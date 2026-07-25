#!/bin/bash
# facetrack — double-click to run. Sets itself up on first launch (a few
# minutes); after that it starts in seconds. Re-runs setup automatically
# if an update changed the requirements or a model file is missing.
cd "$(dirname "$0")"

MARKER=".venv/.facetrack-ready"
REQHASH="$(shasum -a 256 requirements.txt 2>/dev/null | cut -d' ' -f1)"

pause_exit() { echo ""; read -n 1 -s -r -p "Press any key to close..."; exit 1; }

models_ok() {
  [ -s models/face_detection_yunet_2023mar.onnx ] &&
  [ -s models/emotion-ferplus-8.onnx ] &&
  [ -s models/scrfd_10g.onnx ] &&
  [ -s models/human_segmentation_pphumanseg_2023mar.onnx ] &&
  [ -s models/modnet_portrait.onnx ] &&
  [ -s models/rvm_mobilenetv3_fp32.onnx ]
}

ready() {
  [ -x .venv/bin/python ] &&
  [ -f "$MARKER" ] && [ "$(cat "$MARKER" 2>/dev/null)" = "$REQHASH" ] &&
  models_ok
}

have_uv() {
  command -v uv >/dev/null 2>&1 && return 0
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    if [ -x "$d/uv" ]; then export PATH="$d:$PATH"; return 0; fi
  done
  return 1
}

install_uv() {
  echo "Downloading uv (a small tool that fetches Python for this app —"
  echo "installs into your user folder, no password needed)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || return 1
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null 2>&1
}

setup() {
  echo ""
  echo "=== facetrack setup (first run / after an update) ==="
  echo ""

  # Environment strategy: `uv` provides a self-contained Python 3.12 —
  # most reliable, and enables the Syphon output. Fall back to a suitable
  # system python (3.10–3.13); if neither exists, download uv automatically.
  local USE_UV="" PY=""
  if have_uv; then
    USE_UV=1
    echo "Using uv-managed Python 3.12"
  else
    for cand in python3.12 python3.13 python3; do
      if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
        PY="$cand"; break
      fi
    done
    if [ -n "$PY" ]; then
      echo "Using $($PY --version)"
      if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info < (3,13) else 1)' 2>/dev/null; then
        echo "Note: Syphon output needs Python 3.12 (easiest: brew install uv, then"
        echo "delete the .venv folder and re-run this). Everything else works."
      fi
    elif install_uv; then
      USE_UV=1
      echo "Using uv-managed Python 3.12"
    else
      echo "Could not download uv automatically. Check your internet connection,"
      echo "or install Python 3.12 from python.org, then re-run this."
      pause_exit
    fi
  fi

  echo "1/3 Creating the app environment..."
  if [ -n "$USE_UV" ]; then
    if [ ! -d .venv ] || ! ./.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
      rm -rf .venv
      uv venv --seed --python 3.12 .venv || { echo "Could not create environment."; pause_exit; }
    fi
  else
    if [ -d .venv ] && [ "$(./.venv/bin/python -c 'import sys; print(sys.version_info[:2])' 2>/dev/null)" != "$($PY -c 'import sys; print(sys.version_info[:2])')" ]; then
      echo "    (rebuilding environment for $($PY --version))"
      rm -rf .venv
    fi
    [ -d .venv ] || "$PY" -m venv .venv || { echo "Could not create environment."; pause_exit; }
  fi

  # Catch broken interpreter builds early (e.g. Homebrew/macOS library skew).
  if ! ./.venv/bin/python -c 'import pyexpat' 2>/dev/null; then
    echo "This Python build is broken on this macOS version."
    echo "Fix: brew install uv — then double-click this again."
    pause_exit
  fi

  echo "2/3 Installing components (can take a few minutes)..."
  ./.venv/bin/python -m pip install --upgrade pip --quiet
  ./.venv/bin/pip install -r requirements.txt --quiet || { echo "Install failed — check your internet connection and re-run."; pause_exit; }

  echo "3/3 Checking everything works..."
  ./.venv/bin/python -m facetrack.doctor --fix

  echo "$REQHASH" > "$MARKER"
  echo ""
  echo "Setup finished — launching facetrack."
  echo "(macOS will ask for camera permission the first time — click Allow.)"
  echo ""
}

ready || setup
ready || { echo "Setup did not complete cleanly — see messages above."; pause_exit; }

# Run, and auto-restart on crashes (clean quits — Ctrl-C, panel Quit,
# pressing q — end the loop). Ctrl-C during the countdown also stops.
while true; do
  ./.venv/bin/python main.py "$@"
  status=$?
  [ $status -eq 0 ] && break
  echo ""
  echo "facetrack crashed (exit $status) — restarting in 3 seconds. Ctrl-C to stop."
  echo "Details are in logs/facetrack.log. If it keeps crashing, delete the"
  echo ".venv folder and double-click this again to repair."
  sleep 3 || break
done
