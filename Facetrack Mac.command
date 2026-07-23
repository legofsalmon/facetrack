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
  [ -s models/scrfd_10g.onnx ]
}

ready() {
  [ -x .venv/bin/python ] &&
  [ -f "$MARKER" ] && [ "$(cat "$MARKER" 2>/dev/null)" = "$REQHASH" ] &&
  models_ok
}

setup() {
  echo ""
  echo "=== facetrack setup (first run / after an update) ==="
  echo ""

  # Environment strategy: `uv` (if installed) provides a self-contained
  # Python 3.12 — most reliable, and enables the Syphon output. Otherwise
  # use the newest suitable system python (3.12 preferred, for Syphon).
  local USE_UV="" PY=""
  if command -v uv >/dev/null 2>&1; then
    USE_UV=1
    echo "Using uv-managed Python 3.12"
  else
    for cand in python3.12 python3.13 python3; do
      if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
        PY="$cand"; break
      fi
    done
    if [ -z "$PY" ]; then
      echo "Python 3.10+ was not found. Easiest fix: install Homebrew's uv"
      echo "(brew install uv) or Python 3.12 from python.org, then re-run this."
      pause_exit
    fi
    echo "Using $($PY --version)"
    if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info < (3,13) else 1)' 2>/dev/null; then
      echo "Note: Syphon output needs Python 3.12 (easiest: brew install uv, then"
      echo "re-run this). Everything else will work fine."
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

./.venv/bin/python main.py "$@"
status=$?
if [ $status -ne 0 ]; then
  echo ""
  echo "facetrack stopped with an error (see messages above)."
  echo "Tip: delete the .venv folder and double-click this again to repair."
  read -n 1 -s -r -p "Press any key to close..."
fi
