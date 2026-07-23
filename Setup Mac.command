#!/bin/bash
# facetrack one-time setup for macOS — double-click to run.
cd "$(dirname "$0")"
echo ""
echo "=== facetrack setup ==="
echo ""

# Environment strategy: `uv` (if installed) provides a self-contained
# Python 3.12 — most reliable, and enables the Syphon output. Otherwise use
# the newest suitable system python (3.12 preferred, for Syphon).
USE_UV=""
PY=""
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
    read -n 1 -s -r -p "Press any key to close..."; exit 1
  fi
  echo "Using $($PY --version)"
  if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info < (3,13) else 1)' 2>/dev/null; then
    echo "Note: Syphon output needs Python 3.12 (easiest: brew install uv, then"
    echo "re-run this setup). Everything else will work fine."
  fi
fi

echo "1/3 Creating the app environment..."
if [ -n "$USE_UV" ]; then
  if [ ! -d .venv ] || ! ./.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>/dev/null; then
    rm -rf .venv
    uv venv --seed --python 3.12 .venv || { echo "Could not create environment."; read -n 1 -s -r; exit 1; }
  fi
else
  # Rebuild the venv if it exists but was made with a different Python.
  if [ -d .venv ] && [ "$(./.venv/bin/python -c 'import sys; print(sys.version_info[:2])' 2>/dev/null)" != "$($PY -c 'import sys; print(sys.version_info[:2])')" ]; then
    echo "    (rebuilding environment for $($PY --version))"
    rm -rf .venv
  fi
  [ -d .venv ] || "$PY" -m venv .venv || { echo "Could not create environment."; read -n 1 -s -r; exit 1; }
fi

# Catch broken interpreter builds early (e.g. Homebrew/macOS library skew).
if ! ./.venv/bin/python -c 'import pyexpat' 2>/dev/null; then
  echo "This Python build is broken on this macOS version."
  echo "Fix: brew install uv — then double-click this setup again."
  read -n 1 -s -r -p "Press any key to close..."; exit 1
fi

echo "2/3 Installing components (first run can take a few minutes)..."
./.venv/bin/python -m pip install --upgrade pip --quiet
./.venv/bin/pip install -r requirements.txt --quiet || { echo "Install failed — check your internet connection and re-run."; read -n 1 -s -r; exit 1; }

echo "3/3 Checking everything works..."
./.venv/bin/python -m facetrack.doctor --fix

echo ""
echo "Setup finished. Double-click 'Start Mac.command' to launch facetrack."
echo "(macOS will ask for camera permission the first time — click Allow.)"
read -n 1 -s -r -p "Press any key to close..."
