#!/bin/bash
# facetrack one-time setup for macOS — double-click to run.
cd "$(dirname "$0")"
echo ""
echo "=== facetrack setup ==="
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install it from https://www.python.org/downloads/"
  echo "then double-click this file again."
  read -n 1 -s -r -p "Press any key to close..."; exit 1
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  echo "Your Python is too old (need 3.10+). Install the latest from python.org,"
  echo "then double-click this file again."
  read -n 1 -s -r -p "Press any key to close..."; exit 1
fi

echo "1/3 Creating the app environment..."
[ -d .venv ] || python3 -m venv .venv || { echo "Could not create environment."; read -n 1 -s -r; exit 1; }

echo "2/3 Installing components (first run can take a few minutes)..."
./.venv/bin/python -m pip install --upgrade pip --quiet
./.venv/bin/pip install -r requirements.txt --quiet || { echo "Install failed — check your internet connection and re-run."; read -n 1 -s -r; exit 1; }

echo "3/3 Checking everything works..."
./.venv/bin/python -m facetrack.doctor --fix

echo ""
echo "Setup finished. Double-click 'Start Mac.command' to launch facetrack."
echo "(macOS will ask for camera permission the first time — click Allow.)"
read -n 1 -s -r -p "Press any key to close..."
