#!/bin/bash
# Launch facetrack — double-click to run. The control panel opens in your browser.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "facetrack isn't set up yet — double-click 'Setup Mac.command' first."
  read -n 1 -s -r -p "Press any key to close..."; exit 1
fi
./.venv/bin/python main.py "$@"
status=$?
if [ $status -ne 0 ]; then
  echo ""
  echo "facetrack stopped with an error (see messages above)."
  echo "Tip: run 'Setup Mac.command' again to repair, or check the camera is free."
  read -n 1 -s -r -p "Press any key to close..."
fi
