#!/bin/bash
# yewee licence admin — double-click to issue licence keys.
# Vendor tool: never distribute this or the signing key it creates.
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Run the Yewee launcher once first so the environment exists."
  read -n 1 -s -r -p "Press any key to close..."
  exit 1
fi

./.venv/bin/python tools/admin.py
status=$?
if [ $status -ne 0 ]; then
  echo ""
  echo "The licence admin stopped with an error (see above)."
  read -n 1 -s -r -p "Press any key to close..."
fi
