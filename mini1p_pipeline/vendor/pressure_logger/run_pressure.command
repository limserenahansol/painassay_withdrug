#!/bin/bash
# Double-click to start the pressure logger (macOS). It asks for the mouse ID.
cd "$(dirname "$0")" || exit 1
PY=/usr/bin/python3
[ -x "$PY" ] || PY=python3
"$PY" pressure_logger.py --data-dir "/Users/hansollim/Downloads/pressure/data/22/test1016/SBI_1019_Ntsr1_1stcohort" "$@"
echo
read -r -p "Logger closed. Press Enter to close this window..."
