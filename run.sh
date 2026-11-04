#!/usr/bin/env bash
# One-click start (macOS / Linux / WSL). Creates a virtualenv on first run and repairs it if broken.
#   ./run.sh                      -> full pipeline + dashboard
#   ./run.sh --zip ~/Downloads/LCORE-D.zip --open
set -euo pipefail
cd "$(dirname "$0")"
VENV=.venv
# A .venv created by Windows (run.bat) has Scripts\python.exe, not bin/python - use a separate one.
if [ -d .venv ] && [ ! -x .venv/bin/python ]; then VENV=.venv-linux; fi
if [ ! -x "$VENV/bin/python" ]; then
  rm -rf "$VENV"
  if ! python3 -m venv "$VENV"; then
    echo "Could not create a virtualenv. On Ubuntu/WSL run:  sudo apt update && sudo apt install -y python3-venv python3-pip"
    exit 1
  fi
  "$VENV/bin/python" -m pip install -r requirements.txt
fi
if ! "$VENV/bin/python" -c "import flask, sklearn, pandas, numpy, joblib, matplotlib, psutil" 2>/dev/null; then
  echo "Libraries missing or corrupted - reinstalling..."
  "$VENV/bin/python" -m pip install --force-reinstall --no-cache-dir -r requirements.txt
fi
exec "$VENV/bin/python" automate.py all "$@"
