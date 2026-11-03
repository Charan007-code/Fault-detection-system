#!/usr/bin/env bash
# One-click start (macOS / Linux). Creates a virtualenv on first run.
#   ./run.sh                      -> full pipeline + dashboard
#   ./run.sh --zip ~/Downloads/LCORE-D.zip --open
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python automate.py all "$@"
