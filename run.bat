@echo off
REM One-click start (Windows). Creates a virtualenv on first run.
REM   run.bat                      -> full pipeline + dashboard
REM   run.bat --zip "%USERPROFILE%\Downloads\LCORE-D.zip" --open
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\pip install -r requirements.txt
)
.venv\Scripts\python automate.py all %*
