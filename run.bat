@echo off
REM One-click start (Windows). Creates a virtualenv on first run and repairs it if broken.
REM   run.bat                      -> full pipeline + dashboard
REM   run.bat --zip "%USERPROFILE%\Downloads\LCORE-D.zip" --open
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python -c "import flask, sklearn, pandas, numpy, joblib, matplotlib, psutil" 2>nul
if errorlevel 1 (
  echo Libraries missing or corrupted - reinstalling, this takes a few minutes...
  .venv\Scripts\python -m pip install --force-reinstall --no-cache-dir -r requirements.txt
)
.venv\Scripts\python automate.py all %*
