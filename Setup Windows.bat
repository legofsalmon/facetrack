@echo off
rem facetrack one-time setup for Windows - double-click to run.
cd /d "%~dp0"
echo.
echo === facetrack setup ===
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" in the installer, then run this again.
  pause
  exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Your Python is too old - need 3.10 or newer. Install the latest from python.org.
  pause
  exit /b 1
)

echo 1/3 Creating the app environment...
if not exist .venv python -m venv .venv

echo 2/3 Installing components (first run can take a few minutes)...
.venv\Scripts\python -m pip install --upgrade pip --quiet
.venv\Scripts\pip install -r requirements.txt --quiet
if errorlevel 1 (
  echo Install failed - check your internet connection and run this again.
  pause
  exit /b 1
)

echo 3/3 Checking everything works...
.venv\Scripts\python -m facetrack.doctor --fix

echo.
echo Setup finished. Double-click "Start Windows.bat" to launch facetrack.
pause
