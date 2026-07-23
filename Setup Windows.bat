@echo off
rem facetrack one-time setup for Windows - double-click to run.
cd /d "%~dp0"
echo.
echo === facetrack setup ===
echo.

rem Prefer Python 3.13 (needed for the Spout output), then 3.12, then whatever is default.
set "PYCMD="
py -3.13 -c "" >nul 2>nul && set "PYCMD=py -3.13"
if not defined PYCMD py -3.12 -c "" >nul 2>nul && set "PYCMD=py -3.12"
if not defined PYCMD (
  where python >nul 2>nul && set "PYCMD=python"
)
if not defined PYCMD (
  echo Python was not found. Install Python 3.13 from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" in the installer, then run this again.
  pause
  exit /b 1
)
%PYCMD% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Your Python is too old - need 3.10 or newer. Install 3.13 from python.org.
  pause
  exit /b 1
)
%PYCMD% -c "import sys; sys.exit(0 if sys.version_info < (3,14) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Note: Spout output needs Python 3.13 or older - install 3.13 from python.org
  echo and re-run this setup. Everything else will work fine.
)
for /f "delims=" %%v in ('%PYCMD% --version') do echo Using %%v

echo 1/3 Creating the app environment...
if not exist .venv %PYCMD% -m venv .venv

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
