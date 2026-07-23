@echo off
rem facetrack - double-click to run. Sets itself up on first launch (a few
rem minutes); after that it starts in seconds. Re-runs setup automatically
rem if an update changed the requirements or a model file is missing.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "MARKER=.venv\.facetrack-ready"
set "REQHASH="
for /f "skip=1 tokens=1" %%h in ('certutil -hashfile requirements.txt SHA256 2^>nul') do if not defined REQHASH set "REQHASH=%%h"

set "NEEDSETUP="
if not exist .venv\Scripts\python.exe set "NEEDSETUP=1"
if not exist models\face_detection_yunet_2023mar.onnx set "NEEDSETUP=1"
if not exist models\emotion-ferplus-8.onnx set "NEEDSETUP=1"
if not exist models\scrfd_10g.onnx set "NEEDSETUP=1"
set "OLDHASH="
if exist "%MARKER%" set /p OLDHASH=<"%MARKER%"
if not "%REQHASH%"=="%OLDHASH%" set "NEEDSETUP=1"

if defined NEEDSETUP (
  call :setup
  if errorlevel 1 ( pause & exit /b 1 )
)

.venv\Scripts\python main.py %*
if errorlevel 1 (
  echo.
  echo facetrack stopped with an error - see messages above.
  echo Tip: delete the .venv folder and double-click this again to repair.
  pause
)
exit /b 0

:setup
echo.
echo === facetrack setup (first run / after an update) ===
echo.

rem Prefer Python 3.13 (needed for the Spout output), then 3.12, then default.
set "PYCMD="
py -3.13 -c "" >nul 2>nul && set "PYCMD=py -3.13"
if not defined PYCMD py -3.12 -c "" >nul 2>nul && set "PYCMD=py -3.12"
if not defined PYCMD (
  where python >nul 2>nul && set "PYCMD=python"
)
if not defined PYCMD (
  echo Python was not found. Install Python 3.13 from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" in the installer, then run this again.
  exit /b 1
)
%PYCMD% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Your Python is too old - need 3.10 or newer. Install 3.13 from python.org.
  exit /b 1
)
%PYCMD% -c "import sys; sys.exit(0 if sys.version_info < (3,14) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Note: Spout output needs Python 3.13 or older - install 3.13 from python.org
  echo and re-run this. Everything else will work fine.
)
for /f "delims=" %%v in ('%PYCMD% --version') do echo Using %%v

echo 1/3 Creating the app environment...
if not exist .venv %PYCMD% -m venv .venv

echo 2/3 Installing components (can take a few minutes)...
.venv\Scripts\python -m pip install --upgrade pip --quiet
.venv\Scripts\pip install -r requirements.txt --quiet
if errorlevel 1 (
  echo Install failed - check your internet connection and run this again.
  exit /b 1
)

echo 3/3 Checking everything works...
.venv\Scripts\python -m facetrack.doctor --fix

echo %REQHASH%>"%MARKER%"
echo.
echo Setup finished - launching facetrack.
echo.
exit /b 0
