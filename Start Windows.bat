@echo off
rem Launch facetrack - double-click to run. The control panel opens in your browser.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo facetrack isn't set up yet - double-click "Setup Windows.bat" first.
  pause
  exit /b 1
)
.venv\Scripts\python main.py %*
if errorlevel 1 (
  echo.
  echo facetrack stopped with an error (see messages above).
  echo Tip: run "Setup Windows.bat" again to repair, or check the camera is free.
  pause
)
