@echo off
rem yewee licence admin - double-click to issue licence keys.
rem Vendor tool: never distribute this or the signing key it creates.
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Run the Yewee launcher once first so the environment exists.
  pause
  exit /b 1
)

.venv\Scripts\python tools\admin.py
if errorlevel 1 (
  echo.
  echo The licence admin stopped with an error - see above.
  pause
)
