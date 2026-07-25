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
if not exist models\human_segmentation_pphumanseg_2023mar.onnx set "NEEDSETUP=1"
if not exist models\modnet_portrait.onnx set "NEEDSETUP=1"
if not exist models\rvm_mobilenetv3_fp32.onnx set "NEEDSETUP=1"
set "OLDHASH="
if exist "%MARKER%" set /p OLDHASH=<"%MARKER%"
if not "%REQHASH%"=="%OLDHASH%" set "NEEDSETUP=1"

if defined NEEDSETUP (
  call :setup
  if errorlevel 1 ( pause & exit /b 1 )
)

rem Run, and auto-restart on crashes (clean quits end the loop).
:runapp
.venv\Scripts\python main.py %*
if %errorlevel%==0 exit /b 0
echo.
echo facetrack crashed (exit %errorlevel%) - restarting in 3 seconds. Close this
echo window to stop. Details are in logs\facetrack.log. If it keeps crashing,
echo delete the .venv folder and double-click this again to repair.
timeout /t 3 /nobreak >nul
goto :runapp

:setup
echo.
echo === facetrack setup (first run / after an update) ===
echo.

rem Environment strategy: uv (if present) provides a self-contained
rem Python 3.13 - most reliable, and enables the Spout output. Fall back
rem to a suitable system python (3.10-3.13); if neither exists, download
rem uv automatically (user folder, no admin needed).
set "UVCMD="
set "PYCMD="
where uv >nul 2>nul && set "UVCMD=uv"
if not defined UVCMD if exist "%USERPROFILE%\.local\bin\uv.exe" set "UVCMD=%USERPROFILE%\.local\bin\uv.exe"
if defined UVCMD goto :have_env

py -3.13 -c "" >nul 2>nul && set "PYCMD=py -3.13"
if not defined PYCMD py -3.12 -c "" >nul 2>nul && set "PYCMD=py -3.12"
if not defined PYCMD where python >nul 2>nul && set "PYCMD=python"
if not defined PYCMD goto :get_uv
%PYCMD% -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul
if not errorlevel 1 goto :have_env
set "PYCMD="

:get_uv
echo No suitable Python found - downloading uv, a small tool that fetches
echo Python for this app (installs into your user folder, no admin needed)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if exist "%USERPROFILE%\.local\bin\uv.exe" set "UVCMD=%USERPROFILE%\.local\bin\uv.exe"
if not defined UVCMD where uv >nul 2>nul && set "UVCMD=uv"
if not defined UVCMD (
  echo Automatic download failed. Check your internet connection, or install
  echo Python 3.13 from python.org ^(tick "Add python.exe to PATH"^) and re-run.
  exit /b 1
)

:have_env
echo 1/3 Creating the app environment...
if defined UVCMD (
  echo Using uv-managed Python 3.13
  if not exist .venv %UVCMD% venv --seed --python 3.13 .venv
) else (
  for /f "delims=" %%v in ('%PYCMD% --version') do echo Using %%v
  if not exist .venv %PYCMD% -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo Could not create environment.
  exit /b 1
)

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
