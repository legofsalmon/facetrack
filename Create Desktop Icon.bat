@echo off
rem Creates a Facetrack desktop shortcut with the proper icon (Windows).
cd /d "%~dp0"
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$lnk = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Facetrack.lnk');" ^
  "$lnk.TargetPath = '%~dp0Facetrack Windows.bat';" ^
  "$lnk.WorkingDirectory = '%~dp0';" ^
  "$lnk.IconLocation = '%~dp0assets\facetrack.ico';" ^
  "$lnk.Description = 'facetrack - live face tracking to NDI';" ^
  "$lnk.Save()"
if errorlevel 1 (
  echo Could not create the shortcut.
) else (
  echo Facetrack icon created on your desktop.
)
pause
