@echo off
rem Creates a Yewee desktop shortcut with the proper icon (Windows).
cd /d "%~dp0"
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$lnk = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Yewee.lnk');" ^
  "$lnk.TargetPath = '%~dp0Yewee Windows.bat';" ^
  "$lnk.WorkingDirectory = '%~dp0';" ^
  "$lnk.IconLocation = '%~dp0assets\yewee.ico';" ^
  "$lnk.Description = 'yewee - live face tracking to NDI';" ^
  "$lnk.Save()"
if errorlevel 1 (
  echo Could not create the shortcut.
) else (
  echo Yewee icon created on your desktop.
)
pause
