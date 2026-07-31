; Inno Setup script for the yewee Windows installer.
;
;   iscc /DVersion=1.3.0 build\yewee.iss
;
; Expects the PyInstaller output in build\dist\yewee (what build\build.py
; leaves behind on Windows). Produces build\dist\yewee-setup-<version>.exe.
;
; Design decisions worth knowing about:
;  - Installs per-user (no admin prompt) into {autopf}. The app writes its
;    settings and logs to %APPDATA%\yewee (see yewee/paths.py), never into
;    its install directory, so Program Files' read-only convention is fine.
;  - No "run at startup", no services, no PATH changes. It is an app you
;    start before a show, not something that should live in the background.
;  - Uninstall leaves %APPDATA%\yewee alone: that folder holds the licence
;    key, and deleting a paid licence because someone reinstalled would be
;    a support nightmare. The uninstaller says so instead.

#ifndef Version
  #define Version "0.0.0"
#endif

[Setup]
AppId={{C6EDA975-21AA-4C87-9384-9D73BB120747}
AppName=yewee
AppVersion={#Version}
AppPublisher=Colm Hewson
AppPublisherURL=https://yeweetracker.letissier.ie
AppSupportURL=mailto:colly@letissier.ie
DefaultDirName={autopf}\yewee
DefaultGroupName=yewee
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=dist
OutputBaseFilename=yewee-setup-{#Version}
SetupIconFile=..\assets\yewee.ico
UninstallDisplayIcon={app}\yewee.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; The bundle is ~400MB unpacked; make sure Setup checks for room.
ExtraDiskSpaceRequired=52428800

[Files]
Source: "dist\yewee\*"; DestDir: "{app}"; \
  Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\yewee"; Filename: "{app}\yewee.exe"
Name: "{autodesktop}\yewee"; Filename: "{app}\yewee.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
  GroupDescription: "{cm:AdditionalIcons}"

[Run]
Filename: "{app}\yewee.exe"; Description: "Start yewee now"; \
  Flags: nowait postinstall skipifsilent

[UninstallRun]
; Nothing to run — but be explicit that this section is empty on purpose:
; the app installs no services and changes no system state.

[Messages]
; Shown on the uninstaller's confirmation page.
ConfirmUninstall=Remove yewee from this computer?%n%nYour settings and licence key (in %%APPDATA%%\yewee) will be kept, so reinstalling later picks up where you left off.
