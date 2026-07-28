# PyInstaller spec for yewee — driven by build/build.py, which writes
# yewee/_buildinfo.py first and sets YEWEE_DISTRIBUTION.
#
# Not run directly: `python build/build.py` (see build/README.md).
import os
import sys

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
DIST = os.environ.get("YEWEE_DISTRIBUTION") == "1"
MACOS = sys.platform == "darwin"

# ---- data: the panel, and the models we're allowed to ship -------------
datas = [(os.path.join(ROOT, "yewee", "static"), "yewee/static")]

for name in sorted(os.listdir(os.path.join(ROOT, "models"))):
    if not name.endswith(".onnx"):
        continue
    if DIST and name.startswith("rvm_"):
        continue          # GPL-3.0 — internal builds only, see LICENSE
    datas.append((os.path.join(ROOT, "models", name), "models"))

datas += [(os.path.join(ROOT, "LICENSE"), "."),
          (os.path.join(ROOT, "models", "README.md"), "models")]

# ---- packages whose binaries PyInstaller can't infer -------------------
binaries = []
hiddenimports = [
    # uvicorn resolves these by string at runtime
    "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
]
for pkg in ("cyndilib", "onnxruntime"):     # NDI runtime / ORT providers
    try:
        d, b, h = collect_all(pkg)
        # onnxruntime ships sample models and test data we don't need
        d = [(src, dst) for src, dst in d
             if "datasets" not in src and not src.endswith((".onnx", ".pb"))]
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass                                # absent on this platform

if MACOS:
    for pkg in ("syphon", "objc", "Foundation", "Quartz", "Metal"):
        try:
            d, b, h = collect_all(pkg)
            datas += d
            binaries += b
            hiddenimports += h
        except Exception:
            pass
else:
    for pkg in ("SpoutGL", "pygrabber"):
        try:
            d, b, h = collect_all(pkg)
            datas += d
            binaries += b
            hiddenimports += h
        except Exception:
            pass

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "PIL", "pandas", "scipy",
              "pytest", "onnx", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="yewee",
    console=True,          # the launcher window shows feed names and errors
    icon=os.path.join(ROOT, "assets",
                      "yewee.icns" if MACOS else "yewee.ico"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="yewee")

if MACOS:
    app = BUNDLE(
        coll,
        name="Yewee.app",
        icon=os.path.join(ROOT, "assets", "yewee.icns"),
        bundle_identifier="ie.letissier.yewee",
        info_plist={
            # Without this macOS kills the app the moment it opens a camera
            "NSCameraUsageDescription":
                "Yewee reads your camera to find and follow faces. "
                "Video is processed on this machine and never stored.",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "CFBundleShortVersionString": os.environ.get("YEWEE_VERSION", "0.0.0"),
            "CFBundleVersion": os.environ.get("YEWEE_VERSION", "0.0.0"),
        },
    )
