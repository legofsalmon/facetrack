# Packaging yewee

```bash
pip install pyinstaller

# what you keep — everything, no licensing
python build/build.py --version 1.3.0

# what you sell — licensing on, GPL models left out
python build/build.py --distribution --pubkey <hex from Licence Admin> --version 1.3.0
```

Output lands in `build/dist/` — `Yewee.app` on macOS, a `yewee/` folder on
Windows. Roughly **340 MB**; most of it is OpenCV (118 MB), the models
(72 MB) and ONNX Runtime (64 MB).

## What a distribution build changes

`build.py` writes `yewee/_buildinfo.py` (git-ignored, removed afterwards)
carrying `DISTRIBUTION`, your `VENDOR_PUBLIC_KEY` and the version. That is
how a packaged app knows to enforce licensing — environment variables
don't survive packaging.

| | internal | `--distribution` |
|---|---|---|
| Licensing | off, unrestricted | on, 72-hour trial |
| RVM model | included | **excluded** (GPL-3.0) |
| Silhouette models | Fast, Quality, Best | Fast, Quality |

The build fails if RVM ever ends up in a distribution bundle.

## Verified so far

A distribution build on macOS runs standalone with no Python present:
licensing active at 72 hours, only the two shippable silhouette models
offered, pipeline live at ~29 fps, Syphon available.

## Signing and notarisation

Unsigned builds run locally but are blocked or warned about on other
people's machines. Both platforms need paid certificates.

### macOS (Apple Developer Program, $99/yr) — working

```bash
YEWEE_VERSION=1.3.0 build/sign_macos.sh
```

Signs every nested binary, signs the app with the hardened runtime and
`build/entitlements.plist`, verifies, and builds the DMG. Verified with
`Developer ID Application: Colm Hewson (PKN49VCQZQ)`.

Two things the script handles that catch people out:

- **Unsealed contents in `Syphon.framework`.** PyInstaller copies it with
  `Modules/` as a real directory at the framework root instead of a
  symlink into `Versions/Current`. `codesign --verify --strict` refuses
  it. The script restores the symlink before signing.
- **Camera access under the hardened runtime.** Turning the hardened
  runtime on means `NSCameraUsageDescription` is no longer enough —
  `com.apple.security.device.camera` is required too, or capture is
  denied with no useful error. `disable-library-validation` is there
  because PyInstaller loads many extension modules at runtime.

Nothing is written inside the bundle at runtime (see `yewee/paths.py`) —
that would invalidate the signature. A test in `tests/smoke.py` guards it.

**Notarisation** needs credentials stored once, interactively, so that no
password passes through a script or a terminal history:

```bash
xcrun notarytool store-credentials yewee \
    --apple-id <your-apple-id> --team-id PKN49VCQZQ
```

Use an [app-specific password](https://appleid.apple.com), not the Apple
ID password. Then:

```bash
YEWEE_VERSION=1.3.0 build/sign_macos.sh --notarize
```

That submits, waits, and staples the ticket to the DMG so it validates
offline. Until it is notarised, `spctl` reports
`source=Unnotarized Developer ID` and other machines will warn.

### Windows (code-signing certificate, ~£200–400/yr)

Build an installer with [Inno Setup](https://jrsoftware.org/isinfo.php)
from `build/dist/yewee/`, then:

```
signtool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 ^
  /a build\dist\yewee-setup.exe
```

Without a signature SmartScreen warns users off. Reputation builds over
time, or an EV certificate skips the wait.

## CI

`.github/workflows/build.yml` builds both platforms on every tag and
uploads the results as artefacts. It builds **unsigned** — signing runs
locally, or add the certificates as repository secrets and extend the
workflow.
