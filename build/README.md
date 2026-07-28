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

## Signing and notarisation — not done yet

Unsigned builds run locally but will be blocked or warned about on other
people's machines. Both need paid certificates.

### macOS (Apple Developer Program, $99/yr)

```bash
codesign --deep --force --options runtime --timestamp \
  --sign "Developer ID Application: YOUR NAME (TEAMID)" build/dist/Yewee.app

hdiutil create -volname Yewee -srcfolder build/dist/Yewee.app \
  -ov -format UDZO build/dist/Yewee-1.3.0.dmg

xcrun notarytool submit build/dist/Yewee-1.3.0.dmg \
  --apple-id you@example.com --team-id TEAMID --password <app-specific> --wait
xcrun stapler staple build/dist/Yewee-1.3.0.dmg
```

Known snag: PyInstaller leaves "unsealed contents" inside
`Syphon.framework`, which `codesign --deep` complains about. Sign that
framework on its own first if notarisation rejects it.

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
