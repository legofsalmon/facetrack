# Packaging yewee

```bash
pip install pyinstaller

# what you keep — everything, no licensing
python build/build.py

# what you sell — licensing on, GPL models left out
python build/build.py --distribution
```

The version comes from `__version__` in `yewee/__init__.py` (1.0.0 since
the reset on 2026-09-24). `--version` is optional, and a distribution build
refuses one that disagrees with `__version__`, so bump it in the commit you
tag: tag `v1.0.1` builds only once `__version__` says `1.0.1`.

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

## The licence key

`build.py` bakes the vendor **public** key into every distribution build,
defaulting to the one recorded at the top of the script. It prints which
key it used — check that line says "the vendor key" before you ship
anything, because a build carrying the wrong key looks perfect and cannot
be activated by anybody.

The **private** half lives only in the Licence Admin's data directory
(`~/Library/Application Support/yewee-vendor/signing.key`, mode 0600) and
must never be committed, emailed or pasted anywhere. Back it up somewhere
durable and offline: every installed copy verifies against its public
half, so losing it means no existing install can ever be activated again,
and there is no recovery path. That is the design — activation works with
no server and no internet — and the cost of it is that this one file
matters more than the rest of the repository put together.

## Verified so far

A distribution build on macOS runs standalone with no Python present:
licensing active at 72 hours, only the two shippable silhouette models
offered, pipeline live at ~29 fps, Syphon available.

## Signing and notarisation

Unsigned builds run locally but are blocked or warned about on other
people's machines. Both platforms need paid certificates.

### macOS (Apple Developer Program, $99/yr) — working

```bash
build/sign_macos.sh                 # version read from yewee/__init__.py
```

Signs every nested Mach-O binary (found with `file`, so helpers without
an extension are included), then each framework, then the app with the
hardened runtime and `build/entitlements.plist`. It then runs
`codesign --verify --deep --strict` and builds the DMG only if that
passes. The DMG holds `Yewee.app` with `build/TERMS.txt` beside it
(the Windows installer shows the same file as its licence page). Any
nested signing failure stops the script with codesign's own
message rather than leaving a half-signed bundle to fail at notarisation.
Verified with `Developer ID Application: Colm Hewson (PKN49VCQZQ)` before
that change; the stricter version has not yet been run on a Mac.

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
build/sign_macos.sh --notarize
```

That submits, waits, and staples the ticket to the DMG so it validates
offline. Until it is notarised, `spctl` reports
`source=Unnotarized Developer ID` and other machines will warn.

### Windows (code-signing certificate, ~£200–400/yr)

The installer builds itself: CI compiles `build/yewee.iss` with Inno
Setup on every tag and uploads `yewee-setup-<version>.exe` as an
artefact. Locally it is the same two steps on a Windows machine:
`build.py --distribution`, then `ISCC.exe /DVersion=<v> build\yewee.iss`.

**Not done: signing.** With a certificate in place:

```
signtool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 ^
  /a build\dist\yewee-setup-<version>.exe
```

Without a signature SmartScreen warns users off. Reputation builds over
time, or an EV certificate skips the wait.

## CI

`.github/workflows/build.yml` makes **test builds**: unsigned, for both
platforms, kept as run artefacts for 14 days. Start it from Actions.

`.github/workflows/release.yml` **cuts releases**. Run it from Actions →
release → "Run workflow" on `main` with the version (`v1.0.1`), or push a
`v*` tag. In one run it:

1. checks the version matches `__version__`, that no tag of that name
   points elsewhere, and that `docs/releases/<tag>.md` exists;
2. builds, signs and notarises the Mac dmg with `sign_macos.sh`, then
   checks the dmg the way a buyer's Mac will;
3. builds the Windows installer;
4. creates the tag at the commit it built and publishes the release with
   both files and the notes.

If anything fails, nothing is published. See `docs/releases/README.md` for
the steps around it.

### Releasing from CI: the four secrets

The Mac half needs the signing identity and the notary login as repository
secrets (Settings → Secrets and variables → Actions → New repository
secret). The run stops in its first minute, naming what's missing, if any
of them is absent.

| Secret | What goes in it |
|---|---|
| `MACOS_CERTIFICATE` | The Developer ID Application certificate *with its private key*, as base64. In Keychain Access, right-click "Developer ID Application: Colm Hewson (PKN49VCQZQ)" → Export → .p12 with a password, then `base64 -i yewee-signing.p12 \| pbcopy`. |
| `MACOS_CERTIFICATE_PASSWORD` | The password chosen for that .p12. |
| `APPLE_ID` | The Apple ID that notarises. |
| `APPLE_APP_PASSWORD` | An app-specific password for that Apple ID, from appleid.apple.com → Sign-In and Security → App-Specific Passwords. It is never the Apple ID's own password. |

The team id (`PKN49VCQZQ`) is not a secret; it is set in the workflow.
Delete the exported .p12 file once the secret is saved.

While notarisation is being set up, tick **unnotarised** when running the
workflow: it then needs only the first two secrets, and adds a line to the
release notes telling Mac users to right-click → Open.
