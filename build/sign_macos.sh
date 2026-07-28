#!/bin/bash
# Sign (and optionally notarise) the macOS bundle.
#
#   build/sign_macos.sh                       # sign + verify + make a .dmg
#   build/sign_macos.sh --notarize            # ...and submit for notarisation
#
# Needs a "Developer ID Application" certificate in your keychain. For
# --notarize you must first store credentials once (Apple will not accept
# a password on the command line, and you should never paste one into a
# script):
#
#   xcrun notarytool store-credentials yewee \
#       --apple-id you@example.com --team-id <TEAMID>
#
# See build/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."

APP="build/dist/Yewee.app"
VERSION="${YEWEE_VERSION:-0.0.0}"
DMG="build/dist/Yewee-${VERSION}.dmg"
ENTS="build/entitlements.plist"
PROFILE="${NOTARY_PROFILE:-yewee}"
NOTARIZE=0
[ "${1:-}" = "--notarize" ] && NOTARIZE=1

IDENTITY=$(security find-identity -v -p codesigning \
  | grep "Developer ID Application" | head -1 | sed 's/.*"\(.*\)"/\1/')
if [ -z "$IDENTITY" ]; then
  echo "No 'Developer ID Application' certificate found in the keychain."
  echo "Add one from your Apple Developer account, then re-run."
  exit 1
fi
[ -d "$APP" ] || { echo "$APP not found — run build/build.py first."; exit 1; }

echo "  identity: $IDENTITY"
echo "  bundle:   $APP"

# PyInstaller copies Syphon.framework with Modules/ as a real directory at
# the framework root instead of a symlink into Versions/Current. codesign
# calls that "unsealed contents present in the root directory of an
# embedded framework" and refuses to verify. Restore the symlink layout
# every framework is supposed to have.
for fw in $(find "$APP" -name "*.framework" -type d); do
  ver="$fw/Versions/Current"
  [ -d "$ver" ] || continue
  for item in "$fw"/*; do
    name=$(basename "$item")
    [ "$name" = "Versions" ] && continue
    if [ -d "$item" ] && [ ! -L "$item" ] && [ -e "$ver/$name" ]; then
      echo "  repairing $(basename "$fw")/$name -> Versions/Current/$name"
      rm -rf "$item"
      ln -s "Versions/Current/$name" "$item"
    fi
  done
done

# Nested code must be signed before the bundle that contains it. Apple
# discourages --deep, so sign the individual Mach-O objects and the
# frameworks, then let the outer signature seal the rest.
echo "  signing nested binaries (this takes a minute)..."
find "$APP" \( -name "*.dylib" -o -name "*.so" -o -name "*.framework" \) -print0 \
  | xargs -0 -P 4 -I {} codesign --force --timestamp --options runtime \
      --sign "$IDENTITY" {} 2>/dev/null || true

echo "  signing the app..."
codesign --force --timestamp --options runtime --entitlements "$ENTS" \
  --sign "$IDENTITY" "$APP"

echo "  entitlements:"
codesign -d --entitlements - --xml "$APP" 2>/dev/null \
  | plutil -p - | grep -E "com\.apple" | sed 's/^/    /'

echo "  verifying..."
codesign --verify --strict --verbose=2 "$APP" 2>&1 | sed 's/^/    /'
echo "  gatekeeper assessment (expect 'rejected' until notarised):"
spctl --assess --type execute --verbose "$APP" 2>&1 | sed 's/^/    /' || true

echo "  building $DMG ..."
rm -f "$DMG"
hdiutil create -volname "Yewee" -srcfolder "$APP" -ov -format UDZO "$DMG" \
  | tail -2 | sed 's/^/    /'

if [ "$NOTARIZE" = "1" ]; then
  echo "  submitting for notarisation (several minutes)..."
  xcrun notarytool submit "$DMG" --keychain-profile "$PROFILE" --wait
  xcrun stapler staple "$DMG"
  echo "  stapled. Verifying a fresh install would pass:"
  spctl --assess --type install --verbose "$DMG" 2>&1 | sed 's/^/    /' || true
else
  echo ""
  echo "  Signed but NOT notarised — macOS will still warn on other machines."
  echo "  Store credentials once:"
  echo "    xcrun notarytool store-credentials $PROFILE \\"
  echo "        --apple-id <your-apple-id> --team-id <TEAMID>"
  echo "  then: build/sign_macos.sh --notarize"
fi
echo ""
