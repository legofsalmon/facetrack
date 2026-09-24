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
VERSION="${YEWEE_VERSION:-$(python3 build/build.py --print-version 2>/dev/null || echo 0.0.0)}"
DMG="build/dist/Yewee-${VERSION}.dmg"
ENTS="build/entitlements.plist"
PROFILE="${NOTARY_PROFILE:-yewee}"
NOTARIZE=0
[ "${1:-}" = "--notarize" ] && NOTARIZE=1

# (no match is not an error here: the message below explains it, where
# set -e with pipefail would otherwise stop silently)
IDENTITY=$(security find-identity -v -p codesigning \
  | grep "Developer ID Application" | head -1 | sed 's/.*"\(.*\)"/\1/' || true)
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
while IFS= read -r -d '' fw; do
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
done < <(find "$APP" -name "*.framework" -type d -print0)

# Nested code must be signed before the bundle that contains it. Apple
# discourages --deep for signing, so sign every Mach-O object on its own,
# then each framework, innermost first, then the app, whose signature seals
# everything else (data files need no signature of their own).
#
# Every one of these must succeed. A failure used to be hidden
# (2>/dev/null || true), which left a half-signed bundle that only failed
# at notarisation, or on the buyer's Mac. Now it stops the script, with
# codesign's own message.
MAIN_EXE=$(/usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" \
  "$APP/Contents/Info.plist" 2>/dev/null || true)
MACHO_LIST=$(mktemp "${TMPDIR:-/tmp}/yewee-macho.XXXXXX")
trap 'rm -f "$MACHO_LIST"' EXIT
# Candidates: libraries and extension modules by name, and anything
# executable. `file` decides; a .so or .dylib that is not Mach-O is data.
while IFS= read -r -d '' f; do
  [ -n "$MAIN_EXE" ] && [ "$f" = "$APP/Contents/MacOS/$MAIN_EXE" ] && continue
  case "$(file -b "$f")" in
    Mach-O*) printf '%s\0' "$f" >> "$MACHO_LIST" ;;
    *) case "$f" in
         *.so|*.dylib) echo "  note: not Mach-O, sealed as data: ${f#"$APP"/}" ;;
       esac ;;
  esac
done < <(find "$APP/Contents" -type f \
           \( -name "*.dylib" -o -name "*.so" -o -perm -0100 \) -print0)
count=$(tr -cd '\0' < "$MACHO_LIST" | wc -c | tr -d ' ')
echo "  signing $count nested binaries (this takes a minute)..."
if ! xargs -0 -n 20 -P 4 codesign --force --timestamp --options runtime \
       --sign "$IDENTITY" < "$MACHO_LIST"; then
  echo "  ! codesign failed on a nested binary (message above)." >&2
  echo "  ! Stopping: a half-signed bundle fails notarisation or Gatekeeper." >&2
  exit 1
fi

echo "  signing frameworks..."
# -depth lists a directory after its contents, so a framework inside
# another one is signed before the one that holds it.
while IFS= read -r -d '' fw; do
  if ! codesign --force --timestamp --options runtime --sign "$IDENTITY" "$fw"; then
    echo "  ! codesign failed on ${fw#"$APP"/} (message above)." >&2
    exit 1
  fi
done < <(find "$APP" -depth -name "*.framework" -type d -print0)

echo "  signing the app..."
codesign --force --timestamp --options runtime --entitlements "$ENTS" \
  --sign "$IDENTITY" "$APP"

echo "  entitlements:"
codesign -d --entitlements - --xml "$APP" 2>/dev/null \
  | plutil -p - | grep -E "com\.apple" | sed 's/^/    /'

echo "  verifying every signature in the bundle..."
if ! codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | sed 's/^/    /'; then
  echo "  ! codesign --verify --deep --strict failed (above): the bundle is not" >&2
  echo "  ! fully signed. Not building a DMG from it." >&2
  exit 1
fi
echo "  gatekeeper assessment (expect 'rejected' until notarised):"
spctl --assess --type execute --verbose "$APP" 2>&1 | sed 's/^/    /' || true

# The DMG holds the app and TERMS.txt beside it: LeTissier Creative
# Studios Ltd's terms, which the NDI SDK licence (section 3d) requires the
# app to be distributed under. ditto keeps the signed bundle byte-for-byte
# (symlinks, extended attributes), so the signature stays valid.
TERMS="build/TERMS.txt"
[ -f "$TERMS" ] || { echo "  ! $TERMS is missing; not building a DMG without it." >&2; exit 1; }
STAGE="build/dist/dmg-staging"
rm -rf "$STAGE"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/$(basename "$APP")"
cp "$TERMS" "$STAGE/TERMS.txt"

echo "  building $DMG ..."
rm -f "$DMG"
hdiutil create -volname "Yewee" -srcfolder "$STAGE" -ov -format UDZO "$DMG" \
  | tail -2 | sed 's/^/    /'
rm -rf "$STAGE"

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
