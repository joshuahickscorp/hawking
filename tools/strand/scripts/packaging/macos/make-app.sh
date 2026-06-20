#!/bin/zsh
# make-app.sh — reproducible assembly of STRAND.app, the macOS opener that
# makes .sa archives double-clickable.
#
# Usage:
#   ./make-app.sh [-i "<codesign identity>"] [-c <path-to-strand-cli>] [-o <build-dir>]
#
#   -i  codesign identity. Default: best available, auto-detected in order
#       "Developer ID Application" > "Apple Development" > ad-hoc ("-").
#       Pass "-" explicitly to force ad-hoc.
#   -c  path to the release `strand` CLI binary to embed.
#       Default: /tmp/strand-release-bin/strand
#   -o  build output directory. Default: ./build (next to this script)
#
# Produces: <build-dir>/STRAND.app, signed inner-binary-first, verified.
#
# Note: the app currently ships without an icon (.icns). A future nicety —
# add Resources/STRAND.icns and a CFBundleIconFile key to Info.plist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

IDENTITY=""
CLI="/tmp/strand-release-bin/strand"
BUILD="$SCRIPT_DIR/build"

while getopts "i:c:o:h" opt; do
  case $opt in
    i) IDENTITY="$OPTARG" ;;
    c) CLI="$OPTARG" ;;
    o) BUILD="$OPTARG" ;;
    h) sed -n '2,20p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done

# ─── Preflight ────────────────────────────────────────────────────────
[ -x "$CLI" ] || { echo "✗ strand CLI not found/executable at $CLI (override with -c)"; exit 1; }

if ! xcrun -f swiftc >/dev/null 2>&1; then
  echo "✗ swiftc not found. Install Xcode or the Command Line Tools."
  echo "  (An Objective-C fallback would be: clang -framework AppKit ... — not"
  echo "  provided here because this machine has had swiftc available.)"
  exit 1
fi

if [ -z "$IDENTITY" ]; then
  IDS="$(security find-identity -v -p codesigning 2>/dev/null || true)"
  if echo "$IDS" | grep -q "Developer ID Application"; then
    IDENTITY="$(echo "$IDS" | grep "Developer ID Application" | head -1 | sed 's/.*"\(.*\)"/\1/')"
  elif echo "$IDS" | grep -q "Apple Development"; then
    IDENTITY="$(echo "$IDS" | grep "Apple Development" | head -1 | sed 's/.*"\(.*\)"/\1/')"
    echo "⚠ using an Apple Development identity — valid on THIS machine only,"
    echo "  not distributable and not notarizable."
  else
    IDENTITY="-"
    echo "⚠ no signing identity in keychain — falling back to ad-hoc (-s -)."
  fi
fi
echo "→ signing identity: $IDENTITY"

# ─── 1. Compile the opener ────────────────────────────────────────────
echo "→ compiling StrandOpener.swift"
mkdir -p "$BUILD"
xcrun swiftc -O \
  -parse-as-library \
  -target arm64-apple-macos11.0 \
  "$SCRIPT_DIR/StrandOpener.swift" \
  -o "$BUILD/StrandOpener"

# ─── 2. Lay out the bundle ────────────────────────────────────────────
APP="$BUILD/STRAND.app"
echo "→ assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$SCRIPT_DIR/Info.plist"   "$APP/Contents/Info.plist"
printf 'APPL????'           > "$APP/Contents/PkgInfo"
cp "$BUILD/StrandOpener"      "$APP/Contents/MacOS/StrandOpener"
cp "$CLI"                     "$APP/Contents/MacOS/strand"
chmod 755 "$APP/Contents/MacOS/StrandOpener" "$APP/Contents/MacOS/strand"

# ─── 3. Sign: inner binaries first, then the bundle ───────────────────
# --options runtime  = hardened runtime (required for notarization)
# --timestamp        = secure timestamp (required for notarization;
#                      not applicable to ad-hoc signatures)
TSFLAG="--timestamp"
[ "$IDENTITY" = "-" ] && TSFLAG=""

echo "→ codesign inner binary: strand CLI"
codesign --force --options runtime $TSFLAG --sign "$IDENTITY" \
  "$APP/Contents/MacOS/strand"

echo "→ codesign bundle (covers StrandOpener as the main executable)"
codesign --force --options runtime $TSFLAG --sign "$IDENTITY" \
  "$APP"

# ─── 4. Verify ────────────────────────────────────────────────────────
echo "→ codesign --verify --deep --strict"
codesign --verify --deep --strict -vv "$APP"

echo "→ spctl assessment (FAILS until the app is notarized — recorded, not fatal)"
spctl -a -t exec -vv "$APP" || true

echo
echo "✓ built: $APP"
echo "  next: notarize (see README.md), then:"
echo "    ditto \"$APP\" /Applications/STRAND.app   # or drag in Finder"
