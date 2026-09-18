#!/bin/sh
set -eu

# Build one candidate.  This script intentionally does not install or grant
# anything: a candidate must be tested, signed with the same stable identity,
# and promoted by NativeHelperReleaseStore before it can own TCC-sensitive work.
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output=${1:-"$root/hawking-native-helper.candidate"}
compiler=/usr/bin/xcrun
if [ ! -x "$compiler" ]; then
  compiler=swiftc
fi

"$compiler" swiftc -O \
  -framework AppKit \
  -framework ApplicationServices \
  -framework AVFoundation \
  -framework Speech \
  "$root/hawking_macos.swift" \
  -o "$output" \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker "$root/Info.plist"

if [ -n "${HAWKING_CODESIGN_IDENTITY:-}" ]; then
  /usr/bin/codesign --force --sign "$HAWKING_CODESIGN_IDENTITY" --timestamp=none "$output"
else
  echo "candidate built unsigned; set HAWKING_CODESIGN_IDENTITY before promotion" >&2
fi
chmod 755 "$output"
echo "$output"
