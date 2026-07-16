#!/bin/zsh
# renotarize.sh — rebuild STRAND.app with the current release CLI, notarize,
# staple, reinstall to /Applications, and refresh the shippable zip.
# Idempotent; safe to re-run. Requires the strand-notary keychain profile and
# a reachable Apple timestamp server.
set -euo pipefail
cd "$(dirname "$0")"

CLI=${1:-/tmp/strand-release-bin/strand}
./setup-signing-keychain.sh >/dev/null

# Build can hit a transient timestamp-server flake; retry a few times.
built=0
for i in 1 2 3 4 5; do
  if ./make-app.sh -c "$CLI" 2>&1 | grep -q "✓ built"; then built=1; break; fi
  echo "build attempt $i failed; retrying in 10s"; sleep 10
done
[ "$built" = 1 ] || { echo "FAILED: could not build a timestamped app"; exit 1; }

ditto -c -k --keepParent build/STRAND.app /tmp/STRAND-renotarize.zip
OUT=$(xcrun notarytool submit /tmp/STRAND-renotarize.zip --keychain-profile strand-notary --wait 2>&1)
echo "$OUT" | grep -E "status:" | tail -1
if echo "$OUT" | grep -q "status: Accepted"; then
  xcrun stapler staple build/STRAND.app
  ditto build/STRAND.app /Applications/STRAND.app
  ditto -c -k --keepParent build/STRAND.app build/STRAND-notarized.zip
  spctl -a -t exec -vv /Applications/STRAND.app 2>&1 | head -2
  echo "RENOTARIZE-DONE"
else
  SUBID=$(echo "$OUT" | grep -m1 "  id:" | awk '{print $2}')
  echo "NOT ACCEPTED: $SUBID"
  xcrun notarytool log "$SUBID" --keychain-profile strand-notary 2>&1 | head -30
  exit 1
fi
