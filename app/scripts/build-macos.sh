#!/usr/bin/env bash
#
# HIDE - guided signed + notarized macOS build (no Xcode required).
#
# Just run it; it walks you through everything:
#   bash app/scripts/build-macos.sh
#
# It creates the certificate via the WEB (Apple's site, not Xcode), auto-detects your signing
# identity + Team ID from the keychain, and only asks you to paste two things: your Apple ID email
# and an app-specific password. Nothing is written to disk; secrets live only in this run.
#
set -euo pipefail
cd "$(dirname "$0")/.." # -> app/

# ---- pretty output ----------------------------------------------------------------------------
if [ -t 1 ]; then B="$(tput bold)"; D="$(tput dim)"; R="$(tput sgr0)"; G="$(tput setaf 2)"; Y="$(tput setaf 3)"; else B=""; D=""; R=""; G=""; Y=""; fi
hr() { printf '%s\n' "------------------------------------------------------------------------"; }
pause() { read -r -p "$(printf '\n%sPress Enter when done (or Ctrl+C to quit)...%s ' "$D" "$R")"; }

printf '\n%sHIDE - guided signed + notarized macOS build%s\n' "$B" "$R"
hr

# ---- preflight: command line tools (NOT full Xcode) -------------------------------------------
if ! xcrun --find notarytool >/dev/null 2>&1 || ! command -v codesign >/dev/null 2>&1; then
  printf '%sMissing the Xcode Command Line Tools%s (this is NOT the full Xcode app - it is a small\n' "$Y" "$R"
  echo "developer toolkit that provides 'codesign' and 'notarytool'). Install it with:"
  echo
  echo "    ${B}xcode-select --install${R}"
  echo
  echo "Accept the dialog, let it finish, then re-run this script."
  exit 1
fi

# ---- STEP 1/3: signing identity ---------------------------------------------------------------
printf '\n%sSTEP 1/3  -  Developer ID Application certificate%s\n' "$B" "$R"

detect_identity() {
  security find-identity -v -p codesigning 2>/dev/null \
    | grep "Developer ID Application" | head -1 | sed -E 's/.*"(.*)"/\1/'
}

IDENTITY="$(detect_identity || true)"

if [ -z "$IDENTITY" ]; then
  cat <<EOF

No "Developer ID Application" certificate is in your keychain yet. Create one
${B}via the web${R} (no Xcode needed):

  ${B}1. Make a certificate request (CSR)${R}
     - Open ${B}Keychain Access${R} (press Cmd+Space, type "Keychain Access", Enter).
     - Menu bar: ${B}Keychain Access > Certificate Assistant >
       Request a Certificate From a Certificate Authority...${R}
     - User Email Address: your Apple ID email
     - Common Name: your name
     - CA Email Address: leave blank
     - Select ${B}"Saved to disk"${R}, click Continue, and save the
       ${B}CertificateSigningRequest.certSigningRequest${R} file (e.g. to your Desktop).
       (This also quietly creates the matching private key in your keychain.)

  ${B}2. Issue the certificate on Apple's website${R}
     - Open ${B}https://developer.apple.com/account/resources/certificates/add${R}
     - Under "Software", choose ${B}"Developer ID Application"${R}, then Continue.
     - Click "Choose File", upload the .certSigningRequest from step 1, then Continue.
     - Click ${B}Download${R} to get ${B}developerID_application.cer${R}.

  ${B}3. Install it${R}
     - Double-click the downloaded ${B}.cer${R} file. It installs into your login
       keychain and pairs with the private key from step 1.
EOF
  pause
  IDENTITY="$(detect_identity || true)"
fi

if [ -z "$IDENTITY" ]; then
  printf '\n%sStill no Developer ID Application identity found.%s\n' "$Y" "$R"
  echo "Check what is installed with:"
  echo "    ${B}security find-identity -v -p codesigning${R}"
  echo "Make sure the .cer was downloaded AND double-clicked, then re-run this script."
  exit 1
fi

TEAM_ID="$(printf '%s' "$IDENTITY" | sed -E 's/.*\(([A-Z0-9]{10})\).*/\1/')"
printf '  %sFound:%s   %s\n' "$G" "$R" "$IDENTITY"
printf '  %sTeam ID:%s %s\n' "$G" "$R" "$TEAM_ID"
export APPLE_SIGNING_IDENTITY="$IDENTITY"
export APPLE_TEAM_ID="$TEAM_ID"

# ---- STEP 2/3: notarization credentials -------------------------------------------------------
printf '\n%sSTEP 2/3  -  Notarization credentials%s\n' "$B" "$R"
echo "Your Apple ID is the email on your developer account."
read -r -p "  ${B}Apple ID email:${R} " APPLE_ID
export APPLE_ID

cat <<EOF

Now an ${B}app-specific password${R} (a one-off password, NOT your normal Apple login):
  - Open ${B}https://account.apple.com${R} and sign in.
  - Go to ${B}Sign-In and Security > App-Specific Passwords${R}.
  - Click ${B}+${R}, name it "HIDE notarization", and copy the ${B}xxxx-xxxx-xxxx-xxxx${R} value.
EOF
read -r -s -p "  ${B}Paste app-specific password (hidden):${R} " APPLE_PASSWORD
echo
export APPLE_PASSWORD
if [ -z "${APPLE_ID:-}" ] || [ -z "${APPLE_PASSWORD:-}" ]; then
  printf '\n%sApple ID and app-specific password are both required.%s\n' "$Y" "$R"
  exit 1
fi

# ---- confirm ----------------------------------------------------------------------------------
printf '\n%sReady to build, sign, and notarize:%s\n' "$B" "$R"
echo "  identity : $APPLE_SIGNING_IDENTITY"
echo "  team     : $APPLE_TEAM_ID"
echo "  apple id : $APPLE_ID"
echo "  password : (hidden, used only for this run)"
read -r -p "$(printf '\n%sBuild now? [Enter to continue, Ctrl+C to abort]%s ' "$B" "$R")"

# ---- STEP 3/3: build -> sign -> notarize -> staple --------------------------------------------
printf '\n%sSTEP 3/3  -  Building (a few minutes; notarization waits on Apple servers)...%s\n\n' "$B" "$R"
pnpm install
# Bundle the engine alongside the app (externalBin is enabled in tauri.conf.json).
bash scripts/stage-sidecar.sh || echo "warn: sidecar staging skipped (build hide-serve manually if needed)"
pnpm exec tauri build

printf '\n%sDone.%s Signed + notarized artifacts:\n' "$G" "$R"
echo "  app/src-tauri/target/release/bundle/dmg/    (the .dmg to distribute)"
echo "  app/src-tauri/target/release/bundle/macos/  (the .app)"
echo
echo "Tip: the FIRST time codesign uses your key, macOS may pop a keychain dialog -"
echo "click ${B}Always Allow${R} so it does not ask again."
