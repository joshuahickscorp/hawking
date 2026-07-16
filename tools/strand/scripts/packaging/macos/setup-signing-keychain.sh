#!/bin/zsh
# setup-signing-keychain.sh — stand up a dedicated keychain holding the
# Developer ID Application identity, without touching the login keychain.
#
# Mirrors the recipe tailor's CI uses (tailor/.github/workflows/
# build-desktop.yml): dedicated keychain + known password means
# `security set-key-partition-list` works non-interactively, so codesign
# never pops a password dialog.
#
# Source material is always explicit; this script never guesses another local
# repository or a machine-specific secret path.
#
# Usage:
#   STRAND_SIGNING_KEY=/secure/key.pem \
#   STRAND_SIGNING_CER=/secure/developer-id.cer \
#   ./setup-signing-keychain.sh
# Then:  security find-identity -v -p codesigning   # should list the identity
# Undo:  security delete-keychain /tmp/strand-signing.keychain

set -euo pipefail

KEY="${STRAND_SIGNING_KEY:-}"
CER="${STRAND_SIGNING_CER:-}"
KC="/tmp/strand-signing.keychain"
INTERMEDIATE_URL="https://www.apple.com/certificateauthority/DeveloperIDG2CA.cer"
INTERMEDIATE="/tmp/DeveloperIDG2CA.cer"

[ -n "$KEY" ] || { echo "✗ STRAND_SIGNING_KEY is required"; exit 1; }
[ -n "$CER" ] || { echo "✗ STRAND_SIGNING_CER is required"; exit 1; }
[ -f "$KEY" ] || { echo "✗ private key not found: $KEY"; exit 1; }
[ -f "$CER" ] || { echo "✗ certificate not found: $CER"; exit 1; }

# Fresh keychain with a throwaway password (needed for partition-list).
PW="$(openssl rand -base64 24)"
security delete-keychain "$KC" 2>/dev/null || true
security create-keychain -p "$PW" "$KC"
security set-keychain-settings -lut 21600 "$KC"
security unlock-keychain -p "$PW" "$KC"

# Import: private key (unencrypted PKCS#8 PEM) + leaf certificate.
security import "$KEY" -k "$KC" -T /usr/bin/codesign
security import "$CER" -k "$KC" -T /usr/bin/codesign

# The Developer ID G2 intermediate is not preinstalled on this machine;
# codesign needs it to build the chain to Apple's root.
if [ ! -f "$INTERMEDIATE" ]; then
  curl -fsSL -o "$INTERMEDIATE" "$INTERMEDIATE_URL"
fi
security import "$INTERMEDIATE" -k "$KC"

# Pre-authorize Apple's signing tools so codesign doesn't prompt.
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$PW" "$KC" >/dev/null

# Put the keychain in the user search list (idempotent-ish: re-adding is fine).
EXISTING=$(security list-keychains -d user | tr -d '" ')
security list-keychains -d user -s "$KC" ${(f)EXISTING}

echo "✓ signing keychain ready: $KC"
security find-identity -v -p codesigning
