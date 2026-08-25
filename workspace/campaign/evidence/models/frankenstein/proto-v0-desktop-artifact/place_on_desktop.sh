#!/bin/zsh
# Run OUTSIDE the sandboxed agent to land the artifact on Desktop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/desktop-staging/b-linear-hub-artifact"
DEST="${HOME}/Desktop/hawking-frankenstein/proto-frankenstein/b-linear-hub-artifact"
mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -R "$SRC" "$DEST"
echo "Placed: $DEST"
ls -la "$DEST"
