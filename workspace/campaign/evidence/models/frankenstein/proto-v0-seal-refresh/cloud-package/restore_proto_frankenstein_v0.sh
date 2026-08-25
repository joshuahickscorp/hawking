#!/usr/bin/env bash
# One-command local restore for PROTO_FRANKENSTEIN_V0 cloud package.
# Verifies payload hashes against the sealed cloud manifest, then copies to --out.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-$HOME/Desktop/hawking-frankenstein/proto-frankenstein}"
python3.12 -m lab.operators.frankenstein_v0_seal restore \
  --package "$SCRIPT_DIR" \
  --out "$OUT_DIR"
echo "restored to $OUT_DIR"
