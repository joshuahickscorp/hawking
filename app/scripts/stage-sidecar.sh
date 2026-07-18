#!/usr/bin/env bash
# Build the local engine (hide-serve) and stage it as a Tauri sidecar binary, named with the Rust
# target triple Tauri expects. Run before `tauri build` if you enable `bundle.externalBin`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)" # repo root
TRIPLE="$(rustc -Vv | sed -n 's/host: //p')"
DEST="$ROOT/app/src-tauri/binaries"

echo "building hide-serve (release) ..."
cargo build --release -p hide-serve --manifest-path "$ROOT/Cargo.toml"

mkdir -p "$DEST"
cp "$ROOT/target/release/hide-serve" "$DEST/hide-serve-$TRIPLE"
echo "staged: app/src-tauri/binaries/hide-serve-$TRIPLE"
echo "to bundle it, add to tauri.conf.json:  \"bundle\": { \"externalBin\": [\"binaries/hide-serve\"] }"
