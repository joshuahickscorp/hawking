#!/usr/bin/env bash
# Run UB checks (§14.6) on the unsafe-bearing crates via miri.
#
# miri is a NIGHTLY-only component and is not installable on the stable toolchain
# this project pins, so it lives behind this script + the nightly CI job rather
# than in rust-toolchain.toml. (strand-coder / strand-engine currently carry no
# `unsafe`, so this is a guard for when SIMD intrinsics land.)
set -euo pipefail

if ! rustup toolchain list 2>/dev/null | grep -q '^nightly'; then
  echo "miri: installing nightly + miri component..."
  rustup toolchain install nightly --component miri
fi
rustup +nightly component add miri >/dev/null 2>&1 || true

echo "miri: running cargo +nightly miri test on strand-coder, strand-engine"
cargo +nightly miri test -p strand-coder -p strand-engine "$@"
