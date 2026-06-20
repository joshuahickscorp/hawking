#!/usr/bin/env bash
# Build the STRAND WASM artifacts (the §14.1 / §14.7 WASM gate).
#
# Two configurations are built, both byte-identical to the native encoder/decoder
# (STRAND is integer-deterministic, so x86 == aarch64 == wasm output):
#
#   - encode      (default features): the FULL codec — Writer + detect + dispatch +
#                 decode. This is audit item #23: the encoder runs in the browser /
#                 at the edge with bounded blocks, and its output is byte-identical
#                 to the server's (cache keys, client-side dedup). The core encode
#                 path pulls no host-only deps (no std::fs / std::thread / std::time),
#                 so it compiles to wasm32 unchanged; `rayon` is behind the (off)
#                 `parallel` feature and never enters the wasm build.
#   - decode-only (--no-default-features --features decode-only): the lean decoder,
#                 encoder feature-gated out, for a minimal-footprint player.
#
# Why this script exists: on a host where Homebrew's cargo/rustc shadow the
# rustup toolchain on PATH, a plain
#     cargo build --target wasm32-unknown-unknown
# fails with E0463 ("can't find crate for core") because the Homebrew sysroot
# ships no wasm32 std. `rustup run stable cargo …` is *also* insufficient because
# PATH still resolves `rustc` to Homebrew first. The robust fix is to invoke the
# rustup toolchain's own cargo/rustc explicitly, which we do below. On a
# rustup-only host (and in CI) the fallback `cargo` already works.
#
# Usage:
#   scripts/wasm-build.sh            # build BOTH configurations (default)
#   scripts/wasm-build.sh encode     # build only the encode (full-codec) variant
#   scripts/wasm-build.sh decode     # build only the decode-only variant
set -euo pipefail

TARGET=wasm32-unknown-unknown
MODE="${1:-both}"

# Resolve the build driver once: prefer the rustup toolchain's own cargo/rustc
# (so Homebrew's PATH shadow can't break the wasm sysroot lookup), else fall back
# to whatever `cargo` is on PATH (the rustup-only / CI case).
CARGO=cargo
RUSTC_ENV=()
if command -v rustup >/dev/null 2>&1; then
  rustup target add "$TARGET" >/dev/null 2>&1 || true
  RUSTUP_HOME_DIR="${RUSTUP_HOME:-$HOME/.rustup}"
  TC_NAME="$(rustup show active-toolchain 2>/dev/null | awk '{print $1}')"
  TC_DIR="$RUSTUP_HOME_DIR/toolchains/$TC_NAME/bin"
  if [[ -n "$TC_NAME" && -x "$TC_DIR/cargo" && -x "$TC_DIR/rustc" ]]; then
    echo "wasm-build: using rustup toolchain '$TC_NAME' ($TC_DIR)"
    CARGO="$TC_DIR/cargo"
    RUSTC_ENV=(env "RUSTC=$TC_DIR/rustc")
  else
    echo "wasm-build: rustup present but no usable wasm toolchain; using cargo on PATH"
  fi
else
  echo "wasm-build: using default cargo on PATH"
fi

build_encode() {
  echo "wasm-build: building ENCODE (full codec) for $TARGET"
  "${RUSTC_ENV[@]}" "$CARGO" build -p strand-container --target "$TARGET" --release
}

build_decode() {
  echo "wasm-build: building DECODE-ONLY for $TARGET"
  "${RUSTC_ENV[@]}" "$CARGO" build -p strand-container \
    --no-default-features --features decode-only --target "$TARGET" --release
}

case "$MODE" in
  encode) build_encode ;;
  decode|decode-only) build_decode ;;
  both) build_encode; build_decode ;;
  *) echo "wasm-build: unknown mode '$MODE' (use: encode | decode | both)" >&2; exit 2 ;;
esac

echo "wasm-build: OK"
