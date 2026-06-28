#!/usr/bin/env bash
# tools/bench/build_kernel_profile.sh
#
# Generate a Hawking kernel profile for any local GGUF. This is the scale-wide
# companion to compare_sota.sh: every model-size row should have its own
# hardware profile before its Hawking tps is quoted as final.
#
# Usage:
#   tools/bench/build_kernel_profile.sh models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
#   WEIGHTS=models/qwen2.5-1.5b-instruct-q4_k_m.gguf MAX_HOURS=2 tools/bench/build_kernel_profile.sh
#   OUT=profiles/qwen14b-instruct-q4k.m3pro18.json tools/bench/build_kernel_profile.sh models/...

set -euo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS="${WEIGHTS:-${1:-}}"
MAX_HOURS="${MAX_HOURS:-1}"
PROFILE_NAME="${PROFILE_NAME:-m3-pro-18gb}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 64
}

sanitize_stem() {
  basename "$1" .gguf | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]\+/-/g; s/^-//; s/-$//'
}

[ -n "$WEIGHTS" ] || die "missing GGUF path (pass as argv[1] or WEIGHTS=...)"
[ -f "$WEIGHTS" ] || die "weights not found: $WEIGHTS"
[ -x ./target/release/hawking ] || die "target/release/hawking missing; run cargo build --release -p hawking"

stem="$(sanitize_stem "$WEIGHTS")"
OUT="${OUT:-profiles/${stem}.m3pro18.json}"
LOG="${LOG:-reports/${stem}_autotune_$(date +%Y%m%d_%H%M%S).jsonl}"

mkdir -p profiles reports

printf '[autotune] weights:      %s\n' "$WEIGHTS"
printf '[autotune] output:       %s\n' "$OUT"
printf '[autotune] profile-name: %s\n' "$PROFILE_NAME"
printf '[autotune] max-hours:    %s\n' "$MAX_HOURS"
printf '[autotune] log:          %s\n' "$LOG"
printf '[autotune] start:        %s\n' "$(date -u +%FT%TZ)"

nice -n 19 taskpolicy -b ./target/release/hawking autotune \
  --weights "$WEIGHTS" \
  --profile "$PROFILE_NAME" \
  --max-hours "$MAX_HOURS" \
  --out "$OUT" \
  --log "$LOG"

printf '[autotune] done:         %s\n' "$(date -u +%FT%TZ)"
ls -lh "$OUT"
