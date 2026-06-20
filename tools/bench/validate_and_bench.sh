#!/usr/bin/env bash
# tools/bench/validate_and_bench.sh
#
# Full validation + bench sequence. Runs in order:
#   1. Release build
#   2. Lib unit tests (all crates, no GPU/weights required)
#   3. GPU-only integration tests (synthetic data, no model file)
#   4. Weight-gated integration tests (skip gracefully if weights absent)
#   5. bake-sidecar  — writes models/*.dismantle, skips if weights absent
#   6. verify        — prints SHA-256 + sidecar presence, skips if weights absent
#   7. report_card   — full lane bench
#
# Stops on first hard failure. Weight-absent skips are NOT failures.
#
# Usage:
#   tools/bench/validate_and_bench.sh
#
# Env overrides:
#   WEIGHTS   path to GGUF (default: models/qwen2.5-3b-instruct-q4_k_m.gguf)
#   DBIN      dismantle binary (default: ./target/release/hawking)
#   SKIP_BENCH=1   skip report_card (run tests + bake only)
#   ONLY_BENCH=1   skip all tests, jump straight to report_card

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
DBIN="${DBIN:-./target/release/hawking}"
SKIP_BENCH="${SKIP_BENCH:-0}"
ONLY_BENCH="${ONLY_BENCH:-0}"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

pass() { echo -e "${GREEN}  ✓ $*${RESET}"; }
fail() { echo -e "${RED}  ✗ $*${RESET}"; }
skip() { echo -e "${YELLOW}  ↷ $*${RESET}"; }
step() { echo -e "\n${CYAN}${BOLD}▶ $*${RESET}"; }

FAILURES=0
PASSES=0
SKIPS=0

run_step() {
    local label="$1"; shift
    echo -e "  ${BOLD}$label${RESET}"
    if "$@"; then
        pass "$label"
        (( PASSES++ )) || true
    else
        fail "$label"
        (( FAILURES++ )) || true
        echo -e "${RED}  Aborting — fix the failure above before continuing.${RESET}"
        exit 1
    fi
}

# ── Skip to bench if requested ────────────────────────────────────────────────
if [[ "$ONLY_BENCH" == "1" ]]; then
    step "Skipping tests (ONLY_BENCH=1) — jumping to report_card"
    exec tools/bench/report_card.sh
fi

# ── 1. Release build ──────────────────────────────────────────────────────────
step "1/7  Release build"
run_step "cargo build --release --workspace" \
    cargo build --release --workspace

# ── 2. Lib unit tests ─────────────────────────────────────────────────────────
step "2/7  Lib unit tests (no GPU/weights required)"
run_step "cargo test --workspace --lib" \
    cargo test --workspace --lib

# ── 3. GPU-only integration tests ─────────────────────────────────────────────
step "3/7  GPU-only integration tests (synthetic data, macOS only)"
for test_name in rope_qk_fused_parity swiglu_fused_ffn_parity; do
    run_step "  $test_name" \
        cargo test -p hawking-core --test "$test_name" --release
done

# ── 4. Weight-gated integration tests ─────────────────────────────────────────
step "4/7  Weight-gated integration tests"
WEIGHTS_ABS="$(pwd)/$WEIGHTS"
if [[ ! -f "$WEIGHTS_ABS" ]]; then
    skip "Weights not found at $WEIGHTS — skipping 4 integration tests"
    echo "     Set WEIGHTS=<path> to run them."
    (( SKIPS+=4 )) || true
else
    echo "  Weights: $WEIGHTS"
    for test_name in \
        greedy_token_only_parity \
        multiseq_decode_parity \
        multiseq_q4k_lmhead_parity \
        prefill_slot_into_multiseq_parity
    do
        run_step "  $test_name" \
            cargo test -p hawking-core --test "$test_name" --release
    done
fi

# ── 5. bake-sidecar ───────────────────────────────────────────────────────────
step "5/7  bake-sidecar"
if [[ ! -f "$WEIGHTS_ABS" ]]; then
    skip "Weights absent — skipping bake-sidecar"
    (( SKIPS++ )) || true
else
    SIDECAR="${WEIGHTS%.gguf}.dismantle"
    run_step "bake-sidecar → $SIDECAR" \
        "$DBIN" bake-sidecar --weights "$WEIGHTS"
    if [[ -f "$SIDECAR" ]]; then
        pass "Sidecar written: $SIDECAR ($(du -sh "$SIDECAR" | cut -f1))"
    else
        fail "Sidecar file not found at $SIDECAR after bake"
        exit 1
    fi
fi

# ── 6. verify ─────────────────────────────────────────────────────────────────
step "6/7  verify model hash"
if [[ ! -f "$WEIGHTS_ABS" ]]; then
    skip "Weights absent — skipping verify"
    (( SKIPS++ )) || true
else
    run_step "dismantle verify --weights $WEIGHTS" \
        "$DBIN" verify --weights "$WEIGHTS"
fi

# ── 7. report_card bench ──────────────────────────────────────────────────────
step "7/7  report_card bench"
if [[ "$SKIP_BENCH" == "1" ]]; then
    skip "SKIP_BENCH=1 — skipping report_card"
    (( SKIPS++ )) || true
else
    echo "  Note: absolute numbers require a clean room (close Claude before this step)."
    echo "  Paired ratios (lane vs lane) are contamination-robust."
    echo ""
    tools/bench/report_card.sh
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════${RESET}"
echo -e "${BOLD}Validation summary${RESET}"
echo -e "  ${GREEN}passed:  $PASSES${RESET}"
echo -e "  ${YELLOW}skipped: $SKIPS${RESET}"
if [[ $FAILURES -gt 0 ]]; then
    echo -e "  ${RED}failed:  $FAILURES${RESET}"
    exit 1
else
    echo -e "  ${RED}failed:  0${RESET}"
    echo -e "${GREEN}${BOLD}All steps passed.${RESET}"
fi
