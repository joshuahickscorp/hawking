#!/usr/bin/env bash
# tools/training/g1a_v2_expansion_chain.sh
#
# Extended post-G1a chain. This runs after the existing phase2 gate chain and
# queues only work that is independent of the current training run's numerical
# result, plus gated probes that self-skip when their artifacts are absent.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(stamp)] [g1a-v2] $*"; }

FINAL_PPL="${FINAL_PPL:-${1:-unknown}}"
GATE_RESULT="${GATE_RESULT:-${2:-unknown}}"
PHASE2_REPORT="${PHASE2_REPORT:-}"

DATE_TAG="$(date -u '+%Y_%m_%d')"
OUT_DIR="$ROOT/artifacts/lowbit_rwkv7/v2_expansion"
REPORT_OUT="$ROOT/docs/plans/g1a_v2_expansion_results_${DATE_TAG}.md"
mkdir -p "$OUT_DIR" "$ROOT/docs/plans"

FINAL_EXIT=0
RESULT_ROWS=()

run_capture() {
    local label="$1"
    local logfile="$2"
    shift 2
    log "--- $label"
    log "cmd: $*"
    : > "$logfile"
    local rc=0
    "$@" > "$logfile" 2>&1 || rc=$?
    if [[ "$rc" -eq 0 ]]; then
        RESULT_ROWS+=("| $label | PASS | $logfile |")
        log "$label: PASS"
    else
        RESULT_ROWS+=("| $label | FAIL exit $rc | $logfile |")
        FINAL_EXIT=1
        log "$label: FAIL exit $rc"
    fi
}

run_skip_if_missing() {
    local label="$1"
    local required_path="$2"
    local logfile="$3"
    shift 3
    if [[ ! -e "$required_path" ]]; then
        RESULT_ROWS+=("| $label | skipped: missing $required_path | $logfile |")
        log "$label: skipped; missing $required_path"
        return
    fi
    run_capture "$label" "$logfile" "$@"
}

log "=== G1a v2 expansion chain start ==="
log "FINAL_PPL=$FINAL_PPL GATE_RESULT=$GATE_RESULT PHASE2_REPORT=${PHASE2_REPORT:-none}"

# 1. Compile surfaces that the expansion touches: core, serve, and bench.
run_capture "cargo check dismantle-core" \
    "$OUT_DIR/cargo_check_core.log" \
    env CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" cargo check -p dismantle-core

run_capture "cargo check dismantle-serve" \
    "$OUT_DIR/cargo_check_serve.log" \
    env CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" cargo check -p dismantle-serve

run_capture "cargo check dismantle-bench" \
    "$OUT_DIR/cargo_check_bench.log" \
    env CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" cargo check -p dismantle-bench

run_capture "cargo check dismantle-core tq" \
    "$OUT_DIR/cargo_check_core_tq.log" \
    env CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" cargo check -p dismantle-core --features tq

# 2. llama.cpp gap reducers already/partially in-tree.
run_capture "json constraint unit tests" \
    "$OUT_DIR/json_constraint_tests.log" \
    cargo test -p dismantle-core json_constrain --lib

run_capture "mamba2 smoke" \
    "$OUT_DIR/mamba2_smoke.log" \
    cargo test -p dismantle-core --test mamba2_smoke -- --nocapture

# 3. RWKV-7 breadth and flatness. These skip cleanly if weights are absent.
RWKV7_MODEL="${DISMANTLE_RWKV7_GGUF:-$ROOT/models/rwkv7-04/rwkv7-0.4B-world.Q4_K_M.gguf}"
run_skip_if_missing "rwkv7 metal parity" "$RWKV7_MODEL" \
    "$OUT_DIR/rwkv7_metal_parity.log" \
    env DISMANTLE_RWKV7_GGUF="$RWKV7_MODEL" \
    cargo test -p dismantle-core --test rwkv7_metal_parity -- --nocapture --test-threads=1

run_skip_if_missing "rwkv7 flatness quick 16k" "$RWKV7_MODEL" \
    "$OUT_DIR/rwkv7_flatness_16k.log" \
    env DISMANTLE_RWKV7_GGUF="$RWKV7_MODEL" DISMANTLE_RWKV7_MAX_DEPTH="${DISMANTLE_RWKV7_MAX_DEPTH:-16000}" \
    cargo test -p dismantle-core --test rwkv7_metal_bench -- --ignored --nocapture --test-threads=1

run_capture "tq trellis synthetic parity" \
    "$OUT_DIR/tq_trellis_parity.log" \
    cargo test -p dismantle-core --features tq --test tq_trellis_parity -- --nocapture

if [[ "${G1A_V2_FULL_BENCH:-0}" == "1" ]]; then
    run_skip_if_missing "rwkv7 flatness full 64k" "$RWKV7_MODEL" \
        "$OUT_DIR/rwkv7_flatness_64k.log" \
        env DISMANTLE_RWKV7_GGUF="$RWKV7_MODEL" DISMANTLE_RWKV7_MAX_DEPTH=64000 \
        cargo test -p dismantle-core --test rwkv7_metal_bench -- --ignored --nocapture --test-threads=1
else
    RESULT_ROWS+=("| rwkv7 flatness full 64k | skipped: set G1A_V2_FULL_BENCH=1 | $OUT_DIR/rwkv7_flatness_64k.log |")
fi

# 4. TQ artifact gates. These stay result-dependent but harmless: they skip if
# G1a did not produce/export a .tq artifact.
TQ_ARTIFACT="${RWKV7_TQ_MODEL:-$ROOT/artifacts/lowbit_rwkv7/export/g1a/model.tq}"
if [[ -f "$TQ_ARTIFACT" ]]; then
    run_capture "rwkv7 tq loader" "$OUT_DIR/rwkv7_tq_loader.log" \
        env RWKV7_TQ_TEST_ARTIFACT="$TQ_ARTIFACT" \
        cargo test -p dismantle-core --features tq --test rwkv7_tq_loader -- --ignored --nocapture --test-threads=1
    run_capture "rwkv7 tq bench" "$OUT_DIR/rwkv7_tq_bench.log" \
        env RWKV7_TQ_MODEL="$TQ_ARTIFACT" RWKV7_Q4K_MODEL="$RWKV7_MODEL" \
        cargo test -p dismantle-core --features tq --test rwkv7_tq_bench -- --ignored --nocapture --test-threads=1
else
    RESULT_ROWS+=("| rwkv7 tq loader | skipped: no TQ artifact at $TQ_ARTIFACT | $OUT_DIR/rwkv7_tq_loader.log |")
    RESULT_ROWS+=("| rwkv7 tq bench | skipped: no TQ artifact at $TQ_ARTIFACT | $OUT_DIR/rwkv7_tq_bench.log |")
fi

# 5. Optional llama.cpp RWKV baseline. This is intentionally opt-in because it
# requires a local llama.cpp binary and clean-room conditions for credible tps.
if [[ "${G1A_V2_LLAMA_BASELINE:-0}" == "1" ]]; then
    run_skip_if_missing "llama.cpp rwkv7 head-to-head" "$RWKV7_MODEL" \
        "$OUT_DIR/llama_rwkv7_head_to_head.log" \
        env GGUF="$RWKV7_MODEL" SKIP_BATCH="${SKIP_BATCH:-1}" TOKENS="${TOKENS:-128}" \
        tools/bench/llama_head_to_head.sh
else
    RESULT_ROWS+=("| llama.cpp rwkv7 head-to-head | skipped: set G1A_V2_LLAMA_BASELINE=1 | $OUT_DIR/llama_rwkv7_head_to_head.log |")
fi

cat > "$REPORT_OUT" <<REPORT
# G1a V2 Expansion Chain Results
**Date:** $(date -u '+%Y-%m-%d %H:%M UTC')

## Gate Context

| | |
|---|---|
| Final PPL | $FINAL_PPL |
| Gate result | $GATE_RESULT |
| Phase2 report | ${PHASE2_REPORT:-none} |
| Artifact dir | $OUT_DIR |

## Results

| Step | Status | Log |
|---|---|---|
$(printf '%s\n' "${RESULT_ROWS[@]}")

## Interpretation

This chain is deliberately wider than the G1a promote ladder. It keeps
result-dependent TQ work behind artifact checks, while still advancing the
independent surfaces that improve Dismantle against llama.cpp: JSON-mode
constraint scaffolding, Mamba2 architecture breadth, core/serve/bench compile
health, synthetic TQ parity, RWKV-7 parity, and context-depth flatness.

Set \`G1A_V2_FULL_BENCH=1\` for the full 64k flatness sweep, and
\`G1A_V2_LLAMA_BASELINE=1\` for the clean-room llama.cpp RWKV baseline.
REPORT

log "Report written: $REPORT_OUT"
log "=== G1a v2 expansion chain complete ==="

# ---------------------------------------------------------------------------
# Draft-sweep: train 100M/150M/200M/300M RWKV-7 variants for spec-decode.
# Runs after all architecture checks so a Rust build failure cannot block it.
# EPOCHS=1 gives enough signal to rank variants; extend the winner manually.
# ACCEPT_SEQS=50 keeps the per-checkpoint watcher eval under ~8 min on CPU.
# ---------------------------------------------------------------------------
DRAFT_SWEEP="$ROOT/tools/training/launch_draft_sweep.sh"
if [[ -f "$DRAFT_SWEEP" ]]; then
    log "=== launching draft sweep (100M/150M/200M/300M) ==="
    DRAFT_LOG="$ROOT/artifacts/lowbit_rwkv7/draft_sweep.log"
    mkdir -p "$(dirname "$DRAFT_LOG")"
    EPOCHS="${DRAFT_EPOCHS:-1}" \
    ACCEPT_SEQS="${DRAFT_ACCEPT_SEQS:-50}" \
    PYTHON="${PYTHON:-.venv-rwkv/bin/python}" \
        bash "$DRAFT_SWEEP" >> "$DRAFT_LOG" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
        log "Draft sweep complete — results in $ROOT/artifacts/lowbit_rwkv7/runs/custom_*/eval_log.jsonl"
    else
        log "Draft sweep exited with code $rc — check $DRAFT_LOG"
        FINAL_EXIT=1
    fi
else
    log "Draft sweep script not found at $DRAFT_SWEEP — skipping"
fi

exit "$FINAL_EXIT"
