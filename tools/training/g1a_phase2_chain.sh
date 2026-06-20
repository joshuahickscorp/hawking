#!/usr/bin/env bash
# tools/training/g1a_phase2_chain.sh
#
# Phase 2 completion chain — triggered after G1a QAT finishes and PPL is known.
#
# Usage:
#   FINAL_PPL=13.10 GATE_RESULT=PASS_G1B bash tools/training/g1a_phase2_chain.sh
#
#   Or pass as positional args:
#   bash tools/training/g1a_phase2_chain.sh 13.10 PASS_G1B
#
# GATE_RESULT must be one of:
#   PASS_G1B    — PPL <= 13.56 (1.2× baseline) — runs TQ export + full bench
#   PASS_SILVER — PPL <= 15.26 (1.35×)         — skips TQ export, runs bench only
#   PASS_TUNE   — PPL <= 16.95 (1.5×)          — runs bench only
#   FAIL        — PPL > 16.95                  — runs bench only (diagnostic)
#
# Exit: 0 on success, 1 on any hard failure.
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve ROOT regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log()   { echo "[$(stamp)] [phase2] $*"; }

# ---------------------------------------------------------------------------
# Parse inputs: env vars take priority, fall back to positional args
# ---------------------------------------------------------------------------
FINAL_PPL="${FINAL_PPL:-${1:-}}"
GATE_RESULT="${GATE_RESULT:-${2:-}}"

if [[ -z "$FINAL_PPL" ]]; then
    echo "ERROR: FINAL_PPL not set. Export it or pass as first arg." >&2
    exit 1
fi
if [[ -z "$GATE_RESULT" ]]; then
    echo "ERROR: GATE_RESULT not set. Export it or pass as second arg." >&2
    exit 1
fi

# Normalise GATE_RESULT — strip everything after first space so callers can
# pass the full watcher string like "PASS_G1B — no G1b needed, ..."
GATE_RESULT="${GATE_RESULT%% *}"
GATE_RESULT="${GATE_RESULT%%—*}"
GATE_RESULT="${GATE_RESULT%%:*}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RUN="$ROOT/artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8"
EXPORT_OUT="$ROOT/artifacts/lowbit_rwkv7/export/g1a"
VENV="$ROOT/.venv-rwkv"
DOCS_PLANS="$ROOT/docs/plans"
DATE_TAG="$(date -u '+%Y_%m_%d')"
REPORT_OUT="$DOCS_PLANS/phase2_bench_results_${DATE_TAG}.md"

# ---------------------------------------------------------------------------
# Track per-step outcomes for the summary
# ---------------------------------------------------------------------------
TQ_EXPORT_STATUS="skipped"
TQ_BUILD_STATUS="skipped"
TQ_PARITY_STATUS="skipped"
MAMBA2_PARITY_STATUS="skipped"
RWKV7_FLATNESS_STATUS="skipped"
RWKV7_TPS_STATUS="skipped"

TQ_EXPORT_OUTPUT=""
TQ_BUILD_OUTPUT=""
TQ_PARITY_OUTPUT=""
MAMBA2_PARITY_OUTPUT=""
RWKV7_FLATNESS_OUTPUT=""
RWKV7_TPS_OUTPUT=""

FINAL_EXIT=0

# ---------------------------------------------------------------------------
# Helper: run a command, capture output, set status variable
# ---------------------------------------------------------------------------
run_step() {
    local label="$1"
    local status_var="$2"
    local output_var="$3"
    shift 3
    log "--- $label: $*"
    local tmp
    tmp="$(mktemp)"
    local rc=0
    "$@" 2>&1 | tee "$tmp" || rc=$?
    local out
    out="$(cat "$tmp")"
    rm -f "$tmp"
    if [[ $rc -eq 0 ]]; then
        printf -v "$status_var" "PASS"
        log "$label: PASS"
    else
        printf -v "$status_var" "FAIL (exit $rc)"
        printf -v FINAL_EXIT "1"
        log "$label: FAIL (exit $rc)"
    fi
    printf -v "$output_var" "%s" "$out"
}

# ---------------------------------------------------------------------------
# Step 1: TQ export (only if G1b gate passed)
# ---------------------------------------------------------------------------
log "=== Phase 2 chain start ==="
log "FINAL_PPL=$FINAL_PPL  GATE_RESULT=$GATE_RESULT"

if [[ "$GATE_RESULT" == "PASS_G1B" ]]; then
    log "=== Step 1: TQ export ==="

    FINAL_CKPT="$RUN/final"
    if [[ ! -f "$FINAL_CKPT/state_dict.pt" ]]; then
        log "ERROR: checkpoint not found at $FINAL_CKPT/state_dict.pt"
        TQ_EXPORT_STATUS="FAIL (checkpoint missing)"
        FINAL_EXIT=1
    else
        mkdir -p "$EXPORT_OUT"

        # Activate venv if present
        if [[ -f "$VENV/bin/activate" ]]; then
            # shellcheck disable=SC1090
            source "$VENV/bin/activate"
        fi

        # Build the strand quantize-model binary if missing
        if [[ ! -x "$ROOT/target/release/quantize-model" ]]; then
            log "quantize-model binary missing — building..."
            tmp_build_out=""
            run_step "build quantize-model" TQ_BUILD_STATUS tmp_build_out \
                cargo build -p strand-quant --bin quantize-model --release
            TQ_BUILD_OUTPUT="$tmp_build_out"
        fi

        run_step "TQ export" TQ_EXPORT_STATUS TQ_EXPORT_OUTPUT \
            python3 "$ROOT/tools/training/rwkv7_export_strand.py" \
                --checkpoint "$FINAL_CKPT" \
                --out "$EXPORT_OUT" \
                --bits 2 \
                --l 7 \
                --strand-bin "$ROOT/target/release/quantize-model"
    fi
else
    log "Skipping TQ export (gate: $GATE_RESULT, need PASS_G1B)"
fi

# ---------------------------------------------------------------------------
# Step 2: cargo build hawking-core with tq feature (only after successful export)
# ---------------------------------------------------------------------------
if [[ "$TQ_EXPORT_STATUS" == "PASS" ]]; then
    log "=== Step 2: cargo build hawking-core --features tq ==="
    run_step "cargo build tq" TQ_BUILD_STATUS TQ_BUILD_OUTPUT \
        cargo build -p hawking-core --features tq --release
else
    if [[ "$GATE_RESULT" == "PASS_G1B" && "$TQ_EXPORT_STATUS" != "skipped" ]]; then
        log "Skipping cargo build: TQ export did not pass ($TQ_EXPORT_STATUS)"
    fi
fi

# ---------------------------------------------------------------------------
# Step 3: TQ parity test (only if build succeeded)
# ---------------------------------------------------------------------------
if [[ "$TQ_BUILD_STATUS" == "PASS" ]]; then
    log "=== Step 3: RWKV-7 TQ parity test ==="

    # Check test file exists before attempting
    TQ_PARITY_TEST="$ROOT/crates/hawking-core/tests/rwkv7_tq_parity.rs"
    if [[ -f "$TQ_PARITY_TEST" ]]; then
        run_step "rwkv7_tq_parity" TQ_PARITY_STATUS TQ_PARITY_OUTPUT \
            cargo test -p hawking-core --features tq --test rwkv7_tq_parity -- --nocapture
    else
        log "Skipping rwkv7_tq_parity: test file not found at $TQ_PARITY_TEST"
        TQ_PARITY_STATUS="skipped (test file absent)"
    fi
else
    if [[ "$GATE_RESULT" == "PASS_G1B" ]]; then
        log "Skipping TQ parity: build step did not pass ($TQ_BUILD_STATUS)"
    fi
fi

# ---------------------------------------------------------------------------
# Step 4: Mamba2 parity test (gated on test file existence)
# ---------------------------------------------------------------------------
log "=== Step 4: Mamba2 smoke/parity test ==="
MAMBA2_TEST="$ROOT/crates/hawking-core/tests/mamba2_smoke.rs"
if [[ -f "$MAMBA2_TEST" ]]; then
    run_step "mamba2_smoke" MAMBA2_PARITY_STATUS MAMBA2_PARITY_OUTPUT \
        cargo test -p hawking-core --test mamba2_smoke -- --nocapture
else
    log "Skipping mamba2_smoke: $MAMBA2_TEST not found (Phase 2 Mamba2 feature not yet built)"
    MAMBA2_PARITY_STATUS="skipped (test file absent)"
fi

# ---------------------------------------------------------------------------
# Step 5: RWKV-7 flatness bench (context depth sweep to 64k)
# ---------------------------------------------------------------------------
log "=== Step 5: RWKV-7 flatness bench (depth sweep) ==="
METAL_BENCH_TEST="$ROOT/crates/hawking-core/tests/rwkv7_metal_bench.rs"
if [[ -f "$METAL_BENCH_TEST" ]]; then
    run_step "rwkv7_flatness_bench" RWKV7_FLATNESS_STATUS RWKV7_FLATNESS_OUTPUT \
        env HAWKING_RWKV7_MAX_DEPTH=64000 \
        cargo test -p hawking-core --test rwkv7_metal_bench -- \
            --ignored --nocapture --test-threads=1
else
    log "Skipping rwkv7_flatness_bench: $METAL_BENCH_TEST not found"
    RWKV7_FLATNESS_STATUS="skipped (test file absent)"
fi

# ---------------------------------------------------------------------------
# Step 6: RWKV-7 Task #8 tps bench (same binary, different depth cap)
# ---------------------------------------------------------------------------
log "=== Step 6: RWKV-7 Task #8 tps bench ==="
if [[ -f "$METAL_BENCH_TEST" ]]; then
    run_step "rwkv7_tps_bench" RWKV7_TPS_STATUS RWKV7_TPS_OUTPUT \
        cargo test -p hawking-core --test rwkv7_metal_bench -- \
            --ignored --nocapture --test-threads=1
else
    log "Skipping rwkv7_tps_bench: $METAL_BENCH_TEST not found"
    RWKV7_TPS_STATUS="skipped (test file absent)"
fi

# ---------------------------------------------------------------------------
# Step 7: Generate summary markdown
# ---------------------------------------------------------------------------
log "=== Step 7: writing summary to $REPORT_OUT ==="
mkdir -p "$DOCS_PLANS"

# Derive a human-readable gate label
gate_label() {
    case "$1" in
        PASS_G1B)    echo "PASS G1b (PPL ≤ 13.56 — 1.2× baseline) — proceed to TQ export" ;;
        PASS_SILVER) echo "PASS Silver (PPL ≤ 15.26 — 1.35× baseline) — launch G1b for more training" ;;
        PASS_TUNE)   echo "PASS Tune (PPL ≤ 16.95 — 1.5× baseline) — tune-only quality" ;;
        FAIL)        echo "FAIL (PPL > 16.95) — QAT regression, investigate" ;;
        *)           echo "$1" ;;
    esac
}
GATE_LABEL="$(gate_label "$GATE_RESULT")"

# Determine overall chain result
CHAIN_RESULT="PASS"
for s in "$TQ_EXPORT_STATUS" "$TQ_BUILD_STATUS" "$TQ_PARITY_STATUS" \
          "$MAMBA2_PARITY_STATUS" "$RWKV7_FLATNESS_STATUS" "$RWKV7_TPS_STATUS"; do
    if [[ "$s" == FAIL* ]]; then
        CHAIN_RESULT="FAIL"
        break
    fi
done

cat > "$REPORT_OUT" << REPORT_EOF
# Phase 2 Bench Results — $(date -u '+%Y-%m-%d %H:%M UTC')

## G1a Gate Summary

| | |
|---|---|
| Final PPL | **${FINAL_PPL}** |
| Baseline PPL | 11.30 (F32, wikitext2 4k single window) |
| Gate result | **${GATE_LABEL}** |
| G1b gate threshold | ≤ 13.56 (1.2×) |
| Silver gate threshold | ≤ 15.26 (1.35×) |
| Tune gate threshold | ≤ 16.95 (1.5×) |

## Phase 2 Chain Step Results

| Step | Status |
|---|---|
| TQ export | ${TQ_EXPORT_STATUS} |
| cargo build (tq) | ${TQ_BUILD_STATUS} |
| rwkv7_tq_parity | ${TQ_PARITY_STATUS} |
| mamba2_smoke | ${MAMBA2_PARITY_STATUS} |
| rwkv7_flatness_bench (64k) | ${RWKV7_FLATNESS_STATUS} |
| rwkv7_tps_bench (Task #8) | ${RWKV7_TPS_STATUS} |
| **Overall** | **${CHAIN_RESULT}** |

---

## Step Detail

### TQ Export

\`\`\`
${TQ_EXPORT_OUTPUT:-skipped — not applicable for gate $GATE_RESULT}
\`\`\`

### cargo build hawking-core --features tq

\`\`\`
${TQ_BUILD_OUTPUT:-skipped}
\`\`\`

### rwkv7_tq_parity

\`\`\`
${TQ_PARITY_OUTPUT:-skipped}
\`\`\`

### mamba2_smoke

\`\`\`
${MAMBA2_PARITY_OUTPUT:-skipped}
\`\`\`

### rwkv7_flatness_bench (HAWKING_RWKV7_MAX_DEPTH=64000)

\`\`\`
${RWKV7_FLATNESS_OUTPUT:-skipped}
\`\`\`

### rwkv7_tps_bench (Task #8)

\`\`\`
${RWKV7_TPS_OUTPUT:-skipped}
\`\`\`

---

## Next Steps

$(python3 -c "
import sys
ppl = float('${FINAL_PPL}')
gate = '${GATE_RESULT}'
if gate == 'PASS_G1B':
    print('TQ export ran. If rwkv7_tq_parity is green:')
    print('1. Fill any remaining stubs in crates/hawking-core/src/tq.rs')
    print('2. Wire TQ dispatch into the RWKV-7 serving path')
    print('3. Run full integration bench vs Q4_K_M baseline')
    print('4. Commit and push (do not skip TQ parity gate)')
elif gate == 'PASS_SILVER':
    print('G1b launch recommended (~26h FFN ternary, all layers):')
    print('  python3 tools/training/rwkv7_qat.py \\\\')
    print('    --model artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8/final/state_dict.pt \\\\')
    print('    --out artifacts/lowbit_rwkv7/runs/g1b ...')
    print()
    print('Mamba2 and flatness benches above are available for review while G1b trains.')
elif gate == 'PASS_TUNE':
    print('Quality is marginal. Options:')
    print('1. Retrain with smaller requant_every or reduced LR ramp')
    print('2. Accept for tune-only use-case (embedding retrieval, not generation)')
    print('3. Evaluate G1b anyway — more quant steps may recover quality')
else:
    print('FAIL gate — investigate before proceeding:')
    print('1. Check loss curve in artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8/events.jsonl')
    print('2. Look for training instability (loss spike / NaN) near end of run')
    print('3. Consider requant_every=2 or clamp_scale_min increase')
" 2>/dev/null || echo "PPL or gate parse error — review manually")

---

*Generated by tools/training/g1a_phase2_chain.sh at $(stamp)*
REPORT_EOF

log "Report written: $REPORT_OUT"

# ---------------------------------------------------------------------------
# Step 8: launch the expanded v2 chain.
#
# This intentionally runs even when one of the phase2 gates failed: most v2
# checks are independent of TQ quality/artifacts and should still produce useful
# architecture coverage for the final bench wave.
# ---------------------------------------------------------------------------
V2_CHAIN="$ROOT/tools/training/g1a_v2_expansion_chain.sh"
if [[ -f "$V2_CHAIN" ]]; then
    log "=== Step 8: launching G1a v2 expansion chain ==="
    V2_LOG="$ROOT/artifacts/lowbit_rwkv7/g1a_v2_expansion_chain.log"
    FINAL_PPL="$FINAL_PPL" GATE_RESULT="$GATE_RESULT" PHASE2_REPORT="$REPORT_OUT" \
        bash "$V2_CHAIN" >> "$V2_LOG" 2>&1 || {
            rc=$?
            log "G1a v2 expansion chain FAILED (exit $rc); see $V2_LOG"
            FINAL_EXIT=1
        }
    log "G1a v2 expansion chain log: $V2_LOG"
else
    log "G1a v2 expansion chain not found at $V2_CHAIN — skipping"
fi

# ---------------------------------------------------------------------------
# Final exit
# ---------------------------------------------------------------------------
log "=== Phase 2 chain complete — overall: $CHAIN_RESULT ==="
if [[ "$FINAL_EXIT" -ne 0 ]]; then
    log "One or more steps FAILED. Check output above and the report."
    exit 1
fi
exit 0
