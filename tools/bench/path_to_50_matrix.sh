#!/usr/bin/env bash
# tools/bench/path_to_50_matrix.sh
#
# Bench matrix for the path-to-50 consolidation. Runs every lever
# combination (baseline + each lever solo + stacks) and writes a single
# JSON + Markdown delta report.
#
# Levers (each opt-in via flag, default off):
#   - L0  baseline           (no flags)
#   - L1  vocab-prune        --vocab-prune-path artifacts/vocab_prune/<file>.json
#   - L2  mixed-precision    --quant-tier-map-path artifacts/calibration/tier_maps/<file>.json   (Session A)
#   - L3  q8-kv              --q8-kv                                                              (Session C)
#   - L4  spec-decode K=4    --speculate exact-shared --verify-window 4                          (validates Session B fix)
#   - L1+L2+L3 stack         all three production levers together
#
# Session B's TCB-batched verify is always-on in the codebase (not flag-gated)
# so its contribution shows up in the L4 row vs the pre-T2.16 baseline in
# artifacts/runs/overnight/spec_decode_sweep.md.
#
# This script MUST be run with Claude Code's desktop app fully quit
# (Cmd+Q both the app and any CLI sessions) — see
# memory/bench_contamination.md. It uses --strict to enforce this.
#
# Usage:
#   bash tools/bench/path_to_50_matrix.sh                  # full matrix
#   LEVERS=L0,L1,L4 bash tools/bench/path_to_50_matrix.sh  # subset
#
# Output:
#   artifacts/runs/path_to_50_matrix/<utc>/results.json
#   artifacts/runs/path_to_50_matrix/<utc>/report.md
#   artifacts/runs/path_to_50_matrix/latest -> <utc>       (symlink)

set -uo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
PROFILE="${PROFILE:-profiles/deepseek-v2-lite-q4.m3pro18.json}"
TOKENS="${TOKENS:-64}"
TRIALS="${TRIALS:-3}"
BIN="./target/release/dismantle"

# Pick the most recent calibration artifact if user hasn't pinned one.
VOCAB_PRUNE_PATH="${VOCAB_PRUNE_PATH:-$(ls -1t artifacts/vocab_prune/*.json 2>/dev/null | head -1)}"
TIER_MAP_PATH="${TIER_MAP_PATH:-$(ls -1t artifacts/calibration/tier_maps/*.json 2>/dev/null | head -1)}"

LEVERS="${LEVERS:-L0,L1,L2,L3,L4,STACK}"

UTC=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="artifacts/runs/path_to_50_matrix/${UTC}"
mkdir -p "$RUN_DIR"

# --- Pre-flight ---
echo "=== path-to-50 bench matrix ==="
echo "weights:           $WEIGHTS"
echo "profile:           $PROFILE"
echo "tokens:            $TOKENS"
echo "trials per lever:  $TRIALS"
echo "vocab-prune file:  ${VOCAB_PRUNE_PATH:-<not found, L1 will skip>}"
echo "tier-map file:     ${TIER_MAP_PATH:-<not found, L2 will skip>}"
echo "levers:            $LEVERS"
echo "run dir:           $RUN_DIR"
echo

if pgrep -f "Claude.app" > /dev/null 2>&1; then
    echo "❌ Claude desktop app is running. Bench numbers will be 4-5x"
    echo "   contaminated (memory/bench_contamination.md). Quit Claude and re-run."
    echo "   This script is STRICT — it will not run with Claude live."
    exit 64
fi

if [[ ! -f "$BIN" ]]; then
    echo "Building dismantle release binary…"
    if ! cargo build --release -p dismantle >> "$RUN_DIR/cargo_build.log" 2>&1; then
        echo "❌ cargo build failed — see $RUN_DIR/cargo_build.log"
        exit 1
    fi
fi

# --- Bench one lever combination ---
# Args: <lever_id> <label> <extra_dismantle_args...>
run_bench() {
    local lever_id="$1"; shift
    local label="$1"; shift
    local out_json="$RUN_DIR/${lever_id}.json"
    local trial_tps=()

    echo "--- $lever_id: $label ---"
    echo "extra args: $*"

    for i in $(seq 1 "$TRIALS"); do
        # Pause hook — pipeline stops between trials, not mid-trial.
        if [[ -f artifacts/runs/PAUSE ]]; then
            echo "  ⏸  PAUSED (touch artifacts/runs/RESUME to continue)"
            while [[ -f artifacts/runs/PAUSE ]]; do
                sleep 10
                if [[ -f artifacts/runs/RESUME ]]; then
                    rm -f artifacts/runs/PAUSE artifacts/runs/RESUME
                    echo "  ▶  resumed"
                    break
                fi
            done
        fi
        local trial_json="/tmp/p50_matrix_${lever_id}_t${i}.json"
        printf "  trial %d/%d... " "$i" "$TRIALS"
        if perl -e 'alarm 600; exec @ARGV' \
            "$BIN" bench --trace-dispatch \
            --backend dismantle --suite decode \
            --weights "$WEIGHTS" --trials 1 --max-new-tokens "$TOKENS" \
            --kernel-profile "$PROFILE" \
            --json "$trial_json" \
            "$@" >/dev/null 2>>"$RUN_DIR/${lever_id}.stderr"
        then
            local tps
            tps=$(jq -r '.results.trial_stats[0].decode_tps // 0' "$trial_json" 2>/dev/null)
            trial_tps+=("$tps")
            printf "%.2f tps\n" "$tps"
        else
            trial_tps+=("0")
            printf "FAILED (see %s)\n" "$RUN_DIR/${lever_id}.stderr"
        fi
    done

    # Median + spread
    local median
    median=$(printf '%s\n' "${trial_tps[@]}" | sort -n | awk 'NR==int((NF+1)/2)' | head -1)
    if [[ -z "$median" ]]; then
        median=$(printf '%s\n' "${trial_tps[@]}" | sort -n | sed -n '2p')
    fi

    # Pack lever record
    jq -n \
        --arg id "$lever_id" \
        --arg label "$label" \
        --argjson trials "$(printf '%s\n' "${trial_tps[@]}" | jq -R . | jq -s 'map(tonumber)')" \
        --argjson median "${median:-0}" \
        '{lever_id: $id, label: $label, trials: $trials, median_tps: $median}' \
        > "$out_json"

    echo "  median: ${median:-0} tps"
    echo
}

# --- Run requested levers ---
IFS=',' read -ra LEVER_LIST <<< "$LEVERS"
for lever in "${LEVER_LIST[@]}"; do
    case "$lever" in
        L0)
            run_bench "L0" "baseline (B always-on TCB-batched verify, no other flags)"
            ;;
        L1)
            if [[ -z "$VOCAB_PRUNE_PATH" ]]; then
                echo "--- L1: SKIPPED (no vocab-prune file found) ---"
                echo
            else
                run_bench "L1" "vocab-prune only" --vocab-prune-path "$VOCAB_PRUNE_PATH"
            fi
            ;;
        L2)
            if [[ -z "$TIER_MAP_PATH" ]]; then
                echo "--- L2: SKIPPED (no tier-map file found — Session A may not have landed) ---"
                echo
            else
                run_bench "L2" "mixed-precision (Session A)" --quant-tier-map-path "$TIER_MAP_PATH"
            fi
            ;;
        L3)
            run_bench "L3" "q8-kv (Session C)" --q8-kv
            ;;
        L4)
            run_bench "L4" "spec-decode exact-shared K=4 (validates Session B)" \
                --speculate exact-shared --verify-window 4
            ;;
        STACK)
            stack_args=()
            [[ -n "$VOCAB_PRUNE_PATH" ]] && stack_args+=(--vocab-prune-path "$VOCAB_PRUNE_PATH")
            [[ -n "$TIER_MAP_PATH"    ]] && stack_args+=(--quant-tier-map-path "$TIER_MAP_PATH")
            stack_args+=(--q8-kv)
            run_bench "STACK" "L1+L2+L3 stacked (all production levers)" "${stack_args[@]}"
            ;;
        *)
            echo "Unknown lever: $lever (skipping)"
            ;;
    esac
done

# --- Consolidate to results.json ---
jq -s '
    {
        run_ts:    "'"$UTC"'",
        weights:   "'"$WEIGHTS"'",
        profile:   "'"$PROFILE"'",
        tokens:    '"$TOKENS"',
        trials:    '"$TRIALS"',
        results:   .
    }
' "$RUN_DIR"/L*.json "$RUN_DIR"/STACK.json 2>/dev/null > "$RUN_DIR/results.json"

# --- Markdown delta report ---
{
    echo "# path-to-50 bench matrix — $UTC"
    echo
    echo "**Conditions:** $TOKENS tokens × $TRIALS trials per lever, Claude quit, M3 Pro 18 GB."
    echo
    echo "| Lever | Median tps | Δ vs L0 | Δ% | Notes |"
    echo "|---|---:|---:|---:|---|"
    baseline=$(jq -r '.results[] | select(.lever_id=="L0") | .median_tps' "$RUN_DIR/results.json")
    jq -r '.results[] | [.lever_id, .label, .median_tps] | @tsv' "$RUN_DIR/results.json" \
        | while IFS=$'\t' read -r lid label tps; do
            if [[ "$lid" == "L0" || -z "$baseline" || "$baseline" == "0" ]]; then
                printf "| %s | %.2f | — | — | %s |\n" "$lid" "$tps" "$label"
            else
                delta=$(awk -v a="$tps" -v b="$baseline" 'BEGIN { printf "%+.2f", a - b }')
                pct=$(awk   -v a="$tps" -v b="$baseline" 'BEGIN { printf "%+.1f%%", (a - b) / b * 100 }')
                printf "| %s | %.2f | %s | %s | %s |\n" "$lid" "$tps" "$delta" "$pct" "$label"
            fi
        done
    echo
    echo "## Interpretation"
    echo
    echo "- **L0** is the post-Session-B baseline (TCB-batched verify always on)."
    echo "- **L4** vs L0 validates whether Session B brought spec-decode to net-positive at K=4."
    echo "  - Pre-T2.16 sweep (artifacts/runs/overnight/spec_decode_sweep.md): off=24.04 / spec=9.26 → −62%."
    echo "  - Target: L4 ≥ L0 (spec-decode at least breaks even)."
    echo "- **L2** target: ≥ +1 tps vs L0."
    echo "- **L3** target: ≥ +2 tps vs L0."
    echo "- **STACK** target: ≥ L0 + (L1 gain) + (L2 gain) + (L3 gain) × 0.7 (some interaction expected)."
    echo
    echo "## Source artifacts"
    echo
    echo "- Per-lever JSON: \`$RUN_DIR/L*.json\`, \`$RUN_DIR/STACK.json\`"
    echo "- Per-lever stderr (failed runs): \`$RUN_DIR/*.stderr\`"
    echo "- Consolidated: \`$RUN_DIR/results.json\`"
} > "$RUN_DIR/report.md"

# --- latest symlink ---
ln -sfn "$UTC" "artifacts/runs/path_to_50_matrix/latest"

echo "=== done ==="
echo "report:  $RUN_DIR/report.md"
echo "results: $RUN_DIR/results.json"
echo "latest:  artifacts/runs/path_to_50_matrix/latest"
