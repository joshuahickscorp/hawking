#!/usr/bin/env bash
#
# tools/bench/clean_bench.sh — v0.3.6 clean bench harness
#
# USAGE — run from a plain terminal with Claude Code fully quit
#
#   1. Quit the Claude Code desktop app (Cmd+Q in the menu bar).
#   2. Quit any 'claude' CLI sessions (incl. MASTER_LOOP).
#   3. Open a fresh Terminal.app window.
#   4. cd /Users/scammermike/Downloads/dismantle
#   5. ./tools/bench/clean_bench.sh
#   6. Wait ~30–45 min for all 6 trials. Results appear in
#      bench_results/v0.3.6/. Read summary.md.
#   7. Commit + push the bench_results/v0.3.6/ directory.
#   8. Tell the manager: "v0.3.6 bench results landed".
#
# Pass --gates-only to test pre-flight checks without running any trials.

set -euo pipefail
cd "$(dirname "$0")/../.."

GATES_ONLY="${1:-}"
RESULTS_DIR="bench_results/v0.3.6"
PROFILE_BASE="profiles/deepseek-v2-lite-q4.m3pro18.json"
WEIGHTS="models/deepseek-v2-lite-q4.gguf"
BIN="./target/release/dismantle"

ts()  { date -u +%FT%TZ; }
log() { printf '%s %s\n' "$(ts)" "$*"; }

# Portable timeout — macOS doesn't ship `timeout` (GNU coreutils).
# Prefer `gtimeout` if user has `brew install coreutils`; else fall back
# to a Perl alarm-based exec. Either way the wrapper kills after N seconds.
run_with_timeout() {
    local secs="$1"; shift
    if command -v timeout >/dev/null 2>&1; then
        timeout "$secs" "$@"
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$secs" "$@"
    else
        # `alarm` sends SIGALRM after N seconds; default action is process death.
        perl -e 'use strict; my $secs = shift; alarm $secs; exec { $ARGV[0] } @ARGV or die "exec: $!"' \
            "$secs" "$@"
    fi
}

# ─── GATE: slm ────────────────────────────────────────────────────────────────
gate_slm() {
    log "gate/slm: checking..."
    set +e
    slm_pids=$(pgrep -i slm 2>/dev/null | grep -v aslmanager)
    set -e
    if [[ -n "${slm_pids:-}" ]]; then
        printf 'FAIL gate/slm: slm is running (pids: %s). Exit slm and re-run.\n' \
            "$slm_pids" >&2
        exit 1
    fi
    log "gate/slm: pass"
}

# ─── GATE: Spotlight ──────────────────────────────────────────────────────────
gate_spotlight() {
    local max_wait=600 elapsed=0 poll=60
    log "gate/spotlight: checking top non-bench/non-dismantle process..."
    while true; do
        local ps_out top_line top_cpu top_name
        ps_out=$(ps -axo %cpu,comm | sort -nr)
        top_line=$(printf '%s\n' "$ps_out" \
            | awk '$2 !~ /dismantle|bench/ {print; exit}')
        top_cpu=$(printf '%s\n' "$top_line" | awk '{print $1}')
        top_name=$(printf '%s\n' "$top_line" | awk '{print $2}')
        if awk "BEGIN { exit (\"${top_cpu:-0}\" + 0 < 30.0) ? 0 : 1 }"; then
            log "gate/spotlight: pass (top non-bench: ${top_cpu}% ${top_name})"
            return 0
        fi
        if [[ $elapsed -ge $max_wait ]]; then
            printf 'FAIL gate/spotlight: machine still noisy after %ds. Top: %s%% %s\n' \
                "$max_wait" "$top_cpu" "$top_name" >&2
            printf '%s\n' "$ps_out" | head -10 >&2
            exit 1
        fi
        log "gate/spotlight: waiting (top: ${top_cpu}% ${top_name}, elapsed=${elapsed}s of ${max_wait}s)"
        sleep $poll
        elapsed=$((elapsed + poll))
    done
}

# ─── GATE: Claude processes ───────────────────────────────────────────────────
gate_claude() {
    log "gate/claude: checking for Claude Helper / Renderer / GPU / claude CLI..."
    set +e
    # On macOS, ps -axo comm gives full executable paths, so match on path content.
    # Targets: "Claude Helper" (any variant) and the "claude" CLI binary.
    claude_out=$(ps -axo %cpu,comm | awk '
    {
        cpu = $1 + 0
        comm = $0; sub(/^[[:space:]]*[0-9.]+[[:space:]]+/, "", comm)
        if (cpu > 10.0 && (comm ~ /Claude Helper/ || comm ~ /\/claude$/)) {
            print; ec = 1
        }
    }
    END { exit ec+0 }' 2>&1)
    claude_ec=$?
    set -e
    if [[ $claude_ec -ne 0 ]]; then
        printf 'FAIL gate/claude: Claude Code is running. Quit Claude Code (Cmd+Q on the desktop app) and re-run this script from a plain terminal.\nOffending processes:\n%s\n' \
            "$claude_out" >&2
        exit 1
    fi
    log "gate/claude: pass"
}

# ─── BUILD SANITY ─────────────────────────────────────────────────────────────
gate_build() {
    log "gate/build: checking $BIN --version..."
    if ! "$BIN" --version >/dev/null 2>&1; then
        log "gate/build: binary missing or broken, running cargo build --release --workspace..."
        cargo build --release --workspace
        if ! "$BIN" --version >/dev/null 2>&1; then
            printf 'FAIL gate/build: cargo build completed but %s still not executable.\n' \
                "$BIN" >&2
            exit 1
        fi
    fi
    local ver
    ver=$("$BIN" --version 2>&1 | head -1)
    log "gate/build: pass ($ver)"
}

# ─── PRE-FLIGHT ───────────────────────────────────────────────────────────────
log "Capturing pre-flight top-process snapshot..."
TOP_SNAPSHOT=$(ps -axo %cpu,comm | sort -nr | head -10)
printf '%s\n' "$TOP_SNAPSHOT"

gate_slm
gate_claude
gate_spotlight

if [[ "$GATES_ONLY" == "--gates-only" ]]; then
    log "All pre-flight gates passed. Exiting (--gates-only mode)."
    exit 0
fi

gate_build

# ─── PROFILE GENERATION ───────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
log "Generating profile variants in $RESULTS_DIR/..."
jq '.selected.gemm_q4_k_schedule = "simdgroup"' "$PROFILE_BASE" \
    > "$RESULTS_DIR/profile_simd.json"
jq '.selected.gemm_q4_k_schedule = "scalar"' "$PROFILE_BASE" \
    > "$RESULTS_DIR/profile_scalar.json"
log "Profiles written."

# ─── 6-TRIAL PROTOCOL (alternating: simd-1, scalar-1, …, scalar-3) ───────────
TRIAL_ORDER=(
    "simd:simd_t1"
    "scalar:scalar_t1"
    "simd:simd_t2"
    "scalar:scalar_t2"
    "simd:simd_t3"
    "scalar:scalar_t3"
)

declare -a SIMD_TPS=()
declare -a SIMD_TAGS=()
declare -a SCALAR_TPS=()
declare -a SCALAR_TAGS=()
declare -a TABLE_IDS=()
declare -a TABLE_PROFILES=()
declare -a TABLE_TPS=()
declare -a TABLE_TOKENS=()
declare -a TABLE_MS=()
declare -a TABLE_DISCARDED=()

first_trial=true
for entry in "${TRIAL_ORDER[@]}"; do
    PROFILE="${entry%%:*}"
    TAG="${entry##*:}"
    json_out="$RESULTS_DIR/${TAG}.json"
    trace_out="$RESULTS_DIR/${TAG}.trace.json"

    if [[ "$first_trial" != "true" ]]; then
        log "Sleeping 30s between trials..."
        sleep 30
        gate_spotlight
        gate_claude
    fi
    first_trial=false

    log "=== Trial $TAG (profile=$PROFILE) ==="

    for attempt in 1 2; do
        if [[ $attempt -eq 2 ]]; then
            log "Retrying $TAG (attempt 2)..."
            sleep 30
            gate_spotlight
            gate_claude
        fi

        set +e
        DISMANTLE_TRACE_DISPATCH=1 run_with_timeout 600 \
            nice -n 19 taskpolicy -b "$BIN" bench \
            --backend dismantle --suite decode \
            --weights "$WEIGHTS" \
            --trials 1 --max-new-tokens 64 \
            --kernel-profile "$RESULTS_DIR/profile_${PROFILE}.json" \
            --json "$json_out" \
            --trace-json "$trace_out"
        trial_ec=$?
        set -e

        if [[ $trial_ec -ne 0 ]] || [[ ! -f "$json_out" ]]; then
            log "DISCARD $TAG attempt $attempt: command failed (exit=$trial_ec)"
            if [[ $attempt -eq 2 ]]; then
                printf 'ABORT: two consecutive discards on slot %s (command_failure exit=%d)\n' \
                    "$TAG" "$trial_ec" > "$RESULTS_DIR/ABORTED.txt"
                log "ABORT written to $RESULTS_DIR/ABORTED.txt. Exiting."
                exit 2
            fi
            continue
        fi

        # Extract trial fields from JSON
        tps=$(jq -r '(.trial_stats[0].decode_tps // .decode_tps) // 0' "$json_out")
        tokens=$(jq -r '(.trial_stats[0].completion_tokens // 0)' "$json_out")
        ms=$(jq -r '(.trial_stats[0].decode_ms // .decode_ms // 0)' "$json_out")

        # Discard rules (corrected from v0.3.5):
        #   decode_ms > 590000  → timeout-truncation
        #   completion_tokens < 50 → truncation or model error
        discard_reason=""
        if awk "BEGIN { exit ($ms > 590000) ? 0 : 1 }"; then
            discard_reason="decode_ms=${ms} > 590000 (timeout-truncation)"
        fi
        int_tokens=$(printf '%.0f' "$tokens" 2>/dev/null || echo "0")
        if [[ -z "$discard_reason" ]] && [[ "$int_tokens" -lt 50 ]]; then
            discard_reason="completion_tokens=${tokens} < 50 (truncation/model error)"
        fi

        if [[ -n "$discard_reason" ]]; then
            log "DISCARD $TAG attempt $attempt: $discard_reason"
            TABLE_IDS+=("$TAG")
            TABLE_PROFILES+=("$PROFILE")
            TABLE_TPS+=("$tps")
            TABLE_TOKENS+=("$tokens")
            TABLE_MS+=("$ms")
            TABLE_DISCARDED+=("yes — $discard_reason")
            if [[ $attempt -eq 2 ]]; then
                printf 'ABORT: two consecutive discards on slot %s (%s)\n' \
                    "$TAG" "$discard_reason" > "$RESULTS_DIR/ABORTED.txt"
                log "ABORT written to $RESULTS_DIR/ABORTED.txt. Exiting."
                exit 2
            fi
            continue
        fi

        log "ACCEPT $TAG: dec_tps=$tps completion_tokens=$tokens decode_ms=${ms}ms"
        TABLE_IDS+=("$TAG")
        TABLE_PROFILES+=("$PROFILE")
        TABLE_TPS+=("$tps")
        TABLE_TOKENS+=("$tokens")
        TABLE_MS+=("$ms")
        TABLE_DISCARDED+=("no")

        if [[ "$PROFILE" == "simd" ]]; then
            SIMD_TPS+=("$tps")
            SIMD_TAGS+=("$TAG")
        else
            SCALAR_TPS+=("$tps")
            SCALAR_TAGS+=("$TAG")
        fi
        break
    done
done

# ─── SUMMARY COMPUTATION ──────────────────────────────────────────────────────
log "Computing summary statistics..."

SUMFILE=$(mktemp)
trap 'rm -f "$SUMFILE"' EXIT

python3 - \
    "${SIMD_TPS[@]}" "---" "${SCALAR_TPS[@]}" \
    "---" "${SIMD_TAGS[@]}" "---" "${SCALAR_TAGS[@]}" \
    > "$SUMFILE" <<'PYEOF'
import sys, shlex

args = sys.argv[1:]

def split_on(lst, sep):
    i = lst.index(sep)
    return lst[:i], lst[i+1:]

simd_tps_strs, rest = split_on(args, "---")
scalar_tps_strs, rest = split_on(rest, "---")
simd_tag_strs, rest = split_on(rest, "---")
scalar_tag_strs = rest

simd_tps   = [float(x) for x in simd_tps_strs]
scalar_tps = [float(x) for x in scalar_tps_strs]
simd_tags  = simd_tag_strs
scalar_tags = scalar_tag_strs

def median_with_tag(vals, tags):
    pairs = sorted(zip(vals, tags))
    return pairs[1][0], pairs[1][1]

sm,  sm_tag  = median_with_tag(simd_tps,   simd_tags)
scm, scm_tag = median_with_tag(scalar_tps, scalar_tags)

simd_vs_scalar = (sm  - scm)   / scm   * 100
scalar_vs_base = (scm - 1.861) / 1.861 * 100
simd_vs_v033   = (sm  - 1.748) / 1.748 * 100
simd_spread    = (max(simd_tps)   - min(simd_tps))   / sm  if sm  else 0.0
scalar_spread  = (max(scalar_tps) - min(scalar_tps)) / scm if scm else 0.0

if sm >= 1.861:
    bucket = "dec_tps >= 1.861 (matches/exceeds scalar) → pivot wedge family"
elif sm >= 1.748:
    bucket = "1.748 <= dec_tps < 1.861 → next attention coalesce (kv_b+o)"
elif sm >= 1.700:
    bucket = "1.700 <= dec_tps < 1.748 → fp32 GEMV kernel tuning"
else:
    bucket = "dec_tps < 1.700 → revert + bisect v0.3.4"

print(f'SIMD_MEDIAN={sm:.4f}')
print(f'SIMD_MEDIAN_TAG={sm_tag}')
print(f'SCALAR_MEDIAN={scm:.4f}')
print(f'SCALAR_MEDIAN_TAG={scm_tag}')
print(f'SIMD_VS_SCALAR={simd_vs_scalar:+.1f}')
print(f'SCALAR_VS_BASE={scalar_vs_base:+.1f}')
print(f'SIMD_VS_V033={simd_vs_v033:+.1f}')
print(f'SIMD_SPREAD={simd_spread:.4f}')
print(f'SCALAR_SPREAD={scalar_spread:.4f}')
print(f'BUCKET={shlex.quote(bucket)}')
PYEOF

# shellcheck disable=SC1090
source "$SUMFILE"
log "simd_median=$SIMD_MEDIAN scalar_median=$SCALAR_MEDIAN"
log "bucket: $BUCKET"

# Spread warnings
NOISE_WARNINGS=""
if awk "BEGIN { exit ($SIMD_SPREAD > 0.25) ? 0 : 1 }"; then
    pct=$(awk "BEGIN { printf \"%.1f\", $SIMD_SPREAD * 100 }")
    log "WARNING: simd trials spread=${pct}% > 25% — noisy data"
    NOISE_WARNINGS+="WARNING: simd trials (max-min)/median = ${pct}% > 25% — results are noisy."$'\n'
fi
if awk "BEGIN { exit ($SCALAR_SPREAD > 0.25) ? 0 : 1 }"; then
    pct=$(awk "BEGIN { printf \"%.1f\", $SCALAR_SPREAD * 100 }")
    log "WARNING: scalar trials spread=${pct}% > 25% — noisy data"
    NOISE_WARNINGS+="WARNING: scalar trials (max-min)/median = ${pct}% > 25% — results are noisy."$'\n'
fi

# Extract top-6 kernels from simd median trial trace
SIMD_MEDIAN_TRACE="$RESULTS_DIR/${SIMD_MEDIAN_TAG}.trace.json"
if [[ -f "$SIMD_MEDIAN_TRACE" ]]; then
    TOP_KERNELS=$(jq -r '
        .kernel_summary | sort_by(-.total_us) | .[0:6][] |
        "\(.kernel): total_us=\(.total_us) mean_us=\(.mean_us) count=\(.count)"
    ' "$SIMD_MEDIAN_TRACE")
else
    TOP_KERNELS="(trace file not found: $SIMD_MEDIAN_TRACE)"
fi

# ─── WRITE summary.md ─────────────────────────────────────────────────────────
{
    printf '# v0.3.6 bench summary\n\n'

    printf '## Top-process snapshot at start\n```\n%s\n```\n\n' "$TOP_SNAPSHOT"

    printf '## Trial table\n'
    printf '| trial | profile | dec_tps | completion_tokens | decode_ms | discarded? |\n'
    printf '|---|---:|---:|---:|---:|---|\n'
    for i in "${!TABLE_IDS[@]}"; do
        printf '| %s | %s | %s | %s | %s | %s |\n' \
            "${TABLE_IDS[$i]}" "${TABLE_PROFILES[$i]}" "${TABLE_TPS[$i]}" \
            "${TABLE_TOKENS[$i]}" "${TABLE_MS[$i]}" "${TABLE_DISCARDED[$i]}"
    done
    printf '\n'

    printf '## Medians\n'
    printf '- simd median dec_tps : **%s** (trial: %s)\n' "$SIMD_MEDIAN" "$SIMD_MEDIAN_TAG"
    printf '- scalar median dec_tps: **%s** (trial: %s)\n\n' "$SCALAR_MEDIAN" "$SCALAR_MEDIAN_TAG"

    printf '## Delta percentages\n'
    printf '- simd vs scalar                   : %s%%\n' "$SIMD_VS_SCALAR"
    printf '- scalar vs v0.2.2 baseline (1.861): %s%%\n' "$SCALAR_VS_BASE"
    printf '- simd vs v0.3.3 baseline (1.748)  : %s%%\n\n' "$SIMD_VS_V033"

    if [[ -n "$NOISE_WARNINGS" ]]; then
        printf '## Noise warnings\n%s\n' "$NOISE_WARNINGS"
    fi

    printf '## Top-6 kernels by total_us (simd median trial: %s)\n```\n%s\n```\n\n' \
        "$SIMD_MEDIAN_TAG" "$TOP_KERNELS"

    printf '## Decision rubric (from v0.3.5 sub-task 6)\n\n'
    printf '- **dec_tps >= 1.861 (matches/exceeds v0.2.2 scalar):** v0.3.x campaign succeeded.\n'
    printf '  Recommend pivoting from CB-overhead reduction to a new wedge family.\n'
    printf '  Candidates: (a) Q8_0/Q6_K Metal GEMV to pull MoE-down off CPU,\n'
    printf '  (b) prefill TPS optimization, (c) longer-context decode benchmarking.\n\n'
    printf '- **1.748 <= dec_tps < 1.861 (improved over v0.3.3 but still trailing scalar):**\n'
    printf '  v0.3.4 worked. Recommend the next attention GEMV coalesce — pair kv_b_proj + o_proj\n'
    printf '  (different inputs but same CB-commit savings) per the v0.3.4 report follow-ups.\n\n'
    printf '- **1.700 <= dec_tps < 1.748 (within noise of v0.3.3):** v0.3.4 was structurally\n'
    printf '  clean but did not move the needle. Recommend pivoting to fp32 GEMV kernel tuning\n'
    printf '  (TG_SIZE alternatives, shmem-tile shape) since CB-overhead is now < 1 commit/layer.\n\n'
    printf '- **dec_tps < 1.700 (regression vs v0.3.3):** v0.3.4 introduced a regression.\n'
    printf '  Recommend a v0.3.6 revert + bisect wedge to identify the offending change\n'
    printf '  in the q_a/kv_a coalesce.\n\n'
    printf '**Active bucket:** %s\n' "$BUCKET"
} > "$RESULTS_DIR/summary.md"

log "Summary written to $RESULTS_DIR/summary.md"
log "Bench complete. Exit 0."
exit 0
