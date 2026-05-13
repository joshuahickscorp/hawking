#!/usr/bin/env bash
#
# tools/bench/bench_diff.sh — compare bench results across two commits.
#
# Reads bench_results/bench_history.jsonl (append-only log written by
# coexist_bench.sh / quick_bench.sh), finds runs at two commit hashes,
# and reports whether the difference is statistically significant.
#
# Usage:
#   bash tools/bench/bench_diff.sh <commit_a> <commit_b>
#   bash tools/bench/bench_diff.sh HEAD~1 HEAD
#   bash tools/bench/bench_diff.sh abc1234 def5678
#
# Exits 0 always (informational output only).

set -euo pipefail
cd "$(dirname "$0")/../.."

HISTORY="bench_results/bench_history.jsonl"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <commit_a> <commit_b>" >&2
    exit 1
fi

REF_A="$1"
REF_B="$2"

# Resolve git refs to short hashes (7 chars, matching what bench_history uses).
resolve_commit() {
    local ref="$1"
    git rev-parse --short "$ref" 2>/dev/null || echo "$ref"
}

COMMIT_A=$(resolve_commit "$REF_A")
COMMIT_B=$(resolve_commit "$REF_B")

if [[ ! -f "$HISTORY" ]]; then
    echo "ERROR: $HISTORY not found. Run coexist_bench.sh at least twice first." >&2
    exit 1
fi

# Find the most-recent run for each commit prefix (7-char short hash).
find_run() {
    local commit_prefix="$1"
    jq -c --arg c "$commit_prefix" \
        'select(.commit | startswith($c))' "$HISTORY" | tail -1
}

RUN_A=$(find_run "$COMMIT_A")
RUN_B=$(find_run "$COMMIT_B")

if [[ -z "$RUN_A" ]]; then
    echo "ERROR: no bench_history entry found for commit $COMMIT_A ($REF_A)" >&2
    echo "  Run: bash tools/bench/coexist_bench.sh  at that commit first." >&2
    exit 1
fi
if [[ -z "$RUN_B" ]]; then
    echo "ERROR: no bench_history entry found for commit $COMMIT_B ($REF_B)" >&2
    echo "  Run: bash tools/bench/coexist_bench.sh  at that commit first." >&2
    exit 1
fi

# Extract fields from each run.
extract() {
    local run="$1" field="$2"
    printf '%s\n' "$run" | jq -r "$field // \"null\""
}

MED_A=$(extract "$RUN_A" '.results.median')
MED_B=$(extract "$RUN_B" '.results.median')
CI_LO_A=$(extract "$RUN_A" '.results.ci_95_lo')
CI_HI_A=$(extract "$RUN_A" '.results.ci_95_hi')
CI_LO_B=$(extract "$RUN_B" '.results.ci_95_lo')
CI_HI_B=$(extract "$RUN_B" '.results.ci_95_hi')
IQR_A=$(extract "$RUN_A" '.results.iqr')
IQR_B=$(extract "$RUN_B" '.results.iqr')
TS_A=$(extract "$RUN_A" '.timestamp')
TS_B=$(extract "$RUN_B" '.timestamp')
TOOL_A=$(extract "$RUN_A" '.tool')
TOOL_B=$(extract "$RUN_B" '.tool')
TRIALS_A=$(extract "$RUN_A" '.config.trials')
TRIALS_B=$(extract "$RUN_B" '.config.trials')

# Delta and significance check: CIs overlap = not significant.
# CI overlap: lo_b < hi_a  AND  lo_a < hi_b
awk -v med_a="$MED_A" -v med_b="$MED_B" \
    -v ci_lo_a="$CI_LO_A" -v ci_hi_a="$CI_HI_A" \
    -v ci_lo_b="$CI_LO_B" -v ci_hi_b="$CI_HI_B" \
    -v iqr_a="$IQR_A" -v iqr_b="$IQR_B" \
    -v commit_a="$COMMIT_A" -v commit_b="$COMMIT_B" \
    -v ref_a="$REF_A" -v ref_b="$REF_B" \
    -v ts_a="$TS_A" -v ts_b="$TS_B" \
    -v tool_a="$TOOL_A" -v tool_b="$TOOL_B" \
    -v trials_a="$TRIALS_A" -v trials_b="$TRIALS_B" \
'BEGIN {
    printf "\n=== bench_diff: %s vs %s ===\n\n", ref_a, ref_b

    printf "  commit_a (%s)  [%s, %s trial(s)]:\n", commit_a, ts_a, trials_a
    printf "    median: %.3f dec_tps  (95%% CI: [%.3f, %.3f], IQR: %.3f)\n",
        med_a, ci_lo_a, ci_hi_a, iqr_a
    printf "  commit_b (%s)  [%s, %s trial(s)]:\n", commit_b, ts_b, trials_b
    printf "    median: %.3f dec_tps  (95%% CI: [%.3f, %.3f], IQR: %.3f)\n\n",
        med_b, ci_lo_b, ci_hi_b, iqr_b

    if (med_a <= 0) {
        print "  ERROR: commit_a median is 0 or missing"
        exit 0
    }

    delta_pct = (med_b - med_a) / med_a * 100
    sign = delta_pct >= 0 ? "+" : ""
    printf "  delta:  %s%.1f%% (%s%.3f dec_tps)\n", sign, delta_pct, sign, (med_b - med_a)

    # CI overlap check
    overlap = (ci_lo_b < ci_hi_a) && (ci_lo_a < ci_hi_b)
    if (!overlap) {
        verdict = "CIs do NOT overlap → REAL CHANGE (p≈0.05)"
    } else {
        verdict = "CIs overlap → NOT SIGNIFICANT (p>0.05 — may be noise)"
    }
    printf "  verdict: %s\n\n", verdict

    # Directional interpretation
    if (!overlap && delta_pct > 0) {
        printf "  → commit_b is FASTER (regression ruled out)\n"
    } else if (!overlap && delta_pct < 0) {
        printf "  → commit_b is SLOWER (REGRESSION likely)\n"
    } else {
        printf "  → delta is within noise — run more trials to resolve\n"
    }
    printf "\n"
}'
