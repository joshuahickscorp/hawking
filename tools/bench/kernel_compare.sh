#!/usr/bin/env bash
#
# tools/bench/kernel_compare.sh — compare two kernels at a production shape.
#
# Reads bench_results/kernel_perf_history.jsonl (written by
# `dismantle bench-kernel`), finds the most-recent runs of each kernel at the
# given shape, and prints relative performance.
#
# Usage:
#   bash tools/bench/kernel_compare.sh <shape_tag> <kernel_a> <kernel_b>
#
# Example:
#   bash tools/bench/kernel_compare.sh v2_lite_gate_up \
#       gemv_q4_k_m_v2_pinned_tcb gemv_q3_k_pinned_tcb
#
# shape_tag can also be a raw shape like "1408x2048" (matched against .shape).
#
# Exits 0 always (informational output).

set -euo pipefail
cd "$(dirname "$0")/../.."

HISTORY="bench_results/kernel_perf_history.jsonl"

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <shape_tag_or_shape> <kernel_a> <kernel_b>" >&2
    exit 1
fi

SHAPE_KEY="$1"
KERNEL_A="$2"
KERNEL_B="$3"

if [[ ! -f "$HISTORY" ]]; then
    echo "ERROR: $HISTORY not found." >&2
    echo "  Run: dismantle bench-kernel --all --shape <shape>  first." >&2
    exit 1
fi

# Find most-recent run for (kernel, shape_tag_or_shape).
find_run() {
    local kernel="$1" shape_key="$2"
    jq -c --arg k "$kernel" --arg s "$shape_key" \
        'select((.kernel == $k) and (.shape_tag == $s or .shape == $s))' \
        "$HISTORY" | tail -1
}

RUN_A=$(find_run "$KERNEL_A" "$SHAPE_KEY")
RUN_B=$(find_run "$KERNEL_B" "$SHAPE_KEY")

if [[ -z "$RUN_A" ]]; then
    echo "ERROR: no history entry for kernel=$KERNEL_A shape=$SHAPE_KEY" >&2
    echo "  Run: dismantle bench-kernel --kernel $KERNEL_A --shape <shape>" >&2
    exit 1
fi
if [[ -z "$RUN_B" ]]; then
    echo "ERROR: no history entry for kernel=$KERNEL_B shape=$SHAPE_KEY" >&2
    echo "  Run: dismantle bench-kernel --kernel $KERNEL_B --shape <shape>" >&2
    exit 1
fi

extract() {
    local run="$1" field="$2"
    printf '%s\n' "$run" | jq -r "$field // \"null\""
}

MEAN_A=$(extract "$RUN_A" '.mean_us')
P99_A=$(extract "$RUN_A" '.p99_us')
TS_A=$(extract "$RUN_A" '.timestamp')
ITERS_A=$(extract "$RUN_A" '.iterations')

MEAN_B=$(extract "$RUN_B" '.mean_us')
P99_B=$(extract "$RUN_B" '.p99_us')
TS_B=$(extract "$RUN_B" '.timestamp')
ITERS_B=$(extract "$RUN_B" '.iterations')

awk -v ka="$KERNEL_A" -v kb="$KERNEL_B" \
    -v shape="$SHAPE_KEY" \
    -v mean_a="$MEAN_A" -v p99_a="$P99_A" -v ts_a="$TS_A" -v iters_a="$ITERS_A" \
    -v mean_b="$MEAN_B" -v p99_b="$P99_B" -v ts_b="$TS_B" -v iters_b="$ITERS_B" \
'BEGIN {
    printf "\n=== kernel_compare: %s ===\n\n", shape

    printf "  %-40s  mean: %6.1f μs  (p99: %6.1f μs)  [%s, n=%s]\n",
        ka, mean_a, p99_a, ts_a, iters_a
    printf "  %-40s  mean: %6.1f μs  (p99: %6.1f μs)  [%s, n=%s]\n\n",
        kb, mean_b, p99_b, ts_b, iters_b

    if (mean_a <= 0 || mean_b <= 0) {
        print "  ERROR: mean is 0 or missing — run bench-kernel first."
        exit 0
    }

    if (mean_a < mean_b) {
        ratio = mean_b / mean_a
        printf "  %s is %.2fx FASTER (mean: %.1f vs %.1f μs)\n", ka, ratio, mean_a, mean_b
    } else if (mean_b < mean_a) {
        ratio = mean_a / mean_b
        printf "  %s is %.2fx FASTER (mean: %.1f vs %.1f μs)\n", kb, ratio, mean_b, mean_a
    } else {
        print "  kernels are effectively identical in speed."
    }

    pct = (mean_b - mean_a) / mean_a * 100
    sign = pct >= 0 ? "+" : ""
    printf "  delta: %s→%s  %s%.1f%%\n\n", ka, kb, sign, pct
}'
