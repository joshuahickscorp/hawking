#!/usr/bin/env bash
#
# tools/bench/autotune_sweep.sh — bench a list of schedule values
# back-to-back, tabulate dec_tps for each.
#
# Use this when the comprehensive-plan §3.2.1 suggests "rerun autotune
# with v2t_gu_v2 + MoE shapes." The previous overnight session's
# closeout listed this as a recommended next move; this script makes
# it one command.
#
# Each value is patched into a temp profile JSON (the base profile is
# left untouched) and benched independently. Results are appended to
# bench_results/autotune_sweep_<timestamp>.jsonl with one row per run.
#
# Usage:
#   tools/bench/autotune_sweep.sh \
#       --field gemm_q4_k_schedule \
#       --values v2t,v2t_gu_v2,v2,per_shape \
#       --trials 4 --tokens 24
#
#   tools/bench/autotune_sweep.sh \
#       --field lm_head_schedule \
#       --values metal-argmax-token-only,simdgroup-matrix-argmax \
#       --trials 4 --tokens 24
#
# Pair with tools/bench/with_slm_paused.sh if slm trainer is running
# concurrently — it'll pause slm for each individual bench.

set -euo pipefail
cd "$(dirname "$0")/../.."

WEIGHTS="${WEIGHTS:-models/deepseek-v2-lite-q4.gguf}"
BASE_PROFILE="${BASE_PROFILE:-profiles/deepseek-v2-lite-q4.m3pro18.json}"
TRIALS=4
TOKENS=24
FIELD=""
VALUES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --field) FIELD="$2"; shift 2 ;;
        --values) VALUES="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --weights) WEIGHTS="$2"; shift 2 ;;
        --profile) BASE_PROFILE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$FIELD" || -z "$VALUES" ]]; then
    echo "Usage: $0 --field <name> --values v1,v2,v3 [--trials N] [--tokens N]" >&2
    exit 1
fi

stamp=$(date +%Y%m%d-%H%M%S)
out="bench_results/autotune_sweep_${stamp}.jsonl"
mkdir -p bench_results

echo "[autotune_sweep] base profile: $BASE_PROFILE"
echo "[autotune_sweep] field:        $FIELD"
echo "[autotune_sweep] values:       $VALUES"
echo "[autotune_sweep] trials/tokens: $TRIALS / $TOKENS"
echo "[autotune_sweep] output:       $out"
echo

IFS=',' read -ra VALUE_LIST <<< "$VALUES"
for value in "${VALUE_LIST[@]}"; do
    value=$(echo "$value" | tr -d ' ')
    echo "=== $FIELD=$value ==="
    # Patch the field into a temp profile (using python for safe JSON edit).
    tmp_profile=$(mktemp /tmp/dismantle_sweep_XXXXX.json)
    python3 -c "
import json, sys
d = json.load(open('$BASE_PROFILE'))
d['selected']['$FIELD'] = '$value'
json.dump(d, open('$tmp_profile', 'w'), indent=2)
"
    # Run the bench. Plain quick_bench-style invocation; trim to dec_tps.
    bench_json=$(mktemp /tmp/dismantle_sweep_bench_XXXXX.json)
    "./target/release/hawking" bench \
        --weights "$WEIGHTS" \
        --kernel-profile "$tmp_profile" \
        --suite decode \
        --trials "$TRIALS" \
        --max-new-tokens "$TOKENS" \
        --json "$bench_json" > /dev/null 2>&1 || {
            echo "  FAILED for $FIELD=$value (skipping)"
            rm -f "$tmp_profile" "$bench_json"
            continue
        }
    # Extract dec_tps median, append row to sweep output.
    python3 -c "
import json
b = json.load(open('$bench_json'))
def find_tps(obj):
    if isinstance(obj, dict):
        if 'tokens_per_sec_median' in obj:
            return obj['tokens_per_sec_median']
        for v in obj.values():
            r = find_tps(v)
            if r is not None: return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_tps(v)
            if r is not None: return r
    return None
tps = find_tps(b)
row = {
    'field': '$FIELD',
    'value': '$value',
    'tokens_per_sec_median': tps,
    'trials': $TRIALS,
    'tokens': $TOKENS,
    'profile': '$BASE_PROFILE',
}
print(f'  dec_tps_median = {tps:.3f}' if tps else f'  (no tps found in bench output)')
with open('$out', 'a') as f:
    f.write(__import__('json').dumps(row) + '\n')
"
    rm -f "$tmp_profile" "$bench_json"
done

echo
echo "[autotune_sweep] complete. Results: $out"
echo
echo "Sorted by dec_tps_median (best first):"
python3 -c "
import json
rows = []
for line in open('$out'):
    if line.strip(): rows.append(json.loads(line))
rows.sort(key=lambda r: -(r.get('tokens_per_sec_median') or 0))
print(f\"{'value':30s}  {'dec_tps':>10s}\")
print('-' * 45)
for r in rows:
    tps = r.get('tokens_per_sec_median')
    tps_s = f'{tps:.3f}' if tps else 'N/A'
    print(f\"{r['value']:30s}  {tps_s:>10s}\")"
