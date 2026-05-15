#!/usr/bin/env bash
#
# tools/bench/multi_prompt_bench.sh — context-aware dec_tps measurement
#
# Sweeps a set of prompts spanning short / medium / long context lengths
# across N trials each, parses the `[stats]` line from `dismantle generate`,
# and emits a JSON-lines log + a per-prompt summary table.
#
# Motivation: the v2.2.0 `decode` bench uses a fixed 4-token prompt. Several
# rejected levers (A1 flash-attn, A3 add-rmsnorm) may have lost specifically
# because the bench's seq_len stayed in the 5-70 range. This harness measures
# at the workload lengths real users hit.
#
# USAGE
#   tools/bench/multi_prompt_bench.sh <profile-json> <out-dir>
#     [--trials N] [--max-tokens N] [--prompts-file path]
#     [--speculate MODE] [--verify-window N]
#
#   Default trials=3, max-tokens=96, prompts-file=tools/bench/multi_prompt_suite.txt.
#
# OUTPUT
#   $OUT/runs.jsonl            — one line per (prompt, trial)
#   $OUT/summary.md            — markdown table per prompt × median dec_tps
#
# PROMPT FILE FORMAT
#   One prompt per line. Lines starting with `#` ignored. Each prompt
#   gets an auto-generated id (p001, p002, ...). Multi-line prompts are
#   NOT supported (use a single very long line).

set -euo pipefail
cd "$(dirname "$0")/../.."

PROFILE_JSON="${1:-}"
OUT_DIR="${2:-}"
if [[ -z "$PROFILE_JSON" || -z "$OUT_DIR" ]]; then
    echo "usage: $0 <profile-json> <out-dir> [--trials N] [--max-tokens N] [--prompts-file path] [--speculate MODE] [--verify-window N]" >&2
    exit 2
fi
shift 2

TRIALS=3
MAX_TOKENS=96
PROMPTS_FILE="tools/bench/multi_prompt_suite.txt"
SPECULATE=""
VERIFY_WINDOW=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trials)        TRIALS="$2";        shift 2 ;;
        --max-tokens)    MAX_TOKENS="$2";    shift 2 ;;
        --prompts-file)  PROMPTS_FILE="$2";  shift 2 ;;
        --speculate)     SPECULATE="$2";     shift 2 ;;
        --verify-window) VERIFY_WINDOW="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

WEIGHTS="models/deepseek-v2-lite-q4.gguf"
BIN="./target/release/dismantle"

mkdir -p "$OUT_DIR"
RUNS="$OUT_DIR/runs.jsonl"
SUMMARY="$OUT_DIR/summary.md"
: > "$RUNS"

if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "prompts file not found: $PROMPTS_FILE" >&2
    exit 2
fi

# Read prompts, skipping comments and blank lines. macOS ships bash 3.2
# which lacks `mapfile`, so use a portable read loop.
PROMPTS=()
while IFS= read -r line; do
    PROMPTS+=("$line")
done < <(awk 'NF && !/^#/' "$PROMPTS_FILE")
echo "[bench] profile=$PROFILE_JSON trials=$TRIALS max-tokens=$MAX_TOKENS prompts=${#PROMPTS[@]} speculate=${SPECULATE:-off}" >&2

# Parse one `[stats]` line into key=value pairs. Returns a JSON object.
parse_stats() {
    awk '
    /^\[stats\]/ {
        for (i = 2; i <= NF; i++) {
            n = split($i, kv, "=");
            if (n == 2) {
                # quote string values; numbers are not quoted but
                # everything from dismantle is numeric except `reason` and `profile`
                if (kv[1] == "reason" || kv[1] == "profile") {
                    printf "\"%s\":\"%s\"%s", kv[1], kv[2], (i==NF)?"":",";
                } else {
                    printf "\"%s\":%s%s", kv[1], kv[2], (i==NF)?"":",";
                }
            }
        }
        printf "\n";
    }'
}

# Run one (prompt, trial), append a JSONL line.
run_one() {
    local pid="$1" trial="$2" prompt="$3"
    local cmd=("$BIN" generate
        --weights "$WEIGHTS"
        --kernel-profile "$PROFILE_JSON"
        --prompt "$prompt"
        --max-new-tokens "$MAX_TOKENS"
        --temperature 0)
    if [[ -n "$SPECULATE" ]]; then
        cmd+=(--speculate "$SPECULATE" --verify-window "$VERIFY_WINDOW")
    fi
    local out
    out=$(nice -n 19 taskpolicy -b "${cmd[@]}" 2>&1 || true)
    local stats_json
    stats_json=$(printf '%s\n' "$out" | parse_stats)
    if [[ -z "$stats_json" ]]; then
        echo "[bench] WARN $pid trial=$trial: no [stats] line" >&2
        return
    fi
    # Trim trailing comma if any (shouldn't happen) and wrap with prompt id.
    # Note: dismantle's `[stats]` line includes a `profile=<id>` token; we
    # use `profile_path` for the json filename to avoid the duplicate key.
    printf '{"id":"%s","trial":%d,%s,"profile_path":"%s","speculate":"%s","prompt_chars":%d}\n' \
        "$pid" "$trial" "${stats_json%,}" "$PROFILE_JSON" "${SPECULATE:-off}" "${#prompt}" \
        >> "$RUNS"
}

pid_idx=0
for prompt in "${PROMPTS[@]}"; do
    pid_idx=$((pid_idx + 1))
    pid=$(printf 'p%03d' "$pid_idx")
    echo "[bench] $pid len=${#prompt} chars: ${prompt:0:60}..." >&2
    for trial in $(seq 1 "$TRIALS"); do
        run_one "$pid" "$trial" "$prompt"
        sleep 2
    done
done

# Build the summary table. Median dec_tps per prompt id.
python3 - <<PYEOF > "$SUMMARY"
import json, statistics, sys
from collections import defaultdict

rows_by_id = defaultdict(list)
prompt_chars = {}
with open("$RUNS") as f:
    for line in f:
        if not line.strip(): continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = r["id"]
        rows_by_id[rid].append(r)
        prompt_chars[rid] = r.get("prompt_chars", 0)

print("# multi_prompt_bench summary")
print()
print(f"profile: \`$PROFILE_JSON\`  ")
print(f"trials/prompt: $TRIALS  ")
print(f"max-new-tokens: $MAX_TOKENS  ")
print(f"speculate: \`${SPECULATE:-off}\`")
print()
print("| id | prompt chars | prompt tokens | completion | dec_tps (median) | dec_tps (min) | dec_tps (max) | accept rate |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for rid in sorted(rows_by_id):
    rows = rows_by_id[rid]
    tps = sorted(r["dec_tps"] for r in rows)
    med = statistics.median(tps)
    mn, mx = tps[0], tps[-1]
    pt = rows[0].get("prompt", 0)
    ct = rows[0].get("completion", 0)
    da = sum(r.get("draft_accepted", 0) for r in rows)
    dr = sum(r.get("draft_rejected", 0) for r in rows)
    accept = f"{da/(da+dr)*100:.0f}%" if (da+dr) > 0 else "—"
    print(f"| {rid} | {prompt_chars[rid]} | {pt} | {ct} | {med:.2f} | {mn:.2f} | {mx:.2f} | {accept} |")
PYEOF

echo "[bench] done. summary: $SUMMARY  raw: $RUNS"
