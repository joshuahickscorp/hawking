#!/usr/bin/env bash
# expand-baseline.sh — opportunistic overnight token-baseline expansion.
#
# Captures a 12-prompt 3-token greedy regression baseline against the
# real model, waiting for safe memory windows so it coexists with slm
# training. Memory-aware via tools/haul/coexist.sh probe; uses the same
# nice/taskpolicy QoS scheme as the haul runner.
#
# Output:
#   tests/golden/_phase1_token_baseline_expanded.hashes
#   docs/archive/phase-history/_phase1_token_baseline_expanded.log
#
# Behavior:
#   - Outer loop probes every 60s; sleeps when not safe.
#   - When safe, walks the prompt list and captures each not-yet-done
#     entry, re-probing between captures so a closing window aborts
#     the cycle cleanly.
#   - Idempotent: re-runs skip prompt-ids already in the output file.
#   - Wall-clock budget caps total runtime (default 12 hr).
#
# Caveat: do NOT run concurrently with run-gates.sh — both load the
# 9 GB model and would OOM the box together. Run after the haul, or
# while no haul is active.
#
# Tunables (env):
#   WALL_BUDGET_S=43200          # 12 hr default
#   PROBE_INTERVAL_S=60          # outer-loop sleep when not safe
#   INTER_PROMPT_COOLDOWN_S=30   # between captures inside a safe window
#   N_TOKENS=3                   # per spec § CE-4
#   PROMPT_FILE=<path>           # one "<id>:<prompt>" per line, override default 12

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COEXIST="$REPO_ROOT/tools/haul/coexist.sh"
DISMANTLE="$REPO_ROOT/target/release/dismantle"
MODEL="$REPO_ROOT/models/deepseek-v2-lite-q4.gguf"
# OUT_OVERRIDE / LOG_OVERRIDE let the haul runner point this script at
# alternate output files (e.g. tests/golden/_phase1_token_baseline_50.hashes)
# without forking the script. Defaults preserve the legacy expanded path.
OUT="${OUT_OVERRIDE:-$REPO_ROOT/tests/golden/_phase1_token_baseline_expanded.hashes}"
LOG="${LOG_OVERRIDE:-$REPO_ROOT/docs/archive/phase-history/_phase1_token_baseline_expanded.log}"

WALL_BUDGET_S="${WALL_BUDGET_S:-43200}"
PROBE_INTERVAL_S="${PROBE_INTERVAL_S:-60}"
INTER_PROMPT_COOLDOWN_S="${INTER_PROMPT_COOLDOWN_S:-30}"
N_TOKENS="${N_TOKENS:-3}"
# Stall watchdog. The haul-locked value (run-gates smoke gate) is 60_000 ms,
# but cold-mmap prefill on the 9 GB model under any pressure can blow past
# that. For overnight expansion we give the first forward step 3 min before
# bailing. The token-emit fast path is unaffected once prefill clears.
MAX_STALL_MS="${MAX_STALL_MS:-180000}"
STDERR_DIR="$REPO_ROOT/tools/haul/expand-baseline-stderr"
mkdir -p "$STDERR_DIR"

# Default 12-prompt suite: narrative / code / structured / multilingual /
# chat — diverse enough to surface drift across attention heads, MoE
# routing, and tokenizer paths. IDs are stable so re-runs hash to the
# same row.
DEFAULT_PROMPTS=(
    "p001:Once upon a time"
    "p002:def quicksort(arr):"
    "p003:The capital of France is"
    "p004:To be or not to be"
    "p005:import numpy as np"
    "p006:The quick brown fox"
    "p007:SELECT * FROM users WHERE"
    "p008:Roses are red, violets are"
    "p009:fn main() {"
    "p010:Translate to French: hello"
    "p011:Q: What is 2+2? A:"
    "p012:Markdown header"
)

if [[ -n "${PROMPT_FILE:-}" ]]; then
    PROMPTS=()
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        PROMPTS+=("$line")
    done <"$PROMPT_FILE"
else
    PROMPTS=("${DEFAULT_PROMPTS[@]}")
fi

START_TS=$(date +%s)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() {
    local msg="[expand-baseline $(ts)] $*"
    printf '%s\n' "$msg"
    printf '%s\n' "$msg" >>"$LOG"
}

elapsed_s() { echo $(( $(date +%s) - START_TS )); }
over_budget() { [[ $(elapsed_s) -ge $WALL_BUDGET_S ]]; }

if [[ -z "${SLM_PID:-}" ]]; then
    detected=$(pgrep -f 'mamba_byte_train' 2>/dev/null | head -1)
    if [[ -n "$detected" ]]; then
        export SLM_PID="$detected"
    fi
fi

if command -v b3sum >/dev/null 2>&1; then
    HASH_ALGO="blake3"; HASH_CMD=(b3sum --no-names)
else
    HASH_ALGO="sha256"; HASH_CMD=(shasum -a 256)
fi
run_hash() {
    if [[ "$HASH_ALGO" == "blake3" ]]; then "${HASH_CMD[@]}"; else "${HASH_CMD[@]}" | awk '{print $1}'; fi
}

[[ -x "$COEXIST" ]]   || { echo "missing $COEXIST"; exit 1; }
[[ -x "$DISMANTLE" ]] || { echo "missing $DISMANTLE — run cargo build --release"; exit 1; }
[[ -f "$MODEL" ]]     || { echo "missing $MODEL"; exit 1; }

if [[ ! -f "$OUT" ]]; then
    cat >"$OUT" <<EOF
# Phase 1 token-output baseline — expanded suite (overnight)
# Captured by tools/haul/expand-baseline.sh; sibling of the locked
# _phase1_token_baseline.hashes. Accumulates over memory windows.
# Format: <prompt-id> <max-new-tokens> <hash-hex> <prompt-text>
# algo: $HASH_ALGO
# Generation: temp=0 greedy, max_new_tokens=$N_TOKENS, model=DeepSeek-V2-Lite-Chat-Q4_K_M
EOF
fi

captured_ids() { grep -oE '^p[0-9]+' "$OUT" 2>/dev/null | sort -u; }

probe_state() {
    # Capture once; under `set -o pipefail` the probe's nonzero exit
    # would otherwise make `|| echo unknown` fire AND keep the python
    # output, producing a literal "degraded\nunknown" log line.
    local s
    s=$("$COEXIST" probe --json 2>/dev/null \
        | python3 -c '\''import json,sys
try: print(json.load(sys.stdin).get("state","unknown"))
except Exception: print("unknown")'\'' 2>/dev/null) || true
    [[ -z "$s" ]] && s="unknown"
    printf '\''%s\n'\'' "$s"
}

capture_one() {
    local entry="$1"
    local pid="${entry%%:*}"
    local prompt="${entry#*:}"

    if captured_ids | grep -qx "$pid"; then
        return 0
    fi

    log "$pid: capturing (prompt=${prompt:0:40})"
    local out rc
    local stderr_log="$STDERR_DIR/${pid}.stderr"
    out=$("$DISMANTLE" generate \
        --weights "$MODEL" \
        --prompt "$prompt" \
        --max-new-tokens "$N_TOKENS" \
        --temperature 0 \
        --max-stall-ms "$MAX_STALL_MS" \
        2>"$stderr_log") ; rc=$?

    # Pull the engine's [stats] line if present — single source of truth
    # for *why* a capture might have come back empty.
    local stats_line
    stats_line=$(grep -m1 '^\[stats\]' "$stderr_log" 2>/dev/null || true)

    if [[ $rc -ne 0 ]]; then
        log "$pid: dismantle exited rc=$rc — ${stats_line:-(no stats line; see $stderr_log)}"
        return 1
    fi
    if [[ ${#out} -lt 1 ]]; then
        log "$pid: empty output — ${stats_line:-(no stats; see $stderr_log)}"
        return 1
    fi

    local hash; hash=$(printf '%s' "$out" | run_hash)
    local prompt_escaped="${prompt//$'\n'/\\n}"
    printf '%s %s %s %s\n' "$pid" "$N_TOKENS" "$hash" "$prompt_escaped" >>"$OUT"
    log "$pid: captured hash=$hash (${#out}B)"
    return 0
}

trap 'log "signal received — exiting cleanly"; exit 0' INT TERM

log "expand-baseline starting"
log "  out=$OUT"
log "  log=$LOG"
log "  budget=${WALL_BUDGET_S}s probe-interval=${PROBE_INTERVAL_S}s prompts=${#PROMPTS[@]}"
log "  SLM_PID=${SLM_PID:-unset}"
log "  already-captured: $(captured_ids | tr '\n' ' ')"

ITER=0
THIS_RUN=0
LAST_LOGGED_NONSAFE=0

while ! over_budget; do
    ITER=$((ITER + 1))
    n_captured=$(captured_ids | wc -l | tr -d ' ')
    if [[ $n_captured -ge ${#PROMPTS[@]} ]]; then
        log "all ${#PROMPTS[@]} prompts captured — exiting"
        break
    fi

    state=$(probe_state)
    if [[ "$state" != "safe" ]]; then
        # Quiet logging while waiting — every 10th iter (~10 min default).
        if [[ $((ITER - LAST_LOGGED_NONSAFE)) -ge 10 ]]; then
            log "iter=$ITER captured=$n_captured/${#PROMPTS[@]} probe=$state — waiting"
            LAST_LOGGED_NONSAFE=$ITER
        fi
        sleep "$PROBE_INTERVAL_S"
        continue
    fi

    log "iter=$ITER captured=$n_captured/${#PROMPTS[@]} probe=safe — opening window"

    for entry in "${PROMPTS[@]}"; do
        pid_only="${entry%%:*}"
        if captured_ids | grep -qx "$pid_only"; then continue; fi

        cur=$(probe_state)
        if [[ "$cur" != "safe" ]]; then
            log "window closed mid-cycle (state=$cur); back to outer wait"
            break
        fi

        if capture_one "$entry"; then
            THIS_RUN=$((THIS_RUN + 1))
            sleep "$INTER_PROMPT_COOLDOWN_S"
        fi

        if over_budget; then break; fi
    done
done

n_captured=$(captured_ids | wc -l | tr -d ' ')
log "done — this-run=$THIS_RUN total=$n_captured/${#PROMPTS[@]} elapsed=$(elapsed_s)s"
