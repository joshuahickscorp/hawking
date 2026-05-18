#!/usr/bin/env bash
# stage1_remeasurement.sh — path-to-90 step 10, after GPU-side eagle4 capture.
#
# Same workload as the first Stage 1 capture (commit 94f6068's
# stage1_eagle4_measurement.sh) but uses dismantle's own
# `[stats] dec_tps=…` line as the canonical decode-only number instead
# of the script's wall-clock measurement (which includes per-process
# prefill + model load cost and undercounts by ~10× on short prompts).
#
# RUN THIS AFTER Cmd-Q'ing the Claude desktop app. Bench numbers are
# contaminated 4-5× by an active Claude session. Script bails if
# Claude.app is still up.
#
# Usage:
#   cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
#   bash tools/bench/stage1_remeasurement.sh
#
# Markers (for the next Claude session to read):
#   reports/path_to_90/_stage1_remeasurement/STATUS.log     ← progress log
#   reports/path_to_90/_stage1_remeasurement/raw.json       ← parsed metrics
#   reports/path_to_90/_stage1_remeasurement/off_t{N}.log   ← Off output + [stats]
#   reports/path_to_90/_stage1_remeasurement/e4_t{N}.log    ← Eagle4 output + per-step accept/reject
#   reports/path_to_90/_stage1_remeasurement/DONE           ← created on success
#   reports/path_to_90/_stage1_remeasurement/FAILED         ← created on failure
#
# Expected outcome (per commit 679c077's smoke run):
#   Off    : ~27 dec_tps  (unchanged — same Wedge C path as before)
#   Eagle4 : ~2 dec_tps   (up 3.5× from CPU-walk's 0.54)
#   Eagle4 accept: ~85-95 % (up from 2 % with CPU-walk hiddens — head is now
#                            seeing GPU-sourced hiddens, in-distribution
#                            with its MLX bf16 training data)
#   → Stage 1 gate still HALTS at 18 tps but with the architectural
#     progression visible. Next unlock: Metal-accelerated Eagle4Head
#     forward (step 7 from execution_plan.md). The 165 ms/token CPU
#     LM head gemv dominates the remaining decode time.
#
# Estimated runtime:
#   Off:    ~30 s total (3 prompts × 16 tokens × ~37 ms decode + model load)
#   Eagle4: ~5-8 minutes (3 prompts × 16 tokens × ~530 ms decode + model load)

set -uo pipefail
cd "$(dirname "$0")/../.."

OUT_DIR="reports/path_to_90/_stage1_remeasurement"
mkdir -p "$OUT_DIR"
STATUS="$OUT_DIR/STATUS.log"
RAW="$OUT_DIR/raw.json"
DONE_MARKER="$OUT_DIR/DONE"
FAIL_MARKER="$OUT_DIR/FAILED"

rm -f "$DONE_MARKER" "$FAIL_MARKER"
: > "$STATUS"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$STATUS"
}

notify() {
    osascript -e "display notification \"$2\" with title \"$1\" sound name \"$3\"" 2>/dev/null || true
}

fail() {
    log "FAILED: $*"
    touch "$FAIL_MARKER"
    notify "path-to-90 step 10 remeasure" "stage1 remeasure FAILED: $*" "Sosumi"
    (say "stage one remeasurement failed" &) 2>/dev/null || true
    exit 1
}

trap 'fail "interrupted (signal)"' INT TERM

log "=== stage1_remeasurement starting ==="
log "Worktree: $(pwd)"
log "Branch:   $(git branch --show-current)"
log "Commit:   $(git rev-parse HEAD)"

# ── preflight ─────────────────────────────────────────────────────
if pgrep -f "/Applications/Claude.app/Contents/MacOS/Claude" >/dev/null 2>&1; then
    log "Claude.app processes still running:"
    pgrep -af "/Applications/Claude.app/Contents/MacOS/Claude" | head -5 | tee -a "$STATUS"
    fail "Cmd-Q the Claude desktop app first, then re-run this script."
fi
log "✓ Claude.app: not running"

SLM_PID=$(pgrep -f "overnight_shift.py" 2>/dev/null | head -1 || true)
PAUSED_PIDS=""
if [[ -n "$SLM_PID" ]]; then
    SLM_PGID=$(ps -o pgid= -p "$SLM_PID" 2>/dev/null | tr -d ' ')
    PAUSED_PIDS=$(pgrep -g "$SLM_PGID" 2>/dev/null | tr '\n' ' ')
    log "Pausing slm tree (pgid=$SLM_PGID): $PAUSED_PIDS"
    kill -STOP $PAUSED_PIDS 2>/dev/null || true
    sleep 2
    log "✓ slm paused"
else
    log "✓ slm: not running"
fi

restore_slm() {
    if [[ -n "$PAUSED_PIDS" ]]; then
        log "Resuming slm: $PAUSED_PIDS"
        kill -CONT $PAUSED_PIDS 2>/dev/null || true
    fi
}
trap 'restore_slm' EXIT

# ── build ─────────────────────────────────────────────────────────
log "Building --release (incremental) ..."
if ! cargo build --release --workspace 2>>"$STATUS" >/dev/null; then
    fail "cargo build failed — see STATUS.log"
fi
log "✓ build OK"

BIN="./target/release/dismantle"
WEIGHTS="models/deepseek-v2-lite-q4.gguf"
PROFILE="profiles/deepseek-v2-lite-q4.m3pro18.json"
HEAD_NPZ="eagle4/checkpoints/eagle4_v3/best.npz"
FROZEN_NPZ="eagle4/v2lite_frozen.npz"
TOKENS=16

for f in "$BIN" "$WEIGHTS" "$PROFILE" "$HEAD_NPZ" "$FROZEN_NPZ"; do
    [[ -e "$f" ]] || fail "missing artifact: $f"
done
log "✓ all artifacts present"

# Helper: extract `dec_tps=X.XX` from a generate output log.
extract_dec_tps() {
    grep -oE 'dec_tps=[0-9.]+' "$1" | tail -1 | sed 's/.*=//'
}
extract_accept_reject() {
    local a r
    a=$(grep -oE 'draft_accepted=[0-9]+' "$1" | tail -1 | sed 's/.*=//' || echo 0)
    r=$(grep -oE 'draft_rejected=[0-9]+' "$1" | tail -1 | sed 's/.*=//' || echo 0)
    echo "$a/$r"
}

PROMPTS=(
    "The quick brown fox"
    "Once upon a time"
    "def fibonacci(n):"
)

# ── Off-mode baseline ─────────────────────────────────────────────
log ""
log "── Off-mode greedy baseline ($TOKENS tokens × ${#PROMPTS[@]} prompts) ──"
OFF_DECTPS=()
for i in "${!PROMPTS[@]}"; do
    p="${PROMPTS[$i]}"
    out="$OUT_DIR/off_t${i}.log"
    log "  prompt $i: ${p:0:30}..."
    if ! "$BIN" generate \
            --weights "$WEIGHTS" \
            --kernel-profile "$PROFILE" \
            --prompt "$p" \
            --max-new-tokens "$TOKENS" \
            >"$out" 2>&1; then
        fail "Off generate prompt $i failed — see $out"
    fi
    tps=$(extract_dec_tps "$out")
    log "    dec_tps=$tps"
    OFF_DECTPS+=("$tps")
done

# ── Eagle4-mode bench ─────────────────────────────────────────────
log ""
log "── Eagle4 spec decode (K=1, GPU-side capture) ──"
E4_DECTPS=()
E4_AR=()
for i in "${!PROMPTS[@]}"; do
    p="${PROMPTS[$i]}"
    out="$OUT_DIR/e4_t${i}.log"
    log "  prompt $i: ${p:0:30}..."
    if ! DISMANTLE_SPEC_LOG=1 "$BIN" generate \
            --weights "$WEIGHTS" \
            --kernel-profile "$PROFILE" \
            --prompt "$p" \
            --max-new-tokens "$TOKENS" \
            --speculate eagle4 \
            --draft-head "$HEAD_NPZ" \
            --eagle4-frozen "$FROZEN_NPZ" \
            >"$out" 2>&1; then
        fail "Eagle4 generate prompt $i failed — see $out"
    fi
    tps=$(extract_dec_tps "$out")
    ar=$(extract_accept_reject "$out")
    log "    dec_tps=$tps  draft=$ar"
    E4_DECTPS+=("$tps")
    E4_AR+=("$ar")
done

# ── raw.json ──────────────────────────────────────────────────────
OFF_TPS_JSON=$(printf '%s,' "${OFF_DECTPS[@]}" | sed 's/,$//')
E4_TPS_JSON=$(printf '%s,' "${E4_DECTPS[@]}" | sed 's/,$//')
OFF_MEDIAN=$(printf '%s\n' "${OFF_DECTPS[@]}" | sort -n | awk 'NR==int((NR+1)/2)' | head -1)
E4_MEDIAN=$(printf '%s\n' "${E4_DECTPS[@]}" | sort -n | awk 'NR==int((NR+1)/2)' | head -1)

# Total accept rate across all runs.
total_a=0
total_r=0
for ar in "${E4_AR[@]}"; do
    a=${ar%/*}
    r=${ar#*/}
    total_a=$((total_a + a))
    total_r=$((total_r + r))
done
total=$((total_a + total_r))
accept_pct=$(awk -v a="$total_a" -v n="$total" 'BEGIN{ if (n>0) printf "%.1f", 100*a/n; else print "n/a" }')

cat >"$RAW" <<JSON
{
  "captured_at": "$(date -Iseconds)",
  "git_sha": "$(git rev-parse HEAD)",
  "branch": "$(git branch --show-current)",
  "host": { "model": "M3 Pro 18 GB" },
  "tokens_per_run": $TOKENS,
  "prompts": [$(printf '"%s",' "${PROMPTS[@]}" | sed 's/,$//')],
  "off_mode": {
    "dec_tps_per_prompt": [$OFF_TPS_JSON],
    "median_dec_tps": $OFF_MEDIAN
  },
  "eagle4_mode": {
    "dec_tps_per_prompt": [$E4_TPS_JSON],
    "median_dec_tps": $E4_MEDIAN,
    "accept_reject_per_prompt": [$(printf '"%s",' "${E4_AR[@]}" | sed 's/,$//')],
    "total_accept_pct": $accept_pct
  },
  "block_ship_gate": {
    "lower_bound_tps": 18,
    "upper_bound_tps": 24
  }
}
JSON

log ""
log "── Summary ──"
log "Off    median dec_tps: $OFF_MEDIAN"
log "Eagle4 median dec_tps: $E4_MEDIAN"
log "Eagle4 draft acceptance: $accept_pct %  (total accepts ${total_a}/${total} drafts)"
log ""

GATE_PASS=$(awk -v x="$E4_MEDIAN" 'BEGIN{ print (x >= 18) ? "PASS" : "HALT" }')
log "Block-ship gate (≥ 18 tps for Stage 1): $GATE_PASS"
if [[ "$GATE_PASS" == "HALT" ]]; then
    log ""
    log "Stage 1 gate still HALTS at 18 tps. Architectural progression:"
    log "  - Step 8 (CPU walk):       0.54 dec_tps,  2 % accept"
    log "  - Step 10f (GPU capture):  $E4_MEDIAN dec_tps,  $accept_pct % accept"
    log "  - Step 7 (Metal head):     target ~22 dec_tps (still TODO)"
    log ""
    log "Next unlock: route Eagle4Head's LM head gemv through dismantle's"
    log "existing GPU gemv_f16_argmax_dispatch (the V2-Lite GGUF lm_head is"
    log "already pinned as a Metal buffer). Eliminates the 165 ms/token CPU"
    log "LM head call in the head's CPU forward_full."
fi

log ""
log "=== stage1_remeasurement DONE ==="
log "Next: relaunch Claude Code and tell it:"
log "  'step 10 remeasure done — read reports/path_to_90/_stage1_remeasurement/'"

touch "$DONE_MARKER"
notify "path-to-90 step 10 remeasure" "stage1 remeasurement complete — return to Claude" "Glass"
(say "stage one remeasurement complete" &) 2>/dev/null || true
exit 0
