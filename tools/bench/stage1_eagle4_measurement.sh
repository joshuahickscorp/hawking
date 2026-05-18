#!/usr/bin/env bash
# stage1_eagle4_measurement.sh — path-to-90 execution-plan step 10.
#
# Benches `dismantle generate --speculate eagle4` against
# `--speculate off` on a small prompt suite + decoding length, producing
# the Stage 1 measurement number for the block-ship gate (target band:
# 12-22 tok/s; lower bound for "ship": 18 tok/s).
#
# RUN THIS AFTER Cmd-Q'ing the Claude desktop app. Bench numbers are
# contaminated 4-5× by an active Claude session (see bench scripts'
# README in tools/bench/). Script bails if Claude.app is still up.
#
# Usage:
#   cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
#   bash tools/bench/stage1_eagle4_measurement.sh
#
# Markers (for the next Claude session to read):
#   reports/path_to_90/_stage1_capture/STATUS.log    ← progress log
#   reports/path_to_90/_stage1_capture/raw.json      ← parsed metrics
#   reports/path_to_90/_stage1_capture/off_t{N}.log  ← Off mode output
#   reports/path_to_90/_stage1_capture/e4_t{N}.log   ← Eagle4 output
#   reports/path_to_90/_stage1_capture/DONE          ← created on success
#   reports/path_to_90/_stage1_capture/FAILED        ← created on failure
#
# Expected outcome (per architecture_closeout.md):
#   Off    : ~25 tok/s baseline
#   Eagle4 : ~0.2-0.3 tok/s (slow CPU-walk capture path)
#   → Stage 1 HALTS at the speed gate; that's the right signal pointing
#     at GPU-side eagle4 capture as the next architectural unlock.
#
# Estimated runtime: 4-8 minutes for Off (3 prompts × 16 tokens × ~40ms);
# 30-60 minutes for Eagle4 (3 prompts × 16 tokens × ~3.7s).

set -uo pipefail
cd "$(dirname "$0")/../.."

OUT_DIR="reports/path_to_90/_stage1_capture"
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
    notify "path-to-90 step 10" "stage1 FAILED: $*" "Sosumi"
    (say "stage one measurement failed" &) 2>/dev/null || true
    exit 1
}

trap 'fail "interrupted (signal)"' INT TERM

log "=== stage1_eagle4_measurement starting ==="
log "Worktree: $(pwd)"
log "Branch:   $(git branch --show-current)"
log "Commit:   $(git rev-parse HEAD)"

# ── 1) preflight ───────────────────────────────────────────────────
if pgrep -f "/Applications/Claude.app/Contents/MacOS/Claude" >/dev/null 2>&1; then
    log "Claude.app processes still running:"
    pgrep -af "/Applications/Claude.app/Contents/MacOS/Claude" | head -5 | tee -a "$STATUS"
    fail "Cmd-Q the Claude desktop app first, then re-run this script."
fi
log "✓ Claude.app: not running"

# Pause slm if running.
SLM_PID=$(pgrep -f "overnight_shift.py" 2>/dev/null | head -1 || true)
PAUSED_PIDS=""
if [[ -n "$SLM_PID" ]]; then
    SLM_PGID=$(ps -o pgid= -p "$SLM_PID" 2>/dev/null | tr -d ' ')
    PAUSED_PIDS=$(pgrep -g "$SLM_PGID" 2>/dev/null | tr '\n' ' ')
    log "Pausing slm tree (pgid=$SLM_PGID): $PAUSED_PIDS"
    kill -STOP $PAUSED_PIDS 2>/dev/null || true
    sleep 2
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

# ── 2) build ──────────────────────────────────────────────────────
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
TOKENS=16  # Short — Eagle4 mode is slow per the CPU walk.

for f in "$BIN" "$WEIGHTS" "$PROFILE" "$HEAD_NPZ" "$FROZEN_NPZ"; do
    if [[ ! -e "$f" ]]; then
        fail "missing artifact: $f"
    fi
done
log "✓ all artifacts present"

# Prompts: short, representative; each mode runs all three.
PROMPTS=(
    "The quick brown fox"
    "Once upon a time"
    "def fibonacci(n):"
)

# ── 3) Off-mode baseline ──────────────────────────────────────────
log ""
log "── Off-mode greedy baseline ($TOKENS tokens each) ──"
OFF_TIMES=()
OFF_TPS=()
for i in "${!PROMPTS[@]}"; do
    p="${PROMPTS[$i]}"
    out="$OUT_DIR/off_t${i}.log"
    log "  prompt $i: ${p:0:30}..."
    t0=$(date +%s.%N)
    if ! "$BIN" generate \
            --weights "$WEIGHTS" \
            --kernel-profile "$PROFILE" \
            --prompt "$p" \
            --max-new-tokens "$TOKENS" \
            >"$out" 2>&1; then
        fail "Off generate prompt $i failed — see $out"
    fi
    t1=$(date +%s.%N)
    dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{ printf "%.3f", b - a }')
    tps=$(awk -v t="$TOKENS" -v d="$dt" 'BEGIN{ printf "%.3f", t / d }')
    log "    wall=${dt}s  tps≈${tps}"
    OFF_TIMES+=("$dt")
    OFF_TPS+=("$tps")
done

# ── 4) Eagle4-mode bench ──────────────────────────────────────────
log ""
log "── Eagle4 spec decode ($TOKENS tokens each; CPU-walk capture, slow) ──"
E4_TIMES=()
E4_TPS=()
E4_ACCEPT=()
for i in "${!PROMPTS[@]}"; do
    p="${PROMPTS[$i]}"
    out="$OUT_DIR/e4_t${i}.log"
    log "  prompt $i: ${p:0:30}... (this will take several minutes)"
    t0=$(date +%s.%N)
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
    t1=$(date +%s.%N)
    dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{ printf "%.3f", b - a }')
    tps=$(awk -v t="$TOKENS" -v d="$dt" 'BEGIN{ printf "%.3f", t / d }')
    accept=$(grep -oE 'draft_accepted=[0-9]+' "$out" | tail -1 | sed 's/.*=//' || echo "?")
    reject=$(grep -oE 'draft_rejected=[0-9]+' "$out" | tail -1 | sed 's/.*=//' || echo "?")
    log "    wall=${dt}s  tps≈${tps}  draft=${accept}a/${reject}r"
    E4_TIMES+=("$dt")
    E4_TPS+=("$tps")
    E4_ACCEPT+=("${accept}/${reject}")
done

# ── 5) raw.json ───────────────────────────────────────────────────
OFF_TPS_JSON=$(printf '%s,' "${OFF_TPS[@]}" | sed 's/,$//')
E4_TPS_JSON=$(printf '%s,' "${E4_TPS[@]}" | sed 's/,$//')
OFF_MEDIAN=$(printf '%s\n' "${OFF_TPS[@]}" | sort -n | awk 'NR==int((NR+1)/2)' | head -1)
E4_MEDIAN=$(printf '%s\n' "${E4_TPS[@]}" | sort -n | awk 'NR==int((NR+1)/2)' | head -1)

cat >"$RAW" <<JSON
{
  "captured_at": "$(date -Iseconds)",
  "git_sha": "$(git rev-parse HEAD)",
  "branch": "$(git branch --show-current)",
  "host": { "model": "M3 Pro 18 GB" },
  "tokens_per_run": $TOKENS,
  "prompts": [$(printf '"%s",' "${PROMPTS[@]}" | sed 's/,$//')],
  "off_mode": {
    "tps_per_prompt": [$OFF_TPS_JSON],
    "median_tps": $OFF_MEDIAN
  },
  "eagle4_mode": {
    "tps_per_prompt": [$E4_TPS_JSON],
    "median_tps": $E4_MEDIAN,
    "accept_reject_per_prompt": [$(printf '"%s",' "${E4_ACCEPT[@]}" | sed 's/,$//')]
  },
  "block_ship_gate": {
    "lower_bound_tps": 18,
    "upper_bound_tps": 24,
    "passed": false
  }
}
JSON
log ""
log "── Summary ──"
log "Off    median tps: $OFF_MEDIAN"
log "Eagle4 median tps: $E4_MEDIAN"
log ""

# Compare against block-ship gate.
GATE_PASS=$(awk -v x="$E4_MEDIAN" 'BEGIN{ print (x >= 18) ? "PASS" : "HALT" }')
log "Block-ship gate (≥ 18 tok/s for Stage 1): $GATE_PASS"
if [[ "$GATE_PASS" == "HALT" ]]; then
    log ""
    log "Stage 1 HALT triggered — Eagle4 dec_tps below 18 tok/s lower bound."
    log "This is expected per architecture_closeout.md: CPU-walk capture is"
    log "the bottleneck. Next architectural unlock: GPU-side eagle4 capture."
    log "Per execution_plan.md § Stage 1: < 15 tok/s → reinject Stage 0.5 OR"
    log "land GPU-side capture first."
fi

# ── 6) DONE ────────────────────────────────────────────────────────
log ""
log "=== stage1_eagle4_measurement DONE ==="
log ""
log "Next: relaunch Claude Code in the worktree and tell it:"
log "  'step 10 done — read reports/path_to_90/_stage1_capture/'"

touch "$DONE_MARKER"
notify "path-to-90 step 10" "stage1 capture complete — return to Claude" "Glass"
(say "stage one measurement complete" &) 2>/dev/null || true
exit 0
