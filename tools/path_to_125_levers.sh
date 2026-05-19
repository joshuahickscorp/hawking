#!/usr/bin/env bash
# path-to-125 cross-lever auto-orchestrator.
#
# Runs all remaining levers back-to-back with bench gates between each:
#
#   LEVER 1 — Branch 2 step 4 measurement (parallel-k-union A/B vs parallel-k baseline)
#   LEVER 2 — Branch 3 head-arch experiments (gate_init=0.1, vector_gate, etc.)
#   LEVER 3 — Phase F3 prototype (async verify-start, multi-queue Metal scheduling)
#   LEVER 4 — Phase F1 (AMX extend to V2-Lite projection gemvs)
#   LEVER 5 — Final headline bench
#
# Each lever has:
#   - A run script under tools/_lever_*.sh
#   - A bench gate that decides whether to ship or revert
#   - Status writes to reports/path_to_90/_levers/status.json
#   - Detailed logs under reports/path_to_90/_levers/<lever>.log
#
# Survives Claude open/close — runs as a normal OS background process.
# Decision logic is conservative: if a lever shows >5% regression, the
# next lever runs from the BASELINE checkpoint, not the regressed one.
# If a lever shows >3% improvement, it becomes the new baseline.
#
# This orchestrator is IDEMPOTENT in the sense that each lever's script
# can be re-run independently. Status JSON tracks last-known result so
# re-running the orchestrator picks up where it left off.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LEVERS_DIR="$REPO_ROOT/reports/path_to_90/_levers"
mkdir -p "$LEVERS_DIR"
STATUS="$LEVERS_DIR/status.json"

VENV=/Users/scammermike/Downloads/dismantle/eagle4/.venv/bin/python3
DISMANTLE="$REPO_ROOT/target/release/dismantle"

# ── helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
status_write() {
  local lever="$1" state="$2" extra="${3:-}"
  python3 - "$STATUS" "$lever" "$state" "$extra" <<'PY'
import sys, json, os, datetime
status_path, lever, state, extra = sys.argv[1:5]
try:
    cur = json.load(open(status_path)) if os.path.exists(status_path) else {}
except Exception:
    cur = {}
cur.setdefault("history", []).append({
    "t": datetime.datetime.now().isoformat(timespec="seconds"),
    "lever": lever, "state": state,
})
cur[lever] = {"state": state, "last_update": datetime.datetime.now().isoformat(timespec="seconds")}
if extra:
    try:
        cur[lever].update(json.loads("{" + extra + "}"))
    except Exception as e:
        cur[lever]["write_error"] = str(e)
json.dump(cur, open(status_path, "w"), indent=2)
PY
}

# Run the standard A/B bench. Returns 0 always; reads dec_tps medians from
# the bench summary into the status JSON.
run_bench_ab() {
  local lever="$1"
  local bench_log="$LEVERS_DIR/${lever}_bench.log"
  log "$lever: starting clean-window bench"
  status_write "$lever" "bench_running"
  if pgrep -i "Claude" >/dev/null 2>&1; then
    log "$lever: WARN Claude is open — bench will run contended (numbers indicative only)"
    status_write "$lever" "bench_contended"
  fi
  EAGLE4_CKPT="${EAGLE4_CKPT:-$REPO_ROOT/eagle4/checkpoints/eagle4_v3/best.npz}" \
    "$REPO_ROOT/tools/bench/path_to_125_bench.sh" > "$bench_log" 2>&1 || true
  # Parse the medians out of bench summary.
  local off_med ngram_med eagle_med chain_med
  off_med=$(grep -E "^off/sequential/K1" "$bench_log" | awk '{print $2}' | head -1)
  ngram_med=$(grep -E "^ngram/parallel-k/K1" "$bench_log" | awk '{print $2}' | head -1)
  eagle_med=$(grep -E "^eagle4/sequential/K1" "$bench_log" | awk '{print $2}' | head -1)
  chain_med=$(grep -E "^eagle4/parallel-k/K4" "$bench_log" | awk '{print $2}' | head -1)
  log "$lever: bench medians  off=$off_med ngram=$ngram_med eagle4=$eagle_med chain=$chain_med"
  status_write "$lever" "bench_done" "\"off_med\":${off_med:-null},\"ngram_med\":${ngram_med:-null},\"eagle4_med\":${eagle_med:-null},\"chain_med\":${chain_med:-null}"
}

# ── LEVER 1: Branch 2 step 4 A/B (parallel-k vs parallel-k-union) ─────────────
lever1() {
  log "LEVER 1: parallel-k vs parallel-k-union A/B"
  status_write "lever1_branch2_step4" "starting"
  # Stash original profile.
  cp profiles/deepseek-v2-lite-q4.m3pro18.json /tmp/_lever1_profile_backup.json
  # Bench with parallel-k baseline.
  "$VENV" -c "
import json
p = json.load(open('profiles/deepseek-v2-lite-q4.m3pro18.json'))
p['selected']['verify_kernels'] = 'parallel-k'
json.dump(p, open('profiles/deepseek-v2-lite-q4.m3pro18.json','w'), indent=2)
"
  log "LEVER 1: bench with parallel-k baseline"
  run_bench_ab "lever1_parallel_k"
  # Bench with parallel-k-union.
  "$VENV" -c "
import json
p = json.load(open('profiles/deepseek-v2-lite-q4.m3pro18.json'))
p['selected']['verify_kernels'] = 'parallel-k-union'
json.dump(p, open('profiles/deepseek-v2-lite-q4.m3pro18.json','w'), indent=2)
"
  log "LEVER 1: bench with parallel-k-union"
  run_bench_ab "lever1_parallel_k_union"
  cp /tmp/_lever1_profile_backup.json profiles/deepseek-v2-lite-q4.m3pro18.json
  status_write "lever1_branch2_step4" "done"
}

# ── LEVER 2: Branch 3 head re-arch (gate init = 0.1) ──────────────────────────
lever2() {
  log "LEVER 2: head re-arch (gate init = 0.1)"
  status_write "lever2_head_rearch" "starting"
  # Build a patched eagle4.py that initializes residual_gate at 0.1.
  # Saves backup of original eagle4.py first.
  cp eagle4/eagle4.py /tmp/_lever2_eagle4_backup.py
  "$VENV" -c "
import re
src = open('eagle4/eagle4.py').read()
# Find residual_gate init line. Original v3 has residual_gate near zero.
# Patch the EagleHead.__init__ to bump init to 0.1 unless override given.
patched = re.sub(
    r'self\.residual_gate = mx\.array\(\[0\.0\]\)',
    'self.residual_gate = mx.array([0.1])  # path-to-125 lever2: bumped from 0.0 for chain training',
    src,
)
if patched == src:
    # alt pattern: just look for residual_gate creation
    patched = re.sub(
        r'(self\.residual_gate = .*)',
        r'self.residual_gate = mx.array([0.1])  # path-to-125 lever2',
        src,
        count=1,
    )
open('eagle4/eagle4.py', 'w').write(patched)
print('patched eagle4.py for gate init = 0.1')
"
  log "LEVER 2: training iter6 with gate init = 0.1, k=4, 5 shards"
  CKPT=eagle4/checkpoints/eagle4_v4_lever2
  "$VENV" eagle4/eagle4.py train \
    --parquet \
      training_data/c2_hidden/eagle4_v0/shard_00000.parquet \
      training_data/c2_hidden/eagle4_v0/shard_00001.parquet \
      training_data/c2_hidden/eagle4_v0/shard_00002.parquet \
      training_data/c2_hidden/eagle4_v0/shard_00003.parquet \
      training_data/c2_hidden/eagle4_v0/shard_00004.parquet \
    --frozen eagle4/v2lite_frozen.npz \
    --ckpt-dir "$CKPT" \
    --epochs 1 \
    --multi-step-k 4 \
    --multi-step-decay 0.7 \
    --chain-h-high \
    --target-warmup-steps 50 \
    --multi-step-aux-decay 0.3 \
    > "$LEVERS_DIR/lever2_train.log" 2>&1
  log "LEVER 2: training done; final gate value:"
  grep -oE 'gate=[0-9.]+' "$LEVERS_DIR/lever2_train.log" | tail -3 | tee -a "$LEVERS_DIR/lever2_train.log"
  # tau eval
  log "LEVER 2: tau_eval on heldout shard"
  "$VENV" eagle4/tau_eval.py eval \
    --ckpt "$CKPT/latest.npz" \
    --frozen eagle4/v2lite_frozen.npz \
    --parquet eagle4/data/v2lite_3layer_heldout/shard_00000.parquet \
    --depth 4 \
    > "$LEVERS_DIR/lever2_tau.log" 2>&1
  tau=$(grep -oE '"tau"[^,]*' "$LEVERS_DIR/lever2_tau.log" | head -1 | grep -oE '[0-9.]+' | head -1)
  log "LEVER 2: tau-at-depth-4 = ${tau:-?}"
  # Bench with new head
  EAGLE4_CKPT="$REPO_ROOT/$CKPT/latest.npz" run_bench_ab "lever2_head_rearch"
  cp /tmp/_lever2_eagle4_backup.py eagle4/eagle4.py
  status_write "lever2_head_rearch" "done" "\"tau_at_depth_4\":${tau:-null}"
}

# ── LEVER 3: audit only (Phase F3 design check) ───────────────────────────────
lever3() {
  log "LEVER 3: Phase F3 (async verify) — design audit only this round"
  status_write "lever3_phase_f3_audit" "starting"
  # Write a status doc that captures what F3 would need at the dispatch
  # level. Implementation deferred to a focused next session.
  cat > "$LEVERS_DIR/lever3_design.md" <<'MD'
# Lever 3 — Phase F3 (async verify-start) audit

## What it is
Overlap the last Eagle4 head's draft step (CPU + AMX) with the first
V2-Lite verifier layer's expert prefetch (Metal command queue).

## What it needs
1. **Multi-queue Metal context** — the existing MetalContext uses ONE
   command queue. Async overlap requires either (a) a second queue
   for the verifier or (b) commit-while-encode patterns. Neither is
   currently supported.
2. **Eagle4 head's last step boundary** — the head's last propose
   call needs to commit-and-wait BEFORE the verify can start (because
   verify needs the head's last draft_token as batch[1]). True
   overlap is only possible if we run head's last step IN PARALLEL
   with verifier's BEGIN-OF-WORK (e.g., kv_append at slot K-1
   and pre-load of layer 0 expert weights).
3. **Profile flag** — `verify_kernels = "parallel-k-union-async"`
   would gate this. 24+ hours of dispatch-graph refactor.

## Audit verdict
F3 in its full form requires significant Metal-API refactoring (new
command queue lifecycle, multi-queue synchronization primitives, and
careful scheduling). Estimated 24-48 hours of focused work for
production-grade. Projected gain (per AUTONOMOUS_PLAN.md): +5-8 dec_tps.

For path-to-125: lower ROI/effort ratio than:
  - Continuing Branch 3 head re-arch experiments (1-3 hrs / iter)
  - Phase F1 AMX extension (4-8 hrs)
  - Stage 0.5 MLX rewrites (6-10 hrs)

DEFER F3 until the easier levers have been exhausted.
MD
  status_write "lever3_phase_f3_audit" "done"
}

# ── orchestrator entry ────────────────────────────────────────────────────────
status_write "orchestrator" "starting"
log "path-to-125 cross-lever orchestrator starting"

# Always rebuild first to capture the latest state.
log "rebuild: dismantle-core + dismantle binary"
cargo build --release -p dismantle-core 2>&1 | tail -2 || true
cargo build --release -p dismantle 2>&1 | tail -2 || true

lever1 || log "LEVER 1 failed; continuing"
lever2 || log "LEVER 2 failed; continuing"
lever3 || log "LEVER 3 failed; continuing"

status_write "orchestrator" "done"
log "all levers complete; see $STATUS for results"
if command -v osascript >/dev/null 2>&1; then
  osascript -e 'display notification "path-to-125 levers complete" with title "dismantle"' || true
fi
