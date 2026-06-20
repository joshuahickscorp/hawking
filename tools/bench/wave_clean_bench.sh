#!/usr/bin/env bash
# =============================================================================
# wave_clean_bench.sh — clean-bench ONLY the things not yet cleanly measured.
#
#   Quit the Claude desktop app first (Cmd+Q) for clean ABSOLUTE numbers.
#
# Runs (≈15-18 min):
#   [A] PROFILE LADDER : single-stream dec_tps under default/exact/fast/race/efficient
#                        (race/efficient are newly-distinct; default/exact reflect the
#                         E3 default-flip — all previously unmeasured on a clean machine)
#   [B] ENERGY ANCHOR  : measure_joules → clean J/tok (E3-default shifts it via race-to-idle)
#   [C] #6 LONG-CTX     : f32-KV vs F16_KV+FLASH_F16KV (the harness that printed
#                         decode_tps=? is now FIXED — this is its first clean run)
#
# REMOVED as already-solid (don't re-run; see reports/dead_levers.md + execution log):
#   * correctness/parity suite  → `cargo test` (contention-tolerant, all green, not a tps number)
#   * E3 A/B                     → clean-confirmed +9.6% / +9.8% robust; now the default
#   * #0 aggregate v4r-highB     → clean NO-GO (62.78→53.56); base aggregate 47.96 already clean
#   * final_analysis --clean     → re-confirms solid (Q3 dead, fast +18%); anchor covered by [A]+[B]
#   (re-run any of these by calling their own script: paired_lever.sh / batch_aggregate_bench.sh /
#    final_analysis.sh — they are intentionally not in this fast path.)
#
# Tunables: CTXS="4096 16384" (add 32768 for full) · LADDER_TOKENS=128 · SKIP_LONGCTX=1
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

export RUSTFLAGS="${RUSTFLAGS:-} -Awarnings"   # silence harmless objc cfg warnings
export CARGO_TERM_QUIET=true

WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
PROMPT="${PROMPT:-fn fibonacci(n: u64) -> u64 {}"
LADDER_TOKENS="${LADDER_TOKENS:-128}"
JOULE_TOKENS="${JOULE_TOKENS:-256}"
CTXS="${CTXS:-4096 16384}"
# Long-ctx lane is OPT-IN (default SKIP): it's slow (minutes per ctx even on an
# idle GPU) and only gates the long-context KV-footprint levers (f16-KV / flash-
# f16kv / staged int4-KV). Run it with SKIP_LONGCTX=0 when validating those.
SKIP_LONGCTX="${SKIP_LONGCTX:-1}"
STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="reports/bench/wave_clean_${STAMP}.log"
mkdir -p "$(dirname "$LOG")"

run_step() {  # run_step "label" cmd...
  echo; echo "######## $1 ########"; echo "+ ${*:2}"; echo
  "${@:2}" || echo "!!!! STEP FAILED (continuing): $1 !!!!"
}

# Single-stream dec_tps under a named runtime profile. generate prints the
# [stats] line (with dec_tps) to STDERR only when --json is NOT passed.
prof_tps() {  # $1 = profile name
  local prof="$1" serr dec disp con
  serr=$(./target/release/hawking --profile "$prof" generate \
    --weights "$WEIGHTS" --kernel-profile "$PROFILE" --prompt "$PROMPT" \
    --max-new-tokens "$LADDER_TOKENS" --temperature 0 --seed 0 2>&1 >/dev/null) || true
  dec=$(printf '%s\n' "$serr" | sed -n 's/.*dec_tps=\([0-9.][0-9.]*\).*/\1/p' | head -1)
  disp=$(printf '%s\n' "$serr" | sed -n 's/.*dispatches_per_fwd=\([0-9][0-9]*\).*/\1/p' | head -1)
  con=$(printf '%s\n' "$serr" | sed -n 's/.*\[dismantle\] \(profile=[^.]*\).*/\1/p' | head -1)
  printf "  LADDER %-10s dec_tps=%-8s dispatches=%-5s | %s\n" \
    "$prof" "${dec:-?}" "${disp:-?}" "${con:-(no contract)}"
}

{
  echo "################################################################"
  echo "# WAVE CLEAN BENCH (slim: only un-measured lanes)   ${STAMP}"
  echo "# git: $(git rev-parse --short HEAD 2>/dev/null)  +$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') local edits"
  echo "# binary shader-hash:  $(./target/release/hawking shader-hash 2>/dev/null)"
  echo "# profile shader-hash: $(grep '\"shader_hash\"' "$PROFILE" 2>/dev/null | grep -oE '[a-f0-9]{24}')"
  echo "# CTXS=${CTXS}  LADDER_TOKENS=${LADDER_TOKENS}  SKIP_LONGCTX=${SKIP_LONGCTX}"
  echo "################################################################"

  run_step "BUILD (release bin, ensure current vs restamped profiles)" \
    cargo build --release --bin dismantle -q

  # ── [A] PROFILE LADDER ──────────────────────────────────────────────────────
  echo; echo "######## [A] PROFILE LADDER  (single-stream, ${LADDER_TOKENS}-tok greedy) ########"
  echo "  default == exact (both E3-default bit-identical); fast/race/efficient = quality-trade"
  for prof in default exact fast race efficient; do prof_tps "$prof"; done

  # ── [B] ENERGY ANCHOR (clean J/tok) ────────────────────────────────────────
  run_step "[B] ENERGY  measure_joules (--tokens ${JOULE_TOKENS})" \
    tools/bench/measure_joules.sh --tokens "$JOULE_TOKENS"

  # ── [C] #6 LONG-CONTEXT (the previously-broken, now-fixed harness) ──────────
  if [ "$SKIP_LONGCTX" != "1" ]; then
    run_step "[C] #6 long-ctx  f32-KV baseline (CTXS=$CTXS)" \
      env CTXS="$CTXS" tools/bench/long_context_bench.sh
    echo; echo "---- [C] #6 long-ctx  F16_KV + FLASH_F16KV (CTXS=$CTXS) ----"
    env CTXS="$CTXS" HAWKING_QWEN_F16_KV=1 HAWKING_QWEN_FLASH_F16KV=1 \
      tools/bench/long_context_bench.sh \
      || echo "!!!! STEP FAILED (continuing): #6 ON !!!!"
  else
    echo; echo "######## [C] #6 long-ctx SKIPPED ########"
  fi

  echo; echo "################################################################"
  echo "# DONE ${STAMP}"
  echo "################################################################"
} 2>&1 | tee "$LOG"

# ---- KEY RESULTS (also appended to the log) --------------------------------
{
  echo
  echo "================= KEY RESULTS ================="
  echo "--- [A] profile ladder (dec_tps per profile) ---"
  grep -E "^  LADDER " "$LOG" || echo "  (none captured — check the ladder phase above)"
  echo "--- [B] energy (clean J/tok) ---"
  grep -iE "J/token|dec_tps +:|avg .*power" "$LOG" || echo "  (none captured)"
  echo "--- [C] long-context (per-ctx decode_tps + KV share) ---"
  grep -iE "^ +[0-9]+ +[0-9].*%|decode_tps" "$LOG" || echo "  (none captured)"
  echo "==============================================="
  echo "FULL LOG: $LOG"
  echo "Paste the KEY RESULTS block (or the full log) back."
} | tee -a "$LOG"
