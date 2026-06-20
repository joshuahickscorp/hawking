#!/usr/bin/env bash
# =============================================================================
# clean_room_queue.sh — section-close (2026-06-01) DEFERRED absolute-metric
#   bench queue. The benches whose VERDICT was deferred out of the autonomous
#   section-close run because they print ABSOLUTE numbers (dec_tps / TTFT_ms),
#   which an in-session agent inflates ~4-5x (reports/bench_contamination.md).
# =============================================================================
#
#   ⚠⚠⚠  RUN THIS ONLY WITH CLAUDE CODE FULLY QUIT.  ⚠⚠⚠
#
#   This is the SAME clean-room discipline as tools/bench/clean_room_batch.sh
#   (Q3 byte-cut + anchor + energy). This queue is the *complement*: the two
#   absolute benches this section-close run BUILT-and-parity-gated but did NOT
#   measure, because only a clean machine yields a trustworthy tps/TTFT number.
#
# HOW TO RUN (the user does this — NOT an agent):
#   1. Quit the Claude Code desktop app   (Cmd+Q).
#   2. Quit any `claude` CLI / loop sessions.
#   3. Quit slm and any other GPU/RAM-heavy process.
#   4. Open a fresh Terminal.app window.
#   5. cd /Users/scammermike/Downloads/dismantle
#   6. ./tools/bench/clean_room_queue.sh
#   7. Read the per-section verdicts; the JSON artifacts land under reports/bench/.
#
#   Pass --gates-only to print the plan + contamination preflight WITHOUT
#   running any bench (safe to run anytime, even with Claude open).
#
# WHAT IT RUNS:
#   (1) user_draft_3arm_bench.sh — the 3-arm propose-first vs bonus-first vs
#       plain user-ngram draft tps verdict (P1-C). Paired+interleaved so thermal
#       drift cancels, but the per-arm dec_tps is absolute → clean-room only.
#       Decision rule lives in the script header (diagnosis §6).
#   (2) prefill_mma_ttft_bench.sh — the Q4_K simdgroup-MMA prefill-TTFT verdict
#       (P1-A, silicon #8), MMA-on vs MMA-off, BOTH predec-ON (the shipped
#       config). GUARDED: only runs if the built binary recognizes
#       HAWKING_QWEN_Q4K_MMA (i.e. P1-A landed); otherwise prints SKIP.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

BIN="${BIN:-./target/release/hawking}"
GATES_ONLY=0
[[ "${1:-}" == "--gates-only" ]] && GATES_ONLY=1

hr() { printf '%s\n' "----------------------------------------------------------------------"; }

hr
echo "  clean_room_queue.sh — section-close deferred absolute benches (2026-06-01)"
hr

# ---- contamination preflight ------------------------------------------------
if pgrep -if "Claude" >/dev/null 2>&1 || pgrep -if "[c]laude" >/dev/null 2>&1; then
  echo "  ⚠ A Claude process appears to be running. Absolute tps/TTFT will be"
  echo "    contaminated ~4-5x. Quit Claude Code + any claude CLI, then re-run."
  [[ "$GATES_ONLY" -eq 0 ]] && { echo "  Refusing to run (pass --gates-only to override the plan print)."; exit 2; }
else
  echo "  preflight: no Claude process detected — clean to measure."
fi

if [[ ! -x "$BIN" ]]; then
  echo "  build the release binary first:  cargo build --release -p dismantle" >&2
  echo "  (then re-run this queue)." >&2
  [[ "$GATES_ONLY" -eq 0 ]] && exit 1
fi
echo ""

# ===========================================================================
hr
echo "  SECTION 1 — user-draft 3-arm tps (P1-C; tools/bench/user_draft_3arm_bench.sh)"
echo "  GO/NO-GO rule: see that script's header (§6). Paired+interleaved arms;"
echo "  per-arm absolute dec_tps + forward-count is the signal."
hr
if [[ "$GATES_ONLY" -eq 1 ]]; then
  echo "  [gates-only] would run: tools/bench/user_draft_3arm_bench.sh"
else
  tools/bench/user_draft_3arm_bench.sh || echo "  >> SECTION 1: bench exited non-zero — inspect output above."
  echo "  >> SECTION 1 VERDICT: read the C>B (low-accept) / C~=B (high-accept) lines; JSON at reports/bench/user_draft_3arm.json"
fi
echo ""

# ===========================================================================
hr
echo "  SECTION 2 — Q4_K MMA prefill TTFT (P1-A; tools/bench/prefill_mma_ttft_bench.sh)"
echo "  GO/NO-GO rule: MMA-on prefill_ms < MMA-off on the LONG prompt, predec-ON."
hr
# NOTE: dismantle reads HAWKING_QWEN_Q4K_MMA at runtime (not validated by
# clap), so there is no cheap "is it landed" probe from the shell. Instead we
# rely on prefill_mma_ttft_bench.sh's own SWAP-FIRED preflight, which reports
# whether the MMA twin actually engaged the predec branch (a "+0% flat" result
# with no SWAP-FIRED means P1-A did not land / did not wire in).
if [[ ! -x tools/bench/prefill_mma_ttft_bench.sh ]]; then
  echo "  >> SECTION 2 VERDICT: SKIPPED (prefill_mma_ttft_bench.sh missing)."
elif [[ "$GATES_ONLY" -eq 1 ]]; then
  echo "  [gates-only] would run: tools/bench/prefill_mma_ttft_bench.sh"
else
  tools/bench/prefill_mma_ttft_bench.sh || echo "  >> SECTION 2: bench exited non-zero — inspect output above (check the SWAP-FIRED preflight)."
  echo "  >> SECTION 2 VERDICT: read the paired prefill_ms delta (MMA-on vs MMA-off, predec-ON)."
fi
echo ""

hr
echo "  clean_room_queue DONE. Feed the two verdicts back:"
echo "    1 = user-draft propose-first KEEP/default-off (tps win at low-accept?)"
echo "    2 = MMA prefill GO/NO-GO (prefill_ms cut on long prompt, predec-ON?)"
echo "  (For Q3 byte-cut + anchor + energy, run the sibling tools/bench/clean_room_batch.sh.)"
hr
