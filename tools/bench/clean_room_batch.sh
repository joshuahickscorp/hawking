#!/usr/bin/env bash
# =============================================================================
# clean_room_batch.sh — CLEAN-ROOM-ONLY absolute-metric batch
# =============================================================================
#
#   ⚠⚠⚠  RUN THIS ONLY WITH CLAUDE CODE FULLY QUIT.  ⚠⚠⚠
#
# WHY CLEAN-ROOM-ONLY (read this before running):
#   Every number this script prints is an ABSOLUTE metric (GB/s, dec_tps,
#   joules/token) — NOT a paired relative delta. A running Claude Code / agent
#   session inflates throughput by ~4–5× (the bench-contamination finding:
#   reports/bench_contamination.md, memory/bench_contamination.md), and even a
#   mild active session shifts dec_tps (~37 vs ~31 clean, bible §3.0
#   Correction 2). Paired-delta benches cancel this; ABSOLUTE benches DO NOT.
#   So this batch is meaningless unless the machine is clean.
#
#   This is the same discipline as tools/bench/clean_bench.sh and
#   tools/bench/measure_joules.sh — it is just the *Q3-byte-cut + anchor +
#   energy* batch the cheaper-decode-Q3 / QTIP designs need, in one turnkey run.
#
# HOW TO RUN (the user does this — NOT an agent):
#   1. Quit the Claude Code desktop app   (Cmd+Q in the menu bar).
#   2. Quit any `claude` CLI sessions     (incl. any MASTER_LOOP / loop).
#   3. Quit slm and any other GPU/RAM-heavy process.
#   4. Open a fresh Terminal.app window.
#   5. cd /Users/scammermike/Downloads/hawking
#   6. ./tools/bench/clean_room_batch.sh
#   7. Read the three section verdicts printed at the end.
#
#   Pass --gates-only to run the pre-flight contamination checks and print the
#   plan WITHOUT running any bench (safe to run anytime, even with Claude open).
#
# WHAT IT RUNS (three sections, each with an echoed header + GO/NO-GO rule):
#   (A) Q3 byte-cut microbench  — the existing q3k_bytecut_bench; prints the
#       f32-predec-Q3 GB/s and applies the cheaper-decode-Q3 §6.0 decision rule
#       (~50% peak = Q3 byte-cut GO → build f16-predec-Q3 128-B repack;
#        ~30% peak = NO-GO → hmask residual is the wall → QTIP is the only path).
#   (B) Decode-tps anchor reconciliation — a clean standalone `generate` decode
#       run, printed against BOTH open anchors (~39 older-clean vs ~31
#       most-recent-clean; bible §3.0 OPEN item).
#   (C) Energy baseline — joules/token via measure_joules.sh (wraps macmon).
#
# This script does NOT change any perf behavior; it only measures. It is safe to
# re-run. It uses the locked Qwen fast-path env (the shipped decode config).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

# ---- config (override via env) ---------------------------------------------
BIN="${BIN:-./target/release/hawking}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
TOKENS="${TOKENS:-256}"          # decode length for sections B + C
PROMPT_DEFAULT='fn fibonacci(n: u64) -> u64 {'
PROMPT="${PROMPT:-$PROMPT_DEFAULT}"
PEAK_GBPS="${PEAK_GBPS:-150}"    # M3 Pro memory bandwidth peak (bible §0)
ANCHOR_HIGH="${ANCHOR_HIGH:-39}" # older clean anchor (bible §3 envelope)
ANCHOR_LOW="${ANCHOR_LOW:-31}"   # most-recent clean anchor (A1/A4, bible §3.0)

GATES_ONLY=0
[[ "${1:-}" == "--gates-only" ]] && GATES_ONLY=1

# Locked Qwen fast-path (matches quick_bench.sh / measure_joules.sh / path_to_50).
export HAWKING_QWEN_TCB=1 \
       HAWKING_QWEN_VOCAB_PRUNE=32000 \
       HAWKING_QWEN_Q4K_LMHEAD=1 \
       HAWKING_QWEN_FFN_DOWN_Q4K=1 \
       HAWKING_QWEN_Q4K_PREDEC=1

hr()  { printf '%s\n' "================================================================================"; }
die() { echo "error: $*" >&2; exit 64; }

# ---------------------------------------------------------------------------
# PRE-FLIGHT — contamination gates. These are the whole point of "clean-room".
# ---------------------------------------------------------------------------
hr
echo "  clean_room_batch — CLEAN-ROOM-ONLY absolute-metric batch"
echo "  (Claude MUST be quit; absolute tps/GB/s/J inflate ~4-5x under an active session)"
hr

PREFLIGHT_FAIL=0

# Gate 1: Claude desktop app must be quit.
if pgrep -f "Claude.app" >/dev/null 2>&1; then
  echo "  [GATE claude-app]  FAIL — Claude.app is running. Cmd+Q it, then re-run." >&2
  PREFLIGHT_FAIL=1
else
  echo "  [GATE claude-app]  pass (Claude.app not running)"
fi

# Gate 2: no `claude` CLI / loop sessions.
if pgrep -x "claude" >/dev/null 2>&1 || pgrep -f "MASTER_LOOP" >/dev/null 2>&1; then
  echo "  [GATE claude-cli]  FAIL — a 'claude' CLI / MASTER_LOOP session is running. Quit it." >&2
  PREFLIGHT_FAIL=1
else
  echo "  [GATE claude-cli]  pass (no claude CLI / loop session)"
fi

# Gate 3: no slm (co-existence partner — its load contaminates absolute numbers).
# Exact process-name match: `pgrep -i slm` returns PIDs only, so the old
# `| grep -v aslmanager` never filtered (PIDs aren't names) and the macOS daemon
# `aslmanager` (Apple System Log manager) false-FAILed the gate. `-x` matches the
# executable name exactly, so it catches a process literally named `slm` and
# never `aslmanager`/`asl*`/anything-containing-slm.
SLM_PIDS="$(pgrep -xi slm 2>/dev/null || true)"
if [[ -n "$SLM_PIDS" ]]; then
  echo "  [GATE slm]         FAIL — slm is running (pids: $SLM_PIDS). Exit slm, then re-run." >&2
  PREFLIGHT_FAIL=1
else
  echo "  [GATE slm]         pass (slm not running)"
fi

# Gate 4: top non-bench CPU consumer should be quiet (Spotlight/indexers skew tps).
TOP_LINE="$(ps -axo %cpu,comm | sort -nr | awk '$2 !~ /dismantle|bench|clean_room/ {print; exit}')"
TOP_CPU="$(printf '%s\n' "$TOP_LINE" | awk '{print $1}')"
TOP_NAME="$(printf '%s\n' "$TOP_LINE" | awk '{print $2}')"
if awk "BEGIN { exit (\"${TOP_CPU:-0}\" + 0 < 30.0) ? 0 : 1 }"; then
  echo "  [GATE quiet-cpu]   pass (top non-bench: ${TOP_CPU:-0}% ${TOP_NAME:-?})"
else
  echo "  [GATE quiet-cpu]   WARN — ${TOP_NAME:-?} at ${TOP_CPU}% CPU; let it settle for clean absolutes." >&2
fi

# Build artifacts present?
[[ -x "$BIN" ]]     || { echo "  [GATE binary]      FAIL — $BIN missing (cargo build --release --workspace?)" >&2; PREFLIGHT_FAIL=1; }
[[ -f "$WEIGHTS" ]] || { echo "  [GATE weights]     FAIL — $WEIGHTS missing" >&2; PREFLIGHT_FAIL=1; }

echo ""
echo "  plan:"
echo "    (A) Q3 byte-cut microbench   -> f32-predec-Q3 GB/s vs ${PEAK_GBPS} GB/s peak  [GO ~50% / NO-GO ~30%]"
echo "    (B) decode-tps anchor recon  -> clean dec_tps vs anchors ~${ANCHOR_HIGH} (old) / ~${ANCHOR_LOW} (recent)"
echo "    (C) energy baseline          -> joules/token via measure_joules.sh (macmon)"
echo ""

if [[ "$GATES_ONLY" == 1 ]]; then
  if [[ "$PREFLIGHT_FAIL" == 1 ]]; then
    echo "  --gates-only: printed the pre-flight failures and plan. Not running benches."
    echo "  Re-run without the flag only after the FAIL gates pass."
  else
    echo "  --gates-only: pre-flight passed. Not running benches. Re-run without the flag (Claude quit)."
  fi
  exit 0
fi
if [[ "$PREFLIGHT_FAIL" == 1 ]]; then
  echo "  PRE-FLIGHT FAILED — fix the FAIL gates above and re-run. (Absolutes would be garbage.)" >&2
  exit 1
fi
echo "  PRE-FLIGHT PASSED — running the clean-room batch."
echo ""

# ===========================================================================
# SECTION A — Q3 byte-cut microbench (cheaper-decode-Q3 §6.0 free pre-build read)
# ===========================================================================
hr
echo "  SECTION A — Q3 byte-cut microbench (q3k_bytecut_bench)"
echo "  Reads the f32-predec-Q3 GB/s = 'Q3 decode with the 6-bit scale decode hoisted out'."
echo "  DECISION RULE (cheaper_decode_q3_design §6.0 / §8):"
echo "    f32-predec-Q3 >= ~50% of peak (~$(awk -v p="$PEAK_GBPS" 'BEGIN{printf "%.0f", p*0.50}')-$(awk -v p="$PEAK_GBPS" 'BEGIN{printf "%.0f", p*0.55}') GB/s) -> GO:"
echo "        bottleneck flipped to BW; BUILD the f16-predec-Q3 128-B repack."
echo "    f32-predec-Q3 ~= 30% of peak (~$(awk -v p="$PEAK_GBPS" 'BEGIN{printf "%.0f", p*0.30}') GB/s) -> NO-GO:"
echo "        hmask/index residual is the wall; Q3 byte-cut stays footprint-only"
echo "        -> QTIP (gather-free trellis) is the ONLY remaining byte-cut path."
hr

A_LOG="$(mktemp -t q3kbytecut.XXXXXX)"
echo "  running: cargo test -p hawking-core --release --test q3k_bytecut_bench -- --ignored --nocapture"
echo "  (the bench test is #[ignore]-marked, so --ignored is required to run it)"
echo "  ... this builds hawking-core in release if needed, then runs 200 iters/shape x 3 shapes."
echo ""
# The bench prints to stderr (eprintln!); capture both streams.
cargo test -p hawking-core --release --test q3k_bytecut_bench -- --ignored --nocapture \
  >"$A_LOG" 2>&1
A_RC=$?

# Echo the bench's own per-shape lines (GB/s + verdicts a/b/c).
echo "----- q3k_bytecut_bench output -----"
grep -E 'shape |GB/s:|>>> |bytes/block' "$A_LOG" || cat "$A_LOG"
echo "------------------------------------"

if [[ $A_RC -ne 0 ]]; then
  echo "  SECTION A: bench exited non-zero ($A_RC). Tail of log:" >&2
  tail -20 "$A_LOG" >&2
  echo "  >> SECTION A VERDICT: INCONCLUSIVE (bench failed to run cleanly)."
else
  # Pull the f32-predec-Q3 GB/s from each 'GB/s:' line. The bench prints a line:
  #   GB/s:  Q3 fused=..  Q3 fused_2r=..  Q3 predec=NN.N  Q4 predec=..
  # 'Q3 predec' IS the f32-predec kernel (it reads f32 scales today). Take the
  # max across the 3 shapes as the best-case Q3-predec read.
  Q3_PREDEC_MAX="$(grep -E 'GB/s:' "$A_LOG" \
    | grep -oE 'Q3 predec=[0-9.]+' | grep -oE '[0-9.]+' \
    | sort -nr | head -1)"
  if [[ -z "$Q3_PREDEC_MAX" ]]; then
    echo "  >> SECTION A VERDICT: could not parse 'Q3 predec' GB/s — read the raw lines above."
  else
    PCT="$(awk -v g="$Q3_PREDEC_MAX" -v p="$PEAK_GBPS" 'BEGIN{printf "%.1f", g/p*100}')"
    echo ""
    echo "  >> f32-predec-Q3 best-shape GB/s = ${Q3_PREDEC_MAX}  (= ${PCT}% of ${PEAK_GBPS} GB/s peak)"
    awk -v g="$Q3_PREDEC_MAX" -v p="$PEAK_GBPS" 'BEGIN{
      pct = g/p*100;
      if (pct >= 45.0)
        print "  >> SECTION A VERDICT: GO (>=~50% of peak) -> bottleneck flipped to BW.";
      else if (pct <= 35.0)
        print "  >> SECTION A VERDICT: NO-GO (~30% of peak) -> hmask residual is the wall -> QTIP is the only byte-cut path.";
      else
        print "  >> SECTION A VERDICT: AMBIGUOUS (~35-45% of peak) -> read verdict (b)/(c) lines; lean build-and-microbench f16-predec-Q3.";
    }'
    echo "     (GO  => build f16-predec-Q3 128-B repack per cheaper_decode_q3_design §5;"
    echo "      NO-GO => Q3 byte-cut footprint-only; pursue QTIP per qtip_bytecut_design §5-§6.)"
  fi
fi
rm -f "$A_LOG"
echo ""

# ===========================================================================
# SECTION B — decode-tps anchor reconciliation (bible §3.0 OPEN item)
# ===========================================================================
hr
echo "  SECTION B — clean decode-tps anchor reconciliation"
echo "  bible §3.0 Correction 2: the ~39 (older clean) vs ~31 (A1/A4 recent clean)"
echo "  anchors are UNRECONCILED. This clean standalone decode run pins which is right."
echo "  config: greedy temp=0 seed=0, ${TOKENS} tokens, locked fast-path, nice -n 19 taskpolicy -b."
hr

B_OUT="$(mktemp -t cleanroom_gen.XXXXXX)"
echo "  running: nice -n 19 taskpolicy -b $BIN generate --max-new-tokens $TOKENS --temperature 0 --seed 0"
echo ""
nice -n 19 taskpolicy -b "$BIN" generate \
  --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
  --prompt "$PROMPT" --max-new-tokens "$TOKENS" --temperature 0 --seed 0 \
  >"$B_OUT" 2>&1
B_RC=$?

STATLINE="$(grep -E '\[stats\]' "$B_OUT" | tail -1)"
DEC_TPS="$(printf '%s' "$STATLINE" | grep -oE 'dec_tps=[0-9.]+' | grep -oE '[0-9.]+')"
COMP_TOK="$(printf '%s' "$STATLINE" | grep -oE 'completion=[0-9]+' | grep -oE '[0-9]+')"

if [[ $B_RC -ne 0 || -z "$DEC_TPS" ]]; then
  echo "  SECTION B: generate failed or no [stats] line. Tail:" >&2
  tail -10 "$B_OUT" >&2
  echo "  >> SECTION B VERDICT: INCONCLUSIVE."
else
  echo "  stats: $STATLINE"
  echo ""
  echo "  >> clean dec_tps = ${DEC_TPS}  (completion=${COMP_TOK:-?} tokens)"
  awk -v t="$DEC_TPS" -v hi="$ANCHOR_HIGH" -v lo="$ANCHOR_LOW" 'BEGIN{
    dh = (t-hi); if (dh<0) dh=-dh;
    dl = (t-lo); if (dl<0) dl=-dl;
    printf "  >> vs ~%s (older-clean anchor): %+.1f tps (%+.1f%%)\n", hi, t-hi, (t/hi-1)*100;
    printf "  >> vs ~%s (recent-clean anchor): %+.1f tps (%+.1f%%)\n", lo, t-lo, (t/lo-1)*100;
    if (dl < dh)      print "  >> SECTION B VERDICT: clean tps tracks the ~31 recent anchor (the ~39 envelope is optimistic).";
    else if (dh < dl) print "  >> SECTION B VERDICT: clean tps tracks the ~39 older anchor (the ~31 recent reading was low).";
    else              print "  >> SECTION B VERDICT: clean tps sits midway between the two anchors.";
  }'
  echo "     (NOTE: a single clean run; for a thermal-protocol median, run clean_bench.sh / repeat N times.)"
fi
rm -f "$B_OUT"
echo ""

# ===========================================================================
# SECTION C — energy baseline (joules/token via measure_joules.sh / macmon)
# ===========================================================================
hr
echo "  SECTION C — energy baseline (joules/token, bible §8 L4.2)"
echo "  Delegates to tools/bench/measure_joules.sh (wraps macmon, sudo-free)."
echo "  J/tok = avg_power_W * decode_wall_s / tokens. (The 'sips power' branded axis.)"
hr

if [[ ! -x tools/bench/measure_joules.sh ]]; then
  echo "  SECTION C: tools/bench/measure_joules.sh not found/executable — skipping." >&2
  echo "  >> SECTION C VERDICT: SKIPPED (measure_joules.sh missing)."
elif ! command -v macmon >/dev/null 2>&1; then
  echo "  SECTION C: macmon not installed (the sudo-free power source)." >&2
  echo "    install: brew install macmon   then re-run this batch." >&2
  echo "  >> SECTION C VERDICT: SKIPPED (no sudo-free power source; run measure_joules.sh attended with sudo -v for powermetrics)."
else
  echo "  running: tools/bench/measure_joules.sh --tokens $TOKENS"
  echo ""
  TOKENS="$TOKENS" PROMPT="$PROMPT" tools/bench/measure_joules.sh --tokens "$TOKENS"
  echo ""
  echo "  >> SECTION C VERDICT: see the 'J/token' line above — that is the clean energy baseline."
fi
echo ""

# ===========================================================================
hr
echo "  clean_room_batch DONE."
echo "  Feed the three verdicts back:"
echo "    A = Q3 byte-cut GO/NO-GO (build f16-predec-Q3, or QTIP-only)"
echo "    B = which decode-tps anchor is real (~39 vs ~31)"
echo "    C = clean joules/token baseline"
echo "  Designs: plans/cheaper_decode_q3_design_2026_05_31.md , plans/qtip_bytecut_design_2026_05_31.md"
hr
