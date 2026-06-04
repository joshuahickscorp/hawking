#!/usr/bin/env bash
# =============================================================================
# tools/bench/clean_bench_queue.sh — THE single command to run in a clean window.
#
# Clean-room downtime (Claude.app + Colab both quit) is scarce. This script
# batches EVERY pending ABSOLUTE / clean-room-gated measurement into one pass so
# the window is maximally productive, and tees a consolidated report.
#
# It is the deferred-absolute companion to final_analysis.sh (which is the
# contamination-robust day-to-day runner that works with Claude OPEN). Anything
# here needs the clean room because it reports an ABSOLUTE number (tps, J/tok,
# per-kernel GPU-us) where the ~4-5x session inflation does NOT cancel.
#
# USAGE (quit Claude.app AND any Colab/GPU job first):
#   tools/bench/clean_bench_queue.sh                 # run all sections
#   tools/bench/clean_bench_queue.sh --only anchor,energy
#   tools/bench/clean_bench_queue.sh --skip trace    # everything except the MST diff
#   tools/bench/clean_bench_queue.sh --list          # show sections + exit
#
# SECTIONS (default = all, in this order):
#   anchor   clean_room_batch.sh — absolute decode tps + J/tok + Q3 §A byte-cut
#            (the ~30.5 tps / ~0.197 J/tok no-regression anchor).
#   energy   phase_joules.sh --domains --tokens 512 — MEASURED per-domain GPU vs
#            DRAM J/tok (the energy-moat number nobody publishes), baseline AND
#            f16-KV, so the "does f16-KV win on ENERGY?" question is settled.
#   trace    mst_diff.sh — Metal System Trace per-kernel GPU-us/call diff of
#            dismantle vs llama-cli (the SOLE remaining single-stream-tps decider:
#            does llama's mul_mv sustain higher GiB/s per call?).
#   batch    continuous-batching AGGREGATE tps (B=1 vs B=8) — runs only once the
#            multi-seq decode path is built+parity-green; otherwise prints PENDING.
#
# Everything tee's to reports/bench/clean_bench_queue_<ts>.log.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

ALL_SECTIONS="anchor energy trace batch"
ONLY=""; SKIP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)  ONLY="$2"; shift 2 ;;
    --skip)  SKIP="$2"; shift 2 ;;
    --list)  printf 'sections: %s\n' "$ALL_SECTIONS"; exit 0 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) printf 'unknown arg: %s\n' "$1" >&2; exit 64 ;;
  esac
done

want() {  # want <section> -> 0 if it should run
  local s="$1"
  if [[ -n "$ONLY" ]]; then [[ ",$ONLY," == *",$s,"* ]]; return; fi
  if [[ -n "$SKIP" ]]; then [[ ",$SKIP," != *",$s,"* ]]; return; fi
  return 0
}

TS="$(date +%Y%m%dT%H%M%S)"
LOG="reports/bench/clean_bench_queue_${TS}.log"
mkdir -p reports/bench
# Tee all stdout+stderr to the log from here on.
exec > >(tee "$LOG") 2>&1

banner() { printf '\n========== %s ==========\n' "$*"; }
declare -a RESULTS

# --- Preflight: HARD-abort if the room is not clean ---------------------------
banner "PREFLIGHT — clean-room guard"
abort=0
# Match clean_room_batch.sh's canonical gate: the desktop app AND the CLI/agent.
if pgrep -f "Claude.app" >/dev/null 2>&1 || pgrep -xi "claude" >/dev/null 2>&1 \
   || pgrep -f "MASTER_LOOP" >/dev/null 2>&1; then
  echo "FAIL: a Claude session (app or CLI) is running — absolute numbers inflate ~4-5x. Quit it and re-run."
  abort=1
fi
# Any other heavy GPU/CPU hog (Colab via a browser tab won't show here, but a
# local python training job will). Warn loudly on >50% CPU non-dismantle procs.
hog="$(ps -Ao %cpu,comm -r | awk 'NR>1 && $1>50 && $2 !~ /dismantle|clean_bench|tee|ps|awk/ {print; exit}')"
if [[ -n "$hog" ]]; then
  echo "WARN: a heavy process is running (>50% CPU): $hog"
  echo "      Quit local Colab/training jobs for a trustworthy absolute number."
fi
if [[ "$abort" == 1 ]]; then
  echo "Aborting — the room is not clean."
  exit 3
fi
echo "OK: Claude.app not detected. Proceeding."
echo "log: $LOG"

# Self-heal a stale kernel profile: a shader change invalidates the committed
# shader_hash, so every generate-based section (anchor/energy) hard-refuses with
# a "shader hash mismatch". autotune is DETERMINISTIC + ~instant, so always
# regenerate to match the current shader build before the sections run.
QBIN="${BIN:-./target/release/dismantle}"
QWEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
QPROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
if [[ -x "$QBIN" && -f "$QWEIGHTS" ]]; then
  if "$QBIN" autotune --weights "$QWEIGHTS" --out "$QPROFILE" >/dev/null 2>&1; then
    echo "OK: kernel profile refreshed for the current shaders ($QPROFILE)"
  else
    echo "WARN: autotune failed — generate-based sections may hit a stale shader-hash error"
  fi
fi

# --- Section: anchor ----------------------------------------------------------
if want anchor; then
  banner "1  ANCHOR — absolute tps + J/tok + Q3 §A (clean_room_batch.sh)"
  if tools/bench/clean_room_batch.sh; then RESULTS+=("anchor: OK"); else RESULTS+=("anchor: non-zero (read above)"); fi
fi

# --- Section: energy (per-domain, baseline + f16-KV) --------------------------
if want energy; then
  banner "2  ENERGY — per-domain GPU/DRAM J/tok (phase_joules --domains)"
  echo "--- 2a baseline ---"
  tools/bench/phase_joules.sh --domains --tokens 512 \
    && RESULTS+=("energy/baseline: OK") || RESULTS+=("energy/baseline: non-zero")
  echo "--- 2b f16-KV (does halving KV bytes win on ENERGY at depth?) ---"
  DISMANTLE_QWEN_F16_KV=1 tools/bench/phase_joules.sh --domains --tokens 1024 \
    && RESULTS+=("energy/f16kv: OK") || RESULTS+=("energy/f16kv: non-zero")
  echo "COMPARE: f16-KV total J/tok < baseline => real energy lever; >= => footprint-only."
fi

# --- Section: trace (single-stream frontier decider) --------------------------
if want trace; then
  banner "3  FRONTIER — Metal System Trace diff vs llama.cpp (mst_diff.sh)"
  if [[ -x tools/bench/mst_diff.sh ]]; then
    tools/bench/mst_diff.sh && RESULTS+=("trace: OK (see reports/mst_diff_*.md)") \
      || RESULTS+=("trace: non-zero (read above)")
  else
    echo "SKIP: tools/bench/mst_diff.sh not executable."
    RESULTS+=("trace: SKIPPED (script missing)")
  fi
fi

# --- Section: batch (continuous-batching aggregate tps) -----------------------
if want batch; then
  banner "4  AGGREGATE — continuous-batching tps (B=1 vs B=8)"
  # Filled in once the multi-seq decode path lands. The build adds the bench
  # invocation here (placeholder guard so the queue runs cleanly until then).
  if [[ -f tools/bench/batch_aggregate_bench.sh ]]; then
    bash tools/bench/batch_aggregate_bench.sh \
      && RESULTS+=("batch: OK") || RESULTS+=("batch: non-zero")
  else
    echo "PENDING: continuous-batching multi-seq decode not built yet."
    echo "  (batch_ceiling.py predicts ~3.5-5.6x aggregate @ B=8; this section measures the real number once the kernel + per-slot KV land.)"
    RESULTS+=("batch: PENDING (build the multi-seq decode path)")
  fi
fi

# --- Summary ------------------------------------------------------------------
banner "SUMMARY"
for r in "${RESULTS[@]}"; do printf '  - %s\n' "$r"; done

# Catch the fake-pass class: a sub-script can exit 0 yet have logged a
# no-measurement signature (the first clean run reported sections "OK" while
# logging 0.0000 J/tok + a shader-hash mismatch). Scan the whole log and flag
# loudly so a masked failure is never silent — review these before trusting any 'OK'.
sigfile="$(mktemp)"
grep -nE 'kernel profile shader hash mismatch|no \[stats\] line|J/token *: *0\.0000|dec_tps *: *\?|Recording failed|Path not found|FAIL:' "$LOG" > "$sigfile" 2>/dev/null || true
if [[ -s "$sigfile" ]]; then
  echo ""
  echo "  ##  FAILURE SIGNATURES DETECTED IN LOG — a section above may be a FALSE 'OK':"
  sed -n '1,20p' "$sigfile" | sed 's/^/      /'
  echo "      (review these lines before trusting any 'OK' result)"
fi
rm -f "$sigfile"

echo ""
echo "full log: $LOG"
echo "done."
