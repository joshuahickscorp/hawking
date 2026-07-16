#!/usr/bin/env bash
# tools/ci/regression_gate.sh — enforce that Hawking's speed / compression / quality
# wins do NOT regress silently. Correctness is already locked by 193 golden token
# hashes; this is the missing half: it compares MEASURED perf/footprint/quality against
# a committed baseline (tools/ci/regression_baseline.json) and EXITS NON-ZERO
# on a breach. It reuses tools/bench/ratios.sh for measurement (no duplicated logic).
#
# It is a CATEGORY-regression gate, not a micro-benchmark: floors sit ~10-15% below the
# measured warm median because the fresh-process warm-median has a +-several-% noise floor
# (test_matrix.md). It catches a lever silently disabled (predec OFF = -46.7%), a quant
# path regressing, or a quality collapse — without flapping red on noise.
#
# Checks:
#   1. FOOTPRINT  (always, CPU-safe, deterministic): on-disk bytes <= committed ceiling.
#   2. DECODE_TPS (GPU; needs the release binary + model + a free GPU): warm median >= floor.
#   3. QUALITY    (GPU): lever argmax-identity vs the bit-identical default >= floor.
#
# A check whose inputs are unavailable is reported SKIPPED (never a false PASS).
# Exit is non-zero only when an ENFORCED check FAILS.
#
# Env:
#   FOOTPRINT_ONLY=1   run only the CPU-safe footprint gate (no GPU)
#   RUN_GPU=0          skip tps + quality (footprint still runs)
#   GPU_WAIT=1         wait for a busy GPU instead of skipping GPU checks (default: skip)
#   TRIALS=3           warm-median trial count for tps
#   TOK=96             max-new-tokens for tps runs
#   QTOK=80            max-new-tokens for quality runs
#   BASELINE=<path>    override the baseline JSON
#   OUT=<dir>          report dir (default reports/regression/<stamp>)
set -u

REPO="${REPO:-$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)}"
cd "$REPO" || exit 2

BIN="${BIN:-./target/release/hawking}"
RATIOS="tools/bench/ratios.sh"
BASELINE="${BASELINE:-tools/ci/regression_baseline.json}"
TRIALS="${TRIALS:-3}"
export TOK="${TOK:-96}"
QTOK="${QTOK:-80}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-reports/regression/$STAMP}"
mkdir -p "$OUT"
REPORT="$OUT/summary.md"

FAIL=0
ENFORCED=0
ROWS=""   # markdown table rows accumulated then written

if ! command -v jq >/dev/null 2>&1; then
  echo "regression_gate: jq is required" >&2; exit 2
fi
if [ ! -f "$BASELINE" ]; then
  echo "regression_gate: baseline not found: $BASELINE" >&2; exit 2
fi

note() { printf '%s\n' "$*"; }
addrow() { ROWS="${ROWS}| $1 | $2 | $3 | $4 | $5 |
"; }   # check | measured | baseline | result | detail

# floating compare: $1 op $2  (op: ge / le)
fcmp() { awk -v a="$1" -v b="$3" -v op="$2" 'BEGIN{ if(op=="ge") exit !(a+0>=b+0); else exit !(a+0<=b+0) }'; }

gpu_busy() {
  ps ax -o pid= -o command= | awk -v self="$$" '
    $1 != self && /hawking (generate|serve)/ && !/regression_gate/ { f=1 } END { exit f?0:1 }'
}

note "== Hawking regression gate =="
note "baseline: $BASELINE"
note "out:      $OUT"
note ""

# ---------------------------------------------------------------- 1. FOOTPRINT
note "-- footprint ceilings (compression persistence) --"
while IFS= read -r key; do
  ceil="$(jq -r --arg k "$key" '.footprint_ceilings_bytes[$k].bytes' "$BASELINE")"
  tol="$(jq -r --arg k "$key" '.footprint_ceilings_bytes[$k].tol_pct // 0' "$BASELINE")"
  if [ ! -f "$key" ]; then
    note "SKIP  footprint $key (file absent)"
    addrow "footprint: $key" "—" "$ceil B" "SKIP" "file absent"
    continue
  fi
  bytes="$(stat -f%z "$key")"
  limit="$(awk -v c="$ceil" -v t="$tol" 'BEGIN{printf "%d", c*(1+t/100)}')"
  ENFORCED=$((ENFORCED+1))
  if fcmp "$bytes" le "$limit"; then
    note "PASS  footprint $key: $bytes B <= $limit B (ceil $ceil +${tol}%)"
    addrow "footprint: $(basename "$key")" "$bytes B" "<= $limit B" "PASS" "ceil $ceil +${tol}%"
  else
    note "FAIL  footprint $key: $bytes B > $limit B"
    addrow "footprint: $(basename "$key")" "$bytes B" "<= $limit B" "**FAIL**" "grew past ceiling — compression regressed or model swapped"
    FAIL=1
  fi
done < <(jq -r '.footprint_ceilings_bytes | keys[]' "$BASELINE")
note ""

# ---------------------------------------------------------------- GPU gating
DO_GPU=1
[ "${FOOTPRINT_ONLY:-0}" = "1" ] && DO_GPU=0
[ "${RUN_GPU:-1}" = "0" ] && DO_GPU=0
if [ "$DO_GPU" = "1" ] && [ ! -x "$BIN" ]; then
  note "SKIP  GPU checks: release binary $BIN absent (build with: cargo build --release -p hawking)"
  DO_GPU=0
  GPU_SKIP_REASON="release binary absent"
fi
if [ "$DO_GPU" = "1" ] && gpu_busy; then
  if [ "${GPU_WAIT:-0}" = "1" ]; then
    note "GPU busy; GPU_WAIT=1 — waiting..."
    while gpu_busy; do sleep 30; done
  else
    note "SKIP  GPU checks: another hawking generate/serve job is active (set GPU_WAIT=1 to wait)"
    DO_GPU=0
    GPU_SKIP_REASON="GPU busy"
  fi
fi

# ---------------------------------------------------------------- 2. DECODE_TPS
note "-- decode_tps floors (speed persistence) --"
if [ "$DO_GPU" = "1" ]; then
  while IFS= read -r key; do
    model="$(jq -r --arg k "$key" '.decode_tps_floors[$k].model' "$BASELINE")"
    ctx="$(jq -r --arg k "$key" '.decode_tps_floors[$k].ctx' "$BASELINE")"
    floor="$(jq -r --arg k "$key" '.decode_tps_floors[$k].floor' "$BASELINE")"
    if [ ! -f "$model" ]; then
      note "SKIP  tps $key ($model absent)"; addrow "tps: $key" "—" ">= $floor" "SKIP" "model absent"; continue
    fi
    raw="$(M="$model" "$RATIOS" tps "$key" "" "$ctx" "$TRIALS" 2>"$OUT/${key}_tps.err")"
    printf '%s\n' "$raw" >>"$OUT/tps.log"
    tps="$(printf '%s' "$raw" | grep -oE 'tps=[0-9.]+' | head -1 | cut -d= -f2)"
    if [ -z "$tps" ] || [ "$tps" = "ERR" ]; then
      note "SKIP  tps $key (no dec_tps parsed; see $OUT/${key}_tps.err)"; addrow "tps: $key" "ERR" ">= $floor" "SKIP" "measurement failed"; continue
    fi
    ENFORCED=$((ENFORCED+1))
    if fcmp "$tps" ge "$floor"; then
      note "PASS  tps $key: $tps >= $floor (warm median, n=$TRIALS)"
      addrow "tps: $key" "$tps" ">= $floor" "PASS" "warm median n=$TRIALS"
    else
      note "FAIL  tps $key: $tps < $floor"
      addrow "tps: $key" "$tps" ">= $floor" "**FAIL**" "decode regressed (category) — check predec/kernel/state"
      FAIL=1
    fi
  done < <(jq -r '.decode_tps_floors | keys[]' "$BASELINE")
else
  note "SKIPPED (GPU checks disabled: ${GPU_SKIP_REASON:-RUN_GPU=0/FOOTPRINT_ONLY})"
  addrow "tps: (all)" "—" "see baseline" "SKIP" "${GPU_SKIP_REASON:-GPU disabled}"
fi
note ""

# ---------------------------------------------------------------- 3. QUALITY
note "-- quality argmax-identity floors (lever-safety) --"
if [ "$DO_GPU" = "1" ]; then
  # key -> ratios.sh qual invocation (hardcoded map; floors come from the baseline)
  qual_run() { # $1=key -> echoes integer percent
    case "$1" in
      profile_fast) "$RATIOS" qual "" fast "$QTOK" 2>/dev/null ;;
      f16_kv)       "$RATIOS" qual "HAWKING_QWEN_F16_KV=1" "" "$QTOK" 2>/dev/null ;;
    esac
  }
  while IFS= read -r key; do
    [ "$key" = "_note" ] && continue
    floor="$(jq -r --arg k "$key" '.quality_identity_floors[$k].floor' "$BASELINE")"
    out="$(qual_run "$key")"; printf '%s\n' "$out" >>"$OUT/quality.log"
    pct="$(printf '%s' "$out" | grep -oE '[0-9]+%' | head -1 | tr -d '%')"
    if [ -z "$pct" ]; then
      note "SKIP  quality $key (no result parsed)"; addrow "quality: $key" "ERR" ">= $floor" "SKIP" "measurement failed"; continue
    fi
    frac="$(awk -v p="$pct" 'BEGIN{printf "%.2f", p/100}')"
    ENFORCED=$((ENFORCED+1))
    if fcmp "$frac" ge "$floor"; then
      note "PASS  quality $key: ${pct}% (>= $floor argmax-identity vs default)"
      addrow "quality: $key" "${pct}%" ">= $floor" "PASS" "argmax-identity vs bit-identical default"
    else
      note "FAIL  quality $key: ${pct}% < $floor"
      addrow "quality: $key" "${pct}%" ">= $floor" "**FAIL**" "lever fidelity regressed"
      FAIL=1
    fi
  done < <(jq -r '.quality_identity_floors | keys[]' "$BASELINE")
else
  note "SKIPPED (GPU checks disabled)"
  addrow "quality: (all)" "—" "see baseline" "SKIP" "${GPU_SKIP_REASON:-GPU disabled}"
fi
note ""

# ---------------------------------------------------------------- report
{
  echo "# Hawking Regression Gate — $STAMP"
  echo
  echo "- Baseline: \`$BASELINE\`"
  echo "- Binary: \`$BIN\` ($([ -x "$BIN" ] && echo present || echo absent))"
  echo "- GPU checks: $([ "$DO_GPU" = "1" ] && echo RAN || echo "SKIPPED (${GPU_SKIP_REASON:-disabled})")"
  echo "- Enforced checks run: $ENFORCED"
  echo
  echo "| Check | Measured | Baseline | Result | Detail |"
  echo "|---|---|---|---|---|"
  printf '%s' "$ROWS"
  echo
  if [ "$FAIL" = "0" ]; then
    echo "**Result: PASS** — $ENFORCED enforced check(s) within baseline."
  else
    echo "**Result: FAIL** — a measured value breached its committed baseline (see rows marked FAIL)."
  fi
  echo
  echo "## Not yet enforced (honest gaps — do not read as covered)"
  jq -r '.pending_not_enforced | to_entries[] | select(.key!="_note") | "- **\(.key):** \(.value)"' "$BASELINE"
} >"$REPORT"

note "report: $REPORT  (enforced=$ENFORCED fail=$FAIL)"
exit "$FAIL"
