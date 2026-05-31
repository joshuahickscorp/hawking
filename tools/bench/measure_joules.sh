#!/usr/bin/env bash
# measure_joules.sh — joules-per-token baseline for Qwen-3B decode (§8 L4.2).
#
# Reports the "runs cool / sips power" axis the throughput bible says nobody
# else flies: average SoC/GPU power (W), decode throughput (dec_tps), and the
# branded number — joules-per-token:
#
#     J/tok = avg_power_W * decode_wall_s / tokens_generated
#
# This is a MEASUREMENT tool (no perf change). It runs a steady-state Qwen-3B
# decode under the locked fast-path env while sampling package/GPU power, then
# divides energy by tokens.
#
# POWER SOURCE (auto-detected, in preference order):
#   1. macmon   — SUDO-FREE. Reads SoC/GPU power via the IOReport framework
#                 (no kext, no password). Install: `brew install macmon`.
#                 This is the preferred unattended path.
#   2. powermetrics — needs SUDO (a password). Use only attended:
#                 `sudo tools/bench/measure_joules.sh ...` or pre-auth with
#                 `sudo -v` first. An unattended agent CANNOT use this.
#
# If neither is usable the script prints exactly what to install/run and exits
# non-zero WITHOUT attempting sudo (safe for unattended hauls).
#
# Usage:
#   tools/bench/measure_joules.sh                       # baseline, 256 tok
#   tools/bench/measure_joules.sh --tokens 512
#   tools/bench/measure_joules.sh --f16s                # also run the A6.5 f16s lever and compare
#   DISMANTLE_QWEN_PREDEC_F16SCALES=1 tools/bench/measure_joules.sh   # single run with f16s on
#   tools/bench/measure_joules.sh --source powermetrics # force powermetrics (needs sudo)
#
# Co-existence: dismantle runs under `nice -n 19 taskpolicy -b` (background QoS).
set -uo pipefail
cd "$(dirname "$0")/../.."

BIN="${BIN:-./target/release/dismantle}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"

# Locked Qwen fast-path (the shipped decode config). f16s (A6.5) is opt-in and
# passed through from the environment / --f16s flag, not baked in here.
BASE_ENV="DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_VOCAB_PRUNE=32000 \
DISMANTLE_QWEN_Q4K_LMHEAD=1 DISMANTLE_QWEN_FFN_DOWN_Q4K=1 \
DISMANTLE_QWEN_Q4K_PREDEC=1"

TOKENS=256
PROMPT='fn fibonacci(n: u64) -> u64 {'
SOURCE="auto"
DO_F16S=0          # when 1, run baseline AND f16s-on, then compare J/tok
SAMPLE_MS=200      # power sampling interval

die() { echo "error: $*" >&2; exit 64; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokens)  TOKENS="$2"; shift 2;;
    --prompt)  PROMPT="$2"; shift 2;;
    --source)  SOURCE="$2"; shift 2;;   # auto | macmon | powermetrics
    --f16s)    DO_F16S=1; shift;;
    --sample-ms) SAMPLE_MS="$2"; shift 2;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) die "unknown arg: $1";;
  esac
done
[[ -x "$BIN" ]] || die "binary not found/executable: $BIN (cargo build --release?)"
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"

# ---------------------------------------------------------------------------
# Power-source detection
# ---------------------------------------------------------------------------
detect_source() {
  if [[ "$SOURCE" == "macmon" ]]; then echo macmon; return; fi
  if [[ "$SOURCE" == "powermetrics" ]]; then echo powermetrics; return; fi
  # auto
  if command -v macmon >/dev/null 2>&1; then echo macmon; return; fi
  if command -v powermetrics >/dev/null 2>&1; then echo powermetrics; return; fi
  echo none
}
SRC="$(detect_source)"

if [[ "$SRC" == "none" ]]; then
  cat >&2 <<EOF
NO POWER SOURCE AVAILABLE.
  - macmon not installed   -> install (sudo-free):  brew install macmon
  - powermetrics not found -> (unexpected on macOS)
Cannot measure joules-per-token. Exiting without sudo.
EOF
  exit 3
fi

if [[ "$SRC" == "powermetrics" ]]; then
  # powermetrics needs sudo. Refuse to prompt in an unattended context.
  if ! sudo -n true 2>/dev/null; then
    cat >&2 <<EOF
POWER SOURCE = powermetrics, but it requires SUDO (a password).
This is the ATTENDED path. Either:
  (a) install the sudo-free reader:   brew install macmon
      then re-run:                    tools/bench/measure_joules.sh
  (b) run this script attended:       sudo -v && tools/bench/measure_joules.sh
      (the sudo -v caches your password so the background sampler can spawn
       powermetrics without a second prompt)
Refusing to invoke sudo non-interactively. Exiting.
EOF
    exit 4
  fi
fi

echo "=== measure_joules — Qwen-3B decode J/tok baseline (§8 L4.2) ==="
echo "power source : $SRC"
echo "tokens       : $TOKENS"
echo "profile      : $PROFILE"
echo "base env     : (locked fast-path) TCB+vocab32k+Q4K-lmhead+FFN-down-Q4K+predec"

# ---------------------------------------------------------------------------
# Power samplers: each writes one watts-value per line to $OUT_W while the
# decode runs. avg = mean of samples taken during the decode window.
# ---------------------------------------------------------------------------
# macmon JSON pipe mode: `macmon pipe -i <ms>` emits one JSON object per sample.
# We extract package power (cpu+gpu+ane) and, separately, GPU power if present.
sample_macmon() {  # $1 = watts-file, $2 = gpu-watts-file
  macmon pipe -i "$SAMPLE_MS" 2>/dev/null | while IFS= read -r line; do
    # Fields vary slightly by macmon version; pull the common ones defensively.
    pkg=$(printf '%s' "$line" | /usr/bin/python3 -c 'import sys,json
try:
  d=json.load(sys.stdin)
except Exception:
  sys.exit(0)
# macmon emits e.g. {"cpu_power":..,"gpu_power":..,"ane_power":..,"all_power":..}
allp=d.get("all_power")
if allp is None:
  allp=sum(d.get(k,0) or 0 for k in ("cpu_power","gpu_power","ane_power"))
print(f"{allp:.4f}")' )
    gpu=$(printf '%s' "$line" | /usr/bin/python3 -c 'import sys,json
try:
  d=json.load(sys.stdin)
except Exception:
  sys.exit(0)
print(f"{(d.get(\"gpu_power\",0) or 0):.4f}")' )
    [[ -n "$pkg" ]] && echo "$pkg" >> "$1"
    [[ -n "$gpu" ]] && echo "$gpu" >> "$2"
  done
}

# powermetrics: sample the SMC/CPU+GPU power group. "Combined Power (CPU + GPU + ANE): N mW"
sample_powermetrics() {  # $1 = watts-file, $2 = gpu-watts-file
  sudo powermetrics --samplers cpu_power,gpu_power -i "$SAMPLE_MS" 2>/dev/null | \
  while IFS= read -r line; do
    case "$line" in
      *"Combined Power (CPU + GPU + ANE):"*)
        mw=$(printf '%s' "$line" | grep -oE '[0-9]+' | tail -1)
        [[ -n "$mw" ]] && echo "scale=4; $mw/1000" | bc >> "$1" ;;
      *"GPU Power:"*)
        mw=$(printf '%s' "$line" | grep -oE '[0-9]+' | tail -1)
        [[ -n "$mw" ]] && echo "scale=4; $mw/1000" | bc >> "$2" ;;
    esac
  done
}

mean_of() {  # mean of newline-separated numbers; 0 if empty
  awk '{s+=$1; n++} END{ if(n>0) printf "%.4f", s/n; else print "0" }' "$1" 2>/dev/null
}

# ---------------------------------------------------------------------------
# One measured run: returns (via globals) dec_tps, tokens, decode_wall_s,
# avg_pkg_W, avg_gpu_W, joules_per_tok.
# ---------------------------------------------------------------------------
run_one() {  # $1 = label, $2 = extra-env (e.g. "DISMANTLE_QWEN_PREDEC_F16SCALES=1")
  local label="$1" extra="$2"
  local wfile gfile statf
  wfile="$(mktemp)"; gfile="$(mktemp)"; statf="$(mktemp)"
  : > "$wfile"; : > "$gfile"

  # start the sampler in the background
  if [[ "$SRC" == "macmon" ]]; then
    sample_macmon "$wfile" "$gfile" &
  else
    sample_powermetrics "$wfile" "$gfile" &
  fi
  local sampler_pid=$!
  sleep 0.3   # let the sampler warm up

  # run the decode. `generate` always emits a timing summary on stderr:
  #   [stats] reason=.. prompt=.. completion=.. prefill_ms=.. decode_ms=.. dec_tps=.. ...
  # (printed unconditionally on the Done event; no flag needed). 2>&1 merges it.
  env $BASE_ENV $extra nice -n 19 taskpolicy -b "$BIN" generate \
    --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
    --prompt "$PROMPT" --max-new-tokens "$TOKENS" --temperature 0 --seed 0 \
    > "$statf" 2>&1

  # stop the sampler
  kill "$sampler_pid" 2>/dev/null; wait "$sampler_pid" 2>/dev/null

  # parse the stats line
  local statline dec_ms dec_tps comp_tok
  statline=$(grep -E '\[stats\]' "$statf" | tail -1)
  if [[ -z "$statline" ]]; then
    echo "  [$label] WARN: no [stats] line found. Raw tail:" >&2
    tail -3 "$statf" >&2
  fi
  dec_ms=$(printf '%s' "$statline"  | grep -oE 'decode_ms=[0-9.]+'  | grep -oE '[0-9.]+')
  dec_tps=$(printf '%s' "$statline" | grep -oE 'dec_tps=[0-9.]+'    | grep -oE '[0-9.]+')
  comp_tok=$(printf '%s' "$statline"| grep -oE 'completion=[0-9]+'  | grep -oE '[0-9]+')
  [[ -z "$comp_tok" ]] && comp_tok="$TOKENS"
  [[ -z "$dec_ms"  ]] && dec_ms="0"

  local decode_wall_s avg_pkg avg_gpu joules jtok
  decode_wall_s=$(awk -v m="$dec_ms" 'BEGIN{printf "%.4f", m/1000}')
  avg_pkg=$(mean_of "$wfile")
  avg_gpu=$(mean_of "$gfile")
  # energy = power(W) * time(s);  J/tok = energy / tokens
  joules=$(awk -v w="$avg_pkg" -v s="$decode_wall_s" 'BEGIN{printf "%.4f", w*s}')
  jtok=$(awk -v j="$joules" -v t="$comp_tok" 'BEGIN{ if(t>0) printf "%.4f", j/t; else print "0"}')

  printf '\n--- %s ---\n' "$label"
  printf '  dec_tps        : %s\n' "${dec_tps:-?}"
  printf '  tokens         : %s\n' "$comp_tok"
  printf '  decode_wall_s  : %s\n' "$decode_wall_s"
  printf '  avg pkg power  : %s W  (CPU+GPU+ANE)\n' "$avg_pkg"
  printf '  avg GPU power  : %s W\n' "$avg_gpu"
  printf '  decode energy  : %s J\n' "$joules"
  printf '  >> J/token     : %s  <<\n' "$jtok"

  # export for the caller's comparison
  LAST_JTOK="$jtok"; LAST_TPS="$dec_tps"; LAST_PKG="$avg_pkg"
  rm -f "$wfile" "$gfile" "$statf"
}

LAST_JTOK=""; LAST_TPS=""; LAST_PKG=""

# baseline (locked fast-path; honors DISMANTLE_QWEN_PREDEC_F16SCALES if set in env)
run_one "baseline (locked fast-path)" ""
BASE_JTOK="$LAST_JTOK"; BASE_TPS="$LAST_TPS"

if [[ "$DO_F16S" == 1 ]]; then
  run_one "f16s ON (A6.5 lever)" "DISMANTLE_QWEN_PREDEC_F16SCALES=1"
  F_JTOK="$LAST_JTOK"; F_TPS="$LAST_TPS"
  echo ""
  echo "=== comparison: does the faster path also sip less? (finish-sooner-and-idle) ==="
  awk -v bj="$BASE_JTOK" -v fj="$F_JTOK" -v bt="$BASE_TPS" -v ft="$F_TPS" 'BEGIN{
    if (bj>0 && fj>0) printf "  J/tok: %.4f -> %.4f  (%.1f%%)\n", bj, fj, (fj/bj-1)*100;
    if (bt>0 && ft>0) printf "  tps  : %.2f -> %.2f  (%.1f%%)\n", bt, ft, (ft/bt-1)*100;
  }'
fi

echo ""
echo "done. (J/tok = avg_power_W * decode_wall_s / tokens)"
