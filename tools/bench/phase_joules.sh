#!/usr/bin/env bash
# phase_joules.sh — per-phase J/tok energy attribution for Qwen-3B decode.
#
# Reports how decode energy is split between three GPU work categories:
#
#   GEMV         weight-matrix projections (q/k/v/o/gate/up/down + LM-head)
#   Attention    MHA decode kernels (mha_decode_* / flash_attn_*)
#   Trivial-ops  everything else (rmsnorm, rope, silu_mul, add_inplace,
#                memcpy, embed, quantize, sample_argmax, add_rmsnorm_fused)
#
# METHOD (the 0.3 proxy):
#   Energy tracks GPU time. GPU time is proportional to CPU-encode time in
#   the single-CB-per-token model (one MTLCommandBuffer per token, no split).
#   We measure CPU-encode time per kernel via HAWKING_TCB_TRACE=cpu on a
#   short bench warm-up pass, compute phase fractions, then multiply total
#   J/tok (from a normal power-measured decode) by those fractions.
#
#   J/tok_phase = J/tok_total * (cpu_encode_us_phase / cpu_encode_us_total)
#
# CAVEATS:
#   1. The trace pass and the energy pass are SEPARATE runs; minor variance.
#   2. CPU-encode time is a proxy for GPU time, not a direct GPU measurement.
#      For direct GPU-time fractions use HAWKING_TCB_TRACE=gpu_prod on the
#      bench pass (slower; changes absolute tps but preserves relative ratios).
#   3. The energy measurement step runs the binary WITHOUT the trace overhead
#      so measured tps/J/tok are production-representative.
#
# POWER SOURCE (same auto-detection as measure_joules.sh):
#   macmon (sudo-free, preferred) or powermetrics (needs sudo).
#
# Usage:
#   tools/bench/phase_joules.sh                       # 256 tok attribution
#   tools/bench/phase_joules.sh --tokens 512
#   tools/bench/phase_joules.sh --domains             # + per-domain GPU/DRAM J/tok (macmon, no dep)
#   tools/bench/phase_joules.sh --trace-mode gpu_prod  # real GPU fractions
#   tools/bench/phase_joules.sh --source powermetrics  # force power source
#   HAWKING_QWEN_PREDEC_F16SCALES=1 tools/bench/phase_joules.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

BIN="${BIN:-./target/release/hawking}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"

BASE_ENV="HAWKING_QWEN_TCB=1 HAWKING_QWEN_VOCAB_PRUNE=32000 \
HAWKING_QWEN_Q4K_LMHEAD=1 HAWKING_QWEN_FFN_DOWN_Q4K=1 \
HAWKING_QWEN_Q4K_PREDEC=1"

TOKENS=256
PROMPT='fn fibonacci(n: u64) -> u64 {'
SOURCE="auto"
SAMPLE_MS=200
TRACE_MODE="cpu"       # cpu | gpu_prod (see CAVEATS)
TRACE_TOKENS=64        # short trace run; fractions stabilize quickly
ZEUS=0                 # --zeus: ALSO emit MEASURED per-domain mJ/tok (zeus-apple-silicon).
                       # Default OFF -> golden proxy output unchanged.
DOMAINS=0              # --domains: ALSO emit per-domain GPU + DRAM J/tok from macmon
                       # (ram_power, sudo-free, no dep). Default OFF -> proxy output unchanged.

die() { printf 'error: %s\n' "$*" >&2; exit 64; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokens)      TOKENS="$2"; shift 2;;
    --prompt)      PROMPT="$2"; shift 2;;
    --source)      SOURCE="$2"; shift 2;;
    --sample-ms)   SAMPLE_MS="$2"; shift 2;;
    --trace-mode)  TRACE_MODE="$2"; shift 2;;
    --trace-tokens) TRACE_TOKENS="$2"; shift 2;;
    --zeus)        ZEUS=1; shift;;
    --domains)     DOMAINS=1; shift;;
    -h|--help)     sed -n '2,50p' "$0"; exit 0;;
    *) die "unknown arg: $1";;
  esac
done
[[ -x "$BIN" ]] || die "binary not found/executable: $BIN (cargo build --release?)"
[[ -f "$WEIGHTS" ]] || die "weights not found: $WEIGHTS"

# ---------------------------------------------------------------------------
# Power-source detection (identical logic to measure_joules.sh)
# ---------------------------------------------------------------------------
detect_source() {
  if [[ "$SOURCE" == "macmon" ]]; then echo macmon; return; fi
  if [[ "$SOURCE" == "powermetrics" ]]; then echo powermetrics; return; fi
  if command -v macmon >/dev/null 2>&1; then echo macmon; return; fi
  if command -v powermetrics >/dev/null 2>&1; then echo powermetrics; return; fi
  echo none
}
SRC="$(detect_source)"

if [[ "$SRC" == "none" ]]; then
  cat >&2 <<'NOPOWER'
NO POWER SOURCE AVAILABLE.
  - macmon not installed   -> brew install macmon
  - powermetrics not found -> (unexpected on macOS)
Cancel. Install macmon and re-run.
NOPOWER
  exit 3
fi

if [[ "$SRC" == "powermetrics" ]]; then
  if ! sudo -n true 2>/dev/null; then
    cat >&2 <<'NOSUDO'
POWER SOURCE = powermetrics but sudo is required.
  (a) brew install macmon   (sudo-free)
  (b) sudo -v && tools/bench/phase_joules.sh
NOSUDO
    exit 4
  fi
fi

echo "=== phase_joules — per-phase J/tok energy attribution (paradigmshift V.4) ==="
echo "power source : $SRC"
echo "tokens       : $TOKENS (energy run) + $TRACE_TOKENS (trace pass)"
echo "trace mode   : $TRACE_MODE"
echo "base env     : (locked fast-path) TCB+vocab32k+Q4K-lmhead+FFN-down-Q4K+predec"

# ---------------------------------------------------------------------------
# Power samplers (same as measure_joules.sh, with pkill-P orphan fix applied)
# ---------------------------------------------------------------------------
sample_macmon() {  # $1=pkg-W file  $2=gpu-W file  $3=dram-W file (optional)
  macmon pipe -i "$SAMPLE_MS" 2>/dev/null | while IFS= read -r line; do
    /usr/bin/python3 -c 'import sys,json
try:
  d=json.load(sys.stdin)
except Exception:
  sys.exit(0)
allp=d.get("all_power")
if allp is None:
  allp=sum(d.get(k,0) or 0 for k in ("cpu_power","gpu_power","ane_power"))
gpu=d.get("gpu_power",0) or 0
dram=d.get("ram_power",0) or 0
print(f"{allp:.4f}\t{gpu:.4f}\t{dram:.4f}")' <<< "$line" | {
      IFS=$'\t' read -r p g r
      [[ -n "$p" ]] && printf '%s\n' "$p" >> "$1"
      [[ -n "$g" ]] && printf '%s\n' "$g" >> "$2"
      [[ -n "${3:-}" && -n "$r" ]] && printf '%s\n' "$r" >> "$3"
    }
  done
}

sample_powermetrics() {
  sudo powermetrics --samplers cpu_power,gpu_power -i "$SAMPLE_MS" 2>/dev/null | \
  while IFS= read -r line; do
    case "$line" in
      *"Combined Power (CPU + GPU + ANE):"*)
        mw=$(printf '%s' "$line" | grep -oE '[0-9]+' | tail -1)
        [[ -n "$mw" ]] && printf '%s\n' "scale=4; $mw/1000" | bc >> "$1" ;;
      *"GPU Power:"*)
        mw=$(printf '%s' "$line" | grep -oE '[0-9]+' | tail -1)
        [[ -n "$mw" ]] && printf '%s\n' "scale=4; $mw/1000" | bc >> "$2" ;;
    esac
  done
}

stop_sampler() {  # $1 = sampler_pid
  pkill -P "$1" 2>/dev/null || true
  kill "$1" 2>/dev/null
  wait "$1" 2>/dev/null || true
}

mean_of() {
  awk '{s+=$1; n++} END{ if(n>0) printf "%.4f", s/n; else print "0" }' "$1" 2>/dev/null
}

# ---------------------------------------------------------------------------
# STEP 1: Trace pass — collect CPU-encode-time fractions per kernel.
# Uses HAWKING_TCB_TRACE=$TRACE_MODE + bench --trace-json.
# A short run (TRACE_TOKENS) suffices; fractions stabilize over ~20 tokens.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: trace pass (${TRACE_TOKENS} tokens, HAWKING_TCB_TRACE=${TRACE_MODE}) ---"
TRACEF="$(mktemp /tmp/phase_joules_trace.XXXXXX.json)"

# We use 'bench --suite decode' which writes dispatch_samples to --trace-json.
# Extra env is passed verbatim; the trace env var is added here.
env $BASE_ENV \
  HAWKING_TCB_TRACE="$TRACE_MODE" \
  nice -n 19 taskpolicy -b "$BIN" bench \
    --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
    --suite decode --trials 1 --max-new-tokens "$TRACE_TOKENS" \
    --trace-json "$TRACEF" \
    --trace-dispatch \
  > /dev/null 2>&1 || {
    echo "  WARN: trace pass exited non-zero; fractions may be unavailable" >&2
}

# Parse the trace JSON with inline Python3 to get phase fractions.
# Phase classification (matches analyze_tcb_trace.py kernel_name map):
#   GEMV:     gemm_q4_k_*, gemv_*, gemm_q6_k_*, gemm_q4k_fast_*, gemm_q4_k_a8_*
#   Attn:     mha_decode_*, flash_attn_*
#   Trivial:  everything else
PHASE_FRACS=$(/usr/bin/python3 - "$TRACEF" <<'PYEOF'
import sys, json
trace_path = sys.argv[1]
try:
    doc = json.loads(open(trace_path).read())
except Exception as e:
    print(f"0.72 0.15 0.13")  # fallback from §8 energy-model priors
    sys.exit(0)

# Walk the nested JSON to find dispatch_samples.
def find_samples(obj):
    if isinstance(obj, dict):
        if "dispatch_samples" in obj and isinstance(obj["dispatch_samples"], list):
            return obj["dispatch_samples"]
        for v in obj.values():
            r = find_samples(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_samples(v)
            if r is not None:
                return r
    return None

samples = find_samples(doc)
if not samples:
    print("0.72 0.15 0.13")  # fallback
    sys.exit(0)

def classify(name):
    n = name.lower()
    # GEMV: all weight-projection kernels
    if (n.startswith("gemm_q4_k") or n.startswith("gemv_") or
        n.startswith("gemm_q6_k") or n.startswith("gemm_q4k_fast") or
        n.startswith("gemm_q4_k_a8")):
        return "gemv"
    # Attention: MHA decode + flash variants
    if n.startswith("mha_decode") or n.startswith("flash_attn"):
        return "attn"
    # Trivial: rmsnorm, rope, silu_mul, add_inplace, memcpy, embed,
    #          quantize, sample_argmax, add_rmsnorm_fused, etc.
    return "trivial"

us = {"gemv": 0, "attn": 0, "trivial": 0}
for s in samples:
    wall = s.get("wall_us") or 0
    gpu  = s.get("gpu_us")  or 0
    # prefer gpu_us (ProdCbGpu mode) over wall_us (CpuEncode mode)
    t = gpu if gpu > 0 else wall
    phase = classify(s.get("kernel_name", ""))
    us[phase] += t

total = sum(us.values())
if total == 0:
    print("0.72 0.15 0.13")  # fallback
    sys.exit(0)

f_gemv    = us["gemv"]    / total
f_attn    = us["attn"]    / total
f_trivial = us["trivial"] / total
print(f"{f_gemv:.6f} {f_attn:.6f} {f_trivial:.6f}")
print(f"# gemv_us={us['gemv']} attn_us={us['attn']} trivial_us={us['trivial']} total_us={total}",
      file=sys.stderr)
PYEOF
)

# Extract the three fractions (space-separated on first line).
F_GEMV=$(printf '%s' "$PHASE_FRACS" | awk 'NR==1{print $1}')
F_ATTN=$(printf '%s' "$PHASE_FRACS" | awk 'NR==1{print $2}')
F_TRIVIAL=$(printf '%s' "$PHASE_FRACS" | awk 'NR==1{print $3}')

rm -f "$TRACEF"

pct() { awk -v f="$1" 'BEGIN{printf "%.1f", f*100}'; }
printf '  GEMV        fraction: %.4f  (%s%%\n' "$F_GEMV" "$(pct "$F_GEMV")"
printf '  Attention   fraction: %.4f  (%s%%\n' "$F_ATTN" "$(pct "$F_ATTN")"
printf '  Trivial-ops fraction: %.4f  (%s%%\n' "$F_TRIVIAL" "$(pct "$F_TRIVIAL")"

# ---------------------------------------------------------------------------
# STEP 2: Energy-measured decode — same as run_one in measure_joules.sh
# (NO trace overhead; this is the production-representative power reading).
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: energy pass (${TOKENS} tokens, no trace, production path) ---"

wfile="$(mktemp)"; gfile="$(mktemp)"; dfile="$(mktemp)"; statf="$(mktemp)"
: > "$wfile"; : > "$gfile"; : > "$dfile"

if [[ "$SRC" == "macmon" ]]; then
  sample_macmon "$wfile" "$gfile" "$dfile" &
else
  sample_powermetrics "$wfile" "$gfile" &
fi
sampler_pid=$!
sleep 0.3

env $BASE_ENV nice -n 19 taskpolicy -b "$BIN" generate \
  --weights "$WEIGHTS" --kernel-profile "$PROFILE" \
  --prompt "$PROMPT" --max-new-tokens "$TOKENS" --temperature 0 --seed 0 \
  > "$statf" 2>&1

stop_sampler "$sampler_pid"

# Parse stats line.
statline=$(grep -E '\[stats\]' "$statf" | tail -1)
if [[ -z "$statline" ]]; then
  # FAIL LOUD — decode produced no measurement (e.g. stale-profile shader-hash,
  # OOM, or a model error). Returning 0.0000 J/tok as a fake-pass is what burned
  # the first clean-room window; exit nonzero so the queue marks this FAIL.
  echo "  FAIL: no [stats] line — decode produced no measurement. Raw tail:" >&2
  tail -3 "$statf" >&2
  rm -f "$wfile" "$gfile" "$dfile" "$statf"
  exit 1
fi
dec_ms=$(printf '%s' "$statline"  | grep -oE 'decode_ms=[0-9.]+' | grep -oE '[0-9.]+')
dec_tps=$(printf '%s' "$statline" | grep -oE 'dec_tps=[0-9.]+'   | grep -oE '[0-9.]+')
comp_tok=$(printf '%s' "$statline"| grep -oE 'completion=[0-9]+' | grep -oE '[0-9]+')
[[ -z "$comp_tok" ]] && comp_tok="$TOKENS"
[[ -z "$dec_ms"   ]] && dec_ms="0"

decode_wall_s=$(awk -v m="$dec_ms" 'BEGIN{printf "%.4f", m/1000}')
avg_pkg=$(mean_of "$wfile")
avg_gpu=$(mean_of "$gfile")
avg_dram=$(mean_of "$dfile")
joules=$(awk -v w="$avg_pkg" -v s="$decode_wall_s" 'BEGIN{printf "%.4f", w*s}')
jtok=$(awk -v j="$joules" -v t="$comp_tok" 'BEGIN{ if(t>0) printf "%.4f", j/t; else print "0"}')
gpu_jtok=$(awk -v w="$avg_gpu" -v s="$decode_wall_s" -v t="$comp_tok" 'BEGIN{ if(t>0) printf "%.4f", w*s/t; else print "0"}')
dram_jtok=$(awk -v w="$avg_dram" -v s="$decode_wall_s" -v t="$comp_tok" 'BEGIN{ if(t>0) printf "%.4f", w*s/t; else print "0"}')

rm -f "$wfile" "$gfile" "$dfile" "$statf"

printf '  dec_tps        : %s\n' "${dec_tps:-?}"
printf '  tokens         : %s\n' "$comp_tok"
printf '  decode_wall_s  : %s\n' "$decode_wall_s"
printf '  avg pkg power  : %s W  (CPU+GPU+ANE)\n' "$avg_pkg"
printf '  avg GPU power  : %s W\n' "$avg_gpu"
printf '  decode energy  : %s J\n' "$joules"
printf '  >> J/token     : %s  <<\n' "$jtok"

if [[ "$DOMAINS" == 1 ]]; then
  if [[ "$SRC" == "macmon" ]]; then
    printf '  --- MEASURED per-domain J/tok (macmon IOReport power x wall / tokens) ---\n'
    printf '    GPU   : %s J/tok\n' "$gpu_jtok"
    # macmon ram_power is absent on some versions/chips -> avg_dram==0. Print the
    # truth, not a fake 0.0000 DRAM J/tok (P4, bench-harness audit).
    if awk -v d="$avg_dram" 'BEGIN{exit !(d+0>0)}'; then
      printf '  avg DRAM power : %s W  (ram_power, external-DRAM domain)\n' "$avg_dram"
      printf '    DRAM  : %s J/tok\n' "$dram_jtok"
    else
      printf '    DRAM  : unavailable (macmon ram_power absent in this version/chip)\n'
    fi
    printf '    NOTE  : MODEL-ESTIMATE (~1 mJ res); GPU-SRAM not exposed on M3 Pro;\n'
    printf '            power averaged over the whole decode window — use --tokens 512+\n'
    printf '            and a clean room (Claude quit) for a publishable absolute number.\n'
  else
    printf '  --domains: DRAM domain only available via macmon (current source: %s); skipped.\n' "$SRC"
  fi
fi

# ---------------------------------------------------------------------------
# STEP 3: Phase attribution — multiply total J/tok by phase fractions.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: per-phase J/tok attribution (proxy: cpu-encode-time fractions) ---"
/usr/bin/python3 - "$jtok" "$F_GEMV" "$F_ATTN" "$F_TRIVIAL" <<'PYEOF2'
import sys
jtok, f_gemv, f_attn, f_trivial = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])

j_gemv    = jtok * f_gemv
j_attn    = jtok * f_attn
j_trivial = jtok * f_trivial

print(f"  GEMV         : {j_gemv:.4f} J/tok  ({f_gemv*100:.1f}% of decode energy)")
print(f"  Attention    : {j_attn:.4f} J/tok  ({f_attn*100:.1f}% of decode energy)")
print(f"  Trivial-ops  : {j_trivial:.4f} J/tok  ({f_trivial*100:.1f}% of decode energy)")
print(f"  ─────────────────────────────────────────────")
print(f"  Total (check): {j_gemv+j_attn+j_trivial:.4f} J/tok  (== {jtok:.4f}; proxy residual = {abs(jtok-(j_gemv+j_attn+j_trivial)):.6f})")
print()
print("  Proxy: J/tok_phase = J/tok_total * (cpu_encode_us_phase / cpu_encode_us_total)")
print("  For direct GPU-time fractions: --trace-mode gpu_prod (caution: slower bench pass)")
PYEOF2

# ---------------------------------------------------------------------------
# STEP 4 (opt-in, --zeus): MEASURED per-domain energy via zeus-apple-silicon.
# Replaces the Step-3 cpu-encode PROXY with a real IOReport "Energy Model"
# reading (GPU-mJ + DRAM-mJ per token), sudo-free. Default OFF so the golden
# proxy output above is byte-identical without the flag. Runs its OWN locked-
# fast-path decode (same BASE_ENV) wrapped in a zeus begin/end window.
# ---------------------------------------------------------------------------
if [[ "$ZEUS" == 1 ]]; then
  echo ""
  echo "--- Step 4: MEASURED per-domain mJ/tok (zeus-apple-silicon, IOReport Energy Model) ---"
  ZPY="${ZEUS_PYTHON:-/usr/bin/python3}"
  if ! "$ZPY" tools/bench/zeus_joules.py --tokens "$TOKENS" --prompt "$PROMPT"; then
    echo "  NOTE: zeus measured-energy step failed (see recipe above). Proxy result stands." >&2
  fi
fi

echo ""
echo "done."

