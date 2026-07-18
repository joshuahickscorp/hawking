#!/usr/bin/env bash
# =============================================================================
# tools/bench/energy_paired.sh — side-by-side energy + throughput comparison.
#
# Runs COMMAND_A and COMMAND_B with power sampling and prints a table:
#
#   lane      | dec_tps | J/tok_GPU | J/tok_pkg | J/req_GPU | verdict
#   command_a | 31.2    | 0.196     | 0.284     | 0.72      | baseline
#   command_b | 43.1    | 0.168     | 0.251     | 0.61      | -14.3% J/tok +38.1% tps BETTER
#
# NEW metrics vs phase_joules.sh:
#   J/tok      = power_W / dec_tps  (GPU or package)
#   J/req      = avg_power_W * (prefill_ms + decode_ms) / 1000  (whole request)
#   J/eff-tok  = J/req / effective_tokens  (effective_tokens = completion +
#                prefix_cache_hits + spec_accepted, if metrics are available)
#
# CONTAMINATION NOTE: J/tok and tps are absolute; run with the agent closed for
#   publishable numbers. Relative A/B deltas are contamination-robust.
#
# USAGE:
#   # env-var style (recommended):
#   COMMAND_A="./target/release/hawking generate --weights models/... --profile fast"  \
#   COMMAND_B="./target/release/hawking generate --weights models/... --profile race"  \
#   tools/bench/energy_paired.sh
#
#   # or pass directly:
#   tools/bench/energy_paired.sh \
#     --cmd-a "./target/release/hawking generate --weights models/... --profile fast" \
#     --cmd-b "./target/release/hawking generate --weights models/... --profile race"
#
# ENVIRONMENT (all optional):
#   COMMAND_A / COMMAND_B  full shell command strings for the two variants
#   TOKENS                 decode token count injected when --max-new-tokens
#                          is NOT already present in COMMAND_A/B (default: 200)
#   PROMPT                 prompt injected when --prompt is NOT already in cmd
#                          (default: "fn fibonacci(n: u64) -> u64 {")
#   SAMPLE_MS              macmon sampling interval in ms (default: 200)
#   SERVE_MODE             1 = treat commands as "dismantle serve" + send one
#                          HTTP request, poll /metrics for stats (default: 0)
#   METRICS_URL_A/B        override /metrics URL when SERVE_MODE=1
#
# GRACEFUL DEGRADATION:
#   - macmon absent  → J columns print "N/A"
#   - command absent → lane skipped with a warning
#   - no [stats] line → that lane's tps / J values are "?"
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

# ── Defaults ──────────────────────────────────────────────────────────────────
TOKENS="${TOKENS:-200}"
PROMPT="${PROMPT:-fn fibonacci(n: u64) -> u64 {}"
SAMPLE_MS="${SAMPLE_MS:-200}"
SERVE_MODE="${SERVE_MODE:-0}"
CMD_A="${COMMAND_A:-}"
CMD_B="${COMMAND_B:-}"
METRICS_URL_A="${METRICS_URL_A:-http://127.0.0.1:8282/metrics}"
METRICS_URL_B="${METRICS_URL_B:-http://127.0.0.1:8282/metrics}"

die() { printf 'error: %s\n' "$*" >&2; exit 64; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cmd-a)   CMD_A="$2"; shift 2;;
    --cmd-b)   CMD_B="$2"; shift 2;;
    --tokens)  TOKENS="$2"; shift 2;;
    --prompt)  PROMPT="$2"; shift 2;;
    --sample-ms) SAMPLE_MS="$2"; shift 2;;
    --serve)   SERVE_MODE=1; shift;;
    -h|--help) sed -n '2,60p' "$0"; exit 0;;
    *) die "unknown arg: $1";;
  esac
done

[[ -n "$CMD_A" ]] || die "COMMAND_A (or --cmd-a) is required"
[[ -n "$CMD_B" ]] || die "COMMAND_B (or --cmd-b) is required"

# ── macmon detection ──────────────────────────────────────────────────────────
HAS_MACMON=0
command -v macmon >/dev/null 2>&1 && HAS_MACMON=1
if [[ "$HAS_MACMON" -eq 0 ]]; then
  printf 'WARN: macmon not found — J/tok columns will show N/A\n' >&2
  printf '      Install: brew install macmon\n' >&2
fi

# ── Power sampler helpers (same pattern as phase_joules.sh) ───────────────────
sample_macmon() {  # $1=pkg_file  $2=gpu_file
  [[ "$HAS_MACMON" -eq 0 ]] && return
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
print(f"{allp:.4f}\t{gpu:.4f}")' <<< "$line" | {
      IFS=$'\t' read -r p g
      [[ -n "${p:-}" ]] && printf '%s\n' "$p" >> "$1"
      [[ -n "${g:-}" ]] && printf '%s\n' "$g" >> "$2"
    }
  done
}

stop_sampler() {  # $1=pid
  pkill -P "$1" 2>/dev/null || true
  kill "$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
}

mean_of() {
  awk '{s+=$1;n++} END{if(n>0)printf "%.4f",s/n; else print "0"}' "$1" 2>/dev/null || echo "0"
}

# ── Inject --max-new-tokens / --prompt if missing ────────────────────────────
inject_args() {
  local cmd="$1"
  [[ "$cmd" != *"--max-new-tokens"* ]] && cmd="$cmd --max-new-tokens $TOKENS"
  [[ "$cmd" != *"--prompt"* ]]         && cmd="$cmd --prompt '$PROMPT'"
  echo "$cmd"
}

CMD_A_FULL="$(inject_args "$CMD_A")"
CMD_B_FULL="$(inject_args "$CMD_B")"

# ── Single-lane runner — returns stats via named temp files ───────────────────
#   Writes: $TMPD/$label_tps  $TMPD/$label_dec_ms  $TMPD/$label_req_ms
#           $TMPD/$label_avg_pkg  $TMPD/$label_avg_gpu
#           $TMPD/$label_eff_tok  $TMPD/$label_log
TMPD="$(mktemp -d /tmp/energy_paired.XXXXXX)"
cleanup() { rm -rf "$TMPD"; }
trap cleanup EXIT

run_lane() {
  local label="$1"
  local cmd="$2"
  local metrics_url="${3:-}"

  local pkgf="$TMPD/${label}_pkg"
  local gpuf="$TMPD/${label}_gpu"
  local logf="$TMPD/${label}.log"
  : > "$pkgf"; : > "$gpuf"

  printf '\n--- Running %s ---\n' "$label"
  printf '    cmd: %s\n' "$cmd"

  if [[ "$SERVE_MODE" -eq 1 ]]; then
    # SERVE_MODE: the command is a running server; send one request then poll /metrics.
    local resp_f="$TMPD/${label}_resp.sse"
    [[ -n "$metrics_url" ]] || metrics_url="http://127.0.0.1:8282/metrics"

    # Capture metrics snapshot before
    local before_f="$TMPD/${label}_met_before"
    curl -sf "$metrics_url" > "$before_f" 2>/dev/null || : > "$before_f"

    sample_macmon "$pkgf" "$gpuf" &
    local samp_pid=$!
    sleep 0.3

    local t0; t0=$(date +%s%3N)
    curl -sf -N -X POST "$metrics_url/../v1/chat/completions" \
         -H 'Content-Type: application/json' \
         -d "{\"model\":\"dismantle\",\"stream\":true,\"temperature\":0,
              \"max_tokens\":$TOKENS,
              \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" \
         > "$resp_f" 2>&1 || true
    local t1; t1=$(date +%s%3N)
    local req_ms=$(( t1 - t0 ))

    stop_sampler "$samp_pid"

    # metrics delta
    local after_f="$TMPD/${label}_met_after"
    curl -sf "$metrics_url" > "$after_f" 2>/dev/null || : > "$after_f"

    local n_greedy_after n_greedy_before n_logits_after n_logits_before
    local rb_after rb_before dec_steps_after dec_steps_before
    n_greedy_after=$(grep  'hawking_greedy_decode_steps_total' "$after_f"  | grep -oE '[0-9]+$' | tail -1 || echo 0)
    n_greedy_before=$(grep 'hawking_greedy_decode_steps_total' "$before_f" | grep -oE '[0-9]+$' | tail -1 || echo 0)
    rb_after=$(grep  'hawking_gpu_readback_bytes_total' "$after_f"  | grep -oE '[0-9]+$' | tail -1 || echo 0)
    rb_before=$(grep 'hawking_gpu_readback_bytes_total' "$before_f" | grep -oE '[0-9]+$' | tail -1 || echo 0)

    local eff_tok=$(( (n_greedy_after - n_greedy_before) ))
    [[ "$eff_tok" -le 0 ]] && eff_tok="$TOKENS"

    local req_s; req_s=$(awk -v r="$req_ms" 'BEGIN{printf "%.4f", r/1000}')
    printf '%s\n' "$eff_tok"  > "$TMPD/${label}_eff_tok"
    printf '%s\n' "$req_ms"   > "$TMPD/${label}_req_ms"
    printf '%s\n' "$req_s"    > "$TMPD/${label}_req_s"
    # No [stats] line in serve mode — synthesise dec_ms from request - ~50ms prefill est
    local approx_dec_ms=$(awk -v r="$req_ms" 'BEGIN{d=r-50; if(d<1)d=r; printf "%d",d}')
    printf '%s\n' "$approx_dec_ms" > "$TMPD/${label}_dec_ms"
    local approx_tps; approx_tps=$(awk -v tok="$eff_tok" -v ms="$approx_dec_ms" \
      'BEGIN{if(ms>0)printf "%.2f",tok/(ms/1000); else print "0"}')
    printf '%s\n' "$approx_tps" > "$TMPD/${label}_tps"
  else
    # GENERATE_MODE: direct binary invocation; parse [stats] line.
    sample_macmon "$pkgf" "$gpuf" &
    local samp_pid=$!
    sleep 0.3

    eval "$cmd" > "$logf" 2>&1 || true

    stop_sampler "$samp_pid"

    local statline; statline=$(grep -E '\[stats\]' "$logf" | tail -1)
    if [[ -z "$statline" ]]; then
      printf '    WARN: no [stats] line in output — tps/J will be "?"\n'
      printf '?\n'  > "$TMPD/${label}_tps"
      printf '0\n'  > "$TMPD/${label}_dec_ms"
      printf '0\n'  > "$TMPD/${label}_req_ms"
      printf '%s\n' "$TOKENS" > "$TMPD/${label}_eff_tok"
    else
      local dec_ms; dec_ms=$(printf '%s' "$statline" | grep -oE 'decode_ms=[0-9.]+' | grep -oE '[0-9.]+')
      local pref_ms; pref_ms=$(printf '%s' "$statline" | grep -oE 'prefill_ms=[0-9.]+' | grep -oE '[0-9.]+' || echo 0)
      local tps; tps=$(printf '%s' "$statline" | grep -oE 'dec_tps=[0-9.]+' | grep -oE '[0-9.]+')
      local comp_tok; comp_tok=$(printf '%s' "$statline" | grep -oE 'completion=[0-9]+' | grep -oE '[0-9]+')
      [[ -z "$comp_tok" ]] && comp_tok="$TOKENS"
      [[ -z "$dec_ms"   ]] && dec_ms="0"
      [[ -z "$pref_ms"  ]] && pref_ms="0"
      local req_ms; req_ms=$(awk -v d="$dec_ms" -v p="$pref_ms" 'BEGIN{printf "%d", d+p}')

      # effective tokens: completion + any prefix_hits or spec_accepted counts if present
      local prefix_hits; prefix_hits=$(printf '%s' "$statline" | grep -oE 'prefix_hits=[0-9]+' | grep -oE '[0-9]+' || echo 0)
      local spec_acc; spec_acc=$(printf '%s' "$statline"    | grep -oE 'spec_accepted=[0-9]+' | grep -oE '[0-9]+' || echo 0)
      local eff_tok; eff_tok=$(awk -v c="$comp_tok" -v h="${prefix_hits:-0}" -v s="${spec_acc:-0}" \
        'BEGIN{print c+h+s}')

      printf '%s\n' "${tps:-0}" > "$TMPD/${label}_tps"
      printf '%s\n' "$dec_ms"   > "$TMPD/${label}_dec_ms"
      printf '%s\n' "$req_ms"   > "$TMPD/${label}_req_ms"
      printf '%s\n' "$eff_tok"  > "$TMPD/${label}_eff_tok"
    fi
  fi

  local avg_pkg; avg_pkg=$(mean_of "$pkgf")
  local avg_gpu; avg_gpu=$(mean_of "$gpuf")
  printf '%s\n' "$avg_pkg" > "$TMPD/${label}_avg_pkg"
  printf '%s\n' "$avg_gpu" > "$TMPD/${label}_avg_gpu"
}

# ── Run both lanes sequentially ───────────────────────────────────────────────
printf '=== energy_paired — throughput + energy comparison ===\n'
printf 'tokens   : %s\n' "$TOKENS"
printf 'prompt   : %s\n' "$PROMPT"
printf 'macmon   : %s\n' "$([[ $HAS_MACMON -eq 1 ]] && echo present || echo absent)"

run_lane "command_a" "$CMD_A_FULL" "$METRICS_URL_A"
run_lane "command_b" "$CMD_B_FULL" "$METRICS_URL_B"

# ── Compute derived metrics per lane ─────────────────────────────────────────
compute_metrics() {
  local label="$1"
  local tps_f="$TMPD/${label}_tps"
  local dec_ms_f="$TMPD/${label}_dec_ms"
  local req_ms_f="$TMPD/${label}_req_ms"
  local eff_tok_f="$TMPD/${label}_eff_tok"
  local pkg_f="$TMPD/${label}_avg_pkg"
  local gpu_f="$TMPD/${label}_avg_gpu"

  local tps;     tps=$(cat "$tps_f" 2>/dev/null || echo "?")
  local dec_ms;  dec_ms=$(cat "$dec_ms_f" 2>/dev/null || echo "0")
  local req_ms;  req_ms=$(cat "$req_ms_f" 2>/dev/null || echo "0")
  local eff_tok; eff_tok=$(cat "$eff_tok_f" 2>/dev/null || echo "$TOKENS")
  local avg_pkg; avg_pkg=$(cat "$pkg_f" 2>/dev/null || echo "0")
  local avg_gpu; avg_gpu=$(cat "$gpu_f" 2>/dev/null || echo "0")

  if [[ "$HAS_MACMON" -eq 0 ]]; then
    printf 'N/A\tN/A\tN/A\tN/A\tN/A\t%s\t%s\t%s\t%s\t%s' \
      "$tps" "$dec_ms" "$req_ms" "$eff_tok" "$avg_pkg"
    return
  fi

  /usr/bin/python3 - "$tps" "$dec_ms" "$req_ms" "$eff_tok" "$avg_pkg" "$avg_gpu" <<'PYEOF'
import sys
tps_s, dec_ms_s, req_ms_s, eff_tok_s, avg_pkg_s, avg_gpu_s = sys.argv[1:]

def try_float(s, default=0.0):
    try: return float(s)
    except: return default

tps     = try_float(tps_s)
dec_ms  = try_float(dec_ms_s)
req_ms  = try_float(req_ms_s)
eff_tok = try_float(eff_tok_s, 1.0)
avg_pkg = try_float(avg_pkg_s)
avg_gpu = try_float(avg_gpu_s)

def jtok(power_w, tps_val):
    if tps_val > 0 and power_w > 0:
        return f"{power_w / tps_val:.4f}"
    return "N/A"

def jreq(power_w, req_ms_val):
    if req_ms_val > 0 and power_w > 0:
        return f"{power_w * (req_ms_val / 1000):.4f}"
    return "N/A"

def jeff(power_w, req_ms_val, eff_tok_val):
    if eff_tok_val > 0 and req_ms_val > 0 and power_w > 0:
        j = power_w * (req_ms_val / 1000)
        return f"{j / eff_tok_val:.4f}"
    return "N/A"

jtok_gpu = jtok(avg_gpu, tps)
jtok_pkg = jtok(avg_pkg, tps)
jreq_gpu = jreq(avg_gpu, req_ms)
jeff_gpu = jeff(avg_gpu, req_ms, eff_tok)

# output: jtok_gpu jtok_pkg jreq_gpu jeff_gpu (tab-sep) + raw values
print(f"{jtok_gpu}\t{jtok_pkg}\t{jreq_gpu}\t{jeff_gpu}\t"
      f"{tps_s}\t{dec_ms_s}\t{req_ms_s}\t{eff_tok_s}\t{avg_pkg:.4f}")
PYEOF
}

A_METRICS=$(compute_metrics "command_a")
B_METRICS=$(compute_metrics "command_b")

parse_field() { printf '%s' "$1" | cut -f"$2"; }

A_JTOK_GPU=$(parse_field "$A_METRICS" 1)
A_JTOK_PKG=$(parse_field "$A_METRICS" 2)
A_JREQ_GPU=$(parse_field "$A_METRICS" 3)
A_JEFF_GPU=$(parse_field "$A_METRICS" 4)
A_TPS=$(parse_field "$A_METRICS" 5)
A_PKG=$(parse_field "$A_METRICS" 9)

B_JTOK_GPU=$(parse_field "$B_METRICS" 1)
B_JTOK_PKG=$(parse_field "$B_METRICS" 2)
B_JREQ_GPU=$(parse_field "$B_METRICS" 3)
B_JEFF_GPU=$(parse_field "$B_METRICS" 4)
B_TPS=$(parse_field "$B_METRICS" 5)

# ── Build verdict via Python (all the float comparisons in one place) ─────────
VERDICT_B=$(/usr/bin/python3 - "$A_TPS" "$A_JTOK_GPU" "$B_TPS" "$B_JTOK_GPU" <<'PYEOF'
import sys

def try_float(s):
    try: return float(s)
    except: return None

a_tps = try_float(sys.argv[1])
a_jg  = try_float(sys.argv[2])
b_tps = try_float(sys.argv[3])
b_jg  = try_float(sys.argv[4])

parts = []
warn = []

if b_jg is not None and a_jg is not None and a_jg > 0:
    delta_j = (b_jg - a_jg) / a_jg * 100
    parts.append(f"{delta_j:+.1f}% J/tok")
else:
    parts.append("J/tok: N/A")
    delta_j = None

if b_tps is not None and a_tps is not None and a_tps > 0:
    delta_t = (b_tps - a_tps) / a_tps * 100
    parts.append(f"{delta_t:+.1f}% tps")
else:
    delta_t = None

# Efficiency verdict
if delta_j is not None and delta_t is not None:
    tps_better = delta_t > 0
    j_better   = delta_j < 0
    if j_better and tps_better:
        verdict = "BETTER (faster + cheaper)"
    elif j_better and not tps_better:
        verdict = "CHEAPER energy, slower tps"
    elif not j_better and tps_better:
        verdict = "WARNING: higher t/s but worse J/tok — throughput gained, efficiency regressed"
    else:
        verdict = "WORSE (slower + costlier)"
    parts.append(verdict)

print(" | ".join(parts))
PYEOF
)

# ── Print table ───────────────────────────────────────────────────────────────
printf '\n'
printf '%-14s | %-8s | %-10s | %-10s | %-10s | %-12s | %s\n' \
  "lane" "dec_tps" "J/tok_GPU" "J/tok_pkg" "J/req_GPU" "J/eff-tok" "verdict"
printf '%-14s-+-%-8s-+-%-10s-+-%-10s-+-%-10s-+-%-12s-+-%s\n' \
  "--------------" "--------" "----------" "----------" "----------" "------------" "-------"
printf '%-14s | %-8s | %-10s | %-10s | %-10s | %-12s | %s\n' \
  "command_a" "$A_TPS" "$A_JTOK_GPU" "$A_JTOK_PKG" "$A_JREQ_GPU" "$A_JEFF_GPU" "baseline"
printf '%-14s | %-8s | %-10s | %-10s | %-10s | %-12s | %s\n' \
  "command_b" "$B_TPS" "$B_JTOK_GPU" "$B_JTOK_PKG" "$B_JREQ_GPU" "$B_JEFF_GPU" "$VERDICT_B"

# ── Explicit J/tok detail lines ───────────────────────────────────────────────
printf '\n'
printf 'A: %.10s J/tok_GPU | B: %.10s J/tok_GPU\n' "$A_JTOK_GPU" "$B_JTOK_GPU"

if [[ "$HAS_MACMON" -eq 1 ]]; then
  /usr/bin/python3 - "$A_JTOK_GPU" "$B_JTOK_GPU" <<'PYEOF'
import sys
def tf(s):
    try: return float(s)
    except: return None
a, b = tf(sys.argv[1]), tf(sys.argv[2])
if a is not None and b is not None and a > 0:
    d = (b - a) / a * 100
    sym = "B wins on energy" if d < 0 else "B loses on energy"
    print(f"delta: {d:+.1f}% | {sym}")
PYEOF
fi

printf '\nnote: absolute numbers require clean room (macmon + agent closed).\n'
printf '      Relative A/B delta is contamination-robust.\n'
