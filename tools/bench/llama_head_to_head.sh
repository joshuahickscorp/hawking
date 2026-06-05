#!/usr/bin/env bash
# =============================================================================
# tools/bench/llama_head_to_head.sh
#
# dismantle vs llama.cpp — complete competitive benchmark.
# Covers:
#   1. Single-stream decode (same GGUF, greedy, multiple trials, with energy)
#   2. Aggregate throughput sweep B=1..8: dismantle serve vs llama-server --parallel N
#      (apples-to-apples: both engines serving N concurrent slots)
#
# CLEAN ROOM REQUIRED for absolute tps/J numbers — quit Claude and any heavy
# GPU process first. The preflight check aborts if Claude is running.
#
# USAGE:
#   tools/bench/llama_head_to_head.sh
#
# ENVIRONMENT OVERRIDES:
#   GGUF              path to Q4_K_M GGUF (default: models/qwen2.5-3b-instruct-q4_k_m.gguf)
#   DBIN              dismantle binary (default: ./target/release/dismantle)
#   LLAMA_BIN         llama-completion/llama-cli binary; auto-detected if unset
#   LLAMA_SERVER_BIN  llama-server binary; auto-detected if unset
#   PROFILE           kernel profile JSON (default: profiles/qwen3b-instruct-q4k.m3pro18.json)
#   TOKENS            decode tokens per run (default: 256)
#   TRIALS            single-stream trials per engine (default: 3)
#   PROMPT            override prompt text
#   BATCH_SIZES       space-separated B values for aggregate sweep (default: "1 2 4 8")
#   LLAMA_SS_TPS      reference llama single-stream tps for ratio column (default: auto from trial)
#   SAMPLE_MS         macmon sampling interval ms (default: 200)
#   RUN_TIMEOUT_SEC   per-run wall-time cap (default: 600)
#   SKIP_BATCH        set to 1 to skip the batch sweep (faster)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

GGUF="${GGUF:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
DBIN="${DBIN:-./target/release/dismantle}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
TOKENS="${TOKENS:-256}"
TRIALS="${TRIALS:-3}"
PROMPT="${PROMPT:-Explain how unified memory on Apple Silicon changes the GPU programming model.}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
SAMPLE_MS="${SAMPLE_MS:-200}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-600}"
SKIP_BATCH="${SKIP_BATCH:-0}"
NICE=(nice -n 19 taskpolicy -b)

TMPD="/tmp/lhth_$$"
mkdir -p "$TMPD"
cleanup() {
  rm -rf "$TMPD"
  if [[ -n "${SERVE_PID:-}" ]]; then
    kill "${SERVE_PID:-}" 2>/dev/null || true
    wait "${SERVE_PID:-}" 2>/dev/null || true
  fi
  if [[ -n "${LLAMA_SERVE_PID:-}" ]]; then
    kill "${LLAMA_SERVE_PID:-}" 2>/dev/null || true
    wait "${LLAMA_SERVE_PID:-}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# =============================================================================
# 0. PREFLIGHT
# =============================================================================
echo "=== dismantle vs llama.cpp — competitive bench ==="
echo

fail() { echo "FAIL: $*" >&2; exit 3; }
warn() { echo "WARN: $*"; }

command -v macmon >/dev/null 2>&1 || fail "macmon missing → brew install macmon"

if pgrep -f "Claude.app" >/dev/null 2>&1 || pgrep -xi "claude" >/dev/null 2>&1; then
  fail "Claude session is running — absolute tps/J inflate ~4-5×. Quit it and re-run."
fi

[[ -x "$DBIN" ]] || fail "$DBIN not built — cargo build --release --workspace"
[[ -f "$GGUF"  ]] || fail "GGUF not found: $GGUF"
[[ -f "$PROFILE" ]] || fail "kernel profile not found: $PROFILE"

resolve_exe() {
  local cand="$1" resolved
  if [[ "$cand" == */* ]]; then
    [[ -x "$cand" ]] && printf '%s\n' "$cand"
    return
  fi
  resolved=$(command -v "$cand" 2>/dev/null || true)
  [[ -n "$resolved" && -x "$resolved" ]] && printf '%s\n' "$resolved"
}

# Auto-detect llama binary
if [[ -z "${LLAMA_BIN:-}" ]]; then
  for cand in llama-completion llama-cli llama \
               /usr/local/bin/llama-completion /usr/local/bin/llama-cli \
               /opt/homebrew/bin/llama-completion /opt/homebrew/bin/llama-cli /opt/homebrew/bin/llama; do
    LLAMA_BIN=$(resolve_exe "$cand") && [[ -n "$LLAMA_BIN" ]] && break
  done
else
  LLAMA_BIN=$(resolve_exe "$LLAMA_BIN")
fi
[[ -x "${LLAMA_BIN:-}" ]] || fail "llama CLI not found. Set LLAMA_BIN= or install llama.cpp (brew install llama.cpp)."

LLAMA_MODE_ARGS=()
case "$(basename "$LLAMA_BIN")" in
  llama-cli|llama)
    # Current llama-cli is chat-first and rejects -no-cnv; single-turn keeps EOF
    # from becoming an unbounded interactive transcript in redirected logs.
    LLAMA_MODE_ARGS=(--single-turn)
    ;;
  *)
    LLAMA_MODE_ARGS=(-no-cnv)
    ;;
esac

# Auto-detect llama-server binary (for parallel serving sweep)
LLAMA_SERVE_PID=""
if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
  for cand in llama-server \
               /usr/local/bin/llama-server \
               /opt/homebrew/bin/llama-server; do
    LLAMA_SERVER_BIN=$(resolve_exe "$cand") && [[ -n "$LLAMA_SERVER_BIN" ]] && break
  done
else
  LLAMA_SERVER_BIN=$(resolve_exe "$LLAMA_SERVER_BIN")
fi
if [[ -z "${LLAMA_SERVER_BIN:-}" || ! -x "${LLAMA_SERVER_BIN:-}" ]]; then
  warn "llama-server not found — batch sweep will be dismantle-only. Set LLAMA_SERVER_BIN= or: brew install llama.cpp"
  LLAMA_SERVER_BIN=""
fi

profile_shader_hash=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("shader_hash",""))' "$PROFILE" 2>/dev/null || true)
current_shader_hash=$("$DBIN" shader-hash 2>/dev/null || true)
if [[ -n "$profile_shader_hash" && -n "$current_shader_hash" && "$profile_shader_hash" != "$current_shader_hash" ]]; then
  warn "kernel profile shader hash mismatch (profile=$profile_shader_hash current=$current_shader_hash)"
  REFRESHED_PROFILE="$TMPD/kernel_profile.json"
  if "$DBIN" autotune --weights "$GGUF" --out "$REFRESHED_PROFILE" --log "$TMPD/kernel_profile.jsonl" >/dev/null 2>&1; then
    PROFILE="$REFRESHED_PROFILE"
    warn "using refreshed temporary kernel profile: $PROFILE"
  else
    fail "kernel profile is stale and autotune failed. Re-run: $DBIN autotune --weights \"$GGUF\" --out \"$PROFILE\""
  fi
fi

echo "  dismantle      : $DBIN"
echo "  llama CLI      : $LLAMA_BIN"
echo "  llama-server   : ${LLAMA_SERVER_BIN:-NOT FOUND (batch comparison disabled)}"
echo "  model          : $GGUF"
echo "  tokens         : $TOKENS   trials: $TRIALS"
echo

# =============================================================================
# 1. DIAGNOSTICS COLLECTION
# =============================================================================
collect_diagnostics() {
  # --- System ---
  CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model 2>/dev/null || echo "unknown")
  RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
  RAM_GB=$(awk -v b="$RAM_BYTES" 'BEGIN{printf "%d", b/1073741824}')
  MACOS=$(sw_vers -productVersion 2>/dev/null || echo "unknown")

  # --- Build ---
  GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  LOCAL_MODS=$(git status --short 2>/dev/null | wc -l | tr -d ' ')
  BUILD_MTIME=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$DBIN" 2>/dev/null || echo "unknown")

  # --- Model ---
  GGUF_SIZE=$(stat -f "%z" "$GGUF" 2>/dev/null || stat -c "%s" "$GGUF" 2>/dev/null || echo "?")
  GGUF_GB=$(awk -v s="$GGUF_SIZE" 'BEGIN{printf "%.2f", s/1073741824}')
  GGUF_HASH="?"
  if command -v b3sum >/dev/null 2>&1; then
    GGUF_HASH=$(b3sum --no-names "$GGUF" 2>/dev/null | cut -c1-12 || echo "?")
  elif command -v md5 >/dev/null 2>&1; then
    GGUF_HASH=$(md5 -q "$GGUF" 2>/dev/null | cut -c1-12 || echo "?")
  fi

  # --- Kernel profile ---
  PROF_NAME="(none)"
  PROF_GEMM="?"
  if [[ -f "$PROFILE" ]]; then
    PROF_NAME=$(python3 -c "import json,sys; d=json.load(open('$PROFILE')); print(d.get('profile_name','?'))" 2>/dev/null || echo "?")
    PROF_GEMM=$(python3 -c "import json,sys; d=json.load(open('$PROFILE')); s=d.get('selected',{}); gs=s.get('gemm_q4_k_schedule_per_shape',{}); v=list(set(gs.values())); print('+'.join(v))" 2>/dev/null || echo "?")
  fi

  # --- llama version ---
  LLAMA_VER=$("$LLAMA_BIN" --version 2>&1 | head -1 | grep -oE '[Vv]?[0-9]+[^ ]*' | head -1 || echo "unknown")

  # --- macmon version ---
  MACMON_VER=$(macmon --version 2>/dev/null || echo "unknown")
}
collect_diagnostics

# =============================================================================
# 2. MACMON SAMPLER
# =============================================================================
sample_macmon() { # $1=pkg_file $2=gpu_file
  macmon pipe -i "$SAMPLE_MS" 2>/dev/null | while IFS= read -r line; do
    p=$(printf '%s' "$line" | sed -n 's/.*"all_power"[: ]*\([0-9.]*\).*/\1/p')
    g=$(printf '%s' "$line" | sed -n 's/.*"gpu_power"[: ]*\([0-9.]*\).*/\1/p')
    [[ -n "$p" ]] && printf '%s\n' "$p" >> "$1"
    [[ -n "$g" ]] && printf '%s\n' "$g" >> "$2"
  done
}

avg_file() { awk '{s+=$1;n++} END{if(n>0) printf "%.4f", s/n; else printf "0"}' "$1" 2>/dev/null; }

# =============================================================================
# 3. TIMED RUN (single trial)
# =============================================================================
run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    perl -e 'alarm shift; exec { $ARGV[0] } @ARGV or die "exec: $!"' "$secs" "$@"
  fi
}

# tps parsers
parse_tps_dismantle() { grep -oE 'dec_tps=[0-9.]+' "$1" | grep -oE '[0-9.]+' | tail -1; }
parse_pfx_dismantle() { grep -oE 'pfx_tps=[0-9.]+' "$1" | grep -oE '[0-9.]+' | tail -1; }

parse_tps_llama() {
  # --perf format: "eval time = ... tokens per second" (NOT "prompt eval")
  local t
  t=$(tail -200 "$1" | grep -iE 'eval time.*tokens per second' | grep -vi 'prompt eval' | sed -nE 's/.*[[:space:],(]([0-9]+(\.[0-9]+)?|inf)[[:space:]]+tokens per second.*/\1/p' | tail -1)
  [[ -n "$t" ]] && { echo "$t"; return; }
  # fallback: "Generation: X t/s"
  tail -200 "$1" | grep -oiE 'Generation:[^0-9]*[0-9.]+ *t/s' | grep -oE '[0-9.]+' | tail -1
}
parse_pfx_llama() {
  tail -200 "$1" | grep -iE 'prompt eval time.*tokens per second' | sed -nE 's/.*[[:space:],(]([0-9]+(\.[0-9]+)?|inf)[[:space:]]+tokens per second.*/\1/p' | tail -1
}

# run_engine: $1=label $2=tps_parse_fn $3=pfx_parse_fn [rest=command]
# Writes results to $TMPD/${label}_results.txt: "tps pfx_tps avg_gpu avg_pkg wall"
run_engine_trials() {
  local label="$1" tps_fn="$2" pfx_fn="$3"; shift 3
  local tps_sum=0 pfx_sum=0 gpu_sum=0 pkg_sum=0 wall_sum=0
  local tps_min=999999 tps_max=0
  local n_ok=0 pfx_ok=0

  printf '  [%s] running %s trial(s)...\n' "$label" "$TRIALS"
  for i in $(seq 1 "$TRIALS"); do
    local out="$TMPD/${label}_trial${i}.log"
    local pkgf="$TMPD/${label}_t${i}_pkg" gpuf="$TMPD/${label}_t${i}_gpu"
    : > "$pkgf"; : > "$gpuf"
    sample_macmon "$pkgf" "$gpuf" & local smp=$!
    local t0 t1
    t0=$(date +%s.%N)
    run_with_timeout "$RUN_TIMEOUT_SEC" "$@" </dev/null > "$out" 2>&1
    local rc=$?
    t1=$(date +%s.%N)
    pkill -P "$smp" 2>/dev/null || true; kill "$smp" 2>/dev/null || true
    wait "$smp" 2>/dev/null || true

    local tps pfx gpu_avg pkg_avg wall
    tps=$("$tps_fn" "$out") pfx=$("$pfx_fn" "$out")
    gpu_avg=$(avg_file "$gpuf") pkg_avg=$(avg_file "$pkgf")
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')

    if [[ -z "$tps" || "$rc" -eq 124 || "$rc" -eq 142 ]]; then
      warn "trial $i failed (rc=$rc, tps='${tps:-?}') — see $out"
      continue
    fi
    printf '    trial %s: dec_tps=%-7s pfx_tps=%-7s gpu=%-6sW wall=%ss\n' \
      "$i" "$tps" "${pfx:-n/a}" "$gpu_avg" "$wall"
    tps_sum=$(awk -v a="$tps_sum" -v b="$tps" 'BEGIN{printf "%.4f", a+b}')
    if [[ -n "$pfx" ]]; then
      pfx_sum=$(awk -v a="$pfx_sum" -v b="$pfx" 'BEGIN{printf "%.4f", a+b}')
      pfx_ok=$(( pfx_ok + 1 ))
    fi
    gpu_sum=$(awk -v a="$gpu_sum" -v b="$gpu_avg" 'BEGIN{printf "%.4f", a+b}')
    pkg_sum=$(awk -v a="$pkg_sum" -v b="$pkg_avg" 'BEGIN{printf "%.4f", a+b}')
    wall_sum=$(awk -v a="$wall_sum" -v b="$wall" 'BEGIN{printf "%.2f", a+b}')
    tps_min=$(awk -v a="$tps_min" -v b="$tps" 'BEGIN{printf "%.2f", (b<a)?b:a}')
    tps_max=$(awk -v a="$tps_max" -v b="$tps" 'BEGIN{printf "%.2f", (b>a)?b:a}')
    n_ok=$(( n_ok + 1 ))
  done

  if [[ "$n_ok" -eq 0 ]]; then
    echo "ERROR: all $TRIALS trials failed for $label." >&2
    printf 'FAIL 0 0 0 0 0 0 0\n' > "$TMPD/${label}_results.txt"
    return 1
  fi

  local tps_mean pfx_mean gpu_mean pkg_mean wall_mean
  tps_mean=$(awk -v s="$tps_sum" -v n="$n_ok" 'BEGIN{printf "%.2f", s/n}')
  if [[ "$pfx_ok" -gt 0 ]]; then
    pfx_mean=$(awk -v s="$pfx_sum" -v n="$pfx_ok" 'BEGIN{printf "%.2f", s/n}')
  else
    pfx_mean="n/a"
  fi
  gpu_mean=$(awk -v s="$gpu_sum" -v n="$n_ok" 'BEGIN{printf "%.4f", s/n}')
  pkg_mean=$(awk -v s="$pkg_sum" -v n="$n_ok" 'BEGIN{printf "%.4f", s/n}')
  wall_mean=$(awk -v s="$wall_sum" -v n="$n_ok" 'BEGIN{printf "%.2f", s/n}')
  printf '%s %s %s %s %s %s %s\n' \
    "$tps_mean" "$pfx_mean" "$gpu_mean" "$pkg_mean" "$wall_mean" "$tps_min" "$tps_max" \
    > "$TMPD/${label}_results.txt"
}

# =============================================================================
# 4. SECTION 1 — SINGLE-STREAM HEAD-TO-HEAD
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SECTION 1: Single-stream decode  (N=$TOKENS, greedy temp=0, $TRIALS trials each)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

run_engine_trials dismantle parse_tps_dismantle parse_pfx_dismantle \
  "${NICE[@]}" "$DBIN" generate \
    --weights "$GGUF" --kernel-profile "$PROFILE" \
    --prompt "$PROMPT" --max-new-tokens "$TOKENS" \
    --temperature 0 --seed 0 --max-stall-ms 30000
echo

run_engine_trials llama parse_tps_llama parse_pfx_llama \
  "${NICE[@]}" "$LLAMA_BIN" \
    -m "$GGUF" -p "$PROMPT" -n "$TOKENS" \
    --temp 0 --seed 0 -ngl 99 "${LLAMA_MODE_ARGS[@]}" \
    --no-display-prompt --no-warmup --perf
echo

# read results
read -r D_TPS D_PFX D_GPU D_PKG D_WALL D_MIN D_MAX < "$TMPD/dismantle_results.txt" || true
read -r L_TPS L_PFX L_GPU L_PKG L_WALL L_MIN L_MAX < "$TMPD/llama_results.txt"     || true

# ratios
ratio_tps=$(awk -v d="${D_TPS:-0}" -v l="${L_TPS:-0}" 'BEGIN{if(l>0&&d>0) printf "%.3f", d/l; else print "?"}')
ratio_gpu=$(awk -v d="${D_GPU:-0}" -v l="${L_GPU:-0}" 'BEGIN{if(d>0&&l>0) printf "%.3f", l/d; else print "?"}')  # higher=better (llama/dis = efficiency ratio)
ratio_pkg=$(awk -v d="${D_PKG:-0}" -v l="${L_PKG:-0}" 'BEGIN{if(d>0&&l>0) printf "%.3f", l/d; else print "?"}')

D_JTOK_GPU=$(awk -v g="${D_GPU:-0}" -v t="${D_TPS:-0}" 'BEGIN{if(t>0&&g>0) printf "%.4f", g/t; else print "?"}')
D_JTOK_PKG=$(awk -v g="${D_PKG:-0}" -v t="${D_TPS:-0}" 'BEGIN{if(t>0&&g>0) printf "%.4f", g/t; else print "?"}')
L_JTOK_GPU=$(awk -v g="${L_GPU:-0}" -v t="${L_TPS:-0}" 'BEGIN{if(t>0&&g>0) printf "%.4f", g/t; else print "?"}')
L_JTOK_PKG=$(awk -v g="${L_PKG:-0}" -v t="${L_TPS:-0}" 'BEGIN{if(t>0&&g>0) printf "%.4f", g/t; else print "?"}')

LLAMA_SS_TPS="${LLAMA_SS_TPS:-${L_TPS:-0}}"

printf '\n%-12s %9s %9s  %10s %10s  %8s\n' \
  engine dec_tps pfx_tps J/tok_GPU J/tok_pkg wall_s
printf '%s\n' "─────────────────────────────────────────────────────────────────"
printf '%-12s %9s %9s  %10s %10s  %8s\n' \
  "dismantle"  "${D_TPS:-?}" "${D_PFX:-?}" "$D_JTOK_GPU" "$D_JTOK_PKG" "${D_WALL:-?}"
printf '%-12s %9s %9s  %10s %10s  %8s\n' \
  "llama.cpp"  "${L_TPS:-?}" "${L_PFX:-?}" "$L_JTOK_GPU" "$L_JTOK_PKG" "${L_WALL:-?}"
printf '%s\n' "─────────────────────────────────────────────────────────────────"
printf '%-12s %9s %9s  %10s %10s\n' \
  "ratio(d/l)" "${ratio_tps}×" "" "${ratio_gpu}× eff" "${ratio_pkg}× eff"
echo "(J/tok efficiency ratio = llama/dismantle; >1.0 means dismantle uses less energy)"
echo "spread: dismantle ${D_MIN}–${D_MAX} t/s over $TRIALS trials"
echo

# =============================================================================
# 5. SECTION 2 — AGGREGATE THROUGHPUT SWEEP (dismantle serve vs llama-server)
# =============================================================================
if [[ "$SKIP_BATCH" != "1" ]]; then
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SECTION 2: Aggregate throughput sweep  (B concurrent slots, N=$TOKENS/slot)"
echo "  dismantle serve vs llama-server --parallel B  [apples-to-apples]"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

MAX_B=1
for b in $BATCH_SIZES; do [[ "$b" -gt "$MAX_B" ]] && MAX_B="$b"; done

# ── Shared helpers ──────────────────────────────────────────────────────────
fire_sse_to() { # $1=serve_url  $2=logfile
  local url="$1" logf="$2"
  local pj; pj=$(printf '%s' "$PROMPT" | python3 -c "import sys,json;print(json.dumps(sys.stdin.read()))" 2>/dev/null || printf '"%s"' "$PROMPT")
  : > "$logf"
  curl --no-buffer --silent --max-time "$RUN_TIMEOUT_SEC" \
    -X POST "${url}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":${pj},\"max_tokens\":${TOKENS},\"stream\":true,\"temperature\":0,\"seed\":0}" \
    > "$logf" 2>&1 &
  local cpid=$!
  local ss; ss=$(date +%s)
  while :; do
    grep -q '^data: \[DONE\]' "$logf" 2>/dev/null && break
    local n; n=$(awk '/^data: / && $0 != "data: [DONE]"{n++} END{print n+0}' "$logf" 2>/dev/null)
    [[ "$n" -ge "$TOKENS" ]] && break
    local st; st=$(ps -o stat= -p "$cpid" 2>/dev/null || true)
    [[ -z "$st" || "$st" == *Z* ]] && break
    [[ $(( $(date +%s) - ss )) -ge "$RUN_TIMEOUT_SEC" ]] && break
    sleep 0.05
  done
  ps -p "$cpid" >/dev/null 2>&1 && kill "$cpid" 2>/dev/null || true
  wait "$cpid" 2>/dev/null || true
  local seen done
  seen=$(count_sse_toks "$logf")
  done=0; grep -q '^data: \[DONE\]' "$logf" 2>/dev/null && done=1
  [[ "$seen" -gt 0 && ( "$seen" -ge "$TOKENS" || "$done" -eq 1 ) ]]
}

count_sse_toks() { awk '/^data: / && $0 != "data: [DONE]"{n++} END{print n+0}' "$1" 2>/dev/null || echo 0; }

# run_sweep SERVE_URL LABEL PREFIX
# Fires BATCH_SIZES concurrent sweeps against an already-running server.
# Writes results into ${PREFIX}_results array (printed live) and
# sets ${PREFIX}_B<N>_agg for each B for later comparison.
run_sweep() {
  local url="$1" label="$2" pfx="$3"
  printf '  warmup (%s)...' "$label"
  local warm_log="$TMPD/${pfx}_warmup.log"
  if fire_sse_to "$url" "$warm_log"; then
    printf ' done (%s toks)\n' "$(count_sse_toks "$warm_log")"
  else
    printf ' failed (%s toks)\n' "$(count_sse_toks "$warm_log")"
    warn "warmup for $label returned no token events; last SSE lines:"
    tail -8 "$warm_log" >&2 || true
  fi

  for B in $BATCH_SIZES; do
    local logs=() pids=() total=0 slot t0 t1 wall agg per
    for slot in $(seq 0 $(( B - 1 ))); do logs+=("$TMPD/${pfx}_B${B}_s${slot}.log"); done
    t0=$(date +%s.%N)
    for slot in $(seq 0 $(( B - 1 ))); do fire_sse_to "$url" "${logs[$slot]}" & pids+=("$!"); done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    t1=$(date +%s.%N)
    for slot in $(seq 0 $(( B - 1 ))); do
      t=$(count_sse_toks "${logs[$slot]}"); total=$(( total + t ))
      if [[ "$t" -eq 0 ]]; then
        warn "$label B=$B slot=$slot returned zero token events; see ${logs[$slot]}"
      fi
    done
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
    agg=$(awk -v tok="$total" -v w="$wall" 'BEGIN{if(w>0) printf "%.2f", tok/w; else print "?"}')
    per=$(awk -v a="$agg" -v b="$B" 'BEGIN{if(b>0&&a>0) printf "%.2f", a/b; else print "?"}')
    eval "${pfx}_B${B}_agg=${agg}"
    eval "${pfx}_B${B}_per=${per}"
    printf '  B=%-2s  agg=%s t/s  per_slot=%s t/s\n' "$B" "$agg" "$per"
  done
}

# ── dismantle serve ─────────────────────────────────────────────────────────
SERVE_PID=""
SERVE_URL="http://127.0.0.1:${SERVE_PORT:-8181}"
SERVE_CMD=("${NICE[@]}" "$DBIN" serve --weights "$GGUF" --max-batch-size "$MAX_B")
[[ -f "$PROFILE" ]] && SERVE_CMD+=(--kernel-profile "$PROFILE")

printf '\n[dismantle serve]  max-batch-size=%s\n' "$MAX_B"
"${SERVE_CMD[@]}" --addr "127.0.0.1:${SERVE_PORT:-8181}" > "$TMPD/dis_serve.log" 2>&1 &
SERVE_PID=$!
WAIT=0
until curl -sf "${SERVE_URL}/healthz" >/dev/null 2>&1; do
  sleep 1; WAIT=$(( WAIT + 1 ))
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "  FAIL: dismantle serve died"; tail -6 "$TMPD/dis_serve.log" >&2; SERVE_PID=""; break; }
  [[ "$WAIT" -gt 90 ]] && { echo "  FAIL: timeout"; kill "$SERVE_PID" 2>/dev/null; SERVE_PID=""; break; }
done
[[ -n "$SERVE_PID" ]] && printf '  ready in %ss\n' "$WAIT"

DIS_SWEEP_OK=0
if [[ -n "$SERVE_PID" ]]; then
  run_sweep "$SERVE_URL" "dismantle" "dis"
  DIS_SWEEP_OK=1
  kill "$SERVE_PID" 2>/dev/null || true
  wait "$SERVE_PID" 2>/dev/null || true
  SERVE_PID=""
  sleep 2  # let GPU drain before starting llama-server
fi

# ── llama-server parallel ────────────────────────────────────────────────────
LLAMA_SERVE_PID=""
LLAMA_SERVE_OK=0
if [[ -n "${LLAMA_SERVER_BIN:-}" ]]; then
  LLAMA_URL="http://127.0.0.1:${LLAMA_SERVER_PORT:-8182}"
  printf '\n[llama-server]  --parallel %s\n' "$MAX_B"
  # llama-server restarts cleanly per-B for a fair comparison; but starting
  # once with --parallel MAX_B is correct since we fire B<=MAX_B requests.
  "${NICE[@]}" "$LLAMA_SERVER_BIN" \
    -m "$GGUF" -ngl 99 --parallel "$MAX_B" \
    --host 127.0.0.1 --port "${LLAMA_SERVER_PORT:-8182}" \
    --log-disable \
    > "$TMPD/llama_serve.log" 2>&1 &
  LLAMA_SERVE_PID=$!
  WAIT=0
  # llama-server health: /health (returns {"status":"ok"}) or /v1/models
  until curl -sf "${LLAMA_URL}/health" >/dev/null 2>&1 || \
        curl -sf "${LLAMA_URL}/v1/models" >/dev/null 2>&1; do
    sleep 1; WAIT=$(( WAIT + 1 ))
    kill -0 "$LLAMA_SERVE_PID" 2>/dev/null || { echo "  FAIL: llama-server died"; tail -6 "$TMPD/llama_serve.log" >&2; LLAMA_SERVE_PID=""; break; }
    [[ "$WAIT" -gt 120 ]] && { echo "  FAIL: timeout"; kill "$LLAMA_SERVE_PID" 2>/dev/null; LLAMA_SERVE_PID=""; break; }
  done
  [[ -n "$LLAMA_SERVE_PID" ]] && printf '  ready in %ss\n' "$WAIT"

  if [[ -n "$LLAMA_SERVE_PID" ]]; then
    run_sweep "$LLAMA_URL" "llama-server" "ls"
    LLAMA_SERVE_OK=1
    kill "$LLAMA_SERVE_PID" 2>/dev/null || true
    wait "$LLAMA_SERVE_PID" 2>/dev/null || true
    LLAMA_SERVE_PID=""
  fi
fi

# ── head-to-head comparison table ───────────────────────────────────────────
if [[ "$DIS_SWEEP_OK" -eq 1 || "$LLAMA_SERVE_OK" -eq 1 ]]; then
  echo
  echo "── Aggregate head-to-head ──────────────────────────────────────────────"
  if [[ "$DIS_SWEEP_OK" -eq 1 && "$LLAMA_SERVE_OK" -eq 1 ]]; then
    printf '%-4s %14s %14s %10s\n' B "dismantle(t/s)" "llama-srv(t/s)" "dis/llama"
    printf '%s\n' "──────────────────────────────────────────────────────────────"
    for B in $BATCH_SIZES; do
      eval "da=\${dis_B${B}_agg:-?}"
      eval "la=\${ls_B${B}_agg:-?}"
      ratio=$(awk -v d="$da" -v l="$la" 'BEGIN{if(d>0&&l>0) printf "%.3f×", d/l; else print "?"}')
      kernel=""; [[ "$B" -eq 1 ]] && kernel="(gemv)" || { [[ "$B" -le 4 ]] && kernel="(v4r)" || kernel="(v3w)"; }
      printf '%-4s %14s %14s %10s  %s\n' "$B" "$da" "$la" "$ratio" "$kernel"
    done
  elif [[ "$DIS_SWEEP_OK" -eq 1 ]]; then
    printf '%-4s %14s %14s\n' B "dismantle(t/s)" "llama-srv"
    printf '%s\n' "────────────────────────────────────────"
    for B in $BATCH_SIZES; do
      eval "da=\${dis_B${B}_agg:-?}"
      kernel=""; [[ "$B" -eq 1 ]] && kernel="(gemv)" || { [[ "$B" -le 4 ]] && kernel="(v4r)" || kernel="(v3w)"; }
      printf '%-4s %14s %14s  %s\n' "$B" "$da" "N/A" "$kernel"
    done
  fi
  echo
  echo "  kernel note: B=1 → single-vector GEMV fallback; B=2..4 → v4r_predec; B>4 → v3w_predec"
  [[ "$LLAMA_SERVE_OK" -eq 0 && -z "${LLAMA_SERVER_BIN:-}" ]] && \
    echo "  llama-server not found — set LLAMA_SERVER_BIN= or brew install llama.cpp"
fi

fi  # SKIP_BATCH

# =============================================================================
# 6. SUMMARY (shareable)
# =============================================================================
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " SUMMARY — $(date '+%Y-%m-%d %H:%M')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf ' Device  : %s  RAM: %s GB  macOS: %s\n' "$CHIP" "$RAM_GB" "$MACOS"
printf ' Model   : %s  (%s GB, b3:%s)\n' "$(basename "$GGUF")" "$GGUF_GB" "$GGUF_HASH"
printf ' Profile : %s  GEMM: %s\n' "$PROF_NAME" "$PROF_GEMM"
printf ' GitSHA  : %s  local_mods: %s\n' "$GIT_SHA" "$LOCAL_MODS"
printf ' dismantle binary built: %s\n' "$BUILD_MTIME"
printf ' llama   : %s  ver: %s\n' "$(basename "$LLAMA_BIN")" "$LLAMA_VER"
echo
echo " ── Single-stream decode (N=$TOKENS, greedy, $TRIALS trials) ──────────"
printf ' %-13s  dec_tps: %-8s  pfx_tps: %-8s  J/tok(GPU): %-8s  J/tok(pkg): %s\n' \
  "dismantle"  "${D_TPS:-?}" "${D_PFX:-?}" "$D_JTOK_GPU" "$D_JTOK_PKG"
printf ' %-13s  dec_tps: %-8s  pfx_tps: %-8s  J/tok(GPU): %-8s  J/tok(pkg): %s\n' \
  "llama.cpp"  "${L_TPS:-?}" "${L_PFX:-?}" "$L_JTOK_GPU" "$L_JTOK_PKG"
printf ' %-13s  dec_tps: %s×  (energy efficiency: GPU %s× / pkg %s×)\n' \
  "ratio(d÷l)"  "${ratio_tps}" "${ratio_gpu}" "${ratio_pkg}"
echo " spread: dismantle ${D_MIN}–${D_MAX} t/s across $TRIALS trials (clean-room required for valid abs)"
echo
if [[ "$SKIP_BATCH" != "1" && ("${DIS_SWEEP_OK:-0}" -eq 1 || "${LLAMA_SERVE_OK:-0}" -eq 1) ]]; then
echo " ── Aggregate head-to-head (B concurrent slots, N=$TOKENS/slot) ────"
if [[ "${DIS_SWEEP_OK:-0}" -eq 1 && "${LLAMA_SERVE_OK:-0}" -eq 1 ]]; then
  printf ' %-4s %14s %14s %10s\n' B "dismantle(t/s)" "llama-srv(t/s)" "dis/llama"
  printf ' %s\n' "──────────────────────────────────────────────────────"
  for B in $BATCH_SIZES; do
    eval "da=\${dis_B${B}_agg:-?}"
    eval "la=\${ls_B${B}_agg:-?}"
    ratio=$(awk -v d="$da" -v l="$la" 'BEGIN{if(d>0&&l>0) printf "%.3f×", d/l; else print "?"}')
    printf ' %-4s %14s %14s %10s\n' "$B" "$da" "$la" "$ratio"
  done
else
  printf ' %-4s %14s\n' B "dismantle(t/s)"
  for B in $BATCH_SIZES; do
    eval "da=\${dis_B${B}_agg:-?}"
    printf ' %-4s %14s\n' "$B" "$da"
  done
  echo " (llama-server not available for comparison)"
fi
echo " kernel: B=1 → single-vector GEMV fallback; B=2..4 → v4r_predec; B>4 → v3w_predec"
echo
fi
echo " ── Notes ─────────────────────────────────────────────────────────"
echo " • Single-stream: dismantle+llama CLI run same GGUF, greedy temp=0, seed=0."
echo " • Aggregate: dismantle serve vs llama-server --parallel B — same B concurrent"
echo "   requests fired simultaneously to each server. Apples-to-apples serving."
echo " • J/tok = avg_power_W / dec_tps (decode-dominated at N=$TOKENS, short prompt)."
echo " • Clean room (Claude quit) required for valid absolute tps and J/tok numbers."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
