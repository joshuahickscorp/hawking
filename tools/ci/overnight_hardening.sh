#!/usr/bin/env bash
# tools/ci/overnight_hardening.sh — unattended production-hardening runner.
#
# Purpose: keep the Hawking campaign moving overnight without relying on a chat
# transcript. The script is intentionally non-destructive: it does not stage,
# commit, reset, delete, rewrite history, or modify source files. It records
# exact commands, git state, test output, and benchmark output under a timestamped
# report directory so the next agent can resume from evidence.
#
# Defaults are conservative. Override via env:
#   RUN_LIB_TESTS=0          skip `cargo test -p hawking-core --lib`
#   RUN_PARITY=0             skip representative parity tests
#   RUN_CARGO_CHECK=0        skip `cargo check`
#   RUN_PREFLIGHT=0          skip tools/ci/preflight.sh
#   RUN_GPU=0                skip all model/bench jobs
#   RUN_SERVE_SMOKE=0        skip SSM serve smoke after SSM benches
#   RUN_MAMBA_SERVE_SMOKE=1  also serve-smoke mamba2 (RWKV is default)
#   FULL_PREFLIGHT=1         run full tools/ci/preflight.sh instead of FAST=1
#   GPU_WAIT=0              do not wait for existing hawking generate/serve jobs
#   MAX_GPU_WAIT_SECS=7200   max wait for a free GPU/model slot
#   TRIALS=3 TOK=96          benchmark trial count / max-new-tokens
set -u

REPO="${REPO:-$HOME/Downloads/hawking}"
cd "$REPO" || exit 2

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-reports/overnight/$STAMP}"
mkdir -p "$OUT"

COMMAND_LOG="$OUT/commands.log"
SUMMARY="$OUT/summary.md"
FAIL=0

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$COMMAND_LOG"
}

run() {
  local name="$1"; shift
  local safe_name
  safe_name="$(printf '%s' "$name" | tr -cs 'A-Za-z0-9_.-' '_' | sed 's/^_//;s/_$//')"
  local logfile="$OUT/${safe_name}.log"
  log "BEGIN $name"
  log "+ $*"
  if "$@" >"$logfile" 2>&1; then
    log "PASS  $name -> $logfile"
  else
    local rc=$?
    log "FAIL  $name rc=$rc -> $logfile"
    FAIL=1
  fi
}

run_shell() {
  local name="$1"; shift
  local script="$*"
  local safe_name
  safe_name="$(printf '%s' "$name" | tr -cs 'A-Za-z0-9_.-' '_' | sed 's/^_//;s/_$//')"
  local logfile="$OUT/${safe_name}.log"
  log "BEGIN $name"
  log "+ bash -lc ${script}"
  if bash -lc "$script" >"$logfile" 2>&1; then
    log "PASS  $name -> $logfile"
  else
    local rc=$?
    log "FAIL  $name rc=$rc -> $logfile"
    FAIL=1
  fi
}

gpu_busy() {
  ps ax -o pid= -o command= |
    awk -v self="$$" '
      $1 != self && /hawking (generate|serve)/ && !/overnight_hardening/ { found=1 }
      END { exit found ? 0 : 1 }
    '
}

wait_gpu_slot() {
  [ "${GPU_WAIT:-1}" = "1" ] || return 0
  local waited=0
  local max="${MAX_GPU_WAIT_SECS:-7200}"
  while gpu_busy; do
    if [ "$waited" -ge "$max" ]; then
      log "GPU wait timed out after ${waited}s; continuing anyway"
      return 0
    fi
    log "GPU/model job already active; waiting 60s before next GPU step"
    sleep 60
    waited=$((waited + 60))
  done
}

make_prompts() {
  mkdir -p "$OUT/prompts"
  python3 - "$OUT/prompts/short.txt" "$OUT/prompts/ctx2k.txt" "$OUT/prompts/ctx8k.txt" <<'PY'
import sys
p_short, p2, p8 = sys.argv[1], sys.argv[2], sys.argv[3]
base = (
    "The memory bandwidth of a GPU fundamentally limits decode speed because "
    "each token reads model weights, and at long context the KV cache adds "
    "more per-token traffic. "
)
open(p_short, "w").write(
    "Explain how a binary search tree works, including insert, search, and delete."
)
open(p2, "w").write(base * 42 + "Summarize the key constraint in one sentence.")
open(p8, "w").write(base * 150 + "Summarize the key constraint in one sentence.")
PY
}

bench_model_tps() {
  local label="$1" model="$2" prompt="$3" tok="${4:-64}" trials="${5:-3}"
  local logfile="$OUT/${label}.log"
  local values="$OUT/${label}_tps_values.txt"
  : >"$values"
  log "BEGIN bench $label model=$model trials=$trials tok=$tok"
  if {
    echo "label=$label"
    echo "model=$model"
    echo "prompt_file=$prompt"
    echo "tok=$tok trials=$trials"
    for i in $(seq 1 "$trials"); do
      local raw="$OUT/${label}_trial_${i}.raw"
      local rc=0
      local tps=""
      env HAWKING_QWEN_USER_DRAFT=0 ./target/release/hawking generate \
        --weights "$model" \
        --prompt "$(cat "$prompt")" \
        --max-new-tokens "$tok" \
        --temperature 0 \
        --seed 5 >"$raw" 2>&1 || rc=$?
      tps="$(grep -oE 'dec_tps=[0-9.]+' "$raw" | tail -1 | cut -d= -f2 || true)"
      if [ -n "$tps" ]; then
        echo "$tps" >>"$values"
        echo "trial_${i}_dec_tps=$tps raw=$raw"
      else
        echo "trial_${i}_dec_tps=ERR rc=$rc raw=$raw"
      fi
    done
    if [ -s "$values" ]; then
      sort -n "$values" | awk '{a[NR]=$1} END{ print "median_dec_tps=" a[int((NR+1)/2)] }'
    else
      echo "median_dec_tps=ERR"
      false
    fi
  } >"$logfile" 2>&1; then
    log "PASS  bench $label -> $logfile"
  else
    local rc=$?
    log "FAIL  bench $label rc=$rc -> $logfile"
    FAIL=1
  fi
}

log "Hawking overnight hardening start"
log "repo=$REPO out=$OUT"

{
  echo "# Overnight Hardening Run — $STAMP"
  echo
  echo "- Repo: \`$REPO\`"
  echo "- Output: \`$OUT\`"
  echo "- Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "## Initial Git State"
  echo '```'
  git status --short
  echo '```'
} >"$SUMMARY"

run "git_status" git status --short
run "git_diff_stat" git diff --stat

if [ "${RUN_CARGO_CHECK:-1}" = "1" ]; then
  run "cargo_check" cargo check
fi

if [ "${RUN_LIB_TESTS:-1}" = "1" ]; then
  run "lib_tests" cargo test -p hawking-core --lib
fi

if [ "${RUN_PARITY:-1}" = "1" ]; then
  run "perchannel_int4kv_parity" cargo test -p hawking-core --test mha_decode_perchannel_int4kv_parity
  run "q6k_2r_parity" cargo test -p hawking-core --test q6k_swiglu_2r_parity
  run "q6k_4r_parity" cargo test -p hawking-core --test q6k_swiglu_4r_parity
fi

if [ "${RUN_PREFLIGHT:-1}" = "1" ] && [ -x tools/ci/preflight.sh ]; then
  if [ "${FULL_PREFLIGHT:-0}" = "1" ]; then
    run "preflight_full" tools/ci/preflight.sh
  else
    run_shell "preflight_fast" "FAST=1 tools/ci/preflight.sh"
  fi
fi

if [ "${RUN_GPU:-1}" = "1" ] && [ -x ./target/release/hawking ]; then
  make_prompts

  if [ -x tools/bench/ratios.sh ]; then
    wait_gpu_slot
    run "ratios_default_ab" tools/bench/ratios.sh ab "" short "${TRIALS:-3}"
    wait_gpu_slot
    run_shell "ratios_profile_fast_abi" "PROFILE=fast TOK=${TOK:-96} tools/bench/ratios.sh abi \"\" short ${TRIALS:-6}"
    wait_gpu_slot
    run "ratios_f16kv_qual" tools/bench/ratios.sh qual "HAWKING_QWEN_F16_KV=1" "" 80
  fi

  if [ -f models/rwkv7-g1-04-sft-Q4_K_M.gguf ]; then
    wait_gpu_slot
    bench_model_tps "rwkv7_sft_short" "models/rwkv7-g1-04-sft-Q4_K_M.gguf" "$OUT/prompts/short.txt" 64 "${TRIALS:-3}"
    wait_gpu_slot
    bench_model_tps "rwkv7_sft_mid" "models/rwkv7-g1-04-sft-Q4_K_M.gguf" "$OUT/prompts/ctx2k.txt" 64 "${TRIALS:-3}"
    wait_gpu_slot
    bench_model_tps "rwkv7_sft_long" "models/rwkv7-g1-04-sft-Q4_K_M.gguf" "$OUT/prompts/ctx8k.txt" 64 "${TRIALS:-3}"
  fi

  if [ -f models/mamba2-370m-Q4_K_M.gguf ]; then
    wait_gpu_slot
    bench_model_tps "mamba2_short" "models/mamba2-370m-Q4_K_M.gguf" "$OUT/prompts/short.txt" 64 "${TRIALS:-3}"
    wait_gpu_slot
    bench_model_tps "mamba2_mid" "models/mamba2-370m-Q4_K_M.gguf" "$OUT/prompts/ctx2k.txt" 64 "${TRIALS:-3}"
    wait_gpu_slot
    bench_model_tps "mamba2_long" "models/mamba2-370m-Q4_K_M.gguf" "$OUT/prompts/ctx8k.txt" 64 "${TRIALS:-3}"
  fi

  if [ "${RUN_SERVE_SMOKE:-1}" = "1" ] && [ -x tools/ci/ssm_serve_smoke.sh ]; then
    if [ -f models/rwkv7-g1-04-sft-Q4_K_M.gguf ]; then
      wait_gpu_slot
      run_shell "ssm_serve_smoke_rwkv7" "OUT='$OUT/serve_rwkv7' tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf"
    fi
    if [ "${RUN_MAMBA_SERVE_SMOKE:-0}" = "1" ] && [ -f models/mamba2-370m-Q4_K_M.gguf ]; then
      wait_gpu_slot
      run_shell "ssm_serve_smoke_mamba2" "OUT='$OUT/serve_mamba2' tools/ci/ssm_serve_smoke.sh models/mamba2-370m-Q4_K_M.gguf"
    fi
  fi
fi

{
  echo
  echo "## Final Git State"
  echo '```'
  git status --short
  echo '```'
  echo
  echo "## Step Results"
  echo '```'
  cat "$COMMAND_LOG"
  echo '```'
  echo
  if [ "$FAIL" = "0" ]; then
    echo "**Result:** PASS (all enabled steps returned 0)."
  else
    echo "**Result:** FAIL/ATTENTION (one or more enabled steps failed; inspect logs above)."
  fi
  echo
  echo "Next recovery command:"
  echo
  echo '```bash'
  echo "sed -n '1,220p' $SUMMARY"
  echo '```'
} >>"$SUMMARY"

log "Hawking overnight hardening complete fail=$FAIL summary=$SUMMARY"
exit "$FAIL"
