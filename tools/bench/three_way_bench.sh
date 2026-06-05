#!/usr/bin/env bash
# =============================================================================
# tools/bench/three_way_bench.sh — dismantle vs llama.cpp vs MLX, side by side.
#
# Decode throughput (dec_tps) AND energy (per-token joules, GPU + package) for
# all three engines on Qwen2.5-3B, same prompt / token budget / greedy temp=0.
# Energy is read sudo-free from macmon (IOReport), the SAME reader the dismantle
# anchor uses, so the J/tok numbers are directly comparable to our 0.196 J/tok.
#
# CLEAN ROOM REQUIRED — quit Claude.app AND the Claude CLI and any heavy GPU job
# first; absolute tps/J inflate ~4-5x under a live session (the preflight aborts).
#
# USAGE:
#   tools/bench/three_way_bench.sh                  # all three
#   ONLY=dismantle,llama tools/bench/three_way_bench.sh
#   TOKENS=256 tools/bench/three_way_bench.sh
#   RUN_TIMEOUT_SEC=900 tools/bench/three_way_bench.sh
#   MLX_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit tools/bench/three_way_bench.sh
#
# FAIRNESS CAVEATS (read before quoting numbers):
#   - dismantle + llama.cpp run the SAME GGUF (Qwen2.5-3B-Instruct-Q4_K_M).
#   - MLX needs an MLX-format model (default mlx-community/...-4bit) — a DIFFERENT
#     quant (4-bit affine vs Q4_K_M), so MLX is directionally comparable, not
#     bit-for-bit the same weights. mlx_lm downloads it on first run.
#   - J/tok = avg_power_W / dec_tps (decode-dominated at N>=128 with a short
#     prompt). For a prefill-clean number, raise TOKENS.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

TOKENS="${TOKENS:-128}"
PROMPT="${PROMPT:-Explain how a CPU pipeline hazard is resolved by forwarding.}"
GGUF="${GGUF:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
MLX_MODEL="${MLX_MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}"
DBIN="${DBIN:-./target/release/dismantle}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
SAMPLE_MS="${SAMPLE_MS:-200}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-600}"
STALL_MS="${STALL_MS:-30000}"
ONLY="${ONLY:-}"
NICE=(nice -n 19 taskpolicy -b)

# --- preflight ---------------------------------------------------------------
command -v macmon >/dev/null 2>&1 || { echo "FAIL: macmon missing -> brew install macmon"; exit 3; }
if pgrep -f "Claude.app" >/dev/null 2>&1 || pgrep -xi "claude" >/dev/null 2>&1; then
  echo "FAIL: a Claude session (app or CLI) is running — absolute tps/J inflate ~4-5x. Quit it and re-run."
  exit 3
fi
want() { [[ -z "$ONLY" || ",$ONLY," == *",$1,"* ]]; }
have() { command -v "$1" >/dev/null 2>&1; }

pick_mlx_bin() {
  if [[ -n "${MLX_BIN:-}" && -x "${MLX_BIN:-}" ]]; then
    echo "$MLX_BIN"
    return
  fi
  for py in "${MLX_PYTHON:-}" "$(dirname "$0")/.mlxenv/bin/python" /tmp/mlxenv/bin/python \
            /Library/Frameworks/Python.framework/Versions/3.12/bin/python3; do
    [[ -n "$py" && -x "$py" ]] || continue
    if "$py" -c 'import mlx_lm' >/dev/null 2>&1; then
      local bin
      bin="$(dirname "$py")/mlx_lm.generate"
      [[ -x "$bin" ]] && { echo "$bin"; return; }
    fi
  done
  command -v mlx_lm.generate 2>/dev/null || true
}

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    perl -e 'use strict; my $secs = shift; alarm $secs; exec { $ARGV[0] } @ARGV or die "exec: $!"' \
      "$secs" "$@"
  fi
}

# --- macmon sampler (mirrors measure_joules.sh) ------------------------------
# `macmon pipe -i <ms>` emits one JSON object per sample; we pull all_power
# (package: CPU+GPU+ANE) and gpu_power defensively and average post-hoc.
sample_macmon() { # $1 pkg-file  $2 gpu-file
  macmon pipe -i "$SAMPLE_MS" 2>/dev/null | while IFS= read -r line; do
    p=$(printf '%s' "$line" | sed -n 's/.*"all_power"[: ]*\([0-9.]*\).*/\1/p')
    g=$(printf '%s' "$line" | sed -n 's/.*"gpu_power"[: ]*\([0-9.]*\).*/\1/p')
    [[ -n "$p" ]] && echo "$p" >> "$1"
    [[ -n "$g" ]] && echo "$g" >> "$2"
  done
}
avg() { awk '{s+=$1;n++} END{if(n)printf "%.4f", s/n; else printf "0"}' "$1" 2>/dev/null; }

# per-engine dec_tps parsers (tail = the decode/eval phase, not prefill)
tps_dismantle() { grep -oE 'dec_tps=[0-9.]+' "$1" | grep -oE '[0-9.]+' | tail -1; }
tps_llama() {
  local t
  t=$(grep -oiE 'Generation:[^0-9]*[0-9.]+ *t/s' "$1" | grep -oE '[0-9.]+' | tail -1)
  [[ -n "$t" ]] && { echo "$t"; return; }
  grep -oiE 'eval time.*[0-9.]+ *tokens per second' "$1" | grep -oE '[0-9.]+' | tail -1
}
tps_mlx()       { grep -oiE 'generation:.*[0-9.]+ *tokens-per-sec' "$1" | grep -oE '[0-9.]+' | tail -1; }

: > /tmp/tw_results

run() { # $1 name  $2 tps-parser-fn  $3.. command...
  local name="$1"; shift
  local tps_fn="$1"; shift
  local pkgf="/tmp/tw_${name}_pkg" gpuf="/tmp/tw_${name}_gpu" out="/tmp/tw_${name}.log"
  : > "$pkgf"; : > "$gpuf"
  printf '\n--- %s ---\n' "$name"
  printf '  timeout=%ss  log=%s\n' "$RUN_TIMEOUT_SEC" "$out"
  sample_macmon "$pkgf" "$gpuf" & local smp=$!
  local t0 t1; t0=$(date +%s.%N)
  run_with_timeout "$RUN_TIMEOUT_SEC" "$@" </dev/null > "$out" 2>&1
  local rc=$?; t1=$(date +%s.%N)
  pkill -P "$smp" 2>/dev/null || true
  kill "$smp" 2>/dev/null || true
  wait "$smp" 2>/dev/null || true
  local wall tps apkg agpu jg jp
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
  tps=$("$tps_fn" "$out"); [[ -z "$tps" ]] && tps="?"
  apkg=$(avg "$pkgf"); agpu=$(avg "$gpuf")
  jg=$(awk -v p="$agpu" -v t="$tps" 'BEGIN{if(t+0>0)printf "%.4f",p/t; else printf "?"}')
  jp=$(awk -v p="$apkg" -v t="$tps" 'BEGIN{if(t+0>0)printf "%.4f",p/t; else printf "?"}')
  if [[ $rc -eq 124 || $rc -eq 142 ]]; then
    echo "  WARN: ${name} timed out after ${RUN_TIMEOUT_SEC}s — inspect $out (tail below):"
    tail -8 "$out" | sed 's/^/    /'
  elif [[ $rc -ne 0 || "$tps" == "?" ]]; then
    echo "  WARN: ${name} did not produce a clean dec_tps (rc=$rc) — inspect $out (tail below):"
    tail -6 "$out" | sed 's/^/    /'
  fi
  printf '%s|%s|%s|%s|%s|%s\n' "$name" "$tps" "$jg" "$jp" "$agpu" "$wall" >> /tmp/tw_results
  printf "  dec_tps=%s  J/tok(GPU)=%s  J/tok(pkg)=%s  avgGPU=%sW  wall=%ss\n" "$tps" "$jg" "$jp" "$agpu" "$wall"
}

# --- runs --------------------------------------------------------------------
if want dismantle; then
  [[ -x "$DBIN" ]] || echo "WARN: $DBIN not built (cargo build --release) — skipping dismantle."
  [[ -x "$DBIN" ]] && run dismantle tps_dismantle "${NICE[@]}" "$DBIN" generate \
    --weights "$GGUF" --kernel-profile "$PROFILE" \
    --prompt "$PROMPT" --max-new-tokens "$TOKENS" --temperature 0 --seed 0 \
    --max-stall-ms "$STALL_MS"
fi
if want llama; then
  LLAMA_BIN="${LLAMA_BIN:-$(command -v llama-completion 2>/dev/null || true)}"
  [[ -x "$LLAMA_BIN" ]] || echo "WARN: llama-completion not found — skipping llama. (This bench intentionally avoids llama-cli chat mode.)"
  [[ -x "$LLAMA_BIN" ]] && run llama tps_llama "${NICE[@]}" "$LLAMA_BIN" -m "$GGUF" -p "$PROMPT" \
    -n "$TOKENS" --temp 0 --seed 0 -ngl 99 -no-cnv -st --simple-io \
    --no-display-prompt --no-warmup --perf
fi
if want mlx; then
  MLX_BIN_RESOLVED="$(pick_mlx_bin)"
  [[ -x "$MLX_BIN_RESOLVED" ]] || echo "WARN: mlx_lm.generate not found (pip install mlx-lm) — skipping mlx."
  [[ -x "$MLX_BIN_RESOLVED" ]] && run mlx tps_mlx "${NICE[@]}" "$MLX_BIN_RESOLVED" --model "$MLX_MODEL" \
    --prompt "$PROMPT" --max-tokens "$TOKENS" --temp 0 --seed 0 --verbose True
fi

# --- table -------------------------------------------------------------------
echo
echo "================ THREE-WAY (Qwen2.5-3B, N=$TOKENS, greedy temp=0) ================"
printf '%-11s %9s %12s %12s %10s %8s\n' engine dec_tps J/tok_GPU J/tok_pkg avgGPU_W wall_s
while IFS='|' read -r n tps jg jp ag w; do
  printf '%-11s %9s %12s %12s %10s %8s\n' "$n" "$tps" "$jg" "$jp" "$ag" "$w"
done < /tmp/tw_results
echo
echo "J/tok = avg_power_W / dec_tps. dismantle+llama = same Q4_K_M GGUF; MLX = $MLX_MODEL (4-bit, different quant)."
echo "Anchor: dismantle clean single-stream ~32.65 dec_tps / ~0.196 J/tok (pkg). llama target ~49 dec_tps."
echo "done."
