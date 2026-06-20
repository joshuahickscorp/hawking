#!/usr/bin/env bash
# 0.5 — long-context harness. Decodes at growing context (4K/16K/32K) and
# reports per-token decode tps + the KV-cache byte-share of per-token reads.
# GATES Tier-2 fused-KV (2.1): it proves where KV bandwidth actually starts to
# bite. At short ctx the ~1.93 GB/token weight read dominates; KV-share grows
# linearly with ctx until, past ~16K, reading the KV cache rivals the weights —
# that's where inline int4/int8-KV attention (no f16 KV buffer) pays off.
#
# CPU to write; runs on the GPU-bench lane (a 32K-ctx decode is a real GPU job —
# run it when the EAGLE capture frees the M3). Prompt is synthesized by repeating
# a code snippet to ~target tokens (≈4 chars/token); override with PROMPT_FILE.
#
# Usage:
#   tools/bench/long_context_bench.sh                  # 4096 16384 32768
#   CTXS="2048 8192" tools/bench/long_context_bench.sh
#   PROMPT_FILE=big.txt tools/bench/long_context_bench.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

BIN="${BIN:-./target/release/hawking}"
WEIGHTS="${WEIGHTS:-models/qwen2.5-3b-instruct-q4_k_m.gguf}"
PROFILE="${PROFILE:-profiles/qwen3b-instruct-q4k.m3pro18.json}"
CTXS="${CTXS:-4096 16384 32768}"
GEN_TOKENS="${GEN_TOKENS:-32}"      # decode tokens to time at each ctx
OUT="${OUT:-reports/bench/long_context.json}"

# Qwen2.5-3B GQA dims (verify against the GGUF metadata if porting to another model).
N_LAYERS="${N_LAYERS:-36}"
N_KV_HEADS="${N_KV_HEADS:-2}"
HEAD_DIM="${HEAD_DIM:-128}"
KV_DTYPE_BYTES="${KV_DTYPE_BYTES:-2}"          # f16 KV cache
WEIGHT_BYTES_PER_TOKEN="${WEIGHT_BYTES_PER_TOKEN:-2072821924}"  # ~1.93 GiB

[[ -x "$BIN" ]] || { echo "error: build first (cargo build --release -p hawking)"; exit 1; }
[[ -f "$WEIGHTS" ]] || { echo "error: weights not found: $WEIGHTS"; exit 1; }
mkdir -p "$(dirname "$OUT")"

# Per-position KV bytes read each decode step (both K and V, all layers).
kv_bytes_per_pos=$(( 2 * N_LAYERS * N_KV_HEADS * HEAD_DIM * KV_DTYPE_BYTES ))

synth_prompt() {  # $1 = approx target tokens -> stdout (~4 chars/token)
  local target_chars=$(( $1 * 4 )) acc=""
  local unit='fn step(s: &mut State, i: usize) -> u64 { s.acc += i as u64; s.acc } '
  while [[ ${#acc} -lt $target_chars ]]; do acc+="$unit"; done
  printf '%s' "${acc:0:$target_chars}"
}

echo "=== long-context bench (KV f16, $N_LAYERS L × $N_KV_HEADS kv-heads × $HEAD_DIM) ==="
printf "%8s %12s %12s %14s %12s\n" "ctx" "prefill_ms" "decode_tps" "kv_B/token" "kv_share%"
results="[]"
for ctx in $CTXS; do
  if [[ -n "${PROMPT_FILE:-}" ]]; then promptf="$PROMPT_FILE"
  else promptf="/tmp/lcb_prompt_${ctx}.txt"; synth_prompt "$ctx" > "$promptf"; fi
  j="/tmp/lcb_${ctx}.json"
  # generate prints "[stats] ... prefill_ms=NN ... dec_tps=NN" to STDERR — but
  # ONLY when --json is NOT passed (--json suppresses the stats line). So run
  # WITHOUT --json and parse the stats line from captured stderr.
  # --max-seq-len must exceed prompt+gen or generate errors "kv cache full" and
  # emits no stats (the bug that showed decode_tps=0 at ctx>=4096). Size it to
  # ctx + GEN_TOKENS + margin.
  maxseq=$(( ctx + ctx/2 + GEN_TOKENS + 256 ))   # 50% headroom: synth prompt may tokenize denser than ~4 char/tok
  serr=$(nice -n 19 taskpolicy -b "$BIN" generate --weights "$WEIGHTS" \
    --kernel-profile "$PROFILE" --prompt "$(cat "$promptf")" \
    --max-seq-len "$maxseq" \
    --max-new-tokens "$GEN_TOKENS" --temperature 0 --seed 0 2>&1 >/dev/null)
  prefill=$(printf '%s\n' "$serr" | sed -n 's/.*prefill_ms=\([0-9.][0-9.]*\).*/\1/p' | head -1)
  dtps=$(printf '%s\n' "$serr" | sed -n 's/.*dec_tps=\([0-9.][0-9.]*\).*/\1/p' | head -1)
  prefill="${prefill:-0}"; dtps="${dtps:-0}"
  kv_tok=$(( kv_bytes_per_pos * ctx ))
  share=$(awk -v kv="$kv_tok" -v w="$WEIGHT_BYTES_PER_TOKEN" 'BEGIN{printf "%.1f", kv/(kv+w)*100}')
  printf "%8s %12s %12s %14s %11s%%\n" "$ctx" "${prefill:-?}" "${dtps:-?}" "$kv_tok" "$share"
  results=$(jq -c --argjson c "$ctx" --arg p "${prefill:-0}" --arg d "${dtps:-0}" \
    --argjson kv "$kv_tok" --arg s "$share" \
    '. += [{ctx:$c, prefill_ms:($p|tonumber), decode_tps:($d|tonumber), kv_bytes_per_token:$kv, kv_share_pct:($s|tonumber)}]' \
    <<<"$results")
done
jq -n --argjson r "$results" --argjson kvpos "$kv_bytes_per_pos" \
  '{harness:"long_context", kv_bytes_per_position:$kvpos, weight_bytes_per_token:'"$WEIGHT_BYTES_PER_TOKEN"', rows:$r}' \
  > "$OUT"
echo "wrote $OUT"
echo "NB: prefill_ms/decode_tps require dismantle generate --json to emit them; if blank, add the fields or parse from bench --suite decode."
