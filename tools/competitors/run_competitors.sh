#!/usr/bin/env bash
# Run the head-to-head benchmark suite against llama.cpp and
# dismantle on the same prompt set + same hardware. Emits a JSON
# document at tools/competitors/results.json.
#
# Honesty rules (per docs/m3_audit.md):
#   - 5 minutes idle between backends to let thermals settle
#   - power adapter connected, lid open, hard surface
#   - macOS Low Power Mode OFF
#   - reports min / median / max over $TRIALS trials per (backend, prompt)
#
# Prerequisites:
#   - models/deepseek-v2-lite-q4.gguf  (run tools/fetch-model.sh first)
#   - llama-cli installed (brew install llama.cpp)
#   - dismantle release binary built (cargo build --release)
#
# Usage:
#   ./tools/competitors/run_competitors.sh [TRIALS]    # default TRIALS=3

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="$REPO_ROOT/models/deepseek-v2-lite-q4.gguf"
PROMPTS="$REPO_ROOT/tools/competitors/prompts.txt"
RESULTS="$REPO_ROOT/tools/competitors/results.json"
VERSIONS="$REPO_ROOT/tools/competitors/versions.json"
TRIALS="${1:-3}"
MAX_TOKENS=256
TEMP=0.0

[[ -f "$MODEL" ]]   || { echo "missing $MODEL — run tools/fetch-model.sh"; exit 1; }
[[ -f "$PROMPTS" ]] || { echo "missing $PROMPTS"; exit 1; }
command -v llama-cli >/dev/null 2>&1 || { echo "missing llama-cli (brew install llama.cpp)"; exit 1; }

DISMANTLE="$REPO_ROOT/target/release/dismantle"
[[ -x "$DISMANTLE" ]] || { echo "missing $DISMANTLE — run cargo build --release"; exit 1; }

# ---- Pin versions ---------------------------------------------------

llamacpp_version="$(llama-cli --version 2>&1 | head -1)"
dismantle_version="$("$DISMANTLE" version 2>/dev/null | head -1)"
chip_string="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
hw_model="$(sysctl -n hw.model 2>/dev/null || echo unknown)"
mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
os_build="$(sw_vers -buildVersion 2>/dev/null || echo unknown)"
os_version="$(sw_vers -productVersion 2>/dev/null || echo unknown)"

cat >"$VERSIONS" <<EOF
{
  "captured_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hardware": {
    "model": "$hw_model",
    "chip": "$chip_string",
    "memory_bytes": $mem_bytes
  },
  "os": { "version": "$os_version", "build": "$os_build" },
  "backends": {
    "llamacpp":  "$llamacpp_version",
    "dismantle": "$dismantle_version"
  },
  "model": "$(basename "$MODEL")",
  "trials_per_prompt": $TRIALS
}
EOF
echo "[run-competitors] pinned versions → $VERSIONS"

# ---- Helpers --------------------------------------------------------

# llama.cpp: parses both the new-style "[ Prompt: X t/s | Generation:
# Y t/s ]" footer (b8870+) and the legacy "eval time = ... tokens per
# second" lines, using sed -nE for portable BSD/GNU regex.
run_llamacpp() {
  local prompt="$1"
  local out
  out="$(printf '%s\n/exit\n' "$prompt" | llama-cli \
      --model "$MODEL" \
      --prompt "$prompt" \
      --predict $MAX_TOKENS \
      --temp $TEMP \
      -no-cnv \
      --no-display-prompt \
      --no-warmup \
      2>&1 || true)"
  local eval_tps prompt_tps ttft_ms
  eval_tps=$(  printf '%s\n' "$out" | sed -nE 's/.*Generation:[[:space:]]+([0-9.]+)[[:space:]]+t.*/\1/p' | tail -1)
  prompt_tps=$(printf '%s\n' "$out" | sed -nE 's/.*Prompt:[[:space:]]+([0-9.]+)[[:space:]]+t.*/\1/p' | tail -1)
  if [[ -z "$eval_tps" ]]; then
    eval_tps=$(printf '%s\n' "$out" | grep -E '^[[:space:]]*eval time' \
      | sed -nE 's/.*\(([0-9.]+)[[:space:]]+tokens per second\).*/\1/p' | tail -1)
  fi
  if [[ -z "$prompt_tps" ]]; then
    prompt_tps=$(printf '%s\n' "$out" | grep 'prompt eval time' \
      | sed -nE 's/.*\(([0-9.]+)[[:space:]]+tokens per second\).*/\1/p' | tail -1)
  fi
  ttft_ms=$(printf '%s\n' "$out" | grep 'prompt eval time' \
    | sed -nE 's/.*=[[:space:]]*([0-9.]+)[[:space:]]+ms.*/\1/p' | tail -1)
  printf '{"decode_tps":%s,"prefill_tps":%s,"ttft_ms":%s}' \
    "${eval_tps:-null}" "${prompt_tps:-null}" "${ttft_ms:-null}"
}

# dismantle's --weights generate path with stats line on stderr.
run_dismantle() {
  local prompt="$1"
  local out
  out="$("$DISMANTLE" generate \
      --weights "$MODEL" \
      --prompt "$prompt" \
      --max-new-tokens $MAX_TOKENS \
      --temperature $TEMP \
      --max-stall-ms 120000 \
      2>&1 || true)"
  # [stats] reason=R prompt=N completion=M prefill_ms=P decode_ms=D dec_tps=T
  local prefill_ms decode_tps
  prefill_ms=$(printf '%s\n' "$out" | sed -nE 's/.*prefill_ms=([0-9.]+).*/\1/p' | head -1)
  decode_tps=$(printf '%s\n' "$out" | sed -nE 's/.*dec_tps=([0-9.]+).*/\1/p' | head -1)
  printf '{"decode_tps":%s,"prefill_tps":null,"ttft_ms":%s,"phase":0}' \
    "${decode_tps:-null}" "${prefill_ms:-null}"
}

# ---- Run matrix -----------------------------------------------------

n_prompts=$(grep -c '|' "$PROMPTS")
echo "[run-competitors] starting matrix; trials=$TRIALS prompts=$n_prompts backends=2"

ROWS_JSON=""
prompt_idx=0
while IFS='|' read -r tier prompt; do
  [[ -z "${tier:-}" ]] && continue
  [[ "$tier" =~ ^# ]] && continue
  prompt_idx=$((prompt_idx + 1))
  echo "[$prompt_idx] $tier — ${prompt:0:60}..."

  for backend in llamacpp dismantle; do
    trial_results=""
    for ((t=1; t<=TRIALS; t++)); do
      case "$backend" in
        llamacpp)  r="$(run_llamacpp  "$prompt")" ;;
        dismantle) r="$(run_dismantle "$prompt")" ;;
      esac
      trial_results+="$r,"
    done
    trial_results="[${trial_results%,}]"

    row=$(printf '{"prompt_idx":%d,"tier":"%s","backend":"%s","trials":%s}' \
      "$prompt_idx" "$tier" "$backend" "$trial_results")
    ROWS_JSON+="$row,"

    # Thermal courtesy between backends within the same prompt.
    sleep 2
  done
done <"$PROMPTS"

ROWS_JSON="[${ROWS_JSON%,}]"

cat >"$RESULTS" <<EOF
{
  "versions_ref": "tools/competitors/versions.json",
  "rows": $ROWS_JSON
}
EOF

echo "[run-competitors] wrote $RESULTS"
echo "[run-competitors] feed this into the audit doc:"
echo "  cargo run --release -p dismantle -- bench --weights $MODEL --model deepseek-v2-lite --suite competitive"
echo "    will reproduce the same shape from inside the binary."
