#!/usr/bin/env bash
# tools/bench/quality_scout.sh — fixed 6-prompt quality probe across model sizes.
#
# For each WEIGHTS supplied, runs the same 6 prompts at temperature=0 with the
# locked Qwen fast-path env, prints the generations side-by-side so a reader
# can eyeball PASS/PARTIAL/FAIL coherence. No automated scoring — the point
# is to compare 0.5B vs 1.5B vs 3B on the same inputs.
#
# Usage:
#   WEIGHTS_LIST="models/a.gguf,models/b.gguf,models/c.gguf" bash tools/bench/quality_scout.sh
#   or pass the comma-list as first arg.

set -o pipefail
cd "$(dirname "$0")/../.."

BIN="./target/release/dismantle"
WEIGHTS_LIST="${1:-${WEIGHTS_LIST:-}}"
TOKENS="${TOKENS:-80}"
SEED="${SEED:-0}"

if [[ -z "$WEIGHTS_LIST" ]]; then
    echo "usage: $0 <w1.gguf,w2.gguf,...>" >&2
    exit 2
fi

# Locked Qwen fast-path env (set if missing) — same as quick_bench.sh.
: "${DISMANTLE_QWEN_TCB:=1}"
: "${DISMANTLE_QWEN_VOCAB_PRUNE:=32000}"
: "${DISMANTLE_QWEN_Q4K_LMHEAD:=1}"
: "${DISMANTLE_QWEN_FFN_DOWN_Q4K:=1}"
export DISMANTLE_QWEN_TCB DISMANTLE_QWEN_VOCAB_PRUNE DISMANTLE_QWEN_Q4K_LMHEAD DISMANTLE_QWEN_FFN_DOWN_Q4K

# 6 prompts spanning chat, code, math, factual recall, creative, instruction follow.
PROMPTS=(
    "Explain what makes the sky blue in two sentences."
    "Write a Python function reverse_words(s) that reverses the order of words in a string. Include a docstring."
    "What is 47 times 38?"
    "Who painted the ceiling of the Sistine Chapel and in what century?"
    "Continue the story: 'The old lighthouse keeper had never seen the lights flicker like that before.'"
    "List exactly three reasons espresso is bitter. Number them 1, 2, 3."
)

IFS=',' read -ra WEIGHTS <<< "$WEIGHTS_LIST"

for w in "${WEIGHTS[@]}"; do
    if [[ ! -f "$w" ]]; then
        echo "❌ weights not found: $w" >&2
        continue
    fi
    base="$(basename "$w" .gguf)"
    echo
    echo "================================================================"
    echo "MODEL: $base"
    echo "================================================================"
    for i in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$i]}"
        echo
        echo "--- Q$((i+1)): $prompt"
        "$BIN" generate \
            --weights "$w" \
            --prompt "$prompt" \
            --max-new-tokens "$TOKENS" \
            --temperature 0 \
            --seed "$SEED" 2>&1 | grep -v '^$' | head -20
    done
done
