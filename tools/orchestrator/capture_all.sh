#!/usr/bin/env bash
# Capture quantized-residual corpora for all Qwen models at layer n-1.
# Runs the local Metal runtime (the ONLY place quantized residuals exist) so
# the Colab can train heads on the distribution the runtime actually serves.
# Output: _capture/<slug>_corpus.bin  +  _capture/<slug>_corpus_shards/
set -euo pipefail
cd "$(dirname "$0")/../.."

PROMPTS=_capture/corpus_prompts.txt
MAXTOK=64
PY=/tmp/eagle5venv/bin/python
BIN=./target/release/hawking

LOCKED="HAWKING_QWEN_TCB=1 HAWKING_QWEN_VOCAB_PRUNE=32000 HAWKING_QWEN_Q4K_LMHEAD=1 HAWKING_QWEN_FFN_DOWN_Q4K=1 HAWKING_QWEN_Q4K_PREDEC=1"

# slug:model:capture_layer(n-1)
for entry in "q05b:0.5b:23" "q1p5b:1.5b:27" "q7b:7b:27"; do
  slug=${entry%%:*}; rest=${entry#*:}; m=${rest%%:*}; layer=${rest##*:}
  gguf="models/qwen2.5-${m}-instruct-q4_k_m.gguf"
  bin="_capture/${slug}_corpus.bin"
  echo "=== capturing $slug (layer $layer) from $gguf ==="
  rm -f "$bin"
  env $LOCKED \
      HAWKING_QWEN_EAGLE5_CAPTURE=1 \
      HAWKING_QWEN_EAGLE5_CAPTURE_LAYER=$layer \
      HAWKING_QWEN_CAPTURE_CORPUS_PATH="$bin" \
    $BIN generate --weights "$gguf" --prompts-file "$PROMPTS" \
      --max-new-tokens $MAXTOK --temperature 0 >/dev/null 2>"_capture/${slug}_capture.log"
  echo "  packing $slug..."
  rm -rf "_capture/${slug}_corpus_shards"
  $PY tools/orchestrator/pack_corpus.py --in "$bin" \
      --out-dir "_capture/${slug}_corpus_shards" --rows-per-shard 64 2>&1 | tail -1
done
echo "=== ALL CAPTURES DONE ==="
