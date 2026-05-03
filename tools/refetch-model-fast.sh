#!/usr/bin/env bash
# Fast, resumable refetch for dismantle's hero GGUF.
#
# Usage:
#   ./tools/refetch-model-fast.sh
#
# Optional, for faster Hub transfers on a strong connection:
#   python3 -m pip install -U "huggingface_hub[hf_transfer]"
#   HF_FAST=1 ./tools/refetch-model-fast.sh
#
# Output:
#   models/deepseek-v2-lite-q4.gguf

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
TARGET="$MODELS_DIR/deepseek-v2-lite-q4.gguf"

MIRRORS=(
  "mradermacher/DeepSeek-V2-Lite-Chat-GGUF|DeepSeek-V2-Lite-Chat.Q4_K_M.gguf"
  "legraphista/DeepSeek-V2-Lite-Chat-IMat-GGUF|DeepSeek-V2-Lite-Chat.Q4_K.gguf"
)

mkdir -p "$MODELS_DIR"

if [[ -f "$TARGET" ]]; then
  echo "[refetch-model-fast] already present: $TARGET"
  ls -lh "$TARGET"
  exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "[refetch-model-fast] missing Hugging Face CLI."
  echo 'Install: python3 -m pip install -U "huggingface_hub[cli]"'
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "[refetch-model-fast] not logged in to Hugging Face."
  echo "Run: hf auth login"
  exit 1
fi

if [[ "${HF_FAST:-0}" == "1" ]]; then
  if python3 -c 'import hf_transfer' >/dev/null 2>&1; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "[refetch-model-fast] hf_transfer enabled"
  else
    echo "[refetch-model-fast] HF_FAST=1 requested, but hf_transfer is not installed."
    echo 'Install: python3 -m pip install -U "huggingface_hub[hf_transfer]"'
    exit 1
  fi
fi

echo "[refetch-model-fast] target: $TARGET"
echo "[refetch-model-fast] user: $(hf auth whoami 2>/dev/null | head -1)"

for mirror_spec in "${MIRRORS[@]}"; do
  repo="${mirror_spec%%|*}"
  filename="${mirror_spec##*|}"
  tmp="$MODELS_DIR/$filename"

  echo "[refetch-model-fast] downloading $repo / $filename"
  if hf download "$repo" "$filename" --local-dir "$MODELS_DIR"; then
    if [[ -f "$tmp" ]]; then
      mv "$tmp" "$TARGET"
      echo "[refetch-model-fast] complete"
      ls -lh "$TARGET"
      shasum -a 256 "$TARGET"
      exit 0
    fi
  fi

  echo "[refetch-model-fast] mirror failed; trying next"
done

echo "[refetch-model-fast] all mirrors failed"
exit 1
