#!/usr/bin/env bash
# Download the standard DeepSeek-V2-Lite Q4_K_M GGUF for dismantle's
# Phase 0 gate + the head-to-head benchmark suite. ~9 GB.
#
# Idempotent: skips download if file already exists and the sha256
# matches the pinned hash below. Re-run is cheap.
#
# Auth: the HuggingFace CLI path can use `hf auth login`, but the public
# mirrors below are also tried with direct `curl` so a token is not required
# when the mirror's resolve URL is publicly reachable.
#
# Usage:
#   ./tools/fetch-model.sh
#
# Output:
#   models/deepseek-v2-lite-q4.gguf

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
TARGET="$MODELS_DIR/deepseek-v2-lite-q4.gguf"

# Mirror order: mradermacher's standard Q4_K_M is the canonical
# choice; legraphista's IMat (importance-matrix) Q4_K is a slightly
# different quantization but a workable fallback. Both repos are
# public + ungated as of the audit (2026-04-27); HF still requires
# auth on the resolve endpoint regardless, so `hf auth login` first.
#
# Filename convention is .Q4_K_M.gguf (period before quant), not -.
MIRRORS=(
  "mradermacher/DeepSeek-V2-Lite-Chat-GGUF|DeepSeek-V2-Lite-Chat.Q4_K_M.gguf"
  "legraphista/DeepSeek-V2-Lite-Chat-IMat-GGUF|DeepSeek-V2-Lite-Chat.Q4_K.gguf"
)

# Pin the sha256 here once a successful download produces a known-
# good file. Until then, the script accepts whatever it downloads
# and prints the hash for you to paste in.
EXPECTED_SHA256="5d33e5f045c7a03351c319aafc8afdad94b69d07bb68f36dc9bb5af340b343a4"

mkdir -p "$MODELS_DIR"

# ---- Idempotent skip on existing file --------------------------------

if [[ -f "$TARGET" ]]; then
  echo "[fetch-model] $TARGET already exists"
  if [[ "$EXPECTED_SHA256" != "UNVERIFIED-PIN-AFTER-FIRST-DOWNLOAD" ]]; then
    actual=$(shasum -a 256 "$TARGET" | awk '{print $1}')
    if [[ "$actual" == "$EXPECTED_SHA256" ]]; then
      echo "[fetch-model] sha256 OK ($actual)"
      exit 0
    else
      echo "[fetch-model] WARNING sha256 mismatch:"
      echo "  expected: $EXPECTED_SHA256"
      echo "  actual:   $actual"
      echo "  delete the file and re-run if you want a fresh download"
      exit 1
    fi
  else
    echo "[fetch-model] no pinned sha256 yet; using existing file as-is"
    echo "[fetch-model] paste this into EXPECTED_SHA256 to lock it:"
    shasum -a 256 "$TARGET"
    exit 0
  fi
fi

# ---- Try each mirror -------------------------------------------------

echo "[fetch-model] target: $TARGET (~9 GB)"
if command -v hf >/dev/null 2>&1 && hf auth whoami >/dev/null 2>&1; then
  echo "[fetch-model] logged in as: $(hf auth whoami 2>/dev/null | head -1)"
else
  echo "[fetch-model] hf auth unavailable; trying public mirror URLs"
fi

for mirror_spec in "${MIRRORS[@]}"; do
  repo="${mirror_spec%%|*}"
  filename="${mirror_spec##*|}"
  echo "[fetch-model] trying $repo / $filename"

  if command -v hf >/dev/null 2>&1 && hf auth whoami >/dev/null 2>&1 && \
      hf download "$repo" "$filename" --local-dir "$MODELS_DIR" 2>&1 | tail -20; then
    src="$MODELS_DIR/$filename"
    if [[ -f "$src" ]]; then
      mv "$src" "$TARGET"
      echo "[fetch-model] download complete"
      echo "[fetch-model] sha256:"
      shasum -a 256 "$TARGET"
      echo
      echo "[fetch-model] paste the hash above into EXPECTED_SHA256 in this"
      echo "[fetch-model] script to lock the version for future runs."
      exit 0
    fi
  fi

  if curl -L -C - --fail --progress-bar \
      -o "$TARGET" \
      "https://huggingface.co/$repo/resolve/main/$filename"; then
    echo "[fetch-model] download complete"
    actual=$(shasum -a 256 "$TARGET" | awk '{print $1}')
    echo "[fetch-model] sha256: $actual"
    if [[ "$EXPECTED_SHA256" != "UNVERIFIED-PIN-AFTER-FIRST-DOWNLOAD" && "$actual" != "$EXPECTED_SHA256" ]]; then
      echo "[fetch-model] WARNING sha256 mismatch:"
      echo "  expected: $EXPECTED_SHA256"
      echo "  actual:   $actual"
      rm -f "$TARGET"
      exit 1
    fi
    exit 0
  fi

  echo "[fetch-model] $repo failed; trying next mirror"
done

cat <<'EOF'

[fetch-model] FATAL: all mirrors failed.

Check:
  1. Check your network connection and free disk space.
  2. hf auth whoami        # optional, useful if public mirror URLs fail
  3. open the repo URL in a browser; if it shows "Access this model
     by accepting the terms", click accept once.
  4. retry this script.

If the model is permanently moved, edit MIRRORS in this script.
EOF
exit 1
