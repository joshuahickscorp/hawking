#!/usr/bin/env bash
# Download Mixtral 8x7B Instruct v0.1 Q4_K_M GGUF for dismantle's
# memory-differentiation demo. ~26 GB on disk.
#
# Idempotent: skips download if file already exists and the sha256
# matches the pinned hash below (after first verified download).
#
# Usage:
#   ./tools/fetch-mixtral.sh
#
# Output:
#   models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
#
# Disk: ~26 GB free required. Network: ~26 GB download, 15-60 min depending on speed.
#
# After download, run with expert offloading to fit in 18 GB RAM:
#   dismantle generate --weights models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
#       --prompt "Once upon a time" --max-new-tokens 32 \
#       --max-routed-expert-ram-mb 8000

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
TARGET="$MODELS_DIR/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"

# Mirror order: TheBloke's repo is the canonical Q4_K_M; bartowski's is
# a maintained alternative. Both are public + ungated.
MIRRORS=(
  "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF|mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
  "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF|Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf"
)

# Pin the sha256 here once a successful download produces a known-good
# file. Leave as UNVERIFIED until first run, then update with the printed hash.
EXPECTED_SHA256="UNVERIFIED-PIN-AFTER-FIRST-DOWNLOAD"

mkdir -p "$MODELS_DIR"

# ---- Disk space precheck (require 30 GB free as safety margin) ------

free_gb=$(df -g "$MODELS_DIR" | awk 'NR==2 {print $4}')
if [[ "$free_gb" -lt 30 ]]; then
  echo "[fetch-mixtral] FATAL: only ${free_gb} GB free on the target volume."
  echo "[fetch-mixtral] Mixtral 8x7B Q4_K_M is ~26 GB; want 30 GB headroom."
  echo "[fetch-mixtral] Free up space and re-run."
  exit 1
fi
echo "[fetch-mixtral] ${free_gb} GB free; proceeding."

# ---- Idempotent skip on existing file -------------------------------

if [[ -f "$TARGET" ]]; then
  echo "[fetch-mixtral] $TARGET already exists"
  if [[ "$EXPECTED_SHA256" != "UNVERIFIED-PIN-AFTER-FIRST-DOWNLOAD" ]]; then
    actual=$(shasum -a 256 "$TARGET" | awk '{print $1}')
    if [[ "$actual" == "$EXPECTED_SHA256" ]]; then
      echo "[fetch-mixtral] sha256 OK ($actual)"
      exit 0
    else
      echo "[fetch-mixtral] WARNING sha256 mismatch:"
      echo "  expected: $EXPECTED_SHA256"
      echo "  actual:   $actual"
      echo "  delete the file and re-run for fresh download"
      exit 1
    fi
  else
    echo "[fetch-mixtral] no pinned sha256 yet; using existing file as-is"
    echo "[fetch-mixtral] paste this into EXPECTED_SHA256 to lock it:"
    shasum -a 256 "$TARGET"
    exit 0
  fi
fi

# ---- Try each mirror ------------------------------------------------

echo "[fetch-mixtral] target: $TARGET (~26 GB)"
if command -v hf >/dev/null 2>&1 && hf auth whoami >/dev/null 2>&1; then
  echo "[fetch-mixtral] logged in as: $(hf auth whoami 2>/dev/null | head -1)"
else
  echo "[fetch-mixtral] hf auth unavailable; trying public mirror URLs"
fi

for mirror_spec in "${MIRRORS[@]}"; do
  repo="${mirror_spec%%|*}"
  filename="${mirror_spec##*|}"
  echo "[fetch-mixtral] trying $repo / $filename"

  if command -v hf >/dev/null 2>&1 && hf auth whoami >/dev/null 2>&1 && \
      hf download "$repo" "$filename" --local-dir "$MODELS_DIR" 2>&1 | tail -20; then
    src="$MODELS_DIR/$filename"
    if [[ -f "$src" ]]; then
      mv "$src" "$TARGET"
      echo "[fetch-mixtral] download complete"
      echo "[fetch-mixtral] sha256:"
      shasum -a 256 "$TARGET"
      echo
      echo "[fetch-mixtral] paste the hash above into EXPECTED_SHA256 in this"
      echo "[fetch-mixtral] script to lock the version for future runs."
      exit 0
    fi
  fi

  if curl -L -C - --fail --progress-bar \
      -o "$TARGET" \
      "https://huggingface.co/$repo/resolve/main/$filename"; then
    echo "[fetch-mixtral] download complete"
    actual=$(shasum -a 256 "$TARGET" | awk '{print $1}')
    echo "[fetch-mixtral] sha256: $actual"
    if [[ "$EXPECTED_SHA256" != "UNVERIFIED-PIN-AFTER-FIRST-DOWNLOAD" && "$actual" != "$EXPECTED_SHA256" ]]; then
      echo "[fetch-mixtral] WARNING sha256 mismatch:"
      echo "  expected: $EXPECTED_SHA256"
      echo "  actual:   $actual"
      rm -f "$TARGET"
      exit 1
    fi
    exit 0
  fi

  echo "[fetch-mixtral] $repo failed; trying next mirror"
done

cat <<'EOF'

[fetch-mixtral] FATAL: all mirrors failed.

Check:
  1. Network connection and free disk space (need ~30 GB free).
  2. hf auth whoami        # optional, useful if public mirror URLs fail
  3. Open the repo URL in a browser; if it shows "Access this model by
     accepting the terms", click accept once.
  4. Retry this script. Downloads are resumable (-C - flag).

If the model is permanently moved, edit MIRRORS in this script.
EOF
exit 1
