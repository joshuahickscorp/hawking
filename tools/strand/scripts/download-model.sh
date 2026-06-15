#!/usr/bin/env bash
# download-model.sh — fast HuggingFace model downloader.
#
# Uses two speed levers simultaneously:
#   1. hf_transfer (Rust-backed multi-part HTTP, 3-5x faster per file)
#   2. max_workers (parallel file downloads for sharded models)
#
# Usage:
#   ./scripts/download-model.sh <HF_MODEL_ID> [--out <dir>] [--workers N]
#                               [--branch <revision>] [--token <hf_token>]
#                               [--include "*.safetensors"] [--exclude "*.bin"]
#                               [--backend auto|xet|http]
#                               [--xet-range-gets N] [--sequential-write]
#
# Examples:
#   ./scripts/download-model.sh meta-llama/Llama-3.1-8B
#   ./scripts/download-model.sh Qwen/Qwen2.5-7B-Instruct --out scratch/qwen-7b
#   ./scripts/download-model.sh meta-llama/Llama-3.1-8B --token hf_xxx
#   ./scripts/download-model.sh bigscience/bloom --workers 16 --include "*.safetensors"
#   ./scripts/download-model.sh Qwen/Qwen2.5-7B --backend xet --xet-range-gets 64
#   ./scripts/download-model.sh Qwen/Qwen2.5-7B --backend http
#
# Speed notes:
#   Hugging Face Hub now prefers hf-xet. If Xet is slow on a given network,
#   retry with --backend http to disable it and use the standard Hub path.
#
# Auth:
#   Gated models (Llama, Gemma, etc.) require a HF token. Either:
#     - Set HF_TOKEN env var before running
#     - Pass --token hf_xxx
#     - Run `huggingface-cli login` once (token stored in ~/.cache/huggingface)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Python (reuse strand-baseline detection)
# ---------------------------------------------------------------------------
PYTHON="${STRAND_PYTHON:-}"
if [ -z "${PYTHON}" ]; then
    for _c in python3 \
               "${HOME}/Downloads/dismantle/.venv/bin/python3" \
               "${HOME}/.venv/bin/python3" \
               /opt/miniconda3/bin/python3 \
               /usr/local/bin/python3; do
        if "${_c}" -c "import huggingface_hub" 2>/dev/null; then
            PYTHON="${_c}"; break
        fi
    done
fi
if [ -z "${PYTHON}" ]; then
    echo "[download-model] ERROR: no Python with huggingface_hub found." >&2
    echo "[download-model] pip install huggingface_hub  (and optionally hf_transfer)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <HF_MODEL_ID> [--out <dir>] [--workers N] [--branch <rev>]" >&2
    echo "          [--token <hf_token>] [--include <glob>] [--exclude <glob>]" >&2
    echo "          [--backend auto|xet|http] [--xet-range-gets N] [--sequential-write]" >&2
    exit 1
fi

MODEL_ID="$1"; shift
OUT_DIR=""
WORKERS=8
BRANCH=""
TOKEN="${HF_TOKEN:-}"
INCLUDE_PATTERNS=()
EXCLUDE_PATTERNS=()
REQUESTED_BACKEND="${STRAND_HF_BACKEND:-auto}"
XET_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-}"
XET_SEQUENTIAL_WRITE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --out)      OUT_DIR="$2";                    shift 2 ;;
        --workers)  WORKERS="$2";                    shift 2 ;;
        --branch)   BRANCH="$2";                     shift 2 ;;
        --token)    TOKEN="$2";                      shift 2 ;;
        --include)  INCLUDE_PATTERNS+=("$2");        shift 2 ;;
        --exclude)  EXCLUDE_PATTERNS+=("$2");        shift 2 ;;
        --backend)  REQUESTED_BACKEND="$2";          shift 2 ;;
        --xet-range-gets)
                    XET_RANGE_GETS="$2";             shift 2 ;;
        --sequential-write)
                    XET_SEQUENTIAL_WRITE=1;           shift ;;
        *) echo "[download-model] Unknown option: $1" >&2; exit 1 ;;
    esac
done

case "${REQUESTED_BACKEND}" in
    auto|xet|http|standard) ;;
    *) echo "[download-model] ERROR: --backend must be auto, xet, or http." >&2; exit 1 ;;
esac
if [ -n "${XET_RANGE_GETS}" ] && ! [[ "${XET_RANGE_GETS}" =~ ^[0-9]+$ ]]; then
    echo "[download-model] ERROR: --xet-range-gets must be an integer." >&2
    exit 1
fi

# Default output dir: scratch/<model_name>
if [ -z "${OUT_DIR}" ]; then
    MODEL_NAME="$(basename "${MODEL_ID}")"
    OUT_DIR="${REPO_ROOT}/scratch/${MODEL_NAME,,}"
fi
mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Install hf_transfer if missing (silent if already present)
# ---------------------------------------------------------------------------
# Install / detect fast-transfer backend (Xet preferred, hf_transfer fallback)
# ---------------------------------------------------------------------------
FAST_MODE="standard"
DISABLE_XET=0
if [ "${REQUESTED_BACKEND}" = "http" ] || [ "${REQUESTED_BACKEND}" = "standard" ]; then
    DISABLE_XET=1
elif [ "${REQUESTED_BACKEND}" = "xet" ]; then
    if "${PYTHON}" -c "import hf_xet" 2>/dev/null || \
       ("${PYTHON}" -m pip install -q "huggingface_hub[xet]" 2>/dev/null && \
        "${PYTHON}" -c "import hf_xet" 2>/dev/null); then
        FAST_MODE="xet"
    else
        echo "[download-model] ERROR: --backend xet requested, but hf_xet could not be installed." >&2
        exit 1
    fi
else
    if "${PYTHON}" -c "import hf_xet" 2>/dev/null; then
        FAST_MODE="xet"
    elif "${PYTHON}" -m pip install -q "huggingface_hub[xet]" 2>/dev/null && \
         "${PYTHON}" -c "import hf_xet" 2>/dev/null; then
        FAST_MODE="xet"
    elif "${PYTHON}" -c "import hf_transfer" 2>/dev/null; then
        FAST_MODE="hf_transfer"
    elif "${PYTHON}" -m pip install -q hf_transfer 2>/dev/null && \
         "${PYTHON}" -c "import hf_transfer" 2>/dev/null; then
        FAST_MODE="hf_transfer"
    fi
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  STRAND model download                               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  model   : ${MODEL_ID}"
echo "  out     : ${OUT_DIR}"
echo "  workers : ${WORKERS} (parallel files)"
echo "  backend : ${REQUESTED_BACKEND} -> ${FAST_MODE}"
if [ "${FAST_MODE}" = "xet" ] && [ -n "${XET_RANGE_GETS}" ]; then
    echo "  xet gets: ${XET_RANGE_GETS} concurrent range GETs/file"
fi
if [ "${XET_SEQUENTIAL_WRITE}" -eq 1 ]; then
    echo "  writes  : sequential"
fi
if [ -n "${BRANCH}" ];   then echo "  branch  : ${BRANCH}"; fi
if [ -n "${TOKEN}" ];    then echo "  auth    : token set"; fi
echo ""

# ---------------------------------------------------------------------------
# Download via Python (snapshot_download with fast transfer + max_workers)
# ---------------------------------------------------------------------------
"${PYTHON}" - <<PYEOF
import os, sys, time

# Enable fast transfer backend — Xet is the current HF-recommended path;
# hf_transfer is the legacy fallback.  Both suppress the deprecation warning.
_fast_mode = "${FAST_MODE}"
_disable_xet = "${DISABLE_XET}" == "1"
_xet_range_gets = "${XET_RANGE_GETS}"
_xet_sequential_write = "${XET_SEQUENTIAL_WRITE}" == "1"
if _disable_xet:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
if _fast_mode == "xet":
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    if _xet_range_gets:
        os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = _xet_range_gets
    if _xet_sequential_write:
        os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "1"
elif _fast_mode == "hf_transfer":
    import warnings, hf_transfer  # noqa: F401 — import activates it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

model_id      = "${MODEL_ID}"
local_dir     = "${OUT_DIR}"
workers       = ${WORKERS}
revision      = "${BRANCH}" or None
token         = "${TOKEN}" or None

include_pats  = [p.strip() for p in """${INCLUDE_PATTERNS[*]:-}""".split() if p.strip()] or None
exclude_pats  = [p.strip() for p in """${EXCLUDE_PATTERNS[*]:-}""".split() if p.strip()] or None

print(f"[download-model] Starting snapshot_download ...", flush=True)
t0 = time.time()

try:
    path = snapshot_download(
        repo_id          = model_id,
        local_dir        = local_dir,
        revision         = revision,
        token            = token,
        allow_patterns   = include_pats,
        ignore_patterns  = exclude_pats,
        max_workers      = workers,
    )
except RepositoryNotFoundError:
    print(f"[download-model] ERROR: model '{model_id}' not found or is gated.", file=sys.stderr)
    print(f"[download-model] If it's a gated model (Llama, Gemma, etc.):", file=sys.stderr)
    print(f"[download-model]   1. Accept the license at https://huggingface.co/{model_id}", file=sys.stderr)
    print(f"[download-model]   2. Pass --token hf_xxx  or set HF_TOKEN env var", file=sys.stderr)
    sys.exit(1)
except HfHubHTTPError as e:
    if "401" in str(e) or "403" in str(e):
        print(f"[download-model] ERROR: authentication required for '{model_id}'.", file=sys.stderr)
        print(f"[download-model]   Pass --token hf_xxx  or run: huggingface-cli login", file=sys.stderr)
        sys.exit(1)
    raise

elapsed = time.time() - t0

# Report downloaded files and total size.
import pathlib
files = sorted(pathlib.Path(local_dir).rglob("*"))
total_bytes = sum(f.stat().st_size for f in files if f.is_file())
n_files = sum(1 for f in files if f.is_file())

print(f"", flush=True)
print(f"[download-model] Done in {elapsed:.0f}s", flush=True)
print(f"[download-model] {n_files} files, {total_bytes/1e9:.1f} GB → {local_dir}", flush=True)
print(f"[download-model] Average speed: {total_bytes/elapsed/1e6:.0f} MB/s", flush=True)

# Print the run command.
model_name = "${MODEL_ID}".split("/")[-1].lower()
print(f"", flush=True)
print(f"[download-model] Ready to quantize:", flush=True)
print(f"  ./scripts/strand-7b-ppl.sh {local_dir} --bits 3 --quality   # or ./scripts/strand-overnight.sh for the full gate", flush=True)
PYEOF
