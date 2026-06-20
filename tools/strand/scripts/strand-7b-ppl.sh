#!/usr/bin/env bash
# strand-7b-ppl.sh — 7B-safe STRAND quantize → reconstruct → WikiText-2 PPL.
#
# This is the script the measure sweep was a warm-up for: it produces a REAL
# perplexity number for a STRAND-quantized 7B model, without the two traps in
# strand-baseline.sh / eval-7b.sh:
#
#   1. NO 28 GB float32 shard-merge. We quantize each HF shard independently
#      (a tensor never spans shards), so peak RAM is ~one shard, not the whole
#      model merged to f32.
#   2. NO dependency on the missing scratch/eval_ppl.py. The WikiText-2 PPL eval
#      is embedded here and is model-dir-agnostic (loads config+tokenizer+weights
#      straight from the reconstructed dir).
#
# The reconstructed shards are recast f32 → bf16 as they are written, so the
# recon model dir is ~the same size as the original (~15 GB), not ~31 GB. Eval runs
# in bf16 — Qwen2.5 is bf16-native and OVERFLOWS to NaN in fp16. Loading uses the
# same MPS fixes that unblocked calibration (no allocator warmup, eager attention,
# use_cache=False).
#
# ─────────────────────────────────────────────────────────────────────────────
# COST WARNING — read before launching a full run.
#   quantize-model is CPU Viterbi; cost scales with weights × 2^L (L = bits+4,
#   or bits+6 with --quality). On this machine, q4 (no --quality) is ~14 s for a
#   1.8 M-param attention tensor → the full 7B (~6.5 B quantizable params, MLP
#   tensors are 38× bigger) is on the order of HOURS, and --quality / higher bits
#   multiply that. Budget an overnight run under `caffeinate`. Validate the
#   pipeline first with the cheap smokes below.
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   ./scripts/strand-7b-ppl.sh <MODEL_DIR> [options]
#
# Options:
#   --bits N            Scalar target bits (default: 4). Used as the mp-config
#                       fallback when --mp-config is set.
#   --l N               Explicit trellis register width L (default: k+4, or k+6
#                       with --quality). Useful for iso-rate vector runs.
#   --mp-config FILE    Mixed-precision JSON (e.g. scripts/mp-5a3f.json). Sets the
#                       run label and per-tensor bits.
#   --calib FILE        HSDI calibration (default: MODEL_DIR/hessian.hsdi if present).
#   --skip-calib        Quantize unweighted (ignore any hessian.hsdi).
#   --block-hessian F   HSB2 block-Hessian calibration for scalar LDLQ.
#   --vec-dim N         Vector trellis dimension d. Payload rate is bits/d bpw.
#   --learned-codebook  Learn the integer vector LUT per tensor (requires vec-dim>1).
#   --dist D            gaussian|laplace|gennorm|empirical reconstruction LUT.
#   --affine-min MODE   Pass through affine-min mode: auto|on|off.
#   --tail-biting       Enable tail-biting.
#   --no-tail-biting    Disable explicit tail-biting.
#   --label NAME        Override result label / JSON names.
#   --quality           L=k+6 (higher quality, MUCH slower). Appends _quality.
#   --passes N          Viterbi sub-scale passes (default: 1).
#   --threads N         Quantizer threads (default: all cores).
#   --ctx N             PPL context length in tokens (default: 2048).
#   --limit-chunks N    Eval only the first N non-overlapping windows; 0 = all.
#   --device D          auto|cpu|mps for the PPL eval (default: auto).
#   --eval-dtype D      bfloat16 (default), float16, or float32 for the forward pass.
#                       Keep bfloat16 — Qwen2.5 overflows fp16 to NaN.
#   --recon-dtype D     bf16 (default), f16, or f32 for the on-disk reconstructed shards.
#   --max-shards N      Quantize only the first N shards; COPY the rest unchanged.
#                       N=0 (default) quantizes every shard (a real run). Small N
#                       is a fast plumbing smoke — the PPL number is NOT meaningful.
#   --fp16-baseline     Also eval the original (un-quantized) model as the anchor
#                       and report STRAND's PPL delta vs it. Cached per out-dir.
#   --no-quant          Skip quantization. Eval the recon dir if it exists, else
#                       (with --fp16-baseline) just the original. Use to (re)run
#                       the eval harness alone.
#   --resume            Reuse completed reconstructed shards and finished STRAND
#                       PPL JSON in --out-dir; recompute missing/partial pieces.
#   --out-dir DIR       Where recon dir + results land (default: MODEL_DIR/ppl-<label>).
#   --keep-f32-temp     Keep the per-shard f32 temp files (debug; they are large).
#
# Outputs (in --out-dir):
#   recon/                              Reconstructed model dir (config+tokenizer+fp16 shards)
#   ppl_<label>.json                    STRAND PPL result
#   ppl_fp16.json                       FP16 anchor PPL (with --fp16-baseline)
#   strand-7b-ppl-<label>.json          Run manifest
#
# Quick smokes (cheap; do these before a full run):
#   # 1) Eval harness only — loads the original 7B, 4 windows. Gives the FP16 anchor.
#   ./scripts/strand-7b-ppl.sh scratch/qwen-7b --no-quant --fp16-baseline \
#       --limit-chunks 4 --device mps
#
#   # 2) Plumbing smoke — quantize 1 shard, copy the other 3, eval 2 windows.
#   #    (PPL is meaningless here; this only proves quantize→recast→load→eval runs.)
#   ./scripts/strand-7b-ppl.sh scratch/qwen-7b --bits 4 --max-shards 1 \
#       --limit-chunks 2 --device mps
#
# Real run (hours — launch under caffeinate):
#   caffeinate -dimsu ./scripts/strand-7b-ppl.sh scratch/qwen-7b \
#       --bits 4 --fp16-baseline --device mps

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Python interpreter — must have torch + transformers + datasets + safetensors.
# Override via STRAND_PYTHON. Mirrors strand-baseline.sh's detection.
# ---------------------------------------------------------------------------
PYTHON="${STRAND_PYTHON:-}"
if [ -z "${PYTHON}" ]; then
    for _candidate in \
        python3 \
        /usr/local/bin/python3 \
        "${HOME}/.venv/bin/python3" \
        /opt/miniconda3/bin/python3; do
        if "${_candidate}" -c "import torch, transformers, safetensors, datasets" 2>/dev/null; then
            PYTHON="${_candidate}"
            break
        fi
    done
fi
if [ -z "${PYTHON}" ]; then
    echo "[strand-7b-ppl] ERROR: no Python with torch+transformers+datasets+safetensors found." >&2
    echo "[strand-7b-ppl] Set STRAND_PYTHON=/path/to/python3 and retry." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Defaults / arg parse
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 1
fi
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
fi

MODEL_DIR="$(cd "$1" && pwd)"; shift

BITS=4
L_BITS=0
MP_CONFIG=""
MP_FALLBACK=""   # base precision for tensors NO mp-rule matches; default = --bits
                 # (was hardcoded 4 — silently made every mp run uniform-4-bit)
SKIP_CALIB=0   # accepted for compatibility; calibration plumbing was removed (dead lever)
VEC_DIM=1
LEARNED_CODEBOOK=0
OUTLIER_PCT=0
OUTLIER_BITS=8
AFFINE_MIN=""
TAIL_BITING=""
LABEL_OVERRIDE=""
QUALITY=0
THREADS=""
CTX=2048
LIMIT_CHUNKS=0
DEVICE="auto"
EVAL_DTYPE="bfloat16"   # Qwen2.5 is bf16-native; fp16 OVERFLOWS to NaN. Do not default to fp16.
RECON_DTYPE="bf16"
MAX_SHARDS=0
FP16_BASELINE=0
NO_QUANT=0
NO_EVAL=0
RESUME=0
OUT_DIR=""
KEEP_F32_TEMP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --bits)          BITS="$2"; shift 2 ;;
        --l)             L_BITS="$2"; shift 2 ;;
        --mp-config)     MP_CONFIG="$2"; shift 2 ;;
        --mp-fallback)   MP_FALLBACK="$2"; shift 2 ;;
        --skip-calib)    SKIP_CALIB=1; shift ;;
        --vec-dim)       VEC_DIM="$2"; shift 2 ;;
        --learned-codebook)
                         LEARNED_CODEBOOK=1; shift ;;
        --outlier-channel) OUTLIER_PCT="$2"; shift 2 ;;
        --outlier-bits)  OUTLIER_BITS="$2"; shift 2 ;;
        --affine-min)    AFFINE_MIN="$2"; shift 2 ;;
        --tail-biting)   TAIL_BITING="on"; shift ;;
        --no-tail-biting)
                         TAIL_BITING="off"; shift ;;
        --label)         LABEL_OVERRIDE="$2"; shift 2 ;;
        --quality)       QUALITY=1; shift ;;
        --threads)       THREADS="$2"; shift 2 ;;
        --ctx)           CTX="$2"; shift 2 ;;
        --limit-chunks)  LIMIT_CHUNKS="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --eval-dtype)    EVAL_DTYPE="$2"; shift 2 ;;
        --recon-dtype)   RECON_DTYPE="$2"; shift 2 ;;
        --max-shards)    MAX_SHARDS="$2"; shift 2 ;;
        --fp16-baseline) FP16_BASELINE=1; shift ;;
        --no-quant)      NO_QUANT=1; shift ;;
        --no-eval)       NO_EVAL=1; shift ;;
        --resume)        RESUME=1; shift ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        --keep-f32-temp) KEEP_F32_TEMP=1; shift ;;
        --help|-h)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "[strand-7b-ppl] ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${RECON_DTYPE}" in
    f16|bf16|f32) ;;
    *) echo "[strand-7b-ppl] ERROR: --recon-dtype must be f16, bf16, or f32" >&2; exit 1 ;;
esac
case "${EVAL_DTYPE}" in
    bfloat16|float16|float32) ;;
    *) echo "[strand-7b-ppl] ERROR: --eval-dtype must be bfloat16, float16, or float32" >&2; exit 1 ;;
esac
if [ "${EVAL_DTYPE}" = "float16" ]; then
    echo "[strand-7b-ppl] WARNING: --eval-dtype float16 OVERFLOWS to NaN on Qwen2.5 (bf16-native model)." >&2
fi
case "${AFFINE_MIN}" in
    ""|auto|on|off) ;;
    *) echo "[strand-7b-ppl] ERROR: --affine-min must be auto, on, or off" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Resolve label, out dir, recon dir, calibration
# ---------------------------------------------------------------------------
QUALITY_SUFFIX=""
[ "${QUALITY}" -eq 1 ] && QUALITY_SUFFIX="_quality"

if [ -n "${LABEL_OVERRIDE}" ]; then
    LABEL="${LABEL_OVERRIDE}"
elif [ -n "${MP_CONFIG}" ]; then
    MP_BASE="$(basename "${MP_CONFIG}" .json)"   # mp-5a3f
    LABEL="${MP_BASE}${QUALITY_SUFFIX}"
else
    LABEL="q${BITS}${QUALITY_SUFFIX}"
    [ "${L_BITS}" -gt 0 ] && LABEL="${LABEL}_l${L_BITS}"
    [ "${VEC_DIM}" -gt 1 ] && LABEL="${LABEL}_v${VEC_DIM}"
    [ "${LEARNED_CODEBOOK}" -eq 1 ] && LABEL="${LABEL}_learned"
fi

if [ -z "${OUT_DIR}" ]; then
    OUT_DIR="${MODEL_DIR}/ppl-${LABEL}"
fi
RECON_DIR="${OUT_DIR}/recon"
mkdir -p "${RECON_DIR}"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  STRAND 7B  quantize → reconstruct → WikiText-2 PPL  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  model       : ${MODEL_DIR}"
echo "  label       : ${LABEL}"
if [ -n "${MP_CONFIG}" ]; then
    echo "  mp-config   : ${MP_CONFIG} (fallback ${MP_FALLBACK:-${BITS}} bits)"
else
    echo "  bits        : ${BITS}"
fi
echo "  quality     : $([ "${QUALITY}" -eq 1 ] && echo "yes (L=k+6)" || echo "no (L=k+4)")"
echo "  vector      : d=${VEC_DIM}  learned-codebook: $([ "${LEARNED_CODEBOOK}" -eq 1 ] && echo "yes" || echo "no")"
[ -n "${AFFINE_MIN}" ] && echo "  affine-min  : ${AFFINE_MIN}"
[ -n "${TAIL_BITING}" ] && echo "  tail-biting : ${TAIL_BITING}"
[ "${L_BITS}" -gt 0 ] && echo "  L override  : ${L_BITS}"
echo "  ctx         : ${CTX}   limit-chunks: $([ "${LIMIT_CHUNKS}" -gt 0 ] && echo "${LIMIT_CHUNKS}" || echo "all")"
echo "  eval device : ${DEVICE}   eval dtype: ${EVAL_DTYPE}"
echo "  recon dtype : ${RECON_DTYPE}   max-shards: $([ "${MAX_SHARDS}" -gt 0 ] && echo "${MAX_SHARDS} (PLUMBING SMOKE — PPL not meaningful)" || echo "all")"
echo "  resume      : $([ "${RESUME}" -eq 1 ] && echo "yes" || echo "no")"
echo "  out         : ${OUT_DIR}"
echo "  python      : ${PYTHON}"
echo ""

# ---------------------------------------------------------------------------
# Build quantizer
# ---------------------------------------------------------------------------
if [ "${NO_QUANT}" -eq 0 ]; then
    echo "[strand-7b-ppl] Building strand-quant release binary ..."
    cargo build --release --manifest-path "${REPO_ROOT}/Cargo.toml" -p strand-quant >/dev/null
fi
QUANT_BIN="${REPO_ROOT}/target/release/quantize-model"

# ---------------------------------------------------------------------------
# Discover shards (sharded HF model) or a single-file model
# ---------------------------------------------------------------------------
SHARDS=()
SINGLE_FILE=""
if [ -f "${MODEL_DIR}/model.safetensors" ] && ! ls "${MODEL_DIR}"/model-*-of-*.safetensors >/dev/null 2>&1; then
    SINGLE_FILE="${MODEL_DIR}/model.safetensors"
else
    while IFS= read -r f; do SHARDS+=("$f"); done < <(ls "${MODEL_DIR}"/model-*-of-*.safetensors 2>/dev/null | sort)
    if [ "${#SHARDS[@]}" -eq 0 ]; then
        echo "[strand-7b-ppl] ERROR: no model.safetensors and no shards in ${MODEL_DIR}" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Embedded helper: recast a quantizer f32 safetensors → fp16 (or copy f32).
#   argv: <src f32 safetensors> <dst safetensors> <f16|f32>
# ---------------------------------------------------------------------------
recast_shard() {
    local src="$1" dst="$2" dt="$3"
    "${PYTHON}" - "${src}" "${dst}" "${dt}" <<'RECAST_PY'
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst, dt = sys.argv[1], sys.argv[2], sys.argv[3]
target = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[dt]
tensors = {}
with safe_open(src, framework="pt") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        if t.dtype == torch.float32 and target != torch.float32:
            t = t.to(target)
        tensors[k] = t.contiguous()
save_file(tensors, dst, metadata={"format": "pt"})
print(f"[recast] {dst}: {len(tensors)} tensors -> {dt}", flush=True)
RECAST_PY
}

# ---------------------------------------------------------------------------
# Embedded PPL eval: WikiText-2 test, non-overlapping ctx windows, exp(Σnll/Σtok).
#   argv: <load_dir> <ctx> <limit_chunks> <device> <eval_dtype> <tag> <out_json>
# ---------------------------------------------------------------------------
eval_ppl() {
    local load_dir="$1" tag="$2" out_json="$3"
    "${PYTHON}" - "${load_dir}" "${CTX}" "${LIMIT_CHUNKS}" "${DEVICE}" "${EVAL_DTYPE}" "${tag}" "${out_json}" <<'PPL_PY'
import sys, json, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dir, ctx_s, limit_s, device_s, dtype_s, tag, out_json = sys.argv[1:8]
ctx = int(ctx_s); limit = int(limit_s)

if device_s == "auto":
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(device_s)

torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_s]
attn_impl = "eager" if device.type == "mps" else None

print(f"[ppl] loading '{load_dir}' on {device} ({dtype_s})", flush=True)
tok = AutoTokenizer.from_pretrained(load_dir, trust_remote_code=True)
load_kwargs = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True, trust_remote_code=True)
if attn_impl:
    load_kwargs["attn_implementation"] = attn_impl

if device.type == "mps":
    # Same fix that unblocked calibration: the Transformers device_map path does a
    # single allocator warmup sized to the whole model, which fails on Apple MPS for
    # 7B even when tensor-by-tensor loading fits. Disable only that warmup.
    import transformers.modeling_utils as mu
    mu.caching_allocator_warmup = lambda *a, **k: None
    model = AutoModelForCausalLM.from_pretrained(load_dir, device_map=str(device), **load_kwargs)
elif device.type == "cpu":
    model = AutoModelForCausalLM.from_pretrained(load_dir, **load_kwargs)
else:
    model = AutoModelForCausalLM.from_pretrained(load_dir, device_map=str(device), **load_kwargs)
model.eval()
if hasattr(model, "config"):
    model.config.use_cache = False

from datasets import load_dataset
# WikiText-2 load, robust across huggingface_hub versions. Older hubs accept the
# bare legacy id "wikitext" (and it may be cached locally → no network, identical
# PPL); newer hubs REQUIRE a "namespace/name" repo id and raise HfUriError on the
# bare name, so fall back to the canonical "Salesforce/wikitext" mirror (same
# wikitext-2-raw-v1 test split → directly comparable perplexities).
test = None
_ppl_errs = []
for _ds_id in ("wikitext", "Salesforce/wikitext"):
    try:
        test = load_dataset(_ds_id, "wikitext-2-raw-v1", split="test")
        break
    except Exception as _e:
        _ppl_errs.append(f"{_ds_id}: {type(_e).__name__}: {_e}")
if test is None:
    raise SystemExit("[ppl] WikiText-2 load failed for all candidate ids:\n  "
                     + "\n  ".join(_ppl_errs))
enc = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[0]
n_chunks = enc.shape[0] // ctx
if limit > 0:
    n_chunks = min(n_chunks, limit)
if n_chunks == 0:
    raise SystemExit(f"[ppl] ctx={ctx} too large for {enc.shape[0]} tokens")

loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
nll = 0.0
ntok = 0
t0 = time.time()
with torch.no_grad():
    for i in range(n_chunks):
        ids = enc[i * ctx:(i + 1) * ctx].unsqueeze(0).to(device)
        logits = model(ids, use_cache=False).logits
        shift_logits = logits[:, :-1, :].float().reshape(-1, logits.size(-1))
        shift_labels = ids[:, 1:].reshape(-1)
        nll += loss_fct(shift_logits, shift_labels).item()
        ntok += shift_labels.numel()
        print(f"[ppl] {i+1}/{n_chunks}  ppl={math.exp(nll/ntok):.4f}  ({time.time()-t0:.0f}s)", flush=True)

ppl = math.exp(nll / ntok)
res = {"tag": tag, "ppl": ppl, "ctx": ctx, "chunks": n_chunks, "tokens": ntok,
       "device": str(device), "dtype": dtype_s, "model": load_dir}
print("RESULT_JSON " + json.dumps(res), flush=True)
if out_json:
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
PPL_PY
}

# ---------------------------------------------------------------------------
# Step 1: quantize + reconstruct (unless --no-quant)
# ---------------------------------------------------------------------------
copy_aux_files() {
    for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
             vocab.json merges.txt special_tokens_map.json added_tokens.json \
             model.safetensors.index.json; do
        if [ -f "${MODEL_DIR}/${f}" ]; then
            cp -f "${MODEL_DIR}/${f}" "${RECON_DIR}/${f}"
        fi
    done
}

valid_safetensors() {
    local path="$1"
    [ -s "${path}" ] || return 1
    "${PYTHON}" - "${path}" >/dev/null 2>&1 <<'VALID_ST_PY'
import sys
from safetensors import safe_open
try:
    with safe_open(sys.argv[1], framework="pt") as f:
        if not list(f.keys()):
            raise RuntimeError("empty safetensors")
except Exception:
    raise SystemExit(1)
VALID_ST_PY
}

valid_json() {
    local path="$1"
    [ -s "${path}" ] || return 1
    "${PYTHON}" - "${path}" >/dev/null 2>&1 <<'VALID_JSON_PY'
import json, sys
try:
    json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)
VALID_JSON_PY
}

quant_flags() {
    local args=()
    if [ -n "${MP_CONFIG}" ]; then
        args+=(--bits "${MP_FALLBACK:-${BITS}}" --mp-config "${MP_CONFIG}")
    else
        args+=(--bits "${BITS}")
    fi
    [ "${L_BITS}" -gt 0 ] && args+=(--l "${L_BITS}")
    [ "${VEC_DIM}" -gt 1 ] && args+=(--vec-dim "${VEC_DIM}")
    [ "${LEARNED_CODEBOOK}" -eq 1 ] && args+=(--learned-codebook)
    [ "${OUTLIER_PCT}" != "0" ] && args+=(--outlier-channel "${OUTLIER_PCT}" --outlier-bits "${OUTLIER_BITS}")
    [ -n "${AFFINE_MIN}" ] && args+=(--affine-min "${AFFINE_MIN}")
    [ "${TAIL_BITING}" = "on" ] && args+=(--tail-biting)
    [ "${TAIL_BITING}" = "off" ] && args+=(--no-tail-biting)
    [ "${QUALITY}" -eq 1 ] && args+=(--quality)
    [ -n "${THREADS}" ] && args+=(--threads "${THREADS}")
    printf '%s\n' "${args[@]}"
}

if [ "${NO_QUANT}" -eq 0 ]; then
    echo "━━━ Step 1: quantize + reconstruct ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    copy_aux_files

    QFLAGS=()
    while IFS= read -r a; do QFLAGS+=("$a"); done < <(quant_flags)

    if [ -n "${SINGLE_FILE}" ]; then
        TMP_F32="${OUT_DIR}/.tmp-recon.safetensors"
        dst="${RECON_DIR}/model.safetensors"
        sidecar="${RECON_DIR}/model.safetensors.json"
        if [ "${RESUME}" -eq 1 ] && valid_safetensors "${dst}" && valid_json "${sidecar}"; then
            echo "[strand-7b-ppl] single-file model: RESUME reuse ${dst}"
        else
            echo "[strand-7b-ppl] quantizing single-file model ..."
            "${QUANT_BIN}" --in "${SINGLE_FILE}" --out "${TMP_F32}" "${QFLAGS[@]}"
            cp -f "${TMP_F32}.json" "${sidecar}" 2>/dev/null || true
            recast_shard "${TMP_F32}" "${dst}" "${RECON_DTYPE}"
            [ "${KEEP_F32_TEMP}" -eq 0 ] && rm -f "${TMP_F32}"
        fi
    else
        N_SHARDS="${#SHARDS[@]}"
        idx=0
        SHARD_JOBS="${SHARD_JOBS:-1}"
        echo "[strand-7b-ppl] shard concurrency: SHARD_JOBS=${SHARD_JOBS} (independent shards quantized in parallel)"
        _quant_one_shard() {
            local idx="$1" shard="$2"
            local base dst sidecar TMP_F32
            base="$(basename "${shard}")"
            dst="${RECON_DIR}/${base}"
            sidecar="${RECON_DIR}/${base}.json"
            if [ "${MAX_SHARDS}" -gt 0 ] && [ "${idx}" -gt "${MAX_SHARDS}" ]; then
                if [ "${RESUME}" -eq 1 ] && valid_safetensors "${dst}"; then
                    echo "[strand-7b-ppl] shard ${idx}/${N_SHARDS} ${base}: RESUME reuse copied shard"
                else
                    echo "[strand-7b-ppl] shard ${idx}/${N_SHARDS} ${base}: COPY (un-quantized, smoke mode)"
                    cp -f "${shard}" "${dst}"
                fi
                return 0
            fi
            if [ "${RESUME}" -eq 1 ] && valid_safetensors "${dst}" && valid_json "${sidecar}"; then
                echo "[strand-7b-ppl] shard ${idx}/${N_SHARDS} ${base}: RESUME reuse reconstructed shard"
                return 0
            fi
            echo "[strand-7b-ppl] shard ${idx}/${N_SHARDS} ${base}: quantize -> recast(${RECON_DTYPE})"
            TMP_F32="${OUT_DIR}/.tmp-${base}"
            "${QUANT_BIN}" --in "${shard}" --out "${TMP_F32}" "${QFLAGS[@]}"
            cp -f "${TMP_F32}.json" "${sidecar}" 2>/dev/null || true
            recast_shard "${TMP_F32}" "${dst}" "${RECON_DTYPE}"
            [ "${KEEP_F32_TEMP}" -eq 0 ] && rm -f "${TMP_F32}"
        }
        for shard in "${SHARDS[@]}"; do
            idx=$((idx + 1))
            _quant_one_shard "${idx}" "${shard}" &
            while [ "$(jobs -rp | wc -l)" -ge "${SHARD_JOBS}" ]; do wait -n 2>/dev/null || sleep 3; done
        done
        wait
    fi
    echo ""
    echo "[strand-7b-ppl] reconstructed model dir: ${RECON_DIR}"
    du -sh "${RECON_DIR}" 2>/dev/null | sed 's/^/  size: /' || true
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: FP16 anchor PPL (optional, cached)
# ---------------------------------------------------------------------------
FP16_JSON="${OUT_DIR}/ppl_baseline.json"
FP16_PPL=""
if [ "${FP16_BASELINE}" -eq 1 ] && [ "${NO_EVAL}" -eq 0 ]; then
    echo "━━━ Step 2: baseline (${EVAL_DTYPE}) anchor PPL ━━━━━━━━━━━━━━━"
    if [ -f "${FP16_JSON}" ]; then
        echo "[strand-7b-ppl] reusing cached baseline PPL: ${FP16_JSON}"
    else
        eval_ppl "${MODEL_DIR}" "baseline" "${FP16_JSON}"
    fi
    FP16_PPL="$("${PYTHON}" -c "import json;print(f\"{json.load(open('${FP16_JSON}'))['ppl']:.4f}\")" 2>/dev/null || echo "")"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 3: STRAND recon PPL
# ---------------------------------------------------------------------------
STRAND_JSON="${OUT_DIR}/ppl_${LABEL}.json"
STRAND_PPL=""
if [ -d "${RECON_DIR}" ] && ls "${RECON_DIR}"/*.safetensors >/dev/null 2>&1; then
    echo "━━━ Step 3: STRAND ${LABEL} PPL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ "${RESUME}" -eq 1 ] && valid_json "${STRAND_JSON}"; then
        echo "[strand-7b-ppl] reusing cached STRAND PPL: ${STRAND_JSON}"
    elif [ "${NO_EVAL}" -eq 1 ]; then
        echo "[strand-7b-ppl] --no-eval: quant/recon done, PPL eval skipped (resume later evals it)"
    else
        eval_ppl "${RECON_DIR}" "${LABEL}" "${STRAND_JSON}"
    fi
    STRAND_PPL="$("${PYTHON}" -c "import json;print(f\"{json.load(open('${STRAND_JSON}'))['ppl']:.4f}\")" 2>/dev/null || echo "")"
    echo ""
elif [ "${NO_QUANT}" -eq 1 ] && [ "${FP16_BASELINE}" -eq 0 ]; then
    echo "[strand-7b-ppl] --no-quant and no recon dir present — nothing to eval." >&2
fi

# Effective bpw from the per-shard sidecars (weighted by quantized weights).
EFF_BPW="$("${PYTHON}" - "${RECON_DIR}" <<'BPW_PY'
import sys, glob, json, os
recon = sys.argv[1]
num = 0.0; den = 0
for j in glob.glob(os.path.join(recon, "*.safetensors.json")):
    try:
        d = json.load(open(j))
        agg = d.get("aggregate", {})
        n = agg.get("quantized_weights", 0)
        bpw = agg.get("effective_bpw", 0.0)
        num += bpw * n; den += n
    except Exception:
        pass
print(f"{num/den:.4f}" if den else "?")
BPW_PY
)"

# ---------------------------------------------------------------------------
# Step 4: results
# ---------------------------------------------------------------------------
echo "━━━ Results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "  %-26s  %9s  %9s  %s\n" "Config" "PPL" "eff.bpw" "vs base"
printf "  %-26s  %9s  %9s  %s\n" "--------------------------" "---------" "---------" "--------"
if [ -n "${FP16_PPL}" ]; then
    printf "  %-26s  %9s  %9s  %s\n" "baseline (${EVAL_DTYPE})" "${FP16_PPL}" "16.00" "ref"
fi
if [ -n "${STRAND_PPL}" ]; then
    VS="?"
    if [ -n "${FP16_PPL}" ]; then
        VS="$("${PYTHON}" -c "p=${STRAND_PPL};r=${FP16_PPL};d=(p-r)/r*100;print(f'{\"+\" if d>=0 else \"\"}{d:.1f}%')" 2>/dev/null || echo "?")"
    fi
    printf "  %-26s  %9s  %9s  %s\n" "STRAND ${LABEL}" "${STRAND_PPL}" "${EFF_BPW}" "${VS}"
fi
echo ""
echo "  NOTE: a Q4_K_M comparison needs the SAME WikiText-2/ctx harness run on a"
echo "        llama.cpp Q4_K_M GGUF of THIS 7B model. The 0.5B baselines in"
echo "        strand-baseline.sh (Q4_K_M=14.67) do NOT apply at 7B — do not reuse them."
echo ""

# ---------------------------------------------------------------------------
# Step 5: manifest
# ---------------------------------------------------------------------------
MANIFEST="${OUT_DIR}/strand-7b-ppl-${LABEL}.json"
"${PYTHON}" - <<MANIFEST_PY
import json, os
def load(p):
    try: return json.load(open(p))
    except Exception: return None
m = {
    "model_dir": "${MODEL_DIR}",
    "label": "${LABEL}",
    "settings": {
        "bits": ${BITS},
        "l": ${L_BITS},
        "mp_config": "${MP_CONFIG}" or None,
        "mp_fallback": "${MP_FALLBACK}" or None,
        "quality": bool(${QUALITY}),
        "vec_dim": ${VEC_DIM},
        "learned_codebook": bool(${LEARNED_CODEBOOK}),
        "affine_min": "${AFFINE_MIN}" or None,
        "tail_biting": "${TAIL_BITING}" or None,
        "ctx": ${CTX},
        "limit_chunks": ${LIMIT_CHUNKS},
        "device": "${DEVICE}",
        "recon_dtype": "${RECON_DTYPE}",
        "max_shards": ${MAX_SHARDS},
        "resume": bool(${RESUME}),
    },
    "effective_bpw": ("${EFF_BPW}" if "${EFF_BPW}" != "?" else None),
    "strand_commit": os.popen("git -C ${REPO_ROOT} rev-parse --short HEAD").read().strip(),
    "eval_dtype": "${EVAL_DTYPE}",
    "baseline": load("${FP16_JSON}"),
    "strand": load("${STRAND_JSON}"),
}
json.dump(m, open("${MANIFEST}", "w"), indent=2)
print(f"[strand-7b-ppl] manifest: ${MANIFEST}")
MANIFEST_PY

echo ""
echo "[strand-7b-ppl] done."
