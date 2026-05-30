# %% [markdown]
# # Stage 3 — byte-cut quality gate for Qwen2.5-3B
#
# This is the active Colab for the byte-cut question. It gives one f16 code-PPL
# denominator, then tries AWQ W4 and, on suitable GPUs, GPTQ W3. Notebook 03/QTIP
# is deliberately not folded into this run: it needs a separate M3 trellis kernel
# and is not the compute-unit-efficient next step.
#
# **Produces:** `awq_bytecut_results.json`, plus quantized model directories for
# engines that complete.
#
# **GO/NO-GO gate:** candidate code-PPL / f16 code-PPL must be at or below:
# AWQ W4 <= 1.05, GPTQ W3 <= 1.12.
#
# **Compute-unit policy:** run this first. AWQ W4 is the cheap, stable signal. GPTQ
# W3 is skipped by default on pre-Ampere GPUs because GPTQModel may compile CUDA
# extensions there, which can burn a session before producing a quality result.

# %%
# --- 0. Dependency bootstrap + GPU/import smoke test ---
# Keep this as the first code cell. The previous version installed unbounded
# quantization packages with check=False; a partial resolver result let a broken
# numpy/scipy/sklearn stack surface later as an AutoTokenizer import error.
import importlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def run(cmd, *, check=True):
    cmd = [str(x) for x in cmd]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def pip_install(*pkgs, check=True, extra=()):
    return run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *extra, *pkgs],
        check=check,
    )


# Remove optional packages that Transformers probes but this notebook does not
# need. A broken sklearn -> scipy -> numpy chain caused the reported failure.
run([sys.executable, "-m", "pip", "uninstall", "-y", "sklearn", "scikit-learn", "scipy"], check=False)

pip_install(
    "numpy==1.26.4",
    "transformers>=4.57,<5",
    "tokenizers>=0.22,<0.23",
    "accelerate>=1.10,<2",
    "datasets>=3.2,<5",
    "huggingface_hub>=0.34,<1",
    "hf_transfer>=0.1.9",
    "safetensors>=0.5,<0.8",
    "zstandard>=0.22",
)
pip_install("autoawq==0.2.7.post3")

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > GPU."
GPU_NAME = torch.cuda.get_device_name(0)
GPU_CC = torch.cuda.get_device_capability(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"python={platform.python_version()} torch={torch.__version__}")
print(f"numpy={np.__version__} transformers={transformers.__version__}")
print(f"GPU: {GPU_NAME}  cc={GPU_CC[0]}.{GPU_CC[1]}  VRAM={VRAM_GB:.1f} GB")

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# %%
# --- 1. Config ---
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUT_AWQ = "qwen2.5-3b-awq-w4"
OUT_GPTQ = "qwen2.5-3b-gptq-w3"

N_CALIB = 256
SEQ_LEN = 1024
MAX_EVAL_TOKENS = 32768
GATE = {"awq_w4": 1.05, "gptq_w3": 1.12}

RUN_AWQ = True
RUN_GPTQ = GPU_CC[0] >= 8
ALLOW_GPTQ_SOURCE_BUILD = False
if not RUN_GPTQ:
    print("GPTQ W3 default: SKIP on cc<8.0 to avoid long CUDA extension builds.")
    print("Set RUN_GPTQ=True and ALLOW_GPTQ_SOURCE_BUILD=True if you intentionally want that.")

results = {
    "model": MODEL_ID,
    "seed": SEED,
    "gpu": {"name": GPU_NAME, "cc": f"{GPU_CC[0]}.{GPU_CC[1]}", "vram_gb": round(VRAM_GB, 1)},
    "ppl": {},
    "bits": {},
    "verdict": {},
    "errors": {},
}

# %% [markdown]
# ## Calibration + Eval Data
# The notebook uses the same small code corpora as the local oracle when possible.
# Local `/content/calib_trim.txt` and `/content/ppl_trim.txt` override the raw
# GitHub fetch. Calibration and evaluation remain disjoint.

# %%
# --- 2. Load disjoint code corpora ---
RAW = (
    "https://raw.githubusercontent.com/joshuahickscorp/dismantle/"
    "codex/maximal-spec-colab/colab/data"
)


def load_code_blocks():
    import urllib.request

    def get(name):
        local = Path("/content") / name
        if local.exists():
            print("using", local)
            return local.read_text(encoding="utf-8", errors="replace")
        with urllib.request.urlopen(f"{RAW}/{name}", timeout=30) as r:
            return r.read().decode("utf-8", "replace")

    return get("calib_trim.txt"), get("ppl_trim.txt")


CALIB_TEXT, EVAL_TEXT = load_code_blocks()
assert len(CALIB_TEXT) > 10_000, f"calibration text too small: {len(CALIB_TEXT)} chars"
assert len(EVAL_TEXT) > 10_000, f"eval text too small: {len(EVAL_TEXT)} chars"
print(f"calib chars={len(CALIB_TEXT)}  eval chars={len(EVAL_TEXT)}")

# %%
# --- 3. Tokenizer + calibration samples ---
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def make_calib(text, n, seqlen):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(ids) <= seqlen:
        return [tok.decode(ids)]
    usable = len(ids) - seqlen
    step = max(1, usable // max(1, n))
    chunks = [ids[i : i + seqlen] for i in range(0, usable + 1, step)][:n]
    return [tok.decode(c) for c in chunks]


CALIB = make_calib(CALIB_TEXT, N_CALIB, SEQ_LEN)
assert CALIB, "no calibration samples produced"
print("calib samples:", len(CALIB))

# %%
# --- 4. PPL helper ---
@torch.inference_mode()
def ppl(model, text, max_len=2048, stride=512, max_eval_tokens=MAX_EVAL_TOKENS):
    model.eval()
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    if max_eval_tokens:
        ids = ids[:, :max_eval_tokens]
    assert ids.size(1) > 1, "eval text tokenized to <=1 token"
    ids = ids.to(model.device)
    nlls, n_tok = [], 0
    prev = 0
    for beg in range(0, ids.size(1), stride):
        end = min(beg + max_len, ids.size(1))
        trg = end - prev
        inp = ids[:, beg:end]
        tgt = inp.clone()
        tgt[:, :-trg] = -100
        loss = model(inp, labels=tgt).loss
        nlls.append(loss.float() * trg)
        n_tok += trg
        prev = end
        if end == ids.size(1):
            break
    return float(torch.exp(torch.stack(nlls).sum() / n_tok))


def record_error(key, exc):
    results["errors"][key] = repr(exc)
    results["verdict"][key] = "ERROR"
    print(f"{key} FAILED:", repr(exc), flush=True)

# %%
# --- 5. f16 reference PPL (gate denominator) ---
m_f16 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
results["ppl"]["f16"] = ppl(m_f16, EVAL_TEXT)
print("f16 code PPL =", round(results["ppl"]["f16"], 4))
del m_f16
torch.cuda.empty_cache()

# %% [markdown]
# ## AWQ W4
# Stable quality baseline. This is the first result to trust because it avoids
# long extension builds and has the best chance of finishing on T4/L4.

# %%
# --- 6. AWQ 4-bit quantize + save ---
if RUN_AWQ:
    try:
        from awq import AutoAWQForCausalLM

        awq = AutoAWQForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        awq.quantize(
            tok,
            quant_config={"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"},
            calib_data=CALIB,
        )
        awq.save_quantized(OUT_AWQ)
        tok.save_pretrained(OUT_AWQ)
        results["bits"]["awq_w4"] = 4.0
        print("AWQ-W4 saved ->", OUT_AWQ)
        del awq
        torch.cuda.empty_cache()
    except Exception as e:
        record_error("awq_w4", e)
else:
    results["verdict"]["awq_w4"] = "SKIPPED"

# %%
# --- 7. AWQ W4 PPL ---
if RUN_AWQ and Path(OUT_AWQ).is_dir():
    try:
        from awq import AutoAWQForCausalLM

        m = AutoAWQForCausalLM.from_quantized(
            OUT_AWQ,
            fuse_layers=False,
            device_map="cuda",
            trust_remote_code=True,
        )
        awq_model = getattr(m, "model", m)
        results["ppl"]["awq_w4"] = ppl(awq_model, EVAL_TEXT)
        print("AWQ-W4 code PPL =", round(results["ppl"]["awq_w4"], 4))
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        record_error("awq_w4_ppl", e)
else:
    print("AWQ PPL skipped; no AWQ directory.")

# %% [markdown]
# ## GPTQ W3
# This is the byte-cut prize, but it is expensive and package-sensitive. It is
# lazily installed only after the f16/AWQ signal is safely recorded.

# %%
# --- 8. GPTQ 3-bit quantize + save ---
if RUN_GPTQ:
    try:
        if GPU_CC[0] < 8 and not ALLOW_GPTQ_SOURCE_BUILD:
            raise RuntimeError("GPTQ skipped on cc<8.0 unless ALLOW_GPTQ_SOURCE_BUILD=True")
        pip_install("gptqmodel==2.2.0", extra=("--no-build-isolation",))
        importlib.invalidate_caches()
        from gptqmodel import GPTQModel, QuantizeConfig

        qc = QuantizeConfig(bits=3, group_size=128, desc_act=True, device="cuda")
        g = GPTQModel.load(MODEL_ID, qc, trust_remote_code=True)
        g.quantize(CALIB, batch_size=1, auto_gc=True)
        g.save(OUT_GPTQ)
        tok.save_pretrained(OUT_GPTQ)
        results["bits"]["gptq_w3"] = 3.0
        print("GPTQ-W3 saved ->", OUT_GPTQ)
        del g
        torch.cuda.empty_cache()
    except Exception as e:
        record_error("gptq_w3", e)
else:
    results["verdict"]["gptq_w3"] = "SKIPPED_CC_LT_8" if GPU_CC[0] < 8 else "SKIPPED"
    print("GPTQ-W3 skipped by config.")

# %%
# --- 9. GPTQ W3 PPL ---
if RUN_GPTQ and Path(OUT_GPTQ).is_dir():
    try:
        from gptqmodel import GPTQModel

        m = GPTQModel.load(OUT_GPTQ, device="cuda", trust_remote_code=True)
        gptq_model = getattr(m, "model", m)
        results["ppl"]["gptq_w3"] = ppl(gptq_model, EVAL_TEXT)
        print("GPTQ-W3 code PPL =", round(results["ppl"]["gptq_w3"], 4))
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        record_error("gptq_w3_ppl", e)
else:
    print("GPTQ PPL skipped; no GPTQ directory.")

# %%
# --- 10. VERDICT ---
f16 = results["ppl"].get("f16")
print(f"\n{'candidate':10s} {'bits':>5s} {'PPL':>9s} {'PPL/f16':>8s} {'gate':>6s} verdict")
for k, thr in GATE.items():
    if results["verdict"].get(k, "").startswith("SKIPPED"):
        print(f"{k:10s}   --   skipped                 {results['verdict'][k]}")
        continue
    p = results["ppl"].get(k)
    if p is None or not f16:
        results["verdict"].setdefault(k, "MISSING")
        print(f"{k:10s}   --   missing                 {results['verdict'][k]}")
        continue
    ratio = p / f16
    verdict = "GO" if ratio <= thr else "NO-GO"
    results["verdict"][k] = verdict
    print(f"{k:10s} {results['bits'].get(k, 0):5.1f} {p:9.4f} {ratio:8.3f} {thr:6.2f} {verdict}")

results["local_oracle_c_reference"] = {"q4km": 4.485, "q3_imatrix": 5.915, "q3_naive": 11.432}
Path("awq_bytecut_results.json").write_text(json.dumps(results, indent=2))
print("\nsaved awq_bytecut_results.json")
print(json.dumps(results["verdict"], indent=2))

# %%
# --- 11. Package small artifacts for download ---
for d in (OUT_AWQ, OUT_GPTQ):
    if Path(d).is_dir():
        shutil.make_archive(d, "zip", d)
        print("zipped", d + ".zip")
try:
    from google.colab import files

    files.download("awq_bytecut_results.json")
except Exception:
    pass

# %% [markdown]
# ## M3 Integration Boundary
# A GO here is a **quality** verdict, not a shipped dismantle lever.
#
# 1. Convert the winner to GGUF or a dismantle-native packed format, then measure
#    perplexity on the same `ppl_trim.txt` scale as oracle C.
# 2. Implement or reuse the matching M3 packed GEMV loader/kernel.
# 3. Re-run `tools/bench/analyze_tcb_trace.py` on any decode bench.
#
# Until the GGUF/M3 round-trip passes, treat Colab PPL as a selection signal only.
