# %% [markdown]
# # Stage 3 deep — QTIP 3-bit research scaffold
#
# **Default state: DEFERRED.** This notebook is intentionally not part of the
# active compute-unit path. QTIP needs a custom M3 trellis-decode kernel before a
# Colab artifact can become a dismantle lever, and upstream QTIP is Llama-oriented.
# Run `01_awq_bytecut.ipynb` first; only revisit this if W3 quality is promising
# or Stage 2 makes the runtime clearly bandwidth-bound.
#
# This file is kept as a guarded scaffold so "Run all" does not burn a GPU session
# on a known-manual research path.

# %%
# --- 0. Guardrail + GPU preflight ---
import json
import os
import subprocess
import sys
from pathlib import Path

RUN_QTIP = False
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BITS = 3
HESS_DIR = Path("hessians_code")
CKPT_DIR = Path("qwen3b-qtip-3bit")
HF_DIR = Path("qwen3b-qtip-3bit-hf")
PPL_GATE_RATIO = 1.10
results = {
    "model": MODEL_ID,
    "bits": BITS,
    "gate_ratio": PPL_GATE_RATIO,
    "verdict": "DEFERRED",
    "reason": "QTIP is not compute-unit-efficient until AWQ/GPTQ and M3 trellis-kernel work justify it.",
}

if not RUN_QTIP:
    print("QTIP disabled by default. Set RUN_QTIP=True in this cell to use a fresh, separate GPU runtime.")
else:
    import torch

    assert torch.cuda.is_available(), "No GPU."
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("GPU:", name, f"{vram:.0f} GB")
    assert vram >= 24, "QTIP Hessian/quant is heavy; use L4/A100-class VRAM."

# %%
# --- 1. Clone QTIP + install deps (guarded) ---
if RUN_QTIP:
    if not Path("QTIP").is_dir():
        subprocess.run(["git", "clone", "https://github.com/Cornell-RelaxML/QTIP.git"], check=True)
    # QTIP pins its own research stack. Run in a fresh runtime, separate from
    # notebook 01, so scipy/numpy changes cannot poison Transformers imports.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "QTIP/requirements.txt"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "QTIP/qtip-kernels"], check=False)
    print("QTIP present:", Path("QTIP").is_dir())
else:
    print("skipped")

# %%
# --- 2. Config + code corpora fetch (guarded) ---
CALIB_TEXT = Path("/content/calib_trim.txt")
EVAL_TEXT = Path("/content/ppl_trim.txt")
if RUN_QTIP:
    import urllib.request

    raw = (
        "https://raw.githubusercontent.com/joshuahickscorp/dismantle/"
        "codex/maximal-spec-colab/colab/data"
    )
    for p in (CALIB_TEXT, EVAL_TEXT):
        if not p.exists():
            urllib.request.urlretrieve(f"{raw}/{p.name}", p)
            print("fetched", p)
else:
    print("skipped")

# %% [markdown]
# ## Known Manual Patch
# Upstream QTIP's Hessian script samples its default dataset; it does not accept
# the code corpus path used by notebook 01. For a true apples-to-apples code-PPL
# gate, patch `QTIP/quantize_llama/input_hessian_llama.py` to consume
# `/content/calib_trim.txt`, then run the guarded cells below.

# %%
# --- 3. Build Hessians (correct upstream module path, guarded) ---
if RUN_QTIP:
    os.chdir("QTIP")
    hess_cmd = [
        sys.executable,
        "-m",
        "quantize_llama.input_hessian_llama",
        "--base_model",
        MODEL_ID,
        "--save_path",
        "../" + str(HESS_DIR),
        "--devset_size",
        "256",
        "--ctx_size",
        "1024",
    ]
    print(" ".join(hess_cmd))
    r = subprocess.run(hess_cmd)
    results["hessian_rc"] = r.returncode
    os.chdir("..")
    assert r.returncode == 0, "Hessian step failed; likely Qwen2/data-loader patch needed."
else:
    results["hessian_rc"] = "skipped"
    print("skipped")

# %%
# --- 4. Quantize + HF-ize (correct upstream module paths, guarded) ---
if RUN_QTIP:
    os.chdir("QTIP")
    q_cmd = [
        sys.executable,
        "-m",
        "quantize_llama.quantize_finetune_llama",
        "--base_model",
        MODEL_ID,
        "--in_hess_path",
        "../" + str(HESS_DIR),
        "--save_path",
        "../" + str(CKPT_DIR),
        "--codebook",
        "bitshift",
        "--scale_override",
        "0.9",
        "--ft_epochs",
        "5",
        "--td_x",
        "16",
        "--td_y",
        "16",
        "--L",
        "16",
        "--K",
        str(BITS),
        "--V",
        "2",
        "--decode_mode",
        "quantlut_sym",
        "--tlut_bits",
        "9",
    ]
    print(" ".join(q_cmd))
    r = subprocess.run(q_cmd)
    results["quant_rc"] = r.returncode
    assert r.returncode == 0, "Quant step failed."
    hf_cmd = [
        sys.executable,
        "-m",
        "quantize_llama.hfize_llama",
        "--quantized_path",
        "../" + str(CKPT_DIR),
        "--hf_output_path",
        "../" + str(HF_DIR),
    ]
    print(" ".join(hf_cmd))
    r = subprocess.run(hf_cmd)
    results["hfize_rc"] = r.returncode
    os.chdir("..")
    assert r.returncode == 0, "HF conversion failed."
else:
    results["quant_rc"] = "skipped"
    results["hfize_rc"] = "skipped"
    print("skipped")

# %%
# --- 5. PPL gate (guarded) ---
if RUN_QTIP and HF_DIR.is_dir():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    @torch.inference_mode()
    def ppl(model, text, max_len=2048, stride=512, max_eval_tokens=32768):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :max_eval_tokens]
        ids = ids.to(model.device)
        nlls, n, prev = [], 0, 0
        for beg in range(0, ids.size(1), stride):
            end = min(beg + max_len, ids.size(1))
            trg = end - prev
            inp = ids[:, beg:end]
            tgt = inp.clone()
            tgt[:, :-trg] = -100
            nlls.append(model(inp, labels=tgt).loss.float() * trg)
            n += trg
            prev = end
            if end == ids.size(1):
                break
        return float(torch.exp(torch.stack(nlls).sum() / n))

    ev = EVAL_TEXT.read_text(encoding="utf-8", errors="replace")
    m_f16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    results["ppl_f16"] = ppl(m_f16, ev)
    del m_f16
    torch.cuda.empty_cache()
    m_q = AutoModelForCausalLM.from_pretrained(str(HF_DIR), torch_dtype=torch.float16, device_map="cuda")
    results["ppl_qtip"] = ppl(m_q, ev)
    del m_q
    torch.cuda.empty_cache()
else:
    results["ppl_note"] = "skipped; RUN_QTIP=False or HF_DIR missing"
    print("skipped")

# %%
# --- 6. Verdict ---
f16, q = results.get("ppl_f16"), results.get("ppl_qtip")
if f16 and q:
    ratio = q / f16
    results["ratio"] = ratio
    results["verdict"] = "GO" if ratio <= PPL_GATE_RATIO else "NO-GO"
    print(
        f"f16 PPL={f16:.3f} QTIP-{BITS}bit PPL={q:.3f} "
        f"ratio={ratio:.3f} gate<={PPL_GATE_RATIO} -> {results['verdict']}"
    )
else:
    print("QTIP not run; result is deferred.")

results["local_oracle_c_ref"] = {"q4km": 4.485, "q3_imatrix": 5.915}
Path("qtip_3bit_results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

# %% [markdown]
# ## M3 Integration Boundary
# A QTIP GO is still only a quality result. Shipping it requires a native
# trellis-decode GEMV path in Metal plus a loader format. Do not spend Colab CUs
# here until that M3 work is scheduled.
