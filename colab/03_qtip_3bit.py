# %% [markdown]
# # Stage 3 (deep) — QTIP 3-bit lookup-free trellis quant for Qwen2.5-3B
#
# **Bible axis 2, deepest byte-cut.** ~3 bits → ~99 tps wall (vs ~66 at Q4_K_M) — what
# makes triple-digit *dense* tps physically reachable. QTIP's trellis decode is
# **lookup-free + contiguous** (no gather), which is the Apple-GPU-friendly property.
#
# **Read the caveats before running — this is a SCAFFOLD, lower priority than nb 01:**
# 1. **M3 blocker:** QTIP needs a custom **trellis-decode Metal kernel** on the M3.
#    None exists; it's multi-session work. This notebook only produces the artifact +
#    quality verdict, NOT a runnable dismantle lever.
# 2. **Sequencing:** QTIP only pays off once the kernels are **bandwidth-bound**
#    (after Bible Stage 2). Its trellis decode adds compute, so axis-1 × QTIP is
#    sub-multiplicative. Do nb 01 (AWQ/GPTQ) first — it's the practical byte-cut.
# 3. **Quality floor:** a 3B degrades hard below ~3 bits. **Gate: code-PPL within ~10%
#    of f16.** If it fails, QTIP is dead for this model.
# 4. QTIP is research code (Llama-oriented); Qwen2 adaptation may be needed.
#
# **Produces:** a QTIP-quantized Qwen-3B + `qtip_3bit_results.json`. **Needs an A100/L4**
# (Hessian compute is heavy).

# %%
# --- 0. GPU check (QTIP wants real VRAM) ---
import torch, os, sys, subprocess, json
assert torch.cuda.is_available(), "No GPU."
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print("GPU:", name, f"{vram:.0f} GB")
if vram < 24:
    print("WARN: QTIP Hessian/quant is heavy; <24GB may OOM. Prefer A100/L4.")

# %%
# --- 1. Clone QTIP + deps ---
if not os.path.isdir("QTIP"):
    subprocess.run(["git", "clone", "https://github.com/Cornell-RelaxML/QTIP.git"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "accelerate", "datasets", "fast-hadamard-transform",
                "safetensors", "numpy<2.2"], check=False)
# QTIP CUDA kernels (optional for quantization; required for fast eval):
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "QTIP/qtip-kernels"],
               check=False)
print("QTIP present:", os.path.isdir("QTIP"))

# %%
# --- 2. Config ---
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BITS = 3                      # the byte-cut target
HESS_DIR = "hessians_code"
CKPT_DIR = "qwen3b-qtip-3bit"
PPL_GATE_RATIO = 1.10         # code-PPL(QTIP)/code-PPL(f16) must be <= this
CALIB_TEXT = "/content/calib_trim.txt"   # upload repo's artifacts/quant/calib_trim.txt
EVAL_TEXT  = "/content/ppl_trim.txt"     # disjoint holdout
results = {"model": MODEL_ID, "bits": BITS, "gate_ratio": PPL_GATE_RATIO}

# %% [markdown]
# ## Calibration (CODE Hessians)
# QTIP estimates per-layer input Hessians from calibration data. Use a **code**
# corpus so the quant is code-aware (upload the repo's `calib_trim.txt`). QTIP's
# scripts default to RedPajama/wikitext — point them at code instead.

# %%
# --- 3. Build Hessians on code (QTIP hessian_offline) ---
# QTIP ships hessian_offline_llama.py; Qwen2 is architecturally close. If the script
# rejects Qwen2, patch its model-loading to AutoModelForCausalLM(trust_remote_code).
os.chdir("QTIP")
src = CALIB_TEXT if os.path.exists(CALIB_TEXT) else None
hess_cmd = [sys.executable, "hessian_offline_llama.py",
            "--base_model", MODEL_ID, "--save_path", "../" + HESS_DIR,
            "--devset_size", "256", "--ctx_size", "1024"]
if src:
    hess_cmd += ["--devset_path", src]   # code calibration
print(" ".join(hess_cmd))
r = subprocess.run(hess_cmd)
results["hessian_rc"] = r.returncode
os.chdir("..")
assert r.returncode == 0, "Hessian step failed — likely Qwen2 arch patch needed (see md)."

# %%
# --- 4. Quantize (QTIP quantize_finetune, 3-bit trellis) ---
os.chdir("QTIP")
q_cmd = [sys.executable, "quantize_finetune_llama.py",
         "--base_model", MODEL_ID, "--hessian_path", "../" + HESS_DIR,
         "--save_path", "../" + CKPT_DIR,
         "--codebook", "bitshift", "--K", str(BITS), "--td_x", "16", "--td_y", "16"]
print(" ".join(q_cmd))
r = subprocess.run(q_cmd)
results["quant_rc"] = r.returncode
os.chdir("..")
assert r.returncode == 0, "Quant step failed."
print("QTIP ckpt ->", CKPT_DIR)

# %%
# --- 5. PPL gate on CODE (the decision) ---
# Reuse the HF sliding-window PPL so the f16 vs QTIP ratio is fair.
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
def ppl(model, text, max_len=2048, stride=512):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device); import torch
    nlls, n, prev = [], 0, 0
    for beg in range(0, ids.size(1), stride):
        end = min(beg+max_len, ids.size(1)); trg = end-prev
        inp = ids[:, beg:end]; tgt = inp.clone(); tgt[:, :-trg] = -100
        with torch.no_grad(): nlls.append(model(inp, labels=tgt).loss*trg)
        n += trg; prev = end
        if end == ids.size(1): break
    return float(torch.exp(torch.stack(nlls).sum()/n))
ev = open(EVAL_TEXT, encoding="utf-8", errors="replace").read() if os.path.exists(EVAL_TEXT) \
     else open(CALIB_TEXT, encoding="utf-8", errors="replace").read()[:40000]
m_f16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                             device_map="cuda", trust_remote_code=True)
results["ppl_f16"] = ppl(m_f16, ev); del m_f16; torch.cuda.empty_cache()
# Load QTIP model via its eval loader (see QTIP/eval/); falls back to noting manual step.
try:
    sys.path.insert(0, "QTIP")
    from lib.utils.unsafe_import import model_from_hf_path  # QTIP helper
    m_q, _ = model_from_hf_path(CKPT_DIR, use_cuda_graph=False)
    results["ppl_qtip"] = ppl(m_q, ev)
except Exception as e:
    results["ppl_qtip"] = None
    results["ppl_note"] = f"load QTIP model manually via QTIP eval scripts: {e!r}"

# %%
# --- 6. VERDICT ---
f16, q = results.get("ppl_f16"), results.get("ppl_qtip")
if f16 and q:
    ratio = q/f16
    results["ratio"] = ratio
    results["verdict"] = "GO" if ratio <= PPL_GATE_RATIO else "NO-GO"
    print(f"f16 PPL={f16:.3f}  QTIP-{BITS}bit PPL={q:.3f}  ratio={ratio:.3f} "
          f"(gate ≤{PPL_GATE_RATIO}) -> {results['verdict']}")
else:
    results["verdict"] = "INCOMPLETE"
    print("QTIP PPL not measured — finish via QTIP eval scripts, then apply gate.")
results["local_oracle_c_ref"] = {"q4km": 4.485, "q3_imatrix": 5.915}
json.dump(results, open("qtip_3bit_results.json", "w"), indent=2)
print(json.dumps(results, indent=2))

# %% [markdown]
# ## M3 integration boundary (the real blocker)
# A GO here is a **quality** verdict only. To run QTIP in dismantle you must write a
# **trellis-decode GEMV kernel in Metal** (lookup-free bitshift codebook → dequant →
# accumulate). There is **no prior art** in the repo — budget this with the Stage-2
# kernel work, and only after the Q4_K GEMV is bandwidth-bound (else QTIP's decode
# compute pulls you back toward compute-bound). Until then, QTIP stays a quality result.
