# %% [markdown]
# # Stage 3 — sub-4-bit byte-cut (AWQ-4bit + GPTQ-3bit) for Qwen2.5-3B
#
# **Bible axis 2 / silicon #16-#17.** Oracle C (local) proved the byte-cut prize is
# real but needs *smart* quant: code-imatrix Q3 hit PPL 5.91 vs naive Q3's 11.43
# (Q4_K_M baseline 4.49). This notebook produces the smart-quant artifacts from the
# **f16** source (not requant-from-Q4, so quality should beat the local numbers) and
# gates them on **code** perplexity.
#
# **Produces:** `qwen2.5-3b-awq-w4` and `qwen2.5-3b-gptq-w3` model dirs (+ a zip).
# **GO/NO-GO gate (baked in below):** a candidate is GO if its code-PPL ratio to f16
# is ≤ the threshold (W4 ≤1.05, W3 ≤1.12). W3 passing is the prize (≈25% fewer bytes
# than Q4_K_M at usable quality).
# **Runtime:** ~20-40 min on a T4/L4; A100 faster. **Needs a GPU runtime.**
#
# **Integration boundary (read before trusting):** these are HF-format packed-int
# checkpoints. dismantle loads GGUF Q4_K_M. Getting a winner into dismantle is a
# separate M3 task — see the final cell. This notebook's job is the **quality verdict**.

# %%
# --- 0. GPU + environment check (fail fast) ---
import sys, subprocess, json, os, math, random
import torch
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > GPU."
print("GPU:", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("torch", torch.__version__)
SEED = 0
random.seed(SEED); torch.manual_seed(SEED)

# %%
# --- 1. Pinned deps. autoawq for W4, gptqmodel for W3 (3-bit). ---
# If an install flakes, re-run this cell; sections are independent so one engine
# failing still yields a verdict from the other.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.44", "accelerate", "datasets", "huggingface_hub",
                "autoawq", "gptqmodel", "optimum"], check=False)
import transformers
print("transformers", transformers.__version__)

# %%
# --- 2. Config ---
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"     # f16 source
OUT_AWQ  = "qwen2.5-3b-awq-w4"
OUT_GPTQ = "qwen2.5-3b-gptq-w3"
N_CALIB  = 256          # calibration samples
SEQ_LEN  = 1024
# GO/NO-GO PPL-ratio gates (candidate_ppl / f16_ppl):
GATE = {"awq_w4": 1.05, "gptq_w3": 1.12}
results = {"model": MODEL_ID, "seed": SEED, "ppl": {}, "bits": {}, "verdict": {}}

# %% [markdown]
# ## Calibration + eval data (CODE)
# For apples-to-apples with the local oracle, upload the repo's
# `artifacts/quant/calib_trim.txt` and `ppl_trim.txt` to `/content/`. Otherwise we
# fall back to a public code dataset. **Calib and eval are kept disjoint.**

# %%
# --- 3. Build disjoint code calib + eval text blocks ---
def load_code_blocks():
    calib_p, eval_p = "/content/calib_trim.txt", "/content/ppl_trim.txt"
    if os.path.exists(calib_p) and os.path.exists(eval_p):
        calib = open(calib_p, encoding="utf-8", errors="replace").read()
        ev    = open(eval_p, encoding="utf-8", errors="replace").read()
        print("using uploaded repo corpora (comparable to local oracle C)")
    else:
        from datasets import load_dataset
        ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python",
                          split="train", streaming=False)
        texts = [r["content"] for r in ds.select(range(400))]
        cut = len(texts) * 3 // 4
        calib = "\n\n".join(texts[:cut]); ev = "\n\n".join(texts[cut:])
        print("using bigcode/the-stack-smol python (fallback)")
    return calib, ev
CALIB_TEXT, EVAL_TEXT = load_code_blocks()
print(f"calib chars={len(CALIB_TEXT)}  eval chars={len(EVAL_TEXT)}")

# %%
# --- 4. Tokenizer + calibration sample list ---
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
def make_calib(text, n, seqlen):
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = max(1, (len(ids) - seqlen) // n)
    chunks = [ids[i:i+seqlen] for i in range(0, len(ids)-seqlen, step)][:n]
    return [tok.decode(c) for c in chunks]
CALIB = make_calib(CALIB_TEXT, N_CALIB, SEQ_LEN)
print("calib samples:", len(CALIB))

# %%
# --- 5. PPL helper (HF sliding window). Same routine for every model => ratios are fair. ---
def ppl(model, text, max_len=2048, stride=512):
    model.eval()
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    nlls, n_tok = [], 0
    prev = 0
    for beg in range(0, ids.size(1), stride):
        end = min(beg + max_len, ids.size(1))
        trg = end - prev
        inp = ids[:, beg:end]
        tgt = inp.clone(); tgt[:, :-trg] = -100
        with torch.no_grad():
            loss = model(inp, labels=tgt).loss
        nlls.append(loss * trg); n_tok += trg
        prev = end
        if end == ids.size(1): break
    return float(torch.exp(torch.stack(nlls).sum() / n_tok))

# %%
# --- 6. f16 reference PPL (the gate denominator) ---
from transformers import AutoModelForCausalLM
m_f16 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)
results["ppl"]["f16"] = ppl(m_f16, EVAL_TEXT)
print("f16 code PPL =", round(results["ppl"]["f16"], 4))
del m_f16; torch.cuda.empty_cache()

# %% [markdown]
# ## Section B — AWQ W4 (stable baseline; ~same bits as Q4_K_M, higher quality)

# %%
# --- 7. AWQ 4-bit quantize + save ---
try:
    from awq import AutoAWQForCausalLM
    awq = AutoAWQForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    awq.quantize(tok, quant_config={"w_bit": 4, "q_group_size": 128,
                                    "zero_point": True, "version": "GEMM"},
                 calib_data=CALIB)
    awq.save_quantized(OUT_AWQ); tok.save_pretrained(OUT_AWQ)
    results["bits"]["awq_w4"] = 4.0
    print("AWQ-W4 saved ->", OUT_AWQ)
except Exception as e:
    print("AWQ section FAILED:", repr(e))

# %%
# --- 8. AWQ W4 PPL ---
try:
    from awq import AutoAWQForCausalLM
    m = AutoAWQForCausalLM.from_quantized(OUT_AWQ, fuse_layers=False)
    results["ppl"]["awq_w4"] = ppl(m.model, EVAL_TEXT)
    print("AWQ-W4 code PPL =", round(results["ppl"]["awq_w4"], 4))
    del m; torch.cuda.empty_cache()
except Exception as e:
    print("AWQ PPL skipped:", repr(e))

# %% [markdown]
# ## Section C — GPTQ W3 (the byte-cut prize: ~3 bits)

# %%
# --- 9. GPTQ 3-bit quantize + save ---
try:
    from gptqmodel import GPTQModel, QuantizeConfig
    qc = QuantizeConfig(bits=3, group_size=128, desc_act=True)
    g = GPTQModel.load(MODEL_ID, qc)
    g.quantize(CALIB, batch_size=1)
    g.save(OUT_GPTQ); tok.save_pretrained(OUT_GPTQ)
    results["bits"]["gptq_w3"] = 3.0
    print("GPTQ-W3 saved ->", OUT_GPTQ)
except Exception as e:
    print("GPTQ section FAILED:", repr(e))

# %%
# --- 10. GPTQ W3 PPL ---
try:
    from gptqmodel import GPTQModel
    m = GPTQModel.load(OUT_GPTQ)
    results["ppl"]["gptq_w3"] = ppl(m.model, EVAL_TEXT)
    print("GPTQ-W3 code PPL =", round(results["ppl"]["gptq_w3"], 4))
    del m; torch.cuda.empty_cache()
except Exception as e:
    print("GPTQ PPL skipped:", repr(e))

# %%
# --- 11. VERDICT (the checks-and-balances gate) ---
f16 = results["ppl"].get("f16")
print(f"\n{'candidate':10s} {'bits':>5s} {'PPL':>9s} {'PPL/f16':>8s} {'gate':>6s} verdict")
for k, thr in GATE.items():
    p = results["ppl"].get(k)
    if p is None or not f16:
        results["verdict"][k] = "MISSING"; print(f"{k:10s}   --   missing"); continue
    ratio = p / f16
    v = "GO" if ratio <= thr else "NO-GO"
    results["verdict"][k] = v
    print(f"{k:10s} {results['bits'].get(k,0):5.1f} {p:9.4f} {ratio:8.3f} {thr:6.2f} {v}")
results["local_oracle_c_reference"] = {"q4km": 4.485, "q3_imatrix": 5.915, "q3_naive": 11.432}
json.dump(results, open("awq_bytecut_results.json", "w"), indent=2)
print("\nsaved awq_bytecut_results.json")
print("DECISION: a GO on gptq_w3 means ~3-bit at usable code quality -> worth the "
      "M3 low-bit kernel. NO-GO on both means stay at Q4_K_M.")

# %%
# --- 12. Package artifacts for download ---
import shutil
for d in (OUT_AWQ, OUT_GPTQ):
    if os.path.isdir(d):
        shutil.make_archive(d, "zip", d); print("zipped", d + ".zip")
try:
    from google.colab import files
    files.download("awq_bytecut_results.json")
except Exception:
    pass

# %% [markdown]
# ## Integration boundary (M3 follow-up — do NOT skip)
# A GO here is a **quality** verdict, not a shipped lever. To use a winner in dismantle:
# 1. **Easiest signal:** convert the dequantized winner to GGUF and `llama-perplexity`
#    it against `artifacts/quant/ppl_trim.txt` to confirm the gain survives the GGUF
#    round-trip on the *same scale* as local oracle C (4.485 / 5.915 / 11.432).
# 2. **To actually run in dismantle:** either (a) convert AWQ/GPTQ → a GGUF quant type
#    dismantle already loads (re-quantize guided by the AWQ scales), or (b) implement a
#    native packed-int{3,4} GEMV kernel + loader on the M3 (this is the real work; pair
#    it with Stage-2 kernel effort).
# 3. Re-run the §1 gate (`tools/bench/analyze_tcb_trace.py`) on any decode bench.
# **Until step 1 passes on the GGUF scale, treat the Colab PPL as indicative only.**
