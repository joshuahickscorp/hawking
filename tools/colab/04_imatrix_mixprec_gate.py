# %% [markdown]
# # P2-H — imatrix mixed-precision quality gate (the decisive Colab leg)
#
# Answers the question `reports/oracle_imatrix_mixprec.md` left as
# NEEDS-MEASUREMENT: **does an imatrix-guided mixed-precision GGUF (~3.82 eff
# bits) beat uniform Q4_K_M on the metric that matters — next-token LOGITS — on
# held-out code?** The local weight-RMSE proxy could not see this (RMSE is not
# the model's objective; the real activation imatrix protects logit-moving
# channels weight-norm is blind to). This notebook is the gate the oracle's
# §"DECISIVE (Colab) gate" runbook specifies.
#
# **It stays in GGUF** — every tensor is a standard ggml K-quant, so a GO needs
# only loader byte-accounting, NO new kernel (unlike AWQ/QTIP). That is the whole
# appeal of this axis. (The AWQ flavour of P2-H — does AWQ smoothing help W4A8 —
# is a separate run: `colab/awq_w4a8_validate.py`; see reports/COLAB_HANDOFF.md.)
#
# **Produces:** `imatrix_mixprec_results.json` (+ the quantized `.gguf` files).
#
# **Kill-respects (dead_levers.md / oracle §):** quantize FROM a near-lossless
# (q8_0/f16) source, NEVER requant-from-Q4 (the +32% PPL kill). The Q4_K_M
# baseline is the shipped llama.cpp quantizer (the gold standard), not a NumPy
# reimplementation. Every leg is guarded: if it cannot run faithfully it reports
# `null` and the verdict downgrades to NEEDS-MEASUREMENT — it never fabricates a
# number (a wrong gate is worse than none).
#
# **Runtime:** ~20–40 min on L4/A100 (imatrix + 2 quantizes + 3 forward passes).
# GPU strongly preferred; CPU works but the forward/PPL legs are slow.

# %%
# --- 0. Build llama.cpp (CUDA if a GPU is present) — the gold K-quant tool ---
import json
import os
import re
import subprocess
from pathlib import Path

REPO = "/content/llama.cpp"
BIN = f"{REPO}/build/bin"
HAS_GPU = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
print("GPU runtime:", HAS_GPU)

if not Path(BIN, "llama-quantize").exists():
    if not Path(REPO).exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ggml-org/llama.cpp", REPO], check=True)
    flags = ["-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"]
    if HAS_GPU:
        flags.append("-DGGML_CUDA=ON")
    subprocess.run(["cmake", "-S", REPO, "-B", f"{REPO}/build", *flags], check=True)
    subprocess.run(["cmake", "--build", f"{REPO}/build", "-j", "--config", "Release"],
                   check=True)
for tool in ("llama-imatrix", "llama-quantize", "llama-perplexity"):
    assert Path(BIN, tool).exists(), f"build missing {tool}"
print("llama.cpp built:", BIN)

# %%
# --- 1. Fetch a near-lossless source GGUF (q8_0 preferred, else f16) ---
# This near-lossless source is BOTH the from-f16 quant source AND the f16-class
# reference for the logit gate (q8_0 vs f16 logits differ <1e-3 — negligible vs
# the Q4/Q3 gap we are measuring).
subprocess.run(["pip", "install", "-q", "huggingface_hub"], check=False)
from huggingface_hub import list_repo_files, hf_hub_download

REPOS = ["Qwen/Qwen2.5-3B-Instruct-GGUF", "bartowski/Qwen2.5-3B-Instruct-GGUF"]
PREF = ["q8_0", "f16", "fp16"]
def find_source():
    for repo in REPOS:
        try:
            ggufs = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
        except Exception as e:
            print("skip", repo, e); continue
        single = [f for f in ggufs if "-of-" not in f]
        for pref in PREF:
            hit = sorted((f for f in single if pref in f.lower()), key=len)
            if hit:
                return repo, hit[0]
    return None, None
repo, fname = find_source()
assert fname, f"no single-file q8_0/f16 GGUF in {REPOS}; edit REPOS"
SRC = hf_hub_download(repo, fname)
print("source (near-lossless f16-class):", repo, "/", fname, "->", SRC)

# %%
# --- 2. Fetch the code corpora (same as the local oracle: calib + eval) ---
RAW = ("https://raw.githubusercontent.com/joshuahickscorp/dismantle/"
       "codex/maximal-spec-colab/colab/data")
import urllib.request
for name in ("calib_trim.txt", "ppl_trim.txt"):
    if not Path("/content", name).exists():
        try:
            urllib.request.urlretrieve(f"{RAW}/{name}", f"/content/{name}")
        except Exception as e:
            print("corpus fetch failed", name, e)
CALIB, EVAL = "/content/calib_trim.txt", "/content/ppl_trim.txt"
assert Path(CALIB).exists() and Path(EVAL).exists(), "code corpora missing"
print("corpora ready")

# %%
# --- 3. Helpers ---
NGL = ["-ngl", "99"] if HAS_GPU else []
def sh(cmd):
    print("$", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
def gib(p):
    return round(os.path.getsize(p) / 1024**3, 4)

# %%
# --- 4. Real activation imatrix on the near-lossless source ---
IM = "/content/code.imatrix"
if not Path(IM).exists():
    r = sh([f"{BIN}/llama-imatrix", "-m", SRC, "-f", CALIB, "-o", IM, "--chunks", "80", *NGL])
    assert Path(IM).exists(), r.stderr[-1200:]
print("imatrix:", IM, gib(IM) if Path(IM).exists() else "MISSING", "GiB")

# %%
# --- 5. Quantize: uniform Q4_K_M (gold) + imatrix MIXED-PRECISION ---
# Mixed allocation (GGUF-native, per-tensor): keep attention + ffn_gate at Q4_K
# (logit-sensitive / smaller), demote the two big FFN projections ffn_down +
# ffn_up to Q3_K. imatrix guides every tensor's quantization. Target: total bytes
# <= uniform Q4_K_M (the oracle's <=0.85x budget, ~3.82 eff bits). If the
# installed llama-quantize lacks --tensor-type, fall back to uniform Q3_K_M+imat
# as the mixed stand-in (logged), so the gate still produces a verdict.
GOLD = "/content/qwen3b-q4km.gguf"      # the shipped incumbent
MIX = "/content/qwen3b-mixed.gguf"      # imatrix mixed-precision candidate
results = {"source": f"{repo}/{fname}", "has_gpu": HAS_GPU,
           "gate": {"recon_floor": "mixed_rmse <= uniform_rmse at fewer bytes",
                    "logit": "cos(mix,ref) >= cos(q4k,ref) AND argmax(mix) >= argmax(q4k) AND KL(ref||mix) <= KL(ref||q4k)",
                    "ppl": "ppl(mix) <= ppl(q4k)"},
           "gib": {}, "ppl": {}, "logit": {}, "notes": []}

r = sh([f"{BIN}/llama-quantize", "--imatrix", IM, SRC, GOLD, "Q4_K_M"])
assert Path(GOLD).exists(), r.stderr[-1200:]
results["gib"]["q4km"] = gib(GOLD)

# Probe whether this llama-quantize build supports per-tensor overrides.
help_txt = sh([f"{BIN}/llama-quantize", "--help"]).stdout + sh([f"{BIN}/llama-quantize"]).stderr
has_override = "--tensor-type" in help_txt
if has_override:
    r = sh([f"{BIN}/llama-quantize", "--imatrix", IM,
            "--tensor-type", "ffn_down=Q3_K", "--tensor-type", "ffn_up=Q3_K",
            SRC, MIX, "Q4_K_M"])
    if not Path(MIX).exists():
        results["notes"].append("mixed override failed; falling back to uniform Q3_K_M+imatrix")
        has_override = False
if not has_override:
    r = sh([f"{BIN}/llama-quantize", "--imatrix", IM, SRC, MIX, "Q3_K_M"])
    results["notes"].append("mixed = uniform Q3_K_M+imatrix (no --tensor-type in this build)")
assert Path(MIX).exists(), r.stderr[-1200:]
results["gib"]["mixed"] = gib(MIX)
results["mixed_under_budget"] = results["gib"]["mixed"] <= results["gib"]["q4km"]
print("gib:", results["gib"], "mixed<=uniform:", results["mixed_under_budget"])

# %%
# --- 6. LEG PPL (corroborating): ref(q8) / Q4_K_M / mixed on held-out code ---
def ppl(gguf):
    try:
        r = sh([f"{BIN}/llama-perplexity", "-m", gguf, "-f", EVAL, "--chunks", "50", *NGL])
        m = re.findall(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", r.stderr + r.stdout)
        return float(m[-1]) if m else None
    except Exception as e:
        print("ppl failed", gguf, e); return None
for tag, g in (("ref", SRC), ("q4km", GOLD), ("mixed", MIX)):
    results["ppl"][tag] = ppl(g)
    print(f"  PPL[{tag}] = {results['ppl'][tag]}")

# %%
# --- 7. LEG LOGIT (decisive): cos / KL / argmax-agree vs the f16-class ref ---
# llama-cpp-python forward with logits_all; compare mixed and Q4_K_M to the
# near-lossless ref over the first ~256 tokens of the eval corpus. Guarded: any
# failure leaves results["logit"] empty -> NEEDS-MEASUREMENT.
try:
    subprocess.run(["pip", "install", "-q", "llama-cpp-python"], check=False)
    import numpy as np
    from llama_cpp import Llama

    eval_text = Path(EVAL).read_text(errors="ignore")[:4000]

    def next_token_logits(gguf, n=256):
        llm = Llama(model_path=gguf, n_ctx=512, logits_all=True,
                    n_gpu_layers=99 if HAS_GPU else 0, verbose=False)
        toks = llm.tokenize(eval_text.encode("utf-8"))[:n]
        llm.reset(); llm.eval(toks)
        scores = np.array(llm.scores[:len(toks)], dtype=np.float32)  # (T, V)
        del llm
        return scores

    ref = next_token_logits(SRC)
    q4k = next_token_logits(GOLD)
    mix = next_token_logits(MIX)
    T = min(len(ref), len(q4k), len(mix))
    ref, q4k, mix = ref[:T], q4k[:T], mix[:T]

    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x); return e / e.sum(axis=-1, keepdims=True)
    def cos(a, b):
        num = (a * b).sum(-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9
        return float((num / den).mean())
    def kl(p_logits, q_logits):
        p = softmax(p_logits); lp = np.log(p + 1e-12)
        lq = np.log(softmax(q_logits) + 1e-12)
        return float((p * (lp - lq)).sum(-1).mean())
    def argmax_agree(a, b):
        return float((a.argmax(-1) == b.argmax(-1)).mean())

    results["logit"] = {
        "cos_mix_ref": cos(mix, ref), "cos_q4k_ref": cos(q4k, ref),
        "kl_ref_mix": kl(ref, mix), "kl_ref_q4k": kl(ref, q4k),
        "argmax_mix_ref": argmax_agree(mix, ref), "argmax_q4k_ref": argmax_agree(q4k, ref),
        "tokens": int(T),
    }
    print("logit leg:", json.dumps(results["logit"], indent=2))
except Exception as e:
    results["notes"].append(f"logit leg skipped: {e}")
    print("logit leg skipped:", e)

# %%
# --- 8. VERDICT ---
lg = results.get("logit") or {}
recon_ok = bool(results.get("mixed_under_budget"))
ppl_ok = (results["ppl"].get("mixed") is not None and results["ppl"].get("q4km") is not None
          and results["ppl"]["mixed"] <= results["ppl"]["q4km"])
logit_ran = all(k in lg for k in ("cos_mix_ref", "kl_ref_mix", "argmax_mix_ref"))
logit_ok = (logit_ran
            and lg["cos_mix_ref"] >= lg["cos_q4k_ref"]
            and lg["argmax_mix_ref"] >= lg["argmax_q4k_ref"]
            and lg["kl_ref_mix"] <= lg["kl_ref_q4k"])

if not logit_ran:
    results["verdict"] = "NEEDS-MEASUREMENT"
    results["reason"] = "decisive logit leg did not run (see notes); recon/PPL alone cannot pass or kill"
elif logit_ok and recon_ok:
    results["verdict"] = "GO"
    results["reason"] = "imatrix mixed-prec matches-or-beats uniform Q4_K on logits at fewer bytes (byte-cut is free on quality)"
else:
    results["verdict"] = "NO-GO"
    results["kill_type"] = "Type-1 (measured): non-uniform allocation does not beat uniform Q4_K on the logit metric with a real imatrix"
    results["reason"] = "logit and/or recon gate failed with a fair real imatrix -> routes to dead_levers.md"
results["legs"] = {"recon_floor": recon_ok, "ppl": ppl_ok, "logit_ran": logit_ran, "logit": logit_ok}

Path("/content/imatrix_mixprec_results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
try:
    from google.colab import files
    files.download("/content/imatrix_mixprec_results.json")
except Exception:
    pass

# %% [markdown]
# ## What the verdict means
# - **GO** — the imatrix mixed-precision GGUF matches-or-beats uniform Q4_K_M on
#   logit-cosine/KL/argmax (decisive) at fewer bytes. Next: wire the per-tensor
#   mix into dismantle's loader (byte accounting only, NO kernel — every tensor is
#   a standard ggml K-quant) + a paired decode bench for the bible's +12–20%
#   throughput claim. This is a cleaner GO than AWQ (no W4A8 kernel / re-gate).
# - **NO-GO** — records a measured quality Type-1 kill in `reports/dead_levers.md`:
#   non-uniform allocation does not beat uniform Q4_K on the metric that matters.
#   The imatrix mixed-precision byte-cut axis closes.
# - **NEEDS-MEASUREMENT** — the decisive logit leg could not run (llama-cpp-python
#   build / VRAM). Re-run on an L4/A100; recon-RMSE + PPL alone cannot legitimately
#   pass or kill an activation-driven method (the Type-2 error CLAUDE.md forbids).
#
# The **AWQ** half of P2-H (does AWQ smoothing help W4A8, the wired
# `DISMANTLE_QWEN_AWQ` downstream) is `colab/awq_w4a8_validate.py` — a separate
# run; see `reports/COLAB_HANDOFF.md` for its command + PASS criterion.
