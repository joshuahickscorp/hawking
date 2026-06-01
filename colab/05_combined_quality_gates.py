# %% [markdown]
# # Combined sub-Q4 quality gates — ONE run, BOTH verdicts
#
# Folds the two decisive Colab gates into a single "Run all" so the expensive
# setup (build llama.cpp + download Qwen2.5-3B + build the gold Q4_K_M) happens
# **once** instead of twice:
#
# 1. **Gate 2 — imatrix mixed-precision** (fast, reliable, pure llama.cpp): does
#    an imatrix-guided mixed-prec GGUF (~3.82 eff bits) beat the shipped
#    Q4_K_M on next-token logits + PPL? → `imatrix_mixprec_results.json`
# 2. **Gate 1 — QTIP 3-bit trellis** (slow, Drive-resumable): does the real
#    Cornell-RelaxML QTIP codec (RHT+trellis, fit from f16) beat Q4_K_M on
#    weight-RMSE + logits + PPL? → `qtip_3bit_results.json`
#
# Gate 2 runs FIRST (so you get that verdict in ~20-30 min even if the QTIP
# codec is slow/finicky), Gate 1 after. Each gate is independently guarded — a
# failure in one still produces the other's verdict + downloads its JSON. The
# QTIP codec auto-saves to / restores from Google Drive so a disconnect never
# costs the ~20-40 min quant. Both gates compare against the SAME shipped gold
# Q4_K_M (the incumbent we'd replace).

# %%
# --- 0. Config + GPU + the baked-in QTIP decision constants ------------------
import json, os, re, subprocess, sys, time, shutil
from pathlib import Path

import torch
assert torch.cuda.is_available(), "GPU REQUIRED (f16 forward + QTIP quant)."
_gpu = torch.cuda.get_device_name(0)
_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print("GPU:", _gpu, f"{_vram:.0f} GB")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BITS = 3                              # QTIP stored bits/weight target
RUN_REAL_QTIP = _vram >= 24           # need L4/A100-class for the real QTIP codec
# QTIP gate lines (reports/oracle_qtip_quality.md):
PROXY_BITS_NEEDED = 1.20
GATE_BITS_NEEDED = 0.0                # GO: measured bits_needed <= 0
GATE_PPL_RATIO = 1.10                 # GO: QTIP PPL <= 1.10x f16
RHT_BLOCK = 256; QK = 256; SEED = 0

results_qtip = {
    "model": MODEL_ID, "bits": BITS, "gpu": _gpu, "vram_gb": round(_vram, 1),
    "run_real_qtip": RUN_REAL_QTIP, "proxy_bits_needed": PROXY_BITS_NEEDED,
    "gate": {"bits_needed_le": GATE_BITS_NEEDED, "ppl_ratio_le": GATE_PPL_RATIO,
             "logit": "cos>=Q4K & argmax>=Q4K & KL<=Q4K, on code"},
    "verdict": "PENDING",
}
results_im = {
    "model": MODEL_ID, "gpu": _gpu, "vram_gb": round(_vram, 1),
    "gate": {"recon_floor": "mixed_bytes <= uniform_Q4_K_M bytes",
             "logit": "cos(mix,ref) >= cos(q4k,ref) AND argmax >= AND KL <=",
             "ppl": "ppl(mix) <= ppl(q4k)"},
    "gib": {}, "ppl": {}, "logit": {}, "notes": [], "verdict": "PENDING",
}
print("RUN_REAL_QTIP =", RUN_REAL_QTIP)

# %%
# --- 1. Build llama.cpp once (gold Q4_K_M quantizer + imatrix + perplexity) ---
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.44", "accelerate", "safetensors",
                "huggingface_hub", "gguf", "scipy"], check=False)
import numpy as np

REPO = "/content/llama.cpp"; BIN = f"{REPO}/build/bin"
if not Path(BIN, "llama-quantize").exists():
    if not Path(REPO).exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ggml-org/llama.cpp", REPO], check=True)
    subprocess.run(["cmake", "-S", REPO, "-B", f"{REPO}/build",
                    "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON"], check=True)
    subprocess.run(["cmake", "--build", f"{REPO}/build", "-j", "--config", "Release"], check=True)
for tool in ("llama-quantize", "llama-perplexity", "llama-imatrix"):
    assert Path(BIN, tool).exists(), f"build missing {tool}"
print("llama.cpp built:", BIN)

# %%
# --- 2. Fetch near-lossless source + corpora + the SHARED gold Q4_K_M --------
from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download
import urllib.request

REPOS = ["Qwen/Qwen2.5-3B-Instruct-GGUF", "bartowski/Qwen2.5-3B-Instruct-GGUF"]
def find_source():
    for repo in REPOS:
        try:
            ggufs = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
        except Exception as e:
            print("skip", repo, e); continue
        single = [f for f in ggufs if "-of-" not in f]
        for pref in ("q8_0", "f16", "fp16"):
            hit = sorted((f for f in single if pref in f.lower()), key=len)
            if hit:
                return repo, hit[0]
    return None, None
gg_repo, gg_name = find_source()
assert gg_name, f"no single-file q8_0/f16 GGUF in {REPOS}"
SRC_GGUF = hf_hub_download(gg_repo, gg_name)
results_qtip["source_gguf"] = results_im["source_gguf"] = f"{gg_repo}/{gg_name}"
print("near-lossless source:", SRC_GGUF)

RAW = ("https://raw.githubusercontent.com/joshuahickscorp/dismantle/"
       "codex/maximal-spec-colab/colab/data")
for name in ("calib_trim.txt", "ppl_trim.txt"):
    if not Path("/content", name).exists():
        urllib.request.urlretrieve(f"{RAW}/{name}", f"/content/{name}")
CALIB, EVAL = "/content/calib_trim.txt", "/content/ppl_trim.txt"

# SHARED gold Q4_K_M (the shipped incumbent both gates compare against),
# quantized from the near-lossless source (from-f16-class, never from-Q4).
Q4KM_GGUF = "/content/qwen3b-q4_k_m.gguf"
if not Path(Q4KM_GGUF).exists():
    r = subprocess.run([f"{BIN}/llama-quantize", "--allow-requantize",
                        SRC_GGUF, Q4KM_GGUF, "Q4_K_M"], capture_output=True, text=True)
    assert Path(Q4KM_GGUF).exists(), r.stderr[-1000:]
def _gib(p): return round(os.path.getsize(p) / 1024**3, 4)
print("gold Q4_K_M:", Q4KM_GGUF, f"({_gib(Q4KM_GGUF)} GiB)")

# Shared logit/PPL helpers.
def _softmax(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)
def _cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    num = (a * b).sum(-1); den = np.linalg.norm(a, -1) * np.linalg.norm(b, -1)
    return float(np.mean(num / np.maximum(den, 1e-12)))
def _kl(p_logits, q_logits):
    p = _softmax(p_logits.astype(np.float64)); q = _softmax(q_logits.astype(np.float64))
    return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), -1)))
def _argmax_agree(a, b):
    return float(np.mean(a.argmax(-1) == b.argmax(-1)))
def _ppl(gguf):
    try:
        r = subprocess.run([f"{BIN}/llama-perplexity", "-m", gguf, "-f", EVAL,
                            "--chunks", "50", "-ngl", "99"], capture_output=True, text=True)
        m = re.findall(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", r.stderr + r.stdout)
        return float(m[-1]) if m else None
    except Exception as e:
        print("ppl failed", gguf, e); return None
def _gguf_logits(gguf, n=256):
    """Next-token logits for the first n eval tokens via llama-cpp-python."""
    from llama_cpp import Llama
    txt = Path(EVAL).read_text(errors="ignore")[:4000]
    llm = Llama(model_path=gguf, n_ctx=max(n + 8, 512), logits_all=True,
                n_gpu_layers=-1, verbose=False)
    toks = llm.tokenize(txt.encode("utf-8"))[:n]
    llm.reset(); llm.eval(toks)
    sc = np.array(llm.scores[:len(toks)], dtype=np.float32)
    del llm
    return sc

# %%
# ====================== GATE 2 — imatrix mixed-precision =====================
# Build a real activation imatrix on the source, then an imatrix mixed-prec GGUF
# (attn+ffn_gate @ Q4_K, ffn_down+ffn_up @ Q3_K), and test it vs the gold
# Q4_K_M on bytes + PPL + the decisive next-token LOGITS. GGUF-native -> a GO
# needs only loader byte-accounting, no kernel. Guarded as a whole.
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python"], check=False)
    IM = "/content/code.imatrix"
    if not Path(IM).exists():
        r = subprocess.run([f"{BIN}/llama-imatrix", "-m", SRC_GGUF, "-f", CALIB,
                            "-o", IM, "--chunks", "80", "-ngl", "99"], capture_output=True, text=True)
        assert Path(IM).exists(), r.stderr[-1200:]
    MIX = "/content/qwen3b-mixed.gguf"
    results_im["gib"]["q4km"] = _gib(Q4KM_GGUF)
    help_txt = (subprocess.run([f"{BIN}/llama-quantize", "--help"], capture_output=True, text=True).stdout
                + subprocess.run([f"{BIN}/llama-quantize"], capture_output=True, text=True).stderr)
    has_override = "--tensor-type" in help_txt
    if has_override and not Path(MIX).exists():
        r = subprocess.run([f"{BIN}/llama-quantize", "--imatrix", IM,
                            "--tensor-type", "ffn_down=Q3_K", "--tensor-type", "ffn_up=Q3_K",
                            SRC_GGUF, MIX, "Q4_K_M"], capture_output=True, text=True)
        if not Path(MIX).exists():
            has_override = False; results_im["notes"].append("override failed; uniform Q3_K_M+imatrix fallback")
    if not has_override and not Path(MIX).exists():
        subprocess.run([f"{BIN}/llama-quantize", "--imatrix", IM, SRC_GGUF, MIX, "Q3_K_M"],
                       capture_output=True, text=True)
        results_im["notes"].append("mixed = uniform Q3_K_M+imatrix (no --tensor-type in this build)")
    assert Path(MIX).exists(), "mixed GGUF not produced"
    results_im["gib"]["mixed"] = _gib(MIX)
    results_im["mixed_under_budget"] = results_im["gib"]["mixed"] <= results_im["gib"]["q4km"]

    for tag, g in (("ref", SRC_GGUF), ("q4km", Q4KM_GGUF), ("mixed", MIX)):
        results_im["ppl"][tag] = _ppl(g)
    try:
        ref = _gguf_logits(SRC_GGUF); q4k = _gguf_logits(Q4KM_GGUF); mix = _gguf_logits(MIX)
        T = min(len(ref), len(q4k), len(mix)); ref, q4k, mix = ref[:T], q4k[:T], mix[:T]
        results_im["logit"] = {
            "cos_mix_ref": _cos(mix, ref), "cos_q4k_ref": _cos(q4k, ref),
            "kl_ref_mix": _kl(ref, mix), "kl_ref_q4k": _kl(ref, q4k),
            "argmax_mix_ref": _argmax_agree(mix, ref), "argmax_q4k_ref": _argmax_agree(q4k, ref),
            "tokens": int(T)}
    except Exception as e:
        results_im["notes"].append(f"logit leg skipped: {e}")

    lg = results_im.get("logit") or {}
    recon_ok = bool(results_im.get("mixed_under_budget"))
    ppl_ok = (results_im["ppl"].get("mixed") is not None and results_im["ppl"].get("q4km") is not None
              and results_im["ppl"]["mixed"] <= results_im["ppl"]["q4km"])
    logit_ran = all(k in lg for k in ("cos_mix_ref", "kl_ref_mix", "argmax_mix_ref"))
    logit_ok = (logit_ran and lg["cos_mix_ref"] >= lg["cos_q4k_ref"]
                and lg["argmax_mix_ref"] >= lg["argmax_q4k_ref"] and lg["kl_ref_mix"] <= lg["kl_ref_q4k"])
    if not logit_ran:
        results_im["verdict"] = "NEEDS-MEASUREMENT"
        results_im["reason"] = "decisive logit leg did not run; recon/PPL alone cannot pass or kill"
    elif logit_ok and recon_ok:
        results_im["verdict"] = "GO"
    else:
        results_im["verdict"] = "NO-GO"
        results_im["kill_type"] = "Type-1: non-uniform allocation does not beat uniform Q4_K on logits"
    results_im["legs"] = {"recon": recon_ok, "ppl": ppl_ok, "logit_ran": logit_ran, "logit": logit_ok}
except Exception as e:
    results_im["verdict"] = "ERROR"; results_im["error"] = repr(e)[:400]
    print("[GATE 2 imatrix ERROR]", repr(e)[:400])

Path("/content/imatrix_mixprec_results.json").write_text(json.dumps(results_im, indent=2))
print("\n[GATE 2 — imatrix] verdict:", results_im["verdict"])
print(json.dumps(results_im, indent=2))
try:
    from google.colab import files; files.download("/content/imatrix_mixprec_results.json")
except Exception: pass

# %%
# --- GATE 1 setup: the QTIP RHT + Lloyd-Max BRACKET codec (cheap fallback) ----
# Same model as tools/bench/oracle_qtip_quality.py: RHT exact + a scalar at the
# {K, K+1}-bit rate. The bracket never invents a single trellis number; the REAL
# codec (next cell) does. rel_rmse is used by LEG 1.
def _fwht(a):
    a = a.astype(np.float64); orig = a.shape; n = orig[-1]
    a = a.reshape(-1, n).copy(); h = 1
    while h < n:
        a = a.reshape(-1, n // (2 * h), 2, h)
        x = a[:, :, 0, :]; y = a[:, :, 1, :]
        a = np.concatenate([x + y, x - y], -1).reshape(-1, n); h *= 2
    return (a / np.sqrt(n)).reshape(orig)
def _rht_signs(block, seed=SEED):
    return np.random.default_rng(seed).integers(0, 2, size=block).astype(np.float64) * 2 - 1
def _erfinv(y):
    try:
        from scipy.special import erfinv; return erfinv(y)
    except Exception:
        a = 0.147; ln = np.log(np.maximum(1 - y * y, 1e-300)); t = 2 / (np.pi * a) + ln / 2
        return np.sign(y) * np.sqrt(np.sqrt(t * t - ln / a) - t)
_LLOYD = {}
def _lloyd_max_gaussian(nlevels, iters=80, samples=400000, seed=SEED):
    if nlevels in _LLOYD: return _LLOYD[nlevels]
    rng = np.random.default_rng(seed); x = np.sort(rng.standard_normal(samples))
    p = (np.arange(nlevels) + 0.5) / nlevels; lv = np.sqrt(2.0) * _erfinv(2.0 * p - 1.0)
    for _ in range(iters):
        b = (lv[:-1] + lv[1:]) / 2.0; idx = np.searchsorted(b, x)
        sums = np.bincount(idx, weights=x, minlength=nlevels); cnts = np.bincount(idx, minlength=nlevels)
        new = np.where(cnts > 0, sums / np.maximum(cnts, 1), lv)
        if np.allclose(new, lv, atol=1e-8): lv = new; break
        lv = new
    _LLOYD[nlevels] = lv; return lv
def _scalar_q(y, levels):
    b = (levels[:-1] + levels[1:]) / 2.0
    return levels[np.searchsorted(b, y.ravel())].reshape(y.shape)
def qtip_bracket(W, store_bits=BITS, quality_bits=None, rht_block=RHT_BLOCK, seed=SEED):
    if quality_bits is None: quality_bits = store_bits
    levels = _lloyd_max_gaussian(1 << int(quality_bits)); signs = _rht_signs(rht_block, seed)
    R, C = W.shape; nblk = C // rht_block; recon = np.array(W, dtype=np.float64, copy=True)
    if nblk:
        seg = W[:, :nblk * rht_block].reshape(R, nblk, rht_block); rot = _fwht(seg * signs)
        sigma = rot.std(-1, keepdims=True); sigma = np.where(sigma == 0, 1.0, sigma)
        qhat = _scalar_q(rot / sigma, levels) * sigma; derot = _fwht(qhat) * signs
        recon[:, :nblk * rht_block] = derot.reshape(R, nblk * rht_block)
    block_bytes = (store_bits * QK) / 8.0 + (16.0 / 8.0) * (QK / rht_block)
    return recon, store_bits + 16.0 / rht_block, block_bytes
def rel_rmse(recon, ref):
    recon = recon.ravel().astype(np.float64); ref = ref.ravel().astype(np.float64)
    d = np.linalg.norm(ref)
    return float(np.linalg.norm(recon - ref) / d) if d > 0 else float("nan")
print("QTIP bracket ready.")

# %%
# --- GATE 1: REAL upstream QTIP codec (Cornell-RelaxML), Drive-resumable ------
HF_DIR = Path("/content/qwen3b-qtip-3bit-hf")
HESS = Path("/content/hessians_code"); CKPT = Path("/content/qwen3b-qtip-3bit")
results_qtip["real_qtip_ran"] = False
CACHE = None
try:
    from google.colab import drive
    drive.mount("/content/drive")
    CACHE = Path("/content/drive/MyDrive/qtip_gate_cache"); CACHE.mkdir(parents=True, exist_ok=True)
    for art in (HF_DIR, CKPT, HESS):
        c = CACHE / art.name
        if not art.exists() and c.exists():
            (shutil.copytree if c.is_dir() else shutil.copy2)(c, art); print("restored:", art.name)
except Exception as e:
    print("[Drive cache unavailable]", repr(e)[:160])

if HF_DIR.is_dir():
    results_qtip["real_qtip_ran"] = True; print("QTIP HF model present -> skip codec")
elif RUN_REAL_QTIP:
    if not Path("QTIP").is_dir():
        subprocess.run(["git", "clone", "https://github.com/Cornell-RelaxML/QTIP.git"], check=True)
    # Blackwell (sm_120) has no flash-attn wheel -> force SDPA so QTIP doesn't ImportError.
    subprocess.run(["bash", "-lc",
        "grep -rl flash_attention_2 QTIP 2>/dev/null | xargs -r sed -i 's/flash_attention_2/sdpa/g'; "
        "grep -rl 'import flash_attn' QTIP 2>/dev/null | xargs -r sed -i "
        "'s/^import flash_attn.*/flash_attn = None  # disabled on Blackwell/'"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "QTIP/requirements.txt"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "glog", "primefac"], check=False)
    kbuild = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "QTIP/qtip-kernels"],
                            capture_output=True, text=True)
    results_qtip["qtip_kernels_build_rc"] = kbuild.returncode
    if kbuild.returncode != 0:
        print("[warn] qtip-kernels build failed -> bracket only:", kbuild.stderr[-500:])
    else:
        os.chdir("QTIP")
        def _qrun(mod, *a):
            r = subprocess.run([sys.executable, "-m", mod, *a], capture_output=True, text=True)
            if r.returncode != 0:
                results_qtip.setdefault("qtip_errs", {})[mod] = (r.stderr or r.stdout)[-2500:]
                print(f"[{mod} FAILED rc={r.returncode}]\n", (r.stderr or r.stdout)[-2500:])
            return r.returncode
        rc_h = 0 if HESS.exists() else _qrun("quantize_llama.input_hessian_llama",
            "--base_model", MODEL_ID, "--save_path", str(HESS), "--devset_size", "256", "--ctx_size", "1024")
        results_qtip["hessian_rc"] = rc_h; rc_q = rc_hf = 1
        if rc_h == 0:
            rc_q = 0 if CKPT.exists() else _qrun("quantize_llama.quantize_finetune_llama",
                "--base_model", MODEL_ID, "--in_hess_path", str(HESS), "--save_path", str(CKPT),
                "--codebook", "bitshift", "--scale_override", "0.9", "--ft_epochs", "5",
                "--td_x", "16", "--td_y", "16", "--L", "16", "--K", str(BITS), "--V", "2",
                "--decode_mode", "quantlut_sym", "--tlut_bits", "9")
            results_qtip["quant_rc"] = rc_q
        if rc_q == 0:
            rc_hf = _qrun("quantize_llama.hfize_llama", "--quantized_path", str(CKPT),
                          "--hf_output_path", str(HF_DIR))
            results_qtip["hfize_rc"] = rc_hf
        os.chdir("..")
        results_qtip["real_qtip_ran"] = (rc_h == 0 and rc_q == 0 and rc_hf == 0 and HF_DIR.is_dir())
        if results_qtip["real_qtip_ran"] and CACHE is not None:
            try:
                for art in (HF_DIR, CKPT, HESS):
                    if art.exists(): shutil.copytree(art, CACHE / art.name, dirs_exist_ok=True)
                print("saved QTIP artifacts to Drive cache.")
            except Exception as e:
                print("[Drive save failed]", repr(e)[:160])
    print("real QTIP HF model ready:", results_qtip["real_qtip_ran"])
else:
    print("RUN_REAL_QTIP=False -> bracket only; QTIP verdict NEEDS-MEASUREMENT.")

# %%
# --- GATE 1: QTIP LEG 1 (weight-RMSE) + LEG 2 (logits) + LEG 3 (PPL) + VERDICT-
try:
    from safetensors import safe_open
    from gguf import GGUFReader, dequantize
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import gc

    HF_SAMPLE = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight", "model.layers.17.self_attn.o_proj.weight",
        "model.layers.17.mlp.up_proj.weight", "model.layers.35.self_attn.q_proj.weight",
        "model.layers.35.mlp.down_proj.weight"]
    GGUF_OF = {"model.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight",
        "model.layers.0.mlp.gate_proj.weight": "blk.0.ffn_gate.weight",
        "model.layers.0.mlp.down_proj.weight": "blk.0.ffn_down.weight",
        "model.layers.17.self_attn.o_proj.weight": "blk.17.attn_output.weight",
        "model.layers.17.mlp.up_proj.weight": "blk.17.ffn_up.weight",
        "model.layers.35.self_attn.q_proj.weight": "blk.35.attn_q.weight",
        "model.layers.35.mlp.down_proj.weight": "blk.35.ffn_down.weight"}
    snap = snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json"])
    def _open_f16(name):
        for st in Path(snap).glob("*.safetensors"):
            with safe_open(str(st), framework="pt") as f:   # framework=pt: bf16-safe (numpy can't)
                if name in f.keys():
                    return f.get_tensor(name).to(torch.float32).cpu().numpy()
        return None
    q4by = {t.name: t for t in GGUFReader(Q4KM_GGUF).tensors}
    def _deq_flat(g):
        t = q4by.get(g)
        return (None, None) if t is None else (
            dequantize(np.array(t.data), t.tensor_type).astype(np.float32).ravel(), t.tensor_type.name)
    rmse_rows = []
    for hf in HF_SAMPLE:
        Wf16 = _open_f16(hf); flat, disk = _deq_flat(GGUF_OF[hf])
        if Wf16 is None or flat is None or flat.size != Wf16.size or Wf16.shape[1] % QK:
            print("[skip]", hf); continue
        ca = flat.reshape(Wf16.shape); cb = flat.reshape(Wf16.shape[::-1]).T  # GGUF<->HF transpose
        Wq4 = ca if rel_rmse(ca, Wf16) <= rel_rmse(cb, Wf16) else cb
        lo, _, qtb = qtip_bracket(Wf16, BITS, BITS); hi, _, _ = qtip_bracket(Wf16, BITS, BITS + 1)
        row = dict(name=hf, disk=disk, q4k_rmse=rel_rmse(Wq4, Wf16),
                   qtip_bracket_lower=rel_rmse(lo, Wf16), qtip_bracket_upper=rel_rmse(hi, Wf16),
                   qtip_block_bytes=qtb, q4k_block_bytes=144.0)
        if results_qtip.get("real_qtip_ran"):
            Wqt = None
            for st in Path(HF_DIR).glob("*.safetensors"):
                with safe_open(str(st), framework="pt") as f:
                    if hf in f.keys():
                        Wqt = f.get_tensor(hf).to(torch.float32).cpu().numpy(); break
            if Wqt is not None and Wqt.shape == Wf16.shape:
                row["qtip_real_rmse"] = rel_rmse(Wqt, Wf16)
            del Wqt
        rmse_rows.append(row); del Wf16, flat, lo, hi; gc.collect()
    results_qtip["rmse_per_tensor"] = rmse_rows
    med = lambda k: float(np.median([r[k] for r in rmse_rows if k in r])) if rmse_rows else float("nan")
    m_q4 = med("q4k_rmse"); results_qtip["median_q4k_rmse"] = m_q4
    bn = lambda r: float(np.log2(r / m_q4)) if (r > 0 and m_q4 > 0) else float("nan")
    results_qtip["bits_needed_bracket"] = [bn(med("qtip_bracket_lower")), bn(med("qtip_bracket_upper"))]
    if results_qtip.get("real_qtip_ran") and all("qtip_real_rmse" in r for r in rmse_rows):
        results_qtip["bits_needed_real"] = bn(med("qtip_real_rmse"))

    # LEG 2 logits + LEG 3 PPL (f16 via transformers, Q4_K_M/QTIP).
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    code_eval = Path(EVAL).read_text(encoding="utf-8", errors="replace")
    ids = tok(code_eval, return_tensors="pt", add_special_tokens=False).input_ids[:, :1024]
    @torch.inference_mode()
    def _next(model):
        return model(ids.to(model.device)).logits[0].float().cpu().numpy()[:-1]
    m_f16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                                 device_map="cuda", trust_remote_code=True)
    L_f16 = _next(m_f16); del m_f16; torch.cuda.empty_cache()
    L_q4 = _gguf_logits(Q4KM_GGUF, n=ids.shape[1])[:L_f16.shape[0]]
    V = min(L_f16.shape[1], L_q4.shape[1]); L_f16c = L_f16[:, :V]; L_q4 = L_q4[:, :V]
    logit = {"cos_q4k": _cos(L_q4, L_f16c), "kl_q4k": _kl(L_f16c, L_q4), "argmax_q4k": _argmax_agree(L_f16c, L_q4)}
    if results_qtip.get("real_qtip_ran"):
        m_qt = AutoModelForCausalLM.from_pretrained(str(HF_DIR), torch_dtype=torch.float16, device_map="cuda")
        L_qt = _next(m_qt)[:, :V]; del m_qt; torch.cuda.empty_cache()
        logit.update(cos_qtip=_cos(L_qt, L_f16c), kl_qtip=_kl(L_f16c, L_qt), argmax_qtip=_argmax_agree(L_f16c, L_qt))
    results_qtip["logit"] = logit
    for _n in ("L_f16", "L_f16c", "L_q4", "L_qt"): globals().pop(_n, None)
    gc.collect(); torch.cuda.empty_cache()

    results_qtip["ppl_q4k"] = _ppl(Q4KM_GGUF); results_qtip["ppl_f16"] = _ppl(SRC_GGUF)
    if results_qtip.get("real_qtip_ran"):
        @torch.inference_mode()
        def _hfppl(model, text, max_len=2048, stride=512, max_eval=16384):
            iid = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :max_eval].to(model.device)
            nlls, n, prev = [], 0, 0
            for beg in range(0, iid.size(1), stride):
                end = min(beg + max_len, iid.size(1)); trg = end - prev
                inp = iid[:, beg:end]; tgt = inp.clone(); tgt[:, :-trg] = -100
                nlls.append(model(inp, labels=tgt).loss.float() * trg); n += trg; prev = end
                if end == iid.size(1): break
            return float(torch.exp(torch.stack(nlls).sum() / n))
        m_qt = AutoModelForCausalLM.from_pretrained(str(HF_DIR), torch_dtype=torch.float16, device_map="cuda")
        results_qtip["ppl_qtip"] = _hfppl(m_qt, code_eval); del m_qt; torch.cuda.empty_cache()

    lg = results_qtip.get("logit", {})
    leg1 = (results_qtip["bits_needed_real"] <= GATE_BITS_NEEDED) if "bits_needed_real" in results_qtip else None
    leg2 = ((lg["cos_qtip"] >= lg["cos_q4k"] and lg["argmax_qtip"] >= lg["argmax_q4k"]
             and lg["kl_qtip"] <= lg["kl_q4k"]) if "cos_qtip" in lg else None)
    leg3 = ((results_qtip["ppl_qtip"] <= GATE_PPL_RATIO * results_qtip["ppl_f16"]
             and results_qtip["ppl_qtip"] <= results_qtip["ppl_q4k"]) if results_qtip.get("ppl_qtip") else None)
    results_qtip["legs"] = {"weight_rmse_le0": leg1, "logit_ge_q4k": leg2, "ppl_le_gate": leg3}
    if results_qtip.get("real_qtip_ran") and None not in (leg1, leg2, leg3):
        results_qtip["verdict"] = "GO" if (leg1 and leg2 and leg3) else "NO-GO"
        if results_qtip["verdict"] == "NO-GO":
            results_qtip["kill_type"] = "Type-1 (quality): real QTIP-3 from f16 does not match Q4_K_M on code"
    else:
        results_qtip["verdict"] = "NEEDS-MEASUREMENT"
        results_qtip["decisive_gate"] = "real QTIP codec did not run; bracket alone cannot pass/kill (Type-2)."
except Exception as e:
    results_qtip["verdict"] = "ERROR"; results_qtip["error"] = repr(e)[:400]
    print("[GATE 1 QTIP ERROR]", repr(e)[:400])

Path("/content/qtip_3bit_results.json").write_text(json.dumps(results_qtip, indent=2))
print("\n[GATE 1 — QTIP] verdict:", results_qtip["verdict"])
print(json.dumps(results_qtip, indent=2))
try:
    from google.colab import files; files.download("/content/qtip_3bit_results.json")
except Exception: pass

# %% [markdown]
# ## Done — two verdicts, two JSONs
# - `imatrix_mixprec_results.json` (Gate 2) and `qtip_3bit_results.json` (Gate 1)
#   auto-download at the end of their cells.
# - **GO / NO-GO / NEEDS-MEASUREMENT** per gate; each is independently guarded so
#   one failing still yields the other. The QTIP codec is cached to Drive
#   (`MyDrive/qtip_gate_cache`) — a disconnect resumes instantly.
# - Send both JSONs back to resume Phase 3 (act on verdicts → consolidate → wipe).
