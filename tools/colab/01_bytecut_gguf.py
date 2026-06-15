# %% [markdown]
# # Stage 3 — byte-cut quality gate (llama.cpp, dependency-free)
#
# **Restructured for robustness.** No transformers / autoawq / gptqmodel / numpy
# pins — just **llama.cpp** (a C++ binary), so there is no Python dependency hell.
# It quantizes Qwen2.5-3B below 4 bits and measures **code perplexity** with the
# same `llama-perplexity` tool as the local oracle, so the numbers are directly
# comparable (local: Q4_K_M=4.485, Q3-from-Q4=5.915, naive-Q3=11.432).
#
# **Why this is also the *right* tool:** dismantle loads GGUF. These outputs
# (Q3_K_M / IQ3) are directly loadable — unlike AWQ/GPTQ which needed a custom
# M3 kernel. So a GO here is much closer to shippable.
#
# **Produces:** `bytecut_results.json` + the quantized `.gguf` files.
# **Gate:** Q3_K_M(+imatrix, from near-lossless source) should land **well below**
# the local Q3-from-Q4 of 5.915 and **near** Q4_K_M's 4.485 → sub-4-bit is viable.
# **Runtime:** ~15-30 min. GPU runtime speeds perplexity but CPU also works.

# %%
# --- 0. Build llama.cpp (CUDA if a GPU is present, else CPU) ---
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
# huggingface_hub is light and reliable; no model-framework deps.
subprocess.run(["pip", "install", "-q", "huggingface_hub"], check=False)
from huggingface_hub import list_repo_files, hf_hub_download

REPOS = ["Qwen/Qwen2.5-3B-Instruct-GGUF", "bartowski/Qwen2.5-3B-Instruct-GGUF"]
PREF = ["q8_0", "f16", "fp16"]   # near-lossless => acts as the f16 reference
def find_source():
    for repo in REPOS:
        try:
            ggufs = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
        except Exception as e:
            print("skip", repo, e); continue
        single = [f for f in ggufs if "-of-" not in f]   # avoid split shards
        for pref in PREF:
            hit = sorted((f for f in single if pref in f.lower()), key=len)
            if hit:
                return repo, hit[0]
    return None, None
repo, fname = find_source()
assert fname, f"no single-file q8_0/f16 GGUF in {REPOS}; edit REPOS"
print("source:", repo, "/", fname)
SRC = hf_hub_download(repo, fname)
print("downloaded:", SRC)

# %%
# --- 2. Fetch the code corpora (same as local oracle C) ---
RAW = ("https://raw.githubusercontent.com/joshuahickscorp/dismantle/"
       "codex/maximal-spec-colab/colab/data")
import urllib.request
for name in ("calib_trim.txt", "ppl_trim.txt"):
    if not Path("/content", name).exists():
        urllib.request.urlretrieve(f"{RAW}/{name}", f"/content/{name}")
CALIB, EVAL = "/content/calib_trim.txt", "/content/ppl_trim.txt"
print("corpora ready")

# %%
# --- 3. Helpers: imatrix, quantize, perplexity ---
NGL = ["-ngl", "99"] if HAS_GPU else []
def sh(cmd):
    print("$", " ".join(cmd)); return subprocess.run(cmd, capture_output=True, text=True)
def imatrix(out):
    r = sh([f"{BIN}/llama-imatrix", "-m", SRC, "-f", CALIB, "-o", out,
            "--chunks", "80", *NGL])
    assert Path(out).exists(), r.stderr[-800:]
    return out
def quantize(out, qtype, imat=None):
    cmd = [f"{BIN}/llama-quantize", "--allow-requantize"]
    if imat: cmd += ["--imatrix", imat]
    cmd += [SRC, out, qtype]
    r = sh(cmd); assert Path(out).exists(), r.stderr[-800:]
    return out
def ppl(gguf):
    r = sh([f"{BIN}/llama-perplexity", "-m", gguf, "-f", EVAL, "--chunks", "50", *NGL])
    m = re.findall(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", r.stderr + r.stdout)
    return float(m[-1]) if m else None
def gib(p): return round(os.path.getsize(p) / 1024**3, 3)

# %%
# --- 4. Build imatrix + the sub-4-bit candidates, measure code PPL ---
IM = imatrix("/content/code.imatrix")
cands = [
    ("q3_k_m_imatrix", "Q3_K_M", IM),
    ("iq3_m_imatrix",  "IQ3_M",  IM),
    ("q3_k_m_naive",   "Q3_K_M", None),   # shows the imatrix benefit
    ("q2_k_naive",     "Q2_K",   None),   # the floor
]
results = {"source": f"{repo}/{fname}", "has_gpu": HAS_GPU, "ppl": {}, "gib": {},
           "local_oracle_c": {"q4km": 4.485, "q3_from_q4_imatrix": 5.915,
                              "q3_from_q4_naive": 11.432}}
for tag, qtype, imat in cands:
    out = f"/content/qwen3b-{tag}.gguf"
    quantize(out, qtype, imat)
    results["gib"][tag] = gib(out)
    results["ppl"][tag] = ppl(out)
    print(f"  {tag:18s} {qtype:7s} {results['gib'][tag]:.2f} GiB  PPL={results['ppl'][tag]}")

# %%
# --- 5. VERDICT ---
q3 = results["ppl"].get("q3_k_m_imatrix")
q4km = results["local_oracle_c"]["q4km"]          # 4.485 (what we ship)
q3_from_q4 = results["local_oracle_c"]["q3_from_q4_imatrix"]  # 5.915
if q3:
    beats_requant = q3 < q3_from_q4
    near_q4 = q3 <= q4km * 1.15
    results["verdict"] = "GO" if (beats_requant and near_q4) else \
                         ("PARTIAL" if beats_requant else "NO-GO")
    print(f"\nQ3_K_M+imatrix (from near-lossless): PPL={q3:.3f}")
    print(f"  vs local Q3-from-Q4 5.915 -> {'beats' if beats_requant else 'worse'}")
    print(f"  vs Q4_K_M 4.485 (+15% = {q4km*1.15:.3f}) -> {'near' if near_q4 else 'far'}")
    print(f"  VERDICT: {results['verdict']}")
else:
    results["verdict"] = "INCOMPLETE"
    print("PPL parse failed; check the perplexity cell output.")
Path("/content/bytecut_results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
try:
    from google.colab import files
    files.download("/content/bytecut_results.json")
except Exception:
    pass

# %% [markdown]
# ## What a GO means here
# These are **GGUF** files dismantle already understands. A GO (Q3_K_M+imatrix near
# Q4_K_M quality at ~25% fewer bytes) means: download the `.gguf`, confirm
# dismantle's loader handles Q3_K/IQ3 (it parses GGUF k-quants), and bench decode
# under the §1 gate (`tools/bench/analyze_tcb_trace.py`). No custom kernel needed —
# that is the whole advantage of staying in GGUF vs AWQ/GPTQ.
