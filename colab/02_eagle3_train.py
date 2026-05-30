# %% [markdown]
# # Stage 4 — EAGLE spec-decode head for Qwen2.5-3B (retrain on Q4_K_M captures)
#
# **Why this exists (measured this session):** the existing
# `checkpoints/eagle5_final/q3b/head_final.safetensors` *loads* fine but gives
# **0.000 acceptance** on Qwen-3B code and makes decode **4.5× slower** (34→7.6 tps).
# Oracle A also ruled out the free n-gram path (τ=1.43 NO-GO). So spec needs a head
# that actually accepts — this notebook orchestrates the repo's **existing** trainer
# with the two fixes the 0%-accept points to.
#
# **This is an ORCHESTRATOR** of scripts that already exist in the repo
# (`colab/eagle5_train_pytorch.py`, `colab/eagle5_tau_eval_pytorch.py`,
# `colab/mega_calibrate.py`, `tools/training/build_qwen3b_frozen.py`). It does not
# reinvent them — it wires them with the right config + **two hard gates**.
#
# **Two root-cause gates (the checks-and-balances):**
# 1. **Capture provenance** — training residuals MUST come from dismantle's **Q4_K_M**
#    capture mode, not f16. f16-trained / Q4_K_M-served is the distribution shift that
#    kills acceptance (see `memory/eagle5_port_phase_a1_shipped`).
# 2. **Head↔runtime parity** — before trusting τ, confirm the Rust runtime computes the
#    SAME logits as the trained head via `cargo test eagle5_forward_parity` on the M3.
#    If parity fails, the 0% is an integration bug, not the head.
#
# **Gate to ship:** τ (mean accepted length) **≥ 2.5** at depth K on code AND parity passes.

# %%
# --- 0. GPU check ---
import torch, os, sys, subprocess, json, re
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > GPU."
print("GPU:", torch.cuda.get_device_name(0), "torch", torch.__version__)

# %%
# --- 1. Get repo scripts + artifacts onto Colab ---
# Option A (git): set REPO_URL to your pushed dismantle remote, then this clones it.
# Option B (upload): leave REPO_URL empty and upload a tarball of `colab/` +
#   `tools/training/` + the frozen npz + the capture-corpus parquet dir to /content.
REPO_URL = ""  # e.g. "https://<token>@github.com/you/dismantle.git"
if REPO_URL:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/dismantle"], check=True)
    os.chdir("/content/dismantle")
else:
    print("Upload to /content: colab/eagle5_train_pytorch.py, eagle5_tau_eval_pytorch.py,")
    print("mega_calibrate.py, tools/training/build_qwen3b_frozen.py, the frozen .npz,")
    print("and the capture-corpus parquet dir. Then set paths in cell 3.")
sys.path.insert(0, "colab")

# %%
# --- 2. Deps (mirror the trainer's contract) ---
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "numpy<2.2", "pyarrow", "safetensors", "transformers", "huggingface_hub"],
               check=False)
import pyarrow  # noqa

# %% [markdown]
# ## GATE 1 — capture provenance (DO THIS ON THE M3 FIRST)
# The parquet corpus must hold **Q4_K_M** residual captures. Produce on the M3:
# ```bash
# # 1) capture Q4_K_M hidden-state residuals over a CODE corpus
# DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_Q4K_PREDEC=1 \
# DISMANTLE_QWEN_EAGLE5_CAPTURE=1 DISMANTLE_QWEN_EAGLE5_CAPTURE_LAYER=32 \
# DISMANTLE_QWEN_CAPTURE_CORPUS_PATH=artifacts/eagle5/captures \
#   ./target/release/dismantle generate \
#     --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
#     --kernel-profile profiles/qwen3b-instruct-q4k.m3pro18.json \
#     --prompts-file artifacts/quant/calib_code.txt --max-new-tokens 64 --temperature 0
# # 2) pack to parquet shards (trainer input contract)
# python3 colab/mega_calibrate.py --captures artifacts/eagle5/captures --out artifacts/eagle5/corpus
# # 3) frozen base tensors (token_embd / lm_head / output_norm) from the GGUF
# python3 tools/training/build_qwen3b_frozen.py \
#   --gguf models/qwen2.5-3b-instruct-q4_k_m.gguf --out artifacts/eagle5/qwen3b_frozen.npz
# ```
# Upload `artifacts/eagle5/corpus` and `qwen3b_frozen.npz` to Colab.

# %%
# --- 3. Paths + config (Qwen-3B preset per memory/eagle5_train_qwen3b_adapter_notes) ---
CORPUS_DIR = "artifacts/eagle5/corpus"          # parquet shards from mega_calibrate.py
FROZEN     = "artifacts/eagle5/qwen3b_frozen.npz"
CKPT_DIR   = "artifacts/eagle5/ckpt_q4km"
CAPTURE_LAYER = 32      # must match the M3 capture layer
DEPTH      = 4          # τ eval depth (K)
TAU_GATE   = 2.5
for p in (CORPUS_DIR, FROZEN):
    assert os.path.exists(p), f"missing {p} — see GATE 1 (produce on M3, upload)."
# provenance assertion: mega_calibrate stamps source precision; refuse f16 corpus.
meta = os.path.join(CORPUS_DIR, "meta.json")
if os.path.exists(meta):
    src = json.load(open(meta)).get("source_precision", "unknown")
    assert "q4" in src.lower(), f"corpus source_precision={src!r} — MUST be Q4_K_M (GATE 1)."
    print("provenance OK:", src)
else:
    print("WARN: no corpus meta.json — manually confirm residuals are Q4_K_M captures.")

# %%
# --- 4. Train (wraps colab/eagle5_train_pytorch.py) ---
cmd = [sys.executable, "colab/eagle5_train_pytorch.py",
       "--corpus-dir", CORPUS_DIR, "--frozen", FROZEN, "--ckpt-dir", CKPT_DIR,
       "--epochs", "8", "--batch-size", "24", "--seq-len", "16",
       "--capture-layer", str(CAPTURE_LAYER), "--num-blocks", "1",
       "--target-mode", "corpus"]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
HEAD = os.path.join(CKPT_DIR, "head_final.safetensors")
assert os.path.exists(HEAD), "training did not emit head_final.safetensors"
print("trained head ->", HEAD, f"({os.path.getsize(HEAD)/1e6:.0f} MB)")

# %%
# --- 5. GATE 2a — τ (mean accepted length) on a held-out code slice ---
TAU_OUT = "artifacts/eagle5/tau.json"
subprocess.run([sys.executable, "colab/eagle5_tau_eval_pytorch.py",
                "--ckpt", CKPT_DIR, "--frozen", FROZEN, "--corpus", CORPUS_DIR,
                "--out", TAU_OUT, "--depth", str(DEPTH)], check=True)
tau_doc = json.load(open(TAU_OUT))
tau = tau_doc.get("tau") or tau_doc.get("mean_accepted_len") or \
      tau_doc.get(f"tau_depth_{DEPTH}")
print("τ doc:", json.dumps(tau_doc, indent=2)[:600])
verdict = "GO" if (tau and tau >= TAU_GATE) else "NO-GO"
print(f"\nτ@K={DEPTH} = {tau}  (gate ≥{TAU_GATE})  ->  {verdict}")

# %% [markdown]
# ## GATE 2b — head↔runtime parity (run on the M3, BEFORE believing τ)
# τ here is PyTorch-side. The 0%-accept could be a Rust-runtime/head logit mismatch.
# Confirm they agree before shipping:
# ```bash
# # make a fixture from the new head, then run the parity test
# python3 tools/eagle5_forward_dump.py --head <downloaded head_final.safetensors> \
#   --out crates/dismantle-core/tests/fixtures/eagle5_parity_q3b.json --seed 0xea91e5
# cargo test -p dismantle-core --test eagle5_forward_parity -- --nocapture
# ```
# Parity PASS + τ≥2.5 ⇒ ship. Parity FAIL ⇒ fix the runtime forward (`speculate/eagle5_forward.rs`),
# not the head.

# %%
# --- 6. Package + record ---
res = {"head": HEAD, "tau": tau, "depth": DEPTH, "tau_gate": TAU_GATE,
       "verdict": verdict, "capture_layer": CAPTURE_LAYER,
       "prior_head_accept": 0.0, "ngram_oracle_tau": 1.43,
       "next": "M3: eagle5_forward_parity test, then tools/bench/eagle5_paired_bench.sh"}
json.dump(res, open("artifacts/eagle5/eagle3_train_result.json", "w"), indent=2)
print(json.dumps(res, indent=2))
try:
    from google.colab import files
    files.download(HEAD); files.download("artifacts/eagle5/eagle3_train_result.json")
except Exception:
    pass

# %% [markdown]
# ## M3 final measurement (the loop closes here)
# ```bash
# WEIGHTS=models/qwen2.5-3b-instruct-q4_k_m.gguf \
# PROFILE=profiles/qwen3b-instruct-q4k.m3pro18.json \
# EAGLE5_HEAD=<new head_final.safetensors> PROMPT='def quicksort(arr):' \
#   bash tools/bench/eagle5_paired_bench.sh
# ```
# Expect accepted-length ≫ 0 and dec_tps **above** the no-spec baseline (the existing
# head gave 0.000 / 4.5× slower — that is the bar to beat). Report under the §1 gate.
