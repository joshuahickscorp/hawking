# %% [markdown]
# # Stage 4 — EAGLE head retrain on Q4_K_M captures
#
# This notebook is an artifact producer, not a runtime benchmark. It trains a
# Qwen2.5-3B draft head from **dismantle Q4_K_M residual captures**, evaluates a
# runtime-predictive accepted-prefix metric, and writes a safetensors head for the
# M3 parity/runtime loop.
#
# **Why this exists:** the existing `checkpoints/eagle5_final/q3b/head_final.safetensors`
# loaded but produced 0.000 acceptance and made decode 4.5x slower. The likely
# failure mode is f16/capture mismatch or head/runtime parity mismatch.
#
# **Ship gate:** tau >= 2.5 on held-out code captures **and** M3
# `eagle5_forward_parity` passes.

# %%
# --- 0. Dependency + GPU preflight ---
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def run(cmd, *, check=True):
    cmd = [str(x) for x in cmd]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


# numpy==1.26.4 can't load into a kernel that already imported numpy 2.x (Colab
# default) — surfaces as "cannot import name '_center'". Install + restart once;
# a sentinel skips the reinstall on the second pass. ("session crashed" = the
# restart; just Run-All again, installs are cached.)
_DEPS_SENTINEL = "/content/.dismantle_eagle_deps_ready"
if not os.path.exists(_DEPS_SENTINEL):
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "numpy==1.26.4",
            "pyarrow>=15,<25",
            "safetensors>=0.5,<0.8",
            "tqdm>=4.66",
        ]
    )
    Path(_DEPS_SENTINEL).touch()
    print("\n*** deps installed — RESTARTING runtime once for a clean numpy. ***")
    print("*** when it reconnects, just Run-All again (cached). ***", flush=True)
    os.kill(os.getpid(), 9)

import pyarrow  # noqa: F401
import torch

assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > GPU."
GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU_NAME}  VRAM={VRAM_GB:.1f} GB  torch={torch.__version__}")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def run_with_heartbeat(cmd, label, interval_sec=30):
    cmd = [str(x) for x in cmd]
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    stop = threading.Event()
    start = time.time()

    def heartbeat():
        while not stop.wait(interval_sec):
            elapsed = (time.time() - start) / 60
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                ).strip()
                util, mem_used, mem_total = [x.strip() for x in out.split(",")]
                gpu = f"GPU util={util}% mem={mem_used}/{mem_total} MB"
            except Exception as e:
                gpu = f"GPU query failed: {e}"
            print(f"[{label}] RUNNING elapsed={elapsed:.1f}m {gpu}", flush=True)

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()
    try:
        rc = proc.wait()
    finally:
        stop.set()
        t.join(timeout=2)
    elapsed = (time.time() - start) / 60
    print(f"[{label}] finished rc={rc} elapsed={elapsed:.1f}m", flush=True)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def kill_stale_trainers():
    try:
        out = subprocess.check_output(["pgrep", "-f", "eagle5_train_pytorch.py"], text=True).strip()
    except subprocess.CalledProcessError:
        return
    pids = [int(p) for p in out.splitlines() if p.strip()]
    if not pids:
        return
    print("[kill-stale] terminating leftover trainers:", pids, flush=True)
    for pid in pids:
        if pid != os.getpid():
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(3)

# %%
# --- 1. Get repo scripts onto Colab ---
# Default clones the branch that contains the three Bible notebooks. If you are
# testing local edits, upload a repo tarball to /content/dismantle and set
# REPO_URL="" before running this cell.
REPO_URL = "https://github.com/joshuahickscorp/dismantle.git"
REPO_BRANCH = "codex/maximal-spec-colab"
REPO_DIR = Path("/content/dismantle")

if REPO_URL:
    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO_DIR)])
    else:
        run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH, "--depth", "1"])
        run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{REPO_BRANCH}"])
    os.chdir(REPO_DIR)
else:
    assert REPO_DIR.exists(), "Upload repo to /content/dismantle or set REPO_URL."
    os.chdir(REPO_DIR)

required = [
    Path("colab/eagle5_train_pytorch.py"),
    Path("colab/eagle5_tau_eval_pytorch.py"),
    Path("colab/mega_calibrate.py"),
    Path("tools/training/build_qwen3b_frozen.py"),
]
missing = [str(p) for p in required if not p.exists()]
assert not missing, f"repo checkout missing required scripts: {missing}"
sys.path.insert(0, str(Path("colab").resolve()))
print("repo ready:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip())

# %% [markdown]
# ## Gate 1 — Capture Provenance
# Produce these on the M3 first, then upload to `/content/artifacts/eagle5/`:
#
# ```bash
# DISMANTLE_QWEN_TCB=1 DISMANTLE_QWEN_Q4K_PREDEC=1 \
# DISMANTLE_QWEN_EAGLE5_CAPTURE=1 DISMANTLE_QWEN_EAGLE5_CAPTURE_LAYER=32 \
# DISMANTLE_QWEN_CAPTURE_CORPUS_PATH=artifacts/eagle5/captures \
#   ./target/release/dismantle generate \
#     --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
#     --kernel-profile profiles/qwen3b-instruct-q4k.m3pro18.json \
#     --prompts-file artifacts/quant/calib_code.txt --max-new-tokens 64 --temperature 0
#
# python3 colab/mega_calibrate.py --captures artifacts/eagle5/captures --out artifacts/eagle5/corpus
# python3 tools/training/build_qwen3b_frozen.py \
#   --gguf models/qwen2.5-3b-instruct-q4_k_m.gguf --out artifacts/eagle5/qwen3b_frozen.npz
# ```

# %%
# --- 2. Paths + training preset ---
ART = Path("/content/artifacts/eagle5")
CORPUS_DIR = ART / "corpus"
FROZEN = ART / "qwen3b_frozen.npz"
CKPT_DIR = ART / "ckpt_q4km"
CAPTURE_LAYER = 32
DEPTH = 4
TAU_GATE = 2.5
ALLOW_UNSTAMPED_CORPUS = False

# Proven baseline from corrected headbank: chained-hidden rollout is the key
# speedup driver. Ordered for performance per compute unit, not exhaustive grid.
TRAIN = {
    "epochs": 12,
    "batch_size": 24 if VRAM_GB >= 35 else (16 if VRAM_GB >= 22 else 8),
    "seq_len": 16,
    "lr": 1e-3,
    "max_rows": 8000 if VRAM_GB >= 22 else 4000,
    "max_row_tokens": 128,
    "num_blocks": 1,
    "head_heads": 16,
    "head_ff_mult": 4.0,
    "calib_loss_weight": 0.1,
    "residual_delta_loss_weight": 0.0,
    "rollout_loss_weight": 1.0,
    "rollout_depth": 5,
    "rollout_depth_targets": "1,2,3,4",
    "rollout_draft_prob": 0.75,
}

for p in (CORPUS_DIR, FROZEN):
    assert p.exists(), f"missing {p} — produce on M3 and upload before training."

shards = sorted(CORPUS_DIR.glob("shard_*.parquet"))
assert shards, f"no parquet shards found under {CORPUS_DIR}"
meta = CORPUS_DIR / "meta.json"
if meta.exists():
    src = json.loads(meta.read_text()).get("source_precision", "unknown")
    assert "q4" in src.lower(), f"corpus source_precision={src!r}; MUST be Q4_K_M"
    print("provenance OK:", src)
elif not ALLOW_UNSTAMPED_CORPUS:
    raise SystemExit("missing corpus meta.json; set ALLOW_UNSTAMPED_CORPUS=True only after manual Q4_K_M verification")
else:
    print("WARN: proceeding with unstamped corpus by explicit override")

CKPT_DIR.mkdir(parents=True, exist_ok=True)
print("shards:", len(shards))
print("train config:", json.dumps(TRAIN, indent=2))

# %%
# --- 3. Train (writes latest.npz AND head_final.safetensors) ---
kill_stale_trainers()
torch.cuda.empty_cache()
cmd = [
    sys.executable,
    "-u",
    "colab/eagle5_train_pytorch.py",
    "--corpus-dir",
    str(CORPUS_DIR),
    "--frozen",
    str(FROZEN),
    "--ckpt-dir",
    str(CKPT_DIR),
    "--device",
    "cuda",
    "--target-mode",
    "corpus",
    "--capture-layer",
    str(CAPTURE_LAYER),
    "--epochs",
    str(TRAIN["epochs"]),
    "--batch-size",
    str(TRAIN["batch_size"]),
    "--seq-len",
    str(TRAIN["seq_len"]),
    "--lr",
    str(TRAIN["lr"]),
    "--max-rows",
    str(TRAIN["max_rows"]),
    "--max-row-tokens",
    str(TRAIN["max_row_tokens"]),
    "--num-blocks",
    str(TRAIN["num_blocks"]),
    "--head-heads",
    str(TRAIN["head_heads"]),
    "--head-ff-mult",
    str(TRAIN["head_ff_mult"]),
    "--calib-loss-weight",
    str(TRAIN["calib_loss_weight"]),
    "--residual-delta-loss-weight",
    str(TRAIN["residual_delta_loss_weight"]),
    "--rollout-loss-weight",
    str(TRAIN["rollout_loss_weight"]),
    "--rollout-depth",
    str(TRAIN["rollout_depth"]),
    "--rollout-depth-targets",
    TRAIN["rollout_depth_targets"],
    "--rollout-draft-prob",
    str(TRAIN["rollout_draft_prob"]),
    "--rollout-chain-hidden",
    "--save-safetensors",
]
run_with_heartbeat(cmd, "train", interval_sec=30)

HEAD = CKPT_DIR / "head_final.safetensors"
LATEST = CKPT_DIR / "latest.npz"
assert HEAD.exists(), "training did not emit head_final.safetensors"
assert LATEST.exists(), "training did not emit latest.npz"
assert HEAD.stat().st_size > 1_000_000, f"head_final.safetensors too small: {HEAD.stat().st_size}"
print("trained head ->", HEAD, f"({HEAD.stat().st_size / 1e6:.0f} MB)")

# %%
# --- 4. Gate 2a — runtime-predictive tau on held-out windows ---
TAU_OUT = ART / "tau.json"
eval_cmd = [
    sys.executable,
    "-u",
    "colab/eagle5_tau_eval_pytorch.py",
    "--ckpt",
    str(HEAD),
    "--frozen",
    str(FROZEN),
    "--corpus",
    str(CORPUS_DIR),
    "--out",
    str(TAU_OUT),
    "--depth",
    str(DEPTH),
    "--target-mode",
    "corpus",
    "--chain-hidden",
    "--max-windows",
    "4000",
]
run_with_heartbeat(eval_cmd, "tau", interval_sec=30)
tau_doc = json.loads(TAU_OUT.read_text())
tau = None
for key in ("tau", "mean_accepted_len", f"tau_depth_{DEPTH}"):
    if key in tau_doc and tau_doc[key] is not None:
        tau = float(tau_doc[key])
        break
assert tau is not None, f"tau eval did not report a tau field: {tau_doc.keys()}"
verdict = "GO" if tau >= TAU_GATE else "NO-GO"
print("tau doc:", json.dumps(tau_doc, indent=2)[:1200])
print(f"\ntau@K={DEPTH} = {tau:.3f}  gate >= {TAU_GATE}  ->  {verdict}")

# %% [markdown]
# ## Gate 2b — M3 Head/Runtime Parity
# Run this on the M3 before trusting any speedup claim:
#
# ```bash
# python3 tools/eagle5_forward_dump.py --head <downloaded head_final.safetensors> \
#   --out crates/dismantle-core/tests/fixtures/eagle5_parity_q3b.json --seed 0xea91e5
# cargo test -p dismantle-core --test eagle5_forward_parity -- --nocapture
# ```
#
# Parity PASS + tau >= 2.5 means proceed to paired runtime bench. Parity FAIL means
# fix the Rust/Metal forward path, not the Colab head.

# %%
# --- 5. Package + record ---
res = {
    "head": str(HEAD),
    "latest_npz": str(LATEST),
    "tau": tau,
    "depth": DEPTH,
    "tau_gate": TAU_GATE,
    "verdict": verdict,
    "capture_layer": CAPTURE_LAYER,
    "train": TRAIN,
    "prior_head_accept": 0.0,
    "ngram_oracle_tau": 1.43,
    "next": "M3: eagle5_forward_parity, then tools/bench/eagle5_paired_bench.sh",
}
out = ART / "eagle3_train_result.json"
out.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
try:
    from google.colab import files

    files.download(str(HEAD))
    files.download(str(out))
except Exception:
    pass

# %% [markdown]
# ## M3 Final Measurement
# ```bash
# WEIGHTS=models/qwen2.5-3b-instruct-q4_k_m.gguf \
# PROFILE=profiles/qwen3b-instruct-q4k.m3pro18.json \
# EAGLE5_HEAD=<new head_final.safetensors> PROMPT='def quicksort(arr):' \
#   bash tools/bench/eagle5_paired_bench.sh
# ```
#
# Report the paired bench only after `tools/bench/analyze_tcb_trace.py` passes.
