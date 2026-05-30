# %% [markdown]
# # Stage 4 — EAGLE spec-decode draft head for Qwen2.5-3B
#
# Trains the Eagle5 v2 draft head on a parquet capture corpus, evaluates a
# runtime-predictive accepted-prefix metric (τ), and emits a
# `head_final.safetensors` for the M3 parity + paired-bench loop. This notebook
# is an **artifact producer**, not a runtime benchmark — the speedup is measured
# on the M3 afterwards.
#
# **Why this exists.** The existing head
# `checkpoints/eagle5_final/q3b/head_final.safetensors` *loads* but gives **0.000
# acceptance** and makes decode ~4.5× slower on Qwen-3B code. The prime suspect
# (see `plans/eagle_spec_handoff_2026_05_30.md`) is a **capture/serving precision
# mismatch**: the old head was trained on captures from a Hugging Face f16 model,
# but dismantle **serves Q4_K_M** (ggml k-quant), a different weight distribution.
#
# **Do the two cheap checks in the handoff FIRST** (on the M3, before spending any
# cloud time): `cargo test -p dismantle-core --test eagle5_forward_parity` (a
# parity FAIL means the 0% is an integration bug, not a data bug) and decide the
# capture source (next markdown cell). Then run this notebook.
#
# **Ship gate:** τ ≥ 2.5 at depth 4 on held-out captures **and** the M3
# `eagle5_forward_parity` test passes with the NEW head.

# %%
# --- Cell 0: minimal deps + GPU preflight ---
#
# DEPENDENCY POLICY (learned the hard way on notebook 01): use Colab's DEFAULT
# torch + numpy (2.x). DO NOT pin/downgrade numpy. DO NOT uninstall sklearn/
# scipy. DO NOT auto-restart the kernel. The eagle5 train/eval scripts only need
# numpy + torch (both pre-installed on Colab) plus pyarrow + safetensors, so we
# install ONLY those two on top of the stock stack and smoke-test imports here.
import importlib
import json
import os
import shutil
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


# Install only genuinely-missing packages. We probe first so a warm runtime
# (deps already present) is a no-op, and we never touch numpy/torch/sklearn.
_NEEDED = {"pyarrow": "pyarrow>=15", "safetensors": "safetensors>=0.4"}
_to_install = [spec for mod, spec in _NEEDED.items() if importlib.util.find_spec(mod) is None]
if _to_install:
    run([sys.executable, "-m", "pip", "install", "-q", *_to_install])
else:
    print("pyarrow + safetensors already present; skipping install.", flush=True)

# Smoke-test the whole import surface the scripts depend on. If any of these
# fail, the message points at the cell rather than letting the trainer die deep
# in a subprocess.
import numpy as np  # noqa: E402
import pyarrow  # noqa: E402,F401
import pyarrow.parquet  # noqa: E402,F401
import safetensors  # noqa: E402,F401
import torch  # noqa: E402

print(f"numpy={np.__version__}  torch={torch.__version__}  pyarrow={pyarrow.__version__}", flush=True)

assert torch.cuda.is_available(), (
    "No GPU detected. Runtime > Change runtime type > Hardware accelerator > GPU "
    "(L4/A100 ideal; T4 works but slower)."
)
GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {GPU_NAME}  VRAM={VRAM_GB:.1f} GB", flush=True)

# Deterministic across the whole notebook (trainer + eval also re-seed, but this
# pins any incidental host-side RNG and cuDNN/cuBLAS algorithm selection).
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def run_with_heartbeat(cmd, label, interval_sec=30):
    """Run a subprocess, streaming a GPU-utilization heartbeat so a long
    training run doesn't look hung on Colab. Raises on non-zero exit."""
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
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
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
    print(f"[{label}] finished rc={rc} elapsed={(time.time()-start)/60:.1f}m", flush=True)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def kill_stale_trainers():
    """Clear a previous interrupted train subprocess before launching a new one."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "colab/eagle5_train_pytorch.py"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return
    pids = [int(p) for p in out.splitlines() if p.strip()]
    if not pids:
        return
    print(f"[kill-stale] terminating leftover trainer pids: {pids}", flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

# %%
# --- Cell 1: clone the dismantle repo + assert the scripts are present ---
REPO_URL = "https://github.com/joshuahickscorp/dismantle.git"
REPO_BRANCH = "codex/maximal-spec-colab"
REPO_DIR = Path("/content/dismantle")

if not REPO_DIR.exists():
    run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO_DIR)])
else:
    # Warm runtime: refresh to the branch tip so a re-run picks up script fixes.
    run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH, "--depth", "1"], check=False)
    run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{REPO_BRANCH}"], check=False)
os.chdir(REPO_DIR)

# Put the colab dir on the path so eagle5_tau_eval_pytorch can
# `from eagle5_train_pytorch import ...`.
COLAB_DIR = (REPO_DIR / "colab").resolve()
if str(COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(COLAB_DIR))

REQUIRED = [
    Path("colab/eagle5_train_pytorch.py"),
    Path("colab/eagle5_tau_eval_pytorch.py"),
    Path("colab/mega_calibrate.py"),
    Path("tools/orchestrator/pack_corpus.py"),
    Path("tools/training/build_qwen3b_frozen.py"),
]
_missing = [str(p) for p in REQUIRED if not p.exists()]
assert not _missing, f"repo checkout missing required scripts: {_missing}"
print("repo ready @", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(), flush=True)

# %% [markdown]
# ## The capture-source question — the likely 0%-acceptance root cause
#
# **Read this before uploading a corpus.** A draft head is only as good as the
# residual distribution it trains on. dismantle **serves Q4_K_M** at inference,
# so the head must learn from Q4_K_M residuals. There are two ways to produce the
# parquet corpus, and they are NOT interchangeable:
#
# 1. **dismantle-runtime Q4_K_M captures (RECOMMENDED — fixes the suspected bug).**
#    Dump the *exact* residuals the quantized runtime serves on the M3, then pack
#    them into the trainer's parquet schema:
#
#    ```bash
#    # M3 — dump Q4_K_M residuals while greedy-decoding a CODE corpus:
#    DISMANTLE_QWEN_EAGLE5_CAPTURE=1 DISMANTLE_QWEN_EAGLE5_CAPTURE_LAYER=32 \
#    DISMANTLE_QWEN_CAPTURE_CORPUS_PATH=artifacts/eagle5/q3b_residuals.bin \
#      ./target/release/dismantle generate \
#        --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
#        --kernel-profile profiles/qwen3b-instruct-q4k.m3pro18.json \
#        --prompts-file colab/data/calib_trim.txt \
#        --max-new-tokens 64 --temperature 0
#
#    # M3 — pack the raw binary stream into shard_*.parquet (the trainer's schema):
#    python3 tools/orchestrator/pack_corpus.py \
#      --in artifacts/eagle5/q3b_residuals.bin \
#      --out-dir artifacts/eagle5/corpus --rows-per-shard 64
#
#    # M3 — frozen base tensors (token_embd / lm_head / output_norm; tied is fine):
#    python3 tools/training/build_qwen3b_frozen.py \
#      --gguf models/qwen2.5-3b-instruct-q4_k_m.gguf \
#      --out artifacts/eagle5/qwen3b_frozen.npz
#    ```
#
#    Then upload `artifacts/eagle5/corpus/` (the shards) and
#    `artifacts/eagle5/qwen3b_frozen.npz` to `/content/artifacts/eagle5/` here.
#    If you upload only `q3b_residuals.bin`, the next code cell auto-runs
#    `pack_corpus.py` and creates the shards for you.
#
# 2. **HF `mega_calibrate.py` captures (CONVENIENT, but the SUSPECTED 0%-CAUSE).**
#    `colab/mega_calibrate.py --model Qwen/Qwen2.5-3B-Instruct --capture-layer 32
#    --out /content/artifacts/eagle5/corpus` loads a **Hugging Face** model
#    (f16, or bnb-4bit via `--load-4bit`) and captures residuals there. Quick to
#    run entirely in Colab, but it is **not** dismantle's served distribution —
#    this is the prime suspect for the old head's 0% acceptance. Only use it as a
#    fallback / smoke test, and expect the τ gate to be the real arbiter.
#
# The notebook also looks in common Drive locations from the reconciliation run
# (`MyDrive/dismantle/qwen3b_corpus` and
# `MyDrive/dismantle/qwen_reconciliation/qwen3b_frozen.npz`) before failing.
#
# **Both sources emit the same parquet schema** (`tokens`, `residual_q` int8,
# `residual_scale` f32, `residual_shape`, and the `intermediate_*` trio), so the
# training cell below is agnostic — but the *quality* depends entirely on which
# source you picked. The `frozen.npz` always comes from the **GGUF**
# (`build_qwen3b_frozen.py`) regardless of capture source, because the runtime's
# lm_head/embedding is the Q4_K_M one.
#
# See `plans/eagle_spec_handoff_2026_05_30.md` for the full diagnosis.

# %%
# --- Cell 2: artifact prep + training config ---
#
# This cell is deliberately defensive: it can use already-packed shards, auto-pack
# an uploaded raw M3 runtime capture, discover the restored Drive corpus/frozen
# from the reconciliation notebook, or (explicit opt-in) run the HF fallback
# capture. The default still prefers the Q4_K_M runtime corpus because that is
# the suspected fix for the old 0%-acceptance head.
ART = Path(os.environ.get("EAGLE5_ARTIFACT_DIR", "/content/artifacts/eagle5"))
CORPUS_DIR = Path(os.environ.get("EAGLE5_CORPUS_DIR", str(ART / "corpus")))
FROZEN = Path(os.environ.get("EAGLE5_FROZEN", str(ART / "qwen3b_frozen.npz")))
CKPT_DIR = Path(os.environ.get("EAGLE5_CKPT_DIR", str(ART / "ckpt_q4km")))

CAPTURE_LAYER = 32   # Qwen-3B has 36 layers; 32 = near-top (matches the runtime tap)
DEPTH = 4            # τ-eval rollout depth
TAU_GATE = 2.5       # ship gate: accepted-prefix τ must clear this

ART.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


AUTO_MOUNT_DRIVE = _env_bool("EAGLE5_AUTO_MOUNT_DRIVE", True)
AUTO_PACK_RAW_CAPTURE = _env_bool("EAGLE5_AUTO_PACK_RAW_CAPTURE", True)
AUTO_BUILD_FROZEN_FROM_GGUF = _env_bool("EAGLE5_AUTO_BUILD_FROZEN_FROM_GGUF", True)

# Opt-in only: this is convenient for a smoke test but is not the preferred data
# distribution for a production head. Set to True in this cell, or export
# EAGLE5_RUN_HF_FALLBACK_CAPTURE=1 before running the notebook.
RUN_HF_FALLBACK_CAPTURE = _env_bool(
    "EAGLE5_RUN_HF_FALLBACK_CAPTURE",
    bool(globals().get("RUN_HF_FALLBACK_CAPTURE", False)),
)
HF_FALLBACK_MAX_SEQUENCES = int(os.environ.get("EAGLE5_HF_FALLBACK_MAX_SEQUENCES", "2000"))
HF_FALLBACK_BATCH_SIZE = int(
    os.environ.get(
        "EAGLE5_HF_FALLBACK_BATCH_SIZE",
        "4" if VRAM_GB >= 35 else ("2" if VRAM_GB >= 22 else "1"),
    )
)
HF_FALLBACK_MAX_TOKENS = int(
    os.environ.get("EAGLE5_HF_FALLBACK_MAX_TOKENS", "2048" if VRAM_GB >= 22 else "1024")
)
HF_FALLBACK_LM_HEAD_CHUNK_TOKENS = int(
    os.environ.get("EAGLE5_HF_FALLBACK_LM_HEAD_CHUNK_TOKENS", "128" if VRAM_GB >= 35 else "32")
)
HF_FALLBACK_LOAD_4BIT = _env_bool("EAGLE5_HF_FALLBACK_LOAD_4BIT", VRAM_GB < 30)
HF_FALLBACK_DATASET = os.environ.get("EAGLE5_HF_FALLBACK_DATASET", "HuggingFaceH4/ultrachat_200k")
HF_FALLBACK_SPLIT = os.environ.get("EAGLE5_HF_FALLBACK_SPLIT", "train_sft")
_DRIVE_MOUNT_ATTEMPTED = False
_DRIVE_AVAILABLE = None


def _unique_paths(paths):
    out = []
    seen = set()
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _shard_count(path: Path) -> int:
    return len(list(path.glob("shard_*.parquet"))) if path.exists() else 0


def _maybe_mount_drive() -> bool:
    global _DRIVE_MOUNT_ATTEMPTED, _DRIVE_AVAILABLE
    if _DRIVE_AVAILABLE is not None:
        return bool(_DRIVE_AVAILABLE)
    root = Path("/content/drive/MyDrive")
    if root.exists():
        _DRIVE_AVAILABLE = True
        return True
    if not AUTO_MOUNT_DRIVE:
        _DRIVE_AVAILABLE = False
        return False
    if _DRIVE_MOUNT_ATTEMPTED:
        return False
    _DRIVE_MOUNT_ATTEMPTED = True
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print(f"(Drive mount skipped: {e})", flush=True)
    _DRIVE_AVAILABLE = root.exists()
    return bool(_DRIVE_AVAILABLE)


def _discover_file(candidates, *, min_size: int = 1) -> Path | None:
    for p in _unique_paths(candidates):
        if p.exists() and p.is_file() and p.stat().st_size >= min_size:
            return p
    return None


def _path_inventory(paths, *, pattern: str = "shard_*.parquet", max_items: int = 12) -> str:
    lines = []
    for p in _unique_paths(paths)[:max_items]:
        if p.exists():
            if p.is_dir():
                count = len(list(p.glob(pattern)))
                lines.append(f"  - {p}  [dir, {count} x {pattern}]")
            else:
                lines.append(f"  - {p}  [file, {p.stat().st_size / 1e6:.1f} MB]")
        else:
            lines.append(f"  - {p}  [missing]")
    return "\n".join(lines)


def _ensure_python_packages(mod_specs: dict[str, str]) -> None:
    to_install = [
        spec for mod, spec in mod_specs.items()
        if importlib.util.find_spec(mod) is None
    ]
    if to_install:
        run([sys.executable, "-m", "pip", "install", "-q", *to_install])


def _corpus_candidates() -> list[Path]:
    candidates = [
        CORPUS_DIR,
        ART / "corpus",
        Path("/content/artifacts/eagle5/corpus"),
        Path("/content/qwen3b_corpus"),
        # Shipped in-repo (zero upload): the M3-produced Q4_K_M capture corpus,
        # fetched by the `git clone` in cell 1.
        Path("/content/dismantle/colab/data/eagle5_corpus"),
    ]
    if _maybe_mount_drive():
        drive_root = Path("/content/drive/MyDrive")
        candidates.extend([
            drive_root / "dismantle/qwen3b_corpus",
            drive_root / "dismantle/qwen_reconciliation/qwen3b_corpus",
            drive_root / "dismantle/artifacts/eagle5/corpus",
            drive_root / "dismantle/eagle5/corpus",
        ])
    return _unique_paths(candidates)


def _discover_existing_corpus() -> None:
    global CORPUS_DIR
    if _shard_count(CORPUS_DIR) > 0:
        return
    ranked = [(p, _shard_count(p)) for p in _corpus_candidates()]
    ranked = [(p, n) for p, n in ranked if n > 0]
    if not ranked:
        return
    best, count = max(ranked, key=lambda x: x[1])
    if best != CORPUS_DIR:
        print(f"[prep] using discovered corpus: {best} ({count} shards)", flush=True)
        CORPUS_DIR = best


def _raw_capture_candidates() -> list[Path]:
    candidates = [
        Path(os.environ["EAGLE5_RAW_CAPTURE"]) if os.environ.get("EAGLE5_RAW_CAPTURE") else None,
        ART / "q3b_residuals.bin",
        ART / "qwen3b_residuals.bin",
        ART / "residuals.bin",
        Path("/content/q3b_residuals.bin"),
        Path("/content/qwen3b_residuals.bin"),
    ]
    for root in (ART, Path("/content")):
        if root.exists():
            candidates.extend(sorted(root.glob("*residual*.bin")))
    if _maybe_mount_drive():
        drive_root = Path("/content/drive/MyDrive")
        drive_dirs = [
            drive_root / "dismantle/artifacts/eagle5",
            drive_root / "dismantle/eagle5",
        ]
        for root in drive_dirs:
            if root.exists():
                candidates.extend(sorted(root.glob("*residual*.bin")))
    return _unique_paths(candidates)


def _auto_pack_raw_capture() -> None:
    global CORPUS_DIR
    if _shard_count(CORPUS_DIR) > 0 or not AUTO_PACK_RAW_CAPTURE:
        return
    raw = _discover_file(_raw_capture_candidates(), min_size=1024)
    if raw is None:
        return
    if CORPUS_DIR == raw.parent:
        CORPUS_DIR = ART / "corpus"
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[prep] packing raw runtime capture {raw} -> {CORPUS_DIR}", flush=True)
    run_with_heartbeat([
        sys.executable, "-u", "tools/orchestrator/pack_corpus.py",
        "--in", str(raw),
        "--out-dir", str(CORPUS_DIR),
        "--rows-per-shard", "64",
    ], "pack-corpus", interval_sec=30)


def _run_hf_fallback_capture_if_enabled() -> None:
    if _shard_count(CORPUS_DIR) > 0 or not RUN_HF_FALLBACK_CAPTURE:
        return
    print(
        "[prep] RUN_HF_FALLBACK_CAPTURE=True; generating an HF fallback corpus. "
        "This is useful for smoke tests, but Q4_K_M runtime capture is preferred.",
        flush=True,
    )
    deps = {
        "transformers": "transformers>=4.45,<5",
        "datasets": "datasets>=3.0",
        "tqdm": "tqdm>=4.66",
        "accelerate": "accelerate>=1.0",
    }
    if HF_FALLBACK_LOAD_4BIT:
        deps["bitsandbytes"] = "bitsandbytes>=0.43"
    _ensure_python_packages(deps)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "colab/mega_calibrate.py",
        "--model", "Qwen/Qwen2.5-3B-Instruct",
        "--dataset", HF_FALLBACK_DATASET,
        "--split", HF_FALLBACK_SPLIT,
        "--max-sequences", str(HF_FALLBACK_MAX_SEQUENCES),
        "--max-tokens", str(HF_FALLBACK_MAX_TOKENS),
        "--batch-size", str(HF_FALLBACK_BATCH_SIZE),
        "--lm-head-chunk-tokens", str(HF_FALLBACK_LM_HEAD_CHUNK_TOKENS),
        "--capture-layer", str(CAPTURE_LAYER),
        "--out", str(CORPUS_DIR),
    ]
    if HF_FALLBACK_LOAD_4BIT:
        cmd.append("--load-4bit")
    run_with_heartbeat(cmd, "hf-capture", interval_sec=60)


def _validate_corpus(path: Path) -> list[Path]:
    shards = sorted(path.glob("shard_*.parquet")) if path.exists() else []
    if not shards:
        return []
    import pyarrow.parquet as pq
    from eagle5_train_pytorch import _extract_row

    required = {
        "tokens",
        "residual_q", "residual_scale", "residual_shape",
        "intermediate_q", "intermediate_scale", "intermediate_shape",
    }
    first = shards[0]
    try:
        table = pq.read_table(first)
    except Exception as e:
        raise SystemExit(f"Could not read first corpus shard {first}: {e}") from e
    missing = required - set(table.column_names)
    if missing:
        raise SystemExit(f"Corpus shard {first} missing required columns: {sorted(missing)}")
    if table.num_rows <= 0:
        raise SystemExit(f"Corpus shard {first} has zero rows")

    usable = None
    for i in range(min(table.num_rows, 8)):
        row = {c: table[c][i].as_py() for c in table.column_names}
        usable = _extract_row(row, max_row_tokens=128)
        if usable is not None:
            break
    if usable is None:
        raise SystemExit(
            f"Corpus shard {first} has no usable rows; each row needs >=5 tokens "
            "and residual/intermediate tensors matching token length."
        )
    hidden = int(usable["residual"].shape[-1])
    if hidden != 2048:
        raise SystemExit(f"Corpus hidden_dim={hidden}; expected 2048 for Qwen2.5-3B.")
    manifest = path / "manifest.json"
    if manifest.exists():
        try:
            meta = json.loads(manifest.read_text())
            if meta.get("model"):
                print(
                    f"[prep] corpus manifest model={meta.get('model')} "
                    f"capture_layer={meta.get('capture_layer')}; this looks like "
                    "HF mega_calibrate output, not M3 Q4_K_M runtime capture.",
                    flush=True,
                )
        except Exception as e:
            print(f"[prep] WARN: could not parse corpus manifest {manifest}: {e}", flush=True)
    print(
        f"[prep] corpus OK: {len(shards)} shard(s), first={first.name}, "
        f"sample_tokens={len(usable['prev_tokens']) + 1}, hidden={hidden}",
        flush=True,
    )
    return shards


def _frozen_candidates() -> list[Path]:
    candidates = [
        FROZEN,
        ART / "qwen3b_frozen.npz",
        Path("/content/qwen3b_frozen.npz"),
        Path("/content/q3b_frozen.npz"),
    ]
    if _maybe_mount_drive():
        drive_root = Path("/content/drive/MyDrive")
        candidates.extend([
            drive_root / "dismantle/qwen_reconciliation/qwen3b_frozen.npz",
            drive_root / "dismantle/artifacts/eagle5/qwen3b_frozen.npz",
            drive_root / "dismantle/eagle5/qwen3b_frozen.npz",
        ])
    return _unique_paths(candidates)


def _stage_discovered_frozen() -> None:
    global FROZEN
    if FROZEN.exists():
        return
    found = _discover_file(_frozen_candidates(), min_size=100_000_000)
    if found is None:
        return
    local = ART / "qwen3b_frozen.npz"
    if found != local:
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists() or local.stat().st_size != found.stat().st_size:
                print(f"[prep] staging frozen npz {found} -> {local}", flush=True)
                shutil.copy2(found, local)
            FROZEN = local
        except OSError as e:
            print(f"[prep] could not stage frozen locally ({e}); using {found}", flush=True)
            FROZEN = found
    else:
        FROZEN = found


def _gguf_candidates() -> list[Path]:
    roots = [ART, Path("/content"), Path("models")]
    if _maybe_mount_drive():
        drive_root = Path("/content/drive/MyDrive")
        roots.extend([
            drive_root / "dismantle/models",
            drive_root / "dismantle/artifacts/eagle5",
            drive_root / "dismantle/eagle5",
        ])
    names = [
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    ]
    candidates = []
    for root in roots:
        for name in names:
            candidates.append(root / name)
        if root.exists():
            candidates.extend(sorted(root.glob("*3B*Q4_K_M*.gguf")))
            candidates.extend(sorted(root.glob("*3b*q4_k_m*.gguf")))
    return _unique_paths(candidates)


def _download_q4km_gguf() -> "Path | None":
    """Fetch the Qwen2.5-3B Q4_K_M GGUF from HF so the frozen base can be built
    with ZERO upload. The frozen tensors (token_embd/lm_head/output_norm) are the
    same in any Q4_K_M GGUF of this model — only the capture corpus is
    distribution-sensitive, and that ships in-repo."""
    try:
        _ensure_python_packages({"huggingface_hub": "huggingface_hub>=0.34"})
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as e:
        print(f"[prep] cannot import huggingface_hub to fetch GGUF: {e}", flush=True)
        return None
    for repo in ("Qwen/Qwen2.5-3B-Instruct-GGUF", "bartowski/Qwen2.5-3B-Instruct-GGUF"):
        try:
            files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
        except Exception as e:
            print(f"[prep] skip GGUF repo {repo}: {e}", flush=True)
            continue
        hits = sorted((f for f in files if "q4_k_m" in f.lower() and "-of-" not in f), key=len)
        if hits:
            print(f"[prep] downloading {repo}/{hits[0]} for the frozen build...", flush=True)
            try:
                return Path(hf_hub_download(repo, hits[0]))
            except Exception as e:
                print(f"[prep] download from {repo} failed: {e}", flush=True)
    print("[prep] no single-file Q4_K_M GGUF found on HF; upload one or the frozen npz.", flush=True)
    return None


def _build_frozen_from_gguf_if_possible() -> None:
    global FROZEN
    if FROZEN.exists() or not AUTO_BUILD_FROZEN_FROM_GGUF:
        return
    gguf = _discover_file(_gguf_candidates(), min_size=1_000_000_000)
    if gguf is None:
        gguf = _download_q4km_gguf()   # zero-upload path: pull the GGUF from HF
    if gguf is None:
        return
    _ensure_python_packages({"gguf": "gguf>=0.10"})
    FROZEN = ART / "qwen3b_frozen.npz"
    print(f"[prep] building frozen npz from GGUF {gguf} -> {FROZEN}", flush=True)
    run_with_heartbeat([
        sys.executable, "-u", "tools/training/build_qwen3b_frozen.py",
        "--gguf", str(gguf),
        "--out", str(FROZEN),
    ], "build-frozen", interval_sec=30)


def _validate_frozen(path: Path) -> None:
    if not path.exists():
        return
    if path.stat().st_size < 100_000_000:
        raise SystemExit(f"Frozen base looks too small: {path} ({path.stat().st_size} bytes)")
    with np.load(path, allow_pickle=False) as z:
        required = {"token_embd", "lm_head", "output_norm"}
        missing = required - set(z.files)
        if missing:
            raise SystemExit(f"Frozen base {path} missing keys: {sorted(missing)}")
        token_shape = tuple(z["token_embd"].shape)
        lm_shape = tuple(z["lm_head"].shape)
        norm_shape = tuple(z["output_norm"].shape)
    if token_shape != (2048, 151936):
        raise SystemExit(f"token_embd shape {token_shape}; expected (2048, 151936)")
    if lm_shape != (2048, 151936):
        raise SystemExit(f"lm_head shape {lm_shape}; expected (2048, 151936)")
    if norm_shape != (2048,):
        raise SystemExit(f"output_norm shape {norm_shape}; expected (2048,)")
    print(f"[prep] frozen OK: {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)


def _fail_missing_corpus():
    corpus_inventory = _path_inventory(_corpus_candidates(), pattern="shard_*.parquet")
    raw_inventory = _path_inventory(_raw_capture_candidates(), pattern="*.bin")
    raise SystemExit(
        "\n".join([
            f"Corpus dir not found / empty: {CORPUS_DIR}",
            "Provide a corpus with ONE of:",
            f"  (best) upload M3 Q4_K_M shard_*.parquet under {CORPUS_DIR}",
            f"  (best) upload q3b_residuals.bin under {ART}; this cell will auto-pack it",
            "  (fallback)    colab/mega_calibrate.py --model Qwen/Qwen2.5-3B-Instruct "
            f"--capture-layer {CAPTURE_LAYER} --out {CORPUS_DIR}",
            "  (fallback)    set RUN_HF_FALLBACK_CAPTURE=True in this cell and rerun, "
            "or set os.environ['EAGLE5_RUN_HF_FALLBACK_CAPTURE']='1' before a hotfix-loader exec",
            "",
            "Corpus path inventory:",
            corpus_inventory or "  (none)",
            "",
            "Raw-capture file inventory:",
            raw_inventory or "  (none)",
            "See the capture-source markdown cell above.",
        ])
    )


_discover_existing_corpus()
_auto_pack_raw_capture()
_run_hf_fallback_capture_if_enabled()
shards = _validate_corpus(CORPUS_DIR)
if not shards:
    _fail_missing_corpus()

_stage_discovered_frozen()
_build_frozen_from_gguf_if_possible()
if not FROZEN.exists():
    raise SystemExit(
        f"Frozen base not found: {FROZEN}\n"
        "Provide it with ONE of:\n"
        f"  upload qwen3b_frozen.npz to {ART}/\n"
        "  upload qwen2.5-3b-instruct-q4_k_m.gguf to /content/artifacts/eagle5/ "
        "so this cell can build the npz\n"
        "  or produce on the M3 with:\n"
        "  python3 tools/training/build_qwen3b_frozen.py "
        "--gguf models/qwen2.5-3b-instruct-q4_k_m.gguf "
        f"--out {FROZEN}\n"
        "Also checked the common Drive frozen paths from the reconciliation notebook."
    )
_validate_frozen(FROZEN)

# Provenance note: the runtime capture stream + pack_corpus.py do NOT stamp a
# source_precision marker, so we cannot auto-verify Q4_K_M-ness from the files.
# The τ gate below is the real arbiter — an HF-capture head will fail it.
print(f"corpus: {len(shards)} shards under {CORPUS_DIR}", flush=True)
print(f"frozen: {FROZEN} ({FROZEN.stat().st_size / 1e6:.0f} MB)", flush=True)

# Qwen-3B preset. Chained-hidden rollout is the key runtime-speedup driver;
# multi-depth targets (1..4) protect shallow-depth accuracy. batch_size scales
# with VRAM. These flags mirror eagle5_train_pytorch.py's argparse exactly.
TRAIN = {
    "epochs": 12,
    "batch_size": 24 if VRAM_GB >= 35 else (16 if VRAM_GB >= 22 else 8),
    "seq_len": 16,
    "lr": 1e-3,
    "max_rows": 8000 if VRAM_GB >= 22 else 4000,
    "max_row_tokens": 128,
    "num_blocks": 2,  # 2 = block + extra_blocks.0 — the only structure the M3 runtime
                      # forward is parity-verified against; num_blocks=1 trained a head
                      # that scored 80% depth-1 in PyTorch but 0% accept on-device.
    "head_heads": 16,
    "head_ff_mult": 4.0,
    "calib_loss_weight": 0.1,
    "residual_delta_loss_weight": 0.0,
    "rollout_loss_weight": 1.0,
    "rollout_depth": 5,
    "rollout_depth_targets": "1,2,3,4",
    "rollout_draft_prob": 0.75,
}
CKPT_DIR.mkdir(parents=True, exist_ok=True)
print("train config:", json.dumps(TRAIN, indent=2), flush=True)

# %%
# --- Cell 3: TRAIN (writes latest.npz AND head_final.safetensors) ---
kill_stale_trainers()
torch.cuda.empty_cache()
HEAD = CKPT_DIR / "head_final.safetensors"
LATEST = CKPT_DIR / "latest.npz"

train_cmd = [
    sys.executable, "-u", "colab/eagle5_train_pytorch.py",
    "--corpus-dir", str(CORPUS_DIR),
    "--frozen", str(FROZEN),
    "--ckpt-dir", str(CKPT_DIR),
    "--device", "cuda",
    "--target-mode", "corpus",          # train against the model's REAL next token
    "--capture-layer", str(CAPTURE_LAYER),
    "--epochs", str(TRAIN["epochs"]),
    "--batch-size", str(TRAIN["batch_size"]),
    "--seq-len", str(TRAIN["seq_len"]),
    "--lr", str(TRAIN["lr"]),
    "--max-rows", str(TRAIN["max_rows"]),
    "--max-row-tokens", str(TRAIN["max_row_tokens"]),
    "--num-blocks", str(TRAIN["num_blocks"]),
    "--head-heads", str(TRAIN["head_heads"]),
    "--head-ff-mult", str(TRAIN["head_ff_mult"]),
    "--calib-loss-weight", str(TRAIN["calib_loss_weight"]),
    "--residual-delta-loss-weight", str(TRAIN["residual_delta_loss_weight"]),
    "--rollout-loss-weight", str(TRAIN["rollout_loss_weight"]),
    "--rollout-depth", str(TRAIN["rollout_depth"]),
    "--rollout-depth-targets", TRAIN["rollout_depth_targets"],
    "--rollout-draft-prob", str(TRAIN["rollout_draft_prob"]),
    "--rollout-chain-hidden",
    "--seed", str(SEED),
    "--save-safetensors",
]
run_with_heartbeat(train_cmd, "train", interval_sec=30)

assert HEAD.exists(), "training did not emit head_final.safetensors"
assert LATEST.exists(), "training did not emit latest.npz"
assert HEAD.stat().st_size > 1_000_000, f"head_final.safetensors suspiciously small: {HEAD.stat().st_size} bytes"
print(f"trained head -> {HEAD} ({HEAD.stat().st_size / 1e6:.0f} MB)", flush=True)

# %%
# --- Cell 4: τ-EVAL (depth 4) + GO/NO-GO gate ---
TAU_OUT = ART / "tau.json"
eval_cmd = [
    sys.executable, "-u", "colab/eagle5_tau_eval_pytorch.py",
    "--ckpt", str(HEAD),
    "--frozen", str(FROZEN),
    "--corpus", str(CORPUS_DIR),
    "--out", str(TAU_OUT),
    "--depth", str(DEPTH),
    "--target-mode", "corpus",   # accept vs the model's REAL next token (ground truth)
    "--chain-hidden",            # runtime-predictive: feed draft_hidden forward
    "--max-windows", "4000",
    "--seed", str(SEED + 123),
]
run_with_heartbeat(eval_cmd, "tau", interval_sec=30)

tau_doc = json.loads(TAU_OUT.read_text())
tau = None
for key in ("tau", "mean_accepted_len", f"tau_depth_{DEPTH}"):
    if tau_doc.get(key) is not None:
        tau = float(tau_doc[key])
        break
assert tau is not None, f"tau eval did not report a tau field: {list(tau_doc.keys())}"
verdict = "GO" if tau >= TAU_GATE else "NO-GO"
print(json.dumps(tau_doc, indent=2)[:1400], flush=True)
print(f"\nτ@depth={DEPTH} = {tau:.3f}   gate >= {TAU_GATE}   ->   {verdict}", flush=True)
if verdict == "NO-GO":
    print(
        "\nNO-GO triage: (1) confirm the corpus is dismantle-runtime Q4_K_M "
        "captures, NOT HF mega_calibrate — see the capture-source cell; "
        "(2) the prior broken head scored 0.000 accept, the n-gram oracle 1.43.",
        flush=True,
    )

# %%
# --- Cell 5: write result JSON + download the head ---
res = {
    "head": str(HEAD),
    "latest_npz": str(LATEST),
    "tau": tau,
    "depth": DEPTH,
    "tau_gate": TAU_GATE,
    "verdict": verdict,
    "capture_layer": CAPTURE_LAYER,
    "gpu": GPU_NAME,
    "seed": SEED,
    "train": TRAIN,
    "tau_doc": tau_doc,
    "prior_head_accept": 0.0,
    "ngram_oracle_tau": 1.43,
    "next": "M3: eagle5_forward_parity with the new head, then tools/bench/eagle5_paired_bench.sh",
}
RESULT_OUT = ART / "eagle3_train_result.json"
RESULT_OUT.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2), flush=True)

try:
    from google.colab import files
    files.download(str(HEAD))
    files.download(str(RESULT_OUT))
except Exception as e:
    print(f"(auto-download skipped: {e}); fetch {HEAD} + {RESULT_OUT} from the file browser.", flush=True)

# %% [markdown]
# ## M3 verification (run after downloading `head_final.safetensors`)
#
# A τ GO here is **necessary but not sufficient**. Confirm head↔runtime parity
# and a real paired speedup on the M3 before trusting the head:
#
# ```bash
# # 1. Re-dump the parity fixture from the NEW head, then check Rust forward parity:
# python3 tools/eagle5_forward_dump.py \
#   --head <downloaded head_final.safetensors> \
#   --out crates/dismantle-core/tests/fixtures/eagle5_parity_q3b.json --seed 0xea91e5
# cargo test -p dismantle-core --test eagle5_forward_parity -- --nocapture
#
# # 2. Paired runtime bench (baseline vs --speculate eagle5). Must beat the
# #    no-spec dec_tps AND the prior 0.000 acceptance:
# WEIGHTS=models/qwen2.5-3b-instruct-q4_k_m.gguf \
# PROFILE=profiles/qwen3b-instruct-q4k.m3pro18.json \
# EAGLE5_HEAD=<downloaded head_final.safetensors> \
# PROMPT='def quicksort(arr):' \
#   bash tools/bench/eagle5_paired_bench.sh
# ```
#
# Parity FAIL => fix the Rust/Metal forward (`crates/dismantle-core/src/speculate/
# eagle5_forward.rs`), not the Colab head. Only report a speedup after
# `tools/bench/analyze_tcb_trace.py` passes its dispatch-cost gate.
