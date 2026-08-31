#!/usr/bin/env python3
"""Detached ModelLake acquisition watcher.

This watcher is deliberately conservative about state changes:

* it attaches to already-running exact-revision downloads;
* it never deletes, clears, promotes, or creates a second destination for a job;
* it admits the smallest queued specimens alongside the remaining P0 partial;
* every new admission is checked against the physical drive free-space floor;
* it records network, disk, process, and Hugging Face-auth health as JSONL.

The launchd plist is separate so this process survives the terminal/Codex task.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ODYSSEY = REPO_ROOT / "workspace" / "campaign" / "odyssey"
DOWNLOAD_DIR = ODYSSEY / "downloads"
# Keep watcher control metadata on the internal workspace. Writing it on the
# busy external volume can block health sampling behind ModelLake I/O.
MANIFEST_DIR = ODYSSEY / "watch-manifests"
DRIVE = Path("/Volumes/corpdrive")
MODEL_ROOT = Path("/Volumes/corpdrive/hawking-modellake")
HF_BIN = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/hf")
PYTHON_BIN = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")
LOG = DOWNLOAD_DIR / "modellake-watch.jsonl"
LOCK_PATH = ODYSSEY / ".modellake-watch.lock"

FLOOR_BYTES = 100_000_000_000
WARN_BYTES = 250_000_000_000
MAX_DOWNLOAD_JOBS = 4
# The hub CLI defaults to eight file workers.  The two P0 repositories are
# deliberately large, sharded artefacts and this host has a 10 GbE uplink, so
# use a high but bounded fan-out per pinned transfer.  This is persisted in
# the launch watcher, not dependent on an interactive terminal.
MAX_WORKERS = 16
POLL_SECONDS = 0.10
NETWORK_SAMPLE_EMIT_SECONDS = 1.0
STATE_SAMPLE_EMIT_SECONDS = 10.0
STALL_SECONDS = 15 * 60
AUTH_CHECK_SECONDS = 10 * 60
KNOWN_TEMP_BYTES = 20_000_000_000
# A transient interface dip is not evidence that a pinned downloader is bad.
# Transfer rate is telemetry, not sufficient evidence to terminate a live
# exact-revision session.  The hub/Xet client already owns connection retry;
# the watcher recovers an actually exited worker into the same destination.
# A rate dip arms recovery telemetry, but never terminates a session by itself.
# Refresh still requires no durable byte growth, so a slow-but-healthy shard is
# left alone.  The short arm window improves recovery from genuine stalls.
RATE_BASED_REFRESH_ENABLED = True
RATE_ARM_SECONDS = 3
RATE_STALL_REFRESH_SECONDS = 10
# A session with no growth in the actual Hub partial cache for the full
# recovery interval is eligible for one same-destination refresh.  This is
# deliberately separate from aggregate interface-rate telemetry.
STALE_REFRESH_ENABLED = True
RECOVERY_REFRESH_SECONDS = 60
RECOVERY_COOLDOWN_SECONDS = 3 * 60 + 30
LOW_RX_BYTES_PER_SEC = 150_000_000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(value: str) -> str:
    value = re.sub(r"hf_[A-Za-z0-9_-]+", "hf_[REDACTED]", value)
    value = re.sub(r"(?i)(token|authorization|bearer)[=: ]+\S+", r"\1=[REDACTED]", value)
    return value[-1000:]


def emit(event: str, **fields: object) -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    row = {"ts": now(), "event": event, **fields}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def notify(message: str, kind: str = "warning") -> None:
    """Best-effort local notification; the JSONL log is authoritative."""
    emit("notification", kind=kind, message=message)
    title = "Hawking ModelLake"
    safe_message = message.replace("\\", "\\\\").replace('"', '\\"')[:220]
    safe_title = title.replace('"', '\\"')
    script = f'display notification "{safe_message}" with title "{safe_title}"'
    try:
        subprocess.run(["/usr/bin/osascript", "-e", script], timeout=10,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def free_bytes() -> int:
    st = os.statvfs(DRIVE)
    return int(st.f_bavail * st.f_frsize)


def interface_name() -> str | None:
    try:
        out = subprocess.check_output(["/sbin/route", "-n", "get", "default"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return None
    for line in out.splitlines():
        if line.strip().startswith("interface:"):
            return line.split(":", 1)[1].strip()
    return None


def interface_counters(interface: str | None) -> tuple[int, int] | None:
    if not interface:
        return None
    try:
        out = subprocess.check_output(["/usr/sbin/netstat", "-ib"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return None
    for line in out.splitlines():
        fields = line.split()
        if len(fields) >= 10 and fields[0] == interface and fields[2].startswith("<Link"):
            try:
                return int(fields[6]), int(fields[9])
            except ValueError:
                return None
    return None


def process_rows() -> list[tuple[int, str]]:
    try:
        out = subprocess.check_output(["/bin/ps", "-axo", "pid=,command="],
                                      text=True, stderr=subprocess.DEVNULL, timeout=15)
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2:
            try:
                rows.append((int(fields[0]), fields[1]))
            except ValueError:
                pass
    return rows


def matching_pids(job: dict[str, object], rows: list[tuple[int, str]]) -> list[int]:
    repo = str(job["repo"])
    revision = str(job["revision"])
    destination = str(job["destination"])
    needle = f"hf download {repo}"
    return [pid for pid, command in rows
            if needle in command and revision in command and destination in command]


def slug(repo: str, revision: str) -> str:
    return repo.replace("/", "--") + "@" + revision[:12]


def local_destination(repo: str, revision: str) -> Path:
    return MODEL_ROOT / "partial" / slug(repo, revision)


def job(repo: str, revision: str, mode: str, priority: str) -> dict[str, object]:
    return {
        "repo": repo,
        "revision": revision,
        "mode": mode,
        "priority": priority,
        "destination": str(local_destination(repo, revision)),
        "tag": slug(repo, revision),
    }


# P0 exact totals were established from the pinned upstream manifests before
# the current downloads were launched. These are full selected-repository bytes.
# GLM-4.5's existing partial remains on disk as a legacy trial, but is no
# longer resumed. The unfinished slot is intentionally not replaced by
# another giant: the small-to-large queue is admitted alongside Kimi-K3.
P0 = [
    {**job("zai-org/GLM-4.5-Air", "a24ceef6ce4f3536971efe9b778bdaa1bab18daa", "all", "P0"),
     "expected": 220_961_581_797},
    {**job("moonshotai/Kimi-K3", "9f62e4e9fffbd0a83ddd60e1c209d828994b3569", "all", "P0"),
     "expected": 1_560_998_984_390},
]

# Revisions are pinned to the live manifest resolution performed for this
# campaign. The queue is ordered by resolved selected bytes, smallest first;
# the existing four-job cap and storage floor remain authoritative. "safe"
# keeps one canonical safetensors representation and execution metadata,
# avoiding bin/ONNX/checkpoint copies. The intentional BitNet pair uses
# separate repositories.
QUEUE = [
    job("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610", "safe", "P2-A"),
    job("microsoft/bitnet-b1.58-2B-4T", "04c3b9ad9361b824064a1f25ea60a8be9599b127", "safe", "P1-S"),
    job("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "safe", "P1-A"),
    job("depth-anything/Depth-Anything-V2-Large-hf", "7581137eff8d4e94f6e796d3baea0e9fa79b22d2", "safe", "P2-A"),
    job("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "safe", "P1-A"),
    job("answerdotai/ModernBERT-large", "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13", "safe", "P1-A"),
    job("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9", "safe", "P1-A"),
    job("google/timesfm-2.0-500m-pytorch", "dc2443792ce5516872b89b37cf1bc058c3bf0c10", "safe", "P1-A"),
    job("state-spaces/mamba3-siso-1.5b", "5cfc721542ec9ccee768088b2fd6b7e8101219d8", "all", "P1-S"),
    job("state-spaces/mamba3-mimo-1.5b", "bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec", "all", "P1-S"),
    job("google/flan-t5-large", "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a", "safe", "P1-A"),
    job("facebook/vjepa2-vitg-fpc64-256", "875c192b7b704b87d1e1d99345769632dd5f739a", "safe", "P1-S"),
    job("boltz-community/boltz-2", "6fdef46d763fee7fbb83ca5501ccceff43b85607", "boltz", "P1-A"),
    job("microsoft/bitnet-b1.58-2B-4T-bf16", "276681394656abdadb8e80e5b2c3db5e5d7fcaff", "safe", "P1-S"),
    job("LiquidAI/LFM2.5-2.6B-Base", "c57bdaed1ef166fe3095dda07f4a5e789ad5321e", "safe", "P1-A"),
    job("ai21labs/AI21-Jamba2-3B", "525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9", "safe", "P1-A"),
    job("stabilityai/stable-audio-open-1.0", "f21265c1e2710b3bd2386596943f0007f55f802e", "stable_audio", "P1-A"),
    job("arcinstitute/evo2_7b", "bda0089f92582d5baabf0f22d9fc85f3588f6b58", "all", "P1-S"),
    job("facebook/musicgen-large", "15ccdc92099879e47b6da12c350cdb71d4eab3ca", "musicgen", "P2-A"),
    job("lerobot/pi0_base", "25c379b52ba2ff8788cab921758a3cc3fe3f77f2", "safe", "P2-HIGH"),
    job("tencent/HunyuanVideo", "6204ad6aea1a77ff5aba337c88278bb9500eb37d", "hunyuan", "P2-HIGH"),
    job("bigcode/starcoder2-7b", "bb9afde76d7945da5745592525db122d4d729eb1", "safe", "P1-A"),
    job("allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f", "safe", "P1-A"),
    job("Dream-org/Dream-v0-Instruct-7B", "05334cb9faaf763692dcf9d8737c642be2b2a6ae", "safe", "P1-S"),
    job("kyutai/moshika-pytorch-bf16", "a49141e28b3d9c947cf9aa5314431e1b11cbd2f5", "safe", "P1-A"),
    job("GSAI-ML/iLLaDA-8B-Base", "a1b5b5f8a31a3854a46205ee584178c04b45ec9a", "safe", "P1-S"),
    # Metadata is public, but file access still requires explicit upstream
    # approval; keep this fail-closed until a file probe succeeds.
    {**job("nvidia/personaplex-7b-v1", "fdaf4090a61cb315c138a1faee287ffd6c716309", "all", "P2-GATED"),
     "requires_manual_auth": True},
    job("nvidia/audio-flamingo-3", "ee26c58423988d7d7cda7b85dd3ce5d97ee8753d", "all", "P2-HIGH"),
    job("facebook/blt-7b", "b65201dce04b0a824f0dedeb13bb16fc3a918048", "blt", "P1-S"),
    job("RWKV/RWKV7-13.3B-20260805", "64ffe5934178f40fb2c6de13f12cffaf9058f243", "safe", "P1-S"),
    job("microsoft/Phi-4-reasoning-plus", "69baf8528e1bcf05f475034d9e5dd32875ed125f", "safe", "P1-A"),
    job("LiquidAI/LFM2-24B-A2B", "a3bbacd91a678b97712f0e323e52f8c24ba29542", "safe", "P1-A"),
    job("arcinstitute/evo2_40b", "d529aa57c30771814217ad89baaeaf6e2315c7d7", "all", "P2-CONDITIONAL"),
    job("Wan-AI/Wan2.2-T2V-A14B", "c8c270b13ee05bfa474194ac9fb07a5868a97cea", "all", "P2-HIGH"),
    job("zai-org/GLM-5.3-Flash", "04c4e9e95c5da8862dced7e5056455116f83a7e0", "all", "P2-HIGH"),
    job("thinkingmachines/Inkling-Small", "8cc5877b44d343f88b92086aa1fb72897950f06a", "safe", "P1-S"),
    # Fresh diversity wave, appended after the existing queue so current
    # admissions and pinned destinations retain their order.
    job("ibm-granite/granite-4.0-h-micro", "d5f01a3ea75f088947be3aae039f4ad52837dfde", "safe", "P1-DIVERSITY"),
    job("ibm-granite/granite-4.0-h-tiny", "791e0d3d28c86e106c9b6e0b4cecdee0375b6124", "safe", "P1-DIVERSITY"),
    job("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "482adb537c021c86670beed01cd58990d01e72e4", "safe", "P1-DIVERSITY"),
    job("microsoft/Phi-4-multimodal-instruct", "93f923e1a7727d1c4f446756212d9d3e8fcc5d81", "safe", "P1-DIVERSITY"),
    job("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "safe", "P1-DIVERSITY"),
    job("Qwen/Qwen3-Coder-30B-A3B-Instruct", "b2cff646eb4bb1d68355c01b18ae02e7cf42d120", "safe", "P1-DIVERSITY"),
    {**job("google/gemma-3-4b-it", "093f9f388b31de276ce2de164bdc2081324b9767", "safe", "P1-DIVERSITY"),
     "requires_manual_auth": True},
    {**job("meta-llama/Llama-4-Scout-17B-16E-Instruct", "92f3b1597a195b523d8d9e5700e57e4fbb8f20d3", "safe", "P1-DIVERSITY"),
     "requires_manual_auth": True},
    job("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", "safe", "P1-DIVERSITY"),
    # Dense multimodal 20-40B control: distinct from the existing LFM2
    # 24B-A2B MoE and pinned to the verified Apache-2.0 upstream revision.
    job("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "68faf511d618ef198fef186659617cfd2eb8e33a", "safe", "P1-DIVERSITY"),
]


def selected(name: str, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "blt":
        return not name.startswith(".eval_results/")
    if mode == "boltz":
        return name in {".gitattributes", "README.md", "boltz2_aff.ckpt", "boltz2_conf.ckpt"}
    if mode == "stable_audio":
        if name.endswith((".ckpt", ".csv", ".png")):
            return False
        return True
    if mode == "hunyuan":
        return name in {
            "README.md",
            "config.json",
            "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt",
            "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8_map.pt",
            "hunyuan-video-t2v-720p/vae/config.json",
            "hunyuan-video-t2v-720p/vae/pytorch_model.pt",
        }
    if mode == "musicgen":
        return name in {
            "compression_state_dict.bin",
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
    if name.startswith(".eval_results/"):
        return False
    if name.startswith("onnx/") or name.startswith("original/"):
        return False
    if mode == "safe":
        if name.endswith((".bin", ".ckpt", ".pt", ".onnx", ".msgpack", ".h5")):
            return False
        return True
    return True


def manifest_for(item: dict[str, object]) -> tuple[list[str], int, dict[str, int]]:
    """Resolve and pin the selected file list at the already-pinned revision."""
    repo = str(item["repo"])
    revision = str(item["revision"])
    mode = str(item["mode"])
    script = (
        "from huggingface_hub import HfApi; import json; "
        "i=HfApi().model_info(%r, revision=%r, files_metadata=True); "
        "print(json.dumps({'sha': i.sha, 'files': "
        "[{'name': getattr(f, 'rfilename', ''), 'size': getattr(f, 'size', None)} "
        "for f in (i.siblings or [])]}))" % (repo, revision)
    )
    result = subprocess.run([str(PYTHON_BIN), "-c", script], capture_output=True,
                            text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(redact(result.stderr or result.stdout or "manifest lookup failed"))
    data = json.loads(result.stdout)
    if data.get("sha") != revision:
        raise RuntimeError(f"revision mismatch: requested {revision}, got {data.get('sha')}")
    files = []
    sizes: dict[str, int] = {}
    expected = 0
    for entry in data.get("files", []):
        name = str(entry.get("name", ""))
        size = entry.get("size")
        if selected(name, mode):
            if not isinstance(size, int):
                raise RuntimeError(f"unknown remote size for {repo}:{name}")
            files.append(name)
            sizes[name] = size
            expected += size
    if not files or expected <= 0:
        raise RuntimeError(f"empty selected manifest for {repo}")
    cache = MANIFEST_DIR / f"{item['tag']}.json"
    try:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"repo": repo, "revision": revision,
                                     "mode": mode, "expected": expected,
                                     "files": files, "sizes": sizes,
                                     "resolved_sha": data["sha"]},
                                    indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        emit("manifest_cache_error", job=item["tag"], error=redact(str(exc)))
    return files, expected, sizes


def load_cached_manifest(item: dict[str, object]) -> tuple[list[str], int, dict[str, int]] | None:
    """Load a previously captured exact-revision inventory without network I/O."""
    cache = MANIFEST_DIR / f"{item['tag']}.json"
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if (data.get("repo") != item["repo"]
                or data.get("revision") != item["revision"]
                or data.get("resolved_sha") != item["revision"]):
            return None
        files = [str(name) for name in data["files"]]
        sizes = {str(name): int(size) for name, size in data["sizes"].items()}
        expected = sum(sizes[name] for name in files)
        if not files or expected <= 0 or int(data.get("expected", -1)) != expected:
            return None
        return files, expected, sizes
    except (OSError, ValueError, KeyError, TypeError):
        return None


def complete(item: dict[str, object], files: list[str], sizes: dict[str, int]) -> bool:
    root = Path(str(item["destination"]))
    if not root.is_dir():
        return False
    for name in files:
        path = root / name
        try:
            if not path.is_file() or path.stat().st_size != sizes[name]:
                return False
        except (FileNotFoundError, OSError):
            return False
    return True


def durable_bytes(item: dict[str, object], files: list[str], sizes: dict[str, int]) -> int:
    """Return logical bytes visible in one pinned destination.

    This is intentionally a metadata-only progress probe: it never opens model
    contents and it does not treat a partial file as complete.  It stats only
    manifest paths plus Hugging Face's hashed in-flight cache files, avoiding a
    recursive walk through giant sharded destinations on every watcher pass.
    """
    root = Path(str(item["destination"]))
    total = 0
    if not root.is_dir():
        return 0
    for name in files:
        expected = sizes[name]
        path = root / name
        best = 0
        for candidate in (path, Path(str(path) + ".incomplete"),
                          Path(str(path) + ".part"), Path(str(path) + ".tmp")):
            try:
                if candidate.is_file() and not candidate.is_symlink():
                    best = max(best, min(candidate.stat().st_size, expected))
            except OSError:
                continue
        total += best
    # huggingface_hub stores active downloads under hashed names in this cache,
    # so they cannot be matched to manifest paths by filename.  Include their
    # physical partial bytes for progress detection, capped by the manifest's
    # remaining logical bytes to avoid over-counting during a transition.
    cache_dir = root / ".cache" / "huggingface" / "download"
    partial_total = 0
    try:
        for candidate in cache_dir.iterdir():
            if candidate.is_file() and candidate.name.endswith(".incomplete"):
                partial_total += min(candidate.stat().st_size, sum(sizes.values()))
    except OSError:
        pass
    total += min(partial_total, max(0, sum(sizes.values()) - total))
    return total


def launch(item: dict[str, object], files: list[str], log_path: Path) -> subprocess.Popen[str]:
    destination = Path(str(item["destination"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(HF_BIN), "download", str(item["repo"]), *files,
               "--revision", str(item["revision"]), "--local-dir", str(destination),
               "--max-workers", str(MAX_WORKERS), "--format", "json"]
    env = os.environ.copy()
    env.update({
        "HF_XET_HIGH_PERFORMANCE": "1",
        # Retain slow-but-live data-plane connections instead of treating an
        # upstream/CDN pause as a failed transfer.  The watcher still retries
        # a genuine process exit into the exact same destination.
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "HF_HUB_ETAG_TIMEOUT": "30",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "PYTHONUNBUFFERED": "1",
    })
    handle = log_path.open("a", encoding="utf-8")
    handle.write(json.dumps({"ts": now(), "event": "launch", "command": command},
                            sort_keys=True) + "\n")
    handle.flush()
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                               stderr=subprocess.STDOUT, env=env,
                               start_new_session=True, text=True)
    # The child owns the inherited descriptor; closing our copy does not affect it.
    handle.close()
    emit("download_started", job=item["tag"], repo=item["repo"],
         revision=item["revision"], pid=process.pid, max_workers=MAX_WORKERS,
         expected=item.get("expected"), destination=item["destination"])
    return process


def auth_check() -> tuple[str, str]:
    script = "from huggingface_hub import HfApi; HfApi().whoami(); print('ok')"
    try:
        result = subprocess.run([str(PYTHON_BIN), "-c", script], capture_output=True,
                                text=True, timeout=45)
    except subprocess.TimeoutExpired:
        return "network_timeout", "Hugging Face auth probe timed out"
    output = redact((result.stderr or result.stdout or "").strip())
    if result.returncode == 0:
        return "ok", "credential accepted"
    if re.search(r"401|403|unauthorized|invalid token|authentication", output, re.I):
        return "auth_failed", output
    return "network_error", output


def acquire_lock():
    ODYSSEY.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("ModelLake watcher already running")
    handle.write(str(os.getpid()) + "\n")
    handle.flush()
    return handle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="perform one health/admission pass")
    parser.add_argument("--poll-secs", type=float, default=POLL_SECONDS)
    args = parser.parse_args()
    lock = acquire_lock()
    del lock  # held until process exit

    if not HF_BIN.is_file():
        emit("fatal", reason="hf executable missing", path=str(HF_BIN))
        notify("HF executable is missing; watcher cannot acquire models", "fatal")
        return 2

    emit("watcher_started", pid=os.getpid(), floor_bytes=FLOOR_BYTES,
         max_download_jobs=MAX_DOWNLOAD_JOBS, max_workers=MAX_WORKERS)
    children: dict[str, subprocess.Popen[str]] = {}
    manifest_cache: dict[str, tuple[list[str], int, dict[str, int]]] = {}
    retry_after: dict[str, float] = {}
    retry_count: dict[str, int] = {}
    last_job_bytes: dict[str, int] = {}
    last_progress: dict[str, float] = {}
    last_stall_notice: dict[str, float] = {}
    last_auth = 0.0
    last_net = None
    last_net_time = time.monotonic()
    last_network_emit = 0.0
    last_state_emit = 0.0
    last_refresh: dict[str, float] = {}
    low_rx_since: dict[str, float] = {}
    rate_rearmed: set[str] = set()
    refresh_requested: set[str] = set()
    blocked_auth_notice: set[str] = set()
    notified_low_disk = False
    last_p0_done = False
    last_route_notice = 0.0

    while True:
        loop_started = time.monotonic()
        started_this_loop: set[str] = set()
        try:
            free = free_bytes()
        except OSError as exc:
            emit("disk_error", error=redact(str(exc)))
            notify("ModelLake drive is unavailable; downloads are not being admitted", "disk")
            if args.once:
                return 2
            time.sleep(min(args.poll_secs, 60))
            continue

        if free < FLOOR_BYTES:
            emit("disk_floor_breached", free_bytes=free, floor_bytes=FLOOR_BYTES)
            notify("Drive free space is below the 100 GB floor", "disk")
        elif free < WARN_BYTES and not notified_low_disk:
            notified_low_disk = True
            emit("disk_floor_warning", free_bytes=free, floor_bytes=FLOOR_BYTES)
            notify("Drive free space is approaching the 100 GB floor", "disk")
        elif free >= WARN_BYTES:
            notified_low_disk = False

        rows = process_rows()
        all_items = P0 + QUEUE
        p0_tags = {str(item["tag"]) for item in P0}
        active_tags = set()
        active_remaining = 0
        p0_done = True
        states = []

        # The current processes may have been launched by another parent. Attach
        # to them by exact repo/revision/destination; never create a duplicate.
        for item in all_items:
            tag = str(item["tag"])
            pids = matching_pids(item, rows)
            running = bool(pids)
            cached_manifest = manifest_cache.get(tag)
            if cached_manifest is None:
                cached_manifest_for = load_cached_manifest(item)
                if cached_manifest_for is not None:
                    cached_manifest = cached_manifest_for
                    manifest_cache[tag] = cached_manifest
                else:
                    try:
                        cached_manifest = manifest_for(item)
                        manifest_cache[tag] = cached_manifest
                    except Exception as exc:
                        if item in P0:
                            p0_done = False
                        states.append({"job": tag, "state": "manifest_wait", "error": str(exc)})
                        continue
            files, resolved_expected, sizes = cached_manifest
            expected = int(item.get("expected", resolved_expected))
            if expected != resolved_expected:
                emit("manifest_size_reconciled", job=tag,
                     prior_expected=expected, resolved_expected=resolved_expected)
                expected = resolved_expected
                item["expected"] = expected
            # While a process is alive, reserve the entire manifest
            # conservatively; this lightweight metadata probe is only used to
            # tell advancing transfers from genuinely stalled ones.
            present = None
            remaining = expected
            is_done = False
            if running:
                active_tags.add(tag)
                active_remaining += remaining
                observed_bytes = durable_bytes(item, files, sizes)
                previous_bytes = last_job_bytes.get(tag)
                if previous_bytes is None or observed_bytes > previous_bytes:
                    last_progress[tag] = loop_started
                elif loop_started - last_progress.get(tag, loop_started) >= STALL_SECONDS:
                    if loop_started - last_stall_notice.get(tag, 0) >= STALL_SECONDS:
                        last_stall_notice[tag] = loop_started
                        emit("download_stall", job=tag, pids=pids,
                             remaining_bytes=remaining,
                             reason="no durable file growth observed")
                        notify(f"No durable file growth for 15 minutes: {tag}", "network")
                last_job_bytes[tag] = observed_bytes
                states.append({"job": tag, "state": "active", "pids": pids,
                               "present_bytes": present, "remaining_bytes": remaining})
            else:
                is_done = complete(item, files, sizes)
                present = expected if is_done else None
                remaining = 0 if is_done else expected
                states.append({"job": tag, "state": "complete" if is_done else "absent",
                               "present_bytes": present, "remaining_bytes": remaining})
            if item in P0 and (running or not is_done):
                p0_done = False

        if p0_done and not last_p0_done:
            emit("p0_recovery_complete", free_bytes=free, active_remaining_bytes=active_remaining)
        last_p0_done = p0_done

        # Reap children that this watcher started. A nonzero result is left on
        # disk for a retry; no existing partial is cleared.
        for tag, process in list(children.items()):
            result = process.poll()
            if result is not None:
                intentional_refresh = tag in refresh_requested
                refresh_requested.discard(tag)
                emit("download_exit", job=tag, returncode=result,
                     intentional_refresh=intentional_refresh)
                del children[tag]
                if result != 0:
                    if intentional_refresh:
                        retry_count[tag] = 0
                        retry_after[tag] = loop_started + 5
                    else:
                        retry_count[tag] = retry_count.get(tag, 0) + 1
                        retry_after[tag] = loop_started + min(3600, 30 * (2 ** min(retry_count[tag], 6)))
                        notify(f"Download exited with code {result}: {tag}", "download")
                else:
                    retry_count[tag] = 0

        # Recover an interrupted P0 download in the same pinned destination.
        # This path is only reached after the process is absent, never while it
        # is healthy or merely slow.
        for item in P0:
            tag = str(item["tag"])
            if tag in children or matching_pids(item, rows):
                continue
            manifest = manifest_cache.get(tag)
            if manifest is None:
                continue
            files, expected, sizes = manifest
            item["expected"] = expected
            if complete(item, files, sizes) or loop_started < retry_after.get(tag, 0):
                continue
            log_path = DOWNLOAD_DIR / f"watch-{tag}-{datetime.now().strftime('%Y%m%dT%H%M%S%z')}.log"
            children[tag] = launch(item, files, log_path)
            last_job_bytes[tag] = durable_bytes(item, files, sizes)
            last_progress[tag] = loop_started
            started_this_loop.add(tag)

        # Admit queued specimens even while the remaining P0 giant runs. The
        # four-job cap and conservative storage reservation still apply, and
        # QUEUE order is deliberately smallest selected manifest first.
        active_count = len(active_tags) + len(children)
        for item in QUEUE:
            if active_count >= MAX_DOWNLOAD_JOBS:
                break
            tag = str(item["tag"])
            if tag in active_tags or tag in children or matching_pids(item, rows):
                continue
            if item.get("requires_manual_auth"):
                if tag not in blocked_auth_notice:
                    blocked_auth_notice.add(tag)
                    emit("admission_blocked_auth", job=tag,
                         reason="manual upstream terms acceptance required")
                    notify(f"Manual HF access approval required; not auto-downloading: {tag}", "auth")
                continue
            if loop_started < retry_after.get(tag, 0):
                continue
            try:
                manifest = manifest_cache.get(tag)
                if manifest is None:
                    manifest = load_cached_manifest(item) or manifest_for(item)
                    manifest_cache[tag] = manifest
                files, expected, sizes = manifest
                item["expected"] = expected
                # File-by-file exactness prevents partial bytes from being
                # mistaken for a complete specimen.
                if complete(item, files, sizes):
                    emit("already_complete", job=tag, expected_bytes=expected)
                    continue
                # A queued job may have an old partial destination. Reserve
                # the full selected manifest until an exact post-exit check
                # proves otherwise; this is intentionally conservative.
                present = None
                remaining = expected
                scratch = max(10_000_000_000, int(remaining * 0.05))
                uncertainty = max(5_000_000_000, int(remaining * 0.02))
                projected = free - active_remaining - remaining - scratch - uncertainty - KNOWN_TEMP_BYTES
                if projected < FLOOR_BYTES:
                    emit("admission_blocked_storage", job=tag, expected_bytes=expected,
                         remaining_bytes=remaining, free_bytes=free,
                         active_remaining_bytes=active_remaining,
                         projected_free_bytes=projected, floor_bytes=FLOOR_BYTES)
                    continue
                log_path = DOWNLOAD_DIR / f"watch-{tag}-{datetime.now().strftime('%Y%m%dT%H%M%S%z')}.log"
                children[tag] = launch(item, files, log_path)
                last_job_bytes[tag] = durable_bytes(item, files, sizes)
                last_progress[tag] = loop_started
                started_this_loop.add(tag)
                active_count += 1
            except Exception as exc:
                message = str(exc)
                emit("admission_error", job=tag, error=redact(message))
                if re.search(r"401|403|unauthorized|gated|access", message, re.I):
                    notify(f"HF access/authentication needs attention: {tag}", "auth")
                # Continue to smaller/diverse candidates rather than stalling
                # the whole queue on one gated or temporarily unavailable repo.

        counters = interface_counters(interface_name())
        net_in = net_out = 0
        rx_rate = None
        if counters is not None:
            if last_net is not None:
                elapsed = max(0.1, loop_started - last_net_time)
                net_in = max(0, counters[0] - last_net[0])
                net_out = max(0, counters[1] - last_net[1])
                rx_rate = net_in / elapsed
                if loop_started - last_network_emit >= NETWORK_SAMPLE_EMIT_SECONDS:
                    emit("network_sample", rx_bytes=net_in, tx_bytes=net_out,
                         rx_bytes_per_sec=round(rx_rate),
                         tx_bytes_per_sec=round(net_out / elapsed))
                    last_network_emit = loop_started
            last_net = counters
            last_net_time = loop_started
        elif active_tags:
            emit("network_route_missing", active_jobs=sorted(active_tags))
            notify("Network route disappeared while ModelLake downloads are active", "network")

        # A sustained interface-rate decline arms the recovery path.  This is
        # deliberately separate from durable progress: aggregate RX can be
        # high while a particular shard is stalled, and low RX can be benign
        # during a filesystem commit.  The refresh gate below requires both a
        # rate arm and no durable growth before signaling the same pinned job.
        if RATE_BASED_REFRESH_ENABLED and active_tags and rx_rate is not None:
            if rx_rate < LOW_RX_BYTES_PER_SEC:
                low_rx_since.setdefault("__active__", loop_started)
                if (loop_started - low_rx_since["__active__"] >= RATE_ARM_SECONDS
                        and "__active__" not in rate_rearmed):
                    rate_rearmed.add("__active__")
                    emit("network_rate_dwindling",
                         active_jobs=sorted(active_tags),
                         rx_bytes_per_sec=round(rx_rate),
                         arm_seconds=RATE_ARM_SECONDS)
            else:
                if "__active__" in rate_rearmed:
                    emit("network_rate_recovered",
                         active_jobs=sorted(active_tags),
                         rx_bytes_per_sec=round(rx_rate))
                low_rx_since.pop("__active__", None)
                rate_rearmed.discard("__active__")

        # Refresh only a genuinely stalled transfer, never merely because an
        # aggregate interface sample dipped.  Each process is an exact-revision,
        # separate-session downloader started with start_new_session=True; use
        # its process group when available so its Xet child is stopped too.
        active_for_refresh = sorted(active_tags | set(children))
        rate_triggered = (RATE_BASED_REFRESH_ENABLED and rx_rate is not None
                          and rx_rate < LOW_RX_BYTES_PER_SEC)
        rate_armed_stall = (RATE_BASED_REFRESH_ENABLED and "__active__" in rate_rearmed
                            and any(loop_started - last_progress.get(tag, loop_started)
                                    >= RATE_STALL_REFRESH_SECONDS
                                    for tag in active_for_refresh))
        stale_triggered = (STALE_REFRESH_ENABLED and any(
            loop_started - last_progress.get(tag, loop_started)
            >= RECOVERY_REFRESH_SECONDS for tag in active_for_refresh))
        if active_for_refresh and (rate_triggered or stale_triggered):
            refreshed = []
            fresh_rows = process_rows()
            by_tag = {str(item["tag"]): item for item in all_items}
            for tag in active_for_refresh:
                # Freshly launched sessions need a complete observation window
                # before they can be considered degraded.
                if tag in started_this_loop:
                    continue
                if loop_started - last_refresh.get(tag, 0) < RECOVERY_COOLDOWN_SECONDS:
                    continue
                no_growth_for = loop_started - last_progress.get(tag, loop_started)
                if no_growth_for < (RATE_STALL_REFRESH_SECONDS
                                    if rate_armed_stall else RECOVERY_REFRESH_SECONDS):
                    continue
                item = by_tag.get(tag)
                if item is None:
                    continue
                pids = matching_pids(item, fresh_rows)
                stopped = []
                for pid in pids:
                    try:
                        pgid_text = subprocess.check_output(
                            ["/bin/ps", "-o", "pgid=", "-p", str(pid)],
                            text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
                        pgid = int(pgid_text)
                    except Exception:
                        pgid = pid
                    try:
                        if pgid == pid and pgid != os.getpid():
                            os.killpg(pgid, signal.SIGTERM)
                        elif pid != os.getpid():
                            os.kill(pid, signal.SIGTERM)
                        stopped.append(pid)
                    except (ProcessLookupError, PermissionError) as exc:
                        emit("download_recovery_signal_skipped", job=tag, pid=pid,
                             error=redact(str(exc)))
                if stopped:
                    last_refresh[tag] = loop_started
                    retry_after[tag] = loop_started + 10
                    refresh_requested.add(tag)
                    refreshed.append({"job": tag, "pids": stopped,
                                      "no_growth_seconds": round(no_growth_for)})
            if refreshed:
                emit("download_recovery_refresh", jobs=refreshed,
                     low_rx_bytes_per_sec=(round(rx_rate)
                                           if rx_rate is not None else None),
                     rx_bytes_per_sec=(round(rx_rate)
                                       if rx_rate is not None else None),
                     refresh_interval_seconds=RECOVERY_REFRESH_SECONDS,
                     refresh_cooldown_seconds=RECOVERY_COOLDOWN_SECONDS,
                     reason=("confirmed no partial-byte growth after sustained rate decline; "
                             "same pinned destinations preserved" if rate_armed_stall
                             else "confirmed no partial-byte growth; same pinned destinations preserved"))
                notify("Refreshed degraded ModelLake download sessions; pinned partials preserved", "network")
                rate_rearmed.discard("__active__")
                low_rx_since.pop("__active__", None)

        if loop_started - last_auth >= AUTH_CHECK_SECONDS or last_auth == 0:
            status, detail = auth_check()
            last_auth = loop_started
            emit("hf_auth_check", status=status, detail=detail)
            if status == "auth_failed":
                notify("HF token is invalid/expired; re-run hf auth login", "auth")
            elif status == "network_error":
                emit("hf_auth_network_error", detail=detail)

        if args.once or loop_started - last_state_emit >= STATE_SAMPLE_EMIT_SECONDS:
            emit("watcher_sample", free_bytes=free, p0_done=p0_done,
                 active_jobs=sorted(active_tags | set(children)),
                 active_remaining_bytes=active_remaining, states=states)
            last_state_emit = loop_started
        if args.once:
            return 0
        sleep_for = max(0.05, args.poll_secs - (time.monotonic() - loop_started))
        time.sleep(sleep_for)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        emit("watcher_stopped", reason="keyboard_interrupt")
        raise
