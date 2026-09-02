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
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
# Absolute package import, not a bare sibling import: this file is imported
# both as a standalone script (sys.path[0] == this directory) and as
# tools.odyssey.modellake_watch (tests, PYTHONPATH=.). A bare `import
# modellake_promote` would load a second, distinct module object under the
# script-path sys.modules key -- a test that patches
# tools.odyssey.modellake_promote.PARTIAL_ROOT would silently not affect the
# copy this file actually calls. One canonical import identity either way.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.odyssey import modellake_promote  # noqa: E402
ODYSSEY = REPO_ROOT / "workspace" / "campaign" / "odyssey"
DOWNLOAD_DIR = ODYSSEY / "downloads"
# Keep watcher control metadata on the internal workspace. Writing it on the
# busy external volume can block health sampling behind ModelLake I/O.
MANIFEST_DIR = ODYSSEY / "watch-manifests"
DRIVE = Path("/Volumes/corpdrive")
MODEL_ROOT = Path("/Volumes/corpdrive/hawking-modellake")
SPECIMEN_ROOT = MODEL_ROOT / "specimens"
HF_BIN = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/hf")
PYTHON_BIN = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")
LOG = DOWNLOAD_DIR / "modellake-watch.jsonl"
LOCK_PATH = ODYSSEY / ".modellake-watch.lock"
# Keep a dedicated sparse APFS cache volume on the internal SSD.  The image is
# deliberately larger than the requested free-space floor, while admission is
# governed by the live internal-volume free bytes.  The older 50 GB image is
# left mounted for already-running workers and is never moved or modified.
SSD_XET_IMAGE = ODYSSEY / "ssd-xet-cache-large.sparseimage"
SSD_XET_CACHE = ODYSSEY / "ssd-xet-cache-large"
SSD_XET_IMAGE_CAPACITY_BYTES = 410_000_000_000
SSD_XET_CHUNK_CACHE_TARGET_BYTES = 400_000_000_000
SSD_FREE_FLOOR_BYTES = 50_000_000_000
SSD_FREE_GUARD_BYTES = 5_000_000_000

FLOOR_BYTES = 100_000_000_000
WARN_BYTES = 250_000_000_000
# Overridable for the same reason MAX_WORKERS is: concurrent JOBS multiply the
# per-child footprint, and on a host where the lake volume absorbs ~87 MB/s the
# fourth job buys no throughput -- it only adds another ~30 GB of chunks queued
# behind the same disk. Read at import; see the MAX_WORKERS note on kickstart.
MAX_DOWNLOAD_JOBS = max(
    1, int(os.environ.get("HAWKING_MODELLAKE_MAX_DOWNLOAD_JOBS", "4") or 4)
)
# The hub CLI defaults to eight file workers.  The two P0 repositories are
# deliberately large, sharded artefacts and this host has a 10 GbE uplink, so
# use a high but bounded fan-out per pinned transfer.  This is persisted in
# the launch watcher, not dependent on an interactive terminal.
#
# MEASURED COST, 2026-09-01: one `hf download` child at MAX_WORKERS=16 was
# observed holding 20.70 GB RSS (pid 17132, Qwen3-Coder-30B-A3B), against a
# resident body of 1.18 GB.  With MAX_DOWNLOAD_JOBS=4 the worst case is 64
# buffering file workers, and on this host that is what drove free RAM to
# 8.5 GB and pushed the resident into WAITING_FOR_MEMORY against its 14.4 GB
# reserve -- an acquisition starving the science it is acquiring for.
#
# MEASURED, 2026-09-02: throughput vs worker count, from 150,254 network_sample
# rows in this watcher's own log covering 16 through 192 concurrent workers.
# p90 receive rate is FLAT: 16 workers -> 208 MB/s, 32 -> 224, 64 -> 220,
# 96 -> 220, 144 -> 242, 160 -> 270, 192 -> 222.  Twelve times the workers buys
# about six percent; the uplink saturates near 210-270 MB/s (~1.8 Gbit/s) and
# ONE job at 16 workers already reaches 93% of the best rate ever recorded.
# The write path is the real wall: a 2 GB dd to the lake volume under two live
# transfers ran at 19.8 MB/s spare against their ~67 MB/s, so corpdrive absorbs
# roughly 87 MB/s total.  At ~13 MB/s per worker that made 2 jobs x 16 a 4.8:1
# oversubscription, and the surplus is not speed -- it is the ~28 GB physical
# footprint each child holds as chunks queue behind a disk that cannot drain
# them, which is what starved the resident into WAITING_FOR_MEMORY.
#
# NOTE: this is read at IMPORT, so `launchctl setenv` alone does NOT reach a
# running watcher -- os.environ is a snapshot.  Applying a new value needs
# `launchctl kickstart -k gui/$UID/com.hawking.modellake.watch`.  That restart
# is safe: children are spawned with start_new_session=True and are rediscovered
# by argv match in matching_pids(), so a new watcher re-adopts live transfers
# instead of duplicating them.
MAX_WORKERS = max(1, int(os.environ.get("HAWKING_MODELLAKE_MAX_WORKERS", "16")))
POLL_SECONDS = 0.10
NETWORK_SAMPLE_EMIT_SECONDS = 1.0
STATE_SAMPLE_EMIT_SECONDS = 10.0
STALL_SECONDS = 15 * 60
AUTH_CHECK_SECONDS = 10 * 60
# G101: without this, a sealed specimen only becomes a WorkUnit when a human
# remembers to run `python3 tools/future/modellake_events.py --build` by
# hand. A seal is a rare event; this cadence is deliberately far coarser than
# the download-health polling above so the consumer's disk reads never
# compete with the two live transfers this watcher is admitting.
MODELLAKE_EVENTS_INTERVAL_SECONDS = 5 * 60
# When there is nothing admissible left -- every pinned job complete, running,
# or blocked -- the loop used to keep spinning at the 0.1s poll and re-announce
# every finished specimen on every tick. That is where 801,116 of the 1,144,619
# rows in this watcher's log came from, and why the log reached 595 MB. Idle now
# rests for this long and then RE-ARMS: rescan, re-admit, keep going until the
# manifest is satisfied. It never delays live work -- the wait is only entered
# when nothing could be started, and it is cut short by a pending retry.
IDLE_REARM_SECONDS = 30 * 60
# G168: complete(item, ...) returning True used to just mean "skip launching
# a redundant download" -- nothing acted on it, and SPECIMEN_ROOT was never
# written by this module. A model could sit finished in partial/ forever if
# the exact poll tick that noticed completion was missed (process restart,
# a manifest resolved late) because nothing ever looked again. This is that
# second look: a low-frequency, tag-agnostic sweep of partial/, specimens/
# and watch-manifests/ that promotes anything complete-but-unpromoted and
# reports what it cannot safely fix itself. Coarser than the poll loop for
# the same reason MODELLAKE_EVENTS_INTERVAL_SECONDS is: a directory walk
# should not compete with the two live transfers' I/O.
RECONCILE_INTERVAL_SECONDS = 30 * 60
KNOWN_TEMP_BYTES = 20_000_000_000
# A transient interface dip is not evidence that a pinned downloader is bad.
# Transfer rate is telemetry, not sufficient evidence to terminate a live
# exact-revision session.  The hub/Xet client already owns connection retry;
# the watcher recovers an actually exited worker into the same destination.
# A rate dip arms recovery telemetry, but never terminates a session by itself.
# Refresh still requires no durable byte growth, so a slow-but-healthy shard is
# left alone.  The short arm window improves recovery from genuine stalls.
# MEASURED 2026-09-02: this path, NOT the stale path, is what fires on this
# host. In a 15-minute window every one of 7 refreshes carried
# "no partial-byte growth after sustained rate decline", and the GLM lane
# lost 9.25 GB of ALREADY-VERIFIED final shards (confirmed not promotion:
# no promotion event, no specimen on disk). LOW_RX_BYTES_PER_SEC is
# 150 MB/s while the lake volume is a 2.5" USB HDD that absorbs ~87 MB/s
# total -- the floor is above what the hardware can sustain, so the rate
# path can never be satisfied and fires forever. Overridable so it can be
# isolated without editing source. Read at import; needs a kickstart.
RATE_BASED_REFRESH_ENABLED = (
    os.environ.get("HAWKING_MODELLAKE_RATE_REFRESH", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
RATE_ARM_SECONDS = 3
RATE_STALL_REFRESH_SECONDS = 10
# A session with no growth in the actual Hub partial cache for the full
# recovery interval is eligible for one same-destination refresh.  This is
# deliberately separate from aggregate interface-rate telemetry.
STALE_REFRESH_ENABLED = True
# Overridable so a refresh-cadence A/B can be run without editing source
# between arms. Read at import; a change needs `launchctl kickstart -k`.
RECOVERY_REFRESH_SECONDS = max(
    1, int(os.environ.get("HAWKING_MODELLAKE_RECOVERY_REFRESH_SECONDS", "60") or 60)
)
RECOVERY_COOLDOWN_SECONDS = 3 * 60 + 30
LOW_RX_BYTES_PER_SEC = 150_000_000


def tree_bytes(root: Path) -> int:
    """Return regular-file bytes below root without following symlinks."""
    total = 0
    if not root.is_dir():
        return total
    for directory, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            try:
                path = Path(directory) / filename
                if path.is_symlink():
                    continue
                total += path.stat().st_size
            except OSError:
                continue
    return total


def ssd_xet_mounted() -> bool:
    return SSD_XET_CACHE.is_dir() and os.path.ismount(SSD_XET_CACHE)


def ensure_ssd_xet_cache() -> bool:
    """Attach the prepared cache image if it is not already mounted."""
    if ssd_xet_mounted():
        return True
    if not SSD_XET_IMAGE.is_file():
        return False
    try:
        SSD_XET_CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["/usr/bin/hdiutil", "attach", "-quiet", "-nobrowse",
             "-mountpoint", str(SSD_XET_CACHE), str(SSD_XET_IMAGE)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return ssd_xet_mounted()


def internal_ssd_free_bytes() -> int | None:
    """Return free bytes on the internal volume hosting the cache."""
    try:
        stats = os.statvfs(Path.home())
    except OSError:
        return None
    return int(stats.f_bavail * stats.f_frsize)


def ssd_xet_plan() -> tuple[int, int, bool, bool]:
    """Return (current bytes, chunk budget, policy-ok, mounted)."""
    mounted = ssd_xet_mounted()
    if not mounted:
        return 0, 0, False, False
    current = tree_bytes(SSD_XET_CACHE)
    internal_free = internal_ssd_free_bytes()
    if internal_free is None:
        return current, 0, False, True
    available = max(0, internal_free - SSD_FREE_FLOOR_BYTES - SSD_FREE_GUARD_BYTES)
    budget = min(SSD_XET_CHUNK_CACHE_TARGET_BYTES, available)
    policy_ok = (internal_free > SSD_FREE_FLOOR_BYTES + SSD_FREE_GUARD_BYTES
                 and current < SSD_XET_CHUNK_CACHE_TARGET_BYTES)
    return current, budget, policy_ok, True


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(value: str) -> str:
    value = re.sub(r"hf_[A-Za-z0-9_-]+", "hf_[REDACTED]", value)
    value = re.sub(r"(?i)(token|authorization|bearer)[=: ]+\S+", r"\1=[REDACTED]", value)
    return value[-1000:]


# The sweeps moved off the supervision loop write here too, and a
# `watcher_sample` row carrying `states` is far larger than the kernel will
# append atomically. One lock keeps two threads from interleaving a row and
# corrupting the JSONL that every measurement in this file is read from.
_EMIT_LOCK = threading.Lock()


# This log had reached 612.6 MB and was still growing at roughly 10 Hz. It is
# append-only and nothing ever reclaimed it, so on the long horizon this
# watcher is built for it fills the volume it is supposed to be protecting.
# Two generations is enough: _download_history only ever tails 20k lines, and
# the rotated file is kept precisely so that tail still spans a rotation.
LOG_MAX_BYTES = int(os.environ.get("HAWKING_MODELLAKE_LOG_MAX_BYTES", 64_000_000))
LOG_GENERATIONS = 2


def _rotate_log_if_needed() -> None:
    """Roll the event log before it can grow without bound.

    Callers hold _EMIT_LOCK. emit() reopens the file on every write, so a
    rename is clean: the next write recreates LOG rather than continuing to
    an unlinked inode, which is exactly what would happen if the handle were
    held open across the rotation.
    """
    try:
        if LOG.stat().st_size < LOG_MAX_BYTES:
            return
    except OSError:
        return
    for gen in range(LOG_GENERATIONS - 1, 0, -1):
        older = LOG.with_suffix(LOG.suffix + f".{gen + 1}")
        newer = LOG.with_suffix(LOG.suffix + f".{gen}")
        if newer.is_file():
            try:
                newer.replace(older)
            except OSError:
                return
    try:
        LOG.replace(LOG.with_suffix(LOG.suffix + ".1"))
    except OSError:
        return


def emit(event: str, **fields: object) -> None:
    row = {"ts": now(), "event": event, **fields}
    line = json.dumps(row, sort_keys=True) + "\n"
    with _EMIT_LOCK:
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        _rotate_log_if_needed()
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(line)


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
    # REMOVED 2026-09-02: nvidia/personaplex-7b-v1, google/gemma-3-4b-it and
    # meta-llama/Llama-4-Scout-17B-16E-Instruct each need a human to accept
    # upstream terms on the model page before any token can fetch them. The
    # watcher was correctly refusing them and emitting admission_blocked_auth
    # on every pass, which is noise for work that can never start unattended.
    # Re-add them (with requires_manual_auth) once the licences are accepted.
    # REMOVED 2026-09-02: stabilityai/stable-audio-open-1.0 and facebook/blt-7b
    # also answer "Access denied. This repository requires approval." Flagging
    # them requires_manual_auth stopped the wasted relaunches but still left
    # two permanently unstartable entries in a queue whose whole purpose is
    # unattended work -- so ModelLake read as "2 remaining" forever. Removed
    # for the same reason as the three above. Re-add once accepted upstream.
    job("nvidia/audio-flamingo-3", "ee26c58423988d7d7cda7b85dd3ce5d97ee8753d", "all", "P2-HIGH"),
    job("RWKV/RWKV7-13.3B-20260805", "64ffe5934178f40fb2c6de13f12cffaf9058f243", "safe", "P1-S"),
    job("microsoft/Phi-4-reasoning-plus", "69baf8528e1bcf05f475034d9e5dd32875ed125f", "safe", "P1-A"),
    job("LiquidAI/LFM2-24B-A2B", "a3bbacd91a678b97712f0e323e52f8c24ba29542", "safe", "P1-A"),
    job("arcinstitute/evo2_40b", "d529aa57c30771814217ad89baaeaf6e2315c7d7", "all", "P2-CONDITIONAL"),
    job("Wan-AI/Wan2.2-T2V-A14B", "c8c270b13ee05bfa474194ac9fb07a5868a97cea", "all", "P2-HIGH"),
    job("zai-org/GLM-5.3-Flash", "04c4e9e95c5da8862dced7e5056455116f83a7e0", "all", "P2-HIGH"),
    job("thinkingmachines/Inkling-Small", "8cc5877b44d343f88b92086aa1fb72897950f06a", "safe", "P1-S"),
    # Fresh diversity wave.  These public small specimens are placed before
    # the remaining large/gated entries so an available slot fills with useful
    # diversity without changing any active pinned destination.
    job("ibm-granite/granite-4.0-h-micro", "d5f01a3ea75f088947be3aae039f4ad52837dfde", "safe", "P1-DIVERSITY"),
    job("ibm-granite/granite-4.0-h-tiny", "791e0d3d28c86e106c9b6e0b4cecdee0375b6124", "safe", "P1-DIVERSITY"),
    job("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "482adb537c021c86670beed01cd58990d01e72e4", "safe", "P1-DIVERSITY"),
    job("microsoft/Phi-4-multimodal-instruct", "93f923e1a7727d1c4f446756212d9d3e8fcc5d81", "safe", "P1-DIVERSITY"),
    job("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "safe", "P1-DIVERSITY"),
    job("LiquidAI/LFM2.5-1.2B-Instruct", "0f604ada3f766f9f257460c4c9f0b5d6f69d431b", "safe", "P1-SMALL"),
    job("tiiuae/Falcon3-1B-Instruct", "28ba2251970a01dd1edc7ba7dad2eb71216ccfdf", "safe", "P1-SMALL"),
    job("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083", "safe", "P1-SMALL"),
    job("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", "safe", "P1-SMALL"),
    job("Qwen/Qwen3-Coder-30B-A3B-Instruct", "b2cff646eb4bb1d68355c01b18ae02e7cf42d120", "safe", "P1-DIVERSITY"),
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
    # Completed exact-revision specimens may be promoted out of the active
    # partial area.  Check the canonical specimen location as well as the
    # legacy download destination so promotion cannot cause a re-download.
    roots = [Path(str(item["destination"])), SPECIMEN_ROOT / str(item["tag"])]
    for root in roots:
        if not root.is_dir():
            continue
        complete_here = True
        for name in files:
            path = root / name
            try:
                if not path.is_file() or path.stat().st_size != sizes[name]:
                    complete_here = False
                    break
            except (FileNotFoundError, OSError):
                complete_here = False
                break
        if complete_here:
            return True
    return False


def _notify_sealed_source(tag: str, action: object, *, source: str) -> None:
    """Landing path for SLEEPING_SPECIMEN_WU: fire SEALED_SOURCE_READY.

    Disk is authority (specimen directory). A notify/harvest failure must
    not undo a promotion or take down the admission loop.
    """
    if action not in {"PROMOTED", "ALREADY_PROMOTED"}:
        return
    try:
        from tools.future.sleeping_specimens import (
            WAKE_SEALED_SOURCE_READY,
            notify_sealed_source_ready,
        )
        from tools.future.wakeup import harvest_sealed_specimens

        event = notify_sealed_source_ready(
            tag, source=source, specimen_root=SPECIMEN_ROOT
        )
        if not event.get("ready"):
            return
        harvest_sealed_specimens(
            [
                {
                    "id": f"odyssey-i.sleeping.{tag}",
                    "wake_condition": WAKE_SEALED_SOURCE_READY,
                    "modellake_identity": {"tag": tag},
                    "status": "sleeping",
                }
            ],
            specimen_root=SPECIMEN_ROOT,
        )
    except Exception:
        return


def promote_if_needed(tag: str, destination: str) -> dict[str, object] | None:
    """Move a verified-complete partial payload into specimens/.

    Returns modellake_promote.promote()'s outcome dict, or None when there is
    no partial payload left to move -- the common, steady-state case once a
    tag has already been promoted. The one is_dir() check keeps a 100ms poll
    tick from re-stat-ing every file of an already-promoted specimen forever.
    modellake_promote.promote() re-verifies completeness itself before moving
    anything, is idempotent, and never overwrites an existing destination.
    """
    if not Path(destination).is_dir():
        return None
    return modellake_promote.promote(tag, go=True)


_LAST_COMPLETE_NOTICE: dict[str, float] = {}


def _promote_and_report(tag: str, destination: str, expected: int) -> None:
    """Promote a tag complete() just found, and report the outcome.

    Shared by the P0 recovery loop and the QUEUE admission loop -- both hit
    this exact state (complete() is True), and a promotion refusal is
    precisely the kind of thing this watcher exists to surface rather than
    silently `continue` past, which is what both call sites did before.
    """
    outcome = promote_if_needed(tag, destination)
    action = outcome["action"] if outcome is not None else "ALREADY_PROMOTED"
    # A specimen that is already promoted is re-observed on every poll tick and
    # says the same thing every time. Announce a repeat at most once per idle
    # cycle; anything that actually CHANGED (a real promotion, or a refusal
    # needing attention) is never throttled.
    now_s = time.monotonic()
    if action == "ALREADY_PROMOTED":
        if now_s - _LAST_COMPLETE_NOTICE.get(tag, 0.0) < IDLE_REARM_SECONDS:
            return
        _LAST_COMPLETE_NOTICE[tag] = now_s
    else:
        _LAST_COMPLETE_NOTICE[tag] = now_s
    emit("already_complete", job=tag, expected_bytes=expected, promotion=action)
    _notify_sealed_source(
        tag, action, source="tools.odyssey.modellake_watch._promote_and_report"
    )
    if action == "PROMOTED":
        notify(f"Promoted completed specimen out of partial/: {tag}", "modellake")
    elif action != "ALREADY_PROMOTED":
        notify(f"ModelLake promotion needs attention ({action}): {tag}", "modellake")


def _has_live_writer(destination: Path, rows: list[tuple[int, str]]) -> bool:
    """True if any process command line references this destination path.

    Broader than matching_pids(): reconciliation also walks legacy partial
    directories with no P0/QUEUE entry any more (no repo/revision to match
    against), and must never promote (move) a directory a live process is
    still writing into.
    """
    needle = str(destination)
    return any(needle in command for _pid, command in rows)


def _tail_json_lines(path: Path, max_lines: int) -> list[dict[str, object]]:
    """Return up to the last max_lines JSON objects from an append-only
    JSONL log.

    ponytail: reads every line to find the tail (O(n) in log length);
    acceptable at RECONCILE_INTERVAL_SECONDS frequency. Upgrade to a
    reverse/seek read if the log ever grows large enough to make that cost
    matter at this call rate.
    """
    # Read the rotated generation first so the tail spans a rotation. Without
    # this, the roll would erase every download_started this watcher remembers
    # and reconcile() would read "started, and now nothing on disk" -- its
    # vanished-payload signal -- for jobs that are simply older than the roll.
    sources = [
        path.with_suffix(path.suffix + f".{gen}")
        for gen in range(LOG_GENERATIONS, 0, -1)
    ] + [path]
    tail: deque = deque(maxlen=max_lines)
    found = False
    for source in sources:
        if not source.is_file():
            continue
        found = True
        try:
            with source.open("r", encoding="utf-8") as handle:
                tail.extend(handle)
        except OSError:
            continue
    if not found:
        return []
    out = []
    for line in tail:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _download_history(max_lines: int = 20_000) -> tuple[dict[str, int], set[str]]:
    """What this watcher's own JSONL log remembers about each job: the most
    recent exit code per tag, and which tags a download was ever actually
    started for.

    The exit code is returned for one purpose only -- flagging disagreement
    with the manifest-verified truth in reconcile(). It is diagnostic, never
    authority: Qwen2.5-72B recorded exit_code=1, acquired=false,
    bytes_on_disk=0 in a downloader's own bookkeeping while 47/47 files were
    present and correct on disk. The started-tags set exists only so
    reconcile() can tell "never admitted yet" (most of QUEUE, at any given
    moment -- not an anomaly) from "started, and now there is no partial
    directory and no specimen either" (a genuinely vanished payload).
    """
    last_exit: dict[str, int] = {}
    started: set[str] = set()
    for row in _tail_json_lines(LOG, max_lines):
        job = row.get("job")
        if job is None:
            continue
        event = row.get("event")
        if event == "download_exit" and "returncode" in row:
            last_exit[str(job)] = int(row["returncode"])
        elif event == "download_started":
            started.add(str(job))
    return last_exit, started


def reconcile() -> dict[str, object]:
    """Self-healing sweep over partial/, specimens/ and watch-manifests/.

    Real-time admission only notices completion at the instant it happens;
    miss that tick (the watcher was down, a manifest resolved late) and
    nothing ever looks again -- that is exactly how a model sat finished in
    partial/ for seven days. This is the second look: it re-derives status
    from the manifest and the two directories every time it runs, promotes
    anything complete-but-unpromoted, and reports what it cannot safely fix
    by itself instead of dropping it.
    """
    rows = modellake_promote.survey()
    proc_rows = process_rows()
    last_exit, started_tags = _download_history()

    promoted: list[str] = []
    refused: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []
    seen_in_partial = {str(row["tag"]) for row in rows}

    for row in rows:
        tag = str(row["tag"])
        if (SPECIMEN_ROOT / tag).is_dir():
            # Two directories claiming one identity. promote() would already
            # refuse this (never merge, never overwrite) -- name it as its
            # own anomaly so it reads as a conflict needing a human, not an
            # ordinary "still downloading" skip.
            anomalies.append({"kind": "duplicate_source", "tag": tag})
            continue
        if not row["complete"]:
            continue
        destination = modellake_promote.PARTIAL_ROOT / tag
        if _has_live_writer(destination, proc_rows):
            continue
        outcome = modellake_promote.promote(tag, go=True)
        action = outcome["action"]
        _notify_sealed_source(
            tag, action, source="tools.odyssey.modellake_watch.reconcile"
        )
        if action == "PROMOTED":
            promoted.append(tag)
        else:
            refused.append({"tag": tag, "action": action, "reason": outcome.get("reason")})
        code = last_exit.get(tag)
        if code not in (None, 0):
            anomalies.append({"kind": "stale_downloader_state", "tag": tag,
                               "recorded_exit_code": code})

    known_tags = {str(item["tag"]) for item in P0 + QUEUE}
    manifest_tags = ({p.stem for p in MANIFEST_DIR.glob("*.json")}
                     if MANIFEST_DIR.is_dir() else set())
    for tag in sorted(manifest_tags):
        has_specimen = (SPECIMEN_ROOT / tag).is_dir()
        if tag in started_tags and tag not in seen_in_partial and not has_specimen:
            anomalies.append({"kind": "registered_but_missing", "tag": tag})
        if tag not in known_tags:
            anomalies.append({"kind": "orphaned_manifest", "tag": tag})

    result: dict[str, object] = {"surveyed": len(rows), "promoted": promoted,
                                 "refused": refused, "anomalies": anomalies}
    emit("reconciliation_pass", **result)
    if anomalies:
        notify(f"ModelLake reconciliation found {len(anomalies)} anomaly(ies)", "modellake")
    return result


# MEASURED, 2026-09-02: `reconcile()` and `emit_modellake_events_once()` ran
# INLINE in the supervision loop, and both walk the lake. Over one window the
# loop went silent 108 times for a total of 240 MINUTES, the longest single
# blackout 1607s -- attributed by taking the last event emitted before each
# stall, which is `watcher_sample`, i.e. exactly these two calls. A blind
# supervisor does not track growth, does not refresh, and above all does not
# RELAUNCH: a transfer killed just before a sweep stayed dead for the whole
# sweep. That is why refreshes did not faithfully come back up. The refresh
# cadence itself is deliberate and is NOT changed here -- restarts are what
# holds the transfer rate up. Only the blocking is removed: these sweeps now
# run beside the loop, single-flight, so supervision never pauses for them.
_BACKGROUND_SWEEPS: dict[str, threading.Thread] = {}


def run_detached(name: str, fn) -> bool:
    """Run one sweep off the supervision loop. Single-flight per name.

    Returns True if a run was started (or one is still going), so the caller
    can advance its interval clock exactly as it did when the call blocked.
    A sweep still in flight is never started twice, which is what keeps a slow
    lake walk from stacking threads on a 0.1s poll.
    """
    live = _BACKGROUND_SWEEPS.get(name)
    if live is not None and live.is_alive():
        return True

    def body() -> None:
        try:
            fn()
        except Exception as exc:
            emit(f"{name}_error", error=redact(str(exc)))

    thread = threading.Thread(target=body, name=f"modellake-{name}", daemon=True)
    _BACKGROUND_SWEEPS[name] = thread
    thread.start()
    return True


def maybe_reconcile(loop_started: float, last_reconcile: float) -> float:
    """Gate reconcile() to RECONCILE_INTERVAL_SECONDS, firing immediately on
    the very first call -- the watcher's own startup is exactly the "look
    again" moment a killed-and-restarted process needs. Never raises: the two
    live transfers this watcher is admitting must not go down because a
    reconciliation sweep hit a bad manifest or a permissions error.
    """
    if last_reconcile and loop_started - last_reconcile < RECONCILE_INTERVAL_SECONDS:
        return last_reconcile
    run_detached("reconciliation", reconcile)
    return loop_started


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
    ssd_cache_mounted = ensure_ssd_xet_cache()
    ssd_cache_bytes, ssd_chunk_budget, ssd_within_limit, plan_mounted = ssd_xet_plan()
    ssd_cache_mounted = ssd_cache_mounted and plan_mounted
    ssd_free = internal_ssd_free_bytes()
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
        # The final files remain on the external ModelLake destination, but
        # Xet's bounded chunk cache stays on the internal SSD.  Sequential
        # reconstruction is specifically intended for spinning disks and
        # avoids turning the HDD into a random-write bottleneck.
        "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY": "1",
        # The legacy cache contains large open historical logs.  New workers
        # use the dedicated bounded volume and add no Xet log growth there.
        "HF_XET_LOG_DEST": "none",
        "HF_XET_LOG_DIR_MAX_SIZE": "250mb",
    })
    if ssd_cache_mounted:
        env.update({
            "HF_XET_CACHE": str(SSD_XET_CACHE),
            "HF_XET_CHUNK_CACHE_SIZE_BYTES": str(
                ssd_chunk_budget if ssd_within_limit else 0),
        })
    else:
        # Never let an unmounted path become an ordinary uncapped directory.
        # The worker can still download to the canonical HDD destination with
        # the sequential reconstruction setting.
        env["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
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
         expected=item.get("expected"), destination=item["destination"],
         ssd_xet_cache=str(SSD_XET_CACHE), ssd_xet_cache_bytes=ssd_cache_bytes,
         ssd_xet_image_capacity_bytes=SSD_XET_IMAGE_CAPACITY_BYTES,
         ssd_free_bytes=ssd_free,
         ssd_free_floor_bytes=SSD_FREE_FLOOR_BYTES,
         ssd_free_guard_bytes=SSD_FREE_GUARD_BYTES,
         ssd_xet_chunk_cache_budget_bytes=ssd_chunk_budget,
         ssd_xet_cache_policy_ok=ssd_within_limit,
         ssd_xet_cache_mounted=ssd_cache_mounted,
         reconstruct_write_sequentially=True)
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


def emit_modellake_events_once() -> int:
    """Run the seal -> registry -> fingerprint -> role -> WorkUnit consumer
    once, and persist its receipt. Returns the number of specimens for which
    a first-sight seal was emitted this run.

    Imported lazily: a missing or broken sidecar module must not stop this
    watcher from starting, and must not touch the download loop it takes no
    part in. This does no network I/O; it reads lake manifests already on
    disk and this watcher's own JSONL tail.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.future import modellake_events as me
    from tools.future._common import write_receipt

    doc = me.build()
    write_receipt(me.RECEIPT_NAME, doc, me.RECORDED_BY)
    return int(doc["n_emitted_specimens"])


def maybe_emit_modellake_events(loop_started: float, last_events_emit: float) -> float:
    """Gate emit_modellake_events_once() to MODELLAKE_EVENTS_INTERVAL_SECONDS.

    Never raises: a broken sidecar module is logged and retried next
    interval, not once per poll tick, and never crashes the watcher that
    is actively managing the two live downloads. Returns the timestamp to
    remember as last_events_emit for the next call.
    """
    if last_events_emit and loop_started - last_events_emit < MODELLAKE_EVENTS_INTERVAL_SECONDS:
        return last_events_emit

    def sweep() -> None:
        emit("modellake_events_run", n_new_seal_specimens=emit_modellake_events_once())

    run_detached("modellake_events", sweep)
    return loop_started


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
         recovery_refresh_seconds=RECOVERY_REFRESH_SECONDS,
         rate_based_refresh=RATE_BASED_REFRESH_ENABLED,
         max_download_jobs=MAX_DOWNLOAD_JOBS, max_workers=MAX_WORKERS,
         ssd_xet_cache=str(SSD_XET_CACHE),
         ssd_xet_image_capacity_bytes=SSD_XET_IMAGE_CAPACITY_BYTES,
         ssd_free_floor_bytes=SSD_FREE_FLOOR_BYTES,
         ssd_free_guard_bytes=SSD_FREE_GUARD_BYTES,
         ssd_xet_chunk_cache_target_bytes=SSD_XET_CHUNK_CACHE_TARGET_BYTES,
         reconcile_interval_seconds=RECONCILE_INTERVAL_SECONDS,
         reconstruct_write_sequentially=True)
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
    last_events_emit = 0.0
    last_reconcile = 0.0
    last_ssd_cache_bytes = None
    last_ssd_cache_mounted = False
    notified_ssd_cache_policy = False
    last_refresh: dict[str, float] = {}
    low_rx_since: dict[str, float] = {}
    rate_rearmed: set[str] = set()
    refresh_requested: set[str] = set()
    last_idle_notice: float = 0.0
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
            if complete(item, files, sizes):
                _promote_and_report(tag, str(item["destination"]), expected)
                continue
            if loop_started < retry_after.get(tag, 0):
                continue
            log_path = DOWNLOAD_DIR / f"watch-{tag}-{datetime.now().strftime('%Y%m%dT%H%M%S%z')}.log"
            # A refresh that signalled an ADOPTED transfer never reaches the
            # children-reap loop, so its tag would otherwise stay in
            # refresh_requested forever and a later genuine crash would be
            # graded as intentional. Relaunching is where the request is spent.
            refresh_requested.discard(tag)
            children[tag] = launch(item, files, log_path)
            last_job_bytes[tag] = durable_bytes(item, files, sizes)
            last_progress[tag] = loop_started
            started_this_loop.add(tag)

        # Admit queued specimens even while the remaining P0 giant runs. The
        # four-job cap and conservative storage reservation still apply, and
        # QUEUE order is deliberately smallest selected manifest first.
        # A job this watcher launched is in `children` AND is found again by
        # the pid scan that fills `active_tags`. Summing them double-counted
        # every live download, so with MAX_DOWNLOAD_JOBS=2 a single transfer
        # saturated the cap and the queue below was never reached. Lines 1311
        # and 1410 already union these two sets; this was the outlier.
        active_count = len(active_tags | set(children))
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
                    _promote_and_report(tag, str(item["destination"]), expected)
                    continue
                # A queued job may have an old partial destination. Reserve
                # the full selected manifest until an exact post-exit check
                # proves otherwise; this is intentionally conservative.
                # Reserving the FULL manifest for a job that is nearly done
                # is not conservatism, it is a deadlock: Inkling-Small sat at
                # 515.9 of 531.9 GB needing 16 GB, and every admission pass
                # emitted admission_blocked_storage because it reserved 531.9
                # against a 519 GB drive. A model can then never finish on a
                # disk smaller than its own total size, however little is left.
                # durable_bytes() is the same exact-manifest accounting the
                # stall detector already trusts, so use it here too and keep
                # the scratch/uncertainty margins on the REMAINING bytes.
                present = durable_bytes(item, files, sizes)
                remaining = max(0, expected - present)
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
                refresh_requested.discard(tag)
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
            last_ssd_cache_mounted = ssd_xet_mounted()
            last_ssd_cache_bytes = (
                tree_bytes(SSD_XET_CACHE) if last_ssd_cache_mounted else None)
            last_ssd_free = internal_ssd_free_bytes()
            cache_policy_ok = (
                last_ssd_free is not None
                and last_ssd_free > SSD_FREE_FLOOR_BYTES + SSD_FREE_GUARD_BYTES
                and last_ssd_cache_bytes is not None
                and last_ssd_cache_bytes < SSD_XET_CHUNK_CACHE_TARGET_BYTES)
            if not cache_policy_ok:
                if not notified_ssd_cache_policy:
                    notified_ssd_cache_policy = True
                    emit("ssd_xet_cache_policy_blocked",
                         cache_bytes=last_ssd_cache_bytes,
                         internal_free_bytes=last_ssd_free,
                         free_floor_bytes=SSD_FREE_FLOOR_BYTES,
                         free_guard_bytes=SSD_FREE_GUARD_BYTES,
                         chunk_cache_target_bytes=SSD_XET_CHUNK_CACHE_TARGET_BYTES)
            else:
                notified_ssd_cache_policy = False
            emit("watcher_sample", free_bytes=free, p0_done=p0_done,
                 active_jobs=sorted(active_tags | set(children)),
                 active_remaining_bytes=active_remaining, states=states,
                 ssd_xet_cache=str(SSD_XET_CACHE),
                 ssd_xet_cache_bytes=last_ssd_cache_bytes,
                 ssd_xet_image_capacity_bytes=SSD_XET_IMAGE_CAPACITY_BYTES,
                 ssd_xet_chunk_cache_target_bytes=SSD_XET_CHUNK_CACHE_TARGET_BYTES,
                 ssd_free_bytes=last_ssd_free,
                 ssd_free_floor_bytes=SSD_FREE_FLOOR_BYTES,
                 ssd_free_guard_bytes=SSD_FREE_GUARD_BYTES,
                 ssd_free_above_floor_bytes=(
                     max(0, last_ssd_free - SSD_FREE_FLOOR_BYTES)
                     if last_ssd_free is not None else None),
                 ssd_xet_cache_policy_ok=cache_policy_ok,
                 ssd_xet_cache_mounted=last_ssd_cache_mounted,
                 reconstruct_write_sequentially=True)
            last_state_emit = loop_started

        last_events_emit = maybe_emit_modellake_events(loop_started, last_events_emit)
        last_reconcile = maybe_reconcile(loop_started, last_reconcile)

        if args.once:
            return 0

        # Rest only when nothing could be started: no transfer alive, nothing
        # launched this pass. A pending retry_after cuts the wait short, so a
        # backoff still fires on time. Waking is a full rescan and re-admit,
        # which is the "re-arm" -- it repeats until the manifest is satisfied.
        idle = not (active_tags or children or started_this_loop)
        sleep_for = max(0.05, args.poll_secs - (time.monotonic() - loop_started))
        if idle:
            pending = [t for t in retry_after.values() if t > loop_started]
            wait = IDLE_REARM_SECONDS
            if pending:
                wait = min(wait, max(1.0, min(pending) - loop_started))
            if wait > sleep_for:
                if loop_started - last_idle_notice >= IDLE_REARM_SECONDS:
                    last_idle_notice = loop_started
                    emit("idle_rearm_wait", seconds=round(wait, 1),
                         next_retry_in=(round(min(pending) - loop_started, 1)
                                        if pending else None))
                sleep_for = wait
        time.sleep(sleep_for)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        emit("watcher_stopped", reason="keyboard_interrupt")
        raise
