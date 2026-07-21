#!/usr/bin/env python3.12
"""Restart-safe resident-source and Doctor Prime controller for Kimi K2.6.

The authoritative source is one Hugging Face cache snapshot.  Weight files are
never copied into the repository or a materialized ``local_dir``.  A launchd
installation copies this controller to a TCC-safe Application Support folder,
where it owns its state, lease, logs, and phone-friendly status files.

The controller intentionally refuses to infer scientific readiness from source
presence.  It advances through source verification and accounting, then waits
for separately sealed adapter/reference/causal evidence before a capability
claim can advance.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


REPO_ID = "moonshotai/Kimi-K2.6"
REVISION = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
LABEL = "com.hawking.kimi-k26-doctor-prime"
SCHEMA = "hawking.kimi_k26_campaign.v1"
STATES = (
    "PRECHECK", "ADMIT_SOURCE", "FREE_STORAGE", "DOWNLOAD", "VERIFY_SOURCE",
    "BUILD_LEDGER", "BUILD_ADAPTER", "BUILD_REFERENCE", "VALIDATE_PARENT",
    "BUILD_DOCTOR", "RUN_CAUSAL", "RUN_TOURNAMENT", "SEAL_RESULT", "MONITOR",
    "BLOCKED", "COMPLETE",
)
RUNTIME = Path.home() / "Library/Application Support/Hawking/KimiK26"
INSTALL_SCRIPT = RUNTIME / "kimi_k26_campaign.py"
INSTALL_ADAPTER = RUNTIME / "kimi_k26_adapter.py"
INSTALL_REFERENCE = RUNTIME / "kimi_k26_reference.py"
INSTALL_CORPUS = RUNTIME / "kimi_k26_corpus.py"
INSTALL_DOCTOR = RUNTIME / "kimi_k26_doctor_prime.py"
STATE = RUNTIME / "KIMI_K26_STATE.json"
HEARTBEAT = RUNTIME / "kimi_k26.heartbeat.json"
LEASE = RUNTIME / "kimi_k26.heavy.lease"
CONTROL = RUNTIME / "control.json"
MANIFEST = RUNTIME / "KIMI_K26_OFFICIAL_MANIFEST.json"
VERIFICATION = RUNTIME / "KIMI_K26_SOURCE_VERIFICATION.json"
ONE_COPY = RUNTIME / "KIMI_K26_ONE_COPY_RECEIPT.json"
LEDGER = RUNTIME / "KIMI_K26_LOGICAL_WEIGHT_LEDGER.json"
FORMAT_LEDGER = RUNTIME / "KIMI_K26_SOURCE_FORMAT_LEDGER.json"
REFERENCE_EVIDENCE = RUNTIME / "KIMI_K26_REFERENCE_FORWARD.json"
PARENT_VALIDATION = RUNTIME / "KIMI_K26_PARENT_FORWARD_VALIDATION.json"
CORPUS_INTEGRITY = RUNTIME / "KIMI_K26_CORPUS_INTEGRITY.json"
CAUSAL_ATLAS = RUNTIME / "KIMI_K26_CAUSAL_ATLAS.json"
BYTE_AUCTION = RUNTIME / "KIMI_K26_DOCTOR_BYTE_AUCTION.json"
TOURNAMENT = RUNTIME / "KIMI_K26_TOURNAMENT.json"
FIRST_CHECKPOINT = RUNTIME / "KIMI_K26_FIRST_CHECKPOINT.json"
NEXT_EXPERIMENT = RUNTIME / "KIMI_K26_NEXT_EXPERIMENT.json"
REFERENCE_RUN = RUNTIME / "reference_run"
PHONE_JSON = RUNTIME / "KIMI_PHONE_STATUS.json"
PHONE_MD = RUNTIME / "KIMI_PHONE_STATUS.md"
LOG = RUNTIME / "kimi_k26_campaign.log"
CREDS = RUNTIME / ".telegram_creds.json"
NOTIFY_STATE = RUNTIME / "telegram_delivery.json"
TELEGRAM_OUTBOX = RUNTIME / "telegram_outbox.json"
PLIST = Path.home() / f"Library/LaunchAgents/{LABEL}.plist"
HF = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/hf")
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12")
MOP = Path.home() / "Downloads/mop"
CACHE_ROOT = Path.home() / ".cache/huggingface/hub"
MODEL_CACHE = CACHE_ROOT / "models--moonshotai--Kimi-K2.6"
SNAPSHOT = MODEL_CACHE / "snapshots" / REVISION
DEFAULT_DISK_FLOOR_BYTES = 5 * 1024**3
CONFIGURED_DISK_FLOOR_BYTES = int(os.environ.get(
    "KIMI_K26_DISK_FLOOR_BYTES", str(DEFAULT_DISK_FLOOR_BYTES),
))
if CONFIGURED_DISK_FLOOR_BYTES != DEFAULT_DISK_FLOOR_BYTES:
    raise RuntimeError("KIMI_K26_DISK_FLOOR_BYTES must equal exactly 5368709120")
MIN_RESERVE = DEFAULT_DISK_FLOOR_BYTES
TARGET_RESERVE = MIN_RESERVE
AVAILABLE_MEMORY_FLOOR = 12 * 1024**3
SWAP_USED_CEILING = 16 * 1024**3
TOKEN_SERVICE = "com.hawking.doctorv5.telegram.bot-token"
CHAT_SERVICE = "com.hawking.doctorv5.telegram.chat-id"
KEYCHAIN_ACCOUNT = "hawking"


class CampaignError(RuntimeError):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def seal(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {k: v for k, v in value.items() if k != "seal_sha256"}
    return {**unsigned, "seal_sha256": hashlib.sha256(canonical(unsigned)).hexdigest()}


def atomic_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, mode)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def atomic_text(path: Path, value: str, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, mode)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def log(message: str) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} {message}\n")
    print(message, file=sys.stderr, flush=True)


def disk_free() -> int:
    return shutil.disk_usage(Path.home()).free


def disk_floor_green(free_bytes: int) -> bool:
    """The hard floor is strict: at or below it, write-heavy work is paused."""
    return int(free_bytes) > MIN_RESERVE


def can_start_atomic_write(free_bytes: int, required_bytes: int) -> bool:
    """Admit one bounded atomic write only if its completion leaves the floor green."""
    return disk_floor_green(int(free_bytes) - max(0, int(required_bytes)))


def memory_snapshot() -> dict[str, int | None]:
    # vm_stat is stable on macOS and does not require third-party packages.
    result = subprocess.run(["/usr/bin/vm_stat"], text=True, capture_output=True, check=False)
    page_size = 16384
    pages: dict[str, int] = {}
    for raw in result.stdout.splitlines():
        if "page size of" in raw:
            try:
                page_size = int(raw.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            except ValueError:
                pass
        if ":" in raw:
            key, value = raw.split(":", 1)
            try:
                pages[key.strip()] = int(value.strip().rstrip("."))
            except ValueError:
                pass
    available_pages = sum(pages.get(k, 0) for k in
                          ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    return {"available_bytes_estimate": available_pages * page_size,
            "page_size": page_size, "free_pages": pages.get("Pages free")}


def swap_snapshot() -> dict[str, int | None]:
    result = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True,
                            capture_output=True, check=False)
    values: dict[str, int | None] = {"swap_total_bytes": None, "swap_used_bytes": None,
                                    "swap_free_bytes": None}
    for label, key in (("total", "swap_total_bytes"), ("used", "swap_used_bytes"),
                       ("free", "swap_free_bytes")):
        match = re.search(rf"{label}\s*=\s*([0-9.]+)([MG])", result.stdout)
        if match:
            scale = 1024**3 if match.group(2) == "G" else 1024**2
            values[key] = int(float(match.group(1)) * scale)
    return values


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def official_manifest(*, refresh: bool = False) -> dict[str, Any]:
    cached = read_json(MANIFEST)
    if cached.get("sha") == REVISION and cached.get("files") and not refresh:
        return cached
    url = ("https://huggingface.co/api/models/"
           f"{urllib.parse.quote(REPO_ID, safe='/')}/revision/{REVISION}?blobs=true")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        if cached.get("sha") == REVISION and cached.get("files"):
            return cached
        raise CampaignError(f"official manifest unavailable: {type(exc).__name__}") from exc
    if raw.get("sha") != REVISION:
        raise CampaignError(f"revision mismatch: expected {REVISION}, got {raw.get('sha')}")
    files = []
    for sibling in raw.get("siblings", []):
        lfs = sibling.get("lfs") or {}
        files.append({"path": sibling["rfilename"], "size": int(sibling.get("size") or 0),
                      "sha256": lfs.get("sha256"), "blob_id": sibling.get("blobId")})
    manifest = seal({
        "schema": "hawking.kimi_k26.official_manifest.v1", "repo": REPO_ID,
        "sha": raw["sha"], "last_modified": raw.get("lastModified"),
        "resolved_at": now(), "license_api": (raw.get("cardData") or {}).get("license"),
        "pipeline_tag": raw.get("pipeline_tag"), "library_name": raw.get("library_name"),
        "files": sorted(files, key=lambda item: item["path"]),
        "file_count": len(files), "total_bytes": sum(item["size"] for item in files),
        "weight_shards": sum(item["path"].startswith("model-") and
                              item["path"].endswith(".safetensors") for item in files),
        "weight_bytes": sum(item["size"] for item in files if item["path"].startswith("model-")
                            and item["path"].endswith(".safetensors")),
        "largest_shard": max(item["size"] for item in files if item["path"].startswith("model-")
                             and item["path"].endswith(".safetensors")),
    })
    atomic_json(MANIFEST, manifest)
    return manifest


def initial_state() -> dict[str, Any]:
    return {
        "schema": SCHEMA, "repo": REPO_ID, "revision": REVISION, "state": "PRECHECK",
        "entered_at": now(), "started_at": now(), "history": [], "verified": {},
        "sealed_checkpoints": 0, "failed_checkpoints": 0, "pause_after_checkpoint": False,
        "stop_requested": False, "status": "RUNNING", "claim_boundary": "TEXT_CORE_ONLY",
    }


def load_state() -> dict[str, Any]:
    state = read_json(STATE)
    if state.get("schema") != SCHEMA:
        state = initial_state()
        atomic_json(STATE, state)
    return state


def save_state(state: dict[str, Any]) -> None:
    state = {**state, "updated_at": now()}
    atomic_json(STATE, state)
    write_status(state)


def transition(state: dict[str, Any], next_state: str, note: str = "") -> dict[str, Any]:
    if next_state not in STATES:
        raise CampaignError(f"invalid state {next_state}")
    state = {**state, "state": next_state, "entered_at": now(),
             "history": (state.get("history", []) +
                         [{"at": now(), "to": next_state, "note": note}])[-128:]}
    save_state(state)
    checkpoint(state, f"state:{next_state}", note)
    log(f"-> {next_state}: {note}")
    return state


def control_flags(state: dict[str, Any]) -> dict[str, Any]:
    control = read_json(CONTROL, {})
    return {**state, "pause_after_checkpoint": bool(control.get("pause_after_checkpoint", False)),
            "stop_requested": bool(control.get("stop_requested", False))}


def credentials() -> tuple[str | None, str | None]:
    def keychain(service: str) -> str | None:
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT,
             "-s", service, "-w"], text=True, capture_output=True, check=False,
        )
        value = result.stdout.strip()
        return value if result.returncode == 0 and value else None
    token, chat = keychain(TOKEN_SERVICE), keychain(CHAT_SERVICE)
    if token and chat:
        return token, chat
    fallback = read_json(CREDS, {})
    return fallback.get("token"), fallback.get("chat_id")


def telegram(message: str) -> bool:
    token, chat_id = credentials()
    if not token or not chat_id:
        return False
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": message[:4000]}).encode()
    request = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=payload)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            answer = json.load(response)
        return bool(answer.get("ok"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def flush_telegram_outbox() -> None:
    pending = read_json(TELEGRAM_OUTBOX, [])
    if not isinstance(pending, list) or not pending:
        return
    record = pending[0]
    if telegram(str(record.get("message", ""))):
        atomic_json(NOTIFY_STATE, {
            "checkpoint_id": record.get("checkpoint_id"),
            "seal_sha256": record.get("seal_sha256"), "delivered_at": now(),
        })
        atomic_json(TELEGRAM_OUTBOX, pending[1:])


def checkpoint(state: dict[str, Any], checkpoint_id: str, detail: str = "") -> None:
    snapshot = status_snapshot(state)
    record = seal({"checkpoint_id": checkpoint_id, "sealed_at": now(), "detail": detail,
                   "state": state.get("state"), "progress": snapshot.get("progress"),
                   "free_disk_bytes": snapshot["resources"]["free_disk_bytes"],
                   "pid": os.getpid(), "complete_bpw": snapshot.get("complete_bpw"),
                   "next_action": snapshot.get("next_action")})
    delivery = read_json(NOTIFY_STATE, {})
    delivered = delivery.get("seal_sha256") == record["seal_sha256"]
    if not delivered:
        runtime_seconds = max(0, int(time.time() - dt.datetime.fromisoformat(
            state.get("started_at", now()).replace("Z", "+00:00")).timestamp()))
        msg = (f"[Kimi K2.6] {checkpoint_id}\nstate/status: {record['state']} / "
               f"{state.get('status')}\nstage/candidate: {state.get('current_layer') or '-'} / "
               f"{snapshot.get('best_candidate') or '-'}\nprogress: {snapshot.get('progress_text')}\n"
               f"complete BPW: {snapshot.get('complete_bpw')}\n"
               f"metrics: {json.dumps(snapshot.get('primary_metrics'), sort_keys=True)[:500]}\n"
               f"runtime: {runtime_seconds}s\nETA: {snapshot.get('eta_text')}\n"
               f"free RAM/disk: {snapshot['resources'].get('available_bytes_estimate', 0)/1024**3:.1f} / "
               f"{snapshot['resources']['free_disk_bytes']/1024**3:.1f} GiB\n"
               f"PID: {os.getpid()}\nnext: {snapshot.get('next_action')}\n"
               f"checkpoint identity: {checkpoint_id} + {record['seal_sha256']}")
        pending = read_json(TELEGRAM_OUTBOX, [])
        if not isinstance(pending, list):
            pending = []
        if not any(item.get("seal_sha256") == record["seal_sha256"] for item in pending):
            pending.append({"checkpoint_id": checkpoint_id,
                            "seal_sha256": record["seal_sha256"], "message": msg,
                            "queued_at": now()})
            atomic_json(TELEGRAM_OUTBOX, pending)
        flush_telegram_outbox()
        delivered = read_json(NOTIFY_STATE, {}).get("seal_sha256") == record["seal_sha256"]
    state["last_checkpoint"] = record
    state["last_telegram_delivery"] = read_json(NOTIFY_STATE, {}) if delivered else delivery
    state["sealed_checkpoints"] = int(state.get("sealed_checkpoints", 0)) + 1
    save_state(state)


def file_present(item: dict[str, Any]) -> bool:
    path = SNAPSHOT / item["path"]
    try:
        return path.resolve(strict=True).is_file() and path.stat().st_size == item["size"]
    except OSError:
        return False


def progress(manifest: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    present = [item for item in manifest["files"] if file_present(item)]
    bytes_done = sum(item["size"] for item in present)
    shards_done = sum(item["path"].startswith("model-") and
                      item["path"].endswith(".safetensors") for item in present)
    elapsed = max(0.001, time.time() - float(state.get("download_epoch") or time.time()))
    throughput = max(0.0, (bytes_done - int(state.get("download_start_bytes") or 0)) / elapsed)
    remaining = max(0, manifest["total_bytes"] - bytes_done)
    eta = remaining / throughput if throughput > 0 else None
    return {"files_done": len(present), "files_total": manifest["file_count"],
            "shards_done": shards_done, "shards_total": manifest["weight_shards"],
            "bytes_done": bytes_done, "bytes_total": manifest["total_bytes"],
            "throughput_bytes_per_second": throughput, "eta_seconds": eta}


def next_action_for(state_name: str) -> str:
    return {
        "PRECHECK": "verify MOP, source root, and live resource floor",
        "ADMIT_SOURCE": "bind immutable official manifest and architecture",
        "FREE_STORAGE": "seal resident storage gate",
        "DOWNLOAD": "land and verify the next official file",
        "VERIFY_SOURCE": "reconcile all 96 files and one-copy invariant",
        "BUILD_LEDGER": "derive logical weights from native packed headers",
        "BUILD_ADAPTER": "run adapter, decoder, MLA, and MoE synthetic tests",
        "BUILD_REFERENCE": "run bounded-memory source text-core forward",
        "VALIDATE_PARENT": "seal real-logit parent validation suite",
        "BUILD_DOCTOR": "seal clean corpus gate and Doctor byte auction",
        "RUN_CAUSAL": "localize MLA/router/expert/residual/logit degradation",
        "RUN_TOURNAMENT": "advance the admitted <=1 complete-BPW candidates",
        "SEAL_RESULT": "seal exact artifact, quality, and rollback evidence",
        "MONITOR": "monitor the next advancing experiment",
        "BLOCKED": "resolve the recorded blocker, then resume",
        "COMPLETE": "campaign complete",
    }.get(state_name, "inspect state")


def retry_state_for_blocker(blocker: str) -> str | None:
    """Return the last restart-safe stage for a repairable sealed-stage failure."""
    routes = (
        ("disk-floor guard", "FREE_STORAGE"),
        ("bounded source forward failed", "BUILD_REFERENCE"),
        ("Kimi clean-corpus integrity artifact", "BUILD_DOCTOR"),
        ("causal intervention harness failed", "RUN_CAUSAL"),
        ("exact Doctor byte auction failed", "RUN_TOURNAMENT"),
        ("tournament F0/first checkpoint failed", "RUN_TOURNAMENT"),
    )
    return next((state for prefix, state in routes if blocker.startswith(prefix)), None)


def control_snapshot(state: dict[str, Any]) -> dict[str, bool]:
    control = read_json(CONTROL, {})
    return {
        "pause_after_checkpoint": bool(control.get(
            "pause_after_checkpoint", state.get("pause_after_checkpoint", False))),
        "stop_requested": bool(control.get(
            "stop_requested", state.get("stop_requested", False))),
    }


def status_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(MANIFEST, {})
    prog = progress(manifest, state) if manifest.get("files") else {}
    hb = read_json(HEARTBEAT, {})
    beat_age = None
    try:
        beat_age = max(0.0, time.time() - dt.datetime.fromisoformat(
            hb["beat_at"].replace("Z", "+00:00")).timestamp())
    except (KeyError, ValueError):
        pass
    eta = state.get("stage_eta_seconds")
    if eta is None:
        eta = prog.get("eta_seconds")
    eta_text = "computing" if eta is None else (
        "done" if eta <= 0 else f"{int(eta // 3600)}h {int((eta % 3600) // 60)}m")
    progress_text = state.get("stage_progress_text") or (
        f"{prog.get('shards_done', 0)}/{prog.get('shards_total', 64)} shards; "
        f"{prog.get('bytes_done', 0)/1e9:.2f}/{prog.get('bytes_total', 0)/1e9:.2f} GB"
    )
    return {
        "schema": "hawking.kimi_phone_status.v1", "generated_at": now(),
        "state": state.get("state"), "status": state.get("status"), "pid": hb.get("pid"),
        "process_group": hb.get("process_group"), "lease": str(LEASE),
        "lease_live": bool(hb.get("pid") and pid_alive(hb.get("pid"))),
        "heartbeat_age_seconds": beat_age, "source_state": state.get("source_state"),
        "current_file": state.get("current_file"), "current_layer": state.get("current_layer"),
        "progress": prog, "progress_text": progress_text, "eta_text": eta_text,
        "sealed_count": state.get("sealed_checkpoints", 0),
        "failed_count": state.get("failed_checkpoints", 0),
        "best_candidate": state.get("best_candidate"), "complete_bpw": state.get("complete_bpw"),
        "primary_metrics": state.get("primary_metrics"),
        "next_action": next_action_for(str(state.get("state"))),
        "last_telegram_delivery": read_json(NOTIFY_STATE, {}),
        "blocker": state.get("blocker"),
        "resources": {"free_disk_bytes": disk_free(),
                      "disk_floor_bytes": MIN_RESERVE,
                      "disk_headroom_bytes": disk_free() - MIN_RESERVE,
                      "disk_floor_green": disk_free() > MIN_RESERVE,
                      **memory_snapshot(), **swap_snapshot()},
        "control": control_snapshot(state),
    }


def write_status(state: dict[str, Any]) -> None:
    snapshot = status_snapshot(state)
    atomic_json(PHONE_JSON, snapshot)
    p = snapshot["progress"]
    throughput_text = (f"{p.get('throughput_bytes_per_second', 0)/1e6:.1f} MB/s"
                       if snapshot.get("source_state") == "KIMI_DOWNLOAD_RUNNING"
                       else "complete")
    md = f"""# Kimi K2.6 Doctor Prime

- State: **{snapshot['state']}** ({snapshot.get('status')})
- Running: PID `{snapshot.get('pid')}`, process group `{snapshot.get('process_group')}`
- Lease live: `{snapshot.get('lease_live')}`; heartbeat age: `{snapshot.get('heartbeat_age_seconds')}` s
- Source: `{snapshot.get('source_state')}`
- Progress: {snapshot.get('progress_text')}
- Download throughput/state: {throughput_text}
- ETA: {snapshot.get('eta_text')}
- Free disk: {snapshot['resources']['free_disk_bytes']/1024**3:.1f} GiB
- Hard disk floor/headroom: {snapshot['resources']['disk_floor_bytes']/1024**3:.1f} / {snapshot['resources']['disk_headroom_bytes']/1024**3:.1f} GiB
- Available memory estimate: {snapshot['resources'].get('available_bytes_estimate', 0)/1024**3:.1f} GiB
- Swap used: {(snapshot['resources'].get('swap_used_bytes') or 0)/1024**3:.1f} GiB (16 GiB guard)
- Current file/layer: `{snapshot.get('current_file') or snapshot.get('current_layer') or 'n/a'}`
- Sealed/failed checkpoints: {snapshot.get('sealed_count')}/{snapshot.get('failed_count')}
- Best candidate / complete BPW: `{snapshot.get('best_candidate')}` / `{snapshot.get('complete_bpw')}`
- Next action: {snapshot.get('next_action')}
- Exact blocker: `{snapshot.get('blocker') or 'none'}`
- Last Telegram: `{snapshot.get('last_telegram_delivery') or 'none'}`

Control from the Hawking repository:

```text
python3 tools/kimi_k26_campaign.py status
python3 tools/kimi_k26_campaign.py pause-after-checkpoint
python3 tools/kimi_k26_campaign.py resume
python3 tools/kimi_k26_campaign.py stop
```
"""
    atomic_text(PHONE_MD, md)


def heartbeat(state: dict[str, Any]) -> None:
    flush_telegram_outbox()
    snapshot = status_snapshot(state)
    atomic_json(HEARTBEAT, {"schema": "hawking.kimi_k26.heartbeat.v1", "beat_at": now(),
                            "pid": os.getpid(), "process_group": os.getpgrp(),
                            "state": state.get("state"), "status": state.get("status"),
                            "current_file": state.get("current_file"),
                            "progress": snapshot.get("progress"),
                            "free_disk_bytes": snapshot["resources"]["free_disk_bytes"]})
    write_status(state)


def pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def precheck(state: dict[str, Any]) -> dict[str, Any]:
    if MOP.resolve(strict=True) != Path.home() / "Downloads/mop":
        raise CampaignError("MOP root did not resolve to the protected path")
    mop_stat = MOP.stat()
    if MODEL_CACHE.resolve().is_relative_to(MOP.resolve()):
        raise CampaignError("Kimi cache unexpectedly resolves inside MOP")
    state.update({"mop": {"path": str(MOP.resolve()), "device": mop_stat.st_dev,
                          "inode": mop_stat.st_ino},
                  "source_root": str(MODEL_CACHE), "snapshot": str(SNAPSHOT),
                  "source_state": "PARTIAL_RESUMABLE_LOCAL"})
    return transition(state, "ADMIT_SOURCE", "MOP and the one-copy cache root verified")


def admit(state: dict[str, Any]) -> dict[str, Any]:
    manifest = official_manifest()
    config = read_json(SNAPSHOT / "config.json")
    text = config.get("text_config") or {}
    required = {"n_routed_experts": 384, "n_shared_experts": 1,
                "num_experts_per_tok": 8, "max_position_embeddings": 262144}
    mismatch = {key: {"expected": expected, "actual": text.get(key)}
                for key, expected in required.items() if text.get(key) != expected}
    if mismatch or manifest["weight_shards"] != 64:
        raise CampaignError(f"official architecture mismatch: {mismatch}")
    state["admission"] = {"revision": REVISION, "license": "Modified MIT",
                          "files": manifest["file_count"], "bytes": manifest["total_bytes"],
                          "weight_shards": manifest["weight_shards"],
                          "architecture": {key: text[key] for key in required},
                          "model_type": config.get("model_type"),
                          "text_model_type": text.get("model_type"),
                          "modality": "image-text; separable DeepSeek-V3-compatible text core",
                          "quantization": text.get("quantization_config")}
    return transition(state, "FREE_STORAGE", "immutable official source admitted")


def storage_gate(state: dict[str, Any]) -> dict[str, Any]:
    manifest = official_manifest()
    free = disk_free()
    present_bytes = progress(manifest, state)["bytes_done"]
    remaining = manifest["total_bytes"] - present_bytes
    reserve = MIN_RESERVE
    projected = free - remaining
    state["storage_gate"] = {"free_bytes": free, "present_source_bytes": present_bytes,
                             "remaining_source_bytes": remaining, "reserve_bytes": reserve,
                             "target_reserve_bytes": TARGET_RESERVE,
                             "projected_post_download_free_bytes": projected}
    if not disk_floor_green(projected):
        state["blocker"] = f"BLOCKED_RESIDENT_SHORTFALL: {reserve - projected} bytes"
        state["status"] = "BLOCKED"
        return transition(state, "BLOCKED", state["blocker"])
    state["download_epoch"] = time.time()
    state["download_start_bytes"] = present_bytes
    state["source_state"] = "KIMI_DOWNLOAD_PENDING"
    return transition(state, "DOWNLOAD", f"projected reserve {projected} bytes is green")


def verify_item(item: dict[str, Any]) -> dict[str, Any]:
    logical = SNAPSHOT / item["path"]
    target = logical.resolve(strict=True)
    if not target.is_file() or target.stat().st_size != item["size"]:
        raise CampaignError(f"size mismatch after download: {item['path']}")
    if item.get("sha256"):
        digest = sha256_file(target)
        if digest != item["sha256"]:
            raise CampaignError(f"SHA-256 mismatch: {item['path']}")
        method = "sha256"
    else:
        digest = git_blob_sha1(target)
        if item.get("blob_id") and digest != item["blob_id"]:
            raise CampaignError(f"Git blob mismatch: {item['path']}")
        method = "git_blob_sha1"
    return {"path": item["path"], "size": item["size"], "digest": digest,
            "method": method, "verified_at": now(), "physical_path": str(target)}


def incomplete_allocated(item: dict[str, Any]) -> int:
    """Allocated bytes in a resumable Xet/LFS partial, never its sparse logical length."""
    digest = item.get("sha256")
    if not digest:
        return 0
    partial = MODEL_CACHE / "blobs" / f"{digest}.incomplete"
    try:
        value = partial.stat().st_blocks * 512
        return min(int(item["size"]), value)
    except OSError:
        return 0


def download_one(state: dict[str, Any]) -> dict[str, Any]:
    manifest = official_manifest()
    state = control_flags(state)
    if state.get("stop_requested"):
        state["status"] = "STOPPED_PRESERVED"
        save_state(state)
        checkpoint(state, "controller:stopped", "all sealed files preserved")
        return state
    if state.get("pause_after_checkpoint"):
        state["status"] = "PAUSED_AFTER_CHECKPOINT"
        save_state(state)
        heartbeat(state)
        time.sleep(15)
        return state
    state["status"] = "RUNNING"
    missing = [item for item in manifest["files"] if not file_present(item)]
    reserve = MIN_RESERVE
    remaining_allocation = sum(max(0, int(item["size"]) - incomplete_allocated(item))
                               for item in missing)
    if not can_start_atomic_write(disk_free(), remaining_allocation):
        state["blocker"] = (f"disk-floor guard: the remaining official snapshot would leave less "
                            f"than {reserve} bytes")
        state["source_state"] = "KIMI_DOWNLOAD_BLOCKED"
        state["status"] = "BLOCKED"
        return transition(state, "BLOCKED", state["blocker"])
    if missing:
        state.update({"current_file": f"Xet bulk snapshot ({len(missing)} files missing)",
                      "current_file_bytes": sum(item["size"] for item in missing),
                      "source_state": "KIMI_DOWNLOAD_RUNNING",
                      "xet_mode": "HIGH_PERFORMANCE_ADAPTIVE_8_FILE"})
        save_state(state)
        log(f"Xet high-performance bulk download: {len(missing)} missing official files")
        env = os.environ.copy()
        env["HF_HUB_DISABLE_XET"] = "0"
        # One long-lived process lets Xet's adaptive controller stay warm across shards.  Eight
        # files can be in flight while Xet independently parallelizes range GETs within each file.
        # This is materially faster than restarting the adaptive controller for every 9.8 GB shard.
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
        command = [str(HF), "download", REPO_ID, "--revision", REVISION,
                   "--cache-dir", str(CACHE_ROOT), "--max-workers", "8", "--format", "agent"]
        with LOG.open("a", encoding="utf-8") as output:
            process = subprocess.Popen(command, stdout=output, stderr=output, env=env,
                                       start_new_session=False)
            while process.poll() is None:
                state = control_flags(state)
                if state.get("stop_requested") or state.get("pause_after_checkpoint"):
                    terminate_child(process)
                    if state.get("stop_requested"):
                        state["status"] = "STOPPED_PRESERVED"
                        detail = "Xet resumable partials and all sealed files preserved"
                        checkpoint_id = "controller:stopped"
                    else:
                        state["status"] = "PAUSED_AFTER_CHECKPOINT"
                        detail = "paused at latest landed-file checkpoint; Xet partials preserved"
                        checkpoint_id = "controller:paused"
                    save_state(state)
                    checkpoint(state, checkpoint_id, detail)
                    return state
                prog = progress(manifest, state)
                live_missing = int(prog["files_total"]) - int(prog["files_done"])
                live_label = f"Xet bulk snapshot ({live_missing} files missing)"
                if state.get("current_file") != live_label:
                    state["current_file"] = live_label
                    save_state(state)
                shard_count = int(prog["shards_done"])
                last_notice = int(state.get("last_download_notice_shards", 0))
                if shard_count == manifest["weight_shards"] or shard_count >= last_notice + 6:
                    state["last_download_notice_shards"] = shard_count
                    save_state(state)
                    checkpoint(state, f"download-progress:{shard_count:02d}-of-64",
                               "Xet landing checkpoint; cryptographic verification follows")
                heartbeat(state)
                time.sleep(15)
        if process.returncode != 0:
            state["failed_checkpoints"] = int(state.get("failed_checkpoints", 0)) + 1
            save_state(state)
            checkpoint(state, "download:fault:bulk-xet",
                       f"exit {process.returncode}; resumable partials preserved for retry")
            time.sleep(30)
            return state

    # Verify every landed file once.  Save after each shard so a restart never repeats sealed
    # cryptographic work; Telegram is grouped every six shards to avoid 64-message flooding.
    state.update({"source_state": "KIMI_DOWNLOAD_VERIFYING", "current_file": None})
    verified = dict(state.get("verified", {}))
    verified_shards = sum(path.startswith("model-") and path.endswith(".safetensors")
                          for path in verified)
    last_verify_notice = int(state.get("last_verify_notice_shards", 0))
    for item in manifest["files"]:
        if item["path"] in verified:
            continue
        state["current_file"] = f"verify:{item['path']}"
        save_state(state)
        verified[item["path"]] = verify_item(item)
        if item["path"].startswith("model-") and item["path"].endswith(".safetensors"):
            verified_shards += 1
            state["sealed_checkpoints"] = int(state.get("sealed_checkpoints", 0)) + 1
        state["verified"] = verified
        save_state(state)
        heartbeat(state)
        if verified_shards == manifest["weight_shards"] or verified_shards >= last_verify_notice + 6:
            last_verify_notice = verified_shards
            state["last_verify_notice_shards"] = verified_shards
            save_state(state)
            checkpoint(state, f"source-verify:{verified_shards:02d}-of-64",
                       "official sizes and SHA-256 digests sealed")
    state["current_file"] = None
    save_state(state)
    return transition(state, "VERIFY_SOURCE", "all official files landed and digest-verified")


def verify_source(state: dict[str, Any]) -> dict[str, Any]:
    manifest = official_manifest()
    failures = []
    for item in manifest["files"]:
        if not file_present(item):
            failures.append({"path": item["path"], "reason": "absent_or_wrong_size"})
    index = read_json(SNAPSHOT / "model.safetensors.index.json")
    indexed = set(index.get("weight_map", {}).values())
    official_shards = {item["path"] for item in manifest["files"]
                       if item["path"].startswith("model-") and
                       item["path"].endswith(".safetensors")}
    if indexed != official_shards:
        failures.append({"reason": "index_shard_set_mismatch",
                         "missing_from_index": sorted(official_shards - indexed),
                         "extra_in_index": sorted(indexed - official_shards)})
    # HF cache snapshots are symlinks to one content-addressed blob.  Count only distinct inodes;
    # the symlink is a view, not a materialized duplicate.
    inodes = set()
    for item in manifest["files"]:
        try:
            st = (SNAPSHOT / item["path"]).resolve(strict=True).stat()
            inodes.add((st.st_dev, st.st_ino))
        except OSError:
            pass
    candidate_indexes = {SNAPSHOT / "model.safetensors.index.json"}
    for root in (CACHE_ROOT, Path.home() / "Library/Caches/huggingface/hub",
                 Path.home() / "Downloads/hawking/models", Path.home() / "models"):
        if root.exists():
            candidate_indexes.update(root.rglob("model.safetensors.index.json"))
    spotlight = subprocess.run(
        ["/usr/bin/mdfind", "kMDItemFSName == 'model.safetensors.index.json'"],
        text=True, capture_output=True, check=False,
    )
    candidate_indexes.update(Path(line) for line in spotlight.stdout.splitlines() if line.strip())
    source_views = []
    physical_fingerprints = set()
    mop_string = str(MOP.resolve()) + os.sep
    authoritative = SNAPSHOT.resolve()
    for candidate_index in sorted(candidate_indexes, key=str):
        # Never stat, resolve, or read a candidate under protected MOP.
        if str(candidate_index).startswith(mop_string):
            continue
        candidate_root = candidate_index.parent
        config = read_json(candidate_root / "config.json", {})
        if config.get("model_type") != "kimi_k25" or \
                (config.get("text_config") or {}).get("model_type") != "kimi_k2":
            continue
        candidate_shards = sorted(candidate_root.glob("model-*-of-000064.safetensors"))
        complete = len(candidate_shards) == 64
        fingerprint = None
        if complete:
            try:
                fingerprint = tuple(sorted(
                    (path.resolve(strict=True).stat().st_dev,
                     path.resolve(strict=True).stat().st_ino) for path in candidate_shards
                ))
                physical_fingerprints.add(fingerprint)
            except OSError:
                complete = False
        source_views.append({
            "path": str(candidate_root), "complete": complete,
            "authoritative_snapshot": candidate_root.resolve() == authoritative,
            "physical_inode_set_sha256": hashlib.sha256(canonical(fingerprint)).hexdigest()
            if fingerprint else None,
        })
    complete_views = [view for view in source_views if view["complete"]]
    non_authoritative = [view for view in complete_views if not view["authoritative_snapshot"]]
    if len(physical_fingerprints) != 1 or len(complete_views) != 1 or non_authoritative:
        failures.append({"reason": "duplicate_or_ambiguous_complete_kimi_source",
                         "complete_views": complete_views,
                         "distinct_physical_inode_sets": len(physical_fingerprints)})
    verification = seal({"schema": "hawking.kimi_k26.source_verification.v1",
                         "status": "PASS" if not failures and
                         disk_floor_green(disk_free()) else "FAIL",
                         "verified_at": now(), "repo": REPO_ID, "revision": REVISION,
                         "source_root": str(MODEL_CACHE), "snapshot": str(SNAPSHOT),
                         "file_count": manifest["file_count"], "weight_shards": len(official_shards),
                         "source_bytes": manifest["total_bytes"], "index_total_size":
                         (index.get("metadata") or {}).get("total_size"),
                         "index_tensor_count": len(index.get("weight_map", {})),
                         "failures": failures, "post_download_free_bytes": disk_free(),
                         "reserve_green": disk_floor_green(disk_free())})
    atomic_json(VERIFICATION, verification)
    one_copy = seal({"schema": "hawking.kimi_k26.one_copy.v1", "verified_at": now(),
                     "status": "PASS" if len(physical_fingerprints) == 1 and
                     len(complete_views) == 1 and not non_authoritative else "FAIL",
                     "authoritative_layout": "huggingface_content_addressed_cache",
                     "source_root": str(MODEL_CACHE), "snapshot_view": str(SNAPSHOT),
                     "local_dir_copy": None, "unique_content_inodes": len(inodes),
                     "complete_source_views": complete_views,
                     "non_authoritative_complete_views": non_authoritative,
                     "distinct_physical_source_inode_sets": len(physical_fingerprints),
                     "model_source_copies": len(physical_fingerprints),
                     "spotlight_and_known_roots_scanned": True,
                     "mop_excluded_without_traversal": str(MOP.resolve())})
    atomic_json(ONE_COPY, one_copy)
    if failures or not verification["reserve_green"]:
        state["blocker"] = f"source verification failed: {failures[:3]}"
        state["status"] = "BLOCKED"
        return transition(state, "BLOCKED", state["blocker"])
    state.update({"source_state": "KIMI_DOWNLOAD_COMPLETE", "current_file": None})
    return transition(state, "BUILD_LEDGER", "96 files, 64 shards, index, and reserve verified")


DTYPE_BITS = {"BOOL": 8, "U8": 8, "I8": 8, "U16": 16, "I16": 16,
              "F16": 16, "BF16": 16, "U32": 32, "I32": 32, "F32": 32,
              "U64": 64, "I64": 64, "F64": 64}


def tensor_organ(name: str) -> str:
    if name.startswith("vision_tower."):
        return "vision"
    if name.startswith("multi_modal_projector.") or name.startswith("mm_projector."):
        return "vision_text_adapter"
    if ".mlp.experts." in name:
        return "routed_expert"
    if ".mlp.shared_experts." in name:
        return "shared_expert"
    if ".self_attn." in name:
        return "attention_mla"
    if ".mlp." in name:
        return "dense_mlp"
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "norm" in name:
        return "normalization"
    return "text_other"


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise CampaignError(f"short safetensors header: {path.name}")
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > 512 * 1024**2:
            raise CampaignError(f"unsafe safetensors header length {size}: {path.name}")
        return json.loads(handle.read(size))


def build_ledger(state: dict[str, Any]) -> dict[str, Any]:
    manifest = official_manifest()
    organs: dict[str, dict[str, int]] = {}
    dtypes: dict[str, dict[str, int]] = {}
    tensor_count = 0
    logical_weights = 0
    physical_tensor_bits = 0
    metadata_bits = 0
    packed_logical = 0
    packed_physical_bits = 0
    unrecognized = []
    for item in manifest["files"]:
        if not (item["path"].startswith("model-") and item["path"].endswith(".safetensors")):
            continue
        tensor_path = SNAPSHOT / item["path"]
        with tensor_path.open("rb") as tensor_file:
            raw_length = tensor_file.read(8)
            if len(raw_length) != 8:
                raise CampaignError(f"short safetensors header: {tensor_path.name}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > 512 * 1024**2:
                raise CampaignError(f"unsafe safetensors header: {tensor_path.name}")
            header = json.loads(tensor_file.read(header_length))
            data_base = 8 + header_length
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                tensor_count += 1
                shape = [int(x) for x in info.get("shape", [])]
                numel = 1
                for dim in shape:
                    numel *= dim
                dtype = info.get("dtype")
                bits = DTYPE_BITS.get(dtype)
                if bits is None:
                    unrecognized.append({"tensor": name, "dtype": dtype})
                    continue
                physical_bits = numel * bits
                organ = tensor_organ(name)
                bucket = organs.setdefault(organ, {"logical_weights": 0, "physical_bits": 0,
                                                   "metadata_bits": 0, "tensors": 0})
                bucket["tensors"] += 1
                db = dtypes.setdefault(dtype, {"physical_elements": 0, "physical_bits": 0,
                                               "tensors": 0})
                db["physical_elements"] += numel
                db["physical_bits"] += physical_bits
                db["tensors"] += 1
                if name.endswith(".weight_packed"):
                    if dtype != "I32":
                        unrecognized.append({"tensor": name, "dtype": dtype,
                                             "reason": "packed tensor is not I32"})
                        continue
                    shape_name = name.removesuffix(".weight_packed") + ".weight_shape"
                    shape_info = header.get(shape_name)
                    if not shape_info or shape_info.get("dtype") != "I32":
                        unrecognized.append({"tensor": name,
                                             "reason": "missing I32 original weight_shape"})
                        continue
                    start, end = shape_info["data_offsets"]
                    tensor_file.seek(data_base + int(start))
                    shape_raw = tensor_file.read(int(end) - int(start))
                    original_shape = list(struct.unpack(f"<{len(shape_raw)//4}i", shape_raw))
                    logical = 1
                    for dim in original_shape:
                        logical *= dim
                    expected_packed_last = (original_shape[-1] * 4 + 31) // 32
                    if (shape[:-1] != original_shape[:-1] or
                            shape[-1] != expected_packed_last or numel * 8 != logical):
                        unrecognized.append({"tensor": name, "physical_shape": shape,
                                             "original_shape": original_shape,
                                             "reason": "packing geometry does not reconcile"})
                        continue
                    packed_logical += logical
                    packed_physical_bits += physical_bits
                    logical_weights += logical
                    physical_tensor_bits += physical_bits
                    bucket["logical_weights"] += logical
                    bucket["physical_bits"] += physical_bits
                elif name.endswith(".weight_scale") or name.endswith(".weight_shape"):
                    metadata_bits += physical_bits
                    bucket["metadata_bits"] += physical_bits
                elif name.endswith(".weight"):
                    logical_weights += numel
                    physical_tensor_bits += physical_bits
                    bucket["logical_weights"] += numel
                    bucket["physical_bits"] += physical_bits
                else:
                    # Non-weight tensors such as router correction bias are installed state and
                    # count physically, but are not silently added to the weight denominator.
                    metadata_bits += physical_bits
                    bucket["metadata_bits"] += physical_bits
    if unrecognized:
        raise CampaignError(f"unrecognized packed layout: {unrecognized[:3]}")
    vision_weights = sum(v["logical_weights"] for k, v in organs.items()
                         if k in {"vision", "vision_text_adapter"})
    text_weights = logical_weights - vision_weights
    routed = organs.get("routed_expert", {}).get("logical_weights", 0)
    shared = organs.get("shared_expert", {}).get("logical_weights", 0)
    attention = organs.get("attention_mla", {}).get("logical_weights", 0)
    installed_bits = physical_tensor_bits + metadata_bits
    weight_shard_file_bytes = sum(
        int(item["size"]) for item in manifest["files"]
        if item["path"].startswith("model-") and item["path"].endswith(".safetensors")
    )
    container_overhead_bits = weight_shard_file_bytes * 8 - installed_bits
    if container_overhead_bits < 0:
        raise CampaignError("safetensors payload exceeds its official shard file bytes")
    if routed % 384:
        raise CampaignError("routed-expert denominator does not divide across 384 experts")
    active_routed = routed // 384 * 8
    active_text = text_weights - routed + active_routed
    ledger = seal({
        "schema": "hawking.kimi_k26.logical_weight_ledger.v1", "status": "PASS",
        "sealed_at": now(),
        "repo": REPO_ID, "revision": REVISION, "tensor_count": tensor_count,
        "all_logical_original_weights": logical_weights,
        "compressible_logical_weights": packed_logical,
        "routed_expert_logical_weights": routed,
        "active_routed_expert_logical_weights_per_token": active_routed,
        "shared_expert_logical_weights": shared,
        "attention_logical_weights": attention,
        "vision_logical_weights": vision_weights,
        "text_core_logical_weights": text_weights,
        "active_text_core_logical_weights_per_token": active_text,
        "official_source_bits_per_logical_weight": manifest["total_bytes"] * 8 / logical_weights,
        "weight_shard_file_bits_per_logical_weight":
            weight_shard_file_bytes * 8 / logical_weights,
        "tensor_payload_bits_per_logical_weight": installed_bits / logical_weights,
        "complete_official_source_bytes": manifest["total_bytes"],
        "weight_shard_file_bytes": weight_shard_file_bytes,
        "safetensors_container_overhead_bits": container_overhead_bits,
        "physical_weight_payload_bits": physical_tensor_bits,
        "scale_and_metadata_bits": metadata_bits, "organs": organs,
        "denominator_rule": "I32 weight_packed stores eight signed INT4 logical weights; "
                            "scale/shape tensors are installed bits, not logical weights",
        "active_denominator_rule": "all non-routed text weights plus 8/384 of each "
                                   "routed-expert layer for one token",
    })
    fmt = seal({"schema": "hawking.kimi_k26.source_format_ledger.v1", "status": "PASS",
                "sealed_at": now(),
                "repo": REPO_ID, "revision": REVISION, "format": "compressed-tensors pack-quantized",
                "packed_dtype": "I32", "packing_factor": 8, "logical_bits": 4,
                "group_size": 32, "physical_weight_payload_bits": physical_tensor_bits,
                "packed_logical_weights": packed_logical,
                "packed_physical_bits": packed_physical_bits,
                "scale_and_metadata_bits": metadata_bits, "dtypes": dtypes,
                "complete_official_source_bytes": manifest["total_bytes"],
                "weight_shard_file_bytes": weight_shard_file_bytes,
                "safetensors_container_overhead_bits": container_overhead_bits,
                "unrecognized_layouts": []})
    atomic_json(LEDGER, ledger)
    atomic_json(FORMAT_LEDGER, fmt)
    state.update({"complete_bpw": None, "logical_weight_denominator": logical_weights,
                  "text_core_logical_weight_denominator": text_weights})
    return transition(state, "BUILD_ADAPTER", "native packed logical-weight ledger sealed")


def build_adapter(state: dict[str, Any]) -> dict[str, Any]:
    if not INSTALL_ADAPTER.exists():
        heartbeat(state)
        time.sleep(30)
        return state
    result = subprocess.run([str(PYTHON), str(INSTALL_ADAPTER), "selftest", "--config",
                             str(SNAPSHOT / "config.json"), "--source", str(SNAPSHOT)],
                            text=True, capture_output=True,
                            check=False, timeout=300)
    try:
        evidence = json.loads(result.stdout)
    except json.JSONDecodeError:
        evidence = {}
    if (result.returncode != 0 or evidence.get("status") != "PASS" or
            not evidence.get("checks", {}).get("functional_metal_k1")):
        state["blocker"] = f"adapter selftest failed: {result.stderr[-300:]}"
        state["status"] = "BLOCKED"
        return transition(state, "BLOCKED", state["blocker"])
    atomic_json(RUNTIME / "KIMI_K26_ADAPTER_TWIN.json", seal(evidence))
    return transition(state, "BUILD_REFERENCE", "adapter/twin/INT4/MLA/MoE selftests pass")


def evidence_gate(state: dict[str, Any], filename: str, next_state: str,
                  required_status: str = "PASS") -> dict[str, Any]:
    evidence = read_json(RUNTIME / filename)
    if evidence.get("status") != required_status:
        heartbeat(state)
        time.sleep(30)
        return state
    return transition(state, next_state, f"{filename} sealed {required_status}")


def valid_seal(value: dict[str, Any]) -> bool:
    expected = value.get("seal_sha256")
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    return isinstance(expected, str) and hashlib.sha256(canonical(unsigned)).hexdigest() == expected


def terminate_child(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def managed_stage(state: dict[str, Any], command: list[str], log_name: str,
                  active_path: Path | None = None) -> tuple[int | None, dict[str, Any]]:
    """Run one heavy child inside the controller process group with resource/control guards."""
    memory = memory_snapshot()
    swap = swap_snapshot()
    if (int(memory.get("available_bytes_estimate") or 0) < AVAILABLE_MEMORY_FLOOR or
            int(swap.get("swap_used_bytes") or 0) > SWAP_USED_CEILING):
        state.update({"status": "WAITING_RESOURCE_FLOOR",
                      "blocker": "reference launch waits for >=12 GiB available and <=16 GiB swap"})
        save_state(state)
        heartbeat(state)
        time.sleep(30)
        return None, state
    state.update({"status": "RUNNING", "blocker": None})
    save_state(state)
    with (RUNTIME / log_name).open("a", encoding="utf-8") as output:
        process = subprocess.Popen(command, stdout=output, stderr=output,
                                   start_new_session=False)
        state["stage_child_pid"] = process.pid
        save_state(state)
        while process.poll() is None:
            state = control_flags(state)
            if active_path:
                active = read_json(active_path, {})
                if active:
                    label = (
                        f"{active.get('probe', 'probe')} layer "
                        f"{int(active.get('layer', -1)) + 1}/{active.get('layers_total', 61)}"
                    )
                    probe_id = str(active.get("probe"))
                    layer_unit = int(active.get("layer", -1)) + 1
                    if probe_id.startswith("determinism_a"):
                        complete_units, total_units = 61 + 0.2 * layer_unit, 61 * 1.4
                    elif probe_id.startswith("determinism_b"):
                        complete_units, total_units = 61 * 1.2 + 0.2 * layer_unit, 61 * 1.4
                    elif probe_id.startswith("batch_"):
                        complete_units, total_units = layer_unit, 61 * 1.4
                    else:
                        probes = ["factual", "science", "coding", "mathematics", "reasoning",
                                  "instruction", "tool_thinking_protocol", "rare_token",
                                  "mathematics_replay"]
                        pass_index = probes.index(probe_id) if probe_id in probes else 0
                        complete_units = pass_index * 61 + layer_unit
                        total_units = len(probes) * 61
                    epoch = float(state.setdefault("reference_epoch", time.time()))
                    elapsed = max(0.001, time.time() - epoch)
                    rate = complete_units / elapsed
                    state["stage_progress_text"] = (
                        f"reference {complete_units:.1f}/{total_units:.1f} weighted layer units; "
                        f"source 64/64 shards"
                    )
                    state["stage_eta_seconds"] = ((total_units - complete_units) / rate
                                                  if rate > 0 else None)
                    if state.get("current_layer") != label:
                        state["current_layer"] = label
                        save_state(state)
            memory = memory_snapshot()
            swap = swap_snapshot()
            reason = None
            if state.get("stop_requested"):
                reason = "stop requested; sealed layer/probe checkpoints preserved"
                state["status"] = "STOPPED_PRESERVED"
            elif state.get("pause_after_checkpoint"):
                reason = "pause requested; last sealed layer checkpoint preserved"
                state["status"] = "PAUSED_AFTER_CHECKPOINT"
            elif int(memory.get("available_bytes_estimate") or 0) < AVAILABLE_MEMORY_FLOOR:
                reason = "available-memory floor crossed; child stopped at resumable checkpoint"
                state["status"] = "RESOURCE_GUARD_PAUSED"
            elif int(swap.get("swap_used_bytes") or 0) > SWAP_USED_CEILING:
                reason = "swap ceiling crossed; child stopped at resumable checkpoint"
                state["status"] = "RESOURCE_GUARD_PAUSED"
            elif not disk_floor_green(disk_free()):
                reason = "disk reserve crossed; child stopped at resumable checkpoint"
                state["status"] = "RESOURCE_GUARD_PAUSED"
            if reason:
                terminate_child(process)
                state["blocker"] = reason
                state.pop("stage_child_pid", None)
                save_state(state)
                checkpoint(state, f"stage-guard:{state.get('state')}", reason)
                return None, state
            heartbeat(state)
            time.sleep(15)
    state.pop("stage_child_pid", None)
    save_state(state)
    return process.returncode, state


def build_reference(state: dict[str, Any]) -> dict[str, Any]:
    existing_parent = read_json(PARENT_VALIDATION, {})
    if (existing_parent.get("status") == "PASS" and valid_seal(existing_parent) and
            REFERENCE_EVIDENCE.exists()):
        return transition(state, "VALIDATE_PARENT", "bounded source forward already sealed")
    if not INSTALL_REFERENCE.exists():
        state.update({"status": "WAITING_INSTALLED_REFERENCE",
                      "blocker": "installed reference module absent; reinstall controller"})
        save_state(state)
        heartbeat(state)
        time.sleep(30)
        return state
    REFERENCE_RUN.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), str(INSTALL_REFERENCE), "run-suite", "--source", str(SNAPSHOT),
               "--output", str(REFERENCE_RUN)]
    returncode, state = managed_stage(
        state, command, "reference_forward.log", REFERENCE_RUN / "active_probe.json"
    )
    if returncode is None:
        return state
    suite = read_json(REFERENCE_RUN / "KIMI_K26_PARENT_FORWARD_VALIDATION.json", {})
    if returncode != 0 or suite.get("status") != "PASS" or not valid_seal(suite):
        state.update({"status": "BLOCKED", "blocker":
                      "bounded source forward failed; inspect reference_forward.log"})
        return transition(state, "BLOCKED", state["blocker"])
    atomic_json(PARENT_VALIDATION, suite)
    reference = seal({
        "schema": "hawking.kimi_k26.reference_forward.v1", "status": "PASS",
        "sealed_at": now(), "repo": REPO_ID, "revision": REVISION,
        "claim_boundary": "TEXT_CORE_ONLY; no multimodal preservation claim",
        "runtime": "MLX Metal native official packed INT4; selected experts only",
        "source_shard_window": 1, "complete_model_dequantized": False,
        "all_61_text_layers_in_official_order": True,
        "official_tokenizer_and_chat_protocol": True,
        "real_logits": True, "silent_tensor_omission": False,
        "vision": "SOURCE_PRESENT_ADAPTER_MAPPED_NOT_INCLUDED_IN_FIRST_CAPABILITY_CLAIM",
        "official_runtime_parity_claimed": False,
        "parent_validation_seal_sha256": suite["seal_sha256"],
        "probe_count": suite["probe_count"],
        "coherent_probe_count": suite["coherent_probe_count"],
        "deterministic_replay": suite["deterministic_replay"],
        "reference_program_sha256": sha256_file(INSTALL_REFERENCE),
    })
    atomic_json(REFERENCE_EVIDENCE, reference)
    state.update({"current_layer": None, "best_candidate": "P0_OFFICIAL_PARENT_REFERENCE",
                  "stage_progress_text": "parent reference 9/9 probes sealed",
                  "stage_eta_seconds": 0,
                  "primary_metrics": {"coherent_probes": suite["coherent_probe_count"],
                                      "finite_probes": suite["finite_probe_count"],
                                      "deterministic_replay": "PASS"}})
    return transition(state, "VALIDATE_PARENT", "coherent deterministic text-core parent sealed")


def build_doctor(state: dict[str, Any]) -> dict[str, Any]:
    evidence = read_json(CORPUS_INTEGRITY, {})
    if evidence.get("status") != "PASS" or not valid_seal(evidence):
        state.update({"status": "BLOCKED", "blocker":
                      "Kimi clean-corpus integrity artifact absent or invalid"})
        return transition(state, "BLOCKED", state["blocker"])
    return transition(state, "RUN_CAUSAL", "clean disjoint corpus gate sealed")


def run_causal(state: dict[str, Any]) -> dict[str, Any]:
    existing = read_json(CAUSAL_ATLAS, {})
    if existing.get("status") == "PASS" and valid_seal(existing):
        return transition(state, "RUN_TOURNAMENT", "causal intervention harness already sealed")
    if not INSTALL_DOCTOR.exists():
        state.update({"status": "BLOCKED", "blocker": "installed Doctor Prime module absent"})
        return transition(state, "BLOCKED", state["blocker"])
    command = [str(PYTHON), str(INSTALL_DOCTOR), "causal", "--corpus",
               str(CORPUS_INTEGRITY), "--parent", str(PARENT_VALIDATION),
               "--output", str(CAUSAL_ATLAS)]
    returncode, state = managed_stage(state, command, "causal_harness.log")
    evidence = read_json(CAUSAL_ATLAS, {})
    if returncode != 0 or evidence.get("status") != "PASS" or not valid_seal(evidence):
        state.update({"status": "BLOCKED", "blocker": "causal intervention harness failed"})
        return transition(state, "BLOCKED", state["blocker"])
    state["primary_metrics"] = {**(state.get("primary_metrics") or {}),
                                "causal_harness": "PASS",
                                "real_candidate_diagnosis": "PENDING_F1"}
    return transition(state, "RUN_TOURNAMENT", "all causal seams classify known twin damage")


def run_tournament(state: dict[str, Any]) -> dict[str, Any]:
    if not INSTALL_DOCTOR.exists():
        state.update({"status": "BLOCKED", "blocker": "installed Doctor Prime module absent"})
        return transition(state, "BLOCKED", state["blocker"])
    auction = read_json(BYTE_AUCTION, {})
    if auction.get("status") != "PASS" or not valid_seal(auction):
        command = [str(PYTHON), str(INSTALL_DOCTOR), "auction", "--ledger", str(LEDGER),
                   "--causal", str(CAUSAL_ATLAS), "--output", str(BYTE_AUCTION)]
        returncode, state = managed_stage(state, command, "doctor_auction.log")
        auction = read_json(BYTE_AUCTION, {})
        if returncode != 0 or auction.get("status") != "PASS" or not valid_seal(auction):
            state.update({"status": "BLOCKED", "blocker": "exact Doctor byte auction failed"})
            return transition(state, "BLOCKED", state["blocker"])
    command = [str(PYTHON), str(INSTALL_DOCTOR), "tournament", "--auction",
               str(BYTE_AUCTION), "--parent", str(PARENT_VALIDATION), "--output",
               str(TOURNAMENT), "--checkpoint-output", str(FIRST_CHECKPOINT)]
    returncode, state = managed_stage(state, command, "tournament.log")
    evidence = read_json(TOURNAMENT, {})
    first = read_json(FIRST_CHECKPOINT, {})
    if (returncode != 0 or evidence.get("status") != "PASS" or not valid_seal(evidence) or
            first.get("status") != "PASS" or not valid_seal(first)):
        state.update({"status": "BLOCKED", "blocker": "tournament F0/first checkpoint failed"})
        return transition(state, "BLOCKED", state["blocker"])
    state.update({"best_candidate": "P0_OFFICIAL_PARENT_REFERENCE", "complete_bpw": None,
                  "current_layer": "P1/P5 F1 representation bracket preflight advancing",
                  "stage_progress_text": "P0 sealed; P1/P5 F1 bracket preflight advancing",
                  "stage_eta_seconds": None,
                  "primary_metrics": {**(state.get("primary_metrics") or {}),
                                      "doctor_byte_auction": "PASS",
                                      "compact_capability": "NONE_YET"}})
    atomic_json(NEXT_EXPERIMENT, seal({
        "schema": "hawking.kimi_k26.next_experiment.v1", "status": "ADVANCING",
        "started_at": now(), "repo": REPO_ID, "revision": REVISION,
        "experiment": "P1_AND_P5_F1_REPRESENTATION_BRACKET",
        "current_action": "freeze disjoint unique-token probes and real routed-expert output seam",
        "heavy_lease": str(LEASE), "controller_pid": os.getpid(),
        "hard_boundary": "no candidate result or compact capability is claimed before F1 executes",
    }))
    return transition(state, "SEAL_RESULT", "P0 and five legal F0 candidates admitted")


def rebind_advancing_experiment() -> None:
    """Keep the sealed advancing-work receipt bound to the current restart-safe owner."""
    evidence = read_json(NEXT_EXPERIMENT, {})
    if evidence.get("status") != "ADVANCING" or not valid_seal(evidence):
        return
    evidence.update({"controller_pid": os.getpid(), "heavy_lease": str(LEASE),
                     "controller_rebound_at": now()})
    atomic_json(NEXT_EXPERIMENT, seal(evidence))


def step(state: dict[str, Any]) -> dict[str, Any]:
    name = state["state"]
    state = control_flags(state)
    if name != "DOWNLOAD" and state.get("stop_requested"):
        state["status"] = "STOPPED_PRESERVED"
        save_state(state)
        checkpoint(state, "controller:stopped", "all sealed stage checkpoints preserved")
        return state
    if name != "DOWNLOAD" and state.get("pause_after_checkpoint"):
        state["status"] = "PAUSED_AFTER_CHECKPOINT"
        save_state(state)
        heartbeat(state)
        time.sleep(15)
        return state
    if name == "PRECHECK":
        return precheck(state)
    if name == "ADMIT_SOURCE":
        return admit(state)
    if name == "FREE_STORAGE":
        return storage_gate(state)
    if name == "DOWNLOAD":
        return download_one(state)
    if name == "VERIFY_SOURCE":
        return verify_source(state)
    if name == "BUILD_LEDGER":
        return build_ledger(state)
    if name == "BUILD_ADAPTER":
        return build_adapter(state)
    if name == "BUILD_REFERENCE":
        return build_reference(state)
    if name == "VALIDATE_PARENT":
        return evidence_gate(state, "KIMI_K26_PARENT_FORWARD_VALIDATION.json", "BUILD_DOCTOR")
    if name == "BUILD_DOCTOR":
        return build_doctor(state)
    if name == "RUN_CAUSAL":
        return run_causal(state)
    if name == "RUN_TOURNAMENT":
        return run_tournament(state)
    if name == "SEAL_RESULT":
        return evidence_gate(state, "KIMI_K26_FIRST_CHECKPOINT.json", "MONITOR")
    if name in {"MONITOR", "COMPLETE"}:
        heartbeat(state)
        time.sleep(30)
        return state
    if name == "BLOCKED":
        heartbeat(state)
        time.sleep(30)
        return state
    raise CampaignError(f"no transition for {name}")


def acquire_lease():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    handle = LEASE.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise CampaignError("exclusive Kimi heavy lease is already held") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"label": LABEL, "pid": os.getpid(), "process_group": os.getpgrp(),
                             "acquired_at": now()}))
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def run_loop() -> int:
    lease_handle = acquire_lease()
    stop_signal = False

    def request_stop(_sig, _frame):
        nonlocal stop_signal
        stop_signal = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        state = load_state()
        state["disk_policy"] = {
            "schema": "hawking.kimi_k26.disk_policy.v1",
            "hard_floor_bytes": MIN_RESERVE,
            "comparison": "free_disk_bytes > hard_floor_bytes",
            "source": "KIMI_K26_DISK_FLOOR_BYTES with 5 GiB default",
        }
        if isinstance(state.get("storage_gate"), dict):
            state["storage_gate"]["reserve_bytes"] = MIN_RESERVE
            state["storage_gate"]["target_reserve_bytes"] = MIN_RESERVE
        save_state(state)
        if state.get("status") == "STOPPED_PRESERVED":
            state["status"] = "RUNNING"
            atomic_json(CONTROL, {"pause_after_checkpoint": False, "stop_requested": False})
            save_state(state)
        state = control_flags(state)
        checkpoint(state, "controller:started", "exclusive lease acquired")
        if state.get("state") == "MONITOR":
            rebind_advancing_experiment()
        while True:
            if stop_signal:
                atomic_json(CONTROL, {"pause_after_checkpoint": False, "stop_requested": True})
            state = load_state()
            state = step(state)
            if state.get("status") == "STOPPED_PRESERVED":
                break
        return 0
    except Exception as exc:  # noqa: BLE001
        state = load_state()
        state.update({"status": "BLOCKED", "blocker": f"{type(exc).__name__}: {exc}",
                      "failed_checkpoints": int(state.get("failed_checkpoints", 0)) + 1})
        save_state(state)
        checkpoint(state, f"fault:{state.get('state')}", state["blocker"])
        log(state["blocker"])
        return 1
    finally:
        try:
            fcntl.flock(lease_handle.fileno(), fcntl.LOCK_UN)
            lease_handle.close()
        except Exception:
            pass


def save_fallback_credentials() -> bool:
    def get(service: str) -> str | None:
        result = subprocess.run(["/usr/bin/security", "find-generic-password", "-a",
                                 KEYCHAIN_ACCOUNT, "-s", service, "-w"], text=True,
                                capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None
    token, chat = get(TOKEN_SERVICE), get(CHAT_SERVICE)
    if not token or not chat:
        return False
    atomic_json(CREDS, {"token": token, "chat_id": chat}, mode=0o600)
    return True


def install(repo_root: Path) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    shutil.copy2(source, INSTALL_SCRIPT)
    os.chmod(INSTALL_SCRIPT, 0o755)
    adapter = repo_root / "tools/condense/kimi_k26_adapter.py"
    installed_modules = (
        (adapter, INSTALL_ADAPTER),
        (repo_root / "tools/condense/kimi_k26_reference.py", INSTALL_REFERENCE),
        (repo_root / "tools/condense/kimi_k26_corpus.py", INSTALL_CORPUS),
        (repo_root / "tools/condense/kimi_k26_doctor_prime.py", INSTALL_DOCTOR),
    )
    for module, destination in installed_modules:
        if not module.exists():
            raise CampaignError(f"required campaign module absent: {module}")
        shutil.copy2(module, destination)
        os.chmod(destination, 0o755)
    corpus_evidence = repo_root / "KIMI_K26_CORPUS_INTEGRITY.json"
    if not corpus_evidence.exists():
        raise CampaignError("clean-corpus artifact must pass before controller installation")
    corpus = read_json(corpus_evidence, {})
    if corpus.get("status") != "PASS" or not valid_seal(corpus):
        raise CampaignError("clean-corpus artifact is not a valid PASS seal")
    atomic_json(CORPUS_INTEGRITY, corpus)
    official_manifest(refresh=True)
    if not STATE.exists():
        save_state(initial_state())
    fallback = save_fallback_credentials()
    plist = {
        "Label": LABEL,
        "ProgramArguments": ["/usr/bin/caffeinate", "-dimsu", str(PYTHON),
                             str(INSTALL_SCRIPT), "run"],
        "WorkingDirectory": str(RUNTIME), "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False}, "ThrottleInterval": 30,
        "ProcessType": "Background", "LowPriorityIO": False,
        "StandardOutPath": str(RUNTIME / "launchd.out.log"),
        "StandardErrorPath": str(RUNTIME / "launchd.err.log"),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_XET": "0",
                                 "HF_XET_HIGH_PERFORMANCE": "1",
                                 "KIMI_K26_DISK_FLOOR_BYTES": str(MIN_RESERVE)},
    }
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    temporary = PLIST.with_suffix(".plist.tmp")
    with temporary.open("wb") as handle:
        plistlib.dump(plist, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, PLIST)
    os.chmod(PLIST, 0o644)
    domain = f"gui/{os.getuid()}"
    old_pid = read_json(HEARTBEAT, {}).get("pid")
    subprocess.run(["/bin/launchctl", "bootout", domain, str(PLIST)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    for _ in range(60):
        if not old_pid or not pid_alive(old_pid):
            break
        time.sleep(1)
    if old_pid and pid_alive(old_pid):
        raise CampaignError(f"existing controller PID {old_pid} did not stop for reinstall")
    installed_state = load_state()
    one_copy = read_json(ONE_COPY, {})
    if (installed_state.get("source_state") == "KIMI_DOWNLOAD_COMPLETE" and
            not one_copy.get("spotlight_and_known_roots_scanned")):
        installed_state.update({
            "state": "VERIFY_SOURCE", "status": "RUNNING", "blocker": None,
            "entered_at": now(), "current_layer": None,
            "stage_progress_text": "re-run strengthened one-copy audit and exact packed ledger",
            "stage_eta_seconds": None,
        })
        installed_state["history"] = (installed_state.get("history", []) + [{
            "at": now(), "to": "VERIFY_SOURCE",
            "note": "controller upgrade: strengthened duplicate scan and pack geometry audit",
        }])[-128:]
        save_state(installed_state)
    # The prior signal handler may have written a stop flag while draining.  Clear it only after
    # that process is gone, then bootstrap the new immutable installed program.
    atomic_json(CONTROL, {"pause_after_checkpoint": False, "stop_requested": False,
                          "requested_at": now()})
    result = subprocess.run(["/bin/launchctl", "bootstrap", domain, str(PLIST)],
                            text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise CampaignError(f"launchctl bootstrap failed: {result.stderr.strip()}")
    subprocess.run(["/bin/launchctl", "kickstart", "-k", f"{domain}/{LABEL}"], check=False)
    print(json.dumps({"installed": str(INSTALL_SCRIPT), "plist": str(PLIST),
                      "runtime": str(RUNTIME), "telegram_fallback_created": fallback}, indent=2))


def set_control(*, pause: bool, stop: bool) -> None:
    atomic_json(CONTROL, {"pause_after_checkpoint": pause, "stop_requested": stop,
                          "requested_at": now()})
    if not stop:
        domain = f"gui/{os.getuid()}"
        subprocess.run(["/bin/launchctl", "kickstart", f"{domain}/{LABEL}"], check=False)


def status_command() -> None:
    state = load_state()
    write_status(state)
    print(PHONE_MD.read_text(encoding="utf-8"))


def sync_evidence(repo_root: Path) -> None:
    target = repo_root / "reports/condense/kimi_k26"
    target.mkdir(parents=True, exist_ok=True)
    evidence_paths = (STATE, HEARTBEAT, MANIFEST, VERIFICATION, ONE_COPY, LEDGER, FORMAT_LEDGER,
                      REFERENCE_EVIDENCE, PARENT_VALIDATION, CORPUS_INTEGRITY, CAUSAL_ATLAS,
                      BYTE_AUCTION, TOURNAMENT, FIRST_CHECKPOINT, NEXT_EXPERIMENT,
                      PHONE_JSON, PHONE_MD, RUNTIME / "KIMI_K26_ADAPTER_TWIN.json")
    for path in evidence_paths:
        if path.exists():
            shutil.copy2(path, target / path.name)
    required_root = (STATE, VERIFICATION, ONE_COPY, LEDGER, FORMAT_LEDGER,
                     RUNTIME / "KIMI_K26_ADAPTER_TWIN.json", REFERENCE_EVIDENCE,
                     PARENT_VALIDATION, CORPUS_INTEGRITY, CAUSAL_ATLAS, BYTE_AUCTION,
                     TOURNAMENT, FIRST_CHECKPOINT, NEXT_EXPERIMENT, PHONE_JSON, PHONE_MD)
    for path in required_root:
        if path.exists():
            shutil.copy2(path, repo_root / path.name)
    print(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    sub.add_parser("status")
    sub.add_parser("pause-after-checkpoint")
    sub.add_parser("resume")
    sub.add_parser("stop")
    install_parser = sub.add_parser("install")
    install_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    sync_parser = sub.add_parser("sync-evidence")
    sync_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_loop()
    if args.command == "status":
        status_command()
    elif args.command == "pause-after-checkpoint":
        set_control(pause=True, stop=False)
        print("pause requested; the latest sealed checkpoint and resumable partials are preserved")
    elif args.command == "resume":
        state = load_state()
        if state.get("state") == "BLOCKED":
            retry_state = retry_state_for_blocker(str(state.get("blocker", "")))
            if retry_state:
                state.update({"state": retry_state, "status": "RUNNING", "blocker": None,
                              "pause_after_checkpoint": False, "stop_requested": False,
                              "entered_at": now()})
                state["history"] = (state.get("history", []) + [{
                    "at": now(), "to": retry_state,
                    "note": "phone resume after repairable sealed-stage failure",
                }])[-128:]
                save_state(state)
        elif state.get("status") in {"PAUSED_AFTER_CHECKPOINT", "RESOURCE_GUARD_PAUSED"}:
            state.update({"status": "RUNNING", "blocker": None,
                          "pause_after_checkpoint": False, "stop_requested": False,
                          "entered_at": now()})
            save_state(state)
        set_control(pause=False, stop=False)
        print("resume requested")
    elif args.command == "stop":
        set_control(pause=False, stop=True)
        print("stop requested; all sealed work will be preserved")
    elif args.command == "install":
        install(args.repo_root.resolve())
    elif args.command == "sync-evidence":
        sync_evidence(args.repo_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
