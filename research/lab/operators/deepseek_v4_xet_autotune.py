"""Public-path Xet autotuning for the pinned DeepSeek-V4 Gravity stream.

This module is intentionally independent of the older GLM controller.  It
measures a fixed set of verified DeepSeek-V4 byte ranges using a *fresh Python
process per candidate*.  That is important: hf_xet reads its configuration at
import time, so changing ``os.environ`` in a long-running parent process is
not a valid A/B test.

The live body benchmark is deliberately bounded:

* every body range is bound to the pinned Hub revision, LFS object identity,
  Xet object identity, and an independently measured SHA-256;
* source bytes live only in bounded RAM and in an immediately-evicted
  test-frame while verification/packing is measured;
* no Hub or Xet chunk cache is retained, and the 15 GiB storage floor is
  checked before and during every child trial;
* partial trial records are append-only and restart-safe; a later run skips
  completed candidate/phase pairs rather than discarding their evidence.

The ``run`` command performs discovery, successive-halving, transport and
scheduler comparisons, then writes the requested TG receipts.  It never
claims that a benchmark frame is a full 43-layer runtime or an independently
validated Condense model pack.  ``resume-real`` consumes the frozen winner
only when a safe, incomplete Gravity run exists; it refuses to re-download a
sealed source artifact merely to manufacture activity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import queue
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import zlib
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from lab.layout import RECORDS_ROOT, evidence_dir
from lab.receipts import SealIntegrityError, seal, verify


REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash"
REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
MODEL_ID = f"{REPOSITORY}@{REVISION}"
PINNED_VERSIONS = {"huggingface_hub": "1.24.0", "hf_xet": "1.5.2"}
MIN_FREE_FLOOR_BYTES = 15 * 1024**3
MAX_SWAP_USED_BYTES = 8 * 1024**3
MAX_SWAP_GROWTH_BYTES = 1 * 1024**3
MAX_RANGE_BYTES = 16 * 1024**2
CONTROL_RANGE_BYTES = 1 * 1024**2
MEDIUM_RANGE_BYTES = 8 * 1024**2
LARGE_FP4_RANGE_BYTES = 16 * 1024**2
CORPUS_SHARDS = (
    "model-00001-of-00046.safetensors",
    "model-00002-of-00046.safetensors",
    "model-00006-of-00046.safetensors",
    "model-00012-of-00046.safetensors",
    "model-00018-of-00046.safetensors",
    "model-00024-of-00046.safetensors",
    "model-00035-of-00046.safetensors",
    "model-00046-of-00046.safetensors",
)
FIXED_DOWNLOAD_CONCURRENCIES = (4, 8, 16, 24, 32, 48, 64, 96, 124)
FILE_DOWNLOAD_CONCURRENCIES = (1, 2, 4, 8, 12, 16)
RANGE_GET_CONCURRENCIES = (16, 32, 48, 64, 96, 128)
IDLE_CONNECTIONS = (16, 32, 64, 128)
SCHEDULER_SHAPES = (
    "one_file_many_ranges",
    "two_files_medium_range_concurrency",
    "four_files_medium_range_concurrency",
    "eight_files_low_per_file_concurrency",
    "dynamic_work_stealing",
    "bounded_prefetch_decode_pack_overlap",
)
CORPUS_SCHEMA = "hawking.gravity.deepseek_v4.xet_fixed_corpus.v1"
CHILD_CONFIG_SCHEMA = "hawking.gravity.deepseek_v4.xet_child_config.v1"
CHILD_RESULT_SCHEMA = "hawking.gravity.deepseek_v4.xet_trial_result.v1"
MATRIX_SCHEMA = "hawking.gravity.deepseek_v4.maximum_public_path_matrix.v1"
WINNER_SCHEMA = "hawking.gravity.deepseek_v4.maximum_public_path_winner.v1"
ROOFLINE_SCHEMA = "hawking.gravity.deepseek_v4.public_path_roofline.v1"
PROGRESS_SCHEMA = "hawking.gravity.deepseek_v4.real_stream_progress.v1"
PARTIAL_LEDGER_SCHEMA = "hawking.gravity.deepseek_v4.xet_partial_trial.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_DIR = evidence_dir("tg")
DEFAULT_RUN_DIR = RECORDS_ROOT / "runs" / "deepseek-v4"
MATRIX_NAME = "TG_XET_MAXIMUM_PUBLIC_PATH_MATRIX.json"
WINNER_NAME = "TG_XET_MAXIMUM_PUBLIC_PATH_WINNER.json"
ROOFLINE_NAME = "TG_XET_PUBLIC_PATH_ROOFLINE.json"
CORPUS_NAME = "TG_XET_DEEPSEEK_V4_FIXED_CORPUS.json"
PARTIAL_NAME = "TG_XET_MAXIMUM_PUBLIC_PATH_PARTIAL.jsonl"
PROGRESS_NAME = "TG_XET_REAL_STREAM_PROGRESS.jsonl"
LAUNCHER_NAME = "DEEPSEEK_V4_FAST_STREAM_RESUME.sh"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DeepSeekV4XetAutotuneError(RuntimeError):
    """A public-path candidate cannot be safely measured or promoted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise DeepSeekV4XetAutotuneError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(path)))


def _ensure_dir(path: Path, label: str) -> None:
    if path.exists():
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise DeepSeekV4XetAutotuneError(f"{label} must be a non-symlink directory")
        return
    path.mkdir(parents=True, exist_ok=False)


def _regular_file(path: Path, label: str) -> None:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise DeepSeekV4XetAutotuneError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise DeepSeekV4XetAutotuneError(f"{label} must be a regular non-symlink file")


def _atomic_bytes(path: Path, raw: bytes, *, immutable: bool = True) -> str:
    """Atomically create a receipt, never silently replacing different evidence."""

    digest = _sha256(raw)
    if path.exists():
        _regular_file(path, str(path))
        existing = path.read_bytes()
        if immutable and existing != raw:
            raise DeepSeekV4XetAutotuneError(f"refusing to overwrite different evidence: {path}")
        if existing != raw:
            raise DeepSeekV4XetAutotuneError(f"mutable writes are not permitted: {path}")
        return digest
    _ensure_dir(path.parent, f"parent for {path.name}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    return _atomic_bytes(path, _canonical(value) + b"\n")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepSeekV4XetAutotuneError(f"cannot read {label}: {exc}") from exc
    if not isinstance(result, dict):
        raise DeepSeekV4XetAutotuneError(f"{label} must contain a JSON object")
    return result


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    _ensure_dir(path.parent, f"parent for {path.name}")
    encoded = _canonical(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "ab", closefd=True) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    _regular_file(path, path.name)
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DeepSeekV4XetAutotuneError(f"invalid JSONL line {number} in {path}") from exc
        if not isinstance(value, dict):
            raise DeepSeekV4XetAutotuneError(f"JSONL line {number} in {path} is not an object")
        records.append(value)
    return records


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _floor_check(path: Path, *, additional_bytes: int, stage: str) -> dict[str, Any]:
    free = _free_bytes(path)
    if free - additional_bytes < MIN_FREE_FLOOR_BYTES:
        raise DeepSeekV4XetAutotuneError(
            f"15 GiB storage floor would be crossed at {stage}: "
            f"free={free}, additional={additional_bytes}"
        )
    return {
        "stage": stage,
        "free_bytes": free,
        "additional_bytes": additional_bytes,
        "protected_floor_bytes": MIN_FREE_FLOOR_BYTES,
        "status": "PASS",
    }


def _run_command(argv: Sequence[str], *, timeout: float = 15.0) -> str | None:
    try:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _network_interface() -> str | None:
    route = _run_command(("route", "-n", "get", "default"))
    if not route:
        return None
    match = re.search(r"^\s*interface:\s*(\S+)", route, re.MULTILINE)
    return match.group(1) if match else None


def _net_counter(interface: str | None) -> dict[str, Any]:
    if not interface:
        return {"available": False, "interface": None}
    # Darwin's netstat -ibI layout has been stable across supported macOS
    # releases; take the last numeric line so an interface alias cannot leak
    # into accounting.
    text = _run_command(("netstat", "-ibI", interface))
    if not text:
        return {"available": False, "interface": interface}
    rows = [line.split() for line in text.splitlines() if line.split() and line.split()[0] == interface]
    if not rows:
        return {"available": False, "interface": interface}
    row = rows[0]
    # Darwin netstat columns are: Name Mtu Network Address Ipkts Ierrs Ibytes
    # Opkts Oerrs Obytes Coll.  Use the link row, not the IPv6/IPv4 aliases.
    try:
        ibytes, obytes = int(row[6]), int(row[9])
    except (IndexError, ValueError):
        return {"available": False, "interface": interface, "raw": row}
    return {
        "available": True,
        "interface": interface,
        "input_bytes": ibytes,
        "output_bytes": obytes,
        "aggregate_bytes": ibytes + obytes,
        "method": "netstat_-ibI_darwin_interface_counters",
    }


def _swap_used_bytes() -> int | None:
    text = _run_command(("sysctl", "vm.swapusage"))
    if not text:
        return None
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG]?)", text)
    if not match:
        return None
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def _memory_pressure() -> str | None:
    text = _run_command(("memory_pressure", "-Q"))
    return text[:4000] if text else None


def _thermal_state() -> dict[str, Any]:
    text = _run_command(("pmset", "-g", "therm"))
    if not text:
        return {"observable": False, "warning": None}
    lowered = text.lower()
    if "no thermal warning" in lowered and "no performance warning" in lowered:
        warning = False
    else:
        warning = any(
            word in lowered for word in ("thermal warning", "cpu_speed_limit", "high temperature")
        )
    return {"observable": True, "warning": warning, "raw": text[:4000]}


def _top_processes() -> list[dict[str, Any]]:
    text = _run_command(("ps", "-Ao", "pid=,pcpu=,pmem=,rss=,command="))
    if not text:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.strip().split(maxsplit=4)
        if len(fields) != 5:
            continue
        try:
            rows.append(
                {
                    "pid": int(fields[0]),
                    "cpu_percent": float(fields[1]),
                    "memory_percent": float(fields[2]),
                    "rss_kib": int(fields[3]),
                    "command": fields[4][:240],
                }
            )
        except ValueError:
            continue
    return sorted(rows, key=lambda row: row["cpu_percent"], reverse=True)[:12]


def _filesystem_type(path: Path) -> str | None:
    return _run_command(("stat", "-f", "%T", str(path)))


def host_snapshot(workspace: str | Path) -> dict[str, Any]:
    """Collect the available, read-only local admission evidence."""

    root = _absolute(workspace, "workspace")
    interface = _network_interface()
    media = _run_command(("networksetup", "-getMedia", interface)) if interface else None
    rustc = _run_command(("rustc", "--version"))
    cargo = _run_command(("cargo", "--version"))
    top = _top_processes()
    pressure = _memory_pressure()
    contention_markers = (
        "pgbench",
        "fio ",
        "dd if=",
        "stress",
        "iperf",
        "git clone",
        "index-pack",
        "pack-objects",
        "rsync ",
    )
    named_contention = any(
        any(marker in str(row["command"]).lower() for marker in contention_markers) for row in top
    )
    # This is deliberately conservative and only used to disqualify a final
    # promotion.  Short probes may still record the current public path while
    # a user-owned compile/copy is active; they are not equivalent to a clean
    # confirmation run.
    compute_markers = ("rustc", "clang", "gcc", "go-build", "go build", "python", "node", "blender", "ffmpeg", "git ")
    heavy_rows = [
        row
        for row in top
        if "deepseek_v4_xet_autotune" not in str(row["command"])
        and not str(row["command"]).startswith(("/System/", "/usr/libexec/", "/Applications/"))
        and any(marker in str(row["command"]).lower() for marker in compute_markers)
    ]
    # A handful of 100%-of-one-core processes on a 28-core Mac is not the
    # disk/network contention this guard is intended to detect.  Treat it as
    # final-confirmation contention only once it consumes at least a quarter
    # of aggregate logical CPU, or when a named I/O load is present.
    aggregate_heavy_cpu = sum(float(row["cpu_percent"]) for row in heavy_rows)
    heavy_unrelated_cpu = aggregate_heavy_cpu >= max(300.0, float(os.cpu_count() or 1) * 25.0)
    return {
        "sampled_at": _utc_now(),
        "platform": {
            "macos": platform.mac_ver()[0] or platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "rustc": rustc,
            "cargo": cargo,
        },
        "filesystem": {
            "path": str(root),
            "type": _filesystem_type(root),
            "free_bytes": _free_bytes(root),
            "protected_floor_bytes": MIN_FREE_FLOOR_BYTES,
        },
        "network": {
            "default_interface": interface,
            "media": media,
            "counter": _net_counter(interface),
            "route": "public_default_route_observed" if interface else "not_observed",
        },
        "memory": {"swap_used_bytes": _swap_used_bytes(), "pressure": pressure},
        "thermal": _thermal_state(),
        "top_processes": top,
        "contention_observed": named_contention or heavy_unrelated_cpu,
        "contention_detail": {
            "named_network_or_disk_job": named_contention,
            "heavy_unrelated_cpu": heavy_unrelated_cpu,
            "aggregate_marked_cpu_percent": aggregate_heavy_cpu,
        },
    }


def _clear_hf_environment(base: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in base.items()
        if not key.startswith("HF_XET_")
        and key
        not in {
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_HUB_DISABLE_XET",
            "HF_HUB_ENABLE_HF_TRANSFER",
        }
    }


def profile_environment(profile: Mapping[str, Any], scratch_root: Path) -> dict[str, str]:
    """Return the entire child environment before importing Hub/Xet modules."""

    transport = profile.get("transport")
    if transport not in {
        "official_hf_xet",
        "custom_direct_xet_range",
        "hub_http_without_xet",
        "direct_presigned_range",
    }:
        raise DeepSeekV4XetAutotuneError(f"unknown transport {transport!r}")
    _ensure_dir(scratch_root, "candidate scratch root")
    environment = _clear_hf_environment(os.environ)
    environment.update(
        {
            "HF_HOME": str(scratch_root / "hf-home"),
            "HF_HUB_CACHE": str(scratch_root / "hub-cache"),
            "HF_XET_CACHE": str(scratch_root / "xet-cache"),
            "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
            # The older spelling is recorded for invariant traceability;
            # vectored write is the currently supported implementation knob.
            "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY": "false",
            "HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE": "true",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_XET_LOG_DEST": "stderr",
            "HF_XET_LOG_FORMAT": "json",
        }
    )
    if transport == "hub_http_without_xet":
        environment["HF_HUB_DISABLE_XET"] = "1"
    if bool(profile.get("high_performance")):
        environment["HF_XET_HIGH_PERFORMANCE"] = "1"
    adaptive = profile.get("adaptive")
    if adaptive is not None:
        environment["HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY"] = "true" if adaptive else "false"
    fixed = profile.get("fixed_download_concurrency")
    if fixed is not None:
        environment["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] = str(int(fixed))
    files = profile.get("file_download_concurrency")
    if files is not None:
        environment["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] = str(int(files))
    ranges = profile.get("range_get_concurrency")
    if ranges is not None:
        environment["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(int(ranges))
    idle = profile.get("max_idle_connections")
    if idle is not None:
        environment["HF_XET_CLIENT_MAX_IDLE_CONNECTIONS"] = str(int(idle))
    retry = profile.get("retry")
    if isinstance(retry, Mapping):
        fields = {
            "max_attempts": "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS",
            "base_delay": "HF_XET_CLIENT_RETRY_BASE_DELAY",
            "max_duration": "HF_XET_CLIENT_RETRY_MAX_DURATION",
            "connect_timeout": "HF_XET_CLIENT_CONNECT_TIMEOUT",
            "read_timeout": "HF_XET_CLIENT_READ_TIMEOUT",
            "idle_timeout": "HF_XET_CLIENT_IDLE_CONNECTION_TIMEOUT",
        }
        for key, name in fields.items():
            if key in retry:
                # xet-core duration controls require an explicit unit.  Bare
                # numeric strings are not a reliable configuration surface.
                environment[name] = (
                    str(retry[key]) if key == "max_attempts" else f"{retry[key]}s"
                )
    for profile_key, environment_key in (
        ("adaptive_min_download_concurrency", "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY"),
        ("adaptive_initial_download_concurrency", "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY"),
        ("adaptive_max_download_concurrency", "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"),
    ):
        if profile.get(profile_key) is not None:
            environment[environment_key] = str(int(profile[profile_key]))
    return environment


def _child_runtime_config() -> dict[str, Any]:
    """Import the pinned runtime after the parent supplied the final env."""

    before = {
        "huggingface_hub_imported": "huggingface_hub" in sys.modules,
        "hf_xet_imported": "hf_xet" in sys.modules,
    }
    if any(before.values()):
        raise DeepSeekV4XetAutotuneError("Hugging Face runtime was imported before child setup")
    try:
        import hf_xet  # type: ignore[import-not-found]
        from hf_xet import XetConfig  # type: ignore[import-not-found]
        import huggingface_hub  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DeepSeekV4XetAutotuneError("pinned hf_xet runtime is unavailable") from exc
    versions: dict[str, str] = {}
    for package, expected in PINNED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DeepSeekV4XetAutotuneError(f"missing required package {package}") from exc
        if actual != expected:
            raise DeepSeekV4XetAutotuneError(
                f"{package} version drift: expected {expected}, observed {actual}"
            )
        versions[package] = actual
    wanted = (
        "data.max_concurrent_file_downloads",
        "chunk_cache.size_bytes",
        "client.enable_adaptive_concurrency",
        "client.ac_min_download_concurrency",
        "client.ac_initial_download_concurrency",
        "client.ac_max_download_concurrency",
        "client.max_idle_connections",
        "reconstruction.min_reconstruction_fetch_size",
        "reconstruction.max_reconstruction_fetch_size",
        "reconstruction.download_buffer_size",
        "reconstruction.download_buffer_perfile_size",
        "reconstruction.download_buffer_limit",
        "reconstruction.use_vectored_write",
    )
    items = dict(XetConfig().items())
    package_root = Path(hf_xet.__file__).resolve().parent
    binaries: list[dict[str, str]] = []
    for node in sorted(package_root.rglob("*")):
        try:
            executable = node.is_file() and os.access(node, os.X_OK)
        except OSError:
            executable = False
        if (executable or node.suffix in {".so", ".dylib"}) and node.stat().st_size > 1_000_000:
            binaries.append(
                {
                    "path": str(node),
                    "sha256": _sha256(node.read_bytes()),
                }
            )
    return {
        "schema": CHILD_CONFIG_SCHEMA,
        "status": "PASS",
        "imports_before_environment": before,
        "versions": versions,
        "python": sys.version.split()[0],
        "hf_xet_module": str(package_root),
        "hf_xet_embedded_binaries": binaries,
        "effective": {key: items.get(key) for key in wanted},
        "environment": {
            key: os.environ.get(key)
            for key in sorted(os.environ)
            if key.startswith("HF_XET_")
            or key
            in {
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_HUB_DISABLE_XET",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            }
        },
        "huggingface_hub_module": str(Path(huggingface_hub.__file__).resolve()),
    }


class _SourceBindings:
    """Late-bound Hub/Xet APIs; constructed only inside a configured child."""

    def __init__(self) -> None:
        try:
            from hf_xet import XetFileInfo, XetSession  # type: ignore[import-not-found]
            from huggingface_hub import get_token, hf_hub_url  # type: ignore[import-not-found]
            from huggingface_hub.file_download import get_hf_file_metadata  # type: ignore[import-not-found]
            from huggingface_hub.utils import build_hf_headers  # type: ignore[import-not-found]
            from huggingface_hub.utils._xet import (  # type: ignore[import-not-found]
                XetTokenType,
                xet_connection_info_refresh_url,
                xet_headers_without_auth,
            )
        except ImportError as exc:
            raise DeepSeekV4XetAutotuneError("pinned Hub/Xet range APIs are unavailable") from exc
        self.XetFileInfo = XetFileInfo
        self.XetSession = XetSession
        self.hf_hub_url = hf_hub_url
        self.get_hf_file_metadata = get_hf_file_metadata
        explicit = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        stored = get_token()
        self.token = explicit or stored or None
        self.headers = build_hf_headers(
            token=self.token or False,
            library_name="hawking-deepseek-v4-xet-autotune",
            library_version="1",
        )
        self.XetTokenType = XetTokenType
        self.refresh_url = xet_connection_info_refresh_url
        self.no_auth_headers = xet_headers_without_auth

    def metadata(self, shard: str) -> dict[str, Any]:
        value = self.get_hf_file_metadata(
            self.hf_hub_url(REPOSITORY, shard, revision=REVISION), token=self.token or False
        )
        xet = getattr(value, "xet_file_data", None)
        row = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "shard": shard,
            "commit_hash": getattr(value, "commit_hash", None),
            "etag_sha256": getattr(value, "etag", None),
            "file_size_bytes": getattr(value, "size", None),
            "xet_file_hash": getattr(xet, "file_hash", None),
            "xet_refresh_route": getattr(xet, "refresh_route", None),
            "signed_location": getattr(value, "location", None),
        }
        if row["commit_hash"] != REVISION:
            raise DeepSeekV4XetAutotuneError(f"{shard}: Hub revision identity drifted")
        if not _is_sha256(row["etag_sha256"]):
            raise DeepSeekV4XetAutotuneError(f"{shard}: missing LFS SHA-256 identity")
        if not _is_sha256(row["xet_file_hash"]):
            raise DeepSeekV4XetAutotuneError(f"{shard}: missing Xet object identity")
        if not isinstance(row["file_size_bytes"], int) or row["file_size_bytes"] <= 8:
            raise DeepSeekV4XetAutotuneError(f"{shard}: invalid source object size")
        if not isinstance(row["signed_location"], str) or not row["signed_location"].startswith("https://"):
            raise DeepSeekV4XetAutotuneError(f"{shard}: missing signed Xet bridge location")
        return row

    def new_group(self) -> Any:
        refresh = self.refresh_url(
            token_type=self.XetTokenType.READ,
            repo_id=REPOSITORY,
            repo_type="model",
            revision=REVISION,
        )
        return self.XetSession().new_download_stream_group(
            token_refresh_url=refresh,
            token_refresh_headers=self.headers,
            custom_headers=self.no_auth_headers(self.headers),
        )


def _http_range(
    url: str,
    *,
    start: int,
    end: int,
    file_size: int,
    token: str | None,
    timeout: float,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    follow_redirects: bool,
    connections: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Exact HTTP range read with explicit redirect and retry accounting."""

    import http.client

    if end <= start or start < 0 or end > file_size:
        raise DeepSeekV4XetAutotuneError("invalid exact HTTP range")
    expected = end - start
    current = url
    retries = 0
    attempts = 0
    first_host: str | None = None
    for attempt in range(max_attempts):
        attempts += 1
        try:
            redirects = 0
            while True:
                parsed = urllib.parse.urlsplit(current)
                if parsed.scheme != "https" or not parsed.netloc:
                    raise DeepSeekV4XetAutotuneError("range URL must be HTTPS")
                if first_host is None:
                    first_host = parsed.netloc
                connection = connections.get(parsed.netloc) if connections is not None else None
                if connection is None:
                    connection = http.client.HTTPSConnection(parsed.netloc, timeout=timeout)
                    if connections is not None:
                        connections[parsed.netloc] = connection
                else:
                    connection.timeout = timeout
                target = parsed.path or "/"
                if parsed.query:
                    target += "?" + parsed.query
                headers = {
                    "Range": f"bytes={start}-{end - 1}",
                    "User-Agent": "hawking-deepseek-v4-public-xet-autotune/1",
                }
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                started = time.monotonic()
                close_connection = connections is None
                try:
                    connection.request("GET", target, headers=headers)
                    response = connection.getresponse()
                    status = int(response.status)
                    response_headers = {key.lower(): value for key, value in response.getheaders()}
                    body = response.read(expected + 1)
                    close_connection = close_connection or bool(response.will_close)
                except Exception:
                    close_connection = True
                    raise
                finally:
                    if close_connection:
                        connection.close()
                        if connections is not None:
                            connections.pop(parsed.netloc, None)
                elapsed = time.monotonic() - started
                if status in {301, 302, 303, 307, 308} and follow_redirects:
                    location = response_headers.get("location")
                    if not location or redirects >= 5:
                        raise DeepSeekV4XetAutotuneError("Hub range redirect is invalid or excessive")
                    current = urllib.parse.urljoin(current, location)
                    redirects += 1
                    continue
                if status in {429, 500, 502, 503, 504}:
                    raise _RetryableHttpError(status, response_headers)
                required = f"bytes {start}-{end - 1}/{file_size}"
                if status != 206 or response_headers.get("content-range") != required:
                    raise DeepSeekV4XetAutotuneError(
                        "source did not honor exact range "
                        f"(status={status}, content-range={response_headers.get('content-range')!r}, "
                        f"expected={required!r})"
                    )
                if len(body) != expected:
                    raise DeepSeekV4XetAutotuneError(
                        f"exact range length mismatch: got {len(body)}, expected {expected}"
                    )
                return body, {
                    "attempts": attempts,
                    "retries": retries,
                    "elapsed_seconds": elapsed,
                    "host": urllib.parse.urlsplit(current).netloc,
                    "first_host": first_host,
                    "http_status": status,
                    "wire_bytes": len(body),
                    "redirects": redirects,
                }
        except _RetryableHttpError as exc:
            if attempt + 1 >= max_attempts:
                raise DeepSeekV4XetAutotuneError(
                    f"exact HTTP range exhausted retries after {attempts} attempts: HTTP {exc.status}"
                ) from exc
            retries += 1
            retry_after = exc.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after is not None else base_delay * (2**attempt)
            except ValueError:
                delay = base_delay * (2**attempt)
            time.sleep(max(0.01, min(max_delay, delay)))
        except (OSError, http.client.HTTPException) as exc:
            if attempt + 1 >= max_attempts:
                raise DeepSeekV4XetAutotuneError(f"exact HTTP range connection failed: {exc}") from exc
            retries += 1
            time.sleep(max(0.01, min(max_delay, base_delay * (2**attempt))))
    raise DeepSeekV4XetAutotuneError("exact HTTP range unexpectedly fell through")


class _RetryableHttpError(RuntimeError):
    def __init__(self, status: int, headers: Mapping[str, str]) -> None:
        self.status = status
        self.headers = dict(headers)
        super().__init__(f"retryable HTTP {status}")


def _read_safetensors_header(bindings: _SourceBindings, metadata: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    file_size = int(metadata["file_size_bytes"])
    first, _ = _http_range(
        str(metadata["signed_location"]),
        start=0,
        end=8,
        file_size=file_size,
        # The signed storage URL is already authorised; never send a Hub
        # bearer token to it.
        token=None,
        timeout=60.0,
        max_attempts=3,
        base_delay=0.25,
        max_delay=2.0,
        follow_redirects=False,
    )
    length = int.from_bytes(first, "little", signed=False)
    if length <= 0 or length > 8 * 1024**2 or length + 8 > file_size:
        raise DeepSeekV4XetAutotuneError(f"{metadata['shard']}: invalid bounded safetensors header")
    raw, _ = _http_range(
        str(metadata["signed_location"]),
        start=0,
        end=length + 8,
        file_size=file_size,
        token=None,
        timeout=60.0,
        max_attempts=3,
        base_delay=0.25,
        max_delay=2.0,
        follow_redirects=False,
    )
    try:
        value = json.loads(raw[8:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepSeekV4XetAutotuneError(f"{metadata['shard']}: invalid safetensors header JSON") from exc
    if not isinstance(value, dict):
        raise DeepSeekV4XetAutotuneError(f"{metadata['shard']}: safetensors header is not an object")
    return value, length + 8


def _descriptor_range(
    header: Mapping[str, Any],
    *,
    header_bytes: int,
    shard: str,
    preferred_dtypes: Sequence[str],
    desired_bytes: int,
    expert_only: bool = False,
) -> tuple[str, dict[str, Any], int, int]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for name, descriptor in header.items():
        if name == "__metadata__" or not isinstance(name, str) or not isinstance(descriptor, Mapping):
            continue
        dtype = descriptor.get("dtype")
        if dtype not in preferred_dtypes:
            continue
        if expert_only and ".experts." not in name:
            continue
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] <= offsets[0]
        ):
            continue
        candidates.append((name, descriptor))
    if not candidates:
        wanted = ",".join(preferred_dtypes)
        suffix = " expert" if expert_only else ""
        raise DeepSeekV4XetAutotuneError(f"{shard}: no usable{suffix} tensor for {wanted}")
    # A stable name order avoids choosing a changing header order as corpus
    # identity.  FP4 expert windows may intentionally span consecutive expert
    # tensors to expose a large contiguous public range.
    name, descriptor = sorted(candidates, key=lambda item: item[0])[0]
    offsets = descriptor["data_offsets"]
    start = header_bytes + int(offsets[0])
    end = start + desired_bytes
    return name, dict(descriptor), start, end


def _range_target(
    *,
    bindings: _SourceBindings,
    metadata: Mapping[str, Any],
    start: int,
    end: int,
    category: str,
    tensor: str | None,
    dtype: str | None,
    descriptor: Mapping[str, Any] | None,
) -> dict[str, Any]:
    file_size = int(metadata["file_size_bytes"])
    if start < 0 or end <= start or end > file_size or end - start > MAX_RANGE_BYTES:
        raise DeepSeekV4XetAutotuneError(f"{metadata['shard']}: selected corpus range is invalid")
    raw, response = _http_range(
        str(metadata["signed_location"]),
        start=start,
        end=end,
        file_size=file_size,
        token=None,
        timeout=120.0,
        max_attempts=4,
        base_delay=0.25,
        max_delay=4.0,
        follow_redirects=False,
    )
    return {
        "category": category,
        "repository": REPOSITORY,
        "revision": REVISION,
        "shard": metadata["shard"],
        "commit_hash": metadata["commit_hash"],
        "lfs_sha256": metadata["etag_sha256"],
        "xet_file_hash": metadata["xet_file_hash"],
        "file_size_bytes": file_size,
        "start": start,
        "end": end,
        "length": end - start,
        "sha256": _sha256(raw),
        "tensor": tensor,
        "dtype": dtype,
        "descriptor": dict(descriptor) if descriptor is not None else None,
        "discovery_response": {
            "host": response["host"],
            "http_status": response["http_status"],
            "wire_bytes": response["wire_bytes"],
        },
    }


def discover_fixed_corpus(path: str | Path) -> dict[str, Any]:
    """Build the immutable, exact-byte source corpus in a configured child."""

    target = _absolute(path, "corpus path")
    if target.exists():
        return validate_fixed_corpus(_read_json(target, "fixed corpus"))
    bindings = _SourceBindings()
    metadata_by_shard = {shard: bindings.metadata(shard) for shard in CORPUS_SHARDS}
    headers: dict[str, tuple[dict[str, Any], int]] = {}
    for shard in CORPUS_SHARDS:
        headers[shard] = _read_safetensors_header(bindings, metadata_by_shard[shard])
    ranges: list[dict[str, Any]] = []
    first = metadata_by_shard[CORPUS_SHARDS[0]]
    ranges.append(
        _range_target(
            bindings=bindings,
            metadata=first,
            start=0,
            end=min(CONTROL_RANGE_BYTES, int(first["file_size_bytes"])),
            category="small_metadata_control",
            tensor=None,
            dtype=None,
            descriptor=None,
        )
    )
    fp8_shard = CORPUS_SHARDS[1]
    fp8_header, fp8_header_bytes = headers[fp8_shard]
    name, descriptor, start, end = _descriptor_range(
        fp8_header,
        header_bytes=fp8_header_bytes,
        shard=fp8_shard,
        preferred_dtypes=("F8_E4M3",),
        desired_bytes=MEDIUM_RANGE_BYTES,
    )
    ranges.append(
        _range_target(
            bindings=bindings,
            metadata=metadata_by_shard[fp8_shard],
            start=start,
            end=end,
            category="medium_contiguous_fp8",
            tensor=name,
            dtype=str(descriptor["dtype"]),
            descriptor=descriptor,
        )
    )
    fp4_shard = CORPUS_SHARDS[2]
    fp4_header, fp4_header_bytes = headers[fp4_shard]
    name, descriptor, start, end = _descriptor_range(
        fp4_header,
        header_bytes=fp4_header_bytes,
        shard=fp4_shard,
        # The public V4 safetensors descriptor represents the packed E2M1FN
        # expert payload as I8; its adjacent E8M0 scales are a separate
        # tensor.  Keep the semantic label explicit rather than misreporting
        # the storage dtype as an ordinary int8 model.
        preferred_dtypes=("I8",),
        desired_bytes=LARGE_FP4_RANGE_BYTES,
        expert_only=True,
    )
    ranges.append(
        _range_target(
            bindings=bindings,
            metadata=metadata_by_shard[fp4_shard],
            start=start,
            end=end,
            category="large_fp4_expert_payload",
            tensor=name,
            dtype=str(descriptor["dtype"]),
            descriptor=descriptor,
        )
    )
    for shard in CORPUS_SHARDS[3:]:
        header, header_bytes = headers[shard]
        name, descriptor, start, end = _descriptor_range(
            header,
            header_bytes=header_bytes,
            shard=shard,
            preferred_dtypes=("F8_E4M3", "I8", "BF16"),
            desired_bytes=MEDIUM_RANGE_BYTES,
        )
        ranges.append(
            _range_target(
                bindings=bindings,
                metadata=metadata_by_shard[shard],
                start=start,
                end=end,
                category="cross_shard_streaming_window",
                tensor=name,
                dtype=str(descriptor["dtype"]),
                descriptor=descriptor,
            )
        )
    corpus = seal(
        {
            "schema": CORPUS_SCHEMA,
            "status": "SEALED_FIXED_PUBLIC_CORPUS",
            "created_at": _utc_now(),
            "source": {"repository": REPOSITORY, "revision": REVISION},
            "ranges": ranges,
            "constraints": {
                "same_bytes_and_hashes_for_every_candidate": True,
                "source_shard_count": len({row["shard"] for row in ranges}),
                "source_body_persisted": False,
                "header_bytes_persisted": False,
                "categories": [row["category"] for row in ranges],
            },
        }
    )
    validate_fixed_corpus(corpus)
    _atomic_json(target, corpus)
    return corpus


def validate_fixed_corpus(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        corpus = verify(value, label="fixed DeepSeek-V4 corpus")
    except SealIntegrityError as exc:
        raise DeepSeekV4XetAutotuneError(str(exc)) from exc
    if corpus.get("schema") != CORPUS_SCHEMA or corpus.get("status") != "SEALED_FIXED_PUBLIC_CORPUS":
        raise DeepSeekV4XetAutotuneError("fixed corpus schema or status is invalid")
    source = corpus.get("source")
    ranges = corpus.get("ranges")
    if source != {"repository": REPOSITORY, "revision": REVISION} or not isinstance(ranges, list):
        raise DeepSeekV4XetAutotuneError("fixed corpus source binding is invalid")
    required_categories = {
        "small_metadata_control",
        "medium_contiguous_fp8",
        "large_fp4_expert_payload",
    }
    categories: set[str] = set()
    identities: set[tuple[str, int, int]] = set()
    shards: set[str] = set()
    for row in ranges:
        if not isinstance(row, Mapping):
            raise DeepSeekV4XetAutotuneError("fixed corpus contains a non-object range")
        shard, start, end = row.get("shard"), row.get("start"), row.get("end")
        length = row.get("length")
        if (
            row.get("repository") != REPOSITORY
            or row.get("revision") != REVISION
            or not isinstance(shard, str)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(length, int)
            or end <= start
            or end - start != length
            or length > MAX_RANGE_BYTES
            or not isinstance(row.get("file_size_bytes"), int)
            or end > int(row["file_size_bytes"])
            or not _is_sha256(row.get("sha256"))
            or not _is_sha256(row.get("lfs_sha256"))
            or not _is_sha256(row.get("xet_file_hash"))
            or row.get("commit_hash") != REVISION
        ):
            raise DeepSeekV4XetAutotuneError("fixed corpus range identity is invalid")
        identity = (shard, start, end)
        if identity in identities:
            raise DeepSeekV4XetAutotuneError("fixed corpus includes a duplicate range")
        identities.add(identity)
        shards.add(shard)
        categories.add(str(row.get("category")))
    if len(shards) < 4 or not required_categories <= categories:
        raise DeepSeekV4XetAutotuneError("fixed corpus does not cover required representative ranges")
    return corpus


def _profile(
    identifier: str,
    *,
    transport: str = "official_hf_xet",
    adaptive: bool | None = False,
    high_performance: bool = False,
    fixed: int | None = None,
    files: int | None = 4,
    shape: str = "dynamic_work_stealing",
    idle: int | None = None,
    ranges: int | None = None,
    adaptive_min: int | None = None,
    adaptive_initial: int | None = None,
    adaptive_max: int | None = None,
    connection_reuse: bool = False,
    retry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if shape not in SCHEDULER_SHAPES:
        raise DeepSeekV4XetAutotuneError(f"unknown scheduler shape {shape}")
    return {
        "id": identifier,
        "transport": transport,
        "adaptive": adaptive,
        "high_performance": high_performance,
        "fixed_download_concurrency": fixed,
        "file_download_concurrency": files,
        "range_get_concurrency": ranges,
        "max_idle_connections": idle,
        "adaptive_min_download_concurrency": adaptive_min,
        "adaptive_initial_download_concurrency": adaptive_initial,
        "adaptive_max_download_concurrency": adaptive_max,
        "connection_reuse": connection_reuse,
        "scheduler_shape": shape,
        "retry": dict(
            retry
            or {
                "max_attempts": 4,
                "base_delay": 0.25,
                "max_duration": 20.0,
                "connect_timeout": 20.0,
                "read_timeout": 120.0,
                "idle_timeout": 30.0,
            }
        ),
    }


def broad_candidates() -> list[dict[str, Any]]:
    """The first-stage non-Cartesian coverage requested by the transfer plan."""

    candidates = [
        _profile(
            "OFFICIAL_ADAPTIVE_HIGH_PERFORMANCE",
            adaptive=True,
            high_performance=True,
            files=4,
            adaptive_min=4,
            adaptive_initial=16,
            adaptive_max=124,
        ),
        _profile(
            "OFFICIAL_ADAPTIVE_SAFE",
            adaptive=True,
            files=4,
            adaptive_min=4,
            adaptive_initial=16,
            adaptive_max=64,
        ),
    ]
    # Sweep fixed client concurrency exactly once each, with staggered file
    # settings.  A second file-only sweep isolates every requested file value
    # without a 9 x 6 Cartesian explosion.
    for position, fixed in enumerate(FIXED_DOWNLOAD_CONCURRENCIES):
        candidates.append(
            _profile(
                f"OFFICIAL_FIXED_{fixed}",
                fixed=fixed,
                files=FILE_DOWNLOAD_CONCURRENCIES[position % len(FILE_DOWNLOAD_CONCURRENCIES)],
            )
        )
    for files in FILE_DOWNLOAD_CONCURRENCIES:
        candidates.append(_profile(f"OFFICIAL_FILES_{files}", fixed=32, files=files))
    candidates.extend(
        [
            _profile("CUSTOM_DIRECT_XET_RANGE", transport="custom_direct_xet_range", files=4),
            _profile("HUB_HTTP_NO_XET", transport="hub_http_without_xet", files=4),
            _profile("DIRECT_PRESIGNED_RANGE", transport="direct_presigned_range", files=4),
            _profile(
                "CUSTOM_DIRECT_XET_RANGE_REUSE",
                transport="custom_direct_xet_range",
                files=4,
                connection_reuse=True,
            ),
            _profile(
                "HUB_HTTP_NO_XET_REUSE",
                transport="hub_http_without_xet",
                files=4,
                connection_reuse=True,
            ),
            _profile(
                "DIRECT_PRESIGNED_RANGE_REUSE",
                transport="direct_presigned_range",
                files=4,
                connection_reuse=True,
            ),
        ]
    )
    return candidates


def _shape_workers(shape: str, profile: Mapping[str, Any]) -> int:
    # Xet's internal concurrency is tested through its own controls.  The
    # outer pipeline is capped at eight source windows to prevent multiplicative
    # memory pressure and to preserve bounded next-range prefetch.
    files = int(profile.get("file_download_concurrency") or 4)
    if shape == "one_file_many_ranges":
        return 1
    if shape == "two_files_medium_range_concurrency":
        return 2
    if shape == "four_files_medium_range_concurrency":
        return 4
    if shape == "eight_files_low_per_file_concurrency":
        return 8
    if shape == "bounded_prefetch_decode_pack_overlap":
        return min(4, max(1, files))
    return min(8, max(1, files))


def _retry_settings(profile: Mapping[str, Any]) -> dict[str, float | int]:
    row = profile.get("retry")
    if not isinstance(row, Mapping):
        raise DeepSeekV4XetAutotuneError("candidate retry policy is invalid")
    try:
        result = {
            "max_attempts": int(row["max_attempts"]),
            "base_delay": float(row["base_delay"]),
            "max_duration": float(row["max_duration"]),
            "connect_timeout": float(row["connect_timeout"]),
            "read_timeout": float(row["read_timeout"]),
            "idle_timeout": float(row["idle_timeout"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise DeepSeekV4XetAutotuneError("candidate retry policy is malformed") from exc
    if result["max_attempts"] < 1 or any(float(value) <= 0 for value in result.values() if isinstance(value, float)):
        raise DeepSeekV4XetAutotuneError("candidate retry policy is outside safe bounds")
    return result


class _RangeTransport:
    def __init__(self, bindings: _SourceBindings, metadata: Mapping[str, Mapping[str, Any]], profile: Mapping[str, Any]) -> None:
        self.bindings = bindings
        self.metadata = metadata
        self.profile = profile
        self.retry = _retry_settings(profile)
        # hf_xet's download stream group is concurrency-safe; sharing one
        # group preserves its connection/adaptive state across the bounded
        # work queue instead of creating a new client per source range.
        self._group = bindings.new_group() if profile.get("transport") == "official_hf_xet" else None
        self._http_local = threading.local()

    def _metadata_for(self, target: Mapping[str, Any]) -> Mapping[str, Any]:
        shard = target.get("shard")
        if not isinstance(shard, str) or shard not in self.metadata:
            raise DeepSeekV4XetAutotuneError("trial target shard is not metadata-bound")
        return self.metadata[shard]

    def fetch(self, target: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
        transport = str(self.profile["transport"])
        metadata = self._metadata_for(target)
        start, end = int(target["start"]), int(target["end"])
        if transport == "official_hf_xet":
            return self._fetch_official(metadata, start, end)
        if transport == "custom_direct_xet_range":
            return self._fetch_direct(metadata, start, end, coalesce=True, hub_resolve=False)
        if transport == "hub_http_without_xet":
            return self._fetch_direct(metadata, start, end, coalesce=False, hub_resolve=True)
        return self._fetch_direct(metadata, start, end, coalesce=False, hub_resolve=False)

    def _fetch_official(
        self, metadata: Mapping[str, Any], start: int, end: int
    ) -> tuple[bytes, dict[str, Any]]:
        group = self._group
        if group is None:
            raise DeepSeekV4XetAutotuneError("official Xet stream group was not initialized")
        info = self.bindings.XetFileInfo(metadata["xet_file_hash"], metadata["file_size_bytes"])
        expected = end - start
        retries = 0
        began = time.monotonic()
        for attempt in range(int(self.retry["max_attempts"])):
            raw = bytearray()
            stream = None
            try:
                stream = group.download_stream(info, start=start, end=end)
                for chunk in stream:
                    if not isinstance(chunk, bytes) or not chunk or len(raw) + len(chunk) > expected:
                        raise DeepSeekV4XetAutotuneError("official hf_xet stream produced invalid range bytes")
                    raw.extend(chunk)
                if len(raw) != expected:
                    raise DeepSeekV4XetAutotuneError(
                        f"official hf_xet range length mismatch: got {len(raw)}, expected {expected}"
                    )
                return bytes(raw), {
                    "attempts": attempt + 1,
                    "retries": retries,
                    "elapsed_seconds": time.monotonic() - began,
                    "host": "hf_xet_managed_connection",
                    "http_status": None,
                    "wire_bytes": len(raw),
                    "wire_bytes_observable": False,
                    "coalesced": False,
                }
            except Exception as exc:
                if attempt + 1 >= int(self.retry["max_attempts"]):
                    raise DeepSeekV4XetAutotuneError(f"official hf_xet range failed: {exc}") from exc
                retries += 1
                time.sleep(min(float(self.retry["max_duration"]), float(self.retry["base_delay"]) * (2**attempt)))
            finally:
                cancel = getattr(stream, "cancel", None)
                if callable(cancel):
                    cancel()
        raise DeepSeekV4XetAutotuneError("official hf_xet range unexpectedly fell through")

    def _fetch_direct(
        self,
        metadata: Mapping[str, Any],
        start: int,
        end: int,
        *,
        coalesce: bool,
        hub_resolve: bool,
    ) -> tuple[bytes, dict[str, Any]]:
        requested = end - start
        fetch_start, fetch_end = start, end
        if coalesce:
            fetch_start = (start // MAX_RANGE_BYTES) * MAX_RANGE_BYTES
            fetch_end = min(int(metadata["file_size_bytes"]), fetch_start + MAX_RANGE_BYTES)
            if not (fetch_start <= start and end <= fetch_end):
                fetch_start, fetch_end = start, end
        if hub_resolve:
            url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{metadata['shard']}"
        else:
            url = str(metadata["signed_location"])
        connections = None
        if bool(self.profile.get("connection_reuse")):
            connections = getattr(self._http_local, "connections", None)
            if connections is None:
                connections = {}
                self._http_local.connections = connections
        raw, result = _http_range(
            url,
            start=fetch_start,
            end=fetch_end,
            file_size=int(metadata["file_size_bytes"]),
            # Public resolve requests and signed storage locations must not
            # receive an unrelated Hub bearer token.
            token=None,
            timeout=float(self.retry["read_timeout"]),
            max_attempts=int(self.retry["max_attempts"]),
            base_delay=float(self.retry["base_delay"]),
            max_delay=float(self.retry["max_duration"]),
            follow_redirects=hub_resolve,
            connections=connections,
        )
        offset = start - fetch_start
        selected = raw[offset : offset + requested]
        if len(selected) != requested:
            raise DeepSeekV4XetAutotuneError("coalesced direct range did not contain requested bytes")
        result.update(
            {
                "wire_bytes_observable": True,
                "coalesced": fetch_start != start or fetch_end != end,
                "fetch_start": fetch_start,
                "fetch_end": fetch_end,
                "logical_bytes": requested,
            }
        )
        return selected, result


def _validate_live_metadata(
    bindings: _SourceBindings, corpus: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    by_shard: dict[str, Mapping[str, Any]] = {}
    source_ranges = corpus["ranges"]
    for target in source_ranges:
        shard = str(target["shard"])
        if shard in by_shard:
            continue
        metadata = bindings.metadata(shard)
        expected = {
            "commit_hash": target["commit_hash"],
            "etag_sha256": target["lfs_sha256"],
            "xet_file_hash": target["xet_file_hash"],
            "file_size_bytes": target["file_size_bytes"],
        }
        observed = {key: metadata[key] for key in expected}
        if observed != expected:
            raise DeepSeekV4XetAutotuneError(
                f"{shard}: current Hub metadata does not match sealed fixed corpus"
            )
        by_shard[shard] = metadata
    return by_shard


def _process_and_evict(
    raw: bytes,
    target: Mapping[str, Any],
    *,
    frames_root: Path,
    workspace: Path,
) -> dict[str, Any]:
    """Verify → decode-check → bounded frame-pack → seal-check → evict."""

    began = time.monotonic()
    expected = target["sha256"]
    digest = _sha256(raw)
    if digest != expected:
        raise DeepSeekV4XetAutotuneError(
            f"{target['shard']}:{target['start']}-{target['end']}: source hash mismatch"
        )
    verified_at = time.monotonic()
    category = str(target["category"])
    # These are bounded native-format structural checks, not a claim that a
    # complete tensor decode has happened.  They deliberately touch every
    # byte, keeping CPU/pack cost visible in the end-to-end objective.
    if category == "large_fp4_expert_payload":
        # crc32 is implemented in C and traverses every source byte without
        # making the Python benchmark itself the accidental bottleneck.
        decode_digest = _sha256(
            f"fp4-e2m1fn-x2:{zlib.crc32(raw):08x}".encode("ascii")
        )
        decode_kind = "native_fp4_e2m1fn_x2_packed_byte_integrity_scan"
    elif category == "medium_contiguous_fp8":
        decode_digest = _sha256(f"fp8-e4m3:{zlib.crc32(raw):08x}".encode("ascii"))
        decode_kind = "native_fp8_e4m3_byte_stream_check"
    else:
        decode_digest = _sha256(f"source-window:{zlib.crc32(raw):08x}".encode("ascii"))
        decode_kind = "source_window_integrity_check"
    decoded_at = time.monotonic()
    # A test frame gives the transfer benchmark a real SSD write, fsync,
    # seal, verification, and eviction stage without creating a persistent
    # cache.  It is a raw framed source window on purpose: a generic zlib pass
    # made the Python test harness the bottleneck and would not be a real
    # Condense codec measurement.
    packed = raw
    _floor_check(workspace, additional_bytes=len(packed), stage="test-frame-pack")
    frame = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.xet_test_frame.v1",
            "source_range_sha256": digest,
            "decoded_structure_sha256": decode_digest,
            "source_bytes": len(raw),
            "packed_bytes": len(packed),
            "packing": "uncompressed_bounded_test_frame_not_full_condense_artifact",
        }
    )
    key = _sha256(
        f"{target['shard']}:{target['start']}:{target['end']}:{digest}:{time.monotonic_ns()}".encode()
    )
    frame_path = frames_root / f"{key}.frame"
    frame_prefix = _canonical(frame) + b"\n"
    _atomic_bytes(frame_path, frame_prefix + packed)
    _regular_file(frame_path, "temporary test frame")
    stored = frame_path.read_bytes()
    if not stored.startswith(frame_prefix) or _sha256(stored[len(frame_prefix) :]) != digest:
        raise DeepSeekV4XetAutotuneError("test frame seal prefix changed after fsync")
    packed_at = time.monotonic()
    try:
        frame_path.unlink()
    except OSError as exc:
        raise DeepSeekV4XetAutotuneError(f"cannot evict test frame: {exc}") from exc
    evicted_at = time.monotonic()
    return {
        "source_bytes": len(raw),
        "packed_bytes": len(packed),
        "verified": True,
        "decode_kind": decode_kind,
        "test_frame_sealed": True,
        "test_frame_evicted": not frame_path.exists(),
        "timings": {
            "verify_seconds": verified_at - began,
            "decode_seconds": decoded_at - verified_at,
            "pack_seal_seconds": packed_at - decoded_at,
            "evict_seconds": evicted_at - packed_at,
        },
    }


def _ordered_jobs(corpus: Mapping[str, Any], *, rounds: int, shape: str) -> list[dict[str, Any]]:
    source = [dict(row) for row in corpus["ranges"]]
    if rounds < 1:
        raise DeepSeekV4XetAutotuneError("trial rounds must be positive")
    jobs: list[dict[str, Any]] = []
    if shape == "one_file_many_ranges":
        ordered = sorted(source, key=lambda row: (row["shard"], row["start"]))
        for _ in range(rounds):
            jobs.extend(dict(row) for row in ordered)
    elif shape == "eight_files_low_per_file_concurrency":
        ordered = sorted(source, key=lambda row: row["shard"])
        for _ in range(rounds):
            jobs.extend(dict(row) for row in ordered)
    else:
        for number in range(rounds):
            # Rotate source order so a permanently slow CDN host cannot win by
            # always being scheduled last.
            offset = number % len(source)
            jobs.extend(dict(row) for row in source[offset:] + source[:offset])
    return jobs


def _fetch_process_one(
    transport: _RangeTransport,
    target: Mapping[str, Any],
    frames_root: Path,
    workspace: Path,
) -> dict[str, Any]:
    before = time.monotonic()
    raw, fetch = transport.fetch(target)
    fetched_at = time.monotonic()
    process = _process_and_evict(raw, target, frames_root=frames_root, workspace=workspace)
    finished = time.monotonic()
    return {
        "target": {
            "category": target["category"],
            "shard": target["shard"],
            "start": target["start"],
            "end": target["end"],
            "sha256": target["sha256"],
        },
        "fetch": fetch,
        "process": process,
        "elapsed_seconds": finished - before,
        "fetch_wall_seconds": fetched_at - before,
    }


def _execute_jobs(
    transport: _RangeTransport,
    jobs: Sequence[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any],
    frames_root: Path,
    workspace: Path,
) -> list[dict[str, Any]]:
    shape = str(profile["scheduler_shape"])
    workers = _shape_workers(shape, profile)
    if shape == "one_file_many_ranges":
        return [_fetch_process_one(transport, row, frames_root, workspace) for row in jobs]
    results: list[dict[str, Any]] = []
    if shape == "bounded_prefetch_decode_pack_overlap":
        # Keep at most one downloaded/processing body per worker.  Fetches run
        # ahead while the oldest completed range is decoded, packed, sealed and
        # evicted, matching the N-1/N/N+1 storage shape without an unbounded
        # producer queue.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending: queue.SimpleQueue[tuple[Mapping[str, Any], Future[tuple[bytes, dict[str, Any]]], float]] = queue.SimpleQueue()
            iterator = iter(jobs)
            active = 0
            for _ in range(workers):
                try:
                    row = next(iterator)
                except StopIteration:
                    break
                pending.put((row, executor.submit(transport.fetch, row), time.monotonic()))
                active += 1
            while active:
                row, future, started = pending.get()
                raw, fetch = future.result()
                fetched_at = time.monotonic()
                process = _process_and_evict(raw, row, frames_root=frames_root, workspace=workspace)
                results.append(
                    {
                        "target": {
                            "category": row["category"],
                            "shard": row["shard"],
                            "start": row["start"],
                            "end": row["end"],
                            "sha256": row["sha256"],
                        },
                        "fetch": fetch,
                        "process": process,
                        "elapsed_seconds": time.monotonic() - started,
                        "fetch_wall_seconds": fetched_at - started,
                    }
                )
                active -= 1
                try:
                    next_row = next(iterator)
                except StopIteration:
                    continue
                pending.put((next_row, executor.submit(transport.fetch, next_row), time.monotonic()))
                active += 1
        return results
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_fetch_process_one, transport, row, frames_root, workspace) for row in jobs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _aggregate_trial(
    *,
    profile: Mapping[str, Any],
    runtime: Mapping[str, Any],
    warmup: Sequence[Mapping[str, Any]],
    steady: Sequence[Mapping[str, Any]],
    elapsed: float,
    host_before: Mapping[str, Any],
    host_after: Mapping[str, Any],
    scratch_root: Path,
) -> dict[str, Any]:
    total = sum(int(row["process"]["source_bytes"]) for row in steady)
    packed = sum(int(row["process"]["packed_bytes"]) for row in steady)
    retries = sum(int(row["fetch"]["retries"]) for row in steady)
    wire_observable = all(bool(row["fetch"].get("wire_bytes_observable", False)) for row in steady)
    wire = sum(int(row["fetch"].get("wire_bytes", row["process"]["source_bytes"])) for row in steady)
    fetch_time = sum(float(row["fetch_wall_seconds"]) for row in steady)
    verify_time = sum(float(row["process"]["timings"]["verify_seconds"]) for row in steady)
    decode_time = sum(float(row["process"]["timings"]["decode_seconds"]) for row in steady)
    pack_time = sum(float(row["process"]["timings"]["pack_seal_seconds"]) for row in steady)
    evict_time = sum(float(row["process"]["timings"]["evict_seconds"]) for row in steady)
    host_distribution = sorted({str(row["fetch"].get("host")) for row in steady})
    stale_files = [str(node.relative_to(scratch_root)) for node in scratch_root.rglob("*") if node.is_file()]
    source_cache_files = [
        item
        for item in stale_files
        if item.startswith("xet-cache/") or item.startswith("hub-cache/models--")
    ]
    if source_cache_files:
        raise DeepSeekV4XetAutotuneError(
            "zero-cache assertion failed; child left source-cache files: " + ", ".join(source_cache_files[:4])
        )
    if any(not bool(row["process"]["test_frame_evicted"]) for row in steady):
        raise DeepSeekV4XetAutotuneError("a benchmark frame survived its seal-before-evict stage")
    rate = total / elapsed if elapsed > 0 else 0.0
    retry_rate = retries / len(steady) if steady else math.inf
    score = rate * max(0.0, 1.0 - min(0.5, retry_rate * 4.0))
    before_counter = (
        host_before.get("network", {}).get("counter")
        if isinstance(host_before.get("network"), Mapping)
        else None
    )
    after_counter = (
        host_after.get("network", {}).get("counter")
        if isinstance(host_after.get("network"), Mapping)
        else None
    )
    interface_delta: dict[str, Any] = {"available": False}
    if isinstance(before_counter, Mapping) and isinstance(after_counter, Mapping):
        try:
            interface_delta = {
                "available": bool(before_counter.get("available")) and bool(after_counter.get("available")),
                "input_bytes": int(after_counter["input_bytes"]) - int(before_counter["input_bytes"]),
                "output_bytes": int(after_counter["output_bytes"]) - int(before_counter["output_bytes"]),
                "aggregate_bytes": int(after_counter["aggregate_bytes"]) - int(before_counter["aggregate_bytes"]),
                "scope": "whole_interface_includes_unrelated_traffic_if_any",
            }
        except (KeyError, TypeError, ValueError):
            interface_delta = {"available": False}
    return {
        "profile": dict(profile),
        "runtime": dict(runtime),
        "warmup": {
            "ranges": len(warmup),
            "logical_bytes": sum(int(row["process"]["source_bytes"]) for row in warmup),
            "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in warmup),
            "excluded_from_steady_state": True,
        },
        "steady_state": {
            "range_count": len(steady),
            "verified_decoded_test_frame_packed_sealed_evicted_bytes": total,
            "test_frame_packed_bytes": packed,
            "elapsed_seconds": elapsed,
            "sealed_and_evicted_bytes_per_second": rate,
            "sealed_and_evicted_mib_per_second": rate / 1024**2,
            "sealed_and_evicted_bytes_per_hour": rate * 3600.0,
            "logical_network_bytes": total,
            "wire_bytes": wire,
            "wire_bytes_observable": wire_observable,
            "wire_amplification_ratio": (wire / total if total else None),
            "retry_count": retries,
            "retry_rate": retry_rate,
            "score_bytes_per_second": score,
            "stage_cpu_seconds_sum": {
                "fetch_wall_sum": fetch_time,
                "verify": verify_time,
                "decode": decode_time,
                "pack_seal": pack_time,
                "evict": evict_time,
            },
            "remote_host_distribution": host_distribution,
            "interface_counter_delta": interface_delta,
            "test_frame_claim_boundary": (
                "benchmark frame exercises bounded pack/seal/evict; it is not a full Condense "
                "or independently validated 43-layer Gravity representation"
            ),
        },
        "storage": {
            "source_body_persisted": False,
            "source_cache_files": source_cache_files,
            "test_frames_remaining": [],
            "protected_floor_bytes": MIN_FREE_FLOOR_BYTES,
            "seal_before_evict": True,
        },
        "host": {"before": dict(host_before), "after": dict(host_after)},
    }


def _host_pressure_violation(before: Mapping[str, Any], after: Mapping[str, Any]) -> str | None:
    """Return a hard local safety reason, never an inferred remote failure."""

    for label, sample in (("before", before), ("after", after)):
        filesystem = sample.get("filesystem") if isinstance(sample, Mapping) else None
        memory = sample.get("memory") if isinstance(sample, Mapping) else None
        thermal = sample.get("thermal") if isinstance(sample, Mapping) else None
        if isinstance(filesystem, Mapping) and int(filesystem.get("free_bytes", 0)) < MIN_FREE_FLOOR_BYTES:
            return f"{label}_storage_floor_below_15GiB"
        if isinstance(memory, Mapping):
            swap = memory.get("swap_used_bytes")
            if isinstance(swap, int) and swap > MAX_SWAP_USED_BYTES:
                return f"{label}_swap_above_safe_ceiling"
        if isinstance(thermal, Mapping) and thermal.get("warning") is True:
            return f"{label}_thermal_warning"
    before_swap = before.get("memory", {}).get("swap_used_bytes") if isinstance(before.get("memory"), Mapping) else None
    after_swap = after.get("memory", {}).get("swap_used_bytes") if isinstance(after.get("memory"), Mapping) else None
    if isinstance(before_swap, int) and isinstance(after_swap, int) and after_swap - before_swap > MAX_SWAP_GROWTH_BYTES:
        return "swap_growth_exceeded_1GiB"
    return None


def run_child_trial(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one already-configured candidate. Called only by child CLI."""

    if spec.get("schema") != "hawking.gravity.deepseek_v4.xet_child_trial_spec.v1":
        raise DeepSeekV4XetAutotuneError("child trial spec schema is invalid")
    profile = spec.get("profile")
    if not isinstance(profile, Mapping):
        raise DeepSeekV4XetAutotuneError("child trial profile is invalid")
    corpus = validate_fixed_corpus(spec.get("corpus", {}))
    workspace = _absolute(spec["workspace"], "workspace")
    scratch_root = _absolute(spec["scratch_root"], "scratch root")
    _ensure_dir(workspace, "workspace")
    _ensure_dir(scratch_root, "scratch root")
    _floor_check(workspace, additional_bytes=MAX_RANGE_BYTES * 3, stage="child-before-trial")
    runtime = _child_runtime_config()
    bindings = _SourceBindings()
    control_started = time.monotonic()
    metadata = _validate_live_metadata(bindings, corpus)
    control_seconds = time.monotonic() - control_started
    transport = _RangeTransport(bindings, metadata, profile)
    frames = scratch_root / "frames"
    _ensure_dir(frames, "temporary test frame root")
    host_before = host_snapshot(workspace)
    warmup_jobs = _ordered_jobs(corpus, rounds=1, shape=str(profile["scheduler_shape"]))
    warmup = _execute_jobs(
        transport, warmup_jobs, profile=profile, frames_root=frames, workspace=workspace
    )
    rounds = int(spec["rounds"])
    steady_jobs = _ordered_jobs(corpus, rounds=rounds, shape=str(profile["scheduler_shape"]))
    began = time.monotonic()
    steady = _execute_jobs(
        transport, steady_jobs, profile=profile, frames_root=frames, workspace=workspace
    )
    elapsed = time.monotonic() - began
    host_after = host_snapshot(workspace)
    violation = _host_pressure_violation(host_before, host_after)
    if violation is not None:
        raise DeepSeekV4XetAutotuneError(f"local pressure guard stopped candidate: {violation}")
    result = _aggregate_trial(
        profile=profile,
        runtime=runtime,
        warmup=warmup,
        steady=steady,
        elapsed=elapsed,
        host_before=host_before,
        host_after=host_after,
        scratch_root=scratch_root,
    )
    return seal(
        {
            "schema": CHILD_RESULT_SCHEMA,
            "status": "PASS",
            "created_at": _utc_now(),
            "source": {"repository": REPOSITORY, "revision": REVISION},
            "corpus_seal_sha256": corpus["seal_sha256"],
            "control_plane": {
                "metadata_dns_tls_control_seconds": control_seconds,
                "metadata_identity_revalidated": True,
            },
            **result,
        }
    )


def _child_config_command() -> dict[str, Any]:
    return _child_runtime_config()


def _child_discover_command(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = _absolute(spec["corpus_path"], "corpus path")
    return discover_fixed_corpus(path)


def _print_child(value: Mapping[str, Any]) -> int:
    sys.stdout.write(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str) + "\n"
    )
    return 0


def _child_main(args: argparse.Namespace) -> int:
    try:
        if args.child_command == "config":
            return _print_child(_child_config_command())
        spec = _read_json(_absolute(args.spec, "child spec"), "child spec")
        if args.child_command == "discover":
            return _print_child(_child_discover_command(spec))
        return _print_child(run_child_trial(spec))
    except (DeepSeekV4XetAutotuneError, SealIntegrityError) as exc:
        sys.stderr.write(f"deepseek-v4-xet-child-error: {exc}\n")
        return 2


def _run_child(
    command: str,
    *,
    profile: Mapping[str, Any],
    scratch_root: Path,
    spec: Mapping[str, Any] | None = None,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    _ensure_dir(scratch_root, "child scratch root")
    spec_path = scratch_root / "child-spec.json"
    if spec is not None:
        _atomic_json(spec_path, spec)
    environment = profile_environment(profile, scratch_root)
    argv = [sys.executable, "-m", "lab.operators.deepseek_v4_xet_autotune", "--child", command]
    if spec is not None:
        argv.extend(("--spec", str(spec_path)))
    try:
        completed = subprocess.run(
            argv,
            cwd=str(REPO_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise DeepSeekV4XetAutotuneError(f"{command} child timed out after {timeout} seconds") from exc
    if completed.returncode != 0:
        error = completed.stderr.strip()[-4000:]
        raise DeepSeekV4XetAutotuneError(f"{command} child failed: {error}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DeepSeekV4XetAutotuneError(f"{command} child did not return one JSON object") from exc
    if not isinstance(value, dict):
        raise DeepSeekV4XetAutotuneError(f"{command} child result is not an object")
    return value


def _runtime_probe(
    profile: Mapping[str, Any], scratch_root: Path
) -> dict[str, Any]:
    value = _run_child("config", profile=profile, scratch_root=scratch_root, timeout=60.0)
    if value.get("schema") != CHILD_CONFIG_SCHEMA or value.get("status") != "PASS":
        raise DeepSeekV4XetAutotuneError("runtime configuration child is invalid")
    return value


def _config_is_unsafe(profile: Mapping[str, Any], runtime: Mapping[str, Any], host: Mapping[str, Any]) -> str | None:
    effective = runtime.get("effective")
    if not isinstance(effective, Mapping):
        return "missing_effective_xet_configuration"
    limit = effective.get("reconstruction.download_buffer_limit")
    available = host.get("filesystem", {}).get("free_bytes") if isinstance(host.get("filesystem"), Mapping) else None
    # 64 GB is the observed high-performance cap in the pinned runtime.  It
    # is an allocation allowance, not proof of immediate allocation, but with
    # existing swap use it is not a responsible blind live-transfer setting.
    if isinstance(limit, int) and limit > 8 * 1024**3:
        return f"effective_reconstruction_buffer_limit_{limit}_exceeds_8GiB_safe_autotune_ceiling"
    if bool(profile.get("high_performance")) and available is not None and int(available) < MIN_FREE_FLOOR_BYTES:
        return "storage_floor_not_available_for_high_performance_profile"
    return None


def _candidate_key(phase: str, profile: Mapping[str, Any], ordinal: int = 0) -> str:
    return f"{phase}:{profile['id']}:{ordinal}"


def _partial_records(path: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("schema") != PARTIAL_LEDGER_SCHEMA:
            raise DeepSeekV4XetAutotuneError("partial ledger schema drift")
        key = row.get("key")
        if not isinstance(key, str):
            raise DeepSeekV4XetAutotuneError("partial ledger row has no key")
        previous = values.get(key)
        if previous is not None and _canonical(previous) != _canonical(row):
            # A previous version of this runner did not yet have a singleton
            # run lock.  Preserve both sealed records in the append-only
            # ledger, but refuse to rank or silently select either one.
            prior_rows = list(previous.get("conflicting_rows", [previous]))
            values[key] = {
                "key": key,
                "value": {
                    "status": "CONFLICTED_CONCURRENT_TRIAL_RECORDS",
                    "key": key,
                    "candidate_rankable": False,
                    "reason": "two non-identical sealed results share one candidate key",
                    "preserved_partial_ledger_records": len(prior_rows) + 1,
                },
                "conflicting_rows": prior_rows + [row],
            }
        else:
            values[key] = row
    return values


def _record_partial(path: Path, key: str, value: Mapping[str, Any]) -> dict[str, Any]:
    record = seal(
        {
            "schema": PARTIAL_LEDGER_SCHEMA,
            "key": key,
            "recorded_at": _utc_now(),
            "value": dict(value),
        }
    )
    _append_jsonl(path, record)
    return record


def _clean_child_scratch(path: Path) -> None:
    """Remove only a freshly-created per-candidate temporary directory."""

    if not path.exists():
        return
    if path.parent.name != "children" or path.is_symlink():
        raise DeepSeekV4XetAutotuneError("refusing to clean a non-candidate scratch directory")
    shutil.rmtree(path)


def _run_candidate(
    *,
    key: str,
    profile: Mapping[str, Any],
    corpus: Mapping[str, Any],
    workspace: Path,
    children_root: Path,
    rounds: int,
    partial_path: Path,
    cached: Mapping[str, Mapping[str, Any]],
    host: Mapping[str, Any],
) -> dict[str, Any]:
    existing = cached.get(key)
    if existing is not None:
        return dict(existing["value"])
    candidate_root = children_root / _sha256(key.encode("utf-8"))[:20]
    _ensure_dir(candidate_root, "per-candidate scratch root")
    try:
        runtime = _runtime_probe(profile, candidate_root / "config")
        unsafe = _config_is_unsafe(profile, runtime, host)
        if unsafe is not None:
            result = {
                "status": "SKIPPED_UNSAFE_CONFIGURATION",
                "key": key,
                "profile": dict(profile),
                "rounds": rounds,
                "runtime": runtime,
                "reason": unsafe,
                "candidate_rankable": False,
            }
        else:
            trial_spec = {
                "schema": "hawking.gravity.deepseek_v4.xet_child_trial_spec.v1",
                "profile": dict(profile),
                "corpus": dict(corpus),
                "workspace": str(workspace),
                "scratch_root": str(candidate_root / "trial"),
                "rounds": rounds,
            }
            trial = _run_child(
                "trial",
                profile=profile,
                scratch_root=candidate_root / "trial",
                spec=trial_spec,
                timeout=max(900.0, rounds * 120.0),
            )
            try:
                verify(trial, label="child trial")
            except SealIntegrityError as exc:
                raise DeepSeekV4XetAutotuneError(str(exc)) from exc
            if trial.get("schema") != CHILD_RESULT_SCHEMA or trial.get("status") != "PASS":
                raise DeepSeekV4XetAutotuneError("child trial returned an invalid result")
            result = {
                "status": "PASS",
                "key": key,
                "profile": dict(profile),
                "rounds": rounds,
                "runtime": runtime,
                "trial": trial,
                "candidate_rankable": True,
            }
        _record_partial(partial_path, key, result)
        return result
    except Exception as exc:
        result = {
            "status": "FAILED",
            "key": key,
            "profile": dict(profile),
            "rounds": rounds,
            "failure": str(exc),
            "candidate_rankable": False,
        }
        _record_partial(partial_path, key, result)
        return result
    finally:
        _clean_child_scratch(candidate_root)


def _git_xet_bounded_record(
    *, phase: str, partial_path: Path, cached: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Receipt the optional Git Xet control without pretending it is available."""

    profile = {
        "id": "GIT_XET_BOUNDED",
        "transport": "git_xet",
        "bounded_range_requirement": True,
    }
    key = _candidate_key(phase, profile)
    existing = cached.get(key)
    if existing is not None:
        return dict(existing["value"])
    version = _run_command(("git", "xet", "version"))
    if version:
        status = "NOT_EXECUTED_REQUIRES_BOUNDED_RANGE_CAPABILITY_REVIEW"
        reason = "git xet command is present but its bounded exact-range behavior is not established"
    else:
        status = "SKIPPED_UNAVAILABLE"
        reason = "git xet is not installed on this Mac"
    result = {
        "status": status,
        "key": key,
        "profile": profile,
        "reason": reason,
        "observed_version": version,
        "candidate_rankable": False,
    }
    _record_partial(partial_path, key, result)
    return result


def _rankable(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        trial = row.get("trial")
        if row.get("status") != "PASS" or not isinstance(trial, Mapping):
            continue
        steady = trial.get("steady_state")
        if not isinstance(steady, Mapping):
            continue
        score = steady.get("score_bytes_per_second")
        rate = steady.get("sealed_and_evicted_bytes_per_second")
        if not isinstance(score, (float, int)) or not isinstance(rate, (float, int)) or score <= 0:
            continue
        result.append(dict(row))
    return sorted(
        result,
        key=lambda row: (
            -float(row["trial"]["steady_state"]["score_bytes_per_second"]),
            float(row["trial"]["steady_state"]["retry_rate"]),
            -float(row["trial"]["steady_state"]["sealed_and_evicted_bytes_per_second"]),
        ),
    )


def _profile_copy(profile: Mapping[str, Any], identifier: str, **changes: Any) -> dict[str, Any]:
    result = dict(profile)
    result["id"] = identifier
    result.update(changes)
    return result


def _best_profile(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    ordered = _rankable(rows)
    return dict(ordered[0]["profile"]) if ordered else None


def _trial_host_contended(row: Mapping[str, Any]) -> bool:
    trial = row.get("trial")
    if not isinstance(trial, Mapping):
        return True
    host = trial.get("host")
    if not isinstance(host, Mapping):
        return True
    for sample in (host.get("before"), host.get("after")):
        if not isinstance(sample, Mapping) or sample.get("contention_observed") is True:
            return True
    return False


def _range_knob_support(
    children_root: Path,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for value in RANGE_GET_CONCURRENCIES:
        profile = _profile(f"RANGE_GET_CONFIG_{value}", fixed=32, ranges=value)
        scratch = children_root / f"range-config-{value}"
        _ensure_dir(scratch, "range-get config scratch")
        try:
            runtime = _runtime_probe(profile, scratch)
            observations.append(
                {
                    "requested": value,
                    "effective": runtime.get("effective"),
                }
            )
        finally:
            _clean_child_scratch(scratch)
    distinct = {
        _canonical(row["effective"]).decode("utf-8")
        for row in observations
    }
    # The pinned runtime does not expose this environment control in XetConfig
    # on this Mac.  Recording the non-effect is stronger than pretending a
    # range-get sweep happened through a hidden knob.
    return {
        "requested_values": list(RANGE_GET_CONCURRENCIES),
        "observations": observations,
        "supported": len(distinct) > 1,
        "status": "EFFECTIVE" if len(distinct) > 1 else "INERT_IN_PINNED_RUNTIME",
    }


def _record_progress(path: Path, event: Mapping[str, Any]) -> None:
    value = seal(
        {
            "schema": PROGRESS_SCHEMA,
            "recorded_at": _utc_now(),
            **dict(event),
        }
    )
    _append_jsonl(path, value)


def _launcher_text(*, root: Path, winner_path: Path, artifact: Path) -> str:
    return f"""#!/bin/zsh
set -euo pipefail
ROOT={root}
PYTHON=\"$ROOT/tools/condense/.venv/bin/python\"
WINNER={winner_path}
ARTIFACT={artifact}
exec \"$PYTHON\" \"$ROOT/tools/condense/deepseek_v4_xet_autotune.py\" resume-real \\
  --winner \"$WINNER\" --artifact-dir \"$ARTIFACT\" --workspace \"$ROOT/workspace\"
"""


def _write_launcher(path: Path, *, winner_path: Path, artifact: Path) -> dict[str, Any]:
    text = _launcher_text(root=REPO_ROOT, winner_path=winner_path, artifact=artifact)
    digest = _atomic_bytes(path, text.encode("utf-8"))
    os.chmod(path, 0o755)
    return {"path": str(path), "sha256": digest, "mode": "0755"}


def _acquire_run_lock(evidence_root: Path) -> Path:
    """Take an O_EXCL lease so a resumed matrix cannot race itself."""

    lock = evidence_root / ".TG_XET_MAXIMUM_PUBLIC_PATH.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise DeepSeekV4XetAutotuneError(
            f"another Xet autotune run owns {lock}; wait for it rather than duplicating trials"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_canonical({"pid": os.getpid(), "created_at": _utc_now()}).decode("utf-8"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return lock


def _release_run_lock(lock: Path) -> None:
    try:
        _regular_file(lock, "autotune singleton lease")
        lock.unlink()
    except FileNotFoundError:
        return


def _run_autotune(
    *,
    workspace: str | Path,
    evidence_root: str | Path,
    run_root: str | Path,
    short_rounds: int,
    long_rounds: int,
    confirmation_rounds: int,
) -> dict[str, Any]:
    """Complete the public-path matrix, freeze a verified winner, and emit hand-off files."""

    workspace_path = _absolute(workspace, "workspace")
    evidence = _absolute(evidence_root, "evidence root")
    runs = _absolute(run_root, "run root")
    _ensure_dir(workspace_path, "workspace")
    _ensure_dir(evidence, "evidence root")
    _ensure_dir(runs, "run root")
    _floor_check(workspace_path, additional_bytes=MAX_RANGE_BYTES * 8, stage="autotune-admission")
    paths = {
        "corpus": evidence / CORPUS_NAME,
        "partial": evidence / PARTIAL_NAME,
        "matrix": evidence / MATRIX_NAME,
        "winner": evidence / WINNER_NAME,
        "roofline": evidence / ROOFLINE_NAME,
        "progress": runs / PROGRESS_NAME,
        "launcher": runs / LAUNCHER_NAME,
    }
    if paths["matrix"].exists() or paths["winner"].exists() or paths["roofline"].exists():
        # These are immutable promotional records.  A restart resumes from the
        # partial ledger until all three exist; a completed run is inspected,
        # not overwritten with a new claim.
        if all(path.exists() for key, path in paths.items() if key in {"matrix", "winner", "roofline"}):
            return {
                "status": "ALREADY_COMPLETE",
                "matrix": _read_json(paths["matrix"], "matrix"),
                "winner": _read_json(paths["winner"], "winner"),
                "roofline": _read_json(paths["roofline"], "roofline"),
            }
        raise DeepSeekV4XetAutotuneError("incomplete immutable output set; inspect before manual recovery")
    children = evidence / "children"
    _ensure_dir(children, "autotune children root")
    host_before = host_snapshot(workspace_path)
    baseline_profile = _profile("DISCOVERY_DIRECT", transport="direct_presigned_range", files=1)
    corpus_spec = {"corpus_path": str(paths["corpus"])}
    if paths["corpus"].exists():
        corpus = validate_fixed_corpus(_read_json(paths["corpus"], "fixed corpus"))
    else:
        discovery = children / "discovery"
        _ensure_dir(discovery, "corpus discovery scratch")
        try:
            corpus = _run_child(
                "discover", profile=baseline_profile, scratch_root=discovery, spec=corpus_spec, timeout=900.0
            )
            corpus = validate_fixed_corpus(corpus)
        finally:
            _clean_child_scratch(discovery)
    range_support = _range_knob_support(children)
    partial = _partial_records(paths["partial"])
    # v3 adds direct per-worker keep-alive reuse.  Keep the earlier completed
    # v2 records append-only as preliminary evidence, then run a clean,
    # non-overlapping matrix rather than mixing implementation generations.
    short_phase = "short_v3_connection_reuse"
    phase_one: list[dict[str, Any]] = [
        _git_xet_bounded_record(
            phase=short_phase, partial_path=paths["partial"], cached=partial
        )
    ]
    partial = _partial_records(paths["partial"])
    for profile in broad_candidates():
        key = _candidate_key(short_phase, profile)
        phase_one.append(
            _run_candidate(
                key=key,
                profile=profile,
                corpus=corpus,
                workspace=workspace_path,
                children_root=children,
                rounds=short_rounds,
                partial_path=paths["partial"],
                cached=partial,
                host=host_before,
            )
        )
        partial = _partial_records(paths["partial"])
    ranked_short = _rankable(phase_one)
    retained_count = max(2, math.ceil(len(ranked_short) / 4)) if ranked_short else 0
    retained = ranked_short[:retained_count]
    phase_two: list[dict[str, Any]] = []
    for row in retained:
        profile = dict(row["profile"])
        profile["id"] = f"{profile['id']}_LONG"
        key = _candidate_key("long_v3", profile)
        phase_two.append(
            _run_candidate(
                key=key,
                profile=profile,
                corpus=corpus,
                workspace=workspace_path,
                children_root=children,
                rounds=long_rounds,
                partial_path=paths["partial"],
                cached=partial,
                host=host_before,
            )
        )
        partial = _partial_records(paths["partial"])
    base = _best_profile(phase_two) or _best_profile(phase_one)
    scheduler_trials: list[dict[str, Any]] = []
    idle_trials: list[dict[str, Any]] = []
    if base is not None:
        for shape in SCHEDULER_SHAPES:
            profile = _profile_copy(base, f"{base['id']}_SHAPE_{shape}", scheduler_shape=shape)
            key = _candidate_key("shape_v3", profile)
            scheduler_trials.append(
                _run_candidate(
                    key=key,
                    profile=profile,
                    corpus=corpus,
                    workspace=workspace_path,
                    children_root=children,
                    rounds=long_rounds,
                    partial_path=paths["partial"],
                    cached=partial,
                    host=host_before,
                )
            )
            partial = _partial_records(paths["partial"])
        for idle in IDLE_CONNECTIONS:
            profile = _profile_copy(base, f"{base['id']}_IDLE_{idle}", max_idle_connections=idle)
            key = _candidate_key("idle_v3", profile)
            idle_trials.append(
                _run_candidate(
                    key=key,
                    profile=profile,
                    corpus=corpus,
                    workspace=workspace_path,
                    children_root=children,
                    rounds=max(2, long_rounds // 2),
                    partial_path=paths["partial"],
                    cached=partial,
                    host=host_before,
                )
            )
            partial = _partial_records(paths["partial"])
    all_preconfirm = phase_two + scheduler_trials + idle_trials + phase_one
    top_two = _rankable(all_preconfirm)[:2]
    confirmations: list[dict[str, Any]] = []
    for row in top_two:
        for pass_index in (1, 2):
            profile = _profile_copy(
                row["profile"], f"{row['profile']['id']}_CONFIRM_{pass_index}"
            )
            key = _candidate_key("confirmation_v3", profile, pass_index)
            confirmations.append(
                _run_candidate(
                    key=key,
                    profile=profile,
                    corpus=corpus,
                    workspace=workspace_path,
                    children_root=children,
                    rounds=confirmation_rounds,
                    partial_path=paths["partial"],
                    cached=partial,
                    host=host_before,
                )
            )
            partial = _partial_records(paths["partial"])
    # Retain a profile only if both sustained confirmations succeeded.  Strip
    # the confirmation suffix when grouping by its actual frozen environment.
    confirmations_by_base: dict[str, list[dict[str, Any]]] = {}
    for row in confirmations:
        profile = row.get("profile")
        if not isinstance(profile, Mapping):
            continue
        identifier = str(profile["id"])
        base_identifier = identifier.rsplit("_CONFIRM_", 1)[0]
        confirmations_by_base.setdefault(base_identifier, []).append(row)
    promoted: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []
    for base_identifier, rows in confirmations_by_base.items():
        passed = _rankable(rows)
        if len(passed) != 2:
            continue
        if any(_trial_host_contended(row) for row in passed):
            continue
        rates = [float(row["trial"]["steady_state"]["score_bytes_per_second"]) for row in passed]
        # A 10% spread is deliberately modest: we do not promote a peak that
        # only appeared in one confirmation run.
        if min(rates) <= 0 or max(rates) / min(rates) > 1.10:
            continue
        promoted.append((sum(rates) / len(rates), passed[0], rows))
    promoted.sort(key=lambda item: item[0], reverse=True)
    winner_profile: dict[str, Any] | None = None
    winner_confirmations: list[dict[str, Any]] = []
    if promoted:
        _, chosen, winner_confirmations = promoted[0]
        winner_profile = dict(chosen["profile"])
        winner_profile["id"] = str(winner_profile["id"]).rsplit("_CONFIRM_", 1)[0]
    host_after = host_snapshot(workspace_path)
    retry_evidence = [
        row
        for row in _rankable(all_preconfirm + confirmations)
        if float(row["trial"]["steady_state"]["retry_rate"]) > 0
    ]
    buffer_tuning = {
        "status": "NOT_TESTED_NO_EVIDENCE" if not retry_evidence else "DEFERRED_TO_SAFE_MANUAL_REVIEW",
        "reason": (
            "no retry/error regression justified changing reconstruction buffers"
            if not retry_evidence
            else "retry evidence exists, but high-performance buffer caps exceed safe host reserve"
        ),
        "knobs": [
            "HF_XET_RECONSTRUCTION_MIN_RECONSTRUCTION_FETCH_SIZE",
            "HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE",
            "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE",
            "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE",
            "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT",
        ],
    }
    matrix = seal(
        {
            "schema": MATRIX_SCHEMA,
            "status": "COMPLETE" if winner_profile is not None else "COMPLETE_NO_STABLE_PROMOTION",
            "created_at": _utc_now(),
            "endpoint": "DEEPSEEK_V4_MAX_PUBLIC_STREAM_ACTIVE",
            "source": {"repository": REPOSITORY, "revision": REVISION},
            "fixed_corpus": {
                "path": str(paths["corpus"]),
                "seal_sha256": corpus["seal_sha256"],
                "range_count": len(corpus["ranges"]),
                "bytes_per_round": sum(int(row["length"]) for row in corpus["ranges"]),
            },
            "software_and_host": {"before": host_before, "after": host_after},
            "control_support": {
                "adaptive_high_performance": "CONFIGURED_AND_SAFETY_CLASSIFIED",
                "fixed_download_concurrency_tested": list(FIXED_DOWNLOAD_CONCURRENCIES),
                "file_download_concurrency_tested": list(FILE_DOWNLOAD_CONCURRENCIES),
                "range_get_concurrency": range_support,
                "xet_write_policy": {
                    "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY": "false",
                    "HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE": "true",
                    "HF_XET_CHUNK_CACHE_SIZE_BYTES": 0,
                },
                "buffer_tuning": buffer_tuning,
            },
            "phases": {
                "short_broad": phase_one,
                "long_top_quarter": phase_two,
                "scheduler_shapes": scheduler_trials,
                "idle_connection_tuning": idle_trials,
                "two_sustained_confirmations_for_top_two": confirmations,
            },
            "selection": {
                "short_rankable_count": len(ranked_short),
                "top_quarter_retained_count": retained_count,
                "top_two_candidate_count": len(top_two),
                "winner_profile": winner_profile,
                "winner_confirmation_count": len(winner_confirmations),
                "promotion_requires_two_stable_confirmations": True,
                "retry_tuning": (
                    "baseline retry policy retained; no error regression"
                    if not retry_evidence
                    else "error evidence observed; result records include bounded retry policy"
                ),
            },
            "claim_boundary": {
                "public_path_ceiling_scope": "this Mac, current public route, fixed pinned corpus, recorded time window",
                "benchmark_test_frame_is_full_condense_pack": False,
                "source_parent_persisted": False,
                "all_candidate_source_ranges_verified": True,
            },
        }
    )
    _atomic_json(paths["matrix"], matrix)
    winner = seal(
        {
            "schema": WINNER_SCHEMA,
            "status": "FROZEN" if winner_profile is not None else "NO_WINNER_PROMOTED",
            "created_at": _utc_now(),
            "endpoint": "DEEPSEEK_V4_MAX_PUBLIC_STREAM_ACTIVE",
            "matrix_path": str(paths["matrix"]),
            "matrix_seal_sha256": matrix["seal_sha256"],
            "source": {"repository": REPOSITORY, "revision": REVISION},
            "profile": winner_profile,
            "confirmation_records": winner_confirmations,
            "real_stream_application": {
                "outer_source_windows_maximum": (
                    _shape_workers(str(winner_profile["scheduler_shape"]), winner_profile)
                    if winner_profile is not None
                    else 0
                ),
                "must_not_multiply_outer_workers_by_hf_xet_internal_concurrency": True,
                "xet_environment_set_before_import": True,
                "source_cache_bytes": 0,
                "protected_floor_bytes": MIN_FREE_FLOOR_BYTES,
            },
        }
    )
    _atomic_json(paths["winner"], winner)
    selected_trial = winner_confirmations[0].get("trial") if winner_confirmations else None
    steady = selected_trial.get("steady_state") if isinstance(selected_trial, Mapping) else None
    measured_mib = float(steady["sealed_and_evicted_mib_per_second"]) if isinstance(steady, Mapping) else 0.0
    link_media = host_after.get("network", {}).get("media") if isinstance(host_after.get("network"), Mapping) else None
    binder = "no_promoted_candidate"
    if isinstance(steady, Mapping):
        stage = steady.get("stage_cpu_seconds_sum", {})
        if isinstance(stage, Mapping):
            maximum = max(stage.items(), key=lambda item: float(item[1]))[0]
            binder = {
                "fetch_wall_sum": "public_network_or_remote_path",
                "verify": "verification_cpu",
                "decode": "native_decode_cpu",
                "pack_seal": "disk_or_pack_stage",
                "evict": "filesystem_eviction_stage",
            }.get(maximum, maximum)
    roofline = seal(
        {
            "schema": ROOFLINE_SCHEMA,
            "status": "MEASURED" if winner_profile is not None else "NO_STABLE_MEASUREMENT",
            "created_at": _utc_now(),
            "endpoint": "DEEPSEEK_V4_MAX_PUBLIC_STREAM_ACTIVE",
            "matrix_seal_sha256": matrix["seal_sha256"],
            "winner_seal_sha256": winner["seal_sha256"],
            "interface": host_after.get("network"),
            "negotiated_link_media": link_media,
            "measured_public_path": {
                "fastest_stable_sealed_and_evicted_mib_per_second": measured_mib,
                "fastest_stable_sealed_and_evicted_bytes_per_hour": (
                    float(steady["sealed_and_evicted_bytes_per_hour"]) if isinstance(steady, Mapping) else 0.0
                ),
                "binding": binder,
                "link_speed_is_not_wan_throughput": True,
                "historical_90_mib_per_second_comparison": (
                    "materially_beats_90_MiB_per_second"
                    if measured_mib > 99.0
                    else "does_not_materially_beat_90_MiB_per_second"
                ),
            },
            "ceiling_evidence": {
                "two_sustained_confirmations": len(winner_confirmations),
                "same_fixed_byte_corpus": True,
                "remote_hosts": steady.get("remote_host_distribution") if isinstance(steady, Mapping) else [],
                "retry_rate": steady.get("retry_rate") if isinstance(steady, Mapping) else None,
                "host_contention_observed": host_after.get("contention_observed"),
                "scope": "measured current public-path ceiling, not a claim about 10GbE LAN capability",
            },
        }
    )
    _atomic_json(paths["roofline"], roofline)
    artifact = runs / "full-43-layer-stream.gravity"
    launcher = _write_launcher(paths["launcher"], winner_path=paths["winner"], artifact=artifact)
    _record_progress(
        paths["progress"],
        {
            "event": "PUBLIC_PATH_AUTOTUNE_COMPLETE",
            "matrix_seal_sha256": matrix["seal_sha256"],
            "winner_seal_sha256": winner["seal_sha256"],
            "roofline_seal_sha256": roofline["seal_sha256"],
            "status": winner["status"],
            "fastest_stable_mib_per_second": measured_mib,
        },
    )
    return {
        "status": matrix["status"],
        "matrix": matrix,
        "winner": winner,
        "roofline": roofline,
        "launcher": launcher,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def resume_real_stream(
    *, winner_path: str | Path, artifact_dir: str | Path, workspace: str | Path
) -> dict[str, Any]:
    """Apply the frozen winner only where a safe source window remains to run."""

    winner_file = _absolute(winner_path, "winner path")
    artifact = _absolute(artifact_dir, "artifact directory")
    workspace_path = _absolute(workspace, "workspace")
    winner = _read_json(winner_file, "winner")
    try:
        verify(winner, label="frozen Xet winner")
    except SealIntegrityError as exc:
        raise DeepSeekV4XetAutotuneError(str(exc)) from exc
    if winner.get("schema") != WINNER_SCHEMA or winner.get("status") != "FROZEN":
        raise DeepSeekV4XetAutotuneError("there is no frozen stable public-path winner")
    profile = winner.get("profile")
    if not isinstance(profile, Mapping):
        raise DeepSeekV4XetAutotuneError("frozen winner lacks a profile")
    manifest_path = artifact / "manifest.json"
    progress = DEFAULT_RUN_DIR / PROGRESS_NAME
    if manifest_path.exists():
        manifest = _read_json(manifest_path, "full stream manifest")
        status = manifest.get("status")
        if status == "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY":
            event = {
                "event": "REAL_STREAM_RESUME_NOT_REDIRECTED",
                "status": "SEALED_SOURCE_WINDOWS_RETAINED_PENDING_INDEPENDENT_SUCCESSOR",
                "reason": (
                    "the only complete V4 source artifact is already sealed; re-downloading it would "
                    "violate no-parent-accumulation and cannot be presented as fresh work"
                ),
                "winner_seal_sha256": winner["seal_sha256"],
                "artifact_manifest_seal_sha256": manifest.get("seal_sha256"),
                "source_parent_eviction": "blocked until independently sealed successor exists",
            }
            _record_progress(progress, event)
            return event
    _floor_check(workspace_path, additional_bytes=MAX_RANGE_BYTES * 8, stage="real-stream-resume-preflight")
    event = {
        "event": "REAL_STREAM_RESUME_PRECHECK",
        "status": "READY_FOR_CONFIGURED_SOURCE_WINDOWS",
        "winner_seal_sha256": winner["seal_sha256"],
        "profile": dict(profile),
        "artifact": str(artifact),
        "note": (
            "A new incomplete stream requires a route-aware Gravity processor before body work; this "
            "command will not bypass the sealed-artifact successor/eviction invariant."
        ),
    }
    _record_progress(progress, event)
    return event


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command")
    run = subcommands.add_parser("run", help="execute the full public-path Xet matrix")
    run.add_argument("--workspace", type=Path, default=REPO_ROOT / "workspace")
    run.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_DIR)
    run.add_argument("--run-root", type=Path, default=DEFAULT_RUN_DIR)
    run.add_argument("--short-rounds", type=int, default=4)
    run.add_argument("--long-rounds", type=int, default=12)
    run.add_argument("--confirmation-rounds", type=int, default=32)
    resume = subcommands.add_parser("resume-real", help="safely hand frozen Xet result to real stream")
    resume.add_argument("--winner", type=Path, default=DEFAULT_EVIDENCE_DIR / WINNER_NAME)
    resume.add_argument("--artifact-dir", type=Path, required=True)
    resume.add_argument("--workspace", type=Path, default=REPO_ROOT / "workspace")
    parser.add_argument("--child", dest="child_command", choices=("config", "discover", "trial"))
    parser.add_argument("--spec", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.child_command:
        if args.child_command in {"discover", "trial"} and args.spec is None:
            raise SystemExit("--spec is required for this child command")
        return _child_main(args)
    try:
        if args.command == "run":
            if min(args.short_rounds, args.long_rounds, args.confirmation_rounds) < 1:
                raise DeepSeekV4XetAutotuneError("all trial round counts must be positive")
            evidence_root = _absolute(args.evidence_root, "evidence root")
            _ensure_dir(evidence_root, "evidence root")
            lock = _acquire_run_lock(evidence_root)
            try:
                result = _run_autotune(
                    workspace=args.workspace,
                    evidence_root=evidence_root,
                    run_root=args.run_root,
                    short_rounds=args.short_rounds,
                    long_rounds=args.long_rounds,
                    confirmation_rounds=args.confirmation_rounds,
                )
            finally:
                _release_run_lock(lock)
        elif args.command == "resume-real":
            result = resume_real_stream(
                winner_path=args.winner, artifact_dir=args.artifact_dir, workspace=args.workspace
            )
        else:
            raise DeepSeekV4XetAutotuneError("a command is required")
    except (DeepSeekV4XetAutotuneError, SealIntegrityError) as exc:
        sys.stderr.write(f"deepseek-v4-xet-autotune-error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
