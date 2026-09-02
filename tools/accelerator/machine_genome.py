"""MachineGenome — the physical identity of THIS machine. FRONT A (G043, S015 §22).

The steer is explicit that a MachineGenome must carry measured bandwidth, not a
datasheet number, and that all Apple Silicon must not be assumed to behave alike.
So everything here is either read from the machine or measured on it. Fields that
cannot be obtained are ABSENT with a reason; none is guessed.

Bandwidth is measured inside a protected window because the lake fill saturates
disk and network, and a contended sample is not a roof. Repeats are alternated and
the spread is reported alongside the number -- a tight spread is itself evidence,
and a wide one means the number is not yet a measurement.

Architecture (v2, still a superset of the v1 identity bag):
  A genome is a dict of DOMAINS, not a product SKU. CPU / GPU / UMA / ANE /
  STORAGE / NETWORK are filled from THIS machine. FPGA and EXTERNAL_ACCELERATOR
  (DGX, eGPU, U50, ...) are declared slots: present=False here, and a future
  board fills the same record without a schema change. `declare_domain()` is
  the slot-in surface.

  Evidence tiers are never merged: STATIC / FUNCTIONAL_SIM / COST_MODEL /
  CYCLE_APPROX / HARDWARE_MEASURED. A probe that cannot run is BLOCKED with
  a reason, never a fabricated number.

  Storage is part of the genome. ModelLake and Odyssey already showed that
  the mount, not just the SoC, changes science economics. Probes are
  read-only on /Volumes/corpdrive (live hf download workers own that volume).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
# v2 is a superset of the v1 identity bag (soc, cpu_cores, gpu_cores,
# memory_bytes, measured_bandwidth, ...). fusion_bridge.domain_from_machine_genome
# reads memory_bytes and does not inspect schema.
SCHEMA = "hawking.accelerator.machine_genome.v2"
SCHEMA_V1 = "hawking.accelerator.machine_genome.v1"

# Honest evidence labels. Never merge. BLOCKED is a status, not a tier.
EVIDENCE_TIERS = (
    "STATIC",
    "FUNCTIONAL_SIM",
    "COST_MODEL",
    "CYCLE_APPROX",
    "HARDWARE_MEASURED",
)
MATURITY = (
    "ABSENT",
    "DECLARED",
    "PRESENT",
    "PROFILED",
    "MEASURED",
    "QUALIFIED",
)
# Vocabulary, not a closed schema. Extra kinds are legal — that is how a
# future FPGA / DGX / eGPU slots in without a rewrite.
KNOWN_DOMAIN_KINDS = (
    "CPU",
    "GPU",
    "UMA",
    "ANE",
    "STORAGE",
    "NETWORK",
    "FPGA",
    "EXTERNAL_ACCELERATOR",
)

# Cross-cutting axes the Hardware Doctor asks per device. Status is
# orthogonal to evidence_tier: a STATIC brochure number is never a
# HARDWARE_MEASURED bandwidth, even if both sit on the same domain record.
AXES = (
    "presence",
    "identity",
    "capacity",
    "bandwidth",
    "latency",
    "energy",
    "thermal",
    "placement",
    "transport",
    "backend",
)
AXIS_STATUSES = (
    "MEASURED",
    "MODELLED",
    "UNKNOWN",
    "BLOCKED",
    "ABSENT",
    "UNRELIABLE",
    "STATIC_IDENTITY",
)

CORPDRIVE = Path("/Volumes/corpdrive")
# Live hf download workers write here. Genome probes never create, truncate,
# or write any path under this prefix.
NO_WRITE_PREFIXES = (str(CORPDRIVE),)
# Identity fields a lowering actually depends on. Free space and sequential
# rates are live readings and are NOT in the digest — a lake fill must not
# look like a different machine.
DIGEST_IDENTITY_KEYS = (
    "soc", "arch", "cpu_cores", "perf_cores", "efficiency_cores",
    "gpu_cores", "memory_bytes",
)

ANE_LAB_CANDIDATES = (
    REPO / "receipts/headless/HCLI_FORBIDDEN_FRUIT_LAB.json",
    Path("/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FORBIDDEN_FRUIT_LAB.json"),
    REPO / "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json",
)
ANE_GIT_FALLBACK = "HEAD:receipts/headless/APPLE_ANE_DEVICE_PROFILE.json"

# Bounded sequential sample. 64 MiB is long enough to clear a 5ms reliability
# gate at ~10 GB/s and small enough not to contend with live lake fills.
SEQ_SAMPLE_BYTES = 64 << 20
RANDOM_READS = 32
RANDOM_BLOCK = 4096


def _sysctl(key: str) -> str | None:
    try:
        r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gpu_cores() -> Any:
    """ioreg gpu-core-count is a fused-in identity property; system_profiler
    is the slower fallback the v1 genome used."""
    out = _run(["ioreg", "-r", "-d", "1", "-c", "AGXAccelerator", "-w", "0"], timeout=8.0)
    if out:
        m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
        if m:
            return int(m.group(1))
    try:
        r = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                           capture_output=True, text=True, timeout=90)
        m = re.search(r"Total Number of Cores:\s*(\d+)", r.stdout)
        if m:
            return int(m.group(1))
        return {"status": "ABSENT", "reason": "system_profiler reported no core count"}
    except Exception as e:
        return {"status": "ABSENT", "reason": f"system_profiler failed: {type(e).__name__}"}


def _toolchain() -> dict[str, Any]:
    out: dict[str, Any] = {}
    absent_metal = {
        "status": "ABSENT",
        "reason": "xcrun metal is not installed; no Metal developer toolchain, so "
                  "AOT metallib compilation is unavailable and kernels go through "
                  "the MLX JIT"}
    try:
        r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"],
                           capture_output=True, text=True, timeout=30)
        # subprocess does not raise on a non-zero exit, so without this check the
        # xcrun ERROR TEXT got stored as if it were a compiler version.
        out["metal_compiler"] = (r.stdout.strip().splitlines()[0]
                                 if r.returncode == 0 and r.stdout.strip()
                                 else absent_metal)
    except Exception:
        out["metal_compiler"] = absent_metal
    try:
        import mlx.core as mx
        out["mlx"] = getattr(mx, "__version__", "unknown")
    except Exception:
        out["mlx"] = {"status": "ABSENT", "reason": "mlx not importable in this interpreter"}
    out["python"] = sys.version.split()[0]
    return out


def measure_bandwidth(n: int = 1 << 26, reps: int = 30, warmup: int = 8) -> dict[str, Any]:
    """Streaming triad on the GPU: reads 2N f32, writes N f32 => 12N bytes moved.

    A first attempt at this reported best 403 GB/s with a 286% spread across reps,
    which is not a measurement -- it is a distribution with a fast tail. Larger
    buffers, a real warmup and an interquartile spread replace it. The number is
    reported ONLY if the IQR is tight; otherwise it is marked UNRELIABLE rather
    than quoted, because a wide spread means the machine was not actually held
    still.
    """
    try:
        import mlx.core as mx
    except Exception as e:
        return {
            "status": "ABSENT",
            "reason": f"mlx unavailable: {type(e).__name__}",
            "evidence_tier": "STATIC",
        }
    a = mx.random.normal((n,), dtype=mx.float32)
    b = mx.random.normal((n,), dtype=mx.float32)
    mx.eval(a, b)
    bytes_moved = 12 * n
    for _ in range(warmup):                    # compile + clock ramp
        mx.eval(a + b)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(a + b)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    q1 = samples[len(samples) // 4]
    med = samples[len(samples) // 2]
    q3 = samples[(3 * len(samples)) // 4]
    iqr_pct = round((q3 - q1) / q1 * 100, 2)
    reliable = iqr_pct <= 10.0
    out = {
        "pattern": "triad c = a + b, f32",
        "elements": n,
        "bytes_moved_per_rep": bytes_moved,
        "reps": reps, "warmup": warmup,
        "median_gb_s": round(bytes_moved / med / 1e9, 2),
        "q1_gb_s": round(bytes_moved / q3 / 1e9, 2),
        "q3_gb_s": round(bytes_moved / q1 / 1e9, 2),
        "iqr_spread_pct": iqr_pct,
        "full_range_spread_pct": round((samples[-1] - samples[0]) / samples[0] * 100, 2),
        "reliable": reliable,
        "is_theoretical_roof": False,
        "evidence_tier": "HARDWARE_MEASURED",
        "note": "one access pattern on one dtype; not the SoC roof and not a "
                "workload-reachable roof",
    }
    if not reliable:
        out["status"] = "UNRELIABLE"
        out["reason"] = (f"interquartile spread {iqr_pct}% exceeds the 10% gate; the "
                         "machine was not held still enough for this to be a roof")
    return out


def _no_write(path: Path) -> bool:
    resolved = str(path.resolve()) if path.exists() else str(path)
    for prefix in NO_WRITE_PREFIXES:
        if resolved == prefix or resolved.startswith(prefix + os.sep):
            return True
    return False


def _statvfs(path: Path) -> dict[str, Any] | None:
    try:
        s = os.statvfs(path)
    except OSError as e:
        return {
            "status": "BLOCKED",
            "reason": f"statvfs({path}): {e}",
            "evidence_tier": "STATIC",
        }
    return {
        "capacity_bytes": int(s.f_frsize * s.f_blocks),
        "free_bytes": int(s.f_frsize * s.f_bavail),
        "frsize": int(s.f_frsize),
        "evidence_tier": "HARDWARE_MEASURED",
    }


def _parse_mounts() -> list[dict[str, Any]]:
    """Live `mount` table. Identity of what is attached, not a rate."""
    raw = _run(["mount"], timeout=5.0) or ""
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        # /dev/disk5s1 on /Volumes/corpdrive (apfs, local, nodev, nosuid, journaled)
        m = re.match(r"^(\S+) on (\S+) \(([^)]+)\)", line)
        if not m:
            continue
        device, mountpoint, opts = m.group(1), m.group(2), m.group(3)
        parts = [p.strip() for p in opts.split(",")]
        rows.append({
            "device": device,
            "mount": mountpoint,
            "fstype": parts[0] if parts else None,
            "options": parts[1:],
            "read_only": "read-only" in parts,
            "evidence_tier": "STATIC",
        })
    return rows


def _sequential_read(path: Path, nbytes: int = SEQ_SAMPLE_BYTES) -> dict[str, Any]:
    """Read-only sequential sample. Never used as a write target."""
    try:
        size = path.stat().st_size
    except OSError as e:
        return {"status": "BLOCKED", "reason": f"stat {path}: {e}", "evidence_tier": "STATIC"}
    want = min(nbytes, size)
    if want <= 0:
        return {"status": "BLOCKED", "reason": f"{path} is empty", "evidence_tier": "STATIC"}
    samples = []
    try:
        for _ in range(2):
            got = 0
            t0 = time.perf_counter()
            with open(path, "rb") as f:
                while got < want:
                    chunk = f.read(min(1 << 20, want - got))
                    if not chunk:
                        break
                    got += len(chunk)
            dt = time.perf_counter() - t0
            samples.append((got, dt))
    except OSError as e:
        return {"status": "BLOCKED", "reason": f"read {path}: {e}", "evidence_tier": "STATIC"}
    cold_bytes, cold_dt = samples[0]
    warm_bytes, warm_dt = samples[-1]
    def _gb_s(n: int, dt: float) -> float | None:
        if dt <= 0:
            return None
        return round(n / dt / 1e9, 3)
    cold = _gb_s(cold_bytes, cold_dt)
    warm = _gb_s(warm_bytes, warm_dt)
    reliable = cold_dt >= 0.005 and cold is not None
    out: dict[str, Any] = {
        "path": str(path),
        "bytes_sampled": cold_bytes,
        "cold_s": round(cold_dt, 4),
        "warm_s": round(warm_dt, 4),
        "cold_gb_s": cold,
        "warm_gb_s": warm,
        "cache_speedup": (round(cold_dt / warm_dt, 2) if warm_dt > 0 else None),
        "reliable": reliable,
        "evidence_tier": "HARDWARE_MEASURED",
        "note": "bounded sequential read of an existing file; not a disk roof",
    }
    if not reliable:
        out["status"] = "UNRELIABLE"
        out["reason"] = (
            f"sample elapsed {cold_dt:.4f}s; a sub-5ms window is not a rate"
        )
    return out


def _sequential_write_read(dirpath: Path, nbytes: int = SEQ_SAMPLE_BYTES) -> dict[str, Any]:
    if _no_write(dirpath):
        return {
            "status": "BLOCKED",
            "reason": f"write probe refused on {dirpath} (live lake / protected volume)",
            "evidence_tier": "STATIC",
        }
    try:
        fd, name = tempfile.mkstemp(prefix="genome-seq-", dir=str(dirpath))
    except OSError as e:
        return {"status": "BLOCKED", "reason": f"mkstemp {dirpath}: {e}", "evidence_tier": "STATIC"}
    path = Path(name)
    try:
        payload = b"\x5a" * (1 << 20)
        wrote = 0
        t0 = time.perf_counter()
        with os.fdopen(fd, "wb") as f:
            while wrote < nbytes:
                n = min(len(payload), nbytes - wrote)
                f.write(payload[:n])
                wrote += n
            f.flush()
            os.fsync(f.fileno())
        write_dt = time.perf_counter() - t0
        read = _sequential_read(path, nbytes=wrote)
        return {
            "dir": str(dirpath),
            "bytes": wrote,
            "write_s": round(write_dt, 4),
            "write_gb_s": round(wrote / write_dt / 1e9, 3) if write_dt > 0 else None,
            "read": read,
            "evidence_tier": "HARDWARE_MEASURED",
            "note": "tempfile created under a writable mount and unlinked; not a roof",
        }
    except OSError as e:
        return {"status": "BLOCKED", "reason": f"write probe {dirpath}: {e}", "evidence_tier": "STATIC"}
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _random_read(path: Path, n: int = RANDOM_READS, block: int = RANDOM_BLOCK) -> dict[str, Any]:
    """Scattered 4KiB reads. Metadata/random behaviour, not a disk roof."""
    try:
        size = path.stat().st_size
    except OSError as e:
        return {"status": "BLOCKED", "reason": f"stat {path}: {e}", "evidence_tier": "STATIC"}
    if size < block:
        return {"status": "BLOCKED", "reason": f"{path} smaller than {block}", "evidence_tier": "STATIC"}
    offsets = [random.randrange(0, size - block) for _ in range(n)]
    got = 0
    t0 = time.perf_counter()
    try:
        with open(path, "rb") as f:
            for off in offsets:
                f.seek(off)
                chunk = f.read(block)
                got += len(chunk)
    except OSError as e:
        return {"status": "BLOCKED", "reason": f"pread {path}: {e}", "evidence_tier": "STATIC"}
    dt = time.perf_counter() - t0
    return {
        "path": str(path),
        "n": n,
        "block": block,
        "bytes": got,
        "s": round(dt, 4),
        "iops": round(n / dt, 1) if dt > 0 else None,
        "evidence_tier": "HARDWARE_MEASURED",
        "note": "scattered 4KiB reads of an existing file; not a random-IOPS roof",
    }


def _metadata_sample(path: Path, n: int = 32) -> dict[str, Any]:
    try:
        names = list(path.iterdir())[:n]
    except OSError as e:
        return {"status": "BLOCKED", "reason": f"listdir {path}: {e}", "evidence_tier": "STATIC"}
    t0 = time.perf_counter()
    for child in names:
        try:
            child.stat()
        except OSError:
            pass
    dt = time.perf_counter() - t0
    t1 = time.perf_counter()
    try:
        list(path.iterdir())
    except OSError:
        pass
    listdir_dt = time.perf_counter() - t1
    return {
        "path": str(path),
        "n_stat": len(names),
        "stat_s": round(dt, 4),
        "listdir_s": round(listdir_dt, 4),
        "evidence_tier": "HARDWARE_MEASURED",
        "note": "metadata sample of the directory itself, not a recursive walk",
    }


def _corpdrive_read_target() -> Path | None:
    """Existing, complete files only. Never anything under partial/ (live downloads)."""
    protected = CORPDRIVE / "legal-scans-2026-08-23.tar.zst"
    if protected.is_file():
        return protected
    specimens = CORPDRIVE / "hawking-modellake" / "specimens"
    if specimens.is_dir():
        try:
            for spec in specimens.iterdir():
                tok = spec / "tokenizer.json"
                if tok.is_file() and tok.stat().st_size >= 1 << 20:
                    return tok
        except OSError:
            pass
    roadmap = CORPDRIVE / "H-ROADMAP.md"
    if roadmap.is_file():
        return roadmap
    return None


def probe_storage() -> dict[str, Any]:
    """Live mounts, including /Volumes/corpdrive. READ-ONLY on corpdrive.

    Sequential write probes run only under /tmp. Corpdrive is sampled by
    reading a bounded prefix of an existing file that is not in partial/.
    """
    mounts = _parse_mounts()
    wanted = {"/", "/tmp", "/System/Volumes/Data", "/Volumes/corpdrive"}
    by_mount = {m["mount"]: m for m in mounts}
    # Always try corpdrive even if `mount` parsing missed it.
    extra = []
    for p in (Path("/"), Path("/tmp"), Path("/System/Volumes/Data"), CORPDRIVE):
        if p.exists() and str(p) not in by_mount:
            extra.append({
                "device": None,
                "mount": str(p),
                "fstype": None,
                "options": [],
                "read_only": not os.access(p, os.W_OK),
                "evidence_tier": "STATIC",
            })
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in mounts + extra:
        mp = row["mount"]
        if mp in seen:
            continue
        # Skip ephemeral/system noise; keep root, data, /tmp, /Volumes/*, user caches.
        keep = mp in wanted or mp.startswith("/Volumes/") or mp == "/tmp"
        if not keep:
            continue
        seen.add(mp)
        path = Path(mp)
        vs = _statvfs(path) if path.exists() else {
            "status": "BLOCKED",
            "reason": f"{mp} is not present",
            "evidence_tier": "STATIC",
        }
        rec: dict[str, Any] = {
            **row,
            "exists": path.exists(),
            "capacity": vs,
        }
        if not path.exists():
            rec["sequential"] = {
                "status": "BLOCKED",
                "reason": f"{mp} is not mounted",
                "evidence_tier": "STATIC",
            }
            records.append(rec)
            continue
        if mp == "/Volumes/corpdrive" or _no_write(path):
            target = _corpdrive_read_target() if mp == "/Volumes/corpdrive" else None
            if target is not None:
                rec["sequential"] = _sequential_read(target)
                rec["random"] = _random_read(target)
            else:
                rec["sequential"] = {
                    "status": "BLOCKED",
                    "reason": "no existing complete file to sample without writing",
                    "evidence_tier": "STATIC",
                }
            rec["write_probe"] = {
                "status": "BLOCKED",
                "reason": "READ-ONLY on this mount; live downloads / protected volume",
                "evidence_tier": "STATIC",
            }
            rec["metadata"] = _metadata_sample(path)
        elif row.get("read_only") or mp == "/":
            rec["sequential"] = {
                "status": "BLOCKED",
                "reason": f"{mp} is read-only; no write probe, no disposable file",
                "evidence_tier": "STATIC",
            }
            rec["metadata"] = _metadata_sample(path)
        else:
            # Writable: /tmp lives on the Data volume on this host.
            probe_dir = Path("/tmp") if mp in {"/tmp", "/System/Volumes/Data"} else path
            if _no_write(probe_dir) or str(probe_dir).startswith("/Volumes/"):
                rec["sequential"] = {
                    "status": "BLOCKED",
                    "reason": "refusing write probe outside /tmp",
                    "evidence_tier": "STATIC",
                }
            else:
                rec["sequential"] = _sequential_write_read(probe_dir)
            rec["metadata"] = _metadata_sample(path)
        records.append(rec)

    corp = next((r for r in records if r["mount"] == "/Volumes/corpdrive"), None)
    return {
        "kind": "STORAGE",
        "name": "storage",
        "present": any(r.get("exists") for r in records),
        "maturity": "MEASURED" if records else "ABSENT",
        "evidence_tier": "HARDWARE_MEASURED",
        "mounts": records,
        "corpdrive": {
            "mounted": bool(corp and corp.get("exists")),
            "probe": "read-only sequential + statvfs + metadata",
            "wrote": False,
        },
        "laws": [
            {
                "law_id": "STORAGE-CHANGES-SCIENCE-ECONOMICS",
                "statement": (
                    "ModelLake and Odyssey already showed that the storage tier "
                    "(HDD lake vs SSD stage vs UMA resident) changes what science "
                    "is affordable. The genome therefore carries mounts, capacity, "
                    "observed sequential rate, metadata behaviour and cache "
                    "characteristics, not just SoC identity."
                ),
                "evidence_tier": "STATIC",
                "scope": "ARCHITECTURE",
            }
        ],
    }


def _ane_ioreg_present() -> dict[str, Any]:
    out = _run(["ioreg", "-r", "-d", "1", "-c", "H11ANEIn", "-w", "0"], timeout=8.0)
    if out and "H11ANEIn" in out:
        n = len(re.findall(r"H11ANEIn", out))
        return {
            "present": True,
            "ioreg_class": "H11ANEIn",
            "ioreg_mentions": n,
            "evidence_tier": "HARDWARE_MEASURED",
            "note": "presence only; ioreg is not a performance number",
        }
    return {
        "present": False,
        "evidence_tier": "HARDWARE_MEASURED",
        "reason": "no H11ANEIn node from ioreg",
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_ane_lab() -> tuple[dict[str, Any] | None, str | None]:
    """Cite the live MLComputePlan lab receipt rather than inventing ANE fields."""
    for p in ANE_LAB_CANDIDATES:
        d = _load_json(p)
        if d:
            return d, str(p)
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "show", ANE_GIT_FALLBACK],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout), f"git:{ANE_GIT_FALLBACK}"
    except Exception:
        pass
    return None, None


def probe_ane() -> dict[str, Any]:
    """ANE domain. Presence from ioreg; MLComputePlan fields from the lab receipt.

    Does not invent TOPS, joules, or a Flash-residency claim. The lab receipt
    records that the observed preferred device for the add fixture was CPU,
    even though NEURAL_ENGINE is in the supported set.
    """
    presence = _ane_ioreg_present()
    lab, lab_path = load_ane_lab()
    profile = None
    predict = None
    placement = None
    if lab:
        profile = lab.get("device_profile") if "device_profile" in lab else lab
        predict = lab.get("predict")
        placement = lab.get("placement")
    devices = []
    if isinstance(profile, dict):
        devices = list(profile.get("supported_compute_devices") or profile.get("compute_devices") or [])
    rec: dict[str, Any] = {
        "kind": "ANE",
        "name": "ane_0",
        "present": bool(presence.get("present")),
        "maturity": "PROFILED" if (presence.get("present") and profile) else (
            "PRESENT" if presence.get("present") else "ABSENT"
        ),
        "evidence_tier": "HARDWARE_MEASURED",
        "ioreg": presence,
        "lab_receipt": lab_path,
        "neural_engine_present": (
            (profile or {}).get("neural_engine_present")
            if isinstance(profile, dict)
            else presence.get("present")
        ),
        "supported_compute_devices": devices,
        "mlcomputeplan": (profile or {}).get("mlcomputeplan") if isinstance(profile, dict) else None,
        "placement": placement,
        "predict": None,
        "claim_boundary": (
            "Public MLComputePlan operation support and preferred placement, plus "
            "ioreg presence. No TOPS, no energy, no Flash residency claim. "
            "Requested compute units are not placement."
        ),
    }
    if isinstance(predict, dict) and predict.get("status") == "MEASURED":
        rec["predict"] = {
            "status": predict.get("status"),
            "warm_predict_ns_mean": predict.get("warm_predict_ns_mean"),
            "warm_predict_ns_min": predict.get("warm_predict_ns_min"),
            "warm_predict_ns_max": predict.get("warm_predict_ns_max"),
            "repeats": predict.get("repeats"),
            "requested_compute_units": predict.get("requested_compute_units"),
            "evidence_tier": "HARDWARE_MEASURED",
            "provenance": lab_path,
            "note": "cited from the lab receipt, not re-measured in this process",
        }
    if lab is None:
        rec["lab_receipt_status"] = "BLOCKED"
        rec["lab_receipt_reason"] = (
            "HCLI_FORBIDDEN_FRUIT_LAB.json and APPLE_ANE_DEVICE_PROFILE.json "
            "were not readable from this worktree"
        )
    return rec


def probe_network() -> dict[str, Any]:
    """Interface identity only. WAN throughput is BLOCKED while lake fills run."""
    names = (_run(["ifconfig", "-l"], timeout=5.0) or "").split()
    return {
        "kind": "NETWORK",
        "name": "network",
        "present": bool(names),
        "maturity": "PRESENT" if names else "ABSENT",
        "evidence_tier": "STATIC",
        "interfaces": names,
        "wan_throughput": {
            "status": "BLOCKED",
            "reason": (
                "live hf download workers write to /Volumes/corpdrive; a WAN "
                "sample would be contended and could disturb them"
            ),
            "evidence_tier": "STATIC",
        },
    }


def discover_identity() -> dict[str, Any]:
    """Static identity. No bandwidth, no joules, no tps."""
    mem = _sysctl("hw.memsize")
    gpu = _gpu_cores()
    return {
        "soc": _sysctl("machdep.cpu.brand_string"),
        "arch": platform.machine(),
        "cpu_cores": _int_or_none(_sysctl("hw.ncpu")),
        "perf_cores": _int_or_none(_sysctl("hw.perflevel0.physicalcpu")),
        "efficiency_cores": _int_or_none(_sysctl("hw.perflevel1.physicalcpu")),
        "gpu_cores": gpu,
        "memory_bytes": int(mem) if mem else None,
        "os": f"{platform.system()} {platform.release()}",
        "os_product": _sysctl("kern.osproductversion"),
        "discovery_class": "STATIC",
        "evidence_tier": "STATIC",
        "discovery_means": ("sysctl", "ioreg AGXAccelerator", "platform"),
    }


def _domain_cpu(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "CPU",
        "name": "cpu_0",
        "present": identity.get("cpu_cores") not in (None, 0),
        "maturity": "MEASURED" if identity.get("cpu_cores") else "ABSENT",
        "evidence_tier": "HARDWARE_MEASURED",
        "cores": identity.get("cpu_cores"),
        "perf_cores": identity.get("perf_cores"),
        "efficiency_cores": identity.get("efficiency_cores"),
        "arch": identity.get("arch"),
        "soc": identity.get("soc"),
    }


def _domain_gpu(identity: Mapping[str, Any], bandwidth: Mapping[str, Any]) -> dict[str, Any]:
    cores = identity.get("gpu_cores")
    present = isinstance(cores, int) and cores > 0
    bw_ok = isinstance(bandwidth, dict) and bandwidth.get("evidence_tier") == "HARDWARE_MEASURED" \
        and bandwidth.get("status") not in {"ABSENT", "UNRELIABLE", "BLOCKED"}
    return {
        "kind": "GPU",
        "name": "gpu_uma_0",
        "present": present,
        "maturity": "MEASURED" if bw_ok else ("PRESENT" if present else "ABSENT"),
        "evidence_tier": "HARDWARE_MEASURED" if present else "STATIC",
        "gpu_cores": cores,
        "measured_bandwidth": bandwidth,
        "note": "gpu_uma_0 is the host GPU domain name fusion_bridge already uses",
    }


def _domain_uma(identity: Mapping[str, Any]) -> dict[str, Any]:
    soc = str(identity.get("soc") or "")
    arch = str(identity.get("arch") or "")
    apple = arch in {"arm64", "aarch64"} and soc.startswith("Apple")
    mem = identity.get("memory_bytes")
    return {
        "kind": "UMA",
        "name": "uma_0",
        "present": bool(apple and mem),
        "maturity": "PRESENT" if (apple and mem) else "ABSENT",
        "evidence_tier": "STATIC",
        "capacity_bytes": mem,
        "capacity_evidence_tier": "HARDWARE_MEASURED" if mem else "STATIC",
        "internal_coherency": "HARDWARE_UMA" if apple else "NONE",
        "laws": [
            {
                "law_id": "UMA-COPY-ELISION",
                "statement": (
                    "On this unified-memory SoC, CUDA-era host-to-device and "
                    "device-to-host copies are structurally avoidable. There is "
                    "no separate device memory."
                ),
                "evidence_tier": "STATIC",
                "scope": "MACHINE_LOCAL",
                "note": (
                    "topology fact, not a speedup number; it does not transfer "
                    "to a discrete-GPU machine"
                ),
            }
        ],
    }


def _domain_fpga_declared() -> dict[str, Any]:
    return {
        "kind": "FPGA",
        "name": "fpga_hbm_0",
        "present": False,
        "maturity": "DECLARED",
        "evidence_tier": "STATIC",
        "physical": False,
        "product": None,
        "reason": (
            "no FPGA is attached to this host; fpga_hbm_0 is the same declared "
            "domain name fusion_bridge already uses. A future board fills this "
            "record; it does not require a schema change."
        ),
        "performance": "UNKNOWN",
        "wake_condition": "U50_PRESENT",
        "note": "COST_MODEL interconnect knobs live in fusion_bridge; nothing here is a measurement",
    }


def _domain_external_declared() -> dict[str, Any]:
    return {
        "kind": "EXTERNAL_ACCELERATOR",
        "name": "nvidia_dgx_0",
        "present": False,
        "maturity": "DECLARED",
        "evidence_tier": "STATIC",
        "physical": False,
        "product_family": "NVIDIA_DGX",
        "reason": (
            "no DGX, no discrete NVIDIA GPU, no eGPU on this host. The domain "
            "is declared so a future DGX slots in without a schema change."
        ),
        "performance": "UNKNOWN",
        "wake_condition": "DGX_PRESENT",
        "note": "Anything about DGX/FPGA/eGPU on this machine is a model, never a measurement",
    }


def _domain_u50dd_declared() -> dict[str, Any]:
    """Named U50DD slot. Brochure numbers live in hwir, not here.

    Flipping present=True without a board census still leaves every
    performance axis UNKNOWN. Presence is not bandwidth.
    """
    return {
        "kind": "FPGA",
        "name": "u50dd_0",
        "present": False,
        "maturity": "DECLARED",
        "evidence_tier": "STATIC",
        "physical": False,
        "product": "Alveo U50DD",
        "expected_sku": "A-U50DD-P00G-ES3-G",
        "wake_condition": "U50_PRESENT",
        "performance": "UNKNOWN",
        "brochure_lives_in": "tools.future.hwir.u50_family_profile",
        "reason": (
            "no U50/U50DD is attached to this host. Expected SKU is a planning "
            "name, not a local census. Vendor-literature LUT/DSP/HBM figures "
            "live in hwir.u50_family_profile('u50dd') and are STATIC, not "
            "HARDWARE_MEASURED."
        ),
        "note": "Anything about U50DD on this machine is a model, never a measurement",
    }


def _domain_egpu_declared() -> dict[str, Any]:
    return {
        "kind": "EXTERNAL_ACCELERATOR",
        "name": "egpu_0",
        "present": False,
        "maturity": "DECLARED",
        "evidence_tier": "STATIC",
        "physical": False,
        "product_family": "EGPU",
        "wake_condition": "EGPU_PRESENT",
        "performance": "UNKNOWN",
        "reason": (
            "no eGPU enclosure is attached to this host. The Apple SoC GPU "
            "is not an eGPU."
        ),
        "note": "Anything about an eGPU on this machine is a model, never a measurement",
    }


def axis_record(
    axis: str,
    *,
    status: str,
    evidence_tier: str,
    **facts: Any,
) -> dict[str, Any]:
    """One per-axis row. Tiers are never merged; status is not a tier."""
    if axis not in AXES:
        raise ValueError(f"axis {axis!r} is not one of {AXES}")
    if status not in AXIS_STATUSES:
        raise ValueError(f"status {status!r} is not one of {AXIS_STATUSES}")
    if evidence_tier not in EVIDENCE_TIERS:
        raise ValueError(f"evidence_tier {evidence_tier!r} is not one of {EVIDENCE_TIERS}")
    rec = {
        "axis": axis,
        "status": status,
        "evidence_tier": evidence_tier,
        **facts,
    }
    # Presence of a device is not a performance number. Refuse a caller that
    # tries to stamp HARDWARE_MEASURED onto an absent device's bandwidth.
    if rec.get("device_present") is False and evidence_tier == "HARDWARE_MEASURED" and axis not in {"presence"}:
        raise ValueError(
            f"refusing HARDWARE_MEASURED on axis {axis!r} of an absent device; "
            "that would fabricate a measurement"
        )
    return rec


def _probe_status(probe: Any) -> tuple[str, str, str | None]:
    """Map a genome probe dict onto (status, evidence_tier, reason)."""
    if not isinstance(probe, dict):
        return "UNKNOWN", "STATIC", "no probe record"
    tier = probe.get("evidence_tier") if probe.get("evidence_tier") in EVIDENCE_TIERS else "STATIC"
    st = probe.get("status")
    reason = probe.get("reason")
    if st in {"ABSENT", "BLOCKED"}:
        return str(st), tier, reason
    if st == "UNRELIABLE":
        return "UNRELIABLE", tier, reason
    if tier == "HARDWARE_MEASURED" and st not in {"ABSENT", "BLOCKED", "UNRELIABLE"}:
        return "MEASURED", "HARDWARE_MEASURED", reason
    if tier in {"COST_MODEL", "CYCLE_APPROX", "FUNCTIONAL_SIM"}:
        return "MODELLED", tier, reason
    if st == "UNKNOWN" or probe.get("performance") == "UNKNOWN":
        return "UNKNOWN", tier, reason
    return "STATIC_IDENTITY", tier, reason


def _absent_axes(domain: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every performance axis of an absent device is UNKNOWN/ABSENT, never measured."""
    name = domain.get("name")
    kind = domain.get("kind")
    wake = domain.get("wake_condition")
    reason = domain.get("reason") or f"{name} is not attached"
    rows = [
        axis_record(
            "presence",
            status="ABSENT",
            evidence_tier="STATIC",
            device=name,
            kind=kind,
            device_present=False,
            wake_condition=wake,
            reason=reason,
        )
    ]
    for axis in AXES:
        if axis == "presence":
            continue
        rows.append(
            axis_record(
                axis,
                status="ABSENT" if axis != "identity" else "UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=False,
                wake_condition=wake,
                reason=(
                    f"{axis} cannot be measured until {wake or 'the device'} "
                    f"arrives; brochure figures are not this axis"
                ),
            )
        )
    return rows


def axes_for_domain(
    domain: Mapping[str, Any],
    *,
    genome: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Per-axis measured / modelled / unknown inventory for one domain.

    Called by the Hardware Doctor. A domain with present=True and no
    bandwidth probe still reports bandwidth UNKNOWN -- presence is not a
    rate. An absent domain never emits HARDWARE_MEASURED on a performance
    axis.
    """
    if not isinstance(domain, dict):
        raise TypeError("domain must be a dict")
    name = domain.get("name")
    kind = domain.get("kind")
    present = bool(domain.get("present"))
    genome = genome or {}
    thermal = genome.get("thermal_envelope") if isinstance(genome.get("thermal_envelope"), dict) else {}
    sustained = genome.get("sustained_behaviour") if isinstance(genome.get("sustained_behaviour"), dict) else {}

    if not present:
        return _absent_axes(domain)

    rows: list[dict[str, Any]] = [
        axis_record(
            "presence",
            status="MEASURED",
            evidence_tier=domain.get("evidence_tier")
            if domain.get("evidence_tier") in EVIDENCE_TIERS
            else "HARDWARE_MEASURED",
            device=name,
            kind=kind,
            device_present=True,
        )
    ]

    def _thermal() -> dict[str, Any]:
        if thermal.get("status") in {"ABSENT", "BLOCKED"}:
            return axis_record(
                "thermal",
                status=str(thermal.get("status")),
                evidence_tier=thermal.get("evidence_tier")
                if thermal.get("evidence_tier") in EVIDENCE_TIERS
                else "STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason=thermal.get("reason"),
            )
        if sustained.get("status") in {"ABSENT", "BLOCKED"}:
            return axis_record(
                "thermal",
                status=str(sustained.get("status")),
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason=sustained.get("reason"),
            )
        return axis_record(
            "thermal",
            status="UNKNOWN",
            evidence_tier="STATIC",
            device=name,
            kind=kind,
            device_present=True,
            reason="no sustained thermal campaign has been run",
        )

    def _energy() -> dict[str, Any]:
        return axis_record(
            "energy",
            status="UNKNOWN",
            evidence_tier="STATIC",
            device=name,
            kind=kind,
            device_present=True,
            reason="no joule meter / powermetrics campaign is cited on this domain",
        )

    def _backend() -> dict[str, Any]:
        return axis_record(
            "backend",
            status="MEASURED" if domain.get("maturity") in {"MEASURED", "PROFILED", "QUALIFIED", "PRESENT"} else "UNKNOWN",
            evidence_tier=domain.get("evidence_tier")
            if domain.get("evidence_tier") in EVIDENCE_TIERS
            else "STATIC",
            device=name,
            kind=kind,
            device_present=True,
            maturity=domain.get("maturity"),
        )

    if kind == "CPU":
        rows.append(
            axis_record(
                "identity",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                cores=domain.get("cores"),
                arch=domain.get("arch"),
                soc=domain.get("soc"),
            )
        )
        rows.append(
            axis_record(
                "capacity",
                status="MEASURED",
                evidence_tier="HARDWARE_MEASURED",
                device=name,
                kind=kind,
                device_present=True,
                cores=domain.get("cores"),
                perf_cores=domain.get("perf_cores"),
                efficiency_cores=domain.get("efficiency_cores"),
            )
        )
        rows.append(
            axis_record(
                "bandwidth",
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason="CPU STREAM/triad has not been run; the GPU triad is not a CPU roof",
            )
        )
        rows.append(
            axis_record(
                "latency",
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason="no CPU cache/memory latency campaign is cited",
            )
        )
        rows.append(_energy())
        rows.append(_thermal())
        rows.append(
            axis_record(
                "placement",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                note="host CPU is always a legal placement; not a preference measurement",
            )
        )
        rows.append(
            axis_record(
                "transport",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                note="on-package; no host-to-device copy on this SoC",
            )
        )
        rows.append(_backend())
        return rows

    if kind == "GPU":
        rows.append(
            axis_record(
                "identity",
                status="MEASURED",
                evidence_tier="HARDWARE_MEASURED",
                device=name,
                kind=kind,
                device_present=True,
                gpu_cores=domain.get("gpu_cores"),
            )
        )
        rows.append(
            axis_record(
                "capacity",
                status="MEASURED",
                evidence_tier="HARDWARE_MEASURED",
                device=name,
                kind=kind,
                device_present=True,
                gpu_cores=domain.get("gpu_cores"),
                note="core count is identity; it is not a FLOP roof",
            )
        )
        bw = domain.get("measured_bandwidth") if isinstance(domain.get("measured_bandwidth"), dict) else (
            genome.get("measured_bandwidth") if isinstance(genome.get("measured_bandwidth"), dict) else {}
        )
        st, tier, reason = _probe_status(bw)
        rec = axis_record(
            "bandwidth",
            status=st if st != "STATIC_IDENTITY" else "UNKNOWN",
            evidence_tier=tier,
            device=name,
            kind=kind,
            device_present=True,
            reason=reason or bw.get("note"),
            reliable=bw.get("reliable"),
            pattern=bw.get("pattern"),
            note="one triad pattern; not the SoC roof",
        )
        if st == "MEASURED" and "median_gb_s" in bw:
            rec["median_gb_s"] = bw.get("median_gb_s")
            rec["iqr_spread_pct"] = bw.get("iqr_spread_pct")
        rows.append(rec)
        rows.append(
            axis_record(
                "latency",
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason="no dispatch/small-message latency campaign is cited on this domain",
            )
        )
        rows.append(_energy())
        rows.append(_thermal())
        rows.append(
            axis_record(
                "placement",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                note="Metal GPU is the default decode placement on this host",
            )
        )
        rows.append(
            axis_record(
                "transport",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                note="UMA: CUDA-era HtoD/DtoH copies are structurally avoidable (law, not a speedup number)",
            )
        )
        rows.append(_backend())
        return rows

    if kind == "UMA":
        cap_tier = domain.get("capacity_evidence_tier") if domain.get("capacity_evidence_tier") in EVIDENCE_TIERS else "HARDWARE_MEASURED"
        rows.append(
            axis_record(
                "identity",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                internal_coherency=domain.get("internal_coherency"),
            )
        )
        rows.append(
            axis_record(
                "capacity",
                status="MEASURED" if domain.get("capacity_bytes") else "UNKNOWN",
                evidence_tier=cap_tier if domain.get("capacity_bytes") else "STATIC",
                device=name,
                kind=kind,
                device_present=True,
                capacity_bytes=domain.get("capacity_bytes"),
            )
        )
        bw = genome.get("measured_bandwidth") if isinstance(genome.get("measured_bandwidth"), dict) else {}
        st, tier, reason = _probe_status(bw)
        rec = axis_record(
            "bandwidth",
            status=st if st != "STATIC_IDENTITY" else "UNKNOWN",
            evidence_tier=tier,
            device=name,
            kind=kind,
            device_present=True,
            reason=reason or "UMA is the physical medium of the GPU triad; not an independent roof",
            note="same triad as gpu_uma_0; citing it here does not invent a second measurement",
        )
        if st == "MEASURED" and "median_gb_s" in bw:
            rec["median_gb_s"] = bw.get("median_gb_s")
        rows.append(rec)
        rows.append(
            axis_record(
                "latency",
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason="no pointer-chase / cache-line campaign is cited",
            )
        )
        rows.append(_energy())
        rows.append(_thermal())
        rows.append(
            axis_record(
                "placement",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                note="resident bodies that fit capacity_bytes are UMA-legal",
            )
        )
        rows.append(
            axis_record(
                "transport",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                law_id="UMA-COPY-ELISION",
                note="topology fact, not a speedup number; does not transfer to a discrete GPU",
            )
        )
        rows.append(_backend())
        return rows

    if kind == "ANE":
        ioreg = domain.get("ioreg") if isinstance(domain.get("ioreg"), dict) else {}
        rows.append(
            axis_record(
                "identity",
                status="MEASURED" if ioreg.get("present") else "UNKNOWN",
                evidence_tier=ioreg.get("evidence_tier")
                if ioreg.get("evidence_tier") in EVIDENCE_TIERS
                else "HARDWARE_MEASURED",
                device=name,
                kind=kind,
                device_present=True,
                ioreg_class=ioreg.get("ioreg_class"),
                note="presence only; ioreg is not TOPS",
            )
        )
        rows.append(
            axis_record(
                "capacity",
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason="no TOPS, no SRAM census; refusing to invent one",
            )
        )
        rows.append(
            axis_record(
                "bandwidth",
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason="ANE bandwidth is not a public ioreg field and is not guessed",
            )
        )
        rows.append(
            axis_record(
                "latency",
                status="MEASURED" if isinstance(domain.get("predict"), dict) and domain["predict"].get("status") == "MEASURED" else "UNKNOWN",
                evidence_tier="HARDWARE_MEASURED"
                if isinstance(domain.get("predict"), dict) and domain["predict"].get("status") == "MEASURED"
                else "STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason=None
                if isinstance(domain.get("predict"), dict) and domain["predict"].get("status") == "MEASURED"
                else "no predict receipt, or predict was not MEASURED",
                provenance=(domain.get("predict") or {}).get("provenance")
                if isinstance(domain.get("predict"), dict)
                else None,
                note="cited from the lab receipt when present; add-fixture only, not Flash/Qwen",
            )
        )
        rows.append(_energy())
        rows.append(_thermal())
        placement = domain.get("placement")
        supported = domain.get("supported_compute_devices") or []
        rows.append(
            axis_record(
                "placement",
                status="MEASURED" if placement or supported else "UNKNOWN",
                evidence_tier="HARDWARE_MEASURED" if placement or supported else "STATIC",
                device=name,
                kind=kind,
                device_present=True,
                supported_compute_devices=list(supported),
                placement=placement,
                claim_boundary=domain.get("claim_boundary"),
                note="requested compute units are not placement; Flash/Qwen residency is not claimed",
            )
        )
        rows.append(
            axis_record(
                "transport",
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason="ANE host-to-engine transport is not measured here",
            )
        )
        rows.append(_backend())
        return rows

    if kind == "STORAGE":
        mounts = list(domain.get("mounts") or [])
        rows.append(
            axis_record(
                "identity",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                n_mounts=len(mounts),
                mounts=[m.get("mount") for m in mounts],
            )
        )
        cap_measured = any(
            isinstance(m.get("capacity"), dict)
            and m["capacity"].get("evidence_tier") == "HARDWARE_MEASURED"
            for m in mounts
        )
        rows.append(
            axis_record(
                "capacity",
                status="MEASURED" if cap_measured else "UNKNOWN",
                evidence_tier="HARDWARE_MEASURED" if cap_measured else "STATIC",
                device=name,
                kind=kind,
                device_present=True,
                note="statvfs capacity; free space is a live reading, not identity",
            )
        )
        seq_rows = []
        for m in mounts:
            seq = m.get("sequential") if isinstance(m.get("sequential"), dict) else {}
            st, tier, reason = _probe_status(seq)
            seq_rows.append({
                "mount": m.get("mount"),
                "status": st,
                "evidence_tier": tier,
                "reason": reason or seq.get("note"),
                "write_probe": (m.get("write_probe") or {}).get("status")
                if isinstance(m.get("write_probe"), dict)
                else None,
            })
        any_measured = any(s["status"] == "MEASURED" for s in seq_rows)
        any_blocked = any(s["status"] in {"BLOCKED", "ABSENT"} for s in seq_rows)
        bw_status = "MEASURED" if any_measured else ("BLOCKED" if any_blocked else "UNKNOWN")
        rows.append(
            axis_record(
                "bandwidth",
                status=bw_status,
                evidence_tier="HARDWARE_MEASURED" if any_measured else "STATIC",
                device=name,
                kind=kind,
                device_present=True,
                samples=seq_rows,
                note="bounded sequential sample of an existing file; not a disk roof. corpdrive write is BLOCKED",
            )
        )
        rnd_measured = any(
            isinstance(m.get("random"), dict) and m["random"].get("evidence_tier") == "HARDWARE_MEASURED"
            for m in mounts
        )
        rows.append(
            axis_record(
                "latency",
                status="MEASURED" if rnd_measured else "UNKNOWN",
                evidence_tier="HARDWARE_MEASURED" if rnd_measured else "STATIC",
                device=name,
                kind=kind,
                device_present=True,
                note="scattered 4KiB reads / metadata sample when present; not a random-IOPS roof",
            )
        )
        rows.append(_energy())
        rows.append(
            axis_record(
                "thermal",
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason="storage thermal is not in this genome",
            )
        )
        rows.append(
            axis_record(
                "placement",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                note="lake vs SSD vs UMA changes science economics (STORAGE-CHANGES-SCIENCE-ECONOMICS)",
            )
        )
        rows.append(
            axis_record(
                "transport",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                note="mount table is identity of what is attached, not a WAN rate",
            )
        )
        rows.append(_backend())
        return rows

    if kind == "NETWORK":
        wan = domain.get("wan_throughput") if isinstance(domain.get("wan_throughput"), dict) else {}
        rows.append(
            axis_record(
                "identity",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                interfaces=domain.get("interfaces"),
            )
        )
        rows.append(
            axis_record(
                "capacity",
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason="link speed is not scraped from ifconfig here",
            )
        )
        st, tier, reason = _probe_status(wan) if wan else ("BLOCKED", "STATIC", "no wan_throughput record")
        rows.append(
            axis_record(
                "bandwidth",
                status=st if st != "STATIC_IDENTITY" else "BLOCKED",
                evidence_tier=tier,
                device=name,
                kind=kind,
                device_present=True,
                reason=reason or wan.get("reason"),
            )
        )
        rows.append(
            axis_record(
                "latency",
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason="no RTT campaign; WAN probe is blocked while lake fills run",
            )
        )
        rows.append(_energy())
        rows.append(
            axis_record(
                "thermal",
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason="not applicable as a host thermal axis",
            )
        )
        rows.append(
            axis_record(
                "placement",
                status="STATIC_IDENTITY",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
            )
        )
        rows.append(
            axis_record(
                "transport",
                status="BLOCKED" if wan else "UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason=(wan or {}).get("reason"),
            )
        )
        rows.append(_backend())
        return rows

    # Unknown kind, present: identity only, everything else UNKNOWN.
    rows.append(
        axis_record(
            "identity",
            status="STATIC_IDENTITY",
            evidence_tier=domain.get("evidence_tier")
            if domain.get("evidence_tier") in EVIDENCE_TIERS
            else "STATIC",
            device=name,
            kind=kind,
            device_present=True,
        )
    )
    for axis in AXES:
        if axis in {"presence", "identity"}:
            continue
        rows.append(
            axis_record(
                axis,
                status="UNKNOWN",
                evidence_tier="STATIC",
                device=name,
                kind=kind,
                device_present=True,
                reason=f"kind {kind!r} has no specialised axis filler",
            )
        )
    return rows


def devices_exist(genome: Mapping[str, Any]) -> dict[str, Any]:
    """What the genome says is attached. Inventory, not a performance number."""
    present: list[dict[str, Any]] = []
    absent: list[dict[str, Any]] = []
    for name, d in (genome.get("domains") or {}).items():
        if not isinstance(d, dict):
            continue
        row = {
            "name": name,
            "kind": d.get("kind"),
            "present": bool(d.get("present")),
            "maturity": d.get("maturity"),
            "evidence_tier": d.get("evidence_tier"),
            "wake_condition": d.get("wake_condition"),
            "physical": d.get("physical"),
        }
        (present if row["present"] else absent).append(row)
    return {
        "soc": genome.get("soc"),
        "arch": genome.get("arch"),
        "present": present,
        "absent": absent,
        "n_present": len(present),
        "n_absent": len(absent),
        "evidence_tier": "STATIC",
        "note": (
            "present/absent is genome inventory. A declared slot with "
            "present=False is not a measurement of a future board."
        ),
    }


def declare_domain(
    genome: Mapping[str, Any],
    *,
    kind: str,
    name: str,
    present: bool = False,
    maturity: str = "DECLARED",
    evidence_tier: str = "STATIC",
    **facts: Any,
) -> dict[str, Any]:
    """Slot a domain into an existing genome without a schema change.

    FPGA, DGX, eGPU, U50, a future M-series GPU — all go through this. The
    schema string is unchanged; the domains dict is open-keyed.
    """
    if evidence_tier not in EVIDENCE_TIERS:
        raise ValueError(f"evidence_tier {evidence_tier!r} is not one of {EVIDENCE_TIERS}")
    if maturity not in MATURITY:
        raise ValueError(f"maturity {maturity!r} is not one of {MATURITY}")
    out = copy.deepcopy(dict(genome))
    domains = dict(out.get("domains") or {})
    rec = {
        "kind": kind,
        "name": name,
        "present": present,
        "maturity": maturity,
        "evidence_tier": evidence_tier,
        **facts,
    }
    domains[name] = rec
    out["domains"] = domains
    out["genome_digest"] = genome_digest(out)
    return out


def genome_digest(genome: Mapping[str, Any]) -> str:
    """Identity digest. Rates, free space and timestamps are excluded so a
    contended sample cannot look like a different machine."""
    identity = {k: genome.get(k) for k in DIGEST_IDENTITY_KEYS}
    domain_id = []
    for name in sorted(genome.get("domains") or {}):
        d = genome["domains"][name]
        if not isinstance(d, dict):
            continue
        entry: dict[str, Any] = {
            "name": name,
            "kind": d.get("kind"),
            "present": d.get("present"),
            "maturity": d.get("maturity"),
        }
        if d.get("kind") == "STORAGE":
            mounts = []
            for m in d.get("mounts") or []:
                cap = m.get("capacity") if isinstance(m.get("capacity"), dict) else {}
                mounts.append({
                    "mount": m.get("mount"),
                    "device": m.get("device"),
                    "fstype": m.get("fstype"),
                    "capacity_bytes": cap.get("capacity_bytes"),
                })
            entry["mounts"] = mounts
        if d.get("kind") == "UMA":
            entry["capacity_bytes"] = d.get("capacity_bytes")
        if d.get("kind") == "GPU":
            entry["gpu_cores"] = d.get("gpu_cores")
        domain_id.append(entry)
    payload = {"identity": identity, "domains": domain_id}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _maturity_table(domains: Mapping[str, Any]) -> dict[str, str]:
    return {name: d.get("maturity", "ABSENT") for name, d in domains.items()
            if isinstance(d, dict)}


def build(*, contended: bool, contention_note: str) -> dict[str, Any]:
    identity = discover_identity()
    bandwidth = measure_bandwidth()
    storage = probe_storage()
    ane = probe_ane()
    network = probe_network()
    cpu = _domain_cpu(identity)
    gpu = _domain_gpu(identity, bandwidth)
    uma = _domain_uma(identity)
    fpga = _domain_fpga_declared()
    ext = _domain_external_declared()
    u50dd = _domain_u50dd_declared()
    egpu = _domain_egpu_declared()
    domains = {
        cpu["name"]: cpu,
        gpu["name"]: gpu,
        uma["name"]: uma,
        ane["name"]: ane,
        storage["name"]: storage,
        network["name"]: network,
        fpga["name"]: fpga,
        ext["name"]: ext,
        u50dd["name"]: u50dd,
        egpu["name"]: egpu,
    }
    laws = []
    for d in domains.values():
        for law in d.get("laws") or []:
            laws.append(law)
    genome = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "soc": identity.get("soc"),
        "arch": identity.get("arch"),
        "cpu_cores": identity.get("cpu_cores"),
        "perf_cores": identity.get("perf_cores"),
        "efficiency_cores": identity.get("efficiency_cores"),
        "gpu_cores": identity.get("gpu_cores"),
        "memory_bytes": identity.get("memory_bytes"),
        "os": identity.get("os"),
        "os_product": identity.get("os_product"),
        "toolchain": _toolchain(),
        "measured_bandwidth": bandwidth,
        "measurement_conditions": {
            "contended": contended,
            "note": contention_note,
        },
        "thermal_envelope": {
            "status": "ABSENT",
            "reason": "no sustained thermal campaign has been run; sustained behaviour "
                      "is required before any production ADP (G049) and is not claimed here"},
        "sustained_behaviour": {
            "status": "ABSENT",
            "reason": "microbenchmark only; the steer requires sustained evidence to be "
                      "distinguished from a microbenchmark and this is the latter"},
        "knowledge_level": "INSTANCE",
        "domains": domains,
        "laws": laws,
        "backend_maturity": _maturity_table(domains),
        "evidence_tiers_used": sorted({
            t for d in domains.values()
            for t in [d.get("evidence_tier")]
            if t in EVIDENCE_TIERS
        }),
        "open_domain_schema": True,
        "note": (
            "domains is an open dict keyed by name. FPGA / DGX / eGPU / a future "
            "M-series fill a record via declare_domain(); they do not require a "
            "schema bump. This M3 Ultra is the first real profile of that shape."
        ),
    }
    genome["genome_digest"] = genome_digest(genome)
    return genome


if __name__ == "__main__":
    g = build(
        contended=True,
        contention_note="live HCLI daemon and hf download workers; genome probes are identity + bounded storage reads",
    )
    print(json.dumps(g, indent=2, default=str))
