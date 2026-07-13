#!/usr/bin/env python3.12
"""Receipt-driven control plane for terminal Studio frontier milestones.

This process is deliberately *not* an executor.  It samples cheap, read-only
machine state, validates durable receipts, and atomically publishes the next
blocked/ready milestone.  Network downloads and quantization/transcode workers
are represented as commands-to-be-wired, but are never invoked here.

The separation is intentional: architecture support, a bounded streaming
implementation, and restart-safe output receipts must exist before a future
executor may consume any command shown by this queue.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "hawking.terminal_frontier_queue.v1"
PID_SCHEMA = "hawking.terminal_frontier_queue_pid.v1"
ARCH_SCHEMA = "hawking.download_architecture_ready.v1"
DOWNLOAD_SCHEMA = "hawking.frontier_download_verified.v1"
MILESTONE_SCHEMA = "hawking.terminal_frontier_milestone.v1"
STREAM_READY_SCHEMA = "hawking.frontier_stream_processing_ready.v1"
TRANSCODE_READY_SCHEMA = "hawking.remote_stream_transcode_ready.v1"

POLL_SECONDS = max(30.0, float(os.environ.get("HAWKING_TERMINAL_QUEUE_POLL_S", "120")))
DISK_RESERVE_GB = 150.0
# The supervisor is cheap, but every readiness decision is for a future heavy
# worker.  Any existing swap therefore blocks admission; "some swap is okay"
# would hide a degraded unified-memory baseline.
MAX_SWAP_MB = 0.0
MIN_PHYSICAL_RAM_GIB = 90.0
MAX_LOAD_PER_CPU = 1.50

_stop_requested = False


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _report_dir() -> pathlib.Path:
    return ROOT / "reports" / "condense"


def _state_file() -> pathlib.Path:
    return _report_dir() / "terminal_frontier_queue_state.json"


def _pid_file() -> pathlib.Path:
    return _report_dir() / "terminal_frontier_queue.pid.json"


def _lock_file() -> pathlib.Path:
    return _report_dir() / "terminal_frontier_queue.lock"


def _start_lock_file() -> pathlib.Path:
    return _report_dir() / "terminal_frontier_queue.start.lock"


def _log_file() -> pathlib.Path:
    return _report_dir() / "terminal_frontier_queue.log"


def _drain_file() -> pathlib.Path:
    return ROOT / "reports" / "cron" / "studio_drain.request"


def _rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except (OSError, ValueError):
        return str(path)


def _fsync_dir(path: pathlib.Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _sha256(path: pathlib.Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _stamp(doc: dict[str, Any]) -> Any:
    return doc.get("completed_at") or doc.get("generated_at") or doc.get("timestamp")


def _gate(name: str, ok: bool, reason: str, *, path: pathlib.Path | None = None,
          observed: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "ok": bool(ok), "reason": reason}
    if path is not None:
        out["path"] = _rel(path)
        out["sha256"] = _sha256(path) if path.is_file() else None
    if observed is not None:
        out["observed"] = observed
    return out


def _path_from_receipt(value: Any) -> pathlib.Path:
    path = pathlib.Path(str(value or ""))
    return path if path.is_absolute() else ROOT / path


def _download_gate(label: str, hf_id: str, local_dir: str) -> dict[str, Any]:
    path = _report_dir() / "download_state" / f"{label}.verified.json"
    doc = _read_json(path)
    verification = doc.get("verification") if isinstance(doc.get("verification"), dict) else {}
    expected_dir = (ROOT / local_dir).resolve()
    observed_dir = _path_from_receipt(doc.get("local_dir")).resolve()
    ok = bool(
        doc.get("schema") == DOWNLOAD_SCHEMA
        and doc.get("status") == "verified"
        and doc.get("verified_complete") is True
        and doc.get("label") == label
        and doc.get("hf_id") == hf_id
        and doc.get("hf_download_returncode") == 0
        and verification.get("requested") is True
        and verification.get("returncode") == 0
        and observed_dir == expected_dir
        and _stamp(doc)
        and expected_dir.is_dir()
    )
    reason = (
        "verified download marker is bound to the exact model and staged directory"
        if ok else
        "verified download marker/source is missing or fails schema, identity, path, or verification checks"
    )
    return _gate(f"{label}:verified-download", ok, reason, path=path)


def _architecture_gate(label: str, hf_id: str) -> dict[str, Any]:
    path = _report_dir() / f"{label}.architecture_ready.json"
    doc = _read_json(path)
    ok = bool(
        doc.get("schema") == ARCH_SCHEMA
        and doc.get("status") == "pass"
        and doc.get("architecture_ready") is True
        and doc.get("label") == label
        and doc.get("hf_id") == hf_id
        and _stamp(doc)
    )
    return _gate(
        f"{label}:architecture",
        ok,
        "architecture-ready receipt passes" if ok else
        "architecture-ready receipt is absent or invalid; no architecture support is inferred",
        path=path,
    )


def _source_release_gate(label: str, hf_id: str) -> dict[str, Any]:
    path = _report_dir() / f"{label}_studio_evidence_run.json"
    doc = _read_json(path)
    gates = doc.get("gate") if isinstance(doc.get("gate"), dict) else {}
    decision = (
        doc.get("source_release_decision")
        if isinstance(doc.get("source_release_decision"), dict) else {}
    )
    allowed = {
        "delete_source_after_verified_bake",
        "retain_source_due_license",
        "retain_source_for_rebake",
        "not_applicable_prequantized",
    }
    text_fields = (decision.get("command"), decision.get("reason"), decision.get("decided_by"))
    concrete = all(
        isinstance(value, str) and value.strip() and "<" not in value and "TODO" not in value
        for value in text_fields
    )
    ok = bool(
        doc.get("schema") == "hawking.frontier_studio_evidence_run.v1"
        and doc.get("model") == label
        and doc.get("hf_id") == hf_id
        and doc.get("receipt_state") == "final"
        and doc.get("status") == "pass"
        and gates.get("source_release_decision") is True
        and decision.get("decision") in allowed
        and concrete
        and _stamp(doc)
    )
    return _gate(
        f"{label}:source-release-decision",
        ok,
        "signed-off source lifecycle decision exists (this queue still never deletes source)"
        if ok else
        "final evidence receipt with a concrete source lifecycle decision is required",
        path=path,
        observed={"automatic_deletion": False, "decision": decision.get("decision")},
    )


def _stream_capability_gate(label: str, hf_id: str, architecture: dict[str, Any]) -> dict[str, Any]:
    path = _report_dir() / f"{label}.stream_processing_ready.json"
    doc = _read_json(path)
    peak = doc.get("bounded_peak_memory_gb")
    ok = bool(
        architecture.get("ok")
        and doc.get("schema") == STREAM_READY_SCHEMA
        and doc.get("status") == "pass"
        and doc.get("stream_processing_ready") is True
        and doc.get("label") == label
        and doc.get("hf_id") == hf_id
        and doc.get("architecture_gate_sha256") == architecture.get("sha256")
        and doc.get("checkpoint_resume") is True
        and isinstance(peak, (int, float)) and not isinstance(peak, bool)
        and math.isfinite(float(peak)) and float(peak) <= 65.0
        and _stamp(doc)
    )
    return _gate(
        f"{label}:stream-processing-capability",
        ok,
        "bounded, checkpoint-resumable streamed processor is receipt-validated" if ok else
        "stream processor capability receipt is missing/invalid or not bound to the architecture gate",
        path=path,
    )


def _transcode_capability_gate(architecture: dict[str, Any]) -> dict[str, Any]:
    label, hf_id = "DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro-DSpark"
    path = _report_dir() / f"{label}.stream_transcode_ready.json"
    doc = _read_json(path)
    peak = doc.get("bounded_peak_memory_gb")
    raw_targets = doc.get("targets_bpw")
    targets = sorted(round(float(v), 2) for v in raw_targets) if (
        isinstance(raw_targets, list)
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw_targets)
    ) else []
    roundtrip = doc.get("roundtrip_selftest") if isinstance(doc.get("roundtrip_selftest"), dict) else {}
    ok = bool(
        architecture.get("ok")
        and doc.get("schema") == TRANSCODE_READY_SCHEMA
        and doc.get("status") == "pass"
        and doc.get("remote_shard_stream_transcode_ready") is True
        and doc.get("label") == label
        and doc.get("hf_id") == hf_id
        and doc.get("architecture_gate_sha256") == architecture.get("sha256")
        and doc.get("full_install_forbidden") is True
        and doc.get("checkpoint_resume") is True
        and doc.get("quantization_supported") is True
        and doc.get("doctor_supported") is True
        and roundtrip.get("status") == "pass"
        and targets == [0.25, 0.33, 0.5]
        and isinstance(peak, (int, float)) and not isinstance(peak, bool)
        and math.isfinite(float(peak)) and float(peak) <= 65.0
        and _stamp(doc)
    )
    return _gate(
        f"{label}:remote-stream-transcode-capability",
        ok,
        "remote-shard codec+Doctor implementation is bounded and round-trip validated" if ok else
        "no valid bounded remote-shard codec+Doctor capability receipt exists; sub-bit support is not assumed",
        path=path,
    )


def _frontier_120_completion() -> dict[str, Any]:
    path = _report_dir() / "frontier_stream_queue_state.json"
    doc = _read_json(path)
    items = doc.get("items") if isinstance(doc.get("items"), dict) else {}
    row = items.get("120B") if isinstance(items.get("120B"), dict) else {}
    configs = row.get("configs") if isinstance(row.get("configs"), dict) else {}
    completion = row.get("completion") if isinstance(row.get("completion"), dict) else {}
    campaign = completion.get("document") if isinstance(completion.get("document"), dict) else {}
    ok = bool(
        doc.get("schema") == "hawking.frontier_stream_queue.v1"
        and row.get("status") == "research-complete"
        and configs
        and all(isinstance(v, dict) and v.get("status") == "pass" for v in configs.values())
        and campaign.get("schema") == "hawking.frontier_vtq_shard_campaign.v1"
        and campaign.get("status") == "research-complete"
        and campaign.get("label") == "120B"
        and campaign.get("artifact_class") == "reconstruction_oracle"
        and campaign.get("deployable") is False
        and campaign.get("doctor_run") is False
    )
    return _gate(
        "120B:existing-frontier-handoff",
        ok,
        "existing representative-shard campaign is receipt-complete" if ok else
        "existing 120B frontier-stream campaign has not produced all validated research receipts",
        path=path,
    )


def _milestone_gate(label: str, action: str, *, bpw: float | None = None) -> dict[str, Any]:
    suffix = action if bpw is None else f"{action}.{str(bpw).replace('.', 'p')}-bpw"
    path = _report_dir() / "terminal_milestones" / f"{label}.{suffix}.json"
    doc = _read_json(path)
    try:
        target_matches = bpw is None or math.isclose(
            float(doc.get("target_bpw")), bpw, abs_tol=1e-9
        )
    except (TypeError, ValueError, OverflowError):
        target_matches = False
    ok = bool(
        doc.get("schema") == MILESTONE_SCHEMA
        and doc.get("status") == "complete"
        and doc.get("label") == label
        and doc.get("action") == action
        and target_matches
        and doc.get("checkpoint_resume_validated") is True
        and _stamp(doc)
    )
    return _gate(
        f"{label}:{suffix}:completion",
        ok,
        "external milestone receipt validates" if ok else "external milestone completion receipt is absent or invalid",
        path=path,
    )


def _probe(argv: list[str]) -> tuple[int, str]:
    """Run only an allowlisted, local read-only OS probe."""
    if not argv or pathlib.Path(argv[0]).name not in {"sysctl", "pmset"}:
        raise ValueError("non-read-only probe rejected")
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False)
        return result.returncode, (result.stdout or result.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return 127, ""


def _thermal_ok(returncode: int, text: str) -> bool:
    low = text.lower()
    explicit = (
        "no thermal warning level has been recorded" in low
        and "no performance warning level has been recorded" in low
    )
    numeric = {key.lower(): int(value) for key, value in re.findall(r"([A-Za-z_]+)\s*[:=]\s*(\d+)", text)}
    return bool(
        returncode == 0 and (
            explicit or (
                {"cpu_speed_limit", "scheduler_limit", "available_cpus"}.issubset(numeric)
                and numeric["cpu_speed_limit"] >= 100
                and numeric["scheduler_limit"] >= 100
                and numeric["available_cpus"] > 0
            )
        )
    )


def _resource_snapshot() -> dict[str, Any]:
    pressure_rc, pressure_text = _probe(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    swap_rc, swap_text = _probe(["sysctl", "-n", "vm.swapusage"])
    memory_rc, memory_text = _probe(["sysctl", "-n", "hw.memsize"])
    power_rc, power_text = _probe(["pmset", "-g", "batt"])
    thermal_rc, thermal_text = _probe(["pmset", "-g", "therm"])
    try:
        pressure = int(pressure_text) if pressure_rc == 0 else None
    except ValueError:
        pressure = None
    match = re.search(r"used\s*=\s*([0-9.]+)([MGT])", swap_text)
    swap_mb = None
    if swap_rc == 0 and match:
        swap_mb = float(match.group(1)) * {"M": 1.0, "G": 1024.0, "T": 1024.0 ** 2}[match.group(2)]
    try:
        physical_gib = int(memory_text) / (1024.0 ** 3) if memory_rc == 0 else None
    except ValueError:
        physical_gib = None
    usage = shutil.disk_usage(ROOT)
    load = os.getloadavg()
    cpus = os.cpu_count() or 1
    return {
        "schema": "hawking.terminal_frontier_resources.v1",
        "sampled_at": _now(),
        "ok": pressure in {1, 2, 4} and swap_mb is not None and physical_gib is not None,
        "pressure_level": pressure,
        "pressure_name": {1: "normal", 2: "warning", 4: "critical"}.get(pressure, "unknown"),
        "swap_used_mb": round(swap_mb, 3) if swap_mb is not None else None,
        "physical_ram_gib": round(physical_gib, 3) if physical_gib is not None else None,
        "disk_total_gb": round(usage.total / 1e9, 3),
        "disk_free_gb": round(usage.free / 1e9, 3),
        "load_1m": round(load[0], 3),
        "load_5m": round(load[1], 3),
        "logical_cpus": cpus,
        "load_per_cpu_1m": round(load[0] / cpus, 3),
        "power_probe_ok": power_rc == 0,
        "power_source": power_text.splitlines()[0] if power_text else None,
        "thermal_ok": _thermal_ok(thermal_rc, thermal_text),
        "thermal_detail": thermal_text[-1000:],
    }


def _safety_gate(snapshot: dict[str, Any], required_disk_free_gb: float) -> dict[str, Any]:
    blockers: list[str] = []
    if snapshot.get("ok") is not True:
        blockers.append("memory resource probes unavailable")
    if snapshot.get("pressure_level") != 1:
        blockers.append(f"memory pressure is not normal (level={snapshot.get('pressure_level')!r})")
    swap = snapshot.get("swap_used_mb")
    if not isinstance(swap, (int, float)) or isinstance(swap, bool) or not math.isfinite(float(swap)):
        blockers.append("swap measurement unavailable")
    elif float(swap) > MAX_SWAP_MB:
        blockers.append(f"swap {float(swap):.1f}MB > {MAX_SWAP_MB:.1f}MB ceiling")
    ram = snapshot.get("physical_ram_gib")
    if not isinstance(ram, (int, float)) or float(ram) < MIN_PHYSICAL_RAM_GIB:
        blockers.append(f"physical RAM does not validate as the 96 GiB class (observed {ram!r}GiB)")
    disk = snapshot.get("disk_free_gb")
    if not isinstance(disk, (int, float)) or float(disk) < required_disk_free_gb:
        blockers.append(f"disk free {disk!r}GB < required {required_disk_free_gb:.3f}GB")
    if snapshot.get("thermal_ok") is not True:
        blockers.append("thermal/performance state is not explicitly green")
    if "AC Power" not in str(snapshot.get("power_source", "")):
        blockers.append("AC power is not confirmed")
    load_per_cpu = snapshot.get("load_per_cpu_1m")
    if not isinstance(load_per_cpu, (int, float)) or float(load_per_cpu) > MAX_LOAD_PER_CPU:
        blockers.append(f"1-minute load per CPU is unavailable or above {MAX_LOAD_PER_CPU:.2f}")
    return _gate(
        f"resources:disk-{required_disk_free_gb:.3f}GB",
        not blockers,
        "resource envelope is green" if not blockers else "; ".join(blockers),
        observed={
            "required_disk_free_gb": round(required_disk_free_gb, 3),
            "disk_reserve_gb": DISK_RESERVE_GB,
            "pressure_level": snapshot.get("pressure_level"),
            "swap_used_mb": snapshot.get("swap_used_mb"),
            "disk_free_gb": snapshot.get("disk_free_gb"),
            "load_per_cpu_1m": snapshot.get("load_per_cpu_1m"),
        },
    )


def _dependency(name: str, complete: bool) -> dict[str, Any]:
    return _gate(name, complete, "dependency complete" if complete else "prior milestone is incomplete")


def _command(argv: list[str], *, worker_exists: bool) -> dict[str, Any]:
    return {
        "argv": argv,
        "wired": False,
        "worker_exists": bool(worker_exists),
        "execution": "display-only; terminal_frontier_queue never invokes this command",
        "resume_required": True,
    }


def _phase(phase_id: str, label: str, action: str, requirements: list[dict[str, Any]],
           completion: dict[str, Any], command: dict[str, Any], **extra: Any) -> dict[str, Any]:
    complete = completion.get("ok") is True
    gate_ready = all(row.get("ok") is True for row in requirements)
    blockers = [] if complete else [row["reason"] for row in requirements if row.get("ok") is not True]
    if not complete and gate_ready:
        blockers.append("execution adapter is intentionally unwired; operator review and a real worker are required")
    row: dict[str, Any] = {
        "id": phase_id,
        "label": label,
        "action": action,
        "status": "external-complete" if complete else "planned-blocked",
        "gate_ready": bool(gate_ready),
        "blockers": blockers,
        "requirements": requirements,
        "completion": completion,
        "command_to_be_wired": command,
    }
    row.update(extra)
    return row


def build_state(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    resources = snapshot if snapshot is not None else _resource_snapshot()
    p120 = _download_gate("120B", "openai/gpt-oss-120b", "scratch/staging/gpt-oss-120b.partial")
    c120 = _frontier_120_completion()
    a_flash = _architecture_gate("DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-DSpark")
    d_flash = _download_gate(
        "DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-DSpark",
        "scratch/staging/deepseek-v4-flash-dspark.partial",
    )
    stream_flash = _stream_capability_gate(
        "DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-DSpark", a_flash
    )
    c_flash = _milestone_gate("DeepSeek-V4-Flash", "streamed-processing")
    release_flash = _source_release_gate(
        "DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-DSpark"
    )
    a_kimi = _architecture_gate("Kimi-K2.6", "moonshotai/Kimi-K2.6")
    d_kimi = _download_gate("Kimi-K2.6", "moonshotai/Kimi-K2.6", "scratch/staging/kimi-k2.6.partial")
    stream_kimi = _stream_capability_gate("Kimi-K2.6", "moonshotai/Kimi-K2.6", a_kimi)
    c_kimi = _milestone_gate("Kimi-K2.6", "streamed-processing")
    release_kimi = _source_release_gate("Kimi-K2.6", "moonshotai/Kimi-K2.6")
    a_pro = _architecture_gate("DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro-DSpark")
    transcode = _transcode_capability_gate(a_pro)

    phases: list[dict[str, Any]] = []
    phases.append(_phase(
        "120b-handoff", "120B", "existing-frontier-handoff",
        [p120, _safety_gate(resources, DISK_RESERVE_GB + 24.0)], c120,
        _command([sys.executable, "tools/condense/frontier_stream_queue.py", "start"], worker_exists=True),
        mode="existing-handoff", source_download_gb=None,
        claim_limit="representative-shard reconstruction research only; not deployable quality evidence",
    ))
    phases.append(_phase(
        "v4-flash-download", "DeepSeek-V4-Flash", "full-download",
        [_dependency("after:120b-handoff", c120["ok"]), a_flash,
         _safety_gate(resources, DISK_RESERVE_GB + 168.0 + 16.0)], d_flash,
        _command([
            sys.executable, "tools/condense/procure.py", "DeepSeek-V4-Flash",
            "--dir", "scratch/staging/deepseek-v4-flash-dspark.partial", "--verify",
            "--retries", "2", "--progress-interval-s", "60", "--stall-timeout-s", "900",
        ], worker_exists=True),
        mode="full-download", source_download_gb=168.0,
        checkpoint="reports/condense/download_state/DeepSeek-V4-Flash.state.json",
    ))
    phases.append(_phase(
        "v4-flash-stream-process", "DeepSeek-V4-Flash", "streamed-processing",
        [_dependency("after:v4-flash-download", d_flash["ok"]),
         _dependency("after:120b-handoff", c120["ok"]), a_flash, stream_flash,
         _safety_gate(resources, DISK_RESERVE_GB + 47.57 + 32.0)], c_flash,
        _command([
            sys.executable, "tools/condense/terminal_frontier_worker.py", "stream-process",
            "--label", "DeepSeek-V4-Flash", "--source",
            "scratch/staging/deepseek-v4-flash-dspark.partial", "--checkpoint-resume",
        ], worker_exists=(ROOT / "tools/condense/terminal_frontier_worker.py").is_file()),
        mode="streamed-processing", bounded_peak_memory_required=True,
        target_bpw=[2.00, 1.34, 1.00, 0.80, 0.50],
        claim_limit="oracle-to-packed-to-native promotion is required at every target",
        checkpoint="reports/condense/terminal_checkpoints/DeepSeek-V4-Flash",
    ))
    phases.append(_phase(
        "kimi-k2p6-install", "Kimi-K2.6", "full-install",
        [_dependency("after:v4-flash-stream-process", c_flash["ok"]), release_flash, a_kimi,
         _safety_gate(resources, DISK_RESERVE_GB + 595.0 + 32.0)], d_kimi,
        _command([
            sys.executable, "tools/condense/procure.py", "Kimi-K2.6", "--dir",
            "scratch/staging/kimi-k2.6.partial", "--verify", "--retries", "2",
            "--progress-interval-s", "60", "--stall-timeout-s", "900",
        ], worker_exists=True),
        mode="full-install", source_download_gb=595.0,
        checkpoint="reports/condense/download_state/Kimi-K2.6.state.json",
        serve_math={
            "0.50_bpw_weight_gb": 68.75,
            "0.50_bpw": "nominally resident but runtime-overhead tight",
            "0.33_bpw_weight_gb": 45.375,
            "0.33_bpw": "safer resident research target after codec support exists",
        },
    ))

    phases.append(_phase(
        "kimi-k2p6-stream-process", "Kimi-K2.6", "streamed-processing",
        [_dependency("after:kimi-k2p6-install", d_kimi["ok"]), a_kimi, stream_kimi,
         _safety_gate(resources, DISK_RESERVE_GB + 68.75 + 64.0)], c_kimi,
        _command([
            sys.executable, "tools/condense/terminal_frontier_worker.py", "stream-process",
            "--label", "Kimi-K2.6", "--source", "scratch/staging/kimi-k2.6.partial",
            "--target-bpw", "0.80,0.50,0.33,0.25", "--checkpoint-resume",
        ], worker_exists=(ROOT / "tools/condense/terminal_frontier_worker.py").is_file()),
        mode="streamed-processing", bounded_peak_memory_required=True,
        target_bpw=[0.80, 0.50, 0.33, 0.25],
        claim_limit="no deployable claim until packed round trip, native parity, and resident capability evidence",
        checkpoint="reports/condense/terminal_checkpoints/Kimi-K2.6",
    ))

    prior_complete = c_kimi["ok"]
    prior_target_complete = True
    retained_outputs_gb = 0.0
    for bpw, output_gb, residency in (
        (0.50, 100.0, "nonresident: 100GB weights exceed the 78GB resident envelope; out-of-core only"),
        (0.33, 66.0, "nominally resident but borderline after runtime/KV/Doctor overhead"),
        (0.25, 50.0, "safe resident target within the 78GB envelope, subject to measured runtime overhead"),
    ):
        completion = _milestone_gate("DeepSeek-V4-Pro", "remote-shard-stream-transcode", bpw=bpw)
        requirements = [
            _dependency("after:kimi-k2p6-stream-process", prior_complete),
            _dependency("after:previous-v4-pro-target", prior_target_complete),
            release_kimi, a_pro, transcode,
            _safety_gate(resources, DISK_RESERVE_GB + 64.0 + retained_outputs_gb + output_gb),
        ]
        token = str(bpw).replace(".", "p")
        phases.append(_phase(
            f"v4-pro-{token}-bpw", "DeepSeek-V4-Pro", "remote-shard-stream-transcode",
            requirements, completion,
            _command([
                sys.executable, "tools/condense/terminal_frontier_worker.py",
                "remote-shard-stream-transcode", "--label", "DeepSeek-V4-Pro",
                "--hf-id", "deepseek-ai/DeepSeek-V4-Pro-DSpark", "--target-bpw", f"{bpw:.2f}",
                "--never-full-install", "--checkpoint-resume",
            ], worker_exists=(ROOT / "tools/condense/terminal_frontier_worker.py").is_file()),
            mode="remote-shard-stream-transcode", target_bpw=bpw,
            estimated_output_gb=output_gb, residency=residency,
            full_install_permitted=False, source_download_gb=892.763,
            source_policy="remote shards only; consume one bounded shard window, checkpoint, then release it",
            checkpoint=f"reports/condense/terminal_checkpoints/DeepSeek-V4-Pro/{token}-bpw",
        ))
        retained_outputs_gb += output_gb
        prior_target_complete = completion["ok"]

    pending = next((row for row in phases if row["status"] != "external-complete"), None)
    status = "complete" if pending is None else "planned-blocked"
    return {
        "schema": SCHEMA,
        "created_at": _now(),
        "updated_at": _now(),
        "status": status,
        "profile": "Studio-M3Ultra-96GB-1TB",
        "poll_seconds": POLL_SECONDS,
        "executor_policy": {
            "network_calls": False,
            "heavy_compute": False,
            "invoke_planned_commands": False,
            "automatic_source_deletion": False,
            "architecture_inference": False,
            "unsupported_quantization": False,
        },
        "restart_safety": {
            "state_write": "temp+fsync+atomic-replace+directory-fsync",
            "progress_authority": "external immutable receipts, never process exit codes",
            "commands_require_checkpoint_resume": True,
            "power_interruption": "restart queue; it revalidates receipts and resumes at first incomplete phase",
        },
        "resources": resources,
        "drain_requested": _drain_file().exists(),
        "phases": phases,
        "progress": {
            "complete": sum(row["status"] == "external-complete" for row in phases),
            "total": len(phases),
            "next_phase": pending["id"] if pending else None,
            "next_gate_ready": pending["gate_ready"] if pending else None,
            "next_blockers": pending["blockers"] if pending else [],
        },
        "invariants": {
            "v4_pro_full_install_forbidden": True,
            "v4_pro_remote_source_gb": 892.763,
            "v4_pro_targets_bpw": [0.50, 0.33, 0.25],
            "no_action_is_executed_by_this_process": True,
        },
    }


def run_once(snapshot: dict[str, Any] | None = None, *, write: bool = True) -> dict[str, Any]:
    state = build_state(snapshot)
    if write:
        old = _read_json(_state_file())
        if old.get("schema") == SCHEMA and old.get("created_at"):
            state["created_at"] = old["created_at"]
        _atomic_json(_state_file(), state)
    return state


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _daemon_status() -> dict[str, Any]:
    doc = _read_json(_pid_file())
    worker = doc.get("worker_pid")
    wrapper = doc.get("wrapper_pid") or doc.get("pid")
    return {
        "running": _pid_alive(worker) or _pid_alive(wrapper),
        "worker_pid": worker,
        "wrapper_pid": wrapper,
        "pgid": doc.get("pgid"),
        "detached": doc.get("detached"),
        "pid_file": _rel(_pid_file()),
        "log": _rel(_log_file()),
    }


def _sleep_interruptible(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not _stop_requested and time.monotonic() < deadline:
        time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))


def run_queue() -> int:
    global _stop_requested
    _lock_file().parent.mkdir(parents=True, exist_ok=True)
    singleton = open(_lock_file(), "a+", encoding="utf-8")
    try:
        fcntl.flock(singleton.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        singleton.close()
        print("terminal-frontier queue already owns the singleton lock", file=sys.stderr)
        return 2

    def request_stop(_sig: int, _frame: Any) -> None:
        global _stop_requested
        _stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    previous = _read_json(_pid_file())
    detached = bool(previous.get("detached"))
    pgid = previous.get("pgid") if detached else os.getpgrp()
    _atomic_json(_pid_file(), {
        "schema": PID_SCHEMA,
        "worker_pid": os.getpid(),
        "wrapper_pid": previous.get("wrapper_pid"),
        "pgid": pgid,
        "detached": detached,
        "started_at": previous.get("started_at") or _now(),
        "poll_seconds": POLL_SECONDS,
        "log": _rel(_log_file()),
    })
    rc = 0
    try:
        while not _stop_requested:
            state = run_once()
            if state["status"] == "complete":
                break
            _sleep_interruptible(POLL_SECONDS)
        if _stop_requested:
            state = run_once()
            state["status"] = "stopped"
            state["stopped_at"] = _now()
            _atomic_json(_state_file(), state)
            rc = 130
        return rc
    finally:
        info = _read_json(_pid_file())
        if info.get("worker_pid") == os.getpid():
            try:
                _pid_file().unlink()
                _fsync_dir(_pid_file().parent)
            except FileNotFoundError:
                pass
        fcntl.flock(singleton.fileno(), fcntl.LOCK_UN)
        singleton.close()


def _detached_argv() -> list[str]:
    worker = [sys.executable, str(pathlib.Path(__file__).resolve()), "run"]
    caffeinate = shutil.which("caffeinate")
    return [caffeinate, "-dimsu", *worker] if caffeinate else worker


def start_queue() -> int:
    _start_lock_file().parent.mkdir(parents=True, exist_ok=True)
    start_lock = open(_start_lock_file(), "a+", encoding="utf-8")
    fcntl.flock(start_lock.fileno(), fcntl.LOCK_EX)
    try:
        status = _daemon_status()
        if status["running"]:
            print(json.dumps({"status": "already-running", "daemon": status}, indent=2))
            return 0
        _log_file().parent.mkdir(parents=True, exist_ok=True)
        with open(_log_file(), "a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                _detached_argv(), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True,
            )
        _atomic_json(_pid_file(), {
            "schema": PID_SCHEMA,
            "pid": proc.pid,
            "wrapper_pid": proc.pid,
            "worker_pid": None,
            "pgid": proc.pid,
            "detached": True,
            "started_at": _now(),
            "poll_seconds": POLL_SECONDS,
            "command": _detached_argv(),
            "log": _rel(_log_file()),
        })
        print(json.dumps({"status": "started", "pid": proc.pid, "log": _rel(_log_file())}, indent=2))
        return 0
    finally:
        fcntl.flock(start_lock.fileno(), fcntl.LOCK_UN)
        start_lock.close()


def stop_queue() -> int:
    doc = _read_json(_pid_file())
    worker = doc.get("worker_pid")
    wrapper = doc.get("wrapper_pid") or doc.get("pid")
    if not (_pid_alive(worker) or _pid_alive(wrapper)):
        print(json.dumps({"status": "not-running", "daemon": _daemon_status()}, indent=2))
        return 0
    try:
        if doc.get("detached") is True and isinstance(doc.get("pgid"), int):
            os.killpg(int(doc["pgid"]), signal.SIGTERM)
        elif _pid_alive(worker):
            os.kill(int(worker), signal.SIGTERM)
        else:
            os.kill(int(wrapper), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "stop-failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(json.dumps({"status": "stop-signalled", "daemon": _daemon_status()}, indent=2))
    return 0


def show_status(refresh: bool = False) -> int:
    state = run_once(write=False) if refresh or not _state_file().is_file() else _read_json(_state_file())
    print(json.dumps({"daemon": _daemon_status(), "state": state}, indent=2, sort_keys=True))
    return 0


def _green_snapshot(*, disk_free_gb: float = 900.0) -> dict[str, Any]:
    return {
        "schema": "hawking.terminal_frontier_resources.v1",
        "sampled_at": _now(), "ok": True, "pressure_level": 1,
        "pressure_name": "normal", "swap_used_mb": 0.0, "physical_ram_gib": 96.0,
        "disk_total_gb": 1000.0, "disk_free_gb": disk_free_gb,
        "load_1m": 1.0, "load_5m": 1.0, "logical_cpus": 16,
        "load_per_cpu_1m": 0.0625, "power_probe_ok": True,
        "power_source": "Now drawing from 'AC Power'", "thermal_ok": True,
        "thermal_detail": "selftest green",
    }


def selftest() -> int:
    global ROOT
    original_root = ROOT
    try:
        with tempfile.TemporaryDirectory(prefix="hawking-terminal-frontier-") as tmp:
            ROOT = pathlib.Path(tmp)
            state = run_once(_green_snapshot(), write=True)
            assert state["schema"] == SCHEMA
            assert state["status"] == "planned-blocked"
            assert state["progress"]["next_phase"] == "120b-handoff"
            assert _read_json(_state_file())["schema"] == SCHEMA

            staged = ROOT / "scratch/staging/gpt-oss-120b.partial"
            staged.mkdir(parents=True)
            marker = _report_dir() / "download_state/120B.verified.json"
            _atomic_json(marker, {
                "schema": DOWNLOAD_SCHEMA, "status": "verified", "verified_complete": True,
                "label": "120B", "hf_id": "openai/gpt-oss-120b",
                "local_dir": "scratch/staging/gpt-oss-120b.partial",
                "hf_download_returncode": 0,
                "verification": {"requested": True, "returncode": 0},
                "completed_at": _now(),
            })
            state = run_once(_green_snapshot(), write=True)
            first = state["phases"][0]
            assert first["status"] == "planned-blocked" and first["gate_ready"] is True
            assert "intentionally unwired" in first["blockers"][0]

            red = _green_snapshot(disk_free_gb=100.0)
            red["pressure_level"] = 4
            red["swap_used_mb"] = 8192.0
            checked = _safety_gate(red, 174.0)
            assert checked["ok"] is False and "pressure" in checked["reason"]

            pro = [row for row in state["phases"] if row["label"] == "DeepSeek-V4-Pro"]
            assert [row["target_bpw"] for row in pro] == [0.50, 0.33, 0.25]
            assert all(row["full_install_permitted"] is False for row in pro)
            assert all(row["mode"] == "remote-shard-stream-transcode" for row in pro)
            assert all("--never-full-install" in row["command_to_be_wired"]["argv"] for row in pro)
            assert all(row["command_to_be_wired"]["wired"] is False for row in state["phases"])
            assert pro[0]["estimated_output_gb"] == 100.0 and "nonresident" in pro[0]["residency"]
            assert pro[1]["estimated_output_gb"] == 66.0 and "borderline" in pro[1]["residency"]
            assert pro[2]["estimated_output_gb"] == 50.0 and "safe resident" in pro[2]["residency"]
            assert _detached_argv()[-1] == "run"
        print("terminal_frontier_queue selftest: PASS")
        return 0
    finally:
        ROOT = original_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start", help="start the detached read-only supervisor")
    sub.add_parser("run", help="run the read-only supervisor in the foreground")
    status_parser = sub.add_parser("status", help="show persisted state and daemon status")
    status_parser.add_argument("--refresh", action="store_true", help="sample read-only gates without writing state")
    sub.add_parser("stop", help="signal the detached supervisor to stop")
    sub.add_parser("selftest", help="run isolated tests; never starts a daemon or worker")
    args = parser.parse_args(argv)
    if args.command == "start":
        return start_queue()
    if args.command == "run":
        return run_queue()
    if args.command == "status":
        return show_status(refresh=args.refresh)
    if args.command == "stop":
        return stop_queue()
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
