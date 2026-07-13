#!/usr/bin/env python3.12
"""Execute one fail-closed Doctor-v5 Pass-B codec-control cell.

This is the first *executing* package after the immutable Pass-A census.  It is
intentionally narrow: a reviewed STRAND scalar profile, a source-bound request,
phase-durable checkpoints, a complete physical bundle (packed projections plus
lossless pass-through tensors), and real forward-pass observations.  Results are
engineering evidence only; this worker cannot issue a sealed-quality or dominance
claim.

The worker never downloads or deletes a model.  It accepts no command string and
never invokes a shell.  Every child argv is derived from the typed request and the
compiled profile below.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PASS_A_INDEX = ROOT / "reports/condense/doctor_v5_scale/index.json"
DEFAULT_ROOT = ROOT / "reports/condense/doctor_v5_pass_b"
HEAVY_LEASE_FD_ENV = "HAWKING_HEAVY_LEASE_FD"
HEAVY_LOCK = ROOT / "reports/cron/studio_heavy.lock"
REQUEST_SCHEMA = "hawking.doctor_v5_pass_b_pilot_request.v1"
CHECKPOINT_SCHEMA = "hawking.doctor_v5_pass_b_phase_checkpoint.v1"
RECEIPT_SCHEMA = "hawking.doctor_v5_pass_b_execution_receipt.v1"
RESOURCE_SCHEMA = "hawking.doctor_v5_pass_b_resource_sample.v1"
PROFILE = "strand-scalar-quality-rhtcols-v1"
QUANTIZABLE_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)
METADATA_NAMES = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "merges.txt",
    "vocab.json", "tokenizer.model", "added_tokens.json",
)
PHASES = (
    "preflight", "baseline_ppl", "pass_through", "packed_encode",
    "archive_attest", "archive_decode", "reconstruction_ppl", "baseline_capability",
    "reconstruction_capability", "receipt",
)
MAX_JSON_BYTES = 32 * 1024 * 1024
DEFAULT_DISK_RESERVE_BYTES = 150_000_000_000
_STOP_REQUESTED = False


class PassBError(RuntimeError):
    pass


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(tmp, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short atomic write")
            view = view[written:]
        os.fsync(fd)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    finally:
        os.close(fd)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True,
                                  ensure_ascii=False).encode("utf-8") + b"\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_JSON_BYTES:
            raise PassBError(f"invalid JSON file size/type: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PassBError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PassBError(f"JSON root must be an object: {path}")
    return value


def _hash_file(path: Path) -> tuple[str, int]:
    """Hash a stable regular file opened without following its final symlink."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PassBError(f"cannot open {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PassBError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(fd)
        if not (before.st_dev == after.st_dev and before.st_ino == after.st_ino
                and before.st_size == after.st_size == total
                and before.st_mtime_ns == after.st_mtime_ns):
            raise PassBError(f"file changed while hashing: {path}")
        return digest.hexdigest(), total
    finally:
        os.close(fd)


def _inside_workspace(raw: Any, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PassBError("path must be a non-empty string")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(ROOT.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise PassBError(f"path is missing or outside workspace: {raw!r}") from exc
    if must_exist and candidate.is_symlink():
        raise PassBError(f"symlink input forbidden: {raw!r}")
    return resolved


def _system_python(raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PassBError("python_path must be a non-empty string")
    candidate = Path(raw)
    try:
        resolved = candidate.resolve(strict=True)
        current = Path(sys.executable).resolve(strict=True)
        st = candidate.lstat()
    except (OSError, RuntimeError) as exc:
        raise PassBError(f"cannot resolve pinned Python executable: {raw!r}") from exc
    if resolved != current or candidate.is_symlink() or not stat.S_ISREG(st.st_mode):
        raise PassBError("python_path must be this worker's exact regular-file interpreter")
    return resolved


def _command(argv: list[str]) -> dict[str, Any]:
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=20,
                            check=False)
        return {"argv": argv, "returncode": cp.returncode,
                "stdout": cp.stdout.strip(), "stderr": cp.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": str(exc)}


def _swap_used_bytes(text: str) -> int | None:
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", text)
    if not match:
        return None
    scale = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def _resource_sample(output_root: Path) -> dict[str, Any]:
    pressure = _command(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    swap = _command(["sysctl", "-n", "vm.swapusage"])
    power = _command(["pmset", "-g", "batt"])
    thermal = _command(["pmset", "-g", "therm"])
    usage = shutil.disk_usage(output_root)
    try:
        level = int(pressure["stdout"].strip()) if pressure["returncode"] == 0 else None
    except ValueError:
        level = None
    return {
        "schema": RESOURCE_SCHEMA,
        "sampled_at": _now(),
        "memory_pressure_level": level,
        "swap_used_bytes": _swap_used_bytes(swap["stdout"]),
        "ac_power": power["returncode"] == 0 and "AC Power" in power["stdout"],
        "thermal_nominal": thermal["returncode"] == 0 and not re.search(
            r"(warning level|performance warning|CPU power status)\s*:\s*[1-9]",
            thermal["stdout"], re.IGNORECASE),
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "raw": {"pressure": pressure, "swap": swap, "power": power, "thermal": thermal},
    }


def _resource_gate(sample: dict[str, Any], request: dict[str, Any]) -> None:
    resources = request["resources"]
    reserve = resources["disk_reserve_bytes"]
    scratch = resources["scratch_budget_bytes"]
    errors: list[str] = []
    if sample["memory_pressure_level"] != 1:
        errors.append(f"memory pressure is not normal: {sample['memory_pressure_level']!r}")
    if sample["swap_used_bytes"] != 0:
        errors.append(f"swap is not exactly zero: {sample['swap_used_bytes']!r}")
    if not sample["ac_power"]:
        errors.append("AC power is not confirmed")
    if not sample["thermal_nominal"]:
        errors.append("thermal state is not nominal")
    if sample["disk_free_bytes"] < reserve + scratch:
        errors.append(
            f"disk free {sample['disk_free_bytes']} < reserve+scratch {reserve + scratch}"
        )
    if errors:
        raise PassBError("resource admission failed: " + "; ".join(errors))


def _validate_heavy_lease() -> dict[str, Any]:
    raw = os.environ.get(HEAVY_LEASE_FD_ENV)
    if not raw or not raw.isdigit():
        raise PassBError(f"missing inherited {HEAVY_LEASE_FD_ENV}")
    fd = int(raw)
    try:
        fst = os.fstat(fd)
        lst = HEAVY_LOCK.stat()
    except OSError as exc:
        raise PassBError(f"invalid inherited heavy lease fd {fd}: {exc}") from exc
    if not stat.S_ISREG(fst.st_mode):
        raise PassBError("heavy lease fd is not a regular lock file")
    if (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino):
        raise PassBError("heavy lease fd does not identify the canonical heavy lock")
    # The queue owns the inherited open-file description.  Do not unlock it here.
    return {"fd": fd, "path": str(HEAVY_LOCK), "st_dev": fst.st_dev,
            "st_ino": fst.st_ino}


def _install_signal_handlers() -> None:
    def handler(_signum: int, _frame: Any) -> None:
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _artifact(path: Path) -> dict[str, Any]:
    sha, size = _hash_file(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha, "bytes": size}


def _commit_generated(tmp: Path, final: Path) -> None:
    if not tmp.is_file() or tmp.stat().st_size <= 0:
        raise PassBError(f"child did not produce a non-empty file: {tmp}")
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)
    _fsync_dir(final.parent)


def _str2_source_sha256(path: Path) -> str:
    """Return the source digest embedded in the fixed STR2 header."""
    try:
        with path.open("rb") as handle:
            header = handle.read(52)
    except OSError as exc:
        raise PassBError(f"cannot read STR2 header {path}: {exc}") from exc
    if len(header) != 52 or header[:4] != b"STR2":
        raise PassBError("packed archive is truncated or does not have STR2 magic")
    return header[20:52].hex()


def _last_json_line(path: Path) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "event" not in row:
                latest = row
    except OSError as exc:
        raise PassBError(f"cannot read child log {path}: {exc}") from exc
    if latest is None:
        raise PassBError(f"child log has no JSON result: {path}")
    return latest


def _run_logged(argv: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    """Run one fixed argv without a shell; retain a durable, append-only phase log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        header = _canonical_bytes({"event": "child_start", "at": _now(), "argv": argv}) + b"\n"
        log.write(header)
        os.fsync(log.fileno())
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, shell=False,
                                close_fds=True)
        rc = proc.wait()
        log.write(_canonical_bytes({"event": "child_exit", "at": _now(),
                                   "returncode": rc}) + b"\n")
        os.fsync(log.fileno())
    _fsync_dir(log_path.parent)
    if rc != 0:
        raise PassBError(f"child failed with rc={rc}; see {log_path}")


def _request_paths(request: dict[str, Any]) -> dict[str, Path]:
    run_root = _inside_workspace(request["output_root"], must_exist=False)
    return {
        "run": run_root,
        "checkpoint": run_root / "checkpoint.json",
        "receipt": run_root / "execution_receipt.json",
        "logs": run_root / "logs",
        "bundle": run_root / "bundle",
        "packed": run_root / "bundle/projections.strand",
        "passthrough": run_root / "bundle/passthrough.safetensors",
        "recon": run_root / "evaluation/reconstruction.safetensors",
        "attestation": run_root / "bundle/attestation.json",
        "baseline_ppl": run_root / "evaluation/baseline_ppl.json",
        "recon_ppl": run_root / "evaluation/reconstruction_ppl.json",
        "baseline_cap": run_root / "evaluation/baseline_capability.json",
        "recon_cap": run_root / "evaluation/reconstruction_capability.json",
        "bundle_manifest": run_root / "bundle/manifest.json",
    }


def _validate_request(path: Path) -> tuple[dict[str, Any], str, dict[str, Path]]:
    request = _load_json(path)
    expected = {
        "schema", "request_id", "label", "profile", "bits", "source",
        "parameter_manifest", "adapter", "execution", "resources", "output_root",
        "evidence_policy",
    }
    if set(request) != expected:
        raise PassBError(f"request keys mismatch: {sorted(set(request) ^ expected)}")
    if request["schema"] != REQUEST_SCHEMA or request["label"] != "0.5B":
        raise PassBError("pilot request must be schema v1 and label 0.5B")
    if request["profile"] != PROFILE or request["bits"] != 4:
        raise PassBError("first pilot is pinned to the reviewed 4-bit scalar quality profile")
    if not isinstance(request["request_id"], str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9._-]{7,127}", request["request_id"]):
        raise PassBError("invalid request_id")
    if request["evidence_policy"] != {
        "class": "provisional_engineering_evidence",
        "dominance_claim_permitted": False,
        "sealed_quality_claim_permitted": False,
        "source_deletion_permitted": False,
    }:
        raise PassBError("evidence policy must retain every fail-closed limitation")

    source = request["source"]
    if not isinstance(source, dict) or set(source) != {
            "model_dir", "weight_file", "weight_sha256", "weight_bytes",
            "census_report", "census_report_sha256", "pass_a_index_sha256",
            "source_manifest_sha256"}:
        raise PassBError("source binding keys mismatch")
    model_dir = _inside_workspace(source["model_dir"])
    weight = _inside_workspace(source["weight_file"])
    census = _inside_workspace(source["census_report"])
    if model_dir not in weight.parents or model_dir not in census.parents:
        # Census lives in reports, while weight must live under the model.  Correct the
        # second half explicitly; keeping this branch makes the intended split obvious.
        if model_dir not in weight.parents:
            raise PassBError("weight_file must be inside model_dir")
    for key in ("weight_sha256", "census_report_sha256", "pass_a_index_sha256",
                "source_manifest_sha256"):
        if not _is_sha(source[key]):
            raise PassBError(f"invalid source hash: {key}")
    if not isinstance(source["weight_bytes"], int) or source["weight_bytes"] <= 0:
        raise PassBError("invalid weight_bytes")

    parameter = request["parameter_manifest"]
    adapter = request["adapter"]
    execution = request["execution"]
    resources = request["resources"]
    if not isinstance(parameter, dict) or set(parameter) != {"path", "sha256"}:
        raise PassBError("parameter_manifest binding keys mismatch")
    if not isinstance(adapter, dict) or set(adapter) != {
            "registry_path", "registry_sha256", "adapter_id", "adapter_source_sha256"}:
        raise PassBError("adapter binding keys mismatch")
    if not isinstance(execution, dict) or set(execution) != {
            "worker_path", "worker_sha256", "quantizer_path", "quantizer_sha256",
            "attest_path", "attest_sha256", "decoder_path", "decoder_sha256",
            "ppl_bench_path", "ppl_bench_sha256", "multi_eval_path",
            "multi_eval_sha256", "python_path", "python_sha256", "threads"}:
        raise PassBError("execution binding keys mismatch")
    if not isinstance(resources, dict) or set(resources) != {
            "disk_reserve_bytes", "scratch_budget_bytes"}:
        raise PassBError("resource keys mismatch")
    if any(not isinstance(resources[k], int) or resources[k] <= 0 for k in resources):
        raise PassBError("resource values must be positive integers")
    if resources["disk_reserve_bytes"] < DEFAULT_DISK_RESERVE_BYTES:
        raise PassBError("disk reserve cannot be below 150 decimal GB")
    if not isinstance(execution["threads"], int) or not 1 <= execution["threads"] <= 16:
        raise PassBError("threads must be in [1,16]")
    for obj, names in ((parameter, ("sha256",)), (adapter, ("registry_sha256",
                        "adapter_source_sha256")), (execution, ("worker_sha256",
                        "quantizer_sha256", "attest_sha256", "decoder_sha256",
                        "ppl_bench_sha256", "multi_eval_sha256", "python_sha256"))):
        for name in names:
            if not _is_sha(obj[name]):
                raise PassBError(f"invalid bound hash: {name}")

    # Resolve output only after the complete typed shape has passed.
    paths = _request_paths(request)
    if paths["run"].exists() and paths["run"].is_symlink():
        raise PassBError("output_root may not be a symlink")
    request_sha = _hash_file(path)[0]
    return request, request_sha, paths


def _validate_bound_inputs(request: dict[str, Any]) -> dict[str, Any]:
    source, parameter, adapter, execution = (
        request["source"], request["parameter_manifest"], request["adapter"],
        request["execution"],
    )
    bindings = [
        (source["weight_file"], source["weight_sha256"], source["weight_bytes"]),
        (source["census_report"], source["census_report_sha256"], None),
        (str(PASS_A_INDEX), source["pass_a_index_sha256"], None),
        (parameter["path"], parameter["sha256"], None),
        (adapter["registry_path"], adapter["registry_sha256"], None),
        (execution["worker_path"], execution["worker_sha256"], None),
        (execution["quantizer_path"], execution["quantizer_sha256"], None),
        (execution["attest_path"], execution["attest_sha256"], None),
        (execution["decoder_path"], execution["decoder_sha256"], None),
        (execution["ppl_bench_path"], execution["ppl_bench_sha256"], None),
        (execution["multi_eval_path"], execution["multi_eval_sha256"], None),
    ]
    observed: list[dict[str, Any]] = []
    for raw, expected_sha, expected_bytes in bindings:
        path = _inside_workspace(raw)
        digest, size = _hash_file(path)
        if digest != expected_sha or (expected_bytes is not None and size != expected_bytes):
            raise PassBError(f"input identity mismatch: {path}")
        observed.append({"path": str(path.relative_to(ROOT)), "sha256": digest, "bytes": size})
    python = _system_python(execution["python_path"])
    digest, size = _hash_file(python)
    if digest != execution["python_sha256"]:
        raise PassBError("Python interpreter identity mismatch")
    observed.append({"path": str(python), "sha256": digest, "bytes": size})

    index = _load_json(PASS_A_INDEX)
    if index.get("pass_a_complete") is not True or index.get("report_count") != 7:
        raise PassBError("completed seven-rung Pass-A index is required")
    census = _load_json(_inside_workspace(source["census_report"]))
    if census.get("status") != "complete" or census.get("label") != request["label"]:
        raise PassBError("census is not the completed 0.5B report")
    if census.get("source", {}).get("source_manifest_sha256") != source["source_manifest_sha256"]:
        raise PassBError("source-manifest hash is not census-bound")
    parameter_manifest = _load_json(_inside_workspace(parameter["path"]))
    return {"files": observed, "index": index, "census": census,
            "parameter_manifest": parameter_manifest}


def _count_source(weight: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise PassBError("safetensors is required") from exc
    quantized_elements = passthrough_elements = 0
    quantized_names: list[str] = []
    passthrough_names: list[str] = []
    with safe_open(str(weight), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            shape = tuple(int(v) for v in handle.get_slice(name).get_shape())
            elements = math.prod(shape)
            if len(shape) == 2 and name.endswith(QUANTIZABLE_SUFFIXES):
                quantized_names.append(name)
                quantized_elements += elements
            else:
                passthrough_names.append(name)
                passthrough_elements += elements
    return {
        "tensor_count": len(quantized_names) + len(passthrough_names),
        "stored_parameter_count": quantized_elements + passthrough_elements,
        "quantized_tensor_count": len(quantized_names),
        "quantized_parameter_count": quantized_elements,
        "passthrough_tensor_count": len(passthrough_names),
        "passthrough_parameter_count": passthrough_elements,
        "quantized_names_sha256": _sha_value(quantized_names),
        "passthrough_names_sha256": _sha_value(passthrough_names),
    }


def _manifest_count(manifest: dict[str, Any]) -> int | None:
    """Read the conservative exact count without accepting a label-derived estimate."""
    candidates = (
        "exact_distinct_stored_parameter_count", "exact_parameter_count",
        "stored_tensor_element_count", "stored_parameter_count",
    )
    stack: list[Any] = [manifest]
    found: list[int] = []
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if key in candidates and isinstance(child, int) and not isinstance(child, bool):
                    found.append(child)
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    unique = sorted(set(found))
    return unique[0] if len(unique) == 1 and unique[0] > 0 else None


def _initial_checkpoint(request_sha: str) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA, "request_sha256": request_sha,
        "created_at": _now(), "updated_at": _now(), "status": "running",
        "completed_phases": [], "phases": {}, "stop_requested": False,
    }


def _load_checkpoint(path: Path, request_sha: str) -> dict[str, Any]:
    if not path.exists():
        return _initial_checkpoint(request_sha)
    cp = _load_json(path)
    if cp.get("schema") != CHECKPOINT_SCHEMA or cp.get("request_sha256") != request_sha:
        raise PassBError("checkpoint identity does not match request")
    completed = cp.get("completed_phases")
    if not isinstance(completed, list) or any(p not in PHASES for p in completed):
        raise PassBError("invalid checkpoint phase set")
    if completed != list(PHASES[:len(completed)]):
        raise PassBError("checkpoint phases are not a strict prefix")
    return cp


def _save_checkpoint(path: Path, cp: dict[str, Any]) -> None:
    cp["updated_at"] = _now()
    cp["stop_requested"] = bool(_STOP_REQUESTED)
    _atomic_json(path, cp)


def _finish_phase(paths: dict[str, Path], cp: dict[str, Any], phase: str,
                  evidence: dict[str, Any]) -> None:
    expected = PHASES[len(cp["completed_phases"])]
    if phase != expected:
        raise PassBError(f"phase order violation: expected {expected}, got {phase}")
    cp["phases"][phase] = {"completed_at": _now(), **evidence}
    cp["completed_phases"].append(phase)
    _save_checkpoint(paths["checkpoint"], cp)
    if _STOP_REQUESTED and phase != "receipt":
        cp["status"] = "checkpointed-stop"
        _save_checkpoint(paths["checkpoint"], cp)
        raise PassBError("stop requested; exited at a durable phase boundary")


def _phase_done(cp: dict[str, Any], name: str) -> bool:
    return name in cp["completed_phases"]


def _fixed_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "DOCTOR_DEVICE": "cpu",
        "DOCTOR_DTYPE": "bfloat16",
        "STRAND_NO_GPU": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": "0",
    })
    return env


def _extract_passthrough(source: Path, destination: Path) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import save_file
    tensors: dict[str, Any] = {}
    elements = 0
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            shape = tuple(int(v) for v in handle.get_slice(name).get_shape())
            if not (len(shape) == 2 and name.endswith(QUANTIZABLE_SUFFIXES)):
                tensor = handle.get_tensor(name)
                tensors[name] = tensor
                elements += tensor.numel()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    try:
        save_file(tensors, str(tmp), metadata={
            "hawking_artifact_class": "lossless_passthrough_state",
            "hawking_schema": "hawking.doctor_v5_pass_b_passthrough.v1",
        })
        _commit_generated(tmp, destination)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return {"tensor_count": len(tensors), "parameter_count": elements,
            "artifact": _artifact(destination)}


def _copy_metadata(model_dir: Path, bundle_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for name in METADATA_NAMES:
        source = model_dir / name
        if not source.is_file() or source.is_symlink():
            continue
        target = bundle_dir / name
        payload = source.read_bytes()
        _atomic_bytes(target, payload)
        copied.append(_artifact(target))
    if not (bundle_dir / "config.json").is_file():
        raise PassBError("bundle metadata is missing config.json")
    return copied


def _validate_reconstruction(source: Path, recon: Path) -> dict[str, Any]:
    from safetensors import safe_open
    with safe_open(str(source), framework="pt", device="cpu") as a, \
            safe_open(str(recon), framework="pt", device="cpu") as b:
        ak = [name for name in a.keys()
              if len(tuple(a.get_slice(name).get_shape())) == 2
              and name.endswith(QUANTIZABLE_SUFFIXES)]
        bk = list(b.keys())
        if ak != bk:
            raise PassBError("decoded STR2 tensor inventory differs from quantized source subset")
        elements = 0
        for name in ak:
            ashape = tuple(a.get_slice(name).get_shape())
            bshape = tuple(b.get_slice(name).get_shape())
            if ashape != bshape:
                raise PassBError(f"reconstruction shape mismatch: {name}")
            elements += math.prod(ashape)
    return {"tensor_count": len(ak), "parameter_count": elements,
            "artifact_class": "decoded_from_attested_str2_projection_override",
            "artifact": _artifact(recon)}


def _run_eval(script: Path, model_dir: Path, override: Path | None, label: str,
              result_path: Path, log_path: Path, python: Path) -> dict[str, Any]:
    argv = [str(python), str(script), str(model_dir),
            str(override) if override is not None else "-", label]
    _run_logged(argv, log_path, env=_fixed_env())
    row = _last_json_line(log_path)
    _atomic_json(result_path, row)
    return {"result": row, "artifact": _artifact(result_path),
            "log": _artifact(log_path)}


def _bundle_manifest(request: dict[str, Any], paths: dict[str, Path],
                     counts: dict[str, Any]) -> dict[str, Any]:
    payload_files = [_artifact(paths["packed"]), _artifact(paths["passthrough"])]
    metadata = [_artifact(p) for p in sorted(paths["bundle"].iterdir())
                if p.is_file() and p.name not in {
                    paths["packed"].name, paths["passthrough"].name,
                    paths["bundle_manifest"].name}]
    payload_bytes = sum(x["bytes"] for x in payload_files)
    bundle_bytes = payload_bytes + sum(x["bytes"] for x in metadata)
    stored = counts["stored_parameter_count"]
    quantized = counts["quantized_parameter_count"]
    manifest = {
        "schema": "hawking.doctor_v5_pass_b_bundle.v1",
        "created_at": _now(),
        "artifact_class": "physically_complete_weight_payload_bundle",
        "runtime_status": "assembly_required_and_runtime_not_validated",
        "request_id": request["request_id"],
        "source_sha256": request["source"]["weight_sha256"],
        "profile": request["profile"],
        "bits": request["bits"],
        "parameter_accounting": counts,
        "files": {"model_payload": payload_files, "metadata": metadata},
        "physical_accounting": {
            "packed_projection_bpw": payload_files[0]["bytes"] * 8 / quantized,
            "model_payload_bytes": payload_bytes,
            "model_payload_bpw_over_all_stored_parameters": payload_bytes * 8 / stored,
            "full_bundle_bytes": bundle_bytes,
            "full_bundle_bpw_over_all_stored_parameters": bundle_bytes * 8 / stored,
            "scope": "exact physical bytes; dense reconstruction oracle excluded",
        },
        "limitations": [
            "projection STR2 and lossless pass-through state require Hawking runtime assembly",
            "quality observations are provisional and unsealed",
            "no speed, energy, dominance, or source-deletion claim is permitted",
        ],
    }
    _atomic_json(paths["bundle_manifest"], manifest)
    return manifest


def _quality_summary(paths: dict[str, Path]) -> dict[str, Any]:
    baseline_ppl, recon_ppl = _load_json(paths["baseline_ppl"]), _load_json(paths["recon_ppl"])
    baseline_cap, recon_cap = _load_json(paths["baseline_cap"]), _load_json(paths["recon_cap"])
    b_ppl, r_ppl = float(baseline_ppl["ppl"]), float(recon_ppl["ppl"])
    b_cap, r_cap = float(baseline_cap["aggregate"]), float(recon_cap["aggregate"])
    return {
        "evidence_class": "provisional_unsealed_forward_pass_observation",
        "ppl": {"baseline": b_ppl, "reconstruction": r_ppl,
                "relative_delta": r_ppl / b_ppl - 1.0},
        "capability": {"baseline": b_cap, "reconstruction": r_cap,
                       "absolute_delta": r_cap - b_cap},
        "dominance_proven": False,
        "quality_restoration_proven": False,
    }


def _execute(request_path: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    _install_signal_handlers()
    request_path = _inside_workspace(str(request_path))
    request, request_sha, paths = _validate_request(request_path)
    paths["run"].mkdir(parents=True, exist_ok=True)
    for name in ("logs", "bundle"):
        paths[name].mkdir(parents=True, exist_ok=True)
    (paths["run"] / "evaluation").mkdir(parents=True, exist_ok=True)
    cp = _load_checkpoint(paths["checkpoint"], request_sha)

    if not _phase_done(cp, "preflight"):
        lease = _validate_heavy_lease()
        bound = _validate_bound_inputs(request)
        sample = _resource_sample(paths["run"])
        _resource_gate(sample, request)
        counts = _count_source(_inside_workspace(request["source"]["weight_file"]))
        census_count = bound["census"].get("tensor_census", {}).get(
            "stored_tensor_element_count")
        manifest_count = _manifest_count(bound["parameter_manifest"])
        if counts["stored_parameter_count"] != census_count:
            raise PassBError("live parameter count differs from Pass-A census")
        if manifest_count != counts["stored_parameter_count"]:
            raise PassBError("role-separated parameter manifest lacks the exact live count")
        registry = _load_json(_inside_workspace(request["adapter"]["registry_path"]))
        # The registry schema is validated by doctor_v5_adapter_abi when available;
        # this local check still makes the exact adapter/source pair fail closed.
        registry_blob = _canonical_bytes(registry)
        if request["adapter"]["adapter_id"].encode() not in registry_blob \
                or request["adapter"]["adapter_source_sha256"].encode() not in registry_blob:
            raise PassBError("requested adapter pair is absent from the bound registry")
        _finish_phase(paths, cp, "preflight", {
            "heavy_lease": lease, "resources": sample, "parameter_accounting": counts,
            "bound_input_count": len(bound["files"]),
        })
    if preflight_only:
        return {"status": "preflight-complete", "checkpoint": str(paths["checkpoint"])}

    source = _inside_workspace(request["source"]["weight_file"])
    model_dir = _inside_workspace(request["source"]["model_dir"])
    execution = request["execution"]
    python = _system_python(execution["python_path"])
    quantizer = _inside_workspace(execution["quantizer_path"])
    attest = _inside_workspace(execution["attest_path"])
    decoder = _inside_workspace(execution["decoder_path"])
    ppl = _inside_workspace(execution["ppl_bench_path"])
    multi = _inside_workspace(execution["multi_eval_path"])
    counts = cp["phases"]["preflight"]["parameter_accounting"]

    if not _phase_done(cp, "baseline_ppl"):
        evidence = _run_eval(ppl, model_dir, None, "pass-b-baseline",
                             paths["baseline_ppl"], paths["logs"] / "baseline_ppl.log", python)
        _finish_phase(paths, cp, "baseline_ppl", evidence)

    if not _phase_done(cp, "pass_through"):
        evidence = _extract_passthrough(source, paths["passthrough"])
        evidence["metadata"] = _copy_metadata(model_dir, paths["bundle"])
        if evidence["parameter_count"] != counts["passthrough_parameter_count"]:
            raise PassBError("pass-through parameter count mismatch")
        _finish_phase(paths, cp, "pass_through", evidence)

    fixed_quant_args = [
        str(quantizer), "--in", str(source), "--bits", "4", "--threads",
        str(execution["threads"]), "--quality", "--rht-cols",
        "--outlier-channel", "1", "--outlier-bits", "8",
        "--sdsq-sideinfo", "--c2f-outl", "--ragged-v2",
    ]
    if not _phase_done(cp, "packed_encode"):
        tmp = paths["packed"].with_name(f".{paths['packed'].name}.partial.{os.getpid()}")
        try:
            _run_logged(fixed_quant_args + ["--packed-v2-out", str(tmp)],
                        paths["logs"] / "packed_encode.log", env=_fixed_env())
            embedded_source_sha = _str2_source_sha256(tmp)
            if embedded_source_sha != request["source"]["weight_sha256"]:
                raise PassBError("STR2 embedded source digest differs from the pinned source")
            _commit_generated(tmp, paths["packed"])
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        manifest = _bundle_manifest(request, paths, counts)
        _finish_phase(paths, cp, "packed_encode", {
            "artifact": _artifact(paths["packed"]),
            "bundle_manifest": _artifact(paths["bundle_manifest"]),
            "physical_accounting": manifest["physical_accounting"],
        })

    if not _phase_done(cp, "archive_attest"):
        log = paths["logs"] / "archive_attest.log"
        _run_logged([str(attest), str(paths["packed"]), "--roots"], log,
                    env=_fixed_env())
        attestation_text = log.read_text(encoding="utf-8", errors="replace")
        if "self-verify" not in attestation_text or "model_root" not in attestation_text:
            raise PassBError("attestor returned success without root/self-verification evidence")
        attestation = {
            "schema": "hawking.doctor_v5_pass_b_str2_attestation.v1",
            "attested_at": _now(),
            "archive": _artifact(paths["packed"]),
            "attestor": _artifact(attest),
            "log": _artifact(log),
            "status": "passed",
        }
        _atomic_json(paths["attestation"], attestation)
        _finish_phase(paths, cp, "archive_attest", {
            "artifact": _artifact(paths["attestation"]), "log": _artifact(log)})

    if not _phase_done(cp, "archive_decode"):
        tmp = paths["recon"].with_name(f".{paths['recon'].name}.partial.{os.getpid()}")
        try:
            _run_logged([str(decoder), str(paths["packed"]), str(tmp)],
                        paths["logs"] / "archive_decode.log", env=_fixed_env())
            validation = _validate_reconstruction(source, tmp)
            _commit_generated(tmp, paths["recon"])
            validation = _validate_reconstruction(source, paths["recon"])
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        validation["decoder"] = _artifact(decoder)
        validation["attestation"] = _artifact(paths["attestation"])
        _finish_phase(paths, cp, "archive_decode", validation)

    if not _phase_done(cp, "reconstruction_ppl"):
        evidence = _run_eval(ppl, model_dir, paths["recon"], "pass-b-4bit-reconstruction",
                             paths["recon_ppl"], paths["logs"] / "reconstruction_ppl.log",
                             python)
        _finish_phase(paths, cp, "reconstruction_ppl", evidence)

    if not _phase_done(cp, "baseline_capability"):
        evidence = _run_eval(multi, model_dir, None, "pass-b-baseline",
                             paths["baseline_cap"], paths["logs"] / "baseline_capability.log",
                             python)
        _finish_phase(paths, cp, "baseline_capability", evidence)

    if not _phase_done(cp, "reconstruction_capability"):
        evidence = _run_eval(multi, model_dir, paths["recon"],
                             "pass-b-4bit-reconstruction", paths["recon_cap"],
                             paths["logs"] / "reconstruction_capability.log", python)
        _finish_phase(paths, cp, "reconstruction_capability", evidence)

    if not _phase_done(cp, "receipt"):
        before = cp["phases"]["preflight"]["resources"]
        after = _resource_sample(paths["run"])
        bundle = _load_json(paths["bundle_manifest"])
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "completed_at": _now(),
            "status": "complete",
            "request": _artifact(request_path),
            "request_id": request["request_id"],
            "label": request["label"],
            "profile": request["profile"],
            "source": request["source"],
            "adapter": request["adapter"],
            "parameter_accounting": counts,
            "bundle": {"manifest": _artifact(paths["bundle_manifest"]),
                       "physical_accounting": bundle["physical_accounting"]},
            "evaluation_oracle": {
                "artifact": _artifact(paths["recon"]),
                "deployable": False,
                "excluded_from_bundle_bpw": True,
            },
            "quality_observation": _quality_summary(paths),
            "resources": {"before": before, "after": after},
            "resume": {"schema": CHECKPOINT_SCHEMA, "phase_order": list(PHASES),
                       "atomic_replace": True, "fsync_file": True,
                       "fsync_parent_directory": True},
            "claims": {
                "dominance_proven": False,
                "sealed_quality_complete": False,
                "independent_reproduction_complete": False,
                "source_deletion_permitted": False,
            },
        }
        receipt["receipt_sha256"] = _sha_value(receipt)
        _atomic_json(paths["receipt"], receipt)
        _finish_phase(paths, cp, "receipt", {"artifact": _artifact(paths["receipt"]),
                                             "receipt_sha256": receipt["receipt_sha256"]})
    cp["status"] = "complete"
    _save_checkpoint(paths["checkpoint"], cp)
    return _load_json(paths["receipt"])


def _selftest() -> None:
    assert _swap_used_bytes("vm.swapusage: total = 0.00M  used = 0.00M  free = 0.00M") == 0
    assert _swap_used_bytes("used = 1.50G") == int(1.5 * 1024 ** 3)
    assert _sha_value({"b": 2, "a": 1}) == _sha_value({"a": 1, "b": 2})
    assert _manifest_count({"authority": {"exact_parameter_count": 11}}) == 11
    assert _manifest_count({"exact_parameter_count": 11,
                            "stored_parameter_count": 12}) is None
    assert len(PHASES) == len(set(PHASES)) and PHASES[-1] == "receipt"
    print(json.dumps({"status": "ok", "schema": REQUEST_SCHEMA, "profile": PROFILE}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "preflight"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--request", required=True, type=Path)
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    try:
        if args.command == "selftest":
            _selftest()
            return 0
        result = _execute(args.request, preflight_only=args.command == "preflight")
        print(json.dumps(result, sort_keys=True))
        return 0
    except PassBError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
