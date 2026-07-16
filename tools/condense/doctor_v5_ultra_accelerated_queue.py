#!/usr/bin/env python3.12
"""Opt-in accelerated Doctor V5 supervisor entrypoint.

This wrapper leaves the source-bound live queue untouched.  At a quiescent
checkpoint it loads that exact implementation, verifies a two-key stacked-
admission overlay, and replaces only pure admission hooks.  Cell identities,
runtime commands, output validation, lifecycle GC, receipts, and reporter logic
remain the frozen base queue implementation.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shlex
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any

import doctor_v5_accel_loader as accel_loader
import doctor_v5_accelerated_resource_policy as resource_policy
import doctor_v5_stacked_admission as stacked


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "doctor_v5_ultra_queue.py"
BASE_SHA256 = "2c4bd2a6b04cfd5c13bac0f076c9676a6b9db6689ee7de072391fbb5ff7b0b6f"
ACCEL_LOADER_SHA256 = "81adc24e3f6a50cdbd31aa368ed60ab42cf0942b1477075455735984587038f3"
RESOURCE_POLICY_SHA256 = "8ddd44ee84e8c995bdfe9fbcae74d5207f2beb3b1833599e6f1690ec3e0eee47"
STACKED_ADMISSION_SHA256 = "2227c4ebb32039b87d3d9d40f24e8d984a8f96cef4113ed29a48f125b809aaa5"
MARKER = _BASE_MARKER = (
    HERE.parents[1] / "reports/condense/doctor_v5_ultra/staged_acceleration/active_stack.json"
)
PACKET = HERE.parents[1] / (
    "reports/condense/doctor_v5_ultra/staged_acceleration/pending_runtime_packet.json"
)
ACCEL_AUTORESUME = HERE / "doctor_v5_ultra_accelerated_autoresume.py"
MARKER_SCHEMA = "hawking.doctor_v5_acceleration_active_marker.v1"
PACKET_SCHEMA = "hawking.doctor_v5_acceleration_pending_runtime.v1"
BLOCK_LAUNCH_TOKENS = 20.0
COOPERATIVE_MOP_PROGRAM_SCHEMA = "mop-generation1-program/v1"
COOPERATIVE_MOP_CONFIG_SCHEMA = "mop-generation1-context-routing-config/v1"
COOPERATIVE_MOP_CAPSULE_SCHEMA = "mop-generation1-capsule/v1"
COOPERATIVE_MOP_IDLE_WORKERS = 25
COOPERATIVE_MOP_HAWKING_WORKERS = 6
COOPERATIVE_MOP_SHARD_COUNT = 4
COOPERATIVE_MOP_MAX_JSON_BYTES = 8 * 1024 * 1024

_BASE = accel_loader.load_frozen("doctor_v5_ultra_queue_frozen", BASE_PATH, BASE_SHA256)
_ORIGINAL_RESERVATION = _BASE._cell_reservation
_ORIGINAL_SCAN_HEADS = _BASE._scan_runnable_heads
_ORIGINAL_ENFORCE_POOL_BUDGET = _BASE._enforce_pool_budget
_CPU_SAMPLES: deque[float] = deque(maxlen=3)
_SCAN_RESERVATIONS: dict[str, int] = {}
_RESOURCE_STOP_SAMPLES: dict[str, int] = {}
_OVERLAY: dict[str, Any] | None = None
_RECORDED_COOPERATIVE_HANDOFFS: set[str] = set()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: row for name, row in value.items() if name != key}


def _verify_static_bindings() -> None:
    bindings = (
        (Path(accel_loader.__file__), ACCEL_LOADER_SHA256, "acceleration loader"),
        (Path(resource_policy.__file__), RESOURCE_POLICY_SHA256,
         "accelerated resource policy"),
        (Path(stacked.__file__), STACKED_ADMISSION_SHA256, "stacked admission module"),
    )
    for path, expected, label in bindings:
        observed, _ = accel_loader.hash_file(path)
        if observed != expected:
            raise accel_loader.AccelerationBindingError(
                f"{label} source drifted: expected={expected} observed={observed}"
            )


def _read_bound_json(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    path.relative_to(stacked.ROOT.resolve())
    if path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
        raise stacked.OverlayError(f"active acceleration JSON is unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise stacked.OverlayError(f"cannot read active acceleration JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise stacked.OverlayError(f"active acceleration JSON root is invalid: {path}")
    return value


def _artifact_matches(row: Any, *, expected_path: Path | None = None,
                      cache: dict[str, tuple[str, int]] | None = None) -> bool:
    if not isinstance(row, dict) or not {"path", "sha256", "bytes"} <= set(row):
        return False
    try:
        path = Path(row["path"]).resolve(strict=True)
        if expected_path is not None and path != expected_path.resolve(strict=True):
            return False
        key = str(path)
        observed = cache.get(key) if cache is not None else None
        if observed is None:
            observed = accel_loader.hash_file(path)
            if cache is not None:
                cache[key] = observed
    except (OSError, TypeError, ValueError, accel_loader.AccelerationBindingError):
        return False
    return observed == (row["sha256"], row["bytes"])


def _active_generation_errors(overlay: dict[str, Any], *,
                              marker_path: Path = MARKER,
                              packet_path: Path = PACKET) -> list[str]:
    """Validate immutable promotion bindings without pinning mutable queue state."""
    errors: list[str] = []
    try:
        marker = _read_bound_json(marker_path)
        packet = _read_bound_json(packet_path)
    except (OSError, stacked.OverlayError) as exc:
        return [str(exc)]
    marker_keys = {
        "schema", "activated_at", "overlay_path", "overlay_sha256",
        "pending_runtime_generation_sha256", "accelerated_queue",
        "accelerated_autoresume", "marker_sha256",
    }
    if set(marker) != marker_keys or marker.get("schema") != MARKER_SCHEMA \
            or marker.get("marker_sha256") != _hash_value(_without(marker, "marker_sha256")):
        errors.append("active acceleration marker identity is invalid")
    try:
        marker_overlay = Path(marker.get("overlay_path", "")).resolve(strict=True)
    except (OSError, TypeError, ValueError):
        marker_overlay = None
    if marker_overlay != stacked.DEFAULT_OVERLAY.resolve() \
            or marker.get("overlay_sha256") != overlay.get("overlay_sha256"):
        errors.append("active marker does not bind the selected admission overlay")
    if not _artifact_matches(marker.get("accelerated_queue"),
                             expected_path=Path(__file__)):
        errors.append("active marker does not bind this accelerated queue source")
    if not _artifact_matches(marker.get("accelerated_autoresume"),
                             expected_path=ACCEL_AUTORESUME):
        errors.append("active marker does not bind the accelerated autoresume source")
    if packet.get("schema") != PACKET_SCHEMA \
            or packet.get("packet_sha256") != _hash_value(_without(packet, "packet_sha256")) \
            or marker.get("pending_runtime_generation_sha256") != packet.get("packet_sha256"):
        errors.append("active pending-runtime generation identity is invalid")

    plan = _BASE._load_plan()
    state = _BASE._load_state(plan)
    cells = {row["cell_id"]: row for row in plan["cells"]}
    cache: dict[str, tuple[str, int]] = {}
    registry = packet.get("registry")
    if not isinstance(registry, dict) or not _artifact_matches(
            registry.get("staged"), expected_path=_BASE.REGISTRY_PATH, cache=cache):
        # The staged digest is the promoted live registry digest; its embedded
        # path still points at staging, so compare the digest against the exact
        # live target through a synthetic artifact row.
        staged = registry.get("staged") if isinstance(registry, dict) else None
        live_row = ({"path": str(_BASE.REGISTRY_PATH), "sha256": staged.get("sha256"),
                     "bytes": staged.get("bytes")} if isinstance(staged, dict) else None)
        if not _artifact_matches(live_row, expected_path=_BASE.REGISTRY_PATH, cache=cache):
            errors.append("promoted adapter registry differs from the active generation")

    promoted = packet.get("pending_runtime_specs")
    if not isinstance(promoted, list) or not promoted:
        errors.append("active generation has no promoted runtime specs")
        promoted = []
    seen: set[str] = set()
    for index, row in enumerate(promoted):
        cell_id = row.get("cell_id") if isinstance(row, dict) else None
        cell = cells.get(cell_id)
        if cell is None or cell_id in seen:
            errors.append(f"active runtime row[{index}] cell identity is invalid")
            continue
        seen.add(cell_id)
        target = (_BASE.ROOT / cell["runtime_spec_path"]).resolve()
        staged = row.get("staged")
        live_row = ({"path": str(target), "sha256": staged.get("sha256"),
                     "bytes": staged.get("bytes")} if isinstance(staged, dict) else None)
        if not _artifact_matches(live_row, expected_path=target, cache=cache):
            errors.append(f"promoted runtime spec changed: {cell_id}")
            continue
        try:
            document = _read_bound_json(target)
        except stacked.OverlayError as exc:
            errors.append(str(exc)); continue
        runtime, _, runtime_errors = _BASE._validate_runtime_spec(
            cell, document, target, verify_inputs=False
        )
        if runtime is None or runtime_errors:
            errors.append(f"promoted runtime spec is semantically invalid: {cell_id}")
            continue
        for input_row in document.get("inputs", []):
            role = input_row.get("role") if isinstance(input_row, dict) else None
            # Source shards are protected by the separately validated structural
            # seal. Every other runtime/tool/metadata input is small enough to
            # re-hash once per resume (the cache deduplicates shared paths).
            if isinstance(role, str) and role.startswith("source_shard:"):
                continue
            if not _artifact_matches(input_row, cache=cache):
                errors.append(f"accelerator source/tool binding changed: {cell_id}:{role}")

    seal = packet.get("terminal_seal")
    rows = seal.get("rows") if isinstance(seal, dict) else None
    if not isinstance(rows, list) or seal.get("rows_sha256") != _hash_value(rows):
        errors.append("active terminal evidence seal is invalid")
    else:
        for row in rows:
            cell_id = row.get("cell_id") if isinstance(row, dict) else None
            if cell_id not in cells or state["cells"][cell_id]["status"] != row.get("status") \
                    or cells[cell_id]["cell_identity_sha256"] \
                    != row.get("cell_identity_sha256"):
                errors.append(f"sealed terminal cell changed: {cell_id}")
                continue
            if any(not _artifact_matches(artifact, cache=cache)
                   for artifact in row.get("artifacts", [])):
                errors.append(f"sealed terminal evidence changed: {cell_id}")
    return errors


def _evidence() -> dict[str, dict[str, int]]:
    return stacked._observed_tier_rss(_BASE._load_plan())


def _reservation_from_evidence(
        cell: dict[str, Any], evidence: dict[str, dict[str, int]]) -> int:
    """Charge dynamic evidence headroom through the base queue's fixed sum.

    The base loop always adds ``SAFETY_MARGIN_BYTES``.  We set that floor to
    12GB and encode the 20/28GB warm-up delta into the candidate reservation.
    An unseen >16B tier is charged the entire 66GB admission ceiling, making its
    first lane exclusive until a persisted RSS sample exists.
    """
    base = _ORIGINAL_RESERVATION(cell)
    nominal = float(cell.get("nominal_params_b", 0.0))
    if nominal <= stacked.UNKNOWN_LARGE_TIER_THRESHOLD_B:
        return base
    samples = evidence.get(cell["model_label"], {}).get("samples", 0)
    ceiling = _BASE.PROCESS_BUDGET_BYTES - stacked.MIN_DYNAMIC_MARGIN_BYTES
    if samples <= 0:
        return ceiling
    penalty = 8_000_000_000 if samples < 3 else 0
    return min(ceiling, base + penalty)


def _accelerated_reservation(cell: dict[str, Any]) -> int:
    """Return the current scan's sealed reservation without reparsing evidence.

    The base admission loop asks for the reservation again after this wrapper's
    scan hook returns.  Retaining only the per-cell integer lets that call reuse
    the exact decision while the next scan atomically replaces the cache.  A
    direct call for a cell outside the last scan remains fail-closed and loads a
    fresh evidence snapshot.
    """
    cell_id = cell.get("cell_id")
    if isinstance(cell_id, str) and cell_id in _SCAN_RESERVATIONS:
        return _SCAN_RESERVATIONS[cell_id]
    return _reservation_from_evidence(cell, _evidence())


def _global_host_cpu_cores() -> float:
    """Sum CPU across the host so lock-free heavy owners remain visible."""
    rows = stacked._process_rows()
    if not isinstance(rows, list) or not rows:
        raise stacked.OverlayError("global CPU process sample is empty")
    total = 0.0
    for row in rows:
        value = row.get("cpu_percent") if isinstance(row, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise stacked.OverlayError("global CPU process sample is invalid")
        total += float(value) / 100.0
    return round(total, 3)


def _stable_external_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a same-user cooperative scheduler contract without trusting paths.

    The MOP campaign is external to this repository, so it is never granted
    queue or signal authority.  This reader only authenticates its published
    adaptive-worker contract before permitting one launch-only handoff.
    """
    resolved = path.resolve(strict=True)
    before_path = resolved.lstat()
    if resolved.is_symlink() or not stat.S_ISREG(before_path.st_mode) \
            or before_path.st_uid != os.getuid() \
            or not 0 < before_path.st_size <= COOPERATIVE_MOP_MAX_JSON_BYTES:
        raise stacked.OverlayError("cooperative MOP JSON is not a safe same-user file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns
        )
        if identity(opened) != identity(before_path):
            raise stacked.OverlayError("cooperative MOP JSON changed while opening")
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > COOPERATIVE_MOP_MAX_JSON_BYTES:
                raise stacked.OverlayError("cooperative MOP JSON exceeds its ceiling")
        after_fd, after_path = os.fstat(descriptor), resolved.lstat()
        if identity(opened) != identity(after_fd) \
                or identity(opened) != identity(after_path) \
                or len(payload) != opened.st_size:
            raise stacked.OverlayError("cooperative MOP JSON changed while reading")
    finally:
        os.close(descriptor)
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise stacked.OverlayError(f"cooperative MOP JSON is invalid: {exc}") from exc
    if not isinstance(document, dict):
        raise stacked.OverlayError("cooperative MOP JSON root is not an object")
    return document, {
        "path": str(resolved), "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _command_rows() -> list[dict[str, Any]]:
    try:
        process = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,command="],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        raise stacked.OverlayError(f"cooperative process probe failed: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line in process.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        try:
            rows.append({"pid": int(fields[0]), "ppid": int(fields[1]),
                         "command": fields[2]})
        except ValueError:
            continue
    if not rows:
        raise stacked.OverlayError("cooperative process probe is empty")
    return rows


def _argument(tokens: list[str], name: str) -> str:
    indexes = [index for index, value in enumerate(tokens) if value == name]
    if len(indexes) != 1 or indexes[0] + 1 >= len(tokens):
        raise stacked.OverlayError(f"cooperative command has no unique {name}")
    return tokens[indexes[0] + 1]


def _is_descendant(pid: int, ancestor: int, parents: dict[int, int]) -> bool:
    seen: set[int] = set()
    cursor = pid
    while cursor > 1 and cursor not in seen:
        if cursor == ancestor:
            return True
        seen.add(cursor)
        cursor = parents.get(cursor, 0)
    return False


def _cooperative_mop_shard_contracts(
        program: dict[str, Any], repo: Path, config_path: Path,
) -> dict[str, int]:
    """Extract the exact sealed adaptive shard contracts used by labeled MOP."""
    capsules = program.get("capsules")
    if not isinstance(capsules, list):
        raise stacked.OverlayError("cooperative MOP program has no capsule list")
    prefix = "g1_c2_context_routing_shard_"
    shard_capsules = [
        row for row in capsules
        if isinstance(row, dict) and isinstance(row.get("id"), str)
        and row["id"].startswith(prefix)
    ]
    if len(shard_capsules) != COOPERATIVE_MOP_SHARD_COUNT:
        raise stacked.OverlayError("cooperative MOP shard matrix is not exact")
    expected_script = (
        repo / "scripts/generation1_context_routing/run_shard.py"
    ).resolve(strict=True)
    expected_config = config_path.resolve(strict=True)
    contracts: dict[str, int] = {}
    for capsule in shard_capsules:
        capsule_id = capsule["id"]
        suffix = capsule_id.removeprefix(prefix)
        if len(suffix) != 2 or not suffix.isdecimal():
            raise stacked.OverlayError("cooperative MOP shard id is not canonical")
        shard_index = int(suffix)
        if shard_index >= COOPERATIVE_MOP_SHARD_COUNT \
                or capsule_id != f"{prefix}{shard_index:02d}" \
                or capsule.get("schema") != COOPERATIVE_MOP_CAPSULE_SCHEMA \
                or capsule.get("capsule_sha256") != _hash_value(
                    _without(capsule, "capsule_sha256")
                ):
            raise stacked.OverlayError("cooperative MOP shard seal is invalid")
        command = capsule.get("command")
        resources = capsule.get("resources")
        if not isinstance(command, list) \
                or not all(isinstance(value, str) for value in command) \
                or not isinstance(resources, dict):
            raise stacked.OverlayError("cooperative MOP shard contract is invalid")
        script_indexes = [
            index for index, value in enumerate(command)
            if Path(value).name == "run_shard.py"
        ]
        if len(script_indexes) != 1:
            raise stacked.OverlayError("cooperative MOP shard script is ambiguous")
        script_path = Path(command[script_indexes[0]])
        if not script_path.is_absolute():
            script_path = repo / script_path
        bound_config = Path(_argument(command, "--config"))
        if not bound_config.is_absolute():
            bound_config = repo / bound_config
        if script_path.resolve(strict=True) != expected_script \
                or bound_config.resolve(strict=True) != expected_config \
                or _argument(command, "--shard-index") != str(shard_index) \
                or _argument(command, "--idle-workers") \
                != str(COOPERATIVE_MOP_IDLE_WORKERS) \
                or _argument(command, "--hawking-workers") \
                != str(COOPERATIVE_MOP_HAWKING_WORKERS) \
                or capsule.get("kind") != "corpus" \
                or capsule.get("cwd") != "." \
                or type(resources.get("cpu_cores")) is not int \
                or resources.get("cpu_cores") != COOPERATIVE_MOP_IDLE_WORKERS \
                or resources.get("lane") != "cpu" \
                or resources.get("accelerator") != "none" \
                or resources.get("process_marker") != "run_shard.py":
            raise stacked.OverlayError("cooperative MOP shard resources drifted")
        if capsule_id in contracts or shard_index in contracts.values():
            raise stacked.OverlayError("cooperative MOP shard contract is duplicated")
        contracts[capsule_id] = shard_index
    if set(contracts.values()) != set(range(COOPERATIVE_MOP_SHARD_COUNT)):
        raise stacked.OverlayError("cooperative MOP shard indexes are incomplete")
    return contracts


def _cooperative_mop_handoff(plan: dict[str, Any], state: dict[str, Any],
                             *, charged_global_cpu_cores: float) \
        -> dict[str, Any] | None:
    """Authenticate one 25->6 MOP yield before bootstrapping Doctor.

    MOP reads this exact queue state's sealed ``active_cells`` on every work
    submission loop.  With no Doctor child yet, a symmetric "wait for CPU"
    policy deadlocks: MOP stays at 25 workers while Doctor waits for MOP to
    yield.  This narrow exception allows one launch only when the running MOP
    program cryptographically binds the same queue state and promises the
    reviewed 25-idle/6-Hawking transition.  It never signals or edits MOP and
    confers no stop authority.
    """
    if state.get("active_children") or state.get("active_cells") \
            or state.get("plan_sha256") != plan.get("plan_sha256") \
            or not isinstance(charged_global_cpu_cores, (int, float)) \
            or charged_global_cpu_cores < BLOCK_LAUNCH_TOKENS:
        return None
    try:
        rows = _command_rows()
        parents = {row["pid"]: row["ppid"] for row in rows}
        script_rows: list[tuple[dict[str, Any], list[str], int]] = []
        for row in rows:
            try:
                tokens = shlex.split(row["command"])
            except ValueError:
                if "mop_generation1_campaign.py" in row["command"]:
                    return None
                continue
            scripts = [index for index, value in enumerate(tokens)
                       if Path(value).name == "mop_generation1_campaign.py"]
            if not scripts:
                continue
            if len(scripts) != 1:
                return None
            script_rows.append((row, tokens, scripts[0]))
        if len(script_rows) != 1:
            return None
        launcher, tokens, script_index = script_rows[0]
        if script_index == 1 \
                and Path(tokens[0]).name.startswith("python") \
                and tokens[script_index + 1:script_index + 2] == ["run"] \
                and "--execute" in tokens:
            owner_topology = "argv-descendant"
        elif len(tokens) == 8 and script_index == 3 \
                and tokens[0] == "/usr/bin/caffeinate" \
                and tokens[1] == "-ims" \
                and Path(tokens[2]).name.startswith("python") \
                and tokens[4:6] == ["run", "--program"] \
                and tokens[7] == "--execute":
            owner_topology = "setproctitle-caffeinate"
        else:
            return None
        script = Path(tokens[script_index]).resolve(strict=True)
        repo = script.parent.parent
        if owner_topology == "setproctitle-caffeinate" \
                and (not Path(tokens[2]).is_absolute()
                     or Path(tokens[2]).resolve(strict=True)
                     != (repo / ".venv/bin/python").resolve(strict=True)):
            return None
        program_path = Path(_argument(tokens, "--program"))
        if not program_path.is_absolute():
            program_path = repo / program_path
        program, program_artifact = _stable_external_json(program_path)
        if program.get("schema") != COOPERATIVE_MOP_PROGRAM_SCHEMA \
                or program.get("program_sha256") != _hash_value(
                    _without(program, "program_sha256")
                ):
            return None
        authority_rows = [row for row in program.get("authorities", [])
                          if isinstance(row, dict)
                          and row.get("path")
                          == "configs/experiment/generation1_context_routing.json"]
        if len(authority_rows) != 1:
            return None
        config_path = repo / authority_rows[0]["path"]
        config, config_artifact = _stable_external_json(config_path)
        if config_artifact["sha256"] != authority_rows[0].get("sha256") \
                or config.get("schema") != COOPERATIVE_MOP_CONFIG_SCHEMA \
                or config.get("activation_allowed") is not False:
            return None
        resources = config.get("adaptive_resources")
        if not isinstance(resources, dict) or resources != {
            "idle_workers": COOPERATIVE_MOP_IDLE_WORKERS,
            "hawking_workers": COOPERATIVE_MOP_HAWKING_WORKERS,
            "hawking_queue_state": str(_BASE.STATE.resolve()),
            "hawking_plan_sha256": plan["plan_sha256"],
        }:
            return None
        capsule_pid: int | None = None
        observed_worker_count: int | None = None
        active_shard_index: int | None = None
        if owner_topology == "argv-descendant":
            owner = launcher
            shard_rows = []
            for row in rows:
                if not _is_descendant(row["pid"], owner["pid"], parents):
                    continue
                try:
                    shard_tokens = shlex.split(row["command"])
                except ValueError:
                    if "run_shard.py" in row["command"]:
                        return None
                    continue
                if not any(Path(value).name == "run_shard.py"
                           for value in shard_tokens):
                    continue
                if int(_argument(shard_tokens, "--idle-workers")) \
                        != COOPERATIVE_MOP_IDLE_WORKERS \
                        or int(_argument(shard_tokens, "--hawking-workers")) \
                        != COOPERATIVE_MOP_HAWKING_WORKERS:
                    continue
                shard_rows.append(row)
            if len(shard_rows) != 1:
                return None
        else:
            program_id = program.get("program_id")
            if not isinstance(program_id, str) or not program_id:
                return None
            supervisor_label = f"mop-supervisor:{program_id}"
            supervisor_rows = [
                row for row in rows if row["command"] == supervisor_label
            ]
            if len(supervisor_rows) != 1:
                return None
            owner = supervisor_rows[0]
            if owner["pid"] != launcher["ppid"] or owner["ppid"] != 1:
                return None
            contracts = _cooperative_mop_shard_contracts(
                program, repo, config_path
            )
            capsule_prefix = "mop-capsule:g1_c2_context_routing_shard_"
            labeled_capsules = [
                row for row in rows if row["command"].startswith(capsule_prefix)
            ]
            if len(labeled_capsules) != 1:
                return None
            capsule = labeled_capsules[0]
            capsule_id = capsule["command"].removeprefix("mop-capsule:")
            if capsule_id not in contracts or capsule["ppid"] != owner["pid"]:
                return None
            capsule_pid = capsule["pid"]
            active_shard_index = contracts[capsule_id]
            worker_label = f"mop-c2-s{active_shard_index:02d}-worker"
            labeled_workers = [
                row for row in rows if row["command"].startswith("mop-c2-s")
            ]
            if len(labeled_workers) != COOPERATIVE_MOP_IDLE_WORKERS \
                    or any(row["command"] != worker_label
                           or row["ppid"] != capsule_pid
                           for row in labeled_workers):
                return None
            observed_worker_count = len(labeled_workers)
    except (OSError, KeyError, TypeError, ValueError, stacked.OverlayError):
        return None
    decision = {
        "schema": "hawking.doctor_v5_cooperative_mop_cpu_handoff.v1",
        "plan_sha256": plan["plan_sha256"],
        "state_sha256": state.get("state_sha256"),
        "owner_pid": owner["pid"],
        "owner_topology": owner_topology,
        "owner_command_sha256": hashlib.sha256(
            owner["command"].encode("utf-8")
        ).hexdigest(),
        "program": program_artifact, "adaptive_config": config_artifact,
        "idle_workers_before": COOPERATIVE_MOP_IDLE_WORKERS,
        "hawking_workers_after": COOPERATIVE_MOP_HAWKING_WORKERS,
        "charged_global_cpu_cores": round(float(charged_global_cpu_cores), 3),
        "launch_limit": 1, "launch_only": True,
        "external_signal_or_mutation_permitted": False,
        "stop_or_shed_authority": False,
    }
    if owner_topology == "setproctitle-caffeinate":
        decision.update({
            "launcher_pid": launcher["pid"],
            "capsule_pid": capsule_pid,
            "active_shard_index": active_shard_index,
            "observed_idle_worker_count": observed_worker_count,
        })
    decision["decision_sha256"] = _hash_value(decision)
    return decision


def _accelerated_scan_heads(plan: dict[str, Any], state: dict[str, Any]) \
        -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    _SCAN_RESERVATIONS.clear()
    heads, blockers = _ORIGINAL_SCAN_HEADS(plan, state)
    active = state.get("active_children", {})
    if not isinstance(active, dict):
        return [], blockers
    try:
        # Sample the WHOLE host even with an empty Doctor pool. A lock-free MOP
        # or other heavy owner must consume fixed-20 launch headroom exactly as a
        # Doctor child does. This gate only pauses admission; it never sheds a
        # healthy lane.
        _CPU_SAMPLES.append(_global_host_cpu_cores())
        cpu = resource_policy.fixed_thread_cpu_launch_decision(
            list(_CPU_SAMPLES), logical_cores=int(os.cpu_count() or 1),
            guard_cores=stacked.CPU_GUARD_CORES,
            launch_threads=BLOCK_LAUNCH_TOKENS,
        )
    except (resource_policy.ResourcePolicyError, stacked.OverlayError,
            OSError, ValueError):
        # Unknown global utilization fails closed for NEW launches only.
        return [], blockers
    # The base loop can launch every returned head without re-entering this hook.
    # Hand it at most the number of full 20-core block tokens available now, and
    # only candidates that fit the current persisted RAM reservations.  This
    # ramps from an empty pool one lane per scan instead of bursting four 20-way
    # encoders before the first CPU sample exists.
    cooperative = None
    launch_tokens = min(1, int(cpu["launch_tokens"]))
    if launch_tokens <= 0 and heads:
        cooperative = _cooperative_mop_handoff(
            plan, state,
            charged_global_cpu_cores=float(cpu["charged_global_cpu_cores"]),
        )
        if cooperative is not None:
            launch_tokens = 1
    if launch_tokens <= 0:
        return [], blockers
    if not heads:
        return [], blockers
    # CHILD_RESOURCES is model-sized telemetry in long campaigns.  Parse it
    # exactly once for this admission scan, then use the resulting immutable
    # snapshot for sorting, fit filtering, and the base loop's launch charge.
    evidence = _evidence()
    for execution in heads:
        cell = execution["cell"]
        cell_id = cell["cell_id"]
        _SCAN_RESERVATIONS[cell_id] = _reservation_from_evidence(cell, evidence)
    # First-fit decreasing makes the unchanged base loop a best-fill packer: it
    # tries the largest admissible head, then smaller cross-tier heads whenever
    # that candidate does not fit the remaining RAM envelope.
    order = {cell["cell_id"]: index for index, cell in enumerate(plan["cells"])}
    heads.sort(key=lambda execution: (
        -_SCAN_RESERVATIONS[execution["cell"]["cell_id"]],
        order[execution["cell"]["cell_id"]], execution["cell"]["cell_id"],
    ))
    reservations: list[int] = []
    for child in active.values():
        value = child.get("reserved_bytes") if isinstance(child, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return [], blockers
        reservations.append(value)
    ram_free = (_BASE.PROCESS_BUDGET_BYTES - _BASE.SAFETY_MARGIN_BYTES
                - sum(reservations))
    selected = [execution for execution in heads
                if _SCAN_RESERVATIONS[execution["cell"]["cell_id"]] <= ram_free]
    selected = selected[:launch_tokens]
    if selected and cooperative is not None:
        digest = cooperative["decision_sha256"]
        if digest not in _RECORDED_COOPERATIVE_HANDOFFS:
            _BASE._append_event(
                "cooperative-mop-cpu-handoff",
                decision_sha256=digest,
                owner_pid=cooperative["owner_pid"],
                idle_workers_before=cooperative["idle_workers_before"],
                hawking_workers_after=cooperative["hawking_workers_after"],
                charged_global_cpu_cores=cooperative[
                    "charged_global_cpu_cores"
                ],
                selected_cell_id=selected[0]["cell"]["cell_id"],
                launch_only=True, external_signal_or_mutation_permitted=False,
            )
            _RECORDED_COOPERATIVE_HANDOFFS.add(digest)
    return selected, blockers


def _truthful_record_resource_stop(state: dict[str, Any], cell_id: str, *,
                                   sole_live: bool) -> str | None:
    """Apply a stop transition using measured RSS, never lane count as proof."""
    if not isinstance(sole_live, bool):
        raise _BASE.CampaignError("resource stop lane classification is invalid")
    stop = state.get("last_resource_stop")
    reason = stop.get("reason") if isinstance(stop, dict) else None
    measured = _RESOURCE_STOP_SAMPLES.get(cell_id)
    counts = state.get("resource_stop_counts")
    if not isinstance(counts, dict) or measured is None:
        raise _BASE.CampaignError("resource stop measurement context is unavailable")
    previous = counts.get(cell_id, 0)
    try:
        decision = resource_policy.resource_stop_decision(
            reason=reason, measured_cell_rss_bytes=measured,
            process_budget_bytes=_BASE.PROCESS_BUDGET_BYTES,
            previous_consecutive_stops=previous,
            max_resource_stops=_BASE.MAX_RESOURCE_STOPS,
        )
    except resource_policy.ResourcePolicyError as exc:
        raise _BASE.CampaignError(f"invalid resource stop evidence: {exc}") from exc
    next_count = decision["next_consecutive_stops"]
    if next_count:
        counts[cell_id] = next_count
    else:
        counts.pop(cell_id, None)
    if decision["escalate"] is not True:
        return None
    row = state.get("cells", {}).get(cell_id)
    if not isinstance(row, dict) or row.get("status") in _BASE.TERMINAL:
        raise _BASE.CampaignError("resource stop targets an invalid or terminal cell")
    blocker = f"resource-stop ceiling reached: {decision['detail']}"
    row["status"] = "blocked-execution"
    row["error"] = blocker
    row["blockers"] = [blocker]
    _BASE._append_event(
        "resource-stop-escalated", cell_id=cell_id,
        consecutive_stops=decision["consecutive_stops"], sole_live=sole_live,
        reason=decision["reason"], classification=decision["classification"],
        measured_cell_rss_bytes=decision["measured_cell_rss_bytes"],
        cell_at_or_over_budget=decision["cell_at_or_over_budget"],
        blocker=blocker,
    )
    return blocker


def _truthful_enforce_pool_budget(
        plan: dict[str, Any], state: dict[str, Any],
        live_cells: dict[str, Any], samples_by_cell: dict[str, dict[str, Any]],
        aggregate: int) -> list[str]:
    """Give the frozen guard exact per-cell samples during stop classification."""
    if _RESOURCE_STOP_SAMPLES:
        raise _BASE.CampaignError("nested resource stop measurement context")
    for cell_id, sample in samples_by_cell.items():
        measured = sample.get("tree_rss_bytes") if isinstance(sample, dict) else None
        if isinstance(measured, bool) or not isinstance(measured, int) or measured < 0:
            raise _BASE.CampaignError("active child RSS sample is invalid")
        _RESOURCE_STOP_SAMPLES[cell_id] = measured
    try:
        return _ORIGINAL_ENFORCE_POOL_BUDGET(
            plan, state, live_cells, samples_by_cell, aggregate
        )
    finally:
        _RESOURCE_STOP_SAMPLES.clear()


def configure(overlay: dict[str, Any]) -> None:
    global _OVERLAY
    errors = stacked.validate_overlay(overlay)
    if errors:
        raise stacked.OverlayError("invalid accelerated admission overlay: " + "; ".join(errors))
    policy = overlay["policy"]
    _BASE.SAFETY_MARGIN_BYTES = int(policy["minimum_margin_bytes"])
    _BASE.SWAP_TOLERANCE_MB = float(policy["swap_stale_tolerance_mb"])
    _BASE.MAX_LANES = min(int(policy["max_lanes"]), stacked.MAX_LANES)
    _BASE._cell_reservation = _accelerated_reservation
    _BASE._scan_runnable_heads = _accelerated_scan_heads
    _BASE._record_resource_stop = _truthful_record_resource_stop
    _BASE._enforce_pool_budget = _truthful_enforce_pool_budget
    _BASE._owner_alive = _owner_alive
    _BASE.start_queue = _start_queue
    _OVERLAY = overlay


def _owner_alive(record: Any, plan: dict[str, Any]) -> bool:
    if not isinstance(record, dict) or record.get("schema") != _BASE.PID_SCHEMA \
            or record.get("version") != _BASE.VERSION \
            or record.get("plan_sha256") != plan.get("plan_sha256") \
            or record.get("pid_record_sha256") != _BASE._hash_value(
                _BASE._without(record, "pid_record_sha256")
            ):
        return False
    nonce = record.get("ownership_nonce")
    identity = _BASE._process_identity(record.get("pid"))
    if identity is None or not isinstance(nonce, str) \
            or _BASE.NONCE_RE.fullmatch(nonce) is None:
        return False
    command, started = identity
    entrypoint = (
        "doctor_v5_ultra_accelerated_queue.py run" in command
        or "doctor_v5_ultra_queue.py run" in command
    )
    return (started == record.get("process_started")
            and hashlib.sha256(command.encode("utf-8")).hexdigest()
            == record.get("process_command_sha256")
            and entrypoint and f"--nonce {nonce}" in command)


def _start_queue() -> int:
    plan = _BASE._load_plan()
    owner = _BASE._read_json(_BASE.PID_FILE, {})
    if _owner_alive(owner, plan):
        print(f"[doctor-v5-ultra-accelerated] already active pid={owner['pid']}")
        return 0
    control = _BASE._load_control(plan)
    if control["mode"] != "run":
        _BASE.set_control("run")
    nonce = secrets.token_hex(16)
    command = [sys.executable, str(Path(__file__).resolve()), "run", "--nonce", nonce]
    if shutil.which("caffeinate"):
        command = ["caffeinate", "-dimsu", *command]
    _BASE.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    assert _OVERLAY is not None
    env[stacked.ENV_OVERLAY] = str(stacked.DEFAULT_OVERLAY.resolve())
    env[stacked.ENV_OVERLAY_SHA256] = _OVERLAY["overlay_sha256"]
    with _BASE.LOG_FILE.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command, cwd=_BASE.ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            close_fds=True, shell=False,
        )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        record = _BASE._read_json(_BASE.PID_FILE, {})
        if record.get("ownership_nonce") == nonce and _owner_alive(record, plan):
            print(f"[doctor-v5-ultra-accelerated] detached pid={record['pid']} "
                  f"log={_BASE.LOG_FILE}")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    raise _BASE.CampaignError("accelerated detached ownership handshake failed")


def _activation_preflight(overlay: dict[str, Any]) -> dict[str, Any]:
    plan = stacked._read_json(stacked.PLAN)
    campaign = stacked._read_json(stacked.CAMPAIGN)
    state = stacked._read_json(stacked.QUEUE_STATE)
    stacked._validate_live_documents(plan, campaign, state)
    try:
        snapshot = stacked.ram_scheduler.resource_snapshot(str(stacked.ROOT))
    except Exception as exc:
        snapshot = {"error": f"{type(exc).__name__}: {exc}"}
    health = stacked.resource_health(snapshot, stacked._thermal_probe())
    result = stacked.activation_preflight(
        overlay, plan, campaign, state, health,
        singleton_lease_available=stacked._lease_available(stacked.QUEUE_LOCK),
        heavy_lease_available=stacked._lease_available(stacked.HEAVY_LOCK),
    )
    generation_errors = _active_generation_errors(overlay)
    if generation_errors:
        result["blockers"].extend(generation_errors)
        result["ready"] = False
    result["mode"] = "one-time-activation"
    return result


def _terminal_subset_errors(overlay: dict[str, Any], campaign: dict[str, Any]) -> list[str]:
    seal = overlay.get("immutable_terminal_seal", {})
    rows = seal.get("rows") if isinstance(seal, dict) else None
    if not isinstance(rows, list) or seal.get("rows_sha256") != stacked._hash_value(rows):
        return ["admission overlay terminal seal is invalid"]
    current = stacked._terminal_seal(campaign)
    current_rows = {row["cell_id"]: row for row in current["rows"]}
    errors = []
    for row in rows:
        cell_id = row.get("cell_id") if isinstance(row, dict) else None
        if current_rows.get(cell_id) != row:
            errors.append(f"overlay-sealed terminal record changed: {cell_id}")
    return errors


def _resume_preflight(overlay: dict[str, Any]) -> dict[str, Any]:
    """Resume-safe validation: immutable generation yes, mutable snapshots no."""
    blockers = list(stacked.validate_overlay(overlay))
    plan = stacked._read_json(stacked.PLAN)
    campaign = stacked._read_json(stacked.CAMPAIGN)
    state = stacked._read_json(stacked.QUEUE_STATE)
    try:
        stacked._validate_live_documents(plan, campaign, state)
    except stacked.OverlayError as exc:
        blockers.append(str(exc))
    if overlay.get("source_bindings", {}).get("plan_sha256") != plan.get("plan_sha256"):
        blockers.append("live plan differs from the activated overlay")
    if not stacked._reference_matches(
            overlay.get("source_bindings", {}).get("observer_source")):
        blockers.append("reviewed observer source differs from the activated overlay")
    blockers.extend(stacked._observer_structure_errors())
    if not any(row.startswith("live observer") for row in blockers):
        observer = stacked._read_json(stacked.OBSERVER_STATE)
        if overlay.get("simulation", {}).get("gpt_oss_120b_execution_ready") is True \
                and observer.get("gpt_oss_120b_execution_ready") is not True:
            blockers.append("live observer regressed from staged 120B readiness")
    blockers.extend(_terminal_subset_errors(overlay, campaign))
    blockers.extend(_active_generation_errors(overlay))
    try:
        snapshot = stacked.ram_scheduler.resource_snapshot(str(stacked.ROOT))
    except Exception as exc:
        snapshot = {"error": f"{type(exc).__name__}: {exc}"}
    health = stacked.resource_health(snapshot, stacked._thermal_probe())
    if health.get("ok") is not True:
        blockers.extend(f"resource: {row}" for row in health.get("blockers", []))
    return {"ready": not blockers, "blockers": blockers, "mode": "resume-safe"}


def _requires_one_time_preflight(control: dict[str, Any], state: dict[str, Any]) -> bool:
    return (control.get("mode") in {"pause", "drain"}
            and state.get("status") in {"paused", "drained"})


def main() -> int:
    _verify_static_bindings()
    overlay = stacked.load_activated_overlay()
    if overlay is None:
        raise stacked.OverlayError("accelerated queue requires both overlay activation keys")
    plan = _BASE._load_plan()
    control = _BASE._load_control(plan)
    state = _BASE._load_state(plan)
    preflight = (_activation_preflight(overlay)
                 if _requires_one_time_preflight(control, state)
                 else _resume_preflight(overlay))
    if preflight.get("ready") is not True:
        raise stacked.OverlayError(
            "accelerated queue activation refused: " + "; ".join(preflight["blockers"])
        )
    configure(overlay)
    return int(_BASE.main())


def _selftest() -> None:
    fixture = {
        "model_label": "72B", "nominal_params_b": 72.0,
        "parameter_manifest": {"largest_source_shard_bytes": 4_000_000_000,
                               "source_weight_bytes": 144_000_000_000},
        "admission": {"whole_parent_residency_assumed": False},
    }
    old_plan = _BASE._load_plan
    old_margin = _BASE.SAFETY_MARGIN_BYTES
    old_observed = dict(_BASE._OBSERVED_TIER_RSS)
    try:
        _BASE.SAFETY_MARGIN_BYTES = stacked.MIN_DYNAMIC_MARGIN_BYTES
        _BASE._OBSERVED_TIER_RSS.clear()
        _BASE._load_plan = lambda: {"plan_sha256": "0" * 64, "cells": []}
        globals()["_evidence"] = lambda: {}
        assert _accelerated_reservation(fixture) == 66_000_000_000
        globals()["_evidence"] = lambda: {"72B": {"samples": 1, "peak_bytes": 32_000_000_000}}
        _BASE._OBSERVED_TIER_RSS["72B"] = 32_000_000_000
        assert _accelerated_reservation(fixture) == 40_000_000_000
        globals()["_evidence"] = lambda: {"72B": {"samples": 3, "peak_bytes": 32_000_000_000}}
        assert _accelerated_reservation(fixture) == 32_000_000_000
    finally:
        _BASE._load_plan = old_plan
        _BASE.SAFETY_MARGIN_BYTES = old_margin
        _BASE._OBSERVED_TIER_RSS.clear()
        _BASE._OBSERVED_TIER_RSS.update(old_observed)
    print(json.dumps({"status": "ok", "base_sha256": BASE_SHA256}, sort_keys=True))


if __name__ == "__main__":
    try:
        if len(sys.argv) == 2 and sys.argv[1] == "selftest":
            _selftest()
            raise SystemExit(0)
        raise SystemExit(main())
    except (stacked.OverlayError, accel_loader.AccelerationBindingError) as exc:
        print(f"doctor_v5_ultra_accelerated_queue: {exc}", file=sys.stderr)
        raise SystemExit(2)
