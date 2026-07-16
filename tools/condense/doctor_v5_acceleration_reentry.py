#!/usr/bin/env python3.12
"""Prepare, audit, atomically promote, and roll back Doctor V5 acceleration.

The active campaign is never edited by ``status`` or ``audit``.  ``prepare``
requires an already drained/paused queue and freezes each Qwen source once,
then builds pending-only runtime specs and a replacement registry in staging.
``activate`` revalidates parity, terminal evidence, leases, and every staged
hash before changing anything.  Completed runtime specs and result directories
are outside the transaction allowlist.
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
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import doctor_v5_accel_loader as accel_loader
import doctor_v5_adapter_abi as abi
import doctor_v5_block_parallel_config_matrix as config_matrix
import doctor_v5_block_parallel_real_canary as real_canary
import doctor_v5_gc_runtime_transition as gc_transition
import doctor_v5_qwen_treatment_block_parallel_adapter as treatment_accel
import doctor_v5_source_seal as source_seal
import doctor_v5_stacked_admission as stacked
import doctor_v5_strand_ladder_block_parallel_adapter as control_accel
import doctor_v5_ultra_queue as queue


ULTRA_ROOT = ROOT / "reports/condense/doctor_v5_ultra"
PLAN = ULTRA_ROOT / "campaign_plan.json"
STATE = ULTRA_ROOT / "queue_state.json"
CAMPAIGN = ULTRA_ROOT / "campaign.json"
CONTROL = ULTRA_ROOT / "control.json"
REGISTRY = ULTRA_ROOT / "adapter_registry.json"
RESULTS = ULTRA_ROOT / "results"
STAGE_ROOT = ULTRA_ROOT / "staged_acceleration"
GENERATION_ROOT = STAGE_ROOT / "pending_runtime"
PACKET = STAGE_ROOT / "pending_runtime_packet.json"
MARKER = STAGE_ROOT / "active_stack.json"
JOURNAL = STAGE_ROOT / "activation_journal.json"
OVERLAY = stacked.DEFAULT_OVERLAY
ACCEL_QUEUE = HERE / "doctor_v5_ultra_accelerated_queue.py"
ACCEL_AUTORESUME = HERE / "doctor_v5_ultra_accelerated_autoresume.py"
QUANTIZER = ROOT / "build/strand-block-parallel/release/quantize-model-block-parallel"
SERIAL_CANDIDATE = ROOT / "build/strand-block-serial/release/quantize-model"
LIVE_QUANTIZER = ROOT / "vendor/strand-quant/target/release/quantize-model"
PARITY_ROOT = ROOT / "build/strand-block-parallel"
PARITY_RECEIPTS = (
    PARITY_ROOT / "parity-receipt.json",
    PARITY_ROOT / "parity-receipt-8t.json",
    PARITY_ROOT / "parity-receipt-16t.json",
    PARITY_ROOT / "parity-receipt-20t.json",
)
E2E_PARITY_RECEIPT = PARITY_ROOT / "quantize-model-parity-receipt.json"
LAUNCH_AGENT = (
    Path.home() / "Library/LaunchAgents/com.hawking.doctorv5ultra.autoresume.plist"
)
LAUNCH_LABEL = "com.hawking.doctorv5ultra.autoresume"
TERMINAL = frozenset({"complete", "negative", "unsupported"})
PACKET_SCHEMA = "hawking.doctor_v5_acceleration_pending_runtime.v1"
MARKER_SCHEMA = "hawking.doctor_v5_acceleration_active_marker.v1"
JOURNAL_SCHEMA = "hawking.doctor_v5_acceleration_journal.v1"


class ReentryError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: row for name, row in value.items() if name != key}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) \
                or info.st_size > 64 * 1024 * 1024:
            raise ReentryError(f"invalid JSON artifact: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReentryError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReentryError(f"JSON root is not an object: {path}")
    return value


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
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


def _hash_file(path: Path) -> tuple[str, int]:
    return accel_loader.hash_file(path)


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = _hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _validate_artifact(row: Any) -> bool:
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
        return False
    try:
        digest, size = _hash_file(Path(row["path"]))
    except (OSError, ValueError, accel_loader.AccelerationBindingError):
        return False
    return digest == row["sha256"] and size == row["bytes"]


def _load_live() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, state, campaign = _read_json(PLAN), _read_json(STATE), _read_json(CAMPAIGN)
    errors = queue.validate_plan(plan)
    if errors:
        raise ReentryError("live campaign plan is invalid: " + "; ".join(errors))
    if state.get("plan_sha256") != plan["plan_sha256"] \
            or campaign.get("plan_sha256") != plan["plan_sha256"]:
        raise ReentryError("live plan/state/campaign bindings differ")
    return plan, state, campaign


def _quiescent() -> tuple[bool, list[str]]:
    _, state, campaign = _load_live()
    blockers: list[str] = []
    control = _read_json(CONTROL)
    if control.get("mode") not in {"drain", "pause"}:
        blockers.append("control mode is not drain/pause")
    if state.get("active_children") or state.get("active_cells"):
        blockers.append("queue state still records active children")
    if campaign.get("active_children") or campaign.get("active_cells"):
        blockers.append("campaign projection still records active children")
    if state.get("status") not in {"drained", "paused"}:
        blockers.append("queue status is not drained/paused")
    if not stacked._lease_available(stacked.QUEUE_LOCK):
        blockers.append("queue singleton lease is still owned")
    if not stacked._lease_available(stacked.HEAVY_LOCK):
        blockers.append("Studio heavy lease is still owned")
    return not blockers, blockers


def _parity_status() -> dict[str, Any]:
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    gate_sha: str | None = None
    observed_threads: set[int] = set()
    for path in PARITY_RECEIPTS:
        try:
            doc = _read_json(path)
        except ReentryError as exc:
            blockers.append(str(exc)); continue
        if doc.get("schema") != "hawking.strand.block-parallel-parity.v1" \
                or doc.get("status") != "pass" or doc.get("feature") != "block-parallel":
            blockers.append(f"invalid block parity receipt: {path}"); continue
        threads = doc.get("threads")
        if threads not in {4, 8, 16, 20} or doc.get("scratch_budget_bytes") != 256 * 1024 * 1024:
            blockers.append(f"parity receipt controls differ: {path}")
        cases = doc.get("cases")
        if not isinstance(cases, list) or len(cases) != 3 \
                or any(not isinstance(row, dict)
                       or row.get("exact_match") is not True
                       or row.get("serial_sha256") != row.get("parallel_sha256")
                       or not isinstance(row.get("speedup"), (int, float))
                       or float(row["speedup"]) <= 1 for row in cases):
            blockers.append(f"parity receipt cases are not exact/accelerated: {path}")
        binary = Path(str(doc.get("binary_path", "")))
        try:
            observed, _ = _hash_file(binary)
        except Exception:
            observed = None
        if observed != doc.get("binary_sha256"):
            blockers.append(f"parity gate binary identity changed: {path}")
        if gate_sha is not None and gate_sha != doc.get("binary_sha256"):
            blockers.append("parity receipts bind different gate binaries")
        payload = bytearray()
        payload.extend(str(doc.get("schema", "")).encode())
        payload.extend(str(doc.get("binary_sha256", "")).encode())
        if isinstance(threads, int):
            payload.extend(threads.to_bytes(8, "little", signed=False))
        budget = doc.get("scratch_budget_bytes")
        if isinstance(budget, int):
            payload.extend(budget.to_bytes(8, "little", signed=False))
        if isinstance(cases, list):
            for case in cases:
                if isinstance(case, dict):
                    for key in ("name", "input_sha256", "serial_sha256",
                                "parallel_sha256"):
                        payload.extend(str(case.get(key, "")).encode())
        if hashlib.sha256(payload).hexdigest() != doc.get("canonical_payload_sha256"):
            blockers.append(f"parity receipt payload hash differs: {path}")
        gate_sha = doc.get("binary_sha256")
        if isinstance(threads, int):
            observed_threads.add(threads)
        rows.append(_artifact(path))
    if observed_threads != {4, 8, 16, 20}:
        blockers.append("parity receipts do not cover 4/8/16/20 block threads")
    try:
        quantizer = _artifact(QUANTIZER)
        serial = _artifact(SERIAL_CANDIDATE)
        live = _artifact(LIVE_QUANTIZER)
    except Exception as exc:
        blockers.append(f"quantizer artifact missing: {exc}")
        quantizer = serial = live = None
    e2e = None
    try:
        doc = _read_json(E2E_PARITY_RECEIPT)
        e2e = _artifact(E2E_PARITY_RECEIPT)
        if doc.get("schema") != "hawking.strand.quantize-model-block-parallel-parity.v1" \
                or doc.get("status") != "pass":
            blockers.append("end-to-end quantizer parity receipt is not an exact pass")
        if isinstance(quantizer, dict) and doc.get("parallel_binary_sha256") \
                != quantizer["sha256"]:
            blockers.append("end-to-end receipt binds a different accelerated quantizer")
        if isinstance(serial, dict) and doc.get("canonical_binary_sha256") \
                != serial["sha256"]:
            blockers.append("end-to-end receipt binds a different serial candidate")
        fixture = Path(str(doc.get("fixture", "")))
        try:
            fixture_sha, _ = _hash_file(fixture)
        except Exception:
            fixture_sha = None
        if fixture_sha != doc.get("fixture_sha256"):
            blockers.append("end-to-end fixture identity changed")
        exact_keys = ("dense_exact_match", "sidecar_exact_match",
                      "packed_v2_exact_match")
        if any(doc.get(key) is not True for key in exact_keys):
            blockers.append("end-to-end receipt lacks exact dense/sidecar/packed-v2 equality")
        invocation = doc.get("invocation_contract")
        expected_invocation = (
            "STRAND_NO_GPU=1;bits=2;l=8;rht=off;tensor_scope=linear_default;"
            "outer_threads=1;block_threads=20;"
            "block_scratch_budget_bytes=268435456"
        )
        if invocation != expected_invocation:
            blockers.append("end-to-end invocation contract differs")
        payload = bytearray()
        for value in (
            doc.get("schema"), doc.get("canonical_binary_sha256"),
            doc.get("parallel_binary_sha256"), doc.get("fixture_sha256"),
            doc.get("dense_output_sha256"), doc.get("sidecar_sha256"),
            doc.get("packed_v2_archive_sha256"), invocation,
        ):
            payload.extend(str(value or "").encode()); payload.append(0)
        if hashlib.sha256(payload).hexdigest() != doc.get("canonical_payload_sha256"):
            blockers.append("end-to-end receipt payload hash differs")
    except ReentryError:
        blockers.append("end-to-end quantize-model parity receipt is absent")
    return {"ok": not blockers, "blockers": blockers, "receipts": rows,
            "end_to_end_receipt": e2e, "quantizer": quantizer,
            "serial_candidate": serial, "live_quantizer": live}


def _terminal_seal(plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cells = {row["cell_id"]: row for row in plan["cells"]}
    rows: list[dict[str, Any]] = []
    for cell_id, state_row in state["cells"].items():
        if state_row["status"] not in TERMINAL:
            continue
        cell = cells[cell_id]
        if state_row["status"] == "complete":
            required = (
                ROOT / cell["runtime_spec_path"], RESULTS / cell_id / "request.json",
                RESULTS / cell_id / "adapter_registry.json", RESULTS / cell_id / "result.json",
                RESULTS / cell_id / "execution_receipt.json",
            )
        else:
            required = (ROOT / cell["runtime_spec_path"], ROOT / cell["disposition_path"])
        missing = [str(path) for path in required
                   if not path.is_file() or path.is_symlink()]
        if missing:
            raise ReentryError(f"terminal cell lacks required evidence {cell_id}: "
                               + ", ".join(missing))
        artifacts = [_artifact(path) for path in required]
        rows.append({"cell_id": cell_id, "status": state_row["status"],
                     "cell_identity_sha256": cell["cell_identity_sha256"],
                     "state_evidence": {key: state_row.get(key) for key in (
                         "request_sha256", "result_sha256", "execution_receipt_sha256",
                         "disposition_sha256",
                     )},
                     "artifacts": artifacts})
    rows.sort(key=lambda row: row["cell_id"])
    return {"count": len(rows), "rows": rows, "rows_sha256": _hash_value(rows)}


def _validate_terminal_seal(seal: Any, plan: dict[str, Any], state: dict[str, Any]) \
        -> list[str]:
    errors: list[str] = []
    if not isinstance(seal, dict) or seal.get("rows_sha256") != _hash_value(seal.get("rows")):
        return ["terminal seal hash is invalid"]
    current_terminal = {cell_id for cell_id, row in state["cells"].items()
                        if row["status"] in TERMINAL}
    sealed_ids = {row.get("cell_id") for row in seal.get("rows", [])
                  if isinstance(row, dict)}
    if sealed_ids != current_terminal or seal.get("count") != len(current_terminal) \
            or len(seal.get("rows", [])) != len(current_terminal):
        errors.append("terminal seal does not exactly cover current terminal cells")
    cells = {row["cell_id"]: row for row in plan["cells"]}
    for row in seal.get("rows", []):
        if not isinstance(row, dict):
            errors.append("terminal seal row is invalid"); continue
        cell_id = row.get("cell_id")
        if cell_id not in cells:
            errors.append(f"terminal seal cell is unknown: {cell_id}"); continue
        state_row = state["cells"][cell_id]
        if row.get("status") != state_row["status"] \
                or row.get("cell_identity_sha256") != cells[cell_id]["cell_identity_sha256"]:
            errors.append(f"terminal identity/status changed: {cell_id}")
        expected_state = {key: state_row.get(key) for key in (
            "request_sha256", "result_sha256", "execution_receipt_sha256",
            "disposition_sha256",
        )}
        if row.get("state_evidence") != expected_state:
            errors.append(f"terminal state evidence changed: {cell_id}")
        expected_artifact_count = 5 if state_row["status"] == "complete" else 2
        if len(row.get("artifacts", [])) != expected_artifact_count \
                or any(not _validate_artifact(artifact)
                       for artifact in row.get("artifacts", [])):
            errors.append(f"terminal evidence changed: {row.get('cell_id')}")
    return errors


def _freeze_pending_sources(plan: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    labels = sorted({cell["model_label"] for cell in plan["cells"]
                     if cell["model_family"] == "qwen2.5-dense"
                     and state["cells"][cell["cell_id"]]["status"] not in TERMINAL},
                    key=lambda value: float(value.removesuffix("B")))
    rows = []
    for label in labels:
        seal_path = source_seal.default_path(label)
        if seal_path.is_file():
            document = _read_json(seal_path)
            schema = document.get("schema")
            if schema == source_seal.SCHEMA_V1:
                # A reboot may renumber APFS st_dev while every stable file
                # field and byte remains identical.  Migration performs the
                # one required full-content pass, archives the v1 authority,
                # and emits the persistent-volume v2 receipt.  Calling freeze
                # here would discard that transition evidence.
                source_seal.migrate_v1(seal_path, workers=4)
            elif schema == source_seal.SCHEMA:
                errors = source_seal.validate_document(
                    document, verify_structural=True
                )
                if errors:
                    raise ReentryError(
                        f"existing v2 source seal is invalid for {label}: "
                        + "; ".join(errors)
                    )
            else:
                raise ReentryError(
                    f"existing source seal has an unknown schema for {label}"
                )
        else:
            source_seal.freeze(label, workers=4)
        rows.append(_artifact(seal_path))
    return rows


def _staged_registry(output: Path) -> dict[str, Any]:
    current = _read_json(REGISTRY)
    control_ids = {control_accel.ADAPTER_ID}
    treatment_ids = {row["adapter_id"] for row in treatment_accel.OPERATIONS.values()}
    entries = []
    for entry in current["entries"]:
        row = dict(entry)
        if row["adapter_id"] in control_ids | treatment_ids:
            source = (Path(control_accel.__file__) if row["adapter_id"] in control_ids
                      else Path(treatment_accel.__file__)).resolve()
            row["adapter_version"] = "2-block-parallel"
            row["source_path"] = str(source.relative_to(ROOT))
            row["source_sha256"] = _hash_file(source)[0]
            argv = list(row["entrypoint_argv"])
            argv[1] = str(source.relative_to(ROOT))
            row["entrypoint_argv"] = argv
        entries.append(row)
    registry = abi.build_registry(entries, created_at=current["created_at"], output_path=output)
    errors = abi.validate_registry(registry, verify_files=True, base_dir=ROOT)
    if errors:
        raise ReentryError("staged registry is invalid: " + "; ".join(errors))
    return registry


def _stage_specs(plan: dict[str, Any], state: dict[str, Any], root: Path) \
        -> list[dict[str, Any]]:
    rows = []
    for cell in plan["cells"]:
        cell_id = cell["cell_id"]
        if cell["model_family"] != "qwen2.5-dense" \
                or state["cells"][cell_id]["status"] in TERMINAL:
            continue
        target = ROOT / cell["runtime_spec_path"]
        old = _read_json(target)
        staged = root / f"{cell_id}.json"
        kwargs = {
            "label": cell["model_label"], "rate_id": cell["rate_id"],
            "cell_id": cell_id, "cell_identity_sha256": cell["cell_identity_sha256"],
            "program_spec_sha256": old["program_spec_sha256"],
            "resource_admission_sha256": old["resource_admission_sha256"],
            "evaluation_mode": old["evaluation"]["mode"],
            "disk_reserve_bytes": old["resources"]["disk_reserve_bytes"],
            "scratch_budget_bytes": old["resources"]["scratch_budget_bytes"],
            "threads": old["resources"]["threads"], "output_path": staged,
        }
        if cell["branch"] == "codec_control":
            document = control_accel.build_spec(**kwargs)
        else:
            dependencies = [{key: row[key] for key in (
                "branch", "cell_id", "cell_identity_sha256"
            )} for row in old["dependencies"]]
            document = treatment_accel.build_spec(
                operation=cell["branch"], dependencies=dependencies, **kwargs
            )
        if document["program_spec_sha256"] != old["program_spec_sha256"] \
                or queue._runtime_program_payload(document) \
                != queue._runtime_program_payload(old):
            raise ReentryError(f"accelerated runtime changed cell semantics: {cell_id}")
        runtime, _, errors = queue._validate_runtime_spec(
            cell, document, staged, verify_inputs=False
        )
        if errors or runtime is None:
            raise ReentryError(f"staged runtime is invalid for {cell_id}: " + "; ".join(errors))
        rows.append({"cell_id": cell_id, "target": str(target.resolve()),
                     "before": _artifact(target), "staged": _artifact(staged)})
    rows.sort(key=lambda row: row["cell_id"])
    return rows


def _real_canary_receipt() -> dict[str, Any]:
    """Reuse exact current physical evidence; rerun only when it is absent/stale."""
    if real_canary.RECEIPT.is_file() and not real_canary.RECEIPT.is_symlink():
        document = _read_json(real_canary.RECEIPT)
        errors = real_canary.validate(document, verify_files=True)
        if not errors:
            return document
    document = real_canary.run()
    errors = real_canary.validate(document, verify_files=True)
    if errors:
        raise ReentryError("real-tensor canary failed: " + "; ".join(errors))
    return document


def _implementation_generation_seed() -> dict[str, Any]:
    return {
        "acceleration_loader": _artifact(Path(accel_loader.__file__)),
        "control_adapter": _artifact(Path(control_accel.__file__)),
        "treatment_adapter": _artifact(Path(treatment_accel.__file__)),
        "source_seal_module": _artifact(Path(source_seal.__file__)),
        "gc_transition_module": _artifact(Path(gc_transition.__file__)),
        "gc_transition_authority": _artifact(
            gc_transition.DEFAULT_AUTHORITY_PATH
        ),
    }


def prepare(*, packet_path: Path = PACKET) -> dict[str, Any]:
    ready, blockers = _quiescent()
    if not ready:
        raise ReentryError("prepare requires a quiescent checkpoint: " + "; ".join(blockers))
    packet_path = packet_path.resolve()
    try:
        packet_path.relative_to(STAGE_ROOT.resolve())
    except ValueError as exc:
        raise ReentryError("pending packet output must remain below staged_acceleration") \
            from exc
    if packet_path.exists() and packet_path.is_symlink():
        raise ReentryError("pending packet output cannot be a symlink")
    parity = _parity_status()
    if not parity["ok"]:
        raise ReentryError("parity gate is not ready: " + "; ".join(parity["blockers"]))
    gc_transition.validate_authority(gc_transition.DEFAULT_AUTHORITY_PATH)
    plan, state, _ = _load_live()
    source_seals = _freeze_pending_sources(plan, state)
    matrix_document = config_matrix.run()
    matrix_errors = config_matrix.validate(matrix_document, verify_files=True)
    if matrix_errors:
        raise ReentryError("exact 10x4 config parity matrix failed: "
                           + "; ".join(matrix_errors))
    canary_document = _real_canary_receipt()
    terminal = _terminal_seal(plan, state)
    implementation = _implementation_generation_seed()
    generation_seed = {
        "plan_sha256": plan["plan_sha256"], "quantizer": parity["quantizer"],
        "terminal_rows_sha256": terminal["rows_sha256"], "source_seals": source_seals,
        "exact_config_matrix": _artifact(config_matrix.RECEIPT),
        "real_tensor_canary": _artifact(real_canary.RECEIPT),
        "implementation": implementation,
    }
    generation = _hash_value(generation_seed)
    root = GENERATION_ROOT / generation
    specs_root = root / "runtime_specs"
    specs_root.mkdir(parents=True, exist_ok=True)
    staged_registry_path = root / "adapter_registry.json"
    _staged_registry(staged_registry_path)
    specs = _stage_specs(plan, state, specs_root)
    packet = {
        "schema": PACKET_SCHEMA, "created_at": _now(), "generation_id": generation,
        "plan_sha256": plan["plan_sha256"], "quantizer_parity": parity,
        "source_seals": source_seals, "terminal_seal": terminal,
        "exact_config_matrix": _artifact(config_matrix.RECEIPT),
        "real_tensor_canary": _artifact(real_canary.RECEIPT),
        "registry": {"target": str(REGISTRY.resolve()), "before": _artifact(REGISTRY),
                     "staged": _artifact(staged_registry_path)},
        "pending_runtime_specs": specs, "pending_cell_count": len(specs),
        "completed_runtime_specs_mutated": False,
        "source_deletion_permitted": False,
    }
    packet["packet_sha256"] = _hash_value(packet)
    _atomic_json(packet_path, packet)
    return packet


def validate_packet(packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_keys = {
        "schema", "created_at", "generation_id", "plan_sha256",
        "quantizer_parity", "source_seals", "terminal_seal",
        "exact_config_matrix", "real_tensor_canary", "registry",
        "pending_runtime_specs", "pending_cell_count",
        "completed_runtime_specs_mutated", "source_deletion_permitted",
        "packet_sha256",
    }
    if set(packet) != expected_keys:
        errors.append("pending runtime packet keys are invalid")
    if packet.get("schema") != PACKET_SCHEMA \
            or packet.get("packet_sha256") != _hash_value(_without(packet, "packet_sha256")):
        errors.append("pending runtime packet schema/hash is invalid")
    plan, state, _ = _load_live()
    if packet.get("plan_sha256") != plan["plan_sha256"]:
        errors.append("pending runtime packet plan binding changed")
    errors.extend(_validate_terminal_seal(packet.get("terminal_seal"), plan, state))
    parity = _parity_status()
    if not parity["ok"]:
        errors.extend(f"parity: {row}" for row in parity["blockers"])
    if packet.get("quantizer_parity") != parity:
        errors.append("packet parity evidence differs from current frozen evidence")
    for key, module in (("exact_config_matrix", config_matrix),
                        ("real_tensor_canary", real_canary)):
        row = packet.get(key)
        if not _validate_artifact(row):
            errors.append(f"{key} receipt artifact changed")
            continue
        try:
            document = _read_json(Path(row["path"]))
            errors.extend(f"{key}: {item}" for item in
                          module.validate(document, verify_files=True))
        except (ReentryError, OSError, ValueError) as exc:
            errors.append(f"{key} receipt cannot be verified: {exc}")
    for row in packet.get("source_seals", []):
        if not _validate_artifact(row):
            errors.append("a source seal artifact changed")
            continue
        doc = _read_json(Path(row["path"]))
        errors.extend(f"source seal: {item}" for item in
                      source_seal.validate_document(doc, verify_structural=True))
    registry = packet.get("registry")
    if not isinstance(registry, dict) or not _validate_artifact(registry.get("before")) \
            or not _validate_artifact(registry.get("staged")):
        errors.append("registry transaction binding changed")
    elif set(registry) != {"target", "before", "staged"} \
            or Path(registry["target"]).resolve() != REGISTRY.resolve() \
            or Path(registry["before"]["path"]).resolve() != REGISTRY.resolve():
        errors.append("registry transaction paths are outside the allowlist")
    else:
        expected_registry = (GENERATION_ROOT / str(packet.get("generation_id")) /
                             "adapter_registry.json").resolve()
        if Path(registry["staged"]["path"]).resolve() != expected_registry:
            errors.append("staged registry path is outside its generation")
    cells = {row["cell_id"]: row for row in plan["cells"]}
    spec_rows = packet.get("pending_runtime_specs")
    if not isinstance(spec_rows, list):
        errors.append("pending runtime spec inventory is not a list")
        spec_rows = []
    expected_ids = {cell["cell_id"] for cell in plan["cells"]
                    if cell["model_family"] == "qwen2.5-dense"
                    and state["cells"][cell["cell_id"]]["status"] not in TERMINAL}
    actual_ids = [row.get("cell_id") for row in spec_rows if isinstance(row, dict)]
    if any(not isinstance(cell_id, str) for cell_id in actual_ids) \
            or len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        errors.append("pending runtime rows do not uniquely cover the exact pending set")
    generation_root = (GENERATION_ROOT / str(packet.get("generation_id")) /
                       "runtime_specs").resolve()
    input_cache: dict[str, tuple[str, int]] = {}
    source_cache: dict[str, tuple[str, int] | None] = {}
    for row in spec_rows:
        cell_id = row.get("cell_id") if isinstance(row, dict) else None
        if cell_id not in cells or state["cells"][cell_id]["status"] in TERMINAL:
            errors.append(f"runtime target is no longer pending: {cell_id}")
            continue
        if set(row) != {"cell_id", "target", "before", "staged"}:
            errors.append(f"runtime transaction keys are invalid: {cell_id}")
            continue
        canonical_target = (ROOT / cells[cell_id]["runtime_spec_path"]).resolve()
        canonical_staged = (generation_root / f"{cell_id}.json").resolve()
        try:
            target = Path(row["target"]).resolve()
            before_path = Path(row["before"]["path"]).resolve()
            staged_path = Path(row["staged"]["path"]).resolve()
        except (KeyError, TypeError, OSError):
            errors.append(f"runtime transaction paths are invalid: {cell_id}")
            continue
        if target != canonical_target or before_path != canonical_target \
                or staged_path != canonical_staged:
            errors.append(f"runtime transaction path is outside allowlist: {cell_id}")
        if not _validate_artifact(row.get("before")) or not _validate_artifact(row.get("staged")):
            errors.append(f"runtime transaction binding changed: {cell_id}")
            continue
        try:
            current, staged = _read_json(canonical_target), _read_json(canonical_staged)
            if staged.get("campaign_binding", {}).get("cell_id") != cell_id \
                    or queue._runtime_program_payload(staged) \
                    != queue._runtime_program_payload(current):
                errors.append(f"staged runtime semantics changed: {cell_id}")
            runtime, _, runtime_errors = queue._validate_runtime_spec(
                cells[cell_id], staged, canonical_staged, verify_inputs=False
            )
            if runtime is None or runtime_errors:
                errors.append(f"staged runtime validation failed {cell_id}: "
                              + "; ".join(runtime_errors))
            for input_row in staged.get("inputs", []):
                role = input_row.get("role") if isinstance(input_row, dict) else None
                try:
                    path = Path(input_row["path"]).resolve(strict=True)
                    expected_artifact = (input_row["sha256"], input_row["bytes"])
                    cache_key = str(path)
                    if isinstance(role, str) and role.startswith("source_shard:"):
                        if cache_key not in source_cache:
                            source_cache[cache_key] = source_seal.lookup(path)
                        observed = source_cache[cache_key]
                    else:
                        if cache_key not in input_cache:
                            input_cache[cache_key] = _hash_file(path)
                        observed = input_cache[cache_key]
                    if observed != expected_artifact:
                        errors.append(f"staged runtime input changed {cell_id}:{role}")
                except (OSError, KeyError, TypeError, ValueError,
                        source_seal.SourceSealError,
                        accel_loader.AccelerationBindingError) as exc:
                    errors.append(f"staged runtime input unavailable {cell_id}:{role}: {exc}")
        except (ReentryError, OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"staged runtime cannot be semantically verified {cell_id}: {exc}")
    expected = len(expected_ids)
    if packet.get("pending_cell_count") != expected or len(spec_rows) != expected:
        errors.append("pending runtime packet no longer has exact Qwen coverage")
    generation_seed = {
        "plan_sha256": plan["plan_sha256"],
        "quantizer": parity.get("quantizer"),
        "terminal_rows_sha256": packet.get("terminal_seal", {}).get("rows_sha256"),
        "source_seals": packet.get("source_seals"),
        "exact_config_matrix": packet.get("exact_config_matrix"),
        "real_tensor_canary": packet.get("real_tensor_canary"),
        "implementation": _implementation_generation_seed(),
    }
    if packet.get("generation_id") != _hash_value(generation_seed):
        errors.append("runtime generation id differs from its exact evidence seed")
    if packet.get("completed_runtime_specs_mutated") is not False \
            or packet.get("source_deletion_permitted") is not False:
        errors.append("pending runtime packet safety policy overclaims mutation/deletion")
    return errors


def adversarial_audit() -> dict[str, Any]:
    packet = _read_json(PACKET)
    baseline = validate_packet(packet)
    if baseline:
        raise ReentryError("cannot adversarially audit an invalid baseline: "
                           + "; ".join(baseline))

    def clone() -> dict[str, Any]:
        return json.loads(json.dumps(packet))

    def seal(document: dict[str, Any]) -> None:
        document["packet_sha256"] = _hash_value(_without(document, "packet_sha256"))

    probes: list[tuple[str, dict[str, Any], str]] = []
    duplicate = clone()
    duplicate["pending_runtime_specs"][-1] = dict(duplicate["pending_runtime_specs"][0])
    seal(duplicate)
    probes.append(("duplicate-and-omit", duplicate, "uniquely cover"))

    redirected = clone()
    redirected["pending_runtime_specs"][0]["target"] = str(PLAN.resolve())
    redirected["pending_runtime_specs"][0]["before"] = _artifact(PLAN)
    seal(redirected)
    probes.append(("target-redirection", redirected, "outside allowlist"))

    terminal_drop = clone()
    terminal_drop["terminal_seal"]["rows"].pop()
    terminal_drop["terminal_seal"]["count"] -= 1
    terminal_drop["terminal_seal"]["rows_sha256"] = _hash_value(
        terminal_drop["terminal_seal"]["rows"]
    )
    seal(terminal_drop)
    probes.append(("terminal-evidence-omission", terminal_drop,
                   "exactly cover current terminal"))

    overclaim = clone()
    overclaim["source_deletion_permitted"] = True
    seal(overclaim)
    probes.append(("deletion-overclaim", overclaim, "safety policy overclaims"))

    results = []
    for name, candidate, expected in probes:
        errors = validate_packet(candidate)
        if not any(expected in error for error in errors):
            raise ReentryError(f"adversarial probe was not rejected correctly: {name}")
        results.append({"name": name, "rejected": True,
                        "expected_error_fragment": expected,
                        "error_count": len(errors)})
    return {"schema": "hawking.doctor_v5_acceleration_adversarial_audit.v1",
            "ok": True, "probe_count": len(results), "probes": results}


def _acquire(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise ReentryError(f"required lease is owned: {path}") from exc
    return handle


def _acquire_both() -> tuple[Any, Any]:
    queue_lease = _acquire(stacked.QUEUE_LOCK)
    try:
        heavy_lease = _acquire(stacked.HEAVY_LOCK)
    except BaseException:
        queue_lease.close()
        raise
    return queue_lease, heavy_lease


def _replace_file(source: Path, target: Path) -> None:
    payload = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, target); _fsync_dir(target.parent)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass


def _reload_launch_agent() -> None:
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", f"{domain}/{LAUNCH_LABEL}"],
                   capture_output=True, check=False)
    launched = subprocess.run(["launchctl", "bootstrap", domain, str(LAUNCH_AGENT)],
                              capture_output=True, text=True, check=False)
    if launched.returncode != 0:
        raise ReentryError("cannot bootstrap Doctor V5 autoresume: "
                           + launched.stderr.strip())


def _restore_optional(target: Path, backup: Path, existed: bool) -> None:
    if existed:
        if not backup.is_file():
            raise ReentryError(f"required rollback artifact is absent: {backup}")
        _replace_file(backup, target)
    else:
        target.unlink(missing_ok=True)


def _write_marker(packet: dict[str, Any]) -> dict[str, Any]:
    overlay = _read_json(OVERLAY)
    marker = {
        "schema": MARKER_SCHEMA, "activated_at": _now(),
        "overlay_path": str(OVERLAY.resolve()),
        "overlay_sha256": overlay["overlay_sha256"],
        "pending_runtime_generation_sha256": packet["packet_sha256"],
        "accelerated_queue": _artifact(ACCEL_QUEUE),
        "accelerated_autoresume": _artifact(ACCEL_AUTORESUME),
    }
    marker["marker_sha256"] = _hash_value(marker)
    _atomic_json(MARKER, marker)
    return marker


def _configure_launch_agent(backup: Path) -> None:
    if not LAUNCH_AGENT.is_file() or LAUNCH_AGENT.is_symlink():
        raise ReentryError("Doctor V5 autoresume LaunchAgent is missing")
    shutil.copy2(LAUNCH_AGENT, backup)
    with LAUNCH_AGENT.open("rb") as handle:
        doc = plistlib.load(handle)
    argv = doc.get("ProgramArguments")
    if not isinstance(argv, list) or len(argv) < 2:
        raise ReentryError("Doctor V5 autoresume LaunchAgent argv is invalid")
    argv[1] = str(ACCEL_AUTORESUME.resolve())
    doc["ProgramArguments"] = argv
    payload = plistlib.dumps(doc, fmt=plistlib.FMT_XML, sort_keys=True)
    fd, raw = tempfile.mkstemp(prefix=f".{LAUNCH_AGENT.name}.", dir=LAUNCH_AGENT.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, LAUNCH_AGENT); _fsync_dir(LAUNCH_AGENT.parent)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass
    try:
        _reload_launch_agent()
    except BaseException:
        # A failed bootstrap is not allowed to strand the campaign without its
        # old self-heal entrypoint.
        _replace_file(backup, LAUNCH_AGENT)
        try:
            _reload_launch_agent()
        except BaseException:
            pass
        raise


def _restore_transaction(packet: dict[str, Any], backup: Path,
                         moved_results: list[dict[str, str]],
                         *, restore_launch_agent: bool,
                         strict_result_absence: bool) -> None:
    # Never erase evidence produced after promotion.  Recovery is automatic
    # only before any promoted pending cell has emitted a result directory.
    conflicts = ([row["cell_id"] for row in packet["pending_runtime_specs"]
                  if (RESULTS / row["cell_id"]).exists()]
                 if strict_result_absence else [])
    if conflicts:
        raise ReentryError("rollback refused because promoted result paths now exist: "
                           + ", ".join(conflicts))
    _replace_file(backup / "adapter_registry.json", REGISTRY)
    for row in packet["pending_runtime_specs"]:
        _replace_file(backup / f"spec-{row['cell_id']}.json", Path(row["target"]))
    for moved in reversed(moved_results):
        source, target = Path(moved["backup"]), Path(moved["target"])
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
    metadata = _read_json(backup / "transaction_metadata.json")
    _restore_optional(OVERLAY, backup / "overlay.json",
                      bool(metadata["overlay_existed"]))
    _restore_optional(JOURNAL, backup / "journal.json",
                      bool(metadata["journal_existed"]))
    _restore_optional(MARKER, backup / "marker.json",
                      bool(metadata["marker_existed"]))
    if restore_launch_agent:
        _replace_file(backup / "autoresume.plist", LAUNCH_AGENT)
        _reload_launch_agent()


def _wait_for_quiescence(timeout_seconds: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last: list[str] = []
    while time.monotonic() < deadline:
        ready, last = _quiescent()
        if ready:
            return
        time.sleep(0.5)
    raise ReentryError("timed out draining after failed activation: " + "; ".join(last))


def _start_accelerated(marker: dict[str, Any]) -> None:
    env = os.environ.copy()
    env[stacked.ENV_OVERLAY] = marker["overlay_path"]
    env[stacked.ENV_OVERLAY_SHA256] = marker["overlay_sha256"]
    process = subprocess.run([sys.executable, str(ACCEL_QUEUE), "start"], cwd=ROOT,
                             env=env, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise ReentryError("accelerated supervisor start failed: "
                           + (process.stderr or process.stdout).strip())


def _verify_accelerated_owner() -> int:
    plan, _, _ = _load_live()
    record = _read_json(queue.PID_FILE)
    if record.get("schema") != queue.PID_SCHEMA or record.get("version") != queue.VERSION \
            or record.get("plan_sha256") != plan["plan_sha256"] \
            or record.get("pid_record_sha256") != queue._hash_value(
                queue._without(record, "pid_record_sha256")
            ):
        raise ReentryError("accelerated supervisor ownership record is invalid")
    identity = queue._process_identity(record.get("pid"))
    if identity is None:
        raise ReentryError("accelerated supervisor process is absent")
    command, started = identity
    if started != record.get("process_started") \
            or "doctor_v5_ultra_accelerated_queue.py run" not in command \
            or hashlib.sha256(command.encode()).hexdigest() \
            != record.get("process_command_sha256"):
        raise ReentryError("detached owner is not the accelerated supervisor")
    return int(record["pid"])


def activate() -> dict[str, Any]:
    ready, blockers = _quiescent()
    if not ready:
        raise ReentryError("activation requires a quiescent checkpoint: " + "; ".join(blockers))
    packet = _read_json(PACKET)
    errors = validate_packet(packet)
    if errors:
        raise ReentryError("activation packet is invalid: " + "; ".join(errors))
    if MARKER.exists():
        raise ReentryError("an acceleration marker already exists")
    generation = packet["generation_id"]
    backup = (GENERATION_ROOT / generation / "rollback" /
              f"{int(time.time_ns())}-{secrets.token_hex(8)}")
    backup.mkdir(parents=True, exist_ok=False)
    queue_lease, heavy_lease = _acquire_both()
    moved_results: list[dict[str, str]] = []
    transaction_backed_up = False
    try:
        shutil.copy2(REGISTRY, backup / "adapter_registry.json")
        shutil.copy2(LAUNCH_AGENT, backup / "autoresume.plist")
        metadata = {
            "overlay_existed": OVERLAY.is_file(),
            "journal_existed": JOURNAL.is_file(),
            "marker_existed": MARKER.is_file(),
        }
        for target, destination in (
            (OVERLAY, backup / "overlay.json"),
            (JOURNAL, backup / "journal.json"),
            (MARKER, backup / "marker.json"),
        ):
            if target.is_file():
                shutil.copy2(target, destination)
        _atomic_json(backup / "transaction_metadata.json", metadata)
        transaction_backed_up = True
        for row in packet["pending_runtime_specs"]:
            target = Path(row["target"])
            shutil.copy2(target, backup / f"spec-{row['cell_id']}.json")
            result = RESULTS / row["cell_id"]
            if result.exists():
                destination = backup / "results" / row["cell_id"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(result, destination)
                moved_results.append({"cell_id": row["cell_id"],
                                      "backup": str(destination), "target": str(result)})
        _replace_file(Path(packet["registry"]["staged"]["path"]), REGISTRY)
        for row in packet["pending_runtime_specs"]:
            _replace_file(Path(row["staged"]["path"]), Path(row["target"]))
        # Rebuild against the exact drained state after the transaction.
        overlay = stacked.build_overlay()
        overlay_errors = stacked.validate_overlay(overlay)
        if overlay_errors:
            raise ReentryError("fresh admission overlay is invalid: " + "; ".join(overlay_errors))
        _atomic_json(OVERLAY, overlay)
        journal = {
            "schema": JOURNAL_SCHEMA, "generation_id": generation,
            "activated_at": _now(), "packet_sha256": packet["packet_sha256"],
            "backup_root": str(backup.resolve()), "moved_results": moved_results,
            "launch_agent_backup": str((backup / "autoresume.plist").resolve()),
            "status": "files-promoted", "source_deletion_permitted": False,
        }
        journal["journal_sha256"] = _hash_value(journal)
        _atomic_json(JOURNAL, journal)
        marker = _write_marker(packet)
        _configure_launch_agent(backup / "autoresume.plist")
    except BaseException as exc:
        # Files are restored before leases are released; the old supervisor
        # therefore cannot observe a partially promoted runtime generation.
        if transaction_backed_up:
            try:
                _restore_transaction(
                    packet, backup, moved_results,
                    restore_launch_agent=(backup / "autoresume.plist").is_file(),
                    strict_result_absence=False,
                )
            except BaseException as rollback_exc:
                raise ReentryError(f"activation failed ({exc}); rollback also failed: "
                                   f"{rollback_exc}") from rollback_exc
        raise
    finally:
        heavy_lease.close(); queue_lease.close()
    try:
        _start_accelerated(marker)
        supervisor_pid = _verify_accelerated_owner()
        journal = _read_json(JOURNAL)
        journal["status"] = "active"
        journal["supervisor_started_at"] = _now()
        journal["supervisor_pid"] = supervisor_pid
        journal["journal_sha256"] = _hash_value(_without(journal, "journal_sha256"))
        _atomic_json(JOURNAL, journal)
    except BaseException as exc:
        # start_queue may have changed control to run before its ownership
        # handshake failed. Drain, reacquire both leases, and roll back only if
        # no promoted result path has appeared.
        queue.set_control("drain")
        _wait_for_quiescence()
        queue_lease, heavy_lease = _acquire_both()
        try:
            _restore_transaction(packet, backup, moved_results,
                                 restore_launch_agent=True,
                                 strict_result_absence=True)
        except BaseException as rollback_exc:
            raise ReentryError(f"accelerated start failed ({exc}); rollback also failed: "
                               f"{rollback_exc}") from rollback_exc
        finally:
            heavy_lease.close(); queue_lease.close()
        raise
    return {"status": "activated", "generation_id": generation,
            "packet_sha256": packet["packet_sha256"],
            "overlay_sha256": marker["overlay_sha256"],
            "supervisor_pid": supervisor_pid,
            "pending_specs_promoted": len(packet["pending_runtime_specs"]),
            "moved_pending_result_dirs": len(moved_results)}


def rollback() -> dict[str, Any]:
    ready, blockers = _quiescent()
    if not ready:
        raise ReentryError("rollback requires a quiescent checkpoint: "
                           + "; ".join(blockers))
    journal = _read_json(JOURNAL)
    if journal.get("schema") != JOURNAL_SCHEMA \
            or journal.get("journal_sha256") != _hash_value(
                _without(journal, "journal_sha256")
            ):
        raise ReentryError("activation journal schema/hash is invalid")
    packet = _read_json(PACKET)
    if journal.get("packet_sha256") != packet.get("packet_sha256"):
        raise ReentryError("rollback journal/packet binding differs")
    backup = Path(journal["backup_root"])
    moved_results = journal.get("moved_results")
    if not isinstance(moved_results, list):
        raise ReentryError("rollback result inventory is invalid")
    queue_lease, heavy_lease = _acquire_both()
    try:
        _restore_transaction(packet, backup, moved_results,
                             restore_launch_agent=True,
                             strict_result_absence=True)
    finally:
        heavy_lease.close(); queue_lease.close()
    return {"status": "rolled-back", "generation_id": packet["generation_id"],
            "restored_pending_specs": len(packet["pending_runtime_specs"]),
            "source_deletion_permitted": False}


def status() -> dict[str, Any]:
    plan, state, campaign = _load_live()
    parity = _parity_status()
    quiescent, quiescent_blockers = _quiescent()
    packet_errors: list[str] | None = None
    if PACKET.is_file():
        try:
            packet_errors = validate_packet(_read_json(PACKET))
        except ReentryError as exc:
            packet_errors = [str(exc)]
    active = state.get("active_children", {})
    return {
        "schema": "hawking.doctor_v5_acceleration_reentry_status.v1",
        "generated_at": _now(), "plan_sha256": plan["plan_sha256"],
        "counts": campaign.get("counts"), "active_cells": sorted(active),
        "parity": parity, "quiescent": quiescent,
        "quiescent_blockers": quiescent_blockers,
        "source_seals_present": sorted(path.stem for path in source_seal.SEAL_ROOT.glob("*.json"))
            if source_seal.SEAL_ROOT.is_dir() else [],
        "packet_present": PACKET.is_file(), "packet_errors": packet_errors,
        "active_marker_present": MARKER.is_file(),
        "activation_permitted_now": quiescent and parity["ok"] \
            and packet_errors == [],
        "source_deletion_permitted": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--packet", type=Path, default=PACKET)
    sub.add_parser("audit")
    sub.add_parser("activate")
    sub.add_parser("rollback")
    sub.add_parser("adversarial-audit")
    args = parser.parse_args(argv)
    if args.command == "status":
        result = status()
    elif args.command == "prepare":
        result = prepare(packet_path=args.packet)
    elif args.command == "audit":
        if not PACKET.is_file():
            raise ReentryError("pending runtime packet is absent")
        errors = validate_packet(_read_json(PACKET))
        result = {"ok": not errors, "errors": errors}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not errors else 2
    elif args.command == "activate":
        result = activate()
    elif args.command == "adversarial-audit":
        result = adversarial_audit()
    else:
        result = rollback()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReentryError, source_seal.SourceSealError,
            accel_loader.AccelerationBindingError) as exc:
        print(f"doctor_v5_acceleration_reentry: {exc}", file=sys.stderr)
        raise SystemExit(2)
