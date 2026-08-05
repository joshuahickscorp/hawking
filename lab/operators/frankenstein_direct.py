#!/usr/bin/env python3.12
"""Ceremony-free streaming fusion harness for the base Frankenstein.

Direct pipeline (no Ramanujan launcher, no Ed25519 owner green-light, no GPU
lease ceremony).  Provenance is still recorded as engineering hygiene.

Lifecycle for every schedule block:
  disk_floor_check → stream donor window → (fit | DEEPSEEK_FORWARD_PENDING)
  → seal raw output block → evict donor window → advance cursor

The student DeepSeek forward is intentionally stubbed behind
``DEEPSEEK_FORWARD_PENDING`` until the separate runtime lane lands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import struct
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators.frankenstein_fusion_op import (
    BRIDGES,
    DEEPSEEK_V4_FLASH,
    FORWARD_GATE,
    GLM_5_2,
    KIMI_K3,
    TRANSPLANT_POINT_NAMES,
    block_id,
    donor_for_bridge,
    estimated_adapter_archive_bytes,
    fusion_operation_spec,
    layer_map,
    loss_target,
    projection_shape,
    residual_adapter_shape,
)
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "workspace"
CAMPAIGN_ROOT = WORKSPACE_ROOT / "campaign"
EVIDENCE_ROOT = CAMPAIGN_ROOT / "evidence"
RUN_ROOT = CAMPAIGN_ROOT / "records" / "runs" / "frankenstein"
FRANK_EVIDENCE = EVIDENCE_ROOT / "models" / "frankenstein"

MIN_FREE_FLOOR_BYTES = 15 * 1024**3
# Default donor working-set ceiling (one window).  Real GLM 90GB schedule peaks
# near ~22 GiB incremental; we budget 32 GiB as a hard refuse line for this
# harness so the floor cannot be crossed by a runaway window.
DEFAULT_DONOR_WINDOW_BUDGET_BYTES = 32 * 1024**3
DEFAULT_OUTPUT_BLOCK_BUDGET_BYTES = 2 * 1024**3
DEFAULT_SCRATCH_BUDGET_BYTES = 4 * 1024**3

# Durable DeepSeek body (read-only; never rewritten by this harness).
DEFAULT_BODY_PATH = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/full-43-layer-stream.gravity"
)
DEFAULT_BRIDGE_PATH = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_LATENT_BRIDGE_CONTRACT.json"
)
DEFAULT_TRANSPLANT_PATH = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_TRANSPLANT_POINTS.json"
)
DEFAULT_KIMI_ADMISSION = (
    EVIDENCE_ROOT / "models" / "kimi-k3" / "KIMI_K3_SOURCE_ADMISSION.json"
)
DEFAULT_GLM_SCHEDULE = (
    EVIDENCE_ROOT / "models" / "glm52" / "GLM52_STREAMING_SCHEDULE_90GB.json"
)

SCHEDULE_SCHEMA = "hawking.frankenstein.direct_streaming_schedule.v1"
PROGRESS_SCHEMA = "hawking.frankenstein.direct_progress.v1"
BLOCK_RECEIPT_SCHEMA = "hawking.frankenstein.direct_block_receipt.v1"
FUSION_SPEC_SCHEMA = "hawking.frankenstein.fusion_operation_spec.v1"
RUN_MANIFEST_SCHEMA = "hawking.frankenstein.direct_run_manifest.v1"
FIXTURE_WINDOW_MAGIC = b"FRNKFIX1"

PLAN_NAME = "FRANKENSTEIN_DIRECT_STREAMING_SCHEDULE.json"
FUSION_SPEC_NAME = "FRANKENSTEIN_FUSION_OPERATION.json"
PROGRESS_NAME = "FRANKENSTEIN_DIRECT_PROGRESS.jsonl"
RUN_MANIFEST_NAME = "FRANKENSTEIN_DIRECT_RUN_MANIFEST.json"


def schedule_filename(mode: str) -> str:
    """Mode-qualified schedule name so fixture/pilot/full do not clobber each other."""

    if mode == "fixture":
        return "FRANKENSTEIN_DIRECT_STREAMING_SCHEDULE_FIXTURE.json"
    if mode == "pilot":
        return "FRANKENSTEIN_DIRECT_STREAMING_SCHEDULE_PILOT.json"
    if mode == "full":
        return "FRANKENSTEIN_DIRECT_STREAMING_SCHEDULE_FULL.json"
    return PLAN_NAME


class FrankensteinDirectError(RuntimeError):
    """Harness input, disk, or lifecycle gate failed closed."""


@dataclass(frozen=True)
class HarnessPaths:
    workspace: Path
    out_dir: Path
    schedule_path: Path
    progress_path: Path
    fusion_spec_path: Path
    run_manifest_path: Path
    # Trackable sealed JSON copies (records/runs/ is gitignored by the repo-wide
    # ``runs/`` pattern).  Working-set + large raw adapter blocks stay under runs/.
    evidence_receipts_dir: Path
    working_set_dir: Path
    output_archive_dir: Path
    body_path: Path
    bridge_path: Path
    transplant_path: Path
    kimi_admission_path: Path | None
    glm_schedule_path: Path | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise FrankensteinDirectError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(path)))


def _regular_file(path: Path, label: str) -> None:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise FrankensteinDirectError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise FrankensteinDirectError(f"{label} must be a regular non-symlink file")


def _ensure_dir(path: Path) -> None:
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
            raise FrankensteinDirectError(f"not a safe directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=False)


def _atomic_create(path: Path, raw: bytes) -> str:
    if path.exists():
        _regular_file(path, f"existing output {path}")
        existing = path.read_bytes()
        if existing != raw:
            raise FrankensteinDirectError(
                f"refusing to overwrite different immutable evidence: {path}"
            )
        return _sha256(existing)
    _ensure_dir(path.parent)
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
    return _sha256(raw)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    _ensure_dir(path.parent)
    encoded = _canonical(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "ab", closefd=True) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def free_bytes(path: Path) -> int:
    try:
        return int(shutil.disk_usage(path).free)
    except OSError as exc:
        raise FrankensteinDirectError(f"cannot measure free space at {path}: {exc}") from exc


def assert_floor(path: Path, *, need_extra: int = 0, label: str = "workspace") -> dict[str, Any]:
    free = free_bytes(path)
    required = MIN_FREE_FLOOR_BYTES + max(0, int(need_extra))
    if free < required:
        raise FrankensteinDirectError(
            f"{label} free-space floor violated: free={free} need>={required} "
            f"(floor={MIN_FREE_FLOOR_BYTES} + extra={need_extra})"
        )
    return {
        "path": str(path),
        "free_bytes": free,
        "floor_bytes": MIN_FREE_FLOOR_BYTES,
        "need_extra_bytes": int(need_extra),
        "required_bytes": required,
        "headroom_bytes": free - required,
        "status": "FLOOR_OK",
    }


def _bind_json_file(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrankensteinDirectError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise FrankensteinDirectError(f"{label} must be a JSON object")
    seal_sha = document.get("seal_sha256")
    seal_verified = False
    if isinstance(seal_sha, str):
        try:
            verify(document, label=label)
            seal_verified = True
        except SealIntegrityError as exc:
            raise FrankensteinDirectError(str(exc)) from exc
    return {
        "label": label,
        "path": str(path),
        "file_sha256": _sha256(raw),
        "bytes": len(raw),
        "schema": document.get("schema"),
        "status": document.get("status"),
        "seal_sha256": seal_sha if isinstance(seal_sha, str) else None,
        "seal_verified": seal_verified,
        "document": document,
    }


def default_paths(
    *,
    workspace: str | Path = WORKSPACE_ROOT,
    out_dir: str | Path | None = None,
    body_path: str | Path = DEFAULT_BODY_PATH,
    bridge_path: str | Path = DEFAULT_BRIDGE_PATH,
    transplant_path: str | Path = DEFAULT_TRANSPLANT_PATH,
) -> HarnessPaths:
    ws = _absolute(workspace, "workspace")
    root = _absolute(out_dir, "out_dir") if out_dir is not None else (ws / "campaign")
    evidence = root / "evidence" / "models" / "frankenstein"
    runs = root / "records" / "runs" / "frankenstein"
    kimi = DEFAULT_KIMI_ADMISSION if DEFAULT_KIMI_ADMISSION.is_file() else None
    # Prefer worktree evidence; fall back to durable hawking evidence.
    glm_candidates = [
        DEFAULT_GLM_SCHEDULE,
        Path(
            "/Users/scammermike/Downloads/hawking/workspace/campaign/evidence/"
            "models/glm52/GLM52_STREAMING_SCHEDULE_90GB.json"
        ),
    ]
    glm = next((p for p in glm_candidates if p.is_file()), None)
    return HarnessPaths(
        workspace=ws,
        out_dir=root,
        # Default schedule path points at pilot; write_schedule overrides by mode.
        schedule_path=evidence / schedule_filename("pilot"),
        progress_path=evidence / "progress" / PROGRESS_NAME,
        fusion_spec_path=evidence / FUSION_SPEC_NAME,
        run_manifest_path=evidence / RUN_MANIFEST_NAME,
        evidence_receipts_dir=evidence / "receipts",
        working_set_dir=runs / "working-set",
        output_archive_dir=runs / "raw-adapter-archive",
        body_path=_absolute(body_path, "body_path"),
        bridge_path=_absolute(bridge_path, "bridge_path"),
        transplant_path=_absolute(transplant_path, "transplant_path"),
        kimi_admission_path=kimi,
        glm_schedule_path=glm,
    )


def _body_binding(body_path: Path) -> dict[str, Any]:
    if not body_path.exists():
        return {
            "path": str(body_path),
            "present": False,
            "read_only_contract": True,
            "status": "BODY_PATH_ABSENT_REFERENCE_ONLY",
            "note": (
                "DeepSeek body is not in this worktree; harness will not rewrite it. "
                "Schedule still budgets as if the durable 149 GiB body is retained."
            ),
            "budgeted_bytes": 149 * 1024**3,
        }
    if body_path.is_symlink():
        raise FrankensteinDirectError("DeepSeek body path must not be a symlink")
    if body_path.is_dir():
        size = 0
        for dirpath, _dirnames, filenames in os.walk(body_path):
            for name in filenames:
                fp = Path(dirpath) / name
                try:
                    if fp.is_symlink():
                        continue
                    size += fp.stat().st_size
                except OSError:
                    continue
        manifest = body_path / "manifest.json"
        manifest_binding = None
        if manifest.is_file() and not manifest.is_symlink():
            try:
                manifest_binding = _bind_json_file(manifest, "deepseek full manifest")
                # Drop heavy document from schedule binding.
                manifest_binding = {
                    k: v for k, v in manifest_binding.items() if k != "document"
                }
            except FrankensteinDirectError:
                manifest_binding = {
                    "path": str(manifest),
                    "file_sha256": _sha256_file(manifest),
                }
        return {
            "path": str(body_path),
            "present": True,
            "read_only_contract": True,
            "kind": "directory",
            "measured_bytes": size,
            "budgeted_bytes": size,
            "manifest": manifest_binding,
            "status": "BODY_PRESENT_READ_ONLY",
        }
    _regular_file(body_path, "deepseek body")
    size = body_path.stat().st_size
    return {
        "path": str(body_path),
        "present": True,
        "read_only_contract": True,
        "kind": "file",
        "measured_bytes": size,
        "budgeted_bytes": size,
        "file_sha256": _sha256_file(body_path),
        "status": "BODY_PRESENT_READ_ONLY",
    }


def _glm_window_budget(glm_schedule_path: Path | None) -> dict[str, Any]:
    if glm_schedule_path is None or not glm_schedule_path.is_file():
        return {
            "source": "default_harness_ceiling",
            "peak_resident_incremental_bytes": 22 * 1024**3,
            "window_budget_bytes": DEFAULT_DONOR_WINDOW_BUDGET_BYTES,
            "schedule_bound": False,
        }
    binding = _bind_json_file(glm_schedule_path, "glm streaming schedule")
    doc = binding["document"]
    peak = 0
    windows = doc.get("windows")
    if isinstance(windows, list):
        for window in windows:
            if not isinstance(window, Mapping):
                continue
            accounting = window.get("incremental_accounting")
            if not isinstance(accounting, Mapping):
                continue
            resident = accounting.get("resident_incremental_bytes")
            if isinstance(resident, int):
                peak = max(peak, resident)
    envelope = None
    contract = doc.get("incremental_accounting_contract")
    if isinstance(contract, Mapping):
        envelope = contract.get("envelope_bytes")
        peak_inc = contract.get("peak_incremental_bytes")
        if isinstance(peak_inc, int):
            peak = max(peak, peak_inc)
    budget = max(peak, DEFAULT_DONOR_WINDOW_BUDGET_BYTES // 2)
    budget = min(budget, DEFAULT_DONOR_WINDOW_BUDGET_BYTES)
    return {
        "source": str(glm_schedule_path),
        "file_sha256": binding["file_sha256"],
        "seal_sha256": binding["seal_sha256"],
        "peak_resident_incremental_bytes": peak,
        "envelope_bytes": envelope,
        "window_budget_bytes": budget,
        "schedule_bound": True,
        "repo": doc.get("repo"),
        "revision": doc.get("revision"),
        "window_count": (doc.get("partition") or {}).get("window_count")
        if isinstance(doc.get("partition"), Mapping)
        else None,
    }


def build_schedule_rows(
    *,
    bridges: Sequence[str] = BRIDGES,
    points: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    mode: str = "full",
) -> list[dict[str, Any]]:
    """Build a resumable ordered block schedule.

    mode:
      - full: all bridges × points × layers (large; for planning)
      - pilot: one point × first student layer per bridge (default for first step)
      - fixture: single synthetic block for disk-safe harness proof
    """

    if mode == "fixture":
        return [
            {
                "order": 0,
                "block_id": "FIXTURE__post_moe_hidden_state__L00",
                "bridge": "GLM_MATH_BRIDGE",
                "transplant_point": "post_moe_hidden_state",
                "student_layer": 0,
                "donor": "glm_5_2",
                "donor_layer": 0,
                "kind": "fixture",
                "fit_gate": FORWARD_GATE,
            }
        ]

    selected_points = list(points) if points is not None else list(TRANSPLANT_POINT_NAMES)
    if mode == "pilot":
        selected_points = ["post_moe_hidden_state"]
        selected_layers = [0]
    else:
        selected_layers = (
            list(layers)
            if layers is not None
            else list(range(DEEPSEEK_V4_FLASH["num_hidden_layers"]))
        )

    rows: list[dict[str, Any]] = []
    order = 0
    for bridge in bridges:
        if bridge not in BRIDGES:
            raise FrankensteinDirectError(f"unknown bridge: {bridge}")
        donor = donor_for_bridge(bridge)
        for point in selected_points:
            if point not in TRANSPLANT_POINT_NAMES:
                raise FrankensteinDirectError(f"unknown transplant point: {point}")
            for layer in selected_layers:
                mapped = layer_map(donor=donor, student_layer=layer)
                rows.append(
                    {
                        "order": order,
                        "block_id": block_id(
                            bridge=bridge, transplant_point=point, student_layer=layer
                        ),
                        "bridge": bridge,
                        "transplant_point": point,
                        "student_layer": layer,
                        "donor": donor,
                        "donor_layer": mapped["donor_layer"],
                        "kind": "live_or_pending",
                        "fit_gate": FORWARD_GATE,
                        "loss": loss_target(transplant_point=point),
                    }
                )
                order += 1
    return rows


def compute_disk_budget(
    *,
    free_now: int,
    body_budget_bytes: int,
    donor_window_budget: int,
    output_block_budget: int = DEFAULT_OUTPUT_BLOCK_BUDGET_BYTES,
    scratch_budget: int = DEFAULT_SCRATCH_BUDGET_BYTES,
    archive_upper_bound: int,
    body_already_on_volume: bool,
) -> dict[str, Any]:
    """Prove the 15 GiB floor is preserved under the working-set invariant.

    Working set = (body if counted) + one donor window + one output block + scratch.
    Body is already on disk when present; free_now already reflects it.
    """

    working_set = donor_window_budget + output_block_budget + scratch_budget
    # free after allocating working set (body already deducted from free_now if present)
    free_after_ws = free_now - working_set
    # archive grows; worst case we write the entire upper bound while holding WS
    free_after_archive_and_ws = free_after_ws - archive_upper_bound
    floor_ok_ws = free_after_ws >= MIN_FREE_FLOOR_BYTES
    floor_ok_full = free_after_archive_and_ws >= MIN_FREE_FLOOR_BYTES
    # Streaming strategy if full archive does not fit: write then externalize/evict
    # completed blocks to free space (harness refuses before crossing floor).
    max_archive_while_holding_ws = max(0, free_now - working_set - MIN_FREE_FLOOR_BYTES)
    return {
        "free_bytes_now": free_now,
        "floor_bytes": MIN_FREE_FLOOR_BYTES,
        "body_budget_bytes": body_budget_bytes,
        "body_already_on_volume": body_already_on_volume,
        "working_set": {
            "donor_window_budget_bytes": donor_window_budget,
            "output_block_budget_bytes": output_block_budget,
            "scratch_budget_bytes": scratch_budget,
            "total_bytes": working_set,
            "invariant": (
                "DeepSeek body (read-only, already on volume) + at most ONE donor "
                "window + current output block + scratch"
            ),
        },
        "archive_upper_bound_bytes": archive_upper_bound,
        "max_archive_bytes_while_holding_working_set": max_archive_while_holding_ws,
        "free_after_working_set_bytes": free_after_ws,
        "free_after_full_archive_and_working_set_bytes": free_after_archive_and_ws,
        "floor_preserved_with_working_set": floor_ok_ws,
        "floor_preserved_with_full_archive_resident": floor_ok_full,
        "policy_if_archive_exceeds_budget": (
            "refuse next block before write if free - (ws + next_block) < floor; "
            "operator must externalize sealed blocks off-volume or shrink scope"
        ),
        "verdict": (
            "SAFE_UNDER_WORKING_SET"
            if floor_ok_ws
            else "UNSAFE_WORKING_SET_WOULD_BREACH_FLOOR"
        ),
    }


def build_schedule(*, paths: HarnessPaths, mode: str = "pilot") -> dict[str, Any]:
    floor = assert_floor(paths.workspace, label="workspace")
    body = _body_binding(paths.body_path)
    bridge = _bind_json_file(paths.bridge_path, "latent bridge contract")
    transplant = _bind_json_file(paths.transplant_path, "transplant points")
    # Drop bulky documents from sealed schedule; keep hashes.
    bridge_meta = {k: v for k, v in bridge.items() if k != "document"}
    transplant_meta = {k: v for k, v in transplant.items() if k != "document"}

    kimi_meta = None
    if paths.kimi_admission_path is not None and paths.kimi_admission_path.is_file():
        kimi_bind = _bind_json_file(paths.kimi_admission_path, "kimi k3 admission")
        kimi_meta = {k: v for k, v in kimi_bind.items() if k != "document"}

    glm_budget = _glm_window_budget(paths.glm_schedule_path)
    archive = estimated_adapter_archive_bytes(
        layers=1 if mode in {"pilot", "fixture"} else None,
        points=("post_moe_hidden_state",)
        if mode in {"pilot", "fixture"}
        else TRANSPLANT_POINT_NAMES,
    )
    if mode == "fixture":
        archive = {
            **archive,
            "block_count": 1,
            "adapters_total_bytes": residual_adapter_shape()["bytes_bf16"],
            "archive_upper_bound_bytes": residual_adapter_shape()["bytes_bf16"]
            + 1024 * 1024,
        }

    rows = build_schedule_rows(mode=mode)
    free_now = int(floor["free_bytes"])
    disk = compute_disk_budget(
        free_now=free_now,
        body_budget_bytes=int(body.get("budgeted_bytes") or 0),
        donor_window_budget=int(glm_budget["window_budget_bytes"]),
        archive_upper_bound=int(archive["archive_upper_bound_bytes"]),
        body_already_on_volume=bool(body.get("present")),
    )
    if disk["verdict"] != "SAFE_UNDER_WORKING_SET":
        raise FrankensteinDirectError(
            "disk budget refuses schedule: working set would breach 15 GiB floor"
        )

    # Seal only structural disk fields (not live free_bytes) so the schedule is
    # deterministic and byte-identical retries stay admissible.
    disk_sealed = {
        "floor_bytes": disk["floor_bytes"],
        "body_budget_bytes": disk["body_budget_bytes"],
        "body_already_on_volume": disk["body_already_on_volume"],
        "working_set": disk["working_set"],
        "archive_upper_bound_bytes": disk["archive_upper_bound_bytes"],
        "max_archive_bytes_while_holding_working_set_formula": (
            "free_now - working_set.total_bytes - floor_bytes"
        ),
        "required_free_for_working_set_bytes": (
            disk["working_set"]["total_bytes"] + disk["floor_bytes"]
        ),
        "floor_preserved_with_working_set_at_freeze": disk[
            "floor_preserved_with_working_set"
        ],
        "floor_preserved_with_full_archive_resident_at_freeze": disk[
            "floor_preserved_with_full_archive_resident"
        ],
        "policy_if_archive_exceeds_budget": disk["policy_if_archive_exceeds_budget"],
        "verdict": disk["verdict"],
        "proof": {
            "statement": (
                "At freeze, free_bytes_now >= working_set + floor.  Live free is "
                "re-checked before every block and is not part of this seal."
            ),
            "free_bytes_now_not_sealed": True,
            "working_set_bytes": disk["working_set"]["total_bytes"],
            "floor_bytes": disk["floor_bytes"],
        },
    }

    fusion = fusion_operation_spec()
    plan = {
        "schema": SCHEDULE_SCHEMA,
        "status": "DIRECT_SCHEDULE_FROZEN_CEREMONY_FREE",
        "mode": mode,
        "ceremony": {
            "ramanujan_launcher": False,
            "ed25519_owner_green_light": False,
            "gpu_lease_receipt": False,
            "d5_d8_d9_freeze": False,
            "note": (
                "Owner waived Ramanujan governance ceremony for this direct build. "
                "No forged audit records. Provenance is recorded as engineering hygiene."
            ),
        },
        "student_body": {
            **{k: DEEPSEEK_V4_FLASH[k] for k in (
                "repository", "revision", "hidden_size", "num_hidden_layers",
                "n_routed_experts", "num_experts_per_tok", "vocab_size",
                "source_torch_dtype",
            )},
            "artifact": body,
        },
        "donors": {
            "kimi_k3": {
                "repository": KIMI_K3["repository"],
                "revision": KIMI_K3["revision"],
                "hidden_size": KIMI_K3["hidden_size"],
                "num_hidden_layers": KIMI_K3["num_hidden_layers"],
                "weight_shard_bytes": KIMI_K3["weight_shard_bytes"],
                "admission": kimi_meta,
            },
            "glm_5_2": {
                "repository": GLM_5_2["repository"],
                "revision": GLM_5_2["revision"],
                "hidden_size": GLM_5_2["hidden_size"],
                "num_hidden_layers": GLM_5_2["num_hidden_layers"],
                "window_budget": glm_budget,
            },
        },
        "bindings": {
            "latent_bridge": bridge_meta,
            "transplant_points": transplant_meta,
        },
        "fusion_operation": {
            "name": fusion["name"],
            "verdict": fusion["verdict"],
            "impossible_names": [row["name"] for row in fusion["impossible"]],
            "forward_gate": FORWARD_GATE,
            "projections": fusion["projections"],
            "residual_adapter": fusion["residual_adapter"],
        },
        "disk": disk_sealed,
        "archive_estimate": archive,
        "storage_contract": {
            "hard_free_floor_bytes": MIN_FREE_FLOOR_BYTES,
            "max_simultaneous_donor_models": 1,
            "deepseek_body_eviction_authorized": False,
            "gravity_compress_output": False,
            "source_window_lifecycle": [
                "fetch_or_fixture_materialize",
                "verify_identity",
                "fit_or_forward_pending",
                "seal_raw_block",
                "evict_donor_window",
            ],
        },
        "blocks": rows,
        "block_count": len(rows),
        "resume": {
            "cursor_field": "order",
            # Relative to campaign root so the sealed schedule is host-stable.
            "progress_path": "evidence/models/frankenstein/progress/FRANKENSTEIN_DIRECT_PROGRESS.jsonl",
            "completed_block_ids": [],
        },
    }
    return seal(plan)


def write_fusion_spec(*, paths: HarnessPaths) -> dict[str, Any]:
    # Live floor check refuses early; the sealed document is deterministic so
    # byte-identical retries are accepted by _atomic_create.
    assert_floor(paths.workspace, label="workspace")
    spec = {
        "schema": FUSION_SPEC_SCHEMA,
        "status": "FUSION_OPERATION_SPEC_SEALED",
        "floor_policy_bytes": MIN_FREE_FLOOR_BYTES,
        "operation": fusion_operation_spec(),
        "adapter_archive_estimate_full": estimated_adapter_archive_bytes(),
        "adapter_archive_estimate_pilot": estimated_adapter_archive_bytes(
            layers=1, points=("post_moe_hidden_state",)
        ),
    }
    sealed = seal(spec)
    _atomic_create(paths.fusion_spec_path, _canonical(sealed) + b"\n")
    return sealed


def write_schedule(*, paths: HarnessPaths, mode: str = "pilot") -> dict[str, Any]:
    plan = build_schedule(paths=paths, mode=mode)
    schedule_path = paths.schedule_path.parent / schedule_filename(mode)
    _atomic_create(schedule_path, _canonical(plan) + b"\n")
    event = seal(
        {
            "schema": PROGRESS_SCHEMA,
            "recorded_at": _utc_now(),
            "event": "SCHEDULE_FROZEN",
            "status": plan["status"],
            "mode": mode,
            "schedule_path": str(schedule_path),
            "schedule_seal_sha256": plan["seal_sha256"],
            "block_count": plan["block_count"],
            "disk_verdict": plan["disk"]["verdict"],
            "free_bytes_at_event": free_bytes(paths.workspace),
        }
    )
    _append_jsonl(paths.progress_path, event)
    return plan


def _load_schedule(path: Path) -> dict[str, Any]:
    binding = _bind_json_file(path, "direct schedule")
    doc = binding["document"]
    if doc.get("schema") != SCHEDULE_SCHEMA:
        raise FrankensteinDirectError(
            f"schedule schema mismatch: {doc.get('schema')!r}"
        )
    return doc


def _fixture_payload(*, block: Mapping[str, Any], nbytes: int = 64 * 1024) -> bytes:
    """Deterministic synthetic donor window (not model weights)."""

    header = {
        "magic": FIXTURE_WINDOW_MAGIC.decode("ascii"),
        "block_id": block["block_id"],
        "bridge": block["bridge"],
        "donor": block["donor"],
        "donor_layer": block["donor_layer"],
        "transplant_point": block["transplant_point"],
        "student_layer": block["student_layer"],
        "kind": "FIXTURE_DONOR_WINDOW",
        "not_model_weights": True,
        "repository": GLM_5_2["repository"] if block["donor"] == "glm_5_2" else KIMI_K3["repository"],
        "revision": GLM_5_2["revision"] if block["donor"] == "glm_5_2" else KIMI_K3["revision"],
    }
    header_raw = _canonical(header)
    # Expand to nbytes with a deterministic stream derived from the header hash.
    seed = hashlib.sha256(header_raw).digest()
    body = bytearray()
    counter = 0
    while len(body) < nbytes:
        block_bytes = hashlib.sha256(seed + struct.pack(">Q", counter)).digest()
        body.extend(block_bytes)
        counter += 1
    body = bytes(body[:nbytes])
    return FIXTURE_WINDOW_MAGIC + struct.pack(">I", len(header_raw)) + header_raw + body


def _write_zero_adapter_block(path: Path, *, block: Mapping[str, Any]) -> dict[str, Any]:
    """Write a raw, un-gravity adapter block: header + zero-init dense residual.

    This is a *placeholder* fit output under DEEPSEEK_FORWARD_PENDING — not a
    trained graft.  Shape matches residual_adapter_shape(); bytes are real on disk.
    """

    adapter = residual_adapter_shape()
    weight_elems = adapter["parameter_count"]
    # bf16 zeros
    payload = b"\x00" * (weight_elems * 2)
    header = {
        "schema": "hawking.frankenstein.raw_adapter_block.v1",
        "block_id": block["block_id"],
        "bridge": block["bridge"],
        "transplant_point": block["transplant_point"],
        "student_layer": block["student_layer"],
        "donor": block["donor"],
        "donor_layer": block["donor_layer"],
        "weight_shape": adapter["weight_shape"],
        "bias_shape": adapter["bias_shape"],
        "dtype": adapter["dtype"],
        "parameter_count": adapter["parameter_count"],
        "payload_bytes": len(payload),
        "fit_status": FORWARD_GATE,
        "trained": False,
        "gravity_compressed": False,
        "direct_weight_transplant": False,
        "init": "zeros_residual_identity",
        "note": (
            "Zero-init residual adapter (identity at apply-time).  Real fit requires "
            "DeepSeek student forward + donor activations."
        ),
    }
    header_raw = _canonical(header)
    raw = b"FRNKADP1" + struct.pack(">I", len(header_raw)) + header_raw + payload
    _ensure_dir(path.parent)
    path.write_bytes(raw)
    # fsync
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {
        "path": str(path),
        "bytes": len(raw),
        "payload_bytes": len(payload),
        "header_sha256": _sha256(header_raw),
        "file_sha256": _sha256(raw),
        "fit_status": FORWARD_GATE,
        "trained": False,
        "gravity_compressed": False,
    }


def _rm_tree(path: Path) -> dict[str, Any]:
    """Evict a working-set directory tree; refuse symlinks."""

    if not path.exists():
        return {"path": str(path), "existed": False, "removed": False, "bytes_removed": 0}
    node = os.lstat(path)
    if stat.S_ISLNK(node.st_mode):
        raise FrankensteinDirectError(f"refusing to evict symlink working set: {path}")
    bytes_removed = 0
    if path.is_file():
        bytes_removed = path.stat().st_size
        path.unlink()
        return {
            "path": str(path),
            "existed": True,
            "removed": True,
            "bytes_removed": bytes_removed,
        }
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for name in filenames:
            fp = Path(dirpath) / name
            if fp.is_symlink():
                raise FrankensteinDirectError(f"refusing to evict symlink: {fp}")
            bytes_removed += fp.stat().st_size
            fp.unlink()
        for name in dirnames:
            dp = Path(dirpath) / name
            if dp.is_symlink():
                raise FrankensteinDirectError(f"refusing to evict symlink dir: {dp}")
            dp.rmdir()
    path.rmdir()
    return {
        "path": str(path),
        "existed": True,
        "removed": True,
        "bytes_removed": bytes_removed,
    }


def run_block(
    *,
    paths: HarnessPaths,
    schedule: Mapping[str, Any],
    order: int,
    fixture: bool = True,
    fixture_window_bytes: int = 64 * 1024,
) -> dict[str, Any]:
    """Execute one schedule block under the streaming/eviction discipline."""

    blocks = schedule.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise FrankensteinDirectError("schedule has no blocks")
    block = None
    for row in blocks:
        if isinstance(row, Mapping) and row.get("order") == order:
            block = row
            break
    if block is None:
        raise FrankensteinDirectError(f"no schedule block with order={order}")

    # Resume: accept an already-sealed receipt for this block_id (prefer trackable
    # evidence path, fall back to runs/ archive copy).
    receipt_candidates = [
        paths.evidence_receipts_dir / f"{block['block_id']}.receipt.json",
        paths.output_archive_dir / f"{block['block_id']}.receipt.json",
    ]
    for receipt_path in receipt_candidates:
        if not (receipt_path.is_file() and not receipt_path.is_symlink()):
            continue
        existing = _bind_json_file(receipt_path, "existing block receipt")
        doc = existing["document"]
        if (
            doc.get("schema") == BLOCK_RECEIPT_SCHEMA
            and doc.get("block", {}).get("block_id") == block["block_id"]
            and doc.get("eviction", {}).get("exact_eviction_confirmed") is True
            and existing["seal_verified"]
        ):
            evidence_receipt = (
                paths.evidence_receipts_dir / f"{block['block_id']}.receipt.json"
            )
            raw = receipt_path.read_bytes()
            _atomic_create(evidence_receipt, raw)
            adapter_path = paths.output_archive_dir / f"{block['block_id']}.raw_adapter"
            return {
                "status": doc.get("status"),
                "receipt_path": str(evidence_receipt),
                "receipt_seal_sha256": doc.get("seal_sha256"),
                "block_id": block["block_id"],
                "fit_status": (doc.get("fit") or {}).get("status", FORWARD_GATE),
                "eviction_confirmed": True,
                "bytes_streamed": (doc.get("stream") or {}).get("bytes_streamed", 0),
                "adapter_path": str(adapter_path),
                "free_bytes_after": free_bytes(paths.workspace),
                "resumed_existing_receipt": True,
            }

    floor_before = assert_floor(
        paths.workspace,
        need_extra=fixture_window_bytes + residual_adapter_shape()["bytes_bf16"],
        label="workspace-before-block",
    )
    _ensure_dir(paths.working_set_dir)
    _ensure_dir(paths.output_archive_dir)

    window_dir = paths.working_set_dir / f"donor-{block['block_id']}"
    if window_dir.exists():
        _rm_tree(window_dir)
    _ensure_dir(window_dir)

    # --- stream (fixture or refuse live without forward + explicit flag) ---
    if not fixture and block.get("kind") != "fixture":
        # Live donor stream is allowed only as a thin provenance-bound fetch of a
        # *bounded* window; full Kimi materialization is hard-refused.
        if block["donor"] == "kimi_k3":
            raise FrankensteinDirectError(
                "refusing live Kimi-K3 full-body stream; use bounded windows only "
                "and never materialize the 1.56 TB resident set"
            )
        raise FrankensteinDirectError(
            "live donor stream not enabled in this step; re-run with fixture "
            "mode for disk-safe harness proof, or extend the thin direct streamer"
        )

    window_path = window_dir / "window.bin"
    payload = _fixture_payload(block=block, nbytes=fixture_window_bytes)
    window_path.write_bytes(payload)
    fd = os.open(window_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    window_sha = _sha256(payload)
    stream_receipt = {
        "kind": "FIXTURE_DONOR_WINDOW",
        "path": str(window_path),
        "bytes_streamed": len(payload),
        "sha256": window_sha,
        "donor": block["donor"],
        "donor_repository": (
            GLM_5_2["repository"] if block["donor"] == "glm_5_2" else KIMI_K3["repository"]
        ),
        "donor_revision": (
            GLM_5_2["revision"] if block["donor"] == "glm_5_2" else KIMI_K3["revision"]
        ),
        "not_model_weights": True,
        "note": (
            "Fixture donor window for harness/lifecycle proof.  Not a download of "
            "official weight shards."
        ),
    }

    # --- fit (honest stub) ---
    fit_result = {
        "status": FORWARD_GATE,
        "trained": False,
        "reason": (
            "DeepSeek-V4 student forward is owned by a separate lane and is not "
            "available to this harness yet.  Projection + residual fit cannot "
            "measure student activations; writing zero-init reversible adapter "
            "placeholder only."
        ),
        "loss_target": loss_target(transplant_point=str(block["transplant_point"])),
        "projection": projection_shape(donor=str(block["donor"])),
    }

    out_path = paths.output_archive_dir / f"{block['block_id']}.raw_adapter"
    adapter_meta = _write_zero_adapter_block(out_path, block=block)

    # --- evict donor window ---
    free_before_evict = free_bytes(paths.workspace)
    eviction = _rm_tree(window_dir)
    free_after_evict = free_bytes(paths.workspace)
    eviction.update(
        {
            "free_bytes_before_eviction": free_before_evict,
            "free_bytes_after_eviction": free_after_evict,
            "source_window_retained": window_dir.exists(),
            "exact_eviction_confirmed": not window_dir.exists(),
        }
    )
    if window_dir.exists():
        raise FrankensteinDirectError("donor window eviction failed")

    floor_after = assert_floor(paths.workspace, label="workspace-after-block")

    receipt = seal(
        {
            "schema": BLOCK_RECEIPT_SCHEMA,
            "status": "BLOCK_SEALED_FIT_PENDING_FORWARD",
            "recorded_at": _utc_now(),
            "schedule_seal_sha256": schedule.get("seal_sha256"),
            "block": {
                "order": block["order"],
                "block_id": block["block_id"],
                "bridge": block["bridge"],
                "transplant_point": block["transplant_point"],
                "student_layer": block["student_layer"],
                "donor": block["donor"],
                "donor_layer": block["donor_layer"],
                "kind": block.get("kind"),
            },
            "stream": stream_receipt,
            "fit": fit_result,
            "adapter_block": adapter_meta,
            "eviction": eviction,
            "disk": {
                "before": floor_before,
                "after": floor_after,
                "floor_bytes": MIN_FREE_FLOOR_BYTES,
                "floor_preserved": True,
            },
            "claim_boundary": {
                "trained_adapter": False,
                "merged_model_file": False,
                "weight_average": False,
                "gravity_compressed": False,
                "donor_weights_retained": False,
                "deepseek_body_modified": False,
                "frankenstein_capability_claim": False,
            },
        }
    )
    receipt_raw = _canonical(receipt) + b"\n"
    evidence_receipt_path = (
        paths.evidence_receipts_dir / f"{block['block_id']}.receipt.json"
    )
    archive_receipt_path = (
        paths.output_archive_dir / f"{block['block_id']}.receipt.json"
    )
    _atomic_create(evidence_receipt_path, receipt_raw)
    _atomic_create(archive_receipt_path, receipt_raw)

    event = seal(
        {
            "schema": PROGRESS_SCHEMA,
            "recorded_at": _utc_now(),
            "event": "BLOCK_COMPLETED",
            "status": receipt["status"],
            "block_id": block["block_id"],
            "order": block["order"],
            "receipt_path": str(evidence_receipt_path),
            "receipt_seal_sha256": receipt["seal_sha256"],
            "fit_status": FORWARD_GATE,
            "bytes_streamed": stream_receipt["bytes_streamed"],
            "adapter_bytes": adapter_meta["bytes"],
            "eviction_confirmed": eviction["exact_eviction_confirmed"],
            "free_bytes_after": floor_after["free_bytes"],
        }
    )
    _append_jsonl(paths.progress_path, event)
    return {
        "status": receipt["status"],
        "receipt_path": str(evidence_receipt_path),
        "receipt_seal_sha256": receipt["seal_sha256"],
        "block_id": block["block_id"],
        "fit_status": FORWARD_GATE,
        "eviction_confirmed": eviction["exact_eviction_confirmed"],
        "bytes_streamed": stream_receipt["bytes_streamed"],
        "adapter_path": adapter_meta["path"],
        "free_bytes_after": floor_after["free_bytes"],
        "resumed_existing_receipt": False,
    }


def run_first_step(*, paths: HarnessPaths, mode: str = "fixture") -> dict[str, Any]:
    """Freeze fusion spec + schedule, execute block 0, write run manifest."""

    fusion = write_fusion_spec(paths=paths)
    schedule = write_schedule(paths=paths, mode=mode)
    schedule_path = paths.schedule_path.parent / schedule_filename(mode)
    result = run_block(paths=paths, schedule=schedule, order=0, fixture=True)
    # Manifest is mode-qualified so fixture/pilot first-steps do not collide.
    # Written under evidence/ (trackable); records/runs/ holds only working-set.
    run_manifest_path = (
        paths.fusion_spec_path.parent
        / f"FRANKENSTEIN_DIRECT_RUN_MANIFEST_{mode.upper()}.json"
    )
    manifest = seal(
        {
            "schema": RUN_MANIFEST_SCHEMA,
            "status": "DIRECT_HARNESS_FIRST_STEP_SEALED",
            "recorded_at": _utc_now(),
            "ceremony_free": True,
            "fusion_spec": {
                "path": str(paths.fusion_spec_path),
                "seal_sha256": fusion["seal_sha256"],
            },
            "schedule": {
                "path": str(schedule_path),
                "seal_sha256": schedule["seal_sha256"],
                "mode": mode,
                "block_count": schedule["block_count"],
            },
            "first_block": result,
            "disk": schedule["disk"],
            "forward_gate": FORWARD_GATE,
            "claim_boundary": {
                "raw_base_frankenstein_model_complete": False,
                "weight_average_performed": False,
                "gravity_compression_performed": False,
                "trained_inheritance": False,
                "harness_and_schedule_and_first_block": True,
            },
        }
    )
    _atomic_create(run_manifest_path, _canonical(manifest) + b"\n")
    return {
        "status": manifest["status"],
        "fusion_spec_path": str(paths.fusion_spec_path),
        "fusion_spec_seal_sha256": fusion["seal_sha256"],
        "schedule_path": str(schedule_path),
        "schedule_seal_sha256": schedule["seal_sha256"],
        "run_manifest_path": str(run_manifest_path),
        "run_manifest_seal_sha256": manifest["seal_sha256"],
        "first_block": result,
        "disk_verdict": schedule["disk"]["verdict"],
        "forward_gate": FORWARD_GATE,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=WORKSPACE_ROOT,
        help="workspace root used for free-space floor checks",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="campaign root for evidence/records (default: <workspace>/campaign)",
    )
    parser.add_argument(
        "--body-path",
        type=Path,
        default=DEFAULT_BODY_PATH,
        help="read-only DeepSeek body path",
    )
    parser.add_argument(
        "--bridge-path",
        type=Path,
        default=DEFAULT_BRIDGE_PATH,
    )
    parser.add_argument(
        "--transplant-path",
        type=Path,
        default=DEFAULT_TRANSPLANT_PATH,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("fusion-spec", help="seal the concrete fusion operation spec")
    sched = commands.add_parser("schedule", help="freeze the resumable streaming schedule")
    sched.add_argument(
        "--mode",
        choices=("fixture", "pilot", "full"),
        default="pilot",
    )
    first = commands.add_parser(
        "first-step",
        help="fusion-spec + schedule + execute block 0 (fixture donor window)",
    )
    first.add_argument(
        "--mode",
        choices=("fixture", "pilot"),
        default="fixture",
    )
    blk = commands.add_parser("run-block", help="run one schedule block by order")
    blk.add_argument("--order", type=int, default=0)
    blk.add_argument("--schedule", type=Path, default=None)
    blk.add_argument("--fixture", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = default_paths(
        workspace=args.workspace,
        out_dir=args.out_dir,
        body_path=args.body_path,
        bridge_path=args.bridge_path,
        transplant_path=args.transplant_path,
    )
    try:
        if args.command == "fusion-spec":
            result = {
                "status": "FUSION_OPERATION_SPEC_SEALED",
                "path": str(paths.fusion_spec_path),
                "seal_sha256": write_fusion_spec(paths=paths)["seal_sha256"],
            }
        elif args.command == "schedule":
            plan = write_schedule(paths=paths, mode=args.mode)
            schedule_path = paths.schedule_path.parent / schedule_filename(args.mode)
            result = {
                "status": plan["status"],
                "path": str(schedule_path),
                "seal_sha256": plan["seal_sha256"],
                "block_count": plan["block_count"],
                "disk_verdict": plan["disk"]["verdict"],
            }
        elif args.command == "first-step":
            result = run_first_step(paths=paths, mode=args.mode)
        elif args.command == "run-block":
            schedule_path = (
                _absolute(args.schedule, "schedule")
                if args.schedule is not None
                else paths.schedule_path
            )
            schedule = _load_schedule(schedule_path)
            result = run_block(
                paths=paths,
                schedule=schedule,
                order=args.order,
                fixture=bool(args.fixture),
            )
        else:  # pragma: no cover
            raise FrankensteinDirectError(f"unsupported command: {args.command}")
    except FrankensteinDirectError as exc:
        raise SystemExit(f"frankenstein direct error: {exc}") from exc
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
