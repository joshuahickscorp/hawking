#!/usr/bin/env python3.12
"""Fail-closed Kimi -> GLM -> DeepSeek-V4 Frankenstein pipeline preflight.

This operator joins the three-generation manufacturing plan to the completed
public-path Xet measurement without pretending that incompatible model weights
can simply be spliced together.  It is deliberately separate from
``ramanujan.scaffold.research.odyssey``: the Ramanujan completion boundary
remains the authority for any live research or teacher-trace acquisition.

The preflight is useful now because it binds every live receipt, records every
blocked lane, and freezes the only permitted successor architecture:

* one source model/window at a time;
* fetch -> verify -> process/pack -> seal -> independently verify -> evict;
* sealed behavioural or functional capsules only -- never raw donor weights,
  logits, KV caches, hidden states, or arbitrary cross-architecture tensors;
* a route-aware, jointly trained transfer experiment only, never the rejected
  independent-per-layer functional-student approach.

It does *not* acquire a teacher, restream a parent, or manufacture a
Frankenstein capability claim.  Those actions remain unavailable until the
bound gates genuinely pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.operators.kimi_k3_source_admission import (
    ADMISSION_SCHEMA as KIMI_K3_ADMISSION_SCHEMA,
    ADMISSION_STATUS as KIMI_K3_ADMISSION_STATUS,
    REPOSITORY as KIMI_K3_REPOSITORY,
    validate_admission as validate_kimi_k3_admission,
)
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "workspace"
CAMPAIGN_ROOT = WORKSPACE_ROOT / "campaign"
EVIDENCE_ROOT = CAMPAIGN_ROOT / "evidence"
RUN_ROOT = CAMPAIGN_ROOT / "records" / "runs" / "deepseek-v4"

MIN_FREE_FLOOR_BYTES = 15 * 1024**3
PLAN_SCHEMA = "hawking.gravity.deepseek_v4.frankenstein_pipeline_plan.v1"
PROGRESS_SCHEMA = "hawking.gravity.deepseek_v4.frankenstein_pipeline_progress.v1"
PLAN_NAME = "DEEPSEEK_V4_FRANKENSTEIN_PIPELINE_PLAN.json"
K3_ADMITTED_PLAN_NAME = "DEEPSEEK_V4_FRANKENSTEIN_PIPELINE_PLAN_K3_ADMITTED.json"
SUSTAINED_PUBLIC_PATH_PLAN_NAME = "DEEPSEEK_V4_FRANKENSTEIN_PIPELINE_PLAN_K3_ADMITTED_SUSTAINED_PUBLIC_PATH.json"
PROGRESS_NAME = "DEEPSEEK_V4_FRANKENSTEIN_PROGRESS.jsonl"

WINNER_SCHEMA = "hawking.gravity.deepseek_v4.maximum_public_path_winner.v1"
SUSTAINED_WINNER_SCHEMA = "hawking.gravity.deepseek_v4.public_path_sustained_winner.v1"
SUSTAINED_FOLLOWUP_SCHEMA = "hawking.gravity.deepseek_v4.public_path_sustained_followup.v1"
FULL_MANIFEST_SCHEMA = "hawking.gravity.deepseek_v4.full_stream.v1"
FULL_REVERIFY_SCHEMA = "hawking.gravity.deepseek_v4.full_reverify.v1"
FULL_BLOCKER_SCHEMA = "hawking.gravity.deepseek_v4.full_runtime_blocker.v1"
CHILD_BASELINE_SCHEMA = "hawking.gravity.deepseek_v4.child_baseline.v1"
LATENT_BRIDGE_SCHEMA = "hawking.gravity.deepseek_v4.latent_bridge_contract.v1"
HCLI_SUITE_SCHEMA = "hawking.gravity.deepseek_v4.hcli_live_suite.v1"
TPS_GATE_SCHEMA = "hawking.gravity.deepseek_v4.base_tps_gate.v1"
GLM_DECISION_SCHEMA = "hawking.glm52.functional_decision.v1"
CASCADE_SCHEMA = "hawking.deepseek_v4.cascade_decision.v1"
LADDER_SCHEMA = "hawking.ladder.v3"
RAMANUJAN_GATE_SCHEMA = "hawking.ramanujan.hawking_completion_gate.v1"
RAMANUJAN_OFFLINE_SCHEMA = "hawking.ramanujan.offline_manifest.v1"

DEEPSEEK_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash"
DEEPSEEK_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"


class FrankensteinPipelineError(RuntimeError):
    """A preflight input is malformed, substituted, or not safe to promote."""


@dataclass(frozen=True)
class PipelineInputs:
    """All receipts the consolidated plan must bind before it can be frozen."""

    public_winner: Path
    full_manifest: Path
    full_reverify: Path
    full_runtime_blocker: Path
    child_baseline: Path
    latent_bridge: Path
    hcli_live_suite: Path
    base_tps_gate: Path
    glm_decision: Path
    cascade_decision: Path
    kimi_ladder: Path
    kimi_k26_release: Path
    ramanujan_gate: Path
    ramanujan_offline_manifest: Path
    kimi_k3_admission: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise FrankensteinPipelineError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(path)))


def _regular_file(path: Path, label: str) -> None:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise FrankensteinPipelineError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise FrankensteinPipelineError(f"{label} must be a regular non-symlink file")


def _ensure_parent(path: Path) -> None:
    parent = path.parent
    if parent.exists():
        node = os.lstat(parent)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
            raise FrankensteinPipelineError(f"output parent is not a safe directory: {parent}")
        return
    parent.mkdir(parents=True, exist_ok=False)


def _atomic_create(path: Path, raw: bytes) -> str:
    """Create immutable evidence, accepting only byte-identical retries."""

    if path.exists():
        _regular_file(path, f"existing output {path}")
        existing = path.read_bytes()
        if existing != raw:
            raise FrankensteinPipelineError(
                f"refusing to overwrite different immutable evidence: {path}"
            )
        return _sha256(existing)
    _ensure_parent(path)
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
    _ensure_parent(path)
    encoded = _canonical(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "ab", closefd=True) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _read_document(
    path: str | Path,
    label: str,
    *,
    schema: str | tuple[str, ...],
    require_seal: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a receipt and retain both its file hash and optional seal binding."""

    source = _absolute(path, label)
    _regular_file(source, label)
    try:
        raw = source.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrankensteinPipelineError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise FrankensteinPipelineError(f"{label} must contain a JSON object")
    expected_schemas = (schema,) if isinstance(schema, str) else schema
    if document.get("schema") not in expected_schemas:
        raise FrankensteinPipelineError(
            f"{label} schema mismatch: expected {expected_schemas!r}, got {document.get('schema')!r}"
        )
    recorded_seal = document.get("seal_sha256")
    if require_seal and not isinstance(recorded_seal, str):
        raise FrankensteinPipelineError(f"{label} must carry a seal_sha256")
    seal_verified = False
    if isinstance(recorded_seal, str):
        try:
            verify(document, label=label)
        except SealIntegrityError as exc:
            raise FrankensteinPipelineError(str(exc)) from exc
        seal_verified = True
    return document, {
        "label": label,
        "path": str(source),
        "file_sha256": _sha256(raw),
        "schema": str(document.get("schema")),
        "receipt_seal_sha256": recorded_seal,
        "receipt_seal_verified": seal_verified,
        "integrity": "receipt_seal_verified" if seal_verified else "file_sha256_bound_unsealed_legacy_record",
    }


def _expect(document: Mapping[str, Any], field: str, expected: Any, label: str) -> None:
    observed = document.get(field)
    if observed != expected:
        raise FrankensteinPipelineError(
            f"{label} {field} mismatch: expected {expected!r}, got {observed!r}"
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrankensteinPipelineError(f"{label} must be an object")
    return value


def _kimi_f8(ladder: Mapping[str, Any]) -> Mapping[str, Any]:
    rungs = ladder.get("rungs")
    if not isinstance(rungs, list):
        raise FrankensteinPipelineError("Kimi ladder has no rungs list")
    for rung in rungs:
        if isinstance(rung, Mapping) and rung.get("rung") == "F8":
            return rung
    raise FrankensteinPipelineError("Kimi ladder lacks the K3/F8 admission rung")


def default_inputs(workspace: str | Path = WORKSPACE_ROOT) -> PipelineInputs:
    """Return the exact current evidence set used by the first consolidated run."""

    root = _absolute(workspace, "workspace")
    campaign = root / "campaign"
    evidence = campaign / "evidence"
    runs = campaign / "records" / "runs" / "deepseek-v4"
    child = runs / "child-baseline-v2"
    return PipelineInputs(
        public_winner=(
            evidence
            / "runtime"
            / "tg"
            / "TG_XET_PUBLIC_PATH_SUSTAINED_WINNER_RETRY_BALANCED_SCHEDULER.json"
            if (
                evidence
                / "runtime"
                / "tg"
                / "TG_XET_PUBLIC_PATH_SUSTAINED_WINNER_RETRY_BALANCED_SCHEDULER.json"
            ).is_file()
            else evidence / "runtime" / "tg" / "TG_XET_MAXIMUM_PUBLIC_PATH_WINNER.json"
        ),
        full_manifest=runs / "full-43-layer-stream.gravity" / "manifest.json",
        full_reverify=runs / "full-reverify-receipt.json",
        full_runtime_blocker=runs / "full-runtime-blocker-receipt-v2.json",
        child_baseline=child / "DSV4F_CHILD_BASELINE.json",
        latent_bridge=child / "DSV4F_LATENT_BRIDGE_CONTRACT.json",
        hcli_live_suite=evidence / "hide" / "deepseek-v4-live.sBqM7r" / "hcli-live-suite-receipt-v2.json",
        base_tps_gate=runs / "streamed-layer4-diagnostic.gravity" / "base-tps-gate-receipt.json",
        glm_decision=evidence / "models" / "glm52" / "GLM52_FUNCTIONAL_DECISION.json",
        cascade_decision=evidence / "models" / "deepseek-v4" / "DEEPSEEK_V4_FLASH_CASCADE_DECISION.json",
        kimi_ladder=evidence / "systems" / "hawking" / "HAWKING_LADDER_V3.json",
        kimi_k26_release=evidence / "models" / "kimi-k26" / "KIMI_K26_SOURCE_RELEASE_FOR_GLM52.json",
        ramanujan_gate=REPO_ROOT / "ramanujan" / "governance" / "boundary" / "HAWKING_COMPLETION_GATE.json",
        ramanujan_offline_manifest=REPO_ROOT
        / "ramanujan"
        / "governance"
        / "boundary"
        / "RAMANUJAN_OFFLINE_MANIFEST.json",
        kimi_k3_admission=(
            evidence / "models" / "kimi-k3" / "KIMI_K3_SOURCE_ADMISSION.json"
            if (evidence / "models" / "kimi-k3" / "KIMI_K3_SOURCE_ADMISSION.json").is_file()
            else None
        ),
    )


def _stage(
    identifier: str,
    state: str,
    *,
    title: str,
    dependencies: list[str],
    bindings: list[str],
    block_reasons: list[str],
    work_when_ready: list[str],
    prohibited: list[str],
) -> dict[str, Any]:
    return {
        "id": identifier,
        "title": title,
        "state": state,
        "dependencies": dependencies,
        "input_bindings": bindings,
        "block_reasons": block_reasons,
        "work_when_ready": work_when_ready,
        "prohibited": prohibited,
    }


def _floor_check(workspace: Path) -> None:
    try:
        free = shutil.disk_usage(workspace).free
    except OSError as exc:
        raise FrankensteinPipelineError(f"cannot measure workspace free space: {exc}") from exc
    if free < MIN_FREE_FLOOR_BYTES:
        raise FrankensteinPipelineError(
            "15 GiB free-space floor is not satisfied; no pipeline promotion is safe"
        )


def build_plan(*, inputs: PipelineInputs, workspace: str | Path) -> dict[str, Any]:
    """Validate every lane and return one deterministic, sealed pipeline plan.

    A blocked result is a successful preflight outcome.  It means that the
    gates were checked and no model body, teacher trace, or source evidence was
    discarded merely to create apparent activity.
    """

    workspace_path = _absolute(workspace, "workspace")
    if not workspace_path.is_dir():
        raise FrankensteinPipelineError(f"workspace is not a directory: {workspace_path}")
    _floor_check(workspace_path)

    winner, winner_binding = _read_document(
        inputs.public_winner,
        "public Xet winner",
        schema=(WINNER_SCHEMA, SUSTAINED_WINNER_SCHEMA),
        require_seal=True,
    )
    _expect(winner, "status", "FROZEN", "public Xet winner")
    winner_base_binding: dict[str, Any] | None = None
    winner_followup_binding: dict[str, Any] | None = None
    if winner.get("schema") == SUSTAINED_WINNER_SCHEMA:
        followup_path = winner.get("followup_path")
        if not isinstance(followup_path, str):
            raise FrankensteinPipelineError("sustained public Xet winner lacks a followup path")
        followup, winner_followup_binding = _read_document(
            followup_path,
            "sustained public Xet followup",
            schema=SUSTAINED_FOLLOWUP_SCHEMA,
            require_seal=True,
        )
        _expect(
            winner,
            "followup_seal_sha256",
            winner_followup_binding["receipt_seal_sha256"],
            "sustained public Xet winner",
        )
        base = _mapping(followup.get("base_frozen_winner"), "sustained public Xet base winner")
        base_path = base.get("path")
        if not isinstance(base_path, str):
            raise FrankensteinPipelineError("sustained public Xet followup lacks a base winner path")
        base_winner, winner_base_binding = _read_document(
            base_path,
            "base public Xet winner",
            schema=WINNER_SCHEMA,
            require_seal=True,
        )
        _expect(base_winner, "status", "FROZEN", "base public Xet winner")
        _expect(
            winner,
            "base_frozen_winner_seal_sha256",
            winner_base_binding["receipt_seal_sha256"],
            "sustained public Xet winner",
        )
        winner_source = _mapping(base_winner.get("source"), "base public Xet winner source")
    else:
        winner_source = _mapping(winner.get("source"), "public Xet winner source")
    _expect(winner_source, "repository", DEEPSEEK_REPOSITORY, "public Xet winner source")
    _expect(winner_source, "revision", DEEPSEEK_REVISION, "public Xet winner source")
    winner_profile = _mapping(winner.get("profile"), "public Xet winner profile")
    winner_application = _mapping(
        winner.get("real_stream_application"), "public Xet winner application"
    )

    full_manifest, full_manifest_binding = _read_document(
        inputs.full_manifest, "full DeepSeek manifest", schema=FULL_MANIFEST_SCHEMA, require_seal=True
    )
    _expect(
        full_manifest,
        "status",
        "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY",
        "full DeepSeek manifest",
    )
    full_source = _mapping(full_manifest.get("source"), "full DeepSeek manifest source")
    _expect(full_source, "repository", DEEPSEEK_REPOSITORY, "full DeepSeek manifest source")
    _expect(full_source, "revision", DEEPSEEK_REVISION, "full DeepSeek manifest source")

    reverify, reverify_binding = _read_document(
        inputs.full_reverify, "full DeepSeek reverify", schema=FULL_REVERIFY_SCHEMA, require_seal=True
    )
    _expect(
        reverify,
        "status",
        "FULL_MODEL_STREAM_FULLY_REVERIFIED_RUNTIME_PENDING",
        "full DeepSeek reverify",
    )

    blocker, blocker_binding = _read_document(
        inputs.full_runtime_blocker, "full DeepSeek runtime blocker", schema=FULL_BLOCKER_SCHEMA, require_seal=True
    )
    _expect(
        blocker,
        "status",
        "FULL_STREAMED_RUNTIME_NO_REGISTERED_43_LAYER_ADAPTER",
        "full DeepSeek runtime blocker",
    )
    blocker_artifact = _mapping(blocker.get("artifact"), "full DeepSeek runtime blocker artifact")
    _expect(
        blocker_artifact,
        "manifest_seal_sha256",
        full_manifest_binding["receipt_seal_sha256"],
        "full DeepSeek runtime blocker artifact",
    )
    blocker_storage = _mapping(blocker.get("storage_accounting"), "runtime blocker storage")
    if blocker_storage.get("raw_artifact_eviction_authorized") is not False:
        raise FrankensteinPipelineError("runtime blocker unexpectedly authorizes raw artifact eviction")

    child, child_binding = _read_document(
        inputs.child_baseline, "DeepSeek child baseline", schema=CHILD_BASELINE_SCHEMA, require_seal=True
    )
    _expect(
        child,
        "status",
        "DSV4F_CHILD_BASELINE_FROZEN_FULL_STREAM_RUNTIME_PENDING",
        "DeepSeek child baseline",
    )
    child_claims = _mapping(child.get("claim_boundary"), "DeepSeek child baseline claim boundary")
    if any(
        child_claims.get(key) is not False
        for key in (
            "full_43_layer_runtime",
            "full_43_layer_metal_dispatch",
            "base_true_tps",
            "direct_weight_transplant",
            "kimi_or_glm_donor_weights_present",
            "kimi_or_glm_training_performed",
        )
    ):
        raise FrankensteinPipelineError("child baseline claim boundary is no longer fail-closed")

    bridge, bridge_binding = _read_document(
        inputs.latent_bridge, "DeepSeek latent bridge", schema=LATENT_BRIDGE_SCHEMA, require_seal=True
    )
    _expect(
        bridge,
        "status",
        "DSV4F_FUTURE_BRIDGE_INTERFACES_DECLARED_NO_DONOR_INHERITANCE",
        "DeepSeek latent bridge",
    )

    hcli, hcli_binding = _read_document(
        inputs.hcli_live_suite, "DeepSeek HCLI live suite", schema=HCLI_SUITE_SCHEMA, require_seal=True
    )
    _expect(
        hcli,
        "status",
        "HCLI_LIVE_SUITE_EVIDENCE_SEALED_DIAGNOSTIC_ONLY",
        "DeepSeek HCLI live suite",
    )
    hcli_claims = _mapping(hcli.get("claim_boundary"), "HCLI live-suite claim boundary")
    if hcli_claims.get("full_43_layer_runtime") is not False:
        raise FrankensteinPipelineError("HCLI receipt unexpectedly claims a full runtime")

    tps, tps_binding = _read_document(
        inputs.base_tps_gate, "DeepSeek base TPS gate", schema=TPS_GATE_SCHEMA, require_seal=True
    )
    _expect(tps, "status", "BASE_TRUE_TPS_WITHHELD", "DeepSeek base TPS gate")

    glm, glm_binding = _read_document(
        inputs.glm_decision, "GLM functional decision", schema=GLM_DECISION_SCHEMA, require_seal=False
    )
    _expect(glm, "decision", "FUNCTIONAL_PARTIAL_ONLY", "GLM functional decision")
    _expect(glm, "glm_full_stream", "DO_NOT_STREAM", "GLM functional decision")
    glm_gates = _mapping(glm.get("gates"), "GLM functional gates")
    for gate_name in ("FS2_next_layer_propagation", "FS4_cross_layer_sharing", "FS7_full_stream_admission"):
        gate = _mapping(glm_gates.get(gate_name), f"GLM gate {gate_name}")
        if gate.get("passes") is not False:
            raise FrankensteinPipelineError(f"GLM decision no longer records {gate_name} as failed")

    cascade, cascade_binding = _read_document(
        inputs.cascade_decision, "DeepSeek cascade decision", schema=CASCADE_SCHEMA, require_seal=False
    )
    _expect(cascade, "answer", "NO", "DeepSeek cascade decision")
    _expect(cascade, "verdict", "FUNCTIONAL_PARTIAL_ONLY", "DeepSeek cascade decision")

    ladder, ladder_binding = _read_document(
        inputs.kimi_ladder, "Kimi ladder", schema=LADDER_SCHEMA, require_seal=False
    )
    f8 = _kimi_f8(ladder)
    kimi_k3_admission: dict[str, Any] | None = None
    kimi_k3_admission_binding: dict[str, Any] | None = None
    if inputs.kimi_k3_admission is None:
        for key, expected in (
            ("official_repo", "UNRESOLVED"),
            ("revision", "UNRESOLVED"),
            ("license", "UNRESOLVED"),
            ("release_status", "PENDING_OFFICIAL_WEIGHTS"),
            ("readiness_stage", "A0"),
        ):
            _expect(f8, key, expected, "Kimi K3/F8 admission rung")
    else:
        kimi_k3_admission, kimi_k3_admission_binding = _read_document(
            inputs.kimi_k3_admission,
            "official Kimi K3 source admission",
            schema=KIMI_K3_ADMISSION_SCHEMA,
            require_seal=True,
        )
        try:
            validate_kimi_k3_admission(kimi_k3_admission)
        except Exception as exc:
            raise FrankensteinPipelineError(f"invalid official Kimi K3 source admission: {exc}") from exc
        _expect(
            _mapping(kimi_k3_admission.get("source"), "official Kimi K3 source admission source"),
            "repository",
            KIMI_K3_REPOSITORY,
            "official Kimi K3 source admission source",
        )

    kimi_k26, kimi_k26_binding = _read_document(
        inputs.kimi_k26_release,
        "Kimi K2.6 historical release",
        schema="hawking.kimi_k26.source_release_for_glm52.v1",
        require_seal=True,
    )
    _expect(kimi_k26, "status", "RECONCILED_ALREADY_RELEASED", "Kimi K2.6 historical release")
    kimi_k26_source = _mapping(kimi_k26.get("source"), "Kimi K2.6 historical release source")
    _expect(kimi_k26_source, "repo", "moonshotai/Kimi-K2.6", "Kimi K2.6 historical release source")

    ramanujan_gate, ramanujan_gate_binding = _read_document(
        inputs.ramanujan_gate,
        "Ramanujan completion gate",
        schema=RAMANUJAN_GATE_SCHEMA,
        require_seal=False,
    )
    _expect(
        ramanujan_gate,
        "status",
        "BLOCKED_ON_HAWKING_COMPLETION",
        "Ramanujan completion gate",
    )
    authority = _mapping(ramanujan_gate.get("authority"), "Ramanujan authority")
    if authority.get("ramanujan_research_authorized") is not False or authority.get("production_authority") is not False:
        raise FrankensteinPipelineError("Ramanujan authority gate is not currently fail-closed")

    offline, offline_binding = _read_document(
        inputs.ramanujan_offline_manifest,
        "Ramanujan offline manifest",
        schema=RAMANUJAN_OFFLINE_SCHEMA,
        require_seal=False,
    )
    _expect(
        offline,
        "status",
        "LOCAL_SOURCES_PARTIALLY_GENERATED",
        "Ramanujan offline manifest",
    )

    bindings = {
        "public_xet_winner": winner_binding,
        "deepseek_full_manifest": full_manifest_binding,
        "deepseek_full_reverify": reverify_binding,
        "deepseek_runtime_blocker": blocker_binding,
        "deepseek_child_baseline": child_binding,
        "deepseek_latent_bridge": bridge_binding,
        "deepseek_hcli_live_suite": hcli_binding,
        "deepseek_base_tps_gate": tps_binding,
        "glm_functional_decision": glm_binding,
        "deepseek_cascade_decision": cascade_binding,
        "kimi_k3_admission_ladder": ladder_binding,
        "kimi_k26_historical_release": kimi_k26_binding,
        "ramanujan_completion_gate": ramanujan_gate_binding,
        "ramanujan_offline_manifest": offline_binding,
    }
    if winner_base_binding is not None:
        bindings["public_xet_base_winner"] = winner_base_binding
    if winner_followup_binding is not None:
        bindings["public_xet_sustained_followup"] = winner_followup_binding
    if kimi_k3_admission_binding is not None:
        bindings["kimi_k3_official_source_admission"] = kimi_k3_admission_binding

    kimi_source_admitted = kimi_k3_admission is not None

    stages = [
        _stage(
            "PUBLIC_XET_PATH",
            "COMPLETED_FROZEN",
            title="Measured public DeepSeek transfer path",
            dependencies=[],
            bindings=["public_xet_winner"],
            block_reasons=[],
            work_when_ready=[
                "Reuse the frozen direct presigned-range profile only for a new admissible source window.",
                "Keep outer source-window concurrency at or below the frozen eight-worker dynamic work-stealing shape.",
            ],
            prohibited=[
                "Do not infer WAN throughput from the 10GbE negotiated LAN link.",
                "Do not re-download an already sealed V4 source merely to report activity.",
            ],
        ),
        _stage(
            "DEEPSEEK_RUNTIME",
            "BLOCKED",
            title="DeepSeek-V4 child execution body",
            dependencies=["PUBLIC_XET_PATH"],
            bindings=[
                "deepseek_full_manifest",
                "deepseek_full_reverify",
                "deepseek_runtime_blocker",
                "deepseek_child_baseline",
                "deepseek_hcli_live_suite",
                "deepseek_base_tps_gate",
            ],
            block_reasons=[
                "The full 43-layer artifact is sealed and reverified but has no registered native runtime adapter.",
                "The available HCLI suite is a one-layer CPU diagnostic, not a full runtime proof.",
                "Base true TPS is explicitly withheld.",
            ],
            work_when_ready=[
                "Register and validate the 43-layer native V4 execution adapter.",
                "Pass full load, forward, first-token, HCLI, numeric-parity, and true-TPS gates.",
            ],
            prohibited=[
                "Do not evict the raw content-addressed V4 artifact before an independently verified successor exists.",
                "Do not describe diagnostic HCLI evidence as full-model capability.",
            ],
        ),
        _stage(
            "KIMI_K3_ADMISSION",
            "COMPLETED_FROZEN" if kimi_source_admitted else "BLOCKED",
            title="Kimi K3 grandparent source admission",
            dependencies=["PUBLIC_XET_PATH"],
            bindings=(
                ["kimi_k3_admission_ladder", "kimi_k26_historical_release", "kimi_k3_official_source_admission"]
                if kimi_source_admitted
                else ["kimi_k3_admission_ladder", "kimi_k26_historical_release"]
            ),
            block_reasons=(
                []
                if kimi_source_admitted
                else [
                    "No official K3 repository, immutable revision, license, config, tokenizer, index, or shard inventory is bound.",
                    "Kimi K2.6 is historical K2.6 evidence and cannot be relabelled as K3.",
                ]
            ),
            work_when_ready=(
                [
                    "Keep the admitted K3 revision pinned; metadata admission alone does not authorize a model-body stream.",
                    "Admit a bounded one-model source-window protocol only after the separate authority and teacher-quality gates pass.",
                ]
                if kimi_source_admitted
                else [
                    "Bind the official K3 source, license, immutable revision, exact manifest/file hashes, and access contract.",
                    "Admit only a bounded one-model source-window protocol after that binding passes.",
                ]
            ),
            prohibited=[
                "Do not infer K3 tensor names, geometry, dtypes, or release rights from secondary claims.",
                "Do not use K2.6 as a substitute identity for K3.",
            ],
        ),
        _stage(
            "GLM_MATH_DIRECTOR",
            "BLOCKED",
            title="GLM mathematical parent/director",
            dependencies=["KIMI_K3_ADMISSION"],
            bindings=["glm_functional_decision"],
            block_reasons=[
                "The bound GLM decision is FUNCTIONAL_PARTIAL_ONLY and explicitly DO_NOT_STREAM.",
                "Propagation, cross-layer sharing, and full-stream admission gates fail.",
            ],
            work_when_ready=[
                "Establish a separately qualified, verifier-backed mathematical teacher/director.",
                "Use only sealed method capsules, proof actions, repair pairs, and verifier dispositions after authorization.",
            ],
            prohibited=[
                "Do not launch a GLM parent restream from the rejected functional package.",
                "Do not treat bounded calibration scores as mathematical capability or teacher quality.",
            ],
        ),
        _stage(
            "SEALED_INHERITANCE",
            "BLOCKED",
            title="Kimi/GLM behavioural and mathematical inheritance archive",
            dependencies=["DEEPSEEK_RUNTIME", "KIMI_K3_ADMISSION", "GLM_MATH_DIRECTOR"],
            bindings=["deepseek_latent_bridge", "ramanujan_completion_gate", "ramanujan_offline_manifest"],
            block_reasons=[
                "The child bridge declares interfaces only and contains no donor inheritance.",
                "Ramanujan authority explicitly blocks teacher-trace acquisition and parent-model restream.",
                "No approved mathematical teacher is currently available.",
            ],
            work_when_ready=[
                "For each approved capsule: verify provenance and verifier result, compact, seal, independently verify, then evict the source window.",
                "Keep one active model/window, one capsule shard, one trainable state, and one rollback checkpoint at most.",
            ],
            prohibited=[
                "Never retain raw donor weights, logits, KV caches, hidden states, or arbitrary tensors as inheritance payloads.",
                "Never accumulate raw source parents or unsealed teacher bodies.",
            ],
        ),
        _stage(
            "ROUTE_AWARE_TRANSFER",
            "BLOCKED",
            title="Kimi/GLM inheritance into the DeepSeek child",
            dependencies=["DEEPSEEK_RUNTIME", "SEALED_INHERITANCE"],
            bindings=["deepseek_cascade_decision", "deepseek_child_baseline"],
            block_reasons=[
                "Independent layerwise functional students were rejected by the full DeepSeek cascade because routing diverges under accumulated drift.",
                "The child has no full-runtime trace, donor capsule, or qualified adapter training result.",
            ],
            work_when_ready=[
                "Use a jointly trained, route-aware rollout objective with verifier-dispositioned functional/behavioural records.",
                "Keep adapters, compact modules, and checkpoints separately content-addressed, sealed, reversible, and removable.",
            ],
            prohibited=[
                "No arbitrary weight or tensor transplant across architectures.",
                "No independent-per-layer transfer masquerading as a compositional solution.",
            ],
        ),
        _stage(
            "GRAVITY_RECOMPOSITION",
            "BLOCKED",
            title="Gravity cut-and-sew of the executable child",
            dependencies=["ROUTE_AWARE_TRANSFER"],
            bindings=["deepseek_full_manifest", "deepseek_runtime_blocker"],
            block_reasons=[
                "No validated child runtime or route-aware transfer candidate exists yet.",
                "The raw V4 artifact is still the sole verified source representation and cannot be evicted.",
            ],
            work_when_ready=[
                "Choose per-function representation: native child weight, reversible adapter, compact module, retrieval, verifier, tool, or discard.",
                "Seal a successor, independently verify it, then and only then authorize parent/source eviction.",
            ],
            prohibited=[
                "Do not declare a three-model executable artifact before its child runtime and independent successor verification exist.",
            ],
        ),
        _stage(
            "FOUR_WAY_ABLATION",
            "BLOCKED",
            title="Required contribution proof",
            dependencies=["GRAVITY_RECOMPOSITION"],
            bindings=["deepseek_child_baseline"],
            block_reasons=["There is no executable candidate or qualified inherited corpus to compare."],
            work_when_ready=[
                "Compare DeepSeek base, DeepSeek + Kimi inheritance, DeepSeek + Kimi + GLM inheritance, and final Gravity-recomposed child.",
                "Promote only if capability, reliability, runtime, storage, and restart evidence beat the base under the same verified evaluation contract.",
            ],
            prohibited=[
                "Do not claim donor contribution without the four-way ablation.",
            ],
        ),
    ]

    plan = {
        "schema": PLAN_SCHEMA,
        "status": "BLOCKED_BY_REQUIRED_GATES",
        "objective": "one executable DeepSeek-family child carrying only verified Kimi/GLM functional and behavioural inheritance",
        "execution_order": [stage["id"] for stage in stages],
        "public_path": {
            "winner_seal_sha256": winner_binding["receipt_seal_sha256"],
            "winner_schema": winner_binding["schema"],
            "transport": winner_profile.get("transport"),
            "scheduler_shape": winner_profile.get("scheduler_shape"),
            "connection_reuse": winner_profile.get("connection_reuse"),
            "outer_source_windows_maximum": winner_application.get("outer_source_windows_maximum"),
            "source_cache_bytes": winner_application.get("source_cache_bytes"),
        },
        "source_identity": {
            "repository": DEEPSEEK_REPOSITORY,
            "revision": DEEPSEEK_REVISION,
            "full_manifest_seal_sha256": full_manifest_binding["receipt_seal_sha256"],
            "full_stream_status": full_manifest.get("status"),
            "source_parent_retained": False,
        },
        "kimi_k3_source_identity": (
            {
                "repository": kimi_k3_admission["source"]["repository"],
                "revision": kimi_k3_admission["source"]["revision"],
                "admission_seal_sha256": kimi_k3_admission_binding["receipt_seal_sha256"],
                "status": kimi_k3_admission["status"],
                "metadata_only": True,
            }
            if kimi_k3_admission is not None and kimi_k3_admission_binding is not None
            else {"status": "NOT_ADMITTED"}
        ),
        "storage_contract": {
            "hard_free_floor_bytes": MIN_FREE_FLOOR_BYTES,
            "full_persistent_source_cache": False,
            "parent_accumulation": False,
            "max_simultaneous_source_models": 1,
            "source_window_lifecycle": [
                "fetch",
                "verify_exact_range_and_file_hash",
                "process_or_pack",
                "seal",
                "independently_verify_successor",
                "evict",
            ],
            "raw_v4_eviction_authorized": False,
        },
        "transfer_contract": {
            "direct_weight_transplant": False,
            "independent_layerwise_student_transfer": False,
            "allowed_inheritance_after_gates": [
                "verified_behavioral_traces",
                "method_capsules",
                "verifier_dispositions",
                "repair_preference_pairs",
                "route_aware_rollout_targets",
                "separately_sealed_reversible_adapters_or_compact_modules",
            ],
            "prohibited_payloads": [
                "raw_donor_weights",
                "arbitrary_donor_tensors",
                "teacher_logits",
                "teacher_KV_caches",
                "teacher_hidden_states",
            ],
        },
        "ramanujan_boundary": {
            "status": ramanujan_gate.get("status"),
            "research_authorized": authority.get("ramanujan_research_authorized"),
            "production_authority": authority.get("production_authority"),
            "offline_manifest_status": offline.get("status"),
            "current_permitted_work": [
                "static_contract_and_provenance_verification",
                "fixture_only_scaffold_work",
                "local_deterministic_data_checks",
                "runtime_adapter_engineering_not_teacher_acquisition",
            ],
        },
        "input_bindings": bindings,
        "stages": stages,
        "current_blockers": [
            "full DeepSeek V4 runtime adapter/HCLI/TPS gate",
            *( [] if kimi_source_admitted else ["Kimi K3 immutable official source admission"] ),
            "qualified GLM mathematical director",
            "Ramanujan completion and owner authority for live teacher acquisition",
        ],
    }
    return seal(plan)


def write_preflight(
    *,
    inputs: PipelineInputs,
    workspace: str | Path,
    out: str | Path,
    progress: str | Path,
) -> dict[str, Any]:
    """Freeze the deterministic plan and append a timestamped resume-safe event."""

    plan = build_plan(inputs=inputs, workspace=workspace)
    output = _absolute(out, "plan output")
    progress_path = _absolute(progress, "progress output")
    _atomic_create(output, _canonical(plan) + b"\n")
    event = seal(
        {
            "schema": PROGRESS_SCHEMA,
            "recorded_at": _utc_now(),
            "event": "FRANKENSTEIN_PIPELINE_PREFLIGHT",
            "status": plan["status"],
            "plan_path": str(output),
            "plan_seal_sha256": plan["seal_sha256"],
            "stage_states": {stage["id"]: stage["state"] for stage in plan["stages"]},
            "blocked_stage_ids": [
                stage["id"] for stage in plan["stages"] if stage["state"] == "BLOCKED"
            ],
        }
    )
    _append_jsonl(progress_path, event)
    return {
        "status": plan["status"],
        "plan_path": str(output),
        "progress_path": str(progress_path),
        "plan_seal_sha256": plan["seal_sha256"],
        "blocked_stage_ids": [
            stage["id"] for stage in plan["stages"] if stage["state"] == "BLOCKED"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="freeze the consolidated fail-closed pipeline plan")
    preflight.add_argument("--workspace", type=Path, default=WORKSPACE_ROOT)
    preflight.add_argument("--out", type=Path)
    preflight.add_argument("--progress", type=Path, default=RUN_ROOT / PROGRESS_NAME)
    direct = commands.add_parser(
        "direct",
        help=(
            "ceremony-free streaming fusion harness (see lab.operators.frankenstein_direct); "
            "remaining args are forwarded"
        ),
    )
    direct.add_argument(
        "direct_argv",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to frankenstein_direct (e.g. first-step, schedule)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            inputs = default_inputs(args.workspace)
            result = write_preflight(
                inputs=inputs,
                workspace=args.workspace,
                out=args.out
                or EVIDENCE_ROOT
                / "models"
                / "deepseek-v4"
                / (
                    SUSTAINED_PUBLIC_PATH_PLAN_NAME
                    if inputs.public_winner.name == "TG_XET_PUBLIC_PATH_SUSTAINED_WINNER_RETRY_BALANCED_SCHEDULER.json"
                    else K3_ADMITTED_PLAN_NAME if inputs.kimi_k3_admission is not None else PLAN_NAME
                ),
                progress=args.progress,
            )
        elif args.command == "direct":
            # Thin forwarder to the ceremony-free harness.  Strips a leading "--"
            # that argparse.REMAINDER may preserve after `direct -- first-step`.
            from lab.operators.frankenstein_direct import main as direct_main

            forwarded = list(args.direct_argv or [])
            if forwarded and forwarded[0] == "--":
                forwarded = forwarded[1:]
            return direct_main(forwarded)
        else:  # pragma: no cover - argparse makes this unreachable.
            raise FrankensteinPipelineError(f"unsupported command: {args.command}")
    except FrankensteinPipelineError as exc:
        raise SystemExit(f"frankenstein pipeline error: {exc}") from exc
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
