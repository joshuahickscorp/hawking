#!/usr/bin/env python3.12
"""Provenance + inheritance receipts for Stage-1 PROTO_FRANKENSTEIN.

Schemas and builders for sealed receipts that document:
  * verified GLM math-trace shards (or their offline fixtures)
  * method / decomposition / formalization / repair / value-ranking targets
  * reversible bridge / adapter or compact-module manifest
  * capability-ablation-vs-BASE (A vs B)
  * coding / agent / tool non-regression
  * runtime & storage accounting

Also seals the **PROTO → Kimi handoff contract**: stable KIMI_STRATEGIC_BRIDGE
points that Stage 1 must leave untouched so Stage 2 can graft strategic
inheritance without disturbing GLM math.

Training-free and Kimi-free: this module never consumes Kimi weights or runs
training.  It only records governance artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators.frankenstein_ablation import (
    ARM_A,
    ARM_B,
    CLAIM_BOUNDARY,
    MATH_DOMAINS,
    SECONDARY_CAPABILITIES,
    run_avb_from_fixture,
)
from lab.operators.frankenstein_fusion_op import (
    BRIDGE_DTYPE,
    BRIDGE_INPUT_SHAPE,
    BRIDGE_OUTPUT_SHAPE,
    BRIDGES,
    DEEPSEEK_V4_FLASH,
    GLM_5_2,
    TRANSPLANT_POINT_NAMES,
    donor_for_bridge,
    projection_shape,
    residual_adapter_shape,
)
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "workspace"
CAMPAIGN_ROOT = WORKSPACE_ROOT / "campaign"
EVIDENCE_ROOT = CAMPAIGN_ROOT / "evidence"
FRANK_EVIDENCE = EVIDENCE_ROOT / "models" / "frankenstein"

DEFAULT_BRIDGE_PATH = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_LATENT_BRIDGE_CONTRACT.json"
)
DEFAULT_TRANSPLANT_PATH = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_TRANSPLANT_POINTS.json"
)
DEFAULT_GLM_SUBSPACE = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/frankenstein/glm-subspace"
)

# --- schemas ---
INHERITANCE_RECEIPT_SCHEMA = "hawking.frankenstein.stage1_inheritance_receipt.v1"
ADAPTER_MANIFEST_SCHEMA = "hawking.frankenstein.stage1_adapter_manifest.v1"
RUNTIME_STORAGE_SCHEMA = "hawking.frankenstein.stage1_runtime_storage.v1"
HANDOFF_CONTRACT_SCHEMA = "hawking.frankenstein.proto_to_kimi_handoff_contract.v1"
GLM_TRACE_SHARD_SCHEMA = "hawking.frankenstein.glm_math_trace_shard.v1"

HANDOFF_CONTRACT_NAME = "PROTO_TO_KIMI_HANDOFF_CONTRACT.json"

# Bridge points Stage 2 (Kimi) owns — Stage 1 must not mutate these.
KIMI_STRATEGIC_TARGET_FUNCTIONS: tuple[str, ...] = (
    "planning",
    "long_horizon_decomposition",
    "coding_breadth",
    "tool_policy",
    "context_management",
    "critique_and_synthesis",
)

# GLM math targets Stage 1 owns.
GLM_MATH_TARGET_FUNCTIONS: tuple[str, ...] = (
    "method_selection",
    "mathematical_decomposition",
    "formalization",
    "proof_repair",
    "value_ranking",
)

# Transplant points reserved exclusively for Kimi strategic grafting at stage 2.
# Stage 1 may write GLM_MATH_BRIDGE adapters on the remaining points, but must
# not place adapters under bridge=KIMI_STRATEGIC_BRIDGE.
KIMI_PRESERVED_TRANSPLANT_POINTS: tuple[str, ...] = (
    "pre_norm_hidden_state",
    "post_attention_hidden_state",
    "pre_router_hidden_state",
    "router_logits",
    "selected_expert_ids",
    "route_probabilities_and_margins",
    "post_moe_hidden_state",
    "mhc_state",
    "attention_index_state",
    "final_hidden_state",
    "lm_head_logits",
    "hcli_tool_action_decision",
)


class ReceiptError(RuntimeError):
    """Receipt construction or integrity failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _regular_file(path: Path, label: str) -> None:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise ReceiptError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise ReceiptError(f"{label} must be a regular non-symlink file")


def _ensure_dir(path: Path) -> None:
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
            raise ReceiptError(f"not a safe directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> str:
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    encoded = raw.encode("utf-8")
    if path.exists():
        _regular_file(path, f"existing output {path}")
        existing = path.read_bytes()
        if existing != encoded:
            raise ReceiptError(
                f"refusing to overwrite different immutable evidence: {path}"
            )
        return _sha256(existing)
    _ensure_dir(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return _sha256(encoded)


# Content-hash large streaming artifacts only when asked; size alone is enough
# for declarative binding of multi-GB subspace dumps.
_HASH_SIZE_CAP_BYTES = 256 * 1024 * 1024


def _optional_binding(
    path: Path | None,
    *,
    label: str,
    hash_content: bool = True,
) -> dict[str, Any] | None:
    if path is None:
        return None
    if path.is_dir():
        return {
            "label": label,
            "path": str(path.resolve()),
            "present": True,
            "kind": "directory",
            "note": "directory cited; individual shard files bind separately",
        }
    if not path.is_file():
        return {
            "label": label,
            "path": str(path),
            "present": False,
            "note": "path cited but not present on this host; binding is declarative",
        }
    _regular_file(path, label)
    size = path.stat().st_size
    binding: dict[str, Any] = {
        "label": label,
        "path": str(path.resolve()),
        "present": True,
        "bytes": size,
    }
    if hash_content and size <= _HASH_SIZE_CAP_BYTES:
        binding["file_sha256"] = _sha256_file(path)
    elif hash_content and size > _HASH_SIZE_CAP_BYTES:
        binding["file_sha256"] = None
        binding["hash_deferred"] = True
        binding["note"] = (
            f"content hash deferred for artifact > {_HASH_SIZE_CAP_BYTES} bytes; "
            "size and path are the binding"
        )
    return binding


def _bind_glm_subspace(glm_subspace_path: Path | None) -> dict[str, Any] | None:
    if glm_subspace_path is None:
        return None
    if glm_subspace_path.is_dir():
        npz = glm_subspace_path / "subspace_final.npz"
        directory = _optional_binding(glm_subspace_path, label="glm-subspace")
        # Multi-GB streaming dump: bind path+size, do not hash by default.
        artifact = _optional_binding(
            npz, label="glm-subspace/subspace_final.npz", hash_content=True
        )
        return {"directory": directory, "primary_artifact": artifact}
    return _optional_binding(glm_subspace_path, label="glm-subspace")


def glm_math_targets() -> list[dict[str, Any]]:
    """Method / decomposition / formalization / repair / value-ranking targets."""

    return [
        {
            "name": name,
            "bridge": "GLM_MATH_BRIDGE",
            "donor": "glm_5_2",
            "optimization": "maximize",
            "stage": 1,
        }
        for name in GLM_MATH_TARGET_FUNCTIONS
    ]


def build_adapter_manifest(
    *,
    blocks: Sequence[Mapping[str, Any]] | None = None,
    projections_included: bool = True,
) -> dict[str, Any]:
    """Reversible bridge/adapter or compact-module manifest for PROTO.

    Stage 1 may only list adapters under GLM_MATH_BRIDGE.  Any KIMI_STRATEGIC_BRIDGE
    block is a contract violation.
    """

    adapter = residual_adapter_shape()
    listed = list(blocks or ())
    for block in listed:
        bridge = block.get("bridge")
        if bridge == "KIMI_STRATEGIC_BRIDGE":
            raise ReceiptError(
                "stage-1 adapter manifest must not include KIMI_STRATEGIC_BRIDGE "
                "blocks; those points are reserved for stage-2 handoff"
            )
        if bridge not in (None, "GLM_MATH_BRIDGE"):
            raise ReceiptError(f"unknown or disallowed bridge in stage-1 manifest: {bridge!r}")

    document = {
        "schema": ADAPTER_MANIFEST_SCHEMA,
        "recorded_at": _utc_now(),
        "kind": "reversible_residual_adapter_or_compact_module",
        "direct_weight_transplant": False,
        "gravity_compressed": False,
        "trained": False,
        "training_free_closed_form_fit": True,
        "student_body": {
            "repository": DEEPSEEK_V4_FLASH["repository"],
            "revision": DEEPSEEK_V4_FLASH["revision"],
            "hidden_size": DEEPSEEK_V4_FLASH["hidden_size"],
            "num_hidden_layers": DEEPSEEK_V4_FLASH["num_hidden_layers"],
            "read_only": True,
        },
        "donor": {
            "repository": GLM_5_2["repository"],
            "revision": GLM_5_2["revision"],
            "family": GLM_5_2["family"],
        },
        "bridge": "GLM_MATH_BRIDGE",
        "forbidden_bridges_stage1": ["KIMI_STRATEGIC_BRIDGE"],
        "adapter_shape": adapter,
        "projection": projection_shape(donor="glm_5_2") if projections_included else None,
        "io_contract": {
            "input_shape": list(BRIDGE_INPUT_SHAPE),
            "output_shape": list(BRIDGE_OUTPUT_SHAPE),
            "dtype": BRIDGE_DTYPE,
        },
        "transplant_points_available": list(TRANSPLANT_POINT_NAMES),
        "blocks": list(listed),
        "block_count": len(listed),
        "removable": True,
        "reversible": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    return seal(document)


def build_runtime_storage_accounting(
    *,
    adapter_archive_bytes: int = 0,
    working_set_bytes: int = 0,
    free_bytes_at_seal: int | None = None,
    floor_bytes: int = 15 * 1024**3,
    tps_base: float | None = None,
    tps_proto: float | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Runtime & storage accounting for a PROTO candidate (shard-stage)."""

    if adapter_archive_bytes < 0 or working_set_bytes < 0:
        raise ReceiptError("byte counts must be non-negative")
    tps_delta = None
    tps_ok = None
    if tps_base is not None and tps_proto is not None:
        tps_delta = float(tps_proto) - float(tps_base)
        # Runtime TPS is a secondary hard gate when measured; recorded here.
        tps_ok = float(tps_proto) >= float(tps_base) * (1.0 - 0.02)
    document = {
        "schema": RUNTIME_STORAGE_SCHEMA,
        "recorded_at": _utc_now(),
        "storage": {
            "adapter_archive_bytes": int(adapter_archive_bytes),
            "working_set_bytes": int(working_set_bytes),
            "floor_bytes": int(floor_bytes),
            "free_bytes_at_seal": free_bytes_at_seal,
            "student_body_resident_read_only": True,
            "donor_weights_retained": False,
            "gravity_compressed": False,
        },
        "runtime": {
            "tps_base": tps_base,
            "tps_proto": tps_proto,
            "tps_delta": tps_delta,
            "tps_non_regression_within_2pct": tps_ok,
            "full_model_runtime_available": False,
            "measurement_scope": "shard_or_unmeasured",
        },
        "notes": notes
        or (
            "Stage-1 accounting is declarative for unmeasured full-runtime fields. "
            "When TPS is supplied it is treated as a secondary hard gate (±2%)."
        ),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    return seal(document)


def build_glm_trace_shard_receipt(
    *,
    shard_id: str,
    path: Path | str | None = None,
    target_functions: Sequence[str] | None = None,
    student_layers: Sequence[int] | None = None,
    verified: bool = False,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Receipt for one verified (or fixture) GLM math trace shard."""

    binding = None
    if path is not None:
        binding = _optional_binding(Path(path), label=f"glm_trace_shard:{shard_id}")
    targets = list(target_functions or GLM_MATH_TARGET_FUNCTIONS)
    for name in targets:
        if name not in GLM_MATH_TARGET_FUNCTIONS and name not in MATH_DOMAINS:
            raise ReceiptError(f"unknown GLM math target: {name!r}")
    document = {
        "schema": GLM_TRACE_SHARD_SCHEMA,
        "recorded_at": _utc_now(),
        "shard_id": shard_id,
        "bridge": "GLM_MATH_BRIDGE",
        "donor": donor_for_bridge("GLM_MATH_BRIDGE"),
        "target_functions": targets,
        "student_layers": list(student_layers or ()),
        "verified": bool(verified),
        "binding": binding,
        "meta": dict(meta or {}),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    return seal(document)


def build_inheritance_receipt(
    *,
    transfer_module_id: str,
    ablation_report: Mapping[str, Any],
    adapter_manifest: Mapping[str, Any] | None = None,
    glm_shards: Sequence[Mapping[str, Any]] | None = None,
    runtime_storage: Mapping[str, Any] | None = None,
    bridge_path: Path | None = DEFAULT_BRIDGE_PATH,
    transplant_path: Path | None = DEFAULT_TRANSPLANT_PATH,
    glm_subspace_path: Path | None = DEFAULT_GLM_SUBSPACE,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Full Stage-1 inheritance receipt binding ablation + provenance."""

    if ablation_report.get("schema") != "hawking.frankenstein.stage1_avb_ablation.v1":
        raise ReceiptError("ablation_report has unexpected schema")
    verify(ablation_report, label="ablation_report")

    secondary = ablation_report.get("secondary") or {}
    non_regression = {
        "coding_and_repository_work": None,
        "tool_use": None,
        "agentic_planning": None,
        "verdict": secondary.get("verdict"),
        "reject_rule_fired": secondary.get("reject_rule_fired"),
        "domains": secondary.get("domains"),
    }
    # Surface the three named non-regression rows explicitly for auditors.
    for row in secondary.get("domains") or []:
        if row.get("domain") in non_regression:
            non_regression[row["domain"]] = {
                "base": row.get("base"),
                "proto": row.get("proto"),
                "delta": row.get("delta"),
                "gate": row.get("gate"),
            }

    document = {
        "schema": INHERITANCE_RECEIPT_SCHEMA,
        "recorded_at": _utc_now(),
        "stage": 1,
        "transfer_module_id": transfer_module_id,
        "arms": {"A": ARM_A, "B": ARM_B},
        "math_targets": glm_math_targets(),
        "math_domains": list(MATH_DOMAINS),
        "secondary_capabilities": list(SECONDARY_CAPABILITIES),
        "glm_trace_shards": list(glm_shards or ()),
        "adapter_manifest": adapter_manifest,
        "capability_ablation_vs_base": {
            "verdict": ablation_report.get("verdict"),
            "reject_rule_fired": ablation_report.get("reject_rule_fired"),
            "math_mean_gain": (ablation_report.get("math") or {}).get("mean_gain"),
            "report_seal_sha256": ablation_report.get("seal_sha256"),
            "report": ablation_report,
        },
        "coding_agent_tool_non_regression": non_regression,
        "runtime_storage": runtime_storage,
        "bindings": {
            "latent_bridge_contract": _optional_binding(
                bridge_path, label="DSV4F_LATENT_BRIDGE_CONTRACT"
            ),
            "transplant_points": _optional_binding(
                transplant_path, label="DSV4F_TRANSPLANT_POINTS"
            ),
            "glm_subspace": _bind_glm_subspace(glm_subspace_path),
        },
        "kimi_handoff": {
            "contract_name": HANDOFF_CONTRACT_NAME,
            "bridge_preserved": "KIMI_STRATEGIC_BRIDGE",
            "stage1_must_not_mutate": True,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "extra": dict(extra or {}),
    }
    return seal(document)


def kimi_bridge_tensor_locations() -> list[dict[str, Any]]:
    """Stable tensor/state locations for KIMI_STRATEGIC_BRIDGE (stage-2 graft points).

    Locations are drawn from the sealed DSV4F transplant-point vocabulary and
    the latent-bridge I/O contract (student residual stream H=4096, bf16).
    Stage 1 records them as **untouched** so stage 2 can attach adapters
    without disturbing GLM math modules.
    """

    rows: list[dict[str, Any]] = []
    for point in KIMI_PRESERVED_TRANSPLANT_POINTS:
        rows.append(
            {
                "bridge": "KIMI_STRATEGIC_BRIDGE",
                "transplant_point": point,
                "tensor_state": {
                    "input_name": "per_token_hidden_state",
                    "input_shape": list(BRIDGE_INPUT_SHAPE),
                    "output_name": "reversible_residual_adapter_output",
                    "output_shape": list(BRIDGE_OUTPUT_SHAPE),
                    "dtype": BRIDGE_DTYPE,
                },
                "student_layers": {
                    "all": list(range(DEEPSEEK_V4_FLASH["num_hidden_layers"])),
                    "note": (
                        "Stage-2 may select a subset; stage-1 must not write "
                        "KIMI_STRATEGIC_BRIDGE adapters on any of these layers."
                    ),
                },
                "block_id_pattern": f"KIMI_STRATEGIC_BRIDGE__{point}__L{{layer:02d}}",
                "stage1_status": "PRESERVED_UNTOUCHED",
                "stage2_owner": "Kimi K3 strategic inheritance",
                "direct_weight_transplant": False,
                "adapter_requirement": (
                    "separately sealed reversible residual adapter or policy head; "
                    "no direct donor weight replacement"
                ),
            }
        )
    return rows


def build_proto_to_kimi_handoff_contract(
    *,
    bridge_path: Path | None = DEFAULT_BRIDGE_PATH,
    transplant_path: Path | None = DEFAULT_TRANSPLANT_PATH,
) -> dict[str, Any]:
    """Seal the PROTO → Kimi handoff contract.

    Names every KIMI_STRATEGIC_BRIDGE point and its tensor/state location so
    stage 2 can graft without disturbing stage-1 GLM math inheritance.
    """

    bridge_binding = _optional_binding(
        bridge_path, label="DSV4F_LATENT_BRIDGE_CONTRACT"
    )
    transplant_binding = _optional_binding(
        transplant_path, label="DSV4F_TRANSPLANT_POINTS"
    )

    # Validate transplant vocabulary covers every preserved point.
    missing = [
        p for p in KIMI_PRESERVED_TRANSPLANT_POINTS if p not in TRANSPLANT_POINT_NAMES
    ]
    if missing:
        raise ReceiptError(
            f"Kimi preserved points missing from TRANSPLANT_POINT_NAMES: {missing}"
        )

    document = {
        "schema": HANDOFF_CONTRACT_SCHEMA,
        "name": "PROTO_TO_KIMI_HANDOFF_CONTRACT",
        "recorded_at": _utc_now(),
        "stage_from": 1,
        "stage_to": 2,
        "from_artifact": "PROTO_FRANKENSTEIN = DeepSeek-V4-Flash + GLM math",
        "to_artifact": "FINAL_FRANKENSTEIN = PROTO + Kimi K3 strategic",
        "policy": {
            "stage1_must_not_mutate_kimi_bridge": True,
            "stage1_writes_only": "GLM_MATH_BRIDGE",
            "stage2_writes_only": "KIMI_STRATEGIC_BRIDGE",
            "no_kimi_consumption_in_stage1": True,
            "no_math_bridge_overwrite_in_stage2": True,
            "direct_weight_transplant": False,
            "training_free_stage1": True,
            "additive_not_subtractive": True,
        },
        "preserved_bridge": {
            "name": "KIMI_STRATEGIC_BRIDGE",
            "donor_family": "kimi_k3",
            "target_functions": list(KIMI_STRATEGIC_TARGET_FUNCTIONS),
            "source_bridge_contract_functions": [
                "planning",
                "tool_policy",
                "long_horizon_decomposition",
                "critique",
                "context_management",
            ],
            "extended_for_handoff": [
                "coding_breadth",
                "critique_and_synthesis",
            ],
            "note": (
                "coding_breadth and critique_and_synthesis are explicit stage-2 "
                "targets named by the stage-1 handoff so the second stream can "
                "cover coding breadth and critique/synthesis without touching "
                "GLM math adapters."
            ),
        },
        "stage1_math_bridge": {
            "name": "GLM_MATH_BRIDGE",
            "donor_family": "glm_5_2",
            "target_functions": list(GLM_MATH_TARGET_FUNCTIONS),
            "status": "ACTIVE_STAGE1",
        },
        "bridge_points": kimi_bridge_tensor_locations(),
        "declared_bridges": list(BRIDGES),
        "student_geometry": {
            "repository": DEEPSEEK_V4_FLASH["repository"],
            "revision": DEEPSEEK_V4_FLASH["revision"],
            "hidden_size": DEEPSEEK_V4_FLASH["hidden_size"],
            "num_hidden_layers": DEEPSEEK_V4_FLASH["num_hidden_layers"],
            "residual_stream_shape": list(BRIDGE_INPUT_SHAPE),
            "dtype": BRIDGE_DTYPE,
        },
        "bindings": {
            "latent_bridge_contract": bridge_binding,
            "transplant_points": transplant_binding,
        },
        "graft_rules_for_stage2": [
            "Attach Kimi adapters only under bridge=KIMI_STRATEGIC_BRIDGE.",
            "Do not rewrite, average, or zero GLM_MATH_BRIDGE adapter blocks.",
            "Do not perform direct weight transplant into the DeepSeek body.",
            "Keep adapters content-addressed, sealed, reversible, and removable.",
            "Re-run A-vs-B-vs-C ablation after graft; secondary gates remain hard.",
            "Do not consume this contract as permission to load Kimi in stage 1.",
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    return seal(document)


def write_handoff_contract(
    out_dir: Path | str | None = None,
    *,
    bridge_path: Path | None = DEFAULT_BRIDGE_PATH,
    transplant_path: Path | None = DEFAULT_TRANSPLANT_PATH,
) -> tuple[Path, dict[str, Any]]:
    """Build and atomically write PROTO_TO_KIMI_HANDOFF_CONTRACT.json."""

    directory = Path(out_dir) if out_dir is not None else FRANK_EVIDENCE
    _ensure_dir(directory)
    path = directory / HANDOFF_CONTRACT_NAME
    document = build_proto_to_kimi_handoff_contract(
        bridge_path=bridge_path,
        transplant_path=transplant_path,
    )
    _atomic_write_json(path, document)
    return path, document


def write_receipt(path: Path | str, document: Mapping[str, Any]) -> str:
    if "seal_sha256" not in document:
        document = seal(dict(document))
    else:
        verify(document, label="receipt")
    return _atomic_write_json(Path(path), document)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage-1 PROTO inheritance receipts and Kimi handoff contract"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_hand = sub.add_parser(
        "handoff-contract",
        help="Seal PROTO_TO_KIMI_HANDOFF_CONTRACT.json",
    )
    p_hand.add_argument(
        "--out-dir",
        type=Path,
        default=FRANK_EVIDENCE,
        help="Directory for PROTO_TO_KIMI_HANDOFF_CONTRACT.json",
    )
    p_hand.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE_PATH)
    p_hand.add_argument("--transplant", type=Path, default=DEFAULT_TRANSPLANT_PATH)

    p_inh = sub.add_parser(
        "inheritance-receipt",
        help="Build inheritance receipt from an ablation fixture evaluation",
    )
    p_inh.add_argument("--fixture", type=Path, required=True)
    p_inh.add_argument("--transfer-module-id", required=True)
    p_inh.add_argument("--out", type=Path, required=True)
    p_inh.add_argument("--shard-id", action="append", default=[], dest="shard_ids")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "handoff-contract":
            path, document = write_handoff_contract(
                args.out_dir,
                bridge_path=args.bridge,
                transplant_path=args.transplant,
            )
            print(
                json.dumps(
                    {
                        "out": str(path),
                        "seal_sha256": document["seal_sha256"],
                        "bridge_points": len(document["bridge_points"]),
                    }
                )
            )
            return 0

        if args.command == "inheritance-receipt":
            ablation = run_avb_from_fixture(args.fixture)
            shards = [
                build_glm_trace_shard_receipt(shard_id=sid, verified=False)
                for sid in args.shard_ids
            ]
            manifest = build_adapter_manifest(blocks=[])
            runtime = build_runtime_storage_accounting()
            receipt = build_inheritance_receipt(
                transfer_module_id=args.transfer_module_id,
                ablation_report=ablation,
                adapter_manifest=manifest,
                glm_shards=shards,
                runtime_storage=runtime,
            )
            write_receipt(args.out, receipt)
            print(
                json.dumps(
                    {
                        "out": str(args.out),
                        "verdict": ablation["verdict"],
                        "seal_sha256": receipt["seal_sha256"],
                    }
                )
            )
            return 0 if ablation["verdict"] == "ACCEPT" else 2

        raise ReceiptError(f"unknown command: {args.command}")
    except ReceiptError as exc:
        print(json.dumps({"error": "RECEIPT_ERROR", "detail": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
