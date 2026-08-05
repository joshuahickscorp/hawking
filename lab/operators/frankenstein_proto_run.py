#!/usr/bin/env python3.12
"""End-to-end PROTO_FRANKENSTEIN full-run orchestrator.

Wires already-built pieces into one pipeline:

  (a) load the sealed GLM math transfer module
  (b) compose it onto the DeepSeek body transplant points → raw PROTO
      artifact descriptor (per-point residual modules; NOT a byte-merge;
      NOT gravity-compressed)
  (c) invoke the student forward for activations at transplant points
      (gated ``DEEPSEEK_FORWARD_PENDING`` until a callable forward exists)
  (d) run A-vs-B capability ablation (math gain + secondary non-regression)
  (e) seal ``PROTO_FRANKENSTEIN_RUN_RECEIPT.json`` with provenance, verdict,
      runtime/storage accounting, and the Kimi handoff pointer

Training-free only.  Additive-not-subtractive reject rule is enforced by the
ablation module.  Capability validation is fail-closed when the forward is
unavailable — dry-run and full-run never fabricate math-bench scores.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import shutil
import stat
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lab.operators.frankenstein_ablation import (
    ARM_A,
    ARM_B,
    CLAIM_BOUNDARY as ABLATION_CLAIM_BOUNDARY,
    DEFAULT_SECONDARY_TOLERANCE,
    FullModelBenchRefused,
    MATH_DOMAINS,
    SECONDARY_CAPABILITIES,
    ShardBenchRequest,
    make_capability_fixture,
    run_avb_from_fixture,
    run_shard_bench,
    write_report as write_ablation_report,
)
from lab.operators.frankenstein_fusion_op import (
    BRIDGES,
    DEEPSEEK_V4_FLASH,
    FORWARD_GATE,
    GLM_5_2,
    TRANSPLANT_POINT_NAMES,
)
from lab.operators.frankenstein_receipts import (
    HANDOFF_CONTRACT_NAME,
    KIMI_PRESERVED_TRANSPLANT_POINTS,
    build_adapter_manifest,
    build_proto_to_kimi_handoff_contract,
    build_runtime_storage_accounting,
    write_handoff_contract,
)
from lab.operators.frankenstein_transfer import (
    DEFAULT_BODY_PATH,
    DEFAULT_BRIDGE_PATH,
    DEFAULT_TRANSPLANT_PATH,
    TRANSFER_MODULE_SCHEMA,
    assert_no_training_path as transfer_assert_no_training_path,
    frankenstein_transfer_apply,
    load_transfer_module,
)
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "workspace"
CAMPAIGN_ROOT = WORKSPACE_ROOT / "campaign"
EVIDENCE_ROOT = CAMPAIGN_ROOT / "evidence" / "models" / "frankenstein"
RUN_ROOT = CAMPAIGN_ROOT / "records" / "runs" / "frankenstein"

DEFAULT_MODULE_META = (
    EVIDENCE_ROOT / "transfer-module" / "FRANKENSTEIN_TRAINING_FREE_MODULE.json"
)
DEFAULT_MODULE_BIN = (
    EVIDENCE_ROOT / "transfer-module" / "FRANKENSTEIN_TRAINING_FREE_MODULE.raw_module"
)
DEFAULT_GLM_SUBSPACE = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/frankenstein/glm-subspace"
)
DEFAULT_HANDOFF_PATH = EVIDENCE_ROOT / HANDOFF_CONTRACT_NAME

PROTO_RUN_RECEIPT_SCHEMA = "hawking.frankenstein.proto_frankenstein_run.v1"
PROTO_ARTIFACT_SCHEMA = "hawking.frankenstein.proto_artifact_descriptor.v1"
FORWARD_ATTEMPT_SCHEMA = "hawking.frankenstein.proto_forward_attempt.v1"
RUN_RECEIPT_NAME = "PROTO_FRANKENSTEIN_RUN_RECEIPT.json"
PROTO_ARTIFACT_NAME = "PROTO_FRANKENSTEIN_ARTIFACT.json"

# Env hook for when the forward lane exposes a callable.
# Format: "package.module:callable_name"
FORWARD_ENV_VAR = "DEEPSEEK_FORWARD_CALLABLE"

CLAIM_BOUNDARY = {
    **dict(ABLATION_CLAIM_BOUNDARY),
    "proto_output_raw_not_gravity": True,
    "kimi_strategic_bridge_preserved": True,
    "forward_capability_measurement_requires_callable": True,
    "dry_run_does_not_validate_capability": True,
}


class ProtoRunError(RuntimeError):
    """PROTO full-run pipeline failed closed."""


class ForwardGatePending(ProtoRunError):
    """Student forward is not available; capability measurement refused."""


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
        raise ProtoRunError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise ProtoRunError(f"{label} must be a regular non-symlink file")


def _ensure_dir(path: Path) -> None:
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
            raise ProtoRunError(f"not a safe directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> str:
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    encoded = raw.encode("utf-8")
    if path.exists():
        _regular_file(path, f"existing output {path}")
        existing = path.read_bytes()
        if existing != encoded:
            raise ProtoRunError(
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


def _optional_binding(path: Path | None, *, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return {
            "label": label,
            "path": str(path),
            "present": False,
            "note": "path cited but not present; binding is declarative",
        }
    if path.is_dir():
        return {
            "label": label,
            "path": str(path.resolve()),
            "present": True,
            "kind": "directory",
        }
    _regular_file(path, label)
    size = path.stat().st_size
    binding: dict[str, Any] = {
        "label": label,
        "path": str(path.resolve()),
        "present": True,
        "bytes": size,
    }
    # Hash only small governance artifacts; multi-GB dumps bind by path+size.
    if size <= 256 * 1024 * 1024:
        binding["file_sha256"] = _sha256_file(path)
    else:
        binding["file_sha256"] = None
        binding["hash_deferred"] = True
    return binding


# ---------------------------------------------------------------------------
# Forward resolution (fail closed until the GPU lane lands a callable)
# ---------------------------------------------------------------------------


def resolve_deepseek_forward(
    forward: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Resolve a callable DeepSeek student forward, or report the gate.

    Resolution order:
      1. Explicit ``forward`` callable passed by the caller / tests.
      2. ``DEEPSEEK_FORWARD_CALLABLE`` env (``module.path:attr``).
      3. Optional future import ``lab.operators.deepseek_v4_forward.student_forward``
         if present and callable.

    Returns a structured resolution record.  Never fabricates activations.
    """

    if forward is not None:
        if not callable(forward):
            raise ProtoRunError("provided forward is not callable")
        return {
            "status": "FORWARD_RESOLVED",
            "source": "explicit_argument",
            "callable": True,
            "gate": None,
        }

    env_spec = os.environ.get(FORWARD_ENV_VAR, "").strip()
    if env_spec:
        if ":" not in env_spec:
            raise ProtoRunError(
                f"{FORWARD_ENV_VAR} must be 'module.path:attr', got {env_spec!r}"
            )
        module_name, attr = env_spec.split(":", 1)
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            return {
                "status": FORWARD_GATE,
                "source": "env",
                "env_spec": env_spec,
                "callable": False,
                "gate": FORWARD_GATE,
                "detail": f"import failed: {exc}",
            }
        candidate = getattr(mod, attr, None)
        if not callable(candidate):
            return {
                "status": FORWARD_GATE,
                "source": "env",
                "env_spec": env_spec,
                "callable": False,
                "gate": FORWARD_GATE,
                "detail": f"attribute {attr!r} is not callable",
            }
        return {
            "status": "FORWARD_RESOLVED",
            "source": "env",
            "env_spec": env_spec,
            "callable": True,
            "gate": None,
            "_forward": candidate,
        }

    # Future hook once the forward lane lands a stable module.
    try:
        mod = importlib.import_module("lab.operators.deepseek_v4_forward")
    except ImportError:
        return {
            "status": FORWARD_GATE,
            "source": "none",
            "callable": False,
            "gate": FORWARD_GATE,
            "detail": (
                "no callable DeepSeek forward registered; GPU forward lane has "
                "not exposed lab.operators.deepseek_v4_forward or "
                f"{FORWARD_ENV_VAR}"
            ),
        }
    candidate = getattr(mod, "student_forward", None)
    if not callable(candidate):
        return {
            "status": FORWARD_GATE,
            "source": "lab.operators.deepseek_v4_forward",
            "callable": False,
            "gate": FORWARD_GATE,
            "detail": "module present but student_forward is not callable",
        }
    return {
        "status": "FORWARD_RESOLVED",
        "source": "lab.operators.deepseek_v4_forward",
        "callable": True,
        "gate": None,
        "_forward": candidate,
    }


def invoke_student_forward(
    *,
    forward: Callable[..., Any] | None = None,
    module: Mapping[str, Any] | None = None,
    body_path: Path | None = None,
    transplant_points: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Attempt student forward at transplant points; fail closed if gated.

    Dry-run and missing-forward paths return a sealed gate record with
    ``activations_captured=False`` and ``capability_measurement=False``.
    They never invent scores.
    """

    points = list(transplant_points or TRANSPLANT_POINT_NAMES)
    recorded_at = _utc_now()

    if dry_run:
        document = {
            "schema": FORWARD_ATTEMPT_SCHEMA,
            "recorded_at": recorded_at,
            "status": FORWARD_GATE,
            "dry_run": True,
            "activations_captured": False,
            "capability_measurement": False,
            "transplant_points_requested": points,
            "body_path": str(body_path) if body_path is not None else None,
            "module_seal_sha256": (
                module.get("seal_sha256") if module is not None else None
            ),
            "gate": {
                "name": FORWARD_GATE,
                "reason": (
                    "dry-run mode skips the student forward deliberately; "
                    "no activations or capability scores are fabricated"
                ),
            },
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
        return seal(document)

    resolution = resolve_deepseek_forward(forward)
    if resolution["status"] != "FORWARD_RESOLVED":
        document = {
            "schema": FORWARD_ATTEMPT_SCHEMA,
            "recorded_at": recorded_at,
            "status": FORWARD_GATE,
            "dry_run": False,
            "activations_captured": False,
            "capability_measurement": False,
            "transplant_points_requested": points,
            "body_path": str(body_path) if body_path is not None else None,
            "module_seal_sha256": (
                module.get("seal_sha256") if module is not None else None
            ),
            "resolution": {
                k: v for k, v in resolution.items() if not k.startswith("_")
            },
            "gate": {
                "name": FORWARD_GATE,
                "reason": resolution.get("detail")
                or "DeepSeek student forward is not available",
            },
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
        return seal(document)

    # Forward is available — call it.  The callable contract (when the lane
    # lands) is expected to accept keyword args and return a mapping with
    # activations / scores.  We do not invent a fake implementation here.
    fn = forward if forward is not None else resolution.get("_forward")
    if fn is None:
        raise ProtoRunError("forward resolved but callable missing")
    try:
        result = fn(
            module=module,
            body_path=body_path,
            transplant_points=points,
        )
    except Exception as exc:  # noqa: BLE001 — surface forward failures honestly
        document = {
            "schema": FORWARD_ATTEMPT_SCHEMA,
            "recorded_at": recorded_at,
            "status": "FORWARD_FAILED",
            "dry_run": False,
            "activations_captured": False,
            "capability_measurement": False,
            "transplant_points_requested": points,
            "body_path": str(body_path) if body_path is not None else None,
            "error": f"{type(exc).__name__}: {exc}",
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
        return seal(document)

    if not isinstance(result, Mapping):
        raise ProtoRunError("forward callable must return a mapping")
    activations = result.get("activations")
    scores = result.get("scores")
    document = {
        "schema": FORWARD_ATTEMPT_SCHEMA,
        "recorded_at": recorded_at,
        "status": "FORWARD_EXECUTED",
        "dry_run": False,
        "activations_captured": activations is not None,
        "capability_measurement": scores is not None,
        "transplant_points_requested": points,
        "body_path": str(body_path) if body_path is not None else None,
        "module_seal_sha256": (
            module.get("seal_sha256") if module is not None else None
        ),
        "resolution": {
            k: v for k, v in resolution.items() if not k.startswith("_")
        },
        "forward_result_keys": sorted(result.keys()),
        "scores_present": scores is not None,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "extra": {
            k: v
            for k, v in result.items()
            if k not in {"activations"}  # keep receipt JSON-serializable
        },
    }
    return seal(document)


# ---------------------------------------------------------------------------
# Module load + PROTO composition
# ---------------------------------------------------------------------------


def load_sealed_glm_transfer_module(
    meta_path: Path | str,
    module_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load the sealed training-free GLM math transfer module."""

    meta = Path(meta_path)
    _regular_file(meta, "transfer module meta")
    bin_path = Path(module_path) if module_path is not None else None
    if bin_path is not None:
        _regular_file(bin_path, "transfer module binary")
    module = load_transfer_module(meta, bin_path)
    if module.get("schema") != TRANSFER_MODULE_SCHEMA:
        raise ProtoRunError(
            f"transfer module schema mismatch: {module.get('schema')!r}"
        )
    if module.get("trained") is True:
        raise ProtoRunError("transfer module claims trained=true; refuse")
    if module.get("bridge") not in (None, "GLM_MATH_BRIDGE"):
        raise ProtoRunError(
            f"stage-1 module must be GLM_MATH_BRIDGE, got {module.get('bridge')!r}"
        )
    return module


def compose_proto_artifact(
    *,
    module: Mapping[str, Any],
    body_path: Path,
    out_dir: Path,
    student_layers: Sequence[int] | None = None,
    bridge_path: Path | None = DEFAULT_BRIDGE_PATH,
    transplant_path: Path | None = DEFAULT_TRANSPLANT_PATH,
) -> dict[str, Any]:
    """Compose per-transplant-point modules onto the read-only DeepSeek body.

    Produces a **raw** PROTO artifact descriptor (not gravity-compressed).
    The body file is never rewritten; composition is structural reference +
    sealed apply plan from ``frankenstein_transfer_apply``.
    """

    _ensure_dir(out_dir)
    apply_dir = out_dir / "transfer-apply"
    applied = frankenstein_transfer_apply(
        module=module,
        body_path=Path(body_path),
        out_dir=apply_dir,
        student_layers=student_layers,
    )

    # Adapter blocks under GLM_MATH_BRIDGE only — Kimi points stay empty.
    blocks: list[dict[str, Any]] = []
    for row in module.get("per_transplant_point") or []:
        point = row.get("transplant_point")
        if point is None:
            continue
        blocks.append(
            {
                "bridge": "GLM_MATH_BRIDGE",
                "transplant_point": point,
                "apply_mode": row.get("apply_mode"),
                "steering_scale": row.get("steering_scale"),
                "direct_weight_transplant": False,
                "kimi_strategic_bridge": False,
            }
        )
    # Explicitly record Kimi points as preserved / no stage-1 adapter.
    for point in KIMI_PRESERVED_TRANSPLANT_POINTS:
        # Points may appear in both lists; the preservation flag is the contract.
        pass

    manifest = build_adapter_manifest(blocks=blocks)

    body = Path(body_path)
    body_binding = _optional_binding(body, label="deepseek_body")
    module_meta_binding = None
    # Prefer sealed meta path from module if known via apply doc.
    artifact = {
        "schema": PROTO_ARTIFACT_SCHEMA,
        "name": "PROTO_FRANKENSTEIN",
        "recorded_at": _utc_now(),
        "kind": "raw_proto_composition_descriptor",
        "status": "STRUCTURAL_PROTO_COMPOSED_VALIDATION_PENDING",
        "gravity_compressed": False,
        "byte_merge": False,
        "direct_weight_transplant": False,
        "trained": False,
        "composition": {
            "mode": "per_transplant_point_residual_modules_on_read_only_body",
            "formula_residual": "a_out = a_s + a_s @ A + b_steer",
            "formula_router": "logits_out = logits_s + router_bias",
            "body_rewritten": False,
            "reversible": True,
            "removable": True,
        },
        "student_body": {
            "family": DEEPSEEK_V4_FLASH["family"],
            "repository": DEEPSEEK_V4_FLASH["repository"],
            "revision": DEEPSEEK_V4_FLASH["revision"],
            "hidden_size": DEEPSEEK_V4_FLASH["hidden_size"],
            "num_hidden_layers": DEEPSEEK_V4_FLASH["num_hidden_layers"],
            "path": str(body),
            "present": body.exists(),
            "read_only": True,
            "binding": body_binding,
        },
        "math_donor": {
            "family": GLM_5_2["family"],
            "repository": GLM_5_2["repository"],
            "revision": GLM_5_2["revision"],
            "hidden_size": GLM_5_2["hidden_size"],
        },
        "transfer_module": {
            "schema": module.get("schema"),
            "status": module.get("status"),
            "capability_status": module.get("capability_status"),
            "capability_claim": module.get("capability_claim"),
            "seal_sha256": module.get("seal_sha256"),
            "bridge": module.get("bridge") or "GLM_MATH_BRIDGE",
            "trained": module.get("trained"),
            "subspace_rank": (module.get("subspace") or {}).get("rank"),
            "forward_gate": module.get("forward_gate") or FORWARD_GATE,
        },
        "structural_apply": {
            "status": applied.get("status"),
            "apply_path": applied.get("apply_path"),
            "apply_seal_sha256": applied.get("apply_seal_sha256"),
            "packet_path": applied.get("packet_path"),
            "validation_status": applied.get("validation_status"),
            "capability_claim": applied.get("capability_claim"),
        },
        "transplant_applications": list(module.get("per_transplant_point") or []),
        "adapter_manifest": manifest,
        "bridges": {
            "active_stage1": "GLM_MATH_BRIDGE",
            "preserved_untouched_for_stage2": "KIMI_STRATEGIC_BRIDGE",
            "all_declared": list(BRIDGES),
            "kimi_points_preserved": list(KIMI_PRESERVED_TRANSPLANT_POINTS),
            "kimi_stage1_status": "PRESERVED_UNTOUCHED",
        },
        "bindings": {
            "latent_bridge_contract": _optional_binding(
                bridge_path, label="DSV4F_LATENT_BRIDGE_CONTRACT"
            ),
            "transplant_points": _optional_binding(
                transplant_path, label="DSV4F_TRANSPLANT_POINTS"
            ),
        },
        "validation": {
            "status": FORWARD_GATE,
            "math_bench": "NOT_RUN",
            "capability_claim": False,
            "note": (
                "Structural PROTO composition only.  Live capability measurement "
                "requires the DeepSeek student forward."
            ),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    sealed = seal(artifact)
    path = out_dir / PROTO_ARTIFACT_NAME
    _atomic_write_json(path, sealed)
    return {
        "status": sealed["status"],
        "artifact_path": str(path),
        "artifact_seal_sha256": sealed["seal_sha256"],
        "apply": applied,
        "document": sealed,
        "gravity_compressed": False,
        "capability_claim": False,
        "validation_status": FORWARD_GATE,
    }


# ---------------------------------------------------------------------------
# Ablation stage (shard / fixture only)
# ---------------------------------------------------------------------------


def run_ablation_stage(
    *,
    scores_fixture: Path | None,
    transfer_module_id: str | None = None,
    shard_ids: Sequence[str] | None = None,
    bench_scope: str = "SHARD",
    secondary_tolerance: float = DEFAULT_SECONDARY_TOLERANCE,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run A-vs-B ablation on sealed shard/fixture scores (never full model)."""

    if scores_fixture is None:
        return {
            "status": "ABLATION_SKIPPED_NO_FIXTURE",
            "verdict": None,
            "reject_rule_fired": None,
            "note": (
                "No sealed capability fixture provided.  Ablation deferred.  "
                "Live score generation requires the student forward "
                f"({FORWARD_GATE})."
            ),
            "forward_gate": FORWARD_GATE,
        }

    fixture_path = Path(scores_fixture)
    _regular_file(fixture_path, "capability scores fixture")

    try:
        request = ShardBenchRequest(
            scope=bench_scope,
            shard_ids=tuple(shard_ids or ("proto-run-shard",)),
            scores_fixture_path=fixture_path,
        )
        bench = run_shard_bench(request)
    except FullModelBenchRefused as exc:
        raise ProtoRunError(f"full-model bench refused: {exc}") from exc

    ablation = bench.get("ablation") or run_avb_from_fixture(
        fixture_path, secondary_tolerance=secondary_tolerance
    )
    report_path = None
    if out_dir is not None:
        _ensure_dir(out_dir)
        report_path = out_dir / f"STAGE1_AVB_ABLATION__{ablation.get('fixture_id', 'run')}.json"
        write_ablation_report(report_path, ablation)

    return {
        "status": "ABLATION_COMPLETE",
        "verdict": ablation.get("verdict"),
        "reject_rule_fired": ablation.get("reject_rule_fired"),
        "additive_not_subtractive": ablation.get("additive_not_subtractive"),
        "math_mean_gain": (ablation.get("math") or {}).get("mean_gain"),
        "secondary_verdict": (ablation.get("secondary") or {}).get("verdict"),
        "fixture_id": ablation.get("fixture_id"),
        "transfer_module_id": ablation.get("transfer_module_id") or transfer_module_id,
        "bench_scope": bench.get("bench_scope"),
        "bench_seal_sha256": bench.get("seal_sha256"),
        "ablation_seal_sha256": ablation.get("seal_sha256"),
        "report_path": str(report_path) if report_path is not None else None,
        "ablation": ablation,
        "shard_bench": bench,
        "note": (
            "Ablation uses sealed shard/fixture scores.  This is governance "
            "evidence, not a live forward-measured capability claim, unless "
            "the fixture itself was produced by a real forward."
        ),
    }


# ---------------------------------------------------------------------------
# Training-path guard (AST; this module + transfer)
# ---------------------------------------------------------------------------


def assert_no_training_path() -> dict[str, Any]:
    """Static guarantee: proto_run never imports training stacks or calls train APIs."""

    import lab.operators.frankenstein_proto_run as self_mod

    source = Path(self_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    hits: list[dict[str, Any]] = []
    imports_torch = False
    imports_optimizer = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                if node.module:
                    names = [node.module]
            for name in names:
                if name == "torch" or name.startswith("torch."):
                    imports_torch = True
                    hits.append({"kind": "import", "module": name})
                if "optim" in name.split("."):
                    imports_optimizer = True
                    hits.append({"kind": "import", "module": name})
        if isinstance(node, ast.Call):
            func = node.func
            attr = None
            if isinstance(func, ast.Attribute):
                attr = func.attr
            elif isinstance(func, ast.Name):
                attr = func.id
            if attr in {"backward", "zero_grad"}:
                hits.append(
                    {
                        "kind": "call",
                        "name": attr,
                        "lineno": getattr(node, "lineno", None),
                    }
                )
    transfer_guard = transfer_assert_no_training_path()
    return {
        "training_path_present": len(hits) > 0
        or bool(transfer_guard.get("training_path_present")),
        "hits": hits,
        "imports_torch": imports_torch or bool(transfer_guard.get("imports_torch")),
        "imports_optimizer": imports_optimizer
        or bool(transfer_guard.get("imports_optimizer")),
        "transfer_guard": transfer_guard,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def build_proto_run_receipt(
    *,
    stages: Mapping[str, Any],
    dry_run: bool,
    out_dir: Path,
    transfer_module_id: str,
    body_path: Path,
    module_meta_path: Path | None,
    scores_fixture: Path | None,
    glm_subspace_path: Path | None,
    bridge_path: Path | None,
    transplant_path: Path | None,
    handoff_path: Path | None,
    wall_s: float,
    free_bytes_start: int | None,
    free_bytes_end: int | None,
) -> dict[str, Any]:
    """Seal the end-to-end PROTO_FRANKENSTEIN run receipt."""

    forward = stages.get("forward") or {}
    ablation = stages.get("ablation") or {}
    compose = stages.get("compose") or {}
    load = stages.get("load") or {}

    forward_status = forward.get("status") or FORWARD_GATE
    activations_ok = bool(forward.get("activations_captured"))
    capability_from_forward = bool(forward.get("capability_measurement"))

    ablation_verdict = ablation.get("verdict")
    reject_fired = ablation.get("reject_rule_fired")

    # Overall verdict: capability ACCEPT requires a real forward measurement.
    # Fixture ablation can still REJECT a candidate (governance).
    if reject_fired is True or ablation_verdict == "REJECT":
        overall = "REJECT"
        reason = (
            "secondary capability regression beyond sealed tolerance on "
            "shard/fixture scores (math gain cannot override)"
        )
    elif dry_run or forward_status == FORWARD_GATE or not capability_from_forward:
        overall = "FORWARD_GATED"
        reason = (
            f"pipeline wired; student forward status={forward_status!r}; "
            "no capability validation claimed (fail closed, not fabricated)"
        )
    elif ablation_verdict == "ACCEPT" and capability_from_forward:
        overall = "ACCEPT"
        reason = (
            "all secondary gates held on forward-measured scores; "
            "math deltas recorded as optimization signal"
        )
    elif ablation_verdict is None:
        overall = "FORWARD_GATED" if not capability_from_forward else "SCOPED_ONLY"
        reason = "ablation fixture not provided; forward gate still honest"
    else:
        overall = str(ablation_verdict)
        reason = f"ablation verdict={ablation_verdict}"

    adapter_bytes = 0
    if module_meta_path is not None and module_meta_path.is_file():
        bin_candidate = module_meta_path.with_name(
            "FRANKENSTEIN_TRAINING_FREE_MODULE.raw_module"
        )
        if bin_candidate.is_file():
            adapter_bytes = bin_candidate.stat().st_size

    runtime_storage = build_runtime_storage_accounting(
        adapter_archive_bytes=adapter_bytes,
        working_set_bytes=0,
        free_bytes_at_seal=free_bytes_end,
        notes=(
            "PROTO run accounting.  Full-model TPS unmeasured until forward "
            "lane lands.  Adapter archive is the sealed raw transfer module."
        ),
    )

    handoff_binding = _optional_binding(
        handoff_path or DEFAULT_HANDOFF_PATH, label="PROTO_TO_KIMI_HANDOFF_CONTRACT"
    )

    receipt = {
        "schema": PROTO_RUN_RECEIPT_SCHEMA,
        "name": RUN_RECEIPT_NAME,
        "recorded_at": _utc_now(),
        "stage": 1,
        "dry_run": bool(dry_run),
        "status": overall,
        "reason": reason,
        "verdict": overall,
        "reject_rule_fired": bool(reject_fired) if reject_fired is not None else False,
        "additive_not_subtractive": ablation.get("additive_not_subtractive"),
        "forward_gate": FORWARD_GATE,
        "forward_status": forward_status,
        "activations_captured": activations_ok,
        "capability_validated": capability_from_forward and overall == "ACCEPT",
        "capability_claim": False
        if (dry_run or not capability_from_forward)
        else (overall == "ACCEPT"),
        "gravity_compressed": False,
        "trained": False,
        "training_free": True,
        "arms": {"A": ARM_A, "B": ARM_B},
        "math_domains": list(MATH_DOMAINS),
        "secondary_capabilities": list(SECONDARY_CAPABILITIES),
        "transfer_module_id": transfer_module_id,
        "stages": {
            "load_module": {
                "status": load.get("status"),
                "module_seal_sha256": load.get("module_seal_sha256"),
                "meta_path": load.get("meta_path"),
                "bridge": load.get("bridge"),
                "trained": load.get("trained"),
            },
            "compose_proto": {
                "status": compose.get("status"),
                "artifact_path": compose.get("artifact_path"),
                "artifact_seal_sha256": compose.get("artifact_seal_sha256"),
                "apply_path": (compose.get("apply") or {}).get("apply_path"),
                "gravity_compressed": False,
                "validation_status": compose.get("validation_status"),
            },
            "forward_activations": {
                "status": forward_status,
                "dry_run": dry_run,
                "activations_captured": activations_ok,
                "capability_measurement": capability_from_forward,
                "seal_sha256": forward.get("seal_sha256"),
                "gate": forward.get("gate") or {"name": FORWARD_GATE},
            },
            "ablation_avb": {
                "status": ablation.get("status"),
                "verdict": ablation_verdict,
                "reject_rule_fired": reject_fired,
                "math_mean_gain": ablation.get("math_mean_gain"),
                "secondary_verdict": ablation.get("secondary_verdict"),
                "fixture_id": ablation.get("fixture_id"),
                "bench_scope": ablation.get("bench_scope"),
                "report_path": ablation.get("report_path"),
                "ablation_seal_sha256": ablation.get("ablation_seal_sha256"),
            },
        },
        "math_domain_gain": {
            "mean_gain": ablation.get("math_mean_gain"),
            "domains": ((ablation.get("ablation") or {}).get("math") or {}).get(
                "domains"
            ),
            "source": "sealed_shard_fixture"
            if scores_fixture is not None
            else "none",
            "live_forward_measured": capability_from_forward,
        },
        "secondary_non_regression": {
            "verdict": ablation.get("secondary_verdict"),
            "reject_rule_fired": reject_fired,
            "domains": ((ablation.get("ablation") or {}).get("secondary") or {}).get(
                "domains"
            ),
            "source": "sealed_shard_fixture"
            if scores_fixture is not None
            else "none",
        },
        "runtime_storage": runtime_storage,
        "provenance": {
            "body": _optional_binding(Path(body_path), label="deepseek_body"),
            "transfer_module_meta": _optional_binding(
                module_meta_path, label="transfer_module_meta"
            )
            if module_meta_path is not None
            else None,
            "scores_fixture": _optional_binding(
                scores_fixture, label="capability_scores_fixture"
            )
            if scores_fixture is not None
            else None,
            "glm_subspace": _optional_binding(
                glm_subspace_path, label="glm_math_subspace"
            )
            if glm_subspace_path is not None
            else None,
            "latent_bridge_contract": _optional_binding(
                bridge_path, label="DSV4F_LATENT_BRIDGE_CONTRACT"
            ),
            "transplant_points": _optional_binding(
                transplant_path, label="DSV4F_TRANSPLANT_POINTS"
            ),
            "student": {
                "repository": DEEPSEEK_V4_FLASH["repository"],
                "revision": DEEPSEEK_V4_FLASH["revision"],
            },
            "math_donor": {
                "repository": GLM_5_2["repository"],
                "revision": GLM_5_2["revision"],
            },
        },
        "kimi_handoff": {
            "contract_name": HANDOFF_CONTRACT_NAME,
            "path": str(handoff_path or DEFAULT_HANDOFF_PATH),
            "binding": handoff_binding,
            "bridge_preserved": "KIMI_STRATEGIC_BRIDGE",
            "stage1_must_not_mutate": True,
            "pointer": "stage-2 Kimi strategic graft uses PROTO_TO_KIMI_HANDOFF_CONTRACT",
        },
        "wall_s": float(wall_s),
        "disk": {
            "free_bytes_start": free_bytes_start,
            "free_bytes_end": free_bytes_end,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "honest_status": (
            "PROTO pipeline stages executed as far as the forward gate allows. "
            "Capability is NOT validated without a real student forward. "
            "Ablation on sealed fixtures is governance evidence only."
        ),
    }
    sealed = seal(receipt)
    path = out_dir / RUN_RECEIPT_NAME
    _atomic_write_json(path, sealed)
    return {
        "receipt_path": str(path),
        "seal_sha256": sealed["seal_sha256"],
        "verdict": sealed["verdict"],
        "status": sealed["status"],
        "document": sealed,
    }


def run_proto_frankenstein(
    *,
    module_meta: Path | str | None = None,
    module_bin: Path | str | None = None,
    body_path: Path | str = DEFAULT_BODY_PATH,
    out_dir: Path | str | None = None,
    scores_fixture: Path | str | None = None,
    dry_run: bool = False,
    forward: Callable[..., Any] | None = None,
    student_layers: Sequence[int] | None = None,
    bridge_path: Path | str | None = DEFAULT_BRIDGE_PATH,
    transplant_path: Path | str | None = DEFAULT_TRANSPLANT_PATH,
    glm_subspace_path: Path | str | None = DEFAULT_GLM_SUBSPACE,
    handoff_path: Path | str | None = None,
    transfer_module_id: str | None = None,
    ensure_handoff: bool = True,
    shard_ids: Sequence[str] | None = None,
    secondary_tolerance: float = DEFAULT_SECONDARY_TOLERANCE,
    # Test hook: pre-built module dict (skips disk load).
    module: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the PROTO_FRANKENSTEIN full-run pipeline.

    When ``dry_run=True``, every stage except the student forward executes;
    the forward stage is recorded as ``DEEPSEEK_FORWARD_PENDING`` without
    fabricating activations or capability scores.
    """

    t0 = time.perf_counter()
    out = Path(out_dir) if out_dir is not None else (EVIDENCE_ROOT / "proto-run")
    _ensure_dir(out)
    body = Path(body_path)
    bridge = Path(bridge_path) if bridge_path is not None else None
    transplant = Path(transplant_path) if transplant_path is not None else None
    subspace = Path(glm_subspace_path) if glm_subspace_path is not None else None
    scores = Path(scores_fixture) if scores_fixture is not None else None

    free_start = None
    try:
        free_start = int(shutil.disk_usage(out).free)
    except OSError:
        pass

    guard = assert_no_training_path()
    if guard["training_path_present"] or guard["imports_torch"] or guard["imports_optimizer"]:
        raise ProtoRunError(f"training path detected: {guard}")

    # --- (a) load sealed GLM math transfer module ---
    meta_path: Path | None = None
    if module is not None:
        loaded = dict(module)
        load_stage = {
            "status": "MODULE_PROVIDED_IN_MEMORY",
            "module_seal_sha256": loaded.get("seal_sha256"),
            "meta_path": None,
            "bridge": loaded.get("bridge") or "GLM_MATH_BRIDGE",
            "trained": loaded.get("trained"),
            "capability_status": loaded.get("capability_status"),
        }
    else:
        meta_path = Path(module_meta) if module_meta is not None else DEFAULT_MODULE_META
        bin_path = Path(module_bin) if module_bin is not None else None
        if not meta_path.is_file():
            raise ProtoRunError(
                f"sealed transfer module meta not found: {meta_path}; "
                "run training-free transfer first or pass --module-meta"
            )
        loaded = load_sealed_glm_transfer_module(meta_path, bin_path)
        load_stage = {
            "status": "MODULE_LOADED",
            "module_seal_sha256": loaded.get("seal_sha256"),
            "meta_path": str(meta_path),
            "bridge": loaded.get("bridge") or "GLM_MATH_BRIDGE",
            "trained": loaded.get("trained"),
            "capability_status": loaded.get("capability_status"),
        }

    module_id = transfer_module_id or (
        f"proto-{loaded.get('seal_sha256', 'unknown')[:12]}"
        if loaded.get("seal_sha256")
        else "proto-frankenstein-module"
    )

    # --- (b) compose PROTO artifact (raw) ---
    compose = compose_proto_artifact(
        module=loaded,
        body_path=body,
        out_dir=out,
        student_layers=student_layers,
        bridge_path=bridge,
        transplant_path=transplant,
    )

    # --- (c) forward activations (gated) ---
    forward_doc = invoke_student_forward(
        forward=forward,
        module=loaded,
        body_path=body,
        dry_run=dry_run,
    )
    # Persist forward attempt under out_dir for auditability.
    forward_path = out / "PROTO_FORWARD_ATTEMPT.json"
    _atomic_write_json(forward_path, forward_doc)

    # --- (d) A-vs-B ablation on shard fixtures ---
    ablation = run_ablation_stage(
        scores_fixture=scores,
        transfer_module_id=module_id,
        shard_ids=shard_ids,
        out_dir=out / "ablation",
        secondary_tolerance=secondary_tolerance,
    )

    # --- Kimi handoff pointer (must exist; seal if missing and allowed) ---
    handoff = Path(handoff_path) if handoff_path is not None else DEFAULT_HANDOFF_PATH
    if ensure_handoff and not handoff.is_file():
        write_handoff_contract(
            handoff.parent,
            bridge_path=bridge or DEFAULT_BRIDGE_PATH,
            transplant_path=transplant or DEFAULT_TRANSPLANT_PATH,
        )

    free_end = None
    try:
        free_end = int(shutil.disk_usage(out).free)
    except OSError:
        pass

    wall_s = time.perf_counter() - t0
    stages = {
        "load": load_stage,
        "compose": compose,
        "forward": forward_doc,
        "ablation": ablation,
    }
    sealed_run = build_proto_run_receipt(
        stages=stages,
        dry_run=dry_run,
        out_dir=out,
        transfer_module_id=module_id,
        body_path=body,
        module_meta_path=meta_path,
        scores_fixture=scores,
        glm_subspace_path=subspace,
        bridge_path=bridge,
        transplant_path=transplant,
        handoff_path=handoff,
        wall_s=wall_s,
        free_bytes_start=free_start,
        free_bytes_end=free_end,
    )

    return {
        "status": sealed_run["status"],
        "verdict": sealed_run["verdict"],
        "dry_run": dry_run,
        "forward_status": forward_doc.get("status"),
        "forward_gate": FORWARD_GATE,
        "capability_validated": sealed_run["document"]["capability_validated"],
        "capability_claim": sealed_run["document"]["capability_claim"],
        "gravity_compressed": False,
        "reject_rule_fired": sealed_run["document"]["reject_rule_fired"],
        "ablation_verdict": ablation.get("verdict"),
        "receipt_path": sealed_run["receipt_path"],
        "receipt_seal_sha256": sealed_run["seal_sha256"],
        "artifact_path": compose.get("artifact_path"),
        "artifact_seal_sha256": compose.get("artifact_seal_sha256"),
        "forward_attempt_path": str(forward_path),
        "stages": {
            "load": load_stage,
            "compose": {
                "status": compose.get("status"),
                "artifact_path": compose.get("artifact_path"),
            },
            "forward": {
                "status": forward_doc.get("status"),
                "activations_captured": forward_doc.get("activations_captured"),
                "capability_measurement": forward_doc.get("capability_measurement"),
            },
            "ablation": {
                "status": ablation.get("status"),
                "verdict": ablation.get("verdict"),
                "reject_rule_fired": ablation.get("reject_rule_fired"),
            },
        },
        "training_path_guard": guard,
        "document": sealed_run["document"],
    }


def make_reject_fixture_document(
    *,
    fixture_id: str = "math_up_coding_down_REJECT",
    transfer_module_id: str = "proto-run-reject-fixture",
) -> dict[str, Any]:
    """Build the sealed math-up / coding-down fixture used by dry-run demos."""

    base_math = {d: 0.70 for d in MATH_DOMAINS}
    base_sec = {d: 0.70 for d in SECONDARY_CAPABILITIES}
    proto_math = {d: 0.90 for d in MATH_DOMAINS}
    proto_sec = {d: 0.70 for d in SECONDARY_CAPABILITIES}
    proto_sec["coding_and_repository_work"] = 0.60
    return make_capability_fixture(
        fixture_id=fixture_id,
        bench_scope="SHARD",
        base_math=base_math,
        base_secondary=base_sec,
        proto_math=proto_math,
        proto_secondary=proto_sec,
        transfer_module_id=transfer_module_id,
        meta={
            "intent": "prove reject rule fires on math-up/coding-down",
            "expected_verdict": "REJECT",
        },
    )


def make_accept_fixture_document(
    *,
    fixture_id: str = "math_gain_secondaries_hold",
    transfer_module_id: str = "proto-run-accept-fixture",
) -> dict[str, Any]:
    base_math = {d: 0.70 for d in MATH_DOMAINS}
    base_sec = {d: 0.70 for d in SECONDARY_CAPABILITIES}
    proto_math = {d: 0.85 for d in MATH_DOMAINS}
    proto_sec = {d: 0.70 for d in SECONDARY_CAPABILITIES}
    proto_sec["tool_use"] = 0.69  # inside tolerance
    return make_capability_fixture(
        fixture_id=fixture_id,
        bench_scope="SHARD",
        base_math=base_math,
        base_secondary=base_sec,
        proto_math=proto_math,
        proto_secondary=proto_sec,
        transfer_module_id=transfer_module_id,
        meta={"expected_verdict": "ACCEPT"},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "PROTO_FRANKENSTEIN full-run orchestrator: load sealed GLM transfer, "
            "compose raw PROTO artifact, gate on DeepSeek forward, run A-vs-B "
            "ablation, seal run receipt."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--module-meta",
            type=Path,
            default=DEFAULT_MODULE_META,
            help="Path to FRANKENSTEIN_TRAINING_FREE_MODULE.json",
        )
        p.add_argument(
            "--module-bin",
            type=Path,
            default=None,
            help="Path to .raw_module (default: beside meta)",
        )
        p.add_argument("--body-path", type=Path, default=DEFAULT_BODY_PATH)
        p.add_argument(
            "--out-dir",
            type=Path,
            default=None,
            help="Output directory for PROTO artifact + run receipt",
        )
        p.add_argument(
            "--scores-fixture",
            type=Path,
            default=None,
            help="Sealed capability score fixture for A-vs-B ablation",
        )
        p.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE_PATH)
        p.add_argument("--transplant", type=Path, default=DEFAULT_TRANSPLANT_PATH)
        p.add_argument("--glm-subspace", type=Path, default=DEFAULT_GLM_SUBSPACE)
        p.add_argument("--handoff", type=Path, default=None)
        p.add_argument("--transfer-module-id", default=None)
        p.add_argument(
            "--secondary-tolerance",
            type=float,
            default=DEFAULT_SECONDARY_TOLERANCE,
        )
        p.add_argument(
            "--shard-id",
            action="append",
            default=[],
            dest="shard_ids",
            help="Shard id for bench scope (repeatable)",
        )

    p_dry = sub.add_parser(
        "dry-run",
        help=(
            "Execute the full pipeline EXCEPT the forward; prove wiring and "
            "fail closed on DEEPSEEK_FORWARD_PENDING (no fake validation)"
        ),
    )
    add_common(p_dry)
    p_dry.add_argument(
        "--write-reject-fixture",
        action="store_true",
        help="Write a math-up/coding-down fixture into out-dir and ablate it",
    )

    p_run = sub.add_parser(
        "run",
        help=(
            "Full run: same as dry-run but attempts the student forward "
            "(still fails closed if forward is unavailable)"
        ),
    )
    add_common(p_run)

    p_guard = sub.add_parser(
        "guard",
        help="Assert no gradient/optimizer/training path in this module",
    )

    p_forward = sub.add_parser(
        "forward-status",
        help="Report whether a DeepSeek forward callable is currently resolvable",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "guard":
            result = assert_no_training_path()
            print(json.dumps(result, sort_keys=True, indent=2))
            return 1 if result["training_path_present"] else 0

        if args.command == "forward-status":
            resolution = resolve_deepseek_forward()
            public = {k: v for k, v in resolution.items() if not k.startswith("_")}
            print(json.dumps(public, sort_keys=True, indent=2))
            return 0 if resolution.get("status") == "FORWARD_RESOLVED" else 2

        if args.command in {"dry-run", "run"}:
            out_dir = args.out_dir
            if out_dir is None:
                stamp = "dry-run" if args.command == "dry-run" else "full-run"
                out_dir = EVIDENCE_ROOT / "proto-run" / stamp

            scores = args.scores_fixture
            if (
                args.command == "dry-run"
                and getattr(args, "write_reject_fixture", False)
                and scores is None
            ):
                _ensure_dir(Path(out_dir))
                fixture_path = Path(out_dir) / "SHARD_FIXTURE__math_up_coding_down_REJECT.json"
                fixture = make_reject_fixture_document()
                fixture_path.write_text(
                    json.dumps(fixture, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                scores = fixture_path

            result = run_proto_frankenstein(
                module_meta=args.module_meta,
                module_bin=args.module_bin,
                body_path=args.body_path,
                out_dir=out_dir,
                scores_fixture=scores,
                dry_run=(args.command == "dry-run"),
                bridge_path=args.bridge,
                transplant_path=args.transplant,
                glm_subspace_path=args.glm_subspace,
                handoff_path=args.handoff,
                transfer_module_id=args.transfer_module_id,
                shard_ids=args.shard_ids or None,
                secondary_tolerance=args.secondary_tolerance,
            )
            # Public summary (no full document dump).
            summary = {
                "command": args.command,
                "status": result["status"],
                "verdict": result["verdict"],
                "dry_run": result["dry_run"],
                "forward_status": result["forward_status"],
                "forward_gate": result["forward_gate"],
                "capability_validated": result["capability_validated"],
                "capability_claim": result["capability_claim"],
                "reject_rule_fired": result["reject_rule_fired"],
                "ablation_verdict": result["ablation_verdict"],
                "gravity_compressed": result["gravity_compressed"],
                "receipt_path": result["receipt_path"],
                "receipt_seal_sha256": result["receipt_seal_sha256"],
                "artifact_path": result["artifact_path"],
                "stages": result["stages"],
            }
            print(json.dumps(summary, sort_keys=True, indent=2))
            # Exit codes:
            #   0 — pipeline completed its honest path (including FORWARD_GATED)
            #   2 — ablation REJECT
            #   3 — unexpected proto error already raised
            if result["verdict"] == "REJECT":
                return 2
            return 0

        raise ProtoRunError(f"unknown command: {args.command}")
    except FullModelBenchRefused as exc:
        print(json.dumps({"error": "FULL_MODEL_BENCH_REFUSED", "detail": str(exc)}))
        return 3
    except ProtoRunError as exc:
        print(json.dumps({"error": "PROTO_RUN_ERROR", "detail": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
