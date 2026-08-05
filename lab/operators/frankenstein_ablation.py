#!/usr/bin/env python3.12
"""Stage-1 A-vs-B capability ablation for PROTO_FRANKENSTEIN.

Governs whether a candidate PROTO transfer module (DeepSeek-V4-Flash body +
GLM math inheritance) may exist as an additive capability delta over
BASE_DSV4F.

Hard policy (sealed, non-negotiable):
  * Maximize GLM mathematical inheritance (gain domains).
  * Preserve every secondary capability as a HARD gate: any regression beyond
    the sealed tolerance yields REJECT, regardless of math gain.
  * Training-free: this module evaluates sealed score fixtures / transfer
    shards.  It never runs a gradient, loss, or optimizer.
  * Shard-level only: full-model benches are fail-closed until gravity
    compression and a complete runtime exist.

Frozen comparison arms owned by this lane:
  A = BASE_DSV4F
  B = PROTO = DSV4F + GLM math
C (FINAL = PROTO + KIMI) and D (RAMANUJAN) are later stages and are named
here only for cross-stage provenance.

Does not edit transfer machinery, crates/, or ramanujan/.  Does not consume
or simulate Kimi.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators.frankenstein_fusion_op import (
    BRIDGES,
    DEEPSEEK_V4_FLASH,
    GLM_5_2,
    TRANSPLANT_POINT_NAMES,
)
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "workspace"
CAMPAIGN_ROOT = WORKSPACE_ROOT / "campaign"
EVIDENCE_ROOT = CAMPAIGN_ROOT / "evidence"
FRANK_EVIDENCE = EVIDENCE_ROOT / "models" / "frankenstein"
DEFAULT_GLM_SUBSPACE = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/frankenstein/glm-subspace"
)
DEFAULT_BRIDGE_PATH = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_LATENT_BRIDGE_CONTRACT.json"
)
DEFAULT_TRANSPLANT_PATH = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "runs/deepseek-v4/child-baseline-v3/DSV4F_TRANSPLANT_POINTS.json"
)

# --- sealed schemas ---
ABLATION_REPORT_SCHEMA = "hawking.frankenstein.stage1_avb_ablation.v1"
SHARD_BENCH_SCHEMA = "hawking.frankenstein.stage1_shard_bench.v1"
CAPABILITY_FIXTURE_SCHEMA = "hawking.frankenstein.stage1_capability_scores.v1"
GATE_POLICY_SCHEMA = "hawking.frankenstein.stage1_gate_policy.v1"

ARM_A = "A_BASE_DSV4F"
ARM_B = "B_PROTO_DSV4F_GLM"
ARM_C_LATER = "C_FINAL_PROTO_KIMI"
ARM_D_LATER = "D_RAMANUJAN"

# Math domains: maximize inheritance (gain reported, not gated as hard reject).
MATH_DOMAINS: tuple[str, ...] = (
    "method_selection",
    "mathematical_decomposition",
    "formalization",
    "proof_repair",
    "value_ranking",
)

# Secondary capabilities: HARD non-regression gates.
SECONDARY_CAPABILITIES: tuple[str, ...] = (
    "coding_and_repository_work",
    "tool_use",
    "agentic_planning",
    "long_context_reasoning",
    "repair_and_critique",
    "hcli_protocols",
    "general_knowledge_conversation",
    "runtime_tps",
    "bridge_compatibility",
    "routing_stability",
    "context_behavior",
)

# Bench scope tokens.  FULL_MODEL is always refused in this stage.
BENCH_SCOPE_SHARD = "SHARD"
BENCH_SCOPE_FIXTURE = "FIXTURE"
BENCH_SCOPE_BOUNDED = "BOUNDED_FIXTURE"
ALLOWED_BENCH_SCOPES = frozenset(
    {BENCH_SCOPE_SHARD, BENCH_SCOPE_FIXTURE, BENCH_SCOPE_BOUNDED}
)
REFUSED_BENCH_SCOPES = frozenset(
    {
        "FULL_MODEL",
        "FULL",
        "END_TO_END_MODEL",
        "COMPLETE_MODEL",
        "GRAVITY_COMPRESSED_MODEL",
        "LIVE_RUNTIME",
    }
)

# Sealed absolute tolerance on secondary scores in [0, 1] space.
# Proto must be ≥ base − tolerance on every secondary.  Math gain never
# buys a secondary regression past this line.
DEFAULT_SECONDARY_TOLERANCE = 0.02
# Math domains report gain; a soft advisory floor (not reject-gated).
DEFAULT_MATH_GAIN_ADVISORY_MIN = 0.0

CLAIM_BOUNDARY = {
    "stage": 1,
    "proto_frankenstein": "DeepSeek-V4-Flash + GLM math inheritance",
    "kimi_consumed": False,
    "kimi_simulated": False,
    "odyssey_or_ramanujan": False,
    "gravity_compression_performed": False,
    "full_model_bench_permitted": False,
    "training_performed": False,
    "gradient_or_optimizer": False,
    "weight_average_performed": False,
    "direct_weight_transplant": False,
    "capability_claim_requires_sealed_shard_scores": True,
    "lane_owns": [ARM_A, ARM_B],
    "lane_does_not_own": [ARM_C_LATER, ARM_D_LATER],
}


class AblationError(RuntimeError):
    """Input, scope, or policy gate failed closed."""


class FullModelBenchRefused(AblationError):
    """Full-model (or otherwise unbound) bench request is not allowed in stage 1."""


class CapabilityReject(AblationError):
    """Secondary capability regressed beyond sealed tolerance."""


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


def _regular_file(path: Path, label: str) -> None:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise AblationError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise AblationError(f"{label} must be a regular non-symlink file")


def _ensure_dir(path: Path) -> None:
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
            raise AblationError(f"not a safe directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> str:
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    encoded = raw.encode("utf-8")
    if path.exists():
        _regular_file(path, f"existing output {path}")
        existing = path.read_bytes()
        if existing != encoded:
            raise AblationError(
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


def sealed_gate_policy(
    *,
    secondary_tolerance: float = DEFAULT_SECONDARY_TOLERANCE,
    math_gain_advisory_min: float = DEFAULT_MATH_GAIN_ADVISORY_MIN,
) -> dict[str, Any]:
    """Return the sealed additive-not-subtractive gate policy.

    The reject rule is explicit: any secondary score where
    ``proto - base < -secondary_tolerance`` yields REJECT, regardless of
    math-domain gain.
    """

    if not (0.0 <= float(secondary_tolerance) <= 0.25):
        raise AblationError(
            f"secondary_tolerance out of sealed range [0, 0.25]: {secondary_tolerance}"
        )
    policy = {
        "schema": GATE_POLICY_SCHEMA,
        "name": "additive_not_subtractive_stage1",
        "optimization_target": "maximize_glm_mathematical_inheritance",
        "math_domains": list(MATH_DOMAINS),
        "secondary_capabilities": list(SECONDARY_CAPABILITIES),
        "secondary_tolerance_absolute": float(secondary_tolerance),
        "math_gain_advisory_min": float(math_gain_advisory_min),
        "reject_rule": {
            "statement": (
                "REJECT any candidate PROTO checkpoint for which any secondary "
                "capability score falls more than secondary_tolerance below the "
                "BASE_DSV4F score, regardless of math-domain gain."
            ),
            "predicate": "any(proto[s] - base[s] < -secondary_tolerance for s in SECONDARY)",
            "math_gain_cannot_override": True,
            "routing_stability_is_secondary": True,
            "context_behavior_is_secondary": True,
            "runtime_tps_is_secondary": True,
        },
        "accept_rule": {
            "statement": (
                "ACCEPT only when every secondary is ≥ base − tolerance AND the "
                "score fixture is sealed shard-scope evidence (not full-model)."
            ),
            "all_secondaries_non_regressing": True,
            "bench_scope_must_be_shard_or_fixture": True,
        },
        "training_free": True,
        "kimi_out_of_scope": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    return seal(policy)


def normalize_bench_scope(scope: str | None) -> str:
    if scope is None or not str(scope).strip():
        raise AblationError("bench_scope is required")
    token = str(scope).strip().upper().replace("-", "_").replace(" ", "_")
    # Normalize common aliases.
    aliases = {
        "SHARDS": BENCH_SCOPE_SHARD,
        "TRANSFER_SHARD": BENCH_SCOPE_SHARD,
        "TRANSFER_SHARDS": BENCH_SCOPE_SHARD,
        "BOUNDED": BENCH_SCOPE_BOUNDED,
        "BOUNDED_FIXTURES": BENCH_SCOPE_BOUNDED,
        "FIXTURES": BENCH_SCOPE_FIXTURE,
    }
    return aliases.get(token, token)


def assert_shard_bench_scope(scope: str | None) -> str:
    """Fail closed unless the request is an allowed shard/fixture scope.

    Full-model benches are refused: PROTO is not gravity-compressed / not
    fully runnable yet, and this stage must not pretend otherwise.
    """

    token = normalize_bench_scope(scope)
    if token in REFUSED_BENCH_SCOPES or token.startswith("FULL"):
        raise FullModelBenchRefused(
            f"stage-1 ablation refuses full-model / unbound bench scope {scope!r}; "
            f"only {sorted(ALLOWED_BENCH_SCOPES)} are permitted until gravity "
            f"compression and a complete DeepSeek runtime exist"
        )
    if token not in ALLOWED_BENCH_SCOPES:
        raise AblationError(
            f"unknown bench_scope {scope!r}; permitted={sorted(ALLOWED_BENCH_SCOPES)}; "
            f"refused={sorted(REFUSED_BENCH_SCOPES)}"
        )
    return token


def _require_score_map(
    scores: Mapping[str, Any],
    domains: Sequence[str],
    *,
    label: str,
) -> dict[str, float]:
    if not isinstance(scores, Mapping):
        raise AblationError(f"{label} scores must be a mapping")
    out: dict[str, float] = {}
    missing = [d for d in domains if d not in scores]
    if missing:
        raise AblationError(f"{label} missing score domains: {missing}")
    for domain in domains:
        value = scores[domain]
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise AblationError(
                f"{label}.{domain} must be a finite float, got {value!r}"
            ) from exc
        if number != number or number in (float("inf"), float("-inf")):
            raise AblationError(f"{label}.{domain} must be finite, got {number!r}")
        if number < 0.0 or number > 1.0:
            raise AblationError(
                f"{label}.{domain} must lie in [0, 1], got {number}"
            )
        out[domain] = number
    return out


def load_capability_fixture(path: Path | str) -> dict[str, Any]:
    """Load a sealed capability score fixture for A and B arms."""

    fixture_path = Path(path)
    _regular_file(fixture_path, "capability fixture")
    try:
        document = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AblationError(f"capability fixture is not valid JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise AblationError("capability fixture must be a JSON object")
    schema = document.get("schema")
    if schema != CAPABILITY_FIXTURE_SCHEMA:
        raise AblationError(
            f"capability fixture schema mismatch: got {schema!r}, "
            f"want {CAPABILITY_FIXTURE_SCHEMA!r}"
        )
    verify(document, label="capability fixture")
    scope = assert_shard_bench_scope(document.get("bench_scope"))
    base = document.get("base") or document.get(ARM_A) or document.get("A")
    proto = document.get("proto") or document.get(ARM_B) or document.get("B")
    if not isinstance(base, Mapping) or not isinstance(proto, Mapping):
        raise AblationError("fixture must include base and proto score objects")
    base_math = _require_score_map(
        base.get("math") or {}, MATH_DOMAINS, label="base.math"
    )
    base_sec = _require_score_map(
        base.get("secondary") or {}, SECONDARY_CAPABILITIES, label="base.secondary"
    )
    proto_math = _require_score_map(
        proto.get("math") or {}, MATH_DOMAINS, label="proto.math"
    )
    proto_sec = _require_score_map(
        proto.get("secondary") or {},
        SECONDARY_CAPABILITIES,
        label="proto.secondary",
    )
    return {
        "path": str(fixture_path.resolve()),
        "file_sha256": _sha256_file(fixture_path),
        "schema": schema,
        "bench_scope": scope,
        "fixture_id": str(document.get("fixture_id") or fixture_path.stem),
        "transfer_module_id": document.get("transfer_module_id"),
        "base": {"math": base_math, "secondary": base_sec},
        "proto": {"math": proto_math, "secondary": proto_sec},
        "meta": dict(document.get("meta") or {}),
        "seal_sha256": document["seal_sha256"],
    }


def evaluate_secondary_gates(
    *,
    base: Mapping[str, float],
    proto: Mapping[str, float],
    tolerance: float = DEFAULT_SECONDARY_TOLERANCE,
) -> dict[str, Any]:
    """Apply the HARD secondary non-regression gates.

    Returns a structured gate result.  Does not raise; callers decide whether
    to surface CapabilityReject from the verdict.
    """

    base_scores = _require_score_map(base, SECONDARY_CAPABILITIES, label="base.secondary")
    proto_scores = _require_score_map(
        proto, SECONDARY_CAPABILITIES, label="proto.secondary"
    )
    tol = float(tolerance)
    rows: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    for domain in SECONDARY_CAPABILITIES:
        b = base_scores[domain]
        p = proto_scores[domain]
        delta = p - b
        floor = b - tol
        ok = p >= floor
        row = {
            "domain": domain,
            "base": b,
            "proto": p,
            "delta": delta,
            "floor": floor,
            "tolerance": tol,
            "gate": "PASS" if ok else "FAIL",
            "hard": True,
        }
        rows.append(row)
        if not ok:
            regressions.append(row)
    verdict = "PASS" if not regressions else "REJECT"
    return {
        "verdict": verdict,
        "tolerance": tol,
        "domains": rows,
        "regressions": regressions,
        "reject_rule_fired": bool(regressions),
        "reject_rule": (
            "any secondary proto < base − tolerance → REJECT "
            "(math gain cannot override)"
        ),
    }


def evaluate_math_deltas(
    *,
    base: Mapping[str, float],
    proto: Mapping[str, float],
    advisory_min_gain: float = DEFAULT_MATH_GAIN_ADVISORY_MIN,
) -> dict[str, Any]:
    """Report math-domain deltas (optimization target; not hard-reject gated)."""

    base_scores = _require_score_map(base, MATH_DOMAINS, label="base.math")
    proto_scores = _require_score_map(proto, MATH_DOMAINS, label="proto.math")
    rows: list[dict[str, Any]] = []
    total_gain = 0.0
    for domain in MATH_DOMAINS:
        b = base_scores[domain]
        p = proto_scores[domain]
        delta = p - b
        total_gain += delta
        rows.append(
            {
                "domain": domain,
                "base": b,
                "proto": p,
                "delta": delta,
                "gain": delta > 0.0,
                "advisory": "OK" if delta >= advisory_min_gain else "BELOW_ADVISORY",
            }
        )
    mean_gain = total_gain / len(MATH_DOMAINS)
    return {
        "domains": rows,
        "total_gain": total_gain,
        "mean_gain": mean_gain,
        "any_gain": any(r["gain"] for r in rows),
        "advisory_min_gain": advisory_min_gain,
        "optimization_target": "maximize_glm_mathematical_inheritance",
        "hard_reject_on_math": False,
        "note": (
            "Math domains are the optimization target.  They never override a "
            "secondary REJECT."
        ),
    }


def run_avb_ablation(
    *,
    base_math: Mapping[str, float],
    base_secondary: Mapping[str, float],
    proto_math: Mapping[str, float],
    proto_secondary: Mapping[str, float],
    bench_scope: str,
    fixture_id: str = "inline",
    transfer_module_id: str | None = None,
    secondary_tolerance: float = DEFAULT_SECONDARY_TOLERANCE,
    math_gain_advisory_min: float = DEFAULT_MATH_GAIN_ADVISORY_MIN,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce the sealed A-vs-B capability delta report with reject rule applied."""

    scope = assert_shard_bench_scope(bench_scope)
    policy = sealed_gate_policy(
        secondary_tolerance=secondary_tolerance,
        math_gain_advisory_min=math_gain_advisory_min,
    )
    secondary = evaluate_secondary_gates(
        base=base_secondary,
        proto=proto_secondary,
        tolerance=secondary_tolerance,
    )
    math = evaluate_math_deltas(
        base=base_math,
        proto=proto_math,
        advisory_min_gain=math_gain_advisory_min,
    )

    if secondary["verdict"] == "REJECT":
        overall = "REJECT"
        reason = (
            "secondary capability regression beyond sealed tolerance; "
            f"failed={[r['domain'] for r in secondary['regressions']]}"
        )
    else:
        overall = "ACCEPT"
        reason = "all secondary gates held; math deltas recorded as optimization signal"

    report = {
        "schema": ABLATION_REPORT_SCHEMA,
        "recorded_at": _utc_now(),
        "stage": 1,
        "comparison": {
            "A": ARM_A,
            "B": ARM_B,
            "C_later": ARM_C_LATER,
            "D_later": ARM_D_LATER,
            "lane_owns": [ARM_A, ARM_B],
        },
        "fixture_id": fixture_id,
        "transfer_module_id": transfer_module_id,
        "bench_scope": scope,
        "gate_policy": policy,
        "math": math,
        "secondary": secondary,
        "verdict": overall,
        "reason": reason,
        "reject_rule_fired": secondary["reject_rule_fired"],
        "additive_not_subtractive": secondary["verdict"] == "PASS",
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "bodies": {
            "student": {
                "family": DEEPSEEK_V4_FLASH["family"],
                "repository": DEEPSEEK_V4_FLASH["repository"],
                "revision": DEEPSEEK_V4_FLASH["revision"],
                "hidden_size": DEEPSEEK_V4_FLASH["hidden_size"],
                "num_hidden_layers": DEEPSEEK_V4_FLASH["num_hidden_layers"],
            },
            "math_donor": {
                "family": GLM_5_2["family"],
                "repository": GLM_5_2["repository"],
                "revision": GLM_5_2["revision"],
                "hidden_size": GLM_5_2["hidden_size"],
                "num_hidden_layers": GLM_5_2["num_hidden_layers"],
            },
            "strategic_donor_stage2": "kimi_k3 (out of scope for stage 1)",
        },
        "bridges_in_play": {
            "active_stage1": "GLM_MATH_BRIDGE",
            "preserved_untouched_for_stage2": "KIMI_STRATEGIC_BRIDGE",
            "all_declared": list(BRIDGES),
        },
        "extra": dict(extra or {}),
    }
    return seal(report)


def run_avb_from_fixture(
    fixture_path: Path | str,
    *,
    secondary_tolerance: float = DEFAULT_SECONDARY_TOLERANCE,
    math_gain_advisory_min: float = DEFAULT_MATH_GAIN_ADVISORY_MIN,
) -> dict[str, Any]:
    """Load a sealed fixture and run the A-vs-B ablation."""

    fixture = load_capability_fixture(fixture_path)
    return run_avb_ablation(
        base_math=fixture["base"]["math"],
        base_secondary=fixture["base"]["secondary"],
        proto_math=fixture["proto"]["math"],
        proto_secondary=fixture["proto"]["secondary"],
        bench_scope=fixture["bench_scope"],
        fixture_id=fixture["fixture_id"],
        transfer_module_id=fixture.get("transfer_module_id"),
        secondary_tolerance=secondary_tolerance,
        math_gain_advisory_min=math_gain_advisory_min,
        extra={
            "fixture_path": fixture["path"],
            "fixture_sha256": fixture["file_sha256"],
            "fixture_seal_sha256": fixture["seal_sha256"],
            "fixture_meta": fixture["meta"],
        },
    )


@dataclass(frozen=True)
class ShardBenchRequest:
    """Explicit shard-level bench request (never full model)."""

    scope: str
    shard_ids: tuple[str, ...]
    transfer_module_path: Path | None = None
    base_fixture_path: Path | None = None
    proto_fixture_path: Path | None = None
    scores_fixture_path: Path | None = None


def run_shard_bench(request: ShardBenchRequest) -> dict[str, Any]:
    """Interim shard-level bench harness.

    Benchmarks transfer SHARDS / bounded fixtures only.  Full-model requests
    raise FullModelBenchRefused (fail closed).
    """

    scope = assert_shard_bench_scope(request.scope)
    if not request.shard_ids and request.scores_fixture_path is None:
        raise AblationError(
            "shard bench requires at least one shard_id or a scores_fixture_path"
        )

    ablation: dict[str, Any] | None = None
    if request.scores_fixture_path is not None:
        ablation = run_avb_from_fixture(request.scores_fixture_path)

    document = {
        "schema": SHARD_BENCH_SCHEMA,
        "recorded_at": _utc_now(),
        "bench_scope": scope,
        "shard_ids": list(request.shard_ids),
        "transfer_module_path": (
            str(request.transfer_module_path)
            if request.transfer_module_path is not None
            else None
        ),
        "base_fixture_path": (
            str(request.base_fixture_path) if request.base_fixture_path else None
        ),
        "proto_fixture_path": (
            str(request.proto_fixture_path) if request.proto_fixture_path else None
        ),
        "scores_fixture_path": (
            str(request.scores_fixture_path) if request.scores_fixture_path else None
        ),
        "full_model_refused": True,
        "measurement_mode": "shard_or_bounded_fixture_scores",
        "ablation": ablation,
        "verdict": (
            ablation["verdict"]
            if ablation is not None
            else "SCOPED_ONLY_NO_SCORES"
        ),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "limits": (
            "This harness does not load or execute a full DeepSeek-V4-Flash, "
            "GLM, or Kimi model.  It evaluates sealed shard/fixture capability "
            "scores only.  No TPS or capability promotion follows without a "
            "later full-runtime gate."
        ),
    }
    return seal(document)


def make_capability_fixture(
    *,
    fixture_id: str,
    bench_scope: str,
    base_math: Mapping[str, float],
    base_secondary: Mapping[str, float],
    proto_math: Mapping[str, float],
    proto_secondary: Mapping[str, float],
    transfer_module_id: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a sealed capability score fixture (for tests and offline benches)."""

    scope = assert_shard_bench_scope(bench_scope)
    document = {
        "schema": CAPABILITY_FIXTURE_SCHEMA,
        "fixture_id": fixture_id,
        "bench_scope": scope,
        "transfer_module_id": transfer_module_id,
        "base": {
            "math": _require_score_map(base_math, MATH_DOMAINS, label="base.math"),
            "secondary": _require_score_map(
                base_secondary, SECONDARY_CAPABILITIES, label="base.secondary"
            ),
        },
        "proto": {
            "math": _require_score_map(proto_math, MATH_DOMAINS, label="proto.math"),
            "secondary": _require_score_map(
                proto_secondary, SECONDARY_CAPABILITIES, label="proto.secondary"
            ),
        },
        "meta": dict(meta or {}),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    return seal(document)


def default_score_template(fill: float = 0.70) -> dict[str, dict[str, float]]:
    """Uniform score map for fixtures (tests / scaffolding)."""

    if not (0.0 <= fill <= 1.0):
        raise AblationError(f"fill must be in [0,1], got {fill}")
    return {
        "math": {d: float(fill) for d in MATH_DOMAINS},
        "secondary": {d: float(fill) for d in SECONDARY_CAPABILITIES},
    }


def write_report(path: Path | str, report: Mapping[str, Any]) -> str:
    """Atomically write a sealed ablation or bench report."""

    out = Path(path)
    if "seal_sha256" not in report:
        report = seal(dict(report))
    else:
        verify(report, label="ablation report")
    return _atomic_write_json(out, report)


# ---------------------------------------------------------------------------
# Functional-transfer A–G ablation harness (owner steer §9)
# ---------------------------------------------------------------------------

# Extended arms for the trained functional-transfer program.
# Stage-1 still owns only A vs B on shard fixtures; C–G need training/evidence.
ARM_FT_A = "A_BASE_DSV4F"
ARM_FT_B = "B_LINEAR_SUBSPACE_INIT"
ARM_FT_C = "C_BEHAVIOR_DISTILL"
ARM_FT_D = "D_NONLINEAR_BRIDGES"
ARM_FT_E = "E_ROUTER_METHOD_ADAPTERS"
ARM_FT_F = "F_EXPERT_ITERATION"
ARM_FT_G = "G_COMPLETE_PROTO_FRANKENSTEIN"

FUNCTIONAL_TRANSFER_ARMS: tuple[tuple[str, str], ...] = (
    (ARM_FT_A, "Base DSV4F only; no donor inheritance"),
    (ARM_FT_B, "Base + LINEAR_SUBSPACE_INITIALIZATION (not inheritance)"),
    (ARM_FT_C, "Behavior distillation objectives (gated: training loop)"),
    (ARM_FT_D, "Nonlinear reversible bridges (gated: training loop)"),
    (ARM_FT_E, "Router/method adapters (gated: training loop; no GLM router copy)"),
    (ARM_FT_F, "Verified expert iteration (gated: verifier + training)"),
    (ARM_FT_G, "Complete Proto-Frankenstein = full functional transfer stack"),
)

AG_ABLATION_SCHEMA = "hawking.frankenstein.functional_transfer_ag_ablation.v1"


def functional_transfer_arm_catalog() -> dict[str, Any]:
    return {
        "schema": "hawking.frankenstein.functional_transfer_arm_catalog.v1",
        "arms": [
            {"id": aid, "description": desc, "index": i}
            for i, (aid, desc) in enumerate(FUNCTIONAL_TRANSFER_ARMS)
        ],
        "reject_rule": (
            "REJECT any arm that regresses secondary capabilities beyond tolerance, "
            "or that improves imitation but fails proof/computation/repair/transfer/"
            "hidden eval. Math gain cannot override secondary REJECT."
        ),
        "linear_init_is_not_proto_complete": True,
        "g_requires_all_prior_plus_gates": True,
    }


def run_ag_ablation(
    *,
    arm_scores: Mapping[str, Mapping[str, Any]] | None = None,
    secondary_tolerance: float = DEFAULT_SECONDARY_TOLERANCE,
    base_arm: str = ARM_FT_A,
) -> dict[str, Any]:
    """A–G ablation harness.

    Without sealed arm_scores, returns a FRAMEWORK report with PENDING arms
    (never fabricates scores).  With scores, applies secondary reject rule
    pairwise against BASE for each arm.
    """

    catalog = functional_transfer_arm_catalog()
    policy = sealed_gate_policy(secondary_tolerance=secondary_tolerance)
    arm_rows: list[dict[str, Any]] = []

    if arm_scores is None:
        for aid, desc in FUNCTIONAL_TRANSFER_ARMS:
            arm_rows.append(
                {
                    "arm": aid,
                    "description": desc,
                    "status": "PENDING",
                    "verdict": None,
                    "reason": (
                        "no sealed scores; arm not evaluated "
                        "(REQUIRES_BENCHMARK_CORPUS / training / forward as applicable)"
                    ),
                    "fabricated": False,
                }
            )
        document = {
            "schema": AG_ABLATION_SCHEMA,
            "recorded_at": _utc_now(),
            "status": "FRAMEWORK_PENDING",
            "verdict": "PENDING",
            "arms": arm_rows,
            "catalog": catalog,
            "gate_policy": policy,
            "reject_rule_fired": False,
            "fabricated_scores": False,
            "note": (
                "A–G harness is real code; live evaluation is pending corpus + "
                "trained modules.  B is LINEAR_SUBSPACE_INITIALIZATION, not PROTO complete."
            ),
            "claim_boundary": {
                **dict(CLAIM_BOUNDARY),
                "functional_transfer_ag": True,
                "proto_complete_from_linear": False,
            },
        }
        return seal(document)

    if base_arm not in arm_scores:
        raise AblationError(f"arm_scores must include base arm {base_arm!r}")
    base = arm_scores[base_arm]
    base_math = _require_score_map(
        base.get("math") or {}, MATH_DOMAINS, label=f"{base_arm}.math"
    )
    base_sec = _require_score_map(
        base.get("secondary") or {},
        SECONDARY_CAPABILITIES,
        label=f"{base_arm}.secondary",
    )

    any_reject = False
    for aid, desc in FUNCTIONAL_TRANSFER_ARMS:
        if aid not in arm_scores:
            arm_rows.append(
                {
                    "arm": aid,
                    "description": desc,
                    "status": "PENDING",
                    "verdict": None,
                    "reason": "scores not provided for this arm",
                    "fabricated": False,
                }
            )
            continue
        if aid == base_arm:
            arm_rows.append(
                {
                    "arm": aid,
                    "description": desc,
                    "status": "BASE",
                    "verdict": "BASE",
                    "math": base_math,
                    "secondary": base_sec,
                }
            )
            continue
        scores = arm_scores[aid]
        # Optional reject: imitation without proof
        if scores.get("imitation_only_without_proof") is True:
            any_reject = True
            arm_rows.append(
                {
                    "arm": aid,
                    "description": desc,
                    "status": "EVALUATED",
                    "verdict": "REJECT",
                    "reason": (
                        "improves imitation but fails proof/computation/repair/"
                        "transfer/hidden eval"
                    ),
                    "reject_rule_fired": True,
                }
            )
            continue
        report = run_avb_ablation(
            base_math=base_math,
            base_secondary=base_sec,
            proto_math=scores.get("math") or {},
            proto_secondary=scores.get("secondary") or {},
            bench_scope=str(scores.get("bench_scope") or "FIXTURE"),
            fixture_id=f"ag-{aid}",
            transfer_module_id=aid,
            secondary_tolerance=secondary_tolerance,
        )
        if report["verdict"] == "REJECT":
            any_reject = True
        arm_rows.append(
            {
                "arm": aid,
                "description": desc,
                "status": "EVALUATED",
                "verdict": report["verdict"],
                "reject_rule_fired": report["reject_rule_fired"],
                "math_mean_gain": (report.get("math") or {}).get("mean_gain"),
                "ablation_seal_sha256": report.get("seal_sha256"),
            }
        )

    evaluated = [r for r in arm_rows if r.get("status") == "EVALUATED"]
    if not evaluated:
        overall = "PENDING"
    elif any_reject:
        overall = "REJECT"
    elif any(r.get("status") == "PENDING" for r in arm_rows):
        overall = "PARTIAL"
    else:
        overall = "ACCEPT"

    document = {
        "schema": AG_ABLATION_SCHEMA,
        "recorded_at": _utc_now(),
        "status": "EVALUATED" if evaluated else "FRAMEWORK_PENDING",
        "verdict": overall,
        "arms": arm_rows,
        "catalog": catalog,
        "gate_policy": policy,
        "reject_rule_fired": any_reject,
        "fabricated_scores": False,
        "base_arm": base_arm,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "functional_transfer_ag": True,
            "proto_complete_from_linear": False,
        },
    }
    return seal(document)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-1 PROTO_FRANKENSTEIN A-vs-B ablation (additive-not-subtractive). "
            "Shard/fixture scores only; full-model benches are refused."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Evaluate a sealed capability fixture")
    p_eval.add_argument("--fixture", type=Path, required=True)
    p_eval.add_argument("--out", type=Path, required=True)
    p_eval.add_argument(
        "--secondary-tolerance",
        type=float,
        default=DEFAULT_SECONDARY_TOLERANCE,
    )
    p_eval.add_argument(
        "--math-gain-advisory-min",
        type=float,
        default=DEFAULT_MATH_GAIN_ADVISORY_MIN,
    )

    p_bench = sub.add_parser(
        "shard-bench",
        help="Run the interim shard-level bench harness (refuses full model)",
    )
    p_bench.add_argument(
        "--scope",
        required=True,
        help="SHARD | FIXTURE | BOUNDED_FIXTURE (FULL_MODEL refused)",
    )
    p_bench.add_argument(
        "--shard-id",
        action="append",
        default=[],
        dest="shard_ids",
        help="Transfer shard id (repeatable)",
    )
    p_bench.add_argument("--scores-fixture", type=Path, default=None)
    p_bench.add_argument("--transfer-module", type=Path, default=None)
    p_bench.add_argument("--out", type=Path, required=True)

    p_policy = sub.add_parser("policy", help="Emit the sealed gate policy JSON")
    p_policy.add_argument("--out", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "evaluate":
            report = run_avb_from_fixture(
                args.fixture,
                secondary_tolerance=args.secondary_tolerance,
                math_gain_advisory_min=args.math_gain_advisory_min,
            )
            write_report(args.out, report)
            print(json.dumps({"verdict": report["verdict"], "out": str(args.out)}))
            return 0 if report["verdict"] == "ACCEPT" else 2

        if args.command == "shard-bench":
            request = ShardBenchRequest(
                scope=args.scope,
                shard_ids=tuple(args.shard_ids),
                transfer_module_path=args.transfer_module,
                scores_fixture_path=args.scores_fixture,
            )
            document = run_shard_bench(request)
            write_report(args.out, document)
            print(
                json.dumps(
                    {
                        "verdict": document["verdict"],
                        "bench_scope": document["bench_scope"],
                        "out": str(args.out),
                    }
                )
            )
            if document.get("ablation") and document["ablation"]["verdict"] == "REJECT":
                return 2
            return 0

        if args.command == "policy":
            policy = sealed_gate_policy()
            write_report(args.out, policy)
            print(json.dumps({"out": str(args.out), "schema": policy["schema"]}))
            return 0

        raise AblationError(f"unknown command: {args.command}")
    except FullModelBenchRefused as exc:
        print(json.dumps({"error": "FULL_MODEL_BENCH_REFUSED", "detail": str(exc)}))
        return 3
    except AblationError as exc:
        print(json.dumps({"error": "ABLATION_ERROR", "detail": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
