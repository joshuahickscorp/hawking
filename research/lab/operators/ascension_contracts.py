"""Canonical Ascension planning contracts + blocked-state registry validation.

Bible:
  §1  platform decision (Apple-first / Metal / portable / CUDA-deferred)
  §11 complete-token profiler + FLOPS ledger
  §29–§31 family kernel / model ladder / rotation blocked-state registry

This module loads the durable JSON under ``workspace/docs/plans/ascension/``
and asserts structural invariants. It never downloads weights, never claims
Qwen performance, and never flips schedule or completion-state status.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
ASCENSION_DIR = REPO_ROOT / "workspace" / "docs" / "plans" / "ascension"

PLATFORM_DECISION_PATH = ASCENSION_DIR / "ASCENSION_PLATFORM_DECISION.json"
PROFILER_CONTRACT_PATH = ASCENSION_DIR / "ASCENSION_COMPLETE_TOKEN_PROFILER_CONTRACT.json"
BLOCKED_REGISTRY_PATH = ASCENSION_DIR / "ASCENSION_BLOCKED_STATE_REGISTRY.json"
MASTER_SCHEDULE_PATH = ASCENSION_DIR / "ASCENSION_MASTER_SCHEDULE.json"
OVERVIEW_PATH = ASCENSION_DIR / "ASCENSION_PROGRAM_OVERVIEW.md"

PLATFORM_SCHEMA = "hawking.ascension.platform_decision.v1"
PROFILER_SCHEMA = "hawking.ascension.complete_token_profiler_contract.v1"
BLOCKED_REGISTRY_SCHEMA = "hawking.ascension.blocked_state_registry.v1"

REQUIRED_BLOCKER_FIELDS: tuple[str, ...] = (
    "architecture_config_source_admission",
    "gravity_family_support",
    "exact_runtime",
    "profiler_evidence",
    "tg_evidence",
    "tg3_approval",
)

REQUIRED_BOOTSTRAP_ENTRY_IDS: tuple[str, ...] = (
    "qwen3_coder_30b",
    "qwen3_coder_next_80b",
)

REQUIRED_BOOTSTRAP_DISPLAY_NAMES: tuple[str, ...] = (
    "Qwen3-Coder-30B",
    "Qwen3-Coder-Next-80B",
)

MODEL_LADDER_PIPELINE: tuple[str, ...] = (
    "DISCOVER",
    "PREFLIGHT",
    "RESEARCH_DISTINCTION",
    "DOWNLOAD_STREAM",
    "GRAVITY",
    "LOAD",
    "PARITY",
    "CAPABILITY",
    "PROFILE",
    "OPTIMIZE",
    "REVIEW",
    "REPORT",
    "SEAL",
    "EVICT",
    "ROTATE",
)

INITIAL_FAMILIES: tuple[str, ...] = (
    "QWEN3_MOE",
    "QWEN3_NEXT",
    "DEEPSEEK_V4",
    "LLAMA",
    "MISTRAL_MIXTRAL",
    "STATE_SPACE_HYBRID",
)

GLOBAL_LEDGERS: tuple[str, ...] = (
    "PEAK_UTILIZATION",
    "FLOPS_PER_TOKEN",
    "BYTES_PER_FLOP",
    "REUSE_FACTOR",
    "CRITICAL_DEPTH",
    "STATE_TRAFFIC",
)

PORTABLE_INTERFACES: tuple[str, ...] = (
    "model_semantics",
    "gravity_representation_contract",
    "kernel_grammar_ir",
    "scheduler_contract",
    "parity_capability_harness",
    "receipt_schema",
    "backend_neutral_runtime_interfaces",
)

MUST_NOT_REQUIRE_SHARING: tuple[str, ...] = (
    "kernel_source",
    "tile_geometry",
    "memory_layout",
    "graph_implementation",
    "cache_policy",
    "command_topology",
    "autotuning_rules",
)


class AscensionContractError(RuntimeError):
    """Planning-contract or blocked-state registry invariant failed."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AscensionContractError(f"missing contract file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AscensionContractError(f"contract root must be object: {path}")
    return data


def load_platform_decision() -> dict[str, Any]:
    return _load_json(PLATFORM_DECISION_PATH)


def load_profiler_contract() -> dict[str, Any]:
    return _load_json(PROFILER_CONTRACT_PATH)


def load_blocked_state_registry() -> dict[str, Any]:
    return _load_json(BLOCKED_REGISTRY_PATH)


def load_master_schedule() -> dict[str, Any]:
    return _load_json(MASTER_SCHEDULE_PATH)


def _require_keys(obj: Mapping[str, Any], keys: Sequence[str], *, label: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise AscensionContractError(f"{label}: missing keys {missing}")


def _require_subset(
    actual: Sequence[str] | set[str],
    required: Sequence[str],
    *,
    label: str,
) -> None:
    missing = [item for item in required if item not in set(actual)]
    if missing:
        raise AscensionContractError(f"{label}: missing required items {missing}")


def validate_platform_decision(doc: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate bible §1 platform-decision contract structure."""
    data = dict(doc) if doc is not None else load_platform_decision()
    _require_keys(
        data,
        (
            "schema",
            "bible_section",
            "stance",
            "tier_1_production",
            "tier_2_portable_contracts",
            "future_tier_1b_cuda",
            "metal_cuda_may_share",
            "metal_cuda_must_not_require_sharing",
            "forbidden",
            "honesty",
        ),
        label="platform_decision",
    )
    if data["schema"] != PLATFORM_SCHEMA:
        raise AscensionContractError(f"platform_decision: bad schema {data['schema']!r}")
    if data["bible_section"] != 1:
        raise AscensionContractError("platform_decision: bible_section must be 1")

    stance = data["stance"]
    if not isinstance(stance, Mapping):
        raise AscensionContractError("platform_decision: stance must be object")
    for key in (
        "apple_first",
        "metal_dominant",
        "architecture_portable",
        "cuda_deferred",
    ):
        if stance.get(key) is not True:
            raise AscensionContractError(f"platform_decision: stance.{key} must be true")
    if stance.get("cuda_rejected") is not False:
        raise AscensionContractError("platform_decision: cuda_rejected must be false")

    tier1 = data["tier_1_production"]
    if not isinstance(tier1, Mapping) or "Metal" not in str(tier1.get("backend", "")):
        raise AscensionContractError("platform_decision: Tier 1 must be Apple/Metal")

    tier2 = data["tier_2_portable_contracts"]
    if not isinstance(tier2, Mapping):
        raise AscensionContractError("platform_decision: tier_2 must be object")
    interfaces = tier2.get("interfaces")
    if not isinstance(interfaces, list):
        raise AscensionContractError("platform_decision: tier_2.interfaces must be list")
    _require_subset(interfaces, PORTABLE_INTERFACES, label="platform_decision.tier_2")

    must_not = data["metal_cuda_must_not_require_sharing"]
    if not isinstance(must_not, list):
        raise AscensionContractError("platform_decision: must_not_require_sharing must be list")
    _require_subset(
        must_not,
        MUST_NOT_REQUIRE_SHARING,
        label="platform_decision.must_not_share",
    )

    forbidden = data["forbidden"]
    if not isinstance(forbidden, list):
        raise AscensionContractError("platform_decision: forbidden must be list")
    if "lowest_common_denominator_kernel_mandate" not in forbidden:
        raise AscensionContractError(
            "platform_decision: must forbid lowest_common_denominator_kernel_mandate"
        )

    cuda = data["future_tier_1b_cuda"]
    if not isinstance(cuda, Mapping) or cuda.get("stance") != "deferred_not_rejected":
        raise AscensionContractError("platform_decision: CUDA must be deferred_not_rejected")

    honesty = data["honesty"]
    if not isinstance(honesty, Mapping):
        raise AscensionContractError("platform_decision: honesty must be object")
    if honesty.get("stage_ready") is not False:
        raise AscensionContractError("platform_decision: must not mark stage_ready")
    if honesty.get("apple_production_certified") is not False:
        raise AscensionContractError(
            "platform_decision: must not claim apple_production_certified"
        )
    if data.get("implementation_status") not in (None, "NOT_STARTED"):
        # Allow explicit NOT_STARTED only; refuse silent READY claims.
        if data.get("implementation_status") != "NOT_STARTED":
            raise AscensionContractError(
                "platform_decision: implementation_status must be NOT_STARTED"
            )

    return data


def validate_profiler_contract(doc: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate bible §11 complete-token profiler / FLOPS ledger contract."""
    data = dict(doc) if doc is not None else load_profiler_contract()
    _require_keys(
        data,
        (
            "schema",
            "bible_section",
            "target_explained_percent",
            "live_qwen_values",
            "required_stage_inventory_bible",
            "required_per_stage_observations",
            "required_global_ledgers",
            "refusal_conditions",
            "ranking_rule",
            "honesty",
        ),
        label="profiler_contract",
    )
    if data["schema"] != PROFILER_SCHEMA:
        raise AscensionContractError(f"profiler_contract: bad schema {data['schema']!r}")
    if data["bible_section"] != 11:
        raise AscensionContractError("profiler_contract: bible_section must be 11")
    if float(data["target_explained_percent"]) != 98.0:
        raise AscensionContractError("profiler_contract: target_explained_percent must be 98.0")

    live = data["live_qwen_values"]
    if not isinstance(live, Mapping):
        raise AscensionContractError("profiler_contract: live_qwen_values must be object")
    if live.get("exist") is not False:
        raise AscensionContractError("profiler_contract: live_qwen_values.exist must be false")
    if live.get("qwen3_coder_30b") is not None:
        raise AscensionContractError("profiler_contract: no live 30B values permitted")
    if live.get("qwen3_coder_next_80b") is not None:
        raise AscensionContractError("profiler_contract: no live 80B values permitted")

    stages = data["required_stage_inventory_bible"]
    if not isinstance(stages, list) or len(stages) < 10:
        raise AscensionContractError("profiler_contract: stage inventory incomplete")
    for required in ("embedding", "residual", "HCLI stream"):
        if required not in stages:
            raise AscensionContractError(
                f"profiler_contract: missing stage inventory item {required!r}"
            )

    per_stage = data["required_per_stage_observations"]
    if not isinstance(per_stage, list):
        raise AscensionContractError("profiler_contract: per-stage observations must be list")
    for required in (
        "GPU duration",
        "CPU duration",
        "achieved FLOPS",
        "dispatches",
        "p99",
        "fallback",
    ):
        if required not in per_stage:
            raise AscensionContractError(
                f"profiler_contract: missing per-stage observation {required!r}"
            )

    ledgers = data["required_global_ledgers"]
    if not isinstance(ledgers, list):
        raise AscensionContractError("profiler_contract: global ledgers must be list")
    _require_subset(ledgers, GLOBAL_LEDGERS, label="profiler_contract.global_ledgers")

    refusals = data["refusal_conditions"]
    if not isinstance(refusals, list) or not refusals:
        raise AscensionContractError("profiler_contract: refusal_conditions required")
    refusal_ids = {
        r.get("id") for r in refusals if isinstance(r, Mapping)
    }
    for required_id in (
        "REFUSE_UNEXPLAINED_OTHER_ABOVE_TARGET",
        "REFUSE_LIVE_QWEN_CLAIM_WITHOUT_RUNTIME",
        "REFUSE_HIGHER_FLOPS_AS_AUTOMATIC_WIN",
        "REFUSE_BLENDED_TPS",
        "REFUSE_MISSING_GLOBAL_LEDGER",
    ):
        if required_id not in refusal_ids:
            raise AscensionContractError(
                f"profiler_contract: missing refusal condition {required_id}"
            )

    ranking = data["ranking_rule"]
    if not isinstance(ranking, Mapping):
        raise AscensionContractError("profiler_contract: ranking_rule must be object")
    if ranking.get("primary") != "complete_wall_time_and_p99":
        raise AscensionContractError(
            "profiler_contract: primary ranking must be complete_wall_time_and_p99"
        )

    honesty = data["honesty"]
    if not isinstance(honesty, Mapping):
        raise AscensionContractError("profiler_contract: honesty must be object")
    for key in (
        "live_qwen_profile",
        "live_qwen_flops_ledger",
        "base_true_tps_claimed",
        "performance_claims",
        "stage_ready",
    ):
        if honesty.get(key) is not False:
            raise AscensionContractError(f"profiler_contract: honesty.{key} must be false")

    if data.get("implementation_status") != "NOT_STARTED":
        raise AscensionContractError(
            "profiler_contract: implementation_status must be NOT_STARTED"
        )

    return data


def validate_blocked_state_registry(
    doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate bible §§29–31 blocked-state registry structure."""
    data = dict(doc) if doc is not None else load_blocked_state_registry()
    _require_keys(
        data,
        (
            "schema",
            "bible_sections",
            "required_blocker_fields",
            "blocker_field_definitions",
            "family_kernel_architecture",
            "model_ladder_pipeline",
            "rotation_rule",
            "entries",
            "honesty",
        ),
        label="blocked_registry",
    )
    if data["schema"] != BLOCKED_REGISTRY_SCHEMA:
        raise AscensionContractError(f"blocked_registry: bad schema {data['schema']!r}")
    sections = data["bible_sections"]
    if not isinstance(sections, list) or set(sections) != {29, 30, 31}:
        raise AscensionContractError("blocked_registry: bible_sections must be [29,30,31]")

    required = data["required_blocker_fields"]
    if not isinstance(required, list):
        raise AscensionContractError("blocked_registry: required_blocker_fields must be list")
    if tuple(required) != REQUIRED_BLOCKER_FIELDS:
        raise AscensionContractError(
            "blocked_registry: required_blocker_fields must match canonical order/set"
        )

    definitions = data["blocker_field_definitions"]
    if not isinstance(definitions, Mapping):
        raise AscensionContractError("blocked_registry: blocker_field_definitions must be object")
    for field in REQUIRED_BLOCKER_FIELDS:
        if field not in definitions:
            raise AscensionContractError(
                f"blocked_registry: missing blocker definition {field}"
            )

    family = data["family_kernel_architecture"]
    if not isinstance(family, Mapping):
        raise AscensionContractError("blocked_registry: family_kernel_architecture must be object")
    initial = family.get("initial_families")
    if not isinstance(initial, list):
        raise AscensionContractError("blocked_registry: initial_families must be list")
    _require_subset(initial, INITIAL_FAMILIES, label="blocked_registry.families")
    promote_only = family.get("promote_only_by")
    if not isinstance(promote_only, list) or "complete_wall_time" not in promote_only:
        raise AscensionContractError(
            "blocked_registry: promote_only_by must include complete_wall_time"
        )

    ladder = data["model_ladder_pipeline"]
    if not isinstance(ladder, Mapping):
        raise AscensionContractError("blocked_registry: model_ladder_pipeline must be object")
    phases = ladder.get("phases")
    if not isinstance(phases, list) or tuple(phases) != MODEL_LADDER_PIPELINE:
        raise AscensionContractError(
            "blocked_registry: model ladder phases must match bible §30 pipeline"
        )

    rotation = data["rotation_rule"]
    if not isinstance(rotation, Mapping):
        raise AscensionContractError("blocked_registry: rotation_rule must be object")
    for key in ("A", "B", "TG3"):
        if not rotation.get(key):
            raise AscensionContractError(f"blocked_registry: rotation_rule.{key} required")

    entries = data["entries"]
    if not isinstance(entries, list):
        raise AscensionContractError("blocked_registry: entries must be list")
    by_id = {e.get("id"): e for e in entries if isinstance(e, Mapping)}
    for entry_id in REQUIRED_BOOTSTRAP_ENTRY_IDS:
        if entry_id not in by_id:
            raise AscensionContractError(
                f"blocked_registry: missing required entry {entry_id}"
            )

    display_names = {
        e.get("display_name") for e in entries if isinstance(e, Mapping)
    }
    for name in REQUIRED_BOOTSTRAP_DISPLAY_NAMES:
        if name not in display_names:
            raise AscensionContractError(
                f"blocked_registry: missing display_name {name}"
            )

    for entry_id in REQUIRED_BOOTSTRAP_ENTRY_IDS:
        entry = by_id[entry_id]
        if entry.get("status") != "BLOCKED":
            raise AscensionContractError(
                f"blocked_registry: {entry_id} must be BLOCKED (got {entry.get('status')!r})"
            )
        blockers = entry.get("blockers")
        if not isinstance(blockers, Mapping):
            raise AscensionContractError(
                f"blocked_registry: {entry_id}.blockers must be object"
            )
        for field in REQUIRED_BLOCKER_FIELDS:
            if field not in blockers:
                raise AscensionContractError(
                    f"blocked_registry: {entry_id} missing blocker {field}"
                )
            cell = blockers[field]
            if not isinstance(cell, Mapping):
                raise AscensionContractError(
                    f"blocked_registry: {entry_id}.blockers.{field} must be object"
                )
            if cell.get("status") != "BLOCKED":
                raise AscensionContractError(
                    f"blocked_registry: {entry_id}.{field} must be BLOCKED"
                )
            if cell.get("cleared_at") is not None or cell.get("cleared_by") is not None:
                raise AscensionContractError(
                    f"blocked_registry: {entry_id}.{field} must not be cleared yet"
                )
            refs = cell.get("evidence_refs")
            if not isinstance(refs, list) or len(refs) != 0:
                raise AscensionContractError(
                    f"blocked_registry: {entry_id}.{field}.evidence_refs must be empty"
                )
        honesty = entry.get("honesty")
        if not isinstance(honesty, Mapping):
            raise AscensionContractError(
                f"blocked_registry: {entry_id}.honesty must be object"
            )
        for key in (
            "weights_present",
            "live_runtime",
            "performance_claims",
            "sandbox_workforce_admitted",
        ):
            if honesty.get(key) is not False:
                raise AscensionContractError(
                    f"blocked_registry: {entry_id}.honesty.{key} must be false"
                )

    # Family mapping for bootstrap identities
    if by_id["qwen3_coder_30b"].get("family") != "QWEN3_MOE":
        raise AscensionContractError("blocked_registry: 30B family must be QWEN3_MOE")
    if by_id["qwen3_coder_next_80b"].get("family") != "QWEN3_NEXT":
        raise AscensionContractError("blocked_registry: 80B family must be QWEN3_NEXT")

    honesty = data["honesty"]
    if not isinstance(honesty, Mapping):
        raise AscensionContractError("blocked_registry: honesty must be object")
    for key in (
        "any_bootstrap_model_unblocked",
        "family_megakernels_promoted",
        "wider_ladder_active",
        "stage_ready",
    ):
        if honesty.get(key) is not False:
            raise AscensionContractError(f"blocked_registry: honesty.{key} must be false")

    if data.get("implementation_status") != "NOT_STARTED":
        raise AscensionContractError(
            "blocked_registry: implementation_status must be NOT_STARTED"
        )

    return data


def entry_is_fully_blocked(entry: Mapping[str, Any]) -> bool:
    """True when entry status and every required blocker cell is BLOCKED."""
    if entry.get("status") != "BLOCKED":
        return False
    blockers = entry.get("blockers")
    if not isinstance(blockers, Mapping):
        return False
    for field in REQUIRED_BLOCKER_FIELDS:
        cell = blockers.get(field)
        if not isinstance(cell, Mapping) or cell.get("status") != "BLOCKED":
            return False
    return True


def validate_schedule_references_contracts(
    schedule: Mapping[str, Any] | None = None,
) -> None:
    """Ensure master schedule companion_docs cite the new contracts where expected."""
    data = dict(schedule) if schedule is not None else load_master_schedule()
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise AscensionContractError("master_schedule: steps must be list")
    by_id = {s.get("id"): s for s in steps if isinstance(s, Mapping)}

    def _docs(step_id: int) -> list[str]:
        step = by_id.get(step_id)
        if not isinstance(step, Mapping):
            raise AscensionContractError(f"master_schedule: missing step {step_id}")
        docs = step.get("companion_docs") or []
        if not isinstance(docs, list):
            raise AscensionContractError(
                f"master_schedule: step {step_id} companion_docs must be list"
            )
        return [str(d) for d in docs]

    def _assert_not_ready(step_id: int) -> None:
        status = by_id[step_id].get("status")
        if status not in ("NOT_STARTED", "BLOCKED"):
            raise AscensionContractError(
                f"master_schedule: step {step_id} must not be marked ready ({status!r})"
            )

    # Platform contract on Apple production + CUDA steps.
    platform_rel = "workspace/docs/plans/ascension/ASCENSION_PLATFORM_DECISION_CONTRACT.md"
    platform_json = "workspace/docs/plans/ascension/ASCENSION_PLATFORM_DECISION.json"
    for step_id in (12, 32, 33):
        docs = _docs(step_id)
        if platform_rel not in docs and platform_json not in docs:
            raise AscensionContractError(
                f"master_schedule: step {step_id} must reference platform decision contract"
            )
        _assert_not_ready(step_id)

    profiler_rel = (
        "workspace/docs/plans/ascension/ASCENSION_COMPLETE_TOKEN_PROFILER_CONTRACT.md"
    )
    profiler_json = (
        "workspace/docs/plans/ascension/ASCENSION_COMPLETE_TOKEN_PROFILER_CONTRACT.json"
    )
    for step_id in (13, 20):
        docs = _docs(step_id)
        if profiler_rel not in docs and profiler_json not in docs:
            raise AscensionContractError(
                f"master_schedule: step {step_id} must reference profiler contract"
            )
        _assert_not_ready(step_id)

    family_rel = (
        "workspace/docs/plans/ascension/ASCENSION_FAMILY_KERNEL_LADDER_ROTATION_CONTRACT.md"
    )
    blocked_rel = "workspace/docs/plans/ascension/ASCENSION_BLOCKED_STATE_REGISTRY.json"
    for step_id in (28, 29, 30):
        docs = _docs(step_id)
        if family_rel not in docs and blocked_rel not in docs:
            raise AscensionContractError(
                f"master_schedule: step {step_id} must reference family/blocked registry contract"
            )
        _assert_not_ready(step_id)


def validate_overview_references_contracts(overview_text: str | None = None) -> None:
    """Ensure programme overview indexes the three dedicated contracts."""
    text = overview_text if overview_text is not None else OVERVIEW_PATH.read_text(
        encoding="utf-8"
    )
    required_snippets = (
        "ASCENSION_PLATFORM_DECISION",
        "ASCENSION_COMPLETE_TOKEN_PROFILER_CONTRACT",
        "ASCENSION_FAMILY_KERNEL_LADDER_ROTATION_CONTRACT",
        "ASCENSION_BLOCKED_STATE_REGISTRY",
    )
    for snippet in required_snippets:
        if snippet not in text:
            raise AscensionContractError(
                f"overview: missing reference to {snippet}"
            )


def validate_all_ascension_contracts() -> dict[str, Any]:
    """Run full structural validation; return a summary dict (not a readiness claim)."""
    platform = validate_platform_decision()
    profiler = validate_profiler_contract()
    blocked = validate_blocked_state_registry()
    validate_schedule_references_contracts()
    validate_overview_references_contracts()
    return {
        "ok": True,
        "platform_schema": platform["schema"],
        "profiler_schema": profiler["schema"],
        "blocked_registry_schema": blocked["schema"],
        "bootstrap_statuses": {
            e["id"]: e["status"] for e in blocked["entries"] if isinstance(e, Mapping)
        },
        "live_qwen_values_exist": profiler["live_qwen_values"]["exist"],
        "any_stage_ready": False,
    }


__all__ = [
    "ASCENSION_DIR",
    "BLOCKED_REGISTRY_PATH",
    "BLOCKED_REGISTRY_SCHEMA",
    "GLOBAL_LEDGERS",
    "INITIAL_FAMILIES",
    "MODEL_LADDER_PIPELINE",
    "MUST_NOT_REQUIRE_SHARING",
    "PLATFORM_DECISION_PATH",
    "PLATFORM_SCHEMA",
    "PORTABLE_INTERFACES",
    "PROFILER_CONTRACT_PATH",
    "PROFILER_SCHEMA",
    "REQUIRED_BLOCKER_FIELDS",
    "REQUIRED_BOOTSTRAP_DISPLAY_NAMES",
    "REQUIRED_BOOTSTRAP_ENTRY_IDS",
    "AscensionContractError",
    "entry_is_fully_blocked",
    "load_blocked_state_registry",
    "load_master_schedule",
    "load_platform_decision",
    "load_profiler_contract",
    "validate_all_ascension_contracts",
    "validate_blocked_state_registry",
    "validate_overview_references_contracts",
    "validate_platform_decision",
    "validate_profiler_contract",
    "validate_schedule_references_contracts",
]
