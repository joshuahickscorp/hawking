#!/usr/bin/env python3
"""Family-agnostic HCLI product-test harness scaffold (bible §33).

Generalizes the DeepSeek-V4 diagnostic live-suite pattern already proven in:

* ``lab.operators.deepseek_v4_gravity.hcli_live_suite_receipt``
* ``tools/condense/tests/test_deepseek_v4_hcli_live_suite.py``
* ``workspace/campaign/evidence/hide/deepseek-v4-live.sBqM7r/``

This module does **not** start endpoints, load weights, or claim product
readiness.  It owns:

* the durable §33 case catalog
* mapping from DeepSeek diagnostic evidence filenames → catalog ids
* a privacy-preserving product-suite receipt envelope for future families
* metric field definitions (verified tasks / hour first)

DeepSeek-specific sealing remains in ``deepseek_v4_gravity`` so existing
receipts stay valid.  Future Qwen adapters should feed evidence into
``product_suite_receipt`` (or call the family packer with the same shape).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import seal  # noqa: E402

CATALOG_PATH = (
    REPO_ROOT
    / "workspace"
    / "docs"
    / "plans"
    / "ascension"
    / "ASCENSION_HCLI_PRODUCT_TEST_CATALOG.json"
)
MASTER_SCHEDULE_PATH = (
    REPO_ROOT
    / "workspace"
    / "docs"
    / "plans"
    / "ascension"
    / "ASCENSION_MASTER_SCHEDULE.json"
)
COMPLETION_STATES_PATH = (
    REPO_ROOT
    / "workspace"
    / "docs"
    / "plans"
    / "ascension"
    / "ASCENSION_COMPLETION_STATES.json"
)

PRODUCT_SUITE_SCHEMA = "hawking.gravity.hcli_product_suite.v1"
PRODUCT_SUITE_STATUS_SCAFFOLD = "HCLI_PRODUCT_SUITE_CATALOG_SCAFFOLD_NOT_LIVE"
PRODUCT_SUITE_STATUS_SEALED = "HCLI_PRODUCT_SUITE_EVIDENCE_SEALED_FAMILY_BOUND"

# Bible §33 case order (canonical).
BIBLE_CASE_IDS: tuple[str, ...] = (
    "chat",
    "repo_context",
    "coding",
    "planner_act_verify",
    "tool_calls",
    "structured_json",
    "session_restart",
    "endpoint_restart",
    "context_compaction",
    "read_safe_swarm",
    "isolated_write_agent",
    "continuous_batching",
    "search_retrieval",
    "memory_ops",
    "skill_execution",
    "document_perception",
)

PRIMARY_METRIC = "verified_tasks_completed_per_hour"

AGENT_SPEED_CHAIN: tuple[str, ...] = (
    "question",
    "trustworthy_evidence",
    "implemented_result",
    "verified_result",
)

SECONDARY_METRICS: tuple[str, ...] = (
    "per_agent_latency_ms",
    "aggregate_accepted_tps",
    "patch_acceptance_rate",
    "tests_passed",
    "retries",
    "regressions",
    "human_intervention_count",
    "thermal_stability",
    "search_latency_ms",
    "tool_selection_latency_ms",
    "memory_retrieval_cost",
    "planning_failure_rate",
)

# Evidence filenames relative to the DeepSeek live evidence directory
# (or sibling hide/ dirs via ../). Mirrors the sealed v2 aggregate inputs.
DEEPSEEK_V4_CASE_EVIDENCE: dict[str, tuple[str, ...]] = {
    "chat": ("normal-turn.json",),
    "repo_context": (
        "repo-context-turn.json",
        "repo-source-ingest.json",
        "attached-evidence-turn.json",
    ),
    "coding": ("coding-task-turn.json",),
    "planner_act_verify": (
        "hcli-agent-planner-act-verify-receipt.json",
        "hcli-agent-planner-act-verify-6-receipt.json",
    ),
    "tool_calls": ("hcli-agent-planner-act-verify-receipt.json",),
    "structured_json": ("structured-json-turn.json",),
    "session_restart": (
        "recovery-before-turn.json",
        "recovery-after-turn.json",
        "recovery-session-before.json",
        "recovery-session-after.json",
    ),
    "endpoint_restart": (
        "endpoint-health-after-restart.json",
        "capabilities.json",
    ),
    "context_compaction": (
        "context-compaction-turn.json",
        "compaction-source-ingest.json",
    ),
    "read_safe_swarm": (
        "../deepseek-v4-swarm.NvBh4u/hcli-read-safe-swarm-receipt.json",
    ),
    "isolated_write_agent": (
        "../deepseek-v4-write-smoke.Yz13sy/hcli-isolated-write-smoke-receipt.json",
    ),
    "continuous_batching": (
        "hcli-diagnostic-cpu-benchmark-receipt.json",
        "base-tps-gate-receipt.json",
    ),
    "search_retrieval": (),
    "memory_ops": (),
    "skill_execution": (),
    "document_perception": (
        "attached-evidence-turn.json",
        "repo-source-ingest.json",
    ),
}

CUDA_PRESERVE: tuple[str, ...] = (
    "architecture_ir",
    "gravity_tensor_semantics",
    "kernel_grammar",
    "benchmark_contracts",
    "parity_capability_suites",
    "receipt_schema",
    "scheduler_api",
)


class HcliProductTestHarnessError(ValueError):
    """Invalid catalog, schedule, or product-suite input."""


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HcliProductTestHarnessError(f"missing JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise HcliProductTestHarnessError(f"expected object at root: {path}")
    return data


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or CATALOG_PATH)


def load_master_schedule(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or MASTER_SCHEDULE_PATH)


def load_completion_states(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or COMPLETION_STATES_PATH)


def catalog_case_ids(catalog: Mapping[str, Any] | None = None) -> list[str]:
    doc = catalog if catalog is not None else load_catalog()
    cases = doc.get("cases")
    if not isinstance(cases, list) or not cases:
        raise HcliProductTestHarnessError("catalog.cases must be a non-empty list")
    ids: list[str] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise HcliProductTestHarnessError(f"catalog.cases[{index}] must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise HcliProductTestHarnessError(f"catalog.cases[{index}].id must be a non-empty string")
        ids.append(case_id)
    return ids


def validate_catalog(catalog: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate bible §33 catalog integrity. Returns a structured report."""
    doc = dict(catalog) if catalog is not None else load_catalog()
    errors: list[str] = []
    case_ids = []
    try:
        case_ids = catalog_case_ids(doc)
    except HcliProductTestHarnessError as exc:
        errors.append(str(exc))
        return {"ok": False, "errors": errors, "case_ids": []}

    if tuple(case_ids) != BIBLE_CASE_IDS:
        missing = [c for c in BIBLE_CASE_IDS if c not in case_ids]
        extra = [c for c in case_ids if c not in BIBLE_CASE_IDS]
        if missing:
            errors.append(f"catalog missing bible cases: {missing}")
        if extra:
            errors.append(f"catalog has non-bible cases: {extra}")
        if not missing and not extra and list(case_ids) != list(BIBLE_CASE_IDS):
            errors.append("catalog case order must match bible §33 order")

    if doc.get("primary_metric") != PRIMARY_METRIC:
        errors.append(f"primary_metric must be {PRIMARY_METRIC!r}")

    chain = doc.get("agent_speed_chain")
    if list(chain or []) != list(AGENT_SPEED_CHAIN):
        errors.append("agent_speed_chain must match question→…→verified_result")

    for case_id, expected_files in DEEPSEEK_V4_CASE_EVIDENCE.items():
        if case_id not in case_ids:
            continue
        case = next(c for c in doc["cases"] if c["id"] == case_id)
        listed = tuple(case.get("deepseek_v4_evidence") or [])
        if listed != expected_files:
            errors.append(
                f"deepseek map mismatch for {case_id}: catalog={listed} harness={expected_files}"
            )

    return {
        "ok": not errors,
        "errors": errors,
        "case_ids": case_ids,
        "case_count": len(case_ids),
        "primary_metric": PRIMARY_METRIC,
    }


def validate_master_schedule(schedule: Mapping[str, Any] | None = None) -> dict[str, Any]:
    doc = dict(schedule) if schedule is not None else load_master_schedule()
    errors: list[str] = []
    steps = doc.get("steps")
    if not isinstance(steps, list):
        return {"ok": False, "errors": ["steps must be a list"], "step_ids": []}

    step_ids: list[int] = []
    by_id: dict[int, Mapping[str, Any]] = {}
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            errors.append(f"steps[{index}] must be an object")
            continue
        sid = step.get("id")
        if not isinstance(sid, int):
            errors.append(f"steps[{index}].id must be int")
            continue
        if sid in by_id:
            errors.append(f"duplicate step id {sid}")
        by_id[sid] = step
        step_ids.append(sid)
        if step.get("status") != "NOT_STARTED":
            errors.append(f"step {sid} status must seed as NOT_STARTED (found {step.get('status')!r})")
        prereqs = step.get("prerequisites") or []
        if not isinstance(prereqs, list):
            errors.append(f"step {sid} prerequisites must be a list")
            continue
        for pre in prereqs:
            if not isinstance(pre, int):
                errors.append(f"step {sid} has non-int prerequisite {pre!r}")
            elif pre not in by_id and pre not in step_ids:
                # allow forward refs only if they appear later in the same list
                pass

    expected = list(range(0, 34))
    if step_ids != expected:
        errors.append(f"step ids must be 0..33 in order; got {step_ids[:5]}... count={len(step_ids)}")

    for sid, step in by_id.items():
        for pre in step.get("prerequisites") or []:
            if isinstance(pre, int) and pre not in by_id:
                errors.append(f"step {sid} prerequisite {pre} is not a defined step")
            if isinstance(pre, int) and pre >= sid:
                errors.append(f"step {sid} prerequisite {pre} must be a lower id (DAG)")

    cuda = doc.get("cuda_policy") or {}
    preserve = list(cuda.get("preserve") or [])
    if preserve != list(CUDA_PRESERVE):
        errors.append("cuda_policy.preserve must match bible §34 preserve list")

    return {
        "ok": not errors,
        "errors": errors,
        "step_ids": step_ids,
        "step_count": len(step_ids),
    }


def validate_completion_states(states_doc: Mapping[str, Any] | None = None) -> dict[str, Any]:
    doc = dict(states_doc) if states_doc is not None else load_completion_states()
    errors: list[str] = []
    states = doc.get("states")
    if not isinstance(states, list):
        return {"ok": False, "errors": ["states must be a list"], "state_ids": []}

    expected = [
        "PROTO_FRANKENSTEIN_OFFLOADED",
        "HAWKING_RESEARCH_PORTFOLIO_FROZEN",
        "HCLI_AGENT_OS_FOUNDATION_READY",
        "QWEN3_30B_GRAVITY_READY",
        "QWEN3_30B_EXECUTOR_READY",
        "QWEN3_NEXT_80B_GRAVITY_READY",
        "QWEN3_NEXT_80B_REVIEWER_READY",
        "TG3_REVIEW_REQUIRED",
        "HAWKING_OPTION_C_KERNEL_SANDBOX_READY",
        "HCLI_AGENT_PIPELINE_READY",
        "HAWKING_SELF_CONTAINED_MODEL_LADDER_ACTIVE",
        "HAWKING_APPLE_PRODUCTION_RELEASE_READY",
    ]
    # Order in bible listing differs slightly from dependency order; require set equality + count.
    state_ids: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, state in enumerate(states):
        if not isinstance(state, Mapping):
            errors.append(f"states[{index}] must be an object")
            continue
        sid = state.get("id")
        if not isinstance(sid, str):
            errors.append(f"states[{index}].id must be str")
            continue
        if sid in by_id:
            errors.append(f"duplicate state id {sid}")
        by_id[sid] = state
        state_ids.append(sid)
        if state.get("status") != "CANDIDATE":
            errors.append(f"state {sid} must seed as CANDIDATE")
        if state.get("certified_by") is not None:
            errors.append(f"state {sid} must not be pre-certified")
        certifiers = state.get("may_be_certified_by") or []
        if "controller" not in certifiers and "human" not in certifiers:
            errors.append(f"state {sid} must allow controller/human certification")
        for pre in state.get("prerequisites") or []:
            if pre not in expected and pre not in by_id:
                # checked after full load
                pass

    if set(state_ids) != set(expected) or len(state_ids) != 12:
        errors.append(f"must define exactly the 12 bible §36 states; got {state_ids}")

    for sid, state in by_id.items():
        for pre in state.get("prerequisites") or []:
            if pre not in by_id:
                errors.append(f"state {sid} prerequisite {pre} is undefined")

    auth = doc.get("authority") or {}
    if auth.get("sandbox_models") != "may_report_candidate_only":
        errors.append("authority.sandbox_models must be may_report_candidate_only")

    return {
        "ok": not errors,
        "errors": errors,
        "state_ids": state_ids,
        "state_count": len(state_ids),
    }


def cross_check_schedule_and_states(
    schedule: Mapping[str, Any] | None = None,
    states_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every produces_completion_state on a step must exist in the state machine."""
    sched = schedule if schedule is not None else load_master_schedule()
    states = states_doc if states_doc is not None else load_completion_states()
    errors: list[str] = []
    state_ids = {s["id"] for s in states.get("states", []) if isinstance(s, Mapping)}
    for step in sched.get("steps", []):
        if not isinstance(step, Mapping):
            continue
        produced = step.get("produces_completion_state")
        if produced is None:
            continue
        if produced not in state_ids:
            errors.append(
                f"step {step.get('id')} produces unknown completion state {produced!r}"
            )
    return {"ok": not errors, "errors": errors}


def empty_metrics_envelope() -> dict[str, Any]:
    return {
        "primary_metric": PRIMARY_METRIC,
        "primary_value": None,
        "agent_speed_chain": list(AGENT_SPEED_CHAIN),
        "secondary": {name: None for name in SECONDARY_METRICS},
        "note": "Metrics remain null until a live family suite is sealed with verification authority passes.",
    }


def product_suite_receipt(
    *,
    family: str,
    artifact_seal_sha256: str,
    case_results: Sequence[Mapping[str, Any]],
    out: str | Path | None = None,
    claim_boundary: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    live: bool = False,
) -> dict[str, Any]:
    """Seal a family-bound product-suite index (scaffold or live evidence).

    Unlike the DeepSeek diagnostic packer, this does not read endpoint files
    from disk — callers supply already-redacted case result summaries.  Live
    family adapters should hash prompts before calling.
    """
    if not family or not isinstance(family, str):
        raise HcliProductTestHarnessError("family must be a non-empty string")
    if not isinstance(artifact_seal_sha256, str) or len(artifact_seal_sha256) != 64:
        raise HcliProductTestHarnessError("artifact_seal_sha256 must be a 64-char hex digest")
    try:
        int(artifact_seal_sha256, 16)
    except ValueError as exc:
        raise HcliProductTestHarnessError("artifact_seal_sha256 must be hex") from exc

    known = set(BIBLE_CASE_IDS)
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(case_results):
        if not isinstance(raw, Mapping):
            raise HcliProductTestHarnessError(f"case_results[{index}] must be an object")
        case_id = raw.get("id")
        if case_id not in known:
            raise HcliProductTestHarnessError(f"unknown product case id: {case_id!r}")
        if case_id in seen:
            raise HcliProductTestHarnessError(f"duplicate product case id: {case_id}")
        seen.add(str(case_id))
        if raw.get("prompt_text_disclosed"):
            raise HcliProductTestHarnessError(
                f"case {case_id} must not disclose prompt text in the aggregate"
            )
        normalized.append(
            {
                "id": case_id,
                "status": raw.get("status", "NOT_RUN"),
                "evidence_sha256": raw.get("evidence_sha256"),
                "prompt_hashes": list(raw.get("prompt_hashes") or []),
                "prompt_text_disclosed": False,
                "verification": raw.get("verification"),
                "notes": raw.get("notes"),
            }
        )

    boundary = dict(claim_boundary or {})
    boundary.setdefault("product_promotion", False)
    boundary.setdefault("tg_eligible", False)
    boundary.setdefault("controller_certified", False)
    boundary.setdefault("family", family)

    report = seal(
        {
            "schema": PRODUCT_SUITE_SCHEMA,
            "status": PRODUCT_SUITE_STATUS_SEALED if live else PRODUCT_SUITE_STATUS_SCAFFOLD,
            "family": family,
            "artifact": {"seal_sha256": artifact_seal_sha256},
            "cases": normalized,
            "metrics": dict(metrics or empty_metrics_envelope()),
            "prompt_disclosure": {
                "mode": "hash_only",
                "note": "Prompt text and completions are intentionally omitted.",
            },
            "claim_boundary": boundary,
            "cuda_preserve": list(CUDA_PRESERVE),
            "bible_section": 33,
        }
    )
    if out is not None:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def scaffold_deepseek_diagnostic_mapping() -> dict[str, Any]:
    """Return the DeepSeek diagnostic adapter map used by the product catalog."""
    return {
        "family": "deepseek_v4_layer4_diagnostic",
        "packer": "lab.operators.deepseek_v4_gravity:hcli_live_suite_receipt",
        "receipt_schema": "hawking.gravity.deepseek_v4.hcli_live_suite.v1",
        "claim": "diagnostic_pattern_proof_only_not_tg_or_production",
        "cases": {case_id: list(files) for case_id, files in DEEPSEEK_V4_CASE_EVIDENCE.items()},
    }


def validate_all() -> dict[str, Any]:
    catalog = validate_catalog()
    schedule = validate_master_schedule()
    states = validate_completion_states()
    cross = cross_check_schedule_and_states()
    errors = (
        list(catalog.get("errors") or [])
        + list(schedule.get("errors") or [])
        + list(states.get("errors") or [])
        + list(cross.get("errors") or [])
    )
    return {
        "ok": not errors,
        "errors": errors,
        "catalog": catalog,
        "schedule": schedule,
        "completion_states": states,
        "cross_check": cross,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage: hcli_product_test_harness.py validate|catalog-ids|deepseek-map",
            file=sys.stderr,
        )
        return 2
    command = args[0]
    if command == "validate":
        report = validate_all()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    if command == "catalog-ids":
        print(json.dumps(list(BIBLE_CASE_IDS), indent=2))
        return 0
    if command == "deepseek-map":
        print(json.dumps(scaffold_deepseek_diagnostic_mapping(), indent=2, sort_keys=True))
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
