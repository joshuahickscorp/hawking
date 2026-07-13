#!/usr/bin/env python3.12
"""Fail-closed static integration and adversarial audit for Doctor v5.

This stdlib-only auditor validates the immutable v5 package.  It never reads
live Studio state, loads a model, authorizes execution, or treats a synthetic
fixture as evidence.  Every fixture below is a rejection test.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable

import doctor_v5
import doctor_v5_contract as contract
import doctor_v5_root
import quality_battery_v5
import training_ladder_v5


SCHEMA = "hawking.doctor_v5_audit.v5"
VERSION = "doctor-v5-audit.2"
ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "condense"
DEFAULT_OUTPUT = REPORT_DIR / "doctor_v5_audit.json"
ROOT_PATH = REPORT_DIR / "doctor_v5_root.json"
CAMPAIGN_PATH = REPORT_DIR / "doctor_v5_campaign.json"
LADDER_PATH = REPORT_DIR / "training_ladder_v5.json"
BATTERY_PATH = REPORT_DIR / "quality_battery_v5.json"

SOURCE_PATHS = (
    ROOT / "tools" / "condense" / "doctor_v5_contract.py",
    ROOT / "tools" / "condense" / "quality_battery_v5.py",
    ROOT / "tools" / "condense" / "training_ladder_v5.py",
    ROOT / "tools" / "condense" / "doctor_v5.py",
    ROOT / "tools" / "condense" / "doctor_v5_root.py",
    Path(__file__).resolve(),
)
DOCUMENT_PATHS = (
    ROOT / "docs" / "plans" / "DOCTOR_V5.md",
    ROOT / "docs" / "plans" / "DOCTOR_V5_RESEARCH_PASSES.md",
    ROOT / "docs" / "plans" / "TRAINING_LADDER_V5.md",
)
REPORT_SPECS = (
    ("campaign", CAMPAIGN_PATH, doctor_v5.CAMPAIGN_SCHEMA, "campaign_sha256"),
    ("training_ladder", LADDER_PATH, training_ladder_v5.SCHEMA, "ladder_sha256"),
    ("quality_battery", BATTERY_PATH, quality_battery_v5.SCHEMA, "manifest_sha256"),
)

FORBIDDEN_IMPORT_ROOTS = {
    "subprocess", "multiprocessing", "torch", "tensorflow", "jax", "mlx",
    "transformers", "vllm", "llama_cpp", "accelerate",
}
FORBIDDEN_CALLS = {
    "exec", "eval", "__import__", "os.system", "os.popen", "os.fork",
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "multiprocessing.Process", "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell", "importlib.import_module",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _stamp(document: dict[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result.pop(field, None)
    result[field] = contract.hash_value(result)
    return result


def _stamp_battery(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result["manifest_sha256"] = contract.hash_value({
        key: value for key, value in result.items()
        if key not in {"manifest_sha256", "generated_at"}
    })
    return result


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _ast_audit(paths: Iterable[Path]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(f"AST parse failed for {_relative(path)}: {exc}")
            continue
        imports: set[str] = set()
        calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Call):
                name = _dotted(node.func)
                if name in FORBIDDEN_CALLS or (
                    name and any(name.startswith(root + ".") for root in FORBIDDEN_IMPORT_ROOTS)
                ):
                    calls.append(f"{name}@{getattr(node, 'lineno', '?')}")
        forbidden_imports = sorted(imports & FORBIDDEN_IMPORT_ROOTS)
        if forbidden_imports or calls:
            errors.append(
                f"{_relative(path)} is not execution-free: imports={forbidden_imports} calls={calls}"
            )
        rows.append({
            "path": _relative(path), "file_sha256": _file_sha256(path),
            "syntax_valid": True, "forbidden_imports": forbidden_imports,
            "forbidden_calls": calls,
        })
    return {"passed": not errors, "files": rows}, errors


def _load_inputs() -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    errors: list[str] = []
    for label, path in (
        ("root", ROOT_PATH), ("campaign", CAMPAIGN_PATH),
        ("training_ladder", LADDER_PATH), ("quality_battery", BATTERY_PATH),
    ):
        try:
            values[label] = _read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"required input {label} unavailable: {_relative(path)}: {exc}")
    return values, errors


def _root_cross_identity(
    root: dict[str, Any], inputs: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    actual = root.get("report_bindings")
    rows = {row.get("id"): row for row in actual if isinstance(row, dict)} \
        if isinstance(actual, list) else {}
    exact = len(rows) == len(REPORT_SPECS) == len(actual or [])
    for label, path, schema, identity_field in REPORT_SPECS:
        document = inputs.get(label)
        expected = {
            "id": label, "path": _relative(path), "schema": schema,
            "identity_field": identity_field,
            "document_identity": document.get(identity_field) if isinstance(document, dict) else None,
            "file_sha256": _file_sha256(path) if path.is_file() else None,
        }
        if rows.get(label) != expected:
            exact = False
            errors.append(f"root report binding for {label} is not the exact current document/file identity")
    campaign = inputs.get("campaign", {})
    immutable = campaign.get("campaign_input_binding", {}).get("input_reports", {})
    for label, path, _schema, identity_field in REPORT_SPECS[1:]:
        key = "training_ladder" if label == "training_ladder" else "quality_battery"
        document = inputs.get(label, {})
        row = immutable.get(key, {})
        if row.get("document_identity_sha256") != document.get(identity_field) \
                or row.get("file_sha256") != (_file_sha256(path) if path.is_file() else None):
            exact = False
            errors.append(f"campaign immutable input binding for {key} drifted")
    policy = root.get("execution_policy", {})
    flags_false = all(policy.get(field) is False for field in (
        "execution_authorized", "greenlight_recorded", "evidence_complete", "dominance_proven"
    ))
    if not flags_false:
        errors.append("root identity attempts to authorize execution/evidence/dominance")
    return {
        "passed": exact and flags_false,
        "root_manifest_sha256": root.get("root_manifest_sha256"),
        "root_file_sha256": _file_sha256(ROOT_PATH) if ROOT_PATH.is_file() else None,
        "report_bindings_exact": exact,
        "root_authority_flags_false": flags_false,
    }, errors


def _grid_audit(ladder: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    models = [row.get("name") for row in ladder.get("models", [])]
    rates = [row.get("physical_bpw_ceiling") for row in ladder.get("rate_profiles", [])]
    scopes = [row.get("id") for row in ladder.get("claim_tracks", [])]
    observed = {
        (row.get("model"), row.get("physical_bpw_ceiling"), row.get("claim_track"))
        for row in ladder.get("lanes", []) if isinstance(row, dict)
    }
    expected = {(model, rate, scope) for model in models for rate in rates for scope in scopes}
    passed = (
        len(models) == len(set(models)) == 32
        and rates == list(doctor_v5.RATE_POINTS) and len(rates) == 10
        and scopes == list(contract.CLAIM_SCOPES) and len(scopes) == 4
        and observed == expected and len(ladder.get("lanes", [])) == 1280
    )
    if not passed:
        errors.append(
            f"training ladder is not the exact 32x10x4 cross product: "
            f"models={len(models)} rates={len(rates)} scopes={len(scopes)} lanes={len(observed)}"
        )
    return {
        "passed": passed, "models": len(models), "rates": len(rates),
        "claim_tracks": len(scopes), "expected_lanes": 1280,
        "observed_lanes": len(ladder.get("lanes", [])),
        "missing_cells": len(expected - observed), "extra_cells": len(observed - expected),
    }, errors


def _quality_contracts(
    campaign: dict[str, Any], ladder: dict[str, Any], battery: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    canonical_battery = quality_battery_v5.compile_manifest()
    suites = battery.get("suites", [])
    suite_exact = len(suites) == 21 and suites == canonical_battery["suites"]
    if not suite_exact:
        errors.append(f"quality battery must contain the exact 21 canonical suites; found {len(suites)}")
    coverage = battery.get("competitor_coverage")
    direct = coverage.get("direct_implementations") if isinstance(coverage, dict) else None
    campaign_direct = campaign.get("direct_competitor_requirements", {})
    ladder_comp = ladder.get("competitor_requirements", {})
    competitor_exact = bool(direct) and (
        campaign_direct.get("competitor_coverage") == coverage
        and campaign_direct.get("required_methods") == direct
        and ladder_comp.get("required_direct_implementations") == direct
        and ladder_comp.get("required_families") == coverage.get("required_families")
        and ladder_comp.get("required_implementation_ids_by_rate_bpw")
        == coverage.get("required_implementation_ids_by_rate_bpw")
    )
    if not competitor_exact:
        errors.append("campaign/ladder/battery do not embed the exact same direct competitor registry")
    battery_compute = battery.get("matched_test_time_compute")
    ladder_compute = ladder.get("matched_test_time_compute")
    campaign_embedded = campaign.get("campaign_input_binding", {}).get(
        "quality_battery_matched_test_time_compute"
    )
    fields = battery_compute.get("required_fields", {}) if isinstance(battery_compute, dict) else {}
    flattened = [field for group in fields.values() if isinstance(group, list) for field in group]
    compute_exact = (
        battery_compute == canonical_battery["matched_test_time_compute"]
        and ladder_compute == battery_compute and campaign_embedded == battery_compute
        and len(flattened) == len(set(flattened))
        and campaign.get("test_time_compute_contract") == doctor_v5._test_time_compute_contract()
    )
    if not compute_exact:
        errors.append("complete matched-compute registry/fields are not identical across v5 inputs")
    return {
        "passed": suite_exact and competitor_exact and compute_exact,
        "suite_count": len(suites), "exact_21_suites": suite_exact,
        "direct_competitor_count": len(direct or []),
        "exact_competitor_registry_equality": competitor_exact,
        "matched_compute_group_count": len(fields),
        "matched_compute_field_count": len(flattened),
        "complete_matched_compute_equality": compute_exact,
    }, errors


def _fail_closed_audit(
    root: dict[str, Any], campaign: dict[str, Any], ladder: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    bindings = campaign.get("parameter_count_bindings", [])
    unresolved = [row.get("model") for row in bindings if row.get("status") != "verified_exact"]
    unresolved_exact = len(bindings) == 32 and len(unresolved) == 32 and all(
        row.get("exact_parameter_count") is None
        and row.get("source_manifest_sha256") == "required"
        and row.get("usable_as_bpw_denominator") is False for row in bindings
    )
    candidates = campaign.get("candidates", [])
    mechanisms = campaign.get("mechanism_registry", [])
    lanes = ladder.get("lanes", [])
    stages = ladder.get("stages", [])
    launchable = (
        sum(row.get("launchable") is not False for row in candidates)
        + sum(row.get("launch_permitted") is not False for row in lanes)
        + sum(row.get("launch_permitted") is not False for row in stages)
        + sum(row.get("executor", {}).get("wired") is not False for row in mechanisms)
    )
    blockers_exact = all(
        "exact_parameter_count_source_manifest_missing" in row.get("launch_blockers", [])
        for row in candidates
    ) if candidates else False
    passed = unresolved_exact and blockers_exact and launchable == 0 \
        and root.get("execution_policy", {}).get("execution_authorized") is False
    if not passed:
        errors.append(
            "fail-closed gate failed: exact parameter blockers, unwired executors, or zero-launchable invariant"
        )
    return {
        "passed": passed, "unresolved_exact_parameter_models": unresolved,
        "unresolved_exact_parameter_count": len(unresolved),
        "every_candidate_has_exact_parameter_blocker": blockers_exact,
        "launchable_or_wired_violations": launchable,
    }, errors


def _materialization_audit(campaign: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Exercise all scope x failure routes without creating executable programs."""
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    candidates = campaign.get("candidates", [])
    for scope in contract.CLAIM_SCOPES:
        for failure_class in doctor_v5.FAILURE_ROUTES:
            candidate = next((
                row for row in candidates
                if row.get("claim_scope") == scope
                and row.get("failure_class") == failure_class
                and row.get("mandatory_control") is False
            ), None)
            label = f"{scope}::{failure_class}"
            if candidate is None:
                errors.append(f"representative planned candidate missing: {label}")
                continue
            try:
                program = doctor_v5.materialize_program(
                    campaign, candidate["candidate_identity_sha256"]
                )
            except (AssertionError, KeyError, ValueError) as exc:
                errors.append(f"representative materialization failed {label}: {exc}")
                continue
            found = contract.validate_program(program)
            kinds = {node.get("kind") for node in program.get("operators", [])}
            if found:
                errors.extend(f"materialized {label}: {item}" for item in found)
            if program.get("mode") != "planned" \
                    or program.get("campaign_metadata", {}).get("launch_permitted") is not False:
                errors.append(f"materialized {label} crossed planned/unlaunchable boundary")
            if program.get("package_root_schema") != "required" \
                    or program.get("package_root_manifest_sha256") != "required":
                errors.append(f"materialized {label} lacks planned package-root binding")
            if any(node.get("executor", {}).get("wired") is not False
                   for node in program.get("operators", [])):
                errors.append(f"materialized {label} wired an executor")
            if failure_class == "computation_collapse" and "reconstruct" not in kinds:
                errors.append(f"materialized {label} lacks structural reconstruction")
            rows.append({
                "claim_scope": scope, "failure_class": failure_class,
                "program_sha256": program.get("program_sha256"),
                "program_spec_sha256": program.get("program_spec_sha256"),
                "package_root_manifest_sha256": program.get("package_root_manifest_sha256"),
                "launch_permitted": False, "executor_wired": False,
                "contract_valid": not found,
            })
    return {
        "passed": not errors and len(rows) == 8,
        "representative_count": len(rows), "scope_failure_cross_product": rows,
        "execution_authorized": False, "trusted_experimental_evidence": False,
    }, errors


def _fixture_program(
    scope: str = "restorative_training", *, active_b: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    operators = [
        contract._sample_operator("diagnose", "diagnose", []),
        contract._sample_operator("represent", "represent", ["diagnose"]),
        contract._sample_operator("reconstruct", "reconstruct", ["represent"]),
        contract._sample_operator("repair", "repair_static", ["reconstruct"]),
        contract._sample_operator("package", "package", ["repair"]),
        contract._sample_operator("evaluate", "evaluate", ["package"]),
    ]
    planned = contract.planned_program(
        experiment_id=f"dv5-audit-{scope}", candidate_identity_sha256="b" * 64,
        model_id="audit-7b", params_b=7.0, active_b=active_b, target_bpw=0.5,
        claim_scope=scope, failure_class="computation_collapse", operators=operators,
    )
    executable = contract._sample_executable(planned)
    return planned, executable


def _security_envelope(program: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "program_sha256": program["program_sha256"],
        "training_seed_manifest_sha256": program["training_contract"][
            "training_seed_manifest_sha256"
        ],
        "calibration_draw_manifest_sha256": program["training_contract"][
            "calibration_draw_manifest_sha256"
        ],
        "exact_parameter_count": program["model"]["exact_parameter_count"],
        "exact_parameter_count_source_sha256": program["model"].get(
            "exact_parameter_count_source_sha256"
        ),
        "test_time_compute_budget_sha256": contract.hash_value(
            program["evaluation_contract"]["test_time_compute_budget"]
        ),
        "teacher_contract_sha256": contract.hash_value(
            program["training_contract"].get("teacher_contract")
        ),
        "package_root_manifest_sha256": program.get("package_root_manifest_sha256"),
    }
    return contract.stamp(payload, "authorization_envelope_sha256")


def _authorization_replay_errors(program: dict[str, Any], envelope: dict[str, Any]) -> list[str]:
    current = _security_envelope(program)
    errors: list[str] = []
    for field in (
        "program_sha256", "training_seed_manifest_sha256", "calibration_draw_manifest_sha256",
        "exact_parameter_count", "exact_parameter_count_source_sha256",
        "test_time_compute_budget_sha256", "teacher_contract_sha256",
        "package_root_manifest_sha256",
    ):
        if current.get(field) != envelope.get(field):
            errors.append(f"authorization replay rejected: frozen {field} changed")
    return errors


def _power_errors(observation: dict[str, Any]) -> list[str]:
    return [
        f"underpowered PROVEN claim rejected: {domain}.n={row.get('n')}"
        for domain, row in observation.get("capability_metrics", {}).items()
        if not isinstance(row.get("n"), int) or row["n"] <= 1
    ]


def _significance_errors(observation: dict[str, Any]) -> list[str]:
    if observation.get("evidence_state") != "PROVEN":
        return []
    alpha = observation.get("statistical_contract", {}).get("familywise_alpha", 0.05)
    errors: list[str] = []
    for comparison in observation.get("competitor_comparisons", []):
        if comparison.get("macro_holm_adjusted_p_value", 1.0) > alpha:
            errors.append("PROVEN dominance rejected: macro Holm-adjusted p-value is nonsignificant")
        for domain, value in comparison.get("domain_holm_adjusted_p_values", {}).items():
            if value > alpha:
                errors.append(f"PROVEN dominance rejected: {domain} Holm p-value is nonsignificant")
    return errors


def _teacher_errors(program: dict[str, Any]) -> list[str]:
    training = program.get("training_contract", {})
    teachers = training.get("teacher_contract", {})
    parent = teachers.get("parent_teacher") if isinstance(teachers, dict) else None
    elevation = teachers.get("elevation_teacher") if isinstance(teachers, dict) else None
    errors: list[str] = []
    if program.get("claim_scope") == "restorative_training" and elevation not in (None, "forbidden"):
        errors.append("restorative claim contaminated by a stronger/elevation teacher")
    if isinstance(elevation, dict) and not contract.is_sha256(
        elevation.get("provenance_manifest_sha256")
    ):
        errors.append("stronger teacher lacks immutable provenance")
    if not isinstance(parent, dict):
        errors.append("parent teacher provenance is absent")
    return errors


def _package_root_errors(program: dict[str, Any]) -> list[str]:
    value = program.get("package_root_manifest_sha256")
    if program.get("mode") == "planned":
        return [] if value == "required" else ["planned package-root binding drifted from required"]
    execution_root = program.get("execution_contract", {}).get("root_manifest", {}).get(
        "manifest_sha256"
    )
    return [] if value == execution_root and contract.is_sha256(value) else [
        "executable package-root binding differs from authorized root"
    ]


def _rejection_row(
    fixture_id: str, validation_errors: list[str], *, dominance_passed: bool = False,
    targeted: Callable[[list[str]], bool] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    targeted_pass = targeted(validation_errors) if targeted else bool(validation_errors)
    rejected = bool(validation_errors) and not dominance_passed and targeted_pass
    row = {
        "id": fixture_id, "rejected": rejected, "accepted": False,
        "dominance_passed": dominance_passed, "targeted_guard_observed": targeted_pass,
        "error_count": len(validation_errors),
        "errors_sha256": contract.hash_value(validation_errors),
        "representative_errors": validation_errors[:8],
        "synthetic_adversarial_fixture_only": True,
        "trusted_experimental_evidence": False,
    }
    return row, ([] if rejected else [f"adversarial fixture was not rejected: {fixture_id}"])


def _adversarial_fixtures(
    campaign: dict[str, Any] | None, battery: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    planned, executable = _fixture_program()
    program_trust = contract._sample_trust_context(executable)
    observation = contract._sample_observation(executable)
    full_trust = contract._sample_trust_context(executable, observation)

    def add(name: str, found: list[str], dominance: bool = False,
            target: Callable[[list[str]], bool] | None = None) -> None:
        row, errs = _rejection_row(name, found, dominance_passed=dominance, targeted=target)
        rows.append(row)
        failures.extend(errs)

    planned_proven = copy.deepcopy(observation)
    planned_proven["program_sha256"] = planned["program_sha256"]
    planned_proven = contract.stamp(planned_proven, "observation_sha256")
    found = contract.validate_observation(planned_proven, planned)
    add("planned_plus_PROVEN", found, contract.dominance_decision(planned_proven, planned)["passed"])

    failed = copy.deepcopy(observation)
    failed.update({"status": "failed_terminal", "proof_state": "independent_reproduction", "evidence_state": "PROVEN"})
    failed = contract.stamp(failed, "observation_sha256")
    found = contract.validate_observation(failed, executable, trust_context=full_trust)
    add("failed_terminal_plus_PROVEN", found, target=lambda e: any("transition" in x for x in e))

    tiny = copy.deepcopy(observation)
    for metric in tiny["capability_metrics"].values():
        metric["n"] = 1
    tiny = contract.stamp(tiny, "observation_sha256")
    found = contract.validate_observation(tiny, executable, trust_context=full_trust) + _power_errors(tiny)
    add("n_equals_1_underpowered_PROVEN", found, target=lambda e: any("underpowered" in x for x in e))

    forged = copy.deepcopy(observation)
    forged["capability_metrics"]["coding"]["delta"] = 0.5
    forged = contract.stamp(forged, "observation_sha256")
    found = contract.validate_observation(forged, executable, trust_context=full_trust)
    add("arithmetic_delta_forgery", found, target=lambda e: any("candidate minus parent" in x for x in e))

    found = contract.validate_observation(observation, executable)
    add("missing_external_trust", found, target=lambda e: any("external trust" in x for x in e))
    mismatched = {role: {"0" * 64} for role in contract.TRUST_CONTEXT_ROLES}
    found = contract.validate_observation(observation, executable, trust_context=mismatched)
    add("mismatched_external_trust", found, target=lambda e: any("external trust" in x for x in e))

    missing_comp = copy.deepcopy(observation)
    missing_comp["competitor_comparisons"].pop()
    missing_comp = contract.stamp(missing_comp, "observation_sha256")
    found = contract.validate_observation(missing_comp, executable, trust_context=full_trust)
    add("missing_direct_competitor", found, target=lambda e: any("competitor" in x and "missing" in x for x in e))

    unwired = copy.deepcopy(executable)
    unwired["operators"][0]["executor"]["wired"] = False
    unwired = contract.stamp(unwired, "program_sha256")
    found = contract.validate_program(unwired, allow_planned=False, trust_context=program_trust)
    add("executable_unwired", found, target=lambda e: any("must be wired" in x for x in e))

    nonsig = copy.deepcopy(observation)
    for comparison in nonsig["competitor_comparisons"]:
        comparison["macro_raw_p_value"] = 0.8
        comparison["macro_holm_adjusted_p_value"] = 0.9
        comparison["domain_raw_p_values"] = {d: 0.8 for d in contract.CAPABILITY_DOMAINS}
        comparison["domain_holm_adjusted_p_values"] = {d: 0.9 for d in contract.CAPABILITY_DOMAINS}
    nonsig = contract.stamp(nonsig, "observation_sha256")
    found = contract.validate_observation(nonsig, executable, trust_context=full_trust) + _significance_errors(nonsig)
    decision = contract.dominance_decision(nonsig, executable, trust_context=full_trust)
    add("p_0_8_0_9_Holm_nonsignificant_PROVEN", found, decision["passed"],
        lambda e: any("nonsignificant" in x for x in e))

    envelope = _security_envelope(executable)
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("seed", lambda p: p["training_contract"]["training_seeds"].__setitem__(0, 999)),
        ("parameter", lambda p: p["model"].__setitem__("exact_parameter_count", 7_000_000_001)),
        ("compute", lambda p: p["evaluation_contract"]["test_time_compute_budget"].__setitem__("max_output_tokens", 16384)),
        ("teacher", lambda p: p["training_contract"]["teacher_contract"]["parent_teacher"].__setitem__(
            "identity_sha256", "9" * 64
        )),
    ]
    for label, mutate in mutations:
        replay = copy.deepcopy(executable)
        mutate(replay)
        replay = contract.stamp(replay, "program_sha256")
        found = contract.validate_program(replay, allow_planned=False, trust_context=program_trust)
        found += _authorization_replay_errors(replay, envelope)
        add(f"authorization_replay_after_{label}_mutation", found,
            target=lambda e, label=label: any(label in x or "program_sha256" in x for x in e))

    self_asserted = copy.deepcopy(executable)
    self_asserted["model"]["exact_parameter_count"] = int(self_asserted["model"]["params_b"] * 1e9)
    self_asserted["model"]["exact_parameter_count_source_sha256"] = contract.hash_value(
        {"self_asserted_from_rounded_params_b": True}
    )
    self_asserted = contract.stamp(self_asserted, "program_sha256")
    found = _authorization_replay_errors(self_asserted, envelope)
    found.append("self-asserted exact parameter count/manifest is not an external tensor receipt")
    add("self_asserted_exact_parameter_count_and_manifest", found,
        target=lambda e: any("self-asserted" in x for x in e))

    contaminated = copy.deepcopy(executable)
    contaminated["training_contract"]["teacher_contract"]["elevation_teacher"] = {
        "identity_sha256": "9" * 64,
        "revision_sha256": "8" * 64,
        "role": "capability_elevation",
        "output_protocol_sha256": "7" * 64,
        "cache_manifest_sha256": "6" * 64,
        "split_manifest_sha256": "5" * 64,
        "provenance_manifest_sha256": None,
        "training_only": True,
        "authorization_receipt": "required",
    }
    contaminated["training_contract"]["elevation_teacher_allowed"] = True
    contaminated = contract.stamp(contaminated, "program_sha256")
    found = contract.validate_program(contaminated, allow_planned=False, trust_context=program_trust)
    found += _teacher_errors(contaminated)
    add("restorative_stronger_teacher_contamination_missing_provenance", found,
        target=lambda e: any("teacher" in x for x in e))

    planned_drift = copy.deepcopy(planned)
    planned_drift["package_root_schema"] = doctor_v5_root.SCHEMA
    planned_drift["package_root_manifest_sha256"] = "8" * 64
    planned_drift = contract.stamp(planned_drift, "program_sha256")
    found = _package_root_errors(planned_drift)
    executable_drift = copy.deepcopy(executable)
    executable_drift["package_root_manifest_sha256"] = "8" * 64
    executable_drift = contract.stamp(executable_drift, "program_sha256")
    found += _package_root_errors(executable_drift)
    add("planned_executable_package_root_drift", found,
        target=lambda e: len([x for x in e if "package-root" in x]) == 2)

    if battery:
        damaged = copy.deepcopy(battery)
        damaged["matched_test_time_compute"]["freeze_before_final"] = False
        damaged = _stamp_battery(damaged)
        found = quality_battery_v5.validate_manifest(damaged)
        add("restamped_quality_battery_mutation", found,
            target=lambda e: any("canonical" in x or "matched" in x for x in e))

    if campaign:
        damaged = copy.deepcopy(campaign)
        damaged["execution_policy"]["greenlight_recorded"] = True
        damaged = _stamp(damaged, "campaign_sha256")
        found = doctor_v5.validate_campaign(damaged)
        add("restamped_campaign_mutation", found)

        moved = copy.deepcopy(campaign)
        removed = next(row for row in moved["candidates"] if row["mandatory_control"])
        removed_lane = (removed["model"], removed["target_bpw"], removed["claim_scope"], removed["failure_class"])
        added = next(row for row in moved["candidates"] if not row["mandatory_control"] and (
            row["model"], row["target_bpw"], row["claim_scope"], row["failure_class"]
        ) != removed_lane)
        removed["mandatory_control"], removed["control_type"] = False, None
        removed["candidate_identity_sha256"] = contract.hash_value(doctor_v5._candidate_identity_payload(removed))
        removed["experiment_id"] = "dv5-" + removed["candidate_identity_sha256"][:20]
        added["mandatory_control"], added["control_type"] = True, "untreated_same_rate"
        added["candidate_identity_sha256"] = contract.hash_value(doctor_v5._candidate_identity_payload(added))
        added["experiment_id"] = "dv5-" + added["candidate_identity_sha256"][:20]
        moved = _stamp(moved, "campaign_sha256")
        found = doctor_v5.validate_campaign(moved)
        add("relocated_mandatory_control", found,
            target=lambda e: any("control" in x and ("topology" in x or "cross product" in x) for x in e))

        approximate = copy.deepcopy(campaign)
        binding = approximate["parameter_count_bindings"][0]
        binding["exact_parameter_count"] = int(binding["nominal_params_b"] * 1e9)
        binding["parameter_count_binding_sha256"] = contract.hash_value({
            key: value for key, value in binding.items() if key != "parameter_count_binding_sha256"
        })
        approximate = _stamp(approximate, "campaign_sha256")
        found = doctor_v5.validate_campaign(approximate)
        add("approximate_params_b_promoted_to_exact", found,
            target=lambda e: any("parameter" in x for x in e))
    return rows, failures


def _refresh_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(artifact)
    result["byte_ledger"] = {component: 0 for component in contract.ARTIFACT_COMPONENTS}
    for row in result["files"]:
        result["byte_ledger"][row["component"]] += row["bytes"]
    result["component_manifests"] = {}
    for component in contract.ARTIFACT_COMPONENTS:
        payload = [
            {"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]}
            for row in result["files"] if row["component"] == component
        ]
        result["component_manifests"][component] = {
            "bytes": sum(row["bytes"] for row in payload), "file_count": len(payload),
            "files_sha256": contract.hash_value(payload),
        }
    return contract.stamp(result, "artifact_sha256")


def _artifact_fixtures() -> tuple[dict[str, Any], list[str]]:
    _planned, executable = _fixture_program()
    trust = contract._sample_trust_context(executable)
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "artifact.bin"
        path.write_bytes(b"x")
        artifact = contract._sample_artifact(executable)
        artifact["files"][0]["sha256"] = hashlib.sha256(b"x").hexdigest()
        artifact = _refresh_artifact(artifact)
        baseline = contract.validate_artifact(
            artifact, executable, verify_files=True, base_dir=root, trust_context=trust
        )
        if baseline:
            errors.extend("strict artifact baseline: " + item for item in baseline)
        for name, mutate in (
            ("nonexistent_path", lambda a: a["files"][0].__setitem__("path", "missing.bin")),
            ("hash_tamper", lambda a: a["files"][0].__setitem__("sha256", "0" * 64)),
        ):
            damaged = copy.deepcopy(artifact)
            mutate(damaged)
            damaged = _refresh_artifact(damaged)
            found = contract.validate_artifact(
                damaged, executable, verify_files=True, base_dir=root, trust_context=trust
            )
            row, failed = _rejection_row("artifact_" + name, found)
            rows.append(row); errors.extend(failed)
        category = copy.deepcopy(artifact)
        category["files"][0]["component"] = "metadata"
        category = contract.stamp(category, "artifact_sha256")
        found = contract.validate_artifact(
            category, executable, verify_files=True, base_dir=root, trust_context=trust
        )
        row, failed = _rejection_row("artifact_path_category_tamper", found)
        rows.append(row); errors.extend(failed)
        path.write_bytes(b"tampered")
        found = contract.validate_artifact(
            artifact, executable, verify_files=True, base_dir=root, trust_context=trust
        )
        row, failed = _rejection_row("artifact_actual_byte_tamper", found)
        rows.append(row); errors.extend(failed)
    return {
        "passed": not errors, "strict_verify_files_baseline_passed": not baseline,
        "real_temporary_component_files": True, "trust_roles": list(contract.TRUST_CONTEXT_ROLES),
        "rejection_fixtures": rows, "trusted_experimental_evidence": False,
    }, errors


def _resign_smuggled_observation(
    program: dict[str, Any], injection: str,
) -> dict[str, Any]:
    """Rebuild every dependent receipt after inserting one unknown field."""
    base = contract._sample_observation(program)
    old_evidence = base["evidence_bundle"]
    observation = copy.deepcopy(base)
    observation.pop("observation_sha256", None)
    observation.pop("evidence_bundle", None)

    if injection == "observation.top":
        observation["unsafe_override"] = True
    elif injection == "observation.metric":
        observation["capability_metrics"][contract.CAPABILITY_DOMAINS[0]]["unsafe_override"] = True
    elif injection == "observation.statistics":
        observation["statistical_contract"]["unsafe_override"] = True
    elif injection == "observation.comparison":
        observation["competitor_comparisons"][0]["unsafe_override"] = True
    elif injection == "observation.claim_snapshot":
        observation["claim_snapshot"]["unsafe_override"] = True

    compute_payload = {
        key: value for key, value in observation["test_time_compute_receipt"].items()
        if key != "receipt_sha256"
    }
    if injection == "observation.compute_receipt":
        compute_payload["unsafe_override"] = True
    observation["test_time_compute_receipt"] = contract.stamp(
        compute_payload, "receipt_sha256"
    )

    artifact_payload = {
        key: value for key, value in old_evidence["artifact_validation_receipt"].items()
        if key != "receipt_sha256"
    }
    if injection == "evidence.artifact_receipt":
        artifact_payload["unsafe_override"] = True
    artifact_receipt = contract.stamp(artifact_payload, "receipt_sha256")

    raw = {
        key: copy.deepcopy(value) for key, value in old_evidence.items()
        if key not in {
            "artifact_validation_receipt", "raw_evidence_index_sha256",
            "corrected_test_receipt", "evidence_verifier_receipt",
            "sealed_service_attestation", "independent_owner_attestation",
        }
    }
    raw["artifact_validation_receipt"] = artifact_receipt
    raw_index_payload = {
        field: raw[field] for field in (
            "per_item_outputs_sha256", "cluster_assignments_sha256", "cluster_outputs_sha256",
            "data_firewall_receipt_sha256", "parent_parity_receipt_sha256",
            "calibration_draw_results_sha256",
        )
    } | {
        "artifact_validation_receipt_sha256": artifact_receipt["receipt_sha256"],
        "training_seed_results_sha256": raw["training_seed_results_sha256"],
        "training_seed_count": raw["training_seed_count"],
        "calibration_draw_count": raw["calibration_draw_count"],
    }
    raw["raw_evidence_index_sha256"] = contract.hash_value(raw_index_payload)
    summary_sha256 = contract.hash_value(contract._summary_payload(observation))
    comparisons = observation["competitor_comparisons"]
    metrics = observation["capability_metrics"]
    raw_p_payload = {
        "parent": {domain: metrics[domain]["raw_p_value"] for domain in contract.CAPABILITY_DOMAINS},
        "competitors": {
            row["competitor_id"]: {
                "domains": row["domain_raw_p_values"], "macro": row["macro_raw_p_value"],
            } for row in comparisons
        },
    }
    corrected_p_payload = {
        "parent": {
            domain: metrics[domain]["holm_adjusted_p_value"]
            for domain in contract.CAPABILITY_DOMAINS
        },
        "competitors": {
            row["competitor_id"]: {
                "domains": row["domain_holm_adjusted_p_values"],
                "macro": row["macro_holm_adjusted_p_value"],
            } for row in comparisons
        },
    }
    corrected_payload = {
        "procedure": "holm", "familywise_alpha": 0.05,
        "family_size": len(contract.CAPABILITY_DOMAINS)
        + len(comparisons) * (len(contract.CAPABILITY_DOMAINS) + 1),
        "raw_p_values_sha256": contract.hash_value(raw_p_payload),
        "corrected_p_values_sha256": contract.hash_value(corrected_p_payload),
        "summary_sha256": summary_sha256, "all_hypotheses_corrected": True,
    }
    if injection == "evidence.corrected_receipt":
        corrected_payload["unsafe_override"] = True
    corrected = contract.stamp(corrected_payload, "receipt_sha256")

    verifier_payload = {
        "raw_evidence_index_sha256": raw["raw_evidence_index_sha256"],
        "summary_sha256": summary_sha256,
        "corrected_test_receipt_sha256": corrected["receipt_sha256"],
        "verifier_source_sha256": "e" * 64,
        "owner_id": "selftest-independent-owner",
        "recomputed_deltas_cis_pvalues_from_raw": True, "passed": True,
    }
    if injection == "evidence.verifier_receipt":
        verifier_payload["unsafe_override"] = True
    verifier = contract.stamp(verifier_payload, "receipt_sha256")

    sealed_payload = {
        "service_id": "selftest-sealed-service", "owner_id": "selftest-sealed-owner",
        "sealed_final_manifest_sha256": program["data_contract"]["splits"]["sealed_final"][
            "manifest_sha256"
        ],
        "program_sha256": program["program_sha256"],
        "artifact_sha256": observation["artifact_manifest_sha256"],
        "raw_evidence_index_sha256": raw["raw_evidence_index_sha256"],
        "execution_receipt_sha256": "f" * 64, "one_time_nonce_sha256": "1" * 64,
        "consumed_once": True,
    }
    if injection == "evidence.sealed_attestation":
        sealed_payload["unsafe_override"] = True
    sealed = contract.stamp(sealed_payload, "attestation_sha256")

    independent_payload = {
        "owner_id": "selftest-independent-owner", "program_sha256": program["program_sha256"],
        "artifact_sha256": observation["artifact_manifest_sha256"],
        "raw_evidence_index_sha256": raw["raw_evidence_index_sha256"],
        "summary_sha256": summary_sha256,
        "sealed_attestation_sha256": sealed["attestation_sha256"],
        "replication_receipt_sha256": "2" * 64,
        "independently_executed": True, "no_shared_runtime_owner": True,
    }
    if injection == "evidence.independent_attestation":
        independent_payload["unsafe_override"] = True
    independent = contract.stamp(independent_payload, "attestation_sha256")
    raw["corrected_test_receipt"] = corrected
    raw["evidence_verifier_receipt"] = verifier
    raw["sealed_service_attestation"] = sealed
    raw["independent_owner_attestation"] = independent
    if injection == "evidence.bundle":
        raw["unsafe_override"] = True
    observation["evidence_bundle"] = raw
    return contract.stamp(observation, "observation_sha256")


def _schema_smuggling_row(
    fixture_id: str, found: list[str], *, dominance_passed: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    schema_markers = (
        "unsafe_override", "unknown field", "unexpected field", "non-canonical key",
        "exact v5", "exact field set", "must use exactly", "must contain exactly",
        "must exactly", "must exactly name", "must exactly partition",
    )
    schema_error = any(marker in error for marker in schema_markers for error in found)
    stale_markers = (
        "program_spec_sha256 missing or mismatched", "program_sha256 missing or mismatched",
        "observation_sha256 missing or mismatched", "artifact_sha256 missing or mismatched",
        "external trust context does not authorize", "receipt is missing or self-inconsistent",
        "attestation is missing or self-inconsistent",
    )
    freshness = not any(marker in error for marker in stale_markers for error in found)
    row, errors = _rejection_row(
        fixture_id, found, dominance_passed=dominance_passed,
        targeted=lambda _errors: schema_error and freshness,
    )
    row["unknown_field"] = "unsafe_override"
    row["schema_rejection_observed"] = schema_error
    row["fresh_program_spec_receipts_and_trust"] = freshness
    if found and not schema_error:
        errors.append(f"unknown-field fixture lacked explicit schema rejection: {fixture_id}")
    if not freshness:
        errors.append(f"unknown-field fixture failed through stale identity/receipt: {fixture_id}")
    return row, errors


def _resign_executable_program(program: dict[str, Any]) -> dict[str, Any]:
    """Reissue every circular authorization receipt after a semantic edit."""
    result = copy.deepcopy(program)
    metadata = copy.deepcopy(result["campaign_metadata"])
    metadata.pop("metadata_sha256", None)
    result["campaign_metadata"] = contract.stamp(metadata, "metadata_sha256")
    result["program_spec_sha256"] = contract.compute_program_spec_sha256(result)
    spec = result["program_spec_sha256"]

    parameter = {
        key: value for key, value in result["model"]["parameter_manifest_receipt"].items()
        if key != "receipt_sha256"
    }
    parameter.update({
        "program_spec_sha256": spec,
        "exact_parameter_count": result["model"]["exact_parameter_count"],
        "parent_revision_sha256": result["model"]["parent_revision_sha256"],
        "config_sha256": result["model"]["config_sha256"],
    })
    parameter = contract.stamp(parameter, "receipt_sha256")
    result["model"]["parameter_manifest_receipt"] = parameter

    teachers = result["training_contract"]["teacher_contract"]
    for teacher_name in ("parent_teacher", "elevation_teacher"):
        teacher = teachers.get(teacher_name)
        if not isinstance(teacher, dict):
            continue
        authorization = {
            key: value for key, value in teacher["authorization_receipt"].items()
            if key != "receipt_sha256"
        }
        authorization.update({
            "authorized": True, "program_spec_sha256": spec,
            "claim_scope": result["claim_scope"],
            "teacher_identity_sha256": teacher["identity_sha256"],
            "teacher_revision_sha256": teacher["revision_sha256"],
            "teacher_role": teacher["role"],
            "output_protocol_sha256": teacher["output_protocol_sha256"],
            "cache_manifest_sha256": teacher["cache_manifest_sha256"],
            "split_manifest_sha256": teacher["split_manifest_sha256"],
            "provenance_manifest_sha256": teacher["provenance_manifest_sha256"],
            "training_only": True,
        })
        teacher["authorization_receipt"] = contract.stamp(authorization, "receipt_sha256")
    parent_auth = teachers["parent_teacher"]["authorization_receipt"]
    elevation = teachers.get("elevation_teacher")
    elevation_auth = elevation["authorization_receipt"] if isinstance(elevation, dict) else None
    teacher_receipts_sha256 = contract._teacher_receipt_set_sha256(parent_auth, elevation_auth)

    execution = result["execution_contract"]
    root_payload = {
        key: value for key, value in execution["root_manifest"].items()
        if key != "binding_sha256"
    }
    root_payload.update({
        "manifest_sha256": result["package_root_manifest_sha256"],
        "package_root_schema": result["package_root_schema"],
        "program_spec_sha256": spec,
        "parameter_manifest_receipt_sha256": parameter["receipt_sha256"],
        "teacher_authorization_receipts_sha256": teacher_receipts_sha256,
        "experiment_id": result["experiment_binding"]["experiment_id"],
        "candidate_identity_sha256": result["experiment_binding"]["candidate_identity_sha256"],
        "policy_version": contract.POLICY_VERSION,
        "parent_revision_sha256": result["model"]["parent_revision_sha256"],
    })
    root_receipt = contract.stamp(root_payload, "binding_sha256")

    greenlight_payload = {
        key: value for key, value in execution["greenlight_receipt"].items()
        if key != "receipt_sha256"
    }
    greenlight_payload.update({
        "root_manifest_sha256": root_receipt["manifest_sha256"],
        "root_binding_sha256": root_receipt["binding_sha256"],
        "program_spec_sha256": spec,
        "parameter_manifest_receipt_sha256": parameter["receipt_sha256"],
        "teacher_authorization_receipts_sha256": teacher_receipts_sha256,
        "candidate_identity_sha256": result["experiment_binding"]["candidate_identity_sha256"],
        "claim_scope": result["claim_scope"],
    })
    greenlight = contract.stamp(greenlight_payload, "receipt_sha256")

    allowlist_payload = {
        key: value for key, value in execution["adapter_allowlist"].items()
        if key != "allowlist_sha256"
    }
    allowlist_payload.update({
        "program_spec_sha256": spec,
        "operators_adapters_sha256": contract._operators_adapters_sha256(result),
    })
    allowlist = contract.stamp(allowlist_payload, "allowlist_sha256")

    admission_payload = {
        key: value for key, value in execution["resource_admission"].items()
        if key != "receipt_sha256"
    }
    admission_payload.update({
        "root_manifest_sha256": root_receipt["manifest_sha256"],
        "program_spec_sha256": spec,
        "parameter_manifest_receipt_sha256": parameter["receipt_sha256"],
        "exact_parameter_count": result["model"]["exact_parameter_count"],
        "target_ceilings_sha256": contract._target_ceilings_sha256(result),
        "compute_budget_sha256": contract.hash_value(
            result["evaluation_contract"]["test_time_compute_budget"]
        ),
        "operators_adapters_sha256": contract._operators_adapters_sha256(result),
    })
    admission = contract.stamp(admission_payload, "receipt_sha256")
    execution.update({
        "root_manifest": root_receipt, "greenlight_receipt": greenlight,
        "adapter_allowlist": allowlist, "resource_admission": admission,
    })
    return contract.stamp(result, "program_sha256")


def _unknown_field_smuggling_fixtures() -> tuple[dict[str, Any], list[str]]:
    """Audit-E: every unknown field is signed first, then rejected by exact schema."""
    errors: list[str] = []
    program_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []

    program_injections: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
        ("program.top", lambda p: p.__setitem__("unsafe_override", True)),
        ("program.model", lambda p: p["model"].__setitem__("unsafe_override", True)),
        ("program.target", lambda p: p["target"].__setitem__("unsafe_override", True)),
        ("program.diagnostic", lambda p: p["diagnostic_contract"].__setitem__("unsafe_override", True)),
        ("program.operator", lambda p: p["operators"][0].__setitem__("unsafe_override", True)),
        ("program.executor", lambda p: p["operators"][0]["executor"].__setitem__("unsafe_override", True)),
        ("program.data", lambda p: p["data_contract"].__setitem__("unsafe_override", True)),
        ("program.data_split", lambda p: p["data_contract"]["splits"]["calibration"].__setitem__("unsafe_override", True)),
        ("program.training", lambda p: p["training_contract"].__setitem__("unsafe_override", True)),
        ("program.teacher_contract", lambda p: p["training_contract"]["teacher_contract"].__setitem__("unsafe_override", True)),
        ("program.parent_teacher", lambda p: p["training_contract"]["teacher_contract"]["parent_teacher"].__setitem__("unsafe_override", True)),
        ("program.evaluation", lambda p: p["evaluation_contract"].__setitem__("unsafe_override", True)),
        ("program.compute", lambda p: p["evaluation_contract"]["test_time_compute_budget"].__setitem__("unsafe_override", True)),
        ("program.execution", lambda p: p["execution_contract"].__setitem__("unsafe_override", True)),
        ("program.resume", lambda p: p["exact_resume_contract"].__setitem__("unsafe_override", True)),
        ("program.output", lambda p: p["output_contract"].__setitem__("unsafe_override", True)),
        ("program.campaign_metadata", lambda p: p["campaign_metadata"].__setitem__("unsafe_override", True)),
    )
    for fixture_id, inject in program_injections:
        _planned, executable = _fixture_program()
        inject(executable)
        executable = _resign_executable_program(executable)
        trust = contract._sample_trust_context(executable)
        found = contract.validate_program(
            executable, allow_planned=False, trust_context=trust,
            expected_package_root_manifest_sha256=executable["package_root_manifest_sha256"],
        )
        row, failed = _schema_smuggling_row(fixture_id, found)
        program_rows.append(row); errors.extend(failed)

    _planned, executable = _fixture_program()
    program_trust = contract._sample_trust_context(executable)
    artifact_injections: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
        ("artifact.top", lambda a: a.__setitem__("unsafe_override", True)),
        ("artifact.file", lambda a: a["files"][0].__setitem__("unsafe_override", True)),
        ("artifact.byte_ledger", lambda a: a["byte_ledger"].__setitem__("unsafe_override", 0)),
        ("artifact.component_manifest", lambda a: a["component_manifests"]["base"].__setitem__("unsafe_override", True)),
        ("artifact.physical_accounting", lambda a: a["physical_accounting"].__setitem__("unsafe_override", True)),
        ("artifact.runtime_accounting", lambda a: a["runtime_accounting"].__setitem__("unsafe_override", True)),
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "artifact.bin").write_bytes(b"x")
        base_artifact = contract._sample_artifact(executable)
        base_artifact["files"][0]["sha256"] = hashlib.sha256(b"x").hexdigest()
        base_artifact = _refresh_artifact(base_artifact)
        baseline_errors = contract.validate_artifact(
            base_artifact, executable, verify_files=True, base_dir=root,
            trust_context=program_trust,
        )
        errors.extend(f"Audit-E strict artifact baseline: {error}" for error in baseline_errors)
        for fixture_id, inject in artifact_injections:
            artifact = copy.deepcopy(base_artifact)
            inject(artifact)
            artifact = contract.stamp(artifact, "artifact_sha256")
            found = contract.validate_artifact(
                artifact, executable, verify_files=True, base_dir=root,
                trust_context=program_trust,
            )
            row, failed = _schema_smuggling_row(fixture_id, found)
            artifact_rows.append(row); errors.extend(failed)
        _moe_planned, moe_executable = _fixture_program(active_b=1.0)
        moe_trust = contract._sample_trust_context(moe_executable)
        moe_artifact = contract._sample_artifact(moe_executable)
        moe_artifact["files"][0]["sha256"] = hashlib.sha256(b"x").hexdigest()
        moe_artifact = _refresh_artifact(moe_artifact)
        moe_baseline_errors = contract.validate_artifact(
            moe_artifact, moe_executable, verify_files=True, base_dir=root,
            trust_context=moe_trust,
        )
        errors.extend(f"Audit-E strict MoE artifact baseline: {error}" for error in moe_baseline_errors)
        moe_artifact["moe_accounting"]["unsafe_override"] = True
        moe_artifact = contract.stamp(moe_artifact, "artifact_sha256")
        found = contract.validate_artifact(
            moe_artifact, moe_executable, verify_files=True, base_dir=root,
            trust_context=moe_trust,
        )
        row, failed = _schema_smuggling_row("artifact.moe_accounting", found)
        artifact_rows.append(row); errors.extend(failed)

    observation_injections = (
        "observation.top", "observation.metric", "observation.statistics",
        "observation.comparison", "observation.claim_snapshot",
        "observation.compute_receipt", "evidence.bundle",
        "evidence.artifact_receipt", "evidence.corrected_receipt",
        "evidence.verifier_receipt", "evidence.sealed_attestation",
        "evidence.independent_attestation",
    )
    for fixture_id in observation_injections:
        observation = _resign_smuggled_observation(executable, fixture_id)
        trust = contract._sample_trust_context(executable, observation)
        found = contract.validate_observation(
            observation, executable, trust_context=trust,
            expected_package_root_manifest_sha256=executable["package_root_manifest_sha256"],
        )
        decision = contract.dominance_decision(
            observation, executable, trust_context=trust,
            expected_package_root_manifest_sha256=executable["package_root_manifest_sha256"],
        )
        row, failed = _schema_smuggling_row(
            fixture_id, found, dominance_passed=decision["passed"]
        )
        observation_rows.append(row); errors.extend(failed)

    all_rows = [*program_rows, *artifact_rows, *observation_rows]
    return {
        "passed": not errors and all(row["rejected"] for row in all_rows),
        "unknown_field": "unsafe_override", "signed_before_receipt_generation": True,
        "fresh_external_trust_regenerated": True,
        "program_fixtures": program_rows, "artifact_fixtures": artifact_rows,
        "observation_and_nested_receipt_fixtures": observation_rows,
        "fixture_count": len(all_rows), "trusted_experimental_evidence": False,
    }, errors


def build_audit() -> dict[str, Any]:
    inputs, errors = _load_inputs()
    root = inputs.get("root", {})
    campaign = inputs.get("campaign", {})
    ladder = inputs.get("training_ladder", {})
    battery = inputs.get("quality_battery", {})
    validator_errors: dict[str, list[str]] = {
        "root": doctor_v5_root.validate_manifest(root, verify_files=True) if root else ["unavailable"],
        "campaign": doctor_v5.validate_campaign(campaign) if campaign else ["unavailable"],
        "training_ladder": training_ladder_v5.validate_ladder(ladder) if ladder else ["unavailable"],
        "quality_battery": quality_battery_v5.validate_manifest(battery) if battery else ["unavailable"],
    }
    for label, found in validator_errors.items():
        errors.extend(f"{label}: {item}" for item in found)

    root_receipt, found = _root_cross_identity(root, inputs) if root else ({"passed": False}, ["root unavailable"])
    errors.extend(found)
    grid, found = _grid_audit(ladder) if ladder else ({"passed": False}, ["ladder unavailable"])
    errors.extend(found)
    quality, found = _quality_contracts(campaign, ladder, battery) if all((campaign, ladder, battery)) else (
        {"passed": False}, ["quality inputs unavailable"]
    )
    errors.extend(found)
    fail_closed, found = _fail_closed_audit(root, campaign, ladder) if all((root, campaign, ladder)) else (
        {"passed": False}, ["fail-closed inputs unavailable"]
    )
    errors.extend(found)
    materializations, found = _materialization_audit(campaign) \
        if campaign and not validator_errors["campaign"] else (
            {"passed": False, "representative_count": 0}, []
        )
    errors.extend(found)
    adversarial, found = _adversarial_fixtures(
        campaign if not validator_errors["campaign"] else None,
        battery if not validator_errors["quality_battery"] else None,
    )
    errors.extend(found)
    artifacts, found = _artifact_fixtures()
    errors.extend(found)
    audit_e_smuggling, found = _unknown_field_smuggling_fixtures()
    errors.extend(found)
    ast_receipt, found = _ast_audit(SOURCE_PATHS)
    errors.extend(found)

    documents = []
    for path in DOCUMENT_PATHS:
        exists = path.is_file() and path.stat().st_size > 0
        if not exists:
            errors.append(f"required design document missing/empty: {_relative(path)}")
        documents.append({
            "path": _relative(path), "exists": exists,
            "file_sha256": _file_sha256(path) if path.is_file() else None,
        })
    errors = sorted(set(errors))
    static_pass = not errors
    report = {
        "schema": SCHEMA, "version": VERSION,
        "audit_scope": "static_design_integrity_and_adversarial_rejection_only",
        "root_manifest_sha256": root.get("root_manifest_sha256"),
        "campaign_sha256": campaign.get("campaign_sha256"),
        "ladder_sha256": ladder.get("ladder_sha256"),
        "quality_battery_sha256": battery.get("manifest_sha256"),
        "static_integrity_pass": static_pass,
        "design_package_complete": static_pass,
        "execution_authorized": False, "evidence_complete": False, "dominance_proven": False,
        "trusted_experimental_evidence_present": False,
        "input_validators": {key: {"passed": not value, "error_count": len(value)}
                             for key, value in validator_errors.items()},
        "root_cross_identity": root_receipt, "exact_training_grid": grid,
        "quality_and_competitor_contracts": quality, "fail_closed_execution": fail_closed,
        "representative_planned_materializations": materializations,
        "adversarial_rejection_fixtures": adversarial,
        "all_adversarial_fixtures_rejected": bool(adversarial) and all(
            row["rejected"] for row in adversarial
        ),
        "strict_artifact_verification": artifacts,
        "audit_e_unknown_field_smuggling": audit_e_smuggling,
        "documents": documents, "source_ast_audit": ast_receipt,
        "errors": errors,
        "conclusion": (
            "Static Doctor-v5 design package is internally complete; no execution, evidence, or dominance is authorized."
            if static_pass else
            "Static Doctor-v5 integrity audit failed; execution, evidence, and dominance remain blocked."
        ),
    }
    return _stamp(report, "audit_sha256")


def validate_audit(report: Any) -> list[str]:
    if not isinstance(report, dict):
        return ["audit report must be an object"]
    errors: list[str] = []
    if report.get("schema") != SCHEMA or report.get("version") != VERSION:
        errors.append("audit schema/version mismatch")
    static = report.get("errors") == []
    if report.get("static_integrity_pass") is not static \
            or report.get("design_package_complete") is not static:
        errors.append("static/design completion flags must exactly reflect the error list")
    for field in ("execution_authorized", "evidence_complete", "dominance_proven"):
        if report.get(field) is not False:
            errors.append(f"static audit can never set {field}=true")
    if report.get("trusted_experimental_evidence_present") is not False:
        errors.append("static audit cannot contain trusted experimental evidence")
    expected = report.get("audit_sha256")
    payload = copy.deepcopy(report); payload.pop("audit_sha256", None)
    if not contract.is_sha256(expected) or expected != contract.hash_value(payload):
        errors.append("audit_sha256 missing or mismatched")
    return errors


def selftest() -> int:
    report = build_audit()
    assert validate_audit(report) == []
    assert report["execution_authorized"] is False
    assert report["evidence_complete"] is False
    assert report["dominance_proven"] is False
    assert report["all_adversarial_fixtures_rejected"] is True
    if not ROOT_PATH.is_file() or not report["input_validators"]["root"]["passed"]:
        print(
            "doctor_v5_audit.py selftest WAITING: doctor_v5_root.json must validate "
            "and bind the final audit source"
        )
        return 0
    assert report["static_integrity_pass"] is True, report["errors"][:20]
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "audit.json"
        contract.atomic_json(path, report)
        assert validate_audit(_read_json(path)) == []
    print("doctor_v5_audit.py selftest OK", report["audit_sha256"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    commands.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    report = build_audit()
    contract.atomic_json(args.output, report)
    print(json.dumps({
        "output": str(args.output), "audit_sha256": report["audit_sha256"],
        "root_manifest_sha256": report["root_manifest_sha256"],
        "static_integrity_pass": report["static_integrity_pass"],
        "design_package_complete": report["design_package_complete"],
        "execution_authorized": False, "evidence_complete": False,
        "dominance_proven": False, "error_count": len(report["errors"]),
    }, indent=2, sort_keys=True))
    return 0 if report["static_integrity_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
