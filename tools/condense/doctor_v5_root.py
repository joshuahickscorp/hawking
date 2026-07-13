#!/usr/bin/env python3.12
"""Compile the single immutable identity for the Doctor-v5 design package.

The root manifest binds the canonical campaign, training ladder, quality
battery, implementation sources, and specifications.  It is deliberately a
static, stdlib-only compiler: it cannot authorize execution or manufacture
experimental evidence.  Derived program and audit receipts must carry this
manifest's hash; an executable program must additionally verify every bound
file and satisfy the separate Doctor-v5 greenlight contract.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import doctor_v5
import doctor_v5_contract as contract
import quality_battery_v5
import training_ladder_v5


SCHEMA = "hawking.doctor_v5_root.v5"
VERSION = "doctor-v5-root.1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports" / "condense" / "doctor_v5_root.json"

REPORT_SPECS = (
    (
        "campaign",
        ROOT / "reports" / "condense" / "doctor_v5_campaign.json",
        doctor_v5.CAMPAIGN_SCHEMA,
        "campaign_sha256",
    ),
    (
        "training_ladder",
        ROOT / "reports" / "condense" / "training_ladder_v5.json",
        training_ladder_v5.SCHEMA,
        "ladder_sha256",
    ),
    (
        "quality_battery",
        ROOT / "reports" / "condense" / "quality_battery_v5.json",
        quality_battery_v5.SCHEMA,
        "manifest_sha256",
    ),
)

SOURCE_PATHS = (
    ROOT / "tools" / "condense" / "doctor_v5_contract.py",
    ROOT / "tools" / "condense" / "quality_battery_v5.py",
    ROOT / "tools" / "condense" / "training_ladder_v5.py",
    ROOT / "tools" / "condense" / "doctor_v5.py",
    ROOT / "tools" / "condense" / "doctor_v5_root.py",
    ROOT / "tools" / "condense" / "doctor_v5_audit.py",
)

DOCUMENT_PATHS = (
    ROOT / "docs" / "plans" / "DOCTOR_V5.md",
    ROOT / "docs" / "plans" / "DOCTOR_V5_RESEARCH_PASSES.md",
    ROOT / "docs" / "plans" / "TRAINING_LADDER_V5.md",
)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_payload(document: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(document)
    payload.pop("root_manifest_sha256", None)
    return payload


def _stamp(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result["root_manifest_sha256"] = contract.hash_value(_identity_payload(result))
    return result


def _validate_input_reports(reports: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    validators = {
        "campaign": doctor_v5.validate_campaign,
        "training_ladder": training_ladder_v5.validate_ladder,
        "quality_battery": quality_battery_v5.validate_manifest,
    }
    for label, validator in validators.items():
        value = reports.get(label)
        if value is None:
            errors.append(f"{label} report unavailable")
            continue
        errors.extend(f"{label}: {error}" for error in validator(value))
    return errors


def build_manifest() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for label, path, _schema, _identity in REPORT_SPECS:
        if not path.is_file():
            raise ValueError(f"missing root input {_relative(path)}")
        reports[label] = _read_json(path)
    report_errors = _validate_input_reports(reports)
    if report_errors:
        raise ValueError("invalid root inputs:\n- " + "\n- ".join(report_errors))

    missing = [path for path in (*SOURCE_PATHS, *DOCUMENT_PATHS) if not path.is_file()]
    if missing:
        raise ValueError("missing bound files: " + ", ".join(_relative(path) for path in missing))

    report_bindings: list[dict[str, Any]] = []
    for label, path, schema, identity_field in REPORT_SPECS:
        report = reports[label]
        if report.get("schema") != schema:
            raise ValueError(f"{label} schema differs from {schema}")
        identity = report.get(identity_field)
        if not contract.is_sha256(identity):
            raise ValueError(f"{label}.{identity_field} is not a SHA-256 identity")
        report_bindings.append(
            {
                "id": label,
                "path": _relative(path),
                "schema": schema,
                "identity_field": identity_field,
                "document_identity": identity,
                "file_sha256": _file_sha256(path),
            }
        )

    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "root_id": "hawking-doctor-v5-capability-first-quality-proof-package",
        "execution_policy": {
            "static_identity_only": True,
            "execution_authorized": False,
            "greenlight_recorded": False,
            "evidence_complete": False,
            "dominance_proven": False,
            "loads_models": False,
            "launches_processes": False,
        },
        "consumer_contract": {
            "planned_program_must_name_root_schema": True,
            "executable_program_must_bind_root_manifest_sha256": True,
            "executable_program_must_verify_all_bound_files": True,
            "audit_receipt_must_bind_root_manifest_sha256": True,
            "derived_example_must_bind_root_manifest_sha256": True,
            "root_identity_is_necessary_not_sufficient_for_execution": True,
        },
        "report_bindings": report_bindings,
        "source_bindings": [
            {"path": _relative(path), "file_sha256": _file_sha256(path)}
            for path in SOURCE_PATHS
        ],
        "document_bindings": [
            {"path": _relative(path), "file_sha256": _file_sha256(path)}
            for path in DOCUMENT_PATHS
        ],
        "contract_schemas": {
            "program": contract.PROGRAM_SCHEMA,
            "artifact": contract.ARTIFACT_SCHEMA,
            "observation": contract.OBSERVATION_SCHEMA,
            "dominance": contract.DOMINANCE_SCHEMA,
            "campaign": doctor_v5.CAMPAIGN_SCHEMA,
            "training_ladder": training_ladder_v5.SCHEMA,
            "quality_battery": quality_battery_v5.SCHEMA,
        },
        "derived_receipts": {
            "audit": "reports/condense/doctor_v5_audit.json",
            "planned_example": "reports/condense/doctor_v5_first_program.json",
            "must_reference_this_root_hash": True,
            "not_inputs_to_root_hash_to_avoid_circular_identity": True,
        },
    }
    return _stamp(manifest)


def validate_manifest(document: Any, *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["root manifest must be an object"]
    if document.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if document.get("version") != VERSION:
        errors.append(f"version must be {VERSION}")
    expected_hash = document.get("root_manifest_sha256")
    if not contract.is_sha256(expected_hash) or expected_hash != contract.hash_value(
        _identity_payload(document)
    ):
        errors.append("root_manifest_sha256 missing or mismatched")
    policy = document.get("execution_policy", {})
    if policy.get("execution_authorized") is not False \
            or policy.get("greenlight_recorded") is not False \
            or policy.get("evidence_complete") is not False \
            or policy.get("dominance_proven") is not False:
        errors.append("root manifest cannot authorize execution, evidence, or dominance")
    if verify_files:
        try:
            canonical = build_manifest()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"could not rebuild canonical root manifest: {exc}")
        else:
            if document != canonical:
                errors.append("root manifest differs from current canonical bound package")
    return errors


def bind_planned_program(
    program_path: Path,
    *,
    root_manifest_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    """Resolve a derived planned example to the already-compiled package root.

    The program is deliberately not an input to the root hash, avoiding a
    circular identity.  Binding changes only the derived program receipt and
    never changes its mode or execution authority.
    """
    root_manifest = _read_json(root_manifest_path)
    root_errors = validate_manifest(root_manifest)
    if root_errors:
        raise ValueError("invalid package root:\n- " + "\n- ".join(root_errors))
    program = _read_json(program_path)
    program_errors = contract.validate_program(program)
    if program_errors:
        raise ValueError("invalid planned program:\n- " + "\n- ".join(program_errors))
    if program.get("mode") != "planned":
        raise ValueError("root binding command accepts planned programs only")
    root_sha256 = root_manifest["root_manifest_sha256"]
    existing = program.get("package_root_manifest_sha256")
    if existing not in ("required", root_sha256):
        raise ValueError("planned program is already bound to a different package root")
    program["package_root_schema"] = SCHEMA
    program["package_root_manifest_sha256"] = root_sha256
    program["program_spec_sha256"] = contract.compute_program_spec_sha256(program)
    bound = contract.stamp(program, "program_sha256")
    bound_errors = contract.validate_program(
        bound,
        expected_package_root_manifest_sha256=root_sha256,
    )
    if bound_errors:
        raise AssertionError("root-bound planned program became invalid:\n- " + "\n- ".join(bound_errors))
    contract.atomic_json(program_path, bound)
    return bound


def selftest() -> int:
    manifest = build_manifest()
    assert validate_manifest(manifest) == []
    assert manifest == build_manifest()
    assert manifest["execution_policy"]["execution_authorized"] is False

    tampered = copy.deepcopy(manifest)
    tampered["execution_policy"]["execution_authorized"] = True
    tampered = _stamp(tampered)
    assert validate_manifest(tampered)

    tampered = copy.deepcopy(manifest)
    tampered["report_bindings"][0]["document_identity"] = "0" * 64
    tampered = _stamp(tampered)
    assert "root manifest differs from current canonical bound package" in validate_manifest(tampered)

    tampered = copy.deepcopy(manifest)
    tampered["document_bindings"].pop()
    tampered = _stamp(tampered)
    assert "root manifest differs from current canonical bound package" in validate_manifest(tampered)

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "root.json"
        contract.atomic_json(path, manifest)
        assert validate_manifest(_read_json(path)) == []
    print("doctor_v5_root.py selftest OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_REPORT)
    bind_parser = commands.add_parser("bind-program")
    bind_parser.add_argument("program", type=Path)
    bind_parser.add_argument("--root", type=Path, default=DEFAULT_REPORT)
    commands.add_parser("selftest")
    args = parser.parse_args()

    if args.command == "compile":
        manifest = build_manifest()
        contract.atomic_json(args.output, manifest)
        print(json.dumps({
            "ok": True,
            "output": str(args.output),
            "root_manifest_sha256": manifest["root_manifest_sha256"],
            "execution_authorized": False,
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        try:
            value = _read_json(args.path)
        except (OSError, json.JSONDecodeError) as exc:
            errors = [str(exc)]
        else:
            errors = validate_manifest(value)
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    if args.command == "bind-program":
        try:
            bound = bind_planned_program(args.program, root_manifest_path=args.root)
        except (OSError, ValueError, json.JSONDecodeError, AssertionError) as exc:
            print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2, sort_keys=True))
            return 1
        print(json.dumps({
            "ok": True,
            "program": str(args.program),
            "program_sha256": bound["program_sha256"],
            "package_root_manifest_sha256": bound["package_root_manifest_sha256"],
            "execution_authorized": False,
        }, indent=2, sort_keys=True))
        return 0
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
