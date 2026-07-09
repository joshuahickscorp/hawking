#!/usr/bin/env python3.12
"""frontier_evidence_run.py - signed same-box Studio evidence-run bundles.

This is the claim-critical bridge above the individual receipt files. It does not download, bake, serve,
or eval on the laptop. It defines the signed envelope a Studio run must produce when one operator
orchestration command ties together native `.tq` serve, same-box baselines, frozen evals, RAM-cliff
energy, Doctor recovery, artifact inventory, and source-release decision evidence.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "condense"))

from studio_manifest import FRONTIER_MODELS, FrontierModel, frontier_by_label  # noqa: E402
import frontier_coverage  # noqa: E402
import frontier_coverage_runner  # noqa: E402
import frontier_doctor_recovery  # noqa: E402
import frontier_experiments  # noqa: E402
import frontier_experiment_runner  # noqa: E402
import frontier_parity  # noqa: E402
import frontier_parity_runner  # noqa: E402
import frontier_provenance  # noqa: E402
import frontier_receipt_runner  # noqa: E402
import frontier_receipts  # noqa: E402

COND_DIR = pathlib.Path("reports/condense")
SCHEMA = "hawking.frontier_studio_evidence_run.v1"
SIGN_ALG = "sha256-json-v1"
ALLOWED_SOURCE_RELEASE = (
    "delete_source_after_verified_bake",
    "retain_source_due_license",
    "retain_source_for_rebake",
    "not_applicable_prequantized",
)
REQUIRED_EVIDENCE = (
    "source_provenance",
    "architecture_parity",
    "native_tq_serve",
    "same_box_baselines",
    "frozen_eval_coverage",
    "ramcliff_energy",
    "doctor_recovery",
    "experiment_matrix",
    "artifact_inventory",
)
REQUIRED_GATE_KEYS = REQUIRED_EVIDENCE + ("source_release_decision",)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _git_commit(root: pathlib.Path = ROOT) -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label)


def evidence_run_path(root: pathlib.Path, label: str) -> pathlib.Path:
    return root / COND_DIR / f"{_safe_label(label)}_studio_evidence_run.json"


def artifact_inventory_path(root: pathlib.Path, label: str) -> pathlib.Path:
    return root / COND_DIR / f"{_safe_label(label)}_artifact_inventory.json"


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.load(open(path))
    except Exception:
        return None


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_digest(data: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(data)
    unsigned.pop("signature", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _placeholder(value: Any) -> bool:
    if value is None:
        return True
    s = str(value)
    return not s.strip() or "<" in s or "TODO" in s or "..." in s


def _commands(record: dict[str, Any]) -> list[str]:
    out = []
    cmds = record.get("commands")
    if isinstance(cmds, list):
        out.extend(str(cmd) for cmd in cmds if cmd)
    if record.get("command"):
        out.append(str(record["command"]))
    if record.get("runner_command"):
        out.append(str(record["runner_command"]))
    return out


def _model_for_record(record: dict[str, Any], model: FrontierModel | None = None) -> FrontierModel | None:
    if model:
        return model
    label = record.get("model") or record.get("label")
    hf_id = record.get("hf_id")
    if label:
        found = frontier_by_label(str(label))
        if found:
            return found
    if hf_id:
        return frontier_by_label(str(hf_id))
    return None


def signature_status(record: dict[str, Any]) -> dict[str, Any]:
    sig = record.get("signature") if isinstance(record.get("signature"), dict) else {}
    expected = _canonical_digest(record)
    ok = sig.get("algorithm") == SIGN_ALG and sig.get("digest") == expected
    problems = []
    if sig.get("algorithm") != SIGN_ALG:
        problems.append(f"signature algorithm must be {SIGN_ALG}")
    if sig.get("digest") != expected:
        problems.append("signature digest mismatch")
    return {
        "ok": ok,
        "algorithm": sig.get("algorithm"),
        "digest": sig.get("digest"),
        "expected_digest": expected,
        "problems": problems,
    }


def _evidence_path(root: pathlib.Path, model: FrontierModel, key: str) -> pathlib.Path:
    if key == "source_provenance":
        return frontier_provenance.provenance_path(root, model.label)
    if key == "architecture_parity":
        return pathlib.Path(frontier_parity.parity_status(model, root)["record"])
    if key == "same_box_baselines":
        return frontier_coverage.baseline_path(root, model.label)
    if key == "frozen_eval_coverage":
        return frontier_coverage.eval_path(root, model.label)
    if key == "native_tq_serve":
        return frontier_receipts.serve_path(root, model.label)
    if key == "ramcliff_energy":
        return frontier_receipts.ramcliff_path(root, model.label)
    if key == "doctor_recovery":
        return frontier_doctor_recovery.recovery_path(root, model.label)
    if key == "experiment_matrix":
        return frontier_experiments.matrix_path(root, model.label)
    if key == "artifact_inventory":
        return artifact_inventory_path(root, model.label)
    raise KeyError(key)


def _artifact_inventory_status(root: pathlib.Path, model: FrontierModel, record: dict[str, Any] | None) -> dict[str, Any]:
    problems = []
    if not record:
        return {
            "schema": "hawking.frontier_artifact_inventory_status.v1",
            "ok": False,
            "label": model.label,
            "artifact_count": 0,
            "problems": ["artifact inventory missing or unreadable"],
        }
    if record.get("schema") != "hawking.frontier_artifact_inventory.v1":
        problems.append("artifact inventory schema mismatch")
    if record.get("label") != model.label:
        problems.append(f"artifact inventory label must be {model.label}")
    if record.get("hf_id") and record.get("hf_id") != model.hf_id:
        problems.append(f"artifact inventory hf_id must be {model.hf_id}")
    rows = record.get("artifacts") if isinstance(record.get("artifacts"), list) else []
    if not rows:
        problems.append("artifact inventory has no artifact rows")
    for row in rows:
        path_value = row.get("path")
        if _placeholder(path_value):
            problems.append("artifact row path missing or placeholder")
            continue
        artifact = pathlib.Path(str(path_value))
        if not artifact.is_absolute():
            artifact = root / artifact
        if not artifact.exists() or not artifact.is_file():
            problems.append(f"artifact {path_value} missing on disk")
            continue
        try:
            size = artifact.stat().st_size
        except OSError:
            problems.append(f"artifact {path_value} cannot be stat()ed")
            continue
        if row.get("bytes") != size:
            problems.append(f"artifact {path_value} size changed since inventory")
        sha = row.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
            problems.append(f"artifact {path_value} missing valid sha256")
            continue
        actual_sha = _sha256_file(artifact)
        if actual_sha != sha.lower():
            problems.append(f"artifact {path_value} sha256 changed since inventory")
    return {
        "schema": "hawking.frontier_artifact_inventory_status.v1",
        "ok": not problems,
        "label": model.label,
        "artifact_count": len(rows),
        "problems": problems,
    }


def _status_for_key(root: pathlib.Path, model: FrontierModel, key: str, record: dict[str, Any] | None) -> dict[str, Any]:
    if key == "source_provenance":
        return frontier_provenance.record_status(record, model=model, require_signature=True)
    if key == "architecture_parity":
        return frontier_parity_runner.record_status(record, model=model, require_signature=True)
    if key == "same_box_baselines":
        return frontier_coverage_runner.record_status(record, kind="baseline", require_signature=True)
    if key == "frozen_eval_coverage":
        return frontier_coverage_runner.record_status(record, kind="eval", require_signature=True)
    if key == "native_tq_serve":
        return frontier_receipt_runner.record_status(record, kind="serve", require_signature=True)
    if key == "ramcliff_energy":
        return frontier_receipt_runner.record_status(record, kind="ramcliff", require_signature=True)
    if key == "doctor_recovery":
        return frontier_doctor_recovery.record_status(record, model=model, require_signature=True)
    if key == "experiment_matrix":
        return frontier_experiment_runner.record_status(record, label=model.label, require_signature=True)
    if key == "artifact_inventory":
        return _artifact_inventory_status(root, model, record)
    raise KeyError(key)


def evidence_status(root: pathlib.Path, model: FrontierModel, key: str) -> dict[str, Any]:
    path = _evidence_path(root, model, key)
    record = _read_json(path)
    status = _status_for_key(root, model, key, record)
    digest = _sha256_file(path)
    return {
        "key": key,
        "path": str(path),
        "exists": digest is not None,
        "bytes": path.stat().st_size if digest and path.exists() else None,
        "sha256": digest,
        "ok": bool(status.get("ok")),
        "signature_ok": (status.get("signature") or {}).get("ok"),
        "status_schema": status.get("schema"),
        "problems": status.get("problems", []),
    }


def evidence_run_status(root: pathlib.Path, label: str) -> dict[str, Any]:
    model = frontier_by_label(label)
    if not model:
        return {
            "schema": "hawking.frontier_studio_evidence_run_status.v1",
            "ok": False,
            "label": label,
            "problems": [f"unknown frontier label: {label}"],
        }
    path = evidence_run_path(root, model.label)
    record = _read_json(path)
    status = record_status(record, root=root, model=model, require_signature=True)
    status["path"] = str(path)
    status["exists"] = record is not None
    return status


def evidence_run_rollup(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    rows = [evidence_run_status(root, label) for label in labels]
    blocked = [row.get("label") or label for row, label in zip(rows, labels) if not row["ok"]]
    return {
        "schema": "hawking.frontier_studio_evidence_run_rollup.v1",
        "model_count": len(labels),
        "passed_count": len(labels) - len(blocked),
        "blocked_count": len(blocked),
        "blocked_labels": blocked,
        "rows": rows,
        "ok": not blocked,
    }


def _source_release_problems(decision: dict[str, Any] | None) -> list[str]:
    problems = []
    if not isinstance(decision, dict):
        return ["source_release_decision missing"]
    status = decision.get("decision") or decision.get("status")
    if status not in ALLOWED_SOURCE_RELEASE:
        problems.append(f"source_release_decision must be one of: {', '.join(ALLOWED_SOURCE_RELEASE)}")
    if _placeholder(decision.get("command")):
        problems.append("source_release_decision.command missing or placeholder")
    if _placeholder(decision.get("reason")):
        problems.append("source_release_decision.reason missing or placeholder")
    if _placeholder(decision.get("decided_by")):
        problems.append("source_release_decision.decided_by missing or placeholder")
    return problems


def _strict_problems(record: dict[str, Any], root: pathlib.Path, model: FrontierModel | None) -> list[str]:
    problems = []
    if record.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA}")
    if not model:
        problems.append("model label/hf_id must match a frontier manifest label")
    else:
        if (record.get("model") or record.get("label")) != model.label:
            problems.append(f"model/label must be {model.label}")
        if record.get("hf_id") != model.hf_id:
            problems.append(f"hf_id must be {model.hf_id}")
    if record.get("receipt_state") != "final":
        problems.append("receipt_state must be final")
    if str(record.get("source") or record.get("mode") or "").lower() != "measured":
        problems.append("source/mode must be measured")
    if record.get("status") != "pass":
        problems.append("status must be pass")
    if not record.get("machine_class"):
        problems.append("machine_class missing")
    if not (record.get("git_commit") or record.get("hawking_commit")):
        problems.append("git_commit/hawking_commit missing")
    if _placeholder(record.get("run_id")):
        problems.append("run_id missing or placeholder")
    commands = _commands(record)
    if not commands:
        problems.append("runner command missing")
    elif any(_placeholder(cmd) for cmd in commands):
        problems.append("runner command contains placeholder text")

    gate = record.get("gate") if isinstance(record.get("gate"), dict) else {}
    for key in REQUIRED_GATE_KEYS:
        if gate.get(key) is not True:
            problems.append(f"gate.{key} must be true")

    evidence = record.get("evidence") if isinstance(record.get("evidence"), list) else []
    by_key = {row.get("key"): row for row in evidence}
    for key in REQUIRED_EVIDENCE:
        row = by_key.get(key)
        if not row:
            problems.append(f"evidence.{key} missing")
            continue
        path = pathlib.Path(str(row.get("path") or ""))
        if not path.is_absolute():
            path = root / path
        digest = _sha256_file(path)
        if not digest:
            problems.append(f"evidence.{key} file missing: {path}")
            continue
        if row.get("sha256") != digest:
            problems.append(f"evidence.{key} sha256 changed")
        if row.get("ok") is not True:
            problems.append(f"evidence.{key} status is not ok")
        if model:
            current = evidence_status(root, model, key)
            if not current["ok"]:
                problems.extend(f"evidence.{key}: {problem}" for problem in current["problems"])

    problems.extend(_source_release_problems(record.get("source_release_decision")))
    return problems


def record_status(record: dict[str, Any] | None, *, root: pathlib.Path = ROOT,
                  model: FrontierModel | None = None,
                  require_signature: bool = True) -> dict[str, Any]:
    if not record:
        return {
            "schema": "hawking.frontier_studio_evidence_run_status.v1",
            "ok": False,
            "label": model.label if model else None,
            "problems": ["record missing or unreadable"],
        }
    model = _model_for_record(record, model)
    problems = _strict_problems(record, root, model)
    sig = signature_status(record)
    if require_signature and not sig["ok"]:
        problems.extend(sig["problems"])
    evidence = record.get("evidence") if isinstance(record.get("evidence"), list) else []
    return {
        "schema": "hawking.frontier_studio_evidence_run_status.v1",
        "ok": not problems,
        "label": model.label if model else record.get("model") or record.get("label"),
        "hf_id": model.hf_id if model else record.get("hf_id"),
        "receipt_state": record.get("receipt_state"),
        "evidence_ok": sum(1 for row in evidence if row.get("ok")),
        "evidence_required": len(REQUIRED_EVIDENCE),
        "signature": sig,
        "problems": problems,
    }


def sign_record(record: dict[str, Any], *, root: pathlib.Path = ROOT,
                model: FrontierModel | None = None,
                allow_blocked_draft: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    signed = copy.deepcopy(record)
    signed.pop("signature", None)
    signed.setdefault("generated_at", _now())
    signed.setdefault("git_commit", _git_commit(root))
    signed["signed_at"] = _now()
    status = record_status(signed, root=root, model=model, require_signature=False)
    if not status["ok"] and not allow_blocked_draft:
        return signed, status
    signed["signature"] = {"algorithm": SIGN_ALG, "digest": _canonical_digest(signed)}
    return signed, record_status(signed, root=root, model=model, require_signature=True)


def draft_record(root: pathlib.Path, model: FrontierModel,
                 *, machine_class: str = "Studio-M1Ultra-128") -> dict[str, Any]:
    evidence = [evidence_status(root, model, key) for key in REQUIRED_EVIDENCE]
    return {
        "schema": SCHEMA,
        "model": model.label,
        "hf_id": model.hf_id,
        "generated_at": _now(),
        "git_commit": _git_commit(root),
        "machine_class": machine_class,
        "receipt_state": "draft",
        "source": "TODO measured",
        "status": "TODO pass",
        "run_id": f"{model.label}-<studio-run-id>",
        "runner_command": (
            "hawking studio evidence-run-receipt build "
            f"{model.label} --source-release-decision <decision> --source-release-command <exact command>"
        ),
        "commands": ["<exact one-command Studio evidence runner invocation>"],
        "evidence": evidence,
        "source_release_decision": {
            "decision": "<delete_source_after_verified_bake|retain_source_due_license|retain_source_for_rebake|not_applicable_prequantized>",
            "command": "<exact release-source or retain-source command>",
            "reason": "<why this source lifecycle is correct>",
            "decided_by": "<operator>",
        },
        "gate": {key: False for key in REQUIRED_GATE_KEYS},
    }


def build_record(root: pathlib.Path, model: FrontierModel, *, run_id: str, runner_command: str,
                 source_release_decision: str, source_release_command: str,
                 source_release_reason: str, decided_by: str,
                 machine_class: str = "Studio-M1Ultra-128") -> dict[str, Any]:
    evidence = [evidence_status(root, model, key) for key in REQUIRED_EVIDENCE]
    gate = {row["key"]: bool(row["ok"]) for row in evidence}
    source_release = {
        "decision": source_release_decision,
        "command": source_release_command,
        "reason": source_release_reason,
        "decided_by": decided_by,
    }
    gate["source_release_decision"] = not _source_release_problems(source_release)
    return {
        "schema": SCHEMA,
        "model": model.label,
        "hf_id": model.hf_id,
        "generated_at": _now(),
        "git_commit": _git_commit(root),
        "machine_class": machine_class,
        "receipt_state": "final",
        "source": "measured",
        "status": "pass" if all(gate.values()) else "blocked",
        "run_id": run_id,
        "runner_command": runner_command,
        "commands": [runner_command],
        "evidence": evidence,
        "source_release_decision": source_release,
        "gate": gate,
    }


def _selected_models(labels: list[str]) -> list[FrontierModel]:
    if not labels:
        return list(FRONTIER_MODELS)
    out = []
    for label in labels:
        model = frontier_by_label(label)
        if not model:
            raise SystemExit(f"unknown frontier label: {label}")
        out.append(model)
    return out


def evidence_run_plan(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    models = _selected_models(labels)
    return {
        "schema": "hawking.frontier_studio_evidence_run_plan.v1",
        "generated_at": _now(),
        "model_count": len(models),
        "requirements": {
            "schema": SCHEMA,
            "signature": SIGN_ALG,
            "evidence": list(REQUIRED_EVIDENCE),
            "source_release_decisions": list(ALLOWED_SOURCE_RELEASE),
            "one_command_required": True,
        },
        "labels": [
            {
                "label": model.label,
                "hf_id": model.hf_id,
                "path": str(evidence_run_path(root, model.label)),
                "command_template": (
                    f"hawking studio evidence-run-receipt draft {model.label} --sign-draft --force"
                ),
                "evidence_paths": {
                    key: str(_evidence_path(root, model, key)) for key in REQUIRED_EVIDENCE
                },
            }
            for model in models
        ],
        "rollup": evidence_run_rollup(root, [m.label for m in models]),
    }


def dispatch(args, root: pathlib.Path = ROOT) -> int:
    rows = []
    ok = True
    for model in _selected_models(args.label):
        path = evidence_run_path(root, model.label)
        if getattr(args, "out_dir", ""):
            path = pathlib.Path(args.out_dir) / path.name
        if args.evidence_mode == "draft":
            if path.exists() and not args.force:
                rows.append({"label": model.label, "path": str(path), "ok": False,
                             "problems": ["path exists; use --force to overwrite"]})
                ok = False
                continue
            record = draft_record(root, model, machine_class=args.machine_class)
            if args.sign_draft:
                record, status = sign_record(record, root=root, model=model, allow_blocked_draft=True)
            else:
                status = record_status(record, root=root, model=model, require_signature=False)
            _write_json(path, record)
        elif args.evidence_mode == "build":
            record = build_record(
                root,
                model,
                run_id=args.run_id,
                runner_command=args.runner_command,
                source_release_decision=args.source_release_decision,
                source_release_command=args.source_release_command,
                source_release_reason=args.source_release_reason,
                decided_by=args.decided_by,
                machine_class=args.machine_class,
            )
            record, status = sign_record(record, root=root, model=model)
            if status["ok"]:
                _write_json(path, record)
        elif args.evidence_mode == "verify":
            record = _read_json(path)
            status = record_status(record, root=root, model=model, require_signature=True)
        else:
            raise SystemExit(f"unknown evidence-run mode: {args.evidence_mode}")
        rows.append({"label": model.label, "path": str(path), "ok": status["ok"],
                     "problems": status["problems"]})
        ok = ok and status["ok"]
    result = {
        "schema": "hawking.frontier_studio_evidence_run_command.v1",
        "mode": args.evidence_mode,
        "ok": ok,
        "rows": rows,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"# frontier Studio evidence-run receipts {args.evidence_mode}: {'OK' if ok else 'BLOCKED'}")
        for row in rows:
            print(f"{row['label'][:18]:18s} {'OK' if row['ok'] else 'BLOCK':6s} {row['path']}")
            for problem in row["problems"][:8]:
                print(f"  - {problem}")
    return 0 if ok else 1


def _write_complete_selftest_evidence(root: pathlib.Path, model: FrontierModel) -> None:
    cond = root / COND_DIR
    cond.mkdir(parents=True, exist_ok=True)
    source, _ = frontier_provenance.sign_record(frontier_provenance.complete_record(model), model=model)
    _write_json(frontier_provenance.provenance_path(root, model.label), source)
    parity, _ = frontier_parity_runner.sign_record(frontier_parity_runner._complete_record(model), model=model)
    _write_json(pathlib.Path(frontier_parity.parity_status(model, root)["record"]), parity)
    baseline = {
        "schema": frontier_coverage_runner.BASELINE_SCHEMA,
        "model": model.label,
        "receipt_state": "final",
        "source": "measured",
        "machine_class": "Studio-M1Ultra-128",
        "baselines": [
            {
                "name": req["name"],
                "status": "pass",
                "command": f"selftest baseline {i}",
                "artifact": f"selftest://baseline/{i}",
                "metrics": {"tok_s": 1.0 + i},
            }
            for i, req in enumerate(frontier_coverage.BASELINE_REQUIREMENTS)
        ],
    }
    baseline, _ = frontier_coverage_runner.sign_record(baseline, kind="baseline")
    _write_json(frontier_coverage.baseline_path(root, model.label), baseline)
    eval_record = {
        "schema": frontier_coverage_runner.EVAL_SCHEMA,
        "model": model.label,
        "receipt_state": "final",
        "source": "measured",
        "mode": "real",
        "machine_class": "Studio-M1Ultra-128",
        "domains": [
            {
                "domain": req["name"],
                "status": "pass",
                "command": f"selftest eval {i}",
                "receipt": f"selftest://eval/{i}",
                "metrics": {"score": 1.0},
            }
            for i, req in enumerate(frontier_coverage.EVAL_REQUIREMENTS)
        ],
    }
    eval_record, _ = frontier_coverage_runner.sign_record(eval_record, kind="eval")
    _write_json(frontier_coverage.eval_path(root, model.label), eval_record)
    serve = {
        "schema": frontier_receipt_runner.SERVE_SCHEMA,
        "model": model.label,
        "receipt_state": "final",
        "machine_class": "Studio-M1Ultra-128",
        "artifact_sha256": "a" * 64,
        "native_tq": True,
        "tq_strict": True,
        "all_linear": True,
        "gpu_bitslice": True,
        "served_forward_pass": True,
        "rehydrate_f16": False,
        "status": "pass",
        "tok_s": 12.0,
        "memory_peak_gb": 4.0,
        "memory_resident_gb": 3.5,
        "unified_memory_gb": 128.0,
        "resident_memory_ok": True,
        "parity_pass": True,
        "commands": ["selftest serve"],
        "load_receipt": "selftest://load",
        "served_forward_receipt": "selftest://served-forward",
        "parity_receipt": "selftest://serve-parity",
        "git_commit": "selftest",
    }
    serve, _ = frontier_receipt_runner.sign_record(serve, kind="serve")
    _write_json(frontier_receipts.serve_path(root, model.label), serve)
    ramcliff = {
        "schema": frontier_receipt_runner.RAMCLIFF_SCHEMA,
        "model": model.label,
        "receipt_state": "final",
        "machine_class": "Studio-M1Ultra-128",
        "artifact_sha256": "b" * 64,
        "gate": {
            "condensed_resident": True,
            "served_native_tq": True,
            "q4k_overflows_box": True,
            "cliff_x_over_gate": True,
            "resident_lower_energy": True,
        },
        "verdict": "CLIFF-WIN",
        "source": "measured",
        "served_native_tq": True,
        "tok_s_resident": 20.0,
        "tok_s_swapping": 1.0,
        "j_per_tok_resident": 1.0,
        "j_per_tok_swapping": 5.0,
        "cliff_x": 20.0,
        "commands": ["selftest ramcliff"],
        "powermetrics_receipt": "selftest://powermetrics",
        "baseline_receipt": "selftest://q4k-swap",
        "git_commit": "selftest",
    }
    ramcliff, _ = frontier_receipt_runner.sign_record(ramcliff, kind="ramcliff")
    _write_json(frontier_receipts.ramcliff_path(root, model.label), ramcliff)
    doctor, _ = frontier_doctor_recovery.sign_record(frontier_doctor_recovery.complete_record(model),
                                                     model=model)
    _write_json(frontier_doctor_recovery.recovery_path(root, model.label), doctor)
    experiment, _ = frontier_experiment_runner.sign_record(
        frontier_experiment_runner._complete_record(model.label),
        label=model.label,
    )
    _write_json(frontier_experiments.matrix_path(root, model.label), experiment)
    artifact = root / "scratch" / f"{_safe_label(model.label)}.tq"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"selftest-tq-artifact")
    _write_json(artifact_inventory_path(root, model.label), {
        "schema": "hawking.frontier_artifact_inventory.v1",
        "generated_at": _now(),
        "label": model.label,
        "hf_id": model.hf_id,
        "git_commit": "selftest",
        "artifacts": [
            {
                "path": str(artifact),
                "bytes": artifact.stat().st_size,
                "gb": round(artifact.stat().st_size / 1e9, 6),
                "sha256": _sha256_file(artifact),
            }
        ],
    })


def selftest() -> bool:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    model = FRONTIER_MODELS[0]
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        _write_complete_selftest_evidence(root, model)
        built = build_record(
            root,
            model,
            run_id="selftest-run-001",
            runner_command="hawking studio run-next --selftest-evidence-run",
            source_release_decision="delete_source_after_verified_bake",
            source_release_command=f"hawking studio release-source {model.label} --dry-run",
            source_release_reason="selftest source lifecycle decision",
            decided_by="selftest",
        )
        signed, status = sign_record(built, root=root, model=model)
        check("complete Studio evidence run signs and verifies", status["ok"])
        complete_signed = copy.deepcopy(signed)
        _write_json(evidence_run_path(root, model.label), signed)
        check("evidence-run rollup passes complete receipt",
              evidence_run_rollup(root, [model.label])["ok"])
        signed["evidence"][0]["sha256"] = "0" * 64
        check("evidence-run tamper is blocked",
              not record_status(signed, root=root, model=model)["ok"])
        artifact = root / "scratch" / f"{_safe_label(model.label)}.tq"
        artifact.write_bytes(b"changed-artifact")
        check("stale artifact inventory is blocked",
              not record_status(complete_signed, root=root, model=model)["ok"])
        draft = draft_record(root, model)
        draft_signed, draft_status = sign_record(draft, root=root, model=model,
                                                 allow_blocked_draft=True)
        check("signed evidence-run draft stays blocked",
              draft_signed.get("signature") and not draft_status["ok"])
        bad_release = build_record(
            root,
            model,
            run_id="selftest-run-002",
            runner_command="hawking studio run-next --selftest-evidence-run",
            source_release_decision="TODO",
            source_release_command="<command>",
            source_release_reason="<reason>",
            decided_by="<operator>",
        )
        _, bad_release_status = sign_record(bad_release, root=root, model=model)
        check("source-release placeholders are blocked", not bad_release_status["ok"])

    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def cmd_plan(args) -> int:
    data = evidence_run_plan(ROOT, args.label)
    if args.out:
        _write_json(pathlib.Path(args.out), data)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0 if data["rollup"]["ok"] else 1
    print("# frontier Studio evidence-run plan")
    roll = data["rollup"]
    print(f"# evidence-runs {roll['passed_count']}/{roll['model_count']}")
    for row in data["labels"]:
        print(f"{row['label']}: {row['path']}")
    return 0 if data["rollup"]["ok"] else 1


def cmd_selftest(args) -> int:
    return 0 if selftest() else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Draft/build/verify signed Studio evidence-run bundles.")
    sub = ap.add_subparsers(dest="evidence_mode")
    p = sub.add_parser("plan", help="print Studio evidence-run requirements")
    p.add_argument("label", nargs="*", help="frontier label(s); default all")
    p.add_argument("--out", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)
    for mode in ("draft", "build", "verify"):
        p = sub.add_parser(mode, help=f"{mode} signed Studio evidence-run receipts")
        p.add_argument("label", nargs="*", help="frontier label(s); default all")
        p.add_argument("--out-dir", default="")
        p.add_argument("--json", action="store_true")
        if mode == "draft":
            p.add_argument("--force", action="store_true")
            p.add_argument("--sign-draft", action="store_true")
            p.add_argument("--machine-class", default="Studio-M1Ultra-128")
        else:
            p.set_defaults(force=False, sign_draft=False)
        if mode == "build":
            p.add_argument("--run-id", required=True)
            p.add_argument("--runner-command", required=True)
            p.add_argument("--source-release-decision", required=True, choices=ALLOWED_SOURCE_RELEASE)
            p.add_argument("--source-release-command", required=True)
            p.add_argument("--source-release-reason", required=True)
            p.add_argument("--decided-by", required=True)
            p.add_argument("--machine-class", default="Studio-M1Ultra-128")
        elif mode != "draft":
            p.set_defaults(run_id="", runner_command="", source_release_decision="",
                           source_release_command="", source_release_reason="", decided_by="",
                           machine_class="Studio-M1Ultra-128")
        p.set_defaults(func=dispatch)
    p = sub.add_parser("selftest", help="synthetic signed Studio evidence-run receipt tests")
    p.set_defaults(func=cmd_selftest)
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    if not args.evidence_mode:
        args = ap.parse_args(["selftest"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
