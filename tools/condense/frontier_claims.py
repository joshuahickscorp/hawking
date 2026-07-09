#!/usr/bin/env python3.12
"""frontier_claims.py - signed public-claim bundles for frontier Studio results.

This is the final paper-trail layer. It does not run heavy work. It asks whether a frontier claim has
the receipts a public win needs, signs those receipts by hash, and verifies later that the evidence did
not drift.

Default policy is strict: a claim bundle requires source provenance, parity, native `.tq` serve,
same-box baseline coverage, frozen eval coverage, RAM-cliff evidence, Doctor recovery evidence,
expensive-mode experiment depth, a same-run Studio evidence bundle, artifact hashes, and exact commands.
Use `--no-require-ramcliff` only for a non-cliff claim.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "condense"))

from studio_manifest import FRONTIER_MODELS, FrontierModel, frontier_by_label  # noqa: E402
import frontier_coverage  # noqa: E402
import frontier_coverage_runner  # noqa: E402
import frontier_experiments  # noqa: E402
import frontier_experiment_runner  # noqa: E402
import frontier_doctor_recovery  # noqa: E402
import frontier_evidence_run  # noqa: E402
import frontier_parity  # noqa: E402
import frontier_parity_runner  # noqa: E402
import frontier_provenance  # noqa: E402
import frontier_receipt_runner  # noqa: E402
import frontier_receipts  # noqa: E402

COND_DIR = pathlib.Path("reports/condense")
SCHEMA = "hawking.frontier_claim_bundle.v1"
SIGN_ALG = "sha256-json-v1"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _git_commit(root: pathlib.Path = ROOT) -> str:
    try:
        p = __import__("subprocess").run(
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


def claim_bundle_path(root: pathlib.Path, label: str) -> pathlib.Path:
    return root / COND_DIR / f"{_safe_label(label)}_claim_bundle.json"


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.load(open(path))
    except Exception:
        return None


def _sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_digest(data: dict[str, Any]) -> str:
    unsigned = dict(data)
    unsigned.pop("signature", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _commands(row: dict[str, Any], record: dict[str, Any] | None = None) -> list[str]:
    out = []
    for source in (row, record or {}):
        cmds = source.get("commands")
        if isinstance(cmds, list):
            out.extend(str(cmd) for cmd in cmds if cmd)
        cmd = source.get("command")
        if cmd:
            out.append(str(cmd))
    return out


def _placeholder_command(cmd: str) -> bool:
    return "<" in cmd or "TODO" in cmd or "..." in cmd


def _exact_commands_ok(row: dict[str, Any], record: dict[str, Any] | None = None) -> tuple[bool, str]:
    cmds = _commands(row, record)
    if not cmds:
        return False, "exact command(s) missing"
    if any(_placeholder_command(cmd) for cmd in cmds):
        return False, "command contains placeholder text"
    return True, ""


def _coverage_command_problems(root: pathlib.Path, label: str, kind: str) -> list[str]:
    if kind == "baseline":
        path = frontier_coverage.baseline_path(root, label)
        missing = f"{path} missing; cannot verify signed baseline receipt"
    else:
        path = frontier_coverage.eval_path(root, label)
        missing = f"{path} missing; cannot verify signed eval receipt"
    record = _read_json(path)
    if not record:
        return [missing]
    status = frontier_coverage_runner.record_status(record, kind=kind, require_signature=True)
    return [f"{kind}: {problem}" for problem in status["problems"]]


def _native_receipt_problems(root: pathlib.Path, label: str, kind: str) -> list[str]:
    if kind == "serve":
        path = frontier_receipts.serve_path(root, label)
        missing = f"{path} missing; cannot verify signed native serve receipt"
    else:
        path = frontier_receipts.ramcliff_path(root, label)
        missing = f"{path} missing; cannot verify signed RAM-cliff receipt"
    record = _read_json(path)
    if not record:
        return [missing]
    status = frontier_receipt_runner.record_status(record, kind=kind, require_signature=True)
    return [f"{kind}: {problem}" for problem in status["problems"]]


def _parity_receipt_problems(root: pathlib.Path, model: FrontierModel) -> list[str]:
    path = pathlib.Path(frontier_parity.parity_status(model, root)["record"])
    record = _read_json(path)
    if not record:
        return [f"{path} missing; cannot verify signed parity receipt"]
    status = frontier_parity_runner.record_status(record, model=model, require_signature=True)
    return [f"parity: {problem}" for problem in status["problems"]]


def _parity_command_problems(root: pathlib.Path, model: FrontierModel) -> list[str]:
    return _parity_receipt_problems(root, model)


def _experiment_receipt_problems(root: pathlib.Path, label: str) -> list[str]:
    path = frontier_experiments.matrix_path(root, label)
    record = _read_json(path)
    if not record:
        return [f"{path} missing; cannot verify signed experiment matrix"]
    status = frontier_experiment_runner.record_status(record, label=label, require_signature=True)
    return [f"experiment: {problem}" for problem in status["problems"]]


def _experiment_trace_problems(root: pathlib.Path, label: str) -> list[str]:
    return _experiment_receipt_problems(root, label)


def _doctor_recovery_problems(root: pathlib.Path, model: FrontierModel) -> list[str]:
    path = frontier_doctor_recovery.recovery_path(root, model.label)
    record = _read_json(path)
    if not record:
        return [f"{path} missing; cannot verify signed Doctor recovery receipt"]
    status = frontier_doctor_recovery.record_status(record, model=model, require_signature=True)
    return [f"doctor recovery: {problem}" for problem in status["problems"]]


def _evidence_run_problems(root: pathlib.Path, model: FrontierModel) -> list[str]:
    path = frontier_evidence_run.evidence_run_path(root, model.label)
    record = _read_json(path)
    if not record:
        return [f"{path} missing; cannot verify signed Studio evidence-run bundle"]
    status = frontier_evidence_run.record_status(record, root=root, model=model, require_signature=True)
    return [f"studio evidence run: {problem}" for problem in status["problems"]]


def _source_provenance_problems(root: pathlib.Path, model: FrontierModel) -> list[str]:
    path = frontier_provenance.provenance_path(root, model.label)
    record = _read_json(path)
    if not record:
        return [f"{path} missing; cannot verify signed source provenance"]
    status = frontier_provenance.record_status(record, model=model, require_signature=True)
    return [f"source provenance: {problem}" for problem in status["problems"]]


def _evidence_file(path: pathlib.Path) -> dict[str, Any]:
    digest = _sha256_file(path)
    return {
        "path": str(path),
        "exists": digest is not None,
        "bytes": path.stat().st_size if digest and path.exists() else None,
        "sha256": digest,
    }


def _bundle_evidence_files(root: pathlib.Path, model: FrontierModel, require_ramcliff: bool) -> list[dict[str, Any]]:
    paths = [
        frontier_provenance.provenance_path(root, model.label),
        pathlib.Path(frontier_parity.parity_status(model, root)["record"]),
        frontier_coverage.baseline_path(root, model.label),
        frontier_coverage.eval_path(root, model.label),
        frontier_receipts.serve_path(root, model.label),
        frontier_doctor_recovery.recovery_path(root, model.label),
        frontier_experiments.matrix_path(root, model.label),
        frontier_evidence_run.evidence_run_path(root, model.label),
    ]
    if require_ramcliff:
        paths.append(frontier_receipts.ramcliff_path(root, model.label))
    return [_evidence_file(path) for path in paths]


def build_bundle(root: pathlib.Path, model: FrontierModel, *, require_ramcliff: bool = True) -> dict[str, Any]:
    parity = frontier_parity.parity_status(model, root)
    baseline = frontier_coverage.baseline_status(root, model.label)
    eval_cov = frontier_coverage.eval_status(root, model.label)
    serve = frontier_receipts.serve_status(root, model.label)
    doctor = frontier_doctor_recovery.recovery_status(root, model.label)
    experiment = frontier_experiments.experiment_status(root, model.label)
    evidence_run = frontier_evidence_run.evidence_run_status(root, model.label)
    ramcliff = frontier_receipts.ramcliff_status(root, model.label) if require_ramcliff else {
        "label": model.label,
        "ok": True,
        "problems": [],
        "waived": True,
        "reason": "bundle built with --no-require-ramcliff",
    }
    provenance = frontier_provenance.provenance_status(root, model.label)

    blockers = []
    for name, status in (
        ("baseline", baseline),
        ("eval", eval_cov),
    ):
        if not status.get("ok") and status.get("status") != "pass":
            problems = status.get("problems") or [f"{name} gate failed"]
            blockers.extend(f"{name}: {problem}" for problem in problems)

    blockers.extend(_source_provenance_problems(root, model))
    blockers.extend(_parity_receipt_problems(root, model))
    blockers.extend(_coverage_command_problems(root, model.label, "baseline"))
    blockers.extend(_coverage_command_problems(root, model.label, "eval"))
    blockers.extend(_native_receipt_problems(root, model.label, "serve"))
    if require_ramcliff:
        blockers.extend(_native_receipt_problems(root, model.label, "ramcliff"))
    blockers.extend(_doctor_recovery_problems(root, model))
    blockers.extend(_experiment_receipt_problems(root, model.label))
    blockers.extend(_evidence_run_problems(root, model))

    evidence_files = _bundle_evidence_files(root, model, require_ramcliff)
    for item in evidence_files:
        if not item["exists"]:
            blockers.append(f"evidence file missing: {item['path']}")

    bundle = {
        "schema": SCHEMA,
        "generated_at": _now(),
        "root": str(root),
        "git_commit": _git_commit(root),
        "label": model.label,
        "hf_id": model.hf_id,
        "claim_kind": "public-frontier-ramcliff" if require_ramcliff else "public-frontier-serve",
        "require_ramcliff": bool(require_ramcliff),
        "claim_admissible": not blockers,
        "blockers": blockers,
        "gates": {
            "parity": parity,
            "baseline": baseline,
            "eval": eval_cov,
            "serve": serve,
            "ramcliff": ramcliff,
            "doctor_recovery": doctor,
            "experiment": experiment,
            "studio_evidence_run": evidence_run,
            "source_provenance": provenance,
        },
        "public_win_contract": {
            "source_provenance": True,
            "native_tq": True,
            "rehydrate_f16": False,
            "all_linear": True,
            "gpu_bitslice": True,
            "parity": True,
            "tok_s": ">0",
            "baseline_coverage": True,
            "eval_coverage": True,
            "ramcliff_required": bool(require_ramcliff),
            "doctor_recovery_7b_plus": True,
            "studio_evidence_run": True,
            "exact_commands": True,
            "signed_evidence_hashes": True,
        },
        "evidence_files": evidence_files,
    }
    bundle["signature"] = {"algorithm": SIGN_ALG, "digest": _canonical_digest(bundle)}
    return bundle


def write_bundle(path: pathlib.Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(bundle, f, indent=2, sort_keys=True)
        f.write("\n")


def verify_bundle(path: pathlib.Path, root: pathlib.Path = ROOT) -> dict[str, Any]:
    data = _read_json(path)
    problems = []
    if not data:
        return {"path": str(path), "ok": False, "problems": [f"{path} missing or unreadable"]}
    if data.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA}")
    sig = data.get("signature") if isinstance(data.get("signature"), dict) else {}
    if sig.get("algorithm") != SIGN_ALG or sig.get("digest") != _canonical_digest(data):
        problems.append("signature digest mismatch")
    if data.get("claim_admissible") is not True:
        problems.append("claim_admissible is not true")
    for item in data.get("evidence_files") or []:
        p = pathlib.Path(item.get("path", ""))
        if not p.is_absolute():
            p = root / p
        digest = _sha256_file(p)
        if not digest:
            problems.append(f"evidence file missing: {item.get('path')}")
        elif digest != item.get("sha256"):
            problems.append(f"evidence file hash changed: {item.get('path')}")
    return {
        "path": str(path),
        "label": data.get("label"),
        "ok": not problems,
        "problems": problems,
        "claim_admissible": data.get("claim_admissible"),
    }


def bundle_status(root: pathlib.Path, label: str) -> dict[str, Any]:
    return verify_bundle(claim_bundle_path(root, label), root)


def claim_rollup(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    rows = [bundle_status(root, label) for label in labels]
    blocked = [row.get("label") or label for row, label in zip(rows, labels) if not row["ok"]]
    return {
        "schema": "hawking.frontier_claim_rollup.v1",
        "model_count": len(labels),
        "passed_count": len(labels) - len(blocked),
        "blocked_count": len(blocked),
        "blocked_labels": blocked,
        "rows": rows,
        "ok": not blocked,
    }


def _labels(args_labels: list[str]) -> list[FrontierModel]:
    if not args_labels:
        return list(FRONTIER_MODELS)
    models = []
    for label in args_labels:
        model = frontier_by_label(label)
        if not model:
            raise SystemExit(f"unknown frontier label: {label}")
        models.append(model)
    return models


def cmd_build(args) -> int:
    models = _labels(args.label)
    all_ok = True
    written = []
    for model in models:
        bundle = build_bundle(ROOT, model, require_ramcliff=not args.no_require_ramcliff)
        out = pathlib.Path(args.out) if args.out and len(models) == 1 else claim_bundle_path(ROOT, model.label)
        write_bundle(out, bundle)
        written.append(str(out))
        all_ok = all_ok and bundle["claim_admissible"]
        print(f"[frontier-claims] wrote {out}  admissible={bundle['claim_admissible']} "
              f"blockers={len(bundle['blockers'])}", file=sys.stderr)
    if args.json:
        print(json.dumps({"written": written, "ok": all_ok}, indent=2, sort_keys=True))
    return 0 if all_ok else 1


def cmd_verify(args) -> int:
    paths = [pathlib.Path(p) for p in args.path]
    if not paths:
        paths = [claim_bundle_path(ROOT, model.label) for model in FRONTIER_MODELS]
    rows = [verify_bundle(path, ROOT) for path in paths]
    ok = all(row["ok"] for row in rows)
    if args.json:
        print(json.dumps({"schema": "hawking.frontier_claim_verify.v1", "ok": ok, "rows": rows},
                         indent=2, sort_keys=True))
    else:
        print(f"# frontier claim bundle verify: {'OK' if ok else 'BLOCKED'}")
        for row in rows:
            print(f"{(row.get('label') or row['path']):18s} {'OK' if row['ok'] else 'BLOCK'} {row['path']}")
            for problem in row["problems"][:5]:
                print(f"  - {problem}")
    return 0 if ok else 1


def selftest() -> bool:
    import tempfile

    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        model = FRONTIER_MODELS[0]
        missing = build_bundle(root, model)
        check("missing bundle is blocked", not missing["claim_admissible"] and missing["blockers"])
        cond = root / COND_DIR
        cond.mkdir(parents=True)
        artifact_hash = "a" * 64
        common = {
            "model": model.label,
            "machine_class": "Studio-M1Ultra-128",
            "git_commit": "selftest",
            "artifact_sha256": artifact_hash,
            "commands": ["selftest command"],
        }
        provenance_record, _ = frontier_provenance.sign_record(
            frontier_provenance.complete_record(model),
            model=model,
        )
        (cond / f"{model.label}_source_provenance.json").write_text(json.dumps(provenance_record))
        family = frontier_parity._family_spec(model)
        parity_record = {
            "schema": "hawking.frontier_parity.v1",
            "model": model.label,
            "hf_id": model.hf_id,
            "family": family["family"],
            "receipt_state": "final",
            "source": "measured",
            "machine_class": "Studio-M1Ultra-128",
            "status": "pass",
            "reference_backend": "selftest",
            "tokenizer_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "prompt_count": 4,
            "max_logit_abs_err": 0.0,
            "greedy_match_tokens": 16,
            "git_commit": "selftest",
            "commands": ["selftest parity"],
            "reference_receipt": "selftest://reference-logits",
            "hawking_receipt": "selftest://hawking-logits",
            "required_native_features": family["required_native_features"],
            "verified_native_features": family["required_native_features"],
        }
        parity_record, _ = frontier_parity_runner.sign_record(parity_record, model=model)
        (cond / f"{model.label}_parity.json").write_text(json.dumps(parity_record))
        baseline_record = {
            "schema": "hawking.frontier_baselines.v1",
            "model": model.label,
            "receipt_state": "final",
            "source": "real",
            "machine_class": "Studio-M1Ultra-128",
            "same_box": True,
            "baselines": [
                {
                    "name": req["name"],
                    "status": "measured",
                    "same_box": True,
                    "command": f"selftest baseline {i}",
                    "artifact": f"selftest://baseline/{i}",
                    "metrics": {"tok_s": 1.0},
                }
                for i, req in enumerate(frontier_coverage.BASELINE_REQUIREMENTS)
            ],
        }
        baseline_record, _ = frontier_coverage_runner.sign_record(baseline_record, kind="baseline")
        (cond / f"{model.label}_baselines.json").write_text(json.dumps(baseline_record))
        eval_record = {
            "schema": "hawking.frontier_eval_coverage.v1",
            "model": model.label,
            "receipt_state": "final",
            "source": "real",
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
        (cond / f"{model.label}_eval.json").write_text(json.dumps(eval_record))
        serve_record = {
            **common,
            "schema": "hawking.frontier_serve.v1",
            "receipt_state": "final",
            "source": "measured",
            "status": "pass",
            "native_tq": True,
            "rehydrate_f16": False,
            "tq_strict": True,
            "all_linear": True,
            "gpu_bitslice": True,
            "served_forward_pass": True,
            "parity_pass": True,
            "tok_s": 1.0,
            "memory_peak_gb": 4.0,
            "memory_resident_gb": 3.5,
            "unified_memory_gb": 128.0,
            "resident_memory_ok": True,
            "load_receipt": "selftest://serve-load",
            "served_forward_receipt": "selftest://serve-forward",
            "parity_receipt": "selftest://serve-parity",
        }
        serve_record, _ = frontier_receipt_runner.sign_record(serve_record, kind="serve")
        (cond / f"{model.label}_serve.json").write_text(json.dumps(serve_record))
        ramcliff_record = {
            **common,
            "schema": "hawking.frontier_ramcliff.v1",
            "receipt_state": "final",
            "source": "measured",
            "verdict": "CLIFF-WIN",
            "served_native_tq": True,
            "tok_s_resident": 20.0,
            "tok_s_swapping": 1.0,
            "j_per_tok_resident": 1.0,
            "j_per_tok_swapping": 3.0,
            "cliff_x": 20.0,
            "gate": {
                "condensed_resident": True,
                "served_native_tq": True,
                "q4k_overflows_box": True,
                "cliff_x_over_gate": True,
                "resident_lower_energy": True,
            },
            "powermetrics_receipt": "selftest://powermetrics",
            "baseline_receipt": "selftest://q4k-swap",
        }
        ramcliff_record, _ = frontier_receipt_runner.sign_record(ramcliff_record, kind="ramcliff")
        (cond / f"{model.label}_ramcliff.json").write_text(json.dumps(ramcliff_record))
        doctor_record, _ = frontier_doctor_recovery.sign_record(
            frontier_doctor_recovery.complete_record(model),
            model=model,
        )
        (cond / f"{model.label}_doctor_recovery.json").write_text(json.dumps(doctor_record))
        experiment_record = {
            "schema": "hawking.frontier_experiment_matrix.v1",
            "model": model.label,
            "source": "real",
            "receipt_state": "final",
            "machine_class": "Studio-M1Ultra-128",
            "git_commit": "selftest",
            "experiments": {
                "floor_seeds": [
                    {"category": "floor_seed", "seed": seed, "status": "pass", "receipt": "selftest"}
                    for seed in (1, 2, 3)
                ],
                "calibration_ablations": [
                    {"category": "calibration_ablations", "name": name, "status": "pass", "receipt": "selftest"}
                    for name in ("domain_matched_calib", "mixed_domain_calib", "awq_alpha_sweep", "residual_depth_sweep")
                ],
                "bpw_ladder": [
                    {"category": "bpw_ladder", "bpw": bpw, "status": "pass", "receipt": "selftest"}
                    for bpw in (1, 2, 3, 4)
                ],
                "moe_expert_ablation": [
                    {"category": "moe_expert_ablation", "status": "pass", "receipt": "selftest"}
                ],
                "ramcliff_repeats": [
                    {"category": "ramcliff_repeats", "run_type": run_type, "status": "pass", "receipt": "selftest"}
                    for run_type in ("cold", "cold", "cold", "warm", "warm", "warm")
                ],
                "baseline_variants": [
                    {"category": "baseline_variants", "name": name, "status": "pass", "receipt": "selftest"}
                    for name in ("a", "b", "c", "d")
                ],
                "null_certification": [
                    {"category": "null_certification", "name": name, "status": "certified", "receipt": "selftest"}
                    for name in ("null_a", "null_b")
                ],
                "rebake_or_hash_verify": [
                    {"category": "rebake_or_hash_verify", "status": "verified", "receipt": "selftest"}
                ],
            },
        }
        experiment_record, _ = frontier_experiment_runner.sign_record(experiment_record, label=model.label)
        (cond / f"{model.label}_experiment_matrix.json").write_text(json.dumps(experiment_record))
        artifact = root / "scratch" / "selftest.tq"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"selftest-tq")
        inventory = {
            "schema": "hawking.frontier_artifact_inventory.v1",
            "generated_at": _now(),
            "label": model.label,
            "hf_id": model.hf_id,
            "git_commit": "selftest",
            "artifacts": [{
                "path": str(artifact),
                "bytes": artifact.stat().st_size,
                "gb": round(artifact.stat().st_size / 1e9, 6),
                "sha256": _sha256_file(artifact),
            }],
        }
        (cond / f"{model.label}_artifact_inventory.json").write_text(json.dumps(inventory))
        evidence_run = frontier_evidence_run.build_record(
            root,
            model,
            run_id="selftest-studio-run",
            runner_command="hawking studio run-next --selftest-evidence-run",
            source_release_decision="delete_source_after_verified_bake",
            source_release_command=f"hawking studio release-source {model.label} --dry-run",
            source_release_reason="selftest source lifecycle decision",
            decided_by="selftest",
        )
        evidence_run, _ = frontier_evidence_run.sign_record(evidence_run, root=root, model=model)
        (cond / f"{model.label}_studio_evidence_run.json").write_text(json.dumps(evidence_run))
        bundle = build_bundle(root, model)
        check("complete bundle is admissible", bundle["claim_admissible"])
        path = claim_bundle_path(root, model.label)
        write_bundle(path, bundle)
        check("signed bundle verifies", verify_bundle(path, root)["ok"])
        (cond / f"{model.label}_serve.json").write_text("{}")
        check("stale evidence breaks verify", not verify_bundle(path, root)["ok"])
    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def cmd_selftest(args) -> int:
    return 0 if selftest() else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build/verify signed frontier claim bundles.")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("build", help="build signed claim bundle(s)")
    p.add_argument("label", nargs="*", help="frontier label(s); default all")
    p.add_argument("--out", default="", help="output path for a single label")
    p.add_argument("--no-require-ramcliff", action="store_true",
                   help="build a serve-only claim bundle; public RAM-cliff wins should not use this")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("verify", help="verify signed claim bundle(s)")
    p.add_argument("path", nargs="*", help="bundle path(s); default all manifest bundle paths")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify)
    p = sub.add_parser("selftest", help="synthetic claim-bundle tests")
    p.set_defaults(func=cmd_selftest)
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    if not args.cmd:
        args = ap.parse_args(["verify"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
