#!/usr/bin/env python3.12
"""Immutable import of the active/old Doctor-v5 Ultra campaign as evidence + priors.

The directive: "Treat completed and running Doctor cells as immutable evidence... The
old ladder becomes evidence and priors." This module reads the campaign READ-ONLY,
by content hash, and produces a frozen `PriorLedger`. It never writes into a
campaign-owned directory and never reads a non-terminal (running/pending/blocked) cell,
so it is safe to run while the 72B cell is mid-bake.

For every TERMINAL cell it:
  - resolves the authoritative binding from `campaign_plan.json` (whose `plan_sha256`
    must match the pinned generation);
  - validates the on-disk self-seal (`result_sha256` for complete cells,
    `disposition_sha256` for unsupported/negative dispositions) using the campaign's
    exact canonical hashing form;
  - extracts sealed physical-byte evidence and PROVISIONAL quality (the campaign runs
    with `quality_claims_permitted:false`, so quality is engineering evidence only,
    never a sealed win).

The ledger is the prior set the adaptive planner brackets from. Nothing here promotes
a quality claim; that discipline is preserved end to end.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, SCHEMA_PRIOR_LEDGER, CAMPAIGN_PLAN_SHA256,
    read_json_safe, sealed, seal_field, sha_file, now_iso, is_sha256,
)

RESULT_SCHEMA = "hawking.doctor_v5_adapter_result.v1"
DISPOSITION_SCHEMA = "hawking.doctor_v5_ultra_disposition.v1"
TERMINAL_STATUSES = frozenset({"complete", "negative", "unsupported"})


@dataclasses.dataclass(frozen=True)
class ImportConfig:
    campaign_root: Path
    expected_plan_sha256: str = CAMPAIGN_PLAN_SHA256
    validate_evidence_files: bool = False  # re-hash disposition evidence artifacts too

    @property
    def plan_path(self) -> Path:
        return self.campaign_root / "campaign_plan.json"

    @property
    def queue_state_path(self) -> Path:
        return self.campaign_root / "queue_state.json"

    @property
    def results_dir(self) -> Path:
        return self.campaign_root / "results"

    @property
    def dispositions_dir(self) -> Path:
        return self.campaign_root / "dispositions"

    @property
    def campaign_repo_root(self) -> Path:
        # dispositions record evidence paths relative to the repo that owns the campaign.
        return self.campaign_root.parents[2]


def default_config(campaign_root: str | os.PathLike[str] | None = None) -> ImportConfig:
    from eco_common import repo_root
    root = Path(campaign_root) if campaign_root else repo_root() / "reports" / "condense" / "doctor_v5_ultra"
    # resolve the pinned generation at call time so a re-pin (or a test override) takes effect
    return ImportConfig(campaign_root=root, expected_plan_sha256=CAMPAIGN_PLAN_SHA256)


def _cohort(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """The full model cohort from the plan (so parents with no terminal cell yet, e.g.
    72B and 120B, are still visible to the planner as awaiting-evidence)."""
    cohort = plan.get("cohort")
    out: list[dict[str, Any]] = []
    if isinstance(cohort, list):
        for m in cohort:
            if isinstance(m, dict):
                out.append({
                    "label": m.get("label"),
                    "family": m.get("family") or m.get("model_family"),
                    "hf_id": m.get("hf_id"),
                    "nominal_params_b": m.get("nominal_params_b"),
                })
    return out


def _plan_index(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cells = plan.get("cells")
    if not isinstance(cells, list):
        raise EcoError("campaign_plan.cells is not a list")
    index: dict[str, dict[str, Any]] = {}
    for cell in cells:
        cid = cell.get("cell_id")
        if not isinstance(cid, str):
            continue
        index[cid] = {
            "cell_id": cid,
            "model_label": cell.get("model_label"),
            "model_family": cell.get("model_family"),
            "model_name": cell.get("model_name"),
            "hf_id": cell.get("hf_id"),
            "exact_stored_parameter_count": cell.get("exact_stored_parameter_count"),
            "nominal_params_b": cell.get("nominal_params_b"),
            "rate_id": str(cell.get("rate_id")),
            "rate_bpw": cell.get("rate_bpw"),
            "branch": cell.get("branch"),
            "cell_identity_sha256": cell.get("cell_identity_sha256"),
            "dependencies": cell.get("dependencies") or [],
            "priority": cell.get("priority"),
            "quality_claims_permitted": cell.get("quality_claims_permitted"),
            "disposition_path": cell.get("disposition_path"),
        }
    return index


def _import_complete(cid: str, meta: dict[str, Any], cfg: ImportConfig) -> dict[str, Any]:
    rf = cfg.results_dir / cid / "result.json"
    result = read_json_safe(rf)
    reasons: list[str] = []
    if result.get("schema") != RESULT_SCHEMA:
        reasons.append("wrong result schema")
    if result.get("status") != "complete":
        reasons.append("result status not complete")
    if not sealed(result, "result_sha256"):
        reasons.append("result self-seal invalid")
    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    cell = metrics.get("campaign_cell", {}) if isinstance(metrics.get("campaign_cell"), dict) else {}
    if cell.get("cell_identity_sha256") != meta.get("cell_identity_sha256"):
        reasons.append("cell_identity mismatch vs plan")
    phys = metrics.get("physical_accounting", {}) if isinstance(metrics.get("physical_accounting"), dict) else {}
    qual = metrics.get("quality_observation", {}) if isinstance(metrics.get("quality_observation"), dict) else {}
    claims = metrics.get("claims", {}) if isinstance(metrics.get("claims"), dict) else {}
    ppl = qual.get("ppl", {}) if isinstance(qual.get("ppl"), dict) else {}
    cap = qual.get("capability", {}) if isinstance(qual.get("capability"), dict) else {}
    record = {
        **_binding(meta),
        "status": "complete",
        "evidence_grade": _grade(reasons, "result self-seal invalid", "physical_sealed"),
        "seal_reasons": reasons,
        "result_sha256": result.get("result_sha256"),
        "physical": {
            "all_in_model_payload_bpw": phys.get("all_in_model_payload_bpw"),
            "target_physical_bpw": phys.get("target_physical_bpw"),
            "target_met": phys.get("target_met"),
            "model_payload_bytes": phys.get("model_payload_bytes"),
            "packed_2d_tensor_bpw": phys.get("packed_2d_tensor_bpw"),
            "lossless_non_2d_passthrough_bytes": phys.get("lossless_non_2d_passthrough_bytes"),
            "full_bundle_bytes": phys.get("full_bundle_bytes"),
        },
        "quality_provisional": {
            "ppl_relative_delta": ppl.get("relative_delta"),
            "ppl_baseline": ppl.get("baseline"),
            "capability_absolute_delta": cap.get("absolute_delta"),
            "quality_claims_permitted": qual.get("quality_claims_permitted", False),
            "status": qual.get("status"),
        },
        "claims": {
            "dominance": claims.get("dominance"),
            "quality": claims.get("quality"),
            "source_deletion": claims.get("source_deletion"),
            "target_physical_rate_met": claims.get("target_physical_rate_met"),
        },
    }
    return record


def _import_disposition(cid: str, meta: dict[str, Any], status: str, cfg: ImportConfig) -> dict[str, Any]:
    df = cfg.dispositions_dir / f"{cid}.json"
    if not df.exists() and isinstance(meta.get("disposition_path"), str):
        df = cfg.campaign_repo_root / meta["disposition_path"]
    disposition = read_json_safe(df)
    reasons: list[str] = []
    if disposition.get("schema") != DISPOSITION_SCHEMA:
        reasons.append("wrong disposition schema")
    if disposition.get("status") not in TERMINAL_STATUSES:
        reasons.append("disposition status not terminal")
    if not sealed(disposition, "disposition_sha256"):
        reasons.append("disposition self-seal invalid")
    if disposition.get("cell_identity_sha256") != meta.get("cell_identity_sha256"):
        reasons.append("cell_identity mismatch vs plan")
    artifacts = disposition.get("evidence_artifacts") or []
    validated_artifacts = 0
    if cfg.validate_evidence_files and isinstance(artifacts, list):
        for art in artifacts:
            path = cfg.campaign_repo_root / str(art.get("path", ""))
            if path.is_file() and not path.is_symlink():
                digest, _ = sha_file(path)
                if digest == art.get("sha256"):
                    validated_artifacts += 1
    return {
        **_binding(meta),
        "status": status,
        "evidence_grade": _grade(reasons, "disposition self-seal invalid", "disposition_sealed"),
        "seal_reasons": reasons,
        "disposition_sha256": disposition.get("disposition_sha256"),
        "reason_code": disposition.get("reason_code"),
        "detail": disposition.get("detail"),
        "evidence_artifact_count": len(artifacts) if isinstance(artifacts, list) else 0,
        "evidence_artifacts_validated": validated_artifacts,
    }


def _grade(reasons: list[str], seal_reason: str, ok_grade: str) -> str:
    """seal_invalid iff the self-seal specifically failed; binding_mismatch for other
    integrity reasons (schema/status/cell_identity); ok_grade when clean."""
    if not reasons:
        return ok_grade
    if seal_reason in reasons:
        return "seal_invalid"
    return "binding_mismatch"


def _binding(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": meta["cell_id"],
        "model_label": meta.get("model_label"),
        "model_family": meta.get("model_family"),
        "hf_id": meta.get("hf_id"),
        "exact_stored_parameter_count": meta.get("exact_stored_parameter_count"),
        "nominal_params_b": meta.get("nominal_params_b"),
        "rate_id": meta.get("rate_id"),
        "rate_bpw": meta.get("rate_bpw"),
        "branch": meta.get("branch"),
        "cell_identity_sha256": meta.get("cell_identity_sha256"),
        "dependencies": meta.get("dependencies"),
    }


def build_ledger(cfg: ImportConfig) -> dict[str, Any]:
    plan = read_json_safe(cfg.plan_path)
    plan_sha = plan.get("plan_sha256")
    if not is_sha256(plan_sha):
        raise EcoError("campaign_plan.plan_sha256 missing/malformed")
    if plan_sha != cfg.expected_plan_sha256:
        raise EcoError(
            f"plan_sha256 mismatch: on-disk {plan_sha} != pinned {cfg.expected_plan_sha256}; "
            "this scaffold is bound to one immutable generation and refuses a different plan"
        )
    index = _plan_index(plan)
    queue = read_json_safe(cfg.queue_state_path)
    if queue.get("plan_sha256") != plan_sha:
        raise EcoError("queue_state plan_sha256 does not match campaign_plan")
    cells = queue.get("cells")
    if not isinstance(cells, dict):
        raise EcoError("queue_state.cells is not an object")

    imported: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    unreadable: list[dict[str, str]] = []
    for cid, row in cells.items():
        status = row.get("status")
        if status not in TERMINAL_STATUSES:
            key = str(status)  # str-coerce so a missing/None status cannot break the seal sort
            skipped[key] = skipped.get(key, 0) + 1
            continue
        meta = index.get(cid)
        if meta is None:
            unreadable.append({"cell_id": cid, "why": "absent from plan index"})
            continue
        try:
            if status == "complete":
                imported.append(_import_complete(cid, meta, cfg))
            else:
                imported.append(_import_disposition(cid, meta, status, cfg))
        except EcoError as exc:
            unreadable.append({"cell_id": cid, "why": str(exc)})

    imported.sort(key=lambda r: (str(r.get("model_label")), float(r.get("rate_bpw") or 0.0), str(r.get("branch"))))
    seal_ok = sum(1 for r in imported if not r["seal_reasons"])
    ledger = {
        "schema": SCHEMA_PRIOR_LEDGER,
        "campaign_plan_sha256": plan_sha,
        "source_campaign_root": str(cfg.campaign_root),
        "imported_at": now_iso(),
        "matrix": plan.get("matrix"),
        "cohort": _cohort(plan),
        "cell_count_total": len(cells),
        "terminal_imported": len(imported),
        "seal_validated": seal_ok,
        "skipped_nonterminal": skipped,
        "unreadable": unreadable,
        "cells": imported,
    }
    return seal_field(ledger, "ledger_sha256")


def selftest() -> dict[str, Any]:
    """Build a tiny synthetic campaign in a scratch dir and round-trip it."""
    import tempfile
    from eco_common import atomic_write_json, hash_value

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "doctor_v5_ultra"
        (root / "results" / "toy-14b__2bpw__codec-control").mkdir(parents=True)
        (root / "dispositions").mkdir(parents=True)
        cell_identity = hash_value({"cell": "toy-14b__2bpw__codec-control"})
        disp_identity = hash_value({"cell": "toy-14b__0p1bpw__doctor-full"})
        plan = {
            "schema": "hawking.doctor_v5_ultra_campaign_plan.v1",
            "matrix": {"models": 1, "rates": 2, "branches": 1, "cells": 2},
            "cells": [
                {"cell_id": "toy-14b__2bpw__codec-control", "model_label": "14B",
                 "model_family": "qwen2.5-dense", "hf_id": "toy/14b", "rate_id": "2",
                 "rate_bpw": 2.0, "branch": "codec_control", "cell_identity_sha256": cell_identity,
                 "exact_stored_parameter_count": 14_000_000_000, "nominal_params_b": 14.0,
                 "dependencies": []},
                {"cell_id": "toy-14b__0p1bpw__doctor-full", "model_label": "14B",
                 "model_family": "qwen2.5-dense", "hf_id": "toy/14b", "rate_id": "0.1",
                 "rate_bpw": 0.1, "branch": "doctor_full", "cell_identity_sha256": disp_identity,
                 "exact_stored_parameter_count": 14_000_000_000, "nominal_params_b": 14.0,
                 "dependencies": ["codec_control"]},
            ],
        }
        plan = seal_field(plan, "plan_sha256")
        atomic_write_json(root / "campaign_plan.json", plan)

        result = {
            "schema": RESULT_SCHEMA, "status": "complete",
            "metrics": {
                "campaign_cell": {"cell_id": "toy-14b__2bpw__codec-control", "branch": "codec_control",
                                  "model_label": "14B", "rate_id": "2", "cell_identity_sha256": cell_identity},
                "physical_accounting": {"all_in_model_payload_bpw": 2.41, "target_physical_bpw": 2.0,
                                        "target_met": False, "model_payload_bytes": 4_200_000_000,
                                        "packed_2d_tensor_bpw": 2.40,
                                        "lossless_non_2d_passthrough_bytes": 1_000_000,
                                        "full_bundle_bytes": 4_260_000_000},
                "quality_observation": {"ppl": {"relative_delta": 0.06, "baseline": 12.0},
                                        "capability": {"absolute_delta": -0.02},
                                        "quality_claims_permitted": False, "status": "provisional_unsealed"},
                "claims": {"dominance": False, "quality": False, "source_deletion": False,
                           "target_physical_rate_met": False},
            },
        }
        result = seal_field(result, "result_sha256")
        atomic_write_json(root / "results" / "toy-14b__2bpw__codec-control" / "result.json", result)

        disposition = {
            "schema": DISPOSITION_SCHEMA, "status": "unsupported", "version": 1,
            "cell_id": "toy-14b__0p1bpw__doctor-full", "cell_identity_sha256": disp_identity,
            "plan_sha256": plan["plan_sha256"], "reason_code": "empirical-quality-cliff-adaptive-defer",
            "detail": "toy", "quality_claims_permitted": False, "source_deletion_permitted": False,
            "evidence_artifacts": [], "recorded_at": now_iso(),
        }
        disposition = seal_field(disposition, "disposition_sha256")
        atomic_write_json(root / "dispositions" / "toy-14b__0p1bpw__doctor-full.json", disposition)

        queue = {"plan_sha256": plan["plan_sha256"], "cells": {
            "toy-14b__2bpw__codec-control": {"status": "complete"},
            "toy-14b__0p1bpw__doctor-full": {"status": "unsupported"},
        }}
        atomic_write_json(root / "queue_state.json", queue)

        cfg = ImportConfig(campaign_root=root, expected_plan_sha256=plan["plan_sha256"])
        ledger = build_ledger(cfg)
        if ledger["terminal_imported"] != 2 or ledger["seal_validated"] != 2:
            raise EcoError(f"selftest import failed: {ledger['terminal_imported']}/{ledger['seal_validated']}")
        if not sealed(ledger, "ledger_sha256"):
            raise EcoError("selftest ledger not sealed")
        return {"ok": True, "terminal_imported": 2, "seal_validated": 2,
                "ledger_sha256": ledger["ledger_sha256"]}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Immutable campaign import (read-only priors).")
    ap.add_argument("--campaign-root", default=None, help="path to a doctor_v5_ultra campaign root")
    ap.add_argument("--out", default=None, help="write the sealed ledger here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
    else:
        cfg = default_config(args.campaign_root)
        led = build_ledger(cfg)
        if args.out:
            from eco_common import atomic_write_json
            atomic_write_json(args.out, led)
        summary = {k: led[k] for k in ("schema", "campaign_plan_sha256", "terminal_imported",
                                       "seal_validated", "skipped_nonterminal", "ledger_sha256")}
        summary["unreadable_count"] = len(led["unreadable"])
        print(json.dumps(summary, indent=2, sort_keys=True))
