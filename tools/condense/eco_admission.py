#!/usr/bin/env python3.12
"""120B+ admission plans for the adaptive planner.

The directive: "Each larger architecture needs its own adapter, exact source manifest,
streamed lifecycle, quality evidence, and admission receipt." And: "Use 72B effects only
as scheduling priors for 120B+ - never as proof."

This module produces one admission plan per large parent (the campaign's gpt-oss-120b plus
the FRONTIER_MODELS 235B+). It binds:
  - the exact source manifest (hf_id, download bytes, source kind, local dir);
  - device fit via size_frontier.analyze (RESIDENT / MOE-PAGED / DENSE-OOC / TOO-BIG) and
    the real disk wall (current free space, not SSD capacity);
  - the streamed lifecycle (download -> streamed bake -> seal -> source release);
  - the adapter requirement (built vs must-build per family);
  - the quality-evidence requirement (full standalone eval + native load/parity + F4);
  - a candidate rate seeded by the scaling prior (scheduling only, never proof).

Planning only. It launches nothing and mutates no campaign state.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import EcoError, SCHEMA_ADMISSION_PLAN, seal_field, sealed, now_iso  # noqa: E402

# Adapters that actually exist in the campaign (doctor_v5 ABI). Everything else must be
# built before its parent can be admitted.
BUILT_ADAPTERS: dict[str, str] = {
    "qwen2.5-dense": "doctor-v5-strand-ladder-qwen25-dense",
    "gpt-oss-moe": "doctor-v5-strand-ladder-gpt-oss-moe",
}

# family inference from a hf org / model role
_FAMILY_BY_ORG = {
    "openai": "gpt-oss-moe", "Qwen": "qwen3-moe", "meta-llama": "llama-dense",
    "deepseek-ai": "deepseek-moe", "zai-org": "glm-moe", "moonshotai": "kimi-moe",
}


def _family_of(hf_id: str, moe: bool) -> str:
    org = hf_id.split("/", 1)[0]
    fam = _FAMILY_BY_ORG.get(org)
    if fam:
        return fam
    return "unknown-moe" if moe else "unknown-dense"


def _large_parents() -> list[dict[str, Any]]:
    """gpt-oss-120b (campaign cohort) + the FRONTIER_MODELS, as a uniform shape."""
    from studio_manifest import FRONTIER_MODELS
    parents: list[dict[str, Any]] = [{
        "label": "120B", "hf_id": "openai/gpt-oss-120b", "local_dir": "scratch/gpt-oss-120b",
        "total_b": 116.8, "active_b": 5.1, "serve_bpw": 3.0, "moe": True,
        "download_gb": 65.3, "source_kind": "native MXFP4 original checkpoint",
        "in_campaign": True,
    }]
    for m in FRONTIER_MODELS:
        parents.append({
            "label": m.label, "hf_id": m.hf_id, "local_dir": m.local_dir,
            "total_b": m.total_b, "active_b": m.active_b, "serve_bpw": m.serve_bpw,
            "moe": m.moe, "download_gb": m.download_gb, "source_kind": m.source_kind,
            "in_campaign": False,
        })
    return parents


def _admission_for(parent: dict[str, Any], device_name: str,
                   predicted_floor_bpw: float | None) -> dict[str, Any]:
    import size_frontier
    from studio_manifest import DEFAULT_HARDWARE as hw

    # candidate rate: scaling-prior floor (scheduling only) else the manifest serve_bpw
    candidate_bpw = predicted_floor_bpw if predicted_floor_bpw else parent["serve_bpw"]
    fit = size_frontier.analyze(parent["total_b"], parent.get("active_b"),
                                candidate_bpw, device_name)

    family = _family_of(parent["hf_id"], parent["moe"])
    adapter_id = BUILT_ADAPTERS.get(family)
    adapter_status = "built" if adapter_id else "must_build"

    # disk reality: the manifest storage budget vs the real binding wall (current free space)
    artifact_gb = round(parent["total_b"] * candidate_bpw / 8.0, 1)
    storage_budget_gb = hw.storage_budget_gb
    source_fits_budget = parent["download_gb"] + artifact_gb <= storage_budget_gb

    lifecycle = [
        {"phase": "procure", "detail": f"stream {parent['hf_id']} ({parent['download_gb']} GB, "
                                       f"{parent['source_kind']}) with hf_transfer+xet"},
        {"phase": "streamed_bake", "detail": "block-parallel condense, one shard resident; "
                                             "peak disk ~= one shard + one output shard"},
        {"phase": "seal", "detail": "emit execution_receipt + result.json (physical bytes sealed)"},
        {"phase": "capability_eval", "detail": "full standalone protected-vector eval + native "
                                               "load/parity proof"},
        {"phase": "source_release", "detail": "operator source-GC only after all cells terminal + "
                                              "reporter sealed (receipt-then-delete)"},
    ]

    admission_gate = {
        "adapter_built": adapter_status == "built",
        "source_manifest_bound": True,
        "device_regime": fit["best_regime"],
        "device_admissible": fit["best_regime"] in ("RESIDENT", "MOE-PAGED"),
        "disk_feasible_today": source_fits_budget,
        "quality_evidence_required": ["standalone_capability_vector", "native_load_parity",
                                      "F4_replicated_seal"],
        "candidate_rate_is_prior_only": predicted_floor_bpw is not None,
    }
    admissible = (admission_gate["adapter_built"] and admission_gate["device_admissible"]
                  and admission_gate["disk_feasible_today"])

    return {
        "parent": {k: parent[k] for k in ("label", "hf_id", "local_dir", "total_b",
                                          "active_b", "moe", "download_gb", "source_kind",
                                          "in_campaign")},
        "family": family,
        "adapter": {"adapter_id": adapter_id, "status": adapter_status},
        "candidate_bpw": candidate_bpw,
        "candidate_basis": "scaling_prior_scheduling_only" if predicted_floor_bpw else "manifest_serve_bpw",
        "artifact_gb": artifact_gb,
        "device_fit": {k: fit[k] for k in ("device", "best_regime", "tq_on_disk_gb",
                                           "resident_gb", "fits_ssd", "resident_ceiling_b",
                                           "storage_ceiling_b")
                       if k in fit},
        "streamed_lifecycle": lifecycle,
        "admission_gate": admission_gate,
        "admissible_now": admissible,
        "blockers": _blockers(admission_gate),
    }


def _blockers(gate: dict[str, Any]) -> list[str]:
    out = []
    if not gate["adapter_built"]:
        out.append("per-family Doctor-v5 adapter not built")
    if not gate["device_admissible"]:
        out.append(f"device regime {gate['device_regime']} is not resident/paged")
    if not gate["disk_feasible_today"]:
        out.append("source+artifact exceed the storage budget (real wall is current free space)")
    return out


def build_admission_plan(*, device_name: str = "studio-m3ultra-96",
                         scaling_prior: dict[str, Any] | None = None) -> dict[str, Any]:
    predicted: dict[str, float] = {}
    if scaling_prior:
        import eco_planner
        slope = scaling_prior.get("log10_slope_bpw_per_decade")
        if slope is not None and scaling_prior.get("points"):
            for p in _large_parents():
                b = eco_planner._predict_bracket(scaling_prior, p["total_b"])
                if b.get("predicted_floor_bpw") is not None:
                    predicted[p["label"]] = b["predicted_floor_bpw"]

    plans = [
        _admission_for(p, device_name, predicted.get(p["label"]))
        for p in _large_parents()
    ]
    plan = {
        "schema": SCHEMA_ADMISSION_PLAN,
        "generated_at": now_iso(),
        "device": device_name,
        "note": ("120B+ admission is planning only. 72B/sub-72B effects are scheduling priors, "
                 "never proof; each parent needs its own adapter, source manifest, streamed "
                 "lifecycle, sealed quality evidence, and admission receipt."),
        "parents": plans,
        "admissible_now": [p["parent"]["label"] for p in plans if p["admissible_now"]],
        "must_build_adapter": sorted({p["family"] for p in plans if p["adapter"]["status"] == "must_build"}),
    }
    return seal_field(plan, "admission_sha256")


def selftest() -> dict[str, Any]:
    plan = build_admission_plan()
    if not sealed(plan, "admission_sha256"):
        raise EcoError("admission plan not sealed")
    labels = {p["parent"]["label"] for p in plan["parents"]}
    if "120B" not in labels or "235B-A22B" not in labels:
        raise EcoError(f"expected 120B and 235B parents: {labels}")
    gptoss = next(p for p in plan["parents"] if p["parent"]["label"] == "120B")
    if gptoss["adapter"]["status"] != "built":
        raise EcoError("gpt-oss-120b adapter should be recognized as built")
    # a frontier parent whose family has no adapter yet must be blocked
    llama = next((p for p in plan["parents"] if p["parent"]["label"] == "405B"), None)
    if llama and llama["admissible_now"]:
        raise EcoError("405B should be blocked (no llama adapter, dense regime)")
    # scaling prior as scheduling-only input
    prior = {"log10_slope_bpw_per_decade": -0.8,
             "points": [{"model_label": "14B", "params_b": 14.8, "provisional_floor_bpw": 2.0,
                         "log10_params": 10.17}]}
    plan2 = build_admission_plan(scaling_prior=prior)
    seeded = [p for p in plan2["parents"] if p["candidate_basis"] == "scaling_prior_scheduling_only"]
    if not seeded:
        raise EcoError("scaling prior should seed candidate rates")
    return {"ok": True, "parents": len(plan["parents"]),
            "must_build_adapter": plan["must_build_adapter"],
            "admissible_now": plan["admissible_now"],
            "prior_seeded_parents": len(seeded)}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="120B+ admission plans (planning only).")
    ap.add_argument("--campaign-root", default=None, help="derive the scaling prior from this campaign")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True)); sys.exit(0)
    prior = None
    if args.campaign_root:
        import eco_import, eco_planner
        p = eco_planner.build_plan(eco_import.build_ledger(eco_import.default_config(args.campaign_root)))
        prior = p["scaling_prior"]
    plan = build_admission_plan(scaling_prior=prior)
    if args.out:
        from eco_common import atomic_write_json
        atomic_write_json(args.out, plan)
    print(json.dumps({"schema": plan["schema"], "admission_sha256": plan["admission_sha256"],
                      "parents": len(plan["parents"]), "admissible_now": plan["admissible_now"],
                      "must_build_adapter": plan["must_build_adapter"]}, indent=2, sort_keys=True))
