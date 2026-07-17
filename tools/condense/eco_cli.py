#!/usr/bin/env python3.12
"""Unified CLI surface for the Hawking Condenser Ecosystem Frontier scaffold.

Every capability is reachable here without HIDE (the constitution's no-lock-in rule):

  eco_cli.py selftest                      run every module self-check
  eco_cli.py import   [--campaign-root R]  build the immutable prior ledger (read-only)
  eco_cli.py plan     [--campaign-root R]  build the adaptive EXTREME plan (plan-only)
  eco_cli.py pipeline                      show + validate the Press->Summon stage graph
  eco_cli.py passport --selftest           mint + verify an identity passport
  eco_cli.py admission[--campaign-root R]  120B+ admission plans (plan-only)
  eco_cli.py status   [--campaign-root R]  compose the Telegram status (dry; --send --go to deliver)
  eco_cli.py activation gate|status|activate|rollback   the fail-closed activation gate
  eco_cli.py materialize [--campaign-root R] [--out-dir D]   emit the full artifact bundle

Default-off: nothing here activates the ecosystem layer. `materialize` writes plan-only
artifacts under reports/condense/frontier_eco/ and touches no campaign-owned file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _all_selftests() -> dict[str, Any]:
    import eco_passport, eco_import, eco_planner, eco_pipeline, eco_activation, eco_status, eco_admission
    results = {}
    for name, mod in (("passport", eco_passport), ("import", eco_import), ("planner", eco_planner),
                      ("pipeline", eco_pipeline), ("activation", eco_activation),
                      ("status", eco_status), ("admission", eco_admission)):
        try:
            results[name] = mod.selftest()
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    results["all_ok"] = all(r.get("ok") for r in results.values() if isinstance(r, dict))
    return results


def _materialize(campaign_root: str | None, out_dir: str | None) -> dict[str, Any]:
    import eco_import, eco_planner, eco_pipeline, eco_admission, eco_passport
    from eco_common import atomic_write_json, eco_state_root

    out = Path(out_dir) if out_dir else eco_state_root() / "materialized"
    cfg = eco_import.default_config(campaign_root)
    ledger = eco_import.build_ledger(cfg)
    plan = eco_planner.build_plan(ledger)
    spec = eco_pipeline.pipeline_spec()
    admission = eco_admission.build_admission_plan(scaling_prior=plan["scaling_prior"])

    # mint one identity passport per parent that has terminal evidence
    passports = []
    for a in plan["parents"]:
        b = a["binding"]
        proxy = a.get("provisional_floor_proxy_bpw") or 4.0
        phys_bytes = int((b.get("exact_stored_parameter_count") or 0) * proxy / 8) or 1
        facets = {
            "artifact": {"family": b.get("model_family"), "label": b.get("model_label"),
                         "hf_id": b.get("hf_id")},
            "doctor_treatment": {"program": "diagnosis_driven", "controls_retained": 4},
            "physical_bytes": {
                "all_in_model_payload_bpw": proxy,
                "all_in_model_payload_bytes": phys_bytes,
                "byte_breakdown": {"base_payload_bytes": phys_bytes},
                # NOT a sealed measurement: this is the collapse-boundary planning proxy, so a
                # consumer never mistakes a scaffold passport's BPW for sealed campaign bytes.
                "evidence_grade": "planning_proxy",
                "basis": "collapse_boundary_proxy_bpw" if a.get("provisional_floor_proxy_bpw") else "default_4bpw",
            },
            "capability_contract": plan["contract"],
            "context_horizon": {"status": "unmeasured", "layer": "context_system"},
            "session_state": {"status": "unmeasured", "layer": "agent_system"},
            "device_profile": plan["device"],
            "client_compat": {"openai_chat": True, "mcp": True, "hide": True},
        }
        try:
            pp = eco_passport.mint_passport(
                facets, parent_label=b.get("model_label") or "?",
                rate_id=str(proxy), branch="scaffold",
                bindings={"campaign_plan_sha256": plan["campaign_plan_sha256"],
                          "prior_ledger_sha256": ledger.get("ledger_sha256")})
            passports.append({"model_label": b.get("model_label"),
                              "passport_sha256": pp["passport_sha256"]})
            atomic_write_json(out / "passports" / f"{b.get('model_label')}.json", pp)
        except Exception as exc:  # noqa: BLE001
            passports.append({"model_label": b.get("model_label"), "error": str(exc)})

    atomic_write_json(out / "prior_ledger.json", ledger)
    atomic_write_json(out / "adaptive_plan.json", plan)
    atomic_write_json(out / "pipeline_spec.json", spec)
    atomic_write_json(out / "admission_plan.json", admission)
    manifest = {
        "out_dir": str(out),
        "campaign_plan_sha256": plan["campaign_plan_sha256"],
        "prior_ledger_sha256": ledger.get("ledger_sha256"),
        "adaptive_plan_sha256": plan["plan_sha256"],
        "pipeline_spec_sha256": spec["spec_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "passports": passports,
        "parents_with_evidence": len(plan["parents"]),
        "parents_awaiting": [a["model_label"] for a in plan["parents_awaiting_evidence"]],
    }
    atomic_write_json(out / "MANIFEST.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eco_cli.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("selftest")
    for name in ("import", "plan", "admission"):
        p = sub.add_parser(name)
        p.add_argument("--campaign-root", default=None)
        p.add_argument("--out", default=None)
    sub.add_parser("pipeline")
    pp = sub.add_parser("passport"); pp.add_argument("--selftest", action="store_true")
    st = sub.add_parser("status")
    st.add_argument("--campaign-root", default=None)
    st.add_argument("--send", action="store_true"); st.add_argument("--go", action="store_true")
    ac = sub.add_parser("activation")
    ac.add_argument("action", choices=["gate", "status", "activate", "rollback"])
    ac.add_argument("--campaign-root", default=None); ac.add_argument("--go", action="store_true")
    mat = sub.add_parser("materialize")
    mat.add_argument("--campaign-root", default=None); mat.add_argument("--out-dir", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        res = _all_selftests()
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0 if res["all_ok"] else 1
    if args.cmd == "import":
        import eco_import
        led = eco_import.build_ledger(eco_import.default_config(args.campaign_root))
        if args.out:
            from eco_common import atomic_write_json
            atomic_write_json(args.out, led)
        print(json.dumps({k: led[k] for k in ("schema", "campaign_plan_sha256",
              "terminal_imported", "seal_validated", "ledger_sha256")}, indent=2, sort_keys=True))
        return 0
    if args.cmd == "plan":
        import eco_import, eco_planner
        led = eco_import.build_ledger(eco_import.default_config(args.campaign_root))
        plan = eco_planner.build_plan(led)
        if args.out:
            from eco_common import atomic_write_json
            atomic_write_json(args.out, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"],
              "parents": [{"label": a["binding"]["model_label"],
                           "extreme": a["extreme_candidate"].get("status"),
                           "floor_proxy_bpw": a["provisional_floor_proxy_bpw"]} for a in plan["parents"]],
              "awaiting": [a["model_label"] for a in plan["parents_awaiting_evidence"]],
              "scaling_trend": plan["scaling_prior"]["trend"]}, indent=2, sort_keys=True))
        return 0
    if args.cmd == "admission":
        import eco_admission, eco_import, eco_planner
        prior = None
        if args.campaign_root:
            p = eco_planner.build_plan(eco_import.build_ledger(eco_import.default_config(args.campaign_root)))
            prior = p["scaling_prior"]
        plan = eco_admission.build_admission_plan(scaling_prior=prior)
        if args.out:
            from eco_common import atomic_write_json
            atomic_write_json(args.out, plan)
        print(json.dumps({"admission_sha256": plan["admission_sha256"],
              "admissible_now": plan["admissible_now"],
              "must_build_adapter": plan["must_build_adapter"]}, indent=2, sort_keys=True))
        return 0
    if args.cmd == "pipeline":
        import eco_pipeline
        ok, why = eco_pipeline.validate_spec()
        print(json.dumps({"valid": ok, "reasons": why,
              "canonical_order": list(eco_pipeline.CANONICAL_ORDER)}, indent=2, sort_keys=True))
        return 0 if ok else 1
    if args.cmd == "passport":
        import eco_passport
        print(json.dumps(eco_passport.selftest(), indent=2, sort_keys=True))
        return 0
    if args.cmd == "status":
        import eco_status
        cfg = eco_status.default_config(args.campaign_root)
        if args.send and args.go:
            print(json.dumps(eco_status.send_status(cfg), indent=2, sort_keys=True))
        else:
            print(eco_status.compose_status(cfg)["text"])
            print("\n--- (dry run; pass --send --go to deliver) ---")
        return 0
    if args.cmd == "activation":
        import eco_activation
        cfg = eco_activation.default_config(args.campaign_root)
        if args.action == "gate":
            print(json.dumps(eco_activation.supersession_gate(cfg), indent=2, sort_keys=True))
        elif args.action == "activate":
            print(json.dumps(eco_activation.activate(cfg, go=args.go), indent=2, sort_keys=True))
        elif args.action == "rollback":
            print(json.dumps(eco_activation.rollback(cfg), indent=2, sort_keys=True))
        else:
            print(json.dumps(eco_activation.status(cfg), indent=2, sort_keys=True))
        return 0
    if args.cmd == "materialize":
        print(json.dumps(_materialize(args.campaign_root, args.out_dir), indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
