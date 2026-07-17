#!/usr/bin/env python3.12
"""The `successor` CLI: one canonical control surface for the adaptive condenser.

Wires the event-sourced controller (succ_state) over the durable queue (succ_queue), the
source-bound admission probe (succ_admission), the adaptive engine (succ_engine + the
eco_planner frontier), the one-use transition (succ_transition), the Telegram service
(succ_telegram), evidence-closed GC (succ_gc), and the ETA model (succ_eta). Master goal
section 6.2 / 20. Every command emits machine-readable JSON. Nothing here activates the
successor while the legacy campaign runs; `start` boots into WAIT_OLD_RELEASE and arms.
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

from eco_common import repo_root  # noqa: E402

CAMPAIGN_ROOT_DEFAULT = "/Users/scammermike/Downloads/hawking/reports/condense/doctor_v5_ultra"
GENERATION = "gen-1"


def _succ_root() -> Path:
    return repo_root() / "reports" / "condense" / "event_horizon_successor"


def _controller():
    import succ_state
    return succ_state.Controller(_succ_root() / GENERATION, generation=GENERATION)


def cmd_audit(args) -> dict[str, Any]:
    import succ_audit
    out = _succ_root() / "audit"
    out.mkdir(parents=True, exist_ok=True)
    return succ_audit.capture_all(args.campaign_root, out)


def cmd_compile(args) -> dict[str, Any]:
    """Import legacy evidence, build the adaptive plan, probe adapters, materialize the queue,
    and boot the controller into WAIT_OLD_RELEASE (legacy still running)."""
    import eco_import, eco_planner, succ_queue, succ_admission, succ_state
    ledger = eco_import.build_ledger(eco_import.default_config(args.campaign_root))
    plan = eco_planner.build_plan(ledger)

    admissions: dict[str, Any] = {}
    for family, label in (("qwen2.5-dense", "72B"), ("gpt-oss-moe", "120B")):
        try:
            admissions[family] = succ_admission.admit(family, label)
        except Exception as exc:  # noqa: BLE001 - honest degrade
            admissions[family] = {"ready_for_execution": False,
                                  "blockers": [f"probe_error:{type(exc).__name__}"]}

    queue = succ_queue.Queue()
    for row in succ_queue.build_default_rows(admissions, generation=GENERATION):
        queue.upsert(row)

    # boot/advance the controller into WAIT_OLD_RELEASE if not already running
    ctrl = _controller()
    if ctrl.current_state() is None:
        ctrl.boot()
        ctrl.transition("AUDIT")
        ctrl.transition("WAIT_OLD_RELEASE", {"reason": "legacy campaign running"})
    return {
        "compiled": True,
        "terminal_evidence": ledger["terminal_imported"],
        "plan_sha256": plan["plan_sha256"],
        "queue": queue.summary(),
        "admissions": {k: {"ready": v.get("ready_for_execution"), "blockers": v.get("blockers")}
                       for k, v in admissions.items()},
        "controller_state": ctrl.current_state(),
    }


def cmd_start(args) -> dict[str, Any]:
    ctrl = _controller()
    if ctrl.current_state() is None:
        ctrl.boot()
        ctrl.transition("AUDIT")
        ctrl.transition("WAIT_OLD_RELEASE", {"reason": "legacy campaign running"})
    return {"started": True, "state": ctrl.current_state(),
            "note": "booted into WAIT_OLD_RELEASE; detached watcher install: successor watchdog-install --go"}


def cmd_status(args) -> dict[str, Any]:
    import succ_queue, succ_transition, succ_telegram
    ctrl = _controller()
    st = ctrl.status() if ctrl.current_state() is not None else {"state": None, "note": "not started"}
    out: dict[str, Any] = {"controller": st, "queue": succ_queue.Queue().summary()}
    try:
        tcfg = succ_transition.default_config(args.campaign_root)
        out["transition"] = succ_transition.status(tcfg) if hasattr(succ_transition, "status") else {}
    except Exception as exc:  # noqa: BLE001
        out["transition"] = {"error": type(exc).__name__}
    try:
        out["telegram"] = succ_telegram.telegram_status()
    except Exception as exc:  # noqa: BLE001
        out["telegram"] = {"error": type(exc).__name__}
    return out


def cmd_queue(args) -> dict[str, Any]:
    import succ_queue
    q = succ_queue.Queue()
    if args.model:
        row = q.load().get("rows", {}).get(args.model)
        return {"row": row} if row else {"error": f"no such row {args.model}"}
    return {"rows": q.rows()}


def cmd_explain_next(args) -> dict[str, Any]:
    import eco_import, eco_planner, succ_engine
    ledger = eco_import.build_ledger(eco_import.default_config(args.campaign_root))
    plan = eco_planner.build_plan(ledger)
    pick = succ_engine.next_experiment(plan)
    return {"next_experiment": pick} if pick else {"next_experiment": None,
                                                   "note": "no boundary probes; frontier resolved or evidence-less"}


def cmd_ping(args) -> dict[str, Any]:
    ctrl = _controller()
    if ctrl.current_state() != "WAIT_OLD_RELEASE":
        return {"pinged": False, "state": ctrl.current_state(), "note": "ping only in WAIT_OLD_RELEASE"}
    ctrl.transition("WAIT_OLD_RELEASE", {"heartbeat": ctrl.log.next_seq()})
    return {"pinged": True, "state": ctrl.current_state(), "events": ctrl.log.next_seq()}


def cmd_resume(args) -> dict[str, Any]:
    return _controller().resume()


def cmd_drain(args) -> dict[str, Any]:
    ctrl = _controller()
    cur = ctrl.current_state()
    import succ_state
    allowed = succ_state.TRANSITIONS.get(cur, ())
    if "DRAINED" not in allowed:
        return {"drained": False, "state": cur, "note": f"cannot drain from {cur}"}
    ctrl.transition("DRAINED", {"reason": "operator drain"})
    return {"drained": True, "state": ctrl.current_state()}


def cmd_verify(args) -> dict[str, Any]:
    import succ_queue
    ctrl = _controller()
    chain_ok, chain_why = (ctrl.log.verify_chain() if ctrl.current_state() is not None else (True, []))
    queue_ok = True
    queue_why = None
    try:
        succ_queue.Queue().load()
    except Exception as exc:  # noqa: BLE001
        queue_ok, queue_why = False, str(exc)
    return {"event_chain_ok": chain_ok, "event_chain_reasons": chain_why,
            "queue_seals_ok": queue_ok, "queue_reason": queue_why,
            "checkpoint_present": ctrl.checkpoint_path.exists()}


def cmd_arm_transition(args) -> dict[str, Any]:
    import succ_transition
    if not args.intent:
        return {"error": "arm-transition requires --intent <path>"}
    intent = json.loads(Path(args.intent).read_text())
    tcfg = succ_transition.default_config(args.campaign_root)
    gate = succ_transition.evaluate_gate(tcfg, intent) if hasattr(succ_transition, "evaluate_gate") \
        else succ_transition.evaluate_gate(args.campaign_root, intent)
    return {"armed": True, "gate_all_pass": gate.get("all_pass"), "reasons": gate.get("reasons"),
            "note": "armed and bound; activation happens automatically only after all gates pass"}


def cmd_transition_status(args) -> dict[str, Any]:
    import succ_transition
    tcfg = succ_transition.default_config(args.campaign_root)
    return succ_transition.status(tcfg)


def cmd_gc_plan(args) -> dict[str, Any]:
    import succ_gc
    return {"note": "gc-plan is evidence-closed; run succ_gc.gc_plan(candidates, dependency_index) "
            "with a real candidate set", "vocabulary": "see succ_gc.selftest for the safety gates"}


def cmd_frontier(args) -> dict[str, Any]:
    import succ_frontier
    man = succ_frontier.frontier_manifest()
    if args.out:
        from eco_common import atomic_write_json
        atomic_write_json(args.out, man)
    return {"manifest_sha256": man["manifest_sha256"], "row_ids": man["row_ids"],
            "regimes_selected": man["regimes_selected"],
            "heavy_execution_order": man["heavy_execution_order"],
            "parents": [{"row_id": r["row_id"], "status": r["current_status"],
                         "regime": r["selected_regime"], "revision": r["exact_revision"][:12],
                         "source_gb": r["envelopes"]["source_gb"],
                         "resident_ceiling_bpw": r["physical_fit"]["resident_ceiling_bpw"],
                         "blockers": len(r["blockers"])} for r in man["parents"]]}


def cmd_frontier_fit(args) -> dict[str, Any]:
    import succ_frontier
    out = {}
    for p in succ_frontier.PARENTS:
        fit = succ_frontier.physical_fit(p)
        out[p.row_id] = {"regime": fit["selected_regime"],
                         "resident_ceiling_bpw": fit["resident_ceiling_bpw"],
                         "anchor_fit": fit["anchor_fit"], "hybrid": fit["hybrid"],
                         "source_gb": fit["source_bytes_gb"]}
    return {"device_envelope": succ_frontier.default_envelope().as_dict(), "fit": out}


def cmd_frontier_admit(args) -> dict[str, Any]:
    """Upsert the 3 giant-parent rows into the durable controller queue (supervisor-loaded)."""
    import succ_frontier, succ_queue
    q = succ_queue.Queue()
    admitted = []
    for row in succ_frontier.queue_rows():
        q.upsert(row)
        admitted.append(row["parent_label"])
    return {"admitted_to_durable_queue": admitted, "queue": succ_queue.Queue().summary()}


def cmd_frontier_twin(args) -> dict[str, Any]:
    try:
        import succ_twin, succ_frontier
    except Exception as exc:  # noqa: BLE001
        return {"error": f"succ_twin not available: {exc}"}
    out = {}
    for p in succ_frontier.PARENTS:
        try:
            v = succ_twin.validate_twin(p)
            out[p.row_id] = {"all_green": v.get("all_green"),
                             "checks": {k: (val.get("pass") if isinstance(val, dict) else val)
                                        for k, val in (v.get("checks") or {}).items()}}
        except Exception as exc:  # noqa: BLE001
            out[p.row_id] = {"error": type(exc).__name__}
    return out


def cmd_frontier_press_plan(args) -> dict[str, Any]:
    try:
        import succ_press, succ_frontier
    except Exception as exc:  # noqa: BLE001
        return {"error": f"succ_press not available: {exc}"}
    out = {}
    for p in succ_frontier.PARENTS:
        try:
            out[p.row_id] = succ_press.press_plan(p)
        except Exception as exc:  # noqa: BLE001
            out[p.row_id] = {"error": type(exc).__name__}
    return out


def cmd_harvest(args) -> dict[str, Any]:
    import succ_harvest
    ds = succ_harvest.harvest(args.campaign_root)
    if args.out:
        from eco_common import atomic_write_json
        atomic_write_json(args.out, ds)
    return {"harvest_sha256": ds["harvest_sha256"], "terminal_rows": ds["terminal_rows"],
            "classification_counts": ds["classification_counts"]}


def cmd_retire_plan(args) -> dict[str, Any]:
    import succ_harvest, succ_retire
    ds = succ_harvest.harvest(args.campaign_root)
    ledger = succ_retire.build_retirement_ledger(ds)
    if args.out:
        from eco_common import atomic_write_json
        atomic_write_json(args.out, ledger)
    return {"ledger_sha256": ledger["ledger_sha256"], "retired_count": ledger["retired_count"],
            "replicated_collapse_boundaries": ledger["replicated_collapse_boundaries"],
            "applied_to": ledger["applied_to"]}


def cmd_eta(args) -> dict[str, Any]:
    import succ_harvest, succ_eta
    ds = succ_harvest.harvest(args.campaign_root)
    obs = succ_harvest.eta_observations(ds)
    if not obs:
        return {"note": "no completed cells with wall-time in the harvest yet"}
    model = succ_eta.fit_runtime(obs)
    segments = getattr(model, "segments", None) or getattr(model, "as_dict", lambda: {})()
    return {"observations": len(obs), "segments": segments if isinstance(segments, dict) else str(segments),
            "note": "per-(branch, full_cell) medians; never one global seconds-per-billion constant"}


def cmd_calibrate(args) -> dict[str, Any]:
    import succ_calibrate
    prog = succ_calibrate.build_calibration(args.model, campaign_root=args.campaign_root)
    if args.out:
        from eco_common import atomic_write_json
        atomic_write_json(args.out, prog)
    return {"model_label": prog["model_label"], "program_sha256": prog["program_sha256"],
            "extreme_status": prog["extreme_status"],
            "event_horizon_bracket": prog["event_horizon_bracket"],
            "experiments": len(prog["ordered_experiments"]),
            "untreated_frontier": prog["untreated_frontier"],
            "release_binding": prog["release_binding"]["executes"]}


def cmd_watch(args) -> dict[str, Any]:
    import succ_watch
    return succ_watch.watch_once(args.campaign_root, intent_path=args.intent, go=args.go)


def cmd_arm_template(args) -> dict[str, Any]:
    import succ_watch
    return succ_watch.write_intent_template(args.campaign_root, out_path=args.out)


def cmd_watch_plist(args) -> dict[str, Any]:
    import succ_watch
    return succ_watch.write_launchd_plist(out_path=args.out, interval=args.interval,
                                          campaign_root=args.campaign_root)


def cmd_telegram(args) -> dict[str, Any]:
    import succ_telegram
    if args.telegram_action == "status":
        return succ_telegram.telegram_status()
    if args.telegram_action == "test":
        if not args.go:
            return {"note": "telegram test requires --go to actually send; dry-run only",
                    "status": succ_telegram.telegram_status()}
        return succ_telegram.emit("real_test", {"generation": GENERATION}, force=True)
    return {"error": "unknown telegram action"}


def cmd_gravity_status(args) -> dict[str, Any]:
    import succ_gravity_policy as gp
    pol = gp.build_policy_manifest()
    return {"policy_version": pol["policy_version"], "enabled": pol["enabled"],
            "activation_gates": pol["activation_gates"], "invariant": pol["invariant"],
            "parents": {k: v["stress_start"]["label"] for k, v in pol["parent_stress_starts"].items()}}


def cmd_gravity_inspect(args) -> dict[str, Any]:
    import succ_gravity as sg
    import succ_gravity_policy as gp
    return {"parent": args.parent, "stress_start": gp.compute_stress_start(args.parent),
            "state": sg.new_parent_state(args.parent)}


def cmd_gravity_explain_next(args) -> dict[str, Any]:
    import succ_gravity as sg
    import succ_gravity_policy as gp
    ss = gp.parse_rate(gp.compute_stress_start(args.parent)["chosen_stress_rate"]["label"])
    p = gp.prior_for(args.parent)
    cand = [{"model_label": args.parent, "rate": gp.rate_identity(ss)["label"],
             "family": p["representation_families"][0], "near_boundary": True,
             "can_change_extreme": True, "distinguishes_degradation_from_collapse": True}]
    ranked = sg.rank_candidates(cand, {args.parent: sg.new_parent_state(args.parent)})
    return {"parent": args.parent, "direction": "upward from a sub-bit stress point",
            "next": ranked[0]}


def cmd_gravity_materialize(args) -> dict[str, Any]:
    import succ_gravity as sg
    return sg.materialize_live_parent_programs(args.parent, source_manifest_sha256=args.source_manifest_sha256)


def cmd_gravity_validate(args) -> dict[str, Any]:
    import succ_gravity as sg
    return sg.build_validation_doc()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="successor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-root", default=CAMPAIGN_ROOT_DEFAULT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("audit", "compile", "start", "status", "explain-next", "ping", "resume",
                 "drain", "verify", "transition-status", "gc-plan"):
        sub.add_parser(name)
    qp = sub.add_parser("queue"); qp.add_argument("--model", default=None)
    fp = sub.add_parser("frontier"); fp.add_argument("--out", default=None)
    sub.add_parser("frontier-fit")
    sub.add_parser("frontier-admit")
    sub.add_parser("frontier-twin")
    sub.add_parser("frontier-press-plan")
    hp = sub.add_parser("harvest"); hp.add_argument("--out", default=None)
    rp = sub.add_parser("retire-plan"); rp.add_argument("--out", default=None)
    sub.add_parser("eta")
    cp = sub.add_parser("calibrate"); cp.add_argument("--model", default="72B"); cp.add_argument("--out", default=None)
    wp = sub.add_parser("watch")
    wp.add_argument("--once", action="store_true"); wp.add_argument("--go", action="store_true")
    wp.add_argument("--intent", default=None)
    ap_t = sub.add_parser("arm-template"); ap_t.add_argument("--out", default=None)
    wpl = sub.add_parser("watch-plist"); wpl.add_argument("--out", default=None)
    wpl.add_argument("--interval", type=int, default=300)
    at = sub.add_parser("arm-transition"); at.add_argument("--intent", default=None)
    tg = sub.add_parser("telegram")
    tg.add_argument("telegram_action", choices=["test", "status"])
    tg.add_argument("--go", action="store_true")
    sub.add_parser("gravity-status")
    gi = sub.add_parser("gravity-inspect"); gi.add_argument("--parent", default="72B")
    ge = sub.add_parser("gravity-explain-next"); ge.add_argument("--parent", default="72B")
    gm = sub.add_parser("gravity-materialize"); gm.add_argument("--parent", default="72B")
    gm.add_argument("--source-manifest-sha256", default=None)
    sub.add_parser("gravity-validate")
    return ap


DISPATCH = {
    "audit": cmd_audit, "compile": cmd_compile, "start": cmd_start, "status": cmd_status,
    "queue": cmd_queue, "explain-next": cmd_explain_next, "ping": cmd_ping, "resume": cmd_resume,
    "drain": cmd_drain, "verify": cmd_verify, "arm-transition": cmd_arm_transition,
    "transition-status": cmd_transition_status, "gc-plan": cmd_gc_plan, "telegram": cmd_telegram,
    "calibrate": cmd_calibrate, "watch": cmd_watch, "arm-template": cmd_arm_template,
    "watch-plist": cmd_watch_plist, "harvest": cmd_harvest, "retire-plan": cmd_retire_plan,
    "eta": cmd_eta, "frontier": cmd_frontier, "frontier-fit": cmd_frontier_fit,
    "frontier-admit": cmd_frontier_admit, "frontier-twin": cmd_frontier_twin,
    "frontier-press-plan": cmd_frontier_press_plan,
    "gravity-status": cmd_gravity_status, "gravity-inspect": cmd_gravity_inspect,
    "gravity-explain-next": cmd_gravity_explain_next, "gravity-materialize": cmd_gravity_materialize,
    "gravity-validate": cmd_gravity_validate,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fn = DISPATCH.get(args.cmd)
    if fn is None:
        print(json.dumps({"error": f"unknown command {args.cmd}"})); return 1
    try:
        print(json.dumps(fn(args), indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
