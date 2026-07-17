#!/usr/bin/env python3.12
"""Detached release watcher + arming artifacts for unattended operation (master goal 3, 6.4).

`watch_once` is the launchd tick: it takes the singleton lease, resumes (or boots) the
controller, and while in WAIT_OLD_RELEASE it samples resources, emits a heartbeat, and
re-evaluates the one-use transition gate against an armed intent. If (and only if) the gate
passes, the intent is valid, and go is set, it fires the transition ONCE. It launches no
heavy work and never touches the legacy campaign.

The two arming artifacts complete the State-B story without the agent overstepping:
  - `write_intent_template` builds an UNSIGNED transition intent bound to the exact current
    identities; the operator adds a signature (authenticity is theirs to provide).
  - `write_launchd_plist` writes the LaunchAgent plist to a file; the operator runs one
    `launchctl load` to make the watcher live. The agent does not install an auto-activating
    system agent or self-sign.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import EcoError, hash_value, now_iso, read_json_safe, atomic_write_json, repo_root  # noqa: E402

WATCH_LABEL = "com.hawking.successor.watcher"


class WatchError(EcoError):
    """Fail-closed watcher error."""


def _successor_root() -> Path:
    return repo_root() / "reports" / "condense" / "event_horizon_successor"


def watch_once(campaign_root: str, *, intent_path: str | None = None, go: bool = False,
               sender: Callable[[str], dict[str, Any]] | None = None,
               successor_root: str | None = None,
               lease_path: str | None = None) -> dict[str, Any]:
    """One watcher tick. Read-mostly: heartbeat + gate re-check; fires the transition only
    when the gate passes, an intent is armed, and go is set."""
    import succ_state, succ_watchdog, succ_transition
    sroot = Path(successor_root) if successor_root else _successor_root()
    lease_file = Path(lease_path) if lease_path else sroot / "watcher.lease"
    lease_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        lease = succ_watchdog.acquire(lease_file)
    except succ_watchdog.WatchdogError:
        return {"tick": False, "reason": "another watcher holds the lease (already-running)"}

    try:
        ctrl = succ_state.Controller(sroot / "gen-1", generation="gen-1")
        if ctrl.current_state() is None:
            ctrl.boot()
            ctrl.transition("AUDIT")
            ctrl.transition("WAIT_OLD_RELEASE", {"reason": "legacy campaign running"})
        state = ctrl.current_state()

        resources = succ_watchdog.sample_resources(path_for_disk=campaign_root)
        result: dict[str, Any] = {"tick": True, "state": state, "resources": resources}

        if state != "WAIT_OLD_RELEASE":
            result["note"] = f"not waiting (state={state}); no gate check"
            return result

        # heartbeat (dry unless a sender is supplied / go)
        try:
            import succ_telegram
            hb = succ_telegram.compose_event("waiting_old_release_heartbeat",
                                             {"generation": "gen-1", "state": state,
                                              "free_disk_gb": resources.get("free_disk_gb")})
            result["heartbeat"] = {"event_id": hb["event_id"], "sent": False}
            if sender is not None:
                sender(hb["text"])
                result["heartbeat"]["sent"] = True
        except Exception as exc:  # noqa: BLE001 - heartbeat must never break the tick
            result["heartbeat"] = {"error": type(exc).__name__}

        # gate re-check against the armed intent (if any)
        if intent_path and Path(intent_path).exists():
            intent = read_json_safe(intent_path)
            gate = succ_transition.evaluate_gate(campaign_root, intent, state_root=str(sroot))
            result["gate"] = {"all_pass": gate.get("all_pass"), "reasons": gate.get("reasons")}
            if gate.get("all_pass") and go:
                fired = succ_transition.execute_transition(intent, campaign_root, go=True,
                                                           state_root=str(sroot))
                result["transition"] = fired
                if fired.get("executed"):
                    ctrl.transition("IMPORT_LEGACY", {"reason": "release gate passed"})
                    result["state"] = ctrl.current_state()
        else:
            result["gate"] = {"all_pass": False, "reasons": ["no armed intent at intent_path"]}
        return result
    finally:
        lease.__exit__(None, None, None)


INTENT_TEMPLATE_SCHEMA = "hawking.successor.transition_intent_template.v1"


def write_intent_template(campaign_root: str, *, successor_repo: str | None = None,
                          out_path: str | None = None, generation: int = 1) -> dict[str, Any]:
    """Emit an UNSIGNED intent TEMPLATE bound to the exact current identities. It is NOT a
    valid intent (make_intent fail-closes without authorization, by design). The operator
    completes it by supplying an operator_signature and calling succ_transition.make_intent
    with these exact fields. The agent never self-signs."""
    import succ_transition
    queue = read_json_safe(Path(campaign_root) / "queue_state.json")
    legacy_plan = queue.get("plan_sha256")
    cells = queue.get("cells", {})
    expected_terminal = len(cells)  # the release condition is ALL cells terminal
    checkpoints = queue.get("report_checkpoints", {})

    sid = succ_transition.measure_successor_identity(successor_repo)
    template = {
        "schema": INTENT_TEMPLATE_SCHEMA,
        "generated_at": now_iso(),
        "signed": False,
        "make_intent_fields": {
            "legacy_plan_sha256": legacy_plan,
            "accepted_checkpoints": {"required_groups": sorted(checkpoints.keys()) or ["sub-120B", "120B"],
                                     "must_be_sealed_at_release": True},
            "successor_commit": sid.get("successor_commit"),
            "successor_tree_sha256": sid.get("successor_tree_sha256"),
            "successor_config_sha256": sid.get("successor_config_sha256"),
            "successor_queue_root_sha256": sid.get("successor_queue_root_sha256"),
            "expected_terminal_count": expected_terminal,
            "rollback_target": "default_off",
            "authorization_statement": "OPERATOR: authorize automatic successor transition after "
                                       "ALL legacy cells terminal, both group reports sealed, quiescent.",
            "generation": generation,
            "ttl_seconds": 30 * 24 * 3600,
        },
        "operator_todo": [
            "1. Review the bound identities below (legacy_plan_sha256, successor_commit/tree, "
            "expected_terminal_count = full cohort).",
            "2. Produce an operator_signature (a real detached signature over an operator key is "
            "preferred; a permission-restricted authorization_file + self-seal is the fallback, "
            "and is possession/integrity control, not cryptographic authenticity).",
            "3. Build the real sealed intent: "
            "succ_transition.make_intent(**make_intent_fields, operator_signature=<sig>).",
            "4. Arm it: succ_cli.py arm-transition --intent <intent.json>. The watcher then fires "
            "the transition automatically once the gate passes; the agent installs no auto-agent.",
        ],
    }
    out = Path(out_path) if out_path else _successor_root() / "transition" / "intent_template.json"
    atomic_write_json(out, template)
    return {"written": str(out), "legacy_plan_sha256": legacy_plan,
            "expected_terminal_count": expected_terminal,
            "successor_commit": sid.get("successor_commit"),
            "signed": False, "note": "operator must sign; agent does not self-sign"}


def write_launchd_plist(*, out_path: str | None = None, interval: int = 300,
                        campaign_root: str | None = None,
                        intent_path: str | None = None) -> dict[str, Any]:
    """Write the LaunchAgent plist to a FILE (does not install it). The operator runs one
    `launchctl load` to make the watcher live."""
    import succ_watchdog
    cr = campaign_root or "/Users/scammermike/Downloads/hawking/reports/condense/doctor_v5_ultra"
    ip = intent_path or str(_successor_root() / "transition" / "intent_template.json")
    program_args = [sys.executable, str(Path(_HERE) / "succ_cli.py"),
                    "--campaign-root", cr, "watch", "--once", "--go", "--intent", ip]
    plist = succ_watchdog.launchd_plist(WATCH_LABEL, program_args, interval)
    out = Path(out_path) if out_path else _successor_root() / "transition" / f"{WATCH_LABEL}.plist"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(plist)
    return {"written": str(out), "label": WATCH_LABEL, "interval_seconds": interval,
            "install_command": f"launchctl load {out}",
            "uninstall_command": f"launchctl unload {out}",
            "note": "agent wrote the plist but did NOT install it; loading is the operator's "
                    "explicit standing-agent authorization"}


def selftest() -> dict[str, Any]:
    import tempfile
    from eco_common import seal_field
    with tempfile.TemporaryDirectory() as d:
        croot = Path(d) / "doctor_v5_ultra"
        croot.mkdir(parents=True)
        # a running legacy campaign -> gate never passes, no transition
        atomic_write_json(croot / "queue_state.json", {
            "plan_sha256": "a" * 64,
            "cells": {"c1": {"status": "complete"}, "c2": {"status": "running"}},
            "report_checkpoints": {"sub-120B": None, "120B": None}})
        sroot = Path(d) / "successor"

        sent: list[str] = []
        r = watch_once(str(croot), go=True, sender=lambda t: (sent.append(t), {"message_id": 1})[1],
                       successor_root=str(sroot), lease_path=str(sroot / "w.lease"))
        if not r["tick"] or r["state"] != "WAIT_OLD_RELEASE":
            raise WatchError(f"tick should run and wait: {r}")
        if r["heartbeat"].get("sent") is not True:
            raise WatchError("heartbeat should send with an injected sender")
        if r["gate"]["all_pass"] is not False:
            raise WatchError("gate must not pass with no armed intent")
        # a second concurrent tick while the lease is held is refused
        # (acquire+release happen within watch_once, so serial ticks are fine; test the lease)
        import succ_watchdog
        held = succ_watchdog.acquire(sroot / "w2.lease")
        try:
            blocked = watch_once(str(croot), successor_root=str(sroot), lease_path=str(sroot / "w2.lease"))
        finally:
            held.__exit__(None, None, None)
        if blocked.get("tick") is not False:
            raise WatchError("a tick must be refused while the lease is held")

        # intent template is unsigned and bound to the exact plan hash
        tmpl = write_intent_template(str(croot), successor_repo=str(repo_root()),
                                     out_path=str(sroot / "intent.json"))
        if tmpl["signed"] is not False or tmpl["legacy_plan_sha256"] != "a" * 64:
            raise WatchError(f"template should be unsigned + bound: {tmpl}")
        # launchd plist written but not installed
        pl = write_launchd_plist(out_path=str(sroot / "w.plist"), campaign_root=str(croot),
                                 intent_path=str(sroot / "intent.json"))
        if "launchctl load" not in pl["install_command"] or not Path(pl["written"]).exists():
            raise WatchError("plist should be written with an install command")
    return {"ok": True, "tick_waits": True, "heartbeat_sent": True, "gate_blocked_no_intent": True,
            "lease_refuses_concurrent": True, "intent_template_unsigned": True,
            "launchd_written_not_installed": True}


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2, sort_keys=True))
