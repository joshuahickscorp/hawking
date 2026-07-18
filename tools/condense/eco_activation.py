#!/usr/bin/env python3.12
"""Atomic, fail-closed activation + rollback for the ecosystem frontier.

The directive: "activation waits until the generation is terminal, reporter-sealed,
checkpoint-accepted, quiescent, and signed for supersession." And: "Do not merge or
activate before the signed campaign release boundary."

This module is the gate. It is DEFAULT-OFF: with no activation manifest on disk the
ecosystem layer is inactive. `activate` refuses unless EVERY supersession condition holds
AND an operator supersession signature is present AND `--go` is passed. It never signs on
its own behalf: the signature is an out-of-band operator authorization that this tool only
reads and verifies. `rollback` restores the previous manifest (or returns to default-off).

While the live campaign is running (a `running` cell, no reporter checkpoint), the gate
fails and activation is impossible by construction.
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
    EcoError, SCHEMA_ACTIVATION, SCHEMA_ACTIVATION_MANIFEST, CAMPAIGN_PLAN_SHA256,
    read_json_safe, atomic_write_json, hash_value, seal_field, sealed, is_sha256,
    now_iso, eco_state_root,
)

SIGNATURE_SCHEMA = "hawking.eco.supersession_signature.v1"
TERMINAL_STATUSES = frozenset({"complete", "negative", "unsupported"})


@dataclasses.dataclass(frozen=True)
class ActivationConfig:
    campaign_root: Path
    state_root: Path
    expected_plan_sha256: str = CAMPAIGN_PLAN_SHA256

    @property
    def signature_path(self) -> Path:
        return self.state_root / "activation" / "supersession_signature.json"

    @property
    def manifest_path(self) -> Path:
        return self.state_root / "activation" / "current.json"

    @property
    def previous_path(self) -> Path:
        return self.state_root / "activation" / "previous.json"


def default_config(campaign_root: str | os.PathLike[str] | None = None,
                   state_root: str | os.PathLike[str] | None = None) -> ActivationConfig:
    from eco_common import repo_root
    croot = Path(campaign_root) if campaign_root else repo_root() / "reports" / "condense" / "doctor_v5_ultra"
    sroot = Path(state_root) if state_root else eco_state_root()
    # resolve the pinned generation at call time so a re-pin (or a test override) takes effect
    return ActivationConfig(campaign_root=croot, state_root=sroot,
                            expected_plan_sha256=CAMPAIGN_PLAN_SHA256)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def supersession_gate(cfg: ActivationConfig) -> dict[str, Any]:
    """Evaluate the five supersession conditions read-only. Never mutates anything."""
    reasons: list[str] = []
    terminal = reporter_sealed = checkpoint_accepted = quiescent = signed = False

    try:
        queue = read_json_safe(cfg.campaign_root / "queue_state.json")
    except EcoError as exc:
        return {"all_pass": False, "reasons": [f"queue_state unreadable: {exc}"],
                "terminal": False, "reporter_sealed": False, "checkpoint_accepted": False,
                "quiescent": False, "signed": False}

    plan_bound = queue.get("plan_sha256") == cfg.expected_plan_sha256
    if not plan_bound:
        reasons.append("queue plan_sha256 does not match the pinned generation")

    cells = queue.get("cells", {})
    if isinstance(cells, dict) and cells:
        nonterminal = {cid: r.get("status") for cid, r in cells.items()
                       if r.get("status") not in TERMINAL_STATUSES}
        terminal = not nonterminal
        if not terminal:
            reasons.append(f"{len(nonterminal)} cells not terminal (e.g. "
                           f"{next(iter(nonterminal.items()))})")
    else:
        reasons.append("queue_state.cells missing")

    checkpoints = queue.get("report_checkpoints", {})
    if isinstance(checkpoints, dict) and checkpoints:
        reporter_sealed = all(v is not None for v in checkpoints.values())
        # a checkpoint is accepted only if it is a sha256 or a dict that carries a sha256
        # self-seal (an empty dict or a bare non-None marker is NOT acceptance).
        checkpoint_accepted = reporter_sealed and all(_checkpoint_ok(v) for v in checkpoints.values())
        if not reporter_sealed:
            reasons.append(f"report checkpoints not all sealed: {checkpoints}")
        elif not checkpoint_accepted:
            reasons.append("report checkpoints present but not accepted (no sha256 self-seal)")
    else:
        reasons.append("report_checkpoints missing")

    running = [cid for cid, r in cells.items() if r.get("status") == "running"] if isinstance(cells, dict) else []
    pid_path = cfg.campaign_root / "queue.pid.json"
    supervisor_alive = False
    if pid_path.exists():
        try:
            pid_doc = read_json_safe(pid_path)
            pid = pid_doc.get("pid")
            if isinstance(pid, int):
                supervisor_alive = _pid_alive(pid)
        except EcoError:
            supervisor_alive = False
    quiescent = not running and not supervisor_alive
    if running:
        reasons.append(f"{len(running)} cells still running")
    if supervisor_alive:
        reasons.append("campaign supervisor pid is still alive")

    # operator supersession signature (out-of-band; we only read + verify it)
    if cfg.signature_path.exists():
        try:
            sig = read_json_safe(cfg.signature_path)
            signed = _verify_signature(sig, cfg.expected_plan_sha256)
            if not signed:
                reasons.append("supersession signature present but invalid or wrong plan")
        except EcoError as exc:
            reasons.append(f"supersession signature unreadable: {exc}")
    else:
        reasons.append("no operator supersession signature on disk")

    all_pass = (plan_bound and terminal and reporter_sealed and checkpoint_accepted
                and quiescent and signed)
    return {"all_pass": all_pass, "plan_bound": plan_bound, "terminal": terminal,
            "reporter_sealed": reporter_sealed, "checkpoint_accepted": checkpoint_accepted,
            "quiescent": quiescent, "signed": signed, "reasons": reasons}


def _checkpoint_ok(value: Any) -> bool:
    """A reporter checkpoint is accepted only if it is a sha256 string or a dict carrying a
    sha256 self-seal (checkpoint_sha256 / state_sha256). An empty dict is NOT acceptance."""
    if isinstance(value, str):
        return is_sha256(value)
    if isinstance(value, dict):
        return any(is_sha256(value.get(k)) for k in ("checkpoint_sha256", "state_sha256",
                                                      "report_checkpoint_sha256"))
    return False


def _verify_signature(sig: dict[str, Any], expected_plan: str) -> bool:
    if sig.get("schema") != SIGNATURE_SCHEMA:
        return False
    if sig.get("plan_sha256") != expected_plan:
        return False
    return sealed(sig, "signature_sha256")


def make_signature(plan_sha256: str, *, signed_by: str, statement: str) -> dict[str, Any]:
    """Construct a supersession signature object. This is the OPERATOR's authorization,
    exposed as a library function for the operator/tests; the agent does not sign the live
    campaign on its own initiative.

    Note on the trust model: the self-seal proves INTEGRITY (the signature was not altered
    after creation and binds this exact plan_sha256), not cryptographic AUTHENTICITY. The
    authorization control is the operator placing this file at cfg.signature_path out of
    band; a future hardening would replace the self-seal with a real detached signature over
    an operator key. A forged signature still cannot activate a running campaign because the
    other four gate conditions require genuine terminal + reporter-sealed + quiescent state.
    """
    if not is_sha256(plan_sha256):
        raise EcoError("plan_sha256 must be a sha256")
    sig = {
        "schema": SIGNATURE_SCHEMA,
        "plan_sha256": plan_sha256,
        "signed_by": signed_by,
        "signed_at": now_iso(),
        "statement": statement,
    }
    return seal_field(sig, "signature_sha256")


def activate(cfg: ActivationConfig, *, go: bool = False,
             artifacts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Atomically flip the ecosystem layer on. Fail-closed: refuses unless the gate passes
    and go is True."""
    gate = supersession_gate(cfg)
    if not gate["all_pass"] or not go:
        return {"activated": False, "refused": True, "gate": gate,
                "reason": "supersession gate not satisfied or --go not set" if gate["all_pass"]
                          else "supersession conditions unmet"}
    manifest = {
        "schema": SCHEMA_ACTIVATION_MANIFEST,
        "active": True,
        "campaign_plan_sha256": cfg.expected_plan_sha256,
        "activated_at": now_iso(),
        "gate": gate,
        "artifacts": artifacts or {},
    }
    manifest = seal_field(manifest, "manifest_sha256")
    # preserve the previous manifest for rollback
    if cfg.manifest_path.exists():
        prev = read_json_safe(cfg.manifest_path)
        atomic_write_json(cfg.previous_path, prev)
    atomic_write_json(cfg.manifest_path, manifest)
    return {"activated": True, "refused": False, "manifest_sha256": manifest["manifest_sha256"],
            "manifest_path": str(cfg.manifest_path)}


def rollback(cfg: ActivationConfig) -> dict[str, Any]:
    """Restore the previous manifest, or return to default-off if none."""
    if cfg.previous_path.exists():
        prev = read_json_safe(cfg.previous_path)
        atomic_write_json(cfg.manifest_path, prev)
        return {"rolled_back": True, "restored": "previous_manifest",
                "manifest_sha256": prev.get("manifest_sha256")}
    if cfg.manifest_path.exists():
        # revert to explicit default-off
        off = seal_field({"schema": SCHEMA_ACTIVATION_MANIFEST, "active": False,
                          "campaign_plan_sha256": cfg.expected_plan_sha256,
                          "deactivated_at": now_iso()}, "manifest_sha256")
        atomic_write_json(cfg.manifest_path, off)
        return {"rolled_back": True, "restored": "default_off"}
    return {"rolled_back": False, "reason": "already default-off (no manifest)"}


def status(cfg: ActivationConfig) -> dict[str, Any]:
    active = False
    manifest_sha = None
    if cfg.manifest_path.exists():
        try:
            m = read_json_safe(cfg.manifest_path)
            active = bool(m.get("active"))
            manifest_sha = m.get("manifest_sha256")
        except EcoError:
            active = False
    gate = supersession_gate(cfg)
    return {
        "schema": SCHEMA_ACTIVATION,
        "active": active,
        "manifest_sha256": manifest_sha,
        "default_off": not active,
        "gate": gate,
        "can_activate_now": gate["all_pass"],
        "checked_at": now_iso(),
    }


def selftest() -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        croot = Path(d) / "doctor_v5_ultra"
        sroot = Path(d) / "frontier_eco"
        (croot).mkdir(parents=True)
        plan_sha = "a" * 64

        def write_queue(all_terminal: bool, sealed_reports: bool):
            cells = {"c1": {"status": "complete"},
                     "c2": {"status": "complete" if all_terminal else "running"}}
            checkpoints = {"sub-120B": ("b" * 64) if sealed_reports else None,
                           "120B": ("c" * 64) if sealed_reports else None}
            atomic_write_json(croot / "queue_state.json",
                              {"plan_sha256": plan_sha, "cells": cells,
                               "report_checkpoints": checkpoints})

        cfg = ActivationConfig(campaign_root=croot, state_root=sroot, expected_plan_sha256=plan_sha)

        # 1. running campaign, no signature -> gate fails, activation refused
        write_queue(all_terminal=False, sealed_reports=False)
        g1 = supersession_gate(cfg)
        if g1["all_pass"]:
            raise EcoError("gate must fail on a running campaign")
        r1 = activate(cfg, go=True)
        if r1["activated"]:
            raise EcoError("activation must refuse on a running campaign")

        # 2. terminal + sealed but still no signature -> refused
        write_queue(all_terminal=True, sealed_reports=True)
        g2 = supersession_gate(cfg)
        if g2["signed"] or g2["all_pass"]:
            raise EcoError("gate must fail without a signature")

        # 3. add a valid operator signature -> gate passes, activate with go
        sig = make_signature(plan_sha, signed_by="operator", statement="supersede after release")
        atomic_write_json(cfg.signature_path, sig)
        g3 = supersession_gate(cfg)
        if not g3["all_pass"]:
            raise EcoError(f"gate should pass now: {g3['reasons']}")
        r3 = activate(cfg, go=True, artifacts={"plan": "x"})
        if not r3["activated"]:
            raise EcoError("activation should succeed when gate passes and go set")
        st = status(cfg)
        if not st["active"]:
            raise EcoError("status should report active")

        # 4. activate without go -> refused even when gate passes
        r4 = activate(cfg, go=False)
        if r4["activated"]:
            raise EcoError("activation must refuse without go")

        # 5. rollback restores default-off / previous
        rb = rollback(cfg)
        if not rb["rolled_back"]:
            raise EcoError("rollback should succeed")

        # 6. wrong plan signature rejected
        bad = make_signature("d" * 64, signed_by="x", statement="wrong")
        if _verify_signature(bad, plan_sha):
            raise EcoError("signature for wrong plan must be rejected")

    return {"ok": True, "running_refused": True, "unsigned_refused": True,
            "signed_go_activates": True, "no_go_refused": True, "rollback_ok": True,
            "wrong_plan_rejected": True}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Fail-closed ecosystem activation gate.")
    ap.add_argument("--campaign-root", default=None)
    ap.add_argument("--state-root", default=None)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status")
    sub.add_parser("gate")
    act = sub.add_parser("activate"); act.add_argument("--go", action="store_true")
    sub.add_parser("rollback")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True)); sys.exit(0)
    cfg = default_config(args.campaign_root, args.state_root)
    if args.cmd == "gate":
        print(json.dumps(supersession_gate(cfg), indent=2, sort_keys=True))
    elif args.cmd == "activate":
        print(json.dumps(activate(cfg, go=args.go), indent=2, sort_keys=True))
    elif args.cmd == "rollback":
        print(json.dumps(rollback(cfg), indent=2, sort_keys=True))
    else:
        print(json.dumps(status(cfg), indent=2, sort_keys=True))
