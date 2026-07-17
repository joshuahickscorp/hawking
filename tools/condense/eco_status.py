#!/usr/bin/env python3.12
"""Telegram status / ETA for the ecosystem frontier scaffold.

Reuses the campaign notifier's hardened send primitives (Keychain-backed token/chat,
`_telegram`) but keeps its OWN state store under `reports/condense/frontier_eco/status/`
so it never touches the campaign notifier's delivered-event ledger. The sender is
injectable, exactly as the campaign notifier does, so tests exercise composition and
idempotency without any network or Keychain access.

The status message reports, honestly and tersely (house style: no em/en dashes):
  - the supersession gate (terminal / reporter-sealed / quiescent / signed) and that
    activation stays BLOCKED until the signed release boundary;
  - campaign progress and a coarse ETA;
  - the adaptive planner summary (parents with evidence, provisional floors, the
    scaling-prior brackets for 72B / 120B as scheduling only).

Sending a message is gated behind an explicit --go on the CLI and is never performed
automatically.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, SCHEMA_STATUS, read_json_safe, atomic_write_json, hash_value, now_iso,
    eco_state_root,
)

TERMINAL_STATUSES = frozenset({"complete", "negative", "unsupported"})
MAX_MESSAGE_CHARS = 4000


@dataclasses.dataclass(frozen=True)
class StatusConfig:
    campaign_root: Path
    state_root: Path
    include_plan: bool = True

    @property
    def state_path(self) -> Path:
        return self.state_root / "status" / "state.json"


def default_config(campaign_root: str | os.PathLike[str] | None = None,
                   state_root: str | os.PathLike[str] | None = None) -> StatusConfig:
    from eco_common import repo_root
    croot = Path(campaign_root) if campaign_root else repo_root() / "reports" / "condense" / "doctor_v5_ultra"
    sroot = Path(state_root) if state_root else eco_state_root()
    return StatusConfig(campaign_root=croot, state_root=sroot)


def _default_sender(text: str) -> dict[str, Any]:
    """Send via the campaign notifier's hardened, Keychain-backed sender."""
    import importlib.util
    mod_path = Path(_HERE) / "doctor_v5_telegram_rung_notifier.py"
    spec = importlib.util.spec_from_file_location("doctor_v5_telegram_rung_notifier", mod_path)
    if spec is None or spec.loader is None:
        raise EcoError("cannot load the campaign notifier for sending")
    notifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notifier)
    return notifier._send(text)  # type: ignore[attr-defined]


def _campaign_progress(cfg: StatusConfig) -> dict[str, Any]:
    queue = read_json_safe(cfg.campaign_root / "queue_state.json")
    cells = queue.get("cells", {})
    if not isinstance(cells, dict):
        raise EcoError("queue_state.cells missing")
    total = len(cells)
    from collections import Counter
    tally = Counter(str(r.get("status")) for r in cells.values())  # str-coerce for JSON sort safety
    terminal = sum(tally.get(s, 0) for s in TERMINAL_STATUSES)
    running = [cid for cid, r in cells.items() if r.get("status") == "running"]
    checkpoints = queue.get("report_checkpoints", {})
    return {
        "total": total,
        "terminal": terminal,
        "progress_pct": round(100.0 * terminal / total, 1) if total else 0.0,
        "status_tally": dict(tally),
        "running": running,
        "reporter_sealed": isinstance(checkpoints, dict) and bool(checkpoints)
                           and all(v is not None for v in checkpoints.values()),
    }


def compose_status(cfg: StatusConfig) -> dict[str, Any]:
    """Build the status text + a machine summary. Read-only."""
    progress = _campaign_progress(cfg)

    gate_summary = None
    try:
        import eco_activation
        acfg = eco_activation.default_config(str(cfg.campaign_root), str(cfg.state_root))
        gate = eco_activation.supersession_gate(acfg)
        gate_summary = {k: gate[k] for k in ("terminal", "reporter_sealed", "quiescent",
                                             "signed", "all_pass")}
    except Exception as exc:  # noqa: BLE001 - status must be robust
        gate_summary = {"error": type(exc).__name__}

    plan_summary = None
    if cfg.include_plan:
        try:
            import eco_import, eco_planner
            icfg = eco_import.default_config(str(cfg.campaign_root))
            plan = eco_planner.build_plan(eco_import.build_ledger(icfg))
            plan_summary = {
                "parents_with_evidence": len(plan["parents"]),
                "scaling_trend": plan["scaling_prior"]["trend"],
                "awaiting": [{"label": a["model_label"],
                              "predicted_floor_bpw": a["predicted_bracket"].get("predicted_floor_bpw")}
                             for a in plan["parents_awaiting_evidence"]],
            }
        except Exception as exc:  # noqa: BLE001
            plan_summary = {"error": type(exc).__name__}

    lines = ["Hawking Condenser Ecosystem Frontier"]
    lines.append(f"campaign {progress['terminal']}/{progress['total']} terminal ({progress['progress_pct']}%)")
    if progress["running"]:
        lines.append(f"running: {progress['running'][0]}")
    if gate_summary and "error" not in gate_summary:
        lines.append("supersession gate: " + " ".join(
            f"{k}={'Y' if gate_summary[k] else 'N'}" for k in
            ("terminal", "reporter_sealed", "quiescent", "signed")))
        lines.append("activation: " + ("READY" if gate_summary["all_pass"]
                                        else "BLOCKED until signed release"))
    if plan_summary and "error" not in plan_summary:
        lines.append(f"planner: {plan_summary['parents_with_evidence']} parents with evidence, "
                     f"floor trend {plan_summary['scaling_trend']}")
        for a in plan_summary["awaiting"]:
            lines.append(f"  {a['label']} awaiting; scheduling-prior floor ~{a['predicted_floor_bpw']} bpw")
    text = "\n".join(lines)[:MAX_MESSAGE_CHARS]

    return {
        "schema": SCHEMA_STATUS,
        "text": text,
        "summary": {"progress": progress, "gate": gate_summary, "plan": plan_summary},
        "composed_at": now_iso(),
        "event_id": hash_value(text),
    }


def send_status(cfg: StatusConfig, *, sender: Callable[[str], dict[str, Any]] = _default_sender,
                force: bool = False) -> dict[str, Any]:
    """Compose + send. Idempotent: the same text is not resent unless force=True."""
    status = compose_status(cfg)
    state = {"delivered": {}}
    if cfg.state_path.exists():
        try:
            state = read_json_safe(cfg.state_path)
        except EcoError:
            state = {"delivered": {}}
    delivered = state.setdefault("delivered", {})
    if status["event_id"] in delivered and not force:
        return {"status": "already-sent", "event_id": status["event_id"], "sent": 0}
    receipt = sender(status["text"])
    delivered[status["event_id"]] = {"sent_at": now_iso(), "receipt": receipt}
    atomic_write_json(cfg.state_path, state)
    return {"status": "sent", "event_id": status["event_id"], "sent": 1, "receipt": receipt}


def selftest() -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        croot = Path(d) / "doctor_v5_ultra"
        sroot = Path(d) / "frontier_eco"
        croot.mkdir(parents=True)
        atomic_write_json(croot / "queue_state.json", {
            "plan_sha256": "a" * 64,
            "cells": {"c1": {"status": "complete"}, "c2": {"status": "running"},
                      "c3": {"status": "pending"}},
            "report_checkpoints": {"sub-120B": None, "120B": None},
        })
        cfg = StatusConfig(campaign_root=croot, state_root=sroot, include_plan=False)
        st = compose_status(cfg)
        if "BLOCKED until signed release" not in st["text"]:
            raise EcoError("status must show activation blocked")
        sent_texts: list[str] = []

        def fake_sender(text: str) -> dict[str, Any]:
            sent_texts.append(text)
            return {"message_id": len(sent_texts), "sent_at": now_iso()}

        r1 = send_status(cfg, sender=fake_sender)
        r2 = send_status(cfg, sender=fake_sender)  # idempotent
        if r1["sent"] != 1 or r2["sent"] != 0 or len(sent_texts) != 1:
            raise EcoError(f"idempotency failed: {r1} {r2} {sent_texts}")
        r3 = send_status(cfg, sender=fake_sender, force=True)
        if r3["sent"] != 1 or len(sent_texts) != 2:
            raise EcoError("force resend failed")
    return {"ok": True, "activation_blocked_shown": True, "idempotent": True, "force_resend": True}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Ecosystem frontier Telegram status/ETA.")
    ap.add_argument("--campaign-root", default=None)
    ap.add_argument("--state-root", default=None)
    ap.add_argument("--no-plan", action="store_true")
    ap.add_argument("--send", action="store_true", help="send to Telegram")
    ap.add_argument("--go", action="store_true", help="required with --send to actually send")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True)); sys.exit(0)
    cfg = default_config(args.campaign_root, args.state_root)
    cfg = dataclasses.replace(cfg, include_plan=not args.no_plan)
    if args.send and args.go:
        print(json.dumps(send_status(cfg, force=args.force), indent=2, sort_keys=True))
    else:
        st = compose_status(cfg)
        print(st["text"])
        print("\n--- (dry run; pass --send --go to deliver) ---")
