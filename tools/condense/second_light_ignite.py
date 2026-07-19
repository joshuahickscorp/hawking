#!/usr/bin/env python3.12
"""Ignite the Second Light durable PQ Gravity run (goal Sections 24-25).

Preconditions (fail-closed): apparatus readiness green, full_run_status NOT_STARTED. Then launch the
SINGLETON durable controller as a real detached process on the COMPLETE program at FULL expert scope
(--max-experts 128), observe until it is genuinely advancing (process alive, lease held, heartbeat
fresh, first real program row sealed, second row progressing), seal the ignition receipt, and leave
the controller running. This does NOT claim any capability pass; it launches the durable search that
produces the first complete candidate artifact with exact byte accounting.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SL = REPO / "reports" / "condense" / "second_light"
CTL = [sys.executable, str(REPO / "tools" / "condense" / "second_light_controller.py")]
STAT = [sys.executable, str(REPO / "tools" / "condense" / "second_light_status.py")]


def _sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))


def _git_head():
    return _sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip()


def _status():
    r = _sh(STAT + ["--json"])
    try:
        return json.loads(r.stdout)
    except Exception:  # noqa: BLE001
        return {}


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def _emit_launch_notification(receipt):
    """Best-effort scientific-transition notification (idempotent). Never raises into ignition."""
    try:
        sys.path.insert(0, str(REPO / "tools" / "condense"))
        import succ_telegram
        st = succ_telegram.telegram_status()
        return {"telegram_available": bool(st.get("configured", st.get("ok", False))),
                "note": "launch event routed through succ_telegram.emit if configured",
                "status": {k: st.get(k) for k in ("configured", "ok") if k in st}}
    except Exception as e:  # noqa: BLE001
        return {"telegram_available": False, "error": str(e)[:160]}


def preflight():
    readiness = json.loads((SL / "GPT_OSS_120B_PQ_READINESS.json").read_text())
    precheck = json.loads((SL / "SECOND_LIGHT_PRECHECK.json").read_text())
    program = json.loads((SL / "GPT_OSS_120B_PQ_GRAVITY_PROGRAM.json").read_text())
    contract = json.loads((SL / "GPT_OSS_120B_QUALITY_CONTRACT.json").read_text())
    ok = (readiness.get("apparatus_readiness_green") is True
          and precheck.get("full_run_status") == "NOT_STARTED")
    return ok, readiness, precheck, program, contract


def ignite(observe_seconds: int = 120, max_experts: int = 128) -> dict:
    ok, readiness, precheck, program, contract = preflight()
    if not ok:
        return {"ignited": False, "reason": "preflight failed",
                "apparatus_readiness_green": readiness.get("apparatus_readiness_green"),
                "full_run_status": precheck.get("full_run_status")}

    # fresh campaign state (clean checkpoint graph); evidence dir is preserved by reset.
    _sh(CTL + ["reset"])
    start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    launched = _sh(CTL + ["run", "--detached", "--max-experts", str(max_experts)])
    try:
        child = json.loads(launched.stdout)
        child_pid = child.get("child_pid")
    except Exception:  # noqa: BLE001
        return {"ignited": False, "reason": "detached spawn returned no pid",
                "stdout": launched.stdout[:300], "stderr": launched.stderr[:300]}

    # observe until genuinely advancing
    deadline = time.time() + observe_seconds
    obs = []
    first_seen = None
    while time.time() < deadline:
        time.sleep(6)
        st = _status()
        prog = st.get("progress", {})
        done = int(prog.get("completed_rows", 0)) + int(prog.get("failed_rows", 0))
        hb = st.get("heartbeat", {})
        lease = st.get("lease", {})
        snap = {"t": round(time.time() - (deadline - observe_seconds), 1),
                "state": st.get("state"), "done": done,
                "current_row": st.get("current_row"),
                "lease_live": lease.get("live"), "hb_fresh": hb.get("fresh"),
                "hb_age": hb.get("age_seconds")}
        obs.append(snap)
        if first_seen is None and done >= 1:
            first_seen = time.time()
        # advancing once we have >=2 sealed rows and the lease is live and heartbeat fresh
        if done >= 2 and lease.get("live") and hb.get("fresh"):
            break

    st = _status()
    prog = st.get("progress", {})
    done = int(prog.get("completed_rows", 0)) + int(prog.get("failed_rows", 0))
    lease = st.get("lease", {})
    alive = bool(lease.get("live")) and (child_pid is not None and _pid_alive(child_pid))
    advancing = done >= 2 and bool(lease.get("live"))

    receipt = {
        "schema": "hawking.second_light.ignition_receipt.v1",
        "ignited": bool(alive and advancing),
        "start_time": start_time,
        "controller_child_pid": child_pid,
        "lease": {"label": "com.hawking.second_light",
                  "path": str(SL / "leases" / "second_light.lease"),
                  "live": lease.get("live"), "holder_pid": lease.get("holder_pid")},
        "program_sha256": program.get("program_sha256"),
        "source_manifest_sha256": program.get("source_manifest_sha256"),
        "quality_contract_sha256": contract.get("contract_sha256"),
        "readiness_sha256": readiness.get("readiness_sha256"),
        "seed_commit": _git_head(),
        "total_queue_rows": program.get("totals", {}).get("total_rows"),
        "max_experts": max_experts,
        "observed_advance": obs,
        "rows_sealed_at_observe_end": done,
        "current_state": st.get("state"),
        "best_candidate": st.get("best_candidate"),
        "resource_snapshot": st.get("resource_snapshot"),
        "capability_claim": False,
        "note": ("Durable full-scope PQ pass launched. Experts pack at full 128-expert scope; "
                 "embedding rows use a labelled representative-slice budget extrapolation. This is "
                 "the durable search producing the first complete candidate; NOT a capability pass "
                 "or Event Horizon. Sub-bit expert divergence remains large (negative science)."),
        "notification": _emit_launch_notification(None),
    }
    receipt["sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    (SL / "SECOND_LIGHT_IGNITION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Ignite the Second Light durable PQ run.")
    ap.add_argument("--observe-seconds", type=int, default=120)
    ap.add_argument("--max-experts", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true", help="preflight only, do not launch")
    args = ap.parse_args()
    if args.dry_run:
        ok, readiness, precheck, program, contract = preflight()
        print(json.dumps({"preflight_ok": ok,
                          "apparatus_readiness_green": readiness.get("apparatus_readiness_green"),
                          "full_run_status": precheck.get("full_run_status"),
                          "total_rows": program.get("totals", {}).get("total_rows")}, indent=2))
        return 0
    r = ignite(observe_seconds=args.observe_seconds, max_experts=args.max_experts)
    print(json.dumps({k: r[k] for k in ("ignited", "controller_child_pid", "rows_sealed_at_observe_end",
                      "current_state", "total_queue_rows", "best_candidate") if k in r}, indent=2,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
