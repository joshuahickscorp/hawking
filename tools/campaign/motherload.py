#!/usr/bin/env python3.12
"""Live-state audit + gate ledger for the Motherload completion campaign.

Two jobs, deliberately in one file because they share the same authority:

  audit   observe the machine -- git, detached processes, source coverage,
          artifact coverage, host resources, MOP isolation -- and write
          HAWKING_MOTHERLOAD_LIVE_AUDIT.json.
  gate    record a terminal-gate transition in an append-only JSONL ledger and
          rebuild HAWKING_MOTHERLOAD_STATUS.{json,md} from it.

Everything the audit reports is measured at call time. Nothing is carried in
memory across runs, so a crash costs at most the row being written, and a stale
claim cannot survive a re-run: the campaign's own rule is that live process and
artifact state overrides stale prose, including this file's previous output.

    python3.12 tools/campaign/motherload.py audit
    python3.12 tools/campaign/motherload.py gate M04 IN_PROGRESS "note"
    python3.12 tools/campaign/motherload.py status
    python3.12 tools/campaign/motherload.py selftest
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "HAWKING_MOTHERLOAD_LEDGER.jsonl"
STATUS_JSON = ROOT / "HAWKING_MOTHERLOAD_STATUS.json"
STATUS_MD = ROOT / "HAWKING_MOTHERLOAD_STATUS.md"
AUDIT_JSON = ROOT / "HAWKING_MOTHERLOAD_LIVE_AUDIT.json"

APPSUP = Path.home() / "Library/Application Support/Hawking"
GLM = APPSUP / "GLM52Gravity"
FETCH = GLM / "source_fetch"

# Terminal gates, verbatim from HAWKING_MOTHERLOAD_COMPLETION_CAMPAIGN.md §12.
GATES = {
    "M01": "GLM traversal is 282/282",
    "M02": "complete local GLM .gravity artifact exists",
    "M03": "lowest broad parity rate is sealed, sub-bit preferred and H15 maximum",
    "M04": "full GLM token executes from .gravity",
    "M05": "measured base TPS and prefill exist",
    "M06": "acceleration stack is terminal and measured separately",
    "M07": "GLM runs end to end inside HIDE",
    "M08": "Prometheus S0 and source decision are sealed",
    "M09": "Prometheus architecture and profiles are implemented",
    "M10": "equal-budget Claim A is sealed",
    "M11": "General and Math artifacts are selected and verified",
    "M12": "Forge, continuity, sovereignty, and Limit Registry are sealed",
    "M13": "Odyssey substrate and training bundle are complete",
    "M14": "sandbox, roles, Ledger, verifiers, Tribunal, and retrieval are scaffolded",
    "M15": "Lean/Mathlib and evidence environment are pinned",
    "M16": "Odyssey dry-run validation passes",
    "M17": "ODYSSEY_LAUNCH_AUTHORIZED remains false",
    "M18": "rollback/source lifecycle is green",
    "M19": "all campaign commits are pushed",
    "M20": "worktree and process state are clean except intentional detached services",
}
STATES = {"OPEN", "IN_PROGRESS", "BLOCKED", "GREEN", "NEGATIVE_SEALED"}
CLOSED = {"GREEN", "NEGATIVE_SEALED"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sh(*args: str) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------- audit


def _git() -> dict:
    # `_sh` strips, which would eat the leading status column of the first
    # porcelain line; split off the status code by whitespace instead of by
    # a fixed offset so an unstaged and a staged entry parse the same.
    dirty = [l for l in _sh("git", "status", "--porcelain").splitlines() if l.strip()]
    return {
        "branch": _sh("git", "branch", "--show-current"),
        "head": _sh("git", "rev-parse", "HEAD"),
        "head_subject": _sh("git", "log", "-1", "--format=%s"),
        "dirty_paths": [l.split(maxsplit=1)[-1] for l in dirty],
        "dirty_count": len(dirty),
        "upstream": _sh("git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"),
        "unpushed": len([l for l in _sh("git", "log", "--oneline", "@{u}..HEAD").splitlines()
                         if l.strip()]) if _sh("git", "rev-parse", "--abbrev-ref",
                                               "--symbolic-full-name", "@{u}") else None,
    }


def _host() -> dict:
    usage = shutil.disk_usage(str(Path.home()))
    vm = _sh("sysctl", "-n", "hw.memsize")
    swap = _sh("sysctl", "-n", "vm.swapusage")
    return {
        "free_disk_bytes": usage.free,
        "free_disk_gib": round(usage.free / 2**30, 1),
        "physical_memory_bytes": int(vm) if vm.isdigit() else None,
        "swap_raw": swap,
        "cpu_count": os.cpu_count(),
    }


def _processes() -> dict:
    """Detached Hawking lanes, and the MOP lanes we must not disturb."""
    ps = _sh("/bin/ps", "-Ao", "pid=,pgid=,command=")
    hawking, mop = [], 0
    for line in ps.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, pgid, cmd = parts
        if "mop." in cmd or "mop-temporal" in cmd:
            mop += 1
        elif "hawking" in cmd and "Downloads/hawking" in cmd and "motherload.py" not in cmd:
            hawking.append({"pid": int(pid), "pgid": int(pgid), "command": cmd[:180]})
    return {
        "hawking_lanes": hawking,
        "hawking_lane_count": len(hawking),
        "mop_process_count": mop,
        "mop_isolation": "UNTOUCHED -- separate project, not read or signalled by this campaign",
    }


def _source() -> dict:
    """GLM source traversal state, read from the controller's own files."""
    progress = _json(FETCH / "progress.json") or {}
    safe = _json(FETCH / "GLM52_SAFE_TO_LEAVE_STATUS.json") or {}
    ctrl = safe.get("controller", {})
    pids = [int(p) for p in ctrl.get("pids", [])]
    resident = sorted(p.name for p in GLM.joinpath("source").glob("model-*.safetensors")) \
        if GLM.joinpath("source").is_dir() else []
    return {
        "repo": progress.get("repo"),
        "revision": progress.get("revision"),
        "state": progress.get("state"),
        "total_source_shards": progress.get("total_source_shards"),
        "shards_verified": progress.get("shards_verified_total"),
        "shards_probed": progress.get("shards_probed_total"),
        "fraction_verified": progress.get("source_fraction_verified"),
        "window": progress.get("window"),
        "window_count": progress.get("window_count"),
        "resident_shard_files": len(resident),
        "resident_bytes": progress.get("resident_bytes"),
        "controller": {
            "launchd_label": ctrl.get("launchd_label"),
            "launchd_loaded": ctrl.get("launchd_loaded"),
            "lease_held": ctrl.get("lease_held"),
            "lease_path": ctrl.get("lease_path"),
            "pids": pids,
            "pids_alive": {str(p): _alive(p) for p in pids},
            "heartbeat_age_seconds": ctrl.get("heartbeat_age_seconds"),
            "eviction_paused": ctrl.get("eviction_paused"),
        },
        "teacher_capsules": (safe.get("teacher") or {}).get("capsule_count"),
        "faults_total": safe.get("faults_total"),
        "healthy": progress.get("state") == "RUNNING" and any(_alive(p) for p in pids),
    }


def _artifacts() -> dict:
    """Every .gravity artifact under Application Support, with its own header's claims."""
    out = []
    if APPSUP.is_dir():
        sys.path.insert(0, str(ROOT / "tools/condense"))
        try:
            import gravity_format as gravity  # noqa: PLC0415
        except ImportError:
            gravity = None
        for p in sorted(APPSUP.rglob("*.gravity")):
            row = {"path": str(p), "bytes": p.stat().st_size}
            if gravity is not None:
                try:
                    h = gravity.read_header(p)
                    row["tensor_count"] = len(h.get("tensors", []))
                    row["complete_bpw"] = (h.get("compression") or {}).get("complete_bpw")
                    row["model"] = (h.get("model") or {}).get("repo")
                    row["revision"] = (h.get("model") or {}).get("revision")
                    row["body_sha256"] = (h.get("integrity") or {}).get("body_sha256")
                except Exception as exc:  # noqa: BLE001 - a bad header is a finding
                    row["header_error"] = f"{type(exc).__name__}: {exc}"
            out.append(row)
    return {"gravity_artifacts": out, "count": len(out)}


def audit() -> dict:
    a = {
        "schema": "hawking.motherload.live_audit.v1",
        "at": _now(),
        "root": str(ROOT),
        "git": _git(),
        "host": _host(),
        "processes": _processes(),
        "source": _source(),
        "artifacts": _artifacts(),
    }
    AUDIT_JSON.write_text(json.dumps(a, indent=1, sort_keys=True) + "\n")
    return a


# ---------------------------------------------------------------- gates


def _append(row: dict) -> None:
    row["at"] = _now()
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _derive() -> dict:
    gates = {g: {"desc": d, "state": "OPEN", "note": "", "at": None}
             for g, d in GATES.items()}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "gate" and row["gate"] in gates:
                gates[row["gate"]].update(state=row["state"], note=row.get("note", ""),
                                          at=row["at"])
    closed = sum(1 for g in gates.values() if g["state"] in CLOSED)
    return {
        "schema": "hawking.motherload.status.v1",
        "updated_at": _now(),
        "gates_total": len(GATES),
        "gates_closed": closed,
        "completion_fraction": round(closed / len(GATES), 4),
        "gates": gates,
        "odyssey_launch_authorized": False,
        "endpoint": "HAWKING_ODYSSEY_READY" if closed == len(GATES) else "IN_PROGRESS",
    }


def _markdown(status: dict) -> str:
    mark = {"GREEN": "green", "NEGATIVE_SEALED": "sealed-negative", "IN_PROGRESS": "running",
            "BLOCKED": "blocked", "OPEN": "open"}
    lines = [
        "# Hawking Motherload Completion Status",
        "",
        f"endpoint: `{status['endpoint']}`  ",
        f"gates closed: {status['gates_closed']}/{status['gates_total']}  ",
        f"ODYSSEY_LAUNCH_AUTHORIZED: `{str(status['odyssey_launch_authorized']).lower()}`  ",
        f"updated: {status['updated_at']}",
        "",
        "| gate | state | condition | note |",
        "|---|---|---|---|",
    ]
    for gid in sorted(status["gates"]):
        g = status["gates"][gid]
        note = (g["note"] or "").replace("|", "\\|")
        lines.append(f"| {gid} | {mark.get(g['state'], g['state'])} | {g['desc']} | {note} |")
    lines.append("")
    return "\n".join(lines)


def seal() -> dict:
    status = _derive()
    STATUS_JSON.write_text(json.dumps(status, indent=1, sort_keys=True) + "\n")
    STATUS_MD.write_text(_markdown(status))
    return status


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "audit":
        print(json.dumps(audit(), indent=1, sort_keys=True))
        return 0
    if cmd == "gate":
        gate, state = argv[2], argv[3].upper()
        if gate not in GATES:
            raise SystemExit(f"unknown gate {gate}; known: {sorted(GATES)}")
        if state not in STATES:
            raise SystemExit(f"unknown state {state}; known: {sorted(STATES)}")
        _append({"kind": "gate", "gate": gate, "state": state, "note": " ".join(argv[4:])})
    elif cmd != "status":
        raise SystemExit(__doc__)
    print(json.dumps(seal(), indent=1, sort_keys=True))
    return 0


def selftest() -> None:
    """Round-trip the ledger in a temp root, and check the audit observes this process."""
    import tempfile
    global LEDGER, STATUS_JSON, STATUS_MD, AUDIT_JSON
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        LEDGER, STATUS_JSON, STATUS_MD = t / "l.jsonl", t / "s.json", t / "s.md"
        AUDIT_JSON = t / "a.json"

        s = seal()
        assert s["gates_closed"] == 0 and s["endpoint"] == "IN_PROGRESS", s
        assert len(s["gates"]) == 20, "the campaign declares 20 terminal gates"

        _append({"kind": "gate", "gate": "M04", "state": "GREEN", "note": "x"})
        _append({"kind": "gate", "gate": "M04", "state": "BLOCKED", "note": "y"})
        s = seal()
        assert s["gates"]["M04"]["state"] == "BLOCKED", "last write must win"
        assert s["gates_closed"] == 0, s
        assert "| M04 | blocked |" in STATUS_MD.read_text()

        # Every gate closed must flip the endpoint, and only then.
        for g in GATES:
            _append({"kind": "gate", "gate": g, "state": "GREEN", "note": ""})
        s = seal()
        assert s["endpoint"] == "HAWKING_ODYSSEY_READY", s["endpoint"]
        assert s["odyssey_launch_authorized"] is False, "the fence is not a gate"

        a = audit()
        assert a["git"]["head"], "audit must observe a real HEAD"
        assert _alive(os.getpid()) and not _alive(2**22), "liveness probe is backwards"
        assert AUDIT_JSON.is_file()
        print("selftest PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        raise SystemExit(main(sys.argv))
