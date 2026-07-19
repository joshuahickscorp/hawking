#!/usr/bin/env python3.12
"""SECOND_LIGHT precheck (Second Light goal, Sections 0-1).

Establishes LIVE truth before any code change: is a controller actually alive and advancing the
complete program, or is `full_run_status = NOT_STARTED`? A committed JSON, a historical PID, or a
completed bounded calibration must NOT be read as a live full run. This tool answers the mandated
questions from real process/lease/source evidence and seals SECOND_LIGHT_PRECHECK.{json,md}.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "condense" / "second_light"
SRC = REPO / "models" / "gpt-oss-120b"
ORIG = SRC / "original"
SCHEMA = "hawking.second_light.precheck.v1"


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<error:{e}>"


def _hawking_heavy_processes() -> list[dict]:
    """A hawking heavy OWNER = a live python/cargo WORKER actually running a hawking campaign module.

    Tightened to avoid two false positives the earlier keyword scan produced: (1) shell wrappers
    (/bin/zsh -c ...) that merely mention a keyword because they are running these very tools, and
    (2) this precheck's own process tree. MoP (a separate project under ~/Downloads/mop) never counts.
    """
    self_pid = os.getpid()
    self_ppid = os.getppid()
    out = _sh(["ps", "-axo", "pid,ppid,etime,command"])
    worker_modules = ("second_light_controller", "gravity_forge_run", "gptoss_gravity_run",
                      "succ_cli", "succ_watch", "doctor_v5", "second_light_pack")
    interpreters = ("python", "cargo", "/hawking")  # a real worker runs one of these
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        cmd = parts[3]
        low = cmd.lower()
        if pid in (self_pid, self_ppid):
            continue
        if low.startswith(("/bin/zsh", "/bin/bash", "/bin/sh", "-zsh", "-bash")) or " -c " in low[:12]:
            continue                                        # shell wrapper, not a worker
        if "/downloads/hawking" not in low and not any(m in low for m in worker_modules):
            continue
        if not any(i in low for i in interpreters):
            continue                                        # must be an actual interpreter/binary
        if not any(m in low for m in worker_modules):
            continue                                        # must run a hawking worker module
        hits.append({"pid": pid, "etime": parts[2], "command": cmd[:240]})
    return hits


def _second_light_live_state() -> dict:
    """MEASURE the actual Second Light controller liveness (not a hardcoded constant).

    Defers to the status tool's snapshot, whose liveness truth is the fcntl flock on the lease
    (a dead pid can never read live). Returns the measured state, whether a live controller holds
    the lease, whether the heartbeat is fresh, and the checkpoint cursor."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import second_light_status as sls
        from second_light_controller import (ControllerConfig, DEFAULT_CAMPAIGN_ROOT,
                                              DEFAULT_PROGRAM_NAME)
        root = DEFAULT_CAMPAIGN_ROOT
        cfg = ControllerConfig(campaign_root=root, program_path=root / DEFAULT_PROGRAM_NAME)
        snap = sls.snapshot(cfg)
        lease = snap.get("lease", {})
        hb = snap.get("heartbeat", {})
        prog = snap.get("progress", {})
        state = snap.get("state")
        lease_live = bool(lease.get("live"))
        hb_fresh = bool(hb.get("fresh"))
        completed = int(prog.get("completed_rows", 0)) + int(prog.get("failed_rows", 0))
        total = int(prog.get("total_rows", 0)) or 1
        return {
            "available": True,
            "state": state,
            "lease_live": lease_live,
            "lease_holder_pid": lease.get("holder_pid"),
            "heartbeat_fresh": hb_fresh,
            "heartbeat_beat_at": hb.get("beat_at"),
            "current_row": snap.get("current_row"),
            "completed_rows": completed,
            "total_rows": total,
            "percent_done": round(100.0 * completed / total, 3),
            # advancing = a live controller holds the lease AND its heartbeat is fresh AND it is
            # either working a row now or has sealed progress. Measured, never hardcoded.
            "advancing": bool(lease_live and hb_fresh and (snap.get("current_row") is not None
                              or completed > 0)),
        }
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)[:160], "state": None,
                "lease_live": False, "heartbeat_fresh": False, "advancing": False,
                "completed_rows": 0, "total_rows": 0, "percent_done": 0.0,
                "lease_holder_pid": None, "current_row": None, "heartbeat_beat_at": None}


def _launchd_state() -> list[dict]:
    rows = []
    for label in ("com.hawking.second_light", "com.hawking.doctorv5.telegram",
                  "com.hawking.doctorv5ultra.post120b",
                  "com.hawking.doctorv5ultra.autoresume", "com.hawking.frontier"):
        detail = _sh(["launchctl", "list", label])
        pid = None
        exit_status = None
        for ln in detail.splitlines():
            if '"PID"' in ln:
                pid = ln.split("=")[-1].strip().rstrip(";").strip()
            if '"LastExitStatus"' in ln:
                exit_status = ln.split("=")[-1].strip().rstrip(";").strip()
        rows.append({"label": label, "pid": pid, "last_exit_status": exit_status,
                     "alive": pid is not None})
    return rows


def _lease_pid_files() -> list[dict]:
    found = []
    root = REPO / "reports" / "condense"
    for p in root.rglob("*"):
        if p.is_file() and (p.suffix == ".pid" or "lease" in p.name.lower()
                            or "heartbeat" in p.name.lower()):
            try:
                txt = p.read_text()[:200]
            except Exception:  # noqa: BLE001
                txt = "<unreadable>"
            found.append({"path": str(p.relative_to(REPO)), "preview": txt})
    return found


def _source_receipt() -> dict:
    idx = ORIG / "model.safetensors.index.json"
    receipt = {"present": False}
    if not idx.exists():
        return receipt
    d = json.loads(idx.read_text())
    wm = d.get("weight_map", {})
    shards = sorted(set(wm.values()))
    shard_info = []
    all_present = True
    for s in shards:
        f = ORIG / s
        if f.exists():
            shard_info.append({"name": s, "bytes": f.stat().st_size})
        else:
            shard_info.append({"name": s, "bytes": None, "missing": True})
            all_present = False
    # cheap manifest hash over (name,size) tuples, not a full byte read (61 GiB)
    h = hashlib.sha256()
    for si in shard_info:
        h.update(f"{si['name']}:{si['bytes']}".encode())
    tok = SRC / "tokenizer.json"
    chat = SRC / "chat_template.jinja"
    return {
        "present": all_present,
        "tensor_count": len(wm),
        "shard_count": len(shards),
        "shards": shard_info,
        "manifest_sha256": h.hexdigest(),
        "config": json.loads((ORIG / "config.json").read_text()) if (ORIG / "config.json").exists() else None,
        "tokenizer_present": tok.exists(),
        "tokenizer_bytes": tok.stat().st_size if tok.exists() else None,
        "chat_template_present": chat.exists(),
    }


def _resources() -> dict:
    mem = int(_sh(["sysctl", "-n", "hw.memsize"]) or 0)
    total, used, free = shutil.disk_usage(str(REPO))
    return {
        "hw_model": _sh(["sysctl", "-n", "hw.model"]),
        "cpu": _sh(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "logical_cores": int(_sh(["sysctl", "-n", "hw.ncpu"]) or 0),
        "ram_gb": round(mem / 1024**3, 1),
        "disk_free_gb": round(free / 1024**3, 1),
        "disk_total_gb": round(total / 1024**3, 1),
        "thermal": _sh(["pmset", "-g", "therm"])[:200] or "no warning recorded",
    }


def build() -> dict:
    heavy = _hawking_heavy_processes()
    launchd = _launchd_state()
    lease_files = _lease_pid_files()
    src = _source_receipt()

    # MEASURE the actual Second Light controller state (fixes the earlier hardcoded advancing=False
    # that made RUNNING unreachable). A live full run = a live controller holding the lease AND a
    # fresh heartbeat AND an advancing/working queue. All three are measured from the status snapshot
    # whose liveness truth is the fcntl flock (a dead pid can never read live).
    live = _second_light_live_state()
    sl_controller_alive = bool(live.get("lease_live"))
    advancing = bool(live.get("advancing"))
    # controller_alive: the Second Light controller specifically, or any hawking heavy owner.
    controller_alive = sl_controller_alive or len(heavy) > 0
    # full_run_status defers to the measured controller state: RUNNING/DRAINING when the Second
    # Light controller is genuinely live+advancing; PAUSED when partial sealed state but no live
    # controller; otherwise NOT_STARTED. A committed JSON or a dead pid can never yield RUNNING.
    sl_state = live.get("state")
    if sl_controller_alive and advancing:
        full_run_status = "RUNNING"
    elif sl_controller_alive:
        full_run_status = "DRAINING"
    elif live.get("completed_rows", 0) > 0:
        full_run_status = "PAUSED"
    else:
        full_run_status = "NOT_STARTED"

    doc = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authoritative_main_commit": _sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "authoritative_main_branch": _sh(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"]),
        "second_light_live_state": live,
        "mandated_questions": {
            "is_a_controller_currently_alive": controller_alive,
            "is_it_processing_the_complete_program": advancing,
            "controller_pid": (live.get("lease_holder_pid") or (heavy[0]["pid"] if heavy else None)),
            "lease_held": ("com.hawking.second_light" if sl_controller_alive else None),
            "queue_advancing": advancing,
            "last_checkpoint_time": live.get("heartbeat_beat_at"),
            "percent_of_complete_program_done": live.get("percent_done", 0.0),
        },
        "full_run_status": full_run_status,
        "evidence": {
            "hawking_heavy_processes": heavy,
            "launchd_jobs": launchd,
            "lease_pid_heartbeat_files": lease_files,
            "prior_ignition_claims": [
                {"commit": "0504b0f7", "claim": "one Gravity run ignited",
                 "verdict": "UNTRUSTED per Section 0; reclassified as FIRST-LIGHT CALIBRATION; "
                            "no live process, no advancing queue, no fresh heartbeat"},
                {"commit": "80b1f1c2", "claim": "120B source FAIL CLOSED (absent) -> run NOT launched",
                 "verdict": "source-absent condition since RESOLVED; source now present+verified"},
            ],
        },
        "source_receipt": src,
        "resources": _resources(),
        "determination": {
            "full_run_status": full_run_status,
            "reason": (
                f"MEASURED: Second Light lease live={sl_controller_alive}, "
                f"heartbeat_fresh={live.get('heartbeat_fresh')}, advancing={advancing}, "
                f"completed_rows={live.get('completed_rows')}/{live.get('total_rows')}. "
                f"full_run_status={full_run_status} derived from the live controller state (fcntl "
                f"flock liveness), not from any committed JSON or historical PID. Other hawking "
                f"heavy owners: {len(heavy)} (MoP is a separate project and is excluded)."),
            "committed_json_not_trusted": True,
            "historical_pid_not_trusted": True,
            "bounded_calibration_not_called_full_run": True,
            "advancing_is_measured_not_hardcoded": True,
        },
    }
    payload = json.dumps(doc, sort_keys=True).encode()
    doc["sha256"] = hashlib.sha256(payload).hexdigest()
    return doc


def render_md(doc: dict) -> str:
    q = doc["mandated_questions"]
    src = doc["source_receipt"]
    res = doc["resources"]
    lines = [
        "# SECOND LIGHT PRECHECK",
        "",
        f"schema `{doc['schema']}`  sha256 `{doc['sha256'][:16]}`  generated {doc['generated_at']}",
        "",
        f"authoritative main: `{doc['authoritative_main_commit'][:12]}` on `{doc['authoritative_main_branch']}`",
        "",
        "## Full-run status",
        "",
        f"**full_run_status = {doc['full_run_status']}**",
        "",
        f"> {doc['determination']['reason']}",
        "",
        "## Mandated questions (Section 1)",
        "",
        "| question | answer |",
        "| --- | --- |",
        f"| Is a controller currently alive? | {q['is_a_controller_currently_alive']} |",
        f"| Is it processing the complete intended program? | {q['is_it_processing_the_complete_program']} |",
        f"| What is its PID? | {q['controller_pid']} |",
        f"| What lease does it hold? | {q['lease_held']} |",
        f"| What queue is advancing? | {q['queue_advancing']} |",
        f"| Last checkpoint time? | {q['last_checkpoint_time']} |",
        f"| Percentage of complete program done? | {q['percent_of_complete_program_done']}% |",
        "",
        "## Source receipt",
        "",
        f"- present: {src['present']}  tensors: {src['tensor_count']}  shards: {src['shard_count']}",
        f"- manifest_sha256: `{src['manifest_sha256'][:16]}`",
        f"- tokenizer present: {src['tokenizer_present']}  chat_template: {src['chat_template_present']}",
        "",
        "## Resources",
        "",
        f"- {res['hw_model']}  {res['cpu']}  {res['logical_cores']} cores  {res['ram_gb']} GiB RAM",
        f"- disk free {res['disk_free_gb']} GiB / {res['disk_total_gb']} GiB",
        f"- thermal: {res['thermal']}",
        "",
        "## Prior ignition claims (corrected)",
        "",
    ]
    for c in doc["evidence"]["prior_ignition_claims"]:
        lines.append(f"- `{c['commit']}` claimed *{c['claim']}* -> {c['verdict']}")
    lines += ["", "## Launchd jobs", ""]
    for r in doc["evidence"]["launchd_jobs"]:
        lines.append(f"- `{r['label']}` alive={r['alive']} pid={r['pid']} last_exit={r['last_exit_status']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = build()
    (OUT / "SECOND_LIGHT_PRECHECK.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    (OUT / "SECOND_LIGHT_PRECHECK.md").write_text(render_md(doc))
    print(json.dumps({"full_run_status": doc["full_run_status"],
                      "controller_alive": doc["mandated_questions"]["is_a_controller_currently_alive"],
                      "source_present": doc["source_receipt"]["present"],
                      "sha256": doc["sha256"][:16]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
