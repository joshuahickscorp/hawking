#!/usr/bin/env python3.12
"""Successor E0 audit capturer: signed, read-only live-state packets.

Writes the Phase E0 packets required by the master goal (section 4) under a
successor-only namespace, each self-sealed with the campaign's canonical hashing form.
Read-only: it never touches campaign-owned state and never prints secrets.

Packets:
  reports/condense/event_horizon_successor/audit/live_state.json
  reports/condense/event_horizon_successor/audit/process_tree.json
  reports/condense/event_horizon_successor/audit/resource_state.json
  reports/condense/event_horizon_successor/audit/git_state.json
  reports/condense/event_horizon_successor/audit/readiness_matrix.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import seal_field, now_iso, repo_root, read_json_safe, atomic_write_json  # noqa: E402

CAMPAIGN_ROOT_DEFAULT = "/Users/scammermike/Downloads/hawking/reports/condense/doctor_v5_ultra"
AUDIT_DIR = "reports/condense/event_horizon_successor/audit"
TERMINAL = frozenset({"complete", "negative", "unsupported"})


def _run(cmd: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def git_state(main_repo: str) -> dict[str, Any]:
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=main_repo)
    head = _run(["git", "rev-parse", "HEAD"], cwd=main_repo)
    worktrees = _run(["git", "worktree", "list"], cwd=main_repo).splitlines()
    dirty = _run(["git", "status", "--porcelain"], cwd=main_repo).splitlines()
    return seal_field({
        "schema": "hawking.successor.audit.git_state.v1",
        "captured_at": now_iso(),
        "main_repo": main_repo,
        "branch": branch,
        "head": head,
        "worktrees": worktrees,
        "dirty_count": len(dirty),
        "remotes": _run(["git", "remote", "-v"], cwd=main_repo).splitlines()[:4],
    }, "packet_sha256")


def process_tree() -> dict[str, Any]:
    out = _run(["ps", "-Ao", "pid,ppid,pgid,rss,%cpu,etime,command"])
    rows = []
    for line in out.splitlines()[1:]:
        low = line.lower()
        if any(k in low for k in ("quantize-model-block-parallel", "doctor_v5_disk25_successor",
                                  "doctor_v5_strand_ladder", "doctor_v5_ultra_queue",
                                  "doctor_v5_campaign_report")) and "grep" not in low:
            parts = line.split(None, 6)
            if len(parts) >= 7:
                rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "pgid": int(parts[2]),
                             "rss_kb": int(parts[3]), "cpu_pct": float(parts[4]),
                             "etime": parts[5], "command": parts[6][:400]})
    supervisor_alive = any(_pid_alive(r["pid"]) for r in rows if "disk25_successor" in r["command"])
    return seal_field({
        "schema": "hawking.successor.audit.process_tree.v1",
        "captured_at": now_iso(),
        "campaign_processes": rows,
        "campaign_supervisor_alive": supervisor_alive,
        "note": "read-only ps snapshot; the successor never signals or adopts these pids",
    }, "packet_sha256")


def resource_state(main_repo: str) -> dict[str, Any]:
    df = _run(["df", "-k", main_repo]).splitlines()
    avail_gb = None
    if len(df) >= 2:
        f = df[-1].split()
        if len(f) >= 4:
            try:
                avail_gb = round(int(f[3]) / (1024 * 1024), 1)
            except ValueError:
                pass
    swap = _run(["sysctl", "-n", "vm.swapusage"])
    power = "AC" if "AC Power" in _run(["pmset", "-g", "batt"]) else "battery-or-unknown"
    thermal_ok = "No thermal warning" in _run(["pmset", "-g", "therm"]) or not _run(["pmset", "-g", "therm"])
    return seal_field({
        "schema": "hawking.successor.audit.resource_state.v1",
        "captured_at": now_iso(),
        "free_disk_gb": avail_gb,
        "swap_usage": swap,
        "power": power,
        "thermal_ok": thermal_ok,
        "cpu_count": os.cpu_count(),
        "note": "campaign saturates CPU; successor work stays light and defers heavy compiles to CI",
    }, "packet_sha256")


def live_state(campaign_root: str) -> dict[str, Any]:
    root = Path(campaign_root)
    qs = read_json_safe(root / "queue_state.json")
    cells = qs.get("cells", {})
    from collections import Counter
    tally = dict(Counter(str(r.get("status")) for r in cells.values()))
    terminal = sum(tally.get(s, 0) for s in TERMINAL)
    running = [cid for cid, r in cells.items() if r.get("status") == "running"]
    return seal_field({
        "schema": "hawking.successor.audit.live_state.v1",
        "captured_at": now_iso(),
        "campaign_root": campaign_root,
        "plan_sha256": qs.get("plan_sha256"),
        "control_mode": qs.get("control_mode"),
        "campaign_status": qs.get("status"),
        "supervisor_pid": qs.get("supervisor_pid"),
        "cell_total": len(cells),
        "status_tally": tally,
        "terminal_count": terminal,
        "running_cells": running,
        "report_checkpoints": qs.get("report_checkpoints"),
        "released_state": "State_B_legacy_running" if running or terminal < len(cells) else "State_A_released",
    }, "packet_sha256")


def readiness_matrix() -> dict[str, Any]:
    """The corrected scaffold + adapter truth matrix (from the E0 deep audit)."""
    return seal_field({
        "schema": "hawking.successor.audit.readiness_matrix.v1",
        "captured_at": now_iso(),
        "eco_surfaces": {
            "eco_common": "implemented_executable",
            "eco_passport": "implemented_executable",
            "eco_import": "implemented_read_only",
            "eco_planner": "implemented_planning_only",
            "eco_admission": "implemented_planning_only (adapter readiness was id-map; superseded by succ_admission source-bound probe)",
            "eco_pipeline": "declared_not_enforced (validators superseded by succ_state enforced gates)",
            "eco_activation": "implemented_executable (superseded/extended by succ_transition)",
            "eco_status": "implemented_executable (extended by succ_telegram service)",
            "eco_cli": "implemented_executable",
        },
        "adapters": {
            "doctor-v5-strand-ladder-qwen25-dense": {
                "ready_for_execution": True,
                "claim_restricted": True,
                "labels": ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"],
                "executable_treatments": ["none"],
                "unsupported_treatments": ["lora_kd", "blockwise_qat", "strand_hessian"],
                "quality_eval_resident_labels": ["0.5B", "1.5B", "3B", "7B", "14B"],
                "quality_eval_deferred": ["32B", "72B"],
            },
            "doctor-v5-strand-ladder-gpt-oss-moe": {
                "ready_for_execution": False,
                "adapter_version": "0.1-contract",
                "run_refuses_exit": 78,
                "blockers": ["source_to_str2_conversion", "reassembly_provenance",
                             "apple_silicon_moe_str2_loader", "tokenizer_template", "evaluator",
                             "native_load_parity", "disk_infeasible_183GB", "human_review_signoff"],
            },
        },
        "ci_state": {
            "rust_check_job": {"red": True, "root_cause": "cargo fmt --check (158 files, mechanical)",
                               "fix": "cargo fmt (write) + commit; clippy/build/test need heavy compile to confirm"},
            "frontend_job": {"red": True, "root_cause": "app/pnpm-workspace.yaml missing packages: key under pnpm 9",
                             "fix": "delete app/pnpm-workspace.yaml (single-package app)"},
            "pre_existing_on_main": True,
        },
        "frontier_600b_1_1t": {
            "recommended_row": {"label": "671B", "hf_id": "deepseek-ai/DeepSeek-V3", "total_b": 671.0,
                                "active_b": 37.0, "regime": "MOE-PAGED", "staged": False,
                                "blockers": ["waiting_source_authority", "waiting_adapter(deepseek-moe)",
                                             "waiting_disk(1342GB_dl_vs_176GB_free)"]},
            "one_t_class_alt": {"label": "Kimi-K2.6", "hf_id": "moonshotai/Kimi-K2.6", "total_b": 1100.0},
        },
        "discovery_hints": {
            "retry_ceiling_disposition_reconciler.py": "campaign_owned_gitignored_runtime; 72B boundary watcher; must_not_import",
            "doctor_v5_disk25_child_gate_successor.py": "campaign_owned_gitignored_runtime; staged v4 25GB-reserve successor; must_not_import",
        },
    }, "packet_sha256")


def capture_all(campaign_root: str, out_dir: Path) -> dict[str, Any]:
    main_repo = "/Users/scammermike/Downloads/hawking"
    packets = {
        "live_state.json": live_state(campaign_root),
        "process_tree.json": process_tree(),
        "resource_state.json": resource_state(main_repo),
        "git_state.json": git_state(main_repo),
        "readiness_matrix.json": readiness_matrix(),
    }
    for name, packet in packets.items():
        atomic_write_json(out_dir / name, packet)
    return {"written": list(packets), "out_dir": str(out_dir),
            "live_released_state": packets["live_state.json"]["released_state"],
            "campaign_supervisor_alive": packets["process_tree.json"]["campaign_supervisor_alive"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Successor E0 audit capturer (read-only).")
    ap.add_argument("--campaign-root", default=CAMPAIGN_ROOT_DEFAULT)
    ap.add_argument("--out-dir", default=str(repo_root() / AUDIT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(capture_all(args.campaign_root, out), indent=2, sort_keys=True))
