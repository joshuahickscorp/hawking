#!/usr/bin/env python3.12
"""Operational manager for the completion-based Kimi K2.6 Gravity chapter."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time
from typing import Any


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_long_run_manager as legacy  # noqa: E402


f1 = legacy.f1
REPO = Path(__file__).resolve().parents[2]
RUNTIME = legacy.RUNTIME
FLOOR_BYTES = 5 * 1024**3
POLICY_JSON = "KIMI_K26_DISK_POLICY.json"
STATUS_JSON = "KIMI_K26_FINAL_CHAPTER_STATUS.json"
STATUS_MD = "KIMI_K26_FINAL_CHAPTER_STATUS.md"
LEDGER = "KIMI_K26_FINAL_CHAPTER_LEDGER.jsonl"
PARALLEL_LEDGER = "KIMI_K26_PARALLEL_EXECUTION_LEDGER.jsonl"
TELEGRAM_RECEIPT = "KIMI_K26_FINAL_CHAPTER_TELEGRAM.json"
STATIC_PLIST = REPO / "deploy/launchd/com.hawking.kimi-k26-doctor-prime.plist"
INSTALLED_PLIST = Path.home() / "Library/LaunchAgents/com.hawking.kimi-k26-doctor-prime.plist"


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configured_floor(path: Path) -> int | None:
    with path.open("rb") as handle:
        value = plistlib.load(handle)
    raw = (value.get("EnvironmentVariables") or {}).get("KIMI_K26_DISK_FLOOR_BYTES")
    return int(raw) if raw is not None else None


def audit(repo: Path) -> dict[str, Any]:
    base = legacy.audit(repo, notify=False)
    repo_campaign = repo / "tools/kimi_k26_campaign.py"
    repo_doctor = repo / "tools/condense/kimi_k26_doctor_prime.py"
    runtime_campaign = RUNTIME / "kimi_k26_campaign.py"
    runtime_doctor = RUNTIME / "kimi_k26_doctor_prime.py"
    launch = subprocess.run(
        ["/bin/launchctl", "print", f"gui/{os.getuid()}/com.hawking.kimi-k26-doctor-prime"],
        text=True, capture_output=True, check=False,
    )
    enforcement = {
        "floor_bytes": FLOOR_BYTES,
        "strict_comparison": "free_disk_bytes > 5368709120",
        "repo_campaign_sha256": sha256_file(repo_campaign),
        "runtime_campaign_sha256": sha256_file(runtime_campaign),
        "campaign_runtime_matches_repo": sha256_file(repo_campaign) == sha256_file(runtime_campaign),
        "repo_doctor_sha256": sha256_file(repo_doctor),
        "runtime_doctor_sha256": sha256_file(runtime_doctor),
        "doctor_runtime_matches_repo": sha256_file(repo_doctor) == sha256_file(runtime_doctor),
        "static_launchd_floor_bytes": configured_floor(STATIC_PLIST),
        "installed_launchd_floor_bytes": configured_floor(INSTALLED_PLIST),
        "live_launchd_contains_floor": "5368709120" in launch.stdout,
        "manager_floor_bytes": legacy.MIN_FREE,
    }
    failures = list(base["failures"])
    if not all((
        enforcement["campaign_runtime_matches_repo"],
        enforcement["doctor_runtime_matches_repo"],
        enforcement["static_launchd_floor_bytes"] == FLOOR_BYTES,
        enforcement["installed_launchd_floor_bytes"] == FLOOR_BYTES,
        enforcement["live_launchd_contains_floor"],
        enforcement["manager_floor_bytes"] == FLOOR_BYTES,
    )):
        failures.append("DISK_POLICY_ENFORCEMENT_MISMATCH")
    if not base["resources"]["floor_green"]:
        failures.append("DISK_FLOOR_RED")
    return f1.seal({
        "schema": "hawking.kimi_k26.final_chapter_audit.v1",
        "status": "PASS" if not failures else "FAIL",
        "audited_at": f1.now(),
        "failures": sorted(set(failures)),
        "git": base["git"],
        "controller": base["controller"],
        "resources": base["resources"],
        "source": base["source"],
        "mop": base["mop"],
        "enforcement": enforcement,
        "base_audit_seal_sha256": base["seal_sha256"],
    })


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    f1.atomic_json(path, value)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def mirror_json(name: str, value: dict[str, Any]) -> None:
    for root in (REPO, RUNTIME):
        atomic_json(root / name, value)


def append_jsonl(name: str, record: dict[str, Any]) -> dict[str, Any]:
    value = f1.seal({key: item for key, item in record.items() if key != "seal_sha256"})
    line = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    for root in (REPO, RUNTIME):
        with (root / name).open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    return value


def append_ledger(record: dict[str, Any]) -> dict[str, Any]:
    return append_jsonl(LEDGER, {
        "schema": "hawking.kimi_k26.final_chapter_ledger.v1", **record,
    })


def append_parallel(record: dict[str, Any]) -> dict[str, Any]:
    return append_jsonl(PARALLEL_LEDGER, {
        "schema": "hawking.kimi_k26.parallel_execution.v1", **record,
    })


def policy_artifact(audit_value: dict[str, Any]) -> dict[str, Any]:
    return f1.seal({
        "schema": "hawking.kimi_k26.disk_policy.v1",
        "status": "PASS" if audit_value["status"] == "PASS" else "FAIL",
        "sealed_at": f1.now(),
        "hard_floor_bytes": FLOOR_BYTES,
        "hard_floor_gib": 5,
        "green_law": "free_disk_bytes > hard_floor_bytes",
        "atomic_write_law": (
            "admit write only when free_disk_bytes - exact_next_write_bytes > hard_floor_bytes"
        ),
        "at_or_below_action": [
            "finish or checkpoint current atomic write",
            "stop new write-heavy launches",
            "run dependency-safe allowlisted GC",
            "resume only after strict green audit",
        ],
        "protected": [
            "sole Kimi source", "MOP", "credentials", "unrelated user data",
            "current best artifact", "final laws", "reproducibility manifests",
            "code required for accepted results",
        ],
        "enforcement": audit_value["enforcement"],
        "rollback_state": {
            "superseded_policy_bytes": 82 * 1024**3,
            "superseded_report_commit": "2910050ea93bd79309ba2e1af0b059a509e98b67",
            "authoritative_policy_bytes": FLOOR_BYTES,
            "rollback_target": "5 GiB remains authoritative",
            "controller_restart_required_after_code_or_plist restoration": True,
            "never_delete_to_rollback": ["sole Kimi source", "MOP", "credentials"],
        },
        "audit_seal_sha256": audit_value["seal_sha256"],
    })


def status_markdown(value: dict[str, Any]) -> str:
    resource = value.get("resources", {})
    controller = value.get("controller", {})
    return "\n".join([
        "# Kimi K2.6 Gravity Final Chapter Status", "",
        f"- Status: **{value.get('status')}**",
        f"- Phase: `{value.get('phase')}`",
        f"- Started: `{value.get('started_at')}`",
        f"- Updated: `{value.get('updated_at')}`",
        f"- Current best: `{value.get('current_best_candidate')}` / `{value.get('current_best_bpw')}` BPW",
        f"- F2 promotable: `{value.get('f2_promotable')}`",
        f"- Experiments completed: `{value.get('experiments_completed', 0)}`",
        f"- Active heavy lane: `{value.get('active_heavy_lane') or 'none'}`",
        f"- Active light/CPU lanes: `{value.get('active_light_lanes', [])}`",
        f"- Next experiment: `{value.get('next_experiment')}`", "",
        "## Guards", "",
        f"- Disk floor/free/headroom: `{resource.get('disk_floor_bytes', 0)}` / `{resource.get('free_disk_bytes', 0)}` / `{resource.get('disk_headroom_bytes', 0)}` bytes",
        f"- Controller PID/heartbeat/lease: `{controller.get('pid')}` / `{controller.get('heartbeat_current')}` / `{controller.get('lease_matches')}`",
        f"- Source one-copy / MOP: `{value.get('one_copy')}` / `{value.get('mop_protected')}`",
        f"- Primary diagnosis: `{value.get('primary_causal_diagnosis')}`", "",
        "## Latest result", "", "```json",
        json.dumps(value.get("latest_result", {}), indent=2, sort_keys=True), "```", "",
    ])


def write_status(value: dict[str, Any]) -> dict[str, Any]:
    existing = {}
    path = REPO / STATUS_JSON
    if path.exists():
        existing = f1.read_json(path)
    started = value.get("started_at") or existing.get("started_at") or f1.now()
    merged = {
        **existing, **value,
        "schema": "hawking.kimi_k26.final_chapter_status.v1",
        "started_at": started, "updated_at": f1.now(),
        "wall_clock_seconds": (
            dt.datetime.now(dt.timezone.utc) - parse_time(started)
        ).total_seconds(),
    }
    sealed = f1.seal({key: item for key, item in merged.items() if key != "seal_sha256"})
    for root in (REPO, RUNTIME):
        atomic_json(root / STATUS_JSON, sealed)
        atomic_text(root / STATUS_MD, status_markdown(sealed))
    return sealed


def send_telegram(checkpoint: str, message: str) -> dict[str, Any]:
    receipt = legacy.telegram(REPO, checkpoint, message)
    mirror_json(TELEGRAM_RECEIPT, receipt)
    return receipt


def has_start_record() -> bool:
    path = REPO / LEDGER
    if not path.exists():
        return False
    return any(json.loads(line).get("event") == "FINAL_CHAPTER_START"
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def start(repo: Path) -> dict[str, Any]:
    audit_value = audit(repo)
    if audit_value["status"] != "PASS":
        raise RuntimeError(f"final chapter guard failed: {audit_value['failures']}")
    policy = policy_artifact(audit_value)
    mirror_json(POLICY_JSON, policy)
    if not has_start_record():
        append_ledger({
            "event": "FINAL_CHAPTER_START", "at": f1.now(),
            "hypothesis": (
                "A native nonlinear representation can reduce upstream trajectory error more "
                "efficiently than downstream linear repair under 0.98 complete BPW."
            ),
            "audit_seal_sha256": audit_value["seal_sha256"],
            "disk_policy_seal_sha256": policy["seal_sha256"],
            "decision": "START_N1_TO_N6_F0_F1_TOURNAMENT",
        })
        receipt = send_telegram(
            "gravity-final:start",
            ("[Kimi K2.6 Gravity closure] started\n"
             "hard floor: 5 GiB exact and green\n"
             f"free/headroom: {audit_value['resources']['free_disk_bytes']/1024**3:.2f} / "
             f"{audit_value['resources']['disk_headroom_bytes']/1024**3:.2f} GiB\n"
             f"controller: PID {audit_value['controller']['pid']}, heartbeat/lease healthy\n"
             "next: nonlinear representation F0/F1 tournament"),
        )
    else:
        receipt = f1.read_json(REPO / TELEGRAM_RECEIPT)
    status = write_status({
        "status": "MANAGING", "phase": "NONLINEAR_F0_F1_TOURNAMENT",
        "current_best_candidate": "P1_DUAL_PATH_RECOVERY_R16X2_LOCAL_F1_ONLY",
        "current_best_bpw": 0.9085909525553385, "f2_promotable": False,
        "primary_causal_diagnosis": "UPSTREAM_STATE_PRIMARY_ROUTE_SECONDARY_CONDITIONAL_AMPLIFIER",
        "experiments_completed": 0, "active_heavy_lane": None,
        "active_light_lanes": [], "next_experiment": "N1_TO_N6_DISJOINT_NONLINEAR_SCREEN",
        "controller": audit_value["controller"], "resources": audit_value["resources"],
        "one_copy": audit_value["source"]["one_copy"],
        "mop_protected": audit_value["mop"]["matches_baseline"],
        "disk_policy_seal_sha256": policy["seal_sha256"],
        "latest_result": {"event": "FINAL_CHAPTER_START",
                          "telegram_delivered": receipt.get("delivered")},
    })
    return {"audit": audit_value, "policy": policy, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "audit"))
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    result = start(repo) if args.command == "start" else audit(repo)
    print(json.dumps({
        "status": result["status"] if args.command == "audit" else result["status"]["status"],
        "seal_sha256": result["seal_sha256"] if args.command == "audit" else
        result["status"]["seal_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
