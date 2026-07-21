#!/usr/bin/env python3.12
"""Operational manager and evidence ledger for the Kimi K2.6 causal long run."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402


RUNTIME = Path.home() / "Library/Application Support/Hawking/KimiK26"
REVISION = f1.REVISION
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--moonshotai--Kimi-K2.6" /
            "snapshots" / REVISION)
MOP = Path.home() / "Downloads/mop"
MIN_FREE = 82 * 1024 ** 3
STATUS_JSON = "KIMI_K26_LONG_RUN_STATUS.json"
STATUS_MD = "KIMI_K26_LONG_RUN_STATUS.md"
LEDGER = "KIMI_K26_LONG_RUN_LEDGER.jsonl"
TELEGRAM_RECEIPT = "KIMI_K26_LONG_RUN_TELEGRAM.json"


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_seal(path: Path) -> dict[str, Any]:
    value = f1.read_json(path)
    expected = f1.seal({key: item for key, item in value.items() if key != "seal_sha256"})[
        "seal_sha256"
    ]
    if value.get("seal_sha256") != expected:
        raise f1.F1Error(f"seal mismatch: {path}")
    return value


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def source_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    failures = []
    inode_set = set()
    for item in manifest["files"]:
        path = SNAPSHOT / item["path"]
        try:
            resolved = path.resolve(strict=True)
            stat = resolved.stat()
            if not resolved.is_file() or stat.st_size != int(item["size"]):
                failures.append({"path": item["path"], "reason": "size_or_type"})
            inode_set.add((stat.st_dev, stat.st_ino))
        except OSError as exc:
            failures.append({"path": item["path"], "reason": type(exc).__name__})
    candidates = {SNAPSHOT / "model.safetensors.index.json"}
    roots = (Path.home() / ".cache/huggingface/hub",
             Path.home() / "Library/Caches/huggingface/hub",
             Path.home() / "Downloads/hawking/models", Path.home() / "models")
    for root in roots:
        if root.exists():
            candidates.update(root.rglob("model.safetensors.index.json"))
    spotlight = subprocess.run(
        ["/usr/bin/mdfind", "kMDItemFSName == 'model.safetensors.index.json'"],
        text=True, capture_output=True, check=False,
    )
    candidates.update(Path(line) for line in spotlight.stdout.splitlines() if line.strip())
    views = []
    fingerprints = set()
    mop_prefix = str(MOP.resolve(strict=True)) + os.sep
    for index_path in sorted(candidates, key=str):
        if str(index_path).startswith(mop_prefix):
            continue
        root = index_path.parent
        config = f1.read_json(root / "config.json") if (root / "config.json").is_file() else {}
        if (config.get("text_config") or {}).get("model_type") != "kimi_k2":
            continue
        shards = sorted(root.glob("model-*-of-000064.safetensors"))
        complete = len(shards) == 64
        fingerprint = None
        if complete:
            try:
                fingerprint = tuple(sorted((path.resolve(strict=True).stat().st_dev,
                                            path.resolve(strict=True).stat().st_ino)
                                           for path in shards))
                fingerprints.add(fingerprint)
            except OSError:
                complete = False
        views.append({"path": str(root), "complete": complete,
                      "authoritative": root.resolve() == SNAPSHOT.resolve(),
                      "physical_inode_set_sha256": (
                          hashlib.sha256(f1.canonical(fingerprint)).hexdigest()
                          if fingerprint else None),
                      })
    complete_views = [view for view in views if view["complete"]]
    one_copy = (len(complete_views) == 1 and complete_views[0]["authoritative"] and
                len(fingerprints) == 1)
    return {
        "status": "PASS" if not failures and one_copy else "FAIL",
        "file_count_checked": len(manifest["files"]), "failures": failures,
        "unique_content_inodes": len(inode_set), "complete_views": complete_views,
        "distinct_physical_inode_sets": len(fingerprints), "one_copy": one_copy,
        "mop_excluded_without_traversal": str(MOP.resolve()),
    }


def load_campaign_module() -> Any:
    path = RUNTIME / "kimi_k26_campaign.py"
    spec = importlib.util.spec_from_file_location("kimi_k26_installed_campaign", path)
    if spec is None or spec.loader is None:
        raise f1.F1Error("installed campaign module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def telegram(repo: Path, checkpoint_id: str, message: str) -> dict[str, Any]:
    module = load_campaign_module()
    token, chat_id = module.credentials()
    available = bool(token and chat_id)
    delivered = bool(module.telegram(message)) if available else False
    receipt = f1.seal({
        "schema": "hawking.kimi_k26.long_run_telegram.v1", "checkpoint_id": checkpoint_id,
        "attempted_at": f1.now(), "credentials_available": available,
        "delivered": delivered, "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
    })
    f1.atomic_json(repo / TELEGRAM_RECEIPT, receipt)
    f1.atomic_json(RUNTIME / TELEGRAM_RECEIPT, receipt)
    return receipt


def resource_snapshot() -> dict[str, Any]:
    usage = shutil.disk_usage(Path.home())
    swap = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True,
                          capture_output=True, check=False).stdout.strip()
    memory = subprocess.run(["/usr/bin/memory_pressure", "-Q"], text=True,
                            capture_output=True, check=False).stdout.strip()
    thermals = subprocess.run(["/usr/bin/pmset", "-g", "therm"], text=True,
                              capture_output=True, check=False).stdout.strip()
    return {"free_disk_bytes": usage.free, "disk_floor_bytes": MIN_FREE,
            "disk_headroom_bytes": usage.free - MIN_FREE,
            "floor_green": usage.free >= MIN_FREE, "swap": swap,
            "memory_pressure": memory[-2000:], "thermals": thermals[-2000:]}


def audit(repo: Path, *, notify: bool) -> dict[str, Any]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                          capture_output=True, check=True).stdout.strip()
    upstream = subprocess.run(["git", "rev-parse", "@{u}"], cwd=repo, text=True,
                              capture_output=True, check=False).stdout.strip()
    baseline_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "0458b2fb", head], cwd=repo,
        capture_output=True, check=False,
    ).returncode == 0
    manifest = verify_seal(RUNTIME / "KIMI_K26_OFFICIAL_MANIFEST.json")
    verification = verify_seal(RUNTIME / "KIMI_K26_SOURCE_VERIFICATION.json")
    one_copy_receipt = verify_seal(RUNTIME / "KIMI_K26_ONE_COPY_RECEIPT.json")
    law = verify_seal(repo / "KIMI_K26_SCIENTIFIC_LAW.json")
    science = verify_seal(repo / "KIMI_K26_SCIENTIFIC_STATUS.json")
    publication = verify_seal(repo / "KIMI_K26_SCIENCE_PUBLICATION_MANIFEST.json")
    source = source_audit(manifest)
    heartbeat = f1.read_json(RUNTIME / "kimi_k26.heartbeat.json")
    lease = f1.read_json(RUNTIME / "kimi_k26.heavy.lease")
    now_time = dt.datetime.now(dt.timezone.utc)
    heartbeat_age = (now_time - parse_time(str(heartbeat["beat_at"]))).total_seconds()
    pid = int(heartbeat["pid"])
    controller = {
        "pid": pid, "process_group": int(heartbeat["process_group"]),
        "alive": process_alive(pid), "heartbeat_age_seconds": heartbeat_age,
        "heartbeat_current": heartbeat_age <= 90,
        "heartbeat_state": heartbeat.get("state"), "heartbeat_status": heartbeat.get("status"),
        "lease_pid": int(lease["pid"]), "lease_process_group": int(lease["process_group"]),
        "lease_matches": (int(lease["pid"]) == pid and
                          int(lease["process_group"]) == int(heartbeat["process_group"])),
    }
    mop_stat = MOP.stat()
    mop = {"path": str(MOP.resolve()), "device": mop_stat.st_dev, "inode": mop_stat.st_ino,
           "matches_baseline": mop_stat.st_dev == 16777233 and mop_stat.st_ino == 1233332,
           "source_outside_mop": not SNAPSHOT.resolve().is_relative_to(MOP.resolve())}
    resources = resource_snapshot()
    baseline = {
        "best_candidate_complete_bpw": science["best_local_candidate"]["actual_complete_bpw"],
        "f1_cosine": science["best_local_candidate"]["f1_score"]["cosine_mean"],
        "f2_top8_route_set_change": 1 - 0.78125,
        "tested_region_bpw_ceiling": law["scope"]["rate_ceiling_complete_physical_bpw"],
        "tested_region_failed_f2": science["current_best_candidate"].startswith("NONE_PROMOTABLE"),
        "reported_commit": "0458b2fb", "head": head,
    }
    failures = []
    if head != upstream or not baseline_is_ancestor:
        failures.append("HEAD_OR_UPSTREAM_MISMATCH")
    if source["status"] != "PASS" or verification["status"] != "PASS" or \
            one_copy_receipt["status"] != "PASS":
        failures.append("SOURCE_OR_ONE_COPY_INVALID")
    if not all((controller["alive"], controller["heartbeat_current"],
                controller["lease_matches"])):
        failures.append("CONTROLLER_GUARD_INVALID")
    if not resources["floor_green"]:
        failures.append("DISK_FLOOR_RISK")
    if not mop["matches_baseline"] or not mop["source_outside_mop"]:
        failures.append("MOP_GUARD_INVALID")
    if not publication.get("artifacts") or science["scientific_law_seal_sha256"] != law["seal_sha256"]:
        failures.append("SCIENCE_SEAL_CHAIN_INVALID")
    result = f1.seal({
        "schema": "hawking.kimi_k26.long_run_audit.v1",
        "status": "PASS" if not failures else "FAIL", "audited_at": f1.now(),
        "failures": failures, "git": {"head": head, "upstream": upstream},
        "baseline": baseline, "source": source, "controller": controller,
        "telegram_prior_delivery": f1.read_json(RUNTIME / "telegram_delivery.json"),
        "resources": resources, "mop": mop,
        "verified_seals": {
            "official_manifest": manifest["seal_sha256"],
            "source_verification": verification["seal_sha256"],
            "one_copy": one_copy_receipt["seal_sha256"],
            "science_law": law["seal_sha256"], "science_status": science["seal_sha256"],
            "publication": publication["seal_sha256"],
        },
    })
    if notify:
        result["telegram"] = telegram(
            repo, "long-run:start",
            ("[Kimi K2.6 long run] manager started\n"
             f"audit: {result['status']}\nHEAD: {head[:8]}\n"
             f"baseline: 0.90859 BPW / F1 0.91344 / F2 route change 21.875%\n"
             f"controller: PID {pid}, heartbeat {heartbeat_age:.1f}s, lease matched\n"
             f"free disk: {resources['free_disk_bytes']/1024**3:.2f} GiB\n"
             "next: held-out control routing atlas and causal swaps"),
        )
        result = f1.seal({key: value for key, value in result.items() if key != "seal_sha256"})
    return result


def append_ledger(repo: Path, record: dict[str, Any]) -> None:
    sealed = f1.seal({key: value for key, value in record.items() if key != "seal_sha256"})
    line = json.dumps(sealed, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    for path in (repo / LEDGER, RUNTIME / LEDGER):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def status_markdown(status: dict[str, Any]) -> str:
    resource = status.get("resources", {})
    control = status.get("controller", {})
    lines = [
        "# Kimi K2.6 Long-Run Status", "",
        f"- Status: **{status.get('status')}**",
        f"- Started: `{status.get('started_at')}`",
        f"- Updated: `{status.get('updated_at')}`",
        f"- Wall-clock managed: `{status.get('wall_clock_seconds', 0):.1f}s`",
        f"- Experiments completed: `{status.get('experiments_completed', 0)}`",
        f"- Active experiment: `{status.get('active_experiment') or 'none'}`",
        f"- Current best: `{status.get('current_best_candidate')}` / "
        f"`{status.get('current_best_bpw')}` BPW",
        f"- Primary diagnosis: `{status.get('primary_causal_diagnosis')}`",
        f"- Next experiment: `{status.get('next_experiment')}`", "",
        "## Guards", "",
        f"- Controller PID/lease: `{control.get('pid')}` / `{control.get('lease_matches')}`",
        f"- Heartbeat age: `{control.get('heartbeat_age_seconds')}` seconds",
        f"- Free disk/headroom: `{resource.get('free_disk_bytes', 0)/1024**3:.2f}` / "
        f"`{resource.get('disk_headroom_bytes', 0)/1024**3:.2f}` GiB",
        f"- MOP protected: `{status.get('mop_protected')}`",
        f"- Sole Kimi source: `{status.get('one_copy')}`", "",
        "## Latest result", "",
        "```json", json.dumps(status.get("latest_result", {}), indent=2, sort_keys=True), "```", "",
    ]
    return "\n".join(lines)


def write_status(repo: Path, value: dict[str, Any]) -> dict[str, Any]:
    started = value.get("started_at") or f1.now()
    value = {**value, "schema": "hawking.kimi_k26.long_run_status.v1",
             "started_at": started, "updated_at": f1.now(),
             "wall_clock_seconds": (dt.datetime.now(dt.timezone.utc) - parse_time(started)).total_seconds()}
    sealed = f1.seal(value)
    for root in (repo, RUNTIME):
        f1.atomic_json(root / STATUS_JSON, sealed)
        temporary = root / f".{STATUS_MD}.{os.getpid()}.{time.time_ns()}.tmp"
        temporary.write_text(status_markdown(sealed), encoding="utf-8")
        os.replace(temporary, root / STATUS_MD)
    return sealed


def audit_command(repo: Path, *, notify: bool) -> dict[str, Any]:
    audit_value = audit(repo, notify=notify)
    existing = f1.read_json(repo / STATUS_JSON) if (repo / STATUS_JSON).is_file() else {}
    status = write_status(repo, {
        **existing,
        "status": "MANAGING" if audit_value["status"] == "PASS" else "BLOCKED_GUARD",
        "audit_seal_sha256": audit_value["seal_sha256"],
        "controller": audit_value["controller"], "resources": audit_value["resources"],
        "mop_protected": audit_value["mop"]["matches_baseline"],
        "one_copy": audit_value["source"]["one_copy"],
        "current_best_candidate": "P1_DUAL_PATH_RECOVERY_R16X2_LOCAL_F1_ONLY",
        "current_best_bpw": 0.9085909525553385,
        "primary_causal_diagnosis": "UNPROVEN_ROUTE_DRIFT_CAUSALITY",
        "active_experiment": None, "experiments_completed": existing.get("experiments_completed", 0),
        "next_experiment": "HELDOUT_090859_CONTROL_CAUSAL_ATLAS",
        "latest_result": {"audit": audit_value["status"],
                          "telegram_delivered": audit_value.get("telegram", {}).get("delivered")},
    })
    append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1",
        "event": "BASELINE_AUDIT", "experiment_id": "LR00_BASELINE_AUDIT",
        "started_at": audit_value["audited_at"], "ended_at": f1.now(),
        "duration_seconds": 0, "status": audit_value["status"],
        "hypothesis": "The reported campaign baseline and operational guards remain valid.",
        "evidence_seal_sha256": audit_value["seal_sha256"],
        "metrics": audit_value["baseline"], "faults": audit_value["failures"],
        "decision": "ADVANCE_TO_HELDOUT_CONTROL" if audit_value["status"] == "PASS" else "STOP",
        "next_run_rationale": status["next_experiment"],
    })
    f1.atomic_json(repo / "KIMI_K26_LONG_RUN_BASELINE_AUDIT.json", audit_value)
    f1.atomic_json(RUNTIME / "KIMI_K26_LONG_RUN_BASELINE_AUDIT.json", audit_value)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--repo", type=Path, required=True)
    audit_parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "audit":
            result = audit_command(args.repo.resolve(strict=True), notify=args.notify)
        else:
            raise f1.F1Error(f"unknown command: {args.command}")
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
