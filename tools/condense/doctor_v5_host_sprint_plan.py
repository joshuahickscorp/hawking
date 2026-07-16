#!/usr/bin/env python3.12
"""Read-only host-sprint probe and reversible, proposal-only isolation plan.

No command in this module changes fan controls, QoS, nice values, Spotlight,
backup exclusions, launch services, or runtime defaults.  ``probe`` performs
read-only inspection; ``stage`` writes a default-off plan below the inert Doctor
staging root.  Any future application requires explicit user authorization and
must record both the action and its reversal.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import doctor_v5_aggressive_admission_policy as aggressive
import doctor_v5_local_observer as local_observer
import ram_scheduler


ULTRA_ROOT = ROOT / "reports/condense/doctor_v5_ultra"
STAGE_ROOT = ULTRA_ROOT / "staged_acceleration/host_sprint_v1"
ELASTIC_STAGE_ROOT = ULTRA_ROOT / "staged_acceleration/elastic_v1"
DEFAULT_PLAN = STAGE_ROOT / "host_sprint_plan.json"
PLAN_SCHEMA = "hawking.doctor_v5_host_sprint_plan.v1"
PROBE_SCHEMA = "hawking.doctor_v5_host_sprint_probe.v1"
GATE_SCHEMA = "hawking.doctor_v5_host_sprint_gate.v1"
OWNER_SNAPSHOT_SCHEMA = "hawking.doctor_v5_host_owner_lease_snapshot.v1"
OWNER_LEASE_SCHEMA = "hawking.doctor_v5_elastic_owner_lease.v1"
SWAP_DECISION_SCHEMA = "hawking.doctor_v5_elastic_swap_decision_binding.v1"
VERSION = "2026-07-14.1"
MAX_CURRENT_EVIDENCE_AGE_SECONDS = 60.0
SHA256_LENGTH = 64
OWNER_ROLES = frozenset({"prepare", "encoder", "finalizer", "companion"})
REVIEWED_TOPOLOGY = {
    "physical_cores": 28,
    "logical_cores": 28,
    "performance_cores": 20,
    "efficiency_cores": 8,
}
TOPOLOGY_SYSCTLS = {
    "physical_cores": "hw.physicalcpu",
    "logical_cores": "hw.logicalcpu",
    "performance_cores": "hw.perflevel0.physicalcpu",
    "efficiency_cores": "hw.perflevel1.physicalcpu",
}


class HostSprintError(RuntimeError):
    """The host sprint probe/plan is invalid or unsafe."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == SHA256_LENGTH \
        and all(character in "0123456789abcdef" for character in value)


def _hash_matches(value: Any, hash_field: str) -> bool:
    if not isinstance(value, dict) or not _valid_sha(value.get(hash_field)):
        return False
    try:
        return value[hash_field] == _hash_value(_without(value, hash_field))
    except (TypeError, ValueError, OverflowError):
        return False


def _sha_or_none(value: Any) -> str | None:
    return value if _valid_sha(value) else None


def _valid_epoch(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) \
        and math.isfinite(float(value)) and float(value) >= 0


def _safe_validate_swap_state(state: Any, baseline: Any) -> list[str]:
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)) \
            or not math.isfinite(float(baseline)) or float(baseline) < 0:
        return ["sealed aggressive swap baseline is invalid"]
    try:
        return aggressive.validate_swap_state(
            state, sealed_baseline_swap_mb=float(baseline)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return ["aggressive swap state is malformed"]


def _sampled_epoch(value: Any) -> float | None:
    """Parse a timezone-qualified wall timestamp without accepting local time."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError, OSError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    epoch = parsed.timestamp()
    return epoch if math.isfinite(epoch) and epoch >= 0 else None


def _freshness_errors(sampled_epoch: float | None, now_epoch: Any, label: str) \
        -> list[str]:
    if sampled_epoch is None or not _valid_epoch(now_epoch):
        return [f"{label} time is invalid"]
    age = float(now_epoch) - sampled_epoch
    if age < 0:
        return [f"{label} is from the future"]
    if age > MAX_CURRENT_EVIDENCE_AGE_SECONDS:
        return [f"{label} is stale"]
    return []


def _file_reference(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _run_read_only(argv: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        process = subprocess.run(argv, capture_output=True, text=True,
                                 timeout=timeout, check=False)
        output = (process.stdout + process.stderr).strip()
        return {"argv": argv, "returncode": process.returncode,
                "output": output[-4000:], "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {"argv": argv, "returncode": None,
                "output": str(exc)[-4000:], "timed_out": True}
    except OSError as exc:
        return {"argv": argv, "returncode": None,
                "output": f"{type(exc).__name__}: {exc}", "timed_out": False}


def _host_topology() -> dict[str, Any]:
    sysctl = shutil.which("sysctl") or "/usr/sbin/sysctl"
    sources, values = {}, {}
    for field, key in TOPOLOGY_SYSCTLS.items():
        result = _run_read_only([sysctl, "-n", key])
        sources[field] = result
        try:
            value = int(result.get("output", "").strip())
        except (TypeError, ValueError):
            value = None
        values[field] = value if result.get("returncode") == 0 and value and value > 0 else None
    verified = values == REVIEWED_TOPOLOGY \
        and values["performance_cores"] + values["efficiency_cores"] \
        == values["physical_cores"]
    topology: dict[str, Any] = {
        **values,
        "reviewed_expected": REVIEWED_TOPOLOGY,
        "verified_for_doctor_v5": verified,
        "source": "read-only-sysctl",
        "sysctl_receipts": sources,
    }
    topology["topology_sha256"] = _hash_value(topology)
    return topology


def _topology_errors(topology: Any) -> list[str]:
    if not isinstance(topology, dict):
        return ["host topology is not an object"]
    if not _hash_matches(topology, "topology_sha256"):
        return ["host topology hash is invalid"]
    errors: list[str] = []
    if topology.get("verified_for_doctor_v5") is not True \
            or topology.get("source") != "read-only-sysctl" \
            or any(topology.get(field) != value
                   for field, value in REVIEWED_TOPOLOGY.items()):
        errors.append("host topology values/source are not the reviewed 28/20/8 probe")
    receipts = topology.get("sysctl_receipts")
    if not isinstance(receipts, dict):
        return errors + ["host topology sysctl receipts are missing"]
    for field, key in TOPOLOGY_SYSCTLS.items():
        row = receipts.get(field)
        argv = row.get("argv") if isinstance(row, dict) else None
        if not isinstance(row, dict) or row.get("returncode") != 0 \
                or row.get("timed_out") is not False \
                or not isinstance(argv, list) or argv[-2:] != ["-n", key] \
                or str(row.get("output", "")).strip() != str(REVIEWED_TOPOLOGY[field]):
            errors.append(f"host topology sysctl receipt is invalid: {field}")
    return errors


def _probe_errors(probe: Any) -> list[str]:
    if not isinstance(probe, dict):
        return ["host probe is not an object"]
    errors: list[str] = []
    if probe.get("schema") != PROBE_SCHEMA or probe.get("version") != VERSION:
        errors.append("host probe schema/version is invalid")
    if not _hash_matches(probe, "probe_sha256"):
        errors.append("host probe identity is invalid")
    if _sampled_epoch(probe.get("sampled_at")) is None:
        errors.append("host probe sampled_at is not timezone-qualified")
    errors.extend(_topology_errors(probe.get("topology")))
    if probe.get("fan_control_read_or_write_attempted") is not False \
            or probe.get("os_service_mutation_attempted") is not False:
        errors.append("host probe crossed the read-only boundary")
    return errors


def _owner_lease_errors(lease: Any) -> list[str]:
    required = {
        "schema", "version", "contract_sha256", "role", "cell_id",
        "process_identity_sha256", "lease_generation",
        "state_generation_at_acquire", "acquired_epoch", "lease_sha256",
    }
    if not isinstance(lease, dict) or set(lease) != required:
        return ["owner lease keys are invalid"]
    errors: list[str] = []
    if lease.get("schema") != OWNER_LEASE_SCHEMA \
            or lease.get("version") != VERSION:
        errors.append("owner lease schema/version is invalid")
    if not _hash_matches(lease, "lease_sha256"):
        errors.append("owner lease hash is invalid")
    if not _valid_sha(lease.get("contract_sha256")) \
            or not _valid_sha(lease.get("process_identity_sha256")):
        errors.append("owner lease contract/process identity is invalid")
    if lease.get("role") not in OWNER_ROLES \
            or not isinstance(lease.get("cell_id"), str) \
            or not lease.get("cell_id"):
        errors.append("owner lease role/cell identity is invalid")
    for field in ("lease_generation", "state_generation_at_acquire"):
        value = lease.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"owner lease {field} is invalid")
    if not _valid_epoch(lease.get("acquired_epoch")):
        errors.append("owner lease acquisition time is invalid")
    return errors


def _sorted_owner_leases(leases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(leases, key=lambda row: (
        row.get("role", ""), row.get("cell_id", ""),
        row.get("process_identity_sha256", ""), row.get("lease_generation", -1),
    ))


def build_owner_lease_snapshot(plan: dict[str, Any], probe: dict[str, Any],
                               owner_leases: list[dict[str, Any]], *,
                               sampled_at: str | None = None) -> dict[str, Any]:
    """Seal a read-only observation of the exact elastic phase-owner leases."""
    if validate_plan(plan):
        raise HostSprintError("cannot bind owner leases to an invalid host plan")
    probe_errors = _probe_errors(probe)
    if probe_errors:
        raise HostSprintError("cannot bind invalid current probe: "
                              + "; ".join(probe_errors))
    if not isinstance(owner_leases, list):
        raise HostSprintError("owner lease observation must be a list")
    copied = json.loads(json.dumps(owner_leases))
    lease_errors = [error for lease in copied for error in _owner_lease_errors(lease)]
    if lease_errors:
        raise HostSprintError("cannot bind invalid owner lease: "
                              + "; ".join(lease_errors))
    copied = _sorted_owner_leases(copied)
    lease_hashes = [row["lease_sha256"] for row in copied]
    owner_identities = [
        {key: row[key] for key in (
            "role", "cell_id", "process_identity_sha256", "lease_sha256"
        )}
        for row in copied
    ]
    if len(lease_hashes) != len(set(lease_hashes)) \
            or len({row["process_identity_sha256"] for row in copied}) != len(copied):
        raise HostSprintError("owner lease observation contains duplicate identities")
    sampled_at = sampled_at or _now()
    sampled_epoch = _sampled_epoch(sampled_at)
    if sampled_epoch is None \
            or any(float(row["acquired_epoch"]) > sampled_epoch for row in copied):
        raise HostSprintError("owner lease observation time is invalid")
    snapshot: dict[str, Any] = {
        "schema": OWNER_SNAPSHOT_SCHEMA, "version": VERSION,
        "plan_sha256": plan["plan_sha256"],
        "probe_sha256": probe["probe_sha256"],
        "topology_sha256": probe["topology"]["topology_sha256"],
        "sampled_at": sampled_at, "owner_count": len(copied),
        "owner_leases": copied, "owner_lease_sha256s": lease_hashes,
        "owner_identities": owner_identities,
    }
    snapshot["snapshot_sha256"] = _hash_value(snapshot)
    return snapshot


def _owner_snapshot_errors(snapshot: Any, plan: dict[str, Any],
                           probe: dict[str, Any], expected_owner_leases: Any, *,
                           now_epoch: Any) -> list[str]:
    required = {
        "schema", "version", "plan_sha256", "probe_sha256", "topology_sha256",
        "sampled_at", "owner_count", "owner_leases", "owner_lease_sha256s",
        "owner_identities", "snapshot_sha256",
    }
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        return ["owner/lease snapshot keys are invalid"]
    errors: list[str] = []
    if snapshot.get("schema") != OWNER_SNAPSHOT_SCHEMA \
            or snapshot.get("version") != VERSION \
            or not _hash_matches(snapshot, "snapshot_sha256"):
        errors.append("owner/lease snapshot identity is invalid")
    plan_sha = plan.get("plan_sha256") if isinstance(plan, dict) else None
    probe_sha = probe.get("probe_sha256") if isinstance(probe, dict) else None
    probe_topology = probe.get("topology") if isinstance(probe, dict) else None
    topology_sha = (probe_topology.get("topology_sha256")
                    if isinstance(probe_topology, dict) else None)
    if snapshot.get("plan_sha256") != plan_sha \
            or snapshot.get("probe_sha256") != probe_sha \
            or snapshot.get("topology_sha256") != topology_sha:
        errors.append("owner/lease snapshot is replayed across plan/probe/topology")
    sampled_epoch = _sampled_epoch(snapshot.get("sampled_at"))
    errors.extend(_freshness_errors(sampled_epoch, now_epoch,
                                    "owner/lease snapshot"))
    probe_epoch = _sampled_epoch(
        probe.get("sampled_at") if isinstance(probe, dict) else None
    )
    if sampled_epoch is not None and probe_epoch is not None \
            and (sampled_epoch < probe_epoch
                 or sampled_epoch - probe_epoch > MAX_CURRENT_EVIDENCE_AGE_SECONDS):
        errors.append("owner/lease snapshot is not contemporaneous with host probe")
    observed = snapshot.get("owner_leases")
    expected = expected_owner_leases
    if not isinstance(observed, list) or not isinstance(expected, (list, tuple)):
        return errors + ["owner/lease observed or expected inventory is invalid"]
    observed_errors = [error for lease in observed for error in _owner_lease_errors(lease)]
    expected_list = json.loads(json.dumps(list(expected)))
    expected_errors = [error for lease in expected_list
                       for error in _owner_lease_errors(lease)]
    errors.extend(observed_errors)
    if expected_errors:
        errors.append("caller expected-owner lease inventory is invalid")
    if observed_errors or expected_errors \
            or any(not isinstance(row, dict) for row in observed + expected_list):
        return errors
    sorted_observed = _sorted_owner_leases(observed)
    sorted_expected = _sorted_owner_leases(expected_list)
    hashes = [row.get("lease_sha256") for row in sorted_observed]
    identities = [
        {key: row.get(key) for key in (
            "role", "cell_id", "process_identity_sha256", "lease_sha256"
        )}
        for row in sorted_observed
    ]
    if snapshot.get("owner_count") != len(sorted_observed) \
            or snapshot.get("owner_lease_sha256s") != hashes \
            or snapshot.get("owner_identities") != identities:
        errors.append("owner/lease snapshot derived inventory is inconsistent")
    if sorted_observed != sorted_expected:
        errors.append("current owner leases differ from the exact expected owner set")
    if not expected_list and snapshot.get("owner_count") != 0:
        errors.append("owner-free gate observed one or more phase owners")
    if len(hashes) != len(set(hashes)) \
            or len({row.get("process_identity_sha256") for row in sorted_observed}) \
            != len(sorted_observed):
        errors.append("owner/lease snapshot contains duplicate identities")
    if sampled_epoch is not None and any(
            _valid_epoch(row.get("acquired_epoch"))
            and float(row["acquired_epoch"]) > sampled_epoch
            for row in sorted_observed):
        errors.append("owner lease was acquired after the owner snapshot")
    return errors


def _swap_decision_semantic_errors(swap_state: dict[str, Any],
                                   decision: Any) -> list[str]:
    exact_fields = {
        "mode", "allow_launch", "launch_limit", "cpu_scale", "shed_one",
        "reason", "probe_valid", "swap_growth_mb", "swap_rate_mb_min",
        "green_streak", "hard_until_epoch", "running_evidence_invalidated",
        "emergency_action",
    }
    if not isinstance(decision, dict) or set(decision) != exact_fields:
        return ["aggressive swap decision fields are invalid"]
    if not isinstance(swap_state, dict):
        return ["aggressive swap state is not an object"]
    raw_mode = swap_state.get("mode")
    mode = raw_mode if isinstance(raw_mode, str) else None
    expected = {
        "green": (True, aggressive.MAX_LANES, 1.0),
        "soft_throttle": (True, 1, 0.5),
        "hard_stop": (False, 0, 0.0),
        "emergency_shed": (False, 0, 0.0),
    }.get(mode)
    if expected is None:
        return ["aggressive swap state mode is invalid"]
    errors: list[str] = []
    if decision.get("mode") != mode \
            or (decision.get("allow_launch"), decision.get("launch_limit"),
                decision.get("cpu_scale")) != expected \
            or decision.get("green_streak") != swap_state.get("green_streak") \
            or decision.get("hard_until_epoch") != swap_state.get("hard_until_epoch") \
            or decision.get("running_evidence_invalidated") is not False:
        errors.append("aggressive swap decision differs from its successor state")
    probe_valid = decision.get("probe_valid")
    previous_swap = swap_state.get("previous_swap_mb")
    baseline_swap = swap_state.get("baseline_swap_mb")
    state_swap_numbers_valid = all(
        not isinstance(value, bool) and isinstance(value, (int, float))
        and math.isfinite(float(value)) and float(value) >= 0
        for value in (previous_swap, baseline_swap)
    )
    if not state_swap_numbers_valid:
        errors.append("aggressive swap state growth inputs are invalid")
    expected_growth = (round(
        float(previous_swap) - float(baseline_swap), 3
    ) if probe_valid is True and state_swap_numbers_valid else None)
    rate = decision.get("swap_rate_mb_min")
    if probe_valid not in {True, False} \
            or decision.get("swap_growth_mb") != expected_growth \
            or (probe_valid is True and (
                isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate)) or float(rate) < 0
            )) or (probe_valid is False and rate is not None):
        errors.append("aggressive swap decision probe/growth/rate is inconsistent")
    allowed_reasons = {
        "green": {"normal pressure inside swap envelope"},
        "soft_throttle": {
            "soft swap growth/rate bound", "soft-throttle hysteresis/cooldown",
        },
        "hard_stop": {
            "warning pressure or hard swap bound", "resource probe invalid",
            "hard-stop hysteresis/cooldown",
        },
        "emergency_shed": {"critical pressure or emergency swap bound"},
    }
    if decision.get("reason") not in allowed_reasons[mode]:
        errors.append("aggressive swap decision reason is not a controller outcome")
    shed = decision.get("shed_one")
    emergency = decision.get("emergency_action")
    if mode != "emergency_shed" and (shed is not False or emergency is not None):
        errors.append("non-emergency swap decision contains a shed action")
    if mode == "emergency_shed" and (
            shed not in {True, False}
            or emergency != (
                "checkpoint/receipt then shed one largest-RSS lane" if shed else None
            )):
        errors.append("emergency swap shed action is inconsistent")
    if mode == "green" and probe_valid is not True:
        errors.append("green swap decision does not bind a valid probe")
    return errors


def bind_aggressive_swap_decision(swap_state: dict[str, Any],
                                  decision: dict[str, Any]) -> dict[str, Any]:
    """Produce the same wire artifact accepted by the elastic scheduler."""
    baseline = swap_state.get("baseline_swap_mb") if isinstance(swap_state, dict) else None
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)):
        raise HostSprintError("cannot bind invalid aggressive swap state")
    state_errors = _safe_validate_swap_state(swap_state, baseline)
    decision_errors = _swap_decision_semantic_errors(swap_state, decision)
    if state_errors or decision_errors:
        raise HostSprintError("cannot bind aggressive swap evidence: "
                              + "; ".join(state_errors + decision_errors))
    binding: dict[str, Any] = {
        "schema": SWAP_DECISION_SCHEMA, "version": VERSION,
        "swap_state_sha256": swap_state["state_sha256"],
        "controller_policy_sha256": _hash_value(aggressive.swap_policy()),
        "decision": json.loads(json.dumps(decision)),
    }
    binding["binding_sha256"] = _hash_value(binding)
    return binding


def _swap_binding_errors(swap_state: Any, binding: Any, *,
                         sealed_baseline_swap_mb: Any,
                         now_epoch: Any) -> list[str]:
    if isinstance(sealed_baseline_swap_mb, bool) \
            or not isinstance(sealed_baseline_swap_mb, (int, float)) \
            or not math.isfinite(float(sealed_baseline_swap_mb)) \
            or float(sealed_baseline_swap_mb) < 0:
        return ["sealed aggressive swap baseline is invalid"]
    errors = _safe_validate_swap_state(swap_state, sealed_baseline_swap_mb)
    if not isinstance(binding, dict) \
            or binding.get("schema") != SWAP_DECISION_SCHEMA \
            or binding.get("version") != VERSION \
            or not _hash_matches(binding, "binding_sha256") \
            or binding.get("swap_state_sha256") \
            != (swap_state.get("state_sha256") if isinstance(swap_state, dict) else None) \
            or binding.get("controller_policy_sha256") \
            != _hash_value(aggressive.swap_policy()):
        errors.append("aggressive swap controller binding is invalid or stale")
    if isinstance(swap_state, dict):
        errors.extend(_freshness_errors(
            float(swap_state["previous_sample_epoch"])
            if _valid_epoch(swap_state.get("previous_sample_epoch")) else None,
            now_epoch, "aggressive swap controller sample",
        ))
    if isinstance(binding, dict):
        errors.extend(_swap_decision_semantic_errors(
            swap_state if isinstance(swap_state, dict) else {},
            binding.get("decision"),
        ))
        if not errors:
            try:
                expected = bind_aggressive_swap_decision(
                    swap_state, binding["decision"]
                )
            except HostSprintError as exc:
                errors.append(str(exc))
            else:
                if expected != binding:
                    errors.append("aggressive swap binding differs from controller output")
    return errors


def probe_host() -> dict[str, Any]:
    try:
        resources = ram_scheduler.resource_snapshot(str(ROOT))
    except Exception as exc:
        resources = {"error": f"{type(exc).__name__}: {exc}"}
    thermal = _run_read_only(["pmset", "-g", "therm"])
    thermal_green = ram_scheduler.thermal_output_ok(
        thermal.get("returncode") or 0, thermal.get("output", "")
    )
    tools = {name: shutil.which(name) for name in (
        "caffeinate", "taskpolicy", "renice", "mdutil", "tmutil"
    )}
    spotlight = (_run_read_only([tools["mdutil"], "-s", str(ROOT)])
                 if tools["mdutil"] else {"available": False})
    backup = (_run_read_only([tools["tmutil"], "status"])
              if tools["tmutil"] else {"available": False})
    probe: dict[str, Any] = {
        "schema": PROBE_SCHEMA, "version": VERSION, "sampled_at": _now(),
        "topology": _host_topology(),
        "resource_snapshot": resources, "thermal_probe": thermal,
        "thermal_green": thermal_green, "tools": tools,
        "spotlight_status_read_only": spotlight,
        "backup_status_read_only": backup,
        "fan_control_read_or_write_attempted": False,
        "os_service_mutation_attempted": False,
    }
    probe["probe_sha256"] = _hash_value(probe)
    return probe


def build_plan(probe: dict[str, Any]) -> dict[str, Any]:
    if not _hash_matches(probe, "probe_sha256"):
        raise HostSprintError("host probe identity is invalid")
    tools = probe.get("tools", {})
    workspace = str(ROOT.resolve())
    proposals = [
        {
            "id": "caffeinate-supervisor", "class": "process-wrapper",
            "automatic": False, "requires_user_authorization": True,
            "prerequisite": "caffeinate executable is source/path bound",
            "proposal": [tools.get("caffeinate") or "/usr/bin/caffeinate", "-dimsu",
                         "<source-bound-supervisor-command>"],
            "reversal": "wrapper exits with the supervised process; no persistent host change",
            "risk": "low",
        },
        {
            "id": "companion-positive-nice", "class": "owned-process-priority",
            "automatic": False, "requires_user_authorization": True,
            "prerequisite": "target PID/start identity belongs to the companion process",
            "proposal": [tools.get("renice") or "/usr/bin/renice", "5", "-p",
                         "<bound-companion-pid>"],
            "reversal": [tools.get("renice") or "/usr/bin/renice", "0", "-p",
                         "<same-bound-companion-pid>"],
            "risk": "low; lowering companion priority only",
        },
        {
            "id": "companion-background-qos", "class": "owned-process-qos",
            "automatic": False, "requires_user_authorization": True,
            "prerequisite": "taskpolicy semantics are manually verified on this macOS build",
            "proposal": [tools.get("taskpolicy") or "/usr/bin/taskpolicy", "-b",
                         "<source-bound-companion-command>"],
            "reversal": "launch the next companion without background taskpolicy",
            "risk": "medium; proposal only until physical A/B proves benefit",
        },
        {
            "id": "workspace-spotlight-exclusion", "class": "optional-os-indexing-exclusion",
            "automatic": False, "requires_user_authorization": True,
            "supported": False,
            "prerequisite": (
                "unsupported until a read-only preflight proves exact per-folder semantics; "
                "otherwise use the manual macOS System Settings privacy UI only"
            ),
            "proposal": "manual System Settings action; no mdutil command is authorized",
            "reversal": "remove the same manual privacy-list entry",
            "risk": (
                "high; mdutil is volume-oriented and is not accepted as a workspace-folder "
                "execution contract"
            ),
        },
        {
            "id": "workspace-backup-exclusion", "class": "optional-backup-exclusion",
            "automatic": False, "requires_user_authorization": True,
            "prerequisite": "explicit user opt-in and existing backup policy review",
            "proposal": [tools.get("tmutil") or "/usr/bin/tmutil", "addexclusion", workspace],
            "reversal": [tools.get("tmutil") or "/usr/bin/tmutil", "removeexclusion", workspace],
            "risk": "high; proposal only and never changes source retention",
        },
    ]
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA, "version": VERSION, "created_at": _now(),
        "mode": "unbound-default-off-proposals-only",
        "enabled_by_default": False, "automatic_execution_permitted": False,
        "source_bindings": {
            "host_sprint_module": _file_reference(Path(__file__)),
            "local_observer": _file_reference(Path(local_observer.__file__)),
            "local_observer_authority_tools": (
                local_observer.authority_tool_references()
            ),
            "aggressive_policy": _file_reference(Path(aggressive.__file__)),
        },
        "probe": probe, "host_probe_sha256": probe["probe_sha256"],
        "topology": probe.get("topology"), "proposals": proposals,
        "gates": {
            "ac_power_required": True, "thermal_green_required": True,
            "memory_pressure_normal_required": True,
            "aggressive_swap_green_required": True,
            "current_evidence_max_age_seconds": MAX_CURRENT_EVIDENCE_AGE_SECONDS,
            "swap_decision_schema": SWAP_DECISION_SCHEMA,
            "swap_controller_policy_sha256": _hash_value(aggressive.swap_policy()),
            "owner_snapshot_schema": OWNER_SNAPSHOT_SCHEMA,
            "owner_lease_schema": OWNER_LEASE_SCHEMA,
            "exact_expected_owner_set_required": True,
            "sealed_swap_baseline_mb": probe.get(
                "resource_snapshot", {}
            ).get("swap_used_mb"),
            "cooling_policy": (
                "ambient/physical cooling may be improved manually; no fan-control API, "
                "kernel extension, SMC write, or automatic service mutation is permitted"
            ),
        },
        "forbidden_automatic_actions": [
            "fan-control writes", "SMC writes", "launchctl mutation",
            "Spotlight mutation", "backup exclusion mutation", "negative nice escalation",
            "runtime default changes",
        ],
        "rollback": {
            "reversal_receipt_required": True,
            "rollback_requires_owner_free_checkpoint": True,
            "completed_evidence_mutation_permitted": False,
            "parent_source_deletion_permitted": False,
        },
        "benchmark_claim_boundary": {
            "component": "host-sprint-isolation",
            "synthetic_probe_may_change_eta": False,
            "production_owner_free_full_stack_ab_required": True,
            "required_metrics": [
                "program/input/output/receipt identity",
                "invocation/semantic-contract identity",
                "wall_seconds", "cpu_seconds", "gpu_seconds", "peak_rss_bytes",
                "scratch_peak_bytes", "disk free start/end", "swap start/end",
                "memory pressure start/end", "thermal start/end",
                "source/default lifecycle flags",
            ],
        },
    }
    plan["plan_sha256"] = _hash_value(plan)
    return plan


def validate_plan(plan: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["host sprint plan is not an object"]
    if plan.get("schema") != PLAN_SCHEMA or plan.get("version") != VERSION \
            or plan.get("mode") != "unbound-default-off-proposals-only" \
            or plan.get("enabled_by_default") is not False \
            or plan.get("automatic_execution_permitted") is not False:
        errors.append("host sprint schema/default-off identity is invalid")
    if not _hash_matches(plan, "plan_sha256"):
        errors.append("host sprint plan hash mismatch")
    bindings = plan.get("source_bindings")
    if not isinstance(bindings, dict) or any(
            not aggressive._reference_matches(bindings.get(name))
            for name in ("host_sprint_module", "local_observer", "aggressive_policy")
    ):
        errors.append("host sprint source binding is absent or stale")
    try:
        authority_tools = local_observer.authority_tool_references()
    except (OSError, local_observer.LocalObserverError):
        authority_tools = None
    if not isinstance(bindings, dict) \
            or bindings.get("local_observer_authority_tools") != authority_tools:
        errors.append("host sprint authority-tool binding is absent or stale")
    probe = plan.get("probe")
    if _probe_errors(probe) \
            or plan.get("host_probe_sha256") \
            != (probe.get("probe_sha256") if isinstance(probe, dict) else None):
        errors.append("host sprint signed probe binding is invalid")
    topology = probe.get("topology") if isinstance(probe, dict) else None
    if _topology_errors(topology) or plan.get("topology") != topology:
        errors.append("host sprint reviewed 28/20/8 topology binding is invalid")
    proposals = plan.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != 5:
        errors.append("host sprint proposal inventory is incomplete")
    else:
        for proposal in proposals:
            if not isinstance(proposal, dict) or proposal.get("automatic") is not False \
                    or proposal.get("requires_user_authorization") is not True \
                    or not proposal.get("reversal"):
                errors.append("host sprint proposal is not explicitly reversible/default-off")
        spotlight = next((row for row in proposals
                          if row.get("id") == "workspace-spotlight-exclusion"), {})
        if spotlight.get("supported") is not False \
                or isinstance(spotlight.get("proposal"), list) \
                or "/mdutil" in str(spotlight.get("proposal")):
            errors.append("Spotlight folder semantics are overclaimed")
    forbidden = plan.get("forbidden_automatic_actions")
    if not isinstance(forbidden, list) or "fan-control writes" not in forbidden \
            or "Spotlight mutation" not in forbidden \
            or "backup exclusion mutation" not in forbidden:
        errors.append("host sprint forbidden-action boundary is incomplete")
    rollback = plan.get("rollback")
    if not isinstance(rollback, dict) \
            or rollback.get("completed_evidence_mutation_permitted") is not False \
            or rollback.get("parent_source_deletion_permitted") is not False:
        errors.append("host sprint rollback/evidence boundary is invalid")
    gates = plan.get("gates")
    if not isinstance(gates, dict) \
            or gates.get("current_evidence_max_age_seconds") \
            != MAX_CURRENT_EVIDENCE_AGE_SECONDS \
            or gates.get("swap_decision_schema") != SWAP_DECISION_SCHEMA \
            or gates.get("swap_controller_policy_sha256") \
            != _hash_value(aggressive.swap_policy()) \
            or gates.get("owner_snapshot_schema") != OWNER_SNAPSHOT_SCHEMA \
            or gates.get("owner_lease_schema") != OWNER_LEASE_SCHEMA \
            or gates.get("exact_expected_owner_set_required") is not True \
            or isinstance(gates.get("sealed_swap_baseline_mb"), bool) \
            or not isinstance(gates.get("sealed_swap_baseline_mb"), (int, float)) \
            or not math.isfinite(float(gates["sealed_swap_baseline_mb"])) \
            or float(gates["sealed_swap_baseline_mb"]) < 0:
        errors.append("host sprint evidence/controller gate contract is invalid")
    return errors


def evaluate_gate(plan: dict[str, Any], probe: dict[str, Any],
                  aggressive_swap_state: dict[str, Any],
                  aggressive_swap_binding: dict[str, Any] | None = None,
                  owner_lease_snapshot: dict[str, Any] | None = None, *,
                  now_epoch: float | None = None,
                  sealed_baseline_swap_mb: float | None = None,
                  expected_owner_leases: tuple[dict[str, Any], ...] | list[dict[str, Any]]
                  = ()) -> dict[str, Any]:
    """Pure caller-evidence evaluator; never grants production authority."""
    blockers = validate_plan(plan)
    blockers.extend(_probe_errors(probe))
    probe_epoch = _sampled_epoch(probe.get("sampled_at")) \
        if isinstance(probe, dict) else None
    blockers.extend(_freshness_errors(probe_epoch, now_epoch, "current host probe"))
    snapshot = probe.get("resource_snapshot", {}) \
        if isinstance(probe, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    topology = probe.get("topology") if isinstance(probe, dict) else None
    topology = topology if isinstance(topology, dict) else {}
    plan_topology = plan.get("topology") if isinstance(plan, dict) else None
    plan_topology = plan_topology if isinstance(plan_topology, dict) else {}
    if _topology_errors(topology) \
            or topology.get("topology_sha256") \
            != plan_topology.get("topology_sha256"):
        blockers.append("current host topology differs from the signed sprint plan")
    if "AC Power" not in str(snapshot.get("power_source", "")):
        blockers.append("AC power is not confirmed")
    if snapshot.get("pressure_level") != 1:
        blockers.append("memory pressure is not normal")
    if not isinstance(probe, dict) or probe.get("thermal_green") is not True:
        blockers.append("thermal/cooling gate is not explicitly green")
    blockers.extend(_swap_binding_errors(
        aggressive_swap_state, aggressive_swap_binding,
        sealed_baseline_swap_mb=sealed_baseline_swap_mb, now_epoch=now_epoch,
    ))
    bound_decision = aggressive_swap_binding.get("decision", {}) \
        if isinstance(aggressive_swap_binding, dict) else {}
    if bound_decision.get("mode") != "green" \
            or bound_decision.get("allow_launch") is not True:
        blockers.append("aggressive swap controller is not green")
    blockers.extend(_owner_snapshot_errors(
        owner_lease_snapshot, plan, probe, expected_owner_leases,
        now_epoch=now_epoch,
    ))
    # Preserve first occurrence order while making the receipt unambiguous.
    blockers = list(dict.fromkeys(blockers))
    gate: dict[str, Any] = {
        "schema": GATE_SCHEMA, "version": VERSION,
        "plan_sha256": _sha_or_none(
            plan.get("plan_sha256") if isinstance(plan, dict) else None
        ),
        "probe_sha256": _sha_or_none(
            probe.get("probe_sha256") if isinstance(probe, dict) else None
        ),
        "topology_sha256": (
            _sha_or_none(topology.get("topology_sha256"))
            if isinstance(probe, dict) else None
        ),
        "aggressive_swap_state_sha256": (
            _sha_or_none(aggressive_swap_state.get("state_sha256"))
            if isinstance(aggressive_swap_state, dict) else None
        ),
        "aggressive_swap_binding_sha256": (
            _sha_or_none(aggressive_swap_binding.get("binding_sha256"))
            if isinstance(aggressive_swap_binding, dict) else None
        ),
        "swap_controller_policy_sha256": _hash_value(aggressive.swap_policy()),
        "owner_lease_snapshot_sha256": (
            _sha_or_none(owner_lease_snapshot.get("snapshot_sha256"))
            if isinstance(owner_lease_snapshot, dict) else None
        ),
        "expected_owner_lease_sha256s": sorted(
            row.get("lease_sha256") for row in expected_owner_leases
            if isinstance(row, dict) and isinstance(row.get("lease_sha256"), str)
        ) if isinstance(expected_owner_leases, (list, tuple)) else [],
        "evaluated_now_epoch": float(now_epoch) if _valid_epoch(now_epoch) else None,
        "ok": not blockers, "blockers": blockers,
        "evidence_authority": "caller-attested-test-only",
        "production_authorized": False,
        "automatic_actions_executed": False,
        "fan_control_touched": False, "os_services_mutated": False,
        "recorded_at": _now(),
    }
    gate["gate_sha256"] = _hash_value(gate)
    return gate


def _probe_from_local_observer(receipt: dict[str, Any]) -> dict[str, Any]:
    resources = receipt.get("resources", {})
    topology_receipts = resources.get("topology_receipts", {})
    converted = {}
    for field in TOPOLOGY_SYSCTLS:
        row = topology_receipts.get(field, {}) \
            if isinstance(topology_receipts, dict) else {}
        converted[field] = {
            "argv": row.get("argv"), "returncode": row.get("returncode"),
            "output": (str(row.get("stdout", ""))
                       + str(row.get("stderr", ""))).strip(),
            "timed_out": row.get("timed_out"),
        }
    raw_topology = resources.get("topology", {})
    values = {
        field: raw_topology.get(field) if isinstance(raw_topology, dict) else None
        for field in REVIEWED_TOPOLOGY
    }
    topology: dict[str, Any] = {
        **values, "reviewed_expected": REVIEWED_TOPOLOGY,
        "verified_for_doctor_v5": values == REVIEWED_TOPOLOGY,
        "source": "read-only-sysctl", "sysctl_receipts": converted,
    }
    topology["topology_sha256"] = _hash_value(topology)
    thermal = resources.get("thermal_receipt", {})
    thermal_probe = {
        "argv": thermal.get("argv"), "returncode": thermal.get("returncode"),
        "output": (str(thermal.get("stdout", ""))
                   + str(thermal.get("stderr", ""))).strip(),
        "timed_out": thermal.get("timed_out"),
    }
    sampled = dt.datetime.fromtimestamp(
        float(receipt["observed_wall_epoch"]), tz=dt.timezone.utc,
    ).isoformat(timespec="seconds")
    probe: dict[str, Any] = {
        "schema": PROBE_SCHEMA, "version": VERSION, "sampled_at": sampled,
        "topology": topology,
        "resource_snapshot": {
            "schema": "hawking.studio_resource_snapshot.v1",
            "ok": resources.get("probe_valid") is True,
            "sampled_at": sampled,
            "power_source": "AC Power" if resources.get("ac_power") is True else None,
            "pressure_level": resources.get("pressure_level"),
            "swap_used_mb": resources.get("swap_used_mb"),
        },
        "thermal_probe": thermal_probe,
        "thermal_green": resources.get("thermal_green") is True,
        "tools": {}, "spotlight_status_read_only": {"available": False},
        "backup_status_read_only": {"available": False},
        "fan_control_read_or_write_attempted": False,
        "os_service_mutation_attempted": False,
    }
    probe["probe_sha256"] = _hash_value(probe)
    return probe


def evaluate_gate_from_local_observer(plan: dict[str, Any], *,
                                      elastic_state_path: Path,
                                      aggressive_swap_state_path: Path) \
        -> dict[str, Any]:
    """Evaluate trusted host evidence without accepting caller-made observations.

    The paths identify persisted artifacts; their contents, owner inventory,
    process membership, clock, pressure, swap, power, and thermal state are all
    read directly by the source-bound observer under the elastic state lock.
    This remains default-off and executes no host action.
    """
    blockers = validate_plan(plan)
    try:
        state_path = Path(elastic_state_path).resolve(strict=True)
        swap_path = Path(aggressive_swap_state_path).resolve(strict=True)
        state_path.relative_to(ELASTIC_STAGE_ROOT.resolve(strict=True))
        swap_path.relative_to(ELASTIC_STAGE_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise HostSprintError(
            "trusted local gate artifacts must exist below elastic_v1"
        ) from exc
    try:
        observed = local_observer.observe_with_state_lock(
            state_path,
            extra_json_paths={"aggressive_swap_state": swap_path},
        )
    except local_observer.LocalObserverError as exc:
        raise HostSprintError(f"trusted local observer failed closed: {exc}") from exc
    if observed.get("observer_receipt_sha256") != _hash_value(
            _without(observed, "observer_receipt_sha256")
    ) or observed.get("authority") != "trusted-local-observer-under-state-lock":
        blockers.append("trusted local observer receipt identity is invalid")
    if observed.get("stable_file_read_method") \
            != "open-fstat-before-after-no-follow":
        blockers.append("trusted local observer stable-file method is invalid")
    if observed.get("observer_source") \
            != plan.get("source_bindings", {}).get("local_observer"):
        blockers.append("trusted local observer source/artifact differs from plan")
    if observed.get("authority_tools") != plan.get(
            "source_bindings", {}).get("local_observer_authority_tools"):
        blockers.append("trusted local observer authority tools differ from plan")
    lock = observed.get("lock_lease", {})
    if not isinstance(lock, dict) or lock.get("lock_lease_sha256") != _hash_value(
            _without(lock, "lock_lease_sha256")
    ) or lock.get("state_sha256") != observed.get("state_sha256") \
            or lock.get("state_generation") != observed.get("state_generation"):
        blockers.append("trusted local observer lock lease is invalid")
    resources = observed.get("resources", {})
    if not isinstance(resources, dict) \
            or resources.get("resource_sha256") != _hash_value(
                _without(resources, "resource_sha256")
            ) or resources.get("source") != "direct-local-subprocess":
        blockers.append("trusted direct sysctl/pressure/swap receipt is invalid")
    probe = _probe_from_local_observer(observed)
    owner_rows = observed.get("persisted_owner_observations", [])
    owner_leases = [
        row.get("owner_lease") for row in owner_rows if isinstance(row, dict)
    ] if isinstance(owner_rows, list) else []
    owner_snapshot = None
    try:
        owner_snapshot = build_owner_lease_snapshot(
            plan, probe, owner_leases,
            sampled_at=probe["sampled_at"],
        )
    except HostSprintError as exc:
        blockers.append(str(exc))
    if owner_rows:
        blockers.append("trusted state is not owner-free")
    if any(
            not isinstance(row, dict)
            or row.get("process_observation", {}).get("exact_identity_running")
            not in {True, False}
            for row in owner_rows
    ):
        blockers.append("trusted owner/process observation is malformed")
    if observed.get("heavy_owner_count") != len(observed.get("heavy_owners", [])):
        blockers.append("trusted heavy-owner inventory is inconsistent")
    if observed.get("heavy_owner_count") != 0:
        blockers.append("trusted local observer found active heavy owners")

    baseline = plan.get("gates", {}).get("sealed_swap_baseline_mb")
    extra = observed.get("extra_json", {}).get("aggressive_swap_state", {})
    prior = extra.get("value") if isinstance(extra, dict) else None
    state_errors = _safe_validate_swap_state(prior, baseline)
    binding = None
    current_state = prior if isinstance(prior, dict) else {}
    if state_errors:
        blockers.extend(f"persisted swap controller: {row}" for row in state_errors)
    else:
        current_state, decision = aggressive.advance_swap_state(
            prior, {
                "pressure_level": resources.get("pressure_level"),
                "swap_used_mb": resources.get("swap_used_mb"),
            }, now_epoch=float(observed["observed_wall_epoch"]),
            sealed_baseline_swap_mb=float(baseline),
        )
        try:
            binding = bind_aggressive_swap_decision(current_state, decision)
        except HostSprintError as exc:
            blockers.append(str(exc))
    pure = evaluate_gate(
        plan, probe, current_state, binding, owner_snapshot,
        now_epoch=float(observed["observed_wall_epoch"]),
        sealed_baseline_swap_mb=float(baseline) if isinstance(baseline, (int, float))
        and not isinstance(baseline, bool) else None,
        expected_owner_leases=owner_leases,
    )
    blockers.extend(pure["blockers"])
    blockers = list(dict.fromkeys(blockers))
    pure.update({
        "ok": not blockers, "blockers": blockers,
        "evidence_authority": "trusted-local-observer-under-state-lock",
        # The plan is intentionally proposals-only; no gate receipt itself may
        # execute a host action after releasing the lock.
        "production_authorized": False,
        "trusted_evidence_gate_passed": not blockers,
        "trusted_local_observer_receipt_sha256": observed[
            "observer_receipt_sha256"
        ],
        "trusted_local_observer_source": observed["observer_source"],
        "observed_state_sha256": observed["state_sha256"],
        "observed_state_generation": observed["state_generation"],
        "lock_lease_sha256": observed["lock_lease"]["lock_lease_sha256"],
        "observed_monotonic_ns": observed["observed_monotonic_ns"],
        "heavy_owner_count": observed["heavy_owner_count"],
        "heavy_owners": observed["heavy_owners"],
        "automatic_actions_executed": False,
    })
    pure["gate_sha256"] = _hash_value(_without(pure, "gate_sha256"))
    return pure


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True,
                         ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe", help="read-only host sprint probe/plan projection")
    stage = sub.add_parser("stage", help="write only an inert proposal plan")
    stage.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    probe = probe_host()
    plan = build_plan(probe)
    if args.command == "stage":
        output = args.output.resolve()
        try:
            output.relative_to(STAGE_ROOT.resolve())
        except ValueError as exc:
            raise HostSprintError("host sprint staging must remain below host_sprint_v1") from exc
        _atomic_json(output, plan)
    print(json.dumps({
        "probe_sha256": probe["probe_sha256"], "plan_sha256": plan["plan_sha256"],
        "enabled_by_default": plan["enabled_by_default"],
        "automatic_execution_permitted": plan["automatic_execution_permitted"],
        "thermal_green": probe["thermal_green"],
        "validation_errors": validate_plan(plan),
        "proposal_count": len(plan["proposals"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except HostSprintError as exc:
        print(f"doctor_v5_host_sprint_plan: {exc}", file=sys.stderr)
        raise SystemExit(2)
