#!/usr/bin/env python3.12
"""Unbound Doctor V5 aggressive-admission policy for a future generation.

This module is intentionally not imported by the active Doctor supervisor.  It
turns structurally authenticated process-tree samples into deterministic
reservations, selects exact thread profiles for heterogeneous packing, and
implements a bounded-swap hysteresis controller.  ``stage`` writes only a new
document below ``staged_acceleration/aggressive_v2``.  It never edits campaign
state, runtime specifications, evidence, results, active markers, or launch
agents.

Promotion is deliberately a separate quiescent transaction.  The future
generation must bind this source, re-stage against the terminal checkpoint,
prove exact output parity for every selected thread count, and then install the
policy and pending-only runtime generation atomically.
"""
from __future__ import annotations

import argparse
import datetime as dt
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
ULTRA_ROOT = ROOT / "reports/condense/doctor_v5_ultra"
PLAN = ULTRA_ROOT / "campaign_plan.json"
STATE = ULTRA_ROOT / "queue_state.json"
CHILD_RESOURCES = ULTRA_ROOT / "child_resources.jsonl"
STAGE_ROOT = ULTRA_ROOT / "staged_acceleration/aggressive_v2"
DEFAULT_OVERLAY = STAGE_ROOT / "aggressive_admission_overlay.json"
THREAD_PROFILE_CONTRACT_PATH = (
    ROOT / "vendor/strand-quant/tools/thread_profile_contract.py"
)

SCHEMA = "hawking.doctor_v5_aggressive_admission_overlay.v1"
SWAP_STATE_SCHEMA = "hawking.doctor_v5_aggressive_swap_state.v1"
VERSION = "2026-07-14.1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")

PROCESS_BUDGET_BYTES = 78_000_000_000
GLOBAL_RESERVE_BYTES = 12_000_000_000
ADMISSION_CEILING_BYTES = PROCESS_BUDGET_BYTES - GLOBAL_RESERVE_BYTES
RSS_MARGIN_FLOOR_BYTES = 2_000_000_000
RSS_MARGIN_RATIO = 0.15
RSS_ROUND_BYTES = 512_000_000
MIN_AUTHENTICATED_SAMPLES = 5
MIN_SAMPLE_SPAN_SECONDS = 24.0
MAX_PACK_CANDIDATES = 24
MAX_LANES = 8
CPU_BUDGET_CORES = 24
REQUIRED_THREAD_PARITY = (8, 12, 16, 20)
THREAD_PROFILE_CONTRACT = {
    8: {"name": "companion-8", "exclusive": False},
    12: {"name": "balanced-12", "exclusive": False},
    16: {"name": "large-16", "exclusive": False},
    20: {"name": "exclusive-20", "exclusive": True},
}

SWAP_SOFT_GROWTH_MB = 512.0
SWAP_HARD_GROWTH_MB = 1536.0
SWAP_EMERGENCY_GROWTH_MB = 3072.0
SWAP_ABSOLUTE_EMERGENCY_MB = 4096.0
SWAP_SOFT_RATE_MB_MIN = 256.0
SWAP_HARD_RATE_MB_MIN = 1024.0
SWAP_SOFT_RECOVERY_SAMPLES = 2
SWAP_HARD_RECOVERY_SAMPLES = 3
SWAP_SOFT_COOLDOWN_SECONDS = 60.0
SWAP_HARD_COOLDOWN_SECONDS = 180.0
SWAP_SHED_COOLDOWN_SECONDS = 60.0


class PolicyError(RuntimeError):
    """The staged aggressive policy cannot be trusted or promoted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PolicyError(f"JSON root is not an object: {path}")
    return value


def _file_reference(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _reference_matches(reference: Any) -> bool:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "bytes"}:
        return False
    try:
        path = (ROOT / reference["path"]).resolve(strict=True)
        path.relative_to(ROOT.resolve())
        raw = path.read_bytes()
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return len(raw) == reference["bytes"] \
        and hashlib.sha256(raw).hexdigest() == reference["sha256"]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True,
                         ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_time(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _round_up(value: int, quantum: int = RSS_ROUND_BYTES) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 \
            or isinstance(quantum, bool) or not isinstance(quantum, int) or quantum <= 0:
        raise PolicyError("rounding inputs are invalid")
    return ((value + quantum - 1) // quantum) * quantum


def _validate_plan_state(plan: dict[str, Any], state: dict[str, Any]) -> None:
    if plan.get("schema") != "hawking.doctor_v5_ultra_campaign_plan.v1" \
            or plan.get("plan_sha256") != _hash_value(_without(plan, "plan_sha256")):
        raise PolicyError("campaign plan identity is invalid")
    if state.get("schema") != "hawking.doctor_v5_ultra_queue_state.v1" \
            or state.get("plan_sha256") != plan["plan_sha256"] \
            or state.get("state_sha256") != _hash_value(_without(state, "state_sha256")):
        raise PolicyError("queue state identity is invalid")
    cells = plan.get("cells")
    rows = state.get("cells")
    if not isinstance(cells, list) or not isinstance(rows, dict) \
            or {row.get("cell_id") for row in cells if isinstance(row, dict)} != set(rows):
        raise PolicyError("plan/state cell sets differ")


def residency_profile_key(cell: dict[str, Any]) -> str:
    """Exact cross-rate residency class; never borrow a different branch's HWM."""
    label, branch = cell.get("model_label"), cell.get("branch")
    admission = cell.get("admission")
    if not isinstance(label, str) or not label or not isinstance(branch, str) or not branch \
            or not isinstance(admission, dict) \
            or admission.get("whole_parent_residency_assumed") not in {True, False}:
        raise PolicyError("cell residency identity is invalid")
    mode = "whole-parent" if admission["whole_parent_residency_assumed"] else "streaming"
    return f"{label}|{branch}|{mode}"


def _authenticated_sample(row: Any, *, plan_sha256: str,
                          cells: dict[str, dict[str, Any]],
                          state_rows: dict[str, dict[str, Any]]) \
        -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(row, dict):
        return None, "row is not an object"
    cell_id = row.get("cell_id")
    cell, state_row = cells.get(cell_id), state_rows.get(cell_id)
    if cell is None or not isinstance(state_row, dict):
        return None, "cell is not in the bound plan/state"
    if row.get("plan_sha256") != plan_sha256:
        return None, "plan binding mismatch"
    request_sha = row.get("request_sha256")
    if not _valid_hash(request_sha) or request_sha != state_row.get("request_sha256"):
        return None, "request binding mismatch"
    if row.get("process_budget_bytes") != PROCESS_BUDGET_BYTES:
        return None, "process budget binding mismatch"
    sampled_epoch = _parse_time(row.get("sampled_at"))
    if sampled_epoch is None:
        return None, "sample timestamp is invalid"
    pgid, root_pid = row.get("pgid"), row.get("root_pid")
    processes = row.get("processes")
    tree_rss = row.get("tree_rss_bytes")
    maximum = row.get("max_tree_rss_bytes")
    if isinstance(pgid, bool) or not isinstance(pgid, int) or pgid <= 1 or root_pid != pgid \
            or not isinstance(processes, list) or not processes \
            or row.get("process_count") != len(processes) \
            or isinstance(tree_rss, bool) or not isinstance(tree_rss, int) or tree_rss <= 0 \
            or tree_rss > PROCESS_BUDGET_BYTES * 2 \
            or isinstance(maximum, bool) or not isinstance(maximum, int) \
            or maximum < tree_rss:
        return None, "process-tree envelope is invalid"
    pids: set[int] = set()
    rss_sum = 0
    has_root = False
    for process in processes:
        if not isinstance(process, dict):
            return None, "process row is invalid"
        pid, member_pgid, rss = process.get("pid"), process.get("pgid"), \
            process.get("rss_bytes")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1 or pid in pids \
                or member_pgid != pgid or isinstance(rss, bool) \
                or not isinstance(rss, int) or rss < 0:
            return None, "process membership/RSS is invalid"
        pids.add(pid)
        has_root = has_root or pid == root_pid
        rss_sum += rss
    if not has_root or rss_sum != tree_rss:
        return None, "process-tree RSS is not the exact member sum"
    # The active queue records whether the aggregate pool, not this one process
    # tree, was over budget.  It must be typed but is not derivable from a
    # per-lane row and therefore is deliberately not trusted as authentication.
    if not isinstance(row.get("at_or_over_budget"), bool):
        return None, "budget classification is invalid"
    normalized = {
        "cell_id": cell_id, "profile_key": residency_profile_key(cell),
        "request_sha256": request_sha, "sampled_at": row["sampled_at"],
        "sampled_epoch": sampled_epoch, "pgid": pgid,
        "tree_rss_bytes": tree_rss, "max_tree_rss_bytes": maximum,
        "process_count": len(processes),
        "sample_sha256": _hash_value(row),
    }
    return normalized, None


def authenticated_process_tree_evidence(plan: dict[str, Any], state: dict[str, Any],
                                        rows: Iterable[Any]) -> dict[str, Any]:
    """Build a hash-sealed HWM corpus from exact plan/request/process-tree bindings."""
    _validate_plan_state(plan, state)
    cells = {row["cell_id"]: row for row in plan["cells"]}
    profiles: dict[str, dict[str, Any]] = {}
    rejected: dict[str, int] = {}
    seen: set[tuple[Any, ...]] = set()
    accepted_hashes: list[str] = []
    for raw in rows:
        sample, error = _authenticated_sample(
            raw, plan_sha256=plan["plan_sha256"], cells=cells,
            state_rows=state["cells"],
        )
        if sample is None:
            rejected[error or "unknown"] = rejected.get(error or "unknown", 0) + 1
            continue
        identity = (sample["cell_id"], sample["request_sha256"],
                    sample["sampled_at"], sample["pgid"], sample["tree_rss_bytes"])
        if identity in seen:
            rejected["duplicate sample identity"] = \
                rejected.get("duplicate sample identity", 0) + 1
            continue
        seen.add(identity)
        accepted_hashes.append(sample["sample_sha256"])
        profile = profiles.setdefault(sample["profile_key"], {
            "sample_count": 0, "cell_ids": set(), "request_sha256s": set(),
            "first_sample_epoch": sample["sampled_epoch"],
            "last_sample_epoch": sample["sampled_epoch"],
            "high_water_bytes": 0, "sample_sha256s": [],
        })
        profile["sample_count"] += 1
        profile["cell_ids"].add(sample["cell_id"])
        profile["request_sha256s"].add(sample["request_sha256"])
        profile["first_sample_epoch"] = min(profile["first_sample_epoch"],
                                             sample["sampled_epoch"])
        profile["last_sample_epoch"] = max(profile["last_sample_epoch"],
                                            sample["sampled_epoch"])
        profile["high_water_bytes"] = max(profile["high_water_bytes"],
                                           sample["tree_rss_bytes"])
        profile["sample_sha256s"].append(sample["sample_sha256"])
    finalized: dict[str, dict[str, Any]] = {}
    for key, profile in sorted(profiles.items()):
        span = profile["last_sample_epoch"] - profile["first_sample_epoch"]
        high = profile["high_water_bytes"]
        margin = max(RSS_MARGIN_FLOOR_BYTES, math.ceil(high * RSS_MARGIN_RATIO))
        reserve = min(ADMISSION_CEILING_BYTES, _round_up(high + margin))
        calibrated = profile["sample_count"] >= MIN_AUTHENTICATED_SAMPLES \
            and span >= MIN_SAMPLE_SPAN_SECONDS and high < ADMISSION_CEILING_BYTES
        finalized[key] = {
            "sample_count": profile["sample_count"],
            "cell_ids": sorted(profile["cell_ids"]),
            "request_sha256s": sorted(profile["request_sha256s"]),
            "first_sample_epoch": profile["first_sample_epoch"],
            "last_sample_epoch": profile["last_sample_epoch"],
            "sample_span_seconds": round(span, 3),
            "high_water_bytes": high,
            "deterministic_margin_bytes": margin,
            "reservation_bytes": reserve if calibrated else ADMISSION_CEILING_BYTES,
            "calibrated": calibrated,
            "sample_sha256s": sorted(profile["sample_sha256s"]),
        }
    evidence: dict[str, Any] = {
        "plan_sha256": plan["plan_sha256"],
        "state_sha256": state["state_sha256"],
        "accepted_sample_count": len(accepted_hashes),
        "rejected_sample_count": sum(rejected.values()),
        "rejected_reasons": dict(sorted(rejected.items())),
        "accepted_sample_sha256s": sorted(accepted_hashes),
        "profiles": finalized,
        "authentication_rule": (
            "exact plan + current request SHA + process budget + root/PGID membership + "
            "process RSS sum; canonical sample hashes are sealed"
        ),
    }
    evidence["evidence_sha256"] = _hash_value(evidence)
    return evidence


def reservation_for_cell(cell: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Use measured HWM+margin, or an exclusive canary; never a whole-parent guess."""
    if evidence.get("evidence_sha256") != _hash_value(
            _without(evidence, "evidence_sha256")):
        raise PolicyError("measured evidence seal is invalid")
    key = residency_profile_key(cell)
    profile = evidence.get("profiles", {}).get(key)
    calibrated = isinstance(profile, dict) and profile.get("calibrated") is True
    reservation = profile.get("reservation_bytes") if calibrated else ADMISSION_CEILING_BYTES
    if isinstance(reservation, bool) or not isinstance(reservation, int) \
            or not 1 <= reservation <= ADMISSION_CEILING_BYTES:
        raise PolicyError("measured reservation is invalid")
    return {
        "profile_key": key, "reservation_bytes": reservation,
        "exclusive_canary": not calibrated,
        "source": "authenticated-process-tree-high-water" if calibrated
                  else "unmeasured-exclusive-canary",
        "high_water_bytes": profile.get("high_water_bytes") if calibrated else None,
        "deterministic_margin_bytes": (
            profile.get("deterministic_margin_bytes") if calibrated else None
        ),
    }


def _cell_thread_identity(cell: dict[str, Any]) -> tuple[str, str, str]:
    tier, rate = cell.get("model_label"), cell.get("rate_id")
    if not isinstance(tier, str) or not tier or not isinstance(rate, str) or not rate:
        raise PolicyError("cell tier/rate thread-profile identity is invalid")
    key = json.dumps([tier, rate], separators=(",", ":"), ensure_ascii=False)
    return tier, rate, key


def _load_thread_contract() -> Any:
    path = THREAD_PROFILE_CONTRACT_PATH.resolve(strict=True)
    specification = importlib.util.spec_from_file_location(
        f"doctor_v5_thread_profile_contract_{hashlib.sha256(path.read_bytes()).hexdigest()}",
        path,
    )
    if specification is None or specification.loader is None:
        raise PolicyError("cannot load the vendor thread-profile contract")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if tuple(getattr(module, "DEFAULT_THREADS", ())) != REQUIRED_THREAD_PARITY:
        raise PolicyError("vendor contract candidate set is not exact 8/12/16/20")
    return module


def qualify_thread_profile(cells: Iterable[dict[str, Any]], *,
                           profile_path: Path | None,
                           binary_path: Path | None) -> dict[str, Any]:
    """Revalidate and hash-bind one exact vendor tier/rate profile generation."""
    contract_reference = _file_reference(THREAD_PROFILE_CONTRACT_PATH)
    result: dict[str, Any] = {
        "status": "missing", "required_threads": list(REQUIRED_THREAD_PARITY),
        "contract": contract_reference, "profile": None, "binary": None,
        "binary_sha256": None, "selections": {},
        "blockers": ["qualified vendor tier/rate thread profile is not supplied"],
    }
    if profile_path is None and binary_path is None:
        result["qualification_sha256"] = _hash_value(result)
        return result
    if profile_path is None or binary_path is None:
        raise PolicyError("thread profile and exact runtime binary must be supplied together")
    profile_path, binary_path = profile_path.resolve(strict=True), binary_path.resolve(strict=True)
    profile_reference, binary_reference = _file_reference(profile_path), \
        _file_reference(binary_path)
    result.update({"status": "partial", "profile": profile_reference,
                   "binary": binary_reference, "binary_sha256": binary_reference["sha256"],
                   "blockers": []})
    try:
        contract = _load_thread_contract()
        profile = contract.load_json(profile_path)
        if profile.get("schema") != contract.PROFILE_SCHEMA \
                or profile.get("status") != "qualified" \
                or tuple(profile.get("required_threads", ())) != REQUIRED_THREAD_PARITY:
            raise contract.ContractError(
                "profile is not a qualified exact 8/12/16/20 production generation"
            )
        if profile.get("expected_binary_sha256") != binary_reference["sha256"]:
            raise contract.ContractError("profile binary binding differs from the supplied binary")
        identities = sorted({_cell_thread_identity(cell) for cell in cells})
        selections: dict[str, Any] = {}
        for tier, rate, key in identities:
            selected = contract.verify_selection(
                profile, tier=tier, rate=rate,
                binary_sha256=binary_reference["sha256"],
            )
            entry = profile.get("entries", {}).get(key)
            if not isinstance(entry, dict):
                raise contract.ContractError("verified profile entry disappeared")
            measurements: list[dict[str, Any]] = []
            for binding in entry.get("receipt_bindings", []):
                receipt_path = Path(binding["path"]).resolve(strict=True)
                receipt = contract.validate_receipt(
                    contract.load_json(receipt_path),
                    expected_binary_sha256=binary_reference["sha256"],
                    allowed_threads=REQUIRED_THREAD_PARITY,
                )
                contract.validate_pipeline_binding(receipt, receipt_path)
                measurements.append({
                    "threads": receipt["threads"],
                    "wall_seconds": receipt["wall_seconds"],
                    "peak_rss_bytes": receipt["peak_rss_bytes"],
                    "receipt_sha256": binding["sha256"],
                })
            measurements.sort(key=lambda row: row["threads"])
            if [row["threads"] for row in measurements] != list(REQUIRED_THREAD_PARITY):
                raise contract.ContractError("qualified entry does not retain all four candidates")
            selection = {
                "tier": tier, "rate": rate,
                "selected_threads": selected["threads"],
                "selected_wall_seconds": entry["selected_wall_seconds"],
                "selected_peak_rss_bytes": entry["selected_peak_rss_bytes"],
                "scratch_budget_bytes": selected["scratch_budget_bytes"],
                "mode": selected["mode"], "source_sha256": selected["source_sha256"],
                "canonical_output_sha256": selected["canonical_output_sha256"],
                "candidate_measurements": measurements,
                "all_candidates_eligible": True,
                "selection_source": "qualified-vendor-thread-profile-contract",
            }
            selection["selection_sha256"] = _hash_value(selection)
            selections[key] = selection
        result["selections"] = selections
        result["status"] = "qualified"
    except Exception as exc:
        result["blockers"].append(f"thread profile qualification failed: {exc}")
    result["qualification_sha256"] = _hash_value(result)
    return result


def selected_thread_profile(cell: dict[str, Any],
                            qualification: dict[str, Any]) -> dict[str, Any]:
    """Consume the measured vendor winner; no tier default or override exists."""
    if qualification.get("qualification_sha256") != _hash_value(
            _without(qualification, "qualification_sha256")) \
            or qualification.get("status") != "qualified":
        raise PolicyError("thread-profile qualification is absent or invalid")
    tier, rate, key = _cell_thread_identity(cell)
    selection = qualification.get("selections", {}).get(key)
    if not isinstance(selection, dict) \
            or selection.get("selection_sha256") != _hash_value(
                _without(selection, "selection_sha256")) \
            or selection.get("tier") != tier or selection.get("rate") != rate \
            or selection.get("all_candidates_eligible") is not True \
            or [row.get("threads") for row in selection.get("candidate_measurements", [])] \
            != list(REQUIRED_THREAD_PARITY):
        raise PolicyError("cell has no exact qualified tier/rate selection")
    threads, wall = selection.get("selected_threads"), selection.get("selected_wall_seconds")
    if threads not in THREAD_PROFILE_CONTRACT \
            or isinstance(wall, bool) or not isinstance(wall, (int, float)) \
            or not math.isfinite(float(wall)) or wall <= 0:
        raise PolicyError("qualified tier/rate winner is invalid")
    contract = THREAD_PROFILE_CONTRACT[threads]
    return {
        "threads": threads, "profile": contract["name"],
        "exclusive_cpu_profile": contract["exclusive"],
        "projected_wall_seconds": float(wall),
        "selected_peak_rss_bytes": selection["selected_peak_rss_bytes"],
        "candidate_measurements": selection["candidate_measurements"],
        "exact_parity_approved": True,
        "all_four_candidates_eligible": True,
        "thread_selection_sha256": selection["selection_sha256"],
        "selection_source": selection["selection_source"],
    }


def choose_heterogeneous_pack(candidates: Iterable[dict[str, Any]], *,
                              active_reserved_bytes: int = 0,
                              active_threads: int = 0,
                              active_lanes: int = 0,
                              launch_limit: int = MAX_LANES) -> dict[str, Any]:
    """Maximize measured projected throughput under both RAM and CPU budgets.

    Candidate rows are dependency-ready exact tier/rate winners from the bound
    vendor contract.  ``sum(1 / measured_wall_seconds)`` is the primary score;
    CPU/RAM fill only breaks measured-throughput ties.  Therefore a measured
    20-thread winner remains exclusive, but is selected only when its projected
    throughput beats every valid multi-lane pack.
    """
    scalars = (active_reserved_bytes, active_threads, active_lanes, launch_limit)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in scalars):
        raise PolicyError("pack envelope is invalid")
    ram_free = ADMISSION_CEILING_BYTES - active_reserved_bytes
    cpu_free = CPU_BUDGET_CORES - active_threads
    lane_free = min(MAX_LANES - active_lanes, launch_limit)
    if ram_free <= 0 or cpu_free <= 0 or lane_free <= 0:
        return {"selected": [], "selected_cell_ids": [], "ram_free_before": ram_free,
                "cpu_free_before": cpu_free, "reason": "no launch capacity"}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        if not isinstance(row, dict):
            raise PolicyError("pack candidate is invalid")
        cell_id, priority = row.get("cell_id"), row.get("priority")
        reserve, threads = row.get("reservation_bytes"), row.get("threads")
        wall = row.get("projected_wall_seconds")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen \
                or isinstance(priority, bool) or not isinstance(priority, int) \
                or isinstance(reserve, bool) or not isinstance(reserve, int) or reserve <= 0 \
                or isinstance(threads, bool) or not isinstance(threads, int) \
                or threads not in THREAD_PROFILE_CONTRACT \
                or row.get("exact_parity_approved") is not True \
                or row.get("all_four_candidates_eligible") is not True \
                or row.get("selection_source") \
                != "qualified-vendor-thread-profile-contract" \
                or isinstance(wall, bool) or not isinstance(wall, (int, float)) \
                or not math.isfinite(float(wall)) or float(wall) <= 0:
            raise PolicyError(f"pack candidate is not exact/profile bound: {cell_id}")
        if row.get("exclusive_cpu_profile", threads == 20) is not (threads == 20):
            raise PolicyError(f"pack candidate exclusivity is inconsistent: {cell_id}")
        seen.add(cell_id)
        rows.append(dict(row))
    rows.sort(key=lambda row: (row["priority"], row["cell_id"]))
    rows = rows[:MAX_PACK_CANDIDATES]
    best: tuple[tuple[Any, ...], tuple[dict[str, Any], ...]] | None = None
    for count in range(1, min(lane_free, len(rows)) + 1):
        for combination in itertools.combinations(rows, count):
            ram = sum(row["reservation_bytes"] for row in combination)
            cpu = sum(row["threads"] for row in combination)
            if any(row["threads"] == 20 for row in combination) and len(combination) != 1:
                continue
            if ram > ram_free or cpu > cpu_free:
                continue
            throughput = sum(
                (Fraction(1, 1) / Fraction(str(row["projected_wall_seconds"])))
                for row in combination
            )
            wave_seconds = max(Fraction(str(row["projected_wall_seconds"]))
                               for row in combination)
            # Measured throughput dominates. Shorter projected wave, CPU/RAM
            # utilization, and immutable priority only break exact ties.
            score = (throughput, -wave_seconds, cpu, ram, count,
                     -sum(row["priority"] for row in combination))
            ids = tuple(row["cell_id"] for row in combination)
            if best is None or score > best[0][0] \
                    or (score == best[0][0] and ids < best[0][1]):
                best = ((score, ids), combination)
    selected = list(best[1]) if best is not None else []
    selected.sort(key=lambda row: (row["priority"], row["cell_id"]))
    selected_throughput = sum(
        (Fraction(1, 1) / Fraction(str(row["projected_wall_seconds"])))
        for row in selected
    ) if selected else Fraction(0, 1)
    return {
        "selected": selected,
        "selected_cell_ids": [row["cell_id"] for row in selected],
        "ram_free_before": ram_free, "cpu_free_before": cpu_free,
        "reserved_after_bytes": active_reserved_bytes
            + sum(row["reservation_bytes"] for row in selected),
        "threads_after": active_threads + sum(row["threads"] for row in selected),
        "heterogeneous_16_plus_8": sorted(row["threads"] for row in selected) == [8, 16],
        "projected_throughput_cells_per_second": float(selected_throughput),
        "projected_wave_seconds": (
            max(float(row["projected_wall_seconds"]) for row in selected)
            if selected else None
        ),
        "selection_basis": "measured-sum-inverse-wall-seconds",
        "reason": "best measured-throughput exact subset" if selected
                  else "no candidate fits",
    }


def swap_policy() -> dict[str, Any]:
    return {
        "soft_growth_mb": SWAP_SOFT_GROWTH_MB,
        "hard_growth_mb": SWAP_HARD_GROWTH_MB,
        "emergency_growth_mb": SWAP_EMERGENCY_GROWTH_MB,
        "absolute_emergency_mb": SWAP_ABSOLUTE_EMERGENCY_MB,
        "soft_rate_mb_min": SWAP_SOFT_RATE_MB_MIN,
        "hard_rate_mb_min": SWAP_HARD_RATE_MB_MIN,
        "soft_recovery_samples": SWAP_SOFT_RECOVERY_SAMPLES,
        "hard_recovery_samples": SWAP_HARD_RECOVERY_SAMPLES,
        "soft_cooldown_seconds": SWAP_SOFT_COOLDOWN_SECONDS,
        "hard_cooldown_seconds": SWAP_HARD_COOLDOWN_SECONDS,
        "shed_cooldown_seconds": SWAP_SHED_COOLDOWN_SECONDS,
    }


def _swap_state_payload(*, baseline_swap_mb: float, mode: str,
                        previous_swap_mb: float, previous_sample_epoch: float,
                        green_streak: int, hard_until_epoch: float,
                        last_transition_epoch: float, last_shed_epoch: float | None,
                        recovered_from_invalid: bool) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": SWAP_STATE_SCHEMA, "version": VERSION,
        "baseline_swap_mb": round(float(baseline_swap_mb), 3), "mode": mode,
        "previous_swap_mb": round(float(previous_swap_mb), 3),
        "previous_sample_epoch": float(previous_sample_epoch),
        "green_streak": green_streak, "hard_until_epoch": float(hard_until_epoch),
        "last_transition_epoch": float(last_transition_epoch),
        "last_shed_epoch": (None if last_shed_epoch is None else float(last_shed_epoch)),
        "recovered_from_invalid": recovered_from_invalid,
    }
    value["state_sha256"] = _hash_value(value)
    return value


def validate_swap_state(state: Any, *, sealed_baseline_swap_mb: float) -> list[str]:
    errors: list[str] = []
    required = {
        "schema", "version", "baseline_swap_mb", "mode", "previous_swap_mb",
        "previous_sample_epoch", "green_streak", "hard_until_epoch",
        "last_transition_epoch", "last_shed_epoch", "recovered_from_invalid",
        "state_sha256",
    }
    if not isinstance(state, dict) or set(state) != required:
        return ["swap state keys are invalid"]
    if state.get("schema") != SWAP_STATE_SCHEMA or state.get("version") != VERSION:
        errors.append("swap state schema/version mismatch")
    if state.get("state_sha256") != _hash_value(_without(state, "state_sha256")):
        errors.append("swap state hash mismatch")
    if state.get("baseline_swap_mb") != round(float(sealed_baseline_swap_mb), 3):
        errors.append("swap baseline differs from the sealed promotion baseline")
    if state.get("mode") not in {"green", "soft_throttle", "hard_stop",
                                  "emergency_shed"}:
        errors.append("swap mode is invalid")
    numeric = ("previous_swap_mb", "previous_sample_epoch", "hard_until_epoch",
               "last_transition_epoch")
    if any(isinstance(state.get(key), bool) or not isinstance(state.get(key), (int, float))
           or not math.isfinite(float(state[key])) or float(state[key]) < 0
           for key in numeric):
        errors.append("swap state numeric envelope is invalid")
    streak = state.get("green_streak")
    if isinstance(streak, bool) or not isinstance(streak, int) or streak < 0:
        errors.append("swap recovery streak is invalid")
    shed = state.get("last_shed_epoch")
    if shed is not None and (isinstance(shed, bool) or not isinstance(shed, (int, float))
                             or not math.isfinite(float(shed)) or shed < 0):
        errors.append("swap shed timestamp is invalid")
    if state.get("recovered_from_invalid") not in {True, False}:
        errors.append("swap recovery marker is invalid")
    return errors


def initial_swap_state(snapshot: dict[str, Any], *, now_epoch: float) -> dict[str, Any]:
    pressure, swap = snapshot.get("pressure_level"), snapshot.get("swap_used_mb")
    if pressure != 1 or isinstance(swap, bool) or not isinstance(swap, (int, float)) \
            or not math.isfinite(float(swap)) or swap < 0 \
            or float(swap) >= SWAP_ABSOLUTE_EMERGENCY_MB:
        raise PolicyError("promotion requires a finite, normal-pressure swap baseline")
    return _swap_state_payload(
        baseline_swap_mb=float(swap), mode="green", previous_swap_mb=float(swap),
        previous_sample_epoch=now_epoch, green_streak=0, hard_until_epoch=now_epoch,
        last_transition_epoch=now_epoch, last_shed_epoch=None,
        recovered_from_invalid=False,
    )


def advance_swap_state(state: Any, snapshot: dict[str, Any], *, now_epoch: float,
                       sealed_baseline_swap_mb: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bounded shock absorber: throttle, stop launch, then shed one checkpoint lane.

    Invalid state self-heals from the separately sealed baseline into a hard-stop
    cooldown.  It never re-baselines to a larger post-crash swap value.
    """
    if isinstance(now_epoch, bool) or not isinstance(now_epoch, (int, float)) \
            or not math.isfinite(float(now_epoch)) or now_epoch < 0:
        raise PolicyError("swap transition time is invalid")
    errors = validate_swap_state(state, sealed_baseline_swap_mb=sealed_baseline_swap_mb)
    pressure, raw_swap = snapshot.get("pressure_level"), snapshot.get("swap_used_mb")
    probe_valid = pressure in {1, 2, 4} and not isinstance(raw_swap, bool) \
        and isinstance(raw_swap, (int, float)) and math.isfinite(float(raw_swap)) \
        and float(raw_swap) >= 0
    if errors:
        observed = float(raw_swap) if probe_valid else float(sealed_baseline_swap_mb)
        healed = _swap_state_payload(
            baseline_swap_mb=sealed_baseline_swap_mb, mode="hard_stop",
            previous_swap_mb=observed, previous_sample_epoch=float(now_epoch),
            green_streak=0, hard_until_epoch=float(now_epoch) + SWAP_HARD_COOLDOWN_SECONDS,
            last_transition_epoch=float(now_epoch), last_shed_epoch=None,
            recovered_from_invalid=True,
        )
        return healed, {
            "mode": "hard_stop", "allow_launch": False, "launch_limit": 0,
            "cpu_scale": 0.0, "shed_one": False,
            "reason": "invalid controller state self-healed fail-closed",
            "state_errors": errors, "probe_valid": probe_valid,
        }
    assert isinstance(state, dict)
    previous_mode = state["mode"]
    if not probe_valid:
        mode, reason, swap, rate = "hard_stop", "resource probe invalid", \
            state["previous_swap_mb"], None
    else:
        swap = float(raw_swap)
        elapsed = max(1.0, float(now_epoch) - float(state["previous_sample_epoch"]))
        rate = max(0.0, swap - float(state["previous_swap_mb"])) * 60.0 / elapsed
        growth = swap - float(sealed_baseline_swap_mb)
        if pressure == 4 or growth >= SWAP_EMERGENCY_GROWTH_MB \
                or swap >= SWAP_ABSOLUTE_EMERGENCY_MB:
            mode, reason = "emergency_shed", "critical pressure or emergency swap bound"
        elif pressure == 2 or growth >= SWAP_HARD_GROWTH_MB \
                or rate >= SWAP_HARD_RATE_MB_MIN:
            mode, reason = "hard_stop", "warning pressure or hard swap bound"
        elif growth >= SWAP_SOFT_GROWTH_MB or rate >= SWAP_SOFT_RATE_MB_MIN:
            mode, reason = "soft_throttle", "soft swap growth/rate bound"
        else:
            mode, reason = "green", "normal pressure inside swap envelope"
    green_streak = state["green_streak"] + 1 if mode == "green" else 0
    hard_until = float(state["hard_until_epoch"])
    if mode in {"hard_stop", "emergency_shed"}:
        hard_until = max(hard_until, float(now_epoch) + SWAP_HARD_COOLDOWN_SECONDS)
    elif mode == "soft_throttle":
        hard_until = max(hard_until, float(now_epoch) + SWAP_SOFT_COOLDOWN_SECONDS)
    if mode == "green" and previous_mode in {"hard_stop", "emergency_shed"} \
            and (green_streak < SWAP_HARD_RECOVERY_SAMPLES or now_epoch < hard_until):
        mode, reason = "hard_stop", "hard-stop hysteresis/cooldown"
    elif mode == "green" and previous_mode == "soft_throttle" \
            and (green_streak < SWAP_SOFT_RECOVERY_SAMPLES or now_epoch < hard_until):
        mode, reason = "soft_throttle", "soft-throttle hysteresis/cooldown"
    last_shed = state["last_shed_epoch"]
    shed_one = mode == "emergency_shed" and (
        last_shed is None or float(now_epoch) - float(last_shed) >= SWAP_SHED_COOLDOWN_SECONDS
    )
    if shed_one:
        last_shed = float(now_epoch)
    transition = float(now_epoch) if mode != previous_mode else state["last_transition_epoch"]
    updated = _swap_state_payload(
        baseline_swap_mb=sealed_baseline_swap_mb, mode=mode,
        previous_swap_mb=float(swap), previous_sample_epoch=float(now_epoch),
        green_streak=green_streak, hard_until_epoch=hard_until,
        last_transition_epoch=transition, last_shed_epoch=last_shed,
        recovered_from_invalid=state["recovered_from_invalid"],
    )
    decision = {
        "mode": mode, "allow_launch": mode in {"green", "soft_throttle"},
        "launch_limit": MAX_LANES if mode == "green" else (1 if mode == "soft_throttle" else 0),
        "cpu_scale": 1.0 if mode == "green" else (0.5 if mode == "soft_throttle" else 0.0),
        "shed_one": shed_one, "reason": reason, "probe_valid": probe_valid,
        "swap_growth_mb": (round(float(swap) - float(sealed_baseline_swap_mb), 3)
                           if probe_valid else None),
        "swap_rate_mb_min": round(rate, 3) if isinstance(rate, float) else None,
        "green_streak": green_streak, "hard_until_epoch": hard_until,
        "running_evidence_invalidated": False,
        "emergency_action": (
            "checkpoint/receipt then shed one largest-RSS lane" if shed_one else None
        ),
    }
    return updated, decision


def _load_resource_rows(path: Path) -> list[Any]:
    rows: list[Any] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append(None)
    except (OSError, UnicodeError) as exc:
        raise PolicyError(f"cannot read process-tree evidence: {exc}") from exc
    return rows


def _pending_profile_rows(plan: dict[str, Any], state: dict[str, Any],
                          evidence: dict[str, Any],
                          qualification: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cell in plan["cells"]:
        status = state["cells"][cell["cell_id"]]["status"]
        if status in {"complete", "negative", "unsupported"}:
            continue
        reservation = reservation_for_cell(cell, evidence)
        if cell.get("model_family") == "gpt-oss-moe":
            threads = {
                "threads": None, "profile": None, "exclusive_cpu_profile": None,
                "projected_wall_seconds": None, "selected_peak_rss_bytes": None,
                "candidate_measurements": [], "exact_parity_approved": False,
                "all_four_candidates_eligible": False,
                "thread_selection_sha256": None, "selection_source": None,
                "thread_profile_blocker": (
                    "GPT-OSS retains its separately reviewed source-bound thread contract"
                ),
            }
        else:
            try:
                threads = selected_thread_profile(cell, qualification)
            except PolicyError as exc:
                threads = {
                    "threads": None, "profile": None,
                    "exclusive_cpu_profile": None,
                    "projected_wall_seconds": None,
                    "selected_peak_rss_bytes": None,
                    "candidate_measurements": [], "exact_parity_approved": False,
                    "all_four_candidates_eligible": False,
                    "thread_selection_sha256": None, "selection_source": None,
                    "thread_profile_blocker": str(exc),
                }
        output.append({
            "cell_id": cell["cell_id"], "priority": cell["priority"],
            "model_family": cell["model_family"],
            "model_label": cell["model_label"], "rate_id": cell["rate_id"],
            "branch": cell["branch"],
            "current_status": status, **reservation, **threads,
        })
    output.sort(key=lambda row: (row["priority"], row["cell_id"]))
    return output


def build_overlay(plan: dict[str, Any], state: dict[str, Any], rows: Iterable[Any], *,
                  baseline_snapshot: dict[str, Any],
                  thread_profile_path: Path | None = None,
                  thread_binary_path: Path | None = None,
                  plan_path: Path | None = None,
                  state_path: Path | None = None) -> dict[str, Any]:
    _validate_plan_state(plan, state)
    evidence = authenticated_process_tree_evidence(plan, state, rows)
    swap_state = initial_swap_state(baseline_snapshot, now_epoch=dt.datetime.now().timestamp())
    pending_cells = [cell for cell in plan["cells"]
                     if state["cells"][cell["cell_id"]]["status"]
                     not in {"complete", "negative", "unsupported"}]
    qwen_pending_cells = [cell for cell in pending_cells
                          if cell.get("model_family") == "qwen2.5-dense"]
    thread_qualification = qualify_thread_profile(
        qwen_pending_cells, profile_path=thread_profile_path,
        binary_path=thread_binary_path,
    )
    pending = _pending_profile_rows(plan, state, evidence, thread_qualification)
    overlay: dict[str, Any] = {
        "schema": SCHEMA, "version": VERSION, "created_at": _now(),
        "mode": "unbound-pending-only-generation",
        "plan_sha256": plan["plan_sha256"], "state_sha256_at_stage": state["state_sha256"],
        "source_bindings": {
            "policy_module": _file_reference(Path(__file__)),
            "thread_profile_contract": _file_reference(THREAD_PROFILE_CONTRACT_PATH),
            "plan": _file_reference(plan_path) if plan_path is not None else None,
            "queue_state": _file_reference(state_path) if state_path is not None else None,
        },
        "resource_policy": {
            "process_budget_bytes": PROCESS_BUDGET_BYTES,
            "global_reserve_bytes": GLOBAL_RESERVE_BYTES,
            "admission_ceiling_bytes": ADMISSION_CEILING_BYTES,
            "reservation_rule": (
                "authenticated exact-profile process-tree HWM + max(2GB,15%), rounded "
                "to 512MB; uncalibrated profile is an exclusive canary"
            ),
            "cpu_budget_cores": CPU_BUDGET_CORES,
            "candidate_thread_profiles": {
                THREAD_PROFILE_CONTRACT[threads]["name"]: threads
                for threads in REQUIRED_THREAD_PARITY
            },
            "required_exact_parity_thread_counts": list(REQUIRED_THREAD_PARITY),
            "thread_selection_rule": (
                "exact per-tier/rate selected_threads from the hash-bound qualified vendor "
                "contract; no nominal-tier default or fallback"
            ),
            "pack_objective": "maximize measured sum(1/selected_wall_seconds)",
            "swap": swap_policy(),
            "sealed_swap_baseline_mb": swap_state["baseline_swap_mb"],
        },
        "evidence": evidence, "initial_swap_state": swap_state,
        "thread_profile_qualification": thread_qualification,
        "pending_profiles": pending,
        "promotion": {
            "automatic_live_mutation_permitted": False,
            "requires_quiescent_paused_or_drained_state": True,
            "requires_zero_active_children": True,
            "requires_restage_at_checkpoint": True,
            "requires_atomic_pending_runtime_generation": True,
            "requires_exact_parity_for_all_selected_thread_counts": True,
            "requires_bound_production_profiles_8_12_16_20": True,
            "requires_exact_vendor_selected_threads_per_tier_rate": True,
            "completed_evidence_mutation_permitted": False,
            "rollback": (
                "restore the pre-promotion pending-only runtime/queue generation and remove "
                "the two activation keys; never change terminal evidence"
            ),
        },
    }
    overlay["overlay_sha256"] = _hash_value(overlay)
    return overlay


def validate_overlay(overlay: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(overlay, dict):
        return ["overlay is not an object"]
    if overlay.get("schema") != SCHEMA or overlay.get("version") != VERSION \
            or overlay.get("mode") != "unbound-pending-only-generation":
        errors.append("overlay schema/version/mode mismatch")
    if overlay.get("overlay_sha256") != _hash_value(_without(overlay, "overlay_sha256")):
        errors.append("overlay canonical hash mismatch")
    policy = overlay.get("resource_policy")
    if not isinstance(policy, dict) \
            or policy.get("process_budget_bytes") != PROCESS_BUDGET_BYTES \
            or policy.get("global_reserve_bytes") != GLOBAL_RESERVE_BYTES \
            or policy.get("admission_ceiling_bytes") != ADMISSION_CEILING_BYTES \
            or policy.get("cpu_budget_cores") != CPU_BUDGET_CORES \
            or policy.get("required_exact_parity_thread_counts") \
            != list(REQUIRED_THREAD_PARITY) \
            or policy.get("thread_selection_rule") != (
                "exact per-tier/rate selected_threads from the hash-bound qualified vendor "
                "contract; no nominal-tier default or fallback"
            ) or policy.get("pack_objective") \
            != "maximize measured sum(1/selected_wall_seconds)" \
            or policy.get("swap") != swap_policy():
        errors.append("overlay resource policy differs from the reviewed envelope")
    evidence = overlay.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("evidence_sha256") \
            != _hash_value(_without(evidence, "evidence_sha256")):
        errors.append("overlay measured evidence seal is invalid")
    qualification = overlay.get("thread_profile_qualification")
    if not isinstance(qualification, dict) \
            or qualification.get("qualification_sha256") \
            != _hash_value(_without(qualification, "qualification_sha256")) \
            or qualification.get("required_threads") != list(REQUIRED_THREAD_PARITY):
        errors.append("overlay thread-profile qualification seal is invalid")
    promotion = overlay.get("promotion")
    if not isinstance(promotion, dict) \
            or promotion.get("automatic_live_mutation_permitted") is not False \
            or promotion.get("requires_quiescent_paused_or_drained_state") is not True \
            or promotion.get("requires_zero_active_children") is not True \
            or promotion.get("requires_restage_at_checkpoint") is not True \
            or promotion.get("requires_atomic_pending_runtime_generation") is not True \
            or promotion.get("requires_exact_parity_for_all_selected_thread_counts") is not True \
            or promotion.get("requires_bound_production_profiles_8_12_16_20") is not True \
            or promotion.get("requires_exact_vendor_selected_threads_per_tier_rate") is not True \
            or promotion.get("completed_evidence_mutation_permitted") is not False:
        errors.append("overlay promotion contract is not fail-closed")
    pending = overlay.get("pending_profiles")
    if not isinstance(pending, list) or len({row.get("cell_id") for row in pending
                                            if isinstance(row, dict)}) != len(pending):
        errors.append("pending profile generation is invalid")
    return errors


def promotion_gate(overlay: dict[str, Any], plan: dict[str, Any], state: dict[str, Any],
                   resource_rows: Iterable[Any], *, snapshot: dict[str, Any]) \
        -> dict[str, Any]:
    blockers = validate_overlay(overlay)
    try:
        _validate_plan_state(plan, state)
    except PolicyError as exc:
        blockers.append(str(exc))
    if plan.get("plan_sha256") != overlay.get("plan_sha256"):
        blockers.append("current queue state identity differs from the overlay plan")
    try:
        current_evidence = authenticated_process_tree_evidence(plan, state, resource_rows)
    except PolicyError as exc:
        blockers.append(f"cannot revalidate measured evidence: {exc}")
    else:
        if current_evidence != overlay.get("evidence"):
            blockers.append("measured evidence differs; overlay must be re-staged")
    if state.get("state_sha256") != overlay.get("state_sha256_at_stage"):
        blockers.append("overlay must be re-staged at the exact promotion checkpoint")
    if state.get("status") not in {"paused", "drained"}:
        blockers.append("queue is not paused/drained")
    active = state.get("active_children")
    if not isinstance(active, dict) or active:
        blockers.append("active children are not empty")
    bindings = overlay.get("source_bindings")
    if not isinstance(bindings, dict) \
            or not _reference_matches(bindings.get("policy_module")):
        blockers.append("aggressive policy source binding changed")
    # Production overlays bind both checkpoint files.  Synthetic tests may omit
    # them, but such an overlay is never promotion-ready.
    for name in ("plan", "queue_state", "thread_profile_contract"):
        if not isinstance(bindings, dict) or not _reference_matches(bindings.get(name)):
            blockers.append(f"promotion source binding is absent or changed: {name}")
    qualification = overlay.get("thread_profile_qualification", {})
    pending_qwen = [
        cell for cell in plan["cells"]
        if cell.get("model_family") == "qwen2.5-dense"
        and state["cells"][cell["cell_id"]]["status"]
        not in {"complete", "negative", "unsupported"}
    ]
    if pending_qwen and qualification.get("status") != "qualified":
        blockers.append("qualified exact vendor thread profile is absent")
    elif pending_qwen:
        try:
            profile_reference = qualification["profile"]
            binary_reference = qualification["binary"]
            if not _reference_matches(profile_reference) \
                    or not _reference_matches(binary_reference):
                raise PolicyError("profile or runtime binary binding changed/escaped workspace")
            profile_path = (ROOT / profile_reference["path"]).resolve(strict=True)
            binary_path = (ROOT / binary_reference["path"]).resolve(strict=True)
            current_qualification = qualify_thread_profile(
                pending_qwen, profile_path=profile_path, binary_path=binary_path,
            )
        except (KeyError, OSError, TypeError, ValueError, PolicyError) as exc:
            blockers.append(f"cannot revalidate vendor thread profile: {exc}")
        else:
            if current_qualification != qualification:
                blockers.append("vendor thread profile generation differs from staging")
    pressure, swap = snapshot.get("pressure_level"), snapshot.get("swap_used_mb")
    baseline = overlay.get("resource_policy", {}).get("sealed_swap_baseline_mb")
    if pressure != 1 or isinstance(swap, bool) or not isinstance(swap, (int, float)) \
            or not math.isfinite(float(swap)) or not isinstance(baseline, (int, float)) \
            or float(swap) - float(baseline) >= SWAP_SOFT_GROWTH_MB:
        blockers.append("promotion resource snapshot is not inside the green swap envelope")
    return {"ready": not blockers, "blockers": blockers,
            "required_thread_profiles": list(REQUIRED_THREAD_PARITY),
            "thread_profile_status": qualification.get("status")}


def _live_snapshot() -> dict[str, Any]:
    # Import only for the existing cheap sysctl/disk/power probe.  This module is
    # not loaded by the active supervisor and does not mutate scheduler state.
    import ram_scheduler
    return ram_scheduler.resource_snapshot(str(ROOT))


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="read-only unbound policy projection")
    stage = sub.add_parser("stage", help="write only the unbound aggressive-v2 overlay")
    stage.add_argument("--output", type=Path, default=DEFAULT_OVERLAY)
    for command in (status, stage):
        command.add_argument("--thread-profile", type=Path)
        command.add_argument("--thread-binary", type=Path)
    validate = sub.add_parser("validate", help="validate a staged aggressive-v2 overlay")
    validate.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    args = parser.parse_args()
    if args.command == "validate":
        overlay = _read_json(args.overlay.resolve(strict=True))
        errors = validate_overlay(overlay)
        print(json.dumps({"overlay_sha256": overlay.get("overlay_sha256"),
                          "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    plan, state = _read_json(PLAN), _read_json(STATE)
    rows = _load_resource_rows(CHILD_RESOURCES)
    overlay = build_overlay(
        plan, state, rows, baseline_snapshot=_live_snapshot(),
        thread_profile_path=args.thread_profile,
        thread_binary_path=args.thread_binary,
        plan_path=PLAN, state_path=STATE,
    )
    if args.command == "stage":
        output = args.output.resolve()
        try:
            output.relative_to(STAGE_ROOT.resolve())
        except ValueError as exc:
            raise PolicyError("stage output must remain below aggressive_v2") from exc
        _atomic_json(output, overlay)
    summary = {
        "overlay_sha256": overlay["overlay_sha256"],
        "mode": overlay["mode"], "validation_errors": validate_overlay(overlay),
        "authenticated_samples": overlay["evidence"]["accepted_sample_count"],
        "rejected_samples": overlay["evidence"]["rejected_sample_count"],
        "calibrated_profiles": sum(
            row["calibrated"] for row in overlay["evidence"]["profiles"].values()
        ),
        "pending_cells": len(overlay["pending_profiles"]),
        "selected_thread_profiles": sorted({
            row["threads"] for row in overlay["pending_profiles"]
            if isinstance(row.get("threads"), int)
        }),
        "thread_profile_status": overlay["thread_profile_qualification"]["status"],
        "thread_profile_blockers": overlay["thread_profile_qualification"]["blockers"],
        "parity_ready": (
            overlay["thread_profile_qualification"]["status"] == "qualified"
            and all(row["exact_parity_approved"] for row in overlay["pending_profiles"])
        ),
        "promotion_ready": False,
        "promotion_reason": (
            "unbound scaffold only; supply a qualified exact vendor tier/rate profile, "
            "re-stage, and promote atomically at a quiescent checkpoint"
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except PolicyError as exc:
        print(f"doctor_v5_aggressive_admission_policy: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
