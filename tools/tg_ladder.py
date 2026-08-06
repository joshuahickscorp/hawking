#!/usr/bin/env python3.12
"""Storage-bounded, breadth-first controller for the TG latency ladder.

This deliberately does not download, quantize, benchmark, or delete a model.  It
is the campaign's fail-closed control plane: the source, raw adapter, profile,
Gravity adapter, artifact oracle, and latency result are separate gates.  That
keeps a bad forward pass from consuming a TG rung or turning into an endless
decode-kernel investigation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GIB = 1024 ** 3
PROTECTED_FILESYSTEM_FLOOR_BYTES = 200_005_889_556  # 186.27 GiB
MODEL_LANE_CEILING_BYTES = 102_000_000_000  # decimal, inclusive
NORMAL_MODEL_ARTIFACT_CAP_BYTES = 82_000_000_000  # decimal, inclusive
MIN_SCRATCH_BYTES = 15 * GIB
MAX_SCRATCH_BYTES = 25 * GIB
EXECUTION_MODES = frozenset(("direct_full_source", "dependency_complete_window"))
SCHEMA = "hawking.tg.breadth_ladder.v1"
STATE_SCHEMA = "hawking.tg.breadth_ladder_state.v1"
RUNG_ORDER = ("TG20", "TG10", "TG5", "TG4", "TG3", "TG2", "TG1")
FAMILY_ROTATION = ("qwen", "mistral", "llama", "deepseek", "qwen", "mistral", "llama")
TERMINAL_GATE_STATES = frozenset(("PASS", "FAIL", "BLOCKED"))


class LadderError(ValueError):
    """A plan or state tried to claim more than its evidence permits."""


def unified_lane_admission(
    *,
    free_bytes: int,
    model_bytes: int,
    scratch_bytes: int,
    metadata_and_ranges_bytes: int = 0,
    active_model_count: int = 0,
) -> dict[str, Any]:
    """Make one byte-exact admission decision under the unified floor.

    This is deliberately a pure check so callers must sample the filesystem
    immediately before calling it.  The model lane is an incremental cap
    inside the one filesystem floor; it is never added to a second Odyssey or
    parent reserve.
    """
    fields = {
        "free_bytes": free_bytes,
        "model_bytes": model_bytes,
        "scratch_bytes": scratch_bytes,
        "metadata_and_ranges_bytes": metadata_and_ranges_bytes,
        "active_model_count": active_model_count,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in fields.values()):
        raise LadderError("unified lane admission inputs must be non-negative integer bytes/counts")

    incremental_bytes = model_bytes + scratch_bytes + metadata_and_ranges_bytes
    residual_bytes = free_bytes - PROTECTED_FILESYSTEM_FLOOR_BYTES
    failures: list[str] = []
    if active_model_count != 0:
        failures.append("an active model/artifact is already present")
    if model_bytes == 0:
        failures.append("model bytes must be positive")
    if model_bytes > NORMAL_MODEL_ARTIFACT_CAP_BYTES:
        failures.append("model artifact exceeds the normal 82-GB decimal cap")
    if not MIN_SCRATCH_BYTES <= scratch_bytes <= MAX_SCRATCH_BYTES:
        failures.append("scratch must be within the 15–25 GiB in-envelope reserve")
    if incremental_bytes > MODEL_LANE_CEILING_BYTES:
        failures.append("model plus scratch/metadata exceeds the 102-GB decimal lane ceiling")
    if residual_bytes < incremental_bytes:
        failures.append("live residual above the protected filesystem floor is insufficient")

    return {
        "schema": "hawking.tg.unified_lane_admission.v1",
        "status": "ADMITTED" if not failures else "DENIED",
        "free_bytes": free_bytes,
        "protected_floor_bytes": PROTECTED_FILESYSTEM_FLOOR_BYTES,
        "residual_above_floor_bytes": residual_bytes,
        "model_lane_ceiling_bytes": MODEL_LANE_CEILING_BYTES,
        "normal_model_artifact_cap_bytes": NORMAL_MODEL_ARTIFACT_CAP_BYTES,
        "scratch_min_bytes": MIN_SCRATCH_BYTES,
        "scratch_max_bytes": MAX_SCRATCH_BYTES,
        "model_bytes": model_bytes,
        "scratch_bytes": scratch_bytes,
        "metadata_and_ranges_bytes": metadata_and_ranges_bytes,
        "incremental_lane_bytes": incremental_bytes,
        "active_model_count": active_model_count,
        "failures": failures,
    }


def executable_working_set_admission(
    *,
    source_bytes: int,
    execution_mode: str,
    resident_weight_bytes: int,
    resident_kv_bytes: int,
    resident_activation_bytes: int,
    runtime_scratch_bytes: int,
    resident_budget_bytes: int,
    dependency_complete_window: bool = False,
    bounded_next_range_bytes: int = 0,
    active_execution_count: int = 0,
) -> dict[str, Any]:
    """Fail closed before a runtime maps or uploads an unbounded weight body.

    Disk admission and executable residency are different physical constraints.
    This pure gate makes the latter explicit: a direct source route must charge
    the complete source body, while a streaming ``.gravity`` route must name a
    dependency-complete current window and at most one bounded next range.
    The caller supplies a freshly measured device/host resident budget; this
    controller intentionally does not guess from parameter count or RAM size.
    """
    fields = {
        "source_bytes": source_bytes,
        "resident_weight_bytes": resident_weight_bytes,
        "resident_kv_bytes": resident_kv_bytes,
        "resident_activation_bytes": resident_activation_bytes,
        "runtime_scratch_bytes": runtime_scratch_bytes,
        "resident_budget_bytes": resident_budget_bytes,
        "bounded_next_range_bytes": bounded_next_range_bytes,
        "active_execution_count": active_execution_count,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in fields.values()):
        raise LadderError("executable working-set admission inputs must be non-negative integer bytes/counts")
    if execution_mode not in EXECUTION_MODES:
        raise LadderError(f"execution_mode must be one of {sorted(EXECUTION_MODES)}")

    failures: list[str] = []
    if source_bytes == 0:
        failures.append("source bytes must be positive")
    if resident_budget_bytes == 0:
        failures.append("resident budget must be positive and freshly measured")
    if active_execution_count != 0:
        failures.append("an execution working set is already active")
    if execution_mode == "direct_full_source":
        if resident_weight_bytes != source_bytes:
            failures.append("direct full-source execution must charge exactly the complete source body")
        if dependency_complete_window:
            failures.append("direct full-source execution cannot claim a dependency-complete window")
        if bounded_next_range_bytes != 0:
            failures.append("direct full-source execution cannot retain a next range")
    else:
        if not dependency_complete_window:
            failures.append("windowed execution requires a dependency-complete current window")
        if resident_weight_bytes == 0 or resident_weight_bytes >= source_bytes:
            failures.append("windowed execution must retain a non-zero proper subset of source bytes")

    required_resident_bytes = (
        resident_weight_bytes
        + resident_kv_bytes
        + resident_activation_bytes
        + runtime_scratch_bytes
        + bounded_next_range_bytes
    )
    if required_resident_bytes > resident_budget_bytes:
        failures.append("charged executable working set exceeds the freshly measured resident budget")

    return {
        "schema": "hawking.tg.executable_working_set_admission.v1",
        "status": "ADMITTED" if not failures else "DENIED",
        "execution_mode": execution_mode,
        "source_bytes": source_bytes,
        "resident_weight_bytes": resident_weight_bytes,
        "resident_kv_bytes": resident_kv_bytes,
        "resident_activation_bytes": resident_activation_bytes,
        "runtime_scratch_bytes": runtime_scratch_bytes,
        "bounded_next_range_bytes": bounded_next_range_bytes,
        "dependency_complete_window": dependency_complete_window,
        "resident_budget_bytes": resident_budget_bytes,
        "required_resident_bytes": required_resident_bytes,
        "active_execution_count": active_execution_count,
        "failures": failures,
    }


def split_gguf_merge_admission(
    *,
    free_bytes: int,
    part_bytes: Iterable[int],
    merged_bytes: int,
    scratch_bytes: int,
    metadata_and_ranges_bytes: int = 0,
    active_model_count: int = 0,
) -> dict[str, Any]:
    """Account for a split-GGUF representation change without inventing space.

    Standard GGUF split files are individually indexed GGUF containers, not
    byte ranges that may be concatenated.  An ordinary merger that retains all
    input parts until the final output exists needs ``source + output`` bytes
    and is deliberately rejected when that exceeds the unified lane.

    A separately audited streaming merger may write one verified part to the
    destination, fsync it, then evict that *exact* source part before moving to
    the next.  Its physical high-water mark is the complete source set plus
    one largest part awaiting post-write eviction.  This helper only admits
    that bounded plan; it neither authorizes an unverified merger nor deletes
    anything itself.
    """
    parts = tuple(part_bytes)
    if not parts:
        raise LadderError("split-GGUF merge admission requires at least one part size")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in parts):
        raise LadderError("split-GGUF part bytes must be positive integer bytes")
    for label, value in {
        "merged_bytes": merged_bytes,
        "scratch_bytes": scratch_bytes,
        "metadata_and_ranges_bytes": metadata_and_ranges_bytes,
        "active_model_count": active_model_count,
        "free_bytes": free_bytes,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LadderError(f"split-GGUF {label} must be a non-negative integer")
    if merged_bytes <= 0:
        raise LadderError("split-GGUF merged_bytes must be positive")

    source_bytes = sum(parts)
    largest_part_bytes = max(parts)
    # Before a part can be deleted the merger must have durably written that
    # part's payload into the target.  Count both copies at that instant.
    streaming_peak_model_bytes = source_bytes + largest_part_bytes
    traditional_peak_model_bytes = source_bytes + merged_bytes
    streaming = unified_lane_admission(
        free_bytes=free_bytes,
        model_bytes=streaming_peak_model_bytes,
        scratch_bytes=scratch_bytes,
        metadata_and_ranges_bytes=metadata_and_ranges_bytes,
        active_model_count=active_model_count,
    )
    traditional = unified_lane_admission(
        free_bytes=free_bytes,
        model_bytes=traditional_peak_model_bytes,
        scratch_bytes=scratch_bytes,
        metadata_and_ranges_bytes=metadata_and_ranges_bytes,
        active_model_count=active_model_count,
    )
    return {
        "schema": "hawking.tg.split_gguf_merge_admission.v1",
        "status": streaming["status"],
        "required_protocol": [
            "verify every source part against its immutable SHA-256 before use",
            "write and fsync one complete output part before removing that exact input part",
            "record per-part removal and free-byte recovery; abort on any mismatch",
            "verify the complete merged GGUF and its final SHA-256 before loading",
        ],
        "source_part_count": len(parts),
        "source_parts_bytes": source_bytes,
        "largest_source_part_bytes": largest_part_bytes,
        "merged_bytes": merged_bytes,
        "streaming_peak_model_bytes": streaming_peak_model_bytes,
        "traditional_peak_model_bytes": traditional_peak_model_bytes,
        "streaming": streaming,
        "traditional_copy": traditional,
    }


def live_free_bytes(path: Path) -> int:
    """Return usable bytes from the selected filesystem at admission time."""
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


@dataclass(frozen=True)
class NextAction:
    rung: str
    model: str
    family: str
    gate: str
    action: str

    def to_json(self) -> dict[str, str]:
        return {
            "rung": self.rung,
            "model": self.model,
            "family": self.family,
            "gate": self.gate,
            "action": self.action,
        }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_plan_path() -> Path:
    return repo_root() / "workspace/campaign/evidence/runtime/tg/TG_BREADTH_LADDER.json"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LadderError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LadderError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LadderError(f"{path} must contain a JSON object")
    return value


def empty_state() -> dict[str, Any]:
    return {"schema": STATE_SCHEMA, "rungs": {}}


def load_state(path: Path | None) -> dict[str, Any]:
    if path is None:
        return empty_state()
    state = load_json(path)
    if state.get("schema") != STATE_SCHEMA:
        raise LadderError(
            f"state schema must be {STATE_SCHEMA!r}, got {state.get('schema')!r}"
        )
    if not isinstance(state.get("rungs"), dict):
        raise LadderError("state.rungs must be an object keyed by TG rung")
    return state


def plan_rungs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rungs = plan.get("rungs")
    if not isinstance(rungs, list):
        raise LadderError("plan.rungs must be a list")
    return rungs


def _require_string(item: dict[str, Any], key: str, where: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise LadderError(f"{where}.{key} must be a non-empty string")
    return value


def validate_plan(plan: dict[str, Any]) -> None:
    """Validate the breadth, storage, and no-false-promotion contract."""
    if plan.get("schema") != SCHEMA:
        raise LadderError(f"plan schema must be {SCHEMA!r}")

    latency = plan.get("rung_latency_ms")
    if not isinstance(latency, dict) or tuple(latency) != RUNG_ORDER:
        raise LadderError(f"rung_latency_ms must be ordered exactly as {RUNG_ORDER}")
    if any(not isinstance(latency[rung], int) or latency[rung] <= 0 for rung in RUNG_ORDER):
        raise LadderError("every rung latency must be a positive integer millisecond budget")

    storage = plan.get("storage_contract")
    if not isinstance(storage, dict):
        raise LadderError("storage_contract must be an object")
    max_parent = storage.get("max_active_parent_gib")
    max_gravity = storage.get("max_retained_gravity_artifact_gib")
    if not isinstance(max_parent, int) or max_parent <= 0:
        raise LadderError("storage_contract.max_active_parent_gib must be a positive integer")
    if not isinstance(max_gravity, int) or max_gravity <= 0:
        raise LadderError("storage_contract.max_retained_gravity_artifact_gib must be a positive integer")

    required_gates = plan.get("required_gates")
    if not isinstance(required_gates, list) or len(required_gates) < 2:
        raise LadderError("required_gates must name the complete gate sequence")
    if len(set(required_gates)) != len(required_gates):
        raise LadderError("required_gates cannot contain duplicates")
    for required in ("RAW_PARENT_FORWARD", "STATE_ADVANCE", "ADAPTER_ACCOUNTING", "AUTOTUNE_PROFILE", "GRAVITY_ADAPTER", "GRAVITY_ORACLE", "RUNG_MEASUREMENT", "EVICTION_RECEIPT"):
        if required not in required_gates:
            raise LadderError(f"required_gates is missing {required}")

    rungs = plan_rungs(plan)
    if len(rungs) != len(RUNG_ORDER):
        raise LadderError(f"plan must contain exactly {len(RUNG_ORDER)} rungs")
    seen_models: set[str] = set()
    for index, rung in enumerate(rungs):
        where = f"rungs[{index}]"
        if not isinstance(rung, dict):
            raise LadderError(f"{where} must be an object")
        if _require_string(rung, "rung", where) != RUNG_ORDER[index]:
            raise LadderError(f"{where}.rung must be {RUNG_ORDER[index]}")
        if _require_string(rung, "family", where) != FAMILY_ROTATION[index]:
            raise LadderError(
                f"{where}.family must follow breadth rotation {FAMILY_ROTATION[index]!r}"
            )
        model = _require_string(rung, "model", where)
        # A rehydration is deliberately named and can only appear after a complete
        # Qwen/Mistral/Llama/DeepSeek rotation.  It prevents raw weights from being
        # retained merely because the same family gets another representation trial.
        normalized_model = model.removesuffix(" (rehydrated)")
        if normalized_model in seen_models and not model.endswith(" (rehydrated)"):
            raise LadderError(f"{where}.model repeats {normalized_model!r} without rehydration")
        if model.endswith(" (rehydrated)") and index < len(set(FAMILY_ROTATION)):
            raise LadderError(f"{where}.model rehydrates before one full family rotation")
        seen_models.add(normalized_model)

        source_budget = rung.get("source_budget_gib")
        artifact_budget = rung.get("gravity_artifact_budget_gib")
        if not isinstance(source_budget, int) or not 0 < source_budget <= max_parent:
            raise LadderError(f"{where}.source_budget_gib must be within 1..{max_parent}")
        if not isinstance(artifact_budget, int) or not 0 < artifact_budget <= max_gravity:
            raise LadderError(f"{where}.gravity_artifact_budget_gib must be within 1..{max_gravity}")
        _require_string(rung, "quant", where)
        _require_string(rung, "local_candidate", where)
        adapter = rung.get("adapter")
        if not isinstance(adapter, dict):
            raise LadderError(f"{where}.adapter must be an object")
        _require_string(adapter, "raw_engine", f"{where}.adapter")
        _require_string(adapter, "gravity_engine", f"{where}.adapter")
        available = adapter.get("available_tunables")
        missing = adapter.get("missing_tunables")
        if not isinstance(available, list) or not isinstance(missing, list):
            raise LadderError(f"{where}.adapter tunables must be lists")
        if not available and not missing:
            raise LadderError(f"{where}.adapter must name either an available or missing tuning surface")


def _state_row(state: dict[str, Any], rung: str) -> dict[str, Any]:
    row = state["rungs"].get(rung, {})
    if not isinstance(row, dict):
        raise LadderError(f"state.rungs.{rung} must be an object")
    gates = row.get("gates", {})
    if not isinstance(gates, dict):
        raise LadderError(f"state.rungs.{rung}.gates must be an object")
    return row


def validate_state(plan: dict[str, Any], state: dict[str, Any]) -> None:
    """Reject unsupported TG claims and source retention beyond one active parent."""
    validate_plan(plan)
    required = tuple(plan["required_gates"])
    valid_rungs = set(RUNG_ORDER)
    unknown_rungs = set(state["rungs"]) - valid_rungs
    if unknown_rungs:
        raise LadderError(f"state names unknown rungs: {sorted(unknown_rungs)}")

    active_sources = 0
    for rung in RUNG_ORDER:
        row = _state_row(state, rung)
        gates = row.get("gates", {})
        for gate, result in gates.items():
            if gate not in required:
                raise LadderError(f"state.rungs.{rung} names unknown gate {gate!r}")
            if result not in TERMINAL_GATE_STATES:
                raise LadderError(
                    f"state.rungs.{rung}.gates.{gate} must be one of {sorted(TERMINAL_GATE_STATES)}"
                )
        if row.get("source_materialized") is True and gates.get("EVICTION_RECEIPT") != "PASS":
            active_sources += 1
        claim = row.get("claim")
        if claim is not None:
            if claim not in ("MEASURED", "NO_CLAIM"):
                raise LadderError(f"state.rungs.{rung}.claim must be MEASURED or NO_CLAIM")
            if claim == "MEASURED" and any(gates.get(gate) != "PASS" for gate in required):
                missing = [gate for gate in required if gates.get(gate) != "PASS"]
                raise LadderError(
                    f"state.rungs.{rung} claims MEASURED without PASS gates: {missing}"
                )
        bytes_seen = row.get("source_bytes")
        if bytes_seen is not None:
            if not isinstance(bytes_seen, int) or bytes_seen <= 0:
                raise LadderError(f"state.rungs.{rung}.source_bytes must be a positive integer")
            budget = next(r for r in plan_rungs(plan) if r["rung"] == rung)["source_budget_gib"] * GIB
            if bytes_seen > budget:
                raise LadderError(
                    f"state.rungs.{rung}.source_bytes exceeds its {budget // GIB} GiB storage budget"
                )
    if active_sources > 1:
        raise LadderError(
            f"storage policy permits one materialized parent, but state has {active_sources}"
        )


def next_actions(plan: dict[str, Any], state: dict[str, Any]) -> list[NextAction]:
    """Return the first unresolved gate per rung; no kernel work is implied."""
    validate_state(plan, state)
    gate_actions = {
        "SOURCE_SEALED": "resolve one exact quantized source, record revision + bytes + sha256, and materialize it as the only active parent",
        "RAW_PARENT_FORWARD": "capture an external-parent token/logit oracle, then run Hawking against the exact source",
        "STATE_ADVANCE": "run a multi-token exact-profile decode; record the token sequence so repetition cannot be called a latency result",
        "ADAPTER_ACCOUNTING": "record device identity, command-buffer count, and dispatches per forward; an unset counter is a failed accounting gate",
        "AUTOTUNE_PROFILE": "run the family’s declared tuning matrix and seal a profile bound to the exact tensor layout and shader hash",
        "GRAVITY_ADAPTER": "implement or validate the family-specific Gravity header, tensor map, cache layout, and callable forward before packing a campaign artifact",
        "GRAVITY_ORACLE": "run the packed artifact against its own independent oracle and seal complete-BPW plus source coverage",
        "RUNG_MEASUREMENT": "on a quiet machine, benchmark correct raw and Gravity paths against this rung’s latency budget",
        "EVICTION_RECEIPT": "seal terminal result and source identity, then evict the parent; retain only receipts and a qualifying Gravity artifact",
    }
    out: list[NextAction] = []
    for rung in plan_rungs(plan):
        row = _state_row(state, rung["rung"])
        gates = row.get("gates", {})
        # Correctness failures are terminal for a *rung*, not invitations to
        # keep its parent on disk and tunnel further into a kernel.  The
        # receipt preserves the adapter result; later repair work can
        # deliberately rehydrate the named source under a fresh state file.
        fatal = next(
            (
                gate
                for gate in ("RAW_PARENT_FORWARD", "STATE_ADVANCE", "ADAPTER_ACCOUNTING", "GRAVITY_ADAPTER", "GRAVITY_ORACLE")
                if gates.get(gate) in ("FAIL", "BLOCKED")
            ),
            None,
        )
        if fatal is not None:
            if gates.get("EVICTION_RECEIPT") != "PASS":
                out.append(
                    NextAction(
                        rung["rung"],
                        rung["model"],
                        rung["family"],
                        "EVICTION_RECEIPT",
                        f"{gate_actions['EVICTION_RECEIPT']} The terminal gate is {fatal}={gates[fatal]}.",
                    )
                )
            continue
        for gate in plan["required_gates"]:
            if gates.get(gate) != "PASS":
                out.append(
                    NextAction(rung["rung"], rung["model"], rung["family"], gate, gate_actions[gate])
                )
                break
    return out


def status(plan: dict[str, Any], state: dict[str, Any], root: Path) -> dict[str, Any]:
    validate_state(plan, state)
    rows = []
    for rung in plan_rungs(plan):
        state_row = _state_row(state, rung["rung"])
        path = root / rung["local_candidate"]
        source_bytes = path.stat().st_size if path.is_file() else None
        rows.append(
            {
                "rung": rung["rung"],
                "model": rung["model"],
                "family": rung["family"],
                "source_budget_gib": rung["source_budget_gib"],
                "gravity_artifact_budget_gib": rung["gravity_artifact_budget_gib"],
                "local_candidate": str(path),
                "local_source_bytes": source_bytes,
                "source_status": rung["source_status"],
                "gates": state_row.get("gates", {}),
                "claim": state_row.get("claim", "NO_CLAIM"),
            }
        )
    return {
        "schema": SCHEMA,
        "status": plan["status"],
        "storage_contract": plan["storage_contract"],
        "rungs": rows,
        "next_actions": [action.to_json() for action in next_actions(plan, state)],
    }


def render_actions(actions: Iterable[NextAction]) -> str:
    lines = []
    for item in actions:
        lines.append(f"{item.rung} | {item.family} | {item.model} | {item.gate}: {item.action}")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("validate", "status", "next", "admission", "execution-admission", "split-merge-admission"),
    )
    parser.add_argument("--plan", type=Path, default=default_plan_path())
    parser.add_argument("--state", type=Path, help="optional sealed campaign state JSON")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument("--model-bytes", type=int, help="selected model artifact bytes for admission")
    parser.add_argument("--scratch-bytes", type=int, help="reserved scratch/rollback bytes for admission")
    parser.add_argument("--metadata-and-ranges-bytes", type=int, default=0, help="bounded metadata/range bytes for admission")
    parser.add_argument("--active-model-count", type=int, default=0, help="currently materialized full model/artifact count")
    parser.add_argument("--free-bytes", type=int, help="test-only sampled free bytes; omit to sample the plan filesystem")
    parser.add_argument("--source-bytes", type=int, help="complete immutable source body bytes for executable working-set admission")
    parser.add_argument("--execution-mode", choices=sorted(EXECUTION_MODES), help="direct full source or dependency-complete window execution")
    parser.add_argument("--resident-weight-bytes", type=int, help="current executable weight bytes resident/mapped")
    parser.add_argument("--resident-kv-bytes", type=int, default=0, help="current KV/state resident bytes")
    parser.add_argument("--resident-activation-bytes", type=int, default=0, help="current activation resident bytes")
    parser.add_argument("--runtime-scratch-bytes", type=int, default=0, help="current runtime scratch resident bytes")
    parser.add_argument("--resident-budget-bytes", type=int, help="fresh measured resident working-set budget")
    parser.add_argument("--dependency-complete-window", action="store_true", help="attest the current window has every dependency for its scheduled work")
    parser.add_argument("--bounded-next-range-bytes", type=int, default=0, help="one verified next range retained ahead of the current window")
    parser.add_argument("--active-execution-count", type=int, default=0, help="currently active executable working-set count")
    parser.add_argument(
        "--part-bytes",
        type=int,
        action="append",
        help="one immutable split-GGUF source-part size; repeat for every part",
    )
    parser.add_argument(
        "--merged-bytes",
        type=int,
        help="expected size of the verified single-file GGUF output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.command == "execution-admission":
            required = {
                "--source-bytes": args.source_bytes,
                "--execution-mode": args.execution_mode,
                "--resident-weight-bytes": args.resident_weight_bytes,
                "--resident-budget-bytes": args.resident_budget_bytes,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise LadderError("execution-admission requires " + ", ".join(missing))
            result = executable_working_set_admission(
                source_bytes=args.source_bytes,
                execution_mode=args.execution_mode,
                resident_weight_bytes=args.resident_weight_bytes,
                resident_kv_bytes=args.resident_kv_bytes,
                resident_activation_bytes=args.resident_activation_bytes,
                runtime_scratch_bytes=args.runtime_scratch_bytes,
                resident_budget_bytes=args.resident_budget_bytes,
                dependency_complete_window=args.dependency_complete_window,
                bounded_next_range_bytes=args.bounded_next_range_bytes,
                active_execution_count=args.active_execution_count,
            )
            if result["status"] != "ADMITTED":
                raise LadderError("executable working-set admission denied: " + "; ".join(result["failures"]))
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print("TG executable working set: ADMITTED")
            return 0
        if args.command == "admission":
            if args.model_bytes is None or args.scratch_bytes is None:
                raise LadderError("admission requires --model-bytes and --scratch-bytes")
            sampled_free = args.free_bytes if args.free_bytes is not None else live_free_bytes(repo_root())
            result = unified_lane_admission(
                free_bytes=sampled_free,
                model_bytes=args.model_bytes,
                scratch_bytes=args.scratch_bytes,
                metadata_and_ranges_bytes=args.metadata_and_ranges_bytes,
                active_model_count=args.active_model_count,
            )
            if result["status"] != "ADMITTED":
                raise LadderError("unified model-lane admission denied: " + "; ".join(result["failures"]))
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print("TG unified model lane: ADMITTED")
            return 0
        if args.command == "split-merge-admission":
            if not args.part_bytes or args.merged_bytes is None or args.scratch_bytes is None:
                raise LadderError(
                    "split-merge-admission requires --part-bytes (once per part), "
                    "--merged-bytes, and --scratch-bytes"
                )
            sampled_free = args.free_bytes if args.free_bytes is not None else live_free_bytes(repo_root())
            result = split_gguf_merge_admission(
                free_bytes=sampled_free,
                part_bytes=args.part_bytes,
                merged_bytes=args.merged_bytes,
                scratch_bytes=args.scratch_bytes,
                metadata_and_ranges_bytes=args.metadata_and_ranges_bytes,
                active_model_count=args.active_model_count,
            )
            if result["status"] != "ADMITTED":
                failures = result["streaming"]["failures"]
                raise LadderError("split-GGUF streaming merge admission denied: " + "; ".join(failures))
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print("TG split-GGUF streaming merge lane: ADMITTED")
            return 0
        plan = load_json(args.plan)
        state = load_state(args.state)
        validate_state(plan, state)
        if args.command == "validate":
            result: Any = {"schema": SCHEMA, "valid": True, "plan": str(args.plan)}
        elif args.command == "next":
            result = [item.to_json() for item in next_actions(plan, state)]
        else:
            result = status(plan, state, repo_root())
    except LadderError as exc:
        print(f"tg-ladder: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "next":
        print(render_actions(NextAction(**item) for item in result))
    else:
        print("TG breadth ladder: valid" if args.command == "validate" else json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
