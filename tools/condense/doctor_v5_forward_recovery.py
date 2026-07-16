#!/usr/bin/env python3.12
"""Receipt-bound forward recovery for the rebooted Doctor V5 campaign.

This is a deliberately separate transaction above ``doctor_v5_acceleration_reentry``.
Staging never changes the live queue, state, campaign, registry, runtime specs,
results, active marker, canonical reentry packet, or LaunchAgent.  Activation is
allowed only at an owner-free drain checkpoint and only when every staged and
live compare-and-swap binding still matches.

The transaction resets the exact failed-attempt Qwen rows, archives (never
deletes) every extant nonterminal Qwen result directory, and carries only the
two reviewed 14B ``strand_ladder`` checkpoint trees into fresh result roots.
All terminal evidence remains outside the mutation allowlist.  Rollback is
possible only while the post-promotion result surface is byte-for-byte equal to
the two carried checkpoint baselines and all other promoted result roots remain
absent.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, IO, Iterable


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import doctor_v5_acceleration_reentry as reentry
import doctor_v5_gc_runtime_transition as gc_transition
import doctor_v5_stacked_admission as stacked
import doctor_v5_strand_ladder_block_parallel_adapter as checkpoint_bridge
import doctor_v5_ultra_queue as queue


SCHEMA = "hawking.doctor_v5_forward_recovery_packet.v2"
JOURNAL_SCHEMA = "hawking.doctor_v5_forward_recovery_journal.v2"
ACTIVATION_SCHEMA = "hawking.doctor_v5_forward_recovery_activation_receipt.v2"
ROLLBACK_MANIFEST_SCHEMA = "hawking.doctor_v5_forward_recovery_rollback_manifest.v1"
ROLLBACK_RECEIPT_SCHEMA = "hawking.doctor_v5_forward_recovery_rollback_receipt.v1"
LEGACY_SUPERSESSION_SCHEMA = "hawking.doctor_v5_forward_recovery_supersession.v1"
SUPERSESSION_SCHEMA = "hawking.doctor_v5_forward_recovery_supersession.v2"
SUPERSESSION_LOCATOR_SCHEMA = (
    "hawking.doctor_v5_forward_recovery_supersession_locator.v1"
)
WAL_SCHEMA = "hawking.doctor_v5_forward_recovery_wal_entry.v1"
STAGE_NAME = "forward_recovery_v1"
TERMINAL = frozenset({"complete", "negative", "unsupported"})
BRIDGE_IDS = (
    "qwen2-5-14b__3bpw__codec-control",
    "qwen2-5-14b__4bpw__codec-control",
)

# Exact incident-generation observations.  These are checked in addition to
# deriving the sets from the hash-bound plan/state, so a later campaign cannot
# silently inherit this recovery transaction.
EXPECTED_PLAN_SHA256 = (
    "3d254b5f7fcc5f02b55f2a71f306f7f6852839b699fd14ab4ddf5a05dbaa0106"
)
EXPECTED_TERMINAL_COUNT = 88
EXPECTED_RESET_COUNT = 41
EXPECTED_RESET_IDS_SHA256 = (
    "c97562ac08cf806e5b0b3a5b497b88cde6610568f3cdc8bdd77483a2898f81f0"
)
EXPECTED_RESET_BEFORE_SHA256 = (
    "c24e7a11881cbecec1566e4236ab5f2813b344cd7c3b9474d22d188ecfc3bf8d"
)
EXPECTED_RESET_AFTER_SHA256 = (
    "c8252d1d38aec1ab1655282348f327ae2de3305eab5d53203157117b97ed9c01"
)
EXPECTED_ATTEMPTS_TOTAL = 461
EXPECTED_RESULT_DIR_COUNT = 49
EXPECTED_RESULT_IDS_SHA256 = (
    "712f90f3b7db3fa0c2cbf6d6a7670b57091b60be4d9d900827b4780b4eea6059"
)
EXPECTED_BRIDGE_ANCHORS = {
    "qwen2-5-14b__3bpw__codec-control": {
        "request_sha256": (
            "a88ef27ddedcfc74af0e3e01f1f57cff5a4a7d9ee4531b0c1838ba606888a6a1"
        ),
        "request_bytes": 6259,
        "checkpoint_sha256": (
            "c69184954cc0e8b59cce9b0d1a81c9e54c860d4de70506c4441867a8ab2dbf14"
        ),
        "checkpoint_bytes": 12851,
        "completed_unit_count": 11, "completed_artifact_count": 20,
        "completed_artifact_binding_sha256": (
            "360d9f350173bde5c21cd16c5f7bba0976c07ec693e88d7067b60c44b6749f86"
        ),
    },
    "qwen2-5-14b__4bpw__codec-control": {
        "request_sha256": (
            "16974a341131142ba038daeb0c0b93ffd354cc48de61c0f011be7ce7b99d91ef"
        ),
        "request_bytes": 6259,
        "checkpoint_sha256": (
            "b1b78a131d22e06fe6a1c9ea29286809510d91e9eb35f14ac9be9be39dd859ff"
        ),
        "checkpoint_bytes": 25594,
        "completed_unit_count": 32, "completed_artifact_count": 52,
        "completed_artifact_binding_sha256": (
            "8654d1036aabac69b1d5b46408d20fdedb8fe6e0cd5c42f1928df3896b07b65e"
        ),
    },
}
MAX_JSON_BYTES = 64 * 1024 * 1024
CONTENT_VERIFY_ATTEMPTS = 3


def _fault(_label: str) -> None:
    """No-op production fault boundary; patched only by isolated tests."""


class ForwardRecoveryError(RuntimeError):
    """A recovery invariant is missing, stale, or ambiguous."""


@dataclass(frozen=True)
class Paths:
    root: Path
    ultra: Path
    plan: Path
    state: Path
    campaign: Path
    control: Path
    registry: Path
    results: Path
    runtime_specs: Path
    pid_file: Path
    queue_lock: Path
    heavy_lock: Path
    launch_agent: Path
    active_marker: Path
    overlay: Path
    predecessor_journal: Path
    canonical_reentry_packet: Path
    gc_authority: Path
    accelerated_queue: Path
    accelerated_autoresume: Path
    stage_root: Path
    packet: Path
    journal: Path
    transaction_lock: Path


def production_paths(root: Path = ROOT) -> Paths:
    ultra = root / "reports/condense/doctor_v5_ultra"
    stage = ultra / "staged_acceleration" / STAGE_NAME
    return Paths(
        root=root, ultra=ultra, plan=ultra / "campaign_plan.json",
        state=ultra / "queue_state.json", campaign=ultra / "campaign.json",
        control=ultra / "control.json", registry=ultra / "adapter_registry.json",
        results=ultra / "results", runtime_specs=ultra / "runtime_specs",
        pid_file=ultra / "queue.pid.json", queue_lock=ultra / "queue.lock",
        heavy_lock=root / "reports/cron/studio_heavy.lock",
        launch_agent=Path.home() / (
            "Library/LaunchAgents/com.hawking.doctorv5ultra.autoresume.plist"
        ),
        active_marker=ultra / "staged_acceleration/active_stack.json",
        overlay=ultra / "staged_acceleration/stacked_admission_overlay.json",
        predecessor_journal=ultra / "staged_acceleration/activation_journal.json",
        canonical_reentry_packet=ultra / (
            "staged_acceleration/pending_runtime_packet.json"
        ),
        gc_authority=ultra / (
            "staged_acceleration/gc_runtime_transition_authority.json"
        ),
        accelerated_queue=root / "tools/condense/doctor_v5_ultra_accelerated_queue.py",
        accelerated_autoresume=root / (
            "tools/condense/doctor_v5_ultra_accelerated_autoresume.py"
        ),
        stage_root=stage, packet=stage / "recovery_packet.json",
        journal=stage / "activation_journal.json",
        transaction_lock=stage / "transaction.lock",
    )


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _stable_file(path: Path, *, max_bytes: int | None = None) \
        -> tuple[bytes, os.stat_result]:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ForwardRecoveryError(f"bound path is not a regular file: {path}")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                     | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ForwardRecoveryError(f"cannot open bound file {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if max_bytes is not None and before.st_size > max_bytes:
            raise ForwardRecoveryError(f"bound file is too large: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns
        )
        if identity(before) != identity(after):
            raise ForwardRecoveryError(f"bound file changed while reading: {path}")
        return b"".join(chunks), before
    finally:
        os.close(fd)


def _read_json(path: Path) -> dict[str, Any]:
    raw, _ = _stable_file(path, max_bytes=MAX_JSON_BYTES)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ForwardRecoveryError(f"cannot decode bound JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ForwardRecoveryError(f"bound JSON root is not an object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    """Seal one regular file with constant memory and a stable file identity."""
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ForwardRecoveryError(f"bound path is not a regular file: {path}")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                     | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ForwardRecoveryError(f"cannot open bound file {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(fd)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns
        )
        if identity(before) != identity(after) or size != before.st_size:
            raise ForwardRecoveryError(f"bound file changed while hashing: {path}")
        return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
                "bytes": size}
    finally:
        os.close(fd)


def _optional_artifact(path: Path) -> dict[str, Any]:
    if not _lexists(path):
        return {"path": str(path.absolute()), "exists": False}
    return {"path": str(path.resolve()), "exists": True, "artifact": _artifact(path)}


def _artifact_matches(row: Any, expected: Path | None = None) -> bool:
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
        return False
    try:
        path = Path(row["path"]).resolve(strict=True)
        return (expected is None or path == expected.resolve(strict=True)) \
            and _artifact(path) == row
    except (OSError, TypeError, ValueError, ForwardRecoveryError):
        return False


def _artifact_content_matches(row: Any, path: Path) -> bool:
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
        return False
    # Authority sealing remains strict in _artifact/_artifact_matches.  Only a
    # content-equivalence boundary gets bounded retries, so a one-shot metadata
    # identity update cannot reject correctly promoted bytes.  A stable read
    # with different content fails immediately; persistent churn fails closed.
    for _attempt in range(CONTENT_VERIFY_ATTEMPTS):
        try:
            observed = _artifact(path)
        except (OSError, ForwardRecoveryError):
            continue
        return observed["sha256"] == row.get("sha256") \
            and observed["bytes"] == row.get("bytes")
    return False


def _optional_matches(row: Any, expected: Path) -> bool:
    if not isinstance(row, dict) or row.get("path") != str(expected.absolute()) \
            and row.get("path") != str(expected.resolve()):
        return False
    if row.get("exists") is False:
        return set(row) == {"path", "exists"} and not _lexists(expected)
    return set(row) == {"path", "exists", "artifact"} \
        and row.get("exists") is True \
        and _artifact_matches(row.get("artifact"), expected)


def _lexists(path: Path) -> bool:
    """Return lexical existence, including broken symlinks."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        # An uninspectable path must never satisfy an absence CAS.
        return True
    return True


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path: Path, raw: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            if mode is not None:
                os.fchmod(handle.fileno(), mode)
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path); _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n")


def _replace_file(source: Path, target: Path) -> None:
    raw, info = _stable_file(source)
    _atomic_bytes(target, raw, mode=stat.S_IMODE(info.st_mode))


def _tree_manifest(path: Path) -> dict[str, Any]:
    """Hash every regular file and empty directory in a result subtree."""
    if path.is_symlink() or not path.is_dir():
        raise ForwardRecoveryError(f"result root is not a real directory: {path}")
    entries: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*"), key=lambda row: row.relative_to(path).as_posix()):
        relative = candidate.relative_to(path).as_posix()
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ForwardRecoveryError(f"symlink is forbidden in result archive: {candidate}")
        if stat.S_ISDIR(info.st_mode):
            entries.append({"kind": "directory", "relative_path": relative})
        elif stat.S_ISREG(info.st_mode):
            artifact = _artifact(candidate)
            entries.append({"kind": "file", "relative_path": relative,
                            "sha256": artifact["sha256"], "bytes": artifact["bytes"]})
        else:
            raise ForwardRecoveryError(f"special file is forbidden in result archive: {candidate}")
    return {"path": str(path.resolve()), "entry_count": len(entries),
            "entries_sha256": _hash_value(entries), "entries": entries}


def _tree_matches(row: Any, expected: Path) -> bool:
    try:
        return isinstance(row, dict) and row == _tree_manifest(expected)
    except (OSError, ForwardRecoveryError):
        return False


def _tree_repath(row: dict[str, Any], path: Path,
                 entries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    value = copy.deepcopy(row)
    value["path"] = str(path.resolve())
    if entries is not None:
        value["entries"] = copy.deepcopy(entries)
        value["entry_count"] = len(entries)
        value["entries_sha256"] = _hash_value(entries)
    return value


def _empty_tree(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "entry_count": 0,
            "entries_sha256": _hash_value([]), "entries": []}


def _bridge_archive_tree(result_row: dict[str, Any], archive: Path) -> dict[str, Any]:
    entries = [
        copy.deepcopy(row) for row in result_row["tree"]["entries"]
        if row["relative_path"] != "strand_ladder"
        and not row["relative_path"].startswith("strand_ladder/")
    ]
    return _tree_repath(result_row["tree"], archive, entries)


def _tree_submanifest(tree: dict[str, Any], parent: Path, relative_root: str) \
        -> dict[str, Any]:
    """Derive a subtree seal from an already physically hashed parent tree."""
    prefix = relative_root.rstrip("/") + "/"
    entries: list[dict[str, Any]] = []
    for row in tree["entries"]:
        relative = row["relative_path"]
        if not relative.startswith(prefix):
            continue
        child = dict(row)
        child["relative_path"] = relative[len(prefix):]
        entries.append(child)
    subtree = parent / relative_root
    if not any(row["relative_path"] == relative_root
               and row["kind"] == "directory" for row in tree["entries"]):
        raise ForwardRecoveryError(f"sealed subtree is absent: {subtree}")
    return {"path": str(subtree.resolve()), "entry_count": len(entries),
            "entries_sha256": _hash_value(entries), "entries": entries}


def _bridge_anchor(ladder: Path, tree: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fully validate and seal the durable prefix carried across activation."""
    request_path, checkpoint_path = ladder / "request.json", ladder / "checkpoint.json"
    request, checkpoint = _read_json(request_path), _read_json(checkpoint_path)
    tree = tree or _tree_manifest(ladder)
    indexed = {
        row["relative_path"]: row for row in tree["entries"]
        if row.get("kind") == "file"
    }

    def sealed_artifact(path: Path) -> dict[str, Any]:
        try:
            relative = path.resolve().relative_to(ladder.resolve()).as_posix()
        except ValueError as exc:
            raise ForwardRecoveryError(
                f"checkpoint artifact escapes carried ladder: {path}"
            ) from exc
        row = indexed.get(relative)
        if row is None:
            raise ForwardRecoveryError(f"checkpoint artifact is absent from tree seal: {path}")
        return {"path": str(path.resolve()), "sha256": row["sha256"],
                "bytes": row["bytes"]}

    request_artifact = sealed_artifact(request_path)
    checkpoint_artifact = sealed_artifact(checkpoint_path)
    plan, completed, units = (
        checkpoint.get("plan"), checkpoint.get("completed_units"), checkpoint.get("units")
    )
    if not isinstance(plan, list) or not isinstance(completed, list) \
            or not completed or len(completed) != len(set(completed)) \
            or completed != plan[:len(completed)] or not isinstance(units, dict) \
            or checkpoint.get("request_sha256") != request_artifact["sha256"]:
        raise ForwardRecoveryError("checkpoint bridge plan/request prefix is invalid")
    anchored = {"units": {unit: units.get(unit) for unit in completed}}
    if any(not isinstance(value, dict) for value in anchored["units"].values()):
        raise ForwardRecoveryError("checkpoint bridge completed unit lacks evidence")
    references = sorted(
        checkpoint_bridge._BASE._checkpoint_artifact_rows(anchored),
        key=lambda row: (row["path"], row["sha256"], row["bytes"]),
    )
    for row in references:
        try:
            observed = sealed_artifact(Path(row["path"]))
        except (OSError, ForwardRecoveryError) as exc:
            raise ForwardRecoveryError(
                f"checkpoint bridge completed artifact is unsafe: {row.get('path')}: {exc}"
            ) from exc
        if observed != row:
            raise ForwardRecoveryError(
                f"checkpoint bridge completed artifact changed: {row.get('path')}"
            )
    return {
        "request": request_artifact, "checkpoint": checkpoint_artifact,
        "request_sha256": request_artifact["sha256"],
        "request_bytes": request_artifact["bytes"],
        "checkpoint_sha256": checkpoint_artifact["sha256"],
        "checkpoint_bytes": checkpoint_artifact["bytes"],
        "request_id": request.get("request_id"), "checkpoint_status": checkpoint.get("status"),
        "completed_units": completed, "completed_unit_count": len(completed),
        "completed_artifact_count": len(references),
        "completed_artifact_binding_sha256": _hash_value(references),
    }


def _patch_module_paths(paths: Paths) -> dict[str, Any]:
    """Patch queue/overlay globals for a non-production test fixture."""
    return {
        "queue": {
            "ROOT": queue.ROOT, "ULTRA_ROOT": queue.ULTRA_ROOT,
            "RESULTS": queue.RESULTS, "PLAN": queue.PLAN,
            "STATE": queue.STATE, "CAMPAIGN": queue.CAMPAIGN,
            "CONTROL": queue.CONTROL,
        },
        "stacked": {
            "ROOT": stacked.ROOT, "ULTRA_ROOT": stacked.ULTRA_ROOT,
            "PLAN": stacked.PLAN, "CAMPAIGN": stacked.CAMPAIGN,
            "QUEUE_STATE": stacked.QUEUE_STATE, "DEFAULT_OVERLAY": stacked.DEFAULT_OVERLAY,
        },
    }


def _set_module_paths(paths: Paths) -> dict[str, Any]:
    old = _patch_module_paths(paths)
    queue.ROOT = paths.root; queue.ULTRA_ROOT = paths.ultra
    queue.RESULTS = paths.results; queue.PLAN = paths.plan
    queue.STATE = paths.state; queue.CAMPAIGN = paths.campaign; queue.CONTROL = paths.control
    stacked.ROOT = paths.root; stacked.ULTRA_ROOT = paths.ultra
    stacked.PLAN = paths.plan; stacked.CAMPAIGN = paths.campaign
    stacked.QUEUE_STATE = paths.state; stacked.DEFAULT_OVERLAY = paths.overlay
    return old


def _restore_module_paths(old: dict[str, Any]) -> None:
    for module, values in ((queue, old["queue"]), (stacked, old["stacked"])):
        for name, value in values.items():
            setattr(module, name, value)


def _campaign_projection(paths: Paths, plan: dict[str, Any], state: dict[str, Any],
                         created_at: str) -> dict[str, Any]:
    old = _set_module_paths(paths)
    old_now = queue._now
    try:
        queue._now = lambda: created_at
        return queue._campaign_projection(plan, state)
    finally:
        queue._now = old_now
        _restore_module_paths(old)


def _build_overlay(paths: Paths) -> dict[str, Any]:
    old = _set_module_paths(paths)
    try:
        return stacked.build_overlay()
    finally:
        _restore_module_paths(old)


def _build_staged_overlay(paths: Paths, packet: dict[str, Any]) -> dict[str, Any]:
    """Build the post-promotion overlay without changing a live path."""
    old = _patch_module_paths(paths)
    try:
        stacked.ROOT = paths.root; stacked.ULTRA_ROOT = paths.ultra
        stacked.PLAN = paths.plan
        stacked.CAMPAIGN = Path(packet["staged_campaign"]["path"])
        stacked.QUEUE_STATE = Path(packet["staged_state"]["path"])
        stacked.DEFAULT_OVERLAY = paths.overlay
        overlay = stacked.build_overlay()
    finally:
        _restore_module_paths(old)
    for name, staged, live in (
        ("campaign", packet["staged_campaign"], paths.campaign),
        ("queue_state", packet["staged_state"], paths.state),
    ):
        binding = overlay["source_bindings"][name]
        binding["path"] = str(live.resolve())
        binding["sha256"] = staged["sha256"]
        binding["bytes"] = staged["bytes"]
    overlay["overlay_sha256"] = _hash_value(_without(overlay, "overlay_sha256"))
    errors = stacked.validate_overlay(overlay)
    if errors:
        raise ForwardRecoveryError("staged reset-state overlay is invalid: "
                                   + "; ".join(errors))
    return overlay


def _reset_ids(plan: dict[str, Any], state: dict[str, Any]) -> list[str]:
    cells = {row["cell_id"]: row for row in plan["cells"]}
    return sorted(
        cell_id for cell_id, row in state["cells"].items()
        if cells[cell_id].get("model_family") == "qwen2.5-dense"
        and row.get("status") not in TERMINAL
        and (row.get("attempts", 0) > 0 or row.get("status") == "blocked-execution")
    )


def _reset_rows(plan: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell_id in _reset_ids(plan, state):
        before = copy.deepcopy(state["cells"][cell_id])
        # A non-null lifecycle triplet would be real GC evidence, not retry
        # residue, and must never be erased by this operational recovery.
        if (before.get("packed_gc_receipt_sha256"), before.get("payload_released_at"),
                before.get("released_payload_bytes")) != (None, None, 0):
            raise ForwardRecoveryError(
                f"reset row contains lifecycle evidence and is not operational-only: {cell_id}"
            )
        after = queue._state_row()
        rows.append({
            "cell_id": cell_id, "before": before,
            "before_sha256": _hash_value(before), "after": after,
            "after_sha256": _hash_value(after),
            "reason": "reboot-source-seal-operational-retry-reset",
        })
    return rows


def _make_reset_state(plan: dict[str, Any], state: dict[str, Any],
                      control: dict[str, Any], rows: list[dict[str, Any]],
                      created_at: str) -> dict[str, Any]:
    staged = copy.deepcopy(state)
    for row in rows:
        staged["cells"][row["cell_id"]] = copy.deepcopy(row["after"])
    staged["status"] = "drained" if control.get("mode") == "drain" else "paused"
    staged["control_mode"] = control.get("mode")
    staged["supervisor_pid"] = None
    staged["active_cells"] = []
    staged["active_children"] = {}
    staged["last_resource_stop"] = None
    counts = staged.get("resource_stop_counts")
    if isinstance(counts, dict):
        for row in rows:
            counts.pop(row["cell_id"], None)
    staged["last_scan"] = None
    staged["error"] = None
    staged["updated_at"] = created_at
    staged.pop("state_sha256", None)
    staged["state_sha256"] = _hash_value(staged)
    errors = queue._validate_state(staged, plan)
    if errors:
        raise ForwardRecoveryError("staged reset state is invalid: " + "; ".join(errors))
    return staged


def _runtime_inventory(plan: dict[str, Any], paths: Paths) -> list[dict[str, Any]]:
    rows = []
    for cell in sorted(plan["cells"], key=lambda row: row["cell_id"]):
        path = (paths.root / cell["runtime_spec_path"]).resolve()
        rows.append({"cell_id": cell["cell_id"],
                     "binding": _optional_artifact(path)})
    return rows


def _result_inventory(plan: dict[str, Any], state: dict[str, Any],
                      paths: Paths, bridge_ids: Iterable[str]) -> list[dict[str, Any]]:
    bridges = set(bridge_ids)
    cells = {row["cell_id"]: row for row in plan["cells"]}
    rows: list[dict[str, Any]] = []
    for cell_id in sorted(cells):
        cell = cells[cell_id]
        if cell.get("model_family") != "qwen2.5-dense" \
                or state["cells"][cell_id]["status"] in TERMINAL:
            continue
        root = paths.results / cell_id
        if not _lexists(root):
            continue
        row: dict[str, Any] = {
            "cell_id": cell_id, "tree": _tree_manifest(root),
            "checkpoint_bridge": cell_id in bridges,
        }
        if cell_id in bridges:
            ladder = root / "strand_ladder"
            row["strand_ladder"] = _tree_submanifest(
                row["tree"], root, "strand_ladder"
            )
            row["bridge_anchor"] = _bridge_anchor(ladder, row["strand_ladder"])
        rows.append(row)
    missing = [cell_id for cell_id in bridges
               if not any(row["cell_id"] == cell_id for row in rows)]
    if missing:
        raise ForwardRecoveryError(
            "required checkpoint result roots are absent: " + ", ".join(sorted(missing))
        )
    return rows


def _exact_incident_checks(*, plan_sha256: str, reset_rows: list[dict[str, Any]],
                           result_rows: list[dict[str, Any]], terminal_count: int,
                           production: bool) -> None:
    if not production:
        return
    reset_ids = [row["cell_id"] for row in reset_rows]
    before = [{"cell_id": row["cell_id"], "row": row["before"]} for row in reset_rows]
    after = [{"cell_id": row["cell_id"], "row": row["after"]} for row in reset_rows]
    attempts = sum(row["before"]["attempts"] for row in reset_rows)
    result_ids = [row["cell_id"] for row in result_rows]
    checks = (
        (plan_sha256 == EXPECTED_PLAN_SHA256, "plan generation differs"),
        (terminal_count == EXPECTED_TERMINAL_COUNT, "terminal count differs"),
        (len(reset_rows) == EXPECTED_RESET_COUNT, "reset count differs"),
        (_hash_value(reset_ids) == EXPECTED_RESET_IDS_SHA256, "reset ID digest differs"),
        (_hash_value(before) == EXPECTED_RESET_BEFORE_SHA256,
         "reset before-row digest differs"),
        (_hash_value(after) == EXPECTED_RESET_AFTER_SHA256,
         "reset after-row digest differs"),
        (attempts == EXPECTED_ATTEMPTS_TOTAL, "historical attempts total differs"),
        (len(result_rows) == EXPECTED_RESULT_DIR_COUNT, "result archive count differs"),
        (_hash_value(result_ids) == EXPECTED_RESULT_IDS_SHA256,
         "result archive ID digest differs"),
    )
    failures = [message for ok, message in checks if not ok]
    for row in result_rows:
        if row["cell_id"] in EXPECTED_BRIDGE_ANCHORS:
            anchor = row.get("bridge_anchor", {})
            expected = EXPECTED_BRIDGE_ANCHORS[row["cell_id"]]
            if any(anchor.get(name) != value for name, value in expected.items()):
                failures.append(f"checkpoint anchor differs: {row['cell_id']}")
    if failures:
        raise ForwardRecoveryError("incident generation check failed: " + "; ".join(failures))


def _capture_prepared_packet(paths: Paths) -> dict[str, Any]:
    """Run reentry preparation without replacing its active canonical packet."""
    capture = paths.stage_root / f".pending-runtime-capture-{secrets.token_hex(8)}.json"
    try:
        # The authority is immutable input to this recovery.  Staging validates
        # it but never regenerates or replaces it.  Reentry has an explicit
        # generation-local destination, so the canonical active packet is never
        # exposed to a transient successor or changed process-globally.
        gc_transition.validate_authority(paths.gc_authority)
        prepared = reentry.prepare(packet_path=capture)
        captured = _read_json(capture)
        if captured != prepared:
            raise ForwardRecoveryError("reentry prepare return/captured packet differ")
        return prepared
    finally:
        capture.unlink(missing_ok=True)


def _quiescent(paths: Paths) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    try:
        state, campaign, control = (
            _read_json(paths.state), _read_json(paths.campaign), _read_json(paths.control)
        )
    except ForwardRecoveryError as exc:
        return False, [str(exc)]
    if control.get("mode") not in {"drain", "pause"}:
        blockers.append("control is not drain/pause")
    if state.get("status") not in {"drained", "paused"}:
        blockers.append("queue state is not drained/paused")
    if state.get("active_cells") or state.get("active_children") \
            or campaign.get("active_cells") or campaign.get("active_children"):
        blockers.append("queue still records active children")
    return not blockers, blockers


def stage(paths: Paths | None = None, *, production_checks: bool = True,
          bridge_ids: Iterable[str] = BRIDGE_IDS) -> dict[str, Any]:
    paths = paths or production_paths()
    recovery_lease = _acquire(paths.transaction_lock)
    try:
        _supersession_barrier(paths, operation="stage")
        return _stage_impl(
            paths, production_checks=production_checks, bridge_ids=bridge_ids
        )
    finally:
        recovery_lease.close()


def _stage_impl(paths: Paths, *, production_checks: bool,
                bridge_ids: Iterable[str]) -> dict[str, Any]:
    if paths.packet.exists():
        packet = _read_json(paths.packet)
        errors = validate_packet(packet, paths=paths, production_checks=production_checks,
                                 bridge_ids=bridge_ids)
        if errors:
            raise ForwardRecoveryError("existing forward packet is invalid: " + "; ".join(errors))
        return packet
    ready, blockers = _quiescent(paths)
    if not ready:
        raise ForwardRecoveryError("stage requires a quiescent checkpoint: "
                                   + "; ".join(blockers))
    plan, state, control = (
        _read_json(paths.plan), _read_json(paths.state), _read_json(paths.control)
    )
    campaign = _read_json(paths.campaign)
    prepared = _capture_prepared_packet(paths)
    prepared_errors = reentry.validate_packet(prepared)
    if prepared_errors:
        raise ForwardRecoveryError("prepared fixed-20 packet is invalid: "
                                   + "; ".join(prepared_errors))
    authority = gc_transition.validate_authority(paths.gc_authority)
    reset_rows = _reset_rows(plan, state)
    result_rows = _result_inventory(plan, state, paths, bridge_ids)
    terminal = prepared.get("terminal_seal", {})
    _exact_incident_checks(
        plan_sha256=plan["plan_sha256"], reset_rows=reset_rows,
        result_rows=result_rows, terminal_count=terminal.get("count", -1),
        production=production_checks,
    )
    created_at = _now()
    runtime_inventory = _runtime_inventory(plan, paths)
    live_bindings = {
        "plan": _artifact(paths.plan), "state": _artifact(paths.state),
        "campaign": _artifact(paths.campaign), "control": _artifact(paths.control),
        "registry": _artifact(paths.registry),
        "active_marker": _optional_artifact(paths.active_marker),
        "overlay": _optional_artifact(paths.overlay),
        "predecessor_journal": _optional_artifact(paths.predecessor_journal),
        "canonical_reentry_packet": _optional_artifact(paths.canonical_reentry_packet),
        "pid_file": _optional_artifact(paths.pid_file),
        "launch_agent": _artifact(paths.launch_agent),
        "accelerated_queue": _artifact(paths.accelerated_queue),
        "accelerated_autoresume": _artifact(paths.accelerated_autoresume),
        "forward_recovery": _artifact(Path(__file__)),
    }
    seed = {
        "plan_sha256": plan["plan_sha256"], "state": _artifact(paths.state),
        "prepared_packet_sha256": prepared["packet_sha256"],
        "gc_authority_sha256": authority["authority_sha256"],
        "reset_rows_sha256": _hash_value(reset_rows),
        "result_inventory_sha256": _hash_value(result_rows),
        "runtime_inventory_sha256": _hash_value(runtime_inventory),
        "live_bindings_sha256": _hash_value(live_bindings),
    }
    generation_id = _hash_value(seed)
    generation = paths.stage_root / "generations" / generation_id
    generation.mkdir(parents=True, exist_ok=True)
    prepared_path = generation / "pending_runtime_packet.json"
    _atomic_json(prepared_path, prepared)
    staged_state = _make_reset_state(plan, state, control, reset_rows, created_at)
    staged_state_path = generation / "queue_state.json"
    _atomic_json(staged_state_path, staged_state)
    staged_campaign = _campaign_projection(paths, plan, staged_state, created_at)
    staged_campaign_path = generation / "campaign.json"
    _atomic_json(staged_campaign_path, staged_campaign)
    packet: dict[str, Any] = {
        "schema": SCHEMA, "created_at": created_at,
        "generation_id": generation_id, "plan_sha256": plan["plan_sha256"],
        "live_bindings": live_bindings,
        "runtime_spec_inventory": runtime_inventory,
        "prepared_runtime_packet": _artifact(prepared_path),
        "prepared_packet_sha256": prepared["packet_sha256"],
        "gc_transition_authority": _artifact(paths.gc_authority),
        "terminal_seal": terminal,
        "reset_rows": reset_rows,
        "reset_ids_sha256": _hash_value([row["cell_id"] for row in reset_rows]),
        "attempts_before_total": sum(row["before"]["attempts"] for row in reset_rows),
        "last_resource_stop_before": copy.deepcopy(state.get("last_resource_stop")),
        "staged_state": _artifact(staged_state_path),
        "staged_campaign": _artifact(staged_campaign_path),
        "result_archives": result_rows,
        "result_ids_sha256": _hash_value([row["cell_id"] for row in result_rows]),
        "checkpoint_bridge_ids": sorted(bridge_ids),
        "completed_evidence_mutation_permitted": False,
        "source_deletion_permitted": False,
        "automatic_rollback_requires_no_new_output": True,
    }
    packet["packet_sha256"] = _hash_value(packet)
    _atomic_json(paths.packet, packet)
    errors = validate_packet(packet, paths=paths, production_checks=production_checks,
                             bridge_ids=bridge_ids)
    if errors:
        raise ForwardRecoveryError("new forward packet failed audit: " + "; ".join(errors))
    return packet


def _validate_real_directory(path: Path) -> None:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ForwardRecoveryError(f"supersession directory is unsafe: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) \
            or resolved != path.absolute():
        raise ForwardRecoveryError(f"supersession directory is unsafe: {path}")


def _ensure_real_directory(path: Path) -> None:
    _validate_real_directory(path.parent)
    existed = _lexists(path)
    try:
        path.mkdir(exist_ok=True)
    except OSError as exc:
        raise ForwardRecoveryError(f"supersession directory is unsafe: {path}") from exc
    _validate_real_directory(path)
    if not existed:
        _fsync_dir(path.parent)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 \
        and all(character in "0123456789abcdef" for character in value)


def _artifact_shape(value: Any, *, expected_path: Path | None = None) -> bool:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"} \
            or not isinstance(value.get("path"), str) \
            or not _valid_sha256(value.get("sha256")) \
            or isinstance(value.get("bytes"), bool) \
            or not isinstance(value.get("bytes"), int) or value["bytes"] < 0:
        return False
    return expected_path is None or value["path"] == str(expected_path.absolute())


def _validate_supersession_intent(paths: Paths, path: Path) -> dict[str, Any]:
    intent = _read_json(path)
    legacy_keys = {
        "schema", "created_at", "status", "reason", "original_packet",
        "archive_path", "source_deletion_permitted", "supersession_sha256",
    }
    current_keys = legacy_keys | {
        "original_journal", "journal_archive_path", "terminal_rollback_receipt",
    }
    schema = intent.get("schema")
    expected_keys = (legacy_keys if schema == LEGACY_SUPERSESSION_SCHEMA
                     else current_keys)
    original = intent.get("original_packet")
    if set(intent) != expected_keys \
            or schema not in {LEGACY_SUPERSESSION_SCHEMA, SUPERSESSION_SCHEMA} \
            or intent.get("status") != "move-intent" \
            or not isinstance(intent.get("reason"), str) \
            or not intent["reason"].strip() \
            or intent.get("source_deletion_permitted") is not False \
            or intent.get("supersession_sha256") != _hash_value(
                _without(intent, "supersession_sha256")
            ) or not _artifact_shape(original, expected_path=paths.packet):
        raise ForwardRecoveryError("supersession intent schema/hash/identity is invalid")
    archive_root = (
        paths.stage_root / "superseded" / original["sha256"]
    ).absolute()
    expected_archive = (
        paths.stage_root / "superseded" / original["sha256"] / "recovery_packet.json"
    ).absolute()
    if intent.get("archive_path") != str(expected_archive) \
            or path.absolute() != expected_archive.with_name(
                "supersession_intent.json"
            ):
        raise ForwardRecoveryError("supersession intent path escapes its packet archive")
    _validate_real_directory(paths.stage_root)
    _validate_real_directory(paths.stage_root / "superseded")
    _validate_real_directory(archive_root)
    try:
        if path.resolve(strict=True) != path.absolute():
            raise ForwardRecoveryError(
                "supersession intent ancestry is not lexically exact"
            )
    except OSError as exc:
        raise ForwardRecoveryError("supersession intent is absent/unsafe") from exc
    if schema == SUPERSESSION_SCHEMA:
        original_journal = intent.get("original_journal")
        journal_archive = intent.get("journal_archive_path")
        rollback_receipt = intent.get("terminal_rollback_receipt")
        expected_journal_archive = archive_root / "activation_journal.json"
        if original_journal is None:
            if journal_archive is not None or rollback_receipt is not None:
                raise ForwardRecoveryError(
                    "supersession intent terminal transaction fields are inconsistent"
                )
        elif not _artifact_shape(original_journal, expected_path=paths.journal) \
                or journal_archive != str(expected_journal_archive) \
                or not _artifact_shape(rollback_receipt):
            raise ForwardRecoveryError(
                "supersession intent terminal transaction binding is invalid"
            )
    return intent


def _validate_supersession_locator(paths: Paths, path: Path) \
        -> tuple[dict[str, Any], Path, dict[str, Any]]:
    locator = _read_json(path)
    expected_keys = {
        "schema", "created_at", "status", "reason", "original_packet_sha256",
        "intent", "source_deletion_permitted", "locator_sha256",
    }
    intent_ref = locator.get("intent")
    if set(locator) != expected_keys \
            or locator.get("schema") != SUPERSESSION_LOCATOR_SCHEMA \
            or locator.get("status") != "active" \
            or not isinstance(locator.get("reason"), str) \
            or not locator["reason"].strip() \
            or not _valid_sha256(locator.get("original_packet_sha256")) \
            or locator.get("source_deletion_permitted") is not False \
            or locator.get("locator_sha256") != _hash_value(
                _without(locator, "locator_sha256")
            ) or not isinstance(intent_ref, dict) \
            or not isinstance(intent_ref.get("path"), str):
        raise ForwardRecoveryError("supersession locator schema/hash is invalid")
    intent_path = Path(intent_ref["path"])
    expected_intent = (
        paths.stage_root / "superseded" / locator["original_packet_sha256"]
        / "supersession_intent.json"
    ).absolute()
    if intent_path.absolute() != expected_intent:
        raise ForwardRecoveryError("supersession locator intent path is not lexical")
    _validate_real_directory(paths.stage_root)
    _validate_real_directory(paths.stage_root / "superseded")
    _validate_real_directory(expected_intent.parent)
    if not _artifact_matches(intent_ref, intent_path):
        raise ForwardRecoveryError("supersession locator intent binding differs")
    intent = _validate_supersession_intent(paths, intent_path)
    if locator["reason"] != intent["reason"] \
            or locator["original_packet_sha256"] \
            != intent["original_packet"]["sha256"]:
        raise ForwardRecoveryError("supersession locator identity differs")
    return locator, intent_path, intent


def _validate_supersession_receipt(intent: dict[str, Any], intent_path: Path,
                                   receipt_path: Path) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    legacy_keys = {
        "schema", "created_at", "status", "reason", "intent",
        "archived_packet", "original_packet_sha256",
        "source_deletion_permitted", "supersession_sha256",
    }
    current_keys = legacy_keys | {"archived_journal", "terminal_rollback_receipt"}
    schema = receipt.get("schema")
    expected_keys = (legacy_keys if schema == LEGACY_SUPERSESSION_SCHEMA
                     else current_keys)
    archived_packet = Path(intent["archive_path"])
    if set(receipt) != expected_keys \
            or schema not in {LEGACY_SUPERSESSION_SCHEMA, SUPERSESSION_SCHEMA} \
            or schema != intent.get("schema") \
            or receipt.get("status") != "superseded" \
            or receipt.get("reason") != intent["reason"] \
            or receipt.get("original_packet_sha256") \
            != intent["original_packet"]["sha256"] \
            or receipt.get("source_deletion_permitted") is not False \
            or receipt.get("supersession_sha256") != _hash_value(
                _without(receipt, "supersession_sha256")
            ) or not _artifact_matches(receipt.get("intent"), intent_path) \
            or not _artifact_matches(receipt.get("archived_packet"), archived_packet) \
            or not _artifact_content_matches(
                intent["original_packet"], archived_packet
            ):
        raise ForwardRecoveryError("supersession receipt schema/hash/chain is invalid")
    if schema == SUPERSESSION_SCHEMA:
        original_journal = intent.get("original_journal")
        archived_journal = receipt.get("archived_journal")
        rollback_receipt = intent.get("terminal_rollback_receipt")
        if original_journal is None:
            if archived_journal is not None \
                    or receipt.get("terminal_rollback_receipt") is not None:
                raise ForwardRecoveryError(
                    "supersession receipt terminal transaction fields are inconsistent"
                )
        else:
            expected_journal = Path(intent["journal_archive_path"])
            rollback_path = Path(rollback_receipt["path"])
            if not _artifact_matches(archived_journal, expected_journal) \
                    or not _artifact_content_matches(original_journal, expected_journal) \
                    or receipt.get("terminal_rollback_receipt") != rollback_receipt \
                    or not _artifact_matches(rollback_receipt, rollback_path):
                raise ForwardRecoveryError(
                    "supersession receipt terminal transaction chain is invalid"
                )
    return receipt


def _supersession_barrier(paths: Paths, *, operation: str) -> None:
    """Refuse staging/activation while a packet supersession is unresolved."""
    if operation not in {"stage", "apply"}:
        raise ForwardRecoveryError("supersession barrier operation is invalid")
    locator_path = paths.stage_root / "supersession_active.json"
    packet_ref = _artifact(paths.packet) if _lexists(paths.packet) else None
    checked_intents: set[Path] = set()

    def check(intent_path: Path, intent: dict[str, Any]) -> None:
        canonical = intent_path.absolute()
        if canonical in checked_intents:
            return
        checked_intents.add(canonical)
        receipt_path = Path(intent["archive_path"]).with_name(
            "supersession_receipt.json"
        )
        if not _lexists(receipt_path):
            raise ForwardRecoveryError(
                f"{operation} refused while packet supersession is unfinished"
            )
        _validate_supersession_receipt(intent, intent_path, receipt_path)
        if packet_ref is not None \
                and packet_ref["sha256"] == intent["original_packet"]["sha256"]:
            raise ForwardRecoveryError(
                f"{operation} refused because the live packet was already superseded"
            )

    if _lexists(locator_path):
        _, intent_path, intent = _validate_supersession_locator(paths, locator_path)
        check(intent_path, intent)
    if packet_ref is not None:
        deterministic = (
            paths.stage_root / "superseded" / packet_ref["sha256"]
            / "supersession_intent.json"
        )
        if _lexists(deterministic):
            check(deterministic, _validate_supersession_intent(paths, deterministic))


def _terminal_transaction_supersession(
        paths: Paths, packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    journal = _read_json(paths.journal)
    backup, manifest, _, _ = _validate_transaction_chain(paths, packet, journal)
    if journal.get("status") != "rolled-back" \
            or journal.get("live_marker") is not None \
            or journal.get("supervisor_pid") is not None:
        raise ForwardRecoveryError(
            "only a terminal owner-free rolled-back transaction may be superseded"
        )
    rollback_path = backup / "rollback_receipt.json"
    rollback_ref = journal.get("rollback_receipt")
    if not _artifact_matches(rollback_ref, rollback_path):
        raise ForwardRecoveryError("terminal rollback receipt binding differs")
    rollback = _read_json(rollback_path)
    if rollback.get("schema") != ROLLBACK_RECEIPT_SCHEMA \
            or rollback.get("status") != "rolled-back" \
            or rollback.get("forward_packet_sha256") != packet.get("packet_sha256") \
            or rollback.get("receipt_sha256") \
            != _hash_value(_without(rollback, "receipt_sha256")) \
            or rollback.get("restored_result_ids") \
            != [row["cell_id"] for row in manifest["result_moves"]] \
            or rollback.get("source_deletion_permitted") is not False:
        raise ForwardRecoveryError("terminal rollback receipt is invalid")
    return _artifact(paths.journal), rollback_ref


def supersede(*, reason: str, paths: Paths | None = None) -> dict[str, Any]:
    """Crash-convergently archive an obsolete staged packet without deleting it."""
    paths = paths or production_paths()
    if not isinstance(reason, str) or not reason.strip():
        raise ForwardRecoveryError("packet supersession requires a reason")
    reason = reason.strip()
    recovery_lease, queue_lease, heavy_lease = _acquire_all(paths)
    try:
        owner = _verified_owner(paths)
        if owner is not None:
            raise ForwardRecoveryError(
                f"supersession refused because accelerated owner is live: {owner}"
            )
        ready, blockers = _quiescent(paths)
        if not ready:
            raise ForwardRecoveryError("supersession requires an owner-free drain: "
                                       + "; ".join(blockers))
        locator_path = paths.stage_root / "supersession_active.json"
        packet_present = _lexists(paths.packet)
        terminal_transaction: tuple[dict[str, Any], dict[str, Any]] | None = None
        if _lexists(paths.journal):
            if not packet_present:
                raise ForwardRecoveryError(
                    "transaction journal exists without its recovery packet"
                )
            terminal_transaction = _terminal_transaction_supersession(
                paths, _read_json(paths.packet)
            )
        if packet_present:
            original = _artifact(paths.packet)
            predecessor: tuple[dict[str, Any], Path, dict[str, Any]] | None = None
            if _lexists(locator_path):
                predecessor = _validate_supersession_locator(paths, locator_path)
                old_locator, old_intent_path, old_intent = predecessor
                if old_locator["original_packet_sha256"] != original["sha256"]:
                    old_receipt_path = Path(old_intent["archive_path"]).with_name(
                        "supersession_receipt.json"
                    )
                    if not _lexists(old_receipt_path):
                        raise ForwardRecoveryError(
                            "a different supersession transaction is unfinished"
                        )
                    _validate_supersession_receipt(
                        old_intent, old_intent_path, old_receipt_path
                    )
                elif old_locator["reason"] != reason:
                    raise ForwardRecoveryError(
                        "existing supersession transaction reason differs"
                    )
            superseded_root = paths.stage_root / "superseded"
            _ensure_real_directory(superseded_root)
            archive_root = superseded_root / original["sha256"]
            _ensure_real_directory(archive_root)
            archived_packet = archive_root / "recovery_packet.json"
            intent_path = archive_root / "supersession_intent.json"
            if _lexists(intent_path):
                intent = _validate_supersession_intent(paths, intent_path)
                if intent["reason"] != reason or intent["original_packet"] != original:
                    raise ForwardRecoveryError("supersession intent already differs")
            else:
                intent = {
                    "schema": SUPERSESSION_SCHEMA, "created_at": _now(),
                    "status": "move-intent", "reason": reason,
                    "original_packet": original,
                    "archive_path": str(archived_packet.resolve()),
                    "original_journal": (
                        terminal_transaction[0] if terminal_transaction else None
                    ),
                    "journal_archive_path": (
                        str((archive_root / "activation_journal.json").absolute())
                        if terminal_transaction else None
                    ),
                    "terminal_rollback_receipt": (
                        terminal_transaction[1] if terminal_transaction else None
                    ),
                    "source_deletion_permitted": False,
                }
                intent["supersession_sha256"] = _hash_value(intent)
                _atomic_json(intent_path, intent)
                intent = _validate_supersession_intent(paths, intent_path)
            _fault("after:supersede:intent")
            locator = {
                "schema": SUPERSESSION_LOCATOR_SCHEMA, "created_at": _now(),
                "status": "active", "reason": reason,
                "original_packet_sha256": original["sha256"],
                "intent": _artifact(intent_path),
                "source_deletion_permitted": False,
            }
            locator["locator_sha256"] = _hash_value(locator)
            _atomic_json(locator_path, locator)
            _validate_supersession_locator(paths, locator_path)
        else:
            if not _lexists(locator_path):
                raise ForwardRecoveryError(
                    "no staged forward packet or recoverable supersession exists"
                )
            locator, intent_path, intent = _validate_supersession_locator(
                paths, locator_path
            )
            if locator["reason"] != reason:
                raise ForwardRecoveryError("supersession recovery reason differs")
            original = intent["original_packet"]
            archived_packet = Path(intent["archive_path"])
            archive_root = archived_packet.parent
        _fault("after:supersede:locator")

        original_journal = intent.get("original_journal")
        archived_journal = (
            Path(intent["journal_archive_path"])
            if original_journal is not None else None
        )
        if archived_journal is not None:
            if _lexists(archived_journal):
                if not _artifact_content_matches(original_journal, archived_journal):
                    raise ForwardRecoveryError("superseded journal archive differs")
                if _lexists(paths.journal):
                    raise ForwardRecoveryError(
                        "both live and archived transaction journals exist"
                    )
            else:
                if not _artifact_matches(original_journal, paths.journal):
                    raise ForwardRecoveryError(
                        "terminal journal changed before supersession move"
                    )
                os.replace(paths.journal, archived_journal)
                _fsync_dir(paths.journal.parent)
                _fsync_dir(archived_journal.parent)
            _fault("after:supersede:journal-move")
        elif _lexists(paths.journal):
            raise ForwardRecoveryError(
                "unbound transaction journal appeared during packet supersession"
            )

        receipt_path = archive_root / "supersession_receipt.json"
        if _lexists(archived_packet):
            if not _artifact_content_matches(original, archived_packet):
                raise ForwardRecoveryError("superseded packet archive differs")
            if _lexists(paths.packet):
                raise ForwardRecoveryError("both staged and archived packets exist")
        else:
            if not _artifact_matches(original, paths.packet):
                raise ForwardRecoveryError("staged packet changed before supersession move")
            os.replace(paths.packet, archived_packet)
            _fsync_dir(paths.packet.parent)
            _fsync_dir(archived_packet.parent)
        _fault("after:supersede:move")

        if _lexists(receipt_path):
            return _validate_supersession_receipt(intent, intent_path, receipt_path)
        receipt = {
            "schema": intent["schema"], "created_at": _now(),
            "status": "superseded", "reason": reason,
            "intent": _artifact(intent_path),
            "archived_packet": _artifact(archived_packet),
            "original_packet_sha256": original["sha256"],
            "source_deletion_permitted": False,
        }
        if intent["schema"] == SUPERSESSION_SCHEMA:
            receipt["archived_journal"] = (
                _artifact(archived_journal) if archived_journal is not None else None
            )
            receipt["terminal_rollback_receipt"] = intent.get(
                "terminal_rollback_receipt"
            )
        receipt["supersession_sha256"] = _hash_value(receipt)
        _atomic_json(receipt_path, receipt)
        receipt = _validate_supersession_receipt(intent, intent_path, receipt_path)
        _fault("after:supersede:receipt")
        return receipt
    finally:
        heavy_lease.close(); queue_lease.close(); recovery_lease.close()


def _validate_runtime_inventory(rows: Any, plan: dict[str, Any], paths: Paths) -> list[str]:
    errors: list[str] = []
    if not isinstance(rows, list):
        return ["runtime spec inventory is not a list"]
    cells = sorted(plan["cells"], key=lambda row: row["cell_id"])
    if [row.get("cell_id") for row in rows if isinstance(row, dict)] \
            != [row["cell_id"] for row in cells]:
        errors.append("runtime spec inventory does not exactly cover the plan")
        return errors
    for row, cell in zip(rows, cells):
        expected = (paths.root / cell["runtime_spec_path"]).resolve()
        if not isinstance(row, dict) \
                or set(row) != {"cell_id", "binding"} \
                or not _optional_matches(row.get("binding"), expected):
            errors.append(f"runtime spec CAS changed: {cell['cell_id']}")
    return errors


def validate_packet(packet: dict[str, Any], *, paths: Paths | None = None,
                    production_checks: bool = True,
                    bridge_ids: Iterable[str] = BRIDGE_IDS,
                    _observed_results: list[dict[str, Any]] | None = None) -> list[str]:
    paths = paths or production_paths()
    errors: list[str] = []
    expected_keys = {
        "schema", "created_at", "generation_id", "plan_sha256", "live_bindings",
        "runtime_spec_inventory", "prepared_runtime_packet", "prepared_packet_sha256",
        "gc_transition_authority", "terminal_seal", "reset_rows", "reset_ids_sha256",
        "attempts_before_total", "last_resource_stop_before", "staged_state",
        "staged_campaign", "result_archives", "result_ids_sha256",
        "checkpoint_bridge_ids", "completed_evidence_mutation_permitted",
        "source_deletion_permitted", "automatic_rollback_requires_no_new_output",
        "packet_sha256",
    }
    if set(packet) != expected_keys:
        errors.append("forward packet keys differ")
    if packet.get("schema") != SCHEMA \
            or packet.get("packet_sha256") != _hash_value(_without(packet, "packet_sha256")):
        errors.append("forward packet schema/hash is invalid")
    ready, blockers = _quiescent(paths)
    if not ready:
        errors.extend(f"quiescence: {row}" for row in blockers)
    try:
        plan, state, campaign, control = (
            _read_json(paths.plan), _read_json(paths.state), _read_json(paths.campaign),
            _read_json(paths.control),
        )
    except ForwardRecoveryError as exc:
        return errors + [str(exc)]
    if packet.get("plan_sha256") != plan.get("plan_sha256"):
        errors.append("plan identity differs")
    generation_id = packet.get("generation_id")
    generation: Path | None = None
    if not isinstance(generation_id, str) or len(generation_id) != 64 \
            or any(character not in "0123456789abcdef" for character in generation_id):
        errors.append("generation identity is invalid")
    else:
        generation = paths.stage_root / "generations" / generation_id
    bindings = packet.get("live_bindings")
    required_live = {
        "plan": paths.plan, "state": paths.state, "campaign": paths.campaign,
        "control": paths.control, "registry": paths.registry,
        "launch_agent": paths.launch_agent, "accelerated_queue": paths.accelerated_queue,
        "accelerated_autoresume": paths.accelerated_autoresume,
        "forward_recovery": Path(__file__),
    }
    optional_live = {
        "active_marker": paths.active_marker, "overlay": paths.overlay,
        "predecessor_journal": paths.predecessor_journal,
        "canonical_reentry_packet": paths.canonical_reentry_packet,
        "pid_file": paths.pid_file,
    }
    if not isinstance(bindings, dict) \
            or set(bindings) != set(required_live) | set(optional_live):
        errors.append("live CAS binding keys differ")
    else:
        for name, path in required_live.items():
            if not _artifact_matches(bindings[name], path):
                errors.append(f"live CAS binding changed: {name}")
        for name, path in optional_live.items():
            if not _optional_matches(bindings[name], path):
                errors.append(f"optional live CAS binding changed: {name}")
    errors.extend(_validate_runtime_inventory(packet.get("runtime_spec_inventory"), plan, paths))
    prepared_ref = packet.get("prepared_runtime_packet")
    prepared: dict[str, Any] | None = None
    prepared_path = generation / "pending_runtime_packet.json" if generation else None
    if prepared_path is None or not _artifact_matches(prepared_ref, prepared_path):
        errors.append("generation-specific prepared packet artifact changed")
    else:
        try:
            prepared = _read_json(Path(prepared_ref["path"]))
            if prepared.get("packet_sha256") != packet.get("prepared_packet_sha256"):
                errors.append("prepared packet semantic identity differs")
            errors.extend(f"prepared packet: {row}" for row in reentry.validate_packet(prepared))
        except (ForwardRecoveryError, reentry.ReentryError) as exc:
            errors.append(f"prepared packet cannot be validated: {exc}")
    authority: dict[str, Any] | None = None
    try:
        if not _artifact_matches(packet.get("gc_transition_authority"), paths.gc_authority):
            errors.append("GC transition authority artifact changed")
        else:
            authority = gc_transition.validate_authority(paths.gc_authority)
    except Exception as exc:
        errors.append(f"GC transition authority failed validation: {exc}")
    reset_rows = packet.get("reset_rows")
    expected_rows: list[dict[str, Any]] = []
    try:
        expected_rows = _reset_rows(plan, state)
    except ForwardRecoveryError as exc:
        errors.append(str(exc))
    if reset_rows != expected_rows:
        errors.append("reset rows differ from exact live failed-attempt selection")
    if not isinstance(reset_rows, list):
        reset_rows = []
    reset_ids = [row.get("cell_id") for row in reset_rows if isinstance(row, dict)]
    if packet.get("reset_ids_sha256") != _hash_value(reset_ids):
        errors.append("reset ID digest differs")
    if packet.get("attempts_before_total") != sum(
            row.get("before", {}).get("attempts", 0) for row in reset_rows
            if isinstance(row, dict)):
        errors.append("reset historical attempt total differs")
    if packet.get("last_resource_stop_before") != state.get("last_resource_stop"):
        errors.append("last_resource_stop before-value differs")
    try:
        staged_state_ref = packet.get("staged_state")
        staged_campaign_ref = packet.get("staged_campaign")
        staged_state_path = generation / "queue_state.json" if generation else None
        staged_campaign_path = generation / "campaign.json" if generation else None
        if staged_state_path is None or staged_campaign_path is None \
                or not _artifact_matches(staged_state_ref, staged_state_path) \
                or not _artifact_matches(staged_campaign_ref, staged_campaign_path):
            errors.append("staged state/campaign artifact changed")
        else:
            expected_state = _make_reset_state(
                plan, state, control, expected_rows, packet.get("created_at")
            )
            actual_state = _read_json(Path(staged_state_ref["path"]))
            if actual_state != expected_state or actual_state.get("last_resource_stop") is not None:
                errors.append("staged state is not the exact operational reset")
            expected_campaign = _campaign_projection(
                paths, plan, expected_state, packet.get("created_at")
            )
            if _read_json(Path(staged_campaign_ref["path"])) != expected_campaign:
                errors.append("staged campaign is not the exact reset-state projection")
    except (ForwardRecoveryError, TypeError) as exc:
        errors.append(f"staged state/campaign validation failed: {exc}")
    if _observed_results is None:
        try:
            expected_results = _result_inventory(plan, state, paths, bridge_ids)
        except ForwardRecoveryError as exc:
            errors.append(str(exc)); expected_results = []
    else:
        expected_results = _observed_results
    if packet.get("result_archives") != expected_results:
        errors.append("result archive inventory differs from exact live trees")
    archives = packet.get("result_archives")
    if not isinstance(archives, list):
        errors.append("result archive inventory is not a list")
        archives = []
    result_ids = [row.get("cell_id") for row in archives if isinstance(row, dict)]
    if packet.get("result_ids_sha256") != _hash_value(result_ids):
        errors.append("result archive ID digest differs")
    if packet.get("checkpoint_bridge_ids") != sorted(bridge_ids):
        errors.append("checkpoint bridge allowlist differs")
    if prepared is not None:
        terminal_errors = reentry._validate_terminal_seal(
            packet.get("terminal_seal"), plan, state
        )
        errors.extend(f"terminal: {row}" for row in terminal_errors)
        if packet.get("terminal_seal") != prepared.get("terminal_seal"):
            errors.append("terminal seal differs from prepared runtime generation")
    try:
        _exact_incident_checks(
            plan_sha256=plan["plan_sha256"], reset_rows=reset_rows,
            result_rows=packet.get("result_archives", []),
            terminal_count=packet.get("terminal_seal", {}).get("count", -1),
            production=production_checks,
        )
    except ForwardRecoveryError as exc:
        errors.append(str(exc))
    if authority is not None and isinstance(bindings, dict):
        try:
            generation_seed = {
                "plan_sha256": plan["plan_sha256"],
                "state": bindings["state"],
                "prepared_packet_sha256": packet["prepared_packet_sha256"],
                "gc_authority_sha256": authority["authority_sha256"],
                "reset_rows_sha256": _hash_value(expected_rows),
                "result_inventory_sha256": _hash_value(expected_results),
                "runtime_inventory_sha256": _hash_value(
                    packet.get("runtime_spec_inventory")
                ),
                "live_bindings_sha256": _hash_value(bindings),
            }
            if generation_id != _hash_value(generation_seed):
                errors.append("generation identity differs from exact staged inputs")
        except (KeyError, TypeError, ValueError):
            errors.append("generation identity cannot be reconstructed")
    if packet.get("completed_evidence_mutation_permitted") is not False \
            or packet.get("source_deletion_permitted") is not False \
            or packet.get("automatic_rollback_requires_no_new_output") is not True:
        errors.append("forward packet safety policy differs")
    return errors


def adversarial_audit(packet: dict[str, Any] | None = None, *,
                      paths: Paths | None = None, production_checks: bool = True,
                      bridge_ids: Iterable[str] = BRIDGE_IDS) -> dict[str, Any]:
    paths = paths or production_paths()
    recovery_lease, queue_lease, heavy_lease = _acquire_all(paths)
    try:
        ready, blockers = _quiescent(paths)
        if not ready:
            raise ForwardRecoveryError("adversarial audit requires quiescence: "
                                       + "; ".join(blockers))
        packet = packet or _read_json(paths.packet)
        plan, state = _read_json(paths.plan), _read_json(paths.state)
        observed_results = _result_inventory(plan, state, paths, bridge_ids)
        baseline = validate_packet(
            packet, paths=paths, production_checks=production_checks,
            bridge_ids=bridge_ids, _observed_results=observed_results,
        )
        if baseline:
            raise ForwardRecoveryError("cannot adversarially audit invalid baseline: "
                                       + "; ".join(baseline))

        def candidate() -> dict[str, Any]:
            return copy.deepcopy(packet)

        def seal(row: dict[str, Any]) -> None:
            row["packet_sha256"] = _hash_value(_without(row, "packet_sha256"))

        probes: list[tuple[str, dict[str, Any], str]] = []
        row = candidate(); row["reset_rows"] = row["reset_rows"][:-1]; seal(row)
        probes.append(("omit-reset", row, "reset rows differ"))
        row = candidate(); row["result_archives"] = row["result_archives"][:-1]; seal(row)
        probes.append(("omit-result-archive", row, "result archive inventory differs"))
        row = candidate(); row["checkpoint_bridge_ids"] = []; seal(row)
        probes.append(("drop-checkpoint-bridges", row, "checkpoint bridge allowlist differs"))
        row = candidate(); row["source_deletion_permitted"] = True; seal(row)
        probes.append(("permit-source-deletion", row, "safety policy differs"))
        row = candidate(); row["live_bindings"]["state"] = row["live_bindings"]["plan"]; seal(row)
        probes.append(("redirect-state-cas", row, "live CAS binding changed: state"))
        row = candidate(); row["generation_id"] = "0" * 64; seal(row)
        probes.append(("redirect-generation", row,
                       "generation identity differs from exact staged inputs"))
        results = []
        for name, altered, fragment in probes:
            errors = validate_packet(
                altered, paths=paths, production_checks=production_checks,
                bridge_ids=bridge_ids, _observed_results=observed_results,
            )
            if not any(fragment in error for error in errors):
                raise ForwardRecoveryError(f"adversarial probe escaped: {name}")
            results.append({"name": name, "rejected": True,
                            "expected_error_fragment": fragment,
                            "error_count": len(errors)})
        return {"schema": "hawking.doctor_v5_forward_recovery_adversarial_audit.v1",
                "ok": True, "physical_result_inventory_passes": 1,
                "probe_count": len(results), "probes": results}
    finally:
        heavy_lease.close(); queue_lease.close(); recovery_lease.close()


def _acquire(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise ForwardRecoveryError(f"required owner-free lease is held: {path}") from exc
    return handle


def _acquire_all(paths: Paths) -> tuple[IO[str], IO[str], IO[str]]:
    acquired: list[IO[str]] = []
    try:
        for path in (paths.transaction_lock, paths.queue_lock, paths.heavy_lock):
            acquired.append(_acquire(path))
        return acquired[0], acquired[1], acquired[2]
    except BaseException:
        for handle in reversed(acquired):
            handle.close()
        raise


def _restore_optional(target: Path, source: Path, existed: bool) -> None:
    if existed:
        if not source.is_file() or source.is_symlink():
            raise ForwardRecoveryError(f"rollback artifact is absent/unsafe: {source}")
        _replace_file(source, target)
    elif _lexists(target):
        target.unlink()
        _fsync_dir(target.parent)


def _configure_launch_agent(paths: Paths,
                            intent: Callable[[str], None] | None = None) -> None:
    raw = _accelerated_launch_agent_bytes(paths)
    if intent:
        intent("configure-launch-agent")
    _atomic_bytes(paths.launch_agent, raw)
    _reload_launch_agent(paths, intent=intent)


def _accelerated_launch_agent_bytes(paths: Paths) -> bytes:
    with paths.launch_agent.open("rb") as handle:
        document = plistlib.load(handle)
    argv = document.get("ProgramArguments")
    if not isinstance(argv, list) or len(argv) < 2:
        raise ForwardRecoveryError("LaunchAgent ProgramArguments is invalid")
    document["ProgramArguments"] = [
        argv[0], str(Path(__file__).resolve()), "recover",
    ]
    return plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=True)


def _reload_launch_agent(paths: Paths,
                         intent: Callable[[str], None] | None = None) -> None:
    domain = f"gui/{os.getuid()}"
    label = "com.hawking.doctorv5ultra.autoresume"
    if intent:
        intent("launchctl-bootout")
    subprocess.run(["launchctl", "bootout", f"{domain}/{label}"],
                   capture_output=True, check=False)
    if intent:
        intent("launchctl-bootstrap")
    result = subprocess.run(
        ["launchctl", "bootstrap", domain, str(paths.launch_agent)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise ForwardRecoveryError("cannot bootstrap accelerated LaunchAgent: "
                                   + result.stderr.strip())


def _marker_document(paths: Paths, active_packet: dict[str, Any],
                     overlay: dict[str, Any]) -> dict[str, Any]:
    marker = {
        "schema": reentry.MARKER_SCHEMA, "activated_at": _now(),
        "overlay_path": str(paths.overlay.resolve()),
        "overlay_sha256": overlay["overlay_sha256"],
        "pending_runtime_generation_sha256": active_packet["packet_sha256"],
        "accelerated_queue": _artifact(paths.accelerated_queue),
        "accelerated_autoresume": _artifact(paths.accelerated_autoresume),
    }
    marker["marker_sha256"] = _hash_value(marker)
    return marker


def _write_marker(paths: Paths, active_packet: dict[str, Any],
                  overlay: dict[str, Any]) -> dict[str, Any]:
    marker = _marker_document(paths, active_packet, overlay)
    _atomic_json(paths.active_marker, marker)
    return marker


def _bridge_live_baseline(result_row: dict[str, Any], live: Path) -> dict[str, Any]:
    subtree = result_row["strand_ladder"]
    baseline_entries = [{"kind": "directory", "relative_path": "strand_ladder"}]
    for child in subtree["entries"]:
        value = dict(child)
        value["relative_path"] = "strand_ladder/" + child["relative_path"]
        baseline_entries.append(value)
    return {
        "path": str(live.resolve()), "entry_count": len(baseline_entries),
        "entries_sha256": _hash_value(baseline_entries), "entries": baseline_entries,
    }


def _absence_inventory(packet: dict[str, Any], paths: Paths) -> list[dict[str, str]]:
    """Bind every pending Qwen result root that was absent before promotion."""
    prepared = _read_json(Path(packet["prepared_runtime_packet"]["path"]))
    pending = [row.get("cell_id") for row in prepared.get("pending_runtime_specs", [])
               if isinstance(row, dict)]
    archived = [row.get("cell_id") for row in packet.get("result_archives", [])
                if isinstance(row, dict)]
    if any(not isinstance(cell_id, str) for cell_id in pending + archived) \
            or len(pending) != len(set(pending)) or len(archived) != len(set(archived)) \
            or not set(archived).issubset(set(pending)):
        raise ForwardRecoveryError("pending/result rollback identity sets are invalid")
    return [
        {"cell_id": cell_id, "live": str((paths.results / cell_id).resolve())}
        for cell_id in sorted(set(pending) - set(archived))
    ]


def _result_move_plan(packet: dict[str, Any], paths: Paths,
                      backup: Path) -> list[dict[str, Any]]:
    moves: list[dict[str, Any]] = []
    for result_row in packet["result_archives"]:
        cell_id = result_row["cell_id"]
        live = paths.results / cell_id
        archive = backup / "results" / cell_id
        row: dict[str, Any] = {
            "cell_id": cell_id, "live": str(live.resolve()),
            "archive": str(archive.resolve()),
            "checkpoint_bridge": result_row["checkpoint_bridge"],
        }
        if result_row["checkpoint_bridge"]:
            row["live_baseline"] = _bridge_live_baseline(result_row, live)
        moves.append(row)
    return moves


def _backup_specs(paths: Paths) -> list[tuple[str, Path, str, bool]]:
    return [
        ("state.json", paths.state, "state", False),
        ("campaign.json", paths.campaign, "campaign", False),
        ("control.json", paths.control, "control", False),
        ("registry.json", paths.registry, "registry", False),
        ("canonical_packet.json", paths.canonical_reentry_packet,
         "canonical_reentry_packet", True),
        ("marker.json", paths.active_marker, "active_marker", True),
        ("overlay.json", paths.overlay, "overlay", True),
        ("predecessor_journal.json", paths.predecessor_journal,
         "predecessor_journal", True),
        ("pid.json", paths.pid_file, "pid_file", True),
        ("launch_agent.plist", paths.launch_agent, "launch_agent", False),
    ]


def _confined_backup_root(paths: Paths, value: Any, *, must_exist: bool) -> Path:
    if not isinstance(value, str):
        raise ForwardRecoveryError("rollback root is not a path")
    candidate = Path(value)
    parent = (paths.stage_root / "rollback").resolve()
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise ForwardRecoveryError(f"rollback root is absent: {candidate}") from exc
    if resolved.parent != parent or candidate.absolute() != resolved:
        raise ForwardRecoveryError("rollback root escapes its exact transaction parent")
    if must_exist:
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ForwardRecoveryError("rollback root is not a real directory")
    return resolved


def _copy_backup(source: Path, target: Path, source_binding: dict[str, Any]) \
        -> dict[str, Any]:
    _replace_file(source, target)
    observed = _artifact(target)
    if observed["sha256"] != source_binding.get("sha256") \
            or observed["bytes"] != source_binding.get("bytes"):
        raise ForwardRecoveryError(f"rollback backup differs from source CAS: {source}")
    return observed


def _regular_mode(path: Path) -> int:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ForwardRecoveryError(f"mode-bound path is not a regular file: {path}")
    return stat.S_IMODE(info.st_mode)


def _backup_transaction(paths: Paths, packet: dict[str, Any], backup: Path) \
        -> dict[str, Any]:
    backup = _confined_backup_root(paths, str(backup), must_exist=False)
    backup.mkdir(parents=True, exist_ok=False)
    bindings = packet["live_bindings"]
    file_rows: list[dict[str, Any]] = []
    for name, source, binding_name, optional in _backup_specs(paths):
        binding = bindings[binding_name]
        destination = backup / name
        if optional:
            existed = binding.get("exists") is True
            source_artifact = binding.get("artifact") if existed else None
        else:
            existed = True
            source_artifact = binding
        if existed:
            if not isinstance(source_artifact, dict):
                raise ForwardRecoveryError(f"backup source binding is invalid: {binding_name}")
            backup_binding = {
                "path": str(destination.absolute()), "exists": True,
                "artifact": _copy_backup(source, destination, source_artifact),
            }
        else:
            if _lexists(destination):
                raise ForwardRecoveryError(f"absent rollback backup unexpectedly exists: {name}")
            backup_binding = {"path": str(destination.absolute()), "exists": False}
        file_rows.append({
            "name": name, "target": str(source.resolve()),
            "binding_name": binding_name, "optional": optional,
            "source_binding": binding,
            "source_mode": _regular_mode(source) if existed else None,
            "backup_binding": backup_binding,
            "backup_mode": _regular_mode(destination) if existed else None,
        })
    prepared = _read_json(Path(packet["prepared_runtime_packet"]["path"]))
    runtime_rows: list[dict[str, Any]] = []
    runtime_root = backup / "runtime_specs"
    runtime_root.mkdir(parents=True, exist_ok=False)
    for row in prepared["pending_runtime_specs"]:
        target = Path(row["target"])
        destination = runtime_root / f"{row['cell_id']}.json"
        runtime_rows.append({
            "cell_id": row["cell_id"], "target": str(target.resolve()),
            "source_binding": row["before"],
            "backup": _copy_backup(target, destination, row["before"]),
            "source_mode": _regular_mode(target),
            "backup_mode": _regular_mode(destination),
        })
    manifest: dict[str, Any] = {
        "schema": ROLLBACK_MANIFEST_SCHEMA, "created_at": _now(),
        "forward_packet_sha256": packet["packet_sha256"],
        "rollback_root": str(backup), "file_backups": file_rows,
        "runtime_spec_backups": runtime_rows,
        "result_moves": _result_move_plan(packet, paths, backup),
        "absent_results": _absence_inventory(packet, paths),
        "source_deletion_permitted": False,
    }
    manifest["manifest_sha256"] = _hash_value(manifest)
    _atomic_json(backup / "rollback_manifest.json", manifest)
    _validate_rollback_manifest(paths, packet, backup, manifest)
    return manifest


def _validate_rollback_manifest(paths: Paths, packet: dict[str, Any], backup: Path,
                                manifest: dict[str, Any]) -> None:
    expected_keys = {
        "schema", "created_at", "forward_packet_sha256", "rollback_root",
        "file_backups", "runtime_spec_backups", "result_moves", "absent_results",
        "source_deletion_permitted", "manifest_sha256",
    }
    if set(manifest) != expected_keys or manifest.get("schema") != ROLLBACK_MANIFEST_SCHEMA \
            or manifest.get("manifest_sha256") != _hash_value(
                _without(manifest, "manifest_sha256")
            ) or manifest.get("forward_packet_sha256") != packet.get("packet_sha256") \
            or manifest.get("rollback_root") != str(backup) \
            or manifest.get("source_deletion_permitted") is not False:
        raise ForwardRecoveryError("rollback manifest schema/hash/identity is invalid")
    expected_moves = _result_move_plan(packet, paths, backup)
    expected_absent = _absence_inventory(packet, paths)
    if manifest.get("result_moves") != expected_moves \
            or manifest.get("absent_results") != expected_absent:
        raise ForwardRecoveryError("rollback result/absence plan differs")
    file_rows = manifest.get("file_backups")
    specs = _backup_specs(paths)
    if not isinstance(file_rows, list) or len(file_rows) != len(specs):
        raise ForwardRecoveryError("rollback file backup inventory differs")
    bindings = packet["live_bindings"]
    for row, (name, target, binding_name, optional) in zip(file_rows, specs):
        expected_row_keys = {
            "name", "target", "binding_name", "optional", "source_binding",
            "source_mode", "backup_binding", "backup_mode",
        }
        if not isinstance(row, dict) or set(row) != expected_row_keys \
                or row.get("name") != name or row.get("target") != str(target.resolve()) \
                or row.get("binding_name") != binding_name \
                or row.get("optional") is not optional \
                or row.get("source_binding") != bindings[binding_name]:
            raise ForwardRecoveryError(f"rollback file backup row differs: {name}")
        destination = backup / name
        source_binding = bindings[binding_name]
        if optional and source_binding.get("exists") is False:
            if row.get("backup_binding") != {
                    "path": str(destination.absolute()), "exists": False} \
                    or _lexists(destination) or row.get("source_mode") is not None \
                    or row.get("backup_mode") is not None:
                raise ForwardRecoveryError(f"absent rollback binding differs: {name}")
        else:
            source_artifact = (source_binding.get("artifact") if optional
                               else source_binding)
            expected_backup = row.get("backup_binding")
            if not isinstance(expected_backup, dict) \
                    or expected_backup.get("path") != str(destination.absolute()) \
                    or expected_backup.get("exists") is not True \
                    or set(expected_backup) != {"path", "exists", "artifact"} \
                    or not _artifact_matches(expected_backup.get("artifact"), destination) \
                    or not _artifact_content_matches(source_artifact, destination) \
                    or not isinstance(row.get("source_mode"), int) \
                    or row.get("backup_mode") != row.get("source_mode") \
                    or _regular_mode(destination) != row.get("backup_mode"):
                raise ForwardRecoveryError(f"rollback backup artifact differs: {name}")
    prepared = _read_json(Path(packet["prepared_runtime_packet"]["path"]))
    runtime_rows = manifest.get("runtime_spec_backups")
    pending = prepared.get("pending_runtime_specs")
    if not isinstance(runtime_rows, list) or not isinstance(pending, list) \
            or len(runtime_rows) != len(pending):
        raise ForwardRecoveryError("rollback runtime backup inventory differs")
    for manifest_row, prepared_row in zip(runtime_rows, pending):
        cell_id = prepared_row["cell_id"]
        target = Path(prepared_row["target"])
        destination = backup / "runtime_specs" / f"{cell_id}.json"
        expected = {
            "cell_id": cell_id, "target": str(target.resolve()),
            "source_binding": prepared_row["before"],
            "backup": manifest_row.get("backup") if isinstance(manifest_row, dict) else None,
            "source_mode": manifest_row.get("source_mode")
            if isinstance(manifest_row, dict) else None,
            "backup_mode": manifest_row.get("backup_mode")
            if isinstance(manifest_row, dict) else None,
        }
        if not isinstance(manifest_row, dict) or set(manifest_row) != set(expected) \
                or any(manifest_row.get(key) != value for key, value in expected.items()) \
                or not _artifact_matches(manifest_row.get("backup"), destination) \
                or not _artifact_content_matches(prepared_row["before"], destination) \
                or not isinstance(manifest_row.get("source_mode"), int) \
                or manifest_row.get("backup_mode") != manifest_row.get("source_mode") \
                or _regular_mode(destination) != manifest_row.get("backup_mode"):
            raise ForwardRecoveryError(f"rollback runtime backup differs: {cell_id}")


def _validate_wal(backup: Path, forward_packet_sha256: str) -> list[dict[str, Any]]:
    root = backup / "wal"
    if not root.is_dir() or root.is_symlink():
        raise ForwardRecoveryError("transaction WAL directory is absent/unsafe")
    paths = sorted(root.glob("*.json"))
    if any(path.name != f"{index:08d}.json" for index, path in enumerate(paths, 1)):
        raise ForwardRecoveryError("transaction WAL sequence has a gap or extra entry")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    expected_keys = {
        "schema", "created_at", "forward_packet_sha256", "index", "phase",
        "operation", "previous_entry_sha256", "details", "entry_sha256",
    }
    for index, path in enumerate(paths, 1):
        row = _read_json(path)
        if set(row) != expected_keys or row.get("schema") != WAL_SCHEMA \
                or row.get("index") != index \
                or row.get("forward_packet_sha256") != forward_packet_sha256 \
                or row.get("previous_entry_sha256") != previous \
                or row.get("entry_sha256") != _hash_value(
                    _without(row, "entry_sha256")
                ):
            raise ForwardRecoveryError(f"transaction WAL entry is invalid: {path.name}")
        previous = row["entry_sha256"]
        entries.append(row)
    return entries


def _append_wal(backup: Path, forward_packet_sha256: str, *, phase: str,
                operation: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    root = backup / "wal"
    root.mkdir(parents=True, exist_ok=True)
    entries = _validate_wal(backup, forward_packet_sha256)
    index = len(entries) + 1
    path = root / f"{index:08d}.json"
    if _lexists(path):
        raise ForwardRecoveryError("refusing to replace an immutable WAL entry")
    row: dict[str, Any] = {
        "schema": WAL_SCHEMA, "created_at": _now(),
        "forward_packet_sha256": forward_packet_sha256, "index": index,
        "phase": phase, "operation": operation,
        "previous_entry_sha256": entries[-1]["entry_sha256"] if entries else None,
        "details": details or {},
    }
    row["entry_sha256"] = _hash_value(row)
    _atomic_json(path, row)
    if _read_json(path) != row:
        raise ForwardRecoveryError("durable WAL entry differs after write")
    return row


def _seal_journal(journal: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(journal)
    value["updated_at"] = _now()
    value["journal_sha256"] = _hash_value(_without(value, "journal_sha256"))
    return value


def _write_journal(paths: Paths, journal: dict[str, Any], wal: dict[str, Any]) \
        -> dict[str, Any]:
    value = copy.deepcopy(journal)
    value["wal_index"] = wal["index"]
    value["wal_entry_sha256"] = wal["entry_sha256"]
    value["phase"] = wal["phase"]
    value["operation"] = wal["operation"]
    value = _seal_journal(value)
    _atomic_json(paths.journal, value)
    return value


def _journal_step(paths: Paths, backup: Path, journal: dict[str, Any], *, phase: str,
                  operation: str, details: dict[str, Any] | None = None) \
        -> dict[str, Any]:
    wal = _append_wal(
        backup, journal["forward_packet_sha256"], phase=phase,
        operation=operation, details=details,
    )
    return _write_journal(paths, journal, wal)


def _live_step(paths: Paths, backup: Path, journal: dict[str, Any], *, phase: str,
               operation: str, action: Callable[[], None],
               verify: Callable[[], bool]) -> dict[str, Any]:
    journal = _journal_step(
        paths, backup, journal, phase=f"{phase}-intent", operation=operation
    )
    _fault(f"before:{phase}:{operation}")
    action()
    if not verify():
        raise ForwardRecoveryError(f"post-CAS verification failed: {operation}")
    journal = _journal_step(
        paths, backup, journal, phase=f"{phase}-done", operation=operation
    )
    _fault(f"after:{phase}:{operation}")
    return journal


def _stage_activation_bundle(paths: Paths, packet: dict[str, Any], backup: Path,
                             manifest: dict[str, Any]) -> dict[str, Any]:
    staged = backup / "staged"
    staged.mkdir(parents=True, exist_ok=False)
    overlay = _build_staged_overlay(paths, packet)
    overlay_path = staged / "overlay.json"
    _atomic_json(overlay_path, overlay)
    launch_agent_path = staged / "launch_agent.plist"
    _atomic_bytes(launch_agent_path, _accelerated_launch_agent_bytes(paths))
    prepared = _read_json(Path(packet["prepared_runtime_packet"]["path"]))
    promotions = {
        "registry": prepared["registry"]["staged"],
        "runtime_specs": [
            {"cell_id": row["cell_id"], "target": row["target"],
             "staged": row["staged"]}
            for row in prepared["pending_runtime_specs"]
        ],
        "state": packet["staged_state"], "campaign": packet["staged_campaign"],
        "overlay": _artifact(overlay_path),
        "launch_agent": _artifact(launch_agent_path),
    }
    receipt: dict[str, Any] = {
        "schema": ACTIVATION_SCHEMA, "created_at": _now(), "status": "intent",
        "forward_packet_sha256": packet["packet_sha256"],
        "rollback_root": str(backup),
        "rollback_manifest": _artifact(backup / "rollback_manifest.json"),
        "planned_promotions": promotions,
        "result_moves": manifest["result_moves"],
        "absent_results": manifest["absent_results"],
        "completed_evidence_mutated": False,
        "source_deletion_permitted": False,
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    receipt_path = backup / "activation_receipt.json"
    _atomic_json(receipt_path, receipt)
    active_packet = copy.deepcopy(prepared)
    active_packet["forward_recovery"] = _artifact(receipt_path)
    active_packet["packet_sha256"] = _hash_value(
        _without(active_packet, "packet_sha256")
    )
    active_packet_path = staged / "active_runtime_packet.json"
    _atomic_json(active_packet_path, active_packet)
    marker = _marker_document(paths, active_packet, overlay)
    marker_path = staged / "marker.json"
    _atomic_json(marker_path, marker)
    return {
        "receipt": receipt, "receipt_path": receipt_path,
        "overlay": overlay, "overlay_path": overlay_path,
        "launch_agent_path": launch_agent_path,
        "active_packet": active_packet, "active_packet_path": active_packet_path,
        "marker": marker, "marker_path": marker_path,
    }


def _initial_journal(paths: Paths, packet: dict[str, Any], backup: Path,
                     manifest: dict[str, Any], bundle: dict[str, Any]) \
        -> dict[str, Any]:
    created_at = _now()
    journal: dict[str, Any] = {
        "schema": JOURNAL_SCHEMA, "created_at": created_at,
        "updated_at": created_at, "status": "promotion-intent",
        "phase": "transaction-ready", "operation": "transaction-ready",
        "wal_index": 0, "wal_entry_sha256": None,
        "forward_packet_sha256": packet["packet_sha256"],
        "plan_sha256": packet["plan_sha256"],
        "generation_id": packet["generation_id"],
        "backup_root": str(backup),
        "rollback_manifest": _artifact(backup / "rollback_manifest.json"),
        "activation_receipt": _artifact(bundle["receipt_path"]),
        "active_runtime_packet": _artifact(bundle["active_packet_path"]),
        "staged_overlay": _artifact(bundle["overlay_path"]),
        "staged_launch_agent": _artifact(bundle["launch_agent_path"]),
        "staged_marker": _artifact(bundle["marker_path"]),
        "result_moves": manifest["result_moves"],
        "absent_results": manifest["absent_results"],
        "live_marker": None, "supervisor_pid": None,
        "supervisor_started_at": None, "rollback_receipt": None,
        "source_deletion_permitted": False, "journal_sha256": "",
    }
    wal = _append_wal(
        backup, packet["packet_sha256"], phase="transaction-ready",
        operation="transaction-ready",
        details={"rollback_manifest_sha256": manifest["manifest_sha256"]},
    )
    return _write_journal(paths, journal, wal)


def _validate_activation_receipt(packet: dict[str, Any], backup: Path,
                                 manifest: dict[str, Any], receipt: dict[str, Any],
                                 bundle_refs: dict[str, Any]) -> None:
    expected_keys = {
        "schema", "created_at", "status", "forward_packet_sha256",
        "rollback_root", "rollback_manifest", "planned_promotions",
        "result_moves", "absent_results", "completed_evidence_mutated",
        "source_deletion_permitted", "receipt_sha256",
    }
    if set(receipt) != expected_keys or receipt.get("schema") != ACTIVATION_SCHEMA \
            or receipt.get("status") != "intent" \
            or receipt.get("receipt_sha256") != _hash_value(
                _without(receipt, "receipt_sha256")
            ) or receipt.get("forward_packet_sha256") != packet.get("packet_sha256") \
            or receipt.get("rollback_root") != str(backup) \
            or receipt.get("rollback_manifest") != bundle_refs["rollback_manifest"] \
            or receipt.get("result_moves") != manifest.get("result_moves") \
            or receipt.get("absent_results") != manifest.get("absent_results") \
            or receipt.get("completed_evidence_mutated") is not False \
            or receipt.get("source_deletion_permitted") is not False:
        raise ForwardRecoveryError("activation intent receipt differs")
    prepared = _read_json(Path(packet["prepared_runtime_packet"]["path"]))
    expected_promotions = {
        "registry": prepared["registry"]["staged"],
        "runtime_specs": [
            {"cell_id": row["cell_id"], "target": row["target"],
             "staged": row["staged"]}
            for row in prepared["pending_runtime_specs"]
        ],
        "state": packet["staged_state"], "campaign": packet["staged_campaign"],
        "overlay": bundle_refs["staged_overlay"],
        "launch_agent": bundle_refs["staged_launch_agent"],
    }
    if receipt.get("planned_promotions") != expected_promotions:
        raise ForwardRecoveryError("activation promotion plan differs")


def _archive_results(packet: dict[str, Any], paths: Paths, backup: Path,
                     moved: list[dict[str, Any]]) -> dict[str, Any]:
    bridge_baselines: dict[str, Any] = {}
    plan = _result_move_plan(packet, paths, backup)
    if moved and moved != plan:
        raise ForwardRecoveryError("result move inventory differs from deterministic plan")
    if not moved:
        moved.extend(copy.deepcopy(plan))  # intent is recorded before the first rename
    for row, move in zip(packet["result_archives"], moved):
        cell_id = row["cell_id"]
        live = Path(move["live"])
        archived = Path(move["archive"])
        archived.parent.mkdir(parents=True, exist_ok=True)
        os.replace(live, archived)
        _fsync_dir(live.parent); _fsync_dir(archived.parent)
        if row["checkpoint_bridge"]:
            live.mkdir(parents=True, exist_ok=False)
            _fsync_dir(live.parent)
            os.replace(archived / "strand_ladder", live / "strand_ladder")
            _fsync_dir(live); _fsync_dir(archived)
            baseline = move["live_baseline"]
            bridge_baselines[cell_id] = baseline
    return bridge_baselines


def _promote_results(packet: dict[str, Any], paths: Paths, backup: Path,
                     journal: dict[str, Any]) -> dict[str, Any]:
    moves = journal["result_moves"]
    for result_row, move in zip(packet["result_archives"], moves):
        cell_id = result_row["cell_id"]
        live, archive = Path(move["live"]), Path(move["archive"])
        archive.parent.mkdir(parents=True, exist_ok=True)

        def archive_action(live: Path = live, archive: Path = archive) -> None:
            os.replace(live, archive)
            _fsync_dir(live.parent); _fsync_dir(archive.parent)

        expected_archive = _tree_repath(result_row["tree"], archive)
        journal = _live_step(
            paths, backup, journal, phase="promotion",
            operation=f"archive-result:{cell_id}", action=archive_action,
            verify=lambda archive=archive, expected_archive=expected_archive:
                not _lexists(live) and _tree_matches(expected_archive, archive),
        )
        if result_row["checkpoint_bridge"]:
            journal = _live_step(
                paths, backup, journal, phase="promotion",
                operation=f"create-bridge-root:{cell_id}",
                action=lambda live=live: (live.mkdir(parents=True, exist_ok=False),
                                          _fsync_dir(live.parent)),
                verify=lambda live=live: _tree_matches(_empty_tree(live), live),
            )

            def carry_action(live: Path = live, archive: Path = archive) -> None:
                os.replace(archive / "strand_ladder", live / "strand_ladder")
                _fsync_dir(live); _fsync_dir(archive)

            expected_remainder = _bridge_archive_tree(result_row, archive)
            journal = _live_step(
                paths, backup, journal, phase="promotion",
                operation=f"carry-bridge:{cell_id}", action=carry_action,
                verify=lambda live=live, archive=archive, move=move,
                              expected_remainder=expected_remainder:
                    _tree_matches(move["live_baseline"], live)
                    and _tree_matches(expected_remainder, archive),
            )
    return journal


def _no_new_output(moved: list[dict[str, Any]],
                   absent: Iterable[dict[str, str]] = ()) -> list[str]:
    conflicts: list[str] = []
    for row in moved:
        live = Path(row["live"])
        if row["checkpoint_bridge"]:
            if not _tree_matches(row.get("live_baseline"), live):
                conflicts.append(row["cell_id"])
        elif _lexists(live):
            conflicts.append(row["cell_id"])
    for row in absent:
        if _lexists(Path(row["live"])):
            conflicts.append(row["cell_id"])
    return conflicts


def _move_state(result_row: dict[str, Any], move: dict[str, Any]) -> str:
    live, archive = Path(move["live"]), Path(move["archive"])
    live_exists, archive_exists = _lexists(live), _lexists(archive)
    original_live = result_row["tree"]
    original_archive = _tree_repath(original_live, archive)
    if live_exists and not archive_exists and _tree_matches(original_live, live):
        return "pristine"
    if not live_exists and archive_exists and _tree_matches(original_archive, archive):
        return "archived"
    if result_row["checkpoint_bridge"] and live_exists and archive_exists:
        if _tree_matches(_empty_tree(live), live) \
                and _tree_matches(original_archive, archive):
            return "bridge-root-created"
        if _tree_matches(move["live_baseline"], live) \
                and _tree_matches(_bridge_archive_tree(result_row, archive), archive):
            return "bridge-carried"
    return "invalid"


def _preflight_result_surface(packet: dict[str, Any], moves: list[dict[str, Any]],
                              absent: list[dict[str, str]], *, forward_only: bool) \
        -> list[str]:
    states: list[str] = []
    if len(moves) != len(packet["result_archives"]):
        raise ForwardRecoveryError("result move count differs")
    for result_row, move in zip(packet["result_archives"], moves):
        state = _move_state(result_row, move)
        allowed = ({"archived", "bridge-carried"} if forward_only else {
            "pristine", "archived", "bridge-root-created", "bridge-carried"
        })
        if state not in allowed:
            raise ForwardRecoveryError(
                f"result move has unknown/new evidence state: {result_row['cell_id']} ({state})"
            )
        states.append(state)
    conflicts = [row["cell_id"] for row in absent if _lexists(Path(row["live"]))]
    if conflicts:
        raise ForwardRecoveryError(
            "rollback refused because formerly-absent output exists: "
            + ", ".join(conflicts)
        )
    return states


def _binding_matches_live(binding: dict[str, Any], path: Path, *, optional: bool) -> bool:
    if optional:
        if binding.get("exists") is False:
            return not _lexists(path)
        return binding.get("exists") is True \
            and _artifact_content_matches(binding.get("artifact"), path)
    return _artifact_content_matches(binding, path)


def _forward_file_bindings(paths: Paths, receipt: dict[str, Any],
                           journal: dict[str, Any]) -> dict[str, dict[str, Any]]:
    promotions = receipt["planned_promotions"]
    return {
        "state.json": promotions["state"],
        "campaign.json": promotions["campaign"],
        "registry.json": promotions["registry"],
        "canonical_packet.json": journal["active_runtime_packet"],
        "marker.json": journal["staged_marker"],
        "overlay.json": promotions["overlay"],
        "launch_agent.plist": promotions["launch_agent"],
    }


def _preflight_live_files(paths: Paths, manifest: dict[str, Any],
                          receipt: dict[str, Any], journal: dict[str, Any],
                          *, allow_original: bool) -> None:
    forward = _forward_file_bindings(paths, receipt, journal)
    for row in manifest["file_backups"]:
        target = Path(row["target"])
        original_ok = _binding_matches_live(
            row["source_binding"], target, optional=row["optional"]
        )
        forward_ref = forward.get(row["name"])
        forward_ok = (original_ok if forward_ref is None else
                      _artifact_content_matches(forward_ref, target))
        if not forward_ok and not (allow_original and original_ok):
            raise ForwardRecoveryError(
                f"live rollback CAS differs before restore: {row['name']}"
            )
    forward_specs = {
        row["cell_id"]: row["staged"]
        for row in receipt["planned_promotions"]["runtime_specs"]
    }
    for row in manifest["runtime_spec_backups"]:
        target = Path(row["target"])
        if not _artifact_content_matches(forward_specs[row["cell_id"]], target) \
                and not (allow_original and _artifact_content_matches(
                    row["source_binding"], target
                )):
            raise ForwardRecoveryError(
                f"live runtime CAS differs before restore: {row['cell_id']}"
            )


def _restore_transaction(paths: Paths, packet: dict[str, Any], backup: Path,
                         manifest: dict[str, Any], receipt: dict[str, Any],
                         journal: dict[str, Any], *, forward_only: bool) \
        -> dict[str, Any]:
    states = _preflight_result_surface(
        packet, manifest["result_moves"], manifest["absent_results"],
        forward_only=forward_only,
    )
    _preflight_live_files(
        paths, manifest, receipt, journal, allow_original=not forward_only
    )
    journal["status"] = "rolling-back"
    journal["live_marker"] = None
    journal = _journal_step(
        paths, backup, journal, phase="rollback-decision",
        operation="rollback-decision",
        details={"forward_only": forward_only, "live_marker_cleared": True},
    )
    for result_row, move in reversed(list(zip(
            packet["result_archives"], manifest["result_moves"]))):
        cell_id = result_row["cell_id"]
        live, archive = Path(move["live"]), Path(move["archive"])
        state = _move_state(result_row, move)
        if state == "bridge-carried":
            def return_ladder(live: Path = live, archive: Path = archive) -> None:
                os.replace(live / "strand_ladder", archive / "strand_ladder")
                _fsync_dir(live); _fsync_dir(archive)
            journal = _live_step(
                paths, backup, journal, phase="rollback",
                operation=f"return-bridge:{cell_id}", action=return_ladder,
                verify=lambda live=live, archive=archive, result_row=result_row:
                    _tree_matches(_empty_tree(live), live)
                    and _tree_matches(_tree_repath(result_row["tree"], archive), archive),
            )
            state = "bridge-root-created"
        if state == "bridge-root-created":
            journal = _live_step(
                paths, backup, journal, phase="rollback",
                operation=f"remove-empty-bridge-root:{cell_id}",
                action=lambda live=live: (live.rmdir(), _fsync_dir(live.parent)),
                verify=lambda live=live: not _lexists(live),
            )
            state = "archived"
        if state == "archived":
            def restore_root(live: Path = live, archive: Path = archive) -> None:
                os.replace(archive, live)
                _fsync_dir(live.parent); _fsync_dir(archive.parent)
            journal = _live_step(
                paths, backup, journal, phase="rollback",
                operation=f"restore-result:{cell_id}", action=restore_root,
                verify=lambda live=live, result_row=result_row:
                    _tree_matches(result_row["tree"], live),
            )
        if _move_state(result_row, move) != "pristine":
            raise ForwardRecoveryError(f"result rollback did not converge: {cell_id}")

    for row in manifest["file_backups"]:
        target = Path(row["target"]); source = backup / row["name"]
        operation = f"restore-file:{row['name']}"
        journal = _live_step(
            paths, backup, journal, phase="rollback", operation=operation,
            action=lambda target=target, source=source, row=row:
                _restore_optional(target, source,
                                  row["backup_binding"].get("exists") is True),
            verify=lambda target=target, row=row: _binding_matches_live(
                row["source_binding"], target, optional=row["optional"]
            ),
        )
    for row in manifest["runtime_spec_backups"]:
        target = Path(row["target"])
        source = backup / "runtime_specs" / f"{row['cell_id']}.json"
        journal = _live_step(
            paths, backup, journal, phase="rollback",
            operation=f"restore-runtime:{row['cell_id']}",
            action=lambda source=source, target=target: _replace_file(source, target),
            verify=lambda target=target, row=row:
                _artifact_content_matches(row["source_binding"], target),
        )
    journal = _journal_step(
        paths, backup, journal, phase="rollback-intent",
        operation="reload-restored-launch-agent",
    )
    _fault("before:rollback:reload-restored-launch-agent")
    _reload_launch_agent(paths)
    journal = _journal_step(
        paths, backup, journal, phase="rollback-done",
        operation="reload-restored-launch-agent",
    )
    _fault("after:rollback:reload-restored-launch-agent")
    return journal


def _start_detached(paths: Paths, marker: dict[str, Any]) -> int:
    env = os.environ.copy()
    env[stacked.ENV_OVERLAY] = marker["overlay_path"]
    env[stacked.ENV_OVERLAY_SHA256] = marker["overlay_sha256"]
    result = subprocess.run(
        [sys.executable, str(paths.accelerated_queue), "start"], cwd=paths.root,
        env=env, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise ForwardRecoveryError("accelerated detached start failed: "
                                   + (result.stderr or result.stdout).strip())
    pid = _verified_owner(paths)
    if pid is None:
        raise ForwardRecoveryError("accelerated detached owner record/identity is invalid")
    return pid


def _verified_owner(paths: Paths) -> int | None:
    if not _lexists(paths.pid_file):
        return None
    try:
        record = _read_json(paths.pid_file)
        plan = _read_json(paths.plan)
    except ForwardRecoveryError:
        return None
    if record.get("schema") != queue.PID_SCHEMA \
            or record.get("version") != queue.VERSION \
            or record.get("plan_sha256") != plan.get("plan_sha256") \
            or record.get("pid_record_sha256") != _hash_value(
                _without(record, "pid_record_sha256")
            ):
        return None
    nonce = record.get("ownership_nonce")
    identity = queue._process_identity(record.get("pid"))
    if identity is None or not isinstance(nonce, str) \
            or queue.NONCE_RE.fullmatch(nonce) is None:
        return None
    command, started = identity
    entrypoint = "doctor_v5_ultra_accelerated_queue.py run" in command
    if started != record.get("process_started") \
            or hashlib.sha256(command.encode()).hexdigest() \
            != record.get("process_command_sha256") \
            or not entrypoint or f"--nonce {nonce}" not in command:
        return None
    return int(record["pid"])


def _journal_keys() -> set[str]:
    return {
        "schema", "created_at", "updated_at", "status", "phase", "operation",
        "wal_index", "wal_entry_sha256", "forward_packet_sha256", "plan_sha256",
        "generation_id", "backup_root", "rollback_manifest", "activation_receipt",
        "active_runtime_packet", "staged_overlay", "staged_launch_agent",
        "staged_marker", "result_moves", "absent_results", "live_marker",
        "supervisor_pid", "supervisor_started_at", "rollback_receipt",
        "source_deletion_permitted", "journal_sha256",
    }


def _replay_wal_tail(journal: dict[str, Any], wal: list[dict[str, Any]],
                     backup: Path) -> None:
    """Apply immutable entries newer than the replaceable journal summary."""
    index = int(journal["wal_index"])
    for row in wal[index:]:
        journal["wal_index"] = row["index"]
        journal["wal_entry_sha256"] = row["entry_sha256"]
        journal["phase"] = row["phase"]
        journal["operation"] = row["operation"]
        phase = row["phase"]
        if phase == "rollback-decision" or phase.startswith("rollback"):
            journal["status"] = "rolling-back"
            journal["live_marker"] = None
        if phase == "rolled-back":
            journal["status"] = "rolled-back"
            receipt_path = backup / "rollback_receipt.json"
            if _lexists(receipt_path):
                journal["rollback_receipt"] = _artifact(receipt_path)
        elif phase.startswith("active"):
            journal["status"] = "active"
            pid = row.get("details", {}).get("supervisor_pid")
            if isinstance(pid, int) and not isinstance(pid, bool):
                journal["supervisor_pid"] = pid
        elif phase.startswith("start"):
            journal["status"] = "start-intent"
        elif phase == "files-promoted" or phase.startswith("service"):
            journal["status"] = "files-promoted"


def _validate_transaction_chain(paths: Paths, packet: dict[str, Any],
                                journal: dict[str, Any]) \
        -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if set(journal) != _journal_keys() or journal.get("schema") != JOURNAL_SCHEMA \
            or journal.get("journal_sha256") != _hash_value(
                _without(journal, "journal_sha256")
            ) or journal.get("forward_packet_sha256") != packet.get("packet_sha256") \
            or journal.get("plan_sha256") != packet.get("plan_sha256") \
            or journal.get("generation_id") != packet.get("generation_id") \
            or journal.get("source_deletion_permitted") is not False:
        raise ForwardRecoveryError("forward transaction journal is invalid")
    backup = _confined_backup_root(paths, journal.get("backup_root"), must_exist=True)
    manifest_path = backup / "rollback_manifest.json"
    if not _artifact_matches(journal.get("rollback_manifest"), manifest_path):
        raise ForwardRecoveryError("journal rollback-manifest artifact differs")
    manifest = _read_json(manifest_path)
    _validate_rollback_manifest(paths, packet, backup, manifest)
    wal = _validate_wal(backup, packet["packet_sha256"])
    index = journal.get("wal_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 1 \
            or index > len(wal) \
            or journal.get("wal_entry_sha256") != wal[index - 1]["entry_sha256"]:
        raise ForwardRecoveryError("journal WAL tail is invalid")
    _replay_wal_tail(journal, wal, backup)
    receipt_path = backup / "activation_receipt.json"
    active_path = backup / "staged/active_runtime_packet.json"
    overlay_path = backup / "staged/overlay.json"
    launch_path = backup / "staged/launch_agent.plist"
    marker_path = backup / "staged/marker.json"
    exact_refs = {
        "activation_receipt": receipt_path, "active_runtime_packet": active_path,
        "staged_overlay": overlay_path, "staged_launch_agent": launch_path,
        "staged_marker": marker_path,
    }
    for name, expected in exact_refs.items():
        if not _artifact_matches(journal.get(name), expected):
            raise ForwardRecoveryError(f"journal bundle artifact differs: {name}")
    receipt = _read_json(receipt_path)
    _validate_activation_receipt(packet, backup, manifest, receipt, journal)
    active_packet = _read_json(active_path)
    if active_packet.get("packet_sha256") != _hash_value(
            _without(active_packet, "packet_sha256")) \
            or active_packet.get("forward_recovery") != journal["activation_receipt"]:
        raise ForwardRecoveryError("active runtime packet chain differs")
    marker = _read_json(marker_path)
    overlay = _read_json(overlay_path)
    if marker.get("marker_sha256") != _hash_value(_without(marker, "marker_sha256")) \
            or marker.get("pending_runtime_generation_sha256") \
            != active_packet.get("packet_sha256") \
            or marker.get("overlay_sha256") != overlay.get("overlay_sha256"):
        raise ForwardRecoveryError("staged marker chain differs")
    if journal.get("result_moves") != manifest.get("result_moves") \
            or journal.get("absent_results") != manifest.get("absent_results"):
        raise ForwardRecoveryError("journal result inventory differs")
    live_marker = journal.get("live_marker")
    if live_marker is not None and not _artifact_matches(live_marker, paths.active_marker):
        raise ForwardRecoveryError("live activation marker binding differs")
    return backup, manifest, receipt, wal


def _rollback_receipt(paths: Paths, packet: dict[str, Any], backup: Path,
                      manifest: dict[str, Any], journal: dict[str, Any]) \
        -> dict[str, Any]:
    for result_row, move in zip(packet["result_archives"], manifest["result_moves"]):
        if _move_state(result_row, move) != "pristine":
            raise ForwardRecoveryError("rollback receipt cannot seal unrestored results")
    for row in manifest["absent_results"]:
        if _lexists(Path(row["live"])):
            raise ForwardRecoveryError("rollback receipt found formerly-absent output")
    for row in manifest["file_backups"]:
        if not _binding_matches_live(
                row["source_binding"], Path(row["target"]), optional=row["optional"]):
            raise ForwardRecoveryError(f"rollback poststate differs: {row['name']}")
        if row["source_mode"] is not None \
                and _regular_mode(Path(row["target"])) != row["source_mode"]:
            raise ForwardRecoveryError(f"rollback mode differs: {row['name']}")
    for row in manifest["runtime_spec_backups"]:
        if not _artifact_content_matches(row["source_binding"], Path(row["target"])):
            raise ForwardRecoveryError(f"rollback runtime poststate differs: {row['cell_id']}")
        if _regular_mode(Path(row["target"])) != row["source_mode"]:
            raise ForwardRecoveryError(
                f"rollback runtime mode differs: {row['cell_id']}"
            )
    value: dict[str, Any] = {
        "schema": ROLLBACK_RECEIPT_SCHEMA, "created_at": _now(),
        "status": "rolled-back", "forward_packet_sha256": packet["packet_sha256"],
        "rollback_manifest": journal["rollback_manifest"],
        "restored_result_ids": [row["cell_id"] for row in manifest["result_moves"]],
        "restored_file_count": len(manifest["file_backups"]),
        "restored_runtime_spec_count": len(manifest["runtime_spec_backups"]),
        "source_deletion_permitted": False,
    }
    value["receipt_sha256"] = _hash_value(value)
    return value


def _finish_rollback(paths: Paths, packet: dict[str, Any], backup: Path,
                     manifest: dict[str, Any], journal: dict[str, Any]) \
        -> dict[str, Any]:
    receipt = _rollback_receipt(paths, packet, backup, manifest, journal)
    receipt_path = backup / "rollback_receipt.json"
    _atomic_json(receipt_path, receipt)
    _fault("after:rollback:rollback-receipt")
    journal["status"] = "rolled-back"
    journal["rollback_receipt"] = _artifact(receipt_path)
    journal["live_marker"] = None
    journal["supervisor_pid"] = None
    journal["supervisor_started_at"] = None
    journal = _journal_step(
        paths, backup, journal, phase="rolled-back", operation="rolled-back",
        details={"rollback_receipt_sha256": receipt["receipt_sha256"]},
    )
    return journal


def apply(*, packet_sha256: str, plan_sha256: str, paths: Paths | None = None,
          production_checks: bool = True,
          bridge_ids: Iterable[str] = BRIDGE_IDS) -> dict[str, Any]:
    paths = paths or production_paths()
    _supersession_barrier(paths, operation="apply")
    if _lexists(paths.journal):
        return recover(paths=paths)
    packet = _read_json(paths.packet)
    if packet_sha256 != packet.get("packet_sha256") \
            or plan_sha256 != packet.get("plan_sha256"):
        raise ForwardRecoveryError("both exact activation keys are required")
    errors = validate_packet(packet, paths=paths, production_checks=production_checks,
                             bridge_ids=bridge_ids)
    if errors:
        raise ForwardRecoveryError("activation audit failed: " + "; ".join(errors))
    recovery_lease, queue_lease, heavy_lease = _acquire_all(paths)
    backup = paths.stage_root / "rollback" / (
        f"{time.time_ns()}-{secrets.token_hex(8)}"
    )
    manifest: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    journal: dict[str, Any] | None = None
    marker: dict[str, Any] | None = None
    campaign_leases_open = True
    try:
        _supersession_barrier(paths, operation="apply")
        ready, blockers = _quiescent(paths)
        if not ready:
            raise ForwardRecoveryError("under-lock quiescence failed: " + "; ".join(blockers))
        errors = validate_packet(packet, paths=paths, production_checks=production_checks,
                                 bridge_ids=bridge_ids)
        if errors:
            raise ForwardRecoveryError("under-lock CAS failed: " + "; ".join(errors))
        if _lexists(paths.journal):
            raise ForwardRecoveryError("a transaction journal appeared under lock")
        manifest = _backup_transaction(paths, packet, backup)
        if any(_lexists(Path(row["live"])) for row in manifest["absent_results"]):
            raise ForwardRecoveryError("an absence-bound result root appeared under lock")
        bundle = _stage_activation_bundle(paths, packet, backup, manifest)
        receipt = bundle["receipt"]
        journal = _initial_journal(paths, packet, backup, manifest, bundle)
        _validate_transaction_chain(paths, packet, journal)
        _fault("after:promotion:transaction-ready")

        journal = _promote_results(packet, paths, backup, journal)
        prepared = _read_json(Path(packet["prepared_runtime_packet"]["path"]))

        def promote(source: Path, target: Path, binding: dict[str, Any],
                    operation: str) -> None:
            nonlocal journal
            assert journal is not None
            journal = _live_step(
                paths, backup, journal, phase="promotion", operation=operation,
                action=lambda: _replace_file(source, target),
                verify=lambda: _artifact_content_matches(binding, target),
            )

        promote(Path(prepared["registry"]["staged"]["path"]), paths.registry,
                prepared["registry"]["staged"], "promote-registry")
        for row in prepared["pending_runtime_specs"]:
            promote(Path(row["staged"]["path"]), Path(row["target"]), row["staged"],
                    f"promote-runtime:{row['cell_id']}")
        promote(Path(packet["staged_state"]["path"]), paths.state,
                packet["staged_state"], "promote-state")
        promote(Path(packet["staged_campaign"]["path"]), paths.campaign,
                packet["staged_campaign"], "promote-campaign")
        promote(bundle["overlay_path"], paths.overlay,
                _artifact(bundle["overlay_path"]), "promote-overlay")
        promote(bundle["active_packet_path"], paths.canonical_reentry_packet,
                _artifact(bundle["active_packet_path"]), "promote-active-packet")
        promote(bundle["launch_agent_path"], paths.launch_agent,
                _artifact(bundle["launch_agent_path"]), "promote-launch-agent")
        # Marker is the final live-path mutation. Service reload/start happen only after it.
        promote(bundle["marker_path"], paths.active_marker,
                _artifact(bundle["marker_path"]), "promote-marker-last")
        marker = bundle["marker"]
        journal["live_marker"] = _artifact(paths.active_marker)
        journal["status"] = "files-promoted"
        journal = _journal_step(
            paths, backup, journal, phase="files-promoted",
            operation="files-promoted",
        )

        domain = f"gui/{os.getuid()}"
        label = "com.hawking.doctorv5ultra.autoresume"
        journal = _live_step(
            paths, backup, journal, phase="service",
            operation="launchctl-bootout",
            action=lambda: subprocess.run(
                ["launchctl", "bootout", f"{domain}/{label}"],
                capture_output=True, check=False,
            ),
            verify=lambda: True,
        )
        bootstrap: dict[str, Any] = {}
        def bootstrap_action() -> None:
            result = subprocess.run(
                ["launchctl", "bootstrap", domain, str(paths.launch_agent)],
                capture_output=True, text=True, check=False,
            )
            bootstrap["result"] = result
            if result.returncode != 0:
                raise ForwardRecoveryError("cannot bootstrap accelerated LaunchAgent: "
                                           + result.stderr.strip())
        journal = _live_step(
            paths, backup, journal, phase="service",
            operation="launchctl-bootstrap", action=bootstrap_action,
            verify=lambda: bootstrap.get("result") is not None,
        )
        journal["status"] = "start-intent"
        journal = _journal_step(
            paths, backup, journal, phase="start-intent",
            operation="start-detached",
        )
    except BaseException as exc:
        try:
            if journal is not None and manifest is not None and receipt is not None:
                try:
                    journal = _restore_transaction(
                        paths, packet, backup, manifest, receipt, journal,
                        forward_only=False,
                    )
                    _finish_rollback(paths, packet, backup, manifest, journal)
                except BaseException as rollback_exc:
                    raise ForwardRecoveryError(
                        f"activation failed ({exc}); safe rollback also failed: {rollback_exc}"
                    ) from rollback_exc
        finally:
            if campaign_leases_open:
                heavy_lease.close(); queue_lease.close(); campaign_leases_open = False
            recovery_lease.close()
        raise
    assert marker is not None and journal is not None and manifest is not None \
        and receipt is not None
    heavy_lease.close(); queue_lease.close(); campaign_leases_open = False
    try:
        pid = _start_detached(paths, marker)
    except BaseException as exc:
        try:
            queue_lease = _acquire(paths.queue_lock)
            heavy_lease = _acquire(paths.heavy_lock)
            campaign_leases_open = True
            owner = _verified_owner(paths)
            if owner is not None:
                journal["status"] = "active"
                journal["supervisor_pid"] = owner
                journal["supervisor_started_at"] = _now()
                journal = _journal_step(
                    paths, backup, journal, phase="active-reconciled",
                    operation="bind-ambiguous-start-owner",
                    details={"supervisor_pid": owner},
                )
                return {
                    "status": "active", "generation_id": packet["generation_id"],
                    "forward_packet_sha256": packet["packet_sha256"],
                    "supervisor_pid": owner, "reconciled_ambiguous_start": True,
                    "reset_rows": len(packet["reset_rows"]),
                    "archived_result_dirs": len(manifest["result_moves"]),
                    "checkpoint_bridges": sorted(bridge_ids),
                }
            journal = _restore_transaction(
                paths, packet, backup, manifest, receipt, journal,
                forward_only=True,
            )
            _finish_rollback(paths, packet, backup, manifest, journal)
        except BaseException as rollback_exc:
            raise ForwardRecoveryError(
                f"detached start failed ({exc}); rollback refused/failed: {rollback_exc}"
            ) from rollback_exc
        finally:
            if campaign_leases_open:
                heavy_lease.close(); queue_lease.close(); campaign_leases_open = False
            recovery_lease.close()
        raise
    try:
        journal["status"] = "active"
        journal["supervisor_pid"] = pid
        journal["supervisor_started_at"] = _now()
        journal = _journal_step(
            paths, backup, journal, phase="active", operation="supervisor-active",
            details={"supervisor_pid": pid},
        )
    finally:
        recovery_lease.close()
    return {
        "status": "active", "generation_id": packet["generation_id"],
        "forward_packet_sha256": packet["packet_sha256"],
        "supervisor_pid": pid, "reset_rows": len(packet["reset_rows"]),
        "archived_result_dirs": len(manifest["result_moves"]),
        "checkpoint_bridges": sorted(bridge_ids),
    }


def recover(*, paths: Paths | None = None) -> dict[str, Any]:
    """Reconcile a crash-cut transaction or resume its verified active owner."""
    paths = paths or production_paths()
    recovery_lease = _acquire(paths.transaction_lock)
    queue_lease: IO[str] | None = None
    heavy_lease: IO[str] | None = None
    try:
        if not _lexists(paths.journal):
            return {"status": "no-transaction", "source_deletion_permitted": False}
        packet = _read_json(paths.packet)
        if packet.get("schema") != SCHEMA or packet.get("packet_sha256") \
                != _hash_value(_without(packet, "packet_sha256")):
            raise ForwardRecoveryError("recovery packet schema/hash is invalid")
        journal = _read_json(paths.journal)
        backup, manifest, receipt, _ = _validate_transaction_chain(
            paths, packet, journal
        )
        owner = _verified_owner(paths)
        marker_committed = _artifact_content_matches(
            journal["staged_marker"], paths.active_marker
        )
        if owner is not None:
            if not marker_committed or journal["status"] == "rolled-back":
                raise ForwardRecoveryError(
                    "a live accelerated owner exists without the committed marker chain"
                )
            if journal["status"] == "active" \
                    and journal.get("supervisor_pid") == owner \
                    and journal.get("live_marker") is not None:
                return {"status": "active", "supervisor_pid": owner,
                        "reconciled": False, "healthy_noop": True,
                        "source_deletion_permitted": False}
            journal["status"] = "active"
            journal["live_marker"] = _artifact(paths.active_marker)
            journal["supervisor_pid"] = owner
            journal["supervisor_started_at"] = journal.get("supervisor_started_at") or _now()
            journal = _journal_step(
                paths, backup, journal, phase="active-reconciled",
                operation="bind-live-owner", details={"supervisor_pid": owner},
            )
            return {"status": "active", "supervisor_pid": owner,
                    "reconciled": True, "source_deletion_permitted": False}
        if journal["status"] == "rolled-back":
            ref = journal.get("rollback_receipt")
            if not _artifact_matches(ref, backup / "rollback_receipt.json"):
                raise ForwardRecoveryError("rolled-back transaction receipt differs")
            _rollback_receipt(paths, packet, backup, manifest, journal)
            return {"status": "already-rolled-back", "reconciled": True,
                    "source_deletion_permitted": False}
        if journal["status"] == "active":
            control = _read_json(paths.control)
            state = _read_json(paths.state)
            if control.get("mode") != "run" or state.get("status") == "complete":
                return {"status": "active-generation-owner-free",
                        "control_mode": control.get("mode"), "reconciled": True,
                        "source_deletion_permitted": False}
            result = subprocess.run(
                [sys.executable, str(paths.accelerated_autoresume)],
                cwd=paths.root, capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                raise ForwardRecoveryError("active-generation autoresume failed: "
                                           + (result.stderr or result.stdout).strip())
            owner = _verified_owner(paths)
            if owner is None:
                raise ForwardRecoveryError("autoresume returned without a verified owner")
            journal["supervisor_pid"] = owner
            journal["supervisor_started_at"] = _now()
            journal = _journal_step(
                paths, backup, journal, phase="active-resumed",
                operation="resume-active-owner", details={"supervisor_pid": owner},
            )
            return {"status": "active", "supervisor_pid": owner,
                    "resumed": True, "source_deletion_permitted": False}

        queue_lease = _acquire(paths.queue_lock)
        heavy_lease = _acquire(paths.heavy_lock)
        owner = _verified_owner(paths)
        if owner is not None:
            raise ForwardRecoveryError("owner appeared while recovery leases were acquired")
        journal = _read_json(paths.journal)
        backup, manifest, receipt, _ = _validate_transaction_chain(
            paths, packet, journal
        )
        marker_committed = _artifact_content_matches(
            journal["staged_marker"], paths.active_marker
        )
        rollback_direction = journal["status"] == "rolling-back"
        if rollback_direction or not marker_committed:
            ready, blockers = _quiescent(paths)
            if not ready:
                raise ForwardRecoveryError(
                    "partial pre-marker transaction is not quiescent: "
                    + "; ".join(blockers)
                )
            journal = _restore_transaction(
                paths, packet, backup, manifest, receipt, journal,
                forward_only=False,
            )
            journal = _finish_rollback(
                paths, packet, backup, manifest, journal
            )
            return {"status": "rolled-back", "reconciled": True,
                    "restored_result_dirs": len(manifest["result_moves"]),
                    "source_deletion_permitted": False}

        _preflight_result_surface(
            packet, manifest["result_moves"], manifest["absent_results"],
            forward_only=True,
        )
        _preflight_live_files(
            paths, manifest, receipt, journal, allow_original=False
        )
        journal["live_marker"] = _artifact(paths.active_marker)
        journal["status"] = "files-promoted"
        journal = _journal_step(
            paths, backup, journal, phase="service-recovery-intent",
            operation="reload-committed-launch-agent",
        )
        _fault("before:recover:reload-committed-launch-agent")
        _reload_launch_agent(paths)
        journal = _journal_step(
            paths, backup, journal, phase="service-recovery-done",
            operation="reload-committed-launch-agent",
        )
        _fault("after:recover:reload-committed-launch-agent")
        journal["status"] = "start-intent"
        journal = _journal_step(
            paths, backup, journal, phase="start-recovery-intent",
            operation="start-detached",
        )
        heavy_lease.close(); heavy_lease = None
        queue_lease.close(); queue_lease = None
        marker = _read_json(paths.active_marker)
        owner = _start_detached(paths, marker)
        journal["status"] = "active"
        journal["supervisor_pid"] = owner
        journal["supervisor_started_at"] = _now()
        journal = _journal_step(
            paths, backup, journal, phase="active-recovered",
            operation="supervisor-active", details={"supervisor_pid": owner},
        )
        return {"status": "active", "supervisor_pid": owner,
                "reconciled": True, "source_deletion_permitted": False}
    finally:
        if heavy_lease is not None:
            heavy_lease.close()
        if queue_lease is not None:
            queue_lease.close()
        recovery_lease.close()


def rollback(*, paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or production_paths()
    recovery_lease, queue_lease, heavy_lease = _acquire_all(paths)
    try:
        owner = _verified_owner(paths)
        if owner is not None:
            raise ForwardRecoveryError(
                f"rollback refused because accelerated owner is live: {owner}"
            )
        ready, blockers = _quiescent(paths)
        if not ready:
            raise ForwardRecoveryError("rollback requires an owner-free drain: "
                                       + "; ".join(blockers))
        packet = _read_json(paths.packet)
        if packet.get("schema") != SCHEMA or packet.get("packet_sha256") \
                != _hash_value(_without(packet, "packet_sha256")):
            raise ForwardRecoveryError("staged forward packet is invalid")
        journal = _read_json(paths.journal)
        backup, manifest, receipt, _ = _validate_transaction_chain(
            paths, packet, journal
        )
        if journal["status"] == "rolled-back":
            ref = journal.get("rollback_receipt")
            expected = backup / "rollback_receipt.json"
            if not _artifact_matches(ref, expected):
                raise ForwardRecoveryError("rolled-back receipt binding differs")
            recorded = _read_json(expected)
            if recorded.get("schema") != ROLLBACK_RECEIPT_SCHEMA \
                    or recorded.get("receipt_sha256") != _hash_value(
                        _without(recorded, "receipt_sha256")
                    ):
                raise ForwardRecoveryError("rolled-back receipt is invalid")
            _rollback_receipt(paths, packet, backup, manifest, journal)
            return {"status": "already-rolled-back",
                    "restored_result_dirs": len(manifest["result_moves"]),
                    "source_deletion_permitted": False}
        forward_only = journal["status"] in {"files-promoted", "start-intent", "active"}
        journal = _restore_transaction(
            paths, packet, backup, manifest, receipt, journal,
            forward_only=forward_only,
        )
        _finish_rollback(paths, packet, backup, manifest, journal)
    finally:
        heavy_lease.close(); queue_lease.close(); recovery_lease.close()
    return {"status": "rolled-back",
            "restored_result_dirs": len(manifest["result_moves"]),
            "source_deletion_permitted": False}


def status(paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or production_paths()
    ready, blockers = _quiescent(paths)
    supersession_blocker: str | None = None
    try:
        _supersession_barrier(paths, operation="apply")
    except ForwardRecoveryError as exc:
        supersession_blocker = str(exc)
    errors: list[str] | None = None
    if paths.packet.exists() and not paths.journal.exists() \
            and supersession_blocker is None:
        try:
            errors = validate_packet(_read_json(paths.packet), paths=paths)
        except ForwardRecoveryError as exc:
            errors = [str(exc)]
    journal = None
    recovery_scan: dict[str, Any] | None = None
    if paths.journal.exists():
        try:
            journal = _read_json(paths.journal)
            packet = _read_json(paths.packet)
            _validate_transaction_chain(paths, packet, journal)
            owner = _verified_owner(paths)
            marker_committed = _artifact_content_matches(
                journal["staged_marker"], paths.active_marker
            )
            if owner is not None and journal.get("status") == "active" \
                    and journal.get("supervisor_pid") == owner \
                    and journal.get("live_marker") is not None:
                action = "healthy-active-noop"
            elif owner is not None:
                action = "finalize-active-owner"
            elif journal.get("status") == "rolled-back":
                action = "none-rolled-back"
            elif journal.get("status") == "active":
                action = "resume-active-if-control-run"
            elif marker_committed:
                action = "resume-committed-marker"
            else:
                action = "rollback-partial-pre-marker"
            recovery_scan = {
                "chain_valid": True, "verified_owner_pid": owner,
                "marker_committed": marker_committed,
                "recommended_action": action,
            }
        except ForwardRecoveryError as exc:
            journal = {"status": "invalid"}
            recovery_scan = {"chain_valid": False, "errors": [str(exc)],
                             "recommended_action": "fail-closed"}
    return {
        "schema": "hawking.doctor_v5_forward_recovery_status.v1",
        "generated_at": _now(), "quiescent": ready, "quiescent_blockers": blockers,
        "packet_present": paths.packet.exists(), "packet_errors": errors,
        "supersession_blocker": supersession_blocker,
        "journal_status": journal.get("status") if journal else None,
        "recovery_scan": recovery_scan,
        "recover_required": recovery_scan is not None
        and recovery_scan.get("recommended_action") not in {
            "none-rolled-back", "healthy-active-noop"
        },
        "activation_permitted_now": ready and errors == [] and journal is None
        and supersession_blocker is None,
        "source_deletion_permitted": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status"); sub.add_parser("stage"); sub.add_parser("audit")
    sub.add_parser("adversarial-audit"); sub.add_parser("rollback")
    sub.add_parser("recover")
    supersede_parser = sub.add_parser("supersede")
    supersede_parser.add_argument("--reason", required=True)
    activate = sub.add_parser("apply")
    activate.add_argument("--packet-sha256", required=True)
    activate.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    paths = production_paths()
    if args.command == "status":
        result = status(paths)
    elif args.command == "stage":
        result = stage(paths)
    elif args.command == "audit":
        errors = validate_packet(_read_json(paths.packet), paths=paths)
        result = {"ok": not errors, "errors": errors}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not errors else 2
    elif args.command == "adversarial-audit":
        result = adversarial_audit(paths=paths)
    elif args.command == "apply":
        result = apply(packet_sha256=args.packet_sha256,
                       plan_sha256=args.plan_sha256, paths=paths)
    elif args.command == "recover":
        result = recover(paths=paths)
    elif args.command == "supersede":
        result = supersede(reason=args.reason, paths=paths)
    else:
        result = rollback(paths=paths)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ForwardRecoveryError, reentry.ReentryError,
            gc_transition.GCRuntimeTransitionError) as exc:
        print(f"doctor_v5_forward_recovery: {exc}", file=sys.stderr)
        raise SystemExit(2)
