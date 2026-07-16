#!/usr/bin/env python3
"""Measure Hawking LOC without confusing relocation with elimination."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any

import hawking_packs


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "loc_floor.json"
CONTRACT_SCHEMA = "hawking.loc_floor.v2"
LOCK_SCHEMA = "hawking.owned_source_lock.v1"
REPORT_SCHEMA = "hawking.loc_floor_report.v2"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RUST_TEST = re.compile(rb"^[ \t]*#\[(?:tokio::)?test[^\]]*\]", re.MULTILINE)
RUST_IGNORE = re.compile(rb"^[ \t]*#\[ignore[^\]]*\]", re.MULTILINE)


class LocFloorError(RuntimeError):
    """The accounting source or contract is structurally invalid."""


@dataclass(frozen=True)
class Blob:
    mode: str
    oid: str
    path: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _physical_lines(raw: bytes) -> int:
    if not raw:
        return 0
    return raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LocFloorError(f"{path}: top-level value must be an object")
    return value


def _safe_relative_path(raw: object) -> str:
    if not isinstance(raw, str) or not raw:
        raise LocFloorError("inventory path must be a non-empty string")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or str(path) in {".", ""}:
        raise LocFloorError(f"unsafe inventory path: {raw!r}")
    if path.as_posix() != raw:
        raise LocFloorError(f"inventory path is not canonical POSIX form: {raw!r}")
    return raw


class GitObjects:
    """Read committed trees and blobs without consulting the working tree."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._trees: dict[str, dict[str, Blob]] = {}
        self._blobs: dict[str, bytes] = {}

    def _run(self, *args: str) -> bytes:
        result = subprocess.run(
            ["git", *args],
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            detail = result.stderr.decode("utf-8", "replace").strip()
            raise LocFloorError(f"git {' '.join(args)} failed: {detail}")
        return result.stdout

    def resolve_commit(self, ref: str) -> str:
        resolved = self._run("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
        if not COMMIT_RE.fullmatch(resolved):
            raise LocFloorError(f"invalid commit resolved for {ref!r}: {resolved!r}")
        return resolved

    def tree(self, ref: str) -> tuple[str, dict[str, Blob]]:
        commit = self.resolve_commit(ref)
        cached = self._trees.get(commit)
        if cached is not None:
            return commit, cached
        raw = self._run("ls-tree", "-rz", "--full-tree", commit)
        rows: dict[str, Blob] = {}
        for record in raw.split(b"\0"):
            if not record:
                continue
            try:
                metadata, raw_path = record.split(b"\t", 1)
                mode, object_type, raw_oid = metadata.split(b" ", 2)
            except ValueError as exc:
                raise LocFloorError("malformed git ls-tree record") from exc
            if object_type != b"blob" or mode in {b"120000", b"160000"}:
                continue
            path = raw_path.decode("utf-8", "surrogateescape")
            if path in rows:
                raise LocFloorError(f"duplicate path in Git tree: {path}")
            rows[path] = Blob(mode.decode(), raw_oid.decode(), path)
        self._trees[commit] = rows
        return commit, rows

    def read_blobs(self, oids: list[str]) -> dict[str, bytes]:
        missing = sorted(set(oids) - self._blobs.keys())
        if missing:
            process = subprocess.Popen(
                ["git", "cat-file", "--batch"],
                cwd=self.root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            request = b"".join(oid.encode("ascii") + b"\n" for oid in missing)
            stdout, stderr = process.communicate(request)
            if process.returncode:
                detail = stderr.decode("utf-8", "replace").strip()
                raise LocFloorError(f"git cat-file --batch failed: {detail}")
            cursor = 0
            for requested in missing:
                header_end = stdout.find(b"\n", cursor)
                if header_end < 0:
                    raise LocFloorError("truncated git cat-file header")
                header = stdout[cursor:header_end]
                cursor = header_end + 1
                parts = header.split()
                if len(parts) != 3 or parts[1] != b"blob":
                    raise LocFloorError(
                        f"git cat-file did not return a blob for {requested}: {header!r}"
                    )
                size = int(parts[2])
                end = cursor + size
                if end >= len(stdout) or stdout[end:end + 1] != b"\n":
                    raise LocFloorError(f"truncated git blob payload for {requested}")
                self._blobs[requested] = stdout[cursor:end]
                cursor = end + 1
            if cursor != len(stdout):
                raise LocFloorError("unexpected trailing bytes from git cat-file")
        return {oid: self._blobs[oid] for oid in oids}


def _suffixes(contract: dict[str, Any], key: str) -> set[str]:
    policy = contract.get("policy")
    if not isinstance(policy, dict) or not isinstance(policy.get(key), list):
        raise LocFloorError(f"loc_floor.json policy.{key} must be a list")
    values = {str(value).lower() for value in policy[key]}
    if not values or any(not value.startswith(".") for value in values):
        raise LocFloorError(f"loc_floor.json policy.{key} is invalid")
    return values


def _measure_tree(
    git: GitObjects,
    tree: dict[str, Blob],
    suffixes: set[str],
) -> dict[str, Any]:
    selected = [
        blob
        for path, blob in tree.items()
        if PurePosixPath(path).suffix.lower() in suffixes
    ]
    raw_by_oid = git.read_blobs([blob.oid for blob in selected])
    by_suffix: dict[str, int] = {}
    total_bytes = 0
    total_lines = 0
    rust_tests = 0
    rust_ignores = 0
    for blob in selected:
        raw = raw_by_oid[blob.oid]
        suffix = PurePosixPath(blob.path).suffix.lower()
        lines = _physical_lines(raw)
        total_bytes += len(raw)
        total_lines += lines
        by_suffix[suffix] = by_suffix.get(suffix, 0) + lines
        if suffix == ".rs":
            rust_tests += len(RUST_TEST.findall(raw))
            rust_ignores += len(RUST_IGNORE.findall(raw))
    return {
        "loc": total_lines,
        "source_file_count": len(selected),
        "source_bytes": total_bytes,
        "by_suffix": dict(sorted(by_suffix.items())),
        "tracked_regular_blob_count": len(tree),
        "rust_test_attributes": rust_tests,
        "rust_ignore_attributes": rust_ignores,
    }


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LocFloorError(f"{label} must be an integer")
    return value


def _verify_source_lock(
    git: GitObjects,
    contract: dict[str, Any],
    lock: dict[str, Any],
    active_tree: dict[str, Blob],
) -> dict[str, Any]:
    errors: list[str] = []
    if lock.get("schema") != LOCK_SCHEMA:
        errors.append(f"source lock schema must be {LOCK_SCHEMA}")
    policy = contract.get("policy")
    policy_id = policy.get("id") if isinstance(policy, dict) else None
    if lock.get("policy_id") != policy_id:
        errors.append("source lock policy_id differs from loc_floor.json")
    source_suffixes = _suffixes(contract, "source_suffixes")
    units = lock.get("units")
    if not isinstance(units, list):
        raise LocFloorError("source lock units must be a list")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    unit_results: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    relocated_owned_loc = 0

    for index, raw_unit in enumerate(units):
        if not isinstance(raw_unit, dict):
            errors.append(f"unit {index} must be an object")
            continue
        unit_id = raw_unit.get("id")
        if not isinstance(unit_id, str) or not unit_id:
            errors.append(f"unit {index} has invalid id")
            unit_id = f"<unit-{index}>"
        elif unit_id in seen_ids:
            errors.append(f"duplicate unit id: {unit_id}")
        seen_ids.add(unit_id)
        unit_errors: list[str] = []
        if raw_unit.get("ownership") != "project_owned":
            unit_errors.append("ownership must be project_owned")
        if raw_unit.get("accounting") != "relocated":
            unit_errors.append("accounting must be relocated")
        required_by = raw_unit.get("required_by")
        if not isinstance(required_by, list) or not required_by:
            unit_errors.append("required_by must be a non-empty list")
        else:
            for raw_path in required_by:
                try:
                    required_path = _safe_relative_path(raw_path)
                except LocFloorError as exc:
                    unit_errors.append(str(exc))
                    continue
                if required_path not in active_tree:
                    unit_errors.append(f"required_by path is not active: {required_path}")

        source = raw_unit.get("source")
        if not isinstance(source, dict) or source.get("kind") != "self_git":
            unit_errors.append("source.kind must be self_git")
            source_commit = ""
        else:
            source_commit = str(source.get("commit_oid", ""))
        if not COMMIT_RE.fullmatch(source_commit):
            unit_errors.append("source.commit_oid must be a full Git commit")
            source_tree: dict[str, Blob] = {}
        else:
            try:
                resolved, source_tree = git.tree(source_commit)
                if resolved != source_commit:
                    unit_errors.append("source.commit_oid did not resolve exactly")
            except LocFloorError as exc:
                unit_errors.append(str(exc))
                source_tree = {}

        files = raw_unit.get("files")
        if not isinstance(files, list):
            unit_errors.append("files must be a list")
            files = []
        declared_paths: list[str] = []
        for row in files:
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                declared_paths.append(str(row["path"]))
        if declared_paths != sorted(declared_paths):
            unit_errors.append("file inventory must be sorted by path")

        available_oids = [
            source_tree[path].oid
            for path in declared_paths
            if path in source_tree
        ]
        try:
            raw_by_oid = git.read_blobs(available_oids)
        except LocFloorError as exc:
            unit_errors.append(str(exc))
            raw_by_oid = {}

        actual_rows: list[dict[str, object]] = []
        for row_index, raw_row in enumerate(files):
            if not isinstance(raw_row, dict):
                unit_errors.append(f"file row {row_index} must be an object")
                continue
            try:
                path = _safe_relative_path(raw_row.get("path"))
            except LocFloorError as exc:
                unit_errors.append(str(exc))
                continue
            if path in seen_paths:
                unit_errors.append(f"relocated inventory path overlaps another unit: {path}")
            seen_paths.add(path)
            if path in active_tree:
                unit_errors.append(f"relocated inventory overlaps active Git tree: {path}")
            suffix = PurePosixPath(path).suffix.lower()
            if suffix not in source_suffixes:
                unit_errors.append(f"relocated source has uncounted suffix: {path}")
            source_blob = source_tree.get(path)
            if source_blob is None:
                unit_errors.append(f"source path is absent at {source_commit}: {path}")
                continue
            raw = raw_by_oid.get(source_blob.oid)
            if raw is None:
                unit_errors.append(f"source blob could not be read: {path}")
                continue
            actual = {
                "path": path,
                "bytes": len(raw),
                "sha256": _sha256(raw),
                "lines": _physical_lines(raw),
            }
            actual_rows.append(actual)
            for key in ("bytes", "sha256", "lines"):
                if raw_row.get(key) != actual[key]:
                    unit_errors.append(
                        f"{path}: declared {key} {raw_row.get(key)!r} "
                        f"!= actual {actual[key]!r}"
                    )

        actual_bytes = sum(int(row["bytes"]) for row in actual_rows)
        actual_lines = sum(int(row["lines"]) for row in actual_rows)
        inventory_sha256 = _sha256(_canonical_json(actual_rows))
        expected = raw_unit.get("expected")
        if not isinstance(expected, dict):
            unit_errors.append("expected must be an object")
            expected = {}
        comparisons = {
            "file_count": len(actual_rows),
            "bytes": actual_bytes,
            "primary_lines": actual_lines,
            "inventory_sha256": inventory_sha256,
        }
        for key, actual in comparisons.items():
            if expected.get(key) != actual:
                unit_errors.append(
                    f"expected.{key} {expected.get(key)!r} != actual {actual!r}"
                )
        total_files += len(actual_rows)
        total_bytes += actual_bytes
        relocated_owned_loc += actual_lines
        errors.extend(f"{unit_id}: {error}" for error in unit_errors)
        unit_results.append(
            {
                "id": unit_id,
                "valid": not unit_errors,
                "file_count": len(actual_rows),
                "bytes": actual_bytes,
                "primary_lines": actual_lines,
                "inventory_sha256": inventory_sha256,
                "errors": unit_errors,
            }
        )

    actual_totals = {
        "unit_count": len(unit_results),
        "file_count": total_files,
        "bytes": total_bytes,
        "relocated_owned_loc": relocated_owned_loc,
    }
    expected_totals = lock.get("expected_totals")
    if not isinstance(expected_totals, dict):
        errors.append("source lock expected_totals must be an object")
    else:
        for key, actual in actual_totals.items():
            if expected_totals.get(key) != actual:
                errors.append(
                    f"expected_totals.{key} {expected_totals.get(key)!r} "
                    f"!= actual {actual!r}"
                )
    return {
        "valid": not errors,
        "errors": errors,
        **actual_totals,
        "units": unit_results,
    }


def _pack_materialized_paths(
    manifest: dict[str, object],
    row: dict[str, object],
) -> tuple[set[str], set[str]]:
    mappings = row.get("materialize")
    metadata = row.get("unmaterialized_metadata")
    if not isinstance(mappings, list) or not isinstance(metadata, list):
        raise LocFloorError(
            f"{row.get('id')}: materialize and unmaterialized_metadata must be lists"
        )
    metadata_paths = {_safe_relative_path(path) for path in metadata}
    mapped: set[str] = set()
    unmapped: set[str] = set()
    for file_row in hawking_packs.validated_manifest(manifest):
        pack_path = PurePosixPath(str(file_row["path"]))
        destinations: list[str] = []
        for mapping in mappings:
            if not isinstance(mapping, dict):
                raise LocFloorError(f"{row.get('id')}: invalid materialization mapping")
            source = PurePosixPath(_safe_relative_path(mapping.get("source")))
            destination = PurePosixPath(
                _safe_relative_path(mapping.get("destination"))
            )
            if pack_path == source or source in pack_path.parents:
                destinations.append(
                    (destination / pack_path.relative_to(source)).as_posix()
                )
        if len(destinations) > 1:
            raise LocFloorError(f"{row.get('id')}: overlapping mappings for {pack_path}")
        if destinations:
            mapped.add(destinations[0])
        else:
            unmapped.add(pack_path.as_posix())
    if unmapped != metadata_paths:
        raise LocFloorError(
            f"{row.get('id')}: unmaterialized payload differs from declared metadata"
        )
    return mapped, metadata_paths


def _verify_pack_lock(
    contract: dict[str, Any],
    active_tree: dict[str, Blob],
    historical_paths: set[str],
) -> dict[str, Any]:
    lock_path = ROOT / str(contract.get("pack_lock", ""))
    try:
        lock = hawking_packs.load_lock(lock_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "path": lock_path.relative_to(ROOT).as_posix(),
            "relocated_owned_loc": 0,
            "rust_test_attributes": 0,
            "rust_ignore_attributes": 0,
            "packs": [],
        }
    suffixes = _suffixes(contract, "source_suffixes")
    errors: list[str] = []
    total_loc = 0
    total_tests = 0
    total_ignores = 0
    logical_paths: set[str] = set()
    results: list[dict[str, Any]] = []
    for row in hawking_packs.select_rows(lock, []):
        pack_id = str(row.get("id", ""))
        pack_errors: list[str] = []
        destination = hawking_packs.pack_destination(lock, row)
        manifest_path = destination / hawking_packs.MANIFEST_NAME
        manifest: dict[str, object] = {}
        if not manifest_path.is_file():
            pack_errors.append("hydration manifest missing")
        else:
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("manifest must be an object")
                manifest = value
                hawking_packs.validated_manifest(manifest)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                pack_errors.append(str(exc))
        verification: dict[str, object] = {"valid": False}
        materializations: dict[str, object] = {"valid": False}
        mapped_paths: set[str] = set()
        if manifest:
            try:
                verification = hawking_packs.verify_tree(
                    destination, manifest, suffixes
                )
                if not verification["valid"]:
                    pack_errors.extend(str(value) for value in verification["errors"])
                materializations = hawking_packs.verify_materializations(
                    destination, row, suffixes
                )
                if not materializations["valid"]:
                    pack_errors.append("legacy materialization differs")
                if manifest.get("pack_id") != pack_id:
                    pack_errors.append("manifest pack id differs")
                if str(manifest.get("version")) != str(row.get("version")):
                    pack_errors.append("manifest version differs")
                manifest_sha = hawking_packs.sha256_bytes(
                    hawking_packs.canonical_json(manifest)
                )
                if manifest_sha != row.get("manifest_sha256"):
                    pack_errors.append("manifest SHA-256 differs")
                if manifest.get("tree_sha256") != row.get("tree_sha256"):
                    pack_errors.append("tree SHA-256 differs from lock")
                if manifest.get("primary_lines") != row.get("primary_lines"):
                    pack_errors.append("primary LOC differs from lock")
                mapped_paths, _ = _pack_materialized_paths(manifest, row)
            except (OSError, ValueError, KeyError, LocFloorError) as exc:
                pack_errors.append(str(exc))
        active_overlap = mapped_paths & set(active_tree)
        if active_overlap and active_overlap != mapped_paths:
            pack_errors.append(
                f"partial active/materialized overlap "
                f"({len(active_overlap)}/{len(mapped_paths)})"
            )
        accounted = bool(mapped_paths) and not active_overlap
        overlap = mapped_paths & (historical_paths | logical_paths)
        if overlap:
            pack_errors.append(f"owned source path overlaps another unit: {min(overlap)}")
        logical_paths.update(mapped_paths)
        pack_loc = 0
        control_loc = int(row["control_primary_lines"]) if accounted else 0
        pack_tests = 0
        pack_ignores = 0
        if accounted and manifest:
            pack_loc = int(manifest["primary_lines"])
            files = manifest.get("files")
            if isinstance(files, list):
                for file_row in files:
                    if not isinstance(file_row, dict):
                        continue
                    relative = str(file_row.get("path", ""))
                    if PurePosixPath(relative).suffix.lower() != ".rs":
                        continue
                    raw = (destination / relative).read_bytes()
                    pack_tests += len(RUST_TEST.findall(raw))
                    pack_ignores += len(RUST_IGNORE.findall(raw))
        if row.get("ownership") != "project_owned":
            pack_errors.append("pack ownership must be project_owned")
        else:
            total_loc += pack_loc + control_loc
            total_tests += pack_tests
            total_ignores += pack_ignores
        cached = hawking_packs.cached_archive(lock, row)
        archive_cached = cached.is_file()
        archive_valid = (
            archive_cached
            and hawking_packs.sha256_file(cached) == row.get("archive_sha256")
        )
        errors.extend(f"{pack_id}: {error}" for error in pack_errors)
        results.append(
            {
                "id": pack_id,
                "valid": not pack_errors,
                "accounted_as_relocated": accounted,
                "active_overlap_count": len(active_overlap),
                "mapped_file_count": len(mapped_paths),
                "primary_lines": pack_loc,
                "control_primary_lines": control_loc,
                "declared_control_primary_lines": row["control_primary_lines"],
                "owned_primary_lines": pack_loc + control_loc,
                "source_commit": row["source_commit"],
                "rust_test_attributes": pack_tests,
                "rust_ignore_attributes": pack_ignores,
                "archive_cached": archive_cached,
                "archive_sha256_valid": archive_valid,
                "tree": verification,
                "materializations": materializations,
                "errors": pack_errors,
            }
        )
    return {
        "valid": not errors,
        "errors": errors,
        "path": lock_path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(lock_path.read_bytes()),
        "relocated_owned_loc": total_loc,
        "rust_test_attributes": total_tests,
        "rust_ignore_attributes": total_ignores,
        "packs": results,
    }


def _evaluate_rungs(
    contract: dict[str, Any],
    metrics: dict[str, int],
) -> list[dict[str, Any]]:
    raw_rungs = contract.get("rungs")
    if not isinstance(raw_rungs, list):
        raise LocFloorError("loc_floor.json rungs must be a list")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_rung in raw_rungs:
        if not isinstance(raw_rung, dict):
            raise LocFloorError("every rung must be an object")
        rung_id = str(raw_rung.get("id", ""))
        if not rung_id or rung_id in seen:
            raise LocFloorError(f"invalid or duplicate rung id: {rung_id!r}")
        seen.add(rung_id)
        budgets = raw_rung.get("budgets")
        if not isinstance(budgets, dict):
            raise LocFloorError(f"{rung_id}: budgets must be an object")
        checks = {
            "active_repo_loc_max": (
                metrics["active_repo_loc"],
                _integer(budgets.get("active_repo_loc_max"), f"{rung_id}.active max"),
                "max",
            ),
            "hydrated_owned_loc_max": (
                metrics["hydrated_owned_loc"],
                _integer(
                    budgets.get("hydrated_owned_loc_max"),
                    f"{rung_id}.hydrated max",
                ),
                "max",
            ),
            "relocated_owned_loc_max": (
                metrics["relocated_owned_loc"],
                _integer(
                    budgets.get("relocated_owned_loc_max"),
                    f"{rung_id}.relocated max",
                ),
                "max",
            ),
            "eliminated_loc_min": (
                metrics["eliminated_loc"],
                _integer(budgets.get("eliminated_loc_min"), f"{rung_id}.eliminated min"),
                "min",
            ),
        }
        rows: dict[str, dict[str, Any]] = {}
        for key, (current, target, direction) in checks.items():
            met = current <= target if direction == "max" else current >= target
            remaining = (
                max(0, current - target)
                if direction == "max"
                else max(0, target - current)
            )
            rows[key] = {
                "current": current,
                "target": target,
                "met": met,
                "remaining": remaining,
            }
        results.append(
            {
                "id": rung_id,
                "prior": raw_rung.get("prior"),
                "met": all(row["met"] for row in rows.values()),
                "checks": rows,
            }
        )
    return results


def build_report(ref: str = "HEAD") -> dict[str, Any]:
    contract = _load_json(CONTRACT_PATH)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise LocFloorError(f"loc_floor.json schema must be {CONTRACT_SCHEMA}")
    lock_path = ROOT / str(contract.get("source_lock", ""))
    lock = _load_json(lock_path)
    git = GitObjects(ROOT)
    ref_commit, active_tree = git.tree(ref)
    source_suffixes = _suffixes(contract, "source_suffixes")
    legacy_suffixes = _suffixes(contract, "legacy_suffixes")
    active = _measure_tree(git, active_tree, source_suffixes)
    active_legacy = _measure_tree(git, active_tree, legacy_suffixes)

    errors: list[str] = []
    policy = contract.get("policy")
    if not isinstance(policy, dict):
        raise LocFloorError("loc_floor.json policy must be an object")
    heritage = contract.get("heritage_baseline")
    if not isinstance(heritage, dict):
        raise LocFloorError("loc_floor.json heritage_baseline must be an object")
    heritage_ref = str(heritage.get("commit", ""))
    heritage_commit, heritage_tree = git.tree(heritage_ref)
    heritage_measure = _measure_tree(git, heritage_tree, source_suffixes)
    heritage_legacy = _measure_tree(git, heritage_tree, legacy_suffixes)
    if heritage_measure["loc"] != heritage.get("active_repo_loc"):
        errors.append("heritage baseline v2 LOC differs from committed Git blobs")
    if heritage_legacy["loc"] != heritage.get("legacy_primary_lines"):
        errors.append("heritage baseline legacy LOC differs from committed Git blobs")

    lock_verification = _verify_source_lock(git, contract, lock, active_tree)
    errors.extend(lock_verification["errors"])
    historical_paths = {
        str(row["path"])
        for unit in lock.get("units", [])
        if isinstance(unit, dict)
        for row in unit.get("files", [])
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    pack_verification = _verify_pack_lock(contract, active_tree, historical_paths)
    errors.extend(pack_verification["errors"])
    relocated_owned_loc = (
        int(lock_verification["relocated_owned_loc"])
        + int(pack_verification["relocated_owned_loc"])
    )
    active_repo_loc = int(active["loc"])
    hydrated_owned_loc = active_repo_loc + relocated_owned_loc
    baseline_loc = int(heritage_measure["loc"])
    eliminated_loc = baseline_loc - hydrated_owned_loc
    conservation = {
        "hydrated_identity": (
            hydrated_owned_loc == active_repo_loc + relocated_owned_loc
        ),
        "baseline_identity": (
            baseline_loc
            == active_repo_loc + relocated_owned_loc + eliminated_loc
        ),
    }
    conservation["valid"] = all(conservation.values())
    if not conservation["valid"]:
        errors.append("LOC conservation identity failed")
    metrics = {
        "active_repo_loc": active_repo_loc,
        "relocated_owned_loc": relocated_owned_loc,
        "hydrated_owned_loc": hydrated_owned_loc,
        "eliminated_loc": eliminated_loc,
    }
    logical_tests = (
        int(active["rust_test_attributes"])
        + int(pack_verification["rust_test_attributes"])
    )
    logical_ignores = (
        int(active["rust_ignore_attributes"])
        + int(pack_verification["rust_ignore_attributes"])
    )
    expected_tests = contract.get("test_surface")
    if not isinstance(expected_tests, dict):
        raise LocFloorError("loc_floor.json test_surface must be an object")
    test_surface = {
        "rust_test_attributes": {
            "current": logical_tests,
            "expected": _integer(
                expected_tests.get("rust_test_attributes"), "expected Rust tests"
            ),
        },
        "rust_ignore_attributes": {
            "current": logical_ignores,
            "expected": _integer(
                expected_tests.get("rust_ignore_attributes"), "expected Rust ignores"
            ),
        },
    }
    for row in test_surface.values():
        row["met"] = row["current"] == row["expected"]
    test_surface_preserved = all(bool(row["met"]) for row in test_surface.values())
    if not test_surface_preserved:
        errors.append("logical Rust test surface differs")

    foundation = contract.get("foundation_measurement")
    foundation_valid: bool | None = None
    if isinstance(foundation, dict) and ref_commit == foundation.get("commit"):
        foundation_actual = {
            "legacy_primary_lines": int(active_legacy["loc"]),
            **metrics,
        }
        foundation_valid = all(
            foundation.get(key) == value
            for key, value in foundation_actual.items()
        )
        if not foundation_valid:
            errors.append("foundation measurement differs from the pinned foundation commit")

    rungs = _evaluate_rungs(contract, metrics)
    lock_raw = lock_path.read_bytes()
    report = {
        "schema": REPORT_SCHEMA,
        "valid": (
            not errors
            and bool(lock_verification["valid"])
            and bool(pack_verification["valid"])
            and test_surface_preserved
        ),
        "errors": errors,
        "policy_id": policy.get("id"),
        "ref": ref,
        "ref_commit": ref_commit,
        "heritage_baseline_commit": heritage_commit,
        "primary_lines": int(active_legacy["loc"]),
        "legacy_primary_lines": int(active_legacy["loc"]),
        **metrics,
        "conservation": conservation,
        "active_inventory": active,
        "legacy_inventory": active_legacy,
        "heritage_inventory": {
            **heritage_measure,
            "legacy_primary_lines": int(heritage_legacy["loc"]),
        },
        "source_lock": {
            "path": lock_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(lock_raw),
            **lock_verification,
        },
        "pack_lock": pack_verification,
        "logical_test_surface": test_surface,
        "logical_test_surface_preserved": test_surface_preserved,
        "foundation_measurement_valid": foundation_valid,
        "north_star_rung": contract.get("north_star_rung"),
        "fallback_floor_order": contract.get("fallback_floor_order"),
        "rungs": rungs,
    }
    return report


def _rung(report: dict[str, Any], rung_id: str) -> dict[str, Any]:
    for row in report["rungs"]:
        if row["id"] == rung_id:
            return row
    raise LocFloorError(f"unknown rung: {rung_id}")


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        values = ["report"]
    elif values[0] == "--check":
        values[0] = "check"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--ref", default="HEAD")
    verify_parser = sub.add_parser("verify-lock")
    verify_parser.add_argument("--ref", default="HEAD")
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--ref", default="HEAD")
    check_parser.add_argument("--rung")
    args = parser.parse_args(values)
    try:
        result = build_report(args.ref)
        if args.command == "verify-lock":
            output: object = {
                "source_lock": result["source_lock"],
                "pack_lock": result["pack_lock"],
            }
            failed = (
                not result["source_lock"]["valid"]
                or not result["pack_lock"]["valid"]
            )
        else:
            output = result
            failed = args.command == "check" and not result["valid"]
            if args.command == "check" and args.rung:
                selected = _rung(result, args.rung)
                failed = failed or not selected["met"]
        print(json.dumps(output, indent=2, sort_keys=True))
        return 1 if failed else 0
    except (LocFloorError, OSError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "schema": REPORT_SCHEMA,
                    "valid": False,
                    "errors": [str(exc)],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
