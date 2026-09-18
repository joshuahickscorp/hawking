#!/usr/bin/env python3
"""Flash E06: a bounded conditional-information *prefilter*.

This is deliberately a small source-row diagnostic, not a representation or
executor.  It opens safetensors headers through the existing Odyssey gate,
then range-reads one predeclared, disjoint sample under a hard byte cap.  It
measures marginal code occupancy/entropy, held-out conditional code length,
a shuffled-label null, and the predictor/decoder byte obligation separately.

It does not load a model, initialize a device, invoke a decoder/runtime, or
make capability, TPS, EBPW, or compression-admission claims.  A positive
prefilter result is merely a reason to consider a separately controlled next
experiment.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import random
import stat
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The source-root constants and relative-file validator are already the Flash
# owner used by the accepted-decode gate.  E06 borrows only identity/path
# semantics; it does not borrow that gate's native-reference contract.
from tools.odyssey.flash_repeated_accepted_decode import (  # noqa: E402
    DEFAULT_MODEL_ROOT,
    PINNED_REVISION,
    REPO_ID,
    SOURCE_SHARD_LEDGER_SCHEMA,
    SOURCE_SHARD_LEDGER_STATUS,
    _canonical_model_lake_manifest,
    _source_file,
    _verify_shard_ledger,
)
from hawking.persist import atomic_write_json  # noqa: E402
from tools.odyssey.specimen_open import (  # noqa: E402
    SpecimenOpenRefused,
    WeightBytesRefused,
    read_header,
)


SCHEMA = "hawking.flash.e06_conditional_information_prefilter.v1"
DEFAULT_TENSOR = "model.language_model.layers.0.mlp.experts.gate_up_proj"
DEFAULT_ROWS_PER_PARTITION = 1024
DEFAULT_SEED = 20260912
DEFAULT_BIN_COUNT = 8
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 1 * 1024 * 1024
MAX_INDEX_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_PREDICTOR_BUCKETS = 1024
MAX_SYNTHETIC_FIXTURE_SHARD_BYTES = 1 * 1024 * 1024
DEFAULT_SOURCE_SHARD_LEDGER = ROOT / "receipts/headless/FLASH_SOURCE_SHARD_LEDGER_20260912.json"
DEFAULT_EMIT = ROOT / "receipts/headless/FLASH_E06_CONDITIONAL_INFORMATION_PREFILTER_BOUND_20260912.json"
RECEIPT_SEAL_FORMAT = "python_json_compact_utf8_sorted_v1"


class E06Refused(RuntimeError):
    """The bounded E06 diagnostic cannot safely make its requested read."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _fingerprint(st: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(st.st_dev),
        "st_ino": int(st.st_ino),
        "st_size": int(st.st_size),
        "st_mtime_ns": int(st.st_mtime_ns),
    }


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_at_no_follow(path: Path, *, allowed_root: Path | None, label: str) -> int:
    """Open exactly one regular-file candidate without following links.

    ``Path.resolve`` alone is not an adequate source-body boundary: a path may
    be changed after its check.  This mirrors the repository's E13 descriptor
    traversal, but stays local because there is not yet a shared Odyssey
    range-reader API to extend without conflicting with its current owner.
    """
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise E06Refused("NOFOLLOW_OPEN_UNAVAILABLE: safe source row reads require O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow
    target = _absolute(path)
    if allowed_root is None:
        try:
            return os.open(target, flags)
        except OSError as exc:
            code = "SYMLINK_OPEN_REFUSED" if exc.errno == errno.ELOOP else "INPUT_UNAVAILABLE"
            raise E06Refused(f"{code}: cannot open {label}") from exc

    root = _absolute(allowed_root)
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise E06Refused(f"SOURCE_OUTSIDE_ROOT: {label} is outside the selected source root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise E06Refused(f"SOURCE_OUTSIDE_ROOT: unsafe {label} path")

    directory_flags = os.O_RDONLY | nofollow | getattr(os, "O_DIRECTORY", 0)
    directory_fd: int | None = None
    try:
        directory_fd = os.open(root, directory_flags)
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise E06Refused("SOURCE_ROOT_INVALID: selected source root is not a directory")
        for component in relative.parts[:-1]:
            child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
            if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
                raise E06Refused(f"SOURCE_ROOT_INVALID: {label} has a non-directory parent")
        return os.open(relative.parts[-1], flags, dir_fd=directory_fd)
    except E06Refused:
        raise
    except OSError as exc:
        code = "SYMLINK_OPEN_REFUSED" if exc.errno == errno.ELOOP else "INPUT_UNAVAILABLE"
        raise E06Refused(f"{code}: cannot safely open {label}") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _safe_stat(path: Path, *, allowed_root: Path | None, label: str) -> os.stat_result:
    fd = _open_at_no_follow(path, allowed_root=allowed_root, label=label)
    try:
        result = os.fstat(fd)
    finally:
        os.close(fd)
    if not stat.S_ISREG(result.st_mode):
        raise E06Refused(f"REGULAR_FILE_REQUIRED: {label} must be a regular file")
    # A hard-linked control/index/shard could be exchanged under a different
    # name.  This tiny source diagnostic has no need to accept that ambiguity.
    if result.st_nlink != 1:
        raise E06Refused(
            f"HARDLINKED_INPUT_REFUSED: {label} must have one link (got {result.st_nlink})"
        )
    return result


def _read_fd_up_to(fd: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum
    while remaining:
        chunk = os.read(fd, min(64 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_bounded_file(
    path: Path,
    *,
    maximum: int,
    allowed_root: Path | None,
    label: str,
) -> tuple[bytes, dict[str, int]]:
    if maximum <= 0:
        raise E06Refused("INVALID_BOUND: bounded input maximum must be positive")
    fd = _open_at_no_follow(path, allowed_root=allowed_root, label=label)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise E06Refused(f"REGULAR_FILE_REQUIRED: {label} must be a regular file")
        if before.st_nlink != 1:
            raise E06Refused(
                f"HARDLINKED_INPUT_REFUSED: {label} must have one link (got {before.st_nlink})"
            )
        if before.st_size > maximum:
            raise E06Refused(
                f"INPUT_TOO_LARGE: {label} is {before.st_size} bytes, above {maximum} bytes"
            )
        raw = _read_fd_up_to(fd, maximum + 1)
        after = os.fstat(fd)
    except OSError as exc:
        raise E06Refused(f"INPUT_UNAVAILABLE: cannot read {label}") from exc
    finally:
        os.close(fd)
    if _fingerprint(before) != _fingerprint(after):
        raise E06Refused(f"INPUT_CHANGED_DURING_READ: {label} changed during bounded read")
    if len(raw) > maximum:
        raise E06Refused(f"INPUT_TOO_LARGE: {label} grew past its bounded limit")
    if len(raw) != before.st_size:
        raise E06Refused(f"SHORT_INPUT: {label} was shorter than its checked size")
    return raw, _fingerprint(before)


def _read_json(
    path: Path,
    *,
    maximum: int,
    allowed_root: Path | None,
    label: str,
) -> tuple[dict[str, Any], int, str]:
    raw, _ = _read_bounded_file(path, maximum=maximum, allowed_root=allowed_root, label=label)
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise E06Refused(f"INVALID_JSON: {label} is not valid UTF-8 JSON") from exc
    if not isinstance(doc, dict):
        raise E06Refused(f"INVALID_JSON: {label} root must be an object")
    return doc, len(raw), _sha256_bytes(raw)


def _checked_root(root: Path | str, *, require_canonical: bool) -> Path:
    selected = _absolute(root)
    try:
        lstat = os.lstat(selected)
    except OSError as exc:
        raise E06Refused("SOURCE_ROOT_UNAVAILABLE: selected source root is absent") from exc
    if stat.S_ISLNK(lstat.st_mode) or not stat.S_ISDIR(lstat.st_mode):
        raise E06Refused("SOURCE_ROOT_INVALID: selected source root must be a real directory")
    if require_canonical:
        canonical = _absolute(DEFAULT_MODEL_ROOT)
        if selected != canonical:
            raise E06Refused("NONCANONICAL_SOURCE_REFUSED: E06 requires the pinned Flash ModelLake root")
    return selected


def _overlaps_canonical_source_path(path: Path | str) -> bool:
    """Whether ``path`` overlaps the pinned specimen in either direction.

    Fixture helpers must enforce this from the filesystem path, not from the
    mutable ``source_mode`` label carried in a plan.  A plan built with
    ``require_canonical=False`` against the default root is still the real
    Flash specimen and must never become eligible for a fixture range read.
    An ancestor is refused too: it could otherwise name the real shard while
    presenting a noncanonical root string in a manually altered fixture plan.
    """
    selected = _absolute(path)
    canonical = _absolute(DEFAULT_MODEL_ROOT)
    try:
        selected.relative_to(canonical)
        return True
    except ValueError:
        try:
            canonical.relative_to(selected)
            return True
        except ValueError:
            return False


def _safe_owner_source_file(root: Path, value: object, label: str) -> Path:
    """Use Flash's source-root resolver, then retain no-follow protection."""
    try:
        candidate = _source_file(root, value, label)
    except ValueError as exc:
        raise E06Refused(f"SOURCE_FILE_INVALID: {label}: {exc}") from exc
    # The existing helper supplies semantic root containment.  The descriptor
    # fd check supplies the non-follow / regular-file guarantee at use time.
    _safe_stat(candidate, allowed_root=root, label=label)
    return candidate


def _canonical_identity(
    root: Path, *, require_canonical: bool,
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any] | None]:
    """Read only bounded control metadata and bind it to the selected root."""
    index_path = _safe_owner_source_file(root, "model.safetensors.index.json", "safetensors index")
    index, index_bytes, index_sha256 = _read_json(
        index_path,
        maximum=MAX_INDEX_BYTES,
        allowed_root=root,
        label="safetensors index",
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise E06Refused("INDEX_WEIGHT_MAP_MISSING: safetensors index has no non-empty weight map")
    if not all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()):
        raise E06Refused("INDEX_WEIGHT_MAP_INVALID: safetensors index has malformed mappings")

    identity: dict[str, Any] = {
        "repo": REPO_ID if require_canonical else None,
        "pinned_revision": PINNED_REVISION if require_canonical else None,
        "source_root": str(root),
        "source_mode": "canonical_modellake" if require_canonical else "noncanonical_header_only",
        "index_path": str(index_path),
        "index_sha256": index_sha256,
    }
    metadata_bytes = index_bytes
    ledger: dict[str, Any] | None = None
    if require_canonical:
        try:
            manifest_path = _canonical_model_lake_manifest(root)
        except ValueError as exc:
            raise E06Refused(f"CANONICAL_MANIFEST_UNAVAILABLE: {exc}") from exc
        manifest, manifest_bytes, manifest_sha256 = _read_json(
            manifest_path,
            maximum=MAX_MANIFEST_BYTES,
            allowed_root=manifest_path.parent.parent,
            label="canonical ModelLake manifest",
        )
        if (
            manifest.get("repo") != REPO_ID
            or manifest.get("revision") != PINNED_REVISION
            or manifest.get("resolved_sha") != PINNED_REVISION
            or _absolute(str(manifest.get("path") or "")) != root
        ):
            raise E06Refused("CANONICAL_MANIFEST_MISMATCH: manifest does not bind pinned Flash source")
        identity.update(
            {
                "model_lake_manifest_path": str(manifest_path),
                "model_lake_manifest_sha256": manifest_sha256,
            }
        )
        metadata_bytes += manifest_bytes
        ledger_path = _absolute(DEFAULT_SOURCE_SHARD_LEDGER)
        ledger, ledger_bytes, ledger_sha256 = _read_json(
            ledger_path,
            maximum=MAX_MANIFEST_BYTES,
            allowed_root=ROOT,
            label="verified source shard ledger",
        )
        verified_ledger = _verify_shard_ledger(
            {
                "path": str(ledger_path),
                "sha256": ledger_sha256,
                "schema": SOURCE_SHARD_LEDGER_SCHEMA,
                "status": SOURCE_SHARD_LEDGER_STATUS,
                "seal_sha256": ledger.get("seal_sha256"),
            },
            contract_dir=ROOT,
            model_root=root,
            manifest_sha256=manifest_sha256,
            index_path=index_path,
            index_sha256=index_sha256,
        )
        identity["verified_source_shard_ledger"] = verified_ledger
        metadata_bytes += ledger_bytes
    return identity, index, metadata_bytes, ledger


def _positive_int(value: object, *, label: str, minimum: int, maximum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise E06Refused(f"INVALID_{label.upper()}: must be an integer") from exc
    if result < minimum or (maximum is not None and result > maximum):
        suffix = f"..{maximum}" if maximum is not None else "+"
        raise E06Refused(f"INVALID_{label.upper()}: must be in {minimum}{suffix}")
    return result


def _tensor_geometry(view: Mapping[str, Any], tensor: str) -> dict[str, Any]:
    tensors = view.get("tensors")
    if not isinstance(tensors, Mapping) or not isinstance(tensors.get(tensor), Mapping):
        raise E06Refused(f"TENSOR_MISSING: header does not contain {tensor!r}")
    descriptor = tensors[tensor]
    if descriptor.get("dtype") != "BF16":
        raise E06Refused(f"TENSOR_DTYPE_UNSUPPORTED: {tensor!r} is not BF16")
    raw_shape = descriptor.get("shape")
    raw_offsets = descriptor.get("data_offsets")
    if not (
        isinstance(raw_shape, list)
        and len(raw_shape) >= 2
        and all(isinstance(value, int) and value > 0 for value in raw_shape)
        and isinstance(raw_offsets, list)
        and len(raw_offsets) == 2
        and all(isinstance(value, int) and value >= 0 for value in raw_offsets)
        and raw_offsets[0] <= raw_offsets[1]
    ):
        raise E06Refused("TENSOR_GEOMETRY_INVALID: supported tensor must be rank >=2 BF16")
    shape = [int(value) for value in raw_shape]
    rows = math.prod(shape[:-1])
    width = shape[-1]
    row_bytes = width * 2
    payload_bytes = rows * row_bytes
    if int(raw_offsets[1]) - int(raw_offsets[0]) != payload_bytes:
        raise E06Refused("TENSOR_PAYLOAD_MISMATCH: BF16 geometry disagrees with header offsets")
    return {
        "dtype": "BF16",
        "shape": shape,
        "flattened_rows": rows,
        "row_width": width,
        "row_bytes": row_bytes,
        "data_offsets": [int(raw_offsets[0]), int(raw_offsets[1])],
        "tensor_payload_bytes": payload_bytes,
    }


def deterministic_partitions(
    rows: int,
    *,
    rows_per_partition: int,
    seed: int,
    max_payload_bytes: int,
    row_bytes: int,
) -> tuple[list[int], list[int]]:
    """Select disjoint fit/held-out rows before opening a tensor payload."""
    if row_bytes <= 0:
        raise E06Refused("TENSOR_GEOMETRY_INVALID: row byte count must be positive")
    if rows < 4:
        raise E06Refused("INSUFFICIENT_ROWS: E06 needs two fit and two held-out rows")
    capacity = max_payload_bytes // (2 * row_bytes)
    selected_per_partition = min(rows_per_partition, rows // 2, capacity)
    if selected_per_partition < 2:
        raise E06Refused(
            "PAYLOAD_BUDGET_TOO_SMALL: 16 MiB budget cannot form two disjoint partitions"
        )
    rng = random.Random(seed)
    chosen = rng.sample(range(rows), 2 * selected_per_partition)
    fit = sorted(chosen[:selected_per_partition])
    heldout = sorted(chosen[selected_per_partition:])
    if set(fit).intersection(heldout) or not fit or not heldout:
        raise AssertionError("random.sample must form disjoint non-empty partitions")
    return fit, heldout


def build_prefilter_plan(
    *,
    root: Path | str = DEFAULT_MODEL_ROOT,
    tensor: str = DEFAULT_TENSOR,
    rows_per_partition: int = DEFAULT_ROWS_PER_PARTITION,
    seed: int = DEFAULT_SEED,
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
    require_canonical: bool = True,
) -> dict[str, Any]:
    """Perform the E06 metadata-only preflight; this function never reads rows."""
    requested_rows = _positive_int(
        rows_per_partition, label="rows_per_partition", minimum=2, maximum=1_000_000
    )
    effective_seed = _positive_int(seed, label="seed", minimum=0, maximum=(1 << 63) - 1)
    payload_budget = _positive_int(
        max_payload_bytes,
        label="max_payload_bytes",
        minimum=1,
        maximum=MAX_PAYLOAD_BYTES,
    )
    if not isinstance(tensor, str) or not tensor:
        raise E06Refused("INVALID_TENSOR: tensor must be a non-empty header name")
    if require_canonical and (
        tensor != DEFAULT_TENSOR
        or requested_rows != DEFAULT_ROWS_PER_PARTITION
        or effective_seed != DEFAULT_SEED
        or payload_budget != MAX_PAYLOAD_BYTES
    ):
        raise E06Refused(
            "CANONICAL_PREFILTER_NOT_PREREGISTERED: source execution is fixed to the declared "
            "tensor, sample size, seed, and payload cap"
        )
    source_root = _checked_root(root, require_canonical=require_canonical)
    identity, index, control_metadata_bytes, source_shard_ledger = _canonical_identity(
        source_root, require_canonical=require_canonical
    )
    weight_map = index["weight_map"]
    shard_name = weight_map.get(tensor)
    if not isinstance(shard_name, str):
        raise E06Refused(f"TENSOR_NOT_INDEXED: index does not map {tensor!r} to a shard")
    shard = _safe_owner_source_file(source_root, shard_name, "selected tensor shard")
    source_shard_sha256: str | None = None
    if require_canonical:
        if not isinstance(source_shard_ledger, Mapping):
            raise AssertionError("canonical source identity must include the verified shard ledger")
        matching = [
            row
            for row in source_shard_ledger.get("shards", [])
            if isinstance(row, Mapping) and row.get("path") == shard_name
        ]
        if len(matching) != 1 or matching[0].get("verification") != "SHA256_EXACT_FILE_BYTES":
            raise E06Refused("SOURCE_SHARD_LEDGER_MISMATCH: selected shard lacks exact-byte identity")
        source_shard_sha256 = str(matching[0].get("sha256") or "")
        if len(source_shard_sha256) != 64:
            raise E06Refused("SOURCE_SHARD_LEDGER_MISMATCH: selected shard digest is invalid")
    before_header = _fingerprint(_safe_stat(shard, allowed_root=source_root, label="selected tensor shard"))
    try:
        header = read_header(
            shard,
            use_cache=False,
            nocache=False,
            header_cap=MAX_HEADER_BYTES,
        )
    except (SpecimenOpenRefused, WeightBytesRefused) as exc:
        raise E06Refused(f"SAFETENSORS_HEADER_REFUSED: {exc}") from exc
    after_header = _fingerprint(_safe_stat(shard, allowed_root=source_root, label="selected tensor shard"))
    if before_header != after_header:
        raise E06Refused("SOURCE_CHANGED_DURING_HEADER_READ: selected shard changed during header preflight")
    if header.get("touched_weight_bytes") is not False or header.get("bytes_read") != header.get("header_bytes"):
        raise E06Refused("HEADER_GATE_VIOLATION: preflight did not remain header-only")
    geometry = _tensor_geometry(header, tensor)
    fit_rows, heldout_rows = deterministic_partitions(
        geometry["flattened_rows"],
        rows_per_partition=requested_rows,
        seed=effective_seed,
        max_payload_bytes=payload_budget,
        row_bytes=geometry["row_bytes"],
    )
    planned_payload_bytes = (len(fit_rows) + len(heldout_rows)) * geometry["row_bytes"]
    if planned_payload_bytes > payload_budget or planned_payload_bytes > MAX_PAYLOAD_BYTES:
        raise AssertionError("partitioning exceeded its pre-open payload cap")
    header_bytes = int(header["header_bytes"])
    plan = {
        "schema": SCHEMA,
        "status": "DRY_RUN_HEADER_ONLY",
        "source_identity": identity,
        "source_identity_sha256": _sha256_value(identity),
        "tensor": {
            "name": tensor,
            "shard_path": str(shard),
            "source_shard_sha256": source_shard_sha256,
            "shard_fingerprint": after_header,
            "header_bytes": header_bytes,
            "header_bytes_read": int(header["bytes_read"]),
            **geometry,
        },
        "sampling": {
            "algorithm": "random_sample_without_replacement.v1",
            "seed": effective_seed,
            "fit_row_ids": fit_rows,
            "heldout_row_ids": heldout_rows,
            "rows_per_partition": len(fit_rows),
            "partitions_disjoint": True,
            "max_payload_bytes": payload_budget,
            "planned_payload_bytes": planned_payload_bytes,
        },
        "source_read_accounting": {
            "control_metadata_bytes_read": control_metadata_bytes,
            "header_bytes_read": header_bytes,
            "payload_bytes_read": 0,
            "total_source_bytes_read": control_metadata_bytes + header_bytes,
            "header_only": True,
        },
        "claim_boundary": {
            "is_model_load": False,
            "is_gpu_run": False,
            "is_runtime_or_decoder_execution": False,
            "is_capability_claim": False,
            "is_tps_claim": False,
            "is_ebpw_claim": False,
            "is_complete_source_ledger": False,
            "is_representation_artifact": False,
            "meaning": "Header-only E06 plan; this tranche leaves row payload unread.",
        },
    }
    plan["plan_sha256"] = _sha256_value(plan)
    return plan


def _synthetic_fixture_plan(**kwargs: Any) -> dict[str, Any]:
    """Mark a noncanonical plan for this module's fixture-only range tests.

    This is intentionally private and has no CLI switch.  It exists so the
    bounded range reader can be regression-tested with a tiny synthetic
    safetensors file without weakening the canonical source admission path.
    """
    options = dict(kwargs)
    if options.pop("require_canonical", False):
        raise E06Refused("SYNTHETIC_FIXTURE_REQUIRED: fixture helper cannot use canonical source")
    if _overlaps_canonical_source_path(options.get("root", DEFAULT_MODEL_ROOT)):
        raise E06Refused(
            "SYNTHETIC_FIXTURE_REQUIRED: fixture helper refuses the pinned Flash source root"
        )
    plan = build_prefilter_plan(require_canonical=False, **options)
    plan["test_only_synthetic_fixture"] = True
    plan["plan_sha256"] = _sha256_value(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    return plan


def _pread_exact(fd: int, offset: int, length: int) -> bytes:
    return os.pread(fd, length, offset)


def _read_sample_rows(
    plan: Mapping[str, Any], *, fixture_only: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Range-read exactly the rows already fixed by a verified plan."""
    expected_plan_sha256 = plan.get("plan_sha256")
    observed_plan_sha256 = _sha256_value(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if expected_plan_sha256 != observed_plan_sha256:
        raise E06Refused("PLAN_DIGEST_MISMATCH: plan changed after its row ranges were fixed")
    tensor = plan.get("tensor")
    sampling = plan.get("sampling")
    if not isinstance(tensor, Mapping) or not isinstance(sampling, Mapping):
        raise E06Refused("PLAN_INVALID: missing tensor or sampling sections")
    shard = Path(str(tensor.get("shard_path") or ""))
    root_value = (plan.get("source_identity") or {}).get("source_root")
    if not isinstance(root_value, str) or not root_value:
        raise E06Refused("PLAN_INVALID: source identity omitted root")
    source_root = _absolute(root_value)
    source_mode = (plan.get("source_identity") or {}).get("source_mode")
    if fixture_only:
        if plan.get("test_only_synthetic_fixture") is not True:
            raise E06Refused("SYNTHETIC_FIXTURE_REQUIRED: fixture marker is absent")
        if _overlaps_canonical_source_path(source_root):
            raise E06Refused(
                "SYNTHETIC_FIXTURE_REQUIRED: fixture range helper refuses the pinned Flash source root"
            )
        if source_mode != "noncanonical_header_only":
            raise E06Refused("SYNTHETIC_FIXTURE_REQUIRED: fixture helper refuses canonical source identity")
    else:
        if plan.get("test_only_synthetic_fixture") is True:
            raise E06Refused("CANONICAL_SOURCE_REQUIRED: source evaluator refuses a fixture plan")
        if source_root != _absolute(DEFAULT_MODEL_ROOT) or source_mode != "canonical_modellake":
            raise E06Refused("CANONICAL_SOURCE_REQUIRED: source evaluator requires the pinned Flash root")
        ledger = (plan.get("source_identity") or {}).get("verified_source_shard_ledger")
        if not isinstance(ledger, Mapping) or ledger.get("verification") != (
            "sha256_exact_file_bytes (attested by sealed ledger)"
        ):
            raise E06Refused("SOURCE_SHARD_LEDGER_REQUIRED: canonical sample lacks exact-byte ledger identity")
        if not isinstance(tensor.get("source_shard_sha256"), str):
            raise E06Refused("SOURCE_SHARD_LEDGER_REQUIRED: selected shard digest is absent")
    expected = tensor.get("shard_fingerprint")
    if not isinstance(expected, Mapping):
        raise E06Refused("PLAN_INVALID: selected shard fingerprint is absent")
    try:
        expected_shard_bytes = int(expected.get("st_size") or 0)
    except (AttributeError, TypeError, ValueError) as exc:
        raise E06Refused("PLAN_INVALID: synthetic fixture shard size is invalid") from exc
    if expected_shard_bytes <= 0 or (
        fixture_only and expected_shard_bytes > MAX_SYNTHETIC_FIXTURE_SHARD_BYTES
    ):
        raise E06Refused(
            "SYNTHETIC_FIXTURE_REQUIRED: fixture range helper refuses a production-sized shard"
        )
    row_bytes = int(tensor.get("row_bytes") or 0)
    header_bytes = int(tensor.get("header_bytes") or 0)
    offsets = tensor.get("data_offsets")
    if row_bytes <= 0 or header_bytes <= 0 or not isinstance(offsets, list) or len(offsets) != 2:
        raise E06Refused("PLAN_INVALID: selected tensor range is invalid")
    fit_ids = [int(value) for value in sampling.get("fit_row_ids") or []]
    heldout_ids = [int(value) for value in sampling.get("heldout_row_ids") or []]
    if not fit_ids or not heldout_ids or set(fit_ids).intersection(heldout_ids):
        raise E06Refused("PLAN_INVALID: planned partitions are missing or overlap")
    max_payload = int(sampling.get("max_payload_bytes") or 0)
    planned = int(sampling.get("planned_payload_bytes") or 0)
    if max_payload <= 0 or max_payload > MAX_PAYLOAD_BYTES or planned != (len(fit_ids) + len(heldout_ids)) * row_bytes:
        raise E06Refused("PLAN_INVALID: payload accounting is invalid")
    if planned > max_payload or planned > MAX_PAYLOAD_BYTES:
        raise E06Refused("PAYLOAD_BUDGET_EXCEEDED: planned payload is above the hard 16 MiB cap")

    payload_start = header_bytes + int(offsets[0])
    payload_end = header_bytes + int(offsets[1])
    fd = _open_at_no_follow(shard, allowed_root=source_root, label="selected tensor shard")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise E06Refused("SOURCE_FILE_INVALID: selected tensor shard is not a single-link regular file")
        if _fingerprint(before) != {str(k): int(v) for k, v in expected.items()}:
            raise E06Refused("SOURCE_CHANGED_BEFORE_PAYLOAD_READ: shard no longer matches header plan")
        if payload_start < header_bytes or payload_end > before.st_size or payload_start >= payload_end:
            raise E06Refused("PLAN_RANGE_INVALID: selected tensor range is outside the pinned shard")
        payload_read = 0
        ranges: list[dict[str, Any]] = []
        digest = hashlib.sha256()
        partitions: dict[str, list[np.ndarray]] = {"fit": [], "heldout": []}
        for partition, row_ids in (("fit", fit_ids), ("heldout", heldout_ids)):
            for row_id in row_ids:
                offset = payload_start + row_id * row_bytes
                if row_id < 0 or offset < payload_start or offset + row_bytes > payload_end:
                    raise E06Refused("PLAN_RANGE_INVALID: row lies outside selected tensor payload")
                if payload_read + row_bytes > max_payload or payload_read + row_bytes > MAX_PAYLOAD_BYTES:
                    raise E06Refused("PAYLOAD_BUDGET_EXCEEDED: refusing an over-budget row read")
                raw = _pread_exact(fd, offset, row_bytes)
                if len(raw) != row_bytes:
                    raise E06Refused(f"SHORT_RANGE_READ: {partition} row {row_id} was short")
                payload_read += len(raw)
                digest.update(partition.encode("ascii"))
                digest.update(row_id.to_bytes(8, "little", signed=False))
                digest.update(raw)
                partitions[partition].append(
                    (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16))
                    .view("<f4")
                    .copy()
                )
                ranges.append(
                    {
                        "partition": partition,
                        "row_id": row_id,
                        "file_offset": offset,
                        "bytes": row_bytes,
                    }
                )
        after = os.fstat(fd)
    except OSError as exc:
        raise E06Refused("SOURCE_PAYLOAD_UNAVAILABLE: selected row range could not be read") from exc
    finally:
        os.close(fd)
    if _fingerprint(before) != _fingerprint(after):
        raise E06Refused("SOURCE_CHANGED_DURING_PAYLOAD_READ: shard changed during range reads")
    if payload_read != planned:
        raise AssertionError("range reader must consume exactly the preplanned payload bytes")
    return (
        np.stack(partitions["fit"]).astype(np.float32, copy=False),
        np.stack(partitions["heldout"]).astype(np.float32, copy=False),
        {
            "payload_bytes_read": payload_read,
            "ranges": ranges,
            "sample_sha256": digest.hexdigest(),
            "no_mmap": True,
        },
    )


def _fixture_only_read_sample_rows(
    plan: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return _read_sample_rows(plan, fixture_only=True)


def _code_rows(
    fit: np.ndarray,
    heldout: np.ndarray,
    *,
    requested_bins: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    fit_norms = np.linalg.norm(np.asarray(fit, dtype=np.float32), axis=1)
    heldout_norms = np.linalg.norm(np.asarray(heldout, dtype=np.float32), axis=1)
    if not np.isfinite(fit_norms).all() or not np.isfinite(heldout_norms).all():
        raise E06Refused("NONFINITE_ROW_VALUES: E06 refuses non-finite BF16 row norms")
    quantiles = np.linspace(0.0, 1.0, requested_bins + 1, dtype=np.float64)
    try:
        edges = np.quantile(fit_norms, quantiles, method="linear")
    except TypeError:  # NumPy before 1.22 spells this interpolation.
        edges = np.quantile(fit_norms, quantiles, interpolation="linear")
    cuts = np.unique(np.asarray(edges[1:-1], dtype=np.float64))
    fit_codes = np.searchsorted(cuts, fit_norms, side="right").astype(np.int64)
    heldout_codes = np.searchsorted(cuts, heldout_norms, side="right").astype(np.int64)
    if len(cuts) + 1 < 2 or np.unique(heldout_codes).size < 2:
        raise E06Refused(
            "DEGENERATE_CODES: row-norm sample cannot support a nontrivial shuffled-label null"
        )
    return fit_codes, heldout_codes, [float(value) for value in cuts.tolist()]


def _predictor_features(row_ids: Sequence[int], shape: Sequence[int]) -> tuple[np.ndarray, dict[str, Any]]:
    total_rows = math.prod(shape[:-1])
    if len(shape) >= 3:
        leading_axis_size = int(shape[0])
        rows_per_leading_axis = math.prod(shape[1:-1])
        buckets = min(leading_axis_size, MAX_PREDICTOR_BUCKETS)
        axes = np.asarray(row_ids, dtype=np.int64) // rows_per_leading_axis
        labels = (axes * buckets) // leading_axis_size
        detail = {
            "kind": "leading_axis_0_bucket.v1",
            "leading_axis_size": leading_axis_size,
            "rows_per_leading_axis": rows_per_leading_axis,
            "bucket_count": buckets,
        }
    else:
        buckets = min(16, total_rows)
        rows = np.asarray(row_ids, dtype=np.int64)
        labels = (rows * buckets) // total_rows
        detail = {
            "kind": "flattened_row_position_bucket.v1",
            "bucket_count": buckets,
        }
    if np.any(labels < 0) or np.any(labels >= buckets):
        raise AssertionError("predictor bucket calculation escaped its stated range")
    return labels.astype(np.int64), detail


def _bits_from_probabilities(probabilities: np.ndarray, features: np.ndarray, codes: np.ndarray) -> float:
    picked = probabilities[features, codes]
    if np.any(picked <= 0.0) or not np.isfinite(picked).all():
        raise AssertionError("smoothed predictor probabilities must be finite and positive")
    return float(np.mean(-np.log2(picked)))


def _seeded_shuffled_null(codes: np.ndarray, *, seed: int) -> tuple[np.ndarray, list[int]]:
    """Return a seeded, position-deranged label null with the same histogram."""
    count = int(codes.size)
    if count < 2 or np.unique(codes).size < 2:
        raise E06Refused("DEGENERATE_NULL: shuffled null needs at least two distinct held-out codes")
    rng = random.Random(seed)
    permutation = list(range(count))
    for _ in range(128):
        rng.shuffle(permutation)
        if all(source != target for source, target in enumerate(permutation)):
            candidate = codes[np.asarray(permutation, dtype=np.int64)]
            if not np.array_equal(candidate, codes):
                return candidate, permutation
    # A rotation is an explicit derangement fallback, still selected from the
    # seeded RNG.  Try every nonidentity rotation to avoid periodic labels.
    start = rng.randrange(1, count)
    for delta in list(range(start, count)) + list(range(1, start)):
        permutation = [(index + delta) % count for index in range(count)]
        candidate = codes[np.asarray(permutation, dtype=np.int64)]
        if not np.array_equal(candidate, codes):
            return candidate, permutation
    raise E06Refused("DEGENERATE_NULL: unable to form a changed held-out label null")


def _pack_codes(codes: np.ndarray, bits_per_code: int) -> bytes:
    if bits_per_code <= 0:
        return b""
    buffer = 0
    buffered_bits = 0
    packed = bytearray()
    for raw_code in np.asarray(codes, dtype=np.int64):
        code = int(raw_code)
        if code < 0 or code >= (1 << bits_per_code):
            raise AssertionError("code does not fit its declared packed width")
        buffer |= code << buffered_bits
        buffered_bits += bits_per_code
        while buffered_bits >= 8:
            packed.append(buffer & 0xFF)
            buffer >>= 8
            buffered_bits -= 8
    if buffered_bits:
        packed.append(buffer & 0xFF)
    return bytes(packed)


def _evaluate_sample(
    plan: Mapping[str, Any], *, fixture_only: bool,
) -> dict[str, Any]:
    """Evaluate the fixed E06 sample without turning it into a codec claim."""
    fit, heldout, row_read = _read_sample_rows(plan, fixture_only=fixture_only)
    fit_codes, heldout_codes, cuts = _code_rows(
        fit,
        heldout,
        requested_bins=DEFAULT_BIN_COUNT,
    )
    tensor = plan["tensor"]
    sampling = plan["sampling"]
    shape = [int(value) for value in tensor["shape"]]
    fit_features, predictor = _predictor_features(sampling["fit_row_ids"], shape)
    heldout_features, _ = _predictor_features(sampling["heldout_row_ids"], shape)
    code_count = int(max(fit_codes.max(), heldout_codes.max()) + 1)
    if code_count < 2:
        raise E06Refused("DEGENERATE_CODES: fewer than two code symbols were observed")
    predictor_table = np.ones((predictor["bucket_count"], code_count), dtype=np.float32)
    np.add.at(predictor_table, (fit_features, fit_codes), np.float32(1.0))
    predictor_table /= predictor_table.sum(axis=1, keepdims=True)
    global_counts = np.bincount(fit_codes, minlength=code_count).astype(np.float32) + np.float32(1.0)
    global_probabilities = global_counts / global_counts.sum()
    global_bits = float(np.mean(-np.log2(global_probabilities[heldout_codes])))
    conditional_bits = _bits_from_probabilities(predictor_table, heldout_features, heldout_codes)
    null_codes, permutation = _seeded_shuffled_null(
        heldout_codes,
        seed=int(sampling["seed"]) ^ 0xE06,
    )
    if not np.array_equal(
        np.bincount(null_codes, minlength=code_count), np.bincount(heldout_codes, minlength=code_count)
    ):
        raise AssertionError("a label permutation must preserve its code histogram")
    null_conditional_bits = _bits_from_probabilities(predictor_table, heldout_features, null_codes)
    heldout_counts = np.bincount(heldout_codes, minlength=code_count)
    heldout_probability = heldout_counts / heldout_codes.size
    occupancy = float(heldout_probability.max())
    entropy = float(
        -np.sum(heldout_probability[heldout_probability > 0] * np.log2(heldout_probability[heldout_probability > 0]))
    )
    packed_codes = _pack_codes(heldout_codes, max(1, math.ceil(math.log2(code_count))))
    predictor_descriptor = {
        "schema": "hawking.flash.e06_position_predictor.v1",
        "predictor": predictor,
        "code_count": code_count,
        "table_dtype": "float32",
        "table_shape": list(predictor_table.shape),
        "table_sha256": _sha256_bytes(predictor_table.tobytes(order="C")),
    }
    decoder_descriptor = {
        "schema": "hawking.flash.e06_row_norm_code_decoder.v1",
        "bf16_decode": "little_endian_u16_shift_left_16_to_f32.v1",
        "feature": "row_l2_norm.v1",
        "code_cuts": cuts,
        "code_count": code_count,
        "packed_code_bits": max(1, math.ceil(math.log2(code_count))),
        "global_fit_counts_with_laplace_one": [int(value) for value in global_counts.tolist()],
    }
    predictor_descriptor_bytes = _canonical_bytes(predictor_descriptor)
    decoder_descriptor_bytes = _canonical_bytes(decoder_descriptor)
    source_reads = dict(plan["source_read_accounting"])
    source_reads.update(
        {
            "payload_bytes_read": int(row_read["payload_bytes_read"]),
            "total_source_bytes_read": int(source_reads["total_source_bytes_read"])
            + int(row_read["payload_bytes_read"]),
            "header_only": False,
            "range_reads": row_read["ranges"],
            "sample_sha256": row_read["sample_sha256"],
            "no_mmap": True,
        }
    )
    accounting = {
        "sample_code_payload_bytes": len(packed_codes),
        "predictor_probability_table_bytes": int(predictor_table.nbytes),
        "predictor_descriptor_bytes": len(predictor_descriptor_bytes),
        "source_safetensors_header_bytes": int(tensor["header_bytes_read"]),
        "source_control_metadata_bytes": int(source_reads["control_metadata_bytes_read"]),
        "decoder_descriptor_bytes": len(decoder_descriptor_bytes),
    }
    accounting["total_diagnostic_bytes"] = sum(int(value) for value in accounting.values())
    conditional_saving = global_bits - conditional_bits
    null_saving = global_bits - null_conditional_bits
    saving_percent = 100.0 * conditional_saving / global_bits if global_bits > 0.0 else None
    fixed_width_bits = max(1, math.ceil(math.log2(code_count)))
    projected_symbol_count = int(tensor["flattened_rows"])
    global_gross_saving_bits = (fixed_width_bits - global_bits) * projected_symbol_count
    conditional_gross_saving_bits = conditional_saving * projected_symbol_count
    global_auxiliary_bytes = len(decoder_descriptor_bytes)
    conditional_auxiliary_bytes = (
        int(predictor_table.nbytes)
        + len(predictor_descriptor_bytes)
        + len(decoder_descriptor_bytes)
    )
    global_net_saving_bits = global_gross_saving_bits - 8 * global_auxiliary_bytes
    conditional_net_saving_bits = conditional_gross_saving_bits - 8 * conditional_auxiliary_bytes
    actual_minus_null_saving = conditional_saving - null_saving
    occupancy_signal = occupancy >= 0.20 and global_net_saving_bits > 0.0
    conditional_signal = (
        saving_percent is not None
        and saving_percent >= 10.0
        and actual_minus_null_saving > 0.0
        and conditional_net_saving_bits > 0.0
    )
    candidate_for_review = occupancy_signal or conditional_signal
    status = (
        "SYNTHETIC_FIXTURE_EVALUATION__PREFILTER_TEST_ONLY"
        if fixture_only
        else (
            "E06_PREFILTER_SIGNAL_REQUIRES_SEPARATE_SOURCE_OUTPUT_REVIEW"
            if candidate_for_review
            else "E06_PREFILTER_STOP_AUXILIARY_COST_OR_NULL_CONTROL_NOT_CLEARED"
        )
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "plan_sha256": plan["plan_sha256"],
        "source_identity": plan["source_identity"],
        "source_identity_sha256": plan["source_identity_sha256"],
        "tensor": {
            key: tensor[key]
            for key in (
                "name",
                "dtype",
                "shape",
                "flattened_rows",
                "row_width",
                "row_bytes",
                "data_offsets",
                "tensor_payload_bytes",
                "header_bytes",
                "source_shard_sha256",
            )
        },
        "sampling": {
            **sampling,
            "sample_sha256": row_read["sample_sha256"],
            "range_read_count": len(row_read["ranges"]),
        },
        "source_read_accounting": source_reads,
        "conditional_information": {
            "fit_code_count": int(fit_codes.size),
            "heldout_code_count": int(heldout_codes.size),
            "code_count": code_count,
            "heldout_largest_codeword_share": occupancy,
            "heldout_shannon_entropy_bits": entropy,
            "global_code_bits_per_heldout_symbol": global_bits,
            "predictor_code_bits_per_heldout_symbol": conditional_bits,
            "conditional_code_saving_bits_per_heldout_symbol": conditional_saving,
            "conditional_code_saving_percent_of_global": saving_percent,
        },
        "shuffled_null": {
            "algorithm": "seeded_heldout_label_permutation_derangement.v1",
            "seed": int(sampling["seed"]) ^ 0xE06,
            "permutation_sha256": _sha256_value(permutation),
            "null_code_sha256": _sha256_bytes(null_codes.tobytes()),
            "position_derangement": all(index != target for index, target in enumerate(permutation)),
            "histogram_preserved": True,
            "predictor_code_bits_per_heldout_symbol": null_conditional_bits,
            "conditional_code_saving_bits_per_heldout_symbol": null_saving,
            "actual_minus_null_saving_bits_per_symbol": conditional_saving - null_saving,
            "note": "Marginal occupancy and entropy are intentionally invariant under this label permutation; the control tests predictor association.",
        },
        "actual_diagnostic_byte_accounting": accounting,
        "projected_complete_code_accounting": {
            "scope": "heldout_sample_rate_extrapolation_to_selected_tensor_flattened_rows",
            "symbol_count": projected_symbol_count,
            "fixed_width_bits_per_symbol": fixed_width_bits,
            "global_cross_entropy_bits_per_symbol": global_bits,
            "conditional_cross_entropy_bits_per_symbol": conditional_bits,
            "global_gross_saving_bits_vs_fixed_width": global_gross_saving_bits,
            "global_auxiliary_bytes": global_auxiliary_bytes,
            "global_net_saving_bits_vs_fixed_width": global_net_saving_bits,
            "conditional_gross_saving_bits_vs_global": conditional_gross_saving_bits,
            "conditional_auxiliary_bytes": conditional_auxiliary_bytes,
            "conditional_net_saving_bits_vs_global": conditional_net_saving_bits,
            "is_ideal_code_length_not_serialized_codec": True,
        },
        "escalation_signal": {
            "occupancy_at_least_0_20": occupancy >= 0.20,
            "conditional_saving_percent_at_least_10": saving_percent is not None and saving_percent >= 10.0,
            "actual_minus_null_saving_positive": actual_minus_null_saving > 0.0,
            "global_saving_exceeds_decoder_bytes": global_net_saving_bits > 0.0,
            "conditional_saving_exceeds_predictor_and_decoder_bytes": conditional_net_saving_bits > 0.0,
            "marginal_occupancy_lane_signal": occupancy_signal,
            "conditional_predictor_lane_signal": conditional_signal,
            "candidate_for_separate_review": candidate_for_review,
            "not_an_automatic_escalation": True,
        },
        "claim_boundary": {
            "is_model_load": False,
            "is_gpu_run": False,
            "is_runtime_or_decoder_execution": False,
            "is_capability_claim": False,
            "is_tps_claim": False,
            "is_ebpw_claim": False,
            "is_complete_source_ledger": False,
            "is_representation_artifact": False,
            "meaning": "Bounded row-sample conditional-information diagnostic only; not a model, representation, decoder, or performance result.",
        },
    }
    report["report_sha256"] = _sha256_value(report)
    return report


def _fixture_only_evaluate_sample(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise diagnostic math against a synthetic fixture, for tests only."""
    return _evaluate_sample(plan, fixture_only=True)


def _producer_identity() -> dict[str, Any]:
    path = Path(__file__).resolve()
    return {
        "module": "tools.odyssey.flash_e06_conditional_information_prefilter",
        "path": str(path),
        "sha256": _sha256_bytes(path.read_bytes()),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
    }


def verify_prefilter_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("schema") != SCHEMA:
        raise E06Refused("RECEIPT_SCHEMA_MISMATCH: E06 receipt schema is invalid")
    if report.get("seal_format") != RECEIPT_SEAL_FORMAT:
        raise E06Refused("RECEIPT_SEAL_FORMAT_MISMATCH: E06 receipt seal format is invalid")
    body = dict(report)
    recorded_seal = body.pop("seal_sha256", None)
    if recorded_seal != _sha256_value(body):
        raise E06Refused("RECEIPT_SEAL_MISMATCH: E06 receipt seal is invalid")
    recorded_report_sha256 = body.pop("report_sha256", None)
    if recorded_report_sha256 != _sha256_value(body):
        raise E06Refused("REPORT_DIGEST_MISMATCH: E06 report digest is invalid")
    return dict(report)


def run_canonical_prefilter(*, emit: Path | str = DEFAULT_EMIT) -> dict[str, Any]:
    """Run the one preregistered CPU-only source sample and seal its receipt."""
    started_ns = time.time_ns()
    plan = build_prefilter_plan(
        root=DEFAULT_MODEL_ROOT,
        tensor=DEFAULT_TENSOR,
        rows_per_partition=DEFAULT_ROWS_PER_PARTITION,
        seed=DEFAULT_SEED,
        max_payload_bytes=MAX_PAYLOAD_BYTES,
        require_canonical=True,
    )
    report = _evaluate_sample(plan, fixture_only=False)
    report.pop("report_sha256", None)
    report.update(
        {
            "producer": _producer_identity(),
            "started_at_ns": started_ns,
            "finished_at_ns": time.time_ns(),
            "model_loaded": False,
            "gpu_session_started": False,
            "seal_format": RECEIPT_SEAL_FORMAT,
        }
    )
    report["elapsed_ns"] = report["finished_at_ns"] - report["started_at_ns"]
    report["report_sha256"] = _sha256_value(report)
    report["seal_sha256"] = _sha256_value(report)
    verify_prefilter_receipt(report)
    destination = _absolute(emit)
    if destination.exists():
        raise E06Refused(f"RECEIPT_EXISTS: refusing to replace {destination}")
    atomic_write_json(destination, report)
    return report


def _print_json(value: Mapping[str, Any]) -> None:
    print(_canonical_bytes(value).decode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--rows-per-partition", type=int, default=DEFAULT_ROWS_PER_PARTITION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-payload-bytes", type=int, default=MAX_PAYLOAD_BYTES)
    parser.add_argument("--emit", type=Path, default=DEFAULT_EMIT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read bounded control metadata and a safetensors header only; never read row payload",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        try:
            _print_json(
                build_prefilter_plan(
                    root=args.root,
                    tensor=args.tensor,
                    rows_per_partition=args.rows_per_partition,
                    seed=args.seed,
                    max_payload_bytes=args.max_payload_bytes,
                    require_canonical=True,
                )
            )
            return 0
        except E06Refused as exc:
            _print_json({"schema": SCHEMA, "status": "WITHHELD", "reason": str(exc)})
            return 2

    try:
        if (
            _absolute(args.root) != _absolute(DEFAULT_MODEL_ROOT)
            or args.tensor != DEFAULT_TENSOR
            or args.rows_per_partition != DEFAULT_ROWS_PER_PARTITION
            or args.seed != DEFAULT_SEED
            or args.max_payload_bytes != MAX_PAYLOAD_BYTES
        ):
            raise E06Refused(
                "CANONICAL_PREFILTER_NOT_PREREGISTERED: source execution accepts no parameter deviations"
            )
        _print_json(run_canonical_prefilter(emit=args.emit))
        return 0
    except E06Refused as exc:
        _print_json({"schema": SCHEMA, "status": "WITHHELD", "reason": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
