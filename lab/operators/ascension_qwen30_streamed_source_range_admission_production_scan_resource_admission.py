#!/usr/bin/env python3
"""Read-only resource admission for the distinct Q30 production hash scanner.

This module deliberately observes only host accounting and sealed CPU-only
records.  It does not accept a source-root argument, open source data, issue a
lease, or start a child.  Its one purpose is to bind a fresh clean resource
window to the *production* scanner binary, while retaining the older bootstrap
resource observation as ancestry only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_bootstrap_resource_admission_preflight as legacy,
)
from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_production_scan_outer_reaper as outer,
)
from lab.receipts import seal

SCHEMA = outer.PRODUCTION_RESOURCE_SCHEMA
PREPARED_STATUS = outer.PRODUCTION_RESOURCE_STATUS
REFUSED_STATUS = (
    "REFUSED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_"
    "RESOURCE_WINDOW_UNSAFE_OR_UNBOUND"
)
MINIMUM_RECLAIMABLE_BYTES = legacy.MINIMUM_RECLAIMABLE_BYTES


class ProductionResourceAdmissionError(RuntimeError):
    """The new binary or its read-only resource window is not safely bound."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionResourceAdmissionError(f"{label} must be a nonnegative integer")
    return value


def _evidence(document: outer.Document | None) -> dict[str, object]:
    if document is None:
        return {"present": False}
    return {
        "path": str(document.path),
        "raw_document_sha256": document.raw_document_sha256,
        "seal_sha256": document.seal_sha256,
    }


def _pointer(document: outer.Document) -> dict[str, str]:
    return {
        "raw_document_sha256": document.raw_document_sha256,
        "seal_sha256": document.seal_sha256,
    }


def _sample_fields(sample: legacy.HostSample) -> dict[str, object]:
    return {
        "backend": sample.backend,
        "swap_used_bytes": sample.swap_used_bytes,
        "swapouts_pages": sample.swapouts_pages,
        "reclaimable_bytes": sample.reclaimable_bytes,
        "q30_capture_child_count": len(sample.q30_capture_children),
        "q80_capture_child_count": len(sample.q80_capture_children),
        "q30_capture_child_fingerprints": list(sample.q30_capture_children),
        "q80_capture_child_fingerprints": list(sample.q80_capture_children),
    }


def _validate_sample(sample: legacy.HostSample) -> None:
    try:
        legacy._validate_sample(sample)
    except legacy.ResourceAdmissionError as exc:
        raise ProductionResourceAdmissionError(str(exc)) from exc


def _profile(*, safety_floor_bytes: int) -> dict[str, object]:
    return {
        "exactly_one_future_non_inference_hash_scan_child": True,
        "maximum_concurrent_source_hash_scan_children": 1,
        "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
        "maximum_live_raw_bf16_windows": 1,
        "maximum_cached_raw_bf16_bytes": outer.MAX_POSITIONED_READ_BYTES,
        "maximum_shards": outer.SOURCE_SHARDS,
        "maximum_tensors": outer.SOURCE_TENSORS,
        "minimum_reclaimable_bytes_required": safety_floor_bytes,
        "source_teacher_or_logits_allowed": False,
        "source_model_residency_allowed": False,
        "model_gpu_server_hcli_or_tps_allowed": False,
        "source_root_statted_or_opened": False,
    }


def build_resource_admission(
    *,
    bootstrap_preflight_path: Path,
    bootstrap_binary_path: Path,
    bootstrap_resource_path: Path,
    production_binary_path: Path,
    safety_floor_bytes: int = MINIMUM_RECLAIMABLE_BYTES,
    snapshot_provider: Callable[[], legacy.HostSample] = legacy.collect_live_host_sample,
) -> dict[str, Any]:
    """Seal a fresh production-binary-bound observation or a refusal.

    The supplied snapshot provider exists only for synthetic tests.  The
    default takes two small, read-only host samples after the production binary
    is cryptographically validated; neither path has a source-root parameter.
    """
    safety_floor_bytes = _nonnegative_int(
        safety_floor_bytes, label="minimum reclaimable safety floor"
    )
    if safety_floor_bytes == 0:
        raise ProductionResourceAdmissionError("minimum reclaimable safety floor must be positive")

    blockers: list[str] = []
    preflight: outer.Document | None = None
    bootstrap_binary: outer.Document | None = None
    bootstrap_resource: outer.Document | None = None
    production_binary: outer.Document | None = None
    production_binary_sha: str | None = None

    try:
        (
            preflight,
            bootstrap_binary,
            bootstrap_resource,
            _legacy_binary_sha,
            _legacy_window,
        ) = outer._validate_existing_bootstrap_chain(
            preflight_path=bootstrap_preflight_path,
            binary_path=bootstrap_binary_path,
            resource_path=bootstrap_resource_path,
        )
    except outer.ProductionScanOuterError as exc:
        blockers.append(f"legacy_bootstrap_ancestry_invalid:{exc}")

    if preflight is None or bootstrap_binary is None or bootstrap_resource is None:
        blockers.append("production_binary_not_evaluated_without_valid_legacy_ancestry")
    else:
        try:
            production_binary = outer._sealed(
                production_binary_path, label="production binary binding"
            )
            production_binary_sha = outer._validate_production_binary(
                production_binary,
                preflight=preflight,
                bootstrap_binary=bootstrap_binary,
                resource=bootstrap_resource,
            )
        except outer.ProductionScanOuterError as exc:
            blockers.append(f"production_binary_binding_invalid:{exc}")

    before: legacy.HostSample | None = None
    after: legacy.HostSample | None = None
    if production_binary_sha is not None:
        try:
            before = snapshot_provider()
            after = snapshot_provider()
            _validate_sample(before)
            _validate_sample(after)
        except ProductionResourceAdmissionError as exc:
            blockers.append(f"read_only_resource_observation_invalid:{exc}")
        except Exception as exc:  # Test adapters are untrusted too.
            blockers.append(f"read_only_resource_observation_failed:{type(exc).__name__}:{exc}")
    else:
        blockers.append("resource_observation_not_evaluated_without_valid_production_binary")

    swapout_delta: int | None = None
    measured_reclaimable: int | None = None
    if before is not None and after is not None:
        samples = (before, after)
        if any(sample.swap_used_bytes != 0 for sample in samples):
            blockers.append("nonzero_swap_used_bytes")
        swapout_delta = after.swapouts_pages - before.swapouts_pages
        if swapout_delta != 0:
            blockers.append("nonzero_swapout_pages_delta")
        measured_reclaimable = min(sample.reclaimable_bytes for sample in samples)
        if measured_reclaimable < safety_floor_bytes:
            blockers.append("reclaimable_bytes_below_explicit_safety_floor")
        if before.q30_capture_children or after.q30_capture_children:
            blockers.append("active_q30_capture_child_detected")
        if before.q80_capture_children or after.q80_capture_children:
            blockers.append("active_q80_capture_child_detected")

    blockers = sorted(set(blockers))
    admitted = not blockers
    observation = {
        "sample_count": 2 if before is not None and after is not None else 0,
        "before": _sample_fields(before) if before is not None else {"present": False},
        "after": _sample_fields(after) if after is not None else {"present": False},
        "swapouts_pages_delta": swapout_delta,
        "captures_observed_before_and_after": True,
        "source_root_or_payload_observed": False,
    }
    identity = _sha256_json(
        {
            "schema": SCHEMA,
            "production_binary": _evidence(production_binary),
            "bootstrap_resource_ancestry": _evidence(bootstrap_resource),
            "safety_floor_bytes": safety_floor_bytes,
            "observation": observation,
        }
    )
    return seal(
        {
            "schema": SCHEMA,
            "status": PREPARED_STATUS if admitted else REFUSED_STATUS,
            "recorded_at": _utc_now(),
            "prepared": admitted,
            "fresh_observation": before is not None and after is not None,
            "observed_after_production_binary_binding": production_binary_sha is not None,
            "exclusive_clean_window": admitted,
            "zero_swap": before is not None
            and after is not None
            and before.swap_used_bytes == 0
            and after.swap_used_bytes == 0,
            "zero_swapouts": swapout_delta == 0,
            "no_active_q30_or_q80_capture_child": before is not None
            and after is not None
            and not before.q30_capture_children
            and not after.q30_capture_children
            and not before.q80_capture_children
            and not after.q80_capture_children,
            "resource_admitted_for_one_future_child": admitted,
            "source_payload_opened": False,
            "source_model_loaded": False,
            "source_teacher_or_logits_executed": False,
            "native_phase_started": False,
            "gpu_server_hcli_or_tps_action": False,
            "lease_issued_or_consumed": False,
            "child_started": False,
            "swap_used_bytes": after.swap_used_bytes if after is not None else None,
            "swapouts_pages_delta": swapout_delta,
            "reclaimable_bytes": measured_reclaimable,
            "minimum_reclaimable_bytes_required": safety_floor_bytes,
            "production_binary_sha256": production_binary_sha,
            "production_binary_binding": _pointer(production_binary)
            if production_binary is not None
            else {"present": False},
            "bootstrap_resource_ancestry": _pointer(bootstrap_resource)
            if bootstrap_resource is not None
            else {"present": False},
            "production_resource_window_identity_sha256": identity,
            "bounded_one_source_hash_scan_resource_profile": _profile(
                safety_floor_bytes=safety_floor_bytes
            ),
            "read_only_observation": observation,
            "blockers": blockers,
            "execution_boundary": {
                "source_root_argument_or_stat_performed": False,
                "source_payload_opened": False,
                "source_model_loaded": False,
                "capture_child_spawned": False,
                "gpu_server_hcli_or_tps_action": False,
                "lease_issued_or_consumed_or_released": False,
                "q30_or_q80_capture_child_started": False,
            },
            "claim_boundary": "Read-only production-binary-bound resource observation only. A successful record is not a lease, child launch, source scan, source-teacher, runtime admission, native phase, model, GPU, server, HCLI, TPS, TG, or tournament result.",
        }
    )


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise ProductionResourceAdmissionError(
            "--out must be a new absolute path below an existing parent"
        )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-preflight", type=Path, required=True)
    parser.add_argument("--bootstrap-binary", type=Path, required=True)
    parser.add_argument("--bootstrap-resource", type=Path, required=True)
    parser.add_argument("--production-binary", type=Path, required=True)
    parser.add_argument("--minimum-reclaimable-bytes", type=int, default=MINIMUM_RECLAIMABLE_BYTES)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = build_resource_admission(
            bootstrap_preflight_path=args.bootstrap_preflight,
            bootstrap_binary_path=args.bootstrap_binary,
            bootstrap_resource_path=args.bootstrap_resource,
            production_binary_path=args.production_binary,
            safety_floor_bytes=args.minimum_reclaimable_bytes,
        )
        _write_new(args.out, document)
    except ProductionResourceAdmissionError as exc:
        print(f"Q30 production hash-scan resource admission refused: {exc}")
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "status": document["status"],
                "seal_sha256": document["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
