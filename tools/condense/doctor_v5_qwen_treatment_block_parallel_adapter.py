#!/usr/bin/env python3.12
"""Pending-only Doctor treatment adapter for the block-parallel Qwen encoder."""
from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

import doctor_v5_accel_loader as _accel
import doctor_v5_gc_runtime_transition as _gc_transition
import doctor_v5_source_seal as _source_seal


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "doctor_v5_qwen_treatment_adapter.py"
BASE_SHA256 = "731aaaaad56bc9e0659db1ba573a42b4fe3812d28dbb27d8164cf5f8abc3dc7d"
CONTROL_ADAPTER_PATH = HERE / "doctor_v5_strand_ladder_block_parallel_adapter.py"
WORKER_PATH = HERE / "doctor_v5_strand_ladder_block_parallel_worker.py"
QUANTIZER = HERE.parents[1] / "build/strand-block-parallel/release/quantize-model-block-parallel"
ADAPTER_VERSION = "2-block-parallel"
AUTHORITY_PATH = _gc_transition.DEFAULT_AUTHORITY_PATH

_BASE = _accel.load_frozen("doctor_v5_qwen_treatment_adapter_frozen", BASE_PATH,
                           BASE_SHA256)
_BASE.__file__ = str(Path(__file__).resolve())
_BASE.CONTROL_ADAPTER_PATH = CONTROL_ADAPTER_PATH
_BASE.WORKER_PATH = WORKER_PATH
_BASE.QUANTIZER = QUANTIZER
_BASE.ADAPTER_VERSION = ADAPTER_VERSION
_BASE.CONTROL = _BASE._load_module(
    "doctor_v5_strand_ladder_block_parallel_control", CONTROL_ADAPTER_PATH
)
_ORIGINAL_BUILD_SPEC = _BASE.build_spec
_ORIGINAL_LOAD_MODULE = _BASE._load_module
_ORIGINAL_VALIDATE_GC_RECEIPT = _BASE._validate_gc_receipt
_GC_HOOK_LOCK = threading.RLock()
_HISTORICAL_RUNTIME_ERROR = (
    "dependency packed GC successor runtime identity changed"
)


def _load_module(name: str, path: Path) -> Any:
    module = _ORIGINAL_LOAD_MODULE(name, path)
    if Path(path).resolve(strict=False) == _BASE.ABI_PATH.resolve(strict=False):
        _source_seal.install_hash_reuse(module, attribute="_sha_file")
    return module


_BASE._load_module = _load_module


def _validate_gc_receipt(path: Path, *, binding: dict[str, Any],
                         result: dict[str, Any],
                         packed_rows: list[dict[str, Any]],
                         consumer_spec: dict[str, Any]) -> dict[str, Any]:
    """Run the frozen validator, authorizing only two exact runtime transitions."""
    try:
        return _ORIGINAL_VALIDATE_GC_RECEIPT(
            path, binding=binding, result=result, packed_rows=packed_rows,
            consumer_spec=consumer_spec,
        )
    except _BASE.TreatmentError as exc:
        if str(exc) != _HISTORICAL_RUNTIME_ERROR:
            raise

    transition = _gc_transition.validate_transition(
        authority_path=AUTHORITY_PATH, receipt_path=path,
        consumer_spec=consumer_spec,
    )
    successor_path = Path(transition["successor_path"]).resolve(strict=True)
    current = transition["current_runtime"]
    historical = transition["historical_runtime"]

    # The frozen implementation performs all canonical receipt, reporter,
    # deletion-allowlist, retained-evidence, and successor-program checks.  We
    # virtualize only its one historical runtime-file identity read.  Every
    # intercepted read first proves that the live file still has the exact
    # current identity validated by the transition authority.
    with _GC_HOOK_LOCK:
        original_hash_file = _BASE._hash_file

        def historical_hash_file(candidate: Path) -> tuple[str, int]:
            resolved = _BASE._workspace_path(str(candidate))
            if resolved != successor_path:
                return original_hash_file(candidate)
            observed_sha, observed_bytes = original_hash_file(resolved)
            if observed_sha != current["sha256"] or observed_bytes != current["bytes"]:
                raise _BASE.TreatmentError(
                    "authorized GC successor runtime changed during validation"
                )
            return historical["sha256"], historical["bytes"]

        _BASE._hash_file = historical_hash_file
        try:
            receipt = _ORIGINAL_VALIDATE_GC_RECEIPT(
                path, binding=binding, result=result, packed_rows=packed_rows,
                consumer_spec=consumer_spec,
            )
            final_sha, final_bytes = original_hash_file(successor_path)
            if final_sha != current["sha256"] or final_bytes != current["bytes"]:
                raise _BASE.TreatmentError(
                    "authorized GC successor runtime changed during validation"
                )
            return receipt
        finally:
            _BASE._hash_file = original_hash_file


def build_spec(**kwargs: Any) -> dict[str, Any]:
    seal_path = _source_seal.default_path(kwargs["label"])
    if not seal_path.is_file() or seal_path.is_symlink():
        raise _accel.AccelerationBindingError(
            f"hash-bound source seal is required before accelerated wiring: {seal_path}"
        )
    # Production accelerated specs must bind the immutable two-incident
    # transition authority.  No best-effort/no-authority build is available.
    _gc_transition.validate_authority(AUTHORITY_PATH)
    document = _ORIGINAL_BUILD_SPEC(**kwargs)
    document = _accel.bind_extra_inputs(document, (
        _accel.input_row("acceleration_loader", Path(_accel.__file__)),
        _accel.input_row("source_seal_module", Path(_source_seal.__file__)),
        _accel.input_row("source_seal", seal_path),
        _accel.input_row("frozen_treatment_adapter_base", BASE_PATH),
        _accel.input_row("frozen_control_adapter_base",
                         HERE / "doctor_v5_strand_ladder_adapter.py"),
        _accel.input_row("frozen_worker_base", HERE / "doctor_v5_strand_ladder_worker.py"),
        _gc_transition.module_input_row(),
        _gc_transition.authority_input_row(AUTHORITY_PATH),
    ))
    abi = _BASE._load_module("doctor_v5_block_parallel_treatment_spec_writer",
                             _BASE.ABI_PATH)
    abi.atomic_json(_BASE._workspace_path(str(kwargs["output_path"]), must_exist=False),
                    document)
    return document


_BASE.build_spec = build_spec
_BASE._validate_gc_receipt = _validate_gc_receipt
_accel.export_module(
    _BASE, globals(),
    keep={"build_spec", "CONTROL_ADAPTER_PATH", "WORKER_PATH", "QUANTIZER",
          "ADAPTER_VERSION", "AUTHORITY_PATH", "_load_module",
          "_validate_gc_receipt"},
)


if __name__ == "__main__":
    raise SystemExit(_BASE.main())
