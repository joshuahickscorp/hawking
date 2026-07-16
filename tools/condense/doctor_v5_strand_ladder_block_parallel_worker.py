#!/usr/bin/env python3.12
"""Pending-only Qwen worker using the separately built block-parallel encoder."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import doctor_v5_accel_loader as _accel
import doctor_v5_source_seal as _source_seal


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "doctor_v5_strand_ladder_worker.py"
BASE_SHA256 = "e47e924481658b1f016db18c70e5847da2c7ffa7076cc35100064e3c0aade974"
BLOCK_PARALLEL_QUANTIZER = (
    HERE.parents[1] / "build/strand-block-parallel/release/quantize-model-block-parallel"
)
BLOCK_THREADS = 20
BLOCK_SCRATCH_BUDGET_BYTES = 256 * 1024 * 1024

_BASE = _accel.load_frozen("doctor_v5_strand_ladder_worker_frozen", BASE_PATH,
                           BASE_SHA256)
_BASE.__file__ = str(Path(__file__).resolve())
_source_seal.install_hash_reuse(_BASE.BASE)
_ORIGINAL_QUANTIZER_ARGV = _BASE._quantizer_argv
_ORIGINAL_FIXED_ENV = _BASE.BASE._fixed_env


def _fixed_env() -> dict[str, str]:
    env = _ORIGINAL_FIXED_ENV()
    env["STRAND_NO_GPU"] = "1"
    return env


_BASE.BASE._fixed_env = _fixed_env


def _quantizer_argv(request: dict[str, Any], source: Path, output: Path) -> list[str]:
    argv = _ORIGINAL_QUANTIZER_ARGV(request, source, output)
    expected = str(Path(request["execution"]["quantizer_path"]).resolve())
    if expected != str(BLOCK_PARALLEL_QUANTIZER.resolve()):
        raise _BASE.LadderError("block-parallel worker received the wrong quantizer binding")
    threads = request["execution"]["threads"]
    if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 32:
        raise _BASE.LadderError("block-parallel thread binding is invalid")
    return argv + [
        "--block-threads", str(min(BLOCK_THREADS, threads)),
        "--block-scratch-budget-bytes", str(BLOCK_SCRATCH_BUDGET_BYTES),
    ]


_BASE._quantizer_argv = _quantizer_argv
_accel.export_module(_BASE, globals(), keep={"_quantizer_argv"})


if __name__ == "__main__":
    raise SystemExit(_BASE.main())
