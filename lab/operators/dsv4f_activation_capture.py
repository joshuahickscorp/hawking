"""DSV4F activation-X capture helpers.

The Rust writer (`hawking-core` `dsv4f_activation_capture`) emits the Q80
on-disk shape. This module validates rows and hands the run directory to the
existing doctor6 collector unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
    ActivationWeightedRepackError,
    collect_expert_activations,
)

CAPTURE_RESULT_NAME = "capture-result.json"


def read_f32le(path: Path, expected_elements: int) -> np.ndarray:
    """Load one float32-LE row. A short or corrupt file raises; never truncate."""

    path = Path(path)
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise ActivationWeightedRepackError(
            f"corrupt hidden {path}: {len(raw)} bytes is not a multiple of 4"
        )
    x = np.frombuffer(raw, dtype="<f4")
    if int(x.size) != int(expected_elements):
        raise ActivationWeightedRepackError(
            f"hidden size mismatch at {path}: got {int(x.size)} f32 elements, "
            f"expected {int(expected_elements)}"
        )
    return np.array(x, copy=True)


def load_capture_result(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / CAPTURE_RESULT_NAME
    if not path.is_file():
        raise ActivationWeightedRepackError(f"missing {CAPTURE_RESULT_NAME}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def collect_via_doctor6(
    run_dir: Path,
    capture: Mapping[str, Any] | None = None,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """Unchanged doctor6 collector: (layer, expert) → float32 array."""

    return collect_expert_activations(Path(run_dir), capture)
