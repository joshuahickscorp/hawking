"""Pure boundary checks for the HQ30GR2 all-layer preparation contract."""
from __future__ import annotations

from lab.operators.ascension_qwen30_quality_repack_all_layer_current_trace_prepare import (
    STATUS,
    TARGET_POSITION,
    TARGET_PROBE,
    TARGET_TOKEN_COUNT,
)


def test_all_layer_plan_is_one_current_trace_and_one_forced_continuation_only() -> None:
    assert TARGET_PROBE == "literal_hawking"
    assert TARGET_POSITION == 337
    assert TARGET_TOKEN_COUNT == 369
    assert STATUS.endswith("NOT_RUN")
