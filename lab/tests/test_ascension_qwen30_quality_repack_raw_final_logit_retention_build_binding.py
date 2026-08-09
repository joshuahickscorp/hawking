"""Focused pure checks for Q30 raw-final-logit build binding."""
from __future__ import annotations

from pathlib import Path

import pytest

from lab.operators import ascension_qwen30_quality_repack_raw_final_logit_retention_build_binding as binding


def test_regular_requires_absolute_non_symlink_file(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.write_bytes(b"x")
    assert binding._regular(payload.resolve(), label="payload") == payload.resolve()
    with pytest.raises(binding.RawFinalLogitRetentionBuildBindingError, match="absolute"):
        binding._regular(Path("relative"), label="relative")


def test_source_mode_constant_is_stable() -> None:
    assert binding.retention_contract.NATIVE_MODE == "metal-diagnostic-retain-raw-final-logits"
