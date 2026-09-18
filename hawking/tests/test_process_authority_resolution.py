"""The process tool must select the process authority, not the inference CLI."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hawking import processes


def test_process_authority_explicit_override_is_canonical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "hawking-process-authority"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("HAWKING_PROCESS_AUTHORITY_BIN", str(binary))
    monkeypatch.delenv("HAWKING_NATIVE_PROCESS_BIN", raising=False)
    assert processes._native_binary(tmp_path) == binary


def test_inference_binary_name_is_not_a_process_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inference = tmp_path / "workspace/ops/build/rust/release/hawking"
    inference.parent.mkdir(parents=True)
    inference.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    inference.chmod(0o755)
    monkeypatch.delenv("HAWKING_PROCESS_AUTHORITY_BIN", raising=False)
    monkeypatch.delenv("HAWKING_NATIVE_PROCESS_BIN", raising=False)
    # The source checkout may have a real process authority build, which is a
    # valid fallback.  The important regression is that the nearby inference
    # binary is never selected merely because it is named ``hawking``.
    assert processes._native_binary(tmp_path).name != "hawking"
