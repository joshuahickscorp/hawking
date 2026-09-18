"""Focused regression for P20_FINAL_HANDOFF_REVIEW_V2 remote-id recognition.

No-release, no-hardware-qualification boundary: this test only exercises the
pure ``_is_remote_openrouter_model`` predicate in ``hawking.goal_surface``.
It performs no spool mutation, no release, and no hardware qualification.
"""
from __future__ import annotations

import pytest

from hawking.goal_surface import _is_remote_openrouter_model


@pytest.mark.parametrize(
    "value",
    [
        "openrouter:anthropic/claude-3.5-sonnet",
        "openrouter:meta-llama/llama-3.1-70b-instruct",
        "anthropic/claude-3.5-sonnet",
        "meta-llama/llama-3.1-70b-instruct",
    ],
)
def test_remote_openrouter_ids_are_recognized(value):
    assert _is_remote_openrouter_model(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
        "/abs/path/to/model",
        "./relative/model",
        "../relative/model",
        "weights/model.safetensors",
        "weights/model.gguf",
        "weights/model.json",
        "weights/model.onnx",
        "weights/model.bin",
        "weights/model.pt",
        "weights/model.pth",
        "weights/model.npz",
        "C:\\\\models\\\\local",
        "models\\\\local\\\\weights",
        "https://example.com/model",
        "a/b/c",
        "provider/",
        "/model",
        "provider /model",
        "provider/ model",
    ],
)
def test_local_paths_and_artifacts_are_not_remote_ids(value):
    assert _is_remote_openrouter_model(value) is False


def test_single_slash_provider_model_is_remote():
    assert _is_remote_openrouter_model("openai/gpt-4o") is True
    assert _is_remote_openrouter_model("openai/gpt-4o") is True