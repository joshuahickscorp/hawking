from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.odyssey import flash_source_boundary_mlx_worker as worker


class _FakeMx:
    float32 = np.float32

    def __init__(self) -> None:
        self.zeros_calls: list[tuple[tuple[int, ...], object]] = []

    def zeros(self, shape: tuple[int, ...], *, dtype: object) -> np.ndarray:
        self.zeros_calls.append((shape, dtype))
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def eval(_value: object) -> None:
        return None


def _linear_attention() -> SimpleNamespace:
    return SimpleNamespace(
        conv_kernel_size=2,
        conv_dim=3,
        num_v_heads=2,
        head_v_dim=3,
        head_k_dim=4,
    )


def test_empty_provider_cache_is_captured_as_the_source_defined_zero_state(tmp_path) -> None:
    mx = _FakeMx()
    cache = SimpleNamespace(state=[None, None])
    attention_input = np.zeros((1, 1, 5), dtype=np.float16)

    records = worker._capture_layer4_linear_cache(
        tmp_path,
        mx,
        np,
        cache_entry=cache,
        boundary="pre-attention",
        attention_input=attention_input,
        linear_attention=_linear_attention(),
    )

    assert cache.state == [None, None]
    assert mx.zeros_calls == [
        ((1, 1, 3), np.dtype("float16")),
        ((1, 2, 3, 4), np.float32),
    ]
    assert [record["state_role"] for record in records] == ["conv_state", "recurrent_state"]
    assert [record["shape"] for record in records] == [[1, 1, 3], [1, 2, 3, 4]]
    assert all(record["initialization"] == worker._IMPLICIT_ZERO_INITIALIZATION for record in records)
    assert all(record["payload"]["path"].startswith(str(tmp_path)) for record in records)


def test_layer4_cache_rejects_partially_materialized_provider_state(tmp_path) -> None:
    mx = _FakeMx()
    cache = SimpleNamespace(state=[None, np.zeros((1, 2, 3, 4), dtype=np.float32)])

    with pytest.raises(ValueError, match="partially materialized"):
        worker._capture_layer4_linear_cache(
            tmp_path,
            mx,
            np,
            cache_entry=cache,
            boundary="pre-attention",
            attention_input=np.zeros((1, 1, 5), dtype=np.float16),
            linear_attention=_linear_attention(),
        )


def test_capture_array_records_the_source_dtype_and_keeps_trace_files_separate(tmp_path) -> None:
    mx = _FakeMx()
    record = worker._capture_array(
        tmp_path,
        mx,
        np,
        ordinal=0,
        layer=0,
        stage="attention_hyper_connection_mixed",
        value=np.array([[[1.0, 2.0]]], dtype=np.float16),
        file_prefix="layer0-trace",
    )

    assert record["original_dtype"] == "float16"
    assert record["shape"] == [1, 1, 2]
    assert (tmp_path / "layer0-trace-00-attention_hyper_connection_mixed.f32").is_file()
