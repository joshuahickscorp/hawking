from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "llama_q4k_gravity_pack", ROOT / "tools" / "llama_q4k_gravity_pack.py"
)
assert SPEC is not None and SPEC.loader is not None
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


def tensor(dtype: int, *, rows: int, cols: int, byte_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        name="fixture.weight",
        tensor_type=dtype,
        data=SimpleNamespace(nbytes=byte_count),
    )


@pytest.mark.parametrize(
    ("dtype", "codec", "block_bytes", "block_elements"),
    [
        (PACK.Q4_K, "ggml.q4_k", 144, 256),
        (PACK.Q5_0, "ggml.q5_0", 22, 32),
        (PACK.Q6_K, "ggml.q6_k", 210, 256),
        (PACK.Q8_0, "ggml.q8_0", 34, 32),
    ],
)
def test_raw_quant_grammars_preserve_exact_block_geometry(
    dtype: int, codec: str, block_bytes: int, block_elements: int
) -> None:
    rows, cols = 3, block_elements * 4
    source = tensor(dtype, rows=rows, cols=cols, byte_count=rows * 4 * block_bytes)
    assert PACK.codec_and_geometry(source, [rows, cols]) == (codec, block_bytes)


def test_raw_quant_geometry_refuses_truncated_or_misaligned_payload() -> None:
    source = tensor(PACK.Q8_0, rows=1, cols=33, byte_count=34)
    with pytest.raises(ValueError, match="cols%32"):
        PACK.codec_and_geometry(source, [1, 33])

    source = tensor(PACK.Q5_0, rows=2, cols=64, byte_count=43)
    with pytest.raises(ValueError, match="raw bytes"):
        PACK.codec_and_geometry(source, [2, 64])
