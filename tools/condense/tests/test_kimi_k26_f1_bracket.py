from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402


def test_unsigned_bitpack_roundtrip_for_non_byte_width() -> None:
    values = np.array([0, 1, 511, 256, 17, 9], dtype=np.uint32)
    packed = f1.pack_unsigned(values, 9)
    np.testing.assert_array_equal(f1.unpack_unsigned(packed, len(values), 9), values)
    assert len(packed) == (len(values) * 9 + 7) // 8


def test_distribute_preserves_exact_budget() -> None:
    result = f1.distribute(101, [7, 7, 7])
    assert sum(result) == 101
    assert max(result) - min(result) <= 1


def test_fidelity_verdict_is_monotonic() -> None:
    survives = {"cosine_mean": 0.95, "cosine_p10": 0.90, "relative_l2": 0.30}
    degraded = {"cosine_mean": 0.80, "cosine_p10": 0.60, "relative_l2": 0.70}
    collapse = {"cosine_mean": 0.40, "cosine_p10": 0.10, "relative_l2": 1.10}
    assert f1.fidelity_verdict(survives) == "SURVIVES_F1"
    assert f1.fidelity_verdict(degraded) == "DEGRADED_F1"
    assert f1.fidelity_verdict(collapse) == "COLLAPSE_F1"


def test_quality_identical_is_perfect() -> None:
    rng = np.random.default_rng(3)
    value = rng.standard_normal((8, 16), dtype=np.float32)
    metric = f1.quality(value, value.copy())
    assert metric["cosine_mean"] > 0.999999
    assert metric["relative_l2"] == 0


def test_physical_payload_read_checks_every_component(tmp_path: Path) -> None:
    path = tmp_path / "physical.k26f1"
    components = [
        {"name": "base", "role": "base", "encoding": "raw", "data": b"abc"},
        {"name": "doctor", "role": "doctor", "encoding": "raw", "data": b"xyz"},
    ]
    written = f1.write_payload(path, {"sentinel_expert": 0}, components)
    header, decoded = f1.read_payload(path)
    assert header["sentinel_expert"] == 0
    assert [component["data"] for component in decoded] == [b"abc", b"xyz"]
    assert written["bytes"] == path.stat().st_size
