from __future__ import annotations

from tools.future.kimi_ebpw_organ_screen import _screen_one, _tensor_vectors


def test_rowwise_pq_vector_count_is_shape_exact() -> None:
    assert _tensor_vectors([3, 64], 16) == (12, 4)
    assert _tensor_vectors([3, 65], 32) == (9, 3)


def test_organ_screen_bills_support_and_codebooks_without_loading_weights() -> None:
    rows = [
        ("synthetic.down_proj.weight", [4, 64], 512),
        ("synthetic.up_proj.weight", [64, 4], 512),
    ]
    result = _screen_one("fixture", rows)
    assert result["parent_params"] == 512
    assert result["families"]["pq_d16_shared"]["capability_status"] == "NOT_MEASURED"
    pq = result["families"]["pq_d16_shared"]["complete_accounting"]
    assert pq["reconciled"] is True
    assert pq["complete_ebpw"] > 0
    assert any(part["category"] == "tables" for part in pq["parts"])
    assert any(part["category"] == "runtime_auxiliaries" for part in pq["parts"])
    assert result["families"]["pq_d16_shared"]["direct_execution_status"] == "NOT_VALIDATED"
