import numpy as np

from tools.odyssey.layerwise_sparse_repair import binary_reconstruct, packed_bytes, repaired, screen, ternary_reconstruct


def test_ternary_and_sparse_repair_have_monotonic_error_and_billed_indices():
    weight = np.array([[1.0, 0.2, -0.9, 0.0], [0.1, -0.8, 0.0, 0.5]], dtype=np.float32)
    base = ternary_reconstruct(weight)
    assert base.shape == weight.shape
    assert binary_reconstruct(weight).shape == weight.shape
    repaired_weight, k = repaired(weight, base, 0.25)
    assert k == 2
    rows = screen(weight, [0.0, 0.25, 1.0])
    assert rows[0]["relative_l2"] >= rows[1]["relative_l2"] >= rows[2]["relative_l2"]
    assert rows[1]["complete_bytes"] > rows[0]["complete_bytes"]
    assert packed_bytes(5, 2) == 2
