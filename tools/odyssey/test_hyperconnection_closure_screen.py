from __future__ import annotations

import numpy as np
import pytest

from tools.odyssey.hyperconnection_closure_screen import screen_stream_closure


def test_flash_four_stream_hc_is_full_rank() -> None:
    result = screen_stream_closure(streams=4, gate_probe=1.0)
    assert result["status"] == "SYMBOLIC_FULL_JOINT_RANK__QUOTIENT_REJECTED"
    assert result["rank_solution"]["read_mix_rank"] == 4
    assert result["rank_solution"]["joint_observable_rank"] == 4
    assert result["rank_solution"]["joint_observable_nullity"] == 0
    assert result["rank_solution"]["nontrivial_r_less_than_m_possible"] is False
    assert np.isfinite(result["rank_solution"]["read_mix_determinant"])


def test_closure_screen_supports_other_stream_widths() -> None:
    result = screen_stream_closure(streams=7, gate_probe=2.0)
    assert result["rank_solution"]["joint_observable_rank"] == 7
    assert len(result["consumer_system"]["read_mix_probe_matrix"]) == 7


def test_closure_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        screen_stream_closure(streams=1)
    with pytest.raises(ValueError, match="finite and non-zero"):
        screen_stream_closure(streams=4, gate_probe=0.0)
