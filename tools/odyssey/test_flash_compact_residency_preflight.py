from __future__ import annotations

from tools.odyssey.flash_compact_residency_preflight import assess


def _session(source_bytes: int = 30 * 1024**3) -> dict:
    return {
        "model": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
        "execution": {
            "expert_bank_mode": "route_union_compact_teacher_bound",
            "source_payload_bytes_read": source_bytes,
            "state_memory": {"total_persistent_bytes": 128 * 1024**2},
        },
    }


def test_admits_only_when_conservative_double_buffer_budget_fits():
    result = assess(
        _session(), physical_bytes=96 * 1024**3, free_percent=88,
        swap_used_mb=8.0, headroom_bytes=8 * 1024**3,
    )
    assert result["status"] == "ADMISSIBLE_CONSERVATIVE_COMPACT_RESIDENT_EXPERIMENT"
    assert result["budget"]["conservative_host_plus_metal_weight_bytes"] == 60 * 1024**3


def test_withholds_when_swap_or_headroom_make_the_experiment_unsafe():
    result = assess(
        _session(), physical_bytes=96 * 1024**3, free_percent=60,
        swap_used_mb=256.0, headroom_bytes=8 * 1024**3,
    )
    assert result["status"] == "WITHHELD_INSUFFICIENT_HEADROOM"
