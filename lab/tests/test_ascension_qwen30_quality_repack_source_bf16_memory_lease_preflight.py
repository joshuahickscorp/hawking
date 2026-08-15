"""Pure tests for Q30 source-BF16 memory preflight arithmetic."""
from __future__ import annotations

from lab.operators import ascension_qwen30_quality_repack_source_bf16_memory_lease_preflight as preflight


def _snapshot(reclaimable: int, swap: int = 0) -> dict[str, object]:
    return {
        "physical_memory_bytes": 96 * 1024**3,
        "vm_stat": {"reclaimable_bytes": reclaimable},
        "swap": {"used_bytes": swap},
    }


def test_headroom_refuses_insufficient_reclaimable_memory() -> None:
    result = preflight.assess_headroom(_snapshot(60 * 1024**3), source_weight_bytes=61 * 1024**3)
    assert result["status"] == preflight.BLOCKED_STATUS
    assert result["lease_granted"] is False
    assert result["measured_reclaimable_deficit_bytes"] > 0


def test_headroom_refuses_any_existing_swap() -> None:
    result = preflight.assess_headroom(_snapshot(80 * 1024**3, swap=1), source_weight_bytes=61 * 1024**3)
    assert result["status"] == preflight.BLOCKED_STATUS


def test_headroom_is_only_prepared_not_a_lease_when_floor_is_met() -> None:
    result = preflight.assess_headroom(_snapshot(80 * 1024**3), source_weight_bytes=61 * 1024**3)
    assert result["status"] == preflight.READY_STATUS
    assert result["lease_granted"] is False


def test_vm_stat_accepts_macos_occupied_by_compressor_spelling() -> None:
    result = preflight._parse_vm_stat(
        "\n".join(
            [
                "Pages free: 10.",
                "Pages inactive: 20.",
                "Pages speculative: 3.",
                "Pages active: 30.",
                "Pages wired down: 4.",
                "Pages occupied by compressor: 5.",
                "Swapouts: 0.",
            ]
        ),
        page_size=16_384,
    )
    assert result["compressor_pages"] == 5
    assert result["reclaimable_bytes"] == 33 * 16_384
