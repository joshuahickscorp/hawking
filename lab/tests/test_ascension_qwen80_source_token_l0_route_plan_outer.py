from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lab.operators import ascension_qwen80_source_token_l0_route_plan_outer as outer


def test_child_command_is_cpu_material_only_and_carries_each_exact_authority() -> None:
    args = SimpleNamespace(
        probe_bin=Path("/tmp/ascension_qwen80_source_token_l0_router_discriminator"),
        manifest=Path("/tmp/manifest.json"),
        admission_current=Path("/tmp/admission-current.json"),
        first_residual_receipt=Path("/tmp/prefix.json"),
        old_route_plan=Path("/tmp/fixture-plan.json"),
    )
    command = outer._command(args, Path("/tmp/capture/inner/source-token-route-material.json"))
    assert command[0].endswith(outer.EXPECTED_PROBE)
    assert "--first-residual-receipt" in command
    assert "--old-route-plan" in command
    assert "--out" in command
    assert not any("lease" in item.lower() or "metal" in item.lower() for item in command)


def test_outer_uses_immutable_receipt_last_filenames() -> None:
    assert outer.MATERIAL_FILENAME.endswith(".json")
    assert outer.AUTHORITY_FILENAME.endswith(".json")
    assert outer.TERMINAL_FILENAME == "outer-terminal-receipt.json"
