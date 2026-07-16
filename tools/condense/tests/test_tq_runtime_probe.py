from __future__ import annotations

import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools" / "condense" / "tq_runtime_probe.py"
SPEC = importlib.util.spec_from_file_location("tq_runtime_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_probe_is_deterministic_complete_and_nonexecuting() -> None:
    probe = MODULE.build_probe()
    assert probe == MODULE.build_probe()
    assert probe["reads_model_artifacts"] is False
    assert probe["safe_during_active_run"] is True
    assert len(probe["cells"]) == len(MODULE.DEFAULT_SHAPES) * 4 * len(MODULE.MODES)


def test_known_tq3_metadata_arithmetic() -> None:
    stored = MODULE.probe_cell(1, 256, 3, 9, "stored", "unit")
    compact = MODULE.probe_cell(1, 256, 3, 9, "compact", "unit")
    assert stored["bpw"]["payload_plus_metadata"] == 5.625
    assert compact["bpw"]["payload_plus_metadata"] == 4.25
    assert stored["logical_bytes"]["metadata"] == 84
    assert compact["logical_bytes"]["metadata"] == 40


def test_probe_exposes_ragged_gpu_ineligibility_and_future_modes() -> None:
    ragged = MODULE.probe_cell(4864, 896, 3, 9, "stored", "qwen_up", model="qwen", multiplicity=48)
    assert ragged["current_gpu_fused_gemv_eligible"] is False
    assert ragged["ragged_tail_weights_per_row"] == 128
    assert MODULE.MODES["compact_hashed"]["implemented"] is False
    assert MODULE.MODES["compact_computed"]["implemented"] is False
    assert MODULE.MODES["repacked_lut"]["implemented"] is False


def test_model_rollups_charge_projection_multiplicity() -> None:
    probe = MODULE.build_probe()
    row = next(
        item for item in probe["model_rollups"]
        if item["model"] == "qwen_0_5b" and item["k_bits"] == 3 and item["mode"] == "stored"
    )
    assert row["projection_family_count"] == 5
    assert row["multiplicity_weighted_weights"] > 0
    assert row["all_projection_families_gpu_eligible"] is False
    assert any("ffn_up_gate" in label for label in row["ineligible_projection_families"])
