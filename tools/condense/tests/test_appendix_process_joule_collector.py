from __future__ import annotations

import copy
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
TESTS = CONDENSE / "tests"
sys.path.insert(0, str(CONDENSE))
sys.path.insert(0, str(TESTS))

import appendix_process_joule_collector as process_joule  # noqa: E402
from test_appendix_physical_counter_collector import _bundle  # noqa: E402
from test_appendix_physical_counter_normalizer import _capture  # noqa: E402


def test_libproc_contract_pins_layout_headers_library_and_direct_units() -> None:
    value = process_joule.contract()
    assert value["contract_sha256"] == process_joule.CONTRACT_SHA256
    assert value["flavor"] == 6
    assert value["energy_field"] == "ri_energy_nj"
    assert value["energy_input_unit"] == "nanojoule"
    assert value["energy_output_unit"] == "joule"
    assert value["powermetrics_energy_impact_accepted"] is False
    assert value["estimated_or_apportioned_values_accepted"] is False
    provenance = process_joule.library_provenance()
    assert provenance["library_install_name"] == "/usr/lib/libproc.dylib"
    assert len(provenance["dyld_shared_cache_uuid"]) == 32
    assert provenance["collector_contract_sha256"] == process_joule.CONTRACT_SHA256


def test_probe_counter_block_reconstructs_every_self_sampled_phase() -> None:
    bundle = _bundle("device")
    value = _capture(bundle, "device", "energy")
    assert process_joule.probe_counter_block_errors(value, expected_pid=1234) == []
    assert len(value["records"]) == 2
    for record in value["records"]:
        assert record["self_sampled_by_release_probe"] is True
        assert record["energy_nj_delta"] > 0
        assert record["energy_j"] == record["energy_nj_delta"] / 1_000_000_000


@pytest.mark.parametrize("field", ["ri_energy_nj", "ri_penergy_nj", "ri_instructions", "ri_cycles"])
def test_monotone_counter_decrease_or_wrap_is_rejected(field: str) -> None:
    bundle = _bundle("device")
    record = copy.deepcopy(_capture(bundle, "device", "energy")["records"][0])
    record["before"]["counters"][field] = 10
    record["after"]["counters"][field] = 9
    record["before"] = process_joule._stamp(record["before"], "snapshot_sha256")
    record["after"] = process_joule._stamp(record["after"], "snapshot_sha256")
    errors = process_joule.phase_record_errors(record, expected_pid=1234)
    assert any("wrapped or decreased" in error for error in errors)


@pytest.mark.parametrize("mutation", ["uuid", "start", "pid"])
def test_pid_reuse_or_process_identity_change_is_rejected(mutation: str) -> None:
    bundle = _bundle("device")
    record = copy.deepcopy(_capture(bundle, "device", "energy")["records"][0])
    if mutation == "uuid":
        record["after"]["ri_uuid"] = "cd" * 16
    elif mutation == "start":
        record["after"]["counters"]["ri_proc_start_abstime"] += 1
    else:
        record["after"]["pid"] += 1
    record["after"] = process_joule._stamp(record["after"], "snapshot_sha256")
    errors = process_joule.phase_record_errors(record, expected_pid=1234)
    assert any("identity changed" in error for error in errors)


def test_counter_reads_that_do_not_bracket_exact_operation_are_rejected() -> None:
    bundle = _bundle("device")
    record = copy.deepcopy(_capture(bundle, "device", "energy")["records"][0])
    record["before"]["read_ended_at_continuous_ns"] = (
        record["interval_started_at_continuous_ns"] + 1
    )
    record["before"] = process_joule._stamp(record["before"], "snapshot_sha256")
    errors = process_joule.phase_record_errors(record, expected_pid=1234)
    assert any("do not bracket" in error for error in errors)


def test_duplicate_phase_record_and_forged_library_provenance_fail() -> None:
    bundle = _bundle("device")
    value = _capture(bundle, "device", "energy")
    with pytest.raises(process_joule.ProcessJouleError, match="reused"):
        process_joule.build_probe_counter_block(
            records=[value["records"][0], value["records"][0]],
            library=value["library_provenance"], probe_pid=1234,
        )
    forged = copy.deepcopy(value["library_provenance"])
    forged["dyld_shared_cache_uuid"] = "f" * 32
    with pytest.raises(process_joule.ProcessJouleError, match="provenance hash"):
        process_joule.build_probe_counter_block(
            records=value["records"], library=forged, probe_pid=1234,
        )
