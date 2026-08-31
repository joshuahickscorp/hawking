"""Memory traffic probe: enumerate, or refuse. Never invent a byte count.

The load-bearing refusal: asking for actual_read_bytes_per_token without a
counter that counted it raises. Substituting the catalog floor raises. A
receipt that carries a number with byte_counter_available false raises.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import memory_traffic_probe as mtp
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


@pytest.fixture(scope="module")
def receipt_doc():
    path = mtp.build()
    assert path.parent == RECEIPTS
    assert path.name == mtp.RECEIPT
    return json.loads(path.read_text())


def test_build_emits_sealed_receipt(receipt_doc):
    assert receipt_doc["schema"] == mtp.SCHEMA
    assert receipt_doc["version"] == mtp.VERSION
    assert receipt_doc["seal_sha256"]
    assert receipt_doc["bench"]["state"] == "UNKNOWN"
    assert receipt_doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert receipt_doc["bench"]["gpu_authority"] is False
    assert receipt_doc["gpu_authority"] is False
    assert receipt_doc["is_a_measurement"] is False
    assert receipt_doc["evidence_class"] == "STATIC_ONLY"
    _assert_no_hardware_claims(receipt_doc)


def test_actual_read_is_explicit_unknown_not_a_number(receipt_doc):
    value = receipt_doc["actual_read_bytes_per_token"]
    assert value == mtp.UNKNOWN
    assert value == "UNKNOWN"
    assert not isinstance(value, (int, float))
    assert value != mtp.CATALOG_WEIGHT_BYTES_PER_TOKEN
    assert receipt_doc["catalog_weight_bytes_per_token"] == mtp.CATALOG_WEIGHT_BYTES_PER_TOKEN
    assert receipt_doc["catalog_is_not_traffic"] is True
    assert receipt_doc["byte_counter_available"] is False
    assert receipt_doc["status"] == mtp.STATUS_NO_COUNTER


def test_module_raises_rather_than_emitting_unmeasured_number(receipt_doc):
    """NEGATIVE CONTROL: the only door that returns a number must refuse."""
    observed = {
        "counter_set_names": ["timestamp"],
        "counter_sets": [{"name": "timestamp", "counter_names": ["GPUTimestamp"]}],
        "n_counter_sets": 1,
        "supports_counter_sampling": {"AtStageBoundary": True, "AtDispatchBoundary": False},
        "gpu_raw_counter": {"dlopen": "ok", "n_groups": 0},
        "iokit": {"agxaccelerator_g15x_entries": [{"PerformanceStatistics_keys": [
            "Alloc system memory", "In use system memory", "Device Utilization %",
        ]}]},
    }
    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED") as caught:
        mtp.emit_actual_read_bytes(mtp.CATALOG_WEIGHT_BYTES_PER_TOKEN)
    assert str(mtp.CATALOG_WEIGHT_BYTES_PER_TOKEN) in str(caught.value)
    assert "catalog" in str(caught.value).lower() or "accounting" in str(caught.value).lower()

    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED"):
        mtp.emit_actual_read_bytes(0, counted_by=None)

    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED"):
        mtp.emit_actual_read_bytes(1, counted_by="timestamp")

    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED"):
        mtp.emit_actual_read_bytes(
            mtp.CATALOG_WEIGHT_BYTES_PER_TOKEN,
            counted_by="catalog",
            observed=observed,
        )

    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED"):
        mtp.measured_read_bytes_per_token(observed, catalog=mtp.CATALOG_WEIGHT_BYTES_PER_TOKEN)

    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED"):
        mtp.measured_read_bytes_per_token(observed, env={mtp.ENV_FLAG: "1"})


def test_emit_accepts_a_number_only_when_a_named_traffic_counter_counted_it():
    observed = {
        "counter_set_names": ["timestamp"],
        "counter_sets": [{
            "name": "timestamp",
            "counter_names": ["GPUTimestamp", "GPU Memory Read Bytes"],
        }],
        "n_counter_sets": 1,
    }
    assert mtp.emit_actual_read_bytes(
        42, counted_by="GPU Memory Read Bytes", observed=observed
    ) == 42
    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED"):
        mtp.emit_actual_read_bytes(42, counted_by="GPUTimestamp", observed=observed)
    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED"):
        mtp.emit_actual_read_bytes(
            mtp.CATALOG_WEIGHT_BYTES_PER_TOKEN,
            counted_by="GPU Memory Read Bytes",
            observed=observed,
        )


def test_validate_receipt_refuses_a_number_without_a_counter():
    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED"):
        mtp.validate_receipt({
            "actual_read_bytes_per_token": mtp.CATALOG_WEIGHT_BYTES_PER_TOKEN,
            "byte_counter_available": False,
        })
    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED"):
        mtp.validate_receipt({
            "actual_read_bytes_per_token": 1,
            "byte_counter_available": False,
        })
    mtp.validate_receipt({
        "actual_read_bytes_per_token": mtp.UNKNOWN,
        "byte_counter_available": False,
    })


def test_env_flag_does_not_license_an_estimate(receipt_doc):
    assert receipt_doc["env_flag"] == mtp.ENV_FLAG
    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED"):
        mtp.measured_read_bytes_per_token(env={mtp.ENV_FLAG: "1"})
    with pytest.raises(mtp.UnmeasuredMemoryTraffic, match="REFUSED"):
        mtp.measured_read_bytes_per_token(env={})


def test_timestamp_is_not_classified_as_traffic():
    assert mtp.name_looks_like_traffic("GPUTimestamp") is False
    assert mtp.name_looks_like_traffic("timestamp") is False
    assert mtp.name_looks_like_traffic("Alloc system memory") is False
    assert mtp.name_looks_like_traffic("In use system memory") is False
    assert mtp.name_looks_like_traffic("TiledSceneBytes") is False
    assert mtp.name_looks_like_traffic("Device Utilization %") is False
    assert mtp.name_looks_like_traffic("currentAllocatedSize") is False
    assert mtp.name_looks_like_traffic("is_memory_traffic") is False
    assert mtp.name_looks_like_traffic("GPU Memory Read Bytes") is True
    assert mtp.name_looks_like_traffic("dram_read_bytes") is True


def test_does_not_assume_common_counter_sets_are_present(receipt_doc):
    src = mtp.PROBE_SOURCE.read_text()
    assert "counterSets" in src
    assert "MTLCommonCounterSetStageUtilization" in src
    assert "MTLCommonCounterSetStatistic" in src
    present = {_norm for _norm in (receipt_doc["counter_set_names_present"] or [])}
    absent = set(receipt_doc["common_counter_sets_absent_on_device"] or [])
    for name in mtp.COMMON_COUNTER_SETS:
        assert (name in present) ^ (name in absent), name
    # This device: do not treat SDK constants as present sets.
    constants = (receipt_doc["surfaces"]["MTLCommonCounterSet_constants"] or {})
    assert "timestamp" in {v.lower() for v in constants.values()}
    assert "stageutilization" in {v.lower() for v in constants.values()}
    assert "statistic" in {v.lower() for v in constants.values()}
    if "timestamp" in present:
        assert "stageutilization" in absent
        assert "statistic" in absent


def test_enumerated_counter_sets_are_the_real_list(receipt_doc):
    present = receipt_doc["counter_set_names_present"]
    assert isinstance(present, list)
    assert present == ["timestamp"]
    detail = receipt_doc["surfaces"]["MTLCounterSet"]["detail"]
    assert detail[0]["name"] == "timestamp"
    assert "GPUTimestamp" in detail[0]["counters"]
    sampling = receipt_doc["surfaces"]["supports_counter_sampling"]
    assert sampling["AtStageBoundary"] is True
    assert sampling["AtDispatchBoundary"] is False
    assert receipt_doc["surfaces"]["timestamp_counters"]["reports_memory_traffic"] is False
    assert receipt_doc["surfaces"]["stage_utilization"]["set_present"] is False
    assert receipt_doc["surfaces"]["statistic"]["set_present"] is False


def test_gpraw_and_iokit_are_recorded_as_not_traffic(receipt_doc):
    grc = receipt_doc["surfaces"]["GPURawCounter"]
    assert grc["reports_memory_traffic"] is False
    assert grc["dlopen"] == "ok"
    assert grc["n_groups"] == 0
    assert "AGXGPURawCounterSourceGroup" in (grc.get("copy_error") or {}).get("description", "")
    iokit = receipt_doc["surfaces"]["IOKit.IOAccelerator"]
    assert iokit["reports_memory_traffic"] is False
    keys = iokit["PerformanceStatistics_keys"]
    assert "Alloc system memory" in keys
    assert "In use system memory" in keys
    assert "Device Utilization %" in keys
    assert not any(mtp.name_looks_like_traffic(k) for k in keys)
    report = iokit["IOReport_channel_names"]
    assert isinstance(report, list) and len(report) > 0
    assert not any(mtp.name_looks_like_traffic(n) for n in report)
    alloc = receipt_doc["surfaces"]["MTLDevice.currentAllocatedSize"]
    assert alloc["reports_memory_traffic"] is False
    assert alloc["delta"] == 16 * 1024 * 1024
    res = receipt_doc["surfaces"]["MTLResidencySet"]
    assert res["reports_memory_traffic"] is False
    assert receipt_doc["surfaces"]["MTL4CounterHeap"]["memory_traffic_heap_type_in_public_api"] is False


def test_stage_boundary_timestamps_are_time_not_bytes(receipt_doc):
    samples = receipt_doc["stage_boundary_samples"]
    assert samples, "stage-boundary sampling is supported and must be exercised"
    deltas = [s.get("timestamp_delta_ns") for s in samples if s.get("timestamp_delta_ns")]
    assert deltas and all(d > 0 for d in deltas)
    dispatch = receipt_doc["dispatch_boundary_samples"]
    for row in dispatch:
        assert row.get("resolved_u64") == [0, 0]
    # A time delta must not leak into actual_read_bytes_per_token.
    assert receipt_doc["actual_read_bytes_per_token"] == mtp.UNKNOWN


def test_module_never_divides_catalog_into_a_byte_rate():
    src = Path(mtp.__file__).read_text()
    compact = src.replace(" ", "")
    assert "CATALOG_WEIGHT_BYTES_PER_TOKEN/" not in compact
    assert "703.5" not in src
    assert "357.4" not in src


def test_probe_source_queries_the_device_instead_of_hardcoding_the_set_list():
    src = mtp.PROBE_SOURCE.read_text()
    assert "device.counterSets" in src
    assert "newCounterSampleBufferWithDescriptor" in src
    assert "GRCCopyAllCounterSourceGroupWithError" in src
    assert "currentAllocatedSize" in src
    assert "IOAccelerator" in src
    assert "AGXAcceleratorG15X" in src
    # Must not claim the statistic set is present without asking.
    assert "counter_set_names = @[ @\"timestamp\", @\"statistic\"" not in src


def test_selftest_aliases_build():
    assert mtp.selftest is mtp.build


def test_main_measure_refuses_without_emitting_a_number():
    rc = mtp.main(["--measure"])
    assert rc == 2
