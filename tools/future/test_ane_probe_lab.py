"""Tests for the parameterized ANE probe lab.

The load-bearing guards: authoring REFUSES when coremltools is absent rather
than simulating a result, and an inferred placement cannot be recorded as an
observation. Tests must pass whether or not coremltools is installed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tools.future import ane_preboard as preboard
from tools.future import ane_probe_lab as lab
from tools.future._common import RECEIPTS


def test_build_writes_parseable_sealed_receipt():
    out = lab.build()
    assert out.parent == RECEIPTS
    assert out.name == "ANE_PROBE_LAB.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.future.ane_probe_lab.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["does_not_choose_the_experiment"] is True
    assert doc["extends"] == "tools/future/ane_preboard.py"
    assert set(doc["runnable_now"]) == set(lab.MODES)


def test_receipt_states_four_modes_runnability_measured_not_assumed():
    doc = json.loads(lab.build().read_text())
    live = lab.mode_runnability(
        devices=doc["compute_devices"],
        coremltools=doc["coremltools"],
    )
    for mode in lab.MODES:
        assert mode in doc["runnable_now"]
        assert isinstance(doc["runnable_now"][mode], bool)
        row = doc["mode_runnability"][mode]
        assert row["runnable_now"] is doc["runnable_now"][mode]
        assert row["runnable_now"] is live[mode]["runnable_now"]
        assert row["assumed"] is False
        assert row["measured_by"]
        assert "coremltools" in " ".join(row["measured_by"]).lower()
    assert all(token in doc["runnable_now_plain"] for token in lab.MODES)
    st = lab.coremltools_status()
    if not st["present"]:
        assert doc["runnable_now"] == {
            "GPU_ONLY": False,
            "ANE_ONLY": False,
            "SERIAL": False,
            "CONCURRENT": False,
        }
        blob = json.dumps(doc["mode_runnability"]).lower()
        assert "coremltools" in blob


def test_materialize_refuses_rather_than_simulating_when_coremltools_absent(monkeypatch):
    monkeypatch.setattr(
        lab,
        "coremltools_status",
        lambda: {
            "present": False,
            "version": None,
            "error": "ModuleNotFoundError: No module named 'coremltools'",
        },
    )
    with pytest.raises(lab.CoremltoolsUnavailableError, match="coremltools") as caught:
        lab.materialize_graph(op="add", shape=[1, 4], dtype="fp16", structure="elementwise")
    msg = str(caught.value).lower()
    assert "simulat" in msg
    assert "refusing" in msg
    with pytest.raises(lab.CoremltoolsUnavailableError, match="coremltools"):
        lab.run_graph(
            op="add",
            shape=[1, 4],
            dtype="fp16",
            structure="elementwise",
            mode="SERIAL",
            repeats=1,
        )


def test_live_coremltools_absence_is_a_real_import_check():
    st = lab.coremltools_status()
    spec = importlib.util.find_spec("coremltools")
    assert st["present"] is (spec is not None)
    if not st["present"]:
        with pytest.raises(lab.CoremltoolsUnavailableError, match="coremltools"):
            lab.materialize_graph(
                op="add", shape=[1, 4], dtype="fp16", structure="elementwise"
            )
        for name, thunk in lab.execution_entry_points():
            with pytest.raises(lab.ProbeLabRefused):
                thunk()
            assert name  # names stay bound so a silent pass cannot drop the loop


def test_inferred_placement_cannot_be_recorded_as_observation():
    with pytest.raises(lab.EvidencePromotionError, match="cannot be recorded"):
        lab.record_placement(
            preferred="NEURAL_ENGINE",
            supported=["NEURAL_ENGINE"],
            evidence_class="PUBLIC_API_OBSERVED",
            source="wall time was lower than GPU_ONLY so this ran on ANE",
            source_kind="TIMING_DELTA",
            is_live=False,
            api=None,
        )
    with pytest.raises(lab.EvidencePromotionError, match="inference"):
        lab.record_placement(
            preferred="NEURAL_ENGINE",
            supported=["CPU", "NEURAL_ENGINE"],
            evidence_class="PUBLIC_API_OBSERVED",
            source="faster than GPU_ONLY",
            source_kind="TIMING_DELTA",
            is_live=True,
            api="MLComputePlan.computeDeviceUsage(for:)",
        )
    inferred = lab.record_placement(
        preferred="NEURAL_ENGINE",
        supported=["CPU", "NEURAL_ENGINE"],
        evidence_class="PLACEMENT_INFERRED",
        source="faster than GPU_ONLY",
        source_kind="TIMING_DELTA",
    )
    assert inferred["is_observation"] is False
    assert inferred["is_inference"] is True
    assert inferred["evidence_class"] == "PLACEMENT_INFERRED"
    with pytest.raises(lab.EvidencePromotionError, match="cannot be promoted"):
        lab.as_observation(inferred)


def test_fixture_plan_cannot_become_a_live_observation():
    fixture = preboard.mlcomputeplan_fixture()
    assert fixture["is_fixture"] is True
    with pytest.raises(lab.EvidencePromotionError, match="fixture"):
        lab.placement_evidence_from_plan(fixture)
    parsed = preboard.parse_mlcompute_plan(fixture)
    assert parsed["is_live"] is False
    with pytest.raises(lab.EvidencePromotionError):
        lab.placement_evidence_from_plan(parsed)


def test_live_plan_records_as_public_api_observed():
    plan = {
        "api": "MLComputePlan.load(contentsOf:configuration:)",
        "model_structure": "MLProgram",
        "status": "PLANNED",
        "is_fixture": False,
        "is_live": True,
        "operations": [
            {
                "index": 0,
                "operator": "ios16.add",
                "preferred": "NEURAL_ENGINE",
                "supported": ["CPU", "NEURAL_ENGINE"],
                "placement_status": "PLANNED",
            }
        ],
    }
    rec = lab.placement_evidence_from_plan(plan)
    assert rec["evidence_class"] == "PUBLIC_API_OBSERVED"
    assert rec["is_observation"] is True
    assert rec["is_inference"] is False
    assert rec["preferred"] == "NEURAL_ENGINE"
    observed = lab.as_observation(rec)
    assert observed["preferred"] == "NEURAL_ENGINE"


def test_missing_graph_fields_refuse_rather_than_default():
    base = dict(op="add", shape=[1, 8], dtype="fp16", structure="elementwise")
    for drop in ("op", "shape", "dtype", "structure"):
        kwargs = dict(base)
        kwargs[drop] = None
        with pytest.raises(lab.GraphSpecRefused, match=drop):
            lab.validate_graph_spec(**kwargs)
    with pytest.raises(lab.GraphSpecRefused, match="dtype"):
        lab.validate_graph_spec(
            op="add", shape=[1, 8], dtype="int8", structure="elementwise"
        )
    with pytest.raises(lab.GraphSpecRefused, match="substituting"):
        lab.validate_graph_spec(
            op="conv", shape=[1, 8], dtype="fp16", structure="elementwise"
        )
    with pytest.raises(lab.GraphSpecRefused, match="rank"):
        lab.validate_graph_spec(op="matmul", shape=[8], dtype="fp16", structure="linear")


def test_unknown_mode_and_missing_modes_refuse():
    with pytest.raises(lab.ModeRefused, match="missing"):
        lab.documented_compute_units("")
    with pytest.raises(lab.ModeRefused, match="CPU_ONLY"):
        lab.documented_compute_units("CPU_ONLY")
    with pytest.raises(lab.ModeRefused, match="missing"):
        lab.run_matched_modes(
            op="add",
            shape=[1, 4],
            dtype="fp16",
            structure="elementwise",
            modes=None,
            repeats=1,
        )
    with pytest.raises(lab.GraphSpecRefused, match="repeats"):
        lab.run_graph(
            op="add",
            shape=[1, 4],
            dtype="fp16",
            structure="elementwise",
            mode="SERIAL",
            repeats=None,
        )


def test_gpu_only_maps_to_documented_cpu_and_gpu_not_exclusive_gpu():
    row = lab.documented_compute_units("GPU_ONLY")
    assert row["evidence_class"] == "PUBLICLY_DOCUMENTED"
    assert row["coremltools_compute_unit"] == "CPU_AND_GPU"
    assert row["exclusive_without_cpu"] is False
    assert row["is_placement"] is False
    ane = lab.documented_compute_units("ANE_ONLY")
    assert ane["coremltools_compute_unit"] == "CPU_AND_NE"
    assert ane["exclusive_without_cpu"] is False
    with pytest.raises(lab.EvidencePromotionError, match="not a graph placement"):
        lab.record_placement(
            preferred="GPU",
            supported=["GPU"],
            evidence_class="PUBLICLY_DOCUMENTED",
            source="Apple docs",
        )


def test_gpu_only_not_runnable_when_public_api_omits_gpu():
    rows = lab.mode_runnability(
        devices={
            "status": "OBSERVED",
            "kinds": ["CPU", "NEURAL_ENGINE"],
            "api": "MLModel.availableComputeDevices",
            "error": None,
        },
        coremltools={"present": True, "version": "9.0", "error": None},
    )
    assert rows["GPU_ONLY"]["runnable_now"] is False
    assert "GPU" in str(rows["GPU_ONLY"]["why"])
    assert rows["ANE_ONLY"]["runnable_now"] is True
    assert rows["SERIAL"]["runnable_now"] is True
    assert rows["CONCURRENT"]["runnable_now"] is True
    assert rows["SERIAL"]["kind"] == "SCHEDULE"
    assert rows["CONCURRENT"]["kind"] == "SCHEDULE"
    assert rows["GPU_ONLY"]["kind"] == "COMPUTE_UNITS"


def test_lab_does_not_encode_a_flash_or_qwen_campaign():
    src = Path(lab.__file__).read_text()
    for tok in ("flash.expert", "qwen27.mlp", "gate_up_swiglu", "17408", "248320"):
        assert tok not in src, tok
    assert not hasattr(lab, "graph_corpus")
    assert lab.MODES == ("GPU_ONLY", "ANE_ONLY", "SERIAL", "CONCURRENT")
    harness = lab.existing_harness()
    assert harness["path"] == "tools/future/ane_preboard.py"
    assert "Flash / Qwen27 experiment recipe" in harness["this_lab_does_not_import"]


def test_evidence_class_set_is_closed_and_parser_is_reused():
    assert lab.EVIDENCE_CLASSES == (
        "PUBLICLY_DOCUMENTED",
        "PUBLIC_API_OBSERVED",
        "PLACEMENT_INFERRED",
        "TIMING_INFERRED",
        "PHYSICAL_BEHAVIOR_INFERRED",
    )
    assert lab.OBSERVATION_CLASSES.isdisjoint(lab.INFERENCE_CLASSES)
    with pytest.raises(lab.EvidencePromotionError, match="not one of"):
        lab.record_placement(
            preferred="CPU",
            supported=["CPU"],
            evidence_class="OBSERVED",
            source="typo",
            is_live=True,
            api="MLComputePlan.computeDeviceUsage(for:)",
        )
    plan = {
        "api": "MLComputePlan.load(contentsOf:configuration:)",
        "model_structure": "MLProgram",
        "status": "PLANNED",
        "is_live": True,
        "is_fixture": False,
        "operations": [
            {
                "index": 0,
                "operator": "ios16.mul",
                "preferred": "CPU",
                "supported": ["CPU"],
                "placement_status": "PLANNED",
            }
        ],
    }
    parsed = preboard.parse_mlcompute_plan(plan)
    rec = lab.placement_evidence_from_plan(parsed)
    assert rec["preferred"] == "CPU"


def test_valid_spec_does_not_compile_without_coremltools():
    spec = lab.validate_graph_spec(
        op="add", shape=(1, 8), dtype="float16", structure="elementwise"
    )
    assert spec["compiled"] is False
    assert spec["dtype"] == "fp16"
    assert spec["shape"] == [1, 8]
    matmul = lab.validate_graph_spec(
        op="matmul", shape=[1, 8, 8], dtype="fp32", structure="linear"
    )
    assert matmul["op"] == "matmul"
    assert "compiled_model" not in spec
