"""Tests for the ANE / Core ML preboard.

The load-bearing guard is the negative control: every execution entry point
must RAISE while toolchain_available() is False, and no code path may return
a placement or a latency while the toolchain is absent. A guard nobody has
watched fail is not a guard.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from tools.future import ane_preboard as ane
from tools.future._common import HARDWARE_FIELDS, RECEIPTS


def test_build_emits_sealed_receipt():
    out = ane.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ANE_PREBOARD.json"
    assert doc["schema"] == "hawking.future.ane_preboard.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["claim_class"] == "STATIC_ONLY"
    assert "DIAGNOSTIC_RELATIVE" in doc["does_not_produce"]
    assert "PROTECTED_ABSOLUTE" in doc["does_not_produce"]
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["graph_corpus"]
    assert doc["mlstate_cases"]
    assert doc["mlcomputeplan"]["fixture_is_a_fixture"] is True
    assert doc["placement_receipts"]["unknown_today"] is True
    assert doc["test_matrix"]
    assert doc["toolchain"]["available"] is False
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    assert ane.ERAS[-1].startswith("V ")
    assert "VI" not in "".join(ane.ERAS)
    assert len(ane.ODYSSEYS) == 3


def test_selftest_emits_sealed_receipt():
    out = ane.selftest()
    doc = json.loads(out.read_text())
    assert doc["seal_sha256"]
    assert doc["schema"] == ane.SCHEMA
    assert doc["bench"]["state"] == "UNKNOWN"


def test_toolchain_unavailable_is_the_headline():
    probe = ane.probe_toolchain()
    assert probe["available"] is False
    assert ane.toolchain_available() is False
    assert probe["missing"]
    blob = " ".join(probe["missing"]).lower()
    assert "xcode" in blob or "coremltools" in blob or "commandlinetools" in blob
    assert "did_not" in probe
    assert "install any package" in probe["did_not"]
    assert "run xcode-select to change the active developer directory" in probe["did_not"]
    assert "prepared-but-unrun" in probe["headline"]


def test_graph_corpus_uses_real_flash_and_qwen27_shapes():
    corpus = {g["id"]: g for g in ane.graph_corpus()}
    expert = corpus["flash.expert.gate_up_swiglu"]
    assert expert["shapes"]["weight"] == [512, 1280, 2560]
    assert expert["shapes"]["activation"] == [1, 2560]
    assert expert["shapes"]["expert_activation"] == [1, 640]
    assert "FLASH_ORGAN_CENSUS" in expert["source"]

    hc = corpus["flash.hc.grouped_rmsnorm"]
    assert hc["shapes"]["activation"] == [4, 2560]

    gate = corpus["qwen27.mlp.gate_proj"]
    assert gate["shapes"]["activation"] == [1, 5120]
    assert gate["shapes"]["weight"] == [17408, 5120]
    assert "ACCELERATOR_DISPATCH_IS_NOT_THE_COST" in gate["source"]

    down = corpus["qwen27.mlp.down_proj"]
    assert down["shapes"]["weight"] == [5120, 17408]

    lm = corpus["qwen27.lm_head"]
    assert lm["shapes"]["weight"] == [248320, 5120]

    qkv = corpus["qwen27.qkv_and_projection"]
    assert qkv["shapes"]["weight"] == "UNKNOWN"
    assert qkv["shapes"]["n_heads"] is None
    assert qkv["shapes"]["head_dim"] is None

    for graph in corpus.values():
        assert graph["compile"]["status"] == "NOT_RUN"
        assert graph["placement"]["status"] == "UNKNOWN"
        assert graph["placement"]["preferred"] is None
        assert graph["latency"] == {"cold_ns": None, "warm_ns": None, "throughput": None}
        assert graph["energy"]["joules"] is None
        assert graph["evidence_class"] == "STATIC_ONLY"
        assert graph["bench_state"] == "UNKNOWN"


def test_no_invented_qwen27_head_counts():
    geo = ane.recover_qwen27_geometry()
    assert geo["hidden_size"] == 5120
    assert geo["intermediate_size"] == 17408
    assert geo["layers"] == 64
    assert geo["vocab_size"] == 248320
    assert geo["n_heads"] is None
    assert geo["n_kv_heads"] is None
    assert geo["head_dim"] is None
    assert geo["head_counts_status"] == "UNKNOWN_NOT_IN_CENSUS_OR_BUDGET"
    assert "qkv_and_projection" in geo["organs"]
    assert "deltanet_and_recurrent_state" in geo["organs"]


def test_flash_geometry_matches_census_and_atlas():
    geo = ane.recover_flash_geometry()
    assert geo["hidden_size"] == 2560
    assert geo["n_experts"] == 512
    assert geo["expert_intermediate"] == 640
    assert geo["expert_gate_up_shape"] == [512, 1280, 2560]
    assert geo["layer_count"] == 48
    assert geo["experts_per_token"] == 10
    assert geo["full_attention_layers"] == 12
    assert geo["atlas_shapes"]["flash_state"] == [1, 48, 128, 128]
    assert geo["hyperconnection_base"] == [4, 2560]
    assert geo["sdpa_shapes"] == [[1, 2560], [1, 24, 1, 256], [1, 2, 1, 256]]
    assert geo["conv_shapes"] == [[1, 12288, 4], [1, 12288, 1]]


def test_mlstate_cases_declare_what_they_test():
    cases = {c["id"]: c for c in ane.mlstate_cases()}
    assert "flash.deltanet.recurrent_state" in cases
    assert cases["flash.deltanet.recurrent_state"]["shape"] == [1, 48, 128, 128]
    assert "read-modify-write" in cases["flash.deltanet.recurrent_state"]["tests"]
    assert cases["qwen27.deltanet.recurrent_state"]["shape"] == "UNKNOWN"
    assert cases["qwen27.gqa.kv_cache"]["shape"] == "UNKNOWN"
    assert "UNKNOWN" in cases["qwen27.gqa.kv_cache"]["shape_status"]
    for case in cases.values():
        assert case["api"].startswith("Core ML MLState")
        assert case["tests"]
        assert case["source"]


def test_mlcomputeplan_parser_on_fixture():
    fixture = ane.mlcomputeplan_fixture()
    assert fixture["is_fixture"] is True
    assert "fixture" in fixture["fixture_citation"].lower()
    parsed = ane.parse_mlcompute_plan(fixture)
    assert parsed["ok"] is True
    assert parsed["is_fixture"] is True
    assert parsed["is_live"] is False
    assert parsed["n_operations"] >= 1
    op0 = parsed["operations"][0]
    assert op0["operator"] == "ios16.mul"
    assert op0["preferred"] == "CPU"
    assert "CPU" in op0["supported"]
    assert "NEURAL_ENGINE" in op0["supported"]
    assert parsed["evidence_class"] == "STATIC_ONLY"
    assert "not a live placement" in parsed["claim_boundary"]


def test_fixture_is_labeled_fixture_in_receipt():
    doc = json.loads(ane.build().read_text())
    plan = doc["mlcomputeplan"]
    assert plan["fixture_is_a_fixture"] is True
    assert plan["fixture"]["is_fixture"] is True
    assert plan["parsed_fixture"]["is_fixture"] is True
    assert plan["parsed_fixture"]["is_live"] is False


def test_parser_rejects_malformed_plans():
    with pytest.raises(ane.PlanParseError, match="mapping"):
        ane.parse_mlcompute_plan("not a plan")  # type: ignore[arg-type]
    with pytest.raises(ane.PlanParseError, match="missing plan key"):
        ane.parse_mlcompute_plan({"api": "x"})
    with pytest.raises(ane.PlanParseError, match="operations"):
        ane.parse_mlcompute_plan(
            {"api": "x", "model_structure": "MLProgram", "operations": "nope", "status": "PLANNED"}
        )
    with pytest.raises(ane.PlanParseError, match="preferred"):
        ane.parse_mlcompute_plan(
            {
                "api": "x",
                "model_structure": "MLProgram",
                "status": "PLANNED",
                "operations": [
                    {
                        "index": 0,
                        "operator": "ios16.mul",
                        "preferred": "TPU",
                        "supported": ["CPU"],
                        "placement_status": "PLANNED",
                    }
                ],
            }
        )


def test_placement_schema_all_unknown_today():
    blank = ane.blank_placement_receipt("flash.elementwise.multiply")
    assert blank["schema"] == ane.PLACEMENT_SCHEMA
    assert blank["status"] == "UNKNOWN"
    assert blank["preferred_compute_device"] == "UNKNOWN"
    assert blank["supported_compute_devices"] == "UNKNOWN"
    assert blank["placement_status"] == "UNKNOWN"
    assert blank["estimated_cost_weight"] == "UNKNOWN"
    assert blank["latency"] == {"cold_ns": None, "warm_ns": None, "throughput": None}
    assert blank["energy"]["joules"] is None
    assert blank["transfer_sync"]["cpu_ane_bytes"] is None
    assert blank["transfer_sync"]["sync_events"] is None
    assert blank["evidence_class"] == "STATIC_ONLY"
    assert blank["bench_state"] == "UNKNOWN"
    assert blank["gpu_authority"] is False
    schema = ane.placement_schema()
    assert schema["unknown_today"] is True
    for slot in ane.PLACEMENT_SLOTS:
        assert slot in schema["blank"] or slot == "graph_id"


def test_test_matrix_has_required_kinds_and_preconditions():
    matrix = {s["id"]: s for s in ane.test_matrix()}
    assert set(matrix) >= {
        "ane_only",
        "metal_only_control",
        "heterogeneous_control",
        "concurrency",
        "transfer_sync_accounting",
    }
    assert matrix["ane_only"]["kind"] == "ANE_ONLY"
    assert matrix["metal_only_control"]["kind"] == "METAL_ONLY_CONTROL"
    assert matrix["heterogeneous_control"]["kind"] == "HETEROGENEOUS_CONTROL"
    assert matrix["concurrency"]["kind"] == "CONCURRENCY"
    assert matrix["transfer_sync_accounting"]["kind"] == "TRANSFER_SYNC"
    for spec in matrix.values():
        assert spec["runnable_now"] is False
        assert spec["entry_point"] == "execute_test_spec"
        assert spec["preconditions"]
        assert any("toolchain_available()" in p for p in spec["preconditions"])
        assert "latency" in spec["does_not_record"]
        assert "tps" in spec["does_not_record"]


def _assert_raises_toolchain(fn, *args, **kwargs):
    with pytest.raises(ane.ToolchainUnavailableError) as caught:
        fn(*args, **kwargs)
    msg = str(caught.value).lower()
    assert "toolchain unavailable" in msg
    assert caught.value.probe["available"] is False
    assert caught.value.probe["missing"]
    return caught.value


def test_every_execution_entry_point_raises_without_toolchain():
    """Negative control: the refusal must actually fire, and name the missing dep."""
    assert ane.toolchain_available() is False
    fired = []
    for name, thunk in ane.execution_entry_points():
        with pytest.raises(ane.ToolchainUnavailableError) as caught:
            thunk()
        err = caught.value
        assert err.probe["available"] is False, name
        assert err.probe["missing"], name
        assert "unavailable" in str(err).lower(), name
        fired.append(name)
    expected = {
        "compile_mlprogram",
        "load_live_compute_plan",
        "inspect_compute_plan_live",
        "place_graph",
        "compile_and_place",
        "execute_test_spec",
        "placement_from_plan",
    }
    assert set(fired) == expected
    # Named calls with explicit arguments, so a refactor of the thunk table
    # cannot silently drop the guard.
    _assert_raises_toolchain(ane.compile_mlprogram, "flash.elementwise.multiply")
    _assert_raises_toolchain(ane.place_graph, "flash.elementwise.multiply")
    _assert_raises_toolchain(ane.compile_and_place, "qwen27.mlp.gate_proj")
    _assert_raises_toolchain(ane.execute_test_spec, "ane_only")
    _assert_raises_toolchain(ane.load_live_compute_plan, "/nonexistent.mlmodelc")
    _assert_raises_toolchain(ane.inspect_compute_plan_live, "/nonexistent.mlmodelc")
    _assert_raises_toolchain(
        ane.placement_from_plan, "flash.elementwise.multiply", ane.mlcomputeplan_fixture()
    )


def test_estimate_and_measure_always_raise():
    with pytest.raises(ane.AneNumberForbiddenError, match="estimate"):
        ane.estimate_ane_latency(graph_id="flash.elementwise.multiply")
    with pytest.raises(ane.AneNumberForbiddenError, match="PROTECTED_ABSOLUTE"):
        ane.measure_prediction(graph_id="flash.elementwise.multiply")
    for name, thunk in ane.forbidden_number_entry_points():
        with pytest.raises(ane.AneNumberForbiddenError):
            thunk()
        assert name in {"estimate_ane_latency", "measure_prediction"}


def _walk_numeric_hardware(node: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                hits.append(f"{here}={value!r}")
            lowered = key.lower()
            if isinstance(value, (int, float)) and any(
                tok in lowered
                for tok in ("latency", "cold_ns", "warm_ns", "token_ns", "gpu_ns", "tps", "joule", "bandwidth")
            ):
                hits.append(f"{here}={value!r}")
            hits.extend(_walk_numeric_hardware(value, here))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            hits.extend(_walk_numeric_hardware(value, f"{path}[{i}]"))
    return hits


def test_no_placement_or_latency_returned_without_toolchain():
    assert ane.toolchain_available() is False

    static = {
        "corpus": ane.graph_corpus(),
        "mlstate": ane.mlstate_cases(),
        "matrix": ane.test_matrix(),
        "blank": ane.blank_placement_receipt("flash.elementwise.multiply"),
        "schema": ane.placement_schema(),
        "fixture_parse": ane.parse_mlcompute_plan(ane.mlcomputeplan_fixture()),
        "geometry": ane.recover_geometry(),
        "probe": ane.probe_toolchain(),
    }
    hits = _walk_numeric_hardware(static)
    assert hits == [], hits

    for graph in static["corpus"]:
        assert graph["placement"]["preferred"] is None
        assert graph["latency"]["cold_ns"] is None
        assert graph["latency"]["warm_ns"] is None
        assert graph["latency"]["throughput"] is None

    parsed = static["fixture_parse"]
    assert parsed["is_fixture"] is True
    # The fixture may contain preferred CPU: that is the pinned document, not
    # a placement this process produced for a Flash/Qwen27 graph.
    assert parsed["is_live"] is False

    blank = static["blank"]
    assert blank["preferred_compute_device"] == "UNKNOWN"
    assert blank["latency"]["cold_ns"] is None

    for name, thunk in ane.execution_entry_points():
        with pytest.raises(ane.ToolchainUnavailableError):
            result = thunk()
            raise AssertionError(f"{name} returned {result!r} instead of raising")


def test_receipt_has_no_hardware_numbers():
    doc = json.loads(ane.build().read_text())
    hits = _walk_numeric_hardware(doc)
    # estimated_cost_weight on the fixture op is null, not a number.
    assert hits == [], hits
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["placement_receipts"]["blank"]["preferred_compute_device"] == "UNKNOWN"


def test_recovered_implementation_reports_absent_named_seams():
    recovered = {row["path"]: row for row in ane.recovered_implementation()}
    assert recovered["receipts/future/evidence/APPLE_ANE_ATLAS.json"]["on_disk"] is True
    assert recovered["receipts/future/evidence/APPLE_ANE_DEVICE_PROFILE.json"]["on_disk"] is True
    assert recovered["receipts/future/evidence/FLASH_ORGAN_CENSUS.json"]["on_disk"] is True
    assert recovered["receipts/future/evidence/QWEN27_TOKEN_NS_BUDGET.json"]["on_disk"] is True
    # These seams are uncommitted, so a sparse lane worktree sees ABSENT and the
    # primary worktree sees ON_DISK. Which one is a fact about the checkout, not
    # about this module. What must hold in BOTH: every named seam is accounted
    # for with a recognised state, and one that is genuinely absent is surfaced
    # as a negative finding rather than silently dropped.
    named_seams = (
        "hcli/ane_provider.py",
        "hcli/test_ane_provider.py",
        "tools/accelerator/ane_micrograph_author.py",
        "ane_probe.swift",
        "run_ane_lane.sh",
    )
    for seam in named_seams:
        assert recovered[seam]["source"] in {"ABSENT", "ON_DISK", "GIT_HEAD"}, seam
    findings = "\n".join(ane.negative_findings())
    for seam in named_seams:
        if recovered[seam]["source"] == "ABSENT":
            assert seam.split("/")[-1] in findings, f"{seam} absent but not reported"


def test_corpus_is_sorted_and_deterministic():
    a = [g["id"] for g in ane.graph_corpus()]
    b = [g["id"] for g in ane.graph_corpus()]
    assert a == b
    assert a == sorted(a)
    cases = [c["id"] for c in ane.mlstate_cases()]
    assert cases == sorted(cases)


def test_require_toolchain_names_missing_dependency():
    with pytest.raises(ane.ToolchainUnavailableError) as caught:
        ane.require_toolchain()
    text = str(caught.value)
    assert "coremltools" in text or "Xcode" in text or "CommandLineTools" in text
    assert caught.value.probe["missing"]
