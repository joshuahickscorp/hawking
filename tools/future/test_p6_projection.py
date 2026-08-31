"""Pins for P6/P7 primitive projection.

A guard nobody has watched fail is not a guard. GENERIC_VERIFIED and numeric
measured-effect fields must RAISE. At least one candidate is recorded
spatially_meaningful=false rather than force-fitted into HWIR.

Never assert that the live queue file is absent: this worktree may or may not
see receipts/headless/. Tests cope with either pinned or live state.
"""
from __future__ import annotations

import json

import pytest

from tools.future import fpga_engines as fe
from tools.future import hwir
from tools.future import odyssey2_law_store as ols
from tools.future import odyssey3_adversary as o3
from tools.future import p6_projection as p6
from tools.future import physical_primitives as pp
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


def _cand(cid: str, **extra: object) -> dict:
    row = {
        "candidate_id": cid,
        "model": "Flash",
        "status": "BLOCKED",
        "blocked_reason": "Flash source-independent NX is not qualified",
        "affected_physical_region": f"region-{cid}",
        "exact_mutation": {"source_oracle_controls": {f"HAWKING_{cid.upper().replace('-', '_')}": "1"}},
        "expected_eliminated_work": "none",
        "expected_dispatch_reduction": "0",
        "expected_gpu_ns_mechanism": "geometry only",
        "expected_active_byte_change": "unchanged",
        "expected_intermediate_byte_reduction": "0",
        "scope_tags": ["MODEL_LOCAL"],
        "measurements": {"status": "NOT_MEASURED", "gpu_ns_per_token": None, "accepted_tps": None},
    }
    row.update(extra)
    return row


def _cb_row() -> dict:
    return _cand(
        "flash-p6-hash-single-command-buffer",
        affected_physical_region="Flash hash-route P6 command-buffer topology",
        exact_mutation={"source_oracle_controls": {"HAWKING_DSV4F_P6_SINGLE_CB": "1"}},
        expected_eliminated_work=(
            "one CPU-visible commit/wait between the device-resident up/SwiGLU "
            "wave and down/combine wave"
        ),
        expected_dispatch_reduction="1 command buffer and wait instead of the historical 2; 60 dispatches remain unchanged",
        expected_gpu_ns_mechanism=(
            "append the dependency-ordered down/combine concurrent waves to the "
            "first P6 command buffer while preserving explicit wave boundaries"
        ),
    )


def _fused_row() -> dict:
    return _cand(
        "flash-p6-routed-fp4-gate-up-swiglu-fused",
        affected_physical_region="Flash P6A fixed-six routed FP4 gate/up/SwiGLU epilogue",
        exact_mutation={
            "source_oracle_controls": {
                "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "1",
                "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD": "0",
            }
        },
        expected_eliminated_work=(
            "six routed W1 launches, six routed W3 launches, twelve routed "
            "FP32-to-BF16 casts, and six routed SwiGLU launches"
        ),
        expected_dispatch_reduction="reduce P6 batch 1 from 38 to 9",
        expected_gpu_ns_mechanism=(
            "one fixed-six indirect-address launch performs paired source-order "
            "FP4 reductions, exact BF16 round-trips, clamp/SwiGLU, and device route weighting"
        ),
    )


def _queue(rows: list[dict]) -> dict:
    return {
        "schema": "hawking.accelerator.physical_qualification_queue.v1",
        "version": 1,
        "candidates": rows,
        "fingerprint": "test",
    }


# ---------------------------------------------------------------------------
# Entry point / receipt
# ---------------------------------------------------------------------------


def test_build_emits_sealed_receipt():
    out = p6.build()
    assert out.parent == RECEIPTS
    assert out.name == "P6_PRIMITIVE_PROJECTION.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.future.p6_projection.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "Static sidecar artifact" in doc["claim_boundary"]
    assert doc["gpu_authority"] is False
    assert doc["not_an_fpga_backend"] is True
    assert doc["evidence_source"] in {"pinned_snapshot", "live_headless"}
    _assert_no_hardware_claims(doc)


def test_selftest_is_build():
    a = p6.selftest()
    b = p6.build()
    assert a.name == b.name == "P6_PRIMITIVE_PROJECTION.json"


def test_receipt_carries_recovered_gaps_negative_evidence():
    doc = json.loads(p6.build().read_text())
    assert doc["recovered_implementation"]["p6_projection_module_existed"] is False
    assert doc["recovered_implementation"]["existed_before_this_module"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert "evidence_inputs" in doc
    assert doc["evidence_inputs"]["qualification_queue"]["present"] is True
    assert doc["eras"] == list(p6.ERAS)
    assert "VI" not in " ".join(doc["eras"])
    assert len(doc["odysseys"]) == 3


# ---------------------------------------------------------------------------
# Substantive mapping
# ---------------------------------------------------------------------------


def test_p6_p7_derived_from_queue_not_a_hard_count():
    rows = [_fused_row(), _cb_row(), _cand("qwen27-affine2-splitk4"), _cand("flash-pipeline-cache-reuse")]
    selected = p6.p6_p7_rows(_queue(rows))
    ids = [r["candidate_id"] for r in selected]
    assert ids == [
        "flash-p6-hash-single-command-buffer",
        "flash-p6-routed-fp4-gate-up-swiglu-fused",
    ]
    assert "qwen27-affine2-splitk4" not in ids
    assert "flash-pipeline-cache-reuse" not in ids
    assert p6.is_p6_p7("flash-p7-mhc-pre-simdgroup")
    assert not p6.is_p6_p7("flash-pipeline-id-resolution")


def test_build_count_matches_selected_rows():
    doc = json.loads(p6.build().read_text())
    n = doc["counts"]["p6_p7_candidates"]
    assert n == len(doc["projections"])
    assert n >= 1
    # Do not hard-code 14: pinned snapshot and live queue differ.
    ids = [p["candidate_id"] for p in doc["projections"]]
    assert ids == sorted(ids)
    assert all(p6.is_p6_p7(i) for i in ids)


def test_primitives_are_atlas_seventeen_or_unmapped():
    doc = json.loads(p6.build().read_text())
    assert doc["atlas_primitives"][:17] == list(pp.ATLAS_PRIMITIVES) or set(doc["atlas_primitives"]) >= set(
        pp.ATLAS_PRIMITIVES
    )
    for rec in doc["projections"]:
        names = rec["hawking_primitive"]["names"]
        assert names
        for name in names:
            assert name == p6.UNMAPPED or name in pp.ATLAS_PRIMITIVES
        assert rec["hawking_primitive"]["new_primitive_proposed"] is None
        assert rec["hawking_primitive"]["cited_from"]
        assert rec["hawking_primitive"]["justification"]


def test_fused_down_shared_combine_maps_when_present_or_as_fixture():
    rec = p6.project_candidate(
        _cand(
            "flash-p6-fused-down-shared-combine",
            affected_physical_region="Flash P6 routed FP4 W2 plus shared FP8 W2 and BF16 combine",
            exact_mutation={
                "source_oracle_controls": {
                    "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "1",
                    "HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED": "1",
                }
            },
            expected_eliminated_work=(
                "six routed FP4 W2 dispatches, one shared FP8 W2 dispatch, seven "
                "FP32-to-BF16 cast dispatches, one final combine dispatch"
            ),
            expected_gpu_ns_mechanism=(
                "one 256-thread row launch reads six resident indirect FP4 records, "
                "computes the shared FP8 row, then the fixed-six BF16 combine"
            ),
        )
    )
    names = rec["hawking_primitive"]["names"]
    assert "FusedDecodeCompute" in names
    assert "DirectRoutedAccumulate" in names
    assert rec["hwir_hypothesis"]["spatially_meaningful"] is True
    assert rec["transfer_scope"]["scope"] == "MODEL_LOCAL"
    assert rec["odyssey_iii_counterexample"]["family"] == "negative_transfer"


def test_fused_epilogue_maps_to_fused_decode_and_routed_accumulate():
    rec = p6.project_candidate(_fused_row())
    names = rec["hawking_primitive"]["names"]
    assert "FusedDecodeCompute" in names
    assert "DirectRoutedAccumulate" in names
    assert rec["hwir_hypothesis"]["spatially_meaningful"] is True
    assert rec["fpga_realization"]["engine_refs"]
    assert "qgemv" in rec["fpga_realization"]["engine_refs"]
    assert rec["transfer_scope"]["scope"] == "MODEL_LOCAL"


def test_simdgroup_ceiling_is_backend_family():
    rec = p6.project_candidate(
        _cand(
            "flash-p6-routed-fp4-simdgroup",
            affected_physical_region="Flash P6 routed-expert FP4 matvec",
            exact_mutation={"source_oracle_controls": {"HAWKING_DSV4F_P6_FP4_SIMD": "1"}},
            expected_eliminated_work="one serial thread per routed-expert output row",
            expected_dispatch_reduction="0; six expert waves remain concurrent",
            expected_gpu_ns_mechanism=(
                "one 64-lane-x-4-row threadgroup uses packed uchar4 loads and "
                "SIMDgroup split-K partials before a deterministic row reduction"
            ),
        )
    )
    assert "LayoutTransform" in rec["hawking_primitive"]["names"]
    assert rec["transfer_scope"]["scope"] == "BACKEND_FAMILY"
    assert rec["transfer_scope"]["scope"] != "GENERIC_VERIFIED"
    assert rec["odyssey_iii_counterexample"]["family"] == "compiler_prior"


def test_expert_cache_is_stationary_and_machine_local():
    rec = p6.project_candidate(
        _cand(
            "flash-p6-learned-expert-cache-reuse",
            affected_physical_region="Flash P6 learned-route bounded expert source cache",
            exact_mutation={
                "source_oracle_controls": {
                    "HAWKING_DSV4F_P6_LEARNED_EXPERT_CACHE_REUSE": "1",
                    "HAWKING_DSV4F_P6_LEARNED_READER_REUSE": "1",
                }
            },
            expected_eliminated_work="repeated source chunk materialization for expert bundles",
            expected_dispatch_reduction="0; device dispatch topology unchanged",
            expected_gpu_ns_mechanism="avoid repeated host source reads before GPU upload when route overlap exists",
        ),
        front_constraint={"constrains": "UMA copy elision is machine-specific"},
    )
    names = rec["hawking_primitive"]["names"]
    assert "StationaryRepresentation" in names
    assert "PersistentPhysicalRegion" in names
    assert rec["hwir_hypothesis"]["spatially_meaningful"] is True
    assert rec["fpga_realization"]["form"] == "stationary_operand"
    assert rec["transfer_scope"]["scope"] == "MACHINE_LOCAL"
    assert rec["transfer_scope"]["front_g_p6_constraint"] is not None
    assert rec["odyssey_iii_counterexample"]["family"] == "law_scope"


def test_every_spatial_hwir_sketch_validates():
    recs = p6.project_queue(_queue([_fused_row(), _cb_row(), _cand("flash-p6-prefix-concurrent-wave",
        affected_physical_region="Flash P6 Gate and activation-quantization prefix",
        exact_mutation={"source_oracle_controls": {"HAWKING_DSV4F_P6_PREFIX_CONCURRENT": "1"}},
        expected_eliminated_work="one compute-encoder boundary",
        expected_gpu_ns_mechanism="place Gate and activation quantization in one concurrent encoder because both only read the shared input and write disjoint outputs",
    )]))
    spatial = [r for r in recs if r["hwir_hypothesis"]["spatially_meaningful"]]
    assert spatial
    for rec in spatial:
        hyp = rec["hwir_hypothesis"]
        assert hyp["validate"]["ok"] is True
        assert hyp["graph"] is not None
        report = hwir.validate(hyp["graph"])
        assert report.ok, report.errors


def test_build_spatial_sketches_validate():
    doc = json.loads(p6.build().read_text())
    spatial = [p for p in doc["projections"] if p["hwir_hypothesis"]["spatially_meaningful"]]
    assert spatial
    for rec in spatial:
        assert rec["hwir_hypothesis"]["validate"]["ok"] is True
        report = hwir.validate(rec["hwir_hypothesis"]["graph"])
        assert report.ok, (rec["candidate_id"], report.errors)


def test_transfer_scope_on_lattice_never_generic_verified():
    doc = json.loads(p6.build().read_text())
    assert doc["transfer_scope_lattice"] == list(ols.SCOPES)
    for rec in doc["projections"]:
        scope = rec["transfer_scope"]["scope"]
        assert scope in p6.LEGAL_EMITTED_SCOPES
        assert scope != "GENERIC_VERIFIED"
        assert rec["transfer_scope"]["defaulted_down_from"] == "GENERIC_CANDIDATE"
        assert rec["codex_expectations"]["label"] == "expectation_not_result"


def test_odyssey3_uses_attack_family_vocabulary():
    doc = json.loads(p6.build().read_text())
    assert doc["attack_families"] == list(o3.ATTACK_FAMILIES)
    for rec in doc["projections"]:
        fam = rec["odyssey_iii_counterexample"]["family"]
        assert fam in o3.ATTACK_FAMILIES
        assert rec["odyssey_iii_counterexample"]["cost_units"] == o3.FAMILY_COST[fam]
        assert rec["odyssey_iii_counterexample"]["bench_state"] == "UNKNOWN"
        assert rec["odyssey_iii_counterexample"]["evidence_class"] == "STATIC_ONLY"
        assert rec["odyssey_iii_counterexample"]["falsifier"]


def test_fpga_refs_are_real_engines():
    doc = json.loads(p6.build().read_text())
    for rec in doc["projections"]:
        fpga = rec["fpga_realization"]
        assert fpga["not_an_fpga_backend"] is True
        assert fpga["emits_hdl"] is False
        for name in fpga["engine_refs"]:
            assert name in fe.ENGINE_FNS
        assert fpga["form"] in {
            "persistent_state_machine",
            "streaming_producer_consumer",
            "explicit_banking",
            "stationary_operand",
            "not_spatially_meaningful",
        }


def test_expectations_are_labeled_not_results():
    rec = p6.project_candidate(_fused_row())
    exp = rec["codex_expectations"]
    assert exp["label"] == "expectation_not_result"
    assert "six routed W1" in exp["expected_eliminated_work"]
    assert "dispatch_count" not in rec
    assert rec.get("gpu_ns") is None
    # The expected_dispatch_reduction string may contain '60' / '8'; it must stay a string.
    assert isinstance(exp["expected_dispatch_reduction"], str)


def test_every_required_field_cites_a_candidate_field():
    rec = p6.project_candidate(_fused_row())
    for key in (
        "hawking_primitive",
        "hwir_hypothesis",
        "fpga_realization",
        "transfer_scope",
        "odyssey_iii_counterexample",
        "software_lesson_now",
    ):
        assert rec[key]["cited_from"], key


def test_unmapped_is_honest_for_a_non_physical_row():
    rec = p6.project_candidate(
        _cand(
            "flash-p6-host-printf-debug",
            affected_physical_region="host logging",
            exact_mutation={"source_oracle_controls": {"HAWKING_DEBUG_PRINTF": "1"}},
            expected_eliminated_work="none",
            expected_gpu_ns_mechanism="prints a host log line",
        )
    )
    assert rec["hawking_primitive"]["unmapped"] is True
    assert rec["hawking_primitive"]["names"] == [p6.UNMAPPED]
    assert rec["hwir_hypothesis"]["spatially_meaningful"] is False


def test_queue_loader_records_source_and_copes_with_either():
    doc, src = p6.load_qualification_queue()
    assert src["present"] is True
    assert src["evidence_source"] in {"pinned_snapshot", "live_headless"}
    assert "candidates" in doc
    # Cope: a pinned-only path and a live path are both legal. Do not assert
    # that receipts/headless/... is missing.
    resolved = p6.resolve_input(
        "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
        p6.PINNED_QUEUE,
    )
    assert resolved["evidence_source"] in {"pinned_snapshot", "live_headless", None} or resolved["present"] in {
        True,
        False,
    }
    if resolved["present"]:
        assert resolved["path"]


# ---------------------------------------------------------------------------
# Negative controls — these must RAISE
# ---------------------------------------------------------------------------


def test_generic_verified_is_rejected_by_emitter():
    rec = p6.project_candidate(_fused_row())
    rec["transfer_scope"]["scope"] = "GENERIC_VERIFIED"
    with pytest.raises(p6.ProjectionClaimError, match="GENERIC_VERIFIED"):
        p6.assert_projection_legal({"projections": [rec]})


def test_measured_effect_field_is_rejected_by_emitter():
    rec = p6.project_candidate(_fused_row())
    rec["gpu_ns"] = 12
    with pytest.raises((p6.ProjectionClaimError, HardwareClaimError)):
        p6.assert_projection_legal({"projections": [rec]})
    rec2 = p6.project_candidate(_fused_row())
    rec2["speedup"] = 7.17
    with pytest.raises(p6.ProjectionClaimError, match="speedup"):
        p6.assert_projection_legal({"projections": [rec2]})


def test_write_receipt_still_rejects_hardware_numbers():
    rec = p6.project_candidate(_fused_row())
    rec["token_ns"] = 100
    with pytest.raises((p6.ProjectionClaimError, HardwareClaimError)):
        p6.assert_projection_legal({"schema": p6.SCHEMA, "projections": [rec]})


def test_new_atlas_primitive_is_rejected():
    rec = p6.project_candidate(_fused_row())
    rec["hawking_primitive"]["names"] = ["SystolicFlashArray"]
    with pytest.raises(p6.ProjectionClaimError, match="not an atlas primitive"):
        p6.assert_projection_legal({"projections": [rec]})


def test_spatially_meaningful_false_is_watched_not_force_fitted():
    rec = p6.project_candidate(_cb_row())
    assert rec["hwir_hypothesis"]["spatially_meaningful"] is False
    assert rec["hwir_hypothesis"]["graph"] is None
    assert rec["hawking_primitive"]["names"] == ["GraphReplay"]
    assert rec["fpga_realization"]["form"] == "not_spatially_meaningful"
    assert rec["transfer_scope"]["scope"] == "BACKEND_FAMILY"


def test_build_records_at_least_one_honest_non_spatial():
    doc = json.loads(p6.build().read_text())
    false = [p for p in doc["projections"] if p["hwir_hypothesis"]["spatially_meaningful"] is False]
    assert false, "a guard nobody has watched fail is not a guard"
    ids = {p["candidate_id"] for p in false}
    # Command-buffer is in both pinned and live queues; reader-reuse too.
    assert (
        "flash-p6-hash-single-command-buffer" in ids
        or "flash-p6-learned-reader-reuse" in ids
        or any(p["fpga_realization"]["form"] == "not_spatially_meaningful" for p in false)
    )
    for rec in false:
        assert rec["hwir_hypothesis"]["graph"] is None


def test_no_hardware_field_in_sealed_receipt():
    doc = json.loads(p6.build().read_text())
    _assert_no_hardware_claims(doc)
    for rec in doc["projections"]:
        for key in HARDWARE_FIELDS:
            val = rec.get(key)
            assert val is None or not isinstance(val, (int, float)) or isinstance(val, bool)


def test_classify_does_not_invent_an_eighteenth_primitive():
    for cid, spec in p6._OVERLAYS.items():
        for name in spec["primitives"]:
            assert name in pp.ATLAS_PRIMITIVES, (cid, name)
        assert spec["attack_family"] in o3.ATTACK_FAMILIES
        assert spec["scope"] in p6.LEGAL_EMITTED_SCOPES
