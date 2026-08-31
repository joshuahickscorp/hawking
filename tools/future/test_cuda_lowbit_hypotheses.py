"""Tests for CUDA-lowbit Hawking hypotheses.

A guard nobody has watched fail is not a guard. The negative control below
proves admit() REJECTS a numeric speedup/latency and REJECTS a recorded
negative-science scar, while still emitting a structurally different
hypothesis.
"""
from __future__ import annotations

import json

import pytest

from tools.future import cuda_lowbit_hypotheses as clb
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, _assert_no_hardware_claims
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def _valid(**over: object) -> dict:
    """A structurally legal hypothesis that is not an atlas restatement and not a scar."""
    base: dict = {
        "id": "H-TEST-001",
        "hypothesis_family": "dequant_site_tier",
        "behavior_id": "fused_decode_compute",
        "hawking_primitive": "MemoryTierIdentity",
        "compared_primitives": ["FusedDecodeCompute"],
        "target_organ": "mlp",
        "physical_invariant": (
            "Unpacked values occupy more bytes than packed ones; the legal "
            "dequant site is a named memory tier of the consumer."
        ),
        "cheapest_falsifier": (
            "STATIC: REGISTER vs THREADGROUP unpack sites that share a semantic "
            "id must not hash equal."
        ),
        "expected_removed_cost": {
            "class": "DENSE_INTERMEDIATE",
            "direction": "REDUCE",
            "measurement": "unmeasured",
            "mechanism": "avoid writing unpacked values to a slower tier",
        },
        "backend_candidate": "METAL",
        "transfer_scope": "CUDA register vs smem dequant site -> Metal thread vs threadgroup",
        "knowledge_source": "MODEL_KNOWLEDGE",
        "why_not_a_restatement": "site-tier choice is not fused-dequant fusion itself",
        "candidate": "unpack at REGISTER vs THREADGROUP",
        "control": "unpack that writes a dense intermediate to a slower tier",
    }
    base.update(over)
    return base


def _atlas() -> dict:
    return clb.load_evidence(clb.ATLAS_NAME)["doc"]


def test_build_emits_sealed_receipt():
    out = clb.build()
    assert out.parent == RECEIPTS
    assert out.name == "CUDA_LOWBIT_HYPOTHESES.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.future.cuda_lowbit.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "Static sidecar artifact" in doc["claim_boundary"]
    _assert_no_hardware_claims(doc)
    assert doc["n_hypotheses"] == len(doc["hypotheses"])
    assert doc["n_hypotheses"] == len(clb.CATALOG)
    assert doc["n_repatriation_specs"] == doc["n_hypotheses"]
    assert doc["hypotheses"], "catalog must emit at least one genuine gap"
    assert doc["gap_analysis"]["cuda_and_cutlass_already_in_schools"] is True
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert set(doc["vocabulary"]["eras"]) == set(clb.ERAS)
    assert set(doc["vocabulary"]["odysseys"]) == set(clb.ODYSSEYS)
    assert "never its own civilization" in doc["vocabulary"]["fpga_is"]


def test_module_entrypoint_runs_and_emits_sealed_receipt():
    rc = clb.main(["--build"])
    assert rc == 0
    out = RECEIPTS / clb.RECEIPT
    doc = json.loads(out.read_text())
    assert doc["schema"] == clb.SCHEMA
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"


def test_selftest_aliases_build():
    assert clb.selftest is clb.build or clb.selftest().name == clb.RECEIPT


def test_gap_analysis_reports_cuda_cutlass_overlap_and_only_emits_gaps():
    atlas = _atlas()
    gap = clb.analyze_gaps(atlas)
    schools = gap["source_schools"]
    assert "NVIDIA CUDA" in schools
    assert "CUTLASS/CUTE" in schools
    assert gap["n_cuda_cutlass_technique_rows"] == len(gap["cuda_cutlass_coverage"])
    assert gap["n_cuda_cutlass_technique_rows"] >= 1
    techniques = {r["source_technique"] for r in gap["cuda_cutlass_coverage"]}
    # Overlap the contract named: these are already mapped; we must not restate them.
    assert "persistent kernels" in techniques
    assert "fused dequant + compute" in techniques
    assert "shared-memory staging" in techniques
    assert "warp/subgroup specialization" in techniques
    assert "logical/physical/tile/lane layout algebra" in techniques
    assert "double buffering" in techniques
    families = {h["hypothesis_family"] for h in clb.CATALOG}
    covered = set(gap["covered_technique_slugs"]) | set(gap["covered_behavior_slugs"])
    collision = {_slug for _slug in (clb._slug(f) for f in families) if _slug in covered}
    assert not collision, f"catalog restates mapped CUDA/CUTLASS coverage: {collision}"
    # Split-K is a genuine hole in coverage.
    assert not any("split" in (r["source_technique"] or "").lower() for r in gap["cuda_cutlass_coverage"])


def test_each_hypothesis_maps_to_an_atlas_primitive_and_required_fields():
    atlas = _atlas()
    slugs = clb.coverage_slugs(atlas)
    for raw in clb.CATALOG:
        h = clb.admit(raw, atlas=atlas, slugs=slugs)
        for field in clb.REQUIRED_HYPOTHESIS_FIELDS:
            assert h.get(field) or raw.get(field), field
        assert h["hawking_primitive"] in ATLAS_PRIMITIVES
        assert h["knowledge_source"] in clb.KNOWLEDGE_SOURCES
        assert h["backend_candidate"] in clb.BACKENDS
        assert h["expected_removed_cost"]["class"] in clb.COST_CLASSES
        assert h["expected_removed_cost"]["measurement"] in clb.COST_MEASUREMENTS
        assert not isinstance(h["expected_removed_cost"].get("mechanism"), (int, float))
        assert h["measurement_class"] == "STATIC_ONLY"
        assert h["bench_state"] == "UNKNOWN"
        assert h["gpu_authority"] is False
        assert h["status"] == "HYPOTHESIS"
        assert h["why_not_a_restatement"]
        assert "FPGA backend" not in h["transfer_scope"] or "not" in h["transfer_scope"].lower()


def test_both_knowledge_source_labels_appear():
    sources = {h["knowledge_source"] for h in clb.CATALOG}
    assert "KNOWN_FROM_REPO" in sources
    assert "MODEL_KNOWLEDGE" in sources


def test_repatriation_specs_use_queue_vocabulary_and_are_not_ready():
    doc = json.loads(clb.build().read_text())
    queue = clb.load_evidence(clb.QUEUE_NAME)["doc"]
    queue_keys = set()
    for spec in queue.get("specs") or []:
        queue_keys |= set(spec.keys())
    required = {
        "schema",
        "experiment_id",
        "backend",
        "behavior_id",
        "candidate",
        "claim_boundary",
        "falsifier",
        "kernel_lowering",
        "organ",
        "status",
        "target",
        "verification_mode",
        "source_evidence",
        "expected_effect",
        "promotion",
        "runner",
        "benchmark_mode",
        "atlas_fingerprint",
        "control",
        "metrics",
        "command",
    }
    assert required <= queue_keys, f"queue vocabulary drifted: missing {required - queue_keys}"
    for spec in doc["repatriation_specs"]:
        assert required <= set(spec.keys())
        assert spec["schema"] == clb.EXPERIMENT_SPEC_SCHEMA
        assert spec["status"] == "HYPOTHESIS"
        assert spec["status"] != "READY"
        assert spec["runner"]["executable_here"] is False
        assert spec["command"][0] == "PLAN_ONLY"
        assert spec["nx_identity"] == "UNKNOWN"
        assert spec["measurement_class"] == "STATIC_ONLY"
        assert spec["bench_state"] == "UNKNOWN"
        assert spec["kernel_lowering"] in ATLAS_PRIMITIVES
        for payload in spec["expected_effect"].values():
            assert payload["measurement"] == "unmeasured"
            assert not isinstance(payload.get("direction"), (int, float))


def test_evidence_source_is_pinned_or_live_per_input():
    doc = json.loads(clb.build().read_text())
    src = doc["evidence_source"]
    assert src[clb.ATLAS_NAME] in {"pinned_snapshot", "live_headless"}
    assert src[clb.QUEUE_NAME] in {"pinned_snapshot", "live_headless"}
    # Prefer pinned when it resolves. Do not assert the live path is absent.
    atlas_res = doc["evidence_resolution"][clb.ATLAS_NAME]
    if atlas_res["pinned"]["resolved_via"] != "unresolved":
        assert atlas_res["source"] == "pinned_snapshot"
    rec = clb.load_evidence(clb.ATLAS_NAME)
    assert rec["evidence_source"] in {"pinned_snapshot", "live_headless"}
    assert rec["doc"]["source_schools"]
    # Recording both resolutions is how the module copes with sparse checkout.
    assert rec["pinned"]["resolved_via"] in {"disk", "git", "unresolved"}
    assert rec["live_headless"]["resolved_via"] in {"disk", "git", "unresolved"}


def test_no_hardware_numbers_anywhere_in_receipt():
    doc = json.loads(clb.build().read_text())
    _assert_no_hardware_claims(doc)

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS:
                    assert not isinstance(v, (int, float)), here
                if k in clb.NUMERIC_CLAIM_KEYS:
                    assert not isinstance(v, (int, float)), here
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)
    for h in doc["hypotheses"]:
        assert "speedup" not in h
        assert "latency" not in h
        assert "tokens_per_second" not in h
        assert "tps" not in h


def test_negative_index_queried_and_dead_families_not_emitted():
    doc = json.loads(clb.build().read_text())
    emitted = {h["hypothesis_family"] for h in doc["hypotheses"]}
    avoid = {row["family"] for row in doc["avoid_families"]}
    assert clb.AVOID_FAMILIES
    assert set(clb.AVOID_FAMILIES) <= avoid
    assert not (emitted & set(clb.AVOID_FAMILIES))
    # At least one AVOID family must have real refuse-eligible ammunition,
    # otherwise the scar gate would never fire in production.
    assert any(row["n_refuse_eligible"] >= 1 for row in doc["avoid_families"])
    assert "megakernel" in avoid
    assert "cross_expert_structure" in avoid
    assert "learned_codebook" in avoid


def test_counts_are_derived_not_hardcoded():
    atlas = _atlas()
    gap = clb.analyze_gaps(atlas)
    assert gap["n_source_schools"] == len(atlas["source_schools"])
    assert gap["n_source_technique_coverage"] == len(atlas["source_technique_coverage"])
    assert gap["n_atlas_entries"] == len(atlas["entries"])
    doc = json.loads(clb.build().read_text())
    assert doc["n_hypotheses"] == len(doc["hypotheses"]) == len(clb.CATALOG)
    # Atlas school/coverage counts may grow; do not pin 14/48.


def test_unknown_primitive_is_refused():
    with pytest.raises(clb.UnknownPrimitiveError, match="VerificationRegion"):
        clb.admit(_valid(hawking_primitive="VerificationRegion"), slugs={"techniques": set(), "behaviors": set()})


def test_missing_field_is_refused():
    raw = _valid()
    del raw["cheapest_falsifier"]
    with pytest.raises(clb.MissingFieldError, match="cheapest_falsifier"):
        clb.admit(raw, slugs={"techniques": set(), "behaviors": set()})


def test_restatement_of_atlas_cuda_technique_is_refused():
    atlas = _atlas()
    slugs = clb.coverage_slugs(atlas)
    with pytest.raises(clb.OverlapRefusal, match="persistent_kernels"):
        clb.admit(
            _valid(
                id="PROBE-persist",
                hypothesis_family="persistent_kernels",
                hawking_primitive="PersistentPhysicalRegion",
                behavior_id="persistent_physical_region",
                target_organ="decode",
            ),
            atlas=atlas,
            slugs=slugs,
        )
    with pytest.raises(clb.OverlapRefusal):
        clb.admit(
            _valid(
                id="PROBE-fused",
                hypothesis_family="fused_dequant_compute",
                hawking_primitive="FusedDecodeCompute",
            ),
            atlas=atlas,
            slugs=slugs,
        )
    # Structurally different family still admits against the same slugs.
    h = clb.admit(_valid(), atlas=atlas, slugs=slugs)
    assert h["hypothesis_family"] == "dequant_site_tier"


def test_negative_control_rejects_numeric_speedup_and_scar_but_emits_other():
    """THE guard: numeric claim dies, recorded scar dies, a different hypothesis lives."""
    atlas = _atlas()
    slugs = clb.coverage_slugs(atlas)

    with pytest.raises((HardwareClaimError, clb.NumericClaimError), match="speedup"):
        clb.admit(_valid(speedup=2.5), atlas=atlas, slugs=slugs)

    with pytest.raises((HardwareClaimError, clb.NumericClaimError), match="latency"):
        clb.admit(_valid(id="H-TEST-LAT", latency=12.0), atlas=atlas, slugs=slugs)

    with pytest.raises((HardwareClaimError, clb.NumericClaimError)):
        clb.admit(_valid(id="H-TEST-TPS", tps=140.0), atlas=atlas, slugs=slugs)

    with pytest.raises(clb.NumericClaimError, match="expected_removed_cost"):
        clb.admit(
            _valid(id="H-TEST-COST", expected_removed_cost=3.7),
            atlas=atlas,
            slugs=slugs,
        )

    with pytest.raises((HardwareClaimError, clb.NumericClaimError)):
        clb.admit(
            _valid(
                id="H-TEST-PROSE",
                physical_invariant="this kernel is 2x faster on decode GEMV",
            ),
            atlas=atlas,
            slugs=slugs,
        )

    with pytest.raises(clb.ScarRefusal, match="scar_id"):
        clb.admit(
            _valid(
                id="H-TEST-DEAD",
                hypothesis_family="cross_expert_structure",
                model="qwen3-235b-a22b",
                organ="gate",
                target_organ="gate",
                hawking_primitive="DirectRoutedAccumulate",
                behavior_id="direct_routed_accumulate",
                physical_invariant="experts share a global template",
                candidate="shared expert template",
            ),
            atlas=atlas,
            slugs=slugs,
        )

    with pytest.raises(clb.ScarRefusal):
        clb.admit(
            _valid(
                id="H-TEST-MEGA",
                hypothesis_family="megakernel",
                hawking_primitive="PersistentPhysicalRegion",
                target_organ="moe",
                behavior_id="persistent_physical_region",
                physical_invariant="fuse the whole MoE layer into one megakernel",
                candidate="full-layer MoE megakernel",
            ),
            atlas=atlas,
            slugs=slugs,
        )

    # Still emits a structurally different hypothesis after the refusals.
    live = clb.admit(_valid(id="H-TEST-LIVE"), atlas=atlas, slugs=slugs)
    assert live["id"] == "H-TEST-LIVE"
    assert live["hypothesis_family"] == "dequant_site_tier"
    assert live["hawking_primitive"] == "MemoryTierIdentity"
    assert live["expected_removed_cost"]["class"] == "DENSE_INTERMEDIATE"
    assert live["gpu_authority"] is False


def test_catalog_survives_admit_and_scar_gate():
    atlas = _atlas()
    slugs = clb.coverage_slugs(atlas)
    admitted, refusals = clb.emit_catalog(atlas)
    assert len(admitted) == len(clb.CATALOG)
    assert all(r["refused"] is True for r in refusals)
    assert all(r["disposition"] == "OVERLAP_REFUSAL" for r in refusals)
    for h in admitted:
        clb.admit(h, atlas=atlas, slugs=slugs)
        assert clb._slug(h["hypothesis_family"]) not in slugs["techniques"]
        assert h["hypothesis_family"] not in clb.AVOID_FAMILIES


def test_build_does_not_claim_cuda_backend_availability():
    doc = json.loads(clb.build().read_text())
    assert "not an execution backend" in doc["vocabulary"]["cuda_is"]
    for h in doc["hypotheses"]:
        assert h["backend_candidate"] != "CUDA" or h["gpu_authority"] is False
        # Catalog targets Metal (Hawking execution surface), CUDA is the school.
        assert h["backend_candidate"] == "METAL"
        assert h["gpu_authority"] is False
    recovered_paths = {r["path"] for r in doc["recovered_implementation"]}
    assert "tools/accelerator/c2m.py" in recovered_paths
    assert "tools/accelerator/cuda_runtime.py" in recovered_paths
    # resolved_via records which path was taken; do not assert unresolved==absent.
    for row in doc["recovered_implementation"]:
        assert row["resolved_via"] in {"disk", "git", "unresolved"}
