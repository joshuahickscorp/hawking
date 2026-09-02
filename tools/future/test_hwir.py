"""HWIR v1 tests. Negative controls must actually refuse."""
import json

import pytest

from tools.future import hwir
from tools.future._common import RECEIPTS, load_json


def test_build_emits_sealed_receipt():
    out = hwir.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "HWIR_V1.json"
    assert doc["schema"] == "hawking.future.hwir.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["lowered"]["validate"]["ok"] is True
    assert doc["proofs"]["round_trip_equal"] is True
    assert doc["proofs"]["dense_source_rejected"] is True
    assert doc["proofs"]["dangling_edge_rejected"] is True
    assert len(doc["hwir_hypotheses"]) == 15
    assert len(doc["backend_neutral_primitives"]) == 17
    assert doc["not_an_fpga_backend"] is True


def test_selftest_aliases_build():
    assert hwir.selftest is hwir.build or callable(hwir.selftest)
    out = hwir.selftest()
    assert out.name == "HWIR_V1.json"


def test_round_trip_is_byte_stable():
    graph = hwir.from_organ_map(hwir.REPO / hwir.FLASH_ORGAN_MAP, "expert_bank")
    blob1 = graph.to_json()
    blob2 = hwir.HwirGraph.from_json(blob1).to_json()
    assert blob1 == blob2
    assert blob1.encode("utf-8") == blob2.encode("utf-8")
    assert "recorded_at" not in blob1
    assert "generated_at" not in blob1
    parsed = json.loads(blob1)
    assert parsed["fingerprint"] == graph.fingerprint()
    # Re-dumping the loaded dict with sorted keys matches the canonical blob.
    body = {k: v for k, v in parsed.items() if k != "fingerprint"}
    rebuilt = json.dumps(
        {**body, "fingerprint": parsed["fingerprint"]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert rebuilt == blob1


def test_from_organ_map_lowers_real_flash_organ_and_validates():
    graph = hwir.from_organ_map(hwir.REPO / hwir.FLASH_ORGAN_MAP, "expert_bank")
    report = hwir.validate(graph)
    assert report.ok, report.errors
    assert graph.organ == "expert_bank"
    assert graph.model == "flash-next"
    kinds = {n.kind for n in graph.nodes}
    assert "compute" in kinds
    assert "memory" in kinds
    assert "representation-decoder" in kinds
    assert "reduction" in kinds
    assert "dma-transport" in kinds
    assert graph.qualification == "STATIC_ONLY"
    mem = next(n for n in graph.nodes if n.kind == "memory")
    assert mem.per_token_transfer is False
    assert "no_weight_body" in (mem.resident_weight_policy or "")
    assert mem.dense_weight_materialization is False
    assert not any(n.assumes_source_tensor_identity for n in graph.nodes)


def test_from_organ_map_state_organ_has_owner():
    graph = hwir.from_organ_map(
        hwir.REPO / hwir.FLASH_ORGAN_MAP, "deltanet_persistent_state"
    )
    report = hwir.validate(graph)
    assert report.ok, report.errors
    states = [n for n in graph.nodes if n.kind == "state"]
    assert states
    assert all(n.owner for n in states)
    kinds = {n.kind for n in graph.nodes}
    assert "persistent-pipeline" in kinds
    assert "state" in kinds


def test_from_organ_map_qwen_mlp_is_packed_not_dense():
    graph = hwir.from_organ_map(hwir.REPO / hwir.QWEN_ORGAN_MAP, "mlp_gate_up_down")
    report = hwir.validate(graph)
    assert report.ok, report.errors
    compute = next(n for n in graph.nodes if n.kind == "compute")
    assert "low-bit" in compute.mapping.lower()
    assert compute.physical.arithmetic_width == "packed_low_bit"


def test_negative_control_rejects_dense_source_rematerialization():
    graph = hwir.graph_dense_source_rematerialization()
    report = hwir.validate(graph)
    assert report.ok is False
    assert "DENSE_WEIGHT_MATERIALIZATION" in report.codes()
    assert "SOURCE_TENSOR_IDENTITY" in report.codes()
    # Structurally connected: refusal is semantic, not a missing-node accident.
    assert "DANGLING_EDGE" not in report.codes()


def test_negative_control_rejects_dangling_edge():
    graph = hwir.graph_dangling_edge()
    report = hwir.validate(graph)
    assert report.ok is False
    assert "DANGLING_EDGE" in report.codes()
    ghost = [e for e in report.errors if e["code"] == "DANGLING_EDGE"]
    assert any("missing.src" in e["message"] or "missing.dst" in e["message"] for e in ghost)


def test_validate_rejects_unowned_state():
    report = hwir.validate(hwir.graph_state_without_owner())
    assert report.ok is False
    assert "STATE_NO_OWNER" in report.codes()


def test_validate_rejects_resource_over_budget():
    report = hwir.validate(hwir.graph_over_budget())
    assert report.ok is False
    assert "RESOURCE_OVER_BUDGET" in report.codes()


def test_validate_rejects_type_mismatched_edge():
    report = hwir.validate(hwir.graph_type_mismatch())
    assert report.ok is False
    assert "TYPE_MISMATCH" in report.codes()


def test_legal_no_dense_rematerialization_mapping_passes():
    graph = hwir.HwirGraph(
        model="legal",
        organ="decoder",
        nodes=[
            hwir.HwirNode(
                id="mem",
                kind="memory",
                primitive="StationaryRepresentation",
                mapping="packed_native resident shards; no_dense_rematerialization",
                outputs={"out": "compact_representation_fragment"},
                lifetime="persistent",
                per_token_transfer=False,
            ),
            hwir.HwirNode(
                id="dec",
                kind="representation-decoder",
                primitive="FusedDecodeCompute",
                mapping="native decode; no dense rematerialization",
                inputs={"in": "compact_representation_fragment"},
                outputs={"out": "activation"},
            ),
        ],
        edges=[
            hwir.HwirEdge(
                id="e",
                src="mem",
                src_port="out",
                dst="dec",
                dst_port="in",
                frame_kind="compact_representation_fragment",
            )
        ],
    )
    report = hwir.validate(graph)
    assert report.ok, report.errors


def test_in_transit_unpack_compact_to_activation_is_legal():
    graph = hwir.HwirGraph(
        model="legal",
        organ="unpack",
        nodes=[
            hwir.HwirNode(
                id="mem",
                kind="memory",
                outputs={"out": "compact_representation_fragment"},
            ),
            hwir.HwirNode(
                id="cmp",
                kind="compute",
                primitive="TiledProjection",
                inputs={"in": "activation"},
                outputs={"out": "activation"},
            ),
        ],
        edges=[
            hwir.HwirEdge(
                id="e",
                src="mem",
                src_port="out",
                dst="cmp",
                dst_port="in",
                frame_kind="compact_representation_fragment",
                in_transit_transform="unpack",
            )
        ],
    )
    report = hwir.validate(graph)
    assert report.ok, report.errors


def test_kind_aliases_canonicalize():
    node = hwir.HwirNode(id="x", kind="dma_transport", outputs={"out": "partial reduction"})
    assert node.kind == "dma-transport"
    assert node.outputs["out"] == "partial_reduction"


def test_from_organ_map_unknown_organ_raises():
    try:
        hwir.from_organ_map(hwir.REPO / hwir.FLASH_ORGAN_MAP, "not_an_organ")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "not_an_organ" in str(exc)


def test_receipt_has_no_hardware_claim_fields_populated():
    out = hwir.build()
    doc = load_json(out)
    # Sidecar must not smuggle a measured hardware number into a forbidden field.
    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in {"tps", "token_ns", "gpu_ns", "joules_per_token", "bandwidth_gbps", "wall_ns", "dispatch_ns"}:
                    assert not isinstance(v, (int, float)), f"{path}.{k}={v}"
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)


# ---------------------------------------------------------------------------
# Pre-board stack. Mutation target for the overflow budget is REFUSAL_BUDGET.
# ---------------------------------------------------------------------------

# Declared compiler ceiling used by the refusal test. Mutation check raises
# these numbers so the over-wide engine fits; the test must then FAIL.
REFUSAL_BUDGET = {"LUT": 4096, "DSP": 8, "BRAM": 2, "URAM": 0, "hbm_channels": 1}


def test_qgemv_lowers_and_simulates_end_to_end():
    kernel = hwir.canonical_qgemv_kernel()
    device = hwir.synthetic_u50_class()
    graph = hwir.from_qgemv(kernel, device)
    report = hwir.validate(graph)
    assert report.ok, report.errors
    assert graph.organ == "qgemv"
    kinds = {n.kind for n in graph.nodes}
    assert "compute" in kinds
    assert "memory" in kinds
    assert "representation-decoder" in kinds
    assert "reduction" in kinds
    assert "dma-transport" in kinds
    mem = next(n for n in graph.nodes if n.kind == "memory")
    assert mem.per_token_transfer is False
    assert mem.primitive == "StationaryRepresentation"
    assert mem.backed_identity
    cmp = next(n for n in graph.nodes if n.kind == "compute")
    assert cmp.primitive == "TiledProjection"
    assert cmp.physical.arithmetic_width == "packed_low_bit"
    assert not any(n.assumes_source_tensor_identity for n in graph.nodes)
    assert not any(n.dense_weight_materialization for n in graph.nodes)

    doc = hwir.run_qgemv_preboard(kernel, device)
    hwir.assert_no_hardware_measured(doc)
    assert doc["prehardware"] is True
    assert doc["hardware_measured"] is False
    assert doc["evidence_tier"] == "STATIC"
    assert doc["validate"]["ok"] is True
    sim = doc["functional_sim"]
    assert sim["ok"] is True
    assert sim["evidence_tier"] == "FUNCTIONAL_SIM"
    assert sim["engine_symbol"] == "qgemv"
    assert sim["engine"] == "tools.future.fpga_engines.qgemv"
    assert sim["matches_expected"] is True
    assert sim["y"] == [10.0, 1.5]
    est = doc["resource_estimate"]
    assert est["evidence_tier"] == "STATIC"
    assert est["kind"] == "RESOURCE_ESTIMATE"
    assert "ESTIMATE" in est["note"]
    assert doc["resource_fit"]["ok"] is True
    cycles = doc["cycle_approx"]
    assert cycles["evidence_tier"] == "CYCLE_APPROX"
    assert isinstance(cycles["modelled_cycles"], int)
    assert cycles["modelled_cycles"] > 0
    assert cycles["seconds"] is None
    assert cycles["clock_hz"] == "UNKNOWN"
    assert cycles["conversion_to_seconds"] == "REFUSED"
    hbm = doc["hbm_traffic"]
    assert hbm["evidence_tier"] == "COST_MODEL"
    assert hbm["weights_resident"] is True
    xfer = doc["host_device_transfer"]
    assert xfer["evidence_tier"] == "COST_MODEL"
    assert "bandwidth_gbps" not in xfer
    part = doc["partition"]
    assert part["evidence_tier"] == "COST_MODEL"
    assert part["fpga_rows"] == kernel.M
    assert part["transport_policy"] == "activations_and_partial_reductions_only"
    tiers = hwir.collect_evidence_tiers(doc)
    assert tiers <= set(hwir.EVIDENCE_TIERS)
    assert "HARDWARE_MEASURED" not in tiers
    for required in ("STATIC", "FUNCTIONAL_SIM", "COST_MODEL", "CYCLE_APPROX"):
        assert required in tiers


def test_functional_sim_calls_fpga_engines_qgemv(monkeypatch):
    import tools.future.fpga_engines as fe

    calls = []
    real = fe.qgemv

    def wrapped(*args, **kwargs):
        calls.append(("qgemv", args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(fe, "qgemv", wrapped)
    sim = hwir.simulate_qgemv_functional()
    assert calls, "simulate_qgemv_functional must invoke fpga_engines.qgemv (an import is not a call site)"
    assert calls[0][0] == "qgemv"
    assert sim["engine_symbol"] == "qgemv"
    assert sim["matches_expected"] is True


def test_from_qgemv_calls_physical_primitives_instantiate(monkeypatch):
    import tools.future.physical_primitives as pp

    names = []
    real = pp.instantiate

    def wrapped(name, *args, **kwargs):
        names.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(pp, "instantiate", wrapped)
    graph = hwir.from_qgemv(hwir.canonical_qgemv_kernel())
    assert names, "from_qgemv must call physical_primitives.instantiate"
    assert "TiledProjection" in names
    assert "FusedDecodeCompute" in names
    assert "StationaryRepresentation" in names
    assert "SemanticTransportEdge" in names
    assert "CollectiveRegion" in names
    assert all(n.backed_identity for n in graph.nodes)


def test_no_code_path_emits_hardware_measured():
    with pytest.raises(hwir.IllegalEvidenceTier):
        hwir.emit_evidence("HARDWARE_MEASURED", {"ok": True})
    with pytest.raises(hwir.IllegalEvidenceTier):
        hwir.emit_evidence("REAL_HARDWARE", {"ok": True})
    with pytest.raises(hwir.IllegalEvidenceTier):
        hwir.HwirNode(id="x", kind="compute", evidence_tier="HARDWARE_MEASURED")
    doc = hwir.run_qgemv_preboard()
    hwir.assert_no_hardware_measured(doc)
    assert "HARDWARE_MEASURED" not in hwir.collect_evidence_tiers(doc)
    assert doc.get("hardware_measured") is False
    # Public emitters used by the preboard path.
    kernel = hwir.canonical_qgemv_kernel()
    device = hwir.synthetic_u50_class()
    for report in (
        hwir.estimate_qgemv_resources(kernel),
        hwir.fit_kernel_to_device(kernel, device),
        hwir.simulate_qgemv_functional(kernel),
        hwir.model_hbm_traffic(kernel, device),
        hwir.model_host_device_transfer(kernel, device),
        hwir.approximate_cycles(kernel, device),
        hwir.partition_qgemv(kernel, device),
        hwir.floorplan_hints(kernel, device),
        device.to_dict(),
        kernel.to_dict(),
        *hwir.lower_hwir_all(hwir.from_qgemv(kernel, device)).values(),
        hwir.lower_qgemv_targets(kernel, device),
    ):
        hwir.assert_no_hardware_measured(report)
        assert report["evidence_tier"] in hwir.EVIDENCE_TIERS
        assert report["hardware_measured"] is False


def test_resource_estimator_refuses_kernel_exceeding_device_budget():
    kernel = hwir.overflow_probe_kernel()
    used = hwir.estimate_qgemv_resources(kernel)["used"]
    # Engine width is independent of the declared ceiling. Mutation raises
    # REFUSAL_BUDGET, not this figure: the test must then fail at pytest.raises.
    assert used["DSP"] == kernel.mac_lanes * kernel.tile_m == 1024
    device = hwir.synthetic_device(
        lut=REFUSAL_BUDGET["LUT"],
        dsp=REFUSAL_BUDGET["DSP"],
        bram=REFUSAL_BUDGET["BRAM"],
        uram=REFUSAL_BUDGET["URAM"],
        hbm_channels=REFUSAL_BUDGET["hbm_channels"],
    )
    with pytest.raises(hwir.ResourceOverBudget) as exc:
        hwir.fit_kernel_to_device(kernel, device)
    assert "DSP" in exc.value.overflow
    graph = hwir.from_qgemv(kernel, device)
    report = hwir.validate(graph)
    assert report.ok is False
    assert "RESOURCE_OVER_BUDGET" in report.codes()


def test_cycle_approx_refuses_seconds_conversion():
    report = hwir.approximate_cycles(hwir.canonical_qgemv_kernel())
    assert report["seconds"] is None
    assert report["conversion_to_seconds"] == "REFUSED"
    with pytest.raises(hwir.UnmeasuredConversionError):
        hwir.cycles_to_seconds(report["modelled_cycles"], clock_hz=300_000_000)


def test_partitioner_keeps_weights_resident_and_splits_on_hbm_capacity():
    kernel = hwir.QGemvKernel(
        M=32,
        K=64,
        weight_bits=4,
        group_size=16,
        mac_lanes=4,
        tile_m=2,
    )
    # Engine fits; HBM capacity does not hold every resident row.
    bytes_per_row = (kernel.weight_bytes() + kernel.scale_bytes()) // kernel.M
    cap = bytes_per_row * 10
    device = hwir.synthetic_device(
        lut=1_000_000,
        dsp=10_000,
        bram=2_000,
        uram=100,
        hbm_channels=8,
        hbm_capacity_bytes=cap,
    )
    part = hwir.partition_qgemv(kernel, device)
    assert part["fpga_rows"] == 10
    assert part["host_rows"] == 22
    assert part["weights_resident"] is True
    assert part["transfer"]["weights_resident"] is True
    assert part["transfer"]["bytes_h2c"] == kernel.activation_in_bytes()
    assert part["transfer"]["bytes_c2h"] == 10 * 4
    assert "bandwidth_gbps" not in part["transfer"]
    note = part["apple_fpga_prior"]["note"]
    assert "COST_MODEL" in note
    assert "not a measurement" in note
    assert "not applied as physics" in note


def test_synthetic_u50_profile_is_declared_not_measured():
    profile = hwir.synthetic_u50_class().to_dict()
    assert profile["declared_not_measured"] is True
    assert profile["origin"] == "SYNTHETIC_U50_CLASS_DECLARED_NOT_A_BOARD"
    assert profile["evidence_tier"] == "STATIC"
    assert profile["hardware_measured"] is False
    assert "bandwidth_gbps" not in profile


def test_organ_map_loads_when_receipts_are_sparse_absent():
    # This worktree does not materialize receipts/; from_organ_map must git-show.
    graph = hwir.from_organ_map(hwir.FLASH_ORGAN_MAP, "expert_bank")
    assert hwir.validate(graph).ok
    assert graph.source_receipt == hwir.FLASH_ORGAN_MAP
