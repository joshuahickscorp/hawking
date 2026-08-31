"""HWIR v1 tests. Negative controls must actually refuse."""
import json

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
