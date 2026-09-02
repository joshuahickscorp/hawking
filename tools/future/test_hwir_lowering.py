"""Pluggable HWIR lowering targets. Source artifacts only. PREHARDWARE."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from tools.future import hwir


VENDOR_NEEDLES = ("xilinx", "vitis", "alveo", "nvidia", "cuda")
GENERIC_BEGIN = "# === GENERIC LOWERING LAYER BEGIN ==="
GENERIC_END = "# === GENERIC LOWERING LAYER END ==="

_GENERIC_CALLABLES = (
    hwir.LoweringTarget,
    hwir.UnknownLoweringTarget,
    hwir.register_lowering_target,
    hwir.list_lowering_targets,
    hwir.get_lowering_target,
    hwir.lowering_target_manifests,
    hwir.lower_hwir,
    hwir.lower_hwir_all,
    hwir.lower_qgemv_targets,
    hwir._finalize_lowering,
    hwir._source_artifact,
    hwir.lowering_emitted_primitives,
    hwir.lowering_hole_primitives,
)


def _generic_layer_source() -> str:
    text = Path(hwir.__file__).read_text()
    begin = text.index(GENERIC_BEGIN)
    end = text.index(GENERIC_END)
    assert end > begin
    return text[begin:end]


def _probe_graph(primitive: str) -> hwir.HwirGraph:
    kind = hwir.PRIMITIVE_TO_NODE_KIND.get(primitive, "compute")
    if primitive == "hbm_memory_controller":
        kind = "memory"
    elif primitive in {"host_link_phy", "interrupt_doorbell"}:
        kind = "dma-transport"
    elif primitive in {"clock_generator", "dfx_region_wrapper"}:
        kind = "persistent-pipeline"
    elif primitive == "io_pinout_constraints":
        kind = "compute"
    node = hwir.HwirNode(
        id="probe",
        kind=kind,
        primitive=primitive,
        mapping="probe",
        outputs={"out": "activation"},
        owner="probe-owner" if kind == "state" else None,
        evidence_tier="STATIC",
    )
    return hwir.HwirGraph(
        model="probe",
        organ="probe",
        nodes=[node],
        notes=["probe graph for lowering-target honesty"],
    )


def _qgemv_graph() -> hwir.HwirGraph:
    return hwir.from_qgemv(hwir.canonical_qgemv_kernel(), hwir.synthetic_u50_class())


def test_at_least_two_registered_targets():
    ids = hwir.list_lowering_targets()
    assert len(ids) >= 2
    assert "hls_style" in ids
    assert "rust_hdl_style" in ids
    assert ids == tuple(sorted(ids))
    assert hwir.PREFERRED_LOWERING_TARGET is None
    for tid in ids:
        target = hwir.get_lowering_target(tid)
        assert isinstance(target, hwir.LoweringTarget)
        man = target.manifest()
        assert man["preferred"] is False
        assert man["toolchain_choice"] is None
        assert man["target_id"] == tid
        hwir.assert_no_hardware_measured(man)


def test_both_emitters_lower_the_same_graph_identically_at_the_interface():
    graph = _qgemv_graph()
    ids = hwir.list_lowering_targets()
    assert len(ids) >= 2
    results = [hwir.lower_hwir(graph, tid) for tid in ids]
    key_sets = [frozenset(r) for r in results]
    assert all(k == key_sets[0] for k in key_sets)
    assert {r["graph_fingerprint"] for r in results} == {graph.fingerprint()}
    assert {r["kind"] for r in results} == {"HWIR_LOWERING"}
    methods = [
        name
        for name in dir(hwir.LoweringTarget)
        if not name.startswith("_")
    ]
    for tid, result in zip(ids, results):
        target = hwir.get_lowering_target(tid)
        assert isinstance(target, hwir.LoweringTarget)
        for name in ("cannot_express", "emits", "emit_artifacts", "family",
                     "handwritten_hdl", "lower", "manifest", "supported_primitives",
                     "target_id"):
            assert callable(getattr(target, name))
        assert result["prehardware"] is True
        assert result["hardware_measured"] is False
        assert result["qualification"] == hwir.PREHARDWARE
        assert result["evidence_tier"] in hwir.EVIDENCE_TIERS
        assert result["preferred"] is False
        assert result["toolchain_choice"] is None
        assert result["target_id"] == tid
        assert result["artifacts"]
        for art in result["artifacts"]:
            assert art["kind"] == "SOURCE_ARTIFACT"
            assert art["prehardware"] is True
            assert art["hardware_measured"] is False
            assert art["evidence_tier"] == "STATIC"
            assert "PREHARDWARE" in art["body"]
        # Same public interface on every target.
        public = [n for n in dir(target) if not n.startswith("_")]
        assert set(methods) <= set(public)
    bundled = hwir.lower_hwir_all(graph)
    assert set(bundled) == set(ids)
    qdoc = hwir.lower_qgemv_targets()
    assert qdoc["preferred"] is None
    assert qdoc["toolchain_choice"] is None
    assert qdoc["graph_fingerprint"] == graph.fingerprint()


def test_each_emitter_enumerates_primitives_it_cannot_express():
    graph = _qgemv_graph()
    catalog = set(hwir.HARDWARE_PRIMITIVE_CATALOG)
    graph_prims = {n.primitive for n in graph.nodes if n.primitive}
    assert len(hwir.list_lowering_targets()) >= 2
    for tid in hwir.list_lowering_targets():
        target = hwir.get_lowering_target(tid)
        cannot = tuple(target.cannot_express())
        supported = tuple(target.supported_primitives())
        hdl = tuple(target.handwritten_hdl())
        assert cannot, f"{tid} cannot_express is empty"
        assert len(cannot) == len(set(cannot))
        assert set(supported).isdisjoint(cannot), f"{tid} claims and denies {set(supported) & set(cannot)}"
        missing = catalog - set(supported) - set(cannot)
        assert not missing, f"{tid} left catalog primitives undeclared: {sorted(missing)}"
        assert hdl, f"{tid} handwritten_hdl is empty"
        assert set(hdl) <= set(cannot), f"{tid} handwritten_hdl not subset of cannot_express"
        result = hwir.lower_hwir(graph, tid)
        assert tuple(result["cannot_express"]) == cannot
        assert tuple(result["handwritten_hdl"]) == hdl
        emitted = hwir.lowering_emitted_primitives(result)
        for prim in graph_prims & set(supported):
            assert prim in emitted, f"{tid} did not emit body for graph primitive {prim}"
        for prim in cannot:
            assert prim not in emitted, f"{tid} emitted {prim} while listing it as cannot_express"
        for prim in supported:
            probe = _probe_graph(prim)
            out = target.lower(probe)
            got = hwir.lowering_emitted_primitives(out)
            assert prim in got, (
                f"{tid} claims to support {prim} but probe lowering has no "
                f"HWIR_EMITTED:{prim}"
            )
        for prim in cannot:
            probe = _probe_graph(prim)
            out = target.lower(probe)
            got = hwir.lowering_emitted_primitives(out)
            holes = hwir.lowering_hole_primitives(out)
            assert prim not in got, f"{tid} emitted a body for unsupported {prim}"
            assert prim in holes, f"{tid} did not name hole {prim}"


def test_no_code_path_emits_hardware_measured_from_lowered_artifact():
    graph = _qgemv_graph()
    with pytest.raises(hwir.IllegalEvidenceTier):
        hwir.emit_evidence("HARDWARE_MEASURED", {"kind": "SOURCE_ARTIFACT"})
    for tid in hwir.list_lowering_targets():
        doc = hwir.lower_hwir(graph, tid)
        hwir.assert_no_hardware_measured(doc)
        assert "HARDWARE_MEASURED" not in hwir.collect_evidence_tiers(doc)
        assert doc["hardware_measured"] is False
        assert doc["evidence_tier"] != "HARDWARE_MEASURED"
        for art in doc["artifacts"]:
            hwir.assert_no_hardware_measured(art)
            assert art["hardware_measured"] is False
            assert art["evidence_tier"] != "HARDWARE_MEASURED"
    bundled = hwir.lower_qgemv_targets()
    hwir.assert_no_hardware_measured(bundled)
    assert "HARDWARE_MEASURED" not in hwir.collect_evidence_tiers(bundled)


def test_generic_lowering_layer_has_no_vendor_symbols():
    src = _generic_layer_source()
    lowered = src.lower()
    for needle in VENDOR_NEEDLES:
        assert needle not in lowered, (
            f"vendor symbol {needle!r} appears in the generic lowering layer"
        )
    # No vendor-keyed control flow: the needles are absent, so no branch can
    # key on them. Also inspect the public generic callables.
    for obj in _GENERIC_CALLABLES:
        piece = inspect.getsource(obj).lower()
        for needle in VENDOR_NEEDLES:
            assert needle not in piece, (
                f"vendor symbol {needle!r} in {getattr(obj, '__name__', obj)}"
            )
    # Backends (style families) also must not key on a vendor.
    for cls in (hwir.HlsStyleEmitter, hwir.RustHdlEmitter):
        piece = inspect.getsource(cls).lower()
        for needle in VENDOR_NEEDLES:
            assert needle not in piece, f"vendor symbol {needle!r} in {cls.__name__}"
    # File-level rg of the generic span (the required proof).
    hits = []
    for i, line in enumerate(src.splitlines(), 1):
        if re.search(r"xilinx|vitis|alveo|nvidia|cuda", line, re.I):
            hits.append(f"{i}:{line.strip()}")
    assert hits == []


def test_unknown_lowering_target_raises():
    with pytest.raises(hwir.UnknownLoweringTarget):
        hwir.get_lowering_target("not-a-target")
    with pytest.raises(hwir.UnknownLoweringTarget):
        hwir.lower_hwir(_qgemv_graph(), "not-a-target")


def test_empty_cannot_express_is_refused(monkeypatch):
    target = hwir.get_lowering_target("hls_style")
    monkeypatch.setattr(target, "cannot_express", lambda: ())
    with pytest.raises(ValueError, match="cannot_express is empty"):
        target.lower(_qgemv_graph())
