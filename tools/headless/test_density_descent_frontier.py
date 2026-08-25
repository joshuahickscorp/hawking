"""N037 two coupled frontiers: generated from receipts, citations resolve."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from density_descent_frontier import (  # noqa: E402
    ABSENT,
    DENSITY_LADDER,
    GENERATOR,
    RECEIPT,
    REQUIRED_INPUTS,
    SCHEMA,
    build,
    citation_exists,
    numeric,
    unresolved_citations,
    write,
)

DOCS = None


def docs() -> dict:
    global DOCS
    if DOCS is None:
        built = build()
        write(built)
        DOCS = built
    return DOCS


def _disk() -> dict:
    assert RECEIPT.is_file(), (
        f"missing {RECEIPT} — run python3 tools/headless/density_descent_frontier.py"
    )
    return json.loads(RECEIPT.read_text())


def test_generator_writes_schema_and_cpu_discipline():
    d = docs()
    on_disk = _disk()
    assert on_disk["schema"] == SCHEMA
    assert d["schema"] == SCHEMA
    assert on_disk["generated_by"] == GENERATOR
    assert on_disk["hand_authored"] is False
    assert on_disk["did_not_touch_gpu"] is True
    assert on_disk["did_not_run_cargo_or_metal_benchmarks"] is True
    assert on_disk["did_not_load_a_model"] is True
    assert on_disk["did_not_mutate_parent"] is True
    assert on_disk["did_not_rederive_roofs"] is True
    assert on_disk["unmeasured_is_absent"] is True


def test_required_inputs_are_present():
    d = docs()
    paths = [r["path"] for r in d["required_inputs"]]
    assert paths == list(REQUIRED_INPUTS)
    for row in d["required_inputs"]:
        assert row["present"] is True
        assert row["required"] is True
        assert citation_exists(row["path"]), row["path"]


def test_every_citation_resolves_on_disk_or_in_git():
    missing = unresolved_citations(docs())
    assert missing == [], f"frontier citations that do not exist: {missing}"
    missing_disk = unresolved_citations(_disk())
    assert missing_disk == [], missing_disk


def test_citation_walker_fails_a_bogus_receipt():
    fake = {
        "citations": ["receipts/headless/DOES_NOT_EXIST_N037.json"],
        "source": "receipts/headless/ALSO_NOT_A_RECEIPT.json",
    }
    bad = unresolved_citations(fake)
    assert "receipts/headless/DOES_NOT_EXIST_N037.json" in bad
    assert "receipts/headless/ALSO_NOT_A_RECEIPT.json" in bad
    assert citation_exists("receipts/headless/BYTES_FRONTIER.json") is True
    assert citation_exists("receipts/headless/SHARED_BASIS_KERNEL.json") is True
    assert citation_exists("receipts/headless/DISPATCH_LEDGER.json") is True
    assert citation_exists("receipts/headless/ORGAN_ROOF_LEDGER.json") is True


def test_both_frontiers_are_named():
    d = docs()
    coh = d["COHERENCE_FRONTIER"]
    exe = d["EXECUTION_FRONTIER"]
    assert coh["name"]
    assert exe["name"]
    assert coh["candidate_id"] == coh["name"]
    assert exe["candidate_id"] == exe["name"]
    assert coh["coherent"] is True
    assert exe["coherent"] is True
    assert coh["axis"] == "lowest_coherent_complete_ebpw"
    assert exe["axis"] == "lowest_coherent_complete_token_ns"


def test_coherence_frontier_is_q2f_2_25():
    d = docs()
    coh = d["COHERENCE_FRONTIER"]
    assert coh["name"] == "q2f_g64"
    bpw = numeric(coh["complete_ebpw"])
    assert bpw == 2.25
    q2f = next(c for c in d["candidates"] if c["id"] == "q2f_g64")
    assert q2f["coherent"] is True
    assert q2f["on_coherence_frontier"] is True
    assert q2f["composition_rung"]["highest_rung_reached"] == "coherent_generation"


def test_execution_frontier_is_q2f_27_55ms_among_coherent():
    d = docs()
    exe = d["EXECUTION_FRONTIER"]
    assert exe["name"] == "q2f_g64"
    ns = numeric(exe["COMPLETE_TOKEN_NS"])
    assert ns == 27_547_874
    assert abs(exe["ms"] - 27.55) < 0.01
    q2f = next(c for c in d["candidates"] if c["id"] == "q2f_g64")
    assert q2f["on_execution_frontier"] is True
    spread = (q2f["COMPLETE_TOKEN_NS"].get("spread") or {})
    assert spread.get("n") == 7
    assert spread["min"] <= spread["median"] <= spread["max"]


def test_faster_incoherent_bodies_do_not_take_execution_frontier():
    d = docs()
    exe_ns = numeric(d["EXECUTION_FRONTIER"]["COMPLETE_TOKEN_NS"])
    by_id = {c["id"]: c for c in d["candidates"]}
    binary = by_id["binary_g64"]
    k2 = by_id["shared_basis_k2"]
    assert binary["coherent"] is False
    assert k2["coherent"] is False
    b_ns = numeric(binary["COMPLETE_TOKEN_NS"])
    k_ns = numeric(k2["COMPLETE_TOKEN_NS"])
    assert b_ns is not None and b_ns < exe_ns
    assert k_ns is not None and k_ns < exe_ns
    assert binary["on_execution_frontier"] is False
    assert k2["on_execution_frontier"] is False
    faster_ids = {r["id"] for r in d["faster_than_execution_frontier_but_incoherent"]}
    assert "binary_g64" in faster_ids
    assert "shared_basis_k2" in faster_ids


def test_required_candidates_sit_on_both_axes():
    d = docs()
    by_id = {c["id"]: c for c in d["candidates"]}
    required = (
        "q2f_g64",
        "binary_g64",
        "ternary_5in8_g64",
        "shared_basis_k2",
        "shared_basis_k8",
        "binary_csr_2pct",
    )
    for cid in required:
        assert cid in by_id, cid
        c = by_id[cid]
        assert "complete_ebpw" in c, cid
        assert "active_bytes_per_token" in c, cid
        assert "COMPLETE_TOKEN_NS" in c, cid
        assert "composition_rung" in c, cid
        assert isinstance(c["coherent"], bool), cid
        # Unmeasured cells are ABSENT with a reason, never a silent 0.
        for key in ("complete_ebpw", "active_bytes_per_token", "COMPLETE_TOKEN_NS"):
            cell = c[key]
            assert isinstance(cell, dict), (cid, key)
            if cell.get("kind") == ABSENT:
                assert cell.get("value") is None, f"{cid}.{key} ABSENT but has a value"
                assert cell.get("absent_reason"), f"{cid}.{key} ABSENT without a reason"
            else:
                assert cell.get("source"), f"{cid}.{key} numeric without a source"
                assert citation_exists(cell["source"]), cell["source"]
        if c.get("source_receipt"):
            assert citation_exists(c["source_receipt"]), c["source_receipt"]


def test_k8_token_ns_is_absent_not_scaled():
    k8 = next(c for c in docs()["candidates"] if c["id"] == "shared_basis_k8")
    ns = k8["COMPLETE_TOKEN_NS"]
    assert ns["kind"] == ABSENT
    assert ns["value"] is None
    assert "k=2" in (ns.get("absent_reason") or "").lower() or "K=2" in (ns.get("absent_reason") or "")
    ebpw = k8["complete_ebpw"]
    assert ebpw["kind"] == ABSENT
    assert ebpw["value"] is None
    assert "counterfactual" in json.dumps(ebpw).lower() or "untested" in (ebpw.get("absent_reason") or "").lower()


def test_density_ladder_coordinates_are_annotations_not_pass_literals():
    d = docs()
    ladder = d["density_ladder"]
    coords = [r["coordinate_bpw"] for r in ladder["coordinates"]]
    assert coords == list(DENSITY_LADDER)
    for r in ladder["coordinates"]:
        assert r["not_an_obligation"] is True
        assert r["not_a_pass_literal"] is True
        assert r["status"] in {"REACHED_COHERENT", "OCCUPIED_INCOHERENT", "UNREACHED"}
        assert "PASS" not in r["status"]
        assert "FAIL" not in r["status"]
    c225 = next(r for r in ladder["coordinates"] if r["coordinate_bpw"] == 2.25)
    assert c225["status"] == "REACHED_COHERENT"
    assert any(o["id"] == "q2f_g64" and o["coherent"] for o in c225["occupied_by"])
    c125 = next(r for r in ladder["coordinates"] if r["coordinate_bpw"] == 1.25)
    assert c125["status"] == "OCCUPIED_INCOHERENT"
    assert any(o["id"] == "binary_g64" and not o["coherent"] for o in c125["occupied_by"])
    c053 = next(r for r in ladder["coordinates"] if r["coordinate_bpw"] == 0.53)
    assert c053["status"] == "OCCUPIED_INCOHERENT"
    assert any(o["id"] == "shared_basis_k2" for o in c053["occupied_by"])
    c01 = next(r for r in ladder["coordinates"] if r["coordinate_bpw"] == 0.1)
    assert c01["status"] == "UNREACHED"
    assert c01["occupied_by"] == []


def test_missing_optional_healing_receipts_are_absent_not_cited():
    d = docs()
    cites = set()
    for c in d["candidates"]:
        cites.update(c.get("citations") or [])
        if c.get("source_receipt"):
            cites.add(c["source_receipt"])
    cites.update(d.get("citations") or [])
    for row in d["optional_inputs"]:
        if row["present"]:
            continue
        assert row["path"] not in cites, (
            f"missing optional {row['path']} was cited; the citation test would then fail"
        )
        assert row["absent_reason"]
    by_id = {c["id"]: c for c in d["candidates"]}
    for cid, path in (
        ("shared_basis_coherent", "receipts/headless/SHARED_BASIS_COHERENT.json"),
        ("binary_healing", "receipts/headless/BINARY_HEALING.json"),
    ):
        if path in {r["path"] for r in d["optional_inputs"] if r["present"]}:
            continue
        c = by_id[cid]
        assert c["complete_ebpw"]["kind"] == ABSENT
        assert c["COMPLETE_TOKEN_NS"]["kind"] == ABSENT
        assert c.get("not_cited_because_missing") is True
        assert path not in (c.get("citations") or [])


def test_roofs_are_copied_not_rederived():
    d = docs()
    roofs = d["roofs"]
    assert roofs["did_not_rederive"] is True
    assert roofs["source"] == "receipts/headless/ORGAN_ROOF_LEDGER.json"
    three = roofs["three_roofs"]
    assert three["never_collapsed"] is True
    assert three["DEVICE_THEORETICAL"]["value"] == 819.0
    assert three["DEVICE_MEASURED_SUSTAINED"]["value"] == 778.8
    assert three["MODEL_REACHABLE"]["value"] > 700
    assert d["organ_density"]["organs"]
    mlp = next(o for o in d["organ_density"]["organs"] if o["organ"] == "mlp_gate_up")
    assert numeric(mlp["recoverable_token_ns"]) is not None
    assert mlp["recoverable_token_ns"]["source"] == "receipts/headless/ORGAN_ROOF_LEDGER.json"


def test_k2_uses_fused_n033_not_two_pass_n032():
    k2 = next(c for c in docs()["candidates"] if c["id"] == "shared_basis_k2")
    ns = numeric(k2["COMPLETE_TOKEN_NS"])
    assert ns == 24_554_625
    two = k2["superseded_two_pass"]["complete_token_ns"]
    assert two is not None and two > ns
    assert k2["kernel_competent"] is True
    assert k2["composition_rung"]["died_at"] == "held_out_activation"
    assert k2["source_receipt"] == "receipts/headless/SHARED_BASIS_KERNEL.json"


def test_ternary_died_at_complete_token_and_is_not_coherent():
    t = next(c for c in docs()["candidates"] if c["id"] == "ternary_5in8_g64")
    assert t["coherent"] is False
    assert t["composition_rung"]["died_at"] == "complete_token"
    assert numeric(t["complete_ebpw"]) == 1.85
    assert numeric(t["COMPLETE_TOKEN_NS"]) is not None
    assert numeric(t["COMPLETE_TOKEN_NS"]) > 27_547_874
