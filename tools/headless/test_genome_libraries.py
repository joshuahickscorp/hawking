"""N034 genome libraries: generated from receipts, citations resolve, queue recovered."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from genome_libraries import (  # noqa: E402
    FRONTIER_NAMES,
    KERNEL_OUT,
    ODYSSEY_OUT,
    ORGAN_OUT,
    ORGANS,
    REPR_FAMILIES,
    REPR_OUT,
    SCHEMA_KERNEL,
    SCHEMA_ODYSSEY,
    SCHEMA_ORGAN,
    SCHEMA_REPR,
    build_all,
    citation_exists,
    unique_citations,
    unresolved_citations,
    write_all,
)

DOCS = None


def docs() -> dict:
    global DOCS
    if DOCS is None:
        built = build_all()
        write_all(built)
        DOCS = built
    return DOCS


def _disk(path: Path) -> dict:
    assert path.is_file(), f"missing generated receipt {path}"
    return json.loads(path.read_text())


def test_generator_writes_four_receipts_with_schemas():
    d = docs()
    mapping = {
        "organ": (ORGAN_OUT, SCHEMA_ORGAN),
        "kernel": (KERNEL_OUT, SCHEMA_KERNEL),
        "representation": (REPR_OUT, SCHEMA_REPR),
        "odyssey": (ODYSSEY_OUT, SCHEMA_ODYSSEY),
    }
    for key, (path, schema) in mapping.items():
        on_disk = _disk(path)
        assert on_disk["schema"] == schema
        assert d[key]["schema"] == schema
        assert on_disk["generated_by"] == "tools/headless/genome_libraries.py"
        assert on_disk["hand_authored"] is False
        assert on_disk["did_not_touch_gpu"] is True
        assert on_disk["did_not_load_a_model"] is True
        assert on_disk["did_not_mutate_parent"] is True


def test_every_citation_resolves_on_disk_or_in_git():
    """Fails if a library entry cites a receipt that does not exist.

    Sparse-checkout: a missing working-tree file is not evidence of absence.
    citation_exists checks git cat-file as well as Path.is_file.
    """
    missing: dict[str, list[str]] = {}
    for key, doc in docs().items():
        bad = unresolved_citations(doc)
        if bad:
            missing[key] = bad
    assert missing == {}, f"library citations that do not exist: {missing}"


def test_citation_walker_fails_a_bogus_receipt():
    fake = {
        "citations": ["receipts/headless/DOES_NOT_EXIST_N034.json"],
        "source": "receipts/headless/ALSO_NOT_A_RECEIPT.json",
    }
    bad = unresolved_citations(fake)
    assert "receipts/headless/DOES_NOT_EXIST_N034.json" in bad
    assert "receipts/headless/ALSO_NOT_A_RECEIPT.json" in bad
    assert citation_exists("receipts/headless/ORGAN_BANDWIDTH.json") is True


def test_organ_library_has_one_genome_per_contract_organ():
    organs = docs()["organ"]["organs"]
    names = [o["organ"] for o in organs]
    assert names == list(ORGANS)
    for o in organs:
        assert o.get("semantic_function")
        assert o.get("dimensions")
        assert o.get("best_representation_family")
        assert o.get("best_complete_ebpw")
        assert o.get("best_active_ebpw")
        assert o.get("kernel_id")
        assert o.get("measured_bandwidth_gb_s")
        assert o.get("recoverable_token_ns")
        assert o.get("token_ns_contribution")
        assert o.get("capability_sensitivity")
        assert "negative_science" in o
        assert o.get("citations"), f"{o['organ']} has no citations"
        # Unmeasured capability must be ABSENT, never a guessed Doctor score.
        live = o["capability_sensitivity"]["live_capability"]
        assert live["kind"] == "ABSENT"
        assert live["value"] is None


def test_organ_numbers_are_cited_not_guessed():
    mlp = next(o for o in docs()["organ"]["organs"] if o["organ"] == "mlp_gate_up")
    bw = mlp["measured_bandwidth_gb_s"]
    assert bw["kind"] == "CITED"
    assert bw["source"] == "receipts/headless/ORGAN_BANDWIDTH.json"
    assert isinstance(bw["value"], (int, float)) and bw["value"] > 0
    rec = mlp["recoverable_token_ns"]
    assert rec["kind"] in {"CITED", "DERIVED"}
    assert "ORGAN_ROOF_LEDGER" in rec["source"]
    sampling = next(o for o in docs()["organ"]["organs"] if o["organ"] == "sampling")
    assert sampling["best_complete_ebpw"]["kind"] == "ABSENT"
    assert sampling["best_complete_ebpw"]["value"] is None


def test_kernel_library_covers_the_qualified_families():
    kernels = docs()["kernel"]["kernels"]
    names = {k["kernel_identity"] for k in kernels}
    required = {
        "q2f_group64_matvec",
        "affine2_group32_matvec_geo_tpr64_tg128",
        "qwen80_add_residual_rmsnorm_tg",
        "qwen38_gated_delta_decode_vi_simd_ba_f4",
        "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
        "binary_g64_matvec_geo_c5120_tpr64_tg128",
    }
    missing = required - names
    assert not missing, missing
    for k in kernels:
        comp = k["competence"]
        assert comp["kind"] in {"CITED", "ABSENT"}
        if comp["kind"] == "ABSENT":
            assert comp.get("verdict") in (None, "ABSENT")
            assert comp.get("absent_reason")
        else:
            assert comp["verdict"] in {"DEFECTIVE", "SUSPECT", "CLEAR"}
        assert k.get("citations")
        assert k["machine_identity"]["chipset"] == "Apple M3 Ultra"


def test_widen_f4_parity_is_cited_from_deltanet_receipt():
    k = next(
        x
        for x in docs()["kernel"]["kernels"]
        if x["kernel_identity"] == "qwen38_gated_delta_decode_vi_simd_ba_f4"
    )
    assert k["parity"]["value"] == 0.0
    assert k["parity"]["source"] == "receipts/headless/DELTANET_ORGAN.json"
    meas = k["measurements"]["deltanet_widen_f4"]
    assert meas["token_ids_unchanged"] is True


def test_representation_library_covers_seven_families_and_the_bits_law():
    doc = docs()["representation"]
    ids = [f["family"] for f in doc["families"]]
    assert ids == list(REPR_FAMILIES)
    laws = doc["laws"]
    assert any("fewer stored bits" in (law.get("law") or "") for law in laws)
    ev = laws[0]["evidence"]
    assert ev["source"] == "receipts/headless/BYTES_FRONTIER.json"
    assert ev["who_moved"] == "binary_g64"
    ternary = next(f for f in doc["families"] if f["family"] == "ternary")
    assert ternary["fewer_bits_moved_ns"] is False
    binary = next(f for f in doc["families"] if f["family"] == "binary")
    assert binary["fewer_bits_moved_ns"] is True
    low = next(f for f in doc["families"] if f["family"] == "low_rank_sparse")
    assert low["design_verdict"] == "NOT_WORTH_BUILDING"
    req = low["kernel_requirement"]
    assert req["kind"] == "ABSENT"


def test_odyssey_queue_recovers_manifest_patients():
    doc = docs()["odyssey"]
    queue = doc["queue"]
    assert doc["n_patients"] == 14
    assert len(queue) == 14
    oxx = [p["oxx"] for p in queue]
    assert oxx[0] == "O000"
    assert oxx[-1] == "O013"
    o005 = next(p for p in queue if p["oxx"] == "O005")
    assert o005["canonical_source"] == "Qwen/Qwen3-30B-A3B"
    assert o005["n_completions"] > 0
    assert doc["did_not_invent_hf_repo_ids"] is True
    assert doc["did_not_modify_odyssey_state"] is True


def test_frontier_reconciliation_resolves_known_and_leaves_t5v4_unresolved():
    families = {f["shorthand"]: f for f in docs()["odyssey"]["frontier_families"]}
    assert set(families) == set(FRONTIER_NAMES)

    qwen = families["Qwen3.8"]
    assert qwen["identity_status"] == "RESOLVED"
    assert qwen["repository"] == "huihui-ai/Huihui-Qwen3.8-27B-abliterated"

    dsv = families["DeepSeek V4 Flash"]
    assert dsv["identity_status"] == "RESOLVED"
    assert dsv["repository"] == "deepseek-ai/DeepSeek-V4-Flash"
    assert dsv["revision"] == "60d8d70770c6776ff598c94bb586a859a38244f1"

    glm = families["GLM 5.x"]
    assert glm["identity_status"] == "RESOLVED"
    assert glm["repository"] == "zai-org/GLM-5.2"
    assert "O010" in glm["not_odyssey_glm45"]["glm45_oxx"]

    t5 = families["T5V4"]
    assert t5["identity_status"] == "UNRESOLVED"
    assert t5["repository"] is None
    assert t5["revision"] is None
    assert t5.get("canonical_source") is None
    assert "invent" in t5["notes"].lower() or "UNRESOLVED" in t5["notes"]

    kimi = families["Kimi K3"]
    assert kimi["identity_status"] == "RESOLVED"
    assert kimi["repository"] == "moonshotai/Kimi-K3"
    assert kimi["revision"] == "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"

    assert docs()["odyssey"]["unresolved_shorthands"] == ["T5V4"]


def test_libraries_cite_themselves_into_unique_receipts():
    for key, doc in docs().items():
        cites = unique_citations(doc)
        assert cites, f"{key} library has no citations"
        assert all(citation_exists(c) for c in cites)
