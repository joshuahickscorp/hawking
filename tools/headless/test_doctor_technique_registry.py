"""N043 Doctor Technique Registry: hypotheses, Hawking receipts, campaign scars."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from doctor_technique_registry import (  # noqa: E402
    ALLOWED_VERDICT_STATUSES,
    DOCS,
    GENERATOR,
    LITERATURE_STATUS,
    RECEIPT,
    RELATED_NEGATIVE,
    REQUIRED_ENTRY_FIELDS,
    REQUIRED_TECHNIQUE_IDS,
    SCAR_BINARY,
    SCAR_LOWRANK,
    SCAR_SHARED_BASIS,
    SCAR_SPARSE,
    SCAR_TERNARY,
    SCHEMA,
    TESTED_NEGATIVE,
    UNTESTED,
    VERDICT_REQUIRES_RECEIPT,
    build,
    citation_exists,
    is_hawking_receipt_path,
    registry_errors,
    technique_field_errors,
    verdict_errors,
    write_receipt,
)

RECEIPT_DOC: dict | None = None


def receipt() -> dict:
    global RECEIPT_DOC
    if RECEIPT_DOC is None:
        RECEIPT_DOC = write_receipt(build())
    return RECEIPT_DOC


def _disk() -> dict:
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    return json.loads(RECEIPT.read_text())


def _by_id(doc: dict | None = None) -> dict[str, dict]:
    doc = doc or receipt()
    return {t["technique_identity"]["id"]: t for t in doc["techniques"]}


def _blank_entry(**overrides) -> dict:
    entry = {
        "technique_identity": {
            "id": "fake",
            "short_name": "Fake",
            "s026_name": "Fake",
            "s026_family": "DOC-DIAGNOSIS",
            "literature_status": LITERATURE_STATUS,
            "not_authority": True,
        },
        "source_paper": {"title": "Fake", "approx_date": "2024-01"},
        "claimed_mechanism": "x",
        "architecture_assumptions": "x",
        "training_calibration_runtime": "x",
        "storage_vs_execution": "x",
        "expected_useful_organs": ["mlp_down"],
        "expected_physical_win": "x",
        "risks": ["x"],
        "licensing_provenance": {"do_not_copy_third_party_code": True},
        "hawking_experiment_mapping": {"campaign_mechanism_overlap": "none"},
        "current_verdict": {
            "status": UNTESTED,
            "literature_is": LITERATURE_STATUS,
            "hawking_receipts": [],
            "cheapest_hawking_experiment": {"id": "HX-FAKE", "name": "n"},
        },
    }
    entry.update(overrides)
    return entry


def test_receipt_written_with_schema_and_discipline_flags():
    doc = receipt()
    disk = _disk()
    assert disk["schema"] == SCHEMA
    assert doc["schema"] == SCHEMA
    assert doc["generated_by"] == GENERATOR
    assert doc["phase"] == "A"
    assert doc["family"] == "DOC-DIAGNOSIS"
    assert doc["hand_authored"] is False
    assert doc["literature_is"] == LITERATURE_STATUS
    assert doc["literature_is_not_authority"] is True
    assert doc["did_not_load_a_model"] is True
    assert doc["did_not_touch_gpu"] is True
    assert doc["did_not_run_cargo_or_metal_benchmarks"] is True
    assert doc["did_not_mutate_parent"] is True
    assert doc["this_is_the_registry_not_the_experiments"] is True
    assert RECEIPT.name == "DOCTOR_TECHNIQUE_REGISTRY.json"


def test_all_required_s026_mechanisms_are_registered():
    ids = list(_by_id())
    missing = [i for i in REQUIRED_TECHNIQUE_IDS if i not in ids]
    assert not missing, missing
    assert receipt()["n_techniques"] >= 15
    assert ids[:15] == list(REQUIRED_TECHNIQUE_IDS) or set(REQUIRED_TECHNIQUE_IDS) <= set(
        ids
    )


def test_every_technique_is_a_literature_hypothesis_not_authority():
    for tid, t in _by_id().items():
        ident = t["technique_identity"]
        assert ident["literature_status"] == LITERATURE_STATUS, tid
        assert ident["not_authority"] is True, tid
        assert t["current_verdict"]["literature_is"] == LITERATURE_STATUS, tid
        paper = t["source_paper"]
        assert paper.get("title"), tid
        assert paper.get("approx_date"), tid
        assert paper.get("status_in_this_registry") == LITERATURE_STATUS, tid


def test_required_fields_present_on_every_entry():
    for tid, t in _by_id().items():
        missing = [k for k in REQUIRED_ENTRY_FIELDS if k not in t]
        assert not missing, (tid, missing)
        organs = t["expected_useful_organs"]
        assert isinstance(organs, list) and organs, tid
        risks = t["risks"]
        assert isinstance(risks, list) and risks, tid
        lic = t["licensing_provenance"]
        assert lic.get("do_not_copy_third_party_code") is True, tid
        assert lic.get("s026_6") or "no blind" in json.dumps(lic).lower()
        mapping = t["hawking_experiment_mapping"]
        assert mapping.get("campaign_mechanism_overlap"), tid
        v = t["current_verdict"]
        assert v["status"] in ALLOWED_VERDICT_STATUSES, (tid, v["status"])
        storage = t["storage_vs_execution"]
        assert storage, tid


def test_untested_entries_name_a_cheapest_cpu_probe():
    for tid, t in _by_id().items():
        v = t["current_verdict"]
        if v["status"] not in {UNTESTED, RELATED_NEGATIVE}:
            continue
        exp = v.get("cheapest_hawking_experiment") or {}
        assert exp.get("id"), tid
        assert exp.get("name"), tid
        assert exp.get("cpu_only") is True, tid
        assert exp.get("touches_gpu") is False, tid
        assert exp.get("loads_model") is False, tid
        assert exp.get("no_second_27b") is True, tid
        assert exp.get("success_criterion"), tid


def test_verdict_without_hawking_receipt_is_rejected():
    fake = _blank_entry()
    fake["current_verdict"] = {
        "status": TESTED_NEGATIVE,
        "literature_is": LITERATURE_STATUS,
        "hawking_receipts": [],
        "cheapest_hawking_experiment": {"id": "HX", "name": "n"},
    }
    errs = verdict_errors(fake)
    assert any("without a cited Hawking receipt" in e for e in errs), errs


def test_arxiv_url_is_not_a_hawking_receipt():
    assert is_hawking_receipt_path("https://arxiv.org/abs/2405.16406") is False
    fake = _blank_entry()
    fake["current_verdict"] = {
        "status": TESTED_NEGATIVE,
        "literature_is": LITERATURE_STATUS,
        "hawking_receipts": ["https://arxiv.org/abs/2405.16406"],
    }
    errs = verdict_errors(fake)
    assert any("not a Hawking receipt" in e for e in errs), errs


def test_missing_receipt_path_fails_the_validator():
    fake = _blank_entry()
    fake["current_verdict"] = {
        "status": RELATED_NEGATIVE,
        "literature_is": LITERATURE_STATUS,
        "hawking_receipts": ["receipts/headless/DOES_NOT_EXIST_N043.json"],
        "cheapest_hawking_experiment": {"id": "HX", "name": "n"},
    }
    errs = verdict_errors(fake)
    assert any("does not exist" in e for e in errs), errs


def test_untested_does_not_require_a_receipt():
    fake = _blank_entry()
    assert verdict_errors(fake) == []
    assert not technique_field_errors(fake)


def test_every_claimed_verdict_cites_an_existing_hawking_receipt():
    doc = receipt()
    errs = registry_errors(doc)
    assert errs == [], errs
    for tid, t in _by_id(doc).items():
        v = t["current_verdict"]
        if v["status"] not in VERDICT_REQUIRES_RECEIPT:
            continue
        recs = v["hawking_receipts"]
        assert recs, tid
        for path in recs:
            assert is_hawking_receipt_path(path), (tid, path)
            assert citation_exists(path), (tid, path)


def test_campaign_scars_are_seeded_from_named_receipts():
    xref = receipt()["campaign_cross_references"]
    assert xref["shared_basis"]["verdict"] == SCAR_SHARED_BASIS
    assert any("SHARED_BASIS_KERNEL.json" in p for p in xref["shared_basis"]["receipts"])
    assert any("SHARED_BASIS_COHERENT.json" in p for p in xref["shared_basis"]["receipts"])

    assert xref["binary"]["verdict"] == SCAR_BINARY
    assert any("BYTES_FRONTIER.json" in p for p in xref["binary"]["receipts"])
    assert any("BINARY_HEALING.json" in p for p in xref["binary"]["receipts"])

    assert xref["low_rank_residual"]["verdict"] == SCAR_LOWRANK
    assert any("HYBRID_OPERATOR.json" in p for p in xref["low_rank_residual"]["receipts"])

    assert xref["ternary"]["verdict"] == SCAR_TERNARY
    assert any("BYTES_FRONTIER.json" in p for p in xref["ternary"]["receipts"])

    assert xref["sparse_residual"]["verdict"] == SCAR_SPARSE
    assert any("BYTES_FRONTIER.json" in p for p in xref["sparse_residual"]["receipts"])

    for key, block in xref.items():
        assert block.get("receipts"), key
        for path in block["receipts"]:
            assert citation_exists(path), (key, path)


def test_paper_entries_inherit_the_matching_campaign_scar():
    by = _by_id()
    assert by["onebit"]["current_verdict"]["status"] == TESTED_NEGATIVE
    assert by["onebit"]["current_verdict"]["campaign_verdict"] == SCAR_BINARY
    assert by["caldera"]["current_verdict"]["status"] == TESTED_NEGATIVE
    assert by["caldera"]["current_verdict"]["campaign_verdict"] == SCAR_LOWRANK
    assert SCAR_TERNARY in (by["twla"]["current_verdict"]["campaign_verdict"] or "")
    assert SCAR_TERNARY in (by["cat_q"]["current_verdict"]["campaign_verdict"] or "")
    assert SCAR_TERNARY in (by["ptqtp"]["current_verdict"]["campaign_verdict"] or "")
    assert SCAR_SPARSE in (by["squeezellm"]["current_verdict"]["campaign_verdict"] or "")
    assert by["spinquant"]["current_verdict"]["status"] == UNTESTED
    assert by["kivi"]["current_verdict"]["status"] == UNTESTED
    assert by["minicache"]["current_verdict"]["status"] == UNTESTED
    assert by["h2o"]["current_verdict"]["status"] == UNTESTED
    assert by["mixture_of_depths"]["current_verdict"]["status"] == UNTESTED
    assert by["prosparse"]["current_verdict"]["status"] == UNTESTED
    assert by["medusa_mtp"]["current_verdict"]["status"] == UNTESTED


def test_shared_basis_kernel_is_competent_but_dead_below_2_25():
    scar = receipt()["campaign_cross_references"]["shared_basis"]
    m = scar["measured"]
    assert m["kernel_competent"] is True
    assert m["coherent_shared_basis_beats_q2f"] is False
    assert m["k2_active_bpw"] < 2.25


def test_binary_is_faster_and_uniformly_injured():
    scar = receipt()["campaign_cross_references"]["binary"]
    m = scar["measured"]
    assert m["moved_toward_roof"] is True
    assert m["uniformly_injured"] is True
    assert m["heals_coherent"] == 0
    onebit = _by_id()["onebit"]["current_verdict"]["measured_numbers"]
    assert onebit["moved_toward_roof"] is True
    assert onebit["uniformly_injured"] is True


def test_ternary_flips_argmax_and_is_slower():
    scar = receipt()["campaign_cross_references"]["ternary"]
    m = scar["measured"]
    assert m["moved_toward_roof"] is False
    assert m["argmax_agree"] is False
    assert m["teacher_argmax"] == 9714
    assert m["student_argmax"] == 10895
    assert m["complete_token_ns"] > 27547874


def test_sparse_indices_cost_more_than_the_residual_saves():
    scar = receipt()["campaign_cross_references"]["sparse_residual"]
    m = scar["measured"]
    assert m["csr_bytes"] > 0
    assert m["csr_bytes"] > 1e9
    assert m["complete_token_ns"] > 27547874
    assert m["nnz_frac"] == 0.02


def test_lowrank_hybrid_never_heals():
    scar = receipt()["campaign_cross_references"]["low_rank_residual"]
    m = scar["measured"]
    assert m["coherent_hybrid_beats_q2f"] is False
    assert m["died_at"] == "held_out_activation"


def test_mutating_a_verdict_off_its_receipt_fails_the_gate():
    doc = copy.deepcopy(receipt())
    onebit = next(
        t for t in doc["techniques"] if t["technique_identity"]["id"] == "onebit"
    )
    onebit["current_verdict"]["hawking_receipts"] = []
    errs = registry_errors(doc)
    assert any("onebit" in e and "without a cited Hawking receipt" in e for e in errs)


def test_docs_ultragoals_file_exists_and_states_the_law():
    assert DOCS.is_file(), f"missing {DOCS}"
    text = DOCS.read_text()
    assert "HYPOTHESIS" in text
    assert "S026" in text
    assert "not authority" in text.lower() or "not an authority" in text.lower()
    assert "Hawking receipt" in text or "hawking receipt" in text.lower()
    assert "SpinQuant" in text
    assert "Medusa" in text
    assert "DOCTOR_TECHNIQUE_REGISTRY.json" in text
