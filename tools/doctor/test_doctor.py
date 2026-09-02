"""Doctor depth: technique order, fail-able zeros, real artifacts, no weight bytes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.doctor import (
    BROKEN_ORGAN,
    FAIL,
    PASS,
    ROUTED_EXPERT_ORGAN,
    TECHNIQUE_ORDER,
    THREE_ZEROS,
    TIED_EMBED_ORGAN,
    UNKNOWN,
    AccessLog,
    WeightBytesForbidden,
    check_three_zeros,
    diagnose,
    is_weight_file,
    ordinary_quantization,
    walk_order,
    zeros_controls,
)
from tools.doctor.anatomy import FLASH_SLUG, resolve_specimen
from tools.doctor.engine import (
    FLASH_EBPW_RECEIPT,
    QWEN06_SLUG,
    QWEN80_RECEIPT,
    RECEIPT,
    SCHEMA,
    build,
    retrieve_negative,
)
from tools.future._common import RECEIPTS, REPO


def test_technique_order_is_roadmap_9_1():
    names = [t.name for t in walk_order()]
    assert names[0] == "ELIMINATE"
    assert names[1] == "REPARAMETERIZE / coordinate shaping"
    assert names[2] == "SHARE"
    assert names[3] == "FACTORIZE"
    assert names[4] == "GENERATE"
    assert names[5] == "ROUTE"
    assert names[6] == "SENSITIVITY-AWARE INFORMATION ASSIGNMENT"
    assert names[7] == "HEAL"
    assert names[8] == "QUANTIZE"
    assert names[9] == "NATIVE OPERATORS"
    assert names[10] == "RUNTIME STATE"
    assert names[11] == "REMOVE COMPUTE"
    assert names[12] == "REDUCE DECODE FORWARDS"
    assert names[13] == "DEVICE COMPILE"
    assert names[14] == "VERIFY COMPLETE FUNCTION"
    assert [t.index for t in TECHNIQUE_ORDER] == list(range(15))
    assert len(TECHNIQUE_ORDER) == 15


def test_three_zeros_fail_on_broken_and_pass_on_good():
    broken = check_three_zeros(BROKEN_ORGAN)
    assert broken["ZERO_STORAGE"].verdict == FAIL
    assert broken["ZERO_INDEPENDENT_INFORMATION"].verdict == FAIL
    assert broken["ZERO_EXECUTION"].verdict == FAIL
    assert ordinary_quantization(broken) is True

    tied = check_three_zeros(TIED_EMBED_ORGAN)
    assert tied["ZERO_STORAGE"].verdict == PASS
    assert tied["ZERO_INDEPENDENT_INFORMATION"].verdict == PASS
    assert ordinary_quantization(tied) is False

    routed = check_three_zeros(ROUTED_EXPERT_ORGAN)
    assert routed["ZERO_EXECUTION"].verdict == PASS
    assert ordinary_quantization(routed) is False

    controls = zeros_controls()
    assert controls["all_three_fail_on_broken"] is True
    assert controls["storage_pass_on_tied"] is True
    assert controls["info_pass_on_tied"] is True
    assert controls["execution_pass_on_routed"] is True
    assert controls["storage_fail_on_broken"] is True
    assert controls["info_fail_on_broken"] is True
    assert controls["execution_fail_on_broken"] is True


def test_ordinary_quantization_is_false_if_any_zero_passes():
    """FAIL+FAIL+PASS is not ordinary quantization. A constant-True would be a lie."""
    mixed = dict(BROKEN_ORGAN)
    mixed["executes_every_token"] = False
    mixed["n_experts"] = 512
    mixed["experts_per_tok"] = 10
    mixed["organ_class"] = "routed_expert"
    results = check_three_zeros(mixed)
    assert results["ZERO_EXECUTION"].verdict == PASS
    assert results["ZERO_STORAGE"].verdict == FAIL
    assert ordinary_quantization(results) is False


def test_ordinary_quantization_false_on_unknown():
    empty = check_three_zeros(
        {
            "name": "mystery",
            "organ_class": "whole_artifact_ledger",
        }
    )
    assert empty["ZERO_STORAGE"].verdict == UNKNOWN
    assert ordinary_quantization(empty) is False


def test_zeros_carry_evidence_and_uncertainty():
    broken = check_three_zeros(BROKEN_ORGAN)
    for name in THREE_ZEROS:
        row = broken[name]
        assert row.evidence, name
        assert row.uncertainty
        assert row.evidence_tier == "STATIC"
        assert 0.0 <= row.confidence <= 1.0


def _assert_diagnosis(doc: dict) -> None:
    assert doc["weights_opened"] is False
    assert doc["io"]["weight_bytes_loaded"] == 0
    assert doc["io"]["weight_files_opened"] == []
    for p in doc["io"]["metadata_files_opened"]:
        assert is_weight_file(p) is False, p
    assert doc["overall"]["verdict"] != "HEALTHY"
    assert doc["overall"]["never_healthy"] is True
    assert doc["evidence_tier"] == "STATIC"
    assert doc["organs"]
    for organ in doc["organs"]:
        assert organ["evidence_tier"] == "STATIC"
        assert organ["uncertainty"]
        assert set(organ["three_zeros"]) == set(THREE_ZEROS)
        for cell in organ["three_zeros"].values():
            assert cell["verdict"] in {PASS, FAIL, UNKNOWN}
            assert cell["evidence_tier"] == "STATIC"
            assert "uncertainty" in cell
        names = [step["name"] for step in organ["technique_order"]]
        assert names == [t.name for t in TECHNIQUE_ORDER]
        assert all(step["asked"] is True for step in organ["technique_order"])
        assert organ["next_experiment"]["evidence_tier"] == "STATIC"


def test_diagnoses_qwen80_bit_budget_receipt():
    path = REPO / QWEN80_RECEIPT
    assert path.is_file(), "QWEN80_BIT_BUDGET_LEDGER.json must exist"
    doc = diagnose(path, negative_science=False, scars=[])
    _assert_diagnosis(doc)
    assert doc["kind"] == "RECEIPT"
    assert doc["name"].endswith("QWEN80_BIT_BUDGET_LEDGER.json")
    routed = next(o for o in doc["organs"] if "routed" in str(o["organ_class"]).lower())
    assert routed["three_zeros"]["ZERO_EXECUTION"]["verdict"] == PASS
    assert routed["ordinary_quantization"] is False
    shared = next(o for o in doc["organs"] if "shared_expert" in str(o["organ_class"]).lower())
    assert shared["three_zeros"]["ZERO_EXECUTION"]["verdict"] == FAIL
    embed = next(o for o in doc["organs"] if "embed" in str(o["organ_class"]).lower())
    assert embed["three_zeros"]["ZERO_STORAGE"]["verdict"] == FAIL
    assert doc["overall"]["verdict"] in {"ZERO_AVAILABLE", "MIXED"}
    assert doc["ebpw_validate"]["called"] is True
    assert doc["ebpw_validate"]["can_promote"] is False


def test_diagnoses_flash_ebpw_budget_receipt():
    path = REPO / FLASH_EBPW_RECEIPT
    assert path.is_file(), "FLASH_EBPW_BUDGET.json must exist"
    doc = diagnose(path, negative_science=False, scars=[])
    _assert_diagnosis(doc)
    assert doc["kind"] == "RECEIPT"
    names = {o["name"] for o in doc["organs"]}
    assert "routed_experts" in names
    assert "shared_expert" in names
    routed = next(o for o in doc["organs"] if o["name"] == "routed_experts")
    assert routed["three_zeros"]["ZERO_EXECUTION"]["verdict"] == PASS
    shared = next(o for o in doc["organs"] if o["name"] == "shared_expert")
    assert shared["three_zeros"]["ZERO_EXECUTION"]["verdict"] == FAIL
    vision = next(o for o in doc["organs"] if "vision" in str(o["name"]))
    assert vision["three_zeros"]["ZERO_EXECUTION"]["verdict"] == PASS


def test_access_log_refuses_safetensors(tmp_path: Path):
    shard = tmp_path / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"not-a-real-weight-body")
    log = AccessLog()
    with pytest.raises(WeightBytesForbidden):
        log.open_bytes(shard)
    assert log.weight_bytes_loaded == 0
    assert str(shard) in log.refused
    assert is_weight_file(shard) is True
    assert is_weight_file(tmp_path / "model.safetensors.index.json") is False


def test_retrieve_negative_calls_refuse_if_dead(monkeypatch):
    called = []

    def fake(proposal, scars=None):
        called.append({"proposal": proposal, "scars": scars})
        return {"refused": True, "scar_id": "TEST-SCAR"}

    monkeypatch.setattr("tools.future.negative_index.refuse_if_dead", fake)
    out = retrieve_negative(
        model="qwen3.8-27b", organ="mlp", family="shared_basis", scars=[]
    )
    assert out["called_refuse_if_dead"] is True
    assert called, "refuse_if_dead must be invoked, not merely imported"
    assert called[0]["proposal"]["hypothesis_family"] == "shared_basis"


@pytest.mark.skipif(
    resolve_specimen(FLASH_SLUG) is None,
    reason="Flash specimen not mounted",
)
def test_diagnoses_flash_specimen_metadata_only():
    spec = resolve_specimen(FLASH_SLUG)
    assert spec is not None
    doc = diagnose(spec, negative_science=False, scars=[])
    _assert_diagnosis(doc)
    assert doc["kind"] == "SPECIMEN_METADATA"
    assert doc["fingerprint"]["weights_opened"] is False
    assert doc["fingerprint"]["num_experts"] == 512
    assert doc["fingerprint"]["num_experts_per_tok"] == 10
    opened = doc["io"]["metadata_files_opened"]
    assert any(p.endswith("config.json") for p in opened)
    assert not any(p.endswith(".safetensors") and not p.endswith(".index.json") for p in opened)
    routed = next(o for o in doc["organs"] if o["organ_class"] == "routed_expert")
    assert routed["three_zeros"]["ZERO_EXECUTION"]["verdict"] == PASS
    # Prior bank screen: experts are independent.
    if routed["flags"].get("cross_expert_cosine") is not None:
        assert routed["three_zeros"]["ZERO_INDEPENDENT_INFORMATION"]["verdict"] == FAIL
        assert routed["ordinary_quantization"] is False
    embed = next(o for o in doc["organs"] if o["organ_class"] == "embed_tokens")
    assert embed["three_zeros"]["ZERO_STORAGE"]["verdict"] == FAIL  # untied


@pytest.mark.skipif(
    resolve_specimen(QWEN06_SLUG) is None,
    reason="Qwen3-0.6B specimen not mounted",
)
def test_qwen06_tied_embeddings_pass_zero_storage():
    spec = resolve_specimen(QWEN06_SLUG)
    assert spec is not None
    doc = diagnose(spec, negative_science=False, scars=[])
    _assert_diagnosis(doc)
    embed = next(o for o in doc["organs"] if o["organ_class"] == "embed_tokens")
    assert embed["three_zeros"]["ZERO_STORAGE"]["verdict"] == PASS
    assert embed["three_zeros"]["ZERO_INDEPENDENT_INFORMATION"]["verdict"] == PASS
    mlp = next(o for o in doc["organs"] if o["organ_class"] == "dense_mlp")
    assert mlp["three_zeros"]["ZERO_EXECUTION"]["verdict"] == FAIL
    assert mlp["ordinary_quantization"] is True


def test_build_emits_sealed_receipt():
    out = build()
    assert out.parent == RECEIPTS
    assert out.name == RECEIPT
    doc = json.loads(out.read_text())
    assert doc["schema"] == SCHEMA
    assert doc["weight_bytes_loaded"] == 0
    assert doc["weights_opened"] is False
    assert doc["evidence_tier"] == "STATIC"
    assert doc["n_artifacts"] >= 2
    names = doc["artifact_names"]
    assert any("QWEN80_BIT_BUDGET_LEDGER.json" in str(n) for n in names)
    assert any("FLASH_EBPW_BUDGET.json" in str(n) for n in names)
    assert doc["three_zeros_controls"]["all_three_fail_on_broken"] is True
    assert doc["three_zeros_controls"]["storage_pass_on_tied"] is True
    assert doc["three_zeros_controls"]["execution_pass_on_routed"] is True
    assert doc["negative_science_call"]["called_refuse_if_dead"] is True
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    for art in doc["artifacts"]:
        assert art["io"]["weight_bytes_loaded"] == 0
        assert art["overall"]["verdict"] != "HEALTHY"
    assert not any("HEALTHY" == art["overall"]["verdict"] for art in doc["artifacts"])


def test_ebpw_validate_invokes_check_three_zeros(monkeypatch):
    """A module import is not a call site. Count the invocations."""
    import tools.doctor.zeros as z
    from tools.future import ebpw_categories as ec

    seen = {"n": 0}
    real = z.check_three_zeros

    def wrap(organ):
        seen["n"] += 1
        return real(organ)

    monkeypatch.setattr(z, "check_three_zeros", wrap)
    result = ec.doctor_zeros_for_doc({"organs": [BROKEN_ORGAN]})
    assert seen["n"] >= 1
    assert result["called"] == "tools.doctor.zeros.check_three_zeros"
    assert result["any_ordinary_quantization"] is True

    q80 = REPO / QWEN80_RECEIPT
    if q80.is_file():
        seen["n"] = 0
        validated = ec.validate(json.loads(q80.read_text()), source_path=QWEN80_RECEIPT)
        assert seen["n"] >= 1
        assert validated["doctor_three_zeros"]["called"] == "tools.doctor.zeros.check_three_zeros"
        assert validated["can_promote"] is False
        assert validated["doctor_three_zeros"]["any_zero_available"] is True
