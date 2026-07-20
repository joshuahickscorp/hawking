#!/usr/bin/env python3.12
"""Tests for the Gravity Potency Ratchet (tools/foundry/gravity_potency.py)."""
import pathlib
import sys

FOUNDRY = pathlib.Path(__file__).resolve().parents[1]
if str(FOUNDRY) not in sys.path:
    sys.path.insert(0, str(FOUNDRY))

import pytest  # noqa: E402

import gravity_potency as gp  # noqa: E402


@pytest.fixture()
def foundry(tmp_path, monkeypatch):
    monkeypatch.setenv("HAWKING_FOUNDRY_DIR", str(tmp_path))
    gp.seal_v1()
    gp.seal_atlas()
    return tmp_path


def _review(**over):
    body = {"schema": gp.SCHEMA_REVIEW, "parent_id": "qwen3-235b:F1", "reviewer": "frontier",
            "verdict": "accept", "capability_receipt_sha256": "a" * 64,
            "mean_symmetric_kl": 0.04, "argmax_agreement": 0.97}
    body.update(over)
    return gp.seal_field(body, "sha256")


# ── selftest + V1 seal ────────────────────────────────────────────────────────────────
def test_selftest_green():
    assert gp.selftest()["ok"] is True


def test_v1_is_sealed_and_immutable(foundry):
    gen = gp.latest_generation()
    assert gen["method_version"] == "GRAVITY_METHOD_V1"
    assert gen["candidate_priors"]["organ_inversion"]["action"].startswith("allocate bits to gate/up")
    assert gen["storage_policy"]["expert_cache_cap_bytes"] == 20 * 1024 ** 3

    doc = gp.read_json_safe(gp.registry_path())
    doc["generations"]["GRAVITY_METHOD_V1"]["storage_policy"]["expert_cache_cap_bytes"] = 64 * 1024 ** 3
    gp.atomic_write_json(gp.registry_path(), doc)
    with pytest.raises(gp.PotencyError, match="mutated or unsealed"):
        gp.load_registry()


# ── 1. promotion ──────────────────────────────────────────────────────────────────────
def test_promotion_refused_without_review(foundry):
    with pytest.raises(gp.PotencyError, match="evidence review is missing"):
        gp.promote({"generation": {"kernel_set": {}}})


def test_promotion_refused_with_unsealed_review(foundry):
    bad = dict(_review())
    bad["reviewer"] = "tampered-after-seal"
    with pytest.raises(gp.PotencyError, match="not sealed"):
        gp.promote({"review": bad, "generation": {"kernel_set": {}}})


def test_promotion_refused_when_contract_weakened(foundry):
    with pytest.raises(gp.PotencyError, match="weakened"):
        gp.promote({"review": _review(),
                    "generation": {"quality_contract": {"mean_symmetric_kl_max": "1/4"}}})
    with pytest.raises(gp.PotencyError, match="weakened"):
        gp.promote({"review": _review(),
                    "generation": {"quality_contract": {"next_token_argmax_agreement_min": "9/10"}}})
    # 88 calibration tokens is exactly the falsified setting; it may never come back
    with pytest.raises(gp.PotencyError, match="weakened"):
        gp.promote({"review": _review(),
                    "generation": {"quality_contract": {"min_capability_tokens": 88}}})


def test_promotion_with_sealed_review_seals_v2(foundry):
    gen = gp.promote({"review": _review(),
                      "generation": {"source_revision": {"parent": "Qwen/Qwen3-235B-A22B"}}})
    assert gen["method_version"] == "GRAVITY_METHOD_V2"
    assert gen["promoted_by_review_sha256"] == _review()["sha256"]
    assert "qwen3-235b:F1" in gen["parents_completed"]
    assert gp.sealed(gen, "sha256")
    # V1 body survives untouched
    assert gp.load_registry()["generations"]["GRAVITY_METHOD_V1"]["generation"] == 1


# ── 2. potency vector ─────────────────────────────────────────────────────────────────
def test_ledger_append_and_read(foundry):
    gp.append_potency(gp.potency_row("gpt-oss-120b:F0", lowest_physical_bpw="3/10",
                                     lowest_capability_passing_bpw=None, source_bytes=65 * 10 ** 9))
    gp.append_potency(gp.potency_row("qwen3-235b:F1", lowest_physical_bpw="2/5"))
    assert len(gp.read_potency()) == 2
    row = gp.read_potency("gpt-oss-120b:F0")[0]
    assert row["lowest_physical_bpw"] == "3/10"
    assert row["lowest_capability_passing_bpw"] is None
    assert row["energy_joules"] is None
    assert gp.sealed(row, "sha256")


def test_unknown_axis_refused(foundry):
    with pytest.raises(gp.PotencyError, match="unknown potency axes"):
        gp.potency_row("x", vanity_score=9000)


def test_report_prints_vector_and_refuses_one_number(foundry):
    gp.append_potency(gp.potency_row("gpt-oss-120b:F0", lowest_physical_bpw="3/10"))
    text = gp.report_potency()
    for axis in gp.POTENCY_AXES:
        assert axis in text
    assert "REFUSED" in text
    with pytest.raises(gp.PotencyError, match="potency is a vector"):
        gp.collapse_to_score(gp.read_potency())


# ── 3. no-senility law ────────────────────────────────────────────────────────────────
PREV = {"parent_id": "gpt-oss-120b:F0", "lowest_credible_bpw": "2/5", "start_rate": "3/5"}


def test_no_senility_passes_an_aggressive_program(foundry):
    out = gp.check_no_senility(
        {"parent_id": "qwen3-235b:F1", "start_rate": "3/5",
         "rates": ["1/1", "3/5", "2/5", "1/3", "1/4"]}, PREV)
    assert out["ok"] is True, out["failures"]


def test_no_senility_fails_a_timid_program(foundry):
    out = gp.check_no_senility(
        {"parent_id": "qwen3-235b:F1", "start_rate": "7/10", "rates": ["1/1", "17/20", "7/10"]},
        PREV)
    assert out["ok"] is False
    joined = " ".join(out["failures"])
    assert "high sub-bit challenger" in joined
    assert "lowest credible region" in joined
    assert "lower-rate stress point" in joined


def test_no_senility_fails_size_justified_rate_raise(foundry):
    out = gp.check_no_senility(
        {"parent_id": "deepseek-685b:F2", "start_rate": "7/10",
         "start_rate_reason": "685B is much larger than the 120B parent, start safer",
         "start_rate_evidence": "b" * 64,
         "rates": ["1/1", "7/10", "2/5", "1/3", "1/4"]}, PREV)
    assert out["ok"] is False
    assert any("size argument" in f for f in out["failures"])


def test_no_senility_fails_unjustified_rate_raise(foundry):
    out = gp.check_no_senility(
        {"parent_id": "deepseek-685b:F2", "start_rate": "7/10",
         "rates": ["1/1", "7/10", "2/5", "1/3", "1/4"]}, PREV)
    assert any("without a measured start_rate_evidence" in f for f in out["failures"])


def test_no_senility_rejects_off_ladder_and_rounded_rates(foundry):
    out = gp.check_no_senility({"rates": ["1/1", "9/17", "1/4"]}, None)
    assert any("not on the exact rate ladder" in f for f in out["failures"])
    out = gp.check_no_senility({"rates": ["0.85"]}, None)
    assert any("exact rational" in f for f in out["failures"])


# ── 4. rate discipline ────────────────────────────────────────────────────────────────
def _full_sweep(rate):
    return [{"rate": rate, "lever": lever, "exhausted": True} for lever in gp.LEVER_ORDER]


def test_rate_discipline_allows_raise_after_full_sweep(foundry):
    out = gp.check_rate_discipline(_full_sweep("1/4") + _full_sweep("1/3"))
    assert out["ok"] is True, out["failures"]
    assert out["may_raise_rate"] is True


def test_rate_discipline_fails_premature_raise(foundry):
    history = _full_sweep("1/4")[:3] + [{"rate": "1/3", "lever": "representation"}]
    out = gp.check_rate_discipline(history)
    assert out["ok"] is False
    assert "levers unexhausted" in out["failures"][0]
    assert "routing_aware_allocation" in out["failures"][0]


def test_rate_discipline_fails_out_of_order_lever(foundry):
    out = gp.check_rate_discipline([{"rate": "1/4", "lever": "doctor_within_budget"}])
    assert out["ok"] is False
    assert "before" in out["failures"][0]


def test_rate_discipline_names_the_next_lever(foundry):
    out = gp.check_rate_discipline(_full_sweep("1/4")[:2])
    assert out["next_lever"] == "sharing_scope"
    assert out["may_raise_rate"] is False


# ── 5. negative transfer atlas ────────────────────────────────────────────────────────
def test_atlas_blocks_every_dead_lever(foundry):
    for lever in gp.load_atlas()["entries"]:
        assert gp.atlas_check(lever)["blocked"] is True


def test_atlas_block_carries_the_killing_measurement(foundry):
    out = gp.atlas_check("inter_expert_redundancy")
    assert out["blocked"] is True
    assert "1e-4" in out["killed_by"]
    assert "cosine" in out["reopen_condition"]


def test_atlas_reopens_only_on_a_new_parent_diagnosis(foundry):
    same_parent = {"parent_id": "gpt-oss-120b:F0", "reopens": ["large_expert_cache"],
                   "measurement": "cache hit rate 0.31"}
    assert gp.atlas_check("large_expert_cache", same_parent)["blocked"] is True

    no_measurement = {"parent_id": "deepseek-685b:F2", "reopens": ["large_expert_cache"]}
    assert gp.atlas_check("large_expert_cache", no_measurement)["blocked"] is True

    good = {"parent_id": "deepseek-685b:F2", "reopens": ["large_expert_cache"],
            "measurement": "measured cross-layer expert cache hit rate 0.31"}
    out = gp.atlas_check("large_expert_cache", good)
    assert out["blocked"] is False
    assert out["reopened_by"] == "deepseek-685b:F2"


def test_atlas_does_not_block_an_alive_lever(foundry):
    assert gp.atlas_check("row_norm_stratification")["blocked"] is False
