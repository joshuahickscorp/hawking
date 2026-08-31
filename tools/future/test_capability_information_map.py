"""Tests for the capability-sensitive information map.

A guard nobody has watched fail is not a guard. The load-bearing refusal:
a bit reduction is never marked supported without a recorded downstream
measurement, and entropy / W-space distortion alone cannot supply one.
"""
from __future__ import annotations

import json

import pytest

from tools.future import capability_information_map as cim
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def test_synthetic_activations_are_refused():
    with pytest.raises(cim.SyntheticActivationRefuse) as caught:
        cim.refuse_synthetic_activations(
            {"kind": "isotropic gaussian", "real_forward_pass": True, "from_embedding_table": True}
        )
    assert "REFUSED" in str(caught.value)
    with pytest.raises(cim.SyntheticActivationRefuse):
        cim.refuse_synthetic_activations({"kind": "cpu", "real_forward_pass": False})
    with pytest.raises(cim.SyntheticActivationRefuse):
        cim.refuse_synthetic_activations(None)
    src = cim.real_activation_source()
    cim.refuse_synthetic_activations(src)
    assert src["real_forward_pass"] is True
    assert src["synthetic"] is False
    assert src["token_ids"] == list(cim.PROMPT_TOKEN_IDS)


def test_module_refuses_to_mark_reducible_without_downstream_measurement():
    """HARD RULE: entropy or W-space distortion is never a licence."""
    lossless = cim.decide_supported_bit_reduction(
        candidate_bits=1,
        incumbent_bits=2,
        H_q_bits=0.4,
        wspace_relfro=1e-6,
        downstream=None,
    )
    assert lossless["supported"] is False
    assert lossless["reason"] == cim.DOWNSTREAM_UNMEASURED
    assert lossless["downstream_measured"] is False
    assert lossless["lossless_possible"] is True
    assert lossless["entropy_is_not_a_licence"] is True
    assert lossless["wspace_is_not_a_licence"] is True

    measured_false = cim.decide_supported_bit_reduction(
        candidate_bits=1,
        incumbent_bits=2,
        H_q_bits=0.4,
        wspace_relfro=0.0,
        downstream={"measured": False, "layer_output_cosine": 0.999},
    )
    assert measured_false["supported"] is False
    assert measured_false["reason"] == cim.DOWNSTREAM_UNMEASURED

    empty = cim.decide_supported_bit_reduction(
        candidate_bits=1,
        incumbent_bits=2,
        H_q_bits=0.4,
        downstream={"measured": True},
    )
    assert empty["supported"] is False
    assert empty["reason"] == cim.SENSITIVITY_INCOMPLETE

    organ_only = cim.decide_supported_bit_reduction(
        candidate_bits=1,
        incumbent_bits=2,
        H_q_bits=1.9,
        wspace_relfro=0.5,
        downstream={
            "measured": True,
            "layer_output_cosine": 0.999,
        },
    )
    assert organ_only["supported"] is False
    assert organ_only["reason"] == cim.SENSITIVITY_INCOMPLETE


def test_supported_row_without_downstream_raises():
    with pytest.raises(cim.DownstreamRequired) as caught:
        cim.refuse_supported_without_downstream(
            {"supported": True, "downstream_measured": False, "id": "L0.mlp.gate.all"},
            name="L0.mlp.gate.all",
        )
    assert "REFUSED" in str(caught.value)
    assert "downstream" in str(caught.value)
    cim.refuse_supported_without_downstream(
        {"supported": False, "downstream_measured": False, "id": "x"}
    )
    cim.refuse_supported_without_downstream(
        {"supported": True, "downstream_measured": True, "id": "y"}
    )


def test_clears_bar_only_with_organ_and_a_quantity_past_it():
    ok = cim.decide_supported_bit_reduction(
        candidate_bits=1,
        incumbent_bits=2,
        H_q_bits=1.87,
        wspace_relfro=0.4,
        downstream={
            "measured": True,
            "layer_output_cosine": 0.995,
            "hidden_after_n_cosine": 0.993,
            "gate_cosine": 0.994,
            "argmax_identical": True,
            "logits_cosine": 0.991,
        },
    )
    assert ok["supported"] is True
    assert ok["reason"] == cim.SENSITIVITY_CLEARS_BAR
    assert ok["downstream_measured"] is True
    # W-space injury is large and entropy is not lossless; still supported
    # because the downstream bars cleared. The inverse is the load-bearing no.
    assert ok["lossless_possible"] is False

    injured = cim.decide_supported_bit_reduction(
        candidate_bits=1,
        incumbent_bits=2,
        H_q_bits=0.2,
        wspace_relfro=1e-9,
        downstream={
            "measured": True,
            "layer_output_cosine": 0.95,
            "hidden_after_n_cosine": 0.999,
            "argmax_identical": True,
        },
    )
    assert injured["supported"] is False
    assert injured["reason"] == cim.LAYER_OUTPUT_BELOW_BAR
    assert injured["lossless_possible"] is True

    hidden = cim.decide_supported_bit_reduction(
        candidate_bits=3,
        incumbent_bits=4,
        downstream={
            "measured": True,
            "layer_output_cosine": 0.999,
            "hidden_after_n_cosine": 0.90,
        },
    )
    assert hidden["supported"] is False
    assert hidden["reason"] == cim.HIDDEN_AFTER_N_BELOW_BAR

    argmax = cim.decide_supported_bit_reduction(
        candidate_bits=3,
        incumbent_bits=4,
        downstream={
            "measured": True,
            "layer_output_cosine": 0.999,
            "argmax_identical": False,
            "logits_cosine": 0.999,
        },
    )
    assert argmax["supported"] is False
    assert argmax["reason"] == cim.ARGMAX_CHANGED


def test_allocation_refuses_a_supported_row_that_skipped_downstream():
    fake = {
        "measured": True,
        "sensitivity_uniform": False,
        "regions": [
            {
                "id": "fake.drop",
                "supported": True,
                "downstream_measured": False,
                "reason": cim.ENTROPY_OR_WSPACE_ALONE_INSUFFICIENT,
                "bytes_eliminated": 100,
                "apply_to_organ": "mlp",
                "channel": None,
                "layer": 0,
                "organ": "mlp.gate",
                "block": "all",
                "bits": 1,
                "downstream": {},
                "wspace_relfro": 0.0,
            }
        ],
    }
    with pytest.raises(cim.DownstreamRequired):
        cim.allocation_from_sensitivity(fake)


def test_roof_movement_quotes_the_live_budget_without_a_tps_key():
    zero = cim.roof_after_bytes(0)
    assert zero["bytes_eliminated"] == 0
    assert zero["quoted_delta"] == 0.0
    # The test's own name says "quotes the LIVE budget", so compare against the
    # budget, not against 66.54. Pinning the literal made this fail when the budget
    # was corrected to include the unattributed 0.321 ms - the correction this test
    # should have been confirming, not resisting.
    from tools.future import causal_budget_71 as _cb
    assert zero["quoted_roof_on_todays_bytes"] == pytest.approx(
        round(_cb.tps(_cb.token_ms(_cb.CLEAN_GEMV_GB_S)), 2), abs=0.05
    )
    assert zero["seventy_one_reachable_at_roof"] is False
    assert "tps" not in zero
    big = cim.roof_after_bytes(2_139_095_040)
    assert big["quoted_roof_after_allocation"] > zero["quoted_roof_on_todays_bytes"]
    assert "tps" not in big


def test_accounting_reconciles_mlp_and_qkvz():
    snap = cim.accounting()
    assert snap["reconciled"] is True
    assert snap["token_active_bytes"] == 9_878_901_136
    assert snap["mlp_stored_bytes"] == 5_347_795_776
    assert snap["mlp_code_bytes"] == 4_278_190_080
    assert snap["qkvz_stored_bytes"] == 2_139_096_960
    assert snap["n_layers"] == 64
    assert snap["hidden_size"] == 5120


def test_byte_model_is_exact():
    assert cim.bytes_eliminated_at_bits(4_278_190_080, 2, 1) == 2_139_095_040
    assert cim.bytes_eliminated_at_bits(251_658_240, 4, 3) == 251_658_240 // 4
    assert cim.code_bytes_at_bits(1000, 2, 2) == 1000


def test_is_gqa_layer_matches_geometry():
    assert cim.is_gqa_layer(3) is True
    assert cim.is_gqa_layer(63) is True
    assert cim.is_gqa_layer(0) is False
    assert cim.is_gqa_layer(21) is False
    assert cim.is_gqa_layer(42) is False
    assert cim.mixer_kind(3) == "gqa"
    assert cim.mixer_kind(0) == "delta_net"


def _snap() -> dict:
    return cim.snapshot(consult_index=False)


def test_real_prefix_is_not_gaussian():
    cap = cim.capture_real_prefix()
    cim.refuse_synthetic_activations(cap["source"])
    assert cap["n_tokens"] == len(cim.PROMPT_TOKEN_IDS)
    assert cap["source"]["token_ids"] == list(cim.PROMPT_TOKEN_IDS)
    assert cap["snaps"][0]["hidden_in"].shape == (5120,)
    assert cap["snaps"][63]["hidden_out"].shape == (5120,)
    # Residual stream is not a zero vector and not N(0,1/sqrt(h)).
    h = cap["snaps"][0]["hidden_in"]
    assert float(abs(h).mean()) > 1e-4
    assert cap["kits"][0].kind == "delta_net"
    assert cap["kits"][63].kind == "gqa"
    assert len(cap["token_hidden_in"][0]) == cap["n_tokens"]


def test_sensitivity_records_downstream_on_every_region():
    sens = _snap()["sensitivity"]
    assert sens["measured"] is True
    assert sens["real_forward_pass"] is True
    assert sens["synthetic"] is False
    assert sens["n_regions"] > 0
    for r in sens["regions"]:
        assert r["downstream_measured"] is True
        ds = r["downstream"]
        assert ds["layer_output_cosine"] is not None
        assert ds["hidden_after_n_cosine"] is not None
        assert ds["gate_cosine"] is not None
        assert ds["real_forward_pass"] is True
        if r["supported"] and not r["downstream_measured"]:
            raise AssertionError(f"supported drop without downstream: {r['id']}")
        if r["reason"] in {cim.DOWNSTREAM_UNMEASURED, cim.ENTROPY_OR_WSPACE_ALONE_INSUFFICIENT}:
            assert r["supported"] is False
        cim.refuse_supported_without_downstream(r, name=r["id"])


def test_allocation_eliminates_only_licensed_bytes_and_quotes_the_roof():
    alloc = _snap()["allocation"]
    licensed_parent = {
        (r["layer"], r["organ"])
        for r in alloc["regions"]
        if r.get("channel") is None and r["supported"]
    }
    expected = 0
    for r in alloc["regions"]:
        if not r["supported"]:
            continue
        if r.get("channel") is None:
            expected += int(r["bytes_eliminated"])
        elif (r["layer"], r["organ"]) not in licensed_parent:
            expected += int(r["bytes_eliminated"])
    assert alloc["total_bytes_eliminated"] == expected
    if not alloc["any_supported"]:
        assert alloc["total_bytes_eliminated"] == 0
        assert alloc["token_bytes_after"] == 9_878_901_136
        assert alloc["roof_movement"]["quoted_delta"] == 0.0
        assert alloc["roof_movement"]["seventy_one_reachable_at_roof"] is False
        assert alloc["school"] in {
            "CLOSED_UNIFORM_SENSITIVITY",
            "CLOSED_NO_REGION_CLEARS",
        }
    for r in alloc["regions"]:
        if r["supported"]:
            assert r["downstream_measured"] is True
            assert r["reason"] == cim.SENSITIVITY_CLEARS_BAR
        else:
            assert r["bytes_eliminated"] == 0
            assert r["bits"] in {2, 4}
    assert "tps" not in alloc["roof_movement"]


def test_candidates_cover_the_contract_and_refuse_entropy_alone():
    cands = _snap()["candidates"]
    ids = [c["id"] for c in cands]
    assert ids == list(cim.REQUIRED_CANDIDATE_IDS)
    legal = {
        cim.ALREADY_FALSIFIED,
        cim.MEASURED_NEGATIVE,
        cim.OPEN,
        cim.UNMEASURED,
    }
    for row in cands:
        assert row["physical_primitive"] in ATLAS_PRIMITIVES
        assert row["status"] in legal
        assert row["evidence_class"] == "STATIC_ONLY"
        assert row["gpu_authority"] is False
        assert row["mechanism"]
        assert row["byte_model"]
        assert row["cheapest_falsifier"]
    by_id = {c["id"]: c for c in cands}
    assert by_id["entropy_or_wspace_alone"]["status"] == cim.MEASURED_NEGATIVE
    assert by_id["entropy_or_wspace_alone"]["bytes_eliminated_if_true"] == 0
    assert by_id["entropy_or_wspace_alone"]["support"] == "REFUSED"


def test_build_emits_sealed_receipt():
    out = cim.build(consult_index=True)
    assert out.parent == RECEIPTS
    assert out.name == "CAPABILITY_INFORMATION_MAP.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == cim.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    _assert_no_hardware_claims(doc)
    assert doc["accounting"]["reconciled"] is True
    assert doc["capture"]["real_forward_pass"] is True
    assert doc["sensitivity"]["synthetic"] is False
    assert doc["allocation"]["roof_movement"]["source"] == cim.BUDGET_REL
    assert [c["id"] for c in doc["candidates"]] == list(cim.REQUIRED_CANDIDATE_IDS)
    for r in doc["allocation"]["regions"]:
        if r["supported"]:
            assert r["downstream_measured"] is True


def test_module_entrypoint_runs_and_emits_sealed_receipt():
    rc = cim.main(["--build"])
    assert rc == 0
    doc = json.loads((RECEIPTS / cim.RECEIPT).read_text())
    assert doc["schema"] == cim.SCHEMA
    assert doc["seal_sha256"]


def test_selftest_aliases_build():
    assert cim.selftest is cim.build or cim.selftest().name == cim.RECEIPT


def test_hardware_fields_stay_non_numeric_on_the_receipt():
    out = cim.build(consult_index=False)
    doc = json.loads(out.read_text())
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:

        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    here = f"{path}.{k}" if path else k
                    if k in HARDWARE_FIELDS:
                        assert not isinstance(v, (int, float)) or isinstance(v, bool), here
                    walk(v, here)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(doc)
