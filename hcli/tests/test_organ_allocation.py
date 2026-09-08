"""G017: unequal organ allocation at matched complete EBPW.

Concrete numbers are from receipts/future/G017_ORGAN_ALLOCATION.json, measured
on Qwen3-0.6B. The INVERTED control and the matched-EBPW check are assertions
against those numbers, not against a regenerated search.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECEIPT = REPO / "receipts" / "future" / "G017_ORGAN_ALLOCATION.json"
SRC = REPO / "tools" / "future" / "organ_allocation.py"

sys.path.insert(0, str(REPO / "tools" / "future"))
from organ_allocation import (  # noqa: E402
    PRIOR_DEFICIT,
    allocate_organ_ebpw,
    order_organs_by_deficit,
)


def _receipt() -> dict:
    assert RECEIPT.is_file(), f"missing {RECEIPT}"
    return json.loads(RECEIPT.read_text())


def test_order_organs_by_deficit_is_descending_and_ends_at_up_proj():
    order = order_organs_by_deficit()
    assert order[0] == "q_proj"
    assert order[-1] == "up_proj"
    deficits = [PRIOR_DEFICIT[n] for n in order]
    assert deficits == sorted(deficits, reverse=True)


def test_organ_weighted_gives_up_proj_more_bits_per_param_than_q_proj():
    params = {n: 1000 for n in PRIOR_DEFICIT}
    ow = allocate_organ_ebpw("ORGAN_WEIGHTED", params, 4.5)
    inv = allocate_organ_ebpw("INVERTED", params, 4.5)
    uni = allocate_organ_ebpw("UNIFORM", params, 4.5)
    assert ow["up_proj"] > ow["q_proj"]
    assert ow["up_proj"] > ow["k_proj"]
    assert ow["up_proj"] > ow["o_proj"]
    assert inv["q_proj"] > inv["up_proj"]
    assert all(abs(uni[n] - 4.5) < 1e-12 for n in uni)


def test_mutation_anchor_still_orders_by_descending_deficit():
    """A mutation left live in the source is not a mutation test."""
    text = SRC.read_text()
    live = "key=lambda name: -PRIOR_DEFICIT[name]))  # MUTATION_ANCHOR_ORDER_BY_DEFICIT"
    mutated = "key=lambda name: PRIOR_DEFICIT[name]))  # MUTATION_ANCHOR_ORDER_BY_DEFICIT"
    assert live in text
    assert mutated not in text


def test_matched_complete_ebpw_uniform_and_organ_weighted():
    rec = _receipt()
    uni = rec["arms"]["UNIFORM"]
    ow = rec["arms"]["ORGAN_WEIGHTED"]
    inv = rec["arms"]["INVERTED"]
    assert uni["complete_ebpw"] == 9.2619
    assert ow["complete_ebpw"] == 9.2619
    assert inv["complete_ebpw"] == 9.2619
    assert rec["matched_ebpw"]["UNIFORM"] == 9.2619
    assert rec["matched_ebpw"]["ORGAN_WEIGHTED"] == 9.2619
    assert rec["matched_ebpw"]["residual_weighted_vs_uniform"] == 0.0
    assert uni["complete_bytes"] == 870191712
    assert ow["complete_bytes"] == 870191712
    assert inv["complete_bytes"] == 870191712


def test_inverted_control_ran_and_did_not_win():
    rec = _receipt()
    inv = rec["arms"]["INVERTED"]
    ow = rec["arms"]["ORGAN_WEIGHTED"]
    ctrl = rec["inverted_control"]
    assert ctrl["ran"] is True
    assert inv["complete_ebpw"] == 9.2619
    assert inv["capability"]["ppl"] == 12.84
    assert inv["capability"]["median_4gram_repeat"] == 0.8224
    assert inv["ppl_ok"] is False
    assert inv["diversity_ok"] is True
    assert inv["capability_ok"] is False
    assert ctrl["ppl"] == 12.84
    assert ctrl["median_4gram_repeat"] == 0.8224
    assert ctrl["capability_ok"] is False
    # The control that decides the experiment: inverted must not beat or tie
    # organ-weighted. It did not — ppl 12.84 vs 6.1738.
    assert inv["capability"]["ppl"] > ow["capability"]["ppl"]
    assert rec["verdict"]["prior_transferred"] is False


def test_uniform_passes_conjunction_organ_weighted_fails_ppl():
    rec = _receipt()
    uni = rec["arms"]["UNIFORM"]
    ow = rec["arms"]["ORGAN_WEIGHTED"]
    assert uni["capability"]["ppl"] == 5.2278
    assert uni["capability"]["median_4gram_repeat"] == 0.7103
    assert uni["ppl_ok"] is True
    assert uni["diversity_ok"] is True
    assert uni["capability_ok"] is True
    assert uni["capability_status"] == "CANDIDATE_PASS"
    assert ow["capability"]["ppl"] == 6.1738
    assert ow["capability"]["median_4gram_repeat"] == 0.6923
    assert ow["ppl_ok"] is False
    assert ow["diversity_ok"] is True
    assert ow["capability_ok"] is False
    assert ow["capability_status"] == "CAPABILITY_LOSS"
    gate = uni["gate"]
    assert gate["gate_ppl_max"] == 5.9178
    assert gate["gate_r4_max"] == 1.0857
    assert rec["dense_parent_capability"]["ppl"] == 4.7342
    assert rec["dense_parent_capability"]["median_4gram_repeat"] == 0.3619


def test_unequal_allocation_does_not_beat_uniform():
    rec = _receipt()
    assert rec["verdict"]["unequal_beats_uniform"] is False
    assert rec["verdict"]["pareto_weighted_vs_uniform"] == "mixed"
    assert rec["arms"]["UNIFORM"]["complete_ebpw"] == rec["arms"]["ORGAN_WEIGHTED"]["complete_ebpw"] == 9.2619


def test_g017_is_on_the_odyssey_ledger_as_nr_not_nx():
    """G017 is a measured NR candidate and an explicit NX refusal, not a blank."""
    from odyssey_ledger import progress
    led = json.loads((REPO / "receipts" / "future" / "G034_ODYSSEY_LEDGER.json").read_text())
    row = next(r for r in led["specimens"] if r["slug"] == "Qwen--Qwen3-0.6B@c1899de289a0")
    nr = row["axes"]["nr_candidate"]
    nx = row["axes"]["nx_disposition"]
    assert nr["state"] == "MEASURED"
    assert nr["receipt"] == "receipts/future/G017_ORGAN_ALLOCATION.json"
    assert nr["value"]["unequal_beats_uniform"] is False
    assert nr["value"]["uniform_capability_ok"] is True
    assert nr["value"]["organ_weighted_capability_ok"] is False
    assert nx["state"] == "REFUSED"
    assert "Reopen" in nx["reason"]
    assert "kernel" in nx["reason"]
    p = progress(led)
    assert p["per_axis"]["nr_candidate"]["MEASURED"] >= 1
    assert p["axes_owed"] < 392
    assert row["axes"]["anatomy"]["state"] == "MEASURED"


def test_mutation_check_recorded_a_real_hash_move():
    rec = _receipt()
    m = rec["mutation_check"]
    assert m["mutation_applied"] is True
    assert m["mutated_probe_failed_as_required"] is True
    assert m["no_mutation_left_live"] is True
    assert m["sha256_before"] == m["sha256_restored"]
    assert m["sha256_before"] != m["sha256_mutated"]
    assert m["sha256_before"] == "32e315b456a66ab93daeff17c59a9dcaa5effe6b0714e6326f1e84f7cc71282c"
    assert m["sha256_mutated"] == "1c106e58d408218de63dbc8b4fd8ec98a94e95f03e8a32acbe356865ea19e3f4"
    assert "-PRIOR_DEFICIT[name]" in m["anchor_line"]
