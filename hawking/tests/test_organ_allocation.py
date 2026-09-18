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
    # The anchor tracks the SOURCE line, which became `src[name]` when
    # order_organs_by_deficit learned to take a specimen's own deficits for the
    # LOCAL policy. Pinning the stale text would leave the live mutation check
    # unable to find its target and the ordering with no negative control.
    live = "key=lambda name: -src[name]))  # MUTATION_ANCHOR_ORDER_BY_DEFICIT"
    mutated = "key=lambda name: src[name]))  # MUTATION_ANCHOR_ORDER_BY_DEFICIT"
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
    # r4 0.8224 is four-gram repetition, i.e. visibly degenerate text. The old
    # bar was 1.0857 -- ABOVE the metric's own range, so nothing could ever fail
    # it and this assertion used to read True. G020 capped the bar inside the
    # metric (min(3 x dense, midpoint(dense, 1.0)) = 0.681) and the control now
    # fails diversity, which is the whole point of having a control.
    assert inv["diversity_ok"] is False
    assert inv["capability_ok"] is False
    assert ctrl["ppl"] == 12.84
    assert ctrl["median_4gram_repeat"] == 0.8224
    assert ctrl["capability_ok"] is False
    # The control that decides the experiment: inverted must not beat or tie
    # organ-weighted. It did not — ppl 12.84 vs 6.1738.
    assert inv["capability"]["ppl"] > ow["capability"]["ppl"]
    assert rec["verdict"]["prior_transferred"] is False


def test_under_the_g020_bar_uniform_is_a_false_survivor_on_diversity():
    """UNIFORM used to "pass". It passed a bar no arm could fail.

    The pre-G020 diversity bar was 3 x dense = 1.0857, and median 4-gram repeat
    lives in [0, 1] -- so the bar sat outside the metric and the conjunction was
    perplexity alone wearing a second name. Under G020's capped bar (0.681)
    UNIFORM's r4 of 0.7103 fails, which is exactly the false-Pareto-survivor
    class G020 exists to kill.

    The likelihood numbers are unchanged from the pre-G020 receipt to four
    decimal places -- 5.2278 / 6.1738 / 12.8400 -- so nothing about the arms
    moved. Only the ruler did.
    """
    rec = _receipt()
    uni = rec["arms"]["UNIFORM"]
    ow = rec["arms"]["ORGAN_WEIGHTED"]
    assert uni["capability"]["ppl"] == 5.2278
    assert uni["capability"]["median_4gram_repeat"] == 0.7103
    assert uni["ppl_ok"] is True          # likelihood was never the problem
    assert uni["diversity_ok"] is False   # the repetition always was
    assert uni["capability_ok"] is False
    assert uni["capability_status"] == "CAPABILITY_LOSS"
    assert ow["capability"]["ppl"] == 6.1738
    assert ow["capability"]["median_4gram_repeat"] == 0.6923
    assert ow["ppl_ok"] is False
    assert ow["diversity_ok"] is False
    assert ow["capability_ok"] is False
    assert ow["capability_status"] == "CAPABILITY_LOSS"
    gate = uni["gate"]
    assert gate["gate_ppl_max"] == 5.9178
    # The capped bar, and the fact that it IS capped. A bar back above 1.0 here
    # means the vacuous ruler came back.
    assert gate["gate_r4_max"] == 0.681
    assert gate["gate_r4_capped"] is True
    assert gate["gate_r4_max"] < 1.0
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
    # This asserted True until 2026-09-08, pinning a value the cited receipt
    # refutes: G017_ORGAN_ALLOCATION.json records capability_ok=False for EVERY
    # arm. UNIFORM passes perplexity and fails 4-gram diversity, so it fails the
    # conjunction. The ledger had recorded ppl_ok in a field meaning the whole
    # gate, and this test enforced that reading -- a test can hold a defect in
    # place as firmly as the code can.
    assert nr["value"]["uniform_capability_ok"] is False
    assert nr["value"]["organ_weighted_capability_ok"] is False
    # and the numbers the receipt carries must actually be here
    assert nr["value"]["uniform_ppl"] == 5.2278
    assert nr["value"]["organ_weighted_ppl"] == 6.1738
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
    # The two absolute digests that used to sit here pinned the CONTENT of a
    # source file that legitimately changes whenever the module is edited, and
    # they carried nothing the three relations above do not already prove: the
    # mutation landed, it moved the hash, and the file came back byte-identical.
    # They only ever failed on honest edits. What must not drift is the anchor
    # itself -- a checker that cannot find its target has no negative control.
    assert "-src[name]" in m["anchor_line"]
    assert "MUTATION_ANCHOR_ORDER_BY_DEFICIT" in m["anchor_line"]
