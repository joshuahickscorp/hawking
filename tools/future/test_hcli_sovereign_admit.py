"""The sovereign loop's shape boundary: no reply may crash validate().

The loop has taken three crashes on the SHAPE of a reply rather than its content
(n_accepted missing on the parse-failure path; selected_work returned as a dict
and sliced as a list; fourteen iterations that parsed nothing). Each was fixed
where it was found. This is the test that would have caught all three at once:
every shape admit() can produce is fed to validate(), and validate must return a
uniform result for all of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hcli_sovereign as sov  # noqa: E402
from resident_output_contract import SOVEREIGN_REPLY_SCHEMA, admit  # noqa: E402

VALIDATE_KEYS = {"ok", "accepted", "rejected", "n_accepted", "n_rejected"}

# Every shape the body has actually produced, plus the ones it is free to.
REPLIES = [
    "",
    "   ",
    "not json at all",
    "```json\n{}\n```",
    "{}",
    "[]",
    "null",
    '{"selected_work": {"type": "PERTURB"}}',           # dict where a list goes
    '{"selected_work": "PERTURB"}',                     # string where a list goes
    '{"selected_work": [null, 3, "x"]}',                # junk members
    '{"selected_work": [{"type": "DELETE_EVERYTHING"}]}',
    '{"selected_work": [{"type": "PERTURB", "params": {}}]}',
    '{"selected_work": [{"type": "PERTURB", "params": ["up", 0, 0.5]}]}',
    '{"selected_work": [{"type": "PERTURB", "params": "up/0/0.5"}]}',
    '{"selected_work": [{"type": "PERTURB", "params": 7}]}',
    '{"selected_work": [{"type": "PERTURB", "params": '
    '{"tensor": "up", "layer": "NaN", "fraction": "half"}}]}',
    '{"selected_work": [{"type": "PERTURB", "params": '
    '{"tensor": "sideways", "layer": 3, "fraction": 0.5}}]}',
    '{"selected_work": [{"type": "PERTURB", "params": '
    '{"tensor": "up", "layer": 999, "fraction": 0.5}}]}',
    '{"selected_work": [{"type": "PERTURB", "params": '
    '{"tensor": "up", "layer": 0, "fraction": 99.0}}]}',
    '{"belief_update": "H1 dead", "live_hypotheses": [{"id": "H3"}]',  # truncated
    '{"selected_work": [{"type": "PERTURB", "params": '
    '{"tensor": "up", "layer": 0, "fraction": 0.5}}]} then it kept talking '
    'and talking and talking and talking and talking and talking',
    '{"a": 1} {"b": 2}',
    '{"selected_work": [' + ','.join(
        ['{"type": "PERTURB", "params": {"tensor": "up", "layer": 1, '
         '"fraction": 0.5}}'] * 40) + ']',
]


@pytest.mark.parametrize("reply", REPLIES, ids=range(len(REPLIES)))
def test_no_reply_shape_can_crash_the_boundary(reply):
    adm = admit(reply, SOVEREIGN_REPLY_SCHEMA)
    obj = adm["value"] if adm["ok"] else adm["value"]
    v = sov.validate(obj)
    assert set(v) >= VALIDATE_KEYS, "validate must return one shape on every path"
    assert isinstance(v["accepted"], list)
    assert isinstance(v["rejected"], list)
    assert v["n_accepted"] == len(v["accepted"])
    assert v["n_rejected"] == len(v["rejected"])


@pytest.mark.parametrize("reply", REPLIES, ids=range(len(REPLIES)))
def test_admit_never_raises_and_holds_its_key_set(reply):
    adm = admit(reply, SOVEREIGN_REPLY_SCHEMA)
    assert set(adm) == set(admit("{}", SOVEREIGN_REPLY_SCHEMA))
    assert isinstance(adm["value"]["selected_work"], list), (
        "selected_work must be a list on every path - slicing a dict is the "
        "exact crash that killed iteration 2"
    )


def test_the_unparsed_path_still_carries_its_counts():
    """The first of the three crashes. Kept as a named regression."""
    v = sov.validate(None)
    assert v["ok"] is False
    assert v["n_accepted"] == 0 and v["n_rejected"] == 0


def test_a_dict_selected_work_is_coerced_not_sliced():
    """The third crash. A dict must become a one-item list, not a KeyError."""
    v = sov.validate({"selected_work": {
        "type": "PERTURB",
        "params": {"tensor": "up", "layer": 0, "fraction": 0.5}}})
    assert v["n_accepted"] == 1


def test_a_good_reply_is_still_accepted():
    """A boundary that rejects everything is not a boundary, it is a wall."""
    adm = admit(
        '{"belief_update": "H2 refuted", '
        '"live_hypotheses": [], "escalation_needed": false, '
        '"selected_work": [{"type": "PERTURB", "why": "down is most sensitive", '
        '"params": {"tensor": "down", "side": "rows", "layer": 12, '
        '"fraction": 0.4}}]}',
        SOVEREIGN_REPLY_SCHEMA)
    assert adm["ok"] is True, adm["missing"]
    v = sov.validate(adm["value"])
    assert v["n_accepted"] == 1
    assert v["accepted"][0]["params"]["tensor"] == "down"


def test_the_loop_imports_admit_rather_than_parsing_by_hand():
    src = Path(sov.__file__).read_text()
    assert "from resident_output_contract import admit" in src
    assert 'reask_kind = "narrow"' in src, (
        "the narrow re-ask is the point of wiring admit in; without it the "
        "loop still burns a whole turn re-running the scientific prompt"
    )


def test_the_narrow_reask_only_asks_for_what_is_missing():
    adm = admit('{"belief_update": "H1 dead", "live_hypotheses": [{"id": "H3"}]',
                SOVEREIGN_REPLY_SCHEMA)
    assert adm["parse"]["recovered"] is True
    assert adm["reask"]["needed"] is True
    frag = adm["reask"]["prompt_fragment"]
    assert "selected_work" in frag
    assert "belief_update" not in frag, (
        "re-asking for a field the body already supplied is the waste this "
        "path exists to avoid"
    )


def test_a_schema_incomplete_reply_that_carries_work_is_still_executed():
    """admit's schema requires `why` and `side`; validate does not need either
    to run the work. A reply missing only those must NOT cost a re-ask turn."""
    adm = admit(
        '{"belief_update": "x", "live_hypotheses": [], '
        '"escalation_needed": false, "selected_work": [{"type": "PERTURB", '
        '"params": {"tensor": "down", "layer": 12, "fraction": 0.4}}]}',
        SOVEREIGN_REPLY_SCHEMA)
    assert adm["ok"] is False
    assert set(adm["missing"]) <= {"selected_work.params.side", "selected_work.why"}
    assert sov.validate(adm["value"])["n_accepted"] == 1
    src = Path(sov.__file__).read_text()
    assert 'has_work = bool(adm["value"].get("selected_work"))' in src


def test_params_as_a_list_is_rejected_not_a_crash():
    """The FOURTH shape crash, found by the adversarial lane before the body
    produced it. A list is truthy, so `or {}` never fired and .get raised."""
    v = sov.validate({"selected_work": [
        {"type": "PERTURB", "params": ["up", 0, 0.5]}]})
    assert v["n_accepted"] == 0
    assert v["n_rejected"] == 1
    assert "params is list" in v["rejected"][0]["why"]


def test_params_as_a_string_is_rejected_not_a_crash():
    v = sov.validate({"selected_work": [
        {"type": "PERTURB", "params": "up/0/0.5"}]})
    assert v["n_rejected"] == 1
    assert "params is str" in v["rejected"][0]["why"]


def test_params_absent_is_still_a_plain_rejection_not_a_crash():
    """Missing params must stay the ordinary path: rejected on its fields, not
    on its type. Over-tightening the coercion would break the common case."""
    v = sov.validate({"selected_work": [{"type": "PERTURB"}]})
    assert v["n_rejected"] == 1
    assert "params is" not in v["rejected"][0]["why"]


def test_the_context_pack_stays_bounded_as_state_accumulates():
    """The pack that worked was under ~1600 chars and this body degenerates with
    length. Both growing inputs - the hypothesis feed and the tried-params list -
    are capped, so a long campaign cannot inflate the pack without bound."""
    import json
    k = json.loads((sov.REPO / sov.KERNEL_REL).read_text())
    lean = len(sov.context_pack(k))
    fat = dict(k)
    fat["live_hypotheses"] = [{"id": "I" * 200, "claim": "c" * 4000}
                              for _ in range(50)]
    fat["tried_params"] = [f"up/L{i}/rows/0.5" for i in range(500)]
    grown = len(sov.context_pack(fat))
    assert grown - lean < 400, (
        f"pack grew {grown - lean} chars on 50 hypotheses and 500 tried params; "
        "an unbounded feed is how the pack outgrows the body"
    )
    assert grown < 2400


def test_the_terse_pack_is_much_shorter_than_the_full_one():
    import json
    k = json.loads((sov.REPO / sov.KERNEL_REL).read_text())
    assert len(sov.context_pack(k, terse=True)) < len(sov.context_pack(k)) / 1.5


def test_the_schema_placeholder_is_not_copyable_as_a_value():
    """16 of 26 recorded hypothesis ids were literally "x" - the schema's own
    placeholder. An id that is not distinct makes the register unjoinable."""
    import json
    k = json.loads((sov.REPO / sov.KERNEL_REL).read_text())
    p = sov.context_pack(k)
    assert '"id":"x"' not in p
    assert "NAME_THIS_CLAIM" in p


def test_a_runner_that_raises_becomes_a_result_not_a_lost_turn():
    """execute() runs in a pool now. An exception there used to propagate out of
    the turn; it is a RESULT about that runner instead."""
    src = Path(sov.__file__).read_text()
    assert "runner raised" in src
    assert "cf.ThreadPoolExecutor" in src
    assert "as_completed" in src


def test_concurrent_work_is_capped_at_the_same_bound_as_acceptance():
    """Accepting at most MAX_WORK_PER_TURN and then launching more than that
    would be two different limits pretending to be one."""
    src = Path(sov.__file__).read_text()
    assert "max_workers=MAX_WORK_PER_TURN" in src
    assert sov.MAX_WORK_PER_TURN == 3


def test_the_iteration_record_says_how_much_ran_concurrently():
    src = Path(sov.__file__).read_text()
    assert '"n_launched_concurrently"' in src
    assert '"concurrent_window_s"' in src


def test_the_kernel_tells_the_resident_the_CURRENT_body():
    """measured_state is what the context pack says about the body the resident
    is reasoning over. It read 27.2896 ms / 36.644 TPS for hours after the
    promotion measured 21.9464 / 45.566."""
    import json
    k = json.loads((sov.REPO / sov.KERNEL_REL).read_text())
    ms = k["measured_state"]
    absolute = json.loads(
        (sov.REPO / "receipts/future/SEALED_DEFAULT_ABSOLUTE.json").read_text()
    )["measured"]
    assert ms["token_ms"] == absolute["gpu_ms_per_token"]
    assert ms["tps"] == absolute["gpu_tps"]
    assert ms["basis"] == "GPU"


def test_the_old_field_names_are_removed_not_left_beside_the_new_ones():
    """Two names for one quantity is how a reader picks the stale one."""
    import json
    ms = json.loads((sov.REPO / sov.KERNEL_REL).read_text())["measured_state"]
    assert "wall_ms_per_token" not in ms
    assert "wall_tps" not in ms


def test_refresh_changes_only_the_measured_state():
    """The kernel is durable: hypotheses, scars and iterations must survive a
    refresh, or --init would be the only way to pick up a baseline and that
    would discard everything the resident has learned."""
    import json
    k = json.loads((sov.REPO / sov.KERNEL_REL).read_text())
    keep = {kk: json.dumps(k[kk], sort_keys=True) for kk in k
            if kk != "measured_state"}
    sov.refresh_measured_state(k)
    after = {kk: json.dumps(k[kk], sort_keys=True) for kk in k
             if kk != "measured_state"}
    assert after == keep
