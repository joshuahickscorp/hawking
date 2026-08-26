"""Mutation tests for the checkpoint generator and its validator.

Same law as test_validate.py: a validator nobody has watched REFUSE is
indistinguishable from one that always accepts. Every test here breaks a
checkpoint on purpose and requires a violation.
"""
import copy, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import checkpoint as C

HERE = pathlib.Path(__file__).resolve().parent


def good():
    """A checkpoint built from the real ledger with a minimal citing authored block."""
    return C.build(999, {
        "what_became_physically_true": [
            "Qwen3-VL builds from its real config; without prepare_config it raises "
            "(receipts/headless/ACCELERATOR_VL_GAP_CLOSED.json)"],
        "what_was_refuted": [
            "G023's recorded blocker was false as stated (G023)"],
        "what_changed_in_the_roadmap": [
            "the VL priority arrived stale; disk state won (receipts/headless/"
            "ACCELERATOR_RUNTIME_EXECUTES.json)"],
    })


def test_a_real_checkpoint_is_defensible():
    """The anti-vacuity guard. Without this, a validator that refuses EVERYTHING
    would make every other test in this file pass."""
    assert C.validate(good()) == []


def test_a_hand_written_checkpoint_is_REFUSED():
    """CHECKPOINT_001 was hand-written, which is exactly why 002 needed the
    originating prompt repeated."""
    cp = good(); cp["hand_written"] = True
    assert any("hand-written" in b for b in C.validate(cp))


def test_inflating_completion_to_evidence_coverage_is_REFUSED():
    cp = good()
    name = next(iter(cp["civilizations"]))
    cp["civilizations"][name]["completion_pct"] = 100.0
    assert any("inflation" in b for b in C.validate(cp))


def test_an_untraceable_percentage_is_REFUSED():
    """Directive VIII: never report a generated percentage without showing what
    changed in the evidence beneath it."""
    cp = good()
    name = next(iter(cp["civilizations"]))
    cp["civilizations"][name].pop("percentage_traces_to")
    assert any("do not trace" in b for b in C.validate(cp))


def test_COMPLETE_with_open_gates_is_REFUSED():
    cp = good()
    name = next(n for n, c in cp["civilizations"].items() if c["open_gates"])
    cp["civilizations"][name]["status"] = "CIVILIZATION_COMPLETE"
    assert any("open gates" in b for b in C.validate(cp))


def test_an_authored_claim_that_cites_nothing_is_REFUSED():
    """The whole point of the authored/derived split. An adjective is not evidence."""
    cp = good()
    cp["authored"]["what_became_physically_true"] = ["the system is now much more coherent"]
    assert any("cites nothing" in b for b in C.validate(cp))


def test_an_empty_authored_field_is_REFUSED():
    cp = good()
    cp["authored"]["what_was_refuted"] = []
    assert any("what_was_refuted is empty" in b for b in C.validate(cp))


def test_dropping_the_authored_block_entirely_is_REFUSED():
    cp = good(); cp["authored"] = {}
    assert any("no authored block" in b for b in C.validate(cp))


def test_an_unlabelled_civilization_progress_is_REFUSED():
    cp = good(); cp["civilization_progress"]["heuristic"] = False
    assert any("not labelled heuristic" in b for b in C.validate(cp))


def test_dropping_the_next_wave_is_REFUSED():
    """A checkpoint that cannot say what is next leaves the next session needing
    the prompt again -- the exact failure this generator exists to end."""
    cp = good(); cp["next_decisive_wave"] = []
    assert any("next_decisive_wave" in b for b in C.validate(cp))


def test_regressions_are_computed_against_a_NAMED_prior_checkpoint():
    """'No regressions' is only meaningful if something was compared."""
    cp = good()
    assert cp["regressions"]["basis"]
    cp["regressions"]["basis"] = None
    assert any("no basis" in b for b in C.validate(cp))


def test_a_falling_test_count_is_REPORTED_as_a_regression():
    """Built directly, because the real ledger has no regression right now and a
    regression detector that has never fired is assumed vacuous."""
    state = {"last_verified_test_count": 400, "obligation_status_counts": {"VERIFIED": 41},
             "civilization_status": {}}
    prev = ("ERA_I_CHECKPOINT_001.json",
            {"last_verified_test_count": 460, "obligation_status_counts": {"VERIFIED": 41}})
    found = C.regressions(state, prev)["found"]
    assert any("test count fell 460 -> 400" in f for f in found), found


def test_a_falling_completion_pct_is_REPORTED_as_a_regression():
    state = {"civilization_status": {"I-D_ACCELERATOR": {"completion_pct": 10.0}}}
    prev = ("ERA_I_CHECKPOINT_001.json",
            {"civilizations": {"I-D_ACCELERATOR": {"completion_pct": 50.0}}})
    found = C.regressions(state, prev)["found"]
    assert any("completion fell 50.0 -> 10.0" in f for f in found), found


def test_comparing_against_the_HAND_WRITTEN_001_schema_does_not_crash():
    """CHECKPOINT_001 has a different shape. A comparison that crashes on the older
    schema is a comparison nobody will run."""
    p = HERE / "ERA_I_CHECKPOINT_001.json"
    if not p.is_file():
        return
    doc = json.loads(p.read_text())
    state = json.loads(C.STATE.read_text())
    out = C.regressions(state, ("ERA_I_CHECKPOINT_001.json", doc))
    assert isinstance(out["found"], list)


def test_a_claim_citing_the_CONTROL_PLANE_ITSELF_is_accepted():
    """Regression. The citation regex listed tools/ and receipts/ but not
    civilization/, so correctly-cited claims about the control plane were refused
    as prose. Found by running the real authored block through the validator."""
    cp = good()
    cp["authored"]["what_changed_in_the_roadmap"] = [
        "resource_ownership is derived from ps now (civilization/build_state.py)"]
    assert C.validate(cp) == []


def test_widening_the_regex_did_not_make_it_accept_ANYTHING():
    """The anti-vacuity partner of the test above. A citation rule that accepts
    every sentence is worse than no rule, because it looks like a check."""
    cp = good()
    for prose in ["the system is now much more coherent",
                  "performance improved substantially across the board",
                  "we fixed several issues and it is faster now"]:
        cp["authored"]["what_became_physically_true"] = [prose]
        assert any("cites nothing" in b for b in C.validate(cp)), prose
