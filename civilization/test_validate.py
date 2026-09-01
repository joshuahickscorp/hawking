"""Mutation tests. A validator nobody has watched REFUSE is indistinguishable from
one that always accepts -- the check-that-cannot-fail this program has sealed six
times. Every test here breaks the ledger on purpose and requires a violation."""
import copy, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from validate import validate

HERE = pathlib.Path(__file__).resolve().parent
GOAL = (pathlib.Path.home() / ".claude/ultragoal/hawking-odyssey-maxx-ascension/GOAL.md").read_text()
def state(): return json.loads((HERE / "ROADMAP_STATE.json").read_text())


def test_the_real_ledger_is_defensible():
    assert validate(state(), GOAL) == []


def test_canonical_roadmap_identity_is_bound_and_checked():
    s = state()
    assert s["roadmap_version"] == "H-ROADMAP_CRISPR_EXECUTION_SPECIFICATION_2026-08-27"
    assert s["civilizational_coordinate"] == 0.7
    s["roadmap_hash"] = "0" * 64
    assert any("roadmap_hash" in problem for problem in validate(s, GOAL))


def test_all_five_eras_and_twenty_five_programs_are_explicit():
    s = state()
    assert set(s["era_statuses"]) == {"I", "II", "III", "IV", "V"}
    assert len(s["program_statuses"]) == 25
    s["program_statuses"].pop("V-E_PERPETUAL_HAWKING")
    assert any("25 canonical programs" in problem for problem in validate(s, GOAL))


def test_inflating_completion_to_evidence_coverage_is_REFUSED():
    """The exact mistake the first build made: I-D 9/9 categories, 0/8 obligations,
    printed 100%."""
    s = state(); s["civilization_status"]["I-D_ACCELERATOR"]["completion_pct"] = 100.0
    assert any("inflation" in b for b in validate(s, GOAL))


def test_COMPLETE_with_an_open_gate_is_REFUSED():
    s = state(); c = s["civilization_status"]["I-D_ACCELERATOR"]
    c["status"] = "CIVILIZATION_COMPLETE"
    assert any("open gates" in b for b in validate(s, GOAL))


def test_an_unmapped_obligation_is_REFUSED():
    s = state(); s["unmapped_obligations"] = ["G099"]
    assert any("unmapped" in b for b in validate(s, GOAL))


def test_a_status_count_disagreeing_with_GOAL_md_is_REFUSED():
    """Disk state is authority. Retyping a count here must not survive."""
    s = state(); s["obligation_status_counts"]["VERIFIED"] = 58
    assert any("disagree with GOAL.md" in b for b in validate(s, GOAL))


def test_a_fabricated_test_count_is_REFUSED():
    s = state(); s["last_verified_test_count"] = "458"          # a string, not a run
    assert any("not an integer" in b for b in validate(s, GOAL))
    s = state(); s["test_count_is_from_a_run_not_arithmetic"] = False
    assert any("not marked as coming from a run" in b for b in validate(s, GOAL))


def test_evidence_pct_must_match_its_own_table():
    s = state(); s["civilization_status"]["I-B_DOCTOR"]["evidence_pct"] = 100.0
    assert any("does not match its own category table" in b for b in validate(s, GOAL))


def test_dropping_the_named_gates_is_REFUSED():
    s = state(); s["named_gates"] = {}
    assert any("no named gates" in b for b in validate(s, GOAL))


# --- directive VIII fields. Same law: a rule nobody has watched refuse is decoration.

def test_an_unlabelled_civilization_percentage_is_REFUSED():
    """Directive VIII: every percentage is derivable from evidence categories or
    EXPLICITLY labelled heuristic. 1.0% is neither derivable nor honest unlabelled."""
    s = state(); s["civilization_progress"]["heuristic"] = False
    assert any("not labelled heuristic" in b for b in validate(s, GOAL))
    s = state(); s["civilization_progress"].pop("basis")
    assert any("no basis" in b for b in validate(s, GOAL))


def test_an_unquantified_blocker_is_REFUSED():
    """Directive XII: 'no runtime' is not a blocker. Which runtime? Which missing
    semantic? This rule caught a real one on first run --
    SUDO_PURGE_OR_96GiB_WORKING_SET said only 'a repeatable cold read.'"""
    s = state()
    s["blockers"][0]["quantified_as"] = "storage slow"
    assert any("not quantified" in b for b in validate(s, GOAL))


def test_a_blocker_that_blocks_nothing_is_REFUSED():
    s = state(); s["blockers"][0]["blocks"] = []
    assert any("blocks nothing" in b for b in validate(s, GOAL))


def test_a_missing_dependency_entry_is_REFUSED():
    """Absent is not the same as none. A civilization with no key was never considered."""
    s = state(); s["dependencies"].pop("I-C_GRAVITY_NOETIC")
    assert any("no dependency entry" in b for b in validate(s, GOAL))


def test_a_grok_lane_claiming_alive_with_no_process_is_REFUSED():
    """swgrok records the reason in its own source: `grok-run status` carries no pid
    and reports long-dead lanes as running. A status file is not a pid."""
    s = state()
    s["running_lanes"] = [{"lane": "ghost-lane", "executor": "grok",
                           "detection": "definitive", "alive": True,
                           "task_file": "/nonexistent/ghost/task.md"}]
    assert any("no live process" in b for b in validate(s, GOAL))


def test_a_grok_lane_with_no_task_file_is_REFUSED():
    """The vacuous direction, and it was a live bug: an EMPTY task_file made
    `tf in c` true for every process line, so this lane passed silently whenever
    any grok process existed anywhere on the machine."""
    s = state()
    s["running_lanes"] = [{"lane": "no-path-lane", "executor": "grok",
                           "detection": "definitive", "alive": True, "task_file": ""}]
    assert any("names no task_file" in b for b in validate(s, GOAL))


def test_a_claude_lane_claiming_a_DEFINITIVE_check_is_REFUSED():
    """A workflow agent has no pid. Claiming definitive detection claims a check
    that does not exist."""
    s = state()
    s["running_lanes"] = [{"lane": "wf/agent-1", "executor": "claude",
                           "detection": "definitive", "alive": True}]
    assert any("has no pid to check" in b for b in validate(s, GOAL))


def test_a_lane_that_does_not_say_HOW_it_was_detected_is_REFUSED():
    """A definitive pid check and an mtime guess must not read alike in the ledger."""
    s = state()
    s["running_lanes"] = [{"lane": "mystery", "executor": "claude", "alive": True}]
    assert any("does not say how it was detected" in b for b in validate(s, GOAL))


def test_a_lane_with_an_unknown_executor_is_REFUSED():
    """The detector knew only about Grok and reported 0 lanes while three Claude
    agents were mid-edit. An unrecognised executor must be loud, not invisible."""
    s = state()
    s["running_lanes"] = [{"lane": "x", "executor": "cursor", "alive": True,
                           "detection": "definitive"}]
    assert any("no known executor" in b for b in validate(s, GOAL))


def test_gate_ranks_must_be_a_clean_sequence():
    s = state(); s["next_decisive_gates"][0]["rank"] = 7
    assert any("not a clean 1..N" in b for b in validate(s, GOAL))


def test_a_gate_naming_no_resource_is_REFUSED():
    """Resource conflict is half the ranking. Leaving it implicit is how two lanes
    end up fighting over the same bus while both look individually sensible."""
    s = state(); s["next_decisive_gates"][0].pop("resource")
    assert any("names no resource" in b for b in validate(s, GOAL))


def test_mtime_derived_laws_must_be_labelled_heuristic():
    s = state(); s["laws_since_last_checkpoint"]["heuristic"] = False
    assert any("mtime is not provenance" in b for b in validate(s, GOAL))


def test_finding_ZERO_retractions_in_this_corpus_is_REFUSED():
    """Anti-vacuity. This corpus is known to supersede itself -- AMENDED_IN_PLACE
    markers, a retracted 1.495x GEMM win, a separation that DID NOT REPRODUCE. A
    detector that finds none of them is broken, not lucky."""
    s = state(); s["unresolved_retractions"] = []
    assert any("detector is broken" in b for b in validate(s, GOAL))


def test_completion_evidence_without_a_note_is_REFUSED():
    """A bare category count is exactly what inflated I-D to 100% once already."""
    s = state(); s["completion_evidence"]["I-D_ACCELERATOR"].pop("note")
    assert any("no note" in b for b in validate(s, GOAL))


# --- era sovereignty. Without these the central law of the directive is prose.

def test_a_later_era_civilization_reporting_completion_is_REFUSED():
    """Fusion, HMF and eGPU all carry real receipts. Era I is sovereign, so none of
    them may earn civilization completion -- the directive says advance work never
    does, and a law nothing enforces is a comment."""
    s = state()
    s["civilization_status"]["IV-A_FUSION"]["completion_pct"] = 62.0
    assert any("NEVER earns civilization completion" in b for b in validate(s, GOAL))


def test_a_later_era_civilization_listed_ACTIVE_is_REFUSED():
    s = state()
    s["active_civilizations"] = s["active_civilizations"] + ["IV-A_FUSION"]
    assert any("listed active" in b for b in validate(s, GOAL))


def test_a_later_era_civilization_claiming_a_GRADUATED_status_is_REFUSED():
    """EXPLORING is legal for advance work. ADVERSARIALLY_VERIFIED is not --
    tracked, not graduated."""
    s = state()
    s["civilization_status"]["IV-B_HMF_HGVAS"]["status"] = "ADVERSARIALLY_VERIFIED"
    assert any("advance work is" in b for b in validate(s, GOAL))


def test_a_test_count_with_NO_INTERPRETER_is_REFUSED():
    """The same suite reports 5 failed under the default python3 (3.14, no mlx) and
    all passing under the framework 3.12. The number alone is not a measurement."""
    s = state(); s.pop("test_environment", None)
    assert any("no test_environment" in b for b in validate(s, GOAL))


def test_a_count_from_an_interpreter_WITHOUT_MLX_is_REFUSED():
    s = state()
    s["test_environment"]["version_and_mlx"] = "mlx NOT importable under this interpreter"
    assert any("without mlx" in b for b in validate(s, GOAL))


def test_publishing_a_PASSED_count_while_tests_FAILED_is_REFUSED():
    """A passed-count beside unreported failures is a half-truth."""
    s = state(); s["test_environment"]["failed"] = 5
    assert any("tests FAILED in the run" in b for b in validate(s, GOAL))


def test_a_RESIDENT_committer_is_representable_and_must_name_itself():
    """A launchd job was found committing to this branch every five minutes while
    the census reported zero running lanes -- it landed a commit BETWEEN two of one
    session's own commits. A committer the ledger cannot see is the worst kind to
    miss, so 'resident' is a first-class executor and must say how it was found."""
    s = state()
    s["running_lanes"] = [{"lane": "tools/odyssey_driver.sh", "executor": "resident",
                           "alive": True, "detection": "definitive",
                           "judged_by": "live process in ps"}]
    assert validate(s, GOAL) == []
    s["running_lanes"][0].pop("judged_by")
    assert any("must be named exactly" in b for b in validate(s, GOAL))


def test_the_resident_matcher_does_not_catch_an_ordinary_SHELL():
    """Anti-over-reporting. The first matcher used `"hawking" in line and
    "driver.sh" in line`, which caught the session's own `zsh -c source ...` shell.
    A census that over-reports is as useless as one that under-reports."""
    import build_state
    lanes = build_state.running_lanes()
    for L in lanes:
        if L["executor"] == "resident":
            assert L["lane"].endswith("driver.sh"), L["lane"]
            assert " -c " not in L["lane"], f"caught a shell, not a driver: {L['lane']}"
