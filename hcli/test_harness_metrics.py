"""The autonomy metric must be uninflatable.

S008 section 18 makes VERIFIED SCIENTIFIC PROGRESS PER SUPERVISOR INTERVENTION
the critical number. A metric the supervisor can talk its way past measures
nothing, so the arithmetic is pinned here.
"""
from hcli.harness_metrics import autonomy, ladder_stats


def test_below_one_is_named_as_supervisor_produced():
    a = autonomy(1, 20)
    assert a["progress_per_intervention"] == 0.05
    assert "supervisor is producing the science" in a["reading"]


def test_at_or_above_one_is_named_as_harness_carrying():
    assert "harness is carrying" in autonomy(20, 20)["reading"]


def test_harness_repairs_count_as_interventions():
    """Autonomy that needs a human to keep repairing its instrument is not
    autonomy. Excluding repairs would let a broken harness score well."""
    from hcli.harness_metrics import supervisor_interventions
    s = supervisor_interventions("HEAD~20")
    assert s["commits"] >= s["harness_repair"] > 0


def test_zero_interventions_does_not_divide():
    assert autonomy(5, 0)["progress_per_intervention"] is None


def test_ladder_stats_counts_non_executions_too():
    """A loop that is alive but not executing must not look productive."""
    s = ladder_stats()
    assert s["rounds"] > s["executed"], "every round executed? verify the log"
    assert s["rounds"] == sum(
        s.get(k, 0) for k in
        ("EXECUTED", "REFUSED_UNRUNNABLE", "REPEATED_A_SPEC_ALREADY_ON_THE_TABLE",
         "WRONG_DIRECTION", "NO_SPEC_EMITTED", "RUNNER_FAILURE")), "rounds unaccounted"
