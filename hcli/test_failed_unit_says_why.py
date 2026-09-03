"""A unit that failed must record why, on itself.

Three units failed four minutes into a live run and every one of them read:

    failure_context: {}   verifier: None   validation: null

The reason existed. `Mission._integrate` computes it -- the validation that was
rejected, the engine error, the status the model claimed -- and hands it to
`Scheduler.fail`, which attached it exclusively to the REPAIR descendant. The
unit that actually failed kept nothing, so a reader could not tell whether the
verifier bit, the model refused, or the engine errored.

Third time this shape appeared today: tool events recorded `ok: false` with no
reason, structured-output receipts recorded a rejection without the reply. A
failure that does not say why is not a failure anyone can act on.
"""
from __future__ import annotations

from hcli.scheduler import Scheduler
from hcli.workunit import WorkUnit


def _scheduler(tmp_path):
    units = {
        "u1": WorkUnit(id="u1", role="research", description="d", status="ready"),
    }
    return Scheduler(units, 1, workspace=tmp_path)


def test_the_failed_unit_carries_the_validation_that_rejected_it(tmp_path):
    sched = _scheduler(tmp_path)
    sched.units["u1"].status = "running"
    sched.fail(
        "u1",
        context={
            "validation": {"ok": False, "reason": "NO_DETERMINISTIC_VALIDATION"},
            "error": None,
            "status_claimed": "completed",
        },
    )
    ctx = sched.units["u1"].failure_context or {}
    assert ctx["validation"]["reason"] == "NO_DETERMINISTIC_VALIDATION"
    assert ctx["status_claimed"] == "completed", (
        "what the model CLAIMED is evidence too, especially when it disagrees "
        "with the verifier"
    )


def test_a_none_valued_key_is_not_written_as_noise(tmp_path):
    """`error: None` is not information; it crowds out what is."""
    sched = _scheduler(tmp_path)
    sched.units["u1"].status = "running"
    sched.fail("u1", context={"validation": {"ok": False}, "error": None})
    assert "error" not in (sched.units["u1"].failure_context or {})


def test_an_empty_context_leaves_the_unit_alone(tmp_path):
    sched = _scheduler(tmp_path)
    sched.units["u1"].status = "running"
    sched.fail("u1", context={})
    assert not (sched.units["u1"].failure_context or {})


def test_the_repair_descendant_still_gets_it(tmp_path):
    """Negative control: recording it on the parent must not take it from the child."""
    sched = _scheduler(tmp_path)
    sched.units["u1"].status = "running"
    repair = sched.fail("u1", context={"validation": {"ok": False, "reason": "boom"}})
    assert repair is not None, "the repair budget still emits a descendant"
    assert repair.repairs == "u1"
    assert (repair.failure_context or {}).get("validation", {}).get("reason") == "boom"


def test_the_reason_survives_a_reload(tmp_path):
    """It is only diagnosis if it is still there after the worker respawns."""
    sched = _scheduler(tmp_path)
    sched.units["u1"].status = "running"
    sched.fail("u1", context={"validation": {"ok": False, "reason": "verifier red"}})

    reloaded = Scheduler.from_workspace(tmp_path, runtime_count=1)
    ctx = reloaded.units["u1"].failure_context or {}
    assert ctx["validation"]["reason"] == "verifier red"


def test_the_reason_is_bounded_so_repairs_cannot_compound(tmp_path):
    """A repair carries its parent's context into its own prompt.

    Unbounded, an error that quotes its parent's error nests once per repair
    generation. Measured live: 2,531 prompt tokens on the base unit, 12,415 on
    the third repair, entirely on nested copies of one preflight message -- and
    every one of those units then failed the context preflight it was quoting.
    """
    from hcli.scheduler import FAILURE_REASON_CHARS

    sched = _scheduler(tmp_path)
    sched.units["u1"].status = "running"
    sched.fail("u1", context={"error": "E" * 5000, "validation": {"ok": False}})

    stored = (sched.units["u1"].failure_context or {})["error"]
    assert len(stored) < FAILURE_REASON_CHARS + 32
    assert stored.endswith("[truncated]"), "silent truncation hides that there was more"


def test_a_short_reason_is_left_exactly_as_it_is(tmp_path):
    """Negative control: bounding must not mangle the common case."""
    sched = _scheduler(tmp_path)
    sched.units["u1"].status = "running"
    sched.fail("u1", context={"error": "verifier red"})
    assert (sched.units["u1"].failure_context or {})["error"] == "verifier red"


def test_non_string_context_survives_bounding(tmp_path):
    sched = _scheduler(tmp_path)
    sched.units["u1"].status = "running"
    sched.fail("u1", context={"validation": {"ok": False, "reason": "x"}})
    assert (sched.units["u1"].failure_context or {})["validation"]["reason"] == "x"
