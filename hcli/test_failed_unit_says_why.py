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
