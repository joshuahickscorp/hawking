"""An obligation with a verifier needs TWO units: the work, then the gate.

Emitting only the TEST unit produced a mission that could RUN its gates and
never implement anything -- every unit shelled out to pytest on a red gate,
failed in seconds, and repaired. That is the mirror of the cognition-only
failure that preceded it, and equally useless. `_default_dag` already had the
right shape (implement -> validate); the obligation path did not.
"""
from __future__ import annotations

from hcli.goal import GoalCompiler

# MULTI-obligation: a single obligation takes `_default_dag` (implement ->
# validate), which already had the right shape. The obligation loop is what
# lacked it, and it only runs with several obligations.
GOAL = (
    "Operate. "
    "G001 make verification self-hosting. Verified by hcli/test_goal_verifier_synthesis.py "
    "G002 attribute the control-plane gap. Verified by hcli/test_hcli_overhead.py "
    "G003 prove self mutation end to end. Verified by hcli/test_self_mutation_e2e.py "
    "G004 put the context compiler on the runtime path. Verified by hcli/test_context_compiler_runtime.py"
)


def _units(text=GOAL):
    return GoalCompiler().compile(text)["workunits"].units


def test_a_verified_obligation_emits_both_a_gate_and_the_work():
    units = _units()
    gates = [u for u in units.values() if u.verifier]
    work = [u for u in units.values() if not u.verifier]
    assert gates, "no gate unit emitted"
    assert work, (
        "no implementation unit emitted; the mission can run its gates but "
        "can never make one green"
    )


def test_the_gate_depends_on_the_work_not_the_reverse():
    units = _units()
    for gate in [u for u in units.values() if u.verifier]:
        assert gate.dependencies, f"gate {gate.id} depends on nothing"
        assert any(d.endswith(".work") for d in gate.dependencies), (
            f"gate {gate.id} does not depend on its implementation unit"
        )


def test_the_work_is_gpu_decode_and_the_gate_is_cpu_test():
    """One resident at a time; the gate is a command, not a model call."""
    units = _units()
    for u in units.values():
        if u.verifier:
            assert u.resource_class == "TEST", u.resource_class
            assert u.preferred_backend == "cpu", u.preferred_backend
        else:
            assert u.resource_class == "GPU_DECODE", (
                f"{u.id} is {u.resource_class}; cognition units must be admitted one at a time"
            )


def test_an_unverified_obligation_still_emits_a_single_cognition_unit():
    """No verifier means no gate to pair with; do not invent one."""
    units = _units(
        "Operate. G001 make something better. G002 make something else better. "
        "G003 and a third thing, none of them naming a check."
    )
    assert not any(u.id.endswith(".work") for u in units.values()), (
        "a work/gate pair was emitted for an obligation that has no gate"
    )
