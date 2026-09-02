"""A cognition WorkUnit opens an 11 GB model body. Only one may run at a time.

LIVE FAILURE, reproduced twice: the sovereign ultragoal compiled to eight
obligations, every one of them a cognition unit. `WorkUnit.resource_class`
defaults to LIGHT_CONTROL, whose limit is 128, so the scheduler dispatched all
eight at once and each opened its own resident. Free RAM fell to 0.2 GB, swap
delta hit 42.7 GB, and the supervisor correctly refused with
WAITING_FOR_MEMORY. GPU_DECODE has a limit of 1.
"""
from __future__ import annotations

from hcli.goal import GoalCompiler
from hcli.resources import ResourceLimits
from hcli.workunit import identify_ready

# The ACTUAL text that reproduced the runaway, trimmed to its first 8
# numbered priorities. A synthetic fixture did not reproduce the obligation
# split, so this uses the real shape.
MULTI = (
    "HAWKING SOVEREIGN ULTRAGOAL. Objective: VERIFIED FRONTIER MOVEMENT PER "
    "UNATTENDED WALL-CLOCK HOUR. Priorities, highest first. "
    "(1) HCLI_SELF: reduce the ~2x Python overhead between the CLI and the "
    "resident. (2) PREFILL/GEMM: 353 of 755 tensors are f32 and have no "
    "prefill GEMM. (3) CONTEXT COMPILER: extend retrieval so exact source "
    "spans are pulled on demand. (4) RESIDENT PHYSICAL PERFORMANCE: decode "
    "and prefill measured, with receipts. (5) MODELLAKE: acquisition is "
    "healthy; do not re-enable the rate refresh. (6) ODYSSEY: advance the "
    "curriculum only with verified specimens. (7) FORBIDDEN FRUIT: the ANE "
    "lab reports OBSERVED placement only. (8) ROADMAP: S016 remains "
    "NOT_STARTED."
)


def _units(text):
    return GoalCompiler().compile(text)["workunits"].units


def test_a_multi_obligation_goal_makes_many_units():
    units = _units(MULTI)
    # The COUNT depends on the compiler's sentence split and is not the
    # point; more than one cognition unit is what makes concurrency real.
    assert len(units) >= 2, f"fixture must produce several units, got {len(units)}"


def test_cognition_units_are_not_light_control():
    """LIGHT_CONTROL's limit is 128; a unit that opens the resident is not light."""
    units = _units(MULTI)
    cognition = [u for u in units.values() if not u.verifier]
    assert cognition, "fixture produced no cognition units"
    for u in cognition:
        assert u.resource_class != "LIGHT_CONTROL", (
            f"unit {u.id} would be dispatched 128-wide and open 128 model bodies"
        )
        assert u.resource_class == "GPU_DECODE", u.resource_class


def test_the_limit_that_now_applies_is_one():
    limits = ResourceLimits.resolve(repo_root=".")
    by_class = limits.by_class() if hasattr(limits, "by_class") else None
    assert limits.gpu_decode == 1, "GPU_DECODE must admit exactly one resident"
    assert limits.light_control > 8, "fixture assumes LIGHT_CONTROL is wide"


def test_every_compiled_obligation_unit_is_gpu_decode():
    """Every compiled obligation unit is GPU_DECODE, admitted one at a time."""
    units = _units(MULTI)
    assert len(units) >= 2, f"expected several obligation units, got {len(units)}"
    assert {u.resource_class for u in units.values()} == {"GPU_DECODE"}
    assert ResourceLimits.resolve(repo_root=".").gpu_decode == 1
