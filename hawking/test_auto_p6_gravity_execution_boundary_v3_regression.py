"""Focused regression for P6_GRAVITY_EXECUTION_BOUNDARY_V3.

Exercises the executable/readable artifact boundary added to
``hawking.gravity``: only the machine-bound ``.nx`` executable may be handed
to the executor, while ``.nr`` shards and historical ``.gravity`` artifacts
remain readable inputs and are never execution targets.

No release and no hardware qualification is claimed by this test.
"""

from hawking.gravity import (
    EXECUTABLE_SUFFIXES,
    READABLE_SUFFIXES,
    is_executable_artifact,
    is_readable_artifact,
)


def test_executable_suffixes_are_machine_bound_only():
    assert EXECUTABLE_SUFFIXES == (".nx",)
    assert ".nr" not in EXECUTABLE_SUFFIXES
    assert ".gravity" not in EXECUTABLE_SUFFIXES


def test_nx_artifact_is_executable():
    assert is_executable_artifact("build/plan.nx") is True
    assert is_executable_artifact("PLAN.NX") is True
    assert is_executable_artifact("  plan.nx  ") is True


def test_transient_and_historical_artifacts_are_not_executable():
    assert is_executable_artifact("shard.nr") is False
    assert is_executable_artifact("legacy.gravity") is False
    assert is_executable_artifact("plan.nx.bak") is False
    assert is_executable_artifact("plan") is False
    assert is_executable_artifact("") is False
    assert is_executable_artifact(None) is False


def test_readable_artifacts_cover_transient_and_historical():
    assert READABLE_SUFFIXES == (".nr", ".gravity")
    assert is_readable_artifact("shard.nr") is True
    assert is_readable_artifact("legacy.gravity") is True
    assert is_readable_artifact("SHARD.NR") is True
    assert is_readable_artifact("plan.nx") is False
    assert is_readable_artifact(None) is False


def test_boundary_is_disjoint():
    for suffix in EXECUTABLE_SUFFIXES:
        assert suffix not in READABLE_SUFFIXES
    for suffix in READABLE_SUFFIXES:
        assert suffix not in EXECUTABLE_SUFFIXES