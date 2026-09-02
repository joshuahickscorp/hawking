"""A second goal in one workspace must not die on the first goal's DAG.

``GoalCompiler`` names every mission's units ``implement``/``validate`` and
``DagStore`` is ONE workspace-global ``.hcli/dag.json``, so before this the
second Mission built in a workspace raised
``IdentityConflict: WorkUnit id 'implement' already exists with different
content`` from inside ``Mission.__init__``. That is the exact failure the
resident daemon hit: ``last_event=worker_failed`` on every restart, and
``AgentOS._drain_goal_bank`` -- which builds one new Mission per queued goal --
could never complete a second goal, so overnight bank promotion was
structurally impossible.
"""
from __future__ import annotations

import json

from hcli.dag_store import DagStore
from hcli.goal import GoalCompiler
from hcli.mission import Mission


def _units(text: str):
    return GoalCompiler().compile(text)["workunits"].units


def test_two_missions_in_one_workspace_do_not_collide(tmp_path):
    first = Mission(tmp_path, goal="first goal: count the lines in one file")
    assert set(first.scheduler.units) == {"implement", "validate"}
    assert first.retired_dag is None

    # Fails with IdentityConflict without the supersession path.
    second = Mission(tmp_path, goal="second goal: a completely different objective")
    assert set(second.scheduler.units) == {"implement", "validate"}
    assert second.id != first.id


def test_the_superseded_dag_is_archived_not_deleted(tmp_path):
    Mission(tmp_path, goal="first goal: count the lines in one file")
    before = json.loads((tmp_path / ".hcli" / "dag.json").read_text())

    second = Mission(tmp_path, goal="second goal: a completely different objective")

    assert second.retired_dag is not None, "a conflicting DAG must be retired"
    assert second.retired_dag.is_file(), "the retired DAG must survive on disk"
    assert json.loads(second.retired_dag.read_text()) == before
    live = json.loads((tmp_path / ".hcli" / "dag.json").read_text())
    assert live != before, "the live DAG must now belong to the second mission"


def test_a_compatible_dag_is_never_retired(tmp_path):
    """Recovery and re-save of the SAME graph must not archive anything."""
    units = _units("one goal: read a file and report on it")
    DagStore(tmp_path).save(units)
    mission = Mission(tmp_path, goal="one goal: read a file and report on it")
    assert mission.retired_dag is None
    assert not (tmp_path / ".hcli" / "dag-retired").exists()
