"""Abort must persist one coherent checkpoint generation, DAG first.

Pre-fix abort() patched mission/state.json in place (phase, cancel_reason,
last_checkpoint) and never wrote dag.json. The two files then named
different generations: state.json carried a cancelled epoch that dag.json
could not, and neither file received a new shared checkpoint_id.

Mission.checkpoint() is the contract: persist the DAG first, re-raise on
DAG failure, stamp the same checkpoint_id into both files. This suite
fails against the unpaired state write and passes when abort reuses that
contract.
"""
from __future__ import annotations

import json
from pathlib import Path

from hcli import delegate as d
from hcli.dag_store import DagStore
from hcli.mission import mission_state_path
from hcli.workunit import WorkUnit


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dag_path(ws: Path) -> Path:
    return Path(ws) / ".hcli" / "dag.json"


def _mission(tmp_path) -> Path:
    ws = Path(tmp_path) / "abort-atomicity"
    d.run("abort must not split generations", workspace=ws, spawn=False)
    return ws


def test_abort_mints_one_shared_checkpoint_generation(tmp_path):
    """A new checkpoint_id must land in BOTH files, and they must agree."""
    ws = _mission(tmp_path)
    state_path = mission_state_path(ws)
    dag_path = _dag_path(ws)
    pre_state = _load(state_path)
    pre_dag = _load(dag_path)
    pre_id = pre_state.get("checkpoint_id")
    assert isinstance(pre_id, str) and pre_id
    assert pre_dag.get("checkpoint_id") == pre_id

    out = d.abort(ws, reason="operator pulled the plug")
    assert out["verdict"] == "ABORTED"

    post_state = _load(state_path)
    post_dag = _load(dag_path)
    post_id = post_state.get("checkpoint_id")
    assert isinstance(post_id, str) and post_id
    assert post_dag.get("checkpoint_id") == post_id
    assert post_id != pre_id
    assert post_state["phase"] == "cancelled"
    assert post_state["cancel_reason"] == "operator pulled the plug"


def test_abort_cannot_leave_dag_and_state_at_different_generations(tmp_path):
    """Plant a DAG-only unit and a DAG-only marker, then abort.

    Unpaired state writes keep the marker, leave the unit out of state.json,
    and reuse the old checkpoint_id. A real checkpoint rewrites dag.json
    (marker gone), copies the DAG unit into state.json, and names both
    files with a new shared checkpoint_id.
    """
    ws = _mission(tmp_path)
    state_path = mission_state_path(ws)
    dag_path = _dag_path(ws)
    pre_id = _load(state_path)["checkpoint_id"]

    planted = WorkUnit(
        id="planted",
        role="work",
        description="unit that exists only in dag.json before abort",
        resource_class="LIGHT_CONTROL",
    )
    store = DagStore(ws)
    units = store.load(recover_running=False)
    units[planted.id] = planted
    store.save(
        units,
        extra={
            "checkpoint_id": pre_id,
            "abort_probe": "stale-if-dag-not-rewritten",
        },
    )
    split_dag = _load(dag_path)
    split_state = _load(state_path)
    assert "planted" in split_dag["units"]
    assert "planted" not in (split_state.get("units") or {})
    assert split_dag.get("abort_probe") == "stale-if-dag-not-rewritten"

    d.abort(ws, reason="heal the split")

    post_state = _load(state_path)
    post_dag = _load(dag_path)
    assert post_state.get("checkpoint_id") == post_dag.get("checkpoint_id")
    assert post_state.get("checkpoint_id") != pre_id
    assert "abort_probe" not in post_dag
    assert "planted" in post_dag["units"]
    assert "planted" in post_state["units"]
    assert post_state["units"]["planted"]["status"] == post_dag["units"]["planted"]["status"]
    assert post_state["phase"] == "cancelled"
    assert post_state["cancel_reason"] == "heal the split"
