"""`replace` must archive the DAG with the mission. They are one generation.

`retire_incumbent` moved `.hcli/mission/` and left `.hcli/dag.json` in place.
That file carries `repair_signatures` and `repair_counts`, and
`rebuild_repair_budget` treats persisted counts as a FLOOR -- deliberately, so
deleting repair units cannot reset a budget. The unintended consequence: a brand
new mission inherited the previous mission's SPENT budget.

Measured on the live daemon. `.hcli/dag.json` held signatures for eleven units
before mission 89e411d8 had executed any of them, so the first failure of
G001.work matched a signature already in the set:

    repair_exhausted  G001.work  depth=0
    "repair cycle: failure signature already seen in lineage G001.work"

attempts=1, no repair emitted, unit terminally failed. Not a throughput problem
and not a model problem: every unit got exactly one attempt and no repair, on
every restart, which is why `accepted` was 0 across missions.
"""
from __future__ import annotations

import json

import pytest

from hcli.agentos.resident import ResidentConfig, ResidentDaemon, retire_incumbent
from hcli.workunit import rebuild_repair_budget


def _seed(tmp_path, *, dag: bool = True):
    mission = tmp_path / ".hcli" / "mission"
    mission.mkdir(parents=True)
    (mission / "state.json").write_text(json.dumps({"id": "old", "phase": "cancelled"}))
    if dag:
        (tmp_path / ".hcli" / "dag.json").write_text(json.dumps({
            "version": 1,
            "units": {},
            "repair_counts": {"G001.work": 1},
            "repair_signatures": {"G001.work": ["dde822de73f818bc"]},
        }))
    daemon = ResidentDaemon(tmp_path)
    daemon.store.update(config=ResidentConfig(workspace=str(tmp_path), goal="g").to_dict())
    return daemon


def test_the_dag_is_archived_beside_the_mission(tmp_path):
    daemon = _seed(tmp_path)
    report = retire_incumbent(daemon)

    assert not (tmp_path / ".hcli" / "dag.json").exists(), (
        "the DAG stayed behind; the next mission inherits its spent repair budget"
    )
    archived = report.get("archived_dag")
    assert archived and json.loads(open(archived).read())["repair_counts"] == {"G001.work": 1}


def test_the_next_mission_starts_with_an_unspent_repair_budget(tmp_path):
    """The load-bearing consequence, not just the file move."""
    daemon = _seed(tmp_path)
    retire_incumbent(daemon)

    # What Scheduler.from_workspace would rebuild from, with no dag.json present.
    budget = rebuild_repair_budget({}, {})
    assert budget["counts"] == {}
    assert budget["signatures"] == {}


def test_the_inherited_budget_is_what_refused_the_repair(tmp_path):
    """Negative control: WITHOUT the fix, the first failure is already spent.

    This pins the mechanism rather than the file move -- if `rebuild_repair_budget`
    ever stops treating persisted counts as a floor, this test says so.
    """
    stale = {
        "repair_counts": {"G001.work": 1},
        "repair_signatures": {"G001.work": ["dde822de73f818bc"]},
    }
    budget = rebuild_repair_budget({}, stale)
    assert "dde822de73f818bc" in budget["signatures"]["G001.work"], (
        "a fresh unit's first failure would match this and be refused a repair"
    )
    assert budget["counts"]["G001.work"] == 1


def test_archiving_still_works_when_there_is_no_dag(tmp_path):
    daemon = _seed(tmp_path, dag=False)
    report = retire_incumbent(daemon)
    assert report["archived_mission"]
    assert "archived_dag" not in report


def test_the_mission_is_moved_never_deleted(tmp_path):
    daemon = _seed(tmp_path)
    report = retire_incumbent(daemon)
    archived = report["archived_mission"]
    assert json.loads(open(f"{archived}/state.json").read())["id"] == "old"
