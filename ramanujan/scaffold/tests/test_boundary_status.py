"""The local scaffold must stay visibly and operationally Hawking-dependent."""
from __future__ import annotations

from ramanujan.status import local_status


def test_local_status_names_the_closed_hawking_handoff_gate() -> None:
    report = local_status()
    assert report["status"] == "BUILDABLE_BUT_BLOCKED_ON_HAWKING_COMPLETION"
    assert report["handoff"] == {
        "status": "PREPARED_NOT_EXECUTED",
        "trigger": "HAWKING_EVOLUTION_COMPLETE",
        "may_execute_now": False,
        "path": "workspace/campaign/evidence/systems/ramanujan/RAMANUJAN_HANDOFF_CONTRACT.json",
    }
    assert report["authority"] == {
        "ramanujan_research_authorized": False,
        "production_authority": False,
        "self_promotion_forbidden": True,
    }
