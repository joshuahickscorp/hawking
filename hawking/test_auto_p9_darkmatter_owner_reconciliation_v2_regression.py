"""Focused regression for P9_DARKMATTER_OWNER_RECONCILIATION_V2.

Exercises the owner-reconciliation block added to the ModelLake Odyssey
scheduling authority: the authority must declare an enforced owner
reconciliation contract that fails closed on unreconciled owners.
"""

import json
from pathlib import Path

AUTHORITY_PATH = (
    Path(__file__).resolve().parents[1]
    / "workspace"
    / "campaign"
    / "odyssey"
    / "MODELLAKE_SCHEDULING_AUTHORITY.json"
)


def _load_authority():
    with AUTHORITY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_owner_reconciliation_v2_block_is_enforced():
    authority = _load_authority()
    execution = authority["execution"]
    block = execution["owner_reconciliation_v2"]
    assert block["schema"] == "hawking.modellake.owner_reconciliation.v2"
    assert block["status"] == "ENFORCED"
    assert block["owner"] == "hawking.odyssey.scheduling_authority"
    assert block["fail_closed_on_unreconciled_owner"] is True
    assert block["receipt"] == (
        "receipts/odyssey-i/MODELLAKE_OWNER_RECONCILIATION_V2.json"
    )


def test_owner_reconciliation_v2_preserves_fail_closed_execution():
    authority = _load_authority()
    execution = authority["execution"]
    assert execution["fail_closed_if_this_file_is_missing_or_invalid"] is True
    assert execution["deletion_armed"] is False
    assert execution["automatic_deletion"] is False