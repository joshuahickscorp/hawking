"""Receipt check for an early O003 ladder round.

QUARANTINE NOTE (2026-09-06): as written by the resident this file imported
`load_artifact` from `hcli.engine`. That function has never existed there --
`git log -S"def load_artifact" -- hcli/` is empty, and the only definition in the
repo is an unrelated `load_artifact(blob)` in research/lab/operators/glm52_pack.py.
The import raised at COLLECTION time, so `pytest hcli/` aborted before running
any of its 1732 tests. The fabricated assertion is removed; the assertions that
check the real receipt are kept, since the receipt does exist.
"""
import json
import os

RECEIPT = "receipts/future/O003_LADDER1.json"


def test_o003_ladder1_receipt_is_well_formed():
    assert os.path.exists(RECEIPT), "the test must fail when the receipt is absent"
    with open(RECEIPT) as f:
        receipt = json.load(f)
    assert receipt["target_ebpw"] == 1.0
    assert len(receipt["classes"]) >= 3
    for c in receipt["classes"]:
        assert len(c) == 7
