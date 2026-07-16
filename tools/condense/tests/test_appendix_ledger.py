from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "appendix_ledger.py"
SPEC = importlib.util.spec_from_file_location("appendix_ledger", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_static_probe_wraps_as_valid_nonclaiming_receipt() -> None:
    probe = MODULE.tq_runtime_probe.build_probe()
    receipt = MODULE.static_probe_receipt(probe, source_commit="0123456789abcdef")
    assert MODULE.contract.validate_receipt(receipt) == []
    assert receipt["analytic_scope"]["physical_speed_claim"] is False
    assert receipt["analytic_scope"]["cell_count"] == len(probe["cells"])
    assert receipt["resources"]["observed"] is False


def test_rollup_preserves_invalid_receipt_instead_of_hiding_it() -> None:
    receipt = MODULE.static_probe_receipt(
        MODULE.tq_runtime_probe.build_probe(), source_commit="0123456789abcdef"
    )
    broken = copy.deepcopy(receipt)
    broken["receipt_sha256"] = "0" * 64
    rollup = MODULE.rollup_receipts([receipt, broken])
    assert rollup["receipt_count"] == 2
    assert rollup["valid_count"] == 1
    assert rollup["invalid_count"] == 1
    assert rollup["receipts"][1]["errors"]
