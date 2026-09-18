"""Focused regression for the Nova lineage measurement_axes contract.

Exercises the production validator in tools/future/gravity_nova_lineage.py:
a declared measurement_axes sequence that repeats an axis name must be
refused with an explicit error, while a complete, duplicate-free sequence
must not raise that error. This is a plan-only bridge check; it does not
load weights, write artifacts, promote candidates, or qualify hardware.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.gravity_nova_lineage import (  # noqa: E402
    MEASUREMENT_AXES,
    SCHEMA,
    validate,
)

PARENT_HASH = "a" * 64
DESCENDANT_HASH = "b" * 64


def _lineage(axes):
    return {
        "schema": SCHEMA,
        "parent": {"identity": "KIMI_BASE", "artifact_hash": PARENT_HASH},
        "descendant": {"identity": "NOVA_CANDIDATE", "artifact_hash": DESCENDANT_HASH},
        "objective": "bounded plan-only Nova lineage",
        "measurement_axes": axes,
        "transformation": {
            "kind": "lora_merge",
            "changed_tensors": ["layer.0.q_proj"],
            "train_set": ["train-a"],
            "freeze_set": ["freeze-a"],
            "data_class": "synthetic",
            "teacher": "none",
            "optimization_objective": "minimize loss",
            "representation_constraints": "none",
        },
        "measurements": {
            axis: {"before": "b0", "after": "b1", "receipt": "r0"}
            for axis in MEASUREMENT_AXES
        },
        "rollback": {"reversible": True, "plan": "restore parent hash"},
    }


def test_duplicate_axis_is_refused():
    result = validate(_lineage(list(MEASUREMENT_AXES) + [MEASUREMENT_AXES[0]]))
    assert result["status"] == "LINEAGE_REFUSED", result
    assert "measurement_axes must not repeat an axis name" in result["errors"], result


def test_complete_axis_set_is_not_refused_for_duplicates():
    result = validate(_lineage(list(MEASUREMENT_AXES)))
    assert "measurement_axes must not repeat an axis name" not in result["errors"], result


if __name__ == "__main__":
    test_duplicate_axis_is_refused()
    test_complete_axis_set_is_not_refused_for_duplicates()
    print("P8_NOVA_MEASUREMENT_AXES_REGRESSION_OK")