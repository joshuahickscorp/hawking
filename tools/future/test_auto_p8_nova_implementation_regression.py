"""Focused regression for the Nova lineage measurement_axes declaration gate.

Exercises the production behavior added to tools/future/gravity_nova_lineage.py:
a lineage that declares an unknown measurement axis is refused, and a lineage
that omits a canonical axis from its declaration is refused with that axis
named in ``missing``. This is a plan-only validation check; it loads no
weights, writes no artifact, and makes no release or hardware claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.gravity_nova_lineage import MEASUREMENT_AXES, validate

PARENT_HASH = "a" * 64
DESCENDANT_HASH = "b" * 64


def _lineage(**overrides):
    lineage = {
        "schema": "hawking.gravity.nova_lineage.v1",
        "parent": {"identity": "KIMI_BASE", "artifact_hash": PARENT_HASH},
        "descendant": {"identity": "NOVA_CANDIDATE", "artifact_hash": DESCENDANT_HASH},
        "objective": "raise capability without touching the frozen parent",
        "transformation": {
            "kind": "lora_merge",
            "changed_tensors": ["layers.0.attn.q_proj"],
            "train_set": ["nova_train_v1"],
            "freeze_set": ["nova_freeze_v1"],
            "data_class": "synthetic_curriculum",
            "teacher": "KIMI_BASE",
            "optimization_objective": "next_token_ce",
            "representation_constraints": ["fp16_round_trip"],
        },
        "measurements": {
            axis: {"before": "0.0", "after": "1.0", "receipt": f"receipt:{axis}"}
            for axis in MEASUREMENT_AXES
        },
        "rollback": {
            "reversible": True,
            "recipe": "restore parent artifact hash",
            "parent_immutable": True,
        },
    }
    lineage.update(overrides)
    return lineage


def test_baseline_lineage_is_accepted():
    result = validate(_lineage())
    assert result["status"] == "LINEAGE_ACCEPTED", result
    assert result["missing"] == []
    assert result["errors"] == []


def test_unknown_declared_axis_is_refused():
    lineage = _lineage(measurement_axes=list(MEASUREMENT_AXES) + ["vibes"])
    result = validate(lineage)
    assert result["status"] == "LINEAGE_REFUSED", result
    assert any("unknown axes" in error for error in result["errors"]), result
    assert "vibes" in " ".join(result["errors"])


def test_undeclared_canonical_axis_is_missing():
    declared = [axis for axis in MEASUREMENT_AXES if axis != "physical"]
    lineage = _lineage(measurement_axes=declared)
    result = validate(lineage)
    assert result["status"] == "LINEAGE_REFUSED", result
    assert "measurement_axes.physical" in result["missing"], result


def test_non_sequence_declaration_is_refused():
    lineage = _lineage(measurement_axes="capability")
    result = validate(lineage)
    assert result["status"] == "LINEAGE_REFUSED", result
    assert any("must be a sequence" in error for error in result["errors"]), result


if __name__ == "__main__":
    test_baseline_lineage_is_accepted()
    test_unknown_declared_axis_is_refused()
    test_undeclared_canonical_axis_is_missing()
    test_non_sequence_declaration_is_refused()
    print("P8_NOVA_IMPLEMENTATION measurement_axes gate: 4 checks passed")