"""Validate the receipts the saturation-wave Grok lanes are contracted to write.

Each lane's contract names a receipt and its required keys. This is the lane's
VERIFY command, so it has to fail on the two ways a lane actually goes wrong:
the receipt is absent, or it is present and hollow.

A field the lane could not establish is required to be the string
``UNKNOWN: <why>``. That is an honest, passing answer. What does not pass is a
key that is missing, empty, or still carrying a placeholder - a lane that
invents a number to fill a slot is worse than one that admits it has none.

A receipt that does not exist yet SKIPS rather than fails, so one lane's
verification does not fail on another lane's absence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECEIPTS = REPO / "receipts" / "future"

# receipt filename -> keys the contract requires
CONTRACTED = {
    "RESIDENT_TPS_RECON.json": [
        "measured_tps", "measurement_command", "measurement_class", "lever_tried",
        "control", "output_equality", "baseline_token_ns", "variant_token_ns", "verdict",
    ],
    "G015_HARVEST.json": [
        "artifacts_found", "window_change_effect", "invariant_holds",
        "recommendation", "remaining_defect", "cheapest_settling_run",
    ],
    "FORBIDDEN_FRUIT_LAB_READINESS.json": [
        "lab_capabilities", "reachable_by_hcli", "probe_output",
        "smallest_missing_piece", "verdict",
    ],
    "SUB2_TOOLING_READINESS.json": [
        "resident_unsupported_requests", "capability_inventory",
        "built_anything", "justification",
    ],
    # Distillation wave: four subtraction audits. Each recommends rather than
    # deletes, so the receipt IS the deliverable and has to be checkable.
    "AIDER_NAMESPACE_AUDIT.json": [
        "aider_import_call_sites", "aider_binary_invocations",
        "haider_namespace_files", "suite_runs_without_aider",
        "safe_to_remove", "must_keep_as_evidence", "recommended_commands",
    ],
    "DEAD_CALLSITE_SWEEP.json": [
        "capabilities_examined", "zero_caller_modules", "test_only_callers",
        "superseded_paths", "recommended_deletions", "must_not_delete",
    ],
    "CONFIG_ENV_AUDIT.json": [
        "hcli_env_vars", "unused_vars", "conflicting_defaults",
        "body_specific_names", "stale_config_files",
        "recommended_normalization",
    ],
    "NOMENCLATURE_AUDIT.json": [
        "model_specific_names_in_generic_roles", "genuinely_generic_already",
        "safe_renames", "unsafe_renames", "compatibility_aliases_needed",
        "recommended_sequence",
    ],
}

PLACEHOLDERS = {"", "TODO", "TBD", "N/A", "null", "None", "...", "-"}


def _hollow(value) -> bool:
    """A key that is present but says nothing."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in PLACEHOLDERS
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


@pytest.mark.parametrize("name,keys", sorted(CONTRACTED.items()))
def test_contracted_receipt_is_complete_or_honestly_unknown(name, keys):
    path = RECEIPTS / name
    if not path.is_file():
        pytest.skip(f"{name} not written yet")
    data = json.loads(path.read_text(encoding="utf-8"))

    missing = [k for k in keys if k not in data]
    assert not missing, f"{name} missing contracted keys: {missing}"

    hollow = [k for k in keys if _hollow(data[k])]
    assert not hollow, (
        f"{name} has hollow keys {hollow}. A field you could not establish must "
        f'be the string "UNKNOWN: <why>", not empty and not a placeholder.'
    )

    assert data.get("commands"), f"{name} records no commands actually run"
    assert data.get("recorded_at"), f"{name} has no recorded_at"


@pytest.mark.parametrize("name", sorted(CONTRACTED))
def test_numeric_fields_are_numbers_or_declared_unknown(name):
    """A measurement slot holds a number or says why it is empty - never prose."""
    path = RECEIPTS / name
    if not path.is_file():
        pytest.skip(f"{name} not written yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("measured_tps", "baseline_token_ns", "variant_token_ns"):
        if key not in data:
            continue
        value = data[key]
        ok = isinstance(value, (int, float)) or (
            isinstance(value, str) and value.startswith("UNKNOWN:")
        )
        assert ok, (
            f"{name}:{key} is {value!r}. A measurement is a number, or the "
            f'string "UNKNOWN: <why>". Prose in a measurement slot is how a '
            f"campaign ends up citing a figure nobody measured."
        )


if __name__ == "__main__":
    written = 0
    for name, keys in sorted(CONTRACTED.items()):
        if not (RECEIPTS / name).is_file():
            print(f"skip  {name} (not written yet)")
            continue
        test_contracted_receipt_is_complete_or_honestly_unknown(name, keys)
        test_numeric_fields_are_numbers_or_declared_unknown(name)
        print(f"ok    {name}")
        written += 1
    print(f"{written}/{len(CONTRACTED)} contracted receipts present and complete")
