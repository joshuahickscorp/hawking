"""Pure checks for the new current-HCLI L0 route-capture boundary."""
from __future__ import annotations

from lab.operators.ascension_qwen30_quality_repack_current_hcli_route_capture import (
    EXPECTED_PROBES,
    INPUT_SCHEMA,
    STATUS,
    TRACE_STATUS,
)


def test_capture_is_explicitly_new_diagnostic_not_historical() -> None:
    assert TRACE_STATUS == "NEW_DIAGNOSTIC_NOT_HISTORICAL"
    assert "UNQUALIFIED" in STATUS
    assert INPUT_SCHEMA.endswith(".v1")


def test_capture_preserves_exact_protected_probe_order() -> None:
    assert EXPECTED_PROBES == ("literal_hawking", "json_status", "python_add")
