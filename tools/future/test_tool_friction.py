"""The friction meter must count, not estimate, and must never write to .hcli."""
from __future__ import annotations

import pytest

from tools.future import tool_friction as F


@pytest.fixture(scope="module")
def doc():
    return F.census()


def test_the_census_counts_real_calls(doc):
    assert doc["tool_calls"] > 0, "no tool calls found; the log shape changed"
    assert 0 <= doc["failed"] <= doc["tool_calls"]
    # the census rounds to 4 places; compare at that precision
    assert doc["failure_rate"] == pytest.approx(doc["failed"] / doc["tool_calls"], abs=1e-4)


def test_tool_execution_is_a_rounding_error_next_to_model_wall(doc):
    """The finding that makes friction matter: what a failed call wastes is not
    its own execution, it is the model turn that has to recover from it."""
    assert doc["tool_execution_share_of_wall"] < 0.05, doc["tool_execution_share_of_wall"]
    assert doc["model_wall_s"] > doc["tool_execution_wall_s"] * 50


def test_the_file_not_found_split_is_exhaustive(doc):
    fnf = doc["file_not_found"]
    assert fnf["total"] == (fnf["present_now"] + fnf["recoverable_wrong_directory_or_near_name"]
                            + fnf["genuinely_absent"])
    assert fnf["total"] > 0


def test_a_path_that_exists_now_is_classified_present_not_absent(doc):
    """The mangled-filename case. HCLI_SELF_IMPROVEMENT_DIRECTIVE.md was requested
    59 times across 55 goals while the file sat on disk as ...DIRECTifact.md.
    Classifying that as ABSENT would hide the most expensive recoverable failure
    in the whole ledger."""
    present = [r for r in doc["missing_paths"] if r["kind"] == "PRESENT_NOW"]
    assert present, "nothing classified PRESENT_NOW; the rename may have regressed"
    worst = doc["missing_paths"][0]
    assert worst["kind"] in ("PRESENT_NOW", "WRONG_DIRECTORY", "NEAR_NAME", "ABSENT")


def test_the_cost_is_a_range_and_never_a_point_estimate(doc):
    w = F.wasted_model_wall_s(doc)
    assert w["pro_rata_s"] < w["per_turn_upper_bound_s"], w
    assert "invented" in w["note"]
    assert w["calls_per_model_turn"] > 1, (
        "more model calls than tool calls would make the per-turn bound the wrong shape"
    )


def test_the_meter_only_reads(tmp_path, monkeypatch):
    """It inspects another campaign's logs. It must never write to them."""
    import inspect
    src = inspect.getsource(F)
    for forbidden in ("write_text", "open(", "unlink", "mkdir"):
        if forbidden == "open(":
            continue
        assert forbidden not in src.replace("write_receipt", ""), forbidden
