"""G023 block re-verification pins. §101: a block must be re-checked against disk,
not believed on the strength of an old note."""
import json
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
R = RH / "NOETIC_COMPILER_BLOCK_CHECK.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="block check not built")


def rec():
    return json.load(open(R))


def test_the_recorded_blocker_was_checked_and_found_false():
    d = rec()["recorded_blocker_is_false_as_stated"]
    assert d["so_the_stated_blocker_does_not_hold"] is True
    assert "qwen30_complete_runtime.rs" in d["actual"]


def test_the_moe_reader_it_found_is_real_and_wired():
    d = rec()["recheck"]
    assert d["moe_capable_decode_readers_found"]
    root = Path(__file__).resolve().parents[2]
    for r in d["moe_capable_decode_readers_found"]:
        p = root / r["file"]
        assert p.is_file() and "/src/" in r["file"]
        assert r["routing_mentions"] >= 5
    mod = (root / "crates/hawking-core/src/model/mod.rs").read_text()
    assert "qwen30_complete_runtime" in mod, "the reader is not exported from the crate"


def test_the_false_positive_search_is_recorded_not_hidden():
    """The first search matched two Q80 kernel benchmarks and wrongly said 'stale'."""
    d = rec()["recheck"]
    assert d["false_positive_rejected"]
    assert "kernel parity" in d["why_rejected"] or "benchmarks" in d["why_rejected"]
    assert "examples/" in d["method"]


def test_the_search_requires_all_three_conditions():
    m = rec()["recheck"]["method"]
    assert "catalog" in m and "qwen3_moe" in m and "src/" in m
    assert "AND" in m


def test_the_revised_blocker_is_narrower_than_the_recorded_one():
    d = rec()["revised_blocker"]
    assert d["unblock_is_generalization_not_creation"] is True
    assert len(d["revised_scope"]) >= 3
    assert "18,867" in d["what"]


def test_the_superseded_size_estimate_is_marked_superseded():
    u = rec()["unblock_specification"]
    assert u["size_estimate"].startswith("SUPERSEDED")


def test_the_self_certified_pass_was_corrected_in_the_receipt_that_made_it():
    d = rec()["self_certified_pass_corrected"]
    assert d["now"] is False
    pipe = json.load(open(RH / "NOETIC_COMPILER_PIPELINE.json"))
    assert pipe["pass"] is False
    assert pipe["pass_corrected_by"].endswith("NOETIC_COMPILER_BLOCK_CHECK.json")


def test_the_pipeline_stages_are_still_honestly_reported():
    s = rec()["stages"]
    assert s["n_blocked"] == 2
    assert set(s["blocked"]) == {"DeviceCompiler", "NoeticExecutable"}
    assert s["n_manual_interventions"] == 0


def test_acceptance_is_reported_as_unmet():
    a = rec()["acceptance_clauses"]
    assert a["n_met"] < a["n_total"]
    assert a["produced_executable_is_coherent"] is False
