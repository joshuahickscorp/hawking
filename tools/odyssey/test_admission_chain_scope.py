"""G023 step-3 scoping pins."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts/headless/ADMISSION_CHAIN_SCOPE.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="scope not run")


def rec():
    return json.load(open(R))


def test_step_2_is_recorded_as_incomplete():
    """Relaxing a constant is not the same as relaxing a binding."""
    d = rec()["MY_STEP_2_WAS_INCOMPLETE"]
    assert len(d["sites"]) >= 2
    assert d["enum_variant_is_hardcoded"] is True
    assert "one layer below where I looked" in d["consequence"]


def test_the_second_binding_really_exists_in_the_source():
    d = rec()["MY_STEP_2_WAS_INCOMPLETE"]
    for site in d["sites"]:
        path, line = site.rsplit(":", 1)
        src = (REPO / path).read_text().splitlines()
        assert "source_repository()" in src[int(line) - 1], site


def test_the_producer_correction_is_recorded():
    """I claimed no producer existed, having not searched lab/."""
    c = rec()["CORRECTION_THE_PRODUCER_EXISTS"]
    assert "did not search lab/" in c["why_that_was_wrong"]
    assert (REPO / c["the_producer"]).is_file()


def test_the_producer_emits_every_schema_admission_requires():
    c = rec()["CORRECTION_THE_PRODUCER_EXISTS"]
    src = (REPO / c["the_producer"]).read_text()
    for schema in c["emits"].values():
        assert schema in src, schema
    assert "from lab.receipts import seal" in src


def test_the_producer_is_parameterized_for_another_model():
    c = rec()["CORRECTION_THE_PRODUCER_EXISTS"]
    src = (REPO / c["the_producer"]).read_text()
    sig = src[src.index("def __init__"):src.index("self.model_dir")]
    for arg in ("repository", "model_id", "artifact_prefix", "schema"):
        assert arg in sig, arg
    # and the qwen80 wrapper proves it is actually reused
    w = (REPO / "tools/condense/ascension_qwen80_complete_gravity.py").read_text()
    assert "CompleteBinaryGravity" in w
    assert "qwen80_complete_binary_gravity" in w


def test_the_representation_conflict_is_named():
    steps = rec()["corrected_step_3"]
    joined = " ".join(steps)
    assert "1-bit signs" in joined and "uniform_q4_group" in joined


def test_the_over_estimation_pattern_is_recorded():
    d = rec()
    assert "third time this blocker has shrunk" in d["honest_size"]
    assert "overstated" in d["pattern_worth_recording"]
