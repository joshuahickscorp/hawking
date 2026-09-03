"""G023 prerequisite-chain pins."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts/headless/COMPILER_PREREQUISITE_CHAIN.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="chain not mapped")


def rec():
    return json.load(open(R))


def test_every_named_producer_that_claims_to_exist_really_does():
    for c in rec()["chain"]:
        if c.get("producer") and c["producer"].startswith(("research/lab/", "tools/")):
            assert (REPO / c["producer"]).is_file(), c["producer"]
            assert c["producer_exists"] is True


def test_the_audit_producer_really_hardcodes_the_repository():
    """Unlike CompleteBinaryGravity, which takes it as a constructor argument."""
    c = next(x for x in rec()["chain"] if x["step"] == 1)
    src = (REPO / c["producer"]).read_text()
    assert 'SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-30B-A3B-Instruct"' in src
    assert c["constant_use_sites"] >= 3
    sig = src[src.index("def __init__"):src.index("def __init__") + 200]
    assert "repository" not in sig


def test_the_gravity_producer_really_is_parameterized():
    c = next(x for x in rec()["chain"] if x["step"] == 2)
    assert set(c["already_parameterized"]) >= {"repository", "model_id", "schema"}
    assert c["blocker"] is None


def test_the_existing_audit_cannot_be_reused():
    """It audits the Coder model, and the directory it audited is gone."""
    a = rec()["the_existing_audit_is_not_reusable"]
    assert a["covers_repository"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert a["audited_model_dir_still_exists"] is False
    assert not Path(a["audited_model_dir"]).exists()


def test_model_2_source_is_actually_present_for_step_1():
    c = next(x for x in rec()["chain"] if x["step"] == 1)
    assert c["input_present"] is True
    assert Path(c["input"]).is_dir()


def test_admission_work_is_recorded_as_done():
    c = next(x for x in rec()["chain"] if x["step"] == 3)
    assert c["blocker"] is None
    assert c["layers_opened"] == 3


def test_the_one_missing_producer_is_named():
    missing = [c for c in rec()["chain"] if not c["producer_exists"]]
    assert len(missing) == 1
    assert "parity harness" in missing[0]["blocker"]


def test_the_summary_does_not_promise_coherence():
    d = rec()
    assert "not invent" in d["honest_summary"]
    assert "does not compose" in d["what_is_NOT_claimed"]
