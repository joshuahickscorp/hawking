"""G036 pins."""
import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parents[2] / "receipts/headless/TOKENIZER_GRAVITY.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="G036 receipt not built")


def rec():
    return json.load(open(R))


def test_the_four_organs_are_treated_as_one_system():
    o = rec()["organ_system"]
    assert set(o["coupled"]) == {"tokenizer", "vocabulary", "embedding", "lm_head"}
    assert o["combined_bytes"] == o["embedding_bytes"] + o["lm_head_bytes"]
    assert o["payload_share_pct"] > 10


def test_the_corpus_is_a_real_hcli_workload_not_generic_text():
    c = rec()["corpus"]
    for src in ("agent_prompts_and_tool_schemas", "python_source", "json_receipts",
                "model_replies", "paths", "shell"):
        assert c["provenance_chars"].get(src, 0) > 0, src


def test_genome_has_all_four_classes():
    g = rec()["genome"]
    for k in ("REQUIRED", "HOT", "WARM", "COLD"):
        assert g[k] > 0, k


def test_the_byte_alphabet_is_required_and_was_actually_found():
    """Qwen is byte-level BPE with no <0xNN> rows; the first attempt matched that
    sentencepiece pattern, found zero, and would have allowed byte coverage to be
    deleted."""
    g = rec()["genome"]
    assert g["byte_fallback_rows"] == 256
    assert g["REQUIRED"] >= 256


def test_every_candidate_is_scored_on_inflation_not_only_bytes():
    for c in rec()["candidates"]:
        assert "token_inflation_x" in c
        assert "payload_pct_saved" in c
        assert "net_wall_time_x" in c


def test_the_no_change_control_calibrates_the_measurement():
    d = rec()
    ctrl = next(c for c in d["candidates"] if c["candidate"] == "no_change_control")
    assert ctrl["rows_removed"] == 0
    assert ctrl["payload_pct_saved"] == 0
    assert abs(ctrl["token_inflation_x"] - 1.0) < 0.15
    assert d["measurement_is_calibrated"] is True


def test_the_control_is_labelled_a_control_not_a_loss():
    ctrl = next(c for c in rec()["candidates"] if c["candidate"] == "no_change_control")
    assert ctrl["verdict"].startswith("CONTROL")


def test_ascii_condensing_is_reproduced_and_left_unadopted():
    """S011 §10: reproduce as a CONTROL, never adopt."""
    d = rec()
    a = next(c for c in d["candidates"] if c["candidate"] == "CONTROL_ascii_condensed")
    assert a["rows_removed"] > 0 and a["payload_pct_saved"] > 0
    assert d["conclusion"]["adoptable_today"] is None


def test_held_out_language_was_probed_and_is_not_from_the_corpus():
    h = rec()["heldout_probe"]
    assert len(h["samples"]) >= 4
    assert h["results"]
    for cand, r in h["results"].items():
        assert r["worst_inflation_x"] > 1.0, cand


def test_every_reducing_candidate_carries_its_heldout_number():
    for c in rec()["candidates"]:
        if c["rows_removed"] > 0:
            assert "heldout_worst_inflation_x" in c, c["candidate"]


def test_the_corpus_fitted_genome_is_worse_than_ascii_on_a_latin_language():
    """The inversion worth keeping: fitting to English+code is worse for German than a
    crude keep-all-ASCII rule."""
    h = rec()["heldout_probe"]["results"]
    ascii_de = h["CONTROL_ascii_condensed"]["per_sample"]["german_prose"]["inflation_x"]
    hot_de = h["hawking_required_hot"]["per_sample"]["german_prose"]["inflation_x"]
    assert hot_de > ascii_de


def test_the_inflation_method_states_its_own_limit():
    m = rec()["inflation_method"]
    assert "not a re-derived BPE" in m["honest_limit"]
    assert "LOWER bound" in m["honest_limit"]


def test_the_conclusion_says_why_cold_is_unsafe():
    c = rec()["conclusion"]
    assert "fact about the corpus" in c["why_cold_is_not_safe"]
    assert len(c["what_would_make_this_adoptable"]) >= 3
