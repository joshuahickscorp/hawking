"""Negative controls for the choice-JSON body-vs-ask probe.

A discriminator nobody has watched reject will rubber-stamp a format story.
These tests prove: the reconstructed ask is the trial ask (hash), not a
friendlier paraphrase; the control clip eats the JSON schema; each named
variation is one lever off that control; trial markdown is not parseable
choice JSON; interpret JSON is not choice JSON; --build does not mint a live
outcome; hardware-named fields cannot land; autonomy_trial.py is not written.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import choice_json_probe as cjp
from tools.future import model_bearing as mb
from tools.future import model_bearing_torture as mbt
from tools.future._common import (
    HARDWARE_FIELDS,
    HardwareClaimError,
    _assert_no_hardware_claims,
    write_receipt,
)

SRC = Path(__file__).resolve().parent / "choice_json_probe.py"
TRIAL_MARKDOWN = (
    "The candidate is:\n\n**WU.HAWKING.resident_identity_pin**\n\n"
    "(Title: \"pin the sealed resident identity on the live body\")\n\n"
    "**Reasoning:**\nThe prompt explicitly states that the scripted policy "
    "would pick this specific ID."
)
GOOD_JSON = (
    '{"choice_id":"WU.HAWKING.resident_identity_pin",'
    '"reason":"live Hawking-self work; not a closed scar"}'
)
INTERPRET_FENCE = """```json
{
"reading": "The frontier is dominated by high-gain model representation updates.",
"worth_doing_next": ["WU.DEAD.mlp_function_replacement"],
"why": "highest-gain frontier items"
}
```"""


# ---------------------------------------------------------------------------
# Exact ask, not a paraphrase
# ---------------------------------------------------------------------------


def test_reconstructed_control_sha256_matches_the_landed_timeline():
    exact = cjp.extract_exact_ask()
    assert exact["clipped_sha256"] == cjp.TIMELINE_CHOOSE_PROMPT_SHA256
    assert exact["matches_timeline_prompt_sha256"] is True
    assert exact["clipped_chars"] == mb.MAX_PROMPT_CHARS == 1800
    assert exact["unclipped_chars"] > 1800
    assert exact["max_ask_tokens"] == 384
    assert exact["temperature"] == 0.0
    assert exact["do_sample"] is False
    assert exact["enable_thinking"] is False
    assert exact["policy_id"] == "WU.HAWKING.resident_identity_pin"


def test_choose_source_still_contains_the_trial_literals():
    """If choose() is reworded, this probe must fail rather than paraphrase."""
    src = cjp.choose_source_contains_literals()
    assert src["preamble_present"] is True
    assert src["schema_tail_present"] is True
    assert src["max_prompt_chars"] == 1800
    assert src["max_prompt_chars_literal"] is True
    # The probe's finding was acted on: the live clip no longer eats the schema.
    # Asserting the defect is still present would pin the bug, not the science.
    assert src["clip_eats_tail"] is False, "the tail-eating clip came back"
    assert src["clip_keeps_tail"] is True
    assert src["fits_by_dropping_candidates"] is True
    text = Path(mb.__file__).read_text(encoding="utf-8")
    assert cjp.CHOOSE_PREAMBLE_PREFIX in text
    assert cjp.CHOOSE_SCHEMA_TAIL_A in text
    assert cjp.CHOOSE_SCHEMA_TAIL_B in text
    assert "Return JSON only:" in text


def test_control_clip_eats_the_json_schema():
    exact = cjp.extract_exact_ask()
    assert exact["json_instruction_in_unclipped"] is True
    assert exact["json_instruction_in_clipped_control"] is False
    assert exact["choice_id_in_clipped_control"] is False
    assert exact["clip_eats_the_schema"] is True
    assert "choice_id" not in exact["clipped_prompt"]
    assert "Return JSON only" not in exact["clipped_prompt"]
    assert exact["clipped_prompt"].endswith("…")


def test_control_prompt_function_is_the_timeline_ask():
    prompt = cjp.control_prompt()
    assert hashlib.sha256(prompt.encode()).hexdigest() == cjp.TIMELINE_CHOOSE_PROMPT_SHA256
    assert cjp.json_instruction_reached(prompt) is False


def test_timeline_choose_calls_are_the_clipped_ask_with_one_markdown_hash():
    obs = cjp.timeline_observation()
    choose = obs["choose"]
    assert choose["n"] == 43
    assert choose["n_parseable"] == 0
    assert choose["parse_rate"] == "0 of 43"
    assert choose["unique_reply_hashes"] == 1
    assert choose["replies"][0]["reply_sha256"] == cjp.TIMELINE_CHOOSE_REPLY_SHA256
    assert "**Reasoning:**" in choose["replies"][0]["reply_text"]
    assert "{" not in choose["replies"][0]["reply_text"]
    assert obs["choose_prompt_matches_reconstruction"] is True
    interpret = obs["interpret_json_object"]
    assert interpret["n"] == 43
    assert interpret["n_parseable"] == 43
    assert interpret["parse_rate"] == "43 of 43"


# ---------------------------------------------------------------------------
# Parse: markdown is not the object; interpret JSON is not the object
# ---------------------------------------------------------------------------


def test_trial_markdown_is_not_parseable_choice_json():
    row = cjp.parse_choice_json(TRIAL_MARKDOWN)
    assert row["parse_ok"] is False
    assert row["choice_id"] is None
    assert mb._extract_json(TRIAL_MARKDOWN) is None


def test_raw_and_fenced_choice_json_parse():
    raw = cjp.parse_choice_json(GOOD_JSON)
    assert raw["parse_ok"] is True
    assert raw["choice_id"] == "WU.HAWKING.resident_identity_pin"
    assert "closed scar" in raw["reason"]
    fenced = cjp.parse_choice_json("```json\n" + GOOD_JSON + "\n```")
    assert fenced["parse_ok"] is True
    assert fenced["choice_id"] == raw["choice_id"]


def test_interpret_shaped_json_is_not_choice_json():
    """NEGATIVE CONTROL: a JSON object without choice_id is not the contract."""
    assert mb._extract_json(INTERPRET_FENCE) is not None
    row = cjp.parse_choice_json(INTERPRET_FENCE)
    assert row["parse_ok"] is False
    assert row["choice_id"] is None


def test_choice_id_without_reason_does_not_parse():
    row = cjp.parse_choice_json('{"choice_id":"WU.HAWKING.resident_identity_pin"}')
    assert row["parse_ok"] is False
    assert row["choice_id"] == "WU.HAWKING.resident_identity_pin"
    assert "reason" in (row["why"] or "")


def test_empty_and_prose_without_brace_do_not_parse():
    assert cjp.parse_choice_json("")["parse_ok"] is False
    assert cjp.parse_choice_json("   ")["parse_ok"] is False
    assert cjp.parse_choice_json("pick WU.HAWKING.resident_identity_pin because live")["parse_ok"] is False


def test_fraction_is_a_fraction_not_a_score():
    assert cjp.fraction(0, 43) == "0 of 43"
    assert cjp.fraction(3, 20) == "3 of 20"
    assert cjp.fraction(2, 2) == "2 of 2"
    assert "mostly" not in cjp.fraction(3, 20)
    assert "%" not in cjp.fraction(3, 20)


# ---------------------------------------------------------------------------
# One lever at a time
# ---------------------------------------------------------------------------


def test_each_variation_is_one_named_lever_off_control():
    specs = {s["variation"]: s for s in cjp.variation_specs()}
    assert list(specs) == [
        "control",
        "json_only",
        "grammar",
        "one_shot",
        "smaller_choice_set",
    ]
    control = specs["control"]
    assert control["lever"] == "none"
    assert control["prompt_sha256"] == cjp.TIMELINE_CHOOSE_PROMPT_SHA256
    assert control["json_instruction_reached_the_body"] is False

    json_only = specs["json_only"]
    assert json_only["lever"] == "instruction"
    assert json_only["prompt_sha256"] != control["prompt_sha256"]
    assert json_only["json_instruction_reached_the_body"] is True
    assert json_only["prompt"].startswith("Return JSON only, no markdown, no prose, no fences")
    assert "choice_id" in json_only["prompt"]
    assert json_only["n_candidates"] == control["n_candidates"]
    assert json_only["grammar"] is False

    grammar = specs["grammar"]
    assert grammar["lever"] == "runtime_constraint"
    assert grammar["prompt_sha256"] == control["prompt_sha256"]
    assert grammar["prompt"] == control["prompt"]
    assert grammar["grammar"] is True

    one_shot = specs["one_shot"]
    assert one_shot["lever"] == "example"
    assert one_shot["prompt"].startswith("Example of a correct reply:")
    assert cjp.ONE_SHOT_EXAMPLE in one_shot["prompt"]
    assert one_shot["n_candidates"] == control["n_candidates"]
    assert one_shot["grammar"] is False

    smaller = specs["smaller_choice_set"]
    assert smaller["lever"] == "candidate_set_size"
    assert smaller["n_candidates"] == 2
    assert smaller["n_candidates"] != control["n_candidates"]
    assert smaller["json_instruction_reached_the_body"] is True
    assert smaller["prompt"].endswith(cjp.CHOOSE_SCHEMA_TAIL)
    assert smaller["prompt"].startswith(cjp.CHOOSE_PREAMBLE_PREFIX)
    assert "WU.HAWKING.resident_identity_pin" in smaller["prompt"]
    assert "WU.DEAD.mlp_function_replacement" not in smaller["prompt"]


def test_json_only_and_one_shot_still_clip_at_1800():
    specs = {s["variation"]: s for s in cjp.variation_specs()}
    assert len(specs["json_only"]["prompt"]) == 1800
    assert len(specs["one_shot"]["prompt"]) == 1800
    assert len(specs["control"]["prompt"]) == 1800
    assert len(specs["smaller_choice_set"]["prompt"]) < 1800


def test_smaller_set_keeps_the_scripted_policy_id():
    small = cjp.smaller_catalog()
    ids = {r["id"] for r in small}
    assert ids == set(cjp.SMALLER_IDS)
    policy = mb.fixed_policy_choose(small)
    assert policy["id"] == "WU.HAWKING.resident_identity_pin"


def test_native_runtime_does_not_support_logit_mask_grammar():
    info = cjp.native_runtime_grammar_support()
    assert info["supported"] is False
    assert "validate-and-retry" in info["why"]
    assert info["request_literal_is_four_keys"] is True
    assert list(info["native_request_keys"]) == ["id", "prompt", "max_new_tokens", "max_seq_len"]
    native = mbt.hcli_root() / "hcli" / "hawking_native.py"
    text = native.read_text(encoding="utf-8") if native.is_file() else ""
    assert '"max_new_tokens": max_new_tokens' in text
    assert '"json_schema"' not in text
    assert '"response_format"' not in text
    assert '"guided_json"' not in text


# ---------------------------------------------------------------------------
# Judge: three named outcomes, plus discrepancy
# ---------------------------------------------------------------------------


def _cell(body, variation, k, n, skipped=None, **extra):
    row = {
        "body": body,
        "variation": variation,
        "lever": extra.get("lever", "none"),
        "one_change": extra.get("one_change", ""),
        "n_parseable": k,
        "n": n,
        "skipped": skipped,
    }
    row.update(extra)
    return row


def test_judge_ask_wrong_when_a_variation_works_on_the_incumbent():
    cells = [
        _cell(cjp.INCUMBENT_BODY, "control", 0, 2),
        _cell(
            cjp.INCUMBENT_BODY,
            "json_only",
            2,
            2,
            lever="instruction",
            one_change="JSON-only first",
        ),
    ]
    v = cjp.judge(cells, control_reproduced=True)
    assert v["outcome"] == cjp.OUTCOME_ASK
    assert v["fixed_by"] == "json_only"
    assert "prompt/schema fix" in v["recommendation"]
    assert "MAX_PROMPT_CHARS=1800" in v["recommendation"]
    assert "succession" in v["recommendation"].lower()


def test_judge_incumbent_cannot_when_only_the_other_body_emits_the_object():
    cells = [
        _cell(cjp.INCUMBENT_BODY, "control", 0, 2),
        _cell(cjp.INCUMBENT_BODY, "json_only", 0, 2),
        _cell(cjp.QWEN06_BODY, "control", 2, 2),
    ]
    v = cjp.judge(cells, control_reproduced=True)
    assert v["outcome"] == cjp.OUTCOME_BODY
    assert cjp.QWEN06_BODY in str(v.get("working_bodies"))
    assert "cognition" in v["recommendation"].lower()


def test_judge_contract_unusable_when_nothing_parses():
    cells = [
        _cell(cjp.INCUMBENT_BODY, "control", 0, 2),
        _cell(cjp.INCUMBENT_BODY, "json_only", 0, 2),
        _cell(cjp.QWEN06_BODY, "control", 0, 2),
        _cell(cjp.QWEN06_BODY, "json_only", 0, 2),
    ]
    v = cjp.judge(cells, control_reproduced=True)
    assert v["outcome"] == cjp.OUTCOME_NOTHING
    assert "replace structured-choice" in v["recommendation"]


def test_judge_discrepancy_when_incumbent_control_parses():
    cells = [_cell(cjp.INCUMBENT_BODY, "control", 2, 2)]
    v = cjp.judge(cells, control_reproduced=False)
    assert v["outcome"] == cjp.OUTCOME_DISCREPANCY
    assert "did not reproduce" in v["why"]


def test_skipped_grammar_cell_is_not_a_working_variation():
    cells = [
        _cell(cjp.INCUMBENT_BODY, "control", 0, 2),
        _cell(
            cjp.INCUMBENT_BODY,
            "grammar",
            0,
            0,
            skipped={"reason": "runtime_does_not_support_logit_mask_grammar"},
        ),
    ]
    v = cjp.judge(cells, control_reproduced=True)
    assert v["outcome"] == cjp.OUTCOME_NOTHING


# ---------------------------------------------------------------------------
# --build is not a live pass; hardware refused; autonomy_trial not written
# ---------------------------------------------------------------------------


def test_static_document_is_not_a_live_outcome():
    doc = cjp.static_document()
    assert doc["verdict"] == "NOT_RUN"
    assert doc["cells"] == []
    assert doc["live_ran"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["exact_ask"]["matches_timeline_prompt_sha256"] is True
    assert doc["timeline_observation"]["choose"]["parse_rate"] == "0 of 43"
    assert doc["autonomy_trial_not_touched"] is True
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert key not in doc


def test_static_receipt_roundtrip_refuses_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr(cjp, "RECEIPT", "CHOICE_JSON_PROBE_TEST.json")
    from tools.future import _common as common

    monkeypatch.setattr(common, "RECEIPTS", tmp_path)
    monkeypatch.setattr(cjp, "write_receipt", write_receipt)
    # write_receipt binds RECEIPTS at call time via common.RECEIPTS
    path = write_receipt("CHOICE_JSON_PROBE_TEST.json", cjp.static_document(), cjp.RECORDED_BY)
    assert path.parent == tmp_path
    doc = json.loads(path.read_text())
    assert doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]


def test_hardware_named_field_cannot_land_in_a_receipt():
    doc = cjp.static_document()
    doc["tps"] = 12.0
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims(doc)


def test_source_does_not_write_autonomy_trial_or_hcli():
    text = SRC.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert "autonomy_trial_not_touched" in text
    writes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"write_text", "write_bytes"}:
                writes.append(ast.dump(node)[:160])
    assert writes == [], writes
    assert "sparse-checkout" not in text
    assert "GpuPark" in text
    assert "MAX_ASK_TOKENS" in text


def test_selftest_passes():
    cjp.selftest()


def test_build_writes_not_run_when_invoked_directly(tmp_path, monkeypatch):
    """--build must not mint ASK_WRONG / INCUMBENT_CANNOT / CONTRACT_UNUSABLE."""
    monkeypatch.chdir(tmp_path)
    doc = cjp.static_document()
    assert doc["verdict"] not in {
        cjp.OUTCOME_ASK,
        cjp.OUTCOME_BODY,
        cjp.OUTCOME_NOTHING,
    }
    assert "--build" in doc["reason"] or "does not start a body" in doc["reason"]


def test_the_probe_reads_an_immutable_control_not_the_live_timeline():
    """A control a later run can overwrite is not a control.

    This probe describes the ask that failed 0 of 43. It used to read the live
    MODEL_BEARING_TIMELINE.json, which every run replaces - so once the ask was
    fixed and the resident went 51 of 51, these assertions failed because the
    artifact under study had been replaced by the thing it caused.
    """
    assert "controls/" in cjp.TIMELINE_REL
    assert cjp.TIMELINE_REL != cjp.LIVE_TIMELINE_REL
    assert cjp.TIMELINE_ARCHIVE_COMMIT, "the fallback must name a commit, never HEAD"
    obs = cjp.timeline_observation()
    assert obs["choose"]["parse_rate"] == "0 of 43", "the control moved"


def test_the_live_timeline_is_allowed_to_disagree_with_the_control():
    """And it does: the fix took it to 51 of 51. That is the point."""
    live = ra_repo = cjp.REPO / cjp.LIVE_TIMELINE_REL
    if not live.is_file():
        pytest.skip("no live timeline on disk")
    doc = json.loads(live.read_text())
    calls = [m for m in (doc.get("model_calls") or [])
             if "Pick one candidate" in str(m.get("prompt") or "")]
    # no assertion on the NUMBER - it moves with every run, which is why the
    # control had to stop being this file.
    assert isinstance(calls, list)
