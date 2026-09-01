"""A malformed model reply is a result, never an exception.

The load-bearing guards: admit() never raises, always returns RESULT_KEYS,
fills selected_work as a list so a later loop cannot slice a dict, and names
the exact missing fields on a partial object so the caller can re-ask narrowly.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from tools.future import resident_output_contract as roc
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


SCHEMA = roc.SOVEREIGN_REPLY_SCHEMA
VALID = roc.VALID_SOVEREIGN_REPLY
VALID_JSON = json.dumps(VALID, ensure_ascii=False)


def _admit(raw):
    return roc.admit(raw, SCHEMA)


def _assert_shape(result: dict) -> None:
    assert tuple(result) == roc.RESULT_KEYS
    assert tuple(result["parse"]) == roc.PARSE_KEYS
    assert tuple(result["reask"]) == roc.REASK_KEYS
    assert isinstance(result["ok"], bool)
    assert isinstance(result["missing"], list)
    assert isinstance(result["extra"], list)
    assert isinstance(result["coerced"], list)
    assert isinstance(result["errors"], list)
    assert result["reask"]["fields"] == result["missing"]
    assert result["reask"]["needed"] == bool(result["missing"])
    assert isinstance(result["value"], dict)
    assert isinstance(result["value"]["selected_work"], list)
    assert isinstance(result["value"]["live_hypotheses"], list)
    assert "belief_update" in result["value"]
    assert "escalation_needed" in result["value"]


# ---------------------------------------------------------------------------
# One test per malformed shape the contract names.
# ---------------------------------------------------------------------------


def test_valid_json():
    result = _admit(VALID_JSON)
    _assert_shape(result)
    assert result["ok"] is True
    assert result["missing"] == []
    assert result["parse"]["kind"] == "json_object"
    assert result["parse"]["ok"] is True
    assert result["value"]["belief_update"] == VALID["belief_update"]
    assert result["value"]["selected_work"][0]["params"]["layer"] == 0


def test_malformed_json():
    result = _admit('{"belief_update": "x", this is not json')
    _assert_shape(result)
    assert result["ok"] is False
    assert result["parse"]["kind"] in ("malformed", "truncated")
    assert "selected_work" in result["value"]
    assert isinstance(result["value"]["selected_work"], list)


def test_truncated_json():
    result = _admit(VALID_JSON[: max(40, len(VALID_JSON) // 2)])
    _assert_shape(result)
    assert result["ok"] is False or result["parse"]["kind"] == "truncated"
    assert result["parse"]["kind"] in ("truncated", "malformed", "json_object")
    assert isinstance(result["value"]["selected_work"], list)


def test_prose_tail_after_valid_json():
    result = _admit(VALID_JSON + "\nHope this helps! — a degenerate prose tail")
    _assert_shape(result)
    assert result["ok"] is True
    assert result["parse"]["kind"] == "prose_tail"
    assert result["parse"]["recovered"] is True
    assert result["value"]["belief_update"] == VALID["belief_update"]


def test_valid_prefix_with_corrupt_tail():
    result = _admit(VALID_JSON + "{not closed, ::: ###")
    _assert_shape(result)
    assert result["ok"] is True
    assert result["parse"]["kind"] == "corrupt_tail"
    assert result["parse"]["recovered"] is True
    assert result["missing"] == []


def test_dict_where_list_belongs():
    raw = {**VALID, "selected_work": VALID["selected_work"][0]}
    result = _admit(raw)
    _assert_shape(result)
    assert isinstance(result["value"]["selected_work"], list)
    assert len(result["value"]["selected_work"]) == 1
    assert result["value"]["selected_work"][0]["type"] == "PERTURB"
    assert "selected_work" in result["coerced"]
    assert result["ok"] is True


def test_list_where_dict_belongs():
    raw = {
        **VALID,
        "selected_work": [{
            **VALID["selected_work"][0],
            "params": [VALID["selected_work"][0]["params"]],
        }],
    }
    result = _admit(raw)
    _assert_shape(result)
    params = result["value"]["selected_work"][0]["params"]
    assert isinstance(params, dict)
    assert params["tensor"] == "gate"
    assert "selected_work.params" in result["coerced"]
    assert result["ok"] is True


def test_missing_required_fields():
    result = _admit({"belief_update": "only this key"})
    _assert_shape(result)
    assert result["ok"] is False
    assert "live_hypotheses" in result["missing"]
    assert "selected_work" in result["missing"]
    assert "escalation_needed" in result["missing"]
    assert "belief_update" not in result["missing"]
    assert result["value"]["belief_update"] == "only this key"
    assert result["value"]["selected_work"] == []


def test_extra_fields():
    result = _admit({**VALID, "commentary": "drop me"})
    _assert_shape(result)
    assert result["ok"] is True
    assert "commentary" in result["extra"]
    assert "commentary" not in result["value"]


def test_null_values():
    result = _admit({
        "belief_update": None,
        "live_hypotheses": None,
        "selected_work": None,
        "escalation_needed": None,
    })
    _assert_shape(result)
    assert result["ok"] is False
    assert "belief_update" in result["missing"]
    assert "live_hypotheses" in result["missing"]
    assert "selected_work" in result["missing"]
    assert "escalation_needed" in result["missing"]
    assert result["value"]["selected_work"] == []
    assert result["value"]["escalation_needed"] is False


def test_unicode():
    text = "下限は 2.508 BPW — 日本語 🧪 snowman ☃"
    result = _admit({**VALID, "belief_update": text})
    _assert_shape(result)
    assert result["ok"] is True
    assert result["value"]["belief_update"] == text


def test_deeply_nested_junk():
    nested: dict = {"leaf": "junk"}
    for _ in range(24):
        nested = {"n": nested, "xs": [nested]}
    result = _admit({**VALID, "junk": nested})
    _assert_shape(result)
    assert result["ok"] is True
    assert "junk" in result["extra"]
    assert "junk" not in result["value"]


def test_empty_string():
    result = _admit("")
    _assert_shape(result)
    assert result["ok"] is False
    assert result["parse"]["kind"] == "empty"
    assert set(result["missing"]) >= {"belief_update", "live_hypotheses",
                                      "selected_work", "escalation_needed"}


def test_none():
    result = _admit(None)
    _assert_shape(result)
    assert result["ok"] is False
    assert result["parse"]["kind"] == "none"
    assert result["parse"]["source_type"] == "NoneType"
    assert isinstance(result["value"]["selected_work"], list)


# ---------------------------------------------------------------------------
# Narrow re-ask, the three live crashes, property probe, receipt.
# ---------------------------------------------------------------------------


def test_narrow_reask_names_exact_missing_fields():
    partial = {
        "belief_update": "a real sentence",
        "live_hypotheses": [{"id": "H.partial", "claim": "has no falsifier"}],
        "escalation_needed": False,
    }
    result = _admit(partial)
    _assert_shape(result)
    assert result["ok"] is False
    assert result["missing"] == [
        "live_hypotheses.cheapest_falsifier",
        "selected_work",
    ]
    assert result["reask"]["needed"] is True
    assert result["reask"]["fields"] == result["missing"]
    assert "live_hypotheses.cheapest_falsifier" in result["reask"]["prompt_fragment"]
    assert "selected_work" in result["reask"]["prompt_fragment"]
    assert "belief_update" not in result["missing"]
    assert "escalation_needed" not in result["missing"]
    assert result["value"]["live_hypotheses"][0]["id"] == "H.partial"
    assert result["value"]["live_hypotheses"][0]["cheapest_falsifier"] == ""


def test_property_mutated_payloads_never_raise_and_keep_keyset():
    rng = random.Random(0)
    n = 220
    for _ in range(n):
        payload = roc.mutate_payload(rng, VALID)
        try:
            result = roc.admit(payload, SCHEMA)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"admit raised {type(exc).__name__}: {exc}") from exc
        _assert_shape(result)
        # A later sovereign loop does this; it must not KeyError on any payload.
        for work in result["value"]["selected_work"]:
            assert isinstance(work, dict)
            params = work["params"]
            assert isinstance(params, dict)


def test_the_three_live_crash_shapes_keep_every_key():
    """Missing key, dict-not-list, parse-failure: the three live crashes."""
    shapes = [
        {"belief_update": "x"},
        {**VALID, "selected_work": VALID["selected_work"][0]},
        None,
    ]
    for raw in shapes:
        result = _admit(raw)
        _assert_shape(result)
        assert set(result) == set(roc.RESULT_KEYS)


def test_fenced_json():
    result = _admit("Here you go:\n```json\n" + VALID_JSON + "\n```\n")
    _assert_shape(result)
    assert result["ok"] is True
    assert result["parse"]["kind"] == "fenced"
    assert result["value"]["belief_update"] == VALID["belief_update"]


def test_whitespace_only():
    result = _admit(" \n\t ")
    _assert_shape(result)
    assert result["ok"] is False
    assert result["parse"]["kind"] == "empty"


def test_json_array_at_top():
    result = _admit([VALID])
    _assert_shape(result)
    assert isinstance(result["value"], dict)
    assert result["value"]["belief_update"] == VALID["belief_update"]
    assert "<root>" in result["coerced"] or result["ok"] is True


def test_bytes_utf8():
    result = _admit(VALID_JSON.encode("utf-8"))
    _assert_shape(result)
    assert result["ok"] is True
    assert result["value"]["belief_update"] == VALID["belief_update"]


def test_parse_failure_does_not_omit_fields():
    result = _admit("not json at all, no braces")
    _assert_shape(result)
    assert result["ok"] is False
    assert result["parse"]["kind"] == "non_json"
    for key in roc.RESULT_KEYS:
        assert key in result
    assert result["value"]["selected_work"] == []
    assert result["value"]["live_hypotheses"] == []


def test_bool_false_is_not_missing():
    result = _admit({**VALID, "escalation_needed": False})
    _assert_shape(result)
    assert result["ok"] is True
    assert result["value"]["escalation_needed"] is False
    assert "escalation_needed" not in result["missing"]


def test_empty_required_string_is_missing():
    result = _admit({**VALID, "belief_update": "   "})
    _assert_shape(result)
    assert result["ok"] is False
    assert "belief_update" in result["missing"]


def test_compile_schema_refuses_a_missing_schema():
    with pytest.raises(roc.ContractRefused, match="schema is missing"):
        roc.compile_schema(None)
    with pytest.raises(roc.ContractRefused, match="schema is missing"):
        roc.compile_schema({})
    with pytest.raises(roc.ContractRefused, match="schema is missing"):
        roc.compile_schema("")
    with pytest.raises(roc.ContractRefused, match="refusing to infer"):
        roc.compile_schema({"belief_update": "not a schema"})


def test_admit_does_not_raise_without_a_schema():
    result = roc.admit(VALID_JSON, None)
    assert tuple(result) == roc.RESULT_KEYS
    assert result["ok"] is False
    assert any("schema is missing" in e for e in result["errors"])


def test_same_shape_on_every_named_payload():
    for shape in roc.named_shapes():
        result = roc.admit(shape["raw"], SCHEMA)
        _assert_shape(result)


def test_schema_as_json_string():
    result = roc.admit(VALID, json.dumps(SCHEMA))
    _assert_shape(result)
    assert result["ok"] is True


def test_numeric_strings_in_params_coerce():
    raw = json.loads(VALID_JSON)
    raw["selected_work"][0]["params"]["layer"] = "3"
    raw["selected_work"][0]["params"]["fraction"] = "0.25"
    result = _admit(raw)
    _assert_shape(result)
    assert result["ok"] is True
    assert result["value"]["selected_work"][0]["params"]["layer"] == 3
    assert result["value"]["selected_work"][0]["params"]["fraction"] == pytest.approx(0.25)
    assert "selected_work.params.layer" in result["coerced"]
    assert "selected_work.params.fraction" in result["coerced"]


def test_build_writes_a_parsable_receipt():
    doc = roc.build()
    path = roc.main(["--build"])
    # main prints the path; write via build() + write_receipt by invoking --build
    out = RECEIPTS / roc.RECEIPT_NAME
    assert out.is_file()
    parsed = json.loads(out.read_text())
    assert parsed["schema"] == roc.SCHEMA_ID
    assert parsed["question"].startswith("Can any malformed")
    assert parsed["named_always_same_keyset"] is True
    assert parsed["named_selected_work_always_list"] is True
    assert parsed["n_named_shapes"] >= 14
    assert parsed["property_probe"]["n"] >= 200
    assert parsed["property_probe"]["n_raised"] == 0
    assert parsed["property_probe"]["always_same_keyset"] is True
    assert parsed["narrow_reask_example"]["missing"] == [
        "live_hypotheses.cheapest_falsifier",
        "selected_work",
    ]
    assert parsed["gpu_authority"] is False
    assert parsed["evidence_class"] == "STATIC_ONLY"
    assert parsed["seal_sha256"]
    _assert_no_hardware_claims(parsed)
    for field in HARDWARE_FIELDS:
        assert field not in parsed
    assert doc["n_named_shapes"] == parsed["n_named_shapes"]
    assert path == 0


def test_result_keys_are_frozen():
    assert roc.RESULT_KEYS == (
        "ok", "value", "missing", "extra", "coerced", "errors",
        "parse", "schema_id", "reask",
    )
    assert "n_accepted" not in roc.RESULT_KEYS
    # The live crash was omitting keys on one path. Pin the set.
    a = _admit(None)
    b = _admit(VALID)
    c = _admit("{")
    assert tuple(a) == tuple(b) == tuple(c) == roc.RESULT_KEYS


# --- G115: one doubled quote discarded a complete option tree -----------------

def _S():
    return {"id": "t.v1", "type": "object", "required": ["a"],
            "additionalProperties": False,
            "properties": {"a": {"type": "string", "nonempty": True}}}


def test_a_doubled_key_quote_is_repaired():
    """The resident produced a complete four-route tree and `""evidence_status":`
    made all 1662 characters unparseable."""
    r = roc.admit('{"a": "x", ""b": "y"}', _S())
    assert r["parse"]["recovered"] is True
    assert r["value"]["a"] == "x"


def test_the_repair_does_not_touch_json_that_already_parses():
    for good in ('{"a": "x"}', '{"a": ""}', '{"a": "he said \\"hi\\""}'):
        r = roc.admit(good, _S())
        assert r["parse"]["kind"] != "repaired_doubled_quote", good


def test_an_empty_string_VALUE_is_never_mistaken_for_a_stray_quote():
    """`{"a": ""}` is valid and means empty. The pattern requires the doubled
    quote to be followed by a key and a colon, so it cannot match here."""
    assert not roc._DOUBLED_KEY_QUOTE.search('{"a": ""}')
    assert not roc._DOUBLED_KEY_QUOTE.search('{"a": "", "b": 1}')
    assert roc._DOUBLED_KEY_QUOTE.search('{"a": 1, ""b": 2}')


def test_the_repair_is_last_resort_not_first_choice():
    """It runs only after strict parse, raw_decode and truncation salvage have
    all failed, so it can never pre-empt a correct reading."""
    import inspect
    src = inspect.getsource(roc._loads_value)
    assert src.index("_close_truncated") < src.index("_repair_doubled_key_quote")
    assert src.index("json.loads") < src.index("_repair_doubled_key_quote")


def test_an_unrepairable_reply_still_returns_the_uniform_shape():
    r = roc.admit('{"a": ""b": nonsense nonsense', _S())
    assert set(r) == set(roc.admit("{}", _S()))
    assert r["ok"] is False
