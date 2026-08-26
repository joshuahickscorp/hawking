"""A schema rejection that names only what is ABSENT teaches nothing to a backend
that got the shape right and drifted one key.

Measured against the sealed 27B resident on 2026-08-26: it wrote "consequential"
on obligations[0] and "consequent" on [1], [2] and [3]. The rejection said
"missing required property 'consequential'" six times and the model never
converged, because nothing in the message pointed at the key it had written.
"""
import pytest

from hcli.backends import _edit_distance, _near_miss_key, validate_against_schema

SCHEMA = {"type": "object",
          "properties": {"consequential": {"type": "boolean"},
                         "id": {"type": "string"},
                         "angles": {"type": "array"}},
          "required": ["consequential", "id"]}


def test_ANTI_VACUITY_a_valid_object_is_not_rejected():
    assert validate_against_schema(
        {"consequential": True, "id": "x"}, SCHEMA) is None


def test_THE_MEASURED_DRIFT_IS_NAMED():
    err = validate_against_schema({"id": "x", "consequent": True}, SCHEMA)
    assert "consequent" in err and "consequential" in err


def test_A_PLAIN_OMISSION_IS_NOT_DECORATED_WITH_A_GUESS():
    """Nothing resembling the key is present, so the message must not invent one."""
    err = validate_against_schema({"id": "x"}, SCHEMA)
    assert "you wrote" not in err


def test_AN_UNRELATED_EXTRA_KEY_IS_NOT_CALLED_A_MISSPELLING():
    err = validate_against_schema(
        {"id": "x", "completely_unrelated_field": 1}, SCHEMA)
    assert "you wrote" not in err, err


def test_A_LEGITIMATE_SIBLING_PROPERTY_IS_NEVER_THE_NEAR_MISS():
    """`angles` is a real property of this schema. A required key going missing
    does not make a sibling that IS in the schema a misspelling of it."""
    assert _near_miss_key("consequential", {"angles": []}, SCHEMA) is None


def test_SHORT_KEYS_CANNOT_COLLIDE_WITH_EACH_OTHER():
    """The threshold scales with name length, so `id` does not match `is`."""
    sch = {"type": "object", "properties": {"id": {}}, "required": ["id"]}
    assert _near_miss_key("id", {"ix": 1}, sch) is None


@pytest.mark.parametrize("a,b,d", [("", "", 0), ("a", "", 1), ("abc", "abc", 0),
                                   ("consequential", "consequent", 3),
                                   ("kitten", "sitting", 3)])
def test_the_distance_is_the_real_one(a, b, d):
    assert _edit_distance(a, b) == d


# --------------------------------------------------------------------------
# Repair, and the limits on it. A repair that nobody can see is worse than the
# drift it fixes.
# --------------------------------------------------------------------------
import json

from hcli.backends import (StructuredOutputContract, SchemaViolation,
                           repair_near_miss_keys, schema_instruction)

PLAN = {"type": "object", "additionalProperties": False,
        "required": ["obligations"],
        "properties": {"obligations": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["id", "consequential"],
                      "properties": {"id": {"type": "string"},
                                     "consequential": {"type": "boolean"}}}}}}


def contract():
    return StructuredOutputContract(schema=PLAN,
                                    instruction=schema_instruction(PLAN))


def test_ANTI_VACUITY_a_clean_reply_needs_no_repair():
    c = contract()
    c.validate(json.dumps({"obligations": [{"id": "a", "consequential": True}]}))
    assert c.repairs == []


def test_THE_MEASURED_DRIFT_IS_REPAIRED_AND_RECORDED():
    """The exact reply the sealed resident produced: right on obligations[0],
    'consequent' on the rest."""
    c = contract()
    out = c.validate(json.dumps({"obligations": [
        {"id": "a", "consequential": True},
        {"id": "b", "consequent": True}]}))
    assert out["obligations"][1]["consequential"] is True
    assert c.repairs and "'consequent' -> 'consequential'" in c.repairs[0]


def test_A_WHITESPACE_PADDED_KEY_IS_STRIPPED_AND_RECORDED():
    """The drift the retry produced once the first was named: ' angles'."""
    c = contract()
    out = c.validate(json.dumps({"obligations": [{"id": "a", " consequential ": True}]}))
    assert out["obligations"][0]["consequential"] is True
    assert any("whitespace" in r for r in c.repairs)


def test_A_REPAIR_THAT_WOULD_ONLY_RELOCATE_A_TYPE_ERROR_IS_REFUSED():
    """Renaming a string onto a boolean property must not be reported as fixed.
    The re-validation is what refuses it -- there is no separate type guard on the
    rename, because a mutation proved that guard changed no outcome."""
    c = contract()
    with pytest.raises(SchemaViolation):
        c.validate(json.dumps({"obligations": [{"id": "a", "consequent": "yes"}]}))
    assert c.repairs == []


def test_A_MISSING_KEY_WITH_NO_CANDIDATE_IS_STILL_A_VIOLATION():
    c = contract()
    with pytest.raises(SchemaViolation):
        c.validate(json.dumps({"obligations": [{"id": "a"}]}))


def test_REPAIR_REACHES_INSIDE_ARRAYS_not_just_the_top_level():
    """Every real drift here was on an array element, never the root object."""
    log = []
    repair_near_miss_keys({"obligations": [{"id": "a", "consequent": True}]}, PLAN, log)
    assert log and "$.obligations[0]" in log[0]


def test_A_REPAIRED_OBJECT_THAT_IS_STILL_INVALID_IS_STILL_REFUSED():
    """A repair fixes ONE key. If the object is invalid for another reason the
    re-validation is what catches it -- and without that step an invalid object
    would be returned as if it had validated.

    The first version of this file did not have this case, and the mutation that
    deleted the re-validation passed all 18 tests.
    """
    c = contract()
    with pytest.raises(SchemaViolation):
        # 'consequent' is repairable; 'id' is simply absent and nothing resembles it.
        c.validate(json.dumps({"obligations": [{"consequent": True}]}))


def test_A_REPAIR_IS_NEVER_SILENT_IN_THE_RECEIPT():
    """It must reach result.raw, which is what the envelope records -- not only
    the contract object, which nothing downstream reads.

    The first version asserted c.repairs and so passed with the receipt line
    deleted.
    """
    from hcli.backends import CompletionResult
    c = contract()
    reply = json.dumps({"obligations": [{"id": "b", "consequent": True}]})
    out = c.enforce(lambda body, timeout=None: CompletionResult(raw={}, text=reply),
                    {"messages": [{"role": "user", "content": "go"}]})
    assert out.raw["_structured_repairs"], "the repair never reached the receipt"
    assert "structured_output_key_repair" in out.degraded


def test_the_call_timeout_is_configurable_and_defaults_to_180():
    """A local resident that reloads the model per call is exactly the backend the
    hardcoded 180 s was wrong for: a real mission died at 722 s having exceeded it
    on one generate."""
    import os
    from hcli.delegate import _delegate_timeout
    old = os.environ.pop("HCLI_DELEGATE_TIMEOUT_S", None)
    try:
        assert _delegate_timeout() == 180.0
        os.environ["HCLI_DELEGATE_TIMEOUT_S"] = "600"
        assert _delegate_timeout() == 600.0
        os.environ["HCLI_DELEGATE_TIMEOUT_S"] = "not a number"
        assert _delegate_timeout() == 180.0, "a bad value must fall back, not raise"
        os.environ["HCLI_DELEGATE_TIMEOUT_S"] = "0"
        assert _delegate_timeout() >= 1.0, "a zero timeout would fail every call"
    finally:
        os.environ.pop("HCLI_DELEGATE_TIMEOUT_S", None)
        if old is not None:
            os.environ["HCLI_DELEGATE_TIMEOUT_S"] = old


# --------------------------------------------------------------------------
# A syntax error upstream must not be reported as a shape error downstream.
#
# Measured on the sealed 27B resident: it emitted an unescaped quote inside a
# string, the outermost object failed to decode, the scan salvaged the SECOND
# obligation object, and the rejection read "$: missing required property
# 'obligations'". Six retries chased a phantom.
# --------------------------------------------------------------------------

MALFORMED = '''{
  "obligations": [
    {"id": "a", "statement": "s", "agent_role": "test",
     "angles": ["find \\"$ROOT/receipts/head" -type f"], "consequential": true},
    {"id": "b", "statement": "t", "agent_role": "test",
     "angles": ["ok"], "consequential": true}
  ]
}'''


def test_the_fixture_really_is_malformed_and_really_does_salvage():
    """ANTI-VACUITY: if this text parsed, or salvaged nothing, the tests below
    would be asserting against a situation that never happens."""
    with pytest.raises(Exception):
        json.loads(MALFORMED)
    from hcli.backends import extract_json_object
    got = extract_json_object(MALFORMED)
    assert isinstance(got, dict) and got.get("id") == "b", got


def test_A_SALVAGED_FRAGMENT_IS_REPORTED_AS_ONE():
    from hcli.backends import extract_json_object
    diag = []
    extract_json_object(MALFORMED, diag)
    assert diag and "NOT valid JSON" in diag[0] and "INNER object" in diag[0]


def test_THE_REJECTION_NAMES_THE_SYNTAX_ERROR_NOT_ONLY_THE_SHAPE():
    c = contract()
    with pytest.raises(SchemaViolation) as e:
        c.validate(MALFORMED)
    reason = str(e.value)
    assert "NOT valid JSON" in reason, reason
    assert "obligations" in reason, "the shape error is still reported, after the cause"


def test_A_WELL_FORMED_REPLY_CARRIES_NO_SALVAGE_NOTE():
    from hcli.backends import extract_json_object
    diag = []
    extract_json_object(json.dumps({"obligations": []}), diag)
    assert diag == []


def test_PROSE_WRAPPED_JSON_IS_STILL_ACCEPTED_WITHOUT_A_NOTE():
    """The salvage exists for replies that wrap the object in text. That path must
    not start reporting a syntax error it does not have."""
    from hcli.backends import extract_json_object
    diag = []
    got = extract_json_object(
        'Here is the plan:\n' + json.dumps({"obligations": []}), diag)
    assert got == {"obligations": []} and diag == []
