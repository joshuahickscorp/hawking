"""The constitution must be PRESENT, STABLE, and SMALL.

A stable prefix that drifts is worse than no prefix: it breaks reuse while still
costing tokens. A constitution nothing reads is the registration-is-not-capability
trap. Both are tested here, not assumed.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "future"))
TEXT = (ROOT / "hcli" / "CONSTITUTION.txt").read_text()


def test_constitution_is_actually_in_the_compiled_prompt():
    import hcli_ladder as H
    prompt = H.prompt_for([{"spec": "q3-g128-experts", "ebpw": 3.2831, "pass": True}])
    assert TEXT.rstrip() in prompt, "constitution is not reaching the resident"


def test_constitution_is_the_PREFIX_not_buried():
    import hcli_ladder as H
    prompt = H.prompt_for([{"spec": "q3-g128-experts", "ebpw": 3.2831, "pass": True}])
    assert prompt.startswith(TEXT.rstrip()), (
        "constitution must lead the prompt; anything before it shifts every "
        "downstream token and defeats prefix reuse"
    )


def test_constitution_is_byte_stable_across_calls():
    import hcli_ladder as H
    a = H.prompt_for([{"spec": "x", "ebpw": 1.0, "pass": True}])
    b = H.prompt_for([{"spec": "y", "ebpw": 2.0, "pass": False}], ["rejected thing"])
    n = len(TEXT.rstrip())
    assert hashlib.sha256(a[:n].encode()).hexdigest() == \
           hashlib.sha256(b[:n].encode()).hexdigest(), \
        "the prefix changed between calls; prefix reuse is defeated"


def test_constitution_stays_small_enough_to_be_a_prefix():
    # Budget is ~667 usable tokens. Measured at 129 with the resident tokenizer;
    # 4 chars/token is a conservative proxy that does not need the model loaded.
    approx = len(TEXT) / 4
    assert approx < 200, f"~{approx:.0f} tokens is too large for a per-call prefix"


def test_constitution_asserts_the_load_bearing_separations():
    low = TEXT.lower()
    for phrase in ("tools measure", "verifiers decide", "disk is authority",
                   "measured bound", "verify the metric", "change representation"):
        assert phrase in low, f"constitution lost its {phrase!r} clause"


def test_prompt_survives_a_frontier_with_no_passing_candidate():
    """min(..., default=None) then dereferenced. A search whose evidence is all
    failures is the NORMAL state early in a ladder, and it crashed the prompt."""
    import hcli_ladder as H
    p = H.prompt_for([{"spec": "y", "ebpw": 2.0, "pass": False}])
    assert "NOTHING above passes" in p
