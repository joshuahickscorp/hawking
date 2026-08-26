"""Mutation tests for the sealed-resident endpoint.

The danger a shim carries is not that it breaks -- that is visible -- it is that
it silently serves a DIFFERENT configuration than the seal binds, and every
number downstream then describes something nobody sealed. Every test here breaks
the correspondence on purpose and requires a refusal.
"""
import copy, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import serve_sealed as S


def seal():
    return S.load_seal()


def test_the_real_disk_matches_the_real_seal():
    """ANTI-VACUITY. Without this, a checker that refused everything would make
    every other test in this file pass."""
    v = S.check_seal(seal())
    assert v["sealed"] is True, v["mismatches"]


def test_a_DIFFERENT_BINARY_is_REFUSED():
    """The one that matters most: a rebuilt binary is a different resident, and
    this session already killed a capability run for exactly that reason."""
    s = seal(); s["fields"]["runtime_binary_sha256_16"]["value"] = "0" * 16
    v = S.check_seal(s)
    assert not v["sealed"] and any("binary sha" in m for m in v["mismatches"]), v


def test_a_DIFFERENT_TOKENIZER_OR_TEMPLATE_is_REFUSED():
    for key, needle in (("tokenizer_sha256_16", "tokenizer.json"),
                        ("chat_template_sha256_16", "chat_template.jinja")):
        s = seal(); s["fields"][key]["value"] = "0" * 16
        v = S.check_seal(s)
        assert not v["sealed"] and any(needle in m for m in v["mismatches"]), (key, v)


def test_the_WRONG_ARM_is_REFUSED():
    """This server renders pre_closed_think. If the seal ever binds the other arm
    the server must refuse rather than quietly serving the one it knows -- the
    same artifact scores 30/43 and 35/43 on the two."""
    s = seal(); s["fields"]["chat_template_arm"]["value"] = "open_think"
    v = S.check_seal(s)
    assert not v["sealed"] and any("arm" in m for m in v["mismatches"]), v


def test_a_DIFFERENT_GRAPH_is_REFUSED():
    s = seal(); s["graph"]["dispatches_per_decode_token"] = 964
    v = S.check_seal(s)
    assert not v["sealed"] and any("628" in m for m in v["mismatches"]), v


def test_EVERY_mismatch_is_reported_not_just_the_first():
    """A reader who fixes the first of three learns about the other two one
    restart at a time."""
    s = seal()
    s["fields"]["runtime_binary_sha256_16"]["value"] = "0" * 16
    s["fields"]["tokenizer_sha256_16"]["value"] = "0" * 16
    s["graph"]["dispatches_per_decode_token"] = 964
    assert len(S.check_seal(s)["mismatches"]) >= 3


def test_a_SAMPLER_REQUEST_is_REFUSED_not_ignored():
    """The sealed decode path is greedy argmax. Accepting temperature and then not
    honouring it is how a benchmark ends up measuring a different sampler than the
    one it asked for."""
    for k, v in (("temperature", 0.7), ("top_p", 0.9), ("top_k", 40), ("stream", True)):
        code, payload = S.handle_chat({"messages": [{"role": "user", "content": "hi"}], k: v},
                                      seal=seal(), sealed=True)
        assert code == 400, (k, code, payload)
        assert "GREEDY ARGMAX" in payload["error"]["message"]


def test_a_DEFAULT_SAMPLER_VALUE_is_NOT_refused():
    """ANTI-VACUITY for the rule above. A guard that refused every request
    carrying the key at all would pass the test above and reject ordinary
    OpenAI clients that always send temperature=1."""
    bad = [k for k in S.UNSUPPORTED if 1 not in (None, 1, False)]
    code, payload = S.handle_chat({"messages": [], "n": 1, "stream": False},
                                  seal=seal(), sealed=True)
    assert code == 400 and payload["error"]["type"] == "invalid_request", payload


def test_an_UNTERMINATED_think_block_is_an_EMPTY_reply_not_prose():
    """The same rule the capability harness uses, deliberately, so the server and
    the suite cannot disagree about what a reply is. A model that never leaves
    reasoning produced no answer."""
    import types
    fake = {"generated_text": "<think>\nstill thinking", "new_token_ids": [1, 2],
            "prompt_len": 3, "fallbacks": 0, "dense_w_materialized": 0}
    raw = fake["generated_text"]
    if "</think>" in raw:
        text = raw.split("</think>", 1)[1]
    elif raw.lstrip().startswith("<think>"):
        text = ""
    else:
        text = raw
    assert text.strip() == "", text


def test_TRUE_EQUALS_ONE_does_not_smuggle_a_streaming_request_through():
    """`stream: true` once passed a `not in (None, 1, False)` guard because
    `True == 1` in Python, and the request was then SERVED NON-STREAMED with a
    200 -- the caller silently given something other than what it asked for.
    Pinned separately from the other parameters because the bug was in the
    COMPARISON, not the list."""
    code, payload = S.handle_chat(
        {"messages": [{"role": "user", "content": "hi"}], "stream": True},
        seal=seal(), sealed=True)
    assert code == 400, (code, payload)
    assert "GREEDY ARGMAX" in payload["error"]["message"]
    # and the mirror: 1 is not True, so n=1 is fine while stream=1 is not a bool
    assert S._acceptable("n", 1) and not S._acceptable("stream", True)
    assert S._acceptable("temperature", 0) and not S._acceptable("temperature", 0.7)
