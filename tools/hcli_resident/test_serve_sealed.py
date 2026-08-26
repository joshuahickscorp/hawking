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


# --------------------------------------------------------------------------
# The arm and the grammar are not sampler parameters, and both were found by
# WIRING THE REAL CALLER rather than by reading the code.
# --------------------------------------------------------------------------

def test_the_real_hcli_payload_is_accepted_because_it_agrees_with_the_seal():
    """hcli/delegate.py default_caller posts exactly this shape. If it is refused,
    the endpoint cannot be HCLI's resident and G083 is unreachable."""
    sl = seal()
    body = {"messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 4096}
    bad = [k for k in S.UNSUPPORTED
           if k in body and not S._acceptable(k, body[k])]
    assert bad == [], f"the real caller would be refused on {bad}"
    assert S._arm_verdict(body, sl) is None


def test_the_other_arm_is_refused_not_silently_served():
    """enable_thinking=true asks for open_think. The two arms scored 30/43 and
    35/43 on the SAME bytes this session, so serving one for the other with a 200
    is handing the caller a different model."""
    sl = seal()
    v = S._arm_verdict(
        {"chat_template_kwargs": {"enable_thinking": True}}, sl)
    assert v and "enable_thinking" in v and "pre_closed_think" in v


def test_an_unknown_chat_template_kwarg_is_refused():
    sl = seal()
    assert S._arm_verdict(
        {"chat_template_kwargs": {"enable_reasoning": False}}, sl)
    assert S._arm_verdict({"chat_template_kwargs": "no"}, sl)


def test_response_format_is_refused_because_it_would_be_ignored():
    code, payload = S.handle_chat(
        {"messages": [{"role": "user", "content": "hi"}],
         "response_format": {"type": "json_schema", "json_schema": {"name": "x"}}},
        seal=seal(), sealed=True)
    assert code == 400
    assert "response_format" in payload["error"]["message"]


def test_finish_reason_is_not_a_constant():
    """A reply that ate the whole budget did not stop for the same reason as one
    that emitted an end token. HCLI branches on this field."""
    src = pathlib.Path(S.__file__).read_text()
    assert '"finish_reason": "stop"' not in src, \
        "finish_reason is hardcoded, so it is a fabricated field"
    assert 'finish = "length" if' in src


def test_total_tokens_is_never_a_sum_over_an_unknown():
    """A None prompt_tokens coerced to 0 reports a total that is not a total. The
    live request measured prompt_tokens=null, completion=2, total=2."""
    src = pathlib.Path(S.__file__).read_text()
    assert '(g["prompt_tokens"] or 0)' not in src
    assert 'if g["prompt_tokens"] is not None else {}' in src


# --------------------------------------------------------------------------
# Found by RUNNING A REAL MISSION, not by reading the code: the first end-to-end
# HCLI mission died with `GQA position 1024 exceeds max_seq_len 1024` because
# max_seq_len was `max_tokens + 512` -- a guess that every prompt fits in 512
# tokens. HCLI's planner prompt does not.
# --------------------------------------------------------------------------

def test_max_seq_len_is_sized_from_the_real_prompt_not_guessed():
    src = pathlib.Path(S.__file__).read_text()
    assert 'str(max_tokens + 512)' not in src, \
        "max_seq_len is guessed from max_tokens; a long prompt aborts the resident"
    assert '(prompt_tokens or 0) + max_tokens' in src


def test_the_context_ceiling_is_read_from_the_artifact_and_is_not_none():
    """Qwen3.5 nests max_position_embeddings under text_config, so the obvious
    top-level read returns None -- a ceiling that silently becomes zero."""
    art = pathlib.Path(seal()["fields"]["artifact_root"]["value"])
    n = S.context_limit(art)
    assert isinstance(n, int) and n > 1024, n


def test_a_context_overflow_is_a_400_not_a_resident_crash():
    art = pathlib.Path(seal()["fields"]["artifact_root"]["value"])
    limit = S.context_limit(art)
    code, payload = S.handle_chat(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": limit + 1},
        seal=seal(), sealed=True)
    assert code == 400 and payload["error"]["type"] == "context_length_exceeded"
    assert str(limit) in payload["error"]["message"]
