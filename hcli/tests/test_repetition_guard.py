"""Greedy repetition-collapse guard: the web path must not serve a 2048-token
loop as a completion.

Measured on the live sealed-3.14 / ascension-qwen38 greedy resident 2026-09-08:
the prompt "List the first 12 US presidents ..." answered #1 and #2 correctly,
stumbled on #3 ("to be a a"), then collapsed into " (S) (S) (S) ..." for ~2000
tokens until it hit the 2048 cap -- 61 s of generation the single-chunk
connector buffered before returning, so the browser showed a blank spinner then
cap-truncated garbage. This is not model-specific: any body decoded greedily
without a tuned anti-repetition kernel can fall into a short-cycle attractor.
The guard trims the collapse to its coherent prefix and flags it honestly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hcli.serve import trim_repetition_collapse, _content_and_flags


REAL_TAIL = " (S)" * 500  # the observed collapse


def test_real_collapse_is_trimmed_to_the_coherent_prefix():
    good = ("Here are the first 12 U.S. presidents:\n"
            "1. George Washington: unanimously elected.\n"
            "2. John Adams: moved the capital.\n"
            "3. Thomas Jefferson: He was the first president to be a a \"S\"")
    text = good + REAL_TAIL
    trimmed, collapsed = trim_repetition_collapse(text)
    assert collapsed is True
    assert "(S) (S) (S)" not in trimmed          # the loop is gone
    assert "George Washington" in trimmed          # the real answer survives
    assert "John Adams" in trimmed
    assert len(trimmed) < len(text) / 4            # the bulk was garbage


def test_normal_prose_is_never_trimmed():
    # legitimate text with ordinary repetition must NOT trip the guard
    normal = ("The cat sat on the mat. The cat was happy. "
              "Paris is the capital of France. Two plus two is four. "
              "I think, therefore I am. This is a complete answer.")
    trimmed, collapsed = trim_repetition_collapse(normal)
    assert collapsed is False
    assert trimmed == normal


def test_short_text_is_left_alone():
    for s in ("HCLI_WEB_OK", "4", "", "hi there"):
        trimmed, collapsed = trim_repetition_collapse(s)
        assert collapsed is False and trimmed == s


def test_single_token_char_loop_is_caught():
    text = "The answer is " + ("a" * 4000)
    trimmed, collapsed = trim_repetition_collapse(text)
    assert collapsed is True
    assert len(trimmed) < 100


def test_flags_flow_into_the_response_contract():
    class R:
        text = "ok answer here" + REAL_TAIL
        finish_reason = "length"
        degraded = []
        prompt_tokens = 10
        completion_tokens = 2048
        total_tokens = 2058
    content, finish, degraded = _content_and_flags(R())
    assert "(S) (S)" not in content
    assert "repetition_collapse" in degraded       # honest flag for the receipt
