"""Every worked example in the system prompt must satisfy the schema it teaches.

`HCLI_RESULT_SCHEMA` lists `tool_calls` in `required`, but two of the three
worked examples in `_SYSTEM_PROMPT` omitted the key entirely. A model that
copied the example it was shown was rejected for `missing required property
'tool_calls'` -- three times, then the WorkUnit failed. The instruction and the
validator disagreed, and the validator won every argument.

This is a producer test, not a receipt test: it extracts the JSON objects from
the live prompt string and runs the live validator over them. It fails if either
side drifts, and it names which example broke.
"""
from __future__ import annotations

import json

import pytest

from hcli.backends import validate_against_schema
from hcli.engine import HCLI_RESULT_SCHEMA, _SYSTEM_PROMPT


def _worked_examples(text: str):
    """Every balanced top-level {...} block in the prompt, in order."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
                start = -1
    return out


def test_the_prompt_contains_the_examples_we_think_it_does():
    """Guard the extractor itself: a parser that finds nothing passes vacuously."""
    blocks = _worked_examples(_SYSTEM_PROMPT)
    assert len(blocks) >= 3, f"only {len(blocks)} worked examples found in the prompt"
    kinds = sorted(json.loads(b).get("kind") for b in blocks)
    assert kinds == ["answer", "mutation", "tool_use"]


@pytest.mark.parametrize("index", range(3))
def test_each_worked_example_validates(index):
    blocks = _worked_examples(_SYSTEM_PROMPT)
    example = json.loads(blocks[index])
    err = validate_against_schema(example, HCLI_RESULT_SCHEMA)
    assert err is None, (
        f"worked example {index} (kind={example.get('kind')!r}) is rejected by the "
        f"schema the same prompt demands: {err}. A model that copies what it was "
        f"shown burns every retry."
    )


def test_the_probe_reads_the_live_schema_not_a_copy():
    """The measuring instrument had its own stale duplicate with no tool_calls."""
    import tools.headless.structured_output_probe as probe

    assert probe.RESULT_SCHEMA is HCLI_RESULT_SCHEMA
    assert probe.SYSTEM is _SYSTEM_PROMPT
