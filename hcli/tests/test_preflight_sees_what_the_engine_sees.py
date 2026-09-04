"""The preflight must parse a reply the way the engine parses it.

_python_syntax_violation used a bare json.loads. The engine uses
extract_json_object, which tolerates a markdown fence and a sentence of preamble
and then acts on what it finds. So for exactly those replies the preflight
returned None and did nothing -- and every correction built on it, the anchor
retry, the syntax retry and the quoted offending line, was skipped without a
trace in the receipt.

Measured: a 343-character anchor that was correct but for ONE character --
'len(raw}' where 'len(raw)}' belongs -- reached _apply_operations and killed the
unit, while the receipt recorded attempts=2 and errors=[]. The contract had found
nothing to complain about because it had never been shown the reply.

Two parsers disagreeing about what a reply says is one parser too many.
"""
from __future__ import annotations

import json
import pathlib

from hcli.engine import _python_syntax_violation

TARGET = "hcli/tool_registry.py"

# The real anchor with one character wrong, exactly as the resident emitted it.
BAD_ANCHOR = (
    '    clipped = raw[:limit]\n    return {\n        "path": str(path),\n'
    '        "bytes": len(raw),\n        "truncated": len(raw) > limit,\n'
    '        "sha256": _sha256_bytes(raw),\n'
    '        "content": clipped.decode(encoding, errors="replace"),\n'
    '        "artifact": {"kind": "file", "path": str(path), '
    '"sha256": _sha256_bytes(raw), "bytes": len(raw},\n    }'
)


def _reply() -> str:
    return json.dumps({"kind": "mutation", "operations": [{
        "op": "replace", "path": TARGET,
        "old_text": BAD_ANCHOR, "new_text": "    x = 1\n",
    }]})


def test_the_bad_anchor_really_is_absent_from_the_file():
    """Otherwise every case below is vacuous."""
    assert pathlib.Path(TARGET).read_text().count(BAD_ANCHOR) == 0


def test_a_plain_reply_is_checked():
    assert _python_syntax_violation(_reply()) is not None


def test_a_FENCED_reply_is_checked():
    fenced = "```json\n" + _reply() + "\n```"
    assert _python_syntax_violation(fenced) is not None, (
        "a markdown fence silently disabled the whole preflight"
    )


def test_a_reply_with_PREAMBLE_is_checked():
    prefaced = "Here is the change:\n" + _reply()
    assert _python_syntax_violation(prefaced) is not None


def test_an_already_parsed_reply_is_checked():
    assert _python_syntax_violation(json.loads(_reply())) is not None


def test_a_reply_that_is_not_JSON_at_all_is_still_skipped_quietly():
    """Unparseable replies are the schema layer's business, not this one's."""
    assert _python_syntax_violation("I could not complete this task.") is None


def test_a_correct_operation_still_passes_through_a_fence():
    src = pathlib.Path(TARGET).read_text()
    anchor = "    clipped = raw[:limit]\n"
    assert src.count(anchor) == 1
    good = json.dumps({"kind": "mutation", "operations": [{
        "op": "replace", "path": TARGET,
        "old_text": anchor, "new_text": anchor,
    }]})
    assert _python_syntax_violation("```json\n" + good + "\n```") is None
