"""A patch form the model cannot syntactically break.

JSON asks for quotes, escapes, brackets and a closing brace before the edit is
even expressible. Measured with every precondition finally satisfied at once --
correct objective, evidence containing the target region, zero tools, the unique
anchor supplied in the goal, a 2048-token budget -- the resident produced 1,338
to 1,446 tokens on three consecutive attempts and never closed the object.

That is the model's own frontier rather than the harness denying it something,
and no better error message moves it. A block form has nothing to close.
"""
from __future__ import annotations

import json

from hcli.engine import _patch_block_to_operations

BLOCK = """PATH: hcli/tool_registry.py
FIND:
    clipped = raw[:limit]
REPLACE:
    clipped = raw[:limit]
    total = len(text.splitlines())
END
"""


def test_a_block_becomes_one_replace_operation():
    out = _patch_block_to_operations(BLOCK)
    assert out["kind"] == "mutation"
    op = out["operations"][0]
    assert op["op"] == "replace"
    assert op["path"] == "hcli/tool_registry.py"


def test_indentation_survives_verbatim():
    """An edit whose leading spaces were eaten would break the file it patches."""
    op = _patch_block_to_operations(BLOCK)["operations"][0]
    assert op["old_text"] == "    clipped = raw[:limit]"
    assert op["new_text"].startswith("    clipped = raw[:limit]\n    total =")


def test_surrounding_prose_is_ignored():
    out = _patch_block_to_operations("Sure, here is the change.\n\n" + BLOCK + "\nDone.")
    assert out is not None


def test_plain_prose_is_not_a_patch():
    assert _patch_block_to_operations("I could not complete this.") is None


def test_an_empty_find_is_refused():
    """An empty anchor would match everywhere; the applier's uniqueness rule
    is the whole safety property and must not be bypassed by a blank block."""
    assert _patch_block_to_operations("PATH: a.py\nFIND:\n\nREPLACE:\nx = 1\nEND") is None
