"""The escape-free mutation ABI exists. Nothing steers to it, and one applier
cannot read it.

`old_lines` / `new_lines` were added to HCLI_RESULT_SCHEMA precisely because the
resident cannot reliably escape newlines inside a JSON string. The schema comment
says so: "the resident emitted a test body whose bytes literal carried \\n inside
a JSON string and produced 'unexpected character after line continuation
character'". `_operation_text` resolves them and its docstring claims "ONE
resolver for both the applier and the preflight. Two readers disagreeing about
what an operation says is exactly the defect that let a bad anchor reach
_apply_operations".

Both halves of that are still open, and receipt
`.hcli/receipts/97444aca-9035-4dab-85e4-7cd27a794d4d.json` is what it costs:
four resident calls, zero operations, three structured-output rejections reading
"unexpected character after line continuation character at line 24" and
"the reply is NOT valid JSON -- Invalid \\escape". The model emitted `\\\\n` where
it meant `\\n` inside `new_text`.

1. The prompt's mutation EXAMPLE only ever shows `new_text`, so a field the model
   is never shown is a field the model never uses. Built, not wired.
2. `hcli/mutation.py::apply_mutation_operations` reads `op["old_text"]` and
   `op["new_text"]` directly and has no idea `*_lines` exists. It is not on the
   live path today -- `hcli/engine.py::_apply_operations` is -- but
   `tools/future/resident_code_tools.py` wraps it as its PATCH primitive, so a
   line-form operation through that path would silently apply an empty replace.
   That is the exact two-readers defect the docstring names.
"""
from __future__ import annotations

from hcli import engine as eng
from hcli.mutation import apply_mutation_operations


def test_the_mutation_example_teaches_the_escape_free_form():
    # The real prompt, not the module source. Scraping the source let this test
    # find `"kind": "mutation"` in unrelated engine dicts, so it could pass while
    # the prompt the resident actually reads taught nothing.
    prompt = eng._SYSTEM_PROMPT
    assert "new_lines" in prompt, (
        "the schema offers new_lines but no prompt text does; a field the model "
        "is never shown is a field the model never uses"
    )
    # And it must appear in the worked mutation EXAMPLE, not only in a schema
    # comment the model never sees.
    import re
    compact = re.sub(r"\s+", "", prompt)
    example_start = compact.index('"kind":"mutation"')
    example_end = compact.index('"kind":"tool_use"')
    example = compact[example_start:example_end]
    assert "new_lines" in example, (
        "new_lines is mentioned somewhere but not in the worked mutation example, "
        "which is the only part the resident actually imitates"
    )


def test_the_resolver_is_shared_so_two_readers_cannot_disagree():
    """`_operation_text` must be THE resolver, not one of two."""
    import hcli.mutation as mut
    assert hasattr(mut, "operation_text"), (
        "hcli/mutation.py has no line-form resolver of its own and does not "
        "import one, so it reads old_text/new_text directly"
    )
    op = {"op": "replace", "path": "x.py", "new_lines": ["a = 1", "b = 2"]}
    assert mut.operation_text(op, "new_text") == "a = 1\nb = 2\n"
    assert eng._operation_text(op, "new_text") == mut.operation_text(op, "new_text")


def test_the_other_applier_honours_line_form(tmp_path):
    """A line-form replace through mutation.py must change the file."""
    target = tmp_path / "t.py"
    target.write_text("x = 0\n", encoding="utf-8")

    class Guard:
        def resolve(self, path):
            return str(tmp_path / path)

    res = apply_mutation_operations(
        Guard(),
        [{"op": "replace", "path": "t.py", "old_lines": ["x = 0"],
          "new_lines": ["x = 1", "y = 2"]}],
    )
    assert target.read_text(encoding="utf-8") == "x = 1\ny = 2\n", (
        "the line-form operation was read as empty old_text/new_text and applied "
        f"nothing; file still reads {target.read_text(encoding='utf-8')!r}"
    )
    assert res.get("changed") or res.get("files")


def test_string_form_still_works(tmp_path):
    """The escaped-string form must keep working; this is additive."""
    target = tmp_path / "t.py"
    target.write_text("x = 0\n", encoding="utf-8")

    class Guard:
        def resolve(self, path):
            return str(tmp_path / path)

    apply_mutation_operations(
        Guard(),
        [{"op": "replace", "path": "t.py", "old_text": "x = 0", "new_text": "x = 9"}],
    )
    assert target.read_text(encoding="utf-8") == "x = 9\n"
