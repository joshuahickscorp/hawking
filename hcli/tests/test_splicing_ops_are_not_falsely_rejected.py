"""insert/append carry fragments by design; compiling them standalone rejects good work.

_python_syntax_violation reconstructs the resulting file for op='replace' --
the comment there records that compiling the fragment alone reported
"unexpected indent at line 1" and killed three consecutive goals that were
correct. insert_before, insert_after and append were never given the same
treatment: they fell through with candidate = body and got compiled standalone,
so an indented block spliced into an indented context was accused of a syntax
error it had not made.

Measured: two autonomy runs died here. The second was holding a correct guard
and retried the same shape three times, because the rejection described a
problem that did not exist.

Where an anchor makes the resulting file knowable, build it and judge that.
Where it does not, judge nothing -- a check that cannot see the result has
nothing to say about it.
"""
import pytest

from hcli.engine import _python_syntax_violation as violation

TARGET = "hcli/perception/store.py"
ANCHOR = "        src = Path(src)\n"
INDENTED = "        if not src.exists():\n            raise FileNotFoundError(src)\n"


def _mut(op, **kw):
    return violation({"kind": "mutation",
                      "operations": [{"op": op, "path": TARGET, **kw}]})


@pytest.mark.parametrize("op", ["insert_before", "insert_after"])
def test_an_indented_fragment_spliced_at_an_anchor_is_accepted(op):
    assert _mut(op, new_text=INDENTED, old_text=ANCHOR) is None, (
        f"{op} with a correctly-indented body was rejected; this is the false "
        "rejection already fixed for op='replace'")


def test_append_is_judged_against_the_real_file():
    assert _mut("append", new_text="\n\ndef _later():\n    return 1\n") is None


def test_append_of_genuinely_broken_code_is_still_caught():
    """The relaxation must not become a hole: a real syntax error still fails."""
    msg = _mut("append", new_text="\n\ndef broken(:\n    pass\n")
    assert msg and "would not compile" in msg


def test_no_usable_anchor_means_no_verdict():
    """An unresolvable anchor cannot be judged, so it must not be condemned."""
    assert _mut("insert_after", new_text=INDENTED,
                old_text="a string that appears nowhere in that file") is None


def test_replace_file_with_a_fragment_is_still_rejected_and_named():
    msg = _mut("replace_file", new_text=INDENTED)
    assert msg and "replaces the ENTIRE file" in msg and "op='replace'" in msg
