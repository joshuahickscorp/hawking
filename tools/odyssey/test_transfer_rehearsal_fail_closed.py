"""The transfer rehearsal's audit must fail CLOSED, not open.

The module's own docstring states its one safety property: "the audit is real: a
sys.addaudithook records every file the process opens, and any read outside the
allowlist fails the rehearsal". FORBIDDEN_PREFIXES exists to catch reads of
Qwen's private working state -- the docstring calls reading them "smuggling".

A recorded read whose path cannot be resolved is currently dropped, so a read of
a forbidden location is reported clean and the rehearsal passes. An audit that
can be made to pass by handing it a path it cannot parse is not an audit.

This is the SPEC. It fails before the change and passes after.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import transfer_rehearsal as trh  # noqa: E402


def test_an_unresolvable_forbidden_read_is_not_clean():
    """A NUL byte makes Path.resolve() raise. That must not launder the read."""
    smuggled = str(pathlib.Path.home() / "noetic" / "secret\x00file")
    result = trh.audit([smuggled])
    assert not result["clean"], (
        "a forbidden read was reported clean because its path would not resolve"
    )


def test_an_ordinary_forbidden_read_is_still_caught():
    """The path that already worked must keep working."""
    plain = str(pathlib.Path.home() / "noetic" / "secret.txt")
    result = trh.audit([plain])
    assert not result["clean"]
    assert result["n_forbidden_reads"] >= 1


def test_a_harmless_read_is_still_clean():
    """Failing closed must not mean failing always."""
    # A stdlib read: outside the repo and outside home, which the audit
    # deliberately ignores. A repo file would be flagged as outside the receipt
    # allowlist, which is a different and correct complaint.
    result = trh.audit(["/usr/lib/python3/os.py"])
    assert result["clean"]
