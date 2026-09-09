"""A tool named shell.readonly must not be able to write a file.

`sed` was on the read-only allowlist. Its `w` flag writes a file from inside
the script argument, so `_shell_readonly`'s two guards both miss it: the
forbidden-option set only inspects tokens like `-i`/`--in-place`, and the
path-containment check skips any token that is not absolute and does not start
with "." or "/" -- which `s/a/b/w out` does not. A registry holding ONLY
READ_ONLY wrote both a relative and an absolute path through it.

hcli/delegate.py's READ_ONLY_VERBS already excluded sed, with a comment saying
its own negative control had written a protected file. The two allowlists had
drifted apart. This test pins the invariant on the registry side so they cannot
drift again silently.
"""
import os
import shlex
from pathlib import Path

import pytest

import hcli.tool_registry as tr


def _ro_registry(root):
    return tr.default_tool_registry(str(root), repo_root=str(root),
                                    permissions={tr.READ_ONLY})


def test_sed_is_not_on_the_readonly_allowlist():
    assert "sed" not in tr._SAFE_SHELL_COMMANDS


def test_the_two_allowlists_agree_on_what_can_write():
    """delegate.py hardened its list first. Neither may re-admit a writer."""
    from hcli.delegate import READ_ONLY_VERBS
    writers = {"sed", "awk", "sort", "find", "python", "python3", "pytest", "tee", "dd"}
    assert not (writers & tr._SAFE_SHELL_COMMANDS), tr._SAFE_SHELL_COMMANDS & writers
    assert not (writers & READ_ONLY_VERBS), READ_ONLY_VERBS & writers


@pytest.mark.parametrize("out", ["PROOF_REL.txt", None])
def test_readonly_shell_cannot_write_a_file(tmp_path, monkeypatch, out):
    """The exact bypass, both relative and absolute, must now be refused."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src.txt").write_text("hello\n")
    target = tmp_path / (out or "PROOF_ABS.txt")
    spec = str(target) if out is None else out
    reg = _ro_registry(tmp_path)
    res = reg.invoke("shell.readonly",
                     {"command": f"sed {shlex.quote('s/hello/pwned/w ' + spec)} src.txt"}).to_dict()
    assert not res["ok"], f"read-only shell accepted a writing command: {res}"
    assert "allowlist" in str(res.get("error", "")).lower()
    assert not target.exists(), f"shell.readonly WROTE {target}"


def test_a_genuinely_read_only_command_still_works(tmp_path, monkeypatch):
    """Negative control: the fix must not break the tool it hardens."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src.txt").write_text("hello\nworld\n")
    reg = _ro_registry(tmp_path)
    res = reg.invoke("shell.readonly", {"command": "grep hello src.txt"}).to_dict()
    assert res["ok"], res
    assert "hello" in str(res["value"])
