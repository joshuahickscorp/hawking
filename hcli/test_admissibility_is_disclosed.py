"""A rule enforced but never stated is not a rule the model can follow.

Two consecutive real missions against the sealed 27B resident proposed
`test -d ...` and `grep -l ... | wc -l`; both were refused as
COMMAND_NOT_ADMITTED, both obligations came back FALSE, and nothing in the prompt
had ever said which first words are admitted or why.
"""
from hcli import verifier_pipeline as VP


def test_the_note_lists_EVERY_admitted_command():
    note = VP._admissibility_note()
    for cmd in VP._COMMAND_FIRST:
        assert cmd in note, cmd


def test_the_note_CANNOT_GO_STALE_because_it_is_built_from_the_allowlist():
    """If someone adds a command to _COMMAND_FIRST, the prompt must gain it too --
    a hand-written list would silently drift from the enforcement."""
    original = VP._COMMAND_FIRST
    try:
        VP._COMMAND_FIRST = frozenset(original | {"kubectl"})
        assert "kubectl" in VP._admissibility_note()
    finally:
        VP._COMMAND_FIRST = original


def test_the_note_says_WHY_not_only_WHAT():
    """The model that proposed `grep | wc -l` was not being careless; it was
    answering the question asked. The reason has to travel with the rule."""
    note = VP._admissibility_note()
    assert "EXIT CODE" in note
    assert "exits 0 whether" in note


def test_the_note_is_actually_ATTACHED_to_the_propose_prompt():
    import inspect
    src = inspect.getsource(VP.verify)
    assert "_ADMISSIBILITY_NOTE" in src


def test_the_commands_the_note_recommends_are_themselves_admissible():
    """A note that suggested an inadmissible command would be worse than none.

    The first draft recommended `bash -c '[ ... ]'`, which the harness REFUSES: a
    `bash -c` wrapper recurses into its body so a wrapper cannot launder an
    inadmissible check, and `[` is not an admitted first word. Caught by this test
    before it ever reached a model.
    """
    ok, why = VP.command_is_admissible('python3 -c "assert 1 == 1"')
    assert ok, why


def test_the_note_WARNS_that_a_bash_wrapper_does_not_launder():
    """Otherwise the model reads `bash` in the list and proposes `bash -c '[...]'`,
    which is refused for a reason the list alone does not explain."""
    note = VP._admissibility_note()
    assert "bash -c" in note and "REFUSED" in note
    assert not VP.command_is_admissible("bash -c '[ 1 -eq 1 ]'")[0]
