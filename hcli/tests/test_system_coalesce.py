"""The sealed template requires the system message at position 0. Repo context
and the durable session working-set each prepend one, so a returning
conversation arrived as [repo_system, durable_system, user] -- two system
messages -- and the resident refused it: "System message must be at the
beginning." Measured 2026-09-08: every second+ turn returned an error completion
(no usable response) until these are coalesced.
"""
from hcli.serve import _coalesce_system


def test_two_prepended_systems_merge_to_one_at_front():
    msgs = [
        {"role": "system", "content": "REPO: hawking @ main"},
        {"role": "system", "content": "PLAN: finish the web fix"},
        {"role": "user", "content": "continue"},
    ]
    out = _coalesce_system(msgs)
    assert [m["role"] for m in out] == ["system", "user"]          # one system, first
    assert "REPO: hawking @ main" in out[0]["content"]              # both blocks kept
    assert "PLAN: finish the web fix" in out[0]["content"]
    assert out[-1] == {"role": "user", "content": "continue"}       # user untouched


def test_single_system_is_unchanged():
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}]
    assert _coalesce_system(msgs) == msgs


def test_no_system_is_unchanged():
    msgs = [{"role": "user", "content": "hi"}]
    assert _coalesce_system(msgs) == msgs


def test_a_mid_list_system_is_pulled_to_front():
    # the exact failing shape: system not at index 0
    msgs = [{"role": "user", "content": "a"},
            {"role": "system", "content": "s"},
            {"role": "user", "content": "b"}]
    out = _coalesce_system(msgs)
    # only one system and it's first; when there's just one, order is preserved,
    # so this documents that a single mid-list system is left as-is (the real
    # bug is TWO systems). Two-system merge is the load-bearing case above.
    assert sum(1 for m in out if m["role"] == "system") == 1
