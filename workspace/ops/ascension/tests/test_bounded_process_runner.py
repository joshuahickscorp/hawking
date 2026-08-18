from workspace.ops.ascension.bounded_process_runner import (
    descendant_pids,
    parse_ps_rows,
)


def test_parse_and_descendant_closure() -> None:
    rows = parse_ps_rows("10 1 100 2.5\n11 10 200 3.0\n12 11 300 4.0\n99 1 400 5.0\n")
    assert rows[0] == (10, 1, 100 * 1024, 2.5)
    assert descendant_pids(10, rows) == {10, 11, 12}


def test_parse_ignores_malformed_rows() -> None:
    assert parse_ps_rows("garbage\n1 2 nope 3\n") == []
