"""The dashboard must not invent a zero for an instrument that did not report.

Directive XXXII: "Unknown means NOT_INSTRUMENTED. Never silently substitute
zero." ``hcli_metric`` currently sums ``prefill_profile.totals.wall_ns`` with a
default of ``0``, so a model call that carried no prefill profile contributes
nothing and reads as a call whose prefill was free. The second half is worse
than the first: decode is derived as ``wall - prefill``, so the missing prefill
is not merely lost, it is REATTRIBUTED to decode. A reader sees a unit that
spent all its wall decoding.

The contract this pins:

* ``prefill_seconds(calls)`` returns ``None`` when ANY call in the unit lacks a
  prefill measurement -- an unknown addend makes the whole sum unknown.
* ``decode_seconds(calls)`` returns ``None`` whenever prefill is unknown, because
  ``wall - unknown`` is unknown, not ``wall``.
* Both return the real number when every call is instrumented.
"""
from __future__ import annotations

import pytest

from tools import hcli_metric


def _call(wall_s: float, prefill_ns: int | None) -> dict:
    call: dict = {"wall_s": wall_s, "prompt_tokens": 10, "completion_tokens": 5}
    if prefill_ns is not None:
        call["prefill_profile"] = {"totals": {"wall_ns": prefill_ns}}
    return call


def test_fully_instrumented_unit_reports_real_seconds():
    cs = [_call(10.0, 6_000_000_000), _call(4.0, 1_000_000_000)]
    assert hcli_metric.prefill_seconds(cs) == pytest.approx(7.0)
    assert hcli_metric.decode_seconds(cs) == pytest.approx(7.0)


def test_one_uninstrumented_call_makes_the_whole_prefill_unknown():
    cs = [_call(10.0, 6_000_000_000), _call(4.0, None)]
    assert hcli_metric.prefill_seconds(cs) is None, (
        "a call with no prefill_profile contributed 0 and the sum was reported "
        "as if it were measured"
    )


def test_unknown_prefill_is_not_reattributed_to_decode():
    cs = [_call(10.0, 6_000_000_000), _call(4.0, None)]
    assert hcli_metric.decode_seconds(cs) is None, (
        "decode was computed as wall - prefill with prefill defaulted to 0, so "
        "the uninstrumented prefill was credited to decode"
    )


def test_a_unit_with_no_calls_at_all_is_unknown_not_zero():
    assert hcli_metric.prefill_seconds([]) is None
    assert hcli_metric.decode_seconds([]) is None


# ---------------------------------------------------------------------------
# The tests above pin the NEW behaviour. On their own they licensed the DESTRUCTION
# of the old: receipt ecf6d616 was ACCEPTED -- kind mutation, status completed,
# py_compile exit 0, 4 of 4 tests green, red_before_green True -- for a
# `replace_file` that cut tools/hcli_metric.py from 202 lines to 72, deleted the
# entire dashboard, and left two runtime NameErrors (`NOT_INSTRstrumented`,
# `preffill`) that py_compile cannot see because they are not syntax errors.
#
# The verifier did exactly what it was told. The test was the weak link. A suite
# that pins only what you are adding is a licence to delete everything else.


def test_the_dashboard_itself_survives():
    """The module is a DASHBOARD, not two helpers. Pin its actual surface."""
    import tools.hcli_metric as m

    for name in ("main", "load", "accepted", "calls", "interventions",
                 "regime_of", "REGIMES", "RECEIPTS", "LEDGER"):
        assert hasattr(m, name), (
            f"tools/hcli_metric.py lost {name!r}; a mutation that adds two "
            "functions must not remove the tool they belong to"
        )


def test_no_undefined_names_survive_py_compile():
    """py_compile is a SYNTAX gate, not a semantic one.

    `NOT_INSTrumented` and `preffill` both compile. Both are NameError at the
    moment the line runs, and neither of those lines is on the path the four
    tests above exercise, so the whole thing passed.
    """
    import ast
    import builtins
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.joinpath("hcli_metric.py").read_text()
    tree = ast.parse(src)

    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.comprehension):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    defined.add(n.id)

    loaded = {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    # Module dunders are bound by the interpreter, not by any statement.
    unresolved = sorted(
        loaded - defined
        - {"annotations", "__file__", "__name__", "__doc__", "__package__", "__spec__"}
    )
    assert not unresolved, (
        f"names read but never bound anywhere in the module: {unresolved}. "
        "py_compile passes on every one of these."
    )


def test_main_still_runs_end_to_end():
    """The one check that a whole-file rewrite cannot fake."""
    import subprocess
    import sys
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, str(root / "tools" / "hcli_metric.py"), "0"],
        capture_output=True, text=True, timeout=120, cwd=str(root),
    )
    assert proc.returncode in (0, 1), (
        f"hcli_metric.py exited {proc.returncode}\n{proc.stderr[-1500:]}"
    )
    assert "PRIMARY METRIC" in proc.stdout or "no receipts" in proc.stdout, (
        "the dashboard printed neither its primary metric nor its empty-state "
        f"message; stdout was {proc.stdout[:400]!r}"
    )
