"""The gates' _bind_emit() is dead code, and proving that is worth a test.

An isolation audit reported that six agentos gates each do `sc.emit = emit`,
clobbering a shared module's global so "whichever gate ran last wins". I
restated that, and started writing a consolidation to fix it.

It is not true. Every gate's _bind_emit() opens with
`if hasattr(sc, "emit"): return`, and tools/verify/status_causality.py defines
emit natively at module scope. The guard is therefore always True and NO GATE
EVER ASSIGNS. The fallback exists for a status_causality that predates emit.

The consolidation I nearly shipped bound unconditionally -- it would have
REPLACED the real emit with a gate's fallback copy and created the very
clobbering that did not exist. A fix for an imagined bug that installs a real
one.

This pins the actual invariant so neither mistake can be made again.
"""
import importlib
import inspect
from pathlib import Path

GATES = sorted(Path("hcli/agentos").glob("*_gate.py"))


def test_status_causality_owns_emit():
    sc = importlib.import_module("tools.verify.status_causality")
    assert hasattr(sc, "emit"), "the gates' fallback would now actually bind"
    assert inspect.getmodule(sc.emit) is sc, (
        f"sc.emit came from {inspect.getmodule(sc.emit)}, not status_causality -- "
        "something has monkeypatched it after all")


def test_every_gate_guards_its_fallback_bind():
    """A gate that dropped the guard would silently win the race."""
    checked = 0
    for g in GATES:
        src = g.read_text()
        if "sc.emit = emit" not in src:
            continue
        checked += 1
        assert 'if hasattr(sc, "emit"):' in src, (
            f"{g} assigns sc.emit without the hasattr guard; import order now "
            f"decides which implementation wins")
    assert checked >= 5, f"expected the fallback in several gates, found {checked}"


def test_importing_every_gate_leaves_emit_alone():
    """The behavioural check: import them all, emit must still be the real one."""
    sc = importlib.import_module("tools.verify.status_causality")
    before = sc.emit
    for g in GATES:
        try:
            importlib.import_module(f"hcli.agentos.{g.stem}")
        except Exception:
            continue  # a gate that will not import cannot rebind anything
    assert sc.emit is before, "importing the gates rebound status_causality.emit"
