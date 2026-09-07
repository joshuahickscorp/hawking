"""Complete-WorkUnit wall decomposition.

`prefill_profile` already attributes time INSIDE one model call. Nothing
attributed the wall of a whole WorkUnit across its seams, so "the resident is
slow" and "the verifier is slow" were indistinguishable from the outside. This
closes that.

Two properties make it honest, and both are the point:

**Leaf attribution.** Spans nest -- the model call happens inside the WorkUnit
span -- so naive summing double-counts the inner time into both buckets and
produces a total larger than the wall. Each span therefore reports only its OWN
time: elapsed minus whatever its children consumed. Named shares then sum to at
most the total, never more.

**UNEXPLAINED is reported, never folded away.** The remainder between the
outermost span and the sum of the named leaves is time the instrument did not
see. Distributing it across the named buckets, or quietly calling it zero, would
turn missing instrumentation into a confident answer -- which is the failure
this codebase has hit repeatedly. It gets its own line.

Usage:

    from hcli.wall_profile import span, report, reset

    reset()
    with span("workunit", total=True):
        with span("context_compile"):
            ...
        with span("resident"):
            ...
    report()  # {"total_ns":..., "phases":{...}, "unexplained_ns":..., ...}
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

__all__ = ["span", "report", "reset", "enabled", "set_enabled", "phase"]

_lock = threading.Lock()
_phases: Dict[str, int] = {}
_counts: Dict[str, int] = {}
_total_ns: Optional[int] = None
_total_name: Optional[str] = None
# Set from the closing frame just before a record is emitted. It is NOT an
# accumulator across records: each frame carries its own tally, because a child
# record's reset would otherwise wipe the parent's.
_nested_total_ns: int = 0
_enabled = True

_local = threading.local()

# Env-gated so this costs nothing in production and needs no further seam edits:
# when set to a path, every completed WorkUnit appends its own decomposition as
# one JSONL record and the accumulators reset for the next one. Per-unit records,
# not one merged blob -- a merged blob cannot answer "which unit was slow".
_SINK_ENV = "HAWKING_WALL_PROFILE"


def set_enabled(on: bool) -> None:
    """Off by default in production paths; the harness turns it on."""
    global _enabled
    _enabled = bool(on)


def enabled() -> bool:
    return _enabled


def _reset_accumulators() -> None:
    """Clear the phase tallies but NOT the active span stack.

    `reset()` rebinds the thread-local stack, which is right at start-up and
    wrong on emit: a WorkUnit record is emitted while its enclosing mission span
    is still open, and dropping the stack orphans that parent frame -- the next
    unit then finds an empty stack and its time is credited to nobody.
    """
    global _total_ns, _total_name, _nested_total_ns
    with _lock:
        _phases.clear()
        _counts.clear()
        _total_ns = None
        _total_name = None
        _nested_total_ns = 0


def reset() -> None:
    global _total_ns, _total_name, _nested_total_ns
    with _lock:
        _phases.clear()
        _counts.clear()
        _total_ns = None
        _total_name = None
        _nested_total_ns = 0
    _local.stack = []


def _stack() -> List[List[Any]]:
    st = getattr(_local, "stack", None)
    if st is None:
        st = []
        _local.stack = st
    return st


@contextmanager
def span(name: str, *, total: bool = False) -> Iterator[None]:
    """Time a phase. `total=True` marks the outermost span the wall is measured against."""
    if not _enabled:
        yield
        return
    st = _stack()
    frame = [time.perf_counter_ns(), 0, 0]  # [t0, child_ns, nested_total_ns]
    st.append(frame)
    try:
        yield
    finally:
        st.pop()
        elapsed = time.perf_counter_ns() - frame[0]
        own = elapsed - frame[1]
        if own < 0:  # clock skew across threads; never report a negative share
            own = 0
        if st:
            st[-1][1] += elapsed
        global _total_ns, _total_name, _nested_total_ns
        emit_now = False
        with _lock:
            if total:
                # The outermost span defines the wall. Its own leaf time is not
                # a phase -- it is whatever the named children did not cover,
                # which is exactly the unexplained remainder.
                _total_ns = elapsed if _total_ns is None else _total_ns + elapsed
                _total_name = name
                # EVERY total span closes its own record, not just the outermost.
                # A mission wraps many WorkUnits; recording only the outermost
                # would lose per-unit granularity, and recording only the inner
                # ones would make mission-level time (planning, scheduling) --
                # exactly the time that has no WorkUnit to belong to -- invisible.
                emit_now = True
                _nested_total_ns = frame[2]
            else:
                _phases[name] = _phases.get(name, 0) + own
                _counts[name] = _counts.get(name, 0) + 1
        if emit_now:
            _emit_record()
        if total and st:
            # The enclosing span must not charge this one's elapsed time to its
            # own UNEXPLAINED: that time IS explained, by the nested record.
            # Tallied on the parent FRAME, so the child's reset cannot wipe it.
            st[-1][2] += elapsed


def _emit_record(extra: Optional[Dict[str, Any]] = None) -> None:
    """Append one WorkUnit's decomposition, then reset for the next unit."""
    sink = os.environ.get(_SINK_ENV)
    if not sink:
        _reset_accumulators()
        return
    rec = report()
    rec["wall_clock"] = time.time()
    if extra:
        rec.update(extra)
    try:
        os.makedirs(os.path.dirname(sink) or ".", exist_ok=True)
        with open(sink, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        # Telemetry must never end a mission. A lost record is a lost record.
        pass
    _reset_accumulators()


def report() -> Dict[str, Any]:
    """The decomposition. `unexplained_ns` is NOT_INSTRUMENTED time, not zero."""
    with _lock:
        phases = dict(_phases)
        counts = dict(_counts)
        total = _total_ns
        total_name = _total_name
        nested = _nested_total_ns
    named = sum(phases.values())
    if total is None:
        # No outermost span ran. Refuse to invent a denominator: shares against
        # a total we never measured would be fiction.
        return {
            "total_ns": None,
            "total_source": "NOT_INSTRUMENTED",
            "phases_ns": phases,
            "phase_counts": counts,
            "named_ns": named,
            "unexplained_ns": None,
            "shares": {},
            "note": "no span(total=True) was entered; shares are not computable",
        }
    # Time spent inside NESTED total spans is explained by their own records.
    # Charging it to this span's UNEXPLAINED would make a mission look almost
    # entirely uninstrumented when in fact every WorkUnit inside it was
    # measured -- a decomposition that hides its own coverage is worse than none.
    unexplained = max(0, total - named - nested)
    shares = {k: (v / total if total else 0.0) for k, v in phases.items()}
    if nested:
        shares["NESTED_TOTALS"] = nested / total if total else 0.0
    shares["UNEXPLAINED"] = unexplained / total if total else 0.0
    return {
        "total_ns": total,
        "total_source": total_name,
        "phases_ns": phases,
        "phase_counts": counts,
        "named_ns": named,
        "nested_total_ns": nested,
        "unexplained_ns": unexplained,
        "shares": shares,
    }


def phase(name: str, *, total: bool = False):
    """Decorator form, so a seam costs one line and no reindentation."""

    def deco(fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with span(name, total=total):
                return fn(*a, **kw)

        wrapper.__wall_phase__ = name
        return wrapper

    return deco


def _self_check() -> None:
    """Leaf attribution must not double-count, and unexplained must survive.

    Driven through the SINK, because that is the path production uses: a total
    span emits a record and resets, so calling report() afterwards is empty by
    design.
    """
    import json as _json
    import tempfile

    sink = os.path.join(tempfile.mkdtemp(), "wall.jsonl")
    os.environ[_SINK_ENV] = sink
    reset()
    set_enabled(True)

    with span("wu", total=True):
        with span("a"):
            time.sleep(0.02)
            with span("b"):
                time.sleep(0.02)
        time.sleep(0.02)  # deliberately uninstrumented

    @phase("decorated")
    def _work():
        time.sleep(0.02)
        return 7

    with span("wu", total=True):
        assert _work() == 7, "decorator broke the return value"

    recs = [_json.loads(line) for line in open(sink, encoding="utf-8")]
    assert len(recs) == 2, f"expected one record per total span, got {len(recs)}"
    r, r3 = recs

    a, b = r["phases_ns"]["a"], r["phases_ns"]["b"]
    # `a` contains `b`; leaf attribution must charge b's time to b alone.
    assert 0.012e9 < a < 0.032e9, f"a double-counted its child: {a}"
    assert 0.012e9 < b < 0.032e9, f"b wrong: {b}"
    assert r["named_ns"] <= r["total_ns"], "named exceeded the wall"
    # The uninstrumented sleep must show up as UNEXPLAINED, not vanish.
    assert 0.012e9 < r["unexplained_ns"] < 0.032e9, (
        f"uninstrumented time was folded away: {r['unexplained_ns']}"
    )
    assert abs(sum(r["shares"].values()) - 1.0) < 1e-6, "shares do not sum to 1"

    # The decorator must attribute the same way the context manager does, and
    # the second record must not carry the first one's time.
    assert 0.012e9 < r3["phases_ns"]["decorated"] < 0.032e9, "decorator did not time"
    assert "a" not in r3["phases_ns"], "accumulators leaked across WorkUnits"

    # Nested totals: a mission wraps many WorkUnits. Every total closes its own
    # record, and the mission must credit the units' time to NESTED_TOTALS
    # rather than to its own UNEXPLAINED -- otherwise a fully instrumented
    # mission reads as almost entirely unmeasured.
    os.environ[_SINK_ENV] = sink2 = os.path.join(tempfile.mkdtemp(), "nested.jsonl")
    reset()
    with span("mission", total=True):
        time.sleep(0.03)  # planning: mission-level, belonging to no WorkUnit
        for _ in range(2):
            with span("workunit", total=True):
                with span("resident"):
                    time.sleep(0.02)
    nrecs = [_json.loads(line) for line in open(sink2, encoding="utf-8")]
    assert len(nrecs) == 3, f"expected 2 unit + 1 mission record, got {len(nrecs)}"
    nunits, nmission = nrecs[:2], nrecs[2]
    assert all(u["nested_total_ns"] == 0 for u in nunits), "unit records polluted by nesting"
    assert nmission["nested_total_ns"] > 0.04e9, (
        f"mission credited only {nmission['nested_total_ns']} of two units"
    )
    assert 0.02e9 < nmission["unexplained_ns"] < 0.05e9, (
        "mission UNEXPLAINED should be its own planning time, not the units'"
    )
    assert abs(sum(nmission["shares"].values()) - 1.0) < 1e-6, "nested shares do not sum to 1"

    # Without an outermost span, refuse rather than invent a denominator.
    del os.environ[_SINK_ENV]
    reset()
    with span("orphan"):
        pass
    r2 = report()
    assert r2["total_ns"] is None and r2["shares"] == {}, "invented a denominator"
    print("wall_profile self-check PASS")


if __name__ == "__main__":
    _self_check()
