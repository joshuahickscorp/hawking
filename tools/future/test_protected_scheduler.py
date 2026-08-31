"""Pins for separating scheduler capability from window availability.

A guard nobody has watched fail is not a guard. decide() must return
BLOCKED_ON_PROTECTED_WINDOW on HEAVY without marking the scheduler
incapable, RUNNABLE when QUIESCENT+holder are supplied as INPUTS (never as a
written lease), and must refuse to park a unit that does not need protection.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tools.future import protected_scheduler as ps
from tools.future import protected_window as pw
from tools.future import qualification_pipeline as qp
from tools.future._common import RECEIPTS, REPO, _assert_no_hardware_claims


# ---------------------------------------------------------------------------
# Fixtures — simulate INPUTS, never a lock file
# ---------------------------------------------------------------------------


def _gpu(**extra: object) -> dict:
    row = dict(ps.PROBE_UNIT)
    row.update(extra)
    return row


def _cpu(**extra: object) -> dict:
    row = {
        "id": extra.pop("id", "future.cpu.unrelated"),
        "resource_class": "STATIC_ANALYSIS",
        "requires_quiescence": False,
        "description": "CPU-class work that does not need a protected window",
    }
    row.update(extra)
    return row


def _dirty(**extra: object) -> dict:
    row = {
        "id": extra.pop("id", "future.dirty.decode"),
        "resource_class": "GPU_DIRTY_OK",
        "requires_quiescence": False,
    }
    row.update(extra)
    return row


def _cont(klass: str) -> dict:
    return {"contamination_class": klass}


def _lease(*, present: bool, pids: list[int] | None = None) -> dict:
    holder_pids = list(pids if pids is not None else ([99] if present else []))
    return {
        "present": present,
        "holders": {"status": "OK" if holder_pids else "SKIPPED", "pids": holder_pids},
        "lock_file_exists": present,
        "reason": "injected input" if present else "injected: no proven holder",
    }


def _lock_paths() -> list[Path]:
    rels = list(pw.DEFAULT_LOCK_RELS)
    paths: list[Path] = []
    seen: set[str] = set()
    roots = list(pw._checkout_roots())
    for root in roots:
        for rel in rels:
            p = Path(root) / rel
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            paths.append(p)
    paths.append(Path("/tmp/hawking_protected_window.lease"))
    return paths


def _mtime_snapshot() -> dict[str, tuple[int, int] | None]:
    out: dict[str, tuple[int, int] | None] = {}
    for p in _lock_paths():
        try:
            st = p.stat()
            out[str(p)] = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            out[str(p)] = None
    return out


# ---------------------------------------------------------------------------
# recognize
# ---------------------------------------------------------------------------


def test_recognize_gpu_exclusive_is_protected():
    rec = ps.recognize(_gpu())
    assert rec["recognized"] is True
    assert rec["protected_required"] is True
    assert rec["resource_class"] == "GPU_EXCLUSIVE"
    assert rec["guessed"] is False
    assert rec["gpu_authority"] is False


def test_recognize_is_by_declared_class_not_the_name():
    rec = ps.recognize(
        _cpu(id="future.protected-window.staged-qualification", resource_class="STATIC_ANALYSIS")
    )
    assert rec["recognized"] is True
    assert rec["protected_required"] is False
    assert rec["guessed"] is False


def test_recognize_dirty_ok_is_not_protected():
    rec = ps.recognize(_dirty())
    assert rec["protected_required"] is False


def test_recognize_gpu_decode_is_not_protected():
    rec = ps.recognize({"id": "decode", "resource_class": "GPU_DECODE"})
    assert rec["recognized"] is True
    assert rec["protected_required"] is False


def test_recognize_unknown_class_is_not_mapped_to_light_control():
    rec = ps.recognize({"id": "mystery", "resource_class": "NOT_A_CLASS"})
    assert rec["recognized"] is False
    assert rec["protected_required"] is False
    assert rec["resource_class"] is None
    assert "rather than map unknown -> LIGHT_CONTROL" in rec["reason"]


def test_recognize_missing_class_is_refused():
    rec = ps.recognize({"id": "no-class"})
    assert rec["recognized"] is False
    d = ps.decide({"id": "no-class"}, contamination=_cont("QUIESCENT"), lease=_lease(present=True))
    assert d["verdict"] == "REFUSED"


def test_recognize_mutation_is_refused():
    rec = ps.recognize({"id": "mut", "resource_class": "MUTATION"})
    assert rec["recognized"] is False
    d = ps.decide({"id": "mut", "resource_class": "MUTATION"})
    assert d["verdict"] == "REFUSED"


def test_recognize_non_mapping_is_refused():
    rec = ps.recognize("GPU_EXCLUSIVE")
    assert rec["recognized"] is False
    d = ps.decide("GPU_EXCLUSIVE")
    assert d["verdict"] == "REFUSED"


def test_requires_quiescence_true_is_a_declaration():
    rec = ps.recognize(_cpu(requires_quiescence=True))
    assert rec["protected_required"] is True


def test_requires_quiescence_string_is_not_a_bool_declaration():
    rec = ps.recognize(_cpu(requires_quiescence="true"))
    assert rec["protected_required"] is False


# ---------------------------------------------------------------------------
# decide — the load-bearing split
# ---------------------------------------------------------------------------


def test_quiescent_and_lease_input_is_runnable_without_writing_a_lease():
    before = _mtime_snapshot()
    d = ps.decide(
        _gpu(),
        contamination=_cont("QUIESCENT"),
        lease=_lease(present=True, pids=[99]),
    )
    after = _mtime_snapshot()
    assert d["verdict"] == "RUNNABLE"
    assert d["protected_required"] is True
    assert d["window_available"] is True
    assert d["scheduler_capable"] is True
    assert d["gpu_authority"] is False
    assert d["inputs_simulated"] is True
    assert d["wake_condition"] is None
    assert after == before
    for p in _lock_paths():
        if "protected-accelerator-bench.lock" in str(p) or "qwen-protected-bench.lock" in str(p):
            # Injected inputs must not mint a lock in the worktree.
            if REPO in p.parents or p.parent == REPO / ".hcli" / "locks":
                if before.get(str(p)) is None:
                    assert not p.exists()


def test_injected_present_without_pids_is_not_a_lease():
    d = ps.decide(
        _gpu(),
        contamination=_cont("QUIESCENT"),
        lease={"present": True, "holders": {"pids": []}},
    )
    assert d["verdict"] == "BLOCKED_ON_PROTECTED_WINDOW"
    assert d["lease_present"] is False
    assert d["window_available"] is False
    assert d["scheduler_capable"] is True


def test_heavy_blocks_the_window_not_the_scheduler():
    d = ps.decide(
        _gpu(),
        contamination=_cont("HEAVY"),
        lease=_lease(present=False),
    )
    assert d["verdict"] == "BLOCKED_ON_PROTECTED_WINDOW"
    assert d["scheduler_capable"] is True
    assert d["window_available"] is False
    assert d["wake_condition"]["all_of"] == list(ps.WAKE_ALL_OF)
    assert "fcntl.flock" in " ".join(d["wake_condition"]["never"])
    assert "incapable" in " ".join(d["wake_condition"]["never"])


def test_light_and_unknown_are_not_quiescent():
    for klass in ("LIGHT", "UNKNOWN"):
        d = ps.decide(_gpu(), contamination=_cont(klass), lease=_lease(present=True))
        assert d["verdict"] == "BLOCKED_ON_PROTECTED_WINDOW", klass
        assert d["scheduler_capable"] is True, klass
        assert d["window_available"] is False, klass


def test_quiescent_without_lease_is_blocked():
    d = ps.decide(_gpu(), contamination=_cont("QUIESCENT"), lease=_lease(present=False))
    assert d["verdict"] == "BLOCKED_ON_PROTECTED_WINDOW"
    assert d["scheduler_capable"] is True
    assert d["lease_present"] is False


def test_lease_without_quiescent_is_blocked():
    d = ps.decide(_gpu(), contamination=_cont("HEAVY"), lease=_lease(present=True))
    assert d["verdict"] == "BLOCKED_ON_PROTECTED_WINDOW"
    assert d["scheduler_capable"] is True


def test_unprotected_unit_is_runnable_on_a_heavy_machine():
    d = ps.decide(_cpu(), contamination=_cont("HEAVY"), lease=_lease(present=False))
    assert d["verdict"] == "RUNNABLE"
    assert d["protected_required"] is False
    assert d["scheduler_capable"] is True


def test_dirty_ok_is_runnable_on_heavy_and_never_parked():
    d = ps.decide(_dirty(), contamination=_cont("HEAVY"), lease=_lease(present=False))
    assert d["verdict"] == "RUNNABLE"
    parked = ps.park(_dirty(), contamination=_cont("HEAVY"), lease=_lease(present=False))
    assert parked["parked"] is False
    assert parked["protected_required"] is False


# ---------------------------------------------------------------------------
# park / continue_with
# ---------------------------------------------------------------------------


def test_unprotected_unit_is_never_parked():
    parked = ps.park(_cpu(), contamination=_cont("HEAVY"), lease=_lease(present=False))
    assert parked["parked"] is False
    assert parked["wake_condition"] is None
    named = ps.park(
        _cpu(id="future.protected-window.staged-qualification"),
        contamination=_cont("HEAVY"),
        lease=_lease(present=False),
    )
    assert named["parked"] is False


def test_blocked_protected_unit_is_parked_with_wake_condition():
    parked = ps.park(_gpu(), contamination=_cont("HEAVY"), lease=_lease(present=False))
    assert parked["parked"] is True
    assert parked["verdict"] == "BLOCKED_ON_PROTECTED_WINDOW"
    assert parked["scheduler_capable"] is True
    assert parked["unit"]["status"] == "blocked"
    assert parked["unit"]["classification"] == "SLEEPING"
    assert parked["wake_condition"]["all_of"] == list(ps.WAKE_ALL_OF)


def test_runnable_protected_unit_is_not_parked():
    parked = ps.park(
        _gpu(),
        contamination=_cont("QUIESCENT"),
        lease=_lease(present=True),
    )
    assert parked["parked"] is False
    assert parked["verdict"] == "RUNNABLE"


def test_continue_with_is_cpu_work_and_not_protected():
    cont = ps.continue_with()
    assert cont["kind"] == "CONTINUE"
    assert cont["gpu_authority"] is False
    assert isinstance(cont["units"], list)
    for unit in cont["units"]:
        assert unit["resource_class"] != "GPU_EXCLUSIVE"
        assert unit.get("requires_quiescence") is not True
        assert unit.get("id") != ps.PROBE_UNIT["id"]
    if cont["failed_closed"]:
        assert cont["n"] == 0
        assert "refused" in (cont["reason"] or "").lower() or "empty" in (cont["reason"] or "").lower()
    # Either way the scheduler coped; an empty continue is not a skip.


def test_continue_with_excluding_drops_the_named_id():
    live = ps.continue_with()
    if not live["units"]:
        dropped = ps.continue_with(excluding={"does-not-exist"})
        assert dropped["n"] == live["n"]
        return
    ident = str(live["units"][0]["id"])
    dropped = ps.continue_with(excluding={ident})
    assert ident not in [u["id"] for u in dropped["units"]]


# ---------------------------------------------------------------------------
# live path — cope with either window state; capability stays true
# ---------------------------------------------------------------------------


def test_live_inspect_contamination_is_a_real_class():
    cont = ps.inspect_contamination()
    assert cont["contamination_class"] in ps.C.CONTAMINATION_CLASSES
    assert cont["live"] is True
    assert cont["injected_input"] is False
    assert cont["coerced"] is False
    if not cont.get("receipt_found"):
        assert cont["contamination_class"] == "UNKNOWN"


def test_live_inspect_lease_does_not_take_or_fabricate():
    before = _mtime_snapshot()
    worktree_locks = REPO / ".hcli" / "locks"
    existed = worktree_locks.exists()
    lease = ps.inspect_lease()
    after = _mtime_snapshot()
    assert lease["kind"] == "READ"
    assert lease["acquired"] is False
    assert lease["would_flock"] is False
    assert lease["fabricated"] is False
    assert lease["touched_lock_file"] is False
    assert lease["live"] is True
    assert lease["gpu_authority"] is False
    if lease["present"]:
        assert lease["holders"]["pids"], "present requires a proven holder"
    assert after == before
    if not existed:
        assert not worktree_locks.exists(), "inspect_lease must not create .hcli/locks"


def test_live_decide_matches_live_inputs_and_stays_capable():
    d = ps.decide(_gpu())
    cont = ps.inspect_contamination()
    lease = ps.inspect_lease()
    available = (
        cont["contamination_class"] == "QUIESCENT" and bool(lease["present"])
    )
    if available:
        assert d["verdict"] == "RUNNABLE"
    else:
        assert d["verdict"] == "BLOCKED_ON_PROTECTED_WINDOW"
    assert d["scheduler_capable"] is True
    assert d["window_available"] is available
    assert d["gpu_authority"] is False


def test_capability_report_separates_the_two_facts():
    report = ps.capability_report()
    assert "PROTECTED_SCHEDULER_CAPABLE" in report
    assert "PROTECTED_WINDOW_AVAILABLE" in report
    assert report["PROTECTED_SCHEDULER_CAPABLE"] is True
    cont = ps.inspect_contamination()
    lease = ps.inspect_lease()
    expected_available = (
        cont["contamination_class"] == "QUIESCENT" and bool(lease["present"])
    )
    assert report["PROTECTED_WINDOW_AVAILABLE"] is expected_available
    assert report["PROTECTED_WINDOW_AVAILABLE"] is not report["PROTECTED_SCHEDULER_CAPABLE"] or expected_available
    # The category error: capability is not a copy of availability.
    if not expected_available:
        assert report["PROTECTED_SCHEDULER_CAPABLE"] is True
        assert report["live_verdict"] == "BLOCKED_ON_PROTECTED_WINDOW"
        assert report["did_not_mark_incapable_because_window_unavailable"] is True
    assert report["did_not_fabricate_lease"] is True
    assert report["did_not_flock"] is True
    plan = report["odyssey_launch_read_plan"]
    assert plan["do_not_edit"] == "tools/future/odyssey_launch.py"
    assert "PROTECTED_SCHEDULER_CAPABLE" in plan["should_read"]
    assert "PROTECTED_WINDOW_AVAILABLE" in plan["should_read"]
    assert plan["honest_values_today"]["PROTECTED_SCHEDULER_CAPABLE"] is True
    assert plan["honest_values_today"]["gpu_authority"] is False


def test_capability_is_not_derived_from_availability():
    report = ps.capability_report()
    # Directly: the field is not a restatement of the other field's negation or copy
    # except on the one host state where both happen to be true.
    if report["PROTECTED_WINDOW_AVAILABLE"] is False:
        assert report["PROTECTED_SCHEDULER_CAPABLE"] is True


# ---------------------------------------------------------------------------
# never fabricate, never flock, never contend
# ---------------------------------------------------------------------------


def test_refuse_flock_and_acquire_lease_actually_fire():
    with pytest.raises(qp.AuthorityBoundaryError, match="flock"):
        ps.refuse_flock()
    with pytest.raises(qp.AuthorityBoundaryError, match="create_lease"):
        ps.refuse_create_lease()
    with pytest.raises(qp.AuthorityBoundaryError, match="start_benchmark"):
        ps.refuse_start_benchmark()
    with pytest.raises(ps.SchedulerRefused, match="flock"):
        ps.acquire_lease()
    with pytest.raises(ps.SchedulerRefused):
        ps.seize_lease()


def test_inspect_lease_does_not_open_a_lock_path(tmp_path, monkeypatch):
    """A lock file we can see must not be opened, created, or flocked by this module."""
    opened: list[str] = []
    real_open = open

    def wrapped_open(file, *args, **kwargs):
        path = str(file)
        opened.append(path)
        if "lock" in path.lower() and ("bench" in path.lower() or "protected" in path.lower() or ".hcli" in path):
            raise AssertionError(f"protected_scheduler opened a lock path: {path}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", wrapped_open)
    ps.inspect_lease()
    ps.inspect_contamination()
    for path in opened:
        lower = path.lower()
        assert not (
            "protected-accelerator-bench.lock" in lower
            or "qwen-protected-bench.lock" in lower
            or "hawking_protected_window.lease" in lower
        )


FORBIDDEN_CALLS = {
    "flock",
    "lockf",
    "kill",
    "killpg",
    "Popen",
    "run_protected_accelerator_benchmark",
    "_try_lock",
    "LOCK_EX",
    "SIGKILL",
    "SIGSTOP",
    "SIGTERM",
    "SingletonLease",
    "heal",
}
FORBIDDEN_IMPORTS = {
    "fcntl",
    "signal",
    "lab.lease",
    "hcli.agentos.protected_accelerator_benchmark",
    "hcli.agentos.native_mission_gate",
    "tools.future.odyssey_launch",
}
LOCK_NAME_FRAGMENTS = (
    "protected-accelerator-bench.lock",
    "qwen-protected-bench.lock",
    "hawking_protected_window.lease",
)
OPEN_FUNCS = {"open", "write_text", "write_bytes", "touch", "mkdir", "makedirs", "unlink"}


def _imported_modules(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            for alias in node.names:
                out.add(f"{node.module}.{alias.name}")
    return out


def _called_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            out.add(func.id)
        elif isinstance(func, ast.Attribute):
            out.add(func.attr)
    return out


def _const_strings(node: ast.AST) -> list[str]:
    out: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            out.append(child.value)
    return out


def test_module_does_not_import_fcntl_or_the_lease_runner():
    tree = ast.parse(Path(ps.__file__).read_text())
    imported = _imported_modules(tree)
    for name in FORBIDDEN_IMPORTS:
        assert name not in imported, f"forbidden import {name}"
    assert any("protected_window" in m for m in imported)
    assert any("qualification_pipeline" in m for m in imported)
    assert any("contamination" in m for m in imported)
    assert any("frontiers" in m for m in imported)
    assert any("workunit_species" in m for m in imported)


def test_module_does_not_call_flock_lockf_or_o_excl():
    src = Path(ps.__file__).read_text()
    tree = ast.parse(src)
    called = _called_names(tree)
    leaked = FORBIDDEN_CALLS & called
    assert not leaked, f"forbidden calls present: {sorted(leaked)}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"O_EXCL", "LOCK_EX", "LOCK_NB"}:
            raise AssertionError(f"forbidden attribute {node.attr}")
        if isinstance(node, ast.Name) and node.id in {"O_EXCL", "LOCK_EX"}:
            raise AssertionError(f"forbidden name {node.id}")
    for name in (
        "inspect_lease",
        "inspect_contamination",
        "decide",
        "park",
        "continue_with",
        "capability_report",
        "drive",
        "build",
        "acquire_lease",
        "recognize",
    ):
        fn = getattr(ps, name)
        names = set(fn.__code__.co_names)
        hit = FORBIDDEN_CALLS & names
        assert not hit, f"{name} names forbidden {sorted(hit)}"
        assert "fcntl" not in names
        assert "O_EXCL" not in names


def test_no_code_path_opens_or_locks_a_bench_lock_file():
    tree = ast.parse(Path(ps.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        attr = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        if attr not in OPEN_FUNCS and attr != "open":
            continue
        blobs = " ".join(_const_strings(node))
        for frag in LOCK_NAME_FRAGMENTS:
            assert frag not in blobs, f"{attr}() targets lock path fragment {frag!r}"
        if attr in {"open", "touch", "mkdir", "makedirs", "write_text", "write_bytes"}:
            lower = blobs.lower()
            assert ".hcli/locks" not in lower
            assert "bench.lock" not in lower


def test_source_does_not_import_fcntl_even_as_a_string_exec():
    src = Path(ps.__file__).read_text()
    assert "import fcntl" not in src
    assert "from fcntl" not in src
    assert "os.open(" not in src
    # Mentions of flock / O_EXCL in wake-never strings are not calls. The AST
    # walk above is the call/import authority.


# ---------------------------------------------------------------------------
# receipt / shape
# ---------------------------------------------------------------------------


def test_build_emits_sealed_static_only_receipt():
    out = ps.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "PROTECTED_SCHEDULER.json"
    assert doc["schema"] == ps.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["measurement_class"] == "STATIC_ONLY"
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["capability"]["PROTECTED_SCHEDULER_CAPABLE"] is True
    assert "PROTECTED_WINDOW_AVAILABLE" in doc["capability"]
    assert doc["did_not_fabricate_lease"] is True
    assert doc["did_not_flock"] is True
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["entry_point"].endswith("capability_report()")
    assert doc["resident_callable"]["frontier"] == "FT.GPU_KERNELS.ready-protected"
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    plan = doc["odyssey_launch_read_plan"]
    assert plan["do_not_edit"] == "tools/future/odyssey_launch.py"
    assert plan["honest_values_today"]["PROTECTED_SCHEDULER_CAPABLE"] is True
    _assert_no_hardware_claims(doc)


def test_workunit_the_scheduler_emits_is_not_gpu_exclusive():
    out = ps.build()
    doc = json.loads(out.read_text())
    wu = doc["workunit"]
    assert wu["resource_class"] == "STATIC_ANALYSIS"
    assert wu["requires_quiescence"] is False


def test_injected_unknown_contamination_is_unknown_not_quiescent():
    cont = ps.inspect_contamination(injected={"contamination_class": "SPARKLY"})
    assert cont["contamination_class"] == "UNKNOWN"
    d = ps.decide(_gpu(), contamination={"contamination_class": "SPARKLY"}, lease=_lease(present=True))
    assert d["verdict"] == "BLOCKED_ON_PROTECTED_WINDOW"
    assert d["scheduler_capable"] is True
