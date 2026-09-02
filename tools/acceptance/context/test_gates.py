"""Load-bearing acceptance tests for the eight HCLI context-family gates.

These tests call the catalog symbols. A receipt file is not the bar: if
``resolve`` stopped driving both root and worker, or stale evidence stopped
raising, these fail even if a prior receipt still says ACCEPTED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.acceptance.context.common import RECEIPT_DIR, REPO
from tools.acceptance.context.gates import GATE_IDS, run_all

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def bundle():
    return run_all()


def test_every_assigned_gate_has_a_verdict_and_a_receipt(bundle):
    summary = bundle["summary"]
    assert summary["criterion_altered"] is False
    assert set(summary["gates"]) == set(GATE_IDS)
    for gate in GATE_IDS:
        row = bundle["results"][gate]
        assert row["verdict"] in {"ACCEPTED", "BLOCKED"}, gate
        assert row.get("criterion_altered") is False
        quoted = (row.get("criterion") or {}).get("quoted") or ""
        assert quoted.strip(), f"{gate} did not quote the roadmap criterion"
        path = RECEIPT_DIR / f"{gate}.json"
        assert path.is_file(), path
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["gate"] == gate
        assert data["verdict"] == row["verdict"]
        assert data["criterion_altered"] is False
        if data["verdict"] == "BLOCKED":
            assert data.get("blocker"), f"{gate} BLOCKED without naming the missing input"
        if data["verdict"] == "ACCEPTED":
            assert data.get("invocations"), f"{gate} ACCEPTED without citing an invocation"


def test_summary_count_matches_verdicts(bundle):
    summary = bundle["summary"]
    accepted = [g for g, r in bundle["results"].items() if r["verdict"] == "ACCEPTED"]
    blocked = [g for g, r in bundle["results"].items() if r["verdict"] == "BLOCKED"]
    assert summary["accepted_count"] == len(accepted)
    assert summary["blocked_count"] == len(blocked)
    assert summary["accepted_count"] + summary["blocked_count"] == len(GATE_IDS)
    path = RECEIPT_DIR / "SUMMARY.json"
    assert path.is_file()


def test_authority_one_resolve_drives_root_and_worker(bundle):
    from hcli.context_budget import estimate_tokens, preflight, preflight_packet, resolve

    row = bundle["results"]["HCLI_CONTEXT_AUTHORITY_UNIFIED"]
    assert row["verdict"] == "ACCEPTED"
    budget = resolve(n_parallel=3)
    root = preflight(budget, estimate_tokens("root"), kind="root")
    worker = preflight_packet(budget, "worker", kind="worker")
    refuse = preflight(budget, int(budget.per_request_ctx) + 1, kind="root")
    assert id(root.budget) == id(budget) == id(worker.budget)
    assert root.kind == "root" and worker.kind == "worker"
    assert root.per_request_ctx == worker.per_request_ctx == budget.per_request_ctx
    assert root.ok and worker.ok
    assert not refuse.ok and refuse.shortfall > 0
    assert budget.source == root.budget.source == worker.budget.source


def test_focused_worker_packet_excludes_the_roadmap(bundle):
    from hcli.goal import compile_worker_context, refuse_goal_dump
    from hcli.workunit import WorkUnit

    row = bundle["results"]["HCLI_CONTEXT_FOCUSED_WORKUNITS"]
    assert row["verdict"] == "ACCEPTED"
    root = "CIVILIZATION ROADMAP ULTRAGOAL. " + ("keep-out-root-" * 40)
    parent = WorkUnit(id="p", role="research", description="parent neighborhood")
    child = WorkUnit(
        id="c",
        role="implement",
        description="one focused unit",
        dependencies=["p"],
    )
    unrelated = WorkUnit(id="u", role="research", description="SECRET_UNRELATED")
    packet = compile_worker_context(
        child,
        {
            "goal": root,
            "invariants": ["single writer"],
            "acceptance_criteria": ["verifier runs"],
        },
        phase="running",
        units={"p": parent, "c": child, "u": unrelated},
        steering=[],
        root_goal=root,
    )
    refuse_goal_dump(packet.prompt, root)
    assert packet.unit_id == "c"
    assert packet.phase == "running"
    assert root not in packet.prompt
    assert "SECRET_UNRELATED" not in packet.prompt
    assert packet.invariants
    assert packet.acceptance
    assert len(packet.prompt) < len(root)


def test_changed_evidence_refuses_stale_compiled_context(bundle):
    from hcli.goal import StaleEvidenceError, assert_evidence_fresh, compile_worker_context, identity_for_path
    from hcli.workunit import WorkUnit
    from tools.acceptance.context.common import rewrite
    import tempfile

    row = bundle["results"]["HCLI_CONTEXT_INVALIDATION"]
    assert row["verdict"] == "ACCEPTED"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "notes.txt"
        path.write_text("AAAA", encoding="utf-8")
        gathered = identity_for_path(path, root=Path(tmp))
        wu = WorkUnit(id="w", role="work", description="read notes.txt")
        packet = compile_worker_context(
            wu,
            {"goal": "short task about notes.txt must pass", "referenced_files": ["notes.txt"]},
            phase="running",
            units={wu.id: wu},
            steering=[],
            workspace=tmp,
            evidence=[gathered],
        )
        assert_evidence_fresh(packet.evidence, tmp)
        rewrite(path, "BBBB")
        with pytest.raises(StaleEvidenceError, match="stale evidence refused"):
            assert_evidence_fresh(packet.evidence, tmp)


def test_status_physical_is_judged_against_d9_not_a_thinner_contract(bundle):
    row = bundle["results"]["HCLI_STATUS_PHYSICAL"]
    missing = (row.get("d9") or {}).get("missing") or []
    if missing:
        assert row["verdict"] == "BLOCKED"
        assert "D.9" in (row.get("blocker") or "")
        # Load-bearing: repair / persistence digest / resource / lease must
        # not be silently dropped from the bar.
        joined = " ".join(missing)
        assert "REPAIR" in joined
    else:
        assert row["verdict"] == "ACCEPTED"
    rendered = row.get("rendered") or ""
    assert "mission " in rendered
    from hcli.commands import format_status

    text = format_status({"mission_id": "x", "phase": "idle", "goal": "g"})
    assert "mission " in text


def test_mixed_max_does_not_accept_cpu_only_as_mixed(bundle):
    row = bundle["results"]["HCLI_MIXED_MAX"]
    assert row["verdict"] == "BLOCKED"
    assert row.get("blocker")
    assert row["checks"]["did_not_substitute_cpu_only_as_mixed"] is True
    assert row["checks"]["heterogeneous_throughput_measured"] is False
    from hcli.max_policy import grok_pool_snapshot

    snap = grok_pool_snapshot(str(REPO))
    assert "active" in snap and "admitted" in snap


def test_backend_failure_isolation_holds_and_terminate_pid_is_called(bundle):
    row = bundle["results"]["BACKEND_FAILURE_ISOLATION"]
    assert row["verdict"] == "ACCEPTED"
    assert row["checks"]["injected_backend_failure_does_not_stop_independent_units"]
    assert row["checks"]["terminate_pid_invoked_on_owned_child"]
    assert row["terminate_pid"]["gone"] is True
    from hcli.resources import classify_failure

    assert classify_failure({"http_status": 429}).kind == "RATE_LIMIT"
    assert classify_failure({"http_status": 503}).kind == "TRANSIENT_BACKEND"


def test_self_supplement_full_chain_and_unverified_refusal(bundle):
    row = bundle["results"]["HCLI_SELF_SUPPLEMENT"]
    assert row["verdict"] == "ACCEPTED"
    hops = row["hops"]
    assert hops["child_dependencies"] == ["parent"]
    assert hops["unverified_denied"] == []
    assert hops["ghost_absent"] is True
    from hcli.agentos.resident import admit_evidence_children

    denied = admit_evidence_children(
        None,
        {"unit_id": "x", "accepted": True, "validation": {"ok": True}, "child_workunits": []},
    )
    assert denied == []


def test_self_optimization_bootstrap_is_blocked_without_weakening(bundle):
    row = bundle["results"]["HCLI_SELF_OPTIMIZATION_BOOTSTRAP"]
    assert row["verdict"] == "BLOCKED"
    blocker = row.get("blocker") or ""
    assert "prewritten" in blocker.lower() or "D.11" in blocker
    assert row["checks"]["run_autonomy_gate_not_invoked"] is True
    assert row["checks"]["two_linked_iterations_with_independent_verification"] is False
    assert row.get("invocations") == []
