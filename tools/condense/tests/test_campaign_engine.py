#!/usr/bin/env python3.12
"""Table-driven tests for the single campaign/experiment engine (lane A1).

Logical cases formerly scattered across per-campaign controller suites are
expressed here against ExperimentSpec + runtime, lease, checkpoint, and receipt
surfaces. Campaign-specific algorithmic tests that cannot be expressed as a
spec matrix remain in their original files (and may import archived bodies).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CONDENSE = Path(__file__).resolve().parents[1]
import sys

if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

from engine.checkpoint import CheckpointStore, HashChainLog  # noqa: E402
from engine.governor import ResourceGovernor, ResourceLimits  # noqa: E402
from engine.lease import LeaseError, SingletonLease  # noqa: E402
from engine.receipts import Receipt, ReceiptStore, seal_receipt, verify_receipt  # noqa: E402
from engine.runtime import CampaignRuntime, run_campaign  # noqa: E402
from engine.scheduler import Scheduler, WorkStatus  # noqa: E402
from engine.spec import (  # noqa: E402
    SCHEMA,
    SPECS_DIR,
    CampaignPhase,
    SpecError,
    load_all_specs,
    load_spec,
    load_spec_path,
    validate_spec,
)
from engine.state_machine import IllegalTransition, Phase, StateMachine  # noqa: E402

# ---------------------------------------------------------------------------
# Spec matrix — every retired/live campaign loads and validates
# ---------------------------------------------------------------------------

SPEC_CASES = sorted(SPECS_DIR.glob("*.json"))
assert SPEC_CASES, "engine/specs must contain campaign JSON specs"


@pytest.mark.parametrize("spec_path", SPEC_CASES, ids=lambda p: p.stem)
def test_spec_loads_and_validates(spec_path: Path) -> None:
    spec = load_spec_path(spec_path)
    assert spec.schema == SCHEMA
    assert spec.campaign_id
    assert spec_path.stem in {spec.campaign_id, spec.family} or spec.campaign_id.startswith(
        spec_path.stem
    )
    assert spec.phases
    assert all(s.phase in set(spec.phases) | {"resume"} for s in spec.steps)
    assert spec.reproduction
    # Authorization fences must not be claimed open in the spec itself.
    for fence in spec.authorization_fences:
        assert fence  # named
    # Reopen conditions are structured when present.
    for cond in spec.reopen:
        assert cond.id
        assert cond.description


@pytest.mark.parametrize("spec_path", SPEC_CASES, ids=lambda p: p.stem)
def test_spec_dry_run_completes(spec_path: Path, tmp_path: Path) -> None:
    """Every campaign is re-runnable from its spec (record handlers only)."""
    result = run_campaign(
        spec_path,
        work_dir=tmp_path / spec_path.stem,
        acquire_lease=True,
    )
    assert result.status == "PASS", result.to_dict()
    assert result.phase == "complete"
    assert result.receipt_path
    # Resume is idempotent: second run skips completed steps.
    result2 = run_campaign(
        spec_path,
        work_dir=tmp_path / spec_path.stem,
        acquire_lease=True,
    )
    assert result2.status == "PASS"
    assert set(result.completed_steps) <= set(result2.completed_steps)


def test_load_all_specs_covers_historical_families() -> None:
    specs = load_all_specs()
    families = {s.family for s in specs}
    for needed in (
        "glm52",
        "kimi_k26",
        "qwen",
        "gptoss",
        "deepseek_v4",
        "second_light",
        "gravity_frontier",
    ):
        assert needed in families, f"missing family {needed}"


def test_invalid_schema_rejected() -> None:
    with pytest.raises(SpecError):
        validate_spec({"schema": "nope", "campaign_id": "x", "phases": ["precheck"]})


def test_unknown_phase_rejected() -> None:
    with pytest.raises(SpecError):
        validate_spec(
            {
                "schema": SCHEMA,
                "campaign_id": "x",
                "phases": ["precheck"],
                "steps": [{"id": "a", "phase": "teleport"}],
            }
        )


def test_duplicate_step_id_rejected() -> None:
    with pytest.raises(SpecError):
        validate_spec(
            {
                "schema": SCHEMA,
                "campaign_id": "x",
                "phases": ["precheck"],
                "steps": [
                    {"id": "a", "phase": "precheck"},
                    {"id": "a", "phase": "precheck"},
                ],
            }
        )


# ---------------------------------------------------------------------------
# State machine — one FSM, illegal transitions, one-use claims
# ---------------------------------------------------------------------------

PHASE_FORWARD = [
    ("idle", "precheck"),
    ("precheck", "measure"),
    ("measure", "allocate"),
    ("allocate", "pack"),
    ("pack", "seal"),
    ("seal", "report"),
    ("report", "complete"),
]


@pytest.mark.parametrize("source,target", PHASE_FORWARD)
def test_forward_transition(source: str, target: str) -> None:
    sm = StateMachine(campaign_id="t")
    sm.phase = Phase(source)
    sm.transition(target, claim_id=f"{source}->{target}")
    assert sm.phase == Phase(target)


def test_illegal_transition_raises() -> None:
    sm = StateMachine(campaign_id="t")
    with pytest.raises(IllegalTransition):
        sm.transition(Phase.SEAL, claim_id="bad")


def test_one_use_claim() -> None:
    sm = StateMachine(campaign_id="t")
    sm.transition(Phase.PRECHECK, claim_id="once")
    with pytest.raises(IllegalTransition):
        sm.transition(Phase.MEASURE, claim_id="once")


def test_fault_and_resume() -> None:
    sm = StateMachine(campaign_id="t")
    sm.transition(Phase.PRECHECK, claim_id="p")
    sm.transition(Phase.FAULT, claim_id="f", detail={"reason": "boom"})
    assert sm.fault_reason == "boom"
    sm.transition(Phase.RESUME, claim_id="r")
    sm.transition(Phase.PRECHECK, claim_id="p2")


def test_snapshot_roundtrip() -> None:
    sm = StateMachine(campaign_id="t")
    sm.transition(Phase.PRECHECK, claim_id="p")
    sm.mark_step("step-a")
    restored = StateMachine.from_snapshot(sm.snapshot())
    assert restored.phase == Phase.PRECHECK
    assert restored.is_step_done("step-a")
    assert "p" in restored.claims


# ---------------------------------------------------------------------------
# Lease — exclusive, process-local double-hold refused, release frees
# ---------------------------------------------------------------------------


def test_lease_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "c.lease"
    a = SingletonLease(path, campaign_id="c", owner="a")
    b = SingletonLease(path, campaign_id="c", owner="b")
    a.acquire()
    try:
        with pytest.raises(LeaseError):
            b.acquire()
    finally:
        a.release()
    b.acquire()
    b.release()


def test_lease_process_double_hold(tmp_path: Path) -> None:
    path = tmp_path / "c.lease"
    a = SingletonLease(path, campaign_id="c", owner="a")
    a.acquire()
    try:
        c = SingletonLease(path, campaign_id="c", owner="a2")
        with pytest.raises(LeaseError):
            c.acquire()
    finally:
        a.release()


def test_lease_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "c.lease"
    with SingletonLease(path, campaign_id="c") as lease:
        assert lease.held
        owner = lease.read_owner()
        assert owner is not None
        assert owner["campaign_id"] == "c"
    assert not lease.held


def test_assert_held(tmp_path: Path) -> None:
    lease = SingletonLease(tmp_path / "c.lease", campaign_id="c")
    with pytest.raises(LeaseError):
        lease.assert_held()


# ---------------------------------------------------------------------------
# Checkpoint / hash chain — crash resume, seal, split-brain refuse
# ---------------------------------------------------------------------------


def test_hash_chain_append_and_verify(tmp_path: Path) -> None:
    log = HashChainLog(tmp_path / "events.jsonl")
    e1 = log.append({"event": "a", "campaign_id": "c"})
    e2 = log.append({"event": "b", "campaign_id": "c"})
    assert e1["event_sha256"] == e2["prev_sha256"]
    log2 = HashChainLog(tmp_path / "events.jsonl")
    log2.load()
    assert log2.count == 2
    assert log2.head == e2["event_sha256"]


def test_checkpoint_save_load_resume(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, campaign_id="c")
    store.record("start", {})
    sealed = store.save({"phase": "pack", "completed_steps": ["a"], "claims": ["c1"]})
    assert "seal_sha256" in sealed
    loaded = store.load()
    assert loaded is not None
    assert loaded["state"]["phase"] == "pack"
    state = store.resume_state()
    assert state["completed_steps"] == ["a"]


def test_checkpoint_tamper_rejected(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, campaign_id="c")
    store.save({"phase": "idle", "completed_steps": [], "claims": []})
    path = store.checkpoint_path
    raw = json.loads(path.read_text())
    raw["state"]["phase"] = "seal"
    path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match="seal mismatch"):
        store.load()


def test_resume_without_checkpoint(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, campaign_id="c")
    state = store.resume_state()
    assert state["phase"] == "idle"
    assert state["completed_steps"] == []


# ---------------------------------------------------------------------------
# Governor
# ---------------------------------------------------------------------------


def test_governor_passes_zero_floor(tmp_path: Path) -> None:
    gov = ResourceGovernor(ResourceLimits(min_free_disk_bytes=0), root=tmp_path)
    ok, sample, failures = gov.allow()
    assert ok
    assert not failures
    assert sample.free_disk_bytes >= 0


def test_governor_refuses_impossible_floor(tmp_path: Path) -> None:
    gov = ResourceGovernor(
        ResourceLimits(min_free_disk_bytes=10**18),
        root=tmp_path,
    )
    ok, _, failures = gov.allow()
    assert not ok
    assert failures


# ---------------------------------------------------------------------------
# Scheduler — resume skips completed idempotent work
# ---------------------------------------------------------------------------


def test_scheduler_skips_completed() -> None:
    spec = load_spec(
        {
            "schema": SCHEMA,
            "campaign_id": "s",
            "phases": ["precheck", "measure"],
            "steps": [
                {"id": "a", "phase": "precheck", "handler": "record"},
                {"id": "b", "phase": "measure", "handler": "record", "inputs": ["a"]},
            ],
        }
    )
    sched = Scheduler(spec, completed={"a"})
    assert sched.items[0].status is WorkStatus.DONE
    nxt = sched.next_ready()
    assert nxt is not None
    assert nxt.id == "b"


def test_scheduler_plan_order() -> None:
    spec = load_spec_path(SPECS_DIR / "glm52.json")
    sched = Scheduler(spec)
    plan = sched.plan()
    assert plan
    assert plan[0].phase == "precheck"


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------


def test_receipt_seal_verify_roundtrip(tmp_path: Path) -> None:
    store = ReceiptStore(tmp_path)
    receipt = Receipt(
        campaign_id="c",
        status="retired",
        phase="complete",
        summary={"ok": True},
        reproduction="echo hi",
    )
    path = store.write(receipt)
    assert path.is_file()
    loaded = store.read("c")
    verify_receipt(loaded)
    assert loaded["campaign_id"] == "c"


def test_receipt_tamper_rejected() -> None:
    sealed = seal_receipt({"schema": "t", "campaign_id": "c", "status": "x", "phase": "y", "summary": {}})
    sealed["status"] = "mutated"
    with pytest.raises(ValueError, match="seal mismatch"):
        verify_receipt(sealed)


# ---------------------------------------------------------------------------
# Runtime integration — fences closed, fault path, custom handler
# ---------------------------------------------------------------------------


def test_runtime_fence_precheck(tmp_path: Path) -> None:
    spec = load_spec(
        {
            "schema": SCHEMA,
            "campaign_id": "fence",
            "phases": ["precheck", "report"],
            "steps": [
                {"id": "f", "phase": "precheck", "handler": "precheck.fences"},
                {"id": "r", "phase": "report", "handler": "report.summary"},
            ],
            "authorization_fences": ["ODYSSEY_LAUNCH_AUTHORIZED"],
            "reproduction": "true",
        }
    )
    with CampaignRuntime(spec, work_dir=tmp_path, acquire_lease=True) as rt:
        result = rt.run()
    assert result.status == "PASS"


def test_runtime_fault_on_handler_error(tmp_path: Path) -> None:
    def boom(runtime, params):
        raise RuntimeError("injected")

    spec = load_spec(
        {
            "schema": SCHEMA,
            "campaign_id": "faulty",
            "phases": ["precheck"],
            "steps": [{"id": "x", "phase": "precheck", "handler": "boom"}],
            "reproduction": "true",
        }
    )
    with CampaignRuntime(
        spec, work_dir=tmp_path, handlers={"boom": boom}, acquire_lease=True
    ) as rt:
        result = rt.run()
    assert result.status == "FAULT"
    assert result.phase == "fault"


def test_runtime_status_surface(tmp_path: Path) -> None:
    spec = load_spec_path(SPECS_DIR / "deepseek_v4.json")
    with CampaignRuntime(spec, work_dir=tmp_path, acquire_lease=True) as rt:
        status = rt.status()
    assert status["campaign_id"] == "deepseek_v4"
    assert status["reproduction"]
    assert status["reopen"]


# ---------------------------------------------------------------------------
# Lifecycle verb matrix — one implementation covers all phase names
# ---------------------------------------------------------------------------

LIFECYCLE_VERBS = [p.value for p in CampaignPhase]


@pytest.mark.parametrize("verb", LIFECYCLE_VERBS)
def test_lifecycle_verb_is_known_phase(verb: str) -> None:
    assert verb in {p.value for p in CampaignPhase}
    # Spec accepts a single-phase campaign for each verb (resume included).
    phases = [verb] if verb != "resume" else ["precheck"]
    steps = (
        [{"id": "r", "phase": "resume", "handler": "record"}]
        if verb == "resume"
        else [{"id": f"{verb}.0", "phase": verb, "handler": "record"}]
    )
    if verb == "resume":
        # resume steps are allowed even if resume is not in phases list
        phases = ["precheck"]
        steps = [
            {"id": "p", "phase": "precheck", "handler": "record"},
            {"id": "r", "phase": "resume", "handler": "record"},
        ]
    spec = validate_spec(
        {
            "schema": SCHEMA,
            "campaign_id": f"verb_{verb}",
            "phases": phases,
            "steps": steps,
            "reproduction": "true",
        }
    )
    assert spec.campaign_id.startswith("verb_")
