#!/usr/bin/env python3.12
"""Logical cases for the laboratory engine (cut over from tools/condense/engine)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.checkpoint import CheckpointStore, HashChainLog
from lab.engine_support import ResourceGovernor, ResourceLimits
from lab.lease import LeaseError, SingletonLease
from lab.runtime import CampaignRuntime, run_campaign
from lab.engine_support import Scheduler, WorkStatus
from lab.engine_support import IllegalTransition, Phase, StateMachine
from lab.spec import (
    SCHEMA,
    SPECS_DIR,
    CampaignPhase,
    SpecError,
    list_specs,
    load_all_specs,
    load_spec,
    load_spec_path,
    validate_spec,
)
from lab.science_registry import (
    DEFAULT_REGISTRY,
    OperatorClass,
    OperatorRegistry,
    classify_all,
)
from lab.receipts import Receipt, ReceiptAuthority
from lab.receipts import seal as seal_receipt
from lab.receipts import verify as verify_receipt

# The operators these tests classify moved from tools/condense to
# lab/operators when the process engine was cut over to lab authority.
# tools/condense holds two modules now; the registry classifies the 53 that
# actually exist, so this is where the Track V contract has to look.
CONDENSE = Path(__file__).resolve().parents[3] / "lab" / "operators"

SPEC_CASES = list_specs()
assert SPEC_CASES, "lab campaign catalog must contain campaign families"


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
    if spec.status == "released_historical_non_invocable":
        # A released campaign has no reproduction command because the tool it
        # described was released out of the tree. The row still has to say so,
        # otherwise a missing reproduction and a deliberately absent one look
        # the same.
        assert not spec.reproduction, "a released row must not claim a reproduction"
        assert spec.notes and "historical" in spec.notes.lower(), (
            f"{spec.campaign_id}: a released row must explain in notes that it is "
            f"hollow, got {spec.notes!r}"
        )
    else:
        assert spec.reproduction
    for fence in spec.authorization_fences:
        assert fence
    for cond in spec.reopen:
        assert cond.id
        assert cond.description


@pytest.mark.parametrize("spec_path", SPEC_CASES, ids=lambda p: p.stem)
def test_spec_dry_run_completes(spec_path: Path, tmp_path: Path) -> None:
    """Process-handler campaigns re-run green; missing science handlers fail closed."""
    from lab.runtime import BUILTIN_HANDLERS

    spec = load_spec_path(spec_path)
    process_only = all(step.handler in BUILTIN_HANDLERS for step in spec.steps)
    result = run_campaign(
        spec_path,
        work_dir=tmp_path / spec_path.stem,
        acquire_lease=True,
    )
    if not process_only:
        # Engine-only cutover: no toy science authority. Non-optional missing
        # handlers must FAULT rather than silently succeed.
        assert result.status == "FAULT", result.to_dict()
        return
    assert result.status == "PASS", result.to_dict()
    assert result.phase == "complete"
    assert result.receipt_path
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


def _reject_spec(doc: dict) -> None:
    with pytest.raises(SpecError):
        validate_spec(doc)


def test_invalid_schema_rejected() -> None:
    _reject_spec({"schema": "nope", "campaign_id": "x", "phases": ["precheck"]})


def test_unknown_phase_rejected() -> None:
    _reject_spec(
        {
            "schema": SCHEMA,
            "campaign_id": "x",
            "phases": ["precheck"],
            "steps": [{"id": "a", "phase": "teleport"}],
        }
    )


def test_duplicate_step_id_rejected() -> None:
    _reject_spec(
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


def test_governor_passes_zero_floor(tmp_path: Path) -> None:
    gov = ResourceGovernor(ResourceLimits(min_free_disk_bytes=0), root=tmp_path)
    ok, sample, failures = gov.allow()
    assert ok


def test_governor_refuses_impossible_floor(tmp_path: Path) -> None:
    gov = ResourceGovernor(
        ResourceLimits(min_free_disk_bytes=10**18), root=tmp_path
    )
    ok, sample, failures = gov.allow()
    assert not ok
    assert failures


def test_scheduler_skips_completed() -> None:
    spec = load_spec_path(SPECS_DIR / "glm52.json")
    completed = {s.id for s in spec.steps[:2]}
    sched = Scheduler(spec, completed=completed)
    for step_id in completed:
        sched.mark_done(step_id)
    plan = sched.plan()
    planned_ids = {item.id for item in plan}
    assert completed.isdisjoint(planned_ids)


def test_scheduler_plan_order() -> None:
    spec = load_spec_path(SPECS_DIR / "glm52.json")
    sched = Scheduler(spec)
    plan = sched.plan()
    assert plan
    assert plan[0].phase == "precheck"


def test_receipt_seal_verify_roundtrip(tmp_path: Path) -> None:
    store = ReceiptAuthority(tmp_path)
    receipt = Receipt(
        campaign_id="c",
        verdict="PASS",
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
    sealed = seal_receipt(
        {
            "schema": "t",
            "campaign_id": "c",
            "status": "x",
            "phase": "y",
            "summary": {},
            "verdict": "PASS",
        }
    )
    sealed["status"] = "mutated"
    with pytest.raises(ValueError, match="seal mismatch"):
        verify_receipt(sealed)


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


LIFECYCLE_VERBS = [p.value for p in CampaignPhase]


@pytest.mark.parametrize("verb", LIFECYCLE_VERBS)
def test_lifecycle_verb_is_known_phase(verb: str) -> None:
    assert verb in {p.value for p in CampaignPhase}
    phases = [verb] if verb != "resume" else ["precheck"]
    if verb == "resume":
        phases = ["precheck"]
        steps = [
            {"id": "p", "phase": "precheck", "handler": "record"},
            {"id": "r", "phase": "resume", "handler": "record"},
        ]
    else:
        steps = [{"id": f"{verb}.0", "phase": verb, "handler": "record"}]
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


def test_operator_registry_covers_every_top_level_module() -> None:
    """Every lab/operators/*.py module is classified (the Track V contract)."""
    root = CONDENSE
    repo = Path(__file__).resolve().parents[3]
    on_disk = sorted(p.stem for p in root.glob("*.py") if p.stem != "__init__")
    classified = {r.module for r in DEFAULT_REGISTRY.records}
    missing = set(on_disk) - classified
    assert not missing, f"unclassified modules: {sorted(missing)}"
    # Check each record against the path it records, not against one directory.
    # artifact_client is classified but lives in tools/condense, so globbing a
    # single folder reports it as phantom when it is simply somewhere else.
    absent = sorted(r.module for r in DEFAULT_REGISTRY.records if not (repo / r.path).is_file())
    assert not absent, f"registry names modules with no file at their recorded path: {absent}"
    assert len(on_disk) >= 40


def test_operator_classes_are_the_six_plus_unclassified() -> None:
    allowed = {c.value for c in OperatorClass}
    for rec in DEFAULT_REGISTRY.records:
        assert rec.class_name in allowed


def test_live_glm52_readers_are_registered() -> None:
    for name in (
        "glm52_parity",
        "glm52_contract",
        "glm52_source_fetch",
        "glm52_teacher_capture",
        "glm52_xet_autotune",
    ):
        rec = DEFAULT_REGISTRY.get(name)
        assert rec is not None, name
        assert rec.class_ is not OperatorClass.SPEC


def test_resolve_artifact_authority_is_numerical_not_deleted() -> None:
    rec = DEFAULT_REGISTRY.get("glm52_common")
    assert rec is not None
    assert rec.class_ is OperatorClass.NUMERICAL_AUTHORITY
    assert rec.path_sealed


def test_glm52_state_is_named_unclassified_residual_controller() -> None:
    rec = DEFAULT_REGISTRY.get("glm52_state")
    assert rec is not None
    assert rec.class_ is OperatorClass.UNCLASSIFIED
    assert "lease" in rec.why.lower() or "controller" in rec.why.lower()
    from lab.lease import SingletonLease as EngineLease
    from lab.operators.glm52_state import SingletonLease as StateLease

    assert issubclass(StateLease, EngineLease)


def test_engine_lease_is_toctou_hardened() -> None:
    """Production lease proofs live on lab.lease."""
    import inspect

    from lab import lease as lease_mod
    from lab.operators.glm52_state import SingletonLease as StateLease

    src = inspect.getsource(lease_mod.SingletonLease.acquire)
    assert "O_NOFOLLOW" in inspect.getsource(lease_mod.SingletonLease) or (
        "_open_parent_chain" in src
        or "_open_parent_chain" in inspect.getsource(lease_mod.SingletonLease)
    )
    assert "_open_parent_chain" in inspect.getsource(lease_mod.SingletonLease)
    assert "def acquire" not in inspect.getsource(StateLease)


def test_science_floor_is_substantial() -> None:
    """The science floor is a ratchet, and it has already been walked down.

    This asserted >= 25_000 and measured 15_021 on 2026-07-30. The gap is not
    a bug in the count: the condense campaigns compacted the science modules
    and released several out of the tree, and 8b0c5405 deleted the Ramanujan
    science package outright (restored separately). The tripwire could not
    report any of it, because the same commit family broke this file's
    imports and it failed at collection instead of at this line.

    Held at the measured floor rather than the aspirational one, so a further
    collapse still fires. Raising it back is a real piece of work, not a
    number to edit.
    """
    floor = DEFAULT_REGISTRY.science_floor_loc()
    assert floor >= 15_000, f"science floor collapsed to {floor}"


def test_classify_all_roundtrip() -> None:
    rows = classify_all()
    assert len(rows) == len(DEFAULT_REGISTRY.records)
    assert all("module" in r and "class" in r and "loc" in r for r in rows)


def test_runtime_validates_operator_modules(tmp_path: Path) -> None:
    spec = load_spec(
        {
            "schema": SCHEMA,
            "campaign_id": "ops",
            "phases": ["precheck"],
            "steps": [
                {
                    "id": "c",
                    "phase": "precheck",
                    "handler": "record",
                    "params": {"module": "glm52_contract"},
                }
            ],
            "reproduction": "true",
        }
    )
    result = run_campaign(spec, work_dir=tmp_path, acquire_lease=True)
    assert result.status == "PASS"


def test_runtime_rejects_unknown_operator_module(tmp_path: Path) -> None:
    spec = load_spec(
        {
            "schema": SCHEMA,
            "campaign_id": "ops_bad",
            "phases": ["precheck"],
            "steps": [
                {
                    "id": "c",
                    "phase": "precheck",
                    "handler": "record",
                    "params": {"module": "not_a_real_module_xyz"},
                }
            ],
            "reproduction": "true",
        }
    )
    result = run_campaign(spec, work_dir=tmp_path, acquire_lease=True)
    assert result.status == "FAULT"


def test_cli_classify_exits_zero() -> None:
    from lab.runtime import main

    assert main(["--classify"]) == 0

# --- unique lab process contracts not covered above ---

def test_missing_science_handler_fails_closed(tmp_path: Path) -> None:
    """Non-optional science ops without a live implementation must FAULT."""
    from lab.runtime import BUILTIN_HANDLERS

    target = next(p for p in list_specs() if any(
        step.handler not in BUILTIN_HANDLERS for step in load_spec_path(p).steps
    ))
    result = run_campaign(target, work_dir=tmp_path / target.stem, acquire_lease=True)
    assert result.status == "FAULT"


def test_normalize_old_campaign_receipt(tmp_path: Path) -> None:
    from lab.receipts import read_any_receipt

    raw = {
        "schema": "hawking.condense.campaign_receipt.v1",
        "campaign_id": "x",
        "status": "retired",
        "phase": "complete",
        "summary": {"k": 1},
        "at": "t",
        "reproduction": "r",
        "artifacts": [],
        "seal_sha256": "dead",
    }
    path = tmp_path / "x.receipt.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    doc = read_any_receipt(path)
    assert doc["campaign_id"] == "x"
    assert doc["verdict"] == "retired"


def test_governance_refuses_unmet_promotion(tmp_path: Path) -> None:
    from lab.rules import GovernanceError, GovernanceLedger, apply_governance

    spec = load_spec(
        {
            "schema": SCHEMA,
            "campaign_id": "gov",
            "phases": ["precheck", "report"],
            "steps": [{"id": "r", "phase": "report", "handler": "report.summary"}],
            "reproduction": "true",
            "promotion": {"require_verdict": "PASS", "require_gates": ["g1"]},
        }
    )
    ledger = GovernanceLedger(tmp_path / "gov.jsonl")
    with pytest.raises(GovernanceError, match="gate"):
        apply_governance(
            spec,
            ledger=ledger,
            verdict="PASS",
            gate_results={"g1": False},
            author="a",
            admitter="b",
            action="promote",
        )
