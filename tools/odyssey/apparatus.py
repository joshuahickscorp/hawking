#!/usr/bin/env python3.12
"""Orchestrate the Odyssey trainer apparatus on FIXTURE toy models and write receipts.

Produces:
  - odyssey/ODYSSEY_T0_EXECUTABLE_RECEIPT.json
  - odyssey/ODYSSEY_TRAINER_READINESS.json
  - odyssey/ODYSSEY_HEAVY_PREREQUISITES.json

Does not flip the launch fence. Does not train on Math-Preserve. Does not claim
that anything real has been learned.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.odyssey._paths import ODYSSEY, ROOT
from tools.odyssey.checkpoints import CheckpointError, CheckpointStore, content_id
from tools.odyssey.objectives import UnregisteredObjective, get_objective, list_objectives, require_objective
from tools.odyssey.qat import simulate_qat_step
from tools.odyssey.rollback_exec import execute_rollback
from tools.odyssey.scheduler import declare_and_admit
from tools.odyssey import t0_run
from tools.odyssey.toy_model import FIXTURE_LABEL, ToyConfig, ToyMLP
from tools.odyssey.tournament import HALO_DIMENSIONS, Scorecard, compare
from tools.odyssey.trainer import ToyTrainer, run_uninterrupted_vs_resumed
from tools.odyssey.trajectory import TEACHER_LEDGER_FACTS, TrajectoryTracesMissing, require_parent_traces_or_refuse, trajectory_loss_shape

SCHEMA_T0_EXEC = "hawking.odyssey.t0_executable_receipt.v1"
SCHEMA_READINESS = "hawking.odyssey.trainer_readiness.v1"
SCHEMA_HEAVY = "hawking.odyssey.heavy_prerequisites.v1"


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")


def exercise_content_addressed_checkpoints(tmp: Path) -> dict[str, Any]:
    store = CheckpointStore(tmp / "ckpts")
    m = ToyMLP(ToyConfig(seed=1))
    m.train_step()
    state = m.state_dict()
    a = store.save(stage="FIXTURE", step=m.step, objective="capability_weighted_ce", model_state=state)
    b = store.save(stage="FIXTURE", step=m.step, objective="capability_weighted_ce", model_state=state)
    # Same state → same id
    same = a["checkpoint_id"] == b["checkpoint_id"]
    m.train_step()
    c = store.save(
        stage="FIXTURE",
        step=m.step,
        objective="capability_weighted_ce",
        model_state=m.state_dict(),
    )
    different = c["checkpoint_id"] != a["checkpoint_id"]
    return {
        "identical_states_same_id": same,
        "different_states_different_id": different,
        "id_a": a["checkpoint_id"],
        "id_b": b["checkpoint_id"],
        "id_c": c["checkpoint_id"],
        "pass": same and different,
    }


def exercise_failure_injection(tmp: Path) -> dict[str, Any]:
    store = CheckpointStore(tmp / "fail", shard_size=64)
    m = ToyMLP(ToyConfig(seed=2))
    for _ in range(3):
        m.train_step()
    meta = store.save(
        stage="FIXTURE",
        step=m.step,
        objective="capability_weighted_ce",
        model_state=m.state_dict(),
    )
    cid = meta["checkpoint_id"]
    results = {}

    # Corrupt
    store.inject_corrupt_shard(cid, 0)
    try:
        store.load_state(cid)
        results["corrupt"] = {"detected": False}
    except CheckpointError as e:
        results["corrupt"] = {"detected": True, "error": str(e)}

    # Fresh good checkpoint for lose-shard
    m.train_step()
    meta2 = store.save(
        stage="FIXTURE",
        step=m.step,
        objective="capability_weighted_ce",
        model_state=m.state_dict(),
    )
    cid2 = meta2["checkpoint_id"]
    store.inject_lose_shard(cid2, 0)
    try:
        store.load_state(cid2)
        results["lose_shard"] = {"detected": False}
    except CheckpointError as e:
        results["lose_shard"] = {"detected": True, "error": str(e)}

    # Mid-write
    cid3 = store.inject_kill_mid_write(
        stage="FIXTURE",
        step=99,
        objective="capability_weighted_ce",
        model_state=m.state_dict(),
    )
    try:
        store.load_state(cid3)
        results["mid_write"] = {"detected": False}
    except CheckpointError as e:
        results["mid_write"] = {"detected": True, "error": str(e)}

    results["pass"] = all(r.get("detected") for r in results.values() if isinstance(r, dict) and "detected" in r)
    return results


def exercise_tournament() -> dict[str, Any]:
    dims = HALO_DIMENSIONS
    inc = Scorecard("incumbent", 0.70, {d: 0.50 for d in dims})
    # Injected regression on coding
    bad_halo = {d: 0.60 for d in dims}
    bad_halo["coding"] = 0.10
    ch_reg = Scorecard("challenger_regressed", 0.99, bad_halo)
    r_reg = compare(inc, ch_reg)
    # Tie
    ch_tie = Scorecard("challenger_tie", 0.70, {d: 0.50 for d in dims})
    r_tie = compare(inc, ch_tie)
    # Strict win
    ch_win = Scorecard("challenger_win", 0.80, {d: 0.60 for d in dims})
    r_win = compare(inc, ch_win)
    return {
        "regression_winner": r_reg["winner"],
        "regression_keeps_incumbent": r_reg["winner"] == "incumbent",
        "tie_winner": r_tie["winner"],
        "tie_keeps_incumbent": r_tie["winner"] == "incumbent",
        "strict_win_winner": r_win["winner"],
        "strict_win_challenger": r_win["winner"] == "challenger_win",
        "pass": (
            r_reg["winner"] == "incumbent"
            and r_tie["winner"] == "incumbent"
            and r_win["winner"] == "challenger_win"
        ),
    }


def exercise_rollback(tmp: Path) -> dict[str, Any]:
    t = ToyTrainer(tmp / "rb", cfg=ToyConfig(seed=3), checkpoint_every=1)
    t.run(total_steps=5, stage="FIXTURE")
    # Collect step->id mapping by reloading each CURRENT history via store dirs.
    # Simpler: run step by step saving ids.
    t2 = ToyTrainer(tmp / "rb2", cfg=ToyConfig(seed=3), checkpoint_every=1)
    ids = {}
    for target in range(1, 6):
        t2.run(total_steps=target, stage="FIXTURE", resume=(target > 1), emit_receipt=False)
        ids[t2.model.step] = t2.last_checkpoint_id
    early = ids[2]
    late = ids[5]
    assert early and late and early != late
    result = execute_rollback(
        t2.store,
        early,
        stage="FIXTURE",
        objective=t2.objective_name,
        reason="injected_halo_regression",
    )
    t2.load_from_checkpoint(early)
    return {
        "status": result["status"],
        "restored_step": t2.model.step,
        "expected_step": 2,
        "event_recorded": result.get("event", {}).get("kind") == "rollback",
        "entry_gate_ok": (result.get("entry_gate") or {}).get("ok") is True,
        "current": t2.store.current_id(),
        "pass": (
            result["status"] == "PROVEN"
            and t2.model.step == 2
            and t2.store.current_id() == early
        ),
        "g6": result.get("g6_note"),
    }


def exercise_t0_unit() -> dict[str, Any]:
    """Callable T0 unit: static-only path (no heavy shard hashing)."""
    unit = t0_run.run_unit(include_runtime=True)
    return {
        "callable": True,
        "entry": "tools/odyssey/t0_run.py:run_unit",
        "status": unit["status"],
        "summary": unit["summary"],
        "static_ok": unit["summary"]["substrate_static"] == "PASS",
        "data_status": unit["summary"]["data"],
        "runtime_status": unit["summary"]["runtime"],
        "known_failures_status": unit["summary"]["known_failures"],
        "pass": unit["status"] == "PASS",
        "note": "Unit path exercises T0 subunits without a full 282-shard hash sweep.",
    }


def build_heavy_prerequisites() -> dict[str, Any]:
    return {
        "schema": SCHEMA_HEAVY,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what_a_real_run_would_need_that_does_not_exist": [
            {
                "id": "approved_substrate",
                "status": "MISSING",
                "detail": (
                    "odyssey/launch/SUBSTRATE_CAPABILITY.json records Math-Preserve as REFUSED "
                    "(SEMANTIC_COLLAPSE). No substrate currently has capability_verdict=APPROVED. "
                    "Silence is not a pass: unlisted artifacts are UNVERIFIED→REFUSED."
                ),
            },
            {
                "id": "training_corpora",
                "status": "DECLARED_NOT_PRESENT",
                "detail": (
                    "odyssey/data/ODYSSEY_DATA_MANIFEST.json declares math-core, support-language, "
                    "long-horizon, sovereignty-corpus — all present=false / DECLARED_NOT_COLLECTED."
                ),
            },
            {
                "id": "trajectory_traces",
                "status": "MISSING",
                "detail": (
                    f"Teacher ledger: {TEACHER_LEDGER_FACTS['ledger_lines']} lines, "
                    f"{TEACHER_LEDGER_FACTS['per_layer_captures']} per-layer captures, "
                    f"{TEACHER_LEDGER_FACTS['trajectory_traces']} trajectory traces. "
                    "T3 cannot run without parent trajectory traces."
                ),
            },
            {
                "id": "launch_authorization",
                "status": "FALSE",
                "detail": (
                    "ODYSSEY_LAUNCH_AUTHORIZED is false. This is deliberate. "
                    "The apparatus must not flip it."
                ),
            },
            {
                "id": "heavy_window",
                "status": "NOT_GRANTED",
                "detail": (
                    "A real T1+ step against a 92 GB substrate needs a heavy resource window: "
                    "streaming-adapter step ~2.1 GiB working set (or ~86 GiB full-resident), "
                    "wall clock minutes–hours per meaningful step, and sole occupancy of the "
                    "machine (LIGHT_ONLY currently forbids heavy parent work)."
                ),
            },
            {
                "id": "served_capable_model",
                "status": "MISSING",
                "detail": (
                    "Forge F1–F4 and false-refusal metrics need a served model that is not "
                    "semantically collapsed. Math-Preserve is not that model."
                ),
            },
            {
                "id": "qat_on_parent",
                "status": "NOT_IMPLEMENTABLE_HERE",
                "detail": (
                    "Only FakeUniformQuantizer exists. Full QAT recovery on a real substrate "
                    "is not implementable under LIGHT_ONLY and has never been measured here."
                ),
            },
        ],
        "honest_summary": (
            "The trainer apparatus is proven on toy fixtures. A real Odyssey training run "
            "cannot start: no APPROVED substrate, no collected corpora, no trajectory traces, "
            "fence false, no heavy window."
        ),
    }


def run_apparatus(*, write_receipts: bool = True) -> dict[str, Any]:
    """Exercise every apparatus surface on fixtures and emit deliverables."""
    tmp = Path(tempfile.mkdtemp(prefix="odyssey-apparatus-"))
    results: dict[str, Any] = {"tmp": str(tmp), "fixture_label": FIXTURE_LABEL}

    # 1. T0 unit
    results["t0_unit"] = exercise_t0_unit()

    # 2. Content-addressed checkpoints
    results["content_addressed"] = exercise_content_addressed_checkpoints(tmp)

    # 3. Resume bit-identity
    results["resume_bit_identity"] = run_uninterrupted_vs_resumed(
        tmp / "resume", total_steps=8, kill_after_step=4
    )

    # 4. Rollback
    results["rollback"] = exercise_rollback(tmp)

    # 5. Objective registry
    try:
        require_objective("capability_weighted_ce", stage="FIXTURE")
        unreg_refused = False
        try:
            get_objective("not_a_real_objective_xyz")
        except UnregisteredObjective:
            unreg_refused = True
        results["objectives"] = {
            "registered": list_objectives(),
            "unregistered_refused": unreg_refused,
            "pass": unreg_refused,
        }
    except Exception as e:
        results["objectives"] = {"pass": False, "error": str(e)}

    # 6. QAT fake
    qat = simulate_qat_step()
    results["qat"] = {
        "kind": qat["kind"],
        "quantizer_name": qat["quantizer_name"],
        "is_measurement": qat.get("is_measurement"),
        "quantizer_is_fake": qat.get("quantizer_is_fake"),
        "pass": (
            qat["kind"] == "fake_simulation"
            and qat.get("is_measurement") is False
            and qat.get("quantizer_is_fake") is True
        ),
    }

    # 7. Trajectory interface
    traj = trajectory_loss_shape([1, 2, 3, 4], [1, 2, 9, 4], source="fixture_tokens")
    refused = False
    try:
        require_parent_traces_or_refuse()
    except TrajectoryTracesMissing:
        refused = True
    results["trajectory"] = {
        "interface_status": traj["status"],
        "teacher_ledger": TEACHER_LEDGER_FACTS,
        "parent_traces_refused": refused,
        "pass": traj["first_divergence_index"] == 2 and refused,
    }

    # 8. Tournament
    results["tournament"] = exercise_tournament()

    # 9. Failure injection
    results["failure_injection"] = exercise_failure_injection(tmp)

    # 10. Resource scheduler declaration
    sched = declare_and_admit(
        stage="FIXTURE",
        wall_time_budget_s=30.0,
        rss_budget_bytes=256 * 1024 * 1024,
        threads=1,
        heavy=False,
        memory_bytes=32 * 1024 * 1024,
    )
    results["scheduler"] = {
        "status": sched["status"],
        "declared_wall_time_s": sched["declaration"]["declared"]["wall_time_budget_s"],
        "declared_rss_bytes": sched["declaration"]["declared"]["rss_budget_bytes"],
        "declared_threads": sched["declaration"]["declared"]["threads"],
        "pass": (
            sched["status"] == "ADMITTED"
            and "wall_time_budget_s" in sched["declaration"]["declared"]
            and "rss_budget_bytes" in sched["declaration"]["declared"]
            and "threads" in sched["declaration"]["declared"]
        ),
    }

    # 11. Training receipts (from a short run)
    t = ToyTrainer(tmp / "receipts_run", cfg=ToyConfig(seed=9), checkpoint_every=2)
    run_result = t.run(total_steps=4, stage="FIXTURE")
    results["training_receipts"] = {
        "receipt_id": run_result["receipt"]["receipt_id"],
        "fixture": run_result["receipt"]["fixture"],
        "content_addressed": bool(run_result["receipt"]["receipt_id"]),
        "pass": run_result["receipt"]["fixture"] is True and len(run_result["receipt"]["receipt_id"]) == 64,
    }

    # Aggregate
    keys = [
        "t0_unit",
        "content_addressed",
        "resume_bit_identity",
        "rollback",
        "objectives",
        "qat",
        "trajectory",
        "tournament",
        "failure_injection",
        "scheduler",
        "training_receipts",
    ]
    # resume uses bit_identical key
    passes = []
    for k in keys:
        block = results[k]
        if k == "resume_bit_identity":
            passes.append(bool(block.get("bit_identical")))
        else:
            passes.append(bool(block.get("pass")))
    results["all_pass"] = all(passes)
    results["pass_map"] = {
        k: (
            bool(results[k].get("bit_identical"))
            if k == "resume_bit_identity"
            else bool(results[k].get("pass"))
        )
        for k in keys
    }

    # Deliverables
    t0_exec = {
        "schema": SCHEMA_T0_EXEC,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PASS" if results["t0_unit"]["pass"] else "FAIL",
        "fixture_label": FIXTURE_LABEL,
        "t0_unit": results["t0_unit"],
        "launch_authorized": False,
        "note": (
            "Executable T0 unit path (static substrate checks + data classification + "
            "runtime authority + known-failure registry). Does not flip the fence. "
            "Full 282-shard hash sweep remains available via tools/odyssey/t0_run.py --full-substrate."
        ),
        "entry_points": {
            "unit": "tools/odyssey/apparatus.py:exercise_t0_unit",
            "full": "tools/odyssey/t0_run.py:run",
            "gated_runner": "odyssey/training/run.py (refuses while fence is false)",
        },
    }

    readiness = {
        "schema": SCHEMA_READINESS,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "APPARATUS_PROVEN_ON_FIXTURES" if results["all_pass"] else "APPARATUS_FAILED",
        "plain_statement": (
            "The Odyssey trainer apparatus has been exercised on pure-numpy toy models "
            "(a few thousand parameters). It has NEVER trained anything real. "
            "Every fixture is labelled a fixture. Content-addressed checkpoints, kill/resume "
            "bit-identity, rollback with entry-gate re-run, objective registry refusal, "
            "FakeUniformQuantizer, trajectory-loss interface, tournament with injected "
            "regressions, failure injection, resource declarations, and content-addressed "
            "receipts are proven on fixtures only."
        ),
        "never_trained_anything_real": True,
        "launch_authorized": False,
        "substrate_capability": "Math-Preserve remains REFUSED; no APPROVED substrate",
        "pass_map": results["pass_map"],
        "all_apparatus_surfaces_pass": results["all_pass"],
        "toy_model": {
            "n_params": ToyMLP(ToyConfig()).n_params(),
            "fixture_label": FIXTURE_LABEL,
            "backend": "numpy",
            "torch": False,
            "gpu": False,
        },
        "evidence": {
            "resume_bit_identical": results["resume_bit_identity"].get("bit_identical"),
            "rollback_status": results["rollback"].get("status"),
            "qat_kind": results["qat"].get("kind"),
            "trajectory_traces": TEACHER_LEDGER_FACTS["trajectory_traces"],
            "failure_injection_pass": results["failure_injection"].get("pass"),
        },
        "what_this_does_not_establish": [
            "That any real corpus was trained on",
            "That Math-Preserve or any parent improved",
            "That QAT was measured on a substrate",
            "That T3 trajectory stabilization ran against parent traces",
            "That ODYSSEY_LAUNCH_AUTHORIZED may be flipped",
        ],
    }

    heavy = build_heavy_prerequisites()

    if write_receipts:
        _write(ODYSSEY / "ODYSSEY_T0_EXECUTABLE_RECEIPT.json", t0_exec)
        _write(ODYSSEY / "ODYSSEY_TRAINER_READINESS.json", readiness)
        _write(ODYSSEY / "ODYSSEY_HEAVY_PREREQUISITES.json", heavy)

    results["deliverables"] = {
        "ODYSSEY_T0_EXECUTABLE_RECEIPT": t0_exec,
        "ODYSSEY_TRAINER_READINESS": readiness,
        "ODYSSEY_HEAVY_PREREQUISITES": heavy,
    }
    return results


def main(argv: list[str] | None = None) -> int:
    out = run_apparatus(write_receipts=True)
    summary = {
        "all_pass": out["all_pass"],
        "pass_map": out["pass_map"],
        "wrote": [
            "odyssey/ODYSSEY_T0_EXECUTABLE_RECEIPT.json",
            "odyssey/ODYSSEY_TRAINER_READINESS.json",
            "odyssey/ODYSSEY_HEAVY_PREREQUISITES.json",
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
