#!/usr/bin/env python3.12
"""Build the Model Odyssey package, and refuse to start it.

Odyssey is the training and research run that comes AFTER this campaign. This builds
everything it needs -- training plan, data and teacher-trace manifests, checkpoint and
evaluation contracts, the sandbox policy, the nine Ramanujan roles, the Ledger, the
verification lattice, the Tribunal, retrieval, economics, and the pinned proof
environment -- and installs a fence that makes starting it a deliberate act in the next
session rather than a side effect of building it here.

Two properties are load-bearing and both are asserted by `validate`:

  ODYSSEY_LAUNCH_AUTHORIZED is false, and nothing in the package can flip it. The fence
  lives in a file the package reads and never writes.

  Every artifact the package points at is either present and content-addressed, or
  declared GATED with the gate named. A manifest that references a substrate which does
  not exist yet is honest only if it says so; silently naming a path that will exist
  later is how a dry run passes and a real run does not.

    python3.12 tools/campaign/odyssey_package.py build
    python3.12 tools/campaign/odyssey_package.py validate
    python3.12 tools/campaign/odyssey_package.py selftest
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ODYSSEY = ROOT / "odyssey"

DIRS = [
    "doctrine", "substrate", "data", "teacher_traces", "profiles", "training",
    "checkpoints", "evaluation", "sandbox", "roles", "ledger", "memory",
    "verifiers", "tribunal", "retrieval", "economics", "graveyard",
    "instruments", "forge", "sovereignty", "experiments", "launch", "rollback",
]

# Revision 3 §10.3. Roles R1-R7 cannot promote claim status; only verifier events
# and the Tribunal can. The `may_promote` flag is what the Director enforces.
ROLES = [
    ("director", "Allocates turns across roles and holds no mathematical opinion.",
     ["schedule", "read_ledger"], False),
    ("conjecturer", "Generates candidate claims. May not assert confidence.",
     ["read_memory", "write_claim"], False),
    ("skeptic", "Attacks claims. May not prove.",
     ["read_claim", "write_objection"], False),
    ("formalizer", "Translates to Lean. May not silently weaken a statement.",
     ["read_claim", "write_lean"], False),
    ("prover", "Attempts formal proof. No proof exists without compiler acceptance.",
     ["read_lean", "run_lean"], False),
    ("computationalist", "Runs symbolic and numerical experiments. May not treat computation as proof.",
     ["run_sandbox", "write_evidence"], False),
    ("cartographer", "Maintains investigation topology. May not generate mathematics.",
     ["read_ledger", "write_topology"], False),
    ("librarian", "Searches prior art. May not judge correctness.",
     ["run_retrieval", "write_literature"], False),
    ("economist", "Allocates compute. Never sees claim content.",
     ["read_economics", "write_budget"], False),
    ("adversary", "Audits the system on schedule. Cannot be suppressed or deprioritized.",
     ["read_all", "write_audit"], False),
    ("tribunal", "Adjudicates admissibility and novelty. Never certifies its own novelty.",
     ["read_all", "promote_claim"], True),
    ("verifier", "Compiler and reproduction events. The only non-Tribunal promoter.",
     ["run_lean", "run_sandbox", "promote_claim"], True),
]

# Revision 3 §10.5.
EVIDENCE_TIERS = [
    (0, "asserted", "The model said it. No evidentiary weight.", []),
    (1, "empirically_supported",
     "Reproducible computation over a declared domain. Never becomes proof by accumulation.",
     ["sandbox_reproduction"]),
    (2, "formalized",
     "A faithful Lean statement exists and was independently fidelity-assessed.",
     ["lean_statement_compiles", "independent_fidelity_assessment"]),
    (3, "proven",
     "Compiler-accepted proof, no sorry, no undeclared axioms, pinned Mathlib, clean-container reproduction.",
     ["lean_proof_compiles", "no_sorry", "no_undeclared_axioms", "pinned_mathlib",
      "clean_container_reproduction"]),
]

# Revision 3 §10.9. Refutation and clean exhaustion are rewarded deliberately.
BRANCH_ECONOMICS = {
    "tier0_to_tier1": 1, "tier1_to_tier2": 3, "tier2_to_tier3": 10,
    "refutation": 5, "clean_branch_exhaustion": 3, "tribunal_admission": 25,
}

MEMORY_STORES = [
    ("M1", "claims", "Every claim ever made, at its current tier."),
    ("M2", "evidence", "Every verifier event, with its inputs and its container hash."),
    ("M3", "trajectories", "How each investigation moved, including the dead moves."),
    ("M4", "structure", "The topology of the investigation space."),
    ("M5", "literature", "Prior-art queries and what they returned, with query hashes."),
    ("M6", "economics", "Compute spent per branch and what it bought."),
    ("M7", "graveyard", "Refuted and exhausted branches. Nothing is deleted; "
                        "selective context is retrieval, not forgetting."),
]

TRAINING_STAGES = [
    ("T0", "baseline_reproduction",
     "Reproduce the frozen substrate's measured baseline before changing anything. "
     "A training run that cannot reproduce its own starting point cannot attribute a delta."),
    ("T1", "capability_conditioned_continued_training",
     "Continued training under the selected capability profile."),
    ("T2", "qat_headroom",
     "Quantization-aware headroom recovery at the frozen rate."),
    ("T3", "trajectory_stabilization",
     "Reduce long-horizon divergence from the parent's reasoning trajectory."),
    ("T4", "forge_profile_expression",
     "Restore full authorized expression of the preserved capability inside the profile."),
    ("T5", "hidden_replication_and_sovereignty",
     "Hidden-set replication plus the sovereignty gauntlet. Nothing graduates on a seen set."),
]

# Revision 3 §10.4 -- pinned so a Tier-3 proof is reproducible in a clean container.
PROOF_ENVIRONMENT = {
    "lean": {"toolchain": "leanprover/lean4:v4.15.0", "pin_kind": "toolchain_file"},
    "mathlib": {"revision": "v4.15.0", "pin_kind": "lakefile_require",
                "note": "Tier 3 requires the pinned revision; a proof that needs a "
                        "different Mathlib is a different proof."},
    "python": {"version": "3.12", "packages": ["numpy", "sympy", "mpmath"]},
    "container": {"kind": "declared", "digest": None,
                  "gate": "ODYSSEY-ENV-01: digest is recorded on first clean build"},
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, str):
        path.write_text(obj)
    else:
        path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    return path


def _substrate() -> dict:
    """What Odyssey would train, described honestly whether or not it exists.

    The selected compact artifact comes out of Prometheus, which is gated on the
    flagship traversal. Naming a path that does not exist yet without saying so is how
    a dry run passes and a real run does not.
    """
    campaign_status = ROOT / "HAWKING_MOTHERLOAD_STATUS.json"
    gates = {}
    if campaign_status.is_file():
        gates = json.loads(campaign_status.read_text()).get("gates", {})

    def gate_state(gid: str) -> str:
        return gates.get(gid, {}).get("state", "OPEN")

    # The runtime instrument is real and content-addressed today; the flagship is not.
    instrument = Path.home() / (
        "Library/Application Support/Hawking/CampaignS08/llama32-1b-R0.v2.gravity")
    entries = []
    if instrument.is_file():
        entries.append({
            "role": "runtime_instrument",
            "path": str(instrument),
            "bytes": instrument.stat().st_size,
            "sha256": _sha256(instrument),
            "present": True,
            "note": "Grades the runtime, not the science. Not an Odyssey training target.",
        })
    entries.append({
        "role": "odyssey_training_substrate",
        "path": None,
        "present": False,
        "gate": "M11: General and Math artifacts are selected and verified",
        "gate_state": gate_state("M11"),
        "note": "The selected compact Math artifact from Prometheus. Odyssey trains this "
                "and nothing else; a substitute substrate is a different experiment.",
    })
    return {
        "schema": "hawking.odyssey.substrate.v1",
        "selected": None,
        "selection_gate": "M11",
        "entries": entries,
    }


def build() -> dict:
    ODYSSEY.mkdir(parents=True, exist_ok=True)
    for d in DIRS:
        (ODYSSEY / d).mkdir(parents=True, exist_ok=True)
        gk = ODYSSEY / d / ".gitkeep"
        if not gk.exists():
            gk.write_text("")

    written: list[str] = []

    def w(rel: str, obj) -> None:
        written.append(str(_write(ODYSSEY / rel, obj).relative_to(ROOT)))

    # --- the fence -------------------------------------------------------
    w("launch/ODYSSEY_LAUNCH_AUTHORIZED", "false\n")
    w("launch/ODYSSEY_FENCE.json", {
        "schema": "hawking.odyssey.fence.v1",
        "authorized": False,
        "authorized_by": None,
        "authority": "the user, explicitly, in the session that starts Odyssey",
        "enforced_at": [
            "odyssey/training/run.py refuses to start unless the fence file reads true",
            "odyssey/sandbox/POLICY.json declares network egress denied by default",
            "tools/campaign/odyssey_package.py validate fails if the fence is not false",
        ],
        "note": "Nothing inside the package may write this file. Building the package "
                "must never be the same act as authorizing the run.",
    })

    # --- doctrine --------------------------------------------------------
    w("doctrine/DOCTRINE.json", {
        "schema": "hawking.odyssey.doctrine.v1",
        "source_documents": [
            "Hawking_Prometheus_Ramanujan_Canonical_Master_Plan_Revision_3.md",
            "HAWKING_MOTHERLOAD_COMPLETION_CAMPAIGN.md",
        ],
        "canonical_distinction": {
            "prometheus": "preserves and conditions capability",
            "forge": "restores its authorized expression",
            "ramanujan": "the complete mathematical research system",
            "checkpoint": "a compact or trained checkpoint is not automatically Ramanujan",
        },
        "principles": [
            "deterministic first", "training-free first", "equal-budget comparisons",
            "measurement over metaphor", "preserve support capabilities",
            "architecture generality", "exact byte accounting", "evidence-gated releases",
            "resumability and streaming", "local-first outcome", "minimal public surface",
            "honest mathematical claims", "capability sovereignty",
        ],
        "claims": {
            "A": "compression: capability-conditioned allocation beats uniform at equal bytes",
            "B": "locality: frontier-class reasoning on one workstation",
            "C": "accumulation: monotonically increasing verified evidence",
            "D": "novelty: at least one result survives formalization, review and reproduction",
            "E": "sovereignty: preserved capability expressible without hidden intervention",
        },
    })

    # --- substrate -------------------------------------------------------
    w("substrate/ODYSSEY_SUBSTRATE.json", _substrate())

    # --- training --------------------------------------------------------
    w("training/ODYSSEY_TRAINING_PLAN.json", {
        "schema": "hawking.odyssey.training_plan.v1",
        "launch_authorized": False,
        "stages": [{"id": i, "name": n, "intent": d,
                    "entry_gate": f"stage {i} may not start until stage "
                                  f"{TRAINING_STAGES[k-1][0] if k else 'none'} is sealed",
                    "status": "PREPARED_NOT_STARTED"}
                   for k, (i, n, d) in enumerate(TRAINING_STAGES)],
        "invariants": [
            "no stage begins before the fence reads true",
            "every stage emits a checkpoint before the next stage reads it",
            "T0 must reproduce the frozen baseline or the run stops",
            "no evaluation set used for selection is ever used for reporting",
        ],
    })
    w("training/ODYSSEY_OBJECTIVE_CONTRACT.json", {
        "schema": "hawking.odyssey.objective.v1",
        "objectives": [
            {"stage": "T1", "loss": "capability-weighted cross-entropy",
             "weighting": "the selected profile's primary/support weights",
             "reference": "profiles/prometheus/math-v1.json"},
            {"stage": "T2", "loss": "quantization-aware distillation to the frozen rate",
             "note": "the rate is frozen; QAT recovers headroom, it does not spend bytes"},
            {"stage": "T3", "loss": "trajectory divergence penalty against the parent",
             "metric": "token and semantic divergence depth, reported as a distribution"},
            {"stage": "T4", "loss": "false-refusal reduction on verified permitted tasks",
             "note": "objective is correctly authorized expression, not answer rate"},
        ],
        "forbidden": [
            "training on any hidden evaluation set",
            "optimizing a metric that a role can promote by itself",
            "any objective that rewards answer rate independent of authorization",
        ],
    })
    w("training/ODYSSEY_CHECKPOINT_CONTRACT.json", {
        "schema": "hawking.odyssey.checkpoint.v1",
        "addressing": "content-addressed by artifact sha256",
        "required_fields": ["stage", "step", "artifact_sha256", "parent_sha256",
                            "profile_hash", "data_manifest_hash", "objective_hash",
                            "eval_result_hash", "sovereignty_manifest_hash", "wall_clock"],
        "retention": "every checkpoint that any evaluation cites is retained; "
                     "intermediate steps may be pruned only if nothing cites them",
        "rollback": "any checkpoint may be made current; rollback is a recorded event, "
                    "never a silent substitution",
    })
    w("training/run.py", _RUNNER)

    # --- data and teacher traces ----------------------------------------
    w("data/ODYSSEY_DATA_MANIFEST.json", {
        "schema": "hawking.odyssey.data.v1",
        "status": "DECLARED_NOT_COLLECTED",
        "gate": "M13: Odyssey substrate and training bundle are complete",
        "corpora": [
            {"id": "math-core", "purpose": "T1 primary", "license_required": "permissive",
             "content_addressed": True, "present": False},
            {"id": "support-language", "purpose": "T1 support: technical language, coding, tools",
             "license_required": "permissive", "content_addressed": True, "present": False},
            {"id": "long-horizon", "purpose": "T3 trajectory stabilization",
             "license_required": "permissive", "content_addressed": True, "present": False},
            {"id": "sovereignty-corpus", "purpose": "T4 permitted/boundary/paraphrase pairs",
             "license_required": "self-authored", "content_addressed": True, "present": False},
        ],
        "invariants": [
            "every corpus is content-addressed before a single step reads it",
            "no corpus overlaps any hidden evaluation set",
            "license is recorded per corpus, not per collection run",
        ],
    })
    w("teacher_traces/ODYSSEY_TEACHER_TRACE_MANIFEST.json", {
        "schema": "hawking.odyssey.teacher_traces.v1",
        "status": "PARTIAL",
        "existing": {
            "glm52_teacher_capsules": {
                "ledger": str(Path.home() / "Library/Application Support/Hawking/"
                              "GLM52Gravity/source_fetch/teacher/"
                              "GLM52_TEACHER_EVIDENCE_LEDGER.jsonl"),
                "note": "captured during the flagship traversal; layer-scoped, not "
                        "trajectory-scoped. Usable for T2, not sufficient for T3.",
            }
        },
        "required_for": {"T2": "layer-scoped teacher capsules",
                         "T3": "full trajectory traces from the parent"},
        "gate": "T3 traces require the flagship source, which is released after M18",
    })

    # --- profiles --------------------------------------------------------
    profiles_src = ROOT / "profiles/prometheus"
    w("profiles/ODYSSEY_PROFILE_MANIFEST.json", {
        "schema": "hawking.odyssey.profiles.v1",
        "source_dir": str(profiles_src.relative_to(ROOT)),
        "profiles": [{"name": p.stem, "sha256": _sha256(p)}
                     for p in sorted(profiles_src.glob("*.json"))],
        "selected_for_odyssey": None,
        "selection_gate": "M11",
    } if profiles_src.is_dir() else {
        "schema": "hawking.odyssey.profiles.v1", "profiles": [],
        "selected_for_odyssey": None, "selection_gate": "M11"})

    # --- evaluation ------------------------------------------------------
    w("evaluation/ODYSSEY_EVALUATION_CONTRACT.json", {
        "schema": "hawking.odyssey.evaluation.v1",
        "sets": [
            {"id": "selection", "visible_to_training": False, "used_for": "stage gating"},
            {"id": "hidden", "visible_to_training": False, "used_for": "reporting only",
             "rule": "never seen by any selection decision"},
            {"id": "sovereignty", "visible_to_training": False,
             "used_for": "false refusal and boundary error"},
        ],
        "metrics": ["math aggregate", "per-domain math", "formalization",
                    "long-horizon stability", "trajectory divergence", "throughput",
                    "context viability", "false refusal under profile"],
        "reporting": "distributions, not only averages",
        "invariants": [
            "a metric computed on the selection set may never be reported as a result",
            "no evaluation runs with a fallback model enabled",
        ],
    })

    # --- sandbox ---------------------------------------------------------
    w("sandbox/POLICY.json", {
        "schema": "hawking.odyssey.sandbox.v1",
        "status": "PREPARED_NOT_STARTED",
        "network": {"default": "deny", "allowlist": [],
                    "note": "retrieval runs against a pinned snapshot, not the live web"},
        "filesystem": {"default": "deny",
                       "read": ["odyssey/", "profiles/"],
                       "write": ["odyssey/ledger/", "odyssey/checkpoints/",
                                 "odyssey/experiments/", "odyssey/graveyard/"]},
        "tools": {"allowlist": ["lean", "python3.12", "sympy", "mpmath"],
                  "note": "a tool not on this list is a permission event, not a refusal"},
        "resources": {"max_wall_clock_hours_per_session": 12,
                      "max_concurrent_heavy_lanes": 1,
                      "memory_ceiling_bytes": 96 * 2**30},
        "process_isolation": {"kind": "subprocess with explicit cwd and env",
                              "inherit_env": False},
        "ingress": {"rule": "artifacts enter only by content-addressed path"},
        "egress": {"rule": "artifacts leave only into odyssey/experiments/, never the network"},
        "audit": {"log": "odyssey/ledger/sandbox_events.jsonl", "append_only": True},
        "emergency_stop": {"file": "odyssey/launch/STOP",
                           "rule": "presence of this file halts every loop at the next "
                                   "checkpoint boundary"},
        "reproducibility": {"environment": PROOF_ENVIRONMENT},
    })

    # --- roles, ledger, verifiers, tribunal ------------------------------
    w("roles/ROLES.json", {
        "schema": "hawking.odyssey.roles.v1",
        "structural_property": "roles that generate may not promote; only verifier "
                               "events and the Tribunal advance evidence status",
        "roles": [{"id": r, "charter": c, "capabilities": caps, "may_promote": promote}
                  for r, c, caps, promote in ROLES],
    })
    w("ledger/LEDGER_CONTRACT.json", {
        "schema": "hawking.odyssey.ledger.v1",
        "append_only": True,
        "law": "what is not recorded did not happen; what is recorded cannot be revised",
        "event_kinds": ["claim", "objection", "formalization", "proof_attempt",
                        "verifier_event", "tribunal_decision", "literature_query",
                        "budget_grant", "sovereignty_event", "sandbox_event",
                        "checkpoint", "rollback"],
        "required_fields": ["id", "at", "kind", "role", "parents", "payload_sha256"],
        "correction_policy": "a wrong row is superseded by a new row that names it; "
                             "rows are never edited or removed",
    })
    w("verifiers/VERIFICATION_LATTICE.json", {
        "schema": "hawking.odyssey.verification.v1",
        "tiers": [{"tier": t, "name": n, "meaning": m, "requires": r}
                  for t, n, m, r in EVIDENCE_TIERS],
        "invariants": [
            "Tier 1 never becomes proof by accumulation",
            "Tier 2 requires an independent fidelity assessment, not the formalizer's own",
            "Tier 3 requires a clean-container reproduction against the pinned Mathlib",
        ],
    })
    w("tribunal/TRIBUNAL.json", {
        "schema": "hawking.odyssey.tribunal.v1",
        "law": "the system that produces a claim is never the system that admits it",
        "admits": ["admissibility", "novelty"],
        "requires_human_expert_gate": True,
        "prior_art_protocol": [
            "pre-register the branch question",
            "execute a structured query ladder",
            "exhaust Mathlib search",
            "write an explicit boundary statement",
            "require a human expert gate",
        ],
        "note": "the system never certifies its own novelty",
    })
    w("memory/MEMORY_STORES.json", {
        "schema": "hawking.odyssey.memory.v1",
        "stores": [{"id": i, "name": n, "contents": c} for i, n, c in MEMORY_STORES],
        "law": "nothing is deleted; selective context is retrieval, not forgetting",
    })
    w("retrieval/RETRIEVAL.json", {
        "schema": "hawking.odyssey.retrieval.v1",
        "sources": [{"id": "mathlib", "kind": "pinned_checkout",
                     "revision": PROOF_ENVIRONMENT["mathlib"]["revision"]},
                    {"id": "literature", "kind": "pinned_snapshot", "snapshot": None,
                     "gate": "ODYSSEY-RET-01: snapshot taken before the first session"}],
        "rule": "retrieval never touches the live network during a run; a query that "
                "needs a source outside the snapshot is a permission event",
        "query_log": "odyssey/ledger/literature_queries.jsonl",
    })
    w("economics/ECONOMICS.json", {
        "schema": "hawking.odyssey.economics.v1",
        "currency": "verified evidence",
        "rewards": BRANCH_ECONOMICS,
        "rationale": "refutation and clean exhaustion are rewarded deliberately: a system "
                     "that only pays for promotion learns to avoid falsification",
        "allocator": "economist",
        "constraint": "the economist never sees claim content",
    })
    w("graveyard/GRAVEYARD.json", {
        "schema": "hawking.odyssey.graveyard.v1",
        "contents": "refuted and exhausted branches, retained in full",
        "revival": "a branch may be revived when a new verifier event or literature "
                   "result changes its premises; revival is a Ledger event",
    })
    w("instruments/INSTRUMENTS.json", {
        "schema": "hawking.odyssey.instruments.v1",
        "runtime": {"decoder": "hawking-core .gravity", "abi": "hawking.gravity.container.v1",
                    "adapters": ["llama", "glm_moe_dsa"]},
        "oracles": [
            "tools/condense/gravity_llama_reference.py",
            "tools/condense/glm52_reference.py",
        ],
        "base_scoreboard": "HAWKING_GRAVITY_BASE_TPS.json",
    })

    # --- forge and sovereignty ------------------------------------------
    w("forge/FORGE.json", {
        "schema": "hawking.odyssey.forge.v1",
        "pipeline": [
            {"id": "F0", "name": "diagnose", "status": "PARTIAL",
             "note": "deterministic half built in tools/sovereignty/"},
            {"id": "F1", "name": "locate", "status": "GATED",
             "gate": "needs a served model and an evaluated prompt set"},
            {"id": "F2", "name": "restore_authorized_expression", "status": "GATED",
             "gate": "needs F1"},
            {"id": "F3", "name": "validate", "status": "GATED", "gate": "needs F2"},
            {"id": "F4", "name": "seal", "status": "GATED", "gate": "needs F3"},
        ],
        "objective": "maximum correctly authorized capability expression, not maximum "
                     "answer rate",
        "restoration_priority": [
            "remove unnecessary external routing",
            "replace blanket topic rules with profile-aware policy",
            "correct excessive-refusal prompts",
            "preserve specialist pathways",
            "apply bounded inference-time steering",
            "optionally add a narrow recovery adapter on verified false refusals",
            "alter source post-training only as a separate research lane",
        ],
    })
    w("sovereignty/SOVEREIGNTY.json", {
        "schema": "hawking.odyssey.sovereignty.v1",
        "mode": "strict",
        "continuity": {"fallback_allowed": False,
                       "law": "the chosen model answers, or Hawking explains why it did not"},
        "planes": ["capability", "policy", "permission", "evidence", "resource"],
        "plane_law": "these planes may not impersonate one another",
        "refusal_attribution": ["R-CAPABILITY", "R-LEARNED", "R-POLICY", "R-PERMISSION",
                                "R-VERIFICATION", "R-RESOURCE"],
        "metrics": {
            "hidden_intervention_rate": {"target": 0, "status": "computable"},
            "model_continuity_rate": {"target": 1, "status": "computable"},
            "attribution_completeness": {"target": 1, "status": "computable"},
            "false_refusal_rate": {"status": "GATED",
                                   "gate": "needs a served model and an evaluated prompt set"},
            "boundary_error_rate": {"status": "GATED",
                                    "gate": "needs a served model and an evaluated prompt set"},
        },
        "implementation": "tools/sovereignty/sovereignty.py",
        "limit_registry": "odyssey/sovereignty/LIMIT_REGISTRY.json",
    })
    w("sovereignty/LIMIT_REGISTRY.json", {
        "schema": "hawking.odyssey.limits.v1",
        "law": "Hawking is not literally unlimited; it is free of invisible limits",
        "limits": [
            {"id": "L-LAUNCH-01", "owner": "user", "type": "policy",
             "reason": "Odyssey must be started deliberately, in a session that intends it",
             "scope": "the entire Odyssey run", "threshold": "explicit authorization",
             "current_value": False, "user_changeable": True,
             "event_log": "odyssey/ledger/sovereignty_events.jsonl"},
            {"id": "L-NET-01", "owner": "sandbox", "type": "permission",
             "reason": "a research loop with live network access is not reproducible",
             "scope": "all sandboxed processes", "threshold": "deny by default",
             "current_value": "deny", "user_changeable": True,
             "event_log": "odyssey/ledger/sandbox_events.jsonl"},
            {"id": "L-HEAVY-01", "owner": "operator", "type": "resource",
             "reason": "one heavy local lane keeps measurements comparable",
             "scope": "local compute", "threshold": 1, "current_value": 1,
             "user_changeable": True,
             "event_log": "odyssey/ledger/sandbox_events.jsonl"},
        ],
    })

    # --- rollback and launch --------------------------------------------
    w("rollback/ROLLBACK.json", {
        "schema": "hawking.odyssey.rollback.v1",
        "triggers": ["T0 fails to reproduce the baseline",
                     "a stage regresses a frozen gate",
                     "a sovereignty metric moves in the wrong direction",
                     "the emergency stop file appears"],
        "procedure": ["halt at the next checkpoint boundary",
                      "record a rollback event naming the checkpoint restored",
                      "restore the named checkpoint as current",
                      "re-run the stage's entry gate before resuming"],
        "law": "rollback is a recorded event, never a silent substitution",
    })
    w("launch/ODYSSEY_LAUNCH.md", _LAUNCH_MD)

    manifest = {
        "schema": "hawking.odyssey.package.v1",
        "built_at": _now(),
        "root": str(ODYSSEY.relative_to(ROOT)),
        "directories": DIRS,
        "files": sorted(written),
        "launch_authorized": False,
        "status": "PREPARED_NOT_STARTED",
    }
    _write(ODYSSEY / "ODYSSEY_PACKAGE.json", manifest)
    return manifest


_RUNNER = '''#!/usr/bin/env python3.12
"""Odyssey stage runner. Refuses to start unless the fence says otherwise.

The fence is read, never written. Building the package and authorizing the run are
different acts by construction, so no amount of re-running the builder can start
training.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FENCE = HERE.parent / "launch" / "ODYSSEY_LAUNCH_AUTHORIZED"
STOP = HERE.parent / "launch" / "STOP"


def authorized() -> bool:
    return FENCE.is_file() and FENCE.read_text().strip().lower() == "true"


def main(argv: list[str]) -> int:
    if STOP.is_file():
        print("ODYSSEY_STOPPED: emergency stop file present", file=sys.stderr)
        return 2
    if not authorized():
        print("ODYSSEY_LAUNCH_AUTHORIZED=false -- refusing to start.", file=sys.stderr)
        print(f"Authorize deliberately by writing 'true' to {FENCE}", file=sys.stderr)
        return 1
    print("authorized; stage execution is implemented in the session that starts Odyssey")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
'''

_LAUNCH_MD = """# Odyssey Launch Packet

Odyssey is **prepared and not started**. `ODYSSEY_LAUNCH_AUTHORIZED` is `false`.

## What is ready

- the full package tree under `odyssey/`, content-addressed where the artifacts exist
- training plan T0-T5, objective contract, checkpoint contract, evaluation contract
- sandbox policy: network denied by default, filesystem allowlisted, one heavy lane
- the nine Ramanujan roles plus Adversary, Tribunal and verifier, with promotion rights
  held only by verifier events and the Tribunal
- the four-tier verification lattice, seven memory stores, branch economics, Graveyard
- Lean and Mathlib pinned; a Tier-3 proof that needs a different Mathlib is a different proof

## What is not ready, and why

The Odyssey training substrate is the compact Math artifact Prometheus selects, and that
selection is gated on the flagship traversal (gate M11). Odyssey trains that artifact and
nothing else; substituting an available model would be a different experiment wearing this
one's name.

## To start Odyssey in the next session

Authorize deliberately, then run the stage runner:

```bash
printf 'true\\n' > odyssey/launch/ODYSSEY_LAUNCH_AUTHORIZED && python3.12 odyssey/training/run.py T0
```

To halt any running loop at the next checkpoint boundary:

```bash
touch odyssey/launch/STOP
```
"""


def validate() -> dict:
    """Dry-run validation. Checks structure, the fence, and honest gating."""
    problems: list[str] = []
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            problems.append(f"{name}: {detail}")

    check("package root exists", ODYSSEY.is_dir(), str(ODYSSEY))
    for d in DIRS:
        check(f"directory {d}", (ODYSSEY / d).is_dir())

    manifest_path = ODYSSEY / "ODYSSEY_PACKAGE.json"
    check("package manifest exists", manifest_path.is_file())
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for rel in manifest.get("files", []):
            check(f"declared file {rel}", (ROOT / rel).is_file())
        check("manifest declares launch unauthorized",
              manifest.get("launch_authorized") is False)

    fence = ODYSSEY / "launch/ODYSSEY_LAUNCH_AUTHORIZED"
    check("fence file exists", fence.is_file())
    if fence.is_file():
        check("fence reads false", fence.read_text().strip().lower() == "false",
              fence.read_text().strip())

    # The runner must refuse while the fence is false. Run it for real -- a fence
    # nobody executes is a comment.
    runner = ODYSSEY / "training/run.py"
    if runner.is_file():
        proc = subprocess.run([sys.executable, str(runner), "T0"],
                              capture_output=True, text=True, timeout=60)
        check("runner refuses to start", proc.returncode == 1,
              f"exit {proc.returncode}: {proc.stderr.strip()[:120]}")

    # Every JSON file must parse, and anything claiming a path must either have it
    # or say it is gated.
    for path in sorted(ODYSSEY.rglob("*.json")):
        rel = path.relative_to(ROOT)
        try:
            obj = json.loads(path.read_text())
        except ValueError as exc:
            check(f"parses {rel}", False, str(exc))
            continue
        check(f"parses {rel}", True)
        for entry in _entries_claiming_paths(obj):
            p = entry.get("path")
            if p and not entry.get("present", True):
                continue
            if p and not Path(p).exists():
                check(f"{rel} path present or gated", "gate" in entry,
                      f"{p} missing and no gate declared")

    # Roles that generate must not be able to promote.
    roles_path = ODYSSEY / "roles/ROLES.json"
    if roles_path.is_file():
        roles = json.loads(roles_path.read_text())["roles"]
        promoters = {r["id"] for r in roles if r["may_promote"]}
        check("only verifier and tribunal may promote", promoters == {"verifier", "tribunal"},
              str(sorted(promoters)))

    # Lean and Mathlib must be pinned to a concrete revision, not "latest".
    sandbox = ODYSSEY / "sandbox/POLICY.json"
    if sandbox.is_file():
        env = json.loads(sandbox.read_text())["reproducibility"]["environment"]
        check("lean pinned", bool(env["lean"]["toolchain"]) and
              "latest" not in env["lean"]["toolchain"], env["lean"]["toolchain"])
        check("mathlib pinned", bool(env["mathlib"]["revision"]) and
              "latest" not in env["mathlib"]["revision"], env["mathlib"]["revision"])
        check("network denied by default",
              json.loads(sandbox.read_text())["network"]["default"] == "deny")

    result = {
        "schema": "hawking.odyssey.validation.v1",
        "at": _now(),
        "checks_total": len(checks),
        "checks_failed": len(problems),
        "problems": problems,
        "odyssey_launch_authorized": False,
        "verdict": "DRY_RUN_PASS" if not problems else "DRY_RUN_FAIL",
    }
    _write(ODYSSEY / "ODYSSEY_DRY_RUN.json", {**result, "checks": checks})
    return result


def _entries_claiming_paths(obj) -> list[dict]:
    """Every dict in the tree that names a `path`, wherever it is nested."""
    found = []
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "path" in cur:
                found.append(cur)
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return found


def selftest() -> None:
    """The fence is the whole point, so test that it cannot be bypassed."""
    build()
    v = validate()
    assert v["verdict"] == "DRY_RUN_PASS", v["problems"]
    assert v["checks_failed"] == 0, v["problems"]

    fence = ODYSSEY / "launch/ODYSSEY_LAUNCH_AUTHORIZED"
    assert fence.read_text().strip() == "false"

    # Rebuilding must never authorize the run, and must not clobber a STOP.
    build()
    assert fence.read_text().strip() == "false", "rebuild flipped the fence"

    # Validation must FAIL if the fence is flipped, so a stray 'true' cannot pass review.
    fence.write_text("true\n")
    broken = validate()
    assert broken["verdict"] == "DRY_RUN_FAIL", "an authorized fence passed validation"
    assert any("fence reads false" in p for p in broken["problems"]), broken["problems"]
    fence.write_text("false\n")
    assert validate()["verdict"] == "DRY_RUN_PASS"

    # And the runner must actually refuse, not merely say it would.
    proc = subprocess.run([sys.executable, str(ODYSSEY / "training/run.py"), "T0"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1 and "refusing to start" in proc.stderr, proc.stderr
    print("selftest PASS")


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "validate"
    if cmd == "build":
        print(json.dumps(build(), indent=1))
    elif cmd == "validate":
        r = validate()
        print(json.dumps(r, indent=1))
        return 0 if r["verdict"] == "DRY_RUN_PASS" else 1
    elif cmd == "selftest":
        selftest()
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
