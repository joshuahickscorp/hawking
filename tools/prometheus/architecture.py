#!/usr/bin/env python3.12
"""The Prometheus components Revision 3 §7 names, around the allocation spine.

`prometheus.py` already implements the deterministic middle of the pipeline: the model
graph, the profile compiler, the allocation optimizer and the retention reporter. This is
the rest of the fourteen -- source verification, family priors, probe planning,
cartography, the sensitivity model, candidate generation, boundary validation, artifact
selection and provenance sealing -- wired into one pipeline that reports, for every stage,
whether it produced a measurement or is waiting on something.

The discipline that matters here is the same one the campaign applies everywhere: a stage
that cannot measure yet says so and names its gate, rather than emitting a plausible
number. `Stage.gated()` is the only way to produce a gated result and it *requires* a gate
string, so a stage cannot quietly return an unmeasured value that later reads as evidence.

Two things are load-bearing about what is NOT here. Cartography reports coalition SIZE and
refuses to name coalition MEMBERSHIP, because membership is a causal claim and causal
claims need intervention on a served model. And no stage computes a capability retention
number: at equal bytes, retention is the whole of Claim A, and inventing it from an
allocation plan would be assuming the result.

    python3.12 tools/prometheus/architecture.py --profile math-v1
    python3.12 tools/prometheus/architecture.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lab.layout import PROFILES_ROOT, evidence_dir, resolve_workspace_path  # noqa: E402

import prometheus as spine  # noqa: E402

GLM52_EVIDENCE = evidence_dir("glm52")
PROFILE_DIR = PROFILES_ROOT / "prometheus"

# Revision 3 §5.1: a QAT or prequantized source contaminates Claim A, because
# Prometheus would be allocating on top of an unknown prior allocation.
FULL_PRECISION_DTYPES = {"BF16", "F32", "FP32", "bfloat16", "float32"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class Stage:
    """One component's result: measured, or gated with the gate named.

    There is deliberately no way to construct a gated stage without a gate. A stage
    that returns an unmeasured value with no gate is how an allocation plan starts
    reading like an experimental result.
    """
    name: str
    status: str
    value: Any = None
    gate: str | None = None
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def measured(name: str, value: Any, *notes: str) -> "Stage":
        return Stage(name, "MEASURED", value, None, list(notes))

    @staticmethod
    def gated(name: str, gate: str, *notes: str) -> "Stage":
        if not gate:
            raise ValueError(f"{name}: a gated stage must name its gate")
        return Stage(name, "GATED", None, gate, list(notes))

    @staticmethod
    def failed(name: str, why: str) -> "Stage":
        return Stage(name, "FAILED", None, None, [why])

    def as_dict(self) -> dict:
        out = {"component": self.name, "status": self.status}
        if self.status == "MEASURED":
            out["value"] = self.value
        if self.gate:
            out["gate"] = self.gate
        if self.notes:
            out["notes"] = self.notes
        return out


# --------------------------------------------------------------------------- #
# 1. Source Verifier  (Revision 3 §5.7, gate S0)
# --------------------------------------------------------------------------- #

def source_verifier(contract_path: Path, ledger_path: Path) -> Stage:
    """S0.1-S0.6 over whatever the repo has actually sealed about the source.

    Claim-A eligibility turns on one question: were the expert tensors full precision
    before Prometheus touched them? A source that arrives already quantized cannot
    support a claim about allocation, because the vendor already made allocation
    decisions nobody recorded.
    """
    checks: dict[str, Any] = {}
    if not contract_path.is_file():
        return Stage.failed("source_verifier", f"no architecture contract at {contract_path}")
    contract = json.loads(contract_path.read_text())
    expected = (contract.get("config_validation") or {}).get("expected", {})

    dtype = str(expected.get("dtype", "")).strip()
    checks["S0.1_expert_tensor_dtype"] = {
        "declared": dtype or None,
        "full_precision": dtype in FULL_PRECISION_DTYPES,
        "rule": "Revision 3 §5.1: expert tensors must be BF16 or higher for Claim A",
    }
    checks["S0.2_architecture_graph"] = {
        "architecture": contract.get("architecture"),
        "config_validation": (contract.get("config_validation") or {}).get("status"),
        "layers": expected.get("num_hidden_layers"),
        "routed_experts": expected.get("n_routed_experts"),
        "shared_experts": expected.get("n_shared_experts"),
    }
    checks["S0.3_decoder_bit_exactness"] = {
        "oracle": "lab/operators/glm52_reference.py",
        "runtime": "crates/hawking-core/src/gravity_glm.rs",
        "status": "MEASURED_ON_FIXTURE",
        "note": "adapter agrees with the numpy oracle on a tiny model with flagship "
                "semantics; flagship-scale agreement is a separate measurement",
    }
    checks["S0.4_shard_streaming"] = {
        "status": "MEASURED",
        "note": "the flagship traversal is a streaming, resumable, content-addressed run",
    }
    checks["S0.6_non_target_tensor_census"] = (
        {"logical_weight_ledger": str(ledger_path.relative_to(REPO)),
         "present": True} if ledger_path.is_file()
        else {"present": False, "note": "no logical weight ledger sealed yet"}
    )

    eligible = checks["S0.1_expert_tensor_dtype"]["full_precision"]
    return Stage.measured(
        "source_verifier",
        {"checks": checks,
         "claim_a_source_eligible": eligible,
         "hide_eligible": True,
         "separation": "Revision 3 §8.1: HIDE eligibility and Claim-A flagship "
                       "eligibility are different questions and are not blurred"},
        "S0.5 numerical hygiene and S0.7 baseline capability need a served model",
    )


# --------------------------------------------------------------------------- #
# 2. Common Model Graph / 3. Family Prior Loader
# --------------------------------------------------------------------------- #

def common_model_graph(ledger_path: Path) -> Stage:
    if not ledger_path.is_file():
        return Stage.gated("common_model_graph",
                           "M02: a sealed logical weight ledger for the flagship",
                           "the graph is derived from the ledger, not from prose")
    groups = spine.load_ledger_graph(ledger_path)
    census = spine.census(groups)
    return Stage.measured("common_model_graph",
                          {"groups": len(groups), "census": census})


def family_prior_loader(family: str) -> Stage:
    """Priors transfer within a family; across families they are a hypothesis.

    There is no sealed prior for this family yet, and inventing one would put a guess
    where a measurement belongs -- the prior would then be indistinguishable from the
    cartography it is supposed to accelerate.
    """
    return Stage.gated(
        "family_prior_loader",
        "M10: a sealed cartography result for at least one member of this family",
        f"family={family}",
        "priors reduce probe cost; they never substitute for a measurement",
    )


# --------------------------------------------------------------------------- #
# 4. Profile Compiler
# --------------------------------------------------------------------------- #

def profile_compiler() -> Stage:
    """Compile every built-in profile and hash it, so an arm can name its policy exactly."""
    compiled = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        profile = json.loads(path.read_text())
        compiled.append({
            "name": profile["name"],
            "control": bool(profile.get("control")),
            "categories": len(profile.get("category_tiers", {})),
            "tiers": sorted(profile.get("tier_rates", {})),
            "profile_hash": _sha256_json(profile),
        })
    required = {"general-v1", "math-v1", "uniform-v1", "random-v1"}
    have = {c["name"] for c in compiled}
    missing = sorted(required - have)
    stage = Stage.measured("profile_compiler",
                           {"profiles": compiled, "count": len(compiled),
                            "equal_budget_arms_present": not missing})
    if missing:
        stage.notes.append(f"equal-budget Claim A needs these arms: {missing}")
    return stage


# --------------------------------------------------------------------------- #
# 5. Probe Planner
# --------------------------------------------------------------------------- #

PROBE_CLASSES = [
    ("structural", "shapes, counts, connectivity", False),
    ("precision", "per-tensor tolerance to representation loss", False),
    ("ablation", "remove and measure", True),
    ("activation", "what fires, how often, at what margin", True),
    ("behavioural", "task outcome under intervention", True),
    ("long_horizon", "where compressed reasoning departs from the parent", True),
    ("formal", "does the formalizer still produce a faithful statement", True),
    ("sovereignty", "false refusal and boundary error inside the profile", True),
]


def probe_planner(groups_present: bool) -> Stage:
    """Plan the probes; the ones needing a served model are planned, not run."""
    plan = [{"class": name, "measures": what, "needs_served_model": served}
            for name, what, served in PROBE_CLASSES]
    runnable = [p["class"] for p in plan if not p["needs_served_model"]]
    if not groups_present:
        return Stage.gated("probe_planner", "M02: a sealed model graph to plan over",
                           f"plan is defined for {len(plan)} probe classes")
    return Stage.measured("probe_planner",
                          {"classes": plan, "runnable_without_served_model": runnable},
                          "activation frequency is not causal importance; the "
                          "intervention probes are the ones that carry the claim")


# --------------------------------------------------------------------------- #
# 6. Deterministic Cartography / 7. Sensitivity Model
# --------------------------------------------------------------------------- #

def cartography(groups_present: bool) -> Stage:
    """Coalition SIZE is arithmetic. Coalition MEMBERSHIP is a causal claim.

    The allocation spine already needs a size to bill bytes against, and size follows
    from the coalition fraction. Which experts belong is what the whole experiment is
    about, and answering it from activation frequency would be answering a different
    question -- frequency is not causal importance.
    """
    return Stage.gated(
        "deterministic_cartography",
        "S0.8 served runtime + intervention probes",
        "coalition SIZE is available to the allocator and is used",
        "coalition MEMBERSHIP is withheld: naming experts without intervention "
        "would make the control arm meaningless, since a byte-matched random "
        "coalition is only a control against a *measured* one",
    )


def sensitivity_model() -> Stage:
    return Stage.gated(
        "sensitivity_model",
        "M10: paired intervention results at matched bytes",
        "the model maps representation loss to capability loss per block; "
        "fitting it to anything other than measured interventions would make "
        "the optimizer's objective circular",
    )


# --------------------------------------------------------------------------- #
# 8. Allocation Optimizer / 9. Candidate Generator
# --------------------------------------------------------------------------- #

def allocation_and_candidates(ledger_path: Path, profiles: list[str],
                              coalition_fraction: float, coalition_protect_bpw: float,
                              seed: int) -> tuple[Stage, Stage]:
    if not ledger_path.is_file():
        gate = "M02: a sealed logical weight ledger for the flagship"
        return (Stage.gated("allocation_optimizer", gate),
                Stage.gated("candidate_generator", gate))
    candidates = []
    for name in profiles:
        result = spine.run_pipeline(ledger_path, name, coalition_fraction,
                                    coalition_protect_bpw, seed)
        d = result["byte_decomposition"]
        candidates.append({
            "arm": name,
            "control": result["control"],
            "complete_bpw": d["complete_bpw"],
            "physical_bytes": d["effective_total_physical_bytes"],
            "excised_logical_weights": d["excised_logical_weights"],
            "plan_hash": _sha256_json(result["blocks"]),
        })
    # Equal budget is the precondition for the comparison meaning anything, so it is
    # checked here rather than asserted in the write-up.
    byte_counts = {c["physical_bytes"] for c in candidates}
    spread = (max(byte_counts) - min(byte_counts)) if byte_counts else 0
    largest = max(byte_counts) if byte_counts else 0
    return (
        Stage.measured("allocation_optimizer",
                       {"arms": len(candidates), "seed": seed,
                        "coalition_fraction": coalition_fraction}),
        Stage.measured("candidate_generator",
                       {"candidates": candidates,
                        "byte_spread": spread,
                        "byte_spread_fraction": (spread / largest) if largest else 0.0,
                        "equal_budget": spread == 0},
                       "an unequal-budget comparison measures budget, not policy"),
    )


def equal_budget_solver(ledger_path: Path, arms: list[str], target_bytes: int,
                        coalition_fraction: float, coalition_protect_bpw: float,
                        seed: int, den: int = 64) -> Stage:
    """Solve each arm's main rate so every arm spends the same bytes.

    Claim A is a statement about equal storage budget. Arms written by hand do not land
    on the same total: the conditioned profiles pin the embedding and head native and
    excise the vision tower, the uniform profile does neither, and the two effects do not
    cancel. Comparing them as written measures the budget difference and attributes it to
    the policy -- and it does so in the direction that flatters the conditioned arms,
    which is the worst way for a bug to be wrong.

    So the budget is the fixed quantity and the main rate is solved for it. Rates stay
    exact rationals over `den` rather than floats, because the byte ledger is exact and a
    rate that cannot be written down is a rate that cannot be reproduced.
    """
    if not ledger_path.is_file():
        return Stage.gated("equal_budget_solver",
                           "M02: a sealed logical weight ledger for the flagship")

    def bytes_for(arm: str, num: int) -> int:
        profile = spine.load_profile(arm)
        base = profile["_base"] if profile.get("control") else profile
        base["tier_rates"] = dict(base["tier_rates"])
        base["tier_rates"]["T1"] = {"num": num, "den": den}
        if profile.get("control"):
            profile["_base"] = base
        groups = spine.load_ledger_graph(ledger_path)
        blocks = spine.allocate(groups, profile, coalition_fraction,
                                coalition_protect_bpw, seed)
        return spine.byte_decomposition(blocks)["effective_total_physical_bytes"]

    solved = []
    for arm in arms:
        # Monotone in `num`, so bisect on the integer numerator directly. The
        # upper bound has to be generous rather than "a rate a profile would
        # plausibly declare": the uniform arm spends its budget on tensors the
        # conditioned arms carry natively, so matching their total can require a
        # main rate well above one bit. A bound that stops short does not report
        # an impossible match -- it reports a false one.
        lo, hi = 1, den * 16
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            total = bytes_for(arm, mid)
            gap = abs(total - target_bytes)
            if best is None or gap < best[2]:
                best = (mid, total, gap)
            if total < target_bytes:
                lo = mid + 1
            else:
                hi = mid - 1
        num, total, gap = best
        solved.append({
            "arm": arm,
            "solved_main_rate": {"num": num, "den": den},
            "solved_main_bpw": num / den,
            "physical_bytes": total,
            "residual_bytes": total - target_bytes,
            "residual_fraction": (total - target_bytes) / target_bytes,
        })

    residuals = [abs(s["residual_fraction"]) for s in solved]
    worst = max(residuals) if residuals else 0.0
    stage = Stage.measured(
        "equal_budget_solver",
        {"target_bytes": target_bytes, "denominator": den, "arms": solved,
         "worst_residual_fraction": worst,
         # A residual under a tenth of a percent is below the granularity of the
         # rate ladder itself, so tightening it further would be false precision.
         "equal_budget_achievable": worst < 1e-3},
        "the byte budget is fixed and the rate is solved for it, not the reverse",
    )
    if worst >= 1e-3:
        stage.notes.append(
            f"worst arm is off budget by {worst*100:.2f}%; a finer denominator than "
            f"{den} is needed before this comparison is honest")
    return stage


# --------------------------------------------------------------------------- #
# 10. The Forge / 11. Boundary Validator
# --------------------------------------------------------------------------- #

def forge_status() -> Stage:
    impl = REPO / "tools/sovereignty/sovereignty.py"
    return Stage.measured(
        "forge",
        {"implementation": str(impl.relative_to(REPO)) if impl.is_file() else None,
         "F0_diagnose": "PARTIAL: continuity, prompt transformation, refusal "
                        "attribution and the limit registry are deterministic and built",
         "F1_locate": "GATED: needs a served model",
         "F2_restore": "GATED: needs F1",
         "F3_validate": "GATED: needs F2",
         "F4_seal": "GATED: needs F3"},
        "the objective is maximum correctly authorized expression, not answer rate",
    )


def boundary_validator() -> Stage:
    return Stage.gated(
        "boundary_validator",
        "S0.8 served runtime + the permitted/boundary prompt sets",
        "false_refusal_rate and boundary_error_rate are the two metrics that cannot "
        "be computed without generating text, and neither is ever estimated",
    )


# --------------------------------------------------------------------------- #
# 12. Artifact Selector / 13. Retention Reporter / 14. Provenance Sealer
# --------------------------------------------------------------------------- #

def artifact_selector(candidates: Stage) -> Stage:
    """Smallest is not a winner. A candidate that crosses a capability threshold,
    causes hidden fallback, raises false refusals, breaks prompt transparency or
    violates byte accounting has not won anything."""
    return Stage.gated(
        "artifact_selector",
        "M10: capability results at equal bytes",
        "the ranking key is capability retention at fixed bytes, and no retention "
        "number exists yet",
        f"{len((candidates.value or {}).get('candidates', []))} candidate plans are ready "
        "to be ranked once retention is measured",
    )


def retention_reporter(candidates: Stage) -> Stage:
    if candidates.status != "MEASURED":
        return Stage.gated("retention_reporter", candidates.gate or "M02")
    return Stage.measured(
        "retention_reporter",
        {"byte_accounting": [
            {"arm": c["arm"], "complete_bpw": c["complete_bpw"],
             "physical_bytes": c["physical_bytes"]}
            for c in candidates.value["candidates"]],
         "capability_retention": None},
        "byte accounting is complete; capability retention is deliberately null "
        "rather than estimated -- at equal bytes, retention IS Claim A",
    )


def provenance_sealer(stages: list[Stage], extra: dict) -> Stage:
    payload = {"stages": [s.as_dict() for s in stages], **extra}
    return Stage.measured("provenance_sealer",
                          {"seal_sha256": _sha256_json(payload),
                           "sealed_at": _now(),
                           "stages_measured": sum(1 for s in stages if s.status == "MEASURED"),
                           "stages_gated": sum(1 for s in stages if s.status == "GATED"),
                           "stages_failed": sum(1 for s in stages if s.status == "FAILED")})


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #

EQUAL_BUDGET_ARMS = ["uniform-v1", "general-v1", "math-v1", "random-v1"]


def run(ledger_path: Path, contract_path: Path, coalition_fraction: float,
        coalition_protect_bpw: float, seed: int) -> dict:
    # Historic receipts and runbooks still name root-level campaign paths.  Read
    # them through the layout bridge rather than requiring consumers to rewrite
    # sealed metadata after a workspace move.
    ledger_path = resolve_workspace_path(ledger_path)
    contract_path = resolve_workspace_path(contract_path)
    src = source_verifier(contract_path, ledger_path)
    graph = common_model_graph(ledger_path)
    priors = family_prior_loader("glm")
    profiles = profile_compiler()
    probes = probe_planner(graph.status == "MEASURED")
    carto = cartography(graph.status == "MEASURED")
    sens = sensitivity_model()
    alloc, cands = allocation_and_candidates(
        ledger_path, EQUAL_BUDGET_ARMS, coalition_fraction, coalition_protect_bpw, seed)
    # Match every arm to the largest as-written arm. The uniform arm then has to
    # exceed the one bit its profile declares, because the conditioned arms spend
    # part of the same budget on a natively-carried embedding and head. That is
    # the comparison Claim A actually asks for: same bytes, spent differently.
    if cands.status == "MEASURED":
        target = max(c["physical_bytes"] for c in cands.value["candidates"])
        budget = equal_budget_solver(ledger_path, EQUAL_BUDGET_ARMS, target,
                                     coalition_fraction, coalition_protect_bpw, seed)
    else:
        budget = Stage.gated("equal_budget_solver", cands.gate or "M02")
    forge = forge_status()
    boundary = boundary_validator()
    selector = artifact_selector(cands)
    retention = retention_reporter(cands)

    stages = [src, graph, priors, profiles, probes, carto, sens, alloc, cands,
              budget, forge, boundary, selector, retention]
    seal = provenance_sealer(stages, {"seed": seed})
    stages.append(seal)

    return {
        "schema": "hawking.prometheus.architecture.v1",
        "at": _now(),
        "revision": "Hawking Prometheus & Ramanujan Canonical Master Plan, Revision 3 §7",
        "components_required": 14,
        "components_present": len(stages),
        "claim_a": {
            "status": "NOT_SEALED",
            "blocking": sorted({s.gate for s in stages if s.gate}),
            "note": "Claim A is a capability result at equal bytes. The allocation plans "
                    "are ready and byte-matched; nothing here asserts a retention number.",
        },
        "stages": [s.as_dict() for s in stages],
    }


def selftest() -> None:
    """The gating discipline is the point, so test that it cannot be evaded."""
    try:
        Stage.gated("x", "")
        raise AssertionError("a gated stage without a gate was accepted")
    except ValueError:
        pass

    s = Stage.gated("x", "M10")
    assert "value" not in s.as_dict(), "a gated stage must not carry a value"
    assert s.as_dict()["gate"] == "M10"
    assert "value" in Stage.measured("y", 1).as_dict()

    result = run(GLM52_EVIDENCE / "GLM52_LOGICAL_WEIGHT_LEDGER.json",
                 GLM52_EVIDENCE / "GLM52_ARCHITECTURE_CONTRACT.json", 0.05, 4.0, 20260724)
    assert result["components_present"] >= 14, result["components_present"]
    assert result["claim_a"]["status"] == "NOT_SEALED"

    by_name = {s["component"]: s for s in result["stages"]}
    # Cartography must never report membership, and retention must never be a number.
    assert by_name["deterministic_cartography"]["status"] == "GATED"
    ret = by_name["retention_reporter"]
    if ret["status"] == "MEASURED":
        assert ret["value"]["capability_retention"] is None, "retention was estimated"
    # Every gated stage names a gate, and no gated stage smuggles a value.
    for s in result["stages"]:
        if s["status"] == "GATED":
            assert s.get("gate"), s
            assert "value" not in s, s
    # The source verifier must answer the Claim-A precision question either way.
    if by_name["source_verifier"]["status"] == "MEASURED":
        assert "claim_a_source_eligible" in by_name["source_verifier"]["value"]
    print("selftest PASS")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prometheus architecture (Revision 3 §7)")
    ap.add_argument("--ledger", default=str(GLM52_EVIDENCE / "GLM52_LOGICAL_WEIGHT_LEDGER.json"))
    ap.add_argument("--contract", default=str(GLM52_EVIDENCE / "GLM52_ARCHITECTURE_CONTRACT.json"))
    ap.add_argument("--coalition-fraction", type=float, default=0.05)
    ap.add_argument("--coalition-protect-bpw", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    result = run(Path(a.ledger), Path(a.contract), a.coalition_fraction,
                 a.coalition_protect_bpw, a.seed)
    text = json.dumps(result, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
        print(f"{result['components_present']} components -> {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
