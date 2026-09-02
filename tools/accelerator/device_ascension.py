"""Device Ascension — the per-machine campaign. FRONT G (G049, steer S015 §23).

Eleven stages from fingerprint to ADP seal. The stages are real functions rather
than a narrative, and a stage that has not run reports NOT_RUN rather than being
skipped silently.

The machine-specific win this looks for is the one the steer points at in §125:
APPLE UNIFIED MEMORY IS A WEAPON -- what CUDA-era copy can disappear? A CUDA port
allocates device memory, copies host to device, computes, and copies back. On this
architecture there is no separate device memory, so both copies are avoidable. That
win is machine-specific in the strict sense P6 requires: it does not exist on a
discrete-GPU machine at all, because there the copies are not optional.

Arrival cycle (added around the eleven ADP stages, not instead of them):
  discover -> characterize -> economics -> select -> promote -> invalidate-on-change

The eleven ADP stages qualify a profile on a machine that already has laws.
The cycle is how a NEW machine (this M3 Ultra first; a future M-series, FPGA
or DGX later) becomes a textbook: live genome, rebuild economics, pick a
resident as a DECISION RECORD (never an install), and invalidate that
decision when the genome digest moves. tools/future/device_ascension_pipeline.py
is a STATIC sidecar around the same idea; this module actually probes.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

REPO = Path(__file__).resolve().parents[2]

STAGES = ("fingerprint", "benchmark", "roof_measurement", "workload_census",
          "kernel_selection", "autotune", "memory_plan", "dispatch_plan",
          "concurrency_sweep", "sustained_qualification", "adp_seal")

# Arrival cycle. Orthogonal to STAGES. A genome change is a recompile trigger.
CYCLE = ("discover", "characterize", "economics", "select", "promote", "invalidate")

PROFILES = ("INTERACTIVE", "MAX_THROUGHPUT", "MIN_LATENCY", "MIN_MEMORY",
            "SUSTAINED", "BATTERY_AWARE", "HCLI_AUTONOMOUS", "ODYSSEY_RESEARCH")

# NOTE: this is the §26 TUNING-SCOPE vocabulary (general / SoC / exact machine) and it
# is NOT the §80 knowledge-level vocabulary used by receipt.py (INSTANCE, MODEL_FAMILY,
# ARCHITECTURE, ...). They share the phrase "knowledge level" and mean different things:
# §26 asks how widely a TUNING RESULT applies, §80 asks how far a FINDING may be
# promoted. Conflating them was a real bug -- an ADP tuning scope was passed straight
# into a receipt and rejected. Renamed here so the two cannot be swapped by accident.
TUNING_SCOPES = ("EXACT_MACHINE", "SOC_FAMILY", "APPLE_GENERAL")
KNOWLEDGE_LEVELS = TUNING_SCOPES  # kept so existing callers still resolve


def _machine_genome():
    """Import the sibling. Accelerator tests put this directory on sys.path."""
    try:
        import machine_genome as mg
        return mg
    except ImportError:
        accel = str(Path(__file__).resolve().parent)
        if accel not in sys.path:
            sys.path.insert(0, accel)
        import machine_genome as mg
        return mg


def _device_profiles():
    """Import tools/odyssey/device_profiles.py. A module import is not a call
    site; callers below invoke economics_from_genome / select_resident."""
    try:
        from tools.odyssey import device_profiles as dp
        return dp
    except ImportError:
        odyssey = str(REPO / "tools/odyssey")
        if odyssey not in sys.path:
            sys.path.insert(0, odyssey)
        import device_profiles as dp
        return dp


@dataclass
class Ascension:
    machine: dict[str, Any]
    stages: dict[str, Any] = field(default_factory=dict)

    def record(self, stage: str, payload: Any) -> None:
        if stage not in STAGES:
            raise ValueError(f"{stage!r} is not one of the eleven stages")
        self.stages[stage] = payload

    def not_run(self, stage: str, reason: str) -> None:
        self.record(stage, {"status": "NOT_RUN", "reason": reason})

    def completed(self) -> list[str]:
        return [s for s in STAGES
                if s in self.stages
                and not (isinstance(self.stages[s], dict)
                         and self.stages[s].get("status") == "NOT_RUN")]

    # A profile may only be sealed on evidence that speaks to what it CLAIMS. Sealing
    # MAX_THROUGHPUT on sustained evidence alone was a real hole: sustained load says
    # nothing about how the kernel behaves against other work.
    PROFILE_REQUIRES: ClassVar[dict[str, tuple[str, ...]]] = {
        "SUSTAINED": ("sustained_qualification",),
        "MAX_THROUGHPUT": ("sustained_qualification", "concurrency_sweep"),
        "HCLI_AUTONOMOUS": ("sustained_qualification", "concurrency_sweep"),
        "MIN_LATENCY": ("dispatch_plan",),
        "MIN_MEMORY": ("memory_plan",),
        "INTERACTIVE": ("dispatch_plan",),
        "ODYSSEY_RESEARCH": ("sustained_qualification",),
        "BATTERY_AWARE": ("sustained_qualification",),
    }

    def may_seal_adp(self, profile: str = "SUSTAINED") -> tuple[bool, str]:
        """A production ADP needs sustained evidence, not a microbenchmark (§29) -- and
        it needs the evidence its OWN profile depends on."""
        for stage in self.PROFILE_REQUIRES.get(profile, ("sustained_qualification",)):
            got = self.stages.get(stage)
            if not isinstance(got, dict) or got.get("status") == "NOT_RUN":
                return False, (f"profile {profile} requires stage {stage!r}, which has "
                               f"not run; sealing it would claim something no "
                               f"measurement here supports")
        sust = self.stages.get("sustained_qualification")
        if not isinstance(sust, dict) or sust.get("status") == "NOT_RUN":
            return False, ("sustained_qualification has not run; §29 forbids sealing a "
                           "production ADP on microbenchmark evidence")
        if not sust.get("passed"):
            return False, "sustained_qualification did not pass"
        return True, "sustained evidence present"

    def seal(self, profile: str, config: dict[str, Any],
             knowledge_level: str) -> dict[str, Any]:
        if profile not in PROFILES:
            raise ValueError(f"{profile!r} is not a named profile; §24 forbids vague "
                             f"'optimized settings'")
        if knowledge_level not in TUNING_SCOPES:
            raise ValueError(f"{knowledge_level!r} is not a §26 tuning scope; the §80 "
                             f"knowledge levels used by receipts are a DIFFERENT "
                             f"vocabulary and must not be passed here")
        ok, why = self.may_seal_adp(profile)
        return {
            "adp": f"ADP-{self.machine.get('soc','UNKNOWN').replace(' ','')}-{profile}",
            "profile": profile,
            "config": config,
            "knowledge_level": knowledge_level,
            "production_sealed": ok,
            "status": "SEALED" if ok else "PROVISIONAL",
            "why": why,
            "stages_completed": self.completed(),
            "stages_not_run": [s for s in STAGES if s not in self.completed()],
            "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


# --------------------------------------------------------------------------- arrival cycle
#
# discover -> characterize -> economics -> select -> promote -> invalidate
# Each function is a real call, not a narrative. Selection is a decision
# record. Nothing here installs a resident.


def discover(*, live: bool = True) -> dict[str, Any]:
    """Stage 1. Identity of THIS machine. Calls machine_genome.discover_identity."""
    mg = _machine_genome()
    identity = mg.discover_identity() if live else {
        "status": "BLOCKED",
        "reason": "live identity probe skipped",
        "evidence_tier": "STATIC",
    }
    return {
        "stage": "discover",
        "identity": identity,
        "evidence_tier": "STATIC",
        "called": "machine_genome.discover_identity",
    }


def characterize(discovery: Mapping[str, Any] | None = None, *,
                 contended: bool = True,
                 contention_note: str = (
                     "live HCLI daemon and hf download workers; genome probes "
                     "are identity plus bounded read-only storage samples"
                 )) -> dict[str, Any]:
    """Stage 2. Full genome: CPU/GPU/UMA/ANE/storage/network + declared FPGA/DGX.

    Calls machine_genome.build, which itself calls probe_storage / probe_ane /
    probe_network / measure_bandwidth.
    """
    mg = _machine_genome()
    genome = mg.build(contended=contended, contention_note=contention_note)
    return {
        "stage": "characterize",
        "genome": genome,
        "genome_digest": genome.get("genome_digest"),
        "backend_maturity": genome.get("backend_maturity"),
        "evidence_tier": "HARDWARE_MEASURED",
        "called": "machine_genome.build",
        "discovery": discovery,
    }


def economics(characterization: Mapping[str, Any],
              *, profile: str = "INTERACTIVE") -> dict[str, Any]:
    """Stage 3. Rebuild science economics from the live genome.

    Calls device_profiles.economics_from_genome. Storage rates and UMA
    capacity change what is affordable; that is a COST_MODEL overlay on
    HARDWARE_MEASURED inputs, never a fabricated device number.
    """
    dp = _device_profiles()
    genome = characterization.get("genome") or {}
    econ = dp.economics_from_genome(genome, profile=profile)
    return {
        "stage": "economics",
        "economics": econ,
        "genome_digest": genome.get("genome_digest"),
        "evidence_tier": "COST_MODEL",
        "called": "device_profiles.economics_from_genome",
    }


def select(econ: Mapping[str, Any], *,
           candidates: list[dict[str, Any]] | None = None,
           profile: str = "INTERACTIVE") -> dict[str, Any]:
    """Stage 4. Pick a resident as a decision record. Does not install."""
    dp = _device_profiles()
    inner = econ.get("economics") if "economics" in econ else econ
    decision = dp.select_resident(inner, candidates=candidates, profile=profile)
    decision["genome_digest"] = (
        decision.get("genome_digest")
        or econ.get("genome_digest")
        or (inner or {}).get("genome_digest")
    )
    return {
        "stage": "select",
        "decision": decision,
        "installed": False,
        "evidence_tier": decision.get("evidence_tier") or "COST_MODEL",
        "called": "device_profiles.select_resident",
    }


def promote(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Stage 5. Stamp the decision PROVISIONAL. Never installs a resident."""
    decision = selection.get("decision") or selection
    return {
        "stage": "promote",
        "installed": False,
        "status": "PROVISIONAL",
        "decision": decision,
        "genome_digest": decision.get("genome_digest") or selection.get("genome_digest"),
        "note": (
            "Promotion is a decision record, not an installation. No resident "
            "is copied, linked, or made current by this stage."
        ),
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_tier": "STATIC",
    }


def _as_genome(bag: Mapping[str, Any]) -> Mapping[str, Any]:
    """Unwrap a characterize/cycle payload down to the genome dict."""
    if isinstance(bag.get("genome"), dict) and (
        "soc" in bag["genome"] or "domains" in bag["genome"] or "genome_digest" in bag["genome"]
    ):
        return bag["genome"]
    return bag


def genome_changed(prior_genome: Mapping[str, Any],
                   current_genome: Mapping[str, Any]) -> bool:
    """Recompile trigger. A selection bound to a prior genome is invalid when
    the live genome digest moves.

    This is the mutation point the negative test is built on: if this
    function ignores a digest mismatch, the invalidation test MUST FAIL.
    """
    mg = _machine_genome()
    prior = _as_genome(prior_genome)
    current = _as_genome(current_genome)
    prior_digest = mg.genome_digest(prior) if (prior.get("soc") or prior.get("domains")) \
        else prior.get("genome_digest")
    current_digest = mg.genome_digest(current) if (current.get("soc") or current.get("domains")) \
        else current.get("genome_digest")
    if not prior_digest or not current_digest:
        return True  # fail closed: unknown identity is a change
    return prior_digest != current_digest


def invalidate(selection: Mapping[str, Any],
               current_genome: Mapping[str, Any]) -> dict[str, Any]:
    """Stage 6. A genome change INVALIDATES a prior selection (recompile)."""
    decision = selection.get("decision") or selection
    prior = decision.get("genome") or decision
    current = _as_genome(current_genome)
    mg = _machine_genome()
    changed = genome_changed(prior, current)
    prior_digest = prior.get("genome_digest")
    if prior.get("soc") or prior.get("domains"):
        prior_digest = mg.genome_digest(prior)
    current_digest = current.get("genome_digest")
    if current.get("soc") or current.get("domains"):
        current_digest = mg.genome_digest(current)
    return {
        "stage": "invalidate",
        "invalidated": bool(changed),
        "recompile_required": bool(changed),
        "prior_genome_digest": prior_digest,
        "current_genome_digest": current_digest,
        "called": "genome_changed",
        "evidence_tier": "STATIC",
        "note": (
            "A digest mismatch is a recompile trigger: the prior selection "
            "was computed against a different machine identity and must not "
            "be reused. Same digest: the selection still holds."
        ),
    }


def run_cycle(*,
              contended: bool = True,
              contention_note: str | None = None,
              profile: str = "INTERACTIVE",
              candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """End-to-end arrival cycle on THIS machine. Selection is not an install.

    Also fills the ADP fingerprint stage from the live genome so the eleven
    stages are connected rather than forked. Remaining ADP stages stay
    NOT_RUN: this cycle does not claim a production seal.
    """
    note = contention_note or (
        "live HCLI daemon and hf download workers; genome probes are "
        "identity plus bounded read-only storage samples"
    )
    d = discover(live=True)
    c = characterize(d, contended=contended, contention_note=note)
    genome = c["genome"]
    e = economics(c, profile=profile)
    s = select(e, candidates=candidates, profile=profile)
    # Bind the live genome onto the decision so invalidate can re-hash it.
    s["decision"]["genome"] = {
        k: genome.get(k) for k in (
            "schema", "soc", "arch", "cpu_cores", "perf_cores",
            "efficiency_cores", "gpu_cores", "memory_bytes", "domains",
            "genome_digest",
        )
    }
    s["decision"]["genome_digest"] = genome.get("genome_digest")
    p = promote(s)
    inv = invalidate(s, genome)

    a = Ascension(machine=genome)
    a.record("fingerprint", {
        "soc": genome.get("soc"),
        "arch": genome.get("arch"),
        "cpu_cores": genome.get("cpu_cores"),
        "gpu_cores": genome.get("gpu_cores"),
        "memory_bytes": genome.get("memory_bytes"),
        "genome_digest": genome.get("genome_digest"),
        "called": "run_cycle.characterize -> machine_genome.build",
    })
    for stage in STAGES:
        if stage == "fingerprint":
            continue
        a.not_run(stage, "arrival cycle does not run the ADP campaign; fingerprint only")

    return {
        "schema": "hawking.accelerator.device_ascension.cycle.v1",
        "cycle": list(CYCLE),
        "stages": {
            "discover": d,
            "characterize": {k: v for k, v in c.items() if k != "genome"},
            "economics": e,
            "select": s,
            "promote": p,
            "invalidate": inv,
        },
        "genome": genome,
        "selection": s["decision"],
        "promotion": p,
        "invalidation": inv,
        "installed": False,
        "adp": {
            "fingerprint": a.stages.get("fingerprint"),
            "stages_completed": a.completed(),
            "stages_not_run": [st for st in STAGES if st not in a.completed()],
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


if __name__ == "__main__":
    result = run_cycle()
    # Drop the bulky genome mounts listing for the CLI summary; the test
    # keeps the full object.
    summary = {
        "soc": result["genome"].get("soc"),
        "genome_digest": result["genome"].get("genome_digest"),
        "backend_maturity": result["genome"].get("backend_maturity"),
        "selection": {
            k: result["selection"].get(k)
            for k in ("selected", "profile", "installed", "reason", "genome_digest")
        },
        "promotion": {"installed": result["promotion"]["installed"],
                      "status": result["promotion"]["status"]},
        "invalidation": {
            "invalidated": result["invalidation"]["invalidated"],
            "recompile_required": result["invalidation"]["recompile_required"],
        },
        "installed": result["installed"],
    }
    print(json.dumps(summary, indent=2, default=str))
