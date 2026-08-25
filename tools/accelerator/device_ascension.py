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
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

REPO = Path(__file__).resolve().parents[2]

STAGES = ("fingerprint", "benchmark", "roof_measurement", "workload_census",
          "kernel_selection", "autotune", "memory_plan", "dispatch_plan",
          "concurrency_sweep", "sustained_qualification", "adp_seal")

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
