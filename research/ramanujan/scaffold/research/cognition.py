"""Executable cognition: Research Object Graph, Cheapest Falsifier, Calibration, Capsules.

`RAMANUJAN_COGNITION_REGISTER.json` specifies thirteen mechanisms with kill conditions and
self-deception tests.  Four of them are implementable now against fixtures, and are the four
that carry the most weight:

  Research Object Graph  -- what is connected to what, so a refutation propagates instead of
                            leaving dependents quietly standing
  Cheapest Falsifier     -- try to kill a claim before trying to prove it
  Self-Model/Calibration -- score confidence against verifier events only, never self-report
  Research Capsule       -- package an investigation so it reproduces

The Cheapest Falsifier is the one with a debt attached.  Math-Preserve was sealed with
282/282 shards and six green gates, and a single forward pass on "2 + 2 =" refuted the whole
substrate.  That check cost seconds and was not run for weeks.  This mechanism exists so
that never repeats, and its own self-deception test is refutation RATE -- a falsifier suite
that never refutes anything is not cheap, it is empty.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

from ramanujan.evidence import Tier


# --------------------------------------------------------------------------
# Research Object Graph
# --------------------------------------------------------------------------
@dataclass
class ResearchObjectGraph:
    """Claims, evidence and their dependencies.

    The property that matters is PROPAGATION: refuting a claim must mark everything that
    depended on it. A graph that records dependencies but does not propagate a refutation
    is a diagram, and leaves dependents standing on a foundation that is gone.
    """

    nodes: dict[str, dict] = field(default_factory=dict)
    depends_on: dict[str, set[str]] = field(default_factory=dict)

    def add(self, nid: str, kind: str, tier: Tier = Tier.ASSERTED, **meta) -> None:
        self.nodes[nid] = {"kind": kind, "tier": tier, "refuted": False, "undermined_by": [], **meta}
        self.depends_on.setdefault(nid, set())

    def depend(self, dependent: str, dependency: str) -> None:
        if dependent not in self.nodes or dependency not in self.nodes:
            raise KeyError("both nodes must exist before a dependency is declared")
        self.depends_on[dependent].add(dependency)

    def dependents_of(self, nid: str) -> set[str]:
        return {d for d, deps in self.depends_on.items() if nid in deps}

    def refute(self, nid: str, why: str) -> set[str]:
        """Refute a node and propagate transitively. Returns everything undermined."""
        self.nodes[nid]["refuted"] = True
        self.nodes[nid]["refuted_because"] = why
        undermined: set[str] = set()
        frontier = [nid]
        while frontier:
            cur = frontier.pop()
            for dep in self.dependents_of(cur):
                if dep in undermined:
                    continue
                undermined.add(dep)
                self.nodes[dep]["undermined_by"].append(cur)
                frontier.append(dep)
        return undermined

    def standing(self) -> dict[str, list[str]]:
        return {
            "live": [n for n, d in self.nodes.items() if not d["refuted"] and not d["undermined_by"]],
            "refuted": [n for n, d in self.nodes.items() if d["refuted"]],
            "undermined": [n for n, d in self.nodes.items() if d["undermined_by"] and not d["refuted"]],
        }


# --------------------------------------------------------------------------
# Cheapest Falsifier
# --------------------------------------------------------------------------
@dataclass
class Falsifier:
    name: str
    cost: float
    check: Callable[[], bool]  # True == the claim SURVIVED; False == refuted
    what_it_would_show: str


@dataclass
class CheapestFalsifier:
    """Try to kill a claim before trying to prove it, cheapest attempt first.

    Kill condition from the register: if a proposed falsifier is more expensive than the
    proof attempt it precedes, the mechanism is inverted and dies.
    """

    falsifiers: list[Falsifier] = field(default_factory=list)
    attempted: int = 0
    refutations: int = 0

    def register(self, f: Falsifier) -> None:
        self.falsifiers.append(f)

    def run(self, proof_attempt_cost: float) -> dict:
        ordered = sorted(self.falsifiers, key=lambda f: (f.cost, f.name))
        for f in ordered:
            if f.cost >= proof_attempt_cost:
                return {
                    "verdict": "MECHANISM_INVERTED",
                    "why": f"cheapest remaining falsifier {f.name!r} costs {f.cost} against a "
                           f"proof attempt costing {proof_attempt_cost}. The register kills the "
                           f"mechanism in this state rather than letting it run.",
                    "attempted": self.attempted,
                }
            self.attempted += 1
            if not f.check():
                self.refutations += 1
                return {
                    "verdict": "REFUTED",
                    "by": f.name,
                    "cost": f.cost,
                    "shows": f.what_it_would_show,
                    "attempted": self.attempted,
                    "saved": proof_attempt_cost - f.cost,
                }
        return {"verdict": "SURVIVED_ALL", "attempted": self.attempted}

    def refutation_rate(self) -> float:
        """The self-deception test. A suite that never refutes anything is empty, not cheap."""
        return self.refutations / self.attempted if self.attempted else 0.0


# --------------------------------------------------------------------------
# Self-Model and Calibration
# --------------------------------------------------------------------------
@dataclass
class Calibration:
    """Confidence scored against verifier outcomes ONLY.

    The register's self-deception test: calibration measured on self-scored outcomes is
    circular, so `record` refuses an outcome that did not come from a verifier.
    """

    predictions: list[tuple[float, bool]] = field(default_factory=list)

    def record(self, confidence: float, outcome: bool, source: str) -> None:
        if source != "verifier":
            raise ValueError(
                f"calibration outcome came from {source!r}; only 'verifier' is admissible. "
                "Scoring confidence against self-assessed success is circular."
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a probability")
        self.predictions.append((confidence, outcome))

    def brier(self) -> float | None:
        """Lower is better. A constant base-rate predictor is the kill threshold."""
        if not self.predictions:
            return None
        return sum((c - float(o)) ** 2 for c, o in self.predictions) / len(self.predictions)

    def base_rate_brier(self) -> float | None:
        """What predicting the base rate every time would score. The bar to beat."""
        if not self.predictions:
            return None
        base = sum(float(o) for _, o in self.predictions) / len(self.predictions)
        return sum((base - float(o)) ** 2 for _, o in self.predictions) / len(self.predictions)

    def beats_base_rate(self) -> bool:
        b, r = self.brier(), self.base_rate_brier()
        return b is not None and r is not None and b < r

    def overconfident(self) -> bool:
        """Mean confidence materially above realised accuracy."""
        if not self.predictions:
            return False
        mc = sum(c for c, _ in self.predictions) / len(self.predictions)
        acc = sum(float(o) for _, o in self.predictions) / len(self.predictions)
        return mc - acc > 0.10


# --------------------------------------------------------------------------
# Research Capsule
# --------------------------------------------------------------------------
def compile_capsule(investigation: dict, artifacts: dict[str, bytes]) -> dict:
    """Package an investigation so it reproduces.

    The register's self-deception test: a capsule that reproduces because it cached its
    answer proves nothing. So the capsule records input hashes and the expected result
    SEPARATELY, and `verify_capsule` recomputes rather than comparing the cached value to
    itself.
    """
    art = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(artifacts.items())}
    body = {"investigation": investigation, "artifact_hashes": art}
    return {
        "capsule_id": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16],
        **body,
        # The owner research fence is false in the present campaign.  Caller
        # content may never self-upgrade a capsule to research authority.
        "authority": "NON_PRODUCTION_AUTHORITY",
    }


def verify_capsule(capsule: dict, artifacts: dict[str, bytes]) -> tuple[bool, str]:
    """Recompute the artifact hashes from the supplied bytes and compare."""
    recomputed = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(artifacts.items())}
    if recomputed != capsule["artifact_hashes"]:
        missing = set(capsule["artifact_hashes"]) - set(recomputed)
        changed = {k for k in recomputed if k in capsule["artifact_hashes"]
                   and recomputed[k] != capsule["artifact_hashes"][k]}
        return False, f"missing={sorted(missing)} changed={sorted(changed)}"
    return True, "artifact hashes reproduce"
