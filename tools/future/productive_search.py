#!/usr/bin/env python3
"""G085: launch count is not autonomy quality.

S025 §21-26. The model-bearing run four launched 51 units, made 155 decisions
and repeated no ask - a real improvement over 4 launches and 94 identical
questions. Then the launch list: 44 of the 51 were WU.HAWKING.health_probe.NNN.

    51 launches, ~7 real questions.

That is ACTIVITY WITHOUT FRONTIER DIVERSITY, and it is what the no-idle law
cannot see. A daemon launching 100 variants of one trivial probe is less
autonomous than one running four decisive experiments across four uncertainties.

So the complement to NO_RUNNABLE_IDLE is NO_LOW_INFORMATION_BUSY_LOOP, and it
needs an identity that is not a prompt hash. A causal question is what a unit is
FOR: its objective, the hypothesis it moves, the discriminator it applies and the
evidence it would change. Two units asking "is the resident healthy?" with
different numeric suffixes are ONE question asked twice.

    python3 tools/future/productive_search.py --build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/productive_search.py"
RECEIPT_NAME = "PRODUCTIVE_SEARCH.json"

TIMELINE_REL = "receipts/future/MODEL_BEARING_TIMELINE.json"

DEGENERATE = "DEGENERATE_SEARCH"
PRODUCTIVE = "PRODUCTIVE_SEARCH"

# A family may hold at most this share of a window before the search is
# degenerate. 44 of 51 is 0.86; four decisive experiments across four
# uncertainties is 0.25.
DOMINANT_FAMILY_MAX_SHARE = 0.50
MIN_DISTINCT_QUESTIONS = 3

_INDEX = re.compile(r"[._-]\d+$")

# Health probes are useful and are not the mission. A repeat needs one of these.
HEALTH_PROBE_JUSTIFICATIONS = (
    "changed_state",
    "elapsed_risk_interval",
    "new_mutation",
    "observed_anomaly",
)


class SearchRefused(RuntimeError):
    """A question identity built from the wrong thing, or a window with no work."""


def causal_question_id(unit: Mapping[str, Any]) -> str:
    """Objective, hypothesis, discriminator, evidence changed. NEVER a prompt hash.

    A prompt hash makes health_probe.007 and .008 two questions. They are one:
    same objective, same hypothesis, same discriminator, same evidence moved.
    """
    parts = [
        str(unit.get("objective") or _INDEX.sub("", str(unit.get("id") or ""))),
        str(unit.get("hypothesis_family") or unit.get("family") or ""),
        str(unit.get("discriminator") or unit.get("verifier") or ""),
        str(unit.get("evidence_changed") or unit.get("frontier_id") or ""),
    ]
    if not any(p.strip() for p in parts):
        raise SearchRefused(
            "a work unit with no objective, hypothesis, discriminator or frontier "
            "has no causal identity; refusing to hash its prompt instead"
        )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def question_family(unit: Mapping[str, Any]) -> str:
    """Readable family name: the id with its index stripped."""
    return _INDEX.sub("", str(unit.get("id") or "")) or "UNKNOWN"


def classify_window(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One rolling window. Launches are the denominator; questions are the point."""
    if not units:
        raise SearchRefused("an empty window is not evidence of productive search")
    qids = [causal_question_id(u) for u in units]
    fams = [question_family(u) for u in units]
    counts = Counter(fams)
    dominant, n_dominant = counts.most_common(1)[0]
    n_distinct = len(set(qids))
    share = n_dominant / len(units)
    reasons = []
    if share > DOMINANT_FAMILY_MAX_SHARE:
        reasons.append(
            f"{dominant} is {n_dominant} of {len(units)} launches "
            f"({share:.0%}), over the {DOMINANT_FAMILY_MAX_SHARE:.0%} bar"
        )
    if n_distinct < MIN_DISTINCT_QUESTIONS:
        reasons.append(
            f"only {n_distinct} distinct causal questions in {len(units)} launches"
        )
    return {
        "n_launches": len(units),
        "n_distinct_causal_questions": n_distinct,
        "dominant_family": dominant,
        "dominant_family_share": round(share, 4),
        "families": dict(counts),
        "verdict": DEGENERATE if reasons else PRODUCTIVE,
        "why": reasons or ["multiple causal questions, no family dominates"],
        "bar": (
            f"a family may hold at most {DOMINANT_FAMILY_MAX_SHARE:.0%} of a "
            f"window and a window needs at least {MIN_DISTINCT_QUESTIONS} distinct "
            "causal questions. Launches per hour is not the metric."
        ),
    }


def health_probe_budget(
    units: Sequence[Mapping[str, Any]],
    *,
    justifications: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """A repeated health probe must say why it ran again."""
    just = dict(justifications or {})
    probes = [u for u in units if "health_probe" in str(u.get("id") or "")]
    unjustified = []
    for i, probe in enumerate(probes):
        if i == 0:
            continue  # the first is always allowed
        why = just.get(str(probe.get("id")))
        if why not in HEALTH_PROBE_JUSTIFICATIONS:
            unjustified.append(str(probe.get("id")))
    return {
        "n_probes": len(probes),
        "n_unjustified_repeats": len(unjustified),
        "unjustified": unjustified[:12],
        "accepted_justifications": list(HEALTH_PROBE_JUSTIFICATIONS),
        "policy": (
            "the first probe in a window is free; every repeat needs changed "
            "state, an elapsed risk interval, a new mutation or an observed "
            "anomaly. Cheap and safe is not a reason to run something again."
        ),
    }


def units_from_timeline(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for event in doc.get("events") or []:
        if str(event.get("kind")) != "workunit_launched":
            continue
        payload = event.get("payload") or {}
        unit = payload.get("unit") if isinstance(payload.get("unit"), Mapping) else payload
        if isinstance(unit, Mapping) and unit.get("id"):
            out.append(dict(unit))
    return out


def run_four() -> dict[str, Any] | None:
    """The 51/44/7 window this obligation was written about."""
    path = REPO / TIMELINE_REL
    if not path.is_file():
        return None
    units = units_from_timeline(json.loads(path.read_text()))
    if not units:
        return None
    return {
        "source": TIMELINE_REL,
        "window": classify_window(units),
        "health_probes": health_probe_budget(units),
    }


def build() -> dict[str, Any]:
    live = run_four()
    return {
        "schema": "hawking.future.productive_search.v1",
        "version": 1,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "invariants": {
            "NO_RUNNABLE_IDLE": "necessary, and already enforced",
            "NO_LOW_INFORMATION_BUSY_LOOP": (
                "the missing complement. The resident must stay busy on useful "
                "UNCERTAINTY, not merely busy."
            ),
        },
        "identity_rule": (
            "causal_question_id derives from objective, hypothesis family, "
            "discriminator and evidence changed - never a prompt hash. "
            "health_probe.007 and .008 are ONE question asked twice."
        ),
        "live_window": live,
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. The window verdict "
            "is computed from launched unit identities in a timeline; it says "
            "nothing about whether the questions asked were the RIGHT ones, only "
            "whether they were different ones."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps(doc["live_window"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
