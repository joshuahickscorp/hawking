#!/usr/bin/env python3.12
"""The quality contract for the deep architecture foundry.

LAW: a byte plan is not a capability. Only a real parent-vs-packed forward can
select a frontier. A 0.3 BPW file with collapsed logits is not a win.

The gate, both conditions required:
    mean symmetric KL        <= 0.10
    next-token argmax match  >= 0.95

Also recorded on every evaluation, never substituted for the gate: logit cosine,
top-5 overlap, candidate perplexity. Cosine in particular is gain-invariant, so
it cannot detect the scale failure a KL gate catches, and post-hoc scalar gain
correction on a PQ artifact is pinned at exactly 1.0 (k-means reconstruction is
a conditional mean; the residual is orthogonal to the reconstruction).

Evidence classes are ordered and one-directional: a lower class may NEVER be
reported as a higher one. Only CAPABILITY may select a parent frontier.

Thresholds are sealed by hash. A threshold may not be weakened, and least of all
after seeing a failure: `assert_not_weakened` rejects any loosening regardless of
motive, so there is no code path that can soften the gate after a bad result.
"""
from __future__ import annotations

import hashlib
import json

SCHEMA_EVALUATION = "hawking.foundry.quality_evaluation.v1"

# ── sealed thresholds ─────────────────────────────────────────────────────────
CONTRACT = {
    "schema": "hawking.foundry.quality_contract.v1",
    "max_mean_symmetric_kl": 0.10,
    "min_argmax_agreement": 0.95,
    # >= 1000 tokens: at 88 calibration tokens the median routing split is only
    # 63.6% stable and 26.1% of cells never route. 88 tokens is not evidence.
    "min_capability_tokens": 1000,
}
# Direction of each threshold: which way is STRICTER.
_BOUND_KIND = {
    "max_mean_symmetric_kl": "upper",   # smaller is stricter
    "min_argmax_agreement": "lower",    # larger is stricter
    "min_capability_tokens": "lower",   # more tokens is stricter
}


def contract_hash(contract: dict = None) -> str:
    payload = json.dumps(contract or CONTRACT, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


SEALED_CONTRACT_SHA256 = contract_hash(CONTRACT)


class ContractViolation(AssertionError):
    """A threshold was weakened, or a claim outran its evidence."""


def assert_not_weakened(proposed: dict, sealed_sha256: str = SEALED_CONTRACT_SHA256) -> dict:
    """Reject any proposed contract that loosens a sealed threshold.

    Tightening and unchanged both pass (and re-seal). Weakening always raises,
    with or without a failure in hand: there is no "after seeing the failure"
    branch to abuse.
    """
    if contract_hash(proposed) == sealed_sha256:
        return {"ok": True, "change": "unchanged", "sealed_sha256": sealed_sha256}

    weakened, tightened = [], []
    for key, kind in _BOUND_KIND.items():
        if key not in proposed:
            weakened.append(f"{key}: removed")
            continue
        old, new = CONTRACT[key], proposed[key]
        if new == old:
            continue
        loosened = new > old if kind == "upper" else new < old
        (weakened if loosened else tightened).append(f"{key}: {old} -> {new}")

    if weakened:
        raise ContractViolation(
            "quality thresholds may not be weakened: " + "; ".join(weakened)
        )
    return {
        "ok": True,
        "change": "tightened" if tightened else "non_threshold_edit",
        "tightened": tightened,
        "sealed_sha256": contract_hash(proposed),
    }


# ── evidence classes ──────────────────────────────────────────────────────────
# Ordered weakest to strongest. Index is the rank.
EVIDENCE_CLASSES = (
    "PHYSICAL",           # bytes, BPW, checksums. Says nothing about capability.
    "FUNCTIONAL_PROXY",   # weight-space recon error / cosine. Proxy, not output.
    "LAYER",              # per-layer output divergence on real activations.
    "SHORT_END_TO_END",   # real parent-vs-packed forward, short or non-holdout.
    "CAPABILITY",         # real forward, holdout split, gate metrics, >= min tokens.
)
_RANK = {name: i for i, name in enumerate(EVIDENCE_CLASSES)}

SPLITS = ("calibration", "validation", "holdout")

# Domains a frontier claim must not silently drop. A gate that passes on prose
# alone has not been shown to hold for code or long context.
PROTECTED_QUALITY_DOMAINS = ("code", "math", "multilingual", "long_context", "instruction_following")


def classify(evidence: dict) -> str:
    """Highest evidence class the supplied evidence actually supports.

    Never trust a self-declared class: this reads what the evidence contains.
    """
    ev = evidence or {}
    metrics = ev.get("metrics") or {}
    real_forward = bool(ev.get("real_parent_forward")) and bool(ev.get("real_packed_forward"))
    has_gate_metrics = "mean_symmetric_kl" in metrics and "argmax_agreement" in metrics
    tokens = int(ev.get("n_tokens") or 0)

    if (
        real_forward
        and has_gate_metrics
        and ev.get("split") == "holdout"
        and tokens >= CONTRACT["min_capability_tokens"]
        and set(ev.get("domains") or ()) >= set(PROTECTED_QUALITY_DOMAINS)
    ):
        return "CAPABILITY"
    if real_forward and has_gate_metrics:
        return "SHORT_END_TO_END"
    if ev.get("per_layer_divergence") is not None:
        return "LAYER"
    if metrics.get("weight_recon_error") is not None or metrics.get("weight_cosine") is not None:
        return "FUNCTIONAL_PROXY"
    return "PHYSICAL"


def assert_not_overclaimed(claim: str, evidence: dict) -> dict:
    """A lower class may never be reported as a higher one."""
    if claim not in _RANK:
        raise ContractViolation(f"unknown evidence class claimed: {claim!r}")
    supported = classify(evidence)
    if _RANK[claim] > _RANK[supported]:
        raise ContractViolation(
            f"overclaim: reported {claim} but the evidence only supports {supported}"
        )
    return {"claim": claim, "supported": supported, "ok": True}


def may_select_frontier(evidence: dict) -> bool:
    """Only CAPABILITY evidence may select a parent frontier."""
    return classify(evidence) == "CAPABILITY"


# ── splits ────────────────────────────────────────────────────────────────────

def assert_splits_disjoint(splits: dict) -> dict:
    """calibration / validation / holdout must not share a single example."""
    missing = [s for s in SPLITS if not splits.get(s)]
    if missing:
        raise ContractViolation(f"missing or empty splits: {missing}")
    seen, overlaps = {}, []
    for name in SPLITS:
        for item in splits[name]:
            if item in seen:
                overlaps.append((item, seen[item], name))
            seen[item] = name
    if overlaps:
        raise ContractViolation(f"split leakage: {overlaps[:5]}")
    return {"ok": True, "sizes": {s: len(splits[s]) for s in SPLITS}}


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(metrics: dict, evidence: dict = None) -> dict:
    """Apply the gate. Both conditions required; extras are recorded, not gating."""
    kl = metrics.get("mean_symmetric_kl")
    agree = metrics.get("argmax_agreement")
    if kl is None or agree is None:
        raise ContractViolation(
            "gate needs both mean_symmetric_kl and argmax_agreement; partial metrics are not a gate"
        )
    kl_ok = kl <= CONTRACT["max_mean_symmetric_kl"]
    agree_ok = agree >= CONTRACT["min_argmax_agreement"]
    supported = classify(evidence or {"metrics": metrics})
    return {
        "schema": SCHEMA_EVALUATION,
        "passed": kl_ok and agree_ok,
        "mean_symmetric_kl": kl,
        "argmax_agreement": agree,
        "kl_ok": kl_ok,
        "argmax_ok": agree_ok,
        "recorded": {
            "logit_cosine": metrics.get("logit_cosine"),
            "top5_overlap": metrics.get("top5_overlap"),
            "candidate_perplexity": metrics.get("candidate_perplexity"),
        },
        "evidence_class": supported,
        "may_select_frontier": (kl_ok and agree_ok) and supported == "CAPABILITY",
        "contract_sha256": SEALED_CONTRACT_SHA256,
    }


if __name__ == "__main__":
    print(json.dumps({"contract": CONTRACT, "sealed_sha256": SEALED_CONTRACT_SHA256}, indent=1))
