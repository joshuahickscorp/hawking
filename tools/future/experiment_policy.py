"""Typed experiment-policy representation.

This ENABLES a future learned policy. It does not select experiments.

The current deterministic / resident policy remains the authority. A
bandit, posterior sampler or ranking head is not present and must not be
inferred from these types existing.

Typed fields: option, expected payoff, information gain, discriminator,
cost, outcome, posterior / belief update.

    python3 -m pytest tools/future/test_experiment_policy.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

from typing import Any, Mapping

POLICY_AUTHORITY = (
    "The current deterministic/resident policy remains the authority. "
    "This module is a typed representation so a future learned policy can "
    "plug in. It does not rank, sample or select experiments."
)

SCHEMA = "hawking.future.experiment_policy.v1"
VERSION = 1

OPTION_KEYS = (
    "option_id",
    "action",
    "hypothesis_id",
    "expected_payoff",
    "information_gain",
    "discriminator",
    "cost",
    "outcome",
    "belief_update",
    "policy_authority",
    "learned",
)

EVIDENCE_TIERS = (
    "STATIC",
    "FUNCTIONAL_SIM",
    "COST_MODEL",
    "CYCLE_APPROX",
    "HARDWARE_MEASURED",
)


class PolicyRefused(RuntimeError):
    """A policy object is malformed, or a caller asked this module to decide."""


def _tier(value: Any, *, default: str = "STATIC") -> str:
    text = str(value or default)
    if text not in EVIDENCE_TIERS:
        return default
    return text


def expected_payoff(
    *,
    value: float | None = None,
    unit: str | None = None,
    kind: str = "EXPECTED",
    evidence_tier: str = "STATIC",
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "value": value,
        "unit": unit,
        "evidence_tier": _tier(evidence_tier),
        "note": note
        or (
            "unquantified; the field exists so a future learned policy can fill it"
            if value is None
            else None
        ),
    }


def information_gain(
    *,
    about: str,
    expected_bits: float | None = None,
    reduces_uncertainty_in: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "about": about,
        "expected_bits": expected_bits,
        "reduces_uncertainty_in": reduces_uncertainty_in,
        "note": note
        or (
            "unquantified; typed so a future learned policy can fill it"
            if expected_bits is None
            else None
        ),
    }


def discriminator(
    *,
    name: str,
    distinguishes: list[str] | tuple[str, ...] | None = None,
    cheapest_falsifier: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "distinguishes": list(distinguishes or ()),
        "cheapest_falsifier": cheapest_falsifier,
    }


def cost(
    *,
    class_: str,
    quantity: float | None = None,
    unit: str | None = None,
    evidence_tier: str = "COST_MODEL",
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "class": class_,
        "quantity": quantity,
        "unit": unit,
        "evidence_tier": _tier(evidence_tier, default="COST_MODEL"),
        "note": note,
    }


def outcome(
    *,
    option_id: str,
    status: str,
    observed: Any = None,
    hypothesis_id: str | None = None,
    evidence_tier: str = "STATIC",
) -> dict[str, Any]:
    return {
        "option_id": option_id,
        "hypothesis_id": hypothesis_id,
        "status": status,
        "observed": observed,
        "evidence_tier": _tier(evidence_tier),
        "learned": False,
    }


def option(
    *,
    option_id: str,
    action: str,
    hypothesis_id: str | None = None,
    expected_payoff_row: Mapping[str, Any] | None = None,
    information_gain_row: Mapping[str, Any] | None = None,
    discriminator_row: Mapping[str, Any] | None = None,
    cost_row: Mapping[str, Any] | None = None,
    outcome_row: Mapping[str, Any] | None = None,
    belief_update_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not option_id or not action:
        raise PolicyRefused("option_id and action are required")
    row = {
        "option_id": option_id,
        "action": action,
        "hypothesis_id": hypothesis_id,
        "expected_payoff": dict(expected_payoff_row or expected_payoff()),
        "information_gain": dict(
            information_gain_row or information_gain(about=action)
        ),
        "discriminator": dict(
            discriminator_row or discriminator(name=action)
        ),
        "cost": dict(cost_row or cost(class_="UNQUANTIFIED")),
        "outcome": dict(outcome_row) if outcome_row else None,
        "belief_update": dict(belief_update_row) if belief_update_row else None,
        "policy_authority": POLICY_AUTHORITY,
        "learned": False,
    }
    missing = [k for k in OPTION_KEYS if k not in row]
    if missing:
        raise PolicyRefused(f"option missing {missing}")
    return row


def option_from_hypothesis(key_fields: Mapping[str, Any]) -> dict[str, Any]:
    """Lift a corpus hypothesis onto a typed option. Does not rank it."""
    hid = str(key_fields.get("id") or key_fields.get("hypothesis_id") or "")
    if not hid:
        raise PolicyRefused("hypothesis has no id")
    falsifier = key_fields.get("cheapest_falsifier")
    family = key_fields.get("hypothesis_family") or hid
    return option(
        option_id=f"hyp:{hid}",
        action="test_hypothesis",
        hypothesis_id=hid,
        expected_payoff_row=expected_payoff(
            note="hypothesis test; payoff unquantified until an outcome exists"
        ),
        information_gain_row=information_gain(
            about=str(family),
            reduces_uncertainty_in=hid,
        ),
        discriminator_row=discriminator(
            name=str(falsifier or "unspecified_falsifier"),
            distinguishes=(hid, "not-" + hid),
            cheapest_falsifier=str(falsifier) if falsifier else None,
        ),
        cost_row=cost(
            class_=str(
                (key_fields.get("expected_removed_cost") or {}).get("class")
                if isinstance(key_fields.get("expected_removed_cost"), Mapping)
                else "UNQUANTIFIED"
            ),
            evidence_tier="STATIC",
            note="cost class from the hypothesis, not a measured number",
        ),
    )


def option_from_ebpw_bill(
    candidate: Mapping[str, Any],
    billed: Mapping[str, Any],
) -> dict[str, Any]:
    """Typed option for a complete_ebpw candidate. Arithmetic remains the bill.

    Called by complete_ebpw.build. This is not a policy decision.
    """
    cid = str(candidate.get("id") or billed.get("id") or "candidate")
    versus = billed.get("versus") if isinstance(billed.get("versus"), Mapping) else {}
    ms_saved = versus.get("ms_saved")
    return option(
        option_id=f"ebpw:{cid}",
        action="bill_complete_executable_bpw",
        hypothesis_id=None,
        expected_payoff_row=expected_payoff(
            value=float(ms_saved) if isinstance(ms_saved, (int, float)) else None,
            unit="catalog_ms_saved_versus_incumbent" if ms_saved is not None else None,
            evidence_tier="COST_MODEL",
            note=(
                "catalog stream-class ms, not a complete-token remeasure; "
                "complete_ebpw arithmetic is the cost authority"
            ),
        ),
        information_gain_row=information_gain(
            about="complete_executable_bpw",
            reduces_uncertainty_in="candidate_cost",
        ),
        discriminator_row=discriminator(
            name="parts_reconcile_and_stream_class_bill",
            distinguishes=("reconciled_candidate", "unreconciled_or_hidden_free"),
            cheapest_falsifier=(
                "unreconciled stated_total_bytes, missing part category, "
                "or unbilled component"
            ),
        ),
        cost_row=cost(
            class_="WEIGHT_CODES_AND_AUX",
            quantity=billed.get("billed_ms"),
            unit="catalog_ms",
            evidence_tier="COST_MODEL",
            note=(
                "unique declared bytes times ECONOMICS_CALIBRATION stream-class "
                "ms/GB; not a learned prediction"
            ),
        ),
    )


def apply_deterministic_belief_update(
    prior: Mapping[str, Any],
    outcome_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a verdict into the posterior. Not a learned Bayesian update.

    The resident / deterministic record is the authority. A numeric
    posterior that looks trained is refused: we do not have one.
    """
    hid = outcome_row.get("hypothesis_id") or outcome_row.get("option_id")
    if not hid:
        raise PolicyRefused("belief update needs hypothesis_id or option_id")
    status = outcome_row.get("status")
    if not status:
        raise PolicyRefused("belief update needs an outcome status")
    if outcome_row.get("learned") is True:
        raise PolicyRefused(
            "refusing a learned belief update; the deterministic/resident "
            "policy remains the authority"
        )
    prior_hyps = dict(prior.get("hypotheses") or {})
    posterior_hyps = dict(prior_hyps)
    posterior_hyps[str(hid)] = {
        "status": status,
        "update_rule": "deterministic_verdict_write",
        "learned": False,
        "authority": "resident_or_deterministic_record",
        "evidence": outcome_row.get("observed"),
        "evidence_tier": outcome_row.get("evidence_tier") or "STATIC",
    }
    return {
        "hypotheses": posterior_hyps,
        "prior": {"hypotheses": prior_hyps},
        "posterior": {"hypotheses": posterior_hyps},
        "update_rule": "deterministic_verdict_write",
        "learned": False,
        "authority": POLICY_AUTHORITY,
    }


def refuse_to_select(options: list[Mapping[str, Any]]) -> dict[str, Any]:
    """This module does not pick an option. The resident does."""
    return {
        "selected": None,
        "n_options": len(options),
        "learned": False,
        "authority": POLICY_AUTHORITY,
        "reason": (
            "typed options only; ranking is the resident/deterministic policy"
        ),
    }
