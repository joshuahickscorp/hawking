"""PHASE_LISTENERS — spawn Phase II transfer and Phase III attack the moment a qualifying law exists.

The three Odysseys are recurrent, not a sequence. Phase II and Phase III
must start when Phase I emits a qualifying law, not after Phase I finishes.
This module is the shared trigger. It reads the Odyssey II law store and
emits WorkUnits. It does not promote a law, does not apply an attack
result, and does not conclude that a law holds or fails.

An attack whose input falls outside the law's own preconditions is
VACUOUS and is rejected before it is scheduled. A law with no recorded
preconditions is underspecified and cannot be attacked. A transfer onto
the law's origin is not a transfer. An empty store emits zero units.

    python3 tools/future/phase_listeners.py --build
    python3 tools/future/phase_listeners.py --listen
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.future._common import write_receipt
from tools.future import odyssey2_law_store as ols
from tools.future import odyssey3_adversary as o3
from tools.future import workunit_species as wus
from tools.future import workgraph as wg


RECEIPT = "PHASE_LISTENERS.json"
SCHEMA = "hawking.future.phase_listeners.v1"
LAW_STORE_REL = "receipts/future/ODYSSEY2_LAW_STORE.json"

INFO_HIGH, INFO_MEDIUM, INFO_LOW = wg.INFO_HIGH, wg.INFO_MEDIUM, wg.INFO_LOW

VACUOUS = "vacuous"
UNDERSPECIFIED = "underspecified"
NOT_A_TRANSFER = "not_a_transfer"
ATLAS_DEAD = "atlas_dead"
EMPTY_STORE = "empty_store"
USEFUL = "useful"
ACCEPTED = "accepted"
SPAWN = "SPAWN"
REFUSED = "REFUSED"

# Scope -> axes the law is actually claiming. Origin-only fields that are
# not in this map are observations, not preconditions of the claim.
CLAIMED_AXES = {
    "MODEL_LOCAL": ("source_model", "organ_class"),
    "ARCHITECTURE_FAMILY": ("architecture_family", "organ_class"),
    "BACKEND_FAMILY": ("backend", "organ_class"),
    "MACHINE_LOCAL": ("source_device", "organ_class"),
    "GENERIC_CANDIDATE": ("organ_class",),
    "GENERIC_VERIFIED": ("organ_class",),
}

# These organ_class values are labels for multi-organ observations, not a
# tensor identity. They do not constrain an attack's organ.
ORGAN_WILDCARDS = frozenset({"cross_model", "method"})

UNNAMED = frozenset({"", "unknown", "none", "null", "unrecorded", "n/a"})

# Out-of-domain stand-ins. Chosen because they are real names in the
# Odyssey II school/family tables, never invented specimens.
HOSTILE_MODEL_DEFAULT = "Falcon-H1-7B"
HOSTILE_MODEL_IF_FALCON = "Qwen3.8-27B"
HOSTILE_ORGAN = "lm_head"
HOSTILE_FAMILY = "falcon_h1"
HOSTILE_BACKEND = "CUDA"
HOSTILE_DEVICE = "NOT_THE_NAMED_DEVICE"

PHASE_II = "phase_ii_transfer"
PHASE_III = "phase_iii_attack"

LISTEN_RULE = (
    "once Phase I emits a law, Phase II may transfer it and Phase III "
    "may attack it. There is no global barrier between II and III."
)


class VacuousAttackError(ValueError):
    """An attack whose input is outside the law's claimed domain."""


class UnderspecifiedLawError(ValueError):
    """A law with no recorded preconditions cannot be attacked."""


class NotATransferError(ValueError):
    """A target identical to the law's origin is not a transfer."""


# ---------------------------------------------------------------------------
# Law access. The Odyssey II field set is the schema; nothing is invented.
# ---------------------------------------------------------------------------


def _get(law: Any, key: str, default: Any = None) -> Any:
    if isinstance(law, Mapping):
        return law.get(key, default)
    return getattr(law, key, default)


def as_law_dict(law: Any) -> dict[str, Any]:
    if isinstance(law, ols.Law):
        return law.to_dict()
    if isinstance(law, Mapping):
        return dict(law)
    raise TypeError(f"law must be an Odyssey II Law or dict, got {type(law).__name__}")


def _norm(value: Any) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "")).strip("_")


def named(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text.lower() in UNNAMED:
        return None
    return text


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "")).strip("-")
    return s or "item"


def _has_token(blob: str, words: tuple[str, ...]) -> bool:
    for w in words:
        if re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", blob):
            return True
    return False


def _blob(law: Mapping[str, Any]) -> str:
    return " ".join(
        str(_get(law, k, "") or "")
        for k in (
            "statement",
            "organ_class",
            "backend",
            "architecture_family",
            "counterexample_requirement",
            "evidence_strength",
            "scope",
        )
    ).lower()


def _same_model(a: Any, b: Any) -> bool:
    if not named(a) or not named(b):
        return False
    if _norm(a) == _norm(b):
        return True
    sa, sb = ols.school_of_model(str(a)), ols.school_of_model(str(b))
    if sa and sb and sa == sb:
        return True
    na, nb = _norm(a), _norm(b)
    for meta in ols.SCHOOLS.values():
        aliases = {_norm(x) for x in meta["aliases"]}
        aliases.add(_norm(meta["source_model"]))
        aliases.add(_norm(meta["school"]))
        if na in aliases and nb in aliases:
            return True
    return False


def origin_school(law: Mapping[str, Any]) -> str | None:
    return ols.school_of_model(str(_get(law, "source_model") or ""))


def origin_family(law: Mapping[str, Any]) -> str | None:
    family = named(_get(law, "architecture_family"))
    if family:
        return family
    model = named(_get(law, "source_model"))
    if not model:
        return None
    derived = ols.architecture_family_of(model)
    return derived if named(derived) else None


# ---------------------------------------------------------------------------
# Preconditions = the law's claimed domain. Never guessed into existence.
# ---------------------------------------------------------------------------


def _parse_explicit(raw: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if raw is None:
        return out
    if isinstance(raw, Mapping):
        for axis, value in raw.items():
            if isinstance(value, (list, tuple)):
                for item in value:
                    v = named(item)
                    if v:
                        out.append({"axis": str(axis), "value": v})
            else:
                v = named(value)
                if v:
                    out.append({"axis": str(axis), "value": v})
        return out
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, Mapping):
                if "axis" in item and "value" in item:
                    v = named(item.get("value"))
                    if v:
                        out.append({"axis": str(item["axis"]), "value": v})
                else:
                    out.extend(_parse_explicit(item))
            elif isinstance(item, str) and "=" in item:
                axis, _, val = item.partition("=")
                v = named(val)
                if v:
                    out.append({"axis": axis.strip(), "value": v})
        return out
    return out


def _axis_restricts(axis: str, value: str) -> bool:
    if axis == "organ_class" and _norm(value) in {_norm(x) for x in ORGAN_WILDCARDS}:
        return False
    return named(value) is not None


def recorded_preconditions(law: Any) -> dict[str, Any]:
    """Claimed-domain constraints. Explicit empty is underspecified, not derived."""
    d = as_law_dict(law)
    if "preconditions" in d:
        parsed = _parse_explicit(d.get("preconditions"))
        usable = [c for c in parsed if _axis_restricts(c["axis"], c["value"])]
        if not usable:
            return {
                "constraints": [],
                "source": "explicit_empty",
                "underspecified": True,
                "reason": (
                    f"{d.get('law_id', '<no id>')}: preconditions were recorded "
                    "empty; the law is underspecified and cannot be attacked"
                ),
            }
        return {
            "constraints": usable,
            "source": "explicit",
            "underspecified": False,
            "reason": None,
        }
    scope = str(d.get("scope") or "")
    axes = CLAIMED_AXES.get(scope, ("source_model", "organ_class", "architecture_family"))
    constraints: list[dict[str, str]] = []
    for axis in axes:
        value = named(d.get(axis))
        if value and _axis_restricts(axis, value):
            constraints.append({"axis": axis, "value": value})
    if not constraints:
        return {
            "constraints": [],
            "source": "absent",
            "underspecified": True,
            "reason": (
                f"{d.get('law_id', '<no id>')}: no recorded preconditions "
                "(claimed-domain axes are unnamed); the law is underspecified "
                "and cannot be attacked"
            ),
        }
    return {
        "constraints": constraints,
        "source": "derived",
        "underspecified": False,
        "reason": None,
    }


def _axis_match(axis: str, expected: str, actual: Any) -> bool:
    got = named(actual)
    if got is None:
        # Unspecified on the attack inherits the law. Inheritance is in-domain.
        return True
    if axis == "source_model":
        return _same_model(expected, got)
    if axis == "architecture_family":
        if _norm(expected) == _norm(got):
            return True
        # A concrete model named on this axis still matches its family.
        return _norm(ols.architecture_family_of(got)) == _norm(expected)
    return _norm(expected) == _norm(got)


def _attack_inputs(attack: Mapping[str, Any]) -> dict[str, Any]:
    raw = attack.get("inputs")
    if isinstance(raw, Mapping):
        out = dict(raw)
    else:
        out = {}
    apply_on = out.get("apply_on")
    if isinstance(apply_on, Mapping):
        for key in (
            "source_model",
            "target_model",
            "name",
            "organ_class",
            "architecture_family",
            "backend",
            "source_device",
        ):
            if key in apply_on and key not in out:
                out[key] = apply_on[key]
        if "source_model" not in out and apply_on.get("name"):
            out["source_model"] = apply_on["name"]
        if "source_model" not in out and apply_on.get("target_model"):
            out["source_model"] = apply_on["target_model"]
    for alias_src, alias_dst in (
        ("target_model", "source_model"),
        ("model", "source_model"),
        ("organ", "organ_class"),
        ("family", "architecture_family"),
        ("device", "source_device"),
    ):
        if alias_dst not in out and alias_src in out:
            out[alias_dst] = out[alias_src]
    return out


def classify_attack(law: Any, attack: Mapping[str, Any]) -> dict[str, Any]:
    """USEFUL only if every recorded precondition is satisfied. Else VACUOUS / UNDERSPECIFIED."""
    d = as_law_dict(law)
    law_id = str(d.get("law_id") or "<no id>")
    pre = recorded_preconditions(d)
    if pre["underspecified"]:
        return {
            "decision": REFUSED,
            "reason_code": UNDERSPECIFIED,
            "reason": pre["reason"],
            "law_id": law_id,
            "in_domain": False,
            "vacuous": False,
            "underspecified": True,
            "violations": [],
            "preconditions": [],
        }
    inputs = _attack_inputs(attack)
    violations: list[dict[str, str]] = []
    for constraint in pre["constraints"]:
        axis = constraint["axis"]
        expected = constraint["value"]
        actual = inputs.get(axis)
        if not _axis_match(axis, expected, actual):
            violations.append(
                {
                    "axis": axis,
                    "claimed": expected,
                    "attack_input": str(actual),
                }
            )
    if violations:
        return {
            "decision": REFUSED,
            "reason_code": VACUOUS,
            "reason": (
                f"{law_id}: attack is vacuous; input falls outside the law's "
                f"preconditions {violations}. A miss outside the claimed domain "
                "proves nothing — the law already excluded that case."
            ),
            "law_id": law_id,
            "in_domain": False,
            "vacuous": True,
            "underspecified": False,
            "violations": violations,
            "preconditions": pre["constraints"],
        }
    return {
        "decision": ACCEPTED,
        "reason_code": USEFUL,
        "reason": f"{law_id}: attack input satisfies recorded preconditions",
        "law_id": law_id,
        "in_domain": True,
        "vacuous": False,
        "underspecified": False,
        "violations": [],
        "preconditions": pre["constraints"],
    }


def make_vacuous_attack(law: Any) -> dict[str, Any]:
    """A constructed out-of-domain attack. Used so the vacuity guard is watched failing."""
    d = as_law_dict(law)
    pre = recorded_preconditions(d)
    inputs = origin_inputs(d)
    # Contradict the first restricting axis. If none exist this still produces
    # an object; classify_attack will then refuse as underspecified.
    for constraint in pre["constraints"]:
        axis = constraint["axis"]
        if axis == "source_model":
            src = named(d.get("source_model")) or ""
            inputs["source_model"] = (
                HOSTILE_MODEL_IF_FALCON if "falcon" in _norm(src) else HOSTILE_MODEL_DEFAULT
            )
            break
        if axis == "architecture_family":
            inputs["architecture_family"] = HOSTILE_FAMILY
            inputs["source_model"] = HOSTILE_MODEL_DEFAULT
            break
        if axis == "organ_class":
            inputs["organ_class"] = HOSTILE_ORGAN
            break
        if axis == "backend":
            inputs["backend"] = HOSTILE_BACKEND
            break
        if axis == "source_device":
            inputs["source_device"] = HOSTILE_DEVICE
            break
    return {
        "family": "negative_transfer",
        "law_id": d.get("law_id"),
        "inputs": inputs,
        "adversarial_target": inputs.get("source_model") or inputs.get("organ_class"),
        "note": "constructed out-of-domain input; must be rejected as vacuous",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def origin_inputs(law: Mapping[str, Any]) -> dict[str, Any]:
    d = as_law_dict(law)
    out: dict[str, Any] = {}
    for key in ("source_model", "organ_class", "architecture_family", "backend", "source_device"):
        value = named(d.get(key))
        if value:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Store load. Absence is empty, never a fixture fallback.
# ---------------------------------------------------------------------------


def _extract_store_laws(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, list):
        return [x for x in doc if isinstance(x, dict)]
    if not isinstance(doc, dict):
        return []
    raw = doc.get("laws")
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def load_law_store(*, laws: list[Any] | None = None) -> dict[str, Any]:
    """Read the Odyssey II store. Caller-supplied lists win. Missing is empty."""
    if laws is not None:
        records = [as_law_dict(x) for x in laws]
        return {
            "laws": records,
            "n": len(records),
            "present": True,
            "source": "caller",
            "reason": "empty law store; both listeners emit zero units" if not records else None,
            "reason_code": EMPTY_STORE if not records else None,
        }
    doc, err = ols.try_load(LAW_STORE_REL)
    if doc is None:
        return {
            "laws": [],
            "n": 0,
            "present": False,
            "source": "absent",
            "reason": (
                f"{LAW_STORE_REL} absent ({err}); both listeners emit zero units. "
                "Absence is not filled with fixture laws — that would impersonate Phase I."
            ),
            "reason_code": EMPTY_STORE,
        }
    records = _extract_store_laws(doc)
    if not records:
        return {
            "laws": [],
            "n": 0,
            "present": True,
            "source": LAW_STORE_REL,
            "reason": (
                f"{LAW_STORE_REL} present but contains no laws list; "
                "both listeners emit zero units"
            ),
            "reason_code": EMPTY_STORE,
        }
    return {
        "laws": records,
        "n": len(records),
        "present": True,
        "source": LAW_STORE_REL,
        "reason": None,
        "reason_code": None,
    }


def _valid_odyssey_ii(law: Mapping[str, Any]) -> tuple[bool, str | None]:
    lid = named(law.get("law_id"))
    stmt = named(law.get("statement"))
    if not lid or not stmt:
        return False, "missing law_id or statement"
    scope = law.get("scope")
    if scope not in ols.SCOPES:
        return False, f"scope {scope!r} is not on the Odyssey II lattice"
    strength = law.get("evidence_strength")
    if strength not in ols.EVIDENCE_STRENGTHS:
        return False, f"evidence_strength {strength!r} is not an Odyssey II class"
    return True, None


def qualifying_laws(*, laws: list[Any] | None = None) -> dict[str, Any]:
    """Laws on disk that meet the bar for transfer or attack. Does not invent records."""
    store = load_law_store(laws=laws)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for raw in store["laws"]:
        ok, why = _valid_odyssey_ii(raw)
        if not ok:
            skipped.append({"law_id": str(raw.get("law_id") or "<no id>"), "reason": why or "invalid"})
            continue
        pre = recorded_preconditions(raw)
        origin_named = bool(
            named(raw.get("source_model"))
            or origin_school(raw)
            or named(raw.get("architecture_family"))
        )
        q_attack = not pre["underspecified"]
        q_transfer = origin_named
        if not (q_attack or q_transfer):
            skipped.append(
                {
                    "law_id": str(raw.get("law_id")),
                    "reason": "no named origin and no recorded preconditions",
                }
            )
            continue
        rows.append(
            {
                "law_id": raw["law_id"],
                "scope": raw["scope"],
                "source_model": raw.get("source_model"),
                "architecture_family": raw.get("architecture_family"),
                "organ_class": raw.get("organ_class"),
                "evidence_strength": raw.get("evidence_strength"),
                "qualifies_for_transfer": q_transfer,
                "qualifies_for_attack": q_attack,
                "preconditions": pre["constraints"],
                "preconditions_source": pre["source"],
                "law": {k: raw.get(k) for k in ols.LAW_FIELDS if k in raw},
            }
        )
    rows.sort(key=lambda r: str(r["law_id"]))
    reason = store["reason"]
    if not rows and reason is None:
        reason = "no qualifying laws on disk after schema checks"
    return {
        "laws": rows,
        "n": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "store": {
            "source": store["source"],
            "present": store["present"],
            "n_records": store["n"],
            "reason": store["reason"],
            "reason_code": store["reason_code"],
        },
        "reason": reason,
        "reason_code": store["reason_code"] if not rows else None,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "concludes_law": False,
    }


# ---------------------------------------------------------------------------
# Phase II — transfer targets. Identity is rejected, never a self-transfer.
# ---------------------------------------------------------------------------


def _target_school_of(target: Mapping[str, Any]) -> str | None:
    school = named(target.get("target_school"))
    if school:
        return school
    model = named(target.get("target_model") or target.get("source_model") or target.get("name"))
    return ols.school_of_model(model) if model else None


def classify_transfer(law: Any, target: Mapping[str, Any]) -> dict[str, Any]:
    d = as_law_dict(law)
    law_id = str(d.get("law_id") or "<no id>")
    tgt_model = named(
        target.get("target_model") or target.get("source_model") or target.get("name")
    )
    tgt_school = _target_school_of(target)
    src_model = named(d.get("source_model"))
    src_school = origin_school(d)
    if (tgt_school and src_school and tgt_school == src_school) or (
        tgt_model and src_model and _same_model(tgt_model, src_model)
    ):
        return {
            "decision": REFUSED,
            "reason_code": NOT_A_TRANSFER,
            "reason": (
                f"{law_id}: target {tgt_school or tgt_model!r} is the law's origin "
                f"({src_school or src_model!r}); that is not a transfer"
            ),
            "law_id": law_id,
            "target_school": tgt_school,
            "target_model": tgt_model,
        }
    return {
        "decision": ACCEPTED,
        "reason_code": ACCEPTED,
        "reason": f"{law_id}: {tgt_school or tgt_model} is distinct from origin",
        "law_id": law_id,
        "target_school": tgt_school,
        "target_model": tgt_model,
    }


def _ols_law(d: Mapping[str, Any]) -> ols.Law | None:
    try:
        kwargs: dict[str, Any] = {}
        for field in ols.LAW_FIELDS:
            if field not in d and field != "time_to_first_useful_executable_ns":
                return None
            kwargs[field] = d.get(field)
        refs = kwargs.get("evidence_refs") or ()
        kwargs["evidence_refs"] = tuple(refs) if not isinstance(refs, tuple) else refs
        cands = kwargs.get("transfer_candidates") or ()
        if isinstance(cands, tuple):
            kwargs["transfer_candidates"] = cands
        elif isinstance(cands, list):
            kwargs["transfer_candidates"] = tuple(cands)
        else:
            kwargs["transfer_candidates"] = ()
        conf = kwargs.get("transfer_confidence")
        if not isinstance(conf, dict):
            return None
        kwargs["time_to_first_useful_executable_ns"] = None
        return ols.validate_law(ols.Law(**kwargs))
    except Exception:
        return None


def _transfer_gain(law: Mapping[str, Any], target: Mapping[str, Any]) -> int:
    src_family = origin_family(law)
    tgt_family = named(target.get("target_architecture_family"))
    if tgt_family and src_family and _norm(tgt_family) != _norm(src_family):
        return INFO_HIGH
    src_school = origin_school(law)
    tgt_school = _target_school_of(target)
    if tgt_school and src_school and tgt_school != src_school:
        return INFO_HIGH if (tgt_family and src_family and _norm(tgt_family) != _norm(src_family)) else INFO_MEDIUM
    if tgt_family and src_family and _norm(tgt_family) == _norm(src_family):
        return INFO_MEDIUM
    return INFO_LOW


def _plausible_reason(law: Mapping[str, Any], target: Mapping[str, Any], gain: int) -> str:
    lid = law.get("law_id")
    scope = law.get("scope")
    src = law.get("source_model")
    tgt = target.get("target_school") or target.get("target_model")
    family_bit = ""
    src_f, tgt_f = origin_family(law), named(target.get("target_architecture_family"))
    if src_f and tgt_f and _norm(src_f) != _norm(tgt_f):
        family_bit = (
            f" Cross-family ({src_f} -> {tgt_f}): a hold here would be the first "
            "evidence the statement is not origin-local."
        )
    elif _target_school_of(target) and origin_school(law):
        family_bit = (
            f" Cross-school ({origin_school(law)} -> {_target_school_of(target)}): "
            "Flash <-> Qwen27 is the first transfer school."
        )
    gain_word = {INFO_HIGH: "high", INFO_MEDIUM: "medium", INFO_LOW: "low"}[gain]
    return (
        f"{lid} at {scope} was evidenced on {src}; {tgt} is a distinct origin "
        f"where the statement might hold and has not been tested. "
        f"Expected information gain {gain_word}.{family_bit} "
        "A spawned transfer does not conclude that it does hold."
    )


def transfer_targets(law: Any) -> dict[str, Any]:
    """Where this law might hold that it has not been tested. Ranked by information gain."""
    d = as_law_dict(law)
    law_id = str(d.get("law_id") or "<no id>")
    rejected: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()

    candidates: list[dict[str, Any]] = []
    for school, meta in ols.SCHOOLS.items():
        candidates.append(
            {
                "target_school": school,
                "target_model": meta["source_model"],
                "target_architecture_family": meta["architecture_family"],
            }
        )
    for existing in d.get("transfer_candidates") or ():
        if isinstance(existing, Mapping):
            candidates.append(dict(existing))

    ols_law = _ols_law(d)
    for cand in candidates:
        identity = classify_transfer(d, cand)
        if identity["reason_code"] == NOT_A_TRANSFER:
            rejected.append(identity)
            continue
        key = _norm(cand.get("target_school") or cand.get("target_model"))
        if not key or key in seen:
            continue
        if ols_law is not None:
            target_name = cand.get("target_school") or cand.get("target_model")
            try:
                proposals = ols.transfer_candidates(ols_law, str(target_name))
            except ols.NegativeTransferError as exc:
                rejected.append(
                    {
                        "decision": REFUSED,
                        "reason_code": ATLAS_DEAD,
                        "reason": str(exc),
                        "law_id": law_id,
                        "target_school": cand.get("target_school"),
                        "target_model": cand.get("target_model"),
                        "atlas_key": getattr(exc, "atlas_key", None),
                    }
                )
                seen.add(key)
                continue
            except ols.LawStoreError as exc:
                rejected.append(
                    {
                        "decision": REFUSED,
                        "reason_code": "unknown_target",
                        "reason": str(exc),
                        "law_id": law_id,
                        "target_school": cand.get("target_school"),
                        "target_model": cand.get("target_model"),
                    }
                )
                seen.add(key)
                continue
            if not proposals:
                # Engine treats same-school as empty; we already classified identity.
                rejected.append(identity if identity["reason_code"] == NOT_A_TRANSFER else {
                    "decision": REFUSED,
                    "reason_code": NOT_A_TRANSFER,
                    "reason": f"{law_id}: transfer_candidates returned empty for {target_name}",
                    "law_id": law_id,
                    "target_school": cand.get("target_school"),
                    "target_model": cand.get("target_model"),
                })
                seen.add(key)
                continue
            cand = {**cand, **proposals[0]}
        seen.add(key)
        gain = _transfer_gain(d, cand)
        ranked.append(
            {
                "target_school": cand.get("target_school"),
                "target_model": cand.get("target_model"),
                "target_architecture_family": cand.get("target_architecture_family"),
                "expected_information_gain": gain,
                "reason": _plausible_reason(d, cand, gain),
                "source_model": d.get("source_model"),
                "source_school": origin_school(d),
                "law_id": law_id,
                "scope": d.get("scope"),
                "concludes_law": False,
                "evidence_class": "STATIC_ONLY",
                "gpu_authority": False,
            }
        )

    ranked.sort(
        key=lambda r: (
            -int(r["expected_information_gain"]),
            str(r.get("target_school") or ""),
            str(r.get("target_model") or ""),
        )
    )
    return {
        "law_id": law_id,
        "ranked": ranked,
        "rejected": rejected,
        "n_accepted": len(ranked),
        "n_rejected": len(rejected),
        "decision": SPAWN if ranked else REFUSED,
        "reason": (
            None
            if ranked
            else f"{law_id}: no transfer target distinct from origin survived refusal"
        ),
        "concludes_law": False,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


# ---------------------------------------------------------------------------
# Phase III — strongest useful counterexample, in-domain only.
# ---------------------------------------------------------------------------


def _family_gain(law: Mapping[str, Any], family: str) -> int | None:
    blob = _blob(law)
    scope = str(law.get("scope") or "")
    metric = _has_token(blob, ("cosine", "adequacy", "bpw", "rel_fro", "metric", "certificate"))
    fitted = _has_token(
        blob, ("fit", "fitted", "seed", "seeded", "held-out", "holdout", "held_out", "grid", "search")
    )
    layout = _has_token(blob, ("layout", "storage", "packing", "kernel", "representation", "gemv", "codec"))
    compiler = _has_token(blob, ("compiler", "default", "flag", "metallib"))
    causal = _has_token(blob, ("because", "cause", "causal", "mechanism", "therefore", "so the"))
    timing = _has_token(blob, ("wall", "dispatch", "scheduling", "telemetry", "latency"))

    if family == "negative_transfer":
        if scope in {"MODEL_LOCAL", "MACHINE_LOCAL"}:
            return None
        return INFO_HIGH
    if family == "measurement_trap":
        return INFO_HIGH if metric else INFO_LOW
    if family == "blind_holdout":
        return INFO_HIGH if fitted else INFO_MEDIUM
    if family == "representation_overfit":
        return INFO_HIGH if layout else INFO_MEDIUM
    if family == "causal_control":
        return INFO_HIGH if causal else INFO_MEDIUM
    if family == "goodhart":
        return INFO_HIGH if metric else None
    if family == "compiler_prior":
        if compiler or layout:
            return INFO_MEDIUM
        return INFO_LOW
    if family == "law_scope":
        return INFO_MEDIUM
    if family == "contamination_trap":
        return INFO_HIGH if (timing or scope == "MACHINE_LOCAL") else INFO_MEDIUM
    return INFO_LOW


def _prove_pair(family: str, law: Mapping[str, Any]) -> tuple[str, str]:
    lid = law.get("law_id")
    scope = law.get("scope")
    organ = law.get("organ_class")
    model = law.get("source_model")
    if family == "blind_holdout":
        return (
            f"the statement of {lid} does not survive a withheld slice of {organ} on {model}",
            (
                "it would not prove the law fails on a different model, organ, or backend; "
                "it would not be a hardware measurement; it would not widen scope"
            ),
        )
    if family == "measurement_trap":
        return (
            f"the metric named by {lid} can certify a destroyed organ (harness self-measure) "
            f"inside the claimed domain ({model}/{organ})",
            (
                "it would not prove the organ itself is inexpressible; it would not prove a "
                "transfer failure; it would not produce a protected number"
            ),
        )
    if family == "representation_overfit":
        return (
            f"{lid} is packing-specific rather than organ-specific on {model}/{organ}",
            (
                "it would not prove the law fails on another architecture family; "
                "it would not prove the named metric is globally broken"
            ),
        )
    if family == "causal_control":
        return (
            f"the mechanism named by {lid} is not necessary for the effect on {model}/{organ}",
            "it would not prove the effect is absent; it would not speak to a different organ",
        )
    if family == "goodhart":
        return (
            f"optimising the metric of {lid} on {model}/{organ} collapses the underlying capability",
            "it would not prove a transfer miss; it would not be a TPS or energy claim",
        )
    if family == "compiler_prior":
        return (
            f"the effect named by {lid} on {model}/{organ} is a compiler default, not the law",
            "it would not prove the organ cannot express the effect under another default",
        )
    if family == "law_scope":
        return (
            f"{lid} at {scope} fails the cheapest in-domain probe that would force a narrower reading",
            "it would not refute the statement inside a still-narrower already-claimed cell",
        )
    if family == "contamination_trap":
        return (
            f"{lid} vanishes or reverses under a contaminated machine state of the claimed device/organ",
            "it would not prove the organ-level statement is false on a quiet window",
        )
    if family == "negative_transfer":
        return (
            f"{lid} at {scope} does not hold on a distinct in-domain target (second model/family "
            f"still inside the claimed organ {organ})",
            (
                "it would not prove the origin observation was wrong; it would not license a "
                "vacuous miss on an out-of-domain specimen"
            ),
        )
    return (
        f"a successful in-domain attack would show {lid} fails a case it claimed to cover",
        "it would not prove anything the law already excluded as a precondition",
    )


def _in_domain_inputs_for(law: Mapping[str, Any], family: str) -> dict[str, Any]:
    inputs = origin_inputs(law)
    organ = inputs.get("organ_class") or law.get("organ_class")
    model = inputs.get("source_model") or law.get("source_model")
    if family == "blind_holdout":
        inputs.update(
            {
                "fit_fraction": 0.5,
                "holdout_fraction": 0.5,
                "score_on": "holdout_only",
                "forbid_fit_on_holdout": True,
                "organ_class": organ,
                "source_model": model,
            }
        )
    elif family == "measurement_trap":
        inputs.update(
            {
                "controls": ["scale_invariance", "skip_counted_as_pass", "receipt_reread_as_run"],
                "past_failure_shapes": list(o3.PAST_FAILURE_SHAPES[:3]),
                "organ_class": organ,
                "source_model": model,
            }
        )
    elif family == "representation_overfit":
        inputs.update(
            {
                "same_organ": organ,
                "same_model": model,
                "alternate_representation": "repack_or_alternate_layout_of_the_same_organ",
                "organ_class": organ,
                "source_model": model,
            }
        )
    elif family == "negative_transfer":
        # Stay inside claimed domain: second model/family only when the scope
        # actually claims that generality. Origin axes remain on the inputs so
        # organ/backend/device preconditions still hold.
        scope = str(law.get("scope") or "")
        src_school = origin_school(law)
        if scope in {"GENERIC_CANDIDATE", "GENERIC_VERIFIED"}:
            pick = "Flash" if src_school != "Flash" else "Qwen27"
            meta = ols.SCHOOLS[pick]
            inputs["source_model"] = meta["source_model"]
            inputs["architecture_family"] = meta["architecture_family"]
            inputs["apply_on"] = {
                "name": meta["source_model"],
                "kind": "in_domain_second_family",
            }
        elif scope == "ARCHITECTURE_FAMILY":
            # Same family, different model alias if we have one; otherwise a
            # sibling label that classify_attack will still accept via family.
            inputs["architecture_family"] = origin_family(law)
            inputs["source_model"] = model
            inputs["apply_on"] = {
                "name": model,
                "kind": "in_domain_same_family",
            }
        else:
            inputs["apply_on"] = {"name": model, "kind": "origin"}
        inputs["organ_class"] = organ
    elif family == "contamination_trap":
        inputs.update(
            {
                "quiet_arm": "PROTECTED_ABSOLUTE_required_not_available",
                "dirty_arm": "DIAGNOSTIC_RELATIVE_busy_machine",
                "sidecar_cannot_open_protected_lease": True,
            }
        )
    elif family == "law_scope":
        nxt = {
            "GENERIC_VERIFIED": "GENERIC_CANDIDATE",
            "GENERIC_CANDIDATE": "ARCHITECTURE_FAMILY",
            "BACKEND_FAMILY": "ARCHITECTURE_FAMILY",
            "ARCHITECTURE_FAMILY": "MODEL_LOCAL",
            "MACHINE_LOCAL": "MACHINE_LOCAL",
            "MODEL_LOCAL": "MODEL_LOCAL",
        }.get(str(law.get("scope") or ""), "MODEL_LOCAL")
        inputs.update({"current_scope": law.get("scope"), "one_step_narrower": nxt})
    else:
        inputs.setdefault("organ_class", organ)
        inputs.setdefault("source_model", model)
    return inputs


def _attack_spec(law: Mapping[str, Any], family: str, gain: int) -> dict[str, Any]:
    prove, not_prove = _prove_pair(family, law)
    inputs = _in_domain_inputs_for(law, family)
    counter = named(law.get("counterexample_requirement")) or (
        "a case inside the claimed domain where the statement fails"
    )
    return {
        "attack_id": f"{law.get('law_id')}::{family}",
        "family": family,
        "law_id": law.get("law_id"),
        "inputs": inputs,
        "expected_information_gain": gain,
        "would_prove": prove,
        "would_not_prove": not_prove,
        "counterexample_requirement": counter,
        "falsifier": counter,
        "adversarial_target": (
            f"{inputs.get('source_model')}/{inputs.get('organ_class')}#{family}"
        ),
        "selection_rule": (
            "strongest useful in-domain counterexample; cost is not the rank key; "
            "vacuous out-of-domain inputs are rejected before scheduling"
        ),
        "concludes_law": False,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "bench_state": "UNKNOWN",
    }


def attacks(law: Any) -> dict[str, Any]:
    """Strongest useful in-domain counterexamples. Vacuous and underspecified are refused."""
    d = as_law_dict(law)
    law_id = str(d.get("law_id") or "<no id>")
    pre = recorded_preconditions(d)
    if pre["underspecified"]:
        return {
            "law_id": law_id,
            "decision": REFUSED,
            "reason_code": UNDERSPECIFIED,
            "reason": pre["reason"],
            "ranked": [],
            "selected": None,
            "rejected": [],
            "n_useful": 0,
            "concludes_law": False,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }

    rejected: list[dict[str, Any]] = []
    useful: list[dict[str, Any]] = []
    families: Iterable[str] = o3.ATTACK_FAMILIES
    for family in families:
        gain = _family_gain(d, family)
        if gain is None:
            continue
        spec = _attack_spec(d, family, gain)
        verdict = classify_attack(d, spec)
        if verdict["reason_code"] == USEFUL:
            spec["classify"] = {k: verdict[k] for k in ("decision", "reason_code", "in_domain")}
            useful.append(spec)
        else:
            rejected.append({**verdict, "family": family, "attack_id": spec["attack_id"]})

    # The vacuity guard must run on a deliberately out-of-domain input too,
    # otherwise it only ever sees in-domain specs and can silently drift.
    vacuous = make_vacuous_attack(d)
    vacuous_verdict = classify_attack(d, vacuous)
    if vacuous_verdict["reason_code"] != VACUOUS and not pre["underspecified"]:
        raise VacuousAttackError(
            f"{law_id}: vacuity guard failed to reject a constructed out-of-domain attack"
        )
    rejected.append({**vacuous_verdict, "family": vacuous.get("family"), "constructed": True})

    useful.sort(
        key=lambda a: (
            -int(a["expected_information_gain"]),
            str(a["family"]),
        )
    )
    selected = useful[0] if useful else None
    if selected is None:
        return {
            "law_id": law_id,
            "decision": REFUSED,
            "reason_code": "no_useful_in_domain_attack",
            "reason": (
                f"{law_id}: every candidate was vacuous or inapplicable; "
                "refusing rather than scheduling an out-of-domain miss"
            ),
            "ranked": [],
            "selected": None,
            "rejected": rejected,
            "n_useful": 0,
            "concludes_law": False,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }
    return {
        "law_id": law_id,
        "decision": SPAWN,
        "reason_code": USEFUL,
        "reason": (
            f"{law_id}: strongest useful counterexample is {selected['family']} "
            f"(gain={selected['expected_information_gain']}); listeners spawn work, "
            "they do not conclude the law"
        ),
        "ranked": useful,
        "selected": selected,
        "rejected": rejected,
        "n_useful": len(useful),
        "concludes_law": False,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


# ---------------------------------------------------------------------------
# Spawn. Raises on vacuous / underspecified / identity so the guard is a raise
# on the schedule path. listen() classifies first and never raises on empty.
# ---------------------------------------------------------------------------


def spawn_attack_unit(law: Any, attack: Mapping[str, Any]) -> dict[str, Any]:
    d = as_law_dict(law)
    pre = recorded_preconditions(d)
    if pre["underspecified"]:
        raise UnderspecifiedLawError(pre["reason"])
    verdict = classify_attack(d, attack)
    if verdict["reason_code"] == UNDERSPECIFIED:
        raise UnderspecifiedLawError(verdict["reason"])
    if verdict["reason_code"] == VACUOUS:
        raise VacuousAttackError(verdict["reason"])
    lid = _slug(str(d.get("law_id")))
    family = _slug(str(attack.get("family") or "attack"))
    uid = f"future.phase-iii.attack.{lid}.{family}"
    gain = int(attack.get("expected_information_gain") or INFO_MEDIUM)
    extras = {
        "species": "odyssey_iii_adversarial_experiment",
        "listener": PHASE_III,
        "law_id": d.get("law_id"),
        "attack_id": attack.get("attack_id") or f"{d.get('law_id')}::{attack.get('family')}",
        "family": attack.get("family"),
        "would_prove": attack.get("would_prove"),
        "would_not_prove": attack.get("would_not_prove"),
        "expected_information_gain": gain,
        "resource_lane": "CPU_ANALYSIS",
        "mutation_scope": ["receipts/future"],
        "cost_units": 1,
        "spawns_work": True,
        "performs_science": False,
        "concludes_law": False,
        "barrier": None,
        "listens_on": "phase_i_law_emission",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": wus.PROPOSAL_CLAIM_BOUNDARY,
    }
    row = wus.emit_hcli_workunit(
        id=uid,
        role="science",
        description=(
            f"Phase III listener: spawn the strongest useful in-domain attack "
            f"({attack.get('family')}) on {d.get('law_id')}. Does not conclude the law."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.odyssey_iii.adversary",
        provider="future.phase_listeners",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras=extras,
    )
    wus.validate_emitted_unit(row)
    return row


def spawn_transfer_unit(law: Any, target: Mapping[str, Any]) -> dict[str, Any]:
    d = as_law_dict(law)
    identity = classify_transfer(d, target)
    if identity["reason_code"] == NOT_A_TRANSFER:
        raise NotATransferError(identity["reason"])
    lid = _slug(str(d.get("law_id")))
    tgt = _slug(str(target.get("target_school") or target.get("target_model") or "target"))
    uid = f"future.phase-ii.transfer.{lid}.{tgt}"
    gain = int(target.get("expected_information_gain") or INFO_MEDIUM)
    extras = {
        "species": "odyssey_ii_transfer_experiment",
        "listener": PHASE_II,
        "law_id": d.get("law_id"),
        "target_school": target.get("target_school"),
        "target_model": target.get("target_model"),
        "target_architecture_family": target.get("target_architecture_family"),
        "expected_information_gain": gain,
        "reason": target.get("reason"),
        "resource_lane": "CPU_ANALYSIS",
        "mutation_scope": ["receipts/future"],
        "cost_units": 1,
        "spawns_work": True,
        "performs_science": False,
        "concludes_law": False,
        "barrier": None,
        "listens_on": "phase_i_law_emission",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": wus.PROPOSAL_CLAIM_BOUNDARY,
    }
    row = wus.emit_hcli_workunit(
        id=uid,
        role="science",
        description=(
            f"Phase II listener: spawn a transfer of {d.get('law_id')} onto "
            f"{target.get('target_school') or target.get('target_model')}. "
            "Does not conclude the law holds there."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.odyssey_ii.law_scope",
        provider="future.phase_listeners",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras=extras,
    )
    wus.validate_emitted_unit(row)
    return row


def listen(*, laws: list[Any] | None = None) -> dict[str, Any]:
    """Emit the WorkUnits both listeners would spawn right now. Empty store → zero units."""
    qualified = qualifying_laws(laws=laws)
    units: list[dict[str, Any]] = []
    phase_ii: list[dict[str, Any]] = []
    phase_iii: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []

    if qualified["n"] == 0:
        reason = qualified["reason"] or (
            "empty law store; both listeners emit zero units"
        )
        return {
            "units": [],
            "n_units": 0,
            "n_phase_ii": 0,
            "n_phase_iii": 0,
            "phase_ii": [],
            "phase_iii": [],
            "refusals": list(qualified.get("skipped") or []),
            "reason": reason,
            "reason_code": qualified.get("reason_code") or EMPTY_STORE,
            "store": qualified["store"],
            "qualifying_n": 0,
            "barrier": None,
            "phase_ii_depends_on_phase_iii": False,
            "phase_iii_depends_on_phase_ii": False,
            "rule": LISTEN_RULE,
            "concludes_law": False,
            "spawns_work": True,
            "performs_science": False,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }

    for row in qualified["laws"]:
        law = row["law"]
        if row["qualifies_for_transfer"]:
            plan = transfer_targets(law)
            refusals.extend(plan["rejected"])
            for target in plan["ranked"]:
                unit = spawn_transfer_unit(law, target)
                units.append(unit)
                phase_ii.append(
                    {
                        "id": unit["id"],
                        "law_id": law["law_id"],
                        "target_school": target.get("target_school"),
                        "target_model": target.get("target_model"),
                        "expected_information_gain": target["expected_information_gain"],
                    }
                )
        if row["qualifies_for_attack"]:
            plan = attacks(law)
            refusals.extend(
                [r for r in plan["rejected"] if r.get("reason_code") != VACUOUS or r.get("constructed")]
            )
            selected = plan.get("selected")
            if selected is None:
                refusals.append(
                    {
                        "law_id": law["law_id"],
                        "reason_code": plan.get("reason_code"),
                        "reason": plan.get("reason"),
                    }
                )
            else:
                unit = spawn_attack_unit(law, selected)
                units.append(unit)
                phase_iii.append(
                    {
                        "id": unit["id"],
                        "law_id": law["law_id"],
                        "family": selected.get("family"),
                        "attack_id": selected.get("attack_id"),
                        "expected_information_gain": selected["expected_information_gain"],
                        "would_prove": selected.get("would_prove"),
                        "would_not_prove": selected.get("would_not_prove"),
                    }
                )
        elif not row["qualifies_for_attack"]:
            refusals.append(
                {
                    "law_id": law["law_id"],
                    "reason_code": UNDERSPECIFIED,
                    "reason": (
                        f"{law['law_id']}: underspecified; Phase III emits no attack unit"
                    ),
                }
            )

    units.sort(key=lambda u: str(u.get("id") or ""))
    return {
        "units": units,
        "n_units": len(units),
        "n_phase_ii": len(phase_ii),
        "n_phase_iii": len(phase_iii),
        "phase_ii": phase_ii,
        "phase_iii": phase_iii,
        "refusals": refusals,
        "reason": None if units else "qualifying laws produced no spawnable targets or attacks",
        "reason_code": None if units else "nothing_to_spawn",
        "store": qualified["store"],
        "qualifying_n": qualified["n"],
        "qualifying_ids": [r["law_id"] for r in qualified["laws"]],
        "barrier": None,
        "phase_ii_depends_on_phase_iii": False,
        "phase_iii_depends_on_phase_ii": False,
        "rule": LISTEN_RULE,
        "concludes_law": False,
        "spawns_work": True,
        "performs_science": False,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


# ---------------------------------------------------------------------------
# Negative-control selftest. A guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


def _fixture_local() -> dict[str, Any]:
    return {
        "law_id": "LAW-FIXTURE-MODEL-LOCAL-MLP",
        "statement": (
            "On Qwen3.8-27B mlp, a fitted affine seed beats uniform round-to-nearest "
            "at matched bits on held-out activations."
        ),
        "source_model": "Qwen3.8-27B",
        "source_device": "UNKNOWN",
        "architecture_family": "dense_hybrid_transformer",
        "organ_class": "mlp",
        "backend": "Metal",
        "evidence_strength": "DIAGNOSTIC_RELATIVE",
        "evidence_refs": ["receipts/headless/ODYSSEY_TRANSFER_PROVEN.json"],
        "scope": "MODEL_LOCAL",
        "transfer_candidates": [],
        "transfer_confidence": {"value": 0.45, "basis": "fixture"},
        "counterexample_requirement": (
            "a held-out mlp slice on Qwen3.8-27B where the seeded family is not better "
            "at matched bits"
        ),
        "expected_saved_experiments": None,
        "actual_saved_experiments": None,
        "time_to_first_useful_executable_ns": None,
    }


def _fixture_generic() -> dict[str, Any]:
    law = _fixture_local()
    law["law_id"] = "LAW-FIXTURE-GENERIC-KERNEL"
    law["statement"] = (
        "Kernel reuse follows storage layout of moe_expert_kernel across architecture families."
    )
    law["scope"] = "GENERIC_CANDIDATE"
    law["organ_class"] = "moe_expert_kernel"
    law["source_model"] = "Qwen3-30B-A3B"
    law["architecture_family"] = "qwen3_moe"
    law["counterexample_requirement"] = (
        "two specimens with divergent on-disk expert layout whose stored tensors "
        "emit the same kernel without a pack-time transpose"
    )
    return law


def _fixture_underspecified() -> dict[str, Any]:
    return {
        "law_id": "LAW-FIXTURE-UNDERSPECIFIED",
        "statement": "a claim with no claimed domain",
        "source_model": "UNKNOWN",
        "source_device": "UNKNOWN",
        "architecture_family": "UNKNOWN",
        "organ_class": "",
        "backend": "UNKNOWN",
        "evidence_strength": "STATIC",
        "evidence_refs": ["receipts/future/PHASE_LISTENERS.json"],
        "scope": "MODEL_LOCAL",
        "transfer_candidates": [],
        "transfer_confidence": {"value": 0.10, "basis": "fixture"},
        "counterexample_requirement": "",
        "preconditions": [],
        "expected_saved_experiments": None,
        "actual_saved_experiments": None,
        "time_to_first_useful_executable_ns": None,
    }


def selftest() -> dict[str, Any]:
    """Four mandatory negative controls, each watched firing."""
    local = _fixture_local()
    generic = _fixture_generic()
    under = _fixture_underspecified()

    vacuous = make_vacuous_attack(local)
    vacuous_cls = classify_attack(local, vacuous)
    if vacuous_cls["reason_code"] != VACUOUS:
        raise VacuousAttackError("vacuous attack was not rejected as vacuous")
    try:
        spawn_attack_unit(local, vacuous)
        raise VacuousAttackError("vacuous attack was scheduled")
    except VacuousAttackError:
        vacuous_raised = True

    under_cls = attacks(under)
    if under_cls["reason_code"] != UNDERSPECIFIED:
        raise UnderspecifiedLawError("underspecified law was attacked")
    try:
        spawn_attack_unit(under, {"family": "blind_holdout", "inputs": origin_inputs(local)})
        raise UnderspecifiedLawError("underspecified law spawned an attack")
    except UnderspecifiedLawError:
        under_raised = True

    identity = {
        "target_school": "Qwen27",
        "target_model": "Qwen3.8-27B",
        "target_architecture_family": "dense_hybrid_transformer",
    }
    ident_cls = classify_transfer(local, identity)
    if ident_cls["reason_code"] != NOT_A_TRANSFER:
        raise NotATransferError("origin target was accepted as a transfer")
    try:
        spawn_transfer_unit(local, identity)
        raise NotATransferError("origin target spawned a transfer")
    except NotATransferError:
        ident_raised = True
    local_targets = transfer_targets(local)
    if any(
        t.get("target_school") == "Qwen27" or _same_model(t.get("target_model"), "Qwen3.8-27B")
        for t in local_targets["ranked"]
    ):
        raise NotATransferError("origin leaked into ranked transfer targets")

    empty = listen(laws=[])
    if empty["n_units"] != 0 or empty["reason_code"] != EMPTY_STORE:
        raise RuntimeError(f"empty store did not emit zero units: {empty}")

    useful = attacks(local)
    if useful["decision"] != SPAWN or useful["selected"] is None:
        raise RuntimeError(f"in-domain law produced no useful attack: {useful}")
    if useful["selected"]["family"] == "negative_transfer":
        raise VacuousAttackError(
            "MODEL_LOCAL strongest attack was negative_transfer; that family is out of domain"
        )
    in_domain_unit = spawn_attack_unit(local, useful["selected"])
    wus.validate_emitted_unit(in_domain_unit)

    generic_plan = attacks(generic)
    generic_nt = next(
        (a for a in generic_plan["ranked"] if a["family"] == "negative_transfer"),
        None,
    )
    generic_nt_useful = generic_nt is not None

    spawned = listen(laws=[local, generic])
    expect_ii = (
        transfer_targets(local)["n_accepted"] + transfer_targets(generic)["n_accepted"]
    )
    if spawned["n_phase_iii"] < 1:
        raise RuntimeError(f"qualifying laws did not spawn Phase III: {spawned}")
    if spawned["n_phase_ii"] != expect_ii:
        raise RuntimeError(
            f"Phase II unit count {spawned['n_phase_ii']} != accepted targets {expect_ii}"
        )
    if spawned["barrier"] is not None or spawned["phase_ii_depends_on_phase_iii"]:
        raise RuntimeError("listeners introduced a II/III barrier")
    if spawned["concludes_law"] or spawned["performs_science"]:
        raise RuntimeError("listeners claimed to perform science or conclude a law")

    graph_ok = []
    for unit in spawned["units"]:
        made = wg.make_unit(
            id=unit["id"],
            role=unit["role"],
            description=unit["description"],
            dependencies=unit.get("dependencies") or [],
            resource_lane=unit["resource_lane"],
            mutation_scope=unit["mutation_scope"],
            verifier=unit["verifier"],
            expected_information_gain=int(unit["expected_information_gain"]),
            cost_units=int(unit["cost_units"]),
            species=unit.get("species"),
            effect_class=unit.get("effect_class") or "READ_ONLY",
        )
        graph_ok.append(made["id"])

    return {
        "vacuous_attack_rejected": vacuous_cls["reason_code"] == VACUOUS and vacuous_raised,
        "underspecified_refused": under_cls["reason_code"] == UNDERSPECIFIED and under_raised,
        "identity_transfer_rejected": ident_cls["reason_code"] == NOT_A_TRANSFER and ident_raised,
        "empty_store_zero_units": empty["n_units"] == 0,
        "empty_store_reason": empty["reason"],
        "in_domain_attack_spawned": in_domain_unit["id"],
        "in_domain_family": useful["selected"]["family"],
        "generic_negative_transfer_in_domain": generic_nt_useful,
        "listen_n_units": spawned["n_units"],
        "listen_n_phase_ii": spawned["n_phase_ii"],
        "listen_n_phase_iii": spawned["n_phase_iii"],
        "graph_admissible_ids": graph_ok,
        "barrier": spawned["barrier"],
        "concludes_law": False,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def build() -> Path:
    live = listen()
    qualified = qualifying_laws()
    controls = selftest()
    underspecified_ids = [
        r["law_id"] for r in qualified["laws"] if not r["qualifies_for_attack"]
    ]
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Spawn Phase II transfer and Phase III attack WorkUnits the moment a "
            "qualifying Odyssey II law exists. Listeners spawn work. They do not "
            "perform science and they do not conclude anything about a law."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "odysseys_are_recurrent": True,
        "rule": LISTEN_RULE,
        "barrier": None,
        "phase_ii_depends_on_phase_iii": False,
        "phase_iii_depends_on_phase_ii": False,
        "law_schema": list(ols.LAW_FIELDS),
        "law_store": qualified["store"],
        "qualifying_n": qualified["n"],
        "qualifying_ids": [r["law_id"] for r in qualified["laws"]],
        "n_qualifying_for_transfer": sum(1 for r in qualified["laws"] if r["qualifies_for_transfer"]),
        "n_qualifying_for_attack": sum(1 for r in qualified["laws"] if r["qualifies_for_attack"]),
        "underspecified_for_attack": underspecified_ids,
        "listen": {
            "n_units": live["n_units"],
            "n_phase_ii": live["n_phase_ii"],
            "n_phase_iii": live["n_phase_iii"],
            "phase_ii": live["phase_ii"],
            "phase_iii": live["phase_iii"],
            "reason": live["reason"],
            "reason_code": live["reason_code"],
            "unit_ids": [u["id"] for u in live["units"]],
            "concludes_law": False,
            "spawns_work": True,
            "performs_science": False,
        },
        "units": live["units"],
        "negative_controls": controls,
        "recovered_implementation": [
            "tools/future/odyssey2_law_store.py — Law field set, SCOPES, SCHOOLS, transfer_candidates()",
            "tools/future/odyssey3_adversary.py — ATTACK_FAMILIES and PAST_FAILURE_SHAPES as the attack-shape catalog; this module does not call generate_attacks() on Odyssey II records (schema mismatch on transfer_confidence / scope ladder) and does not reuse cheapest-first ranking",
            "tools/future/odyssey_launch.py phase_listen_policy — concurrent II/III with no global barrier; graph dependency only, does not read the law store or reject vacuous attacks",
            "tools/future/workunit_species.py emit_hcli_workunit / odyssey_ii_transfer_experiment / odyssey_iii_adversarial_experiment",
            "tools/future/workgraph.py make_unit / CPU_ANALYSIS lane — spawned units are graph-admissible",
            "tools/future/negative_index.py refuse_if_dead — recovered as the scar query; atlas-dead transfers are refused via odyssey2_law_store.transfer_candidates",
            "tools/future/propagate.py — LAW deltas land in the store; this listener consumes the store rather than ingesting deltas",
            "tools/future/orchestration.py BINDINGS TRANSFER_LAW / ATTACK_LAW — named frontiers this receipt informs; this lane cannot write the bindings table",
        ],
        "gaps_closed": [
            "shared trigger: qualifying_laws() reads the real Odyssey II store",
            "transfer_targets(law) ranked by expected information gain, identity rejected as not a transfer",
            "attacks(law) strongest useful in-domain counterexample, not the cheapest out-of-domain miss",
            "vacuous attack (outside preconditions) rejected before scheduling",
            "underspecified law (no recorded preconditions) cannot be attacked",
            "empty law store emits zero units and says why",
            "listen() emits WorkUnits for both listeners with no II/III barrier",
        ],
        "negative_findings": [
            "listeners spawn WorkUnits; they do not execute them and cannot establish that any law holds or fails",
            "Odyssey III generate_attacks() ranks by cost/p_refutation and does not check claimed-domain preconditions; that generator is recovered as a shape catalog, not driven",
            "Odyssey II Law has no preconditions field; claimed domain is derived from scope axes, or taken from an explicit preconditions key when a caller supplies one",
            "orchestration.BINDINGS cannot be updated from this lane; resident-callability of this module is the functions here plus this receipt",
            "negative_index.refuse_if_dead is not a second law schema; atlas-dead transfers already raise in odyssey2_law_store.transfer_candidates",
            (
                f"law store {qualified['store']['source']}: "
                f"n_records={qualified['store']['n_records']} qualifying={qualified['n']}"
            ),
            (
                "underspecified for attack (wildcard organ and/or unnamed claimed-domain axes): "
                + (", ".join(underspecified_ids) if underspecified_ids else "(none)")
                + "; Phase III emits no unit rather than inventing a domain"
            ),
        ],
        "resident_callable": {
            "entry_point": "tools.future.phase_listeners.listen()",
            "workunit": (
                "one CPU_ANALYSIS unit per accepted transfer target and one "
                "CPU_ANALYSIS unit per law for the strongest useful in-domain attack"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.ODYSSEY_TRANSFER.flash-qwen27 + FT.ODYSSEY_ADVERSARY.attacks",
            "fails_closed": (
                "vacuous -> VacuousAttackError; underspecified -> UnderspecifiedLawError; "
                "identity transfer -> NotATransferError; empty store -> zero units with reason, "
                "never a success shape"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/phase_listeners.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), indent=1, sort_keys=True))
        return 0
    if a.listen:
        result = listen()
        print(json.dumps(
            {
                "n_units": result["n_units"],
                "n_phase_ii": result["n_phase_ii"],
                "n_phase_iii": result["n_phase_iii"],
                "reason": result["reason"],
                "reason_code": result["reason_code"],
                "qualifying_n": result["qualifying_n"],
                "concludes_law": result["concludes_law"],
            },
            indent=1,
            sort_keys=True,
        ))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
