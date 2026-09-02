#!/usr/bin/env python3
"""SCAR REEVALUATOR — which old refutations died at a fidelity bar.

FUNCTIONAL_ROLE_PROBE.json recorded that zeroing 40% of a tensor's output
rows moves the hidden state by ~0.0059 cosine. Many representation scars
in this repo were judged against FIDELITY bars (relative L2, cosine 0.99,
reconstruction error), not against capability.

This module classifies the negative-science corpus from each scar's
recorded method and threshold. It does not choose the next experiment,
does not widen a bar into a relaunch, and does not relaunch anything.
The resident owning SUB2_EBPW decides.

It is also the cheap Odyssey II / III activation: a recovered LAW emits
a TRANSFER_TEST WorkUnit (same mechanism, different specimen/architecture)
and a LAW_ATTACK WorkUnit (adversarial attempt to falsify it). Both carry
the law's measured scope, its falsifier, and the result that would retire
it. Scope is copied, never widened; a dropped binding fails the round trip.

    python3 tools/future/scar_reevaluator.py --build
    python3 tools/future/scar_reevaluator.py --emit-follow-on
    python3 -m pytest tools/future/test_scar_reevaluator.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
import json
import re
from typing import Any, Iterable, Mapping

from tools.future._common import (
    HARDWARE_FIELDS,
    REPO,
    _assert_no_hardware_claims,
    git,
    write_receipt,
)
from tools.future import negative_index as ni


RECEIPT = "SCAR_REEVALUATOR.json"
SCHEMA = "hawking.future.scar_reevaluator.v1"
VERSION = 1
RECORDED_BY = "tools/future/scar_reevaluator.py"
EVIDENCE_CLASS = "STATIC_ONLY"
PROBE_REL = "receipts/future/FUNCTIONAL_ROLE_PROBE.json"
CENSUS_REL = "receipts/future/MLP_BYTE_CENSUS.json"

# Odyssey II / III activation. A recovered law emits these two species.
# They are WorkUnit kinds, not a fork of the ten-species catalog.
TRANSFER_TEST = "TRANSFER_TEST"
LAW_ATTACK = "LAW_ATTACK"
FOLLOW_ON_RECEIPT = "ODYSSEY_II_III_ACTIVATION.json"
FOLLOW_ON_SCHEMA = "hawking.future.odyssey_ii_iii_activation.v1"
FOLLOW_ON_VERSION = 1

# Real law, recovered from the Odyssey II store, seeded from a named
# Odyssey-transfer receipt. Not invented.
NAMED_LAW_ID = "LAW-COLD-CONTROL-BEAT-TRANSFER-SEED"
NAMED_LAW_STORE = "receipts/future/ODYSSEY2_LAW_STORE.json"
NAMED_LAW_EVIDENCE = "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json"

SCOPE_BINDING_AXES = (
    "lattice",
    "models",
    "organs",
    "architecture_family",
    "source_model",
    "organ_class",
    "law_id",
)

STRUCTURALLY_REFUTED = "STRUCTURALLY_REFUTED"
POSSIBLY_REOPENABLE = "POSSIBLY_REOPENABLE"
METHOD_UNRECORDED = "METHOD_UNRECORDED"
NOT_A_REFUTATION = "NOT_A_REFUTATION"
CLASSES = (
    STRUCTURALLY_REFUTED,
    POSSIBLY_REOPENABLE,
    METHOD_UNRECORDED,
    NOT_A_REFUTATION,
)

# Cited from tools/future/aux_capability_screen.py:ORGAN_COSINE_BAR and
# tools/future/capability_information_map.py:COSINE_BAR. Context only —
# never written onto a scar that did not name its own bar.
CITED_CAMPAIGN_ORGAN_COSINE_BAR = 0.990
CITED_CAMPAIGN_ORGAN_COSINE_BAR_SOURCE = (
    "tools/future/aux_capability_screen.py:ORGAN_COSINE_BAR"
)

# Family → theoretical EBPW-cut rank (higher = larger named cut). This is
# a table over the family's stated bit-width, not a measurement that the
# family works. Missing family is UNTESTED, not a silent zero that ranks.
FAMILY_EBPW_RANK: dict[str, int] = {
    "binary_quantization": 5,
    "binary": 5,
    "ternary": 4,
    "uniform_q2": 4,
    "low_rank": 4,
    "shared_basis": 3,
    "hadamard_lattice": 3,
    "activation_corrected_q3": 3,
    "uniform_q3": 3,
    "residual_codebook": 2,
    "kronecker": 2,
    "monarch": 2,
    "butterfly": 2,
    "distilled": 2,
    "mlp_function_replacement": 2,
    "uniform_q4": 1,
}

# Organ → token_ns opportunity rank from which organ the scar names.
# Numeric token_ns is NOT claimed; the rank is an ordinal over named organs.
# mlp_gate_up / mlp_down dominate GAP_LEDGER_60.json; that is the citation.
ORGAN_TOKEN_RANK: dict[str, int] = {
    "gate": 5,
    "up": 5,
    "down": 5,
    "mlp": 5,
    "deltanet": 4,
    "attention": 3,
    "routed_experts": 3,
    "whole_model": 3,
    "kv": 2,
    "lm_head": 2,
    "router": 1,
    "embed": 1,
}

# Lower is cheaper to re-try as a codec that already ran. Missing → UNTESTED.
FAMILY_IMPL_COST_RANK: dict[str, int] = {
    "uniform_q2": 1,
    "uniform_q3": 1,
    "uniform_q4": 1,
    "ternary": 1,
    "binary_quantization": 2,
    "binary": 2,
    "hadamard_lattice": 2,
    "activation_corrected_q3": 2,
    "residual_codebook": 2,
    "low_rank": 3,
    "shared_basis": 3,
    "kronecker": 4,
    "monarch": 4,
    "butterfly": 4,
    "distilled": 4,
    "mlp_function_replacement": 4,
}

LIVE_VERDICT_MARKERS = (
    "live and convergent",
    "live_reopen_holds",
    "not closed",
    "untested",
    "positive entry",
    "artifact_of_method",
    "named exception",
    "live.",
)

# Cosine used as evidence that a shared structure does not exist.
STRUCTURE_COSINE_MARKERS = (
    "pairwise",
    "off-diagonal",
    "mutually orthogonal",
    "near-orthogonal",
    "near orthogonal",
    "shared template",
    "no shared component",
    "no shared direction",
    "no shared basis",
    "inter-expert",
    "cross-expert",
    "head redundancy",
    "mean pairwise expert cosine",
    "row-normalized mean",
    "experts are orthogonal",
    "experts are genuinely mutually orthogonal",
    "experts do not share",
)

FIDELITY_METHOD_MARKERS = (
    "component_sensitive_organ_gate",
    "teacher tensor matvec",
    "organ_output_cosine",
    "organ cosine",
    "functional cosine",
    "weight cosine",
    "held-out relative l2",
    "held-out relative-l2",
    "held out relative l2",
    "held-out error",
    "held-out activation reconstruction",
    "relative l2",
    "relative_l2",
    "rel_fro",
    "rel-fro",
    "reconstruction error",
    "mean-row output cosine",
    "mean row output cosine",
    "cosine >= 0.99",
    "cosine ≥ 0.99",
    "cosine >= 0.990",
    "cosine ≥ 0.990",
)

CAPABILITY_ALREADY_FAILED_MARKERS = (
    "generation incoherent",
    "incoherent generation",
    "draft acceptance",
    "token acceptance alpha = 0",
    "acceptance alpha = 0",
    "gibberish",
)

# Cosine treated as a GO / capability certificate. Changing the number
# does not turn a fidelity screen into capability.
COSINE_AS_CAPABILITY_MARKERS = (
    "as a capability certificate",
    "as evidence the codec preserved function",
    "raw activation cosine",
    "treat_raw_activation_cosine",
    "organ_cosine_0_86_0_90",
)

# Reconstruction of one expert from others is a structure-existence test.
STRUCTURE_RECONSTRUCTION_MARKERS = (
    "surviving expert",
    "omitted moe expert",
    "omitted expert",
    "best-single-survivor",
    "best single surviving",
    "no shared component to subtract",
    "dead on arrival: there is no shared",
    "reconstruct an omitted",
)


class ReevaluatorRefused(RuntimeError):
    """An input is missing; guessing a class would be a fake reopening."""


class FollowOnError(RuntimeError):
    """A law cannot emit follow-on work; guessing a law or widening scope is refused."""


class ScopeBindingDropped(FollowOnError):
    """The emitted WorkUnit no longer carries the law's measured scope."""


_SOURCE_EVIDENCE: dict[str, dict[str, dict[str, Any]]] = {}


def _txt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, default=str)
    return str(v)


def _low(v: Any) -> str:
    return " ".join(_txt(v).lower().split())


def probe_tolerance() -> dict[str, Any]:
    """The robustness number that makes fidelity bars the question.

    REFUSES if the probe receipt is missing or lacks robustness. The
    0.0059 figure is not typed here. Sparse checkout: git show HEAD:rel
    is a real receipt, not a missing one.
    """
    path = REPO / PROBE_REL
    doc: dict[str, Any] | None = None
    if path.is_file():
        doc = json.loads(path.read_text())
    else:
        text, _origin = ni.read_text(PROBE_REL)
        if text is None:
            raise ReevaluatorRefused(
                f"{PROBE_REL} is not on disk and not in git HEAD; a scar "
                "reevaluator with no robustness receipt would be inventing "
                "the capability tolerance"
            )
        doc = json.loads(text)
    rob = doc.get("robustness")
    if not isinstance(rob, dict):
        raise ReevaluatorRefused(f"{PROBE_REL} has no robustness block")
    for key in ("at_fraction_zeroed", "worst_damage", "worst_tensor", "worst_layer"):
        if key not in rob:
            raise ReevaluatorRefused(f"{PROBE_REL} robustness is missing {key}")
    if rob["worst_damage"] is None:
        raise ReevaluatorRefused(f"{PROBE_REL} robustness.worst_damage is missing")
    return {
        "at_fraction_zeroed": rob["at_fraction_zeroed"],
        "worst_damage": rob["worst_damage"],
        "worst_tensor": rob["worst_tensor"],
        "worst_layer": rob["worst_layer"],
        "measure": doc.get("measure"),
        "statement": rob.get("statement"),
        "caveat": rob.get("caveat"),
        "source": PROBE_REL,
    }


def _as_record(scar: Any) -> dict[str, Any]:
    if scar is None:
        raise ReevaluatorRefused("scar is missing; refuse rather than default")
    if isinstance(scar, ni.Scar):
        return scar.to_dict()
    if isinstance(scar, Mapping):
        return dict(scar)
    raise ReevaluatorRefused(
        f"scar must be a Scar or dict, got {type(scar).__name__}"
    )


def _is_live(record: Mapping[str, Any]) -> bool:
    verd = _low(record.get("verdict"))
    status = _low(record.get("status"))
    blob = f"{verd} {status}"
    if any(m in blob for m in LIVE_VERDICT_MARKERS):
        return True
    if "artifact_of_method" in blob:
        return True
    return False


def _has_recorded_method(record: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    for key in (
        "failure_mechanism",
        "claim_refuted",
        "verdict",
        "hypothesis_family",
        "reopen_condition",
    ):
        v = record.get(key)
        if v and v != ni.UNRECORDED:
            return True
    if evidence.get("failure_reason"):
        return True
    if evidence.get("functional_cosine") is not None:
        return True
    if evidence.get("held_out_relative_l2") is not None:
        return True
    return False


def _blob(record: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    parts = [
        record.get("verdict"),
        record.get("failure_mechanism"),
        record.get("claim_refuted"),
        record.get("reopen_condition"),
        record.get("hypothesis_family"),
        evidence.get("failure_reason"),
        evidence.get("functional_claim_boundary"),
        evidence.get("mechanism"),
    ]
    return _low(" ".join(_txt(p) for p in parts if p))


def _is_structure_existence_cosine(blob: str) -> bool:
    if "cosine" not in blob:
        return False
    return any(m in blob for m in STRUCTURE_COSINE_MARKERS)


def _is_structure_existence_method(blob: str) -> bool:
    if _is_structure_existence_cosine(blob):
        return True
    if any(m in blob for m in STRUCTURE_RECONSTRUCTION_MARKERS):
        return True
    if any(m in blob for m in COSINE_AS_CAPABILITY_MARKERS):
        return True
    return False


def _is_fidelity_method(record: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    """True iff the recorded method is a reconstruction/fidelity bar.

    Pairwise expert cosine is a structure-existence test, not a codec bar,
    even though the word cosine appears. Organ-gate teacher-matvec is a
    fidelity bar even when reopen_condition mentions full-model parity.
    """
    reason = _low(evidence.get("failure_reason") or record.get("claim_refuted"))
    if "component_sensitive_organ_gate" in reason:
        return True
    if evidence.get("functional_cosine") is not None:
        return True
    if evidence.get("functional_relative_l2") is not None:
        return True
    blob = _blob(record, evidence)
    if _is_structure_existence_method(blob):
        return False
    if any(m in blob for m in CAPABILITY_ALREADY_FAILED_MARKERS):
        # Already a capability failure. A looser cosine bar does not reopen it.
        return False
    if evidence.get("held_out_relative_l2") is not None:
        return True
    return any(m in blob for m in FIDELITY_METHOD_MARKERS)


def _extract_named_threshold(blob: str) -> dict[str, Any] | None:
    """A numeric bar the scar itself named. None if the scar did not name one."""
    m = re.search(
        r"cosine\s*(?:bar|gate)?\s*(?:>=|≥|>|=|:)?\s*(0\.9\d+)",
        blob,
    )
    if m:
        return {
            "kind": "cosine",
            "bar": float(m.group(1)),
            "named_on_scar": True,
        }
    m = re.search(
        r"(?:relative[\s_-]*l2|rel_fro|rel-err|relative error|held-out error)"
        r"[\s\w]{0,40}?(?:stays above|above|>=|≤|<=|<|>|kill(?:\s+is)?|vs kill)?"
        r"\s*(0\.\d+)",
        blob,
    )
    if m:
        return {
            "kind": "relative_l2",
            "bar": float(m.group(1)),
            "named_on_scar": True,
        }
    m = re.search(
        r"reconstruction error\s*(?:<=|≤|>=|≥|<|>)?\s*(0\.\d+)",
        blob,
    )
    if m:
        return {
            "kind": "reconstruction_error",
            "bar": float(m.group(1)),
            "named_on_scar": True,
        }
    return None


def _died_at(record: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    """The threshold the scar died at, from the record, never invented."""
    out: dict[str, Any] = {
        "summary": "UNRECORDED",
        "kind": "UNRECORDED",
        "measured": None,
        "bar": None,
        "named_on_scar": False,
        "source": "scar_fields",
    }
    if evidence.get("functional_cosine") is not None:
        measured = float(evidence["functional_cosine"])
        rel = evidence.get("functional_relative_l2")
        summary = f"functional cosine {measured}"
        if rel is not None:
            summary += f"; relative_l2 {rel}"
        summary += " (component_sensitive_organ_gate_failed)"
        out.update(
            {
                "summary": summary,
                "kind": "organ_output_cosine",
                "measured": measured,
                "measured_relative_l2": rel,
                "bar": None,
                "named_on_scar": False,
                "source": "measured_outcome.functional",
                "gate_name": "component_sensitive_organ_gate",
                "numeric_bar_on_scar": False,
            }
        )
        return out
    if evidence.get("held_out_relative_l2") is not None:
        measured = float(evidence["held_out_relative_l2"])
        kill = evidence.get("held_out_kill_rel")
        summary = f"held-out relative L2 {measured}"
        if kill is not None:
            summary += f" vs kill {kill}"
        out.update(
            {
                "summary": summary,
                "kind": "held_out_relative_l2",
                "measured": measured,
                "bar": float(kill) if kill is not None else None,
                "named_on_scar": kill is not None,
                "source": "source_receipt.held_out_relative_l2_best",
            }
        )
        return out
    blob = _blob(record, evidence)
    named = _extract_named_threshold(blob)
    if named:
        out.update(
            {
                "summary": f"{named['kind']} bar {named['bar']}",
                "kind": named["kind"],
                "bar": named["bar"],
                "named_on_scar": True,
                "source": "scar_text",
            }
        )
        return out
    if "component_sensitive_organ_gate" in blob:
        out.update(
            {
                "summary": "component_sensitive_organ_gate_failed (numeric bar not on scar)",
                "kind": "organ_output_cosine",
                "gate_name": "component_sensitive_organ_gate",
                "numeric_bar_on_scar": False,
            }
        )
        return out
    claim = _txt(record.get("claim_refuted") or record.get("failure_mechanism"))
    if claim and claim != ni.UNRECORDED:
        out["summary"] = claim[:240]
        out["kind"] = "recorded_claim"
    return out


def consult_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Ask the scoped scar registry whether this candidate is banned.

    Calls tools.future.autonomy_scars.consult (the HCLI entry) and then
    classify() on any in-scope block. An out-of-scope scar is not a ban
    even when classify() would call it STRUCTURALLY_REFUTED.
    """
    from tools.future import autonomy_scars as asc

    verdict = asc.consult(candidate)
    classified: list[dict[str, Any]] = []
    for hit in verdict.get("blocked_by") or []:
        row = classify(hit)
        classified.append(
            {
                "scar_id": hit.get("scar_id"),
                "class": row.get("class"),
                "tolerance_change_reopens": row.get("tolerance_change_reopens"),
                "source_path": hit.get("source_path"),
            }
        )
    return {
        "blocked": bool(verdict.get("blocked")),
        "blocked_by": list(verdict.get("blocked_by") or []),
        "out_of_scope_not_blocked": list(verdict.get("out_of_scope_not_blocked") or []),
        "classifications": classified,
        "laws": list(verdict.get("laws") or []),
        "laws_out_of_scope": list(verdict.get("laws_out_of_scope") or []),
        "entry_point": "tools.future.scar_reevaluator.consult_candidate",
        "consults": "tools.future.autonomy_scars.consult",
        "evidence_class": "STATIC",
    }


def classify(
    scar: Any,
    *,
    evidence: Mapping[str, Any] | None = None,
    fidelity_cosine_bar: float | None = None,
) -> dict[str, Any]:
    """Classify one scar from its recorded method and threshold.

    `fidelity_cosine_bar` is a counterfactual reconstruction bar used only
    to prove that a STRUCTURALLY_REFUTED scar does not flip when the bar
    moves. It is not a default, and it is not a relaunch.
    """
    record = _as_record(scar)
    ev = dict(evidence or {})
    parse_status = record.get("parse_status") or ni.PARSED
    if parse_status != ni.PARSED:
        cls = METHOD_UNRECORDED
        method = "unparsed"
        tolerance_change_reopens = False
    elif _is_live(record):
        cls = NOT_A_REFUTATION
        method = "not_a_current_refutation"
        tolerance_change_reopens = False
    elif not _has_recorded_method(record, ev):
        cls = METHOD_UNRECORDED
        method = "unrecorded"
        tolerance_change_reopens = False
    elif _is_fidelity_method(record, ev):
        cls = POSSIBLY_REOPENABLE
        method = "fidelity_threshold"
        tolerance_change_reopens = True
    else:
        cls = STRUCTURALLY_REFUTED
        method = "non_fidelity"
        tolerance_change_reopens = False

    # A counterfactual cosine bar must not reopen a structural scar.
    if cls == STRUCTURALLY_REFUTED and fidelity_cosine_bar is not None:
        tolerance_change_reopens = False
        cls = STRUCTURALLY_REFUTED

    died = _died_at(record, ev)
    family = record.get("hypothesis_family") or ni.UNRECORDED
    organ = record.get("organ") or ni.UNRECORDED
    row = {
        "scar_id": record.get("scar_id") or record.get("original_id") or "",
        "original_id": record.get("original_id") or "",
        "source_path": record.get("source_path") or "",
        "hypothesis_family": family,
        "organ": organ,
        "model": record.get("model") or ni.UNRECORDED,
        "representation": record.get("representation") or ni.UNRECORDED,
        "verdict": record.get("verdict") or ni.UNRECORDED,
        "class": cls,
        "method": method,
        "died_at": died,
        "tolerance_change_reopens": tolerance_change_reopens,
        "refuse_eligible": bool(record.get("refuse_eligible")),
        "level": record.get("level") or "",
        "reopen_condition": record.get("reopen_condition") or ni.UNRECORDED,
        "claim_refuted": record.get("claim_refuted") or ni.UNRECORDED,
        "failure_mechanism": record.get("failure_mechanism") or ni.UNRECORDED,
    }
    if fidelity_cosine_bar is not None:
        row["counterfactual_fidelity_cosine_bar"] = float(fidelity_cosine_bar)
    return row


def _index_jsonl(text: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        rid = obj.get("record_id") or obj.get("id")
        if not rid:
            continue
        mo = obj.get("measured_outcome") if isinstance(obj.get("measured_outcome"), dict) else {}
        fn = mo.get("functional") if isinstance(mo.get("functional"), dict) else {}
        wt = mo.get("weight") if isinstance(mo.get("weight"), dict) else {}
        ev: dict[str, Any] = {
            "failure_reason": obj.get("failure_reason"),
            "functional_claim_boundary": fn.get("claim_boundary") or mo.get("claim_boundary"),
        }
        if isinstance(fn.get("cosine"), (int, float)):
            ev["functional_cosine"] = float(fn["cosine"])
        if isinstance(fn.get("relative_l2"), (int, float)):
            ev["functional_relative_l2"] = float(fn["relative_l2"])
        if isinstance(wt.get("cosine"), (int, float)):
            ev["weight_cosine"] = float(wt["cosine"])
        out[str(rid)] = ev
    return out


def _index_json(text: str) -> dict[str, dict[str, Any]]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    kill = None
    cands = obj.get("candidates")
    if isinstance(cands, list):
        for c in cands:
            if isinstance(c, dict) and isinstance(c.get("held_out_kill_rel"), (int, float)):
                kill = float(c["held_out_kill_rel"])
                break
    rows = obj.get("scars") or obj.get("entries") or []
    if isinstance(rows, dict):
        items: Iterable[tuple[Any, Any]] = rows.items()
    elif isinstance(rows, list):
        items = [
            (
                (r.get("id") or r.get("family") or r.get("record_id") or f"i{i}"),
                r,
            )
            for i, r in enumerate(rows)
            if isinstance(r, dict)
        ]
    else:
        items = []
    for key, rec in items:
        if not isinstance(rec, dict):
            continue
        ev: dict[str, Any] = {
            "mechanism": rec.get("mechanism") or rec.get("killed_by"),
            "failure_reason": rec.get("failure_reason") or rec.get("claim_refuted"),
        }
        for cand_key in (
            "held_out_relative_l2_best",
            "held_out_relative_l2",
            "best_held_out_relative_l2",
        ):
            if isinstance(rec.get(cand_key), (int, float)):
                ev["held_out_relative_l2"] = float(rec[cand_key])
                break
        if kill is not None:
            ev["held_out_kill_rel"] = kill
        elif isinstance(rec.get("held_out_kill_rel"), (int, float)):
            ev["held_out_kill_rel"] = float(rec["held_out_kill_rel"])
        out[str(key)] = ev
        fam = rec.get("family") or rec.get("id")
        if fam and str(fam) not in out:
            out[str(fam)] = ev
    mo = obj.get("measured_outcome") if isinstance(obj.get("measured_outcome"), dict) else {}
    fn = mo.get("functional") if isinstance(mo.get("functional"), dict) else {}
    if fn or mo:
        rid = obj.get("id") or "doc"
        ev = out.setdefault(str(rid), {})
        if isinstance(fn.get("cosine"), (int, float)):
            ev["functional_cosine"] = float(fn["cosine"])
        if isinstance(fn.get("relative_l2"), (int, float)):
            ev["functional_relative_l2"] = float(fn["relative_l2"])
    return out


def load_source_evidence(rel: str) -> dict[str, dict[str, Any]]:
    """Measured method/threshold from the scar's source. Missing source is empty."""
    if rel in _SOURCE_EVIDENCE:
        return _SOURCE_EVIDENCE[rel]
    text, _origin = ni.read_text(rel)
    if text is None:
        _SOURCE_EVIDENCE[rel] = {}
        return {}
    low = rel.lower()
    if low.endswith(".jsonl"):
        ev = _index_jsonl(text)
    elif low.endswith(".json"):
        ev = _index_json(text)
    else:
        ev = {}
    _SOURCE_EVIDENCE[rel] = ev
    return ev


def _evidence_for(record: Mapping[str, Any]) -> dict[str, Any]:
    rel = str(record.get("source_path") or "")
    if not rel:
        return {}
    idx = load_source_evidence(rel)
    for key in (
        record.get("original_id"),
        record.get("scar_id"),
        (record.get("scar_id") or "").split("#")[-1],
    ):
        if key and str(key) in idx:
            return idx[str(key)]
    return {}


def classify_corpus(scars: list[Any] | None = None) -> list[dict[str, Any]]:
    """Classify every ingested scar. Empty corpus REFUSES."""
    pool = scars if scars is not None else ni.ingest()
    if not pool:
        raise ReevaluatorRefused(
            "negative_index ingested zero scars; classifying an empty "
            "corpus would invent a distribution"
        )
    out: list[dict[str, Any]] = []
    for s in pool:
        record = _as_record(s)
        ev = _evidence_for(record)
        # Fixture dicts may carry measured_outcome directly.
        if not ev and isinstance(record.get("measured_outcome"), dict):
            mo = record["measured_outcome"]
            fn = mo.get("functional") if isinstance(mo.get("functional"), dict) else {}
            ev = {}
            if isinstance(fn.get("cosine"), (int, float)):
                ev["functional_cosine"] = float(fn["cosine"])
            if isinstance(fn.get("relative_l2"), (int, float)):
                ev["functional_relative_l2"] = float(fn["relative_l2"])
            if record.get("claim_refuted"):
                ev["failure_reason"] = record.get("claim_refuted")
        out.append(classify(record, evidence=ev))
    return out


def _census_shares() -> dict[str, float] | None:
    path = REPO / CENSUS_REL
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    rows = (doc.get("census") or {}).get("by_organ")
    if not isinstance(rows, list):
        raise ReevaluatorRefused(f"{CENSUS_REL} is missing census.by_organ")
    shares: dict[str, float] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        organ = str(r.get("organ") or "")
        share = r.get("share_of_active")
        if not organ or not isinstance(share, (int, float)):
            continue
        shares[organ] = float(share)
        fam = str(r.get("family") or "")
        if fam:
            shares[fam] = shares.get(fam, 0.0) + float(share)
    if not shares:
        raise ReevaluatorRefused(f"{CENSUS_REL} by_organ had no usable shares")
    return shares


def _ebpw_rank(family: str) -> tuple[int, str]:
    slug = ni.canon_family(family) if family and family != ni.UNRECORDED else ""
    if slug in FAMILY_EBPW_RANK:
        return FAMILY_EBPW_RANK[slug], slug
    low = (family or "").lower()
    for key, rank in FAMILY_EBPW_RANK.items():
        if key in low:
            return rank, key
    return 0, "UNTESTED"


def _token_rank(organ: str) -> tuple[int, str]:
    if not organ or organ == ni.UNRECORDED:
        return 0, "UNTESTED"
    slug = ni.canon_organ(organ)
    if slug in ORGAN_TOKEN_RANK:
        return ORGAN_TOKEN_RANK[slug], slug
    return 0, "UNTESTED"


def _cost_rank(family: str) -> tuple[int, str]:
    slug = ni.canon_family(family) if family and family != ni.UNRECORDED else ""
    if slug in FAMILY_IMPL_COST_RANK:
        return FAMILY_IMPL_COST_RANK[slug], slug
    low = (family or "").lower()
    for key, rank in FAMILY_IMPL_COST_RANK.items():
        if key in low:
            return rank, key
    return 0, "UNTESTED"


def rank_reopenable(
    rows: list[Mapping[str, Any]],
    *,
    census_shares: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rank POSSIBLY_REOPENABLE rows. Does not relaunch them."""
    ranked: list[dict[str, Any]] = []
    for row in rows:
        if row.get("class") != POSSIBLY_REOPENABLE:
            continue
        family = str(row.get("hypothesis_family") or "")
        organ = str(row.get("organ") or "")
        ebpw_rank, ebpw_key = _ebpw_rank(family)
        token_rank, token_key = _token_rank(organ)
        cost_rank, cost_key = _cost_rank(family)
        share: Any = "UNTESTED"
        if census_shares is not None:
            if token_key != "UNTESTED" and token_key in census_shares:
                share = census_shares[token_key]
            elif organ in census_shares:
                share = census_shares[organ]
        item = {
            "scar_id": row.get("scar_id"),
            "source_path": row.get("source_path"),
            "hypothesis_family": family,
            "organ": organ,
            "model": row.get("model"),
            "method": row.get("method"),
            "died_at_threshold": (row.get("died_at") or {}).get("summary"),
            "died_at": row.get("died_at"),
            "theoretical_ebpw_reduction_rank": ebpw_rank if ebpw_key != "UNTESTED" else 0,
            "theoretical_ebpw_reduction": ebpw_key if ebpw_key != "UNTESTED" else "UNTESTED",
            "token_ns_opportunity_rank": token_rank if token_key != "UNTESTED" else 0,
            "token_ns_opportunity": (
                {5: "HIGH", 4: "HIGH", 3: "MEDIUM", 2: "LOW", 1: "LOW"}.get(token_rank, "UNTESTED")
                if token_key != "UNTESTED"
                else "UNTESTED"
            ),
            "implementation_cost_rank": cost_rank if cost_key != "UNTESTED" else 0,
            "implementation_cost": (
                {1: "LOW", 2: "LOW", 3: "MEDIUM", 4: "HIGH"}.get(cost_rank, "UNTESTED")
                if cost_key != "UNTESTED"
                else "UNTESTED"
            ),
            "cited_organ_share_of_active_bytes": share,
            "ranking_is_not_a_relaunch": True,
        }
        ranked.append(item)
    ranked.sort(
        key=lambda r: (
            -int(r["theoretical_ebpw_reduction_rank"]),
            -int(r["token_ns_opportunity_rank"]),
            int(r["implementation_cost_rank"]) if r["implementation_cost"] != "UNTESTED" else 99,
            str(r["scar_id"]),
        )
    )
    return ranked


def _group_families(ranked: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per hypothesis_family so the top list is not 400 JSONL clones."""
    groups: dict[str, dict[str, Any]] = {}
    for row in ranked:
        fam = str(row.get("hypothesis_family") or "unrecorded")
        g = groups.get(fam)
        if g is None:
            groups[fam] = {
                "hypothesis_family": fam,
                "n_scars": 1,
                "organs": [row.get("organ")],
                "models": [row.get("model")],
                "theoretical_ebpw_reduction_rank": row["theoretical_ebpw_reduction_rank"],
                "theoretical_ebpw_reduction": row["theoretical_ebpw_reduction"],
                "token_ns_opportunity_rank": row["token_ns_opportunity_rank"],
                "token_ns_opportunity": row["token_ns_opportunity"],
                "implementation_cost_rank": row["implementation_cost_rank"],
                "implementation_cost": row["implementation_cost"],
                "representative_scar_id": row.get("scar_id"),
                "representative_source_path": row.get("source_path"),
                "died_at_threshold": row.get("died_at_threshold"),
                "died_at": row.get("died_at"),
                "ranking_is_not_a_relaunch": True,
            }
            continue
        g["n_scars"] += 1
        if row.get("organ") not in g["organs"]:
            g["organs"].append(row.get("organ"))
        if row.get("model") not in g["models"]:
            g["models"].append(row.get("model"))
        # Keep the representative whose measured fidelity is closest to
        # passing (highest cosine / lowest relative L2). Mechanical, not a pick.
        cur = g.get("died_at") or {}
        nxt = row.get("died_at") or {}
        cur_m = cur.get("measured")
        nxt_m = nxt.get("measured")
        take = False
        if nxt_m is not None and cur_m is None:
            take = True
        elif nxt_m is not None and cur_m is not None:
            kind = str(nxt.get("kind") or "")
            if kind in {"organ_output_cosine", "cosine"} and float(nxt_m) > float(cur_m):
                take = True
            if kind in {"held_out_relative_l2", "relative_l2", "reconstruction_error"} and float(nxt_m) < float(cur_m):
                take = True
        if take:
            g["representative_scar_id"] = row.get("scar_id")
            g["representative_source_path"] = row.get("source_path")
            g["died_at_threshold"] = row.get("died_at_threshold")
            g["died_at"] = row.get("died_at")
        g["token_ns_opportunity_rank"] = max(
            int(g["token_ns_opportunity_rank"]), int(row["token_ns_opportunity_rank"])
        )
        if int(row["token_ns_opportunity_rank"]) >= int(g["token_ns_opportunity_rank"]):
            g["token_ns_opportunity"] = row["token_ns_opportunity"]
    grouped = list(groups.values())
    grouped.sort(
        key=lambda r: (
            -int(r["theoretical_ebpw_reduction_rank"]),
            -int(r["token_ns_opportunity_rank"]),
            int(r["implementation_cost_rank"]) if r["implementation_cost"] != "UNTESTED" else 99,
            -int(r["n_scars"]),
            str(r["hypothesis_family"]),
        )
    )
    return grouped


def counts(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = {c: 0 for c in CLASSES}
    for r in rows:
        cls = r.get("class")
        if cls in n:
            n[cls] += 1
        else:
            n[str(cls)] = n.get(str(cls), 0) + 1
    return {
        "n_scars": len(rows),
        "by_class": n,
        "n_structurally_refuted": n[STRUCTURALLY_REFUTED],
        "n_possibly_reopenable": n[POSSIBLY_REOPENABLE],
        "n_method_unrecorded": n[METHOD_UNRECORDED],
        "n_not_a_refutation": n[NOT_A_REFUTATION],
    }


# ---------------------------------------------------------------------------
# Odyssey II / III activation: a LAW emits TRANSFER_TEST and LAW_ATTACK.
# Connects existing gates; does not fork a second lattice or adversary.
# ---------------------------------------------------------------------------


def _ols_from_store_record(raw: Mapping[str, Any]) -> Any:
    """Rebuild the Odyssey II Law dataclass from a store receipt row."""
    from tools.future import odyssey2_law_store as ols

    conf = raw.get("transfer_confidence")
    if not isinstance(conf, Mapping):
        raise FollowOnError(
            f"{raw.get('law_id')}: transfer_confidence is not an object; "
            "refusing to invent a confidence"
        )
    cands_raw = raw.get("transfer_candidates") or ()
    cands: list[dict[str, Any]] = []
    for row in cands_raw:
        if isinstance(row, Mapping):
            cands.append(dict(row))
        else:
            cands.append({"target_model": str(row)})
    law = ols.Law(
        law_id=str(raw.get("law_id") or ""),
        statement=str(raw.get("statement") or ""),
        source_model=str(raw.get("source_model") or ""),
        source_device=str(raw.get("source_device") or "UNKNOWN"),
        architecture_family=str(raw.get("architecture_family") or "UNKNOWN"),
        organ_class=str(raw.get("organ_class") or ""),
        backend=str(raw.get("backend") or "UNKNOWN"),
        evidence_strength=str(raw.get("evidence_strength") or "STATIC"),
        evidence_refs=tuple(str(x) for x in (raw.get("evidence_refs") or ())),
        scope=str(raw.get("scope") or ""),
        transfer_candidates=tuple(cands),
        transfer_confidence=dict(conf),
        counterexample_requirement=str(raw.get("counterexample_requirement") or ""),
        expected_saved_experiments=raw.get("expected_saved_experiments"),
        actual_saved_experiments=raw.get("actual_saved_experiments"),
        time_to_first_useful_executable_ns=None,
    )
    return ols.validate_law(law)


def recover_named_law(law_id: str = NAMED_LAW_ID) -> dict[str, Any]:
    """Load a real law from the Odyssey II store. Never invent one.

    Calls:
      tools.future.odyssey2_law_store.load_repo_json
      tools.future.odyssey2_law_store.validate_law
      tools.future.autonomy_scars.law_from_odyssey2_record
      tools.future.autonomy_scars.round_trip_law
    """
    from tools.future import autonomy_scars as asc
    from tools.future import odyssey2_law_store as ols

    doc = ols.load_repo_json(NAMED_LAW_STORE)
    rows = [r for r in (doc.get("laws") or []) if isinstance(r, Mapping) and r.get("law_id") == law_id]
    if not rows:
        raise FollowOnError(
            f"{law_id} is not in {NAMED_LAW_STORE}; refusing to invent a law"
        )
    raw = dict(rows[0])
    ols_law = _ols_from_store_record(raw)
    ols.validate_law(ols_law)
    asc_law = asc.law_from_odyssey2_record(raw, NAMED_LAW_STORE)
    again = asc.round_trip_law(asc_law)
    if again.identity != asc_law.identity or again.scope.to_dict() != asc_law.scope.to_dict():
        raise FollowOnError(f"{law_id}: autonomy_scars.round_trip_law dropped scope")
    if not ols_law.counterexample_requirement.strip():
        raise FollowOnError(f"{law_id}: law has no falsifier; cannot emit follow-on")
    if ols_law.scope not in ols.SCOPES:
        raise FollowOnError(f"{law_id}: scope {ols_law.scope!r} is not on the II lattice")
    evidence = list(ols_law.evidence_refs)
    return {
        "law_id": ols_law.law_id,
        "statement": ols_law.statement,
        "scope": ols_law.scope,
        "falsifier": ols_law.counterexample_requirement,
        "source_path": NAMED_LAW_STORE,
        "evidence_refs": evidence,
        "source_model": ols_law.source_model,
        "source_device": ols_law.source_device,
        "architecture_family": ols_law.architecture_family,
        "organ_class": ols_law.organ_class,
        "backend": ols_law.backend,
        "evidence_strength": ols_law.evidence_strength,
        "transfer_candidates": [dict(c) for c in ols_law.transfer_candidates],
        "transfer_confidence": dict(ols_law.transfer_confidence),
        "ols_law": ols_law,
        "asc_law": again,
        "store_record": raw,
        "named": law_id == NAMED_LAW_ID,
        "evidence_class": "STATIC",
    }


def law_scope_binding(recovered: Mapping[str, Any]) -> dict[str, Any]:
    """The measured scope that must survive emit -> JSON -> load.

    This is the law's origin, not the transfer target. Writing the target
    specimen into models would silently generalise the law.
    """
    asc_law = recovered["asc_law"]
    scope = asc_law.scope.to_dict()
    return {
        "lattice": recovered["scope"],
        "models": list(scope.get("models") or []),
        "organs": list(scope.get("organs") or []),
        "machine": scope.get("machine") or recovered.get("source_device") or "",
        "architecture_family": recovered["architecture_family"],
        "backend": recovered.get("backend") or "",
        "source_model": recovered["source_model"],
        "source_device": recovered.get("source_device") or "",
        "organ_class": recovered["organ_class"],
        "law_id": recovered["law_id"],
        "source_path": recovered["source_path"],
        "evidence_refs": list(recovered.get("evidence_refs") or []),
        "falsifier": recovered["falsifier"],
    }


def extract_scope_binding(unit: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = unit.get("scope_binding")
    if not isinstance(raw, Mapping):
        return None
    return dict(raw)


def scope_binding_intact(expected: Mapping[str, Any], unit: Mapping[str, Any]) -> bool:
    """True iff every bound axis on the law is present and equal on the unit.

    A missing scope_binding, an empty one, or a rewritten lattice/model is
    a dropped binding — the TRANSFER_TEST would no longer police silent
    generalisation.
    """
    got = extract_scope_binding(unit)
    if not got:
        return False
    want = expected.get("scope_binding") if "scope_binding" in expected else expected
    if not isinstance(want, Mapping):
        return False
    for axis in SCOPE_BINDING_AXES:
        if axis not in got:
            return False
        if got.get(axis) != want.get(axis):
            return False
    if not str(got.get("falsifier") or "").strip():
        return False
    if str(got.get("lattice") or "") != str(want.get("lattice") or recovered_lattice(want)):
        return False
    return True


def recovered_lattice(binding: Mapping[str, Any]) -> str:
    return str(binding.get("lattice") or "")


def require_scope_binding(expected: Mapping[str, Any], unit: Mapping[str, Any]) -> dict[str, Any]:
    got = extract_scope_binding(unit)
    if got is None or not scope_binding_intact(expected, unit):
        raise ScopeBindingDropped(
            f"{unit.get('id')}: scope binding dropped or rewritten; "
            f"expected lattice={expected.get('lattice')!r} models={expected.get('models')!r} "
            f"got={got}"
        )
    return got


def round_trip_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    """JSON round-trip. Scope must survive this or the instrument is a lie."""
    return json.loads(json.dumps(unit, sort_keys=True, default=str))


def _match_sealed_target(
    cand: Mapping[str, Any],
    sealed: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    from tools.future import phase_listeners as pl

    target_model = str(cand.get("target_model") or "")
    target_school = str(cand.get("target_school") or "")
    target_fam = str(cand.get("target_architecture_family") or "")
    for alias, row in sealed.items():
        if row.get("source_of_campaign_laws"):
            continue
        if not row.get("whole_tree_verified"):
            continue
        fam = str(row.get("architecture_family") or "")
        if target_fam and fam == target_fam:
            return alias, dict(row)
        repo = str(row.get("repo") or "")
        sid = str(row.get("specimen_id") or "")
        if target_model and (
            pl._same_model(target_model, repo)
            or pl._same_model(target_model, sid)
            or target_model.lower() in alias.lower()
        ):
            return alias, dict(row)
        if target_school and target_school.lower() in alias.lower():
            return alias, dict(row)
    return None, None


def pick_transfer_target(
    recovered: Mapping[str, Any],
    *,
    target_alias: str | None = None,
) -> dict[str, Any]:
    """A different specimen/architecture. Identity transfer is refused.

    Calls:
      tools.future.odyssey2_transfer.sealed_specimens
      tools.future.odyssey2_transfer.require_sealed
      tools.future.odyssey2_transfer.may_transfer
      tools.future.odyssey2_transfer.odyssey_i_barrier
    """
    from tools.future import odyssey2_transfer as o2t

    if o2t.odyssey_i_barrier() is not None:
        raise FollowOnError("Odyssey I barrier is not None; II must not wait")
    ols_law = recovered["ols_law"]
    if target_alias:
        row = o2t.require_sealed(target_alias)
        if not o2t.may_transfer(ols_law, row):
            raise FollowOnError(
                f"{ols_law.law_id}: may_transfer refused {target_alias!r} "
                f"(identity or unsealed)"
            )
        if str(row.get("architecture_family") or "") == ols_law.architecture_family:
            raise FollowOnError(
                f"{ols_law.law_id}: {target_alias!r} is the same architecture "
                f"family {ols_law.architecture_family!r}; TRANSFER_TEST needs a different one"
            )
        return {**dict(row), "alias": target_alias, "from_transfer_candidate": None}

    sealed = o2t.sealed_specimens()
    for cand in ols_law.transfer_candidates:
        if not isinstance(cand, Mapping):
            continue
        cand_fam = str(cand.get("target_architecture_family") or "")
        if cand_fam and cand_fam == ols_law.architecture_family:
            continue
        alias, row = _match_sealed_target(cand, sealed)
        if alias is None or row is None:
            continue
        row = o2t.require_sealed(alias)
        if str(row.get("architecture_family") or "") == ols_law.architecture_family:
            continue
        if o2t.may_transfer(ols_law, row):
            return {**dict(row), "alias": alias, "from_transfer_candidate": dict(cand)}

    for alias, row in sealed.items():
        if row.get("source_of_campaign_laws"):
            continue
        if str(row.get("architecture_family") or "") == ols_law.architecture_family:
            continue
        if not o2t.may_transfer(ols_law, row):
            continue
        sealed_row = o2t.require_sealed(alias)
        return {**dict(sealed_row), "alias": alias, "from_transfer_candidate": None}

    raise FollowOnError(
        f"{ols_law.law_id}: no sealed B-side of a different architecture family; "
        "TRANSFER_TEST has nowhere to go"
    )


def _result_that_would_retire_transfer(recovered: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "on_the_law": recovered["falsifier"],
        "on_this_unit": (
            "TRANSFER_FAILED: the statement does not hold on "
            f"{target.get('specimen_id') or target.get('alias')} "
            f"(family {target.get('architecture_family')}). The law stays at "
            f"{recovered['scope']} on {recovered['source_model']}/"
            f"{recovered['organ_class']} and must not be quoted as generic."
        ),
        "would_also_retire_the_law": recovered["falsifier"],
        "does_not_delete_the_law": True,
        "does_not_widen_scope": True,
    }


def _result_that_would_retire_attack(recovered: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "on_the_law": recovered["falsifier"],
        "on_this_unit": spec.get("falsifier"),
        "expected_if_law_false": spec.get("expected_if_law_false"),
        "target_scope_if_refuted": spec.get("target_scope_if_refuted"),
        "verdict_that_retires": "REFUTED",
        "does_not_delete_the_law": True,
        "does_not_widen_scope": True,
    }


def emit_transfer_test(
    law_id: str = NAMED_LAW_ID,
    *,
    target_alias: str | None = None,
    recovered: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit a TRANSFER_TEST WorkUnit: same mechanism, different specimen.

    Calls:
      tools.future.odyssey2_transfer.may_transfer
      tools.future.odyssey2_transfer.require_sealed
      tools.future.workunit_species.emit_hcli_workunit
      tools.future.workunit_species.validate_emitted_unit
    """
    from tools.future import odyssey2_transfer as o2t
    from tools.future import workunit_species as ws

    rec = dict(recovered) if recovered is not None else recover_named_law(law_id)
    ols_law = rec["ols_law"]
    target = pick_transfer_target(rec, target_alias=target_alias)
    if not o2t.may_transfer(ols_law, target):
        raise FollowOnError(
            f"{ols_law.law_id}: may_transfer returned False for {target.get('alias')}"
        )
    binding = law_scope_binding(rec)
    retire = _result_that_would_retire_transfer(rec, target)
    uid = (
        f"future.{TRANSFER_TEST}.{ols_law.law_id}."
        f"{target.get('alias') or target.get('specimen_id')}"
    )
    extras = {
        "species": TRANSFER_TEST,
        "catalog_species": "odyssey_ii_transfer_experiment",
        "odyssey": "II",
        "law_id": ols_law.law_id,
        "statement": ols_law.statement,
        "mechanism": ols_law.organ_class,
        "scope_binding": binding,
        "falsifier": rec["falsifier"],
        "result_that_would_retire": retire,
        "source_path": rec["source_path"],
        "evidence_refs": list(rec["evidence_refs"]),
        "target_alias": target.get("alias"),
        "target_specimen": target.get("specimen_id"),
        "target_repo": target.get("repo"),
        "target_architecture_family": target.get("architecture_family"),
        "target_whole_tree_verified": target.get("whole_tree_verified"),
        "from_transfer_candidate": target.get("from_transfer_candidate"),
        "source_architecture_family": ols_law.architecture_family,
        "same_mechanism": True,
        "different_specimen": True,
        "different_architecture": str(target.get("architecture_family") or "")
        != ols_law.architecture_family,
        "does_not_widen_scope": True,
        "scope_after_emit": ols_law.scope,
        "may_transfer": True,
        "evidence_class": "STATIC",
        "gates_invoked": [
            "tools.future.odyssey2_transfer.may_transfer",
            "tools.future.odyssey2_transfer.require_sealed",
            "tools.future.odyssey2_transfer.odyssey_i_barrier",
            "tools.future.workunit_species.emit_hcli_workunit",
            "tools.future.workunit_species.validate_emitted_unit",
        ],
        "claim_boundary": (
            "WorkUnit is a proposal. Scope is the law's measured origin, not "
            "the target. A passing diagnostic does not widen MODEL_LOCAL."
        ),
    }
    row = ws.emit_hcli_workunit(
        id=uid,
        role="science",
        description=(
            f"TRANSFER_TEST of {ols_law.law_id} ({ols_law.organ_class} on "
            f"{ols_law.source_model}, scope {ols_law.scope}) onto "
            f"{target.get('alias')} family={target.get('architecture_family')}. "
            f"Falsifier: {rec['falsifier']}"
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.odyssey_ii.law_scope",
        provider="future.scar_reevaluator",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras=extras,
    )
    # emit_hcli_workunit drops None extras; the barrier being None is the deliverable.
    row["odyssey_i_barrier"] = o2t.odyssey_i_barrier()
    ws.validate_emitted_unit(row)
    require_scope_binding(binding, row)
    return row


def emit_law_attack(
    law_id: str = NAMED_LAW_ID,
    *,
    recovered: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit a LAW_ATTACK WorkUnit: an adversarial attempt to falsify the law.

    Calls:
      tools.future.odyssey3_attack.to_o3_law
      tools.future.odyssey3_adversary.validate_law
      tools.future.odyssey3_adversary.generate_attacks
      tools.future.odyssey3_adversary.emit_for_law
      tools.future.workunit_species.emit_hcli_workunit
      tools.future.workunit_species.validate_emitted_unit
    """
    from tools.future import odyssey2_transfer as o2t
    from tools.future import odyssey3_adversary as o3
    from tools.future import odyssey3_attack as o3a
    from tools.future import workunit_species as ws

    rec = dict(recovered) if recovered is not None else recover_named_law(law_id)
    ols_law = rec["ols_law"]
    o3_law = o3a.to_o3_law(ols_law)
    o3.validate_law(o3_law)
    attacks = o3.generate_attacks(o3_law)
    if not attacks:
        raise FollowOnError(f"{ols_law.law_id}: generate_attacks returned nothing")
    plan = o3.emit_for_law(o3_law)
    if plan["n_attacks"] < 1:
        raise FollowOnError(f"{ols_law.law_id}: emit_for_law refused")
    selected = plan["ranked_attacks"][0]
    binding = law_scope_binding(rec)
    retire = _result_that_would_retire_attack(rec, selected)
    uid = f"future.{LAW_ATTACK}.{ols_law.law_id}.{selected['family']}"
    extras = {
        "species": LAW_ATTACK,
        "catalog_species": "odyssey_iii_adversarial_experiment",
        "odyssey": "III",
        "law_id": ols_law.law_id,
        "statement": ols_law.statement,
        "scope_binding": binding,
        "falsifier": rec["falsifier"],
        "attack_falsifier": selected.get("falsifier"),
        "result_that_would_retire": retire,
        "source_path": rec["source_path"],
        "evidence_refs": list(rec["evidence_refs"]),
        "selected_attack_id": selected["attack_id"],
        "selected_family": selected["family"],
        "o3_spec": {
            "attack_id": selected["attack_id"],
            "family": selected["family"],
            "command": list(selected.get("command") or []),
            "falsifier": selected.get("falsifier"),
            "expected_if_law_holds": selected.get("expected_if_law_holds"),
            "expected_if_law_false": selected.get("expected_if_law_false"),
            "target_scope_if_refuted": selected.get("target_scope_if_refuted"),
            "adversarial_target": selected.get("adversarial_target"),
            "transfer_hypothesis": selected.get("transfer_hypothesis"),
        },
        "n_attacks": plan["n_attacks"],
        "n_generate_attacks": len(attacks),
        "ranked_attack_ids": list(plan["ranked_attack_ids"]),
        "does_not_widen_scope": True,
        "scope_after_emit": ols_law.scope,
        "o3_scope": o3_law["scope"],
        "evidence_class": "STATIC",
        "gates_invoked": [
            "tools.future.odyssey3_attack.to_o3_law",
            "tools.future.odyssey3_adversary.validate_law",
            "tools.future.odyssey3_adversary.generate_attacks",
            "tools.future.odyssey3_adversary.emit_for_law",
            "tools.future.workunit_species.emit_hcli_workunit",
            "tools.future.workunit_species.validate_emitted_unit",
        ],
        "claim_boundary": (
            "WorkUnit is a proposal. A SURVIVED attack is a result. "
            "Scope is the law's measured origin; a hit moves scope DOWN "
            "only through odyssey3_adversary.apply_result, not this emit."
        ),
    }
    row = ws.emit_hcli_workunit(
        id=uid,
        role="science",
        description=(
            f"LAW_ATTACK on {ols_law.law_id} via {selected['family']} "
            f"(scope {ols_law.scope}). Falsifier: {rec['falsifier']}. "
            f"Attack falsifier: {selected.get('falsifier')}"
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.odyssey_iii.adversary",
        provider="future.scar_reevaluator",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras=extras,
    )
    row["odyssey_i_barrier"] = o2t.odyssey_i_barrier()
    ws.validate_emitted_unit(row)
    require_scope_binding(binding, row)
    return row


def mutation_check_scope_binding(expected: Mapping[str, Any], unit: Mapping[str, Any]) -> dict[str, Any]:
    """Artifact mutation: dropping scope_binding must fail the intact check.

    This is the load-bearing negative control. A checker that stays green
    with the binding removed is not evidence that scope survived.
    """
    intact = scope_binding_intact(expected, unit)
    dropped = dict(unit)
    dropped.pop("scope_binding", None)
    fails_when_dropped = not scope_binding_intact(expected, dropped)
    raised = False
    try:
        require_scope_binding(expected, dropped)
    except ScopeBindingDropped:
        raised = True
    emptied = dict(unit)
    emptied["scope_binding"] = {}
    fails_when_emptied = not scope_binding_intact(expected, emptied)
    rewritten = dict(unit)
    rewritten["scope_binding"] = dict(extract_scope_binding(unit) or {})
    rewritten["scope_binding"]["lattice"] = "GENERIC_VERIFIED"
    fails_when_widened = not scope_binding_intact(expected, rewritten)
    if not (intact and fails_when_dropped and raised and fails_when_emptied and fails_when_widened):
        raise ScopeBindingDropped(
            f"{unit.get('id')}: mutation check did not fail "
            f"(intact={intact} dropped={fails_when_dropped} raised={raised} "
            f"emptied={fails_when_emptied} widened={fails_when_widened})"
        )
    return {
        "intact_on_emitted_unit": True,
        "fails_when_scope_binding_dropped": True,
        "raises_when_scope_binding_dropped": True,
        "fails_when_scope_binding_emptied": True,
        "fails_when_lattice_silently_widened": True,
        "evidence_class": "STATIC",
    }


def emit_follow_on(law_id: str = NAMED_LAW_ID) -> dict[str, Any]:
    """Recover a real law and emit both follow-on WorkUnits."""
    rec = recover_named_law(law_id)
    binding = law_scope_binding(rec)
    transfer = emit_transfer_test(law_id, recovered=rec)
    attack = emit_law_attack(law_id, recovered=rec)
    transfer_rt = round_trip_unit(transfer)
    attack_rt = round_trip_unit(attack)
    require_scope_binding(binding, transfer_rt)
    require_scope_binding(binding, attack_rt)
    return {
        "law": {
            "law_id": rec["law_id"],
            "statement": rec["statement"],
            "scope": rec["scope"],
            "falsifier": rec["falsifier"],
            "source_path": rec["source_path"],
            "evidence_refs": list(rec["evidence_refs"]),
            "source_model": rec["source_model"],
            "architecture_family": rec["architecture_family"],
            "organ_class": rec["organ_class"],
            "named": rec["named"],
        },
        "scope_binding": binding,
        "transfer_test": transfer,
        "law_attack": attack,
        "round_trip": {
            "transfer_test": round_trip_unit(transfer),
            "law_attack": round_trip_unit(attack),
            "scope_survived_transfer_test": True,
            "scope_survived_law_attack": True,
        },
        "mutation_check": {
            "transfer_test": mutation_check_scope_binding(binding, transfer_rt),
            "law_attack": mutation_check_scope_binding(binding, attack_rt),
        },
        "evidence_class": "STATIC",
    }


def write_follow_on_receipt(follow: Mapping[str, Any] | None = None) -> Any:
    payload = dict(follow) if follow is not None else emit_follow_on()
    transfer = payload["transfer_test"]
    attack = payload["law_attack"]
    law = payload["law"]
    doc = {
        "schema": FOLLOW_ON_SCHEMA,
        "version": FOLLOW_ON_VERSION,
        "purpose": (
            "Odyssey II/III activation plumbing: a recovered LAW emits a "
            "TRANSFER_TEST WorkUnit (same mechanism, different specimen/"
            "architecture) and a LAW_ATTACK WorkUnit (adversarial falsifier). "
            "Both carry the law's measured scope, its falsifier, and the "
            "result that would retire it. Scope is copied, never widened."
        ),
        "odyssey": "II WHAT DID HAWKING ALREADY LEARN? / III WHERE IS HAWKING WRONG?",
        "authority": (
            "H-ROADMAP §6 the three Odysseys and the work-unit law; "
            "II-A / III-A gene cards; a candidate law emits TRANSFER_TEST "
            "and LAW_ATTACK WorkUnits"
        ),
        "evidence_class": "STATIC",
        "gpu_authority": False,
        "odyssey_i_barrier": None,
        "named_law": law,
        "source_receipt": law["source_path"],
        "evidence_receipt": NAMED_LAW_EVIDENCE,
        "transfer_test": transfer,
        "law_attack": attack,
        "round_trip": {
            "scope_survived_transfer_test": payload["round_trip"]["scope_survived_transfer_test"],
            "scope_survived_law_attack": payload["round_trip"]["scope_survived_law_attack"],
            "transfer_test_lattice": (extract_scope_binding(payload["round_trip"]["transfer_test"]) or {}).get("lattice"),
            "law_attack_lattice": (extract_scope_binding(payload["round_trip"]["law_attack"]) or {}).get("lattice"),
        },
        "mutation_check": payload["mutation_check"],
        "gates_invoked": sorted(
            set(
                list(transfer.get("gates_invoked") or [])
                + list(attack.get("gates_invoked") or [])
                + [
                    "tools.future.odyssey2_law_store.load_repo_json",
                    "tools.future.odyssey2_law_store.validate_law",
                    "tools.future.autonomy_scars.law_from_odyssey2_record",
                    "tools.future.autonomy_scars.round_trip_law",
                ]
            )
        ),
        "recovered_implementation": {
            "odyssey2_law_store": (
                f"{NAMED_LAW_ID} in {NAMED_LAW_STORE}, seeded from {NAMED_LAW_EVIDENCE}; "
                "Law field set and sequential lattice consumed, not forked"
            ),
            "odyssey2_transfer": (
                "may_transfer / require_sealed / sealed B-side / no Phase-I barrier"
            ),
            "odyssey3_adversary": (
                "validate_law / generate_attacks / emit_for_law; nine families, ranking"
            ),
            "odyssey3_attack": "to_o3_law projects the II Law onto the III field set",
            "workunit_species": "emit_hcli_workunit + validate_emitted_unit (HCLI field set)",
            "autonomy_scars": "law_from_odyssey2_record + round_trip_law + Scope axes",
            "scar_reevaluator.consult_candidate": (
                "existing HCLI consult path; this emit path is the follow-on"
            ),
            "not_a_fork": True,
        },
        "gaps_closed": [
            f"Recovered {law['law_id']} from {law['source_path']} (evidence {NAMED_LAW_EVIDENCE}); did not invent a law.",
            f"TRANSFER_TEST {transfer.get('id')} targets {transfer.get('target_alias')} "
            f"family={transfer.get('target_architecture_family')} (source family "
            f"{transfer.get('source_architecture_family')}).",
            f"LAW_ATTACK {attack.get('id')} selected {attack.get('selected_attack_id')}.",
            "Both units carry scope_binding, falsifier, result_that_would_retire.",
            "JSON round-trip preserves the measured lattice; dropping or widening the binding fails.",
            "No Odyssey I completion barrier.",
        ],
        "negative_findings": [
            "STATIC plumbing: neither unit ran a GPU or measured the target.",
            "TRANSFER_TEST does not widen MODEL_LOCAL even if a later measurement holds.",
            "LAW_ATTACK does not apply_result here; a hit still has to go through the III loop.",
            "Flash physical_status in the II school table is metadata_only; the B-side used is the sealed WHOLE_TREE_VERIFIED identity, not a fabricated weight presence.",
        ],
        "claim_boundary_reminder": (
            "STATIC. The law's DIAGNOSTIC_RELATIVE evidence_strength is copied as "
            "identity, not remeasured. Scope on the unit is the origin. The target "
            "specimen is the experiment, not a silent promotion."
        ),
    }
    _assert_no_hardware_claims(doc)
    return write_receipt(FOLLOW_ON_RECEIPT, doc, RECORDED_BY)


def build(scars: list[Any] | None = None) -> Any:
    probe = probe_tolerance()
    pool = scars if scars is not None else ni.ingest(force=True)
    rows = classify_corpus(pool)
    follow = emit_follow_on()
    cov = counts(rows)
    shares = _census_shares()
    ranked = rank_reopenable(rows, census_shares=shares)
    families = _group_families(ranked)
    compact = [
        {
            "scar_id": r["scar_id"],
            "class": r["class"],
            "method": r["method"],
            "hypothesis_family": r["hypothesis_family"],
            "organ": r["organ"],
            "died_at_threshold": (r.get("died_at") or {}).get("summary"),
            "source_path": r["source_path"],
            "tolerance_change_reopens": r["tolerance_change_reopens"],
        }
        for r in rows
    ]
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Mechanical classifier over the negative-science scar corpus: "
            "STRUCTURALLY_REFUTED vs POSSIBLY_REOPENABLE, from each scar's "
            "recorded method and threshold. Not a choice of the next "
            "representation. Nothing was relaunched."
        ),
        "authority": (
            "receipts/future/FUNCTIONAL_ROLE_PROBE.json robustness; "
            "tools/future/negative_index.py ingest"
        ),
        "probe_tolerance": probe,
        "why_fidelity_may_be_stricter_than_capability": (
            "The probe zeroed 40% of a tensor's output rows and moved the "
            "hidden state by the recorded worst_damage cosine. Local organ "
            "cosine / relative-L2 / reconstruction bars judge a different "
            "object than capability. A scar that died only on such a bar is "
            "POSSIBLY_REOPENABLE. A scar whose method is structure-existence, "
            "process identity, or already-failed capability is "
            "STRUCTURALLY_REFUTED: changing a reconstruction tolerance cannot "
            "reopen it."
        ),
        "cited_campaign_fidelity_bars": {
            "organ_cosine": {
                "value": CITED_CAMPAIGN_ORGAN_COSINE_BAR,
                "source": CITED_CAMPAIGN_ORGAN_COSINE_BAR_SOURCE,
                "note": (
                    "Cited as campaign context. Not written onto a scar that "
                    "did not name this bar."
                ),
            }
        },
        "nothing_relaunched": True,
        "nothing_relaunched_statement": (
            "This module classified existing scars and ranked POSSIBLY_REOPENABLE "
            "ones by theoretical EBPW-cut rank, token_ns opportunity rank, and "
            "implementation cost rank. It did not relaunch, reschedule, admit, "
            "or run any candidate. The resident owning SUB2_EBPW decides."
        ),
        "resident_decides": True,
        "counts": cov,
        "top_reopenable_families": families[:25],
        "top_reopenable_scars": ranked[:15],
        "classifications": compact,
        "ranking_rule": (
            "Sort POSSIBLY_REOPENABLE by theoretical EBPW-reduction rank "
            "descending, then token_ns opportunity rank descending, then "
            "implementation cost rank ascending. Missing family/organ is "
            "UNTESTED and sorts last. Ranks are ordinal tables over named "
            "families and organs plus cited byte shares from "
            f"{CENSUS_REL}; they are not hardware measurements and they "
            "are not a recommendation to relaunch."
        ),
        "census_share_source": CENSUS_REL if shares is not None else "UNTESTED",
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "n_scars_ingested": len(pool),
        "scoped_consult": consult_candidate(
            {
                "model": "deepseek-v4-flash",
                "organ": "gate",
                "hypothesis_family": "cross_expert_structure",
            }
        ),
        "follow_on": {
            "receipt": FOLLOW_ON_RECEIPT,
            "law_id": follow["law"]["law_id"],
            "source_receipt": follow["law"]["source_path"],
            "evidence_receipt": NAMED_LAW_EVIDENCE,
            "transfer_test_id": follow["transfer_test"]["id"],
            "law_attack_id": follow["law_attack"]["id"],
            "scope_lattice": follow["scope_binding"]["lattice"],
            "scope_preserved": True,
            "mutation_check_fails_when_binding_dropped": True,
        },
    }
    for key in HARDWARE_FIELDS:
        if key in doc and isinstance(doc[key], (int, float)):
            raise ReevaluatorRefused(f"refused hardware field {key}")
    write_follow_on_receipt(follow)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--emit-follow-on", action="store_true")
    a = ap.parse_args()
    if a.emit_follow_on and not a.build:
        print(write_follow_on_receipt())
        return 0
    if not a.build:
        ap.error("pass --build or --emit-follow-on")
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
