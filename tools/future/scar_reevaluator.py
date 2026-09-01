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

    python3 tools/future/scar_reevaluator.py --build
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

from tools.future._common import HARDWARE_FIELDS, REPO, git, write_receipt
from tools.future import negative_index as ni


RECEIPT = "SCAR_REEVALUATOR.json"
SCHEMA = "hawking.future.scar_reevaluator.v1"
VERSION = 1
RECORDED_BY = "tools/future/scar_reevaluator.py"
EVIDENCE_CLASS = "STATIC_ONLY"
PROBE_REL = "receipts/future/FUNCTIONAL_ROLE_PROBE.json"
CENSUS_REL = "receipts/future/MLP_BYTE_CENSUS.json"

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
    0.0059 figure is not typed here.
    """
    path = REPO / PROBE_REL
    if not path.is_file():
        raise ReevaluatorRefused(
            f"{PROBE_REL} is not on disk; a scar reevaluator with no "
            "robustness receipt would be inventing the capability tolerance"
        )
    doc = json.loads(path.read_text())
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


def build(scars: list[Any] | None = None) -> Any:
    probe = probe_tolerance()
    pool = scars if scars is not None else ni.ingest(force=True)
    rows = classify_corpus(pool)
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
    }
    for key in HARDWARE_FIELDS:
        if key in doc and isinstance(doc[key], (int, float)):
            raise ReevaluatorRefused(f"refused hardware field {key}")
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if not a.build:
        ap.error("pass --build")
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
