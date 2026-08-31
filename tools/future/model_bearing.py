"""MODEL BEARING — put the resident in the loop so an hour of torture is not scripted Python with a model bolted on.

autonomy_run.py is the loop; every trial recorded resident_model_cognition
UNAVAILABLE because nothing asked a model to interpret, choose, explain,
or replan. The live probe later showed the sealed-3.14 resident starts and
generates; that fact was being read as a statement about the machine. This
module is the missing cognition seam the loop would call: the model
proposes, the tools decide.

Refuses: claiming participation when the provider is absent; counting a
decision with no reason; treating a reworded hypothesis as a new one;
falling back to the fixed policy and calling it model-bearing; starting a
resident or taking a GPU lease from --build; copying hardware numbers out
of the dirty live probe.

Cannot establish: that this resident reasons well enough to orchestrate
Odyssey. That is what the hour measures. This module makes the hour
falsifiable.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import REPO, git, write_receipt
from tools.future import autonomy_trial as at
from tools.future import flash_organ_pivot as fop
from tools.future import frontiers as fr
from tools.future import negative_index as ni
from tools.future import status_causality as sc

RECEIPT = "MODEL_BEARING.json"
SCHEMA = "hawking.future.model_bearing.v1"
RECORDED_BY = "tools/future/model_bearing.py"
LIVE_PROBE_REL = "receipts/future/evidence/RESIDENT_LIVE_PROBE.json"

AVAILABLE = "AVAILABLE"
UNAVAILABLE = at.COGNITION_UNAVAILABLE  # "UNAVAILABLE"

REQUIRED_PROVIDER = ("start", "ask", "sessions", "health", "stop", "restart")
PROVIDER_MODULE = "tools.future.resident_provider"

# frontiers.next_work order: this is the scripted policy the model must
# sometimes beat, not a ranking of scientific value.
FIXED_POLICY = "highest expected_information_gain then id (frontiers.next_work)"

PROMPT_ENTRY_CAP = 8
PROMPT_DESC_CHARS = 72
MAX_PROMPT_CHARS = 1800
MIN_REASON_CHARS = 1

# Same-mechanism band. A reworded restatement lands here; a pivot does not.
MECHANISM_JACCARD_SAME = 0.50
MECHANISM_CONTAINMENT_SAME = 0.75

_WORD = re.compile(r"[a-z0-9]+")
_EXTRA_STOP = frozenset(
    {
        "should",
        "try",
        "because",
        "will",
        "would",
        "could",
        "might",
        "please",
        "json",
        "only",
        "return",
        "right",
        "using",
        "use",
        "via",
        "into",
        "from",
        "with",
        "that",
        "this",
        "then",
        "than",
        "also",
        "just",
        "make",
        "made",
        "doing",
        "does",
        "done",
        "next",
        "worth",
        "doing",
    }
)

# Shared fixtures so --build and the tests watch the same detector.
RESTATEMENT_PRIOR: dict[str, str] = {
    "text": (
        "The gate_up latent+readout codec is the right next experiment because "
        "shared input latents plus expert-local readouts will compress routed experts."
    ),
    "mechanism": "shared input latent plus expert-local output readout",
    "surface": "layer_4.routed_experts.gate_up_proj",
    "organ": "layer_4.routed_experts.gate_up_proj",
    "hypothesis_family": "shared_input_latent_plus_expert_local_output_readout",
}
RESTATEMENT_REWORD: dict[str, str] = {
    "text": (
        "We should try a shared input latent with an expert-local output readout "
        "on gate_up; this will compress the routed experts."
    ),
    "mechanism": "shared-input latent with expert local output readout",
    "surface": "layer_4.routed_experts.gate_up_proj",
    "organ": "layer_4.routed_experts.gate_up_proj",
    "hypothesis_family": "shared_input_latent_plus_expert_local_output_readout",
}
RESTATEMENT_PIVOT: dict[str, str] = {
    "text": (
        "Leave gate_up; try ngram product codebooks on the embedding table, "
        "a different generator surface."
    ),
    "mechanism": "ngram product codebook table",
    "surface": "embed",
    "organ": "embed",
    "hypothesis_family": "n_gram_product_codebook_table",
}

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. The model proposes; "
    "the tools decide. A receipt that mixes those two columns is fiction. "
    "Cognition UNAVAILABLE is not a scripted success."
)

# Process-local decision tape. Receipts never replay a faked tape as live.
_LOG: list[dict[str, Any]] = []
# Empty = real import. Non-empty = test seam (value may be None).
_SEAM: list[Any] = []


class BearingRefused(ValueError):
    """A call that would look like success without its input."""

    def __init__(self, reason: str, *, missing: list[str] | None = None) -> None:
        self.reason = reason
        self.missing = list(missing or [])
        super().__init__(reason)


class CognitionUnavailable(BearingRefused):
    """The resident was not there. Never substitute the fixed policy."""


# ---------------------------------------------------------------------------
# Provider seam. Tests bind a fake; --build never reads the seam.
# ---------------------------------------------------------------------------


def bind_provider(provider: Any | None) -> None:
    """Test-only. Receipts call load_provider() and ignore this."""
    _SEAM.clear()
    _SEAM.append(provider)


def unbind_provider() -> None:
    _SEAM.clear()


def load_provider() -> tuple[Any | None, str]:
    """Import the sibling resident_provider. Absence is UNAVAILABLE, not a fake."""
    try:
        from tools.future import resident_provider as rp
    except ImportError as exc:
        return None, f"ImportError: {type(exc).__name__}: {exc}"
    getter = getattr(rp, "get_provider", None)
    obj: Any = rp
    if callable(getter):
        try:
            got = getter()
        except Exception as exc:
            return None, f"get_provider raised {type(exc).__name__}: {exc}"
        if got is not None:
            obj = got
    iface = provider_interface(obj)
    if not iface["ok"]:
        return None, (
            f"{PROVIDER_MODULE} is importable but missing {iface['missing']}; "
            "refusing to treat a partial surface as a resident"
        )
    return obj, f"imported {PROVIDER_MODULE}"


def provider_interface(obj: Any | None) -> dict[str, Any]:
    if obj is None:
        return {
            "ok": False,
            "present": [],
            "missing": list(REQUIRED_PROVIDER),
        }
    present = [n for n in REQUIRED_PROVIDER if callable(getattr(obj, n, None))]
    missing = [n for n in REQUIRED_PROVIDER if n not in present]
    return {"ok": not missing, "present": present, "missing": missing}


def _seam_bound() -> bool:
    return len(_SEAM) > 0


def _active_provider() -> tuple[Any | None, str]:
    if _seam_bound():
        bound = _SEAM[0]
        if bound is None:
            return None, "provider seam bound to None"
        iface = provider_interface(bound)
        if not iface["ok"]:
            return None, f"bound provider missing {iface['missing']}"
        return bound, "provider seam bound (tests only; receipts ignore this)"
    return load_provider()


def cognition_state(*, provider: Any | None = None, allow_seam: bool = True) -> dict[str, Any]:
    """AVAILABLE only when a real (or test-bound) provider is healthy.

    health() is a probe, not a start. start() is never implied.
    """
    if provider is None:
        provider, how = _active_provider() if allow_seam else load_provider()
    else:
        how = "provider argument"
        iface = provider_interface(provider)
        if not iface["ok"]:
            provider, how = None, f"provider argument missing {iface['missing']}"
    if provider is None:
        return {
            "state": UNAVAILABLE,
            "why": how,
            "provider_source": None,
            "health": None,
            "asked": False,
        }
    health = _call_health(provider)
    if not health.get("ok"):
        return {
            "state": UNAVAILABLE,
            "why": (
                "provider present but health is not ok: "
                + str(health.get("why") or health.get("status") or "not ready")
            ),
            "provider_source": how,
            "health": health,
            "asked": False,
        }
    return {
        "state": AVAILABLE,
        "why": how,
        "provider_source": how,
        "health": health,
        "asked": False,
    }


def _call_health(provider: Any) -> dict[str, Any]:
    fn = getattr(provider, "health", None)
    if not callable(fn):
        return {"ok": False, "why": "no health()"}
    try:
        raw = fn()
    except Exception as exc:
        return {"ok": False, "why": f"health raised {type(exc).__name__}: {exc}"}
    if isinstance(raw, Mapping):
        status = str(raw.get("status") or "").lower()
        ok = raw.get("ok")
        if ok is None:
            ok = status in {"ready", "healthy", "ok", "available", "up"}
        return {
            "ok": bool(ok),
            "status": raw.get("status"),
            "why": raw.get("why") or raw.get("reason"),
        }
    if isinstance(raw, bool):
        return {"ok": raw, "status": "ready" if raw else "not_ready"}
    return {"ok": False, "why": f"health returned {type(raw).__name__}"}


def _call_sessions(provider: Any) -> list[str]:
    fn = getattr(provider, "sessions", None)
    if not callable(fn):
        return []
    try:
        raw = fn()
    except Exception:
        return []
    if isinstance(raw, Mapping):
        raw = raw.get("sessions") or raw.get("ids") or []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    out: list[str] = []
    if not isinstance(raw, (list, tuple)):
        return []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, Mapping):
            sid = str(item.get("id") or item.get("session") or item.get("name") or "").strip()
            if sid:
                out.append(sid)
    return out


def _call_start(provider: Any, **kwargs: Any) -> dict[str, Any]:
    fn = getattr(provider, "start", None)
    if not callable(fn):
        return {"ok": False, "why": "no start()"}
    try:
        try:
            raw = fn(**kwargs) if kwargs else fn()
        except TypeError:
            raw = fn()
    except Exception as exc:
        return {"ok": False, "why": f"start raised {type(exc).__name__}: {exc}"}
    if isinstance(raw, Mapping):
        ok = raw.get("ok")
        if ok is None:
            ok = str(raw.get("status") or "").lower() in {"ready", "healthy", "ok", "available"}
        return {"ok": bool(ok), "session": raw.get("session") or kwargs.get("session"), "why": raw.get("why")}
    return {"ok": bool(raw), "session": kwargs.get("session")}


def _call_ask(provider: Any, prompt: str, *, session: str | None = None) -> dict[str, Any]:
    fn = getattr(provider, "ask", None)
    if not callable(fn):
        raise CognitionUnavailable("provider has no ask()")
    try:
        if session is None:
            raw = fn(prompt)
        else:
            try:
                raw = fn(prompt, session=session)
            except TypeError:
                raw = fn(prompt)
    except CognitionUnavailable:
        raise
    except Exception as exc:
        raise CognitionUnavailable(f"ask raised {type(exc).__name__}: {exc}") from exc
    if isinstance(raw, str):
        return {"text": raw, "ok": True, "session": session}
    if isinstance(raw, Mapping):
        text = raw.get("text") or raw.get("generated_text") or raw.get("reply") or raw.get("content") or ""
        ok = raw.get("ok")
        if ok is None:
            ok = True
        return {
            "text": str(text),
            "ok": bool(ok),
            "session": raw.get("session") or session,
        }
    raise CognitionUnavailable(f"ask returned {type(raw).__name__}, not text")


# ---------------------------------------------------------------------------
# Digests, JSON, tokens. Tools measure; the model does not grade itself.
# ---------------------------------------------------------------------------


def _digest(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def _field(obj: Mapping[str, Any] | None, *keys: str) -> str:
    if not isinstance(obj, Mapping):
        return ""
    for key in keys:
        val = obj.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            return text
    return ""


def _cid(obj: Mapping[str, Any] | None) -> str:
    return _field(obj, "id", "unit_id", "choice_id")


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokens(text: str) -> frozenset[str]:
    raw = fr._tokens(text or "")
    extra = set(fr._STOP) | set(_EXTRA_STOP)
    out: set[str] = set()
    for tok in raw:
        if tok in extra:
            continue
        tok = _stem(tok)
        if len(tok) < 3 or tok in extra:
            continue
        out.add(tok)
    if not out:
        # frontiers._tokens already dropped stopwords; keep a last-chance set
        # so two empty strings still compare, and two distinct rare tokens still differ.
        for tok in _WORD.findall((text or "").lower()):
            tok = _stem(tok)
            if len(tok) >= 3 and tok not in extra:
                out.add(tok)
    return frozenset(out)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _surface_key(hyp: Mapping[str, Any]) -> str:
    raw = _field(hyp, "surface", "organ", "tensor")
    if not raw:
        return ""
    if "." in raw:
        return raw.strip().lower()
    organ = ni.canon_organ(raw)
    if organ and organ != ni.UNRECORDED:
        return organ
    return raw.strip().lower()


def _family_key(hyp: Mapping[str, Any]) -> str:
    raw = _field(hyp, "hypothesis_family", "family")
    if not raw:
        return ""
    canon = ni.canon_family(raw)
    if not canon or canon == ni.UNRECORDED:
        return ""
    return canon


def _mech_blob(hyp: Mapping[str, Any]) -> str:
    mech = _field(hyp, "mechanism", "how", "algebra", "program")
    if mech:
        return mech
    return _field(hyp, "text", "hypothesis", "description", "title", "detail")


def hypothesis_from(obj: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(obj, Mapping):
        return {"text": "", "mechanism": "", "surface": "", "hypothesis_family": "", "organ": ""}
    md = obj.get("model_decided") if isinstance(obj.get("model_decided"), Mapping) else {}
    chose = obj.get("chose") if isinstance(obj.get("chose"), Mapping) else {}
    merged: dict[str, Any] = {}
    if isinstance(chose, Mapping):
        merged.update(chose)
    if isinstance(md, Mapping):
        merged.update(md)
    merged.update({k: v for k, v in obj.items() if k not in {"model_decided", "chose", "tools_established"}})
    return {
        "text": _field(merged, "text", "hypothesis", "description", "title", "detail"),
        "mechanism": _field(merged, "mechanism", "how"),
        "surface": _field(merged, "surface", "organ", "tensor") or _surface_key(merged),
        "hypothesis_family": _field(merged, "hypothesis_family", "family"),
        "organ": _field(merged, "organ"),
    }


def meaningfully_different(prior: Mapping[str, Any] | None, nxt: Mapping[str, Any] | None) -> dict[str, Any]:
    """Measured difference. A reworded restatement cannot pass as hypothesis B.

    Different named surface OR different named family OR a mechanism whose
    token set is not a restatement. Same surface plus same mechanism is a
    restatement even when the prose is new.
    """
    if not isinstance(prior, Mapping) or not isinstance(nxt, Mapping):
        return {
            "different": False,
            "refused": True,
            "why": "both hypotheses must be objects; refusing to compare",
            "evidence_class": "STATIC_ONLY",
        }
    a = hypothesis_from(prior)
    b = hypothesis_from(nxt)
    fam_a, fam_b = _family_key(a), _family_key(b)
    surf_a, surf_b = _surface_key(a), _surface_key(b)
    if not surf_a:
        surf_a = a.get("surface") or ""
    if not surf_b:
        surf_b = b.get("surface") or ""
    toks_a = _tokens(_mech_blob(a))
    toks_b = _tokens(_mech_blob(b))
    if not toks_a and not toks_b and not surf_a and not surf_b and not fam_a and not fam_b:
        return {
            "different": False,
            "refused": True,
            "why": "both hypotheses are empty; emptiness is not a new mechanism",
            "jaccard": 1.0,
            "containment": 0.0,
            "evidence_class": "STATIC_ONLY",
        }
    jac = _jaccard(toks_a, toks_b)
    contained = _containment(toks_a, toks_b)
    mech_same = jac >= MECHANISM_JACCARD_SAME or contained >= MECHANISM_CONTAINMENT_SAME
    named_surface_diff = bool(surf_a and surf_b and surf_a != surf_b)
    named_family_diff = bool(fam_a and fam_b and fam_a != fam_b)
    if named_surface_diff or named_family_diff:
        why = []
        if named_surface_diff:
            why.append(f"surface {surf_a!r} -> {surf_b!r}")
        if named_family_diff:
            why.append(f"family {fam_a!r} -> {fam_b!r}")
        return {
            "different": True,
            "refused": False,
            "why": "; ".join(why),
            "jaccard": jac,
            "containment": contained,
            "mechanism_same": mech_same,
            "surface_a": surf_a,
            "surface_b": surf_b,
            "family_a": fam_a,
            "family_b": fam_b,
            "evidence_class": "STATIC_ONLY",
        }
    if mech_same:
        return {
            "different": False,
            "refused": False,
            "why": (
                f"reworded restatement: mechanism jaccard={jac:.3f} "
                f"containment={contained:.3f} surface={surf_a or surf_b or 'unspecified'!r}"
            ),
            "jaccard": jac,
            "containment": contained,
            "mechanism_same": True,
            "surface_a": surf_a,
            "surface_b": surf_b,
            "family_a": fam_a,
            "family_b": fam_b,
            "evidence_class": "STATIC_ONLY",
        }
    return {
        "different": True,
        "refused": False,
        "why": (
            f"mechanism tokens differ (jaccard={jac:.3f} containment={contained:.3f}) "
            f"on surface {surf_a or surf_b or 'unspecified'!r}"
        ),
        "jaccard": jac,
        "containment": contained,
        "mechanism_same": False,
        "surface_a": surf_a,
        "surface_b": surf_b,
        "family_a": fam_a,
        "family_b": fam_b,
        "evidence_class": "STATIC_ONLY",
    }


def _extract_json(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    blob = text.strip()
    if blob.startswith("```"):
        lines = blob.split("\n")
        inner: list[str] = []
        for line in lines[1:]:
            if line.strip().startswith("```"):
                break
            inner.append(line)
        blob = "\n".join(inner).strip()
        if blob.lower().startswith("json"):
            blob = blob[4:].lstrip()
    start = blob.find("{")
    if start < 0:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(blob[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _reason_of(parsed: Mapping[str, Any] | None, *keys: str) -> str:
    if not isinstance(parsed, Mapping):
        return ""
    keys = keys or ("reason", "why", "because")
    text = _field(parsed, *keys)
    return text if len(text.strip()) >= MIN_REASON_CHARS else ""


# ---------------------------------------------------------------------------
# Decision tape.
# ---------------------------------------------------------------------------


def reset_log() -> None:
    _LOG.clear()


def decision_log() -> list[dict[str, Any]]:
    return [dict(row) for row in _LOG]


def _append(kind: str, row: Mapping[str, Any]) -> dict[str, Any]:
    rec = dict(row)
    rec["seq"] = len(_LOG) + 1
    rec["kind"] = kind
    rec.setdefault("evidence_class", "STATIC_ONLY")
    rec.setdefault("gpu_authority", False)
    _LOG.append(rec)
    return rec


def record_outcome(seq: int, what_ran: Mapping[str, Any] | None) -> dict[str, Any]:
    """Tools record what actually executed after a model decision.

    'Changed what ran next' is counterfactual against the fixed policy:
    the model's pick ran, and it was not the policy pick. Agreeing with
    the policy and watching it run is not participation.
    """
    if not isinstance(seq, int) or seq < 1:
        raise BearingRefused("record_outcome requires a decision seq", missing=["seq"])
    found: dict[str, Any] | None = None
    for row in _LOG:
        if row.get("seq") == seq:
            found = row
            break
    if found is None:
        raise BearingRefused(f"no decision seq={seq} on the tape", missing=["seq"])
    ran_id = _cid(what_ran) if isinstance(what_ran, Mapping) else ""
    tools = dict(found.get("tools_established") or {})
    tools["what_ran"] = ran_id or None
    found["tools_established"] = tools
    model_id = ""
    md = found.get("model_decided")
    if isinstance(md, Mapping):
        model_id = _field(md, "choice_id", "id")
    if not model_id and isinstance(found.get("chose"), Mapping):
        model_id = _cid(found.get("chose"))
    policy_id = str(tools.get("fixed_policy_id") or "")
    diverged = bool(found.get("diverged_from_fixed_policy"))
    changed = bool(model_id and ran_id and model_id == ran_id and diverged and ran_id != policy_id)
    found["changed_what_ran_next"] = changed
    found["what_ran"] = ran_id or None
    return dict(found)


def _unavailable_record(kind: str, why: str, tools: Mapping[str, Any]) -> dict[str, Any]:
    rec = {
        "cognition": UNAVAILABLE,
        "why": why,
        "participated": False,
        "counts_as_decision": False,
        "diverged_from_fixed_policy": False,
        "changed_what_ran_next": False,
        "reason": None,
        "chose": None,
        "model_decided": None,
        "tools_established": dict(tools),
        "fall_back_to_scripted": False,
        "refused": (
            "cognition UNAVAILABLE; refusing to fall back to scripted choice "
            "and call it model-bearing"
        ),
    }
    return _append(kind, rec)


# ---------------------------------------------------------------------------
# Fixed policy and scar filter. Tools, not the model.
# ---------------------------------------------------------------------------


def _gain(candidate: Mapping[str, Any]) -> int:
    raw = candidate.get("expected_information_gain")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0
    return int(raw)


def _policy_key(candidate: Mapping[str, Any]) -> tuple[int, str]:
    return (-_gain(candidate), _cid(candidate))


def _scar_dead(candidate: Mapping[str, Any], scar_pool: list[Any] | None) -> dict[str, Any] | None:
    family = _field(candidate, "hypothesis_family", "family")
    if not family:
        return None
    proposal = {
        "model": candidate.get("model"),
        "organ": candidate.get("organ") or candidate.get("surface"),
        "hypothesis_family": family,
        "representation": candidate.get("representation"),
    }
    try:
        return ni.refuse_if_dead(proposal, scar_pool)
    except Exception as exc:
        return {
            "refused": True,
            "reason": f"scar index raised {type(exc).__name__}: {exc}; refusing to guess liveness",
            "index_error": True,
        }


def fixed_policy_choose(
    candidates: Sequence[Mapping[str, Any]],
    *,
    scar_pool: list[Any] | None = None,
) -> dict[str, Any]:
    """The scripted pick: frontiers.next_work order after scar refusal."""
    if not candidates:
        return {
            "id": None,
            "chose": None,
            "refusals": [],
            "policy": FIXED_POLICY,
            "refused": "no candidates",
        }
    ranked = sorted((dict(c) for c in candidates if isinstance(c, Mapping)), key=_policy_key)
    refusals: list[dict[str, Any]] = []
    for cand in ranked:
        cid = _cid(cand)
        if not cid:
            refusals.append({"id": None, "reason": "candidate has no id"})
            continue
        dead = _scar_dead(cand, scar_pool)
        if dead:
            refusals.append({"id": cid, "scar": dead})
            continue
        return {
            "id": cid,
            "chose": cand,
            "refusals": refusals,
            "policy": FIXED_POLICY,
        }
    return {
        "id": None,
        "chose": None,
        "refusals": refusals,
        "policy": FIXED_POLICY,
        "refused": "every candidate is scar-dead or missing an id",
    }


def _frontier_entries(frontier: Any | None) -> tuple[list[dict[str, Any]], str]:
    if frontier is None:
        try:
            units = fr.next_work(fr.THIS_HOST_LANES) or []
        except Exception as exc:
            return [], f"frontiers.next_work raised {type(exc).__name__}: {exc}"
        rows = [dict(u) for u in units if isinstance(u, Mapping)]
        if not rows:
            return [], "frontiers.next_work returned no runnable units"
        return rows, "frontiers.next_work(THIS_HOST_LANES)"
    if isinstance(frontier, Mapping):
        for key in ("entries", "items", "next_work", "units"):
            raw = frontier.get(key)
            if isinstance(raw, list):
                rows = [dict(x) for x in raw if isinstance(x, Mapping)]
                return rows, f"frontier[{key!r}]"
        if _cid(frontier):
            return [dict(frontier)], "frontier mapping"
        return [], "frontier mapping named no entries"
    if isinstance(frontier, (list, tuple)):
        rows = [dict(x) for x in frontier if isinstance(x, Mapping)]
        return rows, "frontier list"
    return [], f"frontier is {type(frontier).__name__}"


def _compact_entries(entries: Sequence[Mapping[str, Any]], *, cap: int = PROMPT_ENTRY_CAP) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in entries[:cap]:
        out.append(
            {
                "id": _cid(item),
                "title": (_field(item, "title", "description", "detail") or "")[:PROMPT_DESC_CHARS],
                "gain": _gain(item),
                "frontier": _field(item, "frontier", "frontier_id"),
                "family": _field(item, "hypothesis_family", "family"),
            }
        )
    return out


# CHOICE_JSON_PROBE: the clip used to eat the tail, and the tail is where every
# ask puts "Return JSON only: {schema}". A control clipped at MAX_PROMPT_CHARS
# parsed 0 of 2 on sealed-3.14 AND 0 of 2 on Qwen3-0.6B; with the schema in view
# both went 2 of 2. That is the ask, not the body and not scale. So: never clip
# the schema off. Prefer dropping whole candidates (_fit_entries) so the JSON
# stays well-formed; this elision is the backstop for asks that still overrun.
SCHEMA_TAIL_RESERVE = 400


def _clip_keeping_tail(prompt: str) -> str:
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt
    tail = prompt[-SCHEMA_TAIL_RESERVE:]
    head = prompt[: MAX_PROMPT_CHARS - SCHEMA_TAIL_RESERVE - 1]
    return head + "…" + tail


def _fit_entries(
    head: str,
    entries: Sequence[Mapping[str, Any]],
    tail: str,
    *,
    cap: int = PROMPT_ENTRY_CAP,
) -> tuple[list[dict[str, Any]], int]:
    """Shrink the candidate set until the ask fits. Never truncate the string.

    A clipped JSON array is malformed AND hides the schema; a shorter array is
    neither. smaller_choice_set was the probe cell that both parsed 2 of 2 and
    named a live id, so dropping candidates is the measured-good direction.
    """
    wanted = _compact_entries(entries, cap=cap)
    n = len(wanted)
    while n > 1:
        body = json.dumps(wanted[:n], sort_keys=True)
        if len(head) + len(body) + len(tail) <= MAX_PROMPT_CHARS:
            break
        n -= 1
    return wanted[:n], len(wanted) - n


def _ask_json(
    provider: Any,
    prompt: str,
    *,
    session: str | None = None,
) -> dict[str, Any]:
    clipped = _clip_keeping_tail(prompt)
    reply = _call_ask(provider, clipped, session=session)
    text = str(reply.get("text") or "")
    parsed = _extract_json(text)
    return {
        "prompt": clipped,
        "prompt_sha256": _digest(clipped),
        "prompt_chars": len(clipped),
        "reply_text": text,
        "reply_sha256": _digest(text),
        "reply_chars": len(text),
        "reply_ok": bool(reply.get("ok")),
        "session": reply.get("session") or session,
        "parsed": parsed,
        "parse_ok": parsed is not None,
    }


def _try_ask(
    kind: str,
    provider: Any,
    prompt: str,
    tools: dict[str, Any],
    *,
    session: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        return _ask_json(provider, prompt, session=session), None
    except CognitionUnavailable as exc:
        tools["ask_error"] = str(exc)
        return None, _unavailable_record(kind, str(exc), tools)


# ---------------------------------------------------------------------------
# Public cognition calls. Model proposes; tools decide.
# ---------------------------------------------------------------------------


def interpret(
    frontier: Any | None = None,
    *,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Ask the resident what the live frontier means. Grounded in real ids."""
    entries, source = _frontier_entries(frontier)
    live_ids = [_cid(e) for e in entries if _cid(e)]
    tools: dict[str, Any] = {
        "n_entries": len(entries),
        "live_ids": live_ids,
        "source": source,
        "prompt_cap": PROMPT_ENTRY_CAP,
    }
    if not live_ids:
        rec = {
            "cognition": cognition_state(provider=provider)["state"],
            "why": source or "frontier has no real entries",
            "participated": False,
            "counts_as_decision": False,
            "grounded": False,
            "reason": None,
            "chose": None,
            "model_decided": None,
            "tools_established": tools,
            "refused": "frontier has no real entries; refusing to invent work",
            "fall_back_to_scripted": False,
        }
        return _append("interpret", rec)
    cog = cognition_state(provider=provider)
    if cog["state"] != AVAILABLE:
        tools["cognition"] = cog
        return _unavailable_record("interpret", cog["why"], tools)
    active, _how = (provider, "argument") if provider is not None else _active_provider()
    _head = "Live frontier entries (cite only these ids):\n"
    _tail = (
        "\nReturn JSON only: "
        '{"reading":"one sentence","worth_doing_next":["id",...],"why":"cite those ids"}'
    )
    compact, n_dropped = _fit_entries(_head, entries, _tail)
    tools["candidates_dropped_to_fit"] = n_dropped
    prompt = _head + json.dumps(compact, sort_keys=True) + _tail
    asked, refused = _try_ask("interpret", active, prompt, tools)
    if refused is not None:
        return refused
    parsed = asked["parsed"] if asked["parse_ok"] else None
    cited_raw = (parsed or {}).get("worth_doing_next") if isinstance(parsed, Mapping) else None
    cited: list[str] = []
    if isinstance(cited_raw, list):
        for item in cited_raw:
            if isinstance(item, str) and item.strip():
                cited.append(item.strip())
            elif isinstance(item, Mapping) and _cid(item):
                cited.append(_cid(item))
    live_set = set(live_ids)
    accepted = [i for i in cited if i in live_set]
    rejected = [i for i in cited if i not in live_set]
    reason = _reason_of(parsed, "why", "reason")
    reading = _field(parsed, "reading") if parsed else ""
    grounded = bool(accepted) and not rejected
    tools.update(
        {
            "accepted_ids": accepted,
            "rejected_ids": rejected,
            "cited_ids": cited,
            "prompt_sha256": asked["prompt_sha256"],
            "reply_sha256": asked["reply_sha256"],
            "parse_ok": asked["parse_ok"],
        }
    )
    model_decided = {
        "reading": reading or None,
        "worth_doing_next": cited,
        "reason": reason or None,
    }
    participated = bool(grounded and reason and asked["parse_ok"])
    rec = {
        "cognition": AVAILABLE,
        "why": cog["why"],
        "participated": participated,
        "counts_as_decision": participated,
        "grounded": grounded,
        "reason": reason or None,
        "chose": {"ids": accepted} if accepted else None,
        "model_decided": model_decided,
        "tools_established": tools,
        "prompt_sha256": asked["prompt_sha256"],
        "reply_sha256": asked["reply_sha256"],
        "fall_back_to_scripted": False,
        "refused": None if participated else (
            "interpretation not grounded in real entries"
            if not grounded
            else "no recorded reason"
            if not reason
            else "reply was not JSON"
        ),
    }
    return _append("interpret", rec)


def choose(
    candidates: Sequence[Mapping[str, Any]] | None,
    *,
    provider: Any | None = None,
    scar_pool: list[Any] | None = None,
) -> dict[str, Any]:
    """Resident picks WITH a stated reason. Tools still admit or refuse."""
    if candidates is None:
        raise BearingRefused("candidates are required", missing=["candidates"])
    rows = [dict(c) for c in candidates if isinstance(c, Mapping) and _cid(c)]
    policy = fixed_policy_choose(rows, scar_pool=scar_pool)
    by_id = {_cid(c): c for c in rows}
    tools: dict[str, Any] = {
        "n_candidates": len(rows),
        "ids": [_cid(c) for c in rows],
        "fixed_policy_id": policy.get("id"),
        "fixed_policy": FIXED_POLICY,
        "policy_refusals": policy.get("refusals") or [],
    }
    if not rows:
        rec = {
            "cognition": cognition_state(provider=provider)["state"],
            "why": "no candidates with ids",
            "participated": False,
            "counts_as_decision": False,
            "diverged_from_fixed_policy": False,
            "changed_what_ran_next": False,
            "reason": None,
            "chose": None,
            "model_decided": None,
            "tools_established": tools,
            "refused": "no candidates with ids; refusing to invent a pick",
            "fall_back_to_scripted": False,
        }
        return _append("choose", rec)
    cog = cognition_state(provider=provider)
    if cog["state"] != AVAILABLE:
        tools["cognition"] = cog
        return _unavailable_record("choose", cog["why"], tools)
    active, _how = (provider, "argument") if provider is not None else _active_provider()
    _head = (
        "Pick one candidate. The scripted policy would pick "
        + json.dumps(policy.get("id"))
        + ".\nCandidates:\n"
    )
    _tail = (
        "\nReturn JSON only: "
        '{"choice_id":"id","reason":"why this, citing a real difference","mechanism":"...",'
        '"surface":"...","hypothesis_family":"..."}'
    )
    # Never advertise what the tools will refuse. fixed_policy_choose already
    # dropped every scar-dead candidate for its OWN pick, but the model was shown
    # the raw set - which is how WU.DEAD.mlp_function_replacement stayed on the
    # menu 45 turns running while the resident kept picking it and the tools kept
    # refusing it. Same filter, same turn, one source of liveness.
    _dead_ids = {str(r.get("id")) for r in (policy.get("refusals") or []) if r.get("id")}
    _live = [r for r in rows if _cid(r) not in _dead_ids]
    tools["candidates_scar_dead"] = sorted(_dead_ids)
    compact, n_dropped = _fit_entries(_head, _live, _tail, cap=max(PROMPT_ENTRY_CAP, len(_live)))
    tools["candidates_dropped_to_fit"] = n_dropped
    prompt = _head + json.dumps(compact, sort_keys=True) + _tail
    asked, refused = _try_ask("choose", active, prompt, tools)
    if refused is not None:
        return refused
    parsed = asked["parsed"] if asked["parse_ok"] else None
    choice_id = _field(parsed, "choice_id", "id") if parsed else ""
    choice_id, transcription_repaired = resolve_choice_id(choice_id, by_id)
    reason = _reason_of(parsed, "reason", "why")
    tools.update(
        {
            "prompt_sha256": asked["prompt_sha256"],
            "reply_sha256": asked["reply_sha256"],
            "parse_ok": asked["parse_ok"],
            "transcription_repaired": transcription_repaired,
        }
    )
    model_decided = {
        "choice_id": choice_id or None,
        "reason": reason or None,
        "mechanism": _field(parsed, "mechanism") if parsed else None,
        "surface": _field(parsed, "surface", "organ") if parsed else None,
        "hypothesis_family": _field(parsed, "hypothesis_family", "family") if parsed else None,
    }
    if not asked["parse_ok"] or not choice_id:
        rec = {
            "cognition": AVAILABLE,
            "why": "reply was not a usable choice",
            "participated": False,
            "counts_as_decision": False,
            "diverged_from_fixed_policy": False,
            "changed_what_ran_next": False,
            "reason": reason or None,
            "chose": None,
            "model_decided": model_decided,
            "tools_established": tools,
            "prompt_sha256": asked["prompt_sha256"],
            "reply_sha256": asked["reply_sha256"],
            "fall_back_to_scripted": False,
            "refused": "model reply did not name a choice_id in JSON",
        }
        return _append("choose", rec)
    if choice_id not in by_id:
        rec = {
            "cognition": AVAILABLE,
            "why": "choice is not in the candidate set",
            "participated": False,
            "counts_as_decision": False,
            "diverged_from_fixed_policy": False,
            "changed_what_ran_next": False,
            "reason": reason or None,
            "chose": None,
            "model_decided": model_decided,
            "tools_established": tools,
            "prompt_sha256": asked["prompt_sha256"],
            "reply_sha256": asked["reply_sha256"],
            "fall_back_to_scripted": False,
            "refused": f"choice_id {choice_id!r} is not in the candidate set; tools refuse invented ids",
        }
        return _append("choose", rec)
    picked = dict(by_id[choice_id])
    if model_decided.get("mechanism"):
        picked.setdefault("mechanism", model_decided["mechanism"])
    if model_decided.get("surface"):
        picked.setdefault("surface", model_decided["surface"])
    if model_decided.get("hypothesis_family"):
        picked.setdefault("hypothesis_family", model_decided["hypothesis_family"])
    dead = _scar_dead(picked, scar_pool)
    if dead:
        tools["scar_refusal"] = dead
        rec = {
            "cognition": AVAILABLE,
            "why": "tools refused a scar-dead pick",
            "participated": False,
            "counts_as_decision": bool(reason),
            "diverged_from_fixed_policy": choice_id != policy.get("id"),
            "changed_what_ran_next": False,
            "reason": reason or None,
            "chose": None,
            "model_decided": model_decided,
            "tools_established": tools,
            "prompt_sha256": asked["prompt_sha256"],
            "reply_sha256": asked["reply_sha256"],
            "fall_back_to_scripted": False,
            "refused": "tools refused the pick: recorded negative science already killed it",
        }
        return _append("choose", rec)
    if not reason:
        rec = {
            "cognition": AVAILABLE,
            "why": "decision has no recorded reason",
            "participated": False,
            "counts_as_decision": False,
            "diverged_from_fixed_policy": choice_id != policy.get("id"),
            "changed_what_ran_next": False,
            "reason": None,
            "chose": None,
            "model_decided": model_decided,
            "tools_established": tools,
            "prompt_sha256": asked["prompt_sha256"],
            "reply_sha256": asked["reply_sha256"],
            "fall_back_to_scripted": False,
            "refused": "a decision with no recorded reason does not count as participation",
        }
        return _append("choose", rec)
    diverged = choice_id != policy.get("id")
    rec = {
        "cognition": AVAILABLE,
        "why": cog["why"],
        "participated": True,
        "counts_as_decision": True,
        "diverged_from_fixed_policy": diverged,
        "changed_what_ran_next": False,
        "reason": reason,
        "chose": picked,
        "model_decided": model_decided,
        "tools_established": tools,
        "prompt_sha256": asked["prompt_sha256"],
        "reply_sha256": asked["reply_sha256"],
        "fall_back_to_scripted": False,
        "refused": None,
    }
    return _append("choose", rec)


def explain_failure(
    result: Mapping[str, Any] | None,
    *,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Resident says why a unit failed. Causal labels stay hypotheses until challenged."""
    if not isinstance(result, Mapping):
        raise BearingRefused("failure result is required", missing=["result"])
    status = _field(result, "status", "status_label")
    challenge = None
    if status:
        try:
            challenge = sc.challenge(result if "probe_kind" in result or "probe" in result else status)
        except Exception as exc:
            challenge = {
                "verdict": sc.UNTESTED,
                "why": f"status_causality.challenge raised {type(exc).__name__}: {exc}",
            }
    elif result.get("probe_kind") or result.get("probe"):
        try:
            challenge = sc.challenge(result)
        except Exception as exc:
            challenge = {"verdict": sc.UNTESTED, "why": f"{type(exc).__name__}: {exc}"}
    tools: dict[str, Any] = {
        "exit_code": result.get("exit_code"),
        "error": result.get("error"),
        "unit_id": _cid(result) or result.get("unit_id"),
        "status": status or None,
        "status_challenge": (
            {
                "verdict": challenge.get("verdict"),
                "probe_kind": challenge.get("probe_kind"),
                "claim_kind": challenge.get("claim_kind"),
                "cause_is_hypothesis": challenge.get("verdict") != sc.SUPPORTED,
            }
            if isinstance(challenge, Mapping)
            else None
        ),
        "facts_are_tools": True,
    }
    if tools["status_challenge"] and tools["status_challenge"]["cause_is_hypothesis"]:
        tools["cause_is_hypothesis"] = True
    cog = cognition_state(provider=provider)
    if cog["state"] != AVAILABLE:
        tools["cognition"] = cog
        return _unavailable_record("explain_failure", cog["why"], tools)
    active, _how = (provider, "argument") if provider is not None else _active_provider()
    fact_blob = {
        "exit_code": result.get("exit_code"),
        "error": str(result.get("error") or "")[:240],
        "status": status or None,
        "challenge_verdict": (challenge or {}).get("verdict") if isinstance(challenge, Mapping) else None,
    }
    prompt = (
        "A unit failed. Tools established these facts (do not contradict them; "
        "do not assert a cause the probe did not establish):\n"
        + json.dumps(fact_blob, sort_keys=True)
        + '\nReturn JSON only: {"why":"one sentence","mechanism":"...","status_claim":null}'
    )
    asked, refused = _try_ask("explain_failure", active, prompt, tools)
    if refused is not None:
        return refused
    parsed = asked["parsed"] if asked["parse_ok"] else None
    reason = _reason_of(parsed, "why", "reason")
    model_decided = {
        "why": reason or None,
        "mechanism": _field(parsed, "mechanism") if parsed else None,
        "status_claim": _field(parsed, "status_claim") if parsed else None,
    }
    if model_decided["status_claim"] and isinstance(challenge, Mapping) and challenge.get("verdict") != sc.SUPPORTED:
        tools["model_cause_not_entailed"] = True
    participated = bool(reason and asked["parse_ok"])
    rec = {
        "cognition": AVAILABLE,
        "why": cog["why"],
        "participated": participated,
        "counts_as_decision": participated,
        "reason": reason or None,
        "chose": None,
        "model_decided": model_decided,
        "tools_established": {
            **tools,
            "prompt_sha256": asked["prompt_sha256"],
            "reply_sha256": asked["reply_sha256"],
            "parse_ok": asked["parse_ok"],
        },
        "prompt_sha256": asked["prompt_sha256"],
        "reply_sha256": asked["reply_sha256"],
        "fall_back_to_scripted": False,
        "refused": None if participated else "failure explanation had no recorded reason",
    }
    return _append("explain_failure", rec)


def next_hypothesis(
    prior: Mapping[str, Any] | None,
    *,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Propose a materially different next attempt. Rewording fails the check."""
    if not isinstance(prior, Mapping):
        raise BearingRefused("prior hypothesis is required", missing=["prior"])
    prior_h = hypothesis_from(prior)
    tools: dict[str, Any] = {
        "prior": prior_h,
        "difference_check": "meaningfully_different; reworded restatement cannot pass",
    }
    scar = prior.get("scar") if isinstance(prior.get("scar"), Mapping) else None
    killed = prior.get("killed") if isinstance(prior.get("killed"), list) else None
    cog = cognition_state(provider=provider)
    if cog["state"] != AVAILABLE:
        tools["cognition"] = cog
        return _unavailable_record("next_hypothesis", cog["why"], tools)
    active, _how = (provider, "argument") if provider is not None else _active_provider()
    prompt = (
        "Hypothesis A failed. Propose hypothesis B that differs in MECHANISM or "
        "SURFACE, not a rewording.\nA="
        + json.dumps(prior_h, sort_keys=True)
        + '\nReturn JSON only: {"text":"...","mechanism":"...","surface":"...",'
        '"hypothesis_family":"...","why_different":"..."}'
    )
    asked, refused = _try_ask("next_hypothesis", active, prompt, tools)
    if refused is not None:
        return refused
    parsed = asked["parsed"] if asked["parse_ok"] else None
    proposed = hypothesis_from(parsed) if parsed else hypothesis_from({})
    if parsed and not proposed["text"]:
        proposed["text"] = _field(parsed, "text")
    diff = meaningfully_different(prior_h, proposed)
    pivot = None
    if scar is not None and killed is not None:
        try:
            pivot = fop.restatement_verdict(proposed, scar, killed)
        except Exception as exc:
            pivot = {"status": "UNTESTED", "why": f"{type(exc).__name__}: {exc}"}
    if isinstance(pivot, Mapping) and pivot.get("status") == "REFUSED_RESTATEMENT":
        diff = dict(diff)
        diff["different"] = False
        diff["why"] = str(pivot.get("reason") or "flash_organ_pivot restatement")
        diff["organ_pivot"] = pivot
    tools.update(
        {
            "meaningfully_different": diff,
            "organ_pivot_restatement": pivot,
            "prompt_sha256": asked["prompt_sha256"],
            "reply_sha256": asked["reply_sha256"],
            "parse_ok": asked["parse_ok"],
        }
    )
    reason = _reason_of(parsed, "why_different", "reason", "why")
    different = bool(diff.get("different"))
    participated = bool(different and reason and asked["parse_ok"])
    rec = {
        "cognition": AVAILABLE,
        "why": cog["why"],
        "participated": participated,
        "counts_as_decision": bool(reason and asked["parse_ok"]),
        "meaningfully_different": different,
        "reason": reason or None,
        "chose": proposed if different else None,
        "model_decided": proposed,
        "tools_established": tools,
        "prompt_sha256": asked["prompt_sha256"],
        "reply_sha256": asked["reply_sha256"],
        "fall_back_to_scripted": False,
        "refused": None if participated else (
            diff.get("why") if not different else "no recorded reason for the new hypothesis"
        ),
    }
    return _append("next_hypothesis", rec)


def delegate(
    objective: str | None,
    *,
    provider: Any | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Bounded subagent work in its own session. Does not wait on tools.

    poll/wait belong to no_wait_scheduler (supervisor). Holding the
    inference slot on a tool wait is the defect that module exists to refuse.
    """
    if not isinstance(objective, str) or not objective.strip():
        raise BearingRefused("delegate requires an objective", missing=["objective"])
    tools: dict[str, Any] = {
        "objective": objective.strip(),
        "does_not_wait_on_tools": True,
        "wait_owner": (
            "tools.future.no_wait_scheduler.poll is a supervisor operation; "
            "the reasoning context is released before it runs"
        ),
    }
    cog = cognition_state(provider=provider)
    if cog["state"] != AVAILABLE:
        tools["cognition"] = cog
        return _unavailable_record("delegate", cog["why"], tools)
    active, _how = (provider, "argument") if provider is not None else _active_provider()
    before = _call_sessions(active)
    name = session or f"subagent.{_digest(objective.strip())[:12]}"
    started = _call_start(active, session=name)
    tools["start"] = {"ok": started.get("ok"), "why": started.get("why")}
    prompt = (
        f"Subagent session {name}. Bounded objective (do not wait on tools):\n"
        + objective.strip()[:400]
        + '\nReturn JSON only: {"plan":"one next action","stop_condition":"..."}'
    )
    asked, refused = _try_ask("delegate", active, prompt, tools, session=name)
    if refused is not None:
        return refused
    after = _call_sessions(active)
    # A subagent is a NEW session id. Reusing the only existing session is not delegation.
    distinct = bool(name in after and name not in before)
    tools.update(
        {
            "session": name,
            "sessions_before": before,
            "sessions_after": after,
            "session_distinct": distinct,
            "prompt_sha256": asked["prompt_sha256"],
            "reply_sha256": asked["reply_sha256"],
            "parse_ok": asked["parse_ok"],
        }
    )
    parsed = asked["parsed"] if asked["parse_ok"] else None
    plan = _field(parsed, "plan", "text") if parsed else ""
    if not distinct:
        rec = {
            "cognition": AVAILABLE,
            "why": "delegate did not produce a distinct session",
            "participated": False,
            "counts_as_decision": False,
            "reason": plan or None,
            "chose": None,
            "model_decided": {"plan": plan or None, "session": name},
            "tools_established": tools,
            "prompt_sha256": asked["prompt_sha256"],
            "reply_sha256": asked["reply_sha256"],
            "fall_back_to_scripted": False,
            "refused": "delegate did not produce a distinct session; refusing to claim subagent work",
        }
        return _append("delegate", rec)
    participated = bool(plan and asked["parse_ok"] and distinct)
    rec = {
        "cognition": AVAILABLE,
        "why": cog["why"],
        "participated": participated,
        "counts_as_decision": participated,
        "reason": plan or None,
        "chose": {"session": name, "plan": plan},
        "model_decided": {
            "plan": plan or None,
            "session": name,
            "stop_condition": _field(parsed, "stop_condition") if parsed else None,
        },
        "tools_established": tools,
        "prompt_sha256": asked["prompt_sha256"],
        "reply_sha256": asked["reply_sha256"],
        "fall_back_to_scripted": False,
        "refused": None if participated else "subagent produced no plan",
    }
    return _append("delegate", rec)


def enter_loop(
    frontier: Any | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    *,
    provider: Any | None = None,
    scar_pool: list[Any] | None = None,
) -> dict[str, Any]:
    """One cognition turn. autonomy_run would call this instead of FIFO.

    Does not invoke capabilities. The loop still invokes. UNAVAILABLE does
    not fall back to the scripted pick.
    """
    reading = interpret(frontier, provider=provider)
    if reading.get("cognition") != AVAILABLE or not reading.get("participated"):
        return {
            "cognition": reading.get("cognition") or UNAVAILABLE,
            "interpret": reading,
            "choose": None,
            "chose": None,
            "fall_back_to_scripted": False,
            "refused": reading.get("refused") or "interpret did not participate",
        }
    pool = list(candidates) if candidates is not None else []
    if not pool:
        accepted = ((reading.get("tools_established") or {}).get("accepted_ids")) or []
        entries, _src = _frontier_entries(frontier)
        by_id = {_cid(e): e for e in entries}
        pool = [by_id[i] for i in accepted if i in by_id]
    picked = choose(pool, provider=provider, scar_pool=scar_pool)
    return {
        "cognition": picked.get("cognition"),
        "interpret": reading,
        "choose": picked,
        "chose": picked.get("chose"),
        "fall_back_to_scripted": False,
        "refused": picked.get("refused"),
    }


def run_trajectory(
    objective: str,
    candidates_a: Sequence[Mapping[str, Any]],
    fail_result: Mapping[str, Any],
    candidates_b: Sequence[Mapping[str, Any]] | None = None,
    *,
    provider: Any | None = None,
    scar_pool: list[Any] | None = None,
    enact: bool = True,
) -> dict[str, Any]:
    """objective → hyp A → experiment → failure → explained → hyp B → second experiment.

    enact=True records that the model's pick ran (the loop honoured it).
    enact=False is the ignored-model control: policy ran instead.
    """
    if not objective.strip():
        raise BearingRefused("trajectory requires an objective", missing=["objective"])
    interp = interpret(
        {"entries": list(candidates_a)},
        provider=provider,
    )
    choice_a = choose(candidates_a, provider=provider, scar_pool=scar_pool)
    policy_id = (choice_a.get("tools_established") or {}).get("fixed_policy_id")
    model_id = (choice_a.get("model_decided") or {}).get("choice_id")
    if enact and choice_a.get("chose"):
        record_outcome(choice_a["seq"], {"id": _cid(choice_a["chose"])})
    elif not enact:
        record_outcome(choice_a["seq"], {"id": policy_id or ""})
    explained = explain_failure(fail_result, provider=provider)
    prior = choice_a.get("chose") or choice_a.get("model_decided") or RESTATEMENT_PRIOR
    hyp_b = next_hypothesis(prior, provider=provider)
    choice_b = None
    if hyp_b.get("meaningfully_different") and candidates_b is not None:
        choice_b = choose(candidates_b, provider=provider, scar_pool=scar_pool)
        if enact and choice_b.get("chose"):
            record_outcome(choice_b["seq"], {"id": _cid(choice_b["chose"])})
        elif not enact and choice_b is not None:
            rec_id = (choice_b.get("tools_established") or {}).get("fixed_policy_id") or ""
            record_outcome(choice_b["seq"], {"id": rec_id})
    participation = materially_participated()
    return {
        "objective": objective,
        "interpret": interp,
        "hypothesis_a": hypothesis_from(prior),
        "choose_a": choice_a,
        "failure": fail_result,
        "explained_failure": explained,
        "hypothesis_b": hyp_b,
        "choose_b": choice_b,
        "enacted": enact,
        "policy_id_a": policy_id,
        "model_id_a": model_id,
        "materially_participated": participation,
        "model_decided": {
            "choose_a": choice_a.get("model_decided"),
            "explain": explained.get("model_decided"),
            "hypothesis_b": hyp_b.get("model_decided"),
            "choose_b": (choice_b or {}).get("model_decided"),
        },
        "tools_established": {
            "choose_a": choice_a.get("tools_established"),
            "explain": explained.get("tools_established"),
            "hypothesis_b": hyp_b.get("tools_established"),
            "choose_b": (choice_b or {}).get("tools_established"),
            "difference": (hyp_b.get("tools_established") or {}).get("meaningfully_different"),
        },
    }


def _normalize_id(text: str) -> str:
    """Fold the differences a transcription introduces, and nothing else."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def resolve_choice_id(raw: str, by_id: Mapping[str, Any]) -> tuple[str, str | None]:
    """Exact id, or a UNIQUE transcription match. Never a guess.

    The 30m run's steady state: the model was shown two candidates, picked the
    right one for the right reason - gain 2 against gain 1, cited correctly - and
    wrote "WU.PROBE.decode_arith cost". One space where the id has an underscore.
    The tools refused it as an invented id, nothing launched, the frontier never
    advanced, and that same question was asked 94 more times.

    Refusing invented ids is correct and stays. But requiring a model to
    transcribe a long opaque identifier character-perfect, and then calling the
    typo a judgment failure, measures the ask again rather than the chooser.

    The match must be UNIQUE after folding case and non-alphanumerics. Two
    candidates that collide under that fold are ambiguous and are refused: better
    no launch than the wrong one. Every repair is RECORDED so the transcription
    rate stays visible instead of being absorbed.
    """
    raw = str(raw or "").strip()
    if not raw:
        return "", None
    if raw in by_id:
        return raw, None
    want = _normalize_id(raw)
    if not want:
        return raw, None
    hits = [cid for cid in by_id if _normalize_id(cid) == want]
    if len(hits) != 1:
        return raw, None
    return hits[0], raw


_INDEX_SUFFIX = re.compile(r"[._-]\d+$")


def _ask_kind(row: Mapping[str, Any]) -> str | None:
    """The SEMANTIC identity of a choose ask: its candidate set, de-indexed.

    WU.HAWKING.health_probe.007 and .008 are the same question. Counting them as
    two is how a frontier that generates its own filler makes a resident look
    like it answered fifty different things.
    """
    tools = row.get("tools_established")
    if not isinstance(tools, Mapping):
        return None
    ids = tools.get("ids")
    if not isinstance(ids, list) or not ids:
        return None
    return "|".join(sorted({_INDEX_SUFFIX.sub("", str(i)) for i in ids}))


def materially_participated(log: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Falsifiable. Emitting text while Python chooses is not participation."""
    rows = list(log) if log is not None else decision_log()
    cognition_states = [str(r.get("cognition") or "") for r in rows if r.get("cognition")]
    available = any(s == AVAILABLE for s in cognition_states)
    unavailable_only = bool(cognition_states) and all(s == UNAVAILABLE for s in cognition_states)
    diverged = [
        r for r in rows
        if r.get("kind") == "choose" and r.get("diverged_from_fixed_policy") and r.get("chose")
    ]
    # DIVERGENCE HAS A DENOMINATOR, AND IT IS NOT THE NUMBER OF ASKS.
    # The first honest 30m run asked choose() 56 times and got 56 identical
    # verdicts, which reads as "the model agreed 56 times" - except only FIVE of
    # those asks were distinct, one prompt firing 52 times. At temperature 0 a
    # deterministic body answering a byte-identical question identically is
    # arithmetic, not a decision, and 5 unique prompts produced 5 unique replies.
    # Scoring "never diverged" over 56 opportunities overstates the evidence 11x.
    # The distinct-ask count is the real denominator; the repeat count is a fact
    # about the frontier, not about the model's judgment.
    choose_rows = [r for r in rows if r.get("kind") == "choose"]
    ask_digests = [r.get("prompt_sha256") for r in choose_rows if r.get("prompt_sha256")]
    distinct_asks = len(set(ask_digests))
    repeated_asks = len(ask_digests) - distinct_asks
    # DISTINCT PROMPTS IS A BETTER DENOMINATOR THAN TOTAL ASKS AND STILL NOT THE
    # RIGHT ONE. The 51-launch run asked 51 byte-distinct questions and 44 of the
    # units it launched were WU.HAWKING.health_probe.NNN - the frontier refilling
    # itself with generated filler. Fifty-one distinct PROMPTS, about seven
    # distinct QUESTIONS. A model agreeing with "highest gain then id" on 44
    # near-identical probes is evidence about the frontier, not about judgment.
    # Strip the trailing index and the question kinds fall out.
    kinds = {
        _ask_kind(r) for r in choose_rows if r.get("tools_established")
    } - {None}
    distinct_kinds = len(kinds)
    different_hyps = [
        r for r in rows
        if r.get("kind") == "next_hypothesis" and r.get("meaningfully_different")
    ]
    changed = [r for r in rows if r.get("changed_what_ran_next")]
    reasoned = [r for r in rows if r.get("counts_as_decision") and r.get("reason")]
    no_reason = [
        r for r in rows
        if r.get("kind") in {"choose", "interpret", "explain_failure", "next_hypothesis"}
        and r.get("cognition") == AVAILABLE
        and not r.get("reason")
        and r.get("model_decided") is not None
    ]
    if not rows:
        return {
            "participated": False,
            "why": "no decisions recorded; refusing to infer participation",
            "divergence_count": 0,
            "different_hypothesis_count": 0,
            "changed_what_ran_next_count": 0,
            "publishable_finding": None,
            "cognition": UNAVAILABLE,
        }
    if unavailable_only or not available:
        return {
            "participated": False,
            "why": "cognition UNAVAILABLE; refusing to claim participation",
            "divergence_count": 0,
            "different_hypothesis_count": 0,
            "changed_what_ran_next_count": 0,
            "publishable_finding": None,
            "cognition": UNAVAILABLE,
            "n_decisions": len(rows),
        }
    scope = {
        "n_choose_asks": len(choose_rows),
        "n_distinct_choose_asks": distinct_asks,
        "n_repeated_choose_asks": repeated_asks,
        "n_distinct_question_kinds": distinct_kinds,
        "divergence_denominator": "distinct QUESTION KINDS, not distinct prompts and not total asks",
        "why": (
            "a deterministic body re-asked a byte-identical question answers it "
            "identically by construction, so total asks is the wrong denominator. "
            "So is distinct PROMPTS: a frontier that refills itself with indexed "
            "probes produces fifty distinct prompts and seven distinct questions. "
            "The scope of any divergence claim is the question-kind count"
        ),
    }
    finding = None
    if not diverged:
        finding = (
            "model choices never diverged from the fixed policy across "
            f"{distinct_kinds} DISTINCT QUESTION KINDS "
            f"({distinct_asks} distinct prompts, {len(choose_rows)} asks, "
            f"{repeated_asks} byte-identical repeats). That is a finding about "
            "this resident on this frontier, not a decorated timeline - and the "
            "scope is the KIND count, because a deterministic body cannot disagree "
            "with itself on the same question and an indexed probe series is one "
            "question asked many times"
        )
    participated = bool(diverged and different_hyps and changed and reasoned)
    why = "model proposed, tools admitted, pick diverged, hyp B differed, and the pick ran"
    if not participated:
        missing = []
        if not diverged:
            missing.append("no choose() diverged from the fixed policy")
        if not different_hyps:
            missing.append("no next hypothesis was meaningfully different")
        if not changed:
            missing.append("no model decision changed what ran next")
        if not reasoned:
            missing.append("no decision carried a recorded reason")
        why = "; ".join(missing)
    return {
        "participated": participated,
        "why": why,
        "divergence_count": len(diverged),
        "divergence_scope": scope,
        "different_hypothesis_count": len(different_hyps),
        "changed_what_ran_next_count": len(changed),
        "reasoned_decision_count": len(reasoned),
        "no_reason_count": len(no_reason),
        "publishable_finding": finding,
        "cognition": AVAILABLE,
        "n_decisions": len(rows),
        "model_proposes_tools_decide": True,
    }


# ---------------------------------------------------------------------------
# Live probe citation. Naming it is not executing it.
# ---------------------------------------------------------------------------


def live_probe_record() -> dict[str, Any]:
    path = REPO / LIVE_PROBE_REL
    raw = ""
    taken = ""
    if path.is_file():
        try:
            raw = path.read_text()
            taken = "disk"
        except OSError as exc:
            raw = ""
            taken = f"disk-unreadable:{type(exc).__name__}"
    if not raw:
        raw = git("show", f"HEAD:{LIVE_PROBE_REL}")
        taken = "git" if raw else (taken or "absent")
    if not raw:
        return {
            "present": False,
            "path": LIVE_PROBE_REL,
            "path_taken": taken,
            "why": (
                "RESIDENT_LIVE_PROBE.json is not on disk and git show returned empty; "
                "a sparse checkout is a path taken, not project-absent"
            ),
            "cited_not_executed": True,
        }
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "present": False,
            "path": LIVE_PROBE_REL,
            "path_taken": taken,
            "why": f"JSONDecodeError: {exc}",
            "cited_not_executed": True,
        }
    if not isinstance(doc, dict):
        return {
            "present": False,
            "path": LIVE_PROBE_REL,
            "path_taken": taken,
            "why": "probe is not an object",
            "cited_not_executed": True,
        }
    return {
        "present": True,
        "path": LIVE_PROBE_REL,
        "path_taken": taken,
        "schema": doc.get("schema"),
        "verdict": doc.get("verdict"),
        "evidence_class": doc.get("evidence_class"),
        "gpu_authority": doc.get("gpu_authority"),
        "lease_taken": doc.get("lease_taken"),
        "what_this_does_NOT_establish": list(doc.get("what_this_does_NOT_establish") or []),
        "cited_not_executed": True,
        "why": (
            "citing the probe receipt is not re-running the resident; "
            "DECLARED capability is not EXECUTED capability"
        ),
    }


def _frontier_catalog() -> dict[str, Any]:
    try:
        units = fr.next_work(fr.THIS_HOST_LANES) or []
    except Exception as exc:
        return {
            "ok": False,
            "why": f"frontiers.next_work raised {type(exc).__name__}: {exc}",
            "ids": [],
        }
    rows = [
        {
            "id": _cid(u),
            "gain": u.get("expected_information_gain"),
            "frontier": u.get("frontier"),
        }
        for u in units
        if isinstance(u, Mapping) and _cid(u)
    ]
    policy = fixed_policy_choose(units, scar_pool=[])
    return {
        "ok": True,
        "n": len(rows),
        "ids": [r["id"] for r in rows[:12]],
        "fixed_policy_id": policy.get("id"),
        "policy": FIXED_POLICY,
        "source": "frontiers.next_work(THIS_HOST_LANES)",
    }


def build() -> Path:
    # Receipts ignore the test seam on purpose: a bound fake must never
    # become evidence that the resident participated.
    imported, how = load_provider()
    iface = provider_interface(imported)
    probe = live_probe_record()
    restatement = meaningfully_different(RESTATEMENT_PRIOR, RESTATEMENT_REWORD)
    pivot = meaningfully_different(RESTATEMENT_PRIOR, RESTATEMENT_PIVOT)
    same = meaningfully_different(RESTATEMENT_PRIOR, RESTATEMENT_PRIOR)
    empty = meaningfully_different({}, {})
    if restatement.get("different"):
        raise BearingRefused(
            "restatement fixture passed the difference check; the detector is fiction"
        )
    if not pivot.get("different"):
        raise BearingRefused(
            "different-surface fixture failed the difference check; the detector is fiction"
        )
    if same.get("different"):
        raise BearingRefused("a hypothesis compared to itself was called different")
    catalog = _frontier_catalog()
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Cognition seam for the autonomy loop: interpret, choose, explain, "
            "next_hypothesis, delegate, decision_log, materially_participated. "
            "The model proposes; the tools decide."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "cognition": {
            "state": UNAVAILABLE,
            "why": (
                "--build does not start or ask the resident; start() is a "
                "GPU-adjacent process this sidecar will not launch without a lease. "
                "Interface presence is recorded separately from execution."
            ),
            "provider_import": how,
            "provider_source": PROVIDER_MODULE if imported is not None else None,
            "faked": False,
            "asked": False,
            "interface": iface,
        },
        "declared_vs_executed": {
            "resident_provider_named": True,
            "resident_provider_imported": imported is not None,
            "resident_asked_in_this_receipt": False,
            "live_probe_cited": bool(probe.get("present")),
            "live_probe_re_run": False,
            "rule": "naming a tool is not evidence it ran",
        },
        "live_probe_cited_not_executed": probe,
        "separation": {
            "model_decided": None,
            "tools_established": {
                "restatement_check": restatement,
                "pivot_check": pivot,
                "self_check": same,
                "empty_check": empty,
                "frontier_catalog": catalog,
                "fixed_policy": FIXED_POLICY,
            },
            "rule": (
                "the model proposes; the tools decide. This receipt has a null "
                "model_decided column because --build did not ask."
            ),
        },
        "model_decided": None,
        "tools_established": {
            "restatement_fails": restatement.get("different") is False,
            "pivot_passes": pivot.get("different") is True,
            "self_not_different": same.get("different") is False,
            "empty_refused": empty.get("refused") is True,
            "frontier_catalog": catalog,
            "fixed_policy": FIXED_POLICY,
            "provider_importable": imported is not None,
        },
        "negative_controls": {
            "reworded_restatement_fails": restatement.get("different") is False,
            "different_surface_passes": pivot.get("different") is True,
            "unavailable_does_not_count": True,
            "no_reason_does_not_count": True,
        },
        "prompt_budget": (
            "a handful of focused JSON decisions per minute; tool waits must not "
            "hold the inference slot (no_wait_scheduler owns poll)"
        ),
        "loop_entry": {
            "autonomy_run": (
                "tools/future/autonomy_run.py is the loop; the model must enter "
                "there. This lane cannot edit it. enter_loop() is the seam."
            ),
            "power_torture_transitions": [
                "REPLAN",
                "SUBAGENT_STATE",
                "NO_WAIT",
                "SCAR_PRUNING",
                "STATUS_CAUSALITY",
            ],
            "wired_into_autonomy_run": False,
        },
        "recovered_implementation": [
            "tools/future/autonomy_run.py — the loop; resident_model_cognition UNAVAILABLE; FIFO after scar filter",
            "tools/future/autonomy_trial.py — COGNITION_UNAVAILABLE; extract_orchestration_and_cognition keeps the fields independent",
            "tools/future/frontiers.py — next_work order is the fixed policy (gain then id); _tokens/_jaccard reused",
            "tools/future/flash_organ_pivot.py — refuse_if_restatement / restatement_verdict; scoped scar is not a restatement on a new surface",
            "tools/future/negative_index.py — refuse_if_dead / canon_family / canon_organ; tools still kill dead ideas",
            "tools/future/status_causality.py — challenge(); a status label is a hypothesis until the probe entails it",
            "tools/future/no_wait_scheduler.py — poll is supervisor-side; the reasoning context is already released",
            "tools/future/power_torture.py — REPLAN / SUBAGENT_STATE transition classes the hour must still demonstrate",
            "tools/future/orchestration.py — BINDINGS / invoke / emit_workunit; this module is not bound (outside WRITE)",
            "tools/future/mutation_engine.py — metabolism the loop still lacks; not claimed executed here",
            "receipts/future/evidence/RESIDENT_LIVE_PROBE.json — RESIDENT_STARTS_AND_GENERATES, cited not re-run",
        ],
        "gaps_closed": [
            "no cognition seam existed: interpret/choose/explain_failure/next_hypothesis/delegate/decision_log",
            "materially_participated() is now a computed predicate (divergence, different hyp B, enacted pick, reason)",
            "a reworded restatement fails the difference check on a shared fixture",
            "provider absence reports cognition UNAVAILABLE and refuses to claim participation",
            "a decision with no recorded reason does not count",
            "receipts keep model_decided and tools_established in separate columns",
        ],
        "negative_findings": [
            "this module is not in orchestration.BINDINGS (that table is outside this WRITE list)",
            "autonomy_run.py is not wired to enter_loop() (outside this WRITE list)",
            "--build did not start or ask the resident; cognition in this receipt is UNAVAILABLE",
            "if resident_provider is not importable, the sibling lane has not landed; that is UNAVAILABLE not a fake",
            "the live probe is SELF_MEASURED_DIRTY and does not establish an hour of load or that the resident reasons well",
            "a resident whose choices never diverge from the fixed policy is a publishable finding, not a pass",
        ],
        "resident_callable": {
            "entry_point": "tools.future.model_bearing.enter_loop() / choose() / interpret()",
            "workunit": (
                "one CPU_ANALYSIS unit; the resident proposes a WorkUnit against "
                "the live frontier, tools admit or refuse, and the tape records "
                "whether the pick diverged and whether it ran"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.CHILD_RESIDENT.launch",
            "fails_closed": (
                "provider absent -> cognition UNAVAILABLE, no scripted fallback; "
                "invented ids refused; scar-dead picks refused by tools; "
                "reworded hyp B fails meaningfully_different; no reason does not count; "
                "start/ask never implied by --build"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
