#!/usr/bin/env python3
"""Mandatory post-parent global review.

TRIGGER: a parent run reaches run_status "complete" or "honest_boundary_sealed".
Until this review has produced its artifacts AND every future Tier-A adapter has
acknowledged the resulting priors, the NEXT parent's heavy scientific controller
may not launch. The next source download MAY run concurrently: it is gated on
storage only, never on the review.

Artifacts per parent P (slug uppercased):
  P_VULTURE_HARVEST.json
  P_GLOBAL_METHODOLOGY_REVIEW.json  +  P_GLOBAL_METHODOLOGY_REVIEW.md
  P_ADAPTER_REBASE_MATRIX.json
  P_KERNEL_REBASE_MATRIX.json
  P_QUALITY_REBASE.json
  P_RESOURCE_REBASE.json
  P_GRAVITY_METHOD_PROMOTION.json
Global:
  CROSS_PARENT_TRANSFER_MATRIX.json
  PROVIDER_ADAPTER_LESSON_LEDGER.jsonl

Anti-drift: every adapter prescription is content hashed. An adapter is stale
unless its registry entry carries a rebase_ack for this parent whose digest
matches the freshly generated prescription. Regenerating the review with changed
priors invalidates every stale ack automatically. No silent methodology drift.

LAW: byte plan is not capability. Only a real parent-vs-packed forward with mean
symmetric KL <= 0.10 AND next-token argmax agreement >= 0.95 selects a frontier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

SCHEMA_NS = "hawking.foundry.post_parent_review"

# LAW thresholds. Never weakened after a failure. Evidence bundles that declare
# looser gates are rejected outright.
CAPABILITY_LAW = {
    "mean_symmetric_kl_max": 0.10,
    "next_token_argmax_agreement_min": 0.95,
    "rule": "byte plan != capability; only a real parent-vs-packed forward selects a frontier",
}

# Measured falsifications that must travel forward as NEGATIVE transfer so no
# future parent spends budget re-deriving them.
NEGATIVE_TRANSFER_PRIORS = [
    {
        "id": "inter_expert_redundancy_zero",
        "claim": "experts share exploitable redundancy",
        "verdict": "FALSIFIED",
        "evidence": "mean pairwise cosine 1e-4; residual 0.9350 vs orthogonal null 0.9354",
        "forbidden": ["expert delta coding", "shared low rank bases", "cluster mean subtraction"],
    },
    {
        "id": "entropy_coding_pq_indices",
        "claim": "entropy coding trained PQ indices buys 10 to 25 percent",
        "verdict": "FALSIFIED",
        "evidence": "index entropy 7.945/8.0; real gain 0.0 to 0.7 percent (Lloyd optimal indices are near uniform)",
        "forbidden": ["range coder on PQ indices as a rate lever"],
    },
    {
        "id": "posthoc_scalar_gain_correction",
        "claim": "a post hoc scalar gain fixes a PQ artifact",
        "verdict": "FALSIFIED",
        "evidence": "optimal gain pinned at exactly 1.0; k-means recon is a conditional mean, residual orthogonal to recon; cosine is gain invariant",
        "forbidden": ["scalar gain rescue pass on PQ output"],
    },
    {
        "id": "ternary_factorization",
        "claim": "ternary factorization beats VQ at matched rate",
        "verdict": "FALSIFIED",
        "evidence": "loses to VQ at matched rate on the measured ladder",
        "forbidden": ["ternary factorization as a primary representation"],
    },
    {
        "id": "aggressive_expert_cache",
        "claim": "a large expert cache accelerates a single lockstep pass",
        "verdict": "FALSIFIED",
        "evidence": "64GiB cap gave 0 evictions, drove RAM 70 to 18 GB and swap free to 906MB; zero cross layer reuse",
        "forbidden": ["cache caps above ~20GiB for single lockstep passes"],
        "replacement": "cap ~20GiB; aggressive RAM only where real reuse exists",
    },
]

REQUIRED_PER_PARENT = [
    "VULTURE_HARVEST.json",
    "GLOBAL_METHODOLOGY_REVIEW.json",
    "GLOBAL_METHODOLOGY_REVIEW.md",
    "ADAPTER_REBASE_MATRIX.json",
    "KERNEL_REBASE_MATRIX.json",
    "QUALITY_REBASE.json",
    "RESOURCE_REBASE.json",
    "GRAVITY_METHOD_PROMOTION.json",
]
REQUIRED_GLOBAL = ["CROSS_PARENT_TRANSFER_MATRIX.json", "PROVIDER_ADAPTER_LESSON_LEDGER.jsonl"]

SEALED_STATUSES = ("complete", "honest_boundary_sealed")


# ---------------------------------------------------------------- helpers


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slug(parent_id):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(parent_id)).strip("_").upper()


def digest(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return path


def _potency_backend():
    """Interop with tools/foundry/gravity_potency.py. Owned by another agent: import only."""
    try:
        import gravity_potency  # type: ignore
    except Exception:
        return None, "gravity_potency absent; local fallback ranking"
    for name in ("rank_methods", "potency", "score_method"):
        fn = getattr(gravity_potency, name, None)
        if callable(fn):
            return (name, fn), "gravity_potency." + name
    return None, "gravity_potency present but exposes no documented entry point; local fallback ranking"


_FALLBACK_POTENCY = {
    "CONFIRMED_MEASURED": 3.0,
    "VERIFIED": 3.0,
    "DIRECTIONAL": 2.0,
    "OPEN": 1.0,
    "UNTESTED": 1.0,
    "FALSIFIED": -1.0,
}


def rank_methods(methods):
    """Return (ranked_methods, backend_note). Never edits gravity_potency."""
    backend, note = _potency_backend()
    if backend and backend[0] == "rank_methods":
        try:
            return list(backend[1](methods)), note
        except Exception as exc:  # a foreign module must never block the review
            note = note + " (call failed: %s; local fallback)" % type(exc).__name__
            backend = None
    scored = []
    for m in methods:
        score = None
        if backend and backend[0] in ("potency", "score_method"):
            try:
                score = float(backend[1](m))
            except Exception:
                score = None
        if score is None:
            score = _FALLBACK_POTENCY.get(str(m.get("status", "OPEN")).upper(), 0.0)
            score *= float(m.get("transfer_breadth", 1.0))
        out = dict(m)
        out["potency"] = round(score, 4)
        scored.append(out)
    scored.sort(key=lambda m: (-m["potency"], m.get("id", "")))
    return scored, note


# ---------------------------------------------------------------- evidence


def load_evidence(path):
    with open(path) as fh:
        ev = json.load(fh)
    return validate_evidence(ev)


def validate_evidence(ev):
    for key in ("parent", "run_status"):
        if key not in ev:
            raise ValueError("evidence bundle missing required key: %s" % key)
    if "id" not in ev["parent"]:
        raise ValueError("evidence bundle parent missing id")
    declared = (ev.get("quality") or {}).get("capability_gate") or {}
    kl = declared.get("mean_symmetric_kl_max")
    ag = declared.get("next_token_argmax_agreement_min")
    if kl is not None and float(kl) > CAPABILITY_LAW["mean_symmetric_kl_max"]:
        raise ValueError("refusing weakened capability gate: sym KL max %s > %s" % (kl, CAPABILITY_LAW["mean_symmetric_kl_max"]))
    if ag is not None and float(ag) < CAPABILITY_LAW["next_token_argmax_agreement_min"]:
        raise ValueError("refusing weakened capability gate: argmax agreement min %s < %s" % (ag, CAPABILITY_LAW["next_token_argmax_agreement_min"]))
    return ev


def is_provisional(ev):
    return ev.get("run_status") not in SEALED_STATUSES


# ---------------------------------------------------------------- artifacts


def build_vulture_harvest(ev):
    p = ev["parent"]
    rep = ev.get("representation") or {}
    org = ev.get("organ_sensitivity") or {}
    return {
        "schema": SCHEMA_NS + ".vulture_harvest.v1",
        "generated_at_utc": _utc(),
        "parent": p,
        "run_status": ev["run_status"],
        "provisional": is_provisional(ev),
        "representation_winners": rep.get("winners", []),
        "rate_response_curve": rep.get("rate_response", []),
        "organ_sensitivity": {
            "organs": org.get("organs", {}),
            "dominant_failure_organ": org.get("dominant_failure_organ"),
            "inversion": org.get("inversion", "sensitive organ is mlp1/gate+up; mlp2/down tolerates more"),
            "inversion_confirmed": org.get("inversion_confirmed"),
        },
        "doctor": {
            "successes": (ev.get("doctor") or {}).get("successes", []),
            "failures": (ev.get("doctor") or {}).get("failures", []),
        },
        "routing_evidence": ev.get("routing", {}),
        "activation_evidence": ev.get("activation", {}),
        "quality_failures": (ev.get("quality") or {}).get("failures", []),
        "capability_gate_result": (ev.get("quality") or {}).get("result", {}),
        "runtime_timings": (ev.get("runtime") or {}).get("timings", []),
        "runtime_dominant_bottleneck": (ev.get("runtime") or {}).get("dominant_bottleneck"),
        "cache_ram_swap_behavior": ev.get("resources", {}),
        "source_format_lessons": (ev.get("source_format") or {}).get("lessons", []),
        "storage_lessons": (ev.get("storage") or {}).get("lessons", []),
        "negative_transfer_constraints": NEGATIVE_TRANSFER_PRIORS + list(ev.get("negative_transfer", [])),
        "law": CAPABILITY_LAW,
    }


_REVIEW_QUESTIONS = [
    ("capability_cliff_moved_with_scale", "did the capability cliff move with scale"),
    ("geometry_transferred", "which geometry transferred from the previous parent"),
    ("hot_cold_routing_mattered", "did hot/cold routing separation matter"),
    ("codebook_sharing_structure_or_amortization", "did codebook sharing help through structure or only amortization"),
    ("higher_dimensional_vq_helped", "did higher dimensional VQ help"),
    ("row_norm_stratification_helped", "did row norm stratification help"),
    ("organs_consuming_most_rescue_bytes", "which organs consumed the most rescue bytes"),
    ("quality_domains_collapsing_first", "which quality domains collapsed first"),
    ("dominant_runtime_bottleneck", "which runtime bottleneck dominated"),
]


def build_methodology_review(ev, harvest):
    answers = ev.get("review_answers") or {}
    assumptions = ev.get("assumptions") or []
    review = {
        "schema": SCHEMA_NS + ".global_methodology_review.v1",
        "generated_at_utc": _utc(),
        "parent": ev["parent"],
        "run_status": ev["run_status"],
        "provisional": is_provisional(ev),
        "assumptions_confirmed": [a for a in assumptions if a.get("verdict") == "CONFIRMED"],
        "assumptions_falsified": [a for a in assumptions if a.get("verdict") == "FALSIFIED"],
        "assumptions_open": [a for a in assumptions if a.get("verdict") not in ("CONFIRMED", "FALSIFIED")],
        "questions": {
            key: answers.get(key, {"answer": "UNANSWERED", "evidence": None, "question": text})
            for key, text in _REVIEW_QUESTIONS
        },
        "capability_gate_result": harvest["capability_gate_result"],
        "law": CAPABILITY_LAW,
    }
    return review, render_methodology_md(review, harvest)


def render_methodology_md(review, harvest):
    p = review["parent"]
    lines = [
        "# Global methodology review: %s" % p.get("label", p["id"]),
        "",
        "generated %s" % review["generated_at_utc"],
        "run_status %s%s" % (review["run_status"], "  PROVISIONAL" if review["provisional"] else ""),
        "",
        "## Law",
        "byte plan is not capability. mean symmetric KL <= %.2f AND next token argmax agreement >= %.2f."
        % (CAPABILITY_LAW["mean_symmetric_kl_max"], CAPABILITY_LAW["next_token_argmax_agreement_min"]),
        "",
        "## Assumptions confirmed",
    ]
    for a in review["assumptions_confirmed"] or [{"statement": "none"}]:
        lines.append("- %s %s" % (a.get("statement", a.get("id", "?")), _ev(a)))
    lines += ["", "## Assumptions falsified"]
    for a in review["assumptions_falsified"] or [{"statement": "none"}]:
        lines.append("- %s %s" % (a.get("statement", a.get("id", "?")), _ev(a)))
    lines += ["", "## Assumptions still open"]
    for a in review["assumptions_open"] or [{"statement": "none"}]:
        lines.append("- %s %s" % (a.get("statement", a.get("id", "?")), _ev(a)))
    lines += ["", "## Mandatory questions"]
    for key, text in _REVIEW_QUESTIONS:
        ans = review["questions"][key]
        lines.append("- %s: %s %s" % (text, ans.get("answer"), _ev(ans)))
    lines += [
        "",
        "## Organ sensitivity",
        "dominant failure organ: %s" % harvest["organ_sensitivity"]["dominant_failure_organ"],
        harvest["organ_sensitivity"]["inversion"],
        "",
        "## Negative transfer (do not re-derive)",
    ]
    for n in harvest["negative_transfer_constraints"]:
        lines.append("- %s: %s [%s]" % (n.get("id"), n.get("claim"), n.get("evidence")))
    lines.append("")
    return "\n".join(lines)


def _ev(d):
    e = d.get("evidence")
    return "[%s]" % e if e else ""


def _rate_priors(ev):
    curve = sorted((ev.get("representation") or {}).get("rate_response", []), key=lambda r: r.get("rate_bpw", 0))
    return {
        "search_start_bpw": curve[-1]["rate_bpw"] if curve else None,
        "measured_curve": curve,
        "not_a_selection": True,
        "note": "rate priors seed the search only; a real forward selects",
    }


def build_adapter_rebase_matrix(ev, harvest, adapters):
    org = harvest["organ_sensitivity"]
    shared = {
        "organ_taxonomy": sorted(org["organs"].keys()) or ["gate", "up", "down", "attn", "embed", "lm_head"],
        "sensitive_organ_prior": org["dominant_failure_organ"] or "gate",
        "organ_inversion": org["inversion"],
        "candidate_ordering": [w.get("name") for w in harvest["representation_winners"]]
        + ["DO_NOT_ATTEMPT:" + n["id"] for n in harvest["negative_transfer_constraints"]],
        "rate_priors": _rate_priors(ev),
        "doctor_eligibility": {
            "eligible": [d.get("target") for d in harvest["doctor"]["successes"]],
            "ineligible": [d.get("target") for d in harvest["doctor"]["failures"]],
        },
        "codebook_sharing_scope": (ev.get("codebook_sharing") or {}).get(
            "scope", "amortization only; structural sharing measured at zero gain"
        ),
        "routing_calibration_plan": {
            "min_calibration_tokens": (ev.get("routing") or {}).get("required_calibration_tokens", 1000),
            "rejected": "88 tokens (63.6 pct stable at median, 26.1 pct of cells never routed)",
        },
        "source_decoder_requirements": (ev.get("source_format") or {}).get("decoder_requirements", []),
        "cache_policy": {"expert_cache_cap_gib": 20, "rationale": "single lockstep pass has zero cross layer reuse"},
        "memory_floor_gib": (ev.get("resources") or {}).get("memory_floor_gib"),
        "storage_mode": (ev.get("storage") or {}).get("mode", "release source after harvest; re-download from pinned revision"),
        "kernel_requirements": (ev.get("runtime") or {}).get("kernel_requirements", []),
        "quality_probes": (ev.get("quality") or {}).get("probes", []),
        "stopping_rules": {
            "capability_law": CAPABILITY_LAW,
            "stop_on": "collapsed logits at any rate; a 0.3 BPW file with collapsed logits is not a win",
        },
        "negative_transfer": harvest["negative_transfer_constraints"],
    }
    entries = {}
    for a in adapters:
        pres = dict(shared)
        pres["adapter_id"] = a["id"]
        pres["consumes_parent_lessons"] = sorted(set(list(a.get("consumes_parent_lessons", [])) + [ev["parent"]["id"]]))
        pres["unverified_assumptions"] = a.get("unverified_assumptions", [])
        pres["falsification_plan"] = a.get("falsification_plan")
        pres["prescription_digest"] = digest(pres)
        entries[a["id"]] = pres
    return {
        "schema": SCHEMA_NS + ".adapter_rebase_matrix.v1",
        "generated_at_utc": _utc(),
        "parent": ev["parent"],
        "provisional": is_provisional(ev),
        "shared_priors_digest": digest(shared),
        "adapters": entries,
    }


def build_kernel_rebase_matrix(ev, harvest):
    rt = ev.get("runtime") or {}
    return {
        "schema": SCHEMA_NS + ".kernel_rebase_matrix.v1",
        "generated_at_utc": _utc(),
        "parent": ev["parent"],
        "provisional": is_provisional(ev),
        "dominant_bottleneck": rt.get("dominant_bottleneck"),
        "timings": rt.get("timings", []),
        "required_kernels": rt.get("kernel_requirements", []),
        "backend_parity_rule": "backends may not silently select different scientific winners",
    }


def build_quality_rebase(ev, harvest):
    q = ev.get("quality") or {}
    return {
        "schema": SCHEMA_NS + ".quality_rebase.v1",
        "generated_at_utc": _utc(),
        "parent": ev["parent"],
        "provisional": is_provisional(ev),
        "law": CAPABILITY_LAW,
        "probes": q.get("probes", []),
        "domains_collapsing_first": q.get("domains_collapsing_first", []),
        "failures": q.get("failures", []),
        "result": q.get("result", {}),
    }


def build_resource_rebase(ev, harvest):
    res = ev.get("resources") or {}
    return {
        "schema": SCHEMA_NS + ".resource_rebase.v1",
        "generated_at_utc": _utc(),
        "parent": ev["parent"],
        "provisional": is_provisional(ev),
        "cache_policy": {"expert_cache_cap_gib": 20, "falsified": "64GiB cap (0 evictions, RAM 70 to 18 GB, swap free 906MB)"},
        "observed": res,
        "memory_floor_gib": res.get("memory_floor_gib"),
        "storage": ev.get("storage", {}),
    }


def build_gravity_method_promotion(ev, harvest):
    methods = list(ev.get("methods") or [])
    for n in harvest["negative_transfer_constraints"]:
        methods.append({"id": n["id"], "name": n.get("claim"), "status": "FALSIFIED", "evidence": n.get("evidence")})
    ranked, backend_note = rank_methods(methods)
    return {
        "schema": SCHEMA_NS + ".gravity_method_promotion.v1",
        "generated_at_utc": _utc(),
        "parent": ev["parent"],
        "provisional": is_provisional(ev),
        "potency_backend": backend_note,
        "promoted": [m for m in ranked if m.get("potency", 0) >= 3.0],
        "retained_candidate": [m for m in ranked if 0 < m.get("potency", 0) < 3.0],
        "retired": [m for m in ranked if m.get("potency", 0) <= 0],
        "ranked": ranked,
    }


def update_cross_parent_matrix(path, ev, harvest, review):
    doc = {"schema": SCHEMA_NS + ".cross_parent_transfer_matrix.v1", "parents": {}}
    if os.path.exists(path):
        with open(path) as fh:
            doc = json.load(fh)
    doc.setdefault("parents", {})[ev["parent"]["id"]] = {
        "label": ev["parent"].get("label"),
        "generation": ev["parent"].get("generation"),
        "run_status": ev["run_status"],
        "provisional": is_provisional(ev),
        "dominant_failure_organ": harvest["organ_sensitivity"]["dominant_failure_organ"],
        "inversion_confirmed": harvest["organ_sensitivity"]["inversion_confirmed"],
        "confirmed": [a.get("id") for a in review["assumptions_confirmed"]],
        "falsified": [a.get("id") for a in review["assumptions_falsified"]],
        "open": [a.get("id") for a in review["assumptions_open"]],
        "reviewed_at_utc": review["generated_at_utc"],
    }
    doc["negative_transfer"] = NEGATIVE_TRANSFER_PRIORS
    doc["law"] = CAPABILITY_LAW
    doc["updated_at_utc"] = _utc()
    return _write_json(path, doc)


def append_lesson_ledger(path, ev, harvest):
    """Append one line per lesson. Dedupe by (parent, lesson_id)."""
    seen = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    seen.add((r.get("parent_id"), r.get("lesson_id")))
    lessons = []
    for n in harvest["negative_transfer_constraints"]:
        lessons.append({"lesson_id": n["id"], "kind": "negative_transfer", "detail": n})
    for w in harvest["representation_winners"]:
        lessons.append({"lesson_id": "winner:" + str(w.get("name")), "kind": "representation", "detail": w})
    for lesson in harvest["source_format_lessons"]:
        lessons.append({"lesson_id": "source_format:" + str(lesson.get("id", lesson))[:40], "kind": "source_format", "detail": lesson})
    for lesson in harvest["storage_lessons"]:
        lessons.append({"lesson_id": "storage:" + str(lesson.get("id", lesson))[:40], "kind": "storage", "detail": lesson})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    added = 0
    with open(path, "a") as fh:
        for lesson in lessons:
            key = (ev["parent"]["id"], lesson["lesson_id"])
            if key in seen:
                continue
            seen.add(key)
            rec = dict(lesson)
            rec.update({
                "schema": SCHEMA_NS + ".lesson.v1",
                "parent_id": ev["parent"]["id"],
                "parent_label": ev["parent"].get("label"),
                "provisional": is_provisional(ev),
                "recorded_at_utc": _utc(),
            })
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            added += 1
    return added


def generate(ev, out_dir, adapters):
    """Write every artifact. Returns {name: path}."""
    p = slug(ev["parent"]["id"])
    harvest = build_vulture_harvest(ev)
    review, review_md = build_methodology_review(ev, harvest)
    written = {}
    written["VULTURE_HARVEST.json"] = _write_json(os.path.join(out_dir, p + "_VULTURE_HARVEST.json"), harvest)
    written["GLOBAL_METHODOLOGY_REVIEW.json"] = _write_json(os.path.join(out_dir, p + "_GLOBAL_METHODOLOGY_REVIEW.json"), review)
    md_path = os.path.join(out_dir, p + "_GLOBAL_METHODOLOGY_REVIEW.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(md_path, "w") as fh:
        fh.write(review_md)
    written["GLOBAL_METHODOLOGY_REVIEW.md"] = md_path
    written["ADAPTER_REBASE_MATRIX.json"] = _write_json(
        os.path.join(out_dir, p + "_ADAPTER_REBASE_MATRIX.json"), build_adapter_rebase_matrix(ev, harvest, adapters)
    )
    written["KERNEL_REBASE_MATRIX.json"] = _write_json(
        os.path.join(out_dir, p + "_KERNEL_REBASE_MATRIX.json"), build_kernel_rebase_matrix(ev, harvest)
    )
    written["QUALITY_REBASE.json"] = _write_json(os.path.join(out_dir, p + "_QUALITY_REBASE.json"), build_quality_rebase(ev, harvest))
    written["RESOURCE_REBASE.json"] = _write_json(os.path.join(out_dir, p + "_RESOURCE_REBASE.json"), build_resource_rebase(ev, harvest))
    written["GRAVITY_METHOD_PROMOTION.json"] = _write_json(
        os.path.join(out_dir, p + "_GRAVITY_METHOD_PROMOTION.json"), build_gravity_method_promotion(ev, harvest)
    )
    written["CROSS_PARENT_TRANSFER_MATRIX.json"] = update_cross_parent_matrix(
        os.path.join(out_dir, "CROSS_PARENT_TRANSFER_MATRIX.json"), ev, harvest, review
    )
    ledger = os.path.join(out_dir, "PROVIDER_ADAPTER_LESSON_LEDGER.jsonl")
    append_lesson_ledger(ledger, ev, harvest)
    written["PROVIDER_ADAPTER_LESSON_LEDGER.jsonl"] = ledger
    return written


# ---------------------------------------------------------------- drift + gate


def find_stale_adapters(parent, adapters, matrix=None):
    """Adapters that have not been rebased against this completed parent.

    Stale when: no ack for the parent, ack digest does not match the freshly
    generated prescription (priors changed under the adapter), or the adapter
    fails to declare consumed lessons / unverified assumptions / a plan to
    falsify the transferred priors.
    """
    parent_id = parent["id"] if isinstance(parent, dict) else parent
    entries = (matrix or {}).get("adapters", {})
    stale = []
    for a in adapters:
        reasons = []
        ack = (a.get("rebase_acks") or {}).get(parent_id)
        if not ack:
            reasons.append("no rebase_ack for parent %s" % parent_id)
        else:
            want = entries.get(a["id"], {}).get("prescription_digest")
            if want and ack.get("prescription_digest") != want:
                reasons.append("prescription digest drift: ack %s want %s" % (ack.get("prescription_digest"), want))
        if not a.get("consumes_parent_lessons"):
            reasons.append("does not declare which completed-parent lessons it consumes")
        if a.get("unverified_assumptions") is None:
            reasons.append("does not declare remaining unverified assumptions")
        if not a.get("falsification_plan"):
            reasons.append("does not declare how it will falsify the transferred priors")
        if reasons:
            stale.append({"adapter_id": a["id"], "reasons": reasons})
    return stale


def build_gate_state(out_dir, ev, adapters, matrix=None, heavy_lease_held=False, storage=None):
    p = slug(ev["parent"]["id"])
    present = {}
    for name in REQUIRED_PER_PARENT:
        present[name] = os.path.exists(os.path.join(out_dir, p + "_" + name))
    for name in REQUIRED_GLOBAL:
        present[name] = os.path.exists(os.path.join(out_dir, name))
    return {
        "completed_parent": ev["parent"]["id"],
        "parent_run_status": ev["run_status"],
        "review_provisional": is_provisional(ev),
        "artifacts_present": present,
        "stale_adapters": find_stale_adapters(ev["parent"], adapters, matrix),
        "heavy_lease_held": heavy_lease_held,
        "storage": storage or {},
    }


def can_launch_next_parent(state):
    """(allowed, reasons) for the NEXT parent's HEAVY scientific controller.

    Download is a separate gate: see can_start_next_download. A pending review
    never blocks a download, and a satisfied review never authorises a heavy
    launch while the single heavy lease is held.
    """
    reasons = []
    if state.get("parent_run_status") not in SEALED_STATUSES:
        reasons.append("parent %s run_status %s is not complete or honest_boundary_sealed" % (state.get("completed_parent"), state.get("parent_run_status")))
    if state.get("review_provisional"):
        reasons.append("post parent review is provisional; a provisional review does not unlock the next parent")
    missing = sorted(n for n, ok in (state.get("artifacts_present") or {}).items() if not ok)
    if missing:
        reasons.append("missing review artifacts: " + ", ".join(missing))
    if not state.get("artifacts_present"):
        reasons.append("post parent review has not run")
    stale = state.get("stale_adapters") or []
    if stale:
        reasons.append("stale adapters (silent methodology drift): " + ", ".join(s["adapter_id"] for s in stale))
    if state.get("heavy_lease_held"):
        reasons.append("the single heavy lease is held by a live campaign")
    return (not reasons), reasons


def can_start_next_download(state):
    """(allowed, reasons). Storage only. Deliberately independent of the review."""
    reasons = []
    st = state.get("storage") or {}
    need = float(st.get("required_gib", 0.0)) + float(st.get("headroom_gib", 0.0))
    free = st.get("free_gib")
    if free is None:
        reasons.append("free_gib unknown")
    elif float(free) < need:
        reasons.append("insufficient storage: free %.1f GiB < required+headroom %.1f GiB" % (float(free), need))
    if st.get("download_in_flight"):
        reasons.append("a download is already in flight")
    return (not reasons), reasons


# ---------------------------------------------------------------- cli


def _load(path, default=None):
    if path is None:
        return default
    with open(path) as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["generate", "gate", "stale"])
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--adapters", help="Tier-A adapter registry JSON (list, or {adapters: [...]})")
    ap.add_argument("--out", default="reports/foundry/post_parent_review")
    ap.add_argument("--heavy-lease-held", action="store_true")
    args = ap.parse_args(argv)

    ev = load_evidence(args.evidence)
    reg = _load(args.adapters, []) or []
    adapters = reg.get("adapters", []) if isinstance(reg, dict) else reg

    if args.command == "generate":
        written = generate(ev, args.out, adapters)
        print(json.dumps({"parent": ev["parent"]["id"], "provisional": is_provisional(ev), "written": written}, indent=1))
        return 0

    matrix_path = os.path.join(args.out, slug(ev["parent"]["id"]) + "_ADAPTER_REBASE_MATRIX.json")
    matrix = _load(matrix_path) if os.path.exists(matrix_path) else None

    if args.command == "stale":
        stale = find_stale_adapters(ev["parent"], adapters, matrix)
        print(json.dumps(stale, indent=1))
        return 1 if stale else 0

    state = build_gate_state(args.out, ev, adapters, matrix, heavy_lease_held=args.heavy_lease_held, storage=(ev.get("next_parent") or {}).get("storage"))
    heavy_ok, heavy_reasons = can_launch_next_parent(state)
    dl_ok, dl_reasons = can_start_next_download(state)
    print(json.dumps({
        "heavy_controller_launch_allowed": heavy_ok,
        "heavy_blocking_reasons": heavy_reasons,
        "source_download_allowed": dl_ok,
        "download_blocking_reasons": dl_reasons,
        "state": state,
    }, indent=1))
    return 0 if heavy_ok else 2


if __name__ == "__main__":
    sys.exit(main())
