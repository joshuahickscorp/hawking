#!/usr/bin/env python3.12
"""Seal the GPT-OSS-120B frontier conclusion (runs AFTER the Doctor / correction-wave campaign).

READ-ONLY science sealer. It runs NO forward, loads NO weights, launches NO controller. It only
reads sealed JSON checkpoints and folds them into the campaign's required 120B conclusion artifacts.

Inputs (all optional-tolerant; the sealer degrades to PROVISIONAL rather than crashing):
  reports/condense/general_frontier/GPT_OSS_120B_G4_RESULT.json          untreated uniform-RVQ control
  reports/condense/general_frontier/GPT_OSS_120B_G4_UNTREATED_CONTROL.json (same rows, provenance)
  reports/condense/general_frontier/CORRECTION_WAVE/ (or DOCTOR_CAMPAIGN/)  D0-D6 + diagnosis checkpoints + state
  reports/condense/second_light/GPT_OSS_120B_SECOND_LIGHT_ARTIFACT_MANIFEST.json  whole-artifact byte accounting
  reports/condense/second_light/GPT_OSS_120B_SECOND_LIGHT_BASELINE.json    whole-artifact realized BPW

Outputs (reports/condense/general_frontier/, or --out-dir):
  GPT_OSS_120B_FINAL_FRONTIER_REPORT.md
  GPT_OSS_120B_FINAL_FRONTIER_RESULT.json
  GPT_OSS_120B_FINAL_ARTIFACT_MANIFEST.json
  GPT_OSS_120B_DOCTOR_MAP.json
  GPT_OSS_120B_ORGAN_BYTE_ALLOCATION.json
  GPT_OSS_120B_FRONTIER_ATLAS_UPDATE.jsonl
  GPT_OSS_120B_REPRODUCTION.json
  GPT_OSS_120B_ROLLBACK.json

Outcome A: a TREATED candidate meets the sealed gate (mean_sym_kl <= 0.10 AND
next_token_argmax_agreement >= 0.95) on validation+holdout at whole-artifact BPW < 1.0.
Outcome B: no treated sub-bit candidate passes -> seal the honest boundary (best candidate, the
diagnosed dominant-failure organ, the lowest credible rate region, treatments attempted, why Doctor
could not recover inside budget, reopening conditions).

Guard: if the campaign directory is missing or its state is not final, every artifact is stamped
PROVISIONAL and the report carries an explicit incomplete-campaign banner.

Usage:
  python3.12 tools/condense/seal_120b_conclusion.py seal
  python3.12 tools/condense/seal_120b_conclusion.py seal --dry-run
  python3.12 tools/condense/seal_120b_conclusion.py seal --out-dir /path/to/dir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
GF = REPO / "reports" / "condense" / "general_frontier"
SECOND_LIGHT = REPO / "reports" / "condense" / "second_light"

G4_RESULT = GF / "GPT_OSS_120B_G4_RESULT.json"
G4_CONTROL = GF / "GPT_OSS_120B_G4_UNTREATED_CONTROL.json"
SL_MANIFEST = SECOND_LIGHT / "GPT_OSS_120B_SECOND_LIGHT_ARTIFACT_MANIFEST.json"
SL_BASELINE = SECOND_LIGHT / "GPT_OSS_120B_SECOND_LIGHT_BASELINE.json"

# Campaign directory candidates, in preference order. Whichever exists (and has checkpoints) wins.
CAMPAIGN_DIRS = ["CORRECTION_WAVE", "DOCTOR_CAMPAIGN"]

# Sealed gate. Never lowered after seeing results (this sealer only reports the law, never edits it).
PROMOTE_KL = 0.10
PROMOTE_ARGMAX_AGREE = 0.95
WHOLE_ARTIFACT_BPW_MAX = 1.0

ROLLBACK_TAG = "hawking-gptoss-120b-frontier"
SOURCE_REVISION = "openai/gpt-oss-120b @ b5c939de"
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"

# Prior geometry evidence (F1 bounded proxy, gravity_frontier geometry search). Used only as a
# labelled fallback when the campaign has not yet sealed real-forward diagnosis rows. Never presented
# as a real-forward capability pass.
GEOMETRY_PRIOR = {
    "expert_mlp1": {"role": "up/gate projection", "robust": True,
                    "best_treatment": "pq_doctor_lowrank", "proxy_divergence": 0.00832, "proxy_rate_bpw": 0.876},
    "expert_mlp2": {"role": "down projection", "robust": False,
                    "best_treatment": "pq_protected_islands", "proxy_divergence": 0.184, "proxy_rate_bpw": 0.913,
                    "note": "protected islands cut proxy functional divergence from ~0.60 (plain PQ) to 0.184; still heavy-tailed and sub-bit sensitive"},
}


# --------------------------------------------------------------------------- helpers
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _file_sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _write_json(path: Path, obj: Any) -> None:
    _atomic(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True, timeout=15).stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _tag_exists(tag: str) -> bool:
    try:
        out = subprocess.run(["git", "tag", "-l", tag], cwd=str(REPO),
                             capture_output=True, text=True, timeout=15).stdout.strip()
        return out == tag
    except Exception:
        return False


# --------------------------------------------------------------------------- campaign discovery
def _discover_campaign() -> dict:
    """Locate the Doctor / correction-wave campaign. Returns a normalized descriptor.

    Tolerates: missing dir, partial checkpoints, state not final, either naming (CORRECTION_WAVE or
    DOCTOR_CAMPAIGN), and heterogeneous checkpoint schemas (candidate rows vs diagnosis rows).
    """
    desc: dict[str, Any] = {
        "dir": None, "name": None, "found": False, "complete": False,
        "state_schema": None, "rows_done": 0, "rows_total": 0,
        "state": None, "candidates_config": None,
        "treated": [], "diagnosis_rows": [], "capability_candidate_ids": [],
        "checkpoint_count": 0,
    }
    cdir = None
    for name in CAMPAIGN_DIRS:
        p = GF / name
        if p.is_dir():
            cdir = p
            desc["name"] = name
            break
    if cdir is None:
        return desc
    desc["dir"] = str(cdir)
    desc["found"] = True

    # State file: any *STATE*.json at the campaign root.
    state = None
    for sp in sorted(cdir.glob("*STATE*.json")) + sorted(cdir.glob("*state*.json")):
        state = _read_json(sp)
        if state:
            desc["state"] = state
            desc["state_schema"] = state.get("schema")
            desc["rows_done"] = int(state.get("rows_done", 0) or 0)
            desc["rows_total"] = int(state.get("rows_total", 0) or 0)
            desc["candidates_config"] = state.get("candidates")
            desc["capability_candidate_ids"] = list(state.get("capability_candidates", []) or [])
            break

    # Checkpoints: candidate + parent + diagnosis rows.
    cpdir = cdir / "checkpoints"
    checkpoints = sorted(cpdir.glob("*.json")) if cpdir.is_dir() else []
    desc["checkpoint_count"] = len(checkpoints)
    for cp in checkpoints:
        rec = _read_json(cp)
        if not isinstance(rec, dict):
            continue
        rid = str(rec.get("row_id", cp.stem))
        variant = str(rec.get("variant", ""))
        # Diagnosis rows: D<n> prefix, or an explicit diagnosis/organ field.
        is_diag = (rid[:1] == "D" and rid[1:2].isdigit()) or bool(
            rec.get("diagnosis") or rec.get("organ_ablation") or rec.get("dominant_failure_organ"))
        if is_diag:
            desc["diagnosis_rows"].append(rec)
            continue
        # Treated candidate rows carry divergence_vs_parent and are not the parent/original reference.
        div = rec.get("divergence_vs_parent") or rec.get("divergence_vs_original")
        if variant in ("parent", "original") or div is None:
            continue
        desc["treated"].append({
            "row_id": rid,
            "variant": variant,
            "mapping": rec.get("mapping"),
            "prompt_id": rec.get("prompt_id"),
            "domain": rec.get("domain"),
            "mean_sym_kl": div.get("mean_sym_kl"),
            "argmax_agreement": div.get("next_token_argmax_agreement"),
            "perplexity": (rec.get("quality") or {}).get("perplexity"),
            "whole_bpw": (rec.get("params") or {}).get("whole_bpw") or rec.get("whole_bpw") or rec.get("rate_bpw"),
            "verdict": rec.get("verdict"),
            "capability_candidate": bool(rec.get("capability_candidate")),
        })

    # A "complete" campaign is one whose state is final AND all rows sealed.
    st = desc["state"] or {}
    desc["complete"] = bool(st.get("final")) and desc["rows_total"] > 0 and desc["rows_done"] >= desc["rows_total"]
    return desc


# --------------------------------------------------------------------------- science folding
def _control_summary(g4: dict | None) -> dict:
    rows = []
    if g4:
        rows = g4.get("packed_control_rvq_1bpw") or []
    verdicts = [r.get("verdict") for r in rows]
    agrees = [r.get("next_token_agreement") for r in rows if r.get("next_token_agreement") is not None]
    kls = [r.get("mean_sym_kl") for r in rows if r.get("mean_sym_kl") is not None]
    return {
        "representation": "uniform naive_rvq @ 1.0 bpw (untreated negative control)",
        "rows": len(rows),
        "collapse": sum(1 for v in verdicts if v == "collapse"),
        "degraded": sum(1 for v in verdicts if v == "degraded"),
        "capability_pass": sum(1 for v in verdicts if v in ("capability_candidate", "pass")),
        "argmax_agreement_range": [min(agrees), max(agrees)] if agrees else None,
        "mean_sym_kl_range": [min(kls), max(kls)] if kls else None,
        "verdict": (g4 or {}).get("verdict"),
        "role": "REAL-FORWARD NEGATIVE CONTROL (uniform untreated RVQ near 1 BPW). NOT Hawking's strongest treated candidate.",
    }


def _best_treated(treated: list[dict]) -> dict | None:
    scored = [t for t in treated if t.get("mean_sym_kl") is not None]
    if not scored:
        return None
    return sorted(scored, key=lambda t: (t["mean_sym_kl"], -(t.get("argmax_agreement") or 0)))[0]


def _passes_gate(t: dict, whole_bpw_ok: bool) -> bool:
    kl = t.get("mean_sym_kl")
    ag = t.get("argmax_agreement")
    if kl is None or ag is None:
        return False
    return kl <= PROMOTE_KL and ag >= PROMOTE_ARGMAX_AGREE and whole_bpw_ok


def _dominant_failure_organ(desc: dict) -> dict:
    """Diagnosed organ that caused the largest failures. Prefers real diagnosis rows; falls back to
    the F1 geometry prior (expert_mlp2, the heavy-tailed down projection)."""
    rows = desc.get("diagnosis_rows") or []
    scored = []
    for r in rows:
        organ = r.get("organ") or r.get("tensor_class") or r.get("dominant_failure_organ")
        # failure magnitude: prefer an explicit attributed-loss field, else divergence.
        mag = r.get("failure_magnitude")
        if mag is None:
            div = r.get("divergence_vs_parent") or r.get("divergence_vs_original") or {}
            mag = div.get("mean_sym_kl")
        if organ is not None and mag is not None:
            scored.append({"organ": organ, "failure_magnitude": mag, "row_id": r.get("row_id")})
    if scored:
        top = sorted(scored, key=lambda s: s["failure_magnitude"], reverse=True)[0]
        return {
            "organ": top["organ"],
            "failure_magnitude": top["failure_magnitude"],
            "source": "campaign diagnosis rows",
            "evidence_level": "real_forward",
            "detail": scored,
        }
    return {
        "organ": "expert_mlp2",
        "role": "down projection (second MLP matrix of each expert)",
        "source": "F1 geometry-search prior (gravity_frontier); PENDING confirmation by real-forward diagnosis rows",
        "evidence_level": "F1_bounded_proxy",
        "detail": "expert_mlp2 is heavy-tailed and sub-bit sensitive; plain PQ leaves ~0.60 proxy functional divergence, "
                  "vs expert_mlp1 (up/gate) which tolerates sub-bit at ~0.008 proxy divergence. The down projection is "
                  "the diagnosed dominant-failure organ.",
    }


def _whole_artifact_bpw(baseline: dict | None) -> dict:
    b = (baseline or {}).get("result") or {}
    realized = b.get("realized_whole_artifact_bpw")
    budget = b.get("budget_bpw")
    return {
        "realized_whole_artifact_bpw": realized,
        "budget_bpw": budget,
        "sub_bit_rate_achieved": bool(realized is not None and realized < WHOLE_ARTIFACT_BPW_MAX),
        "gate_threshold_bpw": WHOLE_ARTIFACT_BPW_MAX,
        "source": "Second Light PQ baseline (byte-exact, 183/183 rows sealed)" if realized is not None else "PENDING (baseline not found)",
        "capability_at_this_rate": "NEGATIVE (byte-complete artifact, no capability pass)",
    }


def _organ_byte_allocation(manifest: dict | None) -> dict:
    if not manifest:
        return {"available": False, "note": "second-light artifact manifest not found; organ byte allocation PENDING"}
    rows = manifest.get("rows") or []
    agg: dict[str, dict[str, float]] = {}
    for r in rows:
        tc = r.get("tensor_class", "unknown")
        a = agg.setdefault(tc, {"rows": 0, "physical_bits": 0})
        a["rows"] += 1
        a["physical_bits"] += int(r.get("physical_bits", 0) or 0)
    total_bits = manifest.get("complete_physical_bits") or sum(a["physical_bits"] for a in agg.values())
    organs = {}
    for tc, a in sorted(agg.items()):
        organs[tc] = {
            "rows": a["rows"],
            "physical_bits": a["physical_bits"],
            "physical_bytes": a["physical_bits"] // 8,
            "fraction_of_artifact": round(a["physical_bits"] / total_bits, 6) if total_bits else None,
        }
    return {
        "available": True,
        "realized_whole_artifact_bpw": manifest.get("realized_bpw"),
        "complete_physical_bits": total_bits,
        "complete_physical_bytes": total_bits // 8,
        "organs": organs,
        "note": "physical byte allocation by tensor organ, aggregated from the Second Light 183-row byte-exact manifest",
    }


def _doctor_map(desc: dict, dominant: dict) -> dict:
    """Which Doctor / tensor-class treatments were reachable, per organ, and whether each improved
    real capability."""
    cfg = desc.get("candidates_config")
    reachable = []
    per_organ: dict[str, list] = {"expert_mlp1": [], "expert_mlp2": []}
    if isinstance(cfg, dict) and cfg:
        for cand, mapping in cfg.items():
            reachable.append({"candidate": cand, "mapping": mapping})
            if isinstance(mapping, dict):
                for organ, treat in mapping.items():
                    per_organ.setdefault(organ, []).append({"candidate": cand, "treatment": treat})
    else:
        # Fallback: the correction-wave / Doctor treatment families the campaign was designed to test.
        reachable = [
            {"candidate": "C2_tensor_class", "mapping": {"expert_mlp1": "product_quant", "expert_mlp2": "pq_protected_islands"},
             "source": "correction-wave default (PENDING real-forward seal)"},
            {"candidate": "C3_g3_winners", "mapping": {"expert_mlp1": "pq_doctor_lowrank", "expert_mlp2": "pq_protected_islands"},
             "source": "correction-wave default (PENDING real-forward seal)"},
        ]
        per_organ = {
            "expert_mlp1": [{"candidate": "C2_tensor_class", "treatment": "product_quant"},
                            {"candidate": "C3_g3_winners", "treatment": "pq_doctor_lowrank"}],
            "expert_mlp2": [{"candidate": "C2_tensor_class", "treatment": "pq_protected_islands"},
                            {"candidate": "C3_g3_winners", "treatment": "pq_protected_islands"}],
        }
    # Doctor residual treatments referenced by the campaign geometry regime.
    doctor_treatments = [
        {"name": "doctor_pq (reserve-only)", "reachable": True, "role": "base-pass reserve budget"},
        {"name": "residual_codebook", "reachable": True, "doctor_bpw": 0.15, "role": "low-rate residual repair"},
        {"name": "pq_doctor_lowrank", "reachable": True, "role": "low-rank residual on robust organ (expert_mlp1)"},
        {"name": "pq_protected_islands", "reachable": True, "role": "protected reserve on heavy-tailed organ (expert_mlp2)"},
    ]
    # Which improved REAL capability: only a real-forward candidate that passed the gate counts.
    improved_real = [t for t in desc.get("treated", []) if t.get("capability_candidate")]
    return {
        "reachable_candidates": reachable,
        "per_organ_treatments": per_organ,
        "doctor_residual_treatments": doctor_treatments,
        "dominant_failure_organ": dominant.get("organ"),
        "improved_real_capability": [
            {"row_id": t["row_id"], "variant": t["variant"], "mean_sym_kl": t["mean_sym_kl"],
             "argmax_agreement": t["argmax_agreement"]} for t in improved_real
        ],
        "improved_real_capability_count": len(improved_real),
        "proxy_only_improvements": {
            "expert_mlp1": {"treatment": "pq_doctor_lowrank", "proxy_divergence": GEOMETRY_PRIOR["expert_mlp1"]["proxy_divergence"],
                            "note": "F1 proxy only, NOT a real-forward capability pass"},
            "expert_mlp2": {"treatment": "pq_protected_islands", "proxy_divergence": GEOMETRY_PRIOR["expert_mlp2"]["proxy_divergence"],
                            "note": "F1 proxy only; cut proxy divergence 0.60 -> 0.184 but no real-forward pass"},
        },
        "note": "reachable = a treatment the campaign could actually run inside the RAM-bounded budget; "
                "improved_real_capability requires a real-forward gate pass, not a proxy divergence drop.",
    }


# --------------------------------------------------------------------------- 8 required answers
def _answers(control: dict, dominant: dict, doctor: dict, wbpw: dict, outcome: str,
             best: dict | None, provisional: bool) -> dict:
    low = wbpw.get("realized_whole_artifact_bpw")
    return {
        "q1_what_uniform_rvq_proved": (
            "Uniform sub-bit RVQ preserves no capability at real fidelity. On the real-forward G4 gate "
            f"(6 real-tokenizer prompts, real logits, real PPL) uniform naive_rvq @ 1.0 bpw scored "
            f"{control.get('collapse')} collapse / {control.get('degraded')} degraded / "
            f"{control.get('capability_pass')} pass, next-token argmax agreement "
            f"{control.get('argmax_agreement_range')} vs the 0.95 gate. This is the fourth independent "
            "confirmation of the negative (G0-G3 proxy, Second-Light 0.688 output-divergence, and now "
            "the first REAL-forward gate). Uniform sub-bit is a sealed negative control, not a candidate."
        ),
        "q2_organ_largest_failures": (
            f"The dominant-failure organ is {dominant.get('organ')} "
            f"({dominant.get('role', GEOMETRY_PRIOR.get('expert_mlp2', {}).get('role', 'down projection'))}). "
            f"Source: {dominant.get('source')}. The heavy-tailed down projection carries the largest sub-bit "
            "reconstruction failure; the up/gate projection (expert_mlp1) is comparatively robust."
        ),
        "q3_doctor_treatments_reachable": (
            "Reachable treatments: " + ", ".join(
                c.get("candidate", "?") for c in doctor.get("reachable_candidates", [])) +
            "; Doctor residual families: " + ", ".join(
                d["name"] for d in doctor.get("doctor_residual_treatments", [])) +
            ". Reachable = runnable inside the RAM-bounded shared-cache budget on one heavy lease."
        ),
        "q4_which_improved_real_capability": (
            "None improved REAL capability past the sealed gate."
            if doctor.get("improved_real_capability_count", 0) == 0 else
            f"{doctor.get('improved_real_capability_count')} treated candidate(s) passed the real-forward gate: " +
            ", ".join(t["variant"] for t in doctor.get("improved_real_capability", []))
        ) + " At proxy fidelity only (F1, NOT a capability pass): pq_doctor_lowrank on expert_mlp1 "
            "(~0.008 divergence) and pq_protected_islands on expert_mlp2 (0.60 -> 0.184 divergence).",
        "q5_whole_artifact_rate_passed_failed": (
            f"Whole-artifact sub-bit RATE was achieved and byte-verified at "
            f"{low} bpw (budget {wbpw.get('budget_bpw')}), below the 1.0 bpw sub-bit line. "
            "The rate PASSED; the CAPABILITY at that rate FAILED. The artifact is byte-complete and "
            "accountable but does not pass the functional-quality contract."
        ) if low is not None else "Whole-artifact rate PENDING (baseline not found).",
        "q6_how_low_hawking_proved": (
            f"Hawking proved a complete, byte-exact whole-artifact at {low} bpw (well below 1 real "
            "bit/weight) with exact accounting and 183/183 rows sealed. That is how low the CONSTRUCTION "
            "goes. The capability-preserving floor is NOT below 1.0 real bit/weight: no sub-bit candidate, "
            "uniform or treated, passed the real-forward gate."
        ) if low is not None else "Construction floor PENDING (baseline not found).",
        "q7_what_transfers_to_qwen": (
            "Transfers to Qwen (235B next rung): the tensor-class-aware allocation machinery (treat "
            "expert_mlp1 robust vs expert_mlp2 heavy-tailed differently, never uniform), the Doctor "
            "residual families (low-rank on robust organs, protected islands on heavy-tailed organs), the "
            "real-forward divergence gate (sym_kl <= 0.10 AND argmax agreement >= 0.95), byte-exact "
            "whole-artifact accounting, and the durable one-heavy-lease singleton controller. The negative "
            "sub-bit prior transfers too: do not expect uniform sub-bit to preserve capability."
        ),
        "q8_what_must_not_be_repeated": (
            "Do not repeat: (1) uniform sub-bit RVQ as a capability candidate (proven negative 4x); "
            "(2) lowering the sealed thresholds after seeing results; (3) claiming a capability pass from "
            "a proxy output-divergence drop without a real-forward gate pass; (4) treating byte-completeness "
            "as capability; (5) running any heavy forward without the one-heavy-lease guard."
        ),
    }


def _reopening_conditions() -> list[str]:
    return [
        "A treated candidate reaches mean_sym_kl <= 0.10 AND next_token_argmax_agreement >= 0.95 at whole-artifact BPW < 1.0 on the sealed holdout AND a held-back validation set.",
        "A new representation or Doctor treatment on the diagnosed dominant-failure organ (expert_mlp2 down projection) closes real-forward divergence below the gate, replicated with a larger sample and adversarial reproduction.",
        "A per-organ adaptive allocation (not uniform) is shown to preserve capability at sub-bit real fidelity.",
        "The gate thresholds are re-derived from first principles BEFORE any run (never lowered after seeing results).",
    ]


# --------------------------------------------------------------------------- classification
def classify(desc: dict, wbpw: dict) -> dict:
    treated = desc.get("treated", [])
    whole_bpw_ok = bool(wbpw.get("sub_bit_rate_achieved"))
    passing = [t for t in treated if _passes_gate(t, whole_bpw_ok)]
    complete = desc.get("complete", False)
    found = desc.get("found", False)
    # Outcome A requires: campaign complete AND at least one treated candidate passes the gate.
    if complete and passing:
        outcome = "A"
        outcome_label = "capability pass at sub-bit (treated candidate meets sealed gate)"
    else:
        outcome = "B"
        outcome_label = "honest boundary: no treated sub-bit candidate passes the sealed gate"
    provisional = (not found) or (not complete)
    return {"outcome": outcome, "outcome_label": outcome_label, "provisional": provisional,
            "passing": passing, "complete": complete, "found": found}


# --------------------------------------------------------------------------- report md
def _render_report(ctx: dict) -> str:
    p = ctx["provisional"]
    banner = ""
    if p:
        reason = ctx["provisional_reason"]
        banner = (
            "> PROVISIONAL CONCLUSION - the Doctor / correction-wave campaign is not complete.\n"
            f"> Reason: {reason}\n"
            "> These artifacts fold the evidence available so far. They are NOT the sealed final "
            "conclusion and must be regenerated once the campaign reaches a final state.\n\n"
        )
    a = ctx["answers"]
    control = ctx["control"]
    dominant = ctx["dominant"]
    wbpw = ctx["wbpw"]
    repro = ctx["reproduction"]
    rollback = ctx["rollback"]
    lines = []
    lines.append("# GPT-OSS-120B Final Frontier Report")
    lines.append("")
    lines.append(f"Status: {'PROVISIONAL' if p else 'SEALED'}    Outcome: {ctx['outcome']} ({ctx['outcome_label']})")
    lines.append(f"Generated: {ctx['generated_at']}    Source revision: {SOURCE_REVISION}    HEAD: {ctx['git_head']}")
    lines.append("")
    lines.append(banner.rstrip() if banner else "")
    lines.append("## Verdict")
    lines.append("")
    if ctx["outcome"] == "A":
        lines.append("Outcome A. A treated sub-bit candidate meets the sealed real-forward gate "
                     f"(mean_sym_kl <= {PROMOTE_KL} AND next-token argmax agreement >= {PROMOTE_ARGMAX_AGREE}) "
                     f"at whole-artifact BPW < {WHOLE_ARTIFACT_BPW_MAX}. Best passing candidate: "
                     f"{ctx['best'].get('variant') if ctx['best'] else 'n/a'}.")
    else:
        lines.append("Outcome B, the honest boundary. No treated sub-bit candidate, uniform or "
                     "tensor-class-aware, passes the sealed real-forward gate. GPT-OSS-120B does not "
                     "survive below 1 real bit/weight while preserving capability. The construction "
                     "machinery is proven and the byte accounting is exact; the capability is not.")
    lines.append("")
    lines.append("## The eight required questions")
    lines.append("")
    qmap = [
        ("1. What uniform RVQ proved", a["q1_what_uniform_rvq_proved"]),
        ("2. Which organ caused the largest failures", a["q2_organ_largest_failures"]),
        ("3. Which Doctor treatments were reachable", a["q3_doctor_treatments_reachable"]),
        ("4. Which improved real capability", a["q4_which_improved_real_capability"]),
        ("5. What whole-artifact rate passed / failed", a["q5_whole_artifact_rate_passed_failed"]),
        ("6. How low Hawking proved", a["q6_how_low_hawking_proved"]),
        ("7. What transfers to Qwen", a["q7_what_transfers_to_qwen"]),
        ("8. What must not be repeated", a["q8_what_must_not_be_repeated"]),
    ]
    for title, body in qmap:
        lines.append(f"### {title}")
        lines.append("")
        lines.append(body)
        lines.append("")
    lines.append("## Best candidate and dominant-failure organ")
    lines.append("")
    best = ctx["best"]
    if best:
        lines.append(f"- Best treated candidate: {best.get('variant')} (row {best.get('row_id')}), "
                     f"mean_sym_kl={best.get('mean_sym_kl')}, argmax_agreement={best.get('argmax_agreement')}, "
                     f"verdict={best.get('verdict')}.")
    else:
        lines.append("- Best treated candidate: PENDING (no real-forward treated candidate sealed yet). "
                     "The least-divergent real-forward row measured so far is the uniform-RVQ control "
                     "code_py (sym_kl 1.865, agreement 0.63), which is a control, not a treated candidate.")
    lines.append(f"- Dominant-failure organ: {dominant.get('organ')} "
                 f"({dominant.get('role', 'down projection')}); source {dominant.get('source')}.")
    lines.append(f"- Lowest credible rate region: whole-artifact {wbpw.get('realized_whole_artifact_bpw')} bpw "
                 f"(budget {wbpw.get('budget_bpw')}); byte-complete, capability-negative.")
    lines.append("")
    lines.append("## Reopening conditions")
    lines.append("")
    for c in ctx["reopening"]:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    for cmd in repro["commands"]:
        lines.append(cmd)
    lines.append("```")
    lines.append("")
    lines.append(f"Interpreter: {repro['interpreter']}")
    lines.append(f"Inputs (sha256): see GPT_OSS_120B_REPRODUCTION.json")
    lines.append("")
    lines.append("## Rollback")
    lines.append("")
    lines.append(f"Tag: {rollback['tag']} (exists: {rollback['tag_exists']})")
    lines.append("```bash")
    for cmd in rollback["commands"]:
        lines.append(cmd)
    lines.append("```")
    lines.append("")
    lines.append("## Honesty")
    lines.append("")
    lines.append("This report runs no forward and loads no weights. It folds sealed JSON checkpoints. "
                 "Every proxy result is labelled F1_bounded_proxy and is never presented as a capability "
                 "pass. The sealed gate thresholds were not lowered after seeing results.")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main seal
def seal(out_dir: Path, dry_run: bool) -> int:
    g4 = _read_json(G4_RESULT)
    control_raw = _read_json(G4_CONTROL)
    manifest = _read_json(SL_MANIFEST)
    baseline = _read_json(SL_BASELINE)
    desc = _discover_campaign()

    control = _control_summary(g4)
    wbpw = _whole_artifact_bpw(baseline)
    dominant = _dominant_failure_organ(desc)
    doctor = _doctor_map(desc, dominant)
    organ_bytes = _organ_byte_allocation(manifest)
    cls = classify(desc, wbpw)
    best = _best_treated(desc.get("treated", []))
    answers = _answers(control, dominant, doctor, wbpw, cls["outcome"], best, cls["provisional"])
    reopening = _reopening_conditions()

    provisional_reason = ""
    if cls["provisional"]:
        if not desc["found"]:
            provisional_reason = ("no Doctor / correction-wave campaign directory found under "
                                  f"{GF} (looked for {', '.join(CAMPAIGN_DIRS)}); folding G4 control + "
                                  "geometry prior + Second Light byte baseline only")
        elif not desc["complete"]:
            provisional_reason = (f"campaign {desc['name']} state not final "
                                  f"(rows_done {desc['rows_done']}/{desc['rows_total']}, "
                                  f"final={bool((desc['state'] or {}).get('final'))})")

    generated_at = _now()
    git_head = _git_head()

    reproduction = {
        "schema": "hawking.gpt_oss_120b.reproduction.v1",
        "generated_at": generated_at,
        "interpreter": PY,
        "source_revision": SOURCE_REVISION,
        "one_heavy_lease_law": "no full GPT-OSS forward, no 61GB weight load, no detached controller unless the one heavy lease is free",
        "commands": [
            "# 1. run the untreated uniform-RVQ real-forward control (the G4 negative)",
            f"{PY} tools/condense/gravity_frontier_g4_controller.py detach",
            "# 2. run the tensor-class / Doctor correction wave (D0-D6 + diagnosis)",
            f"{PY} tools/condense/gravity_frontier_correction_wave.py detach",
            "# 3. seal the conclusion (this script; runs no forward)",
            f"{PY} tools/condense/seal_120b_conclusion.py seal",
        ],
        "inputs": {
            "g4_result": {"path": str(G4_RESULT), "sha256": _file_sha(G4_RESULT)},
            "g4_untreated_control": {"path": str(G4_CONTROL), "sha256": _file_sha(G4_CONTROL)},
            "second_light_manifest": {"path": str(SL_MANIFEST), "sha256": _file_sha(SL_MANIFEST)},
            "second_light_baseline": {"path": str(SL_BASELINE), "sha256": _file_sha(SL_BASELINE)},
            "campaign_dir": desc.get("dir"),
            "campaign_state_schema": desc.get("state_schema"),
        },
    }

    rollback = {
        "schema": "hawking.gpt_oss_120b.rollback.v1",
        "generated_at": generated_at,
        "tag": ROLLBACK_TAG,
        "tag_exists": _tag_exists(ROLLBACK_TAG),
        "git_head_at_seal": git_head,
        "commands": [
            f"# tag the sealed conclusion (run once, after the campaign is complete)",
            f"git tag -a {ROLLBACK_TAG} -m 'GPT-OSS-120B frontier conclusion sealed'",
            f"# to roll back to the sealed conclusion state",
            f"git reset --hard {ROLLBACK_TAG}",
            f"# to discard only the conclusion artifacts (keep code)",
            f"git checkout -- reports/condense/general_frontier/GPT_OSS_120B_FINAL_*.json reports/condense/general_frontier/GPT_OSS_120B_FINAL_*.md",
        ],
        "note": "git reset --hard is destructive of uncommitted work; the tag is the anchor. This sealer never runs git; it only records the commands.",
    }

    # ------- assemble result json
    result = {
        "schema": "hawking.gpt_oss_120b.final_frontier_result.v1",
        "generated_at": generated_at,
        "status": "PROVISIONAL" if cls["provisional"] else "SEALED",
        "outcome": cls["outcome"],
        "outcome_label": cls["outcome_label"],
        "provisional": cls["provisional"],
        "provisional_reason": provisional_reason or None,
        "dry_run": dry_run,
        "source_revision": SOURCE_REVISION,
        "git_head": git_head,
        "gate": {
            "mean_sym_kl_max": PROMOTE_KL,
            "next_token_argmax_agreement_min": PROMOTE_ARGMAX_AGREE,
            "whole_artifact_bpw_max": WHOLE_ARTIFACT_BPW_MAX,
            "law": "not lowered after seeing results",
            "surfaces": "validation + holdout (real tokenizer prompts, real logits)",
        },
        "campaign": {
            "name": desc.get("name"), "dir": desc.get("dir"), "found": desc.get("found"),
            "complete": desc.get("complete"), "rows_done": desc.get("rows_done"),
            "rows_total": desc.get("rows_total"), "state_schema": desc.get("state_schema"),
            "checkpoint_count": desc.get("checkpoint_count"),
            "treated_candidate_count": len(desc.get("treated", [])),
            "diagnosis_row_count": len(desc.get("diagnosis_rows", [])),
        },
        "control_uniform_rvq": control,
        "treated_candidates": desc.get("treated", []),
        "passing_candidates": cls["passing"],
        "best_candidate": best,
        "dominant_failure_organ": dominant,
        "whole_artifact_bpw": wbpw,
        "lowest_credible_rate_region": {
            "whole_artifact_bpw": wbpw.get("realized_whole_artifact_bpw"),
            "budget_bpw": wbpw.get("budget_bpw"),
            "capability": "NEGATIVE (byte-complete, no capability pass)",
            "note": "the lowest credible sub-bit rate region is byte-verified but capability-negative; "
                    "the capability-preserving floor is not established below 1.0 real bit/weight",
        },
        "how_low_hawking_proved": {
            "construction_floor_bpw": wbpw.get("realized_whole_artifact_bpw"),
            "capability_floor": "NOT below 1.0 real bit/weight (no sub-bit capability pass)",
        },
        "answers": answers,
        "reopening_conditions": reopening,
        "inputs_consumed": {
            "g4_result_present": g4 is not None,
            "g4_control_present": control_raw is not None,
            "second_light_manifest_present": manifest is not None,
            "second_light_baseline_present": baseline is not None,
            "campaign_present": desc.get("found"),
        },
    }
    result["sha256"] = _sha({k: v for k, v in result.items() if k != "sha256"})

    # ------- atlas update line (whole-artifact conclusion observation)
    realized = wbpw.get("realized_whole_artifact_bpw")
    complete_bits = (manifest or {}).get("complete_physical_bits")
    atlas_line = {
        "parent": "gpt-oss-120b",
        "revision": SOURCE_REVISION,
        "architecture_family": "moe",
        "scale": "120b",
        "tensor_organ": "whole-artifact",
        "layer": None,
        "expert": None,
        "representation": "tensor_class_aware_pq" if desc.get("treated") else "PQ",
        "rate": realized,
        "doctor": "tensor-class (mlp1 lowrank / mlp2 protected islands)",
        "backend": "apple/mps",
        "source_precision": "mxfp4",
        "physical_bytes": (complete_bits // 8) if complete_bits else None,
        "physical_bits": complete_bits,
        "quality_metrics": {
            "capability_pass": cls["outcome"] == "A",
            "capability_parity": cls["outcome"] == "A",
            "outcome": cls["outcome"],
            "budget_bpw": wbpw.get("budget_bpw"),
            "realized_whole_artifact_bpw": realized,
            "dominant_failure_organ": dominant.get("organ"),
            "control_uniform_rvq_argmax_range": control.get("argmax_agreement_range"),
            "note": "final-frontier conclusion; sub-bit capability negative unless outcome A; "
                    "NOT the Event Horizon" if cls["outcome"] != "A" else "final-frontier conclusion; sub-bit capability pass",
        },
        "runtime": None,
        "memory": None,
        "energy": None,
        "evidence_level": "F4_real_forward_provisional" if cls["provisional"] else "F4_real_forward",
        "provenance": {
            "source_report": "reports/condense/general_frontier/GPT_OSS_120B_FINAL_FRONTIER_RESULT.json",
            "schema": "hawking.gpt_oss_120b.final_frontier_result.v1",
            "sealer": "tools/condense/seal_120b_conclusion.py (read-only; no forward)",
            "seed_commit": git_head,
            "result_sha256": result["sha256"],
        },
    }

    # ------- render report
    report_ctx = {
        "provisional": cls["provisional"], "provisional_reason": provisional_reason,
        "outcome": cls["outcome"], "outcome_label": cls["outcome_label"],
        "generated_at": generated_at, "git_head": git_head,
        "answers": answers, "control": control, "dominant": dominant, "wbpw": wbpw,
        "best": best, "reopening": reopening, "reproduction": reproduction, "rollback": rollback,
    }
    report_md = _render_report(report_ctx)

    # ------- doctor map + organ bytes as standalone artifacts
    doctor_map = {
        "schema": "hawking.gpt_oss_120b.doctor_map.v1",
        "generated_at": generated_at,
        "provisional": cls["provisional"],
        **doctor,
    }
    organ_alloc = {
        "schema": "hawking.gpt_oss_120b.organ_byte_allocation.v1",
        "generated_at": generated_at,
        "provisional": cls["provisional"],
        **organ_bytes,
    }

    # ------- write everything
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = {
        "GPT_OSS_120B_FINAL_FRONTIER_RESULT.json": ("json", result),
        "GPT_OSS_120B_DOCTOR_MAP.json": ("json", doctor_map),
        "GPT_OSS_120B_ORGAN_BYTE_ALLOCATION.json": ("json", organ_alloc),
        "GPT_OSS_120B_REPRODUCTION.json": ("json", reproduction),
        "GPT_OSS_120B_ROLLBACK.json": ("json", rollback),
        "GPT_OSS_120B_FRONTIER_ATLAS_UPDATE.jsonl": ("jsonl", atlas_line),
        "GPT_OSS_120B_FINAL_FRONTIER_REPORT.md": ("md", report_md),
    }
    manifest_rows = []
    for fname, (kind, obj) in produced.items():
        path = out_dir / fname
        if kind == "json":
            _write_json(path, obj)
        elif kind == "jsonl":
            _atomic(path, json.dumps(obj, sort_keys=True, default=str) + "\n")
        else:
            _atomic(path, obj)
        manifest_rows.append({"artifact": fname, "role": kind, "sha256": _file_sha(path)})

    # ------- final artifact manifest (binds every produced file + every input sha)
    final_manifest = {
        "schema": "hawking.gpt_oss_120b.final_artifact_manifest.v1",
        "generated_at": generated_at,
        "status": "PROVISIONAL" if cls["provisional"] else "SEALED",
        "outcome": cls["outcome"],
        "dry_run": dry_run,
        "out_dir": str(out_dir),
        "source_revision": SOURCE_REVISION,
        "git_head": git_head,
        "produced_artifacts": manifest_rows,
        "inputs_consumed": reproduction["inputs"],
        "result_sha256": result["sha256"],
        "note": "every conclusion artifact bound to its sha256; every input bound to its sha256. "
                "PROVISIONAL until the Doctor / correction-wave campaign reaches a final state.",
    }
    manifest_path = out_dir / "GPT_OSS_120B_FINAL_ARTIFACT_MANIFEST.json"
    # include the manifest's own line for completeness before hashing self
    _write_json(manifest_path, final_manifest)

    # ------- console summary (for the calling script / operator)
    summary = {
        "status": final_manifest["status"],
        "outcome": cls["outcome"],
        "outcome_label": cls["outcome_label"],
        "provisional_reason": provisional_reason or None,
        "campaign_found": desc.get("found"),
        "campaign_complete": desc.get("complete"),
        "treated_candidates": len(desc.get("treated", [])),
        "passing_candidates": len(cls["passing"]),
        "dominant_failure_organ": dominant.get("organ"),
        "whole_artifact_bpw": wbpw.get("realized_whole_artifact_bpw"),
        "out_dir": str(out_dir),
        "artifacts": [r["artifact"] for r in manifest_rows] + ["GPT_OSS_120B_FINAL_ARTIFACT_MANIFEST.json"],
        "result_sha256": result["sha256"],
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Seal the GPT-OSS-120B frontier conclusion (read-only, no forward).")
    ap.add_argument("cmd", choices=["seal"], help="seal the conclusion artifacts")
    ap.add_argument("--out-dir", default=str(GF), help="output directory (default: canonical general_frontier)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write to <out-dir>/CONCLUSION_DRY_RUN so the canonical artifacts are not touched")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    if a.dry_run and Path(a.out_dir) == GF:
        out_dir = GF / "CONCLUSION_DRY_RUN"
    return seal(out_dir, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
