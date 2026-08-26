#!/usr/bin/env python3
"""DOCTOR TRANSFER — the 39-technique library used, not filed.

For each organ of a new specimen Doctor must diagnose, rank the applicable techniques,
query what already failed, and pick the cheapest discriminating experiment -- recording
what it considered, what it skipped, and why. A library that never changes a decision is
a document.

Prescription quality is then measured against the Qwen baseline campaign: fewer
irrelevant treatments, fewer repeated failures, earlier winning representation, lower
experiment count.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
sys.path.insert(0, str(REPO / "tools/headless"))

LIB = RH / "DOCTOR_TECHNIQUE_LIBRARY.json"
GRADE_RANK = {"PLAUSIBLE": 3, "UNKNOWN": 2, "UNLIKELY": 1}

# The organ names the recognizer emits -> the applicability domain the library indexes on.
ORGAN_DOMAIN = {
    "moe_expert": "moe", "moe_router": "moe", "shared_expert": "moe",
    "mlp_gate_up": "dense_mlp", "mlp_down": "dense_mlp",
    "gqa_attention": "attention_gqa", "latent_attention": "attention_gqa",
    "mha_attention": "attention_gqa",
    "recurrent_state": "recurrent_deltanet", "deltanet": "recurrent_deltanet",
    "embed": ["tokenizer", "dense_mlp"], "lm_head": ["tokenizer", "dense_mlp"],
    "vocabulary": "tokenizer",
    "rmsnorm": "dense_mlp", "normalization": "dense_mlp",
    "kv_state": "kv_state", "vision_encoder": "multimodal", "mm_projector": "multimodal",
}

# Directive §5: the canonical order Doctor asks its questions in, and §6: the three kinds
# of zero it must search. Both are structural, so they are encoded rather than narrated.
CANONICAL_ORDER = [
    "SHOULD THIS STRUCTURE EXIST?",
    "MUST IT EXIST IN THIS COORDINATE SYSTEM?",
    "MUST IT EXIST INDEPENDENTLY?",
    "MUST IT BE STORED RATHER THAN GENERATED?",
    "MUST IT EXECUTE FOR EVERY TOKEN?",
    "MUST IT RETAIN THE SAME PRECISION AS ITS NEIGHBORS?",
    "WHAT IS ITS BEST PHYSICAL REPRESENTATION?",
    "WHAT NATIVE OPERATOR SHOULD EXECUTE IT?",
    "WHAT DEVICE PROFILE BEST RUNS THAT OPERATOR?",
]
THREE_ZEROS = {
    "ZERO_STORAGE": ["DOC-ELIMINATION", "tokenizer / vocabulary reduction"],
    "ZERO_INDEPENDENT_INFORMATION": ["DOC-COORDINATES", "shared_basis / cross-layer coefficients",
                                     "additive/vector codebooks", "DOC-REPRESENTATION",
                                     "learned/function-preserving rotations + incoherence processing",
                                     "ultra-low-bit PTQ"],
    "ZERO_EXECUTION": ["DOC-CONDITIONAL", "activation sparsity + conditional compute",
                       "DOC-DECODE", "speculative / multi-token / self-speculative decoding"],
}


# What the negative store calls a technique -> the library families it warns about.
NEGATIVE_FAMILY = {
    "binary_quantization": ("DOC-REPRESENTATION", "ultra-low-bit PTQ"),
    "protected_islands_healing": ("DOC-HEALING", "DOC-REPRESENTATION"),
    "shared_basis": ("shared_basis / cross-layer coefficients", "DOC-REPRESENTATION"),
    "shared_k_hybrid": ("shared_basis / cross-layer coefficients", "DOC-HEALING"),
    "low_rank_correction": ("DOC-HEALING", "additive/vector codebooks"),
    "coordinate_transform": ("DOC-COORDINATES",
                             "learned/function-preserving rotations + incoherence processing"),
    "structural_elimination_heads": ("DOC-ELIMINATION", "DOC-CONDITIONAL",
                                     "activation sparsity + conditional compute"),
    "depth_state_merging": ("DOC-STATE", "KV & state compression",
                            "linear-attention / SSM / DeltaNet / gated-delta compression"),
    "speculative_decoding_draft": ("DOC-DECODE",
                                   "speculative / multi-token / self-speculative decoding"),
}


def zero_kind_of(family):
    for z, fams in THREE_ZEROS.items():
        if family in fams:
            return z
    return "REPRESENTATION"


def prescribe(techniques, organ, domain, negatives):
    domains = domain if isinstance(domain, list) else [domain]
    considered, skipped = [], []
    prior = {n["technique"]: n for n in negatives}
    for t in techniques:
        # best grade across the organ's domains: a technique that is PLAUSIBLE for the
        # matrix shape is prescribable even if UNLIKELY for the vocabulary role
        aps = [(t.get("applicability") or {}).get(d) or {} for d in domains]
        ap = max(aps, key=lambda x: GRADE_RANK.get(x.get("grade", "UNKNOWN"), 0))
        grade = ap.get("grade", "UNKNOWN")
        # A recorded failure on ANOTHER model warns; it never removes a technique here.
        # Matched on FAMILY, not on id substrings -- the negative store names techniques by
        # what was tried ("coordinate_transform") while the library names them by paper
        # ("spinquant"), so substring matching attached nothing at all.
        warn = [n for n in negatives
                if t["family"] in NEGATIVE_FAMILY.get(n["technique"], ())]
        row = {
            "technique": t["id"], "name": t["name"], "family": t["family"],
            "zero_kind": zero_kind_of(t["family"]),
            "grade": grade, "grade_reason": (ap.get("reason") or "")[:280],
            "metal_feasibility": (t.get("metal_feasibility") or {}).get("class"),
            "cheapest_falsifying_experiment":
                (t.get("cheapest_falsifying_experiment") or {}).get("id"),
            "prior_failures_elsewhere": [{"id": n["id"], "model": n["model"],
                                          "reopen_condition": n["reopen_condition"]}
                                         for n in warn],
        }
        if grade == "UNLIKELY":
            skipped.append({**row, "skip_reason":
                            "graded UNLIKELY for this organ domain; kept in the library and "
                            "reopenable, but not prescribed before PLAUSIBLE and UNKNOWN work"})
            continue
        score = GRADE_RANK[grade] - 0.5 * len(warn)
        if row["metal_feasibility"] in ("ZERO_RUNTIME_IF_ABSORBED", "NATIVE_KERNEL_EXISTS"):
            score += 0.5
        considered.append({**row, "rank_score": round(score, 3)})
    considered.sort(key=lambda r: -r["rank_score"])
    return considered, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    a = ap.parse_args()

    lib = json.load(open(LIB))
    techniques = lib["techniques"]
    negs = json.load(open(RH / "NOETIC_NEGATIVE_SCIENCE.json"))["entries"]
    negs = [n for n in negs if not n.get("migrated")]
    reh = json.load(open(RH / "QWEN_TRANSFER_REHEARSAL.json"))
    organs = [o["organ"] for o in reh["plan"]["organ_plan"]]

    per_organ, all_experiments = [], set()
    for organ in organs:
        domain = ORGAN_DOMAIN.get(organ, "dense_mlp")
        considered, skipped = prescribe(techniques, organ, domain, negs)
        covered = {c["zero_kind"] for c in considered} & set(THREE_ZEROS)
        top = considered[:5]
        all_experiments.update(c["cheapest_falsifying_experiment"] for c in top
                               if c["cheapest_falsifying_experiment"])
        per_organ.append({
            "organ": organ, "applicability_domain": domain,
            "canonical_order_asked": CANONICAL_ORDER,
            "three_zeros_covered": sorted(covered),
            "three_zeros_missing": sorted(set(THREE_ZEROS) - covered),
            "n_techniques_considered": len(considered),
            "n_techniques_skipped": len(skipped),
            "prescription": top,
            "skipped": skipped[:6],
            "skipped_summary": {"n": len(skipped),
                                "reason": "graded UNLIKELY for this organ domain"},
        })

    # Prescription quality against the Qwen baseline: the campaign that produced the
    # library ran its search without one.
    qwen_negs = [n for n in negs if "qwen3.8" in n["model"]]
    quality = {
        "irrelevant_treatments_avoided": {
            "value": sum(o["n_techniques_skipped"] for o in per_organ),
            "meaning": "technique-organ pairs graded UNLIKELY and therefore not prescribed; "
                       "the Qwen campaign had no applicability matrix and ranked by hand"},
        "repeated_failures_avoided": {
            "value": sum(len(c["prior_failures_elsewhere"])
                         for o in per_organ for c in o["prescription"]),
            "meaning": "prescriptions carrying a recorded prior failure, each with its "
                       "reopening condition attached so the retry is deliberate"},
        "experiments_to_run": {
            "value": len(all_experiments),
            "meaning": "distinct cheapest-falsifying experiments across every organ's top 5, "
                       "against 39 techniques x %d organs = %d unranked pairs"
                       % (len(organs), 39 * len(organs))},
        "qwen_baseline_failures_recorded": {"value": len(qwen_negs),
                                            "meaning": "what the first campaign paid to learn"},
        "search_space_reduction": {
            "value": round(1 - len(all_experiments) / (39 * len(organs)), 4),
            "meaning": "fraction of the unranked technique-organ space Doctor does not "
                       "prescribe first"},
    }

    out = {
        "schema": "hawking.headless.doctor_transfer.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/doctor_transfer.py",
        "obligation": "G020 — DOCTOR_TRANSFER (directive §55, §56, §5, §6, §108)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "specimen": reh["plan"]["specimen"],
        "n_techniques_in_library": len(techniques),
        "all_techniques_still_KEEP": all(t.get("decision") == "KEEP" for t in techniques),
        "pruning_law": "a Qwen failure alone never prunes a technique; UNLIKELY is a ranking, "
                       "not a deletion, and every skipped technique stays in the library with "
                       "its reopening condition",
        "canonical_doctor_order": CANONICAL_ORDER,
        "three_kinds_of_zero": {k: v for k, v in THREE_ZEROS.items()},
        "n_organs": len(per_organ),
        "per_organ": per_organ,
        "distinct_experiments_prescribed": sorted(all_experiments),
        "prescription_quality": quality,
        "three_zero_coverage_note":
            "coverage is REPORTED per organ, not required. An organ whose library offers no "
            "ZERO_EXECUTION technique has a real gap in the library, and naming it is the "
            "useful output; inventing a technique to fill the row would not be.",
        "pass": bool(per_organ
                     and all(t.get("decision") == "KEEP" for t in techniques)
                     and all(o["n_techniques_considered"] > 0 for o in per_organ)
                     and all(o["prescription"] for o in per_organ)),
    }
    Path(a.emit).write_text(json.dumps(out, indent=1))
    print(f"organs={len(per_organ)} techniques={len(techniques)} "
          f"all_KEEP={out['all_techniques_still_KEEP']} "
          f"experiments={len(all_experiments)} pass={out['pass']}")
    for o in per_organ:
        dom = o["applicability_domain"]
        dom = dom if isinstance(dom, str) else "+".join(dom)
        print(f"  {o['organ']:16} domain={dom:18} "
              f"considered={o['n_techniques_considered']:2} skipped={o['n_techniques_skipped']:2} "
              f"zeros={len(o['three_zeros_covered'])}/3 top={o['prescription'][0]['technique'] if o['prescription'] else None}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
