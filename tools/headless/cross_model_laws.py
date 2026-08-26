#!/usr/bin/env python3
"""Classify every Qwen law by how far the evidence actually reaches.

Four levels (directive §91): QWEN_SPECIFIC, FAMILY_TRANSFERRED, ARCHITECTURE_GENERAL,
MACHINE_GENERAL. Promotion above QWEN_SPECIFIC requires independent measurements on
distinct models, and the store refuses a promotion that does not have them -- one
textbook is never enough for a universal law.

Both models here are Qwen, so nothing can honestly reach ARCHITECTURE_GENERAL on
measurement count alone; entries that look architecture-general are marked as candidates
with the third model named as the condition.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
LEVELS = ["QWEN_SPECIFIC", "FAMILY_TRANSFERRED", "ARCHITECTURE_GENERAL", "MACHINE_GENERAL"]
MIN_MODELS = {"QWEN_SPECIFIC": 1, "FAMILY_TRANSFERRED": 2, "ARCHITECTURE_GENERAL": 2,
              "MACHINE_GENERAL": 1}
# Distinct architecture families required before ARCHITECTURE_GENERAL is honest.
MIN_FAMILIES = {"QWEN_SPECIFIC": 1, "FAMILY_TRANSFERRED": 1, "ARCHITECTURE_GENERAL": 2,
                "MACHINE_GENERAL": 1}
FAMILY_OF = {"qwen3.8-27b-abliterated": "qwen", "Qwen/Qwen3-30B-A3B": "qwen",
             "tiiuae/Falcon-H1-7B-Instruct": "falcon_h1"}


class Refused(Exception):
    pass


def resolve(cit):
    rel, _, jp = cit.partition("#")
    f = REPO / rel
    if not f.exists():
        raise Refused(f"missing receipt {rel}")
    if not jp:
        return True
    cur = json.load(open(f))
    for part in jp.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise Refused(f"{cit}: no key {part}")
    return cur


def validate(e):
    for f in ("law", "level", "measured_on_models", "evidence", "why_not_higher"):
        if not e.get(f):
            raise Refused(f"{e.get('id')}: missing {f}")
    if e["level"] not in LEVELS:
        raise Refused(f"{e['id']}: bad level {e['level']}")
    models = set(e["measured_on_models"])
    fams = {FAMILY_OF.get(m, "unknown") for m in models}
    if e["level"] != "MACHINE_GENERAL":
        if len(models) < MIN_MODELS[e["level"]]:
            raise Refused(f"{e['id']}: level {e['level']} needs "
                          f"{MIN_MODELS[e['level']]} models, has {len(models)}")
        if len(fams) < MIN_FAMILIES[e["level"]]:
            raise Refused(f"{e['id']}: level {e['level']} needs {MIN_FAMILIES[e['level']]} "
                          f"distinct architecture families, has {sorted(fams)}")
    for c in e["evidence"]:
        resolve(c)
    return e


LAWS = [
    dict(id="LAW-MLP-FLOOR-2.25",
         law="The coherent MLP information floor is 2.25 bpw.",
         level="QWEN_SPECIFIC",
         measured_on_models=["qwen3.8-27b-abliterated"],
         refuted_on_models=["Qwen/Qwen3-30B-A3B"],
         why_not_higher="It does not survive the very first jump inside its own family. On "
                        "Qwen3-30B-A3B moe_expert the best candidate at 2.25 bpw reaches "
                        "held-out rel_fro 0.498769, nowhere near usable, and even 3.5 bpw "
                        "only reaches 0.238596. The value is a property of that organ in "
                        "that model, not of Qwen and not of dense MLPs.",
         evidence=["receipts/headless/ORGAN_FRONTIER_MATRIX.json#n_measured",
                   "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json#matched_bits_comparison.tiers.0.seeded_family_rel_fro",
                   "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json#full_grid.affine_q3_g64.mean_held_out_rel_fro"]),
    dict(id="LAW-FITTED-AFFINE-BEATS-RTN",
         law="At matched bits per weight, a least-squares-refit affine codec beats generic "
             "grouped-absmax round-to-nearest.",
         level="ARCHITECTURE_GENERAL",
         measured_on_models=["qwen3.8-27b-abliterated", "Qwen/Qwen3-30B-A3B",
                             "tiiuae/Falcon-H1-7B-Instruct"],
         why_not_higher="ARCHITECTURE_GENERAL is the ceiling for a claim about weights; "
                        "MACHINE_GENERAL is reserved for properties of the box. Promoted "
                        "here because the reopening condition this law carried has been "
                        "MET: it was measured on a second architecture family. On "
                        "Falcon-H1-7B -- hybrid mamba+attention, a different vendor and a "
                        "different backbone -- the seeded family wins 3 of 3 matched-bit "
                        "tiers, and the advantage is nearly the same size as on the Qwen "
                        "MoE: 1.828x vs 1.899x at 2.25 bpw, 1.327x vs 1.333x at 3.25, "
                        "1.122x vs 1.138x at 4.25.",
         evidence=["receipts/headless/ODYSSEY_TRANSFER_PROVEN.json#matched_bits_comparison.n_tiers_seeded_wins",
                   "receipts/headless/MATCHED_BITS_FALCON_H1.json#n_tiers_seeded_wins",
                   "receipts/headless/MATCHED_BITS_FALCON_H1.json#law_holds_here",
                   "receipts/headless/MATCHED_BITS_FALCON_H1.json#specimen.architecture_family"]),
    dict(id="LAW-HELDOUT-REAL-ACTIVATIONS",
         law="Rank COMPOSITION choices on held-out real activations. Weight-space error "
             "ranks a codec against a single tensor triple correctly, and ranks the "
             "whole-model consequence of a representation change BACKWARDS.",
         level="FAMILY_TRANSFERRED",
         measured_on_models=["qwen3.8-27b-abliterated", "Qwen/Qwen3-30B-A3B"],
         refined_by=["tiiuae/Falcon-H1-7B-Instruct"],
         why_not_higher=(
             "A Falcon-H1 measurement was taken specifically to promote this and it "
             "REFUSED to: on that dense MLP, weight-space and held-out-activation rankings "
             "are IDENTICAL across all 13 candidates with zero matched-tier winner flips. "
             "So the law is not that weight-space always misranks -- it does not. It is "
             "narrower and sharper than it was written: weight-space error is adequate for "
             "ranking codecs on one tensor triple, and it inverted the answer on the "
             "question that actually mattered, which was WHICH REPRESENTATION CHANGE BREAKS "
             "THE COMPOSED MODEL. There weight-space said the non-MLP drop was 6.2x larger "
             "and therefore dominant; the physical measurement priced the MLP change at 19 "
             "capability points against the non-MLP change's 8. Promotion needs a second "
             "family measured on the COMPOSITION question, not on the codec question."),
         evidence=["receipts/headless/ODYSSEY_TRANSFER_PROVEN.json#activations.real_not_synthetic",
                   "receipts/headless/MATCHED_BITS_FALCON_H1.json#weight_space_vs_activations.orderings_identical",
                   "receipts/headless/COMPOSITION_ATTRIBUTION.json#weight_space_prediction_was_wrong",
                   "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json#counts.total"]),
    dict(id="LAW-PER-ORGAN-FLOOR",
         law="Measure the information floor per ORGAN, never per model. Organs in one model "
             "do not share a floor.",
         level="FAMILY_TRANSFERRED",
         measured_on_models=["qwen3.8-27b-abliterated", "Qwen/Qwen3-30B-A3B"],
         why_not_higher="Two Qwen models. On the first, attention/DeltaNet/embedding survive "
                        "q3_g128 and fail at q2f_g64 while the MLP survives it; on the "
                        "second, the MoE expert floor is far above the dense MLP's. Same "
                        "conclusion twice, same family both times.",
         evidence=["receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.gqa_attention.candidates.5.complete_ebpw",
                   "receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.gqa_attention.candidates.6.complete_ebpw",
                   "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json#pareto_frontier"]),
    dict(id="LAW-MOE-EXPERTS-ARE-ORTHOGONAL",
         law="MoE experts within a layer are mutually near-orthogonal, so there is no cheap "
             "shared substrate to factor out of an expert family.",
         level="QWEN_SPECIFIC",
         measured_on_models=["Qwen/Qwen3-30B-A3B"],
         why_not_higher="Measured on exactly one MoE with a receipt in this repository. A "
                        "similar observation on DeepSeek V4-Flash exists in operator memory "
                        "but has no receipt here, and a model claim is not evidence, so it "
                        "does not count toward promotion. A non-Qwen MoE (GLM-4.5-Air and "
                        "Kimi-VL-A3B are both in the queue) is the condition.",
         evidence=["receipts/headless/EXPERT_FAMILY_GENOME.json#expert_similarity.gate_proj.mean_cosine",
                   "receipts/headless/EXPERT_FAMILY_GENOME.json#noetic_hypothesis_test.mean_residual_across_tensors",
                   "receipts/headless/EXPERT_FAMILY_GENOME.json#shared_subspace.gate_proj.energy_in_top8"]),
    dict(id="LAW-DEVICE-ROOFS",
         law="Device theoretical roof 819.0 GB/s and device measured sustained roof 778.8 "
             "GB/s on this machine.",
         level="MACHINE_GENERAL",
         measured_on_models=["qwen3.8-27b-abliterated"],
         why_not_higher="Already at the top level for what it describes: these are properties "
                        "of the box, not of any model. They are valid only on this machine.",
         evidence=["receipts/headless/BANDWIDTH_ROOF.json#anchor_roof.correction.new_roof_gb_s"]),
    dict(id="LAW-MODEL-REACHABLE-ROOF",
         law="The model-reachable bandwidth roof is 690.8 GB/s.",
         level="QWEN_SPECIFIC",
         measured_on_models=["qwen3.8-27b-abliterated"],
         why_not_higher="It is a property of one executable in one regime. The directive "
                        "forbids copying it into another model, and the transfer rehearsal "
                        "asserts it appears nowhere in a new specimen's seeded genome.",
         evidence=["receipts/headless/WHOLE_MODEL_NATIVE.json#three_roofs.MODEL_REACHABLE",
                   "receipts/headless/QWEN_TRANSFER_REHEARSAL.json#plan.device_genome_init.seeded_from"]),
    dict(id="LAW-COMPETENT-KERNEL-FIRST",
         law="A representation evaluated with an incompetent kernel is not evaluated. Fewer "
             "stored bits is not fewer nanoseconds.",
         level="QWEN_SPECIFIC",
         measured_on_models=["qwen3.8-27b-abliterated"],
         why_not_higher="Only one model has had its kernels profiled on this device. The "
                        "second specimen has no native kernel yet, so the law has one "
                        "measurement.",
         evidence=["receipts/headless/SHARED_BASIS_KERNEL.json#finding.reason",
                   "receipts/headless/KERNEL_LIBRARY.json#n_complete"]),
    dict(id="LAW-HEAD-ORTHOGONALITY-BLOCKS-ELIMINATION",
         law="Attention head sharing is refuted where heads are near-orthogonal.",
         level="QWEN_SPECIFIC",
         measured_on_models=["qwen3.8-27b-abliterated"],
         why_not_higher="One measurement, and the law is explicitly conditional on the "
                        "measured geometry rather than on the technique. Any model with high "
                        "head cosine reopens it.",
         evidence=["receipts/headless/STRUCTURAL_ELIMINATION.json#attention_heads.headline.q_mean_cosine_all_layers"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    ap.add_argument("--try-promote", nargs=2, metavar=("ID", "LEVEL"))
    a = ap.parse_args()

    if a.try_promote:
        lid, lvl = a.try_promote
        e = next((x for x in LAWS if x["id"] == lid), None)
        if not e:
            print("no such law", lid)
            return 2
        try:
            validate({**e, "level": lvl})
            print(f"PROMOTED {lid} to {lvl}")
            return 0
        except Refused as r:
            print(f"PROMOTION REFUSED: {r}")
            return 1

    laws, rejected = [], []
    for e in LAWS:
        try:
            laws.append(validate(e))
        except Refused as r:
            rejected.append(str(r))
    counts = {l: sum(1 for e in laws if e["level"] == l) for l in LEVELS}
    out = {
        "schema": "hawking.headless.cross_model_laws.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/cross_model_laws.py",
        "obligation": "G028 — FIRST_CROSS_MODEL_LAWS_SEALED (directive §91)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "models_measured": ["qwen3.8-27b-abliterated", "Qwen/Qwen3-30B-A3B"],
        "architecture_families_measured": ["qwen", "falcon_h1"],
        "promotion_rule": {"min_models": MIN_MODELS, "min_architecture_families": MIN_FAMILIES,
                           "law": "no universal law from one textbook; ARCHITECTURE_GENERAL "
                                  "additionally needs two DISTINCT architecture families"},
        "counts": counts, "n_laws": len(laws), "n_rejected": len(rejected),
        "rejected": rejected, "laws": laws,
        "pass": bool(laws and not rejected),
    }
    Path(a.emit).write_text(json.dumps(out, indent=1))
    print(json.dumps(counts, indent=1))
    for e in laws:
        print(f"  {e['level']:20} {e['id']}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
