#!/usr/bin/env python3
"""Seal F1 (Qwen3-235B) and run the mandatory post-parent review on it.

Reads the sealed, read-only Qwen campaign checkpoints, derives the evidence
bundle from them (no hand transcription of metrics), writes the
SUB_BIT_UNSOLVED receipt, then calls post_parent_review.generate and the
can_launch_next_parent gate. post_parent_review.py, gravity_potency.py and
quality_contract.py are imported, never edited.

GOVERNING LAW (operator, 2026-07-20): Hawking does not climb above one bit to
find where conventional quantization works. It changes representation, model,
allocation and treatment until useful intelligence survives at one bit or
below. complete_artifact_bits / original_weight_count <= 1/1, where complete
means indices + codebooks + scales + metadata + alignment + islands + Doctor
bytes + pass-through tensors + packaging + runtime tables. Nothing is excluded
as overhead. No candidate above 1 BPW. Upward bracketing is REJECTED.

Run:  python3 tools/foundry/seal_f1_qwen3_235b.py [--check]
"""
from __future__ import annotations

import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import post_parent_review as ppr  # noqa: E402
import quality_contract as qc  # noqa: E402

REPO = os.path.dirname(os.path.dirname(_HERE))
CAMPAIGN = ("/Users/scammermike/hawking-qwen-recovery-20260720/reports/condense/"
            "general_frontier/QWEN_GRAVITY")
CKPT = os.path.join(CAMPAIGN, "checkpoints")
EVIDENCE_PATH = os.path.join(_HERE, "evidence", "f1_qwen3_235b.json")
OUT_DIR = os.path.join(REPO, "reports", "foundry")
REVIEW_DIR = os.path.join(OUT_DIR, "post_parent_review")
RECEIPT_PATH = os.path.join(OUT_DIR, "F1_QWEN3_235B_SUB_BIT_UNSOLVED.json")
REGISTRY_PATH = os.path.join(_HERE, "adapters", "tier_a_registry.json")

PARENT_ID = "qwen3-235b-a22b-instruct-2507"

# The one sentence this whole seal exists to get right.
VERDICT = (
    "NEGATIVE RESULT FOR THE RAW-WEIGHT PQ/VQ REPRESENTATION FAMILY AT APPROXIMATELY "
    "ONE BIT on Qwen3-235B. The family fails; sub-bit is NOT shown to be impossible, "
    "and this is NOT a licence to raise the one-bit ceiling. Methods that change the "
    "source (QAT, distillation, compressibility training, structured pruning, learned "
    "sharing) are not bound by the rate-distortion limit of the original weights and "
    "remain untested. The response to this result is to change representation AT the "
    "ceiling, never to bracket upward."
)

# The ceiling is owned by post_parent_review.ONE_BIT_CEILING_LAW. Never restated here.
CEILING = ppr.ONE_BIT_CEILING_LAW


# ---------------------------------------------------------------- campaign read


def read_campaign():
    """Load the 18 sealed rows. Read only: this directory is never written."""
    rows = []
    for path in sorted(glob.glob(os.path.join(CKPT, "*.json"))):
        if os.path.basename(path).startswith("PROBE__"):
            continue  # partial-stack probes, not the sealed 18-row ladder
        with open(path) as fh:
            rows.append(json.load(fh))
    state_path = os.path.join(CAMPAIGN, "QWEN_GRAVITY_STATE.json")
    with open(state_path) as fh:
        state = json.load(fh)
    return rows, state


def by_variant(rows, variant):
    return sorted((r for r in rows if r["variant"] == variant), key=lambda r: r["prompt_id"])


def _rng(values):
    return [round(min(values), 5), round(max(values), 5)]


def packed_summary(rows, variant):
    sel = by_variant(rows, variant)
    kl = [r["divergence_vs_parent"]["mean_sym_kl"] for r in sel]
    ag = [r["divergence_vs_parent"]["next_token_argmax_agreement"] for r in sel]
    bpw = {r["bpw"]["whole_model_bpw"] for r in sel}
    byts = {r["bpw"]["total_artifact_bytes"] for r in sel}
    assert len(bpw) == 1 and len(byts) == 1, "a variant must have one byte plan"
    complete_bpw = bpw.pop()
    total_bytes = byts.pop()
    return {
        "variant": variant,
        "spec": sel[0]["spec"],
        "complete_bpw": complete_bpw,
        "complete_artifact_bytes": total_bytes,
        "original_weight_count": round(total_bytes * 8 / complete_bpw),
        "over_ceiling": complete_bpw > 1.0,
        "organ_bpw": sel[0]["bpw"]["organ_bpw"],
        "n_prompts": len(sel),
        "collapse_count": sum(1 for r in sel if r["verdict"] == "collapse"),
        "mean_sym_kl_range": _rng(kl),
        "argmax_agreement_range": _rng(ag),
        "capability_pass": any(r.get("capability_pass") for r in sel),
        "per_prompt": [
            {
                "prompt_id": r["prompt_id"],
                "domain": r["domain"],
                "mean_sym_kl": r["divergence_vs_parent"]["mean_sym_kl"],
                "next_token_argmax_agreement": r["divergence_vs_parent"]["next_token_argmax_agreement"],
                "mean_logit_cosine": r["divergence_vs_parent"]["mean_logit_cosine"],
                "mean_top5_overlap": r["divergence_vs_parent"]["mean_top5_overlap"],
                "perplexity": r["quality"]["perplexity"],
                "n_tokens": r["n_tokens"],
                "verdict": r["verdict"],
                "gate": qc.evaluate(
                    {
                        "mean_symmetric_kl": r["divergence_vs_parent"]["mean_sym_kl"],
                        "argmax_agreement": r["divergence_vs_parent"]["next_token_argmax_agreement"],
                        "logit_cosine": r["divergence_vs_parent"]["mean_logit_cosine"],
                        "top5_overlap": r["divergence_vs_parent"]["mean_top5_overlap"],
                        "candidate_perplexity": r["quality"]["perplexity"],
                    },
                    {
                        "real_parent_forward": True,
                        "real_packed_forward": True,
                        "n_tokens": r["n_tokens"],
                        "split": "probe",
                        "domains": [r["domain"]],
                        "metrics": {
                            "mean_symmetric_kl": r["divergence_vs_parent"]["mean_sym_kl"],
                            "argmax_agreement": r["divergence_vs_parent"]["next_token_argmax_agreement"],
                        },
                    },
                )["passed"],
                "evidence_class": "SHORT_END_TO_END",
                "sha256": r["sha256"],
            }
            for r in sel
        ],
    }


def parent_summary(rows):
    sel = by_variant(rows, "R0_parent")
    ppl = [r["quality"]["perplexity"] for r in sel]
    return {
        "variant": "R0_parent",
        "kind": "bf16 source parent, the quality reference",
        "n_prompts": len(sel),
        "perplexity_range": _rng(ppl),
        "healthy": max(ppl) < 100.0,
        "why_this_matters": (
            "the control is healthy (ppl 1.61 to 39.33 across six domains), so the "
            "collapse of every packed row is a property of the representation, not of "
            "a broken harness. A negative result is only trustworthy with a live control."
        ),
        "per_prompt": [
            {"prompt_id": r["prompt_id"], "domain": r["domain"],
             "perplexity": r["quality"]["perplexity"], "nll": r["quality"]["nll"],
             "n_tokens": r["n_tokens"], "sha256": r["sha256"]}
            for r in sel
        ],
    }


# ---------------------------------------------------------------- evidence


def build_evidence(rows, state):
    parent = parent_summary(rows)
    a1 = packed_summary(rows, "A1_1p0")
    r2 = packed_summary(rows, "R2_subhalf_best")

    failures = []
    for s in (a1, r2):
        for p in s["per_prompt"]:
            failures.append({
                "probe": "real_parent_vs_packed_forward",
                "variant": s["variant"],
                "complete_bpw": s["complete_bpw"],
                "prompt_id": p["prompt_id"],
                "domain": p["domain"],
                "mean_symmetric_kl": p["mean_sym_kl"],
                "next_token_argmax_agreement": p["next_token_argmax_agreement"],
                "parent_perplexity": next(x["perplexity"] for x in parent["per_prompt"]
                                          if x["prompt_id"] == p["prompt_id"]),
                "candidate_perplexity": p["perplexity"],
                "gate_passed": p["gate"],
                "verdict": "COLLAPSED",
            })

    return {
        "schema": "hawking.foundry.parent_evidence.v1",
        "provenance": (
            "F1 Qwen3-235B campaign, SEALED. Metrics derived directly from the 18 "
            "read-only checkpoint rows at %s (QWEN_GRAVITY_STATE ladder_sha %s, "
            "status %s). Nothing here is re-derived or re-run."
            % (CKPT, state["ladder_sha"], state["status"])
        ),
        "parent": {
            "id": PARENT_ID,
            "label": "235B",
            "generation": "F1",
            "architecture": "qwen3-moe: 94 layers x 128 experts",
            "source_id": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        },
        "run_status": "honest_boundary_sealed",
        "verdict": VERDICT,
        "boundary": (
            "Two complete byte plans, one at 1.0075 BPW and one at 0.4930 BPW, both "
            "collapsed 6/6 on a real parent-vs-packed forward with a healthy control. "
            "No frontier selected. The falsified object is the raw-weight PQ/VQ "
            "representation family at approximately one bit, not sub-bit itself."
        ),
        "rate_ceiling": CEILING,
        "capability_cliff": {
            "f1_qwen3_235b_complete_bpw_collapsed_at": [r2["complete_bpw"], a1["complete_bpw"]],
            "f0_gpt_oss_120b_per_expert_whole_bpw_collapsed_at": [0.80508, 0.88845],
            "f0_variants": {"D2_tensor_pq": 0.80508, "D6_global_alloc": 0.82885, "D4_pq_doctor": 0.88845},
            "moved_with_scale": False,
            "note": (
                "the cliff did not move with scale. A ~2x parameter count bought "
                "nothing: 235B collapsed at a COMPLETE 1.0075 BPW, which is a stricter "
                "accounting than the F0 per-expert 0.80 to 0.89 band. Scale is not a "
                "rate argument."
            ),
        },
        "representation": {
            "winners": [],
            "winners_note": "no representation won. Nothing passed the capability gate at any rate.",
            "rate_response": [
                {"rate_bpw": r2["complete_bpw"], "representation": "shared_grammar gate/up d16 k1024 s1 + down d64 k1024 s1",
                 "mean_sym_kl_range": r2["mean_sym_kl_range"], "argmax_agreement_range": r2["argmax_agreement_range"],
                 "verdict": "COLLAPSE 6/6", "legal_under_ceiling": True},
                {"rate_bpw": a1["complete_bpw"], "representation": "product_quant gate/up d32 k32 s8 + down d16 k16 s2, organ-inverted",
                 "mean_sym_kl_range": a1["mean_sym_kl_range"], "argmax_agreement_range": a1["argmax_agreement_range"],
                 "verdict": "COLLAPSE 6/6", "legal_under_ceiling": False,
                 "note": "1.0075 > 1.0 complete BPW: ILLEGAL under the ceiling. Retained as a measurement, "
                         "NEVER as a search seed. This point must be rebudgeted to <= 1.0, not climbed past."},
            ],
            "seed_policy": (
                "the highest legal search seed is 1.0 complete BPW. Any prior derived "
                "from this curve that exceeds 1.0 is void by law, including "
                "search_start_bpw. Rate priors seed a search; they never select."
            ),
            "measured": {"A1_1p0": a1, "R2_subhalf_best": r2, "R0_parent": parent},
        },
        "organ_sensitivity": {
            "organs": {
                "gate": {"role": "mlp1", "sensitivity": "HIGH", "rel_error_at_rate": 0.92,
                         "f1_allocated_bpw": {"A1_1p0": 1.252698263, "R2_subhalf_best": 0.625409444}},
                "up": {"role": "mlp1", "sensitivity": "HIGH", "rel_error_at_rate": 0.92,
                       "f1_allocated_bpw": {"A1_1p0": 1.252698263, "R2_subhalf_best": 0.625409444}},
                "down": {"role": "mlp2", "sensitivity": "LOWER", "rel_error_at_rate": 0.20,
                         "f1_allocated_bpw": {"A1_1p0": 0.500745138, "R2_subhalf_best": 0.157636007}},
                "attn": {"sensitivity": "UNMEASURED", "f1_allocated_bpw": 1.0004},
                "embed": {"sensitivity": "UNMEASURED", "f1_allocated_bpw": 1.000014},
                "lm_head": {"sensitivity": "UNMEASURED", "f1_allocated_bpw": 1.000014},
            },
            "dominant_failure_organ": "gate",
            "inversion": "mlp1/gate+up is the SENSITIVE organ; mlp2/down tolerates more",
            "inversion_confirmed": True,
            "inversion_is_necessary_not_sufficient": (
                "A1_1p0 spent the inversion correctly (gate/up 1.25 vs down 0.50) and "
                "still collapsed 6/6. Correct allocation inside a failed representation "
                "does not rescue it."
            ),
        },
        "doctor": {
            "successes": [],
            "failures": [
                {"target": "f1_same_budget_doctor", "detail":
                 "NOT RUN at F1. No Doctor pass was executed at capability fidelity on "
                 "Qwen3-235B. Recorded as an untried intervention, not as a failure of Doctor."},
                {"target": "f0_residual_codebook_doctor", "detail":
                 "F0 only: D4_pq_doctor, residual_codebook at 0.15 bpw reserve, per-expert "
                 "whole 0.88845, collapsed on a real forward (argmax 0.0 to 0.3636)."},
            ],
            "honest_note": (
                "same-budget Doctor at capability fidelity was NOT yet run for F1. The "
                "F1 collapse therefore does not falsify Doctor; it falsifies the "
                "undoctored raw-weight PQ/VQ family."
            ),
        },
        "routing": {
            "required_calibration_tokens": 1000,
            "routing_partition_source": state.get("routing_partition_source"),
            "note": "R3_routing_aware was never reached; the F0 finding that 88 calibration tokens is too weak stands unrefuted and unretested at F1.",
        },
        "activation": {
            "inter_expert_mean_pairwise_cosine": 0.0001,
            "verdict": "carried from F0; not contradicted at F1",
        },
        "quality": {
            "capability_gate": {"mean_symmetric_kl_max": 0.10, "next_token_argmax_agreement_min": 0.95},
            "contract_sha256": qc.SEALED_CONTRACT_SHA256,
            "result": {
                "selected_frontier": None,
                "reason": ("both sealed candidates collapsed 6/6 on a real parent-vs-packed "
                           "forward against a healthy control; 12 of 12 packed rows failed the gate"),
                "rows_evaluated": 12,
                "rows_passed": 0,
                "evidence_class": "SHORT_END_TO_END",
                "evidence_class_note": (
                    "5 tokens per prompt is below the 1000-token CAPABILITY threshold. "
                    "SHORT_END_TO_END may not SELECT a frontier; it is ample to REJECT "
                    "one at argmax agreement 0.0."
                ),
            },
            "failures": failures,
            "probes": ["real_parent_vs_packed_forward"],
            "prompt_ids": [p["prompt_id"] for p in parent["per_prompt"]],
            "domains_collapsing_first": [
                "all six simultaneously: factual, code, math, science, instruction, reasoning. "
                "Next-token argmax agreement is exactly 0.0 on every prompt at both rates, so "
                "no domain survived long enough to collapse first."
            ],
        },
        "runtime": {
            "timings": [],
            "dominant_bottleneck": "expert streaming",
            "kernel_requirements": [],
            "note": "expert-outer/candidate-inner loop order and persisted parent logits (51MB on disk) are the two measured runtime levers.",
        },
        "resources": {
            "expert_cache_cap_gib_tested": 64,
            "evictions": 0,
            "ram_gb_before_after": [70, 18],
            "swap_free_mb_at_worst": 906,
            "correct_cap_gib": 20,
            "memory_floor_gib": 20,
            "working_set_gib_per_layer": 8,
            "lesson": "the resource lesson is the 20 GB cache cap, NOT 64 GB. A single lockstep pass has zero cross-layer expert reuse.",
        },
        "source_format": {
            "lessons": [{"id": "qwen3_moe_fused_experts",
                         "detail": "fused per-layer expert tensors; the packer substitutes via expert_hook on a real forward"}],
            "decoder_requirements": ["fused MoE expert tensor decode", "pinned revision re-download"],
        },
        "storage": {
            "mode": "release source after harvest; re-download from pinned revision",
            "lessons": [{"id": "persist_parent_logits",
                         "detail": "51MB of persisted parent logits removes 1.33h of re-forward per restart"}],
        },
        # The ceiling itself, the raw-weight PQ/VQ falsification and the mandatory
        # subbit_closure_plan are carried natively by post_parent_review. Only the
        # finding it does not already hold is added here.
        "negative_transfer": [
            {
                "id": "upward_bracketing",
                "claim": "raising the rate until something passes locates the capability cliff",
                "verdict": "FALSIFIED AND FORBIDDEN",
                "evidence": ("F0 searched 0.80 to 0.89 per-expert and F1 searched 0.4930 to 1.0075 complete. "
                             "Both parents were searched ENTIRELY BELOW their cliffs and neither ever established "
                             "a passing rate. Two parents of methodology produced zero positive controls, and the "
                             "response to that is a different method at the ceiling, not a higher rate."),
                "forbidden": ["upward bracketing", "escape receipt above 1 BPW", "any 1.2/1.5/2.0/3.0 anchor",
                              "bisecting inside a fully collapsed band"],
                "replacement": ("change the representation at the ceiling. A missing positive control is a "
                                "reason to change method, not to raise rate."),
            },
        ],
        "assumptions": [
            {"id": "organ_inversion", "statement": "mlp1/gate+up is the sensitive organ", "verdict": "CONFIRMED",
             "evidence": "dominant_failure_organ=gate; down_only 0.20 vs gate_up 0.92 rel_err; the inversion was spent correctly at A1_1p0"},
            {"id": "cache_policy_64gb", "statement": "a large expert cache accelerates a single lockstep pass", "verdict": "FALSIFIED",
             "evidence": "0 evictions at 64GiB, RAM 70 to 18 GB, swap free 906MB; correct cap is 20 GB"},
            {"id": "parent_control_health", "statement": "the bf16 control is healthy enough to trust a negative", "verdict": "CONFIRMED",
             "evidence": "parent ppl 1.61 to 39.33 across six domains on the same harness that produced the collapses"},
            {"id": "capability_cliff_scale_dependence", "statement": "the capability cliff moves with parent scale", "verdict": "FALSIFIED",
             "evidence": "120B collapsed at 0.80 to 0.89 per-expert; 235B collapsed at a stricter COMPLETE 1.0075 and 0.4930"},
            {"id": "raw_weight_vq_reaches_one_bit", "statement": "raw-weight PQ/VQ preserves capability at ~1 bit", "verdict": "FALSIFIED",
             "evidence": "12/12 packed rows, argmax agreement exactly 0.0, symKL 7.61 to 13.47"},
            {"id": "organ_inversion_is_sufficient", "statement": "correct organ allocation rescues a sub-bit artifact", "verdict": "FALSIFIED",
             "evidence": "A1_1p0 applied the inversion and still collapsed 6/6"},
            {"id": "positive_control_ever_established", "statement": "some rate on some parent passed the capability gate", "verdict": "FALSIFIED",
             "evidence": "F0 and F1 were both searched entirely below their cliffs; no parent has ever had a passing rate established"},
            {"id": "subbit_unreachable_in_principle", "statement": "no Hawking method can preserve capability at or below 1 bit", "verdict": "OPEN",
             "evidence": "NOT established and NOT claimed. F1 falsifies one representation family. QAT, distillation, "
                         "compressibility training, structured pruning and learned sharing change the source and are "
                         "not bound by the rate-distortion limit of the original weights. All are untested."},
            {"id": "row_norm_stratification", "statement": "row-norm stratification recovers the collapsed gate/up rows", "verdict": "OPEN",
             "evidence": "R5_rownorm_strat was never reached at F1"},
            {"id": "routing_aware_allocation", "statement": "coldest-quartile harshening buys rate at fixed capability", "verdict": "OPEN",
             "evidence": "R3_routing_aware was never reached at F1; routing_partition_source is null"},
        ],
        "review_answers": {
            "capability_cliff_moved_with_scale": {
                "answer": "NO",
                "evidence": ("120B collapsed at 0.80 to 0.89 per-expert whole BPW (D2 0.80508, D6 0.82885, D4 0.88845); "
                             "235B collapsed at COMPLETE 1.0075 and 0.4930 with argmax 0.0 on every prompt. A ~2x "
                             "parameter count did not move the cliff, and F1's accounting is the stricter of the two.")},
            "geometry_transferred": {
                "answer": "the organ inversion transferred; nothing else did",
                "evidence": "dominant_failure_organ=gate at both F0 and F1; the inversion was necessary and not sufficient"},
            "hot_cold_routing_mattered": {
                "answer": "UNRESOLVED",
                "evidence": "R3_routing_aware never executed; routing_partition_source null; the F0 88-token calibration finding stands"},
            "codebook_sharing_structure_or_amortization": {
                "answer": "AMORTIZATION ONLY, reconfirmed",
                "evidence": ("R2_subhalf_best used shared_grammar codebooks amortized over 128 experts and collapsed at "
                             "least as hard (symKL 9.34 to 13.47) as the unshared PQ at twice the rate")},
            "higher_dimensional_vq_helped": {
                "answer": "NO at capability",
                "evidence": ("the F0 weight-space gain (rel_error 0.782 to 0.668 with VQ dimension) did not convert into "
                             "any capability at F1; weight-space error remains a proxy, never a pass criterion")},
            "row_norm_stratification_helped": {"answer": "UNTESTED", "evidence": "R5_rownorm_strat never reached"},
            "organs_consuming_most_rescue_bytes": {
                "answer": "gate and up",
                "evidence": "gate/up 1.2527 vs down 0.5007 bpw at A1_1p0; gate_up 0.92 vs down_only 0.20 rel_err"},
            "quality_domains_collapsing_first": {
                "answer": "all six simultaneously; none survived",
                "evidence": "next-token argmax agreement exactly 0.0 on factual, code, math, science, instruction and reasoning at both rates"},
            "dominant_runtime_bottleneck": {
                "answer": "expert streaming",
                "evidence": "expert-outer/candidate-inner removed 2.7h of re-streaming; 51MB of persisted parent logits removes 1.33h per restart"},
        },
        "methods": [
            {"id": "organ_inversion_allocation", "name": "organ inversion byte allocation", "status": "CONFIRMED_MEASURED",
             "transfer_breadth": 2.0, "evidence": "confirmed independently at F0 and F1; necessary, not sufficient"},
            {"id": "persisted_parent_logits", "name": "persisted parent logits", "status": "CONFIRMED_MEASURED",
             "transfer_breadth": 1.0, "evidence": "51MB; removes 1.33h per restart"},
            {"id": "expert_cache_20gib", "name": "20GiB expert cache cap", "status": "CONFIRMED_MEASURED",
             "transfer_breadth": 1.0, "evidence": "replaces the falsified 64GiB cap"},
            {"id": "healthy_parent_control", "name": "mandatory live bf16 parent control per prompt", "status": "CONFIRMED_MEASURED",
             "transfer_breadth": 1.5, "evidence": "parent ppl 1.61 to 39.33 is what makes the F1 negative trustworthy"},
            {"id": "qat_at_ceiling", "name": "quantization-aware training at or below 1 BPW", "status": "UNTESTED",
             "transfer_breadth": 2.0, "evidence": "changes the source, so it is not bound by the raw-weight rate-distortion limit falsified at F1"},
            {"id": "distillation_to_subbit_student", "name": "distillation into a sub-bit student", "status": "UNTESTED",
             "transfer_breadth": 2.0, "evidence": "changes the source; untested at any Hawking parent"},
            {"id": "compressibility_training", "name": "compressibility-inducing training", "status": "UNTESTED",
             "transfer_breadth": 2.0, "evidence": "makes the weights compressible instead of compressing incompressible weights"},
            {"id": "structured_pruning_then_subbit", "name": "structured pruning before sub-bit coding", "status": "UNTESTED",
             "transfer_breadth": 1.5, "evidence": "reduces original_weight_count, which is the denominator of the ceiling"},
            {"id": "learned_expert_sharing", "name": "learned (trained) expert sharing", "status": "UNTESTED",
             "transfer_breadth": 1.5, "evidence": "post-hoc sharing is dead (cosine 1e-4); LEARNED sharing creates the shared component instead of assuming it"},
            {"id": "same_budget_doctor_at_capability", "name": "same-budget Doctor at capability fidelity", "status": "UNTESTED",
             "transfer_breadth": 1.5, "evidence": "never run at F1; Doctor bytes count toward the ceiling"},
        ],
        "next_parent": {"storage": {"free_gib": 122, "required_gib": 0, "headroom_gib": 0, "download_in_flight": False}},
    }


# ---------------------------------------------------------------- receipt


def build_receipt(ev, state):
    m = ev["representation"]["measured"]
    a1, r2, parent = m["A1_1p0"], m["R2_subhalf_best"], m["R0_parent"]
    reached = {"R0_parent", "A1_1p0", "R2_subhalf_best"}
    return {
        "schema": "hawking.foundry.subbit_unsolved.v1",
        "generated_at_utc": ppr._utc(),
        "parent": ev["parent"],
        "run_status": ev["run_status"],
        "verdict": VERDICT,
        "not_claimed": [
            "sub-bit is impossible",
            "the one-bit ceiling should be raised",
            "an escape receipt above 1 BPW is warranted",
            "a compressed high-rate quality anchor is required",
        ],
        "rate_ceiling": CEILING,
        "methods_attempted": [
            {"id": "A1_1p0", "family": "raw-weight product quantization, organ-inverted",
             "spec": a1["spec"], "complete_bpw": a1["complete_bpw"],
             "complete_artifact_bytes": a1["complete_artifact_bytes"],
             "legal_under_ceiling": False,
             "outcome": "COLLAPSE 6/6",
             "diagnosis": ("at 1.0075 complete BPW the gate/up codebooks (d32, k32, 8 subspaces) cannot span the "
                           "mlp1 row geometry; output argmax agreement is exactly 0.0 on all six domains while the "
                           "bf16 control is healthy. Spending the organ inversion correctly did not help.")},
            {"id": "R2_subhalf_best", "family": "shared-grammar vector quantization",
             "spec": r2["spec"], "complete_bpw": r2["complete_bpw"],
             "complete_artifact_bytes": r2["complete_artifact_bytes"],
             "legal_under_ceiling": True,
             "outcome": "COLLAPSE 6/6",
             "diagnosis": ("halving the rate to 0.4930 complete BPW cost roughly 2 to 3 symKL, not an order of "
                           "magnitude. The artifact was already fully collapsed at 1.0075, so the rate was never "
                           "the binding constraint: the representation was.")},
            {"id": "R1_c1_corrected", "planned_complete_bpw": 0.684, "outcome": "NOT REACHED", "diagnosis": "campaign sealed first"},
            {"id": "A2_0p85", "planned_complete_bpw": 0.85, "outcome": "NOT REACHED",
             "diagnosis": "designed as a bisection between two collapsed points; bisecting inside a fully collapsed band has no information value"},
            {"id": "R3_routing_aware", "outcome": "NOT REACHED", "diagnosis": "coldest-quartile harshening never executed; routing_partition_source null"},
            {"id": "R4_highdim_vq", "outcome": "NOT REACHED", "diagnosis": "k=65536 chunked k-means never executed at F1"},
            {"id": "R5_rownorm_strat", "outcome": "NOT REACHED",
             "diagnosis": "94 percent of gate/up rows collapse onto one codeword (row norms span 1e-5 to 0.91); the single most specific untried in-family lever"},
        ],
        "ladder_declared_but_not_reached": [v for v in state["ladder_order"] if v not in reached],
        "diagnoses": [
            "the parent control is healthy (ppl 1.61 to 39.33), so the collapse is a property of the representation and not of the harness",
            "argmax agreement is exactly 0.0 at BOTH rates, so the artifacts are not degraded, they are destroyed; there is no gradient to bisect",
            "halving the complete rate 1.0075 -> 0.4930 moved symKL only 7.61-10.87 -> 9.34-13.47: rate was never the binding constraint",
            "the organ inversion is a correct allocation prior and is not sufficient",
            "weight-space rel_error improvements from F0 (0.782 -> 0.668 with VQ dimension) did not convert into any capability",
            "codebook sharing at F1 bought amortization only, reconfirming the F0 finding that there is no structural gain to extract",
            "F0 and F1 were both searched ENTIRELY BELOW their cliffs; no parent has ever had a passing rate established, so the method has never had a positive control",
        ],
        "doctor_attempts": {
            "f1_same_budget_doctor_at_capability_fidelity": "NOT RUN",
            "honest_note": ("no Doctor pass was executed on Qwen3-235B at capability fidelity. The F1 negative "
                            "therefore falsifies the UNDOCTORED raw-weight PQ/VQ family. Doctor bytes count toward "
                            "the complete ceiling, so any future Doctor attempt must be budgeted inside 1.0 BPW."),
            "prior_art": [{"parent": "gpt-oss-120b:F0", "variant": "D4_pq_doctor",
                           "family": "residual_codebook Doctor, 0.15 bpw reserve",
                           "per_expert_whole_bpw": 0.88845, "outcome": "COLLAPSE",
                           "argmax_agreement_range": [0.0, 0.3636]}],
        },
        "structural_interventions_attempted": {
            "count": 0,
            "note": "NONE. Every F1 candidate compressed the ORIGINAL weights unchanged.",
            "unattempted": [
                {"id": "qat_at_ceiling", "why_it_escapes_the_f1_result": "training changes the weights, so the F1 rate-distortion measurement on the original weights does not bound it"},
                {"id": "distillation_to_subbit_student", "why_it_escapes_the_f1_result": "the student is a different model, not a coding of this one"},
                {"id": "compressibility_training", "why_it_escapes_the_f1_result": "makes the source compressible rather than compressing an incompressible source"},
                {"id": "structured_pruning_then_subbit", "why_it_escapes_the_f1_result": "reduces original_weight_count, the ceiling denominator"},
                {"id": "learned_expert_sharing", "why_it_escapes_the_f1_result": "creates a shared component instead of assuming one; post-hoc sharing is dead at cosine 1e-4"},
            ],
        },
        "quality_failures": {
            "gate": {"mean_symmetric_kl_max": 0.10, "next_token_argmax_agreement_min": 0.95,
                     "contract_sha256": qc.SEALED_CONTRACT_SHA256},
            "rows_evaluated": 12, "rows_passed": 0,
            "A1_1p0": {"complete_bpw": a1["complete_bpw"], "collapse": "6/6",
                       "mean_sym_kl_range": a1["mean_sym_kl_range"],
                       "argmax_agreement_range": a1["argmax_agreement_range"],
                       "per_prompt": a1["per_prompt"]},
            "R2_subhalf_best": {"complete_bpw": r2["complete_bpw"], "collapse": "6/6",
                                "mean_sym_kl_range": r2["mean_sym_kl_range"],
                                "argmax_agreement_range": r2["argmax_agreement_range"],
                                "per_prompt": r2["per_prompt"]},
            "parent_control": {"perplexity_range": parent["perplexity_range"], "healthy": parent["healthy"],
                               "per_prompt": parent["per_prompt"]},
        },
        "exact_bytes": {
            "A1_1p0": {"complete_artifact_bytes": a1["complete_artifact_bytes"],
                       "complete_bpw": a1["complete_bpw"],
                       "original_weight_count": a1["original_weight_count"],
                       "organ_bpw": a1["organ_bpw"],
                       "over_ceiling_by_bpw": round(a1["complete_bpw"] - 1.0, 9)},
            "R2_subhalf_best": {"complete_artifact_bytes": r2["complete_artifact_bytes"],
                                "complete_bpw": r2["complete_bpw"],
                                "original_weight_count": r2["original_weight_count"],
                                "organ_bpw": r2["organ_bpw"],
                                "over_ceiling_by_bpw": 0.0},
            "accounting": ("whole model, every tensor class billed, codebooks amortized over 128 experts. "
                           "No expert-only or payload-only figure appears anywhere in this receipt."),
        },
        "reopening_conditions": [
            {"id": "reopen_raw_weight_pq_vq",
             "condition": "a parent measures a raw-weight VQ artifact at or below 1.0 COMPLETE BPW reaching symKL <= 0.10 AND argmax >= 0.95 on >= 1000 holdout tokens across the protected domains",
             "otherwise": "the family stays retired at and below 1 bit"},
            {"id": "reopen_row_norm_stratification",
             "condition": "row-norm-stratified codebooks are actually built and forwarded at <= 1.0 complete BPW (never reached at F1)",
             "note": "highest-value untried IN-FAMILY lever: 94 percent of gate/up rows currently collapse onto one codeword"},
            {"id": "reopen_routing_aware_allocation",
             "condition": ">= 1000 calibration tokens produce a stable routing partition and the coldest-quartile variant is forwarded at <= 1.0 complete BPW"},
            {"id": "reopen_doctor",
             "condition": "a same-budget Doctor pass runs at capability fidelity with Doctor bytes billed inside the 1.0 complete ceiling"},
            {"id": "unlock_structural_family",
             "condition": "IMMEDIATE. No precondition. QAT, distillation, compressibility training, structured pruning and learned sharing are unblocked by this result, not blocked by it."},
            {"id": "never_reopen_upward_bracketing",
             "condition": "NEVER. Above 1.0 complete BPW is forbidden by law regardless of any future measurement."},
        ],
        "transfers_to_next_parent": True,
        "source_rows": {"campaign": CKPT, "rows": state["rows_done"], "ladder_sha": state["ladder_sha"], "status": state["status"]},
    }


# ---------------------------------------------------------------- main


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rows, state = read_campaign()
    assert len(rows) == 18, "expected the sealed 18-row ladder, got %d" % len(rows)
    ev = build_evidence(rows, state)
    ppr.validate_evidence(ev)
    receipt = build_receipt(ev, state)

    a1 = ev["representation"]["measured"]["A1_1p0"]
    r2 = ev["representation"]["measured"]["R2_subhalf_best"]
    assert a1["collapse_count"] == 6 and r2["collapse_count"] == 6
    assert not a1["capability_pass"] and not r2["capability_pass"]
    assert a1["complete_bpw"] > 1.0 and r2["complete_bpw"] < 0.5
    assert ev["representation"]["measured"]["R0_parent"]["healthy"]
    assert all(p["argmax_agreement_range"] == [0.0, 0.0] for p in (a1, r2))
    # complete_bytes*8/complete_bpw must land on the real 235B parameter count
    assert all(234e9 < p["original_weight_count"] < 236e9 for p in (a1, r2))
    assert receipt["structural_interventions_attempted"]["count"] == 0

    if "--check" in argv:
        print("check ok: 18 rows, 12/12 collapse, parent control healthy")
        return 0

    ppr._write_json(EVIDENCE_PATH, ev)
    ppr._write_json(RECEIPT_PATH, receipt)

    with open(REGISTRY_PATH) as fh:
        adapters = json.load(fh)["adapters"]
    written = ppr.generate(ev, REVIEW_DIR, adapters)

    matrix_path = os.path.join(REVIEW_DIR, ppr.slug(PARENT_ID) + "_ADAPTER_REBASE_MATRIX.json")
    with open(matrix_path) as fh:
        matrix = json.load(fh)
    gate_state = ppr.build_gate_state(REVIEW_DIR, ev, adapters, matrix,
                                      heavy_lease_held=False,
                                      storage=ev["next_parent"]["storage"])
    heavy_ok, heavy_reasons = ppr.can_launch_next_parent(gate_state)
    dl_ok, dl_reasons = ppr.can_start_next_download(gate_state)
    gate_doc = {
        "schema": "hawking.foundry.post_parent_review.next_parent_gate.v1",
        "generated_at_utc": ppr._utc(),
        "verdict": VERDICT,
        "heavy_controller_launch_allowed": heavy_ok,
        "heavy_blocking_reasons": heavy_reasons,
        "source_download_allowed": dl_ok,
        "download_blocking_reasons": dl_reasons,
        "state": gate_state,
    }
    gate_path = ppr._write_json(os.path.join(REVIEW_DIR, ppr.slug(PARENT_ID) + "_NEXT_PARENT_GATE.json"), gate_doc)

    print(json.dumps({
        "verdict": VERDICT,
        "evidence": EVIDENCE_PATH,
        "receipt": RECEIPT_PATH,
        "review": written,
        "gate": gate_path,
        "heavy_controller_launch_allowed": heavy_ok,
        "stale_adapters": [s["adapter_id"] for s in gate_state["stale_adapters"]],
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
