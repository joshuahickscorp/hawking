#!/usr/bin/env python3
"""NEGATIVE SCIENCE — a failure store a later model queries before spending an experiment.

Two laws (directive §80, §81):

  1. Nine fields per failure: model, organ, technique, representation, kernel, machine,
     capability, physical_reason, reopen_condition. A failure without its reopening
     condition is a dead end pretending to be knowledge.
  2. Three levels -- MODEL_SPECIFIC, FAMILY, GENERAL_PHYSICAL -- and promotion is
     expensive. A single model's failure NEVER promotes and NEVER globally prunes a
     technique. Promotion needs independent measurements on distinct models.

The 31 entries already in the receipt use an older seed/claim_refuted schema. They are
migrated, not rewritten: every original entry is carried verbatim under `source_entry`.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
RECEIPT = RH / "NOETIC_NEGATIVE_SCIENCE.json"

FIELDS = ["model", "organ", "technique", "representation", "kernel", "machine",
          "capability", "physical_reason", "reopen_condition"]
LEVELS = ["MODEL_SPECIFIC", "FAMILY", "GENERAL_PHYSICAL"]
MACHINE = "M3 Ultra 96GB / Metal"

# Promotion rule: a level above MODEL_SPECIFIC needs that many DISTINCT models measured.
MIN_MODELS = {"MODEL_SPECIFIC": 1, "FAMILY": 2, "GENERAL_PHYSICAL": 3}


class Rejected(Exception):
    pass


def _receipt_has(path_and_key):
    """`receipts/headless/FOO.json#a.b.c` -> True when the file exists and the path walks."""
    p, _, jp = path_and_key.partition("#")
    f = REPO / p
    if not f.exists():
        return False, f"missing receipt {p}"
    if not jp:
        return True, "file exists"
    try:
        d = json.load(open(f))
    except Exception as e:
        return False, f"unreadable {p}: {e}"
    cur = d
    for part in jp.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except Exception:
                return False, f"{p}#{jp}: bad list index {part}"
        if not isinstance(cur, dict) or part not in cur:
            return False, f"{p}#{jp}: no key {part}"
        cur = cur[part]
    return True, "path resolves"


def validate(entry):
    missing = [f for f in FIELDS if not entry.get(f)]
    if missing:
        raise Rejected(f"{entry.get('id','<no id>')}: missing required field(s): {missing}")
    lvl = entry.get("level")
    if lvl not in LEVELS:
        raise Rejected(f"{entry.get('id')}: level {lvl!r} not in {LEVELS}")
    models = {m for m in entry.get("measured_on_models", [entry["model"]]) if m}
    if len(models) < MIN_MODELS[lvl]:
        raise Rejected(
            f"{entry.get('id')}: level {lvl} needs {MIN_MODELS[lvl]} independently measured "
            f"models, has {len(models)} ({sorted(models)}). A single model's failure is "
            f"MODEL_SPECIFIC and never prunes a technique.")
    ev = entry.get("evidence") or []
    if not ev:
        raise Rejected(f"{entry.get('id')}: no evidence path")
    bad = [(e, why) for e in ev for ok, why in [_receipt_has(e)] if not ok]
    if bad:
        raise Rejected(f"{entry.get('id')}: evidence does not resolve: {bad}")
    return entry


# The measured Qwen negatives. Every physical_reason quotes the receipt it comes from;
# every one is MODEL_SPECIFIC because exactly one model has measured it.
QWEN = [
    dict(id="QN-BINARY-INJURY", model="qwen3.8-27b-abliterated", organ="mlp_gate_up+mlp_down",
         technique="binary_quantization", representation="1-bit sign code, ~1.25 bpw body",
         kernel="qwen_binary_matvec (sign-code GEMV)", machine=MACHINE,
         capability="generation incoherent",
         physical_reason="the 1.25-bpw binary body is physically fast but generation-injured; "
                         "0 of 4 healing candidates reached coherent generation",
         reopen_condition="a healing scheme that restores coherent generation while the healed "
                          "body stays faster than q2f_g64 at 27.55 ms COMPLETE_TOKEN_NS",
         level="MODEL_SPECIFIC",
         evidence=["receipts/headless/BINARY_HEALING.json#finding.n_that_reached_coherent_generation",
                   "receipts/headless/ONEBIT_FAMILIES.json#verdict.decision"]),
    dict(id="QN-BINARY-HEALING", model="qwen3.8-27b-abliterated", organ="mlp_gate_up+mlp_down",
         technique="protected_islands_healing", representation="binary body + high-precision islands",
         kernel="qwen_binary_matvec + island fallback", machine=MACHINE,
         capability="generation incoherent",
         physical_reason="the injury is broad, not localized: no small protected island cheaply "
                         "restored it; 0/4 candidates reached coherent generation",
         reopen_condition="a sensitivity map that localizes the injury to a region small enough "
                          "that protecting it costs less than the 2.25-bpw q2f body",
         level="MODEL_SPECIFIC",
         evidence=["receipts/headless/BINARY_HEALING.json#finding.best_by_capability_per_tax"]),
    dict(id="QN-SHARED-BASIS-DENSITY", model="qwen3.8-27b-abliterated", organ="mlp_gate_up+mlp_down",
         technique="shared_basis", representation="shared basis, ~0.53 local bpw",
         kernel="fused shared-basis matvec (competent: dispatches 384->192, "
                "COMPLETE_TOKEN_NS 110201749->24554625)", machine=MACHINE,
         capability="held-out activation reconstruction fails; not coherent",
         physical_reason="the KERNEL is competent and the byte win does translate to nanoseconds, "
                         "but no K below ~2.25 bpw composes coherently for the MLP: the local "
                         "functional probe dies at held-out activation",
         reopen_condition="a shared-basis point that is coherent at held-out activation AND beats "
                          "q2f on both density and COMPLETE_TOKEN_NS",
         level="MODEL_SPECIFIC",
         evidence=["receipts/headless/SHARED_BASIS_COHERENT.json#finding.reason",
                   "receipts/headless/SHARED_BASIS_KERNEL.json#finding.reason"]),
    dict(id="QN-LOWRANK-HEALING", model="qwen3.8-27b-abliterated", organ="mlp_gate_up+mlp_down",
         technique="low_rank_correction", representation="low-rank residual under a 1.0 bpw budget",
         kernel="hybrid operator (low-rank + quantized body)", machine=MACHINE,
         capability="held-out activation reconstruction fails",
         physical_reason="no distributed correction under the 1.0 bpw budget restored held-out "
                         "activations on real X; even r=256 at 1.035 extra bpw pushed the body to "
                         "2.285 > 2.25 with rel_fro 0.4798",
         reopen_condition="a correction whose extra bpw keeps the body under 2.25 while rel_fro "
                          "on real held-out X drops below the q2f baseline",
         level="MODEL_SPECIFIC",
         evidence=["receipts/headless/HYBRID_OPERATOR.json#finding.reason"]),
    dict(id="QN-SHARED-K-HYBRID", model="qwen3.8-27b-abliterated", organ="mlp_gate_up+mlp_down",
         technique="shared_k_hybrid", representation="shared K=2 basis plus per-tensor correction",
         kernel="hybrid operator", machine=MACHINE,
         capability="incoherent and slower",
         physical_reason="shared K=2 costs 0.531 extra bpw and still fails to restore held-out "
                         "activations; the hybrid remained slower and incoherent",
         reopen_condition="a shared-K variant that is coherent on held-out X at a total body bpw "
                          "below 2.25",
         level="MODEL_SPECIFIC",
         evidence=["receipts/headless/HYBRID_OPERATOR.json#finding.reason"]),
    dict(id="QN-COORDINATE-TRANSFORM", model="qwen3.8-27b-abliterated", organ="mlp_gate_up+mlp_down",
         technique="coordinate_transform", representation="rotations / structured orthogonal bases",
         kernel="transform-then-quantize matvec", machine=MACHINE,
         capability="floor unchanged",
         physical_reason="the tested rotation families did not materially move the Qwen MLP "
                         "information floor; ~2.25 bpw held under coordinate change",
         reopen_condition="a transform family not in the probe (learned orthogonal, Kronecker, "
                          "head-specific, blockwise) that moves the measured floor below 2.25",
         level="MODEL_SPECIFIC",
         evidence=["receipts/headless/COORDINATE_TRANSFORM_PROBE.json#ROTATION_MOVES_BARRIER"]),
    dict(id="QN-HEAD-REDUNDANCY", model="qwen3.8-27b-abliterated", organ="gqa_attention",
         technique="structural_elimination_heads", representation="shared or removed attention heads",
         kernel="n/a (elimination)", machine=MACHINE,
         capability="refuted before packing",
         physical_reason="Q heads mean cosine 0.0438 and K/V/O similarly near-orthogonal, so there "
                         "is no shared-head structure to exploit; MLP dead channels 0 of 1114112; "
                         "near-identity layers 0",
         reopen_condition="an organ or model where head cosine similarity is high enough that "
                          "sharing costs less capability than the bits it saves",
         level="MODEL_SPECIFIC",
         evidence=["receipts/headless/STRUCTURAL_ELIMINATION.json#verdict.one_line"]),
    dict(id="QN-STATE-MERGING", model="qwen3.8-27b-abliterated", organ="kv_state+deltanet_state",
         technique="depth_state_merging", representation="merged / shared KV across depth",
         kernel="n/a (state)", machine=MACHINE,
         capability="negative under tested conditions",
         physical_reason="depth-state and KV merging measured negative on this Qwen under the "
                         "tested conditions",
         reopen_condition="a state topology (recurrent, latent-attention, or a longer-context "
                          "regime) where merged state preserves capability",
         level="MODEL_SPECIFIC",
         evidence=["receipts/headless/STATE_GRAVITY.json#answer"]),
    dict(id="QN-BINARY-AS-DRAFT", model="qwen3.8-27b-abliterated", organ="whole_model",
         technique="speculative_decoding_draft", representation="1.25-bpw binary body as draft model",
         kernel="binary GEMV draft + full verify", machine=MACHINE,
         capability="draft acceptance zero at position 0",
         physical_reason="the 1.25 binary is not a useful draft: token acceptance alpha = 0 at "
                         "position 0, so no expensive forward pass is saved",
         reopen_condition="a cheap draft whose measured acceptance at position 0 is high enough "
                          "that accepted tokens per expensive forward pass rises",
         level="MODEL_SPECIFIC",
         evidence=["receipts/headless/DECODING_GRAVITY.json#one_line"]),
]


def migrate(old):
    """Carry the 31 seed/claim_refuted entries forward verbatim under source_entry."""
    out = []
    for e in old.get("entries", []):
        sc = e.get("scope") or {}
        out.append({
            "id": e.get("id"),
            "model": sc.get("model") or "UNRECORDED_IN_SOURCE",
            "organ": sc.get("organ") or "UNRECORDED_IN_SOURCE",
            "technique": e.get("seed") or "UNRECORDED_IN_SOURCE",
            "representation": sc.get("codec") or sc.get("regime") or "UNRECORDED_IN_SOURCE",
            "kernel": sc.get("kernel") or "UNRECORDED_IN_SOURCE",
            "machine": sc.get("machine") or MACHINE,
            "capability": e.get("claim_refuted") or "UNRECORDED_IN_SOURCE",
            "physical_reason": e.get("kind_reasoning") or "UNRECORDED_IN_SOURCE",
            "reopen_condition": e.get("reopen_condition") or "UNRECORDED_IN_SOURCE",
            "level": "MODEL_SPECIFIC",
            "kind": e.get("kind"),
            "migrated": True,
            "evidence": [],
            "source_entry": e,
        })
    return out


def prior_failures(entries, organ=None, technique=None, arch_class=None):
    """What a planner asks before spending an experiment."""
    hits = []
    for e in entries:
        if organ and organ not in (e.get("organ") or ""):
            continue
        if technique and technique not in (e.get("technique") or ""):
            continue
        if arch_class and e.get("level") == "MODEL_SPECIFIC" and arch_class not in (e.get("model") or ""):
            # a model-specific negative only warns for that model; it never prunes
            hits.append({**e, "applies": "WARNING_ONLY_DIFFERENT_MODEL"})
            continue
        hits.append({**e, "applies": "APPLIES"})
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", metavar="PATH", default=None)
    ap.add_argument("--query-organ"); ap.add_argument("--query-technique")
    ap.add_argument("--try-promote", metavar="ID")
    a = ap.parse_args()

    old = json.load(open(RECEIPT)) if RECEIPT.exists() else {}
    # Migrate only from the v1 predecessor. Re-migrating our own v2 output would
    # duplicate every entry on each rebuild -- it did, once, and that is why this
    # check exists rather than being assumed.
    migrated = migrate(old) if str(old.get("schema", "")).endswith(".v1") else [
        e for e in old.get("entries", []) if e.get("migrated")]
    accepted, rejected = [], []
    for e in QWEN:
        try:
            accepted.append(validate({**e, "measured_on_models": [e["model"]], "migrated": False}))
        except Rejected as r:
            rejected.append(str(r))
    entries = accepted + migrated

    if a.query_organ or a.query_technique:
        print(json.dumps([{k: h[k] for k in ("id", "technique", "level", "applies",
                                             "reopen_condition")}
                          for h in prior_failures(entries, a.query_organ, a.query_technique)],
                         indent=1))
        return 0

    if a.try_promote:
        tgt = next((e for e in entries if e["id"] == a.try_promote), None)
        if not tgt:
            print(f"no entry {a.try_promote}")
            return 2
        try:
            validate({**tgt, "level": "GENERAL_PHYSICAL"})
            print(f"PROMOTED {a.try_promote} to GENERAL_PHYSICAL")
            return 0
        except Rejected as r:
            print(f"PROMOTION REFUSED: {r}")
            return 1

    if not a.rebuild:
        ap.error("nothing to do")

    counts = {lvl: sum(1 for e in entries if e["level"] == lvl) for lvl in LEVELS}
    out = {
        "schema": "hawking.headless.negative_science.v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/negative_science.py",
        "obligation": "G021 — NEGATIVE_SCIENCE THREE-LEVEL (directive §80, §81)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "required_fields": FIELDS,
        "levels": LEVELS,
        "promotion_rule": {"min_distinct_models": MIN_MODELS,
                           "law": "a single model's failure is MODEL_SPECIFIC and never "
                                  "globally prunes a technique (directive §81)"},
        "counts": {"total": len(entries), "by_level": counts,
                   "qwen_measured": len(accepted), "migrated_from_v1": len(migrated),
                   "rejected_at_admission": len(rejected)},
        "rejected_at_admission": rejected,
        "entries": entries,
        "predecessor_counts": old.get("counts"),
        "pass": bool(accepted and not rejected and counts["GENERAL_PHYSICAL"] == 0),
    }
    Path(a.rebuild).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["counts"], indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
