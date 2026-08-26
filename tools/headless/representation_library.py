#!/usr/bin/env python3
"""REPRESENTATION LIBRARY — candidate families, and failures that stay searchable.

Two jobs (directive §50, §63, §64):

  * `seed(organ, arch_class)` returns a RANKED candidate list for a new organ, with the
    evidence behind each rank and the families excluded with reasons. That is what
    replaces a cold search on specimen #2.
  * `prior_failures(...)` returns what already failed and under what condition it
    reopens. Empty is a valid answer; a wrong one is not.

Every Qwen failure is MODEL_SPECIFIC. It warns a different architecture; it never
prunes a family there. The negative-science store owns the failure records; this
module reads them rather than keeping a second copy that can drift.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
OUT = RH / "REPRESENTATION_LIBRARY.json"
sys.path.insert(0, str(REPO / "tools/headless"))

PARENT = "qwen3.8-27b-abliterated"

# The directive's family space (§50). A family with no measurement here is UNTESTED with
# a reason -- present and searchable, never invented a result for.
SPACE = {
    "conventional_low_bit": "grouped absmax / RTN at 3-4 bits",
    "binary": "1-bit sign code",
    "ternary": "{-1,0,+1}",
    "trit_plane": "ternary decomposed into bit planes",
    "vector_quantization": "codeword per weight vector",
    "additive_codebooks": "sum of codebook entries",
    "adaptive_codebooks": "codebook refit per region",
    "residual_codebooks": "coarse codebook plus residual codebook",
    "shared_basis": "K shared basis vectors plus per-row coefficients",
    "tensor_decomposition": "TT / Tucker / CP factorization",
    "low_rank": "UV factorization",
    "low_rank_plus_sparse": "UV plus a sparse exception set",
    "generated_coefficients": "weights emitted by a small generator rather than stored",
    "structural_elimination": "the structure is removed, not compressed",
    "protected_islands": "ultra-low bulk plus tiny high-precision regions",
    "routed_operators": "the operator executed depends on the input",
    "recurrent_representation": "a transition program rather than a dense matrix",
    "q2_affine": "4-level fitted affine at group 64",
    "leftover_f32": "not quantized at all: kept as flat f32",
}
# Which of the directive's ZERO kinds a family attacks (§6). The planner uses this to
# cover all three rather than searching only representation.
ZERO_KIND = {
    "structural_elimination": "ZERO_STORAGE",
    "shared_basis": "ZERO_INDEPENDENT_INFORMATION",
    "generated_coefficients": "ZERO_INDEPENDENT_INFORMATION",
    "tensor_decomposition": "ZERO_INDEPENDENT_INFORMATION",
    "low_rank": "ZERO_INDEPENDENT_INFORMATION",
    "routed_operators": "ZERO_EXECUTION",
    "recurrent_representation": "ZERO_EXECUTION",
}
# Sensitivity allocation (§64): a plan is a list of regions, each with its own bit rate.
# A schema that forces one bpw per organ cannot express the winning Qwen body, which is
# exactly why this is a list and not a scalar.
ALLOCATION_SCHEMA = {
    "regions": [{"selector": "which weights", "bits_per_weight": "float or 0 for eliminated",
                 "family": "which representation family", "why": "sensitivity evidence"}],
    "permits_zero_bit_regions": True, "permits_high_precision_islands": True,
    "forces_uniform_bpw": False,
}


# Organ KINSHIP is the transfer mechanism: an MoE expert IS a dense MLP that only some
# tokens reach, so a representation that hit 2.25 bpw coherently on mlp_gate_up is a
# strong seed for moe_expert -- far stronger than a family with no density number at all.
# Kinship seeds a search; it never asserts the result will hold.
# Organs measured to be left UNQUANTIZED in the winning Qwen body. 353 of its 755 segments
# are flat f32 leftovers -- norms, A_log, conv1d, dt_bias -- costing 10584840 bytes in total,
# about 0.12% of the 8.75 GB executable. Quantizing them buys nothing and risks capability.
LEFTOVER_F32_ORGANS = ["normalization"]

KINSHIP = {
    "moe_expert": ["mlp_gate_up", "mlp_down"],
    "shared_expert": ["mlp_gate_up", "mlp_down"],
    "latent_attention": ["gqa_attention"],
    "mha_attention": ["gqa_attention"],
    "recurrent_state": ["deltanet"],
    "lm_head": ["embed"],
    "mm_projector": ["mlp_gate_up"],
}


class Refused(Exception):
    pass


def _cite(rel, jp=None):
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
            raise Refused(f"{rel}#{jp}: no key {part}")
    return cur


# v1 family spellings -> the directive's family space. Nothing is dropped: a v1 family
# with no home in the space keeps its own entry under its original name.
V1_ALIAS = {
    "q4_control": "conventional_low_bit",
    "low_rank_sparse": "low_rank_plus_sparse",
    # binary_sparse_residual is NOT low_rank_plus_sparse -- it is a binary body with a
    # sparse correction. Aliasing them collapsed two measured v1 families into one and
    # silently lost a row; it keeps its own name.
}


def existing():
    """The v1 receipt is the measured predecessor. Read it from git if the working copy
    has already been rewritten, so a rebuild is idempotent instead of self-erasing."""
    d = None
    if OUT.exists():
        d = json.load(open(OUT))
    if not d or not str(d.get("schema", "")).endswith(".v1"):
        raw = subprocess.run(["git", "-C", str(REPO), "show",
                              f"HEAD:receipts/headless/REPRESENTATION_LIBRARY.json"],
                             capture_output=True, text=True)
        if raw.returncode == 0:
            d = json.loads(raw.stdout)
    fams = (d or {}).get("families") or []
    out = {}
    for f in fams:
        out[V1_ALIAS.get(f.get("family"), f.get("family"))] = f
    return out


def negatives():
    import negative_science as ns
    p = RH / "NOETIC_NEGATIVE_SCIENCE.json"
    return json.load(open(p))["entries"] if p.exists() else []


TECH_TO_FAMILY = {
    "binary_quantization": "binary", "protected_islands_healing": "protected_islands",
    "shared_basis": "shared_basis", "low_rank_correction": "low_rank",
    "shared_k_hybrid": "shared_basis", "coordinate_transform": None,
    "structural_elimination_heads": "structural_elimination",
    "depth_state_merging": None, "speculative_decoding_draft": None,
}


def build():
    prev = existing()
    negs = negatives()
    fams = []
    # leftover_f32 is a measured Qwen outcome even though the v1 library never named it
    prev.setdefault("leftover_f32", {
        "family": "leftover_f32", "successful_organs": LEFTOVER_F32_ORGANS,
        "failed_organs": [],
        "density_frontier": {"active_bpw": 32.0,
                             "source": "receipts/headless/QWEN_CLEAN_REBUILD.json"},
        "execution_cost": {"share_of_executable_bytes": 10584840 / 8754264307,
                           "n_segments": 353,
                           "source": "receipts/headless/QWEN_CLEAN_REBUILD.json"},
    })
    space = dict(SPACE)
    for k in prev:                       # a v1 family with no home in the space keeps its own
        space.setdefault(k, "carried forward from the v1 library")
    for name, what in sorted(space.items()):
        p = prev.get(name)
        fail = [n for n in negs if TECH_TO_FAMILY.get(n.get("technique")) == name]
        # A family with a recorded failure HAS been tested here. Calling it UNTESTED
        # would send the planner to re-run an experiment that already has an answer.
        status = "MEASURED" if (p or fail) else "UNTESTED"
        e = {
            "family": name, "what": what,
            "zero_kind": ZERO_KIND.get(name, "REPRESENTATION"),
            "status": status,
            "untested_reason": None if status == "MEASURED" else
            "no measurement on any Odyssey specimen on this machine",
            "per_architecture": {},
            "failures": [{"id": n["id"], "level": n["level"], "model": n["model"],
                          "organ": n["organ"], "physical_reason": n["physical_reason"],
                          "reopen_condition": n["reopen_condition"],
                          "evidence": n.get("evidence", [])} for n in fail],
        }
        if p:
            # v1's `successful_organs` means "this representation APPLIES to this organ",
            # not "capability survived". Binary is listed as successful on the MLP and is
            # generation-injured -- seeding from that unqualified would recommend a
            # known-broken representation. Cross-check against the negative store and
            # split the list; a capability failure is not a success.
            hurt = {part for n in fail for part in str(n["organ"]).split("+")}
            cap_failed = sorted(hurt & set(p.get("successful_organs", [])))
            e["per_architecture"][PARENT] = {
                "successful_organs": [o for o in p.get("successful_organs", [])
                                      if o not in cap_failed],
                "capability_failed_organs": cap_failed,
                "applies_to_organs": p.get("successful_organs", []),
                "failed_organs": p.get("failed_organs", []),
                "density_frontier": p.get("density_frontier"),
                "execution_cost": p.get("execution_cost"),
            }
            e["preserved_from_v1"] = True
        fams.append(e)
    return fams


def seed(fams, organ, arch_class):
    """Ranked candidates for an organ Hawking has not compiled before."""
    ranked, excluded = [], []
    for f in fams:
        blocking = [x for x in f["failures"]
                    if x["level"] != "MODEL_SPECIFIC" or arch_class in x["model"]]
        warning = [x for x in f["failures"] if x not in blocking]
        if blocking:
            excluded.append({"family": f["family"], "why": "prior failure applies at this "
                             "architecture", "failures": [x["id"] for x in blocking]})
            continue
        arch = f["per_architecture"].get(PARENT, {})
        ok = arch.get("successful_organs") or []
        bad = set(arch.get("capability_failed_organs") or []) | set(arch.get("failed_organs") or [])
        succeeded_here = organ in ok
        kin = [k for k in KINSHIP.get(organ, []) if k in ok]
        kin_failed = [k for k in KINSHIP.get(organ, []) if k in bad]
        df = (arch.get("density_frontier") or {})
        bpw = df.get("active_bpw")
        score, why = 0.0, []
        if succeeded_here:
            score += 2.0
            why.append(f"measured coherent on {organ} for {PARENT}")
        elif kin:
            score += 1.5
            why.append(f"measured coherent on kin organ(s) {kin} for {PARENT}; "
                       f"{organ} shares their operator shape")
        if bpw is not None:
            # a lower measured density frontier is a better seed, capped so density never
            # outweighs a direct measurement on the organ itself
            score += min(1.0, 2.25 / max(bpw, 0.01)) * 0.5
            why.append(f"measured density frontier {bpw} bpw")
        if f["status"] == "MEASURED":
            score += 0.5
        if kin_failed:
            score -= 0.75
            why.append(f"FAILED on kin organ(s) {kin_failed} for {PARENT}; "
                       f"model-specific, so it is demoted rather than excluded")
        if warning:
            score -= 0.25 * len(warning)
            why.append(f"{len(warning)} model-specific failure(s) elsewhere: caution, not a ban")
        if f["zero_kind"] != "REPRESENTATION":
            score += 0.25
            why.append(f"attacks {f['zero_kind']}, searched before plain representation")
        if not succeeded_here and not kin:
            # Nothing is known about this organ. Start CONSERVATIVE: prefer the family
            # that keeps more bits, and let measurement earn the aggressive one. Seeding
            # a router with binary because the scores happened to tie is bad advice.
            if bpw is not None:
                score += min(bpw, 8.0) / 32.0
                why.append(f"unknown organ: conservative prior favours the higher-precision "
                           f"family ({bpw} bpw)")
        ranked.append({
            "family": f["family"], "score": round(score, 3), "zero_kind": f["zero_kind"],
            "kin_organs_it_succeeded_on": kin, "kin_organs_it_failed_on": kin_failed,
            "measured_density_frontier_bpw": bpw,
            "evidence": "; ".join(why) or f"status {f['status']}",
            "warnings": [{"id": w["id"], "model": w["model"],
                          "reopen_condition": w["reopen_condition"]} for w in warning],
        })
    ranked.sort(key=lambda r: -r["score"])
    return {"organ": organ, "arch_class": arch_class, "ranked": ranked,
            "excluded": excluded,
            "law": "a MODEL_SPECIFIC failure on another architecture warns, it never prunes "
                   "(directive §81)"}


def prior_failures(fams, organ=None, family=None):
    out = []
    for f in fams:
        if family and f["family"] != family:
            continue
        for x in f["failures"]:
            if organ and organ not in x["organ"]:
                continue
            out.append({"family": f["family"], **x})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", metavar="PATH")
    ap.add_argument("--seed", nargs=2, metavar=("ORGAN", "ARCH"))
    ap.add_argument("--failures", nargs="?", const="", metavar="ORGAN")
    ap.add_argument("--refuse-demo", action="store_true")
    a = ap.parse_args()

    if a.refuse_demo:
        for rel, jp in (("receipts/headless/NO_SUCH.json", None),
                        ("receipts/headless/BYTES_FRONTIER.json", "nope.nope")):
            try:
                _cite(rel, jp)
            except Refused as r:
                print("REFUSED:", r)
        return 0

    fams = build()
    if a.seed:
        print(json.dumps(seed(fams, *a.seed), indent=1))
        return 0
    if a.failures is not None:
        print(json.dumps(prior_failures(fams, a.failures or None), indent=1))
        return 0
    if not a.rebuild:
        ap.error("nothing to do")

    n_meas = sum(1 for f in fams if f["status"] == "MEASURED")
    # Every v1 measured value must survive the rebuild. Losing one silently is the
    # failure mode this check exists to make impossible.
    prev = existing()
    kept = {f["family"]: f for f in fams}
    lost = []
    for k, v in prev.items():
        e = kept.get(k, {}).get("per_architecture", {}).get(PARENT)
        if not e or e.get("density_frontier") != v.get("density_frontier") \
                or e.get("execution_cost") != v.get("execution_cost"):
            lost.append(k)
    preservation = {"n_v1_families": len(prev), "n_preserved": len(prev) - len(lost),
                    "lost": lost, "alias_map": V1_ALIAS}
    out = {
        "schema": "hawking.headless.representation_library.v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/representation_library.py",
        "obligation": "G018 — REPRESENTATION_LIBRARY (directive §50, §63, §64)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False, "unmeasured_is_absent": True,
        "permanent_law": "fewer stored bits is not fewer nanoseconds "
                         "(receipts/headless/BYTES_FRONTIER.json)",
        "allocation_schema": ALLOCATION_SCHEMA,
        "three_kinds_of_zero": sorted(set(ZERO_KIND.values())),
        "n_families": len(fams), "n_measured": n_meas,
        "n_untested": len(fams) - n_meas,
        "families": fams,
        "v1_preservation": preservation,
        "pass": bool(n_meas >= 7 and len(fams) >= 17 and not preservation["lost"]),
    }
    Path(a.rebuild).write_text(json.dumps(out, indent=1))
    print(f"families={len(fams)} measured={n_meas} untested={out['n_untested']} "
          f"pass={out['pass']}")
    print("untested:", [f["family"] for f in fams if f["status"] == "UNTESTED"])
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
