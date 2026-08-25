#!/usr/bin/env python3
"""G023 step 1, smallest honest slice: the selected representation applied to REAL
routed-expert weights.

The KernelPlanner selected conventional_low_bit for moe_expert, executed by the
uniform_q4_group kernels. Until now nobody had applied it to a routed expert, and the
packer cannot emit one -- whole_model_native.py resolves `mlp.experts.N.gate_proj.weight`
to `leftover` because is_mlp_gemv only matches names ENDING in `mlp.gate_proj.weight`.

This packs real expert tensors from model #2 and decodes them back. It is the first time
any stage of this pipeline has produced bytes for model #2.

WHAT IT DOES NOT SHOW
---------------------
That model #2 would work. This campaign's own permanent finding is that LOCAL ADEQUACY
DOES NOT COMPOSE: a ternary body whose organs all passed in isolation failed the
whole-model bracket, and variantA scored 0/43 with every organ locally validated. A
per-tensor cosine is a necessary condition and nothing more.
"""
import json, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"


def main():
    probe = json.load(open("/tmp/moe_pack_probe.json"))
    kp = json.load(open(RH / "KERNEL_PLANNER_MODEL2.json"))
    sel = next(r for r in kp["organ_plan"] if r["organ"] == "moe_expert")

    out = {
        "schema": "hawking.odyssey.moe_pack_slice.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/moe_pack_slice.py",
        "obligation": "G023 step 1 — does the selected representation work on routed "
                      "experts?",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "representation_under_test": {
            "selected_by": "receipts/headless/KERNEL_PLANNER_MODEL2.json",
            "family": sel["selected_representation"],
            "downgraded_from": sel["seeded_families_in_score_order"][0]["family"],
            "codec": "uniform q4, group 64, grouped absmax offset-binary, "
                     "bound = 2^(bits-1) - 1",
            "competent_kernels": sel["n_competent_kernels"],
        },
        "result": probe,
        "why_this_is_the_first_bytes": "every AUTOMATIC stage before this emitted an "
                                       "analysis. The audit recorded that no stage "
                                       "produced bytes for model #2; this one does.",
        "packer_gap_this_probes": {
            "resolver": "tools/headless/whole_model_native.py::organ_role",
            "why_experts_fall_through": "is_mlp_gemv matches names ENDING in "
                                        "`mlp.gate_proj.weight`, and an expert tensor is "
                                        "`mlp.experts.101.gate_proj.weight`, so every "
                                        "expert resolves to `leftover` and is kept f32",
            "what_a_real_packer_needs": [
                "is_moe_expert matching mlp.experts.<N>.{gate,up,down}_proj.weight",
                "is_moe_router matching mlp.gate.weight",
                "GENOME entries for both, using the representations already selected",
                "per-expert segmentation in the catalog so the runtime can index by "
                "route id",
            ],
        },
        "LOCAL_ADEQUACY_DOES_NOT_COMPOSE": {
            "law": "a per-tensor cosine is a necessary condition, never a sufficient one",
            "evidence_from_this_campaign": [
                "the ternary whole-model bracket failed at 1.85 bpw with every organ "
                "locally validated",
                "variantA-2.98 scored 0/43 with each organ measured adequate in isolation",
            ],
            "so_this_result_means": "the codec packs and decodes routed-expert weights at "
                                    "the expected rate and fidelity. It does NOT mean a "
                                    "model #2 body at this representation would be "
                                    "coherent, and nothing here should be quoted as if "
                                    "it did.",
        },
        "scope": {
            "tensors_packed": probe["n_tensors_packed"],
            "tensors_in_the_layer": 384,
            "layers_in_the_model": 48,
            "fraction_of_the_organ": round(probe["n_tensors_packed"] / 18432, 6),
            "no_catalog_no_container": True,
        },
    }
    out["pass"] = bool(probe["median_cosine"] > 0.98 and probe["median_bpw"] < 5)
    p = RH / "MOE_PACK_SLICE.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"  representation: {out['representation_under_test']['family']} "
          f"(downgraded from {out['representation_under_test']['downgraded_from']})")
    print(f"  {probe['n_tensors_packed']} real expert tensors packed at "
          f"{probe['median_bpw']} bpw")
    print(f"  median cosine {probe['median_cosine']}  worst {probe['worst_cosine']}")
    print(f"  median rel_fro_err {probe['median_rel_fro_err']}  "
          f"worst {probe['worst_rel_fro_err']}")
    print(f"  {probe['total_source_bytes']:,} -> {probe['total_packed_bytes']:,} bytes "
          f"({probe['total_source_bytes']/probe['total_packed_bytes']:.2f}x)")
    print(f"  scope: {out['scope']['fraction_of_the_organ']:.4%} of the organ, "
          f"no catalog, no container")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
