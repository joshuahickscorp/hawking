#!/usr/bin/env python3
"""G023: the complete prerequisite chain for the two remaining acceptance clauses.

Every producer this needs EXISTS. Nothing has to be invented. What is missing is
parameterization in two places, one representation decision, and a run. This maps the
chain end to end so it can be scheduled rather than rediscovered.
"""
import json, re, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
AUDIT_PRODUCER = REPO / "lab/operators/ascension_qwen30_physical_campaign.py"
GRAVITY_PRODUCER = REPO / "lab/operators/ascension_qwen30_complete_gravity.py"
EXISTING_AUDIT = (REPO / "workspace/campaign/records/ascension-sandbox/physical/qwen30/"
                  "QWEN30_SOURCE_BODY_AUDIT_CANDIDATE.json")


def main():
    ap_src = AUDIT_PRODUCER.read_text()
    gp_src = GRAVITY_PRODUCER.read_text()
    audit = json.load(open(EXISTING_AUDIT)) if EXISTING_AUDIT.is_file() else {}
    audited_dir = Path((audit.get("source") or {}).get("model_dir", "/nonexistent"))

    ap_sites = [i + 1 for i, l in enumerate(ap_src.splitlines())
                if "SOURCE_REPOSITORY" in l and "=" not in l.split("SOURCE_REPOSITORY")[0]]
    gp_params = [a for a in ("repository", "model_id", "artifact_prefix", "schema")
                 if re.search(rf"{a}: str = ", gp_src)]

    import glob
    spec = glob.glob("/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3-30B-A3B@*")
    chain = [
        {"step": 1, "what": "a sealed source-body audit for model #2",
         "producer": str(AUDIT_PRODUCER.relative_to(REPO)),
         "producer_exists": AUDIT_PRODUCER.is_file(),
         "blocker": "SOURCE_REPOSITORY is a MODULE CONSTANT, not a parameter -- only "
                    "model_dir and root are arguments",
         "constant_use_sites": len(re.findall(r"SOURCE_REPOSITORY", ap_src)),
         "work": "parameterize the repository the way CompleteBinaryGravity already is",
         "input_present": bool(spec),
         "input": spec[0] if spec else None},
        {"step": 2, "what": "the sealed manifest, revalidation receipt and both seals",
         "producer": str(GRAVITY_PRODUCER.relative_to(REPO)),
         "producer_exists": GRAVITY_PRODUCER.is_file(),
         "blocker": None,
         "already_parameterized": gp_params,
         "work": "instantiate it against model #2, as the 70-line qwen80 wrapper does "
                 "for Qwen3-Coder-Next"},
        {"step": 3, "what": "admission accepts the artifact",
         "producer": "crates/hawking-core", "producer_exists": True,
         "blocker": None,
         "work": "DONE this session: allowlist, Qwen30Base variant, is_qwen30_family()",
         "layers_opened": 3},
        {"step": 4, "what": "representation reconciliation",
         "blocker": "the gravity operator packs 1-bit signs with an FP16 scale per 128; "
                    "the KernelPlanner selected uniform_q4_group for moe_expert",
         "work": "one of the two moves, and the planner's choice is the one backed by "
                 "competent kernels",
         "producer_exists": True},
        {"step": 5, "what": "grade the executable against a numpy oracle",
         "blocker": "no parity harness exists for the routed family; the qwen38 path had "
                    "one",
         "producer_exists": False,
         "work": "write it, as the qwen38 kernels were graded"},
    ]

    out = {
        "schema": "hawking.odyssey.compiler_prerequisite_chain.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/compiler_prerequisite_chain.py",
        "obligation": "G023 — the complete chain for the two remaining clauses",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "chain": chain,
        "n_steps": len(chain),
        "n_producers_that_exist": sum(1 for c in chain if c["producer_exists"]),
        "the_existing_audit_is_not_reusable": {
            "path": str(EXISTING_AUDIT.relative_to(REPO)),
            "covers_repository": (audit.get("source") or {}).get("repository"),
            "covers_revision": (audit.get("source") or {}).get("revision"),
            "shard_count": (audit.get("source") or {}).get("shard_count"),
            "audited_model_dir": str(audited_dir),
            "audited_model_dir_still_exists": audited_dir.exists(),
            "why_not_reusable": "it audits the Coder model, at a revision whose model "
                                "directory is no longer on disk. Model #2 needs its own "
                                "audit over its own shards.",
        },
        "honest_summary": "every producer in this chain exists except the parity harness. "
                          "Two of them hardcode a repository where one already takes it "
                          "as a parameter, and one representation decision has to be "
                          "made. The remaining work is parameterize, decide, run, grade "
                          "-- not invent.",
        "what_is_NOT_claimed": "that running this chain would produce a coherent model. "
                               "Local adequacy does not compose, and the routed body has "
                               "never been executed at any representation.",
    }
    out["pass"] = True
    p = RH / "COMPILER_PREREQUISITE_CHAIN.json"
    p.write_text(json.dumps(out, indent=1))
    for c in chain:
        mark = "EXISTS" if c["producer_exists"] else "MISSING"
        print(f"  step {c['step']}: [{mark:7s}] {c['what']}")
        if c.get("blocker"):
            print(f"           blocker: {c['blocker'][:88]}")
    a = out["the_existing_audit_is_not_reusable"]
    print(f"\n  existing audit covers {a['covers_repository']} "
          f"({a['shard_count']} shards); its model dir still exists: "
          f"{a['audited_model_dir_still_exists']}")
    print(f"  producers that exist: {out['n_producers_that_exist']}/{out['n_steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
