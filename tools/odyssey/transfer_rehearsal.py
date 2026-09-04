#!/usr/bin/env python3
"""TRANSFER REHEARSAL — seed a new specimen from the libraries alone.

The point is not that a plan comes out. The point is WHAT WENT IN. If the rehearsal can
reach Qwen's private scratch -- its receipts, its captures, its artifacts -- then any
head start it shows is smuggled, not transferred.

So the inputs are an allowlist, and the audit is real: a `sys.addaudithook` records every
file the process opens, and any read outside the allowlist fails the rehearsal. A
declaration that no scratch was used would be worth nothing here.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"

# Everything the rehearsal is allowed to read. Canonical libraries, the transfer report,
# and the specimen's own metadata. Nothing else -- and specifically not the Qwen captures,
# artifacts, or per-experiment receipts.
ALLOWED_RECEIPTS = [
    "ORGAN_FRONTIER_MATRIX.json", "ORGAN_LIBRARY.json", "REPRESENTATION_LIBRARY.json",
    "KERNEL_LIBRARY.json", "SUPEROPERATOR_LIBRARY.json", "NOETIC_NEGATIVE_SCIENCE.json",
    "QWEN_TRANSFER_REPORT.json", "MACHINE_GENOME.json", "RUNTIME_GENOME.json",
    "ARCHITECTURE_RECOGNIZER_FIXTURES.json",
]
ALLOWED_TOOLS = ["arch_recognizer.py", "representation_library.py", "negative_science.py",
                 "organ_library.py", "transfer_report.py", "transfer_rehearsal.py"]
# Reading these would be smuggling: they are Qwen's private working state.
FORBIDDEN_PREFIXES = [
    str(Path.home() / "noetic"), str(Path.home() / "models"),
    str(REPO / "workspace"), str(REPO / "artifacts"), str(REPO / "crates"),
]

_reads: list[str] = []


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            _reads.append(os.fspath(args[0]))
        except TypeError:
            pass


def audit(reads):
    """Classify every path the process actually opened."""
    outside, forbidden = [], []
    for r in reads:
        try:
            p = str(Path(r).resolve())
        except Exception:
            unresable = str(r)
            if any(unresable.startswith(f) for f in FORBIDDEN_PREFIXES):
                forbidden.append(unresable)
            continue
        if not p.startswith(str(REPO)) and not p.startswith(str(Path.home())):
            continue                                   # stdlib, site-packages, /dev, /usr
        if any(p.startswith(f) for f in FORBIDDEN_PREFIXES):
            forbidden.append(p)
            continue
        if p.startswith(str(RH)):
            if Path(p).name not in ALLOWED_RECEIPTS:
                outside.append(p)
        elif p.startswith(str(REPO / "tools")):
            # __pycache__ writes are the interpreter's own bookkeeping, including the
            # atomic temp form `x.pyc.<pid>`; they carry no model information.
            if "__pycache__" in p:
                continue
            if Path(p).name not in ALLOWED_TOOLS:
                outside.append(p)
        elif p.startswith(str(REPO)):
            outside.append(p)
    return {"n_opens_recorded": len(reads),
            "n_repo_reads_outside_allowlist": len(set(outside)),
            "reads_outside_allowlist": sorted(set(outside))[:20],
            "n_forbidden_reads": len(set(forbidden)),
            "forbidden_reads": sorted(set(forbidden))[:20],
            "allowlist_receipts": ALLOWED_RECEIPTS,
            "forbidden_prefixes": FORBIDDEN_PREFIXES,
            "clean": not outside and not forbidden}


def rehearse(oxx, repo, rev):
    sys.path.insert(0, str(REPO / "tools/odyssey"))
    sys.path.insert(0, str(REPO / "tools/headless"))
    import arch_recognizer as ar
    import representation_library as rl

    t0 = time.time()
    fx = json.load(open(RH / "ARCHITECTURE_RECOGNIZER_FIXTURES.json"))["fixtures"].get(repo)
    if fx:
        rec = ar.recognize(repo, rev, fx["config"], fx["tensor_names"])
        src = "cached architecture metadata"
    else:
        rec = ar.recognize(repo, rev)
        src = "fetched architecture metadata"
    t_recog = time.time() - t0

    tr = json.load(open(RH / "QWEN_TRANSFER_REPORT.json"))
    fams = rl.build()
    arch_class = rec.get("model_type") or "unknown"

    organs, avoided = [], []
    fam_lib = json.load(open(RH / "ORGAN_FRONTIER_MATRIX.json"))
    measured = {e["organ"]: e for e in fam_lib["organs"] if e["status"] == "MEASURED"}
    kernels = json.load(open(RH / "KERNEL_LIBRARY.json"))["kernels"]

    # Seeding was purely score-driven and MASS-BLIND, which is a defect on tiny organs:
    # the aggressive families always win on score, so for moe_router -- 0.0412% of model
    # mass, where quantizing at all saves 44.6 MiB -- the three seeds were q2_affine,
    # binary_sparse_residual and ternary and the safe option never reached the table.
    # Below the ceiling the density prize is negligible, so leftover_f32 must at least be
    # a candidate. Found by the KernelPlanner: no seeded family had a competent kernel.
    PASSTHROUGH_MASS_CEILING = 0.005
    shares, counts = _organ_mass_shares(rec)

    for o in rec["organs"]:
        name = o["organ"]
        fam_name = ar.ALIAS_TO_FAMILY.get(name, name)
        seed = rl.seed(fams, fam_name, arch_class)
        top = seed["ranked"][:3]
        share = shares.get(name)
        mass_seeded = False
        if (share is not None and share <= PASSTHROUGH_MASS_CEILING
                and not any(t["family"] == "leftover_f32" for t in top)):
            lf = next((t for t in seed["ranked"] if t["family"] == "leftover_f32"), None)
            top = top + [lf if lf else {
                "family": "leftover_f32", "score": 0.0,
                "evidence": "not ranked by the representation library for this organ",
                "warnings": [], "kin_organs_it_failed_on": []}]
            mass_seeded = True
        # kernels already qualified for the seeded representation
        reps = {t["family"] for t in top}
        reuse = [k["kernel_identity"] for k in kernels
                 if k.get("representation_identity") in reps]
        # Branches the prior science actually moved. Counted over the WHOLE ranked list,
        # not just the top 3 -- a family demoted out of the top 3 by a recorded failure is
        # exactly the experiment that does not get run, and counting only survivors would
        # report zero precisely when the transfer worked best.
        for t in seed["ranked"]:
            for w in t["warnings"]:
                avoided.append({"organ": name, "family": t["family"], "warning": w["id"],
                                "model_it_failed_on": w["model"],
                                "reopen_condition": w["reopen_condition"],
                                "in_top3": t in top,
                                "effect": "demoted out of the seed" if t not in top
                                          else "kept but flagged"})
            if t.get("kin_organs_it_failed_on"):
                avoided.append({"organ": name, "family": t["family"],
                                "warning": "KIN_ORGAN_FAILURE",
                                "kin_organs": t["kin_organs_it_failed_on"],
                                "in_top3": t in top,
                                "effect": "demoted: this family failed on an organ of the "
                                          "same operator shape"})
        for e in seed["excluded"]:
            avoided.append({"organ": name, "family": e["family"], "warning": "EXCLUDED",
                            "why": e["why"], "in_top3": False,
                            "effect": "excluded from the search entirely"})
        organs.append({
            "organ": name, "status": o["status"], "confidence": o["confidence"],
            "n_tensors": o["n_tensors"],
            "library_row": ("MEASURED on qwen3.8-27b-abliterated"
                            if fam_name in measured else "no measured row"),
            "organ_mass_share": (round(share, 6) if share is not None else None),
            "organ_params": counts.get(name),
            "mass_aware_seed_added": mass_seeded,
            "mass_aware_seed_reason": (
                f"organ is {share:.4%} of model mass, at or under the "
                f"{PASSTHROUGH_MASS_CEILING:.1%} ceiling, so an f32 passthrough is a "
                f"legitimate candidate and score-only ranking would never surface it"
                if mass_seeded else None),
            "seeded_representations": [{"family": t["family"], "score": t["score"],
                                        "why": t["evidence"]} for t in top],
            "reusable_kernels": sorted(set(reuse))[:6],
            "excluded_families": [e["family"] for e in seed["excluded"]],
        })

    methods = [{"id": e["id"], "inherit": e["inherit"]}
               for e in tr["entries"] if e["id"].startswith("TR-METHOD")]
    machine = json.load(open(RH / "MACHINE_GENOME.json"))
    runtime = json.load(open(RH / "RUNTIME_GENOME.json"))

    return {
        "specimen": {"oxx": oxx, "repo": repo, "revision": rev},
        "architecture_metadata_source": src,
        "architecture_class": arch_class,
        "architectures": rec.get("architectures"),
        "recognition_wall_s": round(t_recog, 3),
        "n_organs_recognized": len(rec["organs"]),
        "n_organs_known": sum(1 for o in rec["organs"] if o["status"] == "KNOWN"),
        "n_organs_declared_unmeasured": sum(1 for o in rec["organs"]
                                            if o["status"] == "DECLARED_UNMEASURED"),
        "n_organs_novel": sum(1 for o in rec["organs"] if o["status"] == "NOVEL"),
        "n_unrecognized_clusters": len(rec["unrecognized"]),
        "organ_plan": organs,
        "methods_inherited": methods,
        "n_methods_inherited": len(methods),
        "prior_failures_that_shape_the_search": avoided,
        "n_prior_failures_applied": len(avoided),
        "device_genome_init": {
            "machine": {k: machine.get(k) for k in
                        ("chipset", "gpu_cores", "unified_memory_bytes", "metal_family")
                        if k in machine},
            "runtime": {k: runtime.get(k) for k in ("runtime", "backend", "profile")
                        if k in runtime},
            "seeded_from": "MACHINE_GENOME.json + RUNTIME_GENOME.json; roofs are NOT copied "
                           "-- the model-reachable roof is per-executable and must be "
                           "remeasured for this specimen",
        },
        "total_wall_s": round(time.time() - t0, 3),
    }


def _organ_mass_shares(rec):
    """Parameter share per organ, from the specimen's own config. Mass is what decides
    whether quantizing an organ can pay, and the seeder had no access to it."""
    import glob
    repo = (rec.get("repo") or "").replace("/", "--")
    cands = glob.glob(f"/Volumes/corpdrive/hawking-modellake/specimens/{repo}@*/**/"
                      "config.json", recursive=True)
    if not cands:
        return {}, {}
    c = json.load(open(cands[0]))
    t = c.get("text_config", c)
    H, L = t.get("hidden_size"), t.get("num_hidden_layers")
    if not H or not L:
        return {}, {}
    E = t.get("num_experts")
    inter = t.get("moe_intermediate_size") or t.get("intermediate_size") or 0
    V = t.get("vocab_size") or c.get("vocab_size") or 0
    kv, hd = t.get("num_key_value_heads", 0), t.get("head_dim", 0)
    q = t.get("num_attention_heads", 0)
    counts = {
        "moe_expert": L * E * 3 * H * inter if E else 0,
        "moe_router": L * E * H if E else 0,
        "gqa_attention": L * (2 * q * hd * H + 2 * kv * hd * H),
        "embed": V * H, "lm_head": V * H, "rmsnorm": L * 2 * H,
    }
    counts = {k: v for k, v in counts.items() if v}
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oxx", default="O005")
    ap.add_argument("--repo", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision", default="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    ap.add_argument("--emit", required=True)
    ap.add_argument("--smuggle-demo", action="store_true",
                    help="deliberately read Qwen scratch and show the audit catching it")
    a = ap.parse_args()

    sys.addaudithook(_hook)
    if a.smuggle_demo:
        p = Path.home() / "noetic/NOETIC_PARENT_A/MIX_REPORT.json"
        if p.exists():
            json.load(open(p))
        au = audit(_reads)
        print(json.dumps({"clean": au["clean"], "n_forbidden_reads": au["n_forbidden_reads"],
                          "forbidden_reads": au["forbidden_reads"][:2]}, indent=1))
        return 0 if not au["clean"] else 1     # the demo PASSES when the audit catches it

    plan = rehearse(a.oxx, a.repo, a.revision)
    au = audit(_reads)
    out = {
        "schema": "hawking.headless.transfer_rehearsal.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/transfer_rehearsal.py",
        "obligation": "G008 — QWEN_TRANSFER_REHEARSAL (directive §15)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "law": "the rehearsal may read only the canonical libraries, the transfer report and "
               "the specimen's own architecture metadata. Everything else is smuggling.",
        "input_audit": au,
        "plan": plan,
        "pass": bool(au["clean"] and plan["n_organs_recognized"] >= 5),
    }
    Path(a.emit).write_text(json.dumps(out, indent=1))
    print(f"organs={plan['n_organs_recognized']} known={plan['n_organs_known']} "
          f"declared={plan['n_organs_declared_unmeasured']} novel={plan['n_organs_novel']} "
          f"methods={plan['n_methods_inherited']} prior_failures_applied={plan['n_prior_failures_applied']} "
          f"recognition={plan['recognition_wall_s']}s audit_clean={au['clean']} pass={out['pass']}")
    for o in plan["organ_plan"]:
        print(f"  {o['organ']:18} {o['status']:20} seed={[s['family'] for s in o['seeded_representations']]} "
              f"kernels={len(o['reusable_kernels'])}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
