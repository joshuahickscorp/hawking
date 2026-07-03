#!/usr/bin/env python3.12
"""studio_run.py — the chained, RAM-packed driver of the bit-floor-vs-scale experiment (plan §4).

This is the one command the Studio runs. It chains, for EACH model in the ladder, the full recovery
stack and a binary search for that model's bit-floor, emits a receipt per floor point, and after the
whole ladder fits the floor-vs-scale curve. The parallelism is at MODEL granularity: each model's
pipeline runs serially inside one job (peak RAM = that model's doctor), and ram_scheduler packs whole
model-pipelines into the 96 GB box (labs+7B together, 14B bigger, 32B solo). Self-dispatching:
  studio_run.py --plan          # dry-run: print the per-model stages + the RAM-pack wave schedule
  studio_run.py --run           # schedule ALL models (packed) then fit the curve  [STUDIO]
  studio_run.py --model 7B      # run ONE model's full chain serially (what --run dispatches) [STUDIO]

Respects the §0 dead-ends and §6 proof discipline: effective bpw only, multiwindow eval, CPU-bf16
production numbers, judge on 7B+ (0.5B/1.5B are lab points, tagged baseline). Heavy stages are
guarded + checkpointed by the underlying tools; this just orders and packs them.
"""
import os, sys, json, subprocess, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
TC = "tools/condense"
REC = "receipts/official"
FLOORS = "reports/cron/bit_floor_curve.jsonl"     # studio-lane floor curve (back-compat default)

def floors_path(set_name="studio"):
    """Per-lane floor file so the studio and subbit lanes don't overwrite each other's curve."""
    return FLOORS if set_name == "studio" else f"reports/cron/bit_floor_{set_name}.jsonl"

# model ladder: label -> (hf dir, params(B), doctor peak GB, solo?, role)
# These DOCTOR resident on 96GB (f16 ~2x params must fit) -> caps at ~32B. This is the bit-floor
# curve substrate; the 100B+ research targets are SERVE-only (FRONTIER below), a different pipeline.
LADDER = [
    ("0.5B", "scratch/qwen-05b", 0.5, 10, False, "lab"),
    ("1.5B", "scratch/qwen-15b", 1.5, 10, False, "lab"),
    ("7B",   "scratch/qwen-7b",  7.0, 40, False, "substrate"),   # the honest mid; 1-bit judged here
    ("14B",  "scratch/qwen-14b", 14.0, 65, False, "payoff"),
    ("32B",  "scratch/qwen-32b", 32.0, 85, True,  "capstone"),   # solo: needs the whole box
]

# FRONTIER — the 100B+ research targets (the real prize). On the M1 Ultra 128 GB box the pivot from
# the M2-Max plan: 235B (~39GB), 405B-dense (~68GB), and 671B@1.0 (~84GB) all fit RESIDENT under the
# ~112 GB weight budget, so the OOC expert pager (the hardest serve-build item, Type-1 dead in the
# free-RAM regime) is NOT needed for these. The pipeline condenses block-wise-streamed to a serve-fit
# .tq, then serves RESIDENT; quality is the native .tq serve (gated on the serve build). Only 744B and
# the deeper frontier (1T/3T) need the SSD out-of-core path. (f16 parent is streamed, never resident.)
# label, hf dir, total_b, active_b (None=dense), serve_bpw, moe?, role, HF id
FRONTIER = [
    ("235B-A22B", "scratch/qwen3-235b-a22b", 235.0, 22.0, 1.34, True,  "moe-resident",
     "Qwen/Qwen3-235B-A22B"),                                            # ~39GB @1.34  RESIDENT (comfy)
    ("405B",      "scratch/llama31-405b",    405.0, None, 1.34, False, "dense-resident",
     "meta-llama/Llama-3.1-405B-Instruct"),                             # ~68GB @1.34  RESIDENT (no pager)
    ("671B",      "scratch/deepseek-v3",     671.0, 37.0, 1.00, True,  "moe-capstone",
     "deepseek-ai/DeepSeek-V3"),                                        # ~84GB @1.0   RESIDENT (the prize)
    ("744B",      "scratch/glm-744b",        744.0, 32.0, 0.75, True,  "moe-stretch",
     "zai-org/GLM-4.5"),                                                # ~70GB @0.75  RESIDENT (research)
]

# the recovery stack run per model, cheapest-first (plan §2). Each entry: (stage, tool, note).
STACK = [
    ("L0 calib",      f"{TC}/calib_build.py",     "domain-matched corpus (input to all below)"),
    ("L1 AWQ",        f"{TC}/awq_bake.py",        "alpha=0.5 pre-scale + bake"),
    ("L2 mixed-prec", f"{TC}/mixed_precision.py", "output-sensitivity bit allocation"),
    ("L3 residual",   f"{TC}/residual_bake.py",   "full-rank residual (train-free ~1:1)"),
    ("L4 block-QAT",  f"{TC}/doctor_blockwise.py","full-rank per-layer QAT  [the LoRA-plateau fix]"),
    ("L5 GPTQ-Hess",  f"{TC}/doctor_strand.py",   "codec-native error-feedback [sub-residual edge]"),
    ("L6 deep-KD",    f"{TC}/doctor_lora.py",     "logit/feature KD polish on the full-rank base"),
]


def _have(label, set_name="studio"):
    """A model's floor point is done if its floor row already exists in that lane (resumable)."""
    fp = floors_path(set_name)
    if not os.path.exists(fp):
        return False
    for ln in open(fp):
        try:
            if json.loads(ln).get("model") == label:
                return True
        except Exception:
            pass
    return False


def run_model(label, set_name="studio"):
    """Run ONE model's full chain serially: (SUBBIT-0 gate) -> bake+recovery stack -> floor search ->
    receipt. Heavy; STUDIO only. set_name selects audit_ladder's config set ('studio' = the L0-L6
    bit-floor stack; 'subbit' = the sub-1/sub-2-bit frontier lane). Each step is checkpointed."""
    row = next(r for r in LADDER if r[0] == label)
    _, mdir, params, _, _, role = row
    if not os.path.isdir(mdir):
        print(f"[{label}] SKIP — parent not staged at {mdir} (download on the Studio)", file=sys.stderr)
        return 2
    ncpu = str(os.cpu_count() or 8)
    env = {**os.environ, "DOCTOR_DEVICE": "cpu", "DOCTOR_DTYPE": "bfloat16", "STRAND_NO_GPU": "1",
           "MULTIWINDOW": "4", "STUDIO_TRIPWIRE": "1", "DOCTOR_THREADS": ncpu,
           "OMP_NUM_THREADS": ncpu, "MKL_NUM_THREADS": ncpu, "VECLIB_MAXIMUM_THREADS": ncpu,
           # generous swap leashes for the big box (§3); the 18GB 6000MB death must not recur <32B
           "DOCTOR_SWAP_CEIL": os.environ.get("DOCTOR_SWAP_CEIL", "60000"),
           "DOCTOR_SWAP_HARD_CEIL": os.environ.get("DOCTOR_SWAP_HARD_CEIL", "80000")}
    log = f"reports/cron/{set_name}_{label}.log"
    out = f"reports/cron/{set_name}_{label}"
    print(f"[{label}] {set_name} chain start (role={role}, {params}B) -> {log}", file=sys.stderr)
    # Architecture coverage: which Doctor levers are arch-compatible for this model (dense here
    # today; the same check that flags Mamba2/RWKV-7 SSM state + MoE per-expert applicability).
    subprocess.run(["python3.12", f"{TC}/arch_coverage.py", mdir, label], env=env)
    # SUBBIT-0 GATE: measure the entropy/side-info floor first. If sub-1-bit dense is DEAD by the
    # floor, the subbit lane still runs (MoE/residual survive) but the gate is on record per model.
    if set_name == "subbit":
        subprocess.run(["python3.12", f"{TC}/subbit_measure.py", mdir, label], env=env)
        # SUBBIT-4 probe: per-expert sensitivity decides MoE sub-bit allocation (gated to MoE dirs).
        if any(t in label.lower() for t in ("moe", "a22b", "a3b", "deepseek", "mixtral", "glm")):
            subprocess.run(["python3.12", f"{TC}/expert_sensitivity.py", mdir, "--label", label,
                            "--bits", "1,2"], env=env)
    # Stage 1-2: the chosen stack via audit_ladder (bakes + AWQ + mixed + residual + L6 LoRA-KD +
    # L4 block-QAT + L5 GPTQ-Hessian), each checkpointed/guarded, multiwindow ppl + tripwire.
    rc = subprocess.run(["python3.12", f"{TC}/audit_ladder.py", mdir, label, set_name, out],
                        env=env).returncode
    # Stage 3-4: pick the floor (lowest eff-bpw <= +2% ppl) + emit a schema-valid receipt.
    subprocess.run(["python3.12", f"{TC}/scaling_law.py", "--floor", label,
                    f"{out}.jsonl", floors_path(set_name), mdir], env=env)
    subprocess.run(["python3.12", f"{TC}/receipt_verify.py", f"receipts/official/{label}-floor.json"],
                   env=env)
    return rc


def run_all(set_name="studio"):
    """Schedule every model's chain, packed into RAM, then fit the curve."""
    from ram_scheduler import Scheduler, Job
    jobs = [Job(lbl, ["python3.12", f"{TC}/studio_run.py", "--model", lbl, set_name],
                est_gb=gb, solo=solo, done_when=None,
                log=f"reports/cron/{set_name}_{lbl}.log")
            for (lbl, _, _, gb, solo, _) in LADDER if not _have(lbl, set_name)]
    Scheduler(statusf=f"reports/cron/{set_name}_sched.status").run(jobs)
    subprocess.run(["python3.12", f"{TC}/scaling_law.py", "--fit", floors_path(set_name)])


def run_frontier(label):
    """SERVE-oriented frontier pipeline for a 100B+ model (the real research prize). The doctor does
    NOT fit (f16 2x resident overflows 96GB), so this runs what DOES on streamed shards: the SUBBIT-0
    entropy floor + per-expert sensitivity (MoE) + the serve-fit record. The block-wise condense to a
    serve-fit .tq, the native-serve quality number, and the RAM-cliff tps demo are the Rust serve
    build (read_strand into the serve binary + the per-expert .tq writer) — emitted as gated steps."""
    row = next((r for r in FRONTIER if r[0] == label), None)
    if not row:
        print(f"[frontier] unknown {label}", file=sys.stderr); return 2
    _, mdir, total, active, bpw, moe, role, hf_id = row
    artifact = round(total * bpw / 8.0, 1)
    # M1 Ultra 128 GB weight budget ~112 GB (re-derived from the M2-Max 84 GB). The pivot: 235B (~39GB),
    # 405B-dense (~68GB), and 671B@1.0 (~84GB) all fit RESIDENT here, so the OOC expert pager (the
    # hardest serve-build item, and Type-1 dead in the free-RAM regime) is NOT on the critical path.
    WEIGHT_BUDGET = 112.0
    fits = artifact <= WEIGHT_BUDGET
    resident = "RESIDENT (no pager)" if fits else "OVERFLOW (SSD-bound, deep frontier)"
    ncpu = str(os.cpu_count() or 8)
    env = {**os.environ, "DOCTOR_DEVICE": "cpu", "DOCTOR_DTYPE": "bfloat16", "STRAND_NO_GPU": "1",
           "OMP_NUM_THREADS": ncpu, "MKL_NUM_THREADS": ncpu, "VECLIB_MAXIMUM_THREADS": ncpu}
    print(f"[frontier] {label} ({total}B{f', act {active}B MoE' if moe else ' dense'}, role={role}) "
          f"-> {bpw} bpw = {artifact}GB ({resident} on 128GB)", file=sys.stderr)
    if not os.path.isdir(mdir):
        print(f"[frontier] {label} NOT staged at {mdir}. Fastest-SOTA procurement (hf_transfer + hf_xet, "
              f"link-bound): python3.12 {TC}/procure.py {label}  (8 TB SSD; ~3h/671B on gigabit)", file=sys.stderr)
        return 2
    # Auto mode: recommend the bit format + serve regime (RESIDENT / MOE-PAGED / DENSE-OOC) and show
    # the device size ceiling before condensing (the "how big can we pull in" advisor).
    ab = ["python3.12", f"{TC}/auto_bits.py", "--params", str(total), "--label", label]
    if active:
        ab += ["--active", str(active)]
    subprocess.run(ab, env=env)
    sf = ["python3.12", f"{TC}/size_frontier.py", str(total), "--bpw", str(bpw)]
    if active:
        sf += ["--active", str(active)]
    subprocess.run(sf, env=env)
    # The Doctor registry: the auto-composed recovery chain (L0-L6 + per-expert) for this model/bpw.
    dr = ["python3.12", f"{TC}/doctor_registry.py", "--select", str(total), str(bpw)]
    if moe:
        dr.append("--moe")
    subprocess.run(dr, env=env)
    # Runs on streamed shards (no full f16 resident): the entropy floor + the MoE expert decision.
    subprocess.run(["python3.12", f"{TC}/subbit_measure.py", mdir, label], env=env)
    # Architecture coverage: real state geometry (Mamba2/RWKV-7 flat state) + which Doctor levers
    # are arch-compatible for this model, so the selector above never wastes a bake on one that isn't.
    subprocess.run(["python3.12", f"{TC}/arch_coverage.py", mdir, label], env=env)
    if moe:
        subprocess.run(["python3.12", f"{TC}/expert_sensitivity.py", mdir, "--label", label,
                        "--bits", "1,2"], env=env)
        # Hot-expert cache policy: simulate hit-rate/blended-tok/s across cache sizes so the OOC
        # pager's cache size is chosen from a measured sweep, not a guess. n_experts best-effort
        # from config; falls back to a documented default sized to this model's active fraction.
        n_experts = 0
        try:
            n_experts = json.load(open(os.path.join(mdir, "config.json"))).get(
                "n_routed_experts") or json.load(open(os.path.join(mdir, "config.json"))).get(
                "num_local_experts") or 0
        except Exception:
            pass
        n_experts = n_experts or max(8, round(total / max(1.0, active or 1.0)) * 4)
        expert_size_b = total / max(1, n_experts)          # params per single expert
        active_k = max(1, round((active or total * 0.05) / max(0.1, expert_size_b)))
        active_gb_tok = (active or total * 0.05) * bpw / 8.0   # TOTAL active bytes/token (all active experts)
        subprocess.run(["python3.12", f"{TC}/expert_cache_policy.py", "--sim", str(n_experts),
                        str(active_k), "--active-gb", str(round(active_gb_tok, 3))], env=env)
    # the serve-build steps (Rust, gated): block-wise condense + native-serve quality + RAM-cliff tps
    rec = {"model": label, "total_b": total, "active_b": active, "moe": moe, "role": role,
           "serve_bpw": bpw, "artifact_gb": artifact, "serve_fits_resident_112gb": fits,
           "resident_no_pager": fits,
           "condense_cmd": f"# block-wise streamed single-bake (+per-expert if MoE) to {label}.tq @ {bpw}bpw",
           "serve_quality_gated_on": "read_strand wired into hawking-serve binary + native .tq GEMV",
           "ram_cliff_demo": f"serve {label}.tq ({artifact}GB resident) vs Q4_K ({round(total*4.5/8)}GB, overflows->swap)"}
    os.makedirs("reports/condense", exist_ok=True)
    json.dump(rec, open(f"reports/condense/{label}_frontier.json", "w"), indent=2)
    print(f"[frontier] {label} serve-fit recorded; quality+cliff GATED on the native serve build",
          file=sys.stderr)
    return 0


def run_frontier_all():
    """Frontier models are each ~box-filling -> run sequentially (the scheduler would serialize them
    anyway). Skips unstaged. The serve build is the gate on the quality/cliff numbers."""
    for (lbl, *_rest) in FRONTIER:
        run_frontier(lbl)


def plan():
    sys.path.insert(0, TC)
    from ram_scheduler import Scheduler, Job
    print("=" * 78)
    print("CHAINED PER-MODEL PIPELINE (each runs serially inside one scheduler job):")
    for lbl, mdir, params, gb, solo, role in LADDER:
        staged = "staged" if os.path.isdir(mdir) else "NEEDS DOWNLOAD"
        print(f"\n  {lbl} ({params}B, role={role}, doctor~{gb}GB{', SOLO' if solo else ''}, {staged})")
        print(f"    1. bake+ppl ladder (audit_ladder, frontier set: AWQ/mixed/residual/outlier + multiwindow ppl)")
        for stage, tool, note in STACK:
            print(f"    2. {stage:13s} {note}")
        print(f"    3. floor-search: lowest eff-bpw at <=+2% ppl AND multi_eval tripwire pass")
        print(f"    4. emit receipt (repro level; 0.5B/1.5B tagged baseline, never set the verdict)")
    print("\n" + "=" * 78)
    print("RAM-PACK SCHEDULE ACROSS MODELS (M1 Ultra 128GB -> 110GB budget):")
    jobs = [Job(lbl, ["true"], est_gb=gb, solo=solo) for (lbl, _, _, gb, solo, _) in LADDER]
    Scheduler(budget_gb=110).plan(jobs)   # M1 Ultra 128GB -> ~110GB pack budget (was 82 on M2-Max-96)
    print("\nAfter the last model: scaling_law.py fits floor vs log(N), draws the recovered-vs-PTQ")
    print("band, and extrapolates the 70B/405B floor (T3.1) as a pre-registered prediction.")


SPEC_TARGETS = ["7B", "32B"]   # condensed substrate + capstone to revive spec-decode on


def go():
    """THE GO BUTTON — run the ENTIRE studio frontier program end-to-end, RAM-packed, continuous
    (no pauses), resumable via per-lane floor files + receipts. On the Studio, this is the one command:
        python3.12 tools/condense/studio_run.py go
    P1 CONDENSE  bit-floor-vs-scale curve (the headline science) across the ladder.
    P2 SUBBIT    sub-1-bit frontier lane (PTQ1.61 + residual + 1-bit codec-native/recover), SUBBIT-0 gated.
    P3 SPEC      revive speculative decoding on the condensed substrate + capstone (latency x density).
    P4 SYNTH     fit both curves + extrapolate 70B/405B; receipts are the record."""
    print("=" * 78); print("HAWKING STUDIO FRONTIER — GO (continuous, resumable, RAM-packed)"); print("=" * 78)
    print("\n### P0 CODEC TRIAGE — score candidate codec designs for decode parallelism ###", file=sys.stderr)
    subprocess.run(["python3.12", f"{TC}/codec_parallelism.py", "--catalog"])
    print("\n### P1 CONDENSE — bit-floor-vs-scale curve ###", file=sys.stderr)
    run_all("studio")
    print("\n### P2 SUBBIT — sub-1-bit frontier lane ###", file=sys.stderr)
    run_all("subbit")
    print("\n### P3 SPEC — speculative decode revival on condensed models ###", file=sys.stderr)
    for lbl in SPEC_TARGETS:
        row = next((r for r in LADDER if r[0] == lbl), None)
        if row and os.path.isdir(row[1]):
            subprocess.run(["python3.12", f"{TC}/spec_revive.py", row[1], lbl])
        else:
            print(f"[spec] {lbl} parent not staged — skipping", file=sys.stderr)
    print("\n### P4 FRONTIER — the 100B+ research prize (serve-oriented; runs what's staged) ###",
          file=sys.stderr)
    run_frontier_all()
    # ---- the VALUE layer: prove capability, defend the wedge, measure the cliff+energy, map the codec ----
    eval_targets = [(l, m, p) for (l, m, p, g, s, r) in LADDER if l not in ("0.5B", "1.5B")]
    print("\n### P5 EVAL — capability + NIAH + LONG-CONTEXT (extend + aggressive KV frontier) ###", file=sys.stderr)
    for lbl, mdir, params in eval_targets:
        if os.path.isdir(mdir):
            subprocess.run(["python3.12", f"{TC}/eval_suite.py", "--model", mdir, "--label", lbl])
            # long-context: YaRN extension + KV-RAM wall + SSM moat ...
            subprocess.run(["python3.12", f"{TC}/ctx_extend.py", mdir, lbl])
            # ... the AGGRESSIVE KV frontier (int2/trellis KV, SSD-paging, evict, SSM) per regime ...
            subprocess.run(["python3.12", f"{TC}/kv_frontier.py", mdir, lbl, str(params)])
            # ... and STKV, the Hawking-specific hybrid that consolidates all four levers (trellis
            # warm + int8 sink + SSD/SSM tail) into one tiered policy — exact recall + unbounded reach.
            subprocess.run(["python3.12", f"{TC}/kv_hybrid.py", mdir, lbl, str(params)])
    print("\n### P6 BASELINE — wedge gate: IQ1_S/IQ2/MLX-4bit head-to-head at matched bpw ###", file=sys.stderr)
    for lbl, mdir, _params in eval_targets:   # SPINE-0: eval_targets are 3-tuples (label, dir, params)
        if os.path.isdir(mdir):
            subprocess.run(["python3.12", f"{TC}/bench_baselines.py", "--model", mdir, "--label", lbl,
                            "--audit-jsonl", f"reports/cron/studio_{lbl}.jsonl"])
    print("\n### P7 CLIFF — RAM-cliff tok/s + energy J/tok (the headline + the energy moat) ###", file=sys.stderr)
    subprocess.run(["python3.12", f"{TC}/ramcliff_bench.py", "--all"])
    print("\n### P8 CODEC — STRAND vs QTIP/QuIP#/AQLM bakeoff (where we rank) ###", file=sys.stderr)
    for lbl, mdir, _params in eval_targets[:1]:   # one representative (7B) sets the codec rank
        if os.path.isdir(mdir):
            subprocess.run(["python3.12", f"{TC}/codec_bakeoff.py", "--model", mdir, "--label", lbl])
    print("\n### P9 SYNTH + SCORECARD — fit curves + the populated competitive matrix ###", file=sys.stderr)
    subprocess.run(["python3.12", f"{TC}/scaling_law.py", "--fit", floors_path("studio")])
    subprocess.run(["python3.12", f"{TC}/scaling_law.py", "--fit", floors_path("subbit")])
    subprocess.run(["python3.12", f"{TC}/scorecard.py"])
    print("\nGO COMPLETE — SCORECARD at reports/condense/SCORECARD.md; receipts in receipts/official/; "
          "curves in reports/cron/bit_floor_*.jsonl; eval/cliff/codec/frontier in reports/condense/", file=sys.stderr)


def go_plan():
    """Dry overview of the whole GO program (run nothing heavy)."""
    plan()
    print("\n" + "=" * 78); print("FULL GO PROGRAM (studio_run.py go):")
    print("  P0 CODEC     codec_parallelism.py --catalog -> triage new codec designs before Rust build time")
    print("  P1 CONDENSE  run_all('studio')  -> bit_floor_curve.jsonl + receipts")
    print("  P2 SUBBIT    run_all('subbit')  -> SUBBIT-0 gate + sub-1-bit lane -> bit_floor_subbit.jsonl")
    print("  P3 SPEC      spec_revive.py on " + ", ".join(SPEC_TARGETS) + " (lossless gate -> capture-retrain "
          "-> accept -> governor)")
    print("  P4 FRONTIER  run_frontier_all() -> 100B+ research prize (235B-A22B/405B/671B/744B)")
    print("  P5 EVAL      eval_suite.py (capability + NIAH) + ctx_extend.py (YaRN long-ctx + KV-RAM + SSM moat)")
    print("  P6 BASELINE  bench_baselines.py -> wedge gate vs IQ1_S/IQ2/MLX-4bit at matched bpw")
    print("  P7 CLIFF     ramcliff_bench.py --all -> RAM-cliff tok/s + energy J/tok (headline + energy moat)")
    print("  P8 CODEC     codec_bakeoff.py -> STRAND vs QTIP/QuIP#/AQLM (the codec rank map)")
    print("  P9 SCORECARD scorecard.py -> the POPULATED competitive matrix (no WIN cell without a receipt)")
    for lbl in SPEC_TARGETS:
        subprocess.run(["python3.12", f"{TC}/spec_revive.py", "--plan", lbl])


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--plan"
    if a == "go":
        go()
    elif a == "--go-plan":
        go_plan()
    elif a == "--plan":
        plan()
    elif a == "--model":
        # studio_run.py --model <label> [set]   (set = studio | subbit)
        sys.exit(run_model(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "studio"))
    elif a == "--run":
        run_all("studio")
    elif a == "--subbit":
        # studio_run.py --subbit <label>   (one model, sub-1-bit lane, with the SUBBIT-0 gate)
        sys.exit(run_model(sys.argv[2], "subbit"))
    elif a == "--subbit-run":
        run_all("subbit")            # schedule the whole ladder through the sub-1-bit lane
    elif a == "--frontier":
        sys.exit(run_frontier(sys.argv[2]))   # one 100B+ model (serve-oriented)
    elif a == "--frontier-run":
        run_frontier_all()           # all staged 100B+ research targets
    else:
        print(__doc__)
