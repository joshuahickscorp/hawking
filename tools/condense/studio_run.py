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
FLOORS = "reports/cron/bit_floor_curve.jsonl"     # one floor datapoint per model

# model ladder: label -> (hf dir, params(B), doctor peak GB, solo?, role)
LADDER = [
    ("0.5B", "scratch/qwen-05b", 0.5, 10, False, "lab"),
    ("1.5B", "scratch/qwen-15b", 1.5, 10, False, "lab"),
    ("7B",   "scratch/qwen-7b",  7.0, 40, False, "substrate"),   # the honest mid; 1-bit judged here
    ("14B",  "scratch/qwen-14b", 14.0, 65, False, "payoff"),
    ("32B",  "scratch/qwen-32b", 32.0, 85, True,  "capstone"),   # solo: needs the whole box
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


def _have(label):
    """A model's floor point is done if its receipt + floor row already exist (resumable)."""
    if not os.path.exists(FLOORS):
        return False
    for ln in open(FLOORS):
        try:
            if json.loads(ln).get("model") == label:
                return True
        except Exception:
            pass
    return False


def run_model(label):
    """Run ONE model's full chain serially: bakes+ppl ladder -> recovery stack -> floor search ->
    receipt. Heavy; STUDIO only. Each step shells the real tool and is individually checkpointed."""
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
    log = f"reports/cron/studio_{label}.log"
    out = f"reports/cron/studio_{label}"
    print(f"[{label}] chain start (role={role}, {params}B) -> {log}", file=sys.stderr)
    # Stage 1-2: the full L0-L6 stack via audit_ladder's 'studio' set (bakes + AWQ + mixed + residual
    # + L6 LoRA-KD + L4 block-QAT + L5 GPTQ-Hessian), each checkpointed/guarded, multiwindow ppl +
    # capability tripwire on floor candidates.
    rc = subprocess.run(["python3.12", f"{TC}/audit_ladder.py", mdir, label, "studio", out],
                        env=env).returncode
    # Stage 3-4: pick the floor (lowest eff-bpw <= +2% ppl) + emit a schema-valid receipt.
    subprocess.run(["python3.12", f"{TC}/scaling_law.py", "--floor", label,
                    f"{out}.jsonl", FLOORS, mdir], env=env)
    subprocess.run(["python3.12", f"{TC}/receipt_verify.py", f"receipts/official/{label}-floor.json"],
                   env=env)
    return rc


def run_all():
    """Schedule every model's chain, packed into RAM, then fit the curve."""
    from ram_scheduler import Scheduler, Job
    jobs = [Job(lbl, ["python3.12", f"{TC}/studio_run.py", "--model", lbl],
                est_gb=gb, solo=solo, done_when=None,
                log=f"reports/cron/studio_{lbl}.log")
            for (lbl, _, _, gb, solo, _) in LADDER if not _have(lbl)]
    Scheduler(statusf="reports/cron/studio_sched.status").run(jobs)
    subprocess.run(["python3.12", f"{TC}/scaling_law.py", "--fit", FLOORS])


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
    print("RAM-PACK SCHEDULE ACROSS MODELS (Studio 96GB -> 82GB budget):")
    jobs = [Job(lbl, ["true"], est_gb=gb, solo=solo) for (lbl, _, _, gb, solo, _) in LADDER]
    Scheduler(budget_gb=82).plan(jobs)
    print("\nAfter the last model: scaling_law.py fits floor vs log(N), draws the recovered-vs-PTQ")
    print("band, and extrapolates the 70B/405B floor (T3.1) as a pre-registered prediction.")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--plan"
    if a == "--plan":
        plan()
    elif a == "--model":
        sys.exit(run_model(sys.argv[2]))
    elif a == "--run":
        run_all()
    else:
        print(__doc__)
