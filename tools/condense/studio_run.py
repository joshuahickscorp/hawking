#!/usr/bin/env python3.12
"""studio_run.py — the chained, RAM-packed driver of the bit-floor-vs-scale experiment (plan §4).

This is the one command the Studio runs. It chains, for EACH model in the ladder, the safe recovery
stack and a binary search for that model's bit-floor, emits a receipt per floor point, and after the
whole ladder fits the floor-vs-scale curve. The parallelism is at MODEL granularity: each model's
pipeline runs serially inside one job (peak RAM = that model's doctor), and ram_scheduler packs whole
model-pipelines into the M3 Ultra 96 GB Studio. The default interactive-safe ladder runs the staged
0.5B/1.5B/7B wave, then 14B solo when present; 32B is held until a streamed path or an explicit,
measured memory waiver exists. Self-dispatching:
  studio_run.py --plan          # dry-run: print the per-model stages + the RAM-pack wave schedule
  studio_run.py --run           # schedule ALL models (packed) then fit the curve  [STUDIO]
  studio_run.py --model 7B      # run ONE model's full chain serially (what --run dispatches) [STUDIO]

Respects the §0 dead-ends and §6 proof discipline: effective bpw only, multiwindow eval, CPU-bf16
production numbers, judge on 7B+ (0.5B/1.5B are lab points, tagged baseline). Heavy stages are
guarded + checkpointed by the underlying tools. This driver also keeps a durable phase ledger so
an interruption or normal shutdown resumes at the first incomplete phase.
"""
import datetime
import os
import signal
import shutil
import sys
import json
import subprocess
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
TC = "tools/condense"
REC = "receipts/official"
FLOORS = "reports/cron/bit_floor_curve.jsonl"     # studio-lane floor curve (back-compat default)
RUN_STATE = pathlib.Path("reports/cron/studio_run_state.json")
RUN_PID = pathlib.Path("reports/cron/studio_run.pid")
DRAIN_REQUEST = pathlib.Path("reports/cron/studio_drain.request")
RUN_LOG = pathlib.Path("reports/cron/studio_run.log")
sys.path.insert(0, TC)
from studio_manifest import DEFAULT_HARDWARE, FRONTIER_MODELS, frontier_by_label, frontier_labels


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path, value):
    """Durably replace a JSON checkpoint; a power cut leaves the previous complete file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _load_run_state():
    try:
        with open(RUN_STATE) as f:
            state = json.load(f)
        if state.get("schema") == "hawking.studio_run_state.v1":
            return state
    except Exception:
        pass
    return {
        "schema": "hawking.studio_run_state.v1",
        "hardware_profile": DEFAULT_HARDWARE.name,
        "created_at": _now(),
        "updated_at": _now(),
        "phases": {},
    }


def _set_phase(name, status, **extra):
    state = _load_run_state()
    row = dict(state.setdefault("phases", {}).get(name, {}))
    row.update(extra)
    row["status"] = status
    row["updated_at"] = _now()
    if status == "running":
        row["started_at"] = row.get("started_at") or row["updated_at"]
        row.pop("ended_at", None)
    elif status in ("pass", "failed", "interrupted", "skipped"):
        row["ended_at"] = row["updated_at"]
    state["phases"][name] = row
    state["updated_at"] = row["updated_at"]
    _atomic_json(RUN_STATE, state)


def _phase_done(name):
    return _load_run_state().get("phases", {}).get(name, {}).get("status") in ("pass", "skipped")


def _draining():
    return DRAIN_REQUEST.exists()


def _checkpointed_phase(name, fn):
    """Run one phase once; only a durable pass/skipped record is trusted on resume."""
    if _phase_done(name):
        print(f"[studio] {name} checkpoint PASS — skip", file=sys.stderr)
        return 0
    if _draining():
        _set_phase(name, "interrupted", reason="drain requested before launch")
        return 130
    _set_phase(name, "running", pid=os.getpid())
    try:
        rc = int(fn() or 0)
    except KeyboardInterrupt:
        rc = 130
    except Exception as exc:
        _set_phase(name, "failed", error=f"{type(exc).__name__}: {exc}")
        print(f"[studio] {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if rc == 0:
        _set_phase(name, "pass", returncode=0)
    elif rc == 130 or _draining():
        _set_phase(name, "interrupted", returncode=rc, reason="drain/signal")
    else:
        _set_phase(name, "failed", returncode=rc)
    return rc


def _checkpointed_call(key, argv, env=None, acceptable=(0,)):
    """Checkpoint a command within its phase and terminate it cleanly when drain is requested."""
    phase = _load_run_state().get("phases", {}).get(key, {})
    if phase.get("status") == "pass":
        print(f"[studio] {key} checkpoint PASS — skip", file=sys.stderr)
        return 0
    _set_phase(key, "running", command=list(argv), pid=os.getpid())
    proc = subprocess.Popen(argv, env=env, start_new_session=True)
    while proc.poll() is None:
        if _draining():
            print(f"[studio] drain: SIGTERM {key} pid={proc.pid}", file=sys.stderr)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                print(f"[studio] drain grace expired: SIGKILL {key}", file=sys.stderr)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            _set_phase(key, "interrupted", returncode=proc.wait(), reason="drain requested")
            return 130
        time.sleep(1)
    rc = proc.returncode
    if rc in acceptable:
        _set_phase(key, "pass", returncode=rc)
        return 0
    _set_phase(key, "failed", returncode=rc)
    return rc


def _model_complete_path(label, set_name):
    return pathlib.Path(f"reports/cron/{set_name}_{label}.complete.json")

def floors_path(set_name="studio"):
    """Per-lane floor file so the studio and subbit lanes don't overwrite each other's curve."""
    return FLOORS if set_name == "studio" else f"reports/cron/bit_floor_{set_name}.jsonl"

# model ladder: label -> (hf dir, params(B), doctor peak GB, solo?, role)
# Measured/estimated peaks for the M3 Ultra 96 GB box. The scheduler's interactive-safe budget is
# lower than physical RAM so ChatGPT/Codex and macOS stay responsive. 32B remains listed for the
# research plan but is deliberately over budget and will not launch without an explicit override.
LADDER = [
    ("0.5B", "scratch/qwen-05b", 0.5, 10, False, "lab"),
    ("1.5B", "scratch/qwen-15b", 1.5, 10, False, "lab"),
    ("7B",   "scratch/qwen-7b",  7.0, 40, False, "substrate"),   # the honest mid; 1-bit judged here
    ("14B",  "scratch/qwen-14b", 14.0, 65, True,  "payoff-solo"),
    ("32B",  "scratch/qwen-32b", 32.0, 85, True,  "gated-streamed-capstone"),
]

# FRONTIER — the 100B+ research targets (the real prize). Kept in studio_manifest.py so procurement,
# RAM-cliff, and docs share one set of hardware/model facts. On this 96 GB box, only targets whose
# artifacts fit DEFAULT_HARDWARE.weight_budget_gb are resident candidates; the rest require a real
# paging/streaming path. Parent downloads are always governed separately by current free disk.
FRONTIER = [
    (m.label, m.local_dir, m.total_b, m.active_b, m.serve_bpw, m.moe, m.role, m.hf_id)
    for m in FRONTIER_MODELS
]

# the recovery stack run per model, cheapest-first (plan §2). Each entry: (stage, tool, note).
STACK = [
    ("L0 calib",      f"{TC}/calib_build.py",     "domain-matched corpus (input to all below)"),
    ("L1 AWQ",        f"{TC}/awq.py bake",        "alpha=0.5 pre-scale + bake"),
    ("L2 mixed-prec", f"{TC}/mixed_precision.py", "output-sensitivity bit allocation"),
    ("L3 residual",   f"{TC}/residual.py bake",   "full-rank residual (train-free ~1:1)"),
    ("L4 block-QAT",  f"{TC}/doctor.py blockwise","GATED: sharded/dtype/per-layer-resume proof required"),
    ("L5 GPTQ-Hess",  f"{TC}/doctor.py strand",   "GATED: requested-model + durable-state proof required"),
    ("L6 deep-KD",    f"{TC}/doctor.py lora",     "logit/feature KD polish on the full-rank base"),
]


def _have(label, set_name="studio"):
    """Trust a floor only when the model completion manifest says audit+floor+receipt passed."""
    complete_path = _model_complete_path(label, set_name)
    try:
        with open(complete_path) as f:
            complete = json.load(f)
        if not (complete.get("schema") == "hawking.studio_model_complete.v1"
                and complete.get("status") == "pass"
                and complete.get("model") == label
                and complete.get("lane") == set_name):
            return False
    except Exception:
        return False
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
    _, mdir, params, est_gb, _, role = row
    if not os.path.isdir(mdir):
        print(f"[{label}] SKIP — parent not staged at {mdir} (download on the Studio)", file=sys.stderr)
        return 2
    process_budget = float(getattr(DEFAULT_HARDWARE, "process_budget_gb",
                                   DEFAULT_HARDWARE.weight_budget_gb))
    if est_gb > process_budget and os.environ.get("HAWKING_STUDIO_ALLOW_OVER_BUDGET") != "1":
        print(f"[{label}] BLOCKED — estimated {est_gb:.0f}GB exceeds interactive-safe "
              f"{process_budget:.0f}GB; use a streamed path or a measured explicit waiver",
              file=sys.stderr)
        return 3
    ncpu = str(os.cpu_count() or 8)
    env = {**os.environ, "DOCTOR_DEVICE": "cpu", "DOCTOR_DTYPE": "bfloat16", "STRAND_NO_GPU": "1",
           "MULTIWINDOW": "4", "STUDIO_TRIPWIRE": "1", "DOCTOR_THREADS": ncpu,
           "OMP_NUM_THREADS": ncpu, "MKL_NUM_THREADS": ncpu, "VECLIB_MAXIMUM_THREADS": ncpu,
           # Stop launching at yellow pressure and checkpoint at red; do not make tens of GB of swap.
           "DOCTOR_SWAP_CEIL": os.environ.get("DOCTOR_SWAP_CEIL", "2000"),
           "DOCTOR_SWAP_HARD_CEIL": os.environ.get("DOCTOR_SWAP_HARD_CEIL", "6000")}
    log = f"reports/cron/{set_name}_{label}.log"
    out = f"reports/cron/{set_name}_{label}"
    audit_set = f"{set_name}_full" if os.environ.get("HAWKING_STUDIO_RESEARCH_FULL") == "1" else set_name
    print(f"[{label}] {set_name} chain start (audit_set={audit_set}, role={role}, {params}B) -> {log}",
          file=sys.stderr)
    # Architecture coverage: which Doctor levers are arch-compatible for this model (dense here
    # today; the same check that flags Mamba2/RWKV-7 SSM state + MoE per-expert applicability).
    rc = subprocess.run(["python3.12", f"{TC}/arch_coverage.py", mdir, label], env=env).returncode
    if rc != 0:
        return rc
    # SUBBIT-0 GATE: measure the entropy/side-info floor first. If sub-1-bit dense is DEAD by the
    # floor, the subbit lane still runs (MoE/residual survive) but the gate is on record per model.
    if set_name == "subbit":
        rc = subprocess.run(["python3.12", f"{TC}/subbit.py", "measure", mdir, label], env=env).returncode
        if rc != 0:
            return rc
        # SUBBIT-4 probe: per-expert sensitivity decides MoE sub-bit allocation (gated to MoE dirs).
        if any(t in label.lower() for t in ("moe", "a22b", "a3b", "deepseek", "mixtral", "glm")):
            rc = subprocess.run(["python3.12", f"{TC}/expert.py", "sensitivity", mdir, "--label", label,
                                 "--bits", "1,2"], env=env).returncode
            if rc != 0:
                return rc
    # Stage 1-2: the chosen stack via audit_ladder (bakes + AWQ + mixed + residual + L6 LoRA-KD +
    # L4 block-QAT + L5 GPTQ-Hessian), each checkpointed/guarded, multiwindow ppl + tripwire.
    rc = subprocess.run(["python3.12", f"{TC}/audit_ladder.py", mdir, label, audit_set, out],
                        env=env).returncode
    if rc != 0:
        print(f"[{label}] audit failed rc={rc}; floor/receipt deliberately not emitted", file=sys.stderr)
        return rc
    # Stage 3-4: pick the floor (lowest eff-bpw <= +2% ppl) + emit a schema-valid receipt.
    rc = subprocess.run(["python3.12", f"{TC}/scaling_law.py", "--floor", label,
                         f"{out}.jsonl", floors_path(set_name), mdir], env=env).returncode
    if rc != 0:
        return rc
    receipt = f"receipts/official/{label}-floor.json"
    rc = subprocess.run(["python3.12", f"{TC}/receipt_verify.py", receipt], env=env).returncode
    if rc != 0:
        return rc
    _atomic_json(_model_complete_path(label, set_name), {
        "schema": "hawking.studio_model_complete.v1",
        "status": "pass",
        "completed_at": _now(),
        "hardware_profile": DEFAULT_HARDWARE.name,
        "model": label,
        "lane": set_name,
        "audit_set": audit_set,
        "model_dir": mdir,
        "audit_jsonl": f"{out}.jsonl",
        "floor_jsonl": floors_path(set_name),
        "receipt": receipt,
    })
    return 0


def run_all(set_name="studio"):
    """Schedule every model's chain, packed into RAM, then fit the curve."""
    from ram_scheduler import Scheduler, Job
    jobs = []
    for lbl, mdir, _params, gb, solo, _role in LADDER:
        if _have(lbl, set_name):
            continue
        if not os.path.isdir(mdir):
            print(f"[{set_name}] {lbl} not staged — deferred", file=sys.stderr)
            continue
        jobs.append(Job(
            lbl,
            ["python3.12", f"{TC}/studio_run.py", "--model", lbl, set_name],
            est_gb=gb,
            solo=solo,
            done_when=str(_model_complete_path(lbl, set_name)),
            log=f"reports/cron/{set_name}_{lbl}.log",
            checkpoint_safe=True,
        ))
    scheduler = Scheduler(
        budget_gb=float(getattr(DEFAULT_HARDWARE, "process_budget_gb",
                                DEFAULT_HARDWARE.weight_budget_gb)),
        statusf=f"reports/cron/{set_name}_sched.status",
        drain_file=str(DRAIN_REQUEST),
    )
    results = scheduler.run(jobs)
    if _draining():
        return 130
    failed = {name: rc for name, rc in results.items() if rc not in (0, 2)}
    if failed:
        print(f"[{set_name}] model failures/blocks: {failed}", file=sys.stderr)
        return 1
    floor = floors_path(set_name)
    if not os.path.exists(floor):
        print(f"[{set_name}] no completed floor rows yet; leaving curve fit pending", file=sys.stderr)
        return 0
    return subprocess.run(["python3.12", f"{TC}/scaling_law.py", "--fit", floor]).returncode


def run_frontier(label):
    """SERVE-oriented frontier pipeline for a 100B+ model (the real research prize). The doctor does
    NOT fit (f16 2x resident overflows this Studio), so this runs what DOES on streamed shards: the SUBBIT-0
    entropy floor + per-expert sensitivity (MoE) + the serve-fit record. The block-wise condense to a
    serve-fit .tq, the native-serve quality number, and the RAM-cliff tps demo are the Rust serve
    build (read_strand into the serve binary + the per-expert .tq writer) — emitted as gated steps."""
    spec = frontier_by_label(label)
    if not spec:
        print(f"[frontier] unknown {label}; known: {', '.join(frontier_labels())}", file=sys.stderr)
        return 2
    mdir, total, active, bpw, moe, role, hf_id = (
        spec.local_dir, spec.total_b, spec.active_b, spec.serve_bpw, spec.moe, spec.role, spec.hf_id
    )
    artifact = round(spec.artifact_gb(), 1)
    fits = spec.fits_resident(DEFAULT_HARDWARE)
    resident = "RESIDENT (no pager)" if fits else "OVERFLOW (SSD-bound, deep frontier)"
    ncpu = str(os.cpu_count() or 8)
    env = {**os.environ, "DOCTOR_DEVICE": "cpu", "DOCTOR_DTYPE": "bfloat16", "STRAND_NO_GPU": "1",
           "OMP_NUM_THREADS": ncpu, "MKL_NUM_THREADS": ncpu, "VECLIB_MAXIMUM_THREADS": ncpu}
    print(f"[frontier] {spec.label} ({total}B{f', act {active}B MoE' if moe else ' dense'}, role={role}) "
          f"-> {bpw} bpw = {artifact}GB ({resident} on {DEFAULT_HARDWARE.name}; "
          f"weight budget={DEFAULT_HARDWARE.weight_budget_gb:.0f}GB; source={spec.source_kind})",
          file=sys.stderr)
    if not os.path.isdir(mdir):
        print(f"[frontier] {spec.label} NOT staged at {mdir}. Fastest-SOTA procurement (hf_transfer + hf_xet, "
              f"link-bound): python3.12 {TC}/procure.py {spec.label}  "
              f"(download~{spec.download_gb:.0f}GB; verify current free disk + safety reserve first)",
              file=sys.stderr)
        return 2
    # Auto mode: recommend the bit format + serve regime (RESIDENT / MOE-PAGED / DENSE-OOC) and show
    # the device size ceiling before condensing (the "how big can we pull in" advisor).
    ab = ["python3.12", f"{TC}/auto_bits.py", "--params", str(total), "--label", spec.label]
    if active:
        ab += ["--active", str(active)]
    rc = _checkpointed_call(f"P4_FRONTIER/{spec.label}/auto-bits", ab, env=env)
    if rc != 0:
        return rc
    sf = ["python3.12", f"{TC}/size_frontier.py", str(total), "--bpw", str(bpw)]
    if active:
        sf += ["--active", str(active)]
    rc = _checkpointed_call(f"P4_FRONTIER/{spec.label}/size-frontier", sf, env=env)
    if rc != 0:
        return rc
    # The Doctor registry: the auto-composed recovery chain (L0-L6 + per-expert) for this model/bpw.
    dr = ["python3.12", f"{TC}/doctor.py", "registry", "--select", str(total), str(bpw)]
    if moe:
        dr.append("--moe")
    rc = _checkpointed_call(f"P4_FRONTIER/{spec.label}/doctor-registry", dr, env=env)
    if rc != 0:
        return rc
    # Runs on streamed shards (no full f16 resident): the entropy floor + the MoE expert decision.
    rc = _checkpointed_call(
        f"P4_FRONTIER/{spec.label}/subbit-measure",
        ["python3.12", f"{TC}/subbit.py", "measure", mdir, spec.label], env=env)
    if rc != 0:
        return rc
    # Architecture coverage: real state geometry (Mamba2/RWKV-7 flat state) + which Doctor levers
    # are arch-compatible for this model, so the selector above never wastes a bake on one that isn't.
    rc = _checkpointed_call(
        f"P4_FRONTIER/{spec.label}/arch-coverage",
        ["python3.12", f"{TC}/arch_coverage.py", mdir, spec.label], env=env)
    if rc != 0:
        return rc
    if moe:
        rc = _checkpointed_call(
            f"P4_FRONTIER/{spec.label}/expert-sensitivity",
            ["python3.12", f"{TC}/expert.py", "sensitivity", mdir, "--label", spec.label,
             "--bits", "1,2"], env=env)
        if rc != 0:
            return rc
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
        rc = _checkpointed_call(
            f"P4_FRONTIER/{spec.label}/expert-cache",
            ["python3.12", f"{TC}/expert.py", "cache", "--sim", str(n_experts),
             str(active_k), "--active-gb", str(round(active_gb_tok, 3))], env=env)
        if rc != 0:
            return rc
    # the serve-build steps (Rust, gated): block-wise condense + native-serve quality + RAM-cliff tps
    rec = {"model": spec.label, "hf_id": hf_id, "total_b": total, "active_b": active, "moe": moe,
           "role": role, "source_kind": spec.source_kind, "note": spec.note,
           "serve_bpw": bpw, "artifact_gb": artifact, "serve_fits_resident": fits,
           "resident_weight_budget_gb": DEFAULT_HARDWARE.weight_budget_gb,
           "resident_no_pager": fits,
           "condense_cmd": f"# block-wise streamed single-bake (+per-expert if MoE) to {spec.label}.tq @ {bpw}bpw",
           "serve_quality_gated_on": "read_strand wired into hawking-serve binary + native .tq GEMV",
           "ram_cliff_demo": f"serve {spec.label}.tq ({artifact}GB resident) vs Q4_K ({round(total*4.5/8)}GB, overflows->swap)"}
    os.makedirs("reports/condense", exist_ok=True)
    json.dump(rec, open(f"reports/condense/{spec.label}_frontier.json", "w"), indent=2)
    print(f"[frontier] {spec.label} serve-fit recorded; quality+cliff GATED on the native serve build",
          file=sys.stderr)
    return 0


def run_frontier_all():
    """Frontier models are each ~box-filling -> run sequentially (the scheduler would serialize them
    anyway). Skips unstaged. The serve build is the gate on the quality/cliff numbers."""
    failures = {}
    for spec in FRONTIER_MODELS:
        rc = run_frontier(spec.label)
        if rc not in (0, 2):
            failures[spec.label] = rc
    if failures:
        print(f"[frontier] failures: {failures}", file=sys.stderr)
        return 1
    return 0


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
    process_budget = float(getattr(DEFAULT_HARDWARE, "process_budget_gb",
                                   DEFAULT_HARDWARE.weight_budget_gb))
    print(f"RAM-PACK SCHEDULE ({DEFAULT_HARDWARE.name} -> {process_budget:.0f}GB interactive-safe budget):")
    jobs = [Job(lbl, ["true"], est_gb=gb, solo=solo) for (lbl, _, _, gb, solo, _) in LADDER]
    Scheduler(budget_gb=process_budget).plan(jobs)
    print("\nAfter the last model: scaling_law.py fits floor vs log(N), draws the recovered-vs-PTQ")
    print("band, and extrapolates the 70B/405B floor (T3.1) as a pre-registered prediction.")


SPEC_TARGETS = ["7B", "32B"]   # condensed substrate + capstone to revive spec-decode on


def efficiency_baseline():
    """Write the pre-ladder measurement contract from the computational-efficiency agenda."""
    try:
        from ram_scheduler import resource_snapshot
        resources = resource_snapshot()
    except Exception as exc:
        resources = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    staged = {
        label: {
            "model_dir": model_dir,
            "staged": os.path.isdir(model_dir),
            "estimated_peak_gb": peak_gb,
            "role": role,
        }
        for label, model_dir, _params, peak_gb, _solo, role in LADDER
    }
    out = {
        "schema": "hawking.studio_efficiency_baseline.v1",
        "generated_at": _now(),
        "hardware_profile": DEFAULT_HARDWARE.name,
        "research_agenda": "docs/plans/computational_efficiency_paradigms_2026_07_11.md",
        "objective": {
            "primary": "quality-adjusted utility goodput under p95/p99 SLO constraints",
            "pareto_metrics": [
                "capability_per_joule",
                "capability_per_weighted_byte_moved",
                "capability_per_touched_parameter",
                "capability_per_wall_clock_second",
            ],
            "must_charge": [
                "draft_and_rejected_work",
                "download_and_storage_bytes",
                "checkpoint_and_recompute_cost",
                "quality_and_safety_regressions",
            ],
        },
        "resources": resources,
        "ladder": staged,
        "claim_status": "measurement-contract-only",
    }
    _atomic_json("reports/condense/studio_efficiency_baseline.json", out)
    return 0


def _spec_gate():
    gate_path = pathlib.Path("reports/condense/spec_oracle_gate.json")
    try:
        gate = json.loads(gate_path.read_text())
    except Exception:
        gate = {}
    ok = (
        os.environ.get("HAWKING_ENABLE_SPEC_RESEARCH") == "1"
        and gate.get("status") == "pass"
        and float(gate.get("tau", 0.0)) >= 2.5
        and gate.get("one_pass_verifier") is True
    )
    return ok, gate_path, gate


def run_spec_phase():
    ok, gate_path, gate = _spec_gate()
    if not ok:
        print("[spec] gated: require HAWKING_ENABLE_SPEC_RESEARCH=1 plus a passing "
              f"{gate_path} with tau>=2.5 and one_pass_verifier=true; existing EAGLE/n-gram paths "
              "remain below the resurrection gate", file=sys.stderr)
        return 0
    for lbl in SPEC_TARGETS:
        row = next((r for r in LADDER if r[0] == lbl), None)
        if not row or not os.path.isdir(row[1]) or not _have(lbl, "studio"):
            print(f"[spec] {lbl} lacks a completed condensed substrate — defer", file=sys.stderr)
            continue
        rc = _checkpointed_call(
            f"P3_SPEC/{lbl}",
            ["python3.12", f"{TC}/spec_revive.py", row[1], lbl],
        )
        if rc != 0:
            return rc
    return 0


def run_eval_phase():
    targets = [(l, m, p) for (l, m, p, _g, _s, _r) in LADDER
               if l not in ("0.5B", "1.5B") and _have(l, "studio")]
    commands = (
        ("eval", lambda l, m, p: ["python3.12", f"{TC}/eval_suite.py", "--model", m, "--label", l]),
        ("ctx", lambda l, m, p: ["python3.12", f"{TC}/ctx_extend.py", m, l]),
        ("kv-frontier", lambda l, m, p: ["python3.12", f"{TC}/kv.py", "frontier", m, l, str(p)]),
        ("kv-hybrid", lambda l, m, p: ["python3.12", f"{TC}/kv.py", "hybrid", m, l, str(p)]),
    )
    for lbl, mdir, params in targets:
        for substage, build in commands:
            rc = _checkpointed_call(f"P5_EVAL/{lbl}/{substage}", build(lbl, mdir, params))
            if rc != 0:
                return rc
    return 0


def run_baseline_phase():
    targets = [(l, m) for (l, m, _p, _g, _s, _r) in LADDER
               if l not in ("0.5B", "1.5B") and _have(l, "studio")]
    for lbl, mdir in targets:
        rc = _checkpointed_call(
            f"P6_BASELINE/{lbl}",
            ["python3.12", f"{TC}/bench_baselines.py", "--model", mdir, "--label", lbl,
             "--audit-jsonl", f"reports/cron/studio_{lbl}.jsonl"],
        )
        if rc != 0:
            return rc
    return 0


def run_codec_phase():
    target = next(((l, m) for (l, m, _p, _g, _s, _r) in LADDER
                   if l not in ("0.5B", "1.5B") and _have(l, "studio")), None)
    if not target:
        return 0
    lbl, mdir = target
    return _checkpointed_call(
        f"P8_CODEC/{lbl}",
        ["python3.12", f"{TC}/codec_bakeoff.py", "--model", mdir, "--label", lbl],
    )


def run_synthesis_phase():
    for lane in ("studio", "subbit"):
        floor = floors_path(lane)
        if os.path.exists(floor):
            rc = _checkpointed_call(
                f"P9_SYNTH/fit-{lane}",
                ["python3.12", f"{TC}/scaling_law.py", "--fit", floor],
            )
            if rc != 0:
                return rc
    return _checkpointed_call(
        "P9_SYNTH/scorecard",
        ["python3.12", f"{TC}/scorecard.py"],
    )


def go():
    """Run the Studio program from the first incomplete durable phase checkpoint.

    On the Studio, this is the one command:
        python3.12 tools/condense/studio_run.py go
    A hard power loss can only lose work since the last underlying model/config checkpoint; the
    phase marked `running` is rerun, while `pass` phases are skipped."""
    if _draining():
        print(f"[studio] drain is active at {DRAIN_REQUEST}; use `studio_run.py resume` after relocation",
              file=sys.stderr)
        return 130
    gate_rc = _enforce_launch_gate()
    if gate_rc != 0:
        return gate_rc
    if RUN_PID.exists():
        try:
            old = json.loads(RUN_PID.read_text()).get("pid")
            if old and old != os.getpid():
                os.kill(int(old), 0)
                print(f"[studio] another run is live pid={old}", file=sys.stderr)
                return 2
        except (ProcessLookupError, ValueError, json.JSONDecodeError, OSError):
            pass
    _atomic_json(RUN_PID, {"pid": os.getpid(), "started_at": _now(),
                           "hardware_profile": DEFAULT_HARDWARE.name})
    print("=" * 78)
    print(f"HAWKING STUDIO — GO ({DEFAULT_HARDWARE.name}, durable phase resume)")
    print("=" * 78)
    stages = [
        ("P0E_EFFICIENCY", "capability-efficiency baseline/measurement contract", efficiency_baseline),
        ("P0_CODEC", "codec parallelism triage", lambda: _checkpointed_call(
            "P0_CODEC/catalog", ["python3.12", f"{TC}/codec_parallelism.py", "--catalog"])),
        ("P1_CONDENSE", "safe bit-floor ladder", lambda: run_all("studio")),
        ("P2_SUBBIT", "safe sub-bit ladder", lambda: run_all("subbit")),
        ("P3_SPEC", "speculation oracle (gated by tau and one-pass verifier)", run_spec_phase),
        ("P4_FRONTIER", "serve-oriented staged frontier probes", run_frontier_all),
        ("P5_EVAL", "capability, long-context, and state evaluation", run_eval_phase),
        ("P6_BASELINE", "same-box baseline comparison", run_baseline_phase),
        ("P7_CLIFF", "RAM-cliff and energy", lambda: _checkpointed_call(
            "P7_CLIFF/bench", ["python3.12", f"{TC}/ramcliff_bench.py", "--all"])),
        ("P8_CODEC", "codec bakeoff", run_codec_phase),
        ("P9_SYNTH", "curve fit and scorecard", run_synthesis_phase),
    ]
    try:
        for key, description, fn in stages:
            print(f"\n### {key} — {description} ###", file=sys.stderr)
            rc = _checkpointed_phase(key, fn)
            if rc != 0:
                print(f"[studio] stopped at {key} rc={rc}; rerun `resume` after fixing/draining",
                      file=sys.stderr)
                return rc
        print("\nGO COMPLETE — SCORECARD at reports/condense/SCORECARD.md; durable state at "
              f"{RUN_STATE}", file=sys.stderr)
        return 0
    finally:
        try:
            if RUN_PID.exists() and json.loads(RUN_PID.read_text()).get("pid") == os.getpid():
                RUN_PID.unlink()
        except Exception:
            pass


def go_plan():
    """Dry overview of the whole GO program (run nothing heavy)."""
    plan()
    print("\n" + "=" * 78); print("FULL GO PROGRAM (studio_run.py go):")
    print("  P0E EFFICIENCY write the Beyond-FLOPS capability/byte/joule/goodput measurement contract")
    print("  P0 CODEC     codec_parallelism.py --catalog -> triage new codec designs before Rust build time")
    print("  P1 CONDENSE  run_all('studio')  -> bit_floor_curve.jsonl + receipts")
    print("  P2 SUBBIT    run_all('subbit')  -> SUBBIT-0 gate + sub-1-bit lane -> bit_floor_subbit.jsonl")
    print("  P3 SPEC      default-skipped; requires tau>=2.5 oracle + measured one-pass verifier")
    print("  P4 FRONTIER  run_frontier_all() -> 100B+ research prize (" + "/".join(frontier_labels()) + ")")
    print("  P5 EVAL      eval_suite.py (capability + NIAH) + ctx_extend.py (YaRN long-ctx + KV-RAM + SSM moat)")
    print("  P6 BASELINE  bench_baselines.py -> wedge gate vs IQ1_S/IQ2/MLX-4bit at matched bpw")
    print("  P7 CLIFF     ramcliff_bench.py --all -> RAM-cliff tok/s + energy J/tok (headline + energy moat)")
    print("  P8 CODEC     codec_bakeoff.py -> STRAND vs QTIP/QuIP#/AQLM (the codec rank map)")
    print("  P9 SCORECARD scorecard.py -> the POPULATED competitive matrix (no WIN cell without a receipt)")
    print("  P3 remains gated until reports/condense/spec_oracle_gate.json records tau>=2.5 and a "
          "one-pass verifier; the existing EAGLE/n-gram paths are not scheduled by default.")


def _other_heavy_processes():
    """Best-effort guard against declaring a whole-machine unplug safe while other work is live."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,pcpu=,rss=,command="],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) != 4:
            continue
        try:
            pid, cpu, rss_kib = int(parts[0]), float(parts[1]), int(parts[2])
        except ValueError:
            continue
        command = parts[3]
        if pid == os.getpid() or "studio_run.py" in command or "ram_scheduler.py" in command:
            continue
        if cpu >= 50.0 or rss_kib >= 4 * 1024 * 1024:
            rows.append({
                "pid": pid,
                "cpu_percent": cpu,
                "rss_gib": round(rss_kib / (1024 * 1024), 3),
                "command": command[:300],
            })
    return sorted(rows, key=lambda row: (row["cpu_percent"], row["rss_gib"]), reverse=True)[:10]


def _launch_gate(resources=None, other_heavy=None):
    """Return a fail-closed whole-machine launch decision for phone/remote operation."""
    if resources is None:
        try:
            from ram_scheduler import resource_snapshot
            resources = resource_snapshot()
        except Exception as exc:
            resources = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if other_heavy is None:
        other_heavy = _other_heavy_processes()

    blockers = []
    if not resources.get("ok"):
        blockers.append(f"resource snapshot unavailable: {resources.get('error', 'unknown error')}")
    else:
        pressure = int(resources.get("pressure_level", 4))
        swap_mb = float(resources.get("swap_used_mb", 0.0))
        usable_disk = float(resources.get("disk_usable_now_gb", 0.0))
        scratch = float(resources.get("scratch_reserve_gb", DEFAULT_HARDWARE.scratch_reserve_gb))
        if pressure > 1:
            blockers.append(f"memory pressure is {resources.get('pressure_name', pressure)}")
        if swap_mb >= 2048.0:
            blockers.append(f"swap use is {swap_mb:.0f} MB")
        if usable_disk < scratch:
            blockers.append(
                f"only {usable_disk:.1f} GB remains after the disk reserve; {scratch:.1f} GB scratch is required"
            )
    for row in other_heavy:
        blockers.append(
            f"other heavy pid {row['pid']} ({row['cpu_percent']:.0f}% CPU, "
            f"{row['rss_gib']:.1f} GiB): {row['command']}"
        )
    return {
        "schema": "hawking.studio_launch_gate.v1",
        "generated_at": _now(),
        "ok": not blockers,
        "blockers": blockers,
        "resources": resources,
        "other_heavy_processes": other_heavy,
    }


def _record_launch_gate(gate):
    state = _load_run_state()
    state["launch_gate"] = gate
    state["updated_at"] = _now()
    _atomic_json(RUN_STATE, state)


def _enforce_launch_gate():
    gate = _launch_gate()
    _record_launch_gate(gate)
    if gate["ok"] or os.environ.get("HAWKING_STUDIO_ALLOW_CONCURRENT") == "1":
        return 0
    print("[studio] LAUNCH HELD — whole-machine safety gate is red", file=sys.stderr)
    for blocker in gate["blockers"]:
        print(f"  - {blocker}", file=sys.stderr)
    print("[studio] wait for a green `studio_run.py --status`; deliberate overlap requires "
          "HAWKING_STUDIO_ALLOW_CONCURRENT=1", file=sys.stderr)
    return 75


def status():
    try:
        from ram_scheduler import resource_snapshot
        resources = resource_snapshot()
    except Exception as exc:
        resources = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    state = _load_run_state()
    pid_info = {}
    active = False
    try:
        pid_info = json.loads(RUN_PID.read_text())
        os.kill(int(pid_info.get("pid")), 0)
        active = True
    except Exception:
        active = False
    other_heavy = _other_heavy_processes()
    launch_gate = _launch_gate(resources, other_heavy)
    payload = {
        "schema": "hawking.studio_status.v1",
        "generated_at": _now(),
        "hardware_profile": DEFAULT_HARDWARE.name,
        "active": active,
        "pid": pid_info.get("pid"),
        "drain_requested": _draining(),
        "hawking_drained": not active,
        "launch_ready": launch_gate["ok"],
        "launch_gate": launch_gate,
        "safe_to_unplug": not active and not other_heavy,
        "other_heavy_processes": other_heavy,
        "resources": resources,
        "run_state": state,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def drain():
    """Stop new work, let checkpoint-aware children terminate, fsync, then declare move safety."""
    _atomic_json(DRAIN_REQUEST, {"requested_at": _now(), "requested_by_pid": os.getpid()})
    timeout_s = float(os.environ.get("HAWKING_STUDIO_DRAIN_TIMEOUT_S", "300"))
    deadline = time.monotonic() + timeout_s
    active_pid = None
    while time.monotonic() < deadline:
        try:
            info = json.loads(RUN_PID.read_text())
            active_pid = int(info.get("pid"))
            os.kill(active_pid, 0)
        except Exception:
            active_pid = None
            break
        time.sleep(2)
    if active_pid is not None:
        print(f"[studio] DRAIN TIMEOUT — pid {active_pid} still active; NOT safe to unplug", file=sys.stderr)
        return 1
    try:
        os.sync()
    except AttributeError:
        subprocess.run(["sync"], check=False)
    other_heavy = _other_heavy_processes()
    global_safe = not other_heavy
    state = _load_run_state()
    state["drain"] = {
        "status": "safe" if global_safe else "hawking-drained-other-work-active",
        "completed_at": _now(),
        "hawking_drained": True,
        "safe_to_unplug": global_safe,
        "other_heavy_processes": other_heavy,
    }
    state["updated_at"] = _now()
    _atomic_json(RUN_STATE, state)
    if not global_safe:
        print("[studio] HAWKING DRAINED, but NOT globally safe to unplug: another heavy process is active",
              file=sys.stderr)
        for row in other_heavy:
            print(f"  pid={row['pid']} cpu={row['cpu_percent']:.0f}% rss={row['rss_gib']:.1f}GiB "
                  f"{row['command']}", file=sys.stderr)
        return 2
    print("[studio] SAFE TO UNPLUG — no Studio/heavy process is active and checkpoints were synced",
          file=sys.stderr)
    return 0


def resume():
    try:
        DRAIN_REQUEST.unlink()
    except FileNotFoundError:
        pass
    state = _load_run_state()
    state["resume"] = {"requested_at": _now(), "previous_drain_cleared": True}
    state["updated_at"] = _now()
    _atomic_json(RUN_STATE, state)
    return go()


def start_background():
    """Start a detached, caffeinated run that survives phone/app disconnects."""
    try:
        info = json.loads(RUN_PID.read_text())
        os.kill(int(info.get("pid")), 0)
        print(f"[studio] already active pid={info.get('pid')}", file=sys.stderr)
        return 0
    except Exception:
        pass
    gate_rc = _enforce_launch_gate()
    if gate_rc != 0:
        return gate_rc
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(RUN_LOG, "ab", buffering=0)
    python = sys.executable or "python3.12"
    cmd = [python, str(pathlib.Path(__file__).resolve()), "resume"]
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-dimsu", *cmd]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            cwd=ROOT)
    log.close()
    print(f"[studio] detached pid={proc.pid}; log={RUN_LOG}; status: studio_run.py --status",
          file=sys.stderr)
    return 0


def selftest():
    global RUN_STATE, RUN_PID, DRAIN_REQUEST
    import tempfile
    original = (RUN_STATE, RUN_PID, DRAIN_REQUEST)
    try:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            RUN_STATE = root / "state.json"
            RUN_PID = root / "run.pid"
            DRAIN_REQUEST = root / "drain.json"
            _set_phase("P0", "running", command=["synthetic"])
            assert _load_run_state()["phases"]["P0"]["status"] == "running"
            _set_phase("P0", "pass", returncode=0)
            assert _phase_done("P0")
            assert _checkpointed_phase("P0", lambda: 99) == 0
            _atomic_json(DRAIN_REQUEST, {"requested_at": _now()})
            assert _checkpointed_phase("P1", lambda: 0) == 130
            assert _load_run_state()["phases"]["P1"]["status"] == "interrupted"
            normal = {
                "ok": True,
                "pressure_level": 1,
                "pressure_name": "normal",
                "swap_used_mb": 0.0,
                "disk_usable_now_gb": 400.0,
                "scratch_reserve_gb": 64.0,
            }
            assert _launch_gate(normal, [])["ok"]
            assert not _launch_gate({**normal, "pressure_level": 2, "pressure_name": "yellow"}, [])["ok"]
            assert not _launch_gate(normal, [{
                "pid": 7, "cpu_percent": 99.0, "rss_gib": 1.0, "command": "synthetic-heavy"
            }])["ok"]
    finally:
        RUN_STATE, RUN_PID, DRAIN_REQUEST = original
    print("studio_run.py selftest OK")
    return 0


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--plan"
    if a == "go":
        sys.exit(go())
    elif a == "resume":
        sys.exit(resume())
    elif a == "start":
        sys.exit(start_background())
    elif a == "drain":
        sys.exit(drain())
    elif a == "--status":
        sys.exit(status())
    elif a == "--selftest":
        sys.exit(selftest())
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
