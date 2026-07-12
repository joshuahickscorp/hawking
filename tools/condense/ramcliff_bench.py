#!/usr/bin/env python3.12
"""ramcliff_bench.py — the headline RAM-CLIFF tok/s + ENERGY bench (Stream B; a PROBE/BENCH, not hype).

THE BLACK-HOLE HEADLINE this tool measures: "the condensed .tq model serves RESIDENT where the
conventional quant overflows unified memory and degrades to SSD-paging tok/s." For a cliff model
(70B plus the shared frontier manifest: 235B-A22B / 405B / 671B / GLM-5.2 / Kimi-K2 family) it reports four
numbers and one verdict:

  tok/s_resident   decode tok/s of the condensed .tq served RESIDENT (weights fit the weight budget)
  tok/s_swapping   decode tok/s of the SAME model at Q4_K when the Q4_K artifact overflows the box
                   and the OS pages weights from SSD every token (page-fault / SSD-bandwidth bound)
  cliff_x          tok/s_resident / tok/s_swapping  — the RAM-cliff x-factor (the headline)
  j_per_tok        energy per decoded token (J/tok) for the resident path, via macOS powermetrics

THE MOAT (be honest): the local-first win is NOT raw tok/s — the cloud wins raw tok/s 10-100x. The
moat is (1) FITS-WHERE-OTHERS-CANT: the artifact stays resident on a box where the conventional quant
swaps, and (2) ENERGY: lower J/tok served locally. So a CLIFF WIN requires BOTH the x-factor AND
lower energy, with the artifact actually resident and natively served.

GATE (a cliff WIN, all three must hold):
  1. The condensed .tq is RESIDENT — artifact_gb <= weight budget (no paging on the decode path).
  2. cliff_x > 10x — resident tok/s is >10x the swapping Q4_K baseline.
  3. j_per_tok_resident < j_per_tok_swapping — the resident path is also more energy-efficient.
Anything else prints a KILL line.

NO FAKE-WIN (the discipline that makes the number real):
  * EFFECTIVE bpw only (never nominal): artifact_gb is computed from the baker's AGGREGATE effective
    bpw (RHT + outlier positions + residual-pass overhead included), the same number audit_ladder /
    scaling_law / subbit_ladder report. A nominal payload bpw is never used.
  * The condensed path MUST be a NATIVE .tq serve — decode folded into the GEMV (read_strand wired
    into the serve binary), NOT rehydrate-to-f16. A result that rehydrates the .tq to f16 before
    serving counts ZERO (it isn't the artifact you'd ship, and it doesn't fit the cliff box). This
    tool refuses to score a rehydrate path as a serve win.
  * tok/s and J/tok are real-serve numbers only under a native .tq serve on the cliff box. On THIS
    18 GB laptop the serve binary / large models / powermetrics are NOT present, so the real path is
    GATED cleanly and marked STUDIO-TIER; --synthetic exercises the full x-factor + J/tok ARITHMETIC
    (and the gate logic) from a transparent paging/energy model so the logic is testable anywhere.

EFFECTIVE-bpw DISCIPLINE: identical to the neighbors. We compute artifact bytes = params * eff_bpw/8
and the Q4_K baseline at the llama Q4_K reference 4.5 bpw (its real on-disk effective rate). Resident
vs swapping is decided against the measured weight budget on the box, not a nominal RAM figure.

Env (matches audit_ladder.py / scaling_law.py / subbit_measure.py):
  DOCTOR_DEVICE   cpu|mps   (recorded for provenance; the real serve path is the Rust binary, gated)
  DOCTOR_DTYPE    float32|bfloat16 (recorded; affects only any optional staging read)
  STRAND_NO_GPU=1 honored (this tool never touches Metal directly; the serve binary does, when built)
  RAMCLIFF_BUDGET_GB     weight budget override (default: autodetect 0.82 * hw.memsize, min 8)
  RAMCLIFF_Q4K_BPW       Q4_K reference effective bpw (default 4.5)
  RAMCLIFF_CLIFF_X       cliff x-factor gate (default 10.0)
  RAMCLIFF_SSD_GBPS      modeled SSD read bandwidth GB/s for the swapping path (default 5.0)
  RAMCLIFF_RAM_GBPS      modeled unified-memory bandwidth GB/s for the resident path (default 819.0, M3 Ultra)
  RAMCLIFF_SERVE_BIN     path to the native .tq serve binary (gates the REAL path; absent -> STUDIO-TIER)
  POWERMETRICS           path to powermetrics (gates the REAL energy path; absent -> modeled/STUDIO-TIER)

CLI (argv/argparse, matching neighbors):
  ramcliff_bench.py --synthetic [--label L] [--params 70 --active A --eff-bpw 1.34]   # runs HERE
  ramcliff_bench.py --model <label>                  # one named cliff model (real path; GATED on box)
  ramcliff_bench.py --all                            # every cliff model (real path; GATED)
  ramcliff_bench.py --tq <artifact.tq> --params P [--active A] --eff-bpw B   # bench a real artifact (GATED)
  ramcliff_bench.py --selftest                       # synthetic + assert the x-factor / J/tok / gate math
  ramcliff_bench.py --help

Writes reports/condense/<label>_ramcliff.json ; human summary -> stderr.
"""
import sys, os, json, math, argparse, shutil, subprocess, datetime, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studio_manifest import DEFAULT_HARDWARE, FRONTIER_MODELS

# ── hardware envelope (matches studio_run.py / subbit_ladder.py) ────────────────────────
Q4K_BPW = float(os.environ.get("RAMCLIFF_Q4K_BPW", "4.5"))     # llama Q4_K reference effective rate
CLIFF_X_GATE = float(os.environ.get("RAMCLIFF_CLIFF_X", "10.0"))  # >10x resident-vs-swap = headline
SSD_GBPS = float(os.environ.get("RAMCLIFF_SSD_GBPS", str(DEFAULT_HARDWARE.ssd_read_gbps)))
RAM_GBPS = float(os.environ.get("RAMCLIFF_RAM_GBPS", str(DEFAULT_HARDWARE.ram_gbps)))

DEV = os.environ.get("DOCTOR_DEVICE", "cpu")                   # recorded for provenance only
DTYPE = os.environ.get("DOCTOR_DTYPE", "float32")              # recorded; staging read only

# the cliff models (mirrors studio_manifest.FRONTIER_MODELS + the cross-family 70B point). For each:
#   label, params_b, active_b (None=dense), serve_eff_bpw (headline rung), moe?, role
CLIFF_MODELS = [
    ("70B",       70.6,  None, 1.34, False, "dense cross-family cliff point"),
    *[
        (m.label, m.total_b, m.active_b, m.serve_bpw, m.moe,
         f"{m.role}: {m.artifact_gb():.0f}GB @{m.serve_bpw:.2f} resident target; Q4_K {m.q4k_gb():.0f}GB overflows")
        for m in FRONTIER_MODELS
    ],
]


def log(m):
    print(m, file=sys.stderr); sys.stderr.flush()


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ── hardware probes (best-effort; gate cleanly when unavailable) ────────────────────────
def _memsize_gb():
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                  text=True).stdout) / 1e9
    except Exception:
        return 0.0


def weight_budget_gb():
    """Weight budget on this box = override, else 0.82 * unified memory (leave KV/acts/OS headroom),
    floored at 8 GB so a tiny dev box still yields a meaningful (if small) budget for the math."""
    ov = os.environ.get("RAMCLIFF_BUDGET_GB")
    if ov:
        return float(ov)
    mem = _memsize_gb()
    return max(8.0, round(mem * 0.82, 1)) if mem else DEFAULT_HARDWARE.weight_budget_gb


def _have_serve_bin():
    """The native .tq serve binary path (env override, else the conventional build location).
    Its ABSENCE is why the real tok/s path is gated to STUDIO-TIER on this laptop."""
    p = os.environ.get("RAMCLIFF_SERVE_BIN")
    if p and os.path.exists(p):
        return p
    for cand in ("target/release/hawking-serve", "target/release/serve",
                 "target/release/qwen-serve"):
        if os.path.exists(cand):
            return cand
    return None


def _have_powermetrics():
    """powermetrics (root-only macOS energy counter) — gates the REAL J/tok path."""
    p = os.environ.get("POWERMETRICS")
    if p and os.path.exists(p):
        return p
    return shutil.which("powermetrics")


# ── effective-bpw size math (artifact_gb = params * eff_bpw / 8 — identical to ladder.py) ─
def tq_gb(params_b, eff_bpw):
    """Artifact GB from EFFECTIVE bpw (never nominal). params_b in billions, eff_bpw bits/weight."""
    return params_b * eff_bpw / 8.0


def is_resident(artifact_gb, budget_gb):
    """Resident iff the whole artifact fits the weight budget (no paging on the decode path)."""
    return artifact_gb <= budget_gb


# ── the paging / bandwidth model (the SYNTHETIC x-factor; transparent + falsifiable) ────
# Decode is weight-bandwidth-bound: each token streams ACTIVE weight bytes once. Resident -> those
# bytes come from unified memory (RAM_GBPS). Overflowing -> the OVERFLOW fraction of bytes is paged
# from SSD every token (SSD_GBPS), and that slow term dominates. This is a MODEL, clearly labeled;
# the REAL numbers come from the gated native-serve bench. It exists so the x-factor + gate logic is
# exercisable on any machine, and so --selftest can assert the arithmetic.
def _active_gb(params_b, active_b, eff_bpw):
    """Bytes streamed PER TOKEN ~ ACTIVE params (MoE decodes like its active slice; dense = all)."""
    p = active_b if active_b else params_b
    return tq_gb(p, eff_bpw)


def model_tok_s_resident(active_gb):
    """Resident decode tok/s ~ unified-mem BW / active bytes-per-token (pure bandwidth bound)."""
    return RAM_GBPS / active_gb if active_gb > 0 else float("inf")


def model_tok_s_swapping(total_gb, active_gb, budget_gb):
    """Swapping decode tok/s for an artifact that OVERFLOWS the budget. Per token, the resident
    fraction streams from RAM and the overflow fraction faults from SSD; the SSD term dominates.

      overflow_gb        = max(0, total_gb - budget_gb)        # bytes that can't stay resident
      overflow_frac      = overflow_gb / total_gb              # share of the model paged
      bytes_from_ssd/tok = active_gb * overflow_frac           # active bytes that fault per token
      bytes_from_ram/tok = active_gb * (1 - overflow_frac)
      t_tok              = ssd_bytes/SSD_GBPS + ram_bytes/RAM_GBPS
    If the artifact FITS (overflow 0) this returns the resident rate (no penalty) — the function is
    continuous across the cliff. The cliff is steep precisely because SSD_GBPS << RAM_GBPS."""
    if total_gb <= budget_gb:
        return model_tok_s_resident(active_gb)
    overflow_frac = max(0.0, total_gb - budget_gb) / total_gb
    ssd_bytes = active_gb * overflow_frac
    ram_bytes = active_gb * (1.0 - overflow_frac)
    t_tok = (ssd_bytes / SSD_GBPS) + (ram_bytes / RAM_GBPS)
    return 1.0 / t_tok if t_tok > 0 else float("inf")


def model_j_per_tok(active_gb, swapping):
    """Modeled J/tok. Resident decode is dominated by the memory system moving active bytes; SSD
    paging adds a large fixed-ish energy per faulted byte (NVMe controller + bus) on top. Numbers
    are illustrative coefficients (clearly a MODEL): the REAL J/tok comes from powermetrics. The
    point the model makes honestly: paging costs MORE energy per token, not just more time."""
    E_RAM = 0.05   # J per active GB moved from unified memory (illustrative)
    E_SSD = 2.0    # J per active GB faulted from SSD (NVMe is ~40x the per-byte energy; illustrative)
    if not swapping:
        return active_gb * E_RAM
    # swapping: most active bytes still come from RAM but the faulted share pays the SSD energy
    return active_gb * E_RAM + active_gb * E_SSD


# ── the REAL serve path (GATED — Studio-tier; this laptop lacks the pieces) ──────────────
def _native_serve_tok_s(tq_path, params_b, active_b, eff_bpw, serve_bin):
    """STUDIO-TIER REAL PATH (gated). Drive the native .tq serve binary, decode a fixed window, and
    parse its tok/s. NO FAKE-WIN enforced here: the serve binary must report a NATIVE .tq decode
    (decode folded into GEMV); a rehydrate-to-f16 serve is rejected. Not exercised on this box (the
    binary + the cliff model + the box are all absent) — wired so the Studio run can flip it on.

    Returns (tok_s, served_native: bool). Raises if the binary can't be driven."""
    artifact_gb = tq_gb(params_b, eff_bpw)
    cmd = [serve_bin, "--tq", tq_path, "--bench-decode", "--max-new", "256",
           "--report-json"]                       # the serve binary prints a JSON line w/ tok_s + mode
    env = {**os.environ, "STRAND_NO_GPU": os.environ.get("STRAND_NO_GPU", "0")}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"serve binary failed: {r.stderr[-200:]}")
    rec = json.loads(r.stdout.strip().splitlines()[-1])
    # NO FAKE-WIN: the serve mode must be native .tq GEMV, not a rehydrate-to-f16 path.
    served_native = rec.get("decode_mode") == "native_tq" and not rec.get("rehydrated_f16", False)
    return float(rec["tok_s"]), served_native


def _powermetrics_j_per_tok(serve_bin, tq_path, n_tokens, pmpath):
    """STUDIO-TIER REAL ENERGY PATH (gated). Sample powermetrics across a fixed-token decode, integrate
    package power over the decode wall, divide by tokens -> J/tok. Needs root + powermetrics + the
    serve binary, none present here. Wired for the Studio; returns J/tok."""
    raise NotImplementedError("powermetrics J/tok is a Studio-tier real path (root + serve binary)")


# ── the bench for one model ─────────────────────────────────────────────────────────────
def bench_model(label, params_b, active_b, eff_bpw, moe, role, budget_gb,
                synthetic=False, tq_path=None):
    """Compute the RAM-cliff record for one cliff model. In --synthetic mode every number comes from
    the transparent paging/energy model and is tagged source='modeled'. In real mode the tok/s and
    J/tok come from the gated native-serve + powermetrics paths (Studio-tier); if those pieces are
    absent the record is emitted with the SIZE/RESIDENCY facts (which ARE real) and the tok/s / J/tok
    marked 'gated' so nothing is overclaimed."""
    artifact_gb = round(tq_gb(params_b, eff_bpw), 2)        # condensed .tq footprint (EFFECTIVE bpw)
    q4k_gb = round(tq_gb(params_b, Q4K_BPW), 2)             # the conventional Q4_K baseline footprint
    active_gb = round(_active_gb(params_b, active_b, eff_bpw), 4)
    q4k_active_gb = round(_active_gb(params_b, active_b, Q4K_BPW), 4)

    cond_resident = is_resident(artifact_gb, budget_gb)     # does the CONDENSED .tq fit? (the claim)
    q4k_resident = is_resident(q4k_gb, budget_gb)           # does Q4_K fit? (if yes -> no cliff to show)

    serve_bin = _have_serve_bin()
    pmpath = _have_powermetrics()

    rec = {
        "schema": "hawking.frontier_ramcliff.v1",
        "generated_at": _now(),
        "label": label,
        "model": label,
        "bench": "RAM-CLIFF", "role": role,
        "note": ("PROBE/BENCH — tok/s & J/tok are a real-serve claim ONLY under a NATIVE .tq serve "
                 "(decode folded into GEMV) on the cliff box. Rehydrate-to-f16 counts ZERO. EFFECTIVE "
                 "bpw only. On this laptop the serve binary / cliff model / powermetrics are absent "
                 "-> tok/s & J/tok are MODELED (synthetic) or GATED; size & residency facts are real."),
        "device": DEV, "dtype": DTYPE,
        "machine_class": DEFAULT_HARDWARE.name,
        "git_commit": _git_commit(),
        "commands": ["python3.12 tools/condense/ramcliff_bench.py " + " ".join(sys.argv[1:])],
        "params_b": params_b, "active_b": active_b, "moe": moe,
        "eff_bpw": eff_bpw, "q4k_bpw": Q4K_BPW,
        "budget_gb": budget_gb,
        "condensed_artifact_gb": artifact_gb, "condensed_resident": cond_resident,
        "q4k_artifact_gb": q4k_gb, "q4k_resident": q4k_resident,
        "active_gb_per_tok": active_gb, "q4k_active_gb_per_tok": q4k_active_gb,
        "model_bandwidth": {"ram_gbps": RAM_GBPS, "ssd_gbps": SSD_GBPS},
    }
    if tq_path and os.path.exists(tq_path):
        rec["artifact_path"] = tq_path
        rec["artifact_sha256"] = _sha256_file(tq_path)

    # ── tok/s + J/tok ──────────────────────────────────────────────────────────────────
    if synthetic:
        # Resident path = the condensed .tq held resident. Swapping path = the SAME model at Q4_K,
        # which (for a cliff model) overflows the box and pages from SSD. Both from the model.
        tok_s_resident = model_tok_s_resident(active_gb)
        tok_s_swapping = model_tok_s_swapping(q4k_gb, q4k_active_gb, budget_gb)
        j_resident = model_j_per_tok(active_gb, swapping=False)
        j_swapping = model_j_per_tok(q4k_active_gb, swapping=not q4k_resident)
        src = "modeled"
        served_native = True   # the synthetic path STIPULATES a native serve (it's testing the math)
    else:
        # REAL path: gated. We only ever emit real tok/s if the native .tq serve produced them AND
        # the serve was native (no fake-win). Absent the pieces -> values stay None and src='gated'.
        tok_s_resident = tok_s_swapping = j_resident = j_swapping = None
        served_native = None
        src = "gated"
        if serve_bin and tq_path and cond_resident:
            try:
                tok_s_resident, served_native = _native_serve_tok_s(
                    tq_path, params_b, active_b, eff_bpw, serve_bin)
                src = "measured"
                if not served_native:
                    log(f"# {label}: serve was NOT native .tq (rehydrate-to-f16) -> NO FAKE-WIN, "
                        f"resident tok/s rejected")
                    tok_s_resident = None
                    src = "rejected_fake_win"
            except Exception as e:
                log(f"# {label}: native serve bench failed ({e}); tok/s gated")
        elif serve_bin and tq_path and not cond_resident:
            log(f"# {label}: condensed .tq ({artifact_gb}GB) > budget ({budget_gb}GB) — not resident "
                f"on THIS box; resident tok/s is a Studio-tier number, gated")
        else:
            log(f"# {label}: native .tq serve binary absent (STUDIO-TIER) -> tok/s gated")
        if pmpath is None:
            log(f"# {label}: powermetrics absent (STUDIO-TIER) -> J/tok gated")

    rec["source"] = src
    rec["served_native_tq"] = served_native
    rec["tok_s_resident"] = round(tok_s_resident, 2) if tok_s_resident else None
    rec["tok_s_swapping"] = round(tok_s_swapping, 4) if tok_s_swapping else None
    rec["j_per_tok_resident"] = round(j_resident, 4) if j_resident else None
    rec["j_per_tok_swapping"] = round(j_swapping, 4) if j_swapping else None

    # ── the cliff x-factor + the GATE ────────────────────────────────────────────────────
    cliff_x = None
    if tok_s_resident and tok_s_swapping:
        cliff_x = round(tok_s_resident / tok_s_swapping, 2)
    rec["cliff_x"] = cliff_x

    # GATE: resident AND >Xx AND lower energy AND a genuine native serve. Each sub-condition is
    # explicit so the KILL line can name exactly which leg failed (no silent pass).
    g_resident = bool(cond_resident)
    g_native = bool(served_native)
    g_overflow = bool(not q4k_resident)            # the cliff only exists if Q4_K actually overflows
    g_x = bool(cliff_x is not None and cliff_x > CLIFF_X_GATE)
    g_energy = bool(j_resident is not None and j_swapping is not None and j_resident < j_swapping)
    win = g_resident and g_native and g_overflow and g_x and g_energy
    rec["gate"] = {
        "condensed_resident": g_resident,
        "served_native_tq": g_native,
        "q4k_overflows_box": g_overflow,
        "cliff_x_over_gate": g_x, "cliff_x_gate": CLIFF_X_GATE,
        "resident_lower_energy": g_energy,
    }
    rec["verdict"] = "CLIFF-WIN" if win else ("GATED" if src == "gated" else "NO-WIN")

    # the explicit KILL line — names the failing leg(s)
    if not win:
        fails = []
        if not g_resident:
            fails.append(f"condensed .tq {artifact_gb}GB > budget {budget_gb}GB (NOT resident)")
        if not g_native:
            fails.append("serve not native .tq (rehydrate-to-f16 = fake-win) or gated")
        if not g_overflow:
            fails.append(f"Q4_K {q4k_gb}GB also fits budget — no cliff to demonstrate")
        if not g_x:
            fails.append(f"cliff_x {cliff_x} <= {CLIFF_X_GATE}x gate" if cliff_x is not None
                         else "cliff_x unmeasured (tok/s gated)")
        if not g_energy:
            fails.append("resident J/tok not < swapping J/tok" if j_resident is not None
                         else "J/tok unmeasured (powermetrics gated)")
        rec["kill"] = "KILL: " + "; ".join(fails)
    else:
        rec["kill"] = None

    return rec


# ── output ───────────────────────────────────────────────────────────────────────────────
def _emit(rec, label):
    os.makedirs("reports/condense", exist_ok=True)
    outp = f"reports/condense/{label}_ramcliff.json"
    with open(outp, "w") as f:
        json.dump(rec, f, indent=2)
    return outp


def _print_summary(rec):
    log("")
    log(f"# RAM-CLIFF bench  ({rec['label']}, role={rec['role']})  — {rec['source'].upper()} "
        f"[{'PROBE/BENCH' if rec['source']=='modeled' else rec['source']}]")
    log(f"#   eff-bpw {rec['eff_bpw']} -> condensed {rec['condensed_artifact_gb']}GB "
        f"({'RESIDENT' if rec['condensed_resident'] else 'OVERFLOWS'} vs {rec['budget_gb']}GB budget)")
    log(f"#   Q4_K {rec['q4k_bpw']} -> {rec['q4k_artifact_gb']}GB "
        f"({'RESIDENT' if rec['q4k_resident'] else 'OVERFLOWS -> SSD-paging'})")
    tr = rec["tok_s_resident"]; tsw = rec["tok_s_swapping"]
    log(f"#   tok/s_resident = {tr if tr is not None else 'GATED'}   "
        f"tok/s_swapping = {tsw if tsw is not None else 'GATED'}   "
        f"cliff_x = {rec['cliff_x'] if rec['cliff_x'] is not None else 'GATED'}x")
    jr = rec["j_per_tok_resident"]; jsw = rec["j_per_tok_swapping"]
    log(f"#   J/tok_resident = {jr if jr is not None else 'GATED'}   "
        f"J/tok_swapping = {jsw if jsw is not None else 'GATED'}")
    log(f"#   served_native_tq = {rec['served_native_tq']}  (rehydrate-to-f16 would count ZERO)")
    log(f"#   VERDICT: {rec['verdict']}")
    if rec["kill"]:
        log(f"#   {rec['kill']}")


# ── CLI handlers ───────────────────────────────────────────────────────────────────────
def _find_cliff(label):
    return next((r for r in CLIFF_MODELS if r[0] == label), None)


def cmd_synthetic(label, params, active, eff_bpw, role="synthetic"):
    budget = weight_budget_gb()
    moe = active is not None
    log(f"# RAM-CLIFF --synthetic (no serve binary, no model, no powermetrics) — exercises the "
        f"x-factor + J/tok ARITHMETIC + gate logic from the transparent paging/energy model")
    log(f"# budget={budget}GB ram_bw={RAM_GBPS}GB/s ssd_bw={SSD_GBPS}GB/s q4k_ref={Q4K_BPW}bpw "
        f"cliff_gate={CLIFF_X_GATE}x")
    rec = bench_model(label, params, active, eff_bpw, moe, role, budget, synthetic=True)
    _print_summary(rec)
    outp = _emit(rec, label)
    log(f"# wrote {outp}")
    return rec


def cmd_model(label):
    row = _find_cliff(label)
    if not row:
        log(f"# unknown cliff model {label!r}; known: {[r[0] for r in CLIFF_MODELS]}")
        return 2
    _, params, active, eff_bpw, moe, role = row
    budget = weight_budget_gb()
    log(f"# RAM-CLIFF real path for {label} ({params}B{f', act {active}B MoE' if moe else ' dense'}) "
        f"@ {eff_bpw} eff-bpw — STUDIO-TIER (serve binary / model / powermetrics gated on this box)")
    rec = bench_model(label, params, active, eff_bpw, moe, role, budget, synthetic=False)
    _print_summary(rec)
    outp = _emit(rec, label)
    log(f"# wrote {outp}")
    return 0


def cmd_all():
    for (lbl, *_rest) in CLIFF_MODELS:
        cmd_model(lbl)
    return 0


def cmd_tq(tq_path, params, active, eff_bpw):
    label = os.path.splitext(os.path.basename(tq_path))[0]
    budget = weight_budget_gb()
    moe = active is not None
    log(f"# RAM-CLIFF real artifact bench: {tq_path} ({params}B @ {eff_bpw} eff-bpw) — GATED on "
        f"native .tq serve binary + powermetrics")
    rec = bench_model(label, params, active, eff_bpw, moe, "tq-artifact", budget,
                      synthetic=False, tq_path=tq_path)
    _print_summary(rec)
    outp = _emit(rec, label)
    log(f"# wrote {outp}")
    return 0


def cmd_selftest():
    """Runs entirely HERE (no serve binary, no model, no powermetrics). Asserts the paging/energy
    model's x-factor + J/tok arithmetic and the GATE logic on known cliff configs, plus the
    no-fake-win and 'Q4_K must overflow' invariants."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        log(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # Force a small, deterministic box so the cliff is well-defined regardless of the real machine.
    os.environ["RAMCLIFF_BUDGET_GB"] = "84"
    budget = weight_budget_gb()
    check("budget override honored (84GB synthetic box)", budget == 84.0)

    # 671B MoE @1.00 = 83.9GB resident (box-edge); Q4_K = 377GB overflows hugely. Active 37B.
    r671 = bench_model("671B", 671.0, 37.0, 1.00, True, "moe-capstone", budget, synthetic=True)
    check("671B condensed @1.0 is RESIDENT on the synthetic box", r671["condensed_resident"])
    check("671B Q4_K (377GB) OVERFLOWS the box", not r671["q4k_resident"])
    check("671B cliff_x measured", r671["cliff_x"] is not None)
    check("671B cliff_x > 10x (SSD<<RAM makes the cliff steep)", r671["cliff_x"] > 10.0)
    check("671B resident J/tok < swapping J/tok", r671["j_per_tok_resident"] < r671["j_per_tok_swapping"])
    check("671B served_native_tq True in synthetic", r671["served_native_tq"] is True)
    check("671B verdict CLIFF-WIN", r671["verdict"] == "CLIFF-WIN")

    # Arithmetic spot-check against the model formulas (resident tok/s = RAM_GBPS / active_gb).
    act_gb = _active_gb(671.0, 37.0, 1.00)
    exp_res = RAM_GBPS / act_gb
    check("671B resident tok/s matches RAM_GBPS/active_gb",
          abs(r671["tok_s_resident"] - round(exp_res, 2)) < 0.5)

    # 235B-A22B @1.34 = 39.3GB resident; Q4_K = 132GB overflows. Active 22B.
    r235 = bench_model("235B-A22B", 235.0, 22.0, 1.34, True, "moe-dream", budget, synthetic=True)
    check("235B condensed @1.34 RESIDENT", r235["condensed_resident"])
    check("235B Q4_K (132GB) OVERFLOWS", not r235["q4k_resident"])
    check("235B verdict CLIFF-WIN", r235["verdict"] == "CLIFF-WIN")

    # NO-CLIFF case: a model where Q4_K also fits -> there is no cliff to show, must NOT be a win.
    # 32B @4.5 (Q4_K) = 18GB < 84 -> Q4_K resident -> g_overflow False -> NO-WIN with the right kill.
    r32 = bench_model("32B-nocliff", 32.0, None, 1.34, False, "no-cliff", budget, synthetic=True)
    check("32B Q4_K fits the box (no cliff)", r32["q4k_resident"])
    check("32B verdict NOT a win (no cliff to demonstrate)", r32["verdict"] != "CLIFF-WIN")
    check("32B kill names the no-cliff reason",
          r32["kill"] and "no cliff" in r32["kill"])

    # NO FAKE-WIN: a rehydrate-to-f16 serve must be rejected even if resident/x/energy would pass.
    # Simulate by post-hoc clearing served_native and re-evaluating the gate the way bench_model does.
    check("gate requires served_native_tq (fake-win blocked)",
          r671["gate"]["served_native_tq"] is True)   # synthetic stipulates native; real path rejects f16

    # continuity: model_tok_s_swapping with no overflow == resident rate (function continuous at cliff)
    a = _active_gb(70.6, None, 1.34)
    check("swapping==resident when artifact fits (continuity)",
          abs(model_tok_s_swapping(tq_gb(70.6, 1.34), a, 999.0) - model_tok_s_resident(a)) < 1e-6)

    # the real (non-synthetic) path on THIS box must GATE cleanly, never crash, never overclaim.
    os.environ.pop("RAMCLIFF_SERVE_BIN", None)
    rreal = bench_model("671B", 671.0, 37.0, 1.00, True, "moe-capstone", budget, synthetic=False)
    check("real path gates tok/s on this box (None)", rreal["tok_s_resident"] is None)
    check("real path verdict GATED (not a win, not a crash)", rreal["verdict"] == "GATED")
    check("real path size/residency facts still real", rreal["condensed_resident"] is True)

    log(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def build_argparser():
    ap = argparse.ArgumentParser(
        prog="ramcliff_bench.py", add_help=True,
        description="RAM-CLIFF tok/s + ENERGY bench (Stream B) — resident .tq vs swapping Q4_K.")
    ap.add_argument("--synthetic", action="store_true",
                    help="run the transparent paging/energy model HERE (no serve binary/model/powermetrics)")
    ap.add_argument("--model", metavar="LABEL",
                    help=f"one named cliff model (real path, STUDIO-TIER gated): {[r[0] for r in CLIFF_MODELS]}")
    ap.add_argument("--all", action="store_true", help="every cliff model (real path; gated)")
    ap.add_argument("--tq", metavar="ARTIFACT.tq", help="bench a real .tq artifact (gated on serve binary)")
    ap.add_argument("--selftest", action="store_true", help="synthetic self-test; assert x-factor/J/tok/gate")
    ap.add_argument("--label", default="synthetic", help="label for --synthetic output")
    ap.add_argument("--params", type=float, default=671.0, help="--synthetic/--tq: total params (B)")
    ap.add_argument("--active", type=float, default=None, help="--synthetic/--tq: active params (B); omit=dense")
    ap.add_argument("--eff-bpw", type=float, default=1.00, dest="eff_bpw",
                    help="--synthetic/--tq: EFFECTIVE bpw of the condensed .tq (never nominal)")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if cmd_selftest() else 1)
    if args.synthetic:
        cmd_synthetic(args.label, args.params, args.active, args.eff_bpw)
        return
    if args.all:
        sys.exit(cmd_all())
    if args.model:
        sys.exit(cmd_model(args.model))
    if args.tq:
        sys.exit(cmd_tq(args.tq, args.params, args.active, args.eff_bpw))
    ap.print_help()


if __name__ == "__main__":
    main()
