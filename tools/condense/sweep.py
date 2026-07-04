#!/usr/bin/env python3.12
"""Hawking condense parameter-sweep DRIVER — the bit-floor search across the ladder.

Contract (docs/plans/parameter_sweep_pipeline.md): for each model, climb EFFECTIVE bpw across
recipes (single-bake AWQ + residual STRAND_b1+b2) and STOP at the lowest eff-bpw the doctor
holds near-1:1. That recipe × the param count = the smallest artifact with full capability =
the highest tps. Two streams per recipe:
  A (CONDENSE/quality): ppl(f16) → bake(recipe) → ppl(healed) → Δ              [always]
  B (SERVE/runtime):    .tq size + fit/cliff (single-bake serves today)        [proj; tps gated]

Invariants:
  · SAFE DEFAULT — prints the plan; nothing runs/writes without --go.
  · IDEMPOTENT  — skips (model:recipe) cells already in the JSONL (resumable).
  · DEVICE-AWARE — --profile here (cpu/bf16, ≤7B) | studio (mps/bf16, full ladder).
  · TIER-GATED  — naive resident condense only ≤34B; >34B = phase-2 block-wise (pending) →
    never naively loaded (would OOM). Stream-B projection runs for every size (computable).
  · DISK-DISCIPLINED — JIT-fetch f16, drop f16-sized artifacts after recording numbers.
  · bf16 not f16 — fp16 overflows on the 7B CPU forward → nan (observed 2026-06-23).

Usage:
  python tools/condense/sweep.py --profile here                       # PLAN (safe, default)
  python tools/condense/sweep.py --profile studio --only qwen2.5 --go
  python tools/condense/sweep.py --profile studio --max-params 32 --go --serve
  python tools/condense/sweep.py --status
"""
import sys, os, json, time, subprocess, shutil, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ladder as L

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
JSONL = os.path.join(ROOT, "reports", "condense", "ladder.jsonl")
SCRATCH = os.path.join(ROOT, "scratch")
EVAL = os.path.join(SCRATCH, "sweep_eval.txt")
CALIB = os.path.join(SCRATCH, "calib_corpus.txt")
PY = "python3.12"

# bfloat16 (NOT float16): same 2-byte footprint but full fp32 RANGE → no overflow→nan
# (fp16 maxes at 65504; a 7B's activations exceed it on CPU → nan, observed 2026-06-23). 2-byte
# also keeps resident-condense ≤~34B (32B bf16 = 64 GB fits 96; f32 = 128 GB no). bf16 also
# sidesteps the MPS *float16* GQA bug. STUDIO CAVEAT: MPS bf16 is newer — sanity-check vs f32.
PROFILES = {
    "here":   dict(DOCTOR_DEVICE="cpu", DOCTOR_DTYPE="bfloat16"),
    "studio": dict(DOCTOR_DEVICE="mps", DOCTOR_DTYPE="bfloat16"),
}
ALPHA = 0.5                       # AWQ alpha (single-bake); residual uses RHT+outlier


# ── io ────────────────────────────────────────────────────────────────────────────────
def load_rows():
    if not os.path.exists(JSONL):
        return []
    return [json.loads(l) for l in open(JSONL) if l.strip()]

def append_row(row):
    os.makedirs(os.path.dirname(JSONL), exist_ok=True)
    with open(JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")

def done_set():
    return {f"{r.get('model')}:{r.get('recipe')}" for r in load_rows() if r.get("stream") == "A"}

def f16_ppl_for(model):
    return next((r["f16_ppl"] for r in load_rows() if r.get("model") == model and r.get("f16_ppl")), None)


# ── shell ───────────────────────────────────────────────────────────────────────────--
def sh(cmd, env=None, timeout=None):
    e = dict(os.environ); e.update(env or {})
    return subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True, timeout=timeout)

def ppl(hf_dir, override, label, env):
    r = sh([PY, "tools/condense/ppl_bench.py", hf_dir, override or "-", label],
           env={**env, "PPL_TEXT": EVAL})
    for ln in r.stdout.splitlines():
        if ln.startswith("{"):
            try:
                return json.loads(ln)["ppl"]
            except Exception:
                pass
    print(f"    ! ppl_bench failed ({label}): {r.stderr.strip().splitlines()[-1:]}", file=sys.stderr)
    return None


# ── stream A — condense / quality (one recipe → one healed Δ) ──────────────────────────
def stream_a(m, hf_dir, recipe, env):
    name, p = m["name"], m["params_b"]
    kind, a, b = recipe
    lbl, eb = L.recipe_label(recipe), L.eff_bpw(recipe)
    art = os.path.join(SCRATCH, f"{name}-{lbl}.safetensors")
    res = dict(stream="A", model=name, family=m["family"], params_b=p, recipe=lbl, kind=kind,
               b1=a, b2=b, eff_bpw=eb, serves=L.serves(recipe), ts=int(time.time()))
    f16 = f16_ppl_for(name) or ppl(hf_dir, None, "f16", env)
    res["f16_ppl"] = f16
    if kind == "single":          # AWQ base = the artifact (activation-aware, training-free)
        r = sh([PY, "tools/condense/awq_bake.py", hf_dir, art, str(a), str(ALPHA)],
               env={**env, "DOCTOR_CALIB": CALIB})
    else:                          # residual = STRAND_b1 + STRAND_b2(residual), full-rank heal
        r = sh([PY, "tools/condense/residual.py", "bake", hf_dir, art, str(a), str(b)],
               env={**env, "DOCTOR_CALIB": CALIB})
    if not os.path.exists(art):
        res["error"] = f"{kind}_bake failed: " + (r.stderr.strip().splitlines()[-1] if r.stderr else "?")
        return res
    hp = ppl(hf_dir, art, lbl, env)
    res["heal_ppl"] = hp
    res["heal_delta"] = (hp / f16 - 1) if (hp and f16) else None
    os.remove(art)                 # disk discipline — keep the numbers, drop the f16-sized artifact
    return res


# ── stream B — serve / cliff (projection always; tps when serve wired & recipe serves) ──
def stream_b(m, recipe, tq_path=None, env=None, do_measure=False):
    p, eb = m["params_b"], L.eff_bpw(recipe)
    res = dict(stream="B", model=m["name"], params_b=p, recipe=L.recipe_label(recipe), eff_bpw=eb,
               tq_gb=round(L.tq_gb(p, eb), 2), fits96=L.serve_fits(p, eb), serves=L.serves(recipe),
               llama_q4k_gb=round(L.tq_gb(p, 4.50), 2), llama_fits=L.serve_fits(p, 4.50),
               active_b=m.get("active_b"), ts=int(time.time()))
    if not L.serves(recipe):
        res["cliff"] = "residual serve format (two-part .tq) not yet built"
    elif res["fits96"] and not res["llama_fits"]:
        res["cliff"] = "WIN: Hawking fits, llama Q4_K does not"
    elif res["fits96"] and res["llama_fits"]:
        res["cliff"] = "both fit (iso-model density/tps bench)"
    else:
        res["cliff"] = "neither fits at this bpw"
    if do_measure and L.serves(recipe) and tq_path and os.path.exists(tq_path):
        res.update(_measure_tps(tq_path, env))
    else:
        res["tps"] = None
        res["tps_note"] = "projection only" if L.serves(recipe) else "residual serve pending"
    return res

def _measure_tps(tq_path, env):
    hk = shutil.which("hawking") or os.path.join(ROOT, "target", "release", "hawking")
    if not os.path.exists(hk):
        return dict(tps=None, tps_note="hawking binary not built")
    r = sh([hk, "generate", "--weights", tq_path, "--max-tokens", "128", "--bench"], env=env or {})
    for ln in r.stdout.splitlines():
        if "tok/s" in ln or "tps" in ln:
            return dict(tps_raw=ln.strip())
    return dict(tps=None, tps_note="serve ran, no tps parsed")


# ── floor search per model (climb recipes by eff bpw, stop at 1:1) ─────────────────────
def floor_search(m, env, args, done):
    name, p = m["name"], m["params_b"]
    tier = L.condense_tier(p)
    print(f"\n=== {name} ({p}B, {tier}) ===")
    hf_dir = os.path.join(SCRATCH, name)

    if tier != "resident":          # >34B → phase-2 block-wise (pending); still project serve
        print(f"  condense: PENDING (phase-2 block-wise; {tier}). Stream-B projection:")
        for r in L.RECIPES:
            b = stream_b(m, r, env=env, do_measure=args.serve)
            print(f"    {b['recipe']:9s} eff~{b['eff_bpw']:.2f}bpw .tq {b['tq_gb']}GB  "
                  f"{'fits96✓' if b['fits96'] else 'fits96✗'} | {b['cliff']}")
            if args.go:
                append_row(dict(stream="A", model=name, params_b=p, recipe=b["recipe"],
                                eff_bpw=b["eff_bpw"], status="pending-phase2-blockwise",
                                tier=tier, ts=int(time.time())))
                append_row(b)
        return

    if not os.path.isdir(hf_dir):
        if not args.go:
            print(f"  would JIT-fetch {m['hf_id']} → {hf_dir} (f16 ≈ {L.f16_gb(p):.0f} GB)")
            for r in L.RECIPES:
                eb = L.eff_bpw(r)
                print(f"    plan {L.recipe_label(r):9s} eff~{eb:.2f}bpw .tq≈{L.tq_gb(p,eb):.1f}GB "
                      f"{'fits✓' if L.serve_fits(p,eb) else 'fits✗'}"
                      f"{'' if L.serves(r) else ' [residual serve pending]'}")
            return
        if not fetch(m, hf_dir):
            print(f"  ! fetch failed; skipping {name}"); return

    floor_1to1 = floor_win = None
    for r in L.RECIPES:
        lbl, eb = L.recipe_label(r), L.eff_bpw(r)
        key = f"{name}:{lbl}"
        d = None
        if key in done:
            prev = next((x for x in load_rows() if x.get("stream") == "A"
                         and x.get("model") == name and x.get("recipe") == lbl), None)
            d = prev.get("heal_delta") if prev else None
            print(f"  {lbl}: cached (Δ {_pct(d)})")
        elif not args.go:
            print(f"  {lbl}: would run (eff~{eb:.2f}bpw)")
            continue
        else:
            print(f"  {lbl} (eff~{eb:.2f}bpw): condensing…", flush=True)
            a = stream_a(m, hf_dir, r, env)
            append_row(a)
            if L.serves(r):
                append_row(stream_b(m, r, env=env, do_measure=args.serve))
            done.add(key)
            d = a.get("heal_delta")
            print(f"    Δ {_pct(d)}  (.tq~{L.tq_gb(p,eb):.1f}GB"
                  f"{'' if L.serves(r) else ', residual serve pending'})")
        if d is not None:
            if floor_win is None and d <= L.WIN:
                floor_win = (lbl, eb)
            if floor_1to1 is None and d <= L.NEAR_1to1:
                floor_1to1 = (lbl, eb)
        if floor_1to1 and not args.full_climb:
            print(f"  → 1:1 floor = {floor_1to1[0]} @ eff~{floor_1to1[1]:.2f}bpw "
                  f"(stop; --full-climb maps all)")
            break
    f1 = f"{floor_1to1[0]}@{floor_1to1[1]:.2f}bpw" if floor_1to1 else ">top"
    fw = f"{floor_win[0]}@{floor_win[1]:.2f}bpw" if floor_win else ">top"
    print(f"  FLOOR {name}: 1:1={f1}  win={fw}")


# ── fetch / helpers ────────────────────────────────────────────────────────────────────
def fetch(m, hf_dir):
    free = shutil.disk_usage(ROOT).free / 1e9
    need = L.f16_gb(m["params_b"]) * 1.1
    if free < need:
        print(f"  ! disk {free:.0f}GB < need {need:.0f}GB for {m['name']} f16"); return False
    print(f"  fetching {m['hf_id']} → {hf_dir} …")
    r = sh(["hf", "download", m["hf_id"], "--local-dir", hf_dir])
    return os.path.isdir(hf_dir) and r.returncode == 0

def _pct(x):
    return f"{x*100:+.1f}%" if isinstance(x, (int, float)) else "  ?  "

def select(args):
    ms = L.MODELS
    if args.only:
        ms = [m for m in ms if args.only in (m["family"], m["name"])]
    if args.max_params:
        ms = [m for m in ms if m["params_b"] <= args.max_params]
    if args.max_prio is not None:
        ms = [m for m in ms if m["priority"] <= args.max_prio]
    return sorted(ms, key=lambda m: (m["priority"], m["params_b"]))

def status():
    rows = load_rows()
    bym = {}
    for r in rows:
        if r.get("stream") == "A" and r.get("heal_delta") is not None:
            bym.setdefault(r["model"], []).append((r.get("eff_bpw"), r["recipe"], r["heal_delta"]))
    print(f"# sweep status — {len(rows)} rows, {len(bym)} models with quality data")
    for name in sorted(bym, key=lambda n: next((m["params_b"] for m in L.MODELS if m["name"] == n), 0)):
        cells = sorted(bym[name])
        win = next((f"{lbl}@{eb}bpw" for eb, lbl, d in cells if d <= L.WIN), ">top")
        s11 = next((f"{lbl}@{eb}bpw" for eb, lbl, d in cells if d <= L.NEAR_1to1), ">top")
        pts = " ".join(f"{lbl}:{_pct(d)}" for eb, lbl, d in cells)
        print(f"  {name:18s} 1:1={s11:14s} win={win:14s}  {pts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="here")
    ap.add_argument("--only", help="family or model name")
    ap.add_argument("--max-params", type=float, dest="max_params")
    ap.add_argument("--max-prio", type=int, dest="max_prio")
    ap.add_argument("--go", action="store_true", help="EXECUTE (default: plan only)")
    ap.add_argument("--serve", action="store_true", help="measure tps (needs wired GPU TQ serve)")
    ap.add_argument("--full-climb", action="store_true", help="map all recipes, don't stop at floor")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        return status()

    env = dict(PROFILES[args.profile])
    os.makedirs(SCRATCH, exist_ok=True)
    if not os.path.exists(EVAL):
        if args.go:
            p = os.path.join(ROOT, "README.md")
            txt = open(p, errors="ignore").read() if os.path.exists(p) else ""
            open(EVAL, "w").write(txt[:24000] or "the science of operations")
        else:
            print(f"# (would build {EVAL} on --go)")

    done = done_set()
    ms = select(args)
    mode = "EXECUTE" if args.go else "PLAN (use --go to run)"
    print(f"# Hawking sweep — profile={args.profile} ({env['DOCTOR_DEVICE']}/{env['DOCTOR_DTYPE']}) "
          f"· {len(ms)} models · {mode}")
    print(f"# thresholds: 1:1 ≤ +{L.NEAR_1to1*100:.0f}%  win ≤ +{L.WIN*100:.0f}%  · "
          f"recipes (eff bpw ↑): {', '.join(L.recipe_label(r) for r in L.RECIPES)}")
    for m in ms:
        floor_search(m, env, args, done)
    if args.go:
        print(sh([PY, "tools/condense/sweep_render.py"]).stdout.strip())


if __name__ == "__main__":
    main()
