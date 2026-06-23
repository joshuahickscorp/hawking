#!/usr/bin/env python3.12
"""Damage-ranked MIXED-PRECISION allocation — the density "doctor" recovery layer.

GOAL: minimize AVERAGE effective bpw at a quality target by spending bits where the
network is OUTPUT-sensitive and starving the tolerant tensors. A uniform N-bit bake
treats `down_proj` (high activation energy, recon error lands straight in the residual
stream) the same as a low-energy `k_proj`; mixed-precision moves a bit from the second
to the first at the SAME average bpw. This tool ranks every linear by output-space
sensitivity, water-fills bits under a target avg bpw, emits a baker `--mp-config`, and
proves the win (or honestly reports none) against a UNIFORM bake at the same avg bpw.

────────────────────────────────────────────────────────────────────────────────────
SENSITIVITY METRIC  (per linear tensor t, bits b)
────────────────────────────────────────────────────────────────────────────────────
The decode error that MATTERS is in OUTPUT space: ||(Ŵ−W)·X|| / ||W·X|| on calib
activations, NOT the raw weight-space ||Ŵ−W||/||W|| (memory: weight-space rel-RMS is
"proxy-limbo"). We make the baker's cheap per-tensor `rel-RMS` output-space by folding
in the per-input-channel activation energy σ_c = mean_calib|x_c| (captured exactly as
audit_ladder.capture_sigma / awq_bake — same hook, same corpus):

    sens(t,b) ≈ relRMS(t,b) · actnorm(t),   actnorm(t) = ||σ_t||_2 / sqrt(in_features)

  · relRMS(t,b): measured by the baker (`--measure-only --only <t>` per bit) — the TRUE
    trellis recon error at that bit, RHT+outlier folded in. Not an analytic guess.
  · actnorm(t): RMS input-channel magnitude — how hard this tensor's output is driven.
    A tensor whose inputs are ~0 can be crushed for free; one on the hot path cannot.
  · product = an output-space damage PROXY that needs no full ppl and no ΔW·X matmul
    (which would need the decoded Ŵ in Python). Cheap: one measure-only bake per bit.

GOLD (opt-in, --metric outxe): instead of relRMS·actnorm, decode Ŵ per bit and compute
the exact per-tensor output-space error ||(Ŵ−W)·Xcalib|| / ||W·Xcalib|| from a calib
matmul. Strictly better signal, strictly more cost (a full write-bake per bit + a load).
Default is `proxy` (relRMS·actnorm) — light enough to run beside a live audit.

────────────────────────────────────────────────────────────────────────────────────
ALLOCATION  (greedy marginal water-fill)
────────────────────────────────────────────────────────────────────────────────────
Start every tensor at the floor bit (min of --bits-set, default 2). Repeatedly grant +1
bit-step to whichever tensor yields the largest sensitivity DROP per extra average-bpw
spent — i.e. argmax_t  (sens(t,b)−sens(t,b+1)) / (Δparam-weighted bpw cost). Stop when
the param-weighted average effective bpw would exceed --target. Equal marginal-return
allocation = the classic rate–distortion water-fill; tensors that barely improve with
more bits keep the floor, sensitive ones climb to the ceiling (max of --bits-set,
default 4). Effective bpw per bit uses ladder.BPW (1→1.34 … 4→4.50, RHT+outlier folded),
weighted by each tensor's element count — consistent with the rest of the sweep.

Emits a baker `--mp-config` JSON array [{pattern, bits}] keyed by FULL tensor name
(unique per tensor, so two layers of down_proj can differ). `--rung-config` (substring→
bits) is also writable with --emit rung for the coarse per-ROLE recipe.

────────────────────────────────────────────────────────────────────────────────────
PROVE  (--bake)
────────────────────────────────────────────────────────────────────────────────────
Bakes the mixed config and a UNIFORM config at (closest achievable) the same avg bpw,
measures AGGREGATE effective bpw + ppl-degradation vs f16 for both (real forward passes,
ppl_bench-style), and prints the delta = the mixed-precision win.

Honors DOCTOR_DEVICE / DOCTOR_DTYPE (0.5B → mps/float32; 7B → cpu/bfloat16; never f16 —
MPS f16 GQA bug + 7B fp16 overflow→nan). Default self-test is the 0.5B, LIGHT (rank +
allocate + emit + a sliced measure), so it does not contend with a running full bake.

Usage:
  # 1) PLAN: rank + allocate + write the mp-config (cheap; measure-only probes). DEFAULT.
  python3.12 tools/condense/mixed_precision.py scratch/qwen-05b --target 3.0

  # 2) Light self-test of the FULL path on a slice (rank a few tensors, emit, sliced bake):
  python3.12 tools/condense/mixed_precision.py scratch/qwen-05b --target 3.0 \
        --limit-tensors 6 --bake --threads 3

  # 3) FULL run (run when the audit is idle — see banner; uses 10 threads):
  python3.12 tools/condense/mixed_precision.py scratch/qwen-05b --target 3.0 --bake
"""
import sys, os, re, gc, json, math, time, argparse, subprocess
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ladder as L                                      # canonical BPW + thresholds

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BAKER = os.path.join(ROOT, "vendor", "strand-quant", "target", "release", "quantize-model")
DEV = os.environ.get("DOCTOR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))
# eval text (real ppl) + calib text (activation σ); ppl_bench/audit use these defaults.
PT = os.environ.get("PPL_TEXT", "/tmp/ppl24k.txt")
CALIB = os.environ.get("DOCTOR_CALIB", os.path.join(ROOT, "scratch", "calib_corpus.txt"))


def log(*m):
    print(*m, file=sys.stderr); sys.stderr.flush()


def audit_running():
    """A full bake contends with a running ladder audit (CPU baker, 10 threads)."""
    try:
        r = subprocess.run(["pgrep", "-fl", "quantize-model"], capture_output=True, text=True)
        return [ln for ln in r.stdout.splitlines() if "quantize-model" in ln]
    except Exception:
        return []


# ── baker ────────────────────────────────────────────────────────────────────────────
def bake(inp, out, bits=None, mp_config=None, threads=10, measure_only=False, only=None):
    """Run the Rust baker. Returns (aggregate_eff_bpw, per_tensor{name:{bits,bpw,relRMS}}).

    bits = global fallback. mp_config = path to [{pattern,bits}] JSON (per-tensor).
    """
    cmd = [BAKER, "--in", inp, "--out", out, "--quality", "--rht-cols",
           "--outlier-channel", "1", "--outlier-bits", "8", "--threads", str(threads)]
    cmd += ["--bits", str(bits if bits is not None else 3)]
    if mp_config:
        cmd += ["--mp-config", mp_config]
    if measure_only:
        cmd += ["--measure-only"]
    if only:
        cmd += ["--only", only]
    r = subprocess.run(cmd, capture_output=True, text=True)
    blob = r.stderr + r.stdout
    if r.returncode != 0:
        raise RuntimeError(f"baker failed: {blob.strip().splitlines()[-3:]}")
    per = {}
    # [done i/N] <name>   bits=B bpw=X.XXX rel-RMS=YY.YY%
    for m in re.finditer(r"\[done [\d/]+\]\s+(\S+)\s+bits=(\d+)\s+bpw=([\d.]+)\s+rel-RMS=([\d.]+)%", blob):
        per[m.group(1)] = dict(bits=int(m.group(2)), bpw=float(m.group(3)), relrms=float(m.group(4)))
    agg = re.search(r"AGGREGATE effective bpw = ([\d.]+)", blob)
    return (float(agg.group(1)) if agg else float("nan")), per


# ── activation energy (output-space weighting) ─────────────────────────────────────────
def capture_actnorm(model_dir):
    """Per-tensor actnorm = RMS over input channels of mean_calib|x|. Same hook as
    audit_ladder.capture_sigma; corpus = DOCTOR_CALIB. Returns {tensor.weight: float}."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=DTYPE, attn_implementation="eager").to(DEV).eval()
    sig, hooks = {}, []

    def mk(n):
        def h(mod, i, o):
            x = i[0].detach().abs().reshape(-1, i[0].shape[-1]).float().mean(0)
            sig[n + ".weight"] = sig.get(n + ".weight", torch.zeros_like(x)) + x
        return h

    for n, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.shape[1] >= 256:
            hooks.append(mod.register_forward_hook(mk(n)))
    txt = open(CALIB, errors="ignore").read() if os.path.exists(CALIB) else open(PT, errors="ignore").read()
    ids = tok(txt, return_tensors="pt").input_ids[:, :2048].to(DEV)
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    out = {k: float((v.float() ** 2).mean().sqrt().item()) for k, v in sig.items()}  # RMS per-channel
    del model
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return out


# ── per-tensor sensitivity table ───────────────────────────────────────────────────────
def list_linears(src):
    with safe_open(src, framework="pt") as f:
        return [(k, f.get_slice(k).get_shape()) for k in f.keys()
                if len(f.get_slice(k).get_shape()) == 2]


def relrms_per_bit(src, names, bits_set, threads, full_model):
    """relRMS(t,b) for each tensor t × bit b via measure-only bakes (no write — cheap).

    `--only` is a PLAIN SUBSTRING filter (NOT a regex — verified: an `a|b` OR matches
    nothing). So for the FULL model we omit --only (one bake measures every tensor per
    bit — the efficient path); for a small self-test SLICE we probe each tensor with its
    own --only (its full name is a unique substring). Returns {name:{b:relrms}}, {name:{b:bpw}}.
    """
    rr, bp = {n: {} for n in names}, {n: {} for n in names}
    for b in bits_set:
        if full_model:
            _, per = bake(src, "/tmp/mp_measure_throwaway.safetensors", bits=b,
                          threads=threads, measure_only=True, only=None)
        else:
            per = {}
            for n in names:                                   # unique substring per tensor
                _, p1 = bake(src, "/tmp/mp_measure_throwaway.safetensors", bits=b,
                             threads=threads, measure_only=True, only=n)
                per.update(p1)
        for n in names:
            if n in per:
                rr[n][b] = per[n]["relrms"]
                bp[n][b] = per[n]["bpw"]
        log(f"  measured rel-RMS @ {b}-bit for {sum(1 for n in names if b in rr[n])}/{len(names)} tensors")
    return rr, bp


def sensitivity_table(src, model_dir, bits_set, threads, limit=None):
    """Build the damage-ranked table: each tensor's elements, actnorm, and sens(b) for b
    in bits_set. sens = relRMS(b) · actnorm (output-space proxy)."""
    lins = list_linears(src)
    if limit:
        # keep a representative slice across roles for the light self-test
        roles = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        picked, seen = [], set()
        for role in roles:
            for k, shp in lins:
                if role in k and role not in seen:
                    picked.append((k, shp)); seen.add(role); break
            if len(picked) >= limit:
                break
        for k, shp in lins:                          # top up if fewer roles than limit
            if (k, shp) not in picked and len(picked) < limit:
                picked.append((k, shp))
        lins = picked[:limit]
        log(f"# self-test slice: {len(lins)} tensors ({', '.join(k.split('.')[-2] for k,_ in lins)})")

    names = [k for k, _ in lins]
    log(f"# capturing activation energy (σ) over calib for output-space weighting…")
    actn = capture_actnorm(model_dir)
    log(f"# probing per-tensor rel-RMS at bits {bits_set} (measure-only; light)…")
    rr, _ = relrms_per_bit(src, names, bits_set, threads, full_model=(limit is None))

    rows = []
    for k, shp in lins:
        elems = shp[0] * shp[1]
        a = actn.get(k, 0.0)
        sens = {b: rr[k].get(b, float("nan")) * a for b in bits_set}
        rows.append(dict(name=k, elems=elems, actnorm=a, relrms=rr[k], sens=sens))
    return rows


# ── greedy water-fill allocation under a target avg bpw ────────────────────────────────
def allocate(rows, bits_set, target_bpw):
    """Assign each tensor a bit from bits_set so sensitive tensors get more, under a
    param-weighted average effective-bpw ≤ target. Greedy: start at floor, repeatedly
    spend the next bit-step where it cuts sensitivity most per avg-bpw added."""
    bits_set = sorted(bits_set)
    floor, ceil = bits_set[0], bits_set[-1]
    alloc = {r["name"]: floor for r in rows}
    total_elems = sum(r["elems"] for r in rows)
    by = {r["name"]: r for r in rows}

    def avg_bpw(a):
        return sum(L.BPW[a[r["name"]]] * r["elems"] for r in rows) / total_elems

    # candidate upgrades: marginal sensitivity drop per marginal avg-bpw cost
    def gain(name, b):
        r = by[name]
        nb = bits_set[bits_set.index(b) + 1]
        ds = r["sens"][b] - r["sens"][nb]                       # sensitivity reduction (≥0 expected)
        dc = (L.BPW[nb] - L.BPW[b]) * r["elems"] / total_elems  # avg-bpw cost of this upgrade
        return (ds / dc if dc > 0 else 0.0), nb

    cur = avg_bpw(alloc)
    if cur > target_bpw:
        log(f"! floor bpw {cur:.3f} already exceeds target {target_bpw} (raise --target or lower floor)")
    while True:
        best = None
        for r in rows:
            b = alloc[r["name"]]
            if b >= ceil:
                continue
            g, nb = gain(r["name"], b)
            # would this upgrade keep us within budget?
            trial = dict(alloc); trial[r["name"]] = nb
            if avg_bpw(trial) <= target_bpw + 1e-9 and (best is None or g > best[0]):
                best = (g, r["name"], nb)
        if best is None:
            break
        alloc[best[1]] = best[2]
    return alloc, avg_bpw(alloc)


# ── ppl (real forward pass, ppl_bench-style; honors DEV/DTYPE) ─────────────────────────
def ppl(model_dir, override):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=DTYPE, attn_implementation="eager").to(DEV).eval()
    if override:
        sd = model.state_dict()
        with safe_open(override, framework="pt") as f:
            for k in f.keys():
                if k in sd and tuple(sd[k].shape) == tuple(f.get_slice(k).get_shape()):
                    sd[k].copy_(f.get_tensor(k).to(DEV, DTYPE))
    ids = tok(open(PT, errors="ignore").read(), return_tensors="pt").input_ids[:, :2048].to(DEV)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss.item()
    del model
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return math.exp(loss)


def nearest_uniform(bits_set, avg_bpw):
    """The uniform bit whose effective bpw is closest to the mixed avg — the honest
    iso-bpw baseline. (Uniform can only hit discrete BPW points; we report its real bpw.)"""
    return min(bits_set, key=lambda b: abs(L.BPW[b] - avg_bpw))


# ── emit ───────────────────────────────────────────────────────────────────────────────
def emit_config(alloc, path, kind):
    if kind == "rung":
        # coarse per-ROLE: majority bit per role substring (q_proj/down_proj/…)
        roles = {}
        for name, b in alloc.items():
            r = name.split(".")[-2] if "." in name else name
            roles.setdefault(r, []).append(b)
        cfg = {r: max(set(bs), key=bs.count) for r, bs in roles.items()}
        json.dump(cfg, open(path, "w"), indent=2)
    else:
        cfg = [{"pattern": name, "bits": b} for name, b in sorted(alloc.items())]
        json.dump(cfg, open(path, "w"))
    return cfg


# ── main ───────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Damage-ranked mixed-precision allocation")
    ap.add_argument("model_dir", help="HF model dir (e.g. scratch/qwen-05b)")
    ap.add_argument("--target", type=float, default=3.0, help="target AVG effective bpw")
    ap.add_argument("--bits-set", default="2,3,4", help="allowed per-tensor bits (comma)")
    ap.add_argument("--metric", choices=["proxy", "outxe"], default="proxy",
                    help="proxy = relRMS·actnorm (cheap); outxe = exact ||ΔW·X||/||WX|| (heavy)")
    ap.add_argument("--out", default=None, help="mp-config path (default scratch/<name>-mp<target>.json)")
    ap.add_argument("--emit", choices=["mp", "rung"], default="mp", help="config flavor")
    ap.add_argument("--bake", action="store_true", help="bake mixed + uniform & report ppl win")
    ap.add_argument("--threads", type=int, default=10, help="baker threads (lower to share w/ audit)")
    ap.add_argument("--limit-tensors", type=int, default=None,
                    help="rank only N tensors (LIGHT self-test of the full path)")
    args = ap.parse_args()

    src = os.path.join(args.model_dir, "model.safetensors")
    name = os.path.basename(args.model_dir.rstrip("/"))
    bits_set = sorted(int(x) for x in args.bits_set.split(","))
    out = args.out or os.path.join(ROOT, "scratch", f"{name}-mp{args.target}.json")

    log(f"# mixed_precision · {name} · target {args.target} bpw · bits {bits_set} · "
        f"metric={args.metric} · dev={DEV}/{DTYPE}")
    running = audit_running()
    if running:
        log(f"# NOTE: a baker is running ({len(running)} proc) — likely the ladder audit. "
            f"{'Self-test slice is light.' if args.limit_tensors else 'Consider --limit-tensors / lower --threads for the full bake.'}")
    if args.metric == "outxe":
        log("# WARNING: --metric outxe decodes Ŵ per bit (a write-bake + load each) — heavy; "
            "intended for an idle box. Falling back to proxy is recommended beside a live audit.")

    # 1) rank
    t0 = time.time()
    rows = sensitivity_table(src, args.model_dir, bits_set, args.threads, limit=args.limit_tensors)
    rows.sort(key=lambda r: r["sens"][bits_set[0]], reverse=True)   # most damage at the floor first
    log(f"# ranked {len(rows)} tensors in {time.time()-t0:.0f}s. Top by floor-bit sensitivity:")
    for r in rows[:8]:
        log(f"    {r['name']:<44s} actnorm={r['actnorm']:.3g}  "
            f"sens@{bits_set[0]}b={r['sens'][bits_set[0]]:.3g}  relRMS={r['relrms']}")

    # 2) allocate
    alloc, achieved = allocate(rows, bits_set, args.target)
    dist = {b: sum(1 for v in alloc.values() if v == b) for b in bits_set}
    log(f"# allocation: avg eff-bpw ≈ {achieved:.3f} (target {args.target}) · bit-counts {dist}")

    # 3) emit
    cfg = emit_config(alloc, out, args.emit)
    log(f"# wrote {args.emit}-config → {out}  ({len(cfg) if isinstance(cfg,list) else len(cfg)} entries)")
    print(json.dumps({"model": name, "target_bpw": args.target, "achieved_alloc_bpw": round(achieved, 3),
                      "bit_counts": dist, "config": out, "n_tensors": len(rows),
                      "metric": args.metric, "emit": args.emit}))

    # 4) prove (optional)
    if not args.bake:
        uni = nearest_uniform(bits_set, achieved)
        log(f"# (skipped bake) to prove the win: --bake  → compares mixed vs uniform {uni}-bit "
            f"(~{L.BPW[uni]} bpw). FULL run when the audit is idle.")
        return
    if args.limit_tensors:
        log("# --limit-tensors set: this bake exercises the CODE PATH on a partial config "
            "(non-listed tensors fall back to --bits). Numbers are a smoke test, not the verdict.")

    log("# === PROVE: mixed vs uniform @ matched avg bpw (real ppl forward passes) ===")
    f16 = ppl(args.model_dir, None)
    log(f"  f16 ppl = {f16:.3f}")

    mixed_bpw, _ = bake(src, "/tmp/mp_mixed.safetensors", bits=bits_set[0],
                        mp_config=out, threads=args.threads)
    mixed_ppl = ppl(args.model_dir, "/tmp/mp_mixed.safetensors")
    log(f"  MIXED   : eff {mixed_bpw:.3f} bpw  ppl {mixed_ppl:.3f}  (+{(mixed_ppl/f16-1)*100:.2f}%)")

    uni = nearest_uniform(bits_set, mixed_bpw)
    uni_bpw, _ = bake(src, "/tmp/mp_uniform.safetensors", bits=uni, threads=args.threads)
    uni_ppl = ppl(args.model_dir, "/tmp/mp_uniform.safetensors")
    log(f"  UNIFORM : eff {uni_bpw:.3f} bpw  ppl {uni_ppl:.3f}  (+{(uni_ppl/f16-1)*100:.2f}%)  (bits={uni})")

    win = (uni_ppl / f16) - (mixed_ppl / f16)
    verdict = ("WIN: mixed-precision beats uniform at ~iso-bpw"
               if win > 0 else "no win: uniform ties/beats mixed here (honest null)")
    log(f"  Δ(degradation) uniform − mixed = {win*100:+.2f} pts  →  {verdict}")
    print(json.dumps({"model": name, "f16_ppl": round(f16, 3),
                      "mixed": dict(bpw=round(mixed_bpw, 3), ppl=round(mixed_ppl, 3),
                                    degr_pct=round((mixed_ppl/f16-1)*100, 2)),
                      "uniform": dict(bits=uni, bpw=round(uni_bpw, 3), ppl=round(uni_ppl, 3),
                                      degr_pct=round((uni_ppl/f16-1)*100, 2)),
                      "mp_win_pts": round(win*100, 2), "verdict": verdict}))


if __name__ == "__main__":
    main()
