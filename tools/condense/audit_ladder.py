#!/usr/bin/env python3.12
"""4/3/2/1-bit STRAND ladder audit — per-model frontier (quality + REAL effective bpw).

Sweeps RHT / AWQ(alpha) / residual(b1+b2) / AWQ*residual across bit budgets, measures:
  - REAL effective bpw  (parsed from the baker's "AGGREGATE effective bpw" = RHT+outlier+
    residual-pass overhead included — the honest number, not nominal)
  - output-space degradation = ppl(condensed)/ppl(f16) - 1   (real forward passes)
Emits per-config JSONL (for the overlaid curve) + a markdown table (best method per tier).

Memory-safe for the 7B in 19GB: the model is FREED during every bake/build (so the Rust baker
and the Python parent are never both holding ~14GB) and RELOADED per measurement, with overrides
STREAMED into the model in-place (peak ~= one model copy). Slow but fits. DOCTOR_DEVICE/DTYPE
honored (0.5B: mps/float32; 7B: cpu/bfloat16 — fp16 overflows the 7B CPU forward -> nan).

Usage: audit_ladder.py <hf-dir> <label> <full|essential> [out_prefix]
"""
import sys, os, re, gc, json, math, subprocess, shutil, atexit, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file

MODEL = sys.argv[1]; LABEL = sys.argv[2]
SETNAME = sys.argv[3] if len(sys.argv) > 3 else "essential"
OUTP = sys.argv[4] if len(sys.argv) > 4 else f"/tmp/audit_{LABEL}"
BAKER = "vendor/strand-quant/target/release/quantize-model"
DEV = os.environ.get("DOCTOR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))
PT = os.environ.get("PPL_TEXT", "/tmp/ppl24k.txt")
def _resolve_src():
    """The baker needs ONE safetensors. 0.5B is single-file; 7B+ is sharded -> consolidate once.
    Streaming: peak ~one tensor, never loads the full 14GB dict into RAM simultaneously."""
    single = os.path.join(MODEL, "model.safetensors")
    if os.path.exists(single):
        return single
    cons = f"/tmp/aud_{LABEL}_src.safetensors"
    if not os.path.exists(cons):
        idxp = os.path.join(MODEL, "model.safetensors.index.json")
        weight_map = json.load(open(idxp))["weight_map"]
        shards = sorted(set(weight_map.values()))
        shard_fhs = {sh: safe_open(os.path.join(MODEL, sh), framework="pt") for sh in shards}
        all_keys = []
        for sh in shards:
            all_keys.extend(shard_fhs[sh].keys())
        def _spec(k):
            sl = shard_fhs[weight_map[k]].get_slice(k)
            return (_str_dtype(sl.get_dtype()), tuple(sl.get_shape()))
        def _produce(k):
            return shard_fhs[weight_map[k]].get_tensor(k)
        stream_save(cons, all_keys, _spec, _produce)
        gc.collect()
    return cons


SRC = _resolve_src()
T = f"/tmp/aud_{LABEL}"                     # reused temp prefix (overwritten per config)
SIGPATH = f"{T}_sigma.safetensors"


def log(m): print(m, file=sys.stderr); sys.stderr.flush()


# ---- streaming safetensors writer (peak ~one tensor, not the whole ~15GB dict) ----
_ST_DTYPE = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32",
             torch.float64: "F64", torch.int64: "I64", torch.int32: "I32",
             torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL"}
_TORCH_DTYPE = {v: k for k, v in _ST_DTYPE.items()}


def _str_dtype(s):
    """safetensors dtype string (e.g. 'BF16') -> torch dtype."""
    return _TORCH_DTYPE[s]


def _raw_bytes(t):
    """Raw little-endian bytes of a tensor in safetensors layout (works for bf16, which has no numpy dtype)."""
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def stream_save(out_path, names, spec, produce):
    """Write a safetensors file one tensor at a time.

    names   : ordered list of tensor names (write order = data-blob order).
    spec(k) : -> (torch_dtype, shape_tuple) for the FINAL tensor (cheap, no materialization).
    produce(k): -> the final tensor for key k (built on demand, freed right after write).

    Transforms here preserve dtype+shape, so spec() reads them off the lazy source slices;
    only produce() materializes a tensor, and only one at a time. Peak ~= one tensor.
    """
    header, off = {}, 0
    for k in names:
        dt, shape = spec(k)
        nbytes = 1
        for d in shape:
            nbytes *= d
        nbytes *= torch.empty(0, dtype=dt).element_size()
        header[k] = {"dtype": _ST_DTYPE[dt], "shape": list(shape), "data_offsets": [off, off + nbytes]}
        off += nbytes
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-(len(hb) + 8)) % 8                      # 8-byte align the data blob (pad header w/ spaces)
    hb += b" " * pad
    with open(out_path, "wb") as fo:
        fo.write(len(hb).to_bytes(8, "little"))
        fo.write(hb)
        for k in names:
            t = produce(k)
            b = _raw_bytes(t)
            exp = header[k]["data_offsets"][1] - header[k]["data_offsets"][0]
            if len(b) != exp:
                raise RuntimeError(f"stream_save size mismatch {k}: wrote {len(b)} expected {exp}")
            fo.write(b)
            del t, b


def _free_gb(path="/tmp"):
    return shutil.disk_usage(path).free / 1e9


def _clean_temps(keep=(SRC, SIGPATH)):
    """Remove every {T}_* intermediate (b1/b2/rin/scaled/baked/ovr), keep SRC + sigma.
    Used to recover from an interrupted config before retrying."""
    import glob
    for p in glob.glob(f"{T}_*.safetensors"):
        if p not in keep:
            try: os.remove(p)
            except OSError: pass


def _ensure_disk(need_gb, label):
    """Pre-flight: guarantee `need_gb` free on /tmp before a config writes intermediates.
    If short, sweep orphaned temps and recheck; if STILL short, raise so main() logs the
    config as skipped and the ladder CONTINUES — never let a write hit ENOSPC mid-stream."""
    free = _free_gb()
    if free < need_gb:
        log(f"  [disk] {label}: {free:.1f}GB < {need_gb}GB need — sweeping orphan temps")
        _clean_temps()
        free = _free_gb()
    if free < need_gb:
        raise RuntimeError(f"insufficient disk: {free:.1f}GB free < {need_gb}GB needed (skipped, ladder continues)")


def _bake_one(inp, out, bits, rung=None):
    """ONE raw baker invocation. --quality removed (L=k+4 fast, accurate for curve shape);
    STRAND_NO_GPU forces CPU so the baker never fights the model measurement for Metal memory.
    rung = path to a flat JSON {substr: bits} for MIXED-PRECISION (per-layer bit allocation,
    e.g. 4-bit attention / 3-bit FFN); --bits is the fallback for tensors matching no rule."""
    nt = int(os.environ.get("BAKE_THREADS") or 4)
    env = {**os.environ, "STRAND_NO_GPU": "1"}
    cmd = [BAKER, "--in", inp, "--out", out, "--bits", str(bits),
           "--rht-cols", "--outlier-channel", "1", "--outlier-bits", "8", "--threads", str(nt)]
    if rung:
        cmd += ["--rung-config", rung]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"baker failed bits={bits}: {r.stderr[-200:]}")
    # parse the baker's TRUE quantized-weight count too — embed/lm_head pass through unquantized
    # (bpw 0 over 0 weights), so we weight the per-chunk aggregate by the baker's count, not our
    # own _isq guess (which would wrongly include the 1B+ embedding params and dilute eff-bpw).
    m = re.search(r"AGGREGATE effective bpw = ([0-9.]+) over ([0-9]+) quantized", r.stderr + r.stdout)
    if m:
        return float(m.group(1)), int(m.group(2))
    return float("nan"), 0


#   CHUNKED baking — the memory fix for 7B on a 19GB Mac.
#   The Rust baker (quantize-model.rs write_safetensors) accumulates the ENTIRE recon as F32 in
#   RAM before writing — ~28GB for a 7B (F32 = 2× the bf16 model) — and OOM-dies partway
#   (observed 2026-06-23: killed at tensor 70/196, swap pinned ~40GB; thread-count independent).
#   Per-tensor trellis quant is INDEPENDENT, so we feed the baker ONE chunk of tensors at a time:
#   it only ever accumulates one chunk's F32 recon (RAM bounded), and we never materialize a full
#   scaled/baked/rin/ovr file (disk bounded) — each builder streams its transform per chunk into a
#   small bf16 "part", and measure() applies the parts together. Bit-identical per tensor.
def _src_specs(path=None):
    with safe_open(path or SRC, framework="pt") as f:
        names = list(f.keys())
        specs = {k: (_str_dtype(f.get_slice(k).get_dtype()), tuple(f.get_slice(k).get_shape())) for k in names}
    return names, specs


def _isq(specs, k):
    """Approx of the baker's is_quantizable_linear: 2D, both dims >= 256."""
    sh = specs[k][1]
    return len(sh) == 2 and min(sh) >= 256


def _chunks(names, specs):
    """Partition the ordered tensor list into ~BAKE_CHUNKS equal-BYTE groups (whole tensors)."""
    n = int(os.environ.get("BAKE_CHUNKS") or 6)
    esz = {dt: torch.empty(0, dtype=dt).element_size() for dt in {d for d, _ in specs.values()}}
    def nb(k):
        dt, sh = specs[k]; m = 1
        for d in sh: m *= d
        return m * esz[dt]
    target = sum(nb(k) for k in names) / max(1, n)
    out, cur, cb = [], [], 0
    for k in names:
        cur.append(k); cb += nb(k)
        if cb >= target and len(out) < n - 1:
            out.append(cur); cur, cb = [], 0
    if cur: out.append(cur)
    return out


def _bake_chunk(cnames, specs, produce, bits, rung=None):
    """Bake ONE chunk: stream its (pre-transformed, bf16) inputs, run the baker, return the recon
    as a small in-RAM dict (this chunk only) + the chunk's aggregate bpw. Cleans its own temps.
    This is the unit that bounds memory — the baker accumulates only one chunk's F32 recon."""
    cin, cout = f"{T}_ckin.safetensors", f"{T}_ckout.safetensors"
    try:
        stream_save(cin, cnames, lambda k: (DTYPE, specs[k][1]), produce)
        bpw, qcount = _bake_one(cin, cout, bits, rung=rung)
        recon = {}
        with safe_open(cout, framework="pt") as fc:
            for k in fc.keys():
                recon[k] = fc.get_tensor(k).to(DTYPE)
    finally:
        for p in (cin, cout):
            try: os.remove(p)
            except OSError: pass
    return recon, bpw, qcount


def _emit_part(ci, cnames, specs, produce):
    """Write one chunk's final (transformed) tensors as a bf16 override part; return its path."""
    part = f"{T}_p{ci}.safetensors"
    stream_save(part, cnames, lambda k: (DTYPE, specs[k][1]), produce)
    return part


def keys_2d(path):
    with safe_open(path, framework="pt") as f:
        return [k for k in f.keys() if len(f.get_slice(k).get_shape()) == 2]


def quant_keys(base_decoded):
    """Tensors the baker actually quantized = 2D and changed vs SRC."""
    qk = set()
    with safe_open(SRC, framework="pt") as fs, safe_open(base_decoded, framework="pt") as fb:
        bk = set(fb.keys())
        for k in fs.keys():
            if k not in bk:
                continue
            v = fs.get_tensor(k)
            if v.dim() == 2 and not torch.equal(fb.get_tensor(k).to(v.dtype), v):
                qk.add(k)
    return qk


# ---- builders: each returns (list-of-override-parts, effective_bpw). Model NOT loaded here.
#      Every builder loops chunks so RAM stays ~one chunk and disk never holds a full temp. ----
def _wavg(pairs):
    w = sum(b * q for b, q in pairs if not math.isnan(b))
    q = sum(q for b, q in pairs if not math.isnan(b))
    return w / q if q else float("nan")


def build_rht(bits):
    names, specs = _src_specs()
    chunks = _chunks(names, specs)
    log(f"  [bake] {bits}-RHT in {len(chunks)} chunks")
    parts, bpws = [], []
    with safe_open(SRC, framework="pt") as fs:
        for ci, cn in enumerate(chunks):
            _ensure_disk(16, f"{bits}-RHT ck{ci+1}/{len(chunks)}")
            recon, bpw, qc = _bake_chunk(cn, specs, lambda k: fs.get_tensor(k).to(DTYPE), bits)
            parts.append(_emit_part(ci, cn, specs, lambda k: recon[k]))
            bpws.append((bpw, qc))
            del recon; gc.collect()
            log(f"  [bake]  {bits}-RHT ck{ci+1}/{len(chunks)} bpw={bpw:.3f} q={qc} free={_free_gb():.0f}GB")
    return parts, _wavg(bpws)


def build_awq(bits, alpha=0.5, rung=None):
    """AWQ (scale cols by sigma^alpha, bake, unscale). rung = mixed-precision dict {substr: bits}
    written to a temp json and passed to the baker; `bits` is then the per-tensor FALLBACK."""
    sig = {}
    with safe_open(SIGPATH, framework="pt") as f:
        for k in f.keys():
            sig[k] = f.get_tensor(k)
    rpath = None
    if rung:
        rpath = f"{T}_rung.json"
        json.dump(rung, open(rpath, "w"))
    label = ("mp" if rung else f"{bits}") + f"-AWQ.{int(alpha*100)}"
    names, specs = _src_specs()
    chunks = _chunks(names, specs)
    log(f"  [bake] {label} in {len(chunks)} chunks{' (mixed-prec)' if rung else ''}")
    parts, bpws = [], []
    with safe_open(SRC, framework="pt") as fs:
        for ci, cn in enumerate(chunks):
            _ensure_disk(16, f"{label} ck{ci+1}/{len(chunks)}")
            def scaled(k):                                   # scale cols by sigma^alpha pre-bake
                v = fs.get_tensor(k)
                return (v.float() * sig[k].pow(alpha)).to(DTYPE) if k in sig else v.to(DTYPE)
            recon, bpw, qc = _bake_chunk(cn, specs, scaled, bits, rung=rpath)
            def unscaled(k):                                 # undo the scale post-bake
                return (recon[k].float() / sig[k].pow(alpha)).to(DTYPE) if k in sig else recon[k]
            parts.append(_emit_part(ci, cn, specs, unscaled))
            bpws.append((bpw, qc))
            del recon; gc.collect()
            log(f"  [bake]  {label} ck{ci+1}/{len(chunks)} bpw={bpw:.3f} q={qc} free={_free_gb():.0f}GB")
    return parts, _wavg(bpws)


def build_residual(b1, b2):
    """W ≈ STRAND(W) + STRAND(W − STRAND(W)). Two bakes per chunk, fully streamed — never holds a
    full b1/rin/b2/ovr file (the old 52GB-peak that forced residual to skip on this disk)."""
    names, specs = _src_specs()
    chunks = _chunks(names, specs)
    log(f"  [bake] res{b1}+{b2} in {len(chunks)} chunks (2 bakes/chunk)")
    parts, bp1, bp2 = [], [], []
    with safe_open(SRC, framework="pt") as fs:
        for ci, cn in enumerate(chunks):
            _ensure_disk(16, f"res{b1}+{b2} ck{ci+1}/{len(chunks)}")
            r1, w1, qc1 = _bake_chunk(cn, specs, lambda k: fs.get_tensor(k).to(DTYPE), b1)
            def rin(k):                                      # residual = SRC − b1(SRC) on quant tensors
                v = fs.get_tensor(k)
                return (v.float() - r1[k].float()).to(DTYPE) if _isq(specs, k) else v.to(DTYPE)
            r2, w2, qc2 = _bake_chunk(cn, specs, rin, b2)
            def recon(k):                                    # reconstruct = b1 + b2
                return (r1[k].float() + r2[k].float()).to(DTYPE) if _isq(specs, k) else r1[k]
            parts.append(_emit_part(ci, cn, specs, recon))
            bp1.append((w1, qc1)); bp2.append((w2, qc2))
            del r1, r2; gc.collect()
            log(f"  [bake]  res ck{ci+1}/{len(chunks)} bpw={w1:.2f}+{w2:.2f} q={qc1} free={_free_gb():.0f}GB")
    return parts, _wavg(bp1) + _wavg(bp2)


# ---- measurement: load model, stream override in-place, ppl, free ----
def measure(override):
    torch.set_num_threads(os.cpu_count() or 12)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, attn_implementation="eager").to(DEV).eval()
    if override:
        paths = [override] if isinstance(override, str) else list(override)
        sd = model.state_dict()
        for pth in paths:                          # builders now return a LIST of chunk parts
            with safe_open(pth, framework="pt") as f:
                for k in f.keys():
                    if k in sd and tuple(sd[k].shape) == tuple(f.get_slice(k).get_shape()):
                        sd[k].copy_(f.get_tensor(k).to(DEV, DTYPE))
    text = open(PT, errors="ignore").read()
    ids = tok(text, return_tensors="pt").input_ids[:, :2048].to(DEV)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss.item()
    del model; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return math.exp(loss)


def capture_sigma():
    if os.path.exists(SIGPATH):
        log(f"# sigma already exists at {SIGPATH}, skipping capture")
        return
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, attn_implementation="eager").to(DEV).eval()
    sig, hooks = {}, []
    def mk(n):
        def h(m, i, o):
            x = i[0].detach().abs().reshape(-1, i[0].shape[-1]).float().mean(0)
            sig[n + ".weight"] = sig.get(n + ".weight", torch.zeros_like(x)) + x
        return h
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.shape[1] >= 256:
            hooks.append(m.register_forward_hook(mk(n)))
    ids = tok(open(PT, errors="ignore").read(), return_tensors="pt").input_ids[:, :2048].to(DEV)
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    save_file({k: (v.cpu().float() + 1e-6) for k, v in sig.items()}, SIGPATH)
    del model; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()


CONFIGS = {
    "full": [("4-RHT", build_rht, (4,)), ("4-AWQ", build_awq, (4,)),
             ("3-RHT", build_rht, (3,)), ("3-AWQ", build_awq, (3,)),
             ("3-AWQ.25", build_awq, (3, 0.25)), ("3-AWQ.75", build_awq, (3, 0.75)),
             ("2-RHT", build_rht, (2,)), ("2-AWQ", build_awq, (2,)),
             ("1-RHT", build_rht, (1,)), ("1-AWQ", build_awq, (1,)),
             ("res3+2", build_residual, (3, 2)), ("res2+2", build_residual, (2, 2)),
             ("res2+1", build_residual, (2, 1)), ("res1+1", build_residual, (1, 1))],
    "essential": [("4-AWQ", build_awq, (4,)), ("3-AWQ", build_awq, (3,)),
                  ("2-AWQ", build_awq, (2,)), ("1-AWQ", build_awq, (1,)), ("1-RHT", build_rht, (1,)),
                  ("res3+2", build_residual, (3, 2)), ("res2+2", build_residual, (2, 2)),
                  ("res2+1", build_residual, (2, 1)), ("res1+1", build_residual, (1, 1))],
    # comprehensive 4/3/2/1-bit local frontier: mixed-precision (measured density win) + the
    # 3-bit floor alpha-sweep + 2-bit rescue attempts. Known essential results are seeded into
    # the JSONL so they skip; ordered most-valuable-first so the best data lands while watched.
    "frontier": [
        ("mp-4a3f", build_awq, (3, 0.5, {"q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4,
                                          "gate_proj": 3, "up_proj": 3, "down_proj": 3})),
        ("mp-4a2f", build_awq, (2, 0.5, {"q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4,
                                          "gate_proj": 2, "up_proj": 2, "down_proj": 2})),
        ("mp-3a2f", build_awq, (2, 0.5, {"q_proj": 3, "k_proj": 3, "v_proj": 3, "o_proj": 3,
                                          "gate_proj": 2, "up_proj": 2, "down_proj": 2})),
        ("3-AWQ.25", build_awq, (3, 0.25)), ("3-AWQ.75", build_awq, (3, 0.75)),
        ("3-RHT", build_rht, (3,)),
        ("2-AWQ.25", build_awq, (2, 0.25)), ("2-AWQ.75", build_awq, (2, 0.75)),
        ("2-RHT", build_rht, (2,)),
        ("4-RHT", build_rht, (4,)),
        # seeded/known (skip on resume):
        ("4-AWQ", build_awq, (4,)), ("3-AWQ", build_awq, (3,)), ("2-AWQ", build_awq, (2,)),
        ("1-AWQ", build_awq, (1,)), ("1-RHT", build_rht, (1,))],
}


def _try_inject():
    """Execute an inject script if present, then delete it. Drop a Python file at
    {OUTP}_inject.py to modify globals (e.g. change bake thread count, skip a config,
    redefine a build function) between configs without restarting the process."""
    ipath = f"{OUTP}_inject.py"
    if not os.path.exists(ipath):
        return
    code = open(ipath).read()
    os.remove(ipath)
    log(f"# INJECT: {ipath}")
    try:
        exec(code, globals())  # noqa: S102 — intentional, user-controlled
        log("# INJECT: ok")
    except Exception as e:
        log(f"# INJECT: error — {e}")


def _acquire_lock():
    """Airtight single-instance guard. Two ladder processes sharing the {T}_* temp paths
    race the disk to ENOSPC (observed 2026-06-23). Refuse to start if a live instance holds
    the lock; reclaim a stale lock whose PID is dead. Released on clean exit."""
    lock = f"{OUTP}.lock"
    if os.path.exists(lock):
        try:
            other = int(open(lock).read().strip())
            os.kill(other, 0)               # alive?
            log(f"# ABORT: another ladder instance is live (PID {other}, lock {lock}); exiting")
            sys.exit(0)
        except (ValueError, ProcessLookupError, PermissionError):
            log(f"# reclaiming stale lock {lock}")
    open(lock, "w").write(str(os.getpid()) + "\n")
    def _release():
        try:
            if os.path.exists(lock) and int(open(lock).read().strip()) == os.getpid():
                os.remove(lock)
        except Exception:
            pass
    atexit.register(_release)


def main():
    _acquire_lock()
    log(f"# audit {LABEL} dev={DEV} dtype={DTYPE} set={SETNAME} cpu={os.cpu_count()} free={_free_gb():.0f}GB")
    capture_sigma()

    # --- checkpoint resume: scan JSONL for already-completed configs ---
    completed = {}
    if os.path.exists(f"{OUTP}.jsonl"):
        for line in open(f"{OUTP}.jsonl"):
            try:
                rec = json.loads(line.strip())
                cfg = rec.get("config")
                if cfg and "ppl" in rec:
                    completed[cfg] = rec
            except Exception:
                pass
    if completed:
        log(f"# resume: {len(completed)} checkpointed — {sorted(completed)}")

    # --- f16 baseline (resumable) ---
    if "f16" in completed:
        hf = completed["f16"]["ppl"]
        log(f"# f16 ppl = {hf:.3f} (checkpoint)")
    else:
        hf = measure(None)
        log(f"# f16 ppl = {hf:.3f}")
        open(f"{OUTP}.jsonl", "a").write(
            json.dumps({"model": LABEL, "config": "f16", "eff_bpw": 16.0,
                        "ppl": round(hf, 3), "degr_pct": 0.0}) + "\n")

    rows = list(completed.values())
    for name, fn, args in CONFIGS[SETNAME]:
        _try_inject()  # execute inject script if present before each config

        if name in completed:
            log(f"  {name:10s} -> SKIP (checkpoint +{completed[name].get('degr_pct','?')}%)")
            continue

        _clean_temps()   # sweep any leftover chunk parts/inputs from an interrupted run

        try:
            path, bpw = fn(*args)
            p = measure(path)
            rec = {"model": LABEL, "config": name, "eff_bpw": round(bpw, 3),
                   "ppl": round(p, 3), "degr_pct": round((p / hf - 1) * 100, 2)}
        except Exception as e:
            rec = {"model": LABEL, "config": name, "error": str(e)[:140]}
        rows.append(rec)
        log(f"  {name:10s} -> {rec.get('eff_bpw','?')} bpw  +{rec.get('degr_pct','?')}%")
        open(f"{OUTP}.jsonl", "a").write(json.dumps(rec) + "\n")
        _clean_temps()   # free this config's parts before the next one
    # markdown
    with open(f"{OUTP}.md", "w") as o:
        o.write(f"## {LABEL} ladder (f16 ppl {hf:.2f}) — effective bpw vs degradation\n\n")
        o.write("| config | eff bpw | degr vs f16 |\n|---|--:|--:|\n")
        for r in sorted([x for x in rows if "error" not in x], key=lambda x: x["eff_bpw"]):
            o.write(f"| {r['config']} | {r['eff_bpw']:.2f} | +{r['degr_pct']:.1f}% |\n")
        for r in [x for x in rows if "error" in x]:
            o.write(f"| {r['config']} | ERR | {r['error']} |\n")
    log(f"# done -> {OUTP}.md / {OUTP}.jsonl")


main()
