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
import torch.nn.functional as F
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


SRC = None
T = f"/tmp/aud_{LABEL}"                     # reused temp prefix (overwritten per config)
SIGPATH = os.environ.get("LADDER_SIGMA", f"{OUTP}_sigma.safetensors")


def _free_gb(path="/tmp"):
    return shutil.disk_usage(path).free / 1e9


def _append_jsonl_durable(path, row):
    """Commit one completed-config checkpoint before deleting its temporary artifacts."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _ensure_ppl_text():
    """Recreate the small eval/capture corpus if /tmp was wiped by a reboot."""
    if os.path.exists(PT):
        return
    chunks = []
    if os.path.exists("README.md"):
        chunks.append(open("README.md", errors="ignore").read())
    import glob
    for path in sorted(glob.glob("docs/plans/*.md")):
        chunks.append(open(path, errors="ignore").read())
    text = "\n".join(chunks)[:24000]
    if not text:
        raise FileNotFoundError(PT)
    with open(PT, "w") as f:
        f.write(text)


def _clean_temps(keep=None):
    """Remove every {T}_* intermediate (b1/b2/rin/scaled/baked/ovr), keep SRC + sigma.
    Used to recover from an interrupted config before retrying."""
    import glob
    if keep is None:
        keep = (SRC, SIGPATH)
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


def _bake_one(inp, out, bits, rung=None, outlier_pct=1.0):
    """ONE raw baker invocation. --quality removed (L=k+4 fast, accurate for curve shape);
    STRAND_NO_GPU forces CPU so the baker never fights the model measurement for Metal memory.
    rung = path to a flat JSON {substr: bits} for MIXED-PRECISION (per-layer bit allocation,
    e.g. 4-bit attention / 3-bit FFN); --bits is the fallback for tensors matching no rule.
    outlier_pct = % of top-|w| weights kept as an 8-bit sparse channel — the train-free rescue
    for sub-3-bit (at 1-bit most signal dies; keeping the top 5-10% at 8-bit can recover a lot)."""
    nt = int(os.environ.get("BAKE_THREADS") or 4)
    env = {**os.environ, "STRAND_NO_GPU": "1"}
    cmd = [BAKER, "--in", inp, "--out", out, "--bits", str(bits), "--rht-cols",
           "--outlier-channel", str(outlier_pct), "--outlier-bits", "8", "--threads", str(nt)]
    if rung:
        cmd += ["--rung-config", rung]
    # ---- frontier quality levers (env-gated; inert unless a recipe turns them on) ----
    # --actmean: activation-mean output de-bias (baker comment: -28.7% PPL for ~0.014 bpw).
    #   c_i = -Σ_j (recon_ij - orig_ij)·E[x_j]; the actmean json is written by capture_sigma.
    am = os.environ.get("BAKE_ACTMEAN")
    if am and os.path.exists(am):
        cmd += ["--actmean", am]
    # --quality: deeper trellis (L=bits+6 vs +4). Lower PPL, ~4× slower on CPU. Chunked baking
    #   now bounds RAM regardless of L, so the old 7B OOM that forced this off no longer applies.
    if os.environ.get("BAKE_QUALITY") == "1":
        cmd += ["--quality"]
    # --vec-dim N (+--learned-codebook): d-dim vector/codebook trellis (AQLM-class). payload k/d
    #   bpw — the real lever for the sub-3-bit tier where scalar quant collapses.
    vd = int(os.environ.get("BAKE_VECDIM", "1"))
    if vd > 1:
        cmd += ["--vec-dim", str(vd)]
        if os.environ.get("BAKE_LEARNED_CB") == "1":
            cmd += ["--learned-codebook"]
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


def _bake_chunk(cnames, specs, produce, bits, rung=None, outlier_pct=1.0):
    """Bake ONE chunk: stream its (pre-transformed, bf16) inputs, run the baker, return the recon
    as a small in-RAM dict (this chunk only) + the chunk's aggregate bpw. Cleans its own temps.
    This is the unit that bounds memory — the baker accumulates only one chunk's F32 recon."""
    cin, cout = f"{T}_ckin.safetensors", f"{T}_ckout.safetensors"
    try:
        stream_save(cin, cnames, lambda k: (DTYPE, specs[k][1]), produce)
        bpw, qcount = _bake_one(cin, cout, bits, rung=rung, outlier_pct=outlier_pct)
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


def build_rht(bits, outlier_pct=1.0):
    names, specs = _src_specs()
    chunks = _chunks(names, specs)
    log(f"  [bake] {bits}-RHT in {len(chunks)} chunks")
    parts, bpws = [], []
    with safe_open(SRC, framework="pt") as fs:
        for ci, cn in enumerate(chunks):
            _ensure_disk(16, f"{bits}-RHT ck{ci+1}/{len(chunks)}")
            recon, bpw, qc = _bake_chunk(cn, specs, lambda k: fs.get_tensor(k).to(DTYPE), bits, outlier_pct=outlier_pct)
            parts.append(_emit_part(ci, cn, specs, lambda k: recon[k]))
            bpws.append((bpw, qc))
            del recon; gc.collect()
            log(f"  [bake]  {bits}-RHT ck{ci+1}/{len(chunks)} bpw={bpw:.3f} q={qc} free={_free_gb():.0f}GB")
    return parts, _wavg(bpws)


def build_awq(bits, alpha=0.5, rung=None, outlier_pct=1.0):
    """AWQ (scale cols by sigma^alpha, bake, unscale). rung = mixed-precision dict {substr: bits}
    written to a temp json and passed to the baker; `bits` is then the per-tensor FALLBACK.
    outlier_pct = top-|w| 8-bit sparse channel size (sub-3-bit train-free rescue)."""
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
            recon, bpw, qc = _bake_chunk(cn, specs, scaled, bits, rung=rpath, outlier_pct=outlier_pct)
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


def _merge_parts(parts, out):
    """Stream chunk parts into one safetensors (doctor_lora.py needs a single base file)."""
    fhs = {p: safe_open(p, framework="pt") for p in parts}
    try:
        order = [k for p in parts for k in fhs[p].keys()]
        loc = {k: p for p in parts for k in fhs[p].keys()}
        shp = {k: tuple(fhs[loc[k]].get_slice(k).get_shape()) for k in order}
        stream_save(out, order, lambda k: (DTYPE, shp[k]), lambda k: fhs[loc[k]].get_tensor(k))
    finally:
        for fh in fhs.values():
            if hasattr(fh, "close"):
                fh.close()


def _swap_mb():
    """Current macOS swap used in MB."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True).stdout
        return float(out.split()[5].rstrip("M,"))
    except Exception:
        return 0.0


def build_recover(bits, steps=60, rank=64, lr=1e-4, alpha=0.5, rung=None,
                  outlier_pct=1.0, target_regex=None, kd_topk=None):
    """RECOVERY track: AWQ-quantize, then HEAL with LoRA-KD (doctor_lora.py) — the only path to
    usable sub-3-bit, where PTQ alone collapses. The doctor caches+frees the teacher (one model
    in RAM, 19GB-safe) and saves LoRA adapters, so measurement applies base+adapter in-place
    without ever materializing a fused full-weight checkpoint.
    Returns ({"base": base, "adapter": adapter}, base_bpw + LoRA overhead).
    Slow on CPU — wall-clock is not the constraint.

    Self-managing resource guard: monitors doctor subprocess every 60s.
      DOCTOR_TIMEOUT          max wall-clock seconds before checkpointed stop
      DOCTOR_SWAP_CEIL        soft swap telemetry threshold
      DOCTOR_SWAP_HARD_CEIL   hard swap threshold before checkpointed stop
    On breach: preserve the best/latest adapter if one exists and ladder CONTINUES."""
    import time
    # timeout: long leash by default; progress is adapter-checkpointed, so time is useful.
    TIMEOUT   = int(os.environ.get("DOCTOR_TIMEOUT", str(max(28800, steps * 480))))
    # soft swap is telemetry only; hard swap asks the doctor to checkpoint before SIGKILL.
    SWAP_CEIL = float(os.environ.get("DOCTOR_SWAP_CEIL", "12000"))
    HARD_SWAP = float(os.environ.get("DOCTOR_SWAP_HARD_CEIL", "18000"))
    GRACE     = int(os.environ.get("DOCTOR_TERMINATE_GRACE", "600"))
    USE_PARTIAL = os.environ.get("DOCTOR_USE_PARTIAL", "1").lower() in {"1", "true", "yes"}
    # Loading is exactly when a wrong model/dtype can exhaust unified memory; no blind warmup.
    WARMUP    = int(os.environ.get("DOCTOR_RESOURCE_WARMUP", "0"))

    parts, base_bpw = build_awq(bits, alpha, rung, outlier_pct)
    base = f"{T}_rbase.safetensors"
    _merge_parts(parts, base)
    for p in parts:
        try: os.remove(p)
        except OSError: pass
    adapter = f"{T}_adapter.safetensors"
    latest = f"{T}_adapter.latest.safetensors"
    progress = f"{T}_doctor_progress.jsonl"
    dout = f"{T}_doctor_stdout.log"
    derr = f"{T}_doctor_stderr.log"
    env = {**os.environ, "DOCTOR_MODEL": MODEL, "DOCTOR_DTYPE": "bfloat16", "DOCTOR_DEVICE": DEV,
           "DOCTOR_THREADS": str(os.cpu_count() or 8),
           "DOCTOR_GRAD_ACCUM": os.environ.get("DOCTOR_GRAD_ACCUM", "4"),
           "DOCTOR_SAVE_MODE": "adapter",
           "DOCTOR_PROGRESS": progress,
           "DOCTOR_LATEST": latest,
           "KD": "1", "KD_TOPK": str(kd_topk or os.environ.get("KD_TOPK", "64"))}
    if target_regex:
        env["DOCTOR_TARGET_REGEX"] = target_regex
    cal = "scratch/calib_corpus.txt"
    if os.path.exists(cal):
        env["DOCTOR_CALIB"] = cal
    log(f"  [recover] {bits}b base bpw={base_bpw:.3f}; LoRA-KD r{rank} {steps} steps"
        f" (timeout={TIMEOUT}s soft_swap={SWAP_CEIL:.0f}MB hard_swap={HARD_SWAP:.0f}MB)…")
    out_fh, err_fh = open(dout, "w"), open(derr, "w")
    proc = subprocess.Popen(["python3.12", "tools/condense/doctor.py", "lora", base,
                             str(steps), str(lr), str(rank), adapter],
                            stdout=out_fh, stderr=err_fh,
                            text=True, env=env, start_new_session=True)
    t0 = time.monotonic()
    terminating_at = None
    terminated_reason = None
    while proc.poll() is None:
        time.sleep(60)
        elapsed = time.monotonic() - t0
        swap = _swap_mb()
        log(f"  [recover] elapsed={elapsed/60:.0f}m swap={swap:.0f}MB pid={proc.pid}")
        if elapsed > WARMUP and swap > SWAP_CEIL:
            log(f"  [recover] soft swap leash crossed ({swap:.0f}>{SWAP_CEIL:.0f}MB); continuing under watch")
        if terminating_at is None and elapsed > WARMUP and swap > HARD_SWAP:
            terminating_at = time.monotonic()
            terminated_reason = f"hard swap {swap:.0f}MB > {HARD_SWAP:.0f}MB"
            log(f"  [recover] {terminated_reason}; SIGTERM doctor for checkpoint")
            proc.terminate()
        if terminating_at is None and elapsed > TIMEOUT:
            terminating_at = time.monotonic()
            terminated_reason = f"timeout {elapsed/60:.0f}m > {TIMEOUT/60:.0f}m"
            log(f"  [recover] {terminated_reason}; SIGTERM doctor for checkpoint")
            proc.terminate()
        if terminating_at is not None and time.monotonic() - terminating_at > GRACE:
            log(f"  [recover] checkpoint grace expired after {GRACE}s; SIGKILL doctor")
            proc.kill()
    proc.wait()
    out_fh.close(); err_fh.close()
    stdout = open(dout, errors="ignore").read() if os.path.exists(dout) else ""
    stderr = open(derr, errors="ignore").read() if os.path.exists(derr) else ""
    r = subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
    artifact = adapter if os.path.exists(adapter) else latest if os.path.exists(latest) else None
    if r.returncode != 0 and not (USE_PARTIAL and artifact):
        try: os.remove(base)
        except OSError: pass
        raise RuntimeError(f"doctor failed bits={bits}: {r.stderr[-200:]}")
    try:
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        log(f"  [recover] base_ppl={rec.get('base_ppl'):.1f} -> lora_ppl={rec.get('lora_ppl'):.1f}"
            f" (recovered {rec.get('recovery_pct',0):.1f}%)")
    except Exception:
        pass
    if r.returncode != 0 and artifact:
        log(f"  [recover] using checkpointed partial adapter after {terminated_reason or 'doctor nonzero exit'}")
    # eff bpw of the deployed artifact = base + LoRA(rank) overhead, 16-bit adapters
    names, specs = _src_specs()
    qd = [specs[k][1] for k in names if _isq(specs, k)
          and (not target_regex or re.search(target_regex, k[:-7] if k.endswith(".weight") else k))]
    den = sum(o * i for (o, i) in qd)
    lora_add = (16 * rank * sum(o + i for (o, i) in qd)) / den if den else 0.0
    if not artifact:
        try: os.remove(base)
        except OSError: pass
        raise RuntimeError(f"doctor produced no adapter checkpoint bits={bits}")
    return {"base": base, "adapter": artifact}, base_bpw + lora_add


def _bake_shadow(candidates, bits, tag):
    """STRAND-bake a doctor's healed-shadow weights (L4/L5 produce full-precision quant-robust
    weights that must be baked through the real codec). Picks the first shadow file that exists,
    bakes it whole, returns ([baked_override], effective_bpw). Whole-bake peaks at ~2x bf16 RAM —
    fine on the 128GB Studio (this is a Studio-tier stage)."""
    raw = next((c for c in candidates if os.path.exists(c)), None)
    if not raw:
        raise RuntimeError(f"{tag}: doctor produced no shadow ({candidates})")
    baked = f"{T}_{tag}_baked.safetensors"
    bpw, qc = _bake_one(raw, baked, bits)
    for c in candidates:
        try: os.remove(c)
        except OSError: pass
    return [baked], bpw


def _doctor_env():
    env = {**os.environ, "DOCTOR_MODEL": MODEL, "DOCTOR_DEVICE": DEV, "DOCTOR_DTYPE": "bfloat16",
           "STRAND_NO_GPU": "1"}
    cal = "scratch/calib_corpus.txt"
    if os.path.exists(cal):
        env["DOCTOR_CALIB"] = cal
    return env


def build_blockwise(bits, steps=80):
    """L4 — full-rank per-layer QAT (doctor_blockwise): the LoRA-plateau fix. Heals the full weight
    matrix (no rank ceiling, no bpw overhead) so it survives the STRAND cut, then bakes it."""
    out = f"{T}_bw.safetensors"
    log(f"  [L4 blockwise] {bits}b QAT {steps} steps")
    r = subprocess.run(["python3.12", "tools/condense/doctor.py", "blockwise", MODEL, out, str(bits), str(steps)],
                       capture_output=True, text=True, env=_doctor_env())
    if r.returncode != 0:
        raise RuntimeError(f"blockwise failed bits={bits}: {r.stderr[-200:]}")
    return _bake_shadow([out, out.replace(".safetensors", ".raw.safetensors")], bits, "bw")


def build_strand(bits, steps=200, lr=3e-5, req=50):
    """L5 — codec-native GPTQ-Hessian error-feedback (doctor_strand): the sub-residual ceiling
    breaker; quantizes sequentially through STRAND's trellis with Hessian error feedback (no STE)."""
    save = f"{T}_str.safetensors"
    log(f"  [L5 strand] {bits}b GPTQ-Hessian {steps} steps lr {lr} requant/{req}")
    r = subprocess.run(["python3.12", "tools/condense/doctor.py", "strand", str(bits), str(steps),
                        str(lr), str(req), save], capture_output=True, text=True, env=_doctor_env())
    if r.returncode != 0:
        raise RuntimeError(f"strand failed bits={bits}: {r.stderr[-200:]}")
    return _bake_shadow([save.replace(".safetensors", ".raw.safetensors"), save], bits, "str")


def _mp(a, f):
    """mixed-precision rung: a-bit attention, f-bit FFN."""
    return {k: a for k in ("q_proj", "k_proj", "v_proj", "o_proj")} | \
           {k: f for k in ("gate_proj", "up_proj", "down_proj")}


# ---- measurement: load model, stream override in-place, ppl, free ----
def _apply_weight_overrides(model, paths):
    if not paths:
        return
    sd = model.state_dict()
    for pth in paths:
        with safe_open(pth, framework="pt") as f:
            for k in f.keys():
                if k in sd and tuple(sd[k].shape) == tuple(f.get_slice(k).get_shape()):
                    sd[k].copy_(f.get_tensor(k).to(DEV, DTYPE))


def _attach_lora_adapter(model, adapter_path):
    modules = dict(model.named_modules())
    attached = 0
    with safe_open(adapter_path, framework="pt") as f:
        keys = set(f.keys())
        for ak in sorted(k for k in keys if k.endswith(".lora_A")):
            name = ak[:-7]
            bk = name + ".lora_B"
            m = modules.get(name)
            if bk not in keys or not isinstance(m, nn.Linear):
                continue
            m._hawking_lora_A = f.get_tensor(ak).to(DEV, DTYPE)
            m._hawking_lora_B = f.get_tensor(bk).to(DEV, DTYPE)
            m.forward = (lambda x, mm=m: F.linear(
                x, mm.weight + mm._hawking_lora_A @ mm._hawking_lora_B, mm.bias))
            attached += 1
    if attached == 0:
        raise RuntimeError(f"adapter contained no attachable LoRA tensors: {adapter_path}")
    log(f"  [measure] attached {attached} LoRA adapters")


def measure(override):
    _ensure_ppl_text()
    torch.set_num_threads(os.cpu_count() or 12)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, attn_implementation="eager").to(DEV).eval()
    if override:
        adapter = None
        if isinstance(override, dict):
            paths = [override["base"]] if override.get("base") else []
            adapter = override.get("adapter")
        else:
            paths = [override] if isinstance(override, str) else list(override)
        _apply_weight_overrides(model, paths)
        if adapter:
            _attach_lora_adapter(model, adapter)
    text = open(PT, errors="ignore").read()
    all_ids = tok(text, return_tensors="pt").input_ids[0].to(DEV)
    # MULTIWINDOW>1 (env, default 1) evaluates N non-overlapping 2048-token windows and returns
    # the MEAN ppl — a single noisy slice is overfittable (doctor trains on a calib corpus), so the
    # verifier scores on multiple held-out windows to confirm a recovery is real, not memorized.
    # The running ladder leaves this at 1 (consistent fast search); only the verifier sets it.
    nwin = int(os.environ.get("MULTIWINDOW", "1"))
    W = 2048
    ppls = []
    with torch.no_grad():
        for w in range(max(1, nwin)):
            seg = all_ids[w * W:(w + 1) * W]
            if seg.numel() < 16:
                break
            ids = seg.unsqueeze(0)
            ppls.append(math.exp(model(ids, labels=ids).loss.item()))
    del model; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    if len(ppls) > 1:
        log(f"  [measure] {len(ppls)} windows: min={min(ppls):.2f} mean={sum(ppls)/len(ppls):.2f} max={max(ppls):.2f}")
    return sum(ppls) / len(ppls)


def capture_sigma():
    if os.path.exists(SIGPATH):
        log(f"# sigma already exists at {SIGPATH}, skipping capture")
        return
    _ensure_ppl_text()
    torch.set_num_threads(os.cpu_count() or 12)
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
        # --- sub-3-bit rescue suite (the most important: can 1/2-bit be made usable on 7B?) ---
        # train-free: keep top-|w| 5/10% at 8-bit (outlier channel)
        ("2-AWQ-o5", build_awq, (2, 0.5, None, 5.0)), ("2-AWQ-o10", build_awq, (2, 0.5, None, 10.0)),
        ("1-AWQ-o5", build_awq, (1, 0.5, None, 5.0)), ("1-AWQ-o10", build_awq, (1, 0.5, None, 10.0)),
        # recovery: AWQ base + LoRA-KD heal (doctor) — the real sub-3-bit lever
        ("3-AWQ+dr", build_recover, (3,)), ("2-AWQ+dr", build_recover, (2,)),
        ("2-AWQ-o5+dr", build_recover, (2, 60, 64, 1e-4, 0.5, None, 5.0)),
        ("1-AWQ+dr", build_recover, (1,)), ("1-AWQ-o10+dr", build_recover, (1, 60, 64, 1e-4, 0.5, None, 10.0)),
        # seeded/known (skip on resume):
        ("4-AWQ", build_awq, (4,)), ("3-AWQ", build_awq, (3,)), ("2-AWQ", build_awq, (2,)),
        ("1-AWQ", build_awq, (1,)), ("1-RHT", build_rht, (1,))],
    # STUDIO — interruption-safe/default set for the M3 Ultra ladder. L4 blockwise is excluded until
    # it supports sharded sources, requested dtype, and per-layer resume; L5 STRAND is excluded until
    # its requested-model and durable optimizer-state gates pass. `studio_full` retains those research
    # configs behind HAWKING_STUDIO_RESEARCH_FULL=1 in studio_run.py.
    "studio": [
        ("4-AWQ", build_awq, (4,)), ("3-AWQ", build_awq, (3,)),
        ("2-AWQ", build_awq, (2,)), ("1-AWQ", build_awq, (1,)),
        ("mp-4a3f", build_awq, (3, 0.5, _mp(4, 3))), ("mp-3a2f", build_awq, (2, 0.5, _mp(3, 2))),
        ("res3+2", build_residual, (3, 2)), ("res2+1", build_residual, (2, 1)),
        ("3-AWQ+dr", build_recover, (3,)), ("2-AWQ+dr", build_recover, (2,)),
        ("1-AWQ+dr", build_recover, (1,)),
    ],
    # Full L0-L6 research set. Not production-ready on sharded 7B+ parents; explicit opt-in only.
    "studio_full": [
        ("4-AWQ", build_awq, (4,)), ("3-AWQ", build_awq, (3,)),
        ("2-AWQ", build_awq, (2,)), ("1-AWQ", build_awq, (1,)),
        ("mp-4a3f", build_awq, (3, 0.5, _mp(4, 3))), ("mp-3a2f", build_awq, (2, 0.5, _mp(3, 2))),
        ("res3+2", build_residual, (3, 2)), ("res2+1", build_residual, (2, 1)),
        ("3-AWQ+dr", build_recover, (3,)), ("2-AWQ+dr", build_recover, (2,)),
        ("1-AWQ+dr", build_recover, (1,)),
        ("3-bw", build_blockwise, (3,)), ("2-bw", build_blockwise, (2,)),
        ("1-bw", build_blockwise, (1,)),
        ("2-str", build_strand, (2,)), ("1-str", build_strand, (1,)),
    ],
    # SUBBIT — the sub-1-bit / sub-2-bit frontier lane (studio_maximization SUBBIT plan). Gated by
    # subbit_measure.py (SUBBIT-0 entropy floor) upstream. SUBBIT-1 = PTQ1.61: 1-bit bulk + sparse
    # 8-bit outlier channel (serves via the native base .tq + OUTL wire). res1+1 = SUBBIT-3 coarse
    # base + low-rate residual (native parity-green two-part .tq serve). 1-str/1-bw/+dr = the
    # codec-native + full-rank recovery at the 1-bit edge (the UNPROVEN gate-opener the Studio reopens).
    "subbit": [
        ("subbit1-o0.5", build_awq, (1, 0.5, None, 0.5)),
        ("subbit1-o1",   build_awq, (1, 0.5, None, 1.0)),
        ("subbit1-o2",   build_awq, (1, 0.5, None, 2.0)),
        ("res1+1",       build_residual, (1, 1)),
        ("res2+1",       build_residual, (2, 1)),
        ("subbit1-o1+dr", build_recover, (1, 60, 64, 1e-4, 0.5, None, 1.0)),
        ("2-AWQ+dr",     build_recover,  (2,)),
    ],
    "subbit_full": [
        ("subbit1-o0.5", build_awq, (1, 0.5, None, 0.5)),
        ("subbit1-o1",   build_awq, (1, 0.5, None, 1.0)),
        ("subbit1-o2",   build_awq, (1, 0.5, None, 2.0)),
        ("res1+1",       build_residual, (1, 1)),
        ("res2+1",       build_residual, (2, 1)),
        ("1-str",        build_strand,   (1,)),
        ("1-bw",         build_blockwise, (1,)),
        ("subbit1-o1+dr", build_recover, (1, 60, 64, 1e-4, 0.5, None, 1.0)),
        ("2-AWQ+dr",     build_recover,  (2,)),
    ],
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


def _load_verify_set():
    """Build CONFIGS['verify'] from a JSON spec so the verifier can independently REPRODUCE a
    chosen set of top candidates. Spec = [[name, fn_name, [args...]], ...]; fn resolved from
    module globals (build_awq/build_recover/build_rht/build_residual). Lets the verifier re-bake
    + re-doctor the winners from scratch under MULTIWINDOW eval — independent reproduction is the
    strongest 'this result is real' guarantee (catches eval-noise, overfit, and non-determinism)."""
    spec_path = os.environ.get("VERIFY_SPEC")
    if not spec_path or not os.path.exists(spec_path):
        raise RuntimeError(f"SETNAME=verify needs VERIFY_SPEC json (got {spec_path!r})")
    spec = json.load(open(spec_path))
    out = []
    for name, fn_name, args in spec:
        fn = globals().get(fn_name)
        if fn is None:
            raise RuntimeError(f"verify spec: unknown build fn {fn_name!r} for {name}")
        out.append((name, fn, tuple(args)))
    CONFIGS["verify"] = out
    log(f"# verify set: {len(out)} candidates — {[c[0] for c in out]}")


def main():
    global SRC
    _acquire_lock()
    SRC = _resolve_src()
    if SETNAME == "verify":
        _load_verify_set()
    log(f"# audit {LABEL} dev={DEV} dtype={DTYPE} set={SETNAME} cpu={os.cpu_count()} free={_free_gb():.0f}GB")

    # --- checkpoint resume: scan JSONL for already-completed configs ---
    records, completed, failed = {}, {}, {}
    if os.path.exists(f"{OUTP}.jsonl"):
        for line in open(f"{OUTP}.jsonl"):
            try:
                rec = json.loads(line.strip())
                cfg = rec.get("config")
                if cfg:
                    records[cfg] = rec
            except Exception:
                pass
    for cfg, rec in records.items():
        if "ppl" in rec:
            completed[cfg] = rec
        elif "error" in rec:
            failed[cfg] = rec
    if completed:
        log(f"# resume: {len(completed)} checkpointed — {sorted(completed)}")
    retry_errors = os.environ.get("LADDER_RETRY_ERRORS", "0").lower() in {"1", "true", "yes"}
    skip = dict(completed)
    if failed and not retry_errors:
        skip.update(failed)
        log(f"# resume: {len(failed)} error-checkpointed — {sorted(failed)}")

    # Consume a saved inject before deciding whether expensive sigma capture is still needed.
    _try_inject()
    pending = [(name, fn, args) for name, fn, args in CONFIGS[SETNAME] if name not in skip]
    if any(fn in (build_awq, build_recover) for _, fn, _ in pending):
        capture_sigma()
    else:
        log("# sigma not needed; no pending AWQ/recovery configs")

    # --- f16 baseline (resumable) ---
    if "f16" in completed:
        hf = completed["f16"]["ppl"]
        log(f"# f16 ppl = {hf:.3f} (checkpoint)")
    else:
        hf = measure(None)
        log(f"# f16 ppl = {hf:.3f}")
        _append_jsonl_durable(
            f"{OUTP}.jsonl",
            {"model": LABEL, "config": "f16", "eff_bpw": 16.0,
             "ppl": round(hf, 3), "degr_pct": 0.0},
        )

    rows = list(records.values())
    for name, fn, args in CONFIGS[SETNAME]:
        _try_inject()  # execute inject script if present before each config

        if name in skip:
            rec = skip[name]
            if "ppl" in rec:
                log(f"  {name:10s} -> SKIP (checkpoint +{rec.get('degr_pct','?')}%)")
            else:
                log(f"  {name:10s} -> SKIP (error checkpoint: {rec.get('error','')[:60]})")
            continue

        _clean_temps()   # sweep any leftover chunk parts/inputs from an interrupted run

        try:
            path, bpw = fn(*args)
            p = measure(path)
            rec = {"model": LABEL, "config": name, "eff_bpw": round(bpw, 3),
                   "ppl": round(p, 3), "degr_pct": round((p / hf - 1) * 100, 2)}
            # capability tripwire on floor candidates: a floor claim is void if ppl is ~1:1 but a
            # downstream task collapses (§6). Only on parts-based artifacts within the gate, to keep
            # multi_eval cost off the broken tiers. Best-effort: records, never blocks the run.
            tw_gate = float(os.environ.get("STUDIO_TRIPWIRE_GATE", "8.0"))
            if (os.environ.get("STUDIO_TRIPWIRE") == "1" and isinstance(path, list)
                    and rec["degr_pct"] <= tw_gate):
                merged = f"{T}_trip.safetensors"
                try:
                    _merge_parts(path, merged)
                    tw = subprocess.run(["python3.12", "tools/condense/multi_eval.py", MODEL, merged, name],
                                        capture_output=True, text=True, env=os.environ)
                    last = tw.stdout.strip().splitlines()[-1] if tw.stdout.strip() else ""
                    try: rec["tripwire"] = json.loads(last)
                    except Exception: rec["tripwire"] = last[:200] or f"rc={tw.returncode}"
                finally:
                    try: os.remove(merged)
                    except OSError: pass
        except Exception as e:
            rec = {"model": LABEL, "config": name, "error": str(e)[:140]}
        rows.append(rec)
        log(f"  {name:10s} -> {rec.get('eff_bpw','?')} bpw  +{rec.get('degr_pct','?')}%")
        _append_jsonl_durable(f"{OUTP}.jsonl", rec)
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
