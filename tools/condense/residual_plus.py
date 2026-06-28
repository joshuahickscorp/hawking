#!/usr/bin/env python3.12
"""RESIDUAL-PLUS — generalized residual STRAND condense (3 levers over residual_bake.py).

residual_bake.py proves the full-rank ceiling-breaker:  W ≈ STRAND_b1(W) + STRAND_b2(R),
R = W − STRAND_b1(W). The residual term is FULL-RANK (captures the high-rank quant error
LoRA can't) and codec-native (no transfer gap). Costs +b2 bpw. This tool generalizes that
single idea three ways, reusing residual_bake's exact stage logic (copied so the running
ladder audit's residual_bake.py is untouched):

  (a) per-tensor residual DEPTH — residual_bake gives EVERY quantized tensor the same b2.
      But the residual error only matters where the network is output-sensitive (down_proj
      on the hot path) — a tolerant k_proj wastes the +b2 bpw. This spends residual bits
      per tensor from a sensitivity / mixed-precision config: high-sensitivity tensors get
      a (deeper) residual pass, tolerant ones get base only. Lower AVERAGE eff-bpw at equal
      quality — the same rate–distortion win mixed_precision.py makes, applied to residual.

  (b) ITERATED residual — b1 + b2 + b3 (+ …). Each stage quantizes the RUNNING residual
      with STRAND:  R0=W; for stage i: Ŵ += STRAND_bi(R_{i-1}); R_i = W − Ŵ. More passes
      drive the residual down further (diminishing returns) — the knob for the quality
      (Stream-A) ceiling when bpw budget exists.

  (c) AWQ × residual STACK — compute the residual on an AWQ base. The base pass is the
      activation-aware awq_bake (scale columns by σ^alpha → bake → unscale-fold); residual
      passes then correct W − Ŵ_awq with plain STRAND. AWQ shrinks the base error train-
      free, so the residual has less to fix → a better quality/bpw point than residual-on-
      raw. (The ladder calls this the 7B chat's active next step.)

Effective bpw is reported HONESTLY as the SUM of all passes' aggregate bpw (a residual is a
SECOND stored stream — serving sums them in GEMV; that two-part .tq serve path is not yet
built, so today residual is the QUALITY ceiling-breaker, single-bake is the SERVE path).
Degradation is the real ppl forward pass vs f16 — never hidden.

HONORS DOCTOR_DEVICE / DOCTOR_DTYPE for the ppl pass + σ capture (0.5B → mps/float32; 7B →
cpu/bfloat16; NEVER float16 — MPS f16 GQA bug + 7B fp16 overflow→nan).

Usage:
  # ITERATED residual 3+2+2 (base 3-bit, two 2-bit residual passes):
  python3.12 tools/condense/residual_plus.py scratch/qwen-05b out/res_322.safetensors \
        --stages 3,2,2

  # PER-TENSOR depth from a mixed_precision config (tensors it allocated >2 bits get a
  # 2-bit residual pass; the rest stay base-only):
  python3.12 tools/condense/residual_plus.py scratch/qwen-05b out/res_pt.safetensors \
        --base-bits 3 --from-mp scratch/qwen-05b-mp3.0.json --residual-above 2 --residual-bits 2

  # AWQ × residual STACK (AWQ 3-bit base, alpha 0.5, + 2-bit residual):
  python3.12 tools/condense/residual_plus.py scratch/qwen-05b out/awq_res_32.safetensors \
        --stages 3,2 --awq-base --alpha 0.5

  # LIGHT self-test (few tensors, fast non-quality bake — does not contend with the audit):
  python3.12 tools/condense/residual_plus.py scratch/qwen-05b /tmp/resp_selftest.safetensors \
        --stages 3,2 --limit-tensors 4 --fast
"""
import sys, os, re, gc, json, math, time, argparse, subprocess
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BAKER = os.path.join(ROOT, "vendor", "strand-quant", "target", "release", "quantize-model")
sys.path.insert(0, HERE)
import ladder as L                                       # canonical BPW table
DEV = os.environ.get("DOCTOR_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))
PT = os.environ.get("PPL_TEXT", "/tmp/ppl24k.txt")
CALIB = os.environ.get("DOCTOR_CALIB", os.path.join(ROOT, "scratch", "calib_corpus.txt"))
CALIB_MD = os.path.join(ROOT, "scratch", "calib_multidomain.txt")


def log(*m):
    print(*m, file=sys.stderr); sys.stderr.flush()


def audit_running():
    try:
        r = subprocess.run(["pgrep", "-fl", "quantize-model"], capture_output=True, text=True)
        return [ln for ln in r.stdout.splitlines() if "quantize-model" in ln]
    except Exception:
        return []


def model_src(model_dir):
    one = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(one):
        return load_file(one)
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        sd = {}
        for shard in sorted(set(wm.values())):
            sd.update(load_file(os.path.join(model_dir, shard)))
        log(f"# merged {len(set(wm.values()))} shards in-memory")
        return sd
    raise FileNotFoundError(f"no model.safetensors[.index.json] in {model_dir}")


# ── baker (COPIED invocation from residual_bake.py; Metal auto on macOS) ────────────────
def bake(inp, out, bits, quality=True, threads=10):
    cmd = [BAKER, "--in", inp, "--out", out, "--bits", str(bits), "--rht-cols",
           "--outlier-channel", "1", "--outlier-bits", "8", "--threads", str(threads)]
    if quality:
        cmd += ["--quality"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    blob = r.stderr + r.stdout
    if r.returncode != 0:
        raise RuntimeError(f"baker failed: {blob.strip().splitlines()[-4:]}")
    agg = re.search(r"AGGREGATE effective bpw = ([\d.]+)", blob)
    return float(agg.group(1)) if agg else float("nan")


# ── AWQ helpers (COPIED from awq_bake.py — used only for --awq-base) ────────────────────
def calib_text():
    txt = open(CALIB, errors="ignore").read() if os.path.exists(CALIB) else open(PT, errors="ignore").read()
    if os.path.exists(CALIB_MD):
        txt = txt + "\n" + open(CALIB_MD, errors="ignore").read()
    return txt[:20000]


def capture_sigma(model_dir):
    m = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=DTYPE, attn_implementation="eager").to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    sig, hooks = {}, []

    def mk(name):
        def h(mod, inp, out):
            x = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).float().mean(0)
            sig[name] = sig.get(name, torch.zeros_like(x)) + x
        return h

    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.shape[1] >= 256:
            hooks.append(mod.register_forward_hook(mk(name)))
    ids = tok(calib_text(), return_tensors="pt").input_ids[:, :2048].to(DEV)
    with torch.no_grad():
        m(ids)
    for h in hooks:
        h.remove()
    out = {k + ".weight": v.cpu().float() for k, v in sig.items()}
    log(f"# captured activation σ for {len(out)} linears (AWQ base)")
    del m
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
    return out


def awq_bake_base(W, sigma, alpha, bits, qkeys, tag, quality, threads):
    """awq_bake's scale→bake→unscale on the BASE only. Returns (Ŵ_dict, eff_bpw).
    Ŵ for qkeys = unscaled decoded; non-qkeys copied verbatim."""
    scaled, scales = {}, {}
    for k, v in W.items():
        if k in qkeys and k in sigma:
            s = (sigma[k] + 1e-6) ** alpha
            scaled[k] = (v.float() * s).to(torch.float16)
            scales[k] = s
        else:
            scaled[k] = v
    ti, to = f"/tmp/resp_awq_in_{tag}.safetensors", f"/tmp/resp_awq_out_{tag}.safetensors"
    save_file(scaled, ti)
    ebpw = bake(ti, to, bits, quality=quality, threads=threads)
    baked = load_file(to)
    Wh = {}
    for k, v in baked.items():
        Wh[k] = (v.float() / scales[k]) if k in scales else v.float()
    os.remove(ti); os.remove(to)
    return Wh, ebpw


# ── plain STRAND bake of a state dict → decoded floats + eff_bpw ───────────────────────
def strand_bake(sd, bits, tag, quality, threads):
    ti, to = f"/tmp/resp_in_{tag}.safetensors", f"/tmp/resp_out_{tag}.safetensors"
    save_file({k: (v.to(torch.float16) if v.dtype != torch.float16 else v) for k, v in sd.items()}, ti)
    ebpw = bake(ti, to, bits, quality=quality, threads=threads)
    dec = load_file(to)
    os.remove(ti); os.remove(to)
    return {k: v.float() for k, v in dec.items()}, ebpw


def quantized_keys(W, Wh1):
    """Tensors the baker actually quantized (2-D, shape-match, changed) — residual_bake's qkey test."""
    return {k for k, v in W.items()
            if k in Wh1 and v.dim() == 2 and Wh1[k].shape == v.shape and not torch.equal(Wh1[k].float(), v.float())}


# ── ppl (real forward pass; honors DEV/DTYPE) ──────────────────────────────────────────
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


def load_depth_map(args, qkeys):
    """Per-tensor residual depth (b2 list) per quantized tensor name. Sources:
      --depth-config JSON {substr: [b,...] | b}  (explicit residual bits per pattern)
      --from-mp <mp.json> --residual-above N     (tensors mp allocated > N bits → residual)
    Tensors with no rule get the global --stages residual bits (or none if base-only)."""
    depth = {}
    if args.depth_config:
        rules = json.load(open(args.depth_config))
        for k in qkeys:
            for patt, b in rules.items():
                if patt in k:
                    depth[k] = b if isinstance(b, list) else [b]
                    break
    if args.from_mp:
        mp = json.load(open(args.from_mp))            # [{pattern, bits}]
        bits_by = {e["pattern"]: e["bits"] for e in mp}
        for k in qkeys:
            # exact name match first, else substring
            kb = bits_by.get(k)
            if kb is None:
                for patt, b in bits_by.items():
                    if patt in k:
                        kb = b; break
            if kb is not None and kb > args.residual_above:
                depth.setdefault(k, [args.residual_bits])
    return depth


def main():
    ap = argparse.ArgumentParser(description="residual_plus: depth + iterated + AWQ-stack residual")
    ap.add_argument("model_dir")
    ap.add_argument("out")
    # iterated residual: base + residual passes
    ap.add_argument("--stages", default="3,2",
                    help="bits per pass: base,res1,res2,… (residual_bake = '3,2'). Iterated = '3,2,2'.")
    ap.add_argument("--base-bits", type=int, default=None,
                    help="base bits when using --from-mp/--depth-config (overrides --stages[0])")
    # per-tensor depth
    ap.add_argument("--depth-config", default=None,
                    help="JSON {substr: residual_bits | [bits...]} — per-tensor residual depth")
    ap.add_argument("--from-mp", default=None,
                    help="derive depth from a mixed_precision --mp-config JSON ([{pattern,bits}])")
    ap.add_argument("--residual-above", type=int, default=2,
                    help="with --from-mp: tensors allocated > this many bits get a residual pass")
    ap.add_argument("--residual-bits", type=int, default=2, help="residual bits granted by --from-mp")
    # AWQ stack
    ap.add_argument("--awq-base", action="store_true", help="compute the residual on an AWQ base")
    ap.add_argument("--alpha", type=float, default=0.5, help="AWQ alpha for --awq-base")
    # general
    ap.add_argument("--limit-tensors", type=int, default=None, help="restrict to N tensors (LIGHT self-test)")
    ap.add_argument("--fast", action="store_true", help="non-quality bake (L=k+4) — self-test speed")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--no-prove", action="store_true", help="skip the ppl forward pass")
    args = ap.parse_args()

    quality = not args.fast
    stages = [int(x) for x in args.stages.split(",")]
    base_bits = args.base_bits if args.base_bits is not None else stages[0]
    res_bits_global = stages[1:]                       # iterated residual passes (global)
    name = os.path.basename(args.model_dir.rstrip("/"))
    per_tensor_depth = bool(args.depth_config or args.from_mp)

    log(f"# residual_plus · {name} · base={base_bits}b · "
        f"{'AWQ-base(a=%g) ' % args.alpha if args.awq_base else ''}"
        f"{'per-tensor-depth ' if per_tensor_depth else 'iterated res=%s ' % res_bits_global}"
        f"· dev={DEV}/{DTYPE} · {'fast' if args.fast else 'quality'}")
    run = audit_running()
    if run:
        log(f"# NOTE: {len(run)} baker proc running (ladder audit). Probes use Metal (GPU); audit "
            f"baker is CPU — keep --threads modest. {'self-test is light.' if args.limit_tensors else ''}")

    W = model_src(args.model_dir)
    if args.limit_tensors:
        roles = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        keep, seen = set(), set()
        for role in roles:
            for k in W:
                if role in k and k.endswith(".weight") and role not in seen and W[k].dim() == 2:
                    keep.add(k); seen.add(role); break
        # keep everything non-2D (norms/embeddings) so the model still runs; cap 2-D to N
        twoD = [k for k in keep]
        W = {k: v for k, v in W.items() if v.dim() != 2 or k in twoD}
        log(f"# self-test slice: {len(twoD)} quantized tensors {sorted(s for s in seen)}")

    # ── STAGE 0: base (AWQ or plain) ────────────────────────────────────────────────────
    sigma = capture_sigma(args.model_dir) if args.awq_base else None
    qkeys_guess = {k for k, v in W.items() if v.dim() == 2 and v.shape[1] >= 256}
    if args.awq_base:
        Wh, base_bpw = awq_bake_base(W, sigma, args.alpha, base_bits, qkeys_guess, "base", quality, args.threads)
    else:
        Wh, base_bpw = strand_bake(W, base_bits, "base", quality, args.threads)
    qkeys = quantized_keys(W, {k: torch.as_tensor(v) for k, v in Wh.items()})
    log(f"# base {base_bits}-bit{' (AWQ)' if args.awq_base else ''}: {len(qkeys)} quantized tensors, "
        f"eff {base_bpw:.3f} bpw")

    # accumulate decoded estimate Ŵ; running residual R = W − Ŵ on qkeys
    Wh_acc = {k: (Wh[k] if k in Wh else W[k].float()) for k in W}
    pass_bpw = [("base", base_bits, round(base_bpw, 3))]

    # ── residual passes ─────────────────────────────────────────────────────────────────
    if per_tensor_depth:
        depth = load_depth_map(args, qkeys)
        n_res = sum(len(v) for v in depth.values())
        log(f"# per-tensor depth: {len(depth)}/{len(qkeys)} tensors get a residual "
            f"({n_res} residual-tensor-passes). Others stay base-only.")
        # bucket tensors by the residual bits of their NEXT pass, bake each bucket together
        max_passes = max((len(v) for v in depth.values()), default=0)
        for p in range(max_passes):
            buckets = {}
            for k, bs in depth.items():
                if p < len(bs):
                    buckets.setdefault(bs[p], []).append(k)
            for rb, keys in sorted(buckets.items()):
                Rin = {k: (W[k].float() - Wh_acc[k]) for k in keys}
                dec, rb_bpw = strand_bake(Rin, rb, f"ptres{p}_{rb}", quality, args.threads)
                for k in keys:
                    Wh_acc[k] = Wh_acc[k] + dec[k]
                # weight this pass's bpw by the FRACTION of params it covers (honest avg)
                covered = sum(W[k].numel() for k in keys)
                total = sum(W[k].numel() for k in qkeys)
                pass_bpw.append((f"res{p+1}@{rb}b×{len(keys)}t", rb, round(rb_bpw * covered / total, 3)))
                log(f"  residual pass {p+1} @ {rb}-bit over {len(keys)} tensors: "
                    f"raw {rb_bpw:.3f} bpw · param-weighted +{rb_bpw*covered/total:.3f} bpw")
    else:
        for i, rb in enumerate(res_bits_global):
            Rin = {k: (W[k].float() - Wh_acc[k]) for k in qkeys}
            dec, rb_bpw = strand_bake(Rin, rb, f"res{i}_{rb}", quality, args.threads)
            for k in qkeys:
                Wh_acc[k] = Wh_acc[k] + dec[k]
            pass_bpw.append((f"res{i+1}", rb, round(rb_bpw, 3)))
            resnorm = math.sqrt(sum(float((W[k].float()-Wh_acc[k]).pow(2).sum()) for k in qkeys))
            log(f"  iterated residual pass {i+1} @ {rb}-bit: eff {rb_bpw:.3f} bpw · "
                f"‖residual‖₂ now {resnorm:.4g}")

    # ── write summed model (base + residuals on qkeys; originals elsewhere) ──────────────
    out = {}
    for k, v in W.items():
        out[k] = Wh_acc[k].to(torch.float16) if k in qkeys else v
    save_file(out, args.out)
    eff_total = round(sum(b for _, _, b in pass_bpw), 3)
    log(f"# saved → {args.out}  ·  effective bpw = Σpasses = {eff_total}  "
        f"({' + '.join(f'{n}:{b}' for n, _, b in pass_bpw)})")

    result = {"model": name, "base_bits": base_bits, "awq_base": args.awq_base,
              "mode": ("per-tensor-depth" if per_tensor_depth else "iterated"),
              "passes": [{"pass": n, "bits": bt, "bpw_contrib": b} for n, bt, b in pass_bpw],
              "eff_bpw_total": eff_total, "n_quantized": len(qkeys), "out": args.out}

    if not args.no_prove:
        f16 = ppl(args.model_dir, None)
        p = ppl(args.model_dir, args.out)
        degr = (p / f16 - 1) * 100
        # honest quality tier vs the ladder's thresholds
        tier = ("≈1:1 (near-lossless)" if degr <= L.NEAR_1to1 * 100
                else "beats-Q4_K band" if degr <= L.WIN * 100 else "above Q4_K degradation")
        log(f"  f16 ppl = {f16:.3f}  ·  residual_plus ppl = {p:.3f}  (+{degr:.2f}%)  [{tier}]")
        log(f"  INTENDED vs VALIDATED: eff {eff_total} bpw at +{degr:.2f}% degradation "
            f"(compare to ladder: 0.5B res3+2≈+1.6%, res2+2≈+8.9%).")
        result.update(f16_ppl=round(f16, 3), ppl=round(p, 3), degr_pct=round(degr, 2), quality_tier=tier)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
