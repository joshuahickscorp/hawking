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
import sys, os, re, gc, json, math, subprocess, torch, torch.nn as nn
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
SRC = os.path.join(MODEL, "model.safetensors")
T = f"/tmp/aud_{LABEL}"                     # reused temp prefix (overwritten per config)
SIGPATH = f"{T}_sigma.safetensors"


def log(m): print(m, file=sys.stderr); sys.stderr.flush()


def bake(inp, out, bits):
    """Run the Rust baker; return REAL aggregate effective bpw."""
    r = subprocess.run([BAKER, "--in", inp, "--out", out, "--bits", str(bits), "--quality",
                        "--rht-cols", "--outlier-channel", "1", "--outlier-bits", "8",
                        "--threads", "10"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"baker failed bits={bits}: {r.stderr[-200:]}")
    m = re.search(r"AGGREGATE effective bpw = ([0-9.]+)", r.stderr + r.stdout)
    return float(m.group(1)) if m else float("nan")


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


# ---- builders: each returns (override_path, effective_bpw). Model NOT loaded here. ----
def build_rht(bits):
    bpw = bake(SRC, f"{T}_ovr.safetensors", bits)
    return f"{T}_ovr.safetensors", bpw


def build_awq(bits, alpha=0.5):
    sig = {}
    with safe_open(SIGPATH, framework="pt") as f:
        for k in f.keys():
            sig[k] = f.get_tensor(k)
    # scale columns by sigma^alpha (streamed)
    out = {}
    with safe_open(SRC, framework="pt") as f:
        for k in f.keys():
            v = f.get_tensor(k)
            out[k] = (v.float() * sig[k].pow(alpha)).to(DTYPE) if k in sig else v
    save_file(out, f"{T}_scaled.safetensors"); del out; gc.collect()
    bpw = bake(f"{T}_scaled.safetensors", f"{T}_baked.safetensors", bits)
    out = {}
    with safe_open(f"{T}_baked.safetensors", framework="pt") as f:
        for k in f.keys():
            v = f.get_tensor(k)
            out[k] = (v.float() / sig[k].pow(alpha)).to(DTYPE) if k in sig else v
    save_file(out, f"{T}_ovr.safetensors"); del out; gc.collect()
    return f"{T}_ovr.safetensors", bpw


def build_residual(b1, b2):
    bpw1 = bake(SRC, f"{T}_b1.safetensors", b1)
    qk = quant_keys(f"{T}_b1.safetensors")
    rin = {}
    with safe_open(SRC, framework="pt") as fs, safe_open(f"{T}_b1.safetensors", framework="pt") as fb:
        for k in fs.keys():
            v = fs.get_tensor(k)
            rin[k] = (v.float() - fb.get_tensor(k).float()).to(DTYPE) if k in qk else v
    save_file(rin, f"{T}_rin.safetensors"); del rin; gc.collect()
    bpw2 = bake(f"{T}_rin.safetensors", f"{T}_b2.safetensors", b2)
    out = {}
    with safe_open(f"{T}_b1.safetensors", framework="pt") as f1, safe_open(f"{T}_b2.safetensors", framework="pt") as f2:
        for k in f1.keys():
            out[k] = (f1.get_tensor(k).float() + f2.get_tensor(k).float()).to(DTYPE) if k in qk else f1.get_tensor(k)
    save_file(out, f"{T}_ovr.safetensors"); del out; gc.collect()
    return f"{T}_ovr.safetensors", bpw1 + bpw2


# ---- measurement: load model, stream override in-place, ppl, free ----
def measure(override):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE, attn_implementation="eager").to(DEV).eval()
    if override:
        sd = model.state_dict()
        with safe_open(override, framework="pt") as f:
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
}


def main():
    log(f"# audit {LABEL} dev={DEV} dtype={DTYPE} set={SETNAME}")
    capture_sigma()
    hf = measure(None)
    log(f"# f16 ppl = {hf:.3f}")
    open(f"{OUTP}.jsonl", "w").write(json.dumps({"model": LABEL, "config": "f16", "eff_bpw": 16.0, "ppl": round(hf, 3), "degr_pct": 0.0}) + "\n")
    rows = []
    for name, fn, args in CONFIGS[SETNAME]:
        try:
            path, bpw = fn(*args)
            p = measure(path)
            rec = {"model": LABEL, "config": name, "eff_bpw": round(bpw, 3), "ppl": round(p, 3), "degr_pct": round((p / hf - 1) * 100, 2)}
        except Exception as e:
            rec = {"model": LABEL, "config": name, "error": str(e)[:140]}
        rows.append(rec)
        log(f"  {name:10s} -> {rec.get('eff_bpw','?')} bpw  +{rec.get('degr_pct','?')}%")
        open(f"{OUTP}.jsonl", "a").write(json.dumps(rec) + "\n")
        for suf in ("_b1", "_b2", "_rin", "_scaled", "_baked"):
            try: os.remove(f"{T}{suf}.safetensors")
            except OSError: pass
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
