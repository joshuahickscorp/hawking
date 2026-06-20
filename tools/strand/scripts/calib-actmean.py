#!/usr/local/bin/python3
"""calib-actmean.py - projection-input activation means for STRAND de-bias.

This is a STRAND-specific calibration pass, not a generic framework port. It
hooks the exact projection modules that quantize-model emits as
`model.layers.*.{q,k,v,o,gate,up,down}_proj.weight`, forwards WikiText windows,
and records both:

  * scalar mean per module, for the cheap mu_bar rowsum correction
  * per-feature mean vector, for the stronger correction
        c_i = - sum_j (recon_ij - base_ij) * E[x_j]

The vector correction is still a per-output-row additive bias at inference time;
it costs the same as the scalar correction if materialized, but uses the measured
activation direction instead of assuming isotropic DC.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import torch


PROJ_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))$"
)
DATASET_IDS = ("wikitext", "Salesforce/wikitext")
DATASET_CONFIG = "wikitext-2-raw-v1"


def parse_layers(s, n_layers):
    if s.strip().lower() in ("", "all", "*"):
        return set(range(n_layers))
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return {i for i in out if 0 <= i < n_layers}


def load_wikitext(tok, split):
    from datasets import load_dataset

    errs = []
    for ds_id in DATASET_IDS:
        try:
            ds = load_dataset(ds_id, DATASET_CONFIG, split=split)
            text = "\n\n".join(ds["text"])
            fp = hashlib.sha256(text.encode()).hexdigest()[:16]
            ids = tok(text, return_tensors="pt").input_ids[0]
            return ids, ds_id, fp
        except Exception as e:  # noqa: BLE001 - fallback chain is the point
            errs.append(f"{ds_id}: {type(e).__name__}: {e}")
    raise SystemExit("[actmean] WikiText load failed:\n  " + "\n  ".join(errs))


def dtype_from_name(name):
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="scratch/qwen-05b")
    ap.add_argument("--out", default="research/actmean-qwen05b.json")
    ap.add_argument("--split", default="train", choices=["train", "test", "validation"])
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--chunks", type=int, default=8,
                    help="non-overlapping windows to forward; 0 means all")
    ap.add_argument("--layers", default="all",
                    help="all, comma list, or ranges like 0,3,10-12")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--no-feature-means", action="store_true",
                    help="store only scalar mean/rms; disables vector de-bias")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = dtype_from_name(args.dtype)
    dev = torch.device(args.device)
    print(f"[actmean] loading {args.model} ({args.dtype}, {args.device}, eager)", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch_dtype, attn_implementation="eager")
    model.to(dev)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    layers = parse_layers(args.layers, n_layers)
    print(f"[actmean] selected {len(layers)}/{n_layers} layers", flush=True)

    acc = {}
    hooks = []

    def ensure(name, dim):
        if name not in acc:
            acc[name] = {
                "sum": torch.zeros(dim, dtype=torch.float64),
                "sum_sq": torch.zeros(dim, dtype=torch.float64),
                "sum_abs": torch.zeros(dim, dtype=torch.float64),
                "tokens": 0,
                "dim": dim,
            }
        return acc[name]

    def make_hook(name):
        def pre_hook(_module, inputs):
            x = inputs[0].detach().to(torch.float32)
            # Expected [batch, seq, dim]; fall back to flatten-all-but-last for safety.
            dim = x.shape[-1]
            flat = x.reshape(-1, dim).to("cpu", torch.float64)
            a = ensure(name, dim)
            a["sum"] += flat.sum(dim=0)
            a["sum_sq"] += (flat * flat).sum(dim=0)
            a["sum_abs"] += flat.abs().sum(dim=0)
            a["tokens"] += flat.shape[0]
        return pre_hook

    for name, module in model.named_modules():
        m = PROJ_RE.match(name)
        if not m:
            continue
        if int(m.group(1)) not in layers:
            continue
        hooks.append(module.register_forward_pre_hook(make_hook(name)))

    if not hooks:
        raise SystemExit("[actmean] no projection modules were hooked")
    print(f"[actmean] hooked {len(hooks)} projection modules", flush=True)

    ids, dataset_id, dataset_fp = load_wikitext(tok, args.split)
    n_chunks = ids.numel() // args.ctx
    if args.chunks > 0:
        n_chunks = min(n_chunks, args.chunks)
    if n_chunks <= 0:
        raise SystemExit(f"[actmean] ctx={args.ctx} too large for {ids.numel()} tokens")

    t0 = time.time()
    with torch.no_grad():
        for i in range(n_chunks):
            window = ids[i * args.ctx:(i + 1) * args.ctx].unsqueeze(0).to(dev)
            model(window, use_cache=False)
            if args.device == "mps":
                torch.mps.empty_cache()
            print(f"[actmean] chunk {i + 1}/{n_chunks}", flush=True)

    for h in hooks:
        h.remove()

    modules = {}
    for name in sorted(acc):
        a = acc[name]
        n = max(int(a["tokens"]), 1)
        mean_vec = a["sum"] / n
        sq_mean = a["sum_sq"] / n
        abs_mean_vec = a["sum_abs"] / n
        rec = {
            "tensor": name + ".weight",
            "tokens": int(a["tokens"]),
            "dim": int(a["dim"]),
            "mean": float(mean_vec.mean()),
            "abs_mean": float(abs_mean_vec.mean()),
            "rms": float(sq_mean.mean().sqrt()),
            "feature_mean_l2": float(mean_vec.pow(2).sum().sqrt()),
            "feature_mean_absmax": float(mean_vec.abs().max()),
        }
        if not args.no_feature_means:
            rec["feature_mean"] = [round(float(v), 9) for v in mean_vec]
            # per-feature RMS (sqrt of second moment) — the activation-energy weighting
            # for output-space error analysis (error-spectrum.py activation-weighted mode)
            rec["feature_rms"] = [round(float(v), 9) for v in sq_mean.sqrt()]
        modules[name] = rec
        print(f"[actmean] {name}: mean={rec['mean']:.6g} "
              f"rms={rec['rms']:.6g} mu_l2={rec['feature_mean_l2']:.4g}",
              flush=True)

    out = {
        "schema": "strand_actmean_v1",
        "model": os.path.abspath(args.model),
        "split": args.split,
        "dataset_id": dataset_id,
        "dataset_fp": dataset_fp,
        "ctx": args.ctx,
        "chunks": n_chunks,
        "tokens_per_module": n_chunks * args.ctx,
        "layers": sorted(layers),
        "feature_means": not args.no_feature_means,
        "elapsed_s": round(time.time() - t0, 3),
        "modules": modules,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[actmean] wrote {len(modules)} modules -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
