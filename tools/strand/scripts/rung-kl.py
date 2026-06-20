#!/usr/local/bin/python3
"""rung-kl.py - output-space damage screen for STRAND rung allocation.

The allocator must see output damage, not weight MSE: RHT makes rel-RMS flat
across tensor classes, so rel-RMS routing is banned (frontier doc §2). This
screen answers, per tensor: "how much does swapping THIS tensor to its STRAND
recon hurt the model's output distribution?"

Method (hot-swap, one tensor at a time):
  1. load the bf16 base once (CPU), cache base logits on WikiText TRAIN windows
  2. for each projection tensor present in the recon: swap recon weight in,
     forward the same windows, score, swap the base weight back
  3. score = mean token KL(base || swapped) and delta-NLL vs the base
  4. emit ranked json + a RED list (the tensors PV must train) + rung rules

KL uses the base as reference: damage is how far the quantized output drifts
from what the full model would have said. delta-NLL grounds it in real loss.

Output json (research/rung-kl-<tag>.json) is promote.py-compatible: it carries
harness identity (dataset_fp, ctx, chunks) and is meant to feed selective PV
(--pv-tensors) and future mixed-rung configs with measured, output-space evidence.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def recon_files(path):
    if os.path.isfile(path):
        return [path]
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return [single]
    files = sorted(glob.glob(os.path.join(path, "model-*-of-*.safetensors")))
    if not files:
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise SystemExit(f"[rung-kl] no safetensors found in {path}")
    return files


def load_recon_tensors(path):
    from safetensors import safe_open
    out = {}
    for fpath in recon_files(path):
        with safe_open(fpath, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if key.endswith(".weight") and any(key.endswith(s + ".weight") for s in PROJ_SUFFIXES):
                    out[key] = sf.get_tensor(key)
    if not out:
        raise SystemExit(f"[rung-kl] no projection weights in {path}")
    return out


def load_train_windows(tok, ctx, n_windows):
    from datasets import load_dataset
    errs = []
    for ds_id in ("wikitext", "Salesforce/wikitext"):
        try:
            ds = load_dataset(ds_id, "wikitext-2-raw-v1", split="train")
            text = "\n\n".join(ds["text"])
            fp = hashlib.sha256(text.encode()).hexdigest()[:16]
            ids = tok(text, return_tensors="pt").input_ids[0]
            n = min(ids.numel() // ctx, n_windows)
            if n <= 0:
                raise SystemExit(f"[rung-kl] ctx={ctx} too large")
            return [ids[i * ctx:(i + 1) * ctx] for i in range(n)], ds_id, fp
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            errs.append(f"{ds_id}: {type(e).__name__}: {e}")
    raise SystemExit("[rung-kl] WikiText train load failed:\n  " + "\n  ".join(errs))


@torch.no_grad()
def forward_logits(model, window):
    out = model(window.unsqueeze(0), use_cache=False)
    return out.logits[0].float()


@torch.no_grad()
def score_vs_base(model, windows, base_logits, base_nll):
    """Mean token KL(base||current) and delta-NLL across windows."""
    kl_sum, tok_sum, nll_sum = 0.0, 0, 0.0
    for w, bl in zip(windows, base_logits):
        logits = forward_logits(model, w)
        logp = F.log_softmax(logits[:-1], dim=-1)
        base_logp = F.log_softmax(bl[:-1], dim=-1)
        kl = F.kl_div(logp, base_logp, log_target=True, reduction="none").sum(-1)
        kl_sum += float(kl.sum())
        nll = F.cross_entropy(logits[:-1], w[1:], reduction="sum")
        nll_sum += float(nll)
        tok_sum += w.numel() - 1
    return kl_sum / tok_sum, (nll_sum / tok_sum) - base_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="scratch/qwen-05b")
    ap.add_argument("--recon", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--tag", default="rungkl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--red-quantile", type=float, default=0.85,
                    help="tensors above this damage quantile form the RED PV list")
    ap.add_argument("--threads", type=int, default=6,
                    help="torch CPU threads; keep below box width so live MPS runs are not starved")
    args = ap.parse_args()
    out_path = args.out or f"research/rung-kl-{args.tag}.json"

    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[rung-kl] loading base {args.base} (cpu, bf16->fp32 logits)", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()
    model.config.use_cache = False

    recon = load_recon_tensors(args.recon)
    windows, dataset_id, dataset_fp = load_train_windows(tok, args.ctx, args.windows)
    print(f"[rung-kl] {len(recon)} tensors x {len(windows)} train windows @ctx{args.ctx}", flush=True)

    t0 = time.time()
    base_logits = [forward_logits(model, w) for w in windows]
    base_nll = sum(float(F.cross_entropy(bl[:-1], w[1:], reduction="sum"))
                   for w, bl in zip(windows, base_logits))
    base_nll /= sum(w.numel() - 1 for w in windows)
    print(f"[rung-kl] base cached: nll/tok={base_nll:.4f} ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for i, (name, r) in enumerate(sorted(recon.items()), 1):
        mod_name = name[: -len(".weight")]
        module = model.get_submodule(mod_name)
        if tuple(module.weight.shape) != tuple(r.shape):
            raise SystemExit(f"[rung-kl] shape mismatch {name}")
        saved = module.weight.detach().clone()
        with torch.no_grad():
            module.weight.copy_(r.to(module.weight.dtype))
        kl, dnll = score_vs_base(model, windows, base_logits, base_nll)
        with torch.no_grad():
            module.weight.copy_(saved)
        results[name] = {"kl_per_tok": round(kl, 6), "delta_nll": round(dnll, 6)}
        if i % 12 == 0 or i == len(recon):
            print(f"[rung-kl] {i}/{len(recon)} ({time.time()-t0:.0f}s) last={name} kl={kl:.5f}", flush=True)

    # rank, classify, and derive the RED list on KL (the drift metric)
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["kl_per_tok"])
    kls = [v["kl_per_tok"] for _, v in ranked]
    cut = kls[max(0, int(len(kls) * (1 - args.red_quantile)) - 1)] if kls else 0
    red = [n for n, v in ranked if v["kl_per_tok"] >= cut]
    by_class = {}
    for n, v in results.items():
        cls = next((s for s in PROJ_SUFFIXES if n.endswith(s + ".weight")), "other")
        by_class.setdefault(cls, []).append(v["kl_per_tok"])
    class_mean = {c: round(sum(v) / len(v), 6) for c, v in sorted(by_class.items())}

    out = {
        "schema": "strand_rung_kl_v1",
        "base": os.path.abspath(args.base),
        "recon": os.path.abspath(args.recon),
        "dataset_id": dataset_id, "dataset_fp": dataset_fp,
        "ctx": args.ctx, "chunks": len(windows), "split": "train",
        "base_nll_per_tok": round(base_nll, 6),
        "class_mean_kl": class_mean,
        "red_quantile": args.red_quantile,
        "red_tensors": red,
        "pv_tensors_regex": "|".join(
            sorted({n.replace(".weight", "").split("model.layers.")[-1] for n in red})) if red else "",
        "tensors": {n: v for n, v in ranked},
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[rung-kl] class mean KL: {class_mean}", flush=True)
    print(f"[rung-kl] RED ({len(red)} tensors >= q{args.red_quantile}): {red[:5]}...", flush=True)
    print(f"[rung-kl] wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
