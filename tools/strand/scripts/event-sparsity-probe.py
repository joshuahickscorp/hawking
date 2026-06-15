#!/usr/local/bin/python3
"""event-sparsity-probe — extract REAL per-layer activation vectors + tail stats.

The data half of the EVENT-DRIVEN FIRING wave (`crates/strand-decode-kernel/src/
event_mac.rs`, `bin/gate-eventmac`, `research/event-sparsity.md`): runs a small
CPU forward of a pretrained CausalLM (default scratch/qwen-05b, 2 chunks of
WikiText-2 test at ctx 2048 — the canon eval windows), hooks the projection
Linears, and saves CONSECUTIVE-token input-activation vectors per layer (raw
f32 LE, row-major [tokens, dim]) plus measured tail statistics.

Consecutive positions matter: the delta / surprise-coding mode in event_mac.rs
bills |x_t - x_{t-1}|, and teacher-forced consecutive positions are the honest
cheap stand-in for what autoregressive decode sees.

Outputs (default research/event-sparsity-probe/):
  manifest.tsv             tensor_name<TAB>tokens<TAB>dim<TAB>relpath  (gate input)
  acts/<safe_name>.bin     raw little-endian f32, [tokens, dim] row-major
  stats.json               per-module tail stats (kurtosis, |x| quantiles,
                           energy concentration, threshold fire fractions,
                           delta fire fractions) + the aggregate tail table

CPU-ONLY by design (this wave does not touch MPS — a sibling wave owns the GPU).
Numbers from a contended box are fine here: these are distributional
measurements of fixed forward passes, not wall-clock benchmarks.

Usage:
  /usr/local/bin/python3 scripts/event-sparsity-probe.py \
      --model scratch/qwen-05b --chunks 2 --ctx 2048 --tokens 48 \
      --layers 0,5,11,17,23 --out research/event-sparsity-probe
"""

import argparse
import json
import math
import os
import sys

import torch


PROJ_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def load_wikitext_ids(tok, split):
    """Same fallback dance as strand-qat.py (newer hub rejects bare 'wikitext')."""
    from datasets import load_dataset

    for ds_id in ("wikitext", "Salesforce/wikitext"):
        try:
            ds = load_dataset(ds_id, "wikitext-2-raw-v1", split=split)
            text = "\n\n".join(ds["text"])
            return tok(text, return_tensors="pt").input_ids[0]
        except Exception as e:  # noqa: BLE001 - we report and fall through
            print(f"[probe] dataset id {ds_id!r} failed: {e}", flush=True)
    raise SystemExit("[probe] failed to load wikitext-2-raw-v1")


def quantile(sorted_vals, q):
    """Quantile of a pre-sorted 1-D tensor (linear interp)."""
    n = sorted_vals.numel()
    if n == 0:
        return 0.0
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def tail_stats(acts):
    """acts: [tokens, dim] f32. Returns the measured-tail dict the doc cites."""
    flat = acts.reshape(-1).to(torch.float64)
    mean = flat.mean()
    var = flat.var(unbiased=False)
    std = var.sqrt()
    # Excess kurtosis: 0 for a Gaussian; heavy tails >> 0.
    kurt = float((((flat - mean) / (std + 1e-30)) ** 4).mean() - 3.0)

    mags = flat.abs()
    smags = mags.sort().values
    qs = {f"absx_p{int(p*100)}": quantile(smags, p)
          for p in (0.50, 0.75, 0.90, 0.95, 0.99, 0.999)}
    qs["absx_max"] = float(smags[-1])

    # Energy concentration per token, averaged: share of sum(x^2) held by the
    # top {1,5,10,25}% largest-|x| dims.
    e = acts.to(torch.float64) ** 2
    tot = e.sum(dim=1, keepdim=True).clamp_min(1e-30)
    dim = acts.shape[1]
    conc = {}
    sorted_e = e.sort(dim=1, descending=True).values
    csum = sorted_e.cumsum(dim=1)
    for frac in (0.01, 0.05, 0.10, 0.25):
        k = max(1, int(round(dim * frac)))
        conc[f"energy_top{int(frac*100)}pct_dims"] = float(
            (csum[:, k - 1:k] / tot).mean())

    # Threshold fire fractions: fraction of dims with |x| >= tau, where tau is
    # set at |x| quantiles — the same sweep gate-eventmac uses.
    fire = {}
    for p in (0.50, 0.75, 0.90, 0.95, 0.99):
        tau = quantile(smags, p)
        fire[f"fire_frac_at_absx_p{int(p*100)}"] = float((mags >= tau).double().mean())

    # Delta (surprise) tail: |x_t - x_{t-1}| over consecutive tokens.
    delta = {}
    if acts.shape[0] >= 2:
        d = (acts[1:] - acts[:-1]).reshape(-1).abs().to(torch.float64)
        sd = d.sort().values
        for p in (0.50, 0.90, 0.99):
            delta[f"absdx_p{int(p*100)}"] = quantile(sd, p)
        # Fired fraction if tau is set at the ACTIVATION's quantiles — how much
        # smaller the surprise signal is than the salience signal.
        for p in (0.50, 0.75, 0.90):
            tau = quantile(smags, p)
            delta[f"delta_fire_frac_at_absx_p{int(p*100)}"] = float(
                (d >= tau).double().mean())
        delta["mean_absdx_over_mean_absx"] = float(d.mean() / (mags.mean() + 1e-30))

    return {
        "tokens": int(acts.shape[0]),
        "dim": int(acts.shape[1]),
        "mean": float(mean),
        "std": float(std),
        "excess_kurtosis": kurt,
        **qs,
        **conc,
        **fire,
        **delta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="scratch/qwen-05b")
    ap.add_argument("--out", default="research/event-sparsity-probe")
    ap.add_argument("--chunks", type=int, default=2,
                    help="non-overlapping ctx windows to forward (canon protocol)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=48,
                    help="CONSECUTIVE token positions saved per module per chunk")
    ap.add_argument("--start", type=int, default=1024,
                    help="first saved position within each chunk (skip the cold prefix)")
    ap.add_argument("--layers", default="0,5,11,17,23",
                    help="comma-separated layer indices to hook")
    ap.add_argument("--device", default="cpu", choices=["cpu"],
                    help="cpu only — this wave does not use MPS (sibling owns it)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    layers = sorted({int(s) for s in args.layers.split(",") if s.strip() != ""})
    os.makedirs(os.path.join(args.out, "acts"), exist_ok=True)

    print(f"[probe] loading {args.model} (fp32, eager, cpu)…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager")
    model.eval()

    n_layers = model.config.num_hidden_layers
    layers = [l for l in layers if 0 <= l < n_layers]
    print(f"[probe] hooking layers {layers} of {n_layers}", flush=True)

    # name -> list of [tokens, dim] tensors (one per chunk)
    captured = {}
    hooks = []

    def make_hook(name):
        def pre_hook(_module, inputs):
            x = inputs[0]
            # x: [batch, seq, dim] (batch=1). Save a consecutive-token span.
            seq = x.shape[1]
            s = min(args.start, max(0, seq - args.tokens))
            e = min(seq, s + args.tokens)
            captured.setdefault(name, []).append(
                x[0, s:e, :].detach().to(torch.float32).cpu().clone())
        return pre_hook

    for li in layers:
        layer = model.model.layers[li]
        for suffix in PROJ_SUFFIXES:
            obj = layer
            for part in suffix.split("."):
                obj = getattr(obj, part)
            name = f"model.layers.{li}.{suffix}"
            hooks.append(obj.register_forward_pre_hook(make_hook(name)))

    ids = load_wikitext_ids(tok, "test")
    n_chunks = min(args.chunks, ids.numel() // args.ctx)
    print(f"[probe] forwarding {n_chunks} chunk(s) of ctx {args.ctx} (cpu)…", flush=True)
    with torch.no_grad():
        for c in range(n_chunks):
            window = ids[c * args.ctx:(c + 1) * args.ctx].unsqueeze(0)
            model(window)
            print(f"[probe] chunk {c + 1}/{n_chunks} done", flush=True)
    for h in hooks:
        h.remove()

    manifest_lines = []
    stats = {}
    for name in sorted(captured):
        acts = torch.cat(captured[name], dim=0)  # [chunks*tokens, dim]
        safe = name.replace(".", "_")
        rel = os.path.join("acts", f"{safe}.bin")
        path = os.path.join(args.out, rel)
        acts.contiguous().numpy().astype("<f4").tofile(path)
        # tensor name as it appears in the .strand artifact:
        tensor_name = name + ".weight"
        manifest_lines.append(
            f"{tensor_name}\t{acts.shape[0]}\t{acts.shape[1]}\t{rel}")
        st = tail_stats(acts)
        stats[tensor_name] = st
        print(f"[probe] {tensor_name}: tokens={st['tokens']} dim={st['dim']} "
              f"kurtosis={st['excess_kurtosis']:.1f} "
              f"top1%dims-energy={st['energy_top1pct_dims']*100:.1f}% "
              f"mean|dx|/mean|x|={st.get('mean_absdx_over_mean_absx', float('nan')):.3f}",
              flush=True)

    with open(os.path.join(args.out, "manifest.tsv"), "w") as f:
        f.write("\n".join(manifest_lines) + "\n")
    meta = {
        "model": args.model,
        "chunks": n_chunks,
        "ctx": args.ctx,
        "tokens_per_chunk": args.tokens,
        "start": args.start,
        "layers": layers,
        "note": ("teacher-forced consecutive positions; raw (pre-RHT) activation "
                 "space — the per-row RHT means no single rotated space exists "
                 "at deployment (see event_mac.rs module doc)"),
        "modules": stats,
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[probe] wrote {len(manifest_lines)} modules to {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
