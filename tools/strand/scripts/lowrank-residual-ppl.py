#!/usr/local/bin/python3
"""lowrank-residual-ppl.py - the output-space low-rank residual lever (PPL judge).

Doctrine (frontier doc §9): RHT whitens the WEIGHT error (error-spectrum.py
weight-MSE space: rank-1 ~ 0.2%), so weight-MSE residuals are dead. But the
OUTPUT error is low-rank (activation-weighted space: rank-1 ~ 19%), because the
activation distribution is anisotropic. This lever absorbs the top-r output-error
modes with a deterministic bf16 low-rank residual added in the MAC epilogue:

    y = decode(recon) @ x  +  A @ (B @ x)  +  c            # c = de-bias (adopted)

The factors come from the SVD of the activation-weighted error. With
D = diag(feature_rms), F = (recon - orig) * D, top-r SVD F = U_r S_r V_r^T:

    A = -U_r * S_r          (out x r)
    B =  V_r^T * (1/rms)    (r x in)      => A @ B = -(recon-orig) projected to the
                                              r directions that matter for OUTPUT

Billed mass: r * (out + in) * 16 bits, bf16. This is LiftQuant/LittleBit latent
compensation, but chosen by output error (not weight MSE), deterministic on decode,
and computed from STRAND's encode-time triad (orig + recon + activation stats).

A/B is run ON TOP of de-bias (already adopted), so the gate measures what the
residual adds BEYOND the rank-0 mean term. Promote.py-compatible output.
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
if os.path.isdir(os.path.join(TOOLS, "strand_eval")):
    sys.path.insert(0, TOOLS)
from strand_eval.core import (  # noqa: E402
    build_record, eval_chunks, load_model_and_tokenizer, load_wikitext)

PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def recon_files(path):
    if os.path.isfile(path):
        return [path]
    p = os.path.join(path, "model.safetensors")
    if os.path.exists(p):
        return [p]
    f = sorted(glob.glob(os.path.join(path, "model-*-of-*.safetensors"))) or \
        sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not f:
        raise SystemExit(f"[lowrank] no safetensors in {path}")
    return f


def load_recon(path):
    from safetensors import safe_open
    out = {}
    for fp in recon_files(path):
        with safe_open(fp, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if k.endswith(".weight") and any(k.endswith(s + ".weight") for s in PROJ_SUFFIXES):
                    out[k] = sf.get_tensor(k).to(torch.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--actmean", required=True)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--match", default="all", help="restrict residual to a class, e.g. up_proj")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tag", default="lowrank")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or f"research/lowrank-ppl-{args.tag}.json"

    model, tok, device_mode, resolved, input_device = load_model_and_tokenizer(
        args.base, args.device, args.dtype)
    am = json.load(open(args.actmean)).get("modules", {})
    recon = load_recon(args.recon)

    # build de-bias corrections (rank-0) AND low-rank factors (rank-r) from the triad
    debias_c = {}
    lr_factors = {}
    bias_rows = 0
    lr_params = 0
    total_w = 0
    with torch.no_grad():
        for name, r in sorted(recon.items()):
            mod_name = name[:-len(".weight")]
            module = model.get_submodule(mod_name)
            orig = module.weight.detach().to(torch.float32)
            if tuple(orig.shape) != tuple(r.shape):
                raise SystemExit(f"[lowrank] shape mismatch {name}")
            rec = am.get(mod_name) or {}
            mu = torch.tensor(rec.get("feature_mean", [0.0] * orig.shape[1]), dtype=torch.float32)
            rms = torch.tensor(rec.get("feature_rms", [1.0] * orig.shape[1]), dtype=torch.float32)
            E = (r - orig)                                   # recon - orig
            debias_c[mod_name] = -(E @ mu)                   # rank-0 (adopted)
            bias_rows += E.shape[0]
            total_w += E.numel()
            if args.match != "all" and args.match not in name:
                module.weight.copy_(r.to(module.weight.dtype)); continue
            # output-space low-rank: SVD of activation-weighted error
            Ew = (E * rms[None, :]).numpy()
            U, S, Vt = np.linalg.svd(Ew, full_matrices=False)
            rk = min(args.rank, len(S))
            A = -(U[:, :rk] * S[:rk])                        # out x r
            B = Vt[:rk, :] * (1.0 / rms.numpy())[None, :]    # r x in
            lr_factors[mod_name] = (torch.tensor(A, dtype=torch.float32),
                                    torch.tensor(B, dtype=torch.float32))
            lr_params += rk * (E.shape[0] + E.shape[1])
            module.weight.copy_(r.to(module.weight.dtype))   # load recon as the base

    print(f"[lowrank] {len(recon)} tensors; residual on '{args.match}' rank={args.rank}; "
          f"lr_params={lr_params} ({lr_params*16/total_w:.5f} bpw bf16)", flush=True)

    def install(use_lowrank):
        hooks = []
        for mod_name, c in debias_c.items():
            module = model.get_submodule(mod_name)
            A = B = None
            if use_lowrank and mod_name in lr_factors:
                A, B = lr_factors[mod_name]

            def mk(c=c, A=A, B=B):
                def hook(mod, inp, output):
                    x = inp[0]
                    add = c.to(output.device, output.dtype)
                    out = output + (add.view(1, 1, -1) if output.dim() == 3 else add)
                    if A is not None:
                        # A @ (B @ x): deterministic low-rank residual in the MAC epilogue
                        bx = torch.nn.functional.linear(x.to(torch.float32), B.to(x.device))
                        lr = torch.nn.functional.linear(bx, A.t().to(x.device).t())
                        out = out + lr.to(output.dtype)
                    return out
                return hook
            hooks.append(module.register_forward_hook(mk()))
        return hooks

    enc, dataset_id, dataset_fp = load_wikitext(tok, split="test")
    n_chunks = min(enc.shape[0] // args.ctx, args.chunks)
    chunks = [enc[i * args.ctx:(i + 1) * args.ctx] for i in range(n_chunks)]
    ce_slice = 512 if str(resolved).startswith("mps") else 0

    def run(label, use_lr):
        h = install(use_lr)
        nll, ntok = eval_chunks(model, chunks, input_device, ce_slice=ce_slice)
        for x in h:
            x.remove()
        ppl = math.exp(nll / ntok)
        print(f"[lowrank] {label}: ppl={ppl:.6f}", flush=True)
        return ppl, ntok

    # A = de-bias only (the adopted baseline); B = de-bias + low-rank residual
    ppl_a, nt = run("debias_only", False)
    ppl_b, nt2 = run("debias+lowrank", True)
    if nt != nt2:
        raise SystemExit("[lowrank] token mismatch")

    ratio = ppl_b / ppl_a
    result = {
        "schema": "strand_lowrank_residual_ab_v1",
        "base": os.path.abspath(args.base), "recon": os.path.abspath(args.recon),
        "rank": args.rank, "match": args.match,
        "residual_bpw": round(lr_params * 16 / total_w, 6),
        "ppl_debias_only": ppl_a, "ppl_debias_lowrank": ppl_b,
        "delta_ppl": ppl_b - ppl_a, "ratio": ratio, "relative_pct": (ratio - 1) * 100,
        "ctx": args.ctx, "chunks": n_chunks, "dataset_fp": dataset_fp,
        "contamination_warning": abs(ppl_b - ppl_a) < 1e-12,
        "verdict": ("ADOPT" if ratio <= 0.995 else "KILL" if ratio >= 1.0 else "MARGINAL"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"[lowrank] rank={args.rank} {args.match}: {(ratio-1)*100:+.3f}% @ +{result['residual_bpw']} bpw "
          f"-> {result['verdict']}", flush=True)
    print(f"[lowrank] wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
