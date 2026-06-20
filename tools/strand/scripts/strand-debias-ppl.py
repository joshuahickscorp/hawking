#!/usr/local/bin/python3
"""strand-debias-ppl.py - PPL A/B for STRAND activation-mean de-bias.

Loads a full-precision HF base model, replaces any projection weights present in
a STRAND recon safetensors file, then evaluates:

  A. recon baseline
  B. recon + output bias correction

The correction is computed from base/recon weight deltas and the activation means
from scripts/calib-actmean.py. If the calibration file contains feature_mean, the
script uses the stronger vector correction:

    c_i = - sum_j (recon_ij - base_ij) * E[x_j]

otherwise it falls back to scalar mu_bar rowsum correction. The model architecture
is unchanged; corrections are forward hooks, so this is a gate harness rather than
a deployment format change.
"""

import argparse
import glob
import json
import math
import os
import sys

import torch


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
if os.path.isdir(os.path.join(TOOLS, "strand_eval")):
    sys.path.insert(0, TOOLS)

from strand_eval.core import (  # noqa: E402
    build_record,
    eval_chunks,
    load_model_and_tokenizer,
    load_wikitext,
)


PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def recon_files(path):
    if os.path.isfile(path):
        return [path]
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return [single]
    files = sorted(glob.glob(os.path.join(path, "model-*-of-*.safetensors")))
    if files:
        return files
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if files:
        return files
    raise SystemExit(f"[debias-ppl] no safetensors found in {path}")


def sidecar_bpw(path):
    candidates = []
    if os.path.isfile(path):
        candidates.append(path + ".json")
    else:
        candidates += glob.glob(os.path.join(path, "*.safetensors.json"))
        candidates += glob.glob(os.path.join(path, "*.json"))
    vals = []
    for p in candidates:
        try:
            with open(p) as f:
                data = json.load(f)
            v = data.get("aggregate", {}).get("effective_bpw")
            if v is not None:
                vals.append(float(v))
        except Exception:
            continue
    return sum(vals) / len(vals) if vals else None


def load_recon_tensors(path):
    from safetensors import safe_open

    out = {}
    for fpath in recon_files(path):
        with safe_open(fpath, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if key.endswith(".weight") and any(key.endswith(s + ".weight") for s in PROJ_SUFFIXES):
                    out[key] = sf.get_tensor(key).to(torch.float32).contiguous()
    if not out:
        raise SystemExit(f"[debias-ppl] no projection weights found in {path}")
    return out


def module_for_tensor(model, tensor_name):
    mod_name = tensor_name[:-len(".weight")]
    try:
        return model.get_submodule(mod_name), mod_name
    except AttributeError:
        obj = model
        for part in mod_name.split("."):
            obj = getattr(obj, part)
        return obj, mod_name


def load_actmeans(path):
    with open(path) as f:
        data = json.load(f)
    if data.get("schema") != "strand_actmean_v1":
        print(f"[debias-ppl] WARNING: unknown actmean schema {data.get('schema')!r}",
              flush=True)
    return data.get("modules", {})


def correction_for(mod_name, orig, recon, mean_rec, mode):
    delta = recon - orig
    in_features = delta.shape[1]
    if mode == "vector" and mean_rec and "feature_mean" in mean_rec:
        mu = torch.tensor(mean_rec["feature_mean"], dtype=torch.float32)
        if mu.numel() != in_features:
            raise SystemExit(
                f"[debias-ppl] mean dim mismatch for {mod_name}: "
                f"{mu.numel()} vs {in_features}")
        corr = -(delta @ mu)
        kind = "vector"
    else:
        mu_bar = float(mean_rec.get("mean", 0.0) if mean_rec else 0.0)
        corr = -delta.sum(dim=1) * mu_bar
        kind = "scalar"
    return corr.contiguous(), kind


def install_bias_hooks(model, corrections):
    hooks = []
    for mod_name, corr in corrections.items():
        module = model.get_submodule(mod_name)
        module.register_buffer("_strand_debias_correction", corr.detach().cpu(),
                               persistent=False)

        def make_hook():
            def hook(mod, _inputs, output):
                c = mod._strand_debias_correction.to(device=output.device, dtype=output.dtype)
                if output.dim() == 3:
                    return output + c.view(1, 1, -1)
                if output.dim() == 2:
                    return output + c.view(1, -1)
                return output + c
            return hook

        hooks.append(module.register_forward_hook(make_hook()))
    return hooks


def run_ppl(model, chunk_list, input_device, ce_slice, label):
    nll, ntok = eval_chunks(model, chunk_list, input_device, ce_slice=ce_slice)
    ppl = math.exp(nll / ntok)
    print(f"[debias-ppl] {label}: ppl={ppl:.6f} tokens={ntok}", flush=True)
    return ppl, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF base model dir")
    ap.add_argument("--recon", required=True,
                    help="STRAND recon safetensors file or dir containing model*.safetensors")
    ap.add_argument("--actmean", required=True,
                    help="JSON from scripts/calib-actmean.py")
    ap.add_argument("--out", default="research/debias-ppl-ab.json")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=64)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--mode", default="vector", choices=["vector", "scalar"],
                    help="vector uses feature_mean when present; scalar uses module mean")
    ap.add_argument("--tag", default="debias_ab")
    args = ap.parse_args()

    print(f"[debias-ppl] loading base={args.base} recon={args.recon}", flush=True)
    model, tok, device_mode, resolved, input_device = load_model_and_tokenizer(
        args.base, args.device, args.dtype)
    modules = load_actmeans(args.actmean)
    recon = load_recon_tensors(args.recon)

    replaced = 0
    corrections = {}
    correction_kinds = {"vector": 0, "scalar": 0}
    total_corr_l2 = 0.0
    total_corr_absmax = 0.0

    with torch.no_grad():
        for tensor_name, r in sorted(recon.items()):
            module, mod_name = module_for_tensor(model, tensor_name)
            if tuple(module.weight.shape) != tuple(r.shape):
                raise SystemExit(
                    f"[debias-ppl] shape mismatch {tensor_name}: "
                    f"model {tuple(module.weight.shape)} vs recon {tuple(r.shape)}")
            orig = module.weight.detach().to("cpu", torch.float32).contiguous()
            mean_rec = modules.get(mod_name) or modules.get(tensor_name)
            corr, kind = correction_for(mod_name, orig, r, mean_rec, args.mode)
            corrections[mod_name] = corr
            correction_kinds[kind] += 1
            total_corr_l2 += float(corr.pow(2).sum())
            total_corr_absmax = max(total_corr_absmax, float(corr.abs().max()))
            module.weight.copy_(r.to(device=module.weight.device, dtype=module.weight.dtype))
            replaced += 1

    if replaced == 0:
        raise SystemExit("[debias-ppl] no weights replaced")
    print(f"[debias-ppl] replaced {replaced} projection tensors; "
          f"corrections vector={correction_kinds['vector']} scalar={correction_kinds['scalar']} "
          f"corr_l2={math.sqrt(total_corr_l2):.6g} corr_absmax={total_corr_absmax:.6g}",
          flush=True)

    enc, dataset_id, dataset_fp = load_wikitext(tok, split="test")
    n_chunks = enc.shape[0] // args.ctx
    if args.chunks > 0:
        n_chunks = min(n_chunks, args.chunks)
    if n_chunks <= 0:
        raise SystemExit(f"[debias-ppl] ctx={args.ctx} too large for {enc.shape[0]} tokens")
    chunk_list = [enc[i * args.ctx:(i + 1) * args.ctx] for i in range(n_chunks)]
    ce_slice = 512 if str(resolved).startswith("mps") else 0

    baseline_ppl, ntok = run_ppl(model, chunk_list, input_device, ce_slice, "baseline")
    hooks = install_bias_hooks(model, corrections)
    debiased_ppl, ntok2 = run_ppl(model, chunk_list, input_device, ce_slice, "debiased")
    for h in hooks:
        h.remove()

    if ntok != ntok2:
        raise SystemExit("[debias-ppl] token count changed between A/B")

    eff_bpw = sidecar_bpw(args.recon)
    extra = {
        "recon_path": os.path.abspath(args.recon),
        "actmean_path": os.path.abspath(args.actmean),
        "debias_mode": args.mode,
        "projection_tensors_replaced": replaced,
        "correction_kinds": correction_kinds,
        "correction_l2": math.sqrt(total_corr_l2),
        "correction_absmax": total_corr_absmax,
        "eff_bpw": eff_bpw,
    }
    rec_base = build_record(
        model_path=args.base, tag=args.tag + "_baseline", ppl=baseline_ppl,
        ctx=args.ctx, chunks=n_chunks, tokens=ntok, device_resolved=resolved,
        device_mode=device_mode, dtype=args.dtype, dataset_id=dataset_id,
        dataset_fp=dataset_fp, eff_bpw=eff_bpw,
        extra=dict(extra, debias_applied=False))
    rec_deb = build_record(
        model_path=args.base, tag=args.tag + "_debiased", ppl=debiased_ppl,
        ctx=args.ctx, chunks=n_chunks, tokens=ntok, device_resolved=resolved,
        device_mode=device_mode, dtype=args.dtype, dataset_id=dataset_id,
        dataset_fp=dataset_fp, eff_bpw=eff_bpw,
        extra=dict(extra, debias_applied=True))

    ratio = debiased_ppl / baseline_ppl
    result = {
        "schema": "strand_debias_ppl_ab_v1",
        "baseline": rec_base,
        "debiased": rec_deb,
        "delta_ppl": debiased_ppl - baseline_ppl,
        "ratio": ratio,
        "relative_pct": (ratio - 1.0) * 100.0,
        "contamination_warning": abs(debiased_ppl - baseline_ppl) < 1e-12,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[debias-ppl] ratio={ratio:.8f} ({(ratio - 1.0) * 100.0:+.4f}%)", flush=True)
    if result["contamination_warning"]:
        print("[debias-ppl] WARNING: identical PPL; verify hooks/corrections applied",
              flush=True)
    print(f"[debias-ppl] wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
