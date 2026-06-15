#!/usr/local/bin/python3
"""error-spectrum.py - is the STRAND quantization error low-rank? (the scout)

Black-hole doctrine, frontier doc: before spending bits on a low-rank residual
correction (recon + U V^T, deterministic bf16 MAC epilogue), measure whether the
error E = recon - orig even HAS exploitable low-rank structure. RHT whitening
predicts E is near-white (low-rank dead, like diag-Hessian). But de-bias worked,
so the mean (rank-0) survived. This asks: does rank-1..r survive too?

Cheap scout (no PPL, no activations needed): per tensor, thin-SVD of E, report
cumulative energy captured at ranks {1,4,16,64} and the bpw a bf16 rank-r residual
would cost. Decision is made on energy concentration; the PPL A/B is the later judge.

  energy_r = sum(sigma[:r]^2) / sum(sigma^2)        # fraction of error captured
  bpw_r    = r * (rows + cols) * 16 / (rows * cols) # bf16 residual mass per weight

If energy_1 is high (say > 0.3) the error is concentrated -> low-rank lever is ALIVE.
If energy_64 is still ~ 64/min(rows,cols) (i.e. flat/white) -> RHT whitened it -> DEAD.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

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
        raise SystemExit(f"[err-spec] no safetensors in {path}")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="scratch/qwen-05b")
    ap.add_argument("--recon", required=True)
    ap.add_argument("--match", default="down_proj",
                    help="substring filter; 'all' for every projection")
    ap.add_argument("--actmean", default=None,
                    help="calib json; when set, weight error columns by feature_rms "
                         "(output-space spectrum) instead of raw weight-MSE space")
    ap.add_argument("--ranks", default="1,4,16,64")
    ap.add_argument("--tag", default="errspec")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    out_path = args.out or f"research/error-spectrum-{args.tag}.json"
    ranks = [int(r) for r in args.ranks.split(",")]
    torch.set_num_threads(args.threads)
    try:
        import os as _os
        _os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    except Exception:
        pass

    from safetensors import safe_open

    base_f = recon_files(args.base)
    recon_f = recon_files(args.recon)

    def load_all(files):
        d = {}
        for fp in files:
            with safe_open(fp, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    if k.endswith(".weight") and any(k.endswith(s + ".weight") for s in PROJ_SUFFIXES):
                        if args.match == "all" or args.match in k:
                            d[k] = sf.get_tensor(k).to(torch.float32).numpy()
        return d

    base = load_all(base_f)
    recon = load_all(recon_f)
    names = sorted(set(base) & set(recon))
    if not names:
        raise SystemExit(f"[err-spec] no overlapping tensors match '{args.match}'")
    print(f"[err-spec] {len(names)} tensors match '{args.match}'", flush=True)

    actrms = {}
    if args.actmean:
        am = json.load(open(args.actmean)).get("modules", {})
        for mod, rec in am.items():
            if "feature_rms" in rec:
                actrms[mod + ".weight"] = np.asarray(rec["feature_rms"], dtype=np.float32)
        if not actrms:
            raise SystemExit("[err-spec] --actmean has no feature_rms; re-run calib-actmean.py")
        print(f"[err-spec] activation-weighted (output) space: {len(actrms)} feature_rms vectors", flush=True)
    space = "output(act-weighted)" if args.actmean else "weight-MSE"

    rows_out = {}
    agg = {r: [] for r in ranks}
    for i, n in enumerate(names, 1):
        E = recon[n] - base[n]                      # [out, in]
        if args.actmean:
            w = actrms.get(n)
            if w is None or w.shape[0] != E.shape[1]:
                continue
            E = E * w[None, :]                       # column-scale by activation energy
        # thin SVD energy spectrum (singular values^2 = error energy per mode)
        s = np.linalg.svd(E, compute_uv=False)
        total = float((s ** 2).sum()) or 1.0
        rmin = min(E.shape)
        rec = {"shape": list(E.shape), "err_fro": float(np.sqrt(total))}
        for r in ranks:
            er = float((s[:r] ** 2).sum()) / total
            rec[f"energy_r{r}"] = round(er, 4)
            # white baseline: a white matrix captures r/rmin of energy in r modes
            rec[f"lift_r{r}"] = round(er / (min(r, rmin) / rmin), 3)  # >1 = concentrated
            rec[f"bpw_r{r}"] = round(r * (E.shape[0] + E.shape[1]) * 16 / (E.shape[0] * E.shape[1]), 5)
            agg[r].append(er)
        rows_out[n] = rec
        if i % 6 == 0 or i == len(names):
            r1 = rec["energy_r1"]; l1 = rec["lift_r1"]
            print(f"[err-spec] {i}/{len(names)} {n}: E_r1={r1} lift_r1={l1}x", flush=True)

    summary = {}
    for r in ranks:
        vals = agg[r]
        summary[f"mean_energy_r{r}"] = round(float(np.mean(vals)), 4)
        summary[f"median_energy_r{r}"] = round(float(np.median(vals)), 4)

    # verdict: rank-1 lift well above 1.0 and rising energy => concentrated => ALIVE
    mean_e1 = summary["mean_energy_r1"]
    mean_e16 = summary.get("mean_energy_r16", 0)
    verdict = ("ALIVE - error is low-rank concentrated; build the residual lever"
               if mean_e1 >= 0.15 or mean_e16 >= 0.5
               else "DEAD - error is near-white (RHT whitened it); low-rank residual will not pay")

    out = {
        "schema": "strand_error_spectrum_v1",
        "base": os.path.abspath(args.base),
        "recon": os.path.abspath(args.recon),
        "match": args.match,
        "space": space,
        "ranks": ranks,
        "summary": summary,
        "verdict": verdict,
        "tensors": rows_out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[err-spec] summary: {summary}", flush=True)
    print(f"[err-spec] VERDICT: {verdict}", flush=True)
    print(f"[err-spec] wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
