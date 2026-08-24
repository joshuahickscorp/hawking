#!/usr/bin/env python3
"""Resume: embed + lm_head only, streaming residual so peak stays ~6 GB."""
from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

import g1_gen_residual as g

torch.set_num_threads(int(os.environ.get("G1_TORCH_THREADS", "8")))
torch.set_grad_enabled(False)

OUT = Path("/tmp/g1_gen_residual.json")
ROW_BLOCK = 2048


def stream_codec_on_rows(W, bits_or_bin, group, A=None, B=None):
    """Accumulate ||target - Q(target)||^2 and payload.

    If A,B given, target is residual R = W - A@B, and error is vs original W
    (i.e. ||R-Q(R)||^2). If not, target is W.
    """
    m, n = W.shape
    err2 = 0.0
    payload = 0
    for i0 in range(0, m, ROW_BLOCK):
        i1 = min(m, i0 + ROW_BLOCK)
        block = W[i0:i1]
        if A is not None:
            target = block - (A[i0:i1] @ B)
        else:
            target = block
        if bits_or_bin == "bin":
            recon, pay = g.quant_binary(target, 128)
        else:
            recon, pay = g.quant_uniform(target, bits_or_bin, group)
        err2 += g.frob2(target - recon)
        payload += pay
        del target, recon
    return err2, payload


def residual_stats_stream(W, A, B):
    m, n = W.shape
    n_elem = m * n
    s1 = s2 = s4 = sabs = 0.0
    max_abs = 0.0
    # first pass
    for i0 in range(0, m, ROW_BLOCK):
        i1 = min(m, i0 + ROW_BLOCK)
        R = W[i0:i1] - (A[i0:i1] @ B)
        sl = R.reshape(-1)
        s1 += float(sl.sum().item())
        s2 += float(sl.square().sum().item())
        sabs += float(sl.abs().sum().item())
        max_abs = max(max_abs, float(sl.abs().max().item()))
        del R
    mean = s1 / n_elem
    m2 = max(s2 / n_elem - mean * mean, 0.0)
    rms = math.sqrt(s2 / n_elem)
    # second pass m4
    for i0 in range(0, m, ROW_BLOCK):
        i1 = min(m, i0 + ROW_BLOCK)
        R = W[i0:i1] - (A[i0:i1] @ B)
        xc = R.reshape(-1) - mean
        s4 += float((xc * xc * xc * xc).sum().item())
        del R, xc
    kurt = (s4 / n_elem) / (m2 * m2) - 3.0 if m2 > 0 else 0.0
    return {
        "n": n_elem,
        "mean": mean,
        "std": math.sqrt(m2),
        "rms": rms,
        "mean_abs": sabs / n_elem,
        "max_abs": max_abs,
        "peak_over_rms": (max_abs / rms) if rms > 0 else None,
        "excess_kurtosis": kurt,
        "frac_gt_3rms": None,
        "abs_p50": None,
        "abs_p90": None,
        "abs_p99": None,
        "abs_p999": None,
        "note": "streamed; percentiles skipped to keep RSS low",
    }


def analyze_large(idx, name, cls):
    t0 = time.perf_counter()
    g.log(f"LOAD {cls} {name}")
    W = idx.load_f32(name)
    m, n = int(W.shape[0]), int(W.shape[1])
    w_f2 = g.frob2(W)
    w_f = math.sqrt(w_f2)
    w_stats = g.stats(W)
    g.log(f"  shape=({m},{n}) ||W||_F={w_f:.6g} load_s={time.perf_counter()-t0:.2f}")

    t1 = time.perf_counter()
    U, S, Vh, method = g.rsvd(W, 128)
    svd_s = time.perf_counter() - t1
    S_np = S.detach().cpu().numpy().astype(np.float64)
    energy = {
        str(r): float(np.square(S_np[:r]).sum() / w_f2)
        for r in g.R_SWEEP
        if r <= S_np.size
    }
    spec_idx = [0, 1, 3, 7, 15, 31, 63, 127]
    spectrum = {str(i): float(S_np[i]) for i in spec_idx if i < S_np.size}
    decay = {}
    for a, b in ((0, 7), (0, 31), (0, 63), (0, 127)):
        if b < S_np.size and S_np[b] > 0:
            decay[f"s{a+1}_over_s{b+1}"] = float(S_np[a] / S_np[b])
    g.log(f"  svd_s={svd_s:.1f} energy64={energy.get('64')}")

    orig_codecs = {}
    codec_specs = (
        ("binary_g128", "bin", 128),
        ("uniform_q2_g64", 2, 64),
        ("uniform_q3_g64", 3, 64),
        ("uniform_q4_g64", 4, 64),
    )
    for cname, bits, group in codec_specs:
        e2, payload = stream_codec_on_rows(W, bits, group)
        rel = math.sqrt(e2 / w_f2)
        orig_codecs[cname] = {
            "rel_l2": rel,
            "cosine": None,
            "err_frob2": e2,
            "payload_bytes": int(payload),
            "bpw": 8.0 * payload / (m * n),
            "note": "streamed; cosine not computed",
        }
    q4_rel = orig_codecs["uniform_q4_g64"]["rel_l2"]

    residual_ranks = []
    for r in (8, 32, 64, 128):
        if r > S_np.size:
            continue
        A32 = U[:, :r] * S[:r]
        B32 = Vh[:r]
        # lr-only error streamed
        lr32_e2 = 0.0
        lr16_e2 = 0.0
        A16 = A32.half().float()
        B16 = B32.half().float()
        for i0 in range(0, m, ROW_BLOCK):
            i1 = min(m, i0 + ROW_BLOCK)
            G32 = A32[i0:i1] @ B32
            G16 = A16[i0:i1] @ B16
            lr32_e2 += g.frob2(W[i0:i1] - G32)
            lr16_e2 += g.frob2(W[i0:i1] - G16)
            del G32, G16
        g32_rel = math.sqrt(lr32_e2 / w_f2)
        g16_rel = math.sqrt(lr16_e2 / w_f2)
        f16_extra = math.sqrt(max(g16_rel * g16_rel - g32_rel * g32_rel, 0.0))
        r_stats = residual_stats_stream(W, A16, B16)
        res_codecs = {}
        for cname, bits, group in codec_specs:
            e2, payload = stream_codec_on_rows(W, bits, group, A16, B16)
            rec = {
                "rel_l2": math.sqrt(e2 / w_f2),
                "cosine": None,
                "err_frob2": e2,
            }
            fb = g.factor_bytes(m, n, r)
            rec["residual_payload_bytes"] = int(payload)
            rec["factor_bytes_f16"] = int(fb)
            rec["total_bytes"] = int(payload + fb)
            rec["residual_bpw"] = 8.0 * payload / (m * n)
            rec["factor_bpw"] = 8.0 * fb / (m * n)
            rec["total_bpw"] = 8.0 * (payload + fb) / (m * n)
            rec["beats_orig_rel"] = rec["rel_l2"] < orig_codecs[cname]["rel_l2"]
            rec["rel_improvement"] = (
                orig_codecs[cname]["rel_l2"] / rec["rel_l2"] if rec["rel_l2"] > 0 else None
            )
            res_codecs[cname] = rec
        bits_to_target = {}
        for tgt in g.TARGETS + (q4_rel,):
            key = f"rel_l2<={tgt:.6f}"
            hits = [
                (rec["total_bpw"], cname, rec["rel_l2"])
                for cname, rec in res_codecs.items()
                if rec["rel_l2"] <= tgt
            ]
            if hits:
                hits.sort()
                bits_to_target[key] = {
                    "total_bpw": hits[0][0],
                    "codec": hits[0][1],
                    "rel_l2": hits[0][2],
                    "target": tgt,
                }
            else:
                bits_to_target[key] = {
                    "total_bpw": None,
                    "codec": None,
                    "rel_l2": None,
                    "target": tgt,
                    "note": "no tested residual codec hits target",
                }
        residual_ranks.append(
            {
                "r": r,
                "explained_frob": energy.get(str(r)),
                "lr_only_rel_l2_f32": g32_rel,
                "lr_only_rel_l2_f16": g16_rel,
                "f16_factor_extra_rel_l2": f16_extra,
                "residual_stats": r_stats,
                "residual_top32_energy_frac": None,
                "residual_s1_over_s8": None,
                "residual_spectrum_method": "skipped_to_cap_rss",
                "orig_peak_over_rms": w_stats["peak_over_rms"],
                "resid_peak_over_rms": r_stats["peak_over_rms"],
                "orig_excess_kurtosis": w_stats["excess_kurtosis"],
                "resid_excess_kurtosis": r_stats["excess_kurtosis"],
                "codecs": res_codecs,
                "bits_to_target": bits_to_target,
            }
        )
        del A32, B32, A16, B16
        gc.collect()
        g.log(f"  r={r} explained={energy.get(str(r))} lr_rel={g16_rel:.4f}")

    rec = {
        "name": name,
        "class": cls,
        "shape": [m, n],
        "n_elem": m * n,
        "dtype_on_disk": "BF16",
        "frob": w_f,
        "weight_stats": w_stats,
        "svd_method": method,
        "svd_k": int(S_np.size),
        "svd_s": svd_s,
        "approx_captured_energy_of_computed_S": float(np.square(S_np).sum() / w_f2),
        "energy_frac": energy,
        "spectrum": spectrum,
        "decay": decay,
        "orig_codecs": orig_codecs,
        "q4_rel_l2": q4_rel,
        "residual": residual_ranks,
        "wall_s": time.perf_counter() - t0,
        "rss_max_gb_after": g.rss_gb(),
        "path": "streamed_large",
    }
    del W, U, S, Vh
    gc.collect()
    g.log(f"  DONE {cls} energy@64={energy.get('64')} q4_rel={q4_rel:.4f} wall={rec['wall_s']:.1f}s")
    return rec


def summarize(results):
    by = {}
    for t in results["tensors"]:
        by.setdefault(t["class"], []).append(t)
    summary = {}

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(sum(xs) / len(xs)) if xs else None

    gctl = results.get("gaussian_control") or {}
    g64 = (gctl.get("energy_frac") or {}).get("64")
    for cls, rows in by.items():
        e64 = [r["energy_frac"].get("64") for r in rows]
        e128 = [r["energy_frac"].get("128") for r in rows]
        e256 = [r["energy_frac"].get("256") for r in rows]
        q4 = [r["q4_rel_l2"] for r in rows]
        bin_imp, q4_imp = [], []
        for r in rows:
            for rr in r["residual"]:
                if rr["r"] == 64:
                    bin_imp.append(rr["codecs"]["binary_g128"].get("rel_improvement"))
                    q4_imp.append(rr["codecs"]["uniform_q4_g64"].get("rel_improvement"))
        summary[cls] = {
            "n": len(rows),
            "energy64_mean": mean(e64),
            "energy64_min": min(e64) if e64 else None,
            "energy64_max": max(e64) if e64 else None,
            "energy128_mean": mean(e128),
            "energy256_mean": mean(e256),
            "q4_rel_l2_mean": mean(q4),
            "r64_binary_rel_improvement_mean": mean(bin_imp),
            "r64_q4_rel_improvement_mean": mean(q4_imp),
            "flat_vs_gaussian64": (mean(e64) / g64) if (e64 and g64) else None,
        }
    results["summary_by_class"] = summary
    results["rss_max_gb"] = g.rss_gb()


def main():
    g.log("resume embed+lm_head streamed")
    idx = g.ShardIndex(g.BF16_DIR)
    results = json.loads(OUT.read_text())
    have = {t["name"] for t in results["tensors"]}
    todo = [
        ("language_model.model.embed_tokens.weight", "embed"),
        ("language_model.lm_head.weight", "lm_head"),
    ]
    for name, cls in todo:
        if name in have:
            g.log(f"skip already have {name}")
            continue
        rec = analyze_large(idx, name, cls)
        results["tensors"].append(rec)
        OUT.write_text(json.dumps(results, indent=2))
        if g.rss_gb() > 18.0:
            g.log("ABORT rss_max>18G")
            results.setdefault("errors", []).append(
                {"where": "rss_guard_embed", "rss_max_gb": g.rss_gb()}
            )
            break
    summarize(results)
    OUT.write_text(json.dumps(results, indent=2))
    g.log(f"WROTE {OUT} tensors={len(results['tensors'])}")
    print(json.dumps(results["summary_by_class"], indent=2))


if __name__ == "__main__":
    main()
