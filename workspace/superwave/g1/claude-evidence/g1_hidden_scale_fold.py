#!/usr/bin/env python3
"""Codec for the one real profile reuse: shared hidden-channel RMS template.

gate/up: fold col_rms (K=5120) into X, quantize W/s.
down:    fold row_rms (out=5120) into Y, quantize W/s_row.

CPU, real BF16. Appends to /tmp/g1_cross_layer_structure.json.
"""
from __future__ import annotations

import json
import time

src = open("/tmp/g1_cross_layer_structure.py").read()
ns = {}
exec(compile(src.replace("raise SystemExit(main())", "pass"), "/tmp/g1_cross_layer_structure.py", "exec"), ns)

log = ns["log"]
parse_all_headers = ns["parse_all_headers"]
ROOT = ns["ROOT"]
tensor_name = ns["tensor_name"]
load_f32 = ns["load_f32"]
load_hidden = ns["load_hidden"]
iter_row_tiles = ns["iter_row_tiles"]
apply_levels_chunked = ns["apply_levels_chunked"]
lloyd_1d = ns["lloyd_1d"]
uniform_levels = ns["uniform_levels"]
group_amax_and_hist = ns["group_amax_and_hist"]
reconstruct_with_levels = ns["reconstruct_with_levels"]
matvec_cos = ns["matvec_cos"]
cosine = ns["cosine"]
rel_l2 = ns["rel_l2"]
rss_gb = ns["rss_gb"]
OUT = ns["OUT"]

HIDDEN = 5120


def col_rms_from_info(info):
    rows, cols = info["shape"]
    col_ssq = None
    for _, _, tile in iter_row_tiles(info, tile=256):
        s = (tile.astype("float64") ** 2).sum(axis=0)
        col_ssq = s if col_ssq is None else col_ssq + s
    return (col_ssq / rows) ** 0.5


def row_rms_from_info(info):
    rows, cols = info["shape"]
    rms = []
    for _, _, tile in iter_row_tiles(info, tile=256):
        rms.append(((tile.astype("float64") ** 2).mean(axis=1)) ** 0.5)
    import numpy as np
    return np.concatenate(rms).astype("float32")


def main():
    import numpy as np

    t0 = time.time()
    table = parse_all_headers(ROOT)
    report = json.loads(OUT.read_text())
    out = {"schema": "hidden_scale_fold_v1", "tensors": []}

    jobs = [
        # name, layer, fold_axis, probe_layers_for_template
        ("mlp.gate_proj", "mlp.gate_proj.weight", [0, 15, 31, 63], "col"),
        ("mlp.up_proj", "mlp.up_proj.weight", [0, 15, 31, 63], "col"),
        ("mlp.down_proj", "mlp.down_proj.weight", [0, 15, 31, 63], "row"),
        ("self_attn.q_proj", "self_attn.q_proj.weight", [3, 15, 31, 63], "col"),
        ("linear_attn.in_proj_qkv", "linear_attn.in_proj_qkv.weight", [0, 16, 32, 62], "col"),
    ]

    for cname, suffix, layers, axis in jobs:
        log(f"fold {cname} axis={axis}")
        templates = []
        for L in layers:
            info = table[tensor_name(L, suffix)]
            if axis == "col":
                templates.append(col_rms_from_info(info))
            else:
                templates.append(row_rms_from_info(info))
        T = np.stack(templates, 0)
        s_shared = np.maximum(T.mean(axis=0), 1e-12)
        # pairwise centered cosine of templates
        def ctr_cos(a, b):
            ac, bc = a - a.mean(), b - b.mean()
            return cosine(ac, bc)

        pair = []
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                pair.append(
                    {
                        "i": layers[i],
                        "j": layers[j],
                        "centered_cos": ctr_cos(templates[i], templates[j]),
                    }
                )

        # evaluate on first and last
        for L in (layers[0], layers[-1]):
            info = table[tensor_name(L, suffix)]
            W = load_f32(info)
            if axis == "col":
                assert W.shape[1] == s_shared.size, (W.shape, s_shared.size)
                Wn = W / s_shared[None, :].astype(np.float32)
            else:
                assert W.shape[0] == s_shared.size, (W.shape, s_shared.size)
                Wn = W / s_shared[:, None].astype(np.float32)

            rng = np.random.default_rng(0)
            _, _, u = group_amax_and_hist(Wn)
            fit = u[rng.choice(u.size, min(400_000, u.size), replace=False)]
            lv4 = lloyd_1d(fit, 16)
            lv2 = lloyd_1d(fit, 4)

            # score reconstruction in original space
            def score(Wsrc, levels, s_axis):
                rel, cos, _ = apply_levels_chunked(Wsrc, levels)
                # reconstruct in scaled space then fold back
                What_n = reconstruct_with_levels(Wsrc, levels)
                if s_axis == "col":
                    What = What_n * s_shared[None, :].astype(np.float32)
                else:
                    What = What_n * s_shared[:, None].astype(np.float32)
                return rel, cos, What

            # baseline: no fold, same codec on raw W
            _, _, u0 = group_amax_and_hist(W)
            fit0 = u0[rng.choice(u0.size, min(400_000, u0.size), replace=False)]
            lv4_raw = lloyd_1d(fit0, 16)
            r_raw4, c_raw4, _ = apply_levels_chunked(W, lv4_raw)
            r_uni4, c_uni4, _ = apply_levels_chunked(W, uniform_levels(4))
            r_n4, c_n4, What_n4 = score(Wn, lv4, axis)
            # fold-back error vs original W
            rel_fold4 = rel_l2(W, What_n4)
            cos_fold4 = cosine(W, What_n4)

            r_n2, _, What_n2 = score(Wn, lv2, axis)
            rel_fold2 = rel_l2(W, What_n2)

            rec = {
                "class": cname,
                "layer": L,
                "axis": axis,
                "template_n": len(layers),
                "template_pairs_centered_cos_mean": float(np.mean([p["centered_cos"] for p in pair])),
                "raw_priv4_rel_l2": r_raw4,
                "raw_uni4_rel_l2": r_uni4,
                "fold_scaled_priv4_rel_l2_in_scaled_space": r_n4,
                "fold_priv4_rel_l2_original_space": rel_fold4,
                "fold_priv4_cos_original_space": cos_fold4,
                "fold_priv2_rel_l2_original_space": rel_fold2,
                "shared_template_bytes_f16": int(s_shared.size * 2),
                "amortized_bpw": float(s_shared.size * 16) / float(W.size * len(layers) * (64 / len(layers))),
            }
            # WX if possible
            if W.shape[1] == HIDDEN:
                X = load_hidden(L)[192:]
                What_raw = reconstruct_with_levels(W, lv4_raw)
                rec["wx_raw_priv4"] = matvec_cos(W, What_raw, X)
                rec["wx_fold_priv4"] = matvec_cos(W, What_n4, X)
            out["tensors"].append(rec)
            log(
                f"  L{L} raw4={r_raw4:.5f} fold4={rel_fold4:.5f} "
                f"delta={rel_fold4-r_raw4:+.5f} fold2={rel_fold2:.5f}"
            )
            del W, Wn, What_n4, What_n2

    out["template_pairs"] = None
    out["elapsed_s"] = time.time() - t0
    out["rss_gb"] = rss_gb()
    report["hidden_scale_fold"] = out
    OUT.write_text(json.dumps(report, indent=2))
    log(f"fold done {out['elapsed_s']:.1f}s rss={out['rss_gb']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
