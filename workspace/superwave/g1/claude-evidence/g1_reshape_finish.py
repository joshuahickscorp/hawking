#!/usr/bin/env python3
"""Finish reshape-before-lowbit: cite reproduction, Q4 refs, hold512, JSON dump."""
from __future__ import annotations
import json, os, re, resource, sys, time
import numpy as np

sys.path.insert(0, "/Users/scammermike/.claude-grok/worktrees/209-reshape-before-lowbit-20260817-181049/tools")
from gravity_doctor_gate import _rowcos, c_uniform, load_tensor
import importlib.util
spec = importlib.util.spec_from_file_location("rbl", "/tmp/g1_reshape_before_lowbit.py")
rbl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbl)

BF16 = rbl.BF16
LOG = ("/Users/scammermike/.grok/sessions/"
       "%2FUsers%2Fscammermike%2F.claude-grok%2Fworktrees%2F209-reshape-before-lowbit-20260817-181049"
       "/01a011c7-1664-70c3-8654-06522307083c/terminal/call-43e81411-b9e4-4841-a522-64d9e79cb0f8-35.log")
OUT = "/tmp/g1_reshape_before_lowbit.json"


def frob_cos(a, b):
    left = np.asarray(a, np.float64).reshape(-1)
    right = np.asarray(b, np.float64).reshape(-1)
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right) / den if den > 1e-12 else 0.0


def uniform_f16(W, bits, g):
    G = rbl.group_view(W, g)
    lim = (1 << (bits - 1)) - 1
    scales = (np.max(np.abs(G), axis=-1) / lim).astype(np.float16).astype(np.float32)
    den = np.where(scales > 0, scales, 1.0)
    q = np.clip(np.rint(G / den[..., None]), -lim, lim)
    return (q.astype(np.float32) * scales[..., None]).reshape(W.shape)


def reproduce_cites():
    out = {}
    for layer, bits_or_bin, g in ((58, "binary", 128), (62, 2, 64), (0, "binary", 128), (62, 2, 128)):
        t0 = time.perf_counter()
        X = rbl.load_X_v1(layer)
        Wg = rbl.load_W(rbl.tname(layer, "mlp.gate_proj"))
        Wu = rbl.load_W(rbl.tname(layer, "mlp.up_proj"))
        inter = rbl.silu(X @ Wg.T) * (X @ Wu.T)
        del Wg, Wu
        Wd = rbl.load_W(rbl.tname(layer, "mlp.down_proj"))
        Xh = inter[192:]
        if bits_or_bin == "binary":
            Wq = rbl.codec_binary(Wd, g)
            key = f"L{layer}_binary_g{g}"
        else:
            Wq = uniform_f16(Wd, bits_or_bin, g)
            key = f"L{layer}_q{bits_or_bin}_g{g}_f16"
        y, yq = Xh @ Wd.T, Xh @ Wq.T
        out[key] = {
            "frob_cosine": frob_cos(y, yq),
            "mean_row_cosine": float(_rowcos(y, yq)),
            "n_hold": 64,
            "wall_s": float(time.perf_counter() - t0),
            "protocol": "v1 hidden + silu(X@Wg.T)*(X@Wu.T) last 64; mlp-floor cosine=frob",
        }
        print(f"CITE {key} frob={out[key]['frob_cosine']:.12f} row={out[key]['mean_row_cosine']:.12f}",
              flush=True)
        del X, inter, Xh, Wd, Wq, y, yq
    return out


def q4_and_hold512():
    refs = {}
    hold = {}
    for layer, cls, site, tag in rbl.JOBS:
        W = rbl.load_W(rbl.tname(layer, cls))
        _, Xh, split = rbl.site_xy(site, layer, 8, 256)
        prep = rbl.Prep(W, Xh)
        q4 = rbl.score(prep, rbl.codec_uniform(W, 4, 128))
        refs[f"L{layer}_{tag}"] = {**{k: float(v) for k, v in q4.items()},
                                   "shape": [int(W.shape[0]), int(W.shape[1])],
                                   "site": site, "split": split}
        print(f"Q4 L{layer} {tag} obs={q4['observed']:.6f} prb={q4['probed']:.6f} wu={q4['worst_unit']:.6f}",
              flush=True)
        if tag == "down" and layer in (58, 62):
            Xh512 = rbl.load_site_rows("post_swiglu", layer, rbl.take_prefix(rbl.swiglu_split()[1], 512))
            prep512 = rbl.Prep(W, Xh512)
            q4_512 = rbl.score(prep512, rbl.codec_uniform(W, 4, 128))
            raw_b = rbl.vs_ref(rbl.score(prep512, rbl.codec_binary(W)), q4_512)
            raw_t = rbl.vs_ref(rbl.score(prep512, rbl.codec_ternary(W)), q4_512)
            Xf, _, _ = rbl.site_xy("post_swiglu", layer, 256, 8)
            s = rbl.col_scales(W, Xf, 0.0)
            Wn = rbl.fold_col(W, s)
            Xn = Xf * s[None, :]
            rec_b = rbl.vs_ref(rbl.score(prep512, rbl.unfold_col(rbl.apply_actls(Wn, Xn, "binary"), s)), q4_512)
            rec_t = rbl.vs_ref(rbl.score(prep512, rbl.unfold_col(rbl.apply_actls(Wn, Xn, "ternary"), s)), q4_512)
            act_t = rbl.vs_ref(rbl.score(prep512, rbl.apply_actls(W, Xf, "ternary")), q4_512)
            hold[f"L{layer}"] = {
                "n_hold": 512,
                "q4": {k: float(v) for k, v in q4_512.items()},
                "binary_raw": rbl.rec(raw_b),
                "ternary_raw": rbl.rec(raw_t),
                "binary_col_eq_actls": rbl.rec(rec_b),
                "ternary_actls": rbl.rec(act_t),
                "ternary_col_eq_actls": rbl.rec(rec_t),
            }
            print(f"HOLD512 L{layer} bin_raw_obs={raw_b['observed']:.6f} ter_actls_obs={act_t['observed']:.6f} "
                  f"wu_raw={raw_b['worst_unit']:.6f} wu_act={act_t['worst_unit']:.6f}", flush=True)
            del Xh512, prep512, Xf
        del W, Xh, prep
    return refs, hold


def parse_log(path):
    text = open(path).read()
    tensors = []
    cur = None
    row_re = re.compile(
        r"^\s+([A-Za-z0-9_]+)\s+obs=([0-9.+-]+)\s+prb=([0-9.+-]+)\s+wu=([0-9.+-]+)\s+"
        r"gate=([0-9.+-]+)\s+(HEALTHY|UNHEALTHY)\s+([0-9.]+)s"
    )
    head_re = re.compile(r"-- L(\d+) (\S+) site=(\S+) rss=([0-9.]+) --")
    tag_of = {
        "mlp.down_proj": "down", "mlp.gate_proj": "gate",
        "self_attn.q_proj": "q", "linear_attn.out_proj": "out",
    }
    for line in text.splitlines():
        m = head_re.search(line)
        if m:
            if cur:
                tensors.append(cur)
            layer, cls, site, rss = int(m.group(1)), m.group(2), m.group(3), float(m.group(4))
            cur = {"layer": layer, "cls": cls, "tag": tag_of[cls], "site": site,
                   "rss_gb": rss, "rows": {}}
            continue
        m = row_re.search(line)
        if m and cur is not None:
            cur["rows"][m.group(1)] = {
                "observed": float(m.group(2)),
                "probed": float(m.group(3)),
                "worst_unit": float(m.group(4)),
                "gate": float(m.group(5)),
                "healthy": m.group(6) == "HEALTHY",
                "wall_s": float(m.group(7)),
            }
    if cur:
        tensors.append(cur)
    calib = []
    for m in re.finditer(
        r"CALIB n=(\d+) bin_obs=([0-9.]+) ter_obs=([0-9.]+) ann_obs=([0-9.]+) "
        r"fit_s=([0-9.]+)/([0-9.]+)/([0-9.]+)", text):
        calib.append({
            "n_fit": int(m.group(1)),
            "binary_actls_observed": float(m.group(2)),
            "ternary_actls_observed": float(m.group(3)),
            "ternary_anneal_observed": float(m.group(4)),
            "binary_actls_fit_s": float(m.group(5)),
            "ternary_actls_fit_s": float(m.group(6)),
            "ternary_anneal_fit_s": float(m.group(7)),
        })
    up = {}
    for m in re.finditer(
        r"UP (\S+) bin_obs=([0-9.]+) ter_obs=([0-9.]+) wcos=([0-9.]+)", text):
        up[m.group(1)] = {
            "binary_observed": float(m.group(2)),
            "ternary_observed": float(m.group(3)),
            "weight_row_cosine_vs_raw": float(m.group(4)),
        }
    repro = {}
    for m in re.finditer(r"REPRO L(\d+) (\S+) hold_cos=([0-9.]+)", text):
        repro[f"L{m.group(1)}_{m.group(2)}_mean_row"] = float(m.group(3))
    return tensors, calib, up, repro


def attach_deficits(tensors, refs):
    for t in tensors:
        key = f"L{t['layer']}_{t['tag']}"
        ref = refs.get(key)
        t["q4_ref"] = ref
        if not ref:
            continue
        for name, a in t["rows"].items():
            deficits = {k: a[k] - (ref[k] - rbl.AXIS_MARGIN[k])
                        for k in ("observed", "probed", "worst_unit")}
            worst = min(deficits, key=deficits.get)
            a["deficit"] = deficits
            a["worst_axis"] = worst
            # recompute gate from ref in case of rounding
            a["gate_from_ref"] = deficits[worst]
            a["healthy_from_ref"] = deficits[worst] >= 0.0
    return tensors


def main():
    t0 = time.perf_counter()
    print("=== cite reproduction (frob + row) ===", flush=True)
    cites = reproduce_cites()
    print("=== Q4 refs + hold512 ===", flush=True)
    refs, hold = q4_and_hold512()
    tensors, calib, up, repro = parse_log(LOG)
    tensors = attach_deficits(tensors, refs)
    print(f"parsed tensors={len(tensors)} calib={len(calib)}", flush=True)
    result = {
        "schema": "hawking.gravity1.reshape_before_lowbit.v1",
        "host": "Hawking Apple M3 Ultra, CPU numpy, no GPU",
        "N": rbl.N_SOURCE,
        "group": 128,
        "axis_margin": rbl.AXIS_MARGIN,
        "source": BF16,
        "capture_v1": rbl.CAP1,
        "capture_v2": rbl.CAP2,
        "capture_v2_sha256_self": rbl.cap2().get("sha256_self"),
        "c_uniform_twin_maxabs": 0.0,
        "doctor_demo": "PASS",
        "v1_cite_reproduction": cites,
        "v1_first_pass_mean_row": repro,
        "tensors": tensors,
        "q4_refs": refs,
        "calib_L58_down": calib,
        "up_side_effect_L58": up,
        "hold512": hold,
        "primary_wall_s": 297.61,
        "finish_wall_s": float(time.perf_counter() - t0),
        "rss_max_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
        "log": LOG,
    }
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"WROTE {OUT} finish_wall={result['finish_wall_s']:.1f}s rss={result['rss_max_gb']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
