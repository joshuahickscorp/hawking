#!/usr/bin/env python3
"""Continuation: residual-proxy + student-forced MLP walk + packed organs.

Appends into /tmp/g1_screen_vs_generate.json. CPU only.
"""
from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path

import numpy as np

# reuse helpers
import importlib.util

spec = importlib.util.spec_from_file_location("s", "/tmp/g1_screen_vs_generate.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

ART = s.ART
G0 = s.G0
OUT = s.OUT
HIDDEN = s.HIDDEN
INTER = s.INTER
EOS = s.EOS
THINK = s.THINK


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def packed_organs_fixed() -> dict:
    hold = np.arange(192, 256)
    out = {}
    g0_man = json.loads((G0 / "manifest.json").read_text())
    s15_man = json.loads((ART / "mixed-sub15-v1/manifest.json").read_text())
    q3_rows = s.parse_catalog(ART / "mixed-q3mlp-v1")
    q4d_rows = s.parse_catalog(ART / "mixed-q4down-v1")
    p2_rows = s.parse_catalog(ART / "mixed-2p0-v1")

    for layer in (0, 62):
        X = s.load_hidden(layer)[hold]
        Wg = s.load_tensor(f"language_model.model.layers.{layer}.mlp.gate_proj.weight")
        Wu = s.load_tensor(f"language_model.model.layers.{layer}.mlp.up_proj.weight")
        Wd = s.load_tensor(f"language_model.model.layers.{layer}.mlp.down_proj.weight")
        Yg = X @ Wg.T
        Yu = X @ Wu.T
        act = s.silu(Yg) * Yu
        Yd = act @ Wd.T
        rec = {}
        for role, W, Yref, Xin in (
            ("gate_proj", Wg, Yg, X),
            ("up_proj", Wu, Yu, X),
            ("down_proj", Wd, Yd, act),
        ):
            name = f"language_model.model.layers.{layer}.mlp.{role}.weight"
            cell = {}
            g0_t = next(t for t in g0_man["tensors"] if t["name"] == name)
            g0_path = G0 / "tensors" / g0_t["artifact"]
            log(f"G0 decode L{layer} {role}")
            What = s.decode_hq30uq4_rows(g0_path, np.arange(W.shape[0]))
            Yh = Xin @ What.T
            cell["g0"] = dict(
                out_cos=s.mean_row_cosine(Yref, Yh),
                out_min=s.min_row_cosine(Yref, Yh),
                weight_cos=s.cosine(W, What),
            )
            r = s.catalog_row(q3_rows, name)
            log(f"q3mlp decode L{layer} {role}")
            What = s.decode_hgravu_rows(Path(r["path"]), r["off"], r["nbytes"], np.arange(W.shape[0]))
            Yh = Xin @ What.T
            cell["q3mlp"] = dict(
                out_cos=s.mean_row_cosine(Yref, Yh),
                out_min=s.min_row_cosine(Yref, Yh),
                weight_cos=s.cosine(W, What),
            )
            if role == "down_proj":
                r4 = s.catalog_row(q4d_rows, name)
                with open(r4["path"], "rb") as fh:
                    fh.seek(r4["off"])
                    mag = fh.read(8)
                cell["q4down_magic"] = mag.decode("latin1")
                if mag == b"HGRAVU01":
                    What = s.decode_hgravu_rows(Path(r4["path"]), r4["off"], r4["nbytes"], np.arange(W.shape[0]))
                    Yh = Xin @ What.T
                    cell["q4down"] = dict(
                        out_cos=s.mean_row_cosine(Yref, Yh),
                        out_min=s.min_row_cosine(Yref, Yh),
                        weight_cos=s.cosine(W, What),
                    )
            s15_t = next(t for t in s15_man["tensors"] if t["name"] == name)
            s15_path = ART / "mixed-sub15-v1/tensors" / s15_t["artifact"]
            same = s15_path.stat().st_ino == g0_path.stat().st_ino
            cell["sub15_same_file_as_g0"] = same
            if not same:
                log(f"sub15 decode L{layer} {role}")
                What = s.decode_hq30uq4_rows(s15_path, np.arange(W.shape[0]))
                Yh = Xin @ What.T
                cell["sub15"] = dict(
                    out_cos=s.mean_row_cosine(Yref, Yh),
                    out_min=s.min_row_cosine(Yref, Yh),
                    weight_cos=s.cosine(W, What),
                )
            rec[role] = cell
        # binary self-check L0 gate
        if layer == 0:
            Wb = s.binary_g128_recon(Wg)
            rec["binary_g128_L0_gate_weight_cos"] = s.cosine(Wg, Wb)
        out[f"L{layer}"] = rec
        del Wg, Wu, Wd
    return out


def residual_and_walk(ident: dict) -> dict:
    hold = np.arange(192, 256)
    sites = s.capture_sites()
    walk_sites = [x for x in sites if x["kind"] in ("generate_first_token", "after_think")][:4]
    walk_idx = [x["t"] for x in walk_sites]
    log(f"walk tokens {walk_idx} sites {[x['kind'] for x in walk_sites]}")

    recipes = {
        "g0_q4_hq30": dict(gate="hq4", up="hq4", down="hq4"),
        "q3mlp": dict(gate="u3", up="u3", down="u3"),
        "q4down_gatebin_upbin_downu4": dict(gate="bin", up="bin", down="u4"),
        "familyA_gatebin_upbin_downbin": dict(gate="bin", up="bin", down="bin"),
    }

    def recon(W, tag):
        return {
            "hq4": s.hq30uq4_recon,
            "u4": lambda w: s.hgravu_absmax_recon(w, 4),
            "u3": lambda w: s.hgravu_absmax_recon(w, 3),
            "bin": s.binary_g128_recon,
        }[tag](W)

    walk_err = {k: np.zeros((len(walk_idx), HIDDEN), dtype=np.float32) for k in recipes}
    walk_trace = {k: [] for k in recipes}
    per_layer = []

    for layer in range(64):
        t1 = time.time()
        X = s.load_hidden(layer)
        Xh = X[hold]
        Xw = X[walk_idx]
        Wg = s.load_tensor(f"language_model.model.layers.{layer}.mlp.gate_proj.weight")
        Wu = s.load_tensor(f"language_model.model.layers.{layer}.mlp.up_proj.weight")
        Wd = s.load_tensor(f"language_model.model.layers.{layer}.mlp.down_proj.weight")

        Yg = Xh @ Wg.T
        Yu = Xh @ Wu.T
        act = s.silu(Yg) * Yu
        Yd = act @ Wd.T
        res = Xh + Yd

        row = dict(
            layer=layer,
            yd_over_x=float(np.linalg.norm(Yd) / max(np.linalg.norm(Xh), 1e-12)),
            mean_token_yd_over_x=float(
                np.mean(np.linalg.norm(Yd, axis=1) / np.maximum(np.linalg.norm(Xh, axis=1), 1e-12))
            ),
        )

        specs_hold = {
            "g0_q4_hq30": ("hq4", "hq4", "hq4"),
            "q3mlp": ("u3", "u3", "u3"),
            "q4_hgravu": ("u4", "u4", "u4"),
            "q4down_bin_up": ("bin", "bin", "u4"),
            "familyA_bin": ("bin", "bin", "bin"),
        }
        for tag, (a, b, c) in specs_hold.items():
            Wg_h, Wu_h, Wd_h = recon(Wg, a), recon(Wu, b), recon(Wd, c)
            act_h = s.silu(Xh @ Wg_h.T) * (Xh @ Wu_h.T)
            Yd_h = act_h @ Wd_h.T
            row[tag] = dict(
                gate_out_cos=s.mean_row_cosine(Yg, Xh @ Wg_h.T),
                up_out_cos=s.mean_row_cosine(Yu, Xh @ Wu_h.T),
                down_out_cos=s.mean_row_cosine(Yd, Yd_h),
                residual_proxy_cos=s.mean_row_cosine(res, Xh + Yd_h),
                residual_proxy_min=s.min_row_cosine(res, Xh + Yd_h),
                down_rel_l2=s.rel_l2(Yd, Yd_h),
                residual_rel_l2=s.rel_l2(res, Xh + Yd_h),
            )

        Yg_w = Xw @ Wg.T
        Yu_w = Xw @ Wu.T
        act_w = s.silu(Yg_w) * Yu_w
        Yd_w = act_w @ Wd.T
        for rname, spec in recipes.items():
            Xs = Xw + walk_err[rname]
            Wg_h, Wu_h, Wd_h = recon(Wg, spec["gate"]), recon(Wu, spec["up"]), recon(Wd, spec["down"])
            act_s = s.silu(Xs @ Wg_h.T) * (Xs @ Wu_h.T)
            Yd_s = act_s @ Wd_h.T
            walk_err[rname] = walk_err[rname] + (Yd_s - Yd_w)
            walk_trace[rname].append(
                dict(
                    layer=layer,
                    hidden_cos=s.mean_row_cosine(Xw, Xw + walk_err[rname]),
                    per_token_cos=[
                        s.cosine(Xw[i], Xw[i] + walk_err[rname][i]) for i in range(len(walk_idx))
                    ],
                    err_rms=float(np.sqrt(np.mean(walk_err[rname] ** 2))),
                )
            )

        row["wall_s"] = time.time() - t1
        per_layer.append(row)
        if layer % 4 == 0 or layer in (54, 58, 62, 63):
            log(
                f"L{layer:02d} yd/x={row['mean_token_yd_over_x']:.3f} "
                f"q4r={row['g0_q4_hq30']['residual_proxy_cos']:.6f} "
                f"q3r={row['q3mlp']['residual_proxy_cos']:.6f} "
                f"binr={row['familyA_bin']['residual_proxy_cos']:.6f} "
                f"walk_q3={walk_trace['q3mlp'][-1]['hidden_cos']:.6f} "
                f"walk_g0={walk_trace['g0_q4_hq30'][-1]['hidden_cos']:.6f} "
                f"{row['wall_s']:.1f}s"
            )
        del Wg, Wu, Wd

    def prod_key(tag, field):
        xs = [row[tag][field] for row in per_layer]
        p = 1.0
        for x in xs:
            p *= x
        return dict(n=len(xs), min=min(xs), median=sorted(xs)[len(xs) // 2], max=max(xs), prod=p, geomean=p ** (1 / len(xs)))

    # final walked hidden @ lm_head (L63 post-norm + acc err, then model.norm)
    X63 = s.load_hidden(63)
    try:
        w_norm = s.load_tensor("language_model.model.norm.weight")
        norm_note = "rmsnorm(L63 + acc_err, model.norm.weight)"
    except Exception as e:
        w_norm = np.ones(HIDDEN, dtype=np.float32)
        norm_note = f"norm failed {e}"

    walk_logits = {}
    for rname in recipes:
        recs = []
        for i, site in enumerate(walk_sites):
            h = s.rmsnorm(X63[site["t"]] + walk_err[rname][i], w_norm)
            log(f"walk logits {rname} site={site['kind']} t={site['t']}")
            # G0 and HGRAVU lm_head are identical; use G0 (faster nibble unpack)
            logits = s.full_logits_for_hidden(h, "g0", ident)
            st = s.logit_stats(logits)
            st["hidden_cos_vs_bf16_site"] = s.cosine(X63[site["t"]], X63[site["t"]] + walk_err[rname][i])
            st["site"] = site
            recs.append(st)
        walk_logits[rname] = recs

    summary = dict(
        g0_residual_proxy=prod_key("g0_q4_hq30", "residual_proxy_cos"),
        q3_residual_proxy=prod_key("q3mlp", "residual_proxy_cos"),
        u4_residual_proxy=prod_key("q4_hgravu", "residual_proxy_cos"),
        q4down_residual_proxy=prod_key("q4down_bin_up", "residual_proxy_cos"),
        familyA_residual_proxy=prod_key("familyA_bin", "residual_proxy_cos"),
        g0_down_out=prod_key("g0_q4_hq30", "down_out_cos"),
        q3_down_out=prod_key("q3mlp", "down_out_cos"),
        walk_final={k: walk_trace[k][-1] for k in recipes},
        mean_yd_over_x=float(np.mean([r["mean_token_yd_over_x"] for r in per_layer])),
        min_yd_over_x=float(np.min([r["mean_token_yd_over_x"] for r in per_layer])),
        max_yd_over_x=float(np.max([r["mean_token_yd_over_x"] for r in per_layer])),
        norm_note=norm_note,
    )
    return dict(
        summary=summary,
        per_layer=per_layer,
        walk_trace=walk_trace,
        walk_logits=walk_logits,
        walk_sites=walk_sites,
    )


def main() -> None:
    t0 = time.time()
    result = json.loads(OUT.read_text())
    log("packed organs")
    result["packed_organs"] = packed_organs_fixed()
    OUT.write_text(json.dumps(result, indent=2))
    log("residual + walk")
    result["residual_walk"] = residual_and_walk(result["identity"])
    result["walk_wall_s"] = time.time() - t0
    OUT.write_text(json.dumps(result, indent=2))
    log(f"done {result['walk_wall_s']:.1f}s")


if __name__ == "__main__":
    main()
