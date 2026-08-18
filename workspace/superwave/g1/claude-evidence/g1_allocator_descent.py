#!/usr/bin/env python3
"""G1 reverse-pruning allocator. CPU only. Reads landed JSONs. No GPU, no pack."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

MLP_FLOOR = Path("/tmp/g1_mlp_floor.json")
HETERO = Path("/tmp/g1_hetero_alloc_out.json")
SCREEN = Path("/tmp/g1_screen_vs_generate.json")
DESCENT = Path("/tmp/QWEN38_BPW_DESCENT.json")
MSE = Path("/tmp/qwen38_mse_scale_rule.json")
OUT = Path("/tmp/g1_allocator_descent_out.json")

N = 26_895_998_464
N_LAYERS = 64
G0_PRODUCT = 0.4078534106896186
G0_COMPLETE_BPW = 4.252735126866492
A63 = 2.6039602756500244
FLAT_GEO = G0_PRODUCT ** (1.0 / 192.0)  # exact; contract's 0.99527 is the rounded form

# role write coefficients into the residual (PROXY, not a measured Jacobian)
ROLE_WRITE = {
    "gate_proj": 0.5,
    "up_proj": 0.5,
    "down_proj": 1.0,
    "dn.in_proj_qkvz": 0.5,
    "dn.in_proj_ba": 0.1,
    "dn.out_proj": 1.0,
    "gqa.q_proj": 0.5,
    "gqa.k_proj": 0.4,
    "gqa.v_proj": 0.4,
    "gqa.o_proj": 1.0,
    "embed": 0.3,
    "lm_head": 0.0,  # after last residual; does not write the stream
}

IS_GQA = lambda layer: (layer + 1) % 4 == 0


def uniform_qn_bytes(n: int, bits: int, rank: int = 2, group: int = 64) -> int:
    groups = (n + group - 1) // group
    header = 32 + 4 * rank
    return header + groups * 2 + groups * group * bits // 8


def binary_g128_bytes(n: int) -> int:
    groups = (n + 127) // 128
    return 261 + groups * 2 + (n + 7) // 8


def ternary_g128_bytes(n: int) -> int:
    groups = (n + 127) // 128
    return 261 + groups * 2 + groups * 2 + (n * 2 + 7) // 8


def rice_bytes(n: int) -> int:
    # descent L0 gate 1.287565523035386 BPW on 89128960
    return int(round(n * 1.287565523035386 / 8.0))


def f32v2_bytes(n: int) -> int:
    return 8 + n * 4


def lerp(x0, y0, x1, y1, x):
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def pareto(points):
    """points: list of dicts with bpw, hold, rel_l2, bytes, codec. Keep non-dominated by bpw vs hold."""
    pts = [p for p in points if p["hold"] is not None and p["bpw"] is not None]
    pts.sort(key=lambda p: (p["bpw"], -p["hold"]))
    kept = []
    best_hold = -1.0
    for p in pts:
        if p["hold"] > best_hold + 1e-15:
            kept.append(p)
            best_hold = p["hold"]
    return kept


def interp_on_path(path, bpw):
    """path sorted by increasing bpw. Return hold, rel_l2, bytes at this bpw (clamped)."""
    if bpw <= path[0]["bpw"]:
        return path[0]["hold"], path[0]["rel_l2"], path[0]["bytes"], path[0]["codec"], 1.0
    if bpw >= path[-1]["bpw"]:
        return path[-1]["hold"], path[-1]["rel_l2"], path[-1]["bytes"], path[-1]["codec"], 1.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if a["bpw"] <= bpw <= b["bpw"]:
            h = lerp(a["bpw"], a["hold"], b["bpw"], b["hold"], bpw)
            r = lerp(a["bpw"], a["rel_l2"], b["bpw"], b["rel_l2"], bpw)
            by = lerp(a["bpw"], a["bytes"], b["bpw"], b["bytes"], bpw)
            # mix label
            if abs(bpw - b["bpw"]) < abs(bpw - a["bpw"]):
                codec = b["codec"]
                frac_hi = (bpw - a["bpw"]) / (b["bpw"] - a["bpw"])
            else:
                codec = a["codec"]
                frac_hi = (bpw - a["bpw"]) / (b["bpw"] - a["bpw"])
            if 0.02 < frac_hi < 0.98:
                codec = f"mix({a['codec']}->{b['codec']},{frac_hi:.3f})"
            return h, r, by, codec, frac_hi
    return path[-1]["hold"], path[-1]["rel_l2"], path[-1]["bytes"], path[-1]["codec"], 1.0


def nearest_measured(table, layer, role, key_hold="hold"):
    """table: dict (layer, role) -> list of rungs. Interpolate across measured layers for same role."""
    layers = sorted({L for (L, r) in table if r == role})
    if not layers:
        return None
    if layer in layers:
        return table[(layer, role)]
    lo = max((L for L in layers if L <= layer), default=None)
    hi = min((L for L in layers if L >= layer), default=None)
    if lo is None:
        return table[(hi, role)]
    if hi is None:
        return table[(lo, role)]
    a = table[(lo, role)]
    b = table[(hi, role)]
    # match by nearest bpw
    out = []
    for pa in a:
        pb = min(b, key=lambda q: abs(q["bpw"] - pa["bpw"]))
        t = (layer - lo) / (hi - lo)
        out.append({
            "codec": pa["codec"] + "_interp",
            "bpw": (1 - t) * pa["bpw"] + t * pb["bpw"],
            "hold": (1 - t) * pa["hold"] + t * pb["hold"],
            "rel_l2": (1 - t) * pa["rel_l2"] + t * pb["rel_l2"],
            "bytes": None,
            "proxy": True,
        })
    return out


def main():
    mlp = json.loads(MLP_FLOOR.read_text())
    hetero = json.loads(HETERO.read_text())
    screen = json.loads(SCREEN.read_text())
    descent = json.loads(DESCENT.read_text())
    mse = json.loads(MSE.read_text())

    f32_bytes = hetero["f32_catalog_bytes"]
    rms = [L["rms"] for L in hetero["activation"]["layers"]]
    A = [p["mean_token_yd_over_x"] for p in screen["residual_walk"]["per_layer"]]
    assert abs(A[63] - A63) < 1e-12

    # remaining stream gain to L63
    G_stream = [rms[63] / rms[l] for l in range(N_LAYERS)]
    # remaining (1+A) product — loose Jacobian bound, reported, not used as primary
    rem_jac = [1.0] * N_LAYERS
    acc = 1.0
    for l in range(N_LAYERS - 2, -1, -1):
        acc *= (1.0 + A[l + 1])
        rem_jac[l] = acc

    # ----- MLP curves: 192 tensors, 64 layers, MEASURED -----
    mlp_organs = {}
    for org in mlp["organs"]:
        key = (org["layer"], org["role"])
        pts = []
        for c in org["candidates"]:
            # packed-2p0 hold is in-sample (n_fit=256). weight_rsvd is weight-space.
            # Both sit on a naive Pareto front and lie. Honest HGRAVS is act_thin.
            if c["codec"] in ("hgravs01_r160_b3_packed_2p0", "hgravs01_r160_b3_weight_rsvd"):
                continue
            h = c.get("hold_output_cosine")
            if h is None:
                continue
            r = c.get("hold_output_rel_l2")
            if r is None:
                r = math.sqrt(max(0.0, 2.0 * (1.0 - h)))
            pts.append({
                "codec": c["codec"],
                "bpw": c["physical_bpw"],
                "hold": h,
                "rel_l2": r,
                "bytes": c["payload_bytes"],
                "proxy": False,
            })
        mlp_organs[key] = pts

    # verify G0 / q3mlp / q4down / 2p0 products
    def mlp_product(role_codec):
        p = 1.0
        n = 0
        for (L, role), pts in mlp_organs.items():
            want = role_codec[role]
            hit = next(x for x in pts if x["codec"] == want)
            p *= hit["hold"]
            n += 1
        return n, p

    prod_q4 = mlp_product({r: "uniform_q4_g64" for r in ("gate_proj", "up_proj", "down_proj")})
    prod_q3 = mlp_product({r: "uniform_q3_g64" for r in ("gate_proj", "up_proj", "down_proj")})
    prod_q4down = mlp_product({"gate_proj": "binary_g128", "up_proj": "residual_rice_q1_rms_2pct", "down_proj": "uniform_q4_g64"})
    prod_2p0 = mlp_product({"gate_proj": "binary_g128", "up_proj": "residual_rice_q1_rms_2pct", "down_proj": "hgravs01_r160_b3_act_thin"})

    # ----- descent attn_in / attn_out (6 layers) -----
    desc_attn = {}
    for org in descent["organs"]:
        role = org["role"]
        if role not in ("attn_in", "attn_out"):
            continue
        pts = []
        for c in org["candidates"]:
            h = c.get("hold_output_cosine")
            wcos = c.get("weight_cosine")
            if h is None:
                h = wcos  # attn_out is weight-only; PROXY
            if h is None:
                continue
            r = c.get("hold_output_rel_l2")
            if r is None:
                r = c.get("weight_rel_l2")
            if r is None:
                r = math.sqrt(max(0.0, 2.0 * (1.0 - h)))
            pts.append({
                "codec": c["codec"],
                "bpw": c["physical_bpw"],
                "hold": h,
                "rel_l2": r,
                "bytes": c.get("payload_bytes"),
                "proxy": org["quality_space"] != "output",
            })
        desc_attn[(org["layer"], role)] = pts

    # ----- mse-scale Q2/Q3/Q4 output cosine, 34 layers -----
    mse_attn = defaultdict(list)
    mse_mlp = defaultdict(list)
    for row in mse["summary"]["rows"]:
        L = int(row["layer"])
        role = row["role"]
        bits = int(row["bits"])
        hold = row["absmax"]
        hold_mse = row["mse"]
        rel = math.sqrt(max(0.0, 2.0 * (1.0 - hold)))
        rec = {
            "codec": f"uniform_q{bits}_g64",
            "bpw": bits + 0.25,
            "hold": hold,
            "hold_mse": hold_mse,
            "rel_l2": rel,
            "bytes": None,
            "proxy": False,
            "source": "mse_absmax",
        }
        if role in ("gate", "up", "down"):
            mse_mlp[(L, {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[role])].append(rec)
        else:
            mse_attn[(L, role)].append(rec)

    def attn_curve(layer, pack_role):
        """Build an attention curve. Mix mse-scale (Q2/3/4) + descent (binary/ternary/rice)."""
        # map pack_role -> mse role / descent role
        if pack_role == "dn.in_proj_qkvz":
            mse_roles = ["in_proj_qkvz_fused", "in_proj_qkv"]
            desc_role = "attn_in"
        elif pack_role == "dn.out_proj":
            mse_roles = ["out_proj"]
            desc_role = "attn_out"
        elif pack_role == "dn.in_proj_ba":
            mse_roles = ["in_proj_a"]
            desc_role = "attn_in"
        elif pack_role == "gqa.q_proj":
            mse_roles = ["q"]
            desc_role = "attn_in"
        elif pack_role == "gqa.k_proj":
            mse_roles = ["k"]
            desc_role = "attn_in"
        elif pack_role == "gqa.v_proj":
            mse_roles = ["v"]
            desc_role = "attn_in"
        elif pack_role == "gqa.o_proj":
            mse_roles = ["o"]
            desc_role = "attn_out"
        else:
            raise ValueError(pack_role)

        pts = []
        # mse at this layer or interpolated
        mse_pts = None
        for mr in mse_roles:
            if (layer, mr) in mse_attn:
                mse_pts = mse_attn[(layer, mr)]
                break
        if mse_pts is None:
            # interpolate across layers for first matching role
            for mr in mse_roles:
                tab = {(L, r): mse_attn[(L, r)] for (L, r) in mse_attn if r == mr}
                got = nearest_measured(tab, layer, mr)
                if got:
                    mse_pts = got
                    break
        if mse_pts:
            pts.extend(mse_pts)

        # descent binary / ternary / rice at nearest measured layer
        dtab = {(L, r): desc_attn[(L, r)] for (L, r) in desc_attn if r == desc_role}
        dpts = nearest_measured(dtab, layer, desc_role) if dtab else None
        if dpts:
            for p in dpts:
                if p["codec"] in ("binary_g128", "ternary_t0.7_g128", "rice_q1_rms_2pct"):
                    q = dict(p)
                    q["proxy"] = True
                    pts.append(q)
        return pts

    # ----- build free GEMV tensors -----
    tensors = []

    def add_tensor(layer, role, n_elem, pts, screen_member, proxy, mixer):
        # fill bytes where missing
        filled = []
        for p in pts:
            q = dict(p)
            if q.get("bytes") is None:
                codec = q["codec"]
                if codec.startswith("uniform_q") or codec.startswith("mix"):
                    # extract bits if possible
                    bits = None
                    for b in range(2, 9):
                        if f"q{b}" in codec:
                            bits = b
                            break
                    if bits is None:
                        bits = int(round(q["bpw"] - 0.25))
                    q["bytes"] = uniform_qn_bytes(n_elem, max(2, min(8, bits)))
                elif "binary" in codec:
                    q["bytes"] = binary_g128_bytes(n_elem)
                elif "ternary" in codec:
                    q["bytes"] = ternary_g128_bytes(n_elem)
                elif "rice" in codec or "residual" in codec:
                    q["bytes"] = rice_bytes(n_elem)
                else:
                    q["bytes"] = int(round(n_elem * q["bpw"] / 8.0))
            filled.append(q)
        path = pareto(filled)
        if not path:
            return
        w_A = A[layer] * ROLE_WRITE.get(role, 0.5)
        w_AR = w_A * (rms[63] / rms[layer])
        tensors.append({
            "layer": layer,
            "role": role,
            "n": n_elem,
            "path": path,
            "screen": screen_member,
            "proxy": proxy or any(p.get("proxy") for p in path),
            "mixer": mixer,
            "w_A": w_A,
            "w_AR": w_AR,
            "w_flat": 1.0 if screen_member else 0.0,
            "bpw_min": path[0]["bpw"],
            "bpw_max": path[-1]["bpw"],
        })

    E_GATE = 17408 * 5120  # 89128960
    for L in range(N_LAYERS):
        for role in ("gate_proj", "up_proj", "down_proj"):
            add_tensor(L, role, E_GATE, mlp_organs[(L, role)], True, False, "mlp")

    # attention geometry
    for L in range(N_LAYERS):
        if IS_GQA(L):
            add_tensor(L, "gqa.q_proj", 12288 * 5120, attn_curve(L, "gqa.q_proj"), False, True, "gqa")
            add_tensor(L, "gqa.k_proj", 1024 * 5120, attn_curve(L, "gqa.k_proj"), False, True, "gqa")
            add_tensor(L, "gqa.v_proj", 1024 * 5120, attn_curve(L, "gqa.v_proj"), False, True, "gqa")
            add_tensor(L, "gqa.o_proj", 5120 * 6144, attn_curve(L, "gqa.o_proj"), False, True, "gqa")
        else:
            add_tensor(L, "dn.in_proj_qkvz", 16384 * 5120, attn_curve(L, "dn.in_proj_qkvz"), False, True, "dn")
            add_tensor(L, "dn.in_proj_ba", 96 * 5120, attn_curve(L, "dn.in_proj_ba"), False, True, "dn")
            add_tensor(L, "dn.out_proj", 5120 * 6144, attn_curve(L, "dn.out_proj"), False, True, "dn")

    # embed + lm_head: PROXY = mean MLP Qn hold at same bits (descent said weight-l2 stand-in)
    def table_curve(n_elem):
        # build from mean of all MLP holds at each uniform_qn / binary
        buckets = defaultdict(list)
        for pts in mlp_organs.values():
            for p in pts:
                if p["codec"].startswith("uniform_q") or p["codec"] in ("binary_g128", "residual_rice_q1_rms_2pct"):
                    buckets[p["codec"]].append(p)
        pts = []
        for codec, group in buckets.items():
            h = sum(p["hold"] for p in group) / len(group)
            r = sum(p["rel_l2"] for p in group) / len(group)
            bpw = sum(p["bpw"] for p in group) / len(group)
            if codec.startswith("uniform_q"):
                bits = int(codec.split("_")[1][1:])
                by = uniform_qn_bytes(n_elem, bits)
            elif codec == "binary_g128":
                by = binary_g128_bytes(n_elem)
            else:
                by = rice_bytes(n_elem)
            pts.append({"codec": codec + "_tableproxy", "bpw": bpw, "hold": h, "rel_l2": r, "bytes": by, "proxy": True})
        return pts

    E_TAB = 248320 * 5120
    add_tensor(0, "embed", E_TAB, table_curve(E_TAB), False, True, "table")
    add_tensor(63, "lm_head", E_TAB, table_curve(E_TAB), False, True, "table")

    # pin tables at Q4 by construction? No — allow descent but they start at Q4 max if path max is Q8 proxy.
    # Restrict embed/lm_head path to <= Q4 and >= Q3 (generate-adjacent). Still allow prune Q4->Q3.
    for t in tensors:
        if t["role"] in ("embed", "lm_head"):
            t["path"] = [p for p in t["path"] if 3.0 <= p["bpw"] <= 4.3]
            if not t["path"]:
                raise RuntimeError("table path empty")
            t["bpw_min"] = t["path"][0]["bpw"]
            t["bpw_max"] = t["path"][-1]["bpw"]

    print(f"tensors {len(tensors)} screen {sum(1 for t in tensors if t['screen'])} "
          f"proxy {sum(1 for t in tensors if t['proxy'])}")

    # ----- state -----
    # start generous: each tensor at path max (Q8 MLP, Q4 attn/tables)
    for t in tensors:
        t["bpw"] = t["bpw_max"]

    def eval_state(ts):
        bytes_gemv = 0.0
        logP = 0.0
        nP = 0
        eA = 0.0
        eAR = 0.0
        e_flat = 0.0
        holds = []
        details = []
        for t in ts:
            h, r, by, codec, frac = interp_on_path(t["path"], t["bpw"])
            t["_h"] = h
            t["_r"] = r
            t["_by"] = by
            t["_codec"] = codec
            bytes_gemv += by
            if t["screen"]:
                logP += math.log(max(h, 1e-15))
                nP += 1
                holds.append(h)
                e_flat += (1.0 - h)
            eA += t["w_A"] * r
            eAR += t["w_AR"] * r
        complete = 8.0 * (bytes_gemv + f32_bytes) / N
        P = math.exp(logP) if nP else float("nan")
        geo = math.exp(logP / nP) if nP else float("nan")
        return {
            "complete_bpw": complete,
            "gemv_bytes": bytes_gemv,
            "f32_bytes": f32_bytes,
            "product": P,
            "geo": geo,
            "n_screen": nP,
            "eA": eA,
            "eAR": eAR,
            "e_flat": e_flat,
            "min_hold": min(holds) if holds else None,
            "mean_hold": (sum(holds) / len(holds)) if holds else None,
        }

    # G0 reference: force MLP Q4, attn Q4, tables Q4
    def set_codec_prefix(ts, pred, want_substr):
        for t in ts:
            if not pred(t):
                continue
            hits = [p for p in t["path"] if want_substr in p["codec"]]
            if hits:
                # prefer exact uniform_q4
                exact = [p for p in hits if p["codec"].startswith(want_substr) or want_substr == p["codec"]]
                p = min((exact or hits), key=lambda q: abs(q["bpw"] - 4.25) if "q4" in want_substr else q["bpw"])
                t["bpw"] = p["bpw"]

    # snapshot G0-like
    for t in tensors:
        hits = [p for p in t["path"] if "uniform_q4" in p["codec"]]
        if hits:
            t["bpw"] = hits[-1]["bpw"] if isinstance(hits[-1]["bpw"], float) else hits[0]["bpw"]
            t["bpw"] = min(hits, key=lambda p: abs(p["bpw"] - 4.25))["bpw"]
        else:
            t["bpw"] = t["bpw_max"]
    g0_state = eval_state(tensors)
    g0_eA = g0_state["eA"]
    g0_eAR = g0_state["eAR"]
    print("G0-like", {k: g0_state[k] for k in ("complete_bpw", "product", "geo", "eA", "eAR", "min_hold")})

    # reset to generous (path max)
    for t in tensors:
        t["bpw"] = t["bpw_max"]
    generous = eval_state(tensors)
    print("generous", {k: generous[k] for k in ("complete_bpw", "product", "geo", "eA")})

    STEP = 0.0625  # 1/16 bit; descend, do not jump Q4->Q2
    TARGETS = [3.0, 2.5, 2.0, 1.5, 1.2]

    def prune(ts, mode, targets):
        """
        mode:
          'eA'   — cheapest functional bit = smallest Δ(w_A * rel_l2) / Δbits
          'eAR'  — same with w_AR
          'prod' — smallest |Δ log P| / Δbits (MLP screen only; attn/tables cut first)
        """
        for t in ts:
            t["bpw"] = t["bpw_max"]
        snaps = {}
        path_trace = []
        ev = eval_state(ts)
        path_trace.append({"bpw": ev["complete_bpw"], "product": ev["product"], "eA": ev["eA"], "eAR": ev["eAR"]})

        # precompute per-tensor bits = n * bpw
        guard = 0
        max_iters = 200000
        next_targets = list(targets)
        while ev["complete_bpw"] > min(targets) - 1e-6 and guard < max_iters:
            guard += 1
            best = None
            best_score = None
            for i, t in enumerate(ts):
                if t["bpw"] <= t["bpw_min"] + 1e-9:
                    continue
                new_bpw = max(t["bpw_min"], t["bpw"] - STEP)
                # do not skip over a measured rung by more than STEP (already)
                h0, r0, by0, _, _ = interp_on_path(t["path"], t["bpw"])
                h1, r1, by1, _, _ = interp_on_path(t["path"], new_bpw)
                d_bits = 8.0 * (by0 - by1)
                if d_bits <= 1e-6:
                    # force a move to next lower rung
                    lower = [p for p in t["path"] if p["bpw"] < t["bpw"] - 1e-9]
                    if not lower:
                        continue
                    nxt = max(lower, key=lambda p: p["bpw"])
                    new_bpw = nxt["bpw"]
                    h1, r1, by1 = nxt["hold"], nxt["rel_l2"], nxt["bytes"]
                    d_bits = 8.0 * (by0 - by1)
                    if d_bits <= 1e-6:
                        continue
                if mode == "prod":
                    if t["screen"]:
                        d_obj = -(math.log(max(h1, 1e-15)) - math.log(max(h0, 1e-15)))  # product loss
                    else:
                        d_obj = 0.0  # free vs the screen — will be cut first
                elif mode == "eA":
                    d_obj = t["w_A"] * (r1 - r0)
                elif mode == "eAR":
                    d_obj = t["w_AR"] * (r1 - r0)
                else:
                    raise ValueError(mode)
                score = d_obj / d_bits
                # prefer lower score (cheaper)
                if best_score is None or score < best_score - 1e-18 or (
                    abs(score - best_score) <= 1e-18 and t["n"] > ts[best[0]]["n"]
                ):
                    best_score = score
                    best = (i, new_bpw, d_obj, d_bits)
            if best is None:
                break
            i, new_bpw, d_obj, d_bits = best
            prev_bpw = ev["complete_bpw"]
            ts[i]["bpw"] = new_bpw
            ev = eval_state(ts)
            # snapshot crossings
            still = []
            for T in next_targets:
                if prev_bpw > T + 1e-12 and ev["complete_bpw"] <= T + 1e-12:
                    snaps[T] = snapshot(ts, ev, mode)
                else:
                    still.append(T)
            next_targets = still
            if guard % 400 == 0:
                path_trace.append({
                    "bpw": ev["complete_bpw"],
                    "product": ev["product"],
                    "eA": ev["eA"],
                    "eAR": ev["eAR"],
                    "min_hold": ev["min_hold"],
                })
            if not next_targets and ev["complete_bpw"] <= min(targets):
                break
        # if a target was never crossed (started below), ignore
        ev = eval_state(ts)
        path_trace.append({"bpw": ev["complete_bpw"], "product": ev["product"], "eA": ev["eA"], "eAR": ev["eAR"]})
        return snaps, path_trace, ev

    def snapshot(ts, ev, mode):
        hist = Counter()
        by_class = defaultdict(lambda: {"n": 0, "bits": 0.0, "elems": 0, "hold_sum": 0.0, "hold_min": 1.0})
        per_layer_down = []
        mlp_hist = Counter()
        attn_hist = Counter()
        mix_n = 0
        for t in ts:
            h, r, by, codec, frac = interp_on_path(t["path"], t["bpw"])
            # bucket codec
            bucket = codec
            if codec.startswith("mix"):
                mix_n += 1
                # name by nearest rung
                nearest = min(t["path"], key=lambda p: abs(p["bpw"] - t["bpw"]))
                bucket = "near:" + nearest["codec"]
            hist[bucket] += 1
            cls = t["mixer"]
            by_class[cls]["n"] += 1
            by_class[cls]["bits"] += t["n"] * t["bpw"]
            by_class[cls]["elems"] += t["n"]
            if t["screen"]:
                by_class[cls]["hold_sum"] += h
                by_class[cls]["hold_min"] = min(by_class[cls]["hold_min"], h)
                mlp_hist[bucket] += 1
            else:
                attn_hist[bucket] += 1
            if t["role"] == "down_proj":
                per_layer_down.append({
                    "layer": t["layer"],
                    "bpw": t["bpw"],
                    "hold": h,
                    "rel_l2": r,
                    "codec": codec,
                    "A": A[t["layer"]],
                    "w_A": t["w_A"],
                })
        class_bpw = {k: (v["bits"] / v["elems"] if v["elems"] else None) for k, v in by_class.items()}
        return {
            "mode": mode,
            "complete_bpw": ev["complete_bpw"],
            "product": ev["product"],
            "geo": ev["geo"],
            "clears_g0_bar": bool(ev["product"] >= G0_PRODUCT),
            "eA": ev["eA"],
            "eAR": ev["eAR"],
            "eA_vs_g0": ev["eA"] / g0_eA if g0_eA else None,
            "eAR_vs_g0": ev["eAR"] / g0_eAR if g0_eAR else None,
            "min_hold": ev["min_hold"],
            "mean_hold": ev["mean_hold"],
            "class_bpw": class_bpw,
            "codec_hist": dict(hist),
            "mlp_codec_hist": dict(mlp_hist),
            "attn_codec_hist": dict(attn_hist),
            "n_mixed": mix_n,
            "down": per_layer_down,
            "gemv_bytes": ev["gemv_bytes"],
        }

    results = {}
    for mode in ("eA", "prod"):
        print("PRUNE", mode)
        snaps, trace, final = prune(tensors, mode, TARGETS)
        results[mode] = {
            "snapshots": {str(T): snaps.get(T) for T in TARGETS},
            "trace_tail": trace[-8:],
            "final": {k: final[k] for k in ("complete_bpw", "product", "geo", "eA", "eAR", "min_hold")},
        }
        for T in TARGETS:
            s = snaps.get(T)
            if s:
                print(f"  T={T} bpw={s['complete_bpw']:.6f} P={s['product']:.6e} geo={s['geo']:.6f} "
                      f"clears={s['clears_g0_bar']} eA/g0={s['eA_vs_g0']:.3f} min={s['min_hold']:.4f} "
                      f"class={s['class_bpw']}")
            else:
                print(f"  T={T} NO SNAPSHOT")

    # ----- flat vs depth-weighted requirement bits -----
    # Flat MIN: each screen tensor cheapest codec with hold >= FLAT_GEO
    # Flat GEO: G0 all-Q4 already meets product (geo == FLAT_GEO)
    # Depth-weighted: minimize MLP+attn bits s.t. sum w_A * rel_l2 <= sum w_A * rel_l2(hold=FLAT_GEO)
    #   implemented as: start generous, prune by eA, stop when eA would exceed E_budget.

    def rel_of_hold(h):
        return math.sqrt(max(0.0, 2.0 * (1.0 - h)))

    # budget = weighted error if every screen tensor sat at hold=FLAT_GEO
    # plus attention at its G0 (Q4) error — we only redistribute the MLP (screen) budget? 
    # Contract asks about the 192-tensor flat requirement. Restrict this comparison to the 192.
    E_budget_A = 0.0
    E_budget_AR = 0.0
    flat_min_bits = 0.0
    flat_min_bytes = 0
    flat_min_hist = Counter()
    q4_bits = 0.0
    q4_bytes = 0
    n_need_q5 = 0
    n_q3_clears = 0
    n_q4_clears = 0
    for t in tensors:
        if not t["screen"]:
            continue
        q4 = min((p for p in t["path"] if "uniform_q4" in p["codec"]), key=lambda p: abs(p["bpw"] - 4.25))
        q4_bits += t["n"] * q4["bpw"]
        q4_bytes += q4["bytes"]
        # cheapest clearing FLAT_GEO
        ok = [p for p in t["path"] if p["hold"] >= FLAT_GEO]
        if ok:
            p = min(ok, key=lambda q: q["bpw"])
        else:
            p = t["path"][-1]
        flat_min_bits += t["n"] * p["bpw"]
        flat_min_bytes += p["bytes"]
        flat_min_hist[p["codec"]] += 1
        if p["codec"].startswith("uniform_q5"):
            n_need_q5 += 1
        if p["codec"].startswith("uniform_q3"):
            n_q3_clears += 1
        if p["codec"].startswith("uniform_q4"):
            n_q4_clears += 1
        e_tgt = rel_of_hold(FLAT_GEO)
        E_budget_A += t["w_A"] * e_tgt
        E_budget_AR += t["w_AR"] * e_tgt

    # depth-weighted: assign each screen tensor a hold by reverse-pruning ONLY the 192
    # subject to sum w_A * rel_l2 <= E_budget_A. Start at path max, prune eA, stop at budget.
    mlp_only = [t for t in tensors if t["screen"]]
    for t in mlp_only:
        t["bpw"] = t["bpw_max"]

    def mlp_eA():
        s = 0.0
        bits = 0.0
        by = 0.0
        for t in mlp_only:
            h, r, b, _, _ = interp_on_path(t["path"], t["bpw"])
            s += t["w_A"] * r
            bits += t["n"] * t["bpw"]
            by += b
        return s, bits, by

    e, bits, by = mlp_eA()
    guard = 0
    while e < E_budget_A and guard < 100000:
        guard += 1
        best = None
        best_score = None
        for i, t in enumerate(mlp_only):
            if t["bpw"] <= t["bpw_min"] + 1e-9:
                continue
            new_bpw = max(t["bpw_min"], t["bpw"] - STEP)
            h0, r0, by0, _, _ = interp_on_path(t["path"], t["bpw"])
            h1, r1, by1, _, _ = interp_on_path(t["path"], new_bpw)
            d_bits = 8.0 * (by0 - by1)
            if d_bits <= 1e-6:
                continue
            d_e = t["w_A"] * (r1 - r0)
            if e + d_e > E_budget_A + 1e-12:
                continue  # would exceed budget
            score = d_e / d_bits
            if best_score is None or score < best_score:
                best_score = score
                best = (i, new_bpw, d_e)
        if best is None:
            break
        mlp_only[best[0]]["bpw"] = best[1]
        e, bits, by = mlp_eA()

    dw_e, dw_bits, dw_bytes = mlp_eA()
    dw_hist = Counter()
    dw_holds = []
    dw_bpw_by_role = defaultdict(lambda: [0.0, 0])
    dw_bpw_by_layer = [0.0] * N_LAYERS
    dw_n_by_layer = [0] * N_LAYERS
    for t in mlp_only:
        h, r, b, codec, _ = interp_on_path(t["path"], t["bpw"])
        nearest = min(t["path"], key=lambda p: abs(p["bpw"] - t["bpw"]))
        dw_hist[nearest["codec"] if not codec.startswith("mix") else "near:" + nearest["codec"]] += 1
        dw_holds.append(h)
        dw_bpw_by_role[t["role"]][0] += t["bpw"]
        dw_bpw_by_role[t["role"]][1] += 1
        dw_bpw_by_layer[t["layer"]] += t["bpw"]
        dw_n_by_layer[t["layer"]] += 1
    dw_complete = 8.0 * (dw_bytes + 0) / N  # MLP-only contribution
    q4_complete = 8.0 * q4_bytes / N
    flatmin_complete = 8.0 * flat_min_bytes / N

    # per-layer mean bpw under depth-weighted requirement
    layer_req = []
    for L in range(N_LAYERS):
        layer_req.append({
            "layer": L,
            "A": A[L],
            "rms": rms[L],
            "G_stream": G_stream[L],
            "mean_bpw": dw_bpw_by_layer[L] / dw_n_by_layer[L] if dw_n_by_layer[L] else None,
        })

    print("FLAT MIN hist", dict(flat_min_hist), "need_q5", n_need_q5, "q4", n_q4_clears, "q3", n_q3_clears)
    print("FLAT MIN MLP complete-equivalent", flatmin_complete, "bits", flat_min_bits)
    print("Q4 MLP complete-equivalent", q4_complete)
    print("DW eA", dw_e, "budget", E_budget_A, "MLP complete-eq", dw_complete, "bits", dw_bits)
    print("DW hist", dict(dw_hist))
    print("delta bits DW-Q4", dw_bits - q4_bits, "delta complete", dw_complete - q4_complete)
    print("delta bits DW-flatmin", dw_bits - flat_min_bits)

    # ----- inverse at 1.5 -----
    # Use the eA allocation at 1.5 (depth-aware global). If missing, the prod one.
    snap15 = results["eA"]["snapshots"].get("1.5") or results["prod"]["snapshots"].get("1.5")
    # We need the actual holds. Re-run prune eA and capture holds at 1.5.
    # Already have snap15['down'] and product/geo. Reconstruct required α:
    # h'_i = 1 - α (1-h_i), choose α so product = G0_PRODUCT.
    # We don't have all 192 holds in the snapshot — only down. Recompute by pruning again
    # and storing them.

    for t in tensors:
        t["bpw"] = t["bpw_max"]
    snaps_eA, _, _ = prune(tensors, "eA", [1.5])
    holds_15 = []
    rel_15 = []
    bits_15 = []
    meta_15 = []
    for t in tensors:
        if not t["screen"]:
            continue
        h, r, by, codec, _ = interp_on_path(t["path"], t["bpw"])
        holds_15.append(h)
        rel_15.append(r)
        bits_15.append(t["bpw"])
        meta_15.append((t["layer"], t["role"], t["bpw"], h, r, codec, t["w_A"]))

    P15 = 1.0
    for h in holds_15:
        P15 *= h
    geo15 = math.exp(sum(math.log(h) for h in holds_15) / len(holds_15))

    def product_after_alpha(alpha):
        p = 1.0
        for h in holds_15:
            hp = 1.0 - alpha * (1.0 - h)
            if hp <= 1e-15:
                return 0.0
            if hp >= 1.0:
                hp = 1.0 - 1e-16
            p *= hp
        return p

    # bisection on alpha
    lo, hi = 0.0, 1.0
    # if alpha=1 (no improvement) product is P15; need G0. alpha<1 is improvement.
    # if even alpha=0 (perfect holds) product=1 >= G0.
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if product_after_alpha(mid) >= G0_PRODUCT:
            lo = mid  # can worsen more (larger alpha) and still meet? wait
            # we want the REQUIRED improvement: smallest lift, i.e. largest alpha
            # such that product still >= G0. If P15 < G0, we need alpha < 1.
        else:
            hi = mid
    # Actually: h' = 1 - α(1-h). α=1 → original. α=0 → hold=1. We need smallest
    # improvement, i.e. largest α ≤ 1 with product(α) ≥ G0.
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if product_after_alpha(mid) >= G0_PRODUCT:
            lo = mid
        else:
            hi = mid
    alpha_star = lo
    shrink = alpha_star  # remaining fraction of (1-hold)
    improve = 1.0 / alpha_star if alpha_star > 0 else float("inf")

    # per-region required hold at 1.5 bits
    required = []
    for (L, role, bpw, h, r, codec, w) in meta_15:
        hp = 1.0 - alpha_star * (1.0 - h)
        required.append({
            "layer": L,
            "role": role,
            "bpw": bpw,
            "measured_hold": h,
            "required_hold": hp,
            "measured_rel_l2": r,
            "required_rel_l2": math.sqrt(max(0.0, 2.0 * (1.0 - hp))),
            "A": A[L],
            "w_A": w,
            "codec": codec,
        })

    # also: depth-weighted required holds at the SAME bit assignment
    # i.e. what hold would make w_A * rel_l2 contribute equally, summing to g0_eA_mlp
    # skip — the α inverse is the mechanism spec.

    # role/layer summary of inverse
    by_role_gap = defaultdict(list)
    by_layer_gap = defaultdict(list)
    for rec in required:
        gap = rec["required_hold"] - rec["measured_hold"]
        by_role_gap[rec["role"]].append((gap, rec))
        by_layer_gap[rec["layer"]].append(gap)

    inv_role = {}
    for role, items in by_role_gap.items():
        gaps = [g for g, _ in items]
        inv_role[role] = {
            "n": len(items),
            "mean_measured_hold": sum(r["measured_hold"] for _, r in items) / len(items),
            "mean_required_hold": sum(r["required_hold"] for _, r in items) / len(items),
            "mean_gap": sum(gaps) / len(gaps),
            "max_gap": max(gaps),
            "worst": max(items, key=lambda x: x[0])[1],
        }

    print("P15", P15, "geo15", geo15, "alpha", alpha_star, "improve_factor", improve)
    print("inv_role", {k: {kk: vv for kk, vv in v.items() if kk != "worst"} for k, v in inv_role.items()})

    # ----- Q5/Q8 products for report -----
    extra_prods = {}
    for q in range(2, 9):
        p = 1.0
        hs = []
        for pts in mlp_organs.values():
            hit = next(x for x in pts if x["codec"] == f"uniform_q{q}_g64")
            p *= hit["hold"]
            hs.append(hit["hold"])
        extra_prods[f"q{q}"] = {"product": p, "geo": math.exp(sum(math.log(h) for h in hs) / len(hs)), "min": min(hs)}

    out = {
        "schema": "hawking.g1.allocator_descent.v1",
        "inputs": {
            "mlp_floor": str(MLP_FLOOR),
            "mlp_floor_schema": mlp["schema"],
            "mlp_floor_wall_s": mlp["wall_s"],
            "hetero": str(HETERO),
            "screen_walk": str(SCREEN),
            "descent_seal": descent.get("seal_sha256"),
            "G0_PRODUCT": G0_PRODUCT,
            "A63": A63,
            "FLAT_GEO_exact": FLAT_GEO,
            "FLAT_GEO_contract_approx": 0.99527,
        },
        "calibration": {
            "prod_q4": prod_q4[1],
            "prod_q3": prod_q3[1],
            "prod_q4down": prod_q4down[1],
            "prod_2p0_honest": prod_2p0[1],
            "prod_q4_minus_bar": prod_q4[1] - G0_PRODUCT,
            "n_mlp": prod_q4[0],
        },
        "amplification": {
            "A": A,
            "rms": rms,
            "G_stream": G_stream,
            "rem_jac": rem_jac,
            "mean_A": sum(A) / len(A),
            "min_A": min(A),
            "max_A": max(A),
            "argmax_A": int(max(range(64), key=lambda i: A[i])),
            "prod_1pA": math.prod(1.0 + a for a in A),
            "L63_over_L0_rms": rms[63] / rms[0],
            "L63_over_L62_rms": rms[63] / rms[62],
        },
        "g0_like": g0_state,
        "generous": generous,
        "prunes": results,
        "flat_vs_depth": {
            "FLAT_GEO": FLAT_GEO,
            "n_q4_below_geo": n_need_q5,  # assigned Q5
            "n_q4_clears": n_q4_clears,
            "n_q3_clears": n_q3_clears,
            "flat_min_hist": dict(flat_min_hist),
            "flat_min_mlp_bits": flat_min_bits,
            "flat_min_mlp_complete_eq": flatmin_complete,
            "q4_mlp_bits": q4_bits,
            "q4_mlp_complete_eq": q4_complete,
            "E_budget_A": E_budget_A,
            "E_budget_AR": E_budget_AR,
            "dw_eA": dw_e,
            "dw_mlp_bits": dw_bits,
            "dw_mlp_complete_eq": dw_complete,
            "dw_hist": dict(dw_hist),
            "delta_bits_dw_minus_q4": dw_bits - q4_bits,
            "delta_complete_dw_minus_q4": dw_complete - q4_complete,
            "delta_bits_dw_minus_flatmin": dw_bits - flat_min_bits,
            "delta_complete_dw_minus_flatmin": dw_complete - flatmin_complete,
            "layer_req": layer_req,
            "role_mean_bpw": {k: v[0] / v[1] for k, v in dw_bpw_by_role.items()},
        },
        "inverse_1p5": {
            "P15": P15,
            "geo15": geo15,
            "alpha_star": alpha_star,
            "one_minus_hold_must_shrink_to": alpha_star,
            "improve_factor_on_one_minus_hold": improve,
            "required_geo": FLAT_GEO,
            "measured_geo": geo15,
            "role": {k: {kk: (vv if kk != "worst" else {
                "layer": vv["layer"], "role": vv["role"], "measured": vv["measured_hold"],
                "required": vv["required_hold"], "bpw": vv["bpw"],
            }) for kk, vv in v.items()} for k, v in inv_role.items()},
            "worst10": sorted(required, key=lambda r: r["required_hold"] - r["measured_hold"], reverse=True)[:10],
            "late_down": [r for r in required if r["role"] == "down_proj" and r["layer"] in (54, 58, 59, 62, 63)],
        },
        "uniform_products": extra_prods,
        "accounting": mlp["accounting"],
        "n_tensors": len(tensors),
        "n_screen": sum(1 for t in tensors if t["screen"]),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
