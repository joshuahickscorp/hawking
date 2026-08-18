#!/usr/bin/env python3
"""Unified G-ALLOC: heterogeneous representation kinds compete on one budget.

Currency per candidate: IR stored bytes (pool objects counted once), adequacy-gate
damage = 1 - min(observed, probed, worst_unit), native kernel name.

Descent: repeatedly take the single swap with the lowest
  Δ(damage * q_inject(layer)) / Δbytes
across every site and every kind, until each complete-BPW target is met or no
byte-saving swap remains.

Does not modify the repo. Writes /tmp/g1_alloc_unified_out.json.
"""
from __future__ import annotations

import argparse, json, math, os, resource, sys, time
import numpy as np

TOOLS = "/Users/scammermike/.claude-grok/worktrees/206-alloc-unified-20260817-181033/tools"
sys.path.insert(0, TOOLS)

import gravity_doctor_gate as dg
import gravity_allocator as ga
from gravity_ir import (
    Program, Node, quant_tensor, sparse_correction, exact_island, generated_block,
    SOURCE_PARAM_COUNT,
)

SRC = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
CAP = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
PIN = "workspace/superwave/g1/GRAVITY1_SOURCE_PIN.json"
G005 = "/tmp/g005-alloc.json"
OUT = "/tmp/g1_alloc_unified_out.json"

dg.BF16 = SRC
dg.CAPTURE = CAP
ga.BF16_ROOT = SRC

_orig_load_tensor = dg.load_tensor
_orig_load_X = dg.load_X

def load_tensor(name, root=SRC):
    return _orig_load_tensor(name, root=root)

def load_X(layer, capture=CAP):
    return _orig_load_X(layer, capture=capture)

dg.load_tensor = load_tensor
dg.load_X = load_X
ga.load_tensor = load_tensor
ga.load_X = load_X

GROUP = 128
HIDDEN = 5120
ISLAND_CH = (3994, 3456, 310)
RANK_KS = (1, 8, 32, 64, 128, 256)
PLANE_PS = (1, 2, 3)
SPARSE_FRACS = (1e-3, 1e-2)
TARGETS = (3.0, 2.0, 1.5, 1.0, 0.75)
SAMPLE_LAYERS = (0, 31, 32, 63)

# Native consumption path. EXISTS = landed Metal kernel that consumes the packed
# bytes directly. REQUIRED = named kernel the allocation specifies; not a
# pack-then-expand-to-Q4 path.
KERNEL = {
    "q2":   ("qwen_uniform_q2_group128_matvec_geo_tpr64_tg128", False),
    "q3":   ("qwen_uniform_q3_group128_matvec_geo_tpr64_tg128", False),
    "q4":   ("qwen_uniform_q4_group128_matvec_geo_tpr64_tg128", True),
    "q6":   ("qwen_uniform_q6_group128_matvec_geo_tpr64_tg128", False),
    "bin":  ("gk_matvec_binary", True),
    "rank": ("lowrank_gemv_f16", False),
    "sb":   ("fused_basis_gemv", False),
    "plane":("residual_plane_accum_gemv", False),
    "sparse":("sparse_correct_accum", False),
    "island":("static_island_saxpy", False),
}


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():.2f}G {msg}", flush=True)


# ---------------------------------------------------------------- constructors

def c_uniform_fast(W, bits, group=GROUP):
    if bits < 2:
        raise ValueError("uniform bits<2 is empty; use c_binary")
    lim = (1 << (bits - 1)) - 1
    m, d = W.shape
    usable = d - d % group
    Wh = np.empty_like(W)
    if usable:
        blk = np.ascontiguousarray(W[:, :usable]).reshape(m, -1, group)
        amax = np.max(np.abs(blk), axis=2, keepdims=True)
        step = np.maximum(amax, 1e-30) / lim
        q = np.clip(np.rint(blk / step), -lim, lim) * step
        Wh[:, :usable] = q.reshape(m, usable)
    if usable < d:
        Wh[:, usable:] = W[:, usable:]
    return Wh


def c_binary(W, group=GROUP):
    m, d = W.shape
    usable = d - d % group
    Wh = np.empty_like(W)
    if usable:
        blk = np.ascontiguousarray(W[:, :usable]).reshape(m, -1, group)
        scale = np.mean(np.abs(blk), axis=2, keepdims=True)
        scale = np.maximum(scale, 1e-30)
        signs = np.sign(blk)
        signs = np.where(signs == 0, 1.0, signs)
        Wh[:, :usable] = (signs * scale).reshape(m, usable)
    if usable < d:
        rest = W[:, usable:]
        scale = np.maximum(np.mean(np.abs(rest), axis=1, keepdims=True), 1e-30)
        signs = np.sign(rest)
        signs = np.where(signs == 0, 1.0, signs)
        Wh[:, usable:] = signs * scale
    return Wh


def c_planes(W, n_planes, group=GROUP):
    R = W
    acc = np.zeros_like(W)
    for _ in range(n_planes):
        B = c_binary(R, group)
        acc = acc + B
        R = W - acc
    return acc


def rsvd(W, k, rng, p=8):
    m, n = W.shape
    k = int(min(k, m, n))
    if k <= 0:
        raise ValueError("k")
    l = int(min(k + p, min(m, n)))
    Omega = rng.standard_normal((n, l)).astype(np.float32)
    Y = W @ Omega
    Q, _ = np.linalg.qr(Y, mode="reduced")
    B = Q.T @ W
    Uh, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = (Q @ Uh[:, :k]).astype(np.float32)
    return U, S[:k].astype(np.float32), Vt[:k].astype(np.float32)


def f16round(a):
    return a.astype(np.float16).astype(np.float32)


def c_rank_from_svd(U, S, Vt, k):
    return (f16round(U[:, :k]) * f16round(S[:k])) @ f16round(Vt[:k])


def c_sparse_on(W, base, frac):
    R = np.abs(W - base).ravel()
    k = max(1, int(frac * R.size))
    idx = np.argpartition(R, -k)[-k:]
    Wh = base.copy()
    Wh.ravel()[idx] = W.ravel()[idx]
    return Wh, int(k)


def apply_island(W, base):
    Wh = base.copy()
    rows = [r for r in ISLAND_CH if r < W.shape[0]]
    cols = [c for c in ISLAND_CH if c < W.shape[1]]
    n_exact = 0
    if rows:
        Wh[rows] = W[rows]
        n_exact += len(rows) * W.shape[1]
    if cols:
        Wh[:, cols] = W[:, cols]
        n_exact += len(cols) * W.shape[0]
        # overlap counted twice; subtract
        n_exact -= len(rows) * len(cols)
    return Wh, n_exact, rows, cols


def share_side(shape):
    r, c = shape
    if c == HIDDEN:
        return "right"
    if r == HIDDEN:
        return "left"
    return None


# ---------------------------------------------------------------- costing via IR

def q_bytes(elements, bits):
    return quant_tensor(elements, bits, GROUP, "x").stored_bytes


def binary_bytes(elements):
    return q_bytes(elements, 1)


def plane_bytes(elements, n_planes):
    one = binary_bytes(elements) - 40
    return n_planes * one + 40


def rank_bytes(m, n, k):
    return generated_block(m * n, code_bytes=k * (m + n) * 2 + 40,
                           generator_cid="none", kernel="x").stored_bytes


def sb_site_bytes(m, n, k, side):
    # per-site factor against a shared hidden-side basis
    if side == "right":
        code = m * k * 2
    else:
        code = k * n * 2
    return generated_block(m * n, code_bytes=code + 40,
                           generator_cid="none", kernel="x").stored_bytes


def sb_pool_bytes(k):
    return k * HIDDEN * 2


def sparse_term_bytes(n_exc, elements):
    ib = max(1, math.ceil(math.log2(max(elements, 2))))
    return sparse_correction(n_exc, 2, ib, "x").stored_bytes


def island_term_bytes(n_exact):
    return exact_island(n_exact, 2, "x", index_bits=0).stored_bytes


# ---------------------------------------------------------------- scoring

def gate_damage(W, Wh, X, seed=0):
    a = dg.axes(W, Wh, X, seed=seed)
    score = min(a["observed"], a["probed"], a["worst_unit"])
    return {
        "observed": float(a["observed"]),
        "probed": float(a["probed"]),
        "worst_unit": float(a["worst_unit"]),
        "score": float(score),
        "damage": float(max(0.0, 1.0 - score)),
    }


def make_X(W, layer, seed=0):
    probe_only = W.shape[1] != HIDDEN
    if probe_only:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((64, W.shape[1])).astype(np.float32)
    else:
        X = dg.load_X(layer)
    return X, probe_only


def cand(cid, kind, bits_like, kernel_key, exclusive, damage_row, meta,
         shared_key=None, shared_bytes=0, measured=True):
    kn, exists = KERNEL[kernel_key]
    return {
        "id": cid,
        "kind": kind,
        "kernel": kn,
        "kernel_exists": exists,
        "exclusive_bytes": int(exclusive),
        "shared_key": shared_key,
        "shared_bytes": int(shared_bytes),
        "damage": float(damage_row["damage"]),
        "score": float(damage_row["score"]),
        "observed": float(damage_row.get("observed", float("nan"))),
        "probed": float(damage_row.get("probed", float("nan"))),
        "worst_unit": float(damage_row.get("worst_unit", float("nan"))),
        "measured": measured,
        "meta": meta,
    }


# ---------------------------------------------------------------- measure one tensor

def measure_tensor(name, layer, cls, W, reuse_bits=None, rng=None, smoke=False):
    rng = rng or np.random.default_rng(0)
    X, probe_only = make_X(W, layer)
    m, n = W.shape
    e = int(W.size)
    side = share_side(W.shape)
    out = []
    factors = None  # (U, S, Vt) for sharing

    # bit widths: reuse g005 when present
    if reuse_bits:
        for b, row in reuse_bits.items():
            b = int(b)
            out.append(cand(f"q{b}_g128", "QuantTensor", b, f"q{b}",
                            q_bytes(e, b), row, {"bits": b, "group": GROUP}))
    else:
        for b in (2, 3, 4, 6):
            Wh = c_uniform_fast(W, b)
            row = gate_damage(W, Wh, X)
            del Wh
            out.append(cand(f"q{b}_g128", "QuantTensor", b, f"q{b}",
                            q_bytes(e, b), row, {"bits": b, "group": GROUP}))

    # 1-bit binary (c_uniform bits=1 is empty; this is the real 1-bit code)
    Wh = c_binary(W)
    row = gate_damage(W, Wh, X)
    del Wh
    out.append(cand("binary_g128", "QuantTensor", 1, "bin",
                    binary_bytes(e), row, {"bits": 1, "group": GROUP, "rule": "sign*meanabs"}))

    # residual planes
    plane_set = (1,) if smoke else PLANE_PS
    for p in plane_set:
        if p == 1:
            # plane-1 is binary; copy that score, still cost as 1 plane
            src = next(c for c in out if c["id"] == "binary_g128")
            out.append(cand("planes_p1", "Plane", 1, "plane",
                            plane_bytes(e, 1), src, {"n_planes": 1}, measured=True))
            continue
        Wh = c_planes(W, p)
        row = gate_damage(W, Wh, X)
        del Wh
        out.append(cand(f"planes_p{p}", "Plane", p, "plane",
                        plane_bytes(e, p), row, {"n_planes": p}))

    # ranks from one rsvd
    kmax = 32 if smoke else 256
    kmax = min(kmax, m, n)
    U, S, Vt = rsvd(W, kmax, rng)
    factors = (U, S, Vt, side)
    ks = (1, 8, 32) if smoke else RANK_KS
    for k in ks:
        if k > kmax:
            continue
        Wh = c_rank_from_svd(U, S, Vt, k)
        row = gate_damage(W, Wh, X)
        del Wh
        out.append(cand(f"rank_{k}", "Rank", k, "rank",
                        rank_bytes(m, n, k), row,
                        {"k": k, "m": m, "n": n, "energy": float(np.sum(S[:k] ** 2) / (np.sum(W * W) + 1e-30))}))

    # sparse exceptions on binary and on q2
    if not smoke:
        q2W = c_uniform_fast(W, 2)
        binW = c_binary(W)
        for base_name, base in (("q2", q2W), ("binary", binW)):
            base_bytes = q_bytes(e, 2) if base_name == "q2" else binary_bytes(e)
            for frac in SPARSE_FRACS:
                Wh, nex = c_sparse_on(W, base, frac)
                row = gate_damage(W, Wh, X)
                del Wh
                out.append(cand(f"{base_name}+sparse@{frac:g}", "SparseCorrection", frac, "sparse",
                                base_bytes + sparse_term_bytes(nex, e), row,
                                {"base": base_name, "frac": frac, "n_exceptions": nex,
                                 "index_bits": max(1, math.ceil(math.log2(max(e, 2))))}))
        # compile-time exact island on q2 and on binary
        for base_name, base in (("q2", q2W), ("binary", binW)):
            Wh, n_exact, rows, cols = apply_island(W, base)
            if n_exact <= 0:
                continue
            row = gate_damage(W, Wh, X)
            del Wh
            base_bytes = q_bytes(e, 2) if base_name == "q2" else binary_bytes(e)
            out.append(cand(f"{base_name}+island_ct3", "ExactIsland", 0, "island",
                            base_bytes + island_term_bytes(n_exact), row,
                            {"base": base_name, "channels": list(ISLAND_CH),
                             "rows": rows, "cols": cols, "n_exact": n_exact, "index_bits": 0}))
        del q2W, binW

    del X
    return out, factors, probe_only, (m, n, e)


def project_shared(W, V, side):
    V = f16round(V)
    if side == "right":
        C = f16round(W @ V.T)
        return C @ V
    C = f16round(V @ W)
    return V.T @ C


# ---------------------------------------------------------------- descent

def program_bytes(state, menus):
    excl = 0
    used = {}
    for name, cid in state.items():
        c = menus[name][cid]
        excl += c["exclusive_bytes"]
        if c["shared_key"]:
            used[c["shared_key"]] = c["shared_bytes"]
    return excl + sum(used.values())


def weighted_damage(state, menus, meta):
    s = 0.0
    for name, cid in state.items():
        layer = meta[name]["layer"]
        q = ga.q_inject(layer) if layer is not None else ga.q_inject(0)
        s += menus[name][cid]["damage"] * q
    return s


def cheapest_start(menu):
    # lowest damage, then highest bytes (richer) as the start of descent
    return min(menu.values(), key=lambda c: (c["damage"], -c["exclusive_bytes"]))["id"]


def descend(menus, meta, targets, kinds_ok=None):
    """Greedy cheapest swap. kinds_ok=None means every candidate."""
    def filt(menu):
        if kinds_ok is None:
            return menu
        return {k: v for k, v in menu.items() if v["kind"] in kinds_ok}

    fmenus = {n: filt(m) for n, m in menus.items()}
    fmenus = {n: m for n, m in fmenus.items() if m}
    state = {n: cheapest_start(m) for n, m in fmenus.items()}

    # refcounts for shared objects
    def rebuild_refs():
        refs = {}
        for n, cid in state.items():
            sk = fmenus[n][cid]["shared_key"]
            if sk:
                refs[sk] = refs.get(sk, 0) + 1
        return refs

    refs = rebuild_refs()

    def bytes_now():
        excl = 0
        shared = 0
        seen = set()
        for n, cid in state.items():
            c = fmenus[n][cid]
            excl += c["exclusive_bytes"]
            if c["shared_key"] and c["shared_key"] not in seen:
                shared += c["shared_bytes"]
                seen.add(c["shared_key"])
        return excl + shared

    def mix():
        by_kind, by_id = {}, {}
        elems_by_kind = {}
        for n, cid in state.items():
            c = fmenus[n][cid]
            e = meta[n]["elements"]
            by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
            by_id[cid] = by_id.get(cid, 0) + 1
            elems_by_kind[c["kind"]] = elems_by_kind.get(c["kind"], 0) + e
        return {
            "n_by_kind": dict(sorted(by_kind.items())),
            "n_by_id": dict(sorted(by_id.items(), key=lambda kv: -kv[1])),
            "elems_by_kind": dict(sorted(elems_by_kind.items(), key=lambda kv: -kv[1])),
        }

    def snap(tgt, achieved, floor):
        wd = weighted_damage(state, fmenus, meta)
        mx = mix()
        # per-class kind histogram
        class_kind = {}
        for n, cid in state.items():
            cls = meta[n]["cls"]
            k = fmenus[n][cid]["kind"]
            class_kind.setdefault(cls, {}).setdefault(k, 0)
            class_kind[cls][k] += 1
        kernels = {}
        for n, cid in state.items():
            c = fmenus[n][cid]
            kernels.setdefault(c["kernel"], {"n": 0, "exists": c["kernel_exists"]})
            kernels[c["kernel"]]["n"] += 1
        return {
            "target": tgt,
            "achieved_bpw": achieved,
            "reachable_bytes": not floor,
            "floor": floor,
            "total_bytes": bytes_now(),
            "weighted_damage": wd,
            **mx,
            "class_kind": class_kind,
            "kernels": kernels,
        }

    out = {}
    for tgt in sorted(targets, reverse=True):
        guard = 0
        while 8 * bytes_now() / SOURCE_PARAM_COUNT > tgt + 1e-15 and guard < 500000:
            guard += 1
            best = None
            b0 = bytes_now()
            for n, cid in state.items():
                cur = fmenus[n][cid]
                layer = meta[n]["layer"]
                q = ga.q_inject(layer) if layer is not None else ga.q_inject(0)
                d0 = cur["damage"] * q
                for alt_id, alt in fmenus[n].items():
                    if alt_id == cid:
                        continue
                    db_excl = cur["exclusive_bytes"] - alt["exclusive_bytes"]
                    db_sh = 0
                    if cur["shared_key"] != alt["shared_key"]:
                        if cur["shared_key"] and refs.get(cur["shared_key"], 0) == 1:
                            db_sh += cur["shared_bytes"]
                        if alt["shared_key"] and refs.get(alt["shared_key"], 0) == 0:
                            db_sh -= alt["shared_bytes"]
                    db = db_excl + db_sh  # bytes saved
                    if db <= 0:
                        continue
                    d1 = alt["damage"] * q
                    dd = d1 - d0
                    score = dd / db
                    if best is None or score < best[0] - 1e-18 or (
                        abs(score - best[0]) <= 1e-18 and db > best[4]
                    ):
                        best = (score, n, alt_id, dd, db)
            if best is None:
                break
            _, n, alt_id, _, _ = best
            old = fmenus[n][state[n]]
            new = fmenus[n][alt_id]
            if old["shared_key"]:
                refs[old["shared_key"]] -= 1
                if refs[old["shared_key"]] <= 0:
                    del refs[old["shared_key"]]
            if new["shared_key"]:
                refs[new["shared_key"]] = refs.get(new["shared_key"], 0) + 1
            state[n] = alt_id
        achieved = 8 * bytes_now() / SOURCE_PARAM_COUNT
        floor = achieved > tgt + 1e-12
        out[tgt] = snap(tgt, achieved, floor)
        out[tgt]["state_head"] = {n: state[n] for i, n in enumerate(state) if i < 8}
        out[tgt]["n_sites"] = len(state)
    return out, state


def build_ir_program(state, menus, meta, name):
    p = Program(name, source_pin=PIN)
    put = {}
    for n, cid in state.items():
        c = menus[n][cid]
        if c["shared_key"] and c["shared_key"] not in put:
            put[c["shared_key"]] = p.pool.put(
                "SharedBasis", nbytes=c["shared_bytes"],
                content_id=c["shared_key"], key=c["shared_key"])
    for n, cid in state.items():
        c = menus[n][cid]
        e = meta[n]["elements"]
        terms = []
        if c["kind"] == "QuantTensor":
            bits = int(c["meta"].get("bits", 2))
            terms.append(quant_tensor(e, bits, GROUP, c["kernel"]))
        elif c["kind"] == "Rank":
            terms.append(Node("GeneratedBlock", c["kernel"],
                              stored_bytes=c["exclusive_bytes"], elements=e,
                              meta=c["meta"]))
        elif c["kind"] == "SharedBasis":
            terms.append(generated_block(e, c["exclusive_bytes"], put[c["shared_key"]], c["kernel"]))
        elif c["kind"] == "Plane":
            terms.append(Node("Plane", c["kernel"], stored_bytes=c["exclusive_bytes"],
                              elements=e, meta=c["meta"]))
        elif c["kind"] == "SparseCorrection":
            # split approx: base + sparse. Reconstruct from meta.
            base = c["meta"]["base"]
            bb = q_bytes(e, 2) if base == "q2" else binary_bytes(e)
            terms.append(quant_tensor(e, 2 if base == "q2" else 1, GROUP, KERNEL["q2" if base == "q2" else "bin"][0]))
            terms.append(sparse_correction(c["meta"]["n_exceptions"], 2,
                                           c["meta"]["index_bits"], c["kernel"]))
            # force exclusive sum to match (header double-count is in the candidate)
            extra = c["exclusive_bytes"] - sum(t.stored_bytes for t in terms)
            if extra:
                terms[-1].stored_bytes += extra
        elif c["kind"] == "ExactIsland":
            base = c["meta"]["base"]
            terms.append(quant_tensor(e, 2 if base == "q2" else 1, GROUP, KERNEL["q2" if base == "q2" else "bin"][0]))
            terms.append(exact_island(c["meta"]["n_exact"], 2, c["kernel"], index_bits=0))
            extra = c["exclusive_bytes"] - sum(t.stored_bytes for t in terms)
            if extra:
                terms[-1].stored_bytes += extra
        else:
            terms.append(Node(c["kind"], c["kernel"], stored_bytes=c["exclusive_bytes"], elements=e))
        p.add(n, e, terms)
    return p


# ---------------------------------------------------------------- inventory + menus

def load_inv():
    inv = ga.inventory(SRC)
    return inv


def bit_only_floor(inv, curves):
    # reproduce ga.descend against g005 curves
    res = ga.descend(curves, [4.0, 3.0, 2.0, 1.5, 1.0, 0.75], [2, 3, 4, 6], inv=inv)
    return res


def attach_bits_from_g005(curves):
    by = {}
    for c in curves:
        by[(c["layer"], c["cls"])] = {int(b): v for b, v in c["curve"].items()}
    return by


def nearest_curve(layer, cls, sampled):
    layers = [L for (L, c) in sampled if c == cls]
    if not layers:
        return None
    near = min(layers, key=lambda L: abs(L - (layer if layer is not None else 0)))
    return sampled[(near, cls)], near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-measure", action="store_true",
                    help="reuse /tmp/g1_alloc_unified_measure.json if present")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    smoke = args.smoke

    log("inventory")
    inv = load_inv()
    gemv = sum(t["elements"] for t in inv)
    log(f"inventory {len(inv)} GEMV tensors, {gemv} elems ({100*gemv/SOURCE_PARAM_COUNT:.4f}% of N)")

    g005 = json.load(open(G005))
    bit_curves = []
    for c in g005["curves"]:
        cc = dict(c)
        cc["curve"] = {int(b): v for b, v in c["curve"].items()}
        bit_curves.append(cc)
    log("bit-only descend (reproduce 2.5065)")
    bit_res = bit_only_floor(inv, bit_curves)
    for t in sorted(bit_res, reverse=True):
        r = bit_res[t]
        log(f"  bit-only T={t:.2f} achieved={r['bpw']:.10f} wd={r['weighted_damage']:.6f} "
            f"held={r['held_elems']} bits={r['elems_by_bits']}")

    measure_path = "/tmp/g1_alloc_unified_measure.json"
    if args.skip_measure and os.path.exists(measure_path):
        meas = json.load(open(measure_path))
        log(f"loaded measurements {measure_path} n={len(meas['tensors'])}")
    else:
        reuse = attach_bits_from_g005(bit_curves)
        sample_layers = (0,) if smoke else SAMPLE_LAYERS
        tensors_to_do = []
        for t in inv:
            if t["layer"] in sample_layers and t["elements"] >= (1 if smoke else 1_000_000):
                tensors_to_do.append(t)
        if smoke:
            tensors_to_do = [x for x in tensors_to_do if x["cls"] == "mlp.down_proj"][:1]

        meas = {"tensors": [], "factors": {}, "wall_s_measure": None, "rss_max_gb": None}
        rng = np.random.default_rng(0)
        # first pass
        stored_factors = {}  # (layer, cls) -> dict
        for i, t in enumerate(tensors_to_do):
            log(f"measure {i+1}/{len(tensors_to_do)} {t['name']} e={t['elements']}")
            W = dg.load_tensor(t["name"])
            rb = reuse.get((t["layer"], t["cls"]))
            # L32 (and any unsampled) has no g005 bits
            cands, factors, probe_only, (m, n, e) = measure_tensor(
                t["name"], t["layer"], t["cls"], W, reuse_bits=rb, rng=rng, smoke=smoke)
            rec = {
                "name": t["name"], "layer": t["layer"], "cls": t["cls"],
                "elements": e, "shape": [m, n], "probe_only": probe_only,
                "candidates": cands,
            }
            meas["tensors"].append(rec)
            json.dump(meas, open(measure_path, "w"))
            if factors is not None:
                U, S, Vt, side = factors
                stored_factors[(t["layer"], t["cls"])] = {
                    "S": S, "Vt": Vt if side == "right" else None,
                    "U": U if side == "left" else None, "side": side,
                    "shape": (m, n),
                }
            del W
            # free U if left/right we keep the thin factor only
            if factors is not None:
                del U
            log(f"  {len(cands)} candidates; sample dmg "
                + ", ".join(f"{c['id']}={c['damage']:.4f}" for c in cands[:6]))

        # shared-basis second pass
        if not smoke:
            log("shared-basis pass")
            by_cls = {}
            for (L, cls), fac in stored_factors.items():
                by_cls.setdefault(cls, []).append((L, fac))
            shared_V = {}  # (cls, k) -> (V, side)
            for cls, items in by_cls.items():
                sides = {fac["side"] for _, fac in items}
                if None in sides or len(sides) != 1:
                    log(f"  skip shared {cls}: sides={sides}")
                    continue
                side = next(iter(sides))
                parts = []
                for L, fac in items:
                    if side == "right" and fac["Vt"] is not None:
                        parts.append(fac["S"][:, None] * fac["Vt"])
                    elif side == "left" and fac["U"] is not None:
                        # U is (m, k) with m=5120; take U.T * S
                        parts.append(fac["S"][:, None] * fac["U"].T)
                if not parts:
                    continue
                M = np.concatenate(parts, axis=0).astype(np.float32)  # (nL*k, 5120)
                rng2 = np.random.default_rng(1)
                # thin SVD of M to get shared hidden frame
                kmax = min(256, M.shape[0], M.shape[1])
                # M is small-tall: nL*256 x 5120. SVD is cheap.
                _, _, Vh = np.linalg.svd(M, full_matrices=False)
                V = Vh[:kmax].astype(np.float32)
                shared_V[cls] = (V, side)
                log(f"  shared {cls} n_layers={len(items)} V={V.shape} side={side}")

            # score shared membership
            for rec in meas["tensors"]:
                cls = rec["cls"]
                if cls not in shared_V:
                    continue
                V, side = shared_V[cls]
                W = dg.load_tensor(rec["name"])
                X, _ = make_X(W, rec["layer"])
                m, n = W.shape
                e = rec["elements"]
                for k in RANK_KS:
                    if k > V.shape[0]:
                        continue
                    Wh = project_shared(W, V[:k], side)
                    row = gate_damage(W, Wh, X)
                    del Wh
                    rec["candidates"].append(cand(
                        f"shared_k{k}", "SharedBasis", k, "sb",
                        sb_site_bytes(m, n, k, side), row,
                        {"k": k, "side": side, "cls": cls},
                        shared_key=f"sb:{cls}:k{k}",
                        shared_bytes=sb_pool_bytes(k),
                    ))
                del W, X
                log(f"  shared scored {rec['name']}")

        # optional embed / lm_head probe-only
        if not smoke and not args.no_embed:
            for t in inv:
                if t["layer"] is not None:
                    continue
                if t["elements"] < 1_000_000_000:
                    continue
                log(f"measure endpoint {t['name']} e={t['elements']} (batched, probe-only)")
                try:
                    rec = measure_endpoint(t)
                    meas["tensors"].append(rec)
                except Exception as ex:
                    log(f"  SKIP endpoint {t['name']}: {ex}")

        meas["wall_s_measure"] = time.time() - t0
        meas["rss_max_gb"] = rss_gb()
        json.dump(meas, open(measure_path, "w"))
        log(f"wrote {measure_path} ({os.path.getsize(measure_path)} B)")

    # build per-site menus
    log("build menus")
    sampled = {}
    for rec in meas["tensors"]:
        sampled[(rec["layer"], rec["cls"])] = rec

    menus, meta = {}, {}
    n_proxy = 0
    for t in inv:
        rec, src_layer = None, t["layer"]
        key = (t["layer"], t["cls"])
        if key in sampled:
            rec = sampled[key]
            measured = True
        else:
            hit = nearest_curve(t["layer"], t["cls"], sampled)
            if hit is None:
                # held: q6 only, no invented damage — matches gravity_allocator
                continue
            rec, src_layer = hit
            measured = False
            n_proxy += 1
        menu = {}
        scale = t["elements"] / rec["elements"]
        for c in rec["candidates"]:
            cc = dict(c)
            if not measured:
                # re-cost exclusive bytes for this site's element count
                cc = rescale_candidate(c, rec, t)
                cc["measured"] = False
                cc["meta"] = {**cc.get("meta", {}), "src_layer": src_layer, "proxy": True}
            menu[cc["id"]] = cc
        menus[t["name"]] = menu
        meta[t["name"]] = {
            "layer": t["layer"], "cls": t["cls"], "elements": t["elements"],
            "src_layer": src_layer, "measured": measured,
        }

    held = [t for t in inv if t["name"] not in menus]
    held_elems = sum(t["elements"] for t in held)
    # hold leftover 2-D sites at q6 (same policy as gravity_allocator)
    for t in held:
        menus[t["name"]] = {
            "q6_g128": cand("q6_g128", "QuantTensor", 6, "q6",
                            q_bytes(t["elements"], 6),
                            {"damage": 0.0, "score": 1.0, "observed": 1.0,
                             "probed": 1.0, "worst_unit": 1.0},
                            {"bits": 6, "held": True, "reason": "no class curve"},
                            measured=False),
        }
        meta[t["name"]] = {
            "layer": t["layer"], "cls": t["cls"], "elements": t["elements"],
            "src_layer": None, "measured": False, "held": True,
        }

    log(f"menus {len(menus)} sites; proxy {n_proxy}; held-at-q6 {len(held)} elems {held_elems}")

    # unconstrained-held: also offer q2/q4 on held with pessimistic proxy damage
    menus_uncon = {n: dict(m) for n, m in menus.items()}
    if held:
        # worst measured damage at each bit width
        worst = {}
        for rec in meas["tensors"]:
            for c in rec["candidates"]:
                if c["kind"] == "QuantTensor" and "bits" in c["meta"]:
                    b = c["meta"]["bits"]
                    worst[b] = max(worst.get(b, 0.0), c["damage"])
        for t in held:
            for b, kid in ((2, "q2"), (3, "q3"), (4, "q4")):
                if b not in worst:
                    continue
                menus_uncon[t["name"]][f"q{b}_g128"] = cand(
                    f"q{b}_g128", "QuantTensor", b, kid, q_bytes(t["elements"], b),
                    {"damage": worst[b], "score": 1.0 - worst[b],
                     "observed": float("nan"), "probed": float("nan"), "worst_unit": float("nan")},
                    {"bits": b, "held": True, "proxy": "worst_measured_same_bits"},
                    measured=False,
                )

    log("descend bit-only kinds=QuantTensor")
    bit_uni, _ = descend(menus, meta, TARGETS, kinds_ok={"QuantTensor"})
    log("descend unified all kinds")
    uni, state_uni = descend(menus, meta, TARGETS, kinds_ok=None)
    log("descend unified unconstrained-held")
    uni_u, _ = descend(menus_uncon, meta, TARGETS, kinds_ok=None)

    for label, table in (("bit-kinds", bit_uni), ("unified", uni), ("unified-uncon-held", uni_u)):
        log(f"== {label} ==")
        for tgt in TARGETS:
            r = table[tgt]
            kinds = " ".join(f"{k}:{e}" for k, e in r["elems_by_kind"].items())
            log(f"  T={tgt:.2f} ach={r['achieved_bpw']:.6f} floor={r['floor']} "
                f"wd={r['weighted_damage']:.5f} {kinds}")

    # IR cost of unified states at each target (rebuild by re-descending... we have
    # snapshots without full state). Re-run and keep states.
    log("IR programs at each target")
    ir_reports = {}
    # re-descend collecting state per target
    ir_reports = ir_at_targets(menus, meta, TARGETS)

    # theoretical floors
    floors = theoretical_floors(inv, menus, meta)

    out = {
        "schema": "hawking.gravity1.alloc_unified.v1",
        "N": SOURCE_PARAM_COUNT,
        "source": SRC,
        "pin": PIN,
        "q_inject": {str(k): v for k, v in ga.Q_INJECT.items()},
        "q_inject_spread": ga.q_inject(63) / ga.q_inject(0),
        "inventory": {"n_gemv": len(inv), "elems": gemv,
                      "frac_of_N": gemv / SOURCE_PARAM_COUNT},
        "bit_only_reproduce": {str(k): v for k, v in bit_res.items()},
        "sample_layers": list(SAMPLE_LAYERS if not smoke else (0,)),
        "n_measured_tensors": len(meas["tensors"]),
        "measured": [
            {"name": r["name"], "layer": r["layer"], "cls": r["cls"],
             "shape": r["shape"], "probe_only": r["probe_only"],
             "candidates": [
                 {k: c[k] for k in (
                     "id", "kind", "kernel", "kernel_exists", "exclusive_bytes",
                     "shared_key", "shared_bytes", "damage", "score", "observed",
                     "probed", "worst_unit", "measured", "meta")}
                 for c in r["candidates"]
             ]}
            for r in meas["tensors"]
        ],
        "n_sites": len(menus),
        "n_proxy": n_proxy,
        "held_at_q6": {"n": len(held), "elems": held_elems,
                       "names": [t["name"] for t in held[:8]],
                       "classes": sorted({t["cls"] for t in held})},
        "alloc_bit_kinds": {str(k): v for k, v in bit_uni.items()},
        "alloc_unified": {str(k): v for k, v in uni.items()},
        "alloc_unified_uncon_held": {str(k): v for k, v in uni_u.items()},
        "ir": ir_reports,
        "floors": floors,
        "wall_s": time.time() - t0,
        "rss_max_gb": rss_gb(),
        "measure_wall_s": meas.get("wall_s_measure"),
    }
    json.dump(out, open(OUT, "w"))
    log(f"wrote {OUT} ({os.path.getsize(OUT)} B) wall={out['wall_s']:.1f}s")
    return 0


def rescale_candidate(c, rec, t):
    """Recompute exclusive bytes for a same-class site with (usually) equal shape."""
    cc = dict(c)
    e = t["elements"]
    m_src, n_src = rec["shape"]
    # same class => same shape on this model
    if e == rec["elements"]:
        return cc
    # should not happen for this architecture; scale linearly just in case
    ratio = e / rec["elements"]
    cc["exclusive_bytes"] = int(round(c["exclusive_bytes"] * ratio))
    return cc


def ir_at_targets(menus, meta, targets):
    """Re-run unified descent, snapshot IR.report at each crossing."""
    reports = {}
    kinds_ok = None
    def filt(menu):
        return menu
    fmenus = {n: filt(m) for n, m in menus.items()}
    state = {n: cheapest_start(m) for n, m in fmenus.items()}
    refs = {}
    for n, cid in state.items():
        sk = fmenus[n][cid]["shared_key"]
        if sk:
            refs[sk] = refs.get(sk, 0) + 1

    def bytes_now():
        excl = 0
        seen = set()
        sh = 0
        for n, cid in state.items():
            c = fmenus[n][cid]
            excl += c["exclusive_bytes"]
            if c["shared_key"] and c["shared_key"] not in seen:
                sh += c["shared_bytes"]
                seen.add(c["shared_key"])
        return excl + sh

    remaining = list(sorted(targets, reverse=True))
    guard = 0
    while remaining and guard < 500000:
        guard += 1
        tgt = remaining[0]
        if 8 * bytes_now() / SOURCE_PARAM_COUNT <= tgt + 1e-15:
            p = build_ir_program(state, fmenus, meta, f"unified@{tgt}")
            reports[str(tgt)] = p.report()
            remaining.pop(0)
            continue
        best = None
        for n, cid in state.items():
            cur = fmenus[n][cid]
            layer = meta[n]["layer"]
            q = ga.q_inject(layer) if layer is not None else ga.q_inject(0)
            d0 = cur["damage"] * q
            for alt_id, alt in fmenus[n].items():
                if alt_id == cid:
                    continue
                db_excl = cur["exclusive_bytes"] - alt["exclusive_bytes"]
                db_sh = 0
                if cur["shared_key"] != alt["shared_key"]:
                    if cur["shared_key"] and refs.get(cur["shared_key"], 0) == 1:
                        db_sh += cur["shared_bytes"]
                    if alt["shared_key"] and refs.get(alt["shared_key"], 0) == 0:
                        db_sh -= alt["shared_bytes"]
                db = db_excl + db_sh
                if db <= 0:
                    continue
                score = (alt["damage"] * q - d0) / db
                if best is None or score < best[0]:
                    best = (score, n, alt_id)
        if best is None:
            p = build_ir_program(state, fmenus, meta, f"unified@floor")
            for tgt in remaining:
                reports[str(tgt)] = {**p.report(), "floor": True}
            break
        _, n, alt_id = best
        old = fmenus[n][state[n]]
        new = fmenus[n][alt_id]
        if old["shared_key"]:
            refs[old["shared_key"]] -= 1
            if refs[old["shared_key"]] <= 0:
                del refs[old["shared_key"]]
        if new["shared_key"]:
            refs[new["shared_key"]] = refs.get(new["shared_key"], 0) + 1
        state[n] = alt_id
    return reports


def theoretical_floors(inv, menus, meta):
    """Cheapest-bytes assignment per kind family, ignoring damage."""
    def cheapest_of(pred):
        total = 0
        used = {}
        n = 0
        for name, menu in menus.items():
            opts = [c for c in menu.values() if pred(c)]
            if not opts:
                continue
            c = min(opts, key=lambda x: x["exclusive_bytes"] + (
                x["shared_bytes"] if x["shared_key"] and x["shared_key"] not in used else 0))
            total += c["exclusive_bytes"]
            if c["shared_key"]:
                used[c["shared_key"]] = c["shared_bytes"]
            n += 1
        total += sum(used.values())
        return {"sites": n, "bytes": total, "bpw": 8 * total / SOURCE_PARAM_COUNT,
                "shared_objects": len(used)}

    return {
        "all_q6": cheapest_of(lambda c: c["id"] == "q6_g128"),
        "all_q4": cheapest_of(lambda c: c["id"] == "q4_g128"),
        "all_q3": cheapest_of(lambda c: c["id"] == "q3_g128"),
        "all_q2": cheapest_of(lambda c: c["id"] == "q2_g128"),
        "all_binary": cheapest_of(lambda c: c["id"] == "binary_g128"),
        "all_rank1": cheapest_of(lambda c: c["id"] == "rank_1"),
        "all_rank64": cheapest_of(lambda c: c["id"] == "rank_64"),
        "all_rank256": cheapest_of(lambda c: c["id"] == "rank_256"),
        "all_shared_k64": cheapest_of(lambda c: c["id"] == "shared_k64"),
        "all_shared_k256": cheapest_of(lambda c: c["id"] == "shared_k256"),
        "cheapest_any": cheapest_of(lambda c: True),
    }


def measure_endpoint(t):
    """Probe-only scoring of embed/lm_head. Rank + bit widths, batched rows."""
    W = dg.load_tensor(t["name"])
    m, n = W.shape
    e = int(W.size)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, n)).astype(np.float32)
    cands = []

    def score_Wh(Wh):
        return gate_damage(W, Wh, X)

    # bits via row batches into a full Wh would double memory. Score in place
    # by writing a temporary batch reconstruction for axes using maps.
    # We still need Wh for axes() as written. Do bits with in-place overwrite
    # of a working copy, one bit width at a time.
    for b in (2, 3, 4, 6):
        log(f"  endpoint q{b}")
        Wh = c_uniform_fast(W, b)
        row = score_Wh(Wh)
        del Wh
        cands.append(cand(f"q{b}_g128", "QuantTensor", b, f"q{b}",
                          q_bytes(e, b), row, {"bits": b, "group": GROUP, "endpoint": True}))
    log("  endpoint binary")
    Wh = c_binary(W)
    row = score_Wh(Wh)
    del Wh
    cands.append(cand("binary_g128", "QuantTensor", 1, "bin",
                      binary_bytes(e), row, {"bits": 1, "endpoint": True}))

    log("  endpoint rsvd k=64")
    U, S, Vt = rsvd(W, 64, rng, p=8)
    for k in (1, 8, 32, 64):
        Wh = c_rank_from_svd(U, S, Vt, k)
        row = score_Wh(Wh)
        del Wh
        cands.append(cand(f"rank_{k}", "Rank", k, "rank",
                          rank_bytes(m, n, k), row,
                          {"k": k, "m": m, "n": n, "endpoint": True,
                           "energy": float(np.sum(S[:k] ** 2) / (np.sum(W * W) + 1e-30))}))
    del W, U, S, Vt, X
    return {
        "name": t["name"], "layer": t["layer"], "cls": t["cls"],
        "elements": e, "shape": [m, n], "probe_only": True,
        "candidates": cands,
    }


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
    sys.exit(main())
