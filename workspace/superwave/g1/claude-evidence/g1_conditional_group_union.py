#!/usr/bin/env python3
"""Union of per-token dropped Q4 groups at the 0.99-cosine operating point.

If the dropped groups are the same every token, a static mixed-bit layout
can omit them from the high-prec sidecar. If the dropped set rotates, the
union fills and resident BPW does not fall.
"""
from __future__ import annotations

import json
import os
import struct
import time

import numpy as np

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")

CAP = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
BF16 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
OUT = "/tmp/g1_conditional_group_union.json"
INTER, HIDDEN, N_TOK, G = 17408, 5120, 256, 64


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


class ST:
    def __init__(self, d):
        self.base = d
        self.wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}

    def load(self, name):
        shard = self.wm[name]
        if shard not in self.hdr:
            path = os.path.join(self.base, shard)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            self.hdr[shard] = (path, 8 + n, hdr)
        path, off, hdr = self.hdr[shard]
        rec = hdr[name]
        shape = tuple(rec["shape"])
        a, b = rec["data_offsets"]
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=off + a, shape=(b - a,))
        assert rec["dtype"] == "BF16"
        u16 = np.frombuffer(mm, dtype="<u2").reshape(shape).copy()
        out = np.empty(shape, np.float32)
        out.view(np.uint32)[...] = u16.astype(np.uint32) << 16
        return out


def cosine_rows(a, b):
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros(a.shape[0])
    m = den > 0
    out[m] = np.sum(a[m] * b[m], axis=1) / den[m]
    return out


st = ST(BF16)
result = {"layers": {}}
t0 = time.perf_counter()
for L in (0, 32, 63):
    x = np.fromfile(f"{CAP}/hidden/L{L:02d}.f32", dtype=np.float32).reshape(N_TOK, HIDDEN)
    Wg = st.load(f"language_model.model.layers.{L}.mlp.gate_proj.weight")
    Wu = st.load(f"language_model.model.layers.{L}.mlp.up_proj.weight")
    H = silu(x @ Wg.T) * (x @ Wu.T)
    del Wg, Wu
    Wd = st.load(f"language_model.model.layers.{L}.mlp.down_proj.weight")
    Y = H @ Wd.T
    e = (H.astype(np.float64) ** 2).reshape(N_TOK, INTER // G, G).sum(axis=2)
    n_g = e.shape[1]
    rec = {"n_g": n_g}
    for n_drop in (16, 32, 80, 144):
        n_keep = n_g - n_drop
        dropped = np.zeros(n_g, dtype=np.int32)
        union_drop = set()
        union_keep = set()
        growth = []
        for t in range(N_TOK):
            order = np.argsort(e[t])  # coldest first
            d = order[:n_drop]
            k = order[n_drop:]
            dropped[d] += 1
            union_drop.update(int(i) for i in d)
            union_keep.update(int(i) for i in k)
            growth.append(len(union_keep))
        # reconstruct with per-token keep
        Hm = np.zeros_like(H)
        for t in range(N_TOK):
            order = np.argsort(e[t])
            k = order[n_drop:]
            mask = np.zeros(n_g, dtype=bool)
            mask[k] = True
            m = np.repeat(mask, G)
            Hm[t, m] = H[t, m]
        Yk = Hm @ Wd.T
        c = cosine_rows(Yk, Y)
        # static: drop groups that are cold on mean energy
        em = e.mean(axis=0)
        static_drop = set(int(i) for i in np.argsort(em)[:n_drop])
        rec[str(n_drop)] = {
            "n_keep": n_keep,
            "keep_frac": n_keep / n_g,
            "dyn_mean_cos": float(c.mean()),
            "dyn_min_cos": float(c.min()),
            "union_keep": len(union_keep),
            "union_keep_frac": len(union_keep) / n_g,
            "union_drop": len(union_drop),
            "n_never_dropped": int((dropped == 0).sum()),
            "n_always_dropped": int((dropped == N_TOK).sum()),
            "n_dropped_half": int((dropped >= N_TOK // 2).sum()),
            "union_keep_growth": {"16": growth[15], "64": growth[63], "128": growth[127], "256": growth[255]},
            "static_drop_vs_never_dropped_overlap": len(static_drop & set(int(i) for i in np.where(dropped == 0)[0])),
        }
    # residual group union at drop 16 of 80
    eX = (x.astype(np.float64) ** 2).reshape(N_TOK, HIDDEN // G, G).sum(axis=2)
    n_gx = eX.shape[1]
    rec["residual"] = {}
    for n_drop in (8, 16, 32):
        dropped = np.zeros(n_gx, dtype=np.int32)
        uk = set()
        for t in range(N_TOK):
            order = np.argsort(eX[t])
            dropped[order[:n_drop]] += 1
            uk.update(int(i) for i in order[n_drop:])
        rec["residual"][str(n_drop)] = {
            "n_g": n_gx,
            "n_keep": n_gx - n_drop,
            "union_keep": len(uk),
            "n_never_dropped": int((dropped == 0).sum()),
            "n_always_dropped": int((dropped == N_TOK).sum()),
        }
    result["layers"][str(L)] = rec
    print(L, json.dumps(rec, indent=2))
    del H, Wd, Y

result["wall_s"] = time.perf_counter() - t0
json.dump(result, open(OUT, "w"))
print("WROTE", OUT, "wall", result["wall_s"])
