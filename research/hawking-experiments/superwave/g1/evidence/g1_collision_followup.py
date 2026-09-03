#!/usr/bin/env python3
"""Follow-up: shared-r across classes; o_proj remeasured with fat sketch + r_down."""
from __future__ import annotations

import json
import os
import resource
import struct
import time

import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh

BF16 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
OUT = "/tmp/g1_mechanism_collision_followup.json"
RNG = np.random.default_rng(163)
N_COLS_FAT = 2048

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():.3f}G {msg}", flush=True)


def load_index():
    with open(os.path.join(BF16, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


_HDR = {}


def load_f32(weight_map, name):
    shard = weight_map[name]
    if shard not in _HDR:
        path = os.path.join(BF16, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        _HDR[shard] = (8 + n, header, path)
    base, header, path = _HDR[shard]
    info = header[name]
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + start)
        raw = f.read(end - start)
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"]).copy()


def is_gqa(l):
    return (l + 1) % 4 == 0


def I_of(W, r):
    return float(norm(W.T @ r) / (norm(W) + 1e-30))


def recover_r(weight_map, suffix, layers, n_cols):
    W0 = load_f32(weight_map, f"language_model.model.layers.{layers[0]}.{suffix}")
    m, n = W0.shape
    del W0
    idx = RNG.choice(n, size=min(n_cols, n), replace=False)
    idx.sort()
    cols = []
    for i, l in enumerate(layers):
        W = load_f32(weight_map, f"language_model.model.layers.{l}.{suffix}")
        cols.append(np.ascontiguousarray(W[:, idx]))
        del W
        if (i + 1) % 8 == 0 or i == 0:
            log(f"  recover {suffix} {i+1}/{len(layers)}")
    M = np.concatenate(cols, axis=1).astype(np.float32)
    log(f"  stacked {M.shape}")
    G = (M @ M.T).astype(np.float64)
    del M, cols
    w, v = eigh(G, subset_by_index=(0, 3), driver="evr")
    r = np.ascontiguousarray(v[:, 0])
    r /= norm(r)
    log(f"  eigs {w.tolist()}")
    return r.astype(np.float32), [float(x) for x in w], (m, n)


def main():
    t0 = time.time()
    wm = load_index()
    down_edit = list(range(24, 64))
    out_edit = [l for l in range(24, 64) if not is_gqa(l)]
    o_edit = [l for l in range(24, 64) if is_gqa(l)]
    o_ctrl = [l for l in range(0, 24) if is_gqa(l)]
    out_ctrl = [l for l in range(0, 24) if not is_gqa(l)]
    down_ctrl = list(range(0, 24))

    log("recover r_down")
    r_down, eigs_d, sh_d = recover_r(wm, "mlp.down_proj.weight", down_edit, 512)
    log("recover r_out")
    r_out, eigs_o, sh_o = recover_r(wm, "linear_attn.out_proj.weight", out_edit, 512)
    log("recover r_o fat")
    r_oproj, eigs_op, sh_op = recover_r(wm, "self_attn.o_proj.weight", o_edit, N_COLS_FAT)

    cos_do = float(abs(np.dot(r_down, r_out)))
    cos_dopp = float(abs(np.dot(r_down, r_oproj)))
    cos_oopp = float(abs(np.dot(r_out, r_oproj)))
    log(f"cos |down,out|={cos_do:.6f} |down,o|={cos_dopp:.6f} |out,o|={cos_oopp:.6f}")

    def table(suffix, layers, r, tag):
        rows = []
        for l in layers:
            W = load_f32(wm, f"language_model.model.layers.{l}.{suffix}")
            rows.append({"layer": int(l), "I": I_of(W, r), "edited": bool(l >= 24)})
            del W
        Ie = [x["I"] for x in rows if x["edited"]]
        Ic = [x["I"] for x in rows if not x["edited"]]
        log(f"  {tag} I_edit={np.mean(Ie):.6e} I_ctrl={np.mean(Ic):.6e} ratio={np.mean(Ic)/max(np.mean(Ie),1e-30):.2f}")
        return {
            "per_layer": rows,
            "I_edit_mean": float(np.mean(Ie)) if Ie else None,
            "I_ctrl_mean": float(np.mean(Ic)) if Ic else None,
            "ratio": float(np.mean(Ic) / max(np.mean(Ie), 1e-30)) if Ie and Ic else None,
        }

    # I of every class against r_down (the highest-power recovery)
    out = {
        "r_down_eigs": eigs_d,
        "r_out_eigs": eigs_o,
        "r_oproj_fat_eigs": eigs_op,
        "cos_abs": {"down_out": cos_do, "down_oproj": cos_dopp, "out_oproj": cos_oopp},
        "I_down_on_r_down": None,
        "I_out_on_r_down": None,
        "I_oproj_on_r_down": None,
        "I_oproj_on_r_oproj_fat": None,
        "elapsed_s": None,
        "rss_max_gb": None,
    }
    log("I down vs r_down")
    out["I_down_on_r_down"] = table("mlp.down_proj.weight", down_ctrl + down_edit, r_down, "down/r_down")
    log("I out vs r_down")
    out["I_out_on_r_down"] = table("linear_attn.out_proj.weight", out_ctrl + out_edit, r_down, "out/r_down")
    log("I o_proj vs r_down")
    out["I_oproj_on_r_down"] = table("self_attn.o_proj.weight", o_ctrl + o_edit, r_down, "o/r_down")
    log("I o_proj vs r_oproj_fat")
    out["I_oproj_on_r_oproj_fat"] = table("self_attn.o_proj.weight", o_ctrl + o_edit, r_oproj, "o/r_fat")

    # o_proj family-mean restoration under r_down
    log("o_proj mean under r_down")
    acc_all = None
    n_all = 0
    acc_e = None
    n_e = 0
    acc_c = None
    n_c = 0
    tensors = {}
    for l in o_ctrl + o_edit:
        W = load_f32(wm, f"language_model.model.layers.{l}.self_attn.o_proj.weight")
        tensors[l] = W
        if acc_all is None:
            acc_all = np.zeros_like(W, dtype=np.float64)
            acc_e = np.zeros_like(W, dtype=np.float64)
            acc_c = np.zeros_like(W, dtype=np.float64)
        acc_all += W
        n_all += 1
        if l >= 24:
            acc_e += W
            n_e += 1
        else:
            acc_c += W
            n_c += 1
    T_all = (acc_all / n_all).astype(np.float32)
    T_e = (acc_e / n_e).astype(np.float32)
    T_c = (acc_c / n_c).astype(np.float32)
    I_mean_all = [I_of(T_all, r_down) for _ in o_edit]
    I_hats = {
        "family_mean_all": float(np.mean([I_of(T_all, r_down) for _ in o_edit])),
        "family_mean_editonly": float(I_of(T_e, r_down)),
        "family_mean_ctrl": float(I_of(T_c, r_down)),
        "I_edit": out["I_oproj_on_r_down"]["I_edit_mean"],
        "I_ctrl": out["I_oproj_on_r_down"]["I_ctrl_mean"],
    }
    Ie, Ic = I_hats["I_edit"], I_hats["I_ctrl"]
    def rest(Ih):
        return float(np.clip((Ih - Ie) / (Ic - Ie + 1e-30), -0.5, 1.5))
    I_hats["restoration_mean_all"] = rest(I_hats["family_mean_all"])
    I_hats["restoration_mean_editonly"] = rest(I_hats["family_mean_editonly"])
    I_hats["restoration_mean_ctrl"] = rest(I_hats["family_mean_ctrl"])
    # per-layer tying scalar to T_all
    I_scalar = []
    for l in o_edit:
        W = tensors[l]
        s = float(np.vdot(W, T_all) / (np.vdot(T_all, T_all) + 1e-30))
        I_scalar.append(I_of((s * T_all).astype(np.float32), r_down))
    I_hats["tying_scalar_all_mean"] = float(np.mean(I_scalar))
    I_hats["restoration_tying_scalar_all"] = rest(I_hats["tying_scalar_all_mean"])
    out["o_proj_mechs_on_r_down"] = I_hats

    # save r_down for reuse
    np.save("/tmp/g1_r_down.npy", r_down)
    np.save("/tmp/g1_r_out.npy", r_out)
    np.save("/tmp/g1_r_oproj.npy", r_oproj)

    out["elapsed_s"] = time.time() - t0
    out["rss_max_gb"] = rss_gb()
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"WROTE {OUT} {out['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
