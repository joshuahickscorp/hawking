#!/usr/bin/env python3
"""Follow-up: channel ablation, teacher-signed margins, sha256 variants. CPU only."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time

import numpy as np

CAPTURE = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
BF16 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
LM = os.path.join(BF16, "model-00011-of-00011.safetensors")
N, H, V = 256, 5120, 248320
CHUNK = 4096
OUT = "/tmp/g1_decision_margin_follow.json"


def bf16_to_f32(raw):
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_hidden(layer):
    x = np.fromfile(os.path.join(CAPTURE, "hidden", f"L{layer:02d}.f32"), dtype="<f4")
    return x.reshape(N, H).copy()


def parse_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def logits_topk(X, k=5, extra_ids=None):
    header, data0 = parse_header(LM)
    info = header["language_model.lm_head.weight"]
    start = info["data_offsets"][0]
    extra_ids = sorted(set(extra_ids or []))
    extra_z = {tid: np.empty(X.shape[0], np.float32) for tid in extra_ids}
    top_z = np.full((X.shape[0], k), -np.inf, np.float32)
    top_id = np.full((X.shape[0], k), -1, np.int32)
    with open(LM, "rb") as f:
        for row0 in range(0, V, CHUNK):
            row1 = min(V, row0 + CHUNK)
            f.seek(data0 + start + row0 * H * 2)
            W = np.ascontiguousarray(bf16_to_f32(f.read((row1 - row0) * H * 2)).reshape(row1 - row0, H))
            z = X @ W.T
            for tid in extra_ids:
                if row0 <= tid < row1:
                    extra_z[tid] = z[:, tid - row0].copy()
            for i in range(X.shape[0]):
                part = np.argpartition(z[i], -k)[-k:]
                vals = z[i, part]
                ids = part + row0
                all_v = np.concatenate([top_z[i], vals])
                all_i = np.concatenate([top_id[i], ids.astype(np.int32)])
                keep = np.argpartition(all_v, -k)[-k:]
                keep = keep[np.argsort(-all_v[keep])]
                top_z[i] = all_v[keep]
                top_id[i] = all_i[keep]
    return top_z, top_id, extra_z


def main():
    t0 = time.perf_counter()
    meta = json.load(open(os.path.join(CAPTURE, "capture-result.json")))
    ids = []
    pof = []
    for pi, p in enumerate(meta["prompts"]):
        for tok in p["ids"]:
            ids.append(int(tok))
            pof.append(pi)
    ids = np.asarray(ids)
    pof = np.asarray(pof)
    nxt = np.full(N, -1, np.int32)
    last = np.zeros(N, bool)
    for i in range(N):
        if i + 1 < N and pof[i + 1] == pof[i]:
            nxt[i] = ids[i + 1]
        else:
            last[i] = True

    X = load_hidden(63)
    extra = set(int(x) for x in nxt if x >= 0)
    extra |= {248044, 248045, 248046, 248058, 248059, 248066, 248067, 248068, 248069}
    print("base logits", flush=True)
    top_z, top_id, extra_z = logits_topk(X, extra_ids=extra)
    margin = top_z[:, 0] - top_z[:, 1]
    teacher_z = np.full(N, np.nan, np.float32)
    for i in range(N):
        if nxt[i] >= 0:
            teacher_z[i] = extra_z[int(nxt[i])][i]
    # teacher-signed margin: z_next - max_{j!=next} z_j
    tmargin = np.full(N, np.nan, np.float32)
    for i in range(N):
        if nxt[i] < 0:
            continue
        z_t = teacher_z[i]
        # best competitor among topk, excluding next
        comp = -np.inf
        for k in range(5):
            if top_id[i, k] != nxt[i]:
                comp = max(comp, float(top_z[i, k]))
        tmargin[i] = z_t - comp

    tf = nxt >= 0
    print("teacher-signed", float(np.nanmean(tmargin)), "neg_frac", float((tmargin[tf] < 0).mean()),
          "p50", float(np.nanmedian(tmargin)))

    # ablations on last tokens only (and all) by zeroing channels
    last_idx = np.where(last)[0]
    ablates = {
        "zero_3994": [3994],
        "zero_3456": [3456],
        "zero_310": [310],
        "zero_1089": [1089],
        "zero_island3": [3994, 3456, 310],
        "zero_last_top8": [3994, 1236, 3219, 3899, 998, 1089, 2652, 1149],
    }
    abl = {}
    base_last = [int(top_id[i, 0]) for i in last_idx]
    base_last_m = [float(margin[i]) for i in last_idx]
    for name, chs in ablates.items():
        X2 = X.copy()
        for c in chs:
            X2[:, c] = 0.0
        print("ablate", name, flush=True)
        tz, tid, _ = logits_topk(X2, extra_ids=set(base_last))
        pred = [int(tid[i, 0]) for i in last_idx]
        m2 = [float(tz[i, 0] - tz[i, 1]) for i in last_idx]
        flips_last = [int(a != b) for a, b in zip(pred, base_last)]
        flips_all = float((tid[:, 0] != top_id[:, 0]).mean())
        abl[name] = {
            "last_base": base_last,
            "last_pred": pred,
            "last_flip": flips_last,
            "last_flip_n": int(sum(flips_last)),
            "last_m_base": base_last_m,
            "last_m_abl": m2,
            "all_flip_rate": flips_all,
        }

    # sha256 variants
    files = [os.path.join(CAPTURE, "hidden", f"L{l:02d}.f32") for l in range(64)]
    variants = {}
    h = hashlib.sha256()
    for p in files:
        h.update(open(p, "rb").read())
    variants["concat_raw"] = h.hexdigest()
    h = hashlib.sha256()
    for p in files:
        h.update(hashlib.sha256(open(p, "rb").read()).digest())
    variants["concat_perfile_digest"] = h.hexdigest()
    h = hashlib.sha256()
    for p in files:
        h.update(os.path.basename(p).encode())
        h.update(open(p, "rb").read())
    variants["name_plus_raw"] = h.hexdigest()
    # json then payload
    h = hashlib.sha256()
    h.update(open(os.path.join(CAPTURE, "capture-result.json"), "rb").read())
    for p in files:
        h.update(open(p, "rb").read())
    variants["json_plus_concat"] = h.hexdigest()
    claimed = meta["sha256_self"]
    variants["claimed"] = claimed
    variants["any_match"] = any(v == claimed for k, v in variants.items() if k != "claimed")

    # L62 vs L63 cosine (are they almost the same residual?)
    X62 = load_hidden(62)
    c = np.sum(X * X62, 1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(X62, axis=1) + 1e-12)
    # apply final_norm to L63 and rescore last
    idx = json.load(open(os.path.join(BF16, "model.safetensors.index.json")))["weight_map"]

    def load_named(name):
        path = os.path.join(BF16, idx[name])
        header, data0 = parse_header(path)
        info = header[name]
        start, end = info["data_offsets"]
        with open(path, "rb") as f:
            f.seek(data0 + start)
            raw = f.read(end - start)
        if info["dtype"] == "BF16":
            arr = bf16_to_f32(raw)
        else:
            arr = np.frombuffer(raw, dtype="<f4").copy()
        return arr.reshape(info["shape"])

    fw = load_named("language_model.model.norm.weight").reshape(-1).astype(np.float32)
    # mlx RMSNorm eps typically 1e-6
    scale = 1.0 / np.sqrt(np.mean(X ** 2, axis=1, keepdims=True) + 1e-6)
    Xn = X * scale * fw
    print("final_norm proxy logits", flush=True)
    nz, nid, _ = logits_topk(Xn)
    last_fn = [{"i": int(i), "pred": int(nid[i, 0]), "second": int(nid[i, 1]),
                "m": float(nz[i, 0] - nz[i, 1]),
                "same_as_raw_L63": int(nid[i, 0]) == int(top_id[i, 0])} for i in last_idx]

    # teacher acc under final_norm proxy
    tf_hit_fn = 0
    ntf = 0
    for i in range(N):
        if nxt[i] < 0:
            continue
        ntf += 1
        tf_hit_fn += int(nid[i, 0] == nxt[i])

    out = {
        "wall_s": time.perf_counter() - t0,
        "teacher_signed_margin": {
            "n_tf": int(tf.sum()),
            "mean": float(np.nanmean(tmargin)),
            "median": float(np.nanmedian(tmargin)),
            "frac_negative": float((tmargin[tf] < 0).mean()),
            "frac_lt_0": float((tmargin[tf] < 0).mean()),
            "frac_lt_1": float((tmargin[tf] < 1).mean()),
            "p": {str(p): float(np.nanpercentile(tmargin[tf], p)) for p in [5, 10, 25, 50, 75, 90, 95]},
            "note": "z[next]-max_other_in_topk. Negative => L63@lm_head argmax is not the actual next token.",
        },
        "ablation": abl,
        "sha256_variants": variants,
        "L62_L63_cosine_mean": float(c.mean()),
        "L62_L63_cosine_p50": float(np.median(c)),
        "final_norm_on_L63": {
            "last": last_fn,
            "teacher_top1_acc": tf_hit_fn / ntf,
            "n_tf": ntf,
            "note": "Applies language_model.model.norm to L63 captured hidden. If capture is post_attn_norm this is the wrong site.",
        },
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps(out, indent=2)[:8000])
    print("wrote", OUT, "wall", out["wall_s"])


if __name__ == "__main__":
    main()
