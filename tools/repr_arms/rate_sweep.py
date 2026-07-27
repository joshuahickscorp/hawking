#!/usr/bin/env python3.12
"""Rate sweep: at a TARGET complete BPW, how much of a real GLM organ's function survives?

The first pilot ran every arm at the hyperparameters chosen for a 64x64 synthetic fixture.
On real 2048x12288 GLM organs those settings are meaningless: A1 at rank 4 lands at 0.035
BPW and destroys the organ, A2 stores a float16 residual and lands at 16 BPW while
reproducing it perfectly.  Neither is a result about the arm; both are results about
defaults.

The question the tournament actually asks is a rate-distortion curve **in functional
space**: swept across a rate budget, how much of what the organ DOES survives?  That is
what this measures, and it is the only form in which "sub-bit fails for this parent" could
become a supported claim rather than an inference from one artifact's collapse.

Scored on functional preservation against the constant-mean null.  Reconstruction error is
inadmissible for promotion -- the frozen tournament and the sub-bit law both say so -- and
appears here only as a diagnostic.

    .venv/glm52/bin/python tools/repr_arms/rate_sweep.py --organ dense_mlp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.repr_arms.arms import Packed, _pq  # noqa: E402
from tools.repr_arms.pilot import (  # noqa: E402
    SEED, constant_mean_null, functional_error, organ_catalogue, read_tensor, SOURCE,
)

import struct


# --------------------------------------------------------------------------
# Rate-parameterised arms. Each takes an explicit budget knob so the sweep can
# ask "what does this family buy at THIS rate" rather than "what did a default do".
# --------------------------------------------------------------------------
def a1_rank(w: np.ndarray, rank: int):
    m, n = w.shape
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    u, s, vt = u[:, :rank], s[:rank], vt[:rank]
    head = struct.pack("<4sIIH", b"A1FN", m, n, rank)
    body = np.concatenate([u.ravel(), s.ravel(), vt.ravel()]).astype(np.float16).tobytes()
    p = Packed("A1", head + body, m * n, {"header": len(head), "factors": len(body)})
    return p, (u * s) @ vt


def a3_stages(w: np.ndarray, stages: int, ncent: int):
    """Additive multi-codebook. Codes cost log2(ncent) bits/row/stage; codebooks are the
    fixed cost that dominates at small row counts."""
    m, n = w.shape
    rng = np.random.default_rng(SEED)
    resid = w.astype(np.float32).copy()
    books, codes = [], []
    for _ in range(stages):
        idx = rng.choice(m, size=min(ncent, m), replace=False)
        book = resid[idx].copy()
        d = ((resid[:, None, :] - book[None, :, :]) ** 2).sum(-1)
        c = d.argmin(1).astype(np.uint8)
        resid = resid - book[c]
        books.append(book)
        codes.append(c)
    head = struct.pack("<4sIIHH", b"A3AD", m, n, stages, ncent)
    bb = np.stack(books).astype(np.float16).tobytes()
    cb = np.stack(codes, 1).tobytes()
    out = np.zeros_like(w, dtype=np.float32)
    for s_ in range(stages):
        out += books[s_][codes[s_]]
    p = Packed("A3", head + bb + cb, m * n,
               {"header": len(head), "codebooks": len(bb), "codes": len(cb)})
    return p, out


def a2_pq_resid(w: np.ndarray, nsub: int, ncent: int, resid_bits: int):
    """PQ base plus a residual quantised to `resid_bits`, so the residual is a real rate
    choice instead of an unpriced float16 copy."""
    m, n = w.shape
    books, codes, sub = _pq(w, nsub, ncent, "a2sweep")
    approx = np.concatenate([books[j][codes[:, j]] for j in range(nsub)], 1)
    resid = w - approx
    if resid_bits <= 0:
        rq, scale, rb = np.zeros_like(resid), 1.0, b""
    else:
        levels = (1 << resid_bits) - 1
        scale = float(np.abs(resid).max()) or 1.0
        q = np.clip(np.round((resid / scale + 1) * levels / 2), 0, levels).astype(np.uint8)
        rq = (q.astype(np.float32) * 2 / levels - 1) * scale
        rb = np.packbits(np.unpackbits(q[..., None], axis=-1)[..., -resid_bits:].reshape(-1)).tobytes()
    head = struct.pack("<4sIIHHH", b"A2PQ", m, n, nsub, ncent, resid_bits)
    bb = np.concatenate([b.ravel() for b in books]).astype(np.float16).tobytes()
    cb = codes.tobytes()
    p = Packed("A2", head + bb + cb + rb, m * n,
               {"header": len(head), "codebooks": len(bb), "codes": len(cb), "residual": len(rb)})
    return p, approx + rq


def a4_basis(blocks: list[np.ndarray], k: int):
    stack = np.stack([b.ravel() for b in blocks]).astype(np.float32)
    u, s, vt = np.linalg.svd(stack, full_matrices=False)
    k = min(k, vt.shape[0])
    gates = (u[:, :k] * s[:k]).astype(np.float16)
    basis = vt[:k].astype(np.float16)
    m, n = blocks[0].shape
    head = struct.pack("<4sIIIH", b"A4SB", len(blocks), m, n, k)
    p = Packed("A4", head + gates.tobytes() + basis.tobytes(),
               sum(b.size for b in blocks),
               {"header": len(head), "gates": gates.nbytes, "basis": basis.nbytes})
    rec = (gates.astype(np.float32) @ basis.astype(np.float32))
    return p, np.concatenate([rec[i].reshape(m, n) for i in range(len(blocks))], axis=0)


def sweep(organ: str, max_rows: int = 2048) -> dict:
    shard_name, tensor = organ_catalogue()[organ]
    w_full = read_tensor(SOURCE / shard_name, tensor)
    w = np.ascontiguousarray(w_full[:max_rows])
    del w_full
    rng = np.random.default_rng(SEED)
    null = constant_mean_null(w, rng)["constant_mean_cosine_null"]
    m, n = w.shape

    pts = []

    def add(arm, cfg, p, w_hat):
        fe = functional_error(w, w_hat, np.random.default_rng(SEED))
        pts.append({
            "arm": arm, "config": cfg,
            "complete_bpw": round(p.complete_bpw, 5),
            "sub_bit": p.complete_bpw < 1.0,
            "mean_row_cosine": round(fe["mean_row_cosine"], 5),
            "beats_null": fe["mean_row_cosine"] > null,
            "relative_output_error": round(fe["relative_output_error"], 5),
            "accounting_reconciles": p.reconciles(),
        })

    # A1: rank sweep. Rank r costs r*(m+n+1) halves.
    for r in (4, 16, 64, 256, 512):
        if r < min(m, n):
            p, wh = a1_rank(w, r)
            add("A1", {"rank": r}, p, wh)

    # A3: stage/centroid sweep.
    for stages, ncent in ((3, 16), (8, 64), (16, 128), (32, 256)):
        p, wh = a3_stages(w, stages, min(ncent, m))
        add("A3", {"stages": stages, "ncent": min(ncent, m)}, p, wh)

    # A2: residual bit-depth is the rate knob.
    for nsub, ncent, rbits in ((8, 16, 0), (8, 16, 1), (16, 64, 1), (16, 64, 2), (32, 256, 2)):
        p, wh = a2_pq_resid(w, nsub, min(ncent, m), rbits)
        add("A2", {"nsub": nsub, "ncent": min(ncent, m), "resid_bits": rbits}, p, wh)

    # A4: shared basis across 8 equal blocks.
    blocks = [np.ascontiguousarray(b) for b in np.array_split(w, 8, axis=0)]
    if all(b.shape == blocks[0].shape for b in blocks):
        for k in (1, 2, 4, 6):
            p, wh = a4_basis(blocks, k)
            add("A4", {"basis": k}, p, wh)

    sub = [p for p in pts if p["sub_bit"]]
    best_sub = max(sub, key=lambda x: x["mean_row_cosine"], default=None)
    best_any = max(pts, key=lambda x: x["mean_row_cosine"], default=None)
    return {
        "organ": organ, "tensor": tensor, "shape": [m, n],
        "constant_mean_cosine_null": round(null, 5),
        "points": sorted(pts, key=lambda x: (x["arm"], x["complete_bpw"])),
        "best_sub_bit": best_sub,
        "best_any_rate": best_any,
        "sub_bit_points_beating_null": sum(1 for p in sub if p["beats_null"]),
        "sub_bit_points_total": len(sub),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organ", default="dense_mlp", choices=sorted(organ_catalogue()))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    organs = sorted(organ_catalogue()) if a.all else [a.organ]
    t0 = time.time()
    doc = {
        "schema": "hawking.representation.rate_sweep.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "rehydrated GLM-5.2 BF16, verified against the sealed per-file sha256",
        "question": "at a given complete BPW, how much of a real GLM organ's FUNCTION survives?",
        "scoring": "functional cosine against the organ's own constant-mean null. Reconstruction error is INADMISSIBLE for promotion and is reported as a diagnostic only.",
        "results": [sweep(o) for o in organs],
        "seconds": None,
        "scope_caveat": "one organ at a time, 2048 rows, on a 4-of-78-layer window. This can support 'no swept sub-bit configuration of these families preserves this organ's function', which is a real and useful negative. It cannot on its own establish whole-model behaviour, and no whole-model gate is measurable here.",
    }
    doc["seconds"] = round(time.time() - t0, 1)
    out = json.dumps(doc, indent=2)
    if a.out:
        Path(a.out).write_text(out + "\n")
        print(f"wrote {a.out}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
