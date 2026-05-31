#!/usr/bin/env python3
"""Bible Stage-0 oracle B — SVD energy spectrum of the LM head.

Decides whether low-rank screening of the LM head could add anything beyond
lm_head->predec. Reads the real Qwen-3B output (lm_head) tensor from GGUF,
dequantizes, and computes the singular-value energy spectrum via eig(WᵀW).
If 99% energy sits at rank r << hidden, the head is low-rank and a rank-r
screen *might* skip full-vocab logits (then a follow-up needs an activation
top-k recall test). If energy is spread, SVD screening is dead.

Caveat: this is the weight-only half. Top-k *recall* needs real hidden states
(run the model) — flagged as follow-up if the spectrum looks promising. The LM
head is ~4% of decode, so this oracle is intentionally low-priority.

Run with the gguf venv: /tmp/ggufenv/bin/python tools/bench/oracle_svd_lmhead.py
"""
import json
import os
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    out = "reports/oracle/svd_lmhead.json"
    r = GGUFReader(path)
    by_name = {t.name: t for t in r.tensors}
    # Qwen2.5 may tie embeddings; prefer a dedicated output weight, else embd.
    name = next((n for n in ("output.weight", "lm_head.weight") if n in by_name), None)
    tied = False
    if name is None:
        name = "token_embd.weight"
        tied = True
    t = by_name[name]
    qtype = t.tensor_type.name
    W = dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
    if W.ndim != 2:
        W = W.reshape(t.shape[::-1])  # gguf stores reversed dims
    vocab, hidden = (W.shape if W.shape[0] >= W.shape[1] else W.shape[::-1])
    if W.shape[0] < W.shape[1]:
        W = W.T
    # singular values^2 = eigenvalues of WᵀW (hidden x hidden, cheap).
    gram = W.T @ W
    eig = np.linalg.eigvalsh(gram)
    eig = np.clip(eig[::-1], 0, None)  # descending
    sv = np.sqrt(eig)
    energy = eig / eig.sum()
    cum = np.cumsum(energy)
    def rank_for(p):
        return int(np.searchsorted(cum, p) + 1)
    dim = len(sv)
    ranks = {"e90": rank_for(0.90), "e95": rank_for(0.95),
             "e99": rank_for(0.99), "e999": rank_for(0.999)}
    r99_frac = ranks["e99"] / dim
    verdict = ("LOW-RANK (screen candidate — needs activation top-k recall test)"
               if r99_frac < 0.5 else
               "FULL-RANK (SVD screening NO-GO — no compressible structure)")
    res = {
        "oracle": "svd_lmhead_energy",
        "tensor": name, "tied_embedding": tied, "qtype": qtype,
        "shape_vocab_hidden": [int(vocab), int(hidden)],
        "rank_for_energy": ranks, "dim": dim,
        "rank99_fraction_of_dim": round(r99_frac, 3),
        "sigma_max": float(sv[0]), "sigma_min": float(sv[-1]),
        "cond_number": float(sv[0] / max(sv[-1], 1e-12)),
        "verdict": verdict,
        "note": ("Weight-only spectrum. LM head ~4% of decode; oracle is low "
                 "priority. Recall half (top-k preserved under rank-r) needs "
                 "real hidden states — do only if this says LOW-RANK."),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"tensor={name} qtype={qtype} shape=({vocab},{hidden})")
    print(f"rank@90/95/99/99.9% = {ranks['e90']}/{ranks['e95']}/{ranks['e99']}/{ranks['e999']}  of {dim}")
    print(f"rank99 = {r99_frac*100:.1f}% of dim  ->  {verdict}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
