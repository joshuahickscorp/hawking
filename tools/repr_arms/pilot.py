#!/usr/bin/env python3.12
"""Pilot the six representation arms on REAL GLM-5.2 BF16 tensors.

The synthetic matrix run in `arms.py` proved the arms are WIRED.  This runs them on real
weights from the rehydrated source window, which is a different and much sharper question.

## What a representative window can and cannot decide

Shards 1-5 carry roughly four of seventy-eight layers.  A four-layer slice of a model
cannot generate, so on this window the tournament's own gates split:

  MEASURABLE HERE   exact complete-byte accounting; oracle/runtime parity of the codec;
                    functional preservation of an organ under its real input distribution;
                    next-layer propagation of the error the arm introduces
  NOT MEASURABLE    G_math (2 + 2), G_live (prompt-dependent generation), G_halo,
                    long context. All four need the whole model, and no partial-window
                    result may be reported as passing them.

That split is the honest scope of a pilot and is stated in the receipt, so a later reader
cannot mistake a window result for a capability result.  It is the same mistake that let a
byte-perfect Math-Preserve be called ready.

## Why functional, not reconstruction

The frozen tournament rules weight-space reconstruction error INADMISSIBLE as a promotion
signal, and the sub-bit law names it explicitly as evidence it refuses.  So the arms are
scored on what the organ DOES to real activations, not on how closely it reproduces the
parent's numbers.  Reconstruction error is still printed -- it is cheap and diagnostic --
and is marked inadmissible everywhere it appears.

    .venv/glm52/bin/python tools/repr_arms/pilot.py --organ dense_mlp
    .venv/glm52/bin/python tools/repr_arms/pilot.py --list
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.repr_arms.arms import (  # noqa: E402
    pack_a1, unpack_a1, pack_a2, unpack_a2, pack_a3, unpack_a3,
    pack_a4, unpack_a4, pack_a5, pack_a6,
)

SOURCE = Path.home() / "Library/Application Support/Hawking/GLM52Gravity/pilot_source"
SEED = 0x5EED


def read_tensor(shard: Path, name: str) -> np.ndarray:
    """Read one BF16 tensor out of a safetensors shard without loading the shard."""
    with shard.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        info = hdr[name]
        lo, hi = info["data_offsets"]
        f.seek(base + lo)
        raw = f.read(hi - lo)
    # BF16 -> float32: the 16 bits are the high half of the float32.
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(info["shape"])


def organ_catalogue() -> dict[str, tuple[str, str]]:
    """Named organs in the pilot window, one per class the flagship has."""
    s1 = "model-00001-of-00282.safetensors"
    return {
        "dense_mlp": (s1, "model.layers.0.mlp.down_proj.weight"),
        "attn_o": (s1, "model.layers.0.self_attn.o_proj.weight"),
        "embed": (s1, "model.embed_tokens.weight"),
        "lm_head": (s1, "lm_head.weight"),
    }


def functional_error(w: np.ndarray, w_hat: np.ndarray, rng: np.random.Generator,
                     n: int = 64) -> dict:
    """What the organ DOES to real-shaped activations, not how close its numbers are.

    The input distribution is standard normal over the organ's input dimension. That is a
    proxy for the true activation distribution -- a real one needs captured activations,
    which this window does not carry -- and it is labelled as a proxy in the receipt.
    """
    x = rng.standard_normal((n, w.shape[1])).astype(np.float32)
    y, y_hat = x @ w.T, x @ w_hat.T
    num = np.linalg.norm(y - y_hat)
    den = np.linalg.norm(y) + 1e-12
    # Cosine per row, then averaged: direction matters more than scale downstream.
    yn = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    yhn = y_hat / (np.linalg.norm(y_hat, axis=1, keepdims=True) + 1e-12)
    cos = float((yn * yhn).sum(1).mean())
    return {"relative_output_error": float(num / den), "mean_row_cosine": cos}


def constant_mean_null(w: np.ndarray, rng: np.random.Generator, n: int = 64) -> dict:
    """The null every cosine must be read against.

    Raw activation cosine on this parent has a measured constant-mean null of 0.898: a
    predictor that emits the mean row scores that well while carrying no information. A
    cosine below the null is worse than useless, and one just above it is noise.
    """
    x = rng.standard_normal((n, w.shape[1])).astype(np.float32)
    y = x @ w.T
    mean_row = np.repeat(y.mean(0, keepdims=True), n, axis=0)
    yn = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    mn = mean_row / (np.linalg.norm(mean_row, axis=1, keepdims=True) + 1e-12)
    return {"constant_mean_cosine_null": float((yn * mn).sum(1).mean())}


def run_organ(organ: str, max_rows: int = 2048) -> dict:
    shard_name, tensor = organ_catalogue()[organ]
    shard = SOURCE / shard_name
    if not shard.exists():
        raise SystemExit(f"{shard} not present -- rehydrate the window first")

    t0 = time.time()
    w_full = read_tensor(shard, tensor)
    # Bound the pilot: a 154880 x 6144 embedding is 3.8 GB in float32 and the arms are
    # O(rows). A contiguous row block preserves structure a random sample would destroy.
    w = np.ascontiguousarray(w_full[:max_rows]) if w_full.shape[0] > max_rows else w_full
    del w_full
    rng = np.random.default_rng(SEED)
    importance = np.abs(w).sum(1)

    rows = []
    for arm, pack, unpack, args in (
        ("A1", pack_a1, unpack_a1, (w,)),
        ("A2", pack_a2, unpack_a2, (w,)),
        ("A3", pack_a3, unpack_a3, (w,)),
    ):
        p = pack(*args)
        w_hat = unpack(p)
        fe = functional_error(w, w_hat, rng)
        rows.append({
            "arm": arm, "status": "EXECUTED_ON_REAL_TENSOR",
            "complete_bpw": round(p.complete_bpw, 5),
            "total_bytes": p.total_bytes,
            "components": p.component_bytes,
            "accounting_reconciles": p.reconciles(),
            "sub_bit": p.complete_bpw < 1.0,
            **{k: round(v, 6) for k, v in fe.items()},
            "reconstruction_relative_error_INADMISSIBLE": round(
                float(np.linalg.norm(w - w_hat) / (np.linalg.norm(w) + 1e-12)), 6),
        })

    # A4 needs a bank of same-shaped organs; a single tensor is split into blocks so the
    # shared-basis question is asked honestly rather than skipped.
    blocks = [np.ascontiguousarray(b) for b in np.array_split(w, 8, axis=0)]
    if all(b.shape == blocks[0].shape for b in blocks):
        p4 = pack_a4(blocks)
        back = unpack_a4(p4)
        w_hat4 = np.concatenate(back, axis=0)
        fe = functional_error(w, w_hat4, rng)
        rows.append({
            "arm": "A4", "status": "EXECUTED_ON_REAL_TENSOR",
            "complete_bpw": round(p4.complete_bpw, 5), "total_bytes": p4.total_bytes,
            "components": p4.component_bytes, "accounting_reconciles": p4.reconciles(),
            "sub_bit": p4.complete_bpw < 1.0,
            **{k: round(v, 6) for k, v in fe.items()},
            "note": "organ split into 8 equal blocks to pose the shared-basis question",
        })
    else:
        rows.append({"arm": "A4", "status": "SKIPPED", "why": "organ rows not divisible into equal blocks"})

    p5 = pack_a5(w, importance)
    rows.append({
        "arm": "A5", "status": "PACK_ONLY_NO_DECODER",
        "complete_bpw": round(p5.complete_bpw, 5), "total_bytes": p5.total_bytes,
        "components": p5.component_bytes, "accounting_reconciles": p5.reconciles(),
        "sub_bit": p5.complete_bpw < 1.0,
        "gap": "decoder unimplemented, so no functional measurement is possible",
    })

    try:
        pack_a6(w, None)
        rows.append({"arm": "A6", "status": "UNEXPECTEDLY_RAN"})
    except NotImplementedError as e:
        rows.append({"arm": "A6", "status": "DATA_BLOCKED", "why": str(e), "not_a_failure": True})

    return {
        "organ": organ,
        "tensor": tensor,
        "shard": shard_name,
        "shape_used": list(w.shape),
        "rows_bounded_to": max_rows,
        "dtype_source": "BF16 -> float32",
        "null": constant_mean_null(w, rng),
        "arms": rows,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organ", default="dense_mlp", choices=sorted(organ_catalogue()))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.list:
        for k, (s, t) in organ_catalogue().items():
            print(f"{k:12s} {s}  {t}")
        return 0

    organs = sorted(organ_catalogue()) if a.all else [a.organ]
    doc = {
        "schema": "hawking.representation.pilot_real_window.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "rehydrated GLM-5.2 BF16 shards, verified against the sealed per-file sha256",
        "revision": "b4734de4facf877f85769a911abafc5283eab3d9",
        "window": "shards 1-5 of 282, roughly four layers of seventy-eight",
        "results": [run_organ(o) for o in organs],
        "gates_measurable_on_this_window": [
            "exact complete-byte accounting",
            "sub-bit legality at pilot scope",
            "functional preservation of an organ under a proxy input distribution",
        ],
        "gates_NOT_measurable_on_this_window": {
            "G_math": "needs the whole model; four layers cannot complete '2 + 2 ='",
            "G_live": "needs the whole model",
            "G_halo": "needs the whole model and a served endpoint",
            "long_context": "needs the whole model",
            "G_cascade": "needs many more layers than this window carries",
        },
        "admissibility": {
            "functional_error": "ADMISSIBLE as a pilot signal, with the stated proxy caveat",
            "reconstruction_error": "INADMISSIBLE for promotion, per the frozen tournament and the sub-bit law. Printed as a cheap diagnostic only.",
            "cosine": "must be read against the constant-mean null reported per organ; on this parent that null is around 0.898",
        },
        "what_no_result_here_may_be_read_as": "evidence that any arm recovers GLM. That requires a full pack and the whole-model gates.",
    }
    out = json.dumps(doc, indent=2)
    if a.out:
        Path(a.out).write_text(out + "\n")
        print(f"wrote {a.out}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
