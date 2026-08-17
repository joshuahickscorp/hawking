#!/usr/bin/env python3
"""Doctor adversarial adequacy gate.

A functional screen is only trustworthy if it REFUSES deliberately broken
representations. The incumbent screen -- activation-conditioned agreement, i.e.
cosine between X@W and X@W_hat -- does not: a matrix that keeps only the
capture-visible subspace and throws away everything else scores ~1.0 while having
discarded most of its energy. That construction is the reason this gate exists.

Root cause is rank, not sample count. The capture spans r << d directions, so
X@W cannot observe any component of W in the (d - r)-dimensional nullspace. Any
screen built solely on captured activations is blind there by construction, and
adding tokens from the same distribution does not fix it if the distribution
itself is low-rank.

The gate therefore scores every candidate on TWO axes and takes the worse:

  observed  agreement on the captured activation distribution   (what matters in use)
  probed    agreement on isotropic random directions            (what the capture cannot see)

A faithful representation scores high on both. A representation that has been
fitted to, or trivially projected onto, the visible subspace scores high on
`observed` and collapses on `probed`.

Run:  python3 tools/gravity_doctor_gate.py            # full gate, real tensors
      python3 tools/gravity_doctor_gate.py --demo     # fast synthetic self-check
"""
from __future__ import annotations
import argparse, glob, json, os, struct, sys
import numpy as np

BF16 = "workspace/campaign/records/runs/qwen38-27b/bf16"
CAPTURE = "workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
PASS_THRESHOLD = 0.95  # a construction scoring above this is judged HEALTHY by the gate


# ---------------------------------------------------------------- loading

def load_tensor(name, root=BF16):
    idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    shard = idx["weight_map"][name]
    with open(os.path.join(root, shard), "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        meta = hdr[name]
        s, e = meta["data_offsets"]
        f.seek(base + s)
        raw = f.read(e - s)
    dt = meta["dtype"]
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        arr = u16.view(np.float32)
    elif dt == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dt == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        raise ValueError(f"dtype {dt}")
    return arr.reshape(meta["shape"]).astype(np.float32)


def load_X(layer, capture=CAPTURE):
    res = json.load(open(os.path.join(capture, "capture-result.json")))
    per = res["per_layer"][str(layer)]
    X = np.fromfile(per["path"], dtype=np.float32).reshape(per["n_rows"], res["hidden"])
    return X


# ---------------------------------------------------------------- scoring

def _rowcos(A, B):
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    return float(np.mean(num / den))


def observed_score(W, Wh, X):
    """Incumbent screen: agreement on the captured activation distribution."""
    return _rowcos(X @ W.T, X @ Wh.T)


def probed_score(W, Wh, d_in, n=256, seed=0):
    """Agreement on isotropic random directions the capture cannot span."""
    P = _probe(d_in, n, seed)
    return _rowcos(P @ W.T, P @ Wh.T)


def _probe(d_in, n=256, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n, d_in)).astype(np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P


def _worst_unit(A, B):
    """Lowest across-token agreement of any single output unit.

    A mean over output units cannot see a few destroyed ones: deleting 3 rows of
    17408 moves the mean by 2e-4. Output unit j is driven by row j of W, so a
    per-unit score localises sparse row damage that any aggregate dilutes. This
    is the same defect class as a capability metric that folded 1.0 over 402 rows
    whose real value was None.
    """
    num = (A * B).sum(0)
    na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
    live = na > 1e-20                      # units the reference actually drives
    cos = np.zeros_like(num)
    denom = na * nb + 1e-30
    cos[live] = num[live] / denom[live]     # a zeroed candidate unit scores 0, not nan
    return float(cos[live].min()) if live.any() else 1.0


def axes(W, Wh, X, seed=0):
    P = _probe(W.shape[1], seed=seed)
    return {
        "observed": observed_score(W, Wh, X),
        "probed": probed_score(W, Wh, W.shape[1], seed=seed),
        "worst_unit": min(_worst_unit(X @ W.T, X @ Wh.T),
                          _worst_unit(P @ W.T, P @ Wh.T)),
    }


# How far below the honest-codec reference an axis may fall before the candidate
# is judged UNHEALTHY. Absolute thresholds do not work here: `observed` and
# `probed` are means over rows, while `worst_unit` is an extreme order statistic
# over as many as 17408 output units, so they have different natural scales AND
# the scale moves with depth (honest Q4 worst_unit measured 0.9638 at L0 gate,
# 0.9421 at L47 up, 0.9019 at L31 q). Judging against a same-tensor reference is
# the calibration; a flat constant would just be a number chosen to fit.
AXIS_MARGIN = {"observed": 0.02, "probed": 0.02, "worst_unit": 0.10}


def gate(W, Wh, X, ref=None, seed=0):
    """Score a candidate against a same-tensor honest-codec reference.

    `ref` is the axis dict of a faithful cheap codec on this tensor. Every axis
    must land within its margin of that reference. Omitting `ref` falls back to
    an absolute threshold, which is weaker and only appropriate for a synthetic
    self-check where the scale is known.
    """
    a = axes(W, Wh, X, seed=seed)
    if ref is None:
        g = min(a.values())
        return {**a, "gate": g, "healthy": g >= PASS_THRESHOLD, "mode": "absolute"}
    deficits = {k: a[k] - (ref[k] - AXIS_MARGIN[k]) for k in a}
    worst = min(deficits, key=deficits.get)
    return {**a, "ref": ref, "deficit": deficits,
            "gate": deficits[worst], "worst_axis": worst,
            "healthy": deficits[worst] >= 0.0, "mode": "relative"}


# ---------------------------------------------------------------- constructions

def c_visible_subspace(W, X, keep=None):
    """Project W onto the capture-visible row space; discard the nullspace."""
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    r = keep or np.linalg.matrix_rank(X, tol=1e-3 * np.linalg.norm(X, 2))
    B = Vt[:r]
    return W @ (B.T @ B)


def c_unseen_corruption(W, X, scale=1.0, seed=1):
    """Leave the visible subspace exact; corrupt only what the capture cannot see."""
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    r = np.linalg.matrix_rank(X, tol=1e-3 * np.linalg.norm(X, 2))
    B = Vt[:r]
    P_vis = B.T @ B
    rng = np.random.default_rng(seed)
    N = rng.standard_normal(W.shape).astype(np.float32)
    N *= scale * np.linalg.norm(W) / (np.linalg.norm(N) + 1e-30)
    return W @ P_vis + N @ (np.eye(W.shape[1], dtype=np.float32) - P_vis)


def c_channel_deletion(W, k=3, seed=2):
    """Delete the highest-energy output rows outright."""
    Wh = W.copy()
    rows = np.argsort(-np.linalg.norm(W, axis=1))[:k]
    Wh[rows] = 0.0
    return Wh


def c_control_path(W, rows, jitter=0.5, seed=3):
    """Corrupt specific control/stop rows, leaving the bulk untouched.

    Raises if no requested row exists in this tensor. A construction that
    corrupts nothing scores HEALTHY and looks like the gate missing a pathology,
    when in fact the test never ran -- exactly the class of silent no-op this
    whole gate exists to prevent.
    """
    rows = sorted({r for r in rows if 0 <= r < W.shape[0]})
    if not rows:
        raise ValueError(
            f"control_path: none of the requested rows exist in a {W.shape[0]}-row tensor; "
            "pick rows in range or run this construction against lm_head")
    Wh = W.copy()
    rng = np.random.default_rng(seed)
    Wh[rows] += rng.standard_normal((len(rows), W.shape[1])).astype(np.float32) * (
        jitter * np.linalg.norm(W[rows], axis=1, keepdims=True) / np.sqrt(W.shape[1]))
    assert not np.array_equal(Wh, W), "control_path produced an unchanged tensor"
    return Wh


def c_uniform(W, bits=4, group=128):
    """Honest symmetric uniform quantization, group-wise absmax."""
    Wh = W.astype(np.float32).copy()
    lim = (1 << (bits - 1)) - 1
    d = W.shape[1]
    for s in range(0, d - d % group, group):
        blk = Wh[:, s:s + group]
        amax = np.abs(blk).max(axis=1, keepdims=True) + 1e-30
        step = amax / lim
        Wh[:, s:s + group] = np.clip(np.round(blk / step), -lim, lim) * step
    return Wh


def c_faithful_q4(W, group=128):
    """The reference codec. Its own deficit is trivially the margin, so it cannot
    fail -- it calibrates the scale, it does not validate it. Independent controls
    (a richer codec that must pass, a too-cheap one that must fail) do that."""
    return c_uniform(W, 4, group)


# ---------------------------------------------------------------- runner

def run(tensor, layer, verbose=True):
    W = load_tensor(tensor)
    X = load_X(layer)
    if X.shape[1] != W.shape[1]:
        raise SystemExit(f"dim mismatch X {X.shape} vs W {W.shape}")

    rank = int(np.linalg.matrix_rank(X, tol=1e-3 * np.linalg.norm(X, 2)))
    # Real control/stop token rows exist only in lm_head (248320 rows). For any other
    # tensor use an equivalent-sized sparse row set that is actually in range, so the
    # pathology is genuinely constructed rather than silently skipped.
    real_control = [r for r in range(248044, 248077)]
    if max(real_control) < W.shape[0]:
        control_rows, control_label = real_control, "control_path_corruption (real stop-token rows)"
    else:
        step = max(1, W.shape[0] // len(real_control))
        control_rows = [i * step for i in range(len(real_control))]
        control_label = f"sparse_row_corruption ({len(control_rows)} rows, stand-in for control path)"

    ref_W = c_faithful_q4(W)
    ref = axes(ref_W, W, X)          # reference = honest codec vs itself is 1.0; use W vs ref_W
    ref = axes(W, ref_W, X)
    cases = [
        ("q4_g128 = REFERENCE (calibrates scale, cannot fail)", ref_W, True),
        ("q6_g128 (INDEPENDENT POSITIVE, richer, must pass)", c_uniform(W, 6), True),
        ("q2_g128 (INDEPENDENT NEGATIVE, honest but too cheap, must fail)", c_uniform(W, 2), False),
        ("visible_subspace_only", c_visible_subspace(W, X), False),
        ("unseen_subspace_corruption", c_unseen_corruption(W, X), False),
        ("critical_channel_deletion", c_channel_deletion(W), False),
        (control_label, c_control_path(W, control_rows), False),
    ]

    if verbose:
        print(f"tensor {tensor}  shape {tuple(W.shape)}")
        print(f"X {X.shape}  numerical rank {rank} / {X.shape[1]}")
        print(f"  reference (honest Q4 g128): observed={ref['observed']:.6f} probed={ref['probed']:.6f} worst_unit={ref['worst_unit']:.6f}")
        print(f"  margins: {AXIS_MARGIN}")
        print(f"{'construction':<48} {'observed':>9} {'probed':>9} {'worst_u':>9} {'margin':>9}  verdict")

    results, failures = [], []
    for name, Wh, should_pass in cases:
        g = gate(W, Wh, X, ref=ref)
        old = g["observed"] >= PASS_THRESHOLD          # incumbent screen verdict
        new = g["healthy"]                              # this gate's verdict
        ok = (new == should_pass)
        if not ok:
            failures.append(name)
        if verbose:
            v = "HEALTHY" if new else "UNHEALTHY"
            flag = "" if ok else "   <-- GATE FAILS"
            miss = "  (incumbent screen says HEALTHY)" if old and not should_pass else ""
            print(f"{name:<48} {g['observed']:>9.6f} {g['probed']:>9.6f} {g['worst_unit']:>9.6f} {g['gate']:>+9.6f}  {v}{miss}{flag}")
        results.append({"construction": name, "should_pass": should_pass,
                        "incumbent_healthy": old, "gate_healthy": new, **g})

    return {"tensor": tensor, "layer": layer, "x_rank": rank,
            "x_dim": int(X.shape[1]), "results": results, "failures": failures}


def demo():
    """Synthetic self-check: the gate must reject a visible-subspace fit."""
    rng = np.random.default_rng(0)
    d_in, d_out, r = 256, 64, 12
    W = rng.standard_normal((d_out, d_in)).astype(np.float32)
    B = np.linalg.qr(rng.standard_normal((d_in, r)).astype(np.float32))[0].T
    X = (rng.standard_normal((128, r)).astype(np.float32) @ B)

    faithful = c_faithful_q4(W, group=64)
    cheat = W @ (B.T @ B)

    gf, gc = gate(W, faithful, X), gate(W, cheat, X)
    assert gf["healthy"], f"faithful codec rejected: {gf}"
    assert gc["observed"] > 0.99, f"cheat should fool the incumbent screen: {gc}"
    assert not gc["healthy"], f"cheat NOT caught: {gc}"
    print(f"doctor gate demo: PASS")
    print(f"  faithful  observed={gf['observed']:.6f} probed={gf['probed']:.6f} worst_unit={gf['worst_unit']:.6f} -> HEALTHY")
    print(f"  cheat     observed={gc['observed']:.6f} probed={gc['probed']:.6f} worst_unit={gc['worst_unit']:.6f} -> UNHEALTHY")
    print(f"  incumbent screen would have passed the cheat at {gc['observed']:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--tensor", default="language_model.model.layers.0.mlp.gate_proj.weight")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.demo:
        demo()
        sys.exit(0)
    r = run(a.tensor, a.layer)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=2)
    print()
    if r["failures"]:
        print(f"GATE FAILS on {len(r['failures'])}: {r['failures']}")
        sys.exit(1)
    print("GATE ADEQUATE: every pathological construction scored UNHEALTHY, "
          "positive control scored HEALTHY")
