#!/usr/bin/env python3
"""S032 §34 / S033: COMPONENT_PERTURBATION as a callable WorkUnit.

One call, one curve point. The resident names tensor, layer, side and fraction;
this damages exactly that, replays real tokens, and returns the measured effect.
It prints one JSON line so a scheduler can parse it without a receipt round trip.

The resident cannot reach anything except gate/up/down on a real layer, and
cannot write to any artifact. Damage is applied to an in-memory copy.

    python3 tools/future/perturbation_workunit.py --tensor down --layer 21 \
        --side rows --fraction 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TENSORS = ("gate", "up", "down")
SIDES = ("rows", "cols")
N_MORE = 2

# S031 §34 names four perturbation types, and only ZERO existed. The other three
# are what separate "this information can be deleted" from "this information can
# be CHEAPER", which are different questions and only the second is a route to a
# smaller representation.
KINDS = ("zero", "quantize", "noise", "low_rank")
DEFAULT_KIND = "zero"


class PerturbRefused(RuntimeError):
    """The request is outside what this WorkUnit may touch."""


def perturb(W, side: str, fraction: float, kind: str, seed: int):
    """Apply one perturbation to a COPY. Pure, so it is testable without a replay.

    Returns (W2, n_elements_touched, detail). `fraction` means the same thing in
    every kind - the share of rows or columns SELECTED - so two kinds at the same
    fraction touch the same count of elements and are matched by construction.
    Matched-fraction across DIFFERENT-SIZED tensors is still not a control; that
    is what elements_destroyed is reported for.
    """
    import numpy as np

    if kind not in KINDS:
        raise PerturbRefused(f"kind {kind!r} not in {KINDS}")
    rng = np.random.default_rng(seed)
    W2 = np.array(W, copy=True)
    n = W2.shape[0] if side == "rows" else W2.shape[1]
    k = max(1, int(round(n * fraction)))
    idx = rng.choice(n, size=k, replace=False)
    sel = (idx, slice(None)) if side == "rows" else (slice(None), idx)
    block = W2[sel]
    nelem = int(block.size)
    detail: dict[str, Any] = {"kind": kind, "selected": int(k), "of": int(n)}

    if kind == "zero":
        W2[sel] = 0.0
    elif kind == "quantize":
        # Uniform 2-bit per selected row/col, scaled by that slice's own range.
        # Not the production packing - this asks whether the INFORMATION
        # survives coarse quantization, not whether a particular codec does.
        lo = block.min(axis=1, keepdims=True) if side == "rows" else \
            block.min(axis=0, keepdims=True)
        hi = block.max(axis=1, keepdims=True) if side == "rows" else \
            block.max(axis=0, keepdims=True)
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        levels = 3.0
        q = np.rint((block - lo) / span * levels)
        W2[sel] = lo + q / levels * span
        detail["bits"] = 2
    elif kind == "noise":
        # Gaussian at the slice's own scale, so a small-magnitude row is not
        # destroyed while a large one is barely touched.
        sigma = float(np.std(block))
        W2[sel] = block + rng.normal(0.0, sigma, size=block.shape)
        detail["sigma"] = sigma
        detail["relative_sigma"] = 1.0
    elif kind == "low_rank":
        # Replace the selected slice with its best rank-r approximation.
        r = max(1, int(round(min(block.shape) * 0.1)))
        u, sv, vt = np.linalg.svd(np.asarray(block, np.float64),
                                  full_matrices=False)
        W2[sel] = (u[:, :r] * sv[:r]) @ vt[:r]
        detail["rank"] = int(r)
        detail["of_rank"] = int(min(block.shape))
        detail["energy_kept"] = float(
            (sv[:r] ** 2).sum() / (sv ** 2).sum()) if sv.size else 1.0
    return W2, nelem, detail


def run(tensor: str, layer: int, side: str, fraction: float,
        kind: str = DEFAULT_KIND, seed: int = 33) -> dict[str, Any]:
    if tensor not in TENSORS:
        raise PerturbRefused(f"tensor {tensor!r} not in {TENSORS}")
    if side not in SIDES:
        raise PerturbRefused(f"side {side!r} not in {SIDES}")
    if not 0.01 <= fraction <= 0.95:
        raise PerturbRefused(f"fraction {fraction} outside 0.01-0.95")
    if kind not in KINDS:
        raise PerturbRefused(f"kind {kind!r} not in {KINDS}")
    import numpy as np
    from tools.future import capability_information_map as cm

    if not 0 <= layer < 64:
        raise PerturbRefused(f"layer {layer} outside 0-63")

    t0 = time.time()
    cap = cm.capture_real_prefix()
    base = cm.replay_prompt_from(cap, layer, N_MORE)["hidden_after_n"]
    W = cap["kits"][layer].mlp()[tensor]["W"]
    W2, nelem, detail = perturb(W, side, fraction, kind, seed)
    out = cm.replay_prompt_from(cap, layer, N_MORE,
                                mlp_override={tensor: W2}, override_layer=layer)

    def cos(a, b):
        a = np.asarray(a, np.float64).ravel()
        b = np.asarray(b, np.float64).ravel()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    c = cos(base, out["hidden_after_n"])
    return {
        "work_type": "PERTURB",
        "tensor": tensor, "layer": layer, "side": side, "fraction": fraction,
        "kind": kind, "kind_detail": detail,
        "shape": list(W.shape), "elements_destroyed": nelem,
        "hidden_cosine_after_2_layers": c,
        "damage": 1.0 - c,
        "seconds": round(time.time() - t0, 1),
        "measured_level": "LOCAL_FUNCTIONAL_FIDELITY",
        "not_capability": (
            "this is hidden-state cosine two layers downstream. It is NOT a "
            "capability measurement and must not be reported as one."
        ),
    }


def contract() -> dict[str, Any]:
    """What this WorkUnit accepts and what it refuses to claim.

    G119's acceptance: one call produces a curve point. The point is
    (fraction, elements_destroyed, local error) at a named kind - and the
    receipt must say, in its own words, that a local error is not a capability
    effect. capability_stages.py is where the deeper levels live; this unit
    produces the cheapest rung and hands it on.
    """
    return {
        "work_type": "PERTURB",
        "inputs": {
            "tensor": list(TENSORS),
            "layer": "0-63",
            "side": list(SIDES),
            "fraction": "0.01-0.95",
            "kind": list(KINDS),
        },
        "output_is": (
            "one curve point: fraction, elements_destroyed and the hidden-state "
            "cosine two layers downstream, at the named kind"
        ),
        "measured_level": "LOCAL_FUNCTIONAL_FIDELITY",
        "not_capability": (
            "hidden-state cosine two layers downstream is not capability. "
            "FIDELITY_HIERARCHY forbids refusing a capability claim at this "
            "bar, and CAPABILITY_STAGES is where the deeper rungs live."
        ),
        "matched_by_construction": (
            "fraction selects the same share of rows or columns in every kind, "
            "so two kinds at one fraction touch the same element count. Across "
            "DIFFERENT-SIZED tensors it does not, which is why "
            "elements_destroyed is reported on every point."
        ),
        "authority": (
            "gate/up/down on a real layer only. Damage is applied to an "
            "in-memory copy; no artifact is written."
        ),
        "kinds": {
            "zero": "delete the slice - can this information be REMOVED",
            "quantize": "2-bit uniform within the slice's own range - can it be "
                        "CHEAPER, which is the question a smaller representation "
                        "actually asks",
            "noise": "Gaussian at the slice's own scale, so magnitude is not "
                     "confounded with importance",
            "low_rank": "best rank-10% approximation of the slice - is the "
                        "information redundant rather than absent",
        },
        "why_three_were_missing": (
            "only zero existed, and zero can only answer whether information "
            "can be DELETED. A route to 2.0 BPW needs to know whether it can be "
            "CHEAPER, and those are different questions."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--tensor")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--side", default="rows")
    ap.add_argument("--fraction", type=float)
    ap.add_argument("--kind", default=DEFAULT_KIND, choices=list(KINDS))
    ap.add_argument("--contract", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    if a.contract or a.build:
        doc = {"obligation": "G119", **contract()}
        if a.build:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from _common import REPO, write_receipt
            print(write_receipt(
                REPO / "receipts" / "future" / "COMPONENT_PERTURBATION.json",
                doc, "tools/future/perturbation_workunit.py"))
        else:
            print(json.dumps(doc, indent=1))
        return 0
    if a.tensor is None or a.layer is None or a.fraction is None:
        raise PerturbRefused("--tensor, --layer and --fraction are required")
    print(json.dumps(run(a.tensor, a.layer, a.side, a.fraction, a.kind)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
