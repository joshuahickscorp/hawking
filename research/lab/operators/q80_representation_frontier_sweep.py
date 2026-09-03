#!/usr/bin/env python3
"""Matched binary vs uniform-Qn sweep on Q80 experts, against the measured 0.8604 bar.

The low-rank family is refuted at this operating point (measured: every rung clearing 0.8604
costs >= 2.03 expert BPW against a 1.3012 allowance, while the binary incumbent reaches 0.8926
far cheaper). So the open question is whether a *cheaper* family converts bits to cosine well
enough to lift up_proj over the bar inside the allowance.

The prescription's apparent "2-bit is worse than 1-bit" is confounded: the allocator assigns
2 bits only to the organs it judges hardest. This runs 1-bit and 2-bit on the SAME organs,
which is the matched comparison that claim needs.

Uses the X cache written by q80_rank_bits_sweep.py, so it costs seconds, not the 15-minute
1.38 GB JSON parse.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lab.operators.ascension_dual_gravity_worker import _mean_row_cosine
from lab.operators.doctor6.rungs import quant_binary, quant_uniform, quant_residual
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map

MODEL_DIR = Path(os.environ["MODEL_DIR"])
XCACHE = Path(os.environ["XCACHE"]) if os.environ.get("XCACHE") else Path("/tmp/q80_sweep_xcache.npz")
BAR = 0.8604
ALLOWANCE = 1.3011578470468521

# Historical pairs used for gate/up. down_proj needs the post-SwiGLU
# intermediate (512-dim), which this file now collects via x_kind.
LE_PAIRS = [(10, 453), (3, 494)]
COMPONENTS = ("gate_proj", "up_proj", "down_proj")

# complete_bpw = 0.97032 * expert + 0.02968 * nonexpert <= 1.5
F_EXPERT = 0.9703169371044981
F_NONEXPERT = 0.029683062895501933
ALLOWANCE_8BIT_NONEXPERT = (1.5 - F_NONEXPERT * 8.0) / F_EXPERT  # 1.3012
ALLOWANCE_4BIT_NONEXPERT = (1.5 - F_NONEXPERT * 4.0) / F_EXPERT  # 1.4235
RESIDUAL_FRACS = (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03)
HGRAVS01_POINTS = ((192, 3), (256, 3))


def score(W, X, rec, nbytes):
    bpw = 8.0 * nbytes / max(W.size, 1)
    cos = _mean_row_cosine(X @ W.T, X @ rec.astype(np.float32).T)
    return float(cos), float(bpw)


def _verdict(cos, bpw, allowance):
    ok, aff = cos >= BAR, bpw <= allowance
    if ok and aff:
        return "PASS"
    if ok:
        return "over-budget"
    return "fail"


def _load_or_collect_x(kind: str, wanted: set[tuple[int, int]]):
    """kind is 'router_input' or 'swiglu_hidden_routed'."""
    cache_env = "XCACHE" if kind == "router_input" else "XCACHE_SWIGLU"
    default = (
        "/tmp/q80_sweep_xcache.npz"
        if kind == "router_input"
        else "/tmp/q80_sweep_xcache_swiglu.npz"
    )
    cache = Path(os.environ.get(cache_env, default))
    if cache.exists():
        print(f"[sweep] loading {kind} X from {cache}", flush=True)
        z = np.load(cache)
        by_le = {(int(k.split("_")[0]), int(k.split("_")[1])): z[k] for k in z.files}
        return by_le
    capture = os.environ.get("CAPTURE")
    if not capture:
        if kind == "router_input" and XCACHE.exists():
            z = np.load(XCACHE)
            return {(int(k.split("_")[0]), int(k.split("_")[1])): z[k] for k in z.files}
        raise SystemExit(f"need {cache_env} or CAPTURE to materialize {kind} X")
    from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
        collect_expert_activations,
    )
    print(f"[sweep] collecting {kind} from {capture}", flush=True)
    by_le, _prov = collect_expert_activations(
        Path(capture), wanted_keys=wanted, x_kind=kind
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f"{L}_{E}": v for (L, E), v in by_le.items()})
    print(f"[sweep] cached {kind} X to {cache}", flush=True)
    return by_le


def _select_pairs(by_swiglu: dict, min_pairs: int = 4) -> list[tuple[int, int]]:
    env = os.environ.get("LE_PAIRS")
    if env:
        pairs = []
        for tok in env.split(","):
            a, b = tok.split(":")
            pairs.append((int(a), int(b)))
        return pairs
    present = [(k, int(np.asarray(v).shape[0])) for k, v in by_swiglu.items()]
    present.sort(key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    if not present:
        return list(LE_PAIRS)
    # Prefer a spread of layers among the busiest organs.
    picked: list[tuple[int, int]] = []
    used_layers: set[int] = set()
    for (layer, expert), _n in present:
        if layer in used_layers:
            continue
        picked.append((layer, expert))
        used_layers.add(layer)
        if len(picked) >= min_pairs:
            return picked
    for (layer, expert), _n in present:
        if (layer, expert) not in picked:
            picked.append((layer, expert))
        if len(picked) >= min_pairs:
            break
    return picked


def main():
    wanted: set[tuple[int, int]] | None
    env_pairs = os.environ.get("LE_PAIRS")
    if env_pairs:
        wanted = set()
        for tok in env_pairs.split(","):
            a, b = tok.split(":")
            wanted.add((int(a), int(b)))
    else:
        wanted = None

    capture = os.environ.get("CAPTURE")
    if capture:
        from lab.operators.ascension_qwen30_activation_weighted_svd_repack import (
            collect_expert_activations,
        )
        by_swiglu, sw_prov = collect_expert_activations(
            Path(capture), wanted_keys=wanted, x_kind="swiglu_hidden_routed"
        )
        # Cache for reuse.
        sw_cache = Path(os.environ.get("XCACHE_SWIGLU", "/tmp/q80_sweep_xcache_swiglu.npz"))
        np.savez(sw_cache, **{f"{L}_{E}": v for (L, E), v in by_swiglu.items()})
    else:
        by_swiglu = _load_or_collect_x("swiglu_hidden_routed", wanted or set(LE_PAIRS))
        sw_prov = {"source": "xcache"}

    pairs = _select_pairs(by_swiglu, min_pairs=4)
    print(f"[sweep] scoring pairs {pairs}", flush=True)

    wmap = load_weight_map(MODEL_DIR)
    out = {
        "bar": BAR,
        "expert_bpw_allowance_8bit_nonexpert": ALLOWANCE_8BIT_NONEXPERT,
        "expert_bpw_allowance_4bit_nonexpert": ALLOWANCE_4BIT_NONEXPERT,
        "identity": f"complete_bpw = {F_EXPERT}*expert_bpw + {F_NONEXPERT}*nonexpert_bpw <= 1.5",
        "swiglu_formula": "silu(x @ gate_proj.T) * (x @ up_proj.T)",
        "hidden_act": "silu",
        "pairs": [list(p) for p in pairs],
        "activation_provenance": {
            k: sw_prov.get(k)
            for k in (
                "x_kind",
                "packed_swiglu",
                "token_expert_pairs",
                "layer_expert_pairs_with_hits",
                "swiglu_width",
            )
            if isinstance(sw_prov, dict)
        },
        "organs": [],
    }

    from lab.operators.hgravs01_adapter import encode_hgravs01

    for (layer, expert) in pairs:
        X = np.asarray(by_swiglu[(layer, expert)], dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != 512:
            raise SystemExit(
                f"down_proj X for L{layer}.E{expert} has shape {X.shape}; expected (N, 512)"
            )
        key = f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
        W = np.asarray(load_tensor(MODEL_DIR, wmap, key), dtype=np.float32)
        rec_out = {
            "organ_key": key,
            "component": "down_proj",
            "layer": layer,
            "expert": expert,
            "rows": int(X.shape[0]),
            "X_shape": list(X.shape),
            "W_shape": list(W.shape),
            "rank_cap": int(min(X.shape[0], W.shape[0], W.shape[1])),
            "candidates": [],
        }
        print(
            f"\n=== {key}  rows={X.shape[0]}  W={tuple(W.shape)}  X={tuple(X.shape)}",
            flush=True,
        )

        trials: list[tuple[str, object]] = [("binary_g", lambda W=W: quant_binary(W))]
        for frac in RESIDUAL_FRACS:
            label = (
                f"binary+resid_{frac*100:.2f}pct".replace(".00", "")
                if frac * 100 == int(frac * 100)
                else f"binary+resid_{frac*100:g}pct"
            )
            # pretty labels: 0.25 / 0.5 / 1 / 1.5 / 2 / 3
            pct = frac * 100.0
            if abs(pct - round(pct)) < 1e-12:
                label = f"binary+resid_{int(round(pct))}pct"
            elif abs(pct - 0.25) < 1e-12:
                label = "binary+resid_0.25pct"
            elif abs(pct - 0.5) < 1e-12:
                label = "binary+resid_0.5pct"
            elif abs(pct - 1.5) < 1e-12:
                label = "binary+resid_1.5pct"
            trials.append((label, lambda W=W, f=frac: quant_residual(W, outlier_ratio=f)))
        for b in (2, 3):
            trials.append((f"uniform_b{b}", lambda W=W, b=b: quant_uniform(W, bits=b)))

        for name, fn in trials:
            try:
                rec, nbytes = fn()
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<24} ERROR {exc}", flush=True)
                rec_out["candidates"].append({"codec": name, "error": str(exc)})
                continue
            cos, bpw = score(W, X, rec, nbytes)
            rec_out["candidates"].append({
                "codec": name,
                "cosine": cos,
                "expert_bpw": bpw,
                "clears_bar": cos >= BAR,
                "fits_1.3012": bpw <= ALLOWANCE_8BIT_NONEXPERT,
                "fits_1.4235": bpw <= ALLOWANCE_4BIT_NONEXPERT,
                "verdict_vs_1.3012": _verdict(cos, bpw, ALLOWANCE_8BIT_NONEXPERT),
                "verdict_vs_1.4235": _verdict(cos, bpw, ALLOWANCE_4BIT_NONEXPERT),
            })
            print(
                f"  {name:<24} cos={cos:.4f} bpw={bpw:.4f}  "
                f"bar={'Y' if cos >= BAR else 'n'}  "
                f"1.3012={'Y' if bpw <= ALLOWANCE_8BIT_NONEXPERT else 'n'}  "
                f"1.4235={'Y' if bpw <= ALLOWANCE_4BIT_NONEXPERT else 'n'}",
                flush=True,
            )

        for rank, bits in HGRAVS01_POINTS:
            name = f"hgravs01_r{rank}_b{bits}"
            try:
                encoded = encode_hgravs01(W, X, rank=rank, bits=bits)
                rec = np.asarray(encoded["W_hat"], dtype=np.float32)
                nbytes = int(encoded["payload_bytes"])
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<24} ERROR {exc}", flush=True)
                rec_out["candidates"].append({"codec": name, "error": str(exc)})
                continue
            cos, bpw = score(W, X, rec, nbytes)
            rec_out["candidates"].append({
                "codec": name,
                "cosine": cos,
                "expert_bpw": bpw,
                "clears_bar": cos >= BAR,
                "fits_1.3012": bpw <= ALLOWANCE_8BIT_NONEXPERT,
                "fits_1.4235": bpw <= ALLOWANCE_4BIT_NONEXPERT,
                "verdict_vs_1.3012": _verdict(cos, bpw, ALLOWANCE_8BIT_NONEXPERT),
                "verdict_vs_1.4235": _verdict(cos, bpw, ALLOWANCE_4BIT_NONEXPERT),
                "requested_rank": rank,
                "achieved_rank": int(encoded.get("achieved_rank") or rank),
                "rank_clamped_to_n_fit": bool(encoded.get("rank_clamped_to_n_fit")),
                "n_fit_rows": int(encoded.get("n_fit_rows") or X.shape[0]),
            })
            print(
                f"  {name:<24} cos={cos:.4f} bpw={bpw:.4f}  "
                f"bar={'Y' if cos >= BAR else 'n'}  "
                f"1.3012={'Y' if bpw <= ALLOWANCE_8BIT_NONEXPERT else 'n'}  "
                f"1.4235={'Y' if bpw <= ALLOWANCE_4BIT_NONEXPERT else 'n'}  "
                f"rank={encoded.get('achieved_rank')}/{rank}",
                flush=True,
            )
        out["organs"].append(rec_out)

    dest = os.environ.get("OUT", "/tmp/q80_down_proj_frontier_sweep.json")
    Path(dest).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
