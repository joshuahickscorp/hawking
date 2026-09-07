"""OI representational anatomy: how does this organism want to be represented?

S012 §11 requires this BEFORE quantization is attempted, not after it fails.
Everything here was validated on two specimens: it predicted the shared-basis
NR's collapse (ppl 27.77) from spectra alone, and the second specimen's answer
arrived in five minutes instead of hours.

Reads safetensors directly rather than loading a model, so a partially
downloaded specimen still yields an anatomy -- mlx_lm's loader refuses those
outright, and the spectra need one complete expert group, not a runnable model.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import struct
from typing import Any

import mlx.core as mx


class AnatomyUnavailable(RuntimeError):
    """No anatomy could be measured, and saying so is the only honest answer.

    G035. The predecessor of this class was a silent empty dict: a specimen with
    no .safetensors made every loop body skip, and the caller received a
    well-formed result with cross_expert [] and hypotheses [] at exit code 0.
    A campaign driver would have recorded that body as ANATOMISED having
    measured nothing. Four specimens in the lake are in exactly that class
    (evo2_40b, boltz-2, two mamba3 bodies -- .bin, no safetensors), so this was
    not hypothetical: it was 4 of 56 records that would have been fiction.
    """

# Thresholds from the two-specimen family prior. Stated so they can be falsified.
SHARING_LIVE_BELOW = 0.95      # cross-expert participation ratio
LOWRANK_LIVE_BELOW = 0.40      # within-expert participation ratio

_EXPERT_RE = re.compile(r"\.layers\.(\d+)\..*experts?\.(\d+)\.(\w+proj)\.weight$")

# A body 15x larger than RAM must still be anatomisable (G035), so the shard scan
# reads HEADERS ONLY: 8 bytes little-endian u64 header length, then that many bytes
# of JSON mapping tensor name -> {dtype, shape, data_offsets}. Shapes and dtypes
# therefore cost kilobytes per shard instead of gigabytes, and only the shards that
# actually hold the target organ are ever opened for payload.
_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
                "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}


def read_header(path: str) -> dict:
    """Tensor name -> {dtype, shape, data_offsets} without touching the payload."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise AnatomyUnavailable(f"{path}: shorter than a safetensors header")
        n = struct.unpack("<Q", raw)[0]
        if not 0 < n < (1 << 31):
            raise AnatomyUnavailable(f"{path}: implausible header length {n}; not safetensors")
        hdr = json.loads(fh.read(n))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}


def _nbytes(entry: dict) -> int:
    n = 1
    for d in entry.get("shape") or []:
        n *= int(d)
    return n * _DTYPE_BYTES.get(entry.get("dtype", ""), 2)


def _spectrum(X: "mx.array") -> dict:
    Xc = X - mx.mean(X, axis=0, keepdims=True)
    Gm = (Xc @ Xc.T).astype(mx.float32) / max(Xc.shape[1], 1)
    ev = mx.maximum(mx.sort(mx.linalg.eigvalsh(Gm, stream=mx.cpu))[::-1], 0.0)
    tot = float(mx.sum(ev))
    part = ev / max(tot, 1e-30)
    ent = float(-mx.sum(mx.where(part > 0, part * mx.log(mx.maximum(part, 1e-30)), 0.0)))
    cum = mx.cumsum(ev) / max(tot, 1e-30)
    n = int(X.shape[0])
    p = float(mx.exp(mx.array(ent)).item())
    return {"n": n, "participation": round(p, 2), "ratio": round(p / n, 4),
            "rank_90": int(mx.sum(cum < 0.90).item()) + 1,
            "top1_share": round(float(ev[0] / max(tot, 1e-30)), 4)}


def _effective_rank(W: "mx.array") -> dict:
    s = mx.linalg.svd(W.astype(mx.float32), compute_uv=False, stream=mx.cpu)
    s2 = s * s
    tot = float(mx.sum(s2))
    part = s2 / max(tot, 1e-30)
    ent = float(-mx.sum(mx.where(part > 0, part * mx.log(mx.maximum(part, 1e-30)), 0.0)))
    cum = mx.cumsum(s2) / max(tot, 1e-30)
    full = int(min(W.shape))
    p = float(mx.exp(mx.array(ent)).item())
    return {"shape": list(W.shape), "full_rank": full,
            "participation": round(p, 1), "ratio": round(p / full, 4),
            "rank_90": int(mx.sum(cum < 0.90).item()) + 1}


def hypotheses(anatomy: dict) -> list[str]:
    """What the numbers license, in plain terms. This is the OI deliverable."""
    out = []
    ce = anatomy.get("cross_expert") or []
    if ce:
        r = min(x["ratio"] for x in ce)
        if r >= SHARING_LIVE_BELOW:
            out.append(f"cross-expert sharing is DEAD (ratio {r:.4f} >= {SHARING_LIVE_BELOW}): "
                       "skip shared basis, common+delta, clustering, alignment")
        else:
            out.append(f"cross-expert sharing is LIVE (ratio {r:.4f} < {SHARING_LIVE_BELOW}): "
                       "a shared basis may pay -- this specimen contradicts the family prior")
    we = anatomy.get("within_expert")
    if we:
        if we["ratio"] >= LOWRANK_LIVE_BELOW:
            out.append(f"low-rank factorisation is DEAD (ratio {we['ratio']:.4f} >= "
                       f"{LOWRANK_LIVE_BELOW}): a rank-{we['rank_90']} factor costs more than dense")
        else:
            out.append(f"low-rank is LIVE (ratio {we['ratio']:.4f}): rank-{we['rank_90']} may pay")
    cl = anatomy.get("cross_layer")
    if cl:
        verdict = "DEAD" if cl["ratio"] >= SHARING_LIVE_BELOW else "LIVE"
        out.append(f"cross-layer sharing is {verdict} (ratio {cl['ratio']:.4f})")
    if all("DEAD" in h for h in out) and out:
        out.append("NO LINEAR STRUCTURE AVAILABLE: go non-linear (generated coefficients, "
                   "procedural reconstruction, router-conditioned representation) or accept "
                   "quantization as the ceiling for this organism")
    return out


def anatomy_from_safetensors(snapshot: str, layer: int = 0, max_proj: int = 3,
                             max_group_gb: float = 8.0) -> dict:
    """Anatomy of one layer's expert organ, read organ-by-organ from headers.

    Raises AnatomyUnavailable rather than returning an empty anatomy, and never
    holds more than one shard's payload, so a body larger than RAM is reachable.
    """
    snapshot = snapshot.rstrip("/")
    shards = sorted(glob.glob(snapshot + "/*.safetensors"))
    if not shards:
        others = collections.Counter(
            os.path.splitext(f)[1] or "<noext>" for f in os.listdir(snapshot)
        ) if os.path.isdir(snapshot) else None
        if others is None:
            raise AnatomyUnavailable(f"{snapshot}: not a directory")
        raise AnatomyUnavailable(
            f"{snapshot}: no .safetensors weights. Present: "
            + ", ".join(f"{n}x{e}" for e, n in others.most_common(6))
            + ". This body needs format conversion before OI can measure it."
        )

    # Pass 1 -- headers only. Kilobytes per shard even on a 1.4 TiB body.
    index: dict[str, tuple[str, dict]] = {}
    for f in shards:
        for k, e in read_header(f).items():
            index[k] = (f, e)
    if not index:
        raise AnatomyUnavailable(f"{snapshot}: {len(shards)} shards, zero tensors in their headers")

    groups: dict[str, dict[int, str]] = collections.defaultdict(dict)
    stacked: list[str] = []
    layers_seen: set[int] = set()
    for k, (f, e) in index.items():
        m = _EXPERT_RE.search(k)
        if m:
            layers_seen.add(int(m.group(1)))
            if int(m.group(1)) == layer:
                groups[m.group(3)][int(m.group(2))] = k
        elif len(e.get("shape") or []) == 3 and (e["shape"][0] or 0) >= 8 and "expert" in k:
            stacked.append(k)

    if not groups and not stacked:
        expert_keys = [k for k in index if "expert" in k.lower()]
        detail = (f"{len(expert_keys)} keys contain 'expert' but none matched the "
                  f"per-expert or stacked layout (e.g. {expert_keys[0]})"
                  if expert_keys else
                  f"no key contains 'expert'; this looks dense, not MoE "
                  f"(e.g. {sorted(index)[0]})")
        if layers_seen:
            detail += f"; experts exist at layers {min(layers_seen)}..{max(layers_seen)}, not {layer}"
        raise AnatomyUnavailable(
            f"{snapshot}: {len(index)} tensors across {len(shards)} shards, but no expert "
            f"organ is measurable at layer {layer}. {detail}"
        )

    # Pass 2 -- payload, one shard resident at a time.
    held: dict[str, Any] = {}

    def tensor(key: str):
        f = index[key][0]
        if f not in held:
            held.clear()               # organ-by-organ: never two shards at once
            held[f] = mx.load(f)
        return held[f][key]

    out: dict[str, Any] = {"snapshot": snapshot, "layer": layer,
                           "n_shards": len(shards), "n_tensors": len(index),
                           "storage": "per-expert" if groups else "stacked",
                           "cross_expert": []}

    if groups:
        for proj, d in sorted(groups.items())[:max_proj]:
            ids = sorted(d)
            need = sum(_nbytes(index[d[e]][1]) for e in ids) * 2  # bf16 payload -> f32 stack
            n_use = len(ids)
            if need > max_group_gb * 2 ** 30:
                n_use = max(8, int(len(ids) * max_group_gb * 2 ** 30 / need))
                ids = ids[:n_use]
            # Group the loads by shard so each shard is opened once, not once per expert.
            ids.sort(key=lambda e: index[d[e]][0])
            X = mx.stack([tensor(d[e]).astype(mx.float32).reshape(-1) for e in ids])
            r = _spectrum(X)
            r["tensor"] = proj
            r["n_experts_total"] = len(d)
            if n_use < len(d):
                r["subsampled"] = True
            out["cross_expert"].append(r)
            del X
        first = groups[sorted(groups)[0]]
        out["within_expert"] = _effective_rank(tensor(first[sorted(first)[0]]).astype(mx.float32))
    else:
        stacked.sort(key=lambda k: index[k][0])
        for k in stacked[:max_proj]:
            v = tensor(k)
            r = _spectrum(v.astype(mx.float32).reshape(v.shape[0], -1))
            r["tensor"] = k.split(".")[-2]
            r["n_experts_total"] = int(v.shape[0])
            out["cross_expert"].append(r)
        out["within_expert"] = _effective_rank(tensor(stacked[0])[0].astype(mx.float32))

    held.clear()
    out["hypotheses"] = hypotheses(out)
    if not out["hypotheses"]:
        raise AnatomyUnavailable(
            f"{snapshot}: spectra computed but yielded no hypothesis -- refusing to "
            f"record an anatomy that says nothing"
        )
    return out
