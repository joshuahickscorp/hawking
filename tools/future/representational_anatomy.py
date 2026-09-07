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
from typing import Any, NamedTuple

import numpy as np


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
# A DEFICIT against a matched null, in percent -- not a ratio against a constant.
# The old SHARING_LIVE_BELOW = 0.95 was the noise floor at n=20 and called pure
# noise LIVE below it; the old LOWRANK_LIVE_BELOW = 0.40 was compared against a
# null that moves with aspect ratio (0.606 at 1024x1024, 0.847 at 3072x1024).
SHARING_LIVE_DEFICIT_PCT = 2.0
LOWRANK_LIVE_DEFICIT_PCT = 10.0

# Retained so the published-thresholds test still names what was superseded.
SHARING_LIVE_BELOW = 0.95      # SUPERSEDED: noise floor is (n-1)/n
LOWRANK_LIVE_BELOW = 0.40      # SUPERSEDED: the null moves with aspect ratio

# A spectrum over integer/fp8 codes measures the codebook, not the organism.
FLOAT_DTYPES = frozenset({"BF16", "F16", "F32", "F64"})

# Each Gram feature-block is kept at or under this many bytes of float32.
GRAM_BLOCK_BYTES = 256 * 1024 * 1024

_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale", ".scale")


class ExpertKey(NamedTuple):
    """One expert payload tensor, identified from its safetensors key alone."""
    layer: int
    expert_id: int | None  # None => stacked, expert axis is 0 of a 3D tensor
    projection: str
    scheme: str


# Ordered table: first match wins. A seventh layout is one more row.
# Named groups: layer, proj, and expert (absent => stacked).
# `(?:^|\.)layers.` accepts both `model.layers.0` and DeepSeek's `layers.0`.
# `shared_expert(s)` is a different organ and is filtered before this table.
_SCHEMES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "E",
        re.compile(
            r"(?:^|\.)layers\.(?P<layer>\d+)\..*?(?<!shared_)experts?\."
            r"(?P<expert>\d+)\.(?P<proj>\w+)\.weight_packed$"
        ),
    ),
    (
        "D",
        re.compile(
            r"(?:^|\.)layers\.(?P<layer>\d+)\..*?(?<!shared_)experts?\."
            r"(?P<expert>\d+)\.(?P<proj>w\d+)\.weight$"
        ),
    ),
    (
        "A",
        re.compile(
            r"(?:^|\.)layers\.(?P<layer>\d+)\..*?(?<!shared_)experts?\."
            r"(?P<expert>\d+)\.(?P<proj>\w+)\.weight$"
        ),
    ),
    (
        "C",
        re.compile(
            r"(?:^|\.)layers\.(?P<layer>\d+)\..*?(?<!shared_)experts?\."
            r"(?P<proj>w\d+_weight)$"
        ),
    ),
    (
        "B",
        re.compile(
            r"(?:^|\.)layers\.(?P<layer>\d+)\..*?(?<!shared_)experts?\."
            r"(?P<proj>[A-Za-z]\w*)$"
        ),
    ),
)

# A body 15x larger than RAM must still be anatomisable (G035), so the shard scan
# reads HEADERS ONLY: 8 bytes little-endian u64 header length, then that many bytes
# of JSON mapping tensor name -> {dtype, shape, data_offsets}. Shapes and dtypes
# therefore cost kilobytes per shard instead of gigabytes, and only the shards that
# actually hold the target organ are ever opened for payload.
_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
                "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
                "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}


def _read_header_ex(path: str) -> tuple[dict, int]:
    """Header dict plus the JSON header length (payload starts at 8 + that)."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise AnatomyUnavailable(f"{path}: shorter than a safetensors header")
        n = struct.unpack("<Q", raw)[0]
        if not 0 < n < (1 << 31):
            raise AnatomyUnavailable(f"{path}: implausible header length {n}; not safetensors")
        hdr = json.loads(fh.read(n))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}, n


def read_header(path: str) -> dict:
    """Tensor name -> {dtype, shape, data_offsets} without touching the payload."""
    hdr, _ = _read_header_ex(path)
    return hdr


def _load_float_tensor(path: str, header_len: int, entry: dict) -> np.ndarray:
    """Read ONE tensor's payload as float32. Never maps the rest of the shard.

    BF16 is expanded by bit-shift (high 16 bits of f32), which is how the pinned
    Qwen3-30B-A3B receipt was reproduced without Metal.
    """
    dt = entry.get("dtype", "")
    if dt not in FLOAT_DTYPES:
        raise AnatomyUnavailable(f"{path}: refusing to decode non-float dtype {dt}")
    start, end = entry["data_offsets"]
    shape = tuple(int(d) for d in (entry.get("shape") or ()))
    n_el = 1
    for d in shape:
        n_el *= d
    elem = _DTYPE_BYTES[dt]
    # Stream ~1M elements at a time so a BF16 tensor is never held as
    # raw + u32 + f32 together (that triple copy put Inkling over 40 GiB).
    out = np.empty(n_el, dtype=np.float32)
    step = 1 << 20
    with open(path, "rb") as fh:
        fh.seek(8 + header_len + start)
        for s in range(0, n_el, step):
            e = min(n_el, s + step)
            chunk = fh.read((e - s) * elem)
            if dt == "BF16":
                bits = np.frombuffer(chunk, dtype="<u2").astype(np.uint32) << 16
                out[s:e] = bits.view(np.float32)
            elif dt == "F16":
                out[s:e] = np.frombuffer(chunk, dtype="<f2").astype(np.float32)
            elif dt == "F32":
                out[s:e] = np.frombuffer(chunk, dtype="<f4")
            else:
                out[s:e] = np.frombuffer(chunk, dtype="<f8").astype(np.float32)
    return out.reshape(shape)


def _nbytes(entry: dict) -> int:
    n = 1
    for d in entry.get("shape") or []:
        n *= int(d)
    return n * _DTYPE_BYTES.get(entry.get("dtype", ""), 2)


def parse_expert_key(key: str) -> ExpertKey | None:
    """Map a tensor name to (layer, expert_id, projection, scheme), or None.

    Scale tensors and the shared-expert organ are not the routed expert group.
    """
    if "shared_expert" in key:
        return None
    if key.endswith(_SCALE_SUFFIXES):
        return None
    for name, rx in _SCHEMES:
        m = rx.search(key)
        if m:
            gd = m.groupdict()
            eid = gd.get("expert")
            return ExpertKey(int(gd["layer"]), None if eid is None else int(eid),
                             gd["proj"], name)
    return None


def scan_expert_index(index: dict[str, Any]) -> tuple[
        dict[int, dict[str, dict[int | None, str]]],
        dict[int, set[str]],
        bool]:
    """Header pass: group expert payload keys by layer and projection.

    Returns (by_layer[layer][proj][expert_id] = key, schemes_at, shared_expert_present).
    Scale tensors and the shared-expert organ are excluded from by_layer.
    """
    by_layer: dict[int, dict[str, dict[int | None, str]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    schemes_at: dict[int, set[str]] = collections.defaultdict(set)
    shared_expert_present = False
    for k in index:
        if "shared_expert" in k:
            shared_expert_present = True
            continue
        parsed = parse_expert_key(k)
        if parsed is None:
            continue
        by_layer[parsed.layer][parsed.projection][parsed.expert_id] = k
        schemes_at[parsed.layer].add(parsed.scheme)
    return by_layer, schemes_at, shared_expert_present


def resolve_layer(by_layer: dict[int, Any], layer: int | None) -> int:
    """Pick the measured layer. None => lowest layer with a complete expert group."""
    layers_seen = sorted(by_layer)
    if not layers_seen:
        raise KeyError("no expert layers")
    if layer is None:
        return layers_seen[0]
    if layer not in by_layer:
        raise KeyError(layer)
    return layer


def _companion_scale(index: dict[str, Any], key: str) -> str | None:
    """Name of the scale tensor that sits alongside a pre-quantized payload, if any."""
    stems: list[str] = []
    for suf in (".weight_packed", ".weight"):
        if key.endswith(suf):
            stems.append(key[: -len(suf)])
    stems.append(key)
    seen: set[str] = set()
    for stem in stems:
        if stem in seen:
            continue
        seen.add(stem)
        for suf in _SCALE_SUFFIXES:
            cand = stem + suf
            if cand in index and cand != key:
                return cand
    return None


def _gram(Xc: np.ndarray, gram_block_bytes: int | None) -> np.ndarray:
    """Xc @ Xc.T, accumulated along the feature axis in bounded blocks.

    The previous mlx single-matmul of Xc (512, 3_276_800) tripped Metal's
    watchdog (~880 GFLOP, one command buffer). Host GEMM in <=256 MB slices
    is the same algebra and does not depend on a GPU.

    gram_block_bytes is the max float32 footprint of one Xc[:, s:e] slice.
    None (or <= 0) computes the matmul in one shot -- used to prove the chunked
    path matches the unchunked path.
    """
    n, f = int(Xc.shape[0]), int(Xc.shape[1])
    Xc32 = np.asarray(Xc, dtype=np.float32)
    if gram_block_bytes is None or gram_block_bytes <= 0:
        return Xc32 @ Xc32.T
    cols = max(1, gram_block_bytes // (4 * max(n, 1)))
    Gm = np.zeros((n, n), dtype=np.float32)
    for s in range(0, f, cols):
        blk = Xc32[:, s:min(f, s + cols)]
        Gm = Gm + blk @ blk.T
    return Gm


def _null_participation(norms: np.ndarray, d: int, gram_block_bytes: int | None,
                        seed: int = 5) -> float:
    """Participation of a Gaussian null carrying `norms`, built without X.

    Centred INDEPENDENT rows have participation exactly n-1, so ratio = p/n
    scores (n-1)/n on pure noise: 0.750 at n=4, 0.950 at n=20, 0.992 at n=128.
    SHARING_LIVE_BELOW is 0.95, which IS that floor at n=20, so a constant
    threshold reports "sharing is LIVE" on noise for every n <= 20.

    Row-norm heterogeneity depresses the ratio further with no shared direction
    anywhere, so the null must carry the SAME per-row norms. An unmatched
    Gaussian scores 0.9921 where O003's experts score 0.9796 and would license a
    1.27% "shared structure" claim that is entirely an artefact of unequal row
    magnitudes.

    Built from the norms and the shape alone so the real stack can be freed
    first: holding both doubles peak memory for nothing.
    """
    n = int(norms.shape[0])
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, d), dtype=np.float32)
    R *= (norms / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-30))
    R -= R.mean(axis=0, keepdims=True)
    Gm = _gram(R, gram_block_bytes) / max(d, 1)
    del R
    ev = np.maximum(np.linalg.eigvalsh(Gm.astype(np.float64))[::-1], 0.0)
    tot = float(ev.sum())
    part = ev / max(tot, 1e-30)
    pos = part > 0
    return float(np.exp(-(part[pos] * np.log(np.maximum(part[pos], 1e-30))).sum()))


def _spectrum(X: np.ndarray, gram_block_bytes: int | None = GRAM_BLOCK_BYTES,
              with_null: bool = True) -> dict:
    Xc = np.asarray(X, dtype=np.float32)
    if not Xc.flags.writeable:
        Xc = np.array(Xc, dtype=np.float32, copy=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True) if with_null else None
    # In-place center: a second 17 GiB Xc is what put Inkling at 40 GiB RSS.
    Xc -= Xc.mean(axis=0, keepdims=True)
    Gm = _gram(Xc, gram_block_bytes) / max(int(Xc.shape[1]), 1)
    del Xc
    # n x n eigensolve in float64; the Gram itself stays float32. This is the
    # combination that reproduced the pinned Qwen3-30B-A3B receipt exactly.
    ev = np.linalg.eigvalsh(Gm.astype(np.float64))[::-1]
    ev = np.maximum(ev, 0.0)
    tot = float(ev.sum())
    part = ev / max(tot, 1e-30)
    pos = part > 0
    ent = float(-(part[pos] * np.log(np.maximum(part[pos], 1e-30))).sum())
    p = float(np.exp(ent))
    null_p = (_null_participation(norms, int(X.shape[1]), gram_block_bytes)
              if with_null else 0.0)
    cum = np.cumsum(ev) / max(tot, 1e-30)
    n = int(X.shape[0])
    out = {"n": n, "participation": round(p, 2), "ratio": round(p / n, 4),
           "rank_90": int((cum < 0.90).sum()) + 1,
           "top1_share": round(float(ev[0] / max(tot, 1e-30)), 4),
           "noise_floor_ratio": round((n - 1) / n, 4)}
    if with_null:
        out["null_ratio"] = round(null_p / n, 4)
        out["deficit_pct"] = round(100 * (null_p - p) / max(null_p, 1e-30), 3)
    return out


def _rank_stats(W: np.ndarray) -> tuple[float, int, int]:
    s = np.linalg.svd(np.asarray(W, dtype=np.float32), compute_uv=False)
    s2 = s * s
    tot = float(s2.sum())
    part = s2 / max(tot, 1e-30)
    pos = part > 0
    ent = float(-(part[pos] * np.log(np.maximum(part[pos], 1e-30))).sum())
    cum = np.cumsum(s2) / max(tot, 1e-30)
    return float(np.exp(ent)), int((cum < 0.90).sum()) + 1, int(min(W.shape))


def _effective_rank(W: np.ndarray, with_null: bool = True, seed: int = 5) -> dict:
    """Effective rank, and what a RANDOM matrix of the same shape scores.

    The superseded LOWRANK_LIVE_BELOW = 0.40 was a constant compared against a
    null that MOVES WITH ASPECT RATIO: a random matrix scores 0.606 at
    1024x1024 but 0.847 at 3072x1024. It never false-positives, since every null
    is above 0.60, but it is far too conservative for wide organs -- a 3072x1024
    tensor at 0.45 reads DEAD while sitting 47% below its own null.
    """
    p, r90, full = _rank_stats(W)
    out = {"shape": list(W.shape), "full_rank": full,
           "participation": round(p, 1), "ratio": round(p / full, 4),
           "rank_90": r90}
    if with_null:
        rng = np.random.default_rng(seed)
        R = rng.standard_normal(tuple(int(x) for x in W.shape), dtype=np.float32)
        np_, nr90, _ = _rank_stats(R)
        del R
        out["null_ratio"] = round(np_ / full, 4)
        out["null_rank_90"] = nr90
        out["deficit_pct"] = round(100 * (np_ - p) / max(np_, 1e-30), 3)
    return out


def break_even_rank(shape: list) -> float:
    """Rank below which a factorisation stores fewer numbers than the dense tensor.

    m*n dense against r*(m+n) factored, so the break-even is m*n/(m+n). The
    superseded text asserted "a rank-495 factor costs more than dense" for a
    2048x768 tensor: dense is 1,572,864 numbers, the factors are 1,393,920, so it
    SAVES 11.4%. Whether structure EXISTS and whether factoring PAYS are
    different questions and that threshold answered both at once.
    """
    m, n = int(shape[0]), int(shape[1])
    return m * n / max(m + n, 1)


def hypotheses(anatomy: dict) -> list[str]:
    """What the numbers license, in plain terms. This is the OI deliverable."""
    out = []
    ce = anatomy.get("cross_expert") or []
    if ce:
        worst = max(ce, key=lambda x: x.get("deficit_pct", 0.0))
        dfc = worst.get("deficit_pct")
        if dfc is None:
            r = min(x["ratio"] for x in ce)
            out.append(f"cross-expert sharing UNJUDGED (ratio {r:.4f}, no null measured): "
                       "a ratio without its noise floor decides nothing")
        elif dfc >= SHARING_LIVE_DEFICIT_PCT:
            out.append(f"cross-expert sharing is LIVE ({dfc:.2f}% below a norm-matched null, "
                       f"n={worst['n']}): a shared basis may pay -- this specimen contradicts "
                       "the family prior")
        else:
            out.append(f"cross-expert sharing is DEAD ({dfc:.2f}% from a norm-matched null, "
                       f"n={worst['n']}; the old ratio rule's noise floor here is "
                       f"{worst['noise_floor_ratio']}): skip shared basis, common+delta, "
                       "clustering, alignment")
    we = anatomy.get("within_expert")
    if we:
        dfc, r90, shape = we.get("deficit_pct"), we["rank_90"], we["shape"]
        be = break_even_rank(shape)
        pays = r90 < be
        cost = r90 * (shape[0] + shape[1]) / max(shape[0] * shape[1], 1)
        if dfc is None:
            out.append(f"low-rank UNJUDGED (ratio {we['ratio']:.4f}, no null measured)")
        elif dfc >= LOWRANK_LIVE_DEFICIT_PCT:
            out.append(f"low-rank structure is REAL ({dfc:.2f}% below a matched null); a "
                       f"rank-{r90} factor stores {cost:.3f}x dense "
                       f"({'UNDER' if pays else 'OVER'} the {be:.0f} break-even rank). "
                       f"rank_90 is 90% of spectral ENERGY, not of behaviour: this says the "
                       f"BYTES work, not that capability survives")
        else:
            out.append(f"low-rank structure is ABSENT ({dfc:.2f}% from a matched null): the "
                       f"spectrum is what a random matrix of this shape gives, so a rank-{r90} "
                       f"factor at {cost:.3f}x dense would be compressing noise")
    cl = anatomy.get("cross_layer")
    if cl:
        dfc = cl.get("deficit_pct")
        verdict = "UNJUDGED" if dfc is None else ("LIVE" if dfc >= SHARING_LIVE_DEFICIT_PCT else "DEAD")
        out.append(f"cross-layer sharing is {verdict} "
                   f"({'no null' if dfc is None else f'{dfc:.2f}% from a matched null'})")
    if out and all(("DEAD" in h or "ABSENT" in h) for h in out):
        out.append("NO LINEAR STRUCTURE AVAILABLE: go non-linear (generated coefficients, "
                   "procedural reconstruction, router-conditioned representation) or accept "
                   "quantization as the ceiling for this organism")
    return out


def anatomy_from_safetensors(snapshot: str, layer: int | None = 0,
                             max_proj: int = 3, max_group_gb: float = 8.0) -> dict:
    """Anatomy of one layer's expert organ, read organ-by-organ from headers.

    Raises AnatomyUnavailable rather than returning an empty anatomy, and never
    holds more than one shard's payload, so a body larger than RAM is reachable.
    layer=None selects the lowest layer that actually has a complete expert group.
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
    header_lens: dict[str, int] = {}
    for f in shards:
        hdr, n = _read_header_ex(f)
        header_lens[f] = n
        for k, e in hdr.items():
            index[k] = (f, e)
    if not index:
        raise AnatomyUnavailable(f"{snapshot}: {len(shards)} shards, zero tensors in their headers")

    by_layer, schemes_at, shared_expert_present = scan_expert_index(index)
    layers_seen = sorted(by_layer)
    requested = layer

    if not layers_seen:
        expert_keys = [k for k in index if "expert" in k.lower()]
        detail = (f"{len(expert_keys)} keys contain 'expert' but none matched the "
                  f"per-expert or stacked layout (e.g. {expert_keys[0]})"
                  if expert_keys else
                  f"no key contains 'expert'; this looks dense, not MoE "
                  f"(e.g. {sorted(index)[0]})")
        layer_shown = "any layer" if requested is None else requested
        raise AnatomyUnavailable(
            f"{snapshot}: {len(index)} tensors across {len(shards)} shards, but no expert "
            f"organ is measurable at layer {layer_shown}. {detail}"
        )

    try:
        layer = resolve_layer(by_layer, layer)
    except KeyError:
        # Keep the original range clause -- callers use it to rediscover the organ.
        detail = f"experts exist at layers {layers_seen[0]}..{layers_seen[-1]}, not {requested}"
        raise AnatomyUnavailable(
            f"{snapshot}: {len(index)} tensors across {len(shards)} shards, but no expert "
            f"organ is measurable at layer {requested}. {detail}"
        ) from None

    layer_groups = by_layer[layer]

    # Refuse pre-quantized organs from the header dtype -- never load the codes.
    for proj, emap in sorted(layer_groups.items()):
        for eid, key in sorted(emap.items(), key=lambda kv: (kv[0] is None, kv[0])):
            dt = index[key][1].get("dtype", "")
            if dt not in FLOAT_DTYPES:
                scale = _companion_scale(index, key)
                scale_txt = scale if scale is not None else (
                    "no companion .scale/.weight_scale/.weight_scale_inv"
                )
                raise AnatomyUnavailable(
                    f"{snapshot}: expert organ is PRE-QUANTIZED on disk "
                    f"({proj} is {dt}, with {scale_txt} alongside). "
                    "A spectrum over quantization codes measures the codebook, "
                    "not the organism. Dequantize first or record this body as "
                    "anatomy-blocked-by-format."
                )

    indexed = any(eid is not None for emap in layer_groups.values() for eid in emap)
    storage = "per-expert" if indexed else "stacked"
    scheme_name = ",".join(sorted(schemes_at[layer]))

    # Pass 2 -- payload, ONE TENSOR at a time. A 17 GiB shard is never mapped.
    def tensor(key: str) -> np.ndarray:
        f, e = index[key]
        return _load_float_tensor(f, header_lens[f], e)

    out: dict[str, Any] = {"snapshot": snapshot, "layer": layer,
                           "layers_with_experts": [layers_seen[0], layers_seen[-1]],
                           "shared_expert_present": shared_expert_present,
                           "scheme": scheme_name,
                           "n_shards": len(shards), "n_tensors": len(index),
                           "storage": storage,
                           "cross_expert": []}

    if storage == "per-expert":
        groups = {proj: {int(eid): k for eid, k in emap.items() if eid is not None}
                  for proj, emap in layer_groups.items()}
        groups = {p: d for p, d in groups.items() if d}
        for proj, d in sorted(groups.items())[:max_proj]:
            ids = sorted(d)
            need = sum(_nbytes(index[d[e]][1]) for e in ids) * 2  # bf16 payload -> f32 stack
            n_use = len(ids)
            if need > max_group_gb * 2 ** 30:
                n_use = max(8, int(len(ids) * max_group_gb * 2 ** 30 / need))
                ids = ids[:n_use]
            # Group the loads by shard so each file is opened for a run of experts.
            ids.sort(key=lambda e: index[d[e]][0])
            rows = [tensor(d[e]).reshape(-1) for e in ids]
            X = np.stack(rows, axis=0)
            del rows
            r = _spectrum(X)
            r["tensor"] = proj
            r["n_experts_total"] = len(d)
            if n_use < len(d):
                r["subsampled"] = True
            out["cross_expert"].append(r)
            del X
        first = groups[sorted(groups)[0]]
        out["within_expert"] = _effective_rank(tensor(first[sorted(first)[0]]))
    else:
        stacked = [(proj, emap[None]) for proj, emap in sorted(layer_groups.items())
                   if None in emap][:max_proj]
        stacked.sort(key=lambda item: index[item[1]][0])
        within = None
        for i, (proj, k) in enumerate(stacked):
            v = tensor(k)
            n_exp = int(v.shape[0])
            if i == 0:
                within = np.array(v[0], dtype=np.float32, copy=True)
            r = _spectrum(v.reshape(n_exp, -1))
            r["tensor"] = proj
            r["n_experts_total"] = n_exp
            out["cross_expert"].append(r)
            del v
        out["within_expert"] = _effective_rank(within)
    out["hypotheses"] = hypotheses(out)
    if not out["hypotheses"]:
        raise AnatomyUnavailable(
            f"{snapshot}: spectra computed but yielded no hypothesis -- refusing to "
            f"record an anatomy that says nothing"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI so HCLI can own OI as a subprocess without importing mlx.

    Exit 0: JSON anatomy on stdout.
    Exit 2: AnatomyUnavailable on stderr (named refusal, not a crash).
    """
    import argparse
    import sys
    p = argparse.ArgumentParser(
        prog="representational_anatomy",
        description="Odyssey-I representational anatomy of one snapshot directory.",
    )
    p.add_argument("snapshot", help="directory of .safetensors shards")
    p.add_argument("--layer", default="0",
                   help="layer index, or 'auto' for the lowest complete expert group")
    args = p.parse_args(argv)
    layer: int | None
    if str(args.layer).lower() in ("auto", "none"):
        layer = None
    else:
        layer = int(args.layer)
    try:
        out = anatomy_from_safetensors(args.snapshot, layer=layer)
    except AnatomyUnavailable as e:
        sys.stderr.write(str(e) + "\n")
        return 2
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
