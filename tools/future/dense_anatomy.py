"""OI dense-body anatomy: how a body with no expert tensors wants to be represented.

35 of 56 lake specimens are DENSE. representational_anatomy.py measures only
expert organs and correctly refuses all 35; this module is their Odyssey-I
deliverable. Do not "fix" that refusal -- a body with experts is not ours.

A spectral ratio compared against a CONSTANT is not evidence (G028/G029):

  * centred independent rows score participation n-1, so p/n is (n-1)/n on
    pure noise -- 0.95 at n=20. SHARING_LIVE_BELOW = 0.95 reports LIVE on
    noise for every n <= 20.
  * the effective-rank null is aspect-ratio dependent (0.606 square, 0.847
    at 3072x1024). LOWRANK_LIVE_BELOW = 0.40 is a moving target.
  * whether structure EXISTS (deficit vs a same-shape null) and whether
    factoring PAYS (rank_90 vs mn/(m+n)) are different questions. A rank-495
    factor of a 2048x768 tensor SAVES 11.4%; the sibling's hypothesis text
    said it costs more than dense.

Sharing is spectral_null's statistic (calibrated + verdict). Within-tensor
effective rank is the same treatment, which spectral_null does not yet cover:
compare the organ against a random matrix of the SAME SHAPE and report the
deficit. Payloads are streamed tensor-by-tensor from safetensors offsets so a
135 GiB / 37-shard body stays under 20 GiB RSS.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import resource
import struct
import sys
import tempfile
from typing import Any

import numpy as np
from scipy import linalg

from lake_scheme_census import FLOAT_DTYPES, read_header
from spectral_null import GRAM_BLOCK, verdict

LIVE_DEFICIT_PCT = 2.0          # same bar spectral_null.verdict uses
NULL_SEED = 5                   # spectral_null's seed
_GRAM_FEAT_BLOCK = 4096         # columns/rows per blocked W W^T -- Metal timeout class
_CROSS_LAYER_MAX_BYTES = 1 << 30
_CROSS_LAYER_MIN_FEATURES = 1 << 16
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|layer|blocks|block|h)\.(\d+)\.")
_HF_SHARD_RE = re.compile(r"model-\d+-of-\d+\.safetensors$")
_NULL_EIGS: dict[tuple[int, int, int], np.ndarray] = {}


class DenseAnatomyUnavailable(RuntimeError):
    """No dense anatomy could be measured, and saying so is the only honest answer.

    A well-formed empty result ({within_tensor: [], hypotheses: []}) at exit 0
    is how a campaign driver records ANATOMISED having measured nothing. Raise
    a named refusal instead.
    """


# ---------------------------------------------------------------------------
# factoring arithmetic -- existence of structure is a different function
# ---------------------------------------------------------------------------

def factoring_report(m: int, n: int, rank_90: int) -> dict:
    """Bytes of a rank-k factor vs dense. Break-even rank is mn/(m+n)."""
    m, n, rank_90 = int(m), int(n), int(rank_90)
    dense = m * n
    break_even = dense / (m + n) if (m + n) else 0.0
    factored = rank_90 * (m + n)
    byte_ratio = factored / dense if dense else float("inf")
    pays = byte_ratio < 1.0
    return {
        "break_even_rank": round(break_even, 2),
        "rank_90": rank_90,
        "byte_ratio": round(byte_ratio, 4),
        "pays": pays,
        "dense_elems": dense,
        "factored_elems": factored,
        "save_pct": round(100.0 * (1.0 - byte_ratio), 2) if pays else 0.0,
    }


# ---------------------------------------------------------------------------
# headers, shard selection, organ discovery
# ---------------------------------------------------------------------------

def _payload_offset(path: str) -> int:
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise DenseAnatomyUnavailable(f"{path}: shorter than a safetensors header")
        n = struct.unpack("<Q", raw)[0]
    return 8 + n


def _select_shards(shards: list[str]) -> list[str]:
    """Drop a Mistral-style consolidated.safetensors when HF shards sit beside it.

    The two dumps are the same organism under different key schemes. Indexing
    both would double-count every language organ and pull a 48 GiB shard into
    the scan for no new measurement.
    """
    if any(_HF_SHARD_RE.match(os.path.basename(s)) for s in shards):
        shards = [s for s in shards if os.path.basename(s) != "consolidated.safetensors"]
    return shards


def _namespace(snapshot: str, shard: str) -> str:
    rel = os.path.relpath(os.path.dirname(shard), snapshot)
    if rel in (".", ""):
        return ""
    return rel.replace(os.sep, ".")


def _parse_organ(key: str, namespace: str) -> tuple[str, int] | None:
    if not key.endswith(".weight"):
        return None
    if "expert" in key.lower():
        return None
    m = _LAYER_RE.search(key)
    if not m:
        return None
    layer = int(m.group(1))
    organ_key = key[:m.start(1)] + "N" + key[m.end(1):]
    if namespace:
        organ_key = namespace + "::" + organ_key
    return organ_key, layer


def _assign_names(organ_keys: list[str]) -> dict[str, str]:
    """Short unique names: q_proj when unique, self_attn.q when it isn't."""
    def candidates(ok: str) -> list[str]:
        ns, sep, key = ok.partition("::")
        if not sep:
            key, ns = ns, ""
        key = key.removesuffix(".weight")
        m = re.search(r"(?:layers|layer|blocks|block|h)\.N\.(.*)$", key)
        rest = m.group(1) if m else key.rsplit(".", 1)[-1]
        parts = rest.split(".")
        out: list[str] = []
        if parts[-1].isdigit() and len(parts) >= 2:
            out.append(".".join(parts[-2:]))
        else:
            out.append(parts[-1])
        if len(parts) >= 2:
            out.append(".".join(parts[-2:]))
        out.append(rest)
        if ns:
            out.append(f"{ns}.{out[0]}")
            out.append(f"{ns}.{rest}")
        # unique-ify while preserving order
        seen: set[str] = set()
        uniq = []
        for n in out:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        return uniq

    cand = {ok: candidates(ok) for ok in organ_keys}
    assigned: dict[str, str] = {}
    used: set[str] = set()
    depth = 0
    pending = set(organ_keys)
    while pending:
        taken_this = collections.defaultdict(list)
        for ok in list(pending):
            opts = cand[ok]
            name = opts[min(depth, len(opts) - 1)]
            taken_this[name].append(ok)
        for name, oks in taken_this.items():
            if len(oks) == 1 and name not in used:
                assigned[oks[0]] = name
                used.add(name)
                pending.discard(oks[0])
        depth += 1
        if depth > 8:
            for i, ok in enumerate(sorted(pending)):
                assigned[ok] = f"{cand[ok][-1]}#{i}"
            break
    return assigned


def _present_extensions(snapshot: str) -> str:
    counts: collections.Counter = collections.Counter()
    if os.path.isdir(snapshot):
        for root, _dirs, files in os.walk(snapshot):
            for f in files:
                counts[os.path.splitext(f)[1] or "<noext>"] += 1
    if not counts:
        return "(empty)"
    return ", ".join(f"{n}x{e}" for e, n in counts.most_common(6))


# ---------------------------------------------------------------------------
# payload: one tensor, never a whole shard
# ---------------------------------------------------------------------------

def _decode(raw: bytes, dtype: str, shape: list) -> np.ndarray:
    shape = tuple(int(x) for x in shape)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    elif dtype == "F64":
        arr = np.frombuffer(raw, dtype=np.float64).astype(np.float32)
    else:
        raise ValueError(f"internal: non-float dtype {dtype} reached decoder")
    return np.ascontiguousarray(arr.reshape(shape), dtype=np.float32)


def _load_tensor(path: str, entry: dict, payload_off: int) -> np.ndarray:
    start, end = entry["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(payload_off + int(start))
        raw = fh.read(int(end) - int(start))
    return _decode(raw, entry["dtype"], entry["shape"])


_DTYPE_BPE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}


def _load_flat_prefix(path: str, entry: dict, payload_off: int, n_take: int) -> np.ndarray:
    """First n_take flattened elements, without pulling the rest of the tensor.

    Cross-layer participation is a statistic on n_layers rows; a coordinate
    subset is enough, and a 72B down_proj is 484 MiB of bf16 we would otherwise
    reread at every layer only to throw away all but a few million entries.
    """
    dtype = entry["dtype"]
    bpe = _DTYPE_BPE[dtype]
    start, end = int(entry["data_offsets"][0]), int(entry["data_offsets"][1])
    total = (end - start) // bpe
    n_take = min(int(n_take), total)
    with open(path, "rb") as fh:
        fh.seek(payload_off + start)
        raw = fh.read(n_take * bpe)
    return _decode(raw, dtype, [n_take]).reshape(-1)


# ---------------------------------------------------------------------------
# within-tensor effective rank vs a same-shape random null
# ---------------------------------------------------------------------------

def _gram_eigs(W: np.ndarray, block: int = _GRAM_FEAT_BLOCK) -> np.ndarray:
    """Eigenvalues of W W^T (or W^T W), equal to squared singular values."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    m, n = int(W.shape[0]), int(W.shape[1])
    if m <= n:
        G = np.zeros((m, m), dtype=np.float32)
        for s in range(0, n, block):
            B = W[:, s:s + block]
            G += B @ B.T
    else:
        G = np.zeros((n, n), dtype=np.float32)
        for s in range(0, m, block):
            B = W[s:s + block, :]
            G += B.T @ B
    G = 0.5 * (G + G.T)
    ev = linalg.eigvalsh(G, check_finite=False)
    return np.maximum(np.ascontiguousarray(ev[::-1], dtype=np.float64), 0.0)


def _random_gram_eigs(m: int, n: int, seed: int, block: int = _GRAM_FEAT_BLOCK) -> np.ndarray:
    """Same statistic on iid Gaussian of shape (m, n), without holding the matrix."""
    rng = np.random.default_rng(seed)
    if m <= n:
        G = np.zeros((m, m), dtype=np.float32)
        for s in range(0, n, block):
            w = min(block, n - s)
            B = rng.standard_normal((m, w), dtype=np.float32)
            G += B @ B.T
    else:
        G = np.zeros((n, n), dtype=np.float32)
        for s in range(0, m, block):
            h = min(block, m - s)
            B = rng.standard_normal((h, n), dtype=np.float32)
            G += B.T @ B
    G = 0.5 * (G + G.T)
    ev = linalg.eigvalsh(G, check_finite=False)
    return np.maximum(np.ascontiguousarray(ev[::-1], dtype=np.float64), 0.0)


def _null_eigs(m: int, n: int, seed: int = NULL_SEED) -> np.ndarray:
    key = (int(m), int(n), int(seed))
    if key not in _NULL_EIGS:
        _NULL_EIGS[key] = _random_gram_eigs(m, n, seed)
    return _NULL_EIGS[key]


def _participation_from_eigs(ev: np.ndarray) -> tuple[float, int]:
    tot = float(ev.sum())
    p = ev / max(tot, 1e-30)
    pos = p > 0
    ent = float(-(p[pos] * np.log(np.maximum(p[pos], 1e-30))).sum())
    part = float(np.exp(ent))
    cum = np.cumsum(ev) / max(tot, 1e-30)
    rank_90 = int((cum < 0.90).sum()) + 1
    return part, rank_90


NULL_SEEDS = 5                  # draws averaged; one draw carries its own noise


def effective_rank_calibrated(W: np.ndarray, seed: int = NULL_SEED,
                              n_seeds: int = NULL_SEEDS) -> dict:
    """Effective-rank ratio of W, of a same-shape random null, and the deficit.

    THE NULL IS A RANDOM DRAW AND CARRIES ITS OWN NOISE. On a 1024x1024 organ,
    forty seeds give mean 41.297 with stdev 0.045 pp, so any two organs closer
    than about 0.1 pp are NOT ordered by this statistic -- and two adjacent pairs
    in the first Wan2.2-T2V-A14B ordering were 0.04 and 0.05 pp apart, reported
    as ranked when they are not resolved.

    So the deficit now carries `deficit_stderr_pct`, and a caller comparing
    organs must respect it. Nulls are cached by (m, n, seed), so the extra draws
    cost nothing once a shape repeats across layers -- which is every layer.
    """
    W = np.ascontiguousarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"effective rank needs a 2D weight, got shape {W.shape}")
    m, n = int(W.shape[0]), int(W.shape[1])
    full = min(m, n)
    part, rank_90 = _participation_from_eigs(_gram_eigs(W))
    ratio = part / max(full, 1)

    seeds = [seed + i for i in range(max(1, int(n_seeds)))]
    nulls = [_participation_from_eigs(_null_eigs(m, n, s))[0] / max(full, 1)
             for s in seeds]
    null_ratio = float(np.mean(nulls))
    deficits = [100.0 * (nr - ratio) / max(nr, 1e-30) for nr in nulls]
    # standard error of the MEAN deficit: what a comparison between organs may resolve
    stderr = (float(np.std(deficits, ddof=1)) / np.sqrt(len(deficits))
              if len(deficits) > 1 else float("nan"))
    fac = factoring_report(m, n, rank_90)
    return {
        "shape": [m, n],
        "full_rank": full,
        "participation": round(part, 2),
        "ratio": round(ratio, 4),
        "null_ratio": round(null_ratio, 4),
        "null_seeds": len(seeds),
        "deficit_pct": round(100.0 * (null_ratio - ratio) / max(null_ratio, 1e-30), 2),
        "deficit_stderr_pct": round(stderr, 4),
        **fac,
    }


def resolved_ordering(organs: list) -> list:
    """Rank organs by deficit, TYING any pair the null cannot separate.

    `organs` is a list of (name, deficit_pct, stderr_pct). Two organs are ordered
    only when their deficits differ by more than 2x the combined standard error;
    otherwise they share a rank. Reporting an order the instrument cannot resolve
    is the same error class as thresholding a ratio against a constant.
    """
    rows = sorted(organs, key=lambda r: -r[1])
    out, rank = [], 0
    for i, (name, d, se) in enumerate(rows):
        if i:
            pd, pse = rows[i - 1][1], rows[i - 1][2]
            sep = 2.0 * float(np.hypot(se if se == se else 0.0,
                                       pse if pse == pse else 0.0))
            if (pd - d) > sep:
                rank = i
        out.append({"organ": name, "deficit_pct": d, "stderr_pct": se, "rank": rank})
    return out


# ---------------------------------------------------------------------------
# cross-layer sharing -- CPU port of spectral_null.calibrated
# ---------------------------------------------------------------------------

def _centred_gram(X: np.ndarray, block: int = GRAM_BLOCK) -> np.ndarray:
    """(n,n) Gram of column-centred rows, accumulated in feature blocks.

    Faithful port of spectral_null.centred_gram. Blocking is the point: a
    single (512, 3.3M) matmul GPU-timeouts Metal; the same product on CPU in
    one shot is a 20 GiB RSS spike on a 72B down_proj stack.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    n, D = int(X.shape[0]), int(X.shape[1])
    G = np.zeros((n, n), dtype=np.float32)
    for s in range(0, D, block):
        B = X[:, s:s + block] - mu[:, s:s + block]
        G += B @ B.T
    return G / max(D, 1)


def _participation_gram(G: np.ndarray) -> float:
    ev = linalg.eigvalsh(0.5 * (G + G.T), check_finite=False)
    ev = np.maximum(ev[::-1], 0.0)
    tot = float(ev.sum())
    p = ev / max(tot, 1e-30)
    pos = p > 0
    ent = float(-(p[pos] * np.log(np.maximum(p[pos], 1e-30))).sum())
    return float(np.exp(ent))


def _norm_matched_null(X: np.ndarray, seed: int = NULL_SEED) -> np.ndarray:
    """Gaussian rows carrying the SAME per-row norms as X (spectral_null)."""
    norms = np.sqrt((X * X).sum(axis=1, keepdims=True))
    rng = np.random.default_rng(seed)
    R = rng.standard_normal(X.shape, dtype=np.float32)
    R = R / np.sqrt((R * R).sum(axis=1, keepdims=True)) * norms
    return np.ascontiguousarray(R, dtype=np.float32)


def calibrated_stack(X: np.ndarray, seed: int = NULL_SEED) -> dict:
    """Participation of stacked organ rows vs a norm-matched null.

    Same return schema as spectral_null.calibrated, so spectral_null.verdict
    consumes it. Computed on CPU: we never copy a (n_layers, m*n) stack into
    the GPU allocator (72B down_proj is 77 GiB flattened).
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = int(X.shape[0])
    if n < 3:
        raise ValueError(f"participation needs at least 3 rows to be meaningful, got {n}")
    real = _participation_gram(_centred_gram(X))
    null = _participation_gram(_centred_gram(_norm_matched_null(X, seed)))
    return {
        "n": n,
        "d": int(X.shape[1]),
        "participation": round(real, 3),
        "ratio_n": round(real / n, 4),
        "ratio_n_minus_1": round(real / (n - 1), 4),
        "null_participation": round(null, 3),
        "null_ratio_n": round(null / n, 4),
        "deficit_pct": round(100.0 * (null - real) / max(null, 1e-30), 3),
        "noise_floor_ratio_n": round((n - 1) / n, 4),
    }


def _n_features(n_layers: int, d: int) -> int:
    if n_layers * d * 4 <= _CROSS_LAYER_MAX_BYTES:
        return d
    budget = _CROSS_LAYER_MAX_BYTES // (4 * max(n_layers, 1))
    return int(min(d, max(_CROSS_LAYER_MIN_FEATURES, budget)))


# ---------------------------------------------------------------------------
# hypotheses -- the OI deliverable
# ---------------------------------------------------------------------------

def _mid_rows(within: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in within:
        by[r["organ"]].append(r)
    out = []
    for organ, rows in by.items():
        mid = next((x for x in rows if x["depth"] == "middle"), None)
        out.append(mid or rows[0])
    return out


def hypotheses(anatomy: dict) -> list[str]:
    """What the numbers license, in plain terms, grounded in deficits not constants."""
    out: list[str] = []
    within = anatomy.get("within_tensor") or []
    cross = anatomy.get("cross_layer") or []
    mids = _mid_rows(within)

    if mids:
        ordered = sorted(mids, key=lambda r: (-float(r["deficit_pct"]), r["organ"]))
        order_txt = ", ".join(f"{r['organ']} {r['deficit_pct']:.2f}%" for r in ordered)
        top, bot = ordered[0], ordered[-1]
        out.append(
            f"{top['organ']} is the most compressible organ by within-tensor deficit "
            f"({top['deficit_pct']:.2f}% below a same-shape random null at mid-depth); "
            f"full organ ordering: {order_txt}"
        )
        if top["deficit_pct"] >= LIVE_DEFICIT_PCT:
            out.append(
                f"within-tensor structure is ALIVE and strongly organ-dependent "
                f"(mid-depth deficits {bot['deficit_pct']:.2f}%..{top['deficit_pct']:.2f}%) "
                f"-- a constant LOWRANK_LIVE_BELOW would miss the spread"
            )
        else:
            out.append(
                f"within-tensor structure is DEAD: every organ is within "
                f"{top['deficit_pct']:.2f}% of a same-shape random null "
                f"(live bar is {LIVE_DEFICIT_PCT:.0f}% deficit, not a ratio constant)"
            )
        pays = [r for r in ordered if r.get("pays")]
        nopay = [r for r in ordered if not r.get("pays")]
        if pays:
            bits = [
                f"{r['organ']} rank_90={r['rank_90']} vs break-even {r['break_even_rank']} "
                f"(byte ratio {r['byte_ratio']:.3f}, saves {r['save_pct']:.1f}%)"
                for r in pays
            ]
            out.append(
                "low-rank factoring PAYS (break-even arithmetic, separate from whether "
                "structure exists): " + "; ".join(bits)
            )
        if nopay:
            bits = [
                f"{r['organ']} rank_90={r['rank_90']} vs break-even {r['break_even_rank']} "
                f"(byte ratio {r['byte_ratio']:.3f})"
                for r in nopay
            ]
            out.append(
                "low-rank factoring does NOT pay for: " + "; ".join(bits)
                + " -- rank_90 is 90% of spectral energy, not capability"
            )

    if cross:
        hottest = max(cross, key=lambda r: (float(r["deficit_pct"]), r["organ"]))
        v = verdict(hottest, live_deficit_pct=LIVE_DEFICIT_PCT)
        dead_n = sum(1 for r in cross if r["deficit_pct"] < LIVE_DEFICIT_PCT)
        flag = "LIVE" if hottest["deficit_pct"] >= LIVE_DEFICIT_PCT else "DEAD"
        out.append(
            f"cross-layer sharing is {flag} against a norm-matched null, not a "
            f"constant threshold: {v} ({dead_n}/{len(cross)} organs indistinguishable "
            f"from noise; highest-deficit organ is {hottest['organ']})"
        )

    structure = any(r["deficit_pct"] >= LIVE_DEFICIT_PCT for r in mids)
    sharing = any(r["deficit_pct"] >= LIVE_DEFICIT_PCT for r in cross)
    if (mids or cross) and not structure and not sharing:
        out.append(
            "NO LINEAR STRUCTURE AVAILABLE: within-tensor deficits and cross-layer "
            "sharing are both indistinguishable from a matched null. This specimen "
            "is measured-and-negative, not unmeasured."
        )
    return out


# ---------------------------------------------------------------------------
# anatomy
# ---------------------------------------------------------------------------

def _index_snapshot(snapshot: str) -> tuple[list[str], dict[str, tuple[str, dict]], dict[str, int]]:
    shards = _select_shards(sorted(glob.glob(snapshot + "/**/*.safetensors", recursive=True)))
    if not shards:
        raise DenseAnatomyUnavailable(
            f"{snapshot}: no .safetensors weights (searched recursively). Present: "
            + _present_extensions(snapshot)
            + ". This body needs format conversion before OI can measure it."
        )
    index: dict[str, tuple[str, dict]] = {}
    offsets: dict[str, int] = {}
    for f in shards:
        try:
            hdr = read_header(f)
        except (ValueError, json.JSONDecodeError) as exc:
            raise DenseAnatomyUnavailable(f"{f}: unreadable safetensors header ({exc})") from exc
        offsets[f] = _payload_offset(f)
        ns = _namespace(snapshot, f)
        for k, e in hdr.items():
            # namespace only when two subdirs would otherwise collide on the key
            store_k = f"{ns}::{k}" if ns else k
            index[store_k] = (f, e)
    if not index:
        raise DenseAnatomyUnavailable(
            f"{snapshot}: {len(shards)} shards, zero tensors in their headers"
        )
    return shards, index, offsets


def anatomy_from_safetensors(snapshot: str) -> dict:
    """Dense-body anatomy, organ-by-organ, one tensor resident at a time.

    Raises DenseAnatomyUnavailable rather than returning an empty anatomy.
    """
    snapshot = os.path.abspath(os.path.expanduser(str(snapshot))).rstrip("/")
    if not os.path.isdir(snapshot):
        raise DenseAnatomyUnavailable(f"{snapshot}: not a directory")

    _NULL_EIGS.clear()
    shards, index, offsets = _index_snapshot(snapshot)

    # Expert bodies are the other module's job -- refuse even if they also have
    # dense attention tensors.
    #
    # BUT "expert" AS A SUBSTRING IS NOT AN EXPERT ORGAN. pi0_base's keys read
    # paligemma_with_expert.gemma_expert..., which is a SUBMODULE NAME, and 767
    # of them matched. It was refused here as "has experts, the other module owns
    # it" AND refused by representational_anatomy as "not a per-expert layout",
    # so it fell through both and was measured by neither. An MoE organ is an
    # INDEXED set of experts or a stacked tensor whose first axis is the expert
    # axis; lake_scheme_census._is_expert_organ is that test and this defers to it.
    from lake_scheme_census import _is_expert_organ as _real_expert_organ

    expert_keys = [k for k, (_f, _e) in index.items()
                   if _real_expert_organ(k.split("::", 1)[-1], _e)]
    if expert_keys:
        raise DenseAnatomyUnavailable(
            f"{snapshot}: body HAS expert tensors ({len(expert_keys)} keys, e.g. "
            f"{expert_keys[0]}). representational_anatomy.py owns the expert-organ "
            f"case; this module is the dense path."
        )

    organs: dict[str, dict[int, tuple[str, dict]]] = collections.defaultdict(dict)
    organ_dtypes: collections.Counter = collections.Counter()
    layers_seen: set[int] = set()
    for store_k, (f, e) in index.items():
        raw_key = store_k.split("::", 1)[-1]
        parsed = _parse_organ(raw_key, _namespace(snapshot, f))
        if not parsed:
            continue
        if len(e.get("shape") or []) != 2:
            continue
        organ_key, layer = parsed
        organs[organ_key][layer] = (store_k, e)
        organ_dtypes[e.get("dtype", "")] += 1
        layers_seen.add(layer)

    nonfloat = sorted(d for d in organ_dtypes if d not in FLOAT_DTYPES)
    if nonfloat:
        raise DenseAnatomyUnavailable(
            f"{snapshot}: weights are pre-quantized (payload dtype {nonfloat} not in "
            f"BF16/F16/F32/F64) -- a spectrum over quantization codes measures the "
            f"codebook, not the organism"
        )
    if len(layers_seen) < 3:
        raise DenseAnatomyUnavailable(
            f"{snapshot}: fewer than 3 layers (saw {sorted(layers_seen)[:8]} across "
            f"{len(index)} tensors) so a cross-layer statistic is meaningless"
        )
    if not organs:
        raise DenseAnatomyUnavailable(
            f"{snapshot}: {len(index)} tensors across {len(shards)} shards, but no "
            f"2D layer-indexed .weight organ is measurable (e.g. {sorted(index)[0]})"
        )

    names = _assign_names(sorted(organs))
    within: list[dict] = []
    cross: list[dict] = []

    for organ_key in sorted(organs, key=lambda k: names[k]):
        layer_map = organs[organ_key]
        layers = sorted(layer_map)
        display = names[organ_key]
        picks = [("first", layers[0]),
                 ("middle", layers[len(layers) // 2]),
                 ("last", layers[-1])]
        seen_L: set[int] = set()
        for depth, L in picks:
            if L in seen_L:
                continue
            seen_L.add(L)
            store_k, entry = layer_map[L]
            path = index[store_k][0]
            W = _load_tensor(path, entry, offsets[path])
            rec = effective_rank_calibrated(W)
            rec.update(organ=display, organ_key=organ_key, layer=L, depth=depth)
            within.append(rec)
            del W

        if len(layers) >= 3:
            shape0 = tuple(layer_map[layers[0]][1]["shape"])
            d = int(shape0[0]) * int(shape0[1])
            n_take = _n_features(len(layers), d)
            # gather subsampled rows, shard by shard, one tensor at a time
            need_by_shard: dict[str, list[tuple[int, str, dict]]] = collections.defaultdict(list)
            for L in layers:
                store_k, entry = layer_map[L]
                if tuple(entry["shape"]) != shape0:
                    continue
                path = index[store_k][0]
                need_by_shard[path].append((L, store_k, entry))
            rows: dict[int, np.ndarray] = {}
            for path, items in need_by_shard.items():
                items = sorted(items, key=lambda t: int(t[2]["data_offsets"][0]))
                off = offsets[path]
                for L, _store_k, entry in items:
                    rows[L] = _load_flat_prefix(path, entry, off, n_take)
            ordered_rows = [rows[L] for L in layers if L in rows]
            if len(ordered_rows) >= 3:
                X = np.stack(ordered_rows, axis=0)
                c = calibrated_stack(X)
                c.update(organ=display, organ_key=organ_key)
                if n_take < d:
                    c["subsampled_features"] = n_take
                c["verdict"] = verdict(c, live_deficit_pct=LIVE_DEFICIT_PCT)
                cross.append(c)
                del X
            del rows, ordered_rows

    out: dict[str, Any] = {
        "snapshot": snapshot,
        "n_shards": len(shards),
        "n_tensors": len(index),
        "n_layers": len(layers_seen),
        "organs": [names[k] for k in sorted(organs, key=lambda x: names[x])],
        "within_tensor": within,
        "cross_layer": cross,
    }
    out["hypotheses"] = hypotheses(out)
    if not out["hypotheses"]:
        raise DenseAnatomyUnavailable(
            f"{snapshot}: spectra computed but yielded no hypothesis -- refusing to "
            f"record an anatomy that says nothing"
        )
    return out


# ---------------------------------------------------------------------------
# selfcheck writer (also imported by the test module)
# ---------------------------------------------------------------------------

def _write_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict] = {}
    payload = bytearray()
    for k, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        try:
            dt = {np.dtype("float32"): "F32", np.dtype("float16"): "F16",
                  np.dtype("float64"): "F64", np.dtype("int8"): "I8",
                  np.dtype("uint8"): "U8"}[arr.dtype]
        except KeyError as exc:
            raise TypeError(f"unsupported dtype {arr.dtype} for {k}") from exc
        b = arr.tobytes()
        header[k] = {"dtype": dt, "shape": [int(x) for x in arr.shape],
                     "data_offsets": [len(payload), len(payload) + len(b)]}
        payload.extend(b)
    blob = json.dumps(header).encode()
    pad = (8 - (len(blob) % 8)) % 8
    blob += b" " * pad
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(payload)


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss) if sys.platform == "darwin" else int(rss) * 1024


# ---------------------------------------------------------------------------
# selfcheck / lake matrix
# ---------------------------------------------------------------------------

def _selfcheck() -> None:
    rng = np.random.default_rng(0)

    # Break-even arithmetic: the sibling's "costs more than dense" was false.
    fac = factoring_report(2048, 768, 495)
    assert abs(fac["byte_ratio"] - 0.886) < 1e-3, fac
    assert fac["pays"] is True, fac
    assert fac["factored_elems"] < fac["dense_elems"], fac

    # Square random null sits near the Marchenko-Pastur ~0.60 class, not 0.40.
    sq = effective_rank_calibrated(rng.standard_normal((128, 128), dtype=np.float32))
    assert abs(sq["deficit_pct"]) < 5.0, sq
    assert sq["null_ratio"] > 0.50, sq

    # Rank-3 factor is DETECTED. Full-rank random is not.
    B = rng.standard_normal((3, 48), dtype=np.float32)
    C = rng.standard_normal((96, 3), dtype=np.float32)
    low = effective_rank_calibrated(C @ B)
    full = effective_rank_calibrated(rng.standard_normal((96, 48), dtype=np.float32))
    assert low["deficit_pct"] > 50.0, low
    assert abs(full["deficit_pct"]) < 5.0, full

    # Sharing port: independent rows DEAD, rank-3 basis LIVE. This is the
    # spectral_null selfcheck, on CPU, so verdict() is reading our dict.
    R = rng.standard_normal((12, 1 << 14), dtype=np.float32)
    c = calibrated_stack(R)
    assert abs(c["ratio_n"] - 11 / 12) < 2e-2, c
    assert abs(c["deficit_pct"]) < 2.0, c
    assert "DEAD" in verdict(c), verdict(c)
    B = rng.standard_normal((3, 1 << 14), dtype=np.float32)
    C = rng.standard_normal((16, 3), dtype=np.float32)
    c = calibrated_stack(C @ B)
    assert c["deficit_pct"] > 50.0, c
    assert "LIVE" in verdict(c), verdict(c)

    with tempfile.TemporaryDirectory() as d:
        # no safetensors
        open(os.path.join(d, "pytorch_model.bin"), "wb").write(b"\0")
        try:
            anatomy_from_safetensors(d)
            raise AssertionError("empty anatomy leaked for a body with no safetensors")
        except DenseAnatomyUnavailable as exc:
            assert "no .safetensors" in str(exc), exc

        # experts belong to the other module
        _write_safetensors(os.path.join(d, "m.safetensors"), {
            f"model.layers.{i}.mlp.experts.0.down_proj.weight":
                rng.standard_normal((8, 8), dtype=np.float32) for i in range(4)
        })
        try:
            anatomy_from_safetensors(d)
            raise AssertionError("expert body was anatomised by the dense path")
        except DenseAnatomyUnavailable as exc:
            assert "representational_anatomy.py" in str(exc), exc

        # I8 is pre-quantized
        _write_safetensors(os.path.join(d, "m.safetensors"), {
            f"model.layers.{i}.self_attn.q_proj.weight":
                np.zeros((8, 8), dtype=np.int8) for i in range(4)
        })
        try:
            anatomy_from_safetensors(d)
            raise AssertionError("I8 payload was measured as if it were the organism")
        except DenseAnatomyUnavailable as exc:
            assert "pre-quantized" in str(exc) and "I8" in str(exc), exc

        # fewer than 3 layers
        _write_safetensors(os.path.join(d, "m.safetensors"), {
            f"model.layers.{i}.self_attn.q_proj.weight":
                rng.standard_normal((16, 16), dtype=np.float32) for i in range(2)
        })
        try:
            anatomy_from_safetensors(d)
            raise AssertionError("2-layer body produced a cross-layer statistic")
        except DenseAnatomyUnavailable as exc:
            assert "fewer than 3 layers" in str(exc), exc

        # nested safetensors are FOUND
        nest = os.path.join(d, "low_noise_model")
        os.makedirs(nest, exist_ok=True)
        os.remove(os.path.join(d, "m.safetensors"))
        tensors = {}
        B = rng.standard_normal((3, 24), dtype=np.float32)
        for i in range(4):
            C = rng.standard_normal((32, 3), dtype=np.float32)
            tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = C @ B
        _write_safetensors(os.path.join(nest, "m.safetensors"), tensors)
        a = anatomy_from_safetensors(d)
        assert a["n_shards"] == 1, a
        assert a["hypotheses"], a
        mid = next(r for r in a["within_tensor"] if r["depth"] == "middle")
        assert mid["deficit_pct"] > 50.0, mid

    print("selfcheck OK -- break-even saves on 2048x768/r495; rank-3 organ detected; "
          f"sharing uses spectral_null.verdict; nested shards found")


_LAKE = "/Volumes/corpdrive/hawking-modellake/specimens"
_MATRIX = [
    "Qwen--Qwen3-0.6B@c1899de289a0",
    "Qwen--Qwen3-14B@40c069824f42",
    "microsoft--Phi-4-reasoning-plus@69baf8528e1b",
    "mistralai--Mistral-Small-3.1-24B-Instruct-2503@68faf",
    "Qwen--Qwen2.5-72B-Instruct@495f39366efe",
    "Wan-AI--Wan2.2-T2V-A14B@c8c270b13ee0",
    "Qwen--Qwen3-30B-A3B@ad44e777bcd1",
    "microsoft--bitnet-b1.58-2B-4T@04c3b9ad9361",
    "arcinstitute--evo2_40b@d529aa57c307",
]


def _resolve_slug(lake: str, slug: str) -> str | None:
    p = os.path.join(lake, slug)
    if os.path.isdir(p):
        return p
    if not os.path.isdir(lake):
        return None
    hits = sorted(d for d in os.listdir(lake) if d.startswith(slug))
    return os.path.join(lake, hits[0]) if hits else None


def _run_one(snapshot: str) -> None:
    import time
    t0 = time.perf_counter()
    name = os.path.basename(snapshot.rstrip("/"))
    try:
        a = anatomy_from_safetensors(snapshot)
        status = "ANATOMY"
        detail = a["hypotheses"][0] if a.get("hypotheses") else "(no hypothesis)"
        if len(a.get("hypotheses") or []) > 1:
            detail += " | " + a["hypotheses"][1]
    except DenseAnatomyUnavailable as exc:
        status = "REFUSAL"
        detail = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        status = "REFUSAL"
        detail = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    rss = peak_rss_bytes() / 2 ** 30
    print(f"{name:64s}  {status:8s}  wall={wall:7.1f}s  peak_rss={rss:5.2f}GiB  {detail}",
          flush=True)


def _matrix(lake: str = _LAKE) -> None:
    import subprocess
    if not os.path.isdir(lake):
        raise SystemExit(f"LAKE ABSENT -- {lake} is not a directory. Matrix DID NOT RUN.")
    for slug in _MATRIX:
        path = _resolve_slug(lake, slug)
        if path is None:
            print(f"{slug:64s}  REFUSAL   wall={0:7.1f}s  peak_rss={0:5.2f}GiB  "
                  f"DenseAnatomyUnavailable: snapshot absent under {lake}",
                  flush=True)
            continue
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--one", path],
            check=False,
        )
        if proc.returncode != 0:
            print(f"{os.path.basename(path):64s}  REFUSAL   "
                  f"(child exited {proc.returncode})", flush=True)


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    elif "--matrix" in sys.argv:
        lake = _LAKE
        if "--lake" in sys.argv:
            lake = sys.argv[sys.argv.index("--lake") + 1]
        _matrix(lake)
    elif "--one" in sys.argv:
        _run_one(sys.argv[sys.argv.index("--one") + 1])
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        print(json.dumps(anatomy_from_safetensors(sys.argv[1]), indent=1))
    else:
        sys.stderr.write(
            "usage: dense_anatomy.py --selfcheck | --matrix | --one SNAPSHOT | SNAPSHOT\n"
        )
        sys.exit(2)
