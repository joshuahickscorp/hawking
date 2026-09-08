"""Cheapest discriminator for non-linear cross-expert structure.

Linear sharing is dead. Measured across six families, n=64..512, both storage
layouts, every expert stack sits within 0.007% of a norm-matched Gaussian under
the participation ratio -- a SECOND-ORDER statistic. Uncorrelated is not
independent. A spectrum cannot see higher-order dependence, and nothing here
has looked yet.

This probe asks whether the experts have a smaller joint description even
though they do not lie in a low-dimensional linear span. The cheapest place
that difference shows up is the fourth-moment contraction along the feature
axis: the participation of the elementwise-squared stack.

Why a spectrum is blind to this. G_ik = <X_i, X_k> vanishes (up to 1/sqrt(d)
noise) for independent signs times a SHARED magnitude profile m: the linear
Gram is spectrally flat, participation ≈ n-1. The energy Gram
Γ_ik = <X_i², X_k²> is rank-1. That construction -- shared |m|, independent
signs -- is the control that justifies the instrument: the spectrum reports
DEAD, this probe reports LIVE. If we could not construct that case, the probe
would add nothing over the spectrum.

The null is a PER-ROW COORDINATE PERMUTATION of the energy stack, not a
Gaussian with matching energy-row-norms. Permutation preserves every marginal
of every expert's magnitude histogram (L2, kurtosis, sparsity) and destroys
only cross-expert coordinate alignment. A Gaussian energy-null does not
preserve those marginals: on Gaussian rows with exp(N(0,2)) scales it reported
+7.8% "sharing" that the permutation null scores at -0.34%. Same defect class
as the unmatched-norm Gaussian that licensed a 1.27% linear-sharing claim on
Qwen3-30B-A3B.

Linear structure is also caught -- a rank-r stack's squares live in the span
of pairwise products of the basis, rank ≤ r(r+1)/2 -- so a miss on a rank-3
organ would mean this probe is weaker than the spectrum, not complementary.

Several null draws, deficit_stderr_pct reported. LIVE requires the deficit
both to clear the 2% bar AND to exceed 2x the null's own standard error; two
organs 0.04 pp apart were once ranked when the instrument could not separate
them.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any

import numpy as np

from dense_anatomy import (
    LIVE_DEFICIT_PCT,
    NULL_SEED,
    NULL_SEEDS,
    _centred_gram,
    _participation_gram,
    _payload_offset,
    peak_rss_bytes,
)
from lake_scheme_census import FLOAT_DTYPES, read_header
from representational_anatomy import (
    AnatomyUnavailable,
    _load_float_tensor,
    resolve_layer,
    scan_expert_index,
)

# Same bar as spectral_null / dense_anatomy. A constant without the null's
# stderr is how 0.95 became the noise floor at n=20.
LIVE_BAR_PCT = LIVE_DEFICIT_PCT
# Peak stack we will hold (real + one null copy) must fit under 8 GiB RSS
# with room for the interpreter and the shard-read buffer.
MAX_STACK_BYTES = 3 * 2**30


class NonlinearProbeUnavailable(RuntimeError):
    """No probe could be run, and saying so is the only honest answer."""


# ---------------------------------------------------------------------------
# statistic
# ---------------------------------------------------------------------------

def _row_norms(X: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float32)
    return np.sqrt((X * X).sum(axis=1, keepdims=True))


def _stack_participation(X: np.ndarray) -> float:
    """Centred participation of the rows. Zero Gram (exact shared mean) -> 1.0.

    eigvalsh of an all-zero matrix yields tot=0, empty positive set, exp(0)=1,
    which is the right answer: the mean direction was the whole signal and
    centering removed it, leaving one (now-absent) effective direction.
    """
    return _participation_gram(_centred_gram(X))


def _energy_stack(X: np.ndarray) -> np.ndarray:
    """Elementwise squares. This is the map the spectrum cannot see through."""
    return np.square(np.ascontiguousarray(X, dtype=np.float32))


def _permutation_null(Y: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute each row's coordinates.

    Preserves every marginal of every energy profile and destroys only
    cross-expert coordinate alignment. Replacing this with a Gaussian of
    matching energy-row-norms is the known false-positive.
    """
    return np.random.default_rng(seed).permuted(Y, axis=1)


def _gaussian_null_from_norms(norms: np.ndarray, d: int, seed: int) -> np.ndarray:
    """Gaussian rows carrying `norms`, built WITHOUT the real stack."""
    n = int(norms.shape[0])
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, d), dtype=np.float32)
    R *= norms / np.maximum(np.sqrt((R * R).sum(axis=1, keepdims=True)), 1e-30)
    return np.ascontiguousarray(R, dtype=np.float32)


def _calibrated_from_nulls(real: float, nulls: list[float], n: int, d: int) -> dict:
    null_mean = float(np.mean(nulls))
    deficits = [100.0 * (nu - real) / max(nu, 1e-30) for nu in nulls]
    stderr = (float(np.std(deficits, ddof=1) / np.sqrt(len(deficits)))
              if len(deficits) > 1 else float("nan"))
    return {
        "n": n,
        "d": d,
        "participation": round(real, 3),
        "ratio_n": round(real / n, 4),
        "ratio_n_minus_1": round(real / max(n - 1, 1), 4),
        "null_participation": round(null_mean, 3),
        "null_ratio_n": round(null_mean / n, 4),
        "null_seeds": len(nulls),
        "deficit_pct": round(float(np.mean(deficits)), 3),
        "deficit_stderr_pct": round(stderr, 4),
        "noise_floor_ratio_n": round((n - 1) / n, 4),
    }


def is_live(c: dict, bar: float = LIVE_BAR_PCT) -> bool:
    """LIVE only if the deficit clears the bar AND the null can resolve it."""
    d = float(c["deficit_pct"])
    se = float(c["deficit_stderr_pct"])
    if se != se:  # NaN: single draw, fall back to the bar alone
        se = 0.0
    return d >= bar and d > 2.0 * se


def verdict(energy: dict, linear: dict) -> str:
    """What the two statistics license together, in plain terms."""
    e_live, l_live = is_live(energy), is_live(linear)
    e_txt = (f"{energy['deficit_pct']:.2f}% from a per-row permutation null "
             f"(stderr {energy['deficit_stderr_pct']:.4f} pp, n={energy['n']})")
    l_txt = (f"{linear['deficit_pct']:.2f}% from a norm-matched Gaussian "
             f"(stderr {linear['deficit_stderr_pct']:.4f} pp)")
    if e_live and not l_live:
        return (f"non-linear sharing is LIVE: energy profiles sit {e_txt} "
                f"while the spectrum is DEAD at {l_txt} -- a 4th-order "
                f"(coordinate-aligned energy) description may pay")
    if e_live and l_live:
        return (f"non-linear probe is LIVE and so is the spectrum: energy {e_txt}; "
                f"linear {l_txt} -- this may be linear structure the energy "
                f"statistic also sees; it does not by itself license a non-linear "
                f"generator")
    if l_live and not e_live:
        return (f"spectrum is LIVE ({l_txt}) but energy profiles are DEAD "
                f"({e_txt}) -- this probe is weaker than the spectrum on this "
                f"stack, not complementary")
    return (f"non-linear sharing is DEAD: energy profiles sit {e_txt}; "
            f"spectrum {l_txt}. Experts are as energy-independent as randomly "
            f"aligned vectors with the same per-row magnitude histograms. "
            f"No 2nd-order or 4th-order joint description to exploit")


def _flag(energy: dict, linear: dict) -> str:
    e_live, l_live = is_live(energy), is_live(linear)
    if e_live and not l_live:
        return "NONLINEAR_LIVE"
    if e_live and l_live:
        return "BOTH_LIVE"
    if l_live:
        return "SPECTRUM_ONLY"
    return "DEAD"


def probe(X: np.ndarray, n_seeds: int = NULL_SEEDS, seed: int = NULL_SEED,
          consume: bool = False) -> dict:
    """Linear participation AND energy-profile participation of an expert stack.

    `X` is (n_experts, n_features), one flattened expert per row.
    consume=True squares in place after the linear half so a 3 GiB stack is
    not held twice (the lake path). Tests leave it False.
    The energy null is drawn from a disjoint seed range so a correlated RNG
    cannot make the two statistics look independent when they are not.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"probe needs a 2D stack, got shape {X.shape}")
    n, d = int(X.shape[0]), int(X.shape[1])
    if n < 3:
        raise ValueError(f"participation needs at least 3 rows, got {n}")

    lin_real = _stack_participation(X)
    lin_norms = _row_norms(X)
    if consume:
        Y = X
        np.square(Y, out=Y)
    else:
        Y = _energy_stack(X)
    en_real = _stack_participation(Y)

    n_seeds = max(1, int(n_seeds))
    lin_nulls = []
    for i in range(n_seeds):
        R = _gaussian_null_from_norms(lin_norms, d, seed + i)
        lin_nulls.append(_stack_participation(R))
        del R
    en_nulls = []
    for i in range(n_seeds):
        R = _permutation_null(Y, seed + 1000 + i)
        en_nulls.append(_stack_participation(R))
        del R
    del Y

    linear = _calibrated_from_nulls(lin_real, lin_nulls, n, d)
    energy = _calibrated_from_nulls(en_real, en_nulls, n, d)
    linear["statistic"] = "centred participation of X vs a norm-matched Gaussian"
    energy["statistic"] = ("centred participation of X⊙X vs a per-row "
                           "coordinate-permutation null")
    linear["live"] = is_live(linear)
    energy["live"] = is_live(energy)
    return {
        "n": n,
        "d": d,
        "linear": linear,
        "energy": energy,
        "verdict": verdict(energy, linear),
        "flag": _flag(energy, linear),
        "live_bar_pct": LIVE_BAR_PCT,
    }


# ---------------------------------------------------------------------------
# synthetic controls -- shared by selfcheck and the test module
# ---------------------------------------------------------------------------

CONTROL_N = 20
CONTROL_D = 2048


def make_gaussian(n: int, d: int, rng: np.random.Generator,
                  scales: np.ndarray | None = None) -> np.ndarray:
    X = rng.standard_normal((n, d), dtype=np.float32)
    if scales is not None:
        X = X * np.asarray(scales, dtype=np.float32).reshape(n, 1)
    return np.ascontiguousarray(X, dtype=np.float32)


def make_linear(n: int, d: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    B = rng.standard_normal((rank, d), dtype=np.float32)
    C = rng.standard_normal((n, rank), dtype=np.float32)
    return np.ascontiguousarray(C @ B, dtype=np.float32)


def make_sign_magnitude(n: int, d: int, rng: np.random.Generator,
                        shared: bool = True) -> np.ndarray:
    """Independent signs times a magnitude profile.

    shared=True is the 4th control: spectrally flat, energy rank-1.
    shared=False is independent |m| per row -- energy DEAD, a negative.
    """
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, d))
    if shared:
        m = np.abs(rng.standard_normal(d).astype(np.float32)) + 0.1
        return np.ascontiguousarray(signs * m, dtype=np.float32)
    m = np.abs(rng.standard_normal((n, d)).astype(np.float32)) + 0.1
    return np.ascontiguousarray(signs * m, dtype=np.float32)


def make_sparse_mixture(n: int, d: int, rng: np.random.Generator,
                        n_spikes: int = 32, spike_scale: float = 5.0) -> np.ndarray:
    """Half the rows are dense Gaussian, half are k-sparse spikes.

    Independent across rows -- energy alignment is DEAD under a permutation
    null. A Gaussian energy-null of matching energy-row-norms reports LIVE
    here (measured +6.8% at n=20, d=2048) because it does not preserve the
    sparse magnitude histogram. This is the construction that makes that
    false-positive a failing test rather than a comment.
    """
    X = rng.standard_normal((n, d), dtype=np.float32)
    sparse = np.arange(0, n, 2)
    X[sparse] = 0
    n_spikes = min(int(n_spikes), d)
    idx = rng.integers(0, d, size=(len(sparse), n_spikes))
    spikes = rng.standard_normal(idx.shape).astype(np.float32) * spike_scale
    for i, row in enumerate(sparse):
        X[row, idx[i]] = spikes[i]
    return np.ascontiguousarray(X, dtype=np.float32)


def make_tanh_shared(n: int, d: int, rank: int, rng: np.random.Generator,
                     scale: float = 2.0) -> np.ndarray:
    """Rows are tanh(z @ W): a non-linear image of a shared rank-k map.

    Detectable by BOTH probes -- the positive control, not the 4th.
    """
    z = rng.standard_normal((n, rank), dtype=np.float32)
    W = rng.standard_normal((rank, d), dtype=np.float32)
    return np.ascontiguousarray(np.tanh(z @ W * scale), dtype=np.float32)


# ---------------------------------------------------------------------------
# lake loader -- headers via census, one shard resident at a time
# ---------------------------------------------------------------------------

def _pick_projection(layer_groups: dict, projection: str | None) -> str:
    if projection is not None:
        if projection not in layer_groups:
            raise NonlinearProbeUnavailable(
                f"projection {projection!r} not in this layer "
                f"(have {sorted(layer_groups)})"
            )
        return projection
    for cand in ("down_proj", "w2", "w2_weight"):
        if cand in layer_groups:
            return cand
    return sorted(layer_groups)[0]


def _feature_take(d: int, max_features: int | None, seed: int) -> np.ndarray | None:
    if max_features is None or int(max_features) >= d:
        return None
    take = max(8, int(max_features))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(d, size=take, replace=False))


def load_expert_stack(snapshot: str, layer: int | None = 0,
                      projection: str | None = None,
                      max_experts: int | None = None,
                      max_features: int | None = None,
                      feature_seed: int = 0) -> tuple[np.ndarray, dict]:
    """One layer's expert organ as an (n, d) float32 stack.

    Headers through lake_scheme_census.read_header; payload one tensor at a
    time so a 4 GiB shard is never mapped. Peak working set is the stack,
    not the body.
    """
    snapshot = os.path.abspath(os.path.expanduser(str(snapshot))).rstrip("/")
    if not os.path.isdir(snapshot):
        raise NonlinearProbeUnavailable(f"{snapshot}: not a directory")
    shards = sorted(glob.glob(snapshot + "/*.safetensors"))
    if not shards:
        shards = sorted(glob.glob(snapshot + "/**/*.safetensors", recursive=True))
    if not shards:
        raise NonlinearProbeUnavailable(
            f"{snapshot}: no .safetensors weights (searched recursively)"
        )

    index: dict[str, tuple[str, dict]] = {}
    offsets: dict[str, int] = {}
    for f in shards:
        try:
            hdr = read_header(f)
        except (ValueError, json.JSONDecodeError) as exc:
            raise NonlinearProbeUnavailable(
                f"{f}: unreadable safetensors header ({exc})"
            ) from exc
        offsets[f] = _payload_offset(f)
        for k, e in hdr.items():
            index[k] = (f, e)
    if not index:
        raise NonlinearProbeUnavailable(
            f"{snapshot}: {len(shards)} shards, zero tensors in their headers"
        )

    try:
        by_layer, schemes_at, _shared = scan_expert_index(index)
        layer_i = resolve_layer(by_layer, layer)
    except (KeyError, AnatomyUnavailable) as exc:
        raise NonlinearProbeUnavailable(
            f"{snapshot}: no measurable expert organ at layer {layer} ({exc})"
        ) from exc

    layer_groups = by_layer[layer_i]
    proj = _pick_projection(layer_groups, projection)
    emap = layer_groups[proj]
    for eid, key in emap.items():
        dt = index[key][1].get("dtype", "")
        if dt not in FLOAT_DTYPES:
            raise NonlinearProbeUnavailable(
                f"{snapshot}: expert organ is PRE-QUANTIZED ({proj} is {dt}); "
                "a 4th-moment over quantization codes measures the codebook"
            )

    indexed = any(eid is not None for eid in emap)
    storage = "per-expert" if indexed else "stacked"
    scheme = ",".join(sorted(schemes_at.get(layer_i) or [])) or "?"

    def _apply_features(row: np.ndarray, feat_idx: np.ndarray | None) -> np.ndarray:
        row = row.reshape(-1)
        return row[feat_idx] if feat_idx is not None else row

    if indexed:
        ids = sorted(eid for eid in emap if eid is not None)
        if max_experts is not None:
            ids = ids[: max(3, int(max_experts))]
        first_shape = [int(x) for x in index[emap[ids[0]]][1]["shape"]]
        d_full = 1
        for s in first_shape:
            d_full *= s
        feat_idx = _feature_take(d_full, max_features, feature_seed)
        d = int(feat_idx.shape[0]) if feat_idx is not None else d_full
        n = len(ids)
        if n * d * 4 > MAX_STACK_BYTES:
            # shrink features, never experts: the statistic is on n rows
            cap = max(8, MAX_STACK_BYTES // (4 * n))
            feat_idx = _feature_take(d_full, cap, feature_seed)
            d = int(feat_idx.shape[0])
        X = np.empty((n, d), dtype=np.float32)
        by_shard: dict[str, list[tuple[int, int]]] = {}
        for i, eid in enumerate(ids):
            path = index[emap[eid]][0]
            by_shard.setdefault(path, []).append((i, eid))
        for path, items in by_shard.items():
            items.sort(key=lambda t: int(index[emap[t[1]]][1]["data_offsets"][0]))
            header_len = offsets[path] - 8
            for i, eid in items:
                # Streamed decoder: a BF16 tensor is never held as raw+u32+f32.
                W = _load_float_tensor(path, header_len, index[emap[eid]][1])
                X[i] = _apply_features(W, feat_idx)
                del W
        n_total = len([e for e in emap if e is not None])
    else:
        key = emap[None]
        path, entry = index[key]
        W = _load_float_tensor(path, offsets[path] - 8, entry)
        n_total = int(W.shape[0])
        X = np.ascontiguousarray(W.reshape(n_total, -1), dtype=np.float32)
        del W
        if max_experts is not None and int(max_experts) < X.shape[0]:
            X = np.ascontiguousarray(X[: max(3, int(max_experts))], dtype=np.float32)
        d_full = int(X.shape[1])
        feat_idx = _feature_take(d_full, max_features, feature_seed)
        if feat_idx is not None:
            X = np.ascontiguousarray(X[:, feat_idx], dtype=np.float32)
        if X.nbytes > MAX_STACK_BYTES:
            cap = max(8, MAX_STACK_BYTES // (4 * int(X.shape[0])))
            feat_idx = _feature_take(int(X.shape[1]), cap, feature_seed)
            X = np.ascontiguousarray(X[:, feat_idx], dtype=np.float32)
        n, d = int(X.shape[0]), int(X.shape[1])

    n, d = int(X.shape[0]), int(X.shape[1])
    meta = {
        "snapshot": snapshot,
        "layer": layer_i,
        "projection": proj,
        "n": n,
        "d": d,
        "n_experts_total": n_total,
        "storage": storage,
        "scheme": scheme,
        "n_shards": len(shards),
        "subsampled_features": bool(max_features is not None and max_features < d_full)
                               or (d < d_full),
        "d_full": d_full,
    }
    return X, meta


def probe_snapshot(snapshot: str, layer: int | None = 0,
                   projection: str | None = None,
                   n_seeds: int = NULL_SEEDS,
                   max_experts: int | None = None,
                   max_features: int | None = None) -> dict:
    sys.stderr.write(f"loading {snapshot} layer={layer} proj={projection}\n")
    sys.stderr.flush()
    X, meta = load_expert_stack(
        snapshot, layer=layer, projection=projection,
        max_experts=max_experts, max_features=max_features,
    )
    sys.stderr.write(
        f"loaded n={X.shape[0]} d={X.shape[1]} storage={meta['storage']} "
        f"rss={peak_rss_bytes() / 2**30:.2f}GiB -- probing\n"
    )
    sys.stderr.flush()
    out = probe(X, n_seeds=n_seeds, consume=True)
    del X
    out["meta"] = meta
    out["peak_rss_gib"] = round(peak_rss_bytes() / 2**30, 3)
    return out


# ---------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------

def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    n, d = CONTROL_N, CONTROL_D

    # 1. Negative: iid Gaussian is DEAD on both. A false LIVE here is the
    #    0.95-at-n=20 defect repeating at the fourth moment.
    g = probe(make_gaussian(n, d, rng))
    assert not g["linear"]["live"], g["linear"]
    assert not g["energy"]["live"], g["energy"]
    assert abs(g["energy"]["deficit_pct"]) < 2.0, g["energy"]
    assert abs(g["linear"]["deficit_pct"]) < 2.0, g["linear"]
    assert g["flag"] == "DEAD", g
    assert "non-linear sharing is DEAD" in g["verdict"], g["verdict"]

    # 2. Positive: tanh of a shared rank-3 map is DETECTED (by both, that's OK).
    t = probe(make_tanh_shared(n, d, 3, rng))
    assert t["energy"]["live"], t["energy"]
    assert t["energy"]["deficit_pct"] > 20.0, t["energy"]

    # 3. Linear-but-not-only: rank-3 is LIVE on the spectrum AND on energy.
    #    A miss here means this probe is weaker than the spectrum.
    lin = probe(make_linear(n, d, 3, rng))
    assert lin["linear"]["live"], lin["linear"]
    assert lin["energy"]["live"], lin["energy"]
    assert lin["linear"]["deficit_pct"] > 50.0, lin["linear"]
    assert lin["flag"] == "BOTH_LIVE", lin

    # 4. THE control: shared |m|, independent signs. Spectrum DEAD, energy LIVE.
    #    If this fails, the probe adds nothing over the spectrum.
    sm = probe(make_sign_magnitude(n, d, rng, shared=True))
    assert not sm["linear"]["live"], ("spectrum saw sign-magnitude; d too small?", sm["linear"])
    assert sm["energy"]["live"], sm["energy"]
    assert sm["energy"]["deficit_pct"] > 50.0, sm["energy"]
    assert sm["flag"] == "NONLINEAR_LIVE", sm
    assert "non-linear sharing is LIVE" in sm["verdict"], sm["verdict"]

    # Independent |m| per row must NOT read as sharing -- otherwise any
    # non-Gaussian marginal looks like structure.
    ind = probe(make_sign_magnitude(n, d, rng, shared=False))
    assert not ind["energy"]["live"], ind["energy"]
    assert abs(ind["energy"]["deficit_pct"]) < 2.0, ind["energy"]

    # Heterogeneous row norms must not fool the energy null.
    scales = np.exp(rng.standard_normal((n, 1), dtype=np.float32) * 2.0)
    het = probe(make_gaussian(n, d, rng, scales=scales))
    assert not het["energy"]["live"], het["energy"]
    assert abs(het["energy"]["deficit_pct"]) < 2.0, het["energy"]

    # Sparse/dense mixture: permutation DEAD; a Gaussian energy-null is LIVE.
    sp = probe(make_sparse_mixture(n, d, rng))
    assert not sp["energy"]["live"], sp["energy"]
    assert abs(sp["energy"]["deficit_pct"]) < 2.0, sp["energy"]

    # consume=True must not change the number, or the lake path is a
    # different instrument from the tests.
    Xc = make_gaussian(n, d, np.random.default_rng(11))
    a = probe(Xc, consume=False)
    b = probe(np.array(Xc, copy=True), consume=True)
    assert a["energy"]["deficit_pct"] == b["energy"]["deficit_pct"], (a, b)
    assert a["linear"]["deficit_pct"] == b["linear"]["deficit_pct"], (a, b)

    # The null carries its own noise: stderr is a number, and a multi-draw
    # null cannot report zero spread on Gaussian data (the draws differ).
    assert g["energy"]["null_seeds"] >= 2, g["energy"]
    assert g["energy"]["deficit_stderr_pct"] == g["energy"]["deficit_stderr_pct"], "stderr NaN"
    assert g["energy"]["deficit_stderr_pct"] > 0.0, g["energy"]
    assert g["linear"]["deficit_stderr_pct"] > 0.0, g["linear"]

    # Loader: a tiny safetensors organ round-trips into probe().
    import tempfile
    from dense_anatomy import _write_safetensors
    with tempfile.TemporaryDirectory() as td:
        tensors = {
            f"model.layers.0.mlp.experts.{i}.down_proj.weight":
                rng.standard_normal((16, 8), dtype=np.float32)
            for i in range(8)
        }
        _write_safetensors(os.path.join(td, "m.safetensors"), tensors)
        X, meta = load_expert_stack(td, layer=0, projection="down_proj")
        assert X.shape == (8, 128), (X.shape, meta)
        assert meta["storage"] == "per-expert", meta
        r = probe(X)
        assert r["n"] == 8 and r["d"] == 128, r
        # I8 is refused, not measured.
        _write_safetensors(os.path.join(td, "m.safetensors"), {
            f"model.layers.0.mlp.experts.{i}.down_proj.weight":
                np.zeros((8, 8), dtype=np.int8) for i in range(4)
        })
        try:
            load_expert_stack(td, layer=0)
            raise AssertionError("I8 payload was measured as if it were the organism")
        except NonlinearProbeUnavailable as exc:
            assert "PRE-QUANTIZED" in str(exc) and "I8" in str(exc), exc

    print(
        "selfcheck OK -- Gaussian DEAD; tanh detected; rank-3 LIVE on both; "
        f"sign-magnitude spectrum {sm['linear']['deficit_pct']:+.2f}% / "
        f"energy {sm['energy']['deficit_pct']:+.1f}% (NONLINEAR_LIVE); "
        "het-norms not misread; permutation null reports stderr"
    )


_LAKE = "/Volumes/corpdrive/hawking-modellake/specimens"

# The G034 bodies: linear sharing is dead on every one. If energy is also
# dead across this set, the 4th-moment negative is a family law, not a
# one-organ anecdote. Layer/projection match the linear receipt.
_G034_MATRIX = [
    ("moonshotai--Kimi-VL-A3B-Instruct@398eede0903c", 1, "down_proj"),
    ("Qwen--Qwen3-30B-A3B@ad44e777bcd1", 0, "down_proj"),
    ("LiquidAI--LFM2-24B-A2B@a3bbacd91a67", 2, "w1"),
    ("Qwen--Qwen3-Coder-30B-A3B-Instruct@b2cff646eb4b", 0, "down_proj"),
    ("Qwen--Qwen3-VL-30B-A3B-Instruct@9c4b90e1e4ba", 0, "down_proj"),
    ("moonshotai--Kimi-Linear-48B-A3B-Instruct@e1df551a4471", 1, "w1"),
    ("zai-org--GLM-4.5-Air@a24ceef6ce4f", 1, "down_proj"),
    ("windowsxp811203--Qwen3.8-Flash-Next-Abliterated@deb02632504b", 0, "down_proj"),
]


def _matrix(lake: str = _LAKE, seeds: int = NULL_SEEDS) -> int:
    """One child process per body so RSS does not accumulate across organs."""
    import subprocess
    if not os.path.isdir(lake):
        sys.stderr.write(f"LAKE ABSENT -- {lake} is not a directory. Matrix DID NOT RUN.\n")
        return 2
    rc = 0
    for slug, layer, proj in _G034_MATRIX:
        path = os.path.join(lake, slug)
        if not os.path.isdir(path):
            print(f"{slug:64s}  ABSENT", flush=True)
            rc = 1
            continue
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--snapshot", path, "--layer", str(layer), "--projection", proj,
             "--seeds", str(seeds)],
            check=False,
        )
        if proc.returncode != 0:
            print(f"{slug:64s}  REFUSAL  child={proc.returncode}", flush=True)
            rc = proc.returncode or 1
    return rc


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="nonlinear_probe",
        description="Cheap 4th-moment discriminator for cross-expert non-linear sharing.",
    )
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--matrix", action="store_true",
                   help="run the G034 MoE-float set, one body per child")
    p.add_argument("--snapshot", help="directory of .safetensors shards")
    p.add_argument("--layer", default="0",
                   help="layer index, or 'auto' for the lowest complete expert group")
    p.add_argument("--projection", default=None,
                   help="expert projection (default: down_proj if present)")
    p.add_argument("--seeds", type=int, default=NULL_SEEDS)
    p.add_argument("--max-features", type=int, default=None)
    p.add_argument("--max-experts", type=int, default=None)
    p.add_argument("--lake", default=_LAKE)
    args = p.parse_args(argv)
    if args.selfcheck:
        _selfcheck()
        return 0
    if args.matrix:
        return _matrix(args.lake, seeds=args.seeds)
    if not args.snapshot:
        p.print_help()
        return 2
    layer: int | None
    if str(args.layer).lower() in ("auto", "none"):
        layer = None
    else:
        layer = int(args.layer)
    try:
        out = probe_snapshot(
            args.snapshot, layer=layer, projection=args.projection,
            n_seeds=args.seeds, max_experts=args.max_experts,
            max_features=args.max_features,
        )
    except NonlinearProbeUnavailable as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    # The verdict is the thing a human (or a campaign driver) actually wants.
    sys.stderr.write(out["verdict"] + "\n")
    sys.stderr.write(
        f"peak_rss={out['peak_rss_gib']:.2f}GiB  "
        f"n={out['n']} d={out['d']}  flag={out['flag']}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
