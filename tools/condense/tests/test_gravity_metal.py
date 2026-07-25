"""Guards for the Metal decoder that do not need a Metal device.

The three defects this covers are all reachable without a GPU, which is the point: the
buffer-overrun guard, the cache identity and the byte accounting are pure functions of the
geometry, so they are tested as such.  The handful of assertions that genuinely need a
dispatch are skipped when the framework or the device is missing rather than failed, because
the test process is not guaranteed to have either.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_forge as forge  # noqa: E402
import gravity_metal as gm  # noqa: E402


def _codes(rows: int, cols: int, *, D: int = 8, k: int = 128, seed: int = 0) -> dict:
    """A real packed artifact's codes stash, so the shapes are the production ones."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((rows, cols)).astype(np.float32)
    art = forge.pack_product_quant(w, dim=D, subspaces=1, k=k, seed=seed, iters=2)
    return art.config["pq_codes"]


# R0, the production rung: gate/up is [2048, 6144] at dim=8/k=128, nchunk=768.
R0_ROWS, R0_COLS, R0_D, R0_K = 2048, 6144, 8, 128
R0_NCHUNK = R0_COLS // R0_D


def _r0_codes() -> dict:
    """Geometry-only stand-in: matvec_bytes reads shapes, never index values."""
    return {"D": R0_D, "S": 1, "sub": R0_D, "rows": R0_ROWS, "cols": R0_COLS,
            "nchunk": R0_NCHUNK, "rotate": False, "seed": 0,
            "codebooks": [np.zeros((R0_K, R0_D), dtype=np.float32)],
            "indices": np.zeros((R0_ROWS * R0_NCHUNK, 1), dtype=np.int64)}


def _decoder():
    try:
        return gm.decoder()
    except gm.MetalUnavailable as exc:
        pytest.skip(f"no Metal device in this process: {exc}")


# --- mandate 1.1, the buffer-overrun guard ---------------------------------------------

def test_exact_size_x_passes_the_guard():
    xv = gm._validate_x(np.ones(R0_NCHUNK * R0_D, dtype=np.float32),
                        nchunk=R0_NCHUNK, D=R0_D, allocated_bytes=R0_NCHUNK * R0_D * 4)
    assert xv.nbytes == R0_NCHUNK * R0_D * 4
    assert xv.dtype == np.float32


def test_short_x_is_rejected():
    with pytest.raises(gm.GravityMetalInputError, match="elements"):
        gm._validate_x(np.ones(R0_NCHUNK * R0_D - 1, dtype=np.float32),
                       nchunk=R0_NCHUNK, D=R0_D, allocated_bytes=R0_NCHUNK * R0_D * 4)


def test_long_x_is_rejected_before_any_pointer_is_taken():
    # this is the overrun: 4 KB past the end of a fixed nchunk*D*4 allocation
    with pytest.raises(gm.GravityMetalInputError, match="elements"):
        gm._validate_x(np.ones(R0_NCHUNK * R0_D + 1024, dtype=np.float32),
                       nchunk=R0_NCHUNK, D=R0_D, allocated_bytes=R0_NCHUNK * R0_D * 4)


def test_wrong_dtype_is_rejected():
    with pytest.raises(gm.GravityMetalInputError, match="float32"):
        gm._validate_x(np.ones(R0_NCHUNK * R0_D, dtype=np.float64),
                       nchunk=R0_NCHUNK, D=R0_D, allocated_bytes=R0_NCHUNK * R0_D * 4)


def test_wrong_geometry_is_rejected():
    # right element count for some other tensor, wrong one for this cache entry
    with pytest.raises(gm.GravityMetalInputError, match="allocated with"):
        gm._validate_x(np.ones(R0_NCHUNK * R0_D, dtype=np.float32),
                       nchunk=R0_NCHUNK, D=R0_D, allocated_bytes=256 * R0_D * 4)


def test_guard_fires_on_a_real_dispatch():
    gpu = _decoder()
    codes = _codes(256, 128, D=8, k=16)
    key = gm.content_key(codes)
    good = np.ones(128, dtype=np.float32)
    gpu.matvec(codes, good, key=key)
    with pytest.raises(gm.GravityMetalInputError):
        gpu.matvec(codes, np.ones(4096, dtype=np.float32), key=key)


# --- mandate 1.2, explicit cache identity ----------------------------------------------

def test_matvec_without_a_key_raises():
    gpu = _decoder()
    codes = _codes(256, 128, D=8, k=16)
    with pytest.raises(gm.GravityMetalInputError, match="explicit cache key"):
        gpu.matvec(codes, np.ones(128, dtype=np.float32))


def test_same_explicit_key_returns_the_same_entry():
    gpu = _decoder()
    codes = _codes(256, 128, D=8, k=16)
    first = gpu._cache_tensor(codes, "tensor-a")
    assert gpu._cache_tensor(codes, "tensor-a") is first


def test_different_keys_do_not_collide():
    gpu = _decoder()
    a, b = _codes(256, 128, D=8, k=16, seed=1), _codes(256, 128, D=8, k=16, seed=2)
    # keys are namespaced per test: gm.decoder() is a process singleton, so a key reused
    # across tests is the very collision the fingerprint check now refuses
    ea = gpu._cache_tensor(a, "collide-a")
    eb = gpu._cache_tensor(b, "collide-b")
    assert ea is not eb
    x = np.ones(128, dtype=np.float32)
    ya = gpu.matvec(a, x, key="collide-a")
    yb = gpu.matvec(b, x, key="collide-b")
    assert not np.allclose(ya, yb)


def test_reusing_one_key_for_two_tensors_is_refused():
    """The explicit-key rule moves uniqueness to the caller; this is what enforces it."""
    gpu = _decoder()
    a, b = _codes(256, 128, D=8, k=16, seed=5), _codes(256, 128, D=8, k=16, seed=6)
    gpu._cache_tensor(a, "shared-literal")
    with pytest.raises(gm.GravityMetalInputError, match="already holds a different tensor"):
        gpu._cache_tensor(b, "shared-literal")


def test_codebook_width_must_match_declared_D():
    """A narrow codebook is an out-of-bounds device read, not a wrong answer."""
    gpu = _decoder()
    codes = _codes(256, 128, D=8, k=16, seed=7)
    codes["D"] = 16
    with pytest.raises(gm.GravityMetalInputError, match="codebook subvector"):
        gpu._cache_tensor(codes, "narrow-book")


def test_index_beyond_codebook_is_refused_before_the_uint8_cast():
    """uint8 would wrap an out-of-range index and silently select the wrong codeword."""
    gpu = _decoder()
    codes = _codes(256, 128, D=8, k=16, seed=8)
    codes["indices"] = np.asarray(codes["indices"], dtype=np.int64).copy()
    codes["indices"][0] = 300
    with pytest.raises(gm.GravityMetalInputError, match="out of range"):
        gpu._cache_tensor(codes, "wrapping-index")


def test_content_key_is_stable_and_discriminating():
    a = _codes(128, 64, D=8, k=16, seed=3)
    # a byte-identical copy, not a re-pack: pack_product_quant's indices are reproducible but
    # its codebook centroids are not bit-identical across calls (measured 2.4e-7 drift), and
    # a content address is allowed to notice that -- different bytes are a different upload
    same = {**a, "indices": a["indices"].copy(),
            "codebooks": [cb.copy() for cb in a["codebooks"]]}
    other = _codes(128, 64, D=8, k=16, seed=4)
    assert gm.content_key(a) == gm.content_key(a)
    assert gm.content_key(a) == gm.content_key(same)
    assert gm.content_key(a) != gm.content_key(other)
    # a single flipped index has to move the address
    mutated = dict(a)
    idx = a["indices"].copy()
    idx[0, 0] = (idx[0, 0] + 1) % 16
    mutated["indices"] = idx
    assert gm.content_key(mutated) != gm.content_key(a)
    # and so does a single perturbed codeword
    mutated = dict(a)
    book = a["codebooks"][0].copy()
    book[0, 0] += 1.0
    mutated["codebooks"] = [book]
    assert gm.content_key(mutated) != gm.content_key(a)


# --- mandate 1.2, bounded eviction ------------------------------------------------------

def _entry(nbytes: int) -> dict:
    return {"pinned_bytes": nbytes}


def test_lru_evicts_past_the_budget_and_accounts_correctly():
    cache = gm._ByteBudgetCache(budget_bytes=250)
    cache.put("a", _entry(100))
    cache.put("b", _entry(100))
    assert cache.stats()["bytes_pinned"] == 200
    assert cache.stats()["evictions"] == 0
    cache.put("c", _entry(100))                        # 300 > 250, oldest goes
    assert cache.get("a") is None
    assert cache.stats() == {"entries": 2, "bytes_pinned": 200, "budget_bytes": 250,
                             "evictions": 1, "keys": ["b", "c"]}


def test_lru_evicts_by_recency_not_insertion():
    cache = gm._ByteBudgetCache(budget_bytes=250)
    cache.put("a", _entry(100))
    cache.put("b", _entry(100))
    cache.get("a")                                     # a is now the most recent
    cache.put("c", _entry(100))
    assert cache.get("b") is None
    assert cache.get("a") is not None


def test_reinserting_an_evicted_entry_works():
    cache = gm._ByteBudgetCache(budget_bytes=250)
    for key in ("a", "b", "c"):
        cache.put(key, _entry(100))
    fresh = cache.put("a", _entry(100))
    assert cache.get("a") is fresh
    assert cache.stats()["bytes_pinned"] == 200


def test_reput_of_a_live_key_does_not_double_count():
    cache = gm._ByteBudgetCache(budget_bytes=10_000)
    cache.put("a", _entry(100))
    cache.put("a", _entry(300))
    assert cache.stats() == {"entries": 1, "bytes_pinned": 300, "budget_bytes": 10_000,
                             "evictions": 0, "keys": ["a"]}


def test_an_entry_larger_than_the_budget_still_runs():
    cache = gm._ByteBudgetCache(budget_bytes=10)
    entry = cache.put("huge", _entry(1_000))
    assert cache.get("huge") is entry


def test_the_glm_walk_cannot_pin_ninety_gigabytes():
    # 78 layers x 256 experts of R0 gate/up indices is ~90 GB if nothing ever evicts
    cache = gm._ByteBudgetCache(budget_bytes=gm.DEFAULT_CACHE_BUDGET_BYTES)
    per_tensor = R0_ROWS * R0_NCHUNK
    for i in range(4000):
        cache.put(f"t{i}", _entry(per_tensor))
    assert cache.stats()["bytes_pinned"] <= gm.DEFAULT_CACHE_BUDGET_BYTES
    assert cache.stats()["evictions"] > 0


def test_decoder_exposes_the_accounting():
    gpu = _decoder()
    assert set(gpu.cache_stats) == {"entries", "bytes_pinned", "budget_bytes", "evictions",
                                    "keys"}
    assert gpu.cache_stats["budget_bytes"] == gm.DEFAULT_CACHE_BUDGET_BYTES


# --- mandate 1.4, honest byte accounting ------------------------------------------------

def test_r0_gate_up_traffic_is_the_measured_split():
    got = gm.matvec_bytes(_r0_codes())
    assert got["index_bits_billed"] == 7
    assert got["threadgroups"] == 8
    assert got["stage_x"] is True
    assert got["logical_index_bytes"] == 1_376_256
    assert got["logical_codebook_bytes"] == 2_048
    assert got["logical_artifact_bytes"] == 1_378_304
    assert got["executed_index_bytes"] == 1_572_864
    assert got["executed_codebook_bytes"] == 16_384
    assert got["executed_activation_bytes"] == 196_608
    assert got["executed_output_bytes"] == 8_192
    # the number the old scalar should have reported: 1,785,856 not 1,574,912
    assert got["executed_read_bytes"] == 1_785_856
    assert got["executed_total_bytes"] == 1_794_048
    assert got["dense_bf16_bytes"] == 25_165_824
    # the packed_bpw the campaign bills for R0 is 0.87633; the 2e-5 gap is the metadata
    # header, which this function deliberately does not charge to the matvec stream
    assert round(got["logical_bpw"], 5) == 0.87630
    assert round(got["executed_read_bytes"] / got["dense_bf16_bytes"] * 100, 2) == 7.10


def test_executed_exceeds_the_logical_artifact():
    got = gm.matvec_bytes(_r0_codes())
    ratio = got["executed_total_bytes"] / got["logical_artifact_bytes"]
    assert got["executed_total_bytes"] > got["logical_artifact_bytes"]
    assert round(ratio, 4) == round(1_794_048 / 1_378_304, 4) == 1.3016


def test_the_seven_to_eight_bit_gap_is_exactly_eight_sevenths():
    got = gm.matvec_bytes(_r0_codes())
    assert got["executed_index_bytes"] * 7 == got["logical_index_bytes"] * 8


def test_r0_down_projection_traffic():
    codes = _r0_codes()
    codes.update(rows=6144, cols=2048, nchunk=256,
                 indices=np.zeros((6144 * 256, 1), dtype=np.int64))
    got = gm.matvec_bytes(codes)
    assert got["threadgroups"] == 24
    assert got["stage_x"] is True
    assert got["executed_index_bytes"] == 1_572_864
    assert got["executed_codebook_bytes"] == 24 * 2_048
    assert got["executed_activation_bytes"] == 24 * 256 * 8 * 4
    assert got["executed_index_bytes"] * 7 == got["logical_index_bytes"] * 8


def test_unstaged_x_is_charged_once():
    # a tiny threadgroup allotment turns staging off; x is then billed a single read
    codes = _r0_codes()
    got = gm.matvec_bytes(codes, threadgroup_memory_limit=8192)
    assert got["stage_x"] is False
    assert got["executed_activation_bytes"] == R0_NCHUNK * R0_D * 4


def test_accounting_matches_a_real_artifact_and_refuses_multi_subspace():
    codes = _codes(256, 128, D=8, k=16)
    got = gm.matvec_bytes(codes)
    assert got["index_bits_billed"] == 4
    assert got["executed_index_bytes"] == 256 * 16
    assert got["logical_index_bytes"] == 256 * 16 // 2
    codes["S"] = 2
    with pytest.raises(gm.GravityMetalInputError, match="subspaces == 1"):
        gm.matvec_bytes(codes)


def test_bytes_read_per_matvec_returns_the_dict():
    gpu = _decoder()
    got = gpu.bytes_read_per_matvec(_r0_codes())
    assert isinstance(got, dict)
    assert got["executed_read_bytes"] == 1_785_856
