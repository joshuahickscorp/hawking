#!/usr/bin/env python3.12
"""These pin the non-negotiable contracts of the program:"""
from __future__ import annotations
import sys
from pathlib import Path as _Path_repo
_REPO = _Path_repo(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pathlib
import sys
import time

import numpy as np
import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]

from lab.operators import glm52_activation_aware_pack as aap  # noqa: E402

def _synthetic_basis(layer: int = 10, max_rank: int = 32, n_act: int = 512):
    rng = np.random.default_rng(0xA17A7E ^ layer)
    X = rng.standard_normal((n_act, aap.HIDDEN)).astype(np.float32)
    mu = X.mean(0)
    _u, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    max_rank = min(max_rank, vt.shape[0])
    prov = aap.BasisProvenance(
        tensor_layer=layer,
        basis_layer=layer,
        capsule_file=f"L{layer:02d}_L{layer:02d}.npz",
        capsule_key=f"layer_{layer}/pre_router_hidden",
        hidden=aap.HIDDEN,
        n_activation_rows=n_act,
    )
    return aap.ActivationBasis(
        layer=layer,
        basis=vt[:max_rank].T.astype(np.float32),
        singular_values=s[:max_rank].astype(np.float32),
        variance_frac=[
            float(x) for x in np.cumsum(s[:max_rank] ** 2) / (float(np.sum(s ** 2)) + 1e-30)
        ],
        provenance=prov,
        X_hold=X[int(n_act * 0.8):],
        X_fit_mean=mu.astype(np.float32),
    )

def _measured_bundle():
    rng = np.random.default_rng(1)
    basis = _synthetic_basis()
    ranks = (8, 16, 32)
    rows = []
    for i, name in enumerate((
        "model.layers.10.mlp.experts.0.up_proj.weight",
        "model.layers.10.mlp.experts.0.gate_proj.weight",
        "model.layers.10.mlp.experts.0.down_proj.weight",
        "model.layers.10.input_layernorm.weight",
    )):
        if "down_proj" in name:
            W = rng.standard_normal((aap.HIDDEN, 128)).astype(np.float32)
        elif "layernorm" in name:
            W = rng.standard_normal((aap.HIDDEN,)).astype(np.float32)
        else:
            W = rng.standard_normal((128, aap.HIDDEN)).astype(np.float32)
        rows.append(aap.measure_tensor(
            name, W, "BF16", basis, ranks, bill_basis_per_tensor=False,
        ).as_dict())
    return rows

def test_byte_accounting_reconciles_on_payload():
    rng = np.random.default_rng(0)
    m, n, r = 64, aap.HIDDEN, 16
    W = rng.standard_normal((m, n)).astype(np.float32)
    B = rng.standard_normal((n, r)).astype(np.float32)
    B, _ = np.linalg.qr(B)
    L = aap.project_factors(W, B, "input")
    blob = aap.serialize_tensor_payload(
        L, B, side="input", rows=m, cols=n, rank=r,
        basis_layer=10, bill_basis=True,
    )
    parts = aap.factor_bytes(m, n, r, "input")
    assert len(blob) == parts["header"] + parts["coefficients"] + parts["basis"]

    ledger = aap.ByteLedger()
    ledger.add("metadata", parts["header"])
    ledger.add("codebooks", parts["coefficients"] + parts["basis"])
    assert ledger.reconciles(len(blob))
    assert ledger.total_bytes() == len(blob)
    assert sum(ledger.components.values()) == ledger.total_bytes()

    bpw = ledger.complete_bpw(m * n)
    assert bpw.numerator > 0 and bpw.denominator > 0
    # Exact rational string form used by the Math-Preserve receipt style.
    doc = ledger.as_dict(m * n)
    assert doc["complete_bpw_exact"] == f"{bpw.numerator}/{bpw.denominator}"
    assert doc["itemization_reconciles"] is True

def test_basis_provenance_present_on_measured_and_allocated_tensors():
    rows = _measured_bundle()
    aa = [r for r in rows if r["disposition"] == "activation_aware"]
    assert aa, "expected at least one activation-aware tensor"
    for r in aa:
        prov = r["basis_provenance"]
        assert prov is not None
        assert "basis_layer" in prov
        assert "capsule_file" in prov
        assert "capsule_key" in prov
        assert prov["capsule_key"].endswith("pre_router_hidden")
        assert prov["basis_layer"] == 10

    alloc = aap.allocate(rows, aap.Fraction(1, 2), shared_bases=True)
    for row in alloc["allocations"]:
        if row["disposition"] == "activation_aware":
            assert row["basis_provenance"] is not None
            assert row["basis_provenance"]["basis_layer"] == 10
    # Ledger still reconciles after allocation.
    assert alloc["byte_ledger"]["itemization_reconciles"]
    assert sum(alloc["byte_ledger"]["component_bytes"].values()) == alloc["total_bytes"]

def test_disk_floor_halts_rather_than_overruns():
    free = aap.free_bytes()
    with pytest.raises(aap.DiskFloorError):
        aap.assert_disk_floor(extra_bytes=0, floor=free + 1)
    with pytest.raises(aap.DiskFloorError):
        # Even with free space, an extra write that would cross the floor is refused.
        aap.assert_disk_floor(extra_bytes=free, floor=1)
    # Under a zero floor the check is a pure free-space probe and must not raise.
    assert aap.assert_disk_floor(extra_bytes=0, floor=0) >= 0

def test_allocation_is_deterministic_for_same_inputs():
    rows = _measured_bundle()
    target = aap.Fraction(49, 50)
    a1 = aap.allocate(rows, target, shared_bases=True)
    a2 = aap.allocate(rows, target, shared_bases=True)
    ranks1 = [(x["name"], x["rank"], x["disposition"]) for x in a1["allocations"]]
    ranks2 = [(x["name"], x["rank"], x["disposition"]) for x in a2["allocations"]]
    assert ranks1 == ranks2
    assert a1["complete_bpw_exact"] == a2["complete_bpw_exact"]
    assert a1["total_bytes"] == a2["total_bytes"]
    assert a1["basis_rank_by_layer"] == a2["basis_rank_by_layer"]
    # A third run with a deep copy of the inputs still matches.
    import copy
    a3 = aap.allocate(copy.deepcopy(rows), target, shared_bases=True)
    assert [
        (x["name"], x["rank"], x["disposition"]) for x in a3["allocations"]
    ] == ranks1

def test_nearest_basis_layer_is_deterministic():
    avail = [0, 10, 11, 16, 22, 76]
    assert aap.nearest_basis_layer(10, avail) == 10
    assert aap.nearest_basis_layer(12, avail) == 11
    assert aap.nearest_basis_layer(14, avail) == 16
    assert aap.nearest_basis_layer(77, avail) == 76
    # Tie: equal distance prefers the lower index.
    assert aap.nearest_basis_layer(5, [0, 10]) == 0

def test_selftest_entry_point():
    assert aap.selftest() == 0

def test_pass_through_1d_and_output_side_projection():
    rng = np.random.default_rng(2)
    basis = _synthetic_basis()
    # 1-D
    pt = aap.measure_tensor(
        "model.layers.10.post_attention_layernorm.weight",
        rng.standard_normal((aap.HIDDEN,)).astype(np.float32),
        "BF16", basis, ranks=(8, 16),
    )
    assert pt.disposition == "pass_through"
    assert pt.curve[0]["mean_row_cosine"] == 1.0
    # down_proj: output side matches hidden
    down = aap.measure_tensor(
        "model.layers.10.mlp.experts.0.down_proj.weight",
        rng.standard_normal((aap.HIDDEN, 64)).astype(np.float32),
        "BF16", basis, ranks=(8, 16),
        bill_basis_per_tensor=False,
    )
    assert down.disposition == "activation_aware"
    assert down.side == "output"
    assert down.basis_provenance is not None

def test_parallel_measure_emits_in_shard_order_not_completion_order():
    """Allocation must not depend on which worker finished first. This is the real risk in parallelising th"""
    import glm52_activation_aware_pack as m

    shards = [1, 2, 3, 4]
    completion = [4, 2, 3, 1]  # deliberately not the request order

    results = {n: ([f"tensor-of-{n}"], 0.1, f"/tmp/shard{n}") for n in completion}
    per_shard = []
    measurements = []
    for n in shards:                      # the ordering contract, mirrored from phase_measure
        ms, secs, pth = results[n]
        measurements.extend(ms)
        per_shard.append({"shard": n, "path": pth, "n_tensors": len(ms)})

    assert [r["shard"] for r in per_shard] == shards
    assert measurements == [f"tensor-of-{n}" for n in shards]

def test_shared_basis_cache_builds_once_under_contention():
    """Two workers wanting the same layer must build its basis once. Building one is an eigendecomposition """
    import threading
    import glm52_activation_aware_pack as m

    cache = m._SharedBasisCache()
    builds = []
    lock = threading.Lock()

    def build():
        with lock:
            builds.append(1)
        return object()

    out = []
    threads = [threading.Thread(target=lambda: out.append(cache.get(10, build)))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(builds) == 1, f"basis built {len(builds)} times under contention"
    assert len({id(o) for o in out}) == 1, "workers received different basis objects"

def test_prefetcher_preserves_order_under_out_of_order_completion(tmp_path):
    """Pack order must follow the shard list, not which download finished first. Same contract as the measu"""
    import threading
    import time
    from pathlib import Path
    import glm52_activation_aware_pack as m

    shards = [1, 2, 3, 4]
    # Later shards finish first.
    delays = {1: 0.12, 2: 0.08, 3: 0.04, 4: 0.01}
    active = 0
    peak = 0
    lock = threading.Lock()
    deliver_order = []

    def ensure(n, source_dir, fetch=True, floor=0, body_bytes=0, reserve=True):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(delays[n])
        p = Path(source_dir) / f"model-{n:05d}-of-00282.safetensors"
        p.write_bytes(b"x")
        with lock:
            active -= 1
        return p

    pref = m._Prefetcher(
        shards, tmp_path, fetch=True, floor=0,
        workers=4, body_bytes=1, ensure=ensure,
    )
    try:
        got = []
        for n in shards:
            path = pref.get(n)
            deliver_order.append(n)
            got.append(path.name)
            pref.release(n)
    finally:
        pref.close()

    assert deliver_order == shards
    assert got == [f"model-{n:05d}-of-00282.safetensors" for n in shards]
    assert pref.peak_resident <= 4
    assert peak <= 4, f"concurrent ensure calls peaked at {peak}"

def test_prefetcher_residency_bounded_by_workers(tmp_path):
    """Residency must stay bounded by N and not grow with progress."""
    import threading
    import time
    from pathlib import Path
    import glm52_activation_aware_pack as m

    shards = list(range(1, 13))
    workers = 3
    active = 0
    peak = 0
    lock = threading.Lock()
    gate = threading.Barrier(workers)  # force workers to overlap once

    def ensure(n, source_dir, fetch=True, floor=0, body_bytes=0, reserve=True):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        # First wave overlaps; later waves just sleep briefly.
        if n <= workers:
            gate.wait()
        time.sleep(0.02)
        p = Path(source_dir) / f"shard-{n}"
        p.write_bytes(b"y")
        with lock:
            active -= 1
        return p

    pref = m._Prefetcher(
        shards, tmp_path, fetch=True, floor=0,
        workers=workers, body_bytes=1, ensure=ensure,
    )
    try:
        for n in shards:
            pref.get(n)
            # Simulate consumer holding the body briefly, then releasing (evict path).
            time.sleep(0.005)
            pref.release(n)
    finally:
        pref.close()

    assert peak <= workers, f"in-flight ensure peaked at {peak} > {workers}"
    assert pref.peak_resident <= workers, (
        f"occupancy peaked at {pref.peak_resident} > {workers}"
    )

def test_prefetcher_failed_shard_does_not_wedge_siblings(tmp_path):
    """A failed fetch must name its shard and let other in-flight fetches finish."""
    import time
    from pathlib import Path
    import glm52_activation_aware_pack as m

    shards = [10, 11, 12, 13]
    finished = []

    def ensure(n, source_dir, fetch=True, floor=0, body_bytes=0, reserve=True):
        time.sleep(0.03 if n != 11 else 0.01)
        if n == 11:
            raise RuntimeError("simulated CDN reset")
        p = Path(source_dir) / f"ok-{n}"
        p.write_bytes(b"z")
        finished.append(n)
        return p

    pref = m._Prefetcher(
        shards, tmp_path, fetch=True, floor=0,
        workers=4, body_bytes=1, ensure=ensure,
    )
    try:
        p10 = pref.get(10)
        assert p10.name == "ok-10"
        pref.release(10)
        with pytest.raises(m.PackError, match="shard 11"):
            pref.get(11)
        # Siblings admitted in the same window must still be deliverable.
        p12 = pref.get(12)
        assert p12.name == "ok-12"
        pref.release(12)
        p13 = pref.get(13)
        assert p13.name == "ok-13"
        pref.release(13)
    finally:
        pref.close()

    assert 12 in finished and 13 in finished
    assert 11 not in finished

def test_ensure_shard_floor_accounts_for_concurrent_reservations(tmp_path, monkeypatch):
    """N concurrent admissions must reserve N bodies, not one. Two threads each needing a body of size B ag"""
    import threading
    import glm52_activation_aware_pack as m

    body = 1_000_000
    floor = 10_000_000
    # free leaves room for exactly one body above the floor.
    free = floor + body + 1
    monkeypatch.setattr(m, "free_bytes", lambda path=None: free)

    # Reset process-wide reservation so the test is hermetic.
    with m._ENSURE_RESERVE_LOCK:
        m._ENSURE_RESERVED_BYTES = 0

    results = []

    def try_admit(tag):
        try:
            with m._ENSURE_RESERVE_LOCK:
                m.assert_disk_floor(
                    body + m._ENSURE_RESERVED_BYTES,
                    path=tmp_path,
                    floor=floor,
                )
                m._ENSURE_RESERVED_BYTES += body
            results.append((tag, "ok"))
            time.sleep(0.05)  # hold reservation
            with m._ENSURE_RESERVE_LOCK:
                m._ENSURE_RESERVED_BYTES -= body
        except m.DiskFloorError:
            results.append((tag, "floor"))

    t1 = threading.Thread(target=try_admit, args=("a",))
    t2 = threading.Thread(target=try_admit, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    statuses = sorted(s for _, s in results)
    assert statuses == ["floor", "ok"], results
    with m._ENSURE_RESERVE_LOCK:
        assert m._ENSURE_RESERVED_BYTES == 0
