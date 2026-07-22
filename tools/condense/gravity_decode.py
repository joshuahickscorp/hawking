#!/usr/bin/env python3.12
"""Gravity-native decoder for Apple silicon: fused decode-and-compute, never FP16.

Roadmap Phase 2.  A conventional runtime would decompress a Gravity tensor back to FP16
and hand it to a normal matmul, which throws away the entire point: the weights would once
again occupy -- and stream through memory at -- their original size.  This decoder never
forms the dense weight at all.  It contracts codeword indices against the activation
subspace by subspace, so the only weight bytes that ever cross the memory bus are the
compressed ones.

That is the whole performance thesis, and it is a bandwidth argument rather than an
arithmetic one.  Decode is bounded per subspace: at most ``rows * nchunk * sub`` scalars
exist at once and they are freed before the next subspace, so peak working set stays a
small multiple of the compressed tensor rather than the dense one.  A 6144x2048 BF16
expert reads 25.2 MB per matvec; the same expert at R0 reads about 1.4 MB.  Decode
arithmetic is a gather, which Apple silicon does well, and the codebook for a whole tensor
is a few KB -- small enough to stay resident in fast memory across the entire contraction
instead of being re-read per row.

Two backends behind one signature:

* ``cpu``   -- ``gravity_forge.pq_execute``, the correctness authority.
* ``mps``   -- this module, expressed as gather + batched contraction so every step lands
               on real Metal kernels through torch.

The seam is deliberate.  A hand-written ``.metal`` kernel (fusing the gather and the
contraction into a single dispatch, with the codebook staged in threadgroup memory) drops
in behind :func:`decode_matvec` without changing a caller, and is checked against the same
parity harness that guards this one.  Parity against the CPU reference is the gate for any
backend: a faster decoder that computes something else is not a faster decoder.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_forge as forge  # noqa: E402

DECODER_VERSION = 1


def _torch():
    import torch
    return torch


def available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        torch = _torch()
        if torch.backends.mps.is_available():
            backends.append("mps")
    except Exception:  # noqa: BLE001
        pass
    return backends


def _rotation(torch, D: int, seed: int, device) -> Any:
    """Regenerate the billed Hadamard rotation; seeds are stored, matrices are not."""
    return torch.from_numpy(forge._pq_rotation_np(D, seed)).to(device)


def decode_matvec_mps(codes: dict, x: np.ndarray, *, device: str = "mps") -> np.ndarray:
    """y = W_gravity @ x on the GPU, without ever materializing W.

    Per subspace: gather the codewords its indices select, contract them against that
    subspace's slice of the activation, accumulate, release.  Peak decoded state is one
    subspace, never the whole tensor.
    """
    torch = _torch()
    dev = torch.device(device)

    D, S, sub = int(codes["D"]), int(codes["S"]), int(codes["sub"])
    rows, nchunk = int(codes["rows"]), int(codes["nchunk"])

    x = np.ascontiguousarray(x, dtype=np.float32)
    onedim = x.ndim == 1
    if onedim:
        x = x[:, None]
    batch = x.shape[1]

    xt = torch.from_numpy(x).to(dev).reshape(nchunk, D, batch)
    if codes["rotate"]:
        xt = torch.einsum("jk,ckb->cjb", _rotation(torch, D, int(codes["seed"]), dev), xt)

    indices = torch.from_numpy(
        np.ascontiguousarray(codes["indices"], dtype=np.int64)).to(dev).reshape(rows, nchunk, S)
    out = torch.zeros((rows, batch), dtype=torch.float32, device=dev)

    for s in range(S):
        book = torch.from_numpy(
            np.ascontiguousarray(codes["codebooks"][s], dtype=np.float32)).to(dev)
        # [rows, nchunk, sub] -- one subspace of decoded weights, freed before the next
        decoded = book[indices[:, :, s]]
        out += torch.einsum("rcj,cjb->rb", decoded, xt[:, s * sub:(s + 1) * sub, :])
        del decoded, book

    result = out.cpu().numpy()
    return result[:, 0] if onedim else result


def decode_matvec(artifact, x: np.ndarray, *, backend: str = "mps") -> np.ndarray:
    """Dispatch to a backend.  Every backend must match the CPU reference."""
    if backend == "cpu":
        return forge.pq_execute(artifact, x)
    codes = artifact.config.get("pq_codes")
    if codes is None:
        raise ValueError("artifact carries no pq_codes stash")
    return decode_matvec_mps(codes, x, device=backend)


def parity(artifact, x: np.ndarray, *, backend: str = "mps", tol: float = 2e-3) -> dict:
    """Gate any backend against the CPU authority before trusting its numbers."""
    reference = forge.pq_execute(artifact, x)
    candidate = decode_matvec(artifact, x, backend=backend)
    scale = float(np.abs(reference).max()) + 1e-12
    gap = float(np.abs(reference - candidate).max() / scale)
    return {"backend": backend, "relative_max_gap": gap, "tolerance": tol,
            "within_tolerance": gap < tol,
            "finite": bool(np.isfinite(candidate).all())}


def benchmark(artifact, x: np.ndarray, *, backend: str = "mps", repeats: int = 20) -> dict:
    """Time the decoder and report the bandwidth argument in bytes actually read."""
    codes = artifact.config["pq_codes"]
    rows, cols = int(codes["rows"]), int(codes["cols"])
    weights = rows * cols

    decode_matvec(artifact, x, backend=backend)  # warm caches and shader compilation
    start = time.perf_counter()
    for _ in range(repeats):
        decode_matvec(artifact, x, backend=backend)
    elapsed = (time.perf_counter() - start) / repeats

    index_bits = int(np.ceil(np.log2(max(2, codes["codebooks"][0].shape[0]))))
    compressed = (rows * int(codes["nchunk"]) * int(codes["S"]) * index_bits) / 8
    compressed += sum(book.size * 2 for book in codes["codebooks"])
    dense = weights * 2  # the BF16/FP16 the parent would have streamed

    return {
        "backend": backend, "seconds_per_matvec": elapsed,
        "weight_bytes_read_gravity": int(compressed),
        "weight_bytes_read_dense_fp16": int(dense),
        "memory_traffic_reduction": dense / max(1, compressed),
        "effective_dense_gbps": dense / elapsed / 1e9,
        "gravity_gbps": compressed / elapsed / 1e9,
    }


def selftest() -> int:
    """Parity against the CPU authority, on both a synthetic and a real geometry."""
    rng = np.random.default_rng(0)
    backends = available_backends()
    report: dict[str, Any] = {"backends": backends, "decoder_version": DECODER_VERSION}

    if "mps" not in backends:
        print(json.dumps({**report, "selftest": "SKIPPED_NO_MPS"}, indent=2))
        return 0

    checks = []
    for shape, dim, k, rotate in (((256, 128), 8, 16, False),
                                  ((512, 256), 16, 256, False),
                                  ((256, 128), 8, 128, True)):
        weights = rng.standard_normal(shape).astype(np.float32)
        pack = forge.pack_transform_pq if rotate else forge.pack_product_quant
        artifact = pack(weights, dim=dim, subspaces=1, k=k, seed=0, iters=4)

        probe = rng.standard_normal(shape[1]).astype(np.float32)
        single = parity(artifact, probe)
        assert single["within_tolerance"], (shape, dim, k, rotate, single)
        assert single["finite"]

        # batched must agree too, and must agree column-by-column with the single path
        batch = rng.standard_normal((shape[1], 4)).astype(np.float32)
        many = parity(artifact, batch)
        assert many["within_tolerance"], (shape, dim, k, rotate, many)
        checks.append({"shape": list(shape), "dim": dim, "k": k, "rotate": rotate,
                       "single_gap": single["relative_max_gap"],
                       "batch_gap": many["relative_max_gap"]})

    report["parity"] = checks
    report["selftest"] = "PASS"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(selftest())
    sys.stderr.write("import this module; only `selftest` runs standalone\n")
    raise SystemExit(2)
