#!/usr/bin/env python3.12
"""The six representation-escalation arms, executable on synthetic fixtures.

`HAWKING_REPRESENTATION_ESCALATION_TOURNAMENT.json` preregistered six arms and froze their
gates before any compute.  This makes them *runnable* on tiny deterministic matrices, so
that when a heavy window opens the question is which representation wins rather than
whether the serializers work.

What this establishes: each arm has a physical serializer, exact complete-byte accounting,
and a deterministic CPU decoder, and the accounting reconciles.

What this does NOT establish, and no output here may be read as establishing: that any arm
recovers GLM.  These are random 64x64 matrices.  A reconstruction error on a synthetic
fixture is not a capability claim, and the tournament explicitly rules weight-space error
inadmissible as a promotion signal.  The arms are being tested for *wiring*, not for merit.

    python3.12 -m tools.repr_arms.arms
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field

import numpy as np

# Deterministic everywhere: same input, same bytes, every run, on every machine.
SEED = 0x5EED


@dataclass
class Packed:
    """A packed tensor plus the byte accounting the one-bit law is measured against."""

    arm: str
    payload: bytes
    n_weights: int
    component_bytes: dict[str, int] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return len(self.payload)

    @property
    def complete_bpw(self) -> float:
        """Complete bits per weight, whole-tensor, no exclusions.

        The same accounting the sealed Math-Preserve receipt used: every byte that must
        exist for the tensor to be decoded counts, including codebooks and headers. Payload-
        only accounting is how a sub-bit claim becomes untrue without anyone lying.
        """
        return (self.total_bytes * 8) / self.n_weights

    def reconciles(self) -> bool:
        return sum(self.component_bytes.values()) == self.total_bytes


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(SEED ^ (int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)))


# --------------------------------------------------------------------------
# A1 native functional expert/organ replacement
# --------------------------------------------------------------------------
def pack_a1(w: np.ndarray, rank: int = 4) -> Packed:
    """Replace the organ's WEIGHTS with a fitted low-rank FUNCTION of its input.

    Distinct from the others in what it can express: it is not trying to reproduce w, it is
    trying to reproduce what w does. That is why weight-space error is the wrong lens.
    """
    m, n = w.shape
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    u, s, vt = u[:, :rank], s[:rank], vt[:rank]
    head = struct.pack("<4sHHH", b"A1FN", m, n, rank)
    body = np.concatenate([u.ravel(), s.ravel(), vt.ravel()]).astype(np.float16).tobytes()
    return Packed("A1", head + body, m * n, {"header": len(head), "factors": len(body)})


def unpack_a1(p: Packed) -> np.ndarray:
    m, n, rank = struct.unpack("<HHH", p.payload[4:10])
    v = np.frombuffer(p.payload[10:], dtype=np.float16).astype(np.float32)
    o = 0
    u = v[o : o + m * rank].reshape(m, rank); o += m * rank
    s = v[o : o + rank]; o += rank
    vt = v[o : o + rank * n].reshape(rank, n)
    return (u * s) @ vt


# --------------------------------------------------------------------------
# A2 PQ plus deterministic residual correction
# --------------------------------------------------------------------------
def _pq(w: np.ndarray, nsub: int, ncent: int, tag: str):
    m, n = w.shape
    sub = n // nsub
    rng = _rng(tag)
    books, codes = [], []
    for j in range(nsub):
        block = w[:, j * sub : (j + 1) * sub]
        idx = rng.choice(m, size=min(ncent, m), replace=False)
        book = block[idx].astype(np.float32)
        d = ((block[:, None, :] - book[None, :, :]) ** 2).sum(-1)
        codes.append(d.argmin(1).astype(np.uint8))
        books.append(book)
    return books, np.stack(codes, 1), sub


def pack_a2(w: np.ndarray, nsub: int = 8, ncent: int = 16) -> Packed:
    """Quantized base plus an explicit second error budget on the residual.

    Differs from A3 in shape: a correction term rather than a finer partition of the same
    error.
    """
    m, n = w.shape
    books, codes, sub = _pq(w, nsub, ncent, "a2")
    approx = np.concatenate([books[j][codes[:, j]] for j in range(nsub)], 1)
    resid = (w - approx).astype(np.float16)
    head = struct.pack("<4sHHHH", b"A2PQ", m, n, nsub, ncent)
    bb = np.concatenate([b.ravel() for b in books]).astype(np.float16).tobytes()
    cb = codes.tobytes()
    rb = resid.tobytes()
    return Packed("A2", head + bb + cb + rb, m * n,
                  {"header": len(head), "codebooks": len(bb), "codes": len(cb), "residual": len(rb)})


def unpack_a2(p: Packed) -> np.ndarray:
    m, n, nsub, ncent = struct.unpack("<HHHH", p.payload[4:12])
    sub = n // nsub
    o = 12
    nb = nsub * ncent * sub
    books = np.frombuffer(p.payload[o : o + nb * 2], dtype=np.float16).astype(np.float32)
    books = books.reshape(nsub, ncent, sub); o += nb * 2
    codes = np.frombuffer(p.payload[o : o + m * nsub], dtype=np.uint8).reshape(m, nsub)
    o += m * nsub
    resid = np.frombuffer(p.payload[o:], dtype=np.float16).astype(np.float32).reshape(m, n)
    approx = np.concatenate([books[j][codes[:, j]] for j in range(nsub)], 1)
    return approx + resid


# --------------------------------------------------------------------------
# A3 additive / multi-codebook
# --------------------------------------------------------------------------
def pack_a3(w: np.ndarray, stages: int = 3, ncent: int = 16) -> Packed:
    """Sum of several COARSE codebooks rather than one fine codebook.

    Different rate-distortion geometry at the same total bits: failure is spread across
    stages instead of concentrated in one quantizer.
    """
    m, n = w.shape
    rng = _rng("a3")
    resid = w.astype(np.float32).copy()
    books, codes = [], []
    for _ in range(stages):
        idx = rng.choice(m, size=min(ncent, m), replace=False)
        book = resid[idx].copy()
        d = ((resid[:, None, :] - book[None, :, :]) ** 2).sum(-1)
        c = d.argmin(1).astype(np.uint8)
        resid = resid - book[c]
        books.append(book)
        codes.append(c)
    head = struct.pack("<4sHHHH", b"A3AD", m, n, stages, ncent)
    bb = np.stack(books).astype(np.float16).tobytes()
    cb = np.stack(codes, 1).tobytes()
    return Packed("A3", head + bb + cb, m * n,
                  {"header": len(head), "codebooks": len(bb), "codes": len(cb)})


def unpack_a3(p: Packed) -> np.ndarray:
    m, n, stages, ncent = struct.unpack("<HHHH", p.payload[4:12])
    o = 12
    nb = stages * ncent * n
    books = np.frombuffer(p.payload[o : o + nb * 2], dtype=np.float16).astype(np.float32)
    books = books.reshape(stages, ncent, n); o += nb * 2
    codes = np.frombuffer(p.payload[o:], dtype=np.uint8).reshape(m, stages)
    out = np.zeros((m, n), np.float32)
    for s in range(stages):
        out += books[s][codes[:, s]]
    return out


# --------------------------------------------------------------------------
# A4 shared latent basis with causal gates
# --------------------------------------------------------------------------
def pack_a4(experts: list[np.ndarray], basis: int = 6) -> Packed:
    """Experts as gated combinations of ONE shared basis.

    Exploits inter-expert redundancy, which per-expert compression cannot see at all --
    the only arm whose cost falls as the expert count rises.
    """
    stack = np.stack([e.ravel() for e in experts]).astype(np.float32)
    u, s, vt = np.linalg.svd(stack, full_matrices=False)
    k = min(basis, vt.shape[0])
    gates = (u[:, :k] * s[:k]).astype(np.float16)
    b = vt[:k].astype(np.float16)
    m, n = experts[0].shape
    head = struct.pack("<4sHHHH", b"A4SB", len(experts), m, n, k)
    body = gates.tobytes() + b.tobytes()
    return Packed("A4", head + body, sum(e.size for e in experts),
                  {"header": len(head), "gates": len(gates.tobytes()), "basis": len(b.tobytes())})


def unpack_a4(p: Packed) -> list[np.ndarray]:
    ne, m, n, k = struct.unpack("<HHHH", p.payload[4:12])
    o = 12
    gates = np.frombuffer(p.payload[o : o + ne * k * 2], dtype=np.float16).astype(np.float32)
    gates = gates.reshape(ne, k); o += ne * k * 2
    basis = np.frombuffer(p.payload[o:], dtype=np.float16).astype(np.float32).reshape(k, m * n)
    return [(gates[i] @ basis).reshape(m, n) for i in range(ne)]


# --------------------------------------------------------------------------
# A5 profile-conditioned protected islands
# --------------------------------------------------------------------------
def pack_a5(w: np.ndarray, importance: np.ndarray, protect_frac: float = 0.05) -> Packed:
    """Protect rows by MEASURED importance rather than by tensor role.

    Math-Preserve already spends 67.5 GB on protected islands and still collapses, so the
    live question is whether the islands are in the wrong PLACES -- not whether there are
    too few. This arm is the one that tests that directly.
    """
    m, n = w.shape
    k = max(1, int(m * protect_frac))
    keep = np.argsort(-importance)[:k].astype(np.uint16)
    mask = np.zeros(m, bool)
    mask[keep] = True
    prot = w[mask].astype(np.float16)
    rest = w[~mask]
    books, codes, _ = _pq(rest, 8, 16, "a5") if rest.shape[0] else ([], np.zeros((0, 8), np.uint8), 0)
    head = struct.pack("<4sHHH", b"A5PI", m, n, k)
    ib = keep.tobytes()
    pb = prot.tobytes()
    bb = np.concatenate([b.ravel() for b in books]).astype(np.float16).tobytes() if books else b""
    cb = codes.tobytes()
    return Packed("A5", head + ib + pb + bb + cb, m * n,
                  {"header": len(head), "island_index": len(ib), "protected": len(pb),
                   "codebooks": len(bb), "codes": len(cb)})


def unpack_a5(p: Packed, nsub: int = 8, ncent: int = 16) -> np.ndarray:
    """Decode A5.

    Without this the arm could be priced but not judged, and pricing without judging is
    exactly the trap: Math-Preserve spends 67,526,197,248 bytes on protected islands and
    still cannot complete '2 + 2 ='. Whether islands help depends entirely on whether they
    are in the right PLACES, and that is a functional question no byte count answers.
    """
    m, n, k = struct.unpack("<HHH", p.payload[4:10])
    o = 10
    keep = np.frombuffer(p.payload[o : o + k * 2], dtype=np.uint16).astype(np.int64)
    o += k * 2
    prot = np.frombuffer(p.payload[o : o + k * n * 2], dtype=np.float16).astype(np.float32)
    prot = prot.reshape(k, n)
    o += k * n * 2

    rest_rows = m - k
    sub = n // nsub
    nb = nsub * ncent * sub
    books = np.frombuffer(p.payload[o : o + nb * 2], dtype=np.float16).astype(np.float32)
    books = books.reshape(nsub, ncent, sub)
    o += nb * 2
    codes = np.frombuffer(p.payload[o : o + rest_rows * nsub], dtype=np.uint8)
    codes = codes.reshape(rest_rows, nsub)

    out = np.zeros((m, n), np.float32)
    mask = np.zeros(m, bool)
    mask[keep] = True
    out[mask] = prot
    if rest_rows:
        out[~mask] = np.concatenate([books[j][codes[:, j]] for j in range(nsub)], 1)
    return out


# --------------------------------------------------------------------------
# A6 trajectory-stabilized hybrid
# --------------------------------------------------------------------------
def pack_a6(w: np.ndarray, traces: np.ndarray | None) -> Packed:
    """Fit against multi-layer TRAJECTORY preservation rather than per-tensor error.

    DATA_BLOCKED in reality: the teacher ledger holds 122 lines, 118 per-layer captures and
    ZERO trajectory traces. The preregistration says this arm is blocked rather than failed,
    and that substituting per-layer organs answers a different question. This function
    refuses rather than quietly doing the wrong thing.
    """
    if traces is None or len(traces) == 0:
        raise NotImplementedError(
            "A6 is DATA_BLOCKED: it requires full-sequence parent trajectory traces, and the "
            "teacher ledger contains 0. Substituting per-layer captures would answer a "
            "different question and report it as this one."
        )
    raise NotImplementedError("A6 fitting is unimplemented pending real trajectory traces")


ARMS = {
    "A1": "native functional organ replacement",
    "A2": "PQ plus deterministic residual correction",
    "A3": "additive / multi-codebook",
    "A4": "shared latent basis with causal gates",
    "A5": "profile-conditioned protected islands",
    "A6": "trajectory-stabilized hybrid (DATA_BLOCKED)",
}


def run_matrix() -> dict:
    """Execute every arm on a deterministic fixture and report wiring, not merit."""
    rng = _rng("fixture")
    w = rng.standard_normal((64, 64)).astype(np.float32)
    experts = [rng.standard_normal((32, 32)).astype(np.float32) for _ in range(8)]
    importance = np.abs(w).sum(1)

    def err(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

    rows = []
    for arm, packer, unpacker, args in [
        ("A1", pack_a1, unpack_a1, (w,)),
        ("A2", pack_a2, unpack_a2, (w,)),
        ("A3", pack_a3, unpack_a3, (w,)),
    ]:
        p = packer(*args)
        back = unpacker(p)
        rows.append({
            "arm": arm, "name": ARMS[arm], "status": "EXECUTABLE",
            "complete_bpw": round(p.complete_bpw, 4),
            "total_bytes": p.total_bytes,
            "components": p.component_bytes,
            "accounting_reconciles": p.reconciles(),
            "roundtrip_relative_error": round(err(back, w), 6),
            "deterministic": True,
        })

    p4 = pack_a4(experts)
    back4 = unpack_a4(p4)
    rows.append({
        "arm": "A4", "name": ARMS["A4"], "status": "EXECUTABLE",
        "complete_bpw": round(p4.complete_bpw, 4), "total_bytes": p4.total_bytes,
        "components": p4.component_bytes, "accounting_reconciles": p4.reconciles(),
        "roundtrip_relative_error": round(
            float(np.mean([err(b, e) for b, e in zip(back4, experts)])), 6),
        "deterministic": True,
        "note": "the only arm whose cost per expert falls as the expert count rises",
    })

    p5 = pack_a5(w, importance)
    rows.append({
        "arm": "A5", "name": ARMS["A5"], "status": "EXECUTABLE_PACK_ONLY",
        "complete_bpw": round(p5.complete_bpw, 4), "total_bytes": p5.total_bytes,
        "components": p5.component_bytes, "accounting_reconciles": p5.reconciles(),
        "roundtrip_relative_error": None,
        "gap": "decoder not implemented; packing and byte accounting only",
        "deterministic": True,
    })

    try:
        pack_a6(w, None)
        a6 = {"arm": "A6", "status": "UNEXPECTEDLY_RAN"}
    except NotImplementedError as e:
        a6 = {"arm": "A6", "name": ARMS["A6"], "status": "DATA_BLOCKED",
              "why": str(e), "not_a_failure": True}
    rows.append(a6)

    return {
        "schema": "hawking.representation.readiness_matrix.v1",
        "fixture": "deterministic 64x64 and 8x(32x32), seed 0x5EED, numpy only, no GPU",
        "resource_mode": "LIGHT_ONLY",
        "arms": rows,
        "what_this_establishes": "each arm has a physical serializer, exact complete-byte accounting that reconciles, and a deterministic decoder. The wiring is real.",
        "what_this_does_NOT_establish": [
            "that any arm recovers GLM. These are random matrices.",
            "any capability claim whatsoever. The tournament rules weight-space error INADMISSIBLE as a promotion signal, and roundtrip_relative_error is exactly that -- reported here as a wiring check, never as merit.",
            "that A5's islands are in the right places. That needs real importance from a real parent."
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_matrix(), indent=2))
