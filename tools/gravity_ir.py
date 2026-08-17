#!/usr/bin/env python3
"""Gravity IR — a physical model program, not a tensor-codec table.

The existing recipe vocabulary is `tensor: codec`. It cannot express anything
this campaign now wants: structure shared across sites, blocks generated rather
than stored, additive correction stages, exact islands. Promising mechanisms had
nowhere to land even when they measured well.

Design rules, each one a consequence of a failure already paid for:

1. EVERY node reports its own stored bytes. Complete BPW is computed FROM the
   program, so a mechanism cannot claim a density it does not have. A pack once
   reported 3.6138 while carrying an uncounted 1.814 GB leftover.

2. Shared objects are CONTENT-ADDRESSED and counted ONCE no matter how many
   sites reference them. This is the only way sharing can pay: the whole point of
   a dictionary across 64 layers is that it is stored once. Double-counting it
   would make sharing look worthless; forgetting it entirely would make sharing
   look free. Neither is honest.

3. The BPW denominator is the ORIGINAL source parameter count, never the
   candidate's own degrees of freedom. A structural representation with fewer
   variables must still normalise against the source or the number is not
   comparable.

4. Every node names the kernel that consumes it. A representation with no
   execution path is not a representation, it is a compression demo.

Verification is the round-trip: BPW computed from the program must equal BPW
measured from bytes on disk. If it does not, the IR is not describing the artifact.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

SOURCE_PARAM_COUNT = 26_895_998_464


# ---------------------------------------------------------------- shared pool

class SharedPool:
    """Content-addressed store. Identical bytes are stored once and cost once."""

    def __init__(self):
        self.objects: Dict[str, dict] = {}

    def put(self, kind: str, nbytes: int, content_id: Optional[str] = None, **meta) -> str:
        cid = content_id or hashlib.sha256(
            json.dumps({"kind": kind, "nbytes": nbytes, **meta}, sort_keys=True).encode()
        ).hexdigest()[:16]
        if cid in self.objects:
            assert self.objects[cid]["nbytes"] == nbytes, f"content id {cid} collides on size"
        else:
            self.objects[cid] = {"kind": kind, "nbytes": nbytes, **meta}
        return cid

    def bytes_total(self) -> int:
        return sum(o["nbytes"] for o in self.objects.values())


# ---------------------------------------------------------------- nodes

@dataclass
class Node:
    """One term of a site's representation. Terms at a site are additive."""
    kind: str
    kernel: str
    stored_bytes: int = 0              # bytes owned exclusively by this site
    shared_refs: List[str] = field(default_factory=list)   # content ids, cost is pool-side
    active_bytes: Optional[int] = None  # bytes touched per token; defaults to stored
    elements: int = 0                  # source elements this term helps represent
    meta: dict = field(default_factory=dict)

    def active(self) -> int:
        return self.stored_bytes if self.active_bytes is None else self.active_bytes


# node constructors — each is a representation family, each names its kernel

def quant_tensor(elements, bits, group, kernel, scale_bytes_per_group=2, header=40):
    groups = max(1, elements // group)
    code_bytes = (elements * bits + 7) // 8
    return Node("QuantTensor", kernel,
                stored_bytes=code_bytes + groups * scale_bytes_per_group + header,
                elements=elements, meta={"bits": bits, "group": group})


def dense_tensor(elements, dtype_bytes, kernel, header=40):
    return Node("DenseTensor", kernel, stored_bytes=elements * dtype_bytes + header,
                elements=elements, meta={"dtype_bytes": dtype_bytes})


def shared_basis(elements, coeff_bits, basis_cid, kernel, header=40):
    """Per-site coefficients against a basis stored once in the pool."""
    return Node("SharedBasis", kernel,
                stored_bytes=(elements * coeff_bits + 7) // 8 + header,
                shared_refs=[basis_cid], elements=elements,
                meta={"coeff_bits": coeff_bits})


def sparse_correction(n_exceptions, value_bytes, index_bits, kernel, header=40):
    """Exact values on a small set. Index cost is counted -- it usually dominates."""
    return Node("SparseCorrection", kernel,
                stored_bytes=n_exceptions * value_bytes + (n_exceptions * index_bits + 7) // 8 + header,
                elements=0, meta={"n": n_exceptions, "index_bits": index_bits})


def exact_island(n_elements, value_bytes, kernel, index_bits=0, header=40):
    """A compile-time-known region kept exact. index_bits=0 when the set is static."""
    return Node("ExactIsland", kernel,
                stored_bytes=n_elements * value_bytes + (n_elements * index_bits + 7) // 8 + header,
                elements=0, meta={"n": n_elements, "index_bits": index_bits})


def generated_block(elements, code_bytes, generator_cid, kernel, decode_flops_per_elem=0.0):
    """A block computed from a tiny code plus a shared generator."""
    return Node("GeneratedBlock", kernel, stored_bytes=code_bytes,
                shared_refs=[generator_cid], elements=elements,
                active_bytes=code_bytes,
                meta={"decode_flops_per_elem": decode_flops_per_elem})


# ---------------------------------------------------------------- program

@dataclass
class Site:
    name: str
    elements: int
    terms: List[Node]


class Program:
    def __init__(self, name, source_pin=None):
        self.name = name
        self.source_pin = source_pin
        self.pool = SharedPool()
        self.sites: List[Site] = []

    def add(self, name, elements, terms):
        self.sites.append(Site(name, elements, terms))
        return self

    # ---- cost

    def site_bytes(self) -> int:
        return sum(t.stored_bytes for s in self.sites for t in s.terms)

    def total_bytes(self) -> int:
        """Site-exclusive bytes plus every referenced shared object, counted once."""
        used = {cid for s in self.sites for t in s.terms for cid in t.shared_refs}
        shared = sum(self.pool.objects[c]["nbytes"] for c in used)
        return self.site_bytes() + shared

    def complete_bpw(self) -> float:
        return 8 * self.total_bytes() / SOURCE_PARAM_COUNT

    def active_bytes_per_token(self) -> int:
        return sum(t.active() for s in self.sites for t in s.terms)

    def covered_elements(self) -> int:
        return sum(s.elements for s in self.sites)

    def kernels(self):
        return sorted({t.kernel for s in self.sites for t in s.terms})

    def report(self):
        return {
            "program": self.name,
            "source_pin": self.source_pin,
            "sites": len(self.sites),
            "covered_elements": self.covered_elements(),
            "site_bytes": self.site_bytes(),
            "shared_bytes": self.total_bytes() - self.site_bytes(),
            "total_bytes": self.total_bytes(),
            "complete_bpw": self.complete_bpw(),
            "active_bytes_per_token": self.active_bytes_per_token(),
            "kernels": self.kernels(),
            "shared_objects": self.pool.objects,
        }

    def to_json(self):
        return json.dumps({
            "schema": "hawking.gravity1.program.v1",
            "name": self.name,
            "source_pin": self.source_pin,
            "shared_pool": self.pool.objects,
            "sites": [{"name": s.name, "elements": s.elements,
                       "terms": [asdict(t) for t in s.terms]} for s in self.sites],
            "cost": {k: v for k, v in self.report().items() if k != "shared_objects"},
        }, indent=2)
