#!/usr/bin/env python3.12
"""Production execution adapter for ``.gravity`` R0 tensors: bind, refuse, then execute.

The kernel in :mod:`gravity_metal` has zero callers and the worker's
``PRODUCTION_EXECUTION_ADAPTER_REGISTRY`` is empty, so nothing in this repository can
execute a packed tensor under review.  This module is the missing middle: it opens a
sealed shard read-only, carries the header's own claims about a tensor all the way to the
call site, and refuses -- by name -- every case where those claims and the bytes disagree
or where the chosen backend says it cannot run the geometry.

What it deliberately is NOT: a kernel.  The execution backend is injected as a declared
mapping (``BACKEND_PROTOCOL``), so this adapter is constructible and unit-testable with no
GPU and with no kernel selected.  When the kernel phase lands, wiring is one registration
line and one backend mapping -- no change here.

Registry admission
------------------
``glm52_worker._inspect_registered_adapter_source`` AST-inspects a registered adapter's
source and rejects import-time executable effects.  That is why this file has no
``class``, no decorators, no default arguments, no ``__main__`` block, and no computed
module-level assignment: every top-level node is a docstring, an import, a literal
assignment, or a plain ``def``.  Breaking that rule makes the file permanently
unregisterable, so it is a hard constraint, not a style preference.  Because there is no
``sys.path`` bootstrap here, the caller must already have ``tools/condense`` importable
(the worker does exactly that before importing its own modules).

The one line a later, reviewed change adds to
``glm52_worker.PRODUCTION_EXECUTION_ADAPTER_REGISTRY`` -- and not before the evidence
listed in ``reports/condense/breakthrough/GLM52_PRODUCTION_EXECUTION_ADAPTER.json``
exists::

    "gravity_pq_r0": {
        "interface_version": "hawking.glm52.execution_adapter.v1",
        "roles": ["RUN_WINDOW_FORWARD"],
        "entry_points": {"RUN_WINDOW_FORWARD": "execute_tensor"},
        "source_path": "tools/condense/gravity_execution_adapter.py",
        "source_sha256": "<sha256 of this file at review time>",
    },

That line alone does not open the readiness gate: ``_validate_execution_adapters`` still
fails with ``execution adapters lack roles`` until the other seven roles have reviewed
adapters of their own.  This adapter claims exactly one role and claims it honestly.

Honest scope
------------
One tensor, one matvec.  There is no router here and no MoE dispatch: routing weights are
held at source precision as ``CONTROL_SENSITIVE_CANDIDATE`` and carry no PQ payload, so
this adapter refuses them rather than pretending.  Shared and routed experts differ only
in the descriptor's ``category``/``expert`` fields, which are carried through and bound,
not interpreted.  :func:`declare_capabilities` states all of this in machine-readable form
so a receipt can never claim more than the code does.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

import glm52_pack
import gravity_forge
import gravity_format
from glm52_common import Glm52Error

ADAPTER_ID = "gravity_pq_r0"
ADAPTER_INTERFACE = "hawking.glm52.execution_adapter.v1"
ADAPTER_ROLE = "RUN_WINDOW_FORWARD"
BACKEND_PROTOCOL = "hawking.gravity.execution_backend.v1"

# Format/representation the adapter is reviewed for.  Anything outside this is refused by
# name; nothing here is negotiable at runtime.
SUPPORTED_FORMAT_VERSION = 1
REQUIRED_RUNG = "R0"
REQUIRED_SUBSPACES = 1
REQUIRED_ROTATE = False
REQUIRED_CODEC = "gravity-pq"
# Categories whose tensors are a plain [rows, cols] matvec at R0.  Router, router_control,
# embeddings and lm_head are excluded on purpose: the control path is protected at source
# precision and has no packed payload to execute.
EXECUTABLE_CATEGORIES = ("attention", "indexer", "routed_expert", "shared_expert")
# The header's per-tensor bpw is a float rounded through JSON; this is the reconciliation
# tolerance between it and bytes*8/elements, not a quality tolerance.
BPW_TOLERANCE = 1e-6

REFUSAL_NOT_A_SHARD = "GRAVITY_NOT_A_SHARD"
REFUSAL_FORMAT_VERSION = "GRAVITY_FORMAT_VERSION_NEWER_THAN_READER"
REFUSAL_CODEC = "GRAVITY_CODEC_NOT_REVIEWED"
REFUSAL_RUNG = "GRAVITY_RUNG_IS_NOT_PRODUCTION_R0"
REFUSAL_CATEGORY = "GRAVITY_CATEGORY_NOT_EXECUTABLE_BY_THIS_ADAPTER"
REFUSAL_ROTATE = "GRAVITY_ROTATED_GEOMETRY_NOT_REVIEWED"
REFUSAL_SUBSPACES = "GRAVITY_SUBSPACES_IS_NOT_ONE"
REFUSAL_GEOMETRY = "GRAVITY_GEOMETRY_DISAGREES_WITH_DESCRIPTOR"
REFUSAL_RATE = "GRAVITY_RATE_DISAGREES_WITH_DESCRIPTOR"
REFUSAL_BACKEND_PROTOCOL = "GRAVITY_BACKEND_PROTOCOL_MISMATCH"
REFUSAL_BACKEND_GEOMETRY = "GRAVITY_BACKEND_CANNOT_EXECUTE_GEOMETRY"
REFUSAL_INPUT_SHAPE = "GRAVITY_INPUT_LENGTH_DISAGREES_WITH_GEOMETRY"
REFUSAL_OUTPUT_SHAPE = "GRAVITY_BACKEND_RETURNED_WRONG_SHAPE"
REFUSAL_BUDGET = "GRAVITY_ENTRY_EXCEEDS_WHOLE_BYTE_BUDGET"

# Eviction contract, enforced by _admit()/_evict_to_budget() and reported by
# resource_report():
#   1. Cache identity is an explicit string built from the shard path, the tensor name and
#      the descriptor's own sha256.  Never id(): CPython reuses addresses after GC, so an
#      id()-keyed cache can serve one tensor's indices under another tensor's name.
#   2. Entries are evicted least-recently-used first until resident bytes fit the declared
#      budget.  Eviction is always safe: every entry is reconstructible from the shard,
#      which is opened read-only and never mutated.
#   3. Before an entry is dropped, the backend's optional "release" callable is invoked
#      with the same key, so device-side buffers keyed identically are freed in the same
#      step rather than pinning memory for the life of the process.
#   4. An entry that cannot fit the whole budget by itself is refused at admission instead
#      of being retained in violation of the budget.
EVICTION_CONTRACT = "LRU_TO_DECLARED_BYTE_BUDGET_WITH_BACKEND_RELEASE"


def _refuse(reason: str, detail: str) -> None:
    """Every refusal names itself; callers and receipts match on the leading code."""
    raise Glm52Error(f"{reason}: {detail}")


def declare_capabilities() -> dict[str, Any]:
    """The mandate 5.1 binding table: what this adapter binds, and what it refuses to."""
    return {
        "adapter_id": ADAPTER_ID,
        "interface_version": ADAPTER_INTERFACE,
        "roles": [ADAPTER_ROLE],
        "entry_points": {ADAPTER_ROLE: "execute_tensor"},
        "backend_protocol": BACKEND_PROTOCOL,
        "bindings": {
            "gravity_format_version": SUPPORTED_FORMAT_VERSION,
            "tensor_identity": [
                "name", "sha256", "shape", "elements", "category", "expert", "layer",
            ],
            "geometry": ["D", "S", "sub", "k", "rows", "cols", "nchunk", "index_bits"],
            "rate_bpw": "descriptor bpw reconciled against payload_bytes*8/elements",
            "decoder": "glm52_pack.deserialize (bit-exact inverse of pack_indices)",
            "metal_grammar": "UNSELECTED_IN_THIS_PHASE",
            "router": "NOT_EXECUTED: control tensors are protected at source precision",
            "moe_execution": "PER_TENSOR_ONLY: category/expert carried, not dispatched",
            "device_policy": "declared by the injected backend; this module is CPU-only",
            "resource_policy": ["byte_budget", "explicit_cache_key", EVICTION_CONTRACT],
        },
        "executable_categories": list(EXECUTABLE_CATEGORIES),
        "refusals": [
            REFUSAL_NOT_A_SHARD, REFUSAL_FORMAT_VERSION, REFUSAL_CODEC, REFUSAL_RUNG,
            REFUSAL_CATEGORY, REFUSAL_ROTATE, REFUSAL_SUBSPACES, REFUSAL_GEOMETRY,
            REFUSAL_RATE, REFUSAL_BACKEND_PROTOCOL, REFUSAL_BACKEND_GEOMETRY,
            REFUSAL_INPUT_SHAPE, REFUSAL_OUTPUT_SHAPE, REFUSAL_BUDGET,
        ],
    }


def _shard_format_version(path: Path) -> int:
    """Read the 20-byte fixed prefix only, so the version refusal is ours and named.

    gravity_format.read_header raises a generic GravityFormatError for a newer version;
    the worker needs the reason code, so the check happens here first.
    """
    with open(path, "rb") as handle:
        prefix = handle.read(gravity_format.PREFIX_BYTES)
    if len(prefix) != gravity_format.PREFIX_BYTES:
        _refuse(REFUSAL_NOT_A_SHARD, f"{path.name} is shorter than a header prefix")
    magic, version, _ = struct.unpack(gravity_format.PREFIX_STRUCT, prefix)
    if magic != gravity_format.MAGIC:
        _refuse(REFUSAL_NOT_A_SHARD, f"{path.name} does not carry the GRAVITY magic")
    if int(version) > SUPPORTED_FORMAT_VERSION:
        _refuse(
            REFUSAL_FORMAT_VERSION,
            f"{path.name} is format v{int(version)}, this adapter reads "
            f"v{SUPPORTED_FORMAT_VERSION}",
        )
    return int(version)


def _check_backend(backend: Mapping[str, Any]) -> None:
    """A backend is a mapping, not a class: one protocol string and two callables."""
    if not isinstance(backend, Mapping):
        _refuse(REFUSAL_BACKEND_PROTOCOL, "backend is not a mapping")
    if backend.get("protocol") != BACKEND_PROTOCOL:
        _refuse(
            REFUSAL_BACKEND_PROTOCOL,
            f"backend declares protocol {backend.get('protocol')!r}, "
            f"required {BACKEND_PROTOCOL!r}",
        )
    if not isinstance(backend.get("name"), str) or not backend["name"]:
        _refuse(REFUSAL_BACKEND_PROTOCOL, "backend declares no name")
    for field in ("can_execute", "matvec"):
        if not callable(backend.get(field)):
            _refuse(REFUSAL_BACKEND_PROTOCOL, f"backend has no callable {field!r}")
    release = backend.get("release")
    if release is not None and not callable(release):
        _refuse(REFUSAL_BACKEND_PROTOCOL, "backend 'release' is present but not callable")


def open_adapter(
    shard_path: Path,
    backend: Mapping[str, Any],
    byte_budget: int,
) -> dict[str, Any]:
    """Bind one sealed shard, read-only, to one backend under one byte budget.

    No argument has a default: the byte budget and the backend are policy, and policy that
    defaults is policy nobody reviewed.
    """
    path = Path(shard_path)
    _check_backend(backend)
    if not isinstance(byte_budget, int) or byte_budget <= 0:
        _refuse(REFUSAL_BUDGET, f"byte_budget must be a positive int, got {byte_budget!r}")
    version = _shard_format_version(path)
    header, body_offset = gravity_format.open_shard(path)
    codec = header.get("compression", {}).get("codec")
    if codec != REQUIRED_CODEC:
        _refuse(REFUSAL_CODEC, f"{path.name} carries codec {codec!r}, reviewed {REQUIRED_CODEC!r}")
    return {
        "shard_path": str(path),
        "format_version": version,
        "body_offset": int(body_offset),
        "descriptors": {t["name"]: t for t in header["tensors"]},
        "shard_header": {k: v for k, v in header.items() if k != "tensors"},
        "backend": backend,
        "byte_budget": int(byte_budget),
        "cache": {},
        "resident_bytes": 0,
        "evictions": 0,
        "loads": 0,
    }


def cache_key(adapter: Mapping[str, Any], name: str) -> str:
    """Explicit content identity: shard, tensor name, and the descriptor's own digest."""
    descriptor = _descriptor(adapter, name)
    return f"{adapter['shard_path']}|{name}|{descriptor['sha256']}"


def _descriptor(adapter: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    descriptor = adapter["descriptors"].get(name)
    if descriptor is None:
        raise gravity_format.GravityFormatError(
            f"{Path(adapter['shard_path']).name}: no tensor named {name!r}")
    return descriptor


def _check_descriptor(descriptor: Mapping[str, Any]) -> None:
    """Refuse on the header's own claims before a single payload byte is read."""
    name = descriptor["name"]
    if descriptor.get("rung") != REQUIRED_RUNG:
        _refuse(
            REFUSAL_RUNG,
            f"{name} is rung {descriptor.get('rung')!r}, reviewed {REQUIRED_RUNG!r}",
        )
    if descriptor.get("codec") != REQUIRED_CODEC:
        _refuse(REFUSAL_CODEC, f"{name} carries codec {descriptor.get('codec')!r}")
    if descriptor.get("category") not in EXECUTABLE_CATEGORIES:
        _refuse(
            REFUSAL_CATEGORY,
            f"{name} is category {descriptor.get('category')!r}; executable categories "
            f"are {list(EXECUTABLE_CATEGORIES)} (control tensors are held at source "
            "precision and carry no packed payload)",
        )


def _geometry(descriptor: Mapping[str, Any], codes: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-check what the header claims against what the payload actually decodes to."""
    name = descriptor["name"]
    if int(codes["S"]) != REQUIRED_SUBSPACES:
        _refuse(REFUSAL_SUBSPACES, f"{name} decodes S={int(codes['S'])}, reviewed S=1")
    if bool(codes["rotate"]):
        _refuse(REFUSAL_ROTATE, f"{name} decodes rotate=True, reviewed rotate=False")
    rows, cols = int(codes["rows"]), int(codes["cols"])
    shape = [int(v) for v in descriptor["shape"]]
    if shape != [rows, cols]:
        _refuse(
            REFUSAL_GEOMETRY,
            f"{name} descriptor shape {shape} but payload decodes [{rows}, {cols}]",
        )
    if rows * cols != int(descriptor["elements"]):
        _refuse(
            REFUSAL_GEOMETRY,
            f"{name} descriptor claims {int(descriptor['elements'])} elements but "
            f"{rows}*{cols}={rows * cols} decode",
        )
    D, nchunk = int(codes["D"]), int(codes["nchunk"])
    if nchunk * D != cols:
        _refuse(
            REFUSAL_GEOMETRY,
            f"{name} decodes nchunk={nchunk} * D={D} = {nchunk * D}, cols={cols}",
        )
    if int(codes["indices"].shape[0]) != rows * nchunk:
        _refuse(
            REFUSAL_GEOMETRY,
            f"{name} decodes {int(codes['indices'].shape[0])} index rows, "
            f"rows*nchunk={rows * nchunk}",
        )
    observed_bpw = int(descriptor["bytes"]) * 8 / max(1, int(descriptor["elements"]))
    if abs(observed_bpw - float(descriptor["bpw"])) > BPW_TOLERANCE:
        _refuse(
            REFUSAL_RATE,
            f"{name} claims {float(descriptor['bpw'])} bpw but its "
            f"{int(descriptor['bytes'])} bytes over {int(descriptor['elements'])} "
            f"elements are {observed_bpw}",
        )
    book = codes["codebooks"][0]
    return {
        "D": D, "S": int(codes["S"]), "sub": int(codes["sub"]), "k": int(book.shape[0]),
        "rows": rows, "cols": cols, "nchunk": nchunk, "rotate": bool(codes["rotate"]),
        "index_bits": glm52_pack.index_bits(int(book.shape[0])),
        "category": descriptor.get("category"), "expert": descriptor.get("expert"),
        "layer": descriptor.get("layer"), "rung": descriptor.get("rung"),
        "bpw": float(descriptor["bpw"]), "observed_bpw": observed_bpw,
    }


def _entry_bytes(codes: Mapping[str, Any]) -> int:
    return int(codes["indices"].nbytes) + sum(int(cb.nbytes) for cb in codes["codebooks"])


def _evict_to_budget(adapter: dict[str, Any]) -> int:
    """LRU until resident bytes fit the declared budget; release the backend's copy too."""
    release = adapter["backend"].get("release")
    cache = adapter["cache"]
    evicted = 0
    while cache and adapter["resident_bytes"] > adapter["byte_budget"]:
        key = next(iter(cache))
        entry = cache.pop(key)
        adapter["resident_bytes"] -= int(entry["bytes"])
        if release is not None:
            release(key)
        evicted += 1
    adapter["evictions"] += evicted
    return evicted


def evict_all(adapter: dict[str, Any]) -> int:
    """Drop every cached tensor and tell the backend to drop its side of each key."""
    saved = adapter["byte_budget"]
    adapter["byte_budget"] = 0
    evicted = _evict_to_budget(adapter)
    adapter["byte_budget"] = saved
    return evicted


def load_tensor(adapter: dict[str, Any], name: str) -> dict[str, Any]:
    """Read one tensor read-only, decode its geometry, and bind it to its descriptor.

    Returns the whole bound record -- descriptor, geometry, cache key, decoded codes -- so
    a caller that wants to inspect what will execute never has to guess.
    """
    key = cache_key(adapter, name)
    cached = adapter["cache"].get(key)
    if cached is not None:
        adapter["cache"][key] = adapter["cache"].pop(key)
        return cached
    descriptor = _descriptor(adapter, name)
    _check_descriptor(descriptor)
    blob = gravity_format.read_tensor(Path(adapter["shard_path"]), name, verify_hash=True)
    codes = glm52_pack.deserialize(blob)
    geometry = _geometry(descriptor, codes)
    entry = {
        "cache_key": key, "descriptor": dict(descriptor), "geometry": geometry,
        "codes": codes, "bytes": _entry_bytes(codes), "payload_bytes": len(blob),
    }
    if entry["bytes"] > adapter["byte_budget"]:
        _refuse(
            REFUSAL_BUDGET,
            f"{name} needs {entry['bytes']} resident bytes, whole budget is "
            f"{adapter['byte_budget']}",
        )
    adapter["cache"][key] = entry
    adapter["resident_bytes"] += entry["bytes"]
    adapter["loads"] += 1
    _evict_to_budget(adapter)
    return entry


def execute_tensor(adapter: dict[str, Any], name: str, x: np.ndarray) -> np.ndarray:
    """Registry entry point for RUN_WINDOW_FORWARD: y = W_gravity[name] @ x.

    Every refusal is raised before the backend is touched, except the backend's own
    geometry veto, which is asked for explicitly rather than discovered by crashing.
    """
    entry = load_tensor(adapter, name)
    geometry = entry["geometry"]
    vector = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if vector.ndim != 1 or int(vector.shape[0]) != geometry["cols"]:
        _refuse(
            REFUSAL_INPUT_SHAPE,
            f"{name} expects a length-{geometry['cols']} vector, got shape "
            f"{list(vector.shape)} -- a longer x is what silently overruns an unchecked "
            "device buffer",
        )
    backend = adapter["backend"]
    veto = backend["can_execute"](geometry)
    if veto is not None:
        _refuse(REFUSAL_BACKEND_GEOMETRY, f"{name}: backend {backend['name']} says {veto}")
    y = backend["matvec"](entry["codes"], vector, entry["cache_key"])
    result = np.asarray(y, dtype=np.float32)
    if result.shape != (geometry["rows"],):
        _refuse(
            REFUSAL_OUTPUT_SHAPE,
            f"{name}: backend {backend['name']} returned {list(result.shape)}, "
            f"expected [{geometry['rows']}]",
        )
    return result


def resource_report(adapter: Mapping[str, Any]) -> dict[str, Any]:
    """What the policy promised and what it is actually holding, by explicit key."""
    return {
        "byte_budget": int(adapter["byte_budget"]),
        "resident_bytes": int(adapter["resident_bytes"]),
        "entries": len(adapter["cache"]),
        "loads": int(adapter["loads"]),
        "evictions": int(adapter["evictions"]),
        "eviction_contract": EVICTION_CONTRACT,
        "cache_identity": "explicit shard|name|sha256 string, never id()",
        "backend": adapter["backend"]["name"],
        "keys": list(adapter["cache"]),
    }


def cpu_reference_backend() -> dict[str, Any]:
    """The CPU authority every other backend is graded against: gravity_forge.pq_execute.

    Declared through the same protocol as any kernel, so the injected-backend path is the
    only path -- there is no privileged built-in.
    """
    def can_execute(geometry: Mapping[str, Any]) -> str | None:
        if int(geometry["S"]) != 1:
            return f"CPU reference is wired for S=1, got S={int(geometry['S'])}"
        return None

    def matvec(codes: Mapping[str, Any], x: np.ndarray, key: str) -> np.ndarray:
        del key  # the reference path holds no device state, so it evicts nothing
        artifact = gravity_forge.PackedArtifact(
            "product_quant", np.empty((0,), dtype=np.float32),
            int(codes["rows"]) * int(codes["cols"]), gravity_forge.ByteLedger(), 0, 0,
            {"pq_codes": codes},
        )
        return gravity_forge.pq_execute(artifact, x)

    return {
        "name": "cpu_reference_pq_execute",
        "protocol": BACKEND_PROTOCOL,
        "device": "cpu",
        "can_execute": can_execute,
        "matvec": matvec,
    }
