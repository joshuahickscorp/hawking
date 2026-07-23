#!/usr/bin/env python3.12
"""The ``.gravity`` model format: a native container for Gravity-compressed models.

Roadmap Phase 3.  Gravity models are not GGUF with a compression layer bolted on, so they
do not ship as GGUF.  A ``.gravity`` shard is self-describing: given the file alone, a
runtime can enumerate every tensor, learn the architecture and tokenizer it belongs to,
verify integrity, and decode any single tensor by seeking straight to it -- without the
original repository, without an index sidecar, and without reading the whole file.

Layout::

    magic            8 bytes   b"GRAVITY\\x00"
    format_version   u32 LE
    header_length    u64 LE
    header           UTF-8 JSON, header_length bytes
    body             concatenated tensor payloads, each at its declared offset

Everything variable lives in the JSON header; the binary prefix is fixed at 20 bytes so a
reader can locate the header without parsing anything.  Tensor payloads are opaque here --
they are whatever the codec emitted (today ``glm52_pack``'s PQ blob) -- because the
container's job is to describe and locate them, not to reinterpret them.

Integrity is two-level: ``body_sha256`` covers all payload bytes so a truncated or
corrupted shard is caught on open, and every tensor additionally carries its own sha256 so
a single bad tensor can be identified rather than condemning the whole shard.

Design rule that keeps this honest: the header records the MEASURED bits per weight for
every tensor and for the shard, taken from the codec's own ledger.  A ``.gravity`` file
that claims a rate its bytes do not support is malformed, and :func:`verify` says so.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = b"GRAVITY\x00"
FORMAT_VERSION = 1
PREFIX_STRUCT = "<8sIQ"
PREFIX_BYTES = struct.calcsize(PREFIX_STRUCT)
HEADER_SCHEMA = "hawking.gravity.shard_header.v1"


class GravityFormatError(Exception):
    """A .gravity shard is malformed, truncated, or misdescribes its own contents."""


def build_header(*, model: dict, tensors: list[dict], body_sha256: str,
                 compression: dict, tokenizer: dict | None = None,
                 architecture: dict | None = None, shard: dict | None = None) -> dict:
    return {
        "schema": HEADER_SCHEMA,
        "format_version": FORMAT_VERSION,
        "model": model,
        "architecture": architecture or {},
        "tokenizer": tokenizer or {},
        "compression": compression,
        "shard": shard or {},
        "integrity": {"body_sha256": body_sha256, "tensor_count": len(tensors)},
        "tensors": tensors,
    }


def write_shard(path: Path, payloads: list[tuple[dict, bytes]], *, model: dict,
                compression: dict, tokenizer: dict | None = None,
                architecture: dict | None = None, shard: dict | None = None) -> dict:
    """Write one .gravity shard from (descriptor, payload) pairs.

    Offsets are assigned here rather than trusted from callers, so a descriptor can never
    disagree with where its bytes actually landed.
    """
    digest = hashlib.sha256()
    tensors: list[dict] = []
    offset = 0
    for descriptor, blob in payloads:
        digest.update(blob)
        entry = dict(descriptor)
        entry["offset"] = offset
        entry["bytes"] = len(blob)
        entry["sha256"] = hashlib.sha256(blob).hexdigest()
        tensors.append(entry)
        offset += len(blob)

    header = build_header(model=model, tensors=tensors, body_sha256=digest.hexdigest(),
                          compression=compression, tokenizer=tokenizer,
                          architecture=architecture, shard=shard)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as sink:
        sink.write(struct.pack(PREFIX_STRUCT, MAGIC, FORMAT_VERSION, len(encoded)))
        sink.write(encoded)
        for _, blob in payloads:
            sink.write(blob)
    tmp.replace(path)
    return header


def read_header(path: Path) -> dict:
    """Read only the header.  Cheap enough to enumerate a whole model's shards."""
    with open(path, "rb") as handle:
        prefix = handle.read(PREFIX_BYTES)
        if len(prefix) != PREFIX_BYTES:
            raise GravityFormatError(f"{path.name}: shorter than a header prefix")
        magic, version, header_length = struct.unpack(PREFIX_STRUCT, prefix)
        if magic != MAGIC:
            raise GravityFormatError(f"{path.name}: not a .gravity shard")
        if version > FORMAT_VERSION:
            raise GravityFormatError(
                f"{path.name}: format version {version} is newer than this reader ({FORMAT_VERSION})")
        raw = handle.read(header_length)
        if len(raw) != header_length:
            raise GravityFormatError(f"{path.name}: header truncated")
    return json.loads(raw)


def _body_offset(header: dict) -> int:
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return PREFIX_BYTES + len(encoded)


def open_shard(path: Path) -> tuple[dict, int]:
    """Return the header and the absolute offset where tensor payloads begin."""
    with open(path, "rb") as handle:
        prefix = handle.read(PREFIX_BYTES)
        magic, version, header_length = struct.unpack(PREFIX_STRUCT, prefix)
        if magic != MAGIC:
            raise GravityFormatError(f"{path.name}: not a .gravity shard")
    return read_header(path), PREFIX_BYTES + header_length


def read_tensor(path: Path, name: str, *, verify_hash: bool = True) -> bytes:
    """Seek straight to one tensor's payload without reading the rest of the shard."""
    header, base = open_shard(path)
    entry = next((t for t in header["tensors"] if t["name"] == name), None)
    if entry is None:
        raise GravityFormatError(f"{path.name}: no tensor named {name!r}")
    with open(path, "rb") as handle:
        handle.seek(base + int(entry["offset"]))
        blob = handle.read(int(entry["bytes"]))
    if len(blob) != int(entry["bytes"]):
        raise GravityFormatError(f"{path.name}: tensor {name!r} truncated")
    if verify_hash and hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise GravityFormatError(f"{path.name}: tensor {name!r} failed its integrity check")
    return blob


def iter_tensors(path: Path) -> Any:
    """Stream every (descriptor, payload) pair in stored order, one at a time."""
    header, base = open_shard(path)
    with open(path, "rb") as handle:
        handle.seek(base)
        for entry in sorted(header["tensors"], key=lambda t: int(t["offset"])):
            yield entry, handle.read(int(entry["bytes"]))


def verify(path: Path) -> dict:
    """Full integrity check: body digest, per-tensor digests, and rate self-consistency."""
    header, base = open_shard(path)
    digest = hashlib.sha256()
    bad: list[str] = []
    total_bytes = 0
    total_weights = 0
    with open(path, "rb") as handle:
        handle.seek(base)
        for entry in sorted(header["tensors"], key=lambda t: int(t["offset"])):
            blob = handle.read(int(entry["bytes"]))
            if len(blob) != int(entry["bytes"]):
                raise GravityFormatError(f"{path.name}: body truncated at {entry['name']}")
            digest.update(blob)
            if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
                bad.append(entry["name"])
            total_bytes += len(blob)
            total_weights += int(entry.get("elements", 0))

    body_ok = digest.hexdigest() == header["integrity"]["body_sha256"]
    # A shard may not claim a rate its own bytes do not support.  Protected tensors carry
    # no payload here, so they are excluded from the packed-rate reconciliation.
    packed = [t for t in header["tensors"] if int(t.get("bytes", 0)) > 0
              and t.get("elements")]
    observed = (sum(int(t["bytes"]) for t in packed) * 8
                / max(1, sum(int(t["elements"]) for t in packed))) if packed else 0.0
    claimed = float(header["compression"].get("packed_bpw", observed))
    rate_ok = abs(observed - claimed) < 1e-6

    return {
        "path": str(path), "format_version": header["format_version"],
        "tensors": len(header["tensors"]), "body_bytes": total_bytes,
        "body_sha256_ok": body_ok, "bad_tensors": bad,
        "observed_packed_bpw": observed, "claimed_packed_bpw": claimed,
        "rate_self_consistent": rate_ok,
        "ok": body_ok and not bad and rate_ok,
    }


def selftest() -> int:
    """Round-trip, seek, integrity, and tamper-detection on a synthetic shard."""
    import tempfile

    payloads = []
    for index in range(4):
        blob = bytes([(index * 37 + byte) % 256 for byte in range(128 + index)])
        payloads.append(({"name": f"model.layers.{index}.weight", "elements": 1024,
                          "shape": [32, 32], "category": "routed_expert",
                          "codec": "pq", "bpw": len(blob) * 8 / 1024}, blob))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model-00001-of-00001.gravity"
        packed_bpw = (sum(len(b) for _, b in payloads) * 8
                      / sum(d["elements"] for d, _ in payloads))
        write_shard(path, payloads,
                    model={"repo": "zai-org/GLM-5.2", "revision": "b" * 40},
                    architecture={"type": "GlmMoeDsaForCausalLM", "hidden_layers": 78},
                    tokenizer={"kind": "reference", "source": "zai-org/GLM-5.2"},
                    compression={"codec": "gravity-pq", "packed_bpw": packed_bpw},
                    shard={"index": 1, "count": 1})

        header = read_header(path)
        assert header["schema"] == HEADER_SCHEMA
        assert header["model"]["repo"] == "zai-org/GLM-5.2"
        assert header["architecture"]["hidden_layers"] == 78
        assert len(header["tensors"]) == 4

        # every tensor is reachable by name, and the bytes come back identical
        for descriptor, blob in payloads:
            assert read_tensor(path, descriptor["name"]) == blob, descriptor["name"]

        # streaming order matches stored order and covers everything
        streamed = [(d["name"], b) for d, b in iter_tensors(path)]
        assert [n for n, _ in streamed] == [d["name"] for d, _ in payloads]
        assert [b for _, b in streamed] == [b for _, b in payloads]

        report = verify(path)
        assert report["ok"], report
        assert report["rate_self_consistent"], report

        # a single flipped payload byte must be caught, and named
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 0xFF
        tampered = Path(tmp) / "tampered.gravity"
        tampered.write_bytes(bytes(raw))
        damaged = verify(tampered)
        assert not damaged["ok"], "tampering went undetected"
        assert damaged["bad_tensors"] == [payloads[-1][0]["name"]], damaged["bad_tensors"]

        # a shard claiming a rate its bytes do not support is malformed
        lying = Path(tmp) / "lying.gravity"
        write_shard(lying, payloads, model={"repo": "x", "revision": "y"},
                    compression={"codec": "gravity-pq", "packed_bpw": 0.001})
        assert not verify(lying)["rate_self_consistent"], "false rate claim went undetected"

    print(json.dumps({"selftest": "PASS", "format": "gravity", "version": FORMAT_VERSION,
                      "prefix_bytes": PREFIX_BYTES, "seek_by_name": True,
                      "integrity_two_level": True, "false_rate_claim_rejected": True},
                     indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "verify":
        print(json.dumps(verify(Path(sys.argv[2])), indent=2))
        raise SystemExit(0)
    if len(sys.argv) > 2 and sys.argv[1] == "header":
        print(json.dumps(read_header(Path(sys.argv[2])), indent=2)[:4000])
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(selftest())
    sys.stderr.write("usage: gravity_format.py [selftest|header PATH|verify PATH]\n")
    raise SystemExit(2)
