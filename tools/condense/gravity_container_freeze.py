#!/usr/bin/env python3
"""Freeze hawking.gravity.container.v1: schema, test vectors, compatibility.

The spec is prose and prose drifts.  These three artifacts are the machine-checkable half
of the freeze: a JSON Schema a reader can validate a header against, a set of test vectors
generated from the real writer so a future reader can prove it still reads v1, and a
compatibility record naming exactly what v1 promises.

    freeze     regenerate all three from the live implementation
    check      re-derive them and fail if anything drifted
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_format  # noqa: E402

REPO = HERE.parent.parent
OUT = REPO / "docs/gravity"

CONTAINER_ABI = "hawking.gravity.container.v1"

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": f"https://hawking.local/{CONTAINER_ABI}/header.schema.json",
    "title": "hawking.gravity.shard_header.v1",
    "type": "object",
    "required": ["schema", "format_version", "model", "compression", "integrity", "tensors"],
    "additionalProperties": True,
    "properties": {
        "schema": {"const": "hawking.gravity.shard_header.v1"},
        "format_version": {"const": 1},
        "model": {
            "type": "object",
            "required": ["repo", "revision"],
            "properties": {
                "repo": {"type": "string", "minLength": 1},
                "revision": {"type": "string", "minLength": 1},
                "source_shard": {"type": "string"},
            },
        },
        "architecture": {"type": "object"},
        "tokenizer": {"type": "object"},
        "shard": {"type": "object"},
        "compression": {
            "type": "object",
            "required": ["codec", "packed_bpw"],
            "properties": {
                "codec": {"type": "string", "minLength": 1},
                # Both rates are physical claims and verify() reconciles them against the
                # body.  complete_bpw is the campaign's headline and the only rate a
                # candidate may be judged on; it is optional in the schema solely so a
                # single-codec shard with no native organs stays valid.
                "packed_bpw": {"type": "number", "minimum": 0},
                "complete_bpw": {"type": "number", "minimum": 0},
                "production_rung": {"type": "string"},
            },
        },
        "integrity": {
            "type": "object",
            "required": ["body_sha256", "tensor_count"],
            "properties": {
                "body_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "tensor_count": {"type": "integer", "minimum": 0},
            },
        },
        "tensors": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "elements", "offset", "bytes", "sha256"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "category": {"type": "string"},
                    "layer": {"type": ["integer", "null"]},
                    "expert": {"type": ["integer", "null"]},
                    "shape": {"type": "array", "items": {"type": "integer"}},
                    "elements": {"type": "integer", "minimum": 1},
                    "codec": {"type": "string", "minLength": 1},
                    "terminal_state": {
                        "enum": ["PACKED_IN_CORE_ARTIFACT", "PROTECTED_SOURCE_NATIVE"]},
                    "bpw": {"type": "number", "minimum": 0},
                    "offset": {"type": "integer", "minimum": 0},
                    # The load-bearing constraint: a descriptor with no payload is the
                    # Generation A defect and is malformed by schema, not by convention.
                    "bytes": {"type": "integer", "minimum": 1},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
            },
        },
    },
}

COMPATIBILITY = {
    "schema": "hawking.gravity.container_compatibility.v1",
    "abi": CONTAINER_ABI,
    "extension": ".gravity",
    "status": "FROZEN",
    "format_version": gravity_format.FORMAT_VERSION,
    "prefix": {
        "bytes": gravity_format.PREFIX_BYTES,
        "struct": gravity_format.PREFIX_STRUCT,
        "magic": gravity_format.MAGIC.decode("latin-1"),
    },
    "promises": [
        "the 20-byte prefix and its field order",
        "the header is UTF-8 JSON at a declared length",
        "every tensor is locatable by name without reading the body",
        "every tensor carries a non-empty payload and its own digest",
        "packed_bpw and complete_bpw both reconcile against physical bytes",
        "native.<dtype> payloads are exact source bytes and round-trip bit-identically",
    ],
    "does_not_promise": [
        "any particular codec, rung, or geometry",
        "that a given representation id will still be produced",
        "tensor ordering beyond the declared offsets",
        "header fields beyond those named in the schema",
    ],
    "representation_ids": {
        "glm52.pq.r0.v1": "product quantization, the Generation A control family",
        "glm52.functional.block.v1": "native functional block student",
        "glm52.indexshare.student.v1": "IndexShare-aware attention student",
        "glm52.hybrid.doctor.v1": "native base plus a serialized Doctor correction",
        "native.bf16": "exact source bytes, bfloat16",
        "native.f32": "exact source bytes, float32",
    },
    "representation_ids_are_versioned_separately": True,
    "breaking_change_policy": (
        "a change to any promise requires a new format_version and a new entry here; "
        "v1 readers must keep working against v1 files forever"),
}


def build_vectors() -> dict:
    """Generate test vectors from the live writer, so they cannot drift from it silently."""
    import tempfile

    compressed = [
        ({"name": "model.layers.0.mlp.experts.0.gate_proj.weight", "category": "routed_expert",
          "layer": 0, "expert": 0, "shape": [8, 16], "elements": 128,
          "codec": "glm52.pq.r0.v1", "terminal_state": "PACKED_IN_CORE_ARTIFACT",
          "bpw": 0.75}, bytes(range(12))),
        ({"name": "model.layers.0.mlp.experts.1.gate_proj.weight", "category": "routed_expert",
          "layer": 0, "expert": 1, "shape": [8, 16], "elements": 128,
          "codec": "glm52.pq.r0.v1", "terminal_state": "PACKED_IN_CORE_ARTIFACT",
          "bpw": 0.75}, bytes(range(12, 24))),
    ]
    native = [
        ({"name": "model.layers.0.mlp.gate.weight", "category": "router", "layer": 0,
          "expert": None, "shape": [4, 8], "elements": 32, "codec": "native.bf16",
          "terminal_state": "PROTECTED_SOURCE_NATIVE", "bpw": 16.0},
         bytes(range(64))),
        ({"name": "model.layers.0.mlp.gate.e_score_correction_bias",
          "category": "router_control", "layer": 0, "expert": None, "shape": [4],
          "elements": 4, "codec": "native.f32",
          "terminal_state": "PROTECTED_SOURCE_NATIVE", "bpw": 32.0},
         bytes(range(16))),
    ]
    payloads = compressed + native

    packed_bits = sum(len(b) for _, b in compressed) * 8
    packed_elements = sum(d["elements"] for d, _ in compressed)
    all_bits = sum(len(b) for _, b in payloads) * 8
    all_elements = sum(d["elements"] for d, _ in payloads)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model-00001-of-00001.gravity"
        header = gravity_format.write_shard(
            path, payloads,
            model={"repo": "hawking.test/container-v1", "revision": "0" * 40,
                   "source_shard": "model-00001-of-00001.safetensors"},
            architecture={"type": "GlmMoeDsaForCausalLM", "hidden_layers": 78,
                          "routed_experts": 256, "shared_experts": 1, "hidden_size": 6144},
            tokenizer={"kind": "reference", "source": "hawking.test/container-v1"},
            compression={"codec": "gravity-pq", "production_rung": "R0",
                         "packed_bpw": packed_bits / packed_elements,
                         "complete_bpw": all_bits / all_elements},
            shard={"source": "model-00001-of-00001.safetensors", "of": 1})
        raw = path.read_bytes()
        report = gravity_format.verify(path)

    return {
        "schema": "hawking.gravity.container_test_vectors.v1",
        "abi": CONTAINER_ABI,
        "purpose": ("a reader that reproduces these bytes and these verify results reads "
                    "v1 correctly; regenerate only when format_version changes"),
        "vector": {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "file_bytes": len(raw),
            "prefix_hex": raw[:gravity_format.PREFIX_BYTES].hex(),
            "header": header,
            "body_hex": raw[gravity_format._body_offset(header):].hex(),
            "expected_verify": report,
            "expected_tensor_reads": {
                descriptor["name"]: blob.hex() for descriptor, blob in payloads},
        },
        "invariants_exercised": [
            "compressed and native tensors coexist in one shard",
            "complete_bpw exceeds packed_bpw when native organs are carried",
            "every descriptor carries a non-empty payload",
            "each tensor is readable by name at its declared offset",
        ],
    }


def freeze() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "GRAVITY_CONTAINER_SCHEMA.json").write_text(
        json.dumps(SCHEMA, indent=2, sort_keys=True) + "\n")
    (OUT / "GRAVITY_CONTAINER_COMPATIBILITY.json").write_text(
        json.dumps(COMPATIBILITY, indent=2, sort_keys=True) + "\n")
    (OUT / "GRAVITY_CONTAINER_TEST_VECTORS.json").write_text(
        json.dumps(build_vectors(), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"abi": CONTAINER_ABI, "status": "FROZEN",
                      "wrote": sorted(p.name for p in OUT.glob("GRAVITY_CONTAINER_*"))},
                     indent=2))
    return 0


def check() -> int:
    """Re-derive the vectors and confirm the frozen file still describes this writer."""
    frozen_path = OUT / "GRAVITY_CONTAINER_TEST_VECTORS.json"
    if not frozen_path.exists():
        sys.stderr.write("container is not frozen yet; run freeze\n")
        return 2
    frozen = json.loads(frozen_path.read_text())
    live = build_vectors()
    drifted = [key for key in ("file_sha256", "prefix_hex", "body_hex")
               if frozen["vector"][key] != live["vector"][key]]
    if frozen["vector"]["header"] != live["vector"]["header"]:
        drifted.append("header")
    print(json.dumps({"abi": CONTAINER_ABI,
                      "status": "STABLE" if not drifted else "DRIFTED",
                      "drifted_fields": drifted}, indent=2))
    return 0 if not drifted else 1


def selftest() -> int:
    """The frozen vector must validate against the frozen schema and against verify()."""
    import tempfile

    vectors = build_vectors()
    header = vectors["vector"]["header"]

    # Hand-rolled rather than pulling in jsonschema: only the constraints that carry the
    # freeze are checked, and every one of them is a rule Generation A broke.
    assert header["schema"] == "hawking.gravity.shard_header.v1"
    assert header["format_version"] == 1
    for tensor in header["tensors"]:
        assert int(tensor["bytes"]) >= 1, tensor["name"]
        assert len(tensor["sha256"]) == 64
        assert tensor["terminal_state"] in (
            "PACKED_IN_CORE_ARTIFACT", "PROTECTED_SOURCE_NATIVE")
        assert int(tensor["elements"]) >= 1

    report = vectors["vector"]["expected_verify"]
    assert report["ok"], report
    assert report["observed_complete_bpw"] > report["observed_packed_bpw"], report
    assert report["tensors_without_payload"] == []

    # Rebuilding from the recorded bytes must reproduce the recorded reads.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vector.gravity"
        path.write_bytes(bytes.fromhex(vectors["vector"]["prefix_hex"])
                         + json.dumps(header, sort_keys=True,
                                      separators=(",", ":")).encode("utf-8")
                         + bytes.fromhex(vectors["vector"]["body_hex"]))
        for name, expected in vectors["vector"]["expected_tensor_reads"].items():
            assert gravity_format.read_tensor(path, name).hex() == expected, name
        assert gravity_format.verify(path)["ok"]

    print("gravity_container_freeze selftest OK")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "freeze"
    raise SystemExit({"freeze": freeze, "check": check, "selftest": selftest}[command]())
