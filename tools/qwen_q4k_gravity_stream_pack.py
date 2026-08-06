#!/usr/bin/env python3
"""Bounded-memory source-preserving GGUF -> Gravity packer for Qwen2/Llama.

Unlike the legacy packer, this writer never collects tensor payloads or the
whole body in RAM.  It makes exactly one output body copy, computes each
descriptor hash while reading the memory-mapped GGUF, and atomically promotes
the completed Gravity shard.  It is deliberately a format/runtime gate, not a
compression claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Iterator

from gguf import GGUFReader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.llama_q4k_gravity_pack import (  # noqa: E402
    SCHEMA,
    architecture,
    codec_and_geometry,
    mapped_name,
    runtime_shape,
)

MAGIC = b"GRAVITY\0"
FORMAT_VERSION = 1
CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def byte_chunks(array: Any) -> Iterator[memoryview]:
    """Yield contiguous read-only byte views without materializing a tensor."""
    view = array.reshape(-1).view("u1")
    for start in range(0, int(view.nbytes), CHUNK_BYTES):
        yield memoryview(view[start : start + CHUNK_BYTES])


def prepare(
    source: Path, tokenizer_json: Path | None
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Any]]]:
    reader = GGUFReader(str(source), mode="r")
    source_hash = sha256_file(source)
    arch = architecture(reader)
    body_hash = hashlib.sha256()
    entries: list[tuple[dict[str, Any], Any]] = []
    offset = 0
    compressed_bytes = compressed_elements = complete_bytes = complete_elements = 0
    skipped: list[str] = []
    for tensor in reader.tensors:
        target = mapped_name(tensor.name)
        if target is None:
            skipped.append(tensor.name)
            continue
        shape = runtime_shape(tensor)
        codec, _ = codec_and_geometry(tensor, shape)
        tensor_hash = hashlib.sha256()
        for block in byte_chunks(tensor.data):
            tensor_hash.update(block)
            body_hash.update(block)
        nbytes = int(tensor.data.nbytes)
        elements = 1
        for dim in shape:
            elements *= dim
        entries.append(({
            "name": target,
            "shape": shape,
            "codec": codec,
            "elements": elements,
            "bpw": nbytes * 8 / max(1, elements),
            "terminal_state": "SOURCE_QUANT_BYTES_COPIED",
            "source_tensor": tensor.name,
            "offset": offset,
            "bytes": nbytes,
            "sha256": tensor_hash.hexdigest(),
        }, tensor.data))
        offset += nbytes
        complete_bytes += nbytes
        complete_elements += elements
        if not codec.startswith("native."):
            compressed_bytes += nbytes
            compressed_elements += elements
    if not entries:
        raise ValueError("no supported executable tensors")
    names = [descriptor["name"] for descriptor, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("duplicate canonical tensor name")
    tokenizer = {"kind": "embedded-gguf", "source_sha256": source_hash}
    if tokenizer_json is not None:
        if not tokenizer_json.is_file():
            raise FileNotFoundError(f"tokenizer sidecar is not a file: {tokenizer_json}")
        tokenizer = {
            "kind": "tokenizer-json",
            "dir": str(tokenizer_json.parent.resolve()),
            "source": tokenizer_json.name,
            "sha256": sha256_file(tokenizer_json),
        }
    header = {
        "schema": "hawking.gravity.shard_header.v1",
        "format_version": FORMAT_VERSION,
        "model": {"family": arch["model_type"], "source_gguf_sha256": source_hash},
        "architecture": arch,
        "tokenizer": tokenizer,
        "compression": {
            "codec": "ggml-source-q4_k-q5_k-q5_0-q6_k-q8_0",
            "source_quantization_preserved": True,
            "packed_bpw": compressed_bytes * 8 / max(1, compressed_elements),
            "complete_bpw": complete_bytes * 8 / max(1, complete_elements),
            "lossy_fit": False,
        },
        "shard": {"writer": "qwen_q4k_gravity_stream_pack", "stream_chunk_bytes": CHUNK_BYTES},
        "gguf_metadata": {"skipped_source_tensors": skipped},
        "integrity": {"body_sha256": body_hash.hexdigest(), "tensor_count": len(entries)},
        "tensors": [descriptor for descriptor, _ in entries],
    }
    return header, entries


def write(source: Path, output: Path, tokenizer_json: Path | None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to replace existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    header, entries = prepare(source, tokenizer_json)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"remove or inspect prior temporary artifact first: {temporary}")
    try:
        with temporary.open("xb") as destination:
            destination.write(MAGIC)
            destination.write(struct.pack("<I", FORMAT_VERSION))
            destination.write(struct.pack("<Q", len(encoded)))
            destination.write(encoded)
            for _, payload in entries:
                for block in byte_chunks(payload):
                    destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "schema": SCHEMA,
        "status": "WRITTEN_SOURCE_PRESERVING_STREAMED",
        "source": str(source.resolve()),
        "source_sha256": header["model"]["source_gguf_sha256"],
        "output": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "tensor_count": header["integrity"]["tensor_count"],
        "body_sha256": header["integrity"]["body_sha256"],
        "lossy_fit": False,
        "runtime": "GravityLlama direct raw-quant grammar; no capability or performance promotion implied",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    args = parser.parse_args()
    print(json.dumps(write(
        args.source.resolve(),
        args.output.resolve(),
        args.tokenizer_json.resolve() if args.tokenizer_json else None,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
