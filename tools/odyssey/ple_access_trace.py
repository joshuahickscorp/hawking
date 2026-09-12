#!/usr/bin/env python3
"""Source-bound access tracing for Qwen4-Exp PLE n-gram lookup tables.

Qwen4-Exp's PLE table is not 128 independent semantic embedding tables.  The
checkpoint shards concatenate along the global embedding-row axis, while each
accepted token deterministically hashes its current and preceding token IDs to
one row per n-gram head.  This utility binds that address contract to:

* the exact pinned specimen configuration and checkpoint index;
* a versioned reference implementation source file; and
* an optional accepted-token receipt.

It range-reads and hashes only the rows selected by the trace.  It deliberately
does *not* execute PLE projections, use hidden states, construct a compact
artifact, report complete EBPW, or claim capability.  Its job is to make the
next selective-repair discriminator use real lookup addresses rather than
invented "hot" rows.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "hawking.odyssey.ple_access_trace.v1"
SEQUENCE_SCHEMA = "hawking.flash.stateful_complete_token_session.v1"
DEFAULT_SPEC = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
DEFAULT_SEQUENCE_RECEIPT = Path(
    "receipts/headless/FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.json"
)
DEFAULT_REFERENCE = Path(
    os.environ.get(
        "HAWKING_QWEN4_EXP_REFERENCE",
        "/Users/scammermike/.local/share/uv/tools/mlx-vlm/lib/python3.13/"
        "site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py",
    )
)
DEFAULT_SEED = 1234
PRIME_1 = 10007
MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(int(count)):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


@dataclass(frozen=True)
class PLEContract:
    """The small deterministic address contract for one PLE layer."""

    vocab_size: int
    eos_token_id: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_vocab_divisible_by: int
    ple_embed_dim: int
    ple_layer_index: int
    source_layer_index: int
    seed: int
    multipliers: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]
    head_offsets: tuple[int, ...]
    total_vocab_size: int
    padded_vocab_size: int

    @property
    def context_len(self) -> int:
        return self.ngram_size - 1

    @property
    def ngram_heads(self) -> int:
        return self.context_len * self.heads_per_ngram

    @property
    def head_dim(self) -> int:
        return self.ple_embed_dim // self.ngram_heads


def build_contract(
    *,
    vocab_size: int,
    eos_token_id: int,
    ngram_size: int,
    heads_per_ngram: int,
    ngram_vocab_size_base: int,
    make_vocab_divisible_by: int,
    ple_embed_dim: int,
    ple_layer_index: int,
    source_layer_index: int,
    seed: int,
) -> PLEContract:
    """Port the reference's pure PLE address construction with explicit checks."""
    if vocab_size <= 0 or ngram_size < 2 or heads_per_ngram <= 0:
        raise ValueError("invalid PLE vocabulary, n-gram, or head geometry")
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    if ple_embed_dim <= 0 or ple_embed_dim % ngram_heads:
        raise ValueError("ple_embed_dim must divide exactly into the PLE heads")
    if ngram_vocab_size_base < 2 or make_vocab_divisible_by <= 0:
        raise ValueError("invalid PLE lookup vocabulary geometry")
    max_long = (1 << 63) - 1
    multiplier_max = max_long // vocab_size
    half_bound = max(1, multiplier_max // 2)
    base_seed = int(seed) + PRIME_1 * int(ple_layer_index)
    multipliers = tuple(
        2
        * (
            _splitmix64((base_seed + SPLITMIX_GAMMA * (index + 1)) & MASK64)
            % half_bound
        )
        + 1
        for index in range(ngram_size)
    )
    sizes: list[int] = []
    offsets: list[int] = []
    total = 0
    for head_index in range(ngram_heads):
        global_head = ple_layer_index * ngram_heads + head_index
        size = _find_nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    padded = math.ceil(total / make_vocab_divisible_by) * make_vocab_divisible_by
    return PLEContract(
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_vocab_size_base=ngram_vocab_size_base,
        make_vocab_divisible_by=make_vocab_divisible_by,
        ple_embed_dim=ple_embed_dim,
        ple_layer_index=ple_layer_index,
        source_layer_index=source_layer_index,
        seed=seed,
        multipliers=multipliers,
        head_vocab_sizes=tuple(sizes),
        head_offsets=tuple(offsets),
        total_vocab_size=total,
        padded_vocab_size=padded,
    )


def load_contract(spec: Path, ple_layer_index: int) -> tuple[PLEContract, dict[str, Any]]:
    """Bind the deterministic PLE address formula to an exact source config."""
    config_path = spec / "config.json"
    document = json.loads(config_path.read_text(encoding="utf-8"))
    raw = document.get("text_config")
    if not isinstance(raw, dict):
        raise ValueError("Flash source config has no text_config")
    ids = raw.get("ple_layer_ids")
    if not isinstance(ids, list) or not ids or not all(_is_int(item) and item > 0 for item in ids):
        raise ValueError("Flash source config has no valid ple_layer_ids")
    if ple_layer_index < 0 or ple_layer_index >= len(ids):
        raise ValueError(f"ple layer index {ple_layer_index} is outside {ids}")
    required = (
        "vocab_size",
        "ngram_size",
        "heads_per_ngram",
        "ngram_vocab_size_base",
        "make_ngram_vocab_size_divisible_by",
        "ple_embed_dim",
    )
    if any(not _is_int(raw.get(name)) for name in required):
        raise ValueError("Flash source config is missing required integer PLE geometry")
    eos = raw.get("eos_token_id")
    if isinstance(eos, list):
        eos = eos[0] if eos else None
    if not _is_int(eos):
        raise ValueError("Flash source config has no scalar eos_token_id")
    seed = raw.get("seed")
    if seed is None:
        seed = DEFAULT_SEED
    if not _is_int(seed):
        raise ValueError("Flash source config has an invalid PLE seed")
    contract = build_contract(
        vocab_size=int(raw["vocab_size"]),
        eos_token_id=int(eos),
        ngram_size=int(raw["ngram_size"]),
        heads_per_ngram=int(raw["heads_per_ngram"]),
        ngram_vocab_size_base=int(raw["ngram_vocab_size_base"]),
        make_vocab_divisible_by=int(raw["make_ngram_vocab_size_divisible_by"]),
        ple_embed_dim=int(raw["ple_embed_dim"]),
        ple_layer_index=int(ple_layer_index),
        source_layer_index=int(ids[ple_layer_index]) - 1,
        seed=int(seed),
    )
    return contract, {
        "path": str(config_path),
        "sha256": sha256_bytes(config_path.read_bytes()),
        "model_type": raw.get("model_type"),
        "ple_layer_ids": ids,
        "config_seed": raw.get("seed"),
        "effective_seed": seed,
    }


@dataclass(frozen=True)
class LookupShard:
    ordinal: int
    tensor: str
    path: Path
    data_start: int
    payload_offset: int
    rows: int
    columns: int
    global_start: int


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"short safetensors header length: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return header, 8 + int(header_length)


def load_layout(spec: Path, contract: PLEContract, split_parts: int) -> tuple[list[LookupShard], dict[str, Any]]:
    """Resolve the checkpoint's numeric shard order into a combined row space."""
    index_path = spec / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("source safetensors index has no weight_map")
    prefix = (
        f"model.language_model.layers.{contract.source_layer_index}.ple."
        "ple_embedding.ngram_embedding.shard_"
    )
    matcher = re.compile(rf"^{re.escape(prefix)}(\d+)\.weight$")
    numbered: list[tuple[int, str, Path]] = []
    for tensor, source_file in weight_map.items():
        match = matcher.match(str(tensor))
        if match:
            numbered.append((int(match.group(1)), str(tensor), spec / str(source_file)))
    numbered.sort(key=lambda item: item[0])
    if len(numbered) != split_parts or [item[0] for item in numbered] != list(range(split_parts)):
        raise ValueError(
            "PLE checkpoint tensors do not form the configured numeric shard sequence: "
            f"found={[item[0] for item in numbered]} expected=0..{split_parts - 1}"
        )
    headers: dict[Path, tuple[dict[str, Any], int]] = {}
    shards: list[LookupShard] = []
    global_start = 0
    for ordinal, tensor, path in numbered:
        if path not in headers:
            headers[path] = _read_safetensors_header(path)
        header, data_start = headers[path]
        entry = header.get(tensor)
        if not isinstance(entry, dict) or entry.get("dtype") != "BF16":
            raise ValueError(f"{tensor} is not a BF16 PLE lookup tensor")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not (
            isinstance(shape, list)
            and len(shape) == 2
            and all(_is_int(value) and value > 0 for value in shape)
            and isinstance(offsets, list)
            and len(offsets) == 2
            and all(_is_int(value) and value >= 0 for value in offsets)
        ):
            raise ValueError(f"{tensor} has invalid PLE lookup geometry")
        rows, columns = int(shape[0]), int(shape[1])
        if columns != contract.head_dim:
            raise ValueError(
                f"{tensor} width {columns} does not match PLE head dimension {contract.head_dim}"
            )
        payload = int(offsets[1]) - int(offsets[0])
        if payload != rows * columns * 2:
            raise ValueError(f"{tensor} BF16 payload size does not match its geometry")
        shards.append(
            LookupShard(
                ordinal=ordinal,
                tensor=tensor,
                path=path,
                data_start=data_start,
                payload_offset=int(offsets[0]),
                rows=rows,
                columns=columns,
                global_start=global_start,
            )
        )
        global_start += rows
    if global_start != contract.padded_vocab_size:
        raise ValueError(
            "PLE checkpoint combined rows do not match reference padded vocabulary: "
            f"checkpoint={global_start} reference={contract.padded_vocab_size}"
        )
    return shards, {
        "path": str(index_path),
        "sha256": sha256_bytes(index_path.read_bytes()),
        "tensor_prefix": prefix,
        "shard_count": len(shards),
        "combined_rows": global_start,
        "head_dimension": contract.head_dim,
        "source_bytes": sum(shard.rows * shard.columns * 2 for shard in shards),
    }


def locate_combined_row(shards: Sequence[LookupShard], combined_row: int) -> tuple[LookupShard, int]:
    """Map an exact reference embedding row to its physical checkpoint shard."""
    starts = [shard.global_start for shard in shards]
    index = bisect.bisect_right(starts, int(combined_row)) - 1
    if index < 0 or index >= len(shards):
        raise ValueError(f"combined PLE row is outside the checkpoint: {combined_row}")
    shard = shards[index]
    local_row = int(combined_row) - shard.global_start
    if local_row < 0 or local_row >= shard.rows:
        raise ValueError(f"combined PLE row is outside resolved shard: {combined_row}")
    return shard, local_row


def read_source_row(shard: LookupShard, local_row: int) -> bytes:
    """Return one exact BF16 lookup row without materializing its source tensor.

    The row reader is intentionally shared by the address tracer and later PLE
    output controls.  Keeping the byte-offset calculation here prevents a
    selective-repair experiment from silently drifting from the source-bound
    physical layout that the trace already verified.
    """
    if local_row < 0 or local_row >= shard.rows:
        raise ValueError(f"PLE row is outside resolved shard: {shard.tensor} row {local_row}")
    row_bytes = shard.columns * 2
    offset = shard.data_start + shard.payload_offset + local_row * row_bytes
    with shard.path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(row_bytes)
    if len(raw) != row_bytes:
        raise ValueError(f"short PLE row read for {shard.tensor} row {local_row}")
    return raw


def _shift_right_ignore_eos(token_ids: Sequence[int], shift: int, eos_token_id: int) -> list[int]:
    """Exact scalar equivalent of Qwen4Exp's EOS-aware history shifting."""
    tokens = [int(token) for token in token_ids]
    if shift == 0:
        return tokens
    result: list[int] = []
    previous_eos = -1
    for position, token in enumerate(tokens):
        source_position = position - shift
        position_in_segment = position - (previous_eos + 1)
        if position_in_segment >= shift and source_position >= 0:
            result.append(tokens[source_position])
        else:
            result.append(eos_token_id)
        if token == eos_token_id:
            previous_eos = position
    return result


def trace_token_segments(contract: PLEContract, segments: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    """Trace PLE addresses over token chunks while preserving the two-token state."""
    prior_context = [contract.eos_token_id] * contract.context_len
    traced: list[dict[str, Any]] = []
    session_token_index = 0
    for segment_index, raw_segment in enumerate(segments):
        segment = [int(token) for token in raw_segment]
        if not segment:
            continue
        if any(token < 0 or token >= contract.vocab_size for token in segment):
            raise ValueError("token sequence contains an ID outside the source vocabulary")
        history = prior_context + segment
        shifted = [
            _shift_right_ignore_eos(history, shift, contract.eos_token_id)
            for shift in range(contract.ngram_size)
        ]
        for local_index, token_id in enumerate(segment):
            position = contract.context_len + local_index
            accesses: list[dict[str, int]] = []
            for ngram_order in range(2, contract.ngram_size + 1):
                start = (ngram_order - 2) * contract.heads_per_ngram
                stop = start + contract.heads_per_ngram
                mixed_id = shifted[0][position] * contract.multipliers[0]
                for prior in range(1, ngram_order):
                    mixed_id ^= shifted[prior][position] * contract.multipliers[prior]
                for global_head in range(start, stop):
                    combined_row = (mixed_id % contract.head_vocab_sizes[global_head]) + contract.head_offsets[global_head]
                    accesses.append(
                        {
                            "ngram_order": ngram_order,
                            "head_within_ngram": global_head - start,
                            "global_head": global_head,
                            "combined_row": combined_row,
                        }
                    )
            traced.append(
                {
                    "session_token_index": session_token_index,
                    "segment_index": segment_index,
                    "segment_token_index": local_index,
                    "token_id": token_id,
                    "accesses": accesses,
                }
            )
            session_token_index += 1
        prior_context = (prior_context + segment)[-contract.context_len :]
    return traced


def bind_source_rows(
    shards: Sequence[LookupShard], trace: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Range-read every referenced source row and bind exact payload hashes."""
    rows = sorted(
        {
            int(access["combined_row"])
            for token in trace
            for access in token["accesses"]
        }
    )
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    range_read_bytes = 0
    for combined_row in rows:
        shard, local_row = locate_combined_row(shards, combined_row)
        raw = read_source_row(shard, local_row)
        row_bytes = len(raw)
        digest.update(combined_row.to_bytes(8, "little", signed=False))
        digest.update(shard.tensor.encode("utf-8"))
        digest.update(local_row.to_bytes(8, "little", signed=False))
        digest.update(raw)
        range_read_bytes += len(raw)
        records.append(
            {
                "combined_row": combined_row,
                "tensor": shard.tensor,
                "shard_ordinal": shard.ordinal,
                "local_row": local_row,
                "row_bytes": row_bytes,
                "payload_sha256": sha256_bytes(raw),
            }
        )
    return records, {
        "unique_rows": len(records),
        "range_read_bytes": range_read_bytes,
        "range_read_sha256": digest.hexdigest(),
        "source_passes": 1,
    }


def parse_token_csv(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--token-ids must contain at least one integer")
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise ValueError("--token-ids must be comma-separated integers") from exc


def load_sequence(
    *,
    receipt_path: Path | None,
    explicit_tokens: str | None,
    vocab_size: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Load a receipt-bound token history or an explicitly unbound control."""
    if explicit_tokens is not None:
        tokens = parse_token_csv(explicit_tokens)
        return [tokens], {
            "binding": "EXPLICIT_TOKEN_CONTROL__NOT_ACCEPTED_SEQUENCE_EVIDENCE",
            "segments": [tokens],
            "token_count": len(tokens),
        }
    if receipt_path is None or not receipt_path.is_file():
        raise ValueError("a readable --sequence-receipt or explicit --token-ids is required")
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    if document.get("schema") != SEQUENCE_SCHEMA:
        raise ValueError(f"unexpected sequence receipt schema: {document.get('schema')!r}")
    prompt = document.get("prompt_token_ids")
    generated = document.get("reference_generated_token_ids")
    combined = document.get("token_ids")
    if not (
        isinstance(prompt, list)
        and isinstance(generated, list)
        and isinstance(combined, list)
        and all(_is_int(token) for token in prompt + generated + combined)
        and list(prompt) + list(generated) == list(combined)
    ):
        raise ValueError("sequence receipt has inconsistent prompt/generated/token_ids fields")
    segments = [list(map(int, prompt)), list(map(int, generated))]
    if any(token < 0 or token >= vocab_size for segment in segments for token in segment):
        raise ValueError("sequence receipt token is outside the Flash source vocabulary")
    return segments, {
        "binding": "STATEFUL_ACCEPTED_TOKEN_SEQUENCE_RECEIPT",
        "path": str(receipt_path),
        "sha256": sha256_bytes(receipt_path.read_bytes()),
        "schema": document.get("schema"),
        "status": document.get("status"),
        "pinned_revision": document.get("pinned_revision"),
        "segments": segments,
        "token_count": len(combined),
        "execution_boundary": (
            "The receipt binds a real accepted token sequence, but it does not by itself "
            "establish PLE output parity. This trace remains an address/row-binding control."
        ),
    }


def _reference_revision_from_spec(spec: Path) -> str | None:
    marker = "@"
    if marker not in spec.name:
        return None
    return spec.name.rsplit(marker, 1)[1] or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sequence-receipt", type=Path, default=DEFAULT_SEQUENCE_RECEIPT)
    parser.add_argument("--token-ids", help="comma-separated explicit control tokens; omits accepted-sequence binding")
    parser.add_argument("--ple-layer-index", type=int, default=0)
    parser.add_argument("--reference-implementation", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-version", default="transformers 5.17.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    spec = args.spec.expanduser().resolve()
    if not spec.is_dir():
        raise FileNotFoundError(spec)
    if not args.reference_implementation.is_file():
        raise FileNotFoundError(args.reference_implementation)
    contract, config_binding = load_contract(spec, args.ple_layer_index)
    config_document = json.loads((spec / "config.json").read_text(encoding="utf-8"))
    text_config = config_document["text_config"]
    split_parts = text_config.get("split_ngram_parts")
    if not _is_int(split_parts) or split_parts <= 0:
        raise ValueError("Flash source config has invalid split_ngram_parts")
    shards, layout_binding = load_layout(spec, contract, int(split_parts))
    receipt_path = None if args.token_ids is not None else args.sequence_receipt
    segments, sequence_binding = load_sequence(
        receipt_path=receipt_path,
        explicit_tokens=args.token_ids,
        vocab_size=contract.vocab_size,
    )
    receipt_revision = sequence_binding.get("pinned_revision")
    specimen_revision = _reference_revision_from_spec(spec)
    if receipt_revision and specimen_revision and not str(receipt_revision).startswith(specimen_revision):
        raise ValueError(
            "accepted token receipt revision does not match specimen path revision: "
            f"receipt={receipt_revision} specimen={specimen_revision}"
        )
    trace = trace_token_segments(contract, segments)
    source_rows, source_read = bind_source_rows(shards, trace)
    row_locations = {row["combined_row"]: row for row in source_rows}
    for token in trace:
        for access in token["accesses"]:
            location = row_locations[int(access["combined_row"])]
            access.update(
                {
                    "shard_ordinal": location["shard_ordinal"],
                    "local_row": location["local_row"],
                    "source_row_sha256": location["payload_sha256"],
                }
            )
    lookup_events = sum(len(token["accesses"]) for token in trace)
    row_bytes = contract.head_dim * 2
    reference_path = args.reference_implementation.resolve()
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "SOURCE_ALGORITHM_BOUND_TOKEN_ACCESS_TRACE__NOT_PLE_OUTPUT_PARITY",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "specimen": {
            "root": str(spec),
            "model": "Qwen/Qwen3.8-Flash-Next",
            "revision_prefix": specimen_revision,
        },
        "reference_contract": {
            "implementation": str(reference_path),
            "implementation_sha256": sha256_bytes(reference_path.read_bytes()),
            "implementation_version": args.reference_version,
            "implementation_scope": "Qwen4ExpTextNGramEmbedding.forward",
            "config": config_binding,
            "state_contract": {
                "persistent_token_context": contract.context_len,
                "initial_context_token": contract.eos_token_id,
                "eos_aware_history_reset": True,
                "lookup_heads_per_token": contract.ngram_heads,
            },
            "hash_contract": {
                "ngram_size": contract.ngram_size,
                "heads_per_ngram": contract.heads_per_ngram,
                "multipliers": list(contract.multipliers),
                "head_vocab_sizes": list(contract.head_vocab_sizes),
                "head_offsets": list(contract.head_offsets),
                "total_vocab_size": contract.total_vocab_size,
                "padded_vocab_size": contract.padded_vocab_size,
            },
        },
        "checkpoint_layout": layout_binding,
        "sequence": sequence_binding,
        "trace": trace,
        "active_lookup": {
            "token_count": len(trace),
            "lookup_events": lookup_events,
            "lookup_events_per_token": contract.ngram_heads,
            "embedding_values_per_token": contract.ple_embed_dim,
            "source_row_bytes": row_bytes,
            "source_bytes_referenced_across_events": lookup_events * row_bytes,
            "unique_source_rows": source_read["unique_rows"],
            "unique_source_bytes": source_read["range_read_bytes"],
            "source_range_read_sha256": source_read["range_read_sha256"],
            "source_passes": source_read["source_passes"],
            "source_row_records": source_rows,
        },
        "claim_boundary": (
            "This receipt proves the source-config/reference-algorithm token-to-row address "
            "path and binds each selected physical BF16 row. It does not execute the PLE "
            "key/value projections, norms, convolution, hidden-state gates, or any complete "
            "Flash forward. It claims no compact representation, complete EBPW, direct "
            "runtime, TPS, capability, or source PLE output parity."
        ),
        "promotion_allowed": False,
        "next": (
            "Collect disjoint source-bound token sequences to measure real lookup reuse; then "
            "obtain a source PLE output/activation control before testing a non-uniform compact "
            "base plus protected accessed-row repair. Do not materialize uniform global PQ from "
            "this address trace."
        ),
    }
    document["construction"] = {
        "elapsed_ns": time.perf_counter_ns() - started,
        "model_loads": 0,
        "source_tensor_full_materializations": 0,
        "source_read_mode": "checkpoint headers plus exact selected BF16 row range reads",
    }
    document["seal_sha256"] = sha256_bytes(json.dumps(document, sort_keys=True).encode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": document["status"],
                "tokens": len(trace),
                "lookup_events": lookup_events,
                "unique_source_rows": source_read["unique_rows"],
                "out": str(args.output),
                "seal": document["seal_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
