#!/usr/bin/env python3
"""Isolated MLX-VLM worker for raw Flash source logits.

The Hawking parent supplies exact inputs and validates every emitted byte. This
child contains no admission, owner-signature, or native-executor logic.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import sys
import time
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hawking.persist import atomic_write_bytes, atomic_write_json  # noqa: E402


SCHEMA = "hawking.flash.external_greedy_reference_mlx_worker.v1"
EXECUTION_SCHEDULE = "exact_qwen4_exp_row_addressed_ple_and_experts_with_evaluated_sublayer_boundaries"


class RawSourceExpertStore:
    """Read only the routed experts directly from the immutable source shards."""

    def __init__(self, root: Path, mx: Any, np: Any, *, num_experts: int):
        self.root = root.resolve()
        self.mx = mx
        self.np = np
        self.num_experts = num_experts
        index = json.loads((self.root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map = index["weight_map"]
        self._headers: dict[str, tuple[int, dict[str, Any]]] = {}
        self._fds: dict[str, int] = {}
        self._files_opened: set[str] = set()
        self._expert_gets = 0
        self._source_bytes_read = 0

    @staticmethod
    def _names(layer_id: int) -> tuple[str, str]:
        prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
        return f"{prefix}.gate_up_proj", f"{prefix}.down_proj"

    def experts_present(self, layer_id: int) -> bool:
        gate_up, down = self._names(layer_id)
        return gate_up in self.weight_map and down in self.weight_map

    def _tensor_info(self, name: str) -> tuple[str, int, int, dict[str, Any]]:
        filename = self.weight_map[name]
        if filename not in self._fds:
            path = self.root / filename
            fd = os.open(path, os.O_RDONLY)
            prefix = os.pread(fd, 8, 0)
            if len(prefix) != 8:
                os.close(fd)
                raise ValueError(f"truncated Safetensors header length: {path}")
            header_bytes = int.from_bytes(prefix, "little")
            raw_header = os.pread(fd, header_bytes, 8)
            if len(raw_header) != header_bytes:
                os.close(fd)
                raise ValueError(f"truncated Safetensors header: {path}")
            header = json.loads(raw_header.decode("utf-8"))
            self._fds[filename] = fd
            self._headers[filename] = (8 + header_bytes, header)
            self._files_opened.add(filename)
        data_start, header = self._headers[filename]
        info = header.get(name)
        if not isinstance(info, dict):
            raise ValueError(f"source tensor is absent from its indexed shard: {name}")
        return filename, self._fds[filename], data_start, info

    def _expert(self, name: str, expert_id: int) -> Any:
        filename, fd, data_start, info = self._tensor_info(name)
        del filename
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if (
            info.get("dtype") != "BF16"
            or not isinstance(shape, list)
            or len(shape) != 3
            or shape[0] != self.num_experts
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError(f"source expert tensor has an unsupported header: {name}")
        row_elements = math.prod(shape[1:])
        row_bytes = row_elements * 2
        if offsets[1] - offsets[0] != row_bytes * shape[0]:
            raise ValueError(f"source expert tensor byte span differs from its shape: {name}")
        start = data_start + offsets[0] + expert_id * row_bytes
        raw = os.pread(fd, row_bytes, start)
        if len(raw) != row_bytes:
            raise ValueError(f"short source expert byte-range read: {name}[{expert_id}]")
        words = self.np.frombuffer(raw, dtype="<u2").copy()
        array = self.mx.view(self.mx.array(words), self.mx.bfloat16).reshape(shape[1:])
        self.mx.eval(array)
        self._source_bytes_read += row_bytes
        return array

    def get(self, layer_id: int, expert_id: int):
        gate_up_name, down_name = self._names(layer_id)
        gate_up = self._expert(gate_up_name, expert_id)
        down = self._expert(down_name, expert_id)
        midpoint = gate_up.shape[-2] // 2
        self._expert_gets += 1
        plain = lambda value: (value, None, None)
        return (
            plain(gate_up[:midpoint, :]),
            plain(gate_up[midpoint:, :]),
            plain(down),
        )

    def get_all(self, layer_id: int, needed: Any) -> dict[int, Any]:
        return {int(expert_id): self.get(layer_id, int(expert_id)) for expert_id in needed}

    def release_layer(self, layer_id: int) -> None:
        filenames = {self.weight_map[name] for name in self._names(layer_id)}
        for filename in filenames:
            fd = self._fds.pop(filename, None)
            if fd is not None:
                os.close(fd)
            self._headers.pop(filename, None)
        gc.collect()
        self.mx.clear_cache()

    def stats(self) -> dict[str, Any]:
        return {
            "mode": "direct_immutable_source_shard_slices",
            "expert_gets": self._expert_gets,
            "source_bytes_read": self._source_bytes_read,
            "unique_source_files_opened": len(self._files_opened),
            "resident_source_file_descriptors": len(self._fds),
        }


class BF16SourceNGramEmbedding:
    """Exact row-addressed view of the source checkpoint's BF16 PLE table."""

    def __init__(
        self,
        root: Path,
        mx: Any,
        np: Any,
        manifest: dict[str, Any],
        *,
        cache_rows: int = 256,
    ):
        self.root = root.resolve()
        self.mx = mx
        self.np = np
        self.entries = manifest["shards"]
        self.shard_offsets = tuple(
            [entry["row_start"] for entry in self.entries]
            + [self.entries[-1]["row_start"] + self.entries[-1]["row_count"]]
        )
        self.row_count = self.shard_offsets[-1]
        self.row_width = manifest["row_width"]
        self.cache_rows = cache_rows
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._lookups = 0
        self._rows = 0
        self._hits = 0
        self._misses = 0
        self._bytes = 0

    def _read_row(self, row_id: int) -> Any:
        cached = self._cache.get(row_id)
        if cached is not None:
            self._hits += 1
            self._cache.move_to_end(row_id)
            return cached
        shard_index = bisect_right(self.shard_offsets, row_id) - 1
        if shard_index < 0 or shard_index >= len(self.entries):
            raise IndexError("PLE n-gram row is outside the source table")
        entry = self.entries[shard_index]
        local_row = row_id - entry["row_start"]
        row_bytes = self.row_width * 2
        fd = os.open(self.root / entry["file"], os.O_RDONLY)
        try:
            raw = os.pread(fd, row_bytes, entry["offset"] + local_row * row_bytes)
        finally:
            os.close(fd)
        if len(raw) != row_bytes:
            raise ValueError(f"short source PLE row read: {row_id}")
        row = self.np.frombuffer(raw, dtype="<u2").copy()
        self._misses += 1
        self._bytes += row_bytes
        if self.cache_rows:
            self._cache[row_id] = row
            while len(self._cache) > self.cache_rows:
                self._cache.popitem(last=False)
        return row

    def __call__(self, row_ids: Any) -> Any:
        ids = self.np.asarray(row_ids).astype(self.np.int64, copy=False)
        flat = ids.reshape(-1)
        if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= self.row_count):
            raise IndexError("PLE n-gram row is outside the source table")
        rows = self.np.stack([self._read_row(int(row_id)) for row_id in flat])
        self._lookups += 1
        self._rows += int(flat.size)
        values = self.mx.view(self.mx.array(rows), self.mx.bfloat16)
        return values.reshape(*ids.shape, self.row_width)

    lookup = __call__

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "lookups": self._lookups,
            "rows": self._rows,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "bytes_read": self._bytes,
            "cache_rows": len(self._cache),
        }


def _patch_raw_source_experts(language_model: Any, root: Path, mx: Any, np: Any) -> RawSourceExpertStore:
    from mlx_vlm.models.switch_layers import OffloadedSwitchGLU

    store = RawSourceExpertStore(
        root,
        mx,
        np,
        num_experts=language_model.args.num_experts,
    )
    for layer_id, layer in enumerate(language_model.model.layers):
        if not store.experts_present(layer_id):
            raise ValueError(f"source expert tensors are absent for layer {layer_id}")
        original = layer.mlp.switch_mlp
        layer.mlp.switch_mlp = OffloadedSwitchGLU(
            store,
            layer_id,
            (64, 4, "affine"),
            (64, 4, "affine"),
            (64, 4, "affine"),
            activation=original.activation,
        )
    gc.collect()
    mx.clear_cache()
    return store


def _patch_row_addressed_ple(
    language_model: Any,
    root: Path,
    out_dir: Path,
    mx: Any,
    np: Any,
) -> dict[str, Any]:
    manifest_path = out_dir / "provider-ple-range-manifest.json"
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    header_cache: dict[str, tuple[int, dict[str, Any]]] = {}

    def descriptor(name: str) -> dict[str, Any]:
        filename = weight_map[name]
        if filename not in header_cache:
            path = root / filename
            fd = os.open(path, os.O_RDONLY)
            try:
                prefix = os.pread(fd, 8, 0)
                header_bytes = int.from_bytes(prefix, "little")
                raw_header = os.pread(fd, header_bytes, 8)
            finally:
                os.close(fd)
            if len(prefix) != 8 or len(raw_header) != header_bytes:
                raise ValueError(f"truncated source PLE Safetensors header: {path}")
            header_cache[filename] = (8 + header_bytes, json.loads(raw_header.decode("utf-8")))
        data_start, header = header_cache[filename]
        info = header[name]
        if info.get("dtype") != "BF16" or len(info.get("shape", [])) != 2:
            raise ValueError(f"source PLE tensor is not a BF16 matrix: {name}")
        shape = [int(value) for value in info["shape"]]
        offsets = [int(value) for value in info["data_offsets"]]
        if offsets[1] - offsets[0] != math.prod(shape) * 2:
            raise ValueError(f"source PLE tensor span differs from its shape: {name}")
        return {
            "tensor": name,
            "file": filename,
            "offset": data_start + offsets[0],
            "dtype": "BF16",
            "shape": shape,
        }

    patched_layers = []
    manifest_shards: list[dict[str, Any]] = []
    row_start = 0
    row_width = None
    for layer_index, layer in enumerate(language_model.model.layers):
        if "ple" not in layer:
            continue
        embedding = layer.ple.ple_embedding
        source_rows = embedding.ngram_embedding.shard_offsets[-1]
        for shard_index, expected_rows in enumerate(embedding.ngram_embedding.shard_sizes):
            name = (
                f"model.language_model.layers.{layer_index}.ple.ple_embedding."
                f"ngram_embedding.shard_{shard_index}.weight"
            )
            item = descriptor(name)
            if item["shape"][0] != expected_rows:
                raise ValueError("source PLE shard row count differs from the model configuration")
            if row_width is None:
                row_width = item["shape"][1]
            elif item["shape"][1] != row_width:
                raise ValueError("source PLE shard widths differ")
            manifest_shards.append({**item, "row_start": row_start, "row_count": expected_rows})
            row_start += expected_rows
        if row_start != source_rows:
            raise ValueError("row-addressed PLE count differs from the source PLE module")
        patched_layers.append(layer_index)
    if not patched_layers:
        raise ValueError("source model exposed no PLE layer to patch")
    manifest = {
        "schema": "hawking.flash.external_reference_bf16_ple_ranges.v1",
        "status": "INDEXED_IMMUTABLE_SOURCE_PLE_ROWS",
        "version": 1,
        "layout": "safetensors_ranges",
        "source_root": str(root.resolve()),
        "row_count": row_start,
        "row_width": row_width,
        "shards": manifest_shards,
        "source_mutated": False,
        "weight_payload_copied": False,
    }
    atomic_write_json(manifest_path, manifest)
    replacement = BF16SourceNGramEmbedding(root, mx, np, manifest, cache_rows=256)
    for layer_index in patched_layers:
        language_model.model.layers[layer_index].ple.ple_embedding.ngram_embedding = replacement
    gc.collect()
    mx.clear_cache()
    return {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "schema": manifest["schema"],
        "schema_version": manifest["version"],
        "layout": manifest["layout"],
        "row_count": manifest["row_count"],
        "row_width": manifest["row_width"],
        "patched_layer_ids": patched_layers,
        "source_mutated": False,
        "weight_payload_copied": False,
    }


def _cache_offset(cache_entry: Any) -> int:
    offset = getattr(cache_entry, "offset", 0)
    if hasattr(offset, "item"):
        return int(offset.item())
    return int(offset)


def _eval_stage(
    mx: Any,
    tree_flatten: Any,
    outputs: Any,
    cache: list[Any],
    *,
    progress: dict[str, Any],
    capture_path: Path,
    layer_index: int | None,
    stage: str,
    include_cache: bool = False,
) -> None:
    progress["active_layer_index"] = layer_index
    progress["active_stage"] = stage
    progress["updated_unix_ns"] = time.time_ns()
    atomic_write_json(capture_path, progress)
    arrays = [
        value
        for _, value in tree_flatten(outputs)
        if value is not None and hasattr(value, "shape")
    ]
    state_arrays = [] if not include_cache else [
        value
        for _, value in tree_flatten([entry.state for entry in cache])
        if value is not None and hasattr(value, "shape")
    ]
    mx.eval(*arrays, *state_arrays)
    mx.clear_cache()
    progress["last_completed_layer_index"] = layer_index
    progress["last_completed_stage"] = stage
    progress["updated_unix_ns"] = time.time_ns()
    atomic_write_json(capture_path, progress)


def _checkpointed_qwen4_exp_logits(
    language_model: Any,
    expert_store: RawSourceExpertStore,
    inputs: Any,
    cache: list[Any],
    mx: Any,
    tree_flatten: Any,
    *,
    generation_index: int,
    progress: dict[str, Any],
    capture_path: Path,
) -> Any:
    """Run the installed provider's exact layer math with watchdog-safe seams.

    The ordinary provider constructs all 48 layers in one lazy Metal graph.
    Evaluating that graph exceeded the macOS command-buffer watchdog on this
    source. This preserves the installed Qwen4-Exp prefill/decode math while
    materializing embedding, attention, routed-MLP, hidden, and cache seams.
    """
    from mlx_vlm.models.qwen3_5.language import _create_qwen3_5_ssm_mask
    from mlx_vlm.models.qwen4_exp.language import (
        _QWEN4_BATCH_INVARIANT_FORWARD,
        _create_qwen3_5_attention_mask,
        _create_qwen4_exp_attention_mask,
    )

    model = language_model.model
    offset = _cache_offset(cache[model.fa_idx])
    position_ids = mx.arange(
        offset,
        offset + inputs.shape[-1],
        dtype=inputs.dtype,
    )[None, :]
    hidden = model.embed_tokens(inputs)
    _eval_stage(
        mx,
        tree_flatten,
        hidden,
        cache,
        progress=progress,
        capture_path=capture_path,
        layer_index=None,
        stage="embedding",
    )
    hidden = mx.tile(hidden, (1, 1, model.args.hc_count))
    exact_decode = (
        inputs.shape[-1] == 1
        and language_model._supports_batch_invariant_decode()
    )
    fa_mask = (
        _create_qwen3_5_attention_mask(hidden, cache[model.fa_idx])
        if exact_decode
        else _create_qwen4_exp_attention_mask(hidden, cache[model.fa_idx])
    )
    ssm_mask = _create_qwen3_5_ssm_mask(hidden, cache[model.ssm_idx])
    invariant = _QWEN4_BATCH_INVARIANT_FORWARD
    gdn_sink: list[Any] = []
    for layer_index, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
        progress["active_generation_index"] = generation_index
        progress["execution_schedule"] = EXECUTION_SCHEDULE
        layer_mask = ssm_mask if layer.is_linear else fa_mask
        if "ple" in layer:
            ple = (
                invariant._ple(layer.ple, hidden, inputs, layer_cache, layer_mask)
                if exact_decode
                else layer.ple(hidden, inputs, layer_cache, layer_mask)
            )
            hidden = hidden + ple
            _eval_stage(
                mx,
                tree_flatten,
                hidden,
                cache,
                progress=progress,
                capture_path=capture_path,
                layer_index=layer_index,
                stage="ple",
                include_cache=True,
            )

        if exact_decode:
            mixed, hyper_input, injection = invariant._hyper_connection(
                layer.attn_hyper_connection,
                hidden,
            )
        else:
            mixed, hyper_input, injection = layer.attn_hyper_connection(hidden)
        _eval_stage(
            mx,
            tree_flatten,
            (mixed, hyper_input, injection),
            cache,
            progress=progress,
            capture_path=capture_path,
            layer_index=layer_index,
            stage="attention_hyper_connection",
        )

        if exact_decode and layer.is_linear:
            branch = invariant._gated_delta(
                layer.linear_attn,
                mixed,
                layer_mask,
                layer_cache,
                gdn_sink,
            )
        elif exact_decode:
            attention_mask = invariant._qsa_mask(
                layer.self_attn,
                mixed,
                layer_cache,
                position_ids,
                layer_mask,
            )
            branch = invariant._attention(
                layer.self_attn,
                mixed,
                attention_mask,
                layer_cache,
                position_ids,
                None,
            )
        elif layer.is_linear:
            branch = layer.linear_attn(mixed, mask=layer_mask, cache=layer_cache)
        else:
            branch = layer.self_attn(
                mixed,
                mask=layer_mask,
                cache=layer_cache,
                position_ids=position_ids,
            )
        _eval_stage(
            mx,
            tree_flatten,
            branch,
            cache,
            progress=progress,
            capture_path=capture_path,
            layer_index=layer_index,
            stage="attention",
            include_cache=True,
        )
        gdn_sink.clear()

        hidden = (
            invariant._inject(branch, hyper_input, injection)
            if exact_decode
            else hyper_input
            + (branch[..., None, :] * injection[..., None]).reshape(*hyper_input.shape)
        )
        _eval_stage(
            mx,
            tree_flatten,
            hidden,
            cache,
            progress=progress,
            capture_path=capture_path,
            layer_index=layer_index,
            stage="attention_injection",
        )

        if exact_decode:
            mixed, hyper_input, injection = invariant._hyper_connection(
                layer.mlp_hyper_connection,
                hidden,
            )
        else:
            mixed, hyper_input, injection = layer.mlp_hyper_connection(hidden)
        _eval_stage(
            mx,
            tree_flatten,
            (mixed, hyper_input, injection),
            cache,
            progress=progress,
            capture_path=capture_path,
            layer_index=layer_index,
            stage="mlp_hyper_connection",
        )
        progress["active_layer_index"] = layer_index
        progress["active_stage"] = "mlp_route_and_experts"
        progress["updated_unix_ns"] = time.time_ns()
        atomic_write_json(capture_path, progress)
        branch = (
            invariant._feed_forward(layer.mlp, mixed)
            if exact_decode
            else layer.mlp(mixed)
        )
        _eval_stage(
            mx,
            tree_flatten,
            branch,
            cache,
            progress=progress,
            capture_path=capture_path,
            layer_index=layer_index,
            stage="mlp_route_and_experts",
        )
        hidden = (
            invariant._inject(branch, hyper_input, injection)
            if exact_decode
            else hyper_input
            + (branch[..., None, :] * injection[..., None]).reshape(*hyper_input.shape)
        )
        _eval_stage(
            mx,
            tree_flatten,
            hidden,
            cache,
            progress=progress,
            capture_path=capture_path,
            layer_index=layer_index,
            stage="mlp_injection",
        )
        expert_store.release_layer(layer_index)

    if exact_decode:
        logits_hidden = invariant._hyper_connection(model.hyper_connection_mixer, hidden)
    else:
        logits_hidden = model.hyper_connection_mixer(hidden)
    _eval_stage(
        mx,
        tree_flatten,
        logits_hidden,
        cache,
        progress=progress,
        capture_path=capture_path,
        layer_index=None,
        stage="final_hyper_connection_mixer",
    )
    logits = (
        invariant._linear(language_model.lm_head, logits_hidden)
        if exact_decode
        else language_model.lm_head(logits_hidden)
    )
    _eval_stage(
        mx,
        tree_flatten,
        logits,
        cache,
        progress=progress,
        capture_path=capture_path,
        layer_index=None,
        stage="lm_head",
        include_cache=True,
    )
    return logits


def _cache_identity(cache: list[Any], mx: Any, np: Any, tree_flatten: Any) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total = 0
    for name, value in tree_flatten([entry.state for entry in cache]):
        if value is None or not hasattr(value, "shape"):
            continue
        array = (
            np.asarray(value.view(mx.uint16))
            if value.dtype == mx.bfloat16
            else np.asarray(value)
        )
        raw = array.tobytes(order="C")
        total += len(raw)
        records.append({
            "path": str(name),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "hawking.flash.external_source_state_boundary.v1",
        "array_count": len(records),
        "bytes": total,
        "arrays_sha256": hashlib.sha256(canonical).hexdigest(),
        "arrays": records,
    }


def run(root: Path, out_dir: Path, prompt: list[int], steps_count: int, vocab_size: int) -> dict[str, Any]:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.utils import load_model

    if steps_count < 2:
        raise ValueError("source reference requires at least two generated steps")
    out_dir = out_dir.resolve()
    capture_path = out_dir / "provider-capture.json"
    if capture_path.exists():
        raise FileExistsError(f"refusing to overwrite provider capture: {capture_path}")
    started = time.time_ns()
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "root": str(root.resolve()),
        "prompt_token_ids": prompt,
        "generated_token_ids": [],
        "steps": [],
        "started_unix_ns": started,
        "no_sampling": True,
        "execution_schedule": EXECUTION_SCHEDULE,
    }
    atomic_write_json(capture_path, document)

    model = load_model(root, lazy=True, strict=True)
    language_model = model.language_model
    ple_storage = _patch_row_addressed_ple(language_model, root, out_dir, mx, np)
    expert_store = _patch_raw_source_experts(language_model, root, mx, np)
    ple_layers = [index for index, layer in enumerate(language_model.model.layers) if hasattr(layer, "ple")]
    if not ple_layers:
        raise ValueError("external Qwen4-Exp provider instantiated no PLE layers")
    document["ple_storage_manifest"] = ple_storage
    document["updated_unix_ns"] = time.time_ns()
    atomic_write_json(capture_path, document)
    cache = make_prompt_cache(language_model)
    references: list[int] = []
    records: list[dict[str, Any]] = []
    for generation_index in range(steps_count):
        logical_input = [*prompt, *references]
        execution_input = prompt if generation_index == 0 else [references[-1]]
        inputs = mx.array([execution_input], dtype=mx.int64)
        step_started = time.time_ns()
        logits = _checkpointed_qwen4_exp_logits(
            language_model,
            expert_store,
            inputs,
            cache,
            mx,
            tree_flatten,
            generation_index=generation_index,
            progress=document,
            capture_path=capture_path,
        )[0, -1, :].astype(mx.float32)
        mx.eval(logits)
        values = np.asarray(logits, dtype=np.float32)
        if values.ndim != 1 or values.shape[0] != vocab_size:
            raise ValueError(f"external source logits have unexpected shape: {values.shape}")
        if not bool(np.isfinite(values).all()):
            raise ValueError("external source logits contain a non-finite value")
        token_id = int(np.argmax(values))
        raw = values.astype("<f4", copy=False).tobytes(order="C")
        payload = out_dir / f"logits-{generation_index:06d}.f32"
        atomic_write_bytes(payload, raw)
        references.append(token_id)
        records.append({
            "generation_index": generation_index,
            "input_token_ids": logical_input,
            "incremental_execution_input_token_ids": execution_input,
            "argmax_token_id": token_id,
            "logits_f32": {
                "path": str(payload),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "dtype": "F32_LE",
                "elements": values.shape[0],
                "bytes": len(raw),
            },
            "state_boundary": _cache_identity(cache, mx, np, tree_flatten),
            "elapsed_ns": time.time_ns() - step_started,
        })
        document["generated_token_ids"] = list(references)
        document["steps"] = list(records)
        document.pop("active_generation_index", None)
        document.pop("last_completed_layer_index", None)
        document["updated_unix_ns"] = time.time_ns()
        atomic_write_json(capture_path, document)
        mx.clear_cache()

    document.update({
        "status": "CAPTURED_TWO_STEP_GREEDY_F32",
        "generated_token_ids": references,
        "steps": records,
        "ple_layer_ids": ple_layers,
        "incremental_cache_continuity": True,
        "expert_source_access": expert_store.stats(),
        "ple_storage_manifest": ple_storage,
        "ple_storage_stats": [
            layer.ple.ple_embedding.ngram_embedding.stats
            for layer in language_model.model.layers
            if "ple" in layer
        ],
        "finished_unix_ns": time.time_ns(),
        "elapsed_ns": time.time_ns() - started,
        "max_resident_set_size_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "claim_boundary": "external MLX-VLM source execution only; no Hawking admission or owner authority",
    })
    atomic_write_json(capture_path, document)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    args = parser.parse_args(argv)
    prompt = json.loads(args.prompt_json)
    if not isinstance(prompt, list) or not all(isinstance(token, int) and token >= 0 for token in prompt):
        raise ValueError("prompt-json must be a non-negative integer token list")
    try:
        result = run(args.root, args.out_dir, prompt, args.steps, args.vocab_size)
    except Exception as exc:
        capture_path = args.out_dir.resolve() / "provider-capture.json"
        try:
            document = json.loads(capture_path.read_text(encoding="utf-8"))
        except Exception:
            document = {
                "schema": SCHEMA,
                "root": str(args.root.resolve()),
                "prompt_token_ids": prompt,
                "generated_token_ids": [],
                "steps": [],
            }
        document.update({
            "status": "FAILED_EXTERNAL_PROVIDER_RUNTIME__NO_NEW_SOURCE_RESULT",
            "failure": {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "scientific_disposition": "none",
            },
            "finished_unix_ns": time.time_ns(),
            "claim_boundary": "provider/runtime failure only; not source evidence, teacher admission, or a Flash scientific disposition",
        })
        atomic_write_json(capture_path, document)
        raise
    print(json.dumps({
        "status": result["status"],
        "generated_token_ids": result["generated_token_ids"],
        "elapsed_ns": result["elapsed_ns"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
