#!/usr/bin/env python3.12
"""Teacher-evidence capture for the GLM-5.2 BF16 stream.

Eviction is the only irreversible step in the source traversal, and a BF16 body
carries teacher states that nothing downstream can reconstruct once the file is
gone.  This module makes the durable pipeline

    VERIFY -> TEACHER_CAPTURE -> PROBE/PACK -> SEAL -> EVICT

by running the sealed NumPy reference forward (``glm52_reference``) over the
resident BF16 tensors of a dependency window and writing a bounded, sealed,
reloadable capsule of what those weights actually compute.

What a capsule is and is not:

* It IS the block's real function on a sealed calibration input: attention
  selections and output, router logits, top-8 identity/weights, the 8th-vs-9th
  margin, shared and weighted-routed expert outputs, the post-MoE state, and the
  post-block residual, at BF16-sourced float32 precision.
* It is NOT a capability result.  The calibration input is a sealed synthetic
  token-id probe, not natural text, and windows captured before their upstream
  layers were resident are seeded from the embedding table rather than chained.
  Both facts are recorded per capsule and must be carried into any claim.

Bounded by construction: one batch of ``CALIBRATION_TOKENS`` positions, one
tensor read at a time through an exact shard-local ``pread``, no full activation
corpus, and no unbounded array kept in the capsule.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_reference as reference  # noqa: E402
from glm52_adapter import (  # noqa: E402
    IMMUTABLE_REVISION,
    OFFICIAL_INDEXER_TYPES,
    PROFILE_OFFICIAL,
    load_json_strict,
    validate_config,
)
from glm52_common import (  # noqa: E402
    Glm52Error,
    atomic_bytes,
    atomic_json,
    canonical,
    read_sealed_json,
    seal,
    utc_now,
    verify_sealed,
)

ROOT = CONDENSE.parents[1]
SUPPORT = Path(
    os.environ.get(
        "GLM52_SUPPORT_ROOT",
        "/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity",
    )
)
SOURCE_ROOT = SUPPORT / "source"
CONTROL_ROOT = SUPPORT / "control"
FETCH_LEDGER = SUPPORT / "source_fetch" / "SOURCE_FETCH_LEDGER.jsonl"
TEACHER_DIR = SUPPORT / "source_fetch" / "teacher"
CAPSULES = TEACHER_DIR / "capsules"
LEDGER = TEACHER_DIR / "GLM52_TEACHER_EVIDENCE_LEDGER.jsonl"
POLICY_PATH = ROOT / "GLM52_TEACHER_EVIDENCE_POLICY.json"
GRAPH_PATH = ROOT / "GLM52_SHARD_DEPENDENCY_GRAPH.json"
SCHEDULE_PATH = ROOT / "GLM52_STREAMING_SCHEDULE.json"
MANIFEST_PATH = ROOT / "GLM52_OFFICIAL_MANIFEST.json"

# One batch, this many positions.  Small enough that a capsule is megabytes and a
# capture is minutes, large enough that routing sees a real spread of experts.
CALIBRATION_TOKENS = 8
CALIBRATION_SPLITS = ("teacher_fit", "teacher_holdout")
DEFAULT_SPLIT = "teacher_fit"
# Every tensor this forward needs fits well under this; o_proj at ~201 MB is the
# largest.  The cap is what keeps "capture" from becoming "load the checkpoint".
MAX_TENSOR_BYTES = 320 * 1024 * 1024
# Logit-lens vocabulary subset: lm_head is 1.9 GB, so only these rows are read.
LOGIT_LENS_ROWS = 1024

SCHEMA_CAPSULE = "hawking.glm52.teacher_capsule.v1"
SCHEMA_POLICY = "hawking.glm52.teacher_evidence_policy.v1"


class TeacherCaptureError(Glm52Error):
    """Raised when teacher evidence cannot be captured, sealed or reproduced."""


# --------------------------------------------------------------------------- #
# sealed inputs
# --------------------------------------------------------------------------- #


def _graph() -> dict[str, Any]:
    return read_sealed_json(GRAPH_PATH)


def _schedule() -> dict[str, Any]:
    return read_sealed_json(SCHEDULE_PATH)


def official_config() -> dict[str, Any]:
    """Pinned-revision ``config.json`` validated against the frozen contract."""
    path = CONTROL_ROOT / "config.json"
    if not path.exists():
        raise TeacherCaptureError(
            f"pinned config.json is absent: {path} (fetch it at revision "
            f"{IMMUTABLE_REVISION} before capture)"
        )
    config = load_json_strict(path)
    if not isinstance(config, dict):
        raise TeacherCaptureError("config.json must contain a JSON object")
    geometry = validate_config(config, profile=PROFILE_OFFICIAL)
    resolved = dict(config)
    resolved["indexer_types"] = list(OFFICIAL_INDEXER_TYPES)
    resolved["mlp_layer_types"] = [
        "dense" if layer < geometry.first_k_dense_replace else "sparse"
        for layer in range(geometry.num_hidden_layers)
    ]
    return resolved


def _tensor_table(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for tensor in graph["tensors"]:
        if tensor.get("alias_of"):
            continue
        table[tensor["name"]] = {
            "shard": tensor["shard"],
            "absolute_start": int(tensor["absolute_start"]),
            "payload_bytes": int(tensor["payload_bytes"]),
            "dtype": tensor["dtype"],
            "shape": tuple(int(value) for value in tensor["shape"]),
        }
    return table


def layer_of_organ(organ_id: str) -> int | None:
    if organ_id.startswith("text_layer_"):
        return int(organ_id.rsplit("_", 1)[1])
    if organ_id.startswith("mtp_"):
        return None
    return None


def organs_by_shard(graph: dict[str, Any]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for organ in graph["organs"]:
        for shard in organ["source_shards"]:
            mapping.setdefault(shard, set()).add(organ["organ_id"])
    return mapping


def organ_shards(graph: dict[str, Any]) -> dict[str, set[str]]:
    return {organ["organ_id"]: set(organ["source_shards"]) for organ in graph["organs"]}


# --------------------------------------------------------------------------- #
# bounded tensor source
# --------------------------------------------------------------------------- #


class ShardTensorSource:
    """``glm52_reference.TensorSource`` backed by exact shard-local ``pread``.

    One complete tensor per call, never a shard and never a prefix.  Row reads
    exist so the 1.9 GB embedding and lm_head tables are never materialised.
    """

    def __init__(
        self,
        root: Path,
        table: dict[str, dict[str, Any]],
        *,
        max_tensor_bytes: int = MAX_TENSOR_BYTES,
    ) -> None:
        self.root = Path(root)
        self.table = table
        self.max_tensor_bytes = int(max_tensor_bytes)
        self.payload_bytes_read = 0
        self.read_calls = 0

    def _record(self, name: str) -> dict[str, Any]:
        record = self.table.get(name)
        if record is None:
            raise TeacherCaptureError(f"tensor absent from the sealed graph: {name!r}")
        return record

    def resident(self, name: str) -> bool:
        record = self.table.get(name)
        return record is not None and (self.root / record["shard"]).exists()

    def _pread(self, record: dict[str, Any], count: int, offset: int, name: str) -> bytes:
        path = self.root / record["shard"]
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            chunks: list[bytes] = []
            remaining = count
            position = offset
            while remaining:
                block = os.pread(fd, remaining, position)
                if not block:
                    raise TeacherCaptureError(
                        f"short read for {name}: wanted {count} at {offset} in {path.name}"
                    )
                chunks.append(block)
                remaining -= len(block)
                position += len(block)
        finally:
            os.close(fd)
        self.payload_bytes_read += count
        self.read_calls += 1
        return b"".join(chunks)

    @staticmethod
    def _decode(payload: bytes, dtype: str) -> np.ndarray:
        if dtype == "BF16":
            words = np.frombuffer(payload, dtype="<u2")
            return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
        if dtype == "F32":
            return np.frombuffer(payload, dtype="<f4").copy()
        raise TeacherCaptureError(f"unsupported source dtype: {dtype}")

    def tensor(self, name: str) -> np.ndarray:
        record = self._record(name)
        if record["payload_bytes"] > self.max_tensor_bytes:
            raise TeacherCaptureError(
                f"bounded read refused {name}: {record['payload_bytes']} bytes "
                f"exceeds {self.max_tensor_bytes}"
            )
        payload = self._pread(record, record["payload_bytes"], record["absolute_start"], name)
        return self._decode(payload, record["dtype"]).reshape(record["shape"]).copy()

    def rows(self, name: str, ids: Iterable[int]) -> np.ndarray:
        """Read only the requested rows of a 2-D table."""
        record = self._record(name)
        if len(record["shape"]) != 2:
            raise TeacherCaptureError(f"row read requires a 2-D tensor: {name}")
        rows, width = record["shape"]
        item = {"BF16": 2, "F32": 4}[record["dtype"]]
        stride = width * item
        wanted = np.asarray(list(np.asarray(ids).reshape(-1)), dtype=np.int64)
        if wanted.size and (wanted.min() < 0 or wanted.max() >= rows):
            raise TeacherCaptureError(f"row index out of range for {name}")
        out = np.empty((wanted.size, width), dtype=np.float32)
        for slot, row in enumerate(wanted):
            payload = self._pread(
                record, stride, record["absolute_start"] + int(row) * stride, name
            )
            out[slot] = self._decode(payload, record["dtype"])
        return out.reshape(*np.asarray(ids).shape, width)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #


def calibration_ids(split: str, *, vocab_size: int, tokens: int = CALIBRATION_TOKENS) -> np.ndarray:
    """Deterministic, split-disjoint token-id probe bound to the pinned revision."""
    if split not in CALIBRATION_SPLITS:
        raise TeacherCaptureError(f"unknown calibration split: {split!r}")
    stream = hashlib.sha256(
        canonical({"revision": IMMUTABLE_REVISION, "split": split, "tokens": tokens})
    ).digest()
    ids: list[int] = []
    counter = 0
    while len(ids) < tokens:
        block = hashlib.sha256(stream + counter.to_bytes(4, "big")).digest()
        for offset in range(0, len(block), 4):
            if len(ids) == tokens:
                break
            ids.append(int.from_bytes(block[offset:offset + 4], "big") % vocab_size)
        counter += 1
    return np.asarray([ids], dtype=np.int64)


def membership_sha256(ids: np.ndarray, split: str) -> str:
    return hashlib.sha256(
        canonical({"split": split, "token_ids": np.asarray(ids).tolist()})
    ).hexdigest()


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _finite(value: np.ndarray) -> np.ndarray:
    """Replace non-finite entries so a metric is a number, not a NaN."""
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _layer_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    output = _finite(arrays["block_output"])
    metrics = {
        "block_output_l2": float(np.sqrt(np.sum(output * output, dtype=np.float64))),
        "block_output_absmax": float(np.max(np.abs(output))),
        "block_output_mean": float(np.mean(output, dtype=np.float64)),
        "attention_output_l2": float(
            np.sqrt(np.sum(_finite(arrays["attention_output"]) ** 2, dtype=np.float64))
        ),
        "post_moe_l2": float(np.sqrt(np.sum(_finite(arrays["post_moe"]) ** 2, dtype=np.float64))),
    }
    if "router_logits" in arrays:
        logits = _finite(arrays["router_logits"])
        margin = _finite(arrays["topk_margin_8th_vs_9th"])
        metrics.update({
            "router_logit_absmax": float(np.max(np.abs(logits))),
            "router_selected_expert_count": float(len(set(np.asarray(
                arrays["topk_indices"]).ravel().tolist()))),
            "topk_margin_mean": float(np.mean(margin, dtype=np.float64)),
            "topk_margin_min": float(np.min(margin)),
            "shared_output_l2": float(
                np.sqrt(np.sum(_finite(arrays["shared_expert_output"]) ** 2, dtype=np.float64))
            ),
            "routed_output_l2": float(
                np.sqrt(np.sum(_finite(arrays["routed_expert_output"]) ** 2, dtype=np.float64))
            ),
        })
    return {key: value for key, value in sorted(metrics.items())}


def _router_margin(
    router_logits: np.ndarray, correction_bias: np.ndarray, top_k: int
) -> np.ndarray:
    """Gap between the last selected and first rejected corrected expert score."""
    logits = np.asarray(router_logits, dtype=np.float32)
    scores = np.float32(1.0) / (np.float32(1.0) + np.exp(-logits, dtype=np.float32))
    corrected = scores + np.asarray(correction_bias, dtype=np.float32)
    ordered = np.sort(corrected, axis=-1)[..., ::-1]
    if ordered.shape[-1] <= top_k:
        return np.zeros(ordered.shape[:-1], dtype=np.float32)
    return (ordered[..., top_k - 1] - ordered[..., top_k]).astype(np.float32)


def capture_layer(
    hidden: np.ndarray,
    source: Any,
    layer: int,
    config: dict[str, Any],
    previous_topk: np.ndarray | None,
    cache: reference.ReferenceCache,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run one real block and keep exactly the bounded evidence it produced."""
    positions = np.arange(hidden.shape[1], dtype=np.int64)[None, :]
    indexer_type = config["indexer_types"][layer]
    mlp_type = config["mlp_layer_types"][layer]
    output, topk, trace = reference.decoder_layer(
        hidden,
        source,
        layer,
        config,
        positions,
        cache,
        mlp_type=mlp_type,
        indexer_type=indexer_type,
        previous_topk=previous_topk,
    )
    attention = trace["attention"]
    arrays: dict[str, np.ndarray] = {
        "input_hidden": np.asarray(trace["input"], dtype=np.float32),
        "attention_input": np.asarray(trace["attention_input"], dtype=np.float32),
        "index_selection": np.asarray(attention["topk_indices"], dtype=np.int32),
        "index_scores": np.asarray(attention["index_scores"], dtype=np.float32),
        "attention_output": np.asarray(attention["attention_output"], dtype=np.float32),
        "post_attention_hidden": np.asarray(trace["post_attention"], dtype=np.float32),
        "pre_router_hidden": np.asarray(trace["mlp_input"], dtype=np.float32),
        "block_output": np.asarray(output, dtype=np.float32),
    }
    arrays["post_moe"] = arrays["block_output"] - arrays["post_attention_hidden"]
    mlp = trace["mlp"]
    if mlp["kind"] == "sparse":
        bias = source.tensor(f"model.layers.{layer}.mlp.gate.e_score_correction_bias")
        arrays.update({
            "router_logits": np.asarray(mlp["router_logits"], dtype=np.float32),
            "topk_indices": np.asarray(mlp["topk_indices"], dtype=np.int32),
            "topk_weights": np.asarray(mlp["topk_weights"], dtype=np.float32),
            "topk_margin_8th_vs_9th": _router_margin(
                mlp["router_logits"], bias, int(config["num_experts_per_tok"])
            ),
            "shared_expert_output": np.asarray(mlp["shared_output"], dtype=np.float32),
            "routed_expert_output": np.asarray(mlp["routed_output"], dtype=np.float32),
        })
    return output, topk, arrays


def _logit_lens(source: Any, hidden: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Short logits over a sealed vocabulary subset.  Not the model's output."""
    if not source.resident("model.norm.weight") or not source.resident("lm_head.weight"):
        return {}
    normed = reference.rmsnorm(
        hidden, source.tensor("model.norm.weight"), float(config["rms_norm_eps"])
    )
    rows = np.arange(LOGIT_LENS_ROWS, dtype=np.int64)
    head = source.rows("lm_head.weight", rows)
    return {"short_logits": reference.linear(normed, head).astype(np.float32)}


def _previous_capsule(first_layer: int, out_dir: Path) -> dict[str, Any] | None:
    """The sealed capsule whose carry-out feeds this layer run, if one exists."""
    if first_layer == 0 or not out_dir.exists():
        return None
    for path in sorted(out_dir.glob("*.json")):
        receipt = json.loads(path.read_text())
        if receipt.get("last_layer") == first_layer - 1:
            return receipt
    return None


def capsule_id(layers: list[int]) -> str:
    return f"L{layers[0]:02d}_L{layers[-1]:02d}"


def contiguous_runs(layers: Iterable[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for layer in sorted(set(int(value) for value in layers)):
        if runs and layer == runs[-1][-1] + 1:
            runs[-1].append(layer)
        else:
            runs.append([layer])
    return runs


def _append_ledger(row: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def capture_layers(
    layers: list[int],
    *,
    split: str = DEFAULT_SPLIT,
    source_root: Path | None = None,
    graph: dict[str, Any] | None = None,
    schedule: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    capsule_dir: Path | None = None,
) -> dict[str, Any]:
    """Capture, seal and ledger one contiguous layer run as a teacher capsule.

    Refuses rather than approximates: a layer whose tensors are not all resident
    cannot be captured, and eviction must keep refusing it.  The unit is a layer
    run rather than a window because a window can be half-destroyed by the
    pre-capture policy, and the resident half is still worth capturing.
    """
    started = time.time()
    root = Path(source_root) if source_root is not None else SOURCE_ROOT
    graph = graph if graph is not None else _graph()
    schedule = schedule if schedule is not None else _schedule()
    config = config if config is not None else official_config()
    out_dir = Path(capsule_dir) if capsule_dir is not None else CAPSULES

    layers = sorted(int(value) for value in layers)
    if not layers:
        raise TeacherCaptureError("no layers requested")
    if layers != list(range(layers[0], layers[0] + len(layers))):
        raise TeacherCaptureError(f"layers are not contiguous: {layers}")
    identity = capsule_id(layers)
    window_ids = sorted(
        window["window_id"]
        for window in schedule["windows"]
        if {layer_of_organ(organ) for organ in window["organ_ids"]} & set(layers)
    )

    table = _tensor_table(graph)
    source = ShardTensorSource(root, table)
    shards = organ_shards(graph)
    missing = sorted(
        shard
        for layer in layers
        for shard in shards.get(f"text_layer_{layer:02d}", set())
        if not (root / shard).exists()
    )
    if missing:
        raise TeacherCaptureError(
            f"{identity} is not fully resident; cannot capture: missing {missing[:3]}"
        )

    tokens = CALIBRATION_TOKENS
    if tokens > int(config["index_topk"]):
        raise TeacherCaptureError("calibration sequence exceeds index_topk")
    ids = calibration_ids(split, vocab_size=int(config["vocab_size"]), tokens=tokens)
    membership = membership_sha256(ids, split)

    previous = _previous_capsule(layers[0], out_dir)
    if previous is not None:
        prior = np.load(out_dir / f"{previous['capsule_id']}.npz")
        hidden = np.asarray(prior["carry_out_hidden"], dtype=np.float32)
        previous_topk = np.asarray(prior["carry_out_index_selection"], dtype=np.int32)
        provenance = "CHAINED_FROM_PREVIOUS_CAPSULE"
        chain_gap = []
        if previous.get("calibration_membership_sha256") != membership:
            raise TeacherCaptureError(
                "previous capsule was captured on a different calibration membership"
            )
    else:
        hidden = np.asarray(source.rows("model.embed_tokens.weight", ids), dtype=np.float32)
        # A short probe never exceeds index_topk, so "all previous keys" is not an
        # approximation of the IndexShare selection -- it is exactly what any full
        # indexer layer would return at this length.
        previous_topk = np.tile(np.arange(tokens, dtype=np.int32), (1, tokens, 1))
        chain_gap = list(range(layers[0]))
        # Layer zero's real input IS the embedding, so a run starting there has
        # no gap to declare.  Anything deeper is honestly off-distribution.
        provenance = "EMBEDDED_INPUT_EXACT" if not chain_gap else "EMBEDDING_SEEDED_NOT_CHAINED"

    cache = reference.ReferenceCache()
    arrays: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for layer in layers:
        hidden, previous_topk, layer_arrays = capture_layer(
            hidden, source, layer, config, previous_topk, cache
        )
        metrics[f"layer_{layer:02d}"] = _layer_metrics(layer_arrays)
        for key, value in layer_arrays.items():
            arrays[f"layer_{layer:02d}/{key}"] = value

    arrays["carry_out_hidden"] = np.asarray(hidden, dtype=np.float32)
    arrays["carry_out_index_selection"] = np.asarray(previous_topk, dtype=np.int32)
    arrays["calibration_token_ids"] = np.asarray(ids, dtype=np.int64)
    arrays.update({
        f"logit_lens/{key}": value for key, value in _logit_lens(source, hidden, config).items()
    })

    payload = _npz_bytes(arrays)
    capsule_path = out_dir / f"{identity}.npz"
    atomic_bytes(capsule_path, payload)

    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {"files": []}
    by_path = {row["path"]: row for row in manifest.get("files", [])}
    receipt = seal({
        "schema": SCHEMA_CAPSULE,
        "capsule_id": identity,
        "window_ids": window_ids,
        "layers": layers,
        "first_layer": layers[0],
        "last_layer": layers[-1],
        "repo": graph["repo"],
        "revision": graph["revision"],
        "graph_seal_sha256": graph["seal_sha256"],
        "schedule_seal_sha256": schedule["seal_sha256"],
        "reference_commit": _git_head(),
        "source_shards": sorted(
            shard for layer in layers for shard in shards.get(f"text_layer_{layer:02d}", set())
        ),
        "source_shard_sha256": {
            shard: by_path[shard]["lfs_sha256"]
            for layer in layers
            for shard in sorted(shards.get(f"text_layer_{layer:02d}", set()))
            if shard in by_path
        },
        "calibration_split": split,
        "calibration_tokens": tokens,
        "calibration_vocab_size": int(config["vocab_size"]),
        "calibration_membership_sha256": membership,
        "input_provenance": provenance,
        "chain_gap_layers": chain_gap,
        "index_selection_trivially_complete": tokens <= int(config["index_topk"]),
        "capsule_path": str(capsule_path),
        "capsule_bytes": len(payload),
        "capsule_sha256": hashlib.sha256(payload).hexdigest(),
        "array_sha256": {name: _array_sha256(value) for name, value in sorted(arrays.items())},
        "metrics": metrics,
        "payload_bytes_read": source.payload_bytes_read,
        "bounded_read_calls": source.read_calls,
        "capture_seconds": round(time.time() - started, 2),
        "captured_at": utc_now(),
        "capability_claim_permitted": False,
        "note": "Real BF16 block evidence on a sealed synthetic token-id probe. "
                "Not natural text, not an evaluation, not a capability result.",
    })
    atomic_json(out_dir / f"{identity}.json", receipt)
    _append_ledger({
        "event": "TEACHER_CAPTURED",
        "capsule_id": identity,
        "window_ids": window_ids,
        "layers": layers,
        "revision": receipt["revision"],
        "source_shards": receipt["source_shards"],
        "calibration_split": split,
        "calibration_membership_sha256": membership,
        "capsule_path": str(capsule_path),
        "capsule_bytes": receipt["capsule_bytes"],
        "capsule_sha256": receipt["capsule_sha256"],
        "seal_sha256": receipt["seal_sha256"],
        "metrics": metrics,
        "input_provenance": provenance,
        "eviction_authorized_shards": receipt["source_shards"],
        "at": receipt["captured_at"],
    })
    return receipt


def _git_head() -> str:
    head = ROOT / ".git" / "HEAD"
    try:
        value = head.read_text().strip()
    except OSError:
        return "UNKNOWN"
    if value.startswith("ref: "):
        ref = ROOT / ".git" / value[5:]
        try:
            return ref.read_text().strip()
        except OSError:
            packed = ROOT / ".git" / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + value[5:]):
                        return line.split(" ", 1)[0]
            return "UNKNOWN"
    return value


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #


def capture_window(
    window_id: str,
    *,
    schedule: dict[str, Any] | None = None,
    graph: dict[str, Any] | None = None,
    source_root: Path | None = None,
    capsule_dir: Path | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Capture every contiguous fully-resident layer run of one window."""
    schedule = schedule if schedule is not None else _schedule()
    graph = graph if graph is not None else _graph()
    root = Path(source_root) if source_root is not None else SOURCE_ROOT
    window = next(
        (item for item in schedule["windows"] if item["window_id"] == window_id), None
    )
    if window is None:
        raise TeacherCaptureError(f"unknown window: {window_id!r}")
    shards_of = organ_shards(graph)
    resident = [
        layer
        for organ in window["organ_ids"]
        if (layer := layer_of_organ(organ)) is not None
        and all((root / shard).exists() for shard in shards_of.get(organ, set()))
    ]
    if not resident:
        raise TeacherCaptureError(f"{window_id} has no fully resident text layer")
    return [
        capture_layers(
            run, schedule=schedule, graph=graph, source_root=root,
            capsule_dir=capsule_dir, **kwargs,
        )
        for run in contiguous_runs(resident)
    ]


def verify_capsule(identity: str, *, capsule_dir: Path | None = None) -> dict[str, Any]:
    """Reload a capsule in this process and reproduce every declared metric.

    Fail-closed on: broken seal, capsule hash drift, missing or extra array,
    array hash drift, metric drift, lineage drift, membership drift.
    """
    out_dir = Path(capsule_dir) if capsule_dir is not None else CAPSULES
    receipt_path = out_dir / f"{identity}.json"
    if not receipt_path.exists():
        raise TeacherCaptureError(f"no capsule receipt for {identity}")
    receipt = verify_sealed(json.loads(receipt_path.read_text()), label=str(receipt_path))
    capsule_path = out_dir / f"{identity}.npz"
    payload = capsule_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != receipt["capsule_sha256"]:
        raise TeacherCaptureError(f"{identity} capsule bytes do not match the receipt")

    graph = _graph()
    if graph["revision"] != receipt["revision"] or graph["seal_sha256"] != receipt[
        "graph_seal_sha256"
    ]:
        raise TeacherCaptureError(f"{identity} lineage mismatch against the sealed graph")
    ids_expected = calibration_ids(
        receipt["calibration_split"],
        vocab_size=int(receipt["calibration_vocab_size"]),
        tokens=int(receipt["calibration_tokens"]),
    )
    if membership_sha256(ids_expected, receipt["calibration_split"]) != receipt[
        "calibration_membership_sha256"
    ]:
        raise TeacherCaptureError(f"{identity} calibration membership mismatch")

    with np.load(io.BytesIO(payload)) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    declared = receipt["array_sha256"]
    if set(arrays) != set(declared):
        raise TeacherCaptureError(
            f"{identity} array set differs from the receipt: "
            f"missing={sorted(set(declared) - set(arrays))[:3]} "
            f"unexpected={sorted(set(arrays) - set(declared))[:3]}"
        )
    for name, digest in declared.items():
        if _array_sha256(arrays[name]) != digest:
            raise TeacherCaptureError(f"{identity} array hash drift: {name}")
    if not np.array_equal(arrays["calibration_token_ids"], ids_expected):
        raise TeacherCaptureError(f"{identity} stored calibration ids differ from the split")

    reproduced: dict[str, dict[str, float]] = {}
    for layer in receipt["layers"]:
        prefix = f"layer_{layer:02d}/"
        subset = {
            name[len(prefix):]: value
            for name, value in arrays.items()
            if name.startswith(prefix)
        }
        reproduced[f"layer_{layer:02d}"] = _layer_metrics(subset)
    if reproduced != receipt["metrics"]:
        raise TeacherCaptureError(f"{identity} metric reproduction mismatch")
    return {
        "capsule_id": identity,
        "window_ids": receipt["window_ids"],
        "status": "REPRODUCED",
        "layers": receipt["layers"],
        "capsule_sha256": receipt["capsule_sha256"],
        "arrays": len(arrays),
        "metrics": reproduced,
        "verified_at": utc_now(),
    }


def captured_layers(*, capsule_dir: Path | None = None) -> set[int]:
    """Layers with a sealed capsule.  Cheap: reads receipts, not capsules."""
    out_dir = Path(capsule_dir) if capsule_dir is not None else CAPSULES
    if not out_dir.exists():
        return set()
    layers: set[int] = set()
    for path in sorted(out_dir.glob("*.json")):
        try:
            receipt = verify_sealed(json.loads(path.read_text()), label=str(path))
        except (Glm52Error, json.JSONDecodeError, OSError):
            continue
        layers.update(int(value) for value in receipt.get("layers", []))
    return layers


def sealed_capsules(*, capsule_dir: Path | None = None) -> list[dict[str, Any]]:
    out_dir = Path(capsule_dir) if capsule_dir is not None else CAPSULES
    if not out_dir.exists():
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            found.append(verify_sealed(json.loads(path.read_text()), label=str(path)))
        except (Glm52Error, json.JSONDecodeError, OSError):
            continue
    return found


# --------------------------------------------------------------------------- #
# eviction authority
# --------------------------------------------------------------------------- #


def ever_verified_shards(ledger: Path | None = None) -> set[str]:
    """Shards that once held a VERIFIED body, whether or not they still do."""
    path = Path(ledger) if ledger is not None else FETCH_LEDGER
    if not path.exists():
        return set()
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "VERIFIED" and isinstance(row.get("shard"), str):
            seen.add(row["shard"])
    return seen


def eviction_authority(
    shards: Iterable[str],
    *,
    source_root: Path | None = None,
    graph: dict[str, Any] | None = None,
    capsule_dir: Path | None = None,
    ever_verified: set[str] | None = None,
) -> dict[str, Any]:
    """Split candidate shards into authorized, refused and already-unrecoverable.

    The invariant is not "everything must be captured" -- layers whose bodies
    were already destroyed under the pre-capture policy cannot be captured at any
    price short of a refetch, and refusing to evict them costs disk without
    saving evidence.  What is refused is what still could be captured: a layer
    fully resident right now, and equally a layer that is merely incomplete
    because the rest of it has not been fetched yet.  Only a layer every one of
    whose shards was once VERIFIED and is now partly gone counts as lost.
    """
    root = Path(source_root) if source_root is not None else SOURCE_ROOT
    graph = graph if graph is not None else _graph()
    fetched = ever_verified_shards() if ever_verified is None else set(ever_verified)
    by_shard = organs_by_shard(graph)
    shards_of = organ_shards(graph)
    captured = captured_layers(capsule_dir=capsule_dir)

    authorized: list[str] = []
    refused: dict[str, list[str]] = {}
    incomplete: dict[str, list[str]] = {}
    unrecoverable: dict[str, list[str]] = {}
    for shard in sorted(set(shards)):
        blocking: list[str] = []
        pending: list[str] = []
        lost: list[str] = []
        for organ in sorted(by_shard.get(shard, set())):
            layer = layer_of_organ(organ)
            if layer is None or layer in captured:
                continue
            if all((root / name).exists() for name in shards_of[organ]):
                blocking.append(organ)
            elif all(name in fetched for name in shards_of[organ]):
                lost.append(organ)
            else:
                pending.append(organ)
        if blocking:
            refused[shard] = blocking
        elif pending:
            incomplete[shard] = pending
        else:
            authorized.append(shard)
            if lost:
                unrecoverable[shard] = lost
    return {
        "authorized": authorized,
        "refused_uncaptured_but_capturable": refused,
        "refused_incomplete_organs": incomplete,
        "authorized_with_unrecoverable_organs": unrecoverable,
    }


def layers_capturable_now(
    *,
    source_root: Path | None = None,
    graph: dict[str, Any] | None = None,
    capsule_dir: Path | None = None,
) -> list[int]:
    """Text layers fully resident on disk and not yet sealed in a capsule."""
    root = Path(source_root) if source_root is not None else SOURCE_ROOT
    graph = graph if graph is not None else _graph()
    captured = captured_layers(capsule_dir=capsule_dir)
    ready: list[int] = []
    for organ, shards in organ_shards(graph).items():
        layer = layer_of_organ(organ)
        if layer is None or layer in captured:
            continue
        if all((root / shard).exists() for shard in shards):
            ready.append(layer)
    return sorted(ready)


def ensure_captured(
    layers: Iterable[int],
    *,
    source_root: Path | None = None,
    capsule_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    """Capture each contiguous layer run; never raise into the streaming loop."""
    outcome: dict[str, str] = {}
    for run in contiguous_runs(layers):
        identity = capsule_id(run)
        try:
            capture_layers(
                run, source_root=source_root, capsule_dir=capsule_dir, **kwargs
            )
            verify_capsule(identity, capsule_dir=capsule_dir)
            outcome[identity] = "CAPTURED"
        except Exception as exc:  # noqa: BLE001 - a capture failure must not kill the stream
            outcome[identity] = f"{type(exc).__name__}: {exc}"
            _append_ledger({
                "event": "TEACHER_CAPTURE_FAILED",
                "capsule_id": identity,
                "layers": run,
                "error": outcome[identity],
                "at": utc_now(),
            })
    return outcome


def capture_for_eviction(
    shards: Iterable[str],
    *,
    source_root: Path | None = None,
    graph: dict[str, Any] | None = None,
    capsule_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Capture what these shards still owe, then re-decide eviction authority.

    This is the whole gate in one call: the streaming loop hands over its
    eviction candidates and gets back only the shards whose teacher evidence is
    already durable.
    """
    graph = graph if graph is not None else _graph()
    authority = eviction_authority(
        shards, source_root=source_root, graph=graph, capsule_dir=capsule_dir
    )
    blocking = {
        organ
        for organs in authority["refused_uncaptured_but_capturable"].values()
        for organ in organs
    }
    if not blocking:
        return authority
    wanted = sorted(
        layer for organ in blocking if (layer := layer_of_organ(organ)) is not None
    )
    outcome = ensure_captured(
        wanted, source_root=source_root, capsule_dir=capsule_dir, **kwargs
    )
    authority = eviction_authority(
        shards, source_root=source_root, graph=graph, capsule_dir=capsule_dir
    )
    authority["capture_outcome"] = outcome
    return authority


# --------------------------------------------------------------------------- #
# policy + CLI
# --------------------------------------------------------------------------- #


def write_policy() -> dict[str, Any]:
    policy = seal({
        "schema": SCHEMA_POLICY,
        "status": "ARMED",
        "repo": "zai-org/GLM-5.2",
        "revision": IMMUTABLE_REVISION,
        "pipeline": ["VERIFY", "TEACHER_CAPTURE", "PROBE", "PACK", "SEAL", "EVICT"],
        "invariant": "A shard may be evicted only when every text-layer organ it "
                     "carries is either sealed in a teacher capsule or already "
                     "unrecoverable because its sibling shards were destroyed "
                     "under the pre-capture policy.",
        "ledger_path": str(LEDGER),
        "capsule_directory": str(CAPSULES),
        "calibration": {
            "kind": "sealed deterministic synthetic token-id probe",
            "natural_text": False,
            "tokens": CALIBRATION_TOKENS,
            "splits": list(CALIBRATION_SPLITS),
            "disjoint_by_construction": True,
            "membership_hash": "sha256 over canonical {split, token_ids}",
        },
        "captured_evidence": [
            "input hidden state", "attention input", "IndexShare selection",
            "indexer scores", "attention output", "post-attention residual",
            "pre-router hidden state", "router logits", "top-8 indices",
            "top-8 weights", "8th-versus-9th margin", "shared-expert output",
            "weighted routed-expert output", "post-MoE state",
            "post-block residual", "logit-lens short logits over a sealed "
            "vocabulary subset",
        ],
        "bounds": {
            "no_unrestricted_activation_corpus": True,
            "max_single_tensor_bytes": MAX_TENSOR_BYTES,
            "logit_lens_vocabulary_rows": LOGIT_LENS_ROWS,
            "batch": 1,
        },
        "limitations": [
            "The calibration probe is synthetic token ids, not natural text, so "
            "routing statistics are not the natural-text routing distribution.",
            "A window captured before its upstream layers were resident is seeded "
            "from the embedding table; its chain gap is recorded per capsule.",
            "Capsules are teacher evidence for representation fitting. They are "
            "not an evaluation and permit no capability claim.",
        ],
        "capability_claim_permitted": False,
    })
    atomic_json(POLICY_PATH, policy)
    return policy


def recent_ledger(*, limit: int = 20) -> list[dict[str, Any]]:
    """The tail of the evidence ledger, for callers that watch for failures."""
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text().splitlines()[-limit:]:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def status() -> dict[str, Any]:
    receipts = [
        {
            "capsule_id": receipt["capsule_id"],
            "window_ids": receipt["window_ids"],
            "layers": receipt["layers"],
            "capsule_bytes": receipt["capsule_bytes"],
            "capsule_sha256": receipt["capsule_sha256"],
            "input_provenance": receipt["input_provenance"],
            "captured_at": receipt["captured_at"],
        }
        for receipt in sealed_capsules()
    ]
    rows = 0
    if LEDGER.exists():
        rows = sum(1 for line in LEDGER.read_text().splitlines() if line.strip())
    return {
        "capsules": receipts,
        "capsule_count": len(receipts),
        "capsule_bytes_total": sum(row["capsule_bytes"] for row in receipts),
        "captured_layers": sorted(captured_layers()),
        "latest_capsule_seal": max(
            (row["captured_at"] for row in receipts), default=None
        ),
        "ledger_rows": rows,
        "ledger_path": str(LEDGER),
        "capturable_now": layers_capturable_now() if SOURCE_ROOT.exists() else [],
        "policy_sealed": POLICY_PATH.exists(),
        "at": utc_now(),
    }


def selftest() -> dict[str, Any]:
    """No network, no source, no writes: the pure invariants only."""
    ids_fit = calibration_ids("teacher_fit", vocab_size=154_880)
    ids_hold = calibration_ids("teacher_holdout", vocab_size=154_880)
    if np.array_equal(ids_fit, ids_hold):
        raise AssertionError("calibration splits are not disjoint")
    if not np.array_equal(ids_fit, calibration_ids("teacher_fit", vocab_size=154_880)):
        raise AssertionError("calibration ids are not deterministic")
    logits = np.asarray([[[3.0, 2.0, 1.0, 0.0]]], dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32)
    margin = _router_margin(logits, bias, 2)
    scores = 1.0 / (1.0 + np.exp(-logits))
    if not np.isclose(margin[0, 0], scores[0, 0, 1] - scores[0, 0, 2]):
        raise AssertionError("8th-versus-9th margin is not the selection boundary gap")
    return {
        "status": "PASS",
        "splits_disjoint": True,
        "deterministic_ids": True,
        "margin_is_boundary_gap": True,
        "calibration_tokens": CALIBRATION_TOKENS,
    }


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "status"
    if command == "capture":
        target = argv[2]
        receipts = (
            capture_window(target)
            if target.startswith("W")
            else [capture_layers([int(value) for value in target.split(",")])]
        )
        print(json.dumps(
            [{key: value for key, value in receipt.items() if key != "array_sha256"}
             for receipt in receipts],
            indent=2, sort_keys=True,
        ))
    elif command == "verify":
        print(json.dumps(verify_capsule(argv[2]), indent=2, sort_keys=True))
    elif command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
    elif command == "policy":
        print(json.dumps(write_policy(), indent=2, sort_keys=True))
    elif command == "selftest":
        print(json.dumps(selftest(), indent=2, sort_keys=True))
    elif command == "authority":
        print(json.dumps(eviction_authority(argv[2:]), indent=2, sort_keys=True))
    else:
        raise SystemExit(f"unknown command: {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
