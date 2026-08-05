#!/usr/bin/env python3.12
"""Teacher-forced, layer-major GLM-5.2 activation executor (PROTO_FRANKENSTEIN V0).

This is NOT a chat/decode server.  It freezes a batch of sequences, streams one
GLM layer at a time over ALL sequences, captures bounded hidden-state evidence,
atomically seals next-layer carry states, and evicts that layer's weights before
advancing.  Double-buffer accounting: N-1 seal/evict, N execute, N+1 prefetch.

Reuse:
  - ``glm52_reference.decoder_layer`` for exact norms / MLA+DSA attention / MoE
  - ``glm52_adapter`` safetensors BF16 path + inventory validation
  - ``glm52_teacher_capture`` bounded layer arrays + router margin helpers
  - streaming schedule organs for official shard grouping (when present)
  - ``frankenstein_trace_format`` paired-trace schema for GLM-side emission

Honesty: never fabricates activations.  Official 78-layer body requires resident
source shards; when absent the executor fail-closes and reports the deepest
verified layer (synthetic full-stack proves mechanism end-to-end).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from lab.layout import EVIDENCE_ROOT, REPO_ROOT
from lab.operators import glm52_reference as reference
from lab.operators.frankenstein_trace_format import (
    MEMBERSHIP_SPLITS,
    MembershipManager,
    build_paired_trace,
    empty_side,
    index_corpus,
    make_decoded_span,
    make_route_statistics,
)
from lab.operators.glm52_adapter import (
    IMMUTABLE_REVISION,
    OFFICIAL_INDEXER_TYPES,
    PROFILE_OFFICIAL,
    PROFILE_SYNTHETIC,
    REPO_ID,
    BoundedSafetensorsReader,
    Inventory,
    load_json_strict,
    validate_config,
    verify_checkpoint,
)
from lab.operators.glm52_common import (
    Glm52Error,
    atomic_bytes,
    atomic_json,
    canonical,
    seal,
    sha256_file,
    utc_now,
    verify_sealed,
)
from lab.operators.glm52_layer_stream import (
    DEFAULT_CONTROL_ROOT,
    DEFAULT_STREAM_ROOT,
    LayerMajorStreamer,
    LayerStreamError,
    ensure_control_assets,
)
# Intentionally do NOT import glm52_teacher_capture at module load: that module
# resolves sealed campaign graph paths at import time and fails closed when the
# artifact tree is absent.  Bounded helpers below are self-contained copies of
# the same pure arithmetic used by teacher_capture.


SCHEMA_RECEIPT = "hawking.frankenstein.glm_teacher_forced_capture.v1"
SCHEMA_LAYER_SHARD = "hawking.frankenstein.glm_layer_capture_shard.v1"
SCHEMA_CORPUS = "hawking.frankenstein.glm_frozen_corpus.v1"
SCHEMA_EVICTION = "hawking.frankenstein.glm_layer_eviction.v1"

MIN_FREE_FLOOR_BYTES = 25 * 1024**3
CORPUS_LEVELS: dict[str, int] = {
    "L0": 32,
    "L1": 128,
    "L2": 256,
    "L3": 512,
}

# Which layers get full sample dumps of pre/post attn/MoE (steer: early/mid/late).
SAMPLE_TOKEN_SLOTS = ("first", "mid", "last")
DEFAULT_MICROBATCH = 8
DEFAULT_MAX_SEQ = 64
DEFAULT_SAMPLE_HIDDEN = 64  # store first N dims of hidden samples (bounded)

FRANK_EVIDENCE = EVIDENCE_ROOT / "models" / "frankenstein"
DEFAULT_OUT = FRANK_EVIDENCE / "teacher_forced"


class TeacherForcedError(Glm52Error):
    """Fail-closed teacher-forced capture error."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _finite(value: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _router_margin(
    router_logits: np.ndarray, correction_bias: np.ndarray, top_k: int
) -> np.ndarray:
    logits = np.asarray(router_logits, dtype=np.float32)
    scores = np.float32(1.0) / (np.float32(1.0) + np.exp(-logits, dtype=np.float32))
    corrected = scores + np.asarray(correction_bias, dtype=np.float32)
    ordered = np.sort(corrected, axis=-1)[..., ::-1]
    if ordered.shape[-1] <= top_k:
        return np.zeros(ordered.shape[:-1], dtype=np.float32)
    return (ordered[..., top_k - 1] - ordered[..., top_k]).astype(np.float32)


def _layer_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    output = _finite(arrays["block_output"])
    metrics = {
        "block_output_l2": float(np.sqrt(np.sum(output * output, dtype=np.float64))),
        "block_output_absmax": float(np.max(np.abs(output))),
        "block_output_mean": float(np.mean(output, dtype=np.float64)),
        "attention_output_l2": float(
            np.sqrt(np.sum(_finite(arrays["attention_output"]) ** 2, dtype=np.float64))
        ),
        "post_moe_l2": float(
            np.sqrt(np.sum(_finite(arrays["post_moe"]) ** 2, dtype=np.float64))
        ),
    }
    if "router_logits" in arrays:
        logits = _finite(arrays["router_logits"])
        margin = _finite(arrays["topk_margin_8th_vs_9th"])
        metrics.update(
            {
                "router_logit_absmax": float(np.max(np.abs(logits))),
                "router_selected_expert_count": float(
                    len(set(np.asarray(arrays["topk_indices"]).ravel().tolist()))
                ),
                "topk_margin_mean": float(np.mean(margin, dtype=np.float64)),
                "topk_margin_min": float(np.min(margin)),
                "shared_output_l2": float(
                    np.sqrt(
                        np.sum(
                            _finite(arrays["shared_expert_output"]) ** 2, dtype=np.float64
                        )
                    )
                ),
                "routed_output_l2": float(
                    np.sqrt(
                        np.sum(
                            _finite(arrays["routed_expert_output"]) ** 2, dtype=np.float64
                        )
                    )
                ),
            }
        )
    if "expert_contribution_l2" in arrays:
        contribution = _finite(arrays["expert_contribution_l2"])
        hit = np.asarray(arrays["expert_hit_count"])
        hit_experts = hit > 0
        metrics.update(
            {
                "experts_hit_this_batch": float(np.count_nonzero(hit_experts)),
                "expert_contribution_l2_max": float(np.max(contribution))
                if hit_experts.any()
                else 0.0,
                "expert_contribution_l2_mean_over_hit": float(
                    np.mean(contribution[hit_experts])
                )
                if hit_experts.any()
                else 0.0,
            }
        )
    return {key: value for key, value in sorted(metrics.items())}


def _expert_cartography_arrays(
    per_expert: dict[int, dict[str, np.ndarray]], n_routed_experts: int
) -> dict[str, np.ndarray]:
    contribution_l2 = np.zeros(n_routed_experts, dtype=np.float32)
    hit_count = np.zeros(n_routed_experts, dtype=np.int32)
    coselection_count = np.zeros((n_routed_experts, n_routed_experts), dtype=np.int32)
    token_experts: dict[int, list[int]] = {}
    for expert, data in per_expert.items():
        weighted = np.asarray(data["weighted_output"], dtype=np.float32)
        contribution_l2[expert] = float(
            np.sqrt(np.sum(weighted.astype(np.float64) ** 2))
        )
        tokens = np.asarray(data["tokens"]).ravel().tolist()
        hit_count[expert] = len(tokens)
        for token in tokens:
            token_experts.setdefault(int(token), []).append(expert)
    for experts in token_experts.values():
        for e1 in experts:
            for e2 in experts:
                coselection_count[e1, e2] += 1
    return {
        "expert_contribution_l2": contribution_l2,
        "expert_hit_count": hit_count,
        "expert_coselection_count": coselection_count,
    }


# ---------------------------------------------------------------------------
# Disk floor
# ---------------------------------------------------------------------------


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    return int(usage.free)


def assert_floor(path: Path, *, label: str = "workspace") -> dict[str, Any]:
    free = free_bytes(path)
    ok = free >= MIN_FREE_FLOOR_BYTES
    record = {
        "label": label,
        "path": str(path),
        "free_bytes": free,
        "floor_bytes": MIN_FREE_FLOOR_BYTES,
        "floor_preserved": ok,
        "headroom_bytes": free - MIN_FREE_FLOOR_BYTES,
    }
    if not ok:
        raise TeacherForcedError(
            f"25 GiB floor breached under {label}: free={free} floor={MIN_FREE_FLOOR_BYTES}"
        )
    return record


# ---------------------------------------------------------------------------
# Frozen corpus
# ---------------------------------------------------------------------------


@dataclass
class FrozenSequence:
    example_id: str
    membership: str
    prompt_text: str
    token_ids: list[int]
    domain: str = "general"
    token_ids_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.token_ids_sha256:
            self.token_ids_sha256 = hashlib.sha256(
                canonical({"token_ids": list(self.token_ids)})
            ).hexdigest()


@dataclass
class FrozenCorpus:
    level: str
    sequences: list[FrozenSequence]
    membership: MembershipManager
    pad_id: int
    max_sequence: int
    revision: str
    source: str
    seal_sha256: str = ""

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)

    def membership_sha256(self) -> str:
        payload = {
            "level": self.level,
            "assignments": dict(sorted(self.membership.assignments.items())),
            "token_ids_sha256": [s.token_ids_sha256 for s in self.sequences],
        }
        return hashlib.sha256(canonical(payload)).hexdigest()

    def batch_ids(self) -> np.ndarray:
        width = max((len(s.token_ids) for s in self.sequences), default=0)
        width = min(width, self.max_sequence) if width else 0
        batch = np.full((self.n_sequences, width), self.pad_id, dtype=np.int64)
        for i, seq in enumerate(self.sequences):
            ids = seq.token_ids[: self.max_sequence]
            batch[i, : len(ids)] = ids
        return batch

    def attention_lengths(self) -> np.ndarray:
        return np.asarray(
            [min(len(s.token_ids), self.max_sequence) for s in self.sequences],
            dtype=np.int64,
        )

    def document(self) -> dict[str, Any]:
        doc = {
            "schema": SCHEMA_CORPUS,
            "level": self.level,
            "n_sequences": self.n_sequences,
            "target_n": CORPUS_LEVELS[self.level],
            "pad_id": self.pad_id,
            "max_sequence": self.max_sequence,
            "revision": self.revision,
            "source": self.source,
            "membership_sha256": self.membership_sha256(),
            "membership": self.membership.seal_document(),
            "sequences": [
                {
                    "example_id": s.example_id,
                    "membership": s.membership,
                    "domain": s.domain,
                    "n_tokens": len(s.token_ids),
                    "token_ids_sha256": s.token_ids_sha256,
                    "prompt_text_sha256": hashlib.sha256(
                        s.prompt_text.encode("utf-8")
                    ).hexdigest(),
                    "prompt_text": s.prompt_text,
                }
                for s in self.sequences
            ],
            "fabricated": False,
            "frozen_at": utc_now(),
        }
        sealed = seal(doc)
        self.seal_sha256 = sealed["seal_sha256"]
        return sealed


def _synthetic_prompts(n: int) -> list[tuple[str, str, str]]:
    """(example_id, membership, text) for offline synthetic L0/L1 smoke."""
    domains = [
        "algebra",
        "geometry",
        "combinatorics",
        "formal",
        "coding",
        "tools",
        "general",
        "repair",
    ]
    splits = list(MEMBERSHIP_SPLITS)
    rows: list[tuple[str, str, str]] = []
    for i in range(n):
        domain = domains[i % len(domains)]
        # Keep train dominant; leave room for calib/public/hidden.
        membership = splits[0] if i < int(n * 0.7) else splits[1 + (i % 3)]
        text = (
            f"[{domain}] Prove or solve step {i}: "
            f"let n={i + 3}. Select a method, decompose, and answer with certainty."
        )
        rows.append((f"tf_syn_{i:04d}", membership, text))
    return rows


def freeze_corpus(
    *,
    level: str,
    mode: str,
    max_sequence: int = DEFAULT_MAX_SEQ,
    vocab_size: int | None = None,
    pad_id: int = 0,
    seed: int = 0,
) -> FrozenCorpus:
    if level not in CORPUS_LEVELS:
        raise TeacherForcedError(f"unknown corpus level {level!r}; expected {list(CORPUS_LEVELS)}")
    n = CORPUS_LEVELS[level]
    mgr = MembershipManager()
    sequences: list[FrozenSequence] = []

    if mode == "synthetic":
        if vocab_size is None:
            vocab_size = 64
        for example_id, membership, text in _synthetic_prompts(n):
            mgr.assign(example_id, membership)
            # Deterministic tokenisation into the miniature vocab (not HF tokenizer).
            digest = hashlib.sha256(
                canonical({"seed": seed, "example_id": example_id, "text": text})
            ).digest()
            length = 8 + (digest[0] % max(1, min(16, max_sequence - 8)))
            ids = [
                1 + (digest[j % len(digest)] + j * 17) % max(1, vocab_size - 2)
                for j in range(length)
            ]
            sequences.append(
                FrozenSequence(
                    example_id=example_id,
                    membership=membership,
                    prompt_text=text,
                    token_ids=ids,
                    domain=text.split("]")[0].strip("["),
                )
            )
        source = "SYNTHETIC_DETERMINISTIC_PROBE"
        revision = "synthetic-revision"
    elif mode == "official":
        # Prefer real corpus partition when the pinned tokenizer is available.
        try:
            from lab.operators import glm52_capture_program as program
            from lab.operators import glm52_corpus as corpus

            records = list(corpus.build_records(corpus.load_pinned_tokenizer()))
            records.sort(key=lambda r: r.record_id)
            if len(records) < n:
                raise TeacherForcedError(
                    f"corpus has only {len(records)} records; need {n} for {level}"
                )
            bundle = corpus.load_pinned_tokenizer()
            chosen = records[:n]
            for idx, record in enumerate(chosen):
                membership = (
                    "train"
                    if idx < int(n * 0.7)
                    else list(MEMBERSHIP_SPLITS)[1 + (idx % 3)]
                )
                example_id = f"tf_corp_{record.record_id}"
                mgr.assign(example_id, membership)
                text = (
                    record.context_window
                    if getattr(record, "context_rung_tokens", None)
                    else record.prompt
                )
                ids = list(corpus._encode(bundle, text))[:max_sequence]
                if not ids:
                    raise TeacherForcedError(f"empty encoding for {record.record_id}")
                sequences.append(
                    FrozenSequence(
                        example_id=example_id,
                        membership=membership,
                        prompt_text=text if isinstance(text, str) else str(text),
                        token_ids=ids,
                        domain=str(getattr(record, "domain", "general")),
                    )
                )
            source = "NATURAL_CORPUS_PARTITION"
            revision = IMMUTABLE_REVISION
        except Exception as exc:  # noqa: BLE001 — fail closed for official path
            raise TeacherForcedError(
                f"official corpus freeze failed (fail closed): {exc}"
            ) from exc
    else:
        raise TeacherForcedError(f"unknown mode {mode!r}")

    return FrozenCorpus(
        level=level,
        sequences=sequences,
        membership=mgr,
        pad_id=pad_id,
        max_sequence=max_sequence,
        revision=revision,
        source=source,
    )


# ---------------------------------------------------------------------------
# Weight residency + layer-scoped source + eviction
# ---------------------------------------------------------------------------


@dataclass
class LayerWeightPlan:
    """Which shards own tensors for each layer / global organ."""

    root: Path
    inventory: Inventory
    layer_to_shards: dict[int, set[str]]
    global_shards: set[str]
    shard_to_layers: dict[str, set[int]]
    shard_hashes: dict[str, str] = field(default_factory=dict)

    def shards_for_layer(self, layer: int, *, include_global: bool = False) -> set[str]:
        names = set(self.layer_to_shards.get(layer, set()))
        if include_global:
            names |= set(self.global_shards)
        return names

    def resident_shards(self) -> set[str]:
        # Prefer live disk scan so streamed/evicted shards stay accurate even
        # when the inventory is only a resident subset.
        on_disk = {
            p.name
            for p in self.root.glob("model-*.safetensors")
            if p.is_file()
        }
        if on_disk:
            return on_disk
        return {
            name
            for name in self.shard_to_layers
            if (self.root / name).is_file()
        }

    def missing_for_layer(self, layer: int, *, include_global: bool = False) -> list[str]:
        return sorted(
            name
            for name in self.shards_for_layer(layer, include_global=include_global)
            if not (self.root / name).is_file()
        )

    def verify_window_hashes(self, shards: Iterable[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in sorted(shards):
            path = self.root / name
            if not path.is_file():
                raise TeacherForcedError(f"cannot hash absent shard {name}")
            digest = sha256_file(path)
            prior = self.shard_hashes.get(name)
            if prior is not None and prior != digest:
                raise TeacherForcedError(
                    f"shard hash drift for {name}: prior={prior} now={digest}"
                )
            self.shard_hashes[name] = digest
            out[name] = digest
        return out


def build_weight_plan(root: Path, inventory: Inventory) -> LayerWeightPlan:
    layer_to_shards: dict[int, set[str]] = {}
    global_shards: set[str] = set()
    shard_to_layers: dict[str, set[int]] = {}
    for name, record in inventory.tensors.items():
        shard = record.shard
        shard_to_layers.setdefault(shard, set())
        layer = record.spec.layer
        if layer is None:
            global_shards.add(shard)
            continue
        # MTP physical layer is beyond main num_hidden_layers; track separately
        # only if it appears.  Main path uses 0..num_hidden_layers-1.
        layer_to_shards.setdefault(int(layer), set()).add(shard)
        shard_to_layers[shard].add(int(layer))
    return LayerWeightPlan(
        root=Path(root),
        inventory=inventory,
        layer_to_shards=layer_to_shards,
        global_shards=global_shards,
        shard_to_layers=shard_to_layers,
    )


class LayerScopedSource:
    """TensorSource that refuses tensors outside the admitted resident set.

    Loads via the validated inventory's bounded BF16→f32 path.  Embedding/lm_head
    row reads avoid materialising full tables when possible.
    """

    def __init__(
        self,
        plan: LayerWeightPlan,
        *,
        admitted_shards: set[str],
        max_tensor_bytes: int = 320 * 1024 * 1024,
    ) -> None:
        self.plan = plan
        self.admitted_shards = set(admitted_shards)
        self.max_tensor_bytes = int(max_tensor_bytes)
        self.reader = BoundedSafetensorsReader(
            plan.inventory, max_tensor_bytes=max_tensor_bytes
        )
        self.payload_bytes_read = 0
        self.read_calls = 0

    def resident(self, name: str) -> bool:
        record = self.plan.inventory.tensors.get(name)
        if record is None:
            return False
        return record.shard in self.admitted_shards and (
            self.plan.root / record.shard
        ).is_file()

    def tensor(self, name: str) -> np.ndarray:
        record = self.plan.inventory.tensors.get(name)
        if record is None:
            raise TeacherForcedError(f"tensor absent from inventory: {name!r}")
        if record.shard not in self.admitted_shards:
            raise TeacherForcedError(
                f"tensor {name!r} lives in non-admitted shard {record.shard}"
            )
        if not (self.plan.root / record.shard).is_file():
            raise TeacherForcedError(f"shard not resident for {name}: {record.shard}")
        if record.byte_count > self.max_tensor_bytes:
            raise TeacherForcedError(
                f"bounded read refused {name}: {record.byte_count} > {self.max_tensor_bytes}"
            )
        value = self.reader.tensor(name)
        self.payload_bytes_read = self.reader.payload_bytes_read
        self.read_calls = self.reader.read_calls
        return value

    def rows(self, name: str, ids: Iterable[int]) -> np.ndarray:
        """Row gather for 2-D tables (embed / lm_head)."""
        record = self.plan.inventory.tensors.get(name)
        if record is None:
            raise TeacherForcedError(f"tensor absent: {name!r}")
        if record.shard not in self.admitted_shards:
            raise TeacherForcedError(f"non-admitted shard for row read: {name}")
        shape = record.spec.shape
        if len(shape) != 2:
            raise TeacherForcedError(f"row read requires 2-D tensor: {name}")
        wanted = np.asarray(list(np.asarray(ids).reshape(-1)), dtype=np.int64)
        # Prefer full-table load when tiny (synthetic); else pread rows.
        if record.byte_count <= self.max_tensor_bytes:
            table = self.tensor(name)
            return table[wanted].reshape(*np.asarray(ids).shape, shape[1])
        # Bounded row pread path (official-scale tables).
        item = {"BF16": 2, "F32": 4}[record.spec.dtype]
        stride = shape[1] * item
        path = self.plan.root / record.shard
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            out = np.empty((wanted.size, shape[1]), dtype=np.float32)
            for slot, row in enumerate(wanted):
                if row < 0 or row >= shape[0]:
                    raise TeacherForcedError(f"row index out of range for {name}")
                payload = os.pread(
                    fd, stride, record.absolute_start + int(row) * stride
                )
                if len(payload) != stride:
                    raise TeacherForcedError(f"short row read for {name}")
                if record.spec.dtype == "BF16":
                    words = np.frombuffer(payload, dtype="<u2")
                    out[slot] = (words.astype(np.uint32) << np.uint32(16)).view(
                        np.float32
                    )
                else:
                    out[slot] = np.frombuffer(payload, dtype="<f4")
                self.payload_bytes_read += stride
                self.read_calls += 1
        finally:
            os.close(fd)
        return out.reshape(*np.asarray(ids).shape, shape[1])


def shards_evictable_after(
    plan: LayerWeightPlan,
    *,
    completed_layers: set[int],
    remaining_layers: set[int],
    keep_global_until_final: bool = True,
) -> list[str]:
    """Shards whose every layer is completed and not needed by remaining work."""
    evictable: list[str] = []
    for shard, layers in plan.shard_to_layers.items():
        if not layers:
            continue
        if keep_global_until_final and shard in plan.global_shards:
            # Global shards also appear in layer maps sometimes; only pure globals
            # are deferred.  If a shard serves only completed layers, still OK.
            pass
        if layers & remaining_layers:
            continue
        if not layers.issubset(completed_layers):
            continue
        if not (plan.root / shard).is_file():
            continue
        evictable.append(shard)
    return sorted(evictable)


def evict_shards(
    plan: LayerWeightPlan,
    shards: Sequence[str],
    *,
    require_hashes: bool = True,
) -> dict[str, Any]:
    """Physically remove admitted shards after capture seal.  Source-only reclaim."""
    removed: list[dict[str, Any]] = []
    for name in shards:
        path = plan.root / name
        if not path.is_file():
            continue
        digest = plan.shard_hashes.get(name)
        if require_hashes and digest is None:
            digest = sha256_file(path)
            plan.shard_hashes[name] = digest
        size = path.stat().st_size
        path.unlink()
        removed.append({"shard": name, "bytes": size, "sha256": digest})
    return seal(
        {
            "schema": SCHEMA_EVICTION,
            "removed": removed,
            "bytes_reclaimed": int(sum(r["bytes"] for r in removed)),
            "at": utc_now(),
            "policy": "source_only_reclaim_after_atomic_seal",
        }
    )


# ---------------------------------------------------------------------------
# Bounded capture
# ---------------------------------------------------------------------------


def _sample_positions(length: int) -> dict[str, int]:
    if length <= 0:
        return {slot: 0 for slot in SAMPLE_TOKEN_SLOTS}
    return {
        "first": 0,
        "mid": max(0, (length - 1) // 2),
        "last": max(0, length - 1),
    }


def _hidden_sample(hidden: np.ndarray, lengths: np.ndarray, width: int) -> dict[str, Any]:
    """Bounded per-sequence samples + sufficient statistics."""
    batch, seq, dim = hidden.shape
    width = min(width, dim)
    samples = np.zeros((batch, len(SAMPLE_TOKEN_SLOTS), width), dtype=np.float32)
    means = np.zeros((batch, dim), dtype=np.float32)
    vars_ = np.zeros((batch, dim), dtype=np.float32)
    l2 = np.zeros((batch,), dtype=np.float32)
    absmax = np.zeros((batch,), dtype=np.float32)
    for b in range(batch):
        L = int(lengths[b]) if b < len(lengths) else seq
        L = max(1, min(L, seq))
        slice_ = hidden[b, :L]
        means[b] = np.mean(slice_, axis=0)
        vars_[b] = np.var(slice_, axis=0)
        l2[b] = float(np.sqrt(np.sum(slice_.astype(np.float64) ** 2)))
        absmax[b] = float(np.max(np.abs(slice_)))
        pos = _sample_positions(L)
        for i, slot in enumerate(SAMPLE_TOKEN_SLOTS):
            samples[b, i] = slice_[pos[slot], :width]
    return {
        "samples": samples,  # [B, 3, width]
        "mean": means,
        "var": vars_,
        "l2": l2,
        "absmax": absmax,
        # Keep sample_width batch-shaped so microbatch concat stays rank-stable.
        "sample_width": np.full((batch,), width, dtype=np.int32),
    }


def capture_layer_bounded(
    *,
    hidden_in: np.ndarray,
    source: LayerScopedSource,
    layer: int,
    config: dict[str, Any],
    previous_topk: np.ndarray | None,
    cache: reference.ReferenceCache,
    lengths: np.ndarray,
    sample_hidden: int = DEFAULT_SAMPLE_HIDDEN,
    retain_per_expert: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Run exact teacher-forced layer forward; return output + bounded arrays."""
    positions = np.broadcast_to(
        np.arange(hidden_in.shape[1], dtype=np.int64)[None, :], hidden_in.shape[:2]
    ).copy()
    indexer_type = config["indexer_types"][layer]
    mlp_type = config["mlp_layer_types"][layer]
    output, topk, trace = reference.decoder_layer(
        hidden_in,
        source,
        layer,
        config,
        positions,
        cache,
        mlp_type=mlp_type,
        indexer_type=indexer_type,
        previous_topk=previous_topk,
        retain_per_expert=retain_per_expert,
    )
    attention = trace["attention"]
    full: dict[str, np.ndarray] = {
        "input_hidden": np.asarray(trace["input"], dtype=np.float32),
        "attention_input": np.asarray(trace["attention_input"], dtype=np.float32),
        "index_selection": np.asarray(attention["topk_indices"], dtype=np.int32),
        "attention_output": np.asarray(attention["attention_output"], dtype=np.float32),
        "post_attention_hidden": np.asarray(trace["post_attention"], dtype=np.float32),
        "pre_router_hidden": np.asarray(trace["mlp_input"], dtype=np.float32),
        "block_output": np.asarray(output, dtype=np.float32),
    }
    full["post_moe"] = full["block_output"] - full["post_attention_hidden"]
    mlp = trace["mlp"]
    if mlp["kind"] == "sparse":
        bias = source.tensor(f"model.layers.{layer}.mlp.gate.e_score_correction_bias")
        full.update(
            {
                "router_logits": np.asarray(mlp["router_logits"], dtype=np.float32),
                "topk_indices": np.asarray(mlp["topk_indices"], dtype=np.int32),
                "topk_weights": np.asarray(mlp["topk_weights"], dtype=np.float32),
                "topk_margin_8th_vs_9th": _router_margin(
                    mlp["router_logits"], bias, int(config["num_experts_per_tok"])
                ),
                "shared_expert_output": np.asarray(mlp["shared_output"], dtype=np.float32),
                "routed_expert_output": np.asarray(mlp["routed_output"], dtype=np.float32),
            }
        )
        if retain_per_expert and "per_expert" in mlp:
            full.update(
                _expert_cartography_arrays(
                    mlp["per_expert"], int(config["n_routed_experts"])
                )
            )

    # Bounded projection of the full arrays (samples + stats, not full dumps).
    bounded: dict[str, np.ndarray] = {}
    for key in (
        "input_hidden",
        "attention_input",
        "attention_output",
        "post_attention_hidden",
        "pre_router_hidden",
        "post_moe",
        "block_output",
    ):
        stats = _hidden_sample(full[key], lengths, sample_hidden)
        for sk, sv in stats.items():
            bounded[f"{key}/{sk}"] = sv
    bounded["index_selection"] = full["index_selection"]
    if "router_logits" in full:
        # Router: keep full top-k (small) + logit samples at sample positions.
        bounded["topk_indices"] = full["topk_indices"]
        bounded["topk_weights"] = full["topk_weights"]
        bounded["topk_margin_8th_vs_9th"] = full["topk_margin_8th_vs_9th"]
        rlog = full["router_logits"]
        bsz, seq, n_exp = rlog.shape
        width = min(32, n_exp)  # first 32 expert logits as sample dims
        r_samples = np.zeros((bsz, len(SAMPLE_TOKEN_SLOTS), width), dtype=np.float32)
        for b in range(bsz):
            L = int(lengths[b]) if b < len(lengths) else seq
            L = max(1, min(L, seq))
            pos = _sample_positions(L)
            for i, slot in enumerate(SAMPLE_TOKEN_SLOTS):
                r_samples[b, i] = rlog[b, pos[slot], :width]
        bounded["router_logits/samples"] = r_samples
        if "expert_hit_count" in full:
            bounded["expert_hit_count"] = full["expert_hit_count"]
            bounded["expert_contribution_l2"] = full["expert_contribution_l2"]

    meta = {
        "layer": layer,
        "mlp_type": mlp_type,
        "indexer_type": indexer_type,
        "metrics": _layer_metrics(full),
        "array_sha256": {k: _array_sha256(v) for k, v in sorted(bounded.items())},
        "full_metrics_from_unbounded_pass": True,
        "bounded_only_persisted": True,
    }
    return np.asarray(output, dtype=np.float32), topk, bounded, meta


# ---------------------------------------------------------------------------
# Atomic next-layer state seal
# ---------------------------------------------------------------------------


def atomic_seal_state(
    path: Path,
    *,
    hidden: np.ndarray,
    topk: np.ndarray,
    layer_completed: int,
    membership_sha256: str,
    corpus_seal: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish next-layer carry state (npz + sealed json receipt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "carry_hidden": np.ascontiguousarray(hidden, dtype=np.float32),
        "carry_index_selection": np.ascontiguousarray(topk, dtype=np.int32),
    }
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    payload = buf.getvalue()
    npz_path = path.with_suffix(".npz")
    atomic_bytes(npz_path, payload)
    receipt = seal(
        {
            "schema": "hawking.frankenstein.glm_carry_state.v1",
            "layer_completed": int(layer_completed),
            "next_layer": int(layer_completed) + 1,
            "hidden_shape": list(hidden.shape),
            "topk_shape": list(np.asarray(topk).shape),
            "hidden_sha256": _array_sha256(arrays["carry_hidden"]),
            "topk_sha256": _array_sha256(arrays["carry_index_selection"]),
            "npz_sha256": hashlib.sha256(payload).hexdigest(),
            "npz_bytes": len(payload),
            "membership_sha256": membership_sha256,
            "corpus_seal_sha256": corpus_seal,
            "extra": dict(extra or {}),
            "sealed_at": utc_now(),
        }
    )
    atomic_json(path.with_suffix(".json"), receipt)
    # Verify round-trip before caller may evict.
    reloaded = np.load(npz_path)
    if not np.array_equal(reloaded["carry_hidden"], arrays["carry_hidden"]):
        raise TeacherForcedError("carry hidden round-trip mismatch after atomic seal")
    if not np.array_equal(reloaded["carry_index_selection"], arrays["carry_index_selection"]):
        raise TeacherForcedError("carry topk round-trip mismatch after atomic seal")
    verify_sealed(json.loads(path.with_suffix(".json").read_text()), label="carry state")
    return receipt


# ---------------------------------------------------------------------------
# Double-buffer controller
# ---------------------------------------------------------------------------


@dataclass
class DoubleBufferState:
    """N-1 seal/evict · N execute · N+1 prefetch accounting."""

    n_minus_1: int | None = None
    n: int | None = None
    n_plus_1: int | None = None
    prefetched: set[int] = field(default_factory=set)
    sealed: set[int] = field(default_factory=set)
    evicted_layers: set[int] = field(default_factory=set)
    log: list[dict[str, Any]] = field(default_factory=list)

    def advance(self, layer: int, *, last_layer: int) -> dict[str, Any]:
        self.n_minus_1 = layer - 1 if layer > 0 else None
        self.n = layer
        self.n_plus_1 = layer + 1 if layer < last_layer else None
        if self.n_plus_1 is not None:
            self.prefetched.add(self.n_plus_1)
        row = {
            "layer": layer,
            "n_minus_1": self.n_minus_1,
            "n": self.n,
            "n_plus_1": self.n_plus_1,
            "pipeline": "seal_evict(N-1) | execute(N) | prefetch(N+1)",
        }
        self.log.append(row)
        return row


# ---------------------------------------------------------------------------
# Paired-trace GLM side emission
# ---------------------------------------------------------------------------


def _glm_side_from_layers(
    *,
    example_index: int,
    layer_metas: Sequence[Mapping[str, Any]],
    layer_arrays: Mapping[int, Mapping[str, np.ndarray]],
    final_logits_top_k: Sequence[Mapping[str, Any]] | None,
    lengths: np.ndarray,
) -> dict[str, Any]:
    hidden_reps: list[dict[str, Any]] = []
    route_stats: dict[str, Any] | None = None
    for layer, meta in enumerate(layer_metas):
        arr = layer_arrays.get(layer, {})
        if "block_output/samples" in arr:
            sample = arr["block_output/samples"][example_index]
            hidden_reps.append(
                {
                    "layer": layer,
                    "site": "block_output",
                    "mlp_type": meta.get("mlp_type"),
                    "sample_slots": list(SAMPLE_TOKEN_SLOTS),
                    "sample_sha256": _array_sha256(sample),
                    "l2": float(arr["block_output/l2"][example_index]),
                    "absmax": float(arr["block_output/absmax"][example_index]),
                }
            )
        if "topk_indices" in arr and route_stats is None:
            # First sparse layer route snapshot for this example.
            idx = arr["topk_indices"][example_index]
            L = int(lengths[example_index])
            L = max(1, min(L, idx.shape[0]))
            flat = idx[:L].reshape(-1)
            counts: dict[int, int] = {}
            for e in flat.tolist():
                counts[int(e)] = counts.get(int(e), 0) + 1
            route_stats = make_route_statistics(
                expert_counts=counts,
                top_k=int(idx.shape[-1]),
            )
            route_stats["source_layer"] = layer

    side = empty_side("glm")
    side.update(
        {
            "present": True,
            "capture_status": "OK",
            "representative_hidden_states": hidden_reps,
            "route_statistics": route_stats,
            "bounded_logits_top_k": list(final_logits_top_k)
            if final_logits_top_k is not None
            else None,
            "layers_captured": [m["layer"] for m in layer_metas],
            "teacher_forced": True,
            "layer_major": True,
        }
    )
    return side


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------


@dataclass
class ExecutorConfig:
    mode: str  # synthetic | official
    corpus_level: str  # L0 | L1 | ...
    source_root: Path
    output_dir: Path
    max_sequence: int = DEFAULT_MAX_SEQ
    microbatch: int = DEFAULT_MICROBATCH
    sample_hidden: int = DEFAULT_SAMPLE_HIDDEN
    profile: str = PROFILE_SYNTHETIC
    allow_eviction: bool = True
    require_floor: bool = True
    max_layers: int | None = None  # optional early stop (honesty / time)
    # Official streaming (direct hf_hub layer-major; not GLM52 schedule restream).
    stream: bool = False
    control_root: Path | None = None
    prefetch: bool = True
    corpus_jsonl: Path | None = None  # merged v0 L0/L1 jsonl override


def _load_config(source_root: Path, profile: str) -> dict[str, Any]:
    path = Path(source_root) / "config.json"
    if not path.is_file():
        raise TeacherForcedError(f"config.json absent at {path}")
    config = load_json_strict(path)
    if not isinstance(config, dict):
        raise TeacherForcedError("config.json must be an object")
    geometry = validate_config(config, profile=profile)
    resolved = dict(config)
    if profile == PROFILE_OFFICIAL:
        resolved["indexer_types"] = list(OFFICIAL_INDEXER_TYPES)
    else:
        resolved["indexer_types"] = list(
            config.get("indexer_types")
            or [
                "full" if i in {0, 1, 2, 6} else "shared"
                for i in range(geometry.num_hidden_layers)
            ]
        )
    resolved["mlp_layer_types"] = [
        "dense" if layer < geometry.first_k_dense_replace else "sparse"
        for layer in range(geometry.num_hidden_layers)
    ]
    return resolved


def _freeze_official_merged_corpus(
    *,
    level: str,
    max_sequence: int,
    pad_id: int,
    control_root: Path,
    corpus_jsonl: Path | None = None,
) -> FrozenCorpus:
    """Freeze L0/L1 from the sealed PROTO_FRANKENSTEIN_V0 merged corpus."""
    from lab.operators.glm52_adapter import load_official_tokenizer_assembly

    n = CORPUS_LEVELS[level]
    default_l0 = (
        EVIDENCE_ROOT
        / "models"
        / "frankenstein"
        / "corpus"
        / "PROTO_FRANKENSTEIN_V0_L0_CORPUS.jsonl"
    )
    default_l1 = (
        EVIDENCE_ROOT
        / "models"
        / "frankenstein"
        / "corpus"
        / "PROTO_FRANKENSTEIN_V0_L1_CORPUS.jsonl"
    )
    path = Path(
        corpus_jsonl
        or (default_l0 if level == "L0" else default_l1)
    )
    if not path.is_file():
        raise TeacherForcedError(f"merged v0 corpus absent: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TeacherForcedError("corpus jsonl row must be an object")
        rows.append(row)
    if len(rows) < n:
        raise TeacherForcedError(
            f"corpus {path} has only {len(rows)} rows; need {n} for {level}"
        )
    rows = rows[:n]
    assembly = load_official_tokenizer_assembly(Path(control_root))
    mgr = MembershipManager()
    sequences: list[FrozenSequence] = []
    for row in rows:
        example_id = str(row.get("example_id") or row.get("source_id") or "")
        if not example_id:
            raise TeacherForcedError("corpus row missing example_id")
        membership = str(row.get("membership") or "train")
        if membership not in MEMBERSHIP_SPLITS:
            membership = "train"
        text = (
            row.get("surface_text")
            or row.get("prompt")
            or row.get("text")
            or row.get("context_window")
            or ""
        )
        if not isinstance(text, str) or not text.strip():
            raise TeacherForcedError(f"empty surface_text for {example_id}")
        mgr.assign(example_id, membership)
        token_ids = list(
            assembly.tokenizer.encode(text, add_special_tokens=True)
        )[:max_sequence]
        if not token_ids:
            raise TeacherForcedError(f"tokenizer produced empty ids for {example_id}")
        sequences.append(
            FrozenSequence(
                example_id=example_id,
                membership=membership,
                prompt_text=text,
                token_ids=token_ids,
                domain=str(row.get("family") or row.get("domain") or "general"),
            )
        )
    return FrozenCorpus(
        level=level,
        sequences=sequences,
        membership=mgr,
        pad_id=pad_id,
        max_sequence=max_sequence,
        revision=IMMUTABLE_REVISION,
        source="PROTO_FRANKENSTEIN_V0_MERGED_CORPUS",
    )


def _build_plan_from_streamer(
    streamer: LayerMajorStreamer, inventory: Inventory
) -> LayerWeightPlan:
    """LayerWeightPlan over the stream root with full layer map from the index."""
    return LayerWeightPlan(
        root=streamer.stream_root,
        inventory=inventory,
        layer_to_shards={k: set(v) for k, v in streamer.layer_to_shards.items()},
        global_shards=set(streamer.global_shards),
        shard_to_layers={k: set(v) for k, v in streamer.shard_to_layers.items()},
        shard_hashes=dict(streamer.verified_hashes),
    )


def run_teacher_forced(cfg: ExecutorConfig) -> dict[str, Any]:
    """Execute the full layer-major teacher-forced capture pipeline."""
    started = time.time()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    floor_records: list[dict[str, Any]] = []
    if cfg.require_floor:
        floor_records.append(assert_floor(out, label="output_dir_start"))

    profile = cfg.profile
    streamer: LayerMajorStreamer | None = None
    stream_receipt: dict[str, Any] | None = None

    # Official streaming path: control plane + direct hf_hub layer-major body.
    use_stream = bool(cfg.stream) and profile == PROFILE_OFFICIAL
    if use_stream:
        control = Path(cfg.control_root or DEFAULT_CONTROL_ROOT)
        try:
            ensure_control_assets(control, revision=IMMUTABLE_REVISION)
            streamer = LayerMajorStreamer(
                control_root=control,
                stream_root=Path(cfg.source_root),
                require_floor=cfg.require_floor,
            )
        except LayerStreamError as exc:
            raise TeacherForcedError(f"stream bootstrap failed: {exc}") from exc
        config = _load_config(control, profile)
        # Pad id from official config (list form allowed).
        pad_raw = config.get("pad_token_id", 0)
        if isinstance(pad_raw, list):
            pad_id = int(pad_raw[0]) if pad_raw else 0
        else:
            pad_id = int(pad_raw or 0)
        try:
            corpus = _freeze_official_merged_corpus(
                level=cfg.corpus_level,
                max_sequence=cfg.max_sequence,
                pad_id=pad_id,
                control_root=control,
                corpus_jsonl=cfg.corpus_jsonl,
            )
        except Exception as exc:  # noqa: BLE001
            raise TeacherForcedError(
                f"official merged corpus freeze failed: {exc}"
            ) from exc
        n_layers = int(config["num_hidden_layers"])
        if cfg.max_layers is not None:
            n_layers = min(n_layers, int(cfg.max_layers))

        # Ensure globals + layer 0, admit inventory, build plan.
        need0 = streamer.shards_for_layer(0, include_global=True)
        try:
            if cfg.require_floor:
                floor_records.append(assert_floor(out, label="pre_stream_L00"))
            streamer.ensure(need0)
            if cfg.prefetch and n_layers > 1:
                streamer.prefetch(streamer.shards_for_layer(1, include_global=False))
            inventory = streamer.admit_inventory()
        except LayerStreamError as exc:
            raise TeacherForcedError(f"stream ensure L0 failed: {exc}") from exc
        plan = _build_plan_from_streamer(streamer, inventory)
        plan.shard_hashes.update(streamer.verified_hashes)
    else:
        config = _load_config(cfg.source_root, profile)
        try:
            inventory = verify_checkpoint(cfg.source_root, profile=profile, view="full")
        except Exception as exc:  # noqa: BLE001 — incomplete checkpoint is fail-closed
            raise TeacherForcedError(
                f"checkpoint not verifiable at {cfg.source_root}: {exc}"
            ) from exc
        plan = build_weight_plan(cfg.source_root, inventory)
        n_layers = int(config["num_hidden_layers"])
        if cfg.max_layers is not None:
            n_layers = min(n_layers, int(cfg.max_layers))

        corpus = freeze_corpus(
            level=cfg.corpus_level,
            mode=cfg.mode if cfg.mode != "official" else (
                "official" if profile == PROFILE_OFFICIAL else "synthetic"
            ),
            max_sequence=cfg.max_sequence,
            vocab_size=int(config["vocab_size"]),
            pad_id=int(config.get("pad_token_id", 0) or 0)
            if not isinstance(config.get("pad_token_id"), list)
            else int(config.get("pad_token_id")[0]),
        )
        # Official mode with synthetic profile still freezes synthetic prompts.
        if profile == PROFILE_SYNTHETIC:
            corpus = freeze_corpus(
                level=cfg.corpus_level,
                mode="synthetic",
                max_sequence=min(
                    cfg.max_sequence,
                    int(config.get("index_topk", 2))
                    if cfg.mode == "synthetic"
                    else cfg.max_sequence,
                ),
                vocab_size=int(config["vocab_size"]),
                pad_id=0,
            )
            # DSA index_topk for synthetic is 2; keep sequences ≤ index_topk.
            if corpus.max_sequence > int(config["index_topk"]):
                corpus = freeze_corpus(
                    level=cfg.corpus_level,
                    mode="synthetic",
                    max_sequence=int(config["index_topk"]),
                    vocab_size=int(config["vocab_size"]),
                    pad_id=0,
                )

    corpus_doc = corpus.document()
    atomic_json(out / f"FROZEN_CORPUS_{cfg.corpus_level}.json", corpus_doc)

    batch_ids = corpus.batch_ids()
    lengths = corpus.attention_lengths()
    bsz, seqlen = batch_ids.shape
    if seqlen == 0:
        raise TeacherForcedError("frozen corpus produced empty sequences")

    if not use_stream:
        # Embedding requires global shards (resident full-checkpoint path).
        embed_missing = [
            s for s in plan.global_shards if not (plan.root / s).is_file()
        ]
        missing0 = plan.missing_for_layer(0)
        if missing0 or embed_missing:
            raise TeacherForcedError(
                "source weights not resident for embedding/layer0; "
                f"missing={sorted(set(missing0) | set(embed_missing))[:8]}. "
                "Official BF16 body is not on disk — fail closed (never fake activations). "
                "Re-run with --stream for direct hf_hub layer-major fetch."
            )

    # Admit all currently resident shards at start.
    admitted = plan.resident_shards()
    if not admitted:
        raise TeacherForcedError("no weight shards resident")
    if not use_stream:
        plan.verify_window_hashes(admitted)
    else:
        # Streamer already hash-verified; record digests on the plan.
        plan.shard_hashes.update(
            {n: streamer.verified_hashes[n] for n in admitted if n in streamer.verified_hashes}  # type: ignore[union-attr]
        )

    source = LayerScopedSource(plan, admitted_shards=admitted)
    hidden = np.asarray(
        source.rows("model.embed_tokens.weight", batch_ids), dtype=np.float32
    )
    # Seed previous_topk for shared IndexShare when layer 0 is full.
    previous_topk = np.tile(
        np.arange(seqlen, dtype=np.int32), (bsz, seqlen, 1)
    )[:, :, : int(config["index_topk"])]
    # For sequences shorter than index_topk the reference indexer returns min(topk, keys).

    embed_stats = _hidden_sample(hidden, lengths, cfg.sample_hidden)
    embed_arrays = {f"embedding/{k}": v for k, v in embed_stats.items()}
    embed_shard_path = out / "layers" / "embedding"
    embed_shard_path.parent.mkdir(parents=True, exist_ok=True)
    _write_layer_shard(
        embed_shard_path,
        layer_id="embedding",
        arrays=embed_arrays,
        meta={
            "site": "embedding",
            "shape": list(hidden.shape),
            "metrics": {
                "embedding_l2_mean": float(np.mean(embed_stats["l2"])),
            },
        },
        corpus_seal=corpus_doc["seal_sha256"],
    )

    db = DoubleBufferState()
    layer_metas: list[dict[str, Any]] = []
    layer_arrays_by_layer: dict[int, dict[str, np.ndarray]] = {}
    eviction_receipts: list[dict[str, Any]] = []
    layers_captured: list[int] = []
    bytes_captured = 0
    cache = reference.ReferenceCache()
    deepest_blocker: str | None = None

    for layer in range(n_layers):
        if cfg.require_floor:
            floor_records.append(assert_floor(out, label=f"pre_layer_{layer:02d}"))
        db_row = db.advance(layer, last_layer=n_layers - 1)

        # Layer body shards only (globals kept separately for embed/final).
        need = plan.shards_for_layer(layer, include_global=False)
        # Layer 0 may share the embed global shard; always include globals for L0.
        if layer == 0:
            need = plan.shards_for_layer(layer, include_global=True)

        if use_stream and streamer is not None:
            try:
                # Prefetch N+1 before blocking on N (double-buffer).
                if cfg.prefetch and db.n_plus_1 is not None:
                    streamer.prefetch(
                        streamer.shards_for_layer(db.n_plus_1, include_global=False)
                    )
                    db_row["prefetch_status"] = f"SUBMITTED_L{db.n_plus_1:02d}"
                else:
                    db_row["prefetch_status"] = "NONE"
                streamer.ensure(need)
                inventory = streamer.admit_inventory()
                plan.inventory = inventory
                plan.shard_hashes.update(streamer.verified_hashes)
                source = LayerScopedSource(
                    plan, admitted_shards=plan.resident_shards()
                )
                # Carry over payload counters is not required across rebuilds.
            except LayerStreamError as exc:
                deepest_blocker = (
                    f"lab/operators/glm52_teacher_forced_executor.py:"
                    f"stream_ensure layer {layer}: {exc}"
                )
                break
        else:
            missing = plan.missing_for_layer(layer, include_global=(layer == 0))
            if missing:
                deepest_blocker = (
                    f"lab/operators/glm52_teacher_forced_executor.py:"
                    f"missing_for_layer weights not resident for layer {layer}: {missing[:5]}"
                )
                break
            if db.n_plus_1 is not None:
                prefetch_missing = plan.missing_for_layer(
                    db.n_plus_1, include_global=False
                )
                db_row["prefetch_status"] = (
                    "RESIDENT"
                    if not prefetch_missing
                    else f"MISSING_{len(prefetch_missing)}"
                )
            else:
                db_row["prefetch_status"] = "NONE"
            source.admitted_shards = plan.resident_shards()
            plan.verify_window_hashes(
                plan.shards_for_layer(layer, include_global=(layer == 0))
                & source.admitted_shards
            )

        # Microbatch over sequences (layer-major: one layer, all sequences).
        out_hidden = np.zeros_like(hidden)
        out_topk_list: list[np.ndarray] = []
        bounded_acc: dict[str, list[np.ndarray]] = {}
        metrics_acc: list[dict[str, float]] = []
        mb = max(1, int(cfg.microbatch))
        for start in range(0, bsz, mb):
            end = min(bsz, start + mb)
            # Per-microbatch cache (prefill over full sequence, teacher-forced).
            local_cache = reference.ReferenceCache()
            h_in = hidden[start:end]
            # Slice previous topk if shapes allow.
            if previous_topk is not None and previous_topk.shape[0] == bsz:
                pt = previous_topk[start:end]
            else:
                pt = previous_topk
            h_out, topk, bounded, meta = capture_layer_bounded(
                hidden_in=h_in,
                source=source,
                layer=layer,
                config=config,
                previous_topk=pt,
                cache=local_cache,
                lengths=lengths[start:end],
                sample_hidden=cfg.sample_hidden,
            )
            out_hidden[start:end] = h_out
            out_topk_list.append(topk)
            metrics_acc.append(meta["metrics"])
            for k, v in bounded.items():
                bounded_acc.setdefault(k, []).append(v)

        # Concatenate microbatch bounded arrays (skip pure scalars).
        bounded_cat: dict[str, np.ndarray] = {}
        for k, parts in bounded_acc.items():
            if not parts:
                continue
            if all(np.asarray(p).ndim == 0 for p in parts):
                bounded_cat[k] = np.asarray(parts[0])
                continue
            bounded_cat[k] = np.concatenate(
                [np.asarray(p) for p in parts], axis=0
            )

        # Aggregate metrics (mean over microbatches of scalar metrics).
        keys = sorted(metrics_acc[0])
        agg_metrics = {
            k: float(np.mean([m[k] for m in metrics_acc])) for k in keys
        }
        meta_out = {
            "layer": layer,
            "mlp_type": config["mlp_layer_types"][layer],
            "indexer_type": config["indexer_types"][layer],
            "metrics": agg_metrics,
            "array_sha256": {k: _array_sha256(v) for k, v in sorted(bounded_cat.items())},
            "microbatches": int(np.ceil(bsz / mb)),
            "double_buffer": db_row,
            "payload_bytes_read": source.payload_bytes_read,
        }
        layer_path = out / "layers" / f"L{layer:02d}"
        shard_receipt = _write_layer_shard(
            layer_path,
            layer_id=f"L{layer:02d}",
            arrays=bounded_cat,
            meta=meta_out,
            corpus_seal=corpus_doc["seal_sha256"],
        )
        bytes_captured += int(shard_receipt["npz_bytes"])
        layer_metas.append(meta_out)
        layer_arrays_by_layer[layer] = bounded_cat
        layers_captured.append(layer)
        db.sealed.add(layer)

        # Stitch topk across microbatches.
        previous_topk = np.concatenate(out_topk_list, axis=0)
        hidden = out_hidden

        # Atomic seal of next-layer state.
        carry_receipt = atomic_seal_state(
            out / "carry" / f"after_L{layer:02d}",
            hidden=hidden,
            topk=previous_topk,
            layer_completed=layer,
            membership_sha256=corpus.membership_sha256(),
            corpus_seal=corpus_doc["seal_sha256"],
            extra={"layers_captured_so_far": list(layers_captured)},
        )
        meta_out["carry_seal_sha256"] = carry_receipt["seal_sha256"]

        # Evict shards only needed by completed layers (not remaining).
        completed = set(layers_captured)
        remaining = set(range(layer + 1, n_layers))
        # Keep globals until final layer done.
        if cfg.allow_eviction:
            victims = shards_evictable_after(
                plan,
                completed_layers=completed,
                remaining_layers=remaining,
                keep_global_until_final=True,
            )
            # Never touch global shards (embed / final norm / lm_head) inside the
            # layer loop — final bounded logits still need them after L_last.
            victims = [v for v in victims if v not in plan.global_shards]
            # Also keep anything the prefetched next layer still needs.
            if db.n_plus_1 is not None:
                keep_next = plan.shards_for_layer(db.n_plus_1, include_global=False)
                victims = [v for v in victims if v not in keep_next]
            if victims:
                if use_stream and streamer is not None:
                    raw = streamer.evict(victims)
                    ev = seal(
                        {
                            "schema": SCHEMA_EVICTION,
                            "removed": raw["removed"],
                            "bytes_reclaimed": raw["bytes_reclaimed"],
                            "at": raw["at"],
                            "policy": raw["policy"],
                        }
                    )
                    plan.shard_hashes = {
                        k: v
                        for k, v in plan.shard_hashes.items()
                        if k not in {r["shard"] for r in raw["removed"]}
                    }
                else:
                    ev = evict_shards(plan, victims)
                eviction_receipts.append(ev)
                db.evicted_layers.add(layer)
                source.admitted_shards = plan.resident_shards()

        if cfg.require_floor:
            floor_records.append(assert_floor(out, label=f"post_layer_{layer:02d}"))

    # Final norm + logits if we finished all layers and globals still resident.
    final_logits_top_k_per_seq: list[list[dict[str, Any]]] = [[] for _ in range(bsz)]
    final_status = "SKIPPED"
    if len(layers_captured) == n_layers:
        try:
            if use_stream and streamer is not None:
                streamer.ensure(streamer.global_shards)
                inventory = streamer.admit_inventory()
                plan.inventory = inventory
                plan.shard_hashes.update(streamer.verified_hashes)
                source = LayerScopedSource(
                    plan, admitted_shards=plan.resident_shards()
                )
            if source.resident("model.norm.weight") and source.resident(
                "lm_head.weight"
            ):
                normed = reference.rmsnorm(
                    hidden,
                    source.tensor("model.norm.weight"),
                    float(config["rms_norm_eps"]),
                )
                # Bounded short logits over first 64 vocab rows (never full 154k×B×S).
                rows = np.arange(min(64, int(config["vocab_size"])), dtype=np.int64)
                head = source.rows("lm_head.weight", rows)
                short = reference.linear(normed, head).astype(np.float32)
                final_stats = _hidden_sample(normed, lengths, cfg.sample_hidden)
                final_arrays = {
                    **{f"final_norm/{k}": v for k, v in final_stats.items()},
                    "short_logits": short,
                }
                # top-k over short vocab for each sequence last token
                for b in range(bsz):
                    L = int(lengths[b])
                    pos = max(0, min(L - 1, short.shape[1] - 1))
                    row = short[b, pos]
                    k = min(8, row.shape[0])
                    idx = np.argsort(-row)[:k]
                    final_logits_top_k_per_seq[b] = [
                        {"token_id": int(i), "logit": float(row[i]), "rank": r}
                        for r, i in enumerate(idx)
                    ]
                fr = _write_layer_shard(
                    out / "layers" / "final",
                    layer_id="final",
                    arrays=final_arrays,
                    meta={
                        "site": "final_norm_and_short_logits",
                        "short_vocab_rows": int(rows.shape[0]),
                    },
                    corpus_seal=corpus_doc["seal_sha256"],
                )
                bytes_captured += int(fr["npz_bytes"])
                final_status = "CAPTURED_BOUNDED"
                # Evict remaining globals after final capture.
                if cfg.allow_eviction:
                    leftovers = [
                        s
                        for s in plan.resident_shards()
                        if s in plan.global_shards
                        or not (
                            plan.shard_to_layers.get(s, set())
                            & set(range(n_layers, 10_000))
                        )
                    ]
                    victims = shards_evictable_after(
                        plan,
                        completed_layers=set(range(n_layers)),
                        remaining_layers=set(),
                        keep_global_until_final=False,
                    )
                    victims = sorted(set(victims) | set(leftovers))
                    if victims:
                        if use_stream and streamer is not None:
                            raw = streamer.evict(victims)
                            eviction_receipts.append(
                                seal(
                                    {
                                        "schema": SCHEMA_EVICTION,
                                        "removed": raw["removed"],
                                        "bytes_reclaimed": raw["bytes_reclaimed"],
                                        "at": raw["at"],
                                        "policy": raw["policy"],
                                    }
                                )
                            )
                        else:
                            eviction_receipts.append(evict_shards(plan, victims))
            else:
                final_status = "SKIPPED_GLOBALS_ABSENT"
        except Exception as exc:  # noqa: BLE001
            final_status = f"FAILED:{type(exc).__name__}:{exc}"
            if deepest_blocker is None:
                deepest_blocker = (
                    f"lab/operators/glm52_teacher_forced_executor.py:final_norm "
                    f"{type(exc).__name__}: {exc}"
                )

    if streamer is not None:
        stream_receipt = streamer.receipt_block()
        streamer.close()

    # Emit paired traces (GLM side only; DSV4F absent).
    traces_dir = out / "paired_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_paths: list[Path] = []
    for i, seq in enumerate(corpus.sequences):
        glm_side = _glm_side_from_layers(
            example_index=i,
            layer_metas=layer_metas,
            layer_arrays=layer_arrays_by_layer,
            final_logits_top_k=final_logits_top_k_per_seq[i] or None,
            lengths=lengths,
        )
        prompt = seq.prompt_text
        spans = [
            make_decoded_span(
                text=prompt,
                byte_start=0,
                byte_end=len(prompt.encode("utf-8")),
                role="prompt",
                side="glm",
            )
        ]
        trace = build_paired_trace(
            example_id=seq.example_id,
            membership=seq.membership,
            prompt_text=prompt,
            glm_side=glm_side,
            dsv4f_side=empty_side("dsv4f"),
            decoded_spans=spans,
            meta={
                "complete_pair": False,
                "glm_teacher_forced": True,
                "layers_captured": layers_captured,
                "corpus_level": cfg.corpus_level,
            },
        )
        tpath = traces_dir / f"{seq.example_id}.json"
        atomic_json(tpath, trace)
        trace_paths.append(tpath)

    corpus_index = index_corpus(
        [json.loads(p.read_text()) for p in trace_paths]
    )
    atomic_json(out / "PAIRED_TRACE_CORPUS_INDEX.json", corpus_index)

    floor_final = (
        assert_floor(out, label="output_dir_final")
        if cfg.require_floor
        else {
            "floor_bytes": MIN_FREE_FLOOR_BYTES,
            "free_bytes": free_bytes(out),
            "floor_preserved": free_bytes(out) >= MIN_FREE_FLOOR_BYTES,
        }
    )
    floor_records.append(floor_final)

    status = (
        "PASS_FULL_STACK"
        if len(layers_captured) == int(config["num_hidden_layers"])
        and final_status.startswith("CAPTURED")
        else "PASS_PARTIAL"
        if layers_captured
        else "FAIL_CLOSED"
    )
    if profile == PROFILE_SYNTHETIC and status.startswith("PASS"):
        status = "PASS_SYNTHETIC_" + status.split("PASS_")[-1]

    receipt = seal(
        {
            "schema": SCHEMA_RECEIPT,
            "status": status,
            "mode": cfg.mode,
            "profile": profile,
            "repo": REPO_ID if profile == PROFILE_OFFICIAL else "synthetic/GLM-5.2-twin",
            "revision": IMMUTABLE_REVISION
            if profile == PROFILE_OFFICIAL
            else "synthetic-revision",
            "architecture": {
                "num_hidden_layers_config": int(config["num_hidden_layers"]),
                "layers_executed": n_layers,
                "hidden_size": int(config["hidden_size"]),
                "n_routed_experts": int(config["n_routed_experts"]),
                "num_experts_per_tok": int(config["num_experts_per_tok"]),
                "first_k_dense_replace": int(config["first_k_dense_replace"]),
            },
            "corpus": {
                "level": cfg.corpus_level,
                "n_sequences": corpus.n_sequences,
                "source": corpus.source,
                "membership_sha256": corpus.membership_sha256(),
                "seal_sha256": corpus_doc["seal_sha256"],
                "max_sequence": corpus.max_sequence,
            },
            "layers_captured": layers_captured,
            "deepest_layer_verified": (max(layers_captured) if layers_captured else None),
            "layers_total_config": int(config["num_hidden_layers"]),
            "final_status": final_status,
            "bytes_captured": bytes_captured,
            "evictions": eviction_receipts,
            "eviction_count": len(eviction_receipts),
            "bytes_reclaimed": int(
                sum(int(e.get("bytes_reclaimed", 0)) for e in eviction_receipts)
            ),
            "double_buffer_log": db.log,
            "floor": {
                "floor_bytes": MIN_FREE_FLOOR_BYTES,
                "preserved_throughout": all(
                    r.get("floor_preserved") for r in floor_records if "floor_preserved" in r
                ),
                "records": floor_records[-5:],  # tail only in seal
                "final": floor_final,
            },
            "paired_traces": {
                "n": len(trace_paths),
                "index_seal_sha256": corpus_index["seal_sha256"],
                "dir": str(traces_dir),
            },
            "forward": {
                "kind": "teacher_forced_layer_major",
                "autoregressive": False,
                "reference_module": "lab.operators.glm52_reference.decoder_layer",
                "exact_ops": [
                    "rmsnorm",
                    "mla_dsa_attention",
                    "indexshare",
                    "interleaved_rope",
                    "noaux_tc_router",
                    "swiglu_moe",
                ],
                "verification": (
                    "synthetic: exact reference path previously sealed in "
                    "GLM52_REFERENCE_PARITY.json against Transformers; this run "
                    "reuses the same decoder_layer implementation without faking."
                ),
            },
            "blocker": deepest_blocker,
            "resident_shards_remaining": sorted(plan.resident_shards()),
            "stream": stream_receipt,
            "stream_enabled": bool(use_stream),
            "capture_seconds": round(time.time() - started, 3),
            "captured_at": utc_now(),
            "output_dir": str(out),
            "fabricated": False,
            "capability_claim_permitted": False,
            "note": (
                "Teacher-forced layer-major capture. Not a serving stack. "
                "Official path uses direct hf_hub layer-major streaming "
                "(not GLM52_STREAMING_SCHEDULE restream). Synthetic path "
                "proves the mechanism end-to-end on a miniature twin."
            ),
        }
    )
    atomic_json(out / "GLM_TEACHER_FORCED_CAPTURE_RECEIPT.json", receipt)
    return receipt


def _write_layer_shard(
    path: Path,
    *,
    layer_id: str,
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
    corpus_seal: str,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    payload = buf.getvalue()
    npz_path = path.with_suffix(".npz") if path.suffix != ".npz" else path
    if path.suffix == "":
        npz_path = Path(str(path) + ".npz")
        json_path = Path(str(path) + ".json")
    else:
        json_path = path.with_suffix(".json")
    atomic_bytes(npz_path, payload)
    receipt = seal(
        {
            "schema": SCHEMA_LAYER_SHARD,
            "layer_id": layer_id,
            "meta": dict(meta),
            "array_names": sorted(arrays),
            "array_sha256": {k: _array_sha256(v) for k, v in sorted(arrays.items())},
            "npz_sha256": hashlib.sha256(payload).hexdigest(),
            "npz_bytes": len(payload),
            "corpus_seal_sha256": corpus_seal,
            "sealed_at": utc_now(),
        }
    )
    atomic_json(json_path, receipt)
    return receipt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Teacher-forced layer-major GLM-5.2 activation capture"
    )
    parser.add_argument(
        "--mode",
        choices=("synthetic", "official"),
        default="synthetic",
    )
    parser.add_argument(
        "--corpus-level",
        choices=tuple(CORPUS_LEVELS),
        default="L0",
    )
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-sequence", type=int, default=DEFAULT_MAX_SEQ)
    parser.add_argument("--microbatch", type=int, default=DEFAULT_MICROBATCH)
    parser.add_argument("--sample-hidden", type=int, default=DEFAULT_SAMPLE_HIDDEN)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--no-evict", action="store_true")
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Official mode: direct hf_hub layer-major streaming into stream_root "
            "(not GLM52_STREAMING_SCHEDULE restream)"
        ),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=None,
        help="Control plane with config.json + index + tokenizer (revision-bound)",
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable N+1 double-buffer prefetch",
    )
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=None,
        help="Override merged v0 corpus jsonl (default PROTO_FRANKENSTEIN_V0_L0/L1)",
    )
    parser.add_argument(
        "--build-synthetic-fixture",
        type=Path,
        default=None,
        help="Build a fresh synthetic fixture at this path then run against it",
    )
    args = parser.parse_args(argv)

    source_root = args.source_root
    profile = PROFILE_SYNTHETIC if args.mode == "synthetic" else PROFILE_OFFICIAL
    stream = bool(args.stream)

    if args.build_synthetic_fixture is not None or (
        args.mode == "synthetic" and source_root is None
    ):
        from lab.operators.glm52_synthetic import build_synthetic_fixture

        fixture_root = args.build_synthetic_fixture or (
            Path(os.environ.get("TMPDIR", "/tmp"))
            / f"glm52_tf_fixture_{os.getpid()}"
        )
        if fixture_root.exists():
            shutil.rmtree(fixture_root)
        fx = build_synthetic_fixture(fixture_root)
        source_root = fx.full_dir
        profile = PROFILE_SYNTHETIC
        stream = False

    if source_root is None:
        if profile == PROFILE_OFFICIAL and stream:
            source_root = Path(
                os.environ.get("GLM52_STREAM_ROOT", str(DEFAULT_STREAM_ROOT))
            )
        else:
            # Official default location (full body when present).
            source_root = Path(
                os.environ.get(
                    "GLM52_SOURCE_ROOT",
                    "/Users/scammermike/Library/Application Support/Hawking/"
                    "GLM52Gravity/source",
                )
            )

    # Auto-enable streaming for official when the classical source root has no body.
    if profile == PROFILE_OFFICIAL and not stream:
        cfg_path = Path(source_root) / "config.json"
        any_shard = any(Path(source_root).glob("model-*.safetensors"))
        if not cfg_path.is_file() or not any_shard:
            stream = True
            if source_root is None or not any_shard:
                source_root = Path(
                    os.environ.get("GLM52_STREAM_ROOT", str(DEFAULT_STREAM_ROOT))
                )

    output_dir = args.output_dir or (
        DEFAULT_OUT / f"{args.mode}_{args.corpus_level}_{time.strftime('%Y%m%dT%H%M%SZ')}"
    )

    cfg = ExecutorConfig(
        mode=args.mode,
        corpus_level=args.corpus_level,
        source_root=Path(source_root),
        output_dir=Path(output_dir),
        max_sequence=args.max_sequence,
        microbatch=args.microbatch,
        sample_hidden=args.sample_hidden,
        profile=profile,
        allow_eviction=not args.no_evict,
        require_floor=not args.no_floor,
        max_layers=args.max_layers,
        stream=stream,
        control_root=args.control_root,
        prefetch=not args.no_prefetch,
        corpus_jsonl=args.corpus_jsonl,
    )
    try:
        receipt = run_teacher_forced(cfg)
    except TeacherForcedError as exc:
        err = seal(
            {
                "schema": SCHEMA_RECEIPT,
                "status": "FAIL_CLOSED",
                "error": str(exc),
                "mode": args.mode,
                "corpus_level": args.corpus_level,
                "at": utc_now(),
                "fabricated": False,
            }
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        atomic_json(Path(output_dir) / "GLM_TEACHER_FORCED_CAPTURE_RECEIPT.json", err)
        print(json.dumps(err, indent=2, sort_keys=True))
        return 78
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if str(receipt.get("status", "")).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
