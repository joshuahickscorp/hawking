#!/usr/bin/env python3.12
"""Fail-closed GLM-5.2 checkpoint adapter and bounded safetensors reader.

The adapter has two explicit profiles:

``official``
    The immutable ``zai-org/GLM-5.2`` BF16 checkpoint at revision
    ``b4734de4facf877f85769a911abafc5283eab3d9``.  Its architecture-affecting
    config values, tensor names, shapes, dtypes, IndexShare omissions, and
    physical MTP layer are fixed here.

``synthetic``
    A small, architecture-preserving fixture profile.  It deliberately retains
    256 routed experts/top-8, three dense main layers, sparse main layers, the
    full/shared IndexShare pattern, and a separately stored MTP layer.  Only
    matrix dimensions and main-layer count are reduced.

There is no permissive catch-all.  Unknown, missing, duplicated, misplaced, or
wrongly typed tensors fail before any tensor payload is returned.  The reader
uses shard-local offsets and ``pread`` to read exactly one complete tensor at a
time, refuses symlinks and non-regular files, and supports the two source dtypes
actually present in GLM-5.2: BF16 and F32.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np


REPO_ID = "zai-org/GLM-5.2"
IMMUTABLE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
EXPECTED_ARCHITECTURE = "GlmMoeDsaForCausalLM"
EXPECTED_MODEL_TYPE = "glm_moe_dsa"

PROFILE_OFFICIAL = "official"
PROFILE_SYNTHETIC = "synthetic"
VIEW_FULL = "full"
VIEW_CORE = "core"
CORE = "CORE"
MTP = "MTP"

DTYPE_BYTES = {"BF16": 2, "F32": 4}
MAX_HEADER_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_TENSOR_BYTES = 256 * 1024 * 1024

OFFICIAL_TOKENIZER_ASSETS: Mapping[str, tuple[int, str]] = {
    "tokenizer.json": (
        20_217_442,
        "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    ),
    "tokenizer_config.json": (
        761,
        "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
    ),
    "chat_template.jinja": (
        5_076,
        "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
    ),
    "generation_config.json": (
        194,
        "ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55",
    ),
}

GATE_UP_ORDER = ("gate_proj", "up_proj")

_SAFE_SHARD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors$")
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(
    r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


class Glm52AdapterError(RuntimeError):
    """A config, index, shard, or bounded-read invariant was violated."""


@dataclass(frozen=True)
class OfficialTokenizerAssembly:
    """Pinned local tokenizer and chat template with a secret-free receipt."""

    root: Path
    tokenizer: Any
    asset_sha256: Mapping[str, str]
    vocabulary_size: int
    padded_model_vocabulary_size: int
    eos_token_id: int
    pad_token_id: int
    generation_eos_token_ids: tuple[int, ...]
    generation_pad_token_id: int
    model_max_length: int

    def assemble_chat(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        tools: Iterable[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: Literal["high", "max"] = "max",
        clear_thinking: bool = True,
    ) -> dict[str, Any]:
        if not all(isinstance(value, bool) for value in (
            add_generation_prompt, enable_thinking, clear_thinking
        )):
            raise Glm52AdapterError("chat template boolean options must be booleans")
        if reasoning_effort not in {"high", "max"}:
            raise Glm52AdapterError("reasoning_effort must be 'high' or 'max'")
        try:
            rows = json.loads(json.dumps(
                [dict(message) for message in messages],
                ensure_ascii=False,
                allow_nan=False,
            ))
        except (TypeError, ValueError) as exc:
            raise Glm52AdapterError(f"chat messages are not canonical JSON: {exc}") from exc
        if not rows:
            raise Glm52AdapterError("chat assembly requires at least one message")
        allowed_roles = {"system", "user", "assistant", "tool"}
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or "role" not in row or "content" not in row:
                raise Glm52AdapterError(
                    f"chat message {index} must contain role/content"
                )
            if row["role"] not in allowed_roles:
                raise Glm52AdapterError(f"unsupported chat role: {row['role']!r}")
            allowed_keys = {"role", "content"}
            if row["role"] == "assistant":
                allowed_keys.update({"reasoning_content", "tool_calls"})
            unknown = sorted(set(row) - allowed_keys)
            if unknown:
                raise Glm52AdapterError(
                    f"chat message {index} has unsupported fields: {unknown}"
                )
            content = row["content"]
            if content is None and row["role"] != "assistant":
                raise Glm52AdapterError(
                    f"chat content {index} may be null only for assistant tool calls"
                )
            if content is not None and not isinstance(content, (str, list)):
                raise Glm52AdapterError(
                    f"chat content {index} must be text, a content-part list, or assistant null"
                )
            if "reasoning_content" in row and not isinstance(row["reasoning_content"], str):
                raise Glm52AdapterError(
                    f"assistant reasoning_content {index} must be text"
                )
            if "tool_calls" in row:
                calls = row["tool_calls"]
                if not isinstance(calls, list) or not calls:
                    raise Glm52AdapterError(
                        f"assistant tool_calls {index} must be a non-empty list"
                    )
                for call_index, call in enumerate(calls):
                    if not isinstance(call, dict):
                        raise Glm52AdapterError(
                            f"assistant tool call {index}/{call_index} must be an object"
                        )
                    function = call.get("function", call)
                    if not isinstance(function, dict) or not isinstance(
                        function.get("name"), str
                    ) or not function["name"]:
                        raise Glm52AdapterError(
                            f"assistant tool call {index}/{call_index} lacks a function name"
                        )
                    if not isinstance(function.get("arguments"), dict):
                        raise Glm52AdapterError(
                            f"assistant tool call {index}/{call_index} arguments must be an object"
                        )
        try:
            tool_rows = None if tools is None else json.loads(json.dumps(
                [dict(tool) for tool in tools],
                ensure_ascii=False,
                allow_nan=False,
            ))
        except (TypeError, ValueError) as exc:
            raise Glm52AdapterError(f"tool declarations are not canonical JSON: {exc}") from exc
        for index, tool in enumerate(tool_rows or []):
            function = tool.get("function", tool) if isinstance(tool, dict) else None
            if not isinstance(function, dict) or not isinstance(
                function.get("name"), str
            ) or not function["name"]:
                raise Glm52AdapterError(f"tool declaration {index} lacks a function name")
        rendered = self.tokenizer.apply_chat_template(
            rows,
            tokenize=False,
            tools=tool_rows,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
        )
        if not isinstance(rendered, str) or not rendered:
            raise Glm52AdapterError("official chat template returned no text")
        token_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if not isinstance(token_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in token_ids
        ):
            raise Glm52AdapterError("official tokenizer returned invalid token IDs")
        return {
            "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "token_count": len(token_ids),
            "begins_with_glm_mask_sop": rendered.startswith("[gMASK]<sop>"),
            "ends_with_generation_think": (
                rendered.endswith(
                    "<|assistant|><think>" if enable_thinking
                    else "<|assistant|><think></think>"
                )
                if add_generation_prompt
                else None
            ),
            "tool_count": len(tool_rows or []),
            "contains_tool_catalog": "<tools>" in rendered,
            "contains_tool_call": "<tool_call>" in rendered,
            "contains_tool_response": "<tool_response>" in rendered,
            "thinking_enabled": enable_thinking,
            "reasoning_effort": reasoning_effort,
            "generation_eos_token_ids": list(self.generation_eos_token_ids),
            "generation_pad_token_id": self.generation_pad_token_id,
        }


def load_official_tokenizer_assembly(control_root: Path) -> OfficialTokenizerAssembly:
    """Load the immutable official tokenizer/template without any network access."""
    control_root = Path(control_root).resolve()
    if IMMUTABLE_REVISION not in str(control_root):
        raise Glm52AdapterError(
            "official tokenizer root is not bound to the immutable source revision"
        )
    asset_hashes: dict[str, str] = {}
    for name, (expected_bytes, expected_sha256) in OFFICIAL_TOKENIZER_ASSETS.items():
        path = control_root / name
        if not path.exists() or not path.is_file():
            raise Glm52AdapterError(f"official tokenizer asset missing: {name}")
        observed_bytes = path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_bytes != expected_bytes or digest != expected_sha256:
            raise Glm52AdapterError(
                f"official tokenizer asset identity mismatch for {name}: "
                f"bytes/sha256={observed_bytes}/{digest}"
            )
        asset_hashes[name] = digest
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            control_root,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise Glm52AdapterError(f"failed to load pinned local tokenizer: {exc}") from exc
    template = (control_root / "chat_template.jinja").read_text(encoding="utf-8")
    generation = load_json_strict(control_root / "generation_config.json")
    generation_eos = (154_820, 154_827, 154_829)
    if not isinstance(generation, dict) or tuple(generation.get("eos_token_id", ())) != generation_eos \
            or generation.get("pad_token_id") != 154_820:
        raise Glm52AdapterError(
            "official generation stop-token contract mismatch"
        )
    expected = {
        "vocabulary_size": 154_856,
        "padded_model_vocabulary_size": 154_880,
        "eos_token_id": 154_820,
        "pad_token_id": 154_820,
        "model_max_length": 1_048_576,
    }
    observed = {
        "vocabulary_size": len(tokenizer),
        "padded_model_vocabulary_size": OFFICIAL_CONFIG_CONTRACT["vocab_size"],
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "model_max_length": tokenizer.model_max_length,
    }
    if observed != expected or tokenizer.chat_template != template:
        raise Glm52AdapterError(
            f"official tokenizer/chat-template contract mismatch: {observed}"
        )
    return OfficialTokenizerAssembly(
        root=control_root,
        tokenizer=tokenizer,
        asset_sha256=asset_hashes,
        generation_eos_token_ids=generation_eos,
        generation_pad_token_id=154_820,
        **expected,
    )


def _official_indexer_types() -> tuple[str, ...]:
    return tuple(
        "full" if layer in {0, 1, 2} or (layer >= 6 and (layer - 6) % 4 == 0) else "shared"
        for layer in range(78)
    )


OFFICIAL_INDEXER_TYPES = _official_indexer_types()
SYNTHETIC_INDEXER_TYPES = (
    "full", "full", "full", "shared", "shared", "shared", "full",
)

# Every architecture-affecting field is pinned.  Raw Hugging Face configs may
# contain additional serialization/token fields, but they cannot alter these.
OFFICIAL_CONFIG_CONTRACT: Mapping[str, Any] = {
    "architectures": [EXPECTED_ARCHITECTURE],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "ep_size": 1,
    "first_k_dense_replace": 3,
    "head_dim": 192,
    "hidden_act": "silu",
    "hidden_size": 6144,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_share_for_mtp_iteration": True,
    "index_skip_topk_offset": 3,
    "index_topk": 2048,
    "index_topk_freq": 4,
    "index_topk_pattern": None,
    "intermediate_size": 12288,
    "kv_lora_rank": 512,
    "max_position_embeddings": 1_048_576,
    "model_type": EXPECTED_MODEL_TYPE,
    "moe_intermediate_size": 2048,
    "moe_layer_freq": 1,
    "moe_router_dtype": "float32",
    "n_group": 1,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 78,
    "num_key_value_heads": 64,
    "num_nextn_predict_layers": 1,
    "pretraining_tp": 1,
    "q_lora_rank": 2048,
    "qk_head_dim": 256,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-5,
    "rope_interleave": True,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "tie_word_embeddings": False,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "use_cache": True,
    "v_head_dim": 256,
    "vocab_size": 154880,
}

# The deterministic twin is deliberately not "configurable tiny GLM".  A
# single exact profile keeps parity tests reproducible and prevents a fixture
# from silently losing an architecture feature.
SYNTHETIC_CONFIG_CONTRACT: Mapping[str, Any] = {
    "architectures": [EXPECTED_ARCHITECTURE],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "ep_size": 1,
    "first_k_dense_replace": 3,
    "head_dim": 8,
    "hidden_act": "silu",
    "hidden_size": 32,
    "index_head_dim": 4,
    "index_n_heads": 4,
    "index_share_for_mtp_iteration": True,
    "index_skip_topk_offset": 3,
    "index_topk": 2,
    "index_topk_freq": 4,
    "index_topk_pattern": None,
    "intermediate_size": 64,
    "kv_lora_rank": 8,
    "max_position_embeddings": 1_048_576,
    "model_type": EXPECTED_MODEL_TYPE,
    "moe_intermediate_size": 16,
    "moe_layer_freq": 1,
    "moe_router_dtype": "float32",
    "n_group": 1,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 4,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 7,
    "num_key_value_heads": 4,
    "num_nextn_predict_layers": 1,
    "pretraining_tp": 1,
    "q_lora_rank": 16,
    "qk_head_dim": 12,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 4,
    "rms_norm_eps": 1e-5,
    "rope_interleave": True,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "tie_word_embeddings": False,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "use_cache": True,
    "v_head_dim": 8,
    "vocab_size": 64,
}


@dataclass(frozen=True)
class Geometry:
    profile: str
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    first_k_dense_replace: int
    intermediate_size: int
    moe_intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    qk_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    scoring_func: str
    topk_method: str
    norm_topk_prob: bool
    indexer_types: tuple[str, ...]

    @property
    def mtp_layer(self) -> int:
        """Physical checkpoint layer containing the one MTP block."""
        return self.num_hidden_layers

    @property
    def physical_layer_count(self) -> int:
        return self.num_hidden_layers + 1

    def section_for_layer(self, layer: int) -> str:
        if 0 <= layer < self.num_hidden_layers:
            return CORE
        if layer == self.mtp_layer:
            return MTP
        raise Glm52AdapterError(
            f"layer {layer} is outside 0..{self.mtp_layer} (physical MTP boundary)"
        )

    def indexer_type(self, layer: int) -> str:
        self.section_for_layer(layer)
        return "full" if layer == self.mtp_layer else self.indexer_types[layer]

    def indexer_source_layer(self, layer: int) -> int:
        if self.indexer_type(layer) == "full":
            return layer
        for candidate in range(layer - 1, -1, -1):
            if self.indexer_types[candidate] == "full":
                return candidate
        raise Glm52AdapterError(f"shared IndexShare layer {layer} has no prior full indexer")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    organ: str
    section: str
    layer: int | None
    expert: int | None
    indexer_source_layer: int | None

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def byte_count(self) -> int:
        return self.element_count * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class TensorRecord:
    spec: TensorSpec
    shard: str
    relative_start: int
    relative_end: int
    absolute_start: int
    absolute_end: int

    @property
    def byte_count(self) -> int:
        return self.relative_end - self.relative_start


@dataclass(frozen=True)
class ShardRecord:
    name: str
    path: Path
    file_bytes: int
    header_bytes: int
    data_start: int
    payload_bytes: int
    header_sha256: str
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class Inventory:
    root: Path
    profile: str
    view: str
    geometry: Geometry
    config: Mapping[str, Any]
    index: Mapping[str, Any]
    tensors: Mapping[str, TensorRecord]
    shards: Mapping[str, ShardRecord]

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def payload_bytes(self) -> int:
        return sum(record.byte_count for record in self.tensors.values())

    @property
    def core_names(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, row in self.tensors.items() if row.spec.section == CORE))

    @property
    def mtp_names(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, row in self.tensors.items() if row.spec.section == MTP))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Glm52AdapterError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _loads_json_strict(data: bytes | str, *, source: str) -> Any:
    try:
        return json.loads(data, object_pairs_hook=_strict_object)
    except Glm52AdapterError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Glm52AdapterError(f"invalid JSON in {source}: {exc}") from exc


def load_json_strict(path: Path) -> Any:
    path = Path(path)
    return _loads_json_strict(path.read_bytes(), source=str(path))


def _contract_for_profile(profile: str) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    if profile == PROFILE_OFFICIAL:
        return OFFICIAL_CONFIG_CONTRACT, OFFICIAL_INDEXER_TYPES
    if profile == PROFILE_SYNTHETIC:
        return SYNTHETIC_CONFIG_CONTRACT, SYNTHETIC_INDEXER_TYPES
    raise Glm52AdapterError(f"unknown GLM adapter profile: {profile!r}")


def validate_config(config: Mapping[str, Any], *, profile: str = PROFILE_OFFICIAL) -> Geometry:
    """Validate a raw config against one exact architecture profile."""
    contract, expected_indexers = _contract_for_profile(profile)
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in contract.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise Glm52AdapterError(f"{profile} config contract mismatch: {mismatches}")

    actual_indexers = config.get("indexer_types")
    if not isinstance(actual_indexers, list) or tuple(actual_indexers) != expected_indexers:
        raise Glm52AdapterError(
            f"{profile} IndexShare pattern mismatch: expected={expected_indexers!r} "
            f"actual={actual_indexers!r}"
        )
    expected_mlp = tuple(
        "dense" if layer < int(contract["first_k_dense_replace"]) else "sparse"
        for layer in range(int(contract["num_hidden_layers"]))
    )
    actual_mlp = config.get("mlp_layer_types")
    if not isinstance(actual_mlp, list) or tuple(actual_mlp) != expected_mlp:
        raise Glm52AdapterError(
            f"{profile} dense/sparse layer pattern mismatch: expected={expected_mlp!r} "
            f"actual={actual_mlp!r}"
        )
    if int(config["qk_head_dim"]) != (
        int(config["qk_nope_head_dim"]) + int(config["qk_rope_head_dim"])
    ):
        raise Glm52AdapterError("qk_head_dim must equal qk_nope_head_dim + qk_rope_head_dim")
    if int(config["num_experts_per_tok"]) > int(config["n_routed_experts"]):
        raise Glm52AdapterError("num_experts_per_tok exceeds routed expert count")

    return Geometry(
        profile=profile,
        hidden_size=int(config["hidden_size"]),
        vocab_size=int(config["vocab_size"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        first_k_dense_replace=int(config["first_k_dense_replace"]),
        intermediate_size=int(config["intermediate_size"]),
        moe_intermediate_size=int(config["moe_intermediate_size"]),
        num_attention_heads=int(config["num_attention_heads"]),
        num_key_value_heads=int(config["num_key_value_heads"]),
        q_lora_rank=int(config["q_lora_rank"]),
        kv_lora_rank=int(config["kv_lora_rank"]),
        qk_nope_head_dim=int(config["qk_nope_head_dim"]),
        qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        qk_head_dim=int(config["qk_head_dim"]),
        v_head_dim=int(config["v_head_dim"]),
        index_n_heads=int(config["index_n_heads"]),
        index_head_dim=int(config["index_head_dim"]),
        index_topk=int(config["index_topk"]),
        n_routed_experts=int(config["n_routed_experts"]),
        n_shared_experts=int(config["n_shared_experts"]),
        num_experts_per_tok=int(config["num_experts_per_tok"]),
        routed_scaling_factor=float(config["routed_scaling_factor"]),
        scoring_func=str(config["scoring_func"]),
        topk_method=str(config["topk_method"]),
        norm_topk_prob=bool(config["norm_topk_prob"]),
        indexer_types=expected_indexers,
    )


def _top_spec(geometry: Geometry, name: str) -> TensorSpec | None:
    shapes = {
        "model.embed_tokens.weight": ((geometry.vocab_size, geometry.hidden_size), "embeddings"),
        "model.norm.weight": ((geometry.hidden_size,), "normalization"),
        "lm_head.weight": ((geometry.vocab_size, geometry.hidden_size), "lm_head"),
    }
    row = shapes.get(name)
    if row is None:
        return None
    shape, organ = row
    return TensorSpec(name, shape, "BF16", organ, CORE, None, None, None)


def tensor_spec(geometry: Geometry, name: str) -> TensorSpec:
    """Return the one legal schema row for ``name`` or fail closed."""
    top = _top_spec(geometry, name)
    if top is not None:
        return top

    match = _LAYER_RE.fullmatch(name)
    if match is None:
        raise Glm52AdapterError(f"unknown GLM tensor name: {name!r}")
    layer, suffix = int(match.group(1)), match.group(2)
    section = geometry.section_for_layer(layer)
    source = geometry.indexer_source_layer(layer)
    h = geometry.hidden_size

    normalizations = {
        "input_layernorm.weight": (h,),
        "post_attention_layernorm.weight": (h,),
        "self_attn.q_a_layernorm.weight": (geometry.q_lora_rank,),
        "self_attn.kv_a_layernorm.weight": (geometry.kv_lora_rank,),
    }
    if suffix in normalizations:
        return TensorSpec(name, normalizations[suffix], "BF16", "normalization", section,
                          layer, None, source)

    attention = {
        "self_attn.q_a_proj.weight": (geometry.q_lora_rank, h),
        "self_attn.q_b_proj.weight": (
            geometry.num_attention_heads * geometry.qk_head_dim,
            geometry.q_lora_rank,
        ),
        "self_attn.kv_a_proj_with_mqa.weight": (
            geometry.kv_lora_rank + geometry.qk_rope_head_dim,
            h,
        ),
        "self_attn.kv_b_proj.weight": (
            geometry.num_attention_heads * (geometry.qk_nope_head_dim + geometry.v_head_dim),
            geometry.kv_lora_rank,
        ),
        "self_attn.o_proj.weight": (
            h,
            geometry.num_attention_heads * geometry.v_head_dim,
        ),
    }
    if suffix in attention:
        return TensorSpec(name, attention[suffix], "BF16", "attention", section,
                          layer, None, source)

    indexer = {
        "self_attn.indexer.k_norm.bias": (geometry.index_head_dim,),
        "self_attn.indexer.k_norm.weight": (geometry.index_head_dim,),
        "self_attn.indexer.weights_proj.weight": (geometry.index_n_heads, h),
        "self_attn.indexer.wk.weight": (geometry.index_head_dim, h),
        "self_attn.indexer.wq_b.weight": (
            geometry.index_n_heads * geometry.index_head_dim,
            geometry.q_lora_rank,
        ),
    }
    if suffix in indexer:
        if geometry.indexer_type(layer) != "full":
            raise Glm52AdapterError(
                f"shared IndexShare layer {layer} must omit stored indexer tensor {name!r}"
            )
        return TensorSpec(name, indexer[suffix], "BF16", "indexer", section,
                          layer, None, source)

    if suffix in {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}:
        if section == MTP or layer >= geometry.first_k_dense_replace:
            raise Glm52AdapterError(f"dense MLP tensor appears on sparse/MTP layer: {name!r}")
        if suffix == "mlp.down_proj.weight":
            shape = (h, geometry.intermediate_size)
        else:
            shape = (geometry.intermediate_size, h)
        return TensorSpec(name, shape, "BF16", "dense_mlp", section, layer, None, source)

    expert_match = _EXPERT_RE.fullmatch(suffix)
    if expert_match is not None:
        expert, projection = int(expert_match.group(1)), expert_match.group(2)
        if section == CORE and layer < geometry.first_k_dense_replace:
            raise Glm52AdapterError(f"routed expert appears on dense layer: {name!r}")
        if not 0 <= expert < geometry.n_routed_experts:
            raise Glm52AdapterError(f"expert id outside 0..{geometry.n_routed_experts - 1}: {name!r}")
        shape = ((h, geometry.moe_intermediate_size) if projection == "down_proj"
                 else (geometry.moe_intermediate_size, h))
        return TensorSpec(name, shape, "BF16", "routed_expert", section,
                          layer, expert, source)

    if suffix in {
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight",
        "mlp.shared_experts.down_proj.weight",
    }:
        if section == CORE and layer < geometry.first_k_dense_replace:
            raise Glm52AdapterError(f"shared expert appears on dense layer: {name!r}")
        shape = ((h, geometry.moe_intermediate_size) if suffix.endswith("down_proj.weight")
                 else (geometry.moe_intermediate_size, h))
        return TensorSpec(name, shape, "BF16", "shared_expert", section,
                          layer, None, source)

    if suffix == "mlp.gate.weight":
        if section == CORE and layer < geometry.first_k_dense_replace:
            raise Glm52AdapterError(f"router appears on dense layer: {name!r}")
        return TensorSpec(name, (geometry.n_routed_experts, h), "BF16", "router",
                          section, layer, None, source)
    if suffix == "mlp.gate.e_score_correction_bias":
        if section == CORE and layer < geometry.first_k_dense_replace:
            raise Glm52AdapterError(f"router correction appears on dense layer: {name!r}")
        return TensorSpec(name, (geometry.n_routed_experts,), "F32", "router_control",
                          section, layer, None, source)

    if section == MTP:
        mtp_shapes = {
            "eh_proj.weight": ((h, 2 * h), "mtp_projection"),
            "enorm.weight": ((h,), "mtp_normalization"),
            "hnorm.weight": ((h,), "mtp_normalization"),
            "shared_head.norm.weight": ((h,), "mtp_head_norm"),
        }
        mtp = mtp_shapes.get(suffix)
        if mtp is not None:
            shape, organ = mtp
            return TensorSpec(name, shape, "BF16", organ, MTP, layer, None, source)

    raise Glm52AdapterError(f"unknown or misplaced GLM tensor name: {name!r}")


def _layer_names(geometry: Geometry, layer: int) -> Iterable[str]:
    base = f"model.layers.{layer}."
    for suffix in (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight",
    ):
        yield base + suffix
    if geometry.indexer_type(layer) == "full":
        for suffix in (
            "self_attn.indexer.k_norm.bias",
            "self_attn.indexer.k_norm.weight",
            "self_attn.indexer.weights_proj.weight",
            "self_attn.indexer.wk.weight",
            "self_attn.indexer.wq_b.weight",
        ):
            yield base + suffix

    if layer < geometry.first_k_dense_replace and layer < geometry.num_hidden_layers:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            yield f"{base}mlp.{projection}.weight"
    else:
        yield f"{base}mlp.gate.weight"
        yield f"{base}mlp.gate.e_score_correction_bias"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            yield f"{base}mlp.shared_experts.{projection}.weight"
        for expert in range(geometry.n_routed_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                yield f"{base}mlp.experts.{expert}.{projection}.weight"

    if layer == geometry.mtp_layer:
        for suffix in ("eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight"):
            yield base + suffix


def expected_tensor_specs(geometry: Geometry, *, view: str = VIEW_FULL) -> dict[str, TensorSpec]:
    """Generate the exact full or CORE-only tensor schema for a validated profile."""
    if view not in {VIEW_FULL, VIEW_CORE}:
        raise Glm52AdapterError(f"unknown checkpoint view: {view!r}")
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer in range(geometry.num_hidden_layers):
        names.extend(_layer_names(geometry, layer))
    if view == VIEW_FULL:
        names.extend(_layer_names(geometry, geometry.mtp_layer))
    if len(names) != len(set(names)):
        raise Glm52AdapterError("internal schema generator produced duplicate tensor names")
    specs = {name: tensor_spec(geometry, name) for name in names}
    expected_count = 59_585 if geometry.profile == PROFILE_OFFICIAL else 3_978
    if view == VIEW_CORE:
        expected_count -= 791
    if len(specs) != expected_count:
        raise Glm52AdapterError(
            f"internal {geometry.profile}/{view} tensor census drift: "
            f"expected={expected_count} actual={len(specs)}"
        )
    return specs


def load_index(path: Path) -> dict[str, Any]:
    index = load_json_strict(path)
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
        raise Glm52AdapterError("safetensors index must contain exactly metadata and weight_map")
    metadata, weight_map = index["metadata"], index["weight_map"]
    if not isinstance(metadata, dict) or set(metadata) != {"total_size"}:
        raise Glm52AdapterError("index metadata must contain exactly total_size")
    if not isinstance(metadata["total_size"], int) or metadata["total_size"] <= 0:
        raise Glm52AdapterError("index metadata.total_size must be a positive integer")
    if not isinstance(weight_map, dict) or not weight_map:
        raise Glm52AdapterError("index weight_map must be a non-empty object")
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise Glm52AdapterError("index weight_map names and shards must be strings")
        if _SAFE_SHARD_RE.fullmatch(shard) is None or Path(shard).name != shard:
            raise Glm52AdapterError(f"unsafe shard name in index: {shard!r}")
    return index


def _open_regular_no_follow(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise Glm52AdapterError(f"cannot safely open shard {path}: {exc}") from exc
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise Glm52AdapterError(f"shard is not a regular file: {path}")
    return fd


def _pread_exact(fd: int, count: int, offset: int, *, context: str) -> bytes:
    data = os.pread(fd, count, offset)
    if len(data) != count:
        raise Glm52AdapterError(
            f"short bounded read for {context}: expected={count} actual={len(data)}"
        )
    return data


def _parse_shard(
    path: Path,
    shard_name: str,
    expected_names: set[str],
    expected_specs: Mapping[str, TensorSpec],
) -> tuple[ShardRecord, dict[str, TensorRecord]]:
    fd = _open_regular_no_follow(path)
    try:
        st = os.fstat(fd)
        prefix = _pread_exact(fd, 8, 0, context=f"{shard_name} length prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_bytes <= MAX_HEADER_BYTES or 8 + header_bytes > st.st_size:
            raise Glm52AdapterError(f"unsafe safetensors header length in {shard_name}: {header_bytes}")
        header_raw = _pread_exact(fd, header_bytes, 8, context=f"{shard_name} header")
        header = _loads_json_strict(header_raw, source=shard_name)
        if not isinstance(header, dict):
            raise Glm52AdapterError(f"safetensors header is not an object: {shard_name}")
        metadata = header.pop("__metadata__", None)
        if metadata not in (None, {}):
            raise Glm52AdapterError(f"unexpected safetensors metadata in {shard_name}: {metadata!r}")
        actual_names = set(header)
        if actual_names != expected_names:
            raise Glm52AdapterError(
                f"index/header tensor set mismatch in {shard_name}: "
                f"missing={sorted(expected_names - actual_names)[:5]} "
                f"unknown={sorted(actual_names - expected_names)[:5]}"
            )

        data_start = 8 + header_bytes
        records: dict[str, TensorRecord] = {}
        extents: list[tuple[int, int, str]] = []
        for name, entry in header.items():
            if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
                raise Glm52AdapterError(f"invalid safetensors entry schema: {shard_name}:{name}")
            dtype, shape_raw, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
            if dtype not in DTYPE_BYTES:
                raise Glm52AdapterError(f"unsupported dtype {dtype!r}: {shard_name}:{name}")
            if (
                not isinstance(shape_raw, list)
                or not shape_raw
                or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape_raw)
            ):
                raise Glm52AdapterError(f"invalid positive shape: {shard_name}:{name}:{shape_raw!r}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
            ):
                raise Glm52AdapterError(f"invalid data offsets: {shard_name}:{name}:{offsets!r}")
            shape = tuple(shape_raw)
            start, end = offsets
            expected_extent = math.prod(shape) * DTYPE_BYTES[dtype]
            if start < 0 or end <= start or end - start != expected_extent:
                raise Glm52AdapterError(
                    f"shape/dtype extent mismatch: {shard_name}:{name}: "
                    f"shape={shape} dtype={dtype} offsets={offsets}"
                )
            if data_start + end > st.st_size:
                raise Glm52AdapterError(f"tensor exceeds shard EOF: {shard_name}:{name}")
            spec = expected_specs[name]
            if shape != spec.shape or dtype != spec.dtype:
                raise Glm52AdapterError(
                    f"schema mismatch for {name}: expected={spec.dtype}{spec.shape} "
                    f"actual={dtype}{shape}"
                )
            records[name] = TensorRecord(
                spec=spec,
                shard=shard_name,
                relative_start=start,
                relative_end=end,
                absolute_start=data_start + start,
                absolute_end=data_start + end,
            )
            extents.append((start, end, name))

        cursor = 0
        for start, end, name in sorted(extents):
            if start != cursor:
                relation = "overlap" if start < cursor else "gap"
                raise Glm52AdapterError(
                    f"safetensors payload {relation} before {shard_name}:{name}: "
                    f"cursor={cursor} start={start}"
                )
            cursor = end
        if data_start + cursor != st.st_size:
            raise Glm52AdapterError(
                f"safetensors payload does not close at EOF: {shard_name}: "
                f"data_start={data_start} payload={cursor} file={st.st_size}"
            )
        return (
            ShardRecord(
                name=shard_name,
                path=path.resolve(),
                file_bytes=st.st_size,
                header_bytes=header_bytes,
                data_start=data_start,
                payload_bytes=cursor,
                header_sha256=hashlib.sha256(prefix + header_raw).hexdigest(),
                device=st.st_dev,
                inode=st.st_ino,
                mtime_ns=st.st_mtime_ns,
            ),
            records,
        )
    finally:
        os.close(fd)


def verify_checkpoint(
    root: Path,
    *,
    profile: str = PROFILE_OFFICIAL,
    view: str = VIEW_FULL,
) -> Inventory:
    """Validate config, index, every shard header, and the complete tensor census."""
    root = Path(root).resolve()
    config = load_json_strict(root / "config.json")
    if not isinstance(config, dict):
        raise Glm52AdapterError("config.json must contain a JSON object")
    geometry = validate_config(config, profile=profile)
    expected_specs = expected_tensor_specs(geometry, view=view)
    index = load_index(root / "model.safetensors.index.json")
    weight_map: dict[str, str] = index["weight_map"]
    indexed_names = set(weight_map)
    expected_names = set(expected_specs)
    if indexed_names != expected_names:
        raise Glm52AdapterError(
            f"checkpoint tensor census mismatch: "
            f"missing={sorted(expected_names - indexed_names)[:5]} "
            f"unknown={sorted(indexed_names - expected_names)[:5]}"
        )
    expected_payload_bytes = sum(spec.byte_count for spec in expected_specs.values())
    if index["metadata"]["total_size"] != expected_payload_bytes:
        raise Glm52AdapterError(
            "complete source index total_size mismatch before window admission: "
            f"index={index['metadata']['total_size']} expected={expected_payload_bytes}"
        )

    shard_names = set(weight_map.values())
    disk_shards = {path.name for path in root.glob("*.safetensors")}
    if disk_shards != shard_names:
        raise Glm52AdapterError(
            f"checkpoint shard set mismatch: missing={sorted(shard_names - disk_shards)} "
            f"unindexed={sorted(disk_shards - shard_names)}"
        )
    by_shard: dict[str, set[str]] = {name: set() for name in shard_names}
    for name, shard in weight_map.items():
        by_shard[shard].add(name)

    shards: dict[str, ShardRecord] = {}
    tensors: dict[str, TensorRecord] = {}
    for shard_name in sorted(shard_names):
        shard, records = _parse_shard(
            root / shard_name, shard_name, by_shard[shard_name], expected_specs
        )
        overlap = set(tensors).intersection(records)
        if overlap:
            raise Glm52AdapterError(f"duplicate tensors across shards: {sorted(overlap)[:5]}")
        shards[shard_name] = shard
        tensors.update(records)
    if set(tensors) != expected_names:
        raise Glm52AdapterError("global shard/header tensor census differs from schema")
    payload_bytes = sum(row.byte_count for row in tensors.values())
    if payload_bytes != index["metadata"]["total_size"]:
        raise Glm52AdapterError(
            f"index total_size mismatch: index={index['metadata']['total_size']} "
            f"headers={payload_bytes}"
        )
    if sum(shard.payload_bytes for shard in shards.values()) != payload_bytes:
        raise Glm52AdapterError("per-shard and per-tensor payload totals differ")
    return Inventory(
        root=root,
        profile=profile,
        view=view,
        geometry=geometry,
        config=config,
        index=index,
        tensors=tensors,
        shards=shards,
    )


def _streaming_window_shards(
    window_contract: Mapping[str, Any],
    *,
    indexed_shards: set[str],
) -> set[str]:
    """Validate the immutable shard-set algebra for one dependency window."""
    if not isinstance(window_contract, Mapping):
        raise Glm52AdapterError("streaming window contract must be an object")
    window_id = window_contract.get("window_id")
    if not isinstance(window_id, str) or not window_id or window_id != window_id.strip():
        raise Glm52AdapterError("streaming window_id must be a non-empty trimmed string")

    values: dict[str, set[str]] = {}
    for key in (
        "source_shards",
        "carry_in_shards",
        "new_fetch_shards",
        "refetch_shards",
        "carry_out_shards",
        "evict_after_seal_shards",
    ):
        raw = window_contract.get(key)
        if not isinstance(raw, list):
            raise Glm52AdapterError(f"{window_id}.{key} must be a list")
        if any(
            not isinstance(item, str)
            or _SAFE_SHARD_RE.fullmatch(item) is None
            or Path(item).name != item
            for item in raw
        ):
            raise Glm52AdapterError(f"{window_id}.{key} contains an unsafe shard name")
        if len(raw) != len(set(raw)):
            raise Glm52AdapterError(f"{window_id}.{key} contains a duplicate shard")
        values[key] = set(raw)

    source = values["source_shards"]
    carry_in = values["carry_in_shards"]
    new_fetch = values["new_fetch_shards"]
    refetch = values["refetch_shards"]
    carry_out = values["carry_out_shards"]
    evict = values["evict_after_seal_shards"]
    if not source:
        raise Glm52AdapterError(f"{window_id}.source_shards cannot be empty")
    acquisition_sets = (carry_in, new_fetch, refetch)
    if any(
        acquisition_sets[left] & acquisition_sets[right]
        for left in range(len(acquisition_sets))
        for right in range(left + 1, len(acquisition_sets))
    ):
        raise Glm52AdapterError(
            f"{window_id} carry/new-fetch/refetch shard sets must be disjoint"
        )
    if source != carry_in | new_fetch | refetch:
        raise Glm52AdapterError(
            f"{window_id}.source_shards must equal carry-in + new-fetch + refetch"
        )
    if not carry_out.issubset(source):
        raise Glm52AdapterError(f"{window_id}.carry_out_shards is not resident")
    if evict != source - carry_out:
        raise Glm52AdapterError(
            f"{window_id}.evict_after_seal_shards must equal source minus carry-out"
        )
    unknown = source - indexed_shards
    if unknown:
        raise Glm52AdapterError(
            f"{window_id} references shards outside the complete official index: "
            f"{sorted(unknown)[:3]}"
        )
    return source


def verify_streaming_window(
    control_root: Path,
    hydrated_root: Path,
    window_contract: Mapping[str, Any],
    *,
    profile: str = PROFILE_OFFICIAL,
    view: str = VIEW_FULL,
) -> Inventory:
    """Validate one hydrated window while retaining the complete source census.

    ``control_root`` contains the immutable config and full checkpoint index.
    ``hydrated_root`` is a dedicated scratch directory containing exactly the
    resident source shards declared by ``window_contract``.  The global index
    must still match the entire expected model schema; a partial or rewritten
    index is rejected even if the omitted tensor is outside this window.
    """
    control_root = Path(control_root).resolve()
    hydrated_root = Path(hydrated_root).resolve()
    config = load_json_strict(control_root / "config.json")
    if not isinstance(config, dict):
        raise Glm52AdapterError("config.json must contain a JSON object")
    geometry = validate_config(config, profile=profile)
    expected_specs = expected_tensor_specs(geometry, view=view)
    index = load_index(control_root / "model.safetensors.index.json")
    weight_map: dict[str, str] = index["weight_map"]

    expected_names = set(expected_specs)
    indexed_names = set(weight_map)
    if indexed_names != expected_names:
        raise Glm52AdapterError(
            "complete source index tensor census mismatch before window admission: "
            f"missing={sorted(expected_names - indexed_names)[:5]} "
            f"unknown={sorted(indexed_names - expected_names)[:5]}"
        )
    expected_payload_bytes = sum(spec.byte_count for spec in expected_specs.values())
    if index["metadata"]["total_size"] != expected_payload_bytes:
        raise Glm52AdapterError(
            "complete source index total_size mismatch before window admission: "
            f"index={index['metadata']['total_size']} expected={expected_payload_bytes}"
        )
    indexed_shards = set(weight_map.values())
    resident = _streaming_window_shards(
        window_contract, indexed_shards=indexed_shards
    )
    disk_shards = {path.name for path in hydrated_root.glob("*.safetensors")}
    if disk_shards != resident:
        raise Glm52AdapterError(
            "hydrated streaming shard set mismatch: "
            f"missing={sorted(resident - disk_shards)} "
            f"unexpected={sorted(disk_shards - resident)}"
        )

    by_shard: dict[str, set[str]] = {name: set() for name in resident}
    for name, shard in weight_map.items():
        if shard in resident:
            by_shard[shard].add(name)
    expected_window_names = {
        name for name, shard in weight_map.items() if shard in resident
    }
    shards: dict[str, ShardRecord] = {}
    tensors: dict[str, TensorRecord] = {}
    for shard_name in sorted(resident):
        shard, records = _parse_shard(
            hydrated_root / shard_name,
            shard_name,
            by_shard[shard_name],
            expected_specs,
        )
        overlap = set(tensors).intersection(records)
        if overlap:
            raise Glm52AdapterError(
                f"duplicate tensors across streaming shards: {sorted(overlap)[:5]}"
            )
        shards[shard_name] = shard
        tensors.update(records)
    if set(tensors) != expected_window_names:
        raise Glm52AdapterError("streaming window shard/header tensor census differs from index")
    if sum(shard.payload_bytes for shard in shards.values()) != sum(
        record.byte_count for record in tensors.values()
    ):
        raise Glm52AdapterError("streaming window payload totals differ")
    return Inventory(
        root=hydrated_root,
        profile=profile,
        view=view,
        geometry=geometry,
        config=config,
        index=index,
        tensors=tensors,
        shards=shards,
    )


class BoundedSafetensorsReader:
    """Read one validated tensor at a time using its exact shard-local range."""

    def __init__(
        self,
        inventory: Inventory,
        *,
        max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
    ) -> None:
        if (
            not isinstance(max_tensor_bytes, int)
            or isinstance(max_tensor_bytes, bool)
            or max_tensor_bytes <= 0
        ):
            raise Glm52AdapterError("max_tensor_bytes must be a positive integer")
        self.inventory = inventory
        self.max_tensor_bytes = max_tensor_bytes
        self.payload_bytes_read = 0
        self.read_calls = 0
        self.peak_payload_request_bytes = 0
        self._revalidated_shards: set[str] = set()

    def _open_validated_shard(self, shard: ShardRecord) -> int:
        fd = _open_regular_no_follow(shard.path)
        st = os.fstat(fd)
        observed = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        expected = (shard.device, shard.inode, shard.file_bytes, shard.mtime_ns)
        if observed != expected:
            os.close(fd)
            raise Glm52AdapterError(
                f"shard identity changed after validation: {shard.name}: "
                f"expected={expected} actual={observed}"
            )
        if shard.name not in self._revalidated_shards:
            raw = _pread_exact(
                fd, 8 + shard.header_bytes, 0, context=f"{shard.name} header revalidation"
            )
            if hashlib.sha256(raw).hexdigest() != shard.header_sha256:
                os.close(fd)
                raise Glm52AdapterError(f"shard header changed after validation: {shard.name}")
            self._revalidated_shards.add(shard.name)
        return fd

    def raw(self, name: str, *, max_bytes: int | None = None) -> bytes:
        """Return exactly one tensor payload, never a prefix or an entire shard."""
        record = self.inventory.tensors.get(name)
        if record is None:
            raise Glm52AdapterError(f"tensor is absent from validated inventory: {name!r}")
        if max_bytes is not None and (
            not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0
        ):
            raise Glm52AdapterError("max_bytes must be a positive integer")
        limit = self.max_tensor_bytes if max_bytes is None else min(self.max_tensor_bytes, max_bytes)
        if record.byte_count > limit:
            raise Glm52AdapterError(
                f"bounded read refused {name}: tensor={record.byte_count} limit={limit}"
            )
        shard = self.inventory.shards[record.shard]
        fd = self._open_validated_shard(shard)
        try:
            payload = _pread_exact(
                fd, record.byte_count, record.absolute_start, context=f"tensor {name}"
            )
        finally:
            os.close(fd)
        self.payload_bytes_read += len(payload)
        self.read_calls += 1
        self.peak_payload_request_bytes = max(self.peak_payload_request_bytes, len(payload))
        return payload

    def tensor(self, name: str, *, max_bytes: int | None = None) -> np.ndarray:
        """Return one BF16/F32 source tensor decoded to an owned float32 array."""
        record = self.inventory.tensors.get(name)
        if record is None:
            raise Glm52AdapterError(f"tensor is absent from validated inventory: {name!r}")
        payload = self.raw(name, max_bytes=max_bytes)
        if record.spec.dtype == "BF16":
            words = np.frombuffer(payload, dtype="<u2")
            values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
        elif record.spec.dtype == "F32":
            values = np.frombuffer(payload, dtype="<f4")
        else:  # unreachable after header validation; retained as a fail-closed guard.
            raise Glm52AdapterError(f"unsupported validated dtype: {record.spec.dtype}")
        return values.reshape(record.spec.shape).copy()


def expert_gate_up_names(layer: int, expert: int) -> tuple[str, str]:
    base = f"model.layers.{layer}.mlp.experts.{expert}."
    return tuple(base + projection + ".weight" for projection in GATE_UP_ORDER)  # type: ignore[return-value]


def pack_expert_gate_up(
    reader: BoundedSafetensorsReader,
    layer: int,
    expert: int,
    *,
    max_bytes_per_tensor: int | None = None,
) -> np.ndarray:
    """Pack a routed expert as ``[gate rows; up rows]`` with no interleaving.

    This order matches GLM's SiLU(gate) * up convention.  Changing it is a
    semantic corruption even though gate and up have identical shapes.
    """
    gate_name, up_name = expert_gate_up_names(layer, expert)
    gate_record = reader.inventory.tensors.get(gate_name)
    up_record = reader.inventory.tensors.get(up_name)
    if gate_record is None or up_record is None:
        raise Glm52AdapterError(
            f"cannot pack missing routed expert gate/up pair: layer={layer} expert={expert}"
        )
    for record, projection in ((gate_record, "gate_proj"), (up_record, "up_proj")):
        if record.spec.organ != "routed_expert" or projection not in record.spec.name:
            raise Glm52AdapterError("routed expert gate/up schema identity changed")
    gate = reader.tensor(gate_name, max_bytes=max_bytes_per_tensor)
    up = reader.tensor(up_name, max_bytes=max_bytes_per_tensor)
    if gate.shape != up.shape:
        raise Glm52AdapterError(f"gate/up shape mismatch: {gate.shape} != {up.shape}")
    return np.concatenate((gate, up), axis=0)


def split_expert_gate_up(packed: np.ndarray, geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    packed = np.asarray(packed, dtype=np.float32)
    expected = (2 * geometry.moe_intermediate_size, geometry.hidden_size)
    if packed.shape != expected:
        raise Glm52AdapterError(f"packed gate/up shape mismatch: expected={expected} actual={packed.shape}")
    split = geometry.moe_intermediate_size
    return packed[:split].copy(), packed[split:].copy()


def schema_summary(geometry: Geometry) -> dict[str, Any]:
    """Small deterministic receipt useful to downstream controllers/tests."""
    full = expected_tensor_specs(geometry, view=VIEW_FULL)
    core = {name: spec for name, spec in full.items() if spec.section == CORE}
    mtp = {name: spec for name, spec in full.items() if spec.section == MTP}
    return {
        "profile": geometry.profile,
        "main_layers": geometry.num_hidden_layers,
        "physical_layers": geometry.physical_layer_count,
        "mtp_physical_layer": geometry.mtp_layer,
        "full_tensor_count": len(full),
        "core_tensor_count": len(core),
        "mtp_tensor_count": len(mtp),
        "core_logical_elements": sum(spec.element_count for spec in core.values()),
        "mtp_logical_elements": sum(spec.element_count for spec in mtp.values()),
        "indexer_types_main": list(geometry.indexer_types),
        "mtp_indexer_type": geometry.indexer_type(geometry.mtp_layer),
        "index_topk": geometry.index_topk,
        "routed_expert_count": geometry.n_routed_experts,
        "shared_expert_count": geometry.n_shared_experts,
        "experts_per_token": geometry.num_experts_per_tok,
        "router_selection": (
            f"top-{geometry.num_experts_per_tok}-of-{geometry.n_routed_experts}"
        ),
        "router_weight_sum_after_scaling": geometry.routed_scaling_factor,
        "router_scoring_function": geometry.scoring_func,
        "router_topk_method": geometry.topk_method,
        "router_normalizes_topk_probability": geometry.norm_topk_prob,
        "gate_up_order": list(GATE_UP_ORDER),
        "source_dtypes": sorted({spec.dtype for spec in full.values()}),
    }


__all__ = [
    "BoundedSafetensorsReader",
    "CORE",
    "DEFAULT_MAX_TENSOR_BYTES",
    "DTYPE_BYTES",
    "GATE_UP_ORDER",
    "Geometry",
    "Glm52AdapterError",
    "IMMUTABLE_REVISION",
    "Inventory",
    "MTP",
    "OFFICIAL_CONFIG_CONTRACT",
    "OFFICIAL_INDEXER_TYPES",
    "OFFICIAL_TOKENIZER_ASSETS",
    "OfficialTokenizerAssembly",
    "PROFILE_OFFICIAL",
    "PROFILE_SYNTHETIC",
    "REPO_ID",
    "SYNTHETIC_CONFIG_CONTRACT",
    "SYNTHETIC_INDEXER_TYPES",
    "TensorRecord",
    "TensorSpec",
    "VIEW_CORE",
    "VIEW_FULL",
    "expected_tensor_specs",
    "expert_gate_up_names",
    "load_index",
    "load_json_strict",
    "load_official_tokenizer_assembly",
    "pack_expert_gate_up",
    "schema_summary",
    "split_expert_gate_up",
    "tensor_spec",
    "validate_config",
    "verify_checkpoint",
    "verify_streaming_window",
]
