"""Provider- and architecture-neutral model structure recognition.

The recognizer intentionally reads metadata only.  It never loads weights or
claims that a tensor pattern is a working kernel.  Its output is a planning
map for Doctor/Gravity/Accelerator and a provenance record for AgentOS.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = "hcli.architecture.recognizer.v1"
_MAX_METADATA_BYTES = 32 * 1024 * 1024
_MAX_TENSORS = 250_000


def _sha256(path: Path) -> Optional[str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        if path.stat().st_size > _MAX_METADATA_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _nested_values(config: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield the root and common nested text/config mappings."""
    yield config
    for key in ("text_config", "language_config", "vision_config", "model_config", "config"):
        value = config.get(key)
        if isinstance(value, Mapping):
            yield value


def _first_number(config: Mapping[str, Any], keys: Sequence[str]) -> Optional[int]:
    for current in _nested_values(config):
        for key in keys:
            value = current.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _first_text(config: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for current in _nested_values(config):
        for key in keys:
            value = current.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return None


def _profile_metadata(source: Path, value: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract profile-declared architecture without treating it as truth."""
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    architecture = metadata.get("architecture")
    result: Dict[str, Any] = {}
    if isinstance(architecture, Mapping):
        result.update(dict(architecture))
    for key in ("model_type", "architectures", "hidden_size", "num_hidden_layers", "num_experts", "num_experts_per_tok"):
        if key in value and key not in result:
            result[key] = value[key]
    for key in ("model_id", "provider", "profile_id"):
        if key in value and key not in result:
            result[key] = value[key]
    artifact = value.get("artifact_root")
    if artifact:
        artifact_path = Path(str(artifact)).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = (source.parent / artifact_path).resolve(strict=False)
        result["declared_artifact_root"] = str(artifact_path)
    return result


def _source_candidates(source: Path) -> Tuple[List[Path], List[Path], Dict[str, Any]]:
    configs: List[Path] = []
    indexes: List[Path] = []
    profile: Dict[str, Any] = {}
    if source.is_file():
        if source.name in {"config.json", "configuration.json"}:
            configs.append(source)
        elif source.name.endswith(".index.json") or source.name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}:
            indexes.append(source)
        elif source.suffix.lower() == ".json":
            value = _load_json(source)
            if isinstance(value, Mapping) and ("artifact_root" in value or "runtime" in value or "profile_id" in value):
                profile = _profile_metadata(source, value)
                artifact = profile.get("declared_artifact_root")
                if artifact:
                    source = Path(str(artifact))
            elif isinstance(value, Mapping) and "weight_map" in value:
                indexes.append(source)
    if source.is_dir():
        for name in ("config.json", "configuration.json"):
            candidate = source / name
            if candidate.is_file():
                configs.append(candidate)
                break
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json", "model.index.json"):
            candidate = source / name
            if candidate.is_file():
                indexes.append(candidate)
                break
    return configs, indexes, profile


_LAYER_RE = re.compile(r"(?:^|[./_])(?:layers?|h|blocks?|decoder\.layers|encoder\.layer)[./_](\d+)(?:$|[./_])", re.I)
_ORGAN_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("embedding", ("embed", "embedding", "wte", "tok_embeddings")),
    ("attention", ("attention", "self_attn", "attn", "q_proj", "k_proj", "v_proj", "o_proj")),
    ("recurrent_or_deltanet", ("deltanet", "delta_net", "recurrent", "retention", "gated_delta")),
    ("moe_router", ("router", "gate", "moe_gate", "switch")),
    ("moe_experts", ("expert", "experts", "mlp")),
    ("shared_expert", ("shared_expert", r"shared\.expert")),
    ("normalization", ("norm", "layernorm", "rmsnorm")),
    ("ngram_or_lookup", ("ngram", "lookup", "codebook", "embedding_table")),
    ("mtp_or_auxiliary_head", ("mtp", "multi_token", "aux_head", "auxiliary")),
    ("output_head", ("lm_head", "output", "classifier", "logits")),
)


@dataclass(frozen=True)
class ArchitectureReport:
    """Serializable metadata-only architecture result."""

    document: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.document)


class ArchitectureRecognizer:
    """Recognize model organs from config/index metadata."""

    def __init__(self, *, max_tensors: int = _MAX_TENSORS) -> None:
        self.max_tensors = max(1, min(int(max_tensors), _MAX_TENSORS))

    def inspect(self, source: str | os.PathLike[str]) -> Dict[str, Any]:
        requested = Path(source).expanduser().resolve(strict=False)
        configs, indexes, profile = _source_candidates(requested)
        config_path = configs[0] if configs else None
        config = _load_json(config_path) if config_path else {}
        if not isinstance(config, Mapping):
            config = {}
        index_path = indexes[0] if indexes else None
        index = _load_json(index_path) if index_path else {}
        if not isinstance(index, Mapping):
            index = {}
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping):
            weight_map = {}
        names = [str(name) for name in list(weight_map.keys())[: self.max_tensors]]

        model_type = _first_text(config, ("model_type", "architectures", "model_class"))
        architectures = _first_text(config, ("architectures",))
        layers = sorted({int(match.group(1)) for name in names if (match := _LAYER_RE.search(name))})
        hidden_size = _first_number(config, ("hidden_size", "d_model", "dim", "n_embd"))
        layer_count = _first_number(config, ("num_hidden_layers", "n_layer", "n_layers", "num_layers"))
        experts = _first_number(config, ("num_experts", "n_routed_experts", "num_local_experts"))
        active_experts = _first_number(config, ("num_experts_per_tok", "num_selected_experts", "top_k", "num_experts_per_token"))
        context_length = _first_number(config, ("max_position_embeddings", "max_sequence_length", "context_length"))
        vocab_size = _first_number(config, ("vocab_size", "n_vocab"))

        organs: List[Dict[str, Any]] = []
        lower_names = [(name, name.lower()) for name in names]
        for organ, signals in _ORGAN_PATTERNS:
            matched = [name for name, lower in lower_names if any(signal in lower for signal in signals)]
            organs.append({
                "id": organ,
                "present": bool(matched),
                "tensor_count": len(matched),
                "examples": matched[:8],
                "signals": list(signals),
                "confidence": "high" if matched and index_path else "medium" if matched else "unknown",
            })

        unresolved: List[str] = []
        if not config_path:
            unresolved.append("config metadata unavailable")
        if not index_path:
            unresolved.append("tensor index unavailable; organ counts are incomplete")
        if not names:
            unresolved.append("no tensor names available")
        if layer_count is None and layers:
            layer_count = max(layers) + 1
        if layer_count is None:
            unresolved.append("layer count unresolved")
        if experts is not None and active_experts is not None and active_experts > experts:
            unresolved.append("active expert count exceeds total expert count")

        source_records = []
        for path in [config_path, index_path]:
            if path is None:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            source_records.append({
                "path": str(path),
                "kind": "config" if path == config_path else "tensor_index",
                "bytes": size,
                "sha256": _sha256(path),
            })

        confidence = "high" if config_path and index_path else "medium" if config_path or index_path else "low"
        if unresolved and confidence == "high":
            confidence = "medium"
        model_id = str(config.get("_name_or_path") or profile.get("model_id") or requested.name)
        document: Dict[str, Any] = {
            "schema": SCHEMA,
            "source": str(requested),
            "model_id": model_id,
            "architecture": {
                "model_type": model_type,
                "architectures": config.get("architectures") if isinstance(config.get("architectures"), list) else ([architectures] if architectures else []),
                "hidden_size": hidden_size,
                "layers": layer_count,
                "observed_layer_indices": layers[: self.max_tensors],
                "experts": experts,
                "active_experts_per_token": active_experts,
                "context_length": context_length,
                "vocab_size": vocab_size,
            },
            "organs": organs,
            "tensor_count_observed": len(names),
            "tensor_count_truncated": len(weight_map) > len(names),
            "topology": {
                "layer_repetition": bool(layer_count and layer_count > 1),
                "sparse_moe": bool(experts and active_experts),
                "hybrid_recurrent_attention": any(item["id"] == "recurrent_or_deltanet" and item["present"] for item in organs) and any(item["id"] == "attention" and item["present"] for item in organs),
            },
            "hardware_gravity_plan": {
                "memory_tier": "hot-ssd-or-ram-for-active-working-set; cold-store-for-canonical-source",
                "placement_candidates": ["cpu", "gpu", "fpga", "remote"],
                "measurement_required": ["bytes_per_weight", "active_bytes_per_token", "bandwidth", "latency", "capability_loss"],
                "native_kernel_status": "not_claimed_by_metadata_recognizer",
            },
            "confidence": confidence,
            "unresolved": unresolved,
            "profile_declared_metadata": profile,
            "evidence": source_records,
            "qualification": {
                "status": "METADATA_ONLY",
                "weights_loaded": False,
                "native_execution_verified": False,
                "promotion_allowed": False,
            },
        }
        try:
            from .physical_graph import compile_physical_graph

            document["physical_graph"] = compile_physical_graph(document)
        except Exception as exc:  # noqa: BLE001 - recognizer remains useful if planner is absent
            document["physical_graph"] = {
                "schema": "hcli.physical_graph.v1",
                "qualification": "UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
            }
        return document


__all__ = ["ArchitectureRecognizer", "ArchitectureReport", "SCHEMA"]
