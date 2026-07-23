#!/usr/bin/env python3.12
"""Immutable GLM-5.2 source contract and exact header-derived ledgers.

This command downloads only the official non-weight control plane and reads
the safetensors headers with HTTP byte ranges.  It never downloads a tensor
payload.  Every index tensor must appear in exactly one official shard header,
every extent must match its dtype/shape, and every name must match an explicit
GLM-MoE-DSA organ rule.  Unknown layouts fail closed.

The resulting ledgers distinguish three truths that must never be conflated:

* logical weights (the BPW denominator),
* safetensors payload/container bytes, and
* currently allocated local bytes (metadata only at this phase).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import statistics
import struct
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from glm52_common import (  # noqa: E402
    Glm52Error,
    REPO_ROOT,
    allocated_bytes,
    atomic_json,
    atomic_text,
    canonical,
    git_blob_sha1,
    read_sealed_json,
    seal,
    sha256_file,
    utc_now,
)


REPO_ID = "zai-org/GLM-5.2"
REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
EXPECTED_ARCH = "GlmMoeDsaForCausalLM"
EXPECTED_MODEL_TYPE = "glm_moe_dsa"
EXPECTED_SHARDS = 282
EXPECTED_TENSORS = 59_585
RESOLVE_BASE = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}"
TREE_URL = f"https://huggingface.co/{REPO_ID}/tree/{REVISION}"
MANIFEST_API_URL = f"https://huggingface.co/api/models/{REPO_ID}/revision/{REVISION}?blobs=true"
MAX_HEADER_BYTES = 64 * 1024 * 1024
DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
}
WEIGHT_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


@dataclass(frozen=True)
class Classification:
    category: str
    section: str
    layer: int | None
    expert: int | None
    indexshare_group: int | None
    indexer_source_layer: int | None
    provisional_budget_class: str


@dataclass(frozen=True)
class HeaderTensor:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    logical_elements: int
    payload_bytes: int
    header_relative_start: int
    header_relative_end: int
    absolute_start: int
    absolute_end: int
    category: str
    section: str
    layer: int | None
    expert: int | None
    indexshare_group: int | None
    indexer_source_layer: int | None
    provisional_budget_class: str
    alias_of: str | None
    ownership: str
    terminal_coverage_state: str


@dataclass(frozen=True)
class ShardHeader:
    path: str
    file_bytes: int
    header_bytes: int
    data_start: int
    payload_bytes: int
    tensor_count: int
    xet_hash: str
    lfs_sha256: str
    tensors: tuple[HeaderTensor, ...]


def _effective_indexer_types(config: dict[str, Any]) -> list[str]:
    """Return main config modes plus the physically stored MTP full indexer.

    The official config deliberately has one entry per *main* hidden layer.  The
    extra checkpoint layer used for MTP is outside the Transformers main-model
    graph and stores a complete indexer of its own.  Keeping this derivation in
    one helper prevents us from pretending that the 79th mode came from config.
    """
    main_layers = int(config["num_hidden_layers"])
    main = list(config["indexer_types"])
    if len(main) != main_layers:
        raise Glm52Error(
            f"indexer_types must contain exactly {main_layers} main-layer entries"
        )
    if int(config.get("num_nextn_predict_layers", 1)) != 1:
        raise Glm52Error("this checkpoint contract requires exactly one MTP layer")
    return [*main, "full"]


def _previous_full(indexer_types: list[str], layer: int) -> int:
    for candidate in range(layer, -1, -1):
        if indexer_types[candidate] == "full":
            return candidate
    raise Glm52Error(f"shared indexer layer {layer} has no preceding full indexer")


def _group_id(indexer_types: list[str], layer: int) -> int:
    return _previous_full(indexer_types, layer)


def classify_tensor(name: str, config: dict[str, Any]) -> Classification:
    """Classify one real state-dict name; there is intentionally no catch-all."""
    if name == "model.embed_tokens.weight":
        return Classification("embeddings", "main_text", None, None, None, None,
                              "COMPRESSIBLE_CANDIDATE")
    if name == "lm_head.weight":
        return Classification("lm_head", "main_text", None, None, None, None,
                              "COMPRESSIBLE_CANDIDATE")
    if name == "model.norm.weight":
        return Classification("normalization", "main_text", None, None, None, None,
                              "CONTROL_SENSITIVE_CANDIDATE")

    match = LAYER_RE.match(name)
    if not match:
        raise Glm52Error(f"unrecognized GLM tensor name: {name!r}")
    layer = int(match.group(1))
    suffix = match.group(2)
    if not 0 <= layer <= int(config["num_hidden_layers"]):
        raise Glm52Error(f"layer outside main+MTP range: {name!r}")
    mtp_layer = int(config["num_hidden_layers"])
    section = "mtp" if layer == mtp_layer else "main_text"
    indexer_types = _effective_indexer_types(config)
    group = _group_id(indexer_types, layer)
    source_layer = group

    norm_suffixes = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
    }
    if suffix in norm_suffixes:
        return Classification("normalization", section, layer, None, group, source_layer,
                              "CONTROL_SENSITIVE_CANDIDATE")

    attention_suffixes = {
        "self_attn.q_a_proj.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight",
    }
    if suffix in attention_suffixes:
        return Classification("attention", section, layer, None, group, source_layer,
                              "COMPRESSIBLE_CANDIDATE")

    indexer_suffixes = {
        "self_attn.indexer.k_norm.bias",
        "self_attn.indexer.k_norm.weight",
        "self_attn.indexer.weights_proj.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.wq_b.weight",
    }
    if suffix in indexer_suffixes:
        if indexer_types[layer] != "full":
            raise Glm52Error(f"stored indexer tensor appears on shared layer: {name}")
        return Classification("indexer", section, layer, None, group, source_layer,
                              "CONTROL_SENSITIVE_CANDIDATE")

    if suffix in {
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
    }:
        if layer >= int(config["first_k_dense_replace"]):
            raise Glm52Error(f"dense MLP tensor appears on sparse/MTP layer: {name}")
        return Classification("dense_mlp", section, layer, None, group, source_layer,
                              "COMPRESSIBLE_CANDIDATE")

    expert = EXPERT_RE.match(suffix)
    if expert:
        expert_id = int(expert.group(1))
        if not 0 <= expert_id < int(config["n_routed_experts"]):
            raise Glm52Error(f"expert outside configured range: {name}")
        if layer < int(config["first_k_dense_replace"]):
            raise Glm52Error(f"routed expert appears on dense layer: {name}")
        return Classification("routed_expert", section, layer, expert_id, group, source_layer,
                              "COMPRESSIBLE_CANDIDATE")

    if suffix in {
        "mlp.shared_experts.down_proj.weight",
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight",
    }:
        if layer < int(config["first_k_dense_replace"]):
            raise Glm52Error(f"shared expert appears on dense layer: {name}")
        return Classification("shared_expert", section, layer, None, group, source_layer,
                              "COMPRESSIBLE_CANDIDATE")

    if suffix == "mlp.gate.weight":
        if layer < int(config["first_k_dense_replace"]):
            raise Glm52Error(f"router appears on dense layer: {name}")
        return Classification("router", section, layer, None, group, source_layer,
                              "CONTROL_SENSITIVE_CANDIDATE")
    if suffix == "mlp.gate.e_score_correction_bias":
        if layer < int(config["first_k_dense_replace"]):
            raise Glm52Error(f"router correction appears on dense layer: {name}")
        return Classification("router_control", section, layer, None, group, source_layer,
                              "CONTROL_SENSITIVE_CANDIDATE")

    if layer == mtp_layer and suffix == "eh_proj.weight":
        return Classification("mtp_projection", section, layer, None, group, source_layer,
                              "COMPRESSIBLE_CANDIDATE")
    if layer == mtp_layer and suffix in {"enorm.weight", "hnorm.weight"}:
        return Classification("mtp_normalization", section, layer, None, group, source_layer,
                              "CONTROL_SENSITIVE_CANDIDATE")
    if layer == mtp_layer and suffix == "shared_head.norm.weight":
        return Classification("mtp_head_norm", section, layer, None, group, source_layer,
                              "CONTROL_SENSITIVE_CANDIDATE")
    raise Glm52Error(f"unrecognized GLM layer tensor name: {name!r}")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "architectures": [EXPECTED_ARCH],
        "model_type": EXPECTED_MODEL_TYPE,
        "dtype": "bfloat16",
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "moe_router_dtype": "float32",
        "max_position_embeddings": 1_048_576,
        "num_nextn_predict_layers": 1,
        "index_share_for_mtp_iteration": True,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise Glm52Error(f"official config violates architecture contract: {mismatches}")
    indexer_types = config.get("indexer_types")
    if not isinstance(indexer_types, list) or len(indexer_types) != 78:
        raise Glm52Error("official config must declare 78 main-layer indexer types")
    if set(indexer_types) != {"full", "shared"}:
        raise Glm52Error(f"unknown IndexShare modes: {set(indexer_types)!r}")
    if indexer_types.count("full") != 21:
        raise Glm52Error("unexpected full/shared IndexShare pattern")
    return {"status": "PASS", "expected": expected, "mismatches": {}}


def locked_requirement_versions() -> dict[str, str]:
    path = REPO_ROOT / "tools/condense/requirements-glm52.txt"
    locked: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise Glm52Error(f"unlocked GLM runtime requirement: {line!r}")
        name, version = line.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if not version or normalized in locked:
            raise Glm52Error(f"duplicate/invalid GLM runtime requirement: {line!r}")
        locked[normalized] = version
    if not locked:
        raise Glm52Error("GLM runtime requirements lock is empty")
    return locked


def package_versions() -> dict[str, str]:
    locked = locked_requirement_versions()
    versions: dict[str, str] = {}
    for name in sorted(locked):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def ensure_control_plane() -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        ignore_patterns=["*.safetensors"],
        max_workers=8,
    )).resolve()
    if snapshot.name != REVISION:
        raise Glm52Error(f"snapshot path is not pinned to immutable revision: {snapshot}")
    required = {
        "LICENSE",
        "README.md",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = sorted(name for name in required if not (snapshot / name).is_file())
    if missing:
        raise Glm52Error(f"official control plane is incomplete: {missing}")
    return snapshot


def _manifest_info() -> tuple[Any, list[dict[str, Any]]]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise Glm52Error(f"Hub resolved {REVISION} to unexpected SHA {info.sha}")
    rows: list[dict[str, Any]] = []
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        rows.append({
            "path": sibling.rfilename,
            "logical_bytes": int(sibling.size or 0),
            "git_blob_id": getattr(sibling, "blob_id", None),
            "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
            "lfs_pointer_bytes": int(getattr(lfs, "pointer_size", 0) or 0) if lfs else None,
        })
    rows.sort(key=lambda row: row["path"])
    return info, rows


def _role(path: str) -> str:
    if WEIGHT_RE.match(path):
        return "WEIGHT_SHARD"
    if path == "model.safetensors.index.json":
        return "MODEL_INDEX"
    if path == "config.json":
        return "MODEL_CONFIG"
    if path == "tokenizer.json":
        return "TOKENIZER_MODEL"
    if path == "tokenizer_config.json":
        return "TOKENIZER_CONFIG"
    if path == "chat_template.jinja":
        return "CHAT_TEMPLATE"
    if path == "generation_config.json":
        return "GENERATION_CONFIG"
    if path == "LICENSE":
        return "LICENSE"
    if path == "README.md":
        return "MODEL_CARD"
    if path.startswith(".eval_results/") and path.endswith(".yaml"):
        return "OFFICIAL_EVAL_METADATA"
    if path == ".gitattributes":
        return "GIT_ATTRIBUTES"
    raise Glm52Error(f"unrecognized official repository file: {path}")


def validate_remote_manifest(rows: list[dict[str, Any]], index: dict[str, Any]) -> list[str]:
    if len(rows) != 295:
        raise Glm52Error(f"expected 295 official files, found {len(rows)}")
    weight_rows = [row for row in rows if _role(row["path"]) == "WEIGHT_SHARD"]
    if len(weight_rows) != EXPECTED_SHARDS:
        raise Glm52Error(f"expected {EXPECTED_SHARDS} shards, found {len(weight_rows)}")
    numbers = []
    for row in weight_rows:
        match = WEIGHT_RE.match(row["path"])
        assert match is not None
        number, total = map(int, match.groups())
        if total != EXPECTED_SHARDS:
            raise Glm52Error(f"shard total suffix mismatch: {row['path']}")
        numbers.append(number)
        if not row["lfs_sha256"]:
            raise Glm52Error(f"weight shard lacks official LFS SHA-256: {row['path']}")
    if sorted(numbers) != list(range(1, EXPECTED_SHARDS + 1)):
        raise Glm52Error("official weight shard numbering is incomplete")
    index_shards = sorted(set(index["weight_map"].values()))
    manifest_shards = sorted(row["path"] for row in weight_rows)
    if index_shards != manifest_shards:
        raise Glm52Error("manifest shard set and index shard set differ")
    unindexed_weight_files = sorted(set(manifest_shards) - set(index_shards))
    if unindexed_weight_files:
        raise Glm52Error(f"unindexed weight files: {unindexed_weight_files}")
    return manifest_shards


def _response_xet_hash(response: Any) -> str:
    candidates: list[str] = []
    for prior in response.history:
        if prior.headers.get("x-xet-hash"):
            candidates.append(prior.headers["x-xet-hash"].strip('"'))
    if response.headers.get("etag"):
        candidates.append(response.headers["etag"].strip('"'))
    candidates = [item for item in candidates if re.fullmatch(r"[0-9a-f]{64}", item)]
    if not candidates:
        raise Glm52Error("Xet-backed range response did not expose a 64-hex Xet identity")
    if len(set(candidates)) != 1:
        raise Glm52Error(f"conflicting Xet identities in redirect chain: {candidates}")
    return candidates[0]


def _range_get(client: Any, url: str, start: int, end: int, *, attempts: int = 5) -> Any:
    expected = end - start + 1
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            response = client.get(url, headers={"Range": f"bytes={start}-{end}"})
            if response.status_code != 206:
                raise Glm52Error(f"range request returned HTTP {response.status_code}: {url}")
            if len(response.content) != expected:
                raise Glm52Error(
                    f"range length mismatch for {url}: expected={expected} got={len(response.content)}"
                )
            return response
        except BaseException as exc:  # noqa: BLE001 - retried then surfaced
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.25 * (2 ** attempt))
    assert last_error is not None
    raise Glm52Error(f"range request failed after {attempts} attempts: {url}: {last_error}") from last_error


def _parse_content_total(response: Any) -> int:
    value = response.headers.get("content-range", "")
    match = re.fullmatch(r"bytes \d+-\d+/(\d+)", value)
    if not match:
        raise Glm52Error(f"invalid Content-Range: {value!r}")
    return int(match.group(1))


def fetch_shard_header(
    row: dict[str, Any],
    expected_names: set[str],
    config: dict[str, Any],
    *,
    timeout_seconds: float = 60.0,
) -> ShardHeader:
    import httpx

    path = row["path"]
    url = f"{RESOLVE_BASE}/{path}"
    with httpx.Client(follow_redirects=True, timeout=timeout_seconds) as client:
        prefix_response = _range_get(client, url, 0, 7)
        file_bytes = _parse_content_total(prefix_response)
        if file_bytes != int(row["logical_bytes"]):
            raise Glm52Error(
                f"remote size changed for {path}: manifest={row['logical_bytes']} range={file_bytes}"
            )
        header_bytes = struct.unpack("<Q", prefix_response.content)[0]
        if not 2 <= header_bytes <= MAX_HEADER_BYTES:
            raise Glm52Error(f"unsafe safetensors header length {header_bytes} for {path}")
        header_response = _range_get(client, url, 8, 7 + header_bytes)
        if _parse_content_total(header_response) != file_bytes:
            raise Glm52Error(f"remote size changed between header requests: {path}")
        xet_hash = _response_xet_hash(prefix_response)
        if _response_xet_hash(header_response) != xet_hash:
            raise Glm52Error(f"Xet identity changed between header requests: {path}")
    try:
        decoded = json.loads(header_response.content)
    except json.JSONDecodeError as exc:
        raise Glm52Error(f"invalid safetensors header JSON in {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise Glm52Error(f"safetensors header is not an object: {path}")
    metadata = decoded.pop("__metadata__", None)
    if metadata not in (None, {}):
        raise Glm52Error(f"unexpected safetensors metadata in {path}: {metadata!r}")
    actual_names = set(decoded)
    if actual_names != expected_names:
        raise Glm52Error(
            f"index/header tensor-set mismatch in {path}: "
            f"missing={sorted(expected_names - actual_names)[:5]} "
            f"extra={sorted(actual_names - expected_names)[:5]}"
        )
    data_start = 8 + header_bytes
    tensors: list[HeaderTensor] = []
    extents: list[tuple[int, int, str]] = []
    for name, entry in decoded.items():
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise Glm52Error(f"invalid safetensors entry schema: {path}:{name}")
        dtype = entry["dtype"]
        shape = entry["shape"]
        offsets = entry["data_offsets"]
        if dtype not in DTYPE_BYTES:
            raise Glm52Error(f"unsupported dtype {dtype!r}: {path}:{name}")
        if not isinstance(shape, list) or not shape or any(
            not isinstance(dim, int) or dim <= 0 for dim in shape
        ):
            raise Glm52Error(f"invalid positive tensor shape: {path}:{name}:{shape!r}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
            not isinstance(offset, int) for offset in offsets
        ):
            raise Glm52Error(f"invalid tensor offsets: {path}:{name}:{offsets!r}")
        start, end = offsets
        elements = math.prod(shape)
        payload_bytes = elements * DTYPE_BYTES[dtype]
        if start < 0 or end <= start or end - start != payload_bytes:
            raise Glm52Error(
                f"extent/shape mismatch: {path}:{name}: offsets={offsets} "
                f"shape={shape} dtype={dtype}"
            )
        if data_start + end > file_bytes:
            raise Glm52Error(f"tensor extent exceeds shard: {path}:{name}")
        classification = classify_tensor(name, config)
        tensors.append(HeaderTensor(
            name=name,
            shard=path,
            dtype=dtype,
            shape=tuple(shape),
            logical_elements=elements,
            payload_bytes=payload_bytes,
            header_relative_start=start,
            header_relative_end=end,
            absolute_start=data_start + start,
            absolute_end=data_start + end,
            category=classification.category,
            section=classification.section,
            layer=classification.layer,
            expert=classification.expert,
            indexshare_group=classification.indexshare_group,
            indexer_source_layer=classification.indexer_source_layer,
            provisional_budget_class=classification.provisional_budget_class,
            alias_of=None,
            ownership="UNIQUE_STORED_TENSOR",
            terminal_coverage_state="PENDING_CAMPAIGN_PACK_OR_JUSTIFICATION",
        ))
        extents.append((start, end, name))
    extents.sort()
    cursor = 0
    for start, end, name in extents:
        if start != cursor:
            relation = "overlap" if start < cursor else "gap"
            raise Glm52Error(
                f"safetensors payload {relation} before {path}:{name}: cursor={cursor} start={start}"
            )
        cursor = end
    if data_start + cursor != file_bytes:
        raise Glm52Error(
            f"shard payload does not close at EOF: {path}: "
            f"data_start={data_start} payload={cursor} file={file_bytes}"
        )
    return ShardHeader(
        path=path,
        file_bytes=file_bytes,
        header_bytes=header_bytes,
        data_start=data_start,
        payload_bytes=cursor,
        tensor_count=len(tensors),
        xet_hash=xet_hash,
        lfs_sha256=str(row["lfs_sha256"]),
        tensors=tuple(sorted(tensors, key=lambda tensor: tensor.name)),
    )


def fetch_all_headers(
    rows: list[dict[str, Any]],
    index: dict[str, Any],
    config: dict[str, Any],
    *,
    workers: int,
) -> list[ShardHeader]:
    weight_rows = {row["path"]: row for row in rows if WEIGHT_RE.match(row["path"])}
    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for name, shard in index["weight_map"].items():
        expected_by_shard[shard].add(name)
    if set(weight_rows) != set(expected_by_shard):
        raise Glm52Error("cannot fetch headers: manifest/index shard sets differ")

    def task(path: str) -> ShardHeader:
        return fetch_shard_header(weight_rows[path], expected_by_shard[path], config)

    results: list[ShardHeader] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(task, path): path for path in sorted(weight_rows)}
        try:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    results.sort(key=lambda item: item.path)
    all_tensors = [tensor for header in results for tensor in header.tensors]
    names = [tensor.name for tensor in all_tensors]
    if len(names) != EXPECTED_TENSORS or len(set(names)) != EXPECTED_TENSORS:
        raise Glm52Error(
            f"global header tensor count/uniqueness failed: rows={len(names)} unique={len(set(names))}"
        )
    if set(names) != set(index["weight_map"]):
        raise Glm52Error("global index/header tensor sets differ")
    for tensor in all_tensors:
        if index["weight_map"][tensor.name] != tensor.shard:
            raise Glm52Error(f"index/header shard assignment differs: {tensor.name}")
    payload = sum(header.payload_bytes for header in results)
    index_payload = int(index.get("metadata", {}).get("total_size", -1))
    if payload != index_payload:
        raise Glm52Error(f"index/header payload mismatch: index={index_payload} headers={payload}")
    return results


def _summarize_tensors(tensors: Iterable[HeaderTensor], key: str) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "logical_weights": 0, "source_payload_bytes": 0}
    )
    for tensor in tensors:
        label = str(getattr(tensor, key))
        row = grouped[label]
        row["tensor_count"] += 1
        row["logical_weights"] += tensor.logical_elements
        row["source_payload_bytes"] += tensor.payload_bytes
    return dict(sorted(grouped.items()))


def _subset(tensors: list[HeaderTensor], predicate: Any) -> dict[str, int]:
    selected = [tensor for tensor in tensors if predicate(tensor)]
    return {
        "tensor_count": len(selected),
        "logical_weights": sum(tensor.logical_elements for tensor in selected),
        "source_payload_bytes": sum(tensor.payload_bytes for tensor in selected),
    }


def build_logical_weight_ledger(headers: list[ShardHeader]) -> dict[str, Any]:
    tensors = [tensor for header in headers for tensor in header.tensors]
    all_logical = sum(tensor.logical_elements for tensor in tensors)
    all_payload = sum(tensor.payload_bytes for tensor in tensors)
    dtype_summary = _summarize_tensors(tensors, "dtype")
    if dtype_summary != {
        "BF16": {
            "tensor_count": 59_509,
            "logical_weights": 753_329_921_024,
            "source_payload_bytes": 1_506_659_842_048,
        },
        "F32": {
            "tensor_count": 76,
            "logical_weights": 19_456,
            "source_payload_bytes": 77_824,
        },
    }:
        raise Glm52Error(f"unexpected source dtype census: {dtype_summary}")
    if all_logical != 753_329_940_480 or all_payload != 1_506_659_919_872:
        raise Glm52Error(
            f"unexpected logical/source total: logical={all_logical} payload={all_payload}"
        )

    sections = _summarize_tensors(tensors, "section")
    categories = _summarize_tensors(tensors, "category")
    budget_classes = _summarize_tensors(tensors, "provisional_budget_class")
    major = {
        "all_declared_model_weights": _subset(tensors, lambda _: True),
        "main_text_model_logical_weights": _subset(tensors, lambda t: t.section == "main_text"),
        "mtp_logical_weights": _subset(tensors, lambda t: t.section == "mtp"),
        "dense_layer_weights": _subset(tensors, lambda t: t.layer in {0, 1, 2}),
        "sparse_text_layer_weights": _subset(
            tensors, lambda t: t.layer is not None and 3 <= t.layer <= 77
        ),
        "all_routed_expert_weights_including_mtp": _subset(
            tensors, lambda t: t.category == "routed_expert"
        ),
        "main_text_routed_expert_weights": _subset(
            tensors, lambda t: t.category == "routed_expert" and t.section == "main_text"
        ),
        "mtp_routed_expert_weights": _subset(
            tensors, lambda t: t.category == "routed_expert" and t.section == "mtp"
        ),
        "shared_expert_weights": _subset(tensors, lambda t: t.category == "shared_expert"),
        "router_weights": _subset(tensors, lambda t: t.category == "router"),
        "router_control_weights": _subset(tensors, lambda t: t.category == "router_control"),
        "attention_weights": _subset(tensors, lambda t: t.category == "attention"),
        "indexer_weights": _subset(tensors, lambda t: t.category == "indexer"),
        "normalization_weights": _subset(
            tensors,
            lambda t: t.category in {"normalization", "mtp_normalization", "mtp_head_norm"},
        ),
        "embedding_weights": _subset(tensors, lambda t: t.category == "embeddings"),
        "lm_head_weights": _subset(tensors, lambda t: t.category == "lm_head"),
        "provisional_compressible_weights": _subset(
            tensors, lambda t: t.provisional_budget_class == "COMPRESSIBLE_CANDIDATE"
        ),
        "provisional_control_sensitive_weights": _subset(
            tensors, lambda t: t.provisional_budget_class == "CONTROL_SENSITIVE_CANDIDATE"
        ),
    }
    if major["main_text_model_logical_weights"]["logical_weights"] != 743_377_019_904:
        raise Glm52Error("main-text logical denominator differs from verified header audit")
    if major["mtp_logical_weights"]["logical_weights"] != 9_952_920_576:
        raise Glm52Error("MTP logical denominator differs from verified header audit")
    if major["main_text_routed_expert_weights"]["logical_weights"] != 724_775_731_200:
        raise Glm52Error("main-text routed-expert denominator differs from verified header audit")
    if major["mtp_routed_expert_weights"]["logical_weights"] != 9_663_676_416:
        raise Glm52Error("MTP routed-expert denominator differs from verified header audit")

    rate_budgets = {
        "hard_1_bpw": {"numerator": 1, "denominator": 1},
        "planned_0_98_bpw": {"numerator": 49, "denominator": 50},
        "rate_0_75_bpw": {"numerator": 3, "denominator": 4},
        "rate_0_50_bpw": {"numerator": 1, "denominator": 2},
        "rate_0_33_represented_as_one_third_bpw": {"numerator": 1, "denominator": 3},
        "rate_0_25_bpw": {"numerator": 1, "denominator": 4},
    }
    for spec in rate_budgets.values():
        spec["maximum_complete_physical_bytes"] = (
            all_logical * int(spec["numerator"]) // (8 * int(spec["denominator"]))
        )
    return seal({
        "schema": "hawking.glm52.logical_weight_ledger.v1",
        "status": "PASS_HEADER_DERIVED",
        "repo": REPO_ID,
        "revision": REVISION,
        "authority": "Every tensor shape/dtype/extent was read from the immutable official shard header by HTTP Range; no headline parameter count was used as the denominator.",
        "logical_weight_denominator": all_logical,
        "tensor_count": len(tensors),
        "source_payload_bytes": all_payload,
        "source_dtype_summary": dtype_summary,
        "source_dtype_caveat": "The teacher source class is BF16, but 76 e_score_correction_bias tensors (19,456 elements) are physically F32 and are billed exactly.",
        "sections": sections,
        "primary_categories": categories,
        "provisional_budget_classes": budget_classes,
        "major_accounting_views": major,
        "rate_budgets": rate_budgets,
        "alias_policy": {
            "tie_word_embeddings": False,
            "stored_aliases": 0,
            "every_stored_tensor_counted_once": True,
        },
        "mtp_policy": {
            "physically_present": True,
            "included_in_complete_denominator": True,
            "optional_pack_allowed_only_with_billed_bytes": True,
            "official_transformers_runtime_omits_mtp": True,
        },
        "budget_class_warning": "Control-sensitive versus compressible is a provisional pilot allocation label, not permission to omit or store source-native bytes for free.",
    })


def build_source_format_ledger(headers: list[ShardHeader]) -> dict[str, Any]:
    tensors = [tensor for header in headers for tensor in header.tensors]
    container = sum(header.file_bytes for header in headers)
    payload = sum(header.payload_bytes for header in headers)
    framing = container - payload
    if framing != 7_467_536:
        raise Glm52Error(f"unexpected safetensors framing bytes: {framing}")
    return seal({
        "schema": "hawking.glm52.source_format_ledger.v1",
        "status": "PASS_HEADER_DERIVED_BODY_PENDING",
        "repo": REPO_ID,
        "revision": REVISION,
        "format": "safetensors",
        "weight_shards": len(headers),
        "tensor_count": len(tensors),
        "container_logical_bytes": container,
        "tensor_payload_bytes": payload,
        "safetensors_framing_bytes": framing,
        "header_json_bytes": sum(header.header_bytes for header in headers),
        "length_prefix_bytes": 8 * len(headers),
        "dtype_summary": _summarize_tensors(tensors, "dtype"),
        "format_invariants": {
            "all_shapes_positive": True,
            "all_extents_match_shape_times_dtype": True,
            "all_extents_contiguous_without_gap_or_overlap": True,
            "all_payloads_close_at_shard_eof": True,
            "index_header_tensor_sets_equal": True,
            "index_header_shard_assignments_equal": True,
            "remote_file_size_stable_across_header_requests": True,
            "xet_identity_stable_across_header_requests": True,
        },
        "verification_boundary": {
            "headers_fetched_and_verified": True,
            "tensor_payload_bodies_fetched": False,
            "tensor_payload_sha256_verified": False,
            "full_source_verification_must_transition_per_window": True,
        },
        "per_shard": [
            {
                "path": header.path,
                "file_bytes": header.file_bytes,
                "header_bytes": header.header_bytes,
                "data_start": header.data_start,
                "payload_bytes": header.payload_bytes,
                "tensor_count": header.tensor_count,
                "lfs_sha256": header.lfs_sha256,
                "xet_hash": header.xet_hash,
                "body_state": "NOT_FETCHED",
            }
            for header in headers
        ],
    })


def _local_file_state(snapshot: Path, relative: str) -> dict[str, Any]:
    path = snapshot / relative
    if not path.exists():
        return {
            "state": "NOT_PRESENT",
            "logical_bytes": 0,
            "allocated_bytes": 0,
            "sha256": None,
            "git_blob_id": None,
        }
    if not path.is_file():
        raise Glm52Error(f"official snapshot member is not a regular file: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(snapshot.parent.parent / "blobs"):
        raise Glm52Error(f"snapshot symlink does not resolve into its one-copy blob store: {path}")
    return {
        "state": "PRESENT_VERIFIED",
        "logical_bytes": path.stat().st_size,
        "allocated_bytes": allocated_bytes(resolved),
        "sha256": sha256_file(path),
        "git_blob_id": git_blob_sha1(path),
        "resolved_blob": str(resolved),
    }


def build_official_manifest(
    info: Any,
    rows: list[dict[str, Any]],
    headers: list[ShardHeader],
    snapshot: Path,
) -> dict[str, Any]:
    header_by_path = {header.path: header for header in headers}
    file_rows: list[dict[str, Any]] = []
    local_blob_inodes: set[tuple[int, int]] = set()
    for row in rows:
        path = row["path"]
        role = _role(path)
        header = header_by_path.get(path)
        if header:
            dependencies = {
                "tensor_count": header.tensor_count,
                "layers": sorted({
                    tensor.layer for tensor in header.tensors if tensor.layer is not None
                }),
                "categories": sorted({tensor.category for tensor in header.tensors}),
            }
            local = {
                "state": "HEADER_ONLY_REMOTE_RANGE_NO_LOCAL_WEIGHT_BODY",
                "logical_bytes": 0,
                "allocated_bytes": 0,
                "sha256": None,
                "git_blob_id": None,
            }
            verification = "HEADER_VERIFIED_BODY_NOT_FETCHED"
            xet_hash = header.xet_hash
        else:
            dependencies = {"tensor_count": 0, "layers": [], "categories": []}
            local = _local_file_state(snapshot, path)
            if local["state"] == "PRESENT_VERIFIED":
                resolved_stat = Path(local["resolved_blob"]).stat()
                local_blob_inodes.add((resolved_stat.st_dev, resolved_stat.st_ino))
                if local["logical_bytes"] != row["logical_bytes"]:
                    raise Glm52Error(f"local/manifest size mismatch: {path}")
                if row["lfs_sha256"] and local["sha256"] != row["lfs_sha256"]:
                    raise Glm52Error(f"local LFS SHA mismatch: {path}")
                if not row["lfs_sha256"] and local["git_blob_id"] != row["git_blob_id"]:
                    raise Glm52Error(f"local Git blob mismatch: {path}")
            verification = "FULL_FILE_VERIFIED"
            xet_hash = None
        file_rows.append({
            **row,
            "role": role,
            "is_weight": role == "WEIGHT_SHARD",
            "referenced_by_index": role == "WEIGHT_SHARD",
            "xet_hash": xet_hash,
            "dependencies": dependencies,
            "download_state": "NOT_FETCHED" if header else "PRESENT",
            "verification_state": verification,
            "eviction_state": "NOT_APPLICABLE_YET" if header else "PERMANENT_CONTROL_PLANE",
            "local": local,
        })

    weight_rows = [row for row in file_rows if row["is_weight"]]
    nonweight_rows = [row for row in file_rows if not row["is_weight"]]
    logical_total = sum(row["logical_bytes"] for row in file_rows)
    weight_total = sum(row["logical_bytes"] for row in weight_rows)
    nonweight_total = sum(row["logical_bytes"] for row in nonweight_rows)
    current_allocated = sum(
        row["local"]["allocated_bytes"]
        for row in nonweight_rows
        if row["local"]["state"] == "PRESENT_VERIFIED"
    )
    return seal({
        "schema": "hawking.glm52.official_manifest.v1",
        "status": "PASS_CONTROL_PLANE_AND_HEADERS_BODY_PENDING",
        "repo": REPO_ID,
        "revision": REVISION,
        "immutable_tree_url": TREE_URL,
        "manifest_api_url": MANIFEST_API_URL,
        "resolved_at": utc_now(),
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "license": "MIT",
        "file_count": len(file_rows),
        "weight_shards": len(weight_rows),
        "source_logical_bytes": logical_total,
        "weight_container_logical_bytes": weight_total,
        "nonweight_logical_bytes": nonweight_total,
        "source_allocated_bytes_remote": "UNAVAILABLE_UNTIL_MATERIALIZED",
        "current_local_control_plane_allocated_bytes_unique_blob_sum": current_allocated,
        "current_local_control_plane_unique_blob_inodes": len(local_blob_inodes),
        "largest_weight_shard": max(
            ({"path": row["path"], "bytes": row["logical_bytes"]} for row in weight_rows),
            key=lambda row: row["bytes"],
        ),
        "smallest_weight_shard": min(
            ({"path": row["path"], "bytes": row["logical_bytes"]} for row in weight_rows),
            key=lambda row: row["bytes"],
        ),
        "median_weight_shard_bytes": int(statistics.median(row["logical_bytes"] for row in weight_rows)),
        "mean_weight_shard_bytes": statistics.fmean(row["logical_bytes"] for row in weight_rows),
        "one_copy": {
            "authoritative_location": str(snapshot.parent.parent),
            "snapshot_view": str(snapshot),
            "local_dir_copy": None,
            "weight_body_copies": 0,
            "control_plane_content_addressed_copy": 1,
        },
        "files": file_rows,
    })


def build_architecture_contract(config: dict[str, Any], headers: list[ShardHeader]) -> dict[str, Any]:
    tensors = [tensor for header in headers for tensor in header.tensors]
    indexer_types = _effective_indexer_types(config)
    full_main = [index for index, kind in enumerate(indexer_types[:78]) if kind == "full"]
    shared_main = [index for index, kind in enumerate(indexer_types[:78]) if kind == "shared"]
    stored_indexer_layers = sorted({
        tensor.layer for tensor in tensors if tensor.category == "indexer"
    })
    if stored_indexer_layers != [*full_main, 78]:
        raise Glm52Error("stored indexer layers do not match config full-indexer pattern")
    return seal({
        "schema": "hawking.glm52.architecture_contract.v1",
        "status": "PASS_CONFIG_INDEX_AND_HEADERS",
        "repo": REPO_ID,
        "revision": REVISION,
        "architecture": EXPECTED_ARCH,
        "model_type": EXPECTED_MODEL_TYPE,
        "source_class": ["OFFICIAL_BF16_TEACHER", "VULTURE_XET_STREAMING", "TEXT_GENERATION"],
        "config_validation": validate_config(config),
        "geometry": {
            "vocab_size": config["vocab_size"],
            "hidden_size": config["hidden_size"],
            "main_hidden_layers": config["num_hidden_layers"],
            "physically_stored_layers_including_mtp": config["num_hidden_layers"] + 1,
            "dense_layers": config["first_k_dense_replace"],
            "sparse_main_layers": config["num_hidden_layers"] - config["first_k_dense_replace"],
            "dense_intermediate_size": config["intermediate_size"],
            "moe_intermediate_size": config["moe_intermediate_size"],
            "routed_experts": config["n_routed_experts"],
            "shared_experts": config["n_shared_experts"],
            "active_routed_experts_per_token": config["num_experts_per_tok"],
            "attention_heads": config["num_attention_heads"],
            "kv_heads": config["num_key_value_heads"],
            "q_lora_rank": config["q_lora_rank"],
            "kv_lora_rank": config["kv_lora_rank"],
            "qk_head_dim": config["qk_head_dim"],
            "qk_nope_head_dim": config["qk_nope_head_dim"],
            "qk_rope_head_dim": config["qk_rope_head_dim"],
            "v_head_dim": config["v_head_dim"],
            "context_tokens": config["max_position_embeddings"],
            "mtp_layers": config["num_nextn_predict_layers"],
        },
        "routing": {
            "stored_router_matrix_dtype": "BF16",
            "stored_correction_bias_dtype": "F32",
            "router_compute_dtype": config["moe_router_dtype"],
            "scoring": config["scoring_func"],
            "normalize_topk": config["norm_topk_prob"],
            "scaling_factor": config["routed_scaling_factor"],
            "topk_method": config["topk_method"],
        },
        "dsa_indexshare": {
            "index_topk": config["index_topk"],
            "index_heads": config["index_n_heads"],
            "index_head_dim": config["index_head_dim"],
            "index_topk_frequency": config["index_topk_freq"],
            "index_skip_topk_offset": config["index_skip_topk_offset"],
            "main_full_indexer_layers": full_main,
            "main_shared_indexer_layers": shared_main,
            "mtp_checkpoint_indexer_type": indexer_types[78],
            "mtp_iteration_sharing_enabled_by_config": config["index_share_for_mtp_iteration"],
            "mtp_indexer_type_provenance": "derived from physically stored layer-78 indexer tensors, not a 79th config list entry",
            "stored_indexer_layers": stored_indexer_layers,
            "main_rope_interleaved": config["rope_interleave"],
            "indexer_rope_interleaved": config["indexer_rope_interleave"],
        },
        "weights": {
            "tensor_count": len(tensors),
            "logical_elements": sum(tensor.logical_elements for tensor in tensors),
            "dtype_summary": _summarize_tensors(tensors, "dtype"),
            "embeddings_tied_to_lm_head": config["tie_word_embeddings"],
        },
        "runtime_contract": {
            "config_saved_by_transformers": config.get("transformers_version"),
            "official_model_card_transformers_literal": "v0.5.12+",
            "model_card_version_inference": "LIKELY_TYPO_FOR_5.12+",
            "custom_repository_code_required": False,
            "official_transformers_main_model_supported": True,
            "official_transformers_mtp_supported": False,
            "full_source_accounting_requires_separate_mtp_pack": True,
        },
        "unknown_tensor_names": [],
    })


def build_dependency_graph(
    headers: list[ShardHeader],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    tensors = [tensor for header in headers for tensor in header.tensors]
    shard_sizes = {header.path: header.file_bytes for header in headers}
    organs: dict[str, dict[str, Any]] = {}

    def add_organ(organ_id: str, selected: list[HeaderTensor], **extra: Any) -> None:
        if not selected:
            raise Glm52Error(f"dependency organ has no tensors: {organ_id}")
        organs[organ_id] = {
            "organ_id": organ_id,
            "tensor_count": len(selected),
            "logical_weights": sum(tensor.logical_elements for tensor in selected),
            "source_payload_bytes": sum(tensor.payload_bytes for tensor in selected),
            "source_shards": sorted({tensor.shard for tensor in selected}),
            "tensor_names": sorted(tensor.name for tensor in selected),
            **extra,
        }

    add_organ(
        "global_input",
        [tensor for tensor in tensors if tensor.category == "embeddings"],
        execution_order=0,
        section="main_text",
    )
    indexer_types = _effective_indexer_types(config)
    for layer in range(78):
        selected = [tensor for tensor in tensors if tensor.layer == layer]
        source = _previous_full(indexer_types, layer)
        add_organ(
            f"text_layer_{layer:02d}",
            selected,
            execution_order=layer + 1,
            section="main_text",
            layer=layer,
            mlp_type="dense" if layer < 3 else "sparse",
            indexer_type=indexer_types[layer],
            indexshare_source_layer=source,
            runtime_state_dependencies=([] if source == layer else [f"index_selection_from_layer_{source:02d}"]),
        )
    add_organ(
        "global_output",
        [tensor for tensor in tensors if tensor.category in {"normalization", "lm_head"} and tensor.layer is None],
        execution_order=79,
        section="main_text",
    )
    add_organ(
        "mtp_layer_78",
        [tensor for tensor in tensors if tensor.section == "mtp"],
        execution_order=80,
        section="mtp",
        layer=78,
        indexer_type=indexer_types[78],
        indexshare_source_layer=_previous_full(indexer_types, 78),
        runtime_state_dependencies=["final_main_hidden_state", "shared_lm_head"],
    )

    assigned: list[str] = []
    for organ in organs.values():
        assigned.extend(organ["tensor_names"])
    if len(assigned) != len(tensors) or set(assigned) != {tensor.name for tensor in tensors}:
        counts = Counter(assigned)
        duplicate = sorted(name for name, count in counts.items() if count != 1)
        missing = sorted({tensor.name for tensor in tensors} - set(assigned))
        raise Glm52Error(
            f"dependency organs do not partition tensors: duplicates={duplicate[:5]} missing={missing[:5]}"
        )

    shard_records = []
    for header in headers:
        layer_set = sorted({tensor.layer for tensor in header.tensors if tensor.layer is not None})
        organ_ids = sorted(
            organ_id for organ_id, organ in organs.items()
            if header.path in organ["source_shards"]
        )
        shard_records.append({
            "path": header.path,
            "logical_bytes": shard_sizes[header.path],
            "xet_hash": header.xet_hash,
            "lfs_sha256": header.lfs_sha256,
            "tensor_count": header.tensor_count,
            "layers": layer_set,
            "organ_ids": organ_ids,
            "tensor_names": [tensor.name for tensor in header.tensors],
        })
    tensor_records = [asdict(tensor) for tensor in sorted(tensors, key=lambda tensor: tensor.name)]
    graph = seal({
        "schema": "hawking.glm52.shard_dependency_graph.v1",
        "status": "PASS_HEADER_DERIVED",
        "repo": REPO_ID,
        "revision": REVISION,
        "tensor_count": len(tensors),
        "shard_count": len(headers),
        "organ_count": len(organs),
        "coverage_invariant": {
            "every_index_tensor_in_exactly_one_header": True,
            "every_tensor_assigned_to_exactly_one_organ": True,
            "every_official_weight_shard_referenced": True,
            "terminal_states_allowed": [
                "PACKED_IN_CORE_ARTIFACT",
                "PACKED_IN_OPTIONAL_MTP_PACK",
                "PROTECTED_SOURCE_NATIVE_WITH_BILLED_BYTES",
                "INTENTIONALLY_OMITTED_WITH_CAPABILITY_JUSTIFICATION",
            ],
            "current_state": "PENDING_CAMPAIGN_PACK_OR_JUSTIFICATION",
        },
        "organs": [organs[key] for key in sorted(organs, key=lambda key: organs[key]["execution_order"])],
        "shards": shard_records,
        "tensors": tensor_records,
    })
    return graph, {key: set(value["source_shards"]) for key, value in organs.items()}


def _p99(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(0.99 * len(ordered)) - 1]


def build_streaming_schedule(
    headers: list[ShardHeader],
    organ_shards: dict[str, set[str]],
    logical_weights: int,
) -> tuple[dict[str, Any], str]:
    shard_bytes = {header.path: header.file_bytes for header in headers}
    free_bytes = shutil.disk_usage(REPO_ROOT).free
    largest_two = sum(sorted(shard_bytes.values(), reverse=True)[:2])
    hard_floor = 5 * 1024 ** 3
    operational_reserve = max(hard_floor, largest_two)
    artifact_growth = logical_weights * (98 + 75 + 50) // (100 * 8)
    evidence_growth = 20 * 1024 ** 3
    active_scratch = 0
    usable_raw = max(0, free_bytes - operational_reserve - artifact_growth - evidence_growth - active_scratch)
    safety_fraction = 0.70
    p99 = _p99(list(shard_bytes.values()))
    # Two complete raw windows may coexist (active + prefetch).  N-1 is an
    # eviction state, not a third retained body.
    disk_limited = math.floor((safety_fraction * usable_raw / 2) / p99)
    max_single_organ = max(len(value) for value in organ_shards.values())
    if disk_limited < max_single_organ:
        raise Glm52Error(
            f"disk cannot admit one dependency-complete organ: limit={disk_limited} need={max_single_organ}"
        )
    preliminary_target = min(48, disk_limited)

    ordered_organs = [
        "global_input",
        *[f"text_layer_{layer:02d}" for layer in range(78)],
        "global_output",
        "mtp_layer_78",
    ]
    windows: list[dict[str, Any]] = []
    fetched: set[str] = set()
    carried: set[str] = set()
    cursor = 0
    while cursor < len(ordered_organs):
        window_index = len(windows)
        organ_ids: list[str] = []
        required = set(carried)
        next_cursor = cursor
        while next_cursor < len(ordered_organs):
            organ_id = ordered_organs[next_cursor]
            candidate = required | organ_shards[organ_id]
            if organ_ids and len(candidate) > preliminary_target:
                break
            if not organ_ids and len(candidate) > preliminary_target:
                raise Glm52Error(
                    f"window target cannot hold carried state plus organ {organ_id}: "
                    f"need={len(candidate)} target={preliminary_target}"
                )
            organ_ids.append(organ_id)
            required = candidate
            next_cursor += 1
        new = required - fetched
        if (required - carried) != new:
            refetch = (required - carried) - new
            raise Glm52Error(f"preliminary schedule would refetch evicted shards: {sorted(refetch)}")
        fetched |= new
        future = set().union(*(
            organ_shards[organ_id]
            for organ_id in ordered_organs[next_cursor:]
        )) if next_cursor < len(ordered_organs) else set()
        carry_out = required & future
        evict_after = required - carry_out
        windows.append({
            "window_id": f"W{window_index:03d}",
            "organ_ids": organ_ids,
            "source_shards": sorted(required),
            "source_shard_count": len(required),
            "resident_logical_bytes": sum(shard_bytes[path] for path in required),
            "carry_in_shards": sorted(carried),
            "new_fetch_shards": sorted(new),
            "new_fetch_logical_bytes": sum(shard_bytes[path] for path in new),
            "carry_out_shards": sorted(carry_out),
            "evict_after_seal_shards": sorted(evict_after),
            "refetch_shards": [],
        })
        carried = carry_out
        cursor = next_cursor
    if fetched != set(shard_bytes):
        raise Glm52Error(f"schedule does not fetch every shard: missing={sorted(set(shard_bytes)-fetched)}")
    if carried:
        raise Glm52Error(f"final schedule retains source shards: {sorted(carried)}")
    new_counts = Counter(path for window in windows for path in window["new_fetch_shards"])
    if set(new_counts.values()) != {1} or set(new_counts) != set(shard_bytes):
        raise Glm52Error("schedule violates one-fetch preliminary invariant")

    schedule = seal({
        "schema": "hawking.glm52.streaming_schedule.v1",
        "status": "PRELIMINARY_DEPENDENCY_COMPLETE_PENDING_XET_AUTOTUNE",
        "repo": REPO_ID,
        "revision": REVISION,
        "planner_inputs": {
            "free_disk_bytes": free_bytes,
            "hard_floor_bytes": hard_floor,
            "largest_two_source_shards_bytes": largest_two,
            "operational_reserve_bytes": operational_reserve,
            "projected_three_complete_artifacts_bytes_0_98_plus_0_75_plus_0_50": artifact_growth,
            "projected_evidence_bytes": evidence_growth,
            "active_scratch_bytes": active_scratch,
            "usable_raw_window_bytes": usable_raw,
            "safety_fraction": safety_fraction,
            "p99_allocated_proxy_uses_remote_logical_shard_bytes": p99,
            "two_simultaneous_complete_raw_windows": True,
            "disk_limited_shards_per_window": disk_limited,
            "preliminary_target_shards_per_window": preliminary_target,
        },
        "pipeline": {
            "n_minus_1": "verification/sealing/eviction; only carry-out bodies remain",
            "n": "active BF16 teacher/fit/pack/forward",
            "n_plus_1": "prefetch/reconstruction after measured admission",
            "fourth_window": "DISALLOWED_PENDING_MEASUREMENT",
        },
        "window_count": len(windows),
        "maximum_resident_shards_in_one_window": max(window["source_shard_count"] for window in windows),
        "maximum_simultaneous_shards_active_plus_prefetch_upper_bound": 2 * max(
            window["source_shard_count"] for window in windows
        ),
        "source_shards_scheduled": len(fetched),
        "planned_refetches": 0,
        "windows": windows,
        "freeze_boundary": "Window size/concurrency may change only after GLM52_XET_AUTOTUNE; tensor dependencies and one-fetch accounting remain immutable.",
    })
    lines = [
        "# GLM-5.2 preliminary streaming schedule",
        "",
        f"Status: **{schedule['status']}**",
        "",
        f"- Immutable source: `{REPO_ID}@{REVISION}`",
        f"- Windows: `{len(windows)}`",
        f"- Preliminary shard target: `{preliminary_target}`",
        f"- Maximum resident shards in one dependency window: `{schedule['maximum_resident_shards_in_one_window']}`",
        f"- Every official shard scheduled exactly once: `{len(fetched)}/{len(shard_bytes)}`",
        f"- Planned refetches: `0`",
        "",
        "This is a dependency-correct disk admission plan, not an Xet throughput result. The Xet autotuner may resize windows after measuring APFS allocation, reconstruction scratch, swap, thermals, and heavy-lane regression.",
        "",
        "| Window | Organs | Resident shards | New shards | Carry out | New bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for window in windows:
        lines.append(
            f"| `{window['window_id']}` | {len(window['organ_ids'])} | "
            f"{window['source_shard_count']} | {len(window['new_fetch_shards'])} | "
            f"{len(window['carry_out_shards'])} | {window['new_fetch_logical_bytes']} |"
        )
    lines.extend(["", f"Schedule seal: `{schedule['seal_sha256']}`.", ""])
    return schedule, "\n".join(lines)


def build_source_admission(
    *,
    manifest: dict[str, Any],
    architecture: dict[str, Any],
    logical: dict[str, Any],
    source_format: dict[str, Any],
    graph: dict[str, Any],
    schedule: dict[str, Any],
    snapshot: Path,
    main_revision: str,
) -> dict[str, Any]:
    versions = package_versions()
    current_verified = locked_requirement_versions()
    lock_gate = versions == current_verified
    xet_stack = {
        "huggingface-hub": "1.24.0",
        "hf-xet": "1.5.2",
        "transformers": "5.14.1",
        "safetensors": "0.8.0",
        "tokenizers": "0.22.2",
    }
    current_gate = all(versions.get(name) == version for name, version in xet_stack.items())
    environment_root = Path(sys.prefix).resolve()
    environment_config = environment_root / "pyvenv.cfg"
    environment_text = (
        environment_config.read_text(encoding="utf-8")
        if environment_config.is_file() else ""
    )
    isolated_gate = bool(
        sys.prefix != sys.base_prefix
        and re.search(
            r"(?im)^include-system-site-packages\s*=\s*false\s*$",
            environment_text,
        )
    )
    disk = shutil.disk_usage(REPO_ROOT)
    return seal({
        "schema": "hawking.glm52.source_admission.v1",
        "status": "ADMITTED_CONTROL_PLANE_HEADERS_AND_PLAN_BODY_PENDING",
        "admitted_at": utc_now(),
        "repo": REPO_ID,
        "main_resolved_live": main_revision,
        "revision": REVISION,
        "main_matches_pinned_revision_at_admission": main_revision == REVISION,
        "experiment_identity_uses_main": False,
        "license": {
            "spdx": "MIT",
            "path": str(snapshot / "LICENSE"),
            "sha256": sha256_file(snapshot / "LICENSE"),
        },
        "source": {
            "files": manifest["file_count"],
            "logical_bytes": manifest["source_logical_bytes"],
            "weight_shards": manifest["weight_shards"],
            "weight_container_bytes": manifest["weight_container_logical_bytes"],
            "tensor_payload_bytes": source_format["tensor_payload_bytes"],
            "logical_weight_denominator": logical["logical_weight_denominator"],
            "largest_shard": manifest["largest_weight_shard"],
            "source_allocated_bytes": "UNAVAILABLE_UNTIL_MATERIALIZED",
            "source_mode": "VULTURE_XET_STREAMING",
            "complete_source_resident": False,
        },
        "architecture": {
            "architecture": architecture["architecture"],
            "model_type": architecture["model_type"],
            "main_layers": architecture["geometry"]["main_hidden_layers"],
            "mtp_layers": architecture["geometry"]["mtp_layers"],
            "custom_code_required": architecture["runtime_contract"]["custom_repository_code_required"],
            "official_transformers_mtp_supported": architecture["runtime_contract"]["official_transformers_mtp_supported"],
        },
        "local_runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "isolated_environment_required": True,
            "isolated_environment_gate": "PASS" if isolated_gate else "FAIL",
            "requirements_lock": {
                "path": "tools/condense/requirements-glm52.txt",
                "sha256": sha256_file(REPO_ROOT / "tools/condense/requirements-glm52.txt"),
            },
            "packages": versions,
            "current_versions_verified_live_on_2026_07_21": current_verified,
            "complete_requirements_lock_gate": "PASS" if lock_gate else "FAIL",
            "current_hf_xet_stack_gate": "PASS" if current_gate else "FAIL_UPGRADE_REQUIRED",
            "tokenizers_compatibility_note": (
                "Transformers 5.14.1 requires tokenizers>=0.22.0,<=0.23.0; "
                "0.23.0 has no stable PyPI release, so 0.22.2 is the newest "
                "stable compatible release verified for this campaign."
            ),
            "official_checkpoint_saved_by_transformers": "5.12.0",
        },
        "toolchain_binding": {
            "generator": {
                "path": "tools/condense/glm52_contract.py",
                "sha256": sha256_file(Path(__file__)),
            },
            "shared_common": {
                "path": "tools/condense/glm52_common.py",
                "sha256": sha256_file(REPO_ROOT / "tools/condense/glm52_common.py"),
            },
            "requirements_lock": {
                "path": "tools/condense/requirements-glm52.txt",
                "sha256": sha256_file(
                    REPO_ROOT / "tools/condense/requirements-glm52.txt"
                ),
            },
        },
        "xet": {
            "official_xet_objects": 283,
            "weight_xet_objects": 282,
            "tokenizer_xet_objects": 1,
            "hf_transfer_allowed": False,
            "header_range_bytes_read": source_format["safetensors_framing_bytes"],
            "body_bytes_read": 0,
            "autotune_status": "PENDING",
            "fixed_download_concurrency_variable_for_current_stack": "HF_XET_FIXED_DOWNLOAD_CONCURRENCY",
            "deprecated_or_absent_range_get_variable_must_not_be_assumed": "HF_XET_NUM_CONCURRENT_RANGE_GETS",
        },
        "storage": {
            "filesystem_total_bytes": disk.total,
            "filesystem_used_bytes": disk.used,
            "filesystem_free_bytes": disk.free,
            "one_copy_status": manifest["one_copy"],
            "schedule_operational_reserve_bytes": schedule["planner_inputs"]["operational_reserve_bytes"],
            "bf16_body_admitted_now": False,
        },
        "evidence": {
            "official_manifest_seal_sha256": manifest["seal_sha256"],
            "architecture_contract_seal_sha256": architecture["seal_sha256"],
            "logical_weight_ledger_seal_sha256": logical["seal_sha256"],
            "source_format_ledger_seal_sha256": source_format["seal_sha256"],
            "dependency_graph_seal_sha256": graph["seal_sha256"],
            "streaming_schedule_seal_sha256": schedule["seal_sha256"],
        },
        "admission_gates": {
            "immutable_revision": True,
            "license_verified": True,
            "official_file_manifest_complete": True,
            "all_index_tensors_mapped_to_verified_headers": True,
            "exact_logical_weight_denominator": True,
            "unknown_layouts": 0,
            "dependency_schedule_complete": True,
            "current_xet_stack": current_gate,
            "body_stream": False,
        },
        "body_stream_blockers": [
            "Xet throughput/autotune has not run",
            "GLM synthetic twin and official bounded reference parity are not green",
            "physical compact writer/reader/direct execution is not green",
            "crash-safe fetch/verify/capture/pack/seal/evict integration is not green",
        ],
    })


def _load_control(snapshot: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(index, dict):
        raise Glm52Error("official config/index roots must be JSON objects")
    if not isinstance(index.get("weight_map"), dict):
        raise Glm52Error("official index lacks weight_map")
    if len(index["weight_map"]) != EXPECTED_TENSORS:
        raise Glm52Error(
            f"official index tensor count changed: {len(index['weight_map'])} != {EXPECTED_TENSORS}"
        )
    validate_config(config)
    return config, index


def run_admission(*, workers: int) -> dict[str, Any]:
    if not 1 <= workers <= 64:
        raise Glm52Error("header worker count must be between 1 and 64")
    snapshot = ensure_control_plane()
    config, index = _load_control(snapshot)
    info, rows = _manifest_info()
    validate_remote_manifest(rows, index)
    headers = fetch_all_headers(rows, index, config, workers=workers)
    logical = build_logical_weight_ledger(headers)
    source_format = build_source_format_ledger(headers)
    manifest = build_official_manifest(info, rows, headers, snapshot)
    architecture = build_architecture_contract(config, headers)
    graph, organ_shards = build_dependency_graph(headers, config)
    schedule, schedule_md = build_streaming_schedule(
        headers,
        organ_shards,
        logical["logical_weight_denominator"],
    )
    from huggingface_hub import HfApi

    main_revision = HfApi().model_info(REPO_ID).sha
    admission = build_source_admission(
        manifest=manifest,
        architecture=architecture,
        logical=logical,
        source_format=source_format,
        graph=graph,
        schedule=schedule,
        snapshot=snapshot,
        main_revision=main_revision,
    )
    # All construction and cross-checks complete before the first output write.
    atomic_json(REPO_ROOT / "GLM52_OFFICIAL_MANIFEST.json", manifest)
    atomic_json(REPO_ROOT / "GLM52_ARCHITECTURE_CONTRACT.json", architecture)
    atomic_json(REPO_ROOT / "GLM52_LOGICAL_WEIGHT_LEDGER.json", logical)
    atomic_json(REPO_ROOT / "GLM52_SOURCE_FORMAT_LEDGER.json", source_format)
    atomic_json(REPO_ROOT / "GLM52_SHARD_DEPENDENCY_GRAPH.json", graph)
    atomic_json(REPO_ROOT / "GLM52_STREAMING_SCHEDULE.json", schedule)
    atomic_text(REPO_ROOT / "GLM52_STREAMING_SCHEDULE.md", schedule_md)
    atomic_json(REPO_ROOT / "GLM52_SOURCE_ADMISSION.json", admission)
    return {
        "status": admission["status"],
        "revision": REVISION,
        "files": manifest["file_count"],
        "shards": manifest["weight_shards"],
        "tensors": logical["tensor_count"],
        "logical_weights": logical["logical_weight_denominator"],
        "source_logical_bytes": manifest["source_logical_bytes"],
        "windows": schedule["window_count"],
        "admission_seal_sha256": admission["seal_sha256"],
    }


def refresh_source_admission_offline() -> dict[str, Any]:
    """Refresh only mutable local-runtime admission facts without network/body I/O.

    The immutable header-derived ledgers remain byte-for-byte inputs.  This command is
    used after an intentional lock/tooling update so the admission cannot silently pin
    stale runtime provenance.  It never calls ``snapshot_download`` or ``HfApi``.
    """
    artifacts = {
        "manifest": read_sealed_json(REPO_ROOT / "GLM52_OFFICIAL_MANIFEST.json"),
        "architecture": read_sealed_json(REPO_ROOT / "GLM52_ARCHITECTURE_CONTRACT.json"),
        "logical": read_sealed_json(REPO_ROOT / "GLM52_LOGICAL_WEIGHT_LEDGER.json"),
        "source_format": read_sealed_json(REPO_ROOT / "GLM52_SOURCE_FORMAT_LEDGER.json"),
        "graph": read_sealed_json(REPO_ROOT / "GLM52_SHARD_DEPENDENCY_GRAPH.json"),
        "schedule": read_sealed_json(
            REPO_ROOT / "GLM52_STREAMING_SCHEDULE_PRE_AUTOTUNE.json"
        ),
    }
    existing = read_sealed_json(REPO_ROOT / "GLM52_SOURCE_ADMISSION.json")
    expected = {
        "official_manifest_seal_sha256": artifacts["manifest"]["seal_sha256"],
        "architecture_contract_seal_sha256": artifacts["architecture"]["seal_sha256"],
        "logical_weight_ledger_seal_sha256": artifacts["logical"]["seal_sha256"],
        "source_format_ledger_seal_sha256": artifacts["source_format"]["seal_sha256"],
        "dependency_graph_seal_sha256": artifacts["graph"]["seal_sha256"],
        "streaming_schedule_seal_sha256": artifacts["schedule"]["seal_sha256"],
    }
    if existing.get("evidence") != expected:
        raise Glm52Error(
            "offline admission refresh refuses drifted immutable header-ledger inputs"
        )
    license_path = Path(str(existing.get("license", {}).get("path", "")))
    snapshot = license_path.parent.resolve()
    if snapshot.name != REVISION or not license_path.is_file():
        raise Glm52Error("offline admission refresh lacks the pinned local control snapshot")
    if existing.get("main_resolved_live") != REVISION \
            or existing.get("revision") != REVISION:
        raise Glm52Error("offline admission refresh lacks the prior immutable-main binding")
    admission = build_source_admission(
        manifest=artifacts["manifest"],
        architecture=artifacts["architecture"],
        logical=artifacts["logical"],
        source_format=artifacts["source_format"],
        graph=artifacts["graph"],
        schedule=artifacts["schedule"],
        snapshot=snapshot,
        main_revision=REVISION,
    )
    atomic_json(REPO_ROOT / "GLM52_SOURCE_ADMISSION.json", admission)
    return {
        "status": admission["status"],
        "network_access": False,
        "model_body_bytes_read": 0,
        "immutable_input_seals_unchanged": True,
        "requirements_lock_sha256": admission["local_runtime"][
            "requirements_lock"
        ]["sha256"],
        "admission_seal_sha256": admission["seal_sha256"],
    }


def selfcheck() -> dict[str, Any]:
    config = {
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "num_nextn_predict_layers": 1,
        "indexer_types": [
            "full" if layer in {0, 1, 2, *range(6, 78, 4)} else "shared"
            for layer in range(78)
        ],
    }
    rows = {
        "model.embed_tokens.weight": "embeddings",
        "model.layers.0.mlp.gate_proj.weight": "dense_mlp",
        "model.layers.3.mlp.experts.255.down_proj.weight": "routed_expert",
        "model.layers.3.mlp.shared_experts.up_proj.weight": "shared_expert",
        "model.layers.6.self_attn.indexer.wk.weight": "indexer",
        "model.layers.78.eh_proj.weight": "mtp_projection",
        "model.layers.78.mlp.gate.e_score_correction_bias": "router_control",
    }
    classified = {name: classify_tensor(name, config).category for name in rows}
    if classified != rows:
        raise AssertionError(f"classification selfcheck failed: {classified}")
    rejected = []
    for bad in (
        "model.layers.4.mlp.up_proj.weight",
        "model.layers.3.self_attn.indexer.wk.weight",
        "model.layers.3.unknown.weight",
        "model.layers.3.mlp.experts.256.up_proj.weight",
    ):
        try:
            classify_tensor(bad, config)
        except Glm52Error:
            rejected.append(bad)
    if len(rejected) != 4:
        raise AssertionError("fail-closed classifier accepted an invalid layout")
    return {
        "status": "PASS",
        "classified": classified,
        "invalid_layouts_rejected": rejected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    admit = subparsers.add_parser("admit")
    admit.add_argument("--workers", type=int, default=16)
    subparsers.add_parser("refresh-admission-offline")
    subparsers.add_parser("selfcheck")
    args = parser.parse_args()
    if args.command == "selfcheck":
        result = selfcheck()
    elif args.command == "refresh-admission-offline":
        result = refresh_source_admission_offline()
    else:
        result = run_admission(workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
