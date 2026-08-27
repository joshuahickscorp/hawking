"""Pre-runtime Flash-Next architecture and joint-promotion science.

This lane reads only pinned upstream metadata and safetensors headers.  It
creates an architecture fingerprint, an organ graph, structural active-compute
bounds, and a representation-native Gravity/Accelerator worklist.  Header
reads are bounded HTTP range requests; tensor bodies are never downloaded or
loaded.  The lane does not compile a native executable or promote a metadata
observation into a capability/performance claim.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.flash_next import (
    ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN,
    COMPLETE_SYSTEM_EBPW_MAX,
    EXPECTED_BYTES,
    EXPECTED_FILE_COUNT,
    PINNED_REVISION,
    REPO_ID,
    evaluate_flash_promotion,
)
from hcli.persist import atomic_write_json


SCHEMA = "hcli.flash-next.pre-runtime-science.v1"
UPSTREAM_SOURCE = "UPSTREAM_SOURCE"
DERIVED = "DERIVED"
LOCAL_PHYSICAL = "LOCAL_PHYSICAL"
MODEL_PAGE = f"https://huggingface.co/{REPO_ID}"
RAW_BASE = f"https://huggingface.co/{REPO_ID}/resolve/{PINNED_REVISION}/"
METADATA_FILES = (
    ("config.json", 8 * 1024 * 1024),
    ("tokenizer.json", 32 * 1024 * 1024),
    ("tokenizer_config.json", 8 * 1024 * 1024),
    ("model.safetensors.index.json", 8 * 1024 * 1024),
    ("README.md", 8 * 1024 * 1024),
    ("generation_config.json", 8 * 1024 * 1024),
    ("preprocessor_config.json", 8 * 1024 * 1024),
)
HEADER_PROBE_BYTES = 64 * 1024
MAX_HEADER_BYTES = 64 * 1024 * 1024
GRAVITY_LADDER = (
    "ELIMINATE",
    "REPARAMETERIZE",
    "SHARE",
    "FACTORIZE",
    "GENERATE",
    "ROUTE",
    "INFORMATION_ASSIGNMENT",
    "HEAL",
    "QUANTIZE",
    "NATIVE_OPERATORS",
    "STATE_OPTIMIZATION",
    "COMPUTE_REMOVAL",
    "DECODE_STEP_REMOVAL",
    "DEVICE_COMPILE",
    "VERIFY",
)
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "BOOL": 1,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(name: str, limit: int, timeout_s: float) -> Dict[str, Any]:
    url = RAW_BASE + name
    request = urllib.request.Request(url, headers={"User-Agent": "hawking-hcli/1"})
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_s))) as response:
            data = response.read(limit + 1)
            content_type = response.headers.get("Content-Type")
    except Exception as exc:  # noqa: BLE001 - preserve each metadata boundary
        return {"name": name, "url": url, "label": UPSTREAM_SOURCE, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
    if len(data) > limit:
        return {"name": name, "url": url, "label": UPSTREAM_SOURCE, "status": "REFUSED_TOO_LARGE", "max_bytes": limit}
    row: Dict[str, Any] = {
        "name": name,
        "url": url,
        "revision": PINNED_REVISION,
        "label": UPSTREAM_SOURCE,
        "status": "FETCHED",
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "content_type": content_type,
        "data": data,
    }
    if name.endswith(".json"):
        try:
            row["parsed"] = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            row["parse_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _range_bytes(name: str, start: int, end: int, timeout_s: float) -> bytes:
    """Read a bounded pinned-source byte range; never request a weight body."""
    request = urllib.request.Request(
        RAW_BASE + name,
        headers={
            "Range": f"bytes={int(start)}-{int(end)}",
            "User-Agent": "hawking-hcli/1",
        },
    )
    with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_s))) as response:
        status = getattr(response, "status", None)
        content_range = str(response.headers.get("Content-Range") or "")
        if status != 206 or not content_range.startswith(f"bytes {int(start)}-"):
            raise ValueError(f"{name}: range response status={status!r} content-range={content_range!r}")
        data = response.read(int(end) - int(start) + 1)
    expected = int(end) - int(start) + 1
    if len(data) != expected:
        raise ValueError(f"{name}: short range read {len(data)} != {expected}")
    return data


def _fetch_safetensor_header(name: str, timeout_s: float) -> Dict[str, Any]:
    """Fetch and parse only a safetensors header from the pinned source."""
    probe = _range_bytes(name, 0, HEADER_PROBE_BYTES - 1, timeout_s)
    if len(probe) < 8:
        raise ValueError(f"{name}: safetensors header length is absent")
    header_bytes = struct.unpack("<Q", probe[:8])[0]
    if header_bytes <= 0 or header_bytes > MAX_HEADER_BYTES:
        raise ValueError(f"{name}: unsafe header length {header_bytes}")
    required = 8 + int(header_bytes)
    if len(probe) < required:
        probe = _range_bytes(name, 0, required - 1, timeout_s)
    try:
        header = json.loads(probe[8:required].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name}: invalid safetensors header: {exc}") from exc
    if not isinstance(header, Mapping):
        raise ValueError(f"{name}: safetensors header is not an object")
    tensors: Dict[str, Dict[str, Any]] = {}
    payload_bytes = 0
    for tensor_name, raw in header.items():
        if tensor_name == "__metadata__":
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name}: tensor {tensor_name!r} metadata is not an object")
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        dtype = str(raw.get("dtype") or "UNKNOWN")
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"{name}: tensor {tensor_name!r} lacks shape/data_offsets")
        try:
            clean_shape = [int(value) for value in shape]
            begin, end = int(offsets[0]), int(offsets[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: tensor {tensor_name!r} has invalid shape/offsets") from exc
        if any(value < 0 for value in clean_shape) or begin < 0 or end < begin:
            raise ValueError(f"{name}: tensor {tensor_name!r} has invalid shape/offsets")
        tensor_bytes = end - begin
        payload_bytes += tensor_bytes
        tensors[str(tensor_name)] = {
            "shape": clean_shape,
            "dtype": dtype,
            "data_offsets": [begin, end],
            "payload_bytes": tensor_bytes,
            "shard": name,
            "source_label": UPSTREAM_SOURCE,
        }
    return {
        "file": name,
        "status": "PARSED",
        "header_bytes": int(header_bytes),
        "tensor_count": len(tensors),
        "payload_bytes": payload_bytes,
        "tensors": tensors,
    }


def _fetch_tensor_headers(shards: list[str], timeout_s: float) -> Dict[str, Any]:
    """Audit source tensor layouts using parallel bounded header range reads."""
    rows: Dict[str, Dict[str, Any]] = {}
    workers = min(12, max(1, len(shards)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="flash-header") as pool:
        futures = {pool.submit(_fetch_safetensor_header, shard, timeout_s): shard for shard in shards}
        for future in as_completed(futures):
            shard = futures[future]
            try:
                rows[shard] = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve each source boundary
                rows[shard] = {
                    "file": shard,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "header_bytes": None,
                    "tensor_count": 0,
                    "payload_bytes": 0,
                    "tensors": {},
                }
    tensor_headers: Dict[str, Dict[str, Any]] = {}
    duplicate_names: list[str] = []
    for shard in sorted(rows):
        for tensor_name, layout in (rows[shard].get("tensors") or {}).items():
            if tensor_name in tensor_headers:
                duplicate_names.append(tensor_name)
                continue
            tensor_headers[tensor_name] = layout
    return {
        "status": "PARSED" if all(rows[shard].get("status") == "PARSED" for shard in shards) else "PARTIAL",
        "requested_shards": len(shards),
        "parsed_shards": sum(rows[shard].get("status") == "PARSED" for shard in shards),
        "failed_shards": [rows[shard] for shard in sorted(rows) if rows[shard].get("status") != "PARSED"],
        "header_bytes": sum(int(rows[shard].get("header_bytes") or 0) for shard in rows),
        "payload_bytes": sum(int(rows[shard].get("payload_bytes") or 0) for shard in rows),
        "duplicate_tensor_names": sorted(set(duplicate_names)),
        "per_shard": [
            {key: value for key, value in rows[shard].items() if key != "tensors"}
            for shard in sorted(rows)
        ],
        "tensor_headers": tensor_headers,
    }


def _product(values: Any) -> Optional[int]:
    if not isinstance(values, list):
        return None
    result = 1
    try:
        for value in values:
            result *= int(value)
    except (TypeError, ValueError):
        return None
    return result


def _tensor_flops(name: str, layout: Mapping[str, Any]) -> int:
    """Approximate two-operation-per-value arithmetic for source tensors."""
    lower = name.lower()
    shape = layout.get("shape")
    elements = _product(shape)
    if elements is None or len(shape) < 2:
        return 0
    if "ngram_embedding.shard_" in lower or "embed_tokens" in lower or lower.endswith("pos_embed.weight"):
        return 0
    if lower.endswith(".bias") or lower.endswith("a_log") or lower.endswith("dt_bias"):
        return 0
    return int(elements) * 2


def _dtype_bytes(layouts: Mapping[str, Mapping[str, Any]]) -> Optional[int]:
    values = {DTYPE_BYTES.get(str(layout.get("dtype")).upper()) for layout in layouts.values()}
    values.discard(None)
    if len(values) == 1:
        return next(iter(values))
    return None


def _metric(value: Optional[int | float], *, status: str = "STRUCTURAL_ESTIMATE", **extra: Any) -> Dict[str, Any]:
    return {"value": value, "label": DERIVED, "status": status, **extra}


def _gravity_science_plan() -> Dict[str, Any]:
    """Return the ordered, non-mutating Gravity worklist for Flash-Next.

    The ladder is intentionally a plan rather than a claim.  Each candidate
    has to earn source parity, accepted-token accounting, and a complete
    device receipt before it can affect a runtime or promotion decision.
    """
    specs = [
        (
            "ELIMINATE",
            ["duplicate DeltaNet state copies", "unused vision work on text-only decode", "rejected MTP work after acceptance accounting"],
            ["deltanet", "vision_backbone", "mtp"],
            ["state/activation trace", "text-only versus multimodal parity", "accepted and rejected token ledger"],
        ),
        (
            "REPARAMETERIZE",
            ["fused gate/up/SwiGLU layout", "projection and residual layout", "low-rank hyperconnection placement"],
            ["shared_expert", "routed_experts", "residual_hyperconnections"],
            ["exact tensor transform proof", "same-model numerical parity", "no hidden dense rematerialization"],
        ),
        (
            "SHARE",
            ["routed/shared expert basis", "residual hyperconnection basis", "n-gram dictionaries and route metadata"],
            ["routed_experts", "shared_expert", "ngram_engine", "router"],
            ["independent-information accounting", "collision/error bounds", "source-to-runtime ownership map"],
        ),
        (
            "FACTORIZE",
            ["expert bank into shared basis plus residual", "n-gram lookup representation", "candidate vocabulary projection"],
            ["routed_experts", "ngram_engine", "lm_head"],
            ["factorization reconstruction parity", "complete output distribution parity", "measured bytes after representation"],
        ),
        (
            "GENERATE",
            ["compositional n-gram rows", "expert residual reconstruction", "on-device route metadata"],
            ["ngram_engine", "routed_experts", "router"],
            ["generator versus stored-row parity", "latency and bandwidth trace", "no uncounted generated work"],
        ),
        (
            "ROUTE",
            ["router top-k expert assignment", "QSA indexer-budget block traversal", "candidate vocabulary routing"],
            ["router", "sparse_attention", "lm_head"],
            ["route histogram", "selected-index trace", "output capability parity"],
        ),
        (
            "INFORMATION_ASSIGNMENT",
            ["assign tokens to experts", "assign attention budget to blocks", "assign accepted work to decode steps"],
            ["router", "sparse_attention", "mtp"],
            ["zero-independent-information test", "budget conservation", "accepted-token definition"],
        ),
        (
            "HEAL",
            ["residual/hyperconnection correction path", "state rollback on rejected draft", "numerical repair/guard bands"],
            ["residual_hyperconnections", "recurrent_state", "mtp"],
            ["failure injection", "rollback parity", "bounded error and no silent fallback"],
        ),
        (
            "QUANTIZE",
            ["representation-native NF expert storage", "low-bit router/projection candidates", "quantized lookup payloads"],
            ["routed_experts", "router", "ngram_engine"],
            ["same-source dense-vs-NF A/B", "per-organ scales/packing", "capability-preserving quality gate"],
        ),
        (
            "NATIVE_OPERATORS",
            ["NF GEMV and fused epilogue", "DeltaNet state update", "QSA gather/reduction", "n-gram lookup/generator", "MTP accept/reject"],
            ["routed_experts", "deltanet", "sparse_attention", "ngram_engine", "mtp"],
            ["kernel genome", "dispatch trace", "reference-vector parity", "complete-token meter"],
        ),
        (
            "STATE_OPTIMIZATION",
            ["resident DeltaNet F32 state", "sparse KV-cache locality", "MTP state snapshot/rollback", "expert residency"],
            ["recurrent_state", "sparse_attention", "mtp", "routed_experts"],
            ["resident-state bytes", "sequence isolation", "rollback trace", "no per-token weight transfer"],
        ),
        (
            "COMPUTE_REMOVAL",
            ["skip vision backbone for text-only requests", "remove duplicate epilogue launches", "avoid unselected experts"],
            ["vision_backbone", "norms", "routed_experts"],
            ["request-mode proof", "dispatch absence", "selected-expert parity"],
        ),
        (
            "DECODE_STEP_REMOVAL",
            ["MTP accepted-token step reduction", "n-gram-assisted accepted work", "eliminate rejected draft execution"],
            ["mtp", "ngram_engine"],
            ["accepted complete tokens per wall time", "draft/verify/rollback ledger", "no primitive-only TPS"],
        ),
        (
            "DEVICE_COMPILE",
            ["FPGA HBM expert/router placement", "resident DeltaNet state pipeline", "sparse/lookup transport and command graph"],
            ["routed_experts", "router", "recurrent_state", "sparse_attention", "ngram_engine"],
            ["device identity", "bitstream/module hash", "HWIR fingerprint", "transport trace"],
        ),
        (
            "VERIFY",
            ["pinned source identity", "organ ownership and byte ledger", "same-model parity", "capability-preserving complete-token rate"],
            ["all_organs"],
            ["all required byte fields", "fallback disclosure", "protected quiescent receipt", "promotion thresholds"],
        ),
    ]
    stages = []
    for order, (stage, candidates, organs, evidence) in enumerate(specs, start=1):
        stages.append({
            "order": order,
            "stage": stage,
            "display_name": "DECODE-STEP REMOVAL" if stage == "DECODE_STEP_REMOVAL" else stage,
            "label": DERIVED,
            "status": "PLAN_ONLY",
            "candidate_hypotheses": candidates,
            "organs": organs,
            "evidence_required": evidence,
            "source_mutation_allowed": False,
            "native_representation_required": True,
        })
    return {
        "status": "PLAN_ONLY",
        "label": DERIVED,
        "ordered": True,
        "ladder": list(GRAVITY_LADDER),
        "stages": stages,
        "source_mutation_policy": "No source, profile, or pinned artifact mutation is authorized by this pre-runtime plan.",
        "claim_boundary": "Gravity candidates are derived hypotheses. They do not establish quality, speed, capability, or promotion.",
    }


def _three_zero_questions() -> Dict[str, Any]:
    """Record the three zero questions without converting them into claims."""
    return {
        "label": DERIVED,
        "status": "UNRESOLVED_PLAN",
        "storage": {
            "question": "Can required persistent storage be driven to zero without deleting information or changing the accepted function?",
            "status": "NOT_PROVEN",
            "current_answer": "No zero-storage claim. Source weights, recurrent state, cache state, and any accepted-token state remain explicit requirements or conditional costs.",
            "candidates": ["eliminate duplicate state copies", "share/factorize expert and dictionary storage", "remove unused text-only vision storage from the active working set"],
            "evidence_required": ["exact byte ledger", "reconstruction/parity proof", "resident-state and cache receipt"],
        },
        "independent_information": {
            "question": "Can representations share or be generated so that independent information reaches zero while preserving the model function?",
            "status": "NOT_PROVEN",
            "current_answer": "No zero-independent-information claim. Shared bases, residuals, dictionaries, and route metadata are candidates whose residual information must be measured.",
            "candidates": ["shared expert basis plus residual", "low-rank hyperconnection basis", "shared n-gram dictionary and route metadata"],
            "evidence_required": ["rank/residual accounting", "collision and reconstruction bounds", "same-source capability parity"],
        },
        "execution": {
            "question": "Can an execution path be removed to zero work without changing accepted complete-token behavior?",
            "status": "NOT_PROVEN",
            "current_answer": "Only conditional removal is allowed. Text-only vision and rejected speculative work may be skipped only when request mode, state, and accepted-token parity are proven.",
            "candidates": ["text-only vision bypass", "unselected-expert elimination", "MTP rejected-work and decode-step removal"],
            "evidence_required": ["dispatch absence trace", "accepted-token ledger", "state rollback/parity receipt"],
        },
        "claim_boundary": "The three zeros are questions and experiment gates, not achieved reductions.",
    }


def _accelerator_primitive_plan() -> Dict[str, Any]:
    """Map every Flash candidate primitive to a current capability and gap."""
    rows = [
        ("packed_low_bit_gemv", "shared_expert", "packed low-bit GEMV is named in the Qwen27/FPGA pre-board scaffold", "Flash NF packing, scales, and same-model parity are not implemented or measured", "P0"),
        ("native_nf_expert_gemv", "routed_experts", "packed low-bit GEMV schema and expert-bank placement hypothesis", "no Flash-native NF decode kernel, physical dispatch trace, or performance receipt", "P0"),
        ("router_topk_gather", "router", "route metadata/partial-reduction transport is present in the Flash HWIR map", "no runtime top-k/gather kernel or load-balance trace", "P0"),
        ("fused_route_gather", "router", "Flash HWIR keeps route metadata and expert-bank placement adjacent", "no fused route-to-expert gather implementation or route/latency trace", "P0"),
        ("expert_residency_scheduling", "routed_experts", "resident weight-shard policy and expert-bank HBM hypothesis exist", "no residency scheduler, eviction trace, or per-token transfer proof", "P0"),
        ("basis_residual_decode", "routed_experts", "shared-basis/residual is a recorded Gravity candidate", "no exact reconstruction proof or resident decoder", "P0"),
        ("fused_gate_up_swiglu", "shared_expert", "Qwen fusion source audit and norm/epilogue scaffold exist", "Flash tensor ownership and fused dispatch parity are unverified", "P0"),
        ("rmsnorm_epilogue", "norms", "norm/epilogue is a shared pre-board primitive", "no Flash-native kernel or physical fusion trace", "P1"),
        ("persistent_deltanet_state_update", "deltanet", "Qwen27 persistent state/update and resident-state hypothesis exist", "Flash state geometry, sequence isolation, and update parity are unimplemented", "P0"),
        ("deltanet_scan_state", "deltanet", "the Flash organ graph identifies repeated linear-attention layers and virtual recurrent state", "no scan kernel, state transition oracle, or physical state bandwidth trace", "P0"),
        ("qsa_sparse_indexer_kv_gather", "sparse_attention", "Flash HWIR names indexer traversal and sparse KV gather", "no sparse kernel, budget trace, or context-dependent physical bytes", "P1"),
        ("ngram_lookup_generator", "ngram_engine", "Flash organ graph and lookup/generator worklist exist", "no representation-native lookup/generator backend or parity oracle", "P1"),
        ("mtp_accept_reject_rollback", "mtp", "MTP acceptance accounting is specified as a required primitive", "no accepted-token runtime, rollback implementation, or decode-step proof", "P1"),
        ("lm_head_partitioned_topk", "lm_head", "Qwen27 vocabulary reduction/selection is a pre-board precedent", "Flash vocabulary path and complete distribution parity are unverified", "P1"),
        ("embedding_hbm_gather", "embeddings", "HBM gather/row-cache is a derived organ primitive", "no cache locality trace or device implementation", "P2"),
        ("low_rank_hyperconnection_mix", "residual_hyperconnections", "Flash source census identifies low-rank hyperconnections", "no existing accelerator kernel or parity implementation", "P1"),
        ("conditional_vision_bypass", "vision_backbone", "text-only zero-compute candidate is represented in the organ graph", "no request-mode dispatch proof or multimodal parity harness", "P1"),
        ("complete_token_acceptance_meter", "mtp", "HCLI receipt/verifier and unattended complete-work accounting exist", "Flash-native accepted-token and rollback fields are not wired to a runtime", "P0"),
        ("kernel_genome_and_receipt_verifier", "all_organs", "kernel genome/cache and telemetry/receipt verifier are shared pre-board capabilities", "no Flash compiled kernel genome or physical device receipt", "P0"),
        ("fpga_hwir_link_transport", "all_organs", "Qwen27/Flash HWIR, partitioner, and link sensitivity simulation exist", "no selected board, bitstream, DMA bridge, or hardware timing", "P1"),
    ]
    entries = [
        {
            "primitive": primitive,
            "organ": organ,
            "label": DERIVED,
            "status": "PLAN_ONLY",
            "priority": priority,
            "existing_capability": capability,
            "gap": gap,
            "evidence_required": ["same-model reference parity", "capability receipt", "complete-token wall/GPU/dispatch trace"],
        }
        for primitive, organ, capability, gap, priority in rows
    ]
    return {
        "status": "PLAN_ONLY",
        "label": DERIVED,
        "candidate_classes": [
            "low-bit GEMV",
            "expert routing",
            "fused route/gather",
            "expert execution",
            "DeltaNet scan/state",
            "sparse attention",
            "MTP",
            "norms",
            "epilogues",
            "persistent state",
            "expert residency scheduling",
        ],
        "entries": entries,
        "existing_capability_sources": [
            "receipts/headless/HCLI_FPGA_PREBOARD.json",
            "receipts/headless/QWEN27_FPGA_ORGAN_MAP.json",
            "receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json",
            "receipts/headless/HCLI_QWEN38_FUSION_SOURCE_AUDIT.json",
        ],
        "physical_execution_claim": False,
        "claim_boundary": "Capability names are scaffold/pre-board precedents. Every Flash primitive must re-earn same-model parity, physical execution, and complete-token accounting.",
    }


def _state_record(organ: str, text: Mapping[str, Any], dtype_bytes: int) -> Dict[str, Any]:
    if organ == "deltanet":
        key_heads = _as_int(text.get("linear_num_key_heads")) or 0
        key_dim = _as_int(text.get("linear_key_head_dim")) or 0
        value_heads = _as_int(text.get("linear_num_value_heads")) or 0
        value_dim = _as_int(text.get("linear_value_head_dim")) or 0
        conv_kernel = _as_int(text.get("linear_conv_kernel_dim")) or 0
        linear_layers = _as_int(text.get("num_hidden_layers")) or 0
        layer_types = text.get("layer_types") if isinstance(text.get("layer_types"), list) else []
        linear_layers = layer_types.count("linear_attention") or linear_layers
        conv_channels = key_heads * key_dim * 2 + value_heads * value_dim
        conv_per_layer = conv_channels * max(0, conv_kernel - 1) * 4
        recurrent_per_layer = value_heads * key_dim * value_dim * 4
        resident = linear_layers * (conv_per_layer + recurrent_per_layer)
        return _metric(
            resident,
            status="STRUCTURAL_ESTIMATE",
            resident_bytes=resident,
            read_write_bytes_per_token=resident * 2,
            dtype="F32",
            dimensions={
                "linear_layers": linear_layers,
                "conv_channels": conv_channels,
                "conv_state_elements_per_layer": conv_channels * max(0, conv_kernel - 1),
                "recurrent_state_elements_per_layer": value_heads * key_dim * value_dim,
            },
            formula="linear_layers * (conv_channels * (conv_kernel - 1) + value_heads * key_dim * value_dim) * 4",
        )
    if organ == "sparse_attention":
        layers = _as_int(text.get("num_hidden_layers")) or 0
        layer_types = text.get("layer_types") if isinstance(text.get("layer_types"), list) else []
        full_layers = layer_types.count("full_attention")
        kv_heads = _as_int(text.get("num_key_value_heads")) or 0
        head_dim = _as_int(text.get("head_dim")) or 0
        append = full_layers * 2 * kv_heads * head_dim * dtype_bytes
        return _metric(
            None,
            status="PARAMETERIZED_STRUCTURAL_ESTIMATE",
            resident_bytes=None,
            per_token_append_bytes=append,
            context_read_bytes_formula=f"{full_layers} * 2 * {kv_heads} * {head_dim} * {dtype_bytes} * context_length",
            dimensions={"full_attention_layers": full_layers, "kv_heads": kv_heads, "head_dim": head_dim, "all_layers": layers},
        )
    if organ == "recurrent_state":
        deltanet = _state_record("deltanet", text, dtype_bytes)
        return _metric(
            deltanet.get("resident_bytes"),
            status="STRUCTURAL_ESTIMATE",
            resident_bytes=deltanet.get("resident_bytes"),
            read_write_bytes_per_token=deltanet.get("read_write_bytes_per_token"),
            owner="deltanet",
            formula=deltanet.get("formula"),
        )
    if organ == "mtp":
        hidden = _as_int(text.get("hidden_size")) or 0
        return _metric(0, status="CONDITIONAL_STRUCTURAL_ESTIMATE", conditional_state_bytes=hidden * dtype_bytes, conditional_on="MTP proposal/verification path")
    return _metric(0, status="NO_PERSISTENT_STATE_IN_PRE_RUNTIME_MODEL")


def _organ_matchers() -> list[tuple[str, Any]]:
    return [
        ("mtp", lambda name: name.lower().startswith("mtp.")),
        ("vision_backbone", lambda name: name.lower().startswith("model.visual.")),
        ("embeddings", lambda name: name.lower() == "model.language_model.embed_tokens.weight"),
        ("deltanet", lambda name: name.lower().startswith("model.language_model.layers.") and ".linear_attn." in name.lower()),
        ("sparse_attention", lambda name: name.lower().startswith("model.language_model.layers.") and ".self_attn." in name.lower()),
        ("router", lambda name: name.lower().startswith("model.language_model.layers.") and ".mlp.gate.weight" in name.lower()),
        ("routed_experts", lambda name: name.lower().startswith("model.language_model.layers.") and ".mlp.experts." in name.lower()),
        ("shared_expert", lambda name: name.lower().startswith("model.language_model.layers.") and ".mlp.shared_expert" in name.lower()),
        ("norms", lambda name: "norm" in name.lower()),
        ("residual_hyperconnections", lambda name: "hyper_connection" in name.lower()),
        ("lm_head", lambda name: name.lower() == "lm_head.weight"),
        ("ngram_engine", lambda name: ".ple." in name.lower()),
    ]


def _organ_dimensions(organ: str, text: Mapping[str, Any], config: Mapping[str, Any], matched: list[str], layouts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    layer_types = text.get("layer_types") if isinstance(text.get("layer_types"), list) else []
    common = {
        "hidden_size": text.get("hidden_size"),
        "layers": text.get("num_hidden_layers"),
        "experts": text.get("num_experts"),
        "experts_per_token": text.get("num_experts_per_tok"),
        "vocab_size": text.get("vocab_size"),
    }
    dimensions: Dict[str, Any]
    if organ == "embeddings":
        dimensions = {"vocab_size": text.get("vocab_size"), "hidden_size": text.get("hidden_size")}
    elif organ == "deltanet":
        dimensions = {
            "layer_types": layer_types,
            "linear_layers": layer_types.count("linear_attention"),
            "linear_key_head_dim": text.get("linear_key_head_dim"),
            "linear_value_head_dim": text.get("linear_value_head_dim"),
            "linear_num_key_heads": text.get("linear_num_key_heads"),
            "linear_num_value_heads": text.get("linear_num_value_heads"),
            "conv_kernel_dim": text.get("linear_conv_kernel_dim"),
        }
    elif organ == "sparse_attention":
        dimensions = {
            "full_attention_layers": layer_types.count("full_attention"),
            "indexer_budget": text.get("indexer_budget"),
            "indexer_compress_ratio": text.get("indexer_compress_ratio"),
            "indexer_head_dim": text.get("indexer_head_dim"),
            "indexer_kv_heads": text.get("indexer_kv_heads"),
            "indexer_n_heads": text.get("indexer_n_heads"),
            "attention_heads": text.get("num_attention_heads"),
            "kv_heads": text.get("num_key_value_heads"),
            "head_dim": text.get("head_dim"),
        }
    elif organ == "router":
        dimensions = {**common, "output_gate_type": text.get("output_gate_type"), "router_aux_loss_coef": text.get("router_aux_loss_coef")}
    elif organ in {"routed_experts", "shared_expert"}:
        dimensions = {**common, "moe_intermediate_size": text.get("moe_intermediate_size"), "shared_expert_intermediate_size": text.get("shared_expert_intermediate_size")}
    elif organ == "ngram_engine":
        dimensions = {"ngram_vocab_size_base": text.get("ngram_vocab_size_base"), "ngram_size": text.get("ngram_size"), "split_ngram_parts": text.get("split_ngram_parts"), "ple_layer_ids": text.get("ple_layer_ids"), "ple_embed_dim": text.get("ple_embed_dim")}
    elif organ == "mtp":
        dimensions = {"mtp": text.get("mtp"), "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"), "dedicated_embeddings": text.get("mtp_use_dedicated_embeddings")}
    elif organ == "vision_backbone":
        dimensions = config.get("vision_config") if isinstance(config.get("vision_config"), Mapping) else {}
    elif organ == "norms":
        dimensions = {"cross_cutting": True, "norm_tensor_count": len(matched), "overlaps_primary_organs": ["deltanet", "sparse_attention", "ngram_engine", "mtp", "residual_hyperconnections"]}
    elif organ == "residual_hyperconnections":
        dimensions = {"hc_count": text.get("hc_count"), "hc_lowrank": text.get("hc_lowrank"), "hidden_size": text.get("hidden_size")}
    elif organ == "recurrent_state":
        dimensions = {"runtime_only": True, "owner": "deltanet", "dtype": "F32"}
    elif organ == "lm_head":
        dimensions = {"hidden_size": text.get("hidden_size"), "vocab_size": text.get("vocab_size")}
    else:
        dimensions = dict(common)
    dimensions["tensor_shape_coverage"] = {"available": sum(name in layouts for name in matched), "total": len(matched)}
    return dimensions


def _organ_graph(
    config: Mapping[str, Any],
    weight_map: Mapping[str, Any],
    shard_sizes: Mapping[str, int],
    tensor_headers: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> list[Dict[str, Any]]:
    """Build a disjoint source census plus explicit cross-cutting/state views."""
    text = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    names = sorted(str(name) for name in weight_map)
    layouts = {str(name): dict(value) for name, value in (tensor_headers or {}).items()}
    assigned: set[str] = set()
    matched_by_organ: Dict[str, list[str]] = {}
    for organ, matcher in _organ_matchers():
        if organ == "norms":
            # Norms is deliberately a cross-cutting view; its bytes are not additive.
            matched_by_organ[organ] = [name for name in names if matcher(name)]
            continue
        matched = [name for name in names if name not in assigned and matcher(name)]
        assigned.update(matched)
        matched_by_organ[organ] = matched
    matched_by_organ["support_misc"] = [name for name in names if name not in assigned]
    matched_by_organ["recurrent_state"] = []

    plan: Dict[str, Dict[str, Any]] = {
        "embeddings": {
            "regularity": "single dense vocabulary lookup; one row per input token",
            "bottleneck": "embedding lookup locality and token-to-row gather",
            "gravity": ["factorize vocabulary representation", "generate/compress lookup rows", "resident row cache"],
            "primitives": ["partitioned embedding lookup", "HBM gather", "row cache"],
        },
        "deltanet": {
            "regularity": "regular repeated layer schedule with resident recurrent state",
            "bottleneck": "state movement plus projection bandwidth",
            "gravity": ["eliminate redundant state copies", "reparameterize projection layout", "native recurrent operator", "state optimization"],
            "primitives": ["fused projection/rearrange", "persistent state update", "resident F32 stream", "norm/epilogue"],
        },
        "sparse_attention": {
            "regularity": "periodic full-attention layers with budgeted sparse selection",
            "bottleneck": "irregular indexer, KV-cache gather, and reduction",
            "gravity": ["route budgeted blocks", "information assignment", "native sparse traversal", "decode-step accounting"],
            "primitives": ["indexer projection", "top-budget selection", "KV gather", "attention reduction"],
        },
        "router": {
            "regularity": "regular router GEMV followed by irregular top-k assignment",
            "bottleneck": "routing latency and expert-index movement",
            "gravity": ["route before materialization", "information assignment", "share route metadata"],
            "primitives": ["sigmoid router", "top-k selection", "expert index compaction", "route histogram"],
        },
        "routed_experts": {
            "regularity": "irregular selected-expert access over a regular fused expert layout",
            "bottleneck": "HBM bandwidth, gather/scatter, and expert load imbalance",
            "gravity": ["share basis/residual", "factorize expert bank", "route locality", "quantize after representation choice", "native expert GEMV"],
            "primitives": ["packed NF GEMV", "basis/residual decode", "expert gather", "partial reduction", "load balancing"],
        },
        "shared_expert": {
            "regularity": "regular dense path executed with every routed token",
            "bottleneck": "dense projection bandwidth and epilogue synchronization",
            "gravity": ["share with routed path", "reparameterize fused gate/up", "native epilogue"],
            "primitives": ["fused gate/up", "SwiGLU epilogue", "dense GEMV", "partial reduction"],
        },
        "norms": {
            "regularity": "elementwise and small-vector operations, cross-cutting across organs",
            "bottleneck": "launch overhead and synchronization, not tensor volume",
            "gravity": ["eliminate launches", "fuse with producer/consumer", "native epilogue"],
            "primitives": ["RMSNorm/LayerNorm", "elementwise fuse", "activation epilogue"],
        },
        "recurrent_state": {
            "regularity": "persistent per-sequence state with regular read/modify/write",
            "bottleneck": "state bandwidth and residency pressure",
            "gravity": ["state optimization", "eliminate copies", "resident execution", "verify state parity"],
            "primitives": ["persistent state SRAM/HBM", "F32 read/modify/write", "sequence isolation", "rollback"],
        },
        "lm_head": {
            "regularity": "regular vocabulary projection and reduction",
            "bottleneck": "vocabulary bandwidth and final reduction",
            "gravity": ["factorize output projection", "route candidate vocabulary", "native reduction", "verify sampling contract"],
            "primitives": ["partitioned vocabulary GEMV", "top-k reduction", "sampling"],
        },
        "mtp": {
            "regularity": "conditional draft/verify subgraph with rollback boundary",
            "bottleneck": "extra work unless accepted tokens reduce decode steps",
            "gravity": ["decode-step removal", "eliminate rejected work", "verify accepted-token accounting", "state rollback"],
            "primitives": ["draft/verify graph", "accept/reject", "rollback", "complete-token meter"],
        },
        "ngram_engine": {
            "regularity": "irregular bounded lookup plus small projection path",
            "bottleneck": "lookup latency, dictionary locality, and representation expansion",
            "gravity": ["factorize", "generate", "share dictionaries", "information assignment", "native lookup"],
            "primitives": ["partitioned lookup", "generator", "dictionary decode", "gather/reduction"],
        },
        "vision_backbone": {
            "regularity": "regular patch/vision block path, conditional on multimodal input",
            "bottleneck": "conditional vision compute and projection into text space",
            "gravity": ["compute removal for text-only decode", "factorize patch path", "native vision operators"],
            "primitives": ["patch embedding", "vision attention", "vision MLP", "multimodal merger"],
        },
        "residual_hyperconnections": {
            "regularity": "repeated low-rank mixing at each layer boundary",
            "bottleneck": "small GEMV launches and activation movement",
            "gravity": ["eliminate intermediate copies", "share low-rank basis", "fuse residual path"],
            "primitives": ["low-rank mix", "residual combine", "activation routing"],
        },
        "support_misc": {
            "regularity": "source tensors not assigned to a named execution organ",
            "bottleneck": "unresolved until runtime attribution",
            "gravity": ["verify ownership before transformation"],
            "primitives": ["tensor census", "runtime attribution"],
        },
    }

    dtype_bytes = _dtype_bytes(layouts) or 2
    layer_types = text.get("layer_types") if isinstance(text.get("layer_types"), list) else []
    full_layers = layer_types.count("full_attention")
    kv_heads = _as_int(text.get("num_key_value_heads")) or 0
    head_dim = _as_int(text.get("head_dim")) or 0
    kv_append = full_layers * 2 * kv_heads * head_dim * dtype_bytes
    rows: list[Dict[str, Any]] = []
    # D2 order is intentional: it is also the stable map order for handoff readers.
    order = [
        "embeddings", "deltanet", "sparse_attention", "router", "routed_experts",
        "shared_expert", "norms", "recurrent_state", "lm_head", "mtp", "ngram_engine",
        "vision_backbone", "residual_hyperconnections", "support_misc",
    ]
    for organ in order:
        matched = list(matched_by_organ.get(organ) or [])
        organ_plan = plan[organ]
        organ_layouts = {name: layouts[name] for name in matched if name in layouts}
        shards = sorted({str(weight_map[name]) for name in matched if name in weight_map})
        shard_bytes = sum(int(shard_sizes.get(shard) or 0) for shard in shards)
        tensor_layout = [
            {
                "name": name,
                "shape": organ_layouts[name].get("shape") if name in organ_layouts else None,
                "dtype": organ_layouts[name].get("dtype") if name in organ_layouts else None,
                "stored_bytes": organ_layouts[name].get("payload_bytes") if name in organ_layouts else None,
                "shard": str(weight_map.get(name)),
                "label": UPSTREAM_SOURCE if name in organ_layouts else DERIVED,
            }
            for name in matched
        ]
        stored_value = sum(int(layout.get("payload_bytes") or 0) for layout in organ_layouts.values()) if len(organ_layouts) == len(matched) else None
        stored_status = "EXACT_PINNED_SAFETENSOR_HEADER_PAYLOAD" if stored_value is not None and matched else (
            "RUNTIME_STATE_NO_TENSOR_PAYLOAD" if organ == "recurrent_state" else "NOT_MEASURED_HEADERS_INCOMPLETE"
        )
        stored = _metric(
            stored_value if organ != "recurrent_state" else 0,
            status=stored_status,
            source="UPSTREAM_SOURCE safetensors data_offsets" if stored_value is not None else None,
            tensor_payload_bytes=stored_value,
        )
        if organ == "norms":
            stored["accounting_role"] = "CROSS_CUTTING_VIEW_NOT_ADDITIVE"
        if organ == "recurrent_state":
            stored["accounting_role"] = "VIRTUAL_RUNTIME_STATE"
        active_value: Optional[int] = None
        active_lower: Optional[int] = None
        active_upper: Optional[int] = None
        flops_value: Optional[int] = None
        flops_lower: Optional[int] = None
        flops_upper: Optional[int] = None
        components: Dict[str, Any] = {}
        if organ == "recurrent_state":
            state = _state_record(organ, text, dtype_bytes)
            active_value = int(state.get("read_write_bytes_per_token") or 0)
            active_lower = active_value
            active_upper = active_value
            state_dimensions = _state_record("deltanet", text, dtype_bytes).get("dimensions") or {}
            flops_value = int(state_dimensions.get("recurrent_state_elements_per_layer") or 0) * int(state_dimensions.get("linear_layers") or 0) * 2
            flops_lower = flops_value
            flops_upper = flops_value
            components["state_read_write_bytes"] = active_value
        elif organ == "vision_backbone":
            active_value = active_lower = active_upper = 0
            flops_value = flops_lower = flops_upper = 0
            components["text_decode"] = 0
            components["multimodal_prompt"] = "conditional_on_visual_input"
        elif organ == "embeddings" and organ_layouts:
            embed = next(iter(organ_layouts.values()))
            shape = embed.get("shape") or []
            row_bytes = int(embed.get("payload_bytes") or 0) // max(1, int(shape[0])) if shape else 0
            active_value = active_lower = active_upper = row_bytes
            flops_value = flops_lower = flops_upper = 0
            components["one_lookup_row_bytes"] = row_bytes
        elif organ == "ngram_engine" and organ_layouts:
            ngram_shards = [layout for name, layout in organ_layouts.items() if ".ngram_embedding.shard_" in name.lower()]
            auxiliary_bytes = sum(int(layout.get("payload_bytes") or 0) for name, layout in organ_layouts.items() if ".ngram_embedding.shard_" not in name.lower())
            row_sizes = []
            for layout in ngram_shards:
                shape = layout.get("shape") or []
                if shape and int(shape[0]):
                    row_sizes.append(int(layout.get("payload_bytes") or 0) // int(shape[0]))
            row_bytes = min(row_sizes) if row_sizes else 0
            lookup_rows = max(1, _as_int(text.get("ngram_size")) or 1)
            active_lower = auxiliary_bytes + row_bytes
            active_upper = auxiliary_bytes + row_bytes * lookup_rows
            active_value = active_upper
            flops_value = flops_lower = flops_upper = sum(_tensor_flops(name, layout) for name, layout in organ_layouts.items())
            components.update({"auxiliary_projection_bytes": auxiliary_bytes, "lookup_row_bytes": row_bytes, "lookup_rows_lower": 1, "lookup_rows_nominal": lookup_rows})
        elif organ_layouts and len(organ_layouts) == len(matched):
            stored_payload = sum(int(layout.get("payload_bytes") or 0) for layout in organ_layouts.values())
            flops_all = sum(_tensor_flops(name, layout) for name, layout in organ_layouts.items())
            active_value = active_lower = stored_payload
            active_upper = stored_payload
            flops_value = flops_lower = flops_all
            flops_upper = flops_all
            if organ == "routed_experts":
                experts = max(1, _as_int(text.get("num_experts")) or 1)
                topk = max(1, _as_int(text.get("num_experts_per_tok")) or 1)
                active_value = active_lower = int(round(stored_payload * topk / experts))
                active_upper = stored_payload
                flops_value = flops_lower = int(round(flops_all * topk / experts))
                flops_upper = flops_all
                components.update({"total_experts": experts, "routed_experts_per_token": topk, "routed_expert_fraction": topk / experts})
            if organ == "sparse_attention":
                active_value += kv_append
                active_lower += kv_append
                active_upper += kv_append
                components["kv_cache_append_bytes_per_token"] = kv_append
            if organ == "mtp":
                active_lower = 0
                active_value = stored_payload
                active_upper = stored_payload
                flops_lower = 0
                flops_value = flops_all
                flops_upper = flops_all
                components["conditional_on"] = "MTP proposal/verification path"
        elif not matched:
            active_value = active_lower = active_upper = 0
            flops_value = flops_lower = flops_upper = 0
        state = _state_record(organ, text, dtype_bytes)
        metric_extra = {
            "lower_bound": active_lower,
            "upper_bound": active_upper,
            "units": "bytes",
            "basis": "source BF16/declared dtype payloads touched by one token; structural estimate, not a device trace",
            "components": components,
        }
        flop_extra = {
            "lower_bound": flops_lower,
            "upper_bound": flops_upper,
            "units": "FLOPs",
            "basis": "2 operations per matrix value, with routed expert fraction applied; structural estimate, not a device trace",
        }
        if organ == "norms":
            metric_extra["accounting_role"] = "CROSS_CUTTING_VIEW_NOT_ADDITIVE"
            flop_extra["accounting_role"] = "CROSS_CUTTING_VIEW_NOT_ADDITIVE"
        if organ == "recurrent_state":
            metric_extra["accounting_role"] = "VIRTUAL_RUNTIME_STATE"
            flop_extra["accounting_role"] = "VIRTUAL_RUNTIME_STATE"
        row = {
            "id": organ,
            "label": DERIVED,
            "accounting_role": "CROSS_CUTTING_VIEW" if organ == "norms" else ("VIRTUAL_RUNTIME_STATE" if organ == "recurrent_state" else "PRIMARY"),
            "dimensions": _organ_dimensions(organ, text, config, matched, organ_layouts),
            "tensors": len(matched),
            "tensor_names": matched,
            "tensor_name_examples": matched[:10],
            "tensor_layout": tensor_layout,
            "shards": shards,
            "shard_bytes_observed": shard_bytes,
            "shard_bytes_scope": "sum of unique upstream shard files containing this organ; overlaps and is not tensor allocation",
            "stored_bytes": stored,
            "bytes": stored,
            "active_bytes_per_token": _metric(active_value, status="STRUCTURAL_ESTIMATE" if active_value is not None else "NOT_MEASURED_HEADERS_INCOMPLETE", **metric_extra),
            "flops_per_token": _metric(flops_value, status="STRUCTURAL_ESTIMATE" if flops_value is not None else "NOT_MEASURED_HEADERS_INCOMPLETE", **flop_extra),
            "state_bytes": state,
            "regularity": _metric(organ_plan["regularity"], status="STRUCTURAL_ONLY"),
            "execution_regularity": organ_plan["regularity"],
            "bottleneck": _metric(organ_plan["bottleneck"], status="STRUCTURAL_HYPOTHESIS"),
            "bottleneck_class": organ_plan["bottleneck"],
            "candidate_gravity": organ_plan["gravity"],
            "gravity": {"status": "PLAN_ONLY", "techniques": organ_plan["gravity"], "native_representation_required": True},
            "required_accelerator_primitives": organ_plan["primitives"],
            "accelerator": {"status": "PLAN_ONLY", "primitives": organ_plan["primitives"], "native_kernel_required": True},
        }
        if organ == "sparse_attention":
            row["flops_per_token"]["dynamic_context_flops_formula"] = f"4 * {full_layers} * {text.get('num_attention_heads')} * {head_dim} * context_length"
            row["active_bytes_per_token"]["dynamic_context_read_formula"] = f"{full_layers} * 2 * {kv_heads} * {head_dim} * {dtype_bytes} * context_length"
        rows.append(row)
    return rows


def _active_compute_bounds(
    architecture: Mapping[str, Any],
    organ_graph: list[Mapping[str, Any]],
    text: Mapping[str, Any],
    tensor_header_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    primary = [row for row in organ_graph if row.get("accounting_role") == "PRIMARY"]
    base = [row for row in primary if row.get("id") not in {"mtp", "vision_backbone"}]

    def sum_metric(rows: list[Mapping[str, Any]], key: str, bound: str) -> Optional[int]:
        values = []
        for row in rows:
            metric = row.get(key)
            if not isinstance(metric, Mapping):
                return None
            value = metric.get(bound)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            values.append(int(value))
        return sum(values)

    weight_bytes_lower = sum_metric(base, "active_bytes_per_token", "lower_bound")
    weight_bytes_nominal = sum_metric(base, "active_bytes_per_token", "value")
    weight_bytes_upper = sum_metric(base, "active_bytes_per_token", "upper_bound")
    base_flops_lower = sum_metric(base, "flops_per_token", "lower_bound")
    base_flops_nominal = sum_metric(base, "flops_per_token", "value")
    base_flops_upper = sum_metric(base, "flops_per_token", "upper_bound")
    recurrent_state = next((row for row in organ_graph if row.get("id") == "recurrent_state"), {})
    recurrent_metric = recurrent_state.get("active_bytes_per_token") if isinstance(recurrent_state, Mapping) else {}
    recurrent_rw = (recurrent_metric or {}).get("value") if isinstance(recurrent_metric, Mapping) else 0
    recurrent_flops = (recurrent_state.get("flops_per_token") or {}).get("value") if isinstance(recurrent_state, Mapping) and isinstance(recurrent_state.get("flops_per_token"), Mapping) else 0
    base_bytes_lower = (weight_bytes_lower or 0) + int(recurrent_rw or 0)
    base_bytes_nominal = (weight_bytes_nominal or 0) + int(recurrent_rw or 0)
    base_bytes_upper = (weight_bytes_upper or 0) + int(recurrent_rw or 0)
    base_flops_lower = (base_flops_lower or 0) + int(recurrent_flops or 0)
    base_flops_nominal = (base_flops_nominal or 0) + int(recurrent_flops or 0)
    base_flops_upper = (base_flops_upper or 0) + int(recurrent_flops or 0)
    mtp = next((row for row in organ_graph if row.get("id") == "mtp"), {})
    mtp_bytes = ((mtp.get("active_bytes_per_token") or {}).get("upper_bound") if isinstance(mtp.get("active_bytes_per_token"), Mapping) else None)
    mtp_flops = ((mtp.get("flops_per_token") or {}).get("upper_bound") if isinstance(mtp.get("flops_per_token"), Mapping) else None)
    qsa = next((row for row in organ_graph if row.get("id") == "sparse_attention"), {})
    qsa_metric = qsa.get("active_bytes_per_token") if isinstance(qsa, Mapping) else {}
    qsa_append = (qsa_metric or {}).get("components", {}).get("kv_cache_append_bytes_per_token") if isinstance(qsa_metric, Mapping) else None
    if qsa_append is None:
        qsa_append = 0
    layer_types = text.get("layer_types") if isinstance(text.get("layer_types"), list) else []
    full_layers = layer_types.count("full_attention")
    kv_heads = _as_int(text.get("num_key_value_heads")) or 0
    head_dim = _as_int(text.get("head_dim")) or 0
    dtype_bytes = _dtype_bytes(tensor_header_audit.get("tensor_headers") or {}) or 2
    kv_context_read = full_layers * 2 * kv_heads * head_dim * dtype_bytes
    routed = next((row for row in organ_graph if row.get("id") == "routed_experts"), {})
    routed_metric = routed.get("active_bytes_per_token") if isinstance(routed, Mapping) else {}
    routed_upper = (routed_metric or {}).get("upper_bound") if isinstance(routed_metric, Mapping) else None
    routed_lower = (routed_metric or {}).get("lower_bound") if isinstance(routed_metric, Mapping) else None
    routed_flops = routed.get("flops_per_token") if isinstance(routed, Mapping) else {}
    routed_flops_upper = (routed_flops or {}).get("upper_bound") if isinstance(routed_flops, Mapping) else None
    routed_flops_lower = (routed_flops or {}).get("lower_bound") if isinstance(routed_flops, Mapping) else None
    dynamic_context = {
        "qsa_kv_read_bytes_per_context_token": kv_context_read,
        "qsa_kv_read_formula": f"{full_layers} * 2 * {kv_heads} * {head_dim} * {dtype_bytes} * context_length",
        "qsa_attention_flops_formula": f"4 * {full_layers} * {text.get('num_attention_heads')} * {head_dim} * context_length",
    }
    bandwidth_gbps = _number(os.environ.get("HCLI_FLASH_BANDWIDTH_GBPS"))
    roof: Dict[str, Any] = {
        "status": "PARAMETERIZED_ONLY",
        "sustained_bandwidth_gbps": bandwidth_gbps,
        "formula": "sustained_bandwidth_bytes_per_second / bytes_per_token",
        "not_achievable_tps_claim": True,
    }
    if bandwidth_gbps and base_bytes_nominal:
        bytes_per_second = bandwidth_gbps * 1_000_000_000 / 8
        roof["nominal_tps"] = bytes_per_second / base_bytes_nominal
        roof["lower_bytes_tps_upper"] = bytes_per_second / max(1, base_bytes_upper or base_bytes_nominal)
        roof["upper_bytes_tps_lower"] = bytes_per_second / max(1, base_bytes_lower or base_bytes_nominal)
    return {
        "label": DERIVED,
        "status": "STRUCTURAL_ESTIMATE_NOT_PHYSICAL_TRACE",
        "layers": architecture.get("layers"),
        "linear_attention_layers": architecture.get("linear_attention_layers"),
        "full_attention_layers": architecture.get("full_attention_layers"),
        "routed_experts_per_token": architecture.get("experts_per_token"),
        "total_experts": architecture.get("experts"),
        "routed_expert_fraction": (_as_int(text.get("num_experts_per_tok")) or 0) / max(1, _as_int(text.get("num_experts")) or 1),
        "shared_expert_included": True,
        "active_bytes_per_token": _metric(base_bytes_nominal, status="STRUCTURAL_ESTIMATE", basis="text decode nominal bound below"),
        "device_bytes_touched_per_token": {"value": None, "label": LOCAL_PHYSICAL, "status": "NOT_MEASURED"},
        "flops_per_token": _metric(base_flops_nominal, status="STRUCTURAL_ESTIMATE", basis="text decode nominal bound below"),
        "state_bytes_per_token": _metric(recurrent_state.get("state_bytes") if isinstance(recurrent_state, Mapping) else None, status="STRUCTURAL_ESTIMATE"),
        "stored_parameters": {
            "value": architecture.get("index_total_size"),
            "label": UPSTREAM_SOURCE,
            "basis": "safetensors index metadata total_size",
            "parameter_count_assuming_declared_dtype": int(architecture.get("index_total_size") or 0) // max(1, dtype_bytes) if architecture.get("index_total_size") is not None else None,
        },
        "active_parameters_per_token": {
            "lower_bound": max(0, int(weight_bytes_lower or 0) - int(qsa_append or 0)) // max(1, dtype_bytes),
            "nominal": max(0, int(weight_bytes_nominal or 0) - int(qsa_append or 0)) // max(1, dtype_bytes),
            "upper_bound_without_mtp_or_vision": max(0, int(weight_bytes_upper or 0) - int(qsa_append or 0)) // max(1, dtype_bytes),
            "dtype_bytes": dtype_bytes,
            "expert_activation_fraction": (_as_int(text.get("num_experts_per_tok")) or 0) / max(1, _as_int(text.get("num_experts")) or 1),
        },
        "bytes_per_token": {
            "text_decode_lower_bound": base_bytes_lower,
            "text_decode_nominal": base_bytes_nominal,
            "text_decode_upper_bound_without_context_cache": base_bytes_upper,
            "mtp_conditional_extra_bytes": mtp_bytes,
            "dynamic_context": dynamic_context,
            "qsa_cache_append_bytes_per_token": qsa_append,
            "recurrent_state_read_write_bytes_per_token": recurrent_rw,
            "basis": "raw upstream tensor payload traffic plus explicit state/cache terms; excludes any unimplemented NF representation",
        },
        "flops_per_token": {
            "text_decode_lower_bound": base_flops_lower,
            "text_decode_nominal": base_flops_nominal,
            "text_decode_upper_bound_without_context_attention": base_flops_upper,
            "mtp_conditional_extra_flops": mtp_flops,
            "dynamic_context": dynamic_context,
            "routed_expert_lower_flops": routed_flops_lower,
            "routed_expert_upper_flops": routed_flops_upper,
            "basis": "two arithmetic operations per matrix value; no achievable-rate inference",
        },
        "expert_activation_fraction": {
            "routed": (_as_int(text.get("num_experts_per_tok")) or 0) / max(1, _as_int(text.get("num_experts")) or 1),
            "routed_experts_per_token": text.get("num_experts_per_tok"),
            "total_experts": text.get("num_experts"),
            "shared_expert_included": True,
        },
        "ngram_involvement": {
            "ngram_size": text.get("ngram_size"),
            "split_parts": text.get("split_ngram_parts"),
            "lookup_rows_assumed_nominal": max(1, _as_int(text.get("ngram_size")) or 1),
            "status": "STRUCTURAL_LOOKUP_ESTIMATE",
        },
        "mtp_overhead": {
            "status": "CONDITIONAL",
            "extra_bytes": mtp_bytes,
            "extra_flops": mtp_flops,
            "accepted_token_reduction_not_assumed": True,
        },
        "recurrent_state_movement": {
            "organ": "recurrent_state",
            "state_bytes": recurrent_state.get("state_bytes") if isinstance(recurrent_state, Mapping) else None,
            "flops_per_token": recurrent_flops,
        },
        "theoretical_bandwidth_roof": roof,
        "physical_trace": {"label": LOCAL_PHYSICAL, "status": "NOT_MEASURED", "achievable_tps": None, "device_bandwidth": None},
    }


def _metadata_summary(rows: Mapping[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
    result = []
    for name, _limit in METADATA_FILES:
        row = rows.get(name) or {"name": name, "label": UPSTREAM_SOURCE, "status": "MISSING"}
        result.append({key: value for key, value in row.items() if key != "data" and key != "parsed"})
    return result


def _write(report: Dict[str, Any], emit: Optional[str], repo: Path) -> None:
    destination = Path(emit).expanduser() if emit else repo / "receipts" / "headless" / "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    if not destination.is_absolute():
        destination = repo / destination
    report["receipt_path"] = str(destination.resolve())
    atomic_write_json(destination, report)


def run_flash_science_gate(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Fetch pinned metadata and emit an identity-only pre-runtime science receipt."""
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "qualification": "FLASH_NEXT_PRE_RUNTIME_IDENTITY_AND_PLAN_ONLY",
        "started_at": started,
        "source_identity": {
            "repo": REPO_ID,
            "page": MODEL_PAGE,
            "pinned_revision": PINNED_REVISION,
            "expected_file_count": EXPECTED_FILE_COUNT,
            "expected_complete_source_bytes": EXPECTED_BYTES,
            "labels": {"upstream": UPSTREAM_SOURCE, "derived": DERIVED, "local_physical": LOCAL_PHYSICAL},
        },
        "promotion_law": {
            "complete_system_ebpw_max": COMPLETE_SYSTEM_EBPW_MAX,
            "accepted_capability_preserving_tps_min": ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN,
            "dense_parent_execution_fallback": False,
            "hidden_dense_rematerialization": False,
            "text_resident_exception": "FLASH_NEXT_TEXT_RESIDENT may omit vision tensors only when explicitly declared and qualified",
        },
    }
    fetched: Dict[str, Dict[str, Any]] = {}
    try:
        for name, limit in METADATA_FILES:
            fetched[name] = _fetch(name, limit, timeout_s)
        config_row = fetched.get("config.json") or {}
        index_row = fetched.get("model.safetensors.index.json") or {}
        config = config_row.get("parsed") if isinstance(config_row.get("parsed"), Mapping) else {}
        index = index_row.get("parsed") if isinstance(index_row.get("parsed"), Mapping) else {}
        weight_map = index.get("weight_map") if isinstance(index.get("weight_map"), Mapping) else {}
        text_config = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
        remote_sizes: Dict[str, int] = {}
        census_path = repo / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_CENSUS.json"
        census = None
        try:
            census = json.loads(census_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            census = None
        if isinstance(census, Mapping) and isinstance(census.get("remote_manifest"), Mapping):
            for item in census["remote_manifest"].get("files") or []:
                if isinstance(item, Mapping) and item.get("file"):
                    remote_sizes[str(item["file"])] = int(item.get("size") or 0)
        shard_names = sorted({str(shard) for shard in weight_map.values() if shard})
        tensor_header_audit = _fetch_tensor_headers(shard_names, timeout_s) if shard_names else {
            "status": "NOT_RUN",
            "requested_shards": 0,
            "parsed_shards": 0,
            "failed_shards": [],
            "header_bytes": 0,
            "payload_bytes": 0,
            "duplicate_tensor_names": [],
            "per_shard": [],
            "tensor_headers": {},
        }
        expected_tensor_names = {str(name) for name in weight_map}
        observed_tensor_names = set(tensor_header_audit.get("tensor_headers") or {})
        index_total = index.get("metadata", {}).get("total_size") if isinstance(index.get("metadata"), Mapping) else None
        tensor_header_audit.update({
            "index_tensor_count": len(expected_tensor_names),
            "header_tensor_count": len(observed_tensor_names),
            "missing_tensor_count": len(expected_tensor_names - observed_tensor_names),
            "missing_tensor_examples": sorted(expected_tensor_names - observed_tensor_names)[:10],
            "unexpected_tensor_count": len(observed_tensor_names - expected_tensor_names),
            "payload_matches_index": index_total is not None and int(tensor_header_audit.get("payload_bytes") or 0) == int(index_total),
            "complete": (
                tensor_header_audit.get("status") == "PARSED"
                and not tensor_header_audit.get("duplicate_tensor_names")
                and expected_tensor_names == observed_tensor_names
                and index_total is not None
                and int(tensor_header_audit.get("payload_bytes") or 0) == int(index_total)
            ),
            "body_bytes_requested": 0,
            "body_bytes_loaded": 0,
            "claim_boundary": "Only safetensors headers/data_offsets were read; tensor bodies were not requested or loaded.",
        })
        tensor_header_summary = {key: value for key, value in tensor_header_audit.items() if key != "tensor_headers"}
        layer_types = text_config.get("layer_types") if isinstance(text_config.get("layer_types"), list) else []
        architecture = {
            "model_type": config.get("model_type"),
            "text_model_type": text_config.get("model_type"),
            "architectures": config.get("architectures"),
            "hidden_size": text_config.get("hidden_size"),
            "layers": text_config.get("num_hidden_layers"),
            "layer_types": layer_types,
            "full_attention_layers": layer_types.count("full_attention"),
            "linear_attention_layers": layer_types.count("linear_attention"),
            "experts": text_config.get("num_experts"),
            "experts_per_token": text_config.get("num_experts_per_tok"),
            "vocab_size": text_config.get("vocab_size"),
            "ngram_vocab_size_base": text_config.get("ngram_vocab_size_base"),
            "ngram_size": text_config.get("ngram_size"),
            "split_ngram_parts": text_config.get("split_ngram_parts"),
            "indexer_budget": text_config.get("indexer_budget"),
            "mtp": text_config.get("mtp"),
            "vision_config": config.get("vision_config"),
            "observed_index_metadata": index.get("metadata"),
        }
        architecture_fingerprint = _canonical_hash({"architecture": architecture, "tensor_names": sorted(str(name) for name in weight_map), "metadata_sha256": {name: (fetched.get(name) or {}).get("sha256") for name, _ in METADATA_FILES}})
        organ_graph = _organ_graph(config, weight_map, remote_sizes, tensor_header_audit.get("tensor_headers") or {})
        gravity_science = _gravity_science_plan()
        zero_questions = _three_zero_questions()
        accelerator_primitive_plan = _accelerator_primitive_plan()
        report.update({
            "metadata": _metadata_summary(fetched),
            "architecture_fingerprint": {"value": architecture_fingerprint, "algorithm": "sha256(canonical pinned metadata + sorted tensor names)", "label": DERIVED},
            "architecture": {**architecture, "label": DERIVED, "tensor_count": len(weight_map), "index_total_size": index_total},
            "safetensors_header_audit": tensor_header_summary,
            "organ_graph": organ_graph,
            "active_compute_bounds": _active_compute_bounds(
                {**architecture, "index_total_size": index_total},
                organ_graph,
                text_config,
                tensor_header_audit,
            ),
            "gravity_plan": [
                {"organ": "moe_experts", "hypothesis": "cross-expert shared basis plus residual", "native_kernel": "representation-native routed expert decode", "status": "PLAN_ONLY"},
                {"organ": "ngram_embedding", "hypothesis": "factorized/generative lookup rather than generic matrix quantization", "native_kernel": "ngram lookup/generator", "status": "PLAN_ONLY"},
                {"organ": "deltanet", "hypothesis": "persistent resident state-machine execution", "native_kernel": "linear-attention state update", "status": "PLAN_ONLY"},
                {"organ": "qsa_sparse_attention", "hypothesis": "indexer-budget sparse block traversal", "native_kernel": "sparse attention selection and gather", "status": "PLAN_ONLY"},
                {"organ": "mtp", "hypothesis": "accepted drafts reduce effective decode steps only when accepted work is counted", "native_kernel": "MTP accept/reject path", "status": "PLAN_ONLY"},
            ],
            "gravity_science": gravity_science,
            "gravity_ladder": gravity_science["stages"],
            "three_zero_questions": zero_questions,
            "accelerator_primitive_plan": accelerator_primitive_plan,
            "required_primitives": [
                "dense-vs-NF reference comparator",
                "shared-basis/residual decoder",
                "sparse expert router and index lookup",
                "ngram factorized lookup/generator",
                "linear recurrent-state update",
                "QSA sparse traversal",
                "MTP acceptance accounting",
                "complete-token wall meter",
            ],
            "local_physical": {
                "machine": platform.platform(),
                "architecture": platform.machine(),
                "model_lake_target_present": (repo / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_CENSUS.json").is_file() and bool((census or {}).get("flash_target_manifest", {}).get("final_present")),
                "weights_loaded": False,
                "native_executable": False,
                "native_kernel": "NOT_COMPILED",
                "label": LOCAL_PHYSICAL,
            },
            "promotion_gate": evaluate_flash_promotion(None),
            "checks": {
                "all_metadata_fetched": all((fetched.get(name) or {}).get("status") == "FETCHED" for name, _ in METADATA_FILES),
                "config_parsed": isinstance(config, Mapping) and bool(config),
                "index_parsed": isinstance(index, Mapping) and bool(weight_map),
                "pinned_metadata_urls": all((fetched.get(name) or {}).get("revision") == PINNED_REVISION for name, _ in METADATA_FILES if (fetched.get(name) or {}).get("status") == "FETCHED"),
                "no_weights_downloaded": True,
                "safetensors_headers_complete": bool(tensor_header_audit.get("complete")),
                "header_payload_matches_index": bool(tensor_header_audit.get("payload_matches_index")),
                "weight_bodies_not_requested": tensor_header_audit.get("body_bytes_requested") == 0,
                "promotion_not_allowed": evaluate_flash_promotion(None).get("promotion_allowed") is False,
            },
        })
        report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001 - persist the metadata boundary
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        report["metadata"] = _metadata_summary(fetched)
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    _write(report, str(emit) if emit is not None else None, repo)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--emit")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    report = run_flash_science_gate(repo_root=args.repo_root, emit=args.emit, timeout_s=args.timeout_s)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GRAVITY_LADDER",
    "SCHEMA",
    "run_flash_science_gate",
    "_accelerator_primitive_plan",
    "_gravity_science_plan",
    "_three_zero_questions",
]
