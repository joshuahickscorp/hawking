#!/usr/bin/env python3.12
"""Fail-closed bounded architecture admission for DeepSeek-V4-Flash.

This module deliberately consumes only small, local inputs: the pinned
``config.json``, ``tokenizer_config.json``, an optional source-owned chat
protocol artifact, and a safetensors *header-only* capture.  It never fetches
a model, opens a tensor body, decodes a weight, or starts an oracle/runtime.
Its job is narrower: make the architecture fixture requirements and the
source bindings machine-readable before a large source transfer is justified.

An all-green structural fixture is still ``NOT_ADMITTED`` unless a separately
sealed source-exact envelope binds every required fixture result to the exact
repo, revision, and three observed file hashes.  Even an
``ARCHITECTURE_ADMITTED`` result proves only that bounded source evidence
covers the listed architecture behaviours.  It is never evidence of a codec,
source download, complete artifact, CPU oracle, Metal forward, capability, or
throughput result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


SCHEMA = "hawking.gravity.architecture_admission.v1"
SOURCE_ENVELOPE_SCHEMA = "hawking.gravity.source_exact_fixture_envelope.v1"
SOURCE_AUTHORITY_SCHEMA = "hawking.gravity.official_source_authority.v1"
SEAL_KEY = "seal_sha256"
EXPECTED_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash"
MAX_HEADER_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
FIXTURE_LAYER = 4
FIXTURE_SHARD_FILENAME = "model-00006-of-00046.safetensors"
FIXTURE_SHARD_LFS_SHA256 = "51a65e6d9d0ccb70013e25ae70a50b177af8f97e59ac798c2d0ed5ebb169fe7a"
FIXTURE_SHARD_FULL_SIZE_BYTES = 3_590_024_776

# These are evidence behaviours, not implementation capabilities.  A PASS
# must come from a source-exact fixture result supplied by an independent
# future source/oracle lane; this tool only verifies the receipt binding.
REQUIRED_BEHAVIORS: tuple[str, ...] = (
    "native_fp4_expert_decode",
    "native_fp8_control_decode",
    "router_256_top6_shared_expert",
    "mhc_state_transition",
    "compressed_indexed_attention",
    "tokenizer_template_coverage",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_BLOB_ID_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_EXPERT_WEIGHT_RE = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$"
)
_SHARED_WEIGHT_RE = re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.(w[123])\.weight$")


class AdmissionInputError(ValueError):
    """Raised internally for invalid bounded fixture inputs."""


AuthorityVerifier = Callable[[Mapping[str, Any]], Mapping[str, Any]]

# Deliberately unconfigured.  A future private control-plane integration must
# register a signature/key-backed verifier outside this public fixture module.
# The standalone Python API and CLI therefore fail closed today.
_OFFICIAL_SOURCE_AUTHORITY_VERIFIER: AuthorityVerifier | None = None


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical(value)
    return hashlib.sha256(payload).hexdigest()


def seal_document(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic local integrity seal for a source envelope.

    This guards accidental/tampered substitutions.  It is not a remote trust
    authority and does not make untrusted bytes official-source evidence.
    """

    unsigned = {key: item for key, item in value.items() if key != SEAL_KEY}
    return {**unsigned, SEAL_KEY: _sha256(unsigned)}


def _verify_seal(value: Mapping[str, Any]) -> str | None:
    recorded = value.get(SEAL_KEY)
    if not isinstance(recorded, str) or not _SHA256_RE.fullmatch(recorded):
        return "source envelope is missing a valid seal_sha256"
    expected = seal_document(value)[SEAL_KEY]
    if recorded != expected:
        return "source envelope seal_sha256 does not match its contents"
    return None


def _read_json(path: Path, label: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            return None, f"{label} exceeds the {MAX_METADATA_BYTES}-byte bounded metadata limit"
    except OSError as exc:
        return None, f"cannot stat {label}: {exc}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = handle.read(MAX_METADATA_BYTES + 1)
    except OSError as exc:
        return None, f"cannot read {label}: {exc}"
    if len(raw.encode("utf-8")) > MAX_METADATA_BYTES:
        return None, f"{label} exceeds the {MAX_METADATA_BYTES}-byte bounded metadata limit"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {label}: {exc}"
    if not isinstance(value, dict):
        return None, f"{label} must contain a JSON object"
    return value, None


def _file_sha256(path: Path, *, max_bytes: int = MAX_METADATA_BYTES) -> tuple[str | None, str | None]:
    try:
        if path.stat().st_size > max_bytes:
            return None, f"cannot hash {path}: exceeds bounded input limit {max_bytes}"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest(), None
    except OSError as exc:
        return None, f"cannot hash {path}: {exc}"


def _nonempty_file_sha256(path: Path, label: str) -> tuple[str | None, str | None]:
    """Hash one bounded source-owned protocol asset without executing it."""

    try:
        if path.stat().st_size <= 0:
            return None, f"{label} must not be empty"
    except OSError as exc:
        return None, f"cannot stat {label}: {exc}"
    return _file_sha256(path)


def read_header_only(path: Path) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    """Read exactly one safetensors header capture, never a tensor payload.

    The supplied file must contain the standard 8-byte header length followed
    by exactly that many JSON bytes.  A full safetensors shard, a body range,
    or an overlarge header is rejected instead of being partially inspected.
    """

    try:
        capture_bytes = path.stat().st_size
    except OSError as exc:
        return None, f"cannot stat safetensors header: {exc}"
    if capture_bytes < 8:
        return None, "safetensors header capture is shorter than its length prefix"
    # Reject a full source shard before opening it.  The largest permitted
    # input is exactly a small header capture, never a tensor body.
    if capture_bytes > 8 + MAX_HEADER_BYTES:
        return None, "header capture exceeds the bounded header-only limit"
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                return None, "safetensors header capture is shorter than its length prefix"
            header_bytes = struct.unpack("<Q", prefix)[0]
            if header_bytes <= 0 or header_bytes > MAX_HEADER_BYTES:
                return None, f"safetensors header length is outside bounded range: {header_bytes}"
            if capture_bytes != 8 + header_bytes:
                return None, "header capture must contain exactly prefix + JSON header, with no tensor body"
            raw_header = handle.read(header_bytes)
            if len(raw_header) != header_bytes:
                return None, "short safetensors header capture"
    except OSError as exc:
        return None, f"cannot read safetensors header: {exc}"
    try:
        decoded = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"invalid safetensors header JSON: {exc}"
    if not isinstance(decoded, dict):
        return None, "safetensors header JSON must be an object"
    tensors: dict[str, dict[str, Any]] = {}
    for name, row in decoded.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(row, dict):
            return None, "safetensors header contains a non-object tensor descriptor"
        dtype = row.get("dtype")
        shape = row.get("shape")
        offsets = row.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            return None, f"safetensors descriptor {name!r} is missing dtype or shape"
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(item, int) or item < 0 for item in offsets)
            or offsets[0] > offsets[1]
        ):
            return None, f"safetensors descriptor {name!r} has invalid data_offsets"
        tensors[name] = row
    if not tensors:
        return None, "safetensors header contains no tensor descriptors"
    return tensors, None


def _truthy_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _dtype_starts_with(row: Mapping[str, Any] | None, prefix: str) -> bool:
    return isinstance(row, Mapping) and str(row.get("dtype", "")).upper().startswith(prefix)


def _pair_with_scale(
    tensors: Mapping[str, Mapping[str, Any]], regex: re.Pattern[str], *, layer: int = FIXTURE_LAYER
) -> tuple[str | None, str | None]:
    for name in sorted(tensors):
        match = regex.fullmatch(name)
        if match is None or match.group(1) != str(layer):
            continue
        scale = f"{name[:-len('.weight')]}.scale"
        if _dtype_starts_with(tensors.get(name), "I8") and _dtype_starts_with(
            tensors.get(scale), "F8_E8M0"
        ):
            return name, scale
    return None, None


def _fp8_pair(
    tensors: Mapping[str, Mapping[str, Any]], name: str
) -> tuple[str | None, str | None]:
    """Require one observed e4m3 weight / ue8m0 scale pair."""

    scale = f"{name[:-len('.weight')]}.scale"
    if _dtype_starts_with(tensors.get(name), "F8_E4M3") and _dtype_starts_with(
        tensors.get(scale), "F8_E8M0"
    ):
        return name, scale
    return None, None


def _check(status: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": "STRUCTURAL_PASS" if status else "BLOCKED",
        "detail": detail,
        **extra,
    }


def _architecture_checks(
    config: Mapping[str, Any] | None,
    tokenizer: Mapping[str, Any] | None,
    tensors: Mapping[str, Mapping[str, Any]] | None,
    chat_protocol_sha256: str | None,
) -> dict[str, dict[str, Any]]:
    """Check observable architecture shape only; never execute an operator."""

    config = config or {}
    tokenizer = tokenizer or {}
    tensors = tensors or {}
    architectures = config.get("architectures")
    model_identity_ok = (
        config.get("model_type") == "deepseek_v4"
        and isinstance(architectures, list)
        and "DeepseekV4ForCausalLM" in architectures
    )
    quantization = config.get("quantization_config")
    quantization = quantization if isinstance(quantization, Mapping) else {}
    expert_name, expert_scale = _pair_with_scale(tensors, _EXPERT_WEIGHT_RE)
    shared_name, shared_scale = _fp8_pair(
        tensors, f"layers.{FIXTURE_LAYER}.ffn.shared_experts.w1.weight"
    )
    fp8_control_name, fp8_control_scale = _fp8_pair(
        tensors, f"layers.{FIXTURE_LAYER}.attn.indexer.wq_b.weight"
    )
    router_name = next(
        (
            name
            for name in sorted(tensors)
            if re.fullmatch(rf"layers\.{FIXTURE_LAYER}\.ffn\.gate\.weight", name)
            and _dtype_starts_with(tensors.get(name), "BF16")
        ),
        None,
    )

    fp4_ok = str(config.get("expert_dtype", "")).casefold() == "fp4" and expert_name is not None
    fp8_ok = (
        str(quantization.get("quant_method", "")).casefold() == "fp8"
        and str(quantization.get("fmt", "")).casefold() == "e4m3"
        and str(quantization.get("scale_fmt", "")).casefold() == "ue8m0"
        and fp8_control_name is not None
    )
    router_ok = (
        config.get("n_routed_experts") == 256
        and config.get("num_experts_per_tok") == 6
        and config.get("n_shared_experts") == 1
        and config.get("scoring_func") == "sqrtsoftplus"
        and config.get("topk_method") == "noaux_tc"
        and router_name is not None
        and shared_name is not None
    )
    hc_tensor_names = (
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    )
    hc_seen = {
        suffix
        for suffix in hc_tensor_names
        if any(re.fullmatch(rf"layers\.{FIXTURE_LAYER}\.{re.escape(suffix)}", name) for name in tensors)
    }
    mhc_ok = (
        config.get("hc_mult") == 4
        and config.get("hc_sinkhorn_iters") == 20
        and _truthy_number(config.get("hc_eps"))
        and hc_seen == set(hc_tensor_names)
    )
    ratios = config.get("compress_ratios")
    ratio_ok = isinstance(ratios, list) and bool(ratios) and any(
        isinstance(item, int) and item > 0 for item in ratios
    )
    compressed_names = any(
        name.startswith(f"layers.{FIXTURE_LAYER}.attn.compressor.") for name in tensors
    )
    indexed_names = any(
        name.startswith(f"layers.{FIXTURE_LAYER}.attn.indexer.") for name in tensors
    )
    attention_ok = (
        all(
            _truthy_number(config.get(field))
            for field in (
                "q_lora_rank",
                "o_lora_rank",
                "qk_rope_head_dim",
                "compress_rope_theta",
                "index_topk",
                "index_n_heads",
                "index_head_dim",
                "num_hash_layers",
            )
        )
        and ratio_ok
        and compressed_names
        and indexed_names
    )
    tokenizer_class = tokenizer.get("tokenizer_class")
    bos = tokenizer.get("bos_token")
    eos = tokenizer.get("eos_token")
    token_values_ok = all(
        isinstance(value, (str, Mapping)) and bool(value)
        for value in (tokenizer_class, bos, eos)
    )
    # V4-Flash's official tokenizer config declares ``chat_template: null``;
    # its source-owned encoding protocol plus vectors is the authoritative
    # template surface.  A conventional nonempty Jinja template is accepted
    # for future compatible sources, but never invented here.
    template = tokenizer.get("chat_template")
    jinja_template_present = isinstance(template, str) and bool(template.strip())
    protocol_present = isinstance(chat_protocol_sha256, str) and _SHA256_RE.fullmatch(chat_protocol_sha256) is not None
    template_state = (
        "jinja_template"
        if jinja_template_present
        else "source_owned_protocol"
        if protocol_present
        else "not_observed"
    )
    tokenizer_ok = token_values_ok and (jinja_template_present or protocol_present)

    return {
        "model_identity": _check(
            model_identity_ok,
            "requires the declared DeepseekV4ForCausalLM/deepseek_v4 configuration identity",
            model_type=config.get("model_type"),
            architectures=architectures,
            execution="not_executed",
        ),
        "native_fp4_expert_decode": _check(
            fp4_ok,
            "requires source metadata fp4 plus one packed I8 expert/ue8m0-scale pair",
            observed_expert_weight=expert_name,
            observed_expert_scale=expert_scale,
            execution="not_executed",
        ),
        "native_fp8_control_decode": _check(
            fp8_ok,
            "requires source fp8/e4m3/ue8m0 metadata plus one F8_E4M3 indexed-attention control pair",
            observed_fp8_control_weight=fp8_control_name,
            observed_fp8_control_scale=fp8_control_scale,
            execution="not_executed",
        ),
        "router_256_top6_shared_expert": _check(
            router_ok,
            "requires a BF16 256-way learned router, top-6 selection, and one FP8 shared-expert pair",
            n_routed_experts=config.get("n_routed_experts"),
            num_experts_per_tok=config.get("num_experts_per_tok"),
            n_shared_experts=config.get("n_shared_experts"),
            scoring_func=config.get("scoring_func"),
            topk_method=config.get("topk_method"),
            fixture_layer=FIXTURE_LAYER,
            observed_shared_weight=shared_name,
            observed_shared_scale=shared_scale,
            observed_router_weight=router_name,
            execution="not_executed",
        ),
        "mhc_state_transition": _check(
            mhc_ok,
            "requires observed hc field family and all base/function/scale transition tensors",
            hc_mult=config.get("hc_mult"),
            hc_sinkhorn_iters=config.get("hc_sinkhorn_iters"),
            hc_eps=config.get("hc_eps"),
            observed_hc_tensors=sorted(hc_seen),
            execution="not_executed",
        ),
        "compressed_indexed_attention": _check(
            attention_ok,
            "requires compressed-attention and sparse-index configuration plus both tensor families",
            compressed_tensor_family_seen=compressed_names,
            indexed_tensor_family_seen=indexed_names,
            compress_ratios_count=len(ratios) if isinstance(ratios, list) else None,
            index_topk=config.get("index_topk"),
            execution="not_executed",
        ),
        "tokenizer_template_coverage": _check(
            tokenizer_ok,
            "requires source-bound tokenizer class, BOS/EOS, and either a non-empty Jinja template or a source-owned encoding protocol asset",
            tokenizer_class=tokenizer_class,
            chat_template_state=template_state,
            chat_protocol_sha256=chat_protocol_sha256,
            execution="not_executed",
        ),
    }


def _hash_binding_errors(
    source: Mapping[str, Any],
    *,
    config_sha256: str | None,
    tokenizer_sha256: str | None,
    chat_protocol_sha256: str | None,
    header_sha256: str | None,
) -> list[str]:
    errors: list[str] = []
    if source.get("repo") != EXPECTED_REPOSITORY:
        errors.append(f"source repo must be {EXPECTED_REPOSITORY!r}")
    revision = source.get("revision")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        errors.append("source revision must be a 40-character immutable lowercase SHA")
    else:
        expected_tree = f"https://huggingface.co/{EXPECTED_REPOSITORY}/tree/{revision}"
        if source.get("immutable_tree_url") != expected_tree:
            errors.append("immutable_tree_url must exactly bind the declared repo and revision")
    expected_hashes = {
        "config_sha256": config_sha256,
        "tokenizer_config_sha256": tokenizer_sha256,
        "chat_protocol_sha256": chat_protocol_sha256,
        "safetensors_header_sha256": header_sha256,
    }
    for key, actual in expected_hashes.items():
        declared = source.get(key)
        if actual is None:
            if declared is not None:
                errors.append(f"source {key} is declared without an observed local input")
            continue
        if not isinstance(declared, str) or _SHA256_RE.fullmatch(declared) is None:
            errors.append(f"source {key} must be a SHA-256")
        elif actual is None or declared != actual:
            errors.append(f"source {key} does not bind the observed local input")
    return errors


def _result_binding_errors(
    result: Any,
    source: Mapping[str, Any],
    behavior: str,
) -> list[str]:
    if not isinstance(result, Mapping):
        return [f"{behavior}: missing fixture result"]
    errors: list[str] = []
    if result.get("status") != "PASS":
        errors.append(f"{behavior}: source-exact fixture result must be PASS")
    for key in (
        "fixture_receipt_sha256",
        "test_vector_sha256",
        "implementation_sha256",
        "output_sha256",
    ):
        value = result.get(key)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            errors.append(f"{behavior}: {key} must be a SHA-256")
    binding = result.get("source_binding")
    if not isinstance(binding, Mapping):
        return [*errors, f"{behavior}: source_binding is required"]
    for key in (
        "repo",
        "revision",
        "config_sha256",
        "tokenizer_config_sha256",
        "chat_protocol_sha256",
        "safetensors_header_sha256",
    ):
        if binding.get(key) != source.get(key):
            errors.append(f"{behavior}: source_binding.{key} does not match envelope source")
    return errors


def _authority_verifier_gate(
    authority: Mapping[str, Any], *, fixture_attestations_sha256: str | None
) -> dict[str, Any]:
    """Require an external authority verifier; a local seal is never enough.

    This repository intentionally ships no trusted control-plane verifier or
    public-key root for DeepSeek-V4-Flash.  Consequently the standalone CLI
    cannot promote a self-sealed document.  A future control-plane integration
    may inject a verifier, but its result must identify its own durable
    verifier receipt rather than returning a bare boolean.
    """

    verifier = _OFFICIAL_SOURCE_AUTHORITY_VERIFIER
    if verifier is None:
        return {
            "status": "BLOCKED",
            "reason": "INDEPENDENT_AUTHORITY_VERIFIER_UNAVAILABLE",
            "detail": "no trusted official-source authority verifier is configured for this offline harness",
        }
    try:
        result = verifier(authority)
    except Exception as exc:  # pragma: no cover - defensive boundary around external verifier
        return {
            "status": "BLOCKED",
            "reason": "INDEPENDENT_AUTHORITY_VERIFIER_FAILED",
            "detail": f"independent authority verifier raised {type(exc).__name__}",
        }
    if not isinstance(result, Mapping):
        return {
            "status": "BLOCKED",
            "reason": "INDEPENDENT_AUTHORITY_VERIFIER_INVALID",
            "detail": "independent authority verifier must return a structured result",
        }
    verifier_id = result.get("verifier_id")
    verifier_receipt_sha256 = result.get("verifier_receipt_sha256")
    scope = result.get("scope")
    authority_nonce = authority.get("campaign_nonce_sha256")
    if (
        result.get("status") != "PASS"
        or not isinstance(verifier_id, str)
        or not verifier_id.strip()
        or not isinstance(verifier_receipt_sha256, str)
        or _SHA256_RE.fullmatch(verifier_receipt_sha256) is None
        or scope != "official_source_authority"
        or result.get("fixture_attestations_sha256") != fixture_attestations_sha256
        or result.get("campaign_nonce_sha256") != authority_nonce
    ):
        return {
            "status": "BLOCKED",
            "reason": "INDEPENDENT_AUTHORITY_VERIFIER_INVALID",
            "detail": "verifier result must PASS and attest the exact fixture receipt/test-vector/implementation/output digest set plus campaign nonce",
        }
    return {
        "status": "PASS",
        "reason": "INDEPENDENT_AUTHORITY_VERIFIED",
        "verifier_id": verifier_id,
        "verifier_receipt_sha256": verifier_receipt_sha256,
        "scope": scope,
        "fixture_attestations_sha256": fixture_attestations_sha256,
        "campaign_nonce_sha256": authority_nonce,
    }


def _official_source_authority_gate(
    authority: Mapping[str, Any] | None,
    *,
    authority_sha256: str | None,
    envelope_authority_sha256: Any,
    fixture_results: Mapping[str, Any],
    config_sha256: str | None,
    tokenizer_sha256: str | None,
    chat_protocol_sha256: str | None,
    header_sha256: str | None,
    header_capture_bytes: int | None,
) -> dict[str, Any]:
    """Validate a separate control-plane authority record and its bindings."""

    if authority is None:
        return {
            "status": "BLOCKED",
            "reason": "MISSING_OFFICIAL_SOURCE_AUTHORITY",
            "detail": "a separate independently verified official-source authority record is required",
            "errors": [],
        }
    errors: list[str] = []
    if authority.get("schema") != SOURCE_AUTHORITY_SCHEMA:
        errors.append(f"authority schema must be {SOURCE_AUTHORITY_SCHEMA}")
    if authority.get("status") != "PASS":
        errors.append("authority status must be PASS")
    seal_error = _verify_seal(authority)
    if seal_error is not None:
        errors.append(f"authority {seal_error}")
    if (
        not isinstance(envelope_authority_sha256, str)
        or _SHA256_RE.fullmatch(envelope_authority_sha256) is None
        or authority_sha256 is None
        or envelope_authority_sha256 != authority_sha256
    ):
        errors.append("source envelope must bind the exact external authority file SHA-256")

    source = authority.get("source")
    if not isinstance(source, Mapping):
        errors.append("authority source object is required")
        source = {}
    errors.extend(
        _hash_binding_errors(
            source,
            config_sha256=config_sha256,
            tokenizer_sha256=tokenizer_sha256,
            chat_protocol_sha256=chat_protocol_sha256,
            header_sha256=header_sha256,
        )
    )
    blobs = authority.get("verified_blobs")
    if not isinstance(blobs, Mapping):
        errors.append("authority verified_blobs object is required")
        blobs = {}
    for label, observed_sha in (
        ("config", config_sha256),
        ("tokenizer_config", tokenizer_sha256),
        ("chat_protocol", chat_protocol_sha256),
    ):
        if observed_sha is None:
            continue
        blob = blobs.get(label)
        if not isinstance(blob, Mapping):
            errors.append(f"authority verified_blobs.{label} is required")
            continue
        if blob.get("sha256") != observed_sha:
            errors.append(f"authority verified_blobs.{label}.sha256 does not bind observed input")
        blob_id = blob.get("blob_id")
        if not isinstance(blob_id, str) or _BLOB_ID_RE.fullmatch(blob_id) is None:
            errors.append(f"authority verified_blobs.{label}.blob_id must be a pinned 40/64-hex id")

    shard = authority.get("owning_shard")
    if not isinstance(shard, Mapping):
        errors.append("authority owning_shard object is required")
        shard = {}
    filename = shard.get("filename")
    if filename != FIXTURE_SHARD_FILENAME:
        errors.append(f"authority owning_shard.filename must be {FIXTURE_SHARD_FILENAME}")
    lfs_sha = shard.get("lfs_sha256")
    if lfs_sha != FIXTURE_SHARD_LFS_SHA256:
        errors.append("authority owning_shard.lfs_sha256 does not match the selected Base Flash layer-4 shard")
    if shard.get("full_size_bytes") != FIXTURE_SHARD_FULL_SIZE_BYTES:
        errors.append(
            f"authority owning_shard.full_size_bytes must be {FIXTURE_SHARD_FULL_SIZE_BYTES}"
        )
    if shard.get("header_capture_sha256") != header_sha256:
        errors.append("authority owning_shard.header_capture_sha256 does not bind observed header")
    header_range = shard.get("header_range")
    if (
        not isinstance(header_range, Mapping)
        or header_range.get("length_prefix_inclusive") != [0, 7]
        or not isinstance(header_range.get("json_inclusive"), list)
        or len(header_range.get("json_inclusive", [])) != 2
        or header_capture_bytes is None
        or header_range.get("json_inclusive") != [8, header_capture_bytes - 1]
    ):
        errors.append(
            "authority owning_shard.header_range must record [0,7] length prefix plus the dynamic JSON header range"
        )

    authority_info = authority.get("authority")
    if not isinstance(authority_info, Mapping):
        errors.append("authority authority object is required")
    else:
        if authority_info.get("kind") != "official_manifest_control_plane":
            errors.append("authority kind must be official_manifest_control_plane")
        if authority_info.get("independently_verified") is not True:
            errors.append("authority must explicitly state independently_verified=true")
        receipt_sha = authority_info.get("verification_receipt_sha256")
        if not isinstance(receipt_sha, str) or _SHA256_RE.fullmatch(receipt_sha) is None:
            errors.append("authority verification_receipt_sha256 must be a SHA-256")

    freshness = authority.get("freshness")
    if not isinstance(freshness, Mapping):
        errors.append("authority freshness object with verified_at/expires_at is required")
    else:
        verified_at = freshness.get("verified_at")
        expires_at = freshness.get("expires_at")
        try:
            verified_time = datetime.fromisoformat(str(verified_at).replace("Z", "+00:00"))
            expires_time = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
            if (
                verified_time.tzinfo is None
                or expires_time.tzinfo is None
                or expires_time <= verified_time
            ):
                errors.append("authority freshness expires_at must be after verified_at")
            elif expires_time <= datetime.now(timezone.utc):
                errors.append("authority freshness expires_at is stale; a historical authority cannot admit new source evidence")
        except ValueError:
            errors.append("authority freshness verified_at/expires_at must be ISO-8601 timestamps")
    nonce = authority.get("campaign_nonce_sha256")
    if not isinstance(nonce, str) or _SHA256_RE.fullmatch(nonce) is None:
        errors.append("authority campaign_nonce_sha256 must be a SHA-256")

    attestations = authority.get("fixture_attestations")
    if not isinstance(attestations, Mapping):
        errors.append("authority fixture_attestations object is required")
        attestations = {}
    for behavior in REQUIRED_BEHAVIORS:
        attestation = attestations.get(behavior)
        result = fixture_results.get(behavior)
        if not isinstance(attestation, Mapping):
            errors.append(f"authority fixture_attestations.{behavior} is required")
            continue
        if not isinstance(result, Mapping):
            errors.append(f"envelope fixture_results.{behavior} is required for authority comparison")
            continue
        for key in (
            "fixture_receipt_sha256",
            "test_vector_sha256",
            "implementation_sha256",
            "output_sha256",
        ):
            value = attestation.get(key)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                errors.append(f"authority fixture_attestations.{behavior}.{key} must be a SHA-256")
            elif value != result.get(key):
                errors.append(f"authority fixture_attestations.{behavior}.{key} does not bind envelope result")

    verifier_gate = _authority_verifier_gate(
        authority, fixture_attestations_sha256=_sha256(attestations)
    ) if not errors else {
        "status": "BLOCKED",
        "reason": "OFFICIAL_SOURCE_AUTHORITY_INVALID",
        "detail": "authority record did not satisfy its structural contract",
    }
    if verifier_gate["status"] != "PASS" and not errors:
        errors.append(str(verifier_gate["reason"]))
    return {
        "status": "PASS" if not errors and verifier_gate["status"] == "PASS" else "BLOCKED",
        "reason": "OFFICIAL_SOURCE_AUTHORITY_BOUND" if not errors and verifier_gate["status"] == "PASS" else "OFFICIAL_SOURCE_AUTHORITY_INVALID",
        "detail": "separate official-source authority binds manifest blobs, owning shard LFS identity, and header capture" if not errors and verifier_gate["status"] == "PASS" else "official-source authority did not satisfy the complete independent binding contract",
        "errors": errors,
        "verifier": verifier_gate,
        "source": dict(source),
        "fixture_attestations_sha256": _sha256(attestations),
    }


def _source_exact_gate(
    envelope: Mapping[str, Any] | None,
    *,
    envelope_sha256: str | None,
    authority: Mapping[str, Any] | None,
    authority_sha256: str | None,
    config_sha256: str | None,
    tokenizer_sha256: str | None,
    chat_protocol_sha256: str | None,
    header_sha256: str | None,
    header_capture_bytes: int | None,
) -> dict[str, Any]:
    """Validate an external source-bound envelope without manufacturing one."""

    if envelope is None:
        return {
            "status": "BLOCKED",
            "reason": "MISSING_SOURCE_EXACT_EVIDENCE",
            "detail": "a fresh sealed source-exact fixture envelope is required; historical receipts are not substitutes",
        }
    errors: list[str] = []
    if envelope.get("schema") != SOURCE_ENVELOPE_SCHEMA:
        errors.append(f"schema must be {SOURCE_ENVELOPE_SCHEMA}")
    if envelope.get("status") != "SOURCE_EXACT":
        errors.append("status must be SOURCE_EXACT")
    seal_error = _verify_seal(envelope)
    if seal_error is not None:
        errors.append(seal_error)
    source = envelope.get("source")
    if not isinstance(source, Mapping):
        errors.append("source object is required")
        source = {}
    errors.extend(
        _hash_binding_errors(
            source,
            config_sha256=config_sha256,
            tokenizer_sha256=tokenizer_sha256,
            chat_protocol_sha256=chat_protocol_sha256,
            header_sha256=header_sha256,
        )
    )
    results = envelope.get("fixture_results")
    if not isinstance(results, Mapping):
        errors.append("fixture_results object is required")
        results = {}
    for behavior in REQUIRED_BEHAVIORS:
        errors.extend(_result_binding_errors(results.get(behavior), source, behavior))
    authority_gate = _official_source_authority_gate(
        authority,
        authority_sha256=authority_sha256,
        envelope_authority_sha256=envelope.get("official_source_authority_sha256"),
        fixture_results=results,
        config_sha256=config_sha256,
        tokenizer_sha256=tokenizer_sha256,
        chat_protocol_sha256=chat_protocol_sha256,
        header_sha256=header_sha256,
        header_capture_bytes=header_capture_bytes,
    )
    authority_source = authority_gate.get("source")
    if isinstance(authority_source, Mapping):
        for key in (
            "repo",
            "revision",
            "immutable_tree_url",
            "config_sha256",
            "tokenizer_config_sha256",
            "chat_protocol_sha256",
            "safetensors_header_sha256",
        ):
            if source.get(key) != authority_source.get(key):
                errors.append(f"source envelope {key} does not match external official-source authority")
    if authority_gate["status"] != "PASS":
        errors.append(str(authority_gate["reason"]))
    return {
        "status": "PASS" if not errors else "BLOCKED",
        "reason": "SOURCE_EXACT_BOUND" if not errors else "SOURCE_EXACT_EVIDENCE_INVALID",
        "detail": "sealed source identity and every required fixture result bind to observed metadata/header" if not errors else "source-exact evidence did not satisfy the complete binding contract",
        "errors": errors,
        "source": dict(source),
        "official_source_authority": authority_gate,
    }


def evaluate(
    *,
    config_path: Path | None,
    tokenizer_config_path: Path | None,
    chat_protocol_path: Path | None = None,
    safetensors_header_path: Path | None,
    source_envelope_path: Path | None = None,
    source_authority_path: Path | None = None,
) -> dict[str, Any]:
    """Produce a bounded, non-runtime admission receipt in memory.

    Missing inputs are reportable gates, not exceptions.  This keeps status
    useful before any source transfer and ensures the default remains
    ``NOT_ADMITTED``.
    """

    config: dict[str, Any] | None = None
    tokenizer: dict[str, Any] | None = None
    tensors: dict[str, dict[str, Any]] | None = None
    input_errors: list[str] = []
    config_sha: str | None = None
    tokenizer_sha: str | None = None
    chat_protocol_sha: str | None = None
    header_sha: str | None = None
    header_capture_bytes: int | None = None
    authority_sha: str | None = None
    envelope_sha: str | None = None

    if config_path is None:
        input_errors.append("config_path is required")
    else:
        config, error = _read_json(config_path, "config")
        if error:
            input_errors.append(error)
        config_sha, error = _file_sha256(config_path)
        if error:
            input_errors.append(error)
    if tokenizer_config_path is None:
        input_errors.append("tokenizer_config_path is required")
    else:
        tokenizer, error = _read_json(tokenizer_config_path, "tokenizer_config")
        if error:
            input_errors.append(error)
        tokenizer_sha, error = _file_sha256(tokenizer_config_path)
        if error:
            input_errors.append(error)
    if chat_protocol_path is not None:
        chat_protocol_sha, error = _nonempty_file_sha256(
            chat_protocol_path, "chat protocol"
        )
        if error:
            input_errors.append(error)
    if safetensors_header_path is None:
        input_errors.append("safetensors_header_path is required")
    else:
        tensors, error = read_header_only(safetensors_header_path)
        if error:
            input_errors.append(error)
        else:
            header_sha, error = _file_sha256(
                safetensors_header_path, max_bytes=8 + MAX_HEADER_BYTES
            )
            if error:
                input_errors.append(error)
            try:
                header_capture_bytes = safetensors_header_path.stat().st_size
            except OSError as exc:
                input_errors.append(f"cannot stat safetensors header: {exc}")

    envelope: dict[str, Any] | None = None
    if source_envelope_path is not None:
        envelope, error = _read_json(source_envelope_path, "source envelope")
        if error:
            input_errors.append(error)
        else:
            envelope_sha, error = _file_sha256(source_envelope_path)
            if error:
                input_errors.append(error)
    authority: dict[str, Any] | None = None
    if source_authority_path is not None:
        authority, error = _read_json(source_authority_path, "official source authority")
        if error:
            input_errors.append(error)
        authority_sha, error = _file_sha256(source_authority_path)
        if error:
            input_errors.append(error)

    checks = _architecture_checks(config, tokenizer, tensors, chat_protocol_sha)
    structural_ok = not input_errors and all(
        row["status"] == "STRUCTURAL_PASS" for row in checks.values()
    )
    source_gate = _source_exact_gate(
        envelope,
        envelope_sha256=envelope_sha,
        authority=authority,
        authority_sha256=authority_sha,
        config_sha256=config_sha,
        tokenizer_sha256=tokenizer_sha,
        chat_protocol_sha256=chat_protocol_sha,
        header_sha256=header_sha,
        header_capture_bytes=header_capture_bytes,
    )
    admitted = structural_ok and source_gate["status"] == "PASS"
    return {
        "schema": SCHEMA,
        "status": "ARCHITECTURE_ADMITTED" if admitted else "NOT_ADMITTED",
        "admission_scope": "bounded source-exact DeepSeek-V4-Flash architecture fixture only",
        "source_inputs": {
            "config_path": str(config_path) if config_path is not None else None,
            "config_sha256": config_sha,
            "tokenizer_config_path": str(tokenizer_config_path)
            if tokenizer_config_path is not None
            else None,
            "tokenizer_config_sha256": tokenizer_sha,
            "chat_protocol_path": str(chat_protocol_path)
            if chat_protocol_path is not None
            else None,
            "chat_protocol_sha256": chat_protocol_sha,
            "safetensors_header_path": str(safetensors_header_path)
            if safetensors_header_path is not None
            else None,
            "safetensors_header_sha256": header_sha,
            "safetensors_header_capture_bytes": header_capture_bytes,
            "official_source_authority_path": str(source_authority_path)
            if source_authority_path is not None
            else None,
            "official_source_authority_sha256": authority_sha,
            "source_envelope_sha256": envelope_sha,
        },
        "structural_fixture": {
            "status": "STRUCTURAL_PASS" if structural_ok else "BLOCKED",
            "input_errors": input_errors,
            "checks": checks,
        },
        "source_exact_evidence": source_gate,
        "does_not_establish": [
            "source download or source residency",
            "complete model artifact or Condense execution",
            "native FP4 or FP8 codec implementation",
            "CPU/source oracle",
            "Metal forward or kernel support",
            "quality, capability, or route-stability result",
            "runtime residency, latency, or TPS result",
        ],
        "next_gate": (
            "source_exact_fixture_behaviour_review"
            if structural_ok and source_gate["status"] != "PASS"
            else "architecture_fixture_repair"
            if not structural_ok
            else "source_exact_cpu_oracle"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="bounded official config.json capture")
    parser.add_argument(
        "--tokenizer-config", type=Path, help="bounded official tokenizer_config.json capture"
    )
    parser.add_argument(
        "--chat-protocol",
        type=Path,
        help="optional bounded source-owned chat encoding/template protocol artifact",
    )
    parser.add_argument(
        "--safetensors-header",
        type=Path,
        help="header-only safetensors capture: prefix + JSON header, no payload",
    )
    parser.add_argument(
        "--source-envelope",
        type=Path,
        help="sealed source-exact fixture envelope; omitted input intentionally remains NOT_ADMITTED",
    )
    parser.add_argument(
        "--source-authority",
        type=Path,
        help="separate official-manifest/control-plane authority record; CLI still requires an external trusted verifier",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON (the only output format; accepted for command-surface consistency)",
    )
    parser.add_argument("--out", type=Path, help="optional explicit JSON receipt path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = evaluate(
        config_path=args.config,
        tokenizer_config_path=args.tokenizer_config,
        chat_protocol_path=args.chat_protocol,
        safetensors_header_path=args.safetensors_header,
        source_envelope_path=args.source_envelope,
        source_authority_path=args.source_authority,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if receipt["status"] == "ARCHITECTURE_ADMITTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
