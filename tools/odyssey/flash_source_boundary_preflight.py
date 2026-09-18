#!/usr/bin/env python3
"""Seal the next lawful Flash source-boundary discriminator without executing it.

This preflight checks the existing external-reference admission policy.  It
does not load the model, start Metal, accept an unsigned foreign teacher, or
treat cache hashes as hidden-state payloads.  Its purpose is to make the exact
layer/token capture obligation deterministic before the next protected source
session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.odyssey import flash_repeated_accepted_decode as gate  # noqa: E402


SCHEMA = "hawking.flash.source_boundary_capture_preflight.v2"
REFERENCE_SCHEDULE = (
    "exact_qwen4_exp_row_addressed_ple_and_experts_with_evaluated_sublayer_boundaries"
)
NATIVE_SCHEMA = "hawking.flash.layer4_handoff_initialization_control.v1"
NATIVE_STATUS = "PASSED_LAYER4_HANDOFF_INITIALIZATION_CONTROLS"


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"evidence path is not a direct regular file: {path}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "bytes": stat.st_size,
    }


def _candidate_reference(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = gate._json(path.expanduser().resolve(strict=True))
    seal = gate._verify_compact_seal(document, "external reference candidate")
    source = document.get("source")
    provider = document.get("provider")
    prompt = gate._require_token_list(
        document.get("prompt_token_ids"), "external reference prompt_token_ids"
    )
    generated = gate._require_token_list(
        document.get("generated_token_ids"), "external reference generated_token_ids"
    )
    if (
        document.get("schema") != gate.EXTERNAL_REFERENCE_SCHEMA
        or document.get("status") != gate.EXTERNAL_REFERENCE_STATUS
        or not isinstance(source, dict)
        or source.get("model") != gate.REPO_ID
        or source.get("pinned_revision") != gate.PINNED_REVISION
        or source.get("ple_inclusive") is not True
        or prompt != list(gate.PROMPT_IDS)
        or len(generated) < 2
        or not isinstance(provider, dict)
        or provider.get("execution_schedule") != REFERENCE_SCHEDULE
    ):
        raise ValueError("external reference candidate does not bind the canonical PLE-inclusive Flash gate")
    return document, {
        **_binding(path),
        "seal_sha256": seal,
        "prompt_token_ids": prompt,
        "generated_token_ids": generated,
        "candidate_only_until_admitted": True,
    }


def _owner_admission(
    reference: Path,
    model_root: Path,
    authorization: Path | None,
) -> dict[str, Any]:
    try:
        authority = gate.load_admitted_reference_for_boundary(
            reference,
            model_root,
            owner_authorization_path=authorization,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        return {
            "status": "WITHHELD_ADMISSION_AUTHORITY",
            "trust_anchor": str(gate.pinned_owner_public_key_path()),
            "trust_anchor_present": gate.pinned_owner_public_key_path().is_file(),
            "authorization_present": authorization is not None,
            "local_canonical_attempted": authorization is None,
            "admission_reject_reason": str(exc),
        }
    return {
        "status": authority.get("admission_class", "ADMITTED_REFERENCE"),
        "authorization_present": authorization is not None,
        "authorization": authority["owner_authorization"],
        "source_input_identity": authority["source_input_identity"],
        "admission_class": authority.get("admission_class"),
        "science_status": authority.get("science_status", "UNEARNED"),
    }


def _native_frontier(path: Path, token_id: int) -> dict[str, Any]:
    document = gate._json(path.expanduser().resolve(strict=True))
    seal = gate._verify_compact_seal(document, "native layer-4 initialization control")
    target = ((document.get("execution") or {}).get("target") or {})
    tokens = gate._require_token_list(document.get("token_ids"), "native control token_ids")
    if (
        document.get("schema") != NATIVE_SCHEMA
        or document.get("status") != NATIVE_STATUS
        or target.get("layer") != 4
        or not tokens
        or tokens[0] != token_id
    ):
        raise ValueError("native control does not bind the current layer-4/token-0 frontier")
    return {
        **_binding(path),
        "seal_sha256": seal,
        "status": document["status"],
        "layer": 4,
        "token_index": 0,
        "token_id": token_id,
        "raw_intermediate_payloads_present": False,
        "reuse_boundary": (
            "initialization and deterministic-scratch controls only; the historical suffix and "
            "non-PLE trajectory are not source-teacher evidence"
        ),
    }


def _required_seams() -> list[dict[str, Any]]:
    hidden = [1, 1, 10240]
    branch = [1, 1, 2560]
    return [
        {"ordinal": 0, "layer": None, "stage": "embedding_hc_tiled", "shape": hidden},
        {"ordinal": 1, "layer": 0, "stage": "mlp_injection", "shape": hidden},
        {"ordinal": 2, "layer": 1, "stage": "pre_ple", "shape": hidden},
        {"ordinal": 3, "layer": 1, "stage": "ple_additive_output", "shape": hidden},
        {"ordinal": 4, "layer": 1, "stage": "mlp_injection", "shape": hidden},
        {"ordinal": 5, "layer": 2, "stage": "mlp_injection", "shape": hidden},
        {"ordinal": 6, "layer": 3, "stage": "mlp_injection_layer4_input", "shape": hidden},
        {"ordinal": 7, "layer": 4, "stage": "attention_hyper_connection_mixed", "shape": branch},
        {"ordinal": 8, "layer": 4, "stage": "attention_branch", "shape": branch},
        {"ordinal": 9, "layer": 4, "stage": "attention_injection", "shape": hidden},
        {"ordinal": 10, "layer": 4, "stage": "mlp_hyper_connection_mixed", "shape": branch},
        {"ordinal": 11, "layer": 4, "stage": "router_top10_ids_and_weights", "shape": [10]},
        {"ordinal": 12, "layer": 4, "stage": "mlp_route_and_experts", "shape": branch},
        {"ordinal": 13, "layer": 4, "stage": "mlp_injection_layer4_output", "shape": hidden},
    ]


def _required_layer0_trace() -> list[dict[str, Any]]:
    """Declare the layer-0 attention HyperConnection precision trace.

    The prior coarse trace localized the first source/native difference to
    ``attention_hyper_connection_mixed``.  This successor traces only the
    causal attention read that precedes it: source-normalized streams, the
    low-rank projections and activation, gates, mix, injection weights, and
    the attention branch/write.  It deliberately stops before the MLP leg.
    Every payload remains a token-0 F32 observation; this does not establish a
    numerical acceptance bound or change ordinary inference.
    """
    hidden = [1, 1, 2560]
    state = [1, 1, 10240]
    low_rank = [1, 1, 320]
    injection = [1, 1, 4]
    return [
        {"ordinal": 0, "layer": 0, "stage": "attention_hyper_connection_normalized", "shape": state},
        {"ordinal": 1, "layer": 0, "stage": "attention_hyper_connection_down_projection", "shape": low_rank},
        {"ordinal": 2, "layer": 0, "stage": "attention_hyper_connection_low_rank_activation", "shape": low_rank},
        {"ordinal": 3, "layer": 0, "stage": "attention_hyper_connection_up_projection", "shape": state},
        {"ordinal": 4, "layer": 0, "stage": "attention_hyper_connection_mix_weights", "shape": state},
        {"ordinal": 5, "layer": 0, "stage": "attention_hyper_connection_mixed", "shape": hidden},
        {"ordinal": 6, "layer": 0, "stage": "attention_block_injection_weights", "shape": injection},
        {"ordinal": 7, "layer": 0, "stage": "attention_branch", "shape": hidden},
        {"ordinal": 8, "layer": 0, "stage": "post_attention_state", "shape": state},
    ]


def _required_layer4_cache() -> dict[str, list[dict[str, Any]]]:
    """Return the source-defined two-slot GDN cache contract at layer 4."""
    conv = {"state_role": "conv_state", "shape": [1, 3, 10_240]}
    recurrent = {"state_role": "recurrent_state", "shape": [1, 48, 128, 128]}
    return {
        "pre_attention": [
            {**conv, "initialization": "IMPLICIT_SOURCE_ZERO_STATE_MATERIALIZED_BY_CAPTURE"},
            {**recurrent, "initialization": "IMPLICIT_SOURCE_ZERO_STATE_MATERIALIZED_BY_CAPTURE"},
        ],
        "post_attention": [
            {**conv, "initialization": "PROVIDER_MATERIALIZED_AFTER_ATTENTION"},
            {**recurrent, "initialization": "PROVIDER_MATERIALIZED_AFTER_ATTENTION"},
        ],
    }


def _shape_elements(shape: object) -> int:
    if not isinstance(shape, list) or not shape or any(not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("cache shape is not a non-empty positive integer vector")
    elements = 1
    for value in shape:
        elements *= value
    return elements


def build_preflight(
    *,
    reference: Path,
    model_root: Path,
    native_control: Path,
    owner_authorization: Path | None,
) -> dict[str, Any]:
    candidate, reference_binding = _candidate_reference(reference)
    prompt = reference_binding["prompt_token_ids"]
    token_id = prompt[0]
    owner = _owner_admission(reference, model_root, owner_authorization)
    native = _native_frontier(native_control, token_id)
    owner_ready = owner.get("admission_class") in {
        "ADMITTED_OWNER_AUTHORIZED",
        gate.LOCAL_CANONICAL_ADMISSION_STATUS,
    }
    status = (
        "WITHHELD_MISSING_SOURCE_AND_NATIVE_BOUNDARY_PAYLOADS"
        if owner_ready
        else "WITHHELD_ADMISSION_AUTHORITY_AND_BOUNDARY_PAYLOADS"
    )
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "observed_unix_ns": time.time_ns(),
        "source": {
            "model": gate.REPO_ID,
            "pinned_revision": gate.PINNED_REVISION,
            "model_root": str(model_root.expanduser().resolve()),
            "ple_layer_ids": [1],
            "layer_types_0_through_4": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
            ],
        },
        "reference_candidate": reference_binding,
        "owner_admission": owner,
        "science_status": "UNEARNED",
        "science_unearned": True,
        "admission_class": owner.get("admission_class") or owner.get("status"),
        "admission_claim_boundary": (
            (owner.get("authorization") or {}).get("claim_boundary")
            if isinstance(owner.get("authorization"), dict)
            else owner.get("claim_boundary")
        ),
        "native_frontier": native,
        "capture_contract": {
            "token_index": 0,
            "token_id": token_id,
            "initial_cache": "empty; source Qwen GDN defines both layer-4 cache slots as implicit zeros",
            "execution_scope": "one token through embedding and source layers 0..4 only",
            "source_semantics": "installed Qwen4-Exp provider math with layer-1 PLE and immutable source rows",
            "payload_format": "raw little-endian F32 plus shape, byte count, and SHA-256 per seam",
            "cache_payloads": "raw pre/post layer-4 cache arrays normalized to F32, with original dtype and shape retained",
            "layer4_cache": _required_layer4_cache(),
            "route_payloads": "exact ordered top-10 expert ids and F32 weights at layer 4",
            "layer0_trace_contract_version": 5,
            "layer0_trace": _required_layer0_trace(),
            "seams": _required_seams(),
            "source_manifest_required": True,
            "native_manifest_required": True,
            "fresh_output_namespaces_required": True,
            "source_checkpoint_mutation_allowed": False,
        },
        "comparison_contract": {
            "order": (
                "compare the predeclared layer-0 trace in ordinal order, then canonical seams and "
                "layer-4 cache arrays; stop semantic attribution at the first differing payload"
            ),
            "continuous_metrics": ["finite", "max_abs", "relative_l2", "cosine", "exact_sha256"],
            "discrete_metrics": ["ordered_route_ids_exact"],
            "threshold_policy": (
                "diagnostic only until matched provider/native dtype and reduction-order controls establish a "
                "predeclared numerical acceptance bound; exact hashes and routes may localize but cannot alone qualify semantics"
            ),
            "cache_hashes_are_not_hidden_states": True,
        },
        "existing_capture_inventory": {
            "raw_f32_logits_steps": len(((candidate.get("logits_trace") or {}).get("path") or "") and candidate["generated_token_ids"]),
            "intermediate_hidden_payloads": 0,
            "final_cache_identity_only": True,
            "sufficient_for_boundary_comparison": False,
        },
        "next_safe_action": (
            (
                "Provenance admission is ready (owner-authorized or local-canonical); science_status=UNEARNED. "
                "Perform one fresh protected 0..4/token-0 source capture and matching bounded native capture; "
                "stop at the first exact payload/route difference. Do not run another 48-layer replay first."
            )
            if owner_ready
            else (
                "If this is a provenance-complete canonical local capture, local-canonical admission should "
                "succeed without Ed25519; otherwise the machine owner signs the exact existing V2 request. "
                "After admission, perform one fresh protected 0..4/token-0 source capture and matching bounded "
                "native capture; do not run another 48-layer replay first."
            )
        ),
        "runtime_actions": {
            "model_loaded": False,
            "gpu_or_metal_started": False,
            "daemon_restarted": False,
            "kimi_loaded": False,
        },
        "claim_boundary": (
            "Static fail-closed capture preflight only. Admission (owner-authorized or local-canonical) does not "
            "establish source/native parity, execute a model, qualify an NR, measure EBPW/TPS/capability, deploy, "
            "promote Pulsar, or retire Kimi."
            if owner_ready
            else (
                "Static fail-closed capture preflight only. It does not admit a noncanonical unsigned teacher, "
                "establish source parity, execute a model, qualify an NR, measure EBPW/TPS/capability, deploy, "
                "promote Pulsar, or retire Kimi."
            )
        ),
    }
    document["seal_sha256"] = gate._compact_utf8_sorted_seal(document)
    return document


def _write_new(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError(f"preflight output already exists; refusing to overwrite: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-contract", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=gate.DEFAULT_MODEL_ROOT)
    parser.add_argument("--native-control", type=Path, required=True)
    parser.add_argument("--owner-authorization", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = build_preflight(
        reference=args.reference_contract,
        model_root=args.model_root,
        native_control=args.native_control,
        owner_authorization=args.owner_authorization,
    )
    _write_new(args.out, document)
    print(json.dumps({
        "status": document["status"],
        "out": str(args.out),
        "seal_sha256": document["seal_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
