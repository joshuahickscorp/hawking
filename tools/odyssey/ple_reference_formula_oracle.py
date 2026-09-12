#!/usr/bin/env python3
"""Compare the canonical PLE source-formula control with the reference layer.

This is intentionally a bounded oracle rather than a full Flash load.  It
constructs a small placeholder PLE embedding only to instantiate the pinned
Transformers implementation, replaces that embedding with the exact source
rows already bound by the canonical control, and supplies the source BF16
projection/norm/convolution payloads in F32.  The result independently checks
the formula implementation while keeping the 95 GiB global PLE table out of
memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

try:  # Support direct invocation with the reference Python environment.
    from tools.odyssey.ple_access_trace import (
        DEFAULT_REFERENCE,
        load_contract,
        load_layout,
        sha256_bytes,
        trace_token_segments,
    )
    from tools.odyssey.ple_source_output_control import (
        SCHEMA as SOURCE_CONTROL_SCHEMA,
        load_source_lookup_embeddings,
        load_source_support,
    )
except ModuleNotFoundError:  # pragma: no cover - direct source-oracle invocation.
    from ple_access_trace import (
        DEFAULT_REFERENCE,
        load_contract,
        load_layout,
        sha256_bytes,
        trace_token_segments,
    )
    from ple_source_output_control import (
        SCHEMA as SOURCE_CONTROL_SCHEMA,
        load_source_lookup_embeddings,
        load_source_support,
    )


SCHEMA = "hawking.odyssey.ple_reference_formula_oracle.v1"
DEFAULT_CONTROL = Path("receipts/headless/FLASH_PLE_LAYER1_BOS_SOURCE_OUTPUT_CONTROL.json")


class FixedSourceEmbedding(torch.nn.Module):
    """A one-control substitute for the reference module's giant lookup table."""

    def __init__(self, values: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("values", torch.from_numpy(np.asarray(values, dtype=np.float32)))

    def forward(self, input_ids: torch.Tensor, past_key_values: object | None) -> torch.Tensor:
        del past_key_values
        if input_ids.ndim != 2 or tuple(input_ids.shape) != (1, self.values.shape[0]):
            raise ValueError("reference PLE oracle only accepts its sealed one-batch token control")
        return self.values.unsqueeze(0)


def _array_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    candidate = np.asarray(candidate, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if candidate.shape != reference.shape:
        raise ValueError("candidate/reference PLE outputs have different shapes")
    delta = candidate - reference
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    candidate_norm = float(np.linalg.norm(candidate.reshape(-1)))
    denominator = max(reference_norm * candidate_norm, 1e-30)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(math.sqrt(float(np.mean(delta * delta)))),
        "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / max(reference_norm, 1e-30)),
        "cosine": float(np.dot(candidate.reshape(-1), reference.reshape(-1)) / denominator),
        "finite": bool(np.isfinite(candidate).all() and np.isfinite(reference).all()),
    }


def _read_bound_f32(path: Path, record: dict[str, Any]) -> np.ndarray:
    raw = path.read_bytes()
    if (
        record.get("dtype") != "F32_LE"
        or record.get("bytes") != len(raw)
        or record.get("elements") != len(raw) // 4
        or record.get("sha256") != _array_hash(raw)
    ):
        raise ValueError("candidate PLE output file disagrees with its source-control receipt")
    return np.frombuffer(raw, dtype="<f4").copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--reference-implementation", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--max-abs", type=float, default=5e-6)
    parser.add_argument("--relative-l2", type=float, default=5e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("receipts/headless/FLASH_PLE_LAYER1_BOS_REFERENCE_FORMULA_PARITY.json"),
    )
    args = parser.parse_args()
    if args.max_abs <= 0 or args.relative_l2 <= 0:
        raise ValueError("parity tolerances must be positive")
    control_path = args.control.expanduser().resolve()
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if (
        control.get("schema") != SOURCE_CONTROL_SCHEMA
        or control.get("status")
        != "SOURCE_BOUND_PLE_OUTPUT_CONTROL__NO_INDEPENDENT_PLE_OUTPUT_PARITY"
    ):
        raise ValueError("source control receipt does not have the expected bounded formula status")
    reference_path = args.reference_implementation.expanduser().resolve()
    reference_contract = control.get("reference_contract")
    specimen = control.get("specimen")
    input_control = control.get("input_control")
    token_sequence = control.get("token_sequence")
    outputs = control.get("outputs")
    if not all(isinstance(item, dict) for item in (reference_contract, specimen, input_control, token_sequence, outputs)):
        raise ValueError("source control receipt lacks required bindings")
    if reference_contract.get("implementation_sha256") != sha256_bytes(reference_path.read_bytes()):
        raise ValueError("reference implementation hash changed since the source control was sealed")
    spec = Path(str(specimen["root"])).expanduser().resolve()
    contract, config_binding = load_contract(spec, 0)
    if config_binding.get("sha256") != (reference_contract.get("config") or {}).get("sha256"):
        raise ValueError("source control config binding does not match the source specimen")
    config = json.loads((spec / "config.json").read_text(encoding="utf-8"))["text_config"]
    hidden_size = int(config["hidden_size"])
    hc_count = int(config["hc_count"])
    tokens = token_sequence.get("token_ids")
    if not isinstance(tokens, list) or len(tokens) != 1 or not all(isinstance(item, int) for item in tokens):
        raise ValueError("reference PLE oracle currently requires exactly one sealed token")
    input_state_record = input_control.get("state")
    candidate_record = outputs.get("ple_injection")
    if not isinstance(input_state_record, dict) or not isinstance(candidate_record, dict):
        raise ValueError("source control lacks sealed input or PLE injection records")
    input_state = _read_bound_f32(Path(str(input_state_record["path"])), input_state_record)
    candidate = _read_bound_f32(Path(str(candidate_record["path"])), candidate_record)
    if input_state.size != hidden_size * hc_count or candidate.size != hidden_size * hc_count:
        raise ValueError("source control state geometry disagrees with the config")
    split_parts = int(config["split_ngram_parts"])
    shards, _ = load_layout(spec, contract, split_parts)
    trace = trace_token_segments(contract, [tokens])
    embeddings, _, _ = load_source_lookup_embeddings(shards, trace, contract)
    support = load_source_support(spec, contract, hidden_size=hidden_size, hc_count=hc_count)

    # The reference class only needs its full original PLE geometry.  A tiny
    # placeholder n-gram table prevents allocation of the source's 95 GiB
    # global table; FixedSourceEmbedding supplies the exact source rows.
    small_config = Qwen4ExpTextConfig(
        vocab_size=contract.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        hc_count=hc_count,
        hc_lowrank=int(config["hc_lowrank"]),
        ple_layer_ids=[1],
        ple_embed_dim=contract.ple_embed_dim,
        ple_conv_kernel_size=int(config["ple_conv_kernel_size"]),
        ngram_size=contract.ngram_size,
        heads_per_ngram=contract.heads_per_ngram,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=8,
        seed=contract.seed,
        split_ngram_parts=1,
        eos_token_id=contract.eos_token_id,
        layer_types=["linear_attention"],
    )
    layer = Qwen4ExpTextPLELayer(small_config, layer_idx=0, ple_layer_index=0).float()
    layer.ple_embedding = FixedSourceEmbedding(embeddings)
    with torch.no_grad():
        layer.key_proj.weight.copy_(torch.from_numpy(support.key_proj))
        layer.value_proj.weight.copy_(torch.from_numpy(support.value_proj))
        layer.norm_key.weight.copy_(torch.from_numpy(support.norm_key))
        layer.norm_query.weight.copy_(torch.from_numpy(support.norm_query))
        layer.norm_conv.weight.copy_(torch.from_numpy(support.norm_conv))
        layer.conv1d.weight.copy_(torch.from_numpy(support.conv[:, None, :]))
    reference = (
        layer(
            torch.from_numpy(input_state.reshape(1, 1, -1)),
            torch.tensor([tokens], dtype=torch.long),
            None,
        )
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.float32)
    )
    metrics = _metrics(candidate, reference)
    passed = bool(
        metrics["finite"]
        and metrics["max_abs"] <= args.max_abs
        and metrics["relative_l2"] <= args.relative_l2
    )
    reference_path_out = args.output.with_suffix(".reference_ple_injection.f32")
    reference_path_out.parent.mkdir(parents=True, exist_ok=True)
    raw_reference = np.asarray(reference, dtype="<f4").tobytes(order="C")
    reference_path_out.write_bytes(raw_reference)
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "REFERENCE_FRAMEWORK_PLE_FORMULA_PARITY_PASS" if passed else "REFERENCE_FRAMEWORK_PLE_FORMULA_PARITY_FAIL",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_control": {
            "path": str(control_path),
            "sha256": sha256_bytes(control_path.read_bytes()),
            "seal_sha256": control.get("seal_sha256"),
        },
        "reference": {
            "framework": "transformers",
            "framework_version": reference_contract.get("implementation_version"),
            "implementation": str(reference_path),
            "implementation_sha256": sha256_bytes(reference_path.read_bytes()),
            "class": "Qwen4ExpTextPLELayer",
            "device": "cpu",
            "dtype": "float32",
            "lookup_substitution": "FixedSourceEmbedding over the exact source-bound rows; giant source table was not instantiated",
        },
        "candidate": candidate_record,
        "reference_output": {
            "path": str(reference_path_out),
            "dtype": "F32_LE",
            "elements": int(reference.size),
            "bytes": len(raw_reference),
            "sha256": _array_hash(raw_reference),
        },
        "metrics": metrics,
        "thresholds": {"max_abs": args.max_abs, "relative_l2": args.relative_l2},
        "claim_boundary": (
            "This passes or fails only the bounded PLE formula evaluation for one sealed Flash BOS control. "
            "It verifies the canonical F32 formula against the pinned Transformers implementation with exact "
            "source lookup/support values. It does not establish full checkpoint forward parity, layer-1/full-model "
            "execution, a native PLE kernel, repeated token state, complete NR EBPW, TPS, or capability."
        ),
        "promotion_allowed": False,
        "next": (
            "Use this independently checked PLE output as the bounded target for non-uniform lookup representation "
            "and native-runtime discriminators. Preserve its one-control scope until additional source inputs and "
            "token boundaries are tested."
        ),
    }
    document["seal_sha256"] = sha256_bytes(json.dumps(document, sort_keys=True).encode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": document["status"],
                "max_abs": metrics["max_abs"],
                "relative_l2": metrics["relative_l2"],
                "out": str(args.output),
                "seal": document["seal_sha256"],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
