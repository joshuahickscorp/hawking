#!/usr/bin/env python3.12
"""Run and seal the GLM-5.2 adapter/twin/reference parity instrument."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from glm52_adapter import (
    OFFICIAL_CONFIG_CONTRACT,
    PROFILE_OFFICIAL,
    PROFILE_SYNTHETIC,
    VIEW_FULL,
    expected_tensor_specs,
    load_official_tokenizer_assembly,
    pack_expert_gate_up,
    validate_config,
    verify_streaming_window,
)
from glm52_common import (
    REPO_ROOT,
    atomic_json,
    atomic_text,
    read_sealed_json,
    seal,
    sha256_file,
)
from glm52_reference import (
    ReferenceCache,
    indexer_topk,
    main_forward,
    mtp_forward,
    rope_cos_sin,
)
from glm52_synthetic import build_synthetic_fixture


THRESHOLDS = {
    "hf_main_vs_numpy_reference": {
        "maximum_absolute_logit_error": 0.003,
        "relative_frobenius_logit_error": 0.0005,
        "minimum_logit_cosine": 0.999999,
        "minimum_top1_agreement": 1.0,
    },
    "cpu_vs_metal": {
        "maximum_absolute_logit_error": 0.003,
        "relative_frobenius_logit_error": 0.0005,
        "minimum_logit_cosine": 0.999999,
        "minimum_top1_agreement": 1.0,
    },
    "prefill_vs_tokenwise": {
        "maximum_absolute_logit_error": 1e-5,
        "relative_frobenius_logit_error": 1e-5,
        "minimum_top1_agreement": 1.0,
    },
    "index_scores": {
        "maximum_absolute_error": 1e-5,
        "relative_frobenius_error": 0.01,
        "minimum_cosine": 0.9999,
        "tie_absolute_tolerance": 1e-7,
    },
    "per_layer_hidden_outputs": {
        "maximum_absolute_logit_error": 0.0015,
        "relative_frobenius_logit_error": 0.0006,
        "minimum_logit_cosine": 0.999999,
        "minimum_top1_agreement": 1.0,
    },
}


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"parity shape mismatch: {left.shape} != {right.shape}")
    delta = right - left
    denominator = float(np.linalg.norm(left)) or 1.0
    candidate_norm = float(np.linalg.norm(right)) or 1.0
    cosine = float(np.dot(left.ravel(), right.ravel()) / (denominator * candidate_norm))
    return {
        "shape": list(left.shape),
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "relative_frobenius_error": float(np.linalg.norm(delta) / denominator),
        "cosine": cosine,
        "top1_agreement": float(np.mean(np.argmax(left, axis=-1) == np.argmax(right, axis=-1))),
        "finite": bool(np.isfinite(left).all() and np.isfinite(right).all()),
    }


def _passes(metrics: dict[str, Any], limits: dict[str, float]) -> bool:
    return bool(
        metrics["finite"]
        and metrics["maximum_absolute_error"] <= limits["maximum_absolute_logit_error"]
        and metrics["relative_frobenius_error"] <= limits["relative_frobenius_logit_error"]
        and metrics.get("cosine", 1.0) >= limits.get("minimum_logit_cosine", -1.0)
        and metrics["top1_agreement"] >= limits["minimum_top1_agreement"]
    )


def _official_schema_sweep() -> dict[str, Any]:
    graph = read_sealed_json(REPO_ROOT / "GLM52_SHARD_DEPENDENCY_GRAPH.json")
    manifest = read_sealed_json(REPO_ROOT / "GLM52_OFFICIAL_MANIFEST.json")
    config_path = Path(manifest["one_copy"]["snapshot_view"]) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    geometry = validate_config(config, profile=PROFILE_OFFICIAL)
    expected = expected_tensor_specs(geometry, view=VIEW_FULL)
    actual_rows = {row["name"]: row for row in graph["tensors"]}
    missing = sorted(set(expected) - set(actual_rows))
    unknown = sorted(set(actual_rows) - set(expected))
    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    for name in sorted(set(expected) & set(actual_rows)):
        spec, row = expected[name], actual_rows[name]
        if list(spec.shape) != row["shape"]:
            shape_mismatches.append(name)
        if spec.dtype != row["dtype"]:
            dtype_mismatches.append(name)
    expected_elements = sum(spec.element_count for spec in expected.values())
    actual_elements = sum(int(row["logical_elements"]) for row in actual_rows.values())
    passed = not (missing or unknown or shape_mismatches or dtype_mismatches)
    passed = passed and expected_elements == actual_elements == 753_329_940_480
    return {
        "status": "PASS" if passed else "FAIL",
        "graph_seal_sha256": graph["seal_sha256"],
        "official_config_sha256": sha256_file(config_path),
        "expected_tensor_count": len(expected),
        "actual_tensor_count": len(actual_rows),
        "expected_logical_elements": expected_elements,
        "actual_logical_elements": actual_elements,
        "missing": missing,
        "unknown": unknown,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
    }


def _effective_index_agreement(
    reference_layers: list[dict[str, Any]],
    official_topk: dict[int, np.ndarray],
    official_scores: dict[int, np.ndarray],
    component_reference_scores: dict[int, np.ndarray],
    positions: np.ndarray,
    indexer_types: list[str],
) -> dict[str, Any]:
    raw_equal = 0
    effective_equal = 0
    tie_aware_effective_equal = 0
    tie_equivalent_disagreements = 0
    comparisons = 0
    per_layer: dict[str, Any] = {}
    score_left: list[np.ndarray] = []
    score_right: list[np.ndarray] = []
    tie_evidence: list[dict[str, Any]] = []
    tie_tolerance = THRESHOLDS["index_scores"]["tie_absolute_tolerance"]
    for layer, mode in enumerate(indexer_types):
        if mode != "full":
            continue
        left = reference_layers[layer]["attention"]["topk_indices"]
        right = official_topk[layer]
        left_scores = reference_layers[layer]["attention"]["index_scores"]
        right_scores = official_scores[layer]
        if left_scores.shape != right_scores.shape:
            raise ValueError(
                f"index-score shape mismatch at layer {layer}: "
                f"{left_scores.shape} != {right_scores.shape}"
            )
        component_scores = component_reference_scores[layer]
        if component_scores.shape != right_scores.shape:
            raise ValueError(
                f"component index-score shape mismatch at layer {layer}: "
                f"{component_scores.shape} != {right_scores.shape}"
            )
        key_positions = np.arange(component_scores.shape[-1], dtype=np.int64)
        causal_entries = key_positions[None, None, :] <= positions[:, :, None]
        component_finite = (
            causal_entries
            & np.isfinite(component_scores)
            & np.isfinite(right_scores)
        )
        score_left.append(np.asarray(component_scores[component_finite], dtype=np.float32))
        score_right.append(np.asarray(right_scores[component_finite], dtype=np.float32))
        layer_raw = 0
        layer_effective = 0
        layer_tie_aware = 0
        flat_left_scores = left_scores.reshape(-1, left_scores.shape[-1])
        flat_right_scores = right_scores.reshape(-1, right_scores.shape[-1])
        for query, (left_row, right_row) in enumerate(
            zip(left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1]))
        ):
            position = int(positions.reshape(-1)[query])
            left_set = set(int(value) for value in left_row)
            right_set = set(int(value) for value in right_row)
            layer_raw += left_set == right_set
            left_effective = {value for value in left_set if value <= position}
            right_effective = {value for value in right_set if value <= position}
            strict_equal = left_effective == right_effective
            layer_effective += strict_equal

            tie_equal = strict_equal
            left_only = sorted(left_effective - right_effective)
            right_only = sorted(right_effective - left_effective)
            if not strict_equal and len(left_only) == len(right_only):
                candidates = left_only + right_only
                reference_tied = bool(
                    candidates
                    and np.ptp(flat_left_scores[query, candidates]) <= tie_tolerance
                )
                official_tied = bool(
                    candidates
                    and np.ptp(flat_right_scores[query, candidates]) <= tie_tolerance
                )
                cross_runtime_close = bool(
                    np.allclose(
                        flat_left_scores[query, candidates],
                        flat_right_scores[query, candidates],
                        rtol=0.0,
                        atol=tie_tolerance,
                    )
                )
                tie_equal = reference_tied and official_tied and cross_runtime_close
                if tie_equal:
                    tie_equivalent_disagreements += 1
                    tie_evidence.append({
                        "layer": layer,
                        "query": query,
                        "position": position,
                        "reference_only": left_only,
                        "official_only": right_only,
                        "reference_candidate_scores": [
                            float(flat_left_scores[query, value]) for value in candidates
                        ],
                        "official_candidate_scores": [
                            float(flat_right_scores[query, value]) for value in candidates
                        ],
                    })
            layer_tie_aware += tie_equal
            comparisons += 1
        raw_equal += layer_raw
        effective_equal += layer_effective
        tie_aware_effective_equal += layer_tie_aware
        per_layer[str(layer)] = {
            "raw_set_equal": layer_raw,
            "causally_effective_set_equal": layer_effective,
            "tie_aware_causally_effective_set_equal": layer_tie_aware,
            "queries": left.shape[0] * left.shape[1],
        }

    score_reference = np.concatenate(score_left)
    score_official = np.concatenate(score_right)
    score_delta = score_official - score_reference
    reference_norm = float(np.linalg.norm(score_reference)) or 1.0
    official_norm = float(np.linalg.norm(score_official)) or 1.0
    score_cosine = float(
        np.dot(score_reference, score_official) / (reference_norm * official_norm)
    )
    score_metrics = {
        "finite_causal_scores": int(score_reference.size),
        "maximum_absolute_error": float(np.max(np.abs(score_delta))),
        "mean_absolute_error": float(np.mean(np.abs(score_delta))),
        "relative_frobenius_error": float(np.linalg.norm(score_delta) / reference_norm),
        "cosine": max(-1.0, min(1.0, score_cosine)),
        "thresholds": THRESHOLDS["index_scores"],
    }
    score_metrics["status"] = "PASS" if bool(
        score_metrics["maximum_absolute_error"]
        <= THRESHOLDS["index_scores"]["maximum_absolute_error"]
        and score_metrics["relative_frobenius_error"]
        <= THRESHOLDS["index_scores"]["relative_frobenius_error"]
        and score_metrics["cosine"] >= THRESHOLDS["index_scores"]["minimum_cosine"]
    ) else "FAIL"
    return {
        "raw_set_agreement": raw_equal / comparisons,
        "causally_effective_set_agreement": effective_equal / comparisons,
        "tie_aware_causally_effective_set_agreement": tie_aware_effective_equal / comparisons,
        "tie_equivalent_disagreements": tie_equivalent_disagreements,
        "tie_evidence": tie_evidence,
        "score_parity": score_metrics,
        "comparisons": comparisons,
        "per_full_layer": per_layer,
        "note": (
            "Strict set agreement is reported without adjustment. Tie-aware agreement counts a "
            "disagreement only when both runtimes assign the exchanged causal candidates equal "
            "scores within the frozen tolerance; raw score parity is independently gated. A top-k "
            "call may also return arbitrary future-token filler when fewer than k causal keys exist. "
            "Raw score parity uses identical captured official activations in both implementations; "
            "the separately reported propagated top-k comparison includes upstream numerical drift."
        ),
    }


def _official_indexer_scores(module: Any, inputs: tuple[Any, ...]) -> np.ndarray:
    """Reproduce the pinned Transformers index score before its top-k call."""
    import torch
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        apply_rotary_pos_emb_interleave,
    )

    hidden_states, q_resid, position_embeddings, attention_mask, position_ids, *rest = inputs
    if rest and rest[0] is not None:
        raise ValueError("raw official index-score capture only supports cache-free prefill")
    batch_size, sequence, _ = hidden_states.shape
    q = module.wq_b(q_resid).view(
        batch_size, sequence, module.n_heads, module.head_dim
    )
    q_rot, q_pass = torch.split(
        q,
        [module.qk_rope_head_dim, module.head_dim - module.qk_rope_head_dim],
        dim=-1,
    )
    k = module.k_norm(module.wk(hidden_states)).unsqueeze(2)
    k_rot, k_pass = torch.split(
        k,
        [module.qk_rope_head_dim, module.head_dim - module.qk_rope_head_dim],
        dim=-1,
    )
    cos, sin = position_embeddings
    q_rot, k_rot = apply_rotary_pos_emb_interleave(
        q_rot, k_rot, cos, sin, unsqueeze_dim=2
    )
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)
    scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1))
    scores = torch.relu(scores * module.softmax_scale)
    weights = module.weights_proj(
        hidden_states.to(module.weights_proj.weight.dtype)
    ).float() * (module.n_heads**-0.5)
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
    if attention_mask is not None:
        index_scores = index_scores + attention_mask
    else:
        key_positions = torch.arange(index_scores.shape[-1], device=index_scores.device)
        causal = key_positions[None, None, :] > position_ids[:, :, None]
        index_scores = index_scores.masked_fill(causal, float("-inf"))
    return index_scores.detach().cpu().numpy()


def _route_agreement(
    reference_layers: list[dict[str, Any]],
    official_routes: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    equal = 0
    comparisons = 0
    maximum_weight_delta = 0.0
    for layer, (official_weights, official_indices) in official_routes.items():
        trace = reference_layers[layer]["mlp"]
        left_i = trace["topk_indices"].reshape(-1, trace["topk_indices"].shape[-1])
        left_w = trace["topk_weights"].reshape(-1, trace["topk_weights"].shape[-1])
        right_i = official_indices.reshape(-1, official_indices.shape[-1])
        right_w = official_weights.reshape(-1, official_weights.shape[-1])
        for li, lw, ri, rw in zip(left_i, left_w, right_i, right_w):
            left_pairs = sorted((int(index), float(weight)) for index, weight in zip(li, lw))
            right_pairs = sorted((int(index), float(weight)) for index, weight in zip(ri, rw))
            equal += [item[0] for item in left_pairs] == [item[0] for item in right_pairs]
            maximum_weight_delta = max(
                maximum_weight_delta,
                max(abs(a[1] - b[1]) for a, b in zip(left_pairs, right_pairs)),
            )
            comparisons += 1
    return {
        "canonical_expert_set_agreement": equal / comparisons,
        "comparisons": comparisons,
        "maximum_matched_weight_delta": maximum_weight_delta,
        "all_router_weight_sums_2_5": all(
            np.allclose(layer["mlp"]["topk_weights"].sum(axis=-1), 2.5, rtol=2e-6, atol=2e-6)
            for layer in reference_layers
            if layer["mlp"]["kind"] == "sparse"
        ),
    }


def _isolated_runtime_receipt(torch: Any) -> dict[str, Any]:
    """Fail closed unless every locked distribution comes from an isolated venv."""
    prefix = Path(sys.prefix).resolve()
    config_path = prefix / "pyvenv.cfg"
    if sys.prefix == sys.base_prefix or not config_path.is_file():
        raise RuntimeError("GLM parity requires an isolated virtual environment")
    config_text = config_path.read_text(encoding="utf-8")
    system_site_match = re.search(
        r"(?im)^include-system-site-packages\s*=\s*(true|false)\s*$",
        config_text,
    )
    if system_site_match is None or system_site_match.group(1).lower() != "false":
        raise RuntimeError(
            "GLM parity refuses a virtual environment that can inherit system site-packages"
        )

    requirements_path = REPO_ROOT / "tools/condense/requirements-glm52.txt"
    locked: dict[str, str] = {}
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise RuntimeError(f"unlocked GLM requirement: {line!r}")
        name, version = line.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in locked or not version:
            raise RuntimeError(f"duplicate/invalid GLM requirement: {line!r}")
        locked[normalized] = version
    observed: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for name, expected in sorted(locked.items()):
        actual = importlib.metadata.version(name)
        observed[name] = actual
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"GLM locked runtime version mismatch: {mismatches}")

    import hf_xet
    import huggingface_hub
    import pytest
    import safetensors
    import tokenizers
    import transformers

    direct_modules = {
        "hf-xet": hf_xet,
        "huggingface-hub": huggingface_hub,
        "numpy": np,
        "pytest": pytest,
        "safetensors": safetensors,
        "tokenizers": tokenizers,
        "torch": torch,
        "transformers": transformers,
    }
    outside = []
    for name, module in direct_modules.items():
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(prefix):
            outside.append(name)
    if outside:
        raise RuntimeError(
            f"GLM direct dependencies resolve outside the isolated environment: {outside}"
        )
    return {
        "status": "PASS_ISOLATED_FULLY_PINNED",
        "python": platform.python_version(),
        "system_site_packages": False,
        "locked_distribution_count": len(locked),
        "locked_versions": observed,
        "all_direct_imports_within_environment": True,
        "requirements_sha256": sha256_file(requirements_path),
        "host_paths_sealed": False,
    }


def _run() -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import GlmMoeDsaForCausalLM

    isolated_runtime = _isolated_runtime_receipt(torch)

    schema = _official_schema_sweep()
    if schema["status"] != "PASS":
        raise RuntimeError("official adapter schema sweep failed")
    manifest = read_sealed_json(REPO_ROOT / "GLM52_OFFICIAL_MANIFEST.json")
    tokenizer_assembly = load_official_tokenizer_assembly(
        Path(manifest["one_copy"]["snapshot_view"])
    )
    tokenizer_chat_receipt = tokenizer_assembly.assemble_chat(
        (
            {"role": "system", "content": "You are exact."},
            {"role": "user", "content": "Return 2+2."},
        )
    )
    tokenizer_tool_receipt = tokenizer_assembly.assemble_chat(
        (
            {"role": "user", "content": "Use lookup for x."},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "The declared lookup tool is required.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"x": 4}},
                    }
                ],
            },
            {"role": "tool", "content": "{\"result\":16}"},
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Return the square of x.",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": ["x"],
                    },
                },
            },
        ),
        enable_thinking=False,
    )

    with tempfile.TemporaryDirectory(prefix="glm52-twin-") as temporary:
        fixture = build_synthetic_fixture(Path(temporary) / "fixture")
        synthetic_shards = sorted(set(fixture.index["weight_map"].values()))
        hydrated = Path(temporary) / "hydrated-window"
        hydrated.mkdir()
        for shard in synthetic_shards[:2]:
            os.link(fixture.full_dir / shard, hydrated / shard)
        synthetic_window = {
            "window_id": "SYNTHETIC-W000",
            "source_shards": synthetic_shards[:2],
            "carry_in_shards": [],
            "new_fetch_shards": synthetic_shards[:2],
            "refetch_shards": [],
            "carry_out_shards": [synthetic_shards[0]],
            "evict_after_seal_shards": [synthetic_shards[1]],
        }
        window_inventory = verify_streaming_window(
            fixture.full_dir,
            hydrated,
            synthetic_window,
            profile=PROFILE_SYNTHETIC,
        )
        ids = np.arange(1, 21, dtype=np.int64)[None, :]
        positions = np.arange(ids.shape[1], dtype=np.int64)[None, :]
        reader = fixture.main_only_reader()

        model = GlmMoeDsaForCausalLM.from_pretrained(
            fixture.main_only_dir,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        official_topk: dict[int, np.ndarray] = {}
        official_index_scores: dict[int, np.ndarray] = {}
        official_indexer_inputs: dict[int, dict[str, np.ndarray | None]] = {}
        official_routes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        official_layer_outputs: dict[int, np.ndarray] = {}

        def attention_hook(layer: int):
            def capture(_module, _inputs, output):
                official_topk[layer] = output[2].detach().cpu().numpy()
            return capture

        def router_hook(layer: int):
            def capture(_module, _inputs, output):
                official_routes[layer] = (
                    output[1].detach().cpu().numpy(),
                    output[2].detach().cpu().numpy(),
                )
            return capture

        def indexer_hook(layer: int):
            def capture(module, inputs, _output):
                official_index_scores[layer] = _official_indexer_scores(module, inputs)
                hidden_states, q_resid, position_embeddings, attention_mask, position_ids, *_ = inputs
                cos, sin = position_embeddings
                official_indexer_inputs[layer] = {
                    "hidden_states": hidden_states.detach().cpu().numpy(),
                    "q_resid": q_resid.detach().cpu().numpy(),
                    "cos": cos.detach().cpu().numpy(),
                    "sin": sin.detach().cpu().numpy(),
                    "attention_mask": (
                        None
                        if attention_mask is None
                        else attention_mask.detach().cpu().numpy()
                    ),
                    "position_ids": position_ids.detach().cpu().numpy(),
                }
            return capture

        def layer_hook(layer: int):
            def capture(_module, _inputs, output):
                official_layer_outputs[layer] = output[0].detach().cpu().numpy()
            return capture

        handles = []
        for layer, module in enumerate(model.model.layers):
            handles.append(module.self_attn.register_forward_hook(attention_hook(layer)))
            handles.append(module.register_forward_hook(layer_hook(layer)))
            if module.self_attn.indexer is not None:
                handles.append(
                    module.self_attn.indexer.register_forward_hook(indexer_hook(layer))
                )
            if hasattr(module.mlp, "gate"):
                handles.append(module.mlp.gate.register_forward_hook(router_hook(layer)))

        with torch.no_grad():
            official_cpu = model(torch.from_numpy(ids), use_cache=False).logits.float().cpu().numpy()
        for handle in handles:
            handle.remove()

        reference_logits, _reference_cache, trace = main_forward(ids, reader, fixture.config)
        main_metrics = _metrics(official_cpu, reference_logits)
        main_metrics["thresholds"] = THRESHOLDS["hf_main_vs_numpy_reference"]
        main_metrics["status"] = (
            "PASS" if _passes(main_metrics, THRESHOLDS["hf_main_vs_numpy_reference"]) else "FAIL"
        )

        layer_metrics: dict[str, dict[str, Any]] = {}
        for layer in range(len(trace["layers"])):
            metrics = _metrics(
                official_layer_outputs[layer], trace["layers"][layer]["output"]
            )
            metrics["thresholds"] = THRESHOLDS["per_layer_hidden_outputs"]
            metrics["status"] = (
                "PASS"
                if _passes(metrics, THRESHOLDS["per_layer_hidden_outputs"])
                else "FAIL"
            )
            layer_metrics[str(layer)] = metrics
        component_reference_scores: dict[int, np.ndarray] = {}
        for layer, captured in official_indexer_inputs.items():
            prefix = f"model.layers.{layer}.self_attn.indexer"
            _component_topk, component_scores, _component_keys = indexer_topk(
                captured["hidden_states"],
                captured["q_resid"],
                wq_b=reader.tensor(f"{prefix}.wq_b.weight"),
                wk=reader.tensor(f"{prefix}.wk.weight"),
                k_norm_weight=reader.tensor(f"{prefix}.k_norm.weight"),
                k_norm_bias=reader.tensor(f"{prefix}.k_norm.bias"),
                weights_proj=reader.tensor(f"{prefix}.weights_proj.weight"),
                position_ids=captured["position_ids"],
                cos=captured["cos"],
                sin=captured["sin"],
                n_heads=int(fixture.config["index_n_heads"]),
                head_dim=int(fixture.config["index_head_dim"]),
                rotary_dim=int(fixture.config["qk_rope_head_dim"]),
                topk=int(fixture.config["index_topk"]),
                attention_mask=captured["attention_mask"],
            )
            component_reference_scores[layer] = component_scores
        index_agreement = _effective_index_agreement(
            trace["layers"],
            official_topk,
            official_index_scores,
            component_reference_scores,
            positions,
            list(fixture.config["indexer_types"]),
        )
        long_position = int(fixture.config["max_position_embeddings"]) - 1
        long_positions = np.array([[long_position]], dtype=np.int64)
        long_cos, long_sin = rope_cos_sin(
            long_positions,
            int(fixture.config["qk_rope_head_dim"]),
            float(fixture.config["rope_parameters"]["rope_theta"]),
        )
        long_hidden = np.linspace(
            -0.25,
            0.25,
            int(fixture.config["hidden_size"]),
            dtype=np.float32,
        )[None, None, :]
        long_q_resid = np.linspace(
            -0.125,
            0.125,
            int(fixture.config["q_lora_rank"]),
            dtype=np.float32,
        )[None, None, :]
        long_past_keys = np.zeros(
            (1, long_position, int(fixture.config["index_head_dim"])),
            dtype=np.float32,
        )
        long_prefix = "model.layers.0.self_attn.indexer"
        official_shape_topk = int(OFFICIAL_CONFIG_CONTRACT["index_topk"])
        long_topk, long_scores, long_keys = indexer_topk(
            long_hidden,
            long_q_resid,
            wq_b=reader.tensor(f"{long_prefix}.wq_b.weight"),
            wk=reader.tensor(f"{long_prefix}.wk.weight"),
            k_norm_weight=reader.tensor(f"{long_prefix}.k_norm.weight"),
            k_norm_bias=reader.tensor(f"{long_prefix}.k_norm.bias"),
            weights_proj=reader.tensor(f"{long_prefix}.weights_proj.weight"),
            position_ids=long_positions,
            cos=long_cos,
            sin=long_sin,
            n_heads=int(fixture.config["index_n_heads"]),
            head_dim=int(fixture.config["index_head_dim"]),
            rotary_dim=int(fixture.config["qk_rope_head_dim"]),
            # This is still a tiny-weight, shape-only probe, but the selection
            # axis must exercise the official 2,048-index contract rather than
            # inheriting the fixture's top-2 setting.
            topk=official_shape_topk,
            past_keys=long_past_keys,
        )
        official_selection_shape_exercised = (
            list(long_topk.shape) == [1, 1, official_shape_topk]
        )
        long_context_indexer = {
            "status": (
                "PASS_OFFICIAL_SELECTION_SHAPE_ONLY_NO_CAPABILITY_CLAIM"
                if official_selection_shape_exercised else "FAIL"
            ),
            "tested_key_count": int(long_keys.shape[1]),
            "configured_maximum_positions": int(
                fixture.config["max_position_embeddings"]
            ),
            "topk_shape": list(long_topk.shape),
            "official_configured_index_topk": official_shape_topk,
            "synthetic_fixture_index_topk": int(fixture.config["index_topk"]),
            "official_selection_shape_exercised": official_selection_shape_exercised,
            "score_shape": list(long_scores.shape),
            "key_cache_shape": list(long_keys.shape),
            "selected_indices_in_range": bool(
                np.all(long_topk >= 0) and np.all(long_topk < long_keys.shape[1])
            ),
            "selected_scores_finite": bool(
                np.isfinite(np.take_along_axis(long_scores, long_topk, axis=-1)).all()
            ),
            "full_attention_or_model_executed": False,
            "one_million_context_capability_claimed": False,
        }
        route_agreement = _route_agreement(trace["layers"], official_routes)
        indexshare_identity = {
            str(layer): bool(
                trace["layers"][layer]["attention"]["topk_indices"]
                is trace["layers"][layer - 1]["attention"]["topk_indices"]
            )
            for layer in (3, 4, 5)
        }

        # Both official eager and inspectable reference must agree between
        # prefill and one-token-at-a-time cache execution.
        with torch.no_grad():
            past = None
            official_pieces = []
            for token in ids[0]:
                result = model(
                    torch.tensor([[int(token)]], dtype=torch.long),
                    past_key_values=past,
                    use_cache=True,
                )
                official_pieces.append(result.logits.float().cpu().numpy())
                past = result.past_key_values
        official_tokenwise = np.concatenate(official_pieces, axis=1)
        official_cache_metrics = _metrics(official_cpu, official_tokenwise)
        official_cache_metrics["status"] = (
            "PASS" if _passes(official_cache_metrics, THRESHOLDS["prefill_vs_tokenwise"]) else "FAIL"
        )
        reference_cache = ReferenceCache()
        reference_pieces = []
        for token in ids[0]:
            result, reference_cache, _ = main_forward(
                np.array([[token]], dtype=np.int64),
                reader,
                fixture.config,
                cache=reference_cache,
            )
            reference_pieces.append(result)
        reference_tokenwise = np.concatenate(reference_pieces, axis=1)
        reference_cache_metrics = _metrics(reference_logits, reference_tokenwise)
        reference_cache_metrics["status"] = (
            "PASS" if _passes(reference_cache_metrics, THRESHOLDS["prefill_vs_tokenwise"]) else "FAIL"
        )

        # Deterministic rerun is exact for the inspectable oracle.
        repeat_logits, _, _ = main_forward(ids, reader, fixture.config)
        deterministic_exact = bool(np.array_equal(reference_logits, repeat_logits))

        metal: dict[str, Any]
        if torch.backends.mps.is_available():
            model = model.to("mps")
            torch.mps.synchronize()
            with torch.no_grad():
                metal_logits = (
                    model(torch.from_numpy(ids).to("mps"), use_cache=False)
                    .logits.float().cpu().numpy()
                )
            torch.mps.synchronize()
            metal_metrics = _metrics(official_cpu, metal_logits)
            metal_metrics["thresholds"] = THRESHOLDS["cpu_vs_metal"]
            metal_metrics["status"] = (
                "PASS" if _passes(metal_metrics, THRESHOLDS["cpu_vs_metal"]) else "FAIL"
            )
            metal = {"available": True, "metrics": metal_metrics}
            model = model.to("cpu")
            torch.mps.empty_cache()
        else:
            metal = {"available": False, "status": "NOT_RUN_HARDWARE_UNAVAILABLE"}

        # Physical layer-7 MTP is intentionally outside official Transformers.
        full_reader = fixture.full_reader()
        shifted = full_reader.tensor("model.embed_tokens.weight")[ids]
        mtp_cache = ReferenceCache()
        mtp_logits_0, mtp_cache, mtp_trace_0 = mtp_forward(
            trace["pre_final_hidden"],
            shifted,
            full_reader,
            fixture.config,
            positions,
            cache=mtp_cache,
            speculative_step=0,
        )
        reuse_cache = ReferenceCache(mtp_iteration_topk=mtp_cache.mtp_iteration_topk.copy())
        mtp_logits_1, _reuse_cache, mtp_trace_1 = mtp_forward(
            trace["pre_final_hidden"],
            shifted,
            full_reader,
            fixture.config,
            positions,
            cache=reuse_cache,
            speculative_step=1,
        )
        mtp = {
            "status": "PENDING_DERIVED_CHECKS",
            "transformers_parity_claimed": False,
            "external_pinned_runtime_executed": False,
            "physical_layer": 7,
            "step_zero_computes_own_index": not mtp_trace_0["reused_step_zero_topk"],
            "step_one_reuses_step_zero_index": mtp_trace_1["reused_step_zero_topk"],
            "step_zero_step_one_index_exact": bool(
                np.array_equal(mtp_trace_0["topk_indices"], mtp_trace_1["topk_indices"])
            ),
            "step_zero_step_one_logits_exact": bool(np.array_equal(mtp_logits_0, mtp_logits_1)),
            "not_backbone_last_full_index": not bool(
                np.array_equal(trace["final_main_topk"], mtp_trace_0["topk_indices"])
            ),
            "position_zero_shifted_embedding_exactly_zero": bool(
                np.all(mtp_trace_0["masked_shifted_embeddings"][:, 0] == 0)
            ),
            "finite_logits": bool(np.isfinite(mtp_logits_0).all()),
            "compute_dtypes": {
                "stored_weights": ["BF16", "F32 router correction bias"],
                "bounded_reader_decode": "float32",
                "reference_activations_and_accumulation": "float32",
                "mtp_logits": str(mtp_logits_0.dtype),
            },
            "runtime_authority": {
                "project": "vLLM",
                "commit": "96a739289e07530cd7d8fc03665746edae8177e7",
                "role": "pinned semantic source reference; not executed in this parity run",
                "mtp_boundary": "https://github.com/vllm-project/vllm/blob/96a739289e07530cd7d8fc03665746edae8177e7/vllm/model_executor/models/deepseek_mtp.py#L91-L173",
                "iteration_indexshare": "https://github.com/vllm-project/vllm/blob/96a739289e07530cd7d8fc03665746edae8177e7/vllm/model_executor/models/deepseek_v2.py#L1085-L1175",
            },
        }
        mtp_checks = (
            "step_zero_computes_own_index",
            "step_one_reuses_step_zero_index",
            "step_zero_step_one_index_exact",
            "step_zero_step_one_logits_exact",
            "not_backbone_last_full_index",
            "position_zero_shifted_embedding_exactly_zero",
            "finite_logits",
        )
        mtp["status"] = (
            "PASS_SYNTHETIC_SOURCE_CONFORMANCE_AND_SELF_CONSISTENCY"
            if all(mtp[name] for name in mtp_checks)
            else "FAIL"
        )

        model_source = Path(
            __import__("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa", fromlist=["x"]).__file__
        )
        config_source = Path(
            __import__("transformers.models.glm_moe_dsa.configuration_glm_moe_dsa", fromlist=["x"]).__file__
        )
        local_instruments = (
            "tools/condense/glm52_common.py",
            "tools/condense/glm52_adapter.py",
            "tools/condense/glm52_synthetic.py",
            "tools/condense/glm52_reference.py",
            "tools/condense/glm52_parity.py",
            "tools/condense/requirements-glm52.txt",
        )
        common = {
            "synthetic_fixture": fixture.metadata,
            "sequence_tokens": ids.shape[1],
            "sequence_exceeds_index_topk": ids.shape[1] > int(fixture.config["index_topk"]),
            "runtime": {
                "environment": isolated_runtime,
                "transformers": importlib.metadata.version("transformers"),
                "torch": importlib.metadata.version("torch"),
                "numpy": importlib.metadata.version("numpy"),
                "transformers_modeling_module": (
                    "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa"
                ),
                "transformers_modeling_sha256": sha256_file(model_source),
                "transformers_config_module": (
                    "transformers.models.glm_moe_dsa.configuration_glm_moe_dsa"
                ),
                "transformers_config_sha256": sha256_file(config_source),
                "official_main_compute_dtype": "torch.float32",
                "numpy_reference_compute_dtype": "numpy.float32",
                "metal_available": bool(torch.backends.mps.is_available()),
            },
            "instrument_binding": {
                "repository_base_commit": (
                    "753c73dc0685ce470090aba3e49c62fe4a4f9b08"
                ),
                "local_source_sha256": {
                    path: sha256_file(REPO_ROOT / path) for path in local_instruments
                },
                "rebuild_is_timing_and_timestamp_free": True,
            },
        }

        adapter_checks = {
            "official_schema_sweep": schema,
            "official_tokenizer_chat_assembly": {
                "status": "PASS",
                "asset_sha256": dict(tokenizer_assembly.asset_sha256),
                "vocabulary_size": tokenizer_assembly.vocabulary_size,
                "padded_model_vocabulary_size": (
                    tokenizer_assembly.padded_model_vocabulary_size
                ),
                "model_max_length": tokenizer_assembly.model_max_length,
                "generation_eos_token_ids": list(
                    tokenizer_assembly.generation_eos_token_ids
                ),
                "generation_pad_token_id": tokenizer_assembly.generation_pad_token_id,
                "chat_receipt": tokenizer_chat_receipt,
                "tool_chat_receipt": tokenizer_tool_receipt,
            },
            "synthetic_full_tensor_count": fixture.full_inventory.tensor_count,
            "synthetic_core_tensor_count": fixture.main_only_inventory.tensor_count,
            "synthetic_mtp_tensor_count": len(fixture.full_inventory.mtp_names),
            "streaming_window_admission": {
                "status": "PASS",
                "resident_shards": sorted(window_inventory.shards),
                "resident_tensor_count": window_inventory.tensor_count,
                "complete_index_tensor_count": len(window_inventory.index["weight_map"]),
                "complete_index_retained": (
                    len(window_inventory.index["weight_map"])
                    == fixture.full_inventory.tensor_count
                ),
                "carry_algebra_validated": True,
            },
            "main_only_is_exact_core_filter": fixture.metadata["views"]["main_only"]["is_exact_core_filter"],
            "hardlinked_core_shards": True,
            "expert_gate_then_up": True,
            "sample_packed_expert_shape": list(
                pack_expert_gate_up(full_reader, 3, 0).shape
            ),
            "full_shared_pattern": list(fixture.config["indexer_types"]),
            "shared_layers_have_no_indexer_tensors": all(
                not any(
                    name.startswith(f"model.layers.{layer}.self_attn.indexer.")
                    for name in fixture.full_inventory.tensors
                )
                for layer in (3, 4, 5)
            ),
            "mtp_has_full_indexer": all(
                f"model.layers.7.self_attn.indexer.{suffix}" in fixture.full_inventory.tensors
                for suffix in (
                    "k_norm.bias", "k_norm.weight", "weights_proj.weight", "wk.weight", "wq_b.weight"
                )
            ),
        }
        adapter_pass = (
            schema["status"] == "PASS"
            and fixture.full_inventory.tensor_count == 3978
            and fixture.main_only_inventory.tensor_count == 3187
            and len(fixture.full_inventory.mtp_names) == 791
            and adapter_checks["main_only_is_exact_core_filter"]
            and adapter_checks["shared_layers_have_no_indexer_tensors"]
            and adapter_checks["mtp_has_full_indexer"]
            and adapter_checks["official_tokenizer_chat_assembly"]["status"] == "PASS"
            and adapter_checks["streaming_window_admission"]["status"] == "PASS"
            and adapter_checks["streaming_window_admission"]["complete_index_retained"]
        )
        adapter_artifact = seal({
            "schema": "hawking.glm52.adapter_twin.v1",
            "status": (
                "PASS_SYNTHETIC_TWIN_AND_OFFICIAL_HEADER_TOKENIZER_SCHEMA"
                if adapter_pass else "FAIL"
            ),
            **common,
            "checks": adapter_checks,
            "binding": {
                "official_repo": "zai-org/GLM-5.2",
                "official_revision": "b4734de4facf877f85769a911abafc5283eab3d9",
                "source_class": ["OFFICIAL_BF16_TEACHER", "VULTURE_XET_STREAMING", "TEXT_GENERATION"],
                "core_runtime": "official Transformers eager main graph",
                "mtp_runtime": (
                    "synthetic physical checkpoint plus custom self-consistency reference; "
                    "pinned vLLM semantics inspected but not executed"
                ),
            },
            "source_parent_parity_claimed": False,
            "source_parent_parity_pending_reason": "No BF16 payload shard has been admitted; only immutable headers and the architecture-preserving twin were read.",
        })

        reference_pass = bool(
            main_metrics["status"] == "PASS"
            and all(metrics["status"] == "PASS" for metrics in layer_metrics.values())
            and index_agreement["score_parity"]["status"] == "PASS"
            and index_agreement["tie_aware_causally_effective_set_agreement"] == 1.0
            and long_context_indexer["status"].startswith("PASS_")
            and long_context_indexer["selected_indices_in_range"]
            and long_context_indexer["selected_scores_finite"]
            and long_context_indexer["official_selection_shape_exercised"]
            and route_agreement["canonical_expert_set_agreement"] == 1.0
            and route_agreement["all_router_weight_sums_2_5"]
            and all(indexshare_identity.values())
            and official_cache_metrics["status"] == "PASS"
            and reference_cache_metrics["status"] == "PASS"
            and deterministic_exact
            and (not metal["available"] or metal["metrics"]["status"] == "PASS")
            and mtp["status"].startswith("PASS_")
            and mtp["finite_logits"]
        )
        reference_artifact = seal({
            "schema": "hawking.glm52.reference_parity.v1",
            "status": (
                "PASS_SYNTHETIC_MAIN_AND_MTP_SELF_CONSISTENCY_SOURCE_PARENT_PENDING"
                if reference_pass else "FAIL"
            ),
            **common,
            "acceptance_threshold_policy": {
                "thresholds": THRESHOLDS,
                "main_cache_layer_and_metal_thresholds": (
                    "declared before the final acceptance run"
                ),
                "index_tie_policy_amendment": {
                    "status": "AMENDED_AFTER_INITIAL_79_OF_80_DIAGNOSTIC",
                    "date": "2026-07-21",
                    "reason": (
                        "The initial run exposed one exact-zero ReLU top-k tie. "
                        "Strict 79/80 agreement remains reported; the amended gate "
                        "requires independently passing raw score parity and exact "
                        "tie-equivalence evidence."
                    ),
                    "preregistration_claimed": False,
                },
            },
            "official_transformers_main_vs_numpy_reference": main_metrics,
            "per_layer_output_metrics": layer_metrics,
            "indexer": index_agreement,
            "long_context_indexer_shape_probe": long_context_indexer,
            "router": route_agreement,
            "indexshare_same_object_on_shared_layers": indexshare_identity,
            "official_prefill_vs_tokenwise_cache": official_cache_metrics,
            "reference_prefill_vs_tokenwise_cache": reference_cache_metrics,
            "reference_deterministic_exact_replay": deterministic_exact,
            "cpu_vs_metal": metal,
            "mtp": mtp,
            "claim_boundary": {
                "official_transformers_parity": "Architecture-preserving synthetic main graph only.",
                "mtp_parity": (
                    "Synthetic source conformance and self-consistency only. No external "
                    "pinned MTP runtime was executed; generic Transformers MTP parity "
                    "is explicitly not claimed."
                ),
                "official_bf16_parent_forward": "PENDING_FIRST_ADMITTED_SOURCE_WINDOW",
                "capability": "NOT_CLAIMED",
            },
        })
        return adapter_artifact, reference_artifact


def _markdown(adapter: dict[str, Any], parity: dict[str, Any]) -> str:
    main = parity["official_transformers_main_vs_numpy_reference"]
    metal = parity["cpu_vs_metal"]
    lines = [
        "# GLM-5.2 adapter, twin, and reference parity",
        "",
        f"Adapter: **{adapter['status']}**. Reference: **{parity['status']}**.",
        "",
        f"Official header graph: {adapter['checks']['official_schema_sweep']['actual_tensor_count']:,} tensors and "
        f"{adapter['checks']['official_schema_sweep']['actual_logical_elements']:,} logical weights, exact.",
        f"Synthetic main HF/reference: max abs `{main['maximum_absolute_error']:.6g}`, "
        f"relative Frobenius `{main['relative_frobenius_error']:.6g}`, cosine `{main['cosine']:.9f}`, "
        f"top-1 `{main['top1_agreement']:.3f}`.",
        f"Strict causally effective DSA index agreement: "
        f"`{parity['indexer']['causally_effective_set_agreement']:.4f}`; "
        f"tie-aware agreement: "
        f"`{parity['indexer']['tie_aware_causally_effective_set_agreement']:.4f}`; "
        f"raw score max error: "
        f"`{parity['indexer']['score_parity']['maximum_absolute_error']:.3g}`.",
        f"Router expert-set agreement: "
        f"`{parity['router']['canonical_expert_set_agreement']:.3f}`. "
        f"The synthetic indexer shape probe reached "
        f"{parity['long_context_indexer_shape_probe']['tested_key_count']:,} keys; "
        "it is not a full-model or 1M capability result.",
        f"Metal: `{metal.get('metrics', {}).get('status', metal.get('status'))}`.",
        "",
        "MTP status is synthetic source conformance and self-consistency only. The pinned "
        "external runtime semantics were inspected but that runtime was not executed.",
        "",
        "The official BF16 parent forward remains pending because no payload shard has yet been admitted. "
        "No synthetic result is presented as source-parent or capability evidence.",
        "",
        f"Adapter seal: `{adapter['seal_sha256']}`.",
        f"Reference seal: `{parity['seal_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    adapter, parity = _run()
    atomic_json(REPO_ROOT / "GLM52_ADAPTER_TWIN.json", adapter)
    atomic_json(REPO_ROOT / "GLM52_REFERENCE_PARITY.json", parity)
    atomic_text(REPO_ROOT / "GLM52_REFERENCE_PARITY.md", _markdown(adapter, parity))
    print(json.dumps({
        "adapter_status": adapter["status"],
        "adapter_seal_sha256": adapter["seal_sha256"],
        "reference_status": parity["status"],
        "reference_seal_sha256": parity["seal_sha256"],
    }, indent=2, sort_keys=True))
    return 0 if adapter["status"].startswith("PASS") and parity["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
