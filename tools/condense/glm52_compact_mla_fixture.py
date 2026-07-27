#!/usr/bin/env python3.12
"""Build the bounded sparse-MLP direct-u8 fixture for complete-token parity.

This is deliberately separate from ``glm52_gravity_fixture.py``. The ordinary
checked-in fixture must retain its historical R0/native mix, while the compact
absorbed K/V kernels require the Math-Preserve attention geometry:

    D32 / S1 / sub32 / card256 / bits8

The fixture is one sparse layer and 16 vocabulary rows so a full FP64 reference
stays well-conditioned under Numeric Parity V2.1. Its compact attention and
routed/shared expert projection row windows are large enough that product
quantization physically carries all 256 codes. The explicit authority freezes
complete-token logits, DSA positions, and router expert choices. Nothing
produced by this script is a production artifact.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_gravity_fixture as fixture  # noqa: E402
import glm52_gravity_source as source_mod  # noqa: E402
import gravity_format  # noqa: E402

TOKENS = [4, 13]
CONFIG = {
    **fixture.CONFIG,
    "hidden_size": 256,
    "num_hidden_layers": 1,
    "num_attention_heads": 16,
    "kv_lora_rank": 32,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "vocab_size": 16,
    "first_k_dense_replace": 0,
    "indexer_types": ["full"],
    "mlp_layer_types": ["sparse"],
}
DIRECT_U8_GEOMETRY = {
    "dim": 32,
    "subspaces": 1,
    "sub": 32,
    "cardinality": 256,
    "bits": 8,
}


def _fp64_reference_module() -> types.ModuleType:
    """Load the pinned reference with every explicit numpy f32 forced to f64."""
    path = HERE / "glm52_reference.py"
    source = (
        path.read_text()
        .replace("np.float32", "np.float64")
        .replace("np.matmul", "sequential_matmul_f64")
    )
    name = "glm52_reference_compact_mla_f64"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.sequential_matmul_f64 = _sequential_matmul_f64
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def _sequential_matmul_f64(
    left: np.ndarray,
    right: np.ndarray,
    dtype: object | None = None,
) -> np.ndarray:
    """Deterministic f64 matmul with explicit ascending inner reductions."""
    del dtype
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim < 2 or right.ndim < 2 or left.shape[-1] != right.shape[-2]:
        raise ValueError(f"unsupported matmul shapes: {left.shape} x {right.shape}")
    batch_shape = np.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = np.broadcast_to(left, batch_shape + left.shape[-2:])
    right = np.broadcast_to(right, batch_shape + right.shape[-2:])
    rows, inner, cols = left.shape[-2], left.shape[-1], right.shape[-1]
    out = np.empty(batch_shape + (rows, cols), dtype=np.float64)
    for batch_index in np.ndindex(batch_shape):
        for row in range(rows):
            for col in range(cols):
                acc = 0.0
                for k in range(inner):
                    acc += float(left[batch_index + (row, k)]) * float(
                        right[batch_index + (k, col)]
                    )
                out[batch_index + (row, col)] = acc
    return out


def _prompts() -> list[list[int]]:
    # Frozen after screening the artifact-backed f32 and explicit FP64 paths:
    # all four have meaningful-relative margin under V2.1, exact greedy/top-5,
    # exact DSA decisions, and distinct router choices.
    return [TOKENS, [15], [3, 10], [10, 6]]


def _direct_u8_tensor_names() -> list[str]:
    """Compact-attention and expert projections required to be physical R4."""
    layer = "model.layers.0"
    names = [
        f"{layer}.self_attn.kv_b_proj.weight",
        f"{layer}.self_attn.o_proj.weight",
    ]
    for expert in range(int(CONFIG["n_routed_experts"])):
        prefix = f"{layer}.mlp.experts.{expert}"
        names.extend(f"{prefix}.{projection}_proj.weight" for projection in ("gate", "up", "down"))
    shared = f"{layer}.mlp.shared_experts"
    names.extend(f"{shared}.{projection}_proj.weight" for projection in ("gate", "up", "down"))
    return names


def _validate_direct_u8_payloads(artifact: Path, header: dict) -> list[str]:
    """Fail closed unless every compact/expert projection has exact R4 bytes."""
    descriptors = {entry["name"]: entry for entry in header["tensors"]}
    validated = []
    for name in _direct_u8_tensor_names():
        descriptor = descriptors.get(name)
        if descriptor is None:
            raise RuntimeError(f"missing direct-u8 fixture tensor {name}")
        if descriptor.get("codec") != "gravity-pq":
            raise RuntimeError(
                f"{name}: expected gravity-pq, got {descriptor.get('codec')!r}"
            )
        payload = gravity_format.read_tensor(artifact, name)
        codes = fixture.pack.deserialize(payload)
        cardinality = int(codes["codebooks"][0].shape[0])
        actual = {
            "dim": int(codes["D"]),
            "subspaces": int(codes["S"]),
            "sub": int(codes["sub"]),
            "cardinality": cardinality,
            "bits": int(fixture.pack.index_bits(cardinality)),
        }
        if actual != DIRECT_U8_GEOMETRY:
            raise RuntimeError(
                f"{name}: physical PQ geometry {actual} != {DIRECT_U8_GEOMETRY}"
            )
        validated.append(name)
    return validated


def _materialize_fp64_authority_weights(
    weights: source_mod.GravityGlmSource, names: list[str]
) -> dict[str, np.ndarray]:
    """Decode physical fixture tensors once, then widen exactly for f64 math.

    ``GravityGlmSource.matvec`` is intentionally a direct f32 PQ executor and
    therefore cannot serve as an FP64 accumulation authority. This fixture is
    tiny enough to materialize its encoded matrices without weakening the
    production runtime's bounded-decode contract.
    """
    return {
        name: np.asarray(weights.tensor(name), dtype=np.float64) for name in names
    }


def _last_token(values: object, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 2 or array.shape[0] != 1 or array.shape[1] == 0:
        raise RuntimeError(f"{label}: expected non-empty [1, sequence, ...], got {array.shape}")
    return array[0, -1]


def _authority_record(tokens: list[int], logits: np.ndarray, trace: dict) -> dict:
    """Freeze FP64 complete-token values and exact last-token decisions."""
    sparse_layers = []
    router_logits = []
    expert_weights = []
    for layer in trace["layers"]:
        mlp = layer["mlp"]
        if mlp["kind"] != "sparse":
            continue
        sparse_layers.append(
            [
                int(value)
                for value in _last_token(mlp["topk_indices"], "expert choices")
            ]
        )
        router_logits.append(
            [
                float(value)
                for value in _last_token(mlp["router_logits"], "router logits")
            ]
        )
        expert_weights.append(
            [
                float(value)
                for value in _last_token(mlp["topk_weights"], "expert weights")
            ]
        )
    expected_sparse = list(CONFIG["mlp_layer_types"]).count("sparse")
    if len(sparse_layers) != expected_sparse or expected_sparse == 0:
        raise RuntimeError(
            f"authority exercised {len(sparse_layers)} sparse layers, expected {expected_sparse}"
        )
    return {
        "tokens": list(tokens),
        "logits": [
            float(value) for value in _last_token(logits, "complete-token logits")
        ],
        "final_topk": [
            int(value)
            for value in _last_token(trace["final_main_topk"], "final DSA top-k")
        ],
        "expert_choices": sparse_layers,
        "router_logits": router_logits,
        "expert_weights": expert_weights,
    }


def _validate_authorities(authorities: list[dict]) -> dict:
    """Require explicit, finite authorities and non-vacuous router decisions."""
    expected_prompts = _prompts()
    if [row["tokens"] for row in authorities] != expected_prompts:
        raise RuntimeError("FP64 authority prompt order drifted")
    patterns: set[tuple[tuple[int, ...], ...]] = set()
    experts: set[int] = set()
    for row in authorities:
        if len(row["logits"]) != int(CONFIG["vocab_size"]) or not np.isfinite(
            row["logits"]
        ).all():
            raise RuntimeError(f"invalid complete-token logits for {row['tokens']}")
        expected_dsa = min(int(CONFIG["index_topk"]), len(row["tokens"]))
        if len(row["final_topk"]) != expected_dsa or len(set(row["final_topk"])) != expected_dsa:
            raise RuntimeError(f"invalid exact DSA authority for {row['tokens']}")
        if len(row["expert_choices"]) != 1:
            raise RuntimeError(f"sparse router was not exercised for {row['tokens']}")
        choice = row["expert_choices"][0]
        expected_experts = int(CONFIG["num_experts_per_tok"])
        if len(choice) != expected_experts or len(set(choice)) != expected_experts:
            raise RuntimeError(f"invalid exact expert authority for {row['tokens']}")
        if any(not 0 <= expert < int(CONFIG["n_routed_experts"]) for expert in choice):
            raise RuntimeError(f"out-of-range expert authority for {row['tokens']}")
        if len(row["router_logits"][0]) != int(CONFIG["n_routed_experts"]):
            raise RuntimeError(f"router logits missing experts for {row['tokens']}")
        if len(row["expert_weights"][0]) != expected_experts:
            raise RuntimeError(f"router weights missing selected experts for {row['tokens']}")
        pattern = tuple(tuple(layer) for layer in row["expert_choices"])
        patterns.add(pattern)
        experts.update(choice)
    if len(patterns) < 2:
        raise RuntimeError("router authorities are vacuous: every prompt chose the same experts")
    return {
        "selection_patterns": len(patterns),
        "selected_experts": sorted(experts),
    }


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    r4 = next(rung for rung in fixture.pack.LADDER if rung["rung"] == "R4")

    # ``fixture.build`` deliberately selects the entry named R0. Give that
    # local process slot the R4 physical geometry and force matrix admission;
    # the large attention row windows then retain card256/bits8 exactly.
    # Force the fitter to one CPU thread: the normal MPS index_add path is
    # intentionally throughput-oriented and its atomic centroid accumulation
    # is not bit-deterministic enough for a frozen FP64 parity authority.
    torch = fixture.forge._torch()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fixture.forge._device = lambda: torch.device("cpu")
    fixture.pack.LADDER = [{**r4, "rung": "R0"}]
    fixture.pack.rung_is_admissible = lambda _rung, _elements: True
    fixture.CONFIG = dict(CONFIG)
    fixture.TOKENS = list(TOKENS)
    meta = fixture.build(out_dir)

    artifact = out_dir / meta["artifact"]
    header = gravity_format.read_header(artifact)
    direct_u8_names = _validate_direct_u8_payloads(artifact, header)
    weight_map = {entry["name"]: artifact.name for entry in header["tensors"]}
    index = {
        "schema": "hawking.gravity.model_index.v1",
        "model": header["model"],
        "architecture": CONFIG,
        "shards": [artifact.name],
        "shard_count": 1,
        "tensor_count": len(weight_map),
        "weight_map": weight_map,
    }
    (out_dir / "model.gravity.index.json").write_text(json.dumps(index, indent=1) + "\n")

    reference = _fp64_reference_module()
    weights = source_mod.GravityGlmSource(out_dir, single_shard=artifact.name)
    authority_weights = _materialize_fp64_authority_weights(
        weights, sorted(weight_map)
    )
    authorities = []
    for tokens in _prompts():
        logits, _, trace = reference.main_forward(
            np.asarray([tokens], dtype=np.int64),
            authority_weights,
            CONFIG,
        )
        authorities.append(_authority_record(tokens, logits, trace))
    authority_evidence = _validate_authorities(authorities)
    (out_dir / "ref_logits_f64.json").write_text(
        json.dumps(authorities, sort_keys=True, separators=(",", ":")) + "\n"
    )

    receipt = {
        **meta,
        "purpose": "bounded compact MLA + sparse MLP complete-token Numeric Parity V2.1",
        "production_artifact": False,
        "runtime_default_enabled": False,
        "layers": 1,
        "mlp_schedule": list(CONFIG["mlp_layer_types"]),
        "determinism": {
            "weight_seed": int(fixture.SEED),
            "pq_seed": 0,
            "fitter": "CPU",
            "torch_threads": 1,
            "authority_prompt_order": _prompts(),
            "fp64_weight_source": (
                "physical artifact tensors decoded once, exactly widened, "
                "then accumulated in explicit ascending f64 order"
            ),
        },
        "physical_attention_codec": dict(DIRECT_U8_GEOMETRY),
        "physical_routed_expert_codec": {
            **DIRECT_U8_GEOMETRY,
            "routed_experts": int(CONFIG["n_routed_experts"]),
            "shared_experts": int(CONFIG["n_shared_experts"]),
            "projection_tensors": 3
            * (int(CONFIG["n_routed_experts"]) + int(CONFIG["n_shared_experts"])),
        },
        "direct_u8_validation": {
            "status": "PASS",
            "tensors": direct_u8_names,
            "validated_tensors": len(direct_u8_names),
        },
        "fp64_complete_token_authority": {
            "prompts": len(authorities),
            "vocabulary_logits_per_prompt": int(CONFIG["vocab_size"]),
            "exact_dsa_decisions": True,
            "exact_router_expert_decisions": True,
            **authority_evidence,
        },
    }
    (out_dir / "compact_mla_fixture_receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n"
    )
    return receipt


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} OUT_DIR")
    print(json.dumps(build(Path(sys.argv[1])), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
