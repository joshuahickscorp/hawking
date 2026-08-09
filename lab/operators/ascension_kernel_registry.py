"""V3 compiler/Gravity contract for every required kernel family.

The registry is intentionally configuration-only.  It is the durable, exact
list of semantic plugins, shared primitives, model-program keys, Gravity
components, and representation classes that a later measured worker must
cover.  It never calls an existing generic kernel a family qualification and
it never treats a declared plugin as compiled or fast.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.receipts import seal


SCHEMA = "hawking.ascension.v3_kernel_compiler_contract.v1"
FILENAME = "ASCENSION_V3_KERNEL_COMPILER_CONTRACT.json"

SHARED_PRIMITIVES: tuple[str, ...] = (
    "packed_load_decode",
    "scale_codebook_application",
    "dot_reduction",
    "norm",
    "position_operations",
    "routing",
    "top_k",
    "expert_gather",
    "projection_waves",
    "activation",
    "residual",
    "kv_state",
    "sampling",
    "continuous_batching",
    "receipts",
)

FAMILY_PLUGINS: tuple[str, ...] = (
    "QWEN_DENSE",
    "QWEN_MOE",
    "QWEN_NEXT_HYBRID",
    "LLAMA_GQA",
    "MISTRAL_SLIDING",
    "MIXTRAL_MOE",
    "DEEPSEEK_MLA_MOE",
    "DEEPSEEK_V4_OR_DISCOVERED_HYBRID",
    "GLM_MOE_INDEXED_OR_DISCOVERED_VARIANTS",
    "KIMI_RELEASED_ARCHITECTURE_PLUGINS",
    "GEMMA_FAMILY",
    "STATE_SPACE_LINEAR_ATTENTION_HYBRID",
    "GENERIC_HF_REFERENCE",
)

MODEL_PROGRAM_KEY_FIELDS: tuple[str, ...] = (
    "source_revision",
    "artifact_hash",
    "architecture_fingerprint",
    "gravity_plan",
    "device",
    "context_regime",
    "session_regime",
)

ARCHITECTURE_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "dense_or_moe",
    "layer_schedule",
    "hidden_dimensions",
    "attention_state_operators",
    "position_encoding",
    "router_topology",
    "expert_count_top_k",
    "shared_experts",
    "activation_norm",
    "kv_state",
    "native_dtype",
    "tensor_geometry",
    "tokenizer_template",
    "context",
)

GRAVITY_COMPONENTS: tuple[str, ...] = (
    "GRAVITY_CONTROLLER",
    "FORGE",
    "DOCTOR",
    "QAT_ENGINE",
    "KERNEL_COMPILER",
    "SCHEDULER_COMPILER",
    "EVENT_HORIZON",
    "GRAVEYARD",
)

REPRESENTATION_TOURNAMENT_CLASSES: tuple[str, ...] = (
    "native_mixed_precision",
    "organ_aware_q3",
    "special_down_projection",
    "activation_correction",
    "trellis",
    "lattice_incoherence",
    "additive_multi_codebook",
    "binary_ternary",
    "shared_basis",
    "joint_expert_decomposition",
    "low_rank_sparse_residual",
    "protected_outliers",
    "route_group_codebooks",
    "doctor_correction",
    "qat_recovery",
    "kv_state_compression",
    "active_expert_reduction",
    "conditional_depth",
    "functional_expert_replacement",
    "functional_attention_replacement",
    "shared_recurrent_block",
)

FAMILY_STARTING_DOCTRINES: dict[str, tuple[str, ...]] = {
    "QWEN": ("reuse_manager_kernel_evidence", "moe_hybrid_plugins", "expert_shared_expert_representations", "deltanet_state_for_next"),
    "LLAMA": ("exact_tokenizer_bos_rope", "dense_gqa", "reference_parity", "generated_dense_kernels"),
    "MISTRAL_MIXTRAL": ("sliding_window_semantics", "gqa", "moe_routing", "shared_expert_where_present"),
    "DEEPSEEK": ("mla_latent_attention", "moe", "dsa_indexed_variants", "mhc_state", "native_fp4_fp8_when_required", "proto_frankenstein_later"),
    "GLM": ("moe", "dsa", "indexshare_indexed_attention", "long_context", "mtp_control", "streamed_source"),
    "KIMI": ("bind_released_architecture_live", "large_moe", "delta_latent_sparse_attention_when_present", "long_context", "streamed_source"),
    "GEMMA": ("dense_portability", "attention_variants", "exact_model_codegen"),
    "STATE_SPACE_OR_LINEAR_ATTENTION_HYBRID": ("deltanet", "linear_attention", "recurrent_state", "state_checkpoint_restart"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def kernel_compiler_contract(*, bible_sha256: str | None = None) -> dict[str, Any]:
    """Create a sealed configuration contract, not a compiler success receipt."""

    return seal(
        {
            "schema": SCHEMA,
            "status": "CONTROLLER_CONFIGURATION_ONLY",
            "recorded_at": _utc_now(),
            "bible_sha256": bible_sha256,
            "architecture_pipeline": [
                "FINGERPRINT",
                "OPERATOR_GRAPH",
                "FAMILY_PLUGIN",
                "MISSING_OPERATORS",
                "GRAVITY_SEARCH_SPACE",
                "CODEGEN_PLAN",
            ],
            "architecture_fingerprint_fields": list(ARCHITECTURE_FINGERPRINT_FIELDS),
            "shared_primitives": list(SHARED_PRIMITIVES),
            "required_family_plugins": list(FAMILY_PLUGINS),
            "model_program_key_fields": list(MODEL_PROGRAM_KEY_FIELDS),
            "gravity_components": list(GRAVITY_COMPONENTS),
            "representation_tournament_classes": list(REPRESENTATION_TOURNAMENT_CLASSES),
            "family_starting_doctrines": {
                key: list(value) for key, value in FAMILY_STARTING_DOCTRINES.items()
            },
            "model_program_codegen": [
                "loop_bounds",
                "offsets",
                "expert_groups",
                "top_k",
                "quant_grammar",
                "scale_grouping",
                "tile_choices",
                "argument_tables",
                "command_graph",
                "prefetch_plan",
                "residency_plan",
                "state_layout",
            ],
            "claim_boundary": {
                "declaration_is_not_compilation": True,
                "declaration_is_not_exact_model_qualification": True,
                "generic_hf_reference_does_not_replace_a_family_plugin": True,
                "measured_parity_capability_complete_token_p99_and_rollback_remain_required": True,
            },
        }
    )


def write_kernel_compiler_contract(
    root: str | Path, *, bible_sha256: str | None = None
) -> dict[str, Any]:
    """Write the controller-owned full registry alongside continuation files."""

    destination = Path(root).expanduser().resolve() / FILENAME
    document = kernel_compiler_contract(bible_sha256=bible_sha256)
    _atomic_json(destination, document)
    return document


__all__ = [
    "ARCHITECTURE_FINGERPRINT_FIELDS",
    "FAMILY_PLUGINS",
    "FAMILY_STARTING_DOCTRINES",
    "FILENAME",
    "GRAVITY_COMPONENTS",
    "MODEL_PROGRAM_KEY_FIELDS",
    "REPRESENTATION_TOURNAMENT_CLASSES",
    "SCHEMA",
    "SHARED_PRIMITIVES",
    "kernel_compiler_contract",
    "write_kernel_compiler_contract",
]
