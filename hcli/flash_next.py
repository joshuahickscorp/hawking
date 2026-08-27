"""Pinned identity and staging plan for Qwen3.8-Flash-Next.

This module intentionally does not download weights.  It records the exact
upstream revision, the expected ModelLake acquisition, and the organ worklist
so a later acquisition can be resumed and verified without treating an
un-staged model as a qualified resident.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .persist import atomic_write_json
from .providers import CapabilityContract, ResidentProfile


REPO_ID = "Qwen/Qwen3.8-Flash-Next"
REQUESTED_REVISION = "main"
PINNED_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"
MODEL_PAGE = f"https://huggingface.co/{REPO_ID}"
MODEL_TREE = f"{MODEL_PAGE}/tree/{PINNED_REVISION}"
EXPECTED_BYTES = 360_023_286_454
EXPECTED_SAFETENSOR_SHARDS = 131
EXPECTED_FILE_COUNT = 144
SCHEMA = "hcli.flash-next.identity.v1"
PROMOTION_SCHEMA = "hcli.flash-next.promotion.v1"
TEXT_RESIDENT_CONTRACT = "FLASH_NEXT_TEXT_RESIDENT"
COMPLETE_SYSTEM_EBPW_MAX = 1.00
ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN = 50.0

# These are accounting categories, not a license to omit an implementation
# detail.  A final receipt must either carry each category in its byte ledger or
# be refused by the promotion gate.  Keeping the list here makes the denominator
# a product contract rather than a number a candidate can redefine.
COMPLETE_SYSTEM_BYTE_FIELDS = (
    "weight_codes",
    "scales",
    "zero_points",
    "bases",
    "residuals",
    "dictionaries",
    "expert_indices",
    "routing_metadata",
    "generators",
    "lookup_structures",
    "ngram_representation",
    "mtp_representation",
    "required_executable_metadata",
)

ORGANS = (
    {"id": "ngram_embedding", "status": "IDENTITY_ONLY", "target": "20M ngram vocabulary, 3-gram, 128-way split", "source_label": "DERIVED"},
    {"id": "moe_router", "status": "IDENTITY_ONLY", "target": "512 experts, top-10 per token, sigmoid output gate", "source_label": "DERIVED"},
    {"id": "moe_shared_expert", "status": "IDENTITY_ONLY", "target": "shared expert path", "source_label": "DERIVED"},
    {"id": "deltanet", "status": "IDENTITY_ONLY", "target": "36 linear-attention layers, 12 full-attention layers, conv-4 state path", "source_label": "DERIVED"},
    {"id": "qsa_sparse_attention", "status": "IDENTITY_ONLY", "target": "2048 indexer budget, 4 indexer heads, 24 attention heads", "source_label": "DERIVED"},
    {"id": "mtp", "status": "IDENTITY_ONLY", "target": "one hybrid MTP hidden layer", "source_label": "DERIVED"},
    {"id": "lm_head", "status": "IDENTITY_ONLY", "target": "output projection / vocabulary head", "source_label": "DERIVED"},
)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def evaluate_flash_promotion(candidate: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Apply the hard Flash-Next joint promotion law to a candidate receipt.

    The evaluator is intentionally strict and side-effect free.  Missing
    evidence is not treated as zero bytes, zero fallbacks, or a passing
    capability score.  A candidate with measurements but a failed threshold is
    retained as a Pareto-frontier intermediate, never promoted as final.
    """
    body = dict(candidate or {})
    metrics = body.get("metrics")
    metrics = dict(metrics) if isinstance(metrics, Mapping) else {}
    ebpw = _number(body.get("complete_system_ebpw", metrics.get("complete_system_ebpw")))
    accepted_tps = _number(
        body.get(
            "accepted_capability_preserving_tps",
            metrics.get("accepted_capability_preserving_tps"),
        )
    )
    hard_gate = {
        "complete_system_ebpw_at_or_below_1": ebpw is not None and ebpw <= COMPLETE_SYSTEM_EBPW_MAX,
        "accepted_capability_preserving_tps_at_or_above_50": (
            accepted_tps is not None and accepted_tps >= ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN
        ),
    }

    contract = str(body.get("artifact_contract") or "FLASH_NEXT_COMPLETE_MULTIMODAL")
    accounting = body.get("ebpw_accounting")
    accounting = dict(accounting) if isinstance(accounting, Mapping) else {}
    missing: list[str] = []
    if not accounting:
        missing.append("complete-system byte ledger")
    for field in COMPLETE_SYSTEM_BYTE_FIELDS:
        value = accounting.get(field)
        if not isinstance(value, Mapping) or _number(value.get("bytes")) is None:
            missing.append(f"ebpw_accounting.{field}.bytes")
    if accounting.get("all_required_bytes_included") is not True:
        missing.append("ebpw_accounting.all_required_bytes_included=true")
    excluded = accounting.get("excluded_bytes")
    if excluded:
        allowed_text_exception = (
            contract == TEXT_RESIDENT_CONTRACT
            and isinstance(excluded, list)
            and all("vision" in str(item).lower() for item in excluded)
        )
        if not allowed_text_exception:
            missing.append("no unqualified byte exclusions")

    evidence_requirements = {
        "native_executable": "native executable",
        "dense_parent_execution_fallback": "dense-parent fallback declaration",
        "hidden_dense_rematerialization": "hidden dense-rematerialization declaration",
        "capability_contract_passed": "protected capability contract",
        "dense_vs_nf_ab": "dense-versus-NF A/B receipt",
        "whole_model_reference_comparison": "whole-model reference comparison",
        "runtime_kernel_genome_complete": "runtime/kernel genome",
        "mtp_accounting_disclosed": "MTP accounting",
        "reproducible_protected_receipt": "reproducible protected receipt",
    }
    evidence: Dict[str, bool] = {}
    for key, label in evidence_requirements.items():
        value = body.get(key)
        if key in {"dense_parent_execution_fallback", "hidden_dense_rematerialization"}:
            passed = value is False
        else:
            passed = value is True
        evidence[key] = passed
        if not passed:
            missing.append(label)

    fallback_count = body.get("fallback_count")
    fallback_ok = isinstance(fallback_count, (int, float)) and not isinstance(fallback_count, bool)
    if not fallback_ok:
        missing.append("fallback_count disclosure")
    elif int(fallback_count) != 0:
        missing.append("fallback_count=0")

    thresholds_known = ebpw is not None and accepted_tps is not None
    promotable = thresholds_known and all(hard_gate.values()) and not missing
    if promotable:
        status = "PROMOTABLE"
    elif thresholds_known:
        status = "PARETO_FRONTIER_NOT_FINAL"
    else:
        status = "INCOMPLETE"
    return {
        "schema": PROMOTION_SCHEMA,
        "status": status,
        "artifact_contract": contract,
        "hard_gate": hard_gate,
        "measured": {
            "complete_system_ebpw": ebpw,
            "accepted_capability_preserving_tps": accepted_tps,
        },
        "evidence": evidence,
        "missing_or_refused": sorted(set(missing)),
        "promotion_allowed": promotable,
        "ladder": {
            "final": {"complete_system_ebpw_max": COMPLETE_SYSTEM_EBPW_MAX, "accepted_tps_min": ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN},
            "strong": {"complete_system_ebpw_max": 0.75, "accepted_tps_min": 70.0},
            "exceptional": {"complete_system_ebpw_max": 0.60, "accepted_tps_min": 90.0},
            "extreme": {"complete_system_ebpw_max": 0.50, "accepted_tps_min": 120.0},
        },
        "claim_boundary": (
            "Only PROMOTABLE closes the final Flash-Next gate. A measured candidate "
            "that misses either hard threshold remains a research/Pareto intermediate."
        ),
    }


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _local_observation(root: Optional[Path]) -> Dict[str, Any]:
    if root is None:
        return {"status": "NOT_STAGED", "reason": "set HCLI_FLASH_NEXT_ROOT to inspect a local acquisition"}
    root = root.expanduser().resolve()
    if not root.is_dir():
        return {"status": "NOT_STAGED", "path": str(root), "reason": "configured root does not exist"}
    # Do not recursively walk a potentially 360 GB acquisition just to answer
    # an identity command.  Direct children are enough to detect presence; the
    # ModelLake census owns bounded inventory and hash work.
    total = 0
    files = 0
    directories = 0
    entries_truncated = False
    try:
        with os.scandir(root) as entries:
            for index, entry in enumerate(entries):
                if index >= 4096:
                    entries_truncated = True
                    break
                try:
                    if entry.is_file(follow_symlinks=False):
                        files += 1
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        directories += 1
                except OSError:
                    continue
    except OSError:
        pass
    config = root / "config.json"
    config_data: Dict[str, Any] = {}
    if config.is_file():
        try:
            loaded = json.loads(config.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config_data = loaded
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    return {
        "status": "PRESENT_UNVERIFIED",
        "path": str(root),
        "bytes": total,
        "file_count": files,
        "directory_count": directories,
        "bytes_scope": "direct_children_only",
        "entries_truncated": entries_truncated,
        "config_sha256": _sha256(config) if config.is_file() else None,
        "config_model_type": config_data.get("model_type"),
        "revision_verified": False,
        "weights_verified": False,
    }


def model_lake_plan() -> Dict[str, Any]:
    root = Path("/Volumes/corpdrive/hawking-modellake")
    mounted = root.is_dir()
    free = None
    if mounted:
        try:
            free = shutil.disk_usage(root).free
        except OSError:
            free = None
    return {
        "status": "READY_FOR_EXPLICIT_ACQUIRE" if mounted and (free is None or free > EXPECTED_BYTES) else "REFUSED_NOT_READY",
        "root": str(root),
        "mounted": mounted,
        "expected_bytes": EXPECTED_BYTES,
        "expected_file_count": EXPECTED_FILE_COUNT,
        "free_bytes_observed": free,
        "tier": 2,
        "resumable": True,
        "hash_verified": False,
        "hash_status": "NOT_RUN",
        "hash_policy": "verify every acquired shard before atomic publish",
        "atomic_publish": True,
        "command": [
            "hf", "download", REPO_ID, "--revision", PINNED_REVISION,
            "--local-dir", "<ModelLake partial destination>",
        ],
        "human_confirmation_required": True,
        "download_performed": False,
        "reason": "weights are not downloaded by identity census",
    }


def flash_next_profile(local_root: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    root_value = local_root or os.environ.get("HCLI_FLASH_NEXT_ROOT")
    local = _local_observation(Path(root_value) if root_value else None)
    architecture = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "hidden_size": 2560,
        "layers": 48,
        "experts": 512,
        "routed_experts_per_token": 10,
        "vocab_size": 248320,
        "ngram_vocab_size": 20000000,
        "ngram_size": 3,
        "split_ngram_parts": 128,
        "indexer_budget": 2048,
        "indexer_heads": 4,
        "full_attention_layers": 12,
        "linear_attention_layers": 36,
        "mtp_num_hidden_layers": 1,
    }
    architecture_fingerprint = hashlib.sha256(
        json.dumps(architecture, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    pre_runtime_science = {
        "labels": {
            "upstream_source": "UPSTREAM_SOURCE",
            "derived_architecture": "DERIVED",
            "local_physical_measurement": "LOCAL_PHYSICAL",
        },
        "inputs": [
            {"name": "config.json", "source": MODEL_PAGE, "revision": PINNED_REVISION, "label": "UPSTREAM_SOURCE"},
            {"name": "tokenizer.json", "source": MODEL_PAGE, "revision": PINNED_REVISION, "label": "UPSTREAM_SOURCE"},
            {"name": "model.safetensors.index.json", "source": MODEL_PAGE, "revision": PINNED_REVISION, "label": "UPSTREAM_SOURCE"},
            {"name": "README.md", "source": MODEL_PAGE, "revision": PINNED_REVISION, "label": "UPSTREAM_SOURCE"},
            {"name": "architecture fingerprint", "source": "pinned config/index interpretation", "label": "DERIVED"},
        ],
        "architecture_fingerprint": {
            "value": architecture_fingerprint,
            "algorithm": "sha256(canonical architecture metadata)",
            "metadata": architecture,
            "label": "DERIVED",
        },
        "organ_graph": [
            {
                **organ,
                "dimensions": "UNRESOLVED_UNTIL_INDEX_INSPECTION",
                "tensor_bytes": None,
                "active_bytes_per_token": None,
                "flops_per_token": None,
                "state_bytes": None,
                "regularity": "UNMEASURED",
                "bottleneck": "UNRESOLVED",
                "label": "DERIVED",
            }
            for organ in ORGANS
        ],
        "active_compute_bounds": {
            "layers": {"value": 48, "label": "DERIVED"},
            "routed_experts_per_token": {"value": 10, "label": "DERIVED"},
            "total_experts": {"value": 512, "label": "DERIVED"},
            "routed_expert_fraction": {"value": 10 / 512, "label": "DERIVED", "caveat": "excludes shared expert and non-expert path"},
            "active_bytes_per_token": {"value": None, "label": "LOCAL_PHYSICAL", "status": "NOT_MEASURED"},
            "device_bytes_touched_per_token": {"value": None, "label": "LOCAL_PHYSICAL", "status": "NOT_MEASURED"},
            "flops_per_token": {"value": None, "label": "LOCAL_PHYSICAL", "status": "NOT_MEASURED"},
        },
        "gravity_plan": [
            {"organ": "moe_experts", "representation": "cross-expert shared basis plus residual", "status": "PLAN_ONLY", "native_kernel_required": True},
            {"organ": "ngram_embedding", "representation": "factorized/lookup/generative ngram structure", "status": "PLAN_ONLY", "native_kernel_required": True},
            {"organ": "deltanet", "representation": "resident recurrent state machine", "status": "PLAN_ONLY", "native_kernel_required": True},
            {"organ": "qsa_sparse_attention", "representation": "2048-budget sparse block-native attention", "status": "PLAN_ONLY", "native_kernel_required": True},
            {"organ": "mtp", "representation": "accepted-draft accounting, not omitted work", "status": "PLAN_ONLY", "native_kernel_required": True},
        ],
        "required_primitives": [
            "representation-native decode",
            "routing/index lookup",
            "resident recurrent-state update",
            "sparse attention block traversal",
            "MTP accept/reject accounting",
            "complete-token wall instrumentation",
        ],
    }
    profile = ResidentProfile(
        profile_id=f"hf:{REPO_ID}@{PINNED_REVISION[:12]}",
        provider="huggingface",
        model_id=REPO_ID,
        artifact={
            "source_repo": REPO_ID,
            "requested_revision": REQUESTED_REVISION,
            "pinned_revision": PINNED_REVISION,
            "format": "transformers",
            "expected_bytes": EXPECTED_BYTES,
            "expected_file_count": EXPECTED_FILE_COUNT,
            "expected_safetensor_shards": EXPECTED_SAFETENSOR_SHARDS,
            "local": local,
        },
        tokenizer={"status": "UPSTREAM_DECLARED", "source": MODEL_PAGE, "authoritative_after_acquire": True},
        runtime={
            "status": "NOT_COMPILED",
            "compatible_runtimes": ["transformers", "vllm", "sglang"],
            "native_hawking_resident": "NOT_IMPLEMENTED",
        },
        compiler={"status": "NOT_RUN", "required_before_promotion": True},
        representation={"status": "BF16_MODEL_LAKE_TARGET", "organs": list(ORGANS)},
        capabilities=CapabilityContract.from_mapping({
            "schema": "hcli.provider.capabilities.v1",
            "features": {
                "vision": {"state": "unknown", "enforcement": "unknown", "source": "identity only"},
                "tool_calling": {"state": "unknown", "enforcement": "unknown", "source": "identity only"},
                "streaming": {"state": "unknown", "enforcement": "unknown", "source": "identity only"},
            },
        }),
        prompt_contract={"status": "UPSTREAM_CHAT_TEMPLATE_NOT_EXECUTED", "source": MODEL_PAGE},
        generation={"status": "UPSTREAM_GENERATION_CONFIG_NOT_EXECUTED", "source": MODEL_PAGE},
        limits={"context_length": 262144, "native_extensible_context_length": 1000000},
        fallbacks=["no local weights", "no native resident", "no capability qualification"],
        hot_bytes=None,
        machine_genome={},
        receipts=[],
        qualification={"status": "IDENTITY_ONLY", "promotion_allowed": False, "reason": "no local weights or runtime qualification"},
        metadata={
            "model_page": MODEL_PAGE,
            "model_tree": MODEL_TREE,
            "architecture": architecture,
            "observed_at": time.time(),
        },
    )
    return {
        "schema": SCHEMA,
        "profile": profile.to_dict(),
        "model_lake": model_lake_plan(),
        "transfer_ledger": {"source": MODEL_PAGE, "revision": PINNED_REVISION, "bytes_transferred": 0, "verified_bytes": 0},
        "organ_census": list(ORGANS),
        "pre_runtime_science": pre_runtime_science,
        "promotion_gate": evaluate_flash_promotion(None),
        "download_performed": False,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = flash_next_profile(args.local_root)
    if args.emit:
        atomic_write_json(Path(args.emit).expanduser(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_BYTES",
    "EXPECTED_FILE_COUNT",
    "EXPECTED_SAFETENSOR_SHARDS",
    "MODEL_PAGE",
    "ORGANS",
    "PINNED_REVISION",
    "REPO_ID",
    "ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN",
    "COMPLETE_SYSTEM_EBPW_MAX",
    "COMPLETE_SYSTEM_BYTE_FIELDS",
    "PROMOTION_SCHEMA",
    "TEXT_RESIDENT_CONTRACT",
    "evaluate_flash_promotion",
    "flash_next_profile",
    "main",
    "model_lake_plan",
]
