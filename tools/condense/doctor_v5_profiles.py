#!/usr/bin/env python3.12
"""Canonical profiles, replay semantics, and Git-backed Doctor V5 archive."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Sequence

import condense_common as common


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_COMMIT = "1c525380204d61beb8b570516576c5a683d73595"
SCHEMA = "hawking.doctor_v5_condensed_profiles.v1"
REPLAY_SCHEMA = "hawking.doctor_v5_condensed_replay.v1"
TERMINAL = frozenset({"complete", "negative", "unsupported"})
ALLOWED_STATUS = TERMINAL | {"pending", "running", "failed", "resource-stop"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_RE = re.compile(rb"hawking\.doctor_v5_[A-Za-z0-9_.-]+\.v[0-9]+")
REPORT_RE = re.compile(rb"reports/condense/doctor_v5_[A-Za-z0-9_./-]+")
SUBCOMMAND_RE = re.compile(rb"add_parser\([\"']([^\"']+)")
DEFAULT_HANDOFF = (
    ROOT / "reports/condense/doctor_v5_unbound/post120_acceleration/handoff.json"
)
POST120_HANDOFF_SCHEMA = "hawking.doctor_v5_post120_acceleration_handoff.v1"
POST120_BINDINGS = {
    "gptoss_work_plan": "work_plan_sha256",
    "gptoss_pending_wiring": "pending_wiring_sha256",
    "gptoss_reuse_fanout": "fanout_plan_sha256",
    "shared_preprocess_requirements": "requirements_sha256",
    "shared_preprocess_manifest": "manifest_sha256",
    "shared_preprocess_plan": "cache_plan_sha256",
    "tokenizer_binding": "tokenizer_binding_sha256",
    "tokenizer_gate": "gate_sha256",
    "higher_tier_requirements": "requirements_sha256",
    "acceleration_requirements": "requirements_sha256",
    "named_horizons": "horizon_scaffold_sha256",
    "gptoss_acceleration_plan": "acceleration_plan_sha256",
    "physical_ab_plan": "plan_sha256",
}
POST120_FACETS = (
    "rate_thread_profiles",
    "block_parallelism",
    "ordered_phase_overlap",
    "bounded_preprocess_reuse",
    "ram_lane_packing",
    "controlled_swap",
    "disk_lifecycle_gc",
    "native_pgo",
    "metal_preprocess",
    "exact_quality_receipts",
    "rollback_cas",
)
PHYSICAL_AB_FACETS = (
    "release_authority",
    "thread_profiles",
    "block_parallel",
    "ordered_overlap",
    "bounded_reuse",
    "ram_swap_recovery",
    "native_io_pgo",
    "disk_lifecycle",
    "full_stack_parity_ab",
    "post120_appendix_bindings",
)


# name: (profile, canonical replacement, baseline source SHA-256)
ARCHIVED: dict[str, tuple[str, str, str]] = {
    "doctor_v5_acceleration_eta": ("methodology", "doctor.report", "d49b3573f534653aa77182d725b165c066f19a5b905563861b58e89293a10ae7"),
    "doctor_v5_acceleration_reentry": ("recovery", "doctor.recovery", "3d665221ac2379816a0b7beb5e770c721459f3d813ed91bbbcbd8b51a0f1b2af"),
    "doctor_v5_aggressive_admission_policy": ("admission", "doctor.admission", "d9e11003f630c4dbbf0ae68ea4395ae9160b0240d0ba936e01c66ba0995de745"),
    "doctor_v5_audit": ("reporting", "doctor.report", "8ce808e3363b66ff11019bfa793617a4bd898df9e5b67e8a84af0a8be4a54dca"),
    "doctor_v5_block_parallel_config_matrix": ("block_parallel", "doctor.block-parallel", "1f46930330c045ab35f7e942885e67ee455f563fe4af484045849c6a41f36111"),
    "doctor_v5_block_parallel_real_canary": ("block_parallel", "doctor.block-parallel", "c9dc632468e000b6cf4dfa4458623e1404095994350e9f86e7eacf1f628757f0"),
    "doctor_v5_blocked_cell_recovery": ("recovery", "doctor.recovery", "f60bec33b917ff9d2290703a67380a22593161e1df3cb38f2793f73ecc9d5db6"),
    "doctor_v5_condenser_mountain_methodology": ("methodology", "doctor.methodology", "a1222d48eef75ded4881c3dbbcf6bd7aa0f5f92475ef7fc314afd80db5ea3f68"),
    "doctor_v5_distributed_transport": ("post120", "doctor.post120", "d4328c0445b163082263971d8b8c6cd3c8e0c22b80a25b5c617d978db83ef3fa"),
    "doctor_v5_elastic_phase_scheduler": ("admission", "doctor.admission", "70f409ff1582f46935e020ab7c1389a8f922ddb03c256dc33a8a0a6a3c7017d3"),
    "doctor_v5_fixture_phase_validator": ("admission", "doctor.admission", "8741b4bd3ce9d002a401feee56b84849108fa3e41be6bd0250d8fab31b05b503"),
    "doctor_v5_forward_recovery": ("recovery", "doctor.recovery", "f31658c902251eba8c70fb43c8254562cafbffad41523a7afdc372125b85e428"),
    "doctor_v5_gptoss_execution_thread_contract": ("post120", "doctor.post120", "622357978e0aa916759fef32311c31e3f5a0544d0590e728521ac8cc8dd5f8c9"),
    "doctor_v5_gptoss_parallel_scaffold": ("post120", "doctor.post120", "5acd24944dcee8f03107394cbe59f8b6e1197f1987d98bc036d00212e2a8647e"),
    "doctor_v5_gptoss_reuse_fanout": ("post120", "doctor.post120", "c7e728818ab62e4207cd3735d0c4d733a31c7fa9a0d1c6769e218666946614e6"),
    "doctor_v5_gptoss_tokenizer_gate": ("post120", "doctor.post120", "e3a6aac63fc8738d7fec430b12c527fa87e8654e1a43d12cad6dd1fdaea5f949"),
    "doctor_v5_higher_tier_authority": ("post120", "doctor.post120", "dec19a057e8867b34423f3752e85f77c2bd2b64659f5bdf3d8b20ce57e5eeb40"),
    "doctor_v5_higher_tier_scaffold": ("post120", "doctor.post120", "ecffa8759fb785b399540efcd0e1909c83f737bad72a528210572877fa6cbe12"),
    "doctor_v5_host_sprint_plan": ("admission", "doctor.admission", "912d4cd744ea0357069d457a914181ff1afed8b9f32fef6ed320beb5ad025e83"),
    "doctor_v5_inert_phase_launcher": ("admission", "doctor.admission", "644bfd089e3f8e87e45216259ddfa16deeded013b0e769c993c06001c576b7ec"),
    "doctor_v5_mountain_ladder": ("methodology", "doctor.methodology", "78aa8902d765d00e431cc110ab124cbcdb115cff449cd9d5176a3e6bfbbb7d15"),
    "doctor_v5_pass_b_bootstrap": ("supervisor", "doctor.supervisor", "2a02e850f75b2f9cf145406041ed0e6d64f314f198c20486b47ba25f83cf39df"),
    "doctor_v5_pass_b_queue": ("supervisor", "doctor.supervisor", "ae405cd7b2e804367a7288af5bd874f3f38a5de7e544873ccb4d513c0e6b9386"),
    "doctor_v5_post120_acceleration_scaffold": ("post120", "doctor.post120", "da370c6a039c38be4926cb34626ea2f3bd27e3bce287ed6ca2ce96b2a10f65c5"),
    "doctor_v5_production_eta": ("methodology", "doctor.report", "1421c9ae118006928c7cd59812424bb71d98fedbaf39db43f2688a1b433baa03"),
    "doctor_v5_queue": ("supervisor", "doctor.supervisor", "0b65e36279e4db4635d37e63ab3b1b35fffac378ce7cdd4efcde6eebfab1d7a4"),
    "doctor_v5_qwen_shard_window": ("admission", "doctor.admission", "cd46bb05500cad073c5d6c3ef5bc9ee1ff816123be0d18ff1488ecc15ffc26fc"),
    "doctor_v5_qwen_thread_profile_runner": ("admission", "doctor.admission", "6ed08cb57268b8bd9f61413a89bd73b02ac34aa3b46a5d2b4365cc4510b9cc4b"),
    "doctor_v5_remaining_scratch_gate_adapter": ("admission", "doctor.admission", "9fb2379d1db6d26e7628e3bd3591440830b8551196d43c7b3d5354b23ffaf6d8"),
    "doctor_v5_resource_stop_recovery_stage": ("recovery", "doctor.recovery", "c7aeebdc6db9b17d2450c4ac1c91f897dc08b419e15bd528cfb1cd81f51744cb"),
    "doctor_v5_root": ("supervisor", "doctor.supervisor", "6bc3620223ecce84bd109e9dff61ef7ac041d44fecbc4a3c7e49c7546e604547"),
    "doctor_v5_shared_preprocess_cache": ("post120", "doctor.post120", "2bdf39a188409c2b3ed7374084d93fec2bb483c36afd7e6c896136a175021437"),
    "doctor_v5_single_device_benchmark": ("methodology", "doctor.methodology", "45eb2316689e04f973c47e3812bf483c91a592eb358a9fbead9dbd463f2ef5e8"),
    "doctor_v5_single_device_sprint_audit": ("methodology", "doctor.methodology", "37347b5aebf9051e071b85d92a003e7b1c7206e5a2b5aea5c2882525481db4df"),
    "doctor_v5_strand_control_adapter": ("supervisor", "doctor.supervisor", "b288b2433eb1fa15023d9848427a7a67e2650f335c5bb550bb09a88b4c6c0edd"),
    "doctor_v5_streaming_source": ("post120", "doctor.post120", "ba9b0c327b2d0475658b466d541d7fb5ff80c26ab70d5e1ef4310bcec85e91ee"),
    "doctor_v5_ultra_aggressive_autoresume": ("supervisor", "doctor.supervisor", "e18c3b11829f3dfcd3a1a87a82d7bfb20e3204c7175ad816f32650c2db314944"),
    "doctor_v5_ultra_aggressive_queue": ("supervisor", "doctor.supervisor", "371dcf43faf384862983d68fb5a4919fa15d49b79cab12a0a0357cfe87b9b063"),
    "doctor_v5_ultra_autoresume": ("supervisor", "doctor.supervisor", "f5d48608ae2842477b4914c35f5c43d61633825812bfbad3dbb2841585efade1"),
}


PROFILES: dict[str, dict[str, Any]] = {
    "supervisor": {
        "canonical_modules": [
            "doctor_v5_ultra_queue.py",
            "doctor_v5_ultra_accelerated_queue.py",
            "doctor_v5_ultra_accelerated_autoresume.py",
        ],
        "invariants": [
            "single durable queue owner",
            "atomic state/control/PID checkpoints",
            "exact adapter ABI and source seals",
            "resume never changes completed results",
            "resource stop is restartable, not terminal",
        ],
        "outputs": ["reports/condense/doctor_v5_ultra"],
    },
    "admission": {
        "canonical_modules": [
            "doctor_v5_stacked_admission.py",
            "doctor_v5_accelerated_resource_policy.py",
            "doctor_v5_phase_aware_disk_gate.py",
            "doctor_v5_remaining_scratch_ledger.py",
        ],
        "thresholds": {
            "process_budget_bytes": 78_000_000_000,
            "global_reserve_bytes": 12_000_000_000,
            "rss_margin_floor_bytes": 2_000_000_000,
            "rss_margin_ratio": 0.15,
            "disk_reserve_bytes": 150_000_000_000,
            "swap_soft_growth_mb": 512.0,
            "swap_hard_growth_mb": 1536.0,
            "swap_emergency_growth_mb": 3072.0,
            "swap_absolute_emergency_mb": 4096.0,
            "cpu_budget_cores": 24,
            "thread_candidates": [8, 12, 16, 20],
            "max_lanes": 8,
        },
        "invariants": [
            "owner and process identities are authenticated",
            "admission is phase-aware and disk-reserve preserving",
            "swap growth sheds work before promotion",
            "synthetic evidence never gains production authority",
        ],
    },
    "recovery": {
        "canonical_modules": [
            "doctor_v5_disk25_successor.py",
            "doctor_v5_controlled_swap_activation.py",
            "doctor_v5_controlled_swap_successor.py",
            "doctor_v5_controlled_swap_autoresume.py",
        ],
        "terminal_states": sorted(TERMINAL),
        "thresholds": {
            "content_verify_attempts": 3,
            "swap_sample_minimum": 3,
            "swap_sample_interval_seconds": 5.0,
            "swap_proof_max_age_seconds": 120.0,
        },
        "invariants": [
            "WAL intent precedes every activation",
            "CAS recheck occurs beneath the held lease",
            "rollback restores byte-identical prior state",
            "incident-specific pins remain in Git history",
        ],
    },
    "reporting": {
        "canonical_modules": ["doctor_v5_campaign_report.py", "doctor_v5_census.py"],
        "invariants": [
            "result bytes remain immutable",
            "attempt history and negative outcomes remain visible",
            "aggregate status is derivable from receipts",
        ],
    },
    "methodology": {
        "canonical_modules": ["doctor_v5_profiles.py", "doctor_v5_campaign_report.py"],
        "runbook": "docs/OPERATIONS.md",
        "rates": ["4", "3", "2", "1", "0.8", "0.55", "0.5", "0.33", "0.25", "0.1"],
        "thresholds": {
            "mountain_parameter_pass_share": 0.10,
            "good_ppl_delta_max": 0.08,
            "good_capability_delta_min": -0.05,
            "appendix_seconds": [86_400, 259_200],
            "artifact_overhead_ppm": 80_000,
        },
        "invariants": [
            "quality precedes speed at low bpw",
            "ETA is evidence-bounded and explicitly provisional",
            "single-device projections never promote synthetic evidence",
        ],
    },
    "block_parallel": {
        "canonical_modules": [
            "doctor_v5_strand_ladder_block_parallel_adapter.py",
            "doctor_v5_strand_ladder_block_parallel_worker.py",
            "doctor_v5_qwen_treatment_block_parallel_adapter.py",
        ],
        "invariants": [
            "exact output parity is mandatory",
            "serial fallback remains available",
            "real-tensor canaries precede activation",
        ],
    },
    "post120": {
        "canonical_modules": [
            "doctor_v5_post_120b.py",
            "doctor_v5_gptoss_mxfp4.py",
            "doctor_v5_gptoss_moe_adapter.py",
        ],
        "model": {"label": "120B", "hf_id": "openai/gpt-oss-120b"},
        "rates": ["4", "3", "2", "1", "0.8", "0.55", "0.5", "0.33", "0.25", "0.1"],
        "branches": ["codec_control", "doctor_static", "doctor_conditional", "doctor_full"],
        "thresholds": {
            "max_threads": 24,
            "thread_candidates": [8, 12, 16, 20],
            "queue_depth_candidates": [2, 3, 4, 6],
            "layers": 36,
            "experts": 128,
            "experts_per_batch": 8,
            "expected_source_units": 615,
            "expected_output_units": 6150,
            "transport_chunk_bytes": 67_108_864,
            "shared_cache_unit_limit_bytes": 8_000_000_000,
            "shared_cache_max_active_units": 1,
        },
        "outputs": [
            "reports/condense/doctor_v5_unbound/gptoss_120b_parallel",
            "reports/condense/doctor_v5_unbound/post120_acceleration",
            "reports/condense/doctor_v5_unbound/shared_preprocess",
        ],
        "invariants": [
            "native MXFP4 inventory is source authority",
            "tokenizer revision and exact output receipts are bound",
            "fanout shares preprocessing without sharing mutable outputs",
            "remote transport requires chunk hashes, leases, and acceptance receipts",
        ],
    },
    "physical": {
        "canonical_modules": [
            "doctor_v5_local_observer.py",
            "doctor_v5_physical_ab_controller.py",
            "doctor_v5_physical_ab_executor.py",
            "doctor_v5_physical_adapter_registry.py",
            "doctor_v5_physical_counter_barrier.py",
            "doctor_v5_physical_result_authority.py",
        ],
        "invariants": [
            "default off until the signed release boundary",
            "counter and result authority remain separate",
            "same-artifact A/B execution is lease-bound",
        ],
    },
    "notification": {
        "canonical_modules": ["doctor_v5_telegram_rung_notifier.py"],
        "invariants": [
            "one notification per completed rung",
            "next-rung and overall ETA remain distinct",
            "result and optimization opportunity remain explicit",
        ],
    },
}


RETAINED_MODULES = frozenset({
    "doctor_v5_accel_loader.py",
    "doctor_v5_accelerated_resource_policy.py",
    "doctor_v5_adapter_abi.py",
    "doctor_v5_campaign_report.py",
    "doctor_v5_census.py",
    "doctor_v5_contract.py",
    "doctor_v5_controlled_swap_activation.py",
    "doctor_v5_controlled_swap_autoresume.py",
    "doctor_v5_controlled_swap_successor.py",
    "doctor_v5_disk25_successor.py",
    "doctor_v5_gc_runtime_transition.py",
    "doctor_v5_gptoss_moe_adapter.py",
    "doctor_v5_gptoss_mxfp4.py",
    "doctor_v5_local_observer.py",
    "doctor_v5_parameter_manifest.py",
    "doctor_v5_pass_b_worker.py",
    "doctor_v5_phase_aware_disk_gate.py",
    "doctor_v5_physical_ab_controller.py",
    "doctor_v5_physical_ab_executor.py",
    "doctor_v5_physical_adapter_registry.py",
    "doctor_v5_physical_counter_barrier.py",
    "doctor_v5_physical_result_authority.py",
    "doctor_v5_post_120b.py",
    "doctor_v5_qwen_treatment_adapter.py",
    "doctor_v5_qwen_treatment_block_parallel_adapter.py",
    "doctor_v5_remaining_scratch_ledger.py",
    "doctor_v5_sharded_eval.py",
    "doctor_v5_source_seal.py",
    "doctor_v5_stacked_admission.py",
    "doctor_v5_strand_ladder_adapter.py",
    "doctor_v5_strand_ladder_block_parallel_adapter.py",
    "doctor_v5_strand_ladder_block_parallel_worker.py",
    "doctor_v5_strand_ladder_worker.py",
    "doctor_v5_telegram_rung_notifier.py",
    "doctor_v5_ultra_accelerated_autoresume.py",
    "doctor_v5_ultra_accelerated_queue.py",
    "doctor_v5_ultra_queue.py",
})


def _name(raw: str) -> str:
    name = Path(raw).name.removesuffix(".py")
    if not name.startswith("doctor_v5_"):
        name = "doctor_v5_" + name.replace("-", "_")
    return name


def archive_source(name: str) -> bytes:
    normalized = _name(name)
    if normalized not in ARCHIVED:
        raise KeyError(f"unknown retired Doctor module: {name}")
    relative = f"tools/condense/{normalized}.py"
    raw = subprocess.check_output(["git", "show", f"{ARCHIVE_COMMIT}:{relative}"], cwd=ROOT)
    expected = ARCHIVED[normalized][2]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"Git archive hash mismatch for {relative}")
    return raw


def legacy_record(name: str, *, include_source: bool = False) -> dict[str, Any]:
    normalized = _name(name)
    profile, replacement, digest = ARCHIVED[normalized]
    raw = archive_source(normalized)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "legacy_module": normalized + ".py",
        "archive_commit": ARCHIVE_COMMIT,
        "source_sha256": digest,
        "profile": profile,
        "replacement_command": replacement,
        "schemas": sorted(row.decode() for row in set(SCHEMA_RE.findall(raw))),
        "artifact_paths": sorted(row.decode().rstrip("./") for row in set(REPORT_RE.findall(raw))),
        "subcommands": sorted(row.decode() for row in set(SUBCOMMAND_RE.findall(raw))),
    }
    if include_source:
        record["source"] = raw.decode("utf-8")
    return record


def profile_document() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "archive_commit": ARCHIVE_COMMIT,
        "profiles": copy.deepcopy(PROFILES),
        "retained_module_count": len(RETAINED_MODULES),
        "retired_module_count": len(ARCHIVED),
        "retired_modules_sha256": common.canonical_sha256(ARCHIVED),
    }


def _strict_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _bound_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path)
    resolved = source.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 512 * 1024 * 1024:
            raise ValueError(f"invalid or oversized handoff input: {resolved}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"short read: {resolved}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"file grew during read: {resolved}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
    if identity(before) != identity(after):
        raise ValueError(f"file changed during read: {resolved}")
    raw = b"".join(chunks)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {resolved}")
    return value, {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _post120_promotion_gate() -> dict[str, Any]:
    return {
        "currently_permitted": False,
        "all_required": True,
        "facet_receipts": {facet: "missing" for facet in POST120_FACETS},
        "additional_requirements": [
            "source_manifest_and_parent_hashes_revalidated",
            "exact_10x4_matrix_preflighted",
            "all_adapters_reviewed_for_exact_architecture",
            "owner_free_baselines_and_rollback_point_sealed",
            "disk_and_lifecycle_admission_passed",
            "observer_structural_readiness_passed",
            "quiescent_generation_compare_and_swap_succeeded",
            "physical_ab_all_10_facets_green_for_the_exact_segment",
        ],
        "estimates_or_simulations_can_satisfy_gate": False,
    }


def validate_handoff_packet(doc: Any, *, verify_files: bool = True) -> list[str]:
    """Validate the retired post-120B handoff without reviving its file graph."""
    if not isinstance(doc, dict) or doc.get("schema") != POST120_HANDOFF_SCHEMA:
        return ["post-120B handoff schema mismatch"]
    errors: list[str] = []
    expected_fields = {
        "schema", "created_at", "status", "bindings", "coverage", "readiness",
        "promotion_gate", "claim_boundary", "handoff_sha256",
    }
    if set(doc) != expected_fields:
        errors.append("post-120B handoff fields differ")
    unstamped = copy.deepcopy(doc)
    claimed = unstamped.pop("handoff_sha256", None)
    try:
        observed = _strict_sha256(unstamped)
    except (TypeError, ValueError):
        observed = ""
    if not isinstance(claimed, str) or not HEX64.fullmatch(claimed) or claimed != observed:
        errors.append("post-120B handoff hash mismatch")
    boundary = {
        "intermediate_or_structural_evidence_is_final_quality": False,
        "unsupported_or_negative_outcomes_synthesized": False,
        "live_queue_worker_registry_plan_or_runtime_specs_mutated": False,
        "runtime_defaults_changed": False,
        "source_or_parent_deletion_permitted": False,
        "execution_permitted": False,
        "quality_claims_permitted": False,
    }
    if doc.get("status") != "sealed-unbound-handoff-not-executable" \
            or doc.get("claim_boundary") != boundary:
        errors.append("post-120B handoff crosses its claim boundary")
    bindings = doc.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(POST120_BINDINGS):
        errors.append("post-120B handoff binding inventory differs")
    else:
        for name, semantic_field in POST120_BINDINGS.items():
            row = bindings[name]
            if not isinstance(row, dict) or set(row) != {
                "path", "sha256", "bytes", "schema",
                "semantic_hash_field", "semantic_sha256",
            }:
                errors.append(f"handoff artifact binding differs: {name}")
                continue
            if row.get("semantic_hash_field") != semantic_field \
                    or not isinstance(row.get("schema"), str) \
                    or not HEX64.fullmatch(str(row.get("sha256", ""))) \
                    or not HEX64.fullmatch(str(row.get("semantic_sha256", ""))) \
                    or isinstance(row.get("bytes"), bool) \
                    or not isinstance(row.get("bytes"), int) or row["bytes"] < 0 \
                    or not isinstance(row.get("path"), str) \
                    or not Path(row["path"]).is_absolute():
                errors.append(f"handoff artifact binding differs: {name}")
                continue
            if not verify_files:
                continue
            try:
                value, artifact = _bound_json(Path(row["path"]))
                semantic = value.get(semantic_field)
                unstamped_input = copy.deepcopy(value)
                unstamped_input.pop(semantic_field, None)
                if artifact != {key: row[key] for key in ("path", "sha256", "bytes")} \
                        or value.get("schema") != row["schema"] \
                        or semantic != row["semantic_sha256"] \
                        or semantic != _strict_sha256(unstamped_input):
                    errors.append(f"handoff artifact binding differs: {name}")
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append(f"post-120B handoff input verification failed: {name}: {exc}")
    expected_coverage = {
        "gptoss_source_units": 615,
        "gptoss_isolated_jobs": 24_600,
        "gptoss_rates": 10,
        "gptoss_branches": 4,
        "gptoss_pending_cells": 40,
        "named_higher_tier_horizons": 3,
        "named_horizon_cell_templates": 120,
        "aggressive_facets": list(POST120_FACETS),
        "physical_ab_facets": list(PHYSICAL_AB_FACETS),
        "physical_thread_profile_cells": 800,
        "physical_block_parallel_cells": 200,
    }
    if doc.get("coverage") != expected_coverage:
        errors.append("post-120B handoff coverage differs")
    readiness = doc.get("readiness")
    fixed_readiness = {
        "exact_120b_work_plan_structurally_valid": True,
        "exact_40_cell_pending_wiring_structurally_valid": True,
        "exact_source_rate_branch_fanout_structurally_valid": True,
        "shared_preprocess_cache_structurally_valid": True,
        "tokenizer_dual_path_gate_passed_but_not_promotion_reviewed": True,
        "higher_tier_generic_manifest_and_admission_contracts_present": True,
        "higher_tier_exact_source_manifests_present": 0,
        "physical_acceleration_facets_qualified": 0,
        "physical_ab_plan_structurally_valid": True,
        "physical_ab_facets_total": 10,
        "physical_ab_facets_qualified": 0,
        "physical_ab_execution_permitted": False,
        "reviewed_120b_live_adapters": 0,
    }
    dynamic_readiness = {
        "shared_derived_preprocess_consumers_still_unqualified",
        "physical_ab_program_adapters_registered",
    }
    if not isinstance(readiness, dict) \
            or set(readiness) != set(fixed_readiness) | dynamic_readiness \
            or any(readiness.get(key) != value for key, value in fixed_readiness.items()) \
            or any(
                isinstance(readiness.get(key), bool)
                or not isinstance(readiness.get(key), int)
                or readiness[key] < 0
                for key in dynamic_readiness
            ):
        errors.append("post-120B handoff readiness differs")
    if doc.get("promotion_gate") != _post120_promotion_gate():
        errors.append("post-120B handoff promotion gate differs")
    return errors


def validate() -> list[str]:
    errors: list[str] = []
    if len(ARCHIVED) != 39:
        errors.append("retired module inventory must contain 39 modules")
    if any(profile not in PROFILES for profile, _replacement, _digest in ARCHIVED.values()):
        errors.append("retired module references an unknown profile")
    for name, (_profile, _replacement, digest) in ARCHIVED.items():
        if not HEX64.fullmatch(digest):
            errors.append(f"invalid archive SHA-256: {name}")
            continue
        try:
            archive_source(name)
        except (KeyError, OSError, subprocess.SubprocessError, RuntimeError) as exc:
            errors.append(f"archive unavailable: {name}: {exc}")
    current = {path.name for path in (ROOT / "tools/condense").glob("doctor_v5_*.py")}
    expected = set(RETAINED_MODULES) | {"doctor_v5_profiles.py"}
    if current != expected:
        errors.append(
            f"Doctor source layout differs: missing={sorted(expected - current)} "
            f"extra={sorted(current - expected)}"
        )
    if profile_document()["retained_module_count"] != 37:
        errors.append("canonical retained inventory must contain 37 pre-profile modules")
    return errors


def replay(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latest: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"events[{index}] must be an object")
        cell_id, status = event.get("cell_id"), event.get("status")
        attempt = event.get("attempt", 0)
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(f"events[{index}].cell_id must be non-empty")
        if status not in ALLOWED_STATUS:
            raise ValueError(f"events[{index}].status is invalid")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise ValueError(f"events[{index}].attempt must be non-negative")
        latest[cell_id] = copy.deepcopy(event)
    counts = {
        status: sum(row["status"] == status for row in latest.values())
        for status in sorted(ALLOWED_STATUS)
    }
    document = {
        "schema": REPLAY_SCHEMA,
        "cell_count": len(latest),
        "terminal_count": sum(row["status"] in TERMINAL for row in latest.values()),
        "attempts_total": sum(row.get("attempt", 0) for row in latest.values()),
        "status_counts": counts,
        "cells": [latest[cell_id] for cell_id in sorted(latest)],
    }
    document["replay_sha256"] = common.canonical_sha256(document)
    return document


def _selftest() -> int:
    assert not validate()
    result = replay([
        {"cell_id": "a", "status": "running", "attempt": 1},
        {"cell_id": "a", "status": "complete", "attempt": 1},
        {"cell_id": "b", "status": "resource-stop", "attempt": 2},
        {"cell_id": "c", "status": "negative", "attempt": 1},
    ])
    assert result["cell_count"] == 3 and result["terminal_count"] == 2
    assert result["attempts_total"] == 4
    assert result["status_counts"]["resource-stop"] == 1
    print("doctor_v5_profiles.py selftest OK")
    return 0


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _compat(name: str, argv: list[str]) -> int:
    if argv in (["selftest"], ["--selftest"]):
        return _selftest()
    record = legacy_record(name)
    record.update({
        "status": "superseded",
        "compatibility_arguments": argv,
        "mutates_campaign": False,
    })
    _print(record)
    informational = not argv or argv[0] in {
        "status", "plan", "show", "list", "validate", "dry-run", "--dry-run",
    }
    return 0 if informational else 64


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("profile", choices=sorted(PROFILES))
    legacy = sub.add_parser("legacy")
    legacy.add_argument("name")
    legacy.add_argument("--source", action="store_true")
    schemas = sub.add_parser("schemas")
    schemas.add_argument("name")
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("events", type=Path)
    replay_parser.add_argument("--output", type=Path)
    sub.add_parser("validate")
    sub.add_parser("selftest")
    compat = sub.add_parser("compat")
    compat.add_argument("name")
    compat.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "list":
        _print(profile_document())
        return 0
    if args.command == "show":
        _print({"schema": SCHEMA, "profile": args.profile, **PROFILES[args.profile]})
        return 0
    if args.command == "legacy":
        _print(legacy_record(args.name, include_source=args.source))
        return 0
    if args.command == "schemas":
        _print(legacy_record(args.name)["schemas"])
        return 0
    if args.command == "validate":
        errors = validate()
        _print({"ok": not errors, "errors": errors})
        return 0 if not errors else 1
    if args.command == "selftest":
        return _selftest()
    if args.command == "compat":
        return _compat(args.name, args.arguments)
    value = replay(common.read_json(args.events))
    if args.output:
        common.atomic_write_json(args.output, value)
    else:
        _print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
