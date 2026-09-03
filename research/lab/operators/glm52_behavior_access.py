"""GLM-5.2 behavior + activation access: feasibility, architecture map, gates.

This is the long-pole admission surface for obtaining:

* BEHAVIOR — generation trajectories and top-k logits/logprobs (outputs)
* ACTIVATIONS — internal states at transplant points (hidden, router, MoE, IndexShare)

It does not invent a forward, a trajectory, or an activation. Every blocked gate is
named with a concrete local path when one exists.

Stages (program language):

1. Behavior / output distillation — needs OUTPUTS only
2. Trajectory-aligned student fit — needs OUTPUTS (+ optional token ids)
3. Activation-bridge / organ transplant — needs LOCAL activations
4. Full chain teacher re-capture — needs local streaming + reference forward
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from lab.layout import evidence_dir
from lab.operators.glm52_adapter import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_MODEL_TYPE,
    IMMUTABLE_REVISION,
    REPO_ID,
)
from lab.operators.glm52_common import (
    Glm52Error,
    atomic_json,
    read_json,
    seal,
    utc_now,
    verify_sealed,
)

SCHEMA_FEASIBILITY = "hawking.glm52.behavior_access_feasibility.v1"
SCHEMA_ARCHITECTURE_MAP = "hawking.glm52.behavior_architecture_map.v1"
SCHEMA_MULTILANE = "hawking.glm52.behavior_access_multilane_plan.v1"

GLM52_EVIDENCE = evidence_dir("glm52")
SUPPORT = Path(
    os.environ.get(
        "GLM52_SUPPORT_ROOT",
        "/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity",
    )
)
SOURCE_ROOT = SUPPORT / "source"
CONTROL_ROOT = SUPPORT / "control"
TEACHER_DIR = SUPPORT / "source_fetch" / "teacher"
CAPSULES_DIR = TEACHER_DIR / "capsules"
FETCH_PROGRESS = SUPPORT / "source_fetch" / "progress.json"

# Hard floor from the program: keep at least 15 GiB free; operational schedule
# used a larger reserve during body restreams. Never hold 1.5 TB resident.
DISK_FLOOR_BYTES = int(os.environ.get("GLM52_BEHAVIOR_DISK_FLOOR_BYTES", 15 * 1024**3))
SOURCE_PAYLOAD_BYTES = 1_506_659_919_872  # sealed census

# Hosted OpenAI-compatible providers that advertise GLM-5.2 as of 2026-08.
# Identity of the *served* weights is provider-claimed, not byte-verified against
# zai-org/GLM-5.2@b4734de4. That gap is recorded, not papered over.
HOSTED_PROVIDERS: tuple[dict[str, Any], ...] = (
    {
        "id": "zhipu_bigmodel",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model_ids": ("glm-5.2",),
        "api_key_env": ("ZHIPU_API_KEY", "BIGMODEL_API_KEY", "ZAI_API_KEY"),
        "first_party": True,
        "notes": "Zhipu BigModel / Z.ai first-party OpenAI-compatible chat completions",
    },
    {
        "id": "zai_global",
        "base_url": "https://api.z.ai/api/paas/v4",
        "model_ids": ("glm-5.2",),
        "api_key_env": ("ZAI_API_KEY", "ZHIPU_API_KEY"),
        "first_party": True,
        "notes": "Z.ai global OpenAI-compatible surface",
    },
    {
        "id": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model_ids": ("z-ai/glm-5.2",),
        "api_key_env": ("OPENROUTER_API_KEY",),
        "first_party": False,
        "notes": "Router; may land on Z.ai / Fireworks / Together / DeepInfra backends",
    },
    {
        "id": "together",
        "base_url": "https://api.together.xyz/v1",
        "model_ids": ("zai-org/GLM-5.2",),
        "api_key_env": ("TOGETHER_API_KEY",),
        "first_party": False,
        "notes": "Together AI hosted zai-org/GLM-5.2",
    },
    {
        "id": "fireworks",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "model_ids": ("accounts/fireworks/models/glm-5p2",),
        "api_key_env": ("FIREWORKS_API_KEY",),
        "first_party": False,
        "notes": "Fireworks day-zero GLM-5.2 inference",
    },
)

TRANSPLANT_POINTS: tuple[dict[str, Any], ...] = (
    {
        "id": "embed_tokens",
        "layer_scope": "global",
        "shape": "[B, S, 6144]",
        "needs": "local embed_tokens rows or full table",
        "stage": 3,
    },
    {
        "id": "input_hidden",
        "layer_scope": "per_layer",
        "shape": "[B, S, 6144]",
        "needs": "local layer forward or sealed capsule",
        "stage": 3,
    },
    {
        "id": "attention_input",
        "layer_scope": "per_layer",
        "shape": "[B, S, 6144]",
        "needs": "local pre-attention residual after input_layernorm",
        "stage": 3,
    },
    {
        "id": "indexshare_selection",
        "layer_scope": "per_full_indexer_layer",
        "shape": "[B, S, index_topk=2048]",
        "needs": "local DSA indexer; APIs cannot supply",
        "stage": 3,
    },
    {
        "id": "attention_output",
        "layer_scope": "per_layer",
        "shape": "[B, S, 6144]",
        "needs": "local MLA/DSA attention",
        "stage": 3,
    },
    {
        "id": "pre_router_hidden",
        "layer_scope": "sparse_layers",
        "shape": "[B, S, 6144]",
        "needs": "local post-attention-layernorm",
        "stage": 3,
    },
    {
        "id": "router_logits_topk",
        "layer_scope": "sparse_layers",
        "shape": "logits [B,S,256]; top-8 indices+weights",
        "needs": "local MoE router (noaux_tc, sigmoid, scale 2.5)",
        "stage": 3,
    },
    {
        "id": "post_moe",
        "layer_scope": "sparse_layers",
        "shape": "[B, S, 6144]",
        "needs": "local shared+routed expert sum",
        "stage": 3,
    },
    {
        "id": "block_output",
        "layer_scope": "per_layer",
        "shape": "[B, S, 6144]",
        "needs": "local residual add after MoE/dense MLP",
        "stage": 3,
    },
    {
        "id": "final_logits_topk",
        "layer_scope": "global",
        "shape": "[B, S, 154880] or top-k over vocab",
        "needs": "local lm_head OR hosted logprobs (provider-dependent)",
        "stage": 1,
    },
    {
        "id": "generation_trajectory",
        "layer_scope": "global",
        "shape": "token id sequence + optional top-k per step",
        "needs": "hosted chat/completions OR local generate",
        "stage": 1,
    },
)


def _free_bytes(path: Path | None = None) -> int:
    target = path if path is not None else Path.cwd()
    try:
        return int(shutil.disk_usage(str(target)).free)
    except OSError:
        return -1


def _capsule_inventory() -> dict[str, Any]:
    """Metadata-only capsules vs resident NPZ payloads."""
    if not CAPSULES_DIR.is_dir():
        return {
            "directory": str(CAPSULES_DIR),
            "exists": False,
            "metadata_json_count": 0,
            "resident_npz_count": 0,
            "missing_npz_capsule_ids": [],
            "layers_claimed_by_metadata": [],
            "payload_status": "ABSENT",
        }
    metas = sorted(CAPSULES_DIR.glob("*.json"))
    missing: list[str] = []
    resident: list[str] = []
    layers: set[int] = set()
    for meta_path in metas:
        try:
            meta = read_json(meta_path)
        except Glm52Error:
            continue
        capsule_id = str(meta.get("capsule_id") or meta_path.stem)
        npz_path = CAPSULES_DIR / f"{capsule_id}.npz"
        if npz_path.is_file():
            resident.append(capsule_id)
        else:
            # capsule_path may point elsewhere (historical absolute path)
            alt = Path(str(meta.get("capsule_path") or ""))
            if alt.is_file():
                resident.append(capsule_id)
            else:
                missing.append(capsule_id)
        for layer in meta.get("layers") or []:
            layers.add(int(layer))
    return {
        "directory": str(CAPSULES_DIR),
        "exists": True,
        "metadata_json_count": len(metas),
        "resident_npz_count": len(resident),
        "missing_npz_capsule_ids": missing,
        "layers_claimed_by_metadata": sorted(layers),
        "payload_status": (
            "RESIDENT"
            if resident and not missing
            else "METADATA_ONLY_PAYLOADS_EVICTED"
            if metas and not resident
            else "PARTIAL"
            if resident
            else "ABSENT"
        ),
        "honest_note": (
            "JSON receipts claim a full 0..77 chain capture with NATURAL_CORPUS_PARTITION "
            "calibration, but the multi-GB .npz activation payloads are not on disk. "
            "Re-use requires re-stream + re-capture or restore from durable archive."
        ),
    }


def _source_body_inventory() -> dict[str, Any]:
    progress: dict[str, Any] = {}
    if FETCH_PROGRESS.is_file():
        try:
            progress = read_json(FETCH_PROGRESS)
        except Glm52Error:
            progress = {"status": "UNREADABLE"}
    resident_shards = 0
    resident_bytes = 0
    if SOURCE_ROOT.is_dir():
        for path in SOURCE_ROOT.glob("model-*-of-*.safetensors"):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > 0:
                resident_shards += 1
                resident_bytes += size
    return {
        "source_root": str(SOURCE_ROOT),
        "control_config_present": (CONTROL_ROOT / "config.json").is_file(),
        "resident_weight_shards": resident_shards,
        "resident_weight_bytes": resident_bytes,
        "source_payload_bytes_census": SOURCE_PAYLOAD_BYTES,
        "complete_source_resident": resident_bytes >= SOURCE_PAYLOAD_BYTES * 0.99,
        "fetch_progress": {
            "state": progress.get("state"),
            "resident_bytes": progress.get("resident_bytes"),
            "shards_verified_total": progress.get("shards_verified_total"),
            "source_fraction_verified": progress.get("source_fraction_verified"),
            "finished_at": progress.get("finished_at"),
        },
        "status": (
            "VERIFIED_AND_EVICTED"
            if progress.get("state") == "ALL_SHARDS_VERIFIED" and resident_shards == 0
            else "PARTIALLY_RESIDENT"
            if resident_shards
            else "NOT_RESIDENT"
        ),
    }


def _provider_credential_probe() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in HOSTED_PROVIDERS:
        env_names = tuple(provider["api_key_env"])
        present = [name for name in env_names if os.environ.get(name)]
        rows.append(
            {
                "id": provider["id"],
                "base_url": provider["base_url"],
                "model_ids": list(provider["model_ids"]),
                "first_party": provider["first_party"],
                "credential_env_candidates": list(env_names),
                "credential_present": bool(present),
                "credential_env_hit": present[0] if present else None,
                "notes": provider["notes"],
            }
        )
    return rows


def architecture_map() -> dict[str, Any]:
    """Sealed architecture map for behavior + activation access planning."""
    contract_path = GLM52_EVIDENCE / "GLM52_ARCHITECTURE_CONTRACT.json"
    schedule_path = GLM52_EVIDENCE / "GLM52_STREAMING_SCHEDULE.json"
    contract = verify_sealed(read_json(contract_path), label=str(contract_path))
    schedule = verify_sealed(read_json(schedule_path), label=str(schedule_path))
    geometry = contract["geometry"]
    dsa = contract["dsa_indexshare"]
    routing = contract["routing"]
    w0 = schedule["windows"][0]
    payload = {
        "schema": SCHEMA_ARCHITECTURE_MAP,
        "built_at": utc_now(),
        "repo": REPO_ID,
        "revision": IMMUTABLE_REVISION,
        "architecture": EXPECTED_ARCHITECTURE,
        "model_type": EXPECTED_MODEL_TYPE,
        "geometry": {
            "main_hidden_layers": geometry["main_hidden_layers"],
            "hidden_size": geometry["hidden_size"],
            "vocab_size": geometry["vocab_size"],
            "attention_heads": geometry["attention_heads"],
            "kv_heads": geometry["kv_heads"],
            "q_lora_rank": geometry["q_lora_rank"],
            "kv_lora_rank": geometry["kv_lora_rank"],
            "qk_head_dim": geometry["qk_head_dim"],
            "v_head_dim": geometry["v_head_dim"],
            "routed_experts": geometry["routed_experts"],
            "shared_experts": geometry["shared_experts"],
            "active_routed_experts_per_token": geometry["active_routed_experts_per_token"],
            "moe_intermediate_size": geometry["moe_intermediate_size"],
            "dense_layers": geometry["dense_layers"],
            "sparse_main_layers": geometry["sparse_main_layers"],
            "mtp_layers": geometry["mtp_layers"],
            "context_tokens": geometry["context_tokens"],
            "source_payload_bytes": SOURCE_PAYLOAD_BYTES,
        },
        "indexshare": {
            "index_topk": dsa["index_topk"],
            "index_heads": dsa["index_heads"],
            "index_head_dim": dsa["index_head_dim"],
            "main_full_indexer_layers": dsa["main_full_indexer_layers"],
            "main_shared_indexer_layers": dsa["main_shared_indexer_layers"],
            "note": "IndexShare sparse attention is GLM-specific; no local DSV4F-style runtime exists for it yet",
        },
        "routing": {
            "scoring": routing["scoring"],
            "topk_method": routing["topk_method"],
            "scaling_factor": routing["scaling_factor"],
            "normalize_topk": routing["normalize_topk"],
            "router_compute_dtype": routing["router_compute_dtype"],
        },
        "transplant_points": list(TRANSPLANT_POINTS),
        "streaming_schedule": {
            "path": str(schedule_path),
            "seal_sha256": schedule["seal_sha256"],
            "window_count": schedule["window_count"],
            "status": schedule["status"],
            "maximum_resident_shards_in_one_window": schedule[
                "maximum_resident_shards_in_one_window"
            ],
            "W000": {
                "window_id": w0["window_id"],
                "organ_ids": w0["organ_ids"],
                "source_shard_count": w0["source_shard_count"],
                "new_fetch_logical_bytes": w0["new_fetch_logical_bytes"],
                "resident_logical_bytes": w0["resident_logical_bytes"],
                "unblocks": [
                    "global_input (embed_tokens row reads)",
                    "text_layer_00..06 first-window activations",
                    "first-layer reference forward once shards resident",
                ],
            },
        },
        "existing_local_runtime_pieces": {
            "numpy_reference_forward": "lab/operators/glm52_reference.py",
            "teacher_capture": "lab/operators/glm52_teacher_capture.py",
            "range_stream_executor": "lab/operators/glm52_range_stream_executor.py",
            "source_fetch": "lab/operators/glm52_source_fetch.py",
            "parity_suite_synthetic": "lab/operators/glm52_parity.py",
            "honest_scope": (
                "Reference + teacher_capture can run a layer when its shards are resident. "
                "There is no full-model local generate path. Official Transformers main is "
                "supported in principle (config contract) but a 1.5 TB resident load is forbidden; "
                "windowed stream + evict is the only local plan."
            ),
        },
        "contract_seal_sha256": contract["seal_sha256"],
        "status": "MAPPED",
    }
    return seal(payload)


def path_comparison() -> dict[str, Any]:
    """Honest cost/correctness comparison of the three access paths."""
    free = _free_bytes(SOURCE_ROOT if SOURCE_ROOT.exists() else Path.cwd())
    capsules = _capsule_inventory()
    body = _source_body_inventory()
    providers = _provider_credential_probe()
    any_credential = any(row["credential_present"] for row in providers)

    path_a = {
        "id": "local_streaming_inference",
        "label": "(a) local streaming inference of 1.5 TB weights (evicting)",
        "supplies": ["activations", "router/indexshare traces", "local logits", "bounded generation"],
        "does_not_supply_until_built": ["cheap bulk trajectories"],
        "unblocks_stages": [3, 4],
        "also_unblocks_if_generate_works": [1, 2],
        "correctness": (
            "Highest: weights can be hash-verified against the sealed official manifest; "
            "activations come from the same bytes as the admitted revision."
        ),
        "cost": {
            "engineering": (
                "Multi-week, comparable to DSV4F runtime which is still only a BOS forward "
                "after many lanes. Must implement IndexShare DSA, MLA, MoE noaux_tc, MTP, "
                "windowed shard residency, and eviction under the 15 GiB floor."
            ),
            "compute_per_calibration_pass": (
                f"W000 alone is {w0_bytes_gib()} GiB new-fetch logical; full body "
                f"{SOURCE_PAYLOAD_BYTES / 1024**3:.1f} GiB streamed once per full chain "
                "with eviction between windows."
            ),
            "disk_floor_bytes": DISK_FLOOR_BYTES,
            "never_hold_resident": SOURCE_PAYLOAD_BYTES,
        },
        "current_gate": {
            "weight_body": body["status"],
            "resident_weight_shards": body["resident_weight_shards"],
            "capsule_payloads": capsules["payload_status"],
            "first_layer_forward": (
                "BLOCKED: no resident shards under "
                f"{SOURCE_ROOT}; reference decoder exists at "
                "lab/operators/glm52_reference.py:decoder_layer but cannot load tensors"
            ),
            "full_generate": "BLOCKED: no local runtime",
            "reuse": (
                "Streaming schedule + source admission + reference + teacher_capture "
                "already exist; do not re-derive architecture. Restream under sealed "
                "schedule is the next physical step for activations."
            ),
        },
        "ready_now": False,
    }

    path_b = {
        "id": "hosted_outputs_only",
        "label": "(b) hosted GLM-5.2 endpoint for OUTPUTS-ONLY behavior",
        "supplies": [
            "generation trajectories",
            "token sequences",
            "top-k logprobs when the provider exposes them",
        ],
        "does_not_supply": [
            "hidden states",
            "router logits / expert indices",
            "IndexShare selections",
            "any transplant-point activation",
        ],
        "unblocks_stages": [1, 2],
        "correctness": (
            "Sufficient for behavior/output distillation. Weight identity is "
            "provider-claimed (model id string), not byte-equal to "
            f"{REPO_ID}@{IMMUTABLE_REVISION[:12]}. Prefer first-party Zhipu/Z.ai when possible. "
            "Never claim activation parity from hosted outputs."
        ),
        "cost": {
            "engineering": "Days: harness + sealed prompt corpus + logprob capture + provenance",
            "api": "Per-token hosted pricing; no 1.5 TB disk or RAM",
            "disk_floor_bytes": DISK_FLOOR_BYTES,
        },
        "current_gate": {
            "providers_catalogued": len(providers),
            "any_api_credential_present": any_credential,
            "live_capture": (
                "READY_WHEN_CREDENTIAL_SET"
                if any_credential
                else "BLOCKED_NO_API_CREDENTIAL — set one of ZHIPU_API_KEY / ZAI_API_KEY / "
                "OPENROUTER_API_KEY / TOGETHER_API_KEY / FIREWORKS_API_KEY"
            ),
            "harness": "lab/operators/glm52_behavior_capture.py",
        },
        "ready_now": any_credential,
        "providers": providers,
    }

    path_c = {
        "id": "bounded_local_calibration",
        "label": "(c) bounded local generation for a small calibration set",
        "supplies": [
            "small sealed trajectories from local weights",
            "matched activations for the same prompts",
        ],
        "unblocks_stages": [1, 2, 3],
        "correctness": (
            "Highest for the calibration set itself (same process, verified weights). "
            "Does not scale to bulk behavior distillation without repeating the full stream."
        ),
        "cost": {
            "engineering": (
                "Weeks after path (a) first-window forward works: generate loop, KV cache, "
                "lm_head row streaming, top-k over 154880 vocab without materializing full logits "
                "when possible."
            ),
            "compute": "One or more full-window restreams per calibration campaign",
        },
        "current_gate": {
            "depends_on": "path_a first-window forward + lm_head access",
            "status": "BLOCKED_UNTIL_LOCAL_FORWARD",
        },
        "ready_now": False,
    }

    # Primary choice: highest value / lowest arch risk that is actually approachable.
    chosen = {
        "primary": "hosted_outputs_only",
        "primary_why": (
            "Stages 1–2 (behavior/output distillation) are the program's highest-value, "
            "lowest-arch-risk work and need only OUTPUTS. Hosted GLM-5.2 endpoints exist "
            "(Zhipu/Z.ai first-party, OpenRouter, Together, Fireworks). Local 1.5 TB "
            "streaming runtime is a multi-week long pole (DSV4F-comparable) and is still "
            "required for stage 3 activations — run it in parallel, do not wait on it for "
            "behavior capture."
        ),
        "parallel_long_pole": "local_streaming_inference",
        "parallel_why": (
            "APIs cannot emit internal states. Activation-bridge / organ transplant remains "
            "blocked until windowed local serving exists. Capsule metadata claims layers "
            "0–77 were captured once, but NPZ payloads are gone; re-stream is mandatory for "
            "fresh activations."
        ),
        "not_chosen_as_first_increment": "bounded_local_calibration",
        "not_chosen_why": "Strict subset of path (a); unblocks only after first-window forward.",
    }

    return {
        "free_disk_bytes": free,
        "disk_floor_bytes": DISK_FLOOR_BYTES,
        "disk_floor_respected": free < 0 or free >= DISK_FLOOR_BYTES,
        "capsules": capsules,
        "source_body": body,
        "paths": {
            "local_streaming_inference": path_a,
            "hosted_outputs_only": path_b,
            "bounded_local_calibration": path_c,
        },
        "chosen": chosen,
        "stage_matrix": {
            "1_behavior_output_distillation": {
                "needs": "trajectories + optional top-k logits/logprobs",
                "path": "hosted_outputs_only (primary); local generate later",
            },
            "2_trajectory_aligned_student_fit": {
                "needs": "same as stage 1, sealed train/holdout splits",
                "path": "hosted_outputs_only",
            },
            "3_activation_bridge_organ_transplant": {
                "needs": "hidden/router/indexshare/moe activations at transplant points",
                "path": "local_streaming_inference only",
            },
            "4_full_chain_teacher_recapture": {
                "needs": "all windows streamed, reference forward, capsules re-sealed",
                "path": "local_streaming_inference + teacher_capture",
            },
        },
        "prior_program_verdict": {
            "functional_decision": "FUNCTIONAL_PARTIAL_ONLY",
            "evidence": "workspace/campaign/evidence/models/glm52/GLM52_FUNCTIONAL_DECISION.json",
            "reading": (
                "Activation-space functional students diverged (expansive residual). "
                "That strengthens the case for outputs-only behavior distillation as the "
                "near-term bet, while still requiring local activations for any bridge work."
            ),
        },
    }


def w0_bytes_gib() -> float:
    schedule_path = GLM52_EVIDENCE / "GLM52_STREAMING_SCHEDULE.json"
    try:
        schedule = read_json(schedule_path)
        return float(schedule["windows"][0]["new_fetch_logical_bytes"]) / 1024**3
    except (Glm52Error, KeyError, IndexError, TypeError, OSError):
        return 114.8


def multilane_plan() -> dict[str, Any]:
    """Concrete multi-lane plan with honest ETAs (calendar time on this machine class)."""
    return {
        "schema": SCHEMA_MULTILANE,
        "built_at": utc_now(),
        "assumptions": [
            "One capable engineer-agent lane full-time unless noted parallel",
            "Hosted API credential becomes available within days (owner action)",
            "No 1.5 TB resident; 15 GiB free floor held",
            "Reuse sealed schedule, admission, reference, teacher_capture — no re-derive",
            "DSV4F-comparable complexity for local IndexShare/MoE generate",
        ],
        "milestones": [
            {
                "id": "M0_behavior_harness",
                "lane": "behavior-capture",
                "what": "Hosted outputs-only harness + dry-run receipt + fail-closed gates",
                "eta": "this lane (done when receipt sealed)",
                "depends_on": [],
                "unblocks": ["stage 1 live capture once credential present"],
            },
            {
                "id": "M1_live_behavior_pilot",
                "lane": "behavior-capture",
                "what": (
                    "Owner sets API key; capture N≤64 sealed prompts with trajectories; "
                    "record top-k logprobs when provider returns them; seal train/holdout"
                ),
                "eta": "1–3 days after credential (API + bookkeeping)",
                "depends_on": ["M0_behavior_harness", "owner_api_credential"],
                "unblocks": ["stage 1 distillation data"],
            },
            {
                "id": "M2_behavior_bulk",
                "lane": "behavior-capture",
                "what": "Scale to corpus partitions used by glm52_capture_program; top-k where available",
                "eta": "3–7 days (rate limits + cost)",
                "depends_on": ["M1_live_behavior_pilot"],
                "unblocks": ["stage 2 student fit"],
            },
            {
                "id": "M3_output_student_fit",
                "lane": "distill-student",
                "what": "Fit behavior student on sealed trajectories; holdout NLL/top-1/top-k gates",
                "eta": "1–2 weeks (depends on student body choice)",
                "depends_on": ["M2_behavior_bulk"],
                "unblocks": ["behavior-distilled artifact claims under FUNCTIONAL_PARTIAL_ONLY law"],
            },
            {
                "id": "M4_W000_restream",
                "lane": "local-activation",
                "what": (
                    "Restream W000 (~115 GiB logical) under sealed schedule + resource policy; "
                    "evict after seal; no detached daemons"
                ),
                "eta": "2–5 days (Xet throughput + owner restream gates)",
                "depends_on": ["owner_restream_authorization", "disk_floor"],
                "unblocks": ["embed + layers 0–6 weights resident briefly"],
            },
            {
                "id": "M5_first_layer_forward",
                "lane": "local-activation",
                "what": (
                    "First real layer-0 reference forward on official shards via "
                    "ShardTensorSource + glm52_reference.decoder_layer; seal activation receipt"
                ),
                "eta": "3–7 days after W000 resident (wiring + memory discipline)",
                "depends_on": ["M4_W000_restream"],
                "unblocks": ["proof that local activation path works"],
            },
            {
                "id": "M6_windowed_teacher_recapture",
                "lane": "local-activation",
                "what": "Walk remaining windows; re-seal capsules with resident NPZ payloads",
                "eta": "2–4 weeks (stream/compute/evict per window; 20 windows)",
                "depends_on": ["M5_first_layer_forward"],
                "unblocks": ["stage 3 activation-bridge data"],
            },
            {
                "id": "M7_local_generate_topk",
                "lane": "local-runtime",
                "what": (
                    "Bounded local generate + top-k logits for calibration set "
                    "(path c); full-vocab lm_head without 1.5 TB residency"
                ),
                "eta": "3–6 weeks after M5 (KV cache, decode loop, lm_head streaming)",
                "depends_on": ["M5_first_layer_forward"],
                "unblocks": ["byte-verified local trajectories; path c"],
            },
            {
                "id": "M8_full_local_runtime",
                "lane": "local-runtime",
                "what": "Production-grade local serving (IndexShare + MoE + long context) — DSV4F-class",
                "eta": "multi-week to multi-month; do not schedule stage 1 behind this",
                "depends_on": ["M7_local_generate_topk"],
                "unblocks": ["offline behavior without API; full activation+behavior joint capture"],
            },
        ],
        "critical_path_for_stage_1_2": ["M0", "M1", "M2", "M3"],
        "critical_path_for_stage_3": ["M4", "M5", "M6"],
        "honest_summary_eta": {
            "stage_1_pilot_trajectories": "1–3 days after API key (harness already in tree)",
            "stage_2_bulk_behavior_distill_ready": "1–3 weeks",
            "stage_3_first_real_activation": "1–2 weeks after restream authorization",
            "stage_3_full_chain_capsules": "3–6 weeks",
            "full_local_generate_runtime": "multi-week build; comparable to DSV4F BOS-forward program",
        },
    }


def build_feasibility_receipt() -> dict[str, Any]:
    """Assemble the sealed feasibility + plan receipt for this lane."""
    arch = architecture_map()
    comparison = path_comparison()
    plan = multilane_plan()
    payload = {
        "schema": SCHEMA_FEASIBILITY,
        "built_at": utc_now(),
        "repo": REPO_ID,
        "revision": IMMUTABLE_REVISION,
        "status": "FEASIBILITY_SEALED_FIRST_INCREMENT_BEHAVIOR_HARNESS",
        "chosen_path": comparison["chosen"],
        "path_comparison": comparison,
        "architecture_map_seal_sha256": arch["seal_sha256"],
        "architecture_map": arch,
        "multilane_plan": plan,
        "non_claims": [
            "Does not claim a GLM forward was executed",
            "Does not claim a real trajectory was captured unless a live provider receipt exists",
            "Does not claim capsule NPZ payloads are resident",
            "Does not claim hosted model id is byte-identical to the sealed HF revision",
            "Does not authorize restream, Ramanujan research, or capability promotion",
        ],
        "first_increment": {
            "what": "behavior-capture harness + architecture map + feasibility receipt",
            "code": [
                "lab/operators/glm52_behavior_access.py",
                "lab/operators/glm52_behavior_capture.py",
                "tools/condense/glm52_behavior_capture.py",
            ],
            "live_capture_gate": comparison["paths"]["hosted_outputs_only"]["current_gate"][
                "live_capture"
            ],
            "local_forward_gate": comparison["paths"]["local_streaming_inference"]["current_gate"][
                "first_layer_forward"
            ],
        },
    }
    return seal(payload)


def write_feasibility_receipt(path: Path | None = None) -> dict[str, Any]:
    receipt = build_feasibility_receipt()
    target = path or (GLM52_EVIDENCE / "GLM52_BEHAVIOR_ACCESS_FEASIBILITY.json")
    atomic_json(target, receipt)
    return {"path": str(target), "seal_sha256": receipt["seal_sha256"], "status": receipt["status"]}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write sealed feasibility receipt under evidence/models/glm52/",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.write:
        result = write_feasibility_receipt(args.out)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    receipt = build_feasibility_receipt()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
