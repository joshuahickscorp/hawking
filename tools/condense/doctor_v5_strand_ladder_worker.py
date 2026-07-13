#!/usr/bin/env python3.12
"""Checkpointed multi-shard STRAND runtime for Doctor-v5 ladder cells.

Scalar and vector/sub-bit controls now both produce attested, retained STR2
archives.  Vector archives carry source/ordinal-bound SDSC tensor LUTs.  Every
control uses ``--tensor-scope all-2d`` so embeddings and untied heads cannot hide
in a BF16 pass-through channel.  The remaining 1-D tensors are still counted in
the all-in physical model payload.

No candidate geometry is allowed to call its nominal symbol payload an all-in
model rate.  The bundle accounts packed 2-D state, lossless non-2-D state, and
metadata separately, and the canonical ceiling is decided only from measured
physical bytes.  Dense reconstructions are worker-owned ephemera: they are
hash-receipted and removed after evaluation.
"""
from __future__ import annotations

import argparse
import datetime as dt
from fractions import Fraction
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_HELPER_PATH = HERE / "doctor_v5_pass_b_worker.py"
REQUEST_SCHEMA = "hawking.doctor_v5_strand_ladder_request.v1"
CHECKPOINT_SCHEMA = "hawking.doctor_v5_strand_ladder_checkpoint.v1"
RECEIPT_SCHEMA = "hawking.doctor_v5_strand_ladder_receipt.v1"
OVERRIDE_SCHEMA = "hawking.doctor_v5_sharded_override.v1"
CAPABILITY_SCHEMA = "hawking.doctor_v5_strand_ladder_capabilities.v1"
MODEL_FAMILY = "qwen2.5-dense"
SUPPORTED_LABELS = ("0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B")
RESIDENT_LABELS = ("0.5B", "1.5B", "3B", "7B", "14B")
TREATMENT_BRANCHES = ("doctor_static", "doctor_conditional", "doctor_full")
TREATMENT_METHOD = "qwen_packed_reencode_v1"
# Memory budget handed to quantize-model so learned-codebook (vector/sub-bit) encodes run
# multi-threaded across tensors instead of single-threaded. The binary caps workers so peak
# per-worker working set stays under this budget; 12 GiB is safe alongside the largest
# whole-resident tier (14B f32 jobs ~56 GiB) inside the 78 GiB process budget, and the
# per-tensor cap auto-reduces workers for the larger-tensor streamed tiers.
LEARNED_ENCODE_MEM_BUDGET_BYTES = 12 * 1024**3
DEPENDENCY_EVIDENCE_SCHEMA = "hawking.doctor_v5_treatment_dependency_evidence.v1"
BASELINE_CACHE_SCHEMA = "hawking.doctor_v5_shared_baseline_cache.v1"
BASELINE_CACHE_ROOT = ROOT / "reports/condense/doctor_v5_ultra/baseline_cache"
CANONICAL_RATES: dict[str, Fraction] = {
    "4": Fraction(4, 1), "3": Fraction(3, 1), "2": Fraction(2, 1),
    "1": Fraction(1, 1), "0.8": Fraction(4, 5),
    "0.55": Fraction(11, 20), "0.5": Fraction(1, 2),
    "0.33": Fraction(33, 100), "0.25": Fraction(1, 4),
    "0.1": Fraction(1, 10),
}
# Reviewed *candidate* geometries.  Their payload fractions sit at/below the
# physical ceiling, but overhead can still miss it; only the emitted file decides.
# Larger blocks are research/round-trip geometry, not a Metal-serving claim.
RATE_GEOMETRY: dict[str, tuple[int, int, int]] = {
    "4": (4, 1, 256), "3": (3, 1, 256), "2": (2, 1, 256),
    "1": (1, 1, 256), "0.8": (3, 5, 256), "0.55": (1, 3, 256),
    "0.5": (1, 4, 512), "0.33": (1, 8, 1024),
    "0.25": (1, 16, 2048), "0.1": (1, 32, 8192),
}
QUANTIZABLE_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)
METADATA_NAMES = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "merges.txt",
    "vocab.json", "tokenizer.model", "added_tokens.json",
    "model.safetensors.index.json",
)
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_SHARDS = 128
SHA_RE = re.compile(r"[0-9a-f]{64}")
_STOP_REQUESTED = False


class LadderError(RuntimeError):
    pass


def _load_module(name: str, path: Path) -> ModuleType:
    if not path.is_file() or path.is_symlink():
        raise LadderError(f"required helper is missing or symlinked: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LadderError(f"cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_module("doctor_v5_pass_b_worker_ladder_helper", BASE_HELPER_PATH)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES or path.is_symlink():
            raise LadderError(f"invalid JSON file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LadderError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LadderError(f"JSON root is not an object: {path}")
    return value


def _workspace_path(raw: Any, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise LadderError("path must be a non-empty string")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(ROOT.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise LadderError(f"path is missing or outside workspace: {raw!r}") from exc
    if must_exist and candidate.is_symlink():
        raise LadderError(f"symlinked input is forbidden: {raw!r}")
    return resolved


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = BASE._hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _install_signals() -> None:
    def handler(_signum: int, _frame: Any) -> None:
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _rate_geometry(rate_id: str) -> dict[str, Any]:
    if rate_id not in CANONICAL_RATES:
        raise LadderError(f"unknown canonical rate: {rate_id}")
    target = CANONICAL_RATES[rate_id]
    bits, dim, block_len = RATE_GEOMETRY[rate_id]
    nominal = Fraction(bits, dim)
    if nominal > target:
        raise AssertionError(f"reviewed candidate {rate_id} exceeds its payload ceiling")
    return {
        "rate_id": rate_id, "target_fraction": [target.numerator, target.denominator],
        "target_bpw": float(target),
        "artifact_mode": "packed_scalar_control" if dim == 1 else "packed_vector_control",
        "symbol_bits": bits, "vector_dim": dim,
        "block_len": block_len, "tensor_scope": "all-2d",
        "adaptive_scales": rate_id != "0.1",
        "nominal_payload_fraction": [nominal.numerator, nominal.denominator],
        "exact_nominal_match": nominal == target, "candidate_within_payload_ceiling": True,
        "packed_output_supported": True,
        "physical_target_status": "must_measure_full_model_payload",
    }


def capabilities() -> dict[str, Any]:
    rates = [_rate_geometry(rate) for rate in CANONICAL_RATES]
    for row in rates:
        row["all_in_model_target_supported"] = "candidate_only_until_measured"
        row["all_in_blocker"] = (
            "nominal k/d excludes STR2/SDSC/SDSQ framing and lossless non-2-D tensors; "
            "the physical target is unresolved until the complete bundle is measured"
        )
    return {
        "schema": CAPABILITY_SCHEMA,
        "model_family": MODEL_FAMILY,
        "labels": list(SUPPORTED_LABELS),
        "rates": rates,
        "evaluation": {
            "resident_labels": list(RESIDENT_LABELS),
            "32B/72B": (
                "packed codec artifacts supported; even BF16 reconstruction + resident HF "
                "quality evaluation is disk/RAM-gated"
            ),
            "dense_reconstructions": "ephemeral and deleted after evaluation",
        },
        "doctor_hooks": {
            "none": {"supported": True},
            "lora_kd": {"supported": False, "blocker": (
                "doctor.py lora requires one dense base plus resident teacher and is not yet "
                "factored as a hash-bound multi-shard dependent-cell adapter")},
            "blockwise_qat": {"supported": False, "blocker": (
                "current blockwise entrypoint assumes a single resident source and emits no "
                "multi-shard exact-resume artifact")},
            "strand_hessian": {"supported": False, "blocker": (
                "current strand Doctor uses fixed scratch paths and lacks a source-bound shard ABI")},
        },
        "gpt_oss_120b": {
            "supported": False,
            "required_adapter_family": "gpt-oss-moe-mxfp4",
            "requirements": [
                "direct MXFP4 blocks+scales transcoder or a source-bound dequantization stage",
                "semantic mapping for fused qkv/out tensors and expert mlp1/mlp2 block tensors",
                "logical-parameter accounting distinct from serialized U8 element accounting",
                "router-aware total/active expert allocation and calibration",
                "per-expert out-of-core streaming without a dense 120B reconstruction",
                "GPT-OSS runtime/evaluator and tokenizer bindings independent of Qwen2.5",
            ],
        },
        "claims": {"quality": False, "dominance": False, "source_deletion": False},
    }


def _validate_campaign(value: Any, label: str, rate_id: str,
                       branch: str = "codec_control") -> None:
    if not isinstance(value, dict) or set(value) != {
            "cell_id", "cell_identity_sha256", "branch", "target_rate_id",
            "target_rate_bpw", "label"}:
        raise LadderError("campaign_binding keys are invalid")
    if not isinstance(value["cell_id"], str) or not value["cell_id"]:
        raise LadderError("campaign cell_id is invalid")
    if not _is_sha(value["cell_identity_sha256"]):
        raise LadderError("campaign cell identity is invalid")
    if value["branch"] != branch or value["label"] != label \
            or value["target_rate_id"] != rate_id:
        raise LadderError("campaign binding differs from this ladder cell")
    target = CANONICAL_RATES[rate_id]
    observed = value["target_rate_bpw"]
    if isinstance(observed, bool) or not isinstance(observed, (int, float)) \
            or Fraction(str(observed)) != target:
        raise LadderError("campaign target_rate_bpw is not the exact canonical decimal")


def treatment_recipe(branch: str, rate_id: str) -> dict[str, Any]:
    """Return the frozen, production-decodable treatment recipe for one cell.

    These are re-encoding experiments, not asserted quality improvements.  The
    conditional lane deliberately executes a sparse residual proxy and records a
    negative activation-conditioning conclusion because no activation-gated
    production decoder exists yet.  That makes the cell executable and useful
    without relabeling the missing runtime as a successful conditional Doctor.
    """
    if branch not in TREATMENT_BRANCHES:
        raise LadderError(f"unknown treatment branch: {branch}")
    geometry = _rate_geometry(rate_id)
    vector = geometry["vector_dim"] > 1
    if branch == "doctor_static":
        return {
            "recipe_id": ("static_learned_lut_reencode_v1" if vector
                          else "static_sparse_error_reencode_v1"),
            "protocol_class": "executable_static_repair",
            "learned_codebook": vector,
            "outlier_channel_pct": 0.0 if vector else 2.0,
            "outlier_bits": 8, "c2f_outl": not vector,
            "interpretation": (
                "learned tensor-local vector LUT with guarded SSE fallback" if vector else
                "deterministic high-magnitude sparse residual preservation"
            ),
            "activation_conditioned_runtime_claimed": False,
        }
    if branch == "doctor_conditional":
        return {
            "recipe_id": "conditional_sparse_proxy_negative_protocol_v1",
            "protocol_class": "executable_negative_conditional_protocol",
            "learned_codebook": False,
            "outlier_channel_pct": 0.25,
            "outlier_bits": 8, "c2f_outl": True,
            "interpretation": (
                "production-decodable sparse residual proxy; activation-conditioned "
                "selection is explicitly absent and therefore receives no conditional claim"
            ),
            "activation_conditioned_runtime_claimed": False,
        }
    return {
        "recipe_id": ("full_learned_lut_sparse_composition_v1" if vector
                      else "full_scalar_sparse_composition_v1"),
        "protocol_class": "executable_full_composition",
        "learned_codebook": vector,
        "outlier_channel_pct": 0.5 if vector else 3.0,
        "outlier_bits": 8, "c2f_outl": True,
        "interpretation": (
            "learned tensor-local LUT plus sparse residual composition" if vector else
            "scalar trellis plus expanded sparse residual and compressed side-information"
        ),
        "activation_conditioned_runtime_claimed": False,
    }


def _validate_request(path: Path) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    request = _load_json(path)
    expected = {
        "schema", "request_id", "label", "model_family", "campaign_binding",
        "codec", "source", "parameter_manifest", "execution", "evaluation",
        "doctor_hook", "resources", "output_root", "evidence_policy",
    }
    if set(request) != expected or request.get("schema") != REQUEST_SCHEMA:
        raise LadderError("internal request schema/keys are invalid")
    label = request.get("label")
    if label not in SUPPORTED_LABELS or request.get("model_family") != MODEL_FAMILY:
        if label == "120B":
            raise LadderError("120B GPT-OSS requires the separate gpt-oss-moe-mxfp4 adapter")
        raise LadderError("worker accepts only reviewed Qwen2.5 dense ladder labels")
    if not isinstance(request.get("request_id"), str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9._-]{7,159}", request["request_id"]):
        raise LadderError("request_id is invalid")

    campaign = request.get("campaign_binding")
    branch = campaign.get("branch") if isinstance(campaign, dict) else None
    if branch not in ("codec_control", *TREATMENT_BRANCHES):
        raise LadderError("campaign branch is not executable by the Qwen ladder worker")

    codec = request.get("codec")
    if not isinstance(codec, dict) or set(codec) != {
            "rate_id", "artifact_mode", "symbol_bits", "vector_dim", "quality",
            "rht_cols", "outlier_channel_pct", "outlier_bits", "sdsq_sideinfo",
            "c2f_outl", "ragged_v2", "allow_over_ceiling_control", "tensor_scope",
            "block_len", "learned_codebook", "adaptive_scales"}:
        raise LadderError("codec keys are invalid")
    rate_id = codec.get("rate_id")
    geometry = _rate_geometry(rate_id)
    if codec["artifact_mode"] != geometry["artifact_mode"] \
            or codec["symbol_bits"] != geometry["symbol_bits"] \
            or codec["vector_dim"] != geometry["vector_dim"] \
            or codec["block_len"] != geometry["block_len"] \
            or codec["tensor_scope"] != "all-2d" \
            or codec["adaptive_scales"] != geometry["adaptive_scales"]:
        raise LadderError("codec geometry differs from reviewed canonical mapping")
    if not isinstance(codec["learned_codebook"], bool):
        raise LadderError("learned_codebook must be boolean")
    if codec["quality"] is not True or codec["rht_cols"] is not True \
            or codec["ragged_v2"] is not True:
        raise LadderError("quality/rht-cols/ragged-v2 controls are mandatory")
    if codec["allow_over_ceiling_control"] is not True:
        raise LadderError("candidate codec must be explicitly admitted as an unproven-ceiling control")
    pct = codec["outlier_channel_pct"]
    if isinstance(pct, bool) or not isinstance(pct, (int, float)) or pct < 0 or pct > 10:
        raise LadderError("outlier_channel_pct must be in [0,10]")
    if not isinstance(codec["outlier_bits"], int) or isinstance(codec["outlier_bits"], bool) \
            or not 2 <= codec["outlier_bits"] <= 16:
        raise LadderError("outlier_bits must be in [2,16]")
    if codec["sdsq_sideinfo"] is not True:
        raise LadderError("SDSQ side information is mandatory")
    if branch == "codec_control":
        if codec["learned_codebook"] is not False:
            raise LadderError("packed controls use the frozen deterministic LUT")
        if codec["artifact_mode"] == "packed_vector_control":
            if pct != 0 or codec["c2f_outl"]:
                raise LadderError("packed vector control disables OUTL/C2F")
        elif codec["c2f_outl"] is not True or pct <= 0:
            raise LadderError("packed scalar control requires reviewed OUTL/C2F settings")
    else:
        recipe = treatment_recipe(branch, rate_id)
        for key in ("learned_codebook", "outlier_channel_pct", "outlier_bits", "c2f_outl"):
            if codec[key] != recipe[key]:
                raise LadderError(f"treatment codec differs from frozen recipe: {key}")
        if codec["learned_codebook"] and codec["vector_dim"] <= 1:
            raise LadderError("learned codebook treatment requires vector geometry")
    _validate_campaign(request.get("campaign_binding"), label, rate_id, branch)

    evaluation = request.get("evaluation")
    if not isinstance(evaluation, dict) or set(evaluation) != {
            "mode", "retain_dense_reconstruction"} \
            or evaluation.get("mode") not in {"resident", "deferred"} \
            or evaluation.get("retain_dense_reconstruction") is not False:
        raise LadderError("evaluation policy is invalid")
    if label not in RESIDENT_LABELS and evaluation["mode"] != "deferred":
        raise LadderError(
            f"{label} resident evaluation is unsupported: dense BF16 reconstruction would "
            "violate the 96-GiB/150-GB-reserve lifecycle gate"
        )
    doctor = request.get("doctor_hook")
    if branch == "codec_control":
        if doctor != {"method": "none", "dependent_cells_require_packed_base": True}:
            raise LadderError("codec control Doctor hook is invalid")
    else:
        expected_doctor = {
            "method", "operation", "recipe_id", "protocol_class",
            "dependency_evidence", "quality_selection_permitted",
        }
        if not isinstance(doctor, dict) or set(doctor) != expected_doctor \
                or doctor.get("method") != TREATMENT_METHOD \
                or doctor.get("operation") != branch \
                or doctor.get("quality_selection_permitted") is not False:
            raise LadderError("treatment Doctor hook identity/policy is invalid")
        recipe = treatment_recipe(branch, rate_id)
        if doctor.get("recipe_id") != recipe["recipe_id"] \
                or doctor.get("protocol_class") != recipe["protocol_class"]:
            raise LadderError("treatment Doctor hook differs from frozen recipe")
        evidence = doctor.get("dependency_evidence")
        if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256", "count"} \
                or not _is_sha(evidence.get("sha256")) \
                or evidence.get("count") != TREATMENT_BRANCHES.index(branch) + 1:
            raise LadderError("treatment dependency evidence binding is invalid")
        evidence_path = _workspace_path(evidence.get("path"))
        if BASE._hash_file(evidence_path)[0] != evidence["sha256"]:
            raise LadderError("treatment dependency evidence identity mismatch")
        evidence_doc = _load_json(evidence_path)
        if evidence_doc.get("schema") != DEPENDENCY_EVIDENCE_SCHEMA \
                or evidence_doc.get("campaign_binding") != campaign \
                or evidence_doc.get("recipe_id") != recipe["recipe_id"] \
                or not isinstance(evidence_doc.get("dependencies"), list) \
                or len(evidence_doc["dependencies"]) != evidence["count"]:
            raise LadderError("treatment dependency evidence content is invalid")
    evidence = request.get("evidence_policy")
    if evidence != {
            "class": "provisional_engineering_evidence",
            "quality_claims_permitted": False, "dominance_claims_permitted": False,
            "source_deletion_permitted": False}:
        raise LadderError("evidence policy is invalid")

    source = request.get("source")
    if not isinstance(source, dict) or set(source) != {
            "model_dir", "census_path", "census_sha256", "source_manifest_sha256", "shards"}:
        raise LadderError("source keys are invalid")
    model_dir = _workspace_path(source["model_dir"])
    census_path = _workspace_path(source["census_path"])
    if not _is_sha(source["census_sha256"]) or not _is_sha(source["source_manifest_sha256"]):
        raise LadderError("source hashes are invalid")
    census_sha, _ = BASE._hash_file(census_path)
    if census_sha != source["census_sha256"]:
        raise LadderError("census file identity mismatch")
    census = _load_json(census_path)
    if census.get("status") != "complete" or census.get("label") != label \
            or census.get("source", {}).get("model_dir") != str(model_dir) \
            or census.get("source", {}).get("source_manifest_sha256") != source["source_manifest_sha256"]:
        raise LadderError("source differs from completed Pass-A census")

    shards = source.get("shards")
    if not isinstance(shards, list) or not 1 <= len(shards) <= MAX_SHARDS:
        raise LadderError("source shard inventory is invalid")
    verified: list[dict[str, Any]] = []
    census_shards = census.get("source", {}).get("shards")
    if not isinstance(census_shards, list) or len(census_shards) != len(shards):
        raise LadderError("source shard count differs from census")
    for ordinal, (row, census_row) in enumerate(zip(shards, census_shards)):
        if not isinstance(row, dict) or set(row) != {
                "ordinal", "name", "path", "sha256", "bytes"} \
                or row.get("ordinal") != ordinal or census_row.get("ordinal") != ordinal:
            raise LadderError(f"source shard {ordinal} keys/order invalid")
        if row.get("name") != census_row.get("name") or row.get("sha256") != census_row.get(
                "file_sha256") or row.get("bytes") != census_row.get("bytes"):
            raise LadderError(f"source shard {ordinal} differs from census")
        shard = _workspace_path(row["path"])
        if model_dir not in shard.parents or shard.name != Path(row["name"]).name:
            raise LadderError(f"source shard {ordinal} path is unsafe")
        digest, size = BASE._hash_file(shard)
        if digest != row["sha256"] or size != row["bytes"]:
            raise LadderError(f"source shard {ordinal} live identity mismatch")
        verified.append({**row, "path": shard})

    parameter = request.get("parameter_manifest")
    if not isinstance(parameter, dict) or set(parameter) != {"path", "sha256"} \
            or not _is_sha(parameter.get("sha256")):
        raise LadderError("parameter manifest binding is invalid")
    parameter_path = _workspace_path(parameter["path"])
    if BASE._hash_file(parameter_path)[0] != parameter["sha256"]:
        raise LadderError("parameter manifest identity mismatch")
    manifest = _load_json(parameter_path)
    stored = manifest.get("parameter_authority", {}).get(
        "exact_distinct_stored_parameter_count")
    if not isinstance(stored, int) or isinstance(stored, bool) or stored <= 0:
        raise LadderError("parameter manifest lacks authoritative exact stored count")
    if manifest.get("source_manifest_sha256") != source["source_manifest_sha256"]:
        raise LadderError("parameter manifest source identity mismatch")

    execution = request.get("execution")
    required_exec = {
        "worker_path", "worker_sha256", "base_helper_path", "base_helper_sha256",
        "quantizer_path", "quantizer_sha256", "attestor_path", "attestor_sha256",
        "decoder_path", "decoder_sha256", "evaluator_path", "evaluator_sha256",
        "python_path", "python_sha256", "threads",
    }
    if not isinstance(execution, dict) or set(execution) != required_exec:
        raise LadderError("execution bindings are invalid")
    for prefix in ("worker", "base_helper", "quantizer", "attestor", "decoder", "evaluator"):
        tool = _workspace_path(execution[f"{prefix}_path"])
        if BASE._hash_file(tool)[0] != execution[f"{prefix}_sha256"]:
            raise LadderError(f"{prefix} identity mismatch")
    if _workspace_path(execution["worker_path"]) != Path(__file__).resolve() \
            or _workspace_path(execution["base_helper_path"]) != BASE_HELPER_PATH.resolve():
        raise LadderError("worker/helper path binding differs from loaded implementation")
    python = Path(execution["python_path"])
    try:
        python_resolved = python.resolve(strict=True)
        running = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise LadderError(f"cannot resolve Python binding: {exc}") from exc
    if python.is_symlink() or python_resolved != running \
            or BASE._hash_file(python_resolved)[0] != execution["python_sha256"]:
        raise LadderError("Python interpreter identity mismatch")
    if not isinstance(execution["threads"], int) or isinstance(execution["threads"], bool) \
            or not 1 <= execution["threads"] <= 32:
        raise LadderError("threads must be in [1,32]")
    resources = request.get("resources")
    if not isinstance(resources, dict) or set(resources) != {
            "disk_reserve_bytes", "scratch_budget_bytes"} \
            or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                   for v in resources.values()) \
            or resources["disk_reserve_bytes"] < BASE.DEFAULT_DISK_RESERVE_BYTES:
        raise LadderError("resource admission values are invalid")

    output = _workspace_path(request["output_root"], must_exist=False)
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise LadderError("output_root is invalid")
    return request, BASE._hash_file(path)[0], verified


def _quantized_by_scope(name: str, shape: tuple[int, ...], scope: str) -> bool:
    if len(shape) != 2:
        return False
    if scope == "all-2d":
        return True
    if scope == "linear":
        return name.endswith(QUANTIZABLE_SUFFIXES)
    raise LadderError(f"unsupported tensor scope: {scope}")


def _tensor_stats(shard: Path, scope: str) -> dict[str, Any]:
    from safetensors import safe_open
    quantized = passthrough = 0
    qt = pt = 0
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            shape = tuple(int(v) for v in handle.get_slice(name).get_shape())
            count = math.prod(shape)
            if _quantized_by_scope(name, shape, scope):
                quantized += count; qt += 1
            else:
                passthrough += count; pt += 1
    return {"stored_parameters": quantized + passthrough,
            "quantized_parameters": quantized, "passthrough_parameters": passthrough,
            "quantized_tensors": qt, "passthrough_tensors": pt}


_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}


def _stream_passthrough(source: Path, destination: Path, scope: str) -> dict[str, Any]:
    """Write pass-through tensors while holding at most one source tensor."""
    from safetensors import safe_open

    specs: list[tuple[str, str, tuple[int, ...], int]] = []
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            view = handle.get_slice(name)
            shape = tuple(int(v) for v in view.get_shape())
            if _quantized_by_scope(name, shape, scope):
                continue
            dtype = str(view.get_dtype())
            if dtype not in _DTYPE_BYTES:
                raise LadderError(f"unsupported passthrough dtype {dtype}: {name}")
            specs.append((name, dtype, shape, math.prod(shape) * _DTYPE_BYTES[dtype]))
    if not specs:
        return {"tensor_count": 0, "parameter_count": 0, "artifact": None}

    offset = 0
    header: dict[str, Any] = {"__metadata__": {
        "hawking_schema": "hawking.doctor_v5_streamed_passthrough.v1",
        "artifact_class": "lossless_passthrough_state",
    }}
    for name, dtype, shape, size in specs:
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    try:
        with tmp.open("xb", buffering=0) as output:
            output.write(len(encoded).to_bytes(8, "little")); output.write(encoded)
            with safe_open(str(source), framework="pt", device="cpu") as handle:
                for name, _dtype, _shape, size in specs:
                    tensor = handle.get_tensor(name).contiguous().view(-1).view(__import__("torch").uint8)
                    raw = memoryview(tensor.numpy()).cast("B")
                    written = 0
                    while written < len(raw):
                        n = output.write(raw[written:written + 8 * 1024 * 1024])
                        if n is None or n <= 0:
                            raise OSError("short passthrough write")
                        written += n
                    if written != size:
                        raise LadderError(f"passthrough byte count mismatch: {name}")
                    del tensor, raw
            output.flush(); os.fsync(output.fileno())
        os.replace(tmp, destination); BASE._fsync_dir(destination.parent)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass
    return {"tensor_count": len(specs),
            "parameter_count": sum(math.prod(row[2]) for row in specs),
            "artifact": _artifact(destination)}


def _paths(output: Path, shard_count: int) -> dict[str, Any]:
    return {
        "root": output, "checkpoint": output / "checkpoint.json",
        "receipt": output / "execution_receipt.json", "bundle": output / "bundle",
        "logs": output / "logs", "evaluation": output / "evaluation",
        "manifest": output / "bundle/manifest.json",
        "override_manifest": output / "evaluation/override_manifest.json",
        "ephemeral_receipt": output / "evaluation/ephemeral_cleanup.json",
        "shards": [{
            "packed": output / f"bundle/shards/{i:05d}.strand",
            "passthrough": output / f"bundle/shards/{i:05d}.passthrough.safetensors",
            "reconstruction": output / f"evaluation/reconstruction/{i:05d}.safetensors",
            "oracle_sidecar": output / f"evaluation/reconstruction/{i:05d}.safetensors.json",
            "encode_log": output / f"logs/encode-{i:05d}.log",
            "attest_log": output / f"logs/attest-{i:05d}.log",
            "decode_log": output / f"logs/decode-{i:05d}.log",
        } for i in range(shard_count)],
        "baseline_ppl": output / "evaluation/baseline_ppl.json",
        "recon_ppl": output / "evaluation/reconstruction_ppl.json",
        "baseline_cap": output / "evaluation/baseline_capability.json",
        "recon_cap": output / "evaluation/reconstruction_capability.json",
    }


def _plan(request: dict[str, Any], shard_stats: list[dict[str, Any]]) -> list[str]:
    resident = request["evaluation"]["mode"] == "resident"
    units = ["preflight", "metadata"]
    for i, stats in enumerate(shard_stats):
        units.append(f"passthrough:{i:05d}")
        units.append(f"encode:{i:05d}")
        units.append(f"attest:{i:05d}")
        if resident and stats["quantized_tensors"]:
            units.append(f"decode:{i:05d}")
    units.append("bundle_manifest")
    if resident:
        units.extend(["override_manifest", "baseline_ppl", "reconstruction_ppl",
                      "baseline_capability", "reconstruction_capability"])
    if resident:
        units.append("ephemeral_cleanup")
    units.append("receipt")
    return units


def _initial_checkpoint(request_sha: str, plan: list[str]) -> dict[str, Any]:
    return {"schema": CHECKPOINT_SCHEMA, "request_sha256": request_sha,
            "created_at": _now(), "updated_at": _now(), "status": "running",
            "plan": plan, "completed_units": [], "units": {}, "stop_requested": False}


def _artifact_refs(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if set(value) >= {"path", "sha256", "bytes"} \
                and isinstance(value.get("path"), str) \
                and _is_sha(value.get("sha256")) \
                and isinstance(value.get("bytes"), int) \
                and not isinstance(value.get("bytes"), bool):
            rows.append({"path": value["path"], "sha256": value["sha256"],
                         "bytes": value["bytes"]})
        for child in value.values():
            rows.extend(_artifact_refs(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_artifact_refs(child))
    return rows


def _validate_completed_artifacts(cp: dict[str, Any], paths: dict[str, Any],
                                  stats: list[dict[str, Any]]) -> None:
    units = cp.get("units")
    done = cp["completed_units"]
    if not isinstance(units, dict) or set(units) != set(done) \
            or any(not isinstance(units.get(unit), dict) for unit in done):
        raise LadderError("checkpoint unit evidence is incomplete or noncanonical")
    deleted: set[tuple[str, str, int]] = set()
    if "ephemeral_cleanup" in done:
        cleanup = _load_json(paths["ephemeral_receipt"])
        if cleanup.get("schema") != "hawking.doctor_v5_ephemeral_cleanup.v1" \
                or cleanup.get("worker_owned_only") is not True \
                or cleanup.get("source_files_deleted") is not False \
                or not isinstance(cleanup.get("deleted_artifacts"), list):
            raise LadderError("ephemeral cleanup evidence is invalid on resume")
        for row in cleanup["deleted_artifacts"]:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str) \
                    or not _is_sha(row.get("sha256")) \
                    or not isinstance(row.get("bytes"), int):
                raise LadderError("ephemeral cleanup deletion identity is invalid")
            deleted.add((row["path"], row["sha256"], row["bytes"]))
            if Path(row["path"]).exists():
                raise LadderError("ephemeral artifact reappeared after cleanup")
    observed: set[tuple[str, str, int]] = set()
    for unit in done:
        for row in _artifact_refs(units[unit]):
            identity = (row["path"], row["sha256"], row["bytes"])
            if identity in observed:
                continue
            observed.add(identity)
            if identity in deleted:
                continue
            path = _workspace_path(row["path"])
            digest, size = BASE._hash_file(path)
            if digest != row["sha256"] or size != row["bytes"]:
                raise LadderError(f"completed checkpoint artifact changed: {path}")
    for index, shard_stats in enumerate(stats):
        sp = paths["shards"][index]
        encode = f"encode:{index:05d}"
        attest = f"attest:{index:05d}"
        passthrough = f"passthrough:{index:05d}"
        if passthrough in done and not sp["passthrough"].is_file():
            raise LadderError("completed passthrough shard is missing on resume")
        if shard_stats["quantized_tensors"] > 0 and encode in done \
                and not sp["packed"].is_file():
            raise LadderError("completed packed shard is missing on resume")
        if shard_stats["quantized_tensors"] > 0 and attest in done:
            evidence = units[attest]
            archive = evidence.get("archive") if isinstance(evidence, dict) else None
            if not isinstance(archive, dict) or archive.get("path") \
                    != str(sp["packed"].resolve()):
                raise LadderError("attested checkpoint shard binding is missing")


def _checkpoint(path: Path, request_sha: str, plan: list[str],
                paths: dict[str, Any], stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.exists():
        return _initial_checkpoint(request_sha, plan)
    cp = _load_json(path)
    if cp.get("schema") != CHECKPOINT_SCHEMA or cp.get("request_sha256") != request_sha \
            or cp.get("plan") != plan:
        raise LadderError("checkpoint request/plan identity mismatch")
    done = cp.get("completed_units")
    if not isinstance(done, list) or done != plan[:len(done)] or len(done) != len(set(done)):
        raise LadderError("checkpoint units are not a strict plan prefix")
    _validate_completed_artifacts(cp, paths, stats)
    return cp


def _save_cp(path: Path, cp: dict[str, Any]) -> None:
    cp["updated_at"] = _now(); cp["stop_requested"] = bool(_STOP_REQUESTED)
    BASE._atomic_json(path, cp)


def _finish(paths: dict[str, Any], cp: dict[str, Any], unit: str,
            evidence: dict[str, Any]) -> None:
    expected = cp["plan"][len(cp["completed_units"])]
    if unit != expected:
        raise LadderError(f"unit order violation: expected {expected}, got {unit}")
    cp["units"][unit] = {"completed_at": _now(), **evidence}
    cp["completed_units"].append(unit); _save_cp(paths["checkpoint"], cp)
    if _STOP_REQUESTED and unit != "receipt":
        cp["status"] = "checkpointed-stop"; _save_cp(paths["checkpoint"], cp)
        raise LadderError("stop requested; exited at durable shard/phase boundary")


def _done(cp: dict[str, Any], unit: str) -> bool:
    return unit in cp["completed_units"]


def _copy_metadata(model_dir: Path, destination: Path,
                   census: dict[str, Any]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    destination.mkdir(parents=True, exist_ok=True)
    rows = census.get("source", {}).get("auxiliary_files")
    if not isinstance(rows, list):
        raise LadderError("census has no auxiliary-file identity inventory")
    identities = {row.get("name"): row for row in rows if isinstance(row, dict)}
    for name in METADATA_NAMES:
        source = model_dir / name
        if not source.is_file() or source.is_symlink():
            continue
        identity = identities.get(name)
        digest, size = BASE._hash_file(source)
        if not isinstance(identity, dict) or digest != identity.get("sha256") \
                or size != identity.get("bytes"):
            raise LadderError(f"metadata differs from completed census: {name}")
        target = destination / name
        BASE._atomic_bytes(target, source.read_bytes())
        copied.append(_artifact(target))
    if not (destination / "config.json").is_file():
        raise LadderError("model metadata is missing config.json")
    return copied


def _quantizer_argv(request: dict[str, Any], source: Path, output: Path) -> list[str]:
    codec, execution = request["codec"], request["execution"]
    argv = [execution["quantizer_path"], "--in", str(source), "--bits",
            str(codec["symbol_bits"]), "--threads", str(execution["threads"]),
            "--quality", "--rht-cols", "--ragged-v2", "--tensor-scope",
            codec["tensor_scope"], "--block-len", str(codec["block_len"])]
    if codec["vector_dim"] > 1:
        argv += ["--vec-dim", str(codec["vector_dim"])]
        if codec["learned_codebook"]:
            argv.append("--learned-codebook")
            argv += ["--encode-mem-budget-bytes",
                     str(LEARNED_ENCODE_MEM_BUDGET_BYTES)]
    if not codec["adaptive_scales"]:
        argv.append("--no-adaptive-scales")
    if codec["outlier_channel_pct"] > 0:
        argv += ["--outlier-channel", str(codec["outlier_channel_pct"]),
                 "--outlier-bits", str(codec["outlier_bits"])]
    if codec["sdsq_sideinfo"]:
        argv.append("--sdsq-sideinfo")
    if codec["c2f_outl"]:
        argv.append("--c2f-outl")
    argv += ["--packed-v2-out", str(output)]
    return argv


def _validate_decoded(source: Path, decoded: Path, *, scope: str) -> dict[str, Any]:
    from safetensors import safe_open
    with safe_open(str(source), framework="pt", device="cpu") as a, \
            safe_open(str(decoded), framework="pt", device="cpu") as b:
        expected = [name for name in a.keys()
                    if _quantized_by_scope(name, tuple(a.get_slice(name).get_shape()), scope)]
        observed = list(b.keys())
        if expected != observed:
            raise LadderError("decoded tensor inventory differs from expected source subset")
        elements = 0
        for name in expected:
            if tuple(a.get_slice(name).get_shape()) != tuple(b.get_slice(name).get_shape()):
                raise LadderError(f"decoded shape mismatch: {name}")
            elements += math.prod(tuple(a.get_slice(name).get_shape()))
    return {"tensor_count": len(expected), "parameter_count": elements,
            "artifact": _artifact(decoded)}


def _write_override_manifest(path: Path, decoded: list[tuple[int, Path]]) -> dict[str, Any]:
    rows = []
    from safetensors import safe_open
    for ordinal, shard in decoded:
        artifact = _artifact(shard)
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            tensor_count = len(list(handle.keys()))
        rows.append({"ordinal": ordinal, **artifact, "tensor_count": tensor_count})
    doc = {"schema": OVERRIDE_SCHEMA, "shards": rows}
    BASE._atomic_json(path, doc)
    return doc


def _run_eval(request: dict[str, Any], mode: str, override: Path | None,
              output: Path, log: Path, *, label: str | None = None) -> dict[str, Any]:
    argv = [request["execution"]["python_path"], request["execution"]["evaluator_path"],
            "run", "--mode", mode, "--model-dir", request["source"]["model_dir"],
            "--label", label or f"{request['label']}-{request['codec']['rate_id']}"]
    if override is not None:
        argv += ["--override-manifest", str(override)]
    env = BASE._fixed_env(); env["DOCTOR_DTYPE"] = "bfloat16"; env["DOCTOR_DEVICE"] = "cpu"
    BASE._run_logged(argv, log, env=env)
    result = BASE._last_json_line(log); BASE._atomic_json(output, result)
    return {"result": result, "artifact": _artifact(output), "log": _artifact(log)}


def _baseline_cache_identity(request: dict[str, Any], mode: str) -> dict[str, Any]:
    census = _load_json(_workspace_path(request["source"]["census_path"]))
    auxiliary = census.get("source", {}).get("auxiliary_files")
    if not isinstance(auxiliary, list):
        raise LadderError("source census has no auxiliary identity inventory")
    metadata = []
    for row in auxiliary:
        if not isinstance(row, dict) or row.get("name") not in METADATA_NAMES:
            continue
        if not _is_sha(row.get("sha256")) or not isinstance(row.get("bytes"), int):
            raise LadderError("baseline cache metadata identity is invalid")
        metadata.append({"name": row["name"], "sha256": row["sha256"],
                         "bytes": row["bytes"]})
    metadata.sort(key=lambda row: row["name"])
    if not any(row["name"] == "config.json" for row in metadata) \
            or not any(row["name"].startswith("tokenizer") \
                       or row["name"] in {"tokenizer.model", "vocab.json", "merges.txt"}
                       for row in metadata):
        raise LadderError("baseline cache requires config and tokenizer identities")
    corpus: dict[str, Any] = {"kind": "evaluator_embedded_fixture"}
    raw_text = os.environ.get("PPL_TEXT")
    if mode == "ppl" and raw_text and Path(raw_text).is_file():
        text_path = Path(raw_text).resolve(strict=True)
        digest, size = BASE._hash_file(text_path)
        corpus = {"kind": "external_ppl_text", "path": str(text_path),
                  "sha256": digest, "bytes": size}
    return {
        "schema": "hawking.doctor_v5_shared_baseline_identity.v1",
        "model_family": request["model_family"], "label": request["label"],
        "model_dir": request["source"]["model_dir"],
        "source_manifest_sha256": request["source"]["source_manifest_sha256"],
        "parameter_manifest_sha256": request["parameter_manifest"]["sha256"],
        "evaluator_sha256": request["execution"]["evaluator_sha256"],
        "python_sha256": request["execution"]["python_sha256"],
        "mode": mode, "dtype": "bfloat16", "device": "cpu",
        "metadata": metadata, "evaluation_corpus": corpus,
    }


def _validate_baseline_cache(receipt: dict[str, Any], identity: dict[str, Any],
                             result_path: Path, log_path: Path) -> None:
    expected = {"schema", "cache_key_sha256", "identity", "result", "log",
                "created_at", "receipt_sha256"}
    key = _sha_value(identity)
    if set(receipt) != expected or receipt.get("schema") != BASELINE_CACHE_SCHEMA \
            or receipt.get("cache_key_sha256") != key \
            or receipt.get("identity") != identity:
        raise LadderError("shared baseline cache receipt identity is invalid")
    for name, path in (("result", result_path), ("log", log_path)):
        expected_artifact = receipt.get(name)
        if not isinstance(expected_artifact, dict) or set(expected_artifact) != {
                "path", "sha256", "bytes"}:
            raise LadderError("shared baseline cache artifact binding is invalid")
        artifact = _artifact(path)
        if artifact != expected_artifact:
            raise LadderError("shared baseline cache live artifact identity mismatch")
    if receipt.get("receipt_sha256") != _sha_value({
            key: value for key, value in receipt.items() if key != "receipt_sha256"}):
        raise LadderError("shared baseline cache receipt hash mismatch")
    result = _load_json(result_path)
    if result.get("mode") != identity["mode"] or result.get("model") != identity["model_dir"] \
            or result.get("override_manifest") is not None \
            or result.get("label") != f"{identity['label']}-shared-baseline":
        raise LadderError("shared baseline cache result semantics are invalid")


def _run_baseline_eval_cached(request: dict[str, Any], mode: str,
                              output: Path, log: Path) -> dict[str, Any]:
    identity = _baseline_cache_identity(request, mode)
    key = _sha_value(identity)
    cache_dir = BASELINE_CACHE_ROOT / request["label"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir.is_symlink():
        raise LadderError("shared baseline cache directory may not be symlinked")
    result_path = cache_dir / f"{mode}-{key}.json"
    log_path = cache_dir / f"{mode}-{key}.log"
    receipt_path = cache_dir / f"{mode}-{key}.receipt.json"
    lock_path = cache_dir / f"{mode}-{key}.lock"
    hit = False
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        present = [result_path.exists(), log_path.exists(), receipt_path.exists()]
        if any(present) and not all(present):
            raise LadderError("shared baseline cache is partially committed")
        if all(present):
            receipt = _load_json(receipt_path)
            _validate_baseline_cache(receipt, identity, result_path, log_path)
            hit = True
        else:
            tmp_result = result_path.with_name(f".{result_path.name}.partial.{os.getpid()}")
            tmp_log = log_path.with_name(f".{log_path.name}.partial.{os.getpid()}")
            try:
                _run_eval(request, mode, None, tmp_result, tmp_log,
                          label=f"{request['label']}-shared-baseline")
                BASE._commit_generated(tmp_result, result_path)
                BASE._commit_generated(tmp_log, log_path)
            finally:
                for path in (tmp_result, tmp_log):
                    try: path.unlink()
                    except FileNotFoundError: pass
            receipt = {
                "schema": BASELINE_CACHE_SCHEMA, "cache_key_sha256": key,
                "identity": identity, "result": _artifact(result_path),
                "log": _artifact(log_path), "created_at": _now(),
            }
            receipt["receipt_sha256"] = _sha_value(receipt)
            BASE._atomic_json(receipt_path, receipt)
            _validate_baseline_cache(receipt, identity, result_path, log_path)
        cached_result = _load_json(result_path)
        BASE._atomic_json(output, cached_result)
        BASE._atomic_bytes(log, (_canonical({
            "shared_baseline_cache": {"cache_key_sha256": key, "cache_hit": hit,
                                      "receipt": _artifact(receipt_path)}}) + b"\n"))
    return {
        "result": cached_result, "artifact": _artifact(output), "log": _artifact(log),
        "baseline_cache": {
            "schema": BASELINE_CACHE_SCHEMA, "cache_key_sha256": key,
            "cache_hit": hit, "identity": identity,
            "receipt": _artifact(receipt_path),
            "reconstruction_reuse_permitted": False,
        },
    }


def _bundle_manifest(request: dict[str, Any], paths: dict[str, Any],
                     cp: dict[str, Any], totals: dict[str, int]) -> dict[str, Any]:
    metadata = [_artifact(path) for path in sorted(paths["bundle"].iterdir())
                if path.is_file() and path.name != paths["manifest"].name]
    shard_rows: list[dict[str, Any]] = []
    payload_bytes = packed_bytes = passthrough_bytes = 0
    for i, sp in enumerate(paths["shards"]):
        row: dict[str, Any] = {"ordinal": i}
        shard_stats = cp["units"][f"preflight"]["shard_stats"][i]
        packed_required = shard_stats["quantized_tensors"] > 0
        if packed_required and not sp["packed"].is_file():
            raise LadderError(f"bundle manifest refuses missing packed shard {i}")
        if not sp["passthrough"].is_file():
            raise LadderError(f"bundle manifest refuses missing passthrough shard {i}")
        if sp["packed"].is_file():
            row["packed"] = _artifact(sp["packed"]); packed_bytes += row["packed"]["bytes"]
        row["passthrough"] = _artifact(sp["passthrough"])
        passthrough_bytes += row["passthrough"]["bytes"]
        shard_rows.append(row)
    if len(shard_rows) != len(paths["shards"]):
        raise LadderError("bundle manifest shard cardinality mismatch")
    payload_bytes = packed_bytes + passthrough_bytes
    stored, quantized = totals["stored_parameters"], totals["quantized_parameters"]
    target = float(CANONICAL_RATES[request["codec"]["rate_id"]])
    model_bpw = payload_bytes * 8 / stored
    packed_2d_bpw = packed_bytes * 8 / quantized if quantized else None
    metadata_bytes = sum(row["bytes"] for row in metadata)
    full_bundle_bytes = payload_bytes + metadata_bytes
    physical = {
        "packed_2d_tensor_bytes": packed_bytes,
        "lossless_non_2d_passthrough_bytes": passthrough_bytes,
        "model_payload_bytes": payload_bytes,
        "packed_2d_tensor_bpw": packed_2d_bpw,
        "all_in_model_payload_bpw": model_bpw,
        "metadata_bytes_excluded_from_model_bpw": metadata_bytes,
        "full_bundle_bytes": full_bundle_bytes,
        "full_bundle_bpw_over_model_parameters": full_bundle_bytes * 8 / stored,
        "target_physical_bpw": target,
        "target_met": model_bpw <= target,
        "denominator": "exact distinct serialized model parameters",
        "target_scope": "all physical model-weight payload bytes; tokenizer/config metadata excluded",
    }
    artifact_class = "attested_sharded_str2_all_2d_plus_lossless_non_2d"
    doc = {
        "schema": "hawking.doctor_v5_strand_ladder_bundle.v1", "created_at": _now(),
        "campaign_binding": request["campaign_binding"], "label": request["label"],
        "model_family": request["model_family"], "codec": request["codec"],
        "doctor_hook": request["doctor_hook"],
        "artifact_class": artifact_class, "source_manifest_sha256": request["source"][
            "source_manifest_sha256"], "parameter_accounting": totals,
        "shards": shard_rows, "metadata": metadata, "physical_accounting": physical,
        "claims": {"packed_archive_roundtrip_validated": True,
                   "runtime_inference_validated": False, "deployable": False,
                   "quality": False, "dominance": False, "source_deletion": False},
        "limitations": [
            "all-in rate includes lossless pass-through model tensors",
            "tokenizer/config metadata bytes are reported but excluded from model-parameter bpw",
            "dense reconstructions are evaluation ephemera, never retained codec payload",
        ],
    }
    BASE._atomic_json(paths["manifest"], doc)
    return doc


def _cleanup_ephemeral(paths: dict[str, Any]) -> dict[str, Any]:
    deleted: list[dict[str, Any]] = []
    for sp in paths["shards"]:
        for role in ("reconstruction", "oracle_sidecar"):
            path = sp[role]
            if path.is_file() and not path.is_symlink():
                deleted.append({"role": role, **_artifact(path)})
                path.unlink(); BASE._fsync_dir(path.parent)
    if paths["override_manifest"].is_file():
        deleted.append({"role": "override_manifest", **_artifact(paths["override_manifest"])})
        paths["override_manifest"].unlink(); BASE._fsync_dir(paths["override_manifest"].parent)
    receipt = {
        "schema": "hawking.doctor_v5_ephemeral_cleanup.v1", "completed_at": _now(),
        "worker_owned_only": True, "source_files_deleted": False,
        "deleted_artifacts": deleted,
    }
    BASE._atomic_json(paths["ephemeral_receipt"], receipt)
    return receipt


def _quality(paths: dict[str, Any], evaluation_mode: str) -> dict[str, Any]:
    if evaluation_mode != "resident":
        return {"status": "deferred", "quality_claims_permitted": False}
    bp, rp = _load_json(paths["baseline_ppl"]), _load_json(paths["recon_ppl"])
    bc, rc = _load_json(paths["baseline_cap"]), _load_json(paths["recon_cap"])
    return {
        "status": "provisional_unsealed", "ppl": {
            "baseline": bp["ppl"], "reconstruction": rp["ppl"],
            "relative_delta": rp["ppl"] / bp["ppl"] - 1.0,
        }, "capability": {
            "baseline": bc["aggregate"], "reconstruction": rc["aggregate"],
            "absolute_delta": rc["aggregate"] - bc["aggregate"],
        }, "quality_claims_permitted": False,
    }


def _execute(request_path: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    _install_signals()
    request_path = _workspace_path(str(request_path))
    request, request_sha, shards = _validate_request(request_path)
    output = _workspace_path(request["output_root"], must_exist=False)
    output.mkdir(parents=True, exist_ok=True)
    paths = _paths(output, len(shards))
    for key in ("bundle", "logs", "evaluation"):
        paths[key].mkdir(parents=True, exist_ok=True)
    (paths["bundle"] / "shards").mkdir(parents=True, exist_ok=True)
    (paths["evaluation"] / "reconstruction").mkdir(parents=True, exist_ok=True)

    stats = [_tensor_stats(row["path"], request["codec"]["tensor_scope"]) for row in shards]
    totals = {key: sum(row[key] for row in stats) for key in stats[0]}
    plan = _plan(request, stats)
    cp = _checkpoint(paths["checkpoint"], request_sha, plan, paths, stats)
    if not _done(cp, "preflight"):
        lease = BASE._validate_heavy_lease()
        sample = BASE._resource_sample(output); BASE._resource_gate(sample, request)
        parameter = _load_json(_workspace_path(request["parameter_manifest"]["path"]))
        exact = parameter["parameter_authority"]["exact_distinct_stored_parameter_count"]
        if totals["stored_parameters"] != exact:
            raise LadderError("live shard parameter count differs from parameter authority")
        target = float(CANONICAL_RATES[request["codec"]["rate_id"]])
        candidate = request["codec"]["symbol_bits"] / request["codec"]["vector_dim"]
        passthrough_floor = (totals["quantized_parameters"] * candidate
                             + totals["passthrough_parameters"] * 16) / exact
        _finish(paths, cp, "preflight", {
            "heavy_lease": lease, "resources": sample, "parameter_accounting": totals,
            "shard_stats": stats, "candidate_payload_plus_bf16_passthrough_lower_bound_bpw":
                passthrough_floor,
            "canonical_target_preflight_lower_bound_exceeded": passthrough_floor > target,
        })
    if preflight_only:
        return {"status": "preflight-complete", "checkpoint": str(paths["checkpoint"])}

    if not _done(cp, "metadata"):
        census = _load_json(_workspace_path(request["source"]["census_path"]))
        copied = _copy_metadata(_workspace_path(request["source"]["model_dir"]),
                                paths["bundle"], census)
        _finish(paths, cp, "metadata", {"artifacts": copied})

    decoded: list[tuple[int, Path]] = []
    for i, (source_row, shard_stats, sp) in enumerate(zip(shards, stats, paths["shards"])):
        source = source_row["path"]
        unit = f"passthrough:{i:05d}"
        if not _done(cp, unit):
            evidence = _stream_passthrough(source, sp["passthrough"],
                                           request["codec"]["tensor_scope"])
            if evidence["parameter_count"] != shard_stats["passthrough_parameters"]:
                raise LadderError("passthrough parameter count mismatch")
            _finish(paths, cp, unit, evidence)
        unit = f"encode:{i:05d}"
        if not _done(cp, unit):
            if shard_stats["quantized_tensors"] == 0:
                _finish(paths, cp, unit, {"skipped": True, "reason": "no all-2D tensors"})
            else:
                tmp = sp["packed"].with_name(f".{sp['packed'].name}.partial.{os.getpid()}")
                try:
                    BASE._run_logged(_quantizer_argv(request, source, tmp), sp["encode_log"],
                                     env=BASE._fixed_env())
                    if BASE._str2_source_sha256(tmp) != source_row["sha256"]:
                        raise LadderError("STR2 embedded source digest differs from source shard")
                    BASE._commit_generated(tmp, sp["packed"])
                finally:
                    try: tmp.unlink()
                    except FileNotFoundError: pass
                _finish(paths, cp, unit, {"artifact": _artifact(sp["packed"]),
                                          "log": _artifact(sp["encode_log"])})
        unit = f"attest:{i:05d}"
        if not _done(cp, unit):
            if shard_stats["quantized_tensors"] == 0:
                _finish(paths, cp, unit, {"skipped": True, "reason": "no packed archive"})
            else:
                BASE._run_logged([request["execution"]["attestor_path"], str(sp["packed"]),
                                  "--roots"], sp["attest_log"], env=BASE._fixed_env())
                text = sp["attest_log"].read_text(encoding="utf-8", errors="replace")
                if "self-verify" not in text or "model_root" not in text:
                    raise LadderError("attestor returned no self-verify/model-root evidence")
                _finish(paths, cp, unit, {"archive": _artifact(sp["packed"]),
                                          "log": _artifact(sp["attest_log"])})
        unit = f"decode:{i:05d}"
        if unit in plan:
            if not _done(cp, unit):
                tmp = sp["reconstruction"].with_name(
                    f".{sp['reconstruction'].name}.partial.{os.getpid()}")
                try:
                    BASE._run_logged([request["execution"]["decoder_path"], str(sp["packed"]),
                                      str(tmp), "--dtype", "bf16"], sp["decode_log"],
                                     env=BASE._fixed_env())
                    _validate_decoded(source, tmp, scope=request["codec"]["tensor_scope"])
                    BASE._commit_generated(tmp, sp["reconstruction"])
                finally:
                    try: tmp.unlink()
                    except FileNotFoundError: pass
                _finish(paths, cp, unit, _validate_decoded(
                    source, sp["reconstruction"], scope=request["codec"]["tensor_scope"]))
            decoded.append((i, sp["reconstruction"]))

    if not _done(cp, "bundle_manifest"):
        bundle = _bundle_manifest(request, paths, cp, totals)
        _finish(paths, cp, "bundle_manifest", {"artifact": _artifact(paths["manifest"]),
                                                "physical_accounting": bundle["physical_accounting"]})
    if request["evaluation"]["mode"] == "resident":
        if not _done(cp, "override_manifest"):
            _write_override_manifest(paths["override_manifest"], decoded)
            _finish(paths, cp, "override_manifest", {"artifact": _artifact(paths["override_manifest"])})
        eval_units = (
            ("baseline_ppl", "ppl", None, paths["baseline_ppl"]),
            ("reconstruction_ppl", "ppl", paths["override_manifest"], paths["recon_ppl"]),
            ("baseline_capability", "capability", None, paths["baseline_cap"]),
            ("reconstruction_capability", "capability", paths["override_manifest"], paths["recon_cap"]),
        )
        for unit, mode, override, result_path in eval_units:
            if not _done(cp, unit):
                if override is None:
                    evidence = _run_baseline_eval_cached(
                        request, mode, result_path, paths["logs"] / f"{unit}.log")
                else:
                    evidence = _run_eval(request, mode, override, result_path,
                                         paths["logs"] / f"{unit}.log")
                _finish(paths, cp, unit, evidence)
    if "ephemeral_cleanup" in plan and not _done(cp, "ephemeral_cleanup"):
        cleanup = _cleanup_ephemeral(paths)
        _finish(paths, cp, "ephemeral_cleanup", {
            "artifact": _artifact(paths["ephemeral_receipt"]),
            "deleted_count": len(cleanup["deleted_artifacts"]),
        })
    if not _done(cp, "receipt"):
        before = cp["units"]["preflight"]["resources"]
        after = BASE._resource_sample(output)
        bundle = _load_json(paths["manifest"])
        receipt = {
            "schema": RECEIPT_SCHEMA, "completed_at": _now(), "status": "complete",
            "request": _artifact(request_path), "request_id": request["request_id"],
            "campaign_binding": request["campaign_binding"], "label": request["label"],
            "model_family": request["model_family"], "codec": request["codec"],
            "doctor_hook": request["doctor_hook"],
            "source_manifest_sha256": request["source"]["source_manifest_sha256"],
            "parameter_accounting": totals, "bundle": {
                "manifest": _artifact(paths["manifest"]),
                "physical_accounting": bundle["physical_accounting"],
            }, "quality_observation": _quality(paths, request["evaluation"]["mode"]),
            "ephemeral_cleanup": (_artifact(paths["ephemeral_receipt"])
                                  if paths["ephemeral_receipt"].is_file() else None),
            "baseline_cache": ({
                "ppl": cp["units"].get("baseline_ppl", {}).get("baseline_cache"),
                "capability": cp["units"].get(
                    "baseline_capability", {}).get("baseline_cache"),
            } if request["evaluation"]["mode"] == "resident" else None),
            "resources": {"before": before, "after": after},
            "resume": {"unit_order": plan, "atomic_replace": True,
                       "fsync_file": True, "fsync_parent_directory": True,
                       "shard_boundary_checkpointing": True},
            "claims": {"target_physical_rate_met": bundle["physical_accounting"]["target_met"],
                       "quality": False, "dominance": False, "source_deletion": False},
        }
        receipt["receipt_sha256"] = _sha_value(receipt)
        BASE._atomic_json(paths["receipt"], receipt)
        _finish(paths, cp, "receipt", {"artifact": _artifact(paths["receipt"]),
                                        "receipt_sha256": receipt["receipt_sha256"]})
    cp["status"] = "complete"; _save_cp(paths["checkpoint"], cp)
    return _load_json(paths["receipt"])


def _selftest() -> None:
    assert _rate_geometry("4")["packed_output_supported"] is True
    assert _rate_geometry("0.8")["artifact_mode"] == "packed_vector_control"
    assert _rate_geometry("0.33")["exact_nominal_match"] is False
    assert _rate_geometry("0.55")["candidate_within_payload_ceiling"] is True
    assert _rate_geometry("0.1")["vector_dim"] == 32
    caps = capabilities()
    assert len(caps["rates"]) == 10
    assert all(row["all_in_model_target_supported"] == "candidate_only_until_measured"
               for row in caps["rates"])
    assert caps["gpt_oss_120b"]["supported"] is False
    assert "3B" in SUPPORTED_LABELS and "3B" in RESIDENT_LABELS
    scalar = {"codec": {"artifact_mode": "packed_scalar_control"},
              "evaluation": {"mode": "deferred"}}
    plan = _plan(scalar, [{"quantized_tensors": 1}, {"quantized_tensors": 0}])
    assert plan == ["preflight", "metadata", "passthrough:00000", "encode:00000",
                    "attest:00000", "passthrough:00001", "encode:00001",
                    "attest:00001", "bundle_manifest", "receipt"]
    print(json.dumps({"status": "ok", "schema": REQUEST_SCHEMA,
                      "canonical_rate_count": len(CANONICAL_RATES)}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "preflight"):
        child = sub.add_parser(command); child.add_argument("--request", required=True, type=Path)
    sub.add_parser("capabilities"); sub.add_parser("selftest")
    args = parser.parse_args(argv)
    try:
        if args.command == "capabilities":
            print(json.dumps(capabilities(), indent=2, sort_keys=True)); return 0
        if args.command == "selftest":
            _selftest(); return 0
        result = _execute(args.request, preflight_only=args.command == "preflight")
        print(json.dumps(result, sort_keys=True)); return 0
    except (LadderError, BASE.PassBError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
