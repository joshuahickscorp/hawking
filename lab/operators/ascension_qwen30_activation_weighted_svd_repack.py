"""Build a sealed Q30 complete candidate using activation_weighted_svd_low_rank_q.

This branch leaves the admitted binary baseline untouched.  It produces a *new*
complete candidate under quality-candidates/activation-weighted-svd-v1 by:

1. Binding a real L0 routed-activation capture by path + sha256.
2. Selecting each L0 expert gate/up/down organ by surplus-over-null under the
   1.5 component BPW ceiling (weight cosine is reported and used only as the
   distribution-local guard, not the selection metric).
3. Encoding selected organs with activation_weighted_svd_low_rank_q (real
   capture fit; dual-gravity first-class codec).
4. Hard-linking every other tensor from the admitted complete-binary baseline
   so the candidate remains a full 18,867-tensor physical catalog.

Selection metric inversion is the whole finding: the live search optimizes
weight-space cosine; this branch optimizes activation surplus-over-null.

Not a coherence claim.  Not a promotion.  Not a server repoint.  The honest
bar is a candidate for admission; text generation on a served artifact is
run later by the operator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.operators import ascension_dual_gravity_worker as dual
from lab.operators import ascension_qwen30_complete_gravity as complete
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
# Prefer the durable main-tree paths (worktrees often omit multi-GB weights).
MAIN_HAWKING = Path("/Users/scammermike/Downloads/hawking")
MODEL_DIR = MAIN_HAWKING / (
    "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
)
QWEN30_ROOT = MAIN_HAWKING / (
    "workspace/campaign/records/ascension-sandbox/physical/qwen30"
)
BASELINE_ROOT = QWEN30_ROOT / "complete-gravity"
SOURCE_AUDIT = QWEN30_ROOT / "QWEN30_SOURCE_BODY_AUDIT_CANDIDATE.json"
DEFAULT_CAPTURE_RUN = dual.DEFAULT_Q30_ACTIVATION_CAPTURE_RUN
# Prefer the worktree/repo-local candidate root so sandboxed runs can seal
# without writing into the live main-tree quality-candidates control surface.
DEFAULT_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30"
    / "quality-candidates/activation-weighted-svd-v1"
)

SCHEMA = "hawking.ascension.qwen30_activation_weighted_svd_repack_candidate.v1"
SELECTION_SCHEMA = "hawking.ascension.qwen30_activation_weighted_svd_selection.v1"
SOURCE_SNAPSHOT_SCHEMA = "hawking.ascension.qwen30_activation_weighted_svd_source_snapshot.v1"
TERMINAL_SCHEMA = "hawking.ascension.complete_binary_terminal_status.v1"
BRANCH_ID = "qwen30-activation-weighted-svd-v1"
ARTIFACT_PREFIX = "QWEN30_ACTIVATION_WEIGHTED_SVD_V1"
MODEL_ID = "Qwen3-Coder-30B-A3B-Instruct-activation-weighted-svd-v1"
EXPECTED_TENSOR_COUNT = 18_867
CEILING_BPW = 1.5
MIN_TOKENS = 32
HOLD_FRAC = 0.25
SEED = 0xA17A5D
OPERATOR_RECOVERY_WEIGHT_COS = dual.OPERATOR_RECOVERY_WEIGHT_COS
LAYER = 0
COMPONENTS = ("gate_proj", "up_proj", "down_proj")

# Under-ceiling budgets from the measured family probe.  Over-ceiling anchors
# are intentionally excluded from selection.
BUDGET_POINTS: tuple[dict[str, Any], ...] = (
    {"label": "r64_b3", "rank": 64, "bits": 3},
    {"label": "r128_b3", "rank": 128, "bits": 3},
    {"label": "r192_b4", "rank": 192, "bits": 4},
    {"label": "r256_b3", "rank": 256, "bits": 3},
)

BASELINE_MANIFEST_NAME = "QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
BASELINE_ADMISSION_NAME = "QWEN30_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"
BASELINE_REVALIDATION_NAME = "QWEN30_CURRENT_SOURCE_SHARD_REVALIDATION.json"


class ActivationWeightedRepackError(RuntimeError):
    """The activation-weighted complete candidate cannot be sealed safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _raw_sha256(path: Path) -> str:
    return complete._sha256_file(path)


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    raw = complete._read_json(path)
    if raw is None:
        raise ActivationWeightedRepackError(f"missing {label}: {path}")
    try:
        return verify(raw, label=str(path))
    except Exception as exc:
        raise ActivationWeightedRepackError(f"untrustworthy {label}: {exc}") from exc


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    complete._atomic_json(path, payload)


def _file_binding(path: Path, *, label: str) -> dict[str, Any]:
    document = _sealed(path, label=label)
    return {
        "path": str(path.resolve()),
        "document_sha256": _raw_sha256(path),
        "seal_sha256": document["seal_sha256"],
        "file_identity": complete._file_identity(path, label=label),
    }


def _file_identity_loose(path: Path) -> dict[str, int]:
    st = path.stat()
    return {
        "bytes": int(st.st_size),
        "device": int(st.st_dev),
        "inode": int(st.st_ino),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(st.st_ctime_ns),
    }


def silu(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=np.float32), -60.0, 60.0)
    return clipped / (1.0 + np.exp(-clipped))


def holdout_split(X: np.ndarray, *, seed: int, hold_frac: float = HOLD_FRAC) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n < 4:
        return X, X
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = max(1, min(n // 2, int(round(n * hold_frac))))
    hold_idx = perm[:n_hold]
    fit_idx = perm[n_hold:]
    if fit_idx.size == 0:
        fit_idx = perm
    return X[fit_idx], X[hold_idx]


def collect_expert_activations(
    run_dir: Path, capture: Mapping[str, Any]
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    by_expert: dict[int, list[np.ndarray]] = {}
    total_steps = 0
    for probe in capture["probes"]:
        for step in probe["steps"]:
            total_steps += 1
            rel = step["router_input_hidden_f32le"]["relative_path"]
            path = run_dir / rel
            x = np.fromfile(path, dtype="<f4")
            if x.size != step["router_input_hidden_f32le"]["elements"]:
                raise ActivationWeightedRepackError(f"hidden size mismatch at {path}")
            for expert_id in step["selected_expert_ids"]:
                by_expert.setdefault(int(expert_id), []).append(x)
    stacked = {eid: np.stack(rows, axis=0) for eid, rows in by_expert.items()}
    provenance = {
        "total_steps": total_steps,
        "token_expert_pairs": int(sum(v.shape[0] for v in stacked.values())),
        "experts_with_hits": len(stacked),
        "hit_counts": {
            str(k): int(v.shape[0])
            for k, v in sorted(stacked.items(), key=lambda kv: (-kv[1].shape[0], kv[0]))
        },
    }
    return stacked, provenance


def organ_activations(
    *,
    expert: int,
    component: str,
    X_hidden: np.ndarray,
    model_dir: Path,
    weight_map: Mapping[str, str],
) -> np.ndarray:
    if component in ("gate_proj", "up_proj"):
        return X_hidden
    if component != "down_proj":
        raise ActivationWeightedRepackError(f"unsupported component {component}")
    Wg = load_tensor(
        model_dir, weight_map, f"model.layers.{LAYER}.mlp.experts.{expert}.gate_proj.weight"
    ).astype(np.float32, copy=False)
    Wu = load_tensor(
        model_dir, weight_map, f"model.layers.{LAYER}.mlp.experts.{expert}.up_proj.weight"
    ).astype(np.float32, copy=False)
    return silu(X_hidden @ Wg.T) * (X_hidden @ Wu.T)


def select_budget_for_organ(
    *,
    W: np.ndarray,
    X_fit: np.ndarray,
    X_hold: np.ndarray,
    capture_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Pick the under-ceiling budget maximizing surplus-over-null."""

    candidates: list[dict[str, Any]] = []
    for budget in BUDGET_POINTS:
        codec = dual._activation_weighted_svd_low_rank_codec(
            W,
            rank=int(budget["rank"]),
            bits=int(budget["bits"]),
            X_fit=X_fit,
            capture_identity=capture_identity,
            X_hold=X_hold,
        )
        bpw = len(codec.payload) * 8.0 / max(W.size, 1)
        quality = codec.metadata["activation_quality"]
        under = bool(bpw <= CEILING_BPW + 1e-9)
        row = {
            "budget_label": budget["label"],
            "rank": int(codec.metadata["rank"]),
            "bits": int(budget["bits"]),
            "component_bpw": float(bpw),
            "under_ceiling": under,
            "artifact_bytes": len(codec.payload),
            "payload_sha256": hashlib.sha256(codec.payload).hexdigest(),
            "payload": codec.payload,
            "reconstruction": codec.reconstruction,
            "codec_metadata": {
                k: v
                for k, v in codec.metadata.items()
                if k not in {"left", "right"}  # large nested scale tables omitted from selection receipt
            },
            "left_meta": codec.metadata["left"],
            "right_meta": codec.metadata["right"],
            "weight_cosine": float(quality["weight_cosine"]),
            "weight_relative_l2": float(quality["weight_relative_l2"]),
            "output_cosine": float(quality["output_cosine"]),
            "null_baseline": float(quality["null_baseline"]),
            "surplus_over_null": float(quality["surplus_over_null"]),
            "beats_null": bool(quality["beats_null"]),
            "distribution_local_only": bool(quality["distribution_local_only"]),
            "n_hold_tokens": int(quality["n_hold_tokens"]),
            "n_fit_tokens": int(codec.metadata["fit"]["n_fit_tokens"]),
        }
        # Keep full metadata for the winning payload write.
        row["_full_metadata"] = codec.metadata
        candidates.append(row)
    under = [c for c in candidates if c["under_ceiling"]]
    if not under:
        best = max(candidates, key=lambda r: (r["surplus_over_null"], r["weight_cosine"], -r["component_bpw"]))
        raise ActivationWeightedRepackError(
            f"no under-ceiling budget for organ; best surplus budget was "
            f"{best['budget_label']} at bpw={best['component_bpw']:.4f}"
        )
    # Primary: surplus-over-null. Secondary: weight cosine. Tertiary: lower BPW.
    under.sort(
        key=lambda r: (
            -float(r["surplus_over_null"]),
            -float(r["weight_cosine"]),
            float(r["component_bpw"]),
            str(r["budget_label"]),
        )
    )
    winner = under[0]
    winner["sweep"] = [
        {
            k: v
            for k, v in c.items()
            if k not in {"payload", "reconstruction", "_full_metadata", "left_meta", "right_meta"}
        }
        for c in candidates
    ]
    return winner


class ActivationWeightedSvdRepack:
    """Seal a complete activation-weighted-SVD candidate from baseline + capture."""

    def __init__(
        self,
        *,
        model_dir: Path = MODEL_DIR,
        baseline_root: Path = BASELINE_ROOT,
        source_audit: Path = SOURCE_AUDIT,
        capture_run: Path = DEFAULT_CAPTURE_RUN,
        root: Path = DEFAULT_ROOT,
        max_experts: int | None = None,
        workers: int = 4,
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.baseline_root = baseline_root.expanduser().resolve()
        self.source_audit = source_audit.expanduser().resolve()
        self.capture_run = capture_run.expanduser().resolve()
        self.root = root.expanduser().resolve()
        self.max_experts = max_experts
        self.workers = max(1, int(workers))
        self.tensor_dir = self.root / "tensors"
        self.manifest_path = self.root / f"{ARTIFACT_PREFIX}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
        self.selection_path = self.root / f"{ARTIFACT_PREFIX}_SELECTION_RECEIPT.json"
        self.snapshot_path = self.root / f"{ARTIFACT_PREFIX}_SOURCE_BINDING_SNAPSHOT.json"
        self.terminal_path = self.root / f"{ARTIFACT_PREFIX}_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json"
        self.status_path = self.root / f"{ARTIFACT_PREFIX}_COMPLETE_GRAVITY_STATUS.json"
        self.baseline_manifest_path = self.baseline_root / BASELINE_MANIFEST_NAME
        self.baseline_admission_path = self.baseline_root / BASELINE_ADMISSION_NAME
        self.baseline_revalidation_path = self.baseline_root / BASELINE_REVALIDATION_NAME
        self.baseline_tensor_dir = self.baseline_root / "tensors"

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tensor_dir.mkdir(parents=True, exist_ok=True)

        if self.manifest_path.is_file() and self.terminal_path.is_file():
            terminal = _sealed(self.terminal_path, label="existing terminal receipt")
            manifest = _sealed(self.manifest_path, label="existing candidate manifest")
            return {
                "status": "ALREADY_SEALED",
                "complete_physical_bpw": manifest["complete_physical_bpw_ledger"]["complete_physical_bpw"],
                "manifest_path": str(self.manifest_path),
                "terminal_path": str(self.terminal_path),
                "manifest_seal_sha256": manifest["seal_sha256"],
                "terminal_seal_sha256": terminal["seal_sha256"],
            }

        capture_identity = dual._activation_capture_binding(self.capture_run)
        capture = json.loads(Path(capture_identity["capture_result_path"]).read_text(encoding="utf-8"))
        by_expert, act_prov = collect_expert_activations(self.capture_run, capture)
        weight_map = load_weight_map(self.model_dir)
        audit = _sealed(self.source_audit, label="source body audit")
        revalidation = _sealed(self.baseline_revalidation_path, label="baseline source revalidation")
        baseline_manifest = _sealed(self.baseline_manifest_path, label="baseline complete manifest")
        baseline_admission = _sealed(self.baseline_admission_path, label="baseline admission")

        baseline_ledger = baseline_manifest["complete_physical_bpw_ledger"]
        baseline_control = {
            "complete_physical_bpw": baseline_ledger["complete_physical_bpw"],
            "manifest": {
                "path": str(self.baseline_manifest_path),
                "seal_sha256": baseline_manifest["seal_sha256"],
                "document_sha256": _raw_sha256(self.baseline_manifest_path),
            },
            "admission": {
                "path": str(self.baseline_admission_path),
                "seal_sha256": baseline_admission["seal_sha256"],
                "document_sha256": _raw_sha256(self.baseline_admission_path),
            },
            "source_revalidation": {
                "path": str(self.baseline_revalidation_path),
                "seal_sha256": revalidation["seal_sha256"],
                "document_sha256": _raw_sha256(self.baseline_revalidation_path),
            },
            "preserve_as_rollback_control": True,
            "replacement_forbidden_until_all_acceptance_gates_pass": True,
        }

        experts = sorted(
            (e for e, X in by_expert.items() if X.shape[0] >= MIN_TOKENS),
            key=lambda e: (-by_expert[e].shape[0], e),
        )
        if self.max_experts is not None:
            experts = experts[: max(0, int(self.max_experts))]
        if not experts:
            raise ActivationWeightedRepackError("no experts with enough routed tokens for fitting")

        binding = {
            "branch_id": BRANCH_ID,
            "model_id": MODEL_ID,
            "artifact_prefix": ARTIFACT_PREFIX,
            "source_body_audit_seal_sha256": audit["seal_sha256"],
            "immutable_source_revalidation": {
                "path": str(self.baseline_revalidation_path),
                "document_sha256": _raw_sha256(self.baseline_revalidation_path),
                "seal_sha256": revalidation["seal_sha256"],
            },
            "activation_capture": capture_identity,
            "selection_metric": {
                "primary": "surplus_over_null",
                "secondary": "weight_cosine",
                "weight_cosine_role": "distribution_local_guard",
                "operator_recovery_weight_cos_cutoff": OPERATOR_RECOVERY_WEIGHT_COS,
                "ceiling_component_bpw": CEILING_BPW,
            },
            "baseline_control": baseline_control,
            "eligible_experts": experts,
            "components": list(COMPONENTS),
            "budget_points": list(BUDGET_POINTS),
            "min_tokens": MIN_TOKENS,
            "layer": LAYER,
            "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
        }

        snapshot = seal(
            {
                "schema": SOURCE_SNAPSHOT_SCHEMA,
                "status": "EARNED_IMMUTABLE_SOURCE_AND_CAPTURE_BINDING",
                "recorded_at": _utc_now(),
                "binding": binding,
                "activation_provenance": act_prov,
                "claim_boundary": {
                    "baseline_control_is_preserved_and_not_mutated": True,
                    "capture_is_real_routed_activation_not_synthetic_unit_direction": True,
                    "snapshot_is_not_an_artifact_admission": True,
                },
            }
        )
        if not self.snapshot_path.is_file():
            _atomic_json(self.snapshot_path, snapshot)
        else:
            existing = _sealed(self.snapshot_path, label="source snapshot")
            if existing.get("seal_sha256") != snapshot["seal_sha256"] and _canonical_sha256(
                {k: v for k, v in existing.items() if k != "seal_sha256"}
            ) != _canonical_sha256({k: v for k, v in snapshot.items() if k != "seal_sha256"}):
                # Allow re-seal only when content matches after stripping recorded_at/seal.
                pass

        if self.selection_path.is_file():
            selection = _sealed(self.selection_path, label="selection receipt")
        else:
            selection = self._build_selection(
                experts=experts,
                by_expert=by_expert,
                weight_map=weight_map,
                capture_identity=capture_identity,
                binding=binding,
                act_prov=act_prov,
            )
            _atomic_json(self.selection_path, selection)

        selected_organs = selection["selected_representation"]["organs"]
        selected_by_name = {row["tensor_name"]: row for row in selected_organs}

        # Hard-link or copy every baseline tensor, then overwrite selected organs.
        baseline_rows = {
            row["tensor_name"]: row
            for row in baseline_manifest["tensors"]
            if isinstance(row, Mapping) and isinstance(row.get("tensor_name"), str)
        }
        if len(baseline_rows) != EXPECTED_TENSOR_COUNT:
            raise ActivationWeightedRepackError(
                f"baseline manifest tensor count {len(baseline_rows)} != {EXPECTED_TENSOR_COUNT}"
            )

        self._materialize_baseline_tensors(baseline_rows)
        ordered: list[dict[str, Any]] = []
        for tensor_name in sorted(baseline_rows):
            base = baseline_rows[tensor_name]
            if tensor_name in selected_by_name:
                ordered.append(
                    self._write_selected_organ(
                        tensor_name=tensor_name,
                        baseline_row=base,
                        organ=selected_by_name[tensor_name],
                        weight_map=weight_map,
                    )
                )
            else:
                ordered.append(self._row_from_baseline(tensor_name=tensor_name, baseline_row=base))

        artifact_bytes = sum(int(row["artifact_bytes"]) for row in ordered)
        elements = sum(int(row["elements"]) for row in ordered)
        if len(ordered) != EXPECTED_TENSOR_COUNT:
            raise ActivationWeightedRepackError("candidate tensor catalog is incomplete")

        quality_summary = self._quality_summary(ordered, selection=selection, capture_identity=capture_identity)
        representation = {
            "family": "mixed_direct_binary_sign_scale_plus_selected_activation_weighted_svd_low_rank",
            "unchanged_tensor_layout": "HQ30G1B1 binary sign plus FP16 group scale (hard-linked from admitted baseline)",
            "selected_organ_layout": "HGRAVS01 activation_weighted_svd_low_rank factors",
            "selected_family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
            "selected_organs": [row["tensor_name"] for row in selected_organs],
            "selection_metric": "surplus_over_null",
            "weight_cosine_role": "secondary_and_distribution_local_guard",
            "activation_capture": {
                "path": capture_identity["path"],
                "sha256": capture_identity["sha256"],
            },
            "native_reader_requirement": (
                "a native reader that understands HQ30G1B1 plus HGRAVS01 activation-weighted "
                "SVD factors is required before any runtime use"
            ),
            "physical_direct_layout": True,
        }

        preliminary = {
            "schema": SCHEMA,
            "status": "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
            "recorded_at": _utc_now(),
            "source_body_audit_seal_sha256": audit["seal_sha256"],
            "source_revalidation_receipt_path": str(self.baseline_revalidation_path),
            "source_revalidation_receipt_seal_sha256": revalidation["seal_sha256"],
            "source": {
                "repository": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "model_dir": str(self.model_dir),
                "tensor_count": EXPECTED_TENSOR_COUNT,
            },
            "representation": representation,
            "tensors": ordered,
        }

        def build_manifest(*, manifest_bytes_billed: int, ledger_padding: str | None = None) -> dict[str, Any]:
            total_bytes = artifact_bytes + manifest_bytes_billed
            complete_bpw = total_bytes * 8.0 / elements
            ledger: dict[str, Any] = {
                "source_weight_elements": elements,
                "tensor_payload_bytes": artifact_bytes,
                "manifest_bytes_billed": manifest_bytes_billed,
                "all_required_weight_artifact_bytes": total_bytes,
                "complete_physical_bpw": complete_bpw,
                "threshold_bpw": CEILING_BPW,
                "passes_storage_threshold": complete_bpw <= CEILING_BPW,
                "explicitly_excluded_separate_state": [
                    "KV_cache_bytes",
                    "Qwen80_recurrent_state_bytes",
                    "Context_OS_cache_bytes",
                    "Agent_OS_memory_bytes",
                ],
            }
            if ledger_padding is not None:
                ledger["manifest_ledger_padding_bytes"] = len(ledger_padding)
                ledger["manifest_ledger_padding"] = ledger_padding
            if complete_bpw > CEILING_BPW:
                # Honest refusal: never silently trim quality to force the gate.
                raise ActivationWeightedRepackError(
                    f"complete_physical_bpw {complete_bpw:.6f} exceeds ceiling {CEILING_BPW}; "
                    "refusing to seal a trimmed candidate"
                )
            return seal(
                {
                    **preliminary,
                    "complete_physical_bpw_ledger": ledger,
                    "champion_classes": {
                        "current_bpw_candidate": {
                            "candidate": BRANCH_ID,
                            "complete_physical_bpw": complete_bpw,
                            "status": "CANDIDATE_ONLY_NOT_A_BASELINE_REPLACEMENT",
                        },
                        "admitted_baseline_rollback_control": {
                            "complete_physical_bpw": baseline_control["complete_physical_bpw"],
                            "manifest_seal_sha256": baseline_control["manifest"]["seal_sha256"],
                            "status": "PRESERVED_SEPARATE_CONTROL",
                        },
                        "runtime_capability_hcli_tps_tg": {
                            "status": "BLOCKED_BY_REQUIRED_NEW_NATIVE_READER_ADMISSION_AND_FULL_RETEST",
                            "timing_or_capability_transfer_from_baseline_forbidden": True,
                        },
                    },
                    "quality_summary": quality_summary,
                    "activation_weighted_svd_branch": {
                        "branch_id": BRANCH_ID,
                        "selection_receipt": {
                            "path": str(self.selection_path),
                            "document_sha256": _raw_sha256(self.selection_path),
                            "seal_sha256": selection["seal_sha256"],
                        },
                        "source_binding_snapshot": {
                            "path": str(self.snapshot_path),
                            "document_sha256": _raw_sha256(self.snapshot_path),
                            "seal_sha256": _sealed(self.snapshot_path, label="snapshot")["seal_sha256"],
                        },
                        "activation_capture": capture_identity,
                        "changed_organs": [row["tensor_name"] for row in selected_organs],
                        "selection_metric": "surplus_over_null",
                        "admission_state": "NOT_REQUESTED_REQUIRES_ACTIVATION_WEIGHTED_SVD_NATIVE_READER",
                        "baseline_rollback_control": baseline_control,
                    },
                    "claim_boundary": {
                        "complete_physical_tensor_coverage_is_true": True,
                        "complete_bpw_is_real_accounted_bytes_not_a_capability_result": True,
                        "admitted_baseline_is_preserved_and_automatic_replacement_is_forbidden": True,
                        "activation_capture_is_bound_by_path_and_sha256": True,
                        "selection_is_surplus_over_null_not_weight_cosine": True,
                        "native_admission_runtime_generation_hcli_agent_os_tps_tg_and_tournament_are_unearned": True,
                        "coherence_is_not_claimed_without_served_text_generation": True,
                        "raw_source_remains_authority_teacher_only": True,
                    },
                }
            )

        def manifest_document_bytes(document: Mapping[str, Any]) -> int:
            return len(
                json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ) + 1

        manifest_bytes_billed = 0
        candidate: dict[str, Any] | None = None
        for _ in range(16):
            candidate = build_manifest(manifest_bytes_billed=manifest_bytes_billed)
            actual_bytes = manifest_document_bytes(candidate)
            if actual_bytes == manifest_bytes_billed:
                break
            manifest_bytes_billed = actual_bytes
        else:
            target_bytes = max(manifest_bytes_billed, actual_bytes) + 8192
            for _ in range(16):
                padding = ""
                for _ in range(16):
                    candidate = build_manifest(manifest_bytes_billed=target_bytes, ledger_padding=padding)
                    actual_bytes = manifest_document_bytes(candidate)
                    if actual_bytes == target_bytes:
                        break
                    next_padding = len(padding) + target_bytes - actual_bytes
                    if next_padding < 0:
                        break
                    padding = "0" * next_padding
                if actual_bytes == target_bytes:
                    manifest_bytes_billed = target_bytes
                    break
                target_bytes = max(target_bytes, actual_bytes) + 8192
            else:
                raise ActivationWeightedRepackError("manifest-byte ledger did not converge")
        assert candidate is not None
        _atomic_json(self.manifest_path, candidate)

        ledger = candidate["complete_physical_bpw_ledger"]
        terminal = seal(
            {
                "schema": TERMINAL_SCHEMA,
                "status": "EARNED_COMPLETE_PHYSICAL_BINARY_CANDIDATE_UNQUALIFIED",
                "recorded_at": _utc_now(),
                "binding": {
                    "model_id": MODEL_ID,
                    "artifact_prefix": ARTIFACT_PREFIX,
                    "manifest_schema": SCHEMA,
                    "source_body_audit_seal_sha256": audit["seal_sha256"],
                    "source_revalidation_receipt_path": str(self.baseline_revalidation_path),
                    "source_revalidation_receipt_seal_sha256": revalidation["seal_sha256"],
                    "activation_capture": {
                        "path": capture_identity["path"],
                        "sha256": capture_identity["sha256"],
                    },
                    "progress": {
                        "planned_tensors": EXPECTED_TENSOR_COUNT,
                        "completed_tensors": EXPECTED_TENSOR_COUNT,
                        "next_cursor": EXPECTED_TENSOR_COUNT,
                        "next_source_shard": None,
                        "next_tensor_name": None,
                    },
                },
                "candidate": {
                    "manifest_path": str(self.manifest_path),
                    "manifest_seal_sha256": candidate["seal_sha256"],
                    "manifest_document_sha256": _raw_sha256(self.manifest_path),
                    "manifest_file_identity": _file_identity_loose(self.manifest_path),
                    "all_required_weight_artifact_bytes": ledger["all_required_weight_artifact_bytes"],
                    "complete_physical_bpw": ledger["complete_physical_bpw"],
                    "passes_storage_threshold": ledger["passes_storage_threshold"],
                },
                "claim_boundary": {
                    "candidate_remains_unqualified_for_native_runtime_capability_hcli_tps_tg_and_tournament": True,
                    "coherence_not_claimed": True,
                    "promotion_and_server_repoint_forbidden_by_this_receipt": True,
                },
            }
        )
        _atomic_json(self.terminal_path, terminal)
        status = {
            "schema": "hawking.ascension.qwen30_activation_weighted_svd_status.v1",
            "status": "EARNED_COMPLETE_ACTIVATION_WEIGHTED_SVD_CANDIDATE_UNQUALIFIED",
            "recorded_at": _utc_now(),
            "branch_id": BRANCH_ID,
            "complete_physical_bpw": ledger["complete_physical_bpw"],
            "changed_organs": len(selected_organs),
            "manifest_seal_sha256": candidate["seal_sha256"],
            "terminal_seal_sha256": terminal["seal_sha256"],
            "activation_capture_sha256": capture_identity["sha256"],
            "elapsed_seconds": time.perf_counter() - started,
            "claim_boundary": {
                "not_promoted": True,
                "not_served": True,
                "not_a_coherence_claim": True,
            },
        }
        _atomic_json(self.status_path, status)
        return {
            "status": status["status"],
            "complete_physical_bpw": ledger["complete_physical_bpw"],
            "passes_storage_threshold": ledger["passes_storage_threshold"],
            "changed_organs": len(selected_organs),
            "manifest_path": str(self.manifest_path),
            "selection_path": str(self.selection_path),
            "terminal_path": str(self.terminal_path),
            "manifest_seal_sha256": candidate["seal_sha256"],
            "terminal_seal_sha256": terminal["seal_sha256"],
            "activation_capture": capture_identity,
            "quality_summary": quality_summary,
            "elapsed_seconds": status["elapsed_seconds"],
        }

    def _build_selection(
        self,
        *,
        experts: Sequence[int],
        by_expert: Mapping[int, np.ndarray],
        weight_map: Mapping[str, str],
        capture_identity: Mapping[str, Any],
        binding: Mapping[str, Any],
        act_prov: Mapping[str, Any],
    ) -> dict[str, Any]:
        work: list[tuple[int, str]] = [(e, c) for e in experts for c in COMPONENTS]
        organs: list[dict[str, Any]] = []

        def one(expert: int, component: str) -> dict[str, Any]:
            name = f"model.layers.{LAYER}.mlp.experts.{expert}.{component}.weight"
            W = load_tensor(self.model_dir, weight_map, name).astype(np.float32, copy=False)
            X_use = organ_activations(
                expert=expert,
                component=component,
                X_hidden=by_expert[expert],
                model_dir=self.model_dir,
                weight_map=weight_map,
            )
            if W.shape[1] != X_use.shape[1]:
                raise ActivationWeightedRepackError(
                    f"{name} in-dim {W.shape[1]} != activation in-dim {X_use.shape[1]}"
                )
            X_fit, X_hold = holdout_split(
                X_use, seed=SEED ^ (expert * 1009) ^ (hash(component) & 0xFFFF)
            )
            winner = select_budget_for_organ(
                W=W, X_fit=X_fit, X_hold=X_hold, capture_identity=capture_identity
            )
            # Persist payload beside selection for deterministic rewrite.
            payload_dir = self.root / "selected-payloads"
            payload_dir.mkdir(parents=True, exist_ok=True)
            payload_path = payload_dir / f"{hashlib.sha256(name.encode()).hexdigest()}.hgravs01"
            payload_path.write_bytes(winner["payload"])
            return {
                "tensor_name": name,
                "expert": expert,
                "component": component,
                "shape": [int(x) for x in W.shape],
                "elements": int(W.size),
                "source_shard": weight_map[name],
                "source_value_sha256": hashlib.sha256(
                    np.ascontiguousarray(W, dtype="<f4").tobytes()
                ).hexdigest(),
                "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
                "budget_label": winner["budget_label"],
                "rank": winner["rank"],
                "bits": winner["bits"],
                "component_bpw": winner["component_bpw"],
                "under_ceiling": winner["under_ceiling"],
                "weight_cosine": winner["weight_cosine"],
                "weight_relative_l2": winner["weight_relative_l2"],
                "output_cosine": winner["output_cosine"],
                "null_baseline": winner["null_baseline"],
                "surplus_over_null": winner["surplus_over_null"],
                "beats_null": winner["beats_null"],
                "distribution_local_only": winner["distribution_local_only"],
                "n_fit_tokens": winner["n_fit_tokens"],
                "n_hold_tokens": winner["n_hold_tokens"],
                "physical_payload_bytes": winner["artifact_bytes"],
                "physical_payload_sha256": winner["payload_sha256"],
                "payload_path": str(payload_path),
                "selection_metric": "surplus_over_null",
                "budget_sweep": winner["sweep"],
                "codec_metadata": winner["codec_metadata"],
            }

        if self.workers == 1 or len(work) <= 1:
            for expert, component in work:
                organs.append(one(expert, component))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(one, e, c): (e, c) for e, c in work}
                for fut in as_completed(futures):
                    organs.append(fut.result())
        organs.sort(key=lambda r: (int(r["expert"]), str(r["component"])))

        mean_surplus = float(np.mean([r["surplus_over_null"] for r in organs])) if organs else 0.0
        mean_weight = float(np.mean([r["weight_cosine"] for r in organs])) if organs else 0.0
        mean_output = float(np.mean([r["output_cosine"] for r in organs])) if organs else 0.0
        return seal(
            {
                "schema": SELECTION_SCHEMA,
                "status": "EARNED_SURPLUS_FIRST_ACTIVATION_WEIGHTED_SVD_SELECTION_UNQUALIFIED",
                "recorded_at": _utc_now(),
                "binding": binding,
                "source_binding_snapshot": {
                    "path": str(self.snapshot_path),
                    "document_sha256": _raw_sha256(self.snapshot_path),
                    "seal_sha256": _sealed(self.snapshot_path, label="snapshot")["seal_sha256"],
                },
                "selection_method": {
                    "kind": "per_organ_max_surplus_over_null_under_component_bpw_ceiling",
                    "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
                    "budgets_evaluated": list(BUDGET_POINTS),
                    "primary_metric": "surplus_over_null",
                    "secondary_metric": "weight_cosine",
                    "weight_cosine_role": "distribution_local_guard_not_selection",
                    "operator_recovery_weight_cos_cutoff": OPERATOR_RECOVERY_WEIGHT_COS,
                    "ceiling_component_bpw": CEILING_BPW,
                    "holdout_frac": HOLD_FRAC,
                    "min_tokens": MIN_TOKENS,
                    "tie_break": "higher_weight_cosine_then_lower_component_bpw_then_budget_label",
                },
                "activation_capture": capture_identity,
                "activation_provenance": act_prov,
                "selected_representation": {
                    "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
                    "organs": organs,
                    "summary": {
                        "n_organs": len(organs),
                        "mean_surplus_over_null": mean_surplus,
                        "mean_weight_cosine": mean_weight,
                        "mean_output_cosine": mean_output,
                        "frac_beats_null": float(np.mean([1.0 if r["beats_null"] else 0.0 for r in organs])),
                        "frac_distribution_local_only": float(
                            np.mean([1.0 if r["distribution_local_only"] else 0.0 for r in organs])
                        ),
                        "mean_component_bpw": float(np.mean([r["component_bpw"] for r in organs])),
                    },
                },
                "claim_boundary": {
                    "selection_is_source_and_capture_bound_component_measurement_only": True,
                    "not_a_coherence_or_capability_claim": True,
                    "full_artifact_admission_runtime_hcli_tps_tg_and_tournament_are_unearned": True,
                },
            }
        )

    def _materialize_baseline_tensors(self, baseline_rows: Mapping[str, Mapping[str, Any]]) -> None:
        missing = 0
        for tensor_name, row in baseline_rows.items():
            dest = self.tensor_dir / complete._artifact_name(tensor_name)
            if dest.is_file() and dest.stat().st_size == int(row["artifact_bytes"]):
                # Already present (resume).
                continue
            src = Path(str(row["artifact_path"]))
            if not src.is_file():
                # Baseline row path may be absolute to main tree; also try by name.
                src = self.baseline_tensor_dir / complete._artifact_name(tensor_name)
            if not src.is_file():
                missing += 1
                raise ActivationWeightedRepackError(f"baseline tensor missing for {tensor_name}: {src}")
            if dest.is_file() or dest.is_symlink():
                dest.unlink()
            try:
                os.link(src, dest)
            except OSError:
                shutil.copy2(src, dest)
            if dest.stat().st_size != int(row["artifact_bytes"]):
                raise ActivationWeightedRepackError(f"baseline materialize size mismatch for {tensor_name}")
        if missing:
            raise ActivationWeightedRepackError(f"{missing} baseline tensors missing")

    def _row_from_baseline(
        self, *, tensor_name: str, baseline_row: Mapping[str, Any]
    ) -> dict[str, Any]:
        dest = self.tensor_dir / complete._artifact_name(tensor_name)
        digest = complete._sha256_file(dest)
        if digest != baseline_row.get("artifact_sha256"):
            # Hard-link should preserve content; copy path also should.
            raise ActivationWeightedRepackError(f"baseline artifact hash mismatch for {tensor_name}")
        return {
            "tensor_name": tensor_name,
            "source_shard": baseline_row["source_shard"],
            "source_shard_sha256": baseline_row["source_shard_sha256"],
            "source_dtype": baseline_row["source_dtype"],
            "shape": baseline_row["shape"],
            "elements": baseline_row["elements"],
            "artifact_path": str(dest),
            "artifact_bytes": int(baseline_row["artifact_bytes"]),
            "artifact_sha256": digest,
            "layout": dict(baseline_row.get("layout") or {}),
            "component_quality": dict(baseline_row.get("component_quality") or {}),
            "candidate_mutation": {
                "changed_from_admitted_control": False,
                "reason": "hard_linked_from_admitted_binary_baseline",
                "baseline_rollback": {
                    "baseline_manifest_path": str(self.baseline_manifest_path),
                    "baseline_artifact_path": baseline_row.get("artifact_path"),
                    "baseline_artifact_sha256": baseline_row.get("artifact_sha256"),
                    "baseline_artifact_bytes": baseline_row.get("artifact_bytes"),
                    "rollback_action": "use the separately admitted baseline tensor; this candidate never overwrites it",
                },
            },
        }

    def _write_selected_organ(
        self,
        *,
        tensor_name: str,
        baseline_row: Mapping[str, Any],
        organ: Mapping[str, Any],
        weight_map: Mapping[str, str],
    ) -> dict[str, Any]:
        payload_path = Path(str(organ["payload_path"]))
        payload = payload_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != organ["physical_payload_sha256"]:
            raise ActivationWeightedRepackError(f"selected payload hash changed for {tensor_name}")
        if len(payload) != int(organ["physical_payload_bytes"]):
            raise ActivationWeightedRepackError(f"selected payload size changed for {tensor_name}")
        # Decode reconstruction quality from physical bytes.
        rebuilt = dual._decode_activation_weighted_svd_low_rank_codec(payload)
        W = load_tensor(self.model_dir, weight_map, tensor_name).astype(np.float32, copy=False)
        if list(W.shape) != list(organ["shape"]):
            raise ActivationWeightedRepackError(f"source shape changed for {tensor_name}")
        weight_metrics = dual._quality(W, rebuilt)
        dest = self.tensor_dir / complete._artifact_name(tensor_name)
        if dest.is_file() or dest.is_symlink():
            dest.unlink()
        descriptor, temporary = tempfile.mkstemp(prefix=f".{dest.name}.", dir=self.tensor_dir)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o640)
            os.replace(temporary, dest)
            os.chmod(dest, 0o640)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        header, _body = dual._parse_container(payload, expected_magic=dual.MAGIC_ACT_SVD)
        return {
            "tensor_name": tensor_name,
            "source_shard": organ["source_shard"],
            "source_shard_sha256": baseline_row["source_shard_sha256"],
            "source_dtype": baseline_row["source_dtype"],
            "shape": list(organ["shape"]),
            "elements": int(organ["elements"]),
            "artifact_path": str(dest),
            "artifact_bytes": len(payload),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "layout": {
                "family": dual.ACTIVATION_WEIGHTED_SVD_REPRESENTATION,
                "magic": dual.MAGIC_ACT_SVD.decode("ascii"),
                "schema": header.get("schema"),
                "rank": organ["rank"],
                "factor_bits": organ["bits"],
                "budget_label": organ["budget_label"],
                "activation_capture_sha256": header.get("activation_capture", {}).get("sha256"),
            },
            "component_quality": {
                "cosine": float(weight_metrics["cosine"]),
                "relative_l2": float(weight_metrics["relative_l2"]),
                "rmse": float(weight_metrics["rmse"]),
                "finite": True,
                "activation_output_cosine": float(organ["output_cosine"]),
                "activation_null_baseline": float(organ["null_baseline"]),
                "activation_surplus_over_null": float(organ["surplus_over_null"]),
                "beats_null": bool(organ["beats_null"]),
                "distribution_local_only": bool(organ["distribution_local_only"]),
            },
            "candidate_mutation": {
                "changed_from_admitted_control": True,
                "reason": "surplus_first_activation_weighted_svd_low_rank_on_bound_real_capture",
                "selection_receipt_path": str(self.selection_path),
                "organ_selection": {
                    "budget_label": organ["budget_label"],
                    "rank": organ["rank"],
                    "bits": organ["bits"],
                    "component_bpw": organ["component_bpw"],
                    "surplus_over_null": organ["surplus_over_null"],
                    "weight_cosine": organ["weight_cosine"],
                    "output_cosine": organ["output_cosine"],
                    "null_baseline": organ["null_baseline"],
                    "selection_metric": "surplus_over_null",
                },
                "baseline_rollback": {
                    "baseline_manifest_path": str(self.baseline_manifest_path),
                    "baseline_artifact_path": baseline_row.get("artifact_path"),
                    "baseline_artifact_sha256": baseline_row.get("artifact_sha256"),
                    "baseline_artifact_bytes": baseline_row.get("artifact_bytes"),
                    "rollback_action": "use the separately admitted baseline tensor; this candidate never overwrites it",
                },
            },
        }

    @staticmethod
    def _quality_summary(
        ordered: Sequence[Mapping[str, Any]],
        *,
        selection: Mapping[str, Any],
        capture_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        changed = [
            row
            for row in ordered
            if isinstance(row.get("candidate_mutation"), Mapping)
            and row["candidate_mutation"].get("changed_from_admitted_control") is True
        ]
        weight_cos = [float(row["component_quality"]["cosine"]) for row in ordered]
        weight_l2 = [float(row["component_quality"]["relative_l2"]) for row in ordered]
        surplus = [
            float(row["component_quality"]["activation_surplus_over_null"])
            for row in changed
            if "activation_surplus_over_null" in row["component_quality"]
        ]
        out_cos = [
            float(row["component_quality"]["activation_output_cosine"])
            for row in changed
            if "activation_output_cosine" in row["component_quality"]
        ]
        nulls = [
            float(row["component_quality"]["activation_null_baseline"])
            for row in changed
            if "activation_null_baseline" in row["component_quality"]
        ]
        selected_summary = selection.get("selected_representation", {}).get("summary", {})
        return {
            "mean_component_cosine": float(np.mean(weight_cos)) if weight_cos else None,
            "mean_component_relative_l2": float(np.mean(weight_l2)) if weight_l2 else None,
            "changed_organs": len(changed),
            "activation_aware": {
                "selection_metric": "surplus_over_null",
                "mean_surplus_over_null_selected": float(np.mean(surplus)) if surplus else None,
                "mean_output_cosine_selected": float(np.mean(out_cos)) if out_cos else None,
                "mean_null_baseline_selected": float(np.mean(nulls)) if nulls else None,
                "mean_weight_cosine_selected": selected_summary.get("mean_weight_cosine"),
                "frac_beats_null_selected": selected_summary.get("frac_beats_null"),
                "capture_path": capture_identity.get("path"),
                "capture_sha256": capture_identity.get("sha256"),
            },
            "weight_space": {
                "mean_component_cosine_all_tensors": float(np.mean(weight_cos)) if weight_cos else None,
                "mean_component_relative_l2_all_tensors": float(np.mean(weight_l2)) if weight_l2 else None,
                "mean_weight_cosine_selected_organs": selected_summary.get("mean_weight_cosine"),
            },
            "verdict": (
                "SOURCE_AND_CAPTURE_BOUND_ACTIVATION_WEIGHTED_SVD_CANDIDATE_UNQUALIFIED_"
                "REQUIRES_INDEPENDENT_ADMISSION_AND_SERVED_TEXT_BEFORE_COHERENCE_CLAIM"
            ),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
    parser.add_argument("--capture-run", type=Path, default=DEFAULT_CAPTURE_RUN)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--max-experts",
        type=int,
        default=None,
        help="Optional cap on experts (highest-hit first). Default: all with >= min tokens.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Seal selection receipt only; do not materialize the full candidate catalog.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packer = ActivationWeightedSvdRepack(
        model_dir=args.model_dir,
        baseline_root=args.baseline_root,
        source_audit=args.source_audit,
        capture_run=args.capture_run,
        root=args.root,
        max_experts=args.max_experts,
        workers=args.workers,
    )
    if args.selection_only:
        capture_identity = dual._activation_capture_binding(args.capture_run)
        capture = json.loads(Path(capture_identity["capture_result_path"]).read_text(encoding="utf-8"))
        by_expert, act_prov = collect_expert_activations(args.capture_run, capture)
        weight_map = load_weight_map(args.model_dir)
        experts = sorted(
            (e for e, X in by_expert.items() if X.shape[0] >= MIN_TOKENS),
            key=lambda e: (-by_expert[e].shape[0], e),
        )
        if args.max_experts is not None:
            experts = experts[: args.max_experts]
        args.root.mkdir(parents=True, exist_ok=True)
        binding = {
            "branch_id": BRANCH_ID,
            "activation_capture": capture_identity,
            "selection_metric": "surplus_over_null",
            "eligible_experts": experts,
        }
        # Minimal snapshot for selection-only mode.
        snapshot = seal(
            {
                "schema": SOURCE_SNAPSHOT_SCHEMA,
                "status": "EARNED_IMMUTABLE_SOURCE_AND_CAPTURE_BINDING",
                "recorded_at": _utc_now(),
                "binding": binding,
                "activation_provenance": act_prov,
            }
        )
        _atomic_json(packer.snapshot_path, snapshot)
        selection = packer._build_selection(
            experts=experts,
            by_expert=by_expert,
            weight_map=weight_map,
            capture_identity=capture_identity,
            binding=binding,
            act_prov=act_prov,
        )
        _atomic_json(packer.selection_path, selection)
        summary = selection["selected_representation"]["summary"]
        print(json.dumps({"status": "SELECTION_ONLY", "summary": summary, "path": str(packer.selection_path)}, indent=2))
        return 0
    result = packer.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
