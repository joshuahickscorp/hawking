"""ABLITERATION — generate candidate refusal-direction transforms, never an uncensoring switch.

Tabula already owns the behavioural-surgery floor: left-null projection, independent
evaluation, lineage, and the authority lattice. This module does not replace that
floor. It recovers the *candidate generator* the three named upstreams actually
run — direction generation, direction SELECTION, projection, norm preservation,
layer scoping — and binds that generator to Tabula so a run that suppresses
refusals while destroying capability is a Tabula FAILURE, not a success.

Upstream documentation (NousResearch/llm-abliteration README) states explicitly
that abliteration does not guarantee complete removal of refusals. An artifact
that reports a model as "abliterated", "uncensored", or "refusal-free" is
refused here. The honest object is a candidate direction, a stated projection,
and recorded effect on BOTH the target behaviour AND general quality.

This lane does not run a transformation on a specimen, does not take a GPU
lease, and does not write weights. Fitting is a SLEEPING GPU_EXCLUSIVE
WorkUnit. Blocked physical work never becomes a synthetic result.

    python3 tools/future/abliteration.py --build
    python3 -m pytest tools/future/test_abliteration.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hcli.workunit import WorkUnit, is_ready
from tools.future._common import (
    HARDWARE_FIELDS,
    HardwareClaimError,
    git,
    write_receipt,
    _assert_no_hardware_claims,
)
from tools.future import negative_index as ni
from tools.future import tabula as tb
from tools.future.ebpw_categories import CategoryError
from tools.future.external_specimen_seal import (
    is_whole_tree_row,
    load_verification_doc,
    verification_row,
)
from tools.future.workunit_species import emit_hcli_workunit, validate_emitted_unit

RECEIPT = "ABLITERATION.json"
SCHEMA = "hawking.future.abliteration.v1"
VERSION = 1
RECORDED_BY = "tools/future/abliteration.py"
VERIFICATION_REL = "receipts/future/SPECIMEN_VERIFICATION.json"

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. No weight write. "
    "Abliteration is a candidate transformation generator; it does not "
    "guarantee removal of refusals (NousResearch/llm-abliteration README). "
    "A result may name a candidate direction, a stated projection, and the "
    "completion / harmless / loss evaluations that selected it. It may not "
    "report a model as abliterated, uncensored, or refusal-free."
)
SIDECAR_STATUS = "BUILT_NOT_PROMOTED"

# The three evaluations Arditi's select_direction.py actually computes before
# it is allowed to pick. A selection that omits any one is underdetermined.
REQUIRED_EVALS: tuple[str, ...] = ("completion", "harmless", "loss")
GATE_VALUES: frozenset[str] = frozenset({"PASS", "FAIL"})

# Recovered from andyrdt/refusal_direction pipeline/submodules/select_direction.py.
# These are upstream hyperparameters, not measurements this sidecar performed.
ARDITI_DEFAULTS = {
    "kl_threshold": 0.1,
    "induce_refusal_threshold": 0.0,
    "prune_layer_percentage": 0.20,
    "n_train": 128,
    "n_val": 32,
    "n_test": 100,
    "filter_train": True,
    "filter_val": True,
    "source": "andyrdt/refusal_direction pipeline/submodules/select_direction.py + pipeline/config.py",
    "measured_here": False,
}

PROJECTIONS: tuple[str, ...] = (
    "conventional",
    "projected",
    "norm_preserving_frobenius",
    "norm_preserving_biprojected",
)

# Source vs destination are different knobs. Arditi extracts at resid_pre per
# (position, layer) then ablates at ALL layers; FailSpy blacklists destination
# layers; Nous YAML names a measurement layer and a destination layer separately.
LAYER_ROLES: tuple[str, ...] = ("source", "destination")

UPSTREAM = {
    "operational": {
        "repo": "NousResearch/llm-abliteration",
        "role": "HF Transformers implementation: measure, analyze, sharded ablation, YAML marching orders",
        "guarantee": (
            "Abliteration does not guarantee full removal of censorship. "
            "A properly abliterated model will not explicitly refuse, theoretically, "
            "based on the nature of refusals captured in datasets used for abliteration."
        ),
        "projection_flags": ["--projected", "--normpreserve"],
        "full_weight_rule": "subsequent ablation needs to be performed on full-weight models",
    },
    "scientific": {
        "repo": "andyrdt/refusal_direction",
        "paper": "Refusal in Language Models Is Mediated by a Single Direction (arXiv:2406.11717)",
        "pipeline": [
            "generate_directions (mean harmful - mean harmless per position, layer)",
            "select_direction (completion + steering/harmless + KL/loss gates, THEN rank)",
            "completion eval on harmful datasets (jailbreakbench; substring + llamaguard2)",
            "harmless eval (substring matching; act-add should induce refusal)",
            "loss eval (CE on pile, alpaca, alpaca_custom_completions)",
        ],
    },
    "workbench": {
        "repo": "FailSpy/abliterator",
        "role": "barebones experimental workbench (its README's word)",
        "features": [
            "temporary ablation contexts",
            "activation caching (harmful/harmless)",
            "test_dir composite (negative_score, positive_score)",
            "layer blacklist/whitelist; default-blacklist first and last couple",
            "mse_harmless as a quality loss against cached harmless activations",
        ],
        "naive_selection": (
            "find_best_refusal_dir sorts on a single test_dir score. That is the "
            "refusal-suppression-only choice this module makes inexpressible."
        ),
    },
}

# Architecture eligibility is a declared table, not a measurement. Unknown names
# fail closed. Ineligible tokens are matched first so "Qwen3-VL-Instruct" does
# not sneak through on the instruct token.
_INELIGIBLE: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("bitnet",), "quantized_1_58", "Nous: ablation must run on full-weight models"),
    (("embedding",), "embedding_model", "residual-stream chat method; not an embedding model"),
    (("vl-",), "vision_language", "Arditi pipeline is residual-stream decoder-only text"),
    (("vlm",), "vision_language", "Arditi pipeline is residual-stream decoder-only text"),
    (("smolvlm",), "vision_language", "Arditi pipeline is residual-stream decoder-only text"),
    (("multimodal",), "multimodal", "recovered method has no vision tower contract"),
    (("hunyuan",), "video", "not a residual-stream decoder-only chat LM"),
    (("wan2",), "video", "not a residual-stream decoder-only chat LM"),
    (("video",), "video", "not a residual-stream decoder-only chat LM"),
    (("whisper",), "audio", "not a residual-stream decoder-only chat LM"),
    (("musicgen",), "audio", "not a residual-stream decoder-only chat LM"),
    (("flamingo",), "audio", "not a residual-stream decoder-only chat LM"),
    (("sam2",), "vision", "not a residual-stream decoder-only chat LM"),
    (("depth-anything",), "vision", "not a residual-stream decoder-only chat LM"),
    (("vjepa",), "vision", "not a residual-stream decoder-only chat LM"),
    (("timesfm",), "timeseries", "not a residual-stream decoder-only chat LM"),
    (("boltz",), "structure", "not a residual-stream decoder-only chat LM"),
    (("evo2",), "genome", "not a residual-stream decoder-only chat LM"),
    (("mamba",), "state_space", "Arditi residual-stream method is for decoder-only transformers"),
    (("flan-t5",), "encoder_decoder", "T5 is not the recovered residual-stream parent"),
    (("modernbert",), "encoder", "encoder-only; no chat residual stream"),
    (("jamba",), "hybrid_ssm", "hybrid SSM/transformer; not in the recovered 13-model set"),
    (("pi0",), "robotics", "not a residual-stream decoder-only chat LM"),
    (("lfm2",), "liquid_foundation", "not a residual-stream decoder-only transformer"),
)

_ELIGIBLE: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("qwen3-0.6b",), "qwen3_dense", "smallest recovered residual-stream causal LM on this lake"),
    (("qwen3-30b-a3b",), "qwen3_moe", "residual-stream MoE; expert-granular ablation is NOT in the recovered method"),
    (("qwen3.8-27b",), "qwen38_dense", "Tabula patient family; directory name is not a capability result"),
    (("granite-4.0",), "granite", "IBM Granite hybrid-instruct family"),
    (("falcon-h1",), "falcon_h1", "instruct decoder-only"),
    (("mistral-small",), "mistral", "instruct decoder-only"),
)


class UnderdeterminedSelection(ValueError):
    """A direction was offered without harmless or loss evaluation."""


class SelectionEmpty(ValueError):
    """Every candidate failed a first-class gate. Nothing is selected."""


class PlanRefusal(ValueError):
    """plan() refused: specimen, architecture, or verification failed closed."""


class ClaimBoundaryError(ValueError):
    """An artifact asserted a guarantee the method does not have."""


class RunRefused(tb.WeightsFrozen):
    """This lane does not run a transformation on a specimen."""


class MissingEvalError(UnderdeterminedSelection):
    """EvalBundle construction was attempted without a required evaluation."""


def _as_mapping(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise UnderdeterminedSelection(f"{what} must be a mapping, not {type(value).__name__}")
    return dict(value)


@dataclass(frozen=True)
class EvalBundle:
    """The three evaluations a candidate MUST carry. Missing one cannot be expressed.

    Gates are PASS/FAIL labels, never a measured capability number. This sidecar
    has no GPU authority and does not run the specimen; a real run fills the
    same shape later. ABSENT is not a gate value — omit the field and
    construction refuses.
    """

    completion: dict[str, Any]
    harmless: dict[str, Any]
    loss: dict[str, Any]

    def __post_init__(self) -> None:
        for name in REQUIRED_EVALS:
            ev = getattr(self, name)
            if not isinstance(ev, Mapping):
                raise MissingEvalError(
                    f"{name} evaluation is missing; direction selection on "
                    "refusal suppression alone is underdetermined"
                )
            gate = ev.get("gate")
            if gate not in GATE_VALUES:
                raise MissingEvalError(
                    f"{name} evaluation has no PASS/FAIL gate (got {gate!r}); "
                    "harmless and loss evaluations are first-class gates"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "completion": dict(self.completion),
            "harmless": dict(self.harmless),
            "loss": dict(self.loss),
        }

    def all_pass(self) -> bool:
        return all(getattr(self, name).get("gate") == "PASS" for name in REQUIRED_EVALS)

    def failed_gates(self) -> tuple[str, ...]:
        return tuple(
            name for name in REQUIRED_EVALS if getattr(self, name).get("gate") != "PASS"
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "EvalBundle":
        body = dict(mapping)
        missing = [name for name in REQUIRED_EVALS if name not in body]
        if missing:
            raise MissingEvalError(
                f"missing evaluations {missing}; a direction selected on "
                "refusal suppression alone is underdetermined. Required: "
                + ",".join(REQUIRED_EVALS)
            )
        return cls(
            completion=_as_mapping(body["completion"], "completion"),
            harmless=_as_mapping(body["harmless"], "harmless"),
            loss=_as_mapping(body["loss"], "loss"),
        )


def select_by_refusal_suppression_alone(*_args: Any, **_kwargs: Any) -> None:
    """Watched refusal: the naive workbench path cannot be expressed."""
    raise UnderdeterminedSelection(
        "direction selected on refusal suppression alone is underdetermined; "
        "harmless and loss evaluations are first-class gates and cannot be omitted"
    )


def _candidate_id(row: Mapping[str, Any]) -> str:
    cid = str(row.get("id") or row.get("candidate_id") or "")
    if not cid:
        raise UnderdeterminedSelection("every candidate must carry an id")
    return cid


def select(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Multi-objective selection. Harmless and loss are gates, not afterthoughts.

    Recovered from andyrdt/refusal_direction select_direction.filter_fn:
    a candidate is discarded on NaN, on KL above threshold, on failure to
    induce refusal on harmless, or on coming from the last 20% of layers.
    Ranking by refusal suppression happens AFTER those gates, and this
    sidecar cannot perform that ranking without a specimen run — so among
    survivors the pick is a deterministic tie-break, labelled as such.
    Zero survivors is SelectionEmpty, never "pick the least-bad refusal".
    """
    if not candidates:
        raise SelectionEmpty("no candidates; refusing to invent a direction")
    rows: list[dict[str, Any]] = []
    for raw in candidates:
        cid = _candidate_id(raw)
        bundle = EvalBundle.from_mapping(raw.get("evals") if isinstance(raw.get("evals"), Mapping) else raw)
        source_layer = raw.get("source_layer")
        source_position = raw.get("source_position")
        n_layers = raw.get("n_layers")
        pruned = False
        prune_reason = None
        if isinstance(source_layer, int) and isinstance(n_layers, int) and n_layers > 0:
            cutoff = int(n_layers * (1.0 - float(ARDITI_DEFAULTS["prune_layer_percentage"])))
            if source_layer >= cutoff:
                pruned = True
                prune_reason = (
                    f"source_layer {source_layer} is in the last "
                    f"{ARDITI_DEFAULTS['prune_layer_percentage']} of {n_layers} layers "
                    "(andyrdt prune_layer_percentage); discarded as a source"
                )
        failed = bundle.failed_gates()
        rows.append(
            {
                "id": cid,
                "evals": bundle.to_dict(),
                "all_pass": bundle.all_pass() and not pruned,
                "failed_gates": list(failed) + (["source_layer_pruned"] if pruned else []),
                "source_layer": source_layer,
                "source_position": source_position,
                "n_layers": n_layers,
                "pruned_as_source": pruned,
                "prune_reason": prune_reason,
            }
        )
    surviving = [r for r in rows if r["all_pass"]]
    if not surviving:
        raise SelectionEmpty(
            "every candidate failed a first-class gate "
            "(completion, harmless, loss, or source-layer prune); "
            "refusing to select on refusal suppression of the leftovers"
        )
    # Tie-break is identity, not quality. This sidecar did not measure scores.
    surviving_sorted = sorted(
        surviving,
        key=lambda r: (
            r["source_layer"] if isinstance(r["source_layer"], int) else 1 << 30,
            r["source_position"] if isinstance(r["source_position"], int) else 1 << 30,
            r["id"],
        ),
    )
    chosen = surviving_sorted[0]
    return {
        "selected_id": chosen["id"],
        "selected": chosen,
        "n_candidates": len(rows),
        "n_surviving": len(surviving),
        "survivors": [r["id"] for r in surviving_sorted],
        "discarded": [r["id"] for r in rows if not r["all_pass"]],
        "tie_break": "source_layer, source_position, id — NOT a quality ranking",
        "tie_break_is_not_measured_quality": True,
        "gates": list(REQUIRED_EVALS),
        "ranking_by_refusal_suppression": False,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim": (
            "candidate direction selected under completion+harmless+loss gates; "
            "this is not an uncensoring result"
        ),
    }


def orthogonalize_against(v: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: v := v - (v·ĥ)ĥ. Recovered Nous --projected."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    vn = float(np.linalg.norm(v))
    hn = float(np.linalg.norm(h))
    if vn == 0.0 or hn == 0.0:
        raise UnderdeterminedSelection("projected orthogonalize refuses a zero direction")
    h_hat = h / hn
    return v - float(v @ h_hat) * h_hat


def scoped_layers(
    n_layers: int,
    *,
    blacklist: Sequence[int] | None = None,
    whitelist: Sequence[int] | None = None,
    prune_last_fraction: float | None = None,
    role: str = "destination",
) -> dict[str, Any]:
    """FailSpy blacklist/whitelist + Arditi last-fraction prune as SOURCE.

    Destination blacklist defaults to the first and last couple (FailSpy
    README: those layers have dramatic effects). Source prune defaults to
    Arditi's last 20%. A whitelist, if given, is an allow-set; blacklist
    still wins. role must be declared so a source prune cannot be silently
    reused as a destination mask.
    """
    if role not in LAYER_ROLES:
        raise PlanRefusal(f"layer role {role!r} is not source or destination")
    if not isinstance(n_layers, int) or n_layers <= 0:
        raise PlanRefusal(f"n_layers must be a positive int, got {n_layers!r}")
    all_layers = list(range(n_layers))
    blocked: set[int] = set()
    reasons: list[str] = []
    if role == "destination":
        default_bl = {0, 1, n_layers - 2, n_layers - 1} if n_layers >= 4 else {0, n_layers - 1}
        blocked |= set(int(x) for x in (blacklist if blacklist is not None else default_bl))
        reasons.append(
            "FailSpy: blacklist first and last couple destination layers by default"
        )
    if role == "source":
        frac = (
            float(ARDITI_DEFAULTS["prune_layer_percentage"])
            if prune_last_fraction is None
            else float(prune_last_fraction)
        )
        cutoff = int(n_layers * (1.0 - frac))
        blocked |= set(range(max(cutoff, 0), n_layers))
        reasons.append(
            f"andyrdt: prune last {frac} of layers as direction SOURCES (cutoff={cutoff})"
        )
        if blacklist:
            blocked |= set(int(x) for x in blacklist)
    if whitelist is not None:
        allow = set(int(x) for x in whitelist)
        extra = [L for L in all_layers if L not in allow]
        blocked |= set(extra)
        reasons.append("whitelist is an allow-set; layers outside it are blocked")
    writable = [L for L in all_layers if L not in blocked]
    if not writable:
        raise PlanRefusal(
            f"layer scoping for role={role} left zero layers; refusing an empty transform"
        )
    return {
        "role": role,
        "n_layers": n_layers,
        "writable": writable,
        "blocked": sorted(blocked),
        "reasons": reasons,
        "blacklist_wins_over_whitelist": True,
    }


def classify_specimen(name: str) -> dict[str, Any]:
    """Declared eligibility. Unknown names fail closed; a name is not a measurement."""
    if not name or not str(name).strip():
        raise PlanRefusal("specimen name is required")
    n = str(name).lower()
    instruct_in_name = any(tok in n for tok in ("instruct", "-chat", "_chat", "-it@", "-it#"))
    named_abliterated = "abliterated" in n
    for tokens, architecture, why in _INELIGIBLE:
        if all(tok in n for tok in tokens):
            weight_space = "quantized" if architecture.startswith("quantized") else "full_weight"
            return {
                "specimen": name,
                "eligible": False,
                "architecture": architecture,
                "weight_space": weight_space,
                "instruct_in_name": instruct_in_name,
                "named_abliterated": named_abliterated,
                "why": why,
            }
    for tokens, architecture, why in _ELIGIBLE:
        if all(tok in n for tok in tokens):
            return {
                "specimen": name,
                "eligible": True,
                "architecture": architecture,
                "weight_space": "full_weight",
                "instruct_in_name": instruct_in_name,
                "named_abliterated": named_abliterated,
                "why": why,
                "moe_expert_granular_not_in_recovered_method": architecture.endswith("_moe"),
            }
    return {
        "specimen": name,
        "eligible": False,
        "architecture": "unclassified",
        "weight_space": "unknown",
        "instruct_in_name": instruct_in_name,
        "named_abliterated": named_abliterated,
        "why": (
            "unclassified architecture; residual-stream decoder-only eligibility "
            "is a declared table and this name is not on it"
        ),
    }


def require_full_weight(kind: Mapping[str, Any]) -> None:
    """Weight-space projection on a quantized specimen is a category error."""
    if kind.get("weight_space") == "full_weight":
        return
    raise CategoryError(
        f"refused abliteration on weight_space={kind.get('weight_space')!r} "
        f"(architecture={kind.get('architecture')!r}): "
        "NousResearch/llm-abliteration requires full-weight models for ablation; "
        "a quantized / meta-bpw specimen is a different EBPW category"
    )


def contracts() -> dict[str, Any]:
    """Dataset, evaluation, and refusal contracts a run must satisfy.

    Recovered from andyrdt/refusal_direction pipeline/run_pipeline.py (harmful
    and harmless splits; completion, harmless, loss evals) and from
    NousResearch/llm-abliteration (custom harmful/harmless files; --projected;
    --normpreserve; full-weight ablation). Prompt TEXT is not stored here.
    """
    return {
        "schema": "hawking.future.abliteration.contracts.v1",
        "dataset": {
            "harmful_set": {
                "required": True,
                "splits": ["train", "val", "test"],
                "arditi_defaults": {
                    "n_train": ARDITI_DEFAULTS["n_train"],
                    "n_val": ARDITI_DEFAULTS["n_val"],
                    "n_test": ARDITI_DEFAULTS["n_test"],
                    "measured_here": False,
                },
                "nous_formats": [".txt", ".parquet", ".json", ".jsonl"],
                "filter": (
                    "andyrdt filter_train/filter_val: keep harmful examples whose "
                    "baseline refusal score is > 0. An empty harmful set after "
                    "filter is a run refusal, not a direction."
                ),
            },
            "harmless_set": {
                "required": True,
                "splits": ["train", "val", "test"],
                "filter": (
                    "keep harmless examples whose baseline refusal score is < 0. "
                    "An empty harmless set after filter is a run refusal."
                ),
            },
            "both_required": True,
            "harmful_only_is_underdetermined": True,
        },
        "evaluation": {
            "completion_eval": {
                "required": True,
                "on": "harmful prompts",
                "arditi": "jailbreakbench completions; substring_matching + llamaguard2",
                "purpose": "did ablating the candidate reduce explicit refusal on the target set",
            },
            "harmless_eval": {
                "required": True,
                "on": "harmless prompts",
                "arditi": (
                    "substring_matching on harmless completions; act-add of the "
                    "direction should INDUCE refusal (steering_score gate)"
                ),
                "failspy": "positive_score / preserve_harmless; mse_harmless vs cached harmless",
                "purpose": "catch over-refusal and capability collapse on the harmless distribution",
            },
            "loss_eval": {
                "required": True,
                "arditi": "CE loss on pile, alpaca, alpaca_custom_completions vs baseline",
                "failspy": "mse_harmless as a quality loss against cached harmless activations",
                "arditi_kl_gate": {
                    "kl_threshold": ARDITI_DEFAULTS["kl_threshold"],
                    "source": ARDITI_DEFAULTS["source"],
                    "measured_here": False,
                    "rule": "directions with KL above threshold are filtered out BEFORE ranking",
                },
                "purpose": "a transform that removes refusals and destroys next-token quality is a loss",
            },
        },
        "refusal_contracts": {
            "empty_harmful_after_filter": "REFUSE the run; no refusal direction exists to extract",
            "empty_harmless_after_filter": "REFUSE the run; the harmless gate cannot be computed",
            "selection_without_harmless_or_loss": "UnderdeterminedSelection",
            "all_candidates_fail_a_gate": "SelectionEmpty; do not pick the least-bad leftover",
            "artifact_claims_refusals_removed": "ClaimBoundaryError",
            "quantized_parent": "CategoryError (full-weight only)",
            "not_whole_tree_verified": "PlanRefusal",
        },
        "projection": {
            "variants": list(PROJECTIONS),
            "conventional": "W' = (I - v v^T) W — Tabula orthogonal_projection, recovered G123",
            "projected": (
                "first orthogonalize v against the harmless mean (Gram-Schmidt, "
                "Nous --projected), then conventional"
            ),
            "norm_preserving_frobenius": "Tabula norm_preserving: rescale so ||W'||_F = ||W||_F",
            "norm_preserving_biprojected": (
                "Nous modify_tensor_norm_preserved: ablate the ROW DIRECTION and "
                "restore per-row magnitude. This is not Tabula's Frobenius restore."
            ),
            "distinct_norm_preserves": True,
        },
        "layer_scoping": {
            "source_prune": "last 20% of layers discarded as direction sources (andyrdt)",
            "destination_blacklist": "first and last couple blacklisted by default (FailSpy)",
            "nous_yaml": "per-destination layer, optional different measurement layer, scale, sparsity",
            "source_and_destination_are_different_knobs": True,
        },
        "outer_scorer": {
            "module": "tools.future.tabula.evaluate",
            "axes": list(tb.SCORE_AXES),
            "rule": (
                "after a real run, the child is scored on Tabula's independent vector. "
                "Hitting the behavioural target while regressing capability / tool_use / "
                "reasoning / instruction_following is FAILURE. This module does not "
                "fork that scorer."
            ),
        },
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def claim_boundary() -> dict[str, Any]:
    """What an abliteration result may and may not assert."""
    return {
        "object": (
            "a candidate direction, applied under a stated projection, with "
            "recorded effect on BOTH the target behaviour AND general quality"
        ),
        "may_assert": [
            "a candidate direction was generated by difference-in-means on declared harmful/harmless sets",
            "a candidate survived completion + harmless + loss gates under stated hyperparameters",
            "a stated projection (conventional / projected / norm-preserving / biprojected) was applied at declared destination layers",
            "the child carries Tabula lineage and is reversible only if the invert recipe was stored",
            "upstream does not guarantee complete removal of refusals",
        ],
        "must_not_assert": [
            "the model is abliterated",
            "the model is uncensored",
            "refusals were removed",
            "refusals are gone",
            "safety alignment was deleted",
            "the model will never refuse",
            "a capability or hardware number this sidecar did not measure",
        ],
        "upstream_quote": UPSTREAM["operational"]["guarantee"],
        "upstream_source": "NousResearch/llm-abliteration README NOTE",
        "named_abliterated_is_not_a_result": (
            "a directory name containing 'abliterated' is a naming fact, not a "
            "verified behavioural outcome"
        ),
        "status_labels_are_hypotheses": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


_FORBIDDEN_CLAIM_KEYS = frozenset(
    {
        "refusals_removed",
        "refusal_free",
        "uncensored",
        "fully_abliterated",
        "safety_removed",
        "alignment_deleted",
        "never_refuses",
    }
)
_FORBIDDEN_STATUS = frozenset(
    {
        "abliterated",
        "uncensored",
        "refusal_free",
        "refusals_removed",
        "fully_abliterated",
    }
)
_FORBIDDEN_PHRASES = (
    "refusals are removed",
    "refusals were removed",
    "refusals removed",
    "fully uncensored",
    "completely uncensored",
    "guarantee complete removal",
    "refusal-free",
    "refusals are gone",
)


def admit_result(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse an artifact that claims the guarantee upstream does not give."""
    if not isinstance(artifact, Mapping):
        raise ClaimBoundaryError("artifact must be a mapping")
    status = str(artifact.get("status") or "").strip().lower().replace(" ", "_")
    if status in _FORBIDDEN_STATUS:
        raise ClaimBoundaryError(
            f"status {artifact.get('status')!r} asserts a causal outcome the "
            "method cannot guarantee; abliteration is a candidate generator, "
            "not an uncensoring switch"
        )
    for key in _FORBIDDEN_CLAIM_KEYS:
        if artifact.get(key) is True:
            raise ClaimBoundaryError(
                f"artifact sets {key}=True; upstream does not guarantee removal "
                "of refusals and this module will not manufacture that claim"
            )
    blob = json.dumps(artifact, sort_keys=True, default=str).lower()
    for phrase in _FORBIDDEN_PHRASES:
        if phrase in blob:
            raise ClaimBoundaryError(
                f"artifact text claims {phrase!r}; refused. "
                + UPSTREAM["operational"]["guarantee"]
            )
    if artifact.get("gpu_authority") is True:
        raise ClaimBoundaryError("artifact claims gpu_authority; this sidecar has none")
    try:
        _assert_no_hardware_claims(artifact)
    except HardwareClaimError as exc:
        raise ClaimBoundaryError(
            f"artifact writes capability/hardware numbers; {exc}"
        ) from exc
    numeric_caps = []
    for key, value in artifact.items():
        if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
            numeric_caps.append(key)
        if key in {"refusal_rate", "refusal_score", "ce_loss", "perplexity", "tps"} and isinstance(
            value, (int, float)
        ):
            numeric_caps.append(key)
    if numeric_caps:
        raise ClaimBoundaryError(
            f"artifact writes capability/hardware numbers {numeric_caps}; "
            "this sidecar does not measure those"
        )
    return {
        "admitted": True,
        "as": "candidate_transformation",
        "not_as": "uncensoring_switch",
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_class": "STATIC_ONLY",
    }


def method() -> dict[str, Any]:
    """Pipeline recovered from the three named references. Not a run."""
    return {
        "schema": "hawking.future.abliteration.method.v1",
        "belongs_to": "Tabula (tools.future.tabula) — behavioural-surgery science",
        "not_a_rival_floor": True,
        "stages": [
            {
                "id": "generate_directions",
                "from": "andyrdt/refusal_direction pipeline/submodules/generate_directions.py",
                "does": (
                    "mean residual-stream activation on harmful minus mean on "
                    "harmless, per (eoi position, layer). Candidates live on a "
                    "(position, layer, d_model) grid. Nous measure.py adds a "
                    "harmless direction and an optional --projected orthogonalize "
                    "at measurement time."
                ),
            },
            {
                "id": "select_direction",
                "from": "andyrdt/refusal_direction pipeline/submodules/select_direction.py",
                "does": (
                    "for every candidate, compute (1) refusal ablation score on "
                    "harmful, (2) act-add steering score on harmless, (3) KL of "
                    "harmless last-token logits vs baseline. filter_fn discards "
                    "NaN, KL above threshold, steering below threshold, last 20% "
                    "of layers. ONLY THEN sort survivors by refusal suppression. "
                    "FailSpy find_best_refusal_dir skips (2) and (3) as gates — "
                    "that path is inexpressible here."
                ),
                "this_module": "tools.future.abliteration.select",
            },
            {
                "id": "projection",
                "from": "Nous sharded_ablate.py + Tabula project()",
                "does": (
                    "conventional (I-vv^T)W; --projected Gram-Schmidt against "
                    "harmless; Frobenius restore (Tabula); row-norm biprojection "
                    "(Nous). Layer-scoped. Invert recipe stored iff reversible."
                ),
                "reuses": "tools.future.tabula.project",
            },
            {
                "id": "layer_scoping",
                "from": "FailSpy blacklist/whitelist; andyrdt source prune; Nous YAML marching orders",
                "does": "source layers and destination layers are different knobs",
                "this_module": "tools.future.abliteration.scoped_layers",
            },
            {
                "id": "outer_independent_eval",
                "from": "tools.future.tabula.evaluate",
                "does": (
                    "a child that hits the behavioural target and regresses "
                    "capability/tool_use/reasoning/instruction_following is FAILURE"
                ),
            },
        ],
        "projections": list(PROJECTIONS),
        "required_evals": list(REQUIRED_EVALS),
        "recovered_method_defaults": ARDITI_DEFAULTS,
        "upstream": UPSTREAM,
        "fails_closed": [
            "select() without completion+harmless+loss raises UnderdeterminedSelection",
            "select() with every gate failing raises SelectionEmpty",
            "select_by_refusal_suppression_alone() always raises",
            "admit_result() refuses refusals_removed / status=abliterated",
            "plan() refuses a specimen that is not whole-tree verified",
            "require_full_weight() raises CategoryError on quantized parents",
            "run() raises RunRefused; apply_to_weights stays WeightsFrozen",
        ],
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "weights_modified": False,
    }


def _verification_doc(supplied: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    if supplied is not None:
        return dict(supplied), "caller"
    doc = load_verification_doc()
    if isinstance(doc, dict):
        return doc, "external_specimen_seal.load_verification_doc"
    blob = git("show", f"HEAD:{VERIFICATION_REL}")
    if blob:
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            return None, "git:HEAD-unreadable"
        if isinstance(parsed, dict):
            return parsed, "git:HEAD"
    return None, "unlocated"


def smallest_eligible(
    verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Smallest whole-tree-verified residual-stream causal LM. Empty is a refusal."""
    doc, taken = _verification_doc(verification)
    if doc is None:
        raise PlanRefusal(
            f"{VERIFICATION_REL} {taken}; cannot establish whole-tree verification, "
            "so no specimen is eligible"
        )
    ranked: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
    for row in doc.get("results") or []:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("specimen") or "")
        if not name or not is_whole_tree_row(row):
            continue
        kind = classify_specimen(name)
        if not kind["eligible"]:
            continue
        try:
            require_full_weight(kind)
        except CategoryError:
            continue
        bytes_hashed = row.get("bytes_hashed")
        if not isinstance(bytes_hashed, int) or bytes_hashed <= 0:
            continue
        ranked.append((bytes_hashed, name, dict(row), kind))
    canonical = {name.split("#", 1)[0] for _, name, _, _ in ranked if "#" not in name}
    ranked = [
        t for t in ranked if "#" not in t[1] or t[1].split("#", 1)[0] not in canonical
    ]
    if not ranked:
        raise PlanRefusal(
            "no whole-tree-verified residual-stream full-weight causal LM is eligible"
        )
    ranked.sort(key=lambda t: (t[0], t[1]))
    bytes_hashed, name, row, kind = ranked[0]
    return {
        "specimen": name,
        "bytes_hashed": bytes_hashed,
        "n_files": row.get("n_files"),
        "owner": row.get("owner"),
        "kind": kind,
        "verification_path_taken": taken,
        "rule": "smallest bytes_hashed among whole-tree-verified eligible full-weight causal LMs",
        "n_eligible": len(ranked),
    }


def plan(
    specimen: str | None = None,
    *,
    verification: Mapping[str, Any] | None = None,
    n_layers: int | None = None,
    scars: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """What it would take to run this on a real whole-tree-verified specimen.

    Does not run it. Smallest eligible first when specimen is omitted.
    A specimen that is not whole-tree verified is refused, not rounded.
    """
    doc, taken = _verification_doc(verification)
    if doc is None:
        raise PlanRefusal(
            f"{VERIFICATION_REL} {taken}; refusing to plan against an unverified parent"
        )
    chosen_kind: dict[str, Any] | None = None
    if specimen is None:
        pick = smallest_eligible(doc)
        specimen = str(pick["specimen"])
        chosen_kind = pick["kind"]
    row = verification_row(doc, specimen)
    if not is_whole_tree_row(row):
        status = (row or {}).get("status") if row else "ABSENT"
        raise PlanRefusal(
            f"specimen {specimen!r} is not whole-tree verified "
            f"(status={status!r}, path_taken={taken}); "
            "a SIZE-only or partial parent is not a sealed parent"
        )
    kind = chosen_kind or classify_specimen(specimen)
    if not kind["eligible"]:
        raise PlanRefusal(
            f"specimen {specimen!r} is not eligible for the recovered residual-stream "
            f"method: architecture={kind['architecture']} why={kind['why']}"
        )
    require_full_weight(kind)
    layers = None
    if isinstance(n_layers, int) and n_layers > 0:
        layers = {
            "source": scoped_layers(n_layers, role="source"),
            "destination": scoped_layers(n_layers, role="destination"),
        }
    scar = ni.refuse_if_dead(
        {
            "hypothesis_family": "abliteration",
            "technique": "refusal_direction",
            "model": specimen,
        },
        scars=scars,
    )
    return {
        "specimen": specimen,
        "status": "PLAN_ONLY",
        "ran": False,
        "weights_modified": False,
        "kind": kind,
        "verification": {
            "path": VERIFICATION_REL,
            "path_taken": taken,
            "status": row.get("status") if row else None,
            "whole_tree_verified": True,
            "n_files": row.get("n_files") if row else None,
            "owner": row.get("owner") if row else None,
        },
        "named_abliterated_is_not_a_result": bool(kind.get("named_abliterated")),
        "stages": [
            "load full-weight specimen read-only (ModelLake is never mutated)",
            "tokenize declared harmful and harmless sets with the specimen chat template",
            "filter_train/filter_val on baseline refusal scores; empty set refuses the run",
            "generate_directions: mean harmful - mean harmless per (position, layer)",
            "select_direction: completion + harmless + loss gates, then tie-break",
            "project at destination layers under a stated variant; store invert recipe",
            "completion eval, harmless eval, loss eval against the unprojected parent",
            "Tabula evaluate() on the independent vector; behavioural-hit + regression = FAILURE",
            "emit lineage-bearing child; do not promote; do not claim refusals removed",
        ],
        "contracts": "tools.future.abliteration.contracts()",
        "projection_default": "norm_preserving_biprojected",
        "layer_scoping": layers,
        "resource_class": "GPU_EXCLUSIVE",
        "sleep_state": "SLEEPING",
        "wake_condition": tb.fitting_wake_condition(),
        "negative_index": {
            "invoked": True,
            "refuse_if_dead": scar,
            "rule": "a MODEL_SPECIFIC scar does not prune a different named parent",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "outer_scorer": "tools.future.tabula.evaluate",
    }


def run(*_args: Any, **_kwargs: Any) -> None:
    raise RunRefused(
        "this lane does not run a transformation on a specimen; "
        "the run is a SLEEPING GPU_EXCLUSIVE WorkUnit and this sidecar "
        "must not seize a GPU lease or invent a capability result"
    )


def apply_to_weights(*_args: Any, **_kwargs: Any) -> None:
    tb.apply_to_weights()


def projection_reuses_tabula() -> dict[str, Any]:
    """Contract proof: conventional + Frobenius restore is Tabula's project()."""
    rng = np.random.default_rng(0)
    W = tb._matrix(rng, 16, 24)
    v = tb._unit(rng, 16)
    h = tb._unit(np.random.default_rng(1), 16)
    W_conv, _, met_conv = tb.project(W, v, norm_preserve=False, store_component=False)
    W_frob, recipe, met_frob = tb.project(W, v, norm_preserve=True, store_component=True)
    v_proj = orthogonalize_against(v, h)
    # Per-row rescale after the same left projection. Distinct from Tabula's
    # single Frobenius scalar. Nous biprojection is this family, not G123's.
    row_parent = np.linalg.norm(W, axis=1, keepdims=True)
    row_proj = np.linalg.norm(W_conv, axis=1, keepdims=True)
    W_row = W_conv * (row_parent / np.clip(row_proj, 1e-12, None))
    differ = float(np.linalg.norm(W_frob - W_row, ord="fro")) > 1e-9
    return {
        "reused": "tools.future.tabula.project",
        "conventional_left_null": float(np.linalg.norm(v @ W_conv)) < 1e-8,
        "frobenius_restore_error_below_1e8": met_frob["norm_preserve_error"] < 1e-8,
        "row_norm_and_frobenius_are_distinct_operators": differ,
        "projected_orthogonalize_reduces_harmless_alignment": abs(float(v_proj @ (h / np.linalg.norm(h))))
        < abs(float(v @ (h / np.linalg.norm(h)))),
        "invert_recipe_stored": recipe is not None,
        "specimen": "synthetic_contract_proof",
        "weights_modified_on_a_real_specimen": False,
        "metrics_are_geometry_not_capability": True,
        "conventional_parent_residual_nonzero": met_conv["residual_vT_W_parent"] > 0.0,
    }


def emit_workunits(*, capture: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """CPU method unit (pending) and the specimen run (SLEEPING, GPU_EXCLUSIVE)."""
    wake = tb.fitting_wake_condition(capture)
    method_unit = emit_hcli_workunit(
        id="future.abliteration.method",
        role="science",
        description=(
            "Seal the abliteration candidate-generator: method, contracts, "
            "multi-objective selection, claim boundary. STATIC_ONLY. No specimen run."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.abliteration.method",
        provider="future.abliteration",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras={
            "species": "tabula_behavioral_surgery",
            "claim_boundary": CLAIM_BOUNDARY,
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "requires_quiescence": False,
            "may_promote": False,
            "may_modify_verifier": False,
            "may_widen_authority": False,
            "weights_modified": False,
            "command": "python3 tools/future/abliteration.py --build",
            "lane_resource": "CPU_ANALYSIS",
            "belongs_to": "tools.future.tabula",
        },
    )
    select_unit = emit_hcli_workunit(
        id="future.abliteration.select",
        role="science",
        description=(
            "Select a candidate direction under completion+harmless+loss gates. "
            "A selection that omits harmless or loss is inexpressible."
        ),
        dependencies=["future.abliteration.method"],
        resource_class="STATIC_ANALYSIS",
        verifier="future.abliteration.select",
        provider="future.abliteration",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras={
            "species": "tabula_behavioral_surgery",
            "claim_boundary": CLAIM_BOUNDARY,
            "requires_quiescence": False,
            "may_promote": False,
            "required_evals": list(REQUIRED_EVALS),
        },
    )
    run_unit = emit_hcli_workunit(
        id="future.abliteration.run-specimen",
        role="science",
        description=(
            "SLEEPING. Run the recovered pipeline on a whole-tree-verified "
            "specimen when hardware qualifies. This sidecar must not run it, "
            "must not seize a GPU lease, and must not invent a capability result."
        ),
        dependencies=["future.abliteration.method", "future.abliteration.select"],
        resource_class="GPU_EXCLUSIVE",
        verifier="future.abliteration.run.protected",
        provider="future.abliteration",
        effect_class="REVERSIBLE",
        status="sleeping",
        classification="SLEEPING",
        extras={
            "species": "tabula_behavioral_surgery",
            "claim_boundary": CLAIM_BOUNDARY,
            "sleep_state": "SLEEPING",
            "wake_condition": wake,
            "blocked_reason": (
                "no GPU authority on this sidecar host; "
                + "; ".join(tb.PHYSICAL_BLOCKERS[:3])
            ),
            "requires_quiescence": True,
            "may_promote": False,
            "may_modify_verifier": False,
            "may_widen_authority": False,
            "weights_modified": False,
            "smallest_first": True,
        },
    )
    units = [method_unit, select_unit, run_unit]
    for row in units:
        validate_emitted_unit(row)
        WorkUnit.from_dict(dict(row))
    return units


def _prove_negative_controls() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def _trial(name: str, thunk, expected: type[BaseException]) -> None:
        try:
            thunk()
        except expected as exc:
            results.append(
                {
                    "trial": name,
                    "refused": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:320],
                }
            )
            return
        raise AssertionError(f"abliteration guard did not fire for {name}")

    _trial(
        "refusals_removed_artifact",
        lambda: admit_result({"status": "ok", "refusals_removed": True}),
        ClaimBoundaryError,
    )
    _trial(
        "status_abliterated",
        lambda: admit_result({"status": "abliterated"}),
        ClaimBoundaryError,
    )
    _trial(
        "select_by_refusal_suppression_alone",
        lambda: select_by_refusal_suppression_alone([{"id": "d0", "completion": {"gate": "PASS"}}]),
        UnderdeterminedSelection,
    )
    _trial(
        "select_missing_harmless_and_loss",
        lambda: select(
            [
                {
                    "id": "d0",
                    "evals": {"completion": {"gate": "PASS"}},
                }
            ]
        ),
        UnderdeterminedSelection,
    )
    _trial(
        "select_all_gates_fail",
        lambda: select(
            [
                {
                    "id": "d0",
                    "source_layer": 3,
                    "n_layers": 10,
                    "evals": {
                        "completion": {"gate": "PASS"},
                        "harmless": {"gate": "FAIL"},
                        "loss": {"gate": "FAIL"},
                    },
                }
            ]
        ),
        SelectionEmpty,
    )
    _trial(
        "plan_unverified_specimen",
        lambda: plan(
            "not-a-verified-specimen@dead",
            verification={
                "results": [
                    {
                        "specimen": "not-a-verified-specimen@dead",
                        "status": "PARTIAL_NO_REMOTE_DIGEST",
                        "whole_tree_verified": False,
                        "n_files": 2,
                        "verified": 1,
                        "mismatched": 0,
                        "no_remote_digest": 1,
                        "unrecognized_digest": 0,
                        "skipped_time_budget": 0,
                        "bytes_hashed": 99,
                    }
                ]
            },
            scars=[],
        ),
        PlanRefusal,
    )
    _trial(
        "quantized_parent",
        lambda: require_full_weight(
            classify_specimen("microsoft--bitnet-b1.58-2B-4T@04c3b9ad9361")
        ),
        CategoryError,
    )
    _trial("run_refused", lambda: run("Qwen--Qwen3-0.6B@c1899de289a0"), RunRefused)
    _trial("apply_to_weights", lambda: apply_to_weights(), tb.WeightsFrozen)
    return results


def _scar_probe() -> dict[str, Any]:
    try:
        refusal = ni.refuse_if_dead(
            {
                "hypothesis_family": "abliteration",
                "technique": "refusal_direction",
                "model": "qwen3.8-27b",
            }
        )
        hits = ni.query(hypothesis_family="abliteration")
        return {
            "invoked": True,
            "refuse_if_dead": refusal,
            "n_hits": len(hits),
            "hit_ids": [h.get("scar_id") for h in hits[:8]],
            "empty_index_is_not_evidence_the_method_works": refusal is None,
        }
    except Exception as exc:
        return {
            "invoked": True,
            "raised": type(exc).__name__,
            "error": str(exc)[:240],
        }


def resident_callable(*, units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "entry_point": "tools.future.abliteration.method()",
        "callables": [
            "tools.future.abliteration.method()",
            "tools.future.abliteration.plan(specimen)",
            "tools.future.abliteration.contracts()",
            "tools.future.abliteration.claim_boundary()",
            "tools.future.abliteration.select(candidates)",
        ],
        "workunit": (
            "one STATIC_ANALYSIS (CPU_ANALYSIS lane) unit for method/contracts/"
            "plan; the specimen run is a SLEEPING GPU_EXCLUSIVE unit this "
            "sidecar must not wake"
        ),
        "workunit_emitted": [row["id"] for row in units],
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier": "FT.MODEL_CAPABILITY.hard-gates",
        "belongs_to": "tools.future.tabula",
        "this_lane_writes_frontier": False,
        "fails_closed": (
            "select() raises UnderdeterminedSelection without harmless+loss; "
            "admit_result() raises ClaimBoundaryError on refusals_removed; "
            "plan() raises PlanRefusal on a specimen that is not whole-tree "
            "verified; require_full_weight() raises CategoryError on quantized "
            "parents; run() raises RunRefused; write_receipt raises "
            "HardwareClaimError on a numeric hardware field"
        ),
    }


def build() -> Path:
    refusals = _prove_negative_controls()
    if not refusals or not all(r.get("refused") for r in refusals):
        raise AssertionError(f"negative controls did not all fire: {refusals}")
    units = emit_workunits()
    if not tb.sleeping_unit_is_not_ready(units):
        raise AssertionError("sleeping specimen-run unit was ready; HCLI would dispatch a GPU fit")
    geometry = projection_reuses_tabula()
    passing = {
        "id": "cand-mid",
        "source_layer": 4,
        "source_position": -1,
        "n_layers": 12,
        "evals": {
            "completion": {"gate": "PASS"},
            "harmless": {"gate": "PASS"},
            "loss": {"gate": "PASS"},
        },
    }
    pruned = {
        "id": "cand-tail",
        "source_layer": 11,
        "source_position": -1,
        "n_layers": 12,
        "evals": {
            "completion": {"gate": "PASS"},
            "harmless": {"gate": "PASS"},
            "loss": {"gate": "PASS"},
        },
    }
    selection = select([pruned, passing])
    admitted = admit_result(
        {
            "status": "CANDIDATE",
            "selected_id": selection["selected_id"],
            "projection": "norm_preserving_biprojected",
            "gpu_authority": False,
        }
    )
    scar_probe = _scar_probe()
    planned: dict[str, Any] | None
    plan_error: str | None = None
    try:
        planned = plan()
    except PlanRefusal as exc:
        planned = None
        plan_error = str(exc)

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": SIDECAR_STATUS,
        "promoted": False,
        "built": True,
        "purpose": (
            "Candidate transformation generator for Tabula behavioural surgery. "
            "Method, contracts, multi-objective selection, claim boundary. "
            "No specimen run. No weight write."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "claim_rules": claim_boundary(),
        "method": method(),
        "contracts": contracts(),
        "selection_demo": selection,
        "admitted_candidate": admitted,
        "geometry_proof": geometry,
        "plan": planned,
        "plan_error": plan_error,
        "work_units": list(units),
        "refusals_proven": list(refusals),
        "negative_index": scar_probe,
        "weights_modified": False,
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "recovered_implementation": [
            "tools/future/tabula.py — orthogonal projection, Frobenius restore, independent vector, lineage, frozen weights, sleeping GPU fit",
            "tools/future/specimen_verify.py — WHOLE_TREE_VERIFIED is recomputed digests, never size",
            "tools/future/external_specimen_seal.py — is_whole_tree_row / load_verification_doc (status is a hypothesis until the row is strict)",
            "tools/future/negative_index.py — refuse_if_dead / query before proposing",
            "tools/future/ebpw_categories.py — CategoryError: quantized parent is the wrong type for a weight-space projection",
            "NousResearch/llm-abliteration — measure/analyze/sharded_ablate, --projected, --normpreserve, full-weight rule, no-guarantee NOTE",
            "andyrdt/refusal_direction — generate_directions, select_direction.filter_fn, completion/harmless/loss evals",
            "FailSpy/abliterator — temp contexts, cache_activations, blacklist/whitelist, mse_harmless, barebones find_best_refusal_dir",
        ],
        "gaps_closed": [
            "no candidate-generator for Tabula that recovered direction SELECTION as a multi-objective gate",
            "a selection that ignored harmless or loss evaluation could be expressed; it cannot be now",
            "no claim-boundary guard against 'abliterated' / 'refusals_removed' artifacts",
            "no plan() gated on whole-tree specimen verification and full-weight category",
            "Frobenius restore and row-norm biprojection were being conflated; they are distinct operators",
            "source-layer prune and destination blacklist were not distinguished",
        ],
        "negative_findings": [
            "upstream does not guarantee complete removal of refusals; this module will not manufacture that guarantee",
            "FailSpy's find_best_refusal_dir is a single-score sort and is treated as a negative control, not a method",
            "this host has no GPU authority; the specimen run stays SLEEPING and is not simulated as a result",
            "a directory named 'abliterated' is a naming fact, not a verified behavioural outcome",
            "expert-granular MoE ablation is not in the recovered method and is not claimed",
            "this lane produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE",
            "this lane cannot write tools/future/orchestration.py BINDINGS; orchestration.invoke('abliteration.py') will raise UnknownBinding until a later connector bind. emit_workunits() still produces HCLI-shaped units locally",
            plan_error or "smallest eligible whole-tree-verified specimen produced a PLAN_ONLY (not a run)",
        ],
        "resident_callable": resident_callable(units=units),
        "promote_exists": False,
    }
    _assert_no_hardware_claims(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
