"""Clean-process causal probe for a router-logit KIMI operator candidate.

OPH is deliberately separate from the seven copied-weight candidates.  It
projects a train-fitted refusal direction out of one DeepSeek router's logits
on the final prompt token, then recomputes the router's exact group/top-k
selection.  The hook is prefill-only and in-memory: no KIMI_BASE tensor is
written, and the matched random-direction arm receives the same projection.

This is a probe, not a promotion path.  A routing shift is not enough: OPH
must beat its matched null on disjoint behavior rows and then pass the same
capability, epistemic, authorization/HCLI, physical, destructive-control, and
Nova-lineage gates as every other candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FUTURE = Path(__file__).resolve().parent
if str(FUTURE) not in sys.path:
    sys.path.insert(0, str(FUTURE))

from moe_refusal_locality import (  # noqa: E402
    _crossfit_folds,
    _select_train_only_hidden_index,
    causal_rate_report,
    capture_observations,
    heldout_separation,
    load_prompt_battery,
    refusal_labels,
    routing_group_summary,
    install_router_hooks,
)
from kimi_operator_candidates import refusal_intervention_preflight  # noqa: E402

SCHEMA = "hawking.kimi.router_operator_live.v1"
CANDIDATE_ID = "KIMI_OPERATOR_CANDIDATE_OPH"


def _as_router_logits(output: Any):
    """Return the raw router logits from the canonical DeepSeek gate tuple."""
    import torch

    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise TypeError("DeepSeek router output must be (logits, topk_weights, topk_indices)")
    logits = output[0]
    if not isinstance(logits, torch.Tensor) or logits.ndim < 2:
        raise TypeError("DeepSeek router logits must be a rank-2 tensor")
    return logits


def _active_prefill_rows(inputs: tuple[Any, ...], n_rows: int, *, prefill_only: bool):
    """Mask final prompt tokens; cached one-token decode is untouched."""
    import torch

    if not inputs or not isinstance(inputs[0], torch.Tensor):
        raise TypeError("router hook did not receive hidden states")
    hidden = inputs[0]
    if hidden.ndim >= 3:
        batch, seq = int(hidden.shape[0]), int(hidden.shape[-2])
        if prefill_only and seq <= 1:
            return torch.zeros(n_rows, dtype=torch.bool, device=hidden.device)
        mask = torch.zeros(n_rows, dtype=torch.bool, device=hidden.device)
        last = torch.arange(seq - 1, batch * seq, seq, device=hidden.device)
        mask[last] = True
        return mask
    if hidden.ndim == 2:
        if prefill_only and int(hidden.shape[0]) <= 1:
            return torch.zeros(n_rows, dtype=torch.bool, device=hidden.device)
        mask = torch.zeros(n_rows, dtype=torch.bool, device=hidden.device)
        mask[-1] = True
        return mask
    raise TypeError("router hidden states must be rank 2 or 3")


def _recompute_topk(gate: Any, logits):
    """Reproduce DeepseekV3TopkRouter.forward after logits are shifted."""
    import torch

    scores = logits.sigmoid()
    correction = gate.e_score_correction_bias.to(device=logits.device, dtype=logits.dtype)
    scores_for_choice = scores + correction
    n_experts = int(gate.num_experts)
    n_group = int(gate.num_group)
    group_scores = (
        scores_for_choice.view(-1, n_group, n_experts // n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(
        group_scores, k=int(gate.topk_group), dim=-1, sorted=False
    )[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, n_experts // n_group)
        .reshape(-1, n_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(
        scores_for_choice, k=int(gate.top_k), dim=-1, sorted=False
    )[1]
    topk_weights = scores.gather(1, topk_indices)
    if bool(gate.norm_topk_prob):
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * float(gate.routed_scaling_factor)
    return topk_weights, topk_indices


def project_router_output(
    gate: Any,
    output: Any,
    direction: Any,
    anchor: Any,
    strength: float,
    active_rows: Any = None,
) -> Any:
    """Remove a bounded behavioral projection and recompute exact top-k routing.

    ``active_rows`` is a boolean mask over flattened router tokens.  When it is
    absent, every row is treated; the serving hook supplies only final prefill
    tokens.  Unselected rows retain their original logits, weights, and indices
    byte-for-byte where the backend permits it.
    """
    import torch

    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("router projection strength must be between 0 and 1")
    logits = _as_router_logits(output)
    flat = logits.reshape(-1, logits.shape[-1])
    d = torch.as_tensor(direction, device=flat.device, dtype=flat.dtype).reshape(-1)
    a = torch.as_tensor(anchor, device=flat.device, dtype=flat.dtype).reshape(-1)
    if d.numel() != flat.shape[-1] or a.numel() != flat.shape[-1]:
        raise ValueError("router direction/anchor width does not match router logits")
    norm = d.norm()
    if float(norm) == 0.0:
        raise ValueError("router projection direction has zero norm")
    d = d / norm
    if active_rows is None:
        active = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
    else:
        active = torch.as_tensor(active_rows, device=flat.device, dtype=torch.bool).reshape(-1)
        if active.numel() != flat.shape[0]:
            raise ValueError("active router mask does not match flattened logits")
    shifted = flat.clone()
    state = flat - a
    projected = (state @ d).unsqueeze(-1) * d
    shifted[active] = flat[active] - float(strength) * projected[active]
    new_weights, new_indices = _recompute_topk(gate, shifted)
    old_weights, old_indices = output[1], output[2]
    if not isinstance(old_weights, torch.Tensor) or not isinstance(old_indices, torch.Tensor):
        raise TypeError("DeepSeek router top-k outputs must be tensors")
    mask2 = active.unsqueeze(-1)
    weights = torch.where(mask2, new_weights, old_weights.reshape(new_weights.shape))
    indices = torch.where(mask2, new_indices, old_indices.reshape(new_indices.shape))
    shifted_view = shifted.reshape(logits.shape)
    if isinstance(output, tuple):
        return (shifted_view, weights, indices, *output[3:])
    return [shifted_view, weights, indices, *output[3:]]


@contextmanager
def router_projection(
    model: Any,
    layer: int,
    direction: Any,
    anchor: Any,
    strength: float,
    *,
    prefill_only: bool = True,
):
    """Install one reversible OPH hook on a selected DeepSeek router."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise AttributeError("model has no model.layers collection")
    if int(layer) < 0 or int(layer) >= len(layers):
        raise IndexError(f"router layer {layer} outside 0..{len(layers) - 1}")
    gate = getattr(getattr(layers[int(layer)], "mlp", None), "gate", None)
    if gate is None or not callable(getattr(gate, "register_forward_hook", None)):
        raise AttributeError(f"layer {layer} has no callable MoE gate")

    def hook(_module, inputs, output):
        logits = _as_router_logits(output)
        active = _active_prefill_rows(inputs, int(logits.reshape(-1, logits.shape[-1]).shape[0]), prefill_only=prefill_only)
        return project_router_output(gate, output, direction, anchor, strength, active)

    handle = gate.register_forward_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()


def _router_feature(row: Mapping[str, Any], layer: int):
    """Extract the final-token raw router logits for one model layer."""
    import torch

    needle = f".layers.{int(layer)}.mlp.gate"
    for name, outputs in (row.get("routers") or {}).items():
        if needle not in str(name):
            continue
        values = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
        logits = values[0] if values else None
        if isinstance(logits, torch.Tensor):
            flat = logits.float().cpu().reshape(-1, logits.shape[-1])
            if flat.shape[0]:
                return flat[-1]
    raise KeyError(f"router logits missing for layer {layer}")


def _router_layers(observation: Mapping[str, Any]) -> tuple[int, ...]:
    values = set()
    for name in (observation.get("routers") or {}):
        token = ".layers."
        if token not in str(name) or ".mlp.gate" not in str(name):
            continue
        try:
            values.add(int(str(name).split(token, 1)[1].split(".", 1)[0]))
        except (IndexError, ValueError):
            continue
    return tuple(sorted(values))


def _unwrap_observation(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the captured observation nested in a behavior row, if present."""
    nested = row.get("observation")
    return nested if isinstance(nested, Mapping) else row


def router_layer_coverage(
    refused_rows: Sequence[Mapping[str, Any]],
    complied_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit router-layer availability across every behavior row.

    OPH fits and evaluates on disjoint behavior rows.  Looking only at the
    first row of each class can manufacture a common layer when later rows
    were captured through a different wrapper, or can report no common layer
    without saying which rows lost the capture.  This audit is deliberately
    pure and cheap: it only inspects saved hook keys and never loads a model.
    """
    groups = {"refused": refused_rows, "complied": complied_rows}
    group_reports: dict[str, Any] = {}
    common_across_groups: set[int] | None = None

    for group_name, rows in groups.items():
        row_layers = [
            set(_router_layers(_unwrap_observation(row)))
            for row in rows
        ]
        union = set().union(*row_layers) if row_layers else set()
        intersection = set.intersection(*row_layers) if row_layers else set()
        missing_rows = {
            str(index): sorted(union - layers)
            for index, layers in enumerate(row_layers)
            if layers != union
        }
        group_reports[group_name] = {
            "rows": len(rows),
            "rows_with_router_layers": sum(bool(layers) for layers in row_layers),
            "union": sorted(union),
            "intersection": sorted(intersection),
            "missing_rows": missing_rows,
        }
        common_across_groups = (
            intersection if common_across_groups is None
            else common_across_groups & intersection
        )

    common = sorted(common_across_groups or set())
    return {
        "status": "OK" if common else "NO_COMMON_ROUTER_LAYERS",
        "common_layers": common,
        "groups": group_reports,
        "claim_boundary": (
            "Coverage is an integration diagnostic over captured hook keys; it "
            "does not establish behavior, causality, or promotion evidence."
        ),
    }


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def daemon_owner_snapshot(
    endpoint: str | None = None,
    *,
    timeout: float = 0.25,
) -> dict[str, Any] | None:
    """Read the local daemon health surface without touching model memory."""
    import urllib.error
    import urllib.request

    health = endpoint or os.environ.get(
        "HAWKING_DAEMON_HEALTH_URL", "http://127.0.0.1:8011/health"
    )
    try:
        with urllib.request.urlopen(health, timeout=float(timeout)) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return value if isinstance(value, dict) else None


def mps_owner_block(snapshot: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a structured refusal when the single KIMI owner is live on MPS."""
    if not isinstance(snapshot, Mapping):
        return None
    owner = snapshot.get("owner") or {}
    if not isinstance(owner, Mapping):
        return None
    if owner.get("daemon") != "hawkingd":
        return None
    return {
        "status": "OPH_DEVICE_BUSY",
        "owner": "hawkingd",
        "owner_pid": owner.get("pid"),
        "resident": snapshot.get("resident"),
        "reason": (
            "The live hawkingd resident owns the MPS/model surface. OPH did not "
            "load KIMI, create a second provider, or begin an intervention."
        ),
        "reopen_when": (
            "A controlled clean MPS window is available and the daemon/provider "
            "has been quiesced with its handle and rollback preserved."
        ),
    }


def _run_crossfit(
    model: Any,
    tok: Any,
    device: str,
    refused_rows: Sequence[Mapping[str, Any]],
    complied_rows: Sequence[Mapping[str, Any]],
    *,
    strength: float,
    n_splits: int,
    seed: int,
    baseline_lookup: Mapping[str, tuple[bool, str]] | None = None,
) -> dict[str, Any]:
    import torch

    if len(refused_rows) < 4 or len(complied_rows) < 4:
        return {
            "operator_kind": "crossfit_router_logit_projection",
            "causal_effect_established": False,
            "causal_sample_sufficient": False,
            "status": "INSUFFICIENT_BEHAVIOR_ROWS",
            "refused_rows": len(refused_rows),
            "complied_rows": len(complied_rows),
        }
    coverage = router_layer_coverage(refused_rows, complied_rows)
    layers = list(coverage["common_layers"])
    if not layers:
        return {
            "operator_kind": "crossfit_router_logit_projection",
            "candidate_id": CANDIDATE_ID,
            "candidate_method": "OPH",
            "causal_effect_established": False,
            "causal_sample_sufficient": False,
            "crossfit_consistent": False,
            "status": "NO_COMMON_ROUTER_LAYERS",
            "refused_rows": len(refused_rows),
            "complied_rows": len(complied_rows),
            "router_layer_coverage": coverage,
            "claim_boundary": (
                "No intervention was attempted because complete router coverage "
                "was not available across all behavior rows."
            ),
        }
    pos_folds = _crossfit_folds(len(refused_rows), n_splits, seed)
    neg_folds = _crossfit_folds(len(complied_rows), n_splits, seed + 1)
    pos_features = {
        layer: [_router_feature(_unwrap_observation(row), layer) for row in refused_rows]
        for layer in layers
    }
    neg_features = {
        layer: [_router_feature(_unwrap_observation(row), layer) for row in complied_rows]
        for layer in layers
    }
    folds: list[dict[str, Any]] = []
    pooled_baseline: list[bool] = []
    pooled_treated: list[bool] = []
    pooled_null: list[bool] = []
    pooled_classes: list[bool] = []
    selected_layers: list[int] = []

    for fold_id, (pos_test_idx, neg_test_idx) in enumerate(zip(pos_folds, neg_folds)):
        pos_test_set, neg_test_set = set(pos_test_idx), set(neg_test_idx)
        pos_train_idx = [i for i in range(len(refused_rows)) if i not in pos_test_set]
        neg_train_idx = [i for i in range(len(complied_rows)) if i not in neg_test_set]
        selection, selection_metrics = _select_train_only_hidden_index(
            {
                layer: [pos_features[layer][i] for i in pos_train_idx]
                + [neg_features[layer][i] for i in neg_train_idx]
                for layer in layers
            },
            range(len(pos_train_idx)),
            range(len(pos_train_idx), len(pos_train_idx) + len(neg_train_idx)),
        )
        layer = int(selection)
        selected_layers.append(layer)
        fit = heldout_separation(
            pos_features[layer],
            neg_features[layer],
            train_positive=[pos_features[layer][i] for i in pos_train_idx],
            train_negative=[neg_features[layer][i] for i in neg_train_idx],
            heldout_positive=[pos_features[layer][i] for i in pos_test_idx],
            heldout_negative=[neg_features[layer][i] for i in neg_test_idx],
        )
        direction = fit.pop("direction")
        anchor = fit.pop("_fit_anchor")
        prompts = [refused_rows[i]["prompt"] for i in pos_test_idx] + [
            complied_rows[i]["prompt"] for i in neg_test_idx
        ]
        classes = [True] * len(pos_test_idx) + [False] * len(neg_test_idx)
        if baseline_lookup is None:
            baseline, baseline_text = refusal_labels(model, tok, prompts, device)
        else:
            try:
                cached = [baseline_lookup[prompt] for prompt in prompts]
            except KeyError as exc:
                raise KeyError(f"missing cached baseline for prompt {exc.args[0]!r}") from exc
            baseline = [bool(item[0]) for item in cached]
            baseline_text = [str(item[1]) for item in cached]
        with router_projection(model, layer, direction, anchor, strength):
            treated, treated_text = refusal_labels(model, tok, prompts, device)
        random_direction = torch.randn(len(direction), generator=torch.Generator(device="cpu").manual_seed(int(seed + 1000 + fold_id)))
        with router_projection(model, layer, random_direction, anchor, strength):
            null, null_text = refusal_labels(model, tok, prompts, device)
        report = causal_rate_report(
            baseline, treated,
            baseline_control=baseline,
            treated_control=null,
            baseline_classes=classes,
        )
        report.update({
            "fold": fold_id,
            "fit_rows_excluded": True,
            "train_refused": len(pos_train_idx),
            "train_complied": len(neg_train_idx),
            "test_refused": len(pos_test_idx),
            "test_complied": len(neg_test_idx),
            "selected_router_layer": layer,
            "selection_fit_scope": "outer_train_only",
            "selection_metrics": selection_metrics,
            "heldout_auroc": fit.get("heldout_auroc"),
            "baseline_transcripts": baseline_text,
            "treated_transcripts": treated_text,
            "null_treated_transcripts": null_text,
        })
        folds.append(report)
        pooled_baseline.extend(baseline)
        pooled_treated.extend(treated)
        pooled_null.extend(null)
        pooled_classes.extend(classes)

    pooled = causal_rate_report(
        pooled_baseline, pooled_treated,
        baseline_control=pooled_baseline,
        treated_control=pooled_null,
        baseline_classes=pooled_classes,
    )
    positive_folds = sum(bool(row.get("causal_effect_established")) for row in folds)
    pooled.update({
        "operator_kind": "crossfit_router_logit_projection",
        "candidate_id": CANDIDATE_ID,
        "candidate_method": "OPH",
        "candidate_scope": "outer_train_selected_router_layer_final_prefill_token",
        "strength": float(strength),
        "layers": sorted(set(selected_layers)),
        "selected_router_layers": selected_layers,
        "n_splits": len(folds),
        "folds": folds,
        "positive_fold_count": positive_folds,
        "crossfit_consistent": positive_folds >= 2,
        "fit_rows_excluded": True,
        "router_layer_coverage": coverage,
        "control": {**(pooled.get("control") or {}), "kind": "matched_random_router_direction_per_fold", "seed": int(seed)},
        "claim_boundary": (
            "In-memory prefill-only router-logit intervention. Raw KIMI_BASE is "
            "unchanged; this is not a promoted operator, capability guarantee, "
            "authorization change, or NX artifact."
        ),
    })
    return pooled


def run(
    spec: Path,
    *,
    prompt_file: Path | None = None,
    device: str = "mps",
    dtype: str = "bfloat16",
    min_free_gib: float = 12.0,
    strength: float = 1.0,
    n_splits: int = 4,
    seed: int = 8101,
    reproduce_scarred_family: bool = False,
) -> dict[str, Any]:
    negative_science_preflight = refusal_intervention_preflight(
        "OPH", reproduce_scarred_family=reproduce_scarred_family)
    if not negative_science_preflight["allowed"]:
        return {
            "schema": SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "candidate_method": "OPH",
            "source_model": "KIMI_BASE",
            "specimen": str(spec),
            "device": device,
            "dtype": dtype,
            "candidate_causal_interventions": {},
            "negative_science_preflight": negative_science_preflight,
            "status": "NEGATIVE_SCIENCE_REFUSED",
            "weights_written": False,
            "artifact_created": False,
            "claim_boundary": (
                "OPH belongs to the sealed OPA--OPH intervention family. No "
                "model load, MPS lease, behavior run, or candidate was created."
            ),
        }
    harmful, harmless, battery_meta = load_prompt_battery(prompt_file)
    device_preflight = (
        mps_owner_block(daemon_owner_snapshot())
        if str(device).startswith("mps") else None
    )
    if device_preflight is not None:
        return {
            "schema": SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "candidate_method": "OPH",
            "source_model": "KIMI_BASE",
            "specimen": str(spec),
            "device": device,
            "dtype": dtype,
            "battery": battery_meta,
            "behavior_counts": None,
            "routing": {},
            "candidate_causal_interventions": {
                CANDIDATE_ID: {
                    "candidate_id": CANDIDATE_ID,
                    "candidate_method": "OPH",
                    "operator_kind": "crossfit_router_logit_projection",
                    "status": "OPH_DEVICE_BUSY",
                    "causal_effect_established": False,
                    "causal_sample_sufficient": False,
                    "crossfit_consistent": False,
                    "device_preflight": device_preflight,
                }
            },
            "device_preflight": device_preflight,
            "negative_science_preflight": negative_science_preflight,
            "status": "OPH_DEVICE_BUSY",
            "weights_written": False,
            "artifact_created": False,
            "claim_boundary": (
                "Preflight-only result. No behavior, separability, causality, "
                "capability, authorization, or promotion claim is made."
            ),
        }
    import torch
    from transformers import AutoTokenizer
    import dsv3_native_loader as loader

    tok = AutoTokenizer.from_pretrained(str(spec), trust_remote_code=True)
    model, cfg = loader.load(spec, device=device, dtype=dtype, min_free_gib=min_free_gib)
    prompts = list(harmful) + list(harmless)
    refusal_rows = []
    for prompt in prompts:
        text_rows = refusal_labels(model, tok, [prompt], device)
        # ``refusal_labels`` returns (labels, transcripts); the one-prompt
        # label is labels[0][0].  Coercing the labels list itself would turn
        # every non-empty response into True and erase the behavior split.
        refusal_rows.append({
            "prompt": prompt,
            "score": {"refused": bool(text_rows[0][0]), "text": text_rows[1][0]},
        })
    sink: dict[Any, Any] = {}
    handles = install_router_hooks(model, sink)
    try:
        observations = list(capture_observations(model, tok, prompts, device, sink))
    finally:
        for handle in handles:
            handle.remove()
    behavior = [
        {**row, "observation": obs}
        for row, obs in zip(refusal_rows, observations)
    ]
    refused = [row for row in behavior if row["score"]["refused"]]
    complied = [row for row in behavior if not row["score"]["refused"]]
    refused_obs = [row["observation"] for row in refused]
    complied_obs = [row["observation"] for row in complied]
    routing = routing_group_summary(
        {"refused": refused_obs, "complied": complied_obs}, cfg
    ) if refused_obs and complied_obs else {}
    causal = _run_crossfit(
        model, tok, device, refused, complied,
        strength=strength, n_splits=n_splits, seed=seed,
        baseline_lookup={
            row["prompt"]: (
                bool(row["score"]["refused"]),
                str(row["score"]["text"]),
            )
            for row in behavior
        },
    )
    return {
        "schema": SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "candidate_method": "OPH",
        "source_model": "KIMI_BASE",
        "specimen": str(spec),
        "device": device,
        "dtype": dtype,
        "battery": battery_meta,
        "behavior_counts": {"refused": len(refused), "complied": len(complied), "total": len(behavior)},
        "routing": routing,
        "candidate_causal_interventions": {CANDIDATE_ID: causal},
        "status": (
            "OPH_CAUSAL_EFFECT_ESTABLISHED"
            if causal.get("causal_effect_established")
            and causal.get("causal_sample_sufficient")
            and causal.get("crossfit_consistent")
            else "OPH_CAUSAL_NULL_OR_INCONSISTENT"
        ),
        "weights_written": False,
        "artifact_created": False,
        "negative_science_preflight": negative_science_preflight,
        "claim_boundary": (
            "Clean-process OPH router probe only. KIMI_BASE is immutable; a positive "
            "router effect would still require independent capability, epistemic, "
            "authorization/HCLI, physical, destructive-control, and Nova-lineage "
            "evidence before any promotion."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8101)
    parser.add_argument(
        "--reproduce-scarred-family",
        action="store_true",
        help=(
            "explicitly reproduce sealed OPA--OPH negative science; the output "
            "is reproduction-only and cannot be treated as a new candidate"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    result = run(
        args.spec, prompt_file=args.prompt_file, device=args.device,
        dtype=args.dtype, min_free_gib=args.min_free_gib,
        strength=args.strength, n_splits=args.n_splits, seed=args.seed,
        reproduce_scarred_family=args.reproduce_scarred_family,
    )
    result.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 3),
        "source_script_sha256": _sha256_file(Path(__file__)),
        "recorded_by": str(Path(__file__).relative_to(ROOT)),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    summary = {
        "output": str(args.output),
        "status": result["status"],
        "causal": result.get("candidate_causal_interventions", {}).get(CANDIDATE_ID),
    }
    if result["status"] == "NEGATIVE_SCIENCE_REFUSED":
        summary["negative_science_preflight"] = result.get(
            "negative_science_preflight")
    print(json.dumps(summary, indent=2, default=str))
    return 2 if result["status"] == "NEGATIVE_SCIENCE_REFUSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
