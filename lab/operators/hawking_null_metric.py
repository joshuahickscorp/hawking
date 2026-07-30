"""Recomposed science module hawking_null_metric (C-SCI-R1)."""
from __future__ import annotations
import sys as _sys_a1
from pathlib import Path as _Path_a1
import json
import numpy as np
_A1_HERE = _Path_a1(__file__).resolve().parent
_A1_CONDENSE = _A1_HERE.parent if _A1_HERE.name == 'archive' else _A1_HERE
_A1_REPO = _A1_CONDENSE.parents[1]
if str(_A1_CONDENSE) not in _sys_a1.path:
    _sys_a1.path.insert(0, str(_A1_CONDENSE))
SCHEMA = 'hawking.null_corrected_metric.v1'
GATE_SKILL_LOWER = 0.0
GATE_CENTERED_COSINE = 0.5
BOOTSTRAP_RESAMPLES = 512

def _flat(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    return array.reshape(-1, array.shape[-1])

def fit_null(y_fit: np.ndarray) -> dict:
    """Everything the held-out score is allowed to know about the target distribution."""
    flat = _flat(y_fit)
    return {'mean': flat.mean(axis=0), 'count': int(flat.shape[0])}

def _sse_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    difference = a - b
    return np.einsum('ij,ij->i', difference, difference)

def _bootstrap_skill_lower(sse_candidate: np.ndarray, sse_null: np.ndarray, *, resamples: int, alpha: float, seed: int) -> float:
    """Percentile lower bound on the ratio of sums, resampling held-out rows.

    The statistic is a ratio of sums rather than a mean of ratios, because a row whose
    null SSE is near zero would otherwise dominate.
    """
    generator = np.random.default_rng(seed)
    count = sse_candidate.shape[0]
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        pick = generator.integers(0, count, count)
        draws[index] = 1.0 - sse_candidate[pick].sum() / max(sse_null[pick].sum(), 1e-30)
    return float(np.quantile(draws, alpha))

def score(y_true: np.ndarray, y_pred: np.ndarray, null: dict, *, alpha: float=0.05, seed: int=0, resamples: int=BOOTSTRAP_RESAMPLES) -> dict:
    """Score one candidate against a null that was fitted elsewhere."""
    truth = _flat(y_true)
    prediction = _flat(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError(f'shape mismatch: {truth.shape} vs {prediction.shape}')
    mean = np.asarray(null['mean'], dtype=np.float64)
    centred_truth = truth - mean
    centred_prediction = prediction - mean
    denominator = max(float(np.linalg.norm(centred_truth)) * float(np.linalg.norm(centred_prediction)), 1e-30)
    centered_cosine = float(np.tensordot(centred_truth, centred_prediction, axes=2) / denominator)
    raw_denominator = max(float(np.linalg.norm(truth)) * float(np.linalg.norm(prediction)), 1e-30)
    raw_cosine = float(np.tensordot(truth, prediction, axes=2) / raw_denominator)
    sse_candidate = _sse_per_row(prediction, truth)
    sse_null = _sse_per_row(np.broadcast_to(mean, truth.shape), truth)
    total_null = max(float(sse_null.sum()), 1e-30)
    skill = float(1.0 - sse_candidate.sum() / total_null)
    lower = _bootstrap_skill_lower(sse_candidate, sse_null, resamples=resamples, alpha=alpha, seed=seed)
    truth_norm = max(float(np.linalg.norm(truth)), 1e-30)
    relative_l2 = float(np.linalg.norm(prediction - truth) / truth_norm)
    rmse = float(np.sqrt(sse_candidate.sum() / truth.size))
    normalized_rmse = float(rmse / max(float(truth.std()), 1e-30))
    per_row_skill = 1.0 - sse_candidate / np.maximum(sse_null, 1e-30)
    return {'centered_cosine': centered_cosine, 'raw_cosine': raw_cosine, 'skill': skill, 'skill_lower': lower, 'alpha': alpha, 'relative_l2': relative_l2, 'rmse': rmse, 'normalized_rmse': normalized_rmse, 'positions': int(truth.shape[0]), 'per_position_skill': {'p05': float(np.quantile(per_row_skill, 0.05)), 'median': float(np.median(per_row_skill)), 'p95': float(np.quantile(per_row_skill, 0.95)), 'fraction_beating_null': float((per_row_skill > 0).mean())}, 'passes': bool(lower > GATE_SKILL_LOWER and centered_cosine >= GATE_CENTERED_COSINE), 'gate': {'skill_lower_above': GATE_SKILL_LOWER, 'centered_cosine_at_least': GATE_CENTERED_COSINE}, 'schema': SCHEMA}

def constant_null_raw_cosine(y_true: np.ndarray, null: dict) -> float:
    """What the broken metric would have said about predicting the fit-split constant."""
    truth = _flat(y_true)
    mean = np.broadcast_to(np.asarray(null['mean'], dtype=np.float64), truth.shape)
    return float(np.tensordot(truth, mean, axes=2) / max(float(np.linalg.norm(truth)) * float(np.linalg.norm(mean)), 1e-30))
