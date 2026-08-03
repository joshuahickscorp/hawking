from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "tools" / "llama_functional_student_fit.py"
SPEC = importlib.util.spec_from_file_location("llama_functional_student_fit", MODULE_PATH)
assert SPEC and SPEC.loader
fit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fit
SPEC.loader.exec_module(fit)


def paired(rows: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(rows, 6)).astype(np.float32)
    # A rank-two teacher mapping gives this capability test a known answer.
    left = np.array([[1.0, -2.0], [0.5, 0.25], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0], [-1.0, 0.5]], dtype=np.float32)
    right = np.array([[1.0, 2.0, -1.0, 0.0], [0.0, 0.5, 1.0, -2.0]], dtype=np.float32)
    y = x @ left @ right + np.array([0.2, -0.1, 0.5, 1.0], dtype=np.float32)
    heldout = np.zeros(rows, dtype=np.bool_); heldout[::5] = True
    return x, y, heldout


def test_low_rank_gate_scores_unseen_teacher_outputs() -> None:
    x, y, heldout = paired()
    result = fit.fit_low_rank(x, y, heldout, rank=2)
    assert result["heldout_normalized_rmse"] < 1e-5
    assert result["physical"]["executed_macs_per_token"] == 6 * 2 + 2 * 4


def test_small_capture_refuses_promotion_by_default(tmp_path: Path) -> None:
    x, y, heldout = paired(20)
    dataset = tmp_path / "pairs.npz"; np.savez_compressed(dataset, inputs=x, targets=y, heldout=heldout)
    receipt = fit.make_receipt(dataset, None, x, y, heldout, rank=2, max_normalized_rmse=0.1, min_fit_rows=32, min_heldout_rows=8, unsafe_small_probe=False)
    assert receipt["status"] == "REFUSED_INSUFFICIENT_TEACHER_EVIDENCE"
    probe = fit.make_receipt(dataset, None, x, y, heldout, rank=2, max_normalized_rmse=0.1, min_fit_rows=32, min_heldout_rows=8, unsafe_small_probe=True)
    assert probe["status"] == "UNSAFE_SMALL_PROBE_NOT_PROMOTABLE"
    assert probe["tps_claim"] is None


def test_rff_silu_scores_a_fixed_nonlinear_teacher_on_unseen_rows() -> None:
    rng = np.random.default_rng(19)
    x = rng.normal(size=(100, 5)).astype(np.float32)
    heldout = np.zeros(100, dtype=np.bool_); heldout[::5] = True
    basis = np.random.default_rng(17).standard_normal((5, 3), dtype=np.float32) / np.sqrt(np.float32(5))
    features = (x - x[~heldout].mean(axis=0)) @ basis; features = features / (1.0 + np.exp(-features, dtype=np.float32))
    y = features @ np.array([[1.0, -2.0], [0.5, 1.0], [2.0, 0.0]], dtype=np.float32)
    result = fit.fit_rff_silu(x, y, heldout, width=3)
    assert result["heldout_normalized_rmse"] < 1e-5
    assert result["physical"]["activation"] == "silu"
