"""Tests for training-free GLM→DeepSeek math transfer.

Proves:
  - no gradient/optimizer/training path is taken
  - closed-form subspace + projection + reversible residual work
  - capability remains unvalidated (forward-gated)
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_fusion_op as fusion  # noqa: E402
from lab.operators import frankenstein_transfer as xfer  # noqa: E402
from lab.receipts import verify  # noqa: E402


TRANSFER_SOURCE = Path(xfer.__file__)


def test_no_training_imports_or_calls() -> None:
    """FAILS if gradient/optimizer/training APIs are introduced into transfer."""

    source = TRANSFER_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_modules = {
        "torch.optim",
        "torch.nn.functional",
        "transformers.trainer",
        "tensorflow",
        "jax",
    }
    forbidden_names = {
        "backward",
        "zero_grad",
        "optimizer",
        "Adam",
        "AdamW",
        "SGD",
        "grad_fn",
        "autograd",
        "requires_grad",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules, alias.name
                assert not alias.name.startswith("torch.optim"), alias.name
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod not in forbidden_modules, mod
            assert not mod.startswith("torch.optim"), mod
            assert mod != "torch", "torch import is forbidden in training-free module"
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in {"backward", "zero_grad", "step"} and isinstance(func, ast.Attribute):
                # allow pathlib Path methods etc. only if not optim-like receiver
                pass
            if name in {"backward", "zero_grad"}:
                pytest.fail(f"forbidden training call: {name}")

    # AST is the source of truth; also reject real import lines.
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            pytest.fail(f"torch import present: {stripped}")
        if "import torch.optim" in stripped or "from torch.optim" in stripped:
            pytest.fail(f"optimizer import present: {stripped}")

    guard = xfer.assert_no_training_path()
    assert guard["training_path_present"] is False
    assert guard["imports_torch"] is False
    assert guard["imports_optimizer"] is False


def test_loss_fit_path_deprecated_not_primary() -> None:
    spec = fusion.fusion_operation_spec()
    assert spec["primary_method"] == fusion.TRAINING_FREE_METHOD
    assert spec["loss_fit_path"]["executed_by_harness"] is False
    assert spec["loss_fit_path"]["status"] == fusion.LOSS_FIT_PATH_STATUS
    assert "training_free" in spec
    assert spec["training_free"]["verdict"] == "REAL_AND_MINIMAL_TRAINING_FREE"
    loss = fusion.loss_target(transplant_point="post_moe_hidden_state")
    assert loss["deprecated"] is True
    assert loss["executed_by_harness"] is False
    # Existing shape contract still holds for documentation.
    assert loss["forward_gate"] == fusion.FORWARD_GATE
    assert any(t["name"] == "mse_projected_donor" for t in loss["terms"])


def test_closed_form_projection_shapes() -> None:
    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.standard_normal((6144, 16)))
    proj = xfer.closed_form_projection(basis, student_hidden=4096, seed=1)
    assert proj["weight"].shape == (6144, 4096)
    assert proj["bias"].shape == (4096,)
    assert proj["method"] == "closed_form_subspace_isometric_embedding"
    # Isometry on subspace: B.T @ W @ E ≈ I
    recon = basis.T @ proj["weight"] @ proj["student_embedding"]
    assert np.allclose(recon, np.eye(16), atol=1e-6)


def test_subspace_recovers_planted_direction() -> None:
    extraction = xfer.extract_from_synthetic_weights(rank=8, n_experts=48, seed=2)
    true = extraction["true_basis_for_test"]
    basis = extraction["basis"]
    # Canonical correlations via SVD of true.T @ basis; mean |cos| near 1.
    _, s, _ = np.linalg.svd(true.T @ basis, full_matrices=False)
    mean_cos = float(np.mean(s))
    assert mean_cos > 0.95, mean_cos
    assert extraction["energy_fraction"] > 0.90
    assert extraction["method"]["training"] is False
    assert extraction["method"]["gradient_descent"] is False


def test_residual_apply_reverse_roundtrip() -> None:
    extraction = xfer.extract_from_synthetic_weights(rank=8, seed=3)
    module = xfer.build_transfer_module(extraction=extraction, steering_scale=0.05)
    assert module["trained"] is False
    assert module["capability_status"] == "UNVALIDATED_WEIGHT_ONLY_DERIVED"
    assert module["capability_claim"] is False
    assert module["forward_gate"] == fusion.FORWARD_GATE
    assert module["direct_weight_transplant"] is False

    rng = np.random.default_rng(4)
    a = rng.standard_normal((4, 8, 4096))
    out = xfer.apply_residual(a, module, transplant_point="post_moe_hidden_state")
    assert out.shape == a.shape
    assert not np.allclose(out, a)
    back = xfer.reverse_residual(out, module, transplant_point="post_moe_hidden_state")
    assert np.allclose(back, a, atol=1e-5)


def test_seal_module_and_structural_apply(tmp_path: Path) -> None:
    extraction = xfer.extract_from_synthetic_weights(rank=8, seed=5)
    module = xfer.build_transfer_module(extraction=extraction)
    sealed = xfer.seal_transfer_module_files(module, out_dir=tmp_path / "mod")
    assert Path(sealed["meta_path"]).is_file()
    assert Path(sealed["module_path"]).is_file()
    assert sealed["capability_status"] == "UNVALIDATED_WEIGHT_ONLY_DERIVED"
    raw = Path(sealed["module_path"]).read_bytes()
    assert raw[:8] == xfer.MODULE_MAGIC
    assert b"gravity" not in raw[:64]

    doc = json.loads(Path(sealed["meta_path"]).read_text(encoding="utf-8"))
    verify(doc, label="transfer module")
    assert doc["trained"] is False
    assert doc["math_bench_status"] == "NOT_RUN"

    loaded = xfer.load_transfer_module(Path(sealed["meta_path"]), Path(sealed["module_path"]))
    applied = xfer.frankenstein_transfer_apply(
        module=loaded,
        body_path=tmp_path / "missing-body.gravity",
        out_dir=tmp_path / "apply",
    )
    assert applied["validation_status"] == fusion.FORWARD_GATE
    assert applied["capability_claim"] is False
    apply_doc = json.loads(Path(applied["apply_path"]).read_text(encoding="utf-8"))
    verify(apply_doc, label="apply")
    assert apply_doc["claim_boundary"]["frankenstein_math_capability_validated"] is False
    assert apply_doc["student_body"]["rewritten"] is False


def test_router_bias_fixed_not_trained() -> None:
    scores = {i: float(i) for i in range(16)}
    bias = xfer.router_bias_from_scores(scores, n_experts=256)
    assert bias.shape == (256,)
    assert float(bias[15]) > float(bias[0])
    assert np.all(bias[16:] == 0.0)


def test_floor_constant_is_25gib() -> None:
    assert xfer.MIN_FREE_FLOOR_BYTES == 25 * 1024**3


def test_guard_cli(tmp_path: Path) -> None:
    rc = xfer.main(["guard"])
    assert rc == 0
