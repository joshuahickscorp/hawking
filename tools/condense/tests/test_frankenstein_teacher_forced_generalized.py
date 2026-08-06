#!/usr/bin/env python3.12
"""Regression: generalized executor preserves GLM synthetic capture hashes.

Proves the pure architecture-config refactor did not change GLM behavior:
layer array SHA-256s and carry hashes match between the GLM thin wrapper and
the frankenstein core bound to GLM52_ARCHITECTURE. Also pins Kimi architecture
facts as config-only (no weights).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lab.operators import frankenstein_teacher_forced_executor as frank  # noqa: E402
from lab.operators import glm52_synthetic as synthetic  # noqa: E402
from lab.operators import glm52_teacher_forced_executor as glm  # noqa: E402
from lab.operators.glm52_common import verify_sealed  # noqa: E402


@pytest.fixture()
def fixture_env(tmp_path):
    root = tmp_path / "fixture"
    fx = synthetic.build_synthetic_fixture(root)
    return {"fixture": fx, "source": fx.full_dir, "tmp": tmp_path}


def _run_via(module, source, out, **kwargs):
    cfg = module.ExecutorConfig(
        mode="synthetic",
        corpus_level=kwargs.get("level", "L0"),
        source_root=source,
        output_dir=out,
        max_sequence=kwargs.get("max_sequence", 2),
        microbatch=kwargs.get("microbatch", 8),
        sample_hidden=kwargs.get("sample_hidden", 16),
        profile="synthetic",
        allow_eviction=False,  # deterministic residency for dual-run compare
        require_floor=False,
        max_layers=kwargs.get("max_layers", None),
        architecture=module.GLM52_ARCHITECTURE,
    )
    return module.run_teacher_forced(cfg)


def test_glm_architecture_binding_is_config_not_magic():
    arch = frank.GLM52_ARCHITECTURE
    assert arch.family == "glm52"
    assert arch.hidden_size == 6144
    assert arch.num_hidden_layers == 78
    assert arch.attention_variant == "mla_dsa_indexshare"
    assert arch.moe_variant == "noaux_tc_swiglu"
    assert arch.n_routed_experts == 256
    assert arch.num_experts_per_tok == 8
    assert arch.backend == "glm52"
    assert arch.schema_receipt == frank.SCHEMA_RECEIPT
    # Wrapper exposes the same object identity for GLM.
    assert glm.GLM52_ARCHITECTURE is frank.GLM52_ARCHITECTURE
    assert glm.ARCHITECTURE is frank.GLM52_ARCHITECTURE


def test_kimi_architecture_facts_pinned_no_backend():
    arch = frank.KIMI_K3_ARCHITECTURE
    assert arch.repo_id == "moonshotai/Kimi-K3"
    assert arch.immutable_revision == "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
    assert arch.hidden_size == 7168
    assert arch.num_hidden_layers == 93
    assert arch.n_routed_experts == 896
    assert arch.num_experts_per_tok == 16
    assert arch.n_shared_experts == 2
    assert arch.weight_shard_count == 96
    assert arch.weight_shard_bytes == 1_560_936_091_448
    assert arch.tokenizer_kind == "tiktoken"
    assert arch.backend == "kimi_k3"
    # Must not silently run as GLM.
    cfg = frank.ExecutorConfig(
        mode="synthetic",
        corpus_level="L0",
        source_root=Path("/tmp"),
        output_dir=Path("/tmp"),
        architecture=arch,
        require_floor=False,
    )
    with pytest.raises(frank.TeacherForcedError, match="not implemented"):
        frank.run_teacher_forced(cfg)


def test_glm_wrapper_and_core_reproduce_identical_layer_hashes(fixture_env):
    out_glm = fixture_env["tmp"] / "out_glm"
    out_core = fixture_env["tmp"] / "out_core"
    r_glm = _run_via(glm, fixture_env["source"], out_glm)
    r_core = _run_via(frank, fixture_env["source"], out_core)

    assert r_glm["status"].startswith("PASS")
    assert r_core["status"].startswith("PASS")
    assert r_glm["layers_captured"] == r_core["layers_captured"]
    assert r_glm["architecture"] == r_core["architecture"]
    assert r_glm["corpus"]["membership_sha256"] == r_core["corpus"]["membership_sha256"]

    # Per-layer array SHA-256s must match byte-for-byte.
    n_layers = r_glm["architecture"]["num_hidden_layers_config"]
    for layer in range(n_layers):
        a = json.loads((out_glm / "layers" / f"L{layer:02d}.json").read_text())
        b = json.loads((out_core / "layers" / f"L{layer:02d}.json").read_text())
        verify_sealed(a, label=f"glm L{layer:02d}")
        verify_sealed(b, label=f"core L{layer:02d}")
        assert a["array_sha256"] == b["array_sha256"]
        assert a["npz_sha256"] == b["npz_sha256"]

    # Carry after L0 identical.
    c_glm = verify_sealed(
        json.loads((out_glm / "carry" / "after_L00.json").read_text()), label="carry glm"
    )
    c_core = verify_sealed(
        json.loads((out_core / "carry" / "after_L00.json").read_text()), label="carry core"
    )
    assert c_glm["hidden_sha256"] == c_core["hidden_sha256"]
    assert c_glm["topk_sha256"] == c_core["topk_sha256"]
    with np.load(out_glm / "carry" / "after_L00.npz") as z1, np.load(
        out_core / "carry" / "after_L00.npz"
    ) as z2:
        assert np.array_equal(z1["carry_hidden"], z2["carry_hidden"])


def test_glm_synthetic_l0_is_deterministic_under_architecture_config(fixture_env):
    """Two sequential runs via the generalized core yield identical L00 hashes."""
    out1 = fixture_env["tmp"] / "det1"
    out2 = fixture_env["tmp"] / "det2"
    r1 = _run_via(frank, fixture_env["source"], out1)
    r2 = _run_via(frank, fixture_env["source"], out2)
    assert r1["layers_captured"] == r2["layers_captured"]
    a = json.loads((out1 / "layers" / "L00.json").read_text())
    b = json.loads((out2 / "layers" / "L00.json").read_text())
    assert a["array_sha256"] == b["array_sha256"]
    assert a["npz_sha256"] == b["npz_sha256"]
