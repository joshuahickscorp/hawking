"""G091: no impossible runtime path enters Odyssey.

The static preflight reports two kernel_existence ERRORs: the host names
qwen_q2f_group64_matvec_qkv_geo_tpr64_tg128 and ..._pair_geo_tpr64_tg128 and no
shader defines either. q80_mixed_decode.metal defines the sibling family and
stops one short of both.

CLASSIFICATION (S025 §38): not dead, and not taken by sealed-3.14. The gate is
`bits == 2`, and sealed-3.14's GQA q/k/v are codec 3 - uniform q4 - so the q4
fusion wins first and these branches never run. That is why it has been latent
rather than a crash. But this campaign is actively pursuing 2-bit bodies, and the
first artifact with 2-bit attention selects a pipeline that cannot be built.
"""
from __future__ import annotations

from pathlib import Path

from tools.future import _common as common

DECODE = "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
SHADER = "crates/hawking-core/shaders/q80_mixed_decode.metal"
QKV = "qwen_q2f_group64_matvec_qkv_geo_tpr64_tg128"
PAIR = "qwen_q2f_group64_matvec_pair_geo_tpr64_tg128"


def _src(rel: str) -> str:
    return (common.REPO / rel).read_text(encoding="utf-8")


def test_the_two_kernels_really_are_undefined():
    """If someone writes them, this test is the reminder to remove the guard."""
    shader = _src(SHADER)
    assert f"kernel void {QKV}(" not in shader
    assert f"kernel void {PAIR}(" not in shader
    # and the sibling family that stops one short really is there
    assert "kernel void qwen_q2f_group64_matvec_geo_tpr64_tg128(" in shader


def test_both_fused_paths_check_the_kernel_exists_before_selecting():
    src = _src(DECODE)
    for gate, kernel in (("can_fuse_q2f_qkv", "QWEN38_Q2F_QKV_GEO_KERNEL"),
                         ("can_fuse_q2f_pair", "QWEN38_Q2F_PAIR_GEO_KERNEL")):
        i = src.index(f"fn {gate}(")
        head = src[i:i + 260]
        assert "q2f_kernel_available" in head, f"{gate} does not check its kernel"
        assert kernel in head, f"{gate} checks the wrong kernel"
        assert "return false;" in head


def test_the_availability_check_asks_the_pipeline_not_a_list():
    """A hardcoded allowlist would go stale the moment a kernel is written."""
    src = _src(DECODE)
    i = src.index("fn q2f_kernel_available(")
    body = src[i:i + 200]
    assert ".pipeline(name).is_ok()" in body


def test_the_fallback_is_the_unfused_path_not_a_refusal():
    """Falling back must still compute the token, just without the fusion."""
    src = _src(DECODE)
    i = src.index("} else if self.fuse_gqa_qkv && self.can_fuse_q2f_qkv(layer) {")
    after = src[i:i + 400]
    assert "} else {" in after
    assert "encode_named_matvec" in after


def test_the_reason_is_recorded_where_the_guard_is():
    src = _src(DECODE)
    i = src.index("fn q2f_kernel_available(")
    why = src[max(0, i - 1400):i]
    assert "no shader defines either" in why
    assert "codec 3" in why or "uniform q4" in why
    assert "2-bit attention" in why
