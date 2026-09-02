"""Tests for the STATIC_ONLY host/shader ABI preflight.

A checker never seen to fail is not a checker. The negative control below
builds a synthetic kernel+host pair IN THIS FILE (not under crates/) with a
deliberately wrong buffer index and proves the refusal fires.
"""
from __future__ import annotations

import json

from tools.future import static_kernel_verify as skv
from tools.future._common import RECEIPTS, HardwareClaimError, write_receipt


# ---------------------------------------------------------------------------
# Synthetic fixtures (never written under crates/)
# ---------------------------------------------------------------------------

GOOD_METAL = """
#include <metal_stdlib>
using namespace metal;

struct ArgbufN { uint n; float eps; };

kernel void demo_k(
    device const float* x [[buffer(0)]],
    device float* out     [[buffer(1)]],
    constant uint& n      [[buffer(2)]],
    constant float& eps   [[buffer(3)]],
    threadgroup float* sh [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]]
) {}
"""

GOOD_HOST = """
fn go(ctx: &MetalContext, x: &Buffer, out: &Buffer) {
    const TG: u32 = 64;
    ctx.dispatch_threads("demo_k", (64, 1, 1), (TG, 1, 1), |enc| {
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_u32(2, n);
        enc.set_f32(3, eps);
        enc.set_threadgroup_memory_length(0, 256);
    });
}
"""

OFF_BY_ONE_HOST = """
fn go(ctx: &MetalContext, x: &Buffer, out: &Buffer) {
    ctx.dispatch_threads("demo_k", (64, 1, 1), (64, 1, 1), |enc| {
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_u32(3, n);  // WRONG: shader constant uint is buffer(2)
        enc.set_f32(4, eps);
        enc.set_threadgroup_memory_length(0, 256);
    });
}
"""

MISSING_KERNEL_HOST = """
fn go(ctx: &MetalContext) {
    ctx.dispatch_threads("no_such_kernel", (1, 1, 1), (1, 1, 1), |enc| {
        enc.set_buffer(0, Some(x), 0);
    });
}
"""

GENERATED_QWEN_METAL = r"""
#include <metal_stdlib>
using namespace metal;
#define QWEN_UNIFORM_Q4_MATMUL_K(KVAL) \
kernel void qwen_uniform_q4_group64_matmul_k##KVAL##_geo_tpr64_tg128() {}
QWEN_UNIFORM_Q4_MATMUL_K(1)
"""

GENERATED_QWEN_HOST = """
fn go(ctx: &MetalContext) {
    ctx.dispatch_threads(
        "qwen_uniform_q4_group64_matmul_k1_geo_tpr64_tg128",
        (128, 1, 1),
        (128, 1, 1),
        |_enc| {},
    );
}
"""

TYPE_MISMATCH_HOST = """
fn go(ctx: &MetalContext, x: &Buffer, out: &Buffer) {
    ctx.dispatch_threads("demo_k", (64, 1, 1), (64, 1, 1), |enc| {
        enc.set_u32(0, n);  // WRONG kind: shader buffer(0) is device float*
        enc.set_buffer(1, Some(out), 0);
        enc.set_u32(2, n);
        enc.set_f32(3, eps);
        enc.set_threadgroup_memory_length(0, 256);
    });
}
"""

DYNAMIC_INDEX_HOST = """
fn go(ctx: &MetalContext, x: &Buffer, slot: u32) {
    ctx.dispatch_threads("demo_k", (64, 1, 1), (64, 1, 1), |enc| {
        enc.set_buffer(slot, Some(x), 0);  // cannot follow
        enc.set_buffer(1, Some(out), 0);
        enc.set_u32(2, n);
        enc.set_f32(3, eps);
    });
}
"""

OVERSIZE_TG_HOST = """
fn go(ctx: &MetalContext, x: &Buffer, out: &Buffer) {
    ctx.dispatch_threads("demo_k", (2048, 1, 1), (2048, 1, 1), |enc| {
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_u32(2, n);
        enc.set_f32(3, eps);
        enc.set_threadgroup_memory_length(0, 256);
    });
}
"""

ABI_METAL = """
#include <metal_stdlib>
using namespace metal;
struct Pack {
    uint a;
    float b;
    uint c;
};
kernel void pack_k(constant Pack& args [[buffer(0)]]) {}
"""

ABI_RUST_OK = """
#[repr(C)]
pub struct Pack {
    pub a: u32,
    pub b: f32,
    pub c: u32,
}
fn go(ctx: &MetalContext) {
    let mut ab = KernelArgBuffer::new(ctx, &[ArgLayout::U32, ArgLayout::F32, ArgLayout::U32])?;
    ctx.dispatch_threads("pack_k", (1, 1, 1), (1, 1, 1), |enc| {
        enc.set_buffer(0, Some(ab.handle()), 0);
    });
}
"""

ABI_RUST_SWAPPED = """
#[repr(C)]
pub struct Pack {
    pub b: f32,
    pub a: u32,
    pub c: u32,
}
fn go(ctx: &MetalContext) {
    ctx.dispatch_threads("pack_k", (1, 1, 1), (1, 1, 1), |enc| {
        enc.set_buffer(0, Some(ab.handle()), 0);
    });
}
"""

FEATURE_METAL = """
#include <metal_stdlib>
using namespace metal;
kernel void strand_secret(device float* out [[buffer(0)]]) {}
"""

FEATURE_HOST_GATED = """
#[cfg(feature = "tq")]
fn decode_strand_bitslice(ctx: &MetalContext) {
    ctx.dispatch_threads("strand_secret", (1, 1, 1), (1, 1, 1), |enc| {
        enc.set_buffer(0, Some(&out), 0);
    });
}
"""

FEATURE_HOST_UNGATED = """
fn always(ctx: &MetalContext) {
    ctx.dispatch_threads("strand_secret", (1, 1, 1), (1, 1, 1), |enc| {
        enc.set_buffer(0, Some(&out), 0);
    });
}
"""


def _find(raw, severity, check, kernel=None):
    hits = [
        f
        for f in raw["findings"]
        if f.severity == severity and f.check == check and (kernel is None or f.kernel == kernel)
    ]
    return hits


STAGE_SET_HOST = """
fn go(ctx: &MetalContext, x: &Buffer, out: &Buffer) {
    ctx.dispatch_threads("demo_k", (64, 1, 1), (64, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(out), 0);
        encoder.stage_set_u32(2, n);
        encoder.stage_set_f32(3, eps);
        encoder.set_threadgroup_memory_length(0, 256);
    });
}
"""

FREE_SET_HOST = """
fn go(ctx: &MetalContext, x: &Buffer, out: &Buffer) {
    ctx.dispatch_threads("demo_k", (64, 1, 1), (64, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(out), 0);
        set_u32(encoder, 2, n);
        set_f32(encoder, 3, &eps);
        encoder.set_threadgroup_memory_length(0, 256);
    });
}
"""


def test_good_pair_matching_indices_is_not_an_error():
    raw = skv.analyze(
        {"synth.metal": GOOD_METAL},
        {"synth.rs": GOOD_HOST},
    )
    bind_err = _find(raw, "ERROR", "binding_index")
    assert not bind_err, bind_err
    exist_err = _find(raw, "ERROR", "kernel_existence")
    assert not exist_err
    assert raw["binding_checked"] >= 1


def test_negative_control_wrong_buffer_index_is_refused():
    """The guard nobody has watched fail is not a guard.

    Host binds buffer(3)/buffer(4) while the shader declares 0,1,2,3 — a
    classic off-by-one on the scalar tail. The checker MUST flag ERROR
    binding_index and MUST set extra.off_by_one.
    """
    raw = skv.analyze(
        {"synth.metal": GOOD_METAL},
        {"synth.rs": OFF_BY_ONE_HOST},
    )
    hits = _find(raw, "ERROR", "binding_index", kernel="demo_k")
    assert hits, (
        "NEGATIVE CONTROL FAILED: deliberately wrong buffer index was not flagged. "
        f"findings={[(f.severity, f.check, f.message) for f in raw['findings'][:20]]}"
    )
    assert hits[0].host and "synth.rs:" in hits[0].host
    assert hits[0].shader and "synth.metal:" in hits[0].shader
    assert hits[0].extra.get("off_by_one") is True or (
        3 in hits[0].extra.get("extra_on_host", [])
        or 2 in hits[0].extra.get("missing_on_host", [])
    )


def test_missing_kernel_name_is_error():
    raw = skv.analyze(
        {"synth.metal": GOOD_METAL},
        {"synth.rs": MISSING_KERNEL_HOST},
    )
    hits = _find(raw, "ERROR", "kernel_existence", kernel="no_such_kernel")
    assert hits, raw["counts"]


def test_token_pasted_qwen_kernel_name_is_recovered_without_fake_abi():
    raw = skv.analyze(
        {"qwen_uniform_q4.metal": GENERATED_QWEN_METAL},
        {"synth.rs": GENERATED_QWEN_HOST},
    )
    name = "qwen_uniform_q4_group64_matmul_k1_geo_tpr64_tg128"
    assert not _find(raw, "ERROR", "kernel_existence", kernel=name)
    assert raw["generated_kernel_names"][name]["macro"] == "QWEN_UNIFORM_Q4_MATMUL_K"


QWEN_SET_HOST = """
fn go(tcb: &mut TokenCommandBuffer, x: &Buffer, out: &Buffer) {
    tcb.dispatch_threads("demo_k", (64, 1, 1), (64, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(out), 0);
        encoder.qwen_set_u32(2, n);
        encoder.qwen_set_f32(3, eps);
        encoder.set_threadgroup_memory_length(0, 256);
    });
}
"""


def test_stage_set_u32_is_a_real_host_bind():
    raw = skv.analyze({"synth.metal": GOOD_METAL}, {"synth.rs": STAGE_SET_HOST})
    assert not _find(raw, "ERROR", "binding_index"), [
        f.message for f in raw["findings"] if f.severity == "ERROR"
    ]
    assert raw["binding_checked"] >= 1


def test_qwen_set_u32_is_a_real_host_bind():
    raw = skv.analyze({"synth.metal": GOOD_METAL}, {"synth.rs": QWEN_SET_HOST})
    assert not _find(raw, "ERROR", "binding_index"), [
        f.message for f in raw["findings"] if f.severity == "ERROR"
    ]
    assert raw["binding_checked"] >= 1


def test_free_function_set_u32_is_a_real_host_bind():
    raw = skv.analyze({"synth.metal": GOOD_METAL}, {"synth.rs": FREE_SET_HOST})
    assert not _find(raw, "ERROR", "binding_index"), [
        f.message for f in raw["findings"] if f.severity == "ERROR"
    ]
    assert raw["binding_checked"] >= 1


def test_type_width_mismatch_is_error():
    raw = skv.analyze(
        {"synth.metal": GOOD_METAL},
        {"synth.rs": TYPE_MISMATCH_HOST},
    )
    hits = _find(raw, "ERROR", "type_width", kernel="demo_k")
    assert hits, [(f.severity, f.check, f.message) for f in raw["findings"] if f.severity == "ERROR"]
    assert "buffer(0)" in hits[0].message


def test_dynamic_buffer_index_is_unverifiable_never_pass():
    raw = skv.analyze(
        {"synth.metal": GOOD_METAL},
        {"synth.rs": DYNAMIC_INDEX_HOST},
    )
    hits = _find(raw, "UNVERIFIABLE", "binding_index")
    assert hits, "dynamic index must be UNVERIFIABLE, not silent"
    # Must not be reported as a successful matching bind
    assert raw["binding_checked"] == 0


def test_threadgroup_over_device_limit_is_error():
    raw = skv.analyze(
        {"synth.metal": GOOD_METAL},
        {"synth.rs": OVERSIZE_TG_HOST},
    )
    hits = _find(raw, "ERROR", "dispatch_geometry", kernel="demo_k")
    assert hits
    assert "1024" in hits[0].message


def test_repr_c_matches_metal_struct():
    raw = skv.analyze(
        {"synth.metal": ABI_METAL},
        {"synth.rs": ABI_RUST_OK},
    )
    err = _find(raw, "ERROR", "host_shader_abi")
    assert not err, err
    info = [
        f
        for f in raw["findings"]
        if f.check == "host_shader_abi" and f.severity == "INFO" and f.kernel is None
    ]
    assert info, "matching Pack structs should produce an ABI INFO"


def test_repr_c_field_order_mismatch_is_error():
    raw = skv.analyze(
        {"synth.metal": ABI_METAL},
        {"synth.rs": ABI_RUST_SWAPPED},
    )
    hits = _find(raw, "ERROR", "host_shader_abi")
    assert hits, [(f.severity, f.check, f.message) for f in raw["findings"]]
    assert "Pack" in hits[0].message


def test_feature_gate_unreachable_when_off():
    raw = skv.analyze(
        {"crates/hawking-core/shaders/strand_bitslice.metal": FEATURE_METAL},
        {"crates/hawking-core/src/kernels/mod.rs": FEATURE_HOST_GATED},
    )
    hits = [
        f
        for f in raw["findings"]
        if f.check == "feature_gate" and f.kernel == "strand_secret" and f.severity == "INFO"
    ]
    assert hits
    assert "genuinely_unreachable_when_off=True" in hits[0].message


def test_feature_gate_still_reachable_when_ungated_is_error():
    raw = skv.analyze(
        {"crates/hawking-core/shaders/strand_bitslice.metal": FEATURE_METAL},
        {"crates/hawking-core/src/kernels/mod.rs": FEATURE_HOST_UNGATED},
    )
    hits = _find(raw, "ERROR", "feature_gate", kernel="strand_secret")
    assert hits, [(f.severity, f.check, f.message) for f in raw["findings"] if f.check == "feature_gate"]


def test_control_path_production_host_against_library_membership():
    metal = {"crates/hawking-core/shaders/orphan.metal": GOOD_METAL}
    rust = {"crates/hawking-core/src/kernels/mod.rs": GOOD_HOST}
    raw = skv.analyze(
        metal,
        rust,
        library_membership={},  # empty: nothing is in the MetalContext library
    )
    hits = _find(raw, "ERROR", "control_path", kernel="demo_k")
    assert hits
    assert "all_shader_sources" in hits[0].message


def test_plumbing_dispatch_threads_definitions_are_skipped():
    plumbing = """
        pub fn dispatch_threads(
            &self,
            fn_name: &str,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<()> {
            let pipe = self.pipeline(fn_name)?;
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, 1, 1),
                MTLSize::new(tg.0 as u64, 1, 1),
            );
            encode(enc);
            Ok(())
        }
    """
    raw = skv.analyze({"s.metal": GOOD_METAL}, {"metal/mod.rs": plumbing})
    assert raw["dispatches"] == []


def test_decode_family_pick_both_names_must_exist():
    family = '''
pub const MATVEC_BINARY: &str = "gk_matvec_binary";
pub const LEGACY_MATVEC_BINARY: &str = "q80_binary_group_matvec";
fn pick<'a>(family: &'a str, legacy: &'a str) -> &'a str { family }
pub fn matvec_binary() -> &'static str {
    pick(MATVEC_BINARY, LEGACY_MATVEC_BINARY)
}
'''
    metal = """
kernel void gk_matvec_binary(device float* x [[buffer(0)]]) {}
"""
    host = """
fn go(tcb: &mut TokenCommandBuffer) {
    tcb.dispatch_threads(crate::decode_family::matvec_binary(), (1,1,1), (1,1,1), |enc| {
        enc.set_buffer(0, Some(x), 0);
    });
}
"""
    raw = skv.analyze(
        {"gk_family.metal": metal},
        {
            "crates/hawking-core/src/decode_family.rs": family,
            "host.rs": host,
        },
    )
    missing = _find(raw, "ERROR", "kernel_existence", kernel="q80_binary_group_matvec")
    assert missing, "legacy pick name must be required to exist"
    present = _find(raw, "ERROR", "kernel_existence", kernel="gk_matvec_binary")
    assert not present


def test_analyze_does_not_put_hardware_numbers_in_findings():
    raw = skv.analyze({"s.metal": GOOD_METAL}, {"h.rs": GOOD_HOST})
    doc = skv.report_from_analyze(raw)
    # write_receipt must accept this document
    # (uses a throwaway name then we leave it; build() overwrites the real one)
    try:
        write_receipt("_SKV_TEST_THROWNAWAY.json", dict(doc), "test_static_kernel_verify.py")
    except HardwareClaimError as e:
        raise AssertionError(f"preflight document tripped HardwareClaimError: {e}") from e
    p = RECEIPTS / "_SKV_TEST_THROWNAWAY.json"
    if p.exists():
        p.unlink()


def test_build_emits_sealed_receipt_with_required_fields():
    out = skv.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "STATIC_KERNEL_PREFLIGHT.json"
    assert doc["schema"] == "hawking.future.static_kernel_verify.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["does_not_substitute_for_protected_measurement"] is True
    assert "does NOT prove speed" in doc["static_correctness_does_not_prove_speed"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert "recovered_implementation" in doc
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["coverage"]["metal_kernels"] >= 1
    assert "binding count" in " ".join(doc["gaps_closed"]).lower() or any(
        "binding" in g for g in doc["gaps_closed"]
    )
    # Honesty: UNVERIFIABLE is first-class, never silently PASS
    assert "UNVERIFIABLE" in doc["coverage"]["honesty"]
    assert doc["counts"]["ERROR"] >= 0


def test_rust_binary_missing_path_is_none(monkeypatch, tmp_path):
    """A missing artifact must not crash HCLI: scan() falls back to Python."""
    monkeypatch.delenv("HAWKING_SKV_FORCE_PYTHON", raising=False)
    monkeypatch.setenv("HAWKING_STATIC_KERNEL_VERIFY", str(tmp_path / "no-such-bin"))
    assert skv._rust_binary() is None
    assert skv._scan_via_rust(skv.REPO) is None


def test_force_python_disables_rust(monkeypatch):
    monkeypatch.setenv("HAWKING_SKV_FORCE_PYTHON", "1")
    assert skv._rust_binary() is None
