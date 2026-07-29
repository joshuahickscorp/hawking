//! Thin dispatch helpers that are byte-identical across dense model engines.
//!
//! Forward passes stay per-architecture. These cover only the identical Metal/
//! CPU branch choices for rmsnorm and the plain f16 GEMV (no pinned LM-head
//! buffer). DeepSeek's LM-head path keeps its own `gemv_f16_dispatch` because
//! it prefers a pinned buffer when present.

use crate::kernels::{gemv_f16, rmsnorm};
use crate::metal::MetalContext;
use crate::Result;
use half::f16;

/// RMSNorm: Metal when a context is live, otherwise the CPU reference.
pub(crate) fn rmsnorm_dispatch(
    metal_ctx: Option<&MetalContext>,
    x: &[f32],
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
) -> Result<()> {
    #[cfg(target_os = "macos")]
    if let Some(ctx) = metal_ctx {
        return crate::kernels::rmsnorm_metal(ctx, x, weight, eps, out);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = metal_ctx;
    rmsnorm(x, weight, eps, out);
    Ok(())
}

/// Plain f16 GEMV: Metal when a context is live, otherwise the CPU reference.
/// Does not handle a pinned LM-head buffer (DeepSeek keeps that path local).
pub(crate) fn gemv_f16_dispatch(
    metal_ctx: Option<&MetalContext>,
    w_f16: &[f16],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    #[cfg(target_os = "macos")]
    if let Some(ctx) = metal_ctx {
        let w_bytes = bytemuck::cast_slice::<f16, u8>(w_f16);
        return crate::kernels::gemv_f16_metal(ctx, w_bytes, rows, cols, x, out);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = metal_ctx;
    gemv_f16(w_f16, rows, cols, x, out);
    Ok(())
}
