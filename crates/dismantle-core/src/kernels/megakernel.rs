//! Megakernel POC dispatcher (2026-05-25, build/megakernel).
//!
//! SKELETON. Private (not exported). See
//! `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/build_megakernel_design_2026_05_25.md`
//! for the design memo and TODO checklist.
//!
//! This file ships the function signature + argbuf type and the
//! pipeline lookup site. Stage encoding, weight pre-dequant, and KV
//! integration are stubbed. The full implementation is followup.

#![allow(dead_code)]

#[cfg(target_os = "macos")]
pub(crate) mod inner {
    use crate::metal::MetalContext;
    use crate::Result;

    /// Megakernel argbuf — must match `struct MkArgs` in
    /// `shaders/megakernel_qwen3b.metal`.
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct MkArgs {
        pub pos: u32,
        pub seq_len: u32,
        pub max_seq: u32,
    }

    /// Per-layer scalar bundle — matches `struct MkLayerWeights`.
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct MkLayerWeights {
        pub rms_eps: f32,
        pub rope_theta: f32,
        pub has_qbias: u32,
        pub has_kbias: u32,
        pub has_vbias: u32,
    }

    /// Dispatch the 2-layer Qwen-3B megakernel POC.
    ///
    /// **NOT IMPLEMENTED.** Returns `Err(...)` so the parity test fails
    /// loudly until the stage encoder + weight binding is wired. See
    /// memo § "What attended work unblocks" for the next steps.
    pub fn megakernel_2layer_dispatch(_ctx: &MetalContext) -> Result<()> {
        Err(crate::Error::Metal(
            "megakernel_2layer_dispatch: skeleton only — see build_megakernel_design_2026_05_25.md"
                .into(),
        ))
    }
}

#[cfg(target_os = "macos")]
#[allow(unused_imports)]
pub(crate) use inner::megakernel_2layer_dispatch;
