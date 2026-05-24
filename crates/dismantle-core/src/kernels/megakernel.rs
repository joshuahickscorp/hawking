//! Megakernel POC dispatcher (2026-05-25, build/megakernel — day 3+).
//!
//! SKELETON. Private (not exported). See
//! `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/build_megakernel_design_2026_05_25.md`
//! and `build_megakernel_day2_2026_05_25.md` for the design memo and
//! TODO checklist.
//!
//! This file ships:
//!  * argbuf layout (`MkArgs`, `MkLayerArgs`) matching the binding
//!    scheme in `shaders/megakernel_qwen3b.metal` (8 buffer slots
//!    total, within Metal's 30-slot binding limit);
//!  * the dispatcher function signature.
//!
//! Stage encoding, weight upload, and KV integration are stubbed.
//! Day-3+ fills the kernel body (stages A..L per layer) and the
//! dispatcher's Metal buffer assembly. See § "Day-3 entry points"
//! in `build_megakernel_day2_2026_05_25.md`.

#![allow(dead_code)]

#[cfg(target_os = "macos")]
pub(crate) mod inner {
    use crate::metal::MetalContext;
    use crate::Result;

    /// Megakernel scalar argbuf — must match `struct MkArgs` in
    /// `shaders/megakernel_qwen3b.metal`.
    #[repr(C)]
    #[derive(Copy, Clone, Default)]
    pub struct MkArgs {
        pub pos: u32,
        pub seq_len: u32,
        pub max_seq: u32,
        pub _padding: u32,
    }

    /// Per-layer argument buffer — must match `struct MkLayerArgs` in
    /// `shaders/megakernel_qwen3b.metal` byte-for-byte.
    ///
    /// Pointer fields hold Metal 3 GPU virtual addresses, obtained
    /// host-side via [`metal::Buffer::gpu_address`]. The dispatcher
    /// MUST also call `use_resource(buf, MTLResourceUsage::Read)` on
    /// every referenced weight buffer before encoding the compute
    /// dispatch, otherwise the driver will fault on first
    /// dereference.
    ///
    /// `qb` / `kb` / `vb` may be 0 (null) when the corresponding
    /// `has_*bias` flag is 0 (Qwen2 always has Q/K/V biases, but
    /// other models may not).
    ///
    /// Total size: 12 × 8 (pointers) + 2 × 4 (f32) + 4 × 4 (u32)
    ///           = 96 + 8 + 16 = 120 bytes.
    #[repr(C)]
    #[derive(Copy, Clone, Default)]
    pub struct MkLayerArgs {
        // f16 weight pointers (rows × cols, row-major):
        pub qw: u64, // q_proj
        pub kw: u64, // k_proj
        pub vw: u64, // v_proj
        pub ow: u64, // o_proj
        pub gw: u64, // ffn_gate
        pub uw: u64, // ffn_up
        pub dw: u64, // ffn_down
        // f32 norm + bias pointers:
        pub attn_norm: u64,
        pub ffn_norm: u64,
        pub qb: u64,
        pub kb: u64,
        pub vb: u64,
        // Scalars:
        pub rms_eps: f32,
        pub rope_theta: f32,
        pub has_qbias: u32,
        pub has_kbias: u32,
        pub has_vbias: u32,
        pub _padding: u32,
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

    /// Compile-time check that `MkLayerArgs` matches the 120-byte
    /// layout the shader expects. If this fires, the shader struct
    /// has drifted from the Rust struct (or vice-versa).
    const _MK_LAYER_ARGS_SIZE_CHECK: [(); 120] =
        [(); std::mem::size_of::<MkLayerArgs>()];
    const _MK_ARGS_SIZE_CHECK: [(); 16] = [(); std::mem::size_of::<MkArgs>()];
}

#[cfg(target_os = "macos")]
#[allow(unused_imports)]
pub(crate) use inner::megakernel_2layer_dispatch;
