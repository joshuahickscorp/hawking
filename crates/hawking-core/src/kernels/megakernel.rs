//! Megakernel POC dispatcher (2026-05-25, build/megakernel — day 3+).
//!
//! Day-3 wires the upload helper + dispatch harness. The shader body
//! is still a pass-through (stages A..L TODO), so this dispatcher's
//! correctness gate is the pass-through invariant `x_out == x_in`,
//! NOT a full per-layer parity vs CPU. Stage bodies land in the
//! follow-up sessions per
//! project design memory (build_megakernel_day3_2026_05_25).

#![allow(dead_code)]

#[cfg(target_os = "macos")]
pub mod inner {
    use crate::metal::MetalContext;
    use crate::model::qwen_dense::MegakernelLayerWeightsF16;
    use crate::{Error, Result};
    use half::f16;
    use metal::{Buffer, MTLResourceUsage};

    // ── Argbuf layouts (must mirror shaders/megakernel_qwen3b.metal) ────────

    /// Megakernel scalar argbuf — must match `struct MkArgs` in
    /// `shaders/megakernel_qwen3b.metal`.
    ///
    /// `probe_stage` selects which intermediate buffer the shader copies
    /// into `x_out` at the end of the dispatch (dev-only escape hatch so
    /// stages B..L can be parity-tested incrementally without rewiring
    /// the terminal write each commit). Values: see `MK_PROBE_*` consts.
    #[repr(C)]
    #[derive(Copy, Clone, Default)]
    pub struct MkArgs {
        pub pos: u32,
        pub seq_len: u32,
        pub max_seq: u32,
        pub probe_stage: u32,
        /// Number of transformer layers the N-layer kernel runs in one
        /// dispatch. Ignored by `qwen3b_megakernel_2layer` (appended
        /// after the 4 fields that kernel reads).
        pub n_layers: u32,
    }

    /// Probe-stage IDs (must match `MK_PROBE_*` in shader). Each picks
    /// which intermediate the shader emits into `x_out`.
    pub const MK_PROBE_XNORM_A: u32 = 0; // layer-0 stage A (pre-attn rmsnorm)
    pub const MK_PROBE_Q_ROT: u32 = 1; // layer-0 stage D (post-RoPE Q, 2048)
    pub const MK_PROBE_ATTN_OUT: u32 = 2; // layer-0 stage F (MHA out, 2048)
    pub const MK_PROBE_O_PROJ: u32 = 3; // layer-0 stage G (o_proj out, 2048)
    pub const MK_PROBE_XNORM_FFN: u32 = 4; // layer-0 stage H (post-attn rmsnorm)
    pub const MK_PROBE_FFN_DOWN: u32 = 5; // layer-0 stage K (ffn_down out, 2048)
    pub const MK_PROBE_RESIDUAL_L0: u32 = 6; // post-layer-0 residual
    pub const MK_PROBE_RESIDUAL: u32 = 7; // post-layer-1 residual (final 2-layer)

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

    // Compile-time check that `MkLayerArgs`/`MkArgs` match the byte
    // layout the shader expects. If either fires, the shader struct
    // has drifted from the Rust struct (or vice-versa).
    const _MK_LAYER_ARGS_SIZE_CHECK: [(); 120] = [(); std::mem::size_of::<MkLayerArgs>()];
    const _MK_ARGS_SIZE_CHECK: [(); 20] = [(); std::mem::size_of::<MkArgs>()];

    // ── Qwen-3B megakernel shape constants (mirror shader header) ───────────
    pub const MK_HIDDEN: usize = 2048;
    pub const MK_Q_DIM: usize = 2048;
    pub const MK_KV_DIM: usize = 256;
    pub const MK_INTERMEDIATE: usize = 11008;
    /// 8960 halfs = 17920 bytes shmem (see shader).
    const MK_SHMEM_HALFS: usize = 8960;
    const MK_RMS_EPS: f32 = 1e-6;
    const MK_ROPE_THETA: f32 = 1_000_000.0;
    const MK_TG_SIZE: u32 = 256;

    /// All Metal buffers backing one layer's weights, kept alive for
    /// the lifetime of the dispatch (their gpu_addresses live in the
    /// layer argbuf). `qb`/`kb`/`vb` are `None` when the source layer
    /// has no bias for that field.
    pub struct LayerMetalBuffers {
        pub qw: Buffer,
        pub kw: Buffer,
        pub vw: Buffer,
        pub ow: Buffer,
        pub gw: Buffer,
        pub uw: Buffer,
        pub dw: Buffer,
        pub attn_norm: Buffer,
        pub ffn_norm: Buffer,
        pub qb: Option<Buffer>,
        pub kb: Option<Buffer>,
        pub vb: Option<Buffer>,
    }

    impl LayerMetalBuffers {
        /// Allocate `metal::Buffer`s for every weight tensor in `w`,
        /// `memcpy` the host bytes in, and return the bundle paired
        /// with the populated `MkLayerArgs` argbuf descriptor.
        pub fn upload(ctx: &MetalContext, w: &MegakernelLayerWeightsF16) -> (Self, MkLayerArgs) {
            let halfs = |v: &[f16]| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
                };
                ctx.new_buffer_with_bytes(bytes)
            };
            let floats = |v: &[f32]| ctx.new_buffer_with_bytes(bytemuck::cast_slice(v));

            let qw = halfs(&w.q_proj);
            let kw = halfs(&w.k_proj);
            let vw = halfs(&w.v_proj);
            let ow = halfs(&w.o_proj);
            let gw = halfs(&w.ffn_gate);
            let uw = halfs(&w.ffn_up);
            let dw = halfs(&w.ffn_down);
            let attn_norm = floats(&w.attn_norm);
            let ffn_norm = floats(&w.ffn_norm);
            let qb = (!w.q_bias.is_empty()).then(|| floats(&w.q_bias));
            let kb = (!w.k_bias.is_empty()).then(|| floats(&w.k_bias));
            let vb = (!w.v_bias.is_empty()).then(|| floats(&w.v_bias));

            let args = MkLayerArgs {
                qw: qw.gpu_address(),
                kw: kw.gpu_address(),
                vw: vw.gpu_address(),
                ow: ow.gpu_address(),
                gw: gw.gpu_address(),
                uw: uw.gpu_address(),
                dw: dw.gpu_address(),
                attn_norm: attn_norm.gpu_address(),
                ffn_norm: ffn_norm.gpu_address(),
                qb: qb.as_ref().map_or(0, |b| b.gpu_address()),
                kb: kb.as_ref().map_or(0, |b| b.gpu_address()),
                vb: vb.as_ref().map_or(0, |b| b.gpu_address()),
                rms_eps: MK_RMS_EPS,
                rope_theta: MK_ROPE_THETA,
                has_qbias: qb.is_some() as u32,
                has_kbias: kb.is_some() as u32,
                has_vbias: vb.is_some() as u32,
                _padding: 0,
            };
            (
                Self {
                    qw,
                    kw,
                    vw,
                    ow,
                    gw,
                    uw,
                    dw,
                    attn_norm,
                    ffn_norm,
                    qb,
                    kb,
                    vb,
                },
                args,
            )
        }

        /// Mark every weight buffer this layer references as `Read` on
        /// the encoder so the driver keeps them resident across the
        /// dispatch. Used after the argbuf is encoded (the encoder
        /// only sees the argbuf directly).
        pub fn mark_used(&self, enc: &metal::ComputeCommandEncoderRef) {
            let mut refs: Vec<&metal::ResourceRef> = vec![
                &self.qw,
                &self.kw,
                &self.vw,
                &self.ow,
                &self.gw,
                &self.uw,
                &self.dw,
                &self.attn_norm,
                &self.ffn_norm,
            ];
            if let Some(b) = self.qb.as_ref() {
                refs.push(b);
            }
            if let Some(b) = self.kb.as_ref() {
                refs.push(b);
            }
            if let Some(b) = self.vb.as_ref() {
                refs.push(b);
            }
            enc.use_resources(&refs, MTLResourceUsage::Read);
        }
    }

    /// Dispatch the 2-layer Qwen-3B megakernel POC.
    ///
    /// Day-3 status: the dispatch harness is wired (weight upload →
    /// argbuf → `useResource` → dispatch → readback). The shader body
    /// is still a no-op pass-through (`x_out = x_in`), so the parity
    /// invariant this function gates on is the pass-through identity,
    /// NOT a full forward-vs-CPU comparison. Stages A..L land in
    /// follow-up sessions.
    ///
    /// `pos`/`seq_len`/`max_seq` are passed through to the kernel for
    /// when stages D (RoPE) / E (kv-write) / F (MHA) come online.
    pub fn megakernel_2layer_dispatch(
        ctx: &MetalContext,
        layer0: &MegakernelLayerWeightsF16,
        layer1: &MegakernelLayerWeightsF16,
        x_in: &[f16],
        pos: u32,
        seq_len: u32,
        max_seq: u32,
        probe_stage: u32,
    ) -> Result<Vec<f16>> {
        if x_in.len() != MK_HIDDEN {
            return Err(Error::Metal(format!(
                "megakernel_2layer_dispatch: x_in len {} != MK_HIDDEN {}",
                x_in.len(),
                MK_HIDDEN
            )));
        }
        if (max_seq as usize) < 1 {
            return Err(Error::Metal(
                "megakernel_2layer_dispatch: max_seq must be ≥ 1".into(),
            ));
        }

        // Residual buffers.
        let x_in_bytes = unsafe {
            std::slice::from_raw_parts(x_in.as_ptr() as *const u8, std::mem::size_of_val(x_in))
        };
        let x_in_buf = ctx.new_buffer_with_bytes(x_in_bytes);
        let x_out_buf = ctx.new_buffer(MK_HIDDEN * std::mem::size_of::<f16>());

        // KV cache: 2 layers × max_seq × kv_dim halfs each.
        let kv_bytes_per_buf = 2 * (max_seq as usize) * MK_KV_DIM * std::mem::size_of::<f16>();
        let k_cache_buf = ctx.new_buffer(kv_bytes_per_buf);
        let v_cache_buf = ctx.new_buffer(kv_bytes_per_buf);

        // FFN scratch (DRAM spill for stage I/J/K).
        let ffn_scratch_buf = ctx.new_buffer(MK_INTERMEDIATE * std::mem::size_of::<f16>());

        // Upload both layers' weights and build per-layer argbufs.
        let (l0_buffers, l0_args) = LayerMetalBuffers::upload(ctx, layer0);
        let (l1_buffers, l1_args) = LayerMetalBuffers::upload(ctx, layer1);

        let l0_arg_bytes: [u8; std::mem::size_of::<MkLayerArgs>()] =
            unsafe { std::mem::transmute(l0_args) };
        let l1_arg_bytes: [u8; std::mem::size_of::<MkLayerArgs>()] =
            unsafe { std::mem::transmute(l1_args) };
        let l0_arg_buf = ctx.new_buffer_with_bytes(&l0_arg_bytes);
        let l1_arg_buf = ctx.new_buffer_with_bytes(&l1_arg_bytes);

        // Scalar argbuf.
        let scalar_args = MkArgs {
            pos,
            seq_len,
            max_seq,
            probe_stage,
            n_layers: 2,
        };
        let scalar_bytes: [u8; std::mem::size_of::<MkArgs>()] =
            unsafe { std::mem::transmute(scalar_args) };
        let scalar_buf = ctx.new_buffer_with_bytes(&scalar_bytes);

        let shmem_bytes = (MK_SHMEM_HALFS * std::mem::size_of::<f16>()) as u64;

        ctx.dispatch_threads(
            "qwen3b_megakernel_2layer",
            (MK_TG_SIZE, 1, 1),
            (MK_TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&scalar_buf), 0);
                enc.set_buffer(1, Some(&x_in_buf), 0);
                enc.set_buffer(2, Some(&x_out_buf), 0);
                enc.set_buffer(3, Some(&k_cache_buf), 0);
                enc.set_buffer(4, Some(&v_cache_buf), 0);
                enc.set_buffer(5, Some(&ffn_scratch_buf), 0);
                enc.set_buffer(6, Some(&l0_arg_buf), 0);
                enc.set_buffer(7, Some(&l1_arg_buf), 0);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
                // Weight buffers are referenced only via gpu_address
                // inside the argbufs; tell the driver they're live.
                l0_buffers.mark_used(enc);
                l1_buffers.mark_used(enc);
            },
        )?;

        // Readback.
        let mut out = vec![f16::ZERO; MK_HIDDEN];
        let out_ptr = x_out_buf.contents() as *const f16;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, MK_HIDDEN) };
        out.copy_from_slice(out_slice);
        Ok(out)
    }

    /// Dispatch the N-layer Qwen-3B megakernel.
    ///
    /// Runs `layers.len()` transformer layers in a single GPU dispatch
    /// (one threadgroup, shared-memory working set) — the scaling path
    /// past the 2-layer correctness POC. Returns the hidden-state
    /// residual after the last layer.
    ///
    /// `x_in` is the pre-embedded hidden state. `pos` / `seq_len` index
    /// the KV cache as in `megakernel_2layer_dispatch`. `max_seq` is the
    /// per-layer KV stride; it must be ≥ `seq_len` and ≤ 256 (the shmem
    /// `scores` buffer caps attended length at the POC `MK_MAX_SEQ`).
    pub fn megakernel_nlayer_dispatch(
        ctx: &MetalContext,
        layers: &[MegakernelLayerWeightsF16],
        x_in: &[f16],
        pos: u32,
        seq_len: u32,
        max_seq: u32,
    ) -> Result<Vec<f16>> {
        if x_in.len() != MK_HIDDEN {
            return Err(Error::Metal(format!(
                "megakernel_nlayer_dispatch: x_in len {} != MK_HIDDEN {}",
                x_in.len(),
                MK_HIDDEN
            )));
        }
        if layers.is_empty() {
            return Err(Error::Metal(
                "megakernel_nlayer_dispatch: layers must be non-empty".into(),
            ));
        }
        if (max_seq as usize) < 1 || seq_len > max_seq {
            return Err(Error::Metal(format!(
                "megakernel_nlayer_dispatch: need 1 ≤ seq_len ({seq_len}) ≤ max_seq ({max_seq})"
            )));
        }
        let n = layers.len();

        // Residual buffers.
        let x_in_bytes = unsafe {
            std::slice::from_raw_parts(x_in.as_ptr() as *const u8, std::mem::size_of_val(x_in))
        };
        let x_in_buf = ctx.new_buffer_with_bytes(x_in_bytes);
        let x_out_buf = ctx.new_buffer(MK_HIDDEN * std::mem::size_of::<f16>());

        // KV cache: n layers × max_seq × kv_dim halfs per buffer. The
        // shader strides layer `li` by `max_seq * kv_dim`.
        let kv_bytes = n * (max_seq as usize) * MK_KV_DIM * std::mem::size_of::<f16>();
        let k_cache_buf = ctx.new_buffer(kv_bytes);
        let v_cache_buf = ctx.new_buffer(kv_bytes);

        // FFN scratch (reused per layer).
        let ffn_scratch_buf = ctx.new_buffer(MK_INTERMEDIATE * std::mem::size_of::<f16>());

        // Upload every layer's weights; keep the buffer bundles alive
        // for the dispatch (their gpu_addresses live in the argbufs).
        let mut layer_buffers: Vec<LayerMetalBuffers> = Vec::with_capacity(n);
        let mut layer_args: Vec<MkLayerArgs> = Vec::with_capacity(n);
        for w in layers {
            let (bufs, args) = LayerMetalBuffers::upload(ctx, w);
            layer_buffers.push(bufs);
            layer_args.push(args);
        }

        // Pack per-layer argbufs into one contiguous array buffer; the
        // buffer argument table can't hold 36 separate binds, so the
        // shader indexes `layers[li]`. MkLayerArgs is #[repr(C)] 120 B
        // with no padding (size check above), so the Vec is already the
        // wire layout.
        let layer_args_bytes = unsafe {
            std::slice::from_raw_parts(
                layer_args.as_ptr() as *const u8,
                std::mem::size_of_val(layer_args.as_slice()),
            )
        };
        let layer_args_buf = ctx.new_buffer_with_bytes(layer_args_bytes);

        // Scalar argbuf.
        let scalar_args = MkArgs {
            pos,
            seq_len,
            max_seq,
            probe_stage: 0,
            n_layers: n as u32,
        };
        let scalar_bytes: [u8; std::mem::size_of::<MkArgs>()] =
            unsafe { std::mem::transmute(scalar_args) };
        let scalar_buf = ctx.new_buffer_with_bytes(&scalar_bytes);

        let shmem_bytes = (MK_SHMEM_HALFS * std::mem::size_of::<f16>()) as u64;

        ctx.dispatch_threads(
            "qwen3b_megakernel_nlayer",
            (MK_TG_SIZE, 1, 1),
            (MK_TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&scalar_buf), 0);
                enc.set_buffer(1, Some(&x_in_buf), 0);
                enc.set_buffer(2, Some(&x_out_buf), 0);
                enc.set_buffer(3, Some(&k_cache_buf), 0);
                enc.set_buffer(4, Some(&v_cache_buf), 0);
                enc.set_buffer(5, Some(&ffn_scratch_buf), 0);
                enc.set_buffer(6, Some(&layer_args_buf), 0);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
                // Weight buffers are referenced only via gpu_address in
                // the argbufs; tell the driver to keep them resident.
                for lb in &layer_buffers {
                    lb.mark_used(enc);
                }
            },
        )?;

        // Readback.
        let mut out = vec![f16::ZERO; MK_HIDDEN];
        let out_ptr = x_out_buf.contents() as *const f16;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, MK_HIDDEN) };
        out.copy_from_slice(out_slice);
        Ok(out)
    }

    /// Persistent N-layer megakernel runner: uploads weights and
    /// allocates the KV / scratch / I/O buffers ONCE, then runs one
    /// fused dispatch per `step`. This is the shape a decode loop — or a
    /// steady-state bench — needs: re-dequantizing + re-uploading the
    /// f16 layers every token would dwarf the kernel itself. Output is
    /// identical to `megakernel_nlayer_dispatch`; only upload is hoisted.
    pub struct MegakernelRunner {
        layer_buffers: Vec<LayerMetalBuffers>,
        layer_args_buf: Buffer,
        k_cache_buf: Buffer,
        v_cache_buf: Buffer,
        ffn_scratch_buf: Buffer,
        x_in_buf: Buffer,
        x_out_buf: Buffer,
        n_layers: u32,
        max_seq: u32,
    }

    impl MegakernelRunner {
        pub fn new(
            ctx: &MetalContext,
            layers: &[MegakernelLayerWeightsF16],
            max_seq: u32,
        ) -> Result<Self> {
            if layers.is_empty() {
                return Err(Error::Metal(
                    "MegakernelRunner::new: layers must be non-empty".into(),
                ));
            }
            let n = layers.len();
            let mut layer_buffers = Vec::with_capacity(n);
            let mut layer_args = Vec::with_capacity(n);
            for w in layers {
                let (bufs, args) = LayerMetalBuffers::upload(ctx, w);
                layer_buffers.push(bufs);
                layer_args.push(args);
            }
            let layer_args_bytes = unsafe {
                std::slice::from_raw_parts(
                    layer_args.as_ptr() as *const u8,
                    std::mem::size_of_val(layer_args.as_slice()),
                )
            };
            let layer_args_buf = ctx.new_buffer_with_bytes(layer_args_bytes);
            let kv_bytes = n * (max_seq as usize) * MK_KV_DIM * std::mem::size_of::<f16>();
            Ok(Self {
                layer_buffers,
                layer_args_buf,
                k_cache_buf: ctx.new_buffer(kv_bytes),
                v_cache_buf: ctx.new_buffer(kv_bytes),
                ffn_scratch_buf: ctx.new_buffer(MK_INTERMEDIATE * std::mem::size_of::<f16>()),
                x_in_buf: ctx.new_buffer(MK_HIDDEN * std::mem::size_of::<f16>()),
                x_out_buf: ctx.new_buffer(MK_HIDDEN * std::mem::size_of::<f16>()),
                n_layers: n as u32,
                max_seq,
            })
        }

        /// One fused forward over all `n_layers`. Writes `x_in` into the
        /// persistent input buffer, dispatches, returns the residual.
        pub fn step(
            &self,
            ctx: &MetalContext,
            x_in: &[f16],
            pos: u32,
            seq_len: u32,
        ) -> Result<Vec<f16>> {
            if x_in.len() != MK_HIDDEN {
                return Err(Error::Metal(format!(
                    "MegakernelRunner::step: x_in len {} != {}",
                    x_in.len(),
                    MK_HIDDEN
                )));
            }
            unsafe {
                let dst = self.x_in_buf.contents() as *mut f16;
                std::ptr::copy_nonoverlapping(x_in.as_ptr(), dst, MK_HIDDEN);
            }
            let scalar_args = MkArgs {
                pos,
                seq_len,
                max_seq: self.max_seq,
                probe_stage: 0,
                n_layers: self.n_layers,
            };
            let scalar_bytes: [u8; std::mem::size_of::<MkArgs>()] =
                unsafe { std::mem::transmute(scalar_args) };
            let scalar_buf = ctx.new_buffer_with_bytes(&scalar_bytes);
            let shmem_bytes = (MK_SHMEM_HALFS * std::mem::size_of::<f16>()) as u64;
            ctx.dispatch_threads(
                "qwen3b_megakernel_nlayer",
                (MK_TG_SIZE, 1, 1),
                (MK_TG_SIZE, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&scalar_buf), 0);
                    enc.set_buffer(1, Some(&self.x_in_buf), 0);
                    enc.set_buffer(2, Some(&self.x_out_buf), 0);
                    enc.set_buffer(3, Some(&self.k_cache_buf), 0);
                    enc.set_buffer(4, Some(&self.v_cache_buf), 0);
                    enc.set_buffer(5, Some(&self.ffn_scratch_buf), 0);
                    enc.set_buffer(6, Some(&self.layer_args_buf), 0);
                    enc.set_threadgroup_memory_length(0, shmem_bytes);
                    for lb in &self.layer_buffers {
                        lb.mark_used(enc);
                    }
                },
            )?;
            let mut out = vec![f16::ZERO; MK_HIDDEN];
            let out_ptr = self.x_out_buf.contents() as *const f16;
            let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, MK_HIDDEN) };
            out.copy_from_slice(out_slice);
            Ok(out)
        }
    }
}

#[cfg(target_os = "macos")]
#[allow(unused_imports)]
pub use inner::{
    megakernel_2layer_dispatch, megakernel_nlayer_dispatch, LayerMetalBuffers, MegakernelRunner,
    MkArgs, MkLayerArgs, MK_PROBE_ATTN_OUT, MK_PROBE_FFN_DOWN, MK_PROBE_O_PROJ, MK_PROBE_Q_ROT,
    MK_PROBE_RESIDUAL, MK_PROBE_RESIDUAL_L0, MK_PROBE_XNORM_A, MK_PROBE_XNORM_FFN,
};
