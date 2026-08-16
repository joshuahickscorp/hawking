//! Native Qwen3.8 hybrid token graph: Q4 GEMVs + Q80 f32 activation
//! kernels + Q38-forked rearrange/GQA. Dense SwiGLU suffix. Zero fallbacks.

use super::qwen38_64_layer_execution_schedule::qwen38_assert_schedule_intact;
use super::qwen38_geometry::{
    qwen38_deltanet_state_slot, qwen38_gqa_state_slot, qwen38_layer_name, qwen38_mixer_kind,
    Qwen38DeltaNetLayout, Qwen38MixerKind, QWEN38_GQA_HEAD_DIM, QWEN38_GQA_HEADS,
    QWEN38_GQA_KV_HEADS, QWEN38_GQA_LAYERS, QWEN38_GQA_ROTARY_DIM, QWEN38_HIDDEN,
    QWEN38_INTERMEDIATE, QWEN38_LAYERS, QWEN38_RMS_EPS, QWEN38_ROPE_THETA, QWEN38_VOCAB,
};
use super::qwen38_pack::{
    load_qwen38_manifest, read_qwen38_f32_payload, QWEN38_EXPECTED_CATALOG_TENSORS,
};
use super::qwen_complete_binary::{parse_uniform_q4_header, UNIFORM_Q4_GROUP_SIZE};
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub const QWEN38_Q4_MATVEC_KERNEL: &str = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128";
pub const QWEN38_Q4_ADDR_PROBE_KERNEL: &str =
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe";
pub const QWEN38_Q4_DECODE_PROBE_KERNEL: &str =
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_decode_probe";
pub const QWEN38_F32_STREAM_PROBE_KERNEL: &str = "qwen38_f32_stream_probe";

/// Shipped uniform-Q4 matvec bindings. The Qwen3.8 default is the geometry-
/// sweep winner (`geo_tpr64_tg128`), tuned on Q80's 512×2048 organs. The
/// other names are already in `qwen_uniform_q4.metal`; this enum only
/// retargets launch geometry. It does not generate new shaders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38MatvecKernel {
    GeoTpr64Tg128,
    Vecgroup,
    VecgroupX64,
    VecgroupR4,
}

impl Qwen38MatvecKernel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GeoTpr64Tg128 => QWEN38_Q4_MATVEC_KERNEL,
            Self::Vecgroup => "qwen_uniform_q4_group64_matvec_vecgroup",
            Self::VecgroupX64 => "qwen_uniform_q4_group64_matvec_vecgroup_x64",
            Self::VecgroupR4 => "qwen_uniform_q4_group64_matvec_vecgroup_r4",
        }
    }

    /// (grid, threadgroup) for `rows` output elements.
    pub fn launch(self, rows: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
        match self {
            Self::GeoTpr64Tg128 => {
                let tg = 128u32;
                let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
            Self::Vecgroup => {
                let tg = 256u32;
                let grid = rows.div_ceil(8).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
            Self::VecgroupX64 => {
                let tg = 256u32;
                let grid = rows.div_ceil(4).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
            Self::VecgroupR4 => {
                let tg = 256u32;
                let grid = rows.div_ceil(32).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
        }
    }

    pub fn all() -> &'static [Self] {
        &[
            Self::GeoTpr64Tg128,
            Self::Vecgroup,
            Self::VecgroupX64,
            Self::VecgroupR4,
        ]
    }
}

/// Per-class GPU times from residual-correct split command buffers.
/// Each field's `gpu_ns` is `GPUEndTime-GPUStartTime` on that class's CBs.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Qwen38ClassTiming {
    pub embed_gpu_ns: Option<u64>,
    pub embed_wait_ns: u64,
    pub mixer_gpu_ns: u64,
    pub mixer_wait_ns: u64,
    pub mlp_gpu_ns: u64,
    pub mlp_wait_ns: u64,
    pub terminal_gpu_ns: Option<u64>,
    pub terminal_wait_ns: u64,
    pub deltanet_gpu_ns: u64,
    pub gqa_gpu_ns: u64,
    pub sampled: u32,
    pub layer_mlp_gpu_ns: Vec<u64>,
    pub layer_mixer_gpu_ns: Vec<u64>,
}

pub fn render_qwen38_user_chat(user_text: &str) -> String {
    format!("<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
}

pub fn load_qwen38_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
}

#[cfg(target_os = "macos")]
mod device {
    use super::*;
    use crate::kernels::{mha_decode_f32_tcb, qwen_next_add_residual_tcb, sample_argmax_f32_tcb};
    use crate::metal::{CommandBufferTiming, MetalContext, PinnedBuffer, TokenCommandBuffer};
    use std::time::Instant;

    fn zero_buffer(buffer: &PinnedBuffer) {
        let len = buffer.length() as usize;
        unsafe {
            std::ptr::write_bytes(buffer.contents() as *mut u8, 0, len);
        }
    }

    struct Q4Weight {
        rows: usize,
        cols: usize,
        codes: PinnedBuffer,
        scales: PinnedBuffer,
    }

    pub struct Qwen38HybridWorkspace {
        hidden: PinnedBuffer,
        normalized: PinnedBuffer,
        qkvz: PinnedBuffer,
        ba: PinnedBuffer,
        repeated_q: PinnedBuffer,
        repeated_k: PinnedBuffer,
        conv_v: PinnedBuffer,
        z: PinnedBuffer,
        decay: PinnedBuffer,
        beta: PinnedBuffer,
        rec_out: PinnedBuffer,
        gated: PinnedBuffer,
        mixer: PinnedBuffer,
        first_residual: PinnedBuffer,
        q_proj: PinnedBuffer,
        k_proj: PinnedBuffer,
        v_proj: PinnedBuffer,
        query: PinnedBuffer,
        attn: PinnedBuffer,
        gated_attn: PinnedBuffer,
        gate: PinnedBuffer,
        up: PinnedBuffer,
        act: PinnedBuffer,
        down: PinnedBuffer,
        logits: PinnedBuffer,
        sampled: PinnedBuffer,
        conv_state: PinnedBuffer,
        rec_state: PinnedBuffer,
        gqa_key: PinnedBuffer,
        gqa_value: PinnedBuffer,
    }

    impl Qwen38HybridWorkspace {
        fn allocate(ctx: &MetalContext, max_seq_len: usize) -> Result<Self> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let f32b = |n: usize| {
                n.checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))
            };
            let hidden = f32b(QWEN38_HIDDEN)?;
            let qkvz = f32b(layout.qkvz_rows())?;
            let ba = f32b(layout.ba_rows())?;
            let value = f32b(layout.value_elements())?;
            let q_proj = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM * 2)?;
            let kv = f32b(QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)?;
            let query = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM)?;
            let mid = f32b(QWEN38_INTERMEDIATE)?;
            let logits = f32b(QWEN38_VOCAB)?;
            let conv = f32b(48 * layout.conv_state_elements())?;
            let rec = f32b(48 * layout.recurrent_state_elements())?;
            let kv_cache =
                f32b(QWEN38_GQA_LAYERS * max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)?;
            Ok(Self {
                hidden: ctx.new_buffer_checked(hidden)?,
                normalized: ctx.new_buffer_checked(hidden)?,
                qkvz: ctx.new_buffer_checked(qkvz)?,
                ba: ctx.new_buffer_checked(ba)?,
                repeated_q: ctx.new_buffer_checked(value)?,
                repeated_k: ctx.new_buffer_checked(value)?,
                conv_v: ctx.new_buffer_checked(value)?,
                z: ctx.new_buffer_checked(value)?,
                decay: ctx.new_buffer_checked(f32b(layout.value_heads)?)?,
                beta: ctx.new_buffer_checked(f32b(layout.value_heads)?)?,
                rec_out: ctx.new_buffer_checked(value)?,
                gated: ctx.new_buffer_checked(value)?,
                mixer: ctx.new_buffer_checked(hidden)?,
                first_residual: ctx.new_buffer_checked(hidden)?,
                q_proj: ctx.new_buffer_checked(q_proj)?,
                k_proj: ctx.new_buffer_checked(kv)?,
                v_proj: ctx.new_buffer_checked(kv)?,
                query: ctx.new_buffer_checked(query)?,
                attn: ctx.new_buffer_checked(query)?,
                gated_attn: ctx.new_buffer_checked(query)?,
                gate: ctx.new_buffer_checked(mid)?,
                up: ctx.new_buffer_checked(mid)?,
                act: ctx.new_buffer_checked(mid)?,
                down: ctx.new_buffer_checked(hidden)?,
                logits: ctx.new_buffer_checked(logits)?,
                sampled: ctx.new_buffer_checked(std::mem::size_of::<u32>())?,
                conv_state: ctx.new_buffer_checked(conv)?,
                rec_state: ctx.new_buffer_checked(rec)?,
                gqa_key: ctx.new_buffer_checked(kv_cache)?,
                gqa_value: ctx.new_buffer_checked(kv_cache)?,
            })
        }
    }

    pub struct Qwen38HybridDecodeSession {
        #[allow(dead_code)]
        context: MetalContext,
        workspace: Qwen38HybridWorkspace,
        q4: HashMap<String, Q4Weight>,
        f32s: HashMap<String, PinnedBuffer>,
        max_seq_len: usize,
        position: usize,
        pub fallbacks: u32,
        /// Default matches the shipped bring-up binding. Diagnostic lanes may
        /// retarget to another shipped kernel; they must not invent one.
        pub matvec_kernel: Qwen38MatvecKernel,
        /// Overlap independent projections (gate+up, qkvz+ba, q/k/v) in one
        /// concurrent encoder. Off by default so `step` stays bit-identical
        /// to the bring-up vehicle.
        pub concurrent_independent: bool,
        /// Launch one threadgroup per (value-head, value-dim) for the
        /// gated-delta recurrence. Same serial reduction as the Q80 kernel;
        /// the vi columns are independent. Default ON after paired generate
        /// admitted a 42.7→33.4 ms token cut with greedy-identical ids.
        pub deltanet_vi_parallel: bool,
    }

    impl Qwen38HybridDecodeSession {
        pub fn open(root: impl AsRef<Path>, max_seq_len: usize) -> Result<Self> {
            qwen38_assert_schedule_intact()?;
            if max_seq_len == 0 {
                return Err(Error::Model("qwen38 max_seq_len must be positive".into()));
            }
            let root = root.as_ref();
            let (_manifest, rows) = load_qwen38_manifest(root)?;
            if rows.len() != QWEN38_EXPECTED_CATALOG_TENSORS {
                return Err(Error::Model(format!(
                    "qwen38 catalog has {} tensors, expected {QWEN38_EXPECTED_CATALOG_TENSORS}",
                    rows.len()
                )));
            }
            eprintln!(
                "qwen38-decode opening Metal + {} catalog tensors",
                rows.len()
            );
            let context = MetalContext::new()?;
            let workspace = Qwen38HybridWorkspace::allocate(&context, max_seq_len)?;
            let mut q4 = HashMap::new();
            let mut f32s = HashMap::new();
            let tensors_dir = root.join("tensors");
            for (i, row) in rows.iter().enumerate() {
                if i % 50 == 0 {
                    eprintln!("qwen38-decode upload {i}/{}", rows.len());
                }
                let path = tensors_dir.join(&row.artifact);
                let payload = fs::read(&path).map_err(|error| {
                    Error::Model(format!("cannot read {}: {error}", path.display()))
                })?;
                match row.kind.as_str() {
                    "q4" => {
                        let header = parse_uniform_q4_header(&payload)?;
                        let scales = &payload[header.scale_offset..header.sign_offset];
                        let codes = &payload[header.sign_offset..header.payload_bytes];
                        let (rows_n, cols) = match header.shape.as_slice() {
                            [r, c] => (*r, *c),
                            other => {
                                return Err(Error::Model(format!(
                                    "{} Q4 rank {:?} is not a matrix",
                                    row.name, other
                                )))
                            }
                        };
                        q4.insert(
                            row.name.clone(),
                            Q4Weight {
                                rows: rows_n,
                                cols,
                                codes: context.new_buffer_with_bytes_checked(codes)?,
                                scales: context.new_buffer_with_bytes_checked(scales)?,
                            },
                        );
                    }
                    "f32" => {
                        let values = read_qwen38_f32_payload(&payload)?;
                        f32s.insert(
                            row.name.clone(),
                            context.new_buffer_with_bytes_checked(bytemuck::cast_slice(&values))?,
                        );
                    }
                    other => {
                        return Err(Error::Model(format!(
                            "qwen38 catalog kind {other:?} is not q4/f32"
                        )))
                    }
                }
            }
            zero_buffer(&workspace.conv_state);
            zero_buffer(&workspace.rec_state);
            zero_buffer(&workspace.gqa_key);
            zero_buffer(&workspace.gqa_value);
            Ok(Self {
                context,
                workspace,
                q4,
                f32s,
                max_seq_len,
                position: 0,
                fallbacks: 0,
                matvec_kernel: Qwen38MatvecKernel::GeoTpr64Tg128,
                concurrent_independent: false,
                deltanet_vi_parallel: true,
            })
        }

        pub fn reset(&mut self) {
            self.position = 0;
            zero_buffer(&self.workspace.conv_state);
            zero_buffer(&self.workspace.rec_state);
            zero_buffer(&self.workspace.gqa_key);
            zero_buffer(&self.workspace.gqa_value);
        }

        fn q4(&self, name: &str) -> Result<&Q4Weight> {
            self.q4
                .get(name)
                .ok_or_else(|| Error::Model(format!("qwen38 missing Q4 {name}")))
        }

        fn f32(&self, name: &str) -> Result<&PinnedBuffer> {
            self.f32s
                .get(name)
                .ok_or_else(|| Error::Model(format!("qwen38 missing f32 {name}")))
        }

        fn encode_q4_matvec(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            self.encode_q4_matvec_kernel(tcb, name, input, output, self.matvec_kernel.as_str())
        }

        fn encode_q4_matvec_kernel(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
            kernel: &str,
        ) -> Result<()> {
            let weight = self.q4(name)?;
            let groups_per_row = weight.cols.div_ceil(UNIFORM_Q4_GROUP_SIZE) as u32;
            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            let (grid, tg) = self.matvec_kernel.launch(rows);
            tcb.dispatch_threads(kernel, grid, tg, |encoder| {
                encoder.set_buffer(0, Some(&weight.codes), 0);
                encoder.set_buffer(1, Some(&weight.scales), 0);
                encoder.set_buffer(2, Some(input), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &rows as *const u32 as *const _);
                encoder.set_bytes(5, 4, &cols as *const u32 as *const _);
                encoder.set_bytes(6, 4, &groups_per_row as *const u32 as *const _);
            })
        }

        fn encode_independent_q4_pair(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            a_name: &str,
            a_input: &PinnedBuffer,
            a_output: &PinnedBuffer,
            b_name: &str,
            b_input: &PinnedBuffer,
            b_output: &PinnedBuffer,
        ) -> Result<()> {
            if self.concurrent_independent {
                tcb.begin_concurrent_group()?;
            }
            self.encode_q4_matvec(tcb, a_name, a_input, a_output)?;
            self.encode_q4_matvec(tcb, b_name, b_input, b_output)?;
            if self.concurrent_independent {
                tcb.end_concurrent_group()?;
            }
            Ok(())
        }

        fn timed_cb(
            &self,
            encode: impl FnOnce(&mut TokenCommandBuffer<'_>) -> Result<()>,
        ) -> Result<CommandBufferTiming> {
            let mut tcb = TokenCommandBuffer::new(&self.context);
            encode(&mut tcb)?;
            tcb.commit_and_wait_timed()
        }

        fn encode_gated_delta(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            rec_off: u64,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let heads = layout.value_heads as u32;
            let kd = layout.key_head_dim as u32;
            let vd = layout.value_head_dim as u32;
            let (kernel, grid) = if self.deltanet_vi_parallel {
                (
                    "qwen38_gated_delta_decode_vi",
                    (kd, heads, vd),
                )
            } else {
                (
                    "qwen80_gated_delta_decode_tg",
                    (kd, heads, 1),
                )
            };
            tcb.dispatch_threads(kernel, grid, (kd, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&self.workspace.rec_state), rec_off);
                encoder.set_buffer(1, Some(&self.workspace.repeated_q), 0);
                encoder.set_buffer(2, Some(&self.workspace.repeated_k), 0);
                encoder.set_buffer(3, Some(&self.workspace.conv_v), 0);
                encoder.set_buffer(4, Some(&self.workspace.decay), 0);
                encoder.set_buffer(5, Some(&self.workspace.beta), 0);
                encoder.set_buffer(6, Some(&self.workspace.rec_out), 0);
                encoder.set_bytes(7, 4, &heads as *const u32 as *const _);
                encoder.set_bytes(8, 4, &kd as *const u32 as *const _);
                encoder.set_bytes(9, 4, &vd as *const u32 as *const _);
                encoder.set_threadgroup_memory_length(0, 128 * 4);
            })
        }

        fn encode_mixer(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            match qwen38_mixer_kind(layer)? {
                Qwen38MixerKind::DeltaNet => self.encode_deltanet(tcb, layer),
                Qwen38MixerKind::Gqa => self.encode_gqa(tcb, layer),
            }
        }

        fn encode_mixer_gemvs_only(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            match qwen38_mixer_kind(layer)? {
                Qwen38MixerKind::DeltaNet => {
                    self.encode_independent_q4_pair(
                        tcb,
                        &qwen38_layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                        &self.workspace.normalized,
                        &self.workspace.qkvz,
                        &qwen38_layer_name(layer, "linear_attn.in_proj_ba.weight"),
                        &self.workspace.normalized,
                        &self.workspace.ba,
                    )?;
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "linear_attn.out_proj.weight"),
                        &self.workspace.gated,
                        &self.workspace.mixer,
                    )
                }
                Qwen38MixerKind::Gqa => {
                    if self.concurrent_independent {
                        tcb.begin_concurrent_group()?;
                    }
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.q_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.q_proj,
                    )?;
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.k_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.k_proj,
                    )?;
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.v_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.v_proj,
                    )?;
                    if self.concurrent_independent {
                        tcb.end_concurrent_group()?;
                    }
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, "self_attn.o_proj.weight"),
                        &self.workspace.gated_attn,
                        &self.workspace.mixer,
                    )
                }
            }
        }

        fn encode_mlp_matvecs_only(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            self.encode_independent_q4_pair(
                tcb,
                &qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.gate,
                &qwen38_layer_name(layer, "mlp.up_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.up,
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act,
                &self.workspace.down,
            )
        }

        pub fn read_f32_workspace(&self, which: &str, n: usize) -> Result<Vec<f32>> {
            let buffer = match which {
                "gate" => &self.workspace.gate,
                "up" => &self.workspace.up,
                "act" => &self.workspace.act,
                "down" => &self.workspace.down,
                "hidden" => &self.workspace.hidden,
                "normalized" => &self.workspace.normalized,
                "logits" => &self.workspace.logits,
                "mixer" => &self.workspace.mixer,
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 unknown workspace buffer {other}"
                    )))
                }
            };
            let bytes = n
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| Error::Model("qwen38 read overflow".into()))?;
            if buffer.length() < bytes as u64 {
                return Err(Error::Model(format!(
                    "qwen38 {which} is {} bytes, need {bytes}",
                    buffer.length()
                )));
            }
            let mut out = vec![0.0f32; n];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.contents() as *const f32,
                    out.as_mut_ptr(),
                    n,
                );
            }
            Ok(out)
        }

        pub fn measure_named_matvec(&self, name: &str, output: &str) -> Result<CommandBufferTiming> {
            match output {
                "gate" | "up" | "down" | "logits" | "mixer" | "qkvz" | "hidden" => {}
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 unknown matvec output {other}"
                    )))
                }
            }
            self.timed_cb(|tcb| {
                let out_buf = match output {
                    "gate" => &self.workspace.gate,
                    "up" => &self.workspace.up,
                    "down" => &self.workspace.down,
                    "logits" => &self.workspace.logits,
                    "mixer" => &self.workspace.mixer,
                    "qkvz" => &self.workspace.qkvz,
                    _ => &self.workspace.hidden,
                };
                self.encode_q4_matvec(tcb, name, &self.workspace.normalized, out_buf)
            })
        }

        pub fn measure_isolated_mlp_full(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    self.encode_dense_mlp(tcb, layer, &self.workspace.first_residual)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_mlp_matvecs(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    self.encode_mlp_matvecs_only(tcb, layer)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_mlp_one_proj(&self, which: &str) -> Result<CommandBufferTiming> {
            let suffix = match which {
                "gate" => "mlp.gate_proj.weight",
                "up" => "mlp.up_proj.weight",
                "down" => "mlp.down_proj.weight",
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 mlp proj {other} is not gate/up/down"
                    )))
                }
            };
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    let (input, output) = match which {
                        "gate" => (&self.workspace.normalized, &self.workspace.gate),
                        "up" => (&self.workspace.normalized, &self.workspace.up),
                        _ => (&self.workspace.act, &self.workspace.down),
                    };
                    self.encode_q4_matvec(
                        tcb,
                        &qwen38_layer_name(layer, suffix),
                        input,
                        output,
                    )?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_gated_delta(&self) -> Result<CommandBufferTiming> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    if qwen38_mixer_kind(layer)? != Qwen38MixerKind::DeltaNet {
                        continue;
                    }
                    let slot = qwen38_deltanet_state_slot(layer)?;
                    let rec_off = (slot * layout.recurrent_state_elements() * 4) as u64;
                    self.encode_gated_delta(tcb, rec_off)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_mixer_gemvs(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    self.encode_mixer_gemvs_only(tcb, layer)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_lm_head(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                self.encode_q4_matvec(
                    tcb,
                    "language_model.lm_head.weight",
                    &self.workspace.normalized,
                    &self.workspace.logits,
                )
            })
        }

        pub fn measure_isolated_embed(&self, token: u32) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| self.encode_embed(tcb, token))
        }

        pub fn alloc_profile_buffer(&self, bytes: usize) -> Result<PinnedBuffer> {
            self.context.new_buffer_checked(bytes)
        }

        pub fn rec_state_f32_count(&self) -> usize {
            (self.workspace.rec_state.length() as usize) / 4
        }

        pub fn conv_state_f32_count(&self) -> usize {
            (self.workspace.conv_state.length() as usize) / 4
        }

        pub fn gqa_cache_f32_count(&self) -> usize {
            (self.workspace.gqa_key.length() as usize
                + self.workspace.gqa_value.length() as usize)
                / 4
        }

        fn encode_silu(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            let n = QWEN38_INTERMEDIATE as u32;
            tcb.dispatch_threads(
                crate::decode_family::swiglu_f32(),
                (n, 1, 1),
                (n.min(256).max(1), 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.gate), 0);
                    encoder.set_buffer(1, Some(&self.workspace.up), 0);
                    encoder.set_buffer(2, Some(&self.workspace.act), 0);
                    encoder.set_bytes(3, 4, &n as *const u32 as *const _);
                },
            )
        }

        fn encode_rearrange(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let slot = qwen38_deltanet_state_slot(layer)?;
            let conv_off = (slot * layout.conv_state_elements() * 4) as u64;
            let conv_w = self.f32(&qwen38_layer_name(layer, "linear_attn.conv1d.weight"))?;
            tcb.dispatch_threads(
                "qwen38_qkvz_rearrange_conv_l2_f32",
                (256, layout.key_heads as u32, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.qkvz), 0);
                    encoder.set_buffer(1, Some(conv_w), 0);
                    encoder.set_buffer(2, Some(&self.workspace.conv_state), conv_off);
                    encoder.set_buffer(3, Some(&self.workspace.repeated_q), 0);
                    encoder.set_buffer(4, Some(&self.workspace.repeated_k), 0);
                    encoder.set_buffer(5, Some(&self.workspace.conv_v), 0);
                    encoder.set_buffer(6, Some(&self.workspace.z), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    let kd = layout.key_head_dim as u32;
                    let vd = layout.value_head_dim as u32;
                    let ck = layout.conv_kernel as u32;
                    encoder.set_bytes(7, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(8, 4, &vpk as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &kd as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &vd as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &ck as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    encoder.set_threadgroup_memory_length(0, 4 * 256 * 4);
                },
            )
        }

        fn encode_ba_to_decay(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let a_log = self.f32(&qwen38_layer_name(layer, "linear_attn.A_log"))?;
            let dt_bias = self.f32(&qwen38_layer_name(layer, "linear_attn.dt_bias"))?;
            tcb.dispatch_threads(
                "qwen80_ba_to_decay_beta_f32",
                (layout.value_heads as u32, 1, 1),
                (16, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.ba), 0);
                    encoder.set_buffer(1, Some(a_log), 0);
                    encoder.set_buffer(2, Some(dt_bias), 0);
                    encoder.set_buffer(3, Some(&self.workspace.decay), 0);
                    encoder.set_buffer(4, Some(&self.workspace.beta), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    encoder.set_bytes(5, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &vpk as *const u32 as *const _);
                },
            )
        }

        fn encode_gated_rmsnorm(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let norm_w = self.f32(&qwen38_layer_name(layer, "linear_attn.norm.weight"))?;
            tcb.dispatch_threads(
                "qwen80_deltanet_gated_rmsnorm_f32",
                (layout.value_heads as u32, 1, 1),
                (16, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.rec_out), 0);
                    encoder.set_buffer(1, Some(&self.workspace.z), 0);
                    encoder.set_buffer(2, Some(norm_w), 0);
                    encoder.set_buffer(3, Some(&self.workspace.gated), 0);
                    let heads = layout.value_heads as u32;
                    let dim = layout.value_head_dim as u32;
                    encoder.set_bytes(4, 4, &heads as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &dim as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )
        }

        fn encode_rope_cache(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let slot = qwen38_gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = (slot * slot_elems * 4) as u64;
            let q_norm = self.f32(&qwen38_layer_name(layer, "self_attn.q_norm.weight"))?;
            let k_norm = self.f32(&qwen38_layer_name(layer, "self_attn.k_norm.weight"))?;
            tcb.dispatch_threads(
                "qwen38_gqa_qk_norm_rope_cache_f32",
                (QWEN38_GQA_HEADS as u32, 1, 1),
                (QWEN38_GQA_HEADS as u32, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(1, Some(&self.workspace.k_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.v_proj), 0);
                    encoder.set_buffer(3, Some(q_norm), 0);
                    encoder.set_buffer(4, Some(k_norm), 0);
                    encoder.set_buffer(5, Some(&self.workspace.query), 0);
                    encoder.set_buffer(6, Some(&self.workspace.gqa_key), cache_off);
                    encoder.set_buffer(7, Some(&self.workspace.gqa_value), cache_off);
                    let pos = self.position.saturating_sub(1).min(self.max_seq_len - 1) as u32;
                    let nh = QWEN38_GQA_HEADS as u32;
                    let nkv = QWEN38_GQA_KV_HEADS as u32;
                    let hd = QWEN38_GQA_HEAD_DIM as u32;
                    let rd = QWEN38_GQA_ROTARY_DIM as u32;
                    encoder.set_bytes(8, 4, &pos as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &nh as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &nkv as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &hd as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &rd as *const u32 as *const _);
                    encoder.set_bytes(13, 4, &QWEN38_ROPE_THETA as *const f32 as *const _);
                    encoder.set_bytes(14, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )
        }

        fn encode_mha(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let slot = qwen38_gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = (slot * slot_elems * 4) as u64;
            let seq = self.position.max(1).min(self.max_seq_len);
            mha_decode_f32_tcb(
                tcb,
                &self.workspace.query,
                &self.workspace.gqa_key,
                cache_off as usize,
                &self.workspace.gqa_value,
                cache_off as usize,
                &self.workspace.attn,
                seq,
                QWEN38_GQA_HEAD_DIM,
                QWEN38_GQA_HEADS,
                QWEN38_GQA_KV_HEADS,
            )
        }

        fn encode_sigmoid_gate(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            let query_dim = (QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM) as u32;
            let head_dim = QWEN38_GQA_HEAD_DIM as u32;
            tcb.dispatch_threads(
                "qwen38_attention_apply_sigmoid_gate",
                (query_dim, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.attn), 0);
                    encoder.set_buffer(1, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.gated_attn), 0);
                    encoder.set_bytes(3, 4, &query_dim as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &head_dim as *const u32 as *const _);
                },
            )
        }

        fn encode_argmax(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            sample_argmax_f32_tcb(
                tcb,
                &self.workspace.logits,
                &self.workspace.sampled,
                QWEN38_VOCAB,
            )
        }

        fn encode_f32_stream(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            src: &PinnedBuffer,
            dst: &PinnedBuffer,
            n_f32: u32,
        ) -> Result<()> {
            tcb.dispatch_threads(
                QWEN38_F32_STREAM_PROBE_KERNEL,
                (n_f32, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(src), 0);
                    encoder.set_buffer(1, Some(dst), 0);
                    encoder.set_bytes(2, 4, &n_f32 as *const u32 as *const _);
                },
            )
        }

        pub fn measure_isolated_family(
            &self,
            family: &str,
        ) -> Result<CommandBufferTiming> {
            match family {
                "input_norms" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        self.encode_rmsnorm(
                            tcb,
                            &self.workspace.hidden,
                            &qwen38_layer_name(layer, "input_layernorm.weight"),
                            &self.workspace.normalized,
                            QWEN38_HIDDEN as u32,
                        )?;
                    }
                    Ok(())
                }),
                "post_norms" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        self.encode_rmsnorm(
                            tcb,
                            &self.workspace.first_residual,
                            &qwen38_layer_name(layer, "post_attention_layernorm.weight"),
                            &self.workspace.normalized,
                            QWEN38_HIDDEN as u32,
                        )?;
                    }
                    Ok(())
                }),
                "final_norm" => self.timed_cb(|tcb| {
                    self.encode_rmsnorm(
                        tcb,
                        &self.workspace.hidden,
                        "language_model.model.norm.weight",
                        &self.workspace.normalized,
                        QWEN38_HIDDEN as u32,
                    )
                }),
                "silu_64" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_LAYERS {
                        self.encode_silu(tcb)?;
                    }
                    Ok(())
                }),
                "mlp_residual_64" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_LAYERS {
                        qwen_next_add_residual_tcb(
                            tcb,
                            &self.workspace.first_residual,
                            &self.workspace.down,
                            &self.workspace.hidden,
                            QWEN38_HIDDEN,
                        )?;
                    }
                    Ok(())
                }),
                "mixer_residual_64" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_LAYERS {
                        qwen_next_add_residual_tcb(
                            tcb,
                            &self.workspace.hidden,
                            &self.workspace.mixer,
                            &self.workspace.first_residual,
                            QWEN38_HIDDEN,
                        )?;
                    }
                    Ok(())
                }),
                "rearrange_48" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? == Qwen38MixerKind::DeltaNet {
                            self.encode_rearrange(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "ba_to_decay_48" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? == Qwen38MixerKind::DeltaNet {
                            self.encode_ba_to_decay(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "gated_rmsnorm_48" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? == Qwen38MixerKind::DeltaNet {
                            self.encode_gated_rmsnorm(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "rope_cache_16" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? == Qwen38MixerKind::Gqa {
                            self.encode_rope_cache(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "mha_16" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? == Qwen38MixerKind::Gqa {
                            self.encode_mha(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "sigmoid_16" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_GQA_LAYERS {
                        self.encode_sigmoid_gate(tcb)?;
                    }
                    Ok(())
                }),
                "argmax" => self.timed_cb(|tcb| self.encode_argmax(tcb)),
                "dn_gemvs" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? != Qwen38MixerKind::DeltaNet {
                            continue;
                        }
                        self.encode_independent_q4_pair(
                            tcb,
                            &qwen38_layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                            &self.workspace.normalized,
                            &self.workspace.qkvz,
                            &qwen38_layer_name(layer, "linear_attn.in_proj_ba.weight"),
                            &self.workspace.normalized,
                            &self.workspace.ba,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            &qwen38_layer_name(layer, "linear_attn.out_proj.weight"),
                            &self.workspace.gated,
                            &self.workspace.mixer,
                        )?;
                    }
                    Ok(())
                }),
                "gqa_gemvs" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? != Qwen38MixerKind::Gqa {
                            continue;
                        }
                        self.encode_q4_matvec(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.q_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.q_proj,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.k_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.k_proj,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.v_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.v_proj,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.o_proj.weight"),
                            &self.workspace.gated_attn,
                            &self.workspace.mixer,
                        )?;
                    }
                    Ok(())
                }),
                other => Err(Error::Model(format!(
                    "qwen38 unknown isolated family {other}"
                ))),
            }
        }

        pub fn measure_isolated_mlp_one_proj_kernel(
            &self,
            which: &str,
            kernel: &str,
        ) -> Result<CommandBufferTiming> {
            let suffix = match which {
                "gate" => "mlp.gate_proj.weight",
                "up" => "mlp.up_proj.weight",
                "down" => "mlp.down_proj.weight",
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 mlp proj {other} is not gate/up/down"
                    )))
                }
            };
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    let (input, output) = match which {
                        "gate" => (&self.workspace.normalized, &self.workspace.gate),
                        "up" => (&self.workspace.normalized, &self.workspace.up),
                        _ => (&self.workspace.act, &self.workspace.down),
                    };
                    self.encode_q4_matvec_kernel(
                        tcb,
                        &qwen38_layer_name(layer, suffix),
                        input,
                        output,
                        kernel,
                    )?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_class_gemvs_kernel(
            &self,
            class: &str,
            kernel: &str,
        ) -> Result<CommandBufferTiming> {
            match class {
                "mlp" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.gate,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "mlp.up_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.up,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "mlp.down_proj.weight"),
                            &self.workspace.act,
                            &self.workspace.down,
                            kernel,
                        )?;
                    }
                    Ok(())
                }),
                "dn" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? != Qwen38MixerKind::DeltaNet {
                            continue;
                        }
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                            &self.workspace.normalized,
                            &self.workspace.qkvz,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "linear_attn.in_proj_ba.weight"),
                            &self.workspace.normalized,
                            &self.workspace.ba,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "linear_attn.out_proj.weight"),
                            &self.workspace.gated,
                            &self.workspace.mixer,
                            kernel,
                        )?;
                    }
                    Ok(())
                }),
                "gqa" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if qwen38_mixer_kind(layer)? != Qwen38MixerKind::Gqa {
                            continue;
                        }
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.q_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.q_proj,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.k_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.k_proj,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.v_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.v_proj,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            &qwen38_layer_name(layer, "self_attn.o_proj.weight"),
                            &self.workspace.gated_attn,
                            &self.workspace.mixer,
                            kernel,
                        )?;
                    }
                    Ok(())
                }),
                "lm_head" => self.timed_cb(|tcb| {
                    self.encode_q4_matvec_kernel(
                        tcb,
                        "language_model.lm_head.weight",
                        &self.workspace.normalized,
                        &self.workspace.logits,
                        kernel,
                    )
                }),
                other => Err(Error::Model(format!(
                    "qwen38 unknown gemv class {other}"
                ))),
            }
        }

        pub fn measure_f32_stream(
            &self,
            which: &str,
            dest: &PinnedBuffer,
        ) -> Result<CommandBufferTiming> {
            let (src, n) = match which {
                "rec_state" => (&self.workspace.rec_state, self.rec_state_f32_count()),
                "conv_state" => (&self.workspace.conv_state, self.conv_state_f32_count()),
                "gqa_key" => (
                    &self.workspace.gqa_key,
                    (self.workspace.gqa_key.length() as usize) / 4,
                ),
                "gqa_value" => (
                    &self.workspace.gqa_value,
                    (self.workspace.gqa_value.length() as usize) / 4,
                ),
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 unknown stream source {other}"
                    )))
                }
            };
            let n = n.min((dest.length() as usize) / 4);
            self.timed_cb(|tcb| self.encode_f32_stream(tcb, src, dest, n as u32))
        }

        pub fn step_decomposed(&mut self, token: u32) -> Result<(u32, Qwen38ClassTiming)> {
            if self.fallbacks != 0 {
                return Err(Error::Model(
                    "qwen38 decode refuses a run after a fallback".into(),
                ));
            }
            let mut out = Qwen38ClassTiming::default();
            let embed = self.timed_cb(|tcb| self.encode_embed(tcb, token))?;
            out.embed_gpu_ns = embed.gpu_ns;
            out.embed_wait_ns = embed.wait_ns;
            for layer in 0..QWEN38_LAYERS {
                let mixer = self.timed_cb(|tcb| self.encode_mixer(tcb, layer))?;
                let mixer_gpu = mixer.gpu_ns.unwrap_or(0);
                out.mixer_gpu_ns = out.mixer_gpu_ns.saturating_add(mixer_gpu);
                out.mixer_wait_ns = out.mixer_wait_ns.saturating_add(mixer.wait_ns);
                out.layer_mixer_gpu_ns.push(mixer_gpu);
                match qwen38_mixer_kind(layer)? {
                    Qwen38MixerKind::DeltaNet => {
                        out.deltanet_gpu_ns = out.deltanet_gpu_ns.saturating_add(mixer_gpu);
                    }
                    Qwen38MixerKind::Gqa => {
                        out.gqa_gpu_ns = out.gqa_gpu_ns.saturating_add(mixer_gpu);
                    }
                }
                let mlp = self.timed_cb(|tcb| {
                    self.encode_dense_mlp(tcb, layer, &self.workspace.first_residual)
                })?;
                let mlp_gpu = mlp.gpu_ns.unwrap_or(0);
                out.mlp_gpu_ns = out.mlp_gpu_ns.saturating_add(mlp_gpu);
                out.mlp_wait_ns = out.mlp_wait_ns.saturating_add(mlp.wait_ns);
                out.layer_mlp_gpu_ns.push(mlp_gpu);
            }
            let term = self.timed_cb(|tcb| self.encode_terminal(tcb))?;
            out.terminal_gpu_ns = term.gpu_ns;
            out.terminal_wait_ns = term.wait_ns;
            let sampled = unsafe { *(self.workspace.sampled.contents() as *const u32) };
            self.position = self.position.saturating_add(1);
            out.sampled = sampled;
            Ok((sampled, out))
        }

        fn encode_rmsnorm(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            input: &PinnedBuffer,
            weight_name: &str,
            output: &PinnedBuffer,
            hidden: u32,
        ) -> Result<()> {
            let weight = self.f32(weight_name)?;
            tcb.dispatch_threads(
                "qwen80_residual_rmsnorm_f32",
                (256, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(weight), 0);
                    encoder.set_buffer(2, Some(output), 0);
                    encoder.set_bytes(3, 4, &hidden as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    encoder.set_threadgroup_memory_length(0, 256 * 4);
                },
            )
        }

        fn encode_embed(&self, tcb: &mut TokenCommandBuffer<'_>, token: u32) -> Result<()> {
            let weight = self.q4("language_model.model.embed_tokens.weight")?;
            if weight.rows != QWEN38_VOCAB || weight.cols != QWEN38_HIDDEN {
                return Err(Error::Model("qwen38 embed shape drifted".into()));
            }
            let hidden = QWEN38_HIDDEN as u32;
            let vocab = QWEN38_VOCAB as u32;
            let group = UNIFORM_Q4_GROUP_SIZE as u32;
            tcb.dispatch_threads(
                "qwen_uniform_q4_embedding_lookup",
                (hidden, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&weight.codes), 0);
                    encoder.set_buffer(1, Some(&weight.scales), 0);
                    encoder.set_buffer(2, Some(&self.workspace.hidden), 0);
                    encoder.set_bytes(3, 4, &token as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &hidden as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &vocab as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &group as *const u32 as *const _);
                },
            )
        }

        fn encode_dense_mlp(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
            input: &PinnedBuffer,
        ) -> Result<()> {
            let n = QWEN38_INTERMEDIATE as u32;
            self.encode_rmsnorm(
                tcb,
                input,
                &qwen38_layer_name(layer, "post_attention_layernorm.weight"),
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            self.encode_independent_q4_pair(
                tcb,
                &qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.gate,
                &qwen38_layer_name(layer, "mlp.up_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.up,
            )?;
            tcb.dispatch_threads(
                crate::decode_family::swiglu_f32(),
                (n, 1, 1),
                (n.min(256).max(1), 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.gate), 0);
                    encoder.set_buffer(1, Some(&self.workspace.up), 0);
                    encoder.set_buffer(2, Some(&self.workspace.act), 0);
                    encoder.set_bytes(3, 4, &n as *const u32 as *const _);
                },
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act,
                &self.workspace.down,
            )?;
            qwen_next_add_residual_tcb(
                tcb,
                input,
                &self.workspace.down,
                &self.workspace.hidden,
                QWEN38_HIDDEN,
            )
        }

        fn encode_deltanet(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let slot = qwen38_deltanet_state_slot(layer)?;
            let conv_off = (slot * layout.conv_state_elements() * 4) as u64;
            let rec_off = (slot * layout.recurrent_state_elements() * 4) as u64;
            self.encode_rmsnorm(
                tcb,
                &self.workspace.hidden,
                &qwen38_layer_name(layer, "input_layernorm.weight"),
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            self.encode_independent_q4_pair(
                tcb,
                &qwen38_layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                &self.workspace.normalized,
                &self.workspace.qkvz,
                &qwen38_layer_name(layer, "linear_attn.in_proj_ba.weight"),
                &self.workspace.normalized,
                &self.workspace.ba,
            )?;
            let conv_w = self.f32(&qwen38_layer_name(layer, "linear_attn.conv1d.weight"))?;
            tcb.dispatch_threads(
                "qwen38_qkvz_rearrange_conv_l2_f32",
                (256, layout.key_heads as u32, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.qkvz), 0);
                    encoder.set_buffer(1, Some(conv_w), 0);
                    encoder.set_buffer(2, Some(&self.workspace.conv_state), conv_off);
                    encoder.set_buffer(3, Some(&self.workspace.repeated_q), 0);
                    encoder.set_buffer(4, Some(&self.workspace.repeated_k), 0);
                    encoder.set_buffer(5, Some(&self.workspace.conv_v), 0);
                    encoder.set_buffer(6, Some(&self.workspace.z), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    let kd = layout.key_head_dim as u32;
                    let vd = layout.value_head_dim as u32;
                    let ck = layout.conv_kernel as u32;
                    encoder.set_bytes(7, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(8, 4, &vpk as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &kd as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &vd as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &ck as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    encoder.set_threadgroup_memory_length(0, 4 * 256 * 4);
                },
            )?;
            let a_log = self.f32(&qwen38_layer_name(layer, "linear_attn.A_log"))?;
            let dt_bias = self.f32(&qwen38_layer_name(layer, "linear_attn.dt_bias"))?;
            tcb.dispatch_threads(
                "qwen80_ba_to_decay_beta_f32",
                (layout.value_heads as u32, 1, 1),
                (16, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.ba), 0);
                    encoder.set_buffer(1, Some(a_log), 0);
                    encoder.set_buffer(2, Some(dt_bias), 0);
                    encoder.set_buffer(3, Some(&self.workspace.decay), 0);
                    encoder.set_buffer(4, Some(&self.workspace.beta), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    encoder.set_bytes(5, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &vpk as *const u32 as *const _);
                },
            )?;
            self.encode_gated_delta(tcb, rec_off)?;
            let norm_w = self.f32(&qwen38_layer_name(layer, "linear_attn.norm.weight"))?;
            tcb.dispatch_threads(
                "qwen80_deltanet_gated_rmsnorm_f32",
                (layout.value_heads as u32, 1, 1),
                (16, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.rec_out), 0);
                    encoder.set_buffer(1, Some(&self.workspace.z), 0);
                    encoder.set_buffer(2, Some(norm_w), 0);
                    encoder.set_buffer(3, Some(&self.workspace.gated), 0);
                    let heads = layout.value_heads as u32;
                    let dim = layout.value_head_dim as u32;
                    encoder.set_bytes(4, 4, &heads as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &dim as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "linear_attn.out_proj.weight"),
                &self.workspace.gated,
                &self.workspace.mixer,
            )?;
            qwen_next_add_residual_tcb(
                tcb,
                &self.workspace.hidden,
                &self.workspace.mixer,
                &self.workspace.first_residual,
                QWEN38_HIDDEN,
            )
        }

        fn encode_gqa(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            if self.position >= self.max_seq_len {
                return Err(Error::Model(format!(
                    "qwen38 GQA position {} exceeds max_seq_len {}",
                    self.position, self.max_seq_len
                )));
            }
            let slot = qwen38_gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = (slot * slot_elems * 4) as u64;
            self.encode_rmsnorm(
                tcb,
                &self.workspace.hidden,
                &qwen38_layer_name(layer, "input_layernorm.weight"),
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            if self.concurrent_independent {
                tcb.begin_concurrent_group()?;
            }
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.q_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.q_proj,
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.k_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.k_proj,
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.v_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.v_proj,
            )?;
            if self.concurrent_independent {
                tcb.end_concurrent_group()?;
            }
            let q_norm = self.f32(&qwen38_layer_name(layer, "self_attn.q_norm.weight"))?;
            let k_norm = self.f32(&qwen38_layer_name(layer, "self_attn.k_norm.weight"))?;
            tcb.dispatch_threads(
                "qwen38_gqa_qk_norm_rope_cache_f32",
                (QWEN38_GQA_HEADS as u32, 1, 1),
                (QWEN38_GQA_HEADS as u32, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(1, Some(&self.workspace.k_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.v_proj), 0);
                    encoder.set_buffer(3, Some(q_norm), 0);
                    encoder.set_buffer(4, Some(k_norm), 0);
                    encoder.set_buffer(5, Some(&self.workspace.query), 0);
                    encoder.set_buffer(6, Some(&self.workspace.gqa_key), cache_off);
                    encoder.set_buffer(7, Some(&self.workspace.gqa_value), cache_off);
                    let pos = self.position as u32;
                    let nh = QWEN38_GQA_HEADS as u32;
                    let nkv = QWEN38_GQA_KV_HEADS as u32;
                    let hd = QWEN38_GQA_HEAD_DIM as u32;
                    let rd = QWEN38_GQA_ROTARY_DIM as u32;
                    encoder.set_bytes(8, 4, &pos as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &nh as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &nkv as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &hd as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &rd as *const u32 as *const _);
                    encoder.set_bytes(13, 4, &QWEN38_ROPE_THETA as *const f32 as *const _);
                    encoder.set_bytes(14, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )?;
            mha_decode_f32_tcb(
                tcb,
                &self.workspace.query,
                &self.workspace.gqa_key,
                cache_off as usize,
                &self.workspace.gqa_value,
                cache_off as usize,
                &self.workspace.attn,
                self.position + 1,
                QWEN38_GQA_HEAD_DIM,
                QWEN38_GQA_HEADS,
                QWEN38_GQA_KV_HEADS,
            )?;
            let query_dim = (QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM) as u32;
            let head_dim = QWEN38_GQA_HEAD_DIM as u32;
            tcb.dispatch_threads(
                "qwen38_attention_apply_sigmoid_gate",
                (query_dim, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.attn), 0);
                    encoder.set_buffer(1, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.gated_attn), 0);
                    encoder.set_bytes(3, 4, &query_dim as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &head_dim as *const u32 as *const _);
                },
            )?;
            self.encode_q4_matvec(
                tcb,
                &qwen38_layer_name(layer, "self_attn.o_proj.weight"),
                &self.workspace.gated_attn,
                &self.workspace.mixer,
            )?;
            qwen_next_add_residual_tcb(
                tcb,
                &self.workspace.hidden,
                &self.workspace.mixer,
                &self.workspace.first_residual,
                QWEN38_HIDDEN,
            )
        }

        fn encode_layers(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            for layer in 0..QWEN38_LAYERS {
                match qwen38_mixer_kind(layer)? {
                    Qwen38MixerKind::DeltaNet => self.encode_deltanet(tcb, layer)?,
                    Qwen38MixerKind::Gqa => self.encode_gqa(tcb, layer)?,
                }
                self.encode_dense_mlp(tcb, layer, &self.workspace.first_residual)?;
            }
            Ok(())
        }

        fn encode_terminal(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            self.encode_rmsnorm(
                tcb,
                &self.workspace.hidden,
                "language_model.model.norm.weight",
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            self.encode_q4_matvec(
                tcb,
                "language_model.lm_head.weight",
                &self.workspace.normalized,
                &self.workspace.logits,
            )?;
            sample_argmax_f32_tcb(
                tcb,
                &self.workspace.logits,
                &self.workspace.sampled,
                QWEN38_VOCAB,
            )
        }

        pub fn step(&mut self, token: u32) -> Result<(u32, CommandBufferTiming)> {
            if self.fallbacks != 0 {
                return Err(Error::Model(
                    "qwen38 decode refuses a run after a fallback".into(),
                ));
            }
            let encode_t0 = Instant::now();
            let mut tcb = TokenCommandBuffer::new(&self.context);
            self.encode_embed(&mut tcb, token)?;
            self.encode_layers(&mut tcb)?;
            self.encode_terminal(&mut tcb)?;
            let encode_ns = encode_t0.elapsed().as_nanos() as u64;
            let mut timing = tcb.commit_and_wait_timed()?;
            if timing.encode_ns == 0 {
                timing.encode_ns = encode_ns;
            }
            let sampled = unsafe { *(self.workspace.sampled.contents() as *const u32) };
            self.position = self.position.saturating_add(1);
            Ok((sampled, timing))
        }

        /// Same GPU work as [`Self::step`], with host-side Instants around
        /// encode / commit-return / sample readback / position update so
        /// wall − gpu can be named. Timers do not change the command buffer.
        pub fn step_complete(&mut self, token: u32) -> Result<(u32, Qwen38StepWall)> {
            if self.fallbacks != 0 {
                return Err(Error::Model(
                    "qwen38 decode refuses a run after a fallback".into(),
                ));
            }
            let wall = Instant::now();
            let encode_started = Instant::now();
            let mut tcb = TokenCommandBuffer::new(&self.context);
            self.encode_embed(&mut tcb, token)?;
            self.encode_layers(&mut tcb)?;
            self.encode_terminal(&mut tcb)?;
            let encode_ns = encode_started.elapsed().as_nanos() as u64;
            let commit_started = Instant::now();
            let timing = tcb.commit_and_wait_timed()?;
            let commit_return_ns = commit_started.elapsed().as_nanos() as u64;
            let submit_plus_wait = timing.submit_ns.saturating_add(timing.wait_ns);
            let commit_epilogue_ns = commit_return_ns.saturating_sub(submit_plus_wait);
            let readback_started = Instant::now();
            let sampled = unsafe { *(self.workspace.sampled.contents() as *const u32) };
            let sample_readback_ns = readback_started.elapsed().as_nanos() as u64;
            let state_started = Instant::now();
            self.position = self.position.saturating_add(1);
            let state_update_ns = state_started.elapsed().as_nanos() as u64;
            Ok((
                sampled,
                Qwen38StepWall {
                    wall_ns: wall.elapsed().as_nanos() as u64,
                    encode_ns,
                    submit_ns: timing.submit_ns,
                    wait_ns: timing.wait_ns,
                    gpu_ns: timing.gpu_ns,
                    commit_epilogue_ns,
                    sample_readback_ns,
                    state_update_ns,
                    tcb_encode_ns: timing.encode_ns,
                    dispatches: timing.dispatches,
                    command_buffers: 1,
                },
            ))
        }
    }

    pub fn generate_greedy(
        session: &mut Qwen38HybridDecodeSession,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> Result<Qwen38GenerateResult> {
        if prompt.is_empty() {
            return Err(Error::Model("qwen38 prompt is empty".into()));
        }
        session.reset();
        let mut tokens = prompt.to_vec();
        let mut gpu_ns = Vec::new();
        let mut wait_ns = Vec::new();
        let mut encode_ns = Vec::new();
        let mut submit_ns = Vec::new();
        let mut dispatches = Vec::new();
        let mut wall_ns_per_step = Vec::new();
        let wall = Instant::now();
        let mut next = 0u32;
        let prefill = Instant::now();
        let mut first_step_wall_ns = 0u64;
        for (i, &token) in prompt.iter().enumerate() {
            let step_wall = Instant::now();
            let (sampled, timing) = session.step(token)?;
            let step_ns = step_wall.elapsed().as_nanos() as u64;
            if i == 0 {
                first_step_wall_ns = step_ns;
            }
            wall_ns_per_step.push(step_ns);
            gpu_ns.push(timing.gpu_ns);
            wait_ns.push(timing.wait_ns);
            encode_ns.push(timing.encode_ns);
            submit_ns.push(timing.submit_ns);
            dispatches.push(timing.dispatches);
            next = sampled;
        }
        let prefill_wall_ns = prefill.elapsed().as_nanos() as u64;
        tokens.push(next);
        let decode = Instant::now();
        while tokens.len() - prompt.len() < max_new_tokens {
            if next == crate::model::qwen38_geometry::QWEN38_EOS_IM_END
                || next == crate::model::qwen38_geometry::QWEN38_EOS_END_OF_TEXT
            {
                break;
            }
            let step_wall = Instant::now();
            let (sampled, timing) = session.step(next)?;
            wall_ns_per_step.push(step_wall.elapsed().as_nanos() as u64);
            gpu_ns.push(timing.gpu_ns);
            wait_ns.push(timing.wait_ns);
            encode_ns.push(timing.encode_ns);
            submit_ns.push(timing.submit_ns);
            dispatches.push(timing.dispatches);
            tokens.push(sampled);
            next = sampled;
        }
        let decode_wall_ns = decode.elapsed().as_nanos() as u64;
        let decode_steps = tokens.len().saturating_sub(prompt.len()).saturating_sub(1);
        Ok(Qwen38GenerateResult {
            tokens,
            prompt_len: prompt.len(),
            wall_ns: wall.elapsed().as_nanos() as u64,
            gpu_ns,
            wait_ns,
            encode_ns,
            submit_ns,
            dispatches,
            fallbacks: session.fallbacks,
            first_step_wall_ns,
            prefill_wall_ns,
            decode_wall_ns,
            decode_steps,
            wall_ns_per_step,
        })
    }

    pub fn generate_greedy_complete_wall(
        session: &mut Qwen38HybridDecodeSession,
        tokenizer: &Tokenizer,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> Result<Qwen38CompleteWallResult> {
        if prompt.is_empty() {
            return Err(Error::Model("qwen38 prompt is empty".into()));
        }
        let reset_started = Instant::now();
        session.reset();
        let reset_ns = reset_started.elapsed().as_nanos() as u64;
        let mut tokens = prompt.to_vec();
        let mut steps = Vec::new();
        let wall = Instant::now();
        let mut next = 0u32;
        let prefill = Instant::now();
        for (i, &token) in prompt.iter().enumerate() {
            let complete = Instant::now();
            let (sampled, step) = session.step_complete(token)?;
            let last_prompt = i + 1 == prompt.len();
            let (tokenizer_decode_ns, bookkeeping_ns) = if last_prompt {
                finish_new_token(tokenizer, &mut tokens, sampled)?
            } else {
                next = sampled;
                (0, 0)
            };
            if last_prompt {
                next = sampled;
            }
            steps.push(Qwen38CompleteToken {
                role: if last_prompt {
                    "prefill_emits_first_new"
                } else {
                    "prefill"
                }
                .to_owned(),
                step_index: i,
                token_in: token,
                token_out: sampled,
                step,
                tokenizer_decode_ns,
                bookkeeping_ns,
                complete_wall_ns: complete.elapsed().as_nanos() as u64,
            });
        }
        let prefill_wall_ns = prefill.elapsed().as_nanos() as u64;
        let decode = Instant::now();
        while tokens.len() - prompt.len() < max_new_tokens {
            if next == crate::model::qwen38_geometry::QWEN38_EOS_IM_END
                || next == crate::model::qwen38_geometry::QWEN38_EOS_END_OF_TEXT
            {
                break;
            }
            let complete = Instant::now();
            let (sampled, step) = session.step_complete(next)?;
            let (tokenizer_decode_ns, bookkeeping_ns) =
                finish_new_token(tokenizer, &mut tokens, sampled)?;
            steps.push(Qwen38CompleteToken {
                role: "decode".to_owned(),
                step_index: steps.len(),
                token_in: next,
                token_out: sampled,
                step,
                tokenizer_decode_ns,
                bookkeeping_ns,
                complete_wall_ns: complete.elapsed().as_nanos() as u64,
            });
            next = sampled;
        }
        let decode_wall_ns = decode.elapsed().as_nanos() as u64;
        Ok(Qwen38CompleteWallResult {
            tokens,
            prompt_len: prompt.len(),
            wall_ns: wall.elapsed().as_nanos() as u64,
            reset_ns,
            prefill_wall_ns,
            decode_wall_ns,
            fallbacks: session.fallbacks,
            steps,
        })
    }

    fn finish_new_token(
        tokenizer: &Tokenizer,
        tokens: &mut Vec<u32>,
        sampled: u32,
    ) -> Result<(u64, u64)> {
        let tokenizer_started = Instant::now();
        tokenizer.decode(&[sampled], true)?;
        let tokenizer_decode_ns = tokenizer_started.elapsed().as_nanos() as u64;
        let bookkeeping_started = Instant::now();
        tokens.push(sampled);
        let bookkeeping_ns = bookkeeping_started.elapsed().as_nanos() as u64;
        Ok((tokenizer_decode_ns, bookkeeping_ns))
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Qwen38StepWall {
    pub wall_ns: u64,
    pub encode_ns: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub gpu_ns: Option<u64>,
    /// `commit_and_wait_timed` return minus submit minus wait: GPU timestamp
    /// read + command-buffer status check after the host wait returns.
    pub commit_epilogue_ns: u64,
    pub sample_readback_ns: u64,
    pub state_update_ns: u64,
    /// TCB per-dispatch encode sum. Zero unless the cost ledger is recording.
    pub tcb_encode_ns: u64,
    pub dispatches: u64,
    pub command_buffers: u64,
}

impl Qwen38StepWall {
    pub fn named_sum_ns(&self) -> u64 {
        self.encode_ns
            .saturating_add(self.submit_ns)
            .saturating_add(self.wait_ns)
            .saturating_add(self.commit_epilogue_ns)
            .saturating_add(self.sample_readback_ns)
            .saturating_add(self.state_update_ns)
    }

    pub fn residual_ns(&self) -> i64 {
        self.wall_ns as i64 - self.named_sum_ns() as i64
    }

    pub fn wait_minus_gpu_ns(&self) -> Option<i64> {
        Some(self.wait_ns as i64 - self.gpu_ns? as i64)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen38CompleteToken {
    pub role: String,
    pub step_index: usize,
    pub token_in: u32,
    pub token_out: u32,
    pub step: Qwen38StepWall,
    pub tokenizer_decode_ns: u64,
    pub bookkeeping_ns: u64,
    pub complete_wall_ns: u64,
}

impl Qwen38CompleteToken {
    pub fn named_sum_ns(&self) -> u64 {
        self.step
            .named_sum_ns()
            .saturating_add(self.tokenizer_decode_ns)
            .saturating_add(self.bookkeeping_ns)
    }

    pub fn residual_ns(&self) -> i64 {
        self.complete_wall_ns as i64 - self.named_sum_ns() as i64
    }

    pub fn wall_minus_gpu_ns(&self) -> Option<i64> {
        Some(self.complete_wall_ns as i64 - self.step.gpu_ns? as i64)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen38CompleteWallResult {
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub wall_ns: u64,
    pub reset_ns: u64,
    pub prefill_wall_ns: u64,
    pub decode_wall_ns: u64,
    pub fallbacks: u32,
    pub steps: Vec<Qwen38CompleteToken>,
}

impl Qwen38CompleteWallResult {
    pub fn new_tokens(&self) -> &[u32] {
        &self.tokens[self.prompt_len.min(self.tokens.len())..]
    }

    pub fn decode_new(&self, tokenizer: &Tokenizer) -> Result<String> {
        tokenizer.decode(self.new_tokens(), true)
    }

    pub fn first_step(&self) -> Option<&Qwen38CompleteToken> {
        self.steps.first()
    }

    /// Prompt walk, including the last prompt step that emits new-token[0].
    pub fn prefill_steps(&self) -> impl Iterator<Item = &Qwen38CompleteToken> {
        self.steps.iter().filter(|s| s.role != "decode")
    }

    /// New-tokens[1..]: Q80 mixed `steady_state` denominator.
    pub fn steady_decode_steps(&self) -> impl Iterator<Item = &Qwen38CompleteToken> {
        self.steps.iter().filter(|s| s.role == "decode")
    }
}

#[derive(Clone, Debug)]
pub struct Qwen38GenerateResult {
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub wall_ns: u64,
    pub gpu_ns: Vec<Option<u64>>,
    pub wait_ns: Vec<u64>,
    pub encode_ns: Vec<u64>,
    pub submit_ns: Vec<u64>,
    pub dispatches: Vec<u64>,
    pub fallbacks: u32,
    pub first_step_wall_ns: u64,
    pub prefill_wall_ns: u64,
    pub decode_wall_ns: u64,
    pub decode_steps: usize,
    pub wall_ns_per_step: Vec<u64>,
}

impl Qwen38GenerateResult {
    pub fn new_tokens(&self) -> &[u32] {
        &self.tokens[self.prompt_len.min(self.tokens.len())..]
    }

    pub fn decode_new(&self, tokenizer: &Tokenizer) -> Result<String> {
        tokenizer.decode(self.new_tokens(), true)
    }

    pub fn median_gpu_ns_per_token(&self) -> Option<u64> {
        let mut values: Vec<u64> = self.gpu_ns.iter().copied().flatten().collect();
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        Some(values[values.len() / 2])
    }

    pub fn steady_decode_wall_ns_per_token(&self) -> Option<u64> {
        if self.decode_steps == 0 {
            return None;
        }
        Some(self.decode_wall_ns / self.decode_steps as u64)
    }
}

#[cfg(target_os = "macos")]
pub use device::{generate_greedy, generate_greedy_complete_wall, Qwen38HybridDecodeSession};

#[cfg(not(target_os = "macos"))]
pub fn generate_greedy(
    _root: impl AsRef<Path>,
    _prompt: &[u32],
    _max_new: usize,
) -> Result<Qwen38GenerateResult> {
    Err(Error::Model("qwen38 native decode is Metal-only".into()))
}

#[cfg(not(target_os = "macos"))]
pub fn generate_greedy_complete_wall(
    _root: impl AsRef<Path>,
    _tokenizer: &Tokenizer,
    _prompt: &[u32],
    _max_new: usize,
) -> Result<Qwen38CompleteWallResult> {
    Err(Error::Model("qwen38 native decode is Metal-only".into()))
}

#[cfg(test)]
mod complete_wall_identity_tests {
    use super::{Qwen38CompleteToken, Qwen38StepWall};

    #[test]
    fn step_named_sum_plus_residual_equals_wall() {
        let step = Qwen38StepWall {
            wall_ns: 34_000_000,
            encode_ns: 400_000,
            submit_ns: 20_000,
            wait_ns: 33_500_000,
            gpu_ns: Some(33_100_000),
            commit_epilogue_ns: 30_000,
            sample_readback_ns: 2_000,
            state_update_ns: 1_000,
            tcb_encode_ns: 0,
            dispatches: 900,
            command_buffers: 1,
        };
        assert_eq!(
            step.named_sum_ns() as i64 + step.residual_ns(),
            step.wall_ns as i64
        );
        assert_eq!(step.wait_minus_gpu_ns(), Some(400_000));
    }

    #[test]
    fn complete_token_names_tokenizer_and_bookkeeping() {
        let token = Qwen38CompleteToken {
            role: "decode".into(),
            step_index: 12,
            token_in: 1,
            token_out: 2,
            step: Qwen38StepWall {
                wall_ns: 33_953_000,
                encode_ns: 400_000,
                submit_ns: 20_000,
                wait_ns: 33_500_000,
                gpu_ns: Some(33_100_000),
                commit_epilogue_ns: 30_000,
                sample_readback_ns: 2_000,
                state_update_ns: 1_000,
                tcb_encode_ns: 0,
                dispatches: 900,
                command_buffers: 1,
            },
            tokenizer_decode_ns: 8_000,
            bookkeeping_ns: 1_000,
            complete_wall_ns: 33_970_000,
        };
        assert_eq!(
            token.named_sum_ns() as i64 + token.residual_ns(),
            token.complete_wall_ns as i64
        );
        assert_eq!(token.wall_minus_gpu_ns(), Some(870_000));
    }
}
