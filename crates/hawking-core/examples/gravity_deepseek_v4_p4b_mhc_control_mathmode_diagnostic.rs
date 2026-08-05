//! Isolated P4B mHC-control numerical-semantics probe.
//!
//! This is intentionally not an authority path, decoder, runtime, or receipt
//! producer. It uploads the independent CPU oracle's exact P1 `mixes` plus
//! static source scale/base controls into small Metal probes, then compares:
//! CPU Rust `f32::exp`, default Metal fast-math `fast::exp`, default-library
//! `precise::exp`, and a separately compiled `fastMathEnabled(false)` precise
//! candidate. It never uploads a hidden state into the P4B graph, mutates the
//! sealed authority kernel, or claims terminal parity.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "gravity_deepseek_v4_p4b_mhc_control_mathmode_diagnostic requires macOS Metal",
    )
    .into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
    use hawking_core::gravity_deepseek_v4_layer0_continuation::layer0_position1_complete_attention_cpu_oracle;
    use hawking_core::gravity_deepseek_v4_layer0_prefix::{
        HC_EPS, HC_MULT, HC_SINKHORN_ITERS, LAYER0_HC_ATTN_BASE, LAYER0_HC_ATTN_SCALE,
    };
    use metal::{
        CompileOptions, Device, Library, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
    };
    use serde_json::{json, Value};
    use std::error::Error;
    use std::path::PathBuf;

    const FAST_TRACE_KERNEL: &str = "deepseek_v4_p4b_hc_control_fast_exp_trace_candidate";
    const PRECISE_TRACE_KERNEL: &str = "deepseek_v4_p4b_hc_control_precise_exp_trace_candidate";
    const PRECISE_CONTROL_KERNEL: &str = "deepseek_v4_p4b_hc_post_comb_precise_exp_candidate";
    const POST_ULP_REPAIR_KERNEL: &str =
        "deepseek_v4_p4b_hc_post_cpu_exp_ulp_repair_trace_candidate";
    const TRACE_FLOATS: usize = 40;

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Runner {
        device: Device,
        queue: metal::CommandQueue,
        #[allow(dead_code)]
        library: Library,
        fast_math_enabled: bool,
    }

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }

    impl Runner {
        fn new(fast_math_enabled: bool) -> ProbeResult<Self> {
            let device = Device::system_default().ok_or_else(|| failure("no Metal device"))?;
            let options = CompileOptions::new();
            options.set_fast_math_enabled(fast_math_enabled);
            let library = device
                .new_library_with_source(hawking_core::metal::SHADER_MATMUL, &options)
                .map_err(failure)?;
            Ok(Self {
                queue: device.new_command_queue(),
                device,
                library,
                fast_math_enabled,
            })
        }

        fn trace(
            &self,
            kernel: &str,
            mixes: &[f32],
            scale: &[f32],
            base: &[f32],
        ) -> ProbeResult<Vec<f32>> {
            let function = self.library.get_function(kernel, None).map_err(failure)?;
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(failure)?;
            let mixes = self.f32_buffer(mixes);
            let scale = self.f32_buffer(scale);
            let base = self.f32_buffer(base);
            let output = self.zeroed_f32_buffer(TRACE_FLOATS);
            let command = self.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&mixes), 0);
            encoder.set_buffer(1, Some(&scale), 0);
            encoder.set_buffer(2, Some(&base), 0);
            encoder.set_buffer(3, Some(&output), 0);
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            self.check_completed(command.status(), kernel)?;
            read_f32(&output, TRACE_FLOATS)
        }

        fn precise_control(
            &self,
            mixes: &[f32],
            scale: &[f32],
            base: &[f32],
        ) -> ProbeResult<(Vec<f32>, Vec<f32>)> {
            let function = self
                .library
                .get_function(PRECISE_CONTROL_KERNEL, None)
                .map_err(failure)?;
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(failure)?;
            let mixes = self.f32_buffer(mixes);
            let scale = self.f32_buffer(scale);
            let base = self.f32_buffer(base);
            let post = self.zeroed_f32_buffer(HC_MULT);
            let comb = self.zeroed_f32_buffer(HC_MULT * HC_MULT);
            let hc_mult = HC_MULT as u32;
            let mix_width = 24u32;
            let sinkhorn_iters = HC_SINKHORN_ITERS as u32;
            let eps = HC_EPS;
            let command = self.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&mixes), 0);
            encoder.set_buffer(1, Some(&scale), 0);
            encoder.set_buffer(2, Some(&base), 0);
            encoder.set_buffer(3, Some(&post), 0);
            encoder.set_buffer(4, Some(&comb), 0);
            set_u32(encoder, 5, &hc_mult);
            set_u32(encoder, 6, &mix_width);
            set_u32(encoder, 7, &sinkhorn_iters);
            set_f32(encoder, 8, &eps);
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            self.check_completed(command.status(), PRECISE_CONTROL_KERNEL)?;
            Ok((
                read_f32(&post, HC_MULT)?,
                read_f32(&comb, HC_MULT * HC_MULT)?,
            ))
        }

        fn trace_bound_post_ulp_repair(
            &self,
            mixes: &[f32],
            scale: &[f32],
            base: &[f32],
            post_values: &mut [f32],
        ) -> ProbeResult<()> {
            if post_values.len() != HC_MULT {
                return Err(failure("P4B trace repair post geometry"));
            }
            let function = self
                .library
                .get_function(POST_ULP_REPAIR_KERNEL, None)
                .map_err(failure)?;
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(failure)?;
            let mixes = self.f32_buffer(mixes);
            let scale = self.f32_buffer(scale);
            let base = self.f32_buffer(base);
            let post = self.f32_buffer(post_values);
            let command = self.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&mixes), 0);
            encoder.set_buffer(1, Some(&scale), 0);
            encoder.set_buffer(2, Some(&base), 0);
            encoder.set_buffer(3, Some(&post), 0);
            encoder.dispatch_threads(
                MTLSize::new(HC_MULT as u64, 1, 1),
                MTLSize::new(HC_MULT as u64, 1, 1),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            self.check_completed(command.status(), POST_ULP_REPAIR_KERNEL)?;
            post_values.copy_from_slice(&read_f32(&post, HC_MULT)?);
            Ok(())
        }

        fn f32_buffer(&self, values: &[f32]) -> metal::Buffer {
            let buffer = self.device.new_buffer(
                (values.len() * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr() as *const u8,
                    buffer.contents() as *mut u8,
                    values.len() * std::mem::size_of::<f32>(),
                );
            }
            buffer
        }

        fn zeroed_f32_buffer(&self, count: usize) -> metal::Buffer {
            self.f32_buffer(&vec![0.0; count])
        }

        fn check_completed(&self, status: MTLCommandBufferStatus, kernel: &str) -> ProbeResult<()> {
            if status != MTLCommandBufferStatus::Completed {
                return Err(failure(format!(
                    "{kernel} did not complete in fast_math_enabled={}: {status:?}",
                    self.fast_math_enabled
                )));
            }
            Ok(())
        }
    }

    pub fn run() -> ProbeResult<()> {
        let artifact = parse_artifact()?;
        let reader = DeepSeekV4FullStreamReader::admit(&artifact)?;
        let cpu = layer0_position1_complete_attention_cpu_oracle(&reader)?;
        let scale = f32_tensor(&reader, LAYER0_HC_ATTN_SCALE)?;
        let base = f32_tensor(&reader, LAYER0_HC_ATTN_BASE)?;
        let mixes = &cpu.causal.token1_prefix.hc_mixes_f32;
        if mixes.len() != 24 || scale.len() != 3 || base.len() != 24 {
            return Err(failure("P4B control source geometry changed"));
        }
        let cpu_trace = cpu_rust_trace(mixes, &scale, &base);

        let fast = Runner::new(true)?;
        let strict = Runner::new(false)?;
        let fast_exp_trace = fast.trace(FAST_TRACE_KERNEL, mixes, &scale, &base)?;
        let default_precise_trace = fast.trace(PRECISE_TRACE_KERNEL, mixes, &scale, &base)?;
        let strict_precise_trace = strict.trace(PRECISE_TRACE_KERNEL, mixes, &scale, &base)?;
        let (default_post, default_comb) = fast.precise_control(mixes, &scale, &base)?;
        let (strict_post, strict_comb) = strict.precise_control(mixes, &scale, &base)?;
        let mut strict_repaired_post = strict_post.clone();
        strict.trace_bound_post_ulp_repair(mixes, &scale, &base, &mut strict_repaired_post)?;

        let cpu_controls = [
            &cpu.causal.token1_prefix.hc_post_f32[..],
            &cpu.causal.token1_prefix.hc_comb_f32[..],
        ]
        .concat();
        let default_controls = [&default_post[..], &default_comb[..]].concat();
        let strict_controls = [&strict_post[..], &strict_comb[..]].concat();
        let strict_repaired_controls = [&strict_repaired_post[..], &strict_comb[..]].concat();
        let strict_controls_exact = bitwise_equal(&cpu_controls, &strict_repaired_controls);
        let report = json!({
            "schema":"hawking.gravity.deepseek_v4.p4b_mhc_control_mathmode_diagnostic.v1",
            "status":if strict_controls_exact {"STRICT_MATH_CONTROL_VALUES_EXACT_NOT_PROMOTED"} else {"STRICT_MATH_CONTROL_VALUES_NOT_EXACT_NOT_PROMOTED"},
            "artifact_manifest_seal_sha256":reader.manifest_seal_sha256(),
            "cpu_oracle":"independent Rust source-algorithm P1 controls",
            "inputs":{"mixes_f32_bits":f32_words(mixes),"scale_f32_bits":f32_words(&scale),"base_f32_bits":f32_words(&base)},
            "cpu_rust_exp_trace":{"layout":"post_logits[0..4), post_exp[4..8), comb_logits[8..24), comb_exp_after_row_max[24..40)","f32_bits":f32_words(&cpu_trace)},
            "cpu_controls":{"post_f32_bits":f32_words(&cpu.causal.token1_prefix.hc_post_f32),"comb_f32_bits":f32_words(&cpu.causal.token1_prefix.hc_comb_f32)},
            "default_fast_math_enabled":{
                "fast_exp_trace_delta":trace_delta(&cpu_trace, &fast_exp_trace),
                "precise_exp_trace_delta":trace_delta(&cpu_trace, &default_precise_trace),
                "precise_control_delta":control_delta(&cpu_controls, &default_controls),
                "precise_post_f32_bits":f32_words(&default_post),
                "precise_comb_f32_bits":f32_words(&default_comb),
            },
            "strict_fast_math_disabled":{
                "precise_exp_trace_delta":trace_delta(&cpu_trace, &strict_precise_trace),
                "precise_control_delta":control_delta(&cpu_controls, &strict_controls),
                "precise_post_f32_bits":f32_words(&strict_post),
                "precise_comb_f32_bits":f32_words(&strict_comb),
                "trace_bound_ulp_repair_candidate":{
                    "kernel":POST_ULP_REPAIR_KERNEL,
                    "post_f32_bits":f32_words(&strict_repaired_post),
                    "control_delta":control_delta(&cpu_controls, &strict_repaired_controls),
                    "all_controls_exact":strict_controls_exact,
                },
            },
            "smallest_candidate_if_exact":"a separate fastMathEnabled(false) Metal library/pipeline for the existing one-thread 4-lane P4B mHC post+comb control kernel, followed by its trace-bound two-logit ULP repair microkernel; both bind device-resident exact mixes. Terminal BF16 remains untested here and promotion is prohibited.",
            "receipt_written":false,
            "promotion":false,
        });
        println!("{}", serde_json::to_string(&report)?);
        Ok(())
    }

    fn parse_artifact() -> ProbeResult<PathBuf> {
        let mut artifact = None;
        let mut args = std::env::args_os().skip(1);
        while let Some(flag) = args.next() {
            match flag.to_string_lossy().as_ref() {
                "--artifact" => artifact = args.next().map(PathBuf::from),
                other => return Err(failure(format!("unknown argument {other}"))),
            }
        }
        artifact.ok_or_else(|| failure("--artifact is required"))
    }

    fn f32_tensor(reader: &DeepSeekV4FullStreamReader, name: &str) -> ProbeResult<Vec<f32>> {
        let metadata = reader.tensor_metadata(name)?;
        if metadata.dtype != "F32" || metadata.bytes % 4 != 0 {
            return Err(failure(format!("{name} must remain F32")));
        }
        let bytes = reader.read_verified_full(name, metadata.bytes as usize)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect())
    }

    fn cpu_rust_trace(mixes: &[f32], scale: &[f32], base: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; TRACE_FLOATS];
        let mut comb = [0.0; HC_MULT * HC_MULT];
        for lane in 0..HC_MULT {
            let value = mixes[lane + HC_MULT] * scale[1] + base[lane + HC_MULT];
            out[lane] = value;
            out[HC_MULT + lane] = (-value).exp();
        }
        for row in 0..HC_MULT {
            for column in 0..HC_MULT {
                let index = row * HC_MULT + column;
                let source_index = index + 2 * HC_MULT;
                let value = mixes[source_index] * scale[2] + base[source_index];
                comb[index] = value;
                out[2 * HC_MULT + index] = value;
            }
        }
        for row in 0..HC_MULT {
            let start = row * HC_MULT;
            let mut maximum = comb[start];
            for column in 1..HC_MULT {
                maximum = maximum.max(comb[start + column]);
            }
            for column in 0..HC_MULT {
                let index = start + column;
                out[6 * HC_MULT + index] = (comb[index] - maximum).exp();
            }
        }
        out
    }

    fn trace_delta(expected: &[f32], actual: &[f32]) -> Value {
        json!({
            "post_logits":f32_delta(&expected[0..4], &actual[0..4]),
            "post_exp":f32_delta(&expected[4..8], &actual[4..8]),
            "comb_logits":f32_delta(&expected[8..24], &actual[8..24]),
            "comb_exp_after_row_max":f32_delta(&expected[24..40], &actual[24..40]),
        })
    }

    fn control_delta(expected: &[f32], actual: &[f32]) -> Value {
        json!({
            "post":f32_delta(&expected[0..HC_MULT], &actual[0..HC_MULT]),
            "comb":f32_delta(&expected[HC_MULT..], &actual[HC_MULT..]),
        })
    }

    fn f32_delta(expected: &[f32], actual: &[f32]) -> Value {
        assert_eq!(expected.len(), actual.len());
        let mut mismatch_count = 0usize;
        let mut first = None;
        let mut max_absolute_delta = 0.0f32;
        for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            if expected.to_bits() != actual.to_bits() {
                mismatch_count += 1;
                if first.is_none() {
                    first = Some(json!({
                        "index":index,
                        "expected":"0x".to_owned()+&format!("{:08x}",expected.to_bits()),
                        "actual":"0x".to_owned()+&format!("{:08x}",actual.to_bits()),
                        "expected_value":expected.to_string(),
                        "actual_value":actual.to_string(),
                    }));
                }
            }
            max_absolute_delta = max_absolute_delta.max((expected - actual).abs());
        }
        json!({"elements":expected.len(),"bitwise_mismatch_count":mismatch_count,"max_absolute_delta":max_absolute_delta,"first_mismatch":first})
    }

    fn bitwise_equal(expected: &[f32], actual: &[f32]) -> bool {
        expected.len() == actual.len()
            && expected
                .iter()
                .zip(actual)
                .all(|(expected, actual)| expected.to_bits() == actual.to_bits())
    }

    fn f32_words(values: &[f32]) -> Vec<String> {
        values
            .iter()
            .map(|value| format!("0x{:08x}", value.to_bits()))
            .collect()
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            value as *const u32 as *const _,
        );
    }

    fn set_f32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &f32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<f32>() as u64,
            value as *const f32 as *const _,
        );
    }

    fn read_f32(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<f32>> {
        if buffer.length() < (count * std::mem::size_of::<f32>()) as u64 {
            return Err(failure("Metal output read overflow"));
        }
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, count).to_vec() })
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
