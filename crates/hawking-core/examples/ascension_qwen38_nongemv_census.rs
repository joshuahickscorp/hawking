//! G110: the non-GEMV occupancy census -- grid, TG, occupancy and REGISTER PRESSURE.
//!
//! Time attribution for these kernels was already closed at 98.8%. What the
//! acceptance still owes is the structural half: for each kernel, how many threads
//! the hardware will actually let a threadgroup carry, and whether that limit is
//! register pressure rather than choice.
//!
//! maxTotalThreadsPerThreadgroup is the measurement. Metal caps a threadgroup at
//! 1024 threads; a pipeline reporting less than that is REGISTER-LIMITED, and the
//! shortfall is the register pressure the acceptance calls load-bearing. It is
//! already known to be: the R x K tiling regressed at R=16, and G109 measured
//! small organs getting worse as R rose, which is the same wall seen from outside.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(target_os = "macos"))]
    {
        Err(std::io::Error::other("requires macOS Metal").into())
    }
    #[cfg(target_os = "macos")]
    {
        use hawking_core::metal::MetalContext;
        use serde_json::json;
        use std::{env, fs, path::PathBuf};

        // (kernel, threadgroup size the decode path actually dispatches, family)
        const KERNELS: [(&str, u32, &str); 20] = [
            (
                "qwen38_gated_delta_decode_vi_simd",
                128,
                "NON_GEMV recurrence",
            ),
            (
                "qwen38_gated_delta_decode_vi",
                128,
                "NON_GEMV recurrence (superseded)",
            ),
            ("qwen80_deltanet_gated_rmsnorm_tg", 256, "NON_GEMV norm"),
            ("qwen80_deltanet_gated_rmsnorm_f32", 64, "NON_GEMV norm"),
            ("qwen80_residual_rmsnorm_tg", 256, "NON_GEMV norm"),
            ("qwen80_residual_rmsnorm_f32", 64, "NON_GEMV norm"),
            ("qwen38_gqa_qk_norm_rope_cache_tg", 256, "NON_GEMV rope"),
            ("qwen38_gqa_qk_norm_rope_cache_f32", 64, "NON_GEMV rope"),
            (
                "qwen38_qkvz_rearrange_conv_l2_f32",
                64,
                "NON_GEMV conv/rearrange",
            ),
            ("qwen38_attention_apply_sigmoid_gate", 64, "NON_GEMV gate"),
            ("qwen80_ba_to_decay_beta_f32", 64, "NON_GEMV decay"),
            ("sample_argmax_f32_pass1", 256, "NON_GEMV sampling"),
            ("sample_argmax_f32_pass2", 256, "NON_GEMV sampling"),
            (
                "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                128,
                "GEMV reference",
            ),
            // The acceptance states register pressure is load-bearing because "the R x K tiling
            // spilled at R=8". These are the kernels that claim is about, so they are measured
            // rather than taken on trust.
            (
                "qwen_uniform_q4_group64_matmul_r4k1_geo_tpr64_tg128",
                128,
                "GEMV tiling R4",
            ),
            (
                "qwen_uniform_q4_group64_matmul_r8k1_geo_tpr64_tg128",
                128,
                "GEMV tiling R8",
            ),
            (
                "qwen_uniform_q4_group64_matmul_r16k1_geo_tpr64_tg128",
                128,
                "GEMV tiling R16",
            ),
            (
                "qwen_uniform_q4_group64_matmul_r4k4_geo_tpr64_tg128",
                128,
                "GEMV tiling R4K4",
            ),
            (
                "qwen_uniform_q4_group64_matmul_r8k8_geo_tpr64_tg128",
                128,
                "GEMV tiling R8K8",
            ),
            (
                "qwen_uniform_q4_group64_matmul_r16k4_geo_tpr64_tg128",
                128,
                "GEMV tiling R16K4",
            ),
        ];

        let mut out_path: Option<PathBuf> = None;
        let mut it = env::args().skip(1);
        while let Some(a) = it.next() {
            if a == "--out" {
                out_path = it.next().map(PathBuf::from);
            }
        }
        let ctx = MetalContext::new()?;
        let mut rows = Vec::new();
        println!(
            "{:<50}{:>6}{:>8}{:>7}{:>8}  {}",
            "kernel", "TG", "maxTPT", "simd", "occ", "limited?"
        );
        for (name, tg, family) in KERNELS {
            let p = match ctx.pipeline(name) {
                Ok(p) => p,
                Err(e) => {
                    println!("{name:<50}  UNAVAILABLE: {e}");
                    continue;
                }
            };
            let maxt = p.max_total_threads_per_threadgroup() as u32;
            let simd = p.thread_execution_width() as u32;
            let occ = tg as f64 / maxt as f64;
            let limited = maxt < 1024;
            println!(
                "{name:<50}{tg:>6}{maxt:>8}{simd:>7}{occ:>7.2}  {}",
                if limited {
                    format!("REGISTER-LIMITED to {maxt}")
                } else {
                    "no".into()
                }
            );
            rows.push(json!({
                "kernel": name, "family": family, "threadgroup_dispatched": tg,
                "max_total_threads_per_threadgroup": maxt, "thread_execution_width": simd,
                "occupancy_of_its_own_limit": occ,
                "register_limited": limited,
                "headroom_threads": maxt.saturating_sub(tg),
            }));
        }
        let commit = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        let doc = json!({
            "schema": "hawking.nos.nongemv_occupancy_census.v1",
            "obligation": "G110 -- per-kernel grid, TG, occupancy and register pressure",
            "what_max_total_threads_measures":
                "Metal caps a threadgroup at 1024 threads. A pipeline reporting less is \
                 REGISTER-LIMITED, and the shortfall is the register pressure this acceptance \
                 calls load-bearing.",
            "kernels": rows, "commit": commit,
        });
        if let Some(p) = out_path {
            fs::write(p, serde_json::to_string_pretty(&doc)? + "\n")?;
        }
        Ok(())
    }
}
