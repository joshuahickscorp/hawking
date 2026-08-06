//! Smoke-compile the DeepSeek-V4 production Metal kernels used by the
//! general per-layer path. No artifact, no forward, no parity claim.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_deepseek_v4_metal_kernel_smoke requires macOS Metal");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let metal = hawking_core::metal::MetalContext::new()?;
    let kernels = [
        "deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority",
        "deepseek_v4_p7_mhc_ffn_pre_authority",
        "deepseek_v4_p7_ffn_rmsnorm_bf16_authority",
        "deepseek_v4_p7_mhc_ffn_post_authority",
        "deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority",
        "deepseek_v4_p4b_sparse_attention_position1_two_kv_sink_authority",
        "deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate",
        "deepseek_v4_p6a_hash_route_sqrtsoftplus_authority",
    ];
    for kernel in kernels {
        metal.pipeline(kernel)?;
        println!("pipeline_ok {kernel}");
    }
    println!("status PASS_METAL_KERNEL_SMOKE_NOT_FORWARD");
    Ok(())
}
