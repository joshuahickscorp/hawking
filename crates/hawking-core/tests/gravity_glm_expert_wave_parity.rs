#![cfg(target_os = "macos")]
use hawking_core::gravity_glm::{
    estimate_batched_mlp_drains_per_token, estimate_resident_expert_wave_waits_per_token,
    estimate_resident_waits_per_token, gpu_expert_wave_enabled, GlmArch, GPU_EXPERT_WAVE_ENV,
};
use hawking_core::metal::{MetalContext, TokenCommandBuffer};
use hawking_core::numeric_parity::{
    format_score_line, score_pair, silu_mul_f32_host, silu_mul_f64_authority, Bounds, SCHEMA,
};
fn gate_up_pairs(n: usize) -> Vec<(&'static str, Vec<f32>, Vec<f32>)> {
    let gate = |f: fn(usize) -> f32| -> Vec<f32> { (0..n).map(f).collect() };
    vec![
        (
            "ramp",
            gate(|i| (i as f32) * 0.02 - 0.5),
            gate(|i| (i as f32) * 0.01 - 0.25),
        ),
        (
            "mod17",
            gate(|i| ((i * 5 + 3) % 17) as f32 * 0.05 - 0.4),
            gate(|i| ((i * 3 + 1) % 17) as f32 * 0.04 - 0.3),
        ),
        ("ones", vec![1.0; n], vec![1.0; n]),
        ("zeros", vec![0.0; n], vec![0.0; n]),
        (
            "sign_flip",
            gate(|i| if i % 2 == 0 { 0.75 } else { -0.75 }),
            gate(|i| if i % 3 == 0 { 0.5 } else { -0.25 }),
        ),
        (
            "sin",
            gate(|i| (i as f32).sin() * 0.5),
            gate(|i| (i as f32 * 0.7).cos() * 0.4),
        ),
        (
            "large",
            gate(|i| if i % 2 == 0 { 8.0 } else { -8.0 }),
            gate(|i| 0.1 * (i as f32 % 5.0 - 2.0)),
        ),
        (
            "near_zero",
            gate(|i| (i as f32) * 1e-4 - 5e-3),
            gate(|i| 1.0 + (i as f32) * 0.001),
        ),
    ]
}
fn dispatch_silu_device(ctx: &MetalContext, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
    assert_eq!(gate.len(), up.len());
    let n = gate.len() as u32;
    let gate_buf = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(gate))
        .map_err(|e| e.to_string())?;
    let up_buf = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(up))
        .map_err(|e| e.to_string())?;
    let out_buf = ctx
        .new_buffer_checked(gate.len() * 4)
        .map_err(|e| e.to_string())?;
    const TG: u32 = 256;
    let mut tcb = TokenCommandBuffer::new(ctx);
    tcb.dispatch_threads(
        "gravity_silu_mul_f32",
        (n.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        |enc| {
            enc.set_buffer(0, Some(&gate_buf), 0);
            enc.set_buffer(1, Some(&up_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
        },
    )
    .map_err(|e| e.to_string())?;
    tcb.commit_and_wait().map_err(|e| e.to_string())?;
    let ptr = out_buf.contents() as *const f32;
    let out = unsafe { std::slice::from_raw_parts(ptr, gate.len()) }.to_vec();
    Ok(out)
}
#[test]
fn device_silu_mul_v21_against_f64_over_several_vectors() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("shader") && !msg.contains("compile"),
                "Metal is present but the shader failed to compile -- this is a real \
                 failure, not a skip: {msg}"
            );
            eprintln!("skip: no Metal device ({e})");
            return;
        }
    };
    let bounds = Bounds::continuous_only();
    let lengths = [16usize, 64, 257, 2048];
    let mut any_fail = false;
    for &n in &lengths {
        for (name, gate, up) in gate_up_pairs(n) {
            let g64: Vec<f64> = gate.iter().map(|&v| v as f64).collect();
            let u64: Vec<f64> = up.iter().map(|&v| v as f64).collect();
            let reference = silu_mul_f64_authority(&g64, &u64).expect("f64 authority");
            let host = silu_mul_f32_host(&gate, &up).expect("host silu");
            let device = dispatch_silu_device(&ctx, &gate, &up).unwrap_or_else(|e| {
                panic!("device silu failed (n={n} vec={name}): {e} — shader compile issues are hard fails");
            });
            assert_eq!(device.len(), host.len());
            assert_eq!(device.len(), reference.len());
            let paired = score_pair(&host, &device, &reference, &bounds);
            if !paired.pass {
                any_fail = true;
            }
        }
    }
    assert!(
        !any_fail,
        "device silu failed V2.1 continuous gates against FP64 authority"
    );
}
#[test]
fn expert_wave_static_drains_flagship() {
    let raw = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gravity_glm/flagship_arch.json"),
    )
    .expect("flagship_arch.json");
    let header: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let arch = GlmArch::from_header(&header).unwrap();
    assert_eq!(estimate_batched_mlp_drains_per_token(&arch, false), 234);
    assert_eq!(estimate_batched_mlp_drains_per_token(&arch, true), 78);
    assert_eq!(estimate_resident_waits_per_token(&arch), 586);
    assert_eq!(estimate_resident_expert_wave_waits_per_token(&arch), 430);
    let prev = std::env::var_os(GPU_EXPERT_WAVE_ENV);
    std::env::remove_var(GPU_EXPERT_WAVE_ENV);
    assert!(!gpu_expert_wave_enabled());
    match prev {
        Some(v) => std::env::set_var(GPU_EXPERT_WAVE_ENV, v),
        None => std::env::remove_var(GPU_EXPERT_WAVE_ENV),
    }
}
#[test]
fn default_path_never_enables_expert_wave_without_flag() {
    let prev = std::env::var_os(GPU_EXPERT_WAVE_ENV);
    std::env::remove_var(GPU_EXPERT_WAVE_ENV);
    assert!(
        !gpu_expert_wave_enabled(),
        "default resident path must not take the expert-wave branch"
    );
    std::env::set_var(GPU_EXPERT_WAVE_ENV, "0");
    assert!(!gpu_expert_wave_enabled());
    std::env::set_var(GPU_EXPERT_WAVE_ENV, "");
    assert!(!gpu_expert_wave_enabled());
    match prev {
        Some(v) => std::env::set_var(GPU_EXPERT_WAVE_ENV, v),
        None => std::env::remove_var(GPU_EXPERT_WAVE_ENV),
    }
}
