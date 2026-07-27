//! Expert-wave device path — Numeric Parity **V2.1** (FP64 authority).
//!
//! Isolated, additive path: `gate + up → SiLU → down → weighted combine` in one
//! command buffer per MoE layer. Flag `HAWKING_GLM_GPU_EXPERT_WAVE=1`, default
//! off. The default resident three-`matvec_batch` path must stay untouched
//! (Parity V2.1 item 6).
//!
//! ## Why not bit-identical SiLU
//!
//! The previous attempt required Metal `exp` == libm `expf` and failed at 1–2
//! ULP. V2.1 scores both host and device against an **FP64** silu authority
//! with condition-aware continuous metrics. Continuous drift is allowed;
//! discrete decisions (router top-k, selected experts, greedy token) remain
//! exact with no tolerance on the full forward path.
//!
//! ## Requires Metal for device tests
//!
//! The executor sandbox often has none. A missing device may skip; a shader
//! **compile** failure is a hard fail (never green-on-skip for a broken kernel).
//!
//! Controller:
//!
//! ```text
//! # Unit (no Metal):
//! cargo test -p hawking-core numeric_parity -- --nocapture
//! cargo test -p hawking-core flagship_wait_estimates -- --nocapture
//! cargo test -p hawking-core gpu_expert_wave_flag -- --nocapture
//!
//! # Device silu V2.1:
//! cargo test -p hawking-core --test gravity_glm_expert_wave_parity -- --nocapture
//!
//! # Default path must stay 3/3 green with flag OFF:
//! cargo test -p hawking-core --test gravity_glm_resident_parity -- --nocapture
//!
//! # Expert-wave full fixture (tiny; pair with LM_HEAD so experts are device bf16):
//! HAWKING_GLM_GPU_RESIDENT_STATE=1 HAWKING_GLM_GPU_LM_HEAD=1 \
//!   HAWKING_GLM_GPU_EXPERT_WAVE=1 HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS=1 \
//!   cargo test -p hawking-core --test gravity_glm_resident_parity -- --nocapture
//! ```

#![cfg(target_os = "macos")]

use hawking_core::metal::{MetalContext, TokenCommandBuffer};
use hawking_core::numeric_parity::{
    format_score_line, silu_mul_f32_host, silu_mul_f64_authority, score_pair, Bounds, SCHEMA,
};
use hawking_core::gravity_glm::{
    estimate_batched_mlp_drains_per_token, estimate_resident_expert_wave_waits_per_token,
    estimate_resident_waits_per_token, gpu_expert_wave_enabled, GlmArch, GPU_EXPERT_WAVE_ENV,
};

/// Post-norm-like gate/up fixtures (O(1) magnitude).
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

fn dispatch_silu_device(
    ctx: &MetalContext,
    gate: &[f32],
    up: &[f32],
) -> Result<Vec<f32>, String> {
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

/// Device SiLU vs host SiLU, both scored against FP64 authority (V2.1).
/// Bit-identity between Metal and libm is **not** required.
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

    eprintln!("numeric parity schema={SCHEMA}");
    let bounds = Bounds::continuous_only();
    eprintln!(
        "bounds: max_rel_l2={:.1e} max_meaningful_rel={:.1e} max_abs_near={:.1e} min_cos={}",
        bounds.max_relative_l2,
        bounds.max_meaningful_rel,
        bounds.max_abs_near_zero,
        bounds.min_cosine
    );

    let lengths = [16usize, 64, 257, 2048];
    let mut any_fail = false;

    for &n in &lengths {
        for (name, gate, up) in gate_up_pairs(n) {
            let g64: Vec<f64> = gate.iter().map(|&v| v as f64).collect();
            let u64: Vec<f64> = up.iter().map(|&v| v as f64).collect();
            let reference = silu_mul_f64_authority(&g64, &u64).expect("f64 authority");
            let host = silu_mul_f32_host(&gate, &up).expect("host silu");
            let device = dispatch_silu_device(&ctx, &gate, &up).unwrap_or_else(|e| {
                panic!(
                    "device silu failed (n={n} vec={name}): {e} — shader compile issues are hard fails"
                );
            });
            assert_eq!(device.len(), host.len());
            assert_eq!(device.len(), reference.len());

            let paired = score_pair(&host, &device, &reference, &bounds);
            eprintln!(
                "n={n} vec={name} cutoff={:.3e}",
                paired.abs_error_cutoff
            );
            eprintln!("  {}", format_score_line(&paired.host));
            eprintln!("  {}", format_score_line(&paired.device));
            if !paired.pass {
                any_fail = true;
                eprintln!("  FAIL host={:?} device={:?}", paired.host.failures, paired.device.failures);
            }
        }
    }

    assert!(
        !any_fail,
        "device silu failed V2.1 continuous gates against FP64 authority"
    );
}

/// Static drain contract: 234 → 78 from batched_mlp alone; default resident
/// estimator stays frozen at 583.
#[test]
fn expert_wave_static_drains_flagship() {
    let raw = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/gravity_glm/flagship_arch.json"),
    )
    .expect("flagship_arch.json");
    let header: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let arch = GlmArch::from_header(&header).unwrap();

    // Layer: default 3 drains (gate, up, down); wave 1 drain.
    assert_eq!(estimate_batched_mlp_drains_per_token(&arch, false), 234);
    assert_eq!(estimate_batched_mlp_drains_per_token(&arch, true), 78);
    assert_eq!(estimate_resident_waits_per_token(&arch), 583);
    assert_eq!(estimate_resident_expert_wave_waits_per_token(&arch), 430);

    // Flag defaults off — isolation contract.
    let prev = std::env::var_os(GPU_EXPERT_WAVE_ENV);
    std::env::remove_var(GPU_EXPERT_WAVE_ENV);
    assert!(!gpu_expert_wave_enabled());
    match prev {
        Some(v) => std::env::set_var(GPU_EXPERT_WAVE_ENV, v),
        None => std::env::remove_var(GPU_EXPERT_WAVE_ENV),
    }
}

/// Source-level proof that the default path is the non-wave path: with the
/// flag unset, the public enable predicate is false so resident forward never
/// enters `moe_device_wave`.
#[test]
fn default_path_never_enables_expert_wave_without_flag() {
    let prev = std::env::var_os(GPU_EXPERT_WAVE_ENV);
    std::env::remove_var(GPU_EXPERT_WAVE_ENV);
    assert!(
        !gpu_expert_wave_enabled(),
        "default resident path must not take the expert-wave branch"
    );
    // Even empty string / 0 must stay off.
    std::env::set_var(GPU_EXPERT_WAVE_ENV, "0");
    assert!(!gpu_expert_wave_enabled());
    std::env::set_var(GPU_EXPERT_WAVE_ENV, "");
    assert!(!gpu_expert_wave_enabled());
    match prev {
        Some(v) => std::env::set_var(GPU_EXPERT_WAVE_ENV, v),
        None => std::env::remove_var(GPU_EXPERT_WAVE_ENV),
    }
}
