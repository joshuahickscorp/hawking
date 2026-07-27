//! Multi-vector **Numeric Parity V2.1** suite: device-resident `native.bf16`
//! `lm_head` vs host f32 path, both scored against an **FP64 authority**.
//!
//! Canonical policy: root `NUMERIC_PARITY_V2_1.md`.
//!
//! ## Why not max relative error
//!
//! V2 rejected device bf16 at 3.23e-3 max scalar relative error. That rejection
//! is reclassified `NUMERIC_GATE_INSUFFICIENTLY_CONDITIONED`: the test vector
//! spanned denormal-scale to huge values, and relative error on a near-zero
//! element is meaningless. V2.1 uses condition-aware metrics + FP64 authority.
//!
//! ## What this suite does
//!
//! 1. Build controlled bf16 weight matrices (unit-scale exponents — not a 50-order
//!    span synthetic pathology).
//! 2. Use **post-norm-like** activation fixtures (O(1) magnitude, mild structure).
//!    Flagship residual-stream activations after RMSNorm live in this regime;
//!    sealed full-vocab logits from the tiny fixture are O(1) as well
//!    (`tests/fixtures/gravity_glm/ref_logits.f32`: |logit| median ≈ 0.54).
//! 3. Compute the FP64 authority matvec (`numeric_parity::matvec_bf16_f64_authority`).
//! 4. Run host f32 (`matvec_bf16_host`) and device bf16 GEMV.
//! 5. Score **both** against f64 with V2.1 bounds. Reject only if meaningful-scale
//!    logits fail, full-vector relative L2 fails, cosine/KL fail, or a discrete
//!    decision (greedy argmax / exact top-k) differs.
//!
//! ULP distributions are always printed (median / p95 / p99 / max). A large tail
//! on a pass is information, not a silent failure.
//!
//! ## Requires Metal
//!
//! The executor sandbox often has none. Controller:
//!
//! ```text
//! cargo test -p hawking-core --test gravity_glm_lm_head_device_parity -- --nocapture
//!
//! # metric unit tests (no Metal; run in any sandbox):
//! cargo test -p hawking-core numeric_parity -- --nocapture
//! ```
//!
//! Flags for the full model path (not required for this unit suite):
//! - `HAWKING_GLM_GPU_LM_HEAD=1` — device-resident native.bf16 matvecs + GPU head
//! - `HAWKING_GLM_GPU_RESIDENT_STATE=1` — resident activations/KV
//! - `HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS=1` — opt-in full vocab readback for
//!   continuous-logit scoring on the forward path (default is token + top-k only)
//!
//! This unit suite calls `dispatch_gemv_native_bf16_seq` directly, so it always
//! scores the full logit vector against the FP64 authority under V2.1.
//!
//! **Isolation:** this suite does not touch the default resident path.

#![cfg(target_os = "macos")]

use hawking_core::gravity::matvec_bf16_host;
use hawking_core::metal::MetalContext;
use hawking_core::numeric_parity::{
    format_score_line, matvec_bf16_f64_authority, score_pair, Bounds, SCHEMA,
};

/// Post-norm-like activation fixtures: O(1) magnitude, several independent
/// patterns so phase-aligned luck on one vector cannot pass the suite.
///
/// These are **not** a synthetic span of 50 orders of magnitude. They model
/// residual-stream / post-RMSNorm hidden states (flagship hidden = 6144; we
/// use smaller cols for unit shapes and the same statistical regime).
fn activations(cols: usize) -> Vec<(&'static str, Vec<f32>)> {
    vec![
        (
            "ramp",
            (0..cols).map(|c| (c as f32) * 0.01 - 0.3).collect(),
        ),
        (
            "mod17",
            (0..cols)
                .map(|c| ((c * 5 + 3) % 17) as f32 * 0.05 - 0.4)
                .collect(),
        ),
        ("ones", vec![1.0; cols]),
        ("zeros", vec![0.0; cols]),
        (
            "block3",
            (0..cols)
                .map(|c| if c % 3 == 0 { 0.5 } else { -0.25 })
                .collect(),
        ),
        (
            "sin",
            (0..cols).map(|c| ((c as f32).sin()) * 0.1).collect(),
        ),
        (
            "half",
            (0..cols)
                .map(|c| if c < cols / 2 { 0.125 } else { -0.0625 })
                .collect(),
        ),
        (
            "mod31",
            (0..cols)
                .map(|c| ((c * 13 + 7) % 31) as f32 * 0.02 - 0.3)
                .collect(),
        ),
        // Mild heavy-tail but still O(1) after clipping — not denormal pathology.
        (
            "gaussish",
            (0..cols)
                .map(|c| {
                    let u = ((c * 17 + 5) % 100) as f32 / 100.0 - 0.5;
                    (u * 3.0).tanh() // ≈ (-0.9, 0.9)
                })
                .collect(),
        ),
    ]
}

/// Controlled bf16 matrix: exponents near unit scale so row dots stay well
/// conditioned. The previous generator used `bits % 0x7F00`, which freely
/// produced the 1e-32…5e18 span that made max-relative-error unusable.
fn make_bf16_matrix_unit_scale(rows: usize, cols: usize, salt: u32) -> Vec<u8> {
    let mut bits = Vec::with_capacity(rows * cols * 2);
    for i in 0..(rows * cols) {
        // Sign from low bit; exponent biased near 127 (bf16); 7-bit mantissa.
        let h = (i as u32).wrapping_mul(37).wrapping_add(salt);
        let sign = ((h >> 15) & 1) as u16;
        let exp = (120 + (h % 15)) as u16; // bf16 exp in [120, 134] → ~0.0078 … ~256
        let mant = (h & 0x7f) as u16;
        let u = (sign << 15) | (exp << 7) | mant;
        bits.extend_from_slice(&u.to_le_bytes());
    }
    bits
}

#[test]
fn device_bf16_lm_head_v21_against_f64_over_several_vectors() {
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
    let bounds = Bounds::logits();
    eprintln!(
        "bounds: max_rel_l2={:.1e} max_meaningful_rel={:.1e} max_abs_near={:.1e} \
         min_cos={} max_kl={:.1e} top_k={}",
        bounds.max_relative_l2,
        bounds.max_meaningful_rel,
        bounds.max_abs_near_zero,
        bounds.min_cosine,
        bounds.max_kl,
        bounds.top_k
    );

    // Two matrix shapes so one accidental dimension alignment cannot hide bugs.
    let shapes = [(64usize, 32usize), (257usize, 17usize), (16usize, 64usize)];
    let mut any_fail = false;

    for &(rows, cols) in &shapes {
        let weight = make_bf16_matrix_unit_scale(rows, cols, (rows * cols) as u32);
        let w_buf = ctx
            .new_buffer_with_bytes_checked(&weight)
            .expect("upload weight");

        for (vi, (name, x)) in activations(cols).into_iter().enumerate() {
            let reference = matvec_bf16_f64_authority(&weight, cols, &x)
                .unwrap_or_else(|e| panic!("f64 authority failed: {e}"));
            let host = matvec_bf16_host(&weight, cols, &x).expect("host f32");
            let device = hawking_core::gravity_glm::gpu::dispatch_gemv_native_bf16_seq(
                &ctx,
                &w_buf,
                rows as u32,
                cols as u32,
                &x,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "device gemv failed (rows={rows} cols={cols} vec={vi}/{name}): {e} — \
                     if this is a shader compile issue it is a hard fail"
                );
            });

            assert_eq!(device.len(), host.len());
            assert_eq!(device.len(), reference.len());

            let paired = score_pair(&host, &device, &reference, &bounds);
            eprintln!(
                "shape=[{rows},{cols}] vec={vi}/{name} cutoff={:.3e}",
                paired.abs_error_cutoff
            );
            eprintln!("  {}", format_score_line(&paired.host));
            eprintln!("  {}", format_score_line(&paired.device));
            if !paired.device.pass {
                any_fail = true;
                eprintln!("  DEVICE FAIL: {:?}", paired.device.failures);
            }
            if !paired.host.pass {
                // Host vs f64 can fail the same continuous bounds if the host
                // path is poorly conditioned; report it — do not silently ignore.
                any_fail = true;
                eprintln!("  HOST FAIL: {:?}", paired.host.failures);
            }

            // Hard assert on the pair: both backends must pass V2.1.
            assert!(
                paired.pass,
                "rows={rows} cols={cols} vec={vi}/{name}: V2.1 pair failed\n  host: {:?}\n  device: {:?}",
                paired.host.failures,
                paired.device.failures
            );
        }
        eprintln!(
            "ok shape [{rows}, {cols}] over {} vectors (V2.1 / FP64 authority)",
            activations(cols).len()
        );
    }

    assert!(!any_fail, "internal bookkeeping: any_fail set but asserts passed");
}
