//! Phase A Wedge A2 — mla_decode_kernel_batched parity test.
//!
//! Verifies that M=4 batched MLA output matches 4 sequential M=1 calls
//! at atol=1e-3 (order-of-summation drift in attention is expected).
//!
//! This test exercises the Metal kernel directly via its Rust dispatcher,
//! bypassing the full forward pass. Skips on non-macOS or if no Metal device.

#[cfg(target_os = "macos")]
mod tests {
    use dismantle_core::kernels::{mla_decode_metal, mla_decode_metal_batched};
    use dismantle_core::metal::{MetalContext, PinnedBuffer};

    fn metal_ctx() -> Option<MetalContext> {
        MetalContext::new().ok()
    }

    fn make_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(data))
    }

    /// Build a small synthetic MLA decode problem and verify batched vs sequential.
    #[test]
    fn batched_mla_matches_sequential_at_atol_1e3() {
        let Some(ctx) = metal_ctx() else {
            eprintln!("v_a2_mla_batched_parity: no Metal device, skipping");
            return;
        };

        // Tiny dims that still exercise all code paths.
        let n_heads: usize      = 2;
        let qk_nope: usize      = 8;
        let qk_rope: usize      = 4;
        let v_head_dim: usize   = 8;
        let kv_lora: usize      = 16;
        let head_dim_q          = qk_nope + qk_rope;
        let n_batch: usize      = 4;

        // Base KV cache: 3 existing entries (prefix).
        let base_seq: usize = 3;
        let total_seq = base_seq + n_batch;

        let scale = 1.0_f32 / (head_dim_q as f32).sqrt();

        // Synthetic random-ish data (deterministic).
        let seed = |i: usize| -> f32 { ((i as f32 * 1.6180339) % 2.0) - 1.0 };

        // kv_b_proj: [n_heads, (qk_nope + v_head_dim), kv_lora]
        let kv_b_len = n_heads * (qk_nope + v_head_dim) * kv_lora;
        let kv_b_data: Vec<f32> = (0..kv_b_len).map(|i| seed(i * 3 + 7)).collect();
        let kv_b_buf = make_buf(&ctx, &kv_b_data);

        // c_kv: [total_seq, kv_lora] and k_pe: [total_seq, qk_rope].
        let c_kv: Vec<f32> = (0..total_seq * kv_lora).map(|i| seed(i * 2 + 1)).collect();
        let k_pe: Vec<f32> = (0..total_seq * qk_rope).map(|i| seed(i * 5 + 3)).collect();

        // Q batch: [n_batch, n_heads, head_dim_q].
        let q_batch: Vec<f32> = (0..n_batch * n_heads * head_dim_q)
            .map(|i| seed(i * 7 + 11))
            .collect();

        // --- Batched dispatch (A2 kernel) ---
        let mut out_batch = vec![0.0f32; n_batch * n_heads * v_head_dim];
        mla_decode_metal_batched(
            &ctx,
            &q_batch,
            &c_kv,
            &k_pe,
            &kv_b_buf,
            n_heads,
            qk_nope,
            qk_rope,
            v_head_dim,
            kv_lora,
            base_seq,
            n_batch,
            scale,
            &mut out_batch,
        )
        .expect("mla_decode_metal_batched");

        // --- Sequential dispatch (existing single-token kernel) ---
        // Token m uses seq_len = max(base_seq + m, 1): only sees tokens before it.
        let mut out_seq = vec![0.0f32; n_batch * n_heads * v_head_dim];
        for m in 0..n_batch {
            let seq_len_m = (base_seq + m).max(1);
            let q_m: Vec<f32> = q_batch[m * n_heads * head_dim_q..(m + 1) * n_heads * head_dim_q]
                .to_vec();
            let c_kv_m: Vec<f32> = c_kv[..seq_len_m * kv_lora].to_vec();
            let k_pe_m: Vec<f32> = k_pe[..seq_len_m * qk_rope].to_vec();

            let out_off = m * n_heads * v_head_dim;
            mla_decode_metal(
                &ctx,
                &q_m,
                &c_kv_m,
                &k_pe_m,
                &kv_b_buf,
                n_heads,
                qk_nope,
                qk_rope,
                v_head_dim,
                kv_lora,
                seq_len_m,
                scale,
                &mut out_seq[out_off..out_off + n_heads * v_head_dim],
            )
            .expect("mla_decode_metal sequential");
        }

        // --- Compare ---
        let atol = 1e-3_f32;
        for m in 0..n_batch {
            let off = m * n_heads * v_head_dim;
            for j in 0..n_heads * v_head_dim {
                let b = out_batch[off + j];
                let s = out_seq[off + j];
                let diff = (b - s).abs();
                assert!(
                    diff <= atol,
                    "token {m} element {j}: batch={b} seq={s} diff={diff} > {atol}"
                );
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn a2_mla_batched_parity_skipped_non_macos() {
    eprintln!("v_a2_mla_batched_parity: non-macOS, all tests skipped");
}
