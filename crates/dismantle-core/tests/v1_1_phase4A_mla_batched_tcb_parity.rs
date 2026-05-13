//! Phase 4A parity: mla_decode_metal_batched_slots_tcb (K=4) matches
//! K sequential mla_decode_metal dispatches at atol=1e-3.

#[cfg(target_os = "macos")]
mod tests {
    use dismantle_core::kernels::{mla_decode_metal, mla_decode_metal_batched_slots_tcb};
    use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};

    fn metal_ctx() -> Option<MetalContext> {
        MetalContext::new().ok()
    }

    fn make_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(data))
    }

    fn make_u32_buf(ctx: &MetalContext, data: &[u32]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(data))
    }

    #[test]
    fn k4_batched_tcb_matches_sequential() {
        let Some(ctx) = metal_ctx() else {
            eprintln!("v1_1_phase4A_mla_batched_tcb_parity: no Metal device, skipping");
            return;
        };

        let n_heads: usize = 4;
        let qk_nope: usize = 8;
        let qk_rope: usize = 4;
        let v_head_dim: usize = 8;
        let kv_lora: usize = 16;
        let head_dim_q = qk_nope + qk_rope;
        let n_batch: usize = 4; // K = 4
        let slot_stride: usize = 16;
        let slot_offsets_data = [0u32, slot_stride as u32, (2 * slot_stride) as u32, (3 * slot_stride) as u32];
        // Simulate spec-decode: each token sees [1..K] entries (positions 0..3)
        let seq_lens_data = [1u32, 2u32, 3u32, 4u32];
        let packed_entries = 4 * slot_stride;
        let max_seq_len = *seq_lens_data.iter().max().unwrap() as usize;
        let scale = 1.0_f32 / (head_dim_q as f32).sqrt();

        let seed = |i: usize| -> f32 { ((i as f32 * 1.618_034) % 2.0) - 1.0 };

        let kv_b_len = n_heads * (qk_nope + v_head_dim) * kv_lora;
        let kv_b_data: Vec<f32> = (0..kv_b_len).map(|i| seed(i * 3 + 7)).collect();
        let kv_b_buf = make_buf(&ctx, &kv_b_data);

        let c_kv_data: Vec<f32> = (0..packed_entries * kv_lora).map(|i| seed(i * 7 + 3)).collect();
        let k_pe_data: Vec<f32> = (0..packed_entries * qk_rope).map(|i| seed(i * 11 + 5)).collect();
        let q_batch_data: Vec<f32> = (0..n_batch * n_heads * head_dim_q).map(|i| seed(i * 13 + 11)).collect();

        let q_buf = make_buf(&ctx, &q_batch_data);
        let c_kv_buf = make_buf(&ctx, &c_kv_data);
        let k_pe_buf = make_buf(&ctx, &k_pe_data);
        let offsets_buf = make_u32_buf(&ctx, &slot_offsets_data);
        let seq_lens_buf = make_u32_buf(&ctx, &seq_lens_data);
        let out_buf = ctx.new_buffer(n_batch * n_heads * v_head_dim * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(&ctx);
        mla_decode_metal_batched_slots_tcb(
            &mut tcb,
            &q_buf, &c_kv_buf, &k_pe_buf, &kv_b_buf,
            &offsets_buf, &seq_lens_buf, &out_buf,
            n_heads, qk_nope, qk_rope, v_head_dim, kv_lora,
            n_batch, max_seq_len, scale,
        ).expect("mla_decode_metal_batched_slots_tcb");
        tcb.commit_and_wait().expect("TCB commit");

        let out_batch: Vec<f32> = unsafe {
            let ptr = out_buf.contents() as *const f32;
            std::slice::from_raw_parts(ptr, n_batch * n_heads * v_head_dim).to_vec()
        };

        // Sequential reference
        for m in 0..n_batch {
            let seq_len = seq_lens_data[m] as usize;
            let base = slot_offsets_data[m] as usize;
            let q_m = &q_batch_data[m * n_heads * head_dim_q..(m + 1) * n_heads * head_dim_q];
            let c_start = base * kv_lora;
            let c_end = (base + seq_len) * kv_lora;
            let k_start = base * qk_rope;
            let k_end = (base + seq_len) * qk_rope;
            let mut out_seq = vec![0.0f32; n_heads * v_head_dim];

            mla_decode_metal(
                &ctx, q_m,
                &c_kv_data[c_start..c_end],
                &k_pe_data[k_start..k_end],
                &kv_b_buf,
                n_heads, qk_nope, qk_rope, v_head_dim, kv_lora,
                seq_len, scale, &mut out_seq,
            ).expect("mla_decode_metal sequential");

            let off = m * n_heads * v_head_dim;
            let atol = 1e-3_f32;
            for j in 0..n_heads * v_head_dim {
                let b = out_batch[off + j];
                let s = out_seq[j];
                let diff = (b - s).abs();
                assert!(
                    diff <= atol,
                    "slot {m} element {j}: tcb_batch={b:.6} seq={s:.6} diff={diff:.2e} > {atol}"
                );
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn phase4a_mla_batched_tcb_parity_skipped_non_macos() {
    eprintln!("v1_1_phase4A_mla_batched_tcb_parity: non-macOS, all tests skipped");
}
