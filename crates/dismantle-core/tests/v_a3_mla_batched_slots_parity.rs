//! Continuous batching MLA slot-kernel parity test.
//!
//! Verifies that one batched dispatch over independent slot KV ranges matches
//! separate single-token MLA dispatches for each slot.

#[cfg(target_os = "macos")]
mod tests {
    use dismantle_core::kernels::{mla_decode_metal, mla_decode_metal_batched_slots};
    use dismantle_core::metal::{MetalContext, PinnedBuffer};

    fn metal_ctx() -> Option<MetalContext> {
        MetalContext::new().ok()
    }

    fn make_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(data))
    }

    #[test]
    fn batched_slot_mla_matches_independent_sequential_dispatches() {
        let Some(ctx) = metal_ctx() else {
            eprintln!("v_a3_mla_batched_slots_parity: no Metal device, skipping");
            return;
        };

        let n_heads: usize = 2;
        let qk_nope: usize = 8;
        let qk_rope: usize = 4;
        let v_head_dim: usize = 8;
        let kv_lora: usize = 16;
        let head_dim_q = qk_nope + qk_rope;
        let n_batch: usize = 3;
        let slot_stride: usize = 8;
        let slot_offsets = [0u32, slot_stride as u32, (2 * slot_stride) as u32];
        let seq_lens = [3u32, 5u32, 2u32];
        let packed_entries = 3 * slot_stride;
        let scale = 1.0_f32 / (head_dim_q as f32).sqrt();

        let seed = |i: usize| -> f32 { ((i as f32 * 1.324_718) % 2.0) - 1.0 };

        let kv_b_len = n_heads * (qk_nope + v_head_dim) * kv_lora;
        let kv_b_data: Vec<f32> = (0..kv_b_len).map(|i| seed(i * 3 + 5)).collect();
        let kv_b_buf = make_buf(&ctx, &kv_b_data);

        let c_kv: Vec<f32> = (0..packed_entries * kv_lora)
            .map(|i| seed(i * 7 + 1))
            .collect();
        let k_pe: Vec<f32> = (0..packed_entries * qk_rope)
            .map(|i| seed(i * 11 + 9))
            .collect();
        let q_batch: Vec<f32> = (0..n_batch * n_heads * head_dim_q)
            .map(|i| seed(i * 13 + 17))
            .collect();

        let mut out_batch = vec![0.0f32; n_batch * n_heads * v_head_dim];
        mla_decode_metal_batched_slots(
            &ctx,
            &q_batch,
            &c_kv,
            &k_pe,
            &kv_b_buf,
            &slot_offsets,
            &seq_lens,
            n_heads,
            qk_nope,
            qk_rope,
            v_head_dim,
            kv_lora,
            n_batch,
            scale,
            &mut out_batch,
        )
        .expect("mla_decode_metal_batched_slots");

        let mut out_seq = vec![0.0f32; n_batch * n_heads * v_head_dim];
        for m in 0..n_batch {
            let seq_len = seq_lens[m] as usize;
            let base = slot_offsets[m] as usize;
            let q_m = &q_batch[m * n_heads * head_dim_q..(m + 1) * n_heads * head_dim_q];
            let c_start = base * kv_lora;
            let c_end = (base + seq_len) * kv_lora;
            let k_start = base * qk_rope;
            let k_end = (base + seq_len) * qk_rope;
            let out_off = m * n_heads * v_head_dim;

            mla_decode_metal(
                &ctx,
                q_m,
                &c_kv[c_start..c_end],
                &k_pe[k_start..k_end],
                &kv_b_buf,
                n_heads,
                qk_nope,
                qk_rope,
                v_head_dim,
                kv_lora,
                seq_len,
                scale,
                &mut out_seq[out_off..out_off + n_heads * v_head_dim],
            )
            .expect("mla_decode_metal sequential slot");
        }

        let atol = 1e-3_f32;
        for m in 0..n_batch {
            let off = m * n_heads * v_head_dim;
            for j in 0..n_heads * v_head_dim {
                let b = out_batch[off + j];
                let s = out_seq[off + j];
                let diff = (b - s).abs();
                assert!(
                    diff <= atol,
                    "slot {m} element {j}: batch={b} seq={s} diff={diff} > {atol}"
                );
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn a3_mla_batched_slots_parity_skipped_non_macos() {
    eprintln!("v_a3_mla_batched_slots_parity: non-macOS, all tests skipped");
}
