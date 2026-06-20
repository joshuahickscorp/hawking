//! Q4K_FAST parity vs Q4_K v3_8r at q_proj decode shape (rows=2048,
//! cols=2048).
//!
//! Builds a synthetic Q4_K tensor with constraints that keep the
//! per-sub-block products `d * sb_idx[k]` and `dmin * mb_idx[k]` exactly
//! representable in fp16 (so the FAST layout's fp16 sub_scale / sub_min
//! storage is lossless). Runs both kernels and asserts bit-identical
//! per-row output.
//!
//! Run with: `cargo test --release -p hawking-core --test q4k_fast_parity -- --nocapture`

#![cfg(target_os = "macos")]

use hawking_core::kernels;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use hawking_core::q4k_fast::{convert_q4k_tensor_to_fast, Q4K_BLOCK_BYTES, Q4K_FAST_BLOCK_BYTES};
use half::f16;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

fn ctx() -> MetalContext {
    MetalContext::new().expect("Metal device required")
}

/// Build a Q4_K tensor where `d * sb_idx[k]` and `dmin * mb_idx[k]` are
/// guaranteed exactly representable in fp16:
///
/// * `d`    = 1.0 (fp16 exact)
/// * `dmin` = 0.5 (fp16 exact)
/// * `sb_idx[k]` ∈ [0..63]  → product 0..63, all integers ≤ 2^11, fp16 exact
/// * `mb_idx[k]` ∈ [0..63]  → product 0..31.5 in 0.5 steps, fp16 exact
///
/// Random 4-bit nibbles for the rest.
fn make_synthetic_q4k_tensor(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    assert_eq!(cols % 256, 0, "cols must be a multiple of 256");
    let blocks_per_row = cols / 256;
    let n_blocks = rows * blocks_per_row;
    let mut bytes = vec![0u8; n_blocks * Q4K_BLOCK_BYTES];
    let mut rng = Pcg64Mcg::new(seed as u128);

    let d = f16::from_f32(1.0_f32);
    let dmin = f16::from_f32(0.5_f32);

    for b in 0..n_blocks {
        let off = b * Q4K_BLOCK_BYTES;
        bytes[off..off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
        bytes[off + 2..off + 4].copy_from_slice(&dmin.to_bits().to_le_bytes());

        // Pick random sb_idx[k] ∈ [0..63], mb_idx[k] ∈ [0..63] for k in 0..8.
        let sb: [u8; 8] = [
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
        ];
        let mb: [u8; 8] = [
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
            rng.gen::<u8>() & 0x3F,
        ];
        // Repack into Q4_K's bytes [4..16] layout (inverse of
        // q4k_fast::decode_q4k_sb_mb):
        //   bytes[4+sub] = sb[sub] | ((sb[4+sub] >> 4) << 6)   for sub in 0..4
        //   bytes[8+sub] = mb[sub] | ((mb[4+sub] >> 4) << 6)   for sub in 0..4
        //   bytes[12+j]  = (sb[4+j] & 0x0F) | ((mb[4+j] & 0x0F) << 4)  for j in 0..4
        for sub in 0..4 {
            let hi_sb = (sb[4 + sub] >> 4) & 0x03;
            let hi_mb = (mb[4 + sub] >> 4) & 0x03;
            bytes[off + 4 + sub] = (sb[sub] & 0x3F) | (hi_sb << 6);
            bytes[off + 8 + sub] = (mb[sub] & 0x3F) | (hi_mb << 6);
        }
        for j in 0..4 {
            bytes[off + 12 + j] = (sb[4 + j] & 0x0F) | ((mb[4 + j] & 0x0F) << 4);
        }
        // Random nibbles for the 128-byte data section.
        for i in 16..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

fn make_x(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..cols)
        .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
        .collect()
}

#[test]
fn q4k_fast_v1_bit_identical_to_v3_8r_at_qproj_decode_shape() {
    let rows = 2048usize;
    let cols = 2048usize;
    let blocks_per_row = cols / 256;

    let ctx = ctx();

    // Build synthetic Q4_K tensor with fp16-exact sub-products.
    let q4k_bytes = make_synthetic_q4k_tensor(rows, cols, 0xCAFE_F00D_DEAD_BEEFu64);
    let q4k_byte_size = q4k_bytes.len();
    assert_eq!(q4k_byte_size, rows * blocks_per_row * Q4K_BLOCK_BYTES);

    // Convert to Q4K_FAST.
    let fast_bytes = convert_q4k_tensor_to_fast(&q4k_bytes, rows, cols);
    let fast_byte_size = fast_bytes.len();
    assert_eq!(fast_byte_size, rows * blocks_per_row * Q4K_FAST_BLOCK_BYTES);

    // Activation.
    let x = make_x(cols, 0x1234_5678_9ABC_DEF0u64);

    // Pinned buffers.
    let q4k_buf: PinnedBuffer = ctx.new_buffer_with_bytes(&q4k_bytes);
    let fast_buf: PinnedBuffer = ctx.new_buffer_with_bytes(&fast_bytes);
    let x_buf: PinnedBuffer = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
    let out_v3_buf: PinnedBuffer = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let out_fast_buf: PinnedBuffer = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    // Run v3_8r.
    {
        let mut tcb = TokenCommandBuffer::new(&ctx);
        kernels::gemv_q4_k_m_v3_8r_pinned_tcb(
            &mut tcb,
            &q4k_buf,
            0,
            q4k_byte_size,
            rows,
            cols,
            &x_buf,
            &out_v3_buf,
        )
        .expect("v3_8r dispatch");
        tcb.commit_and_wait().expect("v3_8r commit");
    }

    // Run Q4K_FAST v1.
    {
        let mut tcb = TokenCommandBuffer::new(&ctx);
        kernels::gemv_q4k_fast_v1_pinned_tcb(
            &mut tcb,
            &fast_buf,
            0,
            fast_byte_size,
            rows,
            cols,
            &x_buf,
            &out_fast_buf,
        )
        .expect("q4k_fast_v1 dispatch");
        tcb.commit_and_wait().expect("q4k_fast_v1 commit");
    }

    // Read both outputs.
    let out_v3_ptr = out_v3_buf.contents() as *const f32;
    let out_v3 = unsafe { std::slice::from_raw_parts(out_v3_ptr, rows) };
    let out_fast_ptr = out_fast_buf.contents() as *const f32;
    let out_fast = unsafe { std::slice::from_raw_parts(out_fast_ptr, rows) };

    // Bit-identical assertion (bit-pattern equality of f32).
    let mut first_diff = None;
    let mut max_abs_diff = 0.0f32;
    for i in 0..rows {
        let a = out_v3[i];
        let b = out_fast[i];
        let abs_d = (a - b).abs();
        if abs_d > max_abs_diff {
            max_abs_diff = abs_d;
        }
        if a.to_bits() != b.to_bits() && first_diff.is_none() {
            first_diff = Some((i, a, b));
        }
    }

    if let Some((i, a, b)) = first_diff {
        panic!(
            "Q4K_FAST vs v3_8r diverges at row={i}: v3_8r={a} ({:#x}) fast={b} ({:#x}) max_abs_diff={max_abs_diff:.3e}",
            a.to_bits(),
            b.to_bits()
        );
    }
    println!("[q4k_fast_parity] rows={rows} cols={cols}: bit-identical (max_abs_diff=0)");
}
