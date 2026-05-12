//! v0.5.6 foundation suite parity tests.
//!
//! For each `_buf` dispatcher sibling added in v0.5.6, verifies that the
//! new variant produces the same output as the existing function when given
//! the same inputs. Because both paths dispatch the same kernel, expected
//! tolerance is atol=0 (bit-identical f32 outputs).
//!
//! Also tests the four new DecodeArena buffer slots.

#![cfg(target_os = "macos")]

use half::f16;
use dismantle_core::metal::{DecodeArena, MetalContext};
use dismantle_core::kernels::{
    rmsnorm_metal, rmsnorm_metal_buf,
    silu_mul_f16_metal, silu_mul_metal_buf,
    gemv_f32_attn_metal, gemv_f32_attn_metal_buf,
    gemv_f32_attn_metal_pinned, gemv_f32_attn_metal_pinned_buf,
    dispatch_gemv_f32_attn_pinned_pair_batched, gemv_f32_attn_pair_metal_buf,
    gemv_f32_moe_metal, gemv_f32_moe_metal_buf,
    moe_grouped_gemm_q4_metal, moe_grouped_gemm_q4_metal_buf,
};

fn make_ctx() -> MetalContext {
    MetalContext::new().expect("Metal device")
}

fn linspace(n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|i| lo + (hi - lo) * (i as f32) / (n.max(1) - 1).max(1) as f32).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ── rmsnorm_metal_buf ────────────────────────────────────────────────────────

#[test]
fn rmsnorm_metal_buf_matches_rmsnorm_metal() {
    let ctx = make_ctx();
    let hidden = 2048;
    let eps = 1e-6_f32;

    let x_f32: Vec<f32> = linspace(hidden, -1.0, 1.0);
    let w_f32: Vec<f32> = linspace(hidden, 0.5, 1.5);

    // Reference: existing rmsnorm_metal (allocates its own f16 buffers internally)
    let mut ref_out = vec![0.0f32; hidden];
    rmsnorm_metal(&ctx, &x_f32, &w_f32, eps, &mut ref_out).expect("rmsnorm_metal");

    // _buf variant: caller provides pre-existing f16 buffers
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&w_f16));
    let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f16>());
    rmsnorm_metal_buf(&ctx, &x_buf, &w_buf, eps, hidden, &out_buf).expect("rmsnorm_metal_buf");

    // Read f16 output back to f32 for comparison
    let out_ptr = out_buf.contents() as *const f16;
    let buf_out_f16 = unsafe { std::slice::from_raw_parts(out_ptr, hidden) };
    let buf_out: Vec<f32> = buf_out_f16.iter().map(|h| h.to_f32()).collect();

    let diff = max_abs_diff(&ref_out, &buf_out);
    // Same kernel, same f16 inputs → atol=0
    assert!(diff == 0.0, "rmsnorm_metal_buf vs rmsnorm_metal diff={diff} (expected 0)");
}

// ── silu_mul_metal_buf ───────────────────────────────────────────────────────

#[test]
fn silu_mul_metal_buf_matches_silu_mul_f16() {
    let ctx = make_ctx();
    let n = 4096;

    let gate_f32: Vec<f32> = linspace(n, -2.0, 2.0);
    let up_f32:   Vec<f32> = linspace(n, -1.0, 1.0);

    let gate_f16: Vec<f16> = gate_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let up_f16:   Vec<f16> = up_f32.iter().map(|&v| f16::from_f32(v)).collect();

    // Reference: silu_mul_f16_metal (kernel "silu_mul_f16" — same math)
    let g_ref = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&gate_f16));
    let u_ref = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&up_f16));
    let out_ref = ctx.new_buffer(n * std::mem::size_of::<f16>());
    silu_mul_f16_metal(&ctx, &g_ref, &u_ref, &out_ref, n).expect("silu_mul_f16_metal");

    // _buf variant: kernel "silu_mul" (functionally identical)
    let g_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&gate_f16));
    let u_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&up_f16));
    let out_buf = ctx.new_buffer(n * std::mem::size_of::<f16>());
    silu_mul_metal_buf(&ctx, &g_buf, &u_buf, &out_buf, n).expect("silu_mul_metal_buf");

    let ref_ptr = out_ref.contents() as *const f16;
    let buf_ptr = out_buf.contents() as *const f16;
    let ref_out: Vec<f32> = unsafe { std::slice::from_raw_parts(ref_ptr, n) }
        .iter().map(|h| h.to_f32()).collect();
    let buf_out: Vec<f32> = unsafe { std::slice::from_raw_parts(buf_ptr, n) }
        .iter().map(|h| h.to_f32()).collect();

    let diff = max_abs_diff(&ref_out, &buf_out);
    // Same functional kernel, same f16 data → atol=0
    assert!(diff == 0.0, "silu_mul_metal_buf vs silu_mul_f16_metal diff={diff} (expected 0)");
}

// ── gemv_f32_attn_metal_buf ──────────────────────────────────────────────────

#[test]
fn gemv_f32_attn_metal_buf_matches_existing() {
    let ctx = make_ctx();
    let rows = 128;
    let cols = 256;

    let w: Vec<f32> = linspace(rows * cols, -0.5, 0.5);
    let x: Vec<f32> = linspace(cols, -1.0, 1.0);

    // Reference: existing gemv_f32_attn_metal (allocates its own buffers)
    let mut ref_out = vec![0.0f32; rows];
    gemv_f32_attn_metal(&ctx, &w, rows, cols, &x, &mut ref_out).expect("gemv_f32_attn_metal");

    // _buf variant: caller provides pre-existing x_buf and y_buf
    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    gemv_f32_attn_metal_buf(&ctx, &w, rows, cols, &x_buf, &y_buf)
        .expect("gemv_f32_attn_metal_buf");

    let y_ptr = y_buf.contents() as *const f32;
    let buf_out: Vec<f32> = unsafe { std::slice::from_raw_parts(y_ptr, rows) }.to_vec();

    let diff = max_abs_diff(&ref_out, &buf_out);
    assert!(diff == 0.0, "gemv_f32_attn_metal_buf diff={diff} (expected 0)");
}

// ── gemv_f32_attn_metal_pinned_buf ──────────────────────────────────────────

#[test]
fn gemv_f32_attn_metal_pinned_buf_matches_existing() {
    let ctx = make_ctx();
    let rows = 128;
    let cols = 256;

    let w: Vec<f32> = linspace(rows * cols, -0.5, 0.5);
    let x: Vec<f32> = linspace(cols, -1.0, 1.0);

    // Upload w to a pinned buffer (as gemv_f32_attn_metal_pinned expects)
    let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&w));

    // Reference: existing gemv_f32_attn_metal_pinned
    let mut ref_out = vec![0.0f32; rows];
    gemv_f32_attn_metal_pinned(&ctx, &w_buf, rows, cols, &x, &mut ref_out)
        .expect("gemv_f32_attn_metal_pinned");

    // _buf variant: caller provides pre-existing x_buf and y_buf too
    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    gemv_f32_attn_metal_pinned_buf(&ctx, &w_buf, rows, cols, &x_buf, &y_buf)
        .expect("gemv_f32_attn_metal_pinned_buf");

    let y_ptr = y_buf.contents() as *const f32;
    let buf_out: Vec<f32> = unsafe { std::slice::from_raw_parts(y_ptr, rows) }.to_vec();

    let diff = max_abs_diff(&ref_out, &buf_out);
    assert!(diff == 0.0, "gemv_f32_attn_metal_pinned_buf diff={diff} (expected 0)");
}

// ── gemv_f32_attn_pair_metal_buf ─────────────────────────────────────────────

#[test]
fn gemv_f32_attn_pair_metal_buf_matches_existing() {
    let ctx = make_ctx();
    let rows_a = 64;
    let rows_b = 96;
    let cols = 256;

    let wa: Vec<f32> = linspace(rows_a * cols, -0.4, 0.4);
    let wb: Vec<f32> = linspace(rows_b * cols, -0.3, 0.6);
    let x:  Vec<f32> = linspace(cols, -1.0, 1.0);

    let wa_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&wa));
    let wb_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&wb));

    // Reference: existing pair batched dispatcher (allocates output buffers)
    let mut ref_a = vec![0.0f32; rows_a];
    let mut ref_b = vec![0.0f32; rows_b];
    dispatch_gemv_f32_attn_pinned_pair_batched(
        &ctx, &wa_buf, rows_a, &wb_buf, rows_b, cols, &x, &mut ref_a, &mut ref_b,
    ).expect("pair_batched");

    // _buf variant: caller provides all pre-existing buffers
    let x_buf    = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
    let out_a_buf = ctx.new_buffer(rows_a * std::mem::size_of::<f32>());
    let out_b_buf = ctx.new_buffer(rows_b * std::mem::size_of::<f32>());
    gemv_f32_attn_pair_metal_buf(
        &ctx, &wa_buf, rows_a, &wb_buf, rows_b, cols,
        &x_buf, &out_a_buf, &out_b_buf,
    ).expect("gemv_f32_attn_pair_metal_buf");

    let pa = out_a_buf.contents() as *const f32;
    let pb = out_b_buf.contents() as *const f32;
    let buf_a: Vec<f32> = unsafe { std::slice::from_raw_parts(pa, rows_a) }.to_vec();
    let buf_b: Vec<f32> = unsafe { std::slice::from_raw_parts(pb, rows_b) }.to_vec();

    let diff_a = max_abs_diff(&ref_a, &buf_a);
    let diff_b = max_abs_diff(&ref_b, &buf_b);
    assert!(diff_a == 0.0, "pair_buf out_a diff={diff_a} (expected 0)");
    assert!(diff_b == 0.0, "pair_buf out_b diff={diff_b} (expected 0)");
}

// ── gemv_f32_moe_metal_buf ───────────────────────────────────────────────────

#[test]
fn gemv_f32_moe_metal_buf_matches_existing() {
    let ctx = make_ctx();
    let rows = 64;
    let cols = 256;

    let w: Vec<f32> = linspace(rows * cols, -0.5, 0.5);
    let x: Vec<f32> = linspace(cols, -1.0, 1.0);

    // Reference: existing gemv_f32_moe_metal
    let mut ref_out = vec![0.0f32; rows];
    gemv_f32_moe_metal(&ctx, &w, rows, cols, &x, &mut ref_out).expect("gemv_f32_moe_metal");

    // _buf variant
    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    gemv_f32_moe_metal_buf(&ctx, &w, rows, cols, &x_buf, &y_buf)
        .expect("gemv_f32_moe_metal_buf");

    let y_ptr = y_buf.contents() as *const f32;
    let buf_out: Vec<f32> = unsafe { std::slice::from_raw_parts(y_ptr, rows) }.to_vec();

    let diff = max_abs_diff(&ref_out, &buf_out);
    assert!(diff == 0.0, "gemv_f32_moe_metal_buf diff={diff} (expected 0)");
}

// ── moe_grouped_gemm_q4_metal_buf ───────────────────────────────────────────

fn synthetic_q4_k_bytes(n_blocks: usize) -> Vec<u8> {
    // Deterministic synthetic Q4_K_M block stream.
    // Each 144-byte block: 2 bytes d (f16), 2 bytes dmin (f16), 140 bytes nibbles+scales.
    // Use small d so dequant magnitudes stay bounded and accumulation-order
    // differences stay within atol=0 (both paths use same kernel).
    let d_bits = f16::from_f32(0.01).to_bits().to_le_bytes();
    let dmin_bits = f16::from_f32(0.005).to_bits().to_le_bytes();
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        bytes[off..off + 2].copy_from_slice(&d_bits);
        bytes[off + 2..off + 4].copy_from_slice(&dmin_bits);
        // Fill remaining with a deterministic pattern
        for i in 4..144 {
            bytes[off + i] = ((b * 144 + i) & 0xFF) as u8;
        }
    }
    bytes
}

#[test]
fn moe_grouped_gemm_q4_metal_buf_matches_existing() {
    let ctx = make_ctx();
    let rows = 64;
    let cols = 256; // 1 super-block per row
    let blocks = rows * (cols / 256);

    let w_bytes = synthetic_q4_k_bytes(blocks);
    let x: Vec<f32> = linspace(cols, -0.5, 0.5);

    // Reference: existing moe_grouped_gemm_q4_metal (allocates its own x_buf/y_buf)
    let mut ref_out = vec![0.0f32; rows];
    moe_grouped_gemm_q4_metal(&ctx, &w_bytes, rows, cols, &x, &mut ref_out)
        .expect("moe_grouped_gemm_q4_metal");

    // _buf variant: x and y as pre-existing buffers
    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    moe_grouped_gemm_q4_metal_buf(&ctx, &w_bytes, rows, cols, &x_buf, &y_buf)
        .expect("moe_grouped_gemm_q4_metal_buf");

    let y_ptr = y_buf.contents() as *const f32;
    let buf_out: Vec<f32> = unsafe { std::slice::from_raw_parts(y_ptr, rows) }.to_vec();

    let diff = max_abs_diff(&ref_out, &buf_out);
    // Same kernel, same data → bit-identical
    assert!(diff == 0.0, "moe_grouped_gemm_q4_metal_buf diff={diff} (expected 0)");
}

// ── DecodeArena buffer size assertions ───────────────────────────────────────

#[test]
fn arena_buffers_have_correct_sizes() {
    let ctx = make_ctx();
    let hidden = 2048;
    let n_routed_experts = 64;

    let arena = DecodeArena::new(
        &ctx,
        16,   // n_heads
        128,  // qk_nope_head_dim
        64,   // qk_rope_head_dim
        128,  // v_head_dim
        512,  // kv_lora_rank
        hidden,
        512,  // max_seq
        n_routed_experts,
        6,     // top_k_routed
        1408,  // moe_intermediate
        2,     // n_shared_experts
        10944, // ffn_intermediate
        1536,  // q_lora_rank
    );

    let f32_bytes = std::mem::size_of::<f32>();
    assert_eq!(
        arena.x_buf.length() as usize,
        hidden * f32_bytes,
        "x_buf length"
    );
    assert_eq!(
        arena.x_norm_buf.length() as usize,
        hidden * f32_bytes,
        "x_norm_buf length"
    );
    assert_eq!(
        arena.ffn_out_buf.length() as usize,
        hidden * f32_bytes,
        "ffn_out_buf length"
    );
    assert_eq!(
        arena.moe_logits_buf.length() as usize,
        n_routed_experts * f32_bytes,
        "moe_logits_buf length"
    );
    assert_eq!(arena.n_routed_experts, n_routed_experts);
}

#[test]
fn arena_write_x_read_x_roundtrip() {
    let ctx = make_ctx();
    let hidden = 2048;

    let arena = DecodeArena::new(
        &ctx, 16, 128, 64, 128, 512, hidden, 512, 64, 6, 1408, 2, 10944, 1536,
    );

    let data: Vec<f32> = linspace(hidden, -3.0, 3.0);
    arena.write_x(&data);

    let mut readback = vec![0.0f32; hidden];
    arena.read_x(&mut readback);

    // write_x / read_x is a plain memcpy on unified-memory hardware → byte-identical
    assert_eq!(data, readback, "arena write_x/read_x roundtrip failed");
}
