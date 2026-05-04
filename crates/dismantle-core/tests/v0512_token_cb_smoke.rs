//! v0.5.12 smoke test: TokenCommandBuffer fuses two kernel dispatches.
//!
//! Verifies that encoding rmsnorm + add_inplace into a single
//! TokenCommandBuffer produces the same outputs as two sequential
//! ctx.dispatch_threads calls.

#![cfg(target_os = "macos")]

use dismantle_core::metal::{MetalContext, TokenCommandBuffer};
use half::f16;

const TG_SIZE: u32 = 256;

fn make_ctx() -> MetalContext {
    MetalContext::new().expect("Metal device")
}

fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    let f16v: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
    bytemuck::cast_slice::<f16, u8>(&f16v).to_vec()
}

fn f16_buf_to_f32(ptr: *const f16, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(ptr, n) }
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice::<f32, u8>(v).to_vec()
}

fn f32_buf_read(ptr: *const f32, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

#[test]
fn token_command_buffer_matches_sequential() {
    let ctx = make_ctx();
    let n = TG_SIZE as usize;  // 256 elements
    let eps = 1e-5f32;

    // ── prepare input data ────────────────────────────────────────────────────
    let x_f32: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.05).sin()) * 2.0).collect();
    let w_f32: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.001)).collect();

    let a_f32: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let b_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();

    // ── sequential path ───────────────────────────────────────────────────────
    let x_bytes   = f32_to_f16_bytes(&x_f32);
    let w_bytes   = f32_to_f16_bytes(&w_f32);
    let rn_out_seq = ctx.new_buffer(n * std::mem::size_of::<f16>());

    let x_buf_s  = ctx.new_buffer_with_bytes(&x_bytes);
    let w_buf_s  = ctx.new_buffer_with_bytes(&w_bytes);
    let hidden_u32 = n as u32;
    let shmem = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

    // Sequential: rmsnorm dispatch.
    ctx.dispatch_threads("rmsnorm", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
        enc.set_buffer(0, Some(&x_buf_s), 0);
        enc.set_buffer(1, Some(&w_buf_s), 0);
        enc.set_buffer(2, Some(&rn_out_seq), 0);
        enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        enc.set_threadgroup_memory_length(0, shmem);
    }).expect("seq rmsnorm");

    let a_bytes      = f32_bytes(&a_f32);
    let b_bytes      = f32_bytes(&b_f32);
    let a_buf_s      = ctx.new_buffer_with_bytes(&a_bytes);
    let b_buf_s      = ctx.new_buffer_with_bytes(&b_bytes);
    let n_u32        = n as u32;
    let n_tg         = (n_u32 + TG_SIZE - 1) / TG_SIZE;

    // Sequential: add_inplace dispatch.
    ctx.dispatch_threads("add_inplace", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
        enc.set_buffer(0, Some(&a_buf_s), 0);
        enc.set_buffer(1, Some(&b_buf_s), 0);
        enc.set_bytes(2, 4, &n_u32 as *const u32 as *const _);
    }).expect("seq add_inplace");

    // Read sequential results.
    let rn_seq = f16_buf_to_f32(rn_out_seq.contents() as *const f16, n);
    let ai_seq = f32_buf_read(a_buf_s.contents() as *const f32, n);

    // ── TCB path ──────────────────────────────────────────────────────────────
    let x_buf_t  = ctx.new_buffer_with_bytes(&x_bytes);
    let w_buf_t  = ctx.new_buffer_with_bytes(&w_bytes);
    let rn_out_tcb = ctx.new_buffer(n * std::mem::size_of::<f16>());
    let a_buf_t  = ctx.new_buffer_with_bytes(&a_bytes);
    let b_buf_t  = ctx.new_buffer_with_bytes(&b_bytes);

    {
        let mut tcb = TokenCommandBuffer::new(&ctx);

        tcb.dispatch_threads("rmsnorm", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&x_buf_t), 0);
            enc.set_buffer(1, Some(&w_buf_t), 0);
            enc.set_buffer(2, Some(&rn_out_tcb), 0);
            enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);
            enc.set_threadgroup_memory_length(0, shmem);
        }).expect("tcb rmsnorm");

        tcb.dispatch_threads("add_inplace", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&a_buf_t), 0);
            enc.set_buffer(1, Some(&b_buf_t), 0);
            enc.set_bytes(2, 4, &n_u32 as *const u32 as *const _);
        }).expect("tcb add_inplace");

        tcb.commit_and_wait().expect("tcb commit");
    }

    // Read TCB results.
    let rn_tcb = f16_buf_to_f32(rn_out_tcb.contents() as *const f16, n);
    let ai_tcb = f32_buf_read(a_buf_t.contents() as *const f32, n);

    // ── compare ───────────────────────────────────────────────────────────────
    assert_eq!(rn_seq.len(), rn_tcb.len(), "rmsnorm output length mismatch");
    for (i, (&s, &t)) in rn_seq.iter().zip(rn_tcb.iter()).enumerate() {
        assert_eq!(s, t, "rmsnorm[{i}]: seq={s} tcb={t}");
    }

    assert_eq!(ai_seq.len(), ai_tcb.len(), "add_inplace output length mismatch");
    for (i, (&s, &t)) in ai_seq.iter().zip(ai_tcb.iter()).enumerate() {
        assert_eq!(s, t, "add_inplace[{i}]: seq={s} tcb={t}");
    }
}

#[test]
fn token_command_buffer_drop_commits() {
    // Verify that dropping a TCB without commit_and_wait doesn't panic or leak.
    let ctx = make_ctx();
    let n = 64usize;
    let a_f32: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_f32: Vec<f32> = vec![1.0f32; n];
    let a_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&a_f32));
    let b_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&b_f32));
    let n_u32 = n as u32;
    let n_tg  = (n_u32 + TG_SIZE - 1) / TG_SIZE;

    {
        // Drop without explicit commit: Drop impl should commit cleanly.
        let mut tcb = TokenCommandBuffer::new(&ctx);
        tcb.dispatch_threads("add_inplace", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&b_buf), 0);
            enc.set_bytes(2, 4, &n_u32 as *const u32 as *const _);
        }).expect("tcb add_inplace");
        // Dropped here — Drop impl commits.
    }

    // After drop, the GPU work should have completed; a_buf should be updated.
    let result = unsafe { std::slice::from_raw_parts(a_buf.contents() as *const f32, n) };
    for i in 0..n {
        let expected = i as f32 + 1.0;
        assert!((result[i] - expected).abs() < 1e-6, "drop[{i}]: got={} expected={expected}", result[i]);
    }
}
