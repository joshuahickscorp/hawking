//! `mha_decode_f32` must never return a confident wrong answer.
//!
//! It indexes `scores[seq_len]` in THREADGROUP memory, so its shared-memory
//! request grows with the sequence. Past the device limit Metal returns NO
//! ERROR: the dispatch completes and the numbers are garbage. Measured on an
//! Apple M3 Ultra (max_threadgroup_memory_length 32,768 B) before the guard,
//! against the flash kernel on identical inputs:
//!
//!     seq  8192   34,816 B   over, still agreed to 4.8e-7
//!     seq 16384   67,584 B   DIVERGED, relative error 1.65e2
//!     seq 32768  133,120 B   DIVERGED, relative error 3.08e2
//!
//! A hard ceiling would be safe. Silently returning nonsense is not. Over-budget
//! requests now route to the flash kernel, whose `scores_tile` is constant and
//! which takes identical arguments.

#![cfg(target_os = "macos")]

use hawking_core::kernels::{mha_decode_f32_tcb, mha_decode_flash_f32_tcb};
use hawking_core::metal::{MetalContext, TokenCommandBuffer};

const HD: usize = 256;
const NH: usize = 24;
const NKV: usize = 4;

fn fixture(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn run(ctx: &MetalContext, seq: usize, flash: bool) -> Vec<f32> {
    let qv = fixture(NH * HD, 1);
    let kv = fixture(seq * NKV * HD, 2);
    let vv = fixture(seq * NKV * HD, 3);
    let q = ctx.new_buffer_with_bytes(as_bytes(&qv));
    let k = ctx.new_buffer_with_bytes(as_bytes(&kv));
    let v = ctx.new_buffer_with_bytes(as_bytes(&vv));
    let o = ctx.new_buffer(NH * HD * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    if flash {
        mha_decode_flash_f32_tcb(&mut tcb, &q, &k, 0, &v, 0, &o, seq, HD, NH, NKV)
    } else {
        mha_decode_f32_tcb(&mut tcb, &q, &k, 0, &v, 0, &o, seq, HD, NH, NKV)
    }
    .expect("encode");
    tcb.commit_and_wait_timed().expect("run");
    unsafe { std::slice::from_raw_parts(o.contents() as *const f32, NH * HD) }.to_vec()
}

fn max_rel(a: &[f32], b: &[f32]) -> f64 {
    let scale = b
        .iter()
        .map(|x| x.abs() as f64)
        .fold(0.0f64, f64::max)
        .max(1e-6);
    a.iter()
        .zip(b)
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs() / scale)
        .fold(0.0f64, f64::max)
}

#[test]
fn over_budget_sequences_do_not_return_garbage() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return, // no GPU in this environment
    };
    // 16384 and 32768 both exceed 32,768 B of threadgroup memory at the default
    // tg_size of 512, and both diverged by two orders of magnitude before the guard.
    for seq in [16384usize, 32768] {
        let plain = run(&ctx, seq, false);
        let flash = run(&ctx, seq, true);
        let rel = max_rel(&plain, &flash);
        assert!(
            rel < 1e-3,
            "seq {seq}: mha_decode_f32 diverged from flash by {rel:.3e} -- \
             an over-budget dispatch returned a wrong answer instead of refusing"
        );
    }
}

#[test]
fn under_budget_sequences_still_use_the_original_kernel_and_agree() {
    // The negative control. If the guard rerouted EVERYTHING to flash, the test
    // above would pass vacuously by comparing flash against itself. These sizes
    // fit in threadgroup memory, so the two genuinely different kernels run and
    // must still agree.
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    for seq in [2048usize, 7680] {
        let plain = run(&ctx, seq, false);
        let flash = run(&ctx, seq, true);
        let rel = max_rel(&plain, &flash);
        assert!(
            rel < 1e-3,
            "seq {seq}: kernels disagree below the limit by {rel:.3e}"
        );
    }
}
