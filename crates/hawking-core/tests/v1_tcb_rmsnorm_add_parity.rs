#![cfg(target_os = "macos")]
use hawking_core::kernels;
use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};
mod common;
use common::*;
fn write_f32_buf(buf: &PinnedBuffer, data: &[f32]) {
    let ptr = buf.contents() as *mut f32;
    unsafe { ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
}
#[test]
fn tcb_rmsnorm_matches_cpu() {
    let h = 2048usize;
    let eps = 1e-6_f32;
    let ctx = ctx();
    let x = fixed_f32(h, 0xABCD_1234);
    let w = fixed_f32(h, 0xDEAD_BEEF);
    let mut cpu_out = vec![0.0f32; h];
    kernels::rmsnorm(&x, &w, eps, &mut cpu_out);
    let x_buf = new_f32_buf(ctx, &x);
    let w_buf = new_f32_buf(ctx, &w);
    let out_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &w_buf, eps, h, &out_buf)
            .expect("rmsnorm_metal_buf_tcb");
        tcb.commit_and_wait().expect("commit");
    }
    let gpu_out = read_f32_buf(&out_buf, h);
    let diff = max_abs_diff(&cpu_out, &gpu_out);
    assert!(diff < 1e-5, "rmsnorm TCB vs CPU diff {diff:.2e} >= 1e-5");
}
#[test]
fn tcb_add_inplace_matches_cpu() {
    let h = 2048usize;
    let ctx = ctx();
    let mut a_cpu = fixed_f32(h, 0xCAFE_BABE);
    let b = fixed_f32(h, 0x1234_5678);
    kernels::add_inplace(&mut a_cpu, &b);
    let a_buf = new_f32_buf(ctx, &fixed_f32(h, 0xCAFE_BABE));
    let b_buf = new_f32_buf(ctx, &b);
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::add_inplace_metal_tcb(&mut tcb, &a_buf, &b_buf, h).expect("add_inplace_metal_tcb");
        tcb.commit_and_wait().expect("commit");
    }
    let gpu_a = read_f32_buf(&a_buf, h);
    let diff = max_abs_diff(&a_cpu, &gpu_a);
    assert!(
        diff < 1e-6,
        "add_inplace TCB vs CPU diff {diff:.2e} >= 1e-6"
    );
}
#[test]
fn tcb_staggered_loop_matches_cpu() {
    const N_LAYERS: usize = 4;
    let h = 2048usize;
    let eps = 1e-6_f32;
    let ctx = ctx();
    let x_init = fixed_f32(h, 0x1111_2222);
    let deltas: Vec<Vec<f32>> = (0..N_LAYERS)
        .map(|i| fixed_f32(h, 0xAAAA_0000 + i as u64))
        .collect();
    let norms: Vec<Vec<f32>> = (0..N_LAYERS)
        .map(|i| fixed_f32(h, 0xBBBB_0000 + i as u64))
        .collect();
    let mut x_cpu = x_init.clone();
    let mut norm_outs_cpu = vec![vec![0.0f32; h]; N_LAYERS];
    for li in 0..N_LAYERS {
        kernels::add_inplace(&mut x_cpu, &deltas[li]);
        kernels::rmsnorm(&x_cpu, &norms[li], eps, &mut norm_outs_cpu[li]);
    }
    let x_final_cpu = x_cpu;
    let x_buf = new_f32_buf(ctx, &x_init);
    let delta_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
    let out_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
    let norm_bufs: Vec<PinnedBuffer> = norms.iter().map(|n| new_f32_buf(ctx, n)).collect();
    let mut norm_outs_gpu = vec![vec![0.0f32; h]; N_LAYERS];
    for li in 0..N_LAYERS {
        write_f32_buf(&delta_buf, &deltas[li]);
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::add_inplace_metal_tcb(&mut tcb, &x_buf, &delta_buf, h)
                .expect("add_inplace_metal_tcb");
            kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &norm_bufs[li], eps, h, &out_buf)
                .expect("rmsnorm_metal_buf_tcb");
            tcb.commit_and_wait().expect("commit");
        }
        norm_outs_gpu[li] = read_f32_buf(&out_buf, h);
    }
    let x_final_gpu = read_f32_buf(&x_buf, h);
    let x_diff = max_abs_diff(&x_final_cpu, &x_final_gpu);
    assert!(x_diff < 1e-5, "x_final diff {x_diff:.2e} >= 1e-5");
    for li in 0..N_LAYERS {
        let d = max_abs_diff(&norm_outs_cpu[li], &norm_outs_gpu[li]);
        assert!(d < 1e-5, "layer {li} norm_out diff {d:.2e} >= 1e-5");
    }
}
