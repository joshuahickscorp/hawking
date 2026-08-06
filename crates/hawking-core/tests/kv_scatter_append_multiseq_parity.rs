#![cfg(target_os = "macos")]
use hawking_core::kernels;
use hawking_core::metal::TokenCommandBuffer;
mod common;
use common::*;
#[allow(clippy::too_many_arguments)]
fn scatter_per_slot(
    src_k: &[f32],
    src_v: &[f32],
    regions: &[usize],
    positions: &[usize],
    kv_dim: usize,
    slot_stride: usize,
    layer_off: usize,
    cache_elems: usize,
) -> (Vec<f32>, Vec<f32>) {
    let ctx = ctx();
    let b = regions.len();
    let kbuf = new_f32_buf(ctx, &vec![0.0f32; cache_elems]);
    let vbuf = new_f32_buf(ctx, &vec![0.0f32; cache_elems]);
    let sk = new_f32_buf(ctx, src_k);
    let sv = new_f32_buf(ctx, src_v);
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        for bi in 0..b {
            let dst_off = layer_off + regions[bi] * slot_stride + positions[bi] * kv_dim;
            kernels::memcpy_f32_off_tcb(&mut tcb, &sk, &kbuf, bi * kv_dim, dst_off, kv_dim)
                .unwrap();
            kernels::memcpy_f32_off_tcb(&mut tcb, &sv, &vbuf, bi * kv_dim, dst_off, kv_dim)
                .unwrap();
        }
        tcb.commit_and_wait().unwrap();
    }
    (
        read_f32_buf(&kbuf, cache_elems),
        read_f32_buf(&vbuf, cache_elems),
    )
}
#[allow(clippy::too_many_arguments)]
fn scatter_batched(
    src_k: &[f32],
    src_v: &[f32],
    regions: &[usize],
    positions: &[usize],
    kv_dim: usize,
    slot_stride: usize,
    layer_off: usize,
    cache_elems: usize,
) -> (Vec<f32>, Vec<f32>) {
    let ctx = ctx();
    let b = regions.len();
    let kbuf = new_f32_buf(ctx, &vec![0.0f32; cache_elems]);
    let vbuf = new_f32_buf(ctx, &vec![0.0f32; cache_elems]);
    let sk = new_f32_buf(ctx, src_k);
    let sv = new_f32_buf(ctx, src_v);
    let reg_bytes: Vec<u8> = regions
        .iter()
        .flat_map(|&r| (r as u32).to_le_bytes())
        .collect();
    let pos_bytes: Vec<u8> = positions
        .iter()
        .flat_map(|&p| (p as u32).to_le_bytes())
        .collect();
    let reg_buf = ctx.new_buffer_with_bytes(&reg_bytes);
    let pos_buf = ctx.new_buffer_with_bytes(&pos_bytes);
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::kv_scatter_append_multiseq_tcb(
            &mut tcb,
            &sk,
            &sv,
            &kbuf,
            &vbuf,
            &reg_buf,
            &pos_buf,
            kv_dim,
            b,
            slot_stride,
            layer_off,
        )
        .unwrap();
        tcb.commit_and_wait().unwrap();
    }
    (
        read_f32_buf(&kbuf, cache_elems),
        read_f32_buf(&vbuf, cache_elems),
    )
}
#[test]
fn kv_scatter_append_multiseq_matches_per_slot() {
    let kv_dim = 8usize;
    let max_seq = 4usize;
    let max_batch = 4usize;
    let slot_stride = max_seq * kv_dim; // 32 elems per slot per layer
    let layer_kv_stride = max_batch * slot_stride; // one layer of cache = 128 elems
    let regions = [3usize, 1, 0, 2];
    let positions = [2usize, 0, 3, 1];
    let b = regions.len();
    let src_k = fixed_f32(b * kv_dim, 0x1111_2222_3333_4444);
    let src_v = fixed_f32(b * kv_dim, 0x5555_6666_7777_8888);
    {
        let cache_elems = layer_kv_stride;
        let (ek, ev) = scatter_per_slot(
            &src_k,
            &src_v,
            &regions,
            &positions,
            kv_dim,
            slot_stride,
            0,
            cache_elems,
        );
        let (ak, av) = scatter_batched(
            &src_k,
            &src_v,
            &regions,
            &positions,
            kv_dim,
            slot_stride,
            0,
            cache_elems,
        );
        assert_eq!(max_abs_diff(&ek, &ak), 0.0, "layer0 K: batched != per-slot");
        assert_eq!(max_abs_diff(&ev, &av), 0.0, "layer0 V: batched != per-slot");
    }
    {
        let layer_off = layer_kv_stride; // li = 1
        let cache_elems = 2 * layer_kv_stride; // room for 2 layers
        let (ek, ev) = scatter_per_slot(
            &src_k,
            &src_v,
            &regions,
            &positions,
            kv_dim,
            slot_stride,
            layer_off,
            cache_elems,
        );
        let (ak, av) = scatter_batched(
            &src_k,
            &src_v,
            &regions,
            &positions,
            kv_dim,
            slot_stride,
            layer_off,
            cache_elems,
        );
        assert_eq!(max_abs_diff(&ek, &ak), 0.0, "layer1 K: batched != per-slot");
        assert_eq!(max_abs_diff(&ev, &av), 0.0, "layer1 V: batched != per-slot");
    }
}
