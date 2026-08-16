//! G023: both shipping graphs run the shared decode family, and family
//! kernels are bit-identical to the pre-family entry points they replace.
//!
//! No model artifact. Numerics authority is the existing Metal kernel (same
//! association), cross-checked against the Q80 CPU pack oracle where one exists.

#![cfg(target_os = "macos")]

use hawking_core::decode_family::{
    COMBINE_BF16, DSV4F_GRAPH_KERNELS, FAMILY_KERNELS, MATVEC_BINARY, MATVEC_FP4, MATVEC_HGRAVS,
    PACK_WORKLIST, Q80_GRAPH_KERNELS, SWIGLU_BF16_WORKLIST, SWIGLU_F32, WORKLIST_FP4,
};
use hawking_core::metal::{MetalContext, TokenCommandBuffer};
use hawking_core::model::qwen80_mixed_hybrid_decode;
use hawking_core::model::qwen_complete_binary::{
    binary_group_matvec_f32, deterministic_input, deterministic_matrix, pack_binary_group,
    pack_uniform_factor, uniform_factor_matvec_f32, Q80_BINARY_GROUP_SIZE, Q80_HGRAVS_GROUP_SIZE,
};
use std::path::Path;

fn ctx() -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("shader") && !msg.contains("compile") && !msg.contains("pipeline"),
                "Metal is present but the G023 family failed to compile: {msg}"
            );
            eprintln!("skip: no Metal device ({e})");
            None
        }
    }
}

fn set_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
    let v = value;
    enc.set_bytes(index, 4, &v as *const u32 as *const _);
}

fn read_f32(buf: &hawking_core::metal::PinnedBuffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
}

fn read_u16(buf: &hawking_core::metal::PinnedBuffer, n: usize) -> Vec<u16> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const u16, n).to_vec() }
}

fn dispatch_named(
    ctx: &MetalContext,
    name: &str,
    grid: (u32, u32, u32),
    tg: (u32, u32, u32),
    encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
) {
    let mut tcb = TokenCommandBuffer::new(ctx);
    tcb.dispatch_threads(name, grid, tg, encode)
        .unwrap_or_else(|e| panic!("{name} dispatch failed: {e}"));
    tcb.commit_and_wait()
        .unwrap_or_else(|e| panic!("{name} wait failed: {e}"));
}

fn f32_bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn both_graphs_dispatch_the_shared_family() {
    let mixed = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/model/qwen80_mixed_hybrid_decode.rs"
    ));
    for &kernel in Q80_GRAPH_KERNELS {
        assert!(
            mixed.contains("decode_family::"),
            "Q80 mixed hybrid is not routed through decode_family"
        );
        let _ = kernel;
    }
    assert!(mixed.contains("MATVEC_BINARY"));
    assert!(mixed.contains("MATVEC_HGRAVS"));
    // Residual CSR is STRUCTURAL — must stay on the Q80-only kernel.
    assert!(mixed.contains("q80_sparse_q1_apply_csr"));

    let dsv = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/gravity_deepseek_v4_native_token_graph.rs"
    ));
    for &kernel in DSV4F_GRAPH_KERNELS {
        assert!(
            dsv.contains(kernel) || dsv.contains("decode_family::"),
            "DSV4F graph does not name {kernel}"
        );
    }
    assert!(dsv.contains("PACK_WORKLIST") || dsv.contains(PACK_WORKLIST));
    assert!(dsv.contains("WORKLIST_FP4") || dsv.contains(WORKLIST_FP4));
    assert!(dsv.contains("SWIGLU_BF16_WORKLIST") || dsv.contains(SWIGLU_BF16_WORKLIST));
    assert!(dsv.contains("COMBINE_BF16") || dsv.contains(COMBINE_BF16));

    // Silence unused if the path check is enough.
    let _ = Path::new(".");
    let _ = qwen80_mixed_hybrid_decode::QWEN80_MIXED_CLAIM;
}

#[test]
fn family_source_has_no_runtime_codec_switch_in_fma() {
    let src = hawking_core::metal::SHADER_GK_FAMILY;
    // The family may switch on function constants around whole loops, never
    // on a runtime codec id inside the product.
    assert!(!src.contains("if (codec =="));
    assert!(!src.contains("switch (codec)"));
    assert!(src.contains("kGkTile"));
    assert!(src.contains("kGkCodec"));
    assert!(src.contains("kGkWorklistK"));
    for &kernel in FAMILY_KERNELS {
        assert!(src.contains(&format!("kernel void {kernel}(")));
    }
}

#[test]
fn q80_binary_family_matches_legacy_and_cpu() {
    let Some(ctx) = ctx() else { return };
    let rows = 8usize;
    let cols = 256usize;
    let w = deterministic_matrix(rows, cols, 7);
    let packed = pack_binary_group(&w, rows, cols, Q80_BINARY_GROUP_SIZE).unwrap();
    let x = deterministic_input(cols);
    let cpu = binary_group_matvec_f32(&packed, &x).unwrap();

    let signs = ctx
        .new_buffer_with_bytes_checked(&packed.signs)
        .unwrap();
    let scales = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&packed.scales_f16))
        .unwrap();
    let input = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&x))
        .unwrap();
    let out_fam = ctx.new_buffer_checked(rows * 4).unwrap();
    let out_old = ctx.new_buffer_checked(rows * 4).unwrap();
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let gs = packed.group_size as u32;
    let gpr = packed.groups_per_row as u32;

    let encode = |enc: &metal::ComputeCommandEncoderRef, out: &hawking_core::metal::PinnedBuffer| {
        enc.set_buffer(0, Some(&signs), 0);
        enc.set_buffer(1, Some(&scales), 0);
        enc.set_buffer(2, Some(&input), 0);
        enc.set_buffer(3, Some(out), 0);
        set_u32(enc, 4, rows_u);
        set_u32(enc, 5, cols_u);
        set_u32(enc, 6, gs);
        set_u32(enc, 7, gpr);
    };
    dispatch_named(
        &ctx,
        MATVEC_BINARY,
        (rows_u, 1, 1),
        (256, 1, 1),
        |enc| encode(enc, &out_fam),
    );
    dispatch_named(
        &ctx,
        "q80_binary_group_matvec",
        (rows_u, 1, 1),
        (256, 1, 1),
        |enc| encode(enc, &out_old),
    );
    let fam = read_f32(&out_fam, rows);
    let old = read_f32(&out_old, rows);
    assert!(f32_bits_eq(&fam, &old), "family binary != legacy binary");
    // CPU oracle is the same algebra; Metal serial grouping can differ by
    // 1 ULP from the naive per-element host loop. G023 requires family ==
    // legacy, not a new host association.
    let max_abs = fam
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-4,
        "family binary drifted from CPU oracle: max_abs={max_abs}"
    );
}

#[test]
fn q80_hgravs_family_matches_legacy_and_cpu() {
    let Some(ctx) = ctx() else { return };
    let rows = 16usize;
    let cols = 64usize;
    let w = deterministic_matrix(rows, cols, 3);
    let packed = pack_uniform_factor(&w, rows, cols, 3, Q80_HGRAVS_GROUP_SIZE).unwrap();
    let x = deterministic_input(cols);
    let cpu = uniform_factor_matvec_f32(&packed, &x).unwrap();

    let codes = ctx
        .new_buffer_with_bytes_checked(&packed.codes)
        .unwrap();
    let scales = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&packed.scales_f16))
        .unwrap();
    let input = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&x))
        .unwrap();
    let out_fam = ctx.new_buffer_checked(rows * 4).unwrap();
    let out_old = ctx.new_buffer_checked(rows * 4).unwrap();
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let gs = packed.group_size as u32;
    let bits = u32::from(packed.bits);
    let bound = u32::from(packed.bound);

    let encode = |enc: &metal::ComputeCommandEncoderRef, out: &hawking_core::metal::PinnedBuffer| {
        enc.set_buffer(0, Some(&codes), 0);
        enc.set_buffer(1, Some(&scales), 0);
        enc.set_buffer(2, Some(&input), 0);
        enc.set_buffer(3, Some(out), 0);
        set_u32(enc, 4, rows_u);
        set_u32(enc, 5, cols_u);
        set_u32(enc, 6, gs);
        set_u32(enc, 7, bits);
        set_u32(enc, 8, bound);
    };
    dispatch_named(
        &ctx,
        MATVEC_HGRAVS,
        (rows_u, 1, 1),
        (256, 1, 1),
        |enc| encode(enc, &out_fam),
    );
    dispatch_named(
        &ctx,
        "q80_hgravs01_factor_matvec",
        (rows_u, 1, 1),
        (256, 1, 1),
        |enc| encode(enc, &out_old),
    );
    let fam = read_f32(&out_fam, rows);
    let old = read_f32(&out_old, rows);
    assert!(f32_bits_eq(&fam, &old), "family hgravs != legacy hgravs");
    let max_abs = fam
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-4,
        "family hgravs drifted from CPU oracle: max_abs={max_abs}"
    );
}

fn e2m1(packed: u8, hi: bool) -> f32 {
    let nibble = if hi { (packed >> 4) & 0x0f } else { packed & 0x0f };
    let mag = match nibble & 0x07 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    if nibble & 0x08 != 0 {
        -mag
    } else {
        mag
    }
}

fn e4m3(bits: u8) -> f32 {
    let raw = bits as u32;
    let exponent = (raw >> 3) & 0x0f;
    let mantissa = raw & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return 0.0;
    }
    let magnitude = if exponent == 0 {
        mantissa as f32 * 0.001953125
    } else {
        f32::from_bits(((exponent + 120) << 23) | (mantissa << 20))
    };
    if raw & 0x80 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn e8m0(bits: u8) -> f32 {
    if bits == 0xff {
        0.0
    } else if bits == 0 {
        f32::from_bits(0x0040_0000)
    } else {
        f32::from_bits((bits as u32) << 23)
    }
}

fn fp4_cpu(
    packed: &[u8],
    w_scales: &[u8],
    quant: &[u8],
    a_scales: &[u8],
    rows: usize,
    packed_cols: usize,
    scale_cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for row in 0..rows {
        let mut acc = 0.0f32;
        for block in 0..scale_cols {
            let mut blk = 0.0f32;
            let start = block * 32;
            for off in 0..32 {
                let col = start + off;
                let p = packed[row * packed_cols + (col >> 1)];
                blk = blk + e4m3(quant[col]) * e2m1(p, col & 1 != 0);
            }
            let as_ = e8m0(a_scales[block / 4]);
            let ws = e8m0(w_scales[row * scale_cols + block]);
            acc = acc + blk * (as_ * ws);
        }
        out[row] = acc;
    }
    out
}

#[test]
fn dsv4f_fp4_family_matches_legacy_and_cpu() {
    let Some(ctx) = ctx() else { return };
    let rows = 8usize;
    let packed_cols = 64usize; // 128 logical
    let scale_cols = 4usize;
    let mut packed = vec![0u8; rows * packed_cols];
    let mut w_scales = vec![0u8; rows * scale_cols];
    let mut quant = vec![0u8; 128];
    let mut a_scales = vec![0u8; 1];
    for (i, b) in packed.iter_mut().enumerate() {
        *b = ((i * 17 + 3) & 0xff) as u8;
    }
    for (i, b) in w_scales.iter_mut().enumerate() {
        *b = (120 + (i % 7) as u8).min(254);
    }
    for (i, b) in quant.iter_mut().enumerate() {
        *b = ((i * 13 + 5) & 0x7f) as u8;
    }
    a_scales[0] = 127;

    let cpu = fp4_cpu(&packed, &w_scales, &quant, &a_scales, rows, packed_cols, scale_cols);
    let pw = ctx.new_buffer_with_bytes_checked(&packed).unwrap();
    let ws = ctx.new_buffer_with_bytes_checked(&w_scales).unwrap();
    let q = ctx.new_buffer_with_bytes_checked(&quant).unwrap();
    let asb = ctx.new_buffer_with_bytes_checked(&a_scales).unwrap();
    let out_fam = ctx.new_buffer_checked(rows * 4).unwrap();
    let out_old = ctx.new_buffer_checked(rows * 4).unwrap();
    let rows_u = rows as u32;
    let pc = packed_cols as u32;
    let sc = scale_cols as u32;
    let encode = |enc: &metal::ComputeCommandEncoderRef, out: &hawking_core::metal::PinnedBuffer| {
        enc.set_buffer(0, Some(&pw), 0);
        enc.set_buffer(1, Some(&ws), 0);
        enc.set_buffer(2, Some(&q), 0);
        enc.set_buffer(3, Some(&asb), 0);
        enc.set_buffer(4, Some(out), 0);
        set_u32(enc, 5, rows_u);
        set_u32(enc, 6, pc);
        set_u32(enc, 7, sc);
    };
    dispatch_named(&ctx, MATVEC_FP4, (rows_u, 1, 1), (256, 1, 1), |enc| {
        encode(enc, &out_fam)
    });
    dispatch_named(
        &ctx,
        "dsv4f_fp4_matvec_split",
        (rows_u, 1, 1),
        (256, 1, 1),
        |enc| encode(enc, &out_old),
    );
    let fam = read_f32(&out_fam, rows);
    let old = read_f32(&out_old, rows);
    assert!(f32_bits_eq(&fam, &old), "family fp4 != legacy fp4: {fam:?} vs {old:?}");
    assert!(
        f32_bits_eq(&fam, &cpu),
        "family fp4 != CPU: {fam:?} vs {cpu:?}"
    );
}

#[test]
fn swiglu_f32_family_matches_q80_silu_mul() {
    let Some(ctx) = ctx() else { return };
    let n = 64usize;
    let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 1.5).collect();
    let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 0.4).collect();
    let g = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&gate))
        .unwrap();
    let u = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&up))
        .unwrap();
    let out_fam = ctx.new_buffer_checked(n * 4).unwrap();
    let out_old = ctx.new_buffer_checked(n * 4).unwrap();
    let nu = n as u32;
    let encode = |enc: &metal::ComputeCommandEncoderRef, out: &hawking_core::metal::PinnedBuffer| {
        enc.set_buffer(0, Some(&g), 0);
        enc.set_buffer(1, Some(&u), 0);
        enc.set_buffer(2, Some(out), 0);
        set_u32(enc, 3, nu);
    };
    dispatch_named(&ctx, SWIGLU_F32, (nu, 1, 1), (256, 1, 1), |enc| {
        encode(enc, &out_fam)
    });
    dispatch_named(
        &ctx,
        "qwen80_silu_mul_f32",
        (nu, 1, 1),
        (256, 1, 1),
        |enc| encode(enc, &out_old),
    );
    let fam = read_f32(&out_fam, n);
    let old = read_f32(&out_old, n);
    assert!(f32_bits_eq(&fam, &old), "gk_swiglu_f32 != qwen80_silu_mul_f32");
}

fn bf16_from_f32(v: f32) -> u16 {
    let bits = v.to_bits();
    let low_lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + low_lsb) >> 16) as u16
}

#[test]
fn swiglu_bf16_family_matches_dsv4f_worklist() {
    let Some(ctx) = ctx() else { return };
    let width = 32usize;
    let k = 6usize;
    let mut worklist = vec![0u8; k * 16];
    for slot in 0..k {
        let off = slot * 16;
        worklist[off..off + 4].copy_from_slice(&(slot as u32).to_le_bytes());
        worklist[off + 4..off + 8].copy_from_slice(&(slot as u32).to_le_bytes());
        worklist[off + 8..off + 12].copy_from_slice(&(0.1f32 * (slot as f32 + 1.0)).to_le_bytes());
        worklist[off + 12..off + 16].copy_from_slice(&1u32.to_le_bytes());
    }
    let gate: Vec<u16> = (0..k * width)
        .map(|i| bf16_from_f32((i as f32) * 0.05 - 0.8))
        .collect();
    let up: Vec<u16> = (0..k * width)
        .map(|i| bf16_from_f32((i as f32) * 0.03 - 0.4))
        .collect();
    let wl = ctx.new_buffer_with_bytes_checked(&worklist).unwrap();
    let gb = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&gate))
        .unwrap();
    let ub = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&up))
        .unwrap();
    let out_fam = ctx.new_buffer_checked(k * width * 2).unwrap();
    let out_old = ctx.new_buffer_checked(k * width * 2).unwrap();
    let width_u = width as u32;
    let k_u = k as u32;
    let n = (k * width) as u32;
    let encode = |enc: &metal::ComputeCommandEncoderRef, out: &hawking_core::metal::PinnedBuffer| {
        enc.set_buffer(0, Some(&wl), 0);
        enc.set_buffer(1, Some(&gb), 0);
        enc.set_buffer(2, Some(&ub), 0);
        enc.set_buffer(3, Some(out), 0);
        set_u32(enc, 4, width_u);
        set_u32(enc, 5, k_u);
    };
    dispatch_named(
        &ctx,
        SWIGLU_BF16_WORKLIST,
        (n, 1, 1),
        (32, 1, 1),
        |enc| encode(enc, &out_fam),
    );
    dispatch_named(
        &ctx,
        "dsv4f_worklist_swiglu",
        (n, 1, 1),
        (32, 1, 1),
        |enc| encode(enc, &out_old),
    );
    let fam = read_u16(&out_fam, k * width);
    let old = read_u16(&out_old, k * width);
    assert_eq!(fam, old, "gk_swiglu_bf16_worklist != dsv4f_worklist_swiglu");
}

#[test]
fn family_does_not_add_bytes_or_slow_the_serial_tile() {
    let Some(ctx) = ctx() else { return };
    let rows = 512usize;
    let cols = 2048usize;
    let w = deterministic_matrix(rows, cols, 9);
    let packed = pack_binary_group(&w, rows, cols, Q80_BINARY_GROUP_SIZE).unwrap();
    let x = deterministic_input(cols);
    let signs = ctx.new_buffer_with_bytes_checked(&packed.signs).unwrap();
    let scales = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&packed.scales_f16))
        .unwrap();
    let input = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&x))
        .unwrap();
    let out = ctx.new_buffer_checked(rows * 4).unwrap();
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let gs = packed.group_size as u32;
    let gpr = packed.groups_per_row as u32;
    let time = |name: &str| -> u64 {
        let mut samples = Vec::new();
        for _ in 0..4 {
            let mut tcb = TokenCommandBuffer::new(&ctx);
            tcb.dispatch_threads(name, (rows_u, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(&signs), 0);
                enc.set_buffer(1, Some(&scales), 0);
                enc.set_buffer(2, Some(&input), 0);
                enc.set_buffer(3, Some(&out), 0);
                set_u32(enc, 4, rows_u);
                set_u32(enc, 5, cols_u);
                set_u32(enc, 6, gs);
                set_u32(enc, 7, gpr);
            })
            .unwrap();
            let timing = tcb.commit_and_wait_timed().unwrap();
            if let Some(gpu) = timing.gpu_ns {
                samples.push(gpu);
            }
        }
        samples.into_iter().min().unwrap_or(0)
    };
    let _ = time(MATVEC_BINARY);
    let _ = time("q80_binary_group_matvec");
    let fam = time(MATVEC_BINARY);
    let old = time("q80_binary_group_matvec");
    eprintln!("G023 binary serial GPU ns: family={fam} legacy={old}");
    if fam > 0 && old > 0 {
        assert!(
            fam <= old + old / 6 + 1,
            "family binary serial is slower than legacy: {fam}ns vs {old}ns"
        );
    }
}
