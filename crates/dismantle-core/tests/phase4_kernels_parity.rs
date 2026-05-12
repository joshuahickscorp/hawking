//! Phase 4 wedges 4a/4b kernel parity. Both kernels are pure additions;
//! we verify their math against the CPU reference.

#[cfg(target_os = "macos")]
mod metal_tests {
    use dismantle_core::metal::MetalContext;
    use dismantle_core::kernels::{add_inplace, add_inplace_metal, gpu_argmax_logits_metal};

    fn cpu_argmax(logits: &[f32]) -> u32 {
        let mut best = 0u32;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv { best = i as u32; bv = v; }
        }
        best
    }

    #[test]
    fn add_inplace_metal_matches_cpu() {
        let ctx = MetalContext::new().expect("metal ctx");
        for &n in &[1usize, 7, 256, 1024, 4096] {
            let a_cpu: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
            let b_cpu: Vec<f32>     = (0..n).map(|i| (i as f32) * -0.3).collect();
            let mut a_ref = a_cpu.clone();
            add_inplace(&mut a_ref, &b_cpu);

            let bytes_a = bytemuck::cast_slice::<f32, u8>(&a_cpu);
            let bytes_b = bytemuck::cast_slice::<f32, u8>(&b_cpu);
            let a_buf = ctx.new_buffer_with_bytes(bytes_a);
            let b_buf = ctx.new_buffer_with_bytes(bytes_b);
            add_inplace_metal(&ctx, &a_buf, &b_buf, n).expect("dispatch");

            let ptr = a_buf.contents() as *const f32;
            let a_metal: &[f32] = unsafe { std::slice::from_raw_parts(ptr, n) };

            for (i, (m, r)) in a_metal.iter().zip(a_ref.iter()).enumerate() {
                let diff = (m - r).abs();
                assert!(diff < 1e-5, "n={n} i={i} metal={m} ref={r} diff={diff}");
            }
        }
    }

    #[test]
    fn gpu_argmax_logits_matches_cpu() {
        let ctx = MetalContext::new().expect("metal ctx");
        for &vocab in &[1024usize, 32000, 102400] {
            let mut logits: Vec<f32> = (0..vocab).map(|i| {
                let center = vocab / 3;
                let dist = (i as i32 - center as i32).abs() as f32;
                10.0 - dist * 0.001
            }).collect();
            let target = vocab / 2 + 7;
            logits[target] = 100.0;

            let cpu_pick = cpu_argmax(&logits);
            assert_eq!(cpu_pick, target as u32);

            let logits_buf = ctx.new_buffer_with_bytes(
                bytemuck::cast_slice::<f32, u8>(&logits),
            );
            let out_buf = ctx.new_buffer(std::mem::size_of::<u32>());
            gpu_argmax_logits_metal(&ctx, &logits_buf, &out_buf, vocab)
                .expect("dispatch");

            let ptr = out_buf.contents() as *const u32;
            let metal_pick: u32 = unsafe { *ptr };

            assert_eq!(
                metal_pick, cpu_pick,
                "vocab={vocab} cpu={cpu_pick} metal={metal_pick}"
            );
        }
    }
}
