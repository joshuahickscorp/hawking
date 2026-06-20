// gate-bitslice-staged — micro-bench for the NEW coalesced-write decode variant
// (shaders/strand_bitslice_staged.metal). Self-contained Metal host (does NOT touch
// metal.rs). Compiles the deployed bitslice shader AND the staged variant from source,
// proves the staged variant is BYTE-IDENTICAL to decode_tensor_fixed on a tiny tensor,
// then times BOTH on a small shape best-of-3.
//
// HARD SAFETY: a live MPS PV (scripts/strand-qat.py) owns the GPU. This bin REFUSES to
// dispatch anything to the GPU if `pgrep -f strand-qat` matches, unless --force is passed.
// Default run (no GPU) prints the analysis + bench plan and exits 0. It is designed to be a
// <2 s micro-bench when the GPU is free (small shape, best-of-3).
//
// Run when GPU is FREE:   cargo run -p strand-decode-kernel --release --bin gate-bitslice-staged -- --force
// Default (PV live, safe): cargo run -p strand-decode-kernel --release --bin gate-bitslice-staged

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("gate-bitslice-staged: Metal is macOS-only; nothing to do on this target.");
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

#[cfg(target_os = "macos")]
mod macos {
    use metal::{
        Buffer, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
        NSUInteger,
    };
    use std::time::Instant;

    use strand_decode_kernel::metal::{bake_bitslice_entries, BitsliceEntry};
    use strand_quant::codebook::codebook_lut;
    use strand_quant::decode::decode_tensor_fixed;
    use strand_quant::encode::encode_tensor;
    use strand_quant::TrellisConfig;

    const DEPLOYED: &str = include_str!("../../shaders/strand_bitslice.metal");
    const STAGED: &str = include_str!("../../shaders/strand_bitslice_staged.metal");

    fn qat_running() -> bool {
        std::process::Command::new("pgrep")
            .args(["-f", "strand-qat"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn upload<T: Copy>(dev: &Device, data: &[T]) -> Buffer {
        let byte_len = (data.len() * std::mem::size_of::<T>()).max(4);
        let buf = dev.new_buffer(byte_len as NSUInteger, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                data.len() * std::mem::size_of::<T>(),
            );
        }
        buf
    }

    fn upload_payload(dev: &Device, bits: &[u8]) -> Buffer {
        let padded = bits.len().div_ceil(4) * 4 + 8;
        let buf = dev.new_buffer(padded as NSUInteger, MTLResourceOptions::StorageModeShared);
        unsafe {
            let dst = buf.contents() as *mut u8;
            std::ptr::write_bytes(dst, 0, padded);
            std::ptr::copy_nonoverlapping(bits.as_ptr(), dst, bits.len());
        }
        buf
    }

    fn alloc(dev: &Device, byte_len: usize) -> Buffer {
        dev.new_buffer(byte_len.max(4) as NSUInteger, MTLResourceOptions::StorageModeShared)
    }

    fn pipe(dev: &Device, src: &str, name: &str) -> Option<ComputePipelineState> {
        let lib = dev.new_library_with_source(src, &CompileOptions::new()).ok()?;
        let f = lib.get_function(name, None).ok()?;
        dev.new_compute_pipeline_state_with_function(&f).ok()
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_deployed(
        dev: &Device,
        q: &metal::CommandQueue,
        p: &ComputePipelineState,
        w: &Buffer,
        out: &Buffer,
        tbl: &Buffer,
        nb: &Buffer,
        k: &Buffer,
        l: &Buffer,
        lut: &Buffer,
        n_blocks: usize,
        l_bits: u32,
    ) {
        let cmd = q.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_buffer(2, Some(tbl), 0);
        enc.set_buffer(3, Some(nb), 0);
        enc.set_buffer(4, Some(k), 0);
        enc.set_buffer(5, Some(l), 0);
        enc.set_buffer(6, Some(lut), 0);
        enc.set_threadgroup_memory_length(0, ((1usize << l_bits) * 4) as NSUInteger);
        let groups = MTLSize { width: (n_blocks as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
        let _ = dev;
        enc.dispatch_thread_groups(groups, tpg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_staged(
        q: &metal::CommandQueue,
        p: &ComputePipelineState,
        w: &Buffer,
        out: &Buffer,
        tbl: &Buffer,
        nb: &Buffer,
        k: &Buffer,
        l: &Buffer,
        lut: &Buffer,
        n_blocks: usize,
        l_bits: u32,
        bg: u64,
    ) {
        let cmd = q.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_buffer(2, Some(tbl), 0);
        enc.set_buffer(3, Some(nb), 0);
        enc.set_buffer(4, Some(k), 0);
        enc.set_buffer(5, Some(l), 0);
        enc.set_buffer(6, Some(lut), 0);
        // threadgroup(0) = LUT (2^L ints); threadgroup(1) = staging tile (bg*256 ints).
        enc.set_threadgroup_memory_length(0, ((1usize << l_bits) * 4) as NSUInteger);
        enc.set_threadgroup_memory_length(1, (bg as usize * 256 * 4) as NSUInteger);
        let groups = MTLSize { width: (n_blocks as u64).div_ceil(bg) as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: bg as NSUInteger, height: 1, depth: 1 };
        enc.dispatch_thread_groups(groups, tpg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Pick blocks-per-threadgroup so tile + LUT fit the 32 KB TG budget, with headroom.
    fn pick_bg(l_bits: u32) -> u64 {
        let lut_bytes = (1usize << l_bits) * 4;
        let budget = 32 * 1024usize;
        let tile_budget = budget.saturating_sub(lut_bytes);
        let max_blocks = (tile_budget / (256 * 4)) as u64; // ints per block * 4
        // clamp to a SIMD-friendly value: prefer 32, else the largest multiple of 8 that fits
        let mut bg = max_blocks.min(32);
        if bg >= 8 {
            bg -= bg % 8;
        }
        bg.max(1)
    }

    pub fn run() {
        let force = std::env::args().any(|a| a == "--force");
        let plan = || {
            println!("\n=== gate-bitslice-staged: BENCH PLAN (GPU not touched) ===");
            println!("WHY: deployed strand_bitslice_decode is write-bound — each thread stores");
            println!("     out_q12[gidx*256+j]; adjacent SIMD lanes write 1024 B apart => ~1/32");
            println!("     store efficiency. The 4 B/w Q12 buffer is ~85% of decode traffic.");
            println!("FIX: strand_bitslice_decode_staged keeps the identical serial walk but stages");
            println!("     per-block outputs in threadgroup memory, then flushes COALESCED");
            println!("     (consecutive lanes -> consecutive global addresses).");
            println!("RUN (GPU free): cargo run -p strand-decode-kernel --release \\");
            println!("                --bin gate-bitslice-staged -- --force");
            println!("MEASURE: best-of-N decode_q12 on ffn_down 18944x3584 (67.9 Mw), k3 L7 & k2 L12;");
            println!("         report Gw/s and GB/s vs StrandGpu::bench_peak_bw, identical to");
            println!("         gate-bitslice's bench(). Compare staged vs deployed side by side.");
            println!("GATE: staged must (a) be byte-identical to decode_tensor_fixed (asserted");
            println!("      before timing) and (b) beat deployed decode GB/s. Target: >=90% peak.");
        };

        if qat_running() && !force {
            println!("strand-qat MPS PV is LIVE — refusing GPU dispatch (pass --force to override).");
            plan();
            return;
        }

        let Some(dev) = Device::system_default() else {
            println!("no Metal device; nothing to do.");
            plan();
            return;
        };
        let q = dev.new_command_queue();

        let Some(p_deployed) = pipe(&dev, DEPLOYED, "strand_bitslice_decode") else {
            println!("deployed bitslice shader failed to compile");
            return;
        };
        let Some(p_staged) = pipe(&dev, STAGED, "strand_bitslice_decode_staged") else {
            println!("staged bitslice shader failed to compile");
            return;
        };
        // sizeof probe parity (host repr(C) vs both shaders)
        if let Some(p_sz) = pipe(&dev, STAGED, "strand_bitslice_staged_sizeof") {
            let out = alloc(&dev, 4);
            let cmd = q.new_command_buffer();
            let e = cmd.new_compute_command_encoder();
            e.set_compute_pipeline_state(&p_sz);
            e.set_buffer(0, Some(&out), 0);
            let one = MTLSize { width: 1, height: 1, depth: 1 };
            e.dispatch_thread_groups(one, one);
            e.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu_sz = unsafe { *(out.contents() as *const u32) } as usize;
            assert_eq!(
                gpu_sz,
                std::mem::size_of::<BitsliceEntry>(),
                "staged GPU sizeof(BitsliceEntry)={gpu_sz} != host {}",
                std::mem::size_of::<BitsliceEntry>()
            );
            println!("sizeof(BitsliceEntry) parity OK: {gpu_sz} B (host == staged shader)");
        }

        // ---- IDENTITY (small, multiple shapes incl. a partial last block) ----
        let mut checked = 0usize;
        for cfg in [TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw_l(2.0, 12)] {
            let bg = pick_bg(cfg.l_bits);
            for &total in &[256usize, 512, 700, 4096, 65536, 67000] {
                let w: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.0137).sin() * 0.5).collect();
                let enc = encode_tensor(&w, &cfg);
                let want = decode_tensor_fixed(&enc, &cfg);
                let Some(tbl) = bake_bitslice_entries(&enc, &cfg) else { continue };
                let lut = codebook_lut(cfg.l_bits);

                let w_buf = upload_payload(&dev, &enc.bits);
                let out_buf = alloc(&dev, total * 4);
                let tbl_buf = upload(&dev, &tbl);
                let nb = upload(&dev, &[tbl.len() as u32]);
                let kb = upload(&dev, &[cfg.k_bits]);
                let lb = upload(&dev, &[cfg.l_bits]);
                let lut_buf = upload(&dev, lut);

                dispatch_staged(
                    &q, &p_staged, &w_buf, &out_buf, &tbl_buf, &nb, &kb, &lb, &lut_buf,
                    tbl.len(), cfg.l_bits, bg,
                );
                let got = unsafe { std::slice::from_raw_parts(out_buf.contents() as *const i32, total) };
                assert_eq!(
                    got, &want[..],
                    "STAGED IDENTITY VIOLATION: k={} L={} total={total} bg={bg}",
                    cfg.k_bits, cfg.l_bits
                );
                checked += 1;
            }
            println!(
                "identity OK: staged == decode_tensor_fixed for k{} L{} (bg={}) across 6 shapes",
                cfg.k_bits, cfg.l_bits, bg
            );
        }
        println!("identity: {checked} staged cells byte-identical to decode_tensor_fixed");

        // ---- MICRO-BENCH (small shape; best-of-N; <2 s total) ----
        // Use a modest shape to stay well under 2 s even cold: 2048 x 2048 = 4.2 Mw.
        let (rows, cols) = (2048usize, 2048usize);
        let total = rows * cols;
        let peak_dev = strand_decode_kernel::metal::StrandGpu::new();
        let peak = peak_dev
            .as_ref()
            .map(|g| g.bench_peak_bw(16 << 20, 3))
            .unwrap_or(f64::NAN);
        println!(
            "\n== micro-bench {rows}x{cols} = {:.1}M weights (best-of-5) ==  measured peak {:.1} GB/s",
            total as f64 / 1e6,
            peak / 1e9
        );
        for cfg in [TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw_l(2.0, 12)] {
            let bg = pick_bg(cfg.l_bits);
            let w: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.013).sin() * 0.4).collect();
            let enc = encode_tensor(&w, &cfg);
            let Some(tbl) = bake_bitslice_entries(&enc, &cfg) else { continue };
            let lut = codebook_lut(cfg.l_bits);

            let w_buf = upload_payload(&dev, &enc.bits);
            let out_buf = alloc(&dev, total * 4);
            let tbl_buf = upload(&dev, &tbl);
            let nb = upload(&dev, &[tbl.len() as u32]);
            let kb = upload(&dev, &[cfg.k_bits]);
            let lb = upload(&dev, &[cfg.l_bits]);
            let lut_buf = upload(&dev, lut);

            let q12_bytes = (total * 4) as f64;
            let payload_bytes = (total as f64) * (cfg.k_bits as f64) / 8.0;
            let tbl_bytes = (tbl.len() * std::mem::size_of::<BitsliceEntry>()) as f64;
            let moved = q12_bytes + payload_bytes + tbl_bytes;

            // deployed
            let mut best_dep = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                dispatch_deployed(
                    &dev, &q, &p_deployed, &w_buf, &out_buf, &tbl_buf, &nb, &kb, &lb, &lut_buf,
                    tbl.len(), cfg.l_bits,
                );
                let dt = t.elapsed().as_secs_f64();
                if dt > 0.0 && dt < best_dep { best_dep = dt; }
            }
            // staged
            let mut best_stg = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                dispatch_staged(
                    &q, &p_staged, &w_buf, &out_buf, &tbl_buf, &nb, &kb, &lb, &lut_buf,
                    tbl.len(), cfg.l_bits, bg,
                );
                let dt = t.elapsed().as_secs_f64();
                if dt > 0.0 && dt < best_stg { best_stg = dt; }
            }

            let g = |dt: f64| total as f64 / dt / 1e9;
            let bw = |dt: f64| moved / dt / 1e9;
            let pk = |dt: f64| if peak.is_finite() { 100.0 * (moved / dt) / peak } else { f64::NAN };
            println!(
                "  k{} L{} (bg={bg}): deployed {:.2} Gw/s {:.1} GB/s ({:.0}% pk) | staged {:.2} Gw/s {:.1} GB/s ({:.0}% pk) | staged/deployed {:.2}x",
                cfg.k_bits, cfg.l_bits,
                g(best_dep), bw(best_dep), pk(best_dep),
                g(best_stg), bw(best_stg), pk(best_stg),
                best_dep / best_stg,
            );
        }
        println!(
            "(micro-bench on a SMALL shape to bound runtime; the canonical 67.9 Mw ffn_down number\n bench is gate-bitslice's bench() — wire the staged path there for the headline figure.)"
        );
    }
}
