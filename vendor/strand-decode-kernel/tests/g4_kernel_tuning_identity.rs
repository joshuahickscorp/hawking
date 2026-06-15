// G4 DECODE-KERNEL TUNING — bit-identity gate (NEW integration test, no shared-file edits).
//
// LEVER UNDER TEST (sprint G4): "decode-kernel bandwidth tuning — the tuned kernel must
// produce BYTE-IDENTICAL decode to the reference (speed change only, zero output change)."
//
// THE MOAT (why this test exists): STRAND's claim is that decode is bit-identical on every
// device — a frozen integer Q12 LUT, float-free reconstruct, fully deterministic. A
// bandwidth/throughput tuning is *only* allowed to change WHEN/WHERE bytes are read and
// written and HOW work is scheduled across lanes/threads; it must NEVER change the produced
// Q12 integers. The single source of truth is `strand_quant::decode::decode_tensor_fixed`
// (the canonical frozen-LUT integer path). Every tuned decode path must reproduce that
// Vec<i32> EXACTLY (`assert_eq!`, not an epsilon) for every input.
//
// This file is a PURE CONSUMER of the crate's public API + (on macOS) compiles the staged
// Metal variant FROM SOURCE inline. It edits NO shared file (no lib.rs / mod.rs /
// format.rs / encode.rs / decode.rs / metal.rs).
//
// ---- WHAT "bit-identical" means precisely for a tuning ----
// The decoded weight at logical index i is, for the deployed integer path:
//     q[i]  = ( eff_scale_q(scale_q, sub_mult[i/32]) * codebook_lut[L][state_i] ) >> 16
//             ( + eff_min_q offset, when affine-min )
// where state_i is the trellis state after consuming i symbols of k bits each. Every term
// is an integer; the LUT is a frozen Q12 table indexed by an integer state; the products
// use widening i64 and arithmetic shift. A *tuning* (more threads, coalesced writes, a
// staging tile, SIMD lanes, batched dispatch) reorders memory traffic but evaluates the
// SAME integer expression per index, in the SAME within-block order. Therefore the output
// is invariant. This test makes that invariant executable.
//
// ---- TEST STRATA ----
//   (A) CPU tuned paths (ALWAYS run, every platform): the deployed fast/parallel/SIMD
//       decoders (the existing tuned CPU "kernels") vs decode_tensor_fixed. These are the
//       in-tree precedent for "a faster decode that must stay bit-identical".
//   (B) Property/fuzz: a deterministic xorshift PRNG (no proptest dep in this crate) drives
//       adversarial weight tensors x {k,L} configs x {tail-biting, affine-min} x edge sizes
//       (empty / single / sub-block / partial-final-block / multi-block / large).
//   (C) GPU staged kernel (macOS + Metal device only; SKIPS if a live MPS PV owns the GPU;
//       the dispatch test is #[ignore] so `cargo test` never touches the GPU by default):
//       the G4 coalesced-write staged variant must be byte-identical to decode_tensor_fixed
//       AND to the deployed bitslice kernel.
//
// Run the GPU leg explicitly only when the GPU is FREE:
//   cargo test -p strand-decode-kernel --test g4_kernel_tuning_identity -- --ignored

use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts, EncodedTensor};
use strand_quant::TrellisConfig;

use strand_decode_kernel::gemv::decode_q12_fast;
use strand_decode_kernel::gemv_par::{decode_q12_par, decode_q12_simd};

// ----------------------------------------------------------------------------------------
// Deterministic PRNG (xorshift64*) — same family as gate-decode-speed.rs. No external dep,
// reproducible: the SEED below is the only entropy, so a failure is exactly replayable.
// ----------------------------------------------------------------------------------------
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        // avoid the zero fixed-point
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// A weight in roughly [-1, 1] with occasional outliers (stresses scale/sub-scale and,
    /// under affine-min, the min-offset path) and exact-zero plateaus (stresses ties).
    fn next_weight(&mut self) -> f32 {
        let r = self.next_u64();
        match r & 0xF {
            0 => 0.0,                                            // exact zero
            1 => 3.5 * ((r >> 8) as i32 as f32 / i32::MAX as f32), // outlier
            _ => {
                let u = ((r >> 11) as f64) / (1u64 << 53) as f64; // [0,1)
                (u as f32) * 2.0 - 1.0
            }
        }
    }
}

/// All deploy-relevant trellis configs. (k,L) pairs cover: the 3-bit deploy default (k3 L7,
/// the production GPU config), the folded small-L paths (SUB_BLOCK>=num_states), and the
/// large-L unfolded path (k2 L12 — the second production GPU config).
fn deploy_configs() -> Vec<TrellisConfig> {
    vec![
        TrellisConfig::for_bpw(3.0),       // k3 L7  (deploy default)
        TrellisConfig::for_bpw(2.0),       // k2 L?  (default L)
        TrellisConfig::for_bpw(4.0),       // k4 L?
        TrellisConfig::for_bpw_l(2.0, 12), // k2 L12 (deploy alt; large LUT)
        TrellisConfig::for_bpw_l(2.0, 5),  // small L -> folded path
        TrellisConfig::for_bpw_l(4.0, 4),  // small L -> folded path
        TrellisConfig::for_bpw_l(3.0, 7),  // explicit k3 L7
    ]
}

/// Edge-case + general element counts. 0 (empty), 1 (single), 31/32/33 (sub-block boundary
/// = SUB_BLOCK), 255/256/257 (block-len boundary), partial final block, and several full
/// blocks. Kept modest so the debug-build matrix (7 cfgs x sizes x 4 variants x 3 paths)
/// runs in a few seconds; the dedicated `*_large` test below covers the >64k multi-block
/// regime that actually exercises the rayon/SIMD chunking, on a single config.
fn edge_sizes() -> Vec<usize> {
    vec![
        0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 255, 256, 257, 511, 512, 513, 700, 1023, 1024,
        1025, 2048, 4096, 4097, 8192,
    ]
}

/// Build the four encode variants for a weight vector + config: {plain, tail-biting,
/// affine-min, tail-biting+affine-min}. These flip every decode-relevant flag.
fn encode_variants(w: &[f32], cfg: &TrellisConfig) -> Vec<EncodedTensor> {
    vec![
        encode_tensor(w, cfg),
        encode_tensor_with(w, cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
        encode_tensor_with(w, cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
        encode_tensor_with(
            w,
            cfg,
            &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
        ),
    ]
}

// ========================================================================================
// (A)+(B): CPU tuned decode paths must be byte-identical to the canonical integer reference.
// This is the platform-portable core of the G4 moat proof: it runs on CI for every arch
// (the SIMD path is NEON on aarch64 and falls back to the parallel path elsewhere — both
// must match the reference). Pure assert_eq! on Vec<i32>: NO epsilon anywhere.
// ========================================================================================
#[test]
fn cpu_tuned_decode_paths_are_bit_identical_to_reference() {
    let mut rng = Rng::new(_G4_SEED());
    let mut checked_cells: u64 = 0;

    for cfg in deploy_configs() {
        for &n in &edge_sizes() {
            let w: Vec<f32> = (0..n).map(|_| rng.next_weight()).collect();
            for enc in encode_variants(&w, &cfg) {
                // canonical, frozen-LUT, float-free integer reference
                let reference = decode_tensor_fixed(&enc, &cfg);
                assert_eq!(reference.len(), n, "reference length must equal weight count");

                // The deployed CPU tuned paths. Each is a DIFFERENT schedule of the same
                // integer recurrence; all must reproduce `reference` to the bit.
                let fast = decode_q12_fast(&enc, &cfg);
                assert_eq!(
                    fast, reference,
                    "FAST decode diverged from reference (MOAT BREAK): \
                     k={} L={} n={} tail={} affine={}",
                    cfg.k_bits, cfg.l_bits, n, enc.tail_biting, enc.has_affine_min
                );

                let par = decode_q12_par(&enc, &cfg);
                assert_eq!(
                    par, reference,
                    "PARALLEL decode diverged from reference (MOAT BREAK): \
                     k={} L={} n={} tail={} affine={}",
                    cfg.k_bits, cfg.l_bits, n, enc.tail_biting, enc.has_affine_min
                );

                let simd = decode_q12_simd(&enc, &cfg);
                assert_eq!(
                    simd, reference,
                    "SIMD decode diverged from reference (MOAT BREAK): \
                     k={} L={} n={} tail={} affine={}",
                    cfg.k_bits, cfg.l_bits, n, enc.tail_biting, enc.has_affine_min
                );

                checked_cells += n as u64;
            }
        }
    }
    // sanity: we actually exercised a meaningful amount of data
    assert!(checked_cells > 100_000, "too few cells checked ({checked_cells})");
}

// Large multi-block regime (single 3-bit config) so the rayon block-split and the NEON
// 4-block SIMD lane stepping are genuinely exercised at scale, while keeping wall-time low.
// A >256-block tensor with a partial final block; both fast/par/simd vs the reference.
#[test]
fn cpu_tuned_decode_bit_identical_large_multiblock() {
    let mut rng = Rng::new(0xA11C_E5_2026);
    for cfg in [TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw_l(2.0, 12)] {
        let n = 256 * 300 + 173; // 300 full blocks + a 173-wide partial final block
        let w: Vec<f32> = (0..n).map(|_| rng.next_weight()).collect();
        for enc in encode_variants(&w, &cfg) {
            let reference = decode_tensor_fixed(&enc, &cfg);
            assert_eq!(
                decode_q12_fast(&enc, &cfg),
                reference,
                "FAST large diverged k={} L={} tail={} affine={}",
                cfg.k_bits, cfg.l_bits, enc.tail_biting, enc.has_affine_min
            );
            assert_eq!(
                decode_q12_par(&enc, &cfg),
                reference,
                "PAR large diverged k={} L={} tail={} affine={}",
                cfg.k_bits, cfg.l_bits, enc.tail_biting, enc.has_affine_min
            );
            assert_eq!(
                decode_q12_simd(&enc, &cfg),
                reference,
                "SIMD large diverged k={} L={} tail={} affine={}",
                cfg.k_bits, cfg.l_bits, enc.tail_biting, enc.has_affine_min
            );
        }
    }
}

// A focused property: a tuning must be DETERMINISTIC across repeated invocations on the
// same input (no thread-order nondeterminism leaking into the integer output). Run the
// parallel/SIMD paths many times and demand identical bytes each time.
#[test]
fn tuned_decode_is_run_to_run_deterministic() {
    let mut rng = Rng::new(0xD37E_2026_06_13);
    for cfg in [TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw_l(2.0, 12)] {
        // a multi-block tensor so rayon actually splits work across threads
        let n = 256 * 64 + 137;
        let w: Vec<f32> = (0..n).map(|_| rng.next_weight()).collect();
        let enc = encode_tensor(&w, &cfg);
        let reference = decode_tensor_fixed(&enc, &cfg);
        for rep in 0..8 {
            assert_eq!(
                decode_q12_par(&enc, &cfg),
                reference,
                "PAR nondeterministic at rep {rep} (k={} L={})",
                cfg.k_bits,
                cfg.l_bits
            );
            assert_eq!(
                decode_q12_simd(&enc, &cfg),
                reference,
                "SIMD nondeterministic at rep {rep} (k={} L={})",
                cfg.k_bits,
                cfg.l_bits
            );
        }
    }
}

// Cross-path equivalence stated as a transitive closure: fast == par == simd == reference.
// If a future G4 tuning adds a new CPU path, append it to `paths` and the chain catches any
// divergence at the first offending index (the helper reports it).
#[test]
fn all_cpu_paths_agree_pairwise_with_index_report() {
    let mut rng = Rng::new(0x5EED_0BEEF);
    for cfg in deploy_configs() {
        for &n in &[0usize, 1, 32, 256, 257, 4096, 65_537] {
            let w: Vec<f32> = (0..n).map(|_| rng.next_weight()).collect();
            for enc in encode_variants(&w, &cfg) {
                let reference = decode_tensor_fixed(&enc, &cfg);
                let paths: [(&str, Vec<i32>); 3] = [
                    ("fast", decode_q12_fast(&enc, &cfg)),
                    ("par", decode_q12_par(&enc, &cfg)),
                    ("simd", decode_q12_simd(&enc, &cfg)),
                ];
                for (name, got) in &paths {
                    if got != &reference {
                        let first = got
                            .iter()
                            .zip(reference.iter())
                            .position(|(a, b)| a != b)
                            .unwrap_or(usize::MAX);
                        panic!(
                            "{name} != reference (MOAT BREAK) at index {first}: \
                             got {:?} want {:?} | k={} L={} n={} tail={} affine={}",
                            got.get(first),
                            reference.get(first),
                            cfg.k_bits,
                            cfg.l_bits,
                            n,
                            enc.tail_biting,
                            enc.has_affine_min
                        );
                    }
                }
            }
        }
    }
}

// ========================================================================================
// (C): GPU G4 staged-kernel bit-identity.
//
// The staged coalesced-write variant (shaders/strand_bitslice_staged.metal) is the concrete
// G4 tuning in flight. It is NOT yet wired into BitsliceGpu, so this leg compiles the shader
// FROM SOURCE inline (mirroring src/bin/gate-bitslice-staged.rs) rather than editing
// metal.rs. It is #[ignore]d so the default `cargo test` never dispatches to the GPU, and it
// HARD-REFUSES to run if a live MPS PV (scripts/strand-qat*) owns the device — protecting the
// conductor's running job. The assertion is the moat: staged output == decode_tensor_fixed,
// byte for byte, AND == the deployed bitslice kernel.
// ========================================================================================
#[cfg(target_os = "macos")]
#[test]
#[ignore = "GPU leg: run explicitly with --ignored only when the MPS PV is not live"]
fn gpu_staged_kernel_is_bit_identical_to_reference() {
    gpu_staged::run();
}

#[cfg(target_os = "macos")]
mod gpu_staged {
    use metal::{
        Buffer, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
        NSUInteger,
    };
    use strand_decode_kernel::metal::{bake_bitslice_entries, BitsliceEntry};
    use strand_quant::codebook::codebook_lut;
    use strand_quant::decode::decode_tensor_fixed;
    use strand_quant::encode::encode_tensor;
    use strand_quant::TrellisConfig;

    const DEPLOYED: &str = include_str!("../shaders/strand_bitslice.metal");
    const STAGED: &str = include_str!("../shaders/strand_bitslice_staged.metal");

    /// Refuse to touch the GPU if a quant/PV job owns MPS (the conductor's live process).
    fn pv_owns_gpu() -> bool {
        for pat in ["strand-qat", "strand-pv", "strand-quant-7b"] {
            let hit = std::process::Command::new("pgrep")
                .args(["-f", pat])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if hit {
                return true;
            }
        }
        false
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
        // pad to a u32 boundary + 8 slack bytes so the kernel's word reads never run past
        // the buffer (the deployed loader pads identically).
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

    /// Same threadgroup-budget clamp as gate-bitslice-staged.rs / tests/staged_layout.rs.
    fn pick_bg(l_bits: u32) -> u64 {
        let lut_bytes = (1usize << l_bits) * 4;
        let tile_budget = (32 * 1024usize).saturating_sub(lut_bytes);
        let max_blocks = (tile_budget / (256 * 4)) as u64;
        let mut bg = max_blocks.min(32);
        if bg >= 8 {
            bg -= bg % 8;
        }
        bg.max(1)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_deployed(
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
        let groups =
            MTLSize { width: (n_blocks as u64).div_ceil(256) as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: 256, height: 1, depth: 1 };
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
        enc.set_threadgroup_memory_length(0, ((1usize << l_bits) * 4) as NSUInteger);
        enc.set_threadgroup_memory_length(1, (bg as usize * 256 * 4) as NSUInteger);
        let groups =
            MTLSize { width: (n_blocks as u64).div_ceil(bg) as NSUInteger, height: 1, depth: 1 };
        let tpg = MTLSize { width: bg as NSUInteger, height: 1, depth: 1 };
        enc.dispatch_thread_groups(groups, tpg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    pub fn run() {
        if pv_owns_gpu() {
            eprintln!(
                "[g4-identity] a live MPS PV owns the GPU (pgrep strand-qat/-pv hit); \
                 REFUSING to dispatch. Re-run with --ignored only when the GPU is free."
            );
            return;
        }
        let Some(dev) = Device::system_default() else {
            eprintln!("[g4-identity] no Metal device; skipping GPU staged identity leg.");
            return;
        };
        let q = dev.new_command_queue();

        let Some(p_deployed) = pipe(&dev, DEPLOYED, "strand_bitslice_decode") else {
            panic!("deployed bitslice shader failed to compile");
        };
        let Some(p_staged) = pipe(&dev, STAGED, "strand_bitslice_decode_staged") else {
            panic!("staged bitslice shader failed to compile");
        };

        // ABI parity: GPU struct layout must match the host repr(C). If this diverges the
        // row-major tbl stride diverges and EVERY downstream byte is suspect — fail loud.
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
                "staged GPU sizeof(BitsliceEntry)={gpu_sz} != host {} — tbl stride diverges",
                std::mem::size_of::<BitsliceEntry>()
            );
        }

        // Configs are the two production GPU configs. Sizes include every edge: a partial
        // final block (n%256!=0), exactly one threadgroup of blocks (bg*256), one-past
        // (bg*256+1), several threadgroups, and a single-weight tensor.
        let mut total_checked: u64 = 0;
        for cfg in [TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw_l(2.0, 12)] {
            let bg = pick_bg(cfg.l_bits);
            let sizes = [
                1usize,
                255,
                256,
                257,
                512,
                700,
                (bg as usize) * 256,
                (bg as usize) * 256 + 1,
                (bg as usize) * 256 * 3 + 137,
                4096,
                65_536,
                67_000,
            ];
            for &total in &sizes {
                let w: Vec<f32> =
                    (0..total).map(|i| ((i as f32) * 0.0137).sin() * 0.5).collect();
                let enc = encode_tensor(&w, &cfg);
                let want = decode_tensor_fixed(&enc, &cfg);
                let Some(tbl) = bake_bitslice_entries(&enc, &cfg) else { continue };
                let lut = codebook_lut(cfg.l_bits);

                let w_buf = upload_payload(&dev, &enc.bits);
                let nb = upload(&dev, &[tbl.len() as u32]);
                let kb = upload(&dev, &[cfg.k_bits]);
                let lb = upload(&dev, &[cfg.l_bits]);
                let tbl_buf = upload(&dev, &tbl);
                let lut_buf = upload(&dev, lut);

                // deployed kernel -> its own output buffer
                let out_dep = alloc(&dev, total * 4);
                dispatch_deployed(
                    &q, &p_deployed, &w_buf, &out_dep, &tbl_buf, &nb, &kb, &lb, &lut_buf,
                    tbl.len(), cfg.l_bits,
                );
                let got_dep =
                    unsafe { std::slice::from_raw_parts(out_dep.contents() as *const i32, total) };
                assert_eq!(
                    got_dep,
                    &want[..],
                    "DEPLOYED bitslice != reference (precondition): k={} L={} total={total}",
                    cfg.k_bits, cfg.l_bits
                );

                // staged (G4-tuned) kernel -> a FRESH buffer prefilled with a sentinel so an
                // unwritten slot (a flush hole) would surface as a mismatch, not a stale 0.
                let out_stg = alloc(&dev, total * 4);
                unsafe {
                    let p = out_stg.contents() as *mut i32;
                    for i in 0..total {
                        *p.add(i) = i32::MIN; // sentinel: must be fully overwritten
                    }
                }
                dispatch_staged(
                    &q, &p_staged, &w_buf, &out_stg, &tbl_buf, &nb, &kb, &lb, &lut_buf,
                    tbl.len(), cfg.l_bits, bg,
                );
                let got_stg =
                    unsafe { std::slice::from_raw_parts(out_stg.contents() as *const i32, total) };

                // THE MOAT ASSERTION: staged Q12 == frozen-LUT integer reference, bit for bit.
                assert_eq!(
                    got_stg,
                    &want[..],
                    "STAGED (G4) decode != reference (MOAT BREAK): k={} L={} total={total} bg={bg}",
                    cfg.k_bits, cfg.l_bits
                );
                // and staged == deployed (cross-kernel equivalence)
                assert_eq!(
                    got_stg, got_dep,
                    "STAGED != DEPLOYED (cross-kernel divergence): k={} L={} total={total} bg={bg}",
                    cfg.k_bits, cfg.l_bits
                );
                assert!(
                    got_stg.iter().all(|&v| v != i32::MIN) || total == 0,
                    "staged left a sentinel slot unwritten (flush hole): k={} L={} total={total}",
                    cfg.k_bits, cfg.l_bits
                );
                total_checked += total as u64;
            }
        }
        eprintln!("[g4-identity] GPU staged kernel byte-identical over {total_checked} weights");
    }
}

// Helper: the lever prompt forbids editing shared files, so we keep the magic seed literal
// readable. Rust has no `0xG4...` literal, so synthesize the seed from bytes. (Pure const.)
#[allow(non_snake_case)]
const fn _G4_SEED() -> u64 {
    u64::from_le_bytes(*b"G4_IDENT")
}
