// gate-token-buffer — Wave-4 speed bet 2: WHOLE-TOKEN COMMAND BUFFER.
//
// The deployed Metal path runs `new_command_buffer -> encode -> commit ->
// wait_until_completed` ONCE PER TENSOR. A Qwen2.5-0.5B decode step touches 7
// projection tensors per layer * 24 layers = 168 GEMVs, i.e. 168 separate command-
// buffer commits and 168 CPU<->GPU round-trips. docs/STRAND-speed-moonshot-research.md
// §D flags this as mandatory to kill before dismantle integration: "a kernel that is
// 3x faster can lose the win through 100+ tiny commits".
//
// This gate builds a synthetic Qwen-0.5B-shaped token pass (all projection tensors
// prepared once) and times two execution shapes against IDENTICAL prepared work:
//   A. per-tensor  : one command buffer + commit + wait per tensor (today's shape)
//   B. one-buffer  : all 168 tensors encoded into ONE command buffer, one commit,
//                    one wait_until_completed (the target)
// It reports the token-wall both ways and the per-commit overhead that path B kills.
//
// Identity: path B must produce the same per-tensor y as path A (within float MAC
// tolerance) — same kernels, same buffers, only the commit boundary differs.
//
// Run:  cargo run -p strand-decode-kernel --release --bin gate-token-buffer

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gate-token-buffer: Metal-only gate; skipping on non-macOS.");
}

#[cfg(target_os = "macos")]
fn main() {
    use std::time::Instant;
    use strand_decode_kernel::block_walk::gate_proto::{machine_stamp, synth_encoded};
    use strand_decode_kernel::metal::{bake_bitslice_entries, BitsliceGpu, MatvecToken};
    use strand_quant::codebook::codebook_lut;
    use strand_quant::TrellisConfig;

    // ---- Qwen2.5-0.5B projection geometry (out_features, in_features) -----------------
    // 24 layers; per layer: q/o (896x896), k/v (128x896), gate/up (4864x896),
    // down (896x4864). The bitslice kernel needs in_features (cols) a multiple of 256,
    // exactly as the deployed encoder pads; round cols up so the synthetic token matches
    // the real tensor *count* and *scale*, which is what the commit overhead tracks.
    const LAYERS: usize = 24;
    fn round256(n: usize) -> usize {
        n.div_ceil(256) * 256
    }
    let per_layer: [(usize, usize, &str); 7] =
        [(896, 896, "q_proj"), (128, 896, "k_proj"), (128, 896, "v_proj"), (896, 896, "o_proj"), (4864, 896, "gate_proj"), (4864, 896, "up_proj"), (896, 4864, "down_proj")];
    let mut shapes: Vec<(usize, usize)> = Vec::new();
    for _ in 0..LAYERS {
        for (o, i, _) in per_layer.iter() {
            shapes.push((*o, round256(*i)));
        }
    }
    let n_tensors = shapes.len();
    let total_weights: usize = shapes.iter().map(|(o, i)| o * i).sum();

    let gpu = match BitsliceGpu::new() {
        Some(g) => g,
        None => {
            eprintln!("gate-token-buffer: no Metal device; skipping.");
            return;
        }
    };

    println!("gate-token-buffer — Wave-4 bet 2: whole-token command buffer");
    println!("  {}", machine_stamp());
    println!(
        "  synthetic token: {n_tensors} projection tensors ({LAYERS} layers x 7), \
         {:.1}M weights/token",
        total_weights as f64 / 1e6
    );

    // 3-bit deploy (k3 L7) is the integration target shape.
    for (cfg, label) in [(TrellisConfig::for_bpw(3.0), "3-bit deploy (k3 L7)"), (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen (k2 L12)")] {
        println!("\n== {label} ==");
        let lut = codebook_lut(cfg.l_bits).to_vec();

        // Prepare ALL tensors once (upload payload/tbl/lut/x, alloc partials/y). This is the
        // load-time / per-token-prep cost and is identical for both execution shapes.
        let t_prep = Instant::now();
        let mut tokens: Vec<MatvecToken> = Vec::with_capacity(n_tensors);
        for (out_f, in_f) in shapes.iter().copied() {
            let total = out_f * in_f;
            let enc = synth_encoded(total, cfg.k_bits, 256);
            let tbl = bake_bitslice_entries(&enc, &cfg).expect("256-weight blocks bake");
            let x: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.013).sin()).collect();
            tokens.push(gpu.prepare_token_tensor(&enc.bits, &tbl, &lut, out_f as u32, in_f as u32, cfg.k_bits, cfg.l_bits, &x));
        }
        let prep_ms = t_prep.elapsed().as_secs_f64() * 1e3;
        println!("  prepared {n_tensors} tensors in {prep_ms:.1} ms (one-time / per-token prep)");

        // ---- identity: one-buffer y == per-tensor y (within float MAC tolerance) --------
        gpu.run_token_per_tensor(&tokens);
        let y_ref: Vec<Vec<f32>> = tokens.iter().map(|t| t.y()).collect();
        gpu.run_token_one_buffer(&tokens);
        let y_one: Vec<Vec<f32>> = tokens.iter().map(|t| t.y()).collect();
        let mut max_rel = 0.0f64;
        for (a, b) in y_ref.iter().zip(&y_one) {
            for (&ra, &rb) in a.iter().zip(b) {
                let denom = (ra.abs().max(rb.abs()) as f64).max(1e-3);
                max_rel = max_rel.max(((ra - rb).abs() as f64) / denom);
            }
        }
        assert!(max_rel < 1e-4, "IDENTITY VIOLATION: one-buffer token diverged from per-tensor (max rel {max_rel:.2e})");
        println!("  identity OK: one-buffer == per-tensor (max rel diff {max_rel:.2e})");

        // ---- bench both shapes: token-wall -----------------------------------------------
        let iters = 30usize;
        // warmup
        gpu.run_token_per_tensor(&tokens);
        gpu.run_token_one_buffer(&tokens);

        let mut best_per = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            gpu.run_token_per_tensor(&tokens);
            best_per = best_per.min(t.elapsed().as_secs_f64());
        }
        let mut best_one = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            gpu.run_token_one_buffer(&tokens);
            best_one = best_one.min(t.elapsed().as_secs_f64());
        }

        let per_ms = best_per * 1e3;
        let one_ms = best_one * 1e3;
        let overhead_ms = per_ms - one_ms;
        let per_commit_us = (overhead_ms * 1e3) / (n_tensors as f64);
        let tok_s_per = 1.0 / best_per;
        let tok_s_one = 1.0 / best_one;
        println!(
            "  per-tensor commits : {per_ms:>7.3} ms/token  ({tok_s_per:>7.1} tok/s)  \
             [{n_tensors} commits + {n_tensors} waits]"
        );
        println!(
            "  one command buffer : {one_ms:>7.3} ms/token  ({tok_s_one:>7.1} tok/s)  \
             [1 commit + 1 wait]"
        );
        println!(
            "  SPEEDUP {:.2}x  | commit overhead killed = {overhead_ms:.3} ms/token \
             (~{per_commit_us:.1} us per tensor-commit)",
            best_per / best_one
        );
    }

    println!(
        "\nverdict: the one-command-buffer token graph removes per-tensor commit + round-trip\n\
         overhead. This is the mandatory pre-integration shape for dismantle (§D)."
    );
}
