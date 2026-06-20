
#[cfg(test)]
mod tests {
    use crate::gemv::decode_tensor_q12;
    use crate::loader::StrandModel;
    use crate::footprint_bytes;
    use std::io::Write;
    use std::time::Instant;
    use strand_quant::decode::decode_tensor_fixed;
    use strand_quant::encode::{encode_tensor, EncodedTensor};
    use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};
    use strand_quant::TrellisConfig;

    struct Synth {
        name: &'static str,
        out: usize,
        inf: usize,
        bpw: f64,
        cfg: TrellisConfig,
        enc: EncodedTensor,
    }

    fn synth_weights(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = (i as f32) * 0.0007 + (seed as f32) * 1.31;
                
                0.6 * (t).sin() + 0.3 * (2.0 * t + 0.5).cos() + 0.05 * ((i as f32) / (n as f32))
            })
            .collect()
    }

    fn build_model() -> Vec<Synth> {
        
        let specs: &[(&'static str, usize, usize, f64)] = &[
            ("blk.0.attn_q.weight", 32, 256, 3.0),
            ("blk.0.attn_k.weight", 16, 512, 2.0),
            ("blk.0.ffn_down.weight", 24, 768, 4.0),
            ("blk.0.ffn_up.weight", 40, 256, 3.0),
        ];
        specs
            .iter()
            .enumerate()
            .map(|(s, &(name, out, inf, bpw))| {
                let w = synth_weights(s + 1, out * inf);
                let cfg = TrellisConfig::for_bpw(bpw);
                let enc = encode_tensor(&w, &cfg);
                Synth { name, out, inf, bpw, cfg, enc }
            })
            .collect()
    }

    fn write_model(model: &[Synth]) -> (std::path::PathBuf, usize) {
        
        let shapes: Vec<[u64; 2]> = model.iter().map(|s| [s.out as u64, s.inf as u64]).collect();
        let pts: Vec<PackedTensorV2> = model
            .iter()
            .zip(shapes.iter())
            .map(|(s, shape)| PackedTensorV2 {
                base: PackedTensor {
                    name: s.name,
                    shape,
                    rht_seed: 0,
                    l_bits: s.cfg.l_bits as u8,
                    k_bits: s.cfg.k_bits as u8,
                    vec_dim: s.cfg.vec_dim() as u8,
                    enc: &s.enc,
                },
                block_len: s.cfg.block_len as u32,
            })
            .collect();
        let buf = write_strand_v2(&pts, [0u8; 32], true).expect("write_strand_v2 (STRICT)");
        let file_len = buf.len();

        let mut path = std::env::temp_dir();
        let pid = std::process::id();
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!("strand_e2e_{pid}_{uniq}.strand"));
        let mut f = std::fs::File::create(&path).expect("create temp .strand");
        f.write_all(&buf).expect("write temp .strand");
        f.sync_all().ok();
        (path, file_len)
    }

    #[test]
    fn e2e_v2_runtime_decode_is_bit_exact() {
        let model = build_model();
        let (path, file_len) = write_model(&model);

        let loaded = StrandModel::open(&path).expect("StrandModel::open");
        assert_eq!(
            loaded.header().tensors.len(),
            model.len(),
            "tensor count round-trip"
        );
        assert!(
            loaded.header().all_strict(),
            "all shapes are in % block_len == 0, so ALL_STRICT must be set"
        );
        let loaded_names: Vec<&str> = loaded.tensor_names().collect();
        let want_names: Vec<&str> = model.iter().map(|s| s.name).collect();
        assert_eq!(loaded_names, want_names, "tensor name/order round-trip");

        let mut total_weights = 0usize;
        let mut total_q12: usize = 0; 
        let mut decode_secs = 0.0f64;

        println!(
            "\n=== strand-decode-kernel E2E: encode -> .strand v2 ({file_len} bytes) -> mmap-load -> decode ==="
        );
        println!("  loaded {} tensors, ALL_STRICT={}", model.len(), loaded.header().all_strict());

        for s in &model {
            let n = s.out * s.inf;
            total_weights += n;

            let h = loaded.tensor_header(s.name).expect("tensor_header");
            assert_eq!(h.shape, vec![s.out as u64, s.inf as u64], "{}: shape", s.name);
            assert_eq!(h.l_bits as u32, s.cfg.l_bits, "{}: l_bits", s.name);
            assert_eq!(h.k_bits as u32, s.cfg.k_bits, "{}: k_bits", s.name);
            assert_eq!(h.block_len as usize, s.cfg.block_len, "{}: block_len", s.name);
            assert_eq!(h.n_blocks, s.enc.blocks.len(), "{}: n_blocks", s.name);

            let q_ref = decode_tensor_fixed(&s.enc, &s.cfg);
            assert_eq!(q_ref.len(), n, "{}: reference decode length", s.name);

            let cfg_back = loaded.config_for(h);
            let enc_back = loaded.encoded_tensor(s.name).expect("encoded_tensor");
            let q_loader = decode_tensor_fixed(&enc_back, &cfg_back);
            assert_eq!(
                q_loader, q_ref,
                "{}: loader-reconstructed decode_tensor_fixed != reference (DETERMINISM BREAK)",
                s.name
            );

            let t0 = Instant::now();
            let q_runtime = decode_tensor_q12(&loaded, s.name).expect("decode_tensor_q12");
            decode_secs += t0.elapsed().as_secs_f64();
            total_q12 += q_runtime.len();
            assert_eq!(
                q_runtime, q_ref,
                "{}: runtime decode_tensor_q12 (decode_lean) != reference (DETERMINISM BREAK)",
                s.name
            );

            println!(
                "  {:<24} [{:>3}x{:<4}] bpw={:.1} k={} l={} n_blocks={} -> {} Q12 weights, bit-exact OK",
                s.name, s.out, s.inf, s.bpw, s.cfg.k_bits, s.cfg.l_bits, s.enc.blocks.len(), n
            );
        }

        let bytes_per_weight = file_len as f64 / total_weights as f64;
        
        let iters = 50usize;
        let t0 = Instant::now();
        let mut sink = 0i64;
        for _ in 0..iters {
            for s in &model {
                let q = decode_tensor_q12(&loaded, s.name).expect("decode_tensor_q12");
                sink = sink.wrapping_add(q[0] as i64).wrapping_add(q[q.len() - 1] as i64);
            }
        }
        let loop_secs = t0.elapsed().as_secs_f64();
        let mw_per_s = (total_weights * iters) as f64 / loop_secs / 1e6;
        
        assert!(sink != i64::MIN, "sink guard");

        println!(
            "  total weights = {}  file = {} bytes  => {:.3} bytes/weight (whole archive, incl. headers + page padding)",
            total_weights, file_len, bytes_per_weight
        );
        println!(
            "  CPU decode throughput (decode_tensor_q12 over all {} tensors, {} iters): {:.1} Mweights/s ({:.3} ms/full-pass)",
            model.len(),
            iters,
            mw_per_s,
            loop_secs / iters as f64 * 1e3
        );
        
        let _ = total_q12;
        println!(
            "  (single-pass decode of all tensors: {} Q12 weights in {:.3} ms)\n",
            total_q12,
            decode_secs * 1e3
        );

        assert!(
            bytes_per_weight > 0.0 && bytes_per_weight < 8.0,
            "bytes/weight {bytes_per_weight} out of sane range"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn e2e_footprint_bytes_per_token_report() {
        let model = build_model();
        let total_weights: usize = model.iter().map(|s| s.out * s.inf).sum();
        let bf16 = footprint_bytes(total_weights, 16.0);

        println!(
            "\n=== strand-decode-kernel E2E footprint / bytes-per-token ({} synthetic weights across {} tensors) ===",
            total_weights,
            model.len()
        );
        println!("  bf16 baseline: {} bytes/token ({:.2} KB)", bf16, bf16 as f64 / 1024.0);

        for &bpw in &[3.0f64, 2.0, 1.5] {
            let bytes = footprint_bytes(total_weights, bpw);
            let shrink = bf16 as f64 / bytes as f64;
            println!(
                "  bpw={:.1}  {:>7} bytes/token  {:5.2}x smaller than bf16  (~{:.1}x less weight traffic/token)",
                bpw, bytes, shrink, 16.0 / bpw
            );
            
            assert!(bytes < bf16, "bpw {bpw}: packed must be < bf16");
        }

        let f3 = footprint_bytes(total_weights, 3.0);
        let f2 = footprint_bytes(total_weights, 2.0);
        let f15 = footprint_bytes(total_weights, 1.5);
        assert!(f3 > f2 && f2 > f15, "footprint must shrink monotonically with bpw");

        let big = 7_000_000_000usize;
        println!(
            "  (reference: a 7B model is {:.1}x smaller at 3 bpw than bf16 — {:.2} GB vs {:.2} GB)",
            footprint_bytes(big, 16.0) as f64 / footprint_bytes(big, 3.0) as f64,
            footprint_bytes(big, 3.0) as f64 / 1e9,
            footprint_bytes(big, 16.0) as f64 / 1e9,
        );
        assert!(footprint_bytes(big, 3.0) * 5 < footprint_bytes(big, 16.0));
    }
}
