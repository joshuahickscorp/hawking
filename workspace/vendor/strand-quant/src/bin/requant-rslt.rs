
#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use strand_quant::encode::{encode_tensor_with, EncodeOpts, EncodedTensor};
use strand_quant::format::{
    read_strand_v2, PackedTensor, PackedTensorV2,
    write_strand_v2, OwnedTensorV2,
};
use strand_quant::rslt::{self, RsltSection};
use strand_quant::{CodebookMode, TrellisConfig};

fn usage() -> ! {
    eprintln!(
        "usage: requant-rslt --model <model.strand> --rslt <rslt.strand> --out <model-v2.strand>\n\
         \n\
         Re-encode a .strand v2 model using RSLT inference access statistics\n\
         to guide per-block silence pressure.\n\
         \n\
         Options:\n\
           --model   Original model .strand v2 artifact (source of weights)\n\
           --rslt    .strand v2 artifact that carries an RSLT section (may be same\n\
                     file as --model if RSLT was appended in-place)\n\
           --out     Output path for the re-encoded .strand v2 artifact\n\
           --help    Print this message\n\
         \n\
         HEURISTIC NOTE: temperature thresholds (0.3, 0.7) and silence_bonus\n\
         values are heuristic prototypes — validate by PPL comparison before\n\
         production use (see module docs)."
    );
    std::process::exit(1)
}

struct Args {
    model: PathBuf,
    rslt:  PathBuf,
    out:   PathBuf,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        usage();
    }
    let get = |flag: &str| -> PathBuf {
        let pos = args.iter().position(|a| a == flag)
            .unwrap_or_else(|| {
                eprintln!("requant-rslt: missing required flag {flag}");
                usage();
            });
        args.get(pos + 1)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                eprintln!("requant-rslt: {flag} requires a value");
                usage();
            })
    };
    Args {
        model: get("--model"),
        rslt:  get("--rslt"),
        out:   get("--out"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tier {
    Hot,
    Warm,
    Cold,
}

fn classify(temperature: f64) -> Tier {
    if temperature > 0.7 {
        Tier::Hot
    } else if temperature > 0.3 {
        Tier::Warm
    } else {
        Tier::Cold
    }
}

fn tier_silence_bonus(tier: Tier) -> f64 {
    match tier {
        Tier::Hot  => 0.0,   
        Tier::Warm => 0.0,   
        Tier::Cold => 1e-4,  
    }
}

struct TierCounts {
    hot:  usize,
    warm: usize,
    cold: usize,
    never: usize, 
}

fn re_encode_tensor(
    name: &str,
    src: &OwnedTensorV2,
    counts_slice: &[u32],
    max_count: u32,
    default_opts: &EncodeOpts,
    cfg: &TrellisConfig,
) -> (EncodedTensor, TierCounts) {
    
    if max_count == 0 {
        let enc = encode_tensor_with(&recon_f32(src), cfg, default_opts);
        let nb = enc.blocks.len();
        return (enc, TierCounts { hot: 0, warm: 0, cold: 0, never: nb });
    }

    let weights_f32 = recon_f32(src);
    let nb = src.base.enc.blocks.len();
    let block_len = src.block_len as usize;

    let mut hot_n  = 0usize;
    let mut warm_n = 0usize;
    let mut cold_n = 0usize;
    let mut never_n = 0usize;

    let mut merged_bits: Vec<u8> = Vec::new();
    let mut merged_blocks: Vec<strand_quant::encode::BlockMeta> = Vec::new();
    let mut total = 0usize;

    let silence_before: f64 = {
        let s = &src.base.enc;
        silence_rate(s)
    };

    for b in 0..nb {
        let c = if b < counts_slice.len() { counts_slice[b] } else { 0 };
        let temperature = c as f64 / max_count as f64;
        let tier = if c == 0 && max_count > 0 {
            never_n += 1;
            Tier::Cold
        } else {
            classify(temperature)
        };

        match tier {
            Tier::Hot  => hot_n  += 1,
            Tier::Warm => warm_n += 1,
            Tier::Cold => cold_n += 1,
        }

        let bonus = tier_silence_bonus(tier);
        let block_opts = EncodeOpts { silence_bonus: bonus, ..*default_opts };

        let w_start = b * block_len;
        let w_end = ((b + 1) * block_len).min(weights_f32.len());
        let block_weights = &weights_f32[w_start..w_end];

        let enc_b = encode_tensor_with(block_weights, cfg, &block_opts);

        let k = cfg.k_bits as usize;
        let n_steps = enc_b.total;
        let block_bits = (n_steps * k).div_ceil(8);
        merged_bits.extend_from_slice(&enc_b.bits[..block_bits.min(enc_b.bits.len())]);
        
        let mut bm = enc_b.blocks.into_iter().next().unwrap_or_else(|| {
            
            strand_quant::encode::BlockMeta {
                scale_q: 0,
                sub_scales: Vec::new(),
                min_base_q: 0,
                mins: Vec::new(),
                init_state: 0,
                n: 0,
            }
        });
        bm.n = block_weights.len() as u32;
        merged_blocks.push(bm);
        total += block_weights.len();
    }

    while merged_bits.len() % 8 != 0 || merged_bits.len() < 8 {
        merged_bits.push(0);
    }

    let new_enc = EncodedTensor {
        bits: merged_bits,
        blocks: merged_blocks,
        total,
        has_rht_seed: src.base.enc.has_rht_seed,
        tail_biting: default_opts.tail_biting,
        has_affine_min: default_opts.affine_min,
    };

    let silence_after = silence_rate(&new_enc);
    let _ = (silence_before, silence_after, name); 

    (new_enc, TierCounts { hot: hot_n, warm: warm_n, cold: cold_n, never: never_n })
}

fn silence_rate(enc: &EncodedTensor) -> f64 {
    if enc.total == 0 { return 0.0; }
    
    let zero_bytes = enc.bits.iter().filter(|&&b| b == 0).count();
    zero_bytes as f64 / enc.bits.len() as f64
}

fn recon_f32(src: &OwnedTensorV2) -> Vec<f32> {
    use strand_quant::decode::decode_tensor_fixed;
    
    let cfg = TrellisConfig {
        l_bits: src.base.l_bits as u32,
        k_bits: src.base.k_bits as u32,
        block_len: src.block_len as usize,
        vec_dim: src.base.vec_dim as u32,
        codebook_mode: CodebookMode::StoredLut,
    };
    let q12 = decode_tensor_fixed(&src.base.enc, &cfg);
    let inv = 1.0f32 / 4096.0;
    q12.iter().map(|&q| q as f32 * inv).collect()
}

fn main() {
    let args = parse_args();

    let model_buf = fs::read(&args.model).unwrap_or_else(|e| {
        eprintln!("requant-rslt: read model {:?}: {e}", args.model);
        std::process::exit(1);
    });
    let model_tensors = read_strand_v2(&model_buf).unwrap_or_else(|e| {
        eprintln!("requant-rslt: parse model {:?}: {e}", args.model);
        std::process::exit(1);
    });

    let rslt_buf = fs::read(&args.rslt).unwrap_or_else(|e| {
        eprintln!("requant-rslt: read rslt {:?}: {e}", args.rslt);
        std::process::exit(1);
    });
    let rslt: RsltSection = rslt::read_rslt_bytes(&rslt_buf, true)
        .unwrap_or_else(|e| {
            eprintln!("requant-rslt: parse RSLT from {:?}: {e}", args.rslt);
            std::process::exit(1);
        })
        .unwrap_or_else(|| {
            eprintln!(
                "requant-rslt: no RSLT section found in {:?}\n\
                 (has the file been annotated by an inference run?)",
                args.rslt
            );
            std::process::exit(1);
        });

    let max_count: u32 = rslt
        .block_counts
        .iter()
        .flat_map(|v| v.iter())
        .copied()
        .max()
        .unwrap_or(0);

    println!("requant-rslt: loaded {} tensors, RSLT max_count={}", model_tensors.len(), max_count);

    let default_opts = EncodeOpts {
        adaptive: true,
        tail_biting: false,
        affine_min: false,
        silence_bonus: 0.0,
        entropy_bonus_scale: 0.0,
        entropy_bonus_two_pass: false,
    };

    let mut new_tensors: Vec<(OwnedTensorV2, EncodedTensor)> = Vec::new();
    let mut total_hot  = 0usize;
    let mut total_warm = 0usize;
    let mut total_cold = 0usize;
    let mut total_never = 0usize;
    let mut total_blocks_processed = 0usize;

    let start = std::time::Instant::now();

    for (ti, src) in model_tensors.iter().enumerate() {
        let cfg = TrellisConfig {
            l_bits: src.base.l_bits as u32,
            k_bits: src.base.k_bits as u32,
            block_len: src.block_len as usize,
            vec_dim: src.base.vec_dim as u32,
            codebook_mode: CodebookMode::StoredLut,
        };
        let counts_slice: &[u32] = rslt
            .block_counts
            .get(ti)
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        let (new_enc, tiers) = re_encode_tensor(
            &src.base.name,
            src,
            counts_slice,
            max_count,
            &default_opts,
            &cfg,
        );
        total_hot   += tiers.hot;
        total_warm  += tiers.warm;
        total_cold  += tiers.cold;
        total_never += tiers.never;
        total_blocks_processed += src.base.enc.blocks.len();

        let nb = src.base.enc.blocks.len();
        println!(
            "  [{:>3}/{:>3}] {:40}  blocks={:4}  hot={:4} warm={:4} cold={:4} never={:4}",
            ti + 1, model_tensors.len(), &src.base.name,
            nb, tiers.hot, tiers.warm, tiers.cold, tiers.never,
        );

        new_tensors.push((src.clone(), new_enc));
    }

    println!(
        "\nblocks by tier: hot={total_hot} warm={total_warm} cold={total_cold} never={total_never} \
         / {total_blocks_processed} total ({:.1?} elapsed)",
        start.elapsed()
    );

    let source_sha256 = model_buf.get(20..52)
        .and_then(|s| <[u8; 32]>::try_from(s).ok())
        .unwrap_or([0u8; 32]);

    let packed: Vec<PackedTensorV2> = new_tensors
        .iter()
        .map(|(src, new_enc)| PackedTensorV2 {
            base: PackedTensor {
                name: &src.base.name,
                shape: &src.base.shape,
                rht_seed: src.base.rht_seed,
                l_bits: src.base.l_bits,
                k_bits: src.base.k_bits,
                vec_dim: src.base.vec_dim,
                enc: new_enc,
            },
            block_len: src.block_len,
        })
        .collect();

    let out_bytes = write_strand_v2(&packed, source_sha256, false).unwrap_or_else(|e| {
        eprintln!("requant-rslt: write_strand_v2: {e}");
        std::process::exit(1);
    });

    fs::write(&args.out, &out_bytes).unwrap_or_else(|e| {
        eprintln!("requant-rslt: write {:?}: {e}", args.out);
        std::process::exit(1);
    });

    println!("\n=== bpw comparison ===");
    let orig_total_weights: usize = model_tensors.iter().map(|t| t.base.enc.total).sum();
    let new_total_weights: usize = new_tensors.iter().map(|(_, e)| e.total).sum();
    let orig_total_bits: usize = model_tensors.iter().map(|t| t.base.enc.bits.len() * 8).sum();
    let new_total_bits: usize = new_tensors.iter().map(|(_, e)| e.bits.len() * 8).sum();
    let orig_bpw = if orig_total_weights > 0 { orig_total_bits as f64 / orig_total_weights as f64 } else { 0.0 };
    let new_bpw  = if new_total_weights > 0  { new_total_bits as f64  / new_total_weights as f64  } else { 0.0 };

    println!("  original : {orig_total_bits:12} bits  {orig_total_weights:12} weights  {orig_bpw:.4} bpw");
    println!("  re-encoded: {new_total_bits:12} bits  {new_total_weights:12} weights  {new_bpw:.4} bpw");
    println!("  written   : {:?}", args.out);

    println!("\nHEURISTIC NOTE: temperature thresholds (0.3, 0.7) and silence_bonus values");
    println!("require PPL validation against the original model before production use.");
}
