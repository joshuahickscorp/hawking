//! tq_bake — offline TQ (Trellis-Quant) weight baker.
//!
//! Reads a GGUF model, dequantises the selected projection tensors to f32, encodes
//! them with the absorbed strand-quant trellis quantiser, and writes a `.tq` v2
//! artifact (STR2 wire format). See `docs/strand/STRAND-dismantle-wiring.md` Step 4.
//!
//! Quality note: when the source GGUF is itself quantised (e.g. Q4_K), the f32 encode
//! source is the dequantised reconstruction, so the `.tq` is a quant-of-a-quant —
//! fine for validating the pipeline, NOT a quality artifact. Bake from an f16/f32
//! source for a real artifact.
//!
//! When `--rht cols` is passed, each selected weight is rotated with the per-COLUMN-sign
//! RHT (seed derived from the tensor name) BEFORE the trellis encode, and the archive
//! stamps the per-tensor col-RHT flag (bit 3) so the decoder serves it by transforming
//! the activation once instead of inverting the weights per row.
//!
//!   cargo run -p tq_bake_tool -- <in.gguf> <out.tq> [--bpw 3.34] [--match ffn_] [--limit N] [--rht none|cols]

use anyhow::{bail, Context, Result};
use dismantle_core::gguf::GgufFile;
use dismantle_core::quant::dequant_to_f32;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use strand_quant::encode::{encode_tensor, EncodedTensor};
use strand_quant::format::{write_strand_v2, write_strand_v2_rht, PackedTensor, PackedTensorV2};
use strand_quant::gate_utils::rht_seed_for;
use strand_quant::rht::{rht_forward_cols_inplace, RhtConfig};
use strand_quant::TrellisConfig;

/// Weight-domain Randomized Hadamard Transform applied before the trellis encode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RhtMode {
    /// No RHT — encode the dequantised weights as-is (default).
    None,
    /// Per-COLUMN-sign RHT: rotate each row's `in_features` with a column-shared sign
    /// then FWHT, so the decoder can transform the activation once and reuse it for
    /// every output row (the cheap-serving variant — see `strand_quant::rht`).
    Cols,
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    bpw: f64,
    matcher: String,
    limit: usize,
    rht: RhtMode,
}

fn parse_args() -> Result<Args> {
    let mut input = None;
    let mut output = None;
    let mut bpw = 3.34f64;
    let mut matcher = "ffn_".to_string();
    let mut limit = usize::MAX;
    let mut rht = RhtMode::None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bpw" => bpw = it.next().context("--bpw needs a value")?.parse()?,
            "--match" => matcher = it.next().context("--match needs a value")?,
            "--limit" => limit = it.next().context("--limit needs a value")?.parse()?,
            "--rht" => {
                rht = match it.next().context("--rht needs <none|cols>")?.as_str() {
                    "none" => RhtMode::None,
                    "cols" => RhtMode::Cols,
                    other => bail!("--rht must be 'none' or 'cols', got {other:?}"),
                }
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: tq_bake <in.gguf> <out.tq> [--bpw 3.34] [--match ffn_] [--limit N] [--rht none|cols]"
                );
                std::process::exit(0);
            }
            _ if input.is_none() => input = Some(PathBuf::from(a)),
            _ if output.is_none() => output = Some(PathBuf::from(a)),
            _ => bail!("unexpected arg {a:?}"),
        }
    }
    Ok(Args {
        input: input.context("missing <input.gguf>")?,
        output: output.context("missing <output.tq>")?,
        bpw,
        matcher,
        limit,
        rht,
    })
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let gguf =
        GgufFile::open(&args.input).with_context(|| format!("open {}", args.input.display()))?;
    let src_sha: [u8; 32] = Sha256::digest(std::fs::read(&args.input)?).into();
    eprintln!(
        "[tq_bake] {} : {} tensors, src_sha256={}",
        args.input.display(),
        gguf.tensor_order.len(),
        hex(&src_sha)
    );

    let cfg = TrellisConfig::for_bpw(args.bpw);
    // Realized-config honesty: `for_bpw` rounds bpw -> k = round(bpw).clamp(1,4) and
    // L = (k+4).min(14), so a fractional `--bpw` is silently coerced to an integer k.
    // Print exactly what the trellis will use, and warn if it diverged from the ask.
    eprintln!(
        "[tq_bake] trellis config: L={} (num_states={}), k={} (~{} bpw payload), block_len={}",
        cfg.l_bits,
        cfg.num_states(),
        cfg.k_bits,
        cfg.k_bits,
        cfg.block_len
    );
    if (args.bpw - cfg.k_bits as f64).abs() > 0.05 {
        eprintln!(
            "[tq_bake] note: --bpw {:.3} was rounded/capped to k={} bits/weight \
             (for_bpw uses k = round(bpw).clamp(1,4)); realized payload is {} bpw, not {:.3}",
            args.bpw, cfg.k_bits, cfg.k_bits, args.bpw
        );
    }

    // 1. select + dequant + (optional RHT) + encode each matching 2-D projection weight.
    // `ragged` collects tensors whose in_features is NOT a multiple of block_len: those
    // violate the GPU bitslice deploy invariant, so the archive cannot be written strict.
    let mut encoded: Vec<(String, [u64; 2], EncodedTensor)> = Vec::new();
    let mut ragged: Vec<String> = Vec::new();
    // Per-tensor RHT seed, index-aligned with `encoded`; 0 == no RHT for that tensor.
    let mut seeds: Vec<u64> = Vec::new();
    for name in &gguf.tensor_order {
        if encoded.len() >= args.limit {
            break;
        }
        if !(name.contains(&args.matcher) && name.ends_with(".weight")) {
            continue;
        }
        let info = gguf.tensor(name).expect("tensor in order must exist");
        if info.dims.len() != 2 {
            continue;
        }
        let in_f = info.dims[0];
        let out_f = info.dims[1];
        let bytes = gguf.tensor_bytes(name).context("tensor bytes")?;
        let mut w = dequant_to_f32(info, bytes).with_context(|| format!("dequant {name}"))?;

        // Deploy invariant: GPU bitslice serving requires in_features % block_len == 0.
        if in_f % cfg.block_len as u64 != 0 {
            ragged.push(name.clone());
        }

        // Optional weight-domain RHT, applied BEFORE the trellis encode. `cols` mode
        // rotates each row's `in_features` with a column-shared sign + FWHT; the seed is
        // derived deterministically from the tensor name so the decoder can reconstruct
        // the activation transform without storing per-tensor random state inline.
        let rht_seed = match args.rht {
            RhtMode::None => 0u64,
            RhtMode::Cols => {
                let seed = rht_seed_for(name);
                rht_forward_cols_inplace(&mut w, &RhtConfig::from_seed(seed), in_f as usize);
                seed
            }
        };

        let mut enc = encode_tensor(&w, &cfg);
        // The encoder always returns has_rht_seed=false (it never sees the RHT). When we
        // baked a col-RHT in, stamp the seed presence so the wire flag byte's bit-1
        // (has_rht_seed) is set and `total_bpw` bills the 64-bit seed honestly. The
        // per-tensor bit-3 (rht_cols) is set separately via write_strand_v2_rht's mask.
        if args.rht == RhtMode::Cols {
            enc.has_rht_seed = true;
        }
        eprintln!(
            "[tq_bake]   + {name}  [{out_f}x{in_f}] {:?} -> {} blocks{}",
            info.dtype,
            enc.blocks.len(),
            if rht_seed != 0 {
                format!(" (rht=cols seed={rht_seed:#018x})")
            } else {
                String::new()
            }
        );
        encoded.push((name.clone(), [out_f, in_f], enc));
        // Record the seed parallel to `encoded` (index-aligned) for the writer.
        seeds.push(rht_seed);
    }
    if encoded.is_empty() {
        bail!("no tensors matched {:?} (.weight, 2-D)", args.matcher);
    }

    // 2. pack + write the .tq archive (STR2 wire format).
    // Deployability guard: only write `strict` (ALL_STRICT flag) when every tensor's
    // in_features is block_len-aligned. A non-strict archive is NOT GPU-bitslice
    // deployable — say so loudly rather than silently flipping the flag off.
    let strict = ragged.is_empty();
    if !strict {
        eprintln!(
            "[tq_bake] WARNING: {} tensor(s) have in_features not divisible by block_len={} \
             — archive written NON-STRICT (ALL_STRICT off); these are NOT deployable on the \
             GPU bitslice path:",
            ragged.len(),
            cfg.block_len
        );
        for name in &ragged {
            eprintln!("[tq_bake] WARNING:   ragged: {name}");
        }
    }

    let packed: Vec<PackedTensorV2> = encoded
        .iter()
        .zip(seeds.iter())
        .map(|((name, shape, enc), &seed)| PackedTensorV2 {
            base: PackedTensor {
                name: name.as_str(),
                shape: &shape[..],
                rht_seed: seed,
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc,
            },
            block_len: cfg.block_len as u32,
        })
        .collect();
    let file = match args.rht {
        RhtMode::None => write_strand_v2(&packed, src_sha, strict)
            .map_err(|e| anyhow::anyhow!("write_strand_v2: {e}"))?,
        RhtMode::Cols => {
            // Per-tensor bit-3 (col-RHT) for every tensor we rotated. The mask indexes
            // tensors positionally; in `cols` mode every selected tensor was rotated.
            let rht_cols_mask = vec![true; packed.len()];
            write_strand_v2_rht(&packed, src_sha, strict, false, &rht_cols_mask)
                .map_err(|e| anyhow::anyhow!("write_strand_v2_rht: {e}"))?
        }
    };
    std::fs::write(&args.output, &file)
        .with_context(|| format!("write {}", args.output.display()))?;

    // 3. validate: read the archive back and decode tensor 0 (round-trip sanity).
    let back = strand_quant::sideinfo_wire::read_strand_v2_applied(&file)
        .map_err(|e| anyhow::anyhow!("read-back failed: {e}"))?;
    if back.len() != encoded.len() {
        bail!(
            "read-back tensor count {} != written {}",
            back.len(),
            encoded.len()
        );
    }
    let decoded0 = strand_quant::decode::decode_tensor_fixed(&back[0].base.enc, &cfg);
    let total_w: u64 = encoded.iter().map(|(_, s, _)| s[0] * s[1]).sum();
    let bpw_real = (file.len() as f64 * 8.0) / total_w as f64;
    let rht_label = match args.rht {
        RhtMode::None => "none",
        RhtMode::Cols => "cols",
    };
    eprintln!(
        "[tq_bake] wrote {} ({} bytes, {} tensors, ~{:.3} bpw, strict={}, rht={}); \
         read-back OK, tensor0 decoded {} weights",
        args.output.display(),
        file.len(),
        encoded.len(),
        bpw_real,
        strict,
        rht_label,
        decoded0.len()
    );
    Ok(())
}
