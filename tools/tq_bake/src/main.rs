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
//!   cargo run -p tq_bake_tool -- <in.gguf> <out.tq> [--bpw 3.34] [--match ffn_] [--limit N]

use anyhow::{bail, Context, Result};
use dismantle_core::gguf::GgufFile;
use dismantle_core::quant::dequant_to_f32;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use strand_quant::encode::{encode_tensor, EncodedTensor};
use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};
use strand_quant::TrellisConfig;

struct Args {
    input: PathBuf,
    output: PathBuf,
    bpw: f64,
    matcher: String,
    limit: usize,
}

fn parse_args() -> Result<Args> {
    let mut input = None;
    let mut output = None;
    let mut bpw = 3.34f64;
    let mut matcher = "ffn_".to_string();
    let mut limit = usize::MAX;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bpw" => bpw = it.next().context("--bpw needs a value")?.parse()?,
            "--match" => matcher = it.next().context("--match needs a value")?,
            "--limit" => limit = it.next().context("--limit needs a value")?.parse()?,
            "-h" | "--help" => {
                eprintln!(
                    "usage: tq_bake <in.gguf> <out.tq> [--bpw 3.34] [--match ffn_] [--limit N]"
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

    // 1. select + dequant + encode each matching 2-D projection weight.
    let mut encoded: Vec<(String, [u64; 2], EncodedTensor)> = Vec::new();
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
        let w = dequant_to_f32(info, bytes).with_context(|| format!("dequant {name}"))?;
        let enc = encode_tensor(&w, &cfg);
        eprintln!(
            "[tq_bake]   + {name}  [{out_f}x{in_f}] {:?} -> {} blocks",
            info.dtype,
            enc.blocks.len()
        );
        encoded.push((name.clone(), [out_f, in_f], enc));
    }
    if encoded.is_empty() {
        bail!("no tensors matched {:?} (.weight, 2-D)", args.matcher);
    }

    // 2. pack + write the .tq archive (STR2 wire format).
    let packed: Vec<PackedTensorV2> = encoded
        .iter()
        .map(|(name, shape, enc)| PackedTensorV2 {
            base: PackedTensor {
                name: name.as_str(),
                shape: &shape[..],
                rht_seed: 0,
                l_bits: cfg.l_bits as u8,
                k_bits: cfg.k_bits as u8,
                vec_dim: cfg.vec_dim() as u8,
                enc,
            },
            block_len: cfg.block_len as u32,
        })
        .collect();
    let file = write_strand_v2(&packed, src_sha, false)
        .map_err(|e| anyhow::anyhow!("write_strand_v2: {e}"))?;
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
    eprintln!(
        "[tq_bake] wrote {} ({} bytes, {} tensors, ~{:.3} bpw); read-back OK, tensor0 decoded {} weights",
        args.output.display(),
        file.len(),
        encoded.len(),
        bpw_real,
        decoded0.len()
    );
    Ok(())
}
