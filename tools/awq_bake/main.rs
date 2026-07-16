//! awq_bake_sidecar — offline tool: bake AWQ smoothing (W' = W * s) into
//! every Q4_K projection tensor in a GGUF file, re-quantize back to Q4_K,
//! and emit a Q4K_FAST-format sidecar.
//!
//! Usage:
//!     awq_bake_sidecar <input.gguf> <smoothing.json> <output.hawking>
//!
//! Math (AWQ smoothing, alpha already folded into the saved factors):
//!     For each Q4_K weight W of shape [out_rows, in_cols] and a smoothing
//!     vector s of length in_cols:
//!         W'[r, c] = W[r, c] * s[c]      (s broadcasts over the rows axis)
//!
//!     The activation x of shape [in_cols] is divided by s elementwise at
//!     runtime BEFORE per-block int8 quantization, restoring
//!         (x / s) · (W * s).T == x · W.T
//!     while reshaping which channels look "hot" to the int8 quantizer.
//!
//! Output layout: the sidecar is the same wire format the Q4K_FAST runtime
//! loader already understands (`crates/hawking-core/src/q4k_fast.rs`), so
//! the only Rust change needed at decode time is to swap the sidecar path
//! from the plain Q4K_FAST file to this AWQ-baked one when the AWQ env gate
//! is set.

use anyhow::{bail, Context, Result};
use hawking_core::gguf::{GgmlType, GgufFile};
use hawking_core::q4k_fast::{
    convert_q4k_tensor_to_fast, serialize_sidecar, src_hash_from_sha256_first8, WrittenTensor,
    Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS,
};
use hawking_core::quant::{dequant_into, quantize_q4_k};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Deserialize)]
struct AwqSmoothing {
    schema: String,
    model: String,
    n_layers: usize,
    alpha: f32,
    method: String,
    smoothing_factors: HashMap<String, Vec<f32>>,
}

/// Map a GGUF tensor name (e.g. "blk.0.attn_q.weight") to its AWQ
/// smoothing JSON key (e.g. "layer_0_q_proj"). Returns None for tensors
/// that don't have an AWQ entry (LM head, embeddings, norms, biases).
fn awq_key_for_gguf(name: &str) -> Option<String> {
    let s = name.strip_prefix("blk.")?;
    let dot = s.find('.')?;
    let layer: usize = s[..dot].parse().ok()?;
    let rest = &s[dot + 1..];
    let site = match rest {
        "attn_q.weight" => "q_proj",
        "attn_k.weight" => "k_proj",
        "attn_v.weight" => "v_proj",
        "attn_output.weight" => "o_proj",
        "ffn_gate.weight" => "gate_proj",
        "ffn_up.weight" => "up_proj",
        "ffn_down.weight" => "down_proj",
        _ => return None,
    };
    Some(format!("layer_{layer}_{site}"))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().context("missing <input.gguf>")?;
    let smoothing = args.next().context("missing <smoothing.json>")?;
    let output = args.next().context("missing <output.hawking>")?;
    if args.next().is_some() {
        bail!(
            "unexpected extra argument; usage: awq_bake_sidecar \
             <input.gguf> <smoothing.json> <output.hawking>"
        );
    }
    let input_path = PathBuf::from(input);
    let smoothing_path = PathBuf::from(smoothing);
    let output_path = PathBuf::from(output);

    eprintln!("[awq_bake] reading smoothing: {}", smoothing_path.display());
    let smoothing_bytes = fs::read(&smoothing_path)
        .with_context(|| format!("read smoothing {}", smoothing_path.display()))?;
    let awq: AwqSmoothing =
        serde_json::from_slice(&smoothing_bytes).context("parse smoothing JSON")?;
    if awq.schema != "awq-smoothing-v1" {
        bail!("unexpected smoothing schema: {}", awq.schema);
    }
    eprintln!(
        "[awq_bake] schema={} model={} n_layers={} alpha={} method={} factors={}",
        awq.schema,
        awq.model,
        awq.n_layers,
        awq.alpha,
        awq.method,
        awq.smoothing_factors.len()
    );

    eprintln!("[awq_bake] hashing source GGUF: {}", input_path.display());
    let src_bytes =
        fs::read(&input_path).with_context(|| format!("read input {}", input_path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&src_bytes);
    let digest = hasher.finalize();
    let mut first8 = [0u8; 8];
    first8.copy_from_slice(&digest[0..8]);
    let src_hash = src_hash_from_sha256_first8(first8);
    eprintln!("[awq_bake] src_hash = 0x{:016x}", src_hash);

    let gguf = GgufFile::open(&input_path)
        .with_context(|| format!("open GGUF {}", input_path.display()))?;

    let mut written = Vec::<WrittenTensor>::new();
    let mut n_baked = 0usize;
    let mut n_no_awq_skip = 0usize;
    let mut n_dtype_skip = 0usize;
    let mut n_warn_skip = 0usize;

    for (name, info) in gguf.tensors.iter() {
        if info.dtype != GgmlType::Q4_K {
            n_dtype_skip += 1;
            continue;
        }
        if info.dims.len() != 2 {
            eprintln!(
                "[awq_bake] WARN: {name}: unexpected rank {}",
                info.dims.len()
            );
            n_warn_skip += 1;
            continue;
        }
        let cols = info.dims[0] as usize;
        let rows = info.dims[1] as usize;
        if cols % Q4K_BLOCK_ELEMS != 0 {
            eprintln!("[awq_bake] WARN: {name}: cols {cols} not multiple of {Q4K_BLOCK_ELEMS}");
            n_warn_skip += 1;
            continue;
        }
        let blocks_per_row = cols / Q4K_BLOCK_ELEMS;
        let expected_src_bytes = rows * blocks_per_row * Q4K_BLOCK_BYTES;
        if info.byte_size as usize != expected_src_bytes {
            bail!(
                "tensor {name}: byte_size mismatch (got {} expected {})",
                info.byte_size,
                expected_src_bytes
            );
        }

        let key = match awq_key_for_gguf(name) {
            Some(k) => k,
            None => {
                n_no_awq_skip += 1;
                continue;
            }
        };
        let s = match awq.smoothing_factors.get(&key) {
            Some(v) => v,
            None => {
                eprintln!("[awq_bake] WARN: no AWQ entry for {key} (tensor {name}); skipping");
                n_no_awq_skip += 1;
                continue;
            }
        };
        if s.len() != cols {
            bail!(
                "tensor {name}: smoothing key {key} length {} != tensor cols {}",
                s.len(),
                cols
            );
        }

        // 1. Dequantize Q4_K -> f32 row-major [rows, cols].
        let bytes = gguf
            .tensor_bytes(name)
            .with_context(|| format!("tensor_bytes for {name}"))?;
        let n_elems = rows * cols;
        let mut w = vec![0.0f32; n_elems];
        dequant_into(GgmlType::Q4_K, bytes, &mut w).with_context(|| format!("dequant {name}"))?;

        // 2. Apply smoothing: W'[r, c] = W[r, c] * s[c]
        for r in 0..rows {
            let row = &mut w[r * cols..(r + 1) * cols];
            for c in 0..cols {
                row[c] *= s[c];
            }
        }

        // 3. Re-quantize to canonical Q4_K bytes.
        let q4_bytes_total = rows * blocks_per_row * Q4K_BLOCK_BYTES;
        let mut q4_bytes = vec![0u8; q4_bytes_total];
        quantize_q4_k(&w, &mut q4_bytes).with_context(|| format!("requantize {name}"))?;

        // 4. Convert canonical Q4_K -> Q4K_FAST sub-block-contiguous layout.
        let fast_bytes = convert_q4k_tensor_to_fast(&q4_bytes, rows, cols);
        let len = fast_bytes.len() as u64;
        written.push(WrittenTensor {
            name: name.clone(),
            rows: rows as u32,
            cols: cols as u32,
            byte_len: len,
            bytes: fast_bytes,
        });
        n_baked += 1;
        if n_baked % 36 == 0 {
            eprintln!("[awq_bake]   ... baked {n_baked} tensors");
        }
    }

    eprintln!(
        "[awq_bake] baked={} no-awq-skip={} dtype-skip={} warn-skip={}",
        n_baked, n_no_awq_skip, n_dtype_skip, n_warn_skip
    );

    let file = serialize_sidecar(src_hash, written);
    fs::write(&output_path, &file)
        .with_context(|| format!("write sidecar {}", output_path.display()))?;
    eprintln!(
        "[awq_bake] wrote {} ({} bytes, src_hash 0x{:016x})",
        output_path.display(),
        file.len(),
        src_hash
    );
    Ok(())
}
