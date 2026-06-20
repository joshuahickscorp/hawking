//! q4k_fast_recompute — convert all Q4_K tensors in a GGUF file to the
//! Q4K_FAST sub-block-contiguous re-layout, and emit a `.hawking`
//! sidecar file alongside the original.
//!
//! Usage:
//!     q4k_fast_recompute <input.gguf> <output.hawking>
//!
//! The output file contains every Q4_K tensor from the input, re-laid into
//! the Q4K_FAST format (160 bytes per 256-element block). Non-Q4_K tensors
//! are skipped — the runtime loader uses Q4K_FAST as a sidecar overlay,
//! falling back to the source GGUF for tensors not present.
//!
//! Per the haul brief: this tool is built but NOT run on the full 1.8GB
//! model in this session. Build-only validation.

use anyhow::{bail, Context, Result};
use hawking_core::gguf::{GgmlType, GgufFile};
use hawking_core::q4k_fast::{
    convert_q4k_tensor_to_fast, serialize_sidecar, src_hash_from_sha256_first8, WrittenTensor,
    Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS, Q4K_FAST_BLOCK_BYTES,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().context("missing <input.gguf>")?;
    let output = args.next().context("missing <output.hawking>")?;
    if args.next().is_some() {
        bail!(
            "unexpected extra argument; usage: q4k_fast_recompute <input.gguf> <output.hawking>"
        );
    }
    let input_path = PathBuf::from(input);
    let output_path = PathBuf::from(output);

    eprintln!("[q4k_fast] hashing source GGUF: {}", input_path.display());
    let src_bytes =
        fs::read(&input_path).with_context(|| format!("read input {}", input_path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&src_bytes);
    let digest = hasher.finalize();
    let mut first8 = [0u8; 8];
    first8.copy_from_slice(&digest[0..8]);
    let src_hash = src_hash_from_sha256_first8(first8);
    eprintln!("[q4k_fast] src_hash = 0x{:016x}", src_hash);

    let gguf = GgufFile::open(&input_path)
        .with_context(|| format!("open GGUF {}", input_path.display()))?;

    // Collect Q4_K tensors. Skip non-Q4_K (LM head / norms / Q6_K stay in
    // the source file; the sidecar only carries Q4_K re-layouts).
    let mut written = Vec::<WrittenTensor>::new();
    let mut n_q4k = 0usize;
    let mut n_skip = 0usize;
    for (name, info) in gguf.tensors.iter() {
        if info.dtype != GgmlType::Q4_K {
            n_skip += 1;
            continue;
        }
        n_q4k += 1;
        // Q4_K tensors are 2-D in GGUF (row-major). dims = [cols, rows] in
        // GGUF convention, or [n] for vectors. We require rank == 2.
        if info.dims.len() != 2 {
            eprintln!(
                "[q4k_fast] WARN: skipping {name}: unexpected rank {}",
                info.dims.len()
            );
            continue;
        }
        // GGUF stores dims fastest-axis-first: dims[0] = cols, dims[1] = rows.
        let cols = info.dims[0] as usize;
        let rows = info.dims[1] as usize;
        if cols % Q4K_BLOCK_ELEMS != 0 {
            eprintln!(
                "[q4k_fast] WARN: skipping {name}: cols {cols} not multiple of {Q4K_BLOCK_ELEMS}"
            );
            continue;
        }
        let blocks_per_row = cols / Q4K_BLOCK_ELEMS;
        let expected_bytes = rows * blocks_per_row * Q4K_BLOCK_BYTES;
        if info.byte_size as usize != expected_bytes {
            bail!(
                "tensor {name}: byte_size mismatch (got {} expected {})",
                info.byte_size,
                expected_bytes
            );
        }
        let bytes = gguf
            .tensor_bytes(name)
            .with_context(|| format!("tensor_bytes for {name}"))?;
        let fast_bytes = convert_q4k_tensor_to_fast(bytes, rows, cols);
        debug_assert_eq!(
            fast_bytes.len(),
            rows * blocks_per_row * Q4K_FAST_BLOCK_BYTES
        );
        let len = fast_bytes.len() as u64;
        written.push(WrittenTensor {
            name: name.clone(),
            rows: rows as u32,
            cols: cols as u32,
            byte_len: len,
            bytes: fast_bytes,
        });
    }

    eprintln!(
        "[q4k_fast] {} Q4_K tensors converted; {} non-Q4_K tensors skipped",
        n_q4k, n_skip
    );

    let file = serialize_sidecar(src_hash, written);
    fs::write(&output_path, &file)
        .with_context(|| format!("write sidecar {}", output_path.display()))?;
    eprintln!(
        "[q4k_fast] wrote {} ({} bytes)",
        output_path.display(),
        file.len()
    );
    Ok(())
}
