//! Real GPT-OSS-120B parent-bound F2. Declarative adapter over the authoritative safetensors source:
//! it reads the router (`block.N.mlp.gate.weight`, BF16) and ONE selected MoE expert weight
//! (`mlp1_weight`, MXFP4) directly from the mmap'd shard, runs a bounded reference linear path, fits a
//! genuine sub-bit representation, executes it directly through Candidate C's operator, applies a
//! same-budget Doctor rescue, and measures divergence. Bounded: only the router + one expert slice are
//! read; the 120B model is never densely materialized.

use crate::safetensors::SafeTensors;
use crate::{mxfp4, subbit, Result};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct F2Real {
    pub parent: String,
    pub revision: String,
    pub shard: String,
    pub layer: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub selected_experts: Vec<usize>,
    pub expert: usize,
    pub router_tensor: String,
    pub expert_tensor: String,
    pub expert_full_shape: Vec<usize>,
    pub tested_slice: Vec<usize>,
    pub source_bytes_read: usize,
    pub reference_output_checksum: String,
    pub base_compact_bits: usize,
    pub doctor_bits: usize,
    pub metadata_bits: usize,
    pub complete_bpw: f64,
    pub untreated_divergence: f64,
    pub treated_divergence: f64,
    pub subbit_below_one_bpw: bool,
    pub doctor_within_same_rate: bool,
    pub repeatable: bool,
}

fn deterministic(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503))) >> 9) & 0xFF) as f32 / 128.0 - 1.0).collect()
}

/// Run bounded parent-bound F2 on `layer` using the given shard (which must contain that layer's mlp).
/// `slice` bounds the reference component (rows, cols) taken from the real expert weight.
pub fn run(shard_path: &std::path::Path, revision: &str, layer: usize, slice: (usize, usize)) -> Result<F2Real> {
    let st = SafeTensors::open(shard_path)?;
    let router_name = format!("block.{layer}.mlp.gate.weight");
    let expert_name = format!("block.{layer}.mlp.mlp1_weight");

    // --- router (BF16) -> route a deterministic hidden vector -> top-k experts ---
    let gate = st.get(&router_name)?;
    let (n_experts, hidden) = (gate.shape[0], gate.shape[1]);
    let gate_w = st.bf16_f32(&router_name)?;
    let gbias = st.bf16_f32(&format!("block.{layer}.mlp.gate.bias")).unwrap_or_else(|_| vec![0.0; n_experts]);
    let h = deterministic(hidden, 7);
    let mut rlogits = vec![0f32; n_experts];
    for e in 0..n_experts {
        let mut acc = 0f32;
        for i in 0..hidden {
            acc += gate_w[e * hidden + i] * h[i];
        }
        rlogits[e] = acc + gbias.get(e).copied().unwrap_or(0.0);
    }
    let top_k = 4;
    let mut idx: Vec<usize> = (0..n_experts).collect();
    idx.sort_by(|&a, &b| rlogits[b].partial_cmp(&rlogits[a]).unwrap());
    let selected: Vec<usize> = idx.into_iter().take(top_k).collect();
    let e0 = selected[0];

    // --- ONE expert's mlp1 weight (MXFP4) -> dequant a bounded slice to f32 ---
    let blk = st.get(&format!("{expert_name}.blocks"))?; // [n_experts, out, n_block, 16]
    st.get(&format!("{expert_name}.scales"))?; // [n_experts, out, n_block] (existence check)
    let out_features = blk.shape[1];
    let n_block = blk.shape[2];
    let in_features = n_block * 32;
    let bytes_per_row = n_block * 16; // = ceil(in/2)
    let scales_per_row = n_block;
    let blk_bytes = st.bytes(&format!("{expert_name}.blocks"))?;
    let sc_bytes = st.bytes(&format!("{expert_name}.scales"))?;
    let expert_blk_off = e0 * out_features * bytes_per_row;
    let expert_sc_off = e0 * out_features * scales_per_row;

    let (rows, cols) = (slice.0.min(out_features), slice.1.min(in_features));
    let mut wslice = vec![0f32; rows * cols];
    let mut rowbuf = vec![0f32; in_features];
    let mut source_bytes = gate_w.len() * 2; // router bf16 bytes
    for r in 0..rows {
        let bo = expert_blk_off + r * bytes_per_row;
        let so = expert_sc_off + r * scales_per_row;
        mxfp4::dequant_row(&blk_bytes[bo..bo + bytes_per_row], &sc_bytes[so..so + scales_per_row], in_features, &mut rowbuf);
        wslice[r * cols..(r + 1) * cols].copy_from_slice(&rowbuf[..cols]);
        source_bytes += bytes_per_row + scales_per_row;
    }

    // --- reference linear path on the real expert slice ---
    let probes: Vec<Vec<f32>> = (0..4).map(|s| deterministic(cols, s + 20)).collect();
    let mut refsum = 0f64;
    for x in &probes {
        for rr in 0..rows {
            let mut acc = 0f32;
            for c in 0..cols {
                acc += wslice[rr * cols + c] * x[c];
            }
            refsum += acc as f64;
        }
    }
    let refchk = crate::record::sha256_hex(format!("{refsum:.6}").as_bytes());

    // --- fit a genuine sub-bit representation, execute directly, measure divergence ---
    let r = 32.min(rows.min(cols) / 2).max(4);
    let sb = subbit::fit(&wslice, rows, cols, r);
    let untreated = subbit::output_divergence(&wslice, rows, cols, &sb, &probes);
    let base_bits = sb.bits();

    // --- same-budget Doctor rescue (still sub-bit) ---
    let budget = 0.99;
    let k = ((budget * (rows * cols) as f64 - base_bits as f64) / 48.0).max(0.0) as usize;
    let treated = subbit::doctor_rescue(&wslice, rows, cols, subbit::fit(&wslice, rows, cols, r), k, budget);
    let (treated_div, total_bits, doctor_bits) = match &treated {
        Some(t) => (subbit::output_divergence(&wslice, rows, cols, t, &probes), t.bits(), t.bits() - base_bits),
        None => (untreated, base_bits, 0),
    };
    let complete_bpw = total_bits as f64 / (rows * cols) as f64;

    // repeatability: refit + re-diverge must match
    let sb2 = subbit::fit(&wslice, rows, cols, r);
    let repeatable = (subbit::output_divergence(&wslice, rows, cols, &sb2, &probes) - untreated).abs() < 1e-9;

    Ok(F2Real {
        parent: "openai/gpt-oss-120b".into(),
        revision: revision.into(),
        shard: shard_path.file_name().unwrap().to_string_lossy().into(),
        layer,
        n_experts,
        top_k,
        selected_experts: selected,
        expert: e0,
        router_tensor: router_name,
        expert_tensor: format!("{expert_name}.blocks (MXFP4)"),
        expert_full_shape: vec![n_experts, out_features, in_features],
        tested_slice: vec![rows, cols],
        source_bytes_read: source_bytes,
        reference_output_checksum: refchk,
        base_compact_bits: base_bits,
        doctor_bits,
        metadata_bits: 32,
        complete_bpw,
        untreated_divergence: untreated,
        treated_divergence: treated_div,
        subbit_below_one_bpw: complete_bpw < 1.0,
        doctor_within_same_rate: complete_bpw < 1.0,
        repeatable,
    })
}
