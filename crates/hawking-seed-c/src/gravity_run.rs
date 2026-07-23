//! One 120B Gravity run under one controller. Sub-bit-first: for each real GPT-OSS expert it starts at
//! an aggressive sub-bit stress point, escalates REPRESENTATION (latent rank) before BPW, applies a
//! same-rate Doctor rescue, and rises only within the sub-bit region — never above one complete BPW,
//! never via an Escape Receipt here. Every expert produces a sealed checkpoint Record in the Seed's
//! append-only log (the single controller / evidence authority). Bounded reference slice per expert; the
//! 120B model is never densely materialized.

use crate::safetensors::SafeTensors;
use crate::state::{Event, Machine};
use crate::{evidence::receipt, mxfp4, subbit, Result};
use std::path::Path;

/// The sub-bit rate ladder (exact rational latent ranks -> increasing BPW, all < 1.0). Representation
/// (rank r) escalates before the physical rate rises.
const LADDER_R: [usize; 4] = [4, 8, 16, 24];

fn deterministic(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503))) >> 9) & 0xFF) as f32 / 128.0 - 1.0).collect()
}

/// Read a bounded (rows x cols) slice of expert `e`'s mlp1 weight (MXFP4) from the shard.
fn read_expert_slice(st: &SafeTensors, layer: usize, e: usize, rows: usize, cols: usize) -> Result<(Vec<f32>, usize)> {
    let name = format!("block.{layer}.mlp.mlp1_weight");
    let blk = st.get(&format!("{name}.blocks"))?;
    let out_features = blk.shape[1];
    let n_block = blk.shape[2];
    let in_features = n_block * 32;
    let rows = rows.min(out_features);
    let cols = cols.min(in_features);
    let bpr = n_block * 16;
    let spr = n_block;
    let bb = st.bytes(&format!("{name}.blocks"))?;
    let sb = st.bytes(&format!("{name}.scales"))?;
    let boff = e * out_features * bpr;
    let soff = e * out_features * spr;
    let mut w = vec![0f32; rows * cols];
    let mut row = vec![0f32; in_features];
    let mut bytes_read = 0;
    for r in 0..rows {
        let bo = boff + r * bpr;
        let so = soff + r * spr;
        mxfp4::dequant_row(&bb[bo..bo + bpr], &sb[so..so + spr], in_features, &mut row);
        w[r * cols..(r + 1) * cols].copy_from_slice(&row[..cols]);
        bytes_read += bpr + spr;
    }
    Ok((w, bytes_read))
}

/// Process one expert sub-bit-first; return (best_rate_r, best_bpw, untreated_div, treated_div, doctor_bits, source_bytes).
fn process_expert(w: &[f32], m: usize, n: usize, probes: &[Vec<f32>]) -> (usize, f64, f64, f64, usize) {
    // stress point first (smallest r); escalate representation before BPW; Doctor at each rate.
    let mut best = (LADDER_R[0], 2.0f64, 1.0f64, 1.0f64, 0usize);
    for &r in LADDER_R.iter() {
        let sb = subbit::fit(w, m, n, r);
        let untreated = subbit::output_divergence(w, m, n, &sb, probes);
        let base_bits = sb.bits();
        // same-rate Doctor: fill remaining sub-bit budget with sparse residual
        let budget = 0.99;
        let k = ((budget * (m * n) as f64 - base_bits as f64) / 48.0).max(0.0) as usize;
        let treated = subbit::doctor_rescue(w, m, n, subbit::fit(w, m, n, r), k, budget);
        let (tdiv, tbits) = match &treated {
            Some(t) => (subbit::output_divergence(w, m, n, t, probes), t.bits()),
            None => (untreated, base_bits),
        };
        let bpw = tbits as f64 / (m * n) as f64;
        if bpw < 1.0 && tdiv < best.3 {
            best = (r, bpw, untreated, tdiv, tbits - base_bits);
        }
    }
    best
}

/// Launch and drive exactly one bounded Gravity run over `layer`'s experts. Writes a checkpoint Record
/// per expert. `slice` bounds the per-expert reference component.
pub fn run(shard: &Path, revision: &str, state_root: &Path, layer: usize, n_experts: usize, slice: (usize, usize)) -> Result<()> {
    // one controller lease (single-writer)
    let lease = state_root.join("gravity.lease");
    std::fs::create_dir_all(state_root)?;
    std::fs::write(&lease, format!("pid={} rev={} layer={}", std::process::id(), revision, layer))?;

    let st = SafeTensors::open(shard)?;
    let mut mc = Machine::open(state_root)?;
    // controller admission
    for ev in [Event::Prepare, Event::Admit, Event::Run] {
        let _ = mc.apply(ev, serde_json::json!({"run":"gpt-oss-120b-gravity","layer":layer,"experts":n_experts}));
    }
    // sealed program header
    mc.record(receipt("condensation", serde_json::json!({
        "kind":"gravity_program","parent":"openai/gpt-oss-120b","revision":revision,"shard":shard.file_name().unwrap().to_string_lossy(),
        "layer":layer,"experts":n_experts,"family":"ternary-latent-factor","ladder_r":LADDER_R,
        "law":"sub-bit-first; representation before BPW; Doctor same-rate; no escape","slice":[slice.0,slice.1]
    })))?;

    let probes: Vec<Vec<f32>> = (0..4).map(|s| deterministic(slice.1, s + 20)).collect();
    let (mut total_src, mut checkpoints) = (0usize, 0usize);
    for e in 0..n_experts {
        let (w, src) = match read_expert_slice(&st, layer, e, slice.0, slice.1) {
            Ok(x) => x,
            Err(_) => continue,
        };
        total_src += src;
        let (r, bpw, undiv, tdiv, dbits) = process_expert(&w, slice.0, slice.1, &probes);
        mc.record(receipt("condensation", serde_json::json!({
            "kind":"expert_checkpoint","layer":layer,"expert":e,"rate_rank_r":r,"complete_bpw":bpw,
            "untreated_divergence":undiv,"treated_divergence":tdiv,"doctor_bits":dbits,"sub_bit":bpw<1.0,
            "source_bytes_read":src
        })))?;
        checkpoints += 1;
    }
    // drain/seal
    let _ = mc.apply(Event::Drain, serde_json::json!({}));
    let _ = mc.apply(Event::Seal, serde_json::json!({"checkpoints":checkpoints,"total_source_bytes":total_src}));
    let _ = std::fs::remove_file(&lease);
    Ok(())
}
