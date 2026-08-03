//! TEMPORARY measurement (not a shipping gate): runs the REAL c2_final coder over
//! the THREE real side-info streams of an on-disk q2 .strand archive and reports
//! authoritative recoverable bpw per stream + whole-model concat. Decides the C2F
//! scope: does c2_final's coder beat the deployed SDSQ (sideinfo_rans) on scale_q,
//! and how big is each channel on the real model.
//!
//! Run: STRAND_ARCHIVE=/abs/path.strand cargo test -p strand-quant \
//!        --test c2f_realarchive_measure -- --nocapture --ignored measure_real

use strand_quant::c2_final;
use strand_quant::encode::{n_sub_blocks, unpack_sub_scales};
use strand_quant::format::read_strand_v2;
use strand_quant::outlier_wire::read_outl_bytes;
use strand_quant::sideinfo_wire::read_sdsq;

#[test]
#[ignore = "needs a real archive via STRAND_ARCHIVE"]
fn measure_real() {
    let path = std::env::var("STRAND_ARCHIVE").expect("set STRAND_ARCHIVE=/abs/path.strand");
    let buf = std::fs::read(&path).expect("read archive");

    let tensors = read_strand_v2(&buf).expect("read_strand_v2");
    let total_params: usize = tensors.iter().map(|t| t.base.enc.total).sum();
    let total_blocks: usize = tensors.iter().map(|t| t.base.enc.blocks.len()).sum();

    // ---- scale_q (whole-model concat, tensor-then-block) ----
    // prefer SDSQ section if present (packed archives leave inline 0)
    let scale_q: Vec<i32> = match read_sdsq(&path).expect("read sdsq") {
        Some(s) => s.scale_q,
        None => tensors
            .iter()
            .flat_map(|t| t.base.enc.blocks.iter().map(|b| b.scale_q))
            .collect(),
    };
    assert_eq!(scale_q.len(), total_blocks);

    // ---- sub_scale (whole-model concat of unpacked 6-bit codes) ----
    let mut sub_codes: Vec<u8> = Vec::new();
    for t in &tensors {
        for blk in &t.base.enc.blocks {
            let ns = n_sub_blocks(blk.n as usize);
            sub_codes.extend(unpack_sub_scales(&blk.sub_scales, ns));
        }
    }

    // ---- outlier positions (per-tensor sorted; gap reset per tensor) ----
    // Each tensor's positions index into [0, tensor.total); gaps are per-tensor.
    let outl = read_outl_bytes(&buf, true).expect("read outl");
    let mut outl_pos_streams: Vec<Vec<u8>> = Vec::new();
    let mut n_outl_entries = 0usize;
    let mut outl_inline_idx_bits: u64 = 0;
    if let Some(o) = &outl {
        for w in &o.tensors {
            if let Some(w) = w {
                let positions: Vec<u32> = w.entries.iter().map(|&(i, _)| i).collect();
                n_outl_entries += positions.len();
                outl_inline_idx_bits += positions.len() as u64 * w.idx_bits as u64;
                outl_pos_streams.push(c2_final::encode_positions(&positions));
            }
        }
    }

    let p = total_params as f64;

    // scale_q via c2_final (whole-model single section)
    let sq_c2 = c2_final::encode_scale_q(&scale_q).len() as f64 * 8.0;
    let sq_inline = total_blocks as f64 * 32.0;
    // sub_scale via c2_final
    let ss_c2 = c2_final::encode_sub_scales(&sub_codes).len() as f64 * 8.0;
    let ss_inline = sub_codes.len() as f64 * 6.0;
    // outl positions via c2_final (sum of per-tensor sections)
    let op_c2: f64 = outl_pos_streams.iter().map(|s| s.len() as f64 * 8.0).sum();
    let op_inline = outl_inline_idx_bits as f64;

    println!("\n================ C2F REAL-ARCHIVE MEASURE ================");
    println!("archive = {path}");
    println!("params={total_params} blocks={total_blocks} outl_entries={n_outl_entries}");

    println!("\n-- scale_q (c2_final coder, whole-model) --");
    println!("  inline 32-bit bpw  = {:.5}", sq_inline / p);
    println!("  c2_final bpw       = {:.5}", sq_c2 / p);
    println!("  recoverable bpw    = {:.5}", (sq_inline - sq_c2) / p);

    println!("\n-- sub_scale (c2_final coder, whole-model) --");
    println!("  inline 6-bit bpw   = {:.5}", ss_inline / p);
    println!("  c2_final bpw       = {:.5}", ss_c2 / p);
    println!("  recoverable bpw    = {:.5}", (ss_inline - ss_c2) / p);

    println!("\n-- outl positions (c2_final coder, per-tensor) --");
    println!("  inline idx-bits bpw= {:.5}", op_inline / p);
    println!("  c2_final bpw       = {:.5}", op_c2 / p);
    println!("  recoverable bpw    = {:.5}", (op_inline - op_c2) / p);

    let total_rec = (sq_inline - sq_c2 + ss_inline - ss_c2 + op_inline - op_c2) / p;
    println!("\n  TOTAL recoverable (all 3) = {total_rec:.5} bpw");
}
