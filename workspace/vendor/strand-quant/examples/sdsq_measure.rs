// TEMPORARY measurement harness (not committed). Reads a packed/legacy q2 .strand
// archive and reports:
//   1. model-wide scale_q rANS cost vs inline 32-bit (the SDSQ win)
//   2. model-wide sub_scale rANS cost vs inline 6-bit (the ~2.43 subset)
//   3. H per symbol + (achieved - H) overhead  -> tests the <= H+0.05 claim
//   4. full-file bpw before (16-byte records) and projected after (12-byte)
//      accounting for per-tensor page padding exactly.
//
// Usage: cargo run --release --example sdsq_measure -- <archive.strand>

use std::collections::HashMap;

use strand_quant::encode::{n_sub_blocks, unpack_sub_scales, SUB_BLOCK};
use strand_quant::format::{read_strand_v2, read_strand_v2_header, BlockOffsetRecord};
use strand_quant::sideinfo_rans::encode_scale_q;
use strand_quant::sideinfo_wire::read_sdsq;

const PAGE: usize = 4096;

fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}

fn order0_entropy_bits(syms: &[i64]) -> f64 {
    let mut counts: HashMap<i64, u64> = HashMap::new();
    for &s in syms {
        *counts.entry(s).or_insert(0) += 1;
    }
    let n = syms.len() as f64;
    let mut h = 0.0f64;
    for &c in counts.values() {
        let p = c as f64 / n;
        h -= p * p.log2();
    }
    h.max(0.0)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: sdsq_measure <archive.strand>");
    let buf = std::fs::read(&path).expect("read archive");
    let flen = buf.len();

    let hdr = read_strand_v2_header(&buf).expect("header");
    let flags = hdr.flags;
    let n_tensors = hdr.tensors.len();
    let total_params: usize = hdr.tensors.iter().map(|t| t.total).sum();
    let total_blocks: usize = hdr.tensors.iter().map(|t| t.n_blocks).sum();
    eprintln!(
        "archive={path}\n  file_bytes={flen} flags={flags} tensors={n_tensors} \
         params={total_params} blocks={total_blocks}"
    );

    // --- scale_q: prefer SDSQ section; else inline from the seek table ---
    let scale_q: Vec<i32> = match read_sdsq(&path).expect("read sdsq") {
        Some(s) => {
            eprintln!("  scale_q source = SDSQ section ({} values)", s.scale_q.len());
            s.scale_q
        }
        None => {
            eprintln!("  scale_q source = inline seek table");
            hdr.tensors.iter().flat_map(|t| t.table.iter().map(|r| r.scale_q)).collect()
        }
    };
    assert_eq!(scale_q.len(), total_blocks);

    // --- sub_scale: full read, unpack the 6-bit codes model-wide ---
    // (read_strand_v2 handles 12- vs 16-byte stride; scale_q there may be 0 for a
    //  packed archive but we don't use it — we use the SDSQ values above.)
    let tensors = read_strand_v2(&buf).expect("full read");
    let mut sub_codes: Vec<i64> = Vec::new();
    for t in &tensors {
        for blk in &t.base.enc.blocks {
            let ns = n_sub_blocks(blk.n as usize);
            let codes = unpack_sub_scales(&blk.sub_scales, ns);
            for c in codes {
                sub_codes.push(c as i64);
            }
        }
    }
    eprintln!("  sub_scale codes = {} (SUB_BLOCK={SUB_BLOCK})", sub_codes.len());

    // --- scale_q measurement ---
    let scale_raw: Vec<i64> = scale_q.iter().map(|&v| v as i64).collect();
    let sq_stream = encode_scale_q(&scale_q).len();
    let sq_bits = sq_stream as f64 * 8.0;
    let sq_bps = sq_bits / scale_q.len() as f64;
    let sq_h = order0_entropy_bits(&scale_raw);
    let sq_inline_bpw = total_blocks as f64 * 32.0 / total_params as f64;
    let sq_rans_bpw = sq_bits / total_params as f64;

    // --- sub_scale measurement ---
    let ss_stream = strand_quant::sideinfo_rans::encode_stream(&sub_codes).len();
    let ss_bits = ss_stream as f64 * 8.0;
    let ss_bps = ss_bits / sub_codes.len() as f64;
    let ss_h = order0_entropy_bits(&sub_codes);
    let ss_inline_bpw = sub_codes.len() as f64 * 6.0 / total_params as f64;
    let ss_rans_bpw = ss_bits / total_params as f64;

    println!("\n================ SDSQ MEASUREMENT (model-wide concat) ================");
    println!("params={total_params} blocks={total_blocks}");
    println!("\n-- scale_q --");
    println!("  H (order-0)           = {sq_h:.4} bits/sym");
    println!("  rANS achieved         = {sq_bps:.4} bits/sym  (overhead = H+{:.4})", sq_bps - sq_h);
    println!("  <= H+0.05 ?           = {}", if sq_bps <= sq_h + 0.05 { "YES" } else { "NO" });
    println!("  inline 32-bit bpw     = {sq_inline_bpw:.4}");
    println!("  rANS stream bpw       = {sq_rans_bpw:.4}");
    println!("  recoverable bpw       = {:.4}", sq_inline_bpw - sq_rans_bpw);

    println!("\n-- sub_scale --");
    println!("  H (order-0)           = {ss_h:.4} bits/sym");
    println!("  rANS achieved         = {ss_bps:.4} bits/sym  (overhead = H+{:.4})", ss_bps - ss_h);
    println!("  inline 6-bit bpw      = {ss_inline_bpw:.4}");
    println!("  rANS stream bpw       = {ss_rans_bpw:.4}");
    println!("  recoverable bpw       = {:.4}", ss_inline_bpw - ss_rans_bpw);

    // --- full-file bpw before/after the seek-table shrink ---
    // Recompute the table-section bytes (page-padded per tensor) at 16 vs 12 B.
    let mut tbl16 = 0usize;
    let mut tbl12 = 0usize;
    for t in &hdr.tensors {
        tbl16 += align_up(t.n_blocks * BlockOffsetRecord::SIZE, PAGE);
        tbl12 += align_up(t.n_blocks * BlockOffsetRecord::SIZE_PACKED, PAGE);
    }
    let table_saving = tbl16.saturating_sub(tbl12);
    // The current file already carries the SDSQ section (whether dead or live).
    let before_bytes = if flags & 2 != 0 {
        // packed file: "before" (legacy) would be this file + the table saving.
        flen + table_saving
    } else {
        flen
    };
    let after_bytes = if flags & 2 != 0 { flen } else { flen.saturating_sub(table_saving) };
    let before_bpw = before_bytes as f64 * 8.0 / total_params as f64;
    let after_bpw = after_bytes as f64 * 8.0 / total_params as f64;

    println!("\n-- full-file bpw (file_bytes*8/params) --");
    println!("  table bytes @16B (page-padded) = {tbl16}");
    println!("  table bytes @12B (page-padded) = {tbl12}");
    println!("  seek-table saving              = {table_saving} bytes");
    println!("  BEFORE (16-byte records)       = {before_bytes} bytes -> {before_bpw:.4} bpw");
    println!("  AFTER  (12-byte records)       = {after_bytes} bytes -> {after_bpw:.4} bpw");
    println!("  full-file delta                = {:.4} bpw", before_bpw - after_bpw);
}
