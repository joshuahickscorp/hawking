//! Real-archive byte-identity gate for the SDSQ scale_q packing.
//!
//! The synthetic unit gate (`loader::tests::sdsq_packed_reconstruct_is_byte_identical_to_legacy`)
//! proves the invariant on a 2-tensor toy. This proves it on the REAL Qwen2.5-0.5B
//! q2 model: a 12-byte-record + live-SDSQ archive must decode (Q12 integer
//! reconstruct) byte-identically to a 16-byte-record legacy archive of the EXACT
//! same encode. maxabsdiff over every tensor MUST be 0 — recon is container-only.
//!
//! Build both archives with the same encode flags, differing only in
//! `--sdsq-sideinfo`:
//!   q2_legacy.strand : --bits 2 --l 12 --outlier-channel 1
//!   q2_packed.strand : ... --sdsq-sideinfo
//!
//! Run: STRAND_LEGACY=/abs/q2_legacy.strand STRAND_PACKED=/abs/q2_packed.strand \
//!        cargo test -p strand-decode-kernel --test sdsq_realarchive_gate \
//!        -- --ignored --nocapture

use strand_decode_kernel::loader::StrandModel;
use strand_quant::decode::decode_tensor_fixed;

#[test]
#[ignore = "needs real archives via STRAND_LEGACY / STRAND_PACKED"]
fn sdsq_real_archive_reconstruct_is_byte_identical() {
    let legacy_path = std::env::var("STRAND_LEGACY").expect("set STRAND_LEGACY");
    let packed_path = std::env::var("STRAND_PACKED").expect("set STRAND_PACKED");

    let legacy = StrandModel::open(std::path::Path::new(&legacy_path)).expect("open legacy");
    let packed = StrandModel::open(std::path::Path::new(&packed_path)).expect("open packed");

    // The packed archive must actually be packed (flag set); legacy must not be.
    use strand_quant::format::flags_v2;
    assert_eq!(legacy.header().flags & flags_v2::SCALEQ_IN_SDSQ, 0, "legacy archive must NOT set SCALEQ_IN_SDSQ");
    assert_ne!(packed.header().flags & flags_v2::SCALEQ_IN_SDSQ, 0, "packed archive MUST set SCALEQ_IN_SDSQ");

    let names: Vec<String> = legacy.tensor_names().map(|s| s.to_string()).collect();
    let pnames: Vec<String> = packed.tensor_names().map(|s| s.to_string()).collect();
    assert_eq!(names, pnames, "tensor name lists must match");

    let mut max_abs_diff: i64 = 0;
    let mut checked_blocks = 0usize;
    for name in &names {
        let lh = legacy.tensor_header(name).unwrap();
        let ph = packed.tensor_header(name).unwrap();

        // scale_q sourced from SDSQ (packed) must equal inline scale_q (legacy).
        let lq: Vec<i32> = lh.table.iter().map(|r| r.scale_q).collect();
        let pq: Vec<i32> = ph.table.iter().map(|r| r.scale_q).collect();
        assert_eq!(lq, pq, "SDSQ-sourced scale_q != inline scale_q for {name}");
        checked_blocks += lq.len();

        let el = legacy.encoded_tensor(name).expect("legacy enc");
        let ep = packed.encoded_tensor(name).expect("packed enc");
        assert_eq!(el.blocks, ep.blocks, "EncodedTensor.blocks differ for {name}");

        let ql = decode_tensor_fixed(&el, &legacy.config_for(lh));
        let qp = decode_tensor_fixed(&ep, &packed.config_for(ph));
        assert_eq!(ql.len(), qp.len(), "decode length differs for {name}");
        for (a, b) in ql.iter().zip(qp.iter()) {
            let d = (*a as i64 - *b as i64).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }

        // OUTL outlier channel: the reconstructed OutlierWire (entries =
        // (position, code) pairs, idx_bits, val_bits, omax) MUST be byte-identical
        // whether positions are inline-idx (legacy) or C2F gap-coded (packed). This
        // is the C2F outl_pos lever's make-or-break invariant.
        let lo = legacy.outlier(name);
        let po = packed.outlier(name);
        assert_eq!(lo, po, "OUTL outlier channel differs for {name}");
    }
    println!("SDSQ real-archive gate: {} tensors, {checked_blocks} blocks, maxabsdiff={max_abs_diff}", names.len());
    assert_eq!(max_abs_diff, 0, "recon must be byte-identical (container-only)");
}
