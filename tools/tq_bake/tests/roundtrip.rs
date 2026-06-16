//! End-to-end round-trip tests for the TQ baker pipeline.
//!
//! The baker itself is a thin `main()` that wires GGUF dequant -> RHT -> encode ->
//! write -> read-back. Driving it against a real GGUF is heavy (one attn_k tensor is
//! ~70 s of Viterbi), so these tests mirror the baker's encode -> write -> read-back
//! -> decode logic on a small synthetic f32 tensor and assert the two invariants the
//! baker relies on:
//!
//!   1. the pipeline is deterministic (same input -> byte-identical archive + decode), and
//!   2. the decode after a write/read round-trip is bit-identical to decoding the
//!      freshly-encoded tensor (the archive doesn't perturb the payload).
//!
//! It also exercises the two writer paths the baker chooses between — plain
//! `write_strand_v2` (rht=none) and `write_strand_v2_rht` (rht=cols) — and checks the
//! strict / ragged deployability guard, so the wire-flag plumbing the baker depends on
//! is covered without a fixture model.
//!
//! The CPU encoder is forced (`STRAND_NO_GPU`) so the test is deterministic and fast on
//! any host, Metal or not.

use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor, EncodedTensor};
use strand_quant::format::{
    flags_v2, read_strand_v2_header, write_strand_v2, write_strand_v2_rht, PackedTensor,
    PackedTensorV2,
};
use strand_quant::gate_utils::rht_seed_for;
use strand_quant::rht::{rht_forward_cols_inplace, RhtConfig};
use strand_quant::sideinfo_wire::read_strand_v2_applied;
use strand_quant::TrellisConfig;

/// Deterministic synthetic weights with a little structure (so the trellis has work to
/// do) — mirrors the kind of data `dequant_to_f32` would hand the baker.
fn synth_weights(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32 + seed as f32) * 0.0137;
            x.sin() * 0.5 + ((i % 11) as f32) * 0.01
        })
        .collect()
}

/// Drive one tensor through the exact pipeline the baker uses for `--rht none`:
/// encode -> write_strand_v2(strict) -> read_strand_v2_applied -> decode tensor 0.
fn bake_none(
    name: &str,
    weights: &[f32],
    shape: [u64; 2],
    cfg: &TrellisConfig,
    strict: bool,
) -> (Vec<u8>, Vec<i32>, EncodedTensor) {
    let enc = encode_tensor(weights, cfg);
    let pt = PackedTensorV2 {
        base: PackedTensor {
            name,
            shape: &shape,
            rht_seed: 0,
            l_bits: cfg.l_bits as u8,
            k_bits: cfg.k_bits as u8,
            vec_dim: cfg.vec_dim() as u8,
            enc: &enc,
        },
        block_len: cfg.block_len as u32,
    };
    let file = write_strand_v2(&[pt], [0u8; 32], strict).expect("write_strand_v2");
    let back = read_strand_v2_applied(&file).expect("read_strand_v2_applied");
    assert_eq!(back.len(), 1);
    let decoded = decode_tensor_fixed(&back[0].base.enc, cfg);
    (file, decoded, enc)
}

#[test]
fn roundtrip_none_is_deterministic_and_bit_identical() {
    std::env::set_var("STRAND_NO_GPU", "1");
    // in_features=256 == block_len -> deployable / strict.
    let cfg = TrellisConfig::for_bpw(3.0);
    let in_f = 256u64;
    let out_f = 4u64;
    let w = synth_weights((in_f * out_f) as usize, 7);

    let (file1, dec1, enc1) = bake_none("blk.0.attn_k.weight", &w, [out_f, in_f], &cfg, true);
    let (file2, dec2, enc2) = bake_none("blk.0.attn_k.weight", &w, [out_f, in_f], &cfg, true);

    // 1. determinism: identical inputs -> byte-identical archive + identical decode.
    assert_eq!(file1, file2, "archive bytes not deterministic");
    assert_eq!(dec1, dec2, "decode not deterministic");

    // 2. write/read round-trip preserves the payload: decoding the read-back tensor
    //    equals decoding the freshly-encoded tensor.
    let dec_fresh = decode_tensor_fixed(&enc1, &cfg);
    assert_eq!(
        dec1, dec_fresh,
        "round-trip decode diverged from fresh decode"
    );
    assert_eq!(enc1.bits, enc2.bits, "encoded payload not deterministic");

    // archive is a STR2 archive and (256-aligned) strict.
    assert_eq!(&file1[0..4], b"STR2", "bad magic");
    let hdr = read_strand_v2_header(&file1).unwrap();
    assert_ne!(
        hdr.flags & flags_v2::ALL_STRICT,
        0,
        "expected ALL_STRICT for aligned tensor"
    );
    assert!(
        !hdr.tensors[0].rht_cols,
        "rht=none must not set col-RHT flag"
    );
    assert!(
        !hdr.tensors[0].has_rht_seed,
        "rht=none must not set has_rht_seed"
    );
    assert_eq!(
        decode_tensor_fixed(&enc1, &cfg).len(),
        (in_f * out_f) as usize
    );
}

#[test]
fn roundtrip_cols_sets_flags_seed_and_is_bit_identical() {
    std::env::set_var("STRAND_NO_GPU", "1");
    let cfg = TrellisConfig::for_bpw(3.0);
    let in_f = 256usize;
    let out_f = 4usize;
    let name = "blk.0.attn_k.weight";

    // Mirror the baker's --rht cols path: rotate the weights in place, stamp the seed
    // presence on the encoded tensor, then write via the col-RHT writer with a true mask.
    let mut w = synth_weights(in_f * out_f, 9);
    let seed = rht_seed_for(name);
    rht_forward_cols_inplace(&mut w, &RhtConfig::from_seed(seed), in_f);

    let mut enc = encode_tensor(&w, &cfg);
    enc.has_rht_seed = true;
    let shape = [out_f as u64, in_f as u64];
    let pt = PackedTensorV2 {
        base: PackedTensor {
            name,
            shape: &shape,
            rht_seed: seed,
            l_bits: cfg.l_bits as u8,
            k_bits: cfg.k_bits as u8,
            vec_dim: cfg.vec_dim() as u8,
            enc: &enc,
        },
        block_len: cfg.block_len as u32,
    };
    let file =
        write_strand_v2_rht(&[pt], [0u8; 32], true, false, &[true]).expect("write_strand_v2_rht");
    assert_eq!(&file[0..4], b"STR2");

    // Header carries both the per-tensor col-RHT flag (bit 3) and has_rht_seed (bit 1)
    // plus the deterministic name-derived seed.
    let hdr = read_strand_v2_header(&file).expect("read header");
    assert!(hdr.tensors[0].rht_cols, "col-RHT flag (bit 3) not set");
    assert!(hdr.tensors[0].has_rht_seed, "has_rht_seed (bit 1) not set");
    assert_eq!(hdr.tensors[0].rht_seed, seed, "rht_seed not persisted");
    assert_ne!(seed, 0, "rht_seed_for must be non-zero");

    // Full read-back decodes bit-identically to the fresh encode of the rotated weights.
    let back = read_strand_v2_applied(&file).expect("read applied");
    assert_eq!(back[0].base.rht_seed, seed);
    let dec_back = decode_tensor_fixed(&back[0].base.enc, &cfg);
    let dec_fresh = decode_tensor_fixed(&enc, &cfg);
    assert_eq!(dec_back, dec_fresh, "col-RHT round-trip decode diverged");

    // The col-RHT bake is itself deterministic.
    let mut w2 = synth_weights(in_f * out_f, 9);
    rht_forward_cols_inplace(&mut w2, &RhtConfig::from_seed(seed), in_f);
    let enc2 = encode_tensor(&w2, &cfg);
    assert_eq!(enc.bits, enc2.bits, "col-RHT encode not deterministic");
}

#[test]
fn ragged_in_features_forces_non_strict() {
    std::env::set_var("STRAND_NO_GPU", "1");
    let cfg = TrellisConfig::for_bpw(3.0);
    // in_features=100 is NOT a multiple of block_len=256 -> ragged -> must be non-strict.
    let in_f = 100u64;
    let out_f = 2u64;
    let w = synth_weights((in_f * out_f) as usize, 3);
    assert_ne!(
        in_f % cfg.block_len as u64,
        0,
        "test precondition: in_features must be ragged"
    );

    // Writing strict over a ragged tensor must be rejected by the writer (this is the
    // invariant the baker's `strict = ragged.is_empty()` guard avoids tripping).
    let (file, _dec, _enc) = bake_none("ragged.weight", &w, [out_f, in_f], &cfg, false);
    let hdr = read_strand_v2_header(&file).unwrap();
    assert_eq!(
        hdr.flags & flags_v2::ALL_STRICT,
        0,
        "ragged archive must have ALL_STRICT cleared"
    );

    // And the strict writer hard-errors on the same ragged tensor.
    let enc = encode_tensor(&w, &cfg);
    let shape = [out_f, in_f];
    let pt = PackedTensorV2 {
        base: PackedTensor {
            name: "ragged.weight",
            shape: &shape,
            rht_seed: 0,
            l_bits: cfg.l_bits as u8,
            k_bits: cfg.k_bits as u8,
            vec_dim: cfg.vec_dim() as u8,
            enc: &enc,
        },
        block_len: cfg.block_len as u32,
    };
    let err = write_strand_v2(&[pt], [0u8; 32], true).unwrap_err();
    assert!(
        err.contains("ragged.weight"),
        "strict writer err was: {err}"
    );
}
