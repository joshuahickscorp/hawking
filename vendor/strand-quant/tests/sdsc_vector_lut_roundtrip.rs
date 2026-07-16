use std::sync::atomic::{AtomicU64, Ordering};

use strand_quant::decode::decode_tensor_fixed_with_lut;
use strand_quant::encode::{encode_tensor_with_lut, EncodeOpts};
use strand_quant::format::{read_strand_v2, read_strand_v2_header, write_strand_v2, write_strand_v2_rht, PackedTensor, PackedTensorV2};
use strand_quant::selfdesc::{append_sdsc_with_tensor_luts, build_sdsc_for_archive_with_tensor_luts, decode_q12_with_sdsc_at, read_sdsc_bytes, tensor_lut_wire_bytes, TensorLutInput};
use strand_quant::TrellisConfig;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn weights(n: usize, seed: u64) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect()
}

fn first_tensor_flag_offset(buf: &[u8]) -> usize {
    let mut p = 56usize;
    let name_len = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap()) as usize;
    p += 4 + name_len;
    let ndim = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap()) as usize;
    p += 4 + ndim * 8;
    p + 8 + 3 // seed, L, k, d, then the tensor flags byte
}

#[test]
fn learned_vector_luts_survive_hawking_wire_and_reconstruct_exact_q12() {
    let cfg = TrellisConfig::for_bpw_l(1.0, 5).with_vec_dim(2);
    let lut_a: Vec<i32> = (0..cfg.lut_len()).map(|i| (i as i32 - 31) * 211).collect();
    let lut_b: Vec<i32> = (0..cfg.lut_len()).map(|i| (31 - i as i32) * 173 + (i % 2) as i32 * 401).collect();
    let enc_a = encode_tensor_with_lut(&weights(1024, 101), &cfg, &EncodeOpts::default(), &lut_a);
    let enc_b = encode_tensor_with_lut(&weights(1024, 707), &cfg, &EncodeOpts::default(), &lut_b);
    let shape = [4u64, 256u64];
    let packed = [
        PackedTensorV2 {
            base: PackedTensor { name: "vector.same_geometry.a", shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc_a },
            block_len: cfg.block_len as u32,
        },
        PackedTensorV2 {
            base: PackedTensor { name: "vector.same_geometry.b", shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc_b },
            block_len: cfg.block_len as u32,
        },
    ];
    let base = write_strand_v2_rht(&packed, [0x5Au8; 32], true, false, &[true, false]).expect("write col-RHT STR2");
    let path = std::env::temp_dir().join(format!("strand-sdsc-vector-integration-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)));
    std::fs::write(&path, &base).unwrap();

    let inputs = [TensorLutInput { tensor_index: 0, entries: &lut_a }, TensorLutInput { tensor_index: 1, entries: &lut_b }];
    let err = build_sdsc_for_archive_with_tensor_luts(&base, &inputs[..1]).unwrap_err();
    assert!(err.contains("coverage mismatch"), "err was: {err}");
    let written = append_sdsc_with_tensor_luts(&path, &inputs).expect("append SDSC V2");
    assert_eq!(tensor_lut_wire_bytes(&written.tensor_luts[0]), 52 + lut_a.len() * 4);
    let wire = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(&wire[..base.len()], &base);

    let sdsc = read_sdsc_bytes(&wire, true).expect("parse SDSC").expect("SDSC present");
    assert_eq!(sdsc, written);
    let tensors = read_strand_v2(&wire).expect("parse STR2");
    let header = read_strand_v2_header(&wire).expect("parse STR2 header");
    assert!(header.tensors[0].rht_cols);
    assert!(!header.tensors[1].rht_cols);
    assert!(tensors[0].rht_cols, "OwnedTensorV2 lost col-RHT mode");
    assert!(!tensors[1].rht_cols);
    for (index, (tensor, lut)) in tensors.iter().zip([&lut_a, &lut_b]).enumerate() {
        let expected = decode_tensor_fixed_with_lut(&tensor.base.enc, &cfg, lut);
        let actual = decode_q12_with_sdsc_at(&sdsc, index, tensor).expect("SDSC Q12 decode");
        assert_eq!(actual, expected, "tensor {index} reconstructed differently");
    }

    let mut corrupt = wire.clone();
    let digest = written.tensor_luts[0].record_sha256;
    let digest_off = corrupt.windows(digest.len()).position(|w| w == digest).expect("record digest in wire");
    corrupt[digest_off + digest.len()] ^= 0x01;
    let err = read_sdsc_bytes(&corrupt, true).unwrap_err();
    assert!(err.contains("SHA-256 mismatch"), "err was: {err}");

    // The learned LUT digest binds the RHT axis as part of the tensor
    // descriptor.  Flipping col->row without regenerating SDSC must fail.
    let mut wrong_axis = wire.clone();
    let flag = first_tensor_flag_offset(&wrong_axis);
    assert_ne!(wrong_axis[flag] & 8, 0);
    wrong_axis[flag] ^= 8;
    let err = read_sdsc_bytes(&wrong_axis, true).unwrap_err();
    assert!(err.contains("SHA-256 mismatch"), "err was: {err}");
}

#[test]
fn vector_decoder_matches_native_for_d2_d3_d4_d8_ragged_tail_biting() {
    for d in [2u32, 3, 4, 8] {
        let cfg = TrellisConfig::for_bpw_l(1.0, 5).with_vec_dim(d);
        let lut: Vec<i32> = (0..cfg.lut_len())
            .map(|i| {
                let state = i / d as usize;
                let lane = i % d as usize;
                (state as i32 - 15) * 233 + lane as i32 * 97
            })
            .collect();
        let opts = EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() };
        let enc = encode_tensor_with_lut(&weights(517, 900 + d as u64), &cfg, &opts, &lut);
        let shape = [517u64];
        let packed = [PackedTensorV2 {
            base: PackedTensor { name: "vector.ragged.tail", shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
            block_len: cfg.block_len as u32,
        }];
        let base = write_strand_v2(&packed, [d as u8; 32], false).expect("write STR2");
        let path = std::env::temp_dir().join(format!("strand-sdsc-vector-d{d}-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)));
        std::fs::write(&path, &base).unwrap();
        append_sdsc_with_tensor_luts(&path, &[TensorLutInput { tensor_index: 0, entries: &lut }]).expect("append SDSC V2");
        let wire = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let sdsc = read_sdsc_bytes(&wire, true).unwrap().unwrap();
        let tensors = read_strand_v2(&wire).unwrap();
        let expected = decode_tensor_fixed_with_lut(&tensors[0].base.enc, &cfg, &lut);
        let actual = decode_q12_with_sdsc_at(&sdsc, 0, &tensors[0]).unwrap();
        assert_eq!(actual, expected, "SDSC vector decode diverged for d={d}");
    }
}

#[test]
fn interrupted_tensor_lut_append_is_prefix_recoverable_and_idempotent() {
    let cfg = TrellisConfig::for_bpw_l(1.0, 5).with_vec_dim(2);
    let lut: Vec<i32> = (0..cfg.lut_len()).map(|i| i as i32 * 31 - 700).collect();
    let enc = encode_tensor_with_lut(&weights(1024, 404), &cfg, &EncodeOpts::default(), &lut);
    let shape = [4u64, 256u64];
    let packed = [PackedTensorV2 {
        base: PackedTensor { name: "vector.append.recovery", shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
        block_len: cfg.block_len as u32,
    }];
    let base = write_strand_v2(&packed, [0xA5; 32], true).unwrap();
    let input = [TensorLutInput { tensor_index: 0, entries: &lut }];

    let full_path = std::env::temp_dir().join(format!("strand-sdsc-recovery-full-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)));
    std::fs::write(&full_path, &base).unwrap();
    let expected_sdsc = append_sdsc_with_tensor_luts(&full_path, &input).unwrap();
    let full = std::fs::read(&full_path).unwrap();
    // A retry after the final fsync but before acknowledgement is idempotent.
    assert_eq!(append_sdsc_with_tensor_luts(&full_path, &input).unwrap(), expected_sdsc);
    assert_eq!(std::fs::read(&full_path).unwrap(), full);
    std::fs::remove_file(&full_path).ok();

    let tail = &full[base.len()..];
    for cut in [1usize, tail.len() / 2, tail.len() - 1] {
        let path = std::env::temp_dir().join(format!("strand-sdsc-recovery-cut-{cut}-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)));
        let mut interrupted = base.clone();
        interrupted.extend_from_slice(&tail[..cut]);
        std::fs::write(&path, &interrupted).unwrap();
        assert_eq!(append_sdsc_with_tensor_luts(&path, &input).unwrap(), expected_sdsc, "cut={cut}");
        let recovered = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(recovered, full, "cut={cut}");
        assert!(read_sdsc_bytes(&recovered, true).unwrap().is_some());
    }

    // A nonmatching suffix is never truncated under the recovery policy.
    let foreign_path = std::env::temp_dir().join(format!("strand-sdsc-recovery-foreign-{}-{}.strand", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)));
    let mut foreign = base;
    foreign.extend_from_slice(b"not-an-sdsc-prefix");
    std::fs::write(&foreign_path, &foreign).unwrap();
    let err = append_sdsc_with_tensor_luts(&foreign_path, &input).unwrap_err();
    assert!(err.contains("refusing to truncate"), "err was: {err}");
    assert_eq!(std::fs::read(&foreign_path).unwrap(), foreign);
    std::fs::remove_file(&foreign_path).ok();
}
