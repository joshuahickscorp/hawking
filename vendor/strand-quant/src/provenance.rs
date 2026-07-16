use crate::decode::{decode_tensor_fixed_with_lut, eff_min_q, eff_scale_q, reconstruct_q};
use crate::encode::{n_sub_blocks, unpack_sub_scales, unpack_sub_scales_or_unity, EncodedTensor, SUB_BLOCK};
use crate::sha256::sha256;
use crate::trellis::{read_bits, TrellisConfig};

pub const DOMAIN_BLOCK: &[u8; 8] = b"SPV3.BLK";

pub const DOMAIN_TENSOR: &[u8; 8] = b"SPV3.TNS";

pub const DOMAIN_MODEL: &[u8; 8] = b"SPV3.MDL";

pub const DOMAIN_SELECT: &[u8; 8] = b"SPV3.SEL";

pub const DOMAIN_DESC: &[u8; 8] = b"SPV3.DSC";

pub const DOMAIN_OUTL: &[u8; 8] = b"SPV3.OUT";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProvenanceVector {
    pub block_index: u64,

    pub block_hash: [u8; 32],
}

pub fn block_hash(block_index: u64, q12: &[i32]) -> [u8; 32] {
    debug_assert!(q12.len() <= u32::MAX as usize, "block leaf n must fit u32");
    let mut msg = Vec::with_capacity(8 + 8 + 4 + q12.len() * 4);
    msg.extend_from_slice(DOMAIN_BLOCK);
    msg.extend_from_slice(&block_index.to_le_bytes());
    msg.extend_from_slice(&(q12.len() as u32).to_le_bytes());
    for &q in q12 {
        msg.extend_from_slice(&q.to_le_bytes());
    }
    sha256(&msg)
}

pub fn block_hashes(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<[u8; 32]> {
    let q12 = decode_tensor_fixed_with_lut(enc, cfg, lut);
    let mut hashes = Vec::with_capacity(enc.blocks.len());
    let mut off = 0usize;
    for (bi, blk) in enc.blocks.iter().enumerate() {
        let n = blk.n as usize;

        hashes.push(block_hash(bi as u64, &q12[off..off + n]));
        off += n;
    }
    hashes
}

pub fn tensor_root_from_hashes(hashes: &[[u8; 32]]) -> [u8; 32] {
    let mut msg = Vec::with_capacity(8 + 8 + hashes.len() * 32);
    msg.extend_from_slice(DOMAIN_TENSOR);
    msg.extend_from_slice(&(hashes.len() as u64).to_le_bytes());
    for h in hashes {
        msg.extend_from_slice(h);
    }
    sha256(&msg)
}

pub fn tensor_root(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> [u8; 32] {
    tensor_root_from_hashes(&block_hashes(enc, cfg, lut))
}

pub fn model_root_from_tensor_roots<'a, I>(roots: I) -> [u8; 32]
where
    I: IntoIterator<Item = (&'a str, [u8; 32])>,
{
    let mut body = Vec::new();
    let mut count: u64 = 0;
    for (name, root) in roots {
        debug_assert!(name.len() <= u32::MAX as usize, "tensor name must fit u32");
        body.extend_from_slice(&(name.len() as u32).to_le_bytes());
        body.extend_from_slice(name.as_bytes());
        body.extend_from_slice(&root);
        count += 1;
    }
    let mut msg = Vec::with_capacity(8 + 8 + body.len());
    msg.extend_from_slice(DOMAIN_MODEL);
    msg.extend_from_slice(&count.to_le_bytes());
    msg.extend_from_slice(&body);
    sha256(&msg)
}

pub fn model_root<'a, I>(tensors: I) -> [u8; 32]
where
    I: IntoIterator<Item = (&'a str, &'a EncodedTensor, &'a TrellisConfig, &'a [i32])>,
{
    model_root_from_tensor_roots(tensors.into_iter().map(|(name, enc, cfg, lut)| (name, tensor_root(enc, cfg, lut))))
}

pub fn outlier_digest(wire: &crate::outlier_wire::OutlierWire) -> [u8; 32] {
    let mut msg = Vec::with_capacity(8 + 8 + 12 + wire.entries.len() * 8);
    msg.extend_from_slice(DOMAIN_OUTL);
    msg.extend_from_slice(&(wire.entries.len() as u64).to_le_bytes());
    msg.extend_from_slice(&wire.omax_bits.to_le_bytes());
    msg.extend_from_slice(&wire.idx_bits.to_le_bytes());
    msg.extend_from_slice(&wire.val_bits.to_le_bytes());
    for &(i, c) in &wire.entries {
        msg.extend_from_slice(&i.to_le_bytes());
        msg.extend_from_slice(&c.to_le_bytes());
    }
    sha256(&msg)
}

#[allow(clippy::too_many_arguments)]
pub fn descriptor_digest(name: &str, shape: &[u64], rht_seed: u64, l_bits: u8, k_bits: u8, vec_dim: u8, flags: u8, block_len: u32, total: u64, outl: &[u8; 32]) -> [u8; 32] {
    let mut msg = Vec::with_capacity(8 + 4 + name.len() + 4 + shape.len() * 8 + 8 + 4 + 4 + 8 + 32);
    msg.extend_from_slice(DOMAIN_DESC);
    msg.extend_from_slice(&(name.len() as u32).to_le_bytes());
    msg.extend_from_slice(name.as_bytes());
    msg.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &d in shape {
        msg.extend_from_slice(&d.to_le_bytes());
    }
    msg.extend_from_slice(&rht_seed.to_le_bytes());
    msg.push(l_bits);
    msg.push(k_bits);
    msg.push(vec_dim);
    msg.push(flags);
    msg.extend_from_slice(&block_len.to_le_bytes());
    msg.extend_from_slice(&total.to_le_bytes());
    msg.extend_from_slice(outl);
    sha256(&msg)
}

pub fn block_bit_offset(enc: &EncodedTensor, cfg: &TrellisConfig, block_index: usize) -> usize {
    let k = cfg.k_bits as usize;
    enc.blocks[..block_index].iter().map(|b| cfg.num_steps(b.n as usize) * k).sum()
}

pub fn decode_block_q12(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32], block_index: usize) -> Vec<i32> {
    let k = cfg.k_bits;
    let mask = cfg.state_mask();
    let input_mask = cfg.num_inputs() - 1;
    let d = cfg.vec_dim();

    let blk = &enc.blocks[block_index];
    let n = blk.n as usize;
    let n_steps = cfg.num_steps(n);
    let bit_base = block_bit_offset(enc, cfg, block_index);

    let n_sub = n_sub_blocks(n);
    let mults = unpack_sub_scales_or_unity(&blk.sub_scales, n_sub);
    let eff: Vec<i32> = mults.iter().map(|&m| eff_scale_q(blk.scale_q, m)).collect();
    let offs: Vec<i32> = if enc.has_affine_min {
        let codes = unpack_sub_scales(&blk.mins, n_sub);
        codes.iter().map(|&c| eff_min_q(blk.min_base_q, c)).collect()
    } else {
        Vec::new()
    };

    let nk = n_steps * (k as usize);
    let start_state = if enc.tail_biting && nk >= cfg.l_bits as usize {
        let mut s = 0usize;
        let mut c = bit_base;
        for _ in 0..n_steps {
            let sym = read_bits(&enc.bits, c, k) & input_mask;
            c += k as usize;
            s = ((s << k) | sym) & mask;
        }
        s
    } else {
        blk.init_state as usize & mask
    };

    let mut out = Vec::with_capacity(n);
    let mut state = start_state;
    let mut bit_cursor = bit_base;
    let mut produced = 0usize;
    for _ in 0..n_steps {
        let sym = read_bits(&enc.bits, bit_cursor, k) & input_mask;
        bit_cursor += k as usize;
        state = ((state << k) | sym) & mask;

        let base = state * d;
        let emit = (n - produced).min(d);
        for j in 0..emit {
            let i = produced + j;
            let q = lut[base + j];
            let es = eff[i / SUB_BLOCK];
            let off = offs.get(i / SUB_BLOCK).copied().unwrap_or(0);
            out.push(reconstruct_q(es, q) + off);
        }
        produced += emit;
    }
    out
}

pub fn make_test_vectors(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32], k: usize) -> Vec<ProvenanceVector> {
    let n_blocks = enc.blocks.len();
    let k = k.min(n_blocks);
    if k == 0 {
        return Vec::new();
    }

    let mut seed_msg = Vec::with_capacity(8 + 8 + 8 + enc.bits.len());
    seed_msg.extend_from_slice(DOMAIN_SELECT);
    seed_msg.extend_from_slice(&(n_blocks as u64).to_le_bytes());
    seed_msg.extend_from_slice(&(enc.total as u64).to_le_bytes());
    seed_msg.extend_from_slice(&enc.bits);
    let seed = sha256(&seed_msg);

    let mut used = vec![false; n_blocks];
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    let mut ctr: u32 = 0;
    while chosen.len() < k {
        let mut m = [0u8; 36];
        m[..32].copy_from_slice(&seed);
        m[32..].copy_from_slice(&ctr.to_le_bytes());
        ctr += 1;
        let h = sha256(&m);
        let mut idx = (u64::from_le_bytes(h[0..8].try_into().expect("8-byte slice")) % n_blocks as u64) as usize;
        while used[idx] {
            idx = (idx + 1) % n_blocks;
        }
        used[idx] = true;
        chosen.push(idx);
    }
    chosen.sort_unstable();

    chosen.into_iter().map(|bi| ProvenanceVector { block_index: bi as u64, block_hash: block_hash(bi as u64, &decode_block_q12(enc, cfg, lut, bi)) }).collect()
}

pub fn verify_test_vectors(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32], vectors: &[ProvenanceVector]) -> bool {
    for v in vectors {
        let bi = v.block_index as usize;
        if v.block_index >= enc.blocks.len() as u64 {
            return false;
        }
        let q12 = decode_block_q12(enc, cfg, lut, bi);
        if block_hash(v.block_index, &q12) != v.block_hash {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::codebook_lut;
    use crate::decode::{decode_lean_with_lut, decode_tensor_fixed_with_lut};
    use crate::encode::{encode_tensor_with, vector_lut_from_scalar, EncodeOpts};

    fn hex(d: &[u8; 32]) -> String {
        let mut s = String::with_capacity(64);
        for b in d {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    fn test_weights(n: usize, seed: u64) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect()
    }

    fn opt_variants() -> [EncodeOpts; 4] {
        [
            EncodeOpts::default(),
            EncodeOpts { tail_biting: true, ..Default::default() },
            EncodeOpts { affine_min: true, ..Default::default() },
            EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
        ]
    }

    #[test]
    fn kat_canonical_serialization_pinned() {
        let k1 = block_hash(0, &[]);
        assert_eq!(hex(&k1), "2234d6fcd138505967fb0e109d23cc8452e9498895621e3b30bdb943fd72296f");
        let k2 = block_hash(1, &[1, -2, 1000]);
        assert_eq!(hex(&k2), "3990f085d2669dd68b30e9adcfe5a9e69ddcd7ccb2c5d703326aa5c1df7f30c8");
        let k3 = tensor_root_from_hashes(&[]);
        assert_eq!(hex(&k3), "c6f7a00b2d83315a5315962a057a525886a321b5b2c76200fb9694d6db476a53");
        let k4 = tensor_root_from_hashes(&[k1, k2]);
        assert_eq!(hex(&k4), "1492f395a50b7c64691f10646c707c6966e09ae4f017fb023ba37356dfb19edf");
        let k5 = model_root_from_tensor_roots(std::iter::empty());
        assert_eq!(hex(&k5), "b482c517817d47c0d95c6808b6ef48b1d86e14fa7c02b780969b467ea23026a7");
        let k6 = model_root_from_tensor_roots([("alpha", k3), ("beta", k4)]);
        assert_eq!(hex(&k6), "d390075de4f8486e6d3d6e975ec5bea8dc5e5858694817e4d635ec8545508666");
    }

    #[test]
    fn block_decode_is_bit_identical_to_reference() {
        let scalar_cfgs = [TrellisConfig::for_bpw(2.0), TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw(4.0), TrellisConfig::for_bpw_l(2.0, 5)];
        let lens = [1usize, 31, 256, 257, 777];
        for cfg in scalar_cfgs {
            let lut = codebook_lut(cfg.l_bits);
            for &n in &lens {
                for (vi, opts) in opt_variants().iter().enumerate() {
                    let w = test_weights(n, (n as u64) * 31 + vi as u64);
                    let enc = encode_tensor_with(&w, &cfg, opts);
                    let reference = decode_tensor_fixed_with_lut(&enc, &cfg, lut);
                    let lean = decode_lean_with_lut(&enc, &cfg, lut);
                    assert_eq!(reference, lean, "reference decoders disagree?!");
                    let mut cat: Vec<i32> = Vec::with_capacity(n);
                    for bi in 0..enc.blocks.len() {
                        cat.extend_from_slice(&decode_block_q12(&enc, &cfg, lut, bi));
                    }
                    assert_eq!(cat, reference, "per-block decode diverged: L={} k={} n={} variant={}", cfg.l_bits, cfg.k_bits, n, vi);
                }
            }
        }

        let cfg = TrellisConfig::for_bpw(4.0).with_vec_dim(2);
        let vlut = vector_lut_from_scalar(codebook_lut(cfg.l_bits), 2);
        for &n in &[1usize, 33, 257, 600] {
            for opts in &opt_variants() {
                let w = test_weights(n, 0xB1D2);
                let enc = crate::encode::encode_tensor_with_lut(&w, &cfg, opts, &vlut);
                let reference = decode_tensor_fixed_with_lut(&enc, &cfg, &vlut);
                let mut cat: Vec<i32> = Vec::with_capacity(n);
                for bi in 0..enc.blocks.len() {
                    cat.extend_from_slice(&decode_block_q12(&enc, &cfg, &vlut, bi));
                }
                assert_eq!(cat, reference, "d=2 per-block decode diverged at n={n}");
            }
        }
    }

    #[test]
    fn kat_descriptor_and_outlier_digests_pinned() {
        use crate::outlier_wire::OutlierWire;
        let wire = OutlierWire { omax_bits: 0x3F00_0000, entries: vec![(3, 5), (700, -127)], idx_bits: 10, val_bits: 8 };
        let od = outlier_digest(&wire);
        assert_eq!(hex(&od), "d890a1d65968ba070379f9fb0111a64517f1f9f0c9f9cce55449225ca7bb769a");
        let d_no = descriptor_digest("model.layers.0.q_proj", &[4, 256], 0x1234_5678_9ABC_DEF1, 7, 3, 1, 0b001, 256, 1024, &[0u8; 32]);
        assert_eq!(hex(&d_no), "4bf793e1276d4ac5b6a7ff569be975c6dc8a5f60e95d175420f4520a3cc98f69");
        let d_with = descriptor_digest("model.layers.0.q_proj", &[4, 256], 0x1234_5678_9ABC_DEF1, 7, 3, 1, 0b001, 256, 1024, &od);
        assert_eq!(hex(&d_with), "dca0d8cfa44cf419abef03f1f24c687a226256f989e4e59c62b8a52ffd7b03da");

        let d_seed = descriptor_digest("model.layers.0.q_proj", &[4, 256], 0x1234_5678_9ABC_DEF2, 7, 3, 1, 0b001, 256, 1024, &[0u8; 32]);
        assert_eq!(hex(&d_seed), "a00412529f552dc1083ff227c9b875b67a62fbd611c60cf539eaf7888b3f6066");
        assert_ne!(d_no, d_seed, "rht_seed must bind");

        let base = || ("t", vec![2u64, 8], 7u64, 5u8, 2u8, 1u8, 0u8, 256u32, 16u64);
        let (n, s, r, l, k, d, f, b, t) = base();
        let d0 = descriptor_digest(n, &s, r, l, k, d, f, b, t, &[0u8; 32]);
        assert_ne!(d0, descriptor_digest(n, &[8, 2], r, l, k, d, f, b, t, &[0u8; 32]), "shape binds");
        assert_ne!(d0, descriptor_digest(n, &s, r, l + 1, k, d, f, b, t, &[0u8; 32]), "l binds");
        assert_ne!(d0, descriptor_digest(n, &s, r, l, k + 1, d, f, b, t, &[0u8; 32]), "k binds");
        assert_ne!(d0, descriptor_digest(n, &s, r, l, k, d, 1, b, t, &[0u8; 32]), "flags bind");
        assert_ne!(d0, descriptor_digest(n, &s, r, l, k, d, f, 128, t, &[0u8; 32]), "block_len binds");
        assert_ne!(d0, descriptor_digest(n, &s, r, l, k, d, f, b, t + 1, &[0u8; 32]), "total binds");
        assert_ne!(d0, descriptor_digest(n, &s, r, l, k, d, f, b, t, &od), "outlier channel binds");
    }

    #[test]
    fn hashes_and_roots_are_deterministic() {
        let cfg = TrellisConfig::for_bpw(3.0);
        let lut = codebook_lut(cfg.l_bits);
        let enc_a = encode_tensor_with(&test_weights(700, 7), &cfg, &EncodeOpts::default());
        let enc_b = encode_tensor_with(&test_weights(300, 9), &cfg, &EncodeOpts::default());

        let h1 = block_hashes(&enc_a, &cfg, lut);
        let h2 = block_hashes(&enc_a, &cfg, lut);
        assert_eq!(h1, h2, "block hashes are not deterministic");
        assert_eq!(h1.len(), enc_a.blocks.len());

        let r1 = tensor_root(&enc_a, &cfg, lut);
        assert_eq!(r1, tensor_root(&enc_a, &cfg, lut));
        assert_eq!(r1, tensor_root_from_hashes(&h1), "root != composition of leaves");

        let model = [("model.layers.0.q_proj", &enc_a, &cfg, lut), ("model.layers.0.down_proj", &enc_b, &cfg, lut)];
        let m1 = model_root(model);
        let m2 = model_root(model);
        assert_eq!(m1, m2, "model root is not deterministic");
        let m3 = model_root_from_tensor_roots([("model.layers.0.q_proj", tensor_root(&enc_a, &cfg, lut)), ("model.layers.0.down_proj", tensor_root(&enc_b, &cfg, lut))]);
        assert_eq!(m1, m3, "model_root != composition of tensor roots");
    }

    #[test]
    fn payload_bit_flip_flips_exactly_one_leaf_and_all_roots() {
        let cfg = TrellisConfig::for_bpw(3.0);
        let lut = codebook_lut(cfg.l_bits);
        let other = encode_tensor_with(&test_weights(128, 3), &cfg, &EncodeOpts::default());
        let enc = encode_tensor_with(&test_weights(900, 1), &cfg, &EncodeOpts::default());
        assert!(enc.blocks.len() >= 2, "need a multi-block tensor for this test");

        let mut tampered = enc.clone();
        tampered.bits[0] ^= 1;

        let h_ok = block_hashes(&enc, &cfg, lut);
        let h_bad = block_hashes(&tampered, &cfg, lut);
        assert_ne!(h_ok[0], h_bad[0], "leaf 0 must change");
        assert_eq!(h_ok[1..], h_bad[1..], "blocks decode independently — only leaf 0 may change");

        assert_ne!(tensor_root(&enc, &cfg, lut), tensor_root(&tampered, &cfg, lut), "tensor root must change");
        let m_ok = model_root([("t", &enc, &cfg, lut), ("u", &other, &cfg, lut)]);
        let m_bad = model_root([("t", &tampered, &cfg, lut), ("u", &other, &cfg, lut)]);
        assert_ne!(m_ok, m_bad, "model root must change");
    }

    #[test]
    fn leaf_serialization_sensitivity() {
        assert_ne!(block_hash(0, &[5]), block_hash(1, &[5]), "block_index binds");
        assert_ne!(block_hash(0, &[5]), block_hash(0, &[-5]), "sign binds (i32 LE)");
        assert_ne!(block_hash(0, &[5]), block_hash(0, &[6]), "one Q12 LSB binds");
        assert_ne!(block_hash(0, &[1, 2]), block_hash(0, &[2, 1]), "order binds");
        assert_ne!(block_hash(0, &[1, 2]), block_hash(0, &[1]), "length binds");
    }

    #[test]
    fn model_root_is_name_and_order_sensitive() {
        let r1 = [0x11u8; 32];
        let r2 = [0x22u8; 32];
        let a = model_root_from_tensor_roots([("q_proj", r1), ("k_proj", r2)]);
        let b = model_root_from_tensor_roots([("q_proj2", r1), ("k_proj", r2)]);
        let c = model_root_from_tensor_roots([("k_proj", r2), ("q_proj", r1)]);
        assert_ne!(a, b, "tensor name binds");
        assert_ne!(a, c, "tensor order binds");
    }

    #[test]
    fn test_vectors_verify_and_catch_corruption() {
        let cfg = TrellisConfig::for_bpw(3.0);
        let lut = codebook_lut(cfg.l_bits);
        let enc = encode_tensor_with(&test_weights(1500, 5), &cfg, &EncodeOpts::default());
        let n_blocks = enc.blocks.len();
        assert!(n_blocks >= 4, "want >= 4 blocks for a meaningful selection");

        let vs = make_test_vectors(&enc, &cfg, lut, 4);
        assert_eq!(vs.len(), 4);

        for w in vs.windows(2) {
            assert!(w[0].block_index < w[1].block_index, "indices must ascend");
        }
        assert!(vs.iter().all(|v| v.block_index < n_blocks as u64));
        assert_eq!(vs, make_test_vectors(&enc, &cfg, lut, 4), "selection must be deterministic");

        assert!(verify_test_vectors(&enc, &cfg, lut, &vs), "genuine artifact must verify");
        assert!(verify_test_vectors(&enc, &cfg, lut, &[]), "empty vector set is vacuously true");

        let mut bad = vs.clone();
        bad[0].block_hash[0] ^= 0xFF;
        assert!(!verify_test_vectors(&enc, &cfg, lut, &bad));

        let target = vs[0].block_index as usize;
        let bit = block_bit_offset(&enc, &cfg, target);
        let mut tampered = enc.clone();
        tampered.bits[bit / 8] ^= 1 << (bit % 8);
        assert!(!verify_test_vectors(&tampered, &cfg, lut, &vs));

        let oob = [ProvenanceVector { block_index: n_blocks as u64, block_hash: [0u8; 32] }];
        assert!(!verify_test_vectors(&enc, &cfg, lut, &oob));

        assert!(make_test_vectors(&enc, &cfg, lut, 0).is_empty());
        let all = make_test_vectors(&enc, &cfg, lut, n_blocks + 100);
        assert_eq!(all.len(), n_blocks);
        for (i, v) in all.iter().enumerate() {
            assert_eq!(v.block_index, i as u64, "full selection must cover every block");
        }
        assert!(verify_test_vectors(&enc, &cfg, lut, &all));
    }

    #[test]
    fn empty_tensor_is_well_defined() {
        let cfg = TrellisConfig::for_bpw(3.0);
        let lut = codebook_lut(cfg.l_bits);
        let empty = EncodedTensor { bits: Vec::new(), blocks: Vec::new(), total: 0, has_rht_seed: false, tail_biting: false, has_affine_min: false };
        assert!(block_hashes(&empty, &cfg, lut).is_empty());
        assert_eq!(hex(&tensor_root(&empty, &cfg, lut)), "c6f7a00b2d83315a5315962a057a525886a321b5b2c76200fb9694d6db476a53");
        assert!(make_test_vectors(&empty, &cfg, lut, 8).is_empty());
        assert!(verify_test_vectors(&empty, &cfg, lut, &[]));
    }
}
