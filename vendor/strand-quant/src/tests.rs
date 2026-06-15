
use crate::codebook::{codebook_lut, QUANTILE_SHIFT};
use crate::decode::{decode_tensor, decode_tensor_fixed, eff_scale_q, reconstruct_q, SCALE_SHIFT};
use crate::encode::{encode_tensor, n_sub_blocks, unpack_sub_scales, SUB_BLOCK};
use crate::trellis::{read_bits, TrellisConfig};

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
    }
    fn next_u64(&mut self) -> u64 {
        
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn gauss(rng: &mut Lcg) -> f64 {
    let u1 = rng.unit().max(1e-12);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn normal_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| gauss(&mut rng) as f32).collect()
}

fn mse(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = *x as f64 - *y as f64;
        acc += d * d;
    }
    acc / a.len().max(1) as f64
}

fn uniform_quantize_mse(weights: &[f32], b: u32) -> f64 {
    let levels = 1usize << b;
    let absmax = weights.iter().fold(0.0f64, |m, &w| m.max(w.abs() as f64));
    if absmax == 0.0 || levels < 2 {
        return 0.0;
    }
    const MULTS: [f64; 11] = [
        0.55, 0.65, 0.75, 0.85, 0.92, 1.0, 1.08, 1.18, 1.30, 1.45, 1.65,
    ];
    let mut best = f64::INFINITY;
    for &m in &MULTS {
        let clip = absmax * m;
        
        let step = (2.0 * clip) / (levels as f64 - 1.0);
        let mut acc = 0.0f64;
        for &w in weights {
            let x = (w as f64).clamp(-clip, clip);
            let idx = ((x + clip) / step).round();
            let recon = idx * step - clip;
            let d = w as f64 - recon;
            acc += d * d;
        }
        let m = acc / weights.len() as f64;
        if m < best {
            best = m;
        }
    }
    best
}

#[test]
fn determinism_decode_is_bit_identical() {
    let weights = normal_vec(2048, 0xDE7E_4111u64);
    let cfg = TrellisConfig::for_bpw(4.0);
    let enc = encode_tensor(&weights, &cfg);

    let a = decode_tensor_fixed(&enc, &cfg);
    let b = decode_tensor_fixed(&enc, &cfg);
    assert_eq!(a, b, "fixed-point decode is not deterministic");

    let fa = decode_tensor(&enc, &cfg);
    let fb = decode_tensor(&enc, &cfg);
    let pa: Vec<u32> = fa.iter().map(|x| x.to_bits()).collect();
    let pb: Vec<u32> = fb.iter().map(|x| x.to_bits()).collect();
    assert_eq!(pa, pb, "f32 decode is not bit-identical");

    for (&q, &f) in a.iter().zip(&fa) {
        assert_eq!(f.to_bits(), ((q as f32) * (1.0 / 4096.0)).to_bits());
    }
}

#[test]
fn determinism_reencode_same_bits() {
    
    let weights = normal_vec(1024, 12345);
    let cfg = TrellisConfig::for_bpw(3.0);
    let e1 = encode_tensor(&weights, &cfg);
    let e2 = encode_tensor(&weights, &cfg);
    assert_eq!(e1, e2, "re-encoding a fixed input changed the bits");
}

#[test]
fn encode_decode_path_consistency() {
    let weights = normal_vec(777, 999);
    let cfg = TrellisConfig::for_bpw(4.0);
    let enc = encode_tensor(&weights, &cfg);
    let lut = codebook_lut(cfg.l_bits);

    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let mut expected = Vec::with_capacity(enc.total);
    let mut cursor = 0usize;
    for blk in &enc.blocks {
        let n_sub = n_sub_blocks(blk.n as usize);
        let mults = unpack_sub_scales(&blk.sub_scales, n_sub);
        let mut state = blk.init_state as usize & mask;
        for i in 0..blk.n as usize {
            let sym = read_bits(&enc.bits, cursor, k) & input_mask;
            cursor += k as usize;
            state = ((state << k) | sym) & mask;
            let es = eff_scale_q(blk.scale_q, mults[i / SUB_BLOCK]);
            expected.push(reconstruct_q(es, lut[state]));
        }
    }

    let got = decode_tensor_fixed(&enc, &cfg);
    assert_eq!(got, expected, "decoder diverged from the encoder's implied path");
    assert_eq!(got.len(), weights.len());

    let recon = decode_tensor(&enc, &cfg);
    let m = mse(&weights, &recon);
    let signal = weights.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / weights.len() as f64;
    assert!(m < signal * 0.5, "reconstruction MSE {m} not below half signal energy {signal}");
}

#[test]
fn short_final_block_decodes_to_exact_length() {
    
    let weights = normal_vec(600, 7);
    let cfg = TrellisConfig::for_bpw(3.0);
    assert_eq!(cfg.block_len, 256);
    let enc = encode_tensor(&weights, &cfg);
    assert_eq!(enc.blocks.len(), 3);
    assert_eq!(enc.blocks[2].n, 600 - 512);
    let recon = decode_tensor(&enc, &cfg);
    assert_eq!(recon.len(), 600);
}

fn mechanism_at_rate(b: u32, n: usize, seed: u64) -> (f64, f64) {
    let weights = normal_vec(n, seed);
    let cfg = TrellisConfig::for_bpw(b as f64);
    assert_eq!(cfg.k_bits, b, "config did not realise the requested rate");

    let enc = crate::encode::encode_tensor_opts(&weights, &cfg, false);
    let recon = decode_tensor(&enc, &cfg);
    let trellis_mse = mse(&weights, &recon);
    let uniform_mse = uniform_quantize_mse(&weights, b);
    (uniform_mse, trellis_mse)
}

#[test]
fn mechanism_trellis_beats_uniform_at_iso_bits() {
    
    const SEEDS: [u64; 4] = [0xA1, 0xB2, 0xC3, 0xD4];
    for &b in &[3u32, 4u32] {
        let mut uni_sum = 0.0f64;
        let mut tre_sum = 0.0f64;
        for &seed in &SEEDS {
            let (uni, tre) = mechanism_at_rate(b, 8192, seed ^ (b as u64) << 8);
            uni_sum += uni;
            tre_sum += tre;
            
            assert!(
                tre < uni,
                "trellis MSE {tre:.6} not below uniform MSE {uni:.6} at {b} bpw (seed {seed:#x})",
            );
        }
        let uni = uni_sum / SEEDS.len() as f64;
        let tre = tre_sum / SEEDS.len() as f64;
        let advantage_pct = (uni - tre) / uni * 100.0;
        
        let bit_savings = 0.5 * (uni / tre).log2();
        eprintln!(
            "rate={b} bpw  uniform_mse={uni:.6}  trellis_mse={tre:.6}  \
             advantage={advantage_pct:.1}%  equiv_bit_savings={bit_savings:.3}  (mean of {} seeds)",
            SEEDS.len(),
        );
    }
}

#[test]
fn empty_and_tiny_inputs() {
    let cfg = TrellisConfig::for_bpw(4.0);
    assert!(decode_tensor(&encode_tensor(&[], &cfg), &cfg).is_empty());
    let one = vec![0.37f32];
    let r = decode_tensor(&encode_tensor(&one, &cfg), &cfg);
    assert_eq!(r.len(), 1);
}

#[test]
fn scale_shift_and_quantile_shift_are_sane() {
    assert_eq!(SCALE_SHIFT, 16);
    assert_eq!(QUANTILE_SHIFT, 12);
    
    assert_eq!(reconstruct_q(1 << SCALE_SHIFT, 1 << QUANTILE_SHIFT), 1 << QUANTILE_SHIFT);
    assert_eq!(reconstruct_q(0, 12345), 0);
}

#[test]
fn sub_scale_pack_unpack_round_trip() {
    use crate::encode::{pack_sub_scales, unpack_sub_scales};
    
    let codes: Vec<u8> = (0u8..64).collect();
    let packed = pack_sub_scales(&codes);
    assert_eq!(packed.len(), (codes.len() * 6).div_ceil(8)); 
    assert_eq!(unpack_sub_scales(&packed, codes.len()), codes);
    
    let block = [5u8, 63, 0, 31, 32, 17, 63, 1];
    let pb = pack_sub_scales(&block);
    assert_eq!(pb.len(), 6);
    assert_eq!(unpack_sub_scales(&pb, 8), block);
}

#[test]
fn eff_scale_is_integer_and_monotone() {
    use crate::decode::eff_scale_q;
    let s = 1 << SCALE_SHIFT; 
    
    assert_eq!(eff_scale_q(s, 63), s);
    assert_eq!(eff_scale_q(s, 31), s / 2);
    assert_eq!(eff_scale_q(s, 0), s / 64);
    
    let mut prev = i32::MIN;
    for c in 0u8..64 {
        let e = eff_scale_q(s, c);
        assert!(e >= prev, "eff_scale not monotone at code={c}");
        prev = e;
    }
}

#[test]
fn adaptive_decode_is_deterministic_and_consistent() {
    
    let weights = normal_vec(1024, 0xADAB7);
    let cfg = TrellisConfig::for_bpw(4.0);
    let enc = encode_tensor(&weights, &cfg);
    
    assert!(!enc.blocks[0].sub_scales.is_empty());
    let a = decode_tensor_fixed(&enc, &cfg);
    let b = decode_tensor_fixed(&enc, &cfg);
    assert_eq!(a, b);
    let fa = decode_tensor(&enc, &cfg);
    for (&q, &f) in a.iter().zip(&fa) {
        assert_eq!(f.to_bits(), ((q as f32) * (1.0 / 4096.0)).to_bits());
    }
}

#[test]
fn adaptive_scales_beat_single_scale_on_nonstationary_weights() {
    use crate::encode::encode_tensor_opts;
    
    let n = 4096usize;
    let w: Vec<f32> = (0..n)
        .map(|i| {
            let sub = (i / 32) % 8; 
            let amp = 0.02 + (sub as f32) * 0.6; 
            ((i as f32) * 0.3).sin() * amp
        })
        .collect();
    let cfg = TrellisConfig::for_bpw(4.0);

    let flat = encode_tensor_opts(&w, &cfg, false);
    let adapt = encode_tensor_opts(&w, &cfg, true);
    let mse_flat = mse(&w, &decode_tensor(&flat, &cfg));
    let mse_adapt = mse(&w, &decode_tensor(&adapt, &cfg));

    assert!(
        mse_adapt < mse_flat,
        "adaptive sub-scales ({mse_adapt:.3e}) did not beat single-scale ({mse_flat:.3e})"
    );
    
    assert!(
        adapt.total_bpw(&cfg) < 4.5,
        "adaptive total bpw {} exceeded 4.5",
        adapt.total_bpw(&cfg)
    );
}

#[test]
fn rht_then_quant_then_inverse_tracks_input() {
    use crate::rht::{rht_forward, rht_inverse, RhtConfig};
    
    let w = normal_vec(4096, 0x12345);
    let rcfg = RhtConfig::from_seed(0xC0DE_1234);
    let tcfg = TrellisConfig::for_bpw(4.0);

    let fwd = rht_forward(&w, &rcfg);
    let id = rht_inverse(&fwd, &rcfg);
    let id_err = mse(&w, &id);
    assert!(id_err < 1e-6, "RHT round-trip not identity: mse {id_err:.3e}");

    let enc = encode_tensor(&fwd, &tcfg);
    let recon_inc = decode_tensor(&enc, &tcfg);
    let recon = rht_inverse(&recon_inc, &rcfg);
    let m = mse(&w, &recon);
    let signal = w.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / w.len() as f64;
    assert!(m < signal * 0.2, "RHT+quant+inverse mse {m:.3e} not well below signal {signal:.3e}");
}

use crate::encode::{encode_tensor_with, EncodeOpts};

fn replay_fixed(enc: &crate::EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    use crate::decode::eff_min_q;
    let lut = codebook_lut(cfg.l_bits);
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for blk in &enc.blocks {
        let n_sub = n_sub_blocks(blk.n as usize);
        let mults = unpack_sub_scales(&blk.sub_scales, n_sub);
        let offs: Vec<i32> = if enc.has_affine_min {
            unpack_sub_scales(&blk.mins, n_sub)
                .iter()
                .map(|&c| eff_min_q(blk.min_base_q, c))
                .collect()
        } else {
            Vec::new()
        };
        let nk = (blk.n as usize) * (k as usize);
        let start = if enc.tail_biting && nk >= cfg.l_bits as usize {
            let mut s = 0usize;
            let mut c = cursor;
            for _ in 0..blk.n as usize {
                let sym = read_bits(&enc.bits, c, k) & input_mask;
                c += k as usize;
                s = ((s << k) | sym) & mask;
            }
            s
        } else {
            blk.init_state as usize & mask
        };
        let mut state = start;
        for i in 0..blk.n as usize {
            let sym = read_bits(&enc.bits, cursor, k) & input_mask;
            cursor += k as usize;
            state = ((state << k) | sym) & mask;
            let es = eff_scale_q(blk.scale_q, mults[i / SUB_BLOCK]);
            let off = offs.get(i / SUB_BLOCK).copied().unwrap_or(0);
            out.push(reconstruct_q(es, lut[state]) + off);
        }
    }
    out
}

#[test]
fn tail_biting_round_trips_and_drops_init_state_bits() {
    
    let weights = normal_vec(2048, 0x7A11_B171);
    let cfg = TrellisConfig::for_bpw(3.0); 
    let opts = EncodeOpts {
        tail_biting: true,
        ..Default::default()
    };
    let enc = encode_tensor_with(&weights, &cfg, &opts);
    assert!(enc.tail_biting);

    let a = decode_tensor_fixed(&enc, &cfg);
    let b = decode_tensor_fixed(&enc, &cfg);
    assert_eq!(a, b, "tail-biting decode not deterministic");
    assert_eq!(a, replay_fixed(&enc, &cfg), "decoder diverged from tail-biting replay");
    assert_eq!(a.len(), weights.len());

    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let mut cursor = 0usize;
    for blk in &enc.blocks {
        let mut end = 0usize;
        for _ in 0..blk.n as usize {
            let sym = read_bits(&enc.bits, cursor, k) & input_mask;
            cursor += k as usize;
            end = ((end << k) | sym) & mask;
        }
        if (blk.n as usize) * (k as usize) >= cfg.l_bits as usize {
            assert_eq!(
                blk.init_state as usize & mask,
                end,
                "tail-biting start state != end state"
            );
        }
    }

    let nt = encode_tensor_with(
        &weights,
        &cfg,
        &EncodeOpts { tail_biting: false, ..Default::default() },
    );
    let n_blocks = enc.blocks.len() as f64;
    let saved_bpw = nt.total_bpw(&cfg) - enc.total_bpw(&cfg);
    let expected = (cfg.l_bits as f64 * n_blocks) / weights.len() as f64;
    assert!(
        (saved_bpw - expected).abs() < 1e-9,
        "tail-biting saved {saved_bpw} bpw, expected {expected}"
    );
}

#[test]
fn tail_biting_short_final_block_decodes_correctly() {
    
    for &n in &[600usize, 257, 256 * 3 + 2, 256 * 2 + 1] {
        for &bpw in &[2u32, 3, 4] {
            let w = normal_vec(n, 0x7A11_0000 ^ (n as u64) ^ ((bpw as u64) << 32));
            let cfg = TrellisConfig::for_bpw_l(bpw as f64, 12);
            let enc = encode_tensor_with(
                &w,
                &cfg,
                &EncodeOpts { tail_biting: true, ..Default::default() },
            );
            let d = decode_tensor_fixed(&enc, &cfg);
            assert_eq!(d.len(), n, "n={n} bpw={bpw}: wrong length");
            assert_eq!(
                d,
                replay_fixed(&enc, &cfg),
                "tail-biting short-block decode diverged (n={n}, bpw={bpw})"
            );
            assert_eq!(d, decode_tensor_fixed(&enc, &cfg), "not deterministic");
        }
    }
}

#[test]
fn affine_min_round_trips_and_helps_offset_data() {
    
    let n = 4096usize;
    let w: Vec<f32> = (0..n)
        .map(|i| {
            let sub = (i / 32) % 8;
            let dc = -2.0 + (sub as f32) * 0.5; 
            dc + ((i as f32) * 0.21).sin() * 0.1
        })
        .collect();
    let cfg = TrellisConfig::for_bpw(4.0);

    let off = EncodeOpts { affine_min: false, tail_biting: true, ..Default::default() };
    let on = EncodeOpts { affine_min: true, tail_biting: true, ..Default::default() };
    let e_off = encode_tensor_with(&w, &cfg, &off);
    let e_on = encode_tensor_with(&w, &cfg, &on);
    assert!(e_on.has_affine_min && !e_off.has_affine_min);

    let d = decode_tensor_fixed(&e_on, &cfg);
    assert_eq!(d, decode_tensor_fixed(&e_on, &cfg));
    assert_eq!(d, replay_fixed(&e_on, &cfg), "affine-min decoder diverged from replay");

    let m_off = mse(&w, &decode_tensor(&e_off, &cfg));
    let m_on = mse(&w, &decode_tensor(&e_on, &cfg));
    assert!(
        m_on < m_off,
        "affine-min ({m_on:.4e}) did not beat no-offset ({m_off:.4e}) on DC-offset data"
    );
    
    let bpw = e_on.total_bpw(&cfg);
    assert!(bpw < 4.51, "affine-min+tail-biting bpw {bpw} not at Q4_K parity");
}

#[test]
fn eff_min_q_is_integer_and_reaches_base() {
    use crate::decode::eff_min_q;
    
    let base = 12345i32;
    
    assert_eq!(eff_min_q(base, 0), 0);  
    assert_eq!(eff_min_q(base, 32), 0); 
    
    assert_eq!(eff_min_q(base, 31), -base); 
    assert_eq!(eff_min_q(base, 63), base);  
    
    let mut prev = 0i32;
    for c in 0u8..=31 {
        let v = eff_min_q(base, c);
        assert!(v <= prev, "negative side not monotone at code={c}");
        prev = v;
    }
    
    prev = 0;
    for c in 32u8..=63 {
        let v = eff_min_q(base, c);
        assert!(v >= prev, "positive side not monotone at code={c}");
        prev = v;
    }
}

#[test]
fn levers_compose_and_stay_deterministic() {
    
    let n = 3072usize;
    let w = normal_vec(n, 0xC0FF_EE42);
    let cfg = TrellisConfig::for_bpw_l(4.0, 12);
    let opts = EncodeOpts {
        adaptive: true,
        tail_biting: true,
        affine_min: true,
        silence_bonus: 0.0,
        ..Default::default()
    };
    let enc = encode_tensor_with(&w, &cfg, &opts);
    let d = decode_tensor_fixed(&enc, &cfg);
    assert_eq!(d, decode_tensor_fixed(&enc, &cfg), "composed levers not deterministic");
    assert_eq!(d, replay_fixed(&enc, &cfg), "composed-lever decoder diverged from replay");
    assert_eq!(d.len(), n);
    
    let f = decode_tensor(&enc, &cfg);
    for (&q, &x) in d.iter().zip(&f) {
        assert_eq!(x.to_bits(), ((q as f32) * (1.0 / 4096.0)).to_bits());
    }
}

use crate::encode::{encode_tensor_with_lut, vector_lut_from_scalar};
use crate::decode::decode_tensor_fixed_with_lut;
use crate::learned_codebook::train_state_vector_lut;

fn q12_mse(weights: &[f32], q12: &[i32]) -> f64 {
    debug_assert_eq!(weights.len(), q12.len());
    let to_real = 1.0f64 / (1u32 << QUANTILE_SHIFT) as f64;
    let mut acc = 0.0f64;
    for (&w, &q) in weights.iter().zip(q12) {
        let diff = w as f64 - (q as f64) * to_real;
        acc += diff * diff;
    }
    acc / weights.len().max(1) as f64
}

#[test]
fn b1_d1_vector_path_byte_identical_to_scalar() {
    
    let weights = normal_vec(700, 0x0B1D_0001);
    for k in [2u32, 3, 4] {
        let cfg = TrellisConfig::for_bpw(k as f64);
        assert_eq!(cfg.vec_dim, 1, "scalar config must default to d=1");
        let scalar_lut = codebook_lut(cfg.l_bits);

        let enc_ref = encode_tensor(&weights, &cfg);
        let dec_ref = decode_tensor_fixed(&enc_ref, &cfg);

        let enc_lut = encode_tensor_with_lut(&weights, &cfg, &EncodeOpts::default(), scalar_lut);
        let dec_lut = decode_tensor_fixed_with_lut(&enc_lut, &cfg, scalar_lut);
        assert_eq!(enc_lut, enc_ref, "d=1 explicit-LUT encode diverged from scalar (k={k})");
        assert_eq!(dec_lut, dec_ref, "d=1 explicit-LUT decode diverged from scalar (k={k})");

        let bcast = vector_lut_from_scalar(scalar_lut, 1);
        assert_eq!(bcast, scalar_lut, "broadcast at d=1 must equal the scalar LUT");
    }
}

#[test]
fn b1_d2_round_trips_and_is_deterministic() {
    let weights = normal_vec(600, 0x0B1D_0002);
    
    let cfg = TrellisConfig::for_bpw(4.0).with_vec_dim(2);
    assert_eq!(cfg.vec_dim, 2);
    assert_eq!(cfg.k_bits, 4);

    let lut = train_state_vector_lut(&weights, cfg.l_bits, 2, 0xABCD, 40);
    assert_eq!(lut.len(), cfg.num_states() * 2, "vector LUT must be [2^L * d]");

    let enc = encode_tensor_with_lut(&weights, &cfg, &EncodeOpts::default(), &lut);

    let a = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
    let b = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
    assert_eq!(a, b, "d=2 decode is not deterministic");
    assert_eq!(a.len(), weights.len(), "d=2 decode produced wrong length");

    let enc2 = encode_tensor_with_lut(&weights, &cfg, &EncodeOpts::default(), &lut);
    assert_eq!(enc, enc2, "d=2 re-encode changed the bits");

    assert!((enc.payload_bpw(&cfg) - 2.0).abs() < 1e-12, "d=2 payload bpw != k/d");
    let total_syms = enc.index_symbols(&cfg).len();
    let expected_syms: usize = enc.blocks.iter().map(|blk| (blk.n as usize).div_ceil(2)).sum();
    assert_eq!(total_syms, expected_syms, "symbol count != sum ceil(n/2)");

    let to_real = 1.0f64 / (1u32 << QUANTILE_SHIFT) as f64;
    let recon: Vec<f32> = a.iter().map(|&q| (q as f32) * to_real as f32).collect();
    let m = mse(&weights, &recon);
    let signal = weights.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / weights.len() as f64;
    assert!(m < signal, "d=2 reconstruction MSE {m} not below signal energy {signal}");

    let recon_pub: Vec<f32> = a.iter().map(|&q| (q as f32) * (1.0 / 4096.0)).collect();
    for (&q, &x) in a.iter().zip(&recon_pub) {
        assert_eq!(x.to_bits(), ((q as f32) * (1.0 / 4096.0)).to_bits());
    }
}

#[test]
fn b1_d_partial_final_block_exact_length() {
    
    for d in [2usize, 3, 4] {
        let n = 256 + 1; 
        let weights = normal_vec(n, 0x0B1D_0003 + d as u64);
        let cfg = TrellisConfig::for_bpw(4.0).with_vec_dim(d as u32);
        let lut = train_state_vector_lut(&weights, cfg.l_bits, d, 7, 25);
        let enc = encode_tensor_with_lut(&weights, &cfg, &EncodeOpts::default(), &lut);
        let dec = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
        assert_eq!(dec.len(), n, "d={d} partial-final-block decode wrong length");
        
        assert_eq!(dec, decode_tensor_fixed_with_lut(&enc, &cfg, &lut));
    }
}

#[test]
fn b1_d2_tail_biting_round_trips() {
    let weights = normal_vec(1024, 0x0B1D_0004);
    let cfg = TrellisConfig::for_bpw(4.0).with_vec_dim(2);
    let lut = train_state_vector_lut(&weights, cfg.l_bits, 2, 99, 30);
    let opts = EncodeOpts { tail_biting: true, ..Default::default() };
    let enc = encode_tensor_with_lut(&weights, &cfg, &opts, &lut);
    assert!(enc.tail_biting);
    let a = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
    assert_eq!(a, decode_tensor_fixed_with_lut(&enc, &cfg, &lut), "tail-biting decode not deterministic");
    assert_eq!(a.len(), weights.len());
    
    let enc_nt = encode_tensor_with_lut(&weights, &cfg, &EncodeOpts::default(), &lut);
    let b = decode_tensor_fixed_with_lut(&enc_nt, &cfg, &lut);
    let m_tb = q12_mse(&weights, &a);
    let m_nt = q12_mse(&weights, &b);
    assert!(m_tb < m_nt * 1.5 + 1e-9, "tail-biting MSE {m_tb} >> non-tail-biting {m_nt}");
}

fn b1_iso_bpw_scalar_vs_vector(rate: u32, d: u32, n: usize, seed: u64) -> (f64, f64) {
    let weights = normal_vec(n, seed);

    let cfg_s = TrellisConfig::for_bpw(rate as f64);
    assert_eq!(cfg_s.k_bits, rate);
    assert_eq!(cfg_s.vec_dim, 1);
    let lut_s = train_state_vector_lut(&weights, cfg_s.l_bits, 1, 0x5EED_1111, 60);
    let enc_s = encode_tensor_with_lut(&weights, &cfg_s, &EncodeOpts::default(), &lut_s);
    let dec_s = decode_tensor_fixed_with_lut(&enc_s, &cfg_s, &lut_s);
    let mse_s = q12_mse(&weights, &dec_s);

    let k_vec = rate * d;
    assert!(k_vec <= TrellisConfig::MAX_K, "k=rate*d exceeds MAX_K — pick a feasible rate/d");
    let cfg_v = TrellisConfig::for_bpw_l(k_vec as f64, cfg_s.l_bits).with_vec_dim(d);
    assert_eq!(cfg_v.k_bits, k_vec);
    assert_eq!(cfg_v.l_bits, cfg_s.l_bits);
    
    assert!((enc_s.payload_bpw(&cfg_s) - rate as f64).abs() < 1e-12);
    let lut_v = train_state_vector_lut(&weights, cfg_v.l_bits, d as usize, 0x5EED_2222, 60);
    let enc_v = encode_tensor_with_lut(&weights, &cfg_v, &EncodeOpts::default(), &lut_v);
    assert!((enc_v.payload_bpw(&cfg_v) - rate as f64).abs() < 1e-12, "vector payload != iso rate");
    let dec_v = decode_tensor_fixed_with_lut(&enc_v, &cfg_v, &lut_v);
    let mse_v = q12_mse(&weights, &dec_v);

    (mse_s, mse_v)
}

#[test]
fn b1_rung1_learned_d2_beats_scalar_at_iso_bpw() {
    let n = 8192usize; 
    
    let (mse_s2, mse_v2) = b1_iso_bpw_scalar_vs_vector(2, 2, n, 0xD2_2000);
    let gain_db_2 = 10.0 * (mse_s2 / mse_v2).log10();
    eprintln!(
        "[B1 Rung-1] d=2 @ R=2 iso-bpw: scalar MSE {mse_s2:.6e}, vector MSE {mse_v2:.6e}, \
         gain {gain_db_2:.3} dB (full-VQ ceiling +0.41 dB)"
    );
    assert!(
        mse_v2 < mse_s2,
        "learned d=2 trellis did NOT beat scalar at R=2 (scalar {mse_s2:.6e} vs vector {mse_v2:.6e})"
    );
}

#[test]
fn b1_rung1_learned_d4_beats_scalar_at_iso_bpw() {
    let n = 8192usize;
    
    let (mse_s1, mse_v4) = b1_iso_bpw_scalar_vs_vector(1, 4, n, 0xD4_1000);
    let (_, mse_v2) = b1_iso_bpw_scalar_vs_vector(1, 2, n, 0xD4_1000);
    let gain_db_4 = 10.0 * (mse_s1 / mse_v4).log10();
    let gain_db_2 = 10.0 * (mse_s1 / mse_v2).log10();
    eprintln!(
        "[B1 Rung-1] @ R=1 iso-bpw: scalar MSE {mse_s1:.6e}; d=2 MSE {mse_v2:.6e} ({gain_db_2:.3} dB); \
         d=4 MSE {mse_v4:.6e} ({gain_db_4:.3} dB) — captured gain should grow with d"
    );
    assert!(
        mse_v4 < mse_s1,
        "learned d=4 trellis did NOT beat scalar at R=1 (scalar {mse_s1:.6e} vs d=4 {mse_v4:.6e})"
    );
    
    assert!(
        gain_db_4 >= gain_db_2 - 0.05,
        "d=4 captured LESS than d=2 ({gain_db_4:.3} dB vs {gain_db_2:.3} dB) — space-filling ordering violated"
    );
}

// ---------------------------------------------------------------------------
// (vii) COMPUTED-CODEBOOK EQUIVALENCE GATE (T2).
//
// CodebookMode::ComputedAcklam (Variant A) reproduces the frozen codebook
// byte-for-byte, so it must leave BOTH the encoder's emitted bits AND the
// decoder's output completely unchanged vs the default StoredLut, across the
// full matrix of register widths, rates, and quality levers. This is the hard
// gate: a single byte of divergence here means the computed path is not exact.
// ---------------------------------------------------------------------------

use crate::trellis::CodebookMode;

/// Assert an encode→decode round trip is byte-identical under StoredLut and
/// ComputedAcklam for the given weights / config / opts.
#[cfg(test)]
fn assert_codebook_modes_identical(weights: &[f32], cfg: &TrellisConfig, opts: &EncodeOpts) {
    let cfg_stored = cfg.with_codebook_mode(CodebookMode::StoredLut);
    let cfg_computed = cfg.with_codebook_mode(CodebookMode::ComputedAcklam);

    let enc_s = encode_tensor_with(weights, &cfg_stored, opts);
    let enc_c = encode_tensor_with(weights, &cfg_computed, opts);

    // Same emitted payload + side info, field for field.
    assert_eq!(enc_s.bits, enc_c.bits, "L={} k={}: payload bits diverged", cfg.l_bits, cfg.k_bits);
    assert_eq!(enc_s.total, enc_c.total, "L={} k={}: total diverged", cfg.l_bits, cfg.k_bits);
    assert_eq!(
        enc_s.has_affine_min, enc_c.has_affine_min,
        "L={} k={}: affine flag diverged", cfg.l_bits, cfg.k_bits
    );
    assert_eq!(
        enc_s.tail_biting, enc_c.tail_biting,
        "L={} k={}: tail-biting flag diverged", cfg.l_bits, cfg.k_bits
    );
    assert_eq!(
        enc_s.blocks.len(), enc_c.blocks.len(),
        "L={} k={}: block count diverged", cfg.l_bits, cfg.k_bits
    );
    for (bi, (a, b)) in enc_s.blocks.iter().zip(&enc_c.blocks).enumerate() {
        assert_eq!(a, b, "L={} k={} block {bi}: block side-info diverged", cfg.l_bits, cfg.k_bits);
    }
    // The whole struct, for good measure (EncodedTensor: PartialEq).
    assert_eq!(enc_s, enc_c, "L={} k={}: EncodedTensor diverged", cfg.l_bits, cfg.k_bits);

    // Decode must also be byte-identical — both that the two modes agree with each
    // other, AND that decoding ONE encoding under the other mode is unchanged
    // (proves the decode-side gather→compute swap is exact in isolation, not just
    // that encode+decode drift together).
    let d_ss = decode_tensor_fixed(&enc_s, &cfg_stored);
    let d_cc = decode_tensor_fixed(&enc_c, &cfg_computed);
    let d_sc = decode_tensor_fixed(&enc_s, &cfg_computed); // stored bits, computed decode
    let d_cs = decode_tensor_fixed(&enc_c, &cfg_stored); // computed bits, stored decode
    assert_eq!(d_ss, d_cc, "L={} k={}: decode diverged across modes", cfg.l_bits, cfg.k_bits);
    assert_eq!(d_ss, d_sc, "L={} k={}: computed decode of stored bits diverged", cfg.l_bits, cfg.k_bits);
    assert_eq!(d_ss, d_cs, "L={} k={}: stored decode of computed bits diverged", cfg.l_bits, cfg.k_bits);
}

#[test]
fn computed_codebook_encode_decode_equivalence_matrix() {
    // Across every frozen register width L (4..=14), each supported rate k, and
    // each headline option combination, ComputedAcklam == StoredLut byte-for-byte.
    //
    // The *bit-exactness* of every codebook entry is already proven exhaustively
    // by `codebook::computed_codebook_matches_frozen` (all states, all L). This
    // test proves the *wiring*: that encode + decode genuinely route through the
    // value and stay identical. Viterbi cost is ~block_len*2^L*2^k, so we keep
    // full L/k coverage but bound the work per cell (small tensor + full option
    // grid only where it is cheap; a single representative grid entry at high L).
    let weights = normal_vec(320, 0xC0DE_B007_F00Du64);
    // (adaptive, tail_biting, affine_min) headline lever combinations.
    let full_grid: &[(bool, bool, bool)] = &[
        (false, false, false),
        (true, false, false),
        (true, true, false),
        (true, false, true),
        (true, true, true),
    ];
    let cheap_grid: &[(bool, bool, bool)] = &[(true, true, true)];
    // Cover L up to 12 across the full k range and option grid. Every codebook
    // *value* at every state for L up to 14 is already proven bit-identical by
    // `codebook::computed_codebook_matches_frozen`; this loop proves the
    // encode/decode *plumbing* routes through it. L=13/14 only inflate Viterbi
    // cost (~2^L) without exercising any new wiring, so they get a single smoke
    // cell below rather than the full grid.
    for l in TrellisConfig::MIN_L..=12 {
        for k in 1..=TrellisConfig::MAX_K.min(l) {
            let cfg = TrellisConfig::new(l, k, 128);
            // Full option grid only where the Viterbi table is small; one
            // representative (all-levers) combo otherwise.
            let grid = if (l as u64) * (1u64 << k) <= (12 << 3) { full_grid } else { cheap_grid };
            for &(adaptive, tail_biting, affine_min) in grid {
                let opts = EncodeOpts {
                    adaptive,
                    tail_biting,
                    affine_min,
                    ..Default::default()
                };
                assert_codebook_modes_identical(&weights, &cfg, &opts);
            }
        }
    }
    // Highest-width smoke cell: prove the L=13 and L=14 paths are wired too.
    for l in [13u32, 14] {
        let cfg = TrellisConfig::new(l, 2, 128);
        let opts = EncodeOpts { adaptive: true, tail_biting: true, affine_min: true, ..Default::default() };
        assert_codebook_modes_identical(&weights, &cfg, &opts);
    }
}

#[test]
fn computed_codebook_exhaustive_small_trellis_equivalence() {
    // Exhaustive-style stress on the *small* trellises where every state is
    // exercised many times: many seeds, short and long tensors, tiny blocks so
    // short-final-block + tail-biting edge paths are hit. Any state whose
    // computed codebook value differed from the stored one would surface here.
    for seed in [1u64, 2, 7, 42, 1000, 0xFFFF_FFFF] {
        for &n in &[1usize, 2, 17, 64, 255, 256, 257] {
            let w = normal_vec(n, seed);
            for l in [4u32, 5, 7, 8, 10, 12] {
                for k in [1u32, 2, 3] {
                    if k > l {
                        continue;
                    }
                    // small block to exercise more block boundaries
                    let cfg = TrellisConfig::new(l, k, 64);
                    let opts = EncodeOpts { adaptive: true, tail_biting: true, ..Default::default() };
                    assert_codebook_modes_identical(&w, &cfg, &opts);
                }
            }
        }
    }
}

#[test]
fn computed_codebook_gpu_eligible_path_identical() {
    // GPU-eligible configs (no tail-biting / diag_h / affine_min) are routed
    // through the Metal (macOS) or CUDA backend by `encode_tensor_with`. On this
    // branch the GPU backend is the *encode-side* Viterbi: it receives f32
    // reconstruction levels built host-side from the codebook, so under
    // ComputedAcklam those levels are byte-identical to StoredLut and the GPU
    // result must match. This test makes that coverage explicit (it silently
    // runs on the CPU fallback where no GPU is present — still a valid identity
    // check). It does NOT touch a decode-side GPU gather (none exists here; that
    // kernel lives in the separate strand-decode-kernel crate).
    let weights = normal_vec(2000, 0x6907_C0DE_F00Du64);
    for l in [8u32, 10, 12] {
        for k in [2u32, 4] {
            let cfg = TrellisConfig::new(l, k, 256);
            // adaptive on, the other levers off => GPU-eligible.
            let opts = EncodeOpts { adaptive: true, tail_biting: false, affine_min: false, ..Default::default() };
            assert_codebook_modes_identical(&weights, &cfg, &opts);
        }
    }
}

#[test]
fn computed_codebook_cfg_codebook_helper_matches_frozen() {
    // The TrellisConfig::codebook() accessor returns the frozen slice under
    // StoredLut and a byte-identical materialised Vec under ComputedAcklam.
    for l in TrellisConfig::MIN_L..=TrellisConfig::MAX_L {
        let cfg = TrellisConfig::new(l, 2, 256);
        let stored = cfg.with_codebook_mode(CodebookMode::StoredLut).codebook();
        let computed = cfg.with_codebook_mode(CodebookMode::ComputedAcklam).codebook();
        assert_eq!(stored.as_ref(), codebook_lut(l), "stored helper != frozen, L={l}");
        assert_eq!(computed.as_ref(), codebook_lut(l), "computed helper != frozen, L={l}");
    }
}
