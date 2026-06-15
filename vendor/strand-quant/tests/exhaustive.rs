
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::{decode_lean, decode_tensor_fixed};
use strand_quant::encode::{pack_sub_scales, BlockMeta, EncodedTensor};
use strand_quant::TrellisConfig;

fn ref_read_bits(bytes: &[u8], start_bit: usize, nbits: u32) -> usize {
    let mut v = 0usize;
    for i in 0..nbits as usize {
        let bit = start_bit + i;
        let byte = bit / 8;
        if byte < bytes.len() && (bytes[byte] >> (bit % 8)) & 1 == 1 {
            v |= 1 << i;
        }
    }
    v
}

fn ref_unpack6(bytes: &[u8], n: usize) -> Vec<u8> {
    (0..n).map(|i| ref_read_bits(bytes, i * 6, 6) as u8).collect()
}

fn ref_decode(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<i32> {
    let l = cfg.l_bits;
    let k = cfg.k_bits;
    let mask = (1usize << l) - 1;
    let imask = (1usize << k) - 1;
    let mut out = Vec::with_capacity(enc.total);
    let mut cursor = 0usize; 

    for blk in &enc.blocks {
        let n = blk.n as usize;
        let n_sub = n.div_ceil(32);
        let scodes = ref_unpack6(&blk.sub_scales, n_sub);
        let mcodes: Vec<u8> =
            if enc.has_affine_min { ref_unpack6(&blk.mins, n_sub) } else { Vec::new() };

        let mut state = if enc.tail_biting && n * k as usize >= l as usize {
            let mut s = 0usize;
            for i in 0..n {
                s = ((s << k) | (ref_read_bits(&enc.bits, cursor + i * k as usize, k) & imask))
                    & mask;
            }
            s
        } else {
            blk.init_state as usize & mask
        };

        for i in 0..n {
            let sym = ref_read_bits(&enc.bits, cursor, k) & imask;
            cursor += k as usize;
            state = ((state << k) | sym) & mask;
            let q = lut[state] as i64;
            let scode = (scodes[i / 32] & 0x3F) as i64;
            let es = ((blk.scale_q as i64 * (scode + 1)) >> 6) as i64;
            let recon = ((es * q) >> 16) as i32;
            let off = if enc.has_affine_min {
                let c = mcodes[i / 32];
                let mag = (c & 0x1F) as i64;
                if mag == 0 {
                    0i32
                } else {
                    let base = (blk.min_base_q.unsigned_abs()) as i64;
                    let s = if c & 0x20 != 0 { base * mag } else { -(base * mag) };
                    (s / 31) as i32
                }
            } else {
                0
            };
            out.push(recon + off);
        }
    }
    out
}

fn pack_symbols(syms: &[usize], k: u32) -> Vec<u8> {
    let total_bits = syms.len() * k as usize;
    let mut bytes = vec![0u8; total_bits.div_ceil(8)];
    let mut cursor = 0usize;
    for &s in syms {
        for b in 0..k as usize {
            if (s >> b) & 1 == 1 {
                bytes[cursor / 8] |= 1 << (cursor % 8);
            }
            cursor += 1;
        }
    }
    bytes
}

#[allow(clippy::too_many_arguments)]
fn make_tensor(
    block_syms: &[Vec<usize>],
    k: u32,
    init_states: &[u32],
    scale_qs: &[i32],
    sub_codes: &[Vec<u8>],
    tail_biting: bool,
    affine: Option<(&[i32], &[Vec<u8>])>, 
) -> EncodedTensor {
    let all_syms: Vec<usize> = block_syms.iter().flatten().copied().collect();
    let bits = pack_symbols(&all_syms, k);
    let mut blocks = Vec::new();
    let mut total = 0usize;
    for (b, syms) in block_syms.iter().enumerate() {
        let n = syms.len();
        total += n;
        let (min_base_q, mins) = match affine {
            Some((bases, codes)) => (bases[b], pack_sub_scales(&codes[b])),
            None => (0, Vec::new()),
        };
        blocks.push(BlockMeta {
            scale_q: scale_qs[b],
            sub_scales: pack_sub_scales(&sub_codes[b]),
            min_base_q,
            mins,
            init_state: init_states[b],
            n: n as u32,
        });
    }
    EncodedTensor {
        bits,
        blocks,
        total,
        has_rht_seed: false,
        tail_biting,
        has_affine_min: affine.is_some(),
    }
}

fn assert_three_way(enc: &EncodedTensor, cfg: &TrellisConfig, ctx: &str) {
    let lut = codebook_lut(cfg.l_bits);
    let reference = ref_decode(enc, cfg, lut);
    let lean = decode_lean(enc, cfg);
    let fixed = decode_tensor_fixed(enc, cfg);
    assert_eq!(lean, reference, "decode_lean != spec reference [{ctx}]");
    assert_eq!(fixed, reference, "decode_tensor_fixed != spec reference [{ctx}]");
}

const SCALES: [i32; 8] = [1 << 16, 4096, -(1 << 16), 1, -1, 0, i32::MAX, i32::MIN];

#[test]
fn exhaustive_state_stream_equivalence() {
    let mut covered = 0u64;
    for (l, k) in [(4u32, 2u32), (4, 3), (5, 2), (5, 3)] {
        let cfg = TrellisConfig::new(l, k, 256);
        assert_eq!((cfg.l_bits, cfg.k_bits), (l, k), "config clamped unexpectedly");
        let n_states = 1usize << l;
        let n_max = 12 / k as usize; 
        let mut tier = 0u64;
        for n in 1..=n_max {
            let n_streams = 1usize << (n * k as usize);
            for init in 0..n_states {
                for stream in 0..n_streams {
                    let syms: Vec<usize> =
                        (0..n).map(|i| (stream >> (i * k as usize)) & ((1 << k) - 1)).collect();
                    let scale = SCALES[(init + stream) % SCALES.len()];
                    let enc = make_tensor(
                        &[syms],
                        k,
                        &[init as u32],
                        &[scale],
                        &[vec![63u8]], 
                        false,
                        None,
                    );
                    assert_three_way(&enc, &cfg, &format!("L={l} k={k} n={n} init={init} stream={stream}"));
                    tier += 1;
                }
            }
        }
        
        let expect: u64 = (1..=n_max)
            .map(|n| (n_states as u64) * (1u64 << (n * k as usize)))
            .sum();
        assert_eq!(tier, expect, "coverage drifted at L={l} k={k}");
        covered += tier;
    }
    eprintln!("exhaustive (state x stream), non-tail-biting: {covered} tensors");
    
    assert_eq!(covered, 87_360 + 74_880 + 174_720 + 149_760);
}

#[test]
fn exhaustive_state_stream_equivalence_tail_biting() {
    let mut covered = 0u64;
    for (l, k) in [(4u32, 2u32), (4, 3), (5, 2), (5, 3)] {
        let cfg = TrellisConfig::new(l, k, 256);
        let n_states = 1usize << l;
        let n_max = 12 / k as usize;
        for n in 1..=n_max {
            let n_streams = 1usize << (n * k as usize);
            let nk = n * k as usize;
            let states: Vec<usize> =
                if nk >= l as usize { vec![0, n_states - 1] } else { (0..n_states).collect() };
            for stream in 0..n_streams {
                let syms: Vec<usize> =
                    (0..n).map(|i| (stream >> (i * k as usize)) & ((1 << k) - 1)).collect();
                let mut first: Option<Vec<i32>> = None;
                for &init in &states {
                    let scale = SCALES[(stream + n) % SCALES.len()];
                    let enc = make_tensor(
                        std::slice::from_ref(&syms),
                        k,
                        &[init as u32],
                        &[scale],
                        &[vec![63u8]],
                        true,
                        None,
                    );
                    assert_three_way(&enc, &cfg, &format!("TB L={l} k={k} n={n} init={init} stream={stream}"));
                    covered += 1;
                    
                    if nk >= l as usize {
                        let out = decode_lean(&enc, &cfg);
                        match &first {
                            None => first = Some(out),
                            Some(f) => assert_eq!(
                                &out, f,
                                "tail-bitten output depends on stored init_state (L={l} k={k} n={n})"
                            ),
                        }
                    }
                }
            }
        }
    }
    eprintln!("exhaustive (state x stream), tail-biting: {covered} tensors");
}

#[test]
fn exhaustive_sub_scale_codes() {
    let mut covered = 0u64;
    for (l, k) in [(4u32, 2u32), (5, 3)] {
        let cfg = TrellisConfig::new(l, k, 256);
        let n = 33usize;
        
        let syms: Vec<usize> =
            (0..n).map(|i| (i.wrapping_mul(2654435761) >> 7) & ((1 << k) - 1)).collect();
        for c0 in 0u8..64 {
            for c1 in 0u8..64 {
                let scale = SCALES[(c0 as usize * 64 + c1 as usize) % 4]; 
                let enc = make_tensor(
                    std::slice::from_ref(&syms),
                    k,
                    &[3],
                    &[scale],
                    &[vec![c0, c1]],
                    false,
                    None,
                );
                assert_three_way(&enc, &cfg, &format!("subscale L={l} k={k} c0={c0} c1={c1}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 2 * 64 * 64);
}

#[test]
fn exhaustive_affine_min_codes() {
    let cfg = TrellisConfig::new(4, 2, 256);
    let n = 33usize;
    let syms: Vec<usize> = (0..n).map(|i| (i.wrapping_mul(40503) >> 3) & 0x3).collect();
    let bases = [0i32, 1, 4096, 1 << 20];
    let mut covered = 0u64;
    for &base in &bases {
        for m0 in 0u8..64 {
            for m1 in 0u8..64 {
                let enc = make_tensor(
                    std::slice::from_ref(&syms),
                    2,
                    &[9],
                    &[1 << 16],
                    &[vec![63, 17]],
                    false,
                    Some((&[base], &[vec![m0, m1]])),
                );
                assert_three_way(&enc, &cfg, &format!("affine base={base} m0={m0} m1={m1}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 4 * 64 * 64);
}

#[test]
fn boundary_geometries_three_way() {
    let mut covered = 0u64;
    for (l, k) in [(4u32, 2u32), (4, 3), (5, 2), (5, 3)] {
        let cfg = TrellisConfig::new(l, k, 256);
        for tail in [false, true] {
            for affine in [false, true] {
                for &tail_len in &[1usize, 2, 5, 31, 32, 33, 63, 256] {
                    let lens = [256usize, 256, tail_len];
                    let mut block_syms = Vec::new();
                    let mut inits = Vec::new();
                    let mut scales = Vec::new();
                    let mut subc: Vec<Vec<u8>> = Vec::new();
                    let mut bases = Vec::new();
                    let mut minc: Vec<Vec<u8>> = Vec::new();
                    for (b, &n) in lens.iter().enumerate() {
                        let syms: Vec<usize> = (0..n)
                            .map(|i| {
                                ((i + b * 977).wrapping_mul(2654435761) >> 9) & ((1 << k) - 1)
                            })
                            .collect();
                        block_syms.push(syms);
                        inits.push(((b * 7 + tail_len) % (1 << l)) as u32);
                        scales.push(SCALES[(b + tail_len + l as usize) % SCALES.len()]);
                        let ns = n.div_ceil(32);
                        subc.push((0..ns).map(|s| ((s * 11 + b * 3) % 64) as u8).collect());
                        bases.push([0i32, 4096, 1 << 18][b % 3]);
                        minc.push((0..ns).map(|s| ((s * 23 + b * 5) % 64) as u8).collect());
                    }
                    let aff = if affine { Some((&bases[..], &minc[..])) } else { None };
                    let enc =
                        make_tensor(&block_syms, k, &inits, &scales, &subc, tail, aff);
                    assert_three_way(
                        &enc,
                        &cfg,
                        &format!("geom L={l} k={k} tail={tail} affine={affine} tail_len={tail_len}"),
                    );
                    covered += 1;
                }
            }
        }
    }
    eprintln!("boundary geometries: {covered} tensors (3 blocks each)");
    assert_eq!(covered, 4 * 2 * 2 * 8);
}

#[test]
fn f32_wrapper_is_exact_q12() {
    use strand_quant::decode_tensor;
    let cfg = TrellisConfig::new(5, 3, 256);
    let n = 100usize;
    let syms: Vec<usize> = (0..n).map(|i| (i * 5 + 3) & 0x7).collect();
    for &scale in &SCALES {
        let enc = make_tensor(std::slice::from_ref(&syms), 3, &[17], &[scale], &[vec![63, 40, 1, 63]], false, None);
        let fixed = decode_tensor_fixed(&enc, &cfg);
        let f = decode_tensor(&enc, &cfg);
        for (a, b) in fixed.iter().zip(f.iter()) {
            assert_eq!(*b, (*a as f32) * (1.0 / 4096.0), "f32 wrapper drift at scale={scale}");
        }
    }
}
