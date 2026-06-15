
use strand_quant::decode::{eff_min_q, eff_scale_q};
use strand_quant::encode::{n_sub_blocks, unpack_sub_scales, BlockMeta, EncodedTensor};
use strand_quant::trellis::read_bits;
use strand_quant::TrellisConfig;

pub const MAX_SUB: usize = 256;

#[inline]
pub fn exceeds_max_sub(enc: &EncodedTensor) -> bool {
    enc.blocks.iter().any(|b| n_sub_blocks(b.n as usize) > MAX_SUB)
}

pub struct WordReader<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) word_idx: usize,
    pub(crate) acc: u64,
    pub(crate) have: u32,
}

impl<'a> WordReader<'a> {
    
    #[inline]
    pub(crate) fn load_u32_le(bytes: &[u8], wi: usize) -> u32 {
        let b = wi * 4;
        if b + 4 <= bytes.len() {
            u32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
        } else {
            let g = |o: usize| -> u32 {
                if b + o < bytes.len() {
                    bytes[b + o] as u32
                } else {
                    0
                }
            };
            g(0) | (g(1) << 8) | (g(2) << 16) | (g(3) << 24)
        }
    }

    #[inline]
    pub fn new(bytes: &'a [u8], start_bit: usize) -> Self {
        let word_idx = start_bit >> 5;
        let bit_in_w = (start_bit & 31) as u32;
        let acc = (Self::load_u32_le(bytes, word_idx) as u64) >> bit_in_w;
        WordReader { bytes, word_idx, acc, have: 32 - bit_in_w }
    }

    #[inline]
    pub fn pop(&mut self, k: u32) -> usize {
        if self.have < k {
            self.word_idx += 1;
            let nxt = Self::load_u32_le(self.bytes, self.word_idx) as u64;
            self.acc |= nxt << self.have;
            self.have += 32;
        }
        let sym = (self.acc & ((1u64 << k) - 1)) as usize;
        self.acc >>= k;
        self.have -= k;
        sym
    }
}

#[derive(Clone, Copy)]
pub struct BlockPlan {
    
    pub start_bit: usize,
    
    pub out_off: usize,
    
    pub n: usize,
}

pub fn block_plans(enc: &EncodedTensor, k: usize) -> Vec<BlockPlan> {
    let mut plans = Vec::with_capacity(enc.blocks.len());
    let mut prefix_n = 0usize;
    for blk in &enc.blocks {
        let n = blk.n as usize;
        plans.push(BlockPlan { start_bit: prefix_n * k, out_off: prefix_n, n });
        prefix_n += n;
    }
    plans
}

pub struct SideInfo {
    pub(crate) eff: [i32; MAX_SUB],
    pub(crate) off: [i32; MAX_SUB],
    pub(crate) n_sub: usize,
}

impl SideInfo {
    
    #[inline(always)]
    pub fn hoist(blk: &BlockMeta, has_affine: bool) -> Self {
        let n_sub = n_sub_blocks(blk.n as usize);
        debug_assert!(n_sub <= MAX_SUB, "caller must gate on exceeds_max_sub");
        let mut eff = [0i32; MAX_SUB];
        let mut off = [0i32; MAX_SUB];
        let mults = unpack_sub_scales(&blk.sub_scales, n_sub);
        for (e, &m) in eff[..n_sub].iter_mut().zip(mults.iter()) {
            *e = eff_scale_q(blk.scale_q, m);
        }
        if has_affine {
            let codes = unpack_sub_scales(&blk.mins, n_sub);
            for (o, &c) in off[..n_sub].iter_mut().zip(codes.iter()) {
                *o = eff_min_q(blk.min_base_q, c);
            }
        }
        SideInfo { eff, off, n_sub }
    }

    #[inline(always)]
    pub fn eff(&self) -> &[i32] {
        &self.eff[..self.n_sub]
    }

    #[inline(always)]
    pub fn off(&self) -> &[i32] {
        &self.off[..self.n_sub]
    }
}

#[inline]
pub fn block_init_state(
    blk: &BlockMeta,
    bits: &[u8],
    start_bit: usize,
    cfg: &TrellisConfig,
    tail_biting: bool,
) -> usize {
    let mask = cfg.state_mask();
    let k = cfg.k_bits;
    let input_mask = cfg.num_inputs() - 1;
    let n = blk.n as usize;
    let nk = n * (k as usize);
    if tail_biting && nk >= cfg.l_bits as usize {
        let mut s = 0usize;
        let mut c = start_bit;
        for _ in 0..n {
            let sym = read_bits(bits, c, k) & input_mask;
            c += k as usize;
            s = ((s << k) | sym) & mask;
        }
        s
    } else {
        blk.init_state as usize & mask
    }
}

pub mod gate_proto {
    use strand_quant::encode::{BlockMeta, EncodedTensor, SUB_BLOCK};
    use strand_quant::TrellisConfig;

    pub fn pack_unit_sub_scales(n_sub: usize) -> Vec<u8> {
        let total_bits = 6 * n_sub;
        let nbytes = total_bits.div_ceil(8);
        let mut bytes = vec![0u8; nbytes];
        let mut cursor = 0usize;
        for _ in 0..n_sub {
            for b in 0..6 {
                let bit_idx = cursor + b;
                bytes[bit_idx >> 3] |= 1u8 << (bit_idx & 7);
            }
            cursor += 6;
        }
        bytes
    }

    pub fn synth_encoded(total: usize, k: u32, block_len: usize) -> EncodedTensor {
        let n_blocks = total.div_ceil(block_len);
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut emitted = 0usize;
        for _ in 0..n_blocks {
            let n = block_len.min(total - emitted);
            let n_sub = n.div_ceil(SUB_BLOCK);
            blocks.push(BlockMeta {
                scale_q: 1 << 16,
                sub_scales: pack_unit_sub_scales(n_sub),
                min_base_q: 0,
                mins: Vec::new(),
                init_state: 0,
                n: n as u32,
            });
            emitted += n;
        }
        let payload_bits = (total as u64) * (k as u64);
        let payload_bytes = (payload_bits.div_ceil(8)) as usize;
        let mut bits = vec![0u8; payload_bytes + 8];
        let mut x = 0x9E3779B97F4A7C15u64;
        for b in bits.iter_mut().take(payload_bytes) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x >> 33) as u8;
        }
        EncodedTensor {
            bits,
            blocks,
            total,
            has_affine_min: false,
            tail_biting: false,
            has_rht_seed: false,
        }
    }

    pub fn canonical_configs() -> Vec<(TrellisConfig, &'static str)> {
        vec![
            (TrellisConfig::for_bpw(3.0), "k3 L7 (3-bit deploy)"),
            (TrellisConfig::for_bpw(2.0), "k2 L6"),
            (TrellisConfig::for_bpw(4.0), "k4 L8"),
            (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12 (2-bit reopen)"),
            (TrellisConfig::for_bpw_l(2.0, 5), "k2 L5 (fold)"),
            (TrellisConfig::for_bpw_l(3.0, 5), "k3 L5 (fold)"),
            (TrellisConfig::for_bpw_l(4.0, 4), "k4 L4 (fold)"),
        ]
    }

    pub fn machine_stamp() -> String {
        let load = std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "loadavg unavailable".into());
        let co = std::process::Command::new("pgrep")
            .args(["-fl", "strand-delta|strand-qat|quantize-model|strand-7b-ppl"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| {
                let lines: Vec<&str> = s.lines().collect();
                format!("CO-RUNNING ({}): {}", lines.len(), lines.join("; "))
            })
            .unwrap_or_else(|| "no co-running STRAND science jobs".into());
        format!(
            "machine: loadavg {load} | {co} | rayon threads {}",
            rayon::current_num_threads()
        )
    }
}
