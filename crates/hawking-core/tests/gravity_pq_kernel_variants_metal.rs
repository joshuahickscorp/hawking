#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::gravity::{
    parse_pq_header, pq_matvec_f64_authority, pq_matvec_metal, pq_sections, PqMetalKernelVariant,
    PqMetalMatrix,
};
use hawking_core::metal::MetalContext;
use hawking_core::numeric_parity::{format_score_line, score_against_f64, Bounds};
fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn make_bits8_payload(rows: u32, nchunk: u32) -> Vec<u8> {
    const D: u16 = 32;
    const SUB: u16 = 32;
    const CARD: u16 = 256;
    let cols = nchunk * D as u32;
    let mut out =
        Vec::with_capacity(64 + CARD as usize * SUB as usize * 2 + rows as usize * nchunk as usize);
    out.extend_from_slice(b"GLM52CPK");
    push_u16(&mut out, D);
    push_u16(&mut out, 1);
    push_u16(&mut out, SUB);
    push_u16(&mut out, CARD);
    push_u32(&mut out, rows);
    push_u32(&mut out, cols);
    push_u32(&mut out, nchunk);
    push_u32(&mut out, 0x52A1);
    push_u16(&mut out, 8);
    out.push(0);
    out.push(1);
    out.resize(64, 0);
    for code in 0..CARD as usize {
        for j in 0..SUB as usize {
            let raw = ((code * 29 + j * 17 + (code ^ j) * 3) % 509) as f32;
            let v = (raw - 254.0) / 192.0;
            push_u16(&mut out, f16::from_f32(v).to_bits());
        }
    }
    for row in 0..rows as usize {
        for chunk in 0..nchunk as usize {
            let code = row
                .wrapping_mul(73)
                .wrapping_add(chunk.wrapping_mul(41))
                .wrapping_add(row.wrapping_mul(chunk).wrapping_mul(3));
            out.push((code & 255) as u8);
        }
    }
    out
}
fn make_x(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| {
            let a = ((i as f32 + 0.5) * 0.017578125).sin();
            let b = ((i * 31 % 127) as f32 - 63.0) / 256.0;
            a + b
        })
        .collect()
}
#[derive(Clone, Copy, Default)]
struct DoubleSingle {
    hi: f32,
    lo: f32,
}
impl DoubleSingle {
    fn product(a: f32, b: f32) -> Self {
        let hi = a * b;
        let lo = a.mul_add(b, -hi);
        Self { hi, lo }
    }
    fn add(self, rhs: Self) -> Self {
        let sum = self.hi + rhs.hi;
        let rhs_virtual = sum - self.hi;
        let sum_error = (self.hi - (sum - rhs_virtual)) + (rhs.hi - rhs_virtual);
        let tail = (self.lo + rhs.lo) + sum_error;
        let hi = sum + tail;
        let lo = tail - (hi - sum);
        Self { hi, lo }
    }
}
fn emulate_bits8_double_single(payload: &[u8], x: &[f32]) -> Vec<f32> {
    let h = parse_pq_header(payload).expect("header");
    let (cb_bytes, codes) = pq_sections(payload).expect("sections");
    let codebooks: Vec<f32> = cb_bytes
        .chunks_exact(2)
        .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        .collect();
    let mut out = vec![0.0f32; h.rows as usize];
    for row in 0..h.rows as usize {
        let mut lanes = [DoubleSingle::default(); 32];
        for (lane, lane_sum) in lanes.iter_mut().enumerate() {
            for s in 0..h.s as usize {
                for chunk in (lane..h.nchunk as usize).step_by(32) {
                    let flat = (row * h.nchunk as usize + chunk) * h.s as usize + s;
                    let code = codes[flat] as usize;
                    let cb_base = (s * h.card as usize + code) * h.sub as usize;
                    let x_base = chunk * h.d as usize + s * h.sub as usize;
                    for j in 0..h.sub as usize {
                        *lane_sum = lane_sum
                            .add(DoubleSingle::product(codebooks[cb_base + j], x[x_base + j]));
                    }
                }
            }
        }
        for width in [16usize, 8, 4, 2, 1] {
            for lane in 0..width {
                lanes[lane] = lanes[lane].add(lanes[lane + width]);
            }
        }
        out[row] = lanes[0].hi + lanes[0].lo;
    }
    out
}
#[test]
fn bits8_variants_are_deterministic_and_pass_v21_at_real_chunk_counts() {
    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping: no Metal device available: {e}");
            return;
        }
    };
    for (rows, nchunk) in [(3u32, 1u32), (37u32, 64u32), (37u32, 192u32)] {
        let payload = make_bits8_payload(rows, nchunk);
        let x = make_x(nchunk as usize * 32);
        let authority = pq_matvec_f64_authority(&payload, &x).expect("f64 authority");
        let double_single_cpu = emulate_bits8_double_single(&payload, &x);
        let matrix = PqMetalMatrix::from_payload(&ctx, &payload).expect("resident PQ matrix");
        let established_default = pq_matvec_metal(&ctx, &payload, &x).expect("default");
        for variant in PqMetalKernelVariant::ALL {
            let first = matrix
                .matvec(&ctx, variant, &x)
                .expect("candidate first run");
            let second = matrix.matvec(&ctx, variant, &x).expect("candidate repeat");
            assert!(
                first
                    .iter()
                    .zip(&second)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{variant} is not bit-stable for nchunk={nchunk}"
            );
            if variant == PqMetalKernelVariant::Generic {
                assert!(
                    first
                        .iter()
                        .zip(&established_default)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "explicit generic variant diverged from pq_matvec_metal default"
                );
            }
            if variant == PqMetalKernelVariant::Bits8DoubleSingle {
                if let Some((row, (&gpu, &cpu))) = first
                    .iter()
                    .zip(&double_single_cpu)
                    .enumerate()
                    .find(|(_, (gpu, cpu))| gpu.to_bits() != cpu.to_bits())
                {
                    panic!(
                        "double-single Metal output diverged from the explicit CPU tree \
                         for nchunk={nchunk}, row={row}: gpu={gpu:.9e}/0x{:08x}, \
                         cpu={cpu:.9e}/0x{:08x}",
                        gpu.to_bits(),
                        cpu.to_bits(),
                    );
                }
            }
            let score = score_against_f64(
                &first,
                &authority,
                &Bounds::continuous_only(),
                variant.as_str(),
            );
            assert!(
                score.pass,
                "{variant} failed Numeric Parity V2.1 for nchunk={nchunk}: {:?}",
                score.failures
            );
        }
    }
}
