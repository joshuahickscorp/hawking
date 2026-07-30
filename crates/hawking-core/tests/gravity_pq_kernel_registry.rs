use half::f16;
use hawking_core::gravity::{
    parse_pq_header, pq_matvec, pq_matvec_f64_authority, pq_sections, PqHeader,
    PqMetalKernelVariant,
};
use hawking_core::metal::SHADER_GRAVITY_PQ;
use hawking_core::numeric_parity::{score_against_f64, ulp_distance_f32, Bounds};
fn primary_header(bits: u16) -> PqHeader {
    PqHeader {
        d: 32,
        s: 1,
        sub: 32,
        card: if bits == 8 { 256 } else { 128 },
        rows: 17,
        cols: 192,
        nchunk: 6,
        seed: 0,
        bits,
        rotate: 0,
        n_codebooks: 1,
    }
}
fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn tiny_bits8_payload() -> (Vec<u8>, Vec<f32>) {
    let h = primary_header(8);
    let mut out = Vec::new();
    out.extend_from_slice(b"GLM52CPK");
    push_u16(&mut out, h.d);
    push_u16(&mut out, h.s);
    push_u16(&mut out, h.sub);
    push_u16(&mut out, h.card);
    push_u32(&mut out, h.rows);
    push_u32(&mut out, h.cols);
    push_u32(&mut out, h.nchunk);
    push_u32(&mut out, h.seed);
    push_u16(&mut out, h.bits);
    out.push(h.rotate);
    out.push(h.n_codebooks);
    out.resize(64, 0);
    for code in 0..h.card as usize {
        for j in 0..h.sub as usize {
            let v = (((code * 13 + j * 7) % 257) as f32 - 128.0) / 64.0;
            push_u16(&mut out, f16::from_f32(v).to_bits());
        }
    }
    for row in 0..h.rows as usize {
        for chunk in 0..h.nchunk as usize {
            out.push(((row * 37 + chunk * 19 + row * chunk) & 255) as u8);
        }
    }
    let x = (0..h.cols as usize)
        .map(|i| ((i as f32 + 0.25) * 0.03125).sin() + (i % 11) as f32 * 0.0078125)
        .collect();
    (out, x)
}
fn autotune_payload(rows: u32, cols: u32) -> (Vec<u8>, Vec<f32>) {
    let nchunk = cols / 32;
    let mut out = Vec::with_capacity(64 + 256usize * 32 * 2 + rows as usize * nchunk as usize);
    out.extend_from_slice(b"GLM52CPK");
    push_u16(&mut out, 32);
    push_u16(&mut out, 1);
    push_u16(&mut out, 32);
    push_u16(&mut out, 256);
    push_u32(&mut out, rows);
    push_u32(&mut out, cols);
    push_u32(&mut out, nchunk);
    push_u32(&mut out, 0xA17E_2026);
    push_u16(&mut out, 8);
    out.push(0);
    out.push(1);
    out.resize(64, 0);
    for code in 0..256usize {
        for j in 0..32usize {
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
    let x = (0..cols as usize)
        .map(|i| {
            let a = ((i as f32 + 0.5) * 0.017578125).sin();
            let b = ((i * 31 % 127) as f32 - 63.0) / 256.0;
            a + b
        })
        .collect();
    (out, x)
}
fn simd_sum_tree(mut lanes: [f32; 32]) -> f32 {
    for width in [16usize, 8, 4, 2, 1] {
        for lane in 0..width {
            lanes[lane] += lanes[lane + width];
        }
    }
    lanes[0]
}
fn decode_codebooks(payload: &[u8]) -> (PqHeader, Vec<f32>, Vec<u8>) {
    let h = parse_pq_header(payload).expect("header");
    let (cb, codes) = pq_sections(payload).expect("sections");
    let codebooks = cb
        .chunks_exact(2)
        .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
        .collect();
    (h, codebooks, codes.to_vec())
}
fn emulate_bits8_generic(payload: &[u8], x: &[f32]) -> Vec<f32> {
    let (h, cb, codes) = decode_codebooks(payload);
    let mut out = vec![0.0f32; h.rows as usize];
    for row in 0..h.rows as usize {
        let mut lanes = [0.0f32; 32];
        for lane in 0..32usize {
            let mut acc = 0.0f32;
            for chunk in (lane..h.nchunk as usize).step_by(32) {
                let code = codes[row * h.nchunk as usize + chunk] as usize;
                let cb_base = code * 32;
                let x_base = chunk * 32;
                for j in 0..32usize {
                    acc = cb[cb_base + j].mul_add(x[x_base + j], acc);
                }
            }
            lanes[lane] = acc;
        }
        out[row] = simd_sum_tree(lanes);
    }
    out
}
fn emulate_bits8_vec4(payload: &[u8], x: &[f32]) -> Vec<f32> {
    let (h, cb, codes) = decode_codebooks(payload);
    let mut out = vec![0.0f32; h.rows as usize];
    for row in 0..h.rows as usize {
        let mut lanes = [0.0f32; 32];
        for lane in 0..32usize {
            let mut acc = [[0.0f32; 4]; 4];
            for chunk in (lane..h.nchunk as usize).step_by(32) {
                let code = codes[row * h.nchunk as usize + chunk] as usize;
                let cb_base = code * 32;
                let x_base = chunk * 32;
                for q in 0..8usize {
                    let which = q % 4;
                    for component in 0..4usize {
                        let j = q * 4 + component;
                        acc[which][component] =
                            cb[cb_base + j].mul_add(x[x_base + j], acc[which][component]);
                    }
                }
            }
            let mut v = [0.0f32; 4];
            for component in 0..4usize {
                v[component] = (acc[0][component] + acc[1][component])
                    + (acc[2][component] + acc[3][component]);
            }
            lanes[lane] = (v[0] + v[1]) + (v[2] + v[3]);
        }
        out[row] = simd_sum_tree(lanes);
    }
    out
}
fn emulate_bits8_kahan(payload: &[u8], x: &[f32]) -> Vec<f32> {
    let (h, cb, codes) = decode_codebooks(payload);
    let mut out = vec![0.0f32; h.rows as usize];
    for row in 0..h.rows as usize {
        let mut lanes = [0.0f32; 32];
        for lane in 0..32usize {
            let mut sum = 0.0f32;
            let mut correction = 0.0f32;
            for chunk in (lane..h.nchunk as usize).step_by(32) {
                let code = codes[row * h.nchunk as usize + chunk] as usize;
                for j in 0..32usize {
                    let product = cb[code * 32 + j] * x[chunk * 32 + j];
                    let corrected = product - correction;
                    let next = sum + corrected;
                    correction = (next - sum) - corrected;
                    sum = next;
                }
            }
            lanes[lane] = sum;
        }
        out[row] = simd_sum_tree(lanes);
    }
    out
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
        let tail = self.lo + rhs.lo + sum_error;
        let hi = sum + tail;
        let lo = tail - (hi - sum);
        Self { hi, lo }
    }
    fn value(self) -> f32 {
        self.hi + self.lo
    }
}
fn emulate_bits8_double_single(payload: &[u8], x: &[f32]) -> Vec<f32> {
    let (h, cb, codes) = decode_codebooks(payload);
    let mut out = vec![0.0f32; h.rows as usize];
    for row in 0..h.rows as usize {
        let mut lanes = [DoubleSingle::default(); 32];
        for (lane, lane_sum) in lanes.iter_mut().enumerate() {
            for chunk in (lane..h.nchunk as usize).step_by(32) {
                let code = codes[row * h.nchunk as usize + chunk] as usize;
                for j in 0..32usize {
                    *lane_sum =
                        lane_sum.add(DoubleSingle::product(cb[code * 32 + j], x[chunk * 32 + j]));
                }
            }
        }
        for width in [16usize, 8, 4, 2, 1] {
            for lane in 0..width {
                lanes[lane] = lanes[lane].add(lanes[lane + width]);
            }
        }
        out[row] = lanes[0].value();
    }
    out
}
fn worst_meaningful_row(candidate: &[f32], reference: &[f64], cutoff: f64) -> (usize, f64, f64) {
    candidate
        .iter()
        .zip(reference)
        .enumerate()
        .filter(|(_, (_, reference))| reference.abs() >= cutoff)
        .map(|(row, (&candidate, &reference))| {
            let abs = (candidate as f64 - reference).abs();
            (row, abs, abs / reference.abs())
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .expect("at least one meaningful row")
}
fn row_condition(payload: &[u8], x: &[f32], row: usize, reference: f64) -> f64 {
    let (h, cb, codes) = decode_codebooks(payload);
    let mut sum_abs = 0.0f64;
    for chunk in 0..h.nchunk as usize {
        let code = codes[row * h.nchunk as usize + chunk] as usize;
        for j in 0..32usize {
            sum_abs += ((cb[code * 32 + j] as f64) * (x[chunk * 32 + j] as f64)).abs();
        }
    }
    sum_abs / reference.abs()
}
#[test]
fn registry_is_explicit_unique_and_keeps_generic_first() {
    assert_eq!(PqMetalKernelVariant::ALL[0], PqMetalKernelVariant::Generic);
    assert_eq!(
        PqMetalKernelVariant::Generic.kernel_name(),
        "gravity_pq_matvec"
    );
    assert_eq!(
        PqMetalKernelVariant::Bits8DoubleSingle.kernel_name(),
        "gravity_pq_matvec_bits8_double_single"
    );
    assert_eq!(
        PqMetalKernelVariant::Bits8DoubleSingle.dispatches_per_matvec(),
        1
    );
    assert_eq!(PqMetalKernelVariant::Bits8DoubleSingle.split_count(), None);
    let names: std::collections::HashSet<_> = PqMetalKernelVariant::ALL
        .iter()
        .map(|v| v.as_str())
        .collect();
    assert_eq!(names.len(), PqMetalKernelVariant::ALL.len());
    for variant in PqMetalKernelVariant::ALL {
        assert_eq!(
            variant.as_str().parse::<PqMetalKernelVariant>().unwrap(),
            variant
        );
    }
}
#[test]
fn primary_bits8_geometry_admits_all_candidates_but_packed_bits_do_not() {
    let bits8 = primary_header(8);
    assert!(PqMetalKernelVariant::ALL.iter().all(|v| v.supports(&bits8)));
    let bits7 = primary_header(7);
    assert!(PqMetalKernelVariant::Generic.supports(&bits7));
    assert!(PqMetalKernelVariant::ALL[1..]
        .iter()
        .all(|v| !v.supports(&bits7)));
}
#[test]
fn shader_registers_direct_double_single_vector_and_deterministic_reductions() {
    for symbol in [
        "kernel void gravity_pq_matvec_bits8_direct",
        "kernel void gravity_pq_matvec_bits8_double_single",
        "kernel void gravity_pq_matvec_bits8_vec4",
        "kernel void gravity_pq_matvec_bits8_2d",
        "kernel void gravity_pq_reduce_2d",
    ] {
        assert!(SHADER_GRAVITY_PQ.contains(symbol), "missing {symbol}");
    }
    assert!(SHADER_GRAVITY_PQ.contains("uint(codes[flat])"));
    assert!(SHADER_GRAVITY_PQ.contains("out.lo = metal::precise::fma(a, b, -hi)"));
    assert!(SHADER_GRAVITY_PQ.contains("acc = pq_ds_simd_tree(acc, lane)"));
    for sequenced in [
        "volatile float hi",
        "volatile float sum",
        "volatile float sum_error",
        "volatile float tail",
        "volatile float hi_delta",
    ] {
        assert!(
            SHADER_GRAVITY_PQ.contains(sequenced),
            "double-single error-free transform lost {sequenced}"
        );
    }
    for width in [16, 8, 4, 2, 1] {
        assert!(
            SHADER_GRAVITY_PQ.contains(&format!("if (lane < {width}u)")),
            "double-single reduction is missing width {width}"
        );
    }
    let double_single_kernel = SHADER_GRAVITY_PQ
        .split("kernel void gravity_pq_matvec_bits8_double_single")
        .nth(1)
        .expect("double-single kernel body")
        .split("kernel void gravity_pq_matvec_bits8_vec4")
        .next()
        .expect("double-single kernel terminator");
    assert!(
        !double_single_kernel.contains("simd_sum("),
        "double-single candidate must use its explicit compensated tree"
    );
    assert!(SHADER_GRAVITY_PQ.contains("partials[row * splits + split] = acc"));
    assert!(SHADER_GRAVITY_PQ.contains("for (uint split = 0u; split < splits; ++split)"));
    assert!(
        !SHADER_GRAVITY_PQ.contains("atomic_fetch_add"),
        "2D reduction must not use nondeterministic atomics"
    );
}
#[test]
fn pq_fp64_authority_scores_host_candidate_under_v21() {
    let (payload, x) = tiny_bits8_payload();
    let h = parse_pq_header(&payload).expect("header");
    assert_eq!((h.d, h.s, h.sub, h.card, h.bits), (32, 1, 32, 256, 8));
    let host = pq_matvec(&payload, &x).expect("host f32");
    let authority = pq_matvec_f64_authority(&payload, &x).expect("f64 authority");
    let score = score_against_f64(&host, &authority, &Bounds::continuous_only(), "host-f32");
    assert!(score.pass, "V2.1 host score failed: {:?}", score.failures);
}
#[test]
#[ignore = "exact-geometry CPU diagnostic; run explicitly with --ignored --nocapture"]
fn exact_geometry_autotune_payload_failure_preflight() {
    let bounds = Bounds::continuous_only();
    let mut geometry_reports = Vec::new();
    for (rows, cols) in [(2048u32, 6144u32), (6144u32, 2048u32)] {
        let (payload, x) = autotune_payload(rows, cols);
        let authority = pq_matvec_f64_authority(&payload, &x).expect("f64 authority");
        let host = pq_matvec(&payload, &x).expect("host left-to-right");
        let mut candidates = Vec::new();
        for (name, candidate, expected_pass) in [
            ("host-left-to-right", host, false),
            (
                "generic-emulated",
                emulate_bits8_generic(&payload, &x),
                false,
            ),
            ("vec4-emulated", emulate_bits8_vec4(&payload, &x), false),
            (
                "kahan-lanes-emulated",
                emulate_bits8_kahan(&payload, &x),
                false,
            ),
            (
                "double-single-emulated",
                emulate_bits8_double_single(&payload, &x),
                true,
            ),
        ] {
            let score = score_against_f64(&candidate, &authority, &bounds, name);
            let (row, abs_error, meaningful_rel) =
                worst_meaningful_row(&candidate, &authority, score.continuous.abs_error_cutoff);
            let condition = row_condition(&payload, &x, row, authority[row]);
            assert!(
                score.continuous.relative_l2 < bounds.max_relative_l2,
                "headline L2 is not the failure mechanism"
            );
            assert_eq!(
                score.pass, expected_pass,
                "unexpected V2.1 result for {name}: {:?}",
                score.failures
            );
            if !expected_pass {
                assert_eq!(
                    score.failures.len(),
                    1,
                    "meaningful_rel must be the sole failed V2.1 bound: {:?}",
                    score.failures
                );
                assert!(
                    score
                        .failures
                        .iter()
                        .any(|failure| failure.starts_with("meaningful_rel")),
                    "expected exact-geometry cancellation to trip meaningful_rel: {:?}",
                    score.failures
                );
            }
            assert!(score.discrete.greedy_match && score.discrete.top_k_exact_match);
            candidates.push(serde_json::json!({
                "model": name,
                "expected_pass": expected_pass,
                "score": score,
                "worst_meaningful_row": {
                    "row": row,
                    "reference": authority[row],
                    "candidate": candidate[row],
                    "absolute_error": abs_error,
                    "meaningful_relative_error": meaningful_rel,
                    "ulp_to_reference_rounded_f32": ulp_distance_f32(
                        candidate[row],
                        authority[row] as f32,
                    ),
                    "sum_abs_products_over_abs_reference": condition,
                },
            }));
        }
        geometry_reports.push(serde_json::json!({
            "rows": rows,
            "cols": cols,
            "nchunk": cols / 32,
            "candidates": candidates,
        }));
    }
    let receipt = serde_json::json!({
        "schema": "hawking.gravity.pq_cpu_preflight.v1",
        "status": "DIAGNOSTIC_COMPLETE",
        "execution": "CPU_ONLY_NO_METAL",
        "payload": "deterministic exact-geometry autotune payload; D=32,S=1,sub=32,card=256,bits=8",
        "numeric_contract": "Numeric Parity V2.1 unchanged",
        "bounds": bounds,
        "reduction_model": {
            "host-left-to-right": "actual PqTensor f32 left-to-right accumulation",
            "generic-emulated": "per-lane fma plus a fixed binary-tree CPU model of simd_sum",
            "vec4-emulated": "four float4-equivalent fma accumulators plus a fixed binary-tree CPU model of simd_sum",
            "kahan-lanes-emulated": "per-lane Kahan with ordinary f32 product, then the fixed binary-tree reduction model",
            "double-single-emulated": "FMA product residual plus error-free hi/lo summation through the fixed binary-tree reduction model",
            "metal_equivalence_caveat": "The generic/vec4/Kahan/double-single results model an explicit 32-lane binary reduction tree. They are CPU emulations, not measured Metal and not a claim that simd_sum is bit-equivalent to this tree.",
        },
        "finding": {
            "failure_gate": "max_meaningful_rel",
            "cause": "highly conditioned cancellation rows remain above the V2.1 near-zero cutoff",
            "ordinary_kahan_sufficient": false,
            "double_single_only_green_model": true,
            "next_numeric_candidate": "additive Metal bits8 double-single accumulator using two-product (mul + fma residual), error-free hi/lo addition, and deterministic hi/lo lane reduction",
            "promotion": "NONE",
            "expected_throughput_cost": "material and unmeasured: approximately 1 mul + 1 fma + ~11 add/sub operations per accumulated product instead of 1 fma, plus two-register hi/lo state and compensated lane reduction",
        },
        "geometries": geometry_reports,
    });
    let text = serde_json::to_string_pretty(&receipt).expect("serialize receipt");
    if let Some(path) = std::env::var_os("HAWKING_PQ_CPU_PREFLIGHT_OUT") {
        use std::io::Write;
        let path = std::path::PathBuf::from(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create receipt directory");
        }
        let mut file = std::fs::File::create(&path).expect("create CPU preflight receipt");
        file.write_all(text.as_bytes()).expect("write receipt");
        file.write_all(b"\n").expect("terminate receipt");
        file.sync_all().expect("sync receipt");
    }
}
