//! Numeric Parity Contract **V2.1** — condition-aware hybrid metrics.
//!
//! Canonical policy: root `NUMERIC_PARITY_V2_1.md` (user-authorized 2026-07-26).
//!
//! ## Why this module exists
//!
//! V2 judged `lm_head` with a single scalar — max relative error across the
//! logit vector — and rejected device bf16 at 3.23e-3 against a 1e-5 bound.
//! That vector spanned ~1e-32 … ~5e18. Relative error on a denormal-scale
//! element is not a measurement; well-scaled elements agreed to ~1.4e-7
//! (reduction-order divergence). The gate was reclassified
//! `NUMERIC_GATE_INSUFFICIENTLY_CONDITIONED`.
//!
//! ## Authority
//!
//! Score **both** f32 backends (host and device) against an **FP64 reference
//! computation**, not against each other. The host f32 path has its own
//! accumulation order; making it the oracle (including via
//! `host.iter().map(|&v| v as f64)`) bakes its rounding into the contract and
//! is **banned**.
//!
//! ## Metrics (applied together)
//!
//! | Metric | Role |
//! |---|---|
//! | absolute error near zero | hard — relative error uninformative when \|ref\| is tiny |
//! | **relative L2 (full vector)** | **hard headline** continuous agreement |
//! | cosine similarity | hard — direction / full-vector shape |
//! | KL on softmax | hard when `require_kl` (logits) |
//! | exact top-k + greedy argmax | **hard, no tolerance, ever** |
//! | max meaningful-scale relative | hard op-local; **diagnostic** on full multi-layer forward |
//! | ULP distribution (median, p95, p99, max) | diagnostic distribution, never a lone max gate |
//!
//! See root `NUMERIC_PARITY_V2_1.md` for the full-forward vs op-local split.
//! A large ULP / max_meaningful_rel tail on a full-forward **pass** is still
//! reported — information, not a silent failure.

use serde::Serialize;

/// Schema id for V2.1 receipts and logs.
pub const SCHEMA: &str = "hawking.numeric_parity.v2_1";

/// Reclassification of the insufficiently conditioned V2 `lm_head` gate.
pub const RECLASSIFICATION: &str = "NUMERIC_GATE_INSUFFICIENTLY_CONDITIONED";

// ── cutoffs & bounds ───────────────────────────────────────────────────────

/// Data-derived absolute-error regime cutoff.
///
/// Relative error is uninformative when `|r|` is tiny relative to the bulk of
/// the vector (the denormal-scale pathology that broke V2 on `lm_head`).
///
/// ```text
/// cutoff = max(1e-30, ABS_REGIME_FRACTION * median(|r| over finite elements))
/// ```
///
/// Every report states the cutoff that was used. Elements with `|r| < cutoff`
/// are scored with absolute error; elements with `|r| ≥ cutoff` contribute to
/// meaningful-scale relative statistics.
pub const ABS_REGIME_FRACTION: f64 = 1e-6;

/// Absolute floor so an all-zero or empty reference still has a defined regime.
pub const ABS_REGIME_FLOOR: f64 = 1e-30;

/// Default continuous bounds for f32 reduction vs f64 authority on **op-local**
/// matvecs / elementwise kernels (K ~ hundreds to thousands of accumulate steps).
///
/// Relative L2 ~1e-5 is the headline gate: well above pure f32 rounding of a
/// single mul (~1e-7) and below a decision-changing drift on well-conditioned
/// logits. Meaningful-scale **max** relative uses the same 1e-5 that V2 intended
/// — but **only** on elements that survive the absolute-error cutoff, and only
/// as a hard gate when [`Bounds::gate_max_meaningful_rel`] is true (op-local
/// defaults). Full multi-layer forwards use [`Bounds::full_forward_logits`],
/// which reports max_meaningful_rel as diagnostic only — see
/// `NUMERIC_PARITY_V2_1.md`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Bounds {
    /// Max |cand − ref| allowed for |ref| < abs_cutoff.
    pub max_abs_near_zero: f64,
    /// Headline: ‖cand − ref‖₂ / ‖ref‖₂ over the complete vector.
    pub max_relative_l2: f64,
    /// Min cosine similarity (direction).
    pub min_cosine: f64,
    /// Max KL(softmax(ref) ‖ softmax(cand)) in nats; used when `require_kl`.
    pub max_kl: f64,
    /// When true, KL is a hard gate (logit / distribution vectors).
    pub require_kl: bool,
    /// Exact top-k size (indices, ordered by descending value).
    pub top_k: usize,
    /// Max relative error on elements with |ref| ≥ abs_cutoff.
    /// This is a **max over single elements**, not a mean — see
    /// [`ContinuousMetrics::max_meaningful_rel`].
    pub max_meaningful_rel: f64,
    /// When true, `max_meaningful_rel` is a hard gate. When false it is still
    /// computed and printed but never fails the score. Op-local defaults keep
    /// this true; multi-layer full-forward scoring turns it off because the
    /// host f32 path itself exceeds 1e-5 max relative against a true f64
    /// forward while rel_l2 / cos / KL / discrete stay clean.
    pub gate_max_meaningful_rel: bool,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            max_abs_near_zero: 1e-4,
            max_relative_l2: 1e-5,
            min_cosine: 1.0 - 1e-7,
            max_kl: 1e-6,
            require_kl: true,
            top_k: 5,
            max_meaningful_rel: 1e-5,
            gate_max_meaningful_rel: true,
        }
    }
}

impl Bounds {
    /// Continuous arithmetic only (no KL). Discrete top-k / argmax still exact.
    pub fn continuous_only() -> Self {
        Self {
            require_kl: false,
            ..Self::default()
        }
    }

    /// Logit / lm_head **op-local** defaults (KL on softmax + top-5 + greedy).
    /// Max meaningful relative remains a hard gate — appropriate for a single
    /// matvec / silu scored against f64, not for a multi-layer fixture forward.
    pub fn logits() -> Self {
        Self::default()
    }

    /// Full multi-layer fixture logits vs an independent f64 forward.
    ///
    /// Hard continuous gates: relative L2, cosine, KL, abs-near-zero.
    /// Hard discrete gates: greedy argmax, top-k (no tolerance).
    /// Diagnostic only: max_meaningful_rel, ULP distribution,
    /// diagnostic_max_scalar_rel_all.
    pub fn full_forward_logits() -> Self {
        Self {
            gate_max_meaningful_rel: false,
            ..Self::default()
        }
    }
}

/// Derive the absolute-error regime cutoff from the FP64 reference.
pub fn absolute_error_cutoff(reference: &[f64]) -> f64 {
    let mut abs_vals: Vec<f64> = reference
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(f64::abs)
        .collect();
    if abs_vals.is_empty() {
        return ABS_REGIME_FLOOR;
    }
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = abs_vals.len() / 2;
    let median = if abs_vals.len() % 2 == 0 && abs_vals.len() >= 2 {
        0.5 * (abs_vals[mid - 1] + abs_vals[mid])
    } else {
        abs_vals[mid]
    };
    (ABS_REGIME_FRACTION * median).max(ABS_REGIME_FLOOR)
}

// ── metric structs ─────────────────────────────────────────────────────────

/// ULP distance distribution. Always report the quartet — never a lone max.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct UlpDistribution {
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub n: usize,
}

/// Continuous metrics of one candidate against the FP64 reference.
#[derive(Debug, Clone, Serialize)]
pub struct ContinuousMetrics {
    /// Absolute-error regime cutoff used for this comparison (data-derived).
    pub abs_error_cutoff: f64,
    /// How the cutoff was defined (for the report).
    pub abs_error_cutoff_rule: &'static str,
    pub n_near_zero: usize,
    pub n_meaningful: usize,
    /// Max |cand − ref| on the near-zero subset.
    pub max_abs_near_zero: f64,
    /// Max |cand − ref| / |ref| on the meaningful-scale subset.
    ///
    /// This is a **maximum over single elements**, not a mean / average.
    /// Logs and receipts must label it `max_meaningful_rel` (or
    /// `meaningful_rel`), never `mean_rel`.
    pub max_meaningful_rel: f64,
    /// Headline full-vector relative L2.
    pub relative_l2: f64,
    pub ulp: UlpDistribution,
    pub cosine_similarity: f64,
    /// KL(softmax(ref) ‖ softmax(cand)); None if disabled or empty.
    pub kl_divergence: Option<f64>,
    /// Diagnostic: max scalar relative error over the **whole** vector (the
    /// V2 metric). Present so a re-run can show why V2 misfired; **not** a gate.
    pub diagnostic_max_scalar_rel_all: f64,
}

/// Discrete decisions — exact agreement, no tolerance.
#[derive(Debug, Clone, Serialize)]
pub struct DiscreteMetrics {
    pub greedy_argmax_ref: Option<usize>,
    pub greedy_argmax_cand: Option<usize>,
    pub greedy_match: bool,
    pub top_k: usize,
    pub top_k_ref: Vec<usize>,
    pub top_k_cand: Vec<usize>,
    pub top_k_exact_match: bool,
}

/// One backend scored against the FP64 authority.
#[derive(Debug, Clone, Serialize)]
pub struct BackendScore {
    pub backend: String,
    pub continuous: ContinuousMetrics,
    pub discrete: DiscreteMetrics,
    pub pass: bool,
    pub failures: Vec<String>,
}

/// Paired score: host and device against the same FP64 reference.
#[derive(Debug, Clone, Serialize)]
pub struct PairedScore {
    pub schema: &'static str,
    pub abs_error_cutoff: f64,
    pub bounds: Bounds,
    pub host: BackendScore,
    pub device: BackendScore,
    /// Both backends pass every applicable gate.
    pub pass: bool,
}

// ── leaf metrics ───────────────────────────────────────────────────────────

/// Ordered integer mapping so ULP distance is well-defined across sign.
fn float_to_ordered_bits(x: f32) -> i32 {
    let bits = x.to_bits() as i32;
    if bits < 0 {
        bits ^ i32::MAX
    } else {
        bits
    }
}

/// ULP distance between two finite f32 values. Non-finite → `u64::MAX`.
pub fn ulp_distance_f32(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return u64::MAX;
    }
    if a == b {
        return 0;
    }
    let ai = float_to_ordered_bits(a);
    let bi = float_to_ordered_bits(b);
    (ai as i64 - bi as i64).unsigned_abs()
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let p = p.clamp(0.0, 1.0);
    // Nearest-rank, 0-based.
    let idx = ((p * (sorted.len() as f64 - 1.0)).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// ULP distribution of `candidate` vs `reference` cast elementwise to f32.
///
/// The authority is f64; the comparison asks how many f32 ULPs the candidate
/// sits from the reference **rounded to f32** — the representation both
/// backends actually emit.
pub fn ulp_distribution(candidate: &[f32], reference: &[f64]) -> UlpDistribution {
    assert_eq!(candidate.len(), reference.len());
    let mut ulps: Vec<f64> = candidate
        .iter()
        .zip(reference.iter())
        .map(|(&c, &r)| {
            let r32 = r as f32;
            let d = ulp_distance_f32(c, r32);
            if d == u64::MAX {
                f64::INFINITY
            } else {
                d as f64
            }
        })
        .collect();
    let n = ulps.len();
    if n == 0 {
        return UlpDistribution {
            median: 0.0,
            p95: 0.0,
            p99: 0.0,
            max: 0.0,
            n: 0,
        };
    }
    ulps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    UlpDistribution {
        median: percentile_sorted(&ulps, 0.50),
        p95: percentile_sorted(&ulps, 0.95),
        p99: percentile_sorted(&ulps, 0.99),
        max: ulps[n - 1],
        n,
    }
}

/// Relative L2: ‖c − r‖₂ / ‖r‖₂. Empty or zero-ref → 0 if identical, else +∞.
pub fn relative_l2(candidate: &[f32], reference: &[f64]) -> f64 {
    assert_eq!(candidate.len(), reference.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&c, &r) in candidate.iter().zip(reference.iter()) {
        let d = c as f64 - r;
        num += d * d;
        den += r * r;
    }
    if den == 0.0 {
        return if num == 0.0 { 0.0 } else { f64::INFINITY };
    }
    num.sqrt() / den.sqrt()
}

/// Cosine similarity between candidate and reference (f64 arithmetic).
pub fn cosine_similarity(candidate: &[f32], reference: &[f64]) -> f64 {
    assert_eq!(candidate.len(), reference.len());
    if candidate.is_empty() {
        return 1.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&c, &r) in candidate.iter().zip(reference.iter()) {
        let a = c as f64;
        dot += a * r;
        na += a * a;
        nb += r * r;
    }
    if na == 0.0 || nb == 0.0 {
        return if na == nb { 1.0 } else { 0.0 };
    }
    (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0)
}

/// Softmax in f64 (stable).
pub fn softmax_f64(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let m = xs
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        let n = xs.len() as f64;
        return vec![1.0 / n; xs.len()];
    }
    let mut out: Vec<f64> = xs.iter().map(|&v| (v - m).exp()).collect();
    let s: f64 = out.iter().sum();
    if s > 0.0 {
        for v in &mut out {
            *v /= s;
        }
    }
    out
}

/// KL(p ‖ q) = Σ p log(p/q). Both must be probability distributions.
/// Clamps zeros with a tiny floor so isolated underflows do not explode.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    const FLOOR: f64 = 1e-30;
    let mut kl = 0.0f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let pi = pi.max(0.0);
        if pi == 0.0 {
            continue;
        }
        let qi = qi.max(FLOOR);
        kl += pi * (pi / qi).ln();
    }
    kl
}

/// KL on softmax of reference (f64) vs softmax of candidate (promoted).
pub fn kl_softmax_logits(candidate: &[f32], reference: &[f64]) -> f64 {
    let p = softmax_f64(reference);
    let q_src: Vec<f64> = candidate.iter().map(|&c| c as f64).collect();
    let q = softmax_f64(&q_src);
    kl_divergence(&p, &q)
}

/// Greedy argmax index (left-to-right; strictly greater wins, earlier index on ties).
pub fn greedy_argmax_f64(xs: &[f64]) -> Option<usize> {
    if xs.is_empty() {
        return None;
    }
    let mut best_i = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    Some(best_i)
}

/// Greedy argmax for f32 (same left-to-right strict-greater rule).
pub fn greedy_argmax_f32(xs: &[f32]) -> Option<usize> {
    if xs.is_empty() {
        return None;
    }
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    Some(best_i)
}

/// Top-k indices ordered by descending value (stable on ties: lower index wins).
pub fn top_k_indices_f64(xs: &[f64], k: usize) -> Vec<usize> {
    if xs.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(xs.len());
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| {
        xs[b]
            .partial_cmp(&xs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

pub fn top_k_indices_f32(xs: &[f32], k: usize) -> Vec<usize> {
    if xs.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(xs.len());
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| {
        xs[b]
            .partial_cmp(&xs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

// ── FP64 authority elementwise (SiLU ⊙) ────────────────────────────────────

/// FP64 authority for SwiGLU intermediate: `silu(g) * u` with
/// `silu(g) = g / (1 + exp(-g))` in f64.
///
/// Metal `exp` and libm `expf` differ in the last ULP; V2.1 scores both f32
/// backends against this authority rather than requiring bit-identity between
/// them. Continuous drift is allowed; discrete decisions (router top-k,
/// selected experts, greedy token) stay exact elsewhere.
pub fn silu_mul_f64_authority(gate: &[f64], up: &[f64]) -> Result<Vec<f64>, String> {
    if gate.len() != up.len() {
        return Err(format!(
            "silu_mul_f64_authority: gate.len() {} != up.len() {}",
            gate.len(),
            up.len()
        ));
    }
    Ok(gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect())
}

/// Host f32 SiLU⊙ (libm `expf` via `f32::exp`). Same formula as the Metal
/// kernel `gravity_silu_mul_f32` and the resident host oracle.
pub fn silu_mul_f32_host(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
    if gate.len() != up.len() {
        return Err(format!(
            "silu_mul_f32_host: gate.len() {} != up.len() {}",
            gate.len(),
            up.len()
        ));
    }
    Ok(gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect())
}

// ── FP64 authority matvec ──────────────────────────────────────────────────

/// Widen little-endian bf16 payload to f64 (same bit pattern as host widen,
/// promoted once — not widen→f32→f64, which would bake f32 rounding into the
/// authority).
pub fn widen_bf16_to_f64(weight_le: &[u8]) -> Result<Vec<f64>, String> {
    if weight_le.len() % 2 != 0 {
        return Err(format!(
            "bf16 payload length {} is not a multiple of 2",
            weight_le.len()
        ));
    }
    Ok(weight_le
        .chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            // bf16 → f32 bit-widen, then promote to f64. Exact for all finite
            // bf16 (they are a subset of f32).
            f32::from_bits((bits as u32) << 16) as f64
        })
        .collect())
}

/// FP64 authority matvec: Σ_j W\[i,j\] · x\[j\] in f64 left-to-right per row.
///
/// `weight_le` is little-endian bf16 row-major `[rows, cols]`. `x` is f32
/// activations (promoted per multiply). This is the neutral ground truth both
/// host-f32 and device-f32 backends are scored against.
pub fn matvec_bf16_f64_authority(
    weight_le: &[u8],
    cols: usize,
    x: &[f32],
) -> Result<Vec<f64>, String> {
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    matvec_bf16_f64_authority_x64(weight_le, cols, &x64)
}

/// Same as [`matvec_bf16_f64_authority`] but activations are already f64
/// (end-to-end f64 forward intermediates — not f32-cast then re-promoted).
pub fn matvec_bf16_f64_authority_x64(
    weight_le: &[u8],
    cols: usize,
    x: &[f64],
) -> Result<Vec<f64>, String> {
    if x.len() != cols {
        return Err(format!(
            "matvec_bf16_f64_authority_x64: x.len() {} != cols {cols}",
            x.len()
        ));
    }
    if cols == 0 || weight_le.len() % (cols * 2) != 0 {
        return Err(format!(
            "matvec_bf16_f64_authority_x64: payload {} B is not a whole number of {cols}-wide bf16 rows",
            weight_le.len()
        ));
    }
    let w = widen_bf16_to_f64(weight_le)?;
    let rows = w.len() / cols;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f64;
        for (j, &wj) in row.iter().enumerate() {
            acc += wj * x[j];
        }
        out.push(acc);
    }
    Ok(out)
}

/// FP64 authority matvec for a row-major f32 weight matrix (native.f32 /
/// already-widened native.bf16/f16 payload).
///
/// Weights are promoted once per multiply; activations stay in f64. Used by
/// the fixture f64 forward so host and device f32 logits are scored against
/// a real FP64 computation rather than against each other.
pub fn matvec_dense_f64_authority(
    weights: &[f32],
    cols: usize,
    x: &[f64],
) -> Result<Vec<f64>, String> {
    if cols == 0 {
        return Err("matvec_dense_f64_authority: cols must be > 0".into());
    }
    if x.len() != cols {
        return Err(format!(
            "matvec_dense_f64_authority: x.len() {} != cols {cols}",
            x.len()
        ));
    }
    if weights.len() % cols != 0 {
        return Err(format!(
            "matvec_dense_f64_authority: {} weights is not a whole number of {cols}-wide rows",
            weights.len()
        ));
    }
    let rows = weights.len() / cols;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &weights[r * cols..(r + 1) * cols];
        let mut acc = 0.0f64;
        for (j, &wj) in row.iter().enumerate() {
            acc += (wj as f64) * x[j];
        }
        out.push(acc);
    }
    Ok(out)
}

/// RMSNorm in f64: `x * rsqrt(mean(x²) + eps) * weight`.
pub fn rmsnorm_f64(x: &[f64], weight: &[f64], eps: f64) -> Result<Vec<f64>, String> {
    if x.len() != weight.len() {
        return Err(format!(
            "rmsnorm_f64: x.len() {} != weight.len() {}",
            x.len(),
            weight.len()
        ));
    }
    if x.is_empty() {
        return Ok(Vec::new());
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    Ok(x.iter()
        .zip(weight.iter())
        .map(|(&v, &w)| v * inv * w)
        .collect())
}

/// Affine LayerNorm in f64 (DSA indexer key path).
pub fn layernorm_f64(
    x: &[f64],
    weight: &[f64],
    bias: &[f64],
    eps: f64,
) -> Result<Vec<f64>, String> {
    if x.len() != weight.len() || x.len() != bias.len() {
        return Err(format!(
            "layernorm_f64: len mismatch x={} w={} b={}",
            x.len(),
            weight.len(),
            bias.len()
        ));
    }
    if x.is_empty() {
        return Ok(Vec::new());
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let inv = 1.0 / (var + eps).sqrt();
    Ok(x.iter()
        .zip(weight.iter())
        .zip(bias.iter())
        .map(|((&v, &w), &b)| (v - mean) * inv * w + b)
        .collect())
}

// ── scoring ────────────────────────────────────────────────────────────────

const ABS_CUTOFF_RULE: &str =
    "max(1e-30, 1e-6 * median(|reference| over finite elements)); near-zero uses abs error";

/// Score one f32 candidate against an FP64 reference under V2.1 bounds.
pub fn score_against_f64(
    candidate: &[f32],
    reference: &[f64],
    bounds: &Bounds,
    backend: &str,
) -> BackendScore {
    assert_eq!(
        candidate.len(),
        reference.len(),
        "score_against_f64: length mismatch ({} vs {})",
        candidate.len(),
        reference.len()
    );

    let cutoff = absolute_error_cutoff(reference);
    let mut n_near = 0usize;
    let mut n_mean = 0usize;
    let mut max_abs_near = 0.0f64;
    let mut max_mean_rel = 0.0f64;
    let mut max_scalar_rel_all = 0.0f64;

    for (&c, &r) in candidate.iter().zip(reference.iter()) {
        let d = (c as f64 - r).abs();
        let denom_all = r.abs().max((c as f64).abs()).max(f64::MIN_POSITIVE);
        max_scalar_rel_all = max_scalar_rel_all.max(d / denom_all);

        if !r.is_finite() {
            continue;
        }
        if r.abs() < cutoff {
            n_near += 1;
            max_abs_near = max_abs_near.max(d);
        } else {
            n_mean += 1;
            max_mean_rel = max_mean_rel.max(d / r.abs());
        }
    }

    let rel_l2 = relative_l2(candidate, reference);
    let cos = cosine_similarity(candidate, reference);
    let ulp = ulp_distribution(candidate, reference);
    let kl = if bounds.require_kl {
        Some(kl_softmax_logits(candidate, reference))
    } else {
        None
    };

    let continuous = ContinuousMetrics {
        abs_error_cutoff: cutoff,
        abs_error_cutoff_rule: ABS_CUTOFF_RULE,
        n_near_zero: n_near,
        n_meaningful: n_mean,
        max_abs_near_zero: max_abs_near,
        max_meaningful_rel: max_mean_rel,
        relative_l2: rel_l2,
        ulp,
        cosine_similarity: cos,
        kl_divergence: kl,
        diagnostic_max_scalar_rel_all: max_scalar_rel_all,
    };

    let arg_r = greedy_argmax_f64(reference);
    let arg_c = greedy_argmax_f32(candidate);
    let top_r = top_k_indices_f64(reference, bounds.top_k);
    let top_c = top_k_indices_f32(candidate, bounds.top_k);
    let discrete = DiscreteMetrics {
        greedy_argmax_ref: arg_r,
        greedy_argmax_cand: arg_c,
        greedy_match: arg_r == arg_c,
        top_k: bounds.top_k,
        top_k_exact_match: top_r == top_c,
        top_k_ref: top_r,
        top_k_cand: top_c,
    };

    let mut failures = Vec::new();
    if continuous.max_abs_near_zero > bounds.max_abs_near_zero && continuous.n_near_zero > 0 {
        failures.push(format!(
            "abs_near_zero {:.3e} > bound {:.3e} (cutoff={:.3e}, n={})",
            continuous.max_abs_near_zero, bounds.max_abs_near_zero, cutoff, continuous.n_near_zero
        ));
    }
    if continuous.relative_l2 > bounds.max_relative_l2 {
        failures.push(format!(
            "relative_l2 {:.3e} > bound {:.3e}",
            continuous.relative_l2, bounds.max_relative_l2
        ));
    }
    if continuous.cosine_similarity < bounds.min_cosine {
        failures.push(format!(
            "cosine {:.9} < bound {:.9}",
            continuous.cosine_similarity, bounds.min_cosine
        ));
    }
    if bounds.gate_max_meaningful_rel
        && continuous.max_meaningful_rel > bounds.max_meaningful_rel
        && continuous.n_meaningful > 0
    {
        // Prefix stays `meaningful_rel` so existing failure parsers
        // (`starts_with("meaningful_rel")`) keep working. The score-line
        // label is the unambiguous `max_meaningful_rel=`.
        failures.push(format!(
            "meaningful_rel {:.3e} > bound {:.3e} (n={})",
            continuous.max_meaningful_rel, bounds.max_meaningful_rel, continuous.n_meaningful
        ));
    }
    if let Some(klv) = continuous.kl_divergence {
        if bounds.require_kl && klv > bounds.max_kl {
            failures.push(format!("kl {:.3e} > bound {:.3e}", klv, bounds.max_kl));
        }
    }
    if !discrete.greedy_match {
        failures.push(format!(
            "greedy argmax mismatch: ref={:?} cand={:?}",
            discrete.greedy_argmax_ref, discrete.greedy_argmax_cand
        ));
    }
    if !discrete.top_k_exact_match {
        failures.push(format!(
            "top-{} mismatch: ref={:?} cand={:?}",
            bounds.top_k, discrete.top_k_ref, discrete.top_k_cand
        ));
    }

    BackendScore {
        backend: backend.to_string(),
        continuous,
        discrete,
        pass: failures.is_empty(),
        failures,
    }
}

/// Score host and device f32 outputs against the same FP64 reference.
pub fn score_pair(host: &[f32], device: &[f32], reference: &[f64], bounds: &Bounds) -> PairedScore {
    let host_s = score_against_f64(host, reference, bounds, "host_f32");
    let device_s = score_against_f64(device, reference, bounds, "device_f32");
    let cutoff = absolute_error_cutoff(reference);
    PairedScore {
        schema: SCHEMA,
        abs_error_cutoff: cutoff,
        bounds: *bounds,
        pass: host_s.pass && device_s.pass,
        host: host_s,
        device: device_s,
    }
}

/// Compact human-readable line for `--nocapture` logs.
///
/// Label contract: the field printed as `max_meaningful_rel` is
/// [`ContinuousMetrics::max_meaningful_rel`] — a **max over single
/// elements**, never a mean. The historical mislabel `mean_rel` is gone;
/// parsers must accept `max_meaningful_rel=`. Sealed receipts that embed
/// the old `mean_rel=` string are left untouched.
pub fn format_score_line(s: &BackendScore) -> String {
    let c = &s.continuous;
    let kl = c
        .kl_divergence
        .map(|k| format!("{k:.3e}"))
        .unwrap_or_else(|| "n/a".into());
    format!(
        "[{}] pass={} rel_l2={:.3e} max_meaningful_rel={:.3e} abs_near={:.3e} (cut={:.3e}) \
         cos={:.9} kl={} ulp[med/p95/p99/max]={:.0}/{:.0}/{:.0}/{:.0} \
         argmax={:?}/{:?} topk_ok={} diag_max_scalar_rel={:.3e}",
        s.backend,
        s.pass,
        c.relative_l2,
        c.max_meaningful_rel,
        c.max_abs_near_zero,
        c.abs_error_cutoff,
        c.cosine_similarity,
        kl,
        c.ulp.median,
        c.ulp.p95,
        c.ulp.p99,
        c.ulp.max,
        s.discrete.greedy_argmax_ref,
        s.discrete.greedy_argmax_cand,
        s.discrete.top_k_exact_match,
        c.diagnostic_max_scalar_rel_all
    )
}

// ── unit tests (synthetic, no Metal) ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identical_vectors_pass_with_zero_error() {
        let c: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.0).collect();
        let r: Vec<f64> = c.iter().map(|&v| v as f64).collect();
        let s = score_against_f64(&c, &r, &Bounds::logits(), "id");
        assert!(s.pass, "failures: {:?}", s.failures);
        assert_eq!(s.continuous.relative_l2, 0.0);
        assert_eq!(s.continuous.ulp.max, 0.0);
        assert!(s.discrete.greedy_match);
        assert!(s.discrete.top_k_exact_match);
    }
    #[test]
    fn denormal_scale_element_does_not_fail_relative_gate() {
        let mut r: Vec<f64> = (0..128).map(|i| ((i as f64) - 64.0) * 0.05).collect();
        r[3] = 1e-32;
        let mut c: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        r[3] = 1e-20;
        c[3] = 2e-20;
        let s = score_against_f64(&c, &r, &Bounds::logits(), "denorm");
        assert!(
            s.continuous.diagnostic_max_scalar_rel_all > 1e-5,
            "setup broken: V2 diagnostic should be large, got {}",
            s.continuous.diagnostic_max_scalar_rel_all
        );
        assert!(
            s.pass,
            "V2.1 must not reject denormal-scale relative blow-up: {:?}",
            s.failures
        );
        assert!(s.continuous.n_near_zero >= 1);
        assert!(s.continuous.relative_l2 < 1e-5);
    }
    #[test]
    fn silu_mul_f64_authority_matches_formula() {
        let g = [0.0f64, 1.0, -2.0, 0.5];
        let u = [1.0f64, 2.0, 3.0, -1.0];
        let y = silu_mul_f64_authority(&g, &u).unwrap();
        for i in 0..g.len() {
            let expect = (g[i] / (1.0 + (-g[i]).exp())) * u[i];
            assert!((y[i] - expect).abs() < 1e-15, "i={i}");
        }
        let gh: Vec<f32> = g.iter().map(|&v| v as f32).collect();
        let uh: Vec<f32> = u.iter().map(|&v| v as f32).collect();
        let host = silu_mul_f32_host(&gh, &uh).unwrap();
        let ref64: Vec<f64> = g
            .iter()
            .zip(u.iter())
            .map(|(&a, &b)| (a / (1.0 + (-a).exp())) * b)
            .collect();
        let s = score_against_f64(&host, &ref64, &Bounds::continuous_only(), "silu_host");
        assert!(s.pass, "host silu vs f64: {:?}", s.failures);
    }
    #[test]
    fn wrong_argmax_fails_with_no_tolerance() {
        let r: Vec<f64> = vec![1.0, 2.0, 3.0, 0.5];
        let c: Vec<f32> = vec![1.0, 3.5, 3.0, 0.5];
        let s = score_against_f64(&c, &r, &Bounds::logits(), "argmax");
        assert!(!s.pass);
        assert!(!s.discrete.greedy_match);
        assert!(s.failures.iter().any(|f| f.contains("argmax")));
    }
    #[test]
    fn wrong_topk_fails_with_no_tolerance() {
        let r: Vec<f64> = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let c: Vec<f32> = vec![5.0, 4.0, 0.5, 2.0, 1.0, 3.5];
        let mut bounds = Bounds::logits();
        bounds.top_k = 3;
        let s = score_against_f64(&c, &r, &bounds, "topk");
        assert!(!s.discrete.top_k_exact_match);
        assert!(!s.pass);
        assert!(s.failures.iter().any(|f| f.contains("top-")));
    }
    #[test]
    fn large_relative_l2_fails_headline_gate() {
        let r: Vec<f64> = (0..32).map(|i| (i as f64) + 1.0).collect();
        let c: Vec<f32> = r.iter().map(|&v| (v * 1.1) as f32).collect();
        let s = score_against_f64(&c, &r, &Bounds::continuous_only(), "l2");
        assert!(!s.pass);
        assert!(s.continuous.relative_l2 > 1e-5);
        assert!(s.failures.iter().any(|f| f.contains("relative_l2")));
    }
    #[test]
    fn reduction_order_noise_passes() {
        let r: Vec<f64> = (0..256)
            .map(|i| ((i as f64) * 0.017 - 2.0).sin() * 1.5)
            .collect();
        let c: Vec<f32> = r
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let noise = v * 1e-7 * if i % 2 == 0 { 1.0 } else { -1.0 };
                (v + noise) as f32
            })
            .collect();
        let s = score_against_f64(&c, &r, &Bounds::logits(), "reduction");
        assert!(
            s.pass,
            "1e-7 reduction-order noise must pass V2.1: {:?} | {}",
            s.failures,
            format_score_line(&s)
        );
        assert!(s.continuous.ulp.median >= 0.0);
    }
    #[test]
    fn abs_cutoff_is_data_derived_and_stated() {
        let r: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let cut = absolute_error_cutoff(&r);
        assert!((cut - 1e-6).abs() < 1e-12, "cut={cut}");
        let r2: Vec<f64> = vec![1e3, 1e3, 1e3];
        let cut2 = absolute_error_cutoff(&r2);
        assert!((cut2 - 1e-3).abs() < 1e-12, "cut2={cut2}");
    }
    #[test]
    fn ulp_distribution_reports_quartet_not_lone_max() {
        let r: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut c: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        c[0] = f32::from_bits((1.0f32).to_bits() + 50);
        let u = ulp_distribution(&c, &r);
        assert_eq!(u.n, 4);
        assert_eq!(u.median, 0.0);
        assert_eq!(u.max, 50.0);
        assert!(u.p95 >= u.median);
        assert!(u.p99 >= u.p95);
    }
    #[test]
    fn f64_authority_matvec_matches_manual_dot() {
        let one = 0x3f80u16;
        let two = 0x4000u16;
        let mut bits = Vec::new();
        bits.extend_from_slice(&one.to_le_bytes());
        bits.extend_from_slice(&two.to_le_bytes());
        let y = matvec_bf16_f64_authority(&bits, 2, &[3.0, 4.0]).unwrap();
        assert_eq!(y.len(), 1);
        assert!((y[0] - 11.0).abs() < 1e-12, "y={}", y[0]);
    }
    #[test]
    fn pair_scores_both_backends() {
        let r: Vec<f64> = vec![0.1, 0.2, 0.5, -0.1];
        let h: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        let d = h.clone();
        let p = score_pair(&h, &d, &r, &Bounds::logits());
        assert!(p.pass);
        assert_eq!(p.schema, SCHEMA);
        assert!(p.host.pass && p.device.pass);
    }
    #[test]
    fn format_score_line_names_max_meaningful_rel_not_mean_rel() {
        let r: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut c: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        c[1] = 2.0 * (1.0 + 1e-4);
        let s = score_against_f64(&c, &r, &Bounds::continuous_only(), "label");
        let line = format_score_line(&s);
        assert!(
            line.contains("max_meaningful_rel="),
            "score line must label the max, got: {line}"
        );
        assert!(
            !line.contains("mean_rel="),
            "score line must not mislabel max as mean_rel, got: {line}"
        );
        assert!(s.continuous.max_meaningful_rel > 0.0);
    }
    #[test]
    fn full_forward_bounds_report_max_meaningful_rel_without_gating() {
        let n = 4096usize;
        let mut r = vec![1.0f64; n];
        r[0] = 100.0;
        r[1] = 90.0;
        r[2] = 80.0;
        r[3] = 70.0;
        r[4] = 60.0;
        r[100] = 0.01;
        let mut c: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        c[100] = 0.01 + 1e-4;
        let op_local = score_against_f64(&c, &r, &Bounds::logits(), "op_local");
        assert!(
            !op_local.pass,
            "op-local bounds must still gate max_meaningful_rel: {:?}",
            op_local.failures
        );
        assert!(op_local
            .failures
            .iter()
            .any(|f| f.starts_with("meaningful_rel")));
        let full = score_against_f64(&c, &r, &Bounds::full_forward_logits(), "full_fwd");
        assert!(
            full.pass,
            "full-forward bounds report the tail but do not gate it when \
             rel_l2/cos/kl/discrete hold: {:?} | {}",
            full.failures,
            format_score_line(&full)
        );
        assert!(full.continuous.max_meaningful_rel > 1e-5);
        assert!(full.continuous.relative_l2 < 1e-5);
        let line = format_score_line(&full);
        assert!(line.contains("max_meaningful_rel="));
    }
    #[test]
    fn v2_false_reject_reproduced_and_cleared() {
        let mut r = vec![1.0f64; 64];
        r[0] = 1e-32;
        r[1] = 1e10;
        let mut c: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        let tiny = c[0];
        assert!(
            tiny > 0.0 && tiny < 1e-30,
            "expected tiny positive, got {tiny}"
        );
        c[0] = tiny * 2.0;
        for i in 1..64 {
            c[i] = r[i] as f32;
        }
        let s = score_against_f64(&c, &r, &Bounds::logits(), "v2_pathology");
        assert!(
            s.continuous.diagnostic_max_scalar_rel_all > 1e-3,
            "V2 diagnostic should look catastrophic, got {}",
            s.continuous.diagnostic_max_scalar_rel_all
        );
        assert!(s.pass, "V2.1 must clear the false reject: {:?}", s.failures);
    }
}
