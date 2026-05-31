//! L1.1 oracle instrument — attention-mass concentration capture.
//!
//! **This is the oracle, not the lever.** It does not evict, compress, or
//! touch the KV cache. It observes the post-softmax attention distribution
//! the CPU reference attention path (`crate::attn::mha_decode_step`, used by
//! `QwenDense::forward_token` during prefill) already materializes, and
//! accumulates the statistics §2.5 of `plans/stateful_core_design_2026_05_30.md`
//! requires before the [`super::working_set`] bodies may be written:
//!
//!   - per query position, per layer: how many cached positions hold ≥99%
//!     (and ≥90%, ≥99.9%) of the cumulative attention mass,
//!   - how that mass splits between **attention sinks** (the first few
//!     positions), a **recent window** (the trailing positions near the
//!     query), and a **scattered middle** — the StreamingLLM/H2O structure,
//!     re-measured on Qwen2.5-3B rather than assumed.
//!
//! Entirely gated behind `DISMANTLE_QWEN_ATTN_CAPTURE=1`. With the flag
//! unset every entry point here is a cheap `if` that returns immediately, so
//! the production decode path is untouched. The accumulator is process-global
//! (a `Mutex<Option<…>>`) so no `QwenDense` field or constructor change is
//! needed — the capture is a pure side-observer.
//!
//! Output: a compact JSON written to `$DISMANTLE_ATTN_CAPTURE_OUT`
//! (default `reports/bench/attn_capture.json`) on [`flush`], holding the
//! per-layer concentration curve aggregated over all captured query
//! positions. The offline reader `tools/bench/oracle_attn_mass.py` turns it
//! into the GO/NO-GO verdict.

#[cfg(target_os = "macos")]
use std::sync::Mutex;

/// Sink span (leading positions) and recent-window span (trailing positions
/// relative to the query) used to classify where the heavy mass sits. These
/// mirror the StreamingLLM defaults (4 sinks) and a modest recent window; the
/// oracle reports mass under several windows so the verdict is not sensitive
/// to one arbitrary choice.
const SINK_SPAN: usize = 4;
const RECENT_SPANS: [usize; 3] = [32, 64, 128];
/// Mass thresholds whose "minimum positions to reach" we record per query.
const MASS_THRESHOLDS: [f32; 3] = [0.90, 0.99, 0.999];

/// Running per-layer aggregate. All counts are summed over every captured
/// (query-position) sample for that layer; the reader divides by `samples`.
#[cfg(target_os = "macos")]
#[derive(Default, Clone)]
struct LayerAgg {
    /// Number of query positions captured for this layer.
    samples: u64,
    /// Σ over samples of the *current cached length* (== query_pos + 1). Lets
    /// the reader report concentration as a fraction of context.
    sum_ctx_len: u64,
    /// For each mass threshold: Σ over samples of (min #positions whose
    /// descending-sorted mass first reaches the threshold).
    sum_min_pos: [u64; MASS_THRESHOLDS.len()],
    /// For each mass threshold: Σ over samples of (min #positions) / ctx_len,
    /// i.e. the *fraction* of cached positions needed. Reported as a mean.
    sum_min_frac: [f64; MASS_THRESHOLDS.len()],
    /// Σ over samples of attention mass falling on the first `SINK_SPAN`
    /// positions.
    sum_sink_mass: f64,
    /// For each recent-window span: Σ over samples of mass on the trailing
    /// window [ctx_len-span, ctx_len).
    sum_recent_mass: [f64; RECENT_SPANS.len()],
    /// For each recent-window span: Σ over samples of (sink_mass +
    /// recent_mass) — the StreamingLLM "sinks + recent window" coverage.
    sum_sink_plus_recent: [f64; RECENT_SPANS.len()],
    /// Σ over samples of the single largest attention weight (peakiness).
    sum_top1: f64,
}

#[cfg(target_os = "macos")]
struct CaptureState {
    layers: Vec<LayerAgg>,
    /// Only positions at/after this cached length are recorded, so the curve
    /// reflects the long-context regime the lever targets (short prefixes are
    /// trivially "concentrated" and would bias the mean optimistic).
    min_ctx_len: usize,
}

#[cfg(target_os = "macos")]
static STATE: Mutex<Option<CaptureState>> = Mutex::new(None);

/// `true` when `DISMANTLE_QWEN_ATTN_CAPTURE=1`. Cheap; called per layer per
/// prefill token, so it short-circuits the whole instrument when unset.
#[cfg(target_os = "macos")]
pub fn enabled() -> bool {
    crate::env_on("DISMANTLE_QWEN_ATTN_CAPTURE")
}
#[cfg(not(target_os = "macos"))]
pub fn enabled() -> bool {
    false
}

/// Record one query's attention distribution for `layer`.
///
/// `head_weights` is `n_heads` slices, each the post-softmax attention the
/// current query placed over the `ctx_len` cached positions (in
/// retained-position order, position 0 == oldest). We head-average into one
/// distribution (the design's documented reporting choice) and fold its
/// concentration profile into the per-layer aggregate.
///
/// No-op unless [`enabled`]. Allocation-light: one `ctx_len`-sized scratch
/// reused via the descending partial sort.
#[cfg(target_os = "macos")]
pub fn record_layer(layer: usize, n_layers: usize, head_weights: &[&[f32]]) {
    if head_weights.is_empty() {
        return;
    }
    let ctx_len = head_weights[0].len();
    if ctx_len == 0 {
        return;
    }
    let mut guard = match STATE.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let st = guard.get_or_insert_with(|| CaptureState {
        layers: vec![LayerAgg::default(); n_layers],
        min_ctx_len: std::env::var("DISMANTLE_ATTN_CAPTURE_MIN_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512),
    });
    if ctx_len < st.min_ctx_len {
        return;
    }
    if layer >= st.layers.len() {
        return;
    }

    // Head-average into one distribution over cached positions.
    let n_heads = head_weights.len() as f32;
    let mut avg = vec![0.0f32; ctx_len];
    for hw in head_weights {
        // Guard against ragged input; all heads share ctx_len here.
        let m = hw.len().min(ctx_len);
        for (j, &w) in hw[..m].iter().enumerate() {
            avg[j] += w;
        }
    }
    let inv = 1.0 / n_heads;
    for v in avg.iter_mut() {
        *v *= inv;
    }

    // Structural masses (computed on the in-order distribution).
    let sink_mass: f64 = avg[..SINK_SPAN.min(ctx_len)]
        .iter()
        .map(|&w| w as f64)
        .sum();
    let mut recent_mass = [0.0f64; RECENT_SPANS.len()];
    for (ri, &span) in RECENT_SPANS.iter().enumerate() {
        let start = ctx_len.saturating_sub(span);
        recent_mass[ri] = avg[start..].iter().map(|&w| w as f64).sum();
    }

    // Concentration: sort descending, walk the cumulative curve once.
    let mut sorted = avg.clone();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top1 = sorted[0] as f64;
    let mut min_pos = [ctx_len as u64; MASS_THRESHOLDS.len()];
    let mut cum = 0.0f64;
    let mut next_thr = 0usize;
    for (k, &w) in sorted.iter().enumerate() {
        cum += w as f64;
        while next_thr < MASS_THRESHOLDS.len() && cum >= MASS_THRESHOLDS[next_thr] as f64 {
            min_pos[next_thr] = (k + 1) as u64;
            next_thr += 1;
        }
        if next_thr >= MASS_THRESHOLDS.len() {
            break;
        }
    }

    // Fold into the aggregate.
    let a = &mut st.layers[layer];
    a.samples += 1;
    a.sum_ctx_len += ctx_len as u64;
    a.sum_top1 += top1;
    a.sum_sink_mass += sink_mass;
    for ri in 0..RECENT_SPANS.len() {
        a.sum_recent_mass[ri] += recent_mass[ri];
        // sinks ∪ recent — avoid double-counting the overlap when the recent
        // window reaches back into the sink span (only at tiny ctx_len, which
        // min_ctx_len already filters, but stay correct anyway).
        let overlap_start = ctx_len.saturating_sub(RECENT_SPANS[ri]);
        let overlap = if overlap_start < SINK_SPAN.min(ctx_len) {
            avg[overlap_start..SINK_SPAN.min(ctx_len)]
                .iter()
                .map(|&w| w as f64)
                .sum::<f64>()
        } else {
            0.0
        };
        a.sum_sink_plus_recent[ri] += sink_mass + recent_mass[ri] - overlap;
    }
    for ti in 0..MASS_THRESHOLDS.len() {
        a.sum_min_pos[ti] += min_pos[ti];
        a.sum_min_frac[ti] += min_pos[ti] as f64 / ctx_len as f64;
    }
}

#[cfg(not(target_os = "macos"))]
pub fn record_layer(_layer: usize, _n_layers: usize, _head_weights: &[&[f32]]) {}

/// Write the accumulated per-layer concentration curve to
/// `$DISMANTLE_ATTN_CAPTURE_OUT` (default `reports/bench/attn_capture.json`).
/// Safe to call when capture never ran (writes an empty `layers: []`).
#[cfg(target_os = "macos")]
pub fn flush() {
    if !enabled() {
        return;
    }
    let guard = match STATE.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let out_path = std::env::var("DISMANTLE_ATTN_CAPTURE_OUT")
        .unwrap_or_else(|_| "reports/bench/attn_capture.json".to_string());

    let mut s = String::new();
    s.push_str("{\n");
    let (min_ctx, layers): (usize, &[LayerAgg]) = match guard.as_ref() {
        Some(st) => (st.min_ctx_len, &st.layers),
        None => (0, &[]),
    };
    s.push_str(&format!("  \"min_ctx_len\": {},\n", min_ctx));
    s.push_str(&format!(
        "  \"sink_span\": {}, \"recent_spans\": [{}],\n",
        SINK_SPAN,
        RECENT_SPANS
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    s.push_str(&format!(
        "  \"mass_thresholds\": [{}],\n",
        MASS_THRESHOLDS
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    s.push_str("  \"layers\": [\n");
    let mut first = true;
    for (li, a) in layers.iter().enumerate() {
        if a.samples == 0 {
            continue;
        }
        if !first {
            s.push_str(",\n");
        }
        first = false;
        let n = a.samples as f64;
        let mean_ctx = a.sum_ctx_len as f64 / n;
        let mean_min_pos: Vec<String> = a
            .sum_min_pos
            .iter()
            .map(|&x| format!("{:.2}", x as f64 / n))
            .collect();
        let mean_min_frac: Vec<String> = a
            .sum_min_frac
            .iter()
            .map(|&x| format!("{:.5}", x / n))
            .collect();
        let mean_recent: Vec<String> = a
            .sum_recent_mass
            .iter()
            .map(|&x| format!("{:.5}", x / n))
            .collect();
        let mean_sink_recent: Vec<String> = a
            .sum_sink_plus_recent
            .iter()
            .map(|&x| format!("{:.5}", x / n))
            .collect();
        s.push_str(&format!(
            "    {{\"layer\": {}, \"samples\": {}, \"mean_ctx_len\": {:.1}, \
             \"mean_top1\": {:.5}, \"mean_sink_mass\": {:.5}, \
             \"mean_recent_mass\": [{}], \"mean_sink_plus_recent\": [{}], \
             \"mean_min_pos\": [{}], \"mean_min_frac\": [{}]}}",
            li,
            a.samples,
            mean_ctx,
            a.sum_top1 / n,
            a.sum_sink_mass / n,
            mean_recent.join(", "),
            mean_sink_recent.join(", "),
            mean_min_pos.join(", "),
            mean_min_frac.join(", "),
        ));
    }
    s.push_str("\n  ]\n}\n");

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&out_path, &s).is_ok() {
        eprintln!("[attn-capture] wrote {}", out_path);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn flush() {}
