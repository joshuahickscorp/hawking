//! L3.1 oracle instrument — live token-usage + draft accept/reject capture.
//!
//! **This is the oracle's data source, not the lever.** It does not prune the
//! vocab, tune a draft, or touch any logit. It is a pure side-observer of the
//! emitted-token stream the decode loop already produces, accumulating the two
//! statistics §2.2 of `plans/stateful_moat_continuation_design_2026_05_31.md`
//! requires before the L3.1 adaptation bodies (vocab screen / `UserNgramDraft`)
//! are tuned online:
//!
//!   - **(a)** a per-emitted-token argmax-id **frequency histogram** — the
//!     usage-frequency signal for the §2.1a hot set (which tokens a single
//!     user actually emits as the argmax), and
//!   - **(b)** a **draft accept/reject ledger keyed by n-gram context** — the
//!     per-context acceptance signal for §2.1b draft tuning (which `(c_{i-1},
//!     c_i) → next` grams predict well, so the draft proposes aggressively
//!     only where the per-context hit-rate is high).
//!
//! Entirely gated behind `DISMANTLE_QWEN_USAGE_CAPTURE=1`. With the flag unset
//! every entry point here is a cheap `if` that returns immediately, so the
//! production decode path is untouched and parity is unaffected — the observer
//! never changes which token is emitted. The accumulator is process-global (a
//! `Mutex<Option<…>>`) so no `QwenDense` field or constructor change is needed
//! — the capture is a pure side-observer, exactly like
//! [`super::attn_capture`].
//!
//! Output: a compact JSON written to `$DISMANTLE_USAGE_CAPTURE_OUT` (default
//! `reports/bench/usage_capture.json`) on [`flush`], holding the top argmax-id
//! frequencies and the per-context accept/reject ledger aggregated over the
//! whole run. The offline reader (`tools/bench/oracle_vocab_coverage.py` and
//! the user-warm-start extension of `oracle_spec_accept.py`) turns it into the
//! GO/NO-GO verdicts.

#[cfg(target_os = "macos")]
use std::collections::HashMap;
#[cfg(target_os = "macos")]
use std::sync::Mutex;

/// How many of the most-frequent argmax ids to emit in the flushed JSON. The
/// full histogram is retained in memory (it is a single user's session, a few
/// thousand distinct ids at most); only the head is serialized so the oracle's
/// hot-set sizing has the high-frequency tail it needs without an enormous
/// file. Overridable via `DISMANTLE_USAGE_CAPTURE_TOPK`.
#[cfg(target_os = "macos")]
const DEFAULT_TOPK: usize = 4096;
/// How many of the highest-traffic n-gram contexts to emit. Same rationale.
#[cfg(target_os = "macos")]
const DEFAULT_TOP_CTX: usize = 4096;

/// Per-context draft outcome tally. Keyed by the 2-gram context `(prev, cur)`
/// that preceded a draft proposal (matching the `n=2` the spec oracle uses,
/// `reports/oracle/spec_accept.json`). The reader divides `accepted` by
/// `accepted + rejected` to get the per-context hit-rate and reads `next` to
/// learn what continuation the context predicts.
#[cfg(target_os = "macos")]
#[derive(Default, Clone)]
struct CtxTally {
    /// Σ accepted draft tokens proposed under this context.
    accepted: u64,
    /// Σ rejected draft tokens proposed under this context.
    rejected: u64,
    /// The most-recent token the verifier actually emitted after this context
    /// (the "what the user repeats" signal for the warm-start draft).
    last_next: Option<u32>,
}

#[cfg(target_os = "macos")]
struct UsageState {
    /// (a) per-emitted-token argmax-id frequency histogram.
    argmax_freq: HashMap<u32, u64>,
    /// Σ over all recorded emitted tokens (== Σ argmax_freq values). Lets the
    /// reader report coverage as a fraction without re-summing.
    total_emitted: u64,
    /// (b) draft accept/reject ledger keyed by the 2-gram context.
    ctx_ledger: HashMap<(u32, u32), CtxTally>,
    /// Σ accepted / Σ rejected across all contexts (the pooled draft τ inputs).
    total_draft_accepted: u64,
    total_draft_rejected: u64,
    /// Number of draft *proposals* recorded (cycles), for mean-accepted-length.
    total_draft_cycles: u64,
}

#[cfg(target_os = "macos")]
impl UsageState {
    fn new() -> Self {
        UsageState {
            argmax_freq: HashMap::new(),
            total_emitted: 0,
            ctx_ledger: HashMap::new(),
            total_draft_accepted: 0,
            total_draft_rejected: 0,
            total_draft_cycles: 0,
        }
    }
}

#[cfg(target_os = "macos")]
static STATE: Mutex<Option<UsageState>> = Mutex::new(None);

/// `true` when `DISMANTLE_QWEN_USAGE_CAPTURE=1`. Cheap; called per emitted
/// token, so it short-circuits the whole instrument when unset.
#[cfg(target_os = "macos")]
pub fn enabled() -> bool {
    crate::env_on("DISMANTLE_QWEN_USAGE_CAPTURE")
}
#[cfg(not(target_os = "macos"))]
pub fn enabled() -> bool {
    false
}

/// Record one emitted token's argmax id (statistic (a)).
///
/// Called at every emit site (accepted draft, correction, plain greedy token).
/// No-op unless [`enabled`]. A single hash bump; zero allocation on the steady
/// path (the entry exists after first sighting).
#[cfg(target_os = "macos")]
pub fn record_argmax(id: u32) {
    if !enabled() {
        return;
    }
    let mut guard = match STATE.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let st = guard.get_or_insert_with(UsageState::new);
    *st.argmax_freq.entry(id).or_insert(0) += 1;
    st.total_emitted += 1;
}
#[cfg(not(target_os = "macos"))]
pub fn record_argmax(_id: u32) {}

/// Record one draft cycle's accept/reject outcome under its n-gram context
/// (statistic (b)).
///
/// `ctx` is the 2-gram `(prev, cur)` that preceded the draft proposal; `next`
/// is the token the verifier emitted as the first new token after the context
/// (the continuation this context predicts); `accepted` / `rejected` are the
/// draft-token counts from this cycle (`stats.draft_accepted` /
/// `stats.draft_rejected` deltas). No-op unless [`enabled`].
#[cfg(target_os = "macos")]
pub fn record_draft(ctx: (u32, u32), next: Option<u32>, accepted: usize, rejected: usize) {
    if !enabled() {
        return;
    }
    let mut guard = match STATE.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let st = guard.get_or_insert_with(UsageState::new);
    let t = st.ctx_ledger.entry(ctx).or_default();
    t.accepted += accepted as u64;
    t.rejected += rejected as u64;
    if next.is_some() {
        t.last_next = next;
    }
    st.total_draft_accepted += accepted as u64;
    st.total_draft_rejected += rejected as u64;
    st.total_draft_cycles += 1;
}
#[cfg(not(target_os = "macos"))]
pub fn record_draft(_ctx: (u32, u32), _next: Option<u32>, _accepted: usize, _rejected: usize) {}

/// Write the accumulated usage histogram + draft ledger to
/// `$DISMANTLE_USAGE_CAPTURE_OUT` (default `reports/bench/usage_capture.json`).
/// Safe to call when capture never ran (writes empty collections).
#[cfg(target_os = "macos")]
pub fn flush() {
    if !enabled() {
        return;
    }
    let guard = match STATE.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let out_path = std::env::var("DISMANTLE_USAGE_CAPTURE_OUT")
        .unwrap_or_else(|_| "reports/bench/usage_capture.json".to_string());
    let topk = std::env::var("DISMANTLE_USAGE_CAPTURE_TOPK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_TOPK);
    let top_ctx = std::env::var("DISMANTLE_USAGE_CAPTURE_TOP_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_TOP_CTX);

    let mut s = String::new();
    s.push_str("{\n");

    let st_ref = guard.as_ref();
    let total_emitted = st_ref.map(|s| s.total_emitted).unwrap_or(0);
    let distinct_argmax = st_ref.map(|s| s.argmax_freq.len()).unwrap_or(0);
    let total_acc = st_ref.map(|s| s.total_draft_accepted).unwrap_or(0);
    let total_rej = st_ref.map(|s| s.total_draft_rejected).unwrap_or(0);
    let total_cycles = st_ref.map(|s| s.total_draft_cycles).unwrap_or(0);
    // Pooled mean accepted length τ = (accepted + cycles) / cycles: each cycle
    // emits its accepted drafts plus the verifier's one guaranteed token. This
    // is the same τ the spec oracle reports; `1.0` when no drafting ran.
    let pooled_tau = if total_cycles > 0 {
        (total_acc as f64 + total_cycles as f64) / total_cycles as f64
    } else {
        1.0
    };

    s.push_str(&format!("  \"total_emitted\": {},\n", total_emitted));
    s.push_str(&format!(
        "  \"distinct_argmax_ids\": {},\n",
        distinct_argmax
    ));
    s.push_str(&format!("  \"total_draft_accepted\": {},\n", total_acc));
    s.push_str(&format!("  \"total_draft_rejected\": {},\n", total_rej));
    s.push_str(&format!("  \"total_draft_cycles\": {},\n", total_cycles));
    s.push_str(&format!("  \"pooled_tau\": {:.4},\n", pooled_tau));

    // (a) argmax histogram — top-K by frequency, descending.
    s.push_str("  \"argmax_freq_topk\": [\n");
    if let Some(st) = st_ref {
        let mut pairs: Vec<(u32, u64)> = st.argmax_freq.iter().map(|(&k, &v)| (k, v)).collect();
        // Descending by count, then by id for stable output.
        pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut first = true;
        for (id, count) in pairs.into_iter().take(topk) {
            if !first {
                s.push_str(",\n");
            }
            first = false;
            s.push_str(&format!("    {{\"id\": {}, \"count\": {}}}", id, count));
        }
    }
    s.push_str("\n  ],\n");

    // (b) per-context draft ledger — top by traffic (accepted+rejected).
    s.push_str("  \"draft_ctx_ledger\": [\n");
    if let Some(st) = st_ref {
        let mut rows: Vec<(&(u32, u32), &CtxTally)> = st.ctx_ledger.iter().collect();
        rows.sort_unstable_by(|a, b| {
            let ta = a.1.accepted + a.1.rejected;
            let tb = b.1.accepted + b.1.rejected;
            tb.cmp(&ta).then(a.0.cmp(b.0))
        });
        let mut first = true;
        for ((prev, cur), t) in rows.into_iter().take(top_ctx) {
            if !first {
                s.push_str(",\n");
            }
            first = false;
            let total = t.accepted + t.rejected;
            let hit_rate = if total > 0 {
                t.accepted as f64 / total as f64
            } else {
                0.0
            };
            let next = t.last_next.map(|n| n as i64).unwrap_or(-1);
            s.push_str(&format!(
                "    {{\"prev\": {}, \"cur\": {}, \"accepted\": {}, \"rejected\": {}, \
                 \"hit_rate\": {:.4}, \"next\": {}}}",
                prev, cur, t.accepted, t.rejected, hit_rate, next
            ));
        }
    }
    s.push_str("\n  ]\n}\n");

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if std::fs::write(&out_path, &s).is_ok() {
        eprintln!("[usage-capture] wrote {}", out_path);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn flush() {}

/// Test-only reset of the process-global state so unit tests do not leak into
/// each other. Not part of the production surface.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn reset_for_test() {
    if let Ok(mut g) = STATE.lock() {
        *g = None;
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// Serialize the env-var mutation across the tests in this module: they all
    /// flip `DISMANTLE_QWEN_USAGE_CAPTURE`, which is process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn off_is_a_noop() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("DISMANTLE_QWEN_USAGE_CAPTURE");
        reset_for_test();
        assert!(!enabled());
        // Recording while off must not allocate state.
        record_argmax(42);
        record_draft((1, 2), Some(3), 5, 2);
        let guard = STATE.lock().unwrap();
        assert!(
            guard.is_none(),
            "capture state must stay None when the flag is off"
        );
    }

    #[test]
    fn on_records_argmax_and_draft() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::set_var("DISMANTLE_QWEN_USAGE_CAPTURE", "1");
        reset_for_test();
        assert!(enabled());

        record_argmax(7);
        record_argmax(7);
        record_argmax(9);
        record_draft((10, 11), Some(12), 3, 1);
        record_draft((10, 11), Some(12), 2, 0);
        record_draft((20, 21), Some(22), 0, 4);

        let guard = STATE.lock().unwrap();
        let st = guard.as_ref().expect("state must exist when on");
        assert_eq!(st.total_emitted, 3);
        assert_eq!(st.argmax_freq.get(&7), Some(&2));
        assert_eq!(st.argmax_freq.get(&9), Some(&1));
        // Context (10,11): 3+2 accepted, 1+0 rejected, next=12.
        let t = st.ctx_ledger.get(&(10, 11)).expect("ctx (10,11)");
        assert_eq!(t.accepted, 5);
        assert_eq!(t.rejected, 1);
        assert_eq!(t.last_next, Some(12));
        // Context (20,21): 0 accepted, 4 rejected.
        let t2 = st.ctx_ledger.get(&(20, 21)).expect("ctx (20,21)");
        assert_eq!(t2.accepted, 0);
        assert_eq!(t2.rejected, 4);
        assert_eq!(st.total_draft_accepted, 5);
        assert_eq!(st.total_draft_rejected, 5);
        assert_eq!(st.total_draft_cycles, 3);

        drop(guard);
        // Leave the env clean for sibling tests.
        std::env::remove_var("DISMANTLE_QWEN_USAGE_CAPTURE");
        reset_for_test();
    }
}
