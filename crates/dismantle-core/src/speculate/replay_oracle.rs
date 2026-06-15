//! Offline spec replay-oracle for the n-gram user draft (pure CPU, no GPU, no
//! model forward, no training).
//!
//! Track 6 sits low partly because the free n-gram draft's acceptance was never
//! measured *in-tree against the shipped logic*. The external PLD scorer in
//! `reports/oracle/spec_accept.json` measured a python re-implementation; this
//! module instead replays a corpus of token ids through the **actual**
//! [`UserNgramDraft`] (`propose` / `note_token` / `warm_start`) using the
//! corpus's own next tokens as ground truth -- the standard n-gram
//! self-acceptance proxy. Acceptance is a draft-vs-realized-token property, so
//! no forward pass is needed: the realized token *is* the corpus's next token.
//!
//! It reproduces the live `qwen_dense` `'udpf_loop` accept rule exactly:
//!   * at each cycle the draft proposes up to `k` tokens for the current 2-gram
//!     context (`[prev, cur]`);
//!   * `na` = the longest prefix of that proposal equal to the corpus's actual
//!     next tokens (stop at the first mismatch -- the verifier would correct it);
//!   * the stream advances `na + 1` tokens (the `na` accepted drafts plus the
//!     one bonus/correction the verify forward emits), so one forward retires
//!     `na + 1` tokens;
//!   * every retired token is fed back via `note_token`, growing the index
//!     online exactly as the live loop does.
//!
//! The speedup ceiling is `tau = tokens_emitted / forward_cycles` (a free CPU
//! draft means tau is the decode speedup ceiling -- the same definition as the
//! existing oracle: GO>=2.5, MARGINAL>=1.6, else NO-GO). The oracle also runs a
//! [`SpecGovernor`] over the per-cycle accept stream and reports what fraction
//! of cycles it would have *proposed* on -- i.e. whether the governor would have
//! shut the draft off on this corpus.
//!
//! Pure logic, fully unit-tested; the `cargo test -p dismantle-core --lib`
//! gate needs no Metal, no model, no tokenizer.

use super::governor::SpecGovernor;
use super::user_ngram::UserNgramDraft;

/// Per-`k` replay result over one corpus.
#[derive(Debug, Clone, PartialEq)]
pub struct KReport {
    /// Lookahead cap used for this row (`propose(ctx, k)`).
    pub k: usize,
    /// Forward cycles run (one batched verify forward per cycle).
    pub forward_cycles: u64,
    /// Tokens retired (== positions advanced == accepted drafts + bonuses).
    pub tokens_emitted: u64,
    /// Total draft tokens proposed across all cycles.
    pub drafts_proposed: u64,
    /// Total draft tokens accepted across all cycles (sum of per-cycle `na`).
    pub drafts_accepted: u64,
    /// Cycles that proposed at least one draft token (index non-empty / matched).
    pub cycles_with_proposal: u64,
    /// Speedup ceiling: `tokens_emitted / forward_cycles`.
    pub tau: f32,
    /// Mean accepted-run length: `drafts_accepted / forward_cycles` (excludes the
    /// always-present bonus token, so 0.0 means "no draft ever helped").
    pub mean_accepted_len: f32,
    /// Per-cycle accept rate: cycles with `na>0` / `cycles_with_proposal`.
    pub hit_rate: f32,
    /// Proposal coverage: cycles that proposed >=1 / forward_cycles.
    pub proposal_coverage: f32,
    /// Acceptance fraction of *proposed* drafts: `drafts_accepted / drafts_proposed`.
    pub draft_accept_frac: f32,
    /// Histogram of per-cycle accepted-draft counts (`na`); index = na.
    pub accept_hist: Vec<u64>,
    /// Fraction of cycles on which a [`SpecGovernor`] (default thresholds) would
    /// have been *enabled* (i.e. proposed) -- low means the governor would mostly
    /// shut the draft off on this corpus.
    pub governor_propose_frac: f32,
}

/// Aggregate of a grid replay (one [`KReport`] per requested `k`).
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayReport {
    /// Token count of the corpus that was scored (after the warm-start split).
    pub scored_tokens: usize,
    /// Token count used to warm-start the index (prefix of the input).
    pub warm_start_tokens: usize,
    /// One row per `k`, in input order.
    pub per_k: Vec<KReport>,
}

impl ReplayReport {
    /// The row with the highest `tau` (ties -> smaller `k`). `None` if empty.
    pub fn best(&self) -> Option<&KReport> {
        self.per_k.iter().reduce(|a, b| {
            if b.tau > a.tau || (b.tau == a.tau && b.k < a.k) {
                b
            } else {
                a
            }
        })
    }

    /// Gate verdict on the best row's tau, using the same bands as the existing
    /// spec oracle: GO>=2.5, MARGINAL>=1.6, else NO-GO. `"EMPTY"` if no rows.
    pub fn verdict(&self) -> &'static str {
        match self.best() {
            None => "EMPTY",
            Some(b) if b.tau >= 2.5 => "GO",
            Some(b) if b.tau >= 1.6 => "MARGINAL",
            _ => "NO-GO",
        }
    }
}

/// Replay one corpus of token ids through the n-gram draft for a single `k`,
/// returning the [`KReport`].
///
/// `warm_start_tokens` of the leading corpus is fed to the index via
/// [`UserNgramDraft::warm_start`] before scoring begins (model the live
/// prompt-seed); the remainder is the *scored* stream. The first scored token
/// has no predecessor draft (nothing to verify), so scoring begins once a
/// 1-token context exists. `k` is clamped to `>=1`.
pub fn replay_k(corpus: &[u32], k: usize, warm_start_tokens: usize) -> KReport {
    let k = k.max(1);
    let warm = warm_start_tokens.min(corpus.len());
    let (seed, scored) = corpus.split_at(warm);

    let mut idx = UserNgramDraft::new();
    if !seed.is_empty() {
        idx.warm_start(seed);
    }
    // A default-config governor observing the same accept stream.
    let mut gov = SpecGovernor::new(16, 0.35);

    let mut forward_cycles: u64 = 0;
    let mut tokens_emitted: u64 = 0;
    let mut drafts_proposed: u64 = 0;
    let mut drafts_accepted: u64 = 0;
    let mut cycles_with_proposal: u64 = 0;
    let mut cycles_hit: u64 = 0;
    let mut gov_propose_cycles: u64 = 0;
    let mut accept_hist: Vec<u64> = vec![0; k + 1];

    // Rolling 2-gram context [prev, cur]; mirrors `ctx_buf` in 'udpf_loop.
    // `prev`/`cur` track the last two RETIRED tokens.
    let mut prev: Option<u32> = None;
    let mut cur: Option<u32> = None;

    let mut i = 0usize;
    while i < scored.len() {
        // Need a context (>=1 retired token) to propose. Bootstrap: retire the
        // first scored token with no draft (the single non-amortized forward in
        // the live loop), then begin the propose/verify cycles.
        let Some(cur_tok) = cur else {
            let t = scored[i];
            idx.note_token(t);
            prev = cur;
            cur = Some(t);
            // The first scored token is emitted "free" — produced by the
            // prefill-equivalent (the live loop's `carried_true` from the
            // prompt), not by a decode verify forward. Count it as emitted so
            // `tokens_emitted == scored_tokens` holds, but charge it 0 forward
            // cycles (it never amortizes a draft), so tau excludes it.
            tokens_emitted += 1;
            i += 1;
            continue;
        };

        // One verify forward per cycle (retires >=1 token below).
        forward_cycles += 1;

        // Governor decision for THIS cycle (mirrors live `gov_propose`).
        let gov_propose = gov.is_enabled();
        if gov_propose {
            gov_propose_cycles += 1;
        }

        // Build the 2-gram context exactly as the live loop: [prev, cur].
        let ctx: Vec<u32> = match prev {
            Some(p) => vec![p, cur_tok],
            None => vec![cur_tok],
        };
        let lookahead = idx.propose(&ctx, k);
        let dlen = lookahead.len();
        drafts_proposed += dlen as u64;
        if dlen > 0 {
            cycles_with_proposal += 1;
        }

        // Ground truth = the corpus's own next tokens (self-acceptance proxy).
        // The verifier's preds[j] is scored[i + j]. `na` = longest matching
        // prefix, identical to the live `na` loop.
        let avail = scored.len() - i; // tokens remaining incl. the bonus slot
        let mut na = 0usize;
        while na < dlen && na < avail && lookahead[na] == scored[i + na] {
            na += 1;
        }
        // The bonus/correction token preds[na] exists iff there is a token at
        // i+na; at the very tail it may not (the last cycle then retires only
        // the accepted drafts). Live loop always has the extra verify slot.
        let bonus = if i + na < scored.len() {
            1usize
        } else {
            0usize
        };
        let retired = na + bonus;
        debug_assert!(retired >= 1, "a cycle must retire at least one token");

        drafts_accepted += na as u64;
        accept_hist[na] += 1;
        if na > 0 {
            cycles_hit += 1;
        }
        tokens_emitted += retired as u64;

        // Grow the index over every retired token, advancing the context.
        for j in 0..retired {
            let t = scored[i + j];
            idx.note_token(t);
            prev = cur;
            cur = Some(t);
        }

        // Governor step: it only observes proposed cycles' outcomes; on a
        // skipped (disabled) cycle the live loop steps `false`. We always
        // proposed here (offline we never actually skip the forward), so feed
        // the real outcome when enabled and a `false` when disabled -- matching
        // the live `step(na>0)` / `step(false)` split.
        if gov_propose {
            let _ = gov.step(na > 0);
        } else {
            let _ = gov.step(false);
        }

        i += retired;
    }

    let f = forward_cycles.max(1) as f32;
    let tau = tokens_emitted as f32 / f;
    let mean_accepted_len = drafts_accepted as f32 / f;
    let hit_rate = if cycles_with_proposal > 0 {
        cycles_hit as f32 / cycles_with_proposal as f32
    } else {
        0.0
    };
    let proposal_coverage = cycles_with_proposal as f32 / f;
    let draft_accept_frac = if drafts_proposed > 0 {
        drafts_accepted as f32 / drafts_proposed as f32
    } else {
        0.0
    };
    let governor_propose_frac = gov_propose_cycles as f32 / f;

    KReport {
        k,
        forward_cycles,
        tokens_emitted,
        drafts_proposed,
        drafts_accepted,
        cycles_with_proposal,
        tau,
        mean_accepted_len,
        hit_rate,
        proposal_coverage,
        draft_accept_frac,
        accept_hist,
        governor_propose_frac,
    }
}

/// Replay a corpus across a grid of `k` values, returning the aggregate.
/// `ks` is deduplicated-by-position (input order preserved); empty `ks`
/// defaults to the live cap of `[4]` (`user_draft_k.min(7)` default 4).
pub fn replay_grid(corpus: &[u32], ks: &[usize], warm_start_tokens: usize) -> ReplayReport {
    let warm = warm_start_tokens.min(corpus.len());
    let default_ks = [4usize];
    let ks_iter: &[usize] = if ks.is_empty() { &default_ks } else { ks };
    let per_k = ks_iter.iter().map(|&k| replay_k(corpus, k, warm)).collect();
    ReplayReport {
        scored_tokens: corpus.len() - warm,
        warm_start_tokens: warm,
        per_k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a corpus of `reps` repetitions of `pattern` (a predictable stream
    /// the n-gram draft should learn and then draft correctly).
    fn repeated(pattern: &[u32], reps: usize) -> Vec<u32> {
        let mut v = Vec::with_capacity(pattern.len() * reps);
        for _ in 0..reps {
            v.extend_from_slice(pattern);
        }
        v
    }

    #[test]
    fn repetitive_corpus_yields_positive_acceptance() {
        // A highly repetitive corpus (the n-gram's best case): once the index
        // has seen the cycle a few times, propose drafts the whole period and
        // they verify against the corpus's own continuation.
        let corpus = repeated(&[10, 11, 12, 13, 14, 15, 16, 17], 200);
        let rep = replay_grid(&corpus, &[4, 7], 0);
        let best = rep.best().expect("non-empty grid");
        assert!(
            best.drafts_accepted > 0,
            "repetitive corpus must accept drafts (got {})",
            best.drafts_accepted
        );
        assert!(
            best.tau > 1.05,
            "repetitive corpus must beat plain decode (tau {})",
            best.tau
        );
        assert!(
            best.mean_accepted_len > 0.0 && best.proposal_coverage > 0.5,
            "draft should propose on most cycles and accept (mal {}, cov {})",
            best.mean_accepted_len,
            best.proposal_coverage
        );
        // Per-token accounting must close: positions advanced == corpus scored.
        for r in &rep.per_k {
            assert_eq!(
                r.tokens_emitted as usize, rep.scored_tokens,
                "k={} retired {} != scored {}",
                r.k, r.tokens_emitted, rep.scored_tokens
            );
            assert_eq!(
                r.drafts_accepted,
                r.accept_hist
                    .iter()
                    .enumerate()
                    .map(|(na, c)| na as u64 * c)
                    .sum::<u64>(),
                "accept_hist must reconstruct drafts_accepted (k={})",
                r.k
            );
        }
    }

    #[test]
    fn non_repetitive_corpus_yields_near_zero_acceptance() {
        // A strictly increasing stream: every (prev,cur) and cur is seen at most
        // once before it must be predicted, so the index has no successor to
        // propose -> ~zero acceptance, tau ~= 1 (no speedup).
        let corpus: Vec<u32> = (0u32..4000).collect();
        let rep = replay_grid(&corpus, &[4, 7], 0);
        for r in &rep.per_k {
            assert_eq!(
                r.drafts_accepted, 0,
                "an all-unique stream cannot accept any draft (k={}, got {})",
                r.k, r.drafts_accepted
            );
            assert!(
                (r.tau - 1.0).abs() < 1e-3,
                "no acceptance must give tau~=1 (k={}, tau {})",
                r.k,
                r.tau
            );
            assert!(
                r.mean_accepted_len < 1e-6,
                "mean accepted-run length must be ~0 (k={}, {})",
                r.k,
                r.mean_accepted_len
            );
            // Per-token accounting still closes (every cycle retires exactly 1).
            assert_eq!(r.tokens_emitted as usize, rep.scored_tokens);
        }
        assert_eq!(rep.verdict(), "NO-GO", "unique stream is below the gate");
    }

    #[test]
    fn warm_start_split_is_honored_and_helps() {
        // Warm-starting the index from the corpus prefix should let the scored
        // suffix accept immediately (the warm-start lever the oracle measured).
        let corpus = repeated(&[5, 6, 7, 8], 100);
        // Score only the last 40 tokens; warm-start from the rest.
        let warm = corpus.len() - 40;
        let rep = replay_grid(&corpus, &[4], warm);
        assert_eq!(rep.warm_start_tokens, warm);
        assert_eq!(rep.scored_tokens, 40);
        let r = &rep.per_k[0];
        assert!(
            r.drafts_accepted > 0 && r.tau > 1.2,
            "warm-started repetitive suffix should draft well (acc {}, tau {})",
            r.drafts_accepted,
            r.tau
        );
    }

    #[test]
    fn governor_shuts_off_on_unpredictable_corpus() {
        // On a stream with no acceptance, the SpecGovernor's consecutive-miss
        // bail must trip and it should propose on only a small fraction of
        // cycles (it shuts the draft off) -- the governor-projection signal.
        let corpus: Vec<u32> = (0u32..2000).collect();
        let rep = replay_grid(&corpus, &[4], 0);
        let r = &rep.per_k[0];
        assert!(
            r.governor_propose_frac < 0.2,
            "governor must mostly disable on an unpredictable stream (frac {})",
            r.governor_propose_frac
        );
        // ...whereas on a predictable stream it stays mostly enabled.
        let good = repeated(&[1, 2, 3, 4], 500);
        let gr = replay_grid(&good, &[4], 0);
        assert!(
            gr.per_k[0].governor_propose_frac > 0.8,
            "governor must stay enabled on a predictable stream (frac {})",
            gr.per_k[0].governor_propose_frac
        );
    }

    #[test]
    fn empty_and_tiny_corpora_do_not_panic() {
        let empty: Vec<u32> = Vec::new();
        let rep = replay_grid(&empty, &[4], 0);
        assert_eq!(rep.scored_tokens, 0);
        assert_eq!(rep.best().map(|r| r.forward_cycles), Some(0));
        // single token: bootstrap emits it "free" (0 forward cycles), no
        // verify cycle runs.
        let one = replay_grid(&[42u32], &[4], 0);
        assert_eq!(one.per_k[0].forward_cycles, 0);
        assert_eq!(one.per_k[0].tokens_emitted, 1);
    }
}
