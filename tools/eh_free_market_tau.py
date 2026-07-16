#!/usr/bin/env python3
"""Free-market replay-tau measurement tool for Event Horizon proposers.

Runs an offline replay oracle over a corpus of token sequences for three
model-free proposers: user_ngram, suffix_array, and retrieval (REST-style).
Mirrors the logic of the Rust replay_oracle.rs / UserNgramDraft / SuffixArrayDraft /
RetrievalProposer exactly so the tau numbers are comparable to the in-tree oracle.

Output:
  - Table to stdout: proposer -> drafted_total -> accepted_total -> tau -> vs_ngram
  - Markdown report written to reports/free_market_tau.md

Usage:
  python3 tools/eh_free_market_tau.py
  python3 tools/eh_free_market_tau.py --smoke
  python3 tools/eh_free_market_tau.py --corpus tools/training/data/rwkv7_sft_sample.jsonl
  python3 tools/eh_free_market_tau.py --corpus PATH --max-seqs 50 --max-tokens-per-seq 128

tau definition (same as replay_oracle.rs):
  tau = tokens_emitted / forward_cycles
  where a verify forward retires na_accepted + 1 (bonus) tokens per cycle.
  tau < 1.0: proposer slows things down  (impossible for a lossless draft unless
             the batch penalizes short-proposal cycles -- here it means tau ~= 1
             because the draft never helps but the bonus always fires).
  tau > 1.0: speedup; GO >= 2.5, MARGINAL >= 1.6, NO-GO < 1.6.

Thresholds match the live gate (reports/oracle/spec_accept.json + replay_oracle.rs):
  GO      >= 2.5
  MARGINAL >= 1.6
  NO-GO    < 1.6

Python 3, stdlib only (no new deps beyond tools/training/requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GO_TAU = 2.5
MARGINAL_TAU = 1.6

# Lookahead cap k used for drafts (mirrors user_draft_k default 4 in live loop).
DEFAULT_K = 4

# SuffixArrayDraft anchor h (from suffix_array.rs: h=3)
SUFFIX_H = 3
# SuffixArrayDraft window (from suffix_array.rs: window=10_000)
SUFFIX_WINDOW = 10_000

# RetrievalProposer anchor h (from retrieval.rs: h=4)
RETRIEVAL_H = 4
# RetrievalProposer window (from retrieval.rs: window=50_000)
RETRIEVAL_WINDOW = 50_000

# SpecGovernor default thresholds (from replay_oracle.rs: SpecGovernor::new(16, 0.35))
# max_consecutive_rejections=16, disable_below=0.35
GOV_MAX_CONSEC_REJECT = 16
GOV_DISABLE_BELOW = 0.35
GOV_ENABLE_ABOVE = 0.50   # standard: enable_above > disable_below
GOV_COOLDOWN_STEPS = 16
GOV_WINDOW = 32


# ---------------------------------------------------------------------------
# Python re-implementation of UserNgramDraft (from user_ngram.rs)
# ---------------------------------------------------------------------------

class UserNgramDraft:
    """Bigram + unigram index over emitted token ids.

    Mirrors UserNgramDraft in crates/hawking-core/src/speculate_user_ngram.rs
    exactly: argmax by count, ties broken by smallest id.
    """

    def __init__(self) -> None:
        # (prev, cur) -> {next_id: count}
        self.bigram: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # cur -> {next_id: count}
        self.unigram: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.prev: Optional[int] = None
        self.cur: Optional[int] = None
        self.transitions: int = 0

    def note_token(self, token: int) -> None:
        if self.cur is not None:
            self.unigram[self.cur][token] += 1
            if self.prev is not None:
                self.bigram[(self.prev, self.cur)][token] += 1
            self.transitions += 1
        self.prev = self.cur
        self.cur = token

    def warm_start(self, history: List[int]) -> None:
        for t in history:
            self.note_token(t)
        # reset rolling cursor after seeding
        self.prev = None
        self.cur = None

    def reset_context(self) -> None:
        self.prev = None
        self.cur = None

    def _argmax_count(self, m: Dict[int, int]) -> Optional[int]:
        """Largest count, ties broken by smallest id (mirrors argmax_count fn)."""
        if not m:
            return None
        best_id = None
        best_count = -1
        for tok_id, count in m.items():
            # higher count wins; on tie, smaller id wins
            if count > best_count or (count == best_count and tok_id < best_id):
                best_id = tok_id
                best_count = count
        return best_id

    def best_successor(self, prev: Optional[int], cur: int) -> Optional[int]:
        if prev is not None:
            succ = self.bigram.get((prev, cur))
            if succ:
                best = self._argmax_count(succ)
                if best is not None:
                    return best
        uni = self.unigram.get(cur)
        if uni:
            return self._argmax_count(uni)
        return None

    def propose(self, ctx: List[int], k: int) -> List[int]:
        """Greedy chain proposal up to k tokens. Matches user_ngram.rs propose()."""
        if k == 0 or not ctx:
            return []
        cur0 = ctx[-1]
        prev0 = ctx[-2] if len(ctx) >= 2 else None
        out: List[int] = []
        prev = prev0
        cur = cur0
        last_emitted: Optional[int] = None
        repeat_run = 0
        MAX_REPEAT_RUN = 3
        for _ in range(k):
            nxt = self.best_successor(prev, cur)
            if nxt is None:
                break
            if nxt == last_emitted:
                repeat_run += 1
                if repeat_run >= MAX_REPEAT_RUN:
                    break
            else:
                repeat_run = 0
            out.append(nxt)
            last_emitted = nxt
            prev = cur
            cur = nxt
        return out


# ---------------------------------------------------------------------------
# Python re-implementation of SuffixArrayDraft (from suffix_array.rs)
# ---------------------------------------------------------------------------

class SuffixArrayDraft:
    """Rolling-window exact-match suffix proposer.

    Mirrors SuffixArrayDraft in crates/hawking-core/src/speculate_suffix_array.rs:
    h=3, window=10_000, backwards search for most-recent prior occurrence.
    """

    def __init__(self, h: int = SUFFIX_H, window: int = SUFFIX_WINDOW) -> None:
        self.stream: List[int] = []
        self.window = window
        self.h = h

    def observe(self, emitted: List[int]) -> None:
        self.stream.extend(emitted)
        if len(self.stream) > self.window:
            self.stream = self.stream[len(self.stream) - self.window:]

    def warm(self, history: List[int]) -> None:
        self.observe(history)

    def propose(self, ctx: List[int], k: int) -> List[int]:
        """Backward scan for most-recent prior occurrence of the tail h tokens."""
        slen = len(self.stream)
        if k == 0 or slen <= self.h:
            return []
        tail_start = slen - self.h
        tail = self.stream[tail_start:]
        search_end = slen - self.h  # exclusive upper bound for match start
        best_pos: Optional[int] = None
        for i in range(search_end - 1, -1, -1):
            match = True
            for d in range(self.h):
                if self.stream[i + d] != tail[d]:
                    match = False
                    break
            if match:
                best_pos = i
                break
        if best_pos is None:
            return []
        copy_start = best_pos + self.h
        copy_end = min(copy_start + k, slen)
        if copy_start >= copy_end:
            return []
        return self.stream[copy_start:copy_end]


# ---------------------------------------------------------------------------
# Python re-implementation of RetrievalProposer (from retrieval.rs)
# ---------------------------------------------------------------------------

class RetrievalProposer:
    """Wide-corpus exact-match retrieval proposer.

    Mirrors RetrievalProposer in crates/hawking-core/src/speculate_retrieval.rs:
    h=4, window=50_000. The key difference from SuffixArrayDraft is:
      - anchor h=4 (longer, higher precision, lower recall)
      - corpus includes warm()-seeded history in addition to emitted tokens
      - anchor is taken from ctx.tokens tail, not from the internal stream tail
        (so the caller passes the full emitted context as ctx)
    """

    def __init__(self, h: int = RETRIEVAL_H, window: int = RETRIEVAL_WINDOW) -> None:
        self.corpus: List[int] = []
        self.window = window
        self.h = h

    def _append(self, tokens: List[int]) -> None:
        self.corpus.extend(tokens)
        if len(self.corpus) > self.window:
            self.corpus = self.corpus[len(self.corpus) - self.window:]

    def observe(self, emitted: List[int]) -> None:
        self._append(emitted)

    def warm(self, history: List[int]) -> None:
        self._append(history)

    def propose(self, ctx: List[int], k: int) -> List[int]:
        """Backward scan in corpus using last h tokens of ctx as anchor."""
        clen = len(self.corpus)
        if k == 0 or len(ctx) < self.h or clen <= self.h:
            return []
        anchor = ctx[len(ctx) - self.h:]
        search_end = clen - self.h  # exclusive
        best_pos: Optional[int] = None
        for i in range(search_end - 1, -1, -1):
            match = True
            for d in range(self.h):
                if self.corpus[i + d] != anchor[d]:
                    match = False
                    break
            if match:
                best_pos = i
                break
        if best_pos is None:
            return []
        copy_start = best_pos + self.h
        copy_end = min(copy_start + k, clen)
        if copy_start >= copy_end:
            return []
        return self.corpus[copy_start:copy_end]


# ---------------------------------------------------------------------------
# SpecGovernor (simplified, matching replay_oracle.rs SpecGovernor::new(16, 0.35))
# ---------------------------------------------------------------------------

class SpecGovernor:
    """Rolling-accept-rate governor that mirrors SpecGovernor from governor.rs.

    Params from replay_oracle.rs: SpecGovernor::new(16, 0.35)
      max_consecutive_rejections=16, disable_below=0.35
    """

    def __init__(
        self,
        max_consec_reject: int = GOV_MAX_CONSEC_REJECT,
        disable_below: float = GOV_DISABLE_BELOW,
        enable_above: float = GOV_ENABLE_ABOVE,
        cooldown_steps: int = GOV_COOLDOWN_STEPS,
        window: int = GOV_WINDOW,
    ) -> None:
        self.max_consec_reject = max_consec_reject
        self.disable_below = disable_below
        self.enable_above = enable_above
        self.cooldown_steps = cooldown_steps
        self.window = window
        self._enabled = True
        self._consec_reject = 0
        self._history: List[bool] = []  # rolling window of accept outcomes
        self._cooldown_remaining = 0

    def is_enabled(self) -> bool:
        return self._enabled

    def _rolling_rate(self) -> Optional[float]:
        if len(self._history) < self.window:
            return None
        return sum(self._history) / len(self._history)

    def step(self, accepted: bool) -> bool:
        """Record an outcome and return the new enabled state."""
        self._history.append(accepted)
        if len(self._history) > self.window:
            self._history.pop(0)

        if self._enabled:
            if not accepted:
                self._consec_reject += 1
            else:
                self._consec_reject = 0
            rate = self._rolling_rate()
            if (self._consec_reject >= self.max_consec_reject or
                    (rate is not None and rate <= self.disable_below)):
                self._enabled = False
                self._cooldown_remaining = self.cooldown_steps
                self._consec_reject = 0
        else:
            # cooling down
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                rate = self._rolling_rate()
                if rate is None or rate >= self.enable_above:
                    self._enabled = True
                    self._consec_reject = 0
        return self._enabled


# ---------------------------------------------------------------------------
# Core replay function
# ---------------------------------------------------------------------------

def replay_proposer(
    corpus: List[int],
    proposer_name: str,
    k: int = DEFAULT_K,
    warm_start_tokens: int = 0,
) -> Dict:
    """Replay a single proposer over corpus, returning per-K stats dict.

    Follows the same replay loop as replay_oracle.rs replay_k():
      - bootstrap: first scored token emitted free (0 forward cycles)
      - each cycle: propose k tokens, compute na (accepted prefix), advance na+1
      - tau = tokens_emitted / forward_cycles

    proposer_name: one of "user_ngram", "suffix_array", "retrieval"
    """
    warm = min(warm_start_tokens, len(corpus))
    seed = corpus[:warm]
    scored = corpus[warm:]

    # Initialise the proposer
    if proposer_name == "user_ngram":
        proposer = UserNgramDraft()
        if seed:
            proposer.warm_start(seed)
        # for the replay we track the rolling 2-gram context [prev, cur] ourselves
        # (same as the Rust loop's ctx_buf)
        use_ngram = True
    elif proposer_name == "suffix_array":
        proposer = SuffixArrayDraft()
        if seed:
            proposer.warm(seed)
        use_ngram = False
    elif proposer_name == "retrieval":
        proposer = RetrievalProposer()
        if seed:
            proposer.warm(seed)
        use_ngram = False
    else:
        raise ValueError(f"unknown proposer: {proposer_name!r}")

    gov = SpecGovernor()

    forward_cycles = 0
    tokens_emitted = 0
    drafts_proposed = 0
    drafts_accepted = 0
    cycles_with_proposal = 0
    cycles_hit = 0
    gov_propose_cycles = 0
    accept_hist = [0] * (k + 1)

    # Rolling context for user_ngram (mirrors ctx_buf [prev, cur] in 'udpf_loop)
    prev: Optional[int] = None
    cur: Optional[int] = None

    # For suffix_array / retrieval we maintain the full emitted context window
    # as a list (bounded to RETRIEVAL_WINDOW for retrieval's propose() call)
    emitted_ctx: List[int] = list(seed)

    i = 0
    while i < len(scored):
        # Bootstrap: retire the first scored token with no draft forward cycle.
        # Mirrors the Rust loop's "let Some(cur_tok) = cur else { ... }" bootstrap.
        if cur is None and use_ngram:
            t = scored[i]
            proposer.note_token(t)
            prev = cur
            cur = t
            tokens_emitted += 1
            i += 1
            continue
        elif not use_ngram and len(emitted_ctx) < (SUFFIX_H if proposer_name == "suffix_array" else RETRIEVAL_H):
            # Need enough context to even attempt a proposal; bootstrap token(s).
            t = scored[i]
            emitted_ctx.append(t)
            proposer.observe([t])
            tokens_emitted += 1
            i += 1
            continue

        # One verify forward per cycle.
        forward_cycles += 1

        # Governor decision for this cycle.
        gov_propose = gov.is_enabled()
        if gov_propose:
            gov_propose_cycles += 1

        # Build context and propose.
        if use_ngram:
            ctx_for_propose = [prev, cur] if prev is not None else [cur]
            draft = proposer.propose(ctx_for_propose, k)
        elif proposer_name == "suffix_array":
            # SuffixArrayDraft.propose uses its internal stream tail; ctx arg unused
            draft = proposer.propose(emitted_ctx, k)
        else:
            # RetrievalProposer: anchor from ctx.tokens (last h tokens of emitted_ctx)
            draft = proposer.propose(emitted_ctx, k)

        dlen = len(draft)
        drafts_proposed += dlen
        if dlen > 0:
            cycles_with_proposal += 1

        # Compute na: longest prefix of draft matching corpus ground-truth continuation.
        avail = len(scored) - i
        na = 0
        while na < dlen and na < avail and draft[na] == scored[i + na]:
            na += 1
        bonus = 1 if (i + na < len(scored)) else 0
        retired = na + bonus

        # Should not happen in practice but guard defensively.
        if retired == 0:
            retired = 1

        drafts_accepted += na
        accept_hist[min(na, k)] += 1
        if na > 0:
            cycles_hit += 1
        tokens_emitted += retired

        # Retire tokens: grow the index and advance context.
        retired_slice = scored[i:i + retired]
        if use_ngram:
            for t in retired_slice:
                proposer.note_token(t)
                prev = cur
                cur = t
        else:
            proposer.observe(retired_slice)
            emitted_ctx.extend(retired_slice)
            # Keep emitted_ctx bounded (avoid unbounded growth on long corpora)
            max_ctx = RETRIEVAL_WINDOW
            if len(emitted_ctx) > max_ctx:
                emitted_ctx = emitted_ctx[len(emitted_ctx) - max_ctx:]

        # Governor step: same as replay_oracle.rs
        if gov_propose:
            gov.step(na > 0)
        else:
            gov.step(False)

        i += retired

    f = max(forward_cycles, 1)
    tau = tokens_emitted / f
    mean_accepted_len = drafts_accepted / f
    hit_rate = (cycles_hit / cycles_with_proposal) if cycles_with_proposal > 0 else 0.0
    proposal_coverage = cycles_with_proposal / f
    draft_accept_frac = (drafts_accepted / drafts_proposed) if drafts_proposed > 0 else 0.0
    governor_propose_frac = gov_propose_cycles / f

    return {
        "proposer": proposer_name,
        "k": k,
        "warm_start_tokens": warm,
        "scored_tokens": len(scored),
        "forward_cycles": forward_cycles,
        "tokens_emitted": tokens_emitted,
        "drafts_proposed": drafts_proposed,
        "drafts_accepted": drafts_accepted,
        "cycles_with_proposal": cycles_with_proposal,
        "tau": round(tau, 4),
        "mean_accepted_len": round(mean_accepted_len, 4),
        "hit_rate": round(hit_rate, 4),
        "proposal_coverage": round(proposal_coverage, 4),
        "draft_accept_frac": round(draft_accept_frac, 4),
        "governor_propose_frac": round(governor_propose_frac, 4),
        "accept_hist": accept_hist,
    }


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def load_jsonl_sequences(
    path: str,
    max_seqs: int,
    max_tokens_per_seq: int,
) -> List[List[int]]:
    """Load corpus from a .jsonl file as word-split proxy token ids.

    Reads "text" (or "response") fields. Uses whitespace split as a proxy
    for tokens (or tiktoken if available). Maps each word to a stable integer
    id via a local vocab dict.
    """
    try:
        import tiktoken as _tiktoken  # type: ignore
        enc = _tiktoken.get_encoding("cl100k_base")

        def tokenize(text: str) -> List[int]:
            return enc.encode(text)[:max_tokens_per_seq]
    except ImportError:
        # Whitespace-split proxy: map each unique word to an integer id.
        vocab: Dict[str, int] = {}

        def tokenize(text: str) -> List[int]:  # type: ignore[misc]
            ids = []
            for w in text.split():
                if w not in vocab:
                    vocab[w] = len(vocab)
                ids.append(vocab[w])
                if len(ids) >= max_tokens_per_seq:
                    break
            return ids

    seqs: List[List[int]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Prefer "text" field; fall back to "response", then "messages" last turn
            text = None
            if isinstance(obj, dict):
                if "text" in obj and isinstance(obj["text"], str):
                    text = obj["text"]
                elif "response" in obj and isinstance(obj["response"], str):
                    text = obj["response"]
                elif "messages" in obj and isinstance(obj["messages"], list):
                    msgs = obj["messages"]
                    for m in reversed(msgs):
                        if isinstance(m, dict) and m.get("role") == "assistant":
                            text = m.get("content", "")
                            break
                    if text is None and msgs:
                        last = msgs[-1]
                        if isinstance(last, dict):
                            text = last.get("content", "")
            if not text:
                continue
            ids = tokenize(text)
            if len(ids) >= 4:
                seqs.append(ids)
            if len(seqs) >= max_seqs:
                break
    return seqs


def hardcoded_smoke_sequences() -> List[List[int]]:
    """Five short Rust-boilerplate-like proxy sequences for --smoke fallback.

    Token ids are arbitrary integers mimicking code structure repetition
    (fn declarations, struct patterns, use statements, etc.).
    The sequences have deliberate repetition to exercise the proposers.
    """
    # Sequence 1: fn foo() + fn bar() boilerplate with repeated tokens
    s1 = [1, 2, 3, 4, 5, 6, 7,   # fn foo() -> bool {
          8, 9, 10, 11,             # let x = true;
          12, 13, 14,               # }
          1, 2, 15, 4, 5, 6, 7,    # fn bar() -> bool {
          8, 9, 10, 11,
          12, 13, 14,
          1, 2, 16, 4, 5, 6, 7,
          8, 9, 10, 11, 12, 13, 14]
    # Sequence 2: struct + impl boilerplate
    s2 = [20, 21, 22, 23, 24,      # struct Foo {
          25, 26, 27, 28, 29,       # field: u32,
          30, 31,                   # }
          32, 33, 22, 34, 35, 36,   # impl Foo {
          37, 38, 39, 40, 41,       # pub fn new() -> Self {
          42, 43, 44,               # Self { field: 0 }
          30, 31,                   # }
          37, 38, 45, 40, 41,       # pub fn get() -> u32 {
          46, 47, 25, 30, 31, 30, 31]
    # Sequence 3: repeated use + match arms
    s3 = [50, 51, 52, 53, 54,      # use std::collections::HashMap;
          50, 51, 52, 53, 55,       # use std::collections::HashSet;
          56, 57, 58, 59, 60, 61,  # match x {
          62, 63, 64, 65, 66,       # Ok(v) => v,
          67, 68, 69, 70, 71,       # Err(e) => panic!(...),
          72, 73,                   # }
          56, 57, 58, 59, 60, 61,
          62, 63, 64, 65, 66,
          67, 68, 69, 70, 71, 72, 73]
    # Sequence 4: for loop over vec pattern (highly repetitive)
    s4 = [80, 81, 82, 83, 84, 85,  # for item in items {
          86, 87, 88, 89, 90,       # process(item);
          91, 92,                   # }
          80, 81, 82, 83, 84, 85,
          86, 87, 88, 89, 90, 91, 92,
          80, 81, 82, 83, 84, 85,
          86, 87, 88, 89, 90, 91, 92]
    # Sequence 5: let binding + assert pattern
    s5 = [100, 101, 102, 103, 104,  # let result = compute();
          105, 106, 107, 108,        # assert!(result.is_ok());
          100, 109, 110, 103, 111,   # let value = result.unwrap();
          112, 113, 114, 115, 116,   # assert_eq!(value, expected);
          100, 101, 102, 103, 104,
          105, 106, 107, 108,
          100, 109, 110, 103, 111,
          112, 113, 114, 115, 116]
    return [s1, s2, s3, s4, s5]


# ---------------------------------------------------------------------------
# tau verdict helper
# ---------------------------------------------------------------------------

def verdict(tau: float) -> str:
    if tau >= GO_TAU:
        return "GO"
    if tau >= MARGINAL_TAU:
        return "MARGINAL"
    return "NO-GO"


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(
    results: List[Dict],
    out_path: str,
    corpus_source: str,
    n_seqs: int,
    smoke: bool,
) -> None:
    """Write a Markdown report to out_path."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Aggregate per proposer across sequences
    totals: Dict[str, Dict] = {}
    for r in results:
        pname = r["proposer"]
        if pname not in totals:
            totals[pname] = {
                "drafted": 0, "accepted": 0, "forward_cycles": 0,
                "tokens_emitted": 0, "scored_tokens": 0,
                "seqs": 0,
            }
        totals[pname]["drafted"] += r["drafts_proposed"]
        totals[pname]["accepted"] += r["drafts_accepted"]
        totals[pname]["forward_cycles"] += r["forward_cycles"]
        totals[pname]["tokens_emitted"] += r["tokens_emitted"]
        totals[pname]["scored_tokens"] += r["scored_tokens"]
        totals[pname]["seqs"] += 1

    rows = []
    ngram_tau = None
    for pname in ["user_ngram", "suffix_array", "retrieval"]:
        if pname not in totals:
            continue
        t = totals[pname]
        fc = max(t["forward_cycles"], 1)
        tau_val = t["tokens_emitted"] / fc
        daf = (t["accepted"] / t["drafted"]) if t["drafted"] > 0 else 0.0
        rows.append({
            "proposer": pname,
            "seqs": t["seqs"],
            "scored_tokens": t["scored_tokens"],
            "drafted_total": t["drafted"],
            "accepted_total": t["accepted"],
            "forward_cycles": fc,
            "tau": round(tau_val, 4),
            "draft_accept_frac": round(daf, 4),
            "verdict": verdict(tau_val),
        })
        if pname == "user_ngram":
            ngram_tau = tau_val

    # vs_baseline column
    for r in rows:
        if ngram_tau is not None and ngram_tau > 0:
            r["vs_ngram"] = f"{(r['tau'] / ngram_tau):+.3f}x"
        else:
            r["vs_ngram"] = "n/a"

    lines: List[str] = []
    lines.append("# Free-Market Replay-τ Measurement")
    lines.append("")
    lines.append(f"**Corpus**: {corpus_source}  ")
    lines.append(f"**Sequences scored**: {n_seqs}  ")
    lines.append(f"**Mode**: {'smoke (fast)' if smoke else 'full'}  ")
    lines.append(f"**k** (lookahead cap): {DEFAULT_K}  ")
    lines.append(f"**Thresholds**: GO ≥ {GO_TAU}, MARGINAL ≥ {MARGINAL_TAU}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| proposer | scored_tok | drafted | accepted | tau | draft_acc% | vs_ngram | verdict |")
    lines.append("|----------|-----------|---------|----------|-----|-----------|---------|---------|")
    for r in rows:
        dap = f"{r['draft_accept_frac']*100:.1f}%"
        lines.append(
            f"| {r['proposer']} | {r['scored_tokens']} | {r['drafted_total']} | "
            f"{r['accepted_total']} | {r['tau']:.4f} | {dap} | {r['vs_ngram']} | **{r['verdict']}** |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `tau = tokens_emitted / forward_cycles` (each verify cycle retires `na_accepted + 1` tokens).")
    lines.append("- `vs_ngram` is `tau(proposer) / tau(user_ngram)` (relative to the n-gram baseline).")
    lines.append("- Token sequences are whitespace-split proxy ids (or tiktoken cl100k_base if available).")
    lines.append("- This is an offline oracle; live tau depends on the actual decoder + token distribution.")
    lines.append(f"- `user_ngram`: bigram+unigram, chains greedily, h=2/1, window=unbounded.")
    lines.append(f"- `suffix_array`: rolling exact match, h={SUFFIX_H}, window={SUFFIX_WINDOW}.")
    lines.append(f"- `retrieval`: wide-corpus exact match, h={RETRIEVAL_H}, window={RETRIEVAL_WINDOW}.")
    lines.append("")
    lines.append("## Raw per-sequence data")
    lines.append("")
    lines.append("| seq | proposer | scored_tok | drafted | accepted | tau | verdict |")
    lines.append("|-----|----------|-----------|---------|----------|-----|---------|")
    for i, r in enumerate(results):
        fc = max(r["forward_cycles"], 1)
        tau_val = r["tokens_emitted"] / fc
        lines.append(
            f"| {i // 3} | {r['proposer']} | {r['scored_tokens']} | "
            f"{r['drafts_proposed']} | {r['drafts_accepted']} | "
            f"{tau_val:.4f} | {verdict(tau_val)} |"
        )
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--corpus",
        default=None,
        help="Path to .jsonl corpus (default: tools/training/data/rwkv7_sft_sample.jsonl)",
    )
    ap.add_argument(
        "--max-seqs",
        type=int,
        default=20,
        help="Maximum number of sequences to score (default: 20)",
    )
    ap.add_argument(
        "--max-tokens-per-seq",
        type=int,
        default=64,
        help="Maximum tokens per sequence (default: 64)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Draft lookahead cap (default: {DEFAULT_K})",
    )
    ap.add_argument(
        "--warm-start-frac",
        type=float,
        default=0.0,
        help="Fraction of each sequence to use as warm-start seed (0.0 = cold, default)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 2 sequences, all 3 proposers, fast exit",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output report path (default: reports/free_market_tau.md)",
    )
    args = ap.parse_args()

    # Resolve repo root and default paths relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)  # tools/ is one level below repo root

    default_corpus = os.path.join(repo_root, "tools", "training", "data", "rwkv7_sft_sample.jsonl")
    default_report = os.path.join(repo_root, "reports", "free_market_tau.md")

    corpus_path = args.corpus or default_corpus
    report_path = args.out or default_report
    k = args.k

    max_seqs = 2 if args.smoke else args.max_seqs
    max_tokens = args.max_tokens_per_seq

    # Load sequences
    if args.smoke and not os.path.isfile(corpus_path):
        seqs = hardcoded_smoke_sequences()[:2]
        corpus_source = "hardcoded smoke sequences (5 Rust-boilerplate patterns)"
        print(f"[smoke] no corpus at {corpus_path!r}, using hardcoded smoke sequences")
    elif os.path.isfile(corpus_path):
        seqs = load_jsonl_sequences(corpus_path, max_seqs, max_tokens)
        corpus_source = corpus_path
        if not seqs:
            print(f"WARNING: no usable sequences in {corpus_path!r}; using hardcoded fallback")
            seqs = hardcoded_smoke_sequences()[:max_seqs]
            corpus_source = "hardcoded smoke fallback"
    else:
        print(f"WARNING: corpus not found at {corpus_path!r}; using hardcoded sequences")
        seqs = hardcoded_smoke_sequences()[:max_seqs]
        corpus_source = "hardcoded smoke sequences (corpus not found)"

    if args.smoke:
        seqs = seqs[:2]

    print(f"Corpus  : {corpus_source}")
    print(f"Seqs    : {len(seqs)}")
    print(f"k       : {k}")
    print(f"Mode    : {'smoke' if args.smoke else 'full'}")
    print()

    proposer_names = ["user_ngram", "suffix_array", "retrieval"]
    all_results: List[Dict] = []

    for seq_idx, seq in enumerate(seqs):
        warm = int(len(seq) * args.warm_start_frac)
        for pname in proposer_names:
            r = replay_proposer(seq, pname, k=k, warm_start_tokens=warm)
            r["seq_idx"] = seq_idx
            all_results.append(r)

    # Aggregate and print table
    totals: Dict[str, Dict] = {}
    for r in all_results:
        pname = r["proposer"]
        if pname not in totals:
            totals[pname] = {"drafted": 0, "accepted": 0, "forward_cycles": 0,
                             "tokens_emitted": 0, "scored_tokens": 0}
        totals[pname]["drafted"] += r["drafts_proposed"]
        totals[pname]["accepted"] += r["drafts_accepted"]
        totals[pname]["forward_cycles"] += r["forward_cycles"]
        totals[pname]["tokens_emitted"] += r["tokens_emitted"]
        totals[pname]["scored_tokens"] += r["scored_tokens"]

    ngram_tau: Optional[float] = None
    summary_rows = []
    for pname in proposer_names:
        if pname not in totals:
            continue
        t = totals[pname]
        fc = max(t["forward_cycles"], 1)
        tau_val = t["tokens_emitted"] / fc
        daf = (t["accepted"] / t["drafted"]) if t["drafted"] > 0 else 0.0
        if pname == "user_ngram":
            ngram_tau = tau_val
        summary_rows.append((pname, t["drafted"], t["accepted"], tau_val, daf))

    # Print summary table
    hdr = f"{'proposer':<14} {'drafted':>9} {'accepted':>9} {'tau':>7} {'draft_acc%':>10} {'vs_ngram':>9} {'verdict':>10}"
    print(hdr)
    print("-" * len(hdr))
    for pname, drafted, accepted, tau_val, daf in summary_rows:
        vs = f"{(tau_val / ngram_tau):+.3f}x" if (ngram_tau and ngram_tau > 0) else "n/a"
        verd = verdict(tau_val)
        print(
            f"{pname:<14} {drafted:>9} {accepted:>9} {tau_val:>7.4f} "
            f"{daf*100:>9.1f}% {vs:>9}  {verd}"
        )
    print()

    # Write report
    write_report(
        all_results,
        report_path,
        corpus_source,
        len(seqs),
        args.smoke,
    )
    print(f"Report  : {report_path}")


if __name__ == "__main__":
    main()
