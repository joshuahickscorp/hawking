# Hawking Event Horizon — As-Built Status (2026-06-20)

Grounded against the committed source in `crates/hawking-core/src/speculate/` and the
test file `crates/hawking-core/tests/user_draft_parity_e2e.rs`.

Commits that built this:
- `4b7ffc1` feat(speculate): Event Horizon Phase 0/1 — unified lossless proposal market
- `6e9f271` feat(speculate): Event Horizon Phases 2-8 — proposal-market modules

---

## Phase matrix

| Phase | Description | Status | Flag / Gate | What exists in code |
|-------|-------------|--------|-------------|---------------------|
| 0 | Unify Proposer / Router / Verifier / Telemetry | FUNCTIONAL | `HAWKING_QWEN_EVENT_HORIZON` (default OFF) | `proposal.rs` Proposer trait + Ctx/Budget/Proposal/Telemetry; `verifier.rs` ExactTarget + Verifier::verify_line; `router.rs` ProposalRouter::new/plan/record; `impl ExactTarget for QwenDense` + flag-gated `'ud_loop` seam in `qwen_dense.rs`; parity gate passing (16 tok, CPU fp16 + GPU pruned-Q4K) |
| 1 | Universal base (n-gram + suffix-array) | FUNCTIONAL | `HAWKING_QWEN_EVENT_HORIZON` | `user_ngram.rs` NgramProposer adapter (τ=1.43 base) + `suffix_array.rs` SuffixArrayDraft; both registered via `add_free_slot` with `oracle_cleared=true`; two-proposer max-gain arbitration active in `plan()` |
| 2 | Retrieval (REST-style local corpus) | FUNCTIONAL | `HAWKING_QWEN_EVENT_HORIZON` / `ProposerId::Retrieval` | `retrieval.rs` RetrievalProposer fully implemented (h=4, window=50 000, warm/propose/observe lifecycle); registered as a free slot in the router; no datastore build tooling yet |
| 3 | Router v1 (cost-aware arbiter) | FUNCTIONAL CORE / TIMING WIRING DEFERRED | `HAWKING_QWEN_EVENT_HORIZON` | `router.rs` uses per-slot EWMA acceptance and measured-or-bootstrap B=0..8 verifier cost, charges draft/retokenize/sync plus wall-clock hysteresis, and advances disabled dwell without fake misses. The production Qwen loop still feeds placeholder target cost and zero timing fields; Appendix C owns the post-ladder wiring/receipt gate. |
| 4 | Neural gated (EAGLE-3-H hidden tap) | SCAFFOLD | NEVER ENABLED — τ=0.877 < 2.5 gate | `eagle_proposer.rs` EagleProposer wraps Eagle5Head behind Proposer trait (requires_hidden=true); enable_neural_slot refuses verdict≠"GO" at construction — no runtime path exists to enable it; tests exercise the adapter but the kill-ledger verdict is permanent |
| 5 | Parallel drafter (P-EAGLE / DFlash) | NOT YET BUILT | `HAWKING_EH_PARALLEL_DRAFT` (not yet) | No file, no ProposerId entry, no router slot. The Phase 5 P-EAGLE/DFlash parallel-block approach was deferred; the commit message confirms "Phase 5 deferred" |
| 6 | Tree verify (DDTree / SpecInfer) | SCAFFOLD | `ProposerId::Tree` | `tree.rs` TokenTreeBuilder + ancestor-mask computation + CPU linear fallback verifier; `supports_tree_verify()=false` (no Metal kernel); ProposerId::Tree slot in enum; no wiring to the production loop |
| 7 | Cross-tokenizer bridge (UAG / OmniDraft) | SCAFFOLD | `ProposerId::CrossTokenizer` — enable_neural_slot refuses without GO | `cross_tokenizer.rs` CrossTokenizerProposer (requires_text_bridge=true); span_map populated via warm()/learn_span; disabled by default — the 0.58–0.70× Apple Silicon slowdown risk (ref 2604.16368) means enable_neural_slot will deny it without an oracle GO |
| 8 | Online policy (bandit / RL) | FUNCTIONAL (additive only) | `plan_bandit()` — not wired to production loop | `policy.rs` BanditPolicy + UCB1 arm selection; ProposalRouter::plan_bandit() is a fully working alternate plan path; BanditPolicy field sits in ProposalRouter and arms track per-slot reward; plan_bandit() is NOT called by the production `'ud_loop` — plan() (expected-gain) is |

---

## Kill ledger (from docs/dead_levers.md)

**EAGLE-3 trained draft head (Eagle5 v3) — permanent NO-GO.**

Evidence (`docs/dead_levers.md`, entry dated 2026-05-31, doubly confirmed):
- Held-out τ = **0.877** against the gate of **τ ≥ 2.5**.
  Per-position accept: [0.523, 0.195, 0.097, 0.062].
- On-device paired bench (Qwen-3B, code prompt, locked env):
  baseline 36.9 dec\_tps; spec K=2/4/8 = 14.9/11.1/7.6 → **0.40×/0.30×/0.21×**.
  Net-negative, worse at larger K.
- Device accept 6.5% vs PyTorch held-out 52% (~8× forward-parity gap).
- **The free n-gram draft (τ=1.43) beats the trained head (0.877) on code.**

Resurrection check (verbatim from the ledger): "do NOT re-train the EAGLE head
expecting a win without an oracle first showing achievable τ≥2.5 on the target
workload."

This is enforced structurally: `ProposalRouter::enable_neural_slot` returns
`Err("gated proposer denied: oracle verdict not GO (tau<2.5)")` for any
`requires_hidden` or `requires_text_bridge` slot whose `oracle_verdict != "GO"`.
There is no code path to schedule Eagle5 at runtime.

---

## This build's additions (2026-06-20)

| Item | File | Type | Description |
|------|------|------|-------------|
| A | `crates/hawking-core/src/speculate/proposal.rs` | FUNCTIONAL | Proposer trait + Ctx/Budget/Proposal/Telemetry contracts |
| B | `crates/hawking-core/src/speculate/verifier.rs` | FUNCTIONAL | ExactTarget trait + Verifier::verify_line; unit-tested with mock target (full-accept, mid-reject, empty-draft cases) |
| C | `crates/hawking-core/src/speculate/router.rs` | FUNCTIONAL | ProposalRouter with EWMA CostModel, expected_gain arbiter, per-slot SpecGovernor, add_free_slot, plan_bandit; 3 unit tests |
| D | `crates/hawking-core/src/speculate/suffix_array.rs` | FUNCTIONAL | SuffixArrayDraft rolling-window exact-match proposer (h=3, window=10 000); impl Proposer with observe/warm/reset |
| E | `crates/hawking-core/src/speculate/user_ngram.rs` (appended) | FUNCTIONAL | NgramProposer adapter over the live UserNgramDraft (τ=1.43 base) |
| F | `crates/hawking-core/src/speculate/retrieval.rs` | FUNCTIONAL | RetrievalProposer — wider-corpus complement to SuffixArrayDraft (h=4, window=50 000) |
| G | `crates/hawking-core/src/speculate/eagle_proposer.rs` | SCAFFOLD | EagleProposer wrapping Eagle5Head; requires_hidden=true; permanently NO-GO via kill ledger |
| H | `crates/hawking-core/src/speculate/tree.rs` | SCAFFOLD | TokenTreeBuilder + ancestor-mask helpers + CPU linear fallback verifier; Metal single-pass kernel is a documented stub |
| I | `crates/hawking-core/src/speculate/cross_tokenizer.rs` + `policy.rs` | SCAFFOLD + FUNCTIONAL | CrossTokenizerProposer (scaffold, gated); BanditPolicy UCB1 (functional, additive only) |
| — | `crates/hawking-core/src/model/qwen_dense.rs` | FIX / FUNCTIONAL | `impl ExactTarget for QwenDense` (UFCS); flag-gated `'ud_loop` seam replacing inline accept block with Verifier::verify_line + ProposalRouter::plan (P0.4/0.6/0.7) |
| — | `crates/hawking-core/tests/user_draft_parity_e2e.rs` | MEASUREMENT | `event_horizon_bit_identical_default` + `event_horizon_bit_identical_fast_pruned_q4k` parity gates (P0.6); 16-token bit-identical EH-OFF vs EH-ON on CPU fp16 and GPU pruned-Q4K paths |
| — | `docs/plans/hawking_event_horizon_proposal_engine.md` | DOCS | Frontier survey + phase plan + kill/resurrection checklist |
| — | `docs/plans/hawking_event_horizon_phase0_blueprint.md` | DOCS | Phase 0/1 implementation blueprint, grounded signatures |
| — | `docs/plans/hawking_event_horizon_status.md` (this file) | DOCS | As-built functional/scaffold matrix |

---

## Measurement study plan

The following measurements are blocked only by the KD job occupying the GPU.
CPU-only items can run immediately.

### τ sweep across Eagle5Head layer triplets (item G / Phase 4 prerequisite)
- What it measures: offline accept rate (τ) at each candidate low/mid/high hidden-layer
  triplet to find whether any configuration reaches the τ ≥ 2.5 promotion gate.
- Needs: GPU (model forward pass per triplet), model weights
  (`models/qwen2.5-3b-instruct-q4_k_m.gguf`), small capture corpus
  (`tools/training/data/rwkv7_sft_sample.jsonl` or equivalent).
- Tooling: `tools/training/eagle5_tau_eval.py` (pre-existing).
- GPU pressure: low to moderate — one forward pass per triplet per corpus sample;
  run at N=20 prompts for a quick screen, N=100 for a gate.
- Constraint: kill-ledger gate is τ ≥ 2.5. The existing head measured τ=0.877;
  **no triplet result should be acted on before N ≥ 100 prompts on the target workload.**

### Free-market replay-τ (Phase 3 / router acceptance calibration)
- What it measures: per-proposer accept rate (τ) and expected\_gain distribution
  across the proposal market (n-gram + suffix-array + retrieval) on a realistic corpus.
- Needs: CPU only — `replay_oracle.rs` ReplayReport, the existing capture corpus.
- Tooling: `tools/bench/draft_accept_oracle.py` (pre-existing).
- Deliverable: per-proposer τ by context class (code, JSON, prose); expected\_gain ns
  distribution vs target\_ns\_per\_token; informs `margin_ns` and `alpha` defaults in
  the router (currently placeholder values 1.0 and 0.10).
- Can run now (no GPU needed).

### Market payoff bench (Phase 1/3 — accepted-tokens/s improvement)
- What it measures: end-to-end tok/s with EH-ON vs EH-OFF on the repetitive-code
  and JSON/tool-call workloads that the n-gram + suffix-array market targets.
- Needs: GPU, model weights, 1–2 prompts for smokes (already runnable; full bench
  requires N ≥ 50 prompts and max\_new\_tokens ≥ 256).
- **Current smoke status:** the parity gate runs at max\_new\_tokens=16; payoff is
  not visible at 16 tokens (too few draft cycles). Full bench deferred until after KD.
- Tooling: extend `user_draft_parity_e2e.rs` with a timed variant, or use
  `tools/bench/tps_bench.py` with `HAWKING_QWEN_EVENT_HORIZON=1`.

### Full parity property (Phase 0 — beyond 16 tokens)
- What it asserts: bit-identical output EH-OFF vs EH-ON at max\_new\_tokens=256,
  exercising many more accept/reject boundaries and KV-rewind cycles.
- Needs: GPU; max\_new\_tokens currently capped at 16 in the test to keep the
  parity run short.
- **Current status:** 16-token gate is green on both CPU fp16 and GPU pruned-Q4K
  paths (`event_horizon_bit_identical_default` + `event_horizon_bit_identical_fast_pruned_q4k`).
  Un-cap to 256 after KD completes.

---

## What the GPU lane still needs (after KD job finishes)

The following items are blocked ONLY by the KD job occupying MPS. None require
new code — they require lifted caps or longer runs on existing infrastructure.

1. **Full market payoff bench** — `HAWKING_QWEN_EVENT_HORIZON=1`, N ≥ 50 prompts,
   max\_new\_tokens ≥ 256. Validates that the free n-gram + suffix-array market
   yields a measured tok/s improvement on code/JSON workloads vs the legacy loop.
   Paired bench: `accepted_tokens/s` with EH-ON vs EH-OFF, locking env identically.

2. **Full τ sweep for Eagle5Head layer triplets** — `eagle5_tau_eval.py`,
   N ≥ 100 prompts on the target workload (Qwen-3B + code), all candidate
   low/mid/high hidden-layer triplets. The only purpose is to determine whether
   any triplet reaches τ ≥ 2.5; if none do, the neural slot remains permanently
   gated. Do not act on N < 100 results.

3. **Full parity property** — un-cap `MAX_NEW_TOKENS` in
   `event_horizon_bit_identical_default` and `event_horizon_bit_identical_fast_pruned_q4k`
   from 16 to 256; re-run the gate. The 64-token `user_draft_propose_first_lossless_long`
   test already exercises more KV-rewind boundaries but does not cover the EH seam.

4. **Paired bench for the two-proposer market** — suffix-array + n-gram composite vs
   n-gram alone on code/JSON/agent-loop corpus. This is the Phase 1 deliverable gate
   (plan spec §1.4 bench) and requires a timed run, not just a parity assert.
