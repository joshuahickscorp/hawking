# Speculative-decode re-entry appendix — 2026-07-14

This is Appendix C under `docs/plans/APPENDIX.md`.

## Relationship to the ladder and prior plan

This is the second additive appendage to the current ladder. Compression and
Doctor recovery remain the current work. When those heavy runs finish, native TQ
hardware interpretation is measured first, then speculative speed work is tested
against the exact served TQ/Doctor distribution. No existing ladder phase or
document numbering needs to change.

`spec_decode_studio_readiness_2026_07_12.md` remains the canonical fail-closed
admission contract. This appendix adds code scaffolding and a concrete experiment
matrix; it does not weaken `spec_revive.py`, flip `RUNNER_IMPLEMENTED`, enable a
dead proposer, or launch work during the active run.

## Current code audit

The exact verifier and proposal-market skeleton are substantial, but “spec
decode” is not one finished path:

- `forward_tokens_verify` has a B=1 canonical-greedy correction and B=2..8
  batched path. Its projection dispatcher now checks the same offset-keyed TQ
  ownership map first. A TQ-owned projection runs a batch-major B=1..8 bitslice
  kernel for stored, compact, hashed, or computed interpretation; residual TQ
  passes accumulate, and the incompatible Q4_K fused-down shortcut is disabled
  when TQ owns `ffn_down`. Proof mode requires all seven linears per layer to be
  TQ-owned and GPU-resident. This closes the code bypass, but device parity is
  still unproven until the post-run receipt is green.
- User n-gram and suffix proposers are wired. Retrieval exists but is not
  registered in the production Qwen loop.
- `ParallelDraftProposer` emits zero token IDs and reports zero cost. Its runtime
  and oracle gates correctly keep it disabled; it is interface scaffolding, not a
  draft model.
- Token-tree construction, ancestor masks, and a CPU path exist. There is no
  single-pass Metal tree verifier, and the CPU fallback cannot support a speed
  claim.
- Before this slice, the router carried draft/verify/retokenize/sync timings but
  selected from a fixed historical verifier curve. Disabled slots were fed
  synthetic failures, poisoning their acceptance window rather than learning a
  counterfactual. The production loop still needs post-ladder timing wiring.
- `spec_revive.py` correctly refuses execution until a real checkpointed runner
  exists and hash-bound TQ parity/cost evidence is ready.

## Scaffold added now

The pure-logic router can now accept a validated B=0..8 verifier-cost curve and
uses measured draft, retokenize, sync, and per-B verifier-extra time when
available. Historical costs remain bootstrap values only. A disabled dwell tick
no longer invents a miss. Exact shadow outcomes can enter through
`record_shadow`; absence of shadow evidence cannot re-enable an arm.

`tools/condense/spec_reentry_scaffold.py` creates a deterministic, dependency-
checked matrix with stable cell IDs. It is deliberately non-executing and marks
every cell `deferred`. Its phases are:

| Phase | Question |
|---|---|
| P0 | Is B=1..8 TQ batched verification exactly equal to TQ single-token greedy for every TQ runtime mode? |
| P1 | What is the measured verifier curve for stored, compact, hashed, and computed TQ serving? |
| P2 | Do exact/free user-ngram, suffix, or retrieval proposers clear the cost gate on held-out code, prose, and tool JSON? |
| P3 | What does a conventional 0.5B AR draft control do as draft precision moves from Q4 through TQ3/TQ2/TQ1? |
| P4 | Can P-EAGLE-style parallel MTP, DFlash-style block diffusion, or SpecBlock-style path-dependent blocks trained on the served TQ/Doctor distribution beat the AR control? |
| P5 | Can a Metal token-tree transaction produce exact commits cheaply enough to matter? |
| P6 | Which admitted proposer composes with each TQ hardware interpretation on accepted tokens/s, latency, joules, bytes, rejected work, and memory? |

Useful no-GPU commands:

```sh
python3.12 tools/condense/spec_reentry_scaffold.py --selftest
python3.12 tools/condense/spec_reentry_scaffold.py --plan 7B
python3.12 tools/condense/spec_reentry_scaffold.py --status
```

`--status` is read-only and reports active heavy owners. `--write-plan` can later
checkpoint a matrix, but there is intentionally no execute command.

Every receipt name in that matrix now has an executable payload contract in
`tools/condense/spec_receipt_contract.py`. It enforces B=1..8 parity/curve rows,
zero skips and mismatches, all-linear TQ GPU coverage, named target/verifier
identity, physical counter sources for the curve, workload coverage, minimum
corpus sizes, charged lookup/miss cost, zero placeholder tokens, exact tree
rollback, and the 1.10 per-workload speedup lower bound. Completed speculative
evidence is validated as an outer canonical Appendix receipt plus its
schema-specific payload.

P0/P1 now have an executable but lease-gated adapter. The compiled
`hawking-tq-spec-probe` accepts explicit GGUF, TQ, and token-prompt files; fails
when any is absent; requires all-linear GPU TQ coverage; generates the
single-token TQ oracle once per prompt; and verifies at least 20 prompts x 256
tokens for every B=1..8 with zero skips. `spec_tq_runner.py` rechecks the heavy
lease and pressure/swap/thermal state, binds hashes and source commit, and only
finalizes parity plus verifier-curve receipts after a physical energy/GPU-time/
byte counter bundle is bound to the raw run. No proposer is executed by this
adapter.

The corpus prompt file is intentionally simple and tokenizer-bound. Each prompt
may carry either raw text (tokenized by the loaded target) or pre-tokenized IDs:

```json
{
  "schema": "hawking.spec_token_prompts.v1",
  "tokenizer_sha256": "<tokenizer.json hash, or GGUF hash for embedded tokenizer>",
  "prompts": [
    {"id": "corpus-case-0001", "text": "..."},
    {"id": "corpus-case-0002", "token_ids": [1, 2, 3]}
  ]
}
```

Exactly one of `text` or `token_ids` is allowed per row. This lets the frozen
corpus remain the source without trusting a foreign tokenizer.

## Research synthesis

The research changes priorities, not the admission bar:

- Model-free proposals deserve the first oracle. REST reports retrieval drafts
  without another model, while SuffixDecoding and SAM Decoding turn generated or
  reference text into exact suffix structures
  ([REST](https://arxiv.org/abs/2311.08252),
  [SuffixDecoding](https://arxiv.org/abs/2411.04975),
  [SAM Decoding](https://arxiv.org/abs/2411.10666)). Hawking already owns most of
  these primitives. Their published wins do not override Hawking's prior local
  losses; they justify workload-specific P2 cells and delayed exact shadow
  traces before any target verification cost.
- EAGLE-3 replaces feature prediction with direct token prediction and fuses
  multiple target-layer features
  ([EAGLE-3](https://arxiv.org/abs/2503.01840)). Hawking's current single hidden
  tap is therefore not a faithful modern learned-drafter input. P4 now requires
  early/middle/last capture from the **served TQ target**, with capture bytes and
  synchronization charged.
- P-EAGLE generates multiple depths in one pass and reports a 1.10–1.36x gain
  over autoregressive EAGLE-3 in its evaluated large-model setups
  ([P-EAGLE](https://arxiv.org/abs/2602.01469)). That modest incremental range is
  exactly why Hawking's 1.10 lower-confidence gate and precision sweep matter: a
  conventional dense parallel head can lose its entire margin to unified-memory
  traffic and capture overhead.
- DFlash uses block diffusion for parallel drafting, while the newer SpecBlock
  proposal reintroduces path dependence through small iterative blocks and a
  learned rank head
  ([DFlash](https://arxiv.org/abs/2602.06036),
  [SpecBlock](https://arxiv.org/abs/2605.07243)). These are P4 research controls,
  not reasons to replace the current zero-token stub prematurely. One-pass
  latency, acceptance decay by position, training complexity, and draft-state
  memory must be separated.
- Sequoia treats verification time as a hardware-dependent function and optimizes
  tree shape against it
  ([Sequoia](https://arxiv.org/abs/2402.12374)). That directly supports the new
  injectable verifier curve: tree width/depth must be solved from M3 Ultra
  receipts, not copied from an NVIDIA paper.
- MagicDec shows that the relevant bottleneck changes with context length and
  batch size and uses a sparse draft KV cache for long-context regimes
  ([MagicDec](https://arxiv.org/abs/2408.11049)). The Hawking matrix should
  stratify P1/P3/P4 by short, medium, and long context after the first B=1..8
  gate; a single average verifier curve is insufficient.

The most promising Hawking-specific combination is consequently not “load a
smaller dense model and hope.” It is: exact suffix/retrieval first; then a
quantized one-pass or small-block head conditioned on a few charged TQ target
features; then a hardware-shaped tree/line verifier that consumes the same TQ
weights as greedy decode. Each layer survives only if its incremental accepted
tokens are worth its incremental bytes and synchronization.

## Required execution and implementation order after the heavy run

1. Freeze the corpus, derive a hash-bound real token-prompt set, build both
   release probes, and run the implemented non-skipping 20 x 256 parity adapter
   over B=1..8 for each TQ runtime policy.
2. Bind physical counters, finalize parity and proposer-free verifier curves,
   and feed their conservative bounds
   into `VerifierCostCurve`; do not tune speculation using the historical array.
3. Implement delayed exact shadow evaluation for cheap proposers. Store a
   proposed line keyed by token position, compare it with later committed target
   tokens, then call `record_shadow`. Do not run target verification merely to
   observe a disabled arm.
4. Register retrieval only behind the same exactness, timing, and workload gates.
5. Use the 0.5B AR model strictly as a control. Prove tokenizer/vocabulary
   identity and charge dual residency, draft synchronization, and rejected work.
6. Replace the zero-token parallel stub only after an offline trace says a
   one-forward head can clear the 1.10 speedup lower bound in every target
   workload. Train against the actual served target distribution.
7. Compare P-EAGLE-style, block-diffusion, and block-iterative heads at matched
   resident bytes and measured draft latency. Include context-length strata and
   acceptance-by-position, not only mean accepted length.
8. Implement tree verification only as an exact commit transaction with explicit
   KV branch/rollback semantics; compare it to the best linear policy, not to a
   weak baseline.
9. Add checkpointed P2-P6 runners only after their prerequisites. They must use
   the same shared lease, hash/source bindings, pressure/swap tripwires, atomic
   evidence, and deferred-on-skip semantics already implemented for P0/P1.

## Two-ply composition rule

The FLOPS/TQ appendix changes the denominator of speculative decoding. A cheaper
B-token target pass reduces the acceptance length needed to win; a slower
computed-codebook mode raises it. Conversely, accepting several tokens per
target pass amortizes TQ payload, metadata, RHT, and outlier traffic over useful
tokens. P6 therefore measures combinations rather than crowning a TQ kernel and a
proposer independently.

The final unit is an exact token-commit transaction:

1. propose from local/retrieved/learned state;
2. verify multiple conditional positions against the same served TQ bytes;
3. commit the longest target-confirmed prefix and its KV state;
4. account for all bytes, joules, synchronization, and rejected branches.

That attaches speed research to the condenser ladder without redefining it.
