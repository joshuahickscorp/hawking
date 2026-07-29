# Temporal Gravity GLM live-token path into HIDE

Repair the serving seam that prevents fast GLM token decisions from reaching
HIDE and forces every GLM request out of continuous batching. Work in an
isolated worktree with source-body-free fixtures. Do not read real weights,
run a heavy benchmark, touch MOP, flip `HIDE_KERNEL_TURN`, or claim a TG/HIDE
latency promotion.

## Current evidence to verify

Trace the current path:

`HIDE SubmitTurn -> hawking serve -> /v1/hawking/generate -> BatchDriver ->
single-stream fallback -> GravityEngine::generate -> GravityGlmGpu`.

Verify before editing:

- `GravityModel::prefill_slot` is intentionally unimplemented, so batch-1 GLM
  serve falls back to `Engine::generate`;
- the GPU lm-head token-only path returns `GlmTrace::sample_token`;
- the direct TPS runner consumes that token;
- `GravityEngine` discards the trace and invokes `Sampler` on returned logits;
- an empty logit slice can deterministically produce token zero;
- `healthz.fallback_present=false` describes fallback-model presence, not
  serving-path fallback.

If live source contradicts any premise, report the contradiction and implement
the smallest fix that satisfies the behavior below.

## 1. One exact token-output contract

Define one explicit output contract between `GravityGlmGpu` and
`GravityEngine`:

- full-logit mode returns complete logits and the engine samples them once;
- token-only mode returns no fake logits and a typed, validated committed token
  decision plus its trace;
- empty logits without a valid typed token decision is an error, never token
  zero;
- if both logits and a trace token are present for a parity/debug path, require
  greedy/full-logit argmax to equal the trace token before committing;
- the committed token is used consistently for output, next-token state, stop
  handling, SSE publication, accounting, and persistence;
- no downstream layer resamples or substitutes a token.

Preserve Numeric Parity V2.1. Bind the exact artifact, runtime flags, decision
mode, token, and trace identity. Token-only output remains unavailable when its
parity prerequisites are not met.

Add a hard invariant at the `Sampler` boundary: an empty candidate/logit input
cannot silently yield any token.

This invariant also applies to the batch trait. The generic
`Engine::forward_multiseq_greedy_tokens` implementation must refuse an empty
logit vector instead of `unwrap_or(0)`. `GravityEngine` must provide a typed
override that preserves and commits the GLM trace decision. Test the real
`BatchDriver::decode_ready_once` path with an empty GLM output; it must return a
typed refusal and never token zero.

## 2. Real batch-1 Gravity prefill/decode path

Implement the smallest correct `GravityModel` batch-1 support required by
serve's default batch size:

- replace or extend the bare-`u32` prefill/decode trait results with typed
  results carrying token ID, decision mode/identity, trace evidence, and
  terminal/error provenance;
- `prefill_slot` consumes the exact rendered/tokenized prompt and establishes
  the same model/KV/state as the direct engine path;
- the next decode operation returns the same greedy token as the direct
  `GravityEngine` path under identical artifact, flags, prompt, seed, and
  context;
- slot reset/cancel/error paths release or invalidate state deterministically;
- batch size greater than one remains explicitly unsupported unless fully
  implemented and tested;
- unsupported conditions refuse or take a separately receipted fallback before
  token publication.

Freeze token-only batch eligibility to exact greedy-compatible settings.
Because prefill currently receives no sampling policy, either extend the typed
API so first-token semantics receive and apply the exact request policy, or
refuse every non-exact-greedy GLM batch request. A raw greedy prefill token may
not seed a temperature/top-p/repetition-penalty request.

The ordinary batch-1 serve path must no longer use the generic single-stream
fallback merely because `prefill_slot` is absent.

Production GLM/HIDE requests fail closed on any batch admission, prefill,
decode, state, tokenizer, or receipt error. Do not catch every prefill error
and rerun `Engine::generate`. Any diagnostic fallback must be explicit opt-in,
per-request, separately receipted, and terminally disqualifying for TG/HIDE
qualification. One slot failure must not silently downgrade an entire cohort.
Decode failure must notify and release/poison the affected slot rather than
being logged and retried forever.

Native initial and deferred admission must distinguish `Err` from `Ok(None)`.
Patterns equivalent to `.ok().flatten()` are forbidden. A native
tokenizer/engine refusal emits a typed SSE error, releases state, and never
enters the capacity wait queue or fallback path.

## 3. Honest serve/HIDE evidence

Expose machine-readable per-request evidence:

- exact artifact/index hash and runtime build identity;
- complete resolved GLM flags;
- `engine_path` with a canonical value for the batch-1 path;
- separate fallback-model and serving-path fallback counters/reasons;
- token decision mode (`full_logits` or `gpu_token_trace`);
- trace/policy seal and committed token trace hash;
- queue, prefill, TTFT, and inter-token decode durations separately;
- prompt/tokenizer/seed identity and output-token hash.

Do not call time before prefill finishes "decode-only." HIDE must receive and
persist the run/request evidence needed to prove which path produced the turn.
This task may wire the data path and use fixture tests; it cannot produce a
capable-model HIDE receipt without an approved artifact.

All evidence originates from the raw per-request engine execution record.
Requested settings, environment reconstruction, global counters, or evidence
from an earlier request cannot prove the actual engine path, resolved flags,
decision mode, fallback status, token IDs/hash, or counter deltas.

Replace the untyped `SlotToken = Result<String, ()>` transport with versioned
typed channel/SSE events equivalent to:

- `Token { id, text, decision_evidence }`;
- `Done { reason, generation_receipt }`;
- `Error { typed_reason, partial_evidence }`.

The terminal reason distinguishes EOS, stop string, max tokens, cancellation,
refusal, and error. An error, cancellation, decoder failure, missing evidence,
or disconnect cannot emit a passing receipt or ordinary `[DONE]`.

Extend the HIDE protocol/runtime schema so it can carry the versioned
generation evidence. Exactly one complete terminal receipt is required before
turn success. Missing, malformed, duplicate, out-of-order, unrecognized, or
zero-filled synthetic terminal evidence refuses. `HttpModelProvider` may not
invent zero statistics on missing receipt, and `run_turn_core` must durably
persist the complete receipt keyed to HIDE run ID and serve request ID.

Freeze monotonic timing boundaries:

- request acceptance to admission/queue completion;
- prefill start/end;
- first committed-token TTFT;
- each engine decode interval;
- SSE publish intervals;
- total server wall time;
- separately labelled HIDE-observed transport time.

Bind every phase to the stable request/run ID. Recompute phase aggregates from
samples and require arithmetic reconciliation with total wall time; queued
requests retain their original start and identity.

Persist a durable HIDE generation-start record followed by exactly one terminal
success, refusal, or abort receipt. Each binds artifact/build/request/output
hashes and predecessor identity. Restart/replay must leave interrupted turns
visibly incomplete or aborted, never convert them into unreceipted successes.

## Tests

Add source-body-free deterministic tests for:

- empty logits plus no trace refuses instead of returning token zero;
- the generic multi-sequence greedy helper and Gravity override both refuse
  empty untyped output; `BatchDriver::decode_ready_once` never returns token
  zero from it;
- full logits commit one greedy token;
- token-only trace commits that exact token;
- dual evidence agrees; disagreement refuses;
- committed token reaches stop handling, SSE/output, next-token state, and
  persistence unchanged;
- direct engine and batch-1 `prefill_slot`/decode parity for prefill plus
  several steps;
- exact prompt/history preservation;
- slot reset, cancellation, injected error, and repeated request isolation;
- non-greedy/full-logit first-token policy either executes with exact semantics
  through the typed API or refuses before token publication;
- prefill/decode/tokenizer/state errors fail closed without automatic
  single-stream fallback, cohort downgrade, retry loop, or wait-queue entry;
- native immediate and deferred admission errors remain errors rather than
  `Ok(None)`;
- `engine_path`, flags, hashes, timing phases, decision mode, and both fallback
  counters are present and internally consistent;
- raw per-request evidence cannot be substituted by environment flags, request
  settings, or global counters;
- typed Token/Done/Error SSE ordering, exactly one terminal receipt, and
  missing/malformed/duplicate/out-of-order receipt refusal;
- cross-token stop strings, no EOS token publication, `max_tokens=0`,
  tokenizer-decode failure, and disconnect during the first token; after any
  terminal condition no token/count/passing receipt may appear;
- timing boundaries and phase/total reconciliation;
- HIDE start plus terminal success/refusal/abort durability, crash recovery,
  and replay linkage;
- the real local HTTP/SSE route uses batch-1 rather than the serving fallback;
- HIDE's model-provider fixture consumes and persists the returned evidence.

Tests must assert they exercised the intended path. A fake server, skipped
model test, or healthz fallback Boolean alone is not evidence.

## Authorized files

Change only the smallest necessary subset of:

- `crates/hawking-core/src/model/engine.rs`;
- `crates/hawking-core/src/model/gravity_engine.rs`;
- directly corresponding hawking-core tests;
- batch driver/scheduler types and directly corresponding tests;
- `crates/hawking-serve/src/lib.rs`;
- `crates/hawking-serve/src/http.rs`;
- directly corresponding serve tests;
- `crates/hide-core/src/runtime.rs` or the equivalent canonical protocol/event
  type needed for generation evidence;
- HIDE model-provider/host evidence plumbing and directly corresponding tests,
  only if needed to carry the already-produced receipt.

Do not modify GLM kernels, model artifacts, capability/status/launch receipts,
authorization fences, or unrelated HIDE surfaces.

## Acceptance

Run targeted and affected Rust tests, formatting/lints available in the
repository, deterministic replay, and `git diff --check`. Report exact
files/hashes, token-contract semantics, engine paths exercised, fallback
counters, fixture limitations, and the remaining capable-artifact live gate.

State explicitly that no real model, TG milestone, HIDE production promotion,
or MOP action occurred and all fences remain false.
