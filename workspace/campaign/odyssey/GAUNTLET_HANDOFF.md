# Gravity gauntlet handoff

Recorded 2026-09-05 from the shared checkout. This is a continuation state;
historical receipts were not rewritten.

## State reconstruction

- HEAD: `baa262e489516fff457f0673db1d7adc5f8542e9`
- Branch: `main`, ahead of `origin/main` by 116 commits.
- Worktree: dirty. At handoff: 204 deleted entries, 6 modified tracked files,
  and the new gauntlet artifacts. The large deletion set and the two unrelated
  modified files (`hcli/tests/test_acceptance_integrity.py` and
  `receipts/future/MODELLAKE_EVENTS.json`) predate this gauntlet work and were
  preserved.
- No patient-runner or Gravity search process is active in this checkout.
  `modellake_watch` PID 4183 and the resident `mlx_lm.server` PID 43912 are
  alive. HCLI PID 84230 is working in `.worktrees/ascension` on an unrelated
  G002 attribution mission; it was not touched.
- `ODYSSEY_STATE.json`: 14 patients, 6 READY, 8 ACQUIRING, 2 marked on disk;
  79 work entries VERIFIED, 11 READY, 1 BLOCKED. The queue is stale relative
  to the receipt corpus; it is not used as proof of candidate execution.
- `HCLI_LEDGER.json` does not exist. Existing law/scar authorities are the
  24-entry `receipts/future/ODYSSEY2_LAW_STORE.json`, 9 inferred Gravity rules
  in `GRAVITY_RULEBASE.json`, 7 campaign scars, and 4 autonomy scars.

## Gap and implementation

The old `tools/odyssey_patient_runner.py --gravity SPEC` still evaluates one
mix at a time. It is an evaluator, not a target-seeking search. The new
HCLI-owned `hcli/gravity_gauntlet.py` supplies the missing bounded loop:

`candidate -> measured receipt -> evidence-guided next candidate -> checkpoint`

It records candidate identity, parent, mutation, expected effect, persistent
bytes, complete EBPW, capability/execution/verifier state, timing/resources,
reject reasons, and NR release evidence. Checkpoints are atomic and refuse a
changed candidate space on resume. The callable HCLI surface is
`odyssey.gravity_gauntlet`, registered as `odyssey.gravity_gauntlet`; mutation
requires explicit confirmation.

Verifier rules now enforced by the loop:

- `TARGET_HIT` requires measured complete EBPW `<= 1`, complete accounting,
  capability success, complete execution, an independent verifier, measured
  magnitude adequacy, and validated utilization.
- `PROVEN_UNABLE` requires an explicit bound with limiting mechanism, measured
  evidence, assumptions, search region, and reopen condition.
- Otherwise the exact terminal disposition is `BUDGET_EXHAUSTED`; it is never a
  pass.
- The `0.01W` negative control is rejected even when direction similarity is
  near one. Missing magnitude evidence is unvalidated, not a pass.

## Canary results

Three sequential canaries reused existing measured patient receipts and wrote
their own checkpoints under `workspace/campaign/odyssey/gauntlets/`:

| specimen | role | search | best complete EBPW | terminal |
|---|---|---:|---:|---|
| O001 | small/cheap hybrid | q4-g64-attn-mlp -> q3-g64-attn-mlp | 5.6745 | BUDGET_EXHAUSTED |
| O005 | current relevant MoE | q3-g32-experts -> q2-g32-experts | 3.0759 | BUDGET_EXHAUSTED |
| O003 | materially different multimodal MoE | q3-g32-experts -> q2-g32-experts | 3.0639 | BUDGET_EXHAUSTED |

The O003 search demonstrates learning: complete EBPW improved from `3.9412` to
`3.0639` while the external Doctor signal remained `CANDIDATE_PASS`. This is
not a complete-system target hit: all three reused receipts have
`execution_complete=false`, `verifier_independent=false`, and no measured
magnitude/utilization field. No candidate was promoted to NX.

Measured Tier-1 economics are in
`workspace/campaign/odyssey/ODYSSEY_COST_MODEL.json`: 6 candidate evaluations,
14.7372 s mean per candidate, and 29.4743 s mean per two-candidate search.
The 14-roster extrapolation is explicitly derived and incomplete; acquisition,
static census, deep search, verification, offload, and hardware utilization are
not included.

## Current truth and blockers

`FLASH_COMPLETE_EBPW_LE_1` is mechanism-implemented but currently blocked:
the complete incumbent is `3.139300850311054`, against the target `<= 1`.
Sub-bit complete-system Gravity is **NOT ACHIEVED**. Historical receipts remain
unchanged; current derived state is not allowed to imply otherwise.

The ModelLake acquisition path also refuses O010 in dry-run because measured
free space is 85.5 GiB while the next specimen needs 205.8 GiB plus a 45 GiB
floor. The existing `hcli/test_odyssey.py` dry-run expectation for that path is
currently red for this real disk-hold reason; this gauntlet change did not
alter it.

## Next WorkUnit

Open a new budgeted O003 Gravity episode seeded from
`O003-q2-g32-experts`, with explicit additional candidates such as
`q2-g64-experts` and a sensitivity-driven mixed family. Before any target claim,
add independent execution/capability verification, magnitude adequacy, and
validated utilization fields to the evaluator receipt. If a real bound becomes
available, submit it through `finalize_proven_unable`; otherwise preserve the
truthful terminal status and continue with a new checkpoint rather than editing
the closed history.

## Shadow-supervision continuation — 2026-09-05

The full HCLI WorkUnit specification is preserved at
`workspace/campaign/odyssey/HCLI_O003_WORKUNIT.md`; the compact dispatch prompt
used to avoid context duplication is
`workspace/campaign/odyssey/HCLI_O003_COMPACT_WORKUNIT.md`.

Two HCLI dispatch attempts were made and are recorded as transport failures,
not Odyssey scientific dispositions:

- Mission `2ce220f8-ffb7-437e-bfba-a95dd684c3a0` used the local native resident.
  The resident staged `sealed-3.14` but emitted no model-call completion or
  tool progress for more than 30 minutes; the duplicate resident was then
  interrupted and the mission durably recorded `cancelled`.
- Mission `2af53ce5-f532-4fe2-8775-5871cb013c60` used the existing MLX endpoint
  at `127.0.0.1:9999`. G001 timed out after 600 seconds with a measured
  9,806-token prompt; the endpoint health route was responsive but generation
  returned no result.
- Compact retry mission `f69837fd-c4d5-4d07-8558-4b170a1c2f3b` reduced the
  first model prompt to 7,137 measured tokens and still timed out at 600
  seconds. A direct 30-second `/v1/chat/completions` probe returned HTTP 000
  with zero bytes, confirming an MLX generation-service failure rather than a
  context-overflow classification.

The separate HCLI task in `.worktrees/ascension` remains active on an unrelated
G002 attribution mission and was not steered or modified. Do not treat its
native resident as available for O003 until that mission reaches a terminal
state. Do not claim any of these dispatches as target-seeking Gravity or as new
O003 evidence. The next safe action is to resume the compact O003 WorkUnit
through a functioning HCLI cognition backend, then execute the Doctor-directed
representation discriminators and deep search described above.

The MLX backend root cause was subsequently confirmed: PID 43912's recorded
model path `/private/var/folders/yc/1c930rns2rz1_trh4xcd9_m00000gq/T/tmpk0uglm2s/4bit`
no longer exists. Restarting the exact command therefore produced no server;
no authoritative replacement model artifact was found in the local model/cache
roots. The endpoint's health process was stale, not a usable HCLI cognition
backend. Reacquisition or a valid resident handoff is required before another
HCLI O003 dispatch.

The main-worktree runner now records conversion economics in each Gravity
receipt under `gravity_conversion`: whether the mlx artifact was reused,
whether `mlx_lm.convert` actually ran, its measured converter wall time, and
the total conversion-lifecycle wall time. This closes the previously observed
unmeasured `convert_gravity()` build phase. It is measurement-only and does
not create new O003 evidence or change the HCLI execution boundary.

## Explicit native continuation dispatched — 2026-09-05

The sealed native profile was probed independently with a bounded text-only
request and returned `READY` in 43 seconds. The full O003 WorkUnit was then
dispatched through HCLI using
`hcli/hawking-native.sealed-3.14.json` rather than the stale default
llama-server/MLX routes. Mission:
`4f19d74c-0492-4524-af4a-2ca7a95bacde`.

At the latest observation the mission is `running`, with `G001` in flight and
zero accepted units. Its native resident is actively inside constrained Metal
generation; this is live transport/model work, not a terminal timeout. No new
O003 receipt or scientific result is claimed yet. The unrelated ascension
mission and its resident remain untouched.

After approximately 1,022 seconds with no model-call receipt, accepted unit,
or O003 evidence, the explicit-native dispatch was cancelled as a transport
recovery action. Its resident was stopped; the unrelated ascension resident
was not touched. The mission log records shutdown while its last durable state
still says `running`/G001, so this run must be treated as an inconclusive
transport interruption, never as an Odyssey disposition. The next dispatch is
the same O003 WorkUnit through the compact prompt and explicit native profile.

That compact recovery is now HCLI mission
`7e36ebb4-4850-459a-a87a-bf3c5b57f5f9`. It is `running` with G001 in flight;
the new native resident is actively executing constrained Metal generation.
No acceptance, model-call receipt, or O003 science has been observed yet.

Subsequent supervision observed the O003 resident complete an active
`generate_constrained`/Metal phase, after which the resident became idle while
the HCLI parent/reader remained waiting. The mission is still authoritative
`running` with G001 in flight and no accepted unit, no model-call receipt, and
no O003 science. Treat this as a live transport/integration wait until HCLI
reaches a terminal state; do not restart or convert it into an Odyssey
disposition solely because observation has been slow.

The compact run was later cancelled after approximately 483 seconds without
a receipt. A final bounded HCLI recovery is now running as mission
`f38f3802-f1fa-404d-ac69-805dc305ff04`, using the same compact O003 WorkUnit,
the explicit sealed native profile, and `HCLI_MODEL_TOKENS=512` to bound the
cognition response. It is still `running` with G001 in flight; no scientific
evidence has been promoted from any interrupted run.

## HCLI regression discriminator — 2026-09-05

The final 512-token O003 control did not stall in the native transport. Its
durable receipt is `.hcli/receipts/aa8ba4bd-83f9-4855-b145-d38683a836a7.json`.
It completed native generation and returned five replies to HCLI, but every
reply hit the effective 256-token grant (`max_tokens=512`,
`max_new_tokens_granted=256`, `stop_reason=budget`). The structured contract
received those replies and exhausted its three retries while the prompt grew
from 2,358 to 4,816 tokens. The first divergent boundary is therefore the
generation ceiling/structured contract, not pipe EOF or resident liveness.

The exact matched WorkUnit is
`workspace/campaign/odyssey/HCLI_BOUNDARY_AB_WORKUNIT.md`. With the same
explicit profile, tokenizer, renderer, resident binary, grammar mode, context,
and no tools:

- A, last-known-good token environment (no `HCLI_MODEL_TOKENS`), completed in
  `.hcli/receipts/040886bd-cd62-4d41-8172-a5c19944ac93.json`; it was granted
  1,446 tokens and completed the structured answer in 169 tokens.
- B, current Odyssey environment (`HCLI_MODEL_TOKENS=512`), completed in
  `.hcli/receipts/e6b5b79a-067d-4d1b-a15d-c8f52ae73236.json`; it was granted
  256 tokens and completed the same answer in 169 tokens.

Both A and B crossed request construction, submission, resident receipt,
prefill, first decode token, last decode token, termination, response-byte
emission, HCLI parent receive, structured-parser close, and receipt write.
Therefore the minimal current source/runtime path is not regressed. The O003
contract is causal at its current budget: its larger tool/measurement answer
does not fit the 256-token grant, retries append parser errors, and the prompt
then expands. Do not shrink to 256/128; restore the last-known-good effective
grant and resume the exact O003 WorkUnit.

The opt-in trace instrumentation is in `hcli/hawking_native.py`,
`hcli/engine.py`, `crates/hawking-core/src/model/qwen38_hybrid_decode.rs`, and
`crates/hawking-core/examples/ascension_qwen38_resident.rs`. The current
resident binary was rebuilt with that instrumentation. Historical accepted
receipts bind the same profile, artifact, tokenizer, resident binary hash,
grammar mode, resident topology, fusion environment, and `runtime_env={}`;
they do not carry a git source commit, so no historical source commit is
invented here.

The `f38f3802-f1fa-404d-ac69-805dc305ff04` process was stopped after G001
failed and the mission began an automatic G002 repair; that repair was
interrupted before any work was accepted. Its stale mission state must not be
treated as an active scientific continuation.

## Exact O003 resumed after discriminator — 2026-09-05

The exact full `HCLI_O003_WORKUNIT.md` was resumed through the restored
last-known-good effective grant (no `HCLI_MODEL_TOKENS` override), with the
new boundary trace enabled. The first request used 2,786 prompt tokens and a
1,446-token grant, completed 1,261 generated tokens, crossed parent receive,
and closed the structured parser. Subsequent requests likewise crossed the
native and HCLI boundaries; several parser retries recovered on attempt 2.
The trace recorded nine model requests beginning before the outer 1,800-second
bound; the ninth was still decoding when the wrapper timed out. No WorkUnit
acceptance or scientific O003 receipt was promoted (`accepted_count=0`).

This is partial HCLI/O003 progress and transport evidence only. The mission
state is stale `running`/G001 because the outer timeout interrupted it, and
the orphaned resident was stopped by its exact PID. The permanent unrelated
`.worktrees/ascension` resident was not targeted. Resume from the durable
O003 mission/checkpoint only after the next controlled dispatch; do not treat
the partial model turns as Doctor or Gravity results.

## Exact O003 durable recovery after the A/B discriminator — 2026-09-05

The exact mission `becb8728-a947-40b6-85aa-fa8ad0387e7d` was recovered through
HCLI `AgentOS.recover_mission()`/`continue_mission()` rather than creating a
new mission. It used the explicit sealed-3.14 native profile, no
`HCLI_MODEL_TOKENS` override, and boundary trace
`/tmp/hcli-boundary-O003-resume-20260905-2.jsonl`. Recovery preserved the
mission ID and reclassified the orphaned G001 as interrupted/rerunnable.

The run produced complete measured boundaries for many native requests. For
example, prompt sizes 2,788, 5,494, 5,287, 3,983, 4,259, 4,333, 5,364,
4,446, 5,536, 5,610, and 5,572 were received by the resident; each observed
generation emitted response bytes, reached HCLI parent receive, and generally
closed the structured parser. Cold prefill ranged from 79.87 s to 229.02 s;
same-context prefix reuse was measured at 2.71 s. This is direct prefill/decode
evidence, not heartbeat or process-aliveness evidence.

The recovered graph wrote durable failures for the exact WorkUnit contract:

- G002 failed after native generation completed at 784 tokens because
  `tool_calls[7].arguments[1]` was a string where the structured schema
  required an object.
- G003 failed after repeated 1,365-token budget completions without closing
  the JSON object.
- G004 failed after a 1,405-token budget completion without closing the JSON
  object.

Those receipts are parser/structured-contract evidence, not transport failures;
the trace contains response-byte emission, HCLI parent receive, parser end, and
receipt write for each. HCLI continued its durable repair graph, but the outer
3,600-second bound expired while the next recovered request was in prefill.
The mission remains authoritative `running` with `accepted_count=0`; no
Doctor, Gravity, or O003 scientific acceptance is claimed. The exact native
resident orphan from this run was stopped by PID after the timeout. All prior
stalled receipts remain preserved as transport/runtime evidence.

## O003 latency mutation: local structured repair and live context measurement — 2026-09-05

The first latency mutation targets a measured O003 waste, not synthetic
prefill: `StructuredOutputContract.enforce()` previously deep-copied the full
working Odyssey payload and appended each parser error for every retry. The
repair call therefore replayed the complete goal/evidence/context and could
consume the same cold prefill that caused the failure. `hcli/backends.py` now
keeps the stable system/developer prefix, replaces the mutable user turn with
the schema instruction, exact bounded parser error, and a head/tail fragment
of the rejected reply, and caps the repair completion at 1,024 tokens. The
new regression test proves the retry payload is materially smaller while
preserving the actionable failure and stable prefix. Targeted structured-output
tests pass (`12` tests).

The exact O003 mission was then recovered in place with no
`HCLI_MODEL_TOKENS` override and trace
`/tmp/hcli-boundary-O003-latency-b.jsonl`. The first live turn measured:

- 2,959 prompt tokens; request constructed/submitted and resident received;
- 95.918 s prefill; first decode token immediately after `prefill_end`;
- 302 generated tokens; `constraint_done` termination;
- 181,831 response bytes; HCLI parent receive; structured parser success on
  attempt 1.

The next O003 cognition turn grew to 4,601 prompt tokens and reached resident
`prefill_begin`, but had not reached `prefill_end` or first decode after more
than 120 s. The owned run was stopped at that measured boundary; no transport
EOF failure was inferred and no scientific acceptance was fabricated. The
authoritative mission remains `running`, `accepted_count=0`, with the current
orphan reclassified on the next HCLI recovery. This run makes context growth
between live O003 decisions the next optimization frontier; it does not reopen
generic KV work or change the O003 scientific objective.

The same live trace also showed the round-dependent compact tool catalog was
part of the request builder's mutable prefix: its focus included tool names
from prior observations, so the catalog could change between cognition turns.
`hcli/engine.py` now focuses that catalog only on the stable WorkUnit prompt;
observations remain the final mutable suffix. The added regression test proves
the pre-observation catalog bytes are identical across an observation round.
The combined targeted suite passes (`29` tests). This is a prefix-stability
mutation for the O003 loop; it does not claim a prefill gain yet because the
current `grammar=json` resident path records constrained generation as a reset.
The next live O003 trace must measure whether the stable request shape changes
the actual reuse/prefill boundary before promoting this as a latency Law.

The post-change exact recovery used trace
`/tmp/hcli-boundary-O003-latency-c.jsonl` and the same mission/profile. Its
first live turn again closed normally: 2,963 prompt tokens, 96.480 s prefill,
301 generated tokens, `constraint_done`, 181,978 response bytes, HCLI parent
receive, and structured-parser success on attempt 1. The following request
was 4,605 prompt tokens and reached resident `prefill_begin` before the
bounded 140-second wrapper stopped; the orphan resident was stopped by its
exact PID. This is an A/B safety result for the prefix mutation, not a claim
of speedup: constrained `grammar=json` still resets the native generation
path, and no WorkUnit was accepted (`accepted_count=0`).

## O003 live tool-history reducer — 2026-09-05

The next exact recovery isolated a second, concrete O003 path defect. The
first model turn had completed, but its tool-follow-up was assembled through
`history`, while `_fit_payload_to_budget()` only reduced evidence,
`context_memory`, and legacy trailing observations. The resulting live
failure was `demand 25607 exceeds per-request ctx 8192`; its traceback points
to the second `_call_model()` round, not to native transport. The change in
`hcli/engine.py` adds history to that same reducer: it keeps the newest
assistant/observation pair when possible, then the newest observation, and
reports dropped history in the reduction record. A focused regression test
forces an oversized tool history and passes. The combined targeted context,
tool-loop, structured-output, and observation tests pass (`33` tests).

The exact mission `becb8728-a947-40b6-85aa-fa8ad0387e7d` was recovered in
place with trace `/tmp/hcli-boundary-O003-history-reducer.jsonl`. This run
crossed all physical boundaries for the live repair sequence:

- the ordinary O003 decision was constructed at 3,865 estimated prompt
  tokens; the resident received 2,946 tokens, prefilling cold for 95.421 s;
- it decoded 1,446 tokens and terminated explicitly with `stop_reason=budget`,
  emitted 248,849 response bytes, reached HCLI parent receive, and entered
  local structured repair;
- the repair request was 2,798 estimated / 1,843 resident prompt tokens,
  prefilling for 52.687 s; it generated 177 tokens with
  `stop_reason=constraint_done`, emitted 113,428 bytes, reached parent
  receive, and closed the structured parser on attempt 2;
- the following ordinary O003 decision was 6,379 estimated / 4,852 resident
  prompt tokens and entered prefill. It was stopped at that measured boundary
  by bounded supervision; the parent then recorded `pipe_stdout_eof` during
  resident teardown. This is cancellation evidence, not a transport EOF
  failure: the prior two requests had already crossed parent receive and parser
  close.

The reducer defect is therefore repaired and demonstrated on the real O003
loop: the prior 25,607-token refusal is gone, and a malformed/budgeted turn's
repair is materially smaller. The remaining dominant cost is ordinary cold
O003 prefill plus oversized first-turn structured generation: in this run the
first 2,946-token resident prompt consumed 95.421 s and the model exhausted
the 1,446-token grant before the compact repair succeeded. Prefix telemetry for
these requests was `prefix_source=cold`, `prefix_reused_tokens=0`; this run
did not produce a same-WorkUnit accepted mutation and did not prove a prefix
hit. Mission state remains the same exact mission, `cancelled`,
`accepted_count=0`; all prior stalled receipts remain preserved as runtime /
transport evidence. The next O003 dispatch should use this reducer and target
the first-turn compact operation contract / actual same-turn prefix boundary,
not reopen generic KV science.

## O003 compact-action A/B — 2026-09-05

The next live mutation added only a worker-packet output constraint in
`hcli/goal.py`: one next verifiable action, no research-essay restatement, and
at most four distinct tool calls. It does not alter the O003 scientific
objective or the global result schema. The focused suite remained green (`41`
tests).

On the same exact mission/profile, trace
`/tmp/hcli-boundary-O003-compact-action.jsonl` measured a materially better
first turn: 2,938 resident prompt tokens, 95.103 s cold prefill, 439 generated
tokens, `constraint_done`, response bytes, parent receive, and parser close.
The preceding live shape had exhausted 1,446 generated tokens and required a
repair. This is a real O003 action-shape improvement, although no WorkUnit was
accepted in the bounded slice.

The next request was again 2,938 resident prompt tokens and entered
`prefill_begin` cold with `prefix_reused_tokens=0`; supervision stopped there
and recorded `pipe_stdout_eof` during teardown. This establishes the next
divergence as prefix/context continuity between successive O003 requests,
not model generation completion or parent transport. The exact mission
remains `cancelled`, `accepted_count=0`; no scientific result is promoted.

## O003 compact-history reducer and bounded recovery — 2026-09-05

The prior live history trace showed `history_messages=2` while the posted
payload still collapsed to `system,user`. The reducer was fitting the raw
assistant/tool-result pair, then dropping the entire pair when the O003 base
packet plus degraded-schema reserve left too little room. `hcli/engine.py`
now tries the newest raw pair, then a bounded newest pair (assistant content
<=1200 characters, observation content <=2400 characters), before dropping
history. The compact-history regression is covered by the context-reduction
suite; the focused context/tool/structured/observation/goal suite passes with
`42` tests.

The exact mission `becb8728-a947-40b6-85aa-fa8ad0387e7d` was resumed with the
existing sealed-3.14 native profile under trace
`/tmp/hcli-boundary-O003-history-compacted.jsonl`. Recovery selected G004
before G003 because of the persisted DAG state; this was still the exact O003
mission and its receipt is retained. The measured boundaries were:

- first request: 4,470 estimated / 3,762 resident prompt tokens; cold prefill
  `133.314 s`; first decode; `1,446` generated tokens; explicit `budget`
  termination after `93.394 s` decode; `295,144` response bytes; HCLI parent
  receive;
- local structured repair: 2,396 resident prompt tokens; `73.663 s`
  prefill; `35` tokens; `constraint_done`; parent receive;
- second repair attempt: `895` checkpoint-restored tokens, `818` fresh
  prefill tokens, `25.739 s` prefill, `35` tokens, `constraint_done`, parent
  receive, and parser exhaustion after three attempts with
  `missing required property 'tool_calls'`;
- receipt `6a346159-b0ed-4103-a44b-44eadb4c82de.json` was written as failed;
  no accepted scientific mutation was fabricated;
- the resident and parent were stopped cleanly after the bounded slice;
  mission state is `cancelled`, `accepted_count=0`.

This run closes the transport question again: native generation, response
bytes, parent receive, local repair, checkpoint restore, parser end, and
receipt write all occurred without a spontaneous pipe/EOF failure. The live
dominant costs are now first-turn cold prefill and a broad degraded structured
contract that exhausted its 1,446-token budget before producing a valid
tool-use envelope. The compact-history change is verified offline but was not
reached in this slice because the WorkUnit failed during structured parsing
before a successful tool round. The next O003 change should therefore make
the worker action/structured contract cheaper and more reliable, then rerun
the same bounded WorkUnit; do not reopen generic KV/context optimization.

## O003 prefill floor measurements and rejected compact-cap experiment — 2026-09-06

The exact mission `becb8728-a947-40b6-85aa-fa8ad0387e7d` was resumed through
the current native resident with `HAWKING_QWEN38_PREFILL_CHUNK=4` and complete
boundary tracing. The control crossed every physical transport boundary:
request construction/submission, resident receive, prefill end, first and last
decode, generation termination, response bytes, HCLI parent receive, and the
next repair request. No EOF or reader defect occurred.

The best current cold O003 measurement in this slice was trace
`/tmp/hcli-boundary-O003-agentic-compact-bounded.jsonl`:

- `3,169` resident prompt tokens;
- `83.103 s` prefill;
- `384` generated tokens in `20.238 s`;
- explicit `budget` termination and parent receipt;
- `0` prefix-reused/checkpoint-restored tokens.

The prior 768-token attempt measured `3,073` resident tokens and `79.405 s`
prefill. The earlier real repair measurement remains the reusable-prefix
control: `818` fresh tokens in `25.739 s` after `895` checkpoint-restored
tokens. Therefore `<30 s` is already real for the reused path, but not for a
cold 3K-token O003 prompt. At the current cold rate, approximate fresh-token
budgets for `30/10/5/1 s` are `<=1,140/380/190/38` respectively; these are
decision-suffix targets, not claims that the full cold prompt can meet them.

An agentic first-turn completion cap was experimentally lowered from the
historical effective grant to `768`, then `384`, with a compact schema. Both
reached their generation ceiling without closing JSON, so the experiment was
reverted. No acceptance or scientific result was promoted. The production
path retains the prior effective generation behavior; the failed cap traces
are runtime/structured-output evidence only.

The physical floor is now bounded by the resident's fused-k4 prefill command:
at the measured O003 rate, `1 s` requires roughly ten k4 commands / forty fresh
tokens, `5 s` roughly fifty commands / two hundred fresh tokens, and `10 s`
roughly one hundred commands / four hundred fresh tokens. The next valid
Odyssey optimization is to compile the real O003 stable prefix and mutable
decision suffix so ordinary turns enter this range; do not claim `10/5/1 s`
until a traced O003 request records those fresh-token and wall boundaries.

## O003 worker-envelope prefill A/B — 2026-09-06

The worker path now uses a compact agentic system envelope, compact degraded
structured-output instruction, stable minimal tool catalog, and a worker-only
inline evidence cache capped at `1,200` characters. The named evidence files
remain disk authority. The cap is applied around both evidence gathering and
`Engine.execute`; the regular/direct model path is unchanged. Focused
regression coverage is green: `57 passed`.

The exact mission was resumed with no evidence-budget override under trace
`/tmp/hcli-boundary-O003-worker-default-20260906.jsonl`:

- request estimate `1,556` tokens; resident prompt `1,142` tokens;
- resident prefill `22.266 s` (`51.2` resident tokens/s);
- first decode token followed immediately;
- the bounded run was stopped before completion because no parser/receipt
  boundary had appeared; accepted scientific work remains `0` in this slice.

This is the first verified cold/default O003 prefill under `30 s`, with
`7.734 s` of margin. A zero-inline-evidence A/B was effectively identical:
`1,134` resident tokens and `22.047 s`, proving that evidence bytes are no
longer the dominant floor. A chunk-8 A/B was also effectively identical:
`1,139` resident tokens and `22.161 s`; chunk size alone is not the next
frontier. A `HCLI_NO_TOOLS=1` probe was rejected immediately because it took
the fallback path and sent `4,849` resident tokens; it is not an optimization.

At the measured compact-path rate, approximate resident-token budgets are
`~1,536` for `30 s`, `~512` for `10 s`, `~256` for `5 s`, and `~51` for `1 s`.
The current fixed worker envelope is about `1,134` tokens, so `10/5/1 s` are
not yet demonstrated and cannot be claimed as the current floor. The next
Odyssey-native frontier is to eliminate fixed prompt/contract tokens while
preserving the tool/structured boundary, then rerun the same bounded O003
trace. The live floor is presently measured at `22.047 s`; an absolute
hardware/model floor is not established until the fixed envelope is reduced.

The next packet-cap A/B was deliberately rejected. A `1,400`-character
compiler packet can be produced offline (`1,331` characters), but both the
full-mission and direct `execute_workunit` probes unexpectedly rendered
`~4,683` resident tokens. They were stopped before prefill completion. This
is a dispatch/compiled-context fallback defect, not evidence that the compact
packet is scientifically safe or faster. The packet-cap experiment was not
made default and the temporary mission knob was removed. The verified
production result remains the `22.047--22.266 s` compact-envelope O003 floor;
the next implementation must trace why a manually compact packet is expanded
before attempting another 10-second A/B.

## O003 compact-contract and worker-memory correction — 2026-09-06

The expansion boundary was found and repaired. The agentic path selected the
compact schema, but the backend contract factory could clone the schema; an
identity check therefore failed and the full pretty-printed schema was added
to the user message. Separately, the worker was replaying an 18,269-character
historical `context_memory` object. The worker now uses the compact agentic
schema instruction independent of schema object identity and does not replay
that whole-history cache. Disk state, named evidence, and the bounded WorkUnit
packet remain authoritative.

The real O003 dispatcher was then measured with a reversible
`HCLI_WORKER_PACKET_CHAR_CAP` A/B and zero inline evidence:

- packet cap `1,400`: `838` resident tokens, `15.503 s` prefill;
- packet cap `900`: `669` resident tokens, `12.032 s` prefill;
- compact micro packet cap `700` (honest compiler output `634` chars): `446`
  resident tokens, `7.746 s` prefill, first decode reached.

The last measurement is the first traced O003 path under `10 s`, a `14.301 s`
improvement over the prior `22.047 s` floor. It was stopped before generation
completion, so it proves prefill only and promotes no scientific result. The
`600`-character full-mission attempt was correctly refused by the packet
compiler because the minimum honest packet for the affected units was
`631--638` characters; it performed no inference and is a rejected A/B.

The stable worker envelope is now `384` system characters plus the minimal
packet/contract. Approximate measured budgets are `~575` resident tokens for
`10 s`, `~288` for `5 s`, and `~58` for `1 s`. The current honest micro packet
is already below the 10-second budget. Five and one seconds still require a
separate micro-packet contract or resident-kernel improvement and are not
claimed. Focused regression coverage remains `57 passed`; two broader legacy
context-compiler tests still fail on pre-existing over-cap expectations and
were not altered by this lane.
