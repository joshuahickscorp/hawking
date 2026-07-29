# TG device-resident three-batch MLP — Revision 4 live-hit repair

Revise the Revision-3 device-only candidate in place. Preserve its device SiLU,
ordinary device-SiLU reference, one-buffer transaction, default-off isolation,
no-wave/no-table/no-replay/no-ICB rules, exact arithmetic, poison/reset,
source-body-free fixtures, and every false authorization/MOP/TG/HIDE fence.

The only authorized predecessor has base
`ba2ca65b1765e833ec381b454ee1d68b48534656` and these exact blobs:

- `gravity_glm.rs`: `a4b76fb49e6a5c682a24d0aed9a6e97bb385cd61`;
- `gravity_glm_resident.rs`:
  `f53d56c35d0b639e328af5120aeffa6b2442f8d5`;
- `metal/mod.rs`: `ee137f4824e8b090684b6f57c0a3f74b0889ab8b`;
- `cost_ledger.rs`: `88ae09e370667cb20722c99338fdebc93544d500`;
- `numeric_parity.rs`: `db4dbf2222999b3a7d25a68f76e83e664c6801c4`;
- `gravity_pq.metal`: `b555803b9c0246739bc493815f721ae5bb273776`;
- focused test: `7ea7838e48fafba159b881ea9f363e6cdbb1f893`.

Refuse before editing if any differs. Modify only those seven paths. Remove
`.serena`.

## 1. Freeze the live Revision-3 failure

The unsandboxed Apple M3 Ultra run produced:

```text
7 passed, 1 failed
candidate: cmds=44 encoders=44 mapped_r=416 mapped_w=328 waits=0
device_silu_ref: cmds=44 encoders=44 mapped_r=416 mapped_w=328 waits=0
panic: topology hit count
```

The earlier parity loop observed `three_batch_hit=128`, but the topology block
warmed each stateful model, reset the global probe, and forwarded the same
prompt again without resetting the generation/session position. That cached
forward did not enter the candidate MLP. The reference call then completed the
whole-token trace. Equal 44-command traces are therefore not MLP topology
evidence.

Revision 3 remains negative/default-off. Do not delete or weaken the hit
assertion, move the reset to conceal the issue, count the earlier untraced hit,
or report whole-token traffic from a cached forward as the candidate receipt.

## 2. Deterministic real-hit lifecycle

Give candidate and reference separate resident models/sessions with identical
fixture tensors and generation state. Warm only immutable weights, pipelines,
scratch allocations, and bind-once resources. Before every measured topology
or timing sample:

- reset the sequence/KV/session state through its verified reset while retaining
  the explicitly allowed immutable warm resources;
- set the same prompt/token position and route fixture;
- reset probes and begin the physical trace immediately before the forward;
- run exactly one forward that must execute the intended MLP;
- capture that mode's probes and terminal trace immediately afterward, before
  any other mode or reset;
- assert candidate hit positive/fallback zero and reference candidate-hit zero.

If the implementation cannot warm immutable resources separately from
sequence/KV state, construct fresh sessions around one immutable model/cache
owner and prove the cache/address generations. Never reuse a completed prompt
as the measured token.

Add a mutation test that restores the Revision-3 cached-forward mistake and
must fail the hit gate.

## 3. Stage-scoped physical attribution

Bind physical events to run, mode, token, layer, expert/shared prefix, MLP stage,
command buffer, encoder, buffer role, tensor/address generation, byte range,
completion status, and predecessor hash.

For every actual candidate hit prove:

- gate, up, device SiLU, down, ordered combine, shared-last, and residual stages
  each execute in the claimed order;
- zero mapped CPU reads/writes or D2H/H2D for gate/up/activation/down
  intermediates;
- zero host activation `Vec`, host SiLU, host-zero, per-token allocation, or
  weight upload;
- exactly one candidate MLP command-buffer dependency and checked success;
- every wave/table/replay/ICB entry/scratch probe is zero.

Whole-token mapped traffic from attention/router/head is reported separately
and cannot prove the MLP zero. Labels alone are insufficient: events bind the
actual buffer roles/address generations.

The ordinary device-SiLU reference must execute its independent non-three-batch
gate/up/SiLU/down flow, with its own positive reference-stage probes. A no-op,
fallback, cached token, or candidate-wrapper call refuses.

## 4. Governing comparison and speed gate

Repeat full residual/logit bit identity, exact router/DSA/head decisions,
device-SiLU V2.1, host-SiLU non-regression, failure/poison/reset, and codec
matrices on the corrected real-hit lifecycle.

Then run the frozen live protocol:

- 20 warmups plus at least 200 randomized/interleaved paired measured samples;
- reset to equivalent pre-token state before every sample;
- ledger off for timing and a separate ledger-on physical run;
- raw wall/GPU samples, nearest-rank p50/p95, paired confidence intervals;
- candidate and reference hit/stage counters for every measured sample.

Promotion requires all Revision-3 gates plus:

- candidate MLP command buffers/waits strictly fewer than the independently
  executed reference;
- total commands/token no greater;
- p50 and p95 each within the 2% non-regression limit;
- at least one metric with a strictly lower paired 95% confidence bound.

Otherwise return a negative receipt and keep the flag off. A corrected test,
topology-only win, or tiny fixture is never `BASE_TRUE_TPS`, TG2, TG1, HIDE
promotion, or capable-provider evidence.

## 5. Exit

Return exact Git blobs/SHA-256 identities, the reproduced R3 failure, complete
real-hit test output, per-mode probe/stage tables, physical event reconciliation,
raw live samples, and disposition. Keep all protected flags/claims false.
