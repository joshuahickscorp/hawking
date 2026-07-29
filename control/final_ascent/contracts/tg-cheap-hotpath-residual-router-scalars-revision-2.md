# Temporal Gravity cheap hot-path — Revision 2 candidate disposition

Revise the frozen Revision 1 candidate in place. Preserve all prior default-off,
non-wave, no-ICB, Numeric Parity V2.1, no-real-model/no-MOP, physical-evidence,
transaction, and false-fence rules.

An unsandboxed local Metal run of Revision 1 produced:

- baseline: 39 waits;
- C1: 39 waits only because device residual was killed and host residual kept;
- C2: 102 waits and test failure (`102 > 39`);
- baseline/C1/C2 `score_pair.pass=false`;
- C2 `meaningful_rel = 4.853e-3` versus V2.1 bound `1.0e-5`;
- an alternate mode-vs-peer comparison was printed as rationale even though
  the governing device verdict remained false.

Revision 1 blobs:

- `gravity_glm.rs` `7f73debe095c3bd91a829ccd58d5957ef5519741`;
- `gravity_glm_resident.rs` `dc925f9b2b8986ba4a9f2cccf864d53fc75b649f`;
- test `0ef9b214ded5ee581a365fbd4847c9d2b42d3f08`.

## 1. Final disposition of C1 and C2

C1 device residual is not an acceleration and must be absent from every
promotable/combined path. A diagnostic-only flag may remain only if:

- its name/schema explicitly says diagnostic/non-promotable;
- production/hot-path flag resolution cannot enable device residual;
- runtime receipts report `candidate_disposition=REJECTED`;
- no TG or combined mechanism counts it as a closure.

C2 device router as currently implemented is also rejected as an acceleration:
102 waits versus 39 is a hard kill. Remove it from the combined/promotable path
and mark the candidate/receipt rejected. Do not merely leave a failing test or
hide it behind default-off.

The ordinary production baseline remains host router until a new versioned
implementation demonstrates:

- true FP64 near-tie authority;
- exact k IDs+k weights;
- zero wave/table/replay/ICB calls;
- waits/CBs no greater than host baseline;
- complete V2.1 pass and positive live Metal wall evidence.

Any future C2 must use a new flag/version and new qualification receipt; this
rejected implementation cannot be silently revived.

## 2. No parity proxy or alternate pass

Complete V2.1 `score_pair.pass` against the frozen authoritative computation is
the gate. `host_pass`, greedy/top-k, relative-L2 alone, mode≈peer, same-GPU
output, or a printed explanation cannot replace it.

If the tiny fixture lacks a valid original-input f64 authority for a domain,
mark that domain `AUTHORITY_UNAVAILABLE` and refuse qualification. Do not call
the baseline f32 widened to f64, another GPU mode, or a peer comparison
authority.

Every continuous and discrete field/failure in the governing V2.1 verdict is
asserted. A nonempty failure list or top-level false fails the candidate test.
Do not loosen meaningful-relative or other bounds.

## 3. Isolate C3

Split baseline/C1/C2/C3/combined tests so a killed C2 does not prevent the C3
gate from running and reporting. C3 compares:

- identical device-head mode;
- identical full-logit/token-only setting;
- identical fixture/prompt/tokens/cache state;
- baseline without persistence versus C3 persistence only.

C3 must prove:

- complete V2.1 pass from valid authority;
- cold exactly one final-norm bind, warm zero unchanged norm uploads;
- scalar arena writes only changed fields;
- real address/storage generation, complete model/artifact/tensor/dtype/shape/
  device/source/destination identity;
- in-flight ring/fence no-overwrite;
- no ICB, full-logit change, replay, or final-head replay;
- waits/CBs no greater than same-head baseline;
- physical byte reduction exactly matches removed warm norm/scalar writes;
- randomized/interleaved wall and GPU p50/p95 non-regression/improvement.

If C3 fails any gate, mark it rejected too and restore the exact baseline path.

## 4. Combined path and receipts

Until a candidate individually passes every gate, the combined acceleration
path is exactly the baseline. Do not include a diagnostic/rejected candidate in
combined timing, wait reduction, or byte claims.

Emit a machine-readable candidate disposition for C1/C2/C3:

- `ACCEPTED_FIXTURE_ONLY`;
- `REJECTED_PARITY`;
- `REJECTED_PHYSICS`;
- `AUTHORITY_UNAVAILABLE`;
- `NOT_RUN`.

Only `ACCEPTED_FIXTURE_ONLY` may proceed to a later capable-model gate, and it
still cannot claim TG.

## 5. Live Metal gates

Run each candidate independently so all dispositions are collected:

- complete V2.1;
- actual call-path probes;
- physical topology/counters;
- randomized/interleaved p50/p95;
- edge/failure/poison matrix.

The test suite itself must exit nonzero if code labels a rejected candidate
accepted, masks parity, or includes it in combined. Tests for an intentionally
rejected candidate may pass only by asserting the exact rejected disposition
and baseline restoration.

Report exact files/hashes and live disposition table. No product TPS/TG/HIDE
promotion, real model body, MOP action, or authorization transition is
permitted.
