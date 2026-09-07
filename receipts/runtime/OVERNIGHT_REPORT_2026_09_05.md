# HCLI overnight supergoal — report — 2026-09-05

Worktree: .worktrees/ascension   branch: ascension-isolated   HEAD: 3305b0a2e3e7af9a28aa88123b52312290ba136e
Working tree: 0 uncommitted path(s) at the time of writing.

## 1. ACCEPTED MUTATIONS

HCLI-authored, accepted by a verifier that could actually check them: **ZERO**.

One HCLI mutation was accepted at 07:2x and then REVERTED by the supervisor. It deleted the
correctness guard in `qwen38_batched_prefill_allowed` and was accepted on the evidence of
an unrelated Python test, because the verifier had no checker for Rust (section 9). It does
not count and is not counted.

Supervisor-authored substrate repairs, each with negative controls, all committed:

    6a1270bd3  close the FrontierEngine repair lineage (74 -> 64 failures, none new)
    fa01ed0eb  G012 decode producer
    af01fa42d  G012: fresh process per arm (a median over a degrading resident is not a rate)
    47c1d9359  the verifier now compiles Rust; uncheckable SOURCE fails closed
    5ab1065a2  G012 receipt: complete decode 36.67 tok/s
    3305b0a2e  the 9-cell retrieval matrix
    81fae6a91  frontier input: prefill vs prefix checkpoint
    2b2e7b062  frontier input: the KV cache is f32 while the model is bf16
    d5b18b5d7  frontier input: the resident degrades 4.5x across requests

## 2. REJECTED MUTATIONS

    HCLI, guard deletion in qwen38_batched_prefill_allowed   REVERTED by supervisor
    HCLI, non-compiling Rust (mission 3e0ee71e)              REFUSED by cargo_check exit 101
    HCLI, unit naming the protected G005 gate as proof        REFUSED, gate exit 1 (correct:
                                                              the receipt it demands is absent)

## 3. PHYSICAL MEASUREMENTS

Complete decode, marginal method, fresh process, two-point (wall(576)-wall(64))/512:

    36.668 tok/s     repeat 36.635 tok/s     run-to-run spread 0.090%
    aggregate across 2 streams 36.724 tok/s  ->  scaling 1.002x
    swapouts delta during measurement 0

Fresh prefill, fresh process, sequential route (the only route HCLI ever gets):

     7,737 tok    159.9 s    48.4 tok/s
    15,465 tok    444.3 s    34.8 tok/s
    30,921 tok  1,437.0 s    21.5 tok/s
    scaling ~n^1.48 between 8K and 16K, steepening to ~n^1.59 by 32K

Resident degradation across requests in ONE process:

    16K @25%  first call, fresh          446.0 s
    16K @25%  last call after 7 others   445.6 s     (fresh process)
    16K @50%  third call, shared        1,997.8 s     4.5x
    16K @90%  fourth call, shared         >870 s and still running when stopped

KV cache, exact from geometry constants (16 GQA layers x 4 kv_heads x 256 head_dim x 2):

    f32  (today)              131,072 B/token = 128 KB/token
    bf16 (model's own dtype)   65,536 B/token =  64 KB/token

## 4. CAPABILITY CHANGES

Nothing was made faster tonight. What changed is what can be VERIFIED:

  - The mutation loop can now verify Rust. Before 07:33 it could not compile the language
    every remaining Hawking target lives in, so no Rust mutation it ever accepted was
    actually checked.
  - One sovereign gate discharged: G012_resident_perf. 4 of 14 -> 5 of 14.
  - The 9-cell retrieval matrix replaces a stale, wrong capability claim with a measured one.

## 5. CONTEXT / STATE CHANGES

    9 of 9 retrieval cells RETRIEVED, byte-exact, at 8K / 16K / 32K x 25% / 50% / 90%.

The superseded conclusion ("retrieval fails at 16K at all depths") is OVERTURNED. The real
boundary is above 30,921 tokens and was not located. What binds today is the committed
envelope: 9,728 tokens, which RAISES rather than truncates above that limit.

No state reduction was landed. The KV lever is identified exactly (128 -> 64 KB/token) and
belongs to HCLI to author; it is also the enabling condition for G006's 262,144 rung
(34.4 GB of KV at f32 against ~49 GB free plus an ~11 GB body; 17.2 GB at bf16).

## 6. PREFILL / DECODE CHANGES

No improvement landed. The prefill gap is now EXPLAINED rather than open, and it is two
stacked defects, not one:

    F009  a prefix checkpoint (restored OR taken) disqualifies batched prefill, and HCLI
          checkpoints on every call, so HCLI never gets the batched route
    F017  the resident then degrades ~4.5x across the requests of a long mission

This closes the standing "roughly 400 s unaccounted INSIDE the call" question in
FRONTIER_INPUT_2026_09_05.md: that document costed HCLI's calls at 74.8 tok/s, a rate
measured on the batched route HCLI cannot reach.

## 7. TOTAL WORKUNIT WALL IMPROVEMENT

None measured, and the historical baseline is now known to be unsound: every per-WorkUnit
wall taken late in a long mission is inflated by an unknown, position-dependent amount
(F017). The 21,527 s over 41 units and the 525 s/unit average are degraded-process figures.

## 8. LAWS

  L1  A mutation may not be accepted on evidence that cannot cover it. Evidence in one
      language does not verify a change in another. Silence from a missing checker is not
      a passing check.
  L2  Any resident measurement must control for PROCESS HISTORY. One fresh process per arm.
      A median over a degrading series is not a rate.
  L3  A rate divided out of a whole call wall includes model load and prefill. Measure
      decode MARGINALLY between two token counts so both cancel.
  L4  Prefix reuse and batched prefill are mutually exclusive under the current admission
      rule. Any prefill claim must say which route it was measured on.

## 9. SCARS

  S1  A "capability ceiling" was actually a configuration ceiling. The 8K/16K retrieval
      limit came from the wrong renderer plus a window that refused the prompt. Before
      calling a limit physical, check what refused the request.
  S2  Reported field names lie. `prompt_token_count_source: estimated_without_transformers`
      was a MISLABEL over an exact resident count -- hawking_native.py:841 overwrites the
      label whenever the transformers tokenizer is absent, even when an exact count was
      already obtained. Arithmetic caught it: 13108 chars / 3231 tokens = 4.06, not chars/3.
  S3  I made the same class of error twice in one instrument in twenty minutes: dividing a
      whole call wall by its tokens (reported 10.4 tok/s of setup as "decode"), then
      multiplying an already-aggregate rate by the stream count (reported 73.1 tok/s and
      would have published "2 streams nearly double throughput", the exact opposite of the
      truth). Both were caught only by reading the raw arm data. Ask what the number
      measures, then read the rows.

## 10. BLOCKERS AND REOPEN CONDITIONS

  B1  Odyssey gate: fresh prefill >= 100 tok/s. MEASURED BELOW FLOOR at every length
      (48.4 / 34.8 / 21.5). REOPEN when a checkpointed call can use the batched route.
  B2  Odyssey gate: complete decode >= 40 tok/s. MEASURED 36.67, 8.3% short. REOPEN on any
      decode change; note DeltaNet is 33.8% of the token and KV only 11.7%, so decode work
      belongs in DeltaNet, not KV.
  B3  Bootstrap streak (3 consecutive accepted HCLI mutations). NOT CLOSED. Zero accepted
      by the repaired verifier. REOPEN condition: it is now running against a verifier
      worth passing; mission 3e0ee71e is live and detached.
  B4  G006_long_context needs rungs 131,072 and 262,144, not 32K. Blocked on the KV dtype
      change. NOT attempted tonight.
  B5  F017's MECHANISM is unknown. Three candidates recorded (restore-then-sequential,
      high-water sizing, Metal allocation) with a two-call discriminator specified.

## 11. CURRENT RESIDENT

    sealed-3.14, qwen3.8-27b, hawking-native, physical_ebpw 3.1393
    envelope ascension_envelope.hawking.json, max_seq_len 9,728
    binary workspace/ops/build/rust/release-fast/examples/ascension_qwen38_resident
    built 2026-09-05 00:20:21, AFTER the last source edit -- it includes uncommitted Rust
    changes and is NOT HEAD.

## 12. CLEAN / DIRTY

    0 uncommitted path(s). Every finding above is committed.

## 13. EXACT HEAD

    3305b0a2e3e7af9a28aa88123b52312290ba136e  on ascension-isolated

## 14. NEXT HIGHEST-VALUE FRONTIER

  1. Let mission 3e0ee71e run. It is detached (ppid 1) and survives this session. It is the
     bootstrap streak's first honest attempt.
  2. F009: batched prefill for the eligible part of a checkpointed prompt. This is the
     critical path to the Odyssey prefill floor, and no other lever reaches 100 tok/s.
  3. F017's mechanism, via the two-call discriminator. It contaminates every wall number
     the campaign has, so it compounds.
  4. F012/F014: KV f32 -> bf16. Exact, large, and the enabling condition for the 262K rung.
     Must be qualified by semantic execution, never reconstruction arithmetic.
  5. NOT next: decode micro-optimization. One stream already saturates the GPU (1.002x on
     two), so there is no scheduling headroom to harvest; the work is inside DeltaNet.

---

# ADDENDUM — three more substrate defects, found after the report above was written

The report above concluded the bootstrap streak had not closed. Continuing to work it
uncovered WHY, and it was never the model.

## The bootstrap loop could not do the job, in three independent ways

    F026  the verifier never compiled Rust
          _validate recorded "no_checker_available" for every non-.py path and continued
          without setting ok = False. HCLI deleted a correctness guard in Rust and was
          ACCEPTED on `test hcli/test_engine_tool_loop.py exit 0`.
          FIXED: .rs runs cargo check (13.5 s); uncheckable SOURCE fails closed.

    F028  an unwindowed fs.read returns 0.77% of the file
          qwen38_hybrid_decode.rs is 141,009 tokens, 14.5x the resident's ENTIRE 9,728-token
          window. fs.read with no window returns 4,001 of 520,266 characters and sets
          "truncated": true. fs.read has taken start_line/end_line all along and fs.search
          reports match lines -- the PROMPT's worked example showed neither.
          FIXED: the example searches, then reads a line window. Paid for by trimming prose,
          because adding it verbatim broke a previously-fitting closed turn.

    F030  the first call of every unit was cut at 947 tokens
          The planning call -- the one that has to choose tool calls and reason -- hit
          finish_reason "length" at exactly the grant, every time, with 5,802 tokens
          physically available. Units died at cognition with checks: 0.
          FIXED: HCLI_MODEL_TOKENS 1536 -> 4096 raises the grant to 3,507. The model then
          emits 500-633 tokens and stops on its own, well inside the directive's <1000
          preference. It was never rambling; it was being cut off mid-thought.

None of the three is visible on a Python-on-Python task, which is what every earlier
bootstrap success in this campaign was.

## What the loop does now

    80ccbd51  accepted a wrong Rust mutation on Python evidence
    3e0ee71e  refused non-compiling Rust (cargo_check exit 101); died on `length`
    4d6ae0f5  finishes its replies; rejected with reason NO_OP_MUTATION

The failure moved one layer up at each step and now sits on SUBSTANCE: the semantic no-op
rejection firing on a change that does nothing. That is the first legitimate failure of the
night.

## Corrected in the report above

Section 6 attributes the prefill gap to F009 and F017. That stands, but the commit message
for 81fae6a91 attributes the whole "missing 400 s" to F009 alone and overstates it -- F017
(process degradation) is part of it. Recorded in the ledger as F021 rather than rewritten.

## Still true

Nothing was made faster. Decode 36.67 tok/s and fresh prefill 48.4/34.8/21.5 tok/s remain
below both Odyssey floors. G001 remains open with zero accepted mutations. What changed is
that the loop is now capable of the task it is being asked to do.

---

# CLOSING — what four missions established

    80ccbd51  accepted a wrong Rust mutation on Python evidence   -> exposed F026
    3e0ee71e  refused non-compiling Rust; died on `length`        -> exposed F030
    4d6ae0f5  (1536 ceiling) died on `length` again               -> two equivalent failures
    4d6ae0f5  (4096 ceiling) finishes replies; NO_OP_MUTATION     -> substance, not plumbing

Four substrate defects, each the binding constraint in turn:

    F026  the verifier never compiled Rust
    F028  an unwindowed fs.read returned 0.77% of a file 14.5x the whole context
    F030  the planning call was cut at 947 of 5,802 available tokens
    F032  ...because of an HCLI_MODEL_TOKENS=1536 typed on a command line and then
          inherited unexamined across sessions. Unset, the engine derives ~4096.

All four fixed, each with a negative control. Every element section 0 demands -- semantic
no-op rejection, independent verification, rollback, one writer, durable receipts -- is
active and was observed firing.

## The streak did not close, and that is a measurement

Zero accepted nontrivial HCLI mutations on the Rust target across four attempts with a
repaired loop. The remaining failures are cognition, not plumbing.

The Odyssey contract already frames this correctly: the current dense resident is a
CALIBRATION SPECIMEN, explicitly "NOT presumed to be the final HCLI resident", and section 18
makes HCLI residency a question to be settled by measuring verified progress per wall time
per resource. Tonight measured this specimen against a hard Rust target with the substrate
finally out of the way. It did not close it in four tries.

That is input for Odyssey I, not a reason to keep re-running the same specimen.

## Verified obligations at close

    G002  the 9-cell context matrix          9 of 9 RETRIEVED, boundary above 30,921 tokens
    G010  failure discipline                 every failure carries evidence, mechanism,
                                             negative control, and a Law or Scar
    G011  this report
    G013  HCLI lifetime independent of the session -- mission at ppid 1, and a fresh
          process reconstructed the live mission from disk with nothing restarted

## Not met, with numbers

    G014  decode        36.67 tok/s   against a >=40 floor
    G015  fresh prefill 48.4 / 34.8 / 21.5 tok/s at 8K / 16K / 32K, against a >=100 floor
    G001  bootstrap streak: 0 of 3

---

# LATE FINDING — the prefix optimization has two paths pointing opposite ways

An anomaly in the 9-cell timings turned into the night's most actionable physical result.

    F017 observed   the same 16K prompt took 446 s fresh and 1,998 s as a third call
    F036 refuted    process age costs NOTHING. With disjoint-vocabulary prompts sharing no
                    prefix, a short call made second and third in a process that had already
                    served a 14,487-token request ran 60.35 s and 60.32 s -- FASTER than the
                    66.28 s fresh control.
    F037 explained  it is the prefix path, and it is a pessimization:

        cold                        3,911 tok    66.46 s    58.84 tok/s
        shares prefix with call 1   3,912 tok   150.62 s    25.97 tok/s   2.27x SLOWER
        shares prefix with call 2   3,911 tok     0.80 s  4905.66 tok/s   ~83x FASTER

    APPEND   the prompt strictly extends what the session already holds     ~83x
    RESTORE  the prompt diverges but still matches the stored checkpoint    2.27x SLOWER
             than having no checkpoint at all

Not route selection: cold and restoring calls both set snapshot_at and both ran the
sequential route. The 2.27x is on top of that.

The needle cells were the same filler with the needle MOVED, so consecutive prompts diverged
at the needle and repeatedly took the restore path. Every number that looked like
"degradation" tonight was a restore.

## Why this is the top frontier item

HCLI's tool loop appends an observation and re-sends, which is APPEND-shaped and should hit
the 83x path. Its traces instead show prefix_source "checkpoint_restore" repeatedly
(1,151 / 1,421 / 3,130 / 3,222 reused tokens). It is landing on the slow path in production.

Making HCLI's packets strictly append-shaped -- stable prefix, new material only at the tail
-- is a CONTROL-PLANE change, needs no kernel work, and on these numbers is worth more than
anything else measured tonight.

Frontier input: receipts/runtime/FRONTIER_INPUT_RESTORE_IS_A_PESSIMIZATION.md

## Corrections to earlier sections of this report

F017 was stated too broadly as "the resident degrades ~4.5x across requests in one process",
and that was used to claim every A/B ordering and every late-mission WorkUnit wall is biased.
The degradation requires a SHARED PREFIX; process age alone costs nothing. The practical
consequence survives -- HCLI missions share prefixes heavily, so real WorkUnits do take the
affected path -- but the mechanism in the earlier text is wrong and is corrected here.

The fresh-process measurement protocol used throughout remains correct: it avoids the path.

---

# ODYSSEY READINESS — what the launch contract's own gates measure

Working the contract's gates directly surfaced two more defects in the machinery Odyssey
depends on, plus one fix.

## FIXED: a detached supervisor's steer never reached the running mission

Contract section 4 treats detached supervision as normal: Claude attaches, injects, detaches.
Measured before the fix, on a live mission:

    mission session_id               3f254d6d-...
    the steer landed in              c22b3aa4-...   (the INJECTING process's session)
    file for the mission's session   did not exist

SteeringQueue is keyed by session id and a mission polls only its OWN file, so any steer from
a process that did not start the mission was orphaned -- and the CLI printed a success tick
regardless. A silent success is the worst form of this: the supervisor believes it steered a
running experiment and it did not.

Fixed (4e7bab9f6): the no-mission branch now reads the LIVE mission's identity from
.hcli/mission/state.json and enqueues to the session the mission actually polls, recording
mission_id and source_session_id. Verified live: the next steer landed in the mission's own
file with mission_id 9075bdf8. Negative control: a mission in phase "failed" is not a target.

Not claimed: the steer still reads applied=false. Delivery is fixed; CONSUMPTION into
WorkUnits is not demonstrated, so the gate does not pass.

## NOT FIXED: an interrupted mission is REPLACED, not resumed

    BEFORE interrupt  id 9075bdf8  units {implement: running, validate: pending}
    kill              state SURVIVES intact
    reissue the SAME goal
    AFTER             id 28e36b34  -- a NEW mission; 9075bdf8 gone from state.json,
                                     checkpoint re-keyed, prior DAG retired

Mission.from_workspace restores units from disk, so the capability exists at the API level;
the /mission CLI path does not use it. Contract section 10 requires resume to verify identity
and continue from the first unfinished unit, precisely because Odyssey may run for days.

Left to HCLI on purpose: contract section 1 assigns missing Odyssey infrastructure to HCLI,
not to the supervisor. This is machinery, not a transport repair.

## They compound

Because the mission is replaced rather than resumed, the restart also mints a NEW session id.
So a steer correctly aimed at the pre-interrupt mission is orphaned a second time -- by the
replacement, not by the routing bug that was fixed. A supervisor steering a long Odyssey
across an interruption loses the steer twice.

## Launch summary, measured

    [x] corrected 9-cell context qualification      9 of 9 RETRIEVED, capability survives at 16K
    [x] prefix reuse physically verified            six numbers reported separately
    [x] no material swap contamination              swapouts delta 0
    [x] memory/state accounting valid               131,181 B/token measured vs 131,072 derived
    [x] Claude detach does not stop HCLI            mission at ppid 1
    [x] new session attaches without restarting     /status reconstructed a live mission,
                                                    resident count 1 before and after
    [ ] complete decode >= 40 tok/s                 36.67 MEASURED BELOW
    [ ] fresh prefill >= 100 tok/s                  48.4 / 34.8 / 21.5 MEASURED BELOW
    [ ] 3 consecutive accepted HCLI mutations       0 of 3
    [ ] checkpoint/resume proven                    FAILS -- replaced, not resumed
    [ ] specimen lifecycle / canaries / ODYSSEY_READY   not reached

ODYSSEY_READY = NOT VERIFIED. Two floors are measured below, the streak is zero, and resume
does not exist. None of that is close to a judgement call.

---

# FINAL — six missions, four substrate fixes, zero accepted mutations

## The bootstrap result

    80ccbd51  Rust    ACCEPTED a guard deletion on Python evidence -> exposed the verifier
                      defect; reverted
    3e0ee71e  Rust    refused non-compiling Rust (cargo_check 101); died on `length`
    4d6ae0f5a Rust    died on `length` again -- two equivalent failures, so the variable
                      was changed rather than retried
    4d6ae0f5b Rust    replies finish; rejected NO_OP_MUTATION
    28e36b34  Rust    killed by me: I stopped the resident under it before a cargo run
    6bc1f526  PYTHON  tractable target, clear spec, named test -> repair budget exhausted

Substrate fixed along the way, each with its own negative control:

    the verifier now COMPILES Rust                            47c1d9359
    the prompt teaches windowed reads                         11a241f72
    the planning call is no longer starved at 947 tokens      (launch flag, F030/F032)
    a detached steer REACHES the mission                      4e7bab9f6
    ...and the running mission now READS it                   392fc6eab

Every element section 0 demands -- semantic no-op rejection, independent verification,
rollback, one writer, durable receipts -- was observed active and firing. The loop works.
This resident did not close a mutation through it.

## Two errors of mine worth recording

I spent five missions on the hardest target in the repo before trying a tractable one.
Section 0 asks for three accepted nontrivial mutations and never says they must be Rust, and
every historical success in this campaign was Python-on-Python. The tractable attempt also
failed, so the conclusion stands -- but the ordering was wrong and cost hours.

I also stopped the resident out from under a running mission before a cargo run, killing it.
That is why 28e36b34 has no result.

## A refinement that changes how to read finish_reason

Raising the completion ceiling removed the starvation, and on some units the model then
emitted 500-633 tokens and stopped on its own. On the G002 task it emitted 3,507 and 3,426
and was CUT AGAIN at the new ceiling. The model expands to fill the budget on some tasks.
Starvation and rambling both present as finish_reason "length" and are not the same defect.

## Final ledger

    VERIFIED   13 of 22
      context     9-cell matrix, capability survives at 16K, state census, prefix reuse
      physical    prefill A/B, whole decode token, command-buffer hypothesis refuted
      control     repair lineage, failure discipline, HCLI lifetime independent of the session
      VMCP        image geometry cross-checked against an independent tool

    BLOCKED, each with its missing input named
      G014 decode 36.67 vs >=40      host overhead is 3.6%; zeroing it gives 40.56
      G015 prefill 65.10 peak vs >=100   measured on BOTH routes, token-identical
      G017/G018/G020/G021            Odyssey machinery the contract assigns to HCLI

    FAILED
      G001 bootstrap streak 0 of 3
      G019 resume -- an interrupted mission is replaced, not resumed
      G022 steer delivery FIXED and verified; consumption into a WorkUnit not observed

ODYSSEY_READY = NOT VERIFIED, on four independent counts, none of them a judgement call.

## What tonight actually established

Nothing was made faster. What changed is what can be TRUSTED: a mutation loop that can
compile the language it works in, read files larger than its context, think without being
truncated, and receive a steer from a detached supervisor. And a set of measurements that
say where the remaining work is -- inside the model, not around it. The decode host share is
3.6%. Batched prefill peaks at 65 tok/s. Two streams scale 1.002x. None of those leave room
for scheduling cleverness.

The Odyssey contract's own framing fits the result: this dense resident is a calibration
specimen, and section 18 makes HCLI residency a question for measurement. Tonight measured
it against a repaired loop and the answer was zero accepted mutations in six attempts. That
is Odyssey I input.

---

# ADDENDUM 2 — supervisor injection closed, three defects deep

Contract section 4 asks that a supervisor be able to steer a live mission. That path had
THREE independent breaks, each invisible behind the one before it:

    ROUTING     the steer landed in the INJECTING process's session file, which a mission
                never reads -- and the CLI printed a success tick either way   4e7bab9f6
    STALENESS   the mission's queue loads once at construction and all() returned that
                cached list, so a correctly-routed steer was still unseen       392fc6eab
    PROVENANCE  to_dict wrote mission_id/source_session_id; from_dict dropped both, losing
                them on reload -- an orphan indistinguishable from a delivery   7dff43118

The third was my own half-finished fix. The end-to-end test caught it, because it asserts the
whole path rather than either half: injected from a second process, absorbed by a queue
constructed BEFORE the injection, present in compile_worker_context's output as
steering=('[knowledge] ...'). Two working halves and a broken joint is still a steer nobody
acted on.

I also committed that test one commit before the code that makes it pass -- I committed
without reading the result. Recorded because it is the kind of thing that turns a green suite
into a lie.

Verified live on two separate missions with the mission id unchanged, one writer process
throughout, and the pre-fix orphan still visible: history added to, never rewritten.

# FINAL DISPOSITION

VERIFIED 14 of 22. All eight remaining obligations are BLOCKED with exact missing inputs:

    G001  bootstrap streak 0 of 3       a resident that can author a mutation
    G014  decode 36.67 vs >=40          model-side work; host share is only 3.6%
    G015  prefill 65.10 peak vs >=100   work beyond batching, measured on BOTH routes
    G017  specimen lifecycle            HCLI-owned Odyssey machinery
    G018  probe classification          probes to classify
    G019  checkpoint/resume             an HCLI-authored resume
    G020  three canaries                G017
    G021  ODYSSEY_READY                 the four gates above

Not one of them is blocked on difficulty or cost. Every one needs either a Hawking target
mutation or Odyssey machinery that this directive and the Odyssey contract reserve for HCLI,
and I authored neither -- which was the point.

REOPEN CONDITION: one accepted nontrivial HCLI-authored mutation.
