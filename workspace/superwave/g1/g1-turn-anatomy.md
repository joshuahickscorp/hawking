# G1 turn anatomy — one HCLI worker turn

Lane: `117-turn-anatomy`. Write scope: this file only.
No GPU, no inference, no launchd/resident/daemon mutation, no two-writer edits.

Label key: `MEASURED` this process from live logs/receipts. `SOURCE` file:line.
`DERIVED` arithmetic on those. `PROJECTED` scale under a stated assumption.
`ESTIMATED` not stopwatch-closed. A component microbenchmark is not a turn.

Primary turn (closed budget): kernel / `child_b` seal
`fc9da9479fa3faef95bc4c3a8a5d934a236de6ebb2a13c78f1b6d2c486db5654`
`recorded_at` 2026-08-17T17:28:09Z.

Claim specimen (the "207 s" turn): kernel session
2026-08-17T16:11:41.229778Z → 16:15:08.862862Z.

---

## 0. Verdict

The 207 s claim is **SUPPORTED** as a wall, and **SUPPORTED** as prefill-dominated
once the later instrumented receipts are admitted. It is **FALSIFIED** as
"the three sealed contract files (16 414 + 11 912 + 4 871 B) are re-sent".
What is re-sent is the compiled runtime capsule (1 402 tokens) plus tools
preamble plus a compact worker task. `generate_greedy` still `session.reset()`
and teacher-forces every prompt token via `session.step` on every propose.

Budget for the primary turn **closes** to 3 µs inside native wall and 0.7 ms
inside the session span. A budget that does not close would have been a finding.

Fair-window organism rate (17:08:13Z–17:36:10Z, 7 completed turns):
**15.02 turns/h** DERIVED, not 17.4. Gaps in that window are 8–18 s (11 % of
a turn), not rivals. Pre-17:08 gaps of 7–37 min **did** rival the turns
(DEFERRED + 180 s cooldown + resident restart). That is the number that
governed progress this afternoon, not TOKEN_NS and not BPW.

---

## 1. Claim check

### 1.1 "One HCLI worker turn takes about 207 seconds"

**SUPPORTED.** Kernel session events:

```
2026-08-17T16:11:41.229778Z running n=6
2026-08-17T16:15:08.862862Z idle    n=8
```

MEASURED span **207.633084 s**. Same tick native `wall_ns=206949279500`
(206.949 s). Source: `workspace/ops/genesis-agentos-sessions/genesis-kernel/events.jsonl`
and `genesis-agentos.log` object recorded_at 16:15:08Z seal
`15d902e3538adcb2f16498ff95bc727980a3e6e63be00b06b8cb9b012bcd752f`.
That receipt has no `prefill_wall_ns` (instrumentation not yet on the wire).

Later same path, with split (all MEASURED, `act.native`):

| recorded_at | worker | prompt_len | prefill_s | decode_s | wall_s | prefill/wall | n_new | ssi | serve | ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16:31:20Z | kernel | 3222 | 205.143 | 3.099 | 208.242 | 0.9851 | 36 | 2 | 5 | T |
| 16:43:10Z | kernel | 3468 | 222.971 | 3.274 | 226.244 | 0.9855 | 37 | 3 | 7 | T |
| 16:46:49Z | gravity | 3209 | 200.963 | 3.237 | 204.199 | 0.9842 | 38 | 1 | 8 | F |
| 17:12:05Z | kernel | 3529 | 228.521 | 3.488 | 232.010 | 0.9850 | 39 | 4 | 10 | T |
| 17:15:56Z | gravity | 3413 | 218.616 | 2.800 | 221.417 | 0.9874 | 32 | 2 | 11 | F |
| 17:20:09Z | kernel | 3534 | 229.460 | 4.410 | 233.870 | 0.9811 | 49 | 5 | 12 | T |
| 17:23:59Z | gravity | 3446 | 219.999 | 2.791 | 222.790 | 0.9875 | 32 | 3 | 13 | F |
| 17:28:09Z | kernel | 3527 | 227.924 | 4.390 | 232.314 | 0.9811 | 49 | 6 | 14 | T |
| 17:32:01Z | gravity | 3452 | 221.109 | 2.814 | 223.923 | 0.9874 | 32 | 4 | 15 | F |
| 17:36:10Z | kernel | 3515 | 227.192 | 4.386 | 231.578 | 0.9811 | 49 | 7 | 16 | T |

A later kernel tick at 17:44:14Z reports `prompt_len=3541` MEASURED, equal to
the reconstructed wire token count in §1.3 (same compiler, same durable
state). That is the cross-check that the capsule+tools+task split is the
thing being prefills, not the 33 197 B files.

`used_gpu_lane_lock` is `false` on every HCLI receipt. The body still takes
`/tmp/hawking-gpu-lane.lock` (see §4). `fallbacks=0`, `transport=genesis_resident`.

### 1.2 "Dominated by prompt prefill because each request resets and re-prefills"

**SUPPORTED.**

`crates/hawking-core/src/model/qwen38_hybrid_decode.rs:3652` `session.reset();`
then `:3664-3678` `for token in prompt { session.step(token)? }`.
`step` (`:3577-3593`) is one full encode_embed + encode_layers + encode_terminal
+ `commit_and_wait_timed` per token. Prefill is sequential decode-shaped GEMV,
not a batched GEMM.

`session.reset` (`:1451-1457`) zeros conv/rec/GQA KV and sets `position=0`.
`compile_worker_context` writes `"ephemeral_kv_reused": false`
(`lab/lineage/continuity.py:240`). `session_serve_index` climbs 1→7 on kernel
while `prompt_len` stays 3222–3534 and `prefill_wall_ns` stays 205–229 s:
the session is reused as a workspace, not as a KV prefix.

Prefill per token on the primary turn: 227.923884208 / 3527 = **64.622 ms/tok**
DERIVED. Short-context G0 TOKEN_NS is 39.326 ms (RECEIPT, not this lane).
Long-context decode on the same turn: 4.390303542 / 48 = **91.465 ms/step**
DERIVED (`decode_steps = n_new-1` because the last prefill step emits new-token[0];
SOURCE `qwen38_hybrid_decode.rs:3680-3700`).

Parent-propose 79.00–79.76 s / 1558 tok (RECEIPT `g1-resident-headroom.md:370-372`)
is the same algorithm at a shorter prompt: 50.7–51.2 ms/tok DERIVED. Worker
turns are longer because the HCLI wire prompt is ~2.3× that length, not because
a different kernel ran.

### 1.3 "The sealed Genesis system contract set is re-sent every turn"

**FALSIFIED as file bytes. SUPPORTED as compiled capsule.**

On-disk files MEASURED:

```
QWEN38_GENESIS_SYSTEM_DIRECTIVE.md  bytes=16414 sha256=881ae469e0287cf386467002d3fc7951524b47054ac6d7f753b94a8e4e3ceff7
GENESIS_CONTINUITY_DIRECTIVE.md     bytes=11912 sha256=c4a58bc06575effb8f759dbb22c49abfc65e1957910b18917d45d02592d1fdbc
GENESIS_OUTPUT_LAW.md               bytes=4871  sha256=9679490e8ae623a6fdb408fd906a15d676bc55926580f6d7ed60e9ea610c9ada
binding_sha256=3ef47426958200ff830ea2ec5adce53d3b3347098d459bd7fcddc9a5dc9a179f
```

Binding formula SOURCE `tools/agentos/genesis_contract.py:172-180`
(`relative_path\0sha256\0size_bytes\n` over the three). Matches live
`contract_provenance()`.

What the body actually sees: `inject_runtime_contract` (`:219-271`) prepends
`runtime_capsule` (`:184-217`), not the three files. Capsule MEASURED 5 119 B /
**1 402 tokens** (CPU `tokenizers` encode of the live compiler output; not a
GPU turn). Full files are 33 197 B and are not on the wire.

Reconstructed current kernel wire (same compiler, current durable state):

| piece | chars | tokens | class |
|---|---:|---:|---|
| system / capsule + chat tags | 5138 | 1405 | MEASURED encode |
| user / tools preamble | 1167 | 263 | MEASURED encode |
| user / worker task | 4680 | 1839 | MEASURED encode |
| assistant think-close prefix | 130 | 28 | MEASURED encode |
| **wire total** | **11145** | **3541** | MEASURED encode |

Live primary-turn `prompt_len=3527`. Reconstruct is +14 tok because durable
`NEXT_ACTION` / last-turn excerpt drifted after that tick. Same order of
magnitude; not a turn-level claim that the 17:24 prompt was 3541.

`genesis_contract.py:3-6` still says "4096-token context". Live argv is
`--max-seq-len 8192` (resident log + `genesis_forever.sh:70`). Stale comment.

---

## 2. Closed budget — kernel 17:24:16.970465Z → 17:28:09.805394Z

Sources: `genesis-kernel/events.jsonl`, sealed tick in `genesis-agentos.log`,
`generate_greedy` timers, `propose` order in `genesis_body/src/main.rs:622-742`.

### 2.1 Session span (HCLI)

```
17:24:16.935672Z idle     n=21   # refresh_context save
17:24:16.970040Z idle     n=21
17:24:16.970465Z running  n=22   # user turn appended, backend.complete starts
17:28:09.792760Z running  n=23   # assistant text saved
17:28:09.793456Z running  n=23
17:28:09.805394Z idle     n=24   # tool ran, act() returned
```

| phase | t0 → t1 | seconds | class | source |
|---|---|---:|---|---|
| claim / lineage / capsule hash / compile context / refresh_context / session save | 17:24:16.935672 → 17:24:16.970465 | 0.034793 | MEASURED | events |
| tokenize + GPU-lock acquire + `reset()` + generate | 17:24:16.970465 → 17:28:09.792760 | 232.822295 | MEASURED | events |
| parse tool + `read` + persist + idle | 17:28:09.793456 → 17:28:09.805394 | 0.011938 | MEASURED | events |
| **session running→idle** | 17:24:16.970465 → 17:28:09.805394 | **232.834929** | MEASURED | events |
| residual (extra running save) | — | 0.000696 | MEASURED | 232.834929 − 232.822295 − 0.011938 |

Tool was `read` of `.../kernel/lab/runtime.py` offset=1 limit=200, `ok=true`
(sealed tick `act.results[0]`).

### 2.2 Native generate (inside the 232.822 s window)

`generate_greedy` `wall` starts **after** `session.reset()` (`:3652` then `:3660`).
`prefill` / `decode` partition that wall. Native residual is the timer gap.

| phase | ns | seconds | fraction of wall | class |
|---|---:|---:|---:|---|
| prefill (`step` × 3527) | 227_923_884_208 | 227.923884208 | 0.981102 | MEASURED |
| decode (`step` × 48) | 4_390_303_542 | 4.390303542 | 0.018898 | MEASURED |
| native residual | 2_625 | 0.000002625 | 0.000000011 | DERIVED |
| **native wall** | **232_314_190_375** | **232.314190375** | 1 | MEASURED |

227.923884208 + 4.390303542 + 0.000002625 = 232.314190375. **Closes.**

### 2.3 Host-around-generate (session window − native wall)

232.822295 − 232.314190375 = **0.508105 s** MEASURED.

`propose` order (`genesis_body/src/main.rs:653-706`): tokenize →
`GpuLaneGuard::acquire` → `t0` → `generate_greedy` (reset then wall).
So the 0.508 s is tokenize + lock acquire + `reset()` + `decode_new` + JSON.
Not separately timed on the live path (`reset_ns` exists only on
`generate_greedy_complete_wall`, `:3728-3730`, unused by the resident).

Attribution inside 0.508 s is **INFERRED**, not measured:

| piece | seconds | class | why |
|---|---:|---|---|
| `tokenizer.encode` of ~11 KiB | 0.004 | MEASURED this process, Python tokenizers | rust encode not timed; same order |
| `GpuLaneGuard::acquire` uncontended | <0.001 | ESTIMATED | `create_dir`; lock was free 18 s |
| `session.reset()` zero of 1 232 326 404 B workspace | ~0.50 | INFERRED remainder | 4× `zero_buffer` write_bytes; live RSS workspace RECEIPT |
| `decode_new` + JSON of 49 ids | <0.001 | ESTIMATED | |

REOPEN_IF `reset_ns` / `tokenize_ns` / `lock_ns` are plumbed onto the propose
reply. Do not treat the 0.50 s split as a turn-level measurement.

### 2.4 Host setup (0.034793 s) — this process microbenches, not the turn

| call | seconds | class | note |
|---|---:|---|---|
| `contract_provenance()` | 0.000179 | MEASURED this process | 33 KB, 3× SHA-256 |
| `inject_runtime_contract` | 0.000208 | MEASURED this process | includes another load+hash |
| `project_context` on kernel worktree | 0.043 | MEASURED this process | turn used this path; 35 ms session slot is the same order |
| `project_context` on main repo | 0.103 | MEASURED this process | **not** the turn path |
| `_git_snapshot` worktree | 0.050 | MEASURED this process | dirty=false |
| workers.json 94 148 B SHA | 0.000093 | MEASURED this process | |
| checkpoint 73 649 B SHA | 0.000078 | MEASURED this process | |

`contract_provenance` is invoked at least three times per successful tick
(`run_once:414`, `propose:294`, `health`/`body_is_up:255-278`). Still
sub-millisecond. Identity-hash-on-hot-path is **present and not the wall**.

### 2.5 Second decode

**Absent in-turn.** `dispatch_agentos_turn` hard-codes `--max-rounds 1`
(`ascent_daemon.py:948-949`). `act()` would feed `<tool_result>` back only
when `max_rounds>1` (`special_unit.py:2207-2218`). Next turn sees the tool
output only as a 500-char `LAST_OBSERVED_TURN` excerpt (`genesis_agentos.py:296-304`).
Raising `max_rounds` without deleting `session.reset()` would **double**
prefill in one tick. Not a win until prefix KV is kept.

### 2.6 GPU lock / health / scheduling around this turn

| phase | seconds | class | source |
|---|---:|---|---|
| previous worker (gravity) idle → this running | 17.935 | MEASURED | 17:23:59.035143 → 17:24:16.970465 |
| this idle → next worker (gravity) running | 7.741 | MEASURED | 17:28:09.805394 → 17:28:17.545994 |
| start-to-start organism cycle | 240.576 | DERIVED | 17:24:16.970 → 17:28:17.546 |
| cooldown remaining after a 232 s turn | 0 | DERIVED | `180 − 232 < 0` (`ascent_daemon.py:911-917`) |

`POLL_SECONDS=60` (`ascent_daemon.py:64`). Measured post-success gaps 8–18 s
are the remainder of that poll plus process start plus claim, not a 180 s
cooldown.

Supervisor (`tools/genesis_forever.sh:115-127`) polls `health` every 5 s.
During generate the accept loop is blocked, so health hits
`CONNECT_TIMEOUT_S=0.25` (`genesis_resident.py:42`) and logs
`health_busy_process_alive`, then `health_recovered_after_busy`. That is
honest. It does **not** restart the body. ~46 failed 0.25 s connects per
230 s turn sit off the generate critical path (different process).

---

## 3. Scheduling overhead — gaps vs turns

### 3.1 Fair window (after RR + resident stay-up)

17:08:13.099113Z–17:36:10.607842Z: 1 677.509 s, 7 completed turns.
**239.64 s/turn, 15.02 / h** DERIVED. Alternation is kernel/gravity every
~8–18 s. Per-worker period ~481 s (~7.5 / h / worker). Decode ceiling 1
makes that the organism rate, not 2×.

Gaps do **not** rival turns here (25 s / 233 s ≈ 11 %).

### 3.2 Whole afternoon 15:36–17:36

11 completed native turns / 7 200 s ≈ **5.5 / h**. The missing 9.5 / h is
gaps, not prefill.

Same-worker gaps MEASURED (events):

| worker | from | to | gap_s | cause |
|---|---|---|---:|---|
| gravity | 15:36:14 idle | 15:48:41 | 746.6 | early reject + cooldown + other work |
| gravity | 15:48:41 | 16:05:39 | 1017.7 | same |
| gravity | 16:05:39 paused | 16:43:24 | 2264.9 | refuse + kernel monopoly + DEFERRED |
| gravity | 16:46:49 | 17:05:08 | 1099.2 | DEFERRED cluster §3.3 |
| gravity | 17:05:08 paused | 17:12:14 | 425.4 | 180 s cooldown then kernel 232 s |
| kernel | 16:15:08 | 16:20:15 | 306.3 | cooldown + DEFERRED |
| kernel | 16:20:15 paused | 16:27:51 | 455.8 | refuse + cooldown |
| kernel | 16:43:10 | 17:08:13 | 1503.1 | DEFERRED cluster + resident restart (log mtime 12:08 local) |

### 3.3 DEFERRED_RESIDENT_NOT_READY × 180 s cooldown

`genesis-agentos.log` sealed outcomes `DEFERRED_RESIDENT_NOT_READY` at
16:15:42, 16:24:44, 16:33:57, 16:47:25, 16:50:30, 16:53:35, 16:58:59, 17:02:04.

`body_is_up` → `health(..., timeout=0.25)` (`genesis_resident.py:255-278`).
A busy body looks identical to a dead one. `run_once` then seals DEFERRED
**before** claiming (`genesis_agentos.py:373-381`). The dispatcher still
wrote `started_at` (`ascent_daemon.py:964-974`). `agentos_turn_status`
then reports `cooldown` for `180 − (now − started_at)` s (`:911-917`).
Default `GENESIS_AGENTOS_MIN_INTERVAL_S=180` (`:879`).

16:47:25 + 180 s = 16:50:25 ≈ 16:50:30. 16:50:30 + 180 s = 16:53:30 ≈ 16:53:35.
The cluster is the cooldown, not a 3-minute poll.

8 DEFERRED × 180 s = **1 440 s** DERIVED scheduled dead time. That is the
gap class that actually beat the 207 s turn this afternoon.

Fast `ResidentRefused` (health timeout after claim) is the same 0.25 s
paused window (gravity 16:05:39.280→.537, 17:05:08.441→.694; kernel
16:20:15.160→.414) and also arms the 180 s cooldown. 17:05:08 + 180 s =
17:08:08; kernel started 17:08:13. MEASURED.

### 3.4 "queue dry" vs underlying counts

Daemon hold histogram this log (394 JSON lines): `queue dry` **164**,
AgentOS running/started **64**, cooldown **20**, memory-cap holds the rest.

Live underlying counts MEASURED this process:

| object | value | source |
|---|---:|---|
| PROMOTION_QUEUE.entries | 388 | `receipts/ascent-2026-08-16/PROMOTION_QUEUE.json` |
| promoted | 0 | same |
| MERGE_READY unpromoted | 134 | same |
| NEEDS_COMPOSITION | 15 | same |
| NO_REPORT_MANUAL_REVIEW | 236 | same |
| ASCENT_STATE.targets | 222 | `ASCENT_STATE.json` 532 738 B |
| pending | 0 | status Counter |
| running | 0 | status Counter |
| stale_no_process | 127 | 101 q80 + 13 dsv4f + 12 qwen38 + 1 both |
| mechanism_refused | 66 | all qwen38 |
| launch_failed | 7 | all qwen38 |

`hold: "queue dry - harvest supplied no new pending target"`
(`ascent_daemon.py:1643-1646`) is true about **pending targets** and a lie
about work remaining. Same failure class as G011
(`receipts/ascent-2026-08-16/G011_AUTONOMOUS_LOOP_STARVED.json`):
the loop reported healthy with `launched: null` while harvest had been
starved. Today harvest works (`harvest()` 0.025 s, 190 reports, 240 task
dirs MEASURED this process). The dry hold is now "no pending target after
66 mechanism_refused + 0 running", with 388 undrained promotion entries
sitting next to it.

`one_pass` still `load(STATE)` 533 KB + `harvest()` + `load(QUEUE)` on
**every** 60 s tick, including while an AgentOS turn is running (`:1474-1478`
before the running early-return at `:1568`). Cost MEASURED this process:
1.4 ms + 25 ms + 0.4 ms. Not the wall. Present.

---

## 4. Failure classes (checked, not assumed)

| class | present? | cost | where |
|---|---|---|---|
| Work repeated per turn that does not change | **YES** | 227.9 s MEASURED | `session.reset` + full `step` prefill of a 1 402-tok static capsule + 263-tok static tools preamble |
| Identity / hash on hot path | YES, tiny | 0.18 ms × ≥3 MEASURED | `load_genesis_contracts` every inject/health/propose/`run_once` |
| Large JSON parsed per turn | **NO as turn wall** | tokenizer.json 19 989 325 B loaded **once** at resident start; ASCENT_STATE 533 KB / 1.4 ms per **daemon** tick | KILL as turn-dominating |
| Serialized wait where a concurrent path exists | YES | decode ceiling 1 is physical; **two workers cannot overlap generate**. CPU tool work on the other worker is possible but `max_rounds=1` never gets there without a generate | `genesis_body` comment L4-6; `decode_concurrency=1` |
| Cooldown / backoff dominates real work | YES, conditionally | 1 440 s DERIVED this afternoon on DEFERRED; 0 s after a 232 s success | `ascent_daemon.py:876-917` |
| Retry loop hidden behind a status field | YES | `queue dry` vs 388 queued / 66 mechanism_refused / 0 pending; `used_gpu_lane_lock:false` vs lock owner `genesis-resident:child_a` | QUEUE + lock dir; HCLI `:1953` |
| Health says down while process is busy | YES, now honest in supervisor | 0.25 s fail-fast; AgentOS still treats it as NOT_READY | `genesis_forever.sh:120-127` vs `genesis_agentos.py:107-114` |
| Daemon healthy while starved | HISTORICAL G011; **live form** is queue-dry with 388 undrained + 66 refused | not a turn-ns number | G011 receipt + §3.4 |

GPU lock LIVE this process: `/tmp/hawking-gpu-lane.lock/owner` =
`genesis-resident:child_a`, `pid` 50196 (gravity in flight while this
file was written). HCLI receipts still say `used_gpu_lane_lock: false`
because `GenesisResidentBackend` hard-codes it (`special_unit.py:1953`).
The adapter does not wrap `gpu_lane_lock.sh`; the **body** acquires the
same dir mutex (`genesis_body/src/main.rs:50-91, 674`). Accounting lie,
not a skipped lock.

`tokenizer.json` 19.99 MB: Python `Tokenizer.from_file` 0.236 s MEASURED
this process. Resident loads it once (`genesis_body` `load_qwen38_tokenizer`).
KILL as a per-turn parse. REOPEN_IF propose re-reads it.

Gravity's four complete turns all `read` a **missing**
`workspace/ops/genesis-candidates` (dir absent MEASURED). Each costs ~220 s
prefill to rediscover the same hole. Model/task waste, not kernel waste.
~880 s DERIVED this afternoon.

---

## 5. Ranked removable cost

Seconds are **per successful HCLI turn** unless marked otherwise.
"Daemon" = `ascent_daemon.py` / `genesis_forever.sh` / `genesis_agentos.py`.
"Resident" = `genesis_body` + `qwen38_hybrid_decode.rs`.
"HCLI" = `lab/hcli/special_unit.py`.

| rank | seconds | class | layer | already running? | fix |
|---:|---:|---|---|---|---|
| 1 | **227.9** MEASURED (98.1 % of native wall) | full reset + sequential prefill of 3527 tok | Resident | no | Keep KV. Match prefix; `step` only the suffix. Do not call `session.reset()` on a continuing session. `ephemeral_kv_reused` already exists and is false. |
| 2 | **~86** PROJECTED of rank 1 | static prefix (capsule 1402 + tools 263) | Resident | no | Even without full continuity, cache the compiled capsule+preamble prefix. Early tokens are the cheap ones (`c0≈39.7 ms` ESTIMATED from short TOKEN_NS + this turn's 91.5 ms tail). |
| 3 | **most of 227.9** PROJECTED, orthogonal | sequential `step` vs batched GEMM prefill | Resident | **not running**; sibling lane `116-prefill-amortization` is assigned the 79 s / 1558-tok parent-propose form of this | Wire a real prefill kernel if/when it exists. Do not wait on it to ship rank 1. |
| 4 | **180 per fast fail** MEASURED mechanism; **1 440** DERIVED this afternoon | cooldown armed by DEFERRED / 0.25 s refuse | Daemon | no (bug) | `agentos_turn_status`: apply 180 s only to completed model ticks (`TURN_COMPLETE` / `TURN_FAILED` after a generate), never to `DEFERRED_*` / `IDLE`. Treat health-timeout + `_resident_process_alive()` as BUSY, not NOT_READY. |
| 5 | **232 per wasted gravity tick** MEASURED × 4 | missing inbox path | Daemon + worker task | no | Create `workspace/ops/genesis-candidates` (or stop prompting it). Four identical failed `read`s. |
| 6 | **~4.4** MEASURED | decode of 49 new tokens at long context | Resident | decode kernel is a different campaign | Not the first lever. Falls with rank 1 (shorter context → ~39 ms/tok). |
| 7 | **8–18** MEASURED post-success; up to **60** SOURCE | `POLL_SECONDS` | Daemon | running at 60 | After AgentOS process exit, skip the sleep or poll 1–2 s. Fair-window 11 % overhead. |
| 8 | **0.51** MEASURED host-around | reset memset + tokenize + lock | Resident | no | Falls out of rank 1 (no reset). Until then, emit `reset_ns` from the live path (`generate_greedy_complete_wall` already times it). |
| 9 | **0.035** MEASURED | claim / compile / session save | Daemon / HCLI | compact prompt already running | Leave. `project_context` on the 35 MB worktree is 43 ms this process. |
| 10 | **0.012** MEASURED | tool exec + persist | HCLI | — | A successful `read`. Not a lever. |
| 11 | **0.0002 × N** MEASURED | contract SHA every health/propose | Daemon + client | integrity required | Cache the verified `GenesisContractSet` in-process. Do not drop the check. |
| 12 | status lie, not ns | `queue dry` / `used_gpu_lane_lock:false` | Daemon / HCLI | G011 text is in comments; live hold string is unchanged | Split the hold: `pending=0 refused=66 queued=388 merge_ready=134`. Report `gpu_lock_owner` from the dir, not a constant false. |

**Do not buy speed by shrinking the sealed capsule authority, dropping
zero-fallback, or skipping the protected-GPU refuse.** Rank 1 reuses KV
for a prompt the compiler already made; it does not delete the contract.

### Already fixed and running

- Capsule instead of 33 197 B files (`inject_runtime_contract`). Running.
- Compact worker prompt 2 400 + 1 600 char caps (`genesis_agentos.py:315-320`). Running. Still 1 839 tok of task.
- Think-close prefix so new tokens can be the tool call (`render_tool_prompt`). Running. Decode 32–49 tok not 256.
- Supervisor does not restart a busy body (`genesis_forever.sh:120-127`). Running.
- Phantom `running` targets reconciled (`ascent_daemon.py:1506-1531`). Running.
- Worker RR gravity↔kernel (`_next_agentos_worker`). Running (fair window).
- Isolated ~35 MB worktrees, HCLI resident backend, protected-GPU refuse on the adapter, zero-fallback. Running.
- `generate_greedy` now forwards `prefill_wall_ns` / `decode_wall_ns` onto the tick (absent on the 16:15 207 s receipt, present from 16:31). Running.

### Already fixed in comments / tests, not governing live rate

- `MAX_GENERATED` is an active-pool cap, not a lifetime cap (comment `:1197-1204`). Live pending=0 for a different reason (mechanism_refused).
- G011 harvest preservation. Live harvest returns 190 reports. Not the wall.

### Not implemented (do not claim they are "already fixed")

- Prefix KV reuse.
- Batched prefill in `generate_greedy`.
- Busy-vs-dead in `body_is_up` / `_resident_is_ready`.
- Cooldown exemption for DEFERRED.
- `reset_ns` on the live propose reply.
- `max_rounds>1` with continuity.

---

## 6. Change list (for Codex / a later lane)

Do not edit these here. Two-writer files are live.

### Resident — rank 1 (the turn)

1. `crates/hawking-core/src/model/qwen38_hybrid_decode.rs:3652` and `:3729`.
   Stop unconditional `session.reset()`. Add a prefix-match path: if
   `session.position>0` and `prompt` starts with the previously consumed
   ids, `step` only `prompt[position..]`. If it does not match, reset.
   Keep the reset path for parent propose and first serve.
2. Same file `:3664-3678`. Prefill loop is the sequential GEMV. When a
   batched prefill exists, call it here. Do not block rank 1 on that.
3. `tools/agentos/genesis_body/src/main.rs:696-729`. Forward `reset_ns`,
   `tokenize_ns`, `lock_wait_ns` on the propose reply (mirror
   `generate_greedy_complete_wall:3728-3730`). HCLI already copies unknown
   native keys only from an allow-list — extend
   `special_unit.py:2150-2174` and `:2056-2080`.
4. `tools/agentos/genesis_body/src/main.rs` accept loop. A `health` op
   must answer while `propose` runs (second thread or non-blocking accept
   + "busy" JSON). Then `CONNECT_TIMEOUT_S=0.25` stops aliasing BUSY as DEAD.

### Daemon — ranks 4, 7, 12

5. `tools/ascent_daemon.py:876-917` `agentos_turn_status`.
   Cooldown only if last dispatch produced a model tick. Read the last
   AgentOS log object / controller result. `DEFERRED_*` and `IDLE` →
   status `idle` immediately.
6. `tools/ascent_daemon.py:1573-1578` dispatch predicate. Require
   `_resident_process_alive()` **and** (health-ok **or** last health was
   busy-alive). Do not launch a tick that will only seal DEFERRED.
7. `tools/genesis_agentos.py:107-114` `_resident_is_ready`. If `health`
   is None and `genesis-resident` pid is alive, return a distinct
   `BUSY` and let `run_once` wait/retry, not DEFERRED.
8. `tools/ascent_daemon.py:1643-1646`. Replace the hold string with the
   underlying counters: `pending`, `refused`, `queued`, `merge_ready`,
   `agentos`. Stop saying "dry" when 388 entries sit unpromoted.
9. `tools/ascent_daemon.py:64` / `:1701`. After AgentOS child exit, do
   not sleep a full `POLL_SECONDS` before the next dispatch.
10. `tools/ascent_daemon.py:948-949`. Leave `--max-rounds 1` until rank 1
    ships. Then raise to 2 so the tool result is consumed without a
    second 228 s prefill.

### HCLI — accounting + stale constants

11. `lab/hcli/special_unit.py:1953`. Stop hard-coding
    `used_gpu_lane_lock: false`. Read `/tmp/hawking-gpu-lane.lock/owner`
    after propose (body holds it during generate; receipt should say so).
12. `lab/hcli/special_unit.py:88` `TOOL_MAX_SEQ_LEN=768` and comment
    `:83-84` ("~10 s decode plus prefill"). Live resident is 8192 and
    230 s. Comment is now a lie; the constant is unused by the resident
    backend. Fix the comment. Do not silently cap the resident at 768.
13. `tools/agentos/genesis_contract.py:3-6`. "4096-token context" → 8192
    live. Capsule-not-files comment is already correct.

### Do not touch

- Sealed contract bytes / binding.
- Zero-fallback reject (`validate_resident_reply:1806-1810`).
- Protected-GPU refuse (`GenesisResidentBackend._refuse_if_protected`).
- Promotion gates / lineage slots.
- Live G0 artifact.

---

## 7. KILLS / REOPEN_IF

| id | statement | status |
|---|---|---|
| K1 | 207 s is decode / TOKEN_NS | **KILL**. 98.1 % prefill on every split receipt. |
| K2 | Full 33 197 B contract files are on the wire | **KILL**. Capsule 5 119 B / 1 402 tok. |
| K3 | Contract SHA / workers.json hash / ASCENT_STATE parse is the turn wall | **KILL**. 0.18 ms / 0.09 ms / 1.4 ms. |
| K4 | tokenizer.json 20 MB is parsed per turn | **KILL**. Once at resident start. |
| K5 | Fair-window gaps rival the turns | **KILL**. 8–18 s vs 232 s. |
| K6 | Two worker sessions give two concurrent generates | **KILL**. `step` is serial; lock is exclusive. |
| K7 | Raising `max_rounds` alone cuts prefill | **KILL** until reset is gone. |
| K8 | `used_gpu_lane_lock:false` means the mutex was not taken | **KILL**. Body takes it. |
| R1 | Prefix KV wired | REOPEN if `session_serve_index>1` and `prefill_wall_ns` drops to suffix×~80 ms. |
| R2 | Batched prefill wired | REOPEN if dispatch count per prefill falls from `prompt_len × L × G` to `L × G`. Discriminator already written on the 1558-tok parent-propose lane. |
| R3 | `reset_ns` on propose | REOPEN the 0.508 s split. |
| R4 | tokenizer re-read on propose | REOPEN K4. |
| R5 | `GENESIS_AGENTOS_MIN_INTERVAL_S` changed live | REOPEN the 180 s arithmetic. |

Cheapest experiment not run (GPU lock is held by a sibling / live gravity):
pipe `reset_ns` onto one propose reply. Do not take the device lock for it.

---

## 8. Evidence index

- Live logs: `workspace/ops/ascent-daemon.log` (394 lines),
  `genesis-agentos.log` (23 sealed objects), `genesis-resident.log`
  (load/listen only; no per-serve lines),
  `genesis-agentos-sessions/genesis-{gravity,kernel}/events.jsonl`,
  `genesis-workers.json`, `genesis-agentos-controller.json`,
  `genesis-worker-checkpoints/{gravity,kernel}/checkpoint.json`.
- Contracts: `contracts/genesis/*.md` bytes+sha as §1.3.
- Body: `tools/agentos/genesis_body/src/main.rs`.
- Prefill: `crates/hawking-core/src/model/qwen38_hybrid_decode.rs:1451-1457,3577-3593,3644-3716`.
- Client: `tools/agentos/genesis_resident.py:42-43,255-343`.
- Compiler: `tools/agentos/genesis_contract.py:164-271`.
- AgentOS: `tools/genesis_agentos.py:90-485`.
- HCLI: `lab/hcli/special_unit.py:954-988,1746-1972,2089-2218`.
- Daemon: `tools/ascent_daemon.py:64,363-414,718-765,876-975,1469-1701`.
- Supervisor: `tools/genesis_forever.sh:62-145`.
- Continuity: `lab/lineage/continuity.py:191-245` (`ephemeral_kv_reused: false`).
- G011: `receipts/ascent-2026-08-16/G011_AUTONOMOUS_LOOP_STARVED.json`.
- Parent 79 s: `workspace/superwave/g1/g1-resident-headroom.md:368-374`.
- G0 TOKEN_NS 39 326 090 / TPS 25.4284 / artifact
  `d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df`:
  worker `bound_generation` + lineage CURRENT (MEASURED this process).

No network. No GPU. No service restart. No two-writer edit.
