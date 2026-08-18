# G1 worker concurrency — one body, four sessions, one profitable decode

Lane: `118-worker-concurrency`. Write scope: this file only.
No GPU, no inference, no health RPC, no launchd/resident/daemon mutation.
Two-writer files were read (live parent checkout and `git show HEAD`); none were edited.

Label key: `MEASURED` = numbers already on disk this lane. `RECEIPT` = prior sealed receipt, not re-timed. `SOURCE` = file:line. `DERIVED` = arithmetic on those. `PROJECTED` = scale under a stated assumption. `ESTIMATED` = not timed. A component microbench is not a turn.

Live two-writer snapshots observed here (Codex is still writing them; HEAD ≠ live):

| path | HEAD sha256 (`git show \| shasum -a 256`) | live parent sha256 |
|---|---|---|
| `tools/ascent_daemon.py` | `5bd2a10b7f5e68f8036a76f9e81d276eca9ac4c06aeb2e20ee8524a5a20c485f` | `f305d848374693c36c67ac0463352876632cedbe279a6553a0d73918625ea055` |
| `tools/genesis_agentos.py` | `a5d3d03134823d3c24ece2cc86470c6ea599b0e5a7b95475d97ac1a0f6b07968` | `026d86915131d02089067d14592c2d6eef4a2a1e0c9057a265a1d1447f654723` |
| `lab/hcli/special_unit.py` | `a902642f6a824e780f089193571e1d42686614ed22eff9d9fc157d4ffc450228` | `f50b3d08d632a7994898cc9ded6c0f15d6264b400f7aa3a6b90bce8c9c080f95` |
| `tools/agentos/genesis_body/src/main.rs` | `9b846109cc7c011dab96d221d9aab104f246bf4e55a15acc74e4b3fddef82878` | identical |

Line numbers for the first three files below are the **live parent**. Resident body lines are HEAD=parent.

---

## 0. Verdict

The organism **cannot profitably run more than one worker generate at a time**. That is **physical**, not merely a queue.

It **can** attach four isolated sessions (already does) and **can** run CPU tools without the GPU lock (already does). It **does not** overlap those tools with another worker's generate, because the scheduler admits one AgentOS process and that process sits in `propose()` for ~200 s.

Concurrent in-process decode is a **MEASURED_NEGATIVE** for aggregate tokens/s (4 sessions: 9.43 vs 26.65). The 4× KV at 8192 is **already paid** (~4.93 GB workspace). Turning on a second decode stream costs throughput and would contaminate a promotion wall. It does not cost extra RAM.

The 207 s turn is **not** tools under the lock. It is `generate_greedy` calling `session.reset()` then `step()` once per prompt token. Live MEASURED: prompt 3209–3534, prefill 201–229 s (**98.4 %** of `wall_ns`), decode 2.8–4.4 s, then a file `read`. Cutting prefill (sibling 116) raises **both** latency and actions/hour. Concurrent decode raises neither.

Live cadence MEASURED 241 s / AgentOS dispatch = **14.94 actions/hour**, not the 17/hour that 3600/207 projects. Serialization is parent ⊕ child_a ⊕ child_b on one body.

---

## 1. What is actually serial, and why

Four layers. Do not collapse them.

| layer | serial? | why | class |
|---|---|---|---|
| Weight set | shared | one `Arc<Qwen38HybridWeights>`, four `attach` | SOURCE |
| Metal command queue | shared | `MetalContext` clones `Arc<Inner>` with **one** `device.new_command_queue()` | SOURCE |
| `step()` | serial per session | 964-dispatch CB + one `commit_and_wait_timed` | SOURCE |
| In-process N-session generate | **worse than 1** | same unique-once DRAM stream; 4× lm_head still ~4× GPU ns | RECEIPT |
| Resident accept loop | serial | one `accept` → blocking `serve_connection` → one `propose` | SOURCE |
| `/tmp/hawking-gpu-lane.lock` | exclusive, **per body** | directory mutex; owner `genesis-resident:{role}` | SOURCE |
| AgentOS dispatch | serial | daemon `one_pass` returns while a tick is live; round-robin next | SOURCE MEASURED |
| Tools / seal | not GPU-serial | run after `propose` returns; lock already dropped | SOURCE |

### 1.1 One body, four sessions, decode_concurrency=1

`SOURCE` `tools/agentos/genesis_body/src/main.rs:3-6,30,474-476,612`

```
Loads Qwen3.8 weights once, attaches four isolated logical sessions
SESSION_ROLES = parent, child_a, child_b, protected_test
Decode remains serialized because the measured concurrent-decode ceiling is 1.
```

`health.decode_concurrency` is a **constant `1`**, not a measurement the process updates.

Live health (prior lane, **not** re-queried here — no RPC this lane):
`/tmp/g1-baseline-remeasure/resident_measure_v2.json` `health_before` pid 74869:

```
decode_concurrency     1
session_count          4
max_seq_len            8192
resident_weight_bytes  14297675776
workspace_bytes        4929305616
session_workspace_bytes 1232326404 × 4
session_serve_counts   parent=6, protected_test=1, child_a=0, child_b=0
artifact_sha           d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
```

Workers have since been served (this-lane `genesis-workers.json`). That health snapshot is the seating, not today's serve counts.

`SOURCE` `tools/genesis_forever.sh:70` starts the body with `--max-seq-len 8192`.

### 1.2 Accept loop is one connection at a time

`SOURCE` `main.rs:755-808,898-901`

```
match listener.accept() {
    Ok((mut stream, _)) => serve_connection(&mut stream, &mut body, &mut stop),
```

`serve_connection` is synchronous. `propose` runs `generate_greedy` to completion before the next `accept`. A second Unix client sits in the kernel backlog. `health` during a generate times out (prior harvest: `HEALTH_ERROR: TimeoutError`; daemon logs `health_busy_process_alive`). Structural.

### 1.3 `generate_greedy` wipes KV every request

`SOURCE` `crates/hawking-core/src/model/qwen38_hybrid_decode.rs:1451-1456,3644-3679,3577-3589`

```
session.reset();          // zeros conv, rec, gqa_key, gqa_value; position=0
for token in prompt { session.step(token)?; }   // this IS prefill
```

`step` encodes the full 964-dispatch graph and `commit_and_wait_timed`. There is **no** batched Qwen3.8 prefill on this path (`rg p3_batched qwen38_hybrid_decode.rs` → empty). Prefill of session A overlapping decode of session B is the same primitive as concurrent decode.

`generate_greedy_parallel` (`:3873-3903`) exists, is `Send`, and is **not** called by the resident.

### 1.4 One shared command queue

`SOURCE` `crates/hawking-core/src/metal/mod.rs:1044-1045,2373-2381,2429-2430`
`SOURCE` `qwen38_hybrid_decode.rs:1238` `context: weights.context.clone()`

Four sessions share one `MTLCommandQueue`. Concurrent `step` from two threads submits to that queue. That is what the shared-sessions probe ran.

---

## 2. Physical ceiling — concurrent decode is a loss

`RECEIPT` `receipts/ascent-2026-08-16/QWEN38_SHARED_SESSIONS.json` (seq=128, lock_held, uniform-q4-v1):

| quantity | value |
|---|---:|
| 1-session tokens/s | 26.653 |
| 1-session median GPU ns | 36,099,333 |
| 4-session aggregate tokens/s | **9.427** |
| 4-session median GPU ns / token | 79–82 M |
| lm_head 1× GPU ns | 1,013,791 |
| lm_head 4× serial GPU ns | 4,144,541 = 4.09× |
| lm_head 4× concurrent GPU ns | 4,022,124 = 3.97× |
| `weights_ptr_shared` | true |
| marginal RSS / session | 173,703,168 B |

`RECEIPT` summary `concurrent_decode_ceiling: 1`.

DERIVED: 9.427 / 26.653 = **0.354×** aggregate. Four streams are not 4×; they are a third of one stream. Concurrent encoder does not amortize the unique-once 13.61 GB GEMV (`g1-residency-reuse.md`, `g1-kv-and-host-gaps.md`).

Process-pool fallback (`g1-resident-headroom.md` / `GENESIS_CHILDREN_CAPACITY`): N=2 aggregate **1.37×** at 55.3 vs 37.9 ms/token, N=4 swap. That is a second Metal copy, ~16 GB each, not the live design.

**KILLS:** "four sessions ⇒ four concurrent worker decodes ⇒ 4× actions/hour."
**KILLS:** "a concurrent Metal encoder reuses the weight stream enough to win a token."
**REOPEN_IF:** quiet-device `generate_greedy_parallel` **N=2** (unmeasured; N=4 is not N=2). Cheapest experiment: same probe as `ascension_qwen38_shared_sessions` with `--sessions 2` after the sibling GPU lane releases the device. Do not run it into a promotion wall.

---

## 3. GPU lock — per-body, whole generate, not whole turn

### 3.1 Who takes it

`SOURCE` `main.rs:31-32,50-90,674-696`

- Path: `/tmp/hawking-gpu-lane.lock` (mkdir-atomic). Same path as `tools/gpu_lane_lock.sh`.
- Acquire is **per `propose`**, owner string `genesis-resident:{session_role}`.
- Held across **entire** `generate_greedy` (reset + prefill steps + decode steps).
- Dropped when `GpuLaneGuard` drops at end of `propose` — **before** the JSON line is even the interesting part: tools have not started.
- Wait: **1500 s** on a live foreign owner, then error. Does **not** inspect protected prefixes. A worker propose that reaches `acquire` will sit behind a promotion lock for up to 25 minutes.

There is no per-session lock. Two roles cannot hold two locks. One directory.

### 3.2 Who does *not* take it

`SOURCE` live `lab/hcli/special_unit.py` `GenesisResidentBackend` (1844-1845, 1881-1884, 1926-1953)

```
Does not acquire gpu_lane_lock.sh
_refuse_if_protected()   # only if owner matches PROTECTED_OWNER_PREFIXES
used_gpu_lane_lock: False
```

`used_gpu_lane_lock: false` on every live turn means the **HCLI client** did not wrap `gpu_lane_lock.sh`. The **resident** still took `GpuLaneGuard`. Do not read that field as "generate ran unlocked."

`ResourceGate.admit(CPU)` is always true (`:332-335`). `ToolExecutor` refuses GPU-shaped argv and never takes the lock (`:627-641`).

`NativeQwen38Backend` (cold binary + `gpu_lane_lock.sh`) is **not** the live worker path.

### 3.3 Scheduler around the lock

Live `tools/ascent_daemon.py`:

| call | condition | effect |
|---|---|---|
| `genesis_proposes` `:1111` | `gpu_lane_busy() or protected_gpu_target_active()` | no parent propose |
| `dispatch_agentos_turn` `:1573-1578` | idle AgentOS **and** `not gpu_lane_busy()` **and** `not protected_gpu_target_running()` | start one worker |
| `one_pass` `:1568-1572` | AgentOS running | **return** — no new protected launch, no second worker |

`protected_gpu_target_active` = pending ∪ running GPU classes (`GPU_DIRTY`, `GPU-LAB`, `GPU_PROTECTED`, `GPU_EXCLUSIVE`, `MIXED`).
`protected_gpu_target_running` = running only.

Asymmetry: a **pending** promotion reserves the body against parent proposals, but **not** against AgentOS. An in-flight AgentOS then blocks the promotion launch for the rest of the 241 s tick. That is not complete yield.

Live `tools/genesis_agentos.py:360-368`: if the lock exists and its pid is alive, the tick returns `DEFERRED_GPU_LANE_BUSY` **before** claiming a worker. Complete skip, including CPU tools.

---

## 4. Live turn anatomy — prefill is the turn

### 4.1 Nine native receipts (MEASURED)

From `workspace/ops/genesis-workers.json` unique `(session, serve_index, prefill_wall_ns)`:

| session | serve | prompt_len | prefill_s | decode_s | wall_s | new_tok | fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| child_a | 8 | 3209 | 200.963 | 3.237 | 204.199 | 38 | 0 |
| child_a | 11 | 3413 | 218.616 | 2.800 | 221.417 | 32 | 0 |
| child_a | 13 | 3446 | 219.999 | 2.791 | 222.790 | 32 | 0 |
| child_a | 15 | 3452 | 221.109 | 2.814 | 223.923 | 32 | 0 |
| child_b | 5 | 3222 | 205.143 | 3.099 | 208.242 | 36 | 0 |
| child_b | 7 | 3468 | 222.971 | 3.274 | 226.244 | 37 | 0 |
| child_b | 10 | 3529 | 228.521 | 3.488 | 232.010 | 39 | 0 |
| child_b | 12 | 3534 | 229.460 | 4.410 | 233.870 | 49 | 0 |
| child_b | 14 | 3527 | 227.924 | 4.390 | 232.314 | 49 | 0 |

First completed child_a turn (`704a50e015a8…`, 2026-08-17T16:46:49Z):

```
prefill_wall_ns 200962729833
decode_wall_ns    3236761958
wall_ns         204199496458
prompt_len 3209
max_new_tokens 256
used_gpu_lane_lock false
transport genesis_resident
session_role child_a
fallbacks 0
```

DERIVED on that row: prefill **98.41 %** of `wall_ns`; decode **1.59 %**.
Dirty-box prefill 200.96 s / 3209 = **62.62 ms/token**. G0 frontier TOKEN_NS is **39.326 ms** (`g1-baseline-remeasure.md`). Do not report 62.62 as a clean token. PROJECTED clean prefill of the same 3209 tokens: 3209 × 39.326e-6 = **126.2 s**. The extra ~75 s is a dirty device (this morning's live_lanes 1→10 in the daemon log), not a second organ.

Prompt grows 3209 → 3534. That is the worker task + last-turn excerpt, not a second copy of the 33 KB contract files.

### 4.2 What is actually on the wire

The sealed set is **hashed into a capsule**, not inlined.

`SOURCE` live `tools/agentos/genesis_contract.py:21-38,184-217,219-263`

| file | bytes | sha256 |
|---|---:|---|
| `contracts/genesis/QWEN38_GENESIS_SYSTEM_DIRECTIVE.md` | 16414 | `881ae469e0287cf386467002d3fc7951524b47054ac6d7f753b94a8e4e3ceff7` |
| `contracts/genesis/GENESIS_CONTINUITY_DIRECTIVE.md` | 11912 | `c4a58bc06575effb8f759dbb22c49abfc65e1957910b18917d45d02592d1fdbc` |
| `contracts/genesis/GENESIS_OUTPUT_LAW.md` | 4871 | `9679490e8ae623a6fdb408fd906a15d676bc55926580f6d7ed60e9ea610c9ada` |

THIS_LANE `load_genesis_contracts()`: `binding_sha256 = 3ef47426958200ff830ea2ec5adce53d3b3347098d459bd7fcddc9a5dc9a179f`.
`runtime_capsule(child_a)` = **5119 B**. System prefix = **5149 B**.
AgentOS tool preamble (no bash) = **1167 B** (`render_tools_preamble`).

`genesis_resident.propose` (`:304-310`) always `inject_runtime_contract` then `raw=True`, except `protected_capability` on `protected_test`.

`SpecialUnit.act` (`:2111,2131-2135`) rebuilds **this user_text only** via `render_tool_prompt(..., user_only=True)`. It does not replay the HCLI transcript into the model. Combined with `session.reset()`, every tick re-prefills capsule + tools + task from position 0.

`max_rounds`: live daemon launches `--max-rounds 1` (`ascent_daemon.py:948-949`). HEAD `genesis_agentos.py` default is 2; live parent default is 1. Live path is one generate + tools, no second generate.

### 4.3 Cadence (MEASURED)

Daemon `started_at` for the last 7 AgentOS launches (ascent-daemon.log):

```
1786986492 kernel
1786986733 gravity   +240.983
1786986974 kernel    +240.981
1786987215 gravity   +240.853
1786987456 kernel    +241.171
1786987697 gravity   +240.645
1786987938 kernel    +240.801
mean 240.906 s  →  14.94 dispatches / hour
```

Round-robin `_next_agentos_worker` (`:921-928`). `GENESIS_AGENTOS_MIN_INTERVAL_S` default 180 is a no-op once a turn exceeds 180 s.

3600/207 = 17.39 is a **projection** from a single-turn wall, not the live cadence.

---

## 5. Concurrent KV cost at 8192 × 4

### 5.1 Formula (SOURCE + DERIVED)

`SOURCE` `qwen38_hybrid_decode.rs:646-714,4500-4509` (`kv_is_the_seq_len_term`: only GQA grows with seq).

RECEIPT identity at seq=128: act 1,691,396 + ΔN 156,893,184 + GQA 16,777,216 = **175,361,796**.

GQA = `16 * seq * 4 * 256 * 4 * 2` = `131072 * seq`.

| seq | GQA K+V | workspace | class |
|---:|---:|---:|---|
| 128 | 16,777,216 | 175,361,796 | RECEIPT + THIS_LANE |
| 2048 | 268,435,456 | 427,020,036 | DERIVED |
| 4096 | 536,870,912 | 695,455,492 | DERIVED |
| 8192 | 1,073,741,824 | **1,232,326,404** | DERIVED = live health |

4 × 1,232,326,404 = **4,929,305,616** = live `workspace_bytes`.
4 × 1,073,741,824 = **4,294,967,296** GQA (4.00 GiB).
Weights 14,297,675,776 + workspace 4,929,305,616 = **19,226,981,392 B**.

| unit | value |
|---|---:|
| bytes | 19,226,981,392 |
| GB (1e9) | **19.227** |
| GiB (1024³) | **17.907** |

The "19.3 GiB" seating number is **19.227 GB**, not 19.3 GiB. This lane did not re-run `footprint(1)` or health (no RPC). RSS of pid 74869 was previously untrustworthy (`pti_resident_size` 13 MB vs 14.3 GB Metal). Use health weights+workspace.

96 GiB box: 103,079,215,104 − 19,226,981,392 = **78.09 GiB** formula headroom if the body were alone. It is not alone. Memory is **not** the bind on 4 concurrent sessions. They are already attached.

### 5.2 Allocated vs live occupancy

Every generate `reset()`s. Live seq ≈ prompt_len + new ≈ 3250–3580, not 8192.
DERIVED used GQA at seq=3250 × 4 = 1,703,936,000 B (1.59 GiB) vs 4.00 GiB allocated.
`mha_decode_f32` TG cap seq ≤ 8064 (`g1-kv-and-host-gaps.md`). Live turns never approach it because they reset.

Enabling concurrent **decode** of the four already-attached sessions is memory-neutral.
A second **process** is not: RECEIPT ~15.66 GB phys at seq=128, ~16.72 GB at 8192.

---

## 6. Non-model work is already off the GPU lock

A worker tick (`genesis_agentos.run_once` live `:344-443`):

1. Stopfile / gpu-busy / resident-ready checks (CPU).
2. CAS-claim one worker; checkpoint (`reason=AgentOS HCLI turn before model action`).
3. `unit.act(...)` → `GenesisResidentBackend.complete` → `propose` → resident lock + generate.
4. Parse `<tool_call>`; `ToolExecutor.run` (read/write/grep/pytest/cargo_test/prepare_candidate/submit_candidate).
5. `_finish_worker` + `lab.receipts.seal`.

Steps 4–5 run **after** `GpuLaneGuard` drop.

Live tools on the completed turns are almost all `read`. Gravity repeatedly `read`s a missing `workspace/ops/genesis-candidates` (fail, milliseconds). Kernel `read`s worktree dirs/files (success, still a file read).

ESTIMATED non-generate remainder from cadence − wall: 241 − 224 ≈ **17 s** (gravity last) and 241 − 232 ≈ **9 s** (kernel last). That remainder is process start + `body_is_up` health + context compile + inject + tool + seal, **not** a second generate. It is **not** a turn-level measurement of any one of those.

**KILLS:** "tools run under the GPU lock and steal decode from every other worker."
They steal the **AgentOS dispatch slot** (~241 s, of which ~230 s is generate). They do not hold `/tmp/hawking-gpu-lane.lock`.

Moving tools further "outside the lock" does not create decode time. The lock is already down. Overlapping worker B's `read` with worker A's generate is legal and would save ESTIMATED <20 s per 241 s (**<8 %**). It does not change the 15 actions/hour ceiling.

CPU tests (`pytest`, `cargo_test` minus hawking-core) are also legal during a generate. Today they cannot start, because the only AgentOS process is blocked in `propose`.

`body_is_up()` (`genesis_resident.py:276-277`) is a **health RPC**. During another generate it times out and `propose` returns `None` → `ResidentRefused: resident client returned no result` (live gravity 16:05:39Z, 17:05:08Z). That is a structural refuse, not a queue.

---

## 7. Throughput vs latency

Different objectives.

| lever | single-turn latency | organism actions/hour | contamination risk |
|---|---|---|---|
| Stop `reset()` + keep capsule KV (sibling 116) | **large drop** (PROJECTED 204 s → ~3–4 s decode-only on a dirty box; ~1.3 s at clean 39.3 ms × 32 tok) | **large rise** (PROJECTED tens–hundreds / hour) | none if still one generate |
| Batched prefill kernel (does not exist for Qwen3.8) | unknown; would have to beat per-token `step` | same | measurement only on quiet device |
| Overlap B-tools with A-generate | none on A's generate | **<8 % ESTIMATED** | none (CPU) |
| Two in-process generates | **worse** (N=4: ~2.25× GPU ns/token) | **worse** (0.35× aggregate) | **yes** — second CB on the promotion GPU |
| Second process-pool body | worse per token (N=2: 55 vs 38 ms) | +37 % RECEIPT dirty N=2 | **yes** + ~16 GB |
| Fairness already landed (parent max_new 512, 1 propose/pass, round-robin) | parent no longer 2600-token monopolies | workers get the body at all | none |

Fairness `SOURCE` live `ascent_daemon.py:571-580,921-928,1563-1587`. This is why child_a/child_b now have serve_index > 0. It does not make two decodes cheap.

**Do not buy throughput with a second decode stream.** The DRAM roof is unique-once. Two streams time-share the same 13.61 GB.

---

## 8. Protected measurement — yield must be complete

Promotion-grade complete-token requires a quiet device. Sharing the GPU "a little" is contamination.

### 8.1 What already yields

- `ResourceGate` never waits on the lock; it pauses (`special_unit.py:311-314`).
- `GenesisResidentBackend._refuse_if_protected` raises `ResidentRefused` if the owner matches prefixes (`:1881-1884`).
- `run_once` returns `DEFERRED_GPU_LANE_BUSY` if any live lock owner (`genesis_agentos.py:360-368`).
- `genesis_proposes` returns "" if lock busy **or** a GPU target is pending/running (`:1111`).
- `one_pass` will not **launch** a new protected target while AgentOS is running (`:1568-1572`).

### 8.2 What does not yield completely

1. **Prefix hole.** `PROTECTED_OWNER_PREFIXES` (`:149-155`) is `q80-`, `qwen80-`, `dsv4f-`, `dsv4-`, `auto-dsv4f-`, `deepseek`. A Qwen3.8 complete-token owner such as `qwen38-…`, `g0-…`, or `genesis-resident:protected_test` is **not** protected at the HCLI gate. Relying on `gpu_lane_busy()` in AgentOS covers the *next* tick, not a client that already passed the check.

2. **Pending ≠ reserved for AgentOS.** `dispatch_agentos_turn` uses `protected_gpu_target_running` only. A `pending` `GPU_PROTECTED` promotion can lose the race to a 241 s worker tick, then sit behind it. Parent proposals already treat pending as reserved. Workers do not.

3. **No preemption.** An in-flight `step()` owns the queue until `commit_and_wait_timed` returns. There is no yield inside the 964-dispatch CB. A 3209-token prefill is ~200 s of uninterruptible Metal. Shortening the turn (prefix KV) shrinks the window. Concurrent decode **widens** it.

4. **Resident waits 1500 s** instead of refusing a protected owner (`main.rs:32,74-86`). If anything reached `acquire` while a promotion held the lock, it would poll every 200 ms for 25 minutes. Live AgentOS usually never gets there (`DEFERRED_GPU_LANE_BUSY` / `body_is_up` fail). The wait is still the wrong default next to a promotion lock.

5. **Health during generate** is not a second decode, but it is a second client on a single-threaded accept loop. Not contamination; it is why health times out.

**Do not** add a second generate, a second queue, or a "reduced share" scheduler for promotion. The only legal concurrency beside a promotion wall is CPU work that never opens Metal.

---

## 9. KILLS / REOPEN_IF

| ID | claim | status |
|---|---|---|
| K1 | Four attached sessions can decode concurrently for more actions/hour | **KILLS** (0.35× aggregate RECEIPT) |
| K2 | Concurrent encoder amortizes lm_head / weight DRAM | **KILLS** (3.97× vs 4.09×) |
| K3 | Tools hold the GPU lock and steal decode | **KILLS** (lock dropped before `ToolExecutor`) |
| K4 | Re-sending the 33 KB contract files is the 3209-token prompt | **KILLS** (capsule is 5119 B; files are hashed) |
| K5 | 19.3 GiB is a GiB figure that forbids a second session | **KILLS** (17.91 GiB / 19.23 GB already includes 4 sessions) |
| K6 | Host identity bookkeeping is this turn's wall | **KILLS** (prior genome; here prefill `step` count is the wall) |
| R1 | N=2 in-process parallel generate on a quiet device | **REOPEN_IF** measured |
| R2 | A Qwen3.8 batched-prefill kernel that is not per-token `step` | **REOPEN_IF** it exists and beats 39.3 ms × prompt_len |
| R3 | Prefix-KV / no-reset (lane 116) lands | **REOPEN_IF** — then the serial object is ~30–50 decode tokens, and CPU-overlap + fairness matter more than they do today |
| R4 | Promotion owner strings that are not in `PROTECTED_OWNER_PREFIXES` | **REOPEN_IF** a G0 complete-token lane name is chosen; fix prefixes first |

Cheapest experiment this lane **did not** run (GPU authority held elsewhere): N=2 `generate_greedy_parallel` after the sibling lane releases `/tmp/hawking-gpu-lane.lock`. Until then treat N>1 in-process decode as a loss.

---

## 10. Change list (for Codex / a later lane — not applied here)

Do **not** touch the two-writer set from this lane. Apply on a quiet writer.

### 10.1 Do not implement

1. `tools/agentos/genesis_body/src/main.rs:898-901` — do not accept a second stream and `propose` it while another `generate_greedy` is live.
2. `main.rs:696` — do not call `generate_greedy_parallel`.
3. Do not add a second `new_command_queue()` for child sessions (`metal/mod.rs:2373`). Same DRAM.
4. Do not spawn a process-pool second Qwen3.8 body to raise HCLI actions/hour.
5. Do not weaken `validate_resident_reply` fallbacks==0, capsule injection, or `protected_capability` confinement (`special_unit.py:1746-1814`, `genesis_resident.py:295-310`).

### 10.2 Protected yield — IMPLEMENT_READY

1. **`tools/ascent_daemon.py:1573-1578`** (`one_pass` AgentOS admit). Today: `not protected_gpu_target_running(state)`. Change: also refuse when a **promotion-grade** target is `pending` (`GPU_PROTECTED` / explicit `PROMOTION_MEASUREMENT` class). Keep the existing "pending catalog GPU_DIRTY must not starve AgentOS forever" behavior behind a distinct class so this is not a blanket `protected_gpu_target_active()`. `genesis_proposes:1111` already uses `active()`.
2. **`lab/hcli/special_unit.py:149-155`** `PROTECTED_OWNER_PREFIXES`. Add every owner string a G0 complete-token / quiet-device lane will write (`qwen38-`, `g0-`, `complete-token`, `genesis-protected`). Until the exact owner is named, fail closed on any lock owner that is **not** `genesis-resident:parent|child_a|child_b`.
3. **`tools/agentos/genesis_body/src/main.rs:64-86`** `GpuLaneGuard::acquire`. If `owner` is a protected prefix, **return Err immediately**. Do not wait 1500 s. Leave the 1500 s wait only for `genesis-resident:*` so a worker can serialize behind a sibling worker, not behind a promotion.
4. **Do not** add mid-`step` preemption. Document: in-flight generate is uninterruptible; the yield is "never start." Prefix-KV (116) is what shrinks the uninterruptible window.

### 10.3 Optional, small, after 116 — not a 15→30/hour lever today

1. **`tools/genesis_agentos.py:360-368`**. Split `DEFERRED_GPU_LANE_BUSY` into: refuse **model** complete; still allow a CPU-only tool tick that does not call `propose`. Wins ESTIMATED <8 % until prefill dies.
2. **`tools/ascent_daemon.py:1563-1587`**. After 116, two AgentOS processes may run if and only if at most one is in `propose`. Today a second `propose` is a physical loss.

### 10.4 Not this lane (owned elsewhere)

- `qwen38_hybrid_decode.rs:3652` `session.reset()` every request — prefill amortization (116).
- Batched prefill — no Qwen3.8 kernel to wire.
- Parent `max_new` 512 / 1 propose/pass / worker round-robin — already live.

---

## 11. Evidence (raw pointers)

### 11.1 Shared-sessions receipt (physical kill)

`receipts/ascent-2026-08-16/QWEN38_SHARED_SESSIONS.json`:
`baseline_1_session.tokens_per_s = 26.653226333067778`
`parallel_n_session.aggregate_tokens_per_s = 9.42718104951944`
`fanout_n_concurrent.gpu_ns = 4022124` / `fanout_1.gpu_ns = 1013791`
`marginal_rss_per_session_bytes[].delta = 173703168` (×3)
`workspace.total_bytes = 175361796` at `max_seq_len = 128`
`resident_weight_bytes = 14297675776`

`QWEN38_SHARED_SESSIONS_SUMMARY.json:24` `"concurrent_decode_ceiling": 1`

### 11.2 Live HCLI native block (first successful gravity generate)

`workspace/ops/genesis-workers.json` gravity `tool_results[3].act.native` (recorded_at 2026-08-17T16:46:49Z):

```
prefill_wall_ns 200962729833
decode_wall_ns    3236761958
wall_ns         204199496458
prompt_len 3209
new_tokens len 38
fallbacks 0
session_role child_a
used_gpu_lane_lock false
transport genesis_resident
```

Tool that followed: `read` `.../workspace/ops/genesis-candidates` → `ok: false` `missing path`.

### 11.3 Resident lock + reset + step

```
main.rs:674  let _gpu_guard = GpuLaneGuard::acquire(session_role)
main.rs:696  generate_greedy(session, &prompt_ids, max_new)
qwen38_hybrid_decode.rs:3652  session.reset()
qwen38_hybrid_decode.rs:3664-3678  for prompt token: session.step(token)
qwen38_hybrid_decode.rs:3589  tcb.commit_and_wait_timed()
```

### 11.4 Seating health (prior, no RPC this lane)

`/tmp/g1-baseline-remeasure/resident_measure_v2.json` `health_before`:
`decode_concurrency=1`, `session_count=4`, `max_seq_len=8192`,
`resident_weight_bytes=14297675776`, `workspace_bytes=4929305616`.

### 11.5 This lane did not

- send `{"op":"health"}` or `propose`
- take `/tmp/hawking-gpu-lane.lock` (observed absent at one poll; not used as a liveness claim)
- stop/restart launchd, resident, or ascent_daemon
- edit any two-writer file or any tracked file
- run N=2/N=4 generate
