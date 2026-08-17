# G1 resident harvest — 2026-08-17 ~10:59 local

Read-only harvest of live Genesis G0 on this machine.
Write scope: this file only. No GPU measurement. No process mutation.
`ps` is Seatbelt-denied here; liveness is `pgrep -lf` + `os.kill(pid, 0)`.

Harvest clock: local 2026-08-17. File ages below are from `stat` at ~10:58–10:59.

---

## 1. Is G0 running? Healthy? Starved?

### 1.1 Process table (kernel, not a status field)

`pgrep -lf` at harvest time (second snapshot ~10:58 after first propose PID died):

```
74858 /bin/bash /Users/scammermike/Downloads/hawking/tools/genesis_forever.sh
74912 .../Python .../Downloads/hawking/tools/ascent_daemon.py loop
74869 /Users/scammermike/Downloads/hawking/workspace/ops/build/rust/release/genesis-resident
      --artifact-root .../qwen38-27b/uniform-q4-v1
      --tokenizer .../qwen38-27b/bf16/tokenizer.json
      --socket .../workspace/ops/genesis-resident.sock
50752 .../Python .../genesis_resident.py propose --session parent --max-new-tokens 2600
      MEASURED BOTTLENECK NOW: GPU body of the 964-dispatch token, 35950374 ns ...
```

`os.kill(pid, 0)`:

```
26266 DEAD          # first propose seen at ~10:53 (attention-codec bottleneck)
50752 alive (perm)  # second propose, in flight
74869 alive (perm)  # body
74858 alive (perm)  # forever wrapper
74912 alive (perm)  # ascent_daemon loop
```

launchd: `~/Library/LaunchAgents/com.hawking.genesis.plist` Label `com.hawking.genesis`, ProgramArguments `tools/genesis_forever.sh`, KeepAlive.SuccessfulExit=false, WorkingDirectory `/Users/scammermike/Downloads/hawking`. No `GENESIS_STOP` file (`ls` → No such file).

### 1.2 Health RPC (do not trust; it timed out)

Unix socket exists: `workspace/ops/genesis-resident.sock` (srwxr-xr-x, inode born 10:31:24).

Health RPC `{"op":"health"}` with 0.75s timeout:

```
HEALTH_ERROR: TimeoutError timed out
```

This is **not** evidence the body is dead. A parent `propose` was on the socket. Daemon itself logs the same condition as `health_busy_process_alive` then `health_recovered_after_busy` (`workspace/ops/ascent-daemon.log` lines 221–225). `genesis_resident.py` documents: liveness is process state / socket answering, never a status file.

No second long health RPC was sent (would queue behind the live generate).

### 1.3 GPU lock

```
/tmp/hawking-gpu-lane.lock/owner = genesis-resident:parent
/tmp/hawking-gpu-lane.lock/pid   = 74869
lock mtime 10:58:35 (refreshed when propose 50752 started)
```

The resident **is** the GPU-lock owner. A propose is a generate on the live Qwen3.8 body. Concurrent G1 GPU lanes will block on this mutex.

### 1.4 Body load (resident log, last boot)

`workspace/ops/genesis-resident.log` lines 266–287 (last cycle):

```
genesis-resident: loading .../qwen38-27b/uniform-q4-v1
qwen38-decode opening Metal + 755 catalog tensors
... upload 0/755 .. 750/755
genesis-resident: body resident 3.435s weight_bytes=14297675776
genesis-resident: listening .../genesis-resident.sock pid=74869
```

`weight_bytes=14297675776` = 13.315 GiB. Matches uniform-q4-v1 on disk, not a G1 pack.

The same log shows **many** earlier boots today (pids 6728, 23519, 91218, 85802, 34122, 34583, 40132, 40316, 43872, 44834, 49708, 54526, then 74869), each `exit 0` then reload. Repeated "lineage identity is stale; loading measured artifact ... sha d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df".

### 1.5 Summary-field vs counts (the prior-session failure mode, live again)

Last **completed** daemon tick that wrote a full report (`ascent-daemon.log` line 220, mtime 10:52:56; no newer tick line because `one_pass` is blocked inside `generate_targets` → `genesis_proposes`):

```
{"disk_free_gib": 318.6, "live_lanes_all_repos": 0, "queued": 295,
 "merge_ready": 132, "needs_composition": 15,
 "phantom_targets_reconciled": 1, "generated": 0, "pending": 0,
 "dead_lanes_found": 10, "our_live_lanes": 0, "launched": null,
 "hold": "queue dry - harvest supplied no new pending target"}
```

| field (summary) | value | underlying count | verdict |
|---|---|---|---|
| `hold` | `queue dry - harvest supplied no new pending target` | PROMOTION_QUEUE.entries = **295** | dry string is about ASCENT_STATE pending, **not** the promotion queue |
| `queued` | 295 | 295 entries, `promoted=False` on **all 295** | queue is full of unpromoted science |
| `merge_ready` | 132 | disposition MERGE_READY ∧ ¬promoted = 132 | merge backlog, not dry |
| `needs_composition` | 15 | 15 | composition backlog |
| `pending` | 0 | ASCENT_STATE targets with status=pending = **0** | this field is true |
| `generated` | 0 | new targets created this tick = 0 | true for last completed tick |
| `our_live_lanes` | 0 | last completed tick; a propose is now live | stale by the time you read it |
| CURRENT.live / launched | true / true | lineage `resident_pid` = **40316**; live body = **74869** | lineage flags do not match the live PID |
| CURRENT.research_state | null | — | no research-state object on the seated instance |

ASCENT_STATE.json `targets` n=220, counted 2026-08-17 10:57:

```
status: stale_no_process=125, mechanism_refused=66, retained*=15,
        launch_failed=7, deauthorised=6
ACTIVE pending+running = 0
qwen38 by status: mechanism_refused=66, stale_no_process=10, launch_failed=7
auto_generated ACTIVE = 0
```

Finished **successful** G0 lanes from this morning (deltanet, try13, try14, normalization, hcli-backfill) are recorded as `stale_no_process`, not SHIPPED. The daemon closes a `running` target when the launcher PID dies (`ascent_daemon.py` 1243–1267). A completed grok-run looks the same as a crash.

PROMOTION_QUEUE disposition × status (all 295, `promoted` Counter `{False: 295}`):

```
111 MERGE_READY × SHIPPED
 18 MERGE_READY × PARTIAL
  1 MERGE_READY × PROVEN_NEGATIVE
  1 MERGE_READY × WIN
  1 MERGE_READY × MEASURED
105 NO_REPORT_MANUAL_REVIEW × UNKNOWN
 18 NO_REPORT_MANUAL_REVIEW × NO_REPORT_exit_143
 15 other NO_REPORT_* / CLOSED / BLOCKED / …
 15 NEEDS_COMPOSITION
  1 CHECK_FOR_UNCOMMITTED_WORK
```

model split: q80=237, qwen38=43, dsv4f=15. q80/dsv4f are sealed (weights deleted); they still occupy the queue.

### 1.6 Health verdict

G0 is **not process-dead**. Body, forever wrapper, and daemon loop are up. A parent propose is in flight. The organism produced five finished qwen38 lanes this morning.

G0 **is launch-starved** in the sense the prior session warned about:

1. ASCENT_STATE active pool is empty (`pending=0`, `running=0`).
2. `one_pass` writes `hold: queue dry` whenever the post-admission pending list is empty (`ascent_daemon.py` 1320–1323), even with 295 queued / 132 merge-ready.
3. `generate_targets` walks harvested NEXT_BOTTLENECK strings **serially**, each calling `genesis_proposes` (timeout 1800 s). Daemon log has not grown since 10:52:56 (~7 min) because the current tick is stuck in that call. Two proposes observed: pid 26266 (attention-codec bottleneck, now dead) then pid 50752 (964-dispatch GPU-body bottleneck, live).
4. 66 qwen38 targets are `mechanism_refused` (empty mechanism or semantic duplicate of try12 N=2 GEMM). 12 attempts already exist on the attention-codec bottleneck (4 named, 8 empty).
5. Named receipts from this morning's science **did not land** on the canonical `receipts/ascent-2026-08-16/` tree (see §4).
6. Lineage CURRENT pid 40316 is a prior body; live body is 74869. `research_state` is null. CANDIDATE is empty. Nothing has been promoted.

Not the G011 "five faults, daemon logs healthy" corpse. Closer: **body busy generating proposals; harvest-launch pipeline empty; promotion pipeline never drains.**

**Blocker, precise:** `tools/ascent_daemon.py:generate_targets` is inside a blocking `genesis_proposes` against the single-threaded resident socket, while ASCENT_STATE has zero `pending`/`running` GPU targets. The GPU lock is held by `genesis-resident:parent` for that propose. Superwave G1 GPU lanes and the next G0 measurement lane both wait. This is a scheduler serialisation, not a dead body.

Cheapest experiment to settle health-RPC independently: after the current propose exits, one `{"op":"health"}` with 0.75 s timeout. Do not send it while a propose is live.

---

## 2. Declared NEXT_BOTTLENECK

There is no single field. Four layers disagree. Do not collapse them.

### 2.1 What the live body is answering **right now** (authoritative for "current declared")

In-flight propose pid 50752, text from `pgrep -lf 'genesis_resident.py propose'`:

```
MEASURED BOTTLENECK NOW: GPU body of the 964-dispatch token, 35950374 ns
(35.950 ms, 98.1% of the 36.642 ms complete wall). Named-fixed residue is
wait_minus_gpu 574618 ns (rose +157719 vs baseline). Remaining
encode_host_prepare is 82044 ns.
```

Source of that string: PROMOTION_QUEUE entry `q38-genome-icb-rebase-20260816-184205` status=WIN, disposition=MERGE_READY.

The prompt is built by `_genesis_prompt` (`tools/ascent_daemon.py` 564–594). It also injects standing negatives (GEMV-fusion +10.68 ms; hot/cold 2.5%; N independent GEMVs still 4×) and the still-open item: single dispatch `W @ [x1..xN]`.

### 2.2 What the previous propose (pid 26266, now dead) was answering

```
MEASURED BOTTLENECK NOW: Qwen3.8 complete token still 38.217 ms (38216792 ns)
at 4.253 BPW — attention+embed+norm stay 5.20 GB / 74%. Coherent generate
below that needs an MLP that is not r160_b3 down_proj at 0.132 BPW;
attention below Q4 at 0.99 is closed.
```

Source: `q38-attention-codec-below-q4-20260816-184819` STATUS=PROVEN_NEGATIVE.

### 2.3 Most recently **measured** NEXT_BOTTLENECK from a finished G0 lane (this morning)

From harvested grok-reports, newest last:

| finished | STATUS | NEXT_BOTTLENECK (as harvested, ≤400 chars) | label |
|---|---|---|---|
| try12 03:01 | SHIPPED | `weight_addressing 19952248 ns (mlp 12934375)` | DIRTY_ENGINEERING addr_probe |
| deltanet 09:25 | SHIPPED | `gated_rmsnorm_48 still 1330791 ns GPU (48×16-wide); weight_addressing remains ~21e6 ns existential` | DIRTY isolated + dirty wall |
| hcli-backfill 09:34 | SHIPPED | `live act() through genesis_resident.propose still fail-closes ... 100-TPS remains the protected DeltaNet kernel lane.` | not a token cost |
| try13 09:45 | SHIPPED | `weight_addressing unique-once Q4 DRAM 19486582 ns (mlp class 12651708 ns)` | DIRTY addr_probe |
| try14 10:01 | SHIPPED | `weight_addressing 19818414 ns (DIRTY_ENGINEERING); unique-once Q4 DRAM, mlp 12733416 ns` | DIRTY addr_probe |
| normalization 10:20 | PARTIAL | `normalization 1-TG launch tax 2270249 ns isolated` | DIRTY isolated |

Harvest regex is `^NEXT_BOTTLENECK:\s*(.+)$` (`ascent_daemon.py` 334, 387). Normalization's grok-report contains **two** such lines; harvest takes the first (`normalization 1-TG launch tax 2270249 ns isolated`). The second is more precise: `residual RMSNorm 1-TG launch tax 2270249 ns isolated (production still pays ~18 us x 129 serial waves; GEMV fusion already REFUTED)`.

### 2.4 Synthesis catalog (what actually launched this morning)

Recent launched ids in ASCENT_STATE (all now `stale_no_process`):

- `auto-qwen38-weight-addressing` ← `qwen38-synthesis-per_layer_per_head_assignment` / `weight_addressing 21293103 ns`
- `auto-qwen38-deltanet` ← `qwen38-synthesis-fuse-deltanet-activation-tails` / `deltanet`
- `auto-qwen38-weight-addressing-try13` ← `qwen38-synthesis-addressing_layout_not_codec`
- `auto-qwen38-weight-addressing-try14` ← `qwen38-synthesis-host_gpu_partition_of_addressing`
- `auto-qwen38-normalization` ← `qwen38-synthesis-collapse_rmsnorm_launches` / `normalization 2367415 ns`

`dry_synthesis_source` runs when qwen38 history exists and no active GPU target (`ascent_daemon.py` 974–987). That is why this morning's work was catalog items, not the 38.217 ms attention-codec text.

A contract file `workspace/ops/ascent-lanes/auto-qwen38-live-act-through-genesis.md` mtime 10:46:39 exists (source lane `genesis-agentos-hcli-backfill`, mechanism = the still-open W@X GEMM). **No** matching grok-run task dir. Not in PROMOTION_QUEUE (queue last write 10:42:33). Written, not launched.

### 2.5 Historical G024 ledger (DIRTY_ENGINEERING, **partially superseded**)

`G024_QWEN38_TOKEN_NS.json` ranks (lines 46–58):

```
weight_addressing 21293103 ns  60.44%  EXISTENTIAL
deltanet          3732795 ns   10.60%
gqa               2443471 ns    6.94%
normalization     2367415 ns    6.72%
```

G024 also claims `401.6 / 411.51 GB/s = 97.6%` (line 63). **That ceiling is REFUTED** by `HONEST_ROOF_WEIGHT_ADDRESSING.md` and by the organism's own propose prompt. Do not consume 97.6% / 411.51 as a floor. Consume the ranked **ns** as a DIRTY 2026-08-16 ledger, then prefer this-morning addr_probe medians (~19.5–19.8 ms) as the current dirty addressing measurement.

---

## 3. Lineage

File: `/Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json`
mtime 2026-08-17 09:15:23 (age ~1.7 h at harvest). schema `hawking.lineage.state.v1`.

```
armed: true
valid_count: 2
zero_valid_genesis: false
CANDIDATE: null
```

Events: `install` genesis-qwen38-g0 at 2026-08-17T13:10:59Z; `snapshot_lkg` same instant; `bind_live_observed` at 2026-08-17T13:15:23Z.

### CURRENT (lines 61–105)

| field | value | note |
|---|---|---|
| instance_id | genesis-qwen38-g0 | |
| generation | 0 | no G1 nominated |
| live / launched / valid / terminated | true / true / true / false | flags stale vs live PID |
| representation_bpw | 4.2527 | rounded |
| physical_bpw | 4.252735126866492 | from live bind |
| complete_token_ns | 37879375 | from current-main wall receipt |
| tps | 26.399590806342502 | 1e9/37879375 |
| artifact | qwen38-27b/uniform-q4-v1 | sha d650a757… |
| kernel_genome_sha | 51abdf7be388d62ba080d13a1f97a18ab8b1114c0a6968e9d0f04d109d3efcd1 | qwen_uniform_q4.metal |
| runtime_sha | ae0bc8defd84a8a1a5cd1c4598224f370c0cfce83a0904e275cbb33df84d32c2 | genesis-resident binary |
| identity.resident_pid | **40316** | **not** live 74869 |
| research_state | **null** | |
| silent_fallback_ids | [] | |
| capability | coherence=1, complete_token_discipline=1, engineering=1 | self-declared, not re-verified here |
| timing_label on bind | DIRTY_ENGINEERING | |

Live-bind payload also records `lineage_children: 0`, `load_count: 1` (that body's first load), `session_roles: [parent, child_a, child_b, protected_test]`.

### LAST_KNOWN_GOOD (lines 107–141)

Same instance_id and generation. **Different numbers:**

| field | LKG | CURRENT |
|---|---|---|
| complete_token_ns | **35227918** | 37879375 |
| tps | **28.386576805362157** | 26.399590806342502 |
| physical_bpw | null | 4.252735… |
| kernel_genome_sha | 688d8b87bddc6baa7bd083229f1b1c7c96ea01adb893c42a98ad534c3341cd7e | 51abdf7b… |
| runtime_sha | ecfc1cac8742d51dac35bca3c702520a7409089914b9dd637d7927baae0cfe72 | ae0bc8de… |
| live / launched | false / false | true / true |

LKG 35227918 ns is G024's closed ledger (`G024_QWEN38_TOKEN_NS.json` `sum_components_ns` 35227918 / `total_token_ns` 35227917). CURRENT is the later protected current-main wall. LKG is the install snapshot **before** live bind, not a better genome.

### What lineage does **not** contain

- No CANDIDATE. No G1 occupant.
- No research_state (no bottleneck, no bus cursor, no last proposal).
- No pid of the live body.
- No promotion events after install+bind.

---

## 4. Receipts and negative science

### 4.1 Canonical receipts tree, today

`find receipts/ -mtime 0` on the main repo produced **17** files. Science-relevant:

| mtime | path | what it is |
|---|---|---|
| 10:42:33 | PROMOTION_QUEUE.json | harvest index, 295 entries |
| 10:41:32 | ASCENT_STATE.json | 220 targets; **generated field still "2026-08-16"** |
| 10:31:02 | MECHANISM_DEDUP_EVIDENCE.json | gate fixtures; synthesis example `per_layer_per_head_assignment` |
| 09:15:23 | GENESIS_LINEAGE_CURRENT.json | seated G0 |
| 01:17:57 | QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.verification.json | identity check, **did not re-run GPU** |
| 00:54:08 | QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.json | protected DIRTY wall used by lineage CURRENT |
| 00:43:25 | GENESIS_GENERATOR_RESIDUAL_ADJUDICATION.json | shared_r64 REFUTED |
| 00:35:37 | HONEST_ROOF_WEIGHT_ADDRESSING.{json,md,reduced.json} | 411.51/97.6% REFUTED |
| 00:30:10 | genesis-resident/* , GENESIS_RESIDENT_BODY.* | body bring-up, not token science |

NEGATIVE_SCIENCE_REGISTER.json mtime **2026-08-16 15:58:42** (age ~19 h). It has **not** absorbed this morning's kills. It ends at NS-038. The attention-codec grok-report names `QWEN38_ATTENTION_BELOW_Q4.json (NS-039)` — that file is **not** on the canonical tree; it exists only in the worktree (below). Register `counts`: entries=38, listed_items_covered=10, unlisted_found=28, contradictions=8, p0_contradictions=3.

### 4.2 Protected current-main complete-token wall (lineage CURRENT authority)

`QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.json`:

```
timing_label: DIRTY_ENGINEERING
vehicle.complete_physical_bpw: 4.252735126866492
vehicle.role: PROFILING_ORACLE_ONLY
vehicle.not_optimized: true
authority.headline_complete_wall_ns_per_token: 37879375
authority.headline_complete_wall_ms_per_token: 37.879375
authority.headline_complete_tps: 26.399590806342502
authority.headline_gpu_ns_per_token: 36635958
authority.pooled_steady_complete_wall_ns: n=186, median=37879542,
  min=37286584, max=39345500, mean=37983095.23
fallbacks: 0
kernel: qwen_uniform_q4_group64_matvec_geo_tpr64_tg128
```

Verification receipt `claim_boundary`: `gpu_work_launched: false`, `runtime_executed: false`, `wall_rederived_from_raw_capture: true`. So the 37.879 ms figure is a **re-derived identity of an existing capture**, not a new 01:17 GPU run.

`gpu_proxy_verdict` contains projected TPS from GPU-only (56–63). Those are **projections**, not measured complete-token TPS. Measured complete TPS is 26.40.

Historical campaign claim (4.2527 BPW / 26.4 TPS / ~37.9e6 ns) **matches CURRENT**, not G024's 35.228 ms. Still DIRTY_ENGINEERING. This harvest did not remeasure.

### 4.3 This morning's G0 science — receipts live in worktrees, not on main

| receipt (named in grok-report) | on main receipts/ | on lane worktree |
|---|---|---|
| QWEN38_NORM_COLLAPSE.json | **missing** | present |
| AUTO_QWEN38_WEIGHT_ADDRESSING_TRY14.json | **missing** | present |
| AUTO_QWEN38_WEIGHT_ADDRESSING_TRY13.json | **missing** | present |
| AUTO_QWEN38_WEIGHT_ADDRESSING_TRY12.json | **missing** | missing there too (only grok-report) |
| QWEN38_DELTANET_ACTIVATION_TAILS.json | **missing** | present |
| QWEN38_ATTENTION_BELOW_Q4.json | **missing** | present (Aug 16 19:09) |

Consume the grok-reports + worktree receipts. Do not look at an empty main path and conclude the experiment was not run.

### 4.4 Kills this organism produced (MEASURED_NEGATIVE) — consume these

All DIRTY_ENGINEERING unless noted. Component microbenchmarks are labelled. Complete-token claims are labelled.

**KILLS — execution genome, this morning**

1. **N=2 Q4 GEMM as a cut of single-stream weight_addressing** — try12.
   - addr A med 19952248 vs B N=2 med 19843915 (B/A 0.995). Naive 2×GEMV ~1.984×.
   - Full-kernel B/A 1.36× (unpack amortizes) — **not** the 19.95 ms class.
   - tile64 addr med 27127997 (worse).
   - KILLS. REOPEN_IF: a decode that actually has N independent columns in one token, or a kernel whose tile strategy is shown to re-read W (the still-open W@[x1..xN] discriminator).
   - Evidence: `~/.claude-grok/tasks/auto-qwen38-weight-addressing-this-run-try12-20260817-030101/grok-report.md` lines 1–37.

2. **Addressing layout (blocked / Morton / organ-contiguous VA) without codec change** — try13.
   - addr_probe_sum med 19486582 [19401457, 19694248] vs named 21293103.
   - mlp 12651708 (82.2% of mlp full), dn 4554833, gqa 1426624, lm_head 852166.
   - warmed A/B: layout deltas noise or regression (organ-contig gate d=+26625).
   - KILLS. REOPEN_IF: a different codec whose unpack stream is layout-sensitive, or a measured catalog/single-GEMV topology change.
   - Evidence: try13 grok-report lines 1–31; worktree `AUTO_QWEN38_WEIGHT_ADDRESSING_TRY13.json`.

3. **Host/GPU partition of addressing / bind-to-ICB / serial encoder as an addressing win** — try14.
   - addr_probe_sum med 19818414 [19694831, 19826248].
   - mlp host/serial/icb_barrier/icb_conc med 12733416 / 12779458 / 12829000 / 11440583. Concurrent isolated overlap is **not** a production token lever (try14 next_bottleneck note).
   - production GPU host/serial 36282833 / 36277791, delta −5042 ns.
   - host encode isolated med 66083 (not in GPU addressing).
   - KILLS. REOPEN_IF: a production path that actually moves bind off the GPU critical path **and** that shows up in complete-token GPU, not isolated-catalog overlap.
   - Evidence: try14 grok-report lines 1–26; worktree receipt.

4. **Fuse DeltaNet tails (gated_rmsnorm / ba_to_decay / rearrange) into vi or one encoder** — deltanet 09:25.
   - isolated gated_rmsnorm_48 GPU 1316875 / 1373541 / 1330791 (largest tail). ba 136666; rearrange 340583.
   - limiter = 48 underfilled 16-wide launches, **not** host encoder tax.
   - fused_vi: slower (isolated ~8.1 ms vs split ~7.4 ms) **and** greedy ids drifted; `memory_order_device` absent.
   - one_encoder: bit-identical; complete-wall deltas 133–245 µs inside DIRTY pair noise.
   - KILLS fusion-into-vi and encoder-share as a 1.0–1.5 ms complete-token win. REOPEN_IF: one TG/head × 128-thread gated_rmsnorm (still a separate dispatch) — explicitly untested.
   - Evidence: deltanet grok-report lines 1–21; worktree `QWEN38_DELTANET_ACTIVATION_TAILS.json`.

5. **Collapse 129 RMSNorms by persistent/serial encoder or fold-following** — normalization 10:20.
   - isolated all_norms multi-encoder GPU [2281166, 2229999, 2270249] med 2270249 (G024 2367415).
   - serial-encoder [2229833, 2266666, 2241208] med 2241208 — **equal**.
   - one_norm 19999 ns; 2270249/129 = 17.6 µs/launch vs floor 19260 ns; 118× over floor; 1 TG of 256 on hidden=5120.
   - complete-token decode GPU: baseline [37022499, 36877833, 36631374]; fold and persistent flip sign vs baseline; fuse_add_rms deltas [−290333, −272667, −220583] (~0.25 ms) — does not recover 2.27 ms.
   - fuse_add_rms bit-identical, greedy+seal pass, fallbacks=0. Default path unchanged.
   - KILLS encoder-collapse. REOPEN_IF: a mechanism that removes the 1-TG launch tax itself (not another encoder-share). Do not retry GEMV+RMSNorm fusion (prior −10.68 ms).
   - Evidence: normalization grok-report lines 1–47; worktree `QWEN38_NORM_COLLAPSE.json`.

**KILLS — representation, 2026-08-16 (not in the NS register as NS-039)**

6. **Attention below Q4 at the 0.99 hold-cosine bar** — q38-attention-codec-below-q4.
   - L0 out_proj Q3 g64 cosine 0.953; Q3+4% high-RMS 0.967; Q3 g16 0.975; Q4 **0.992**.
   - Q3 residual is **dense**, not a kurtosis tail / minority of heads / 2–4% sidecar.
   - HGRAVP01 degenerates to per-head Q3-or-Q4; 27B pack refused (~1% complete-BPW shave).
   - Sibling: mixed-2p0 MLP + Q8 attention still incoherent. Generate floor is crushed MLP (`down_proj` 0.132 BPW), not attention Q4.
   - Probes are **CPU cosine on real X**, not token-ns. KILLS. REOPEN_IF: a bar change with generate-id evidence, or an MLP that is not r160_b3 0.132 BPW so attention bits can be spent again.
   - Evidence: `~/.claude-grok/tasks/q38-attention-codec-below-q4-20260816-184819/grok-report.md` lines 7–41. Receipt only in that worktree.

7. **Heterogeneous per-tensor Q3-where-it-passes** — genesis-attention-codec.
   - Attention-GEMV policy 4.1879 BPW vs incumbent 4.250 (56.2 MB / 1.46% on attention GEMVs). 84.8% of those elements stay Q4. `out_proj`/`o_proj` never leave Q4 (0/64).
   - Generate coherence **not run**. Complete token **not moved**. ESTIMATE only: complete artifact 4.2527 → 4.236 if only those GEMVs change.
   - STATUS: MEASURED, no generate win. Do not promote 4.1879 as a token result.

8. **shared_r64 generator residual as a 1.125 BPW / net-byte win** — `GENESIS_GENERATOR_RESIDUAL_ADJUDICATION.json`.
   - classification NEGATIVE_SCIENCE, headline REFUTED.
   - explained fraction 0.040492 (energy-weighted). HQ30UQ4 residual == original Q4 bytes (14,287,109,840). Adding generator **increases** storage 726,151,136 bytes. Corrected total BPW 4.466.
   - No artifact, decode hook, GPU reconstruction, or complete-token A/B.
   - KILLS. REOPEN_IF: a residual coder whose store is not HQ30UQ4-sized, with GPU reconstruct + complete-token A/B.

**Standing negatives already in the register (qwen38-touching)**

- NS-004: 535–637 (560–647) reuse band is **not** a decode ceiling. Unique-once in real CB topology is the honest control.
- NS-009: Gaussian / synthetic X for codec ranking. Never as promotion input.
- NS-023: shader compile is not the live token wall.
- NS-038: do not ask "is Qwen3.8 at its wall?" by comparing to Q80 gather, then "correct" onto gather-vs-sequential. Right axis is reuse vs no-reuse.
- Register date 2026-08-16. NS-038 still cites the **refuted** 411.51 / 98.7% pair as "measured". Treat NS-038's *axis* (reuse vs gather) as settled; treat its *ceiling number* as superseded by HONEST_ROOF.

**Standing execution negatives the propose prompt already injects** (do not re-propose):

- Fuse tiny kernels into the following GEMV: +10.68 ms REGRESSION.
- Cross-token cache reuse: hot/cold gap 2.5%.
- N **independent** dispatches against one weight body: still N× (4 GEMVs on lm_head cost 4× by construction).

**STILL OPEN** (prompt + live-act contract): a **single** dispatch `W @ [x1..xN]` that reads W once. try12 killed N=2 as a cut of **single-stream** decode (W already unique-once for N=1). The open item is N>1 columns in one dispatch, with REJECT_IF ≥3× 1×GEMV.

### 4.5 HONEST_ROOF (consume; do not re-derive 411.51)

`HONEST_ROOF_WEIGHT_ADDRESSING.md` label **GPU_PROTECTED_CPU_CONTENDED** (absolute roof provisional):

```
defended bytes:            13,611,663,360
sealed addressing time:    21.293 ms
sealed addressing rate:    639.25 GB/s
single-GEMV addr roof:     699.57 GB/s   (91.4% of that roof)
401-organ catalog addr/full: 530.65 / 505.81 GB/s
unique_once at 13.6 GB:    375.65 GB/s
refuted ceiling:           411.51 GB/s
```

The 97.6% claim mixed (a) 13,618,141,856 geometry-active bytes including norms+embed, (b) 33.91 ms whole production GPU token, (c) the flattering 512 MiB unique_once point. Correct attribution is 13,611,663,360 / 21.293 ms = 639.25 GB/s.

Consequence for this wave: density is **a** lever (unique-once Q4 DRAM is still the existential ns bucket). It is **not** the only lever "because we are at 97.6% of a physical ceiling." Catalog vs single-GEMV (531 vs 700) is remaining dispatch/topology headroom.

---

## 5. Promotion queue and what G1 should treat as landed

PROMOTION_QUEUE schema `hawking.ascent.promotion_queue.v1`, 295 entries, **promoted=false on every row**. Nothing has crossed the promotion gate into CURRENT.

Newest qwen38 MERGE_READY rows (the ones this wave can steal science from, not merge):

```
auto-qwen38-weight-addressing-this-run-try12  SHIPPED  weight_addressing 19952248 ns (mlp 12934375)
auto-qwen38-deltanet                          SHIPPED  gated_rmsnorm_48 1330791; addressing ~21e6 existential
genesis-agentos-hcli-backfill                 SHIPPED  act() capsule collision; not a token lever
auto-qwen38-weight-addressing-try13           SHIPPED  unique-once Q4 DRAM 19486582 (mlp 12651708)
auto-qwen38-weight-addressing-try14           SHIPPED  19818414 unique-once Q4 DRAM, mlp 12733416
auto-qwen38-normalization                     PARTIAL  1-TG launch tax 2270249 isolated
q38-attention-codec-below-q4                  PROVEN_NEGATIVE  38.217 ms / 4.253 BPW; attn<Q4 closed
q38-genome-icb-rebase                         WIN      GPU body 35950374 of 36642-us wall
q38-genome-tokenns                            SHIPPED  addressing 20119736 (53.8% of 37377125)
q38-genome-kv-residency                       SHIPPED  addressing 21293103; four genome levers did not move it
```

ASCENT_STATE `generated` date field is still `"2026-08-16"` even though the file was rewritten today. Do not trust that date.

---

## 6. Representation / kernel findings this wave should consume

Priority order is **measured mechanism**, not G024's 97.6% slogan.

### 6.1 Physical model (what the bytes are)

- Active vehicle: `qwen38-27b/uniform-q4-v1`, manifest sha `d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df`.
- Complete physical BPW **4.252735126866492** (lineage CURRENT + wall receipt). Role: PROFILING_ORACLE_ONLY.
- Defended GEMV payload **13,611,663,360** bytes (HONEST_ROOF). Ledger `active_bytes` 13,618,141,856 includes norms + one embed row that addressing does not stream.
- Attention+embed+norm mass **5.20 GB / 74%** (attention-codec NEXT_BOTTLENECK; not re-weighed here).
- Body resident weight_bytes **14,297,675,776**.

### 6.2 Token wall (which number is which)

| number | what it is | label |
|---|---|---|
| 37,879,375 ns / 26.400 TPS | lineage CURRENT; current-main headline complete wall | DIRTY_ENGINEERING, re-derived at 01:17 without GPU |
| 36,635,958 ns | same receipt, GPU portion | DIRTY |
| 35,227,918 ns / 28.387 TPS | LKG and G024 closed ledger | DIRTY 2026-08-16, different kernel sha |
| 36.5–38.1 ms | this morning's dirty complete-token GPU/wall spreads on try12/13/14/deltanet/norm | DIRTY, production default unchanged |
| 56–63 TPS | `gpu_proxy_verdict` arithmetic from GPU-only | **projection**, not a token |

G1 target 10,000,000 ns / 100 TPS is **~3.8×** below the CURRENT complete wall. No G0 mechanism this morning moved the production default.

### 6.3 Existential bucket

weight_addressing, unique-once Q4 DRAM of `geo_tpr64_tg128`.

- G024: 21,293,103 ns, 60.44% of 35.228 ms wall. addr_probe 83–92% of every GEMV class.
- try13 dirty addr_probe_sum **19,486,582** (mlp **12,651,708** = 82.2% of mlp full).
- try14 dirty addr_probe_sum **19,818,414** (mlp **12,733,416**).
- Class split try13: mlp 12.65 ms, dn 4.55 ms, gqa 1.43 ms, lm_head 0.85 ms.

This morning's execution attacks (layout, host/GPU bind, N=2 GEMM, encoder-share) **did not cut this class**. The remaining honest levers named by the organism:

1. **Fewer unique-once bytes** (codec / allocation), consumed by a representation-specific kernel — not expand-to-Q4-then-generic-GEMV.
2. **Catalog vs single-GEMV topology** (531 vs 700 GB/s) — dispatch/kernel topology, still headroom, not a bytes/quoted-bandwidth floor.
3. **Still-open W@[x1..xN] single dispatch** — only if the workload actually has N columns. N=2 did not cut N=1 decode.

### 6.4 Representation doors that are closed

- Uniform or per-head attention **below Q4 at 0.99 cosine**: closed. out_proj/o_proj never leave Q4.
- r160_b3 down_proj at 0.132 BPW: not a coherent generate path. Any low-BPW MLP must beat this with generate-id evidence.
- shared_r64 + HQ30UQ4 residual: net **worse** bytes; explained 4.05%.
- 4.1879 attention-GEMV BPW heterogeneous pack: cosine-only, generate unrun, ESTIMATE complete 4.236. Not a G1 candidate by itself.
- Per-layer / per-head assignment (synthesis catalog): **launched** as `auto-qwen38-weight-addressing` 09:02; that task dir has **no grok-report** (only metadata/status). Result unknown. Do not treat as measured.

### 6.5 Kernel doors that are closed on this genome

- Fuse tiny kernels into the following GEMV (+10.68 ms).
- Fuse DeltaNet tails into vi (wrong + slower).
- Persistent/serial encoder collapse of 129 RMSNorms (isolated GPU unchanged).
- GEMV+RMSNorm fusion (prior kill; normalization reconfirmed do-not-retry).
- Addressing layout remap (Morton/blocked/organ-contig).
- Bind/address host-GPU partition / ICB as an addressing win.
- N=2 GEMM as a single-stream addressing cut.
- N independent GEMVs as DRAM amortization.
- Cross-token cache reuse (2.5%).

### 6.6 Kernel doors still open (small vs existential)

- RMSNorm **1-TG launch tax** ~2.27 ms isolated (129 × ~18 µs). Not encoder-share. Not GEMV fusion. Research-class, not existential.
- gated_rmsnorm_48 **1.33 ms** isolated: one TG/head × 128-thread reduction, still a separate dispatch. Untested.
- Catalog/single-GEMV topology (24% below single-GEMV addr roof).
- W@[x1..xN] single dispatch when N>1 is real.
- A **new MLP representation** that is not r160_b3 0.132, with a representation-specific Metal kernel, generate-id gate, complete-token A/B. This is the only door that can move both BPW and the 19.5 ms addressing bucket.

### 6.7 Genome (do not silently change)

```
Qwen38HybridDecodeSession
+ qwen_uniform_q4_group64_matvec_geo_tpr64_tg128
+ qwen38_gated_delta_decode_vi
deltanet_vi_parallel=true
concurrent_independent=false
1 CB / 964 dispatches
Say hi. greedy 16:
[248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149, 1061, 369, 264, 1546]
fallbacks=0
```

This morning's lanes left that default in place (opt-in arms only).

---

## 7. What this wave should not do

- Do not treat `hold: queue dry` as an empty research program.
- Do not treat CURRENT.live / lineage pid 40316 as the live body.
- Do not treat G024 35.228 ms or 97.6%/411.51 as the G0 baseline. CURRENT is 37.879 ms DIRTY; 411.51 is refuted.
- Do not treat gpu_proxy 56–63 TPS as measured.
- Do not resurrect Q80/DSV4F as vehicles. Steal only the negative-science *shape* (unique-once vs reuse; component ≠ token).
- Do not launch another weight_addressing layout / bind-partition / N=2 / encoder-collapse / GEMV-fusion lane.
- Do not send health/propose RPCs that contend with the live parent generate.
- Do not assume this morning's receipts are under `receipts/ascent-2026-08-16/` on main — they are in the lane worktrees.

---

## 8. Cheapest unanswered questions

1. Health JSON of pid 74869 after propose 50752 exits (one 0.75 s RPC).
2. Did propose 26266 emit a named mechanism or an empty/truncated think? (daemon has not written a new ASCENT_STATE tick yet.)
3. What happened to `auto-qwen38-weight-addressing` 09:02 (per-layer/per-head assignment)? No grok-report.
4. Land or copy this morning's five receipts onto the canonical tree so the next harvest is not worktree-only.
5. Independent complete-token remeasure is owned by another lane; this harvest did not take the GPU.

---

# Completion report

```
STATUS
INCONCLUSIVE

CLAIMS
C1. G0 body+daemon+forever are live; artifact is qwen38-27b/uniform-q4-v1; a parent propose is in flight. EVIDENCE: §1.1 pgrep/os.kill; resident.log:284-286; sock exists.
C2. Health RPC timed out because the body is busy generating, not because it is dead. EVIDENCE: TimeoutError 0.75s; propose PIDs 26266 then 50752; daemon.log:221-225 health_busy_process_alive.
C3. Scheduler launch pool is empty (pending=0, running=0) while PROMOTION_QUEUE has 295 entries / 132 merge-ready / 0 promoted. The "queue dry" hold is the pending-list predicate, not the queue. EVIDENCE: daemon.log:220; ASCENT_STATE status Counter §1.5; PROMOTION_QUEUE promoted Counter {False:295}; ascent_daemon.py:1320-1323.
C4. Lineage CURRENT is G0, CANDIDATE empty, research_state null, seated pid 40316 ≠ live 74869. LKG still carries G024 35.228 ms; CURRENT carries current-main 37.879 ms. EVIDENCE: GENESIS_LINEAGE_CURRENT.json lines 61-141.
C5. Current declared NEXT_BOTTLENECK of the in-flight propose is the 964-dispatch GPU body 35.950 ms / 35,950,374 ns. Previous propose (now dead) was the attention-codec 38.217 ms / 4.253 BPW string. Most recently measured next cost from a finished lane is RMSNorm 1-TG tax 2,270,249 ns isolated; existential bucket remains unique-once Q4 DRAM ~19.5-19.8 ms dirty. EVIDENCE: §2 pgrep text; grok-reports try13/try14/normalization; G024 ranked_by_ns.
C6. This morning G0 produced five measured negatives (N=2 GEMM, layout remap, host/GPU bind, DeltaNet-tail fusion, RMSNorm encoder-collapse) and did not move the production default token. Receipts are in worktrees, not on main receipts/. EVIDENCE: §4.3-4.4.
C7. Attention below Q4 at 0.99 is closed; r160_b3 0.132 MLP is not a coherent generate path; shared_r64 net-byte win is REFUTED; 411.51/97.6% roof is REFUTED. EVIDENCE: attention-codec grok-report; GENESIS_GENERATOR_RESIDUAL_ADJUDICATION.json; HONEST_ROOF_WEIGHT_ADDRESSING.md.
C8. Historical G0 claim 4.2527 BPW / 26.4 TPS / ~37.9e6 ns matches lineage CURRENT / current-main wall and is DIRTY_ENGINEERING, not independently remeasured by this lane. EVIDENCE: wall receipt authority.headline_*; verification claim_boundary.gpu_work_launched=false.

EVIDENCE
- pgrep/os.kill/health RPC/lock/file ages: this harvest's shell transcripts, §1.
- /Users/scammermike/Downloads/hawking/workspace/ops/ascent-daemon.log lines 1-225 (last full tick line 220).
- /Users/scammermike/Downloads/hawking/workspace/ops/genesis-resident.log lines 266-287.
- /Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json
- /Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/PROMOTION_QUEUE.json (295 entries)
- /Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/ASCENT_STATE.json (220 targets)
- /Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json (38 entries, date 2026-08-16)
- /Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.json authority.headline_*
- /Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json lines 46-80
- /Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/TOKEN_NS_QWEN38.json weight_addressing 21293102.52 ns
- /Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.md lines 15-45
- /Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/GENESIS_GENERATOR_RESIDUAL_ADJUDICATION.json
- /Users/scammermike/Downloads/hawking/tools/ascent_daemon.py harvest/generate_targets/queue-dry/_genesis_prompt
- grok-reports under ~/.claude-grok/tasks/auto-qwen38-{normalization,weight-addressing-try13,weight-addressing-try14,deltanet,weight-addressing-this-run-try12}-* and q38-attention-codec-below-q4-*
- worktree receipts listed in §4.3

CHANGES
Created workspace/superwave/g1/g1-resident-harvest.md only.

TESTS
$ test -s workspace/superwave/g1/g1-resident-harvest.md && echo 'test -s: PASS'
test -s: PASS

$ wc -l workspace/superwave/g1/g1-resident-harvest.md
     641 workspace/superwave/g1/g1-resident-harvest.md

$ git status --porcelain
?? workspace/superwave/g1/g1-resident-harvest.md

RISKS
- Live propose holds the GPU lock; this wave's GPU-authority lane will wait or dirty-contend.
- Daemon tick is currently blocked up to 1800s inside genesis_proposes; "last tick" ages quickly.
- ASCENT_STATE.generated date field is stale ("2026-08-16").
- G024 97.6%/411.51 still sits in G024_QWEN38_TOKEN_NS.json and NS-038; easy to re-cite.
- This-morning receipts are worktree-only; a sparse checkout of main will look empty.

UNRESOLVED
- Health JSON of the live body (RPC not taken during propose).
- Emit of propose 26266 (empty vs named) — no new ASCENT_STATE tick yet.
- Outcome of auto-qwen38-weight-addressing 09:02 (no grok-report).
- Independent complete-token remeasure (other lane).
- Whether generate_targets will admit any new pending target after the current propose, or refuse it as a try12 duplicate.

NEXT
G1 should attack unique-once bytes with a representation-specific kernel (MLP that is not r160_b3 0.132, generate-id gated), and/or catalog-vs-single-GEMV topology, and/or the still-open W@[x1..xN] discriminator if N>1 is real. Do not replay this morning's killed execution mechanisms. Do not treat queue-dry or CURRENT.live as health.
```
