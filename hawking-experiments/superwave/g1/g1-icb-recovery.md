# G1 ICB recovery — 659,766 ns named-fixed, commit `7400acf1b`

Lane: `73-icb-recovery`. HEAD `bf0b4dc02`. No GPU run this lane.
Every number is tagged MEASURED / PROJECTED / ESTIMATED / GIVEN / DERIVED.
A component microbench is not a token-level claim. A cross-session
subtraction is not a paired A/B.

---

## 0. Verdict

`7400acf1b` is a real, already-paid-for named-fixed deletion of
**659,766 ns DERIVED** on the Q4 production graph. It is **not** an
ancestor of HEAD. The Q4 `step` path at HEAD is still 964 dispatches /
964 encoders / 1 `wait_until_completed` / no ICB symbol. The Metal
replay substrate **is** on HEAD and is **not** called from
`Qwen38HybridDecodeSession::step`.

The 659,766 ns is a **token-level named-component sum**, not isolated
kernel time, and **not** the complete-token wall delta (that wall
moved 1.533 ms and includes dirty GPU). The before-side is the
complete-wall authority receipt, not a same-process CPU-encode
control. Encode 886 µs → 91 µs is too large to be session noise.

Ceremony 1,652,510 ns (G024 organ regrouping, rounded to 1.653 ms in
the sealed ledger) **cannot give this 659,766 twice**. ICB deletes
from the encode/wait/submit members of that class. After ICB, remaining
named-fixed is 670,934 ns. The 336,926 ns intra-CB GPU gap is **not**
inside the 659,766.

Port is IMPLEMENT_READY as a surgical splice, not a `git cherry-pick`.
`git merge-tree` against HEAD: 3 files changed-in-both, 5 conflict
hunks, all “keep both”. ESTIMATED most of the 0.66 ms named-fixed
survives on the G0 Q4 path. 0 of it applies to mixed-sub15 until a
second graph is written. Do not spend the 1.53 ms wall delta.

---

## 1. Ancestry (MEASURED by git)

```
HEAD     bf0b4dc0250041a8ff9237065c89404eff48506b
ICB      7400acf1b46e3f2ebf4b6bde2a1caec8bd4a6f1c
PARENT   9c87c500dc4c1e4b96962a99a1b017a23f273906
```

| probe | result | evidence |
|---|---|---|
| `git cat-file -t 7400acf1b` | `commit` | this lane |
| `git merge-base --is-ancestor 7400acf1b HEAD` | exit 1, NOT_ANCESTOR | this lane |
| `git merge-base --is-ancestor 9c87c500 HEAD` | exit 0, PARENT_IS_ANCESTOR | this lane |
| `git merge-base HEAD 7400acf1b` | `9c87c500dc4c1e4b96962a99a1b017a23f273906` | this lane |
| `git rev-list --count 9c87c500..HEAD` | **86** | this lane |
| `git rev-list --count HEAD..7400acf1b` | **1** | this lane |
| branch containing ICB | `grok/qwen38-kill-fixed-overhead-20260816-165048` | `git branch -a --contains` |

ICB commit message: `qwen38-kill-fixed-overhead: ICB replay, encode 886us -> 91us`
(2026-08-16 17:28:59 -0400). Parent message: `Measure the backlog: 139 of 161 unmerged branches are stale`.

Receipt `commit` field is the **parent**, not `7400acf1b`. DIRTY_ENGINEERING:
the run was on a dirty tree based on `9c87c500`, then committed.

---

## 2. What `7400acf1b` changed

`git diff --stat 7400acf1b^..7400acf1b`:

```
 crates/hawking-core/examples/ascension_qwen38_hybrid_greedy.rs |  77 ++
 crates/hawking-core/src/metal/mod.rs                           |  60 +-
 crates/hawking-core/src/model/qwen38_hybrid_decode.rs          | 949 ++++++++++++++++++++-
 3 files changed, 1058 insertions(+), 28 deletions(-)
```

Receipts were written on the dirty tree and are on HEAD via other
preserves (`files_touched` in `QWEN38_FIXED_OVERHEAD_DELETED.json`).
They are **not** in the commit tree of `7400acf1b` itself.

### 2.1 `qwen38_hybrid_decode.rs` (the payload)

Inserted after `mod device` opens (`7400acf1b` `:131–940`):

| symbol | role |
|---|---|
| `write_u32` / `write_f32` | host poke into the scalar slab |
| `Qwen38Schedule` {`IndirectCommandBuffer`, `SerialGroup`, `MultiEncoder`} | `HAWKING_QWEN38_SCHEDULE` / `HAWKING_QWEN38_ICB=0` |
| `ScalarTable` 64 KiB, 256 B align | intern static `set_bytes` payloads once |
| `Qwen38TokenReplay` | ICB + 3 mutable slot offsets |
| `ReplayWorkspace` | cloned `PinnedBuffer` handles (Q4 workspace only) |
| `ReplayGraphBuilder` | frozen 964-stage graph |

`ReplayGraphBuilder` is a second encoder of the **same** Q4 graph
`encode_embed` / `encode_deltanet` / `encode_gqa` / `encode_dense_mlp`
/ `encode_terminal` already emit:

- GEMV: `session.matvec_kernel.as_str()` + `launch(rows)` (default
  `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`)
- embed: `qwen_uniform_q4_embedding_lookup`
- rearrange: `qwen38_qkvz_rearrange_conv_l2_f32`
- decay: `qwen80_ba_to_decay_beta_f32`
- delta: `qwen38_gated_delta_decode_vi` if `deltanet_vi_parallel` else
  `qwen80_gated_delta_decode_tg`
- gated rms: `qwen80_deltanet_gated_rmsnorm_f32`
- GQA rope: `qwen38_gqa_qk_norm_rope_cache_f32`
- MHA: `mha_decode_f32` with **fixed** shmem `(max_seq_len+128)*4` and
  packed arg u32s in the scalar slab (`seq_len` slot written per token)
- gate: `qwen38_attention_apply_sigmoid_gate`
- silu: `qwen80_silu_mul_f32`
- residual: `qwen_next_add_residual`
- sample: `sample_argmax_f32`
- rmsnorm: `qwen80_residual_rmsnorm_f32`

Barriers on producer→consumer. Omitted on independent pairs
(gate/up, qkvz/ba, k/v after q) — same pairs HEAD can optionally
overlap with `concurrent_independent` (default off).

`finish()` **refuses** unless `stages.len() == 964`
(`QWEN38_EXPECTED_DISPATCHES`). Fail-closed: `ensure_replay` prints
and falls back to `SerialGroup` (does **not** abort generate).

`step` / `step_complete` (`7400acf1b` `:2543–2626`):

```
ensure_replay()?;
if replay.is_some() && schedule == IndirectCommandBuffer {
    encode_token_icb  // write 3 u32s; tcb.execute_replayable_graph
} else {
    encode_token_cpu  // encode_embed + encode_layers + encode_terminal
}
tcb.commit_and_wait_timed()
```

`encode_token_icb` writes `token`, `position`, `mha_seq_len=position+1`
into the slab, then one `execute_replayable_graph`.

Default `Qwen38Schedule::from_env` is **ICB on**. Disable:
`HAWKING_QWEN38_ICB=0` or `HAWKING_QWEN38_SCHEDULE=serial|multi`.

MHA shmem cap: `(max_seq_len+128)*4 ≤ 32 KiB` ⇒ `max_seq_len ≤ 8064`.
Greedy default `--max-seq-len` is 128 (`ascension_qwen38_hybrid_greedy.rs:68`).
Genesis long-ctx above 8064 would fail ICB build and fall back.

### 2.2 `metal/mod.rs` (load-bearing for the 91 µs)

`ReplayableComputeGraph` already existed on the parent. ICB only
changed `execute_replayable_graph_group`: when `graphs.len()==1`,
reuse the encode-time residency list instead of allocating a
`HashMap` and cloning every buffer **per token**. Multi-graph
barriers skipped when `len < 2`.

HEAD still has the HashMap-per-token path
(`crates/hawking-core/src/metal/mod.rs:4696–4718`). The substrate
comment at `:3570–3574` still says it is “intentionally not wired
into decode selection yet.”

Without this 20-line change, ICB encode would still beat 964
encoder creates, but would **not** be the measured 91 µs.

### 2.3 greedy example

Prints `SCHEDULE`, `ICB_COMMANDS`, `ICB_ERROR`, and aggregates the
5 named-fixed means across the 6 warm reps into `NAMED_FIXED_SUM_*`.
That aggregation **is** how `after_mean_ns=670934` was produced.

---

## 3. The 659,766 ns — what it is, and what it is not

Receipt: `receipts/ascent-2026-08-16/QWEN38_FIXED_OVERHEAD_DELETED.json`
(on HEAD; produced against parent `9c87c500`). Full wall:
`QWEN38_FIXED_OVERHEAD_ICB_WALL.json`. Baseline:
`QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json`.

### 3.1 Arithmetic (DERIVED, this lane)

Named-fixed members, receipt `named_fixed_components_ns`:

| member | before_mean | after_mean | Δ |
|---|---:|---:|---:|
| encode_host_prepare | 886,200 | 90,981 | −795,219 |
| wait_minus_gpu | 425,900 | 561,994 | **+136,094** |
| submit | 10,500 | 9,420 | −1,080 |
| tokenizer | 6,300 | 6,831 | +531 |
| epilogue | 1,800 | 1,708 | −92 |
| **sum** | **1,330,700** | **670,934** | **−659,766** |

```
886200+425900+10500+6300+1800 = 1_330_700
90981+561994+9420+6831+1708   =   670_934
1_330_700 - 670_934           =   659_766
```

The receipt never stores `659766`. It stores `before_authority_mean_ms=1.3307`
and `after_mean_ns=670934`. Byte-deletion lane computed the difference
(`g1-byte-deletions.md:524–525`). Fusion-persistent rounded the same
delta to −660,000 (`g1-fusion-persistent.md:410`).

### 3.2 Token-level, not a component microbench (MEASURED)

- Vehicle: `uniform-q4-v1`, complete physical BPW **4.252735126866492**
  (same artifact as G0).
- Binary: `ascension_qwen38_hybrid_greedy` release `lto=fat`.
- Regime: 1 discarded cold generate; 3 alternating A/B pairs = 6
  in-process generates; headline = median of 6 per-rep medians of
  **decode** steps; 31 steady decode steps/rep; prefill excluded.
- After headline wall: 36,683,916 ns, TPS 27.2599, GPU 36,012,250 ns.
- After encode spread across 6 reps: min 85,012 / median 89,980 / max 96,317.
- Coherence: `tools/coherence_gate.py verify` PASS, 3 prompts
  id-identical to `QWEN38_COHERENCE_SEAL.json`, 0 fallbacks.
- GPU authority: `GPUEndTime−GPUStartTime` after wait.

These are host Instants around encode / wait−gpu / submit / tokenizer
/ epilogue **inside** the complete-token generate. Not
`measure_isolated_*`. Not a 15-command ICB fusion microbench
(`replayable_icb_fifteen_command_fusion_encode_benchmark`).

### 3.3 It is not the complete TOKEN_NS delta

| quantity | before | after | Δ | tag |
|---|---:|---:|---:|---|
| named-fixed sum | 1,330,700 ns | 670,934 ns | **−659,766 ns** | MEASURED cross-session components |
| headline complete wall | 38.216792 ms | 36.683916 ms | −1.533 ms | DIRTY (GPU moved) |
| headline GPU | 36.987458 ms | 36.01225 ms | −0.975 ms | DIRTY, unclaimed |
| wall−gpu | 1.229334 ms | 0.671666 ms | −0.558 ms | close to named-fixed after |

The receipt itself says wait−gpu **rose** 136 µs (“ENCODE FELL AND WAIT
ROSE … Reported, not hidden”). Fusion-persistent: “Wall Δ includes dirty
GPU movement … is **not** a clean 1.53 ms ceremony claim.”

Do not spend 1.533 ms as the ICB TOKEN_NS recovery.

### 3.4 Cross-session, not paired A/B

`before_*` copies `QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json`
(`encode_host_prepare` 0.8862 ms, `wait_minus_gpu` 0.4259 ms,
headline 38.216792 ms). The ICB wall JSON is schedule=`icb` only.
There is no same-process CPU-encode control in that receipt.

Label: **token-level named-component subtraction across two DIRTY
sessions on the same vehicle**. Encode 886→91 is a 10× drop against
an after-spread of ±6 µs; that piece is robust. Wait−gpu +136 µs is
the ICB-attributed tail; it could also contain session movement.
Net 659,766 is dominated by encode.

---

## 4. Genome vs HEAD

### 4.1 Same

- Artifact `uniform-q4-v1`, BPW 4.252735126866492.
- Production kernel `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`.
- `deltanet_vi_parallel=true`, `concurrent_independent=false`.
- 964 dispatches, 1 CB. Ledger
  `QWEN38_TOKEN_NS_LEDGER.json` `dispatches.total=964`,
  `production_command_buffers=1`.
- `step` still ends in `commit_and_wait_timed` →
  `cmd.wait_until_completed()` (`metal/mod.rs:5371`).
- Parent `9c87c500` **is** an ancestor of HEAD. The Q4 encode bodies
  (`encode_embed` / `encode_deltanet` / `encode_gqa` / `encode_dense_mlp`
  / `encode_terminal`) still emit the same kernels and grids; HEAD
  only added `if !self.weights.mixed.is_empty() { mixed }` early
  returns. G0 mixed is empty.

### 4.2 Not the same

| axis | ICB measurement | HEAD now |
|---|---|---|
| source commit | dirty `9c87c500` + ICB patch | `bf0b4dc02` = parent + **86** commits |
| decode file | 3,027 lines | 3,905 lines (`git show \| wc -l`) |
| weights | session-owned `HashMap<String,Q4Weight>` | `Arc<Qwen38HybridWeights>` + `attach` / share |
| extra workspace | none | `hgravs_mid`, `split_qkv`, `split_b`, `split_a` |
| extra paths | none | `encode_*_mixed`, `dispatch_{binary,residual,hgravs,uniform}`, fanout |
| ICB in `step` | default on | **absent** (`rg` on HEAD decode file: no match) |
| metal single-graph residency | ICB patch | **absent** (HashMap every replay) |
| complete-token wall | authority 38.217 ms / ICB 36.684 ms | GIVEN live G0 median 39,326,090 ns (6 paired reps, 1.83% spread) |
| TOKEN_NS ledger | G024 35,227,917 ns (commit `57ee82cce`, 61 commits **before** ICB parent) | not re-measured this lane |

Three walls on the same vehicle, three sessions:

| source | ns/token | what it is |
|---|---:|---|
| G024 / `TOKEN_NS_QWEN38` | 35,227,917 | 12-component GPU-closed ledger |
| complete-wall authority | 38,216,792 | complete token, ICB **before** |
| ICB after | 36,683,916 | complete token, ICB **after** |
| GIVEN live G0 today | 39,326,090 | seated organism, this campaign |

`57ee82cce` is an ancestor of `9c87c500` (61 commits). HEAD is 86
further. Kernel **family** matches. Execution **session** does not.
A roof is conditioned on the current execution genome: the 659,766
is conditioned on the authority+ICB pair, not on today’s 39.33 ms
organism.

---

## 5. Ceremony 1,653,000 ns — no double count

G024 `ranked_by_ns` + token-anatomy regrouping
(`g1-token-anatomy.md:198–202, 302`; G024 `closure.named_residual`):

| G024 row | ns | in ICB named-fixed? |
|---|---:|---|
| host_preparation (encode) | 919,250 | yes (authority 886,200) |
| synchronization (wait−gpu) | 384,250 | yes (authority 425,900) |
| command_submission | 12,084 | yes (authority 10,500) |
| intra-CB encoder-transition gap | 336,926 | **no** (GPU-side; sits in `unattributed_residual` 341,925 with ~5 µs embed) |
| tokenizer / epilogue | — | in named-fixed (6,300 + 1,800); **not** in G024 1.653 ms |
| **anatomy ceremony** | **1,652,510** | sealed ledger “1,653,000 ns / 4.69%” |

Overlap of the two **classes** is encode + wait−gpu + submit =
1,315,584 ns (G024) vs 1,322,600 ns (authority, before tokenizer/
epilogue). ICB’s 659,766 is almost entirely encode − wait-rise
inside that overlap.

What “cannot give up 659,766 twice” means, precisely:

1. **Do not** subtract 659,766 from the 1,653,000 **and** bank 659,766
   as a second TOKEN_NS recovery outside ceremony. It **is** a ceremony
   deletion.
2. **Do not** arithmetically do `1_652_510 - 659_766` and call the
   remainder MEASURED. G024 encode 919,250 ≠ authority encode 886,200
   (Δ 33,050 ns, different session, 61 commits apart).
3. **Do not** add the 336,926 intra-CB gap onto the 659,766. ICB
   *may* collapse encoder transitions (964 encoders → 1
   `executeCommandsInBuffer`); the receipt did not claim that GPU
   piece. The 0.975 ms GPU drop is dirty and includes whatever
   that collapse is worth plus session movement.
4. After ICB, remaining **named-fixed** is 670,934 ns MEASURED (ICB
   session). Remaining G024-style ceremony ESTIMATED ≈ 0.67 ms
   named-fixed + leftover intra-CB (unknown, possibly smaller).
   The 1.653 ms figure is pre-ICB G024, not a second well.

Tokenizer +531 and epilogue −92 are noise. They are not in the
1.653 ms bucket.

Room check: 659,766 < G024 encode alone (919,250). Ceremony **can**
give this once. It cannot give it a second time.

---

## 6. Current HEAD step path (still the ICB target)

`Qwen38HybridDecodeSession::step` (`qwen38_hybrid_decode.rs:3292–3312`):

```
encode_embed; encode_layers; encode_terminal; commit_and_wait_timed
```

`encode_layers` (`:2822–2829`): `for layer in 0..64` mixer + dense MLP.
Q4 early-return: mixed only if `!weights.mixed.is_empty()`.

`commit_and_wait_timed` → `commit_and_wait_split` → `cmd.commit()` +
`cmd.wait_until_completed()` (`metal/mod.rs:5371`). One wait.

`mha_decode_f32_tcb` (`kernels/mod.rs:10506–10562`) still allocates a
**new `KernelArgBuffer` every token** and sizes shmem as
`(seq_len+128)*4` (grows with position). ICB replaces both with the
interned slab + oversized fixed shmem. That is why the graph is
replayable.

HEAD decode ICB-symbol scan (`rg -i 'icb|IndirectCommand|Qwen38Schedule|ReplayGraph|HAWKING_QWEN38_ICB'`
on `HEAD:crates/hawking-core/src/model/qwen38_hybrid_decode.rs`): **no
matches**. Contract claim “no ICB symbol present in the hybrid decode
path” is SUPPORTED.

`ReplayableComputeGraph` / `executeCommandsInBuffer` exist in
`metal/mod.rs` and are used by GLM opt-in paths and ignored unit
tests, not by Qwen3.8 `step`.

---

## 7. Conflicts at HEAD (`git merge-tree` dry)

`git merge-tree $(git merge-base HEAD 7400acf1b) HEAD 7400acf1b`
→ 3 `changed in both`, 5 `<<<<<<<` hunks. `--write-tree` is blocked
in this sandbox (`sparse-checkout.lock` / temp file); the textual
merge-tree is enough.

### C1 — `ascension_qwen38_hybrid_greedy.rs` (1 hunk)

HEAD prints `DENSE_W_MATERIALIZED: 0`. ICB prints `SCHEDULE` /
`ICB_COMMANDS` / `ICB_ERROR`. **Keep both.**

ICB’s `named_across_warm_reps` block around the complete-wall
summary auto-merges (added-only). Confirm `mean_u64` / `spread_u64`
still exist at the insertion point before compiling.

### C2 — decode imports

HEAD: `TokenCommandBuffer` + `use std::thread`.
ICB: add `ReplayBufferBinding`, `ReplayComputeStage`,
`ReplayableComputeGraph` + `use std::env`.
**Keep both.**

### C3 — `encode_mixer_gemvs_only_mixed` vs `schedule_name`/`encode_token_*`

Same insertion site, different functions. **Keep HEAD mixed helper.
Add ICB methods after it.** Do not replace mixed diagnostics.

### C4 — `encode_mlp_matvecs_only_mixed` vs `build_token_replay`

Same. **Keep HEAD. Add `build_token_replay` next to `ensure_replay`.**

### C5 — `pub use device`

HEAD exports `generate_greedy_parallel`, `measure_shared_weight_fanout`,
`Qwen38HybridWeights`, `Qwen38WeightFanout`.
ICB exports `Qwen38Schedule`.
**Keep HEAD list; append `Qwen38Schedule`.**

`metal/mod.rs` is `changed in both` with **0** conflict markers: the
single-graph residency patch applies cleanly onto HEAD’s
`execute_replayable_graph_group`. Still required.

The 817-line `ReplayGraphBuilder` block (ICB insert at start of
`mod device`) is an add vs HEAD’s mixed-type add at the same region.
merge-tree auto-places it; verify it lands **inside** `mod device`
after imports, before `Q4Weight`, and that `ReplayWorkspace::clone_from`
still names only fields that exist (they do; HEAD’s 4 extra buffers
are unused by the Q4 graph).

`PinnedBuffer` on macOS is `pub use metal::Buffer as PinnedBuffer`
(`metal/mod.rs:979`). `ReplayBufferBinding::{read,write,read_write}`
take `&Buffer`. Compiles as-is.

`session.q4` / `session.f32` at HEAD go through `self.weights.q4` /
`self.weights.f32s` (`:1182–1195`). `ReplayGraphBuilder` calls
`self.session.q4(name)` — still valid.

---

## 8. Port plan (surgical; do not cherry-pick)

Estimated **~1,050 lines added**, 0 shader changes, 0 new kernel
family. Bound: 250–500 was the mixed-native estimate in Context;
ICB is larger because it duplicates the encode graph.

### P1 — `crates/hawking-core/src/metal/mod.rs`

Function: `TokenCommandBuffer::execute_replayable_graph_group`
(`:4627`). Apply the `graphs.len()==1` residency reuse from
`7400acf1b`. ~20 lines. No API change.

### P2 — `crates/hawking-core/src/model/qwen38_hybrid_decode.rs`

Copy from `7400acf1b`, do not take ICB’s `open()` (HEAD `attach` +
`Qwen38HybridWeights` wins).

1. Imports (C2).
2. `write_u32`/`write_f32`, `Qwen38Schedule`, `ScalarTable`,
   `Qwen38TokenReplay`, `ReplayWorkspace`, `ReplayGraphBuilder`
   (`7400acf1b` `:131–940`).
3. Session fields: `schedule`, `replay`, `replay_error`. Init in
   `attach()` via `Qwen38Schedule::from_env()`.
4. Methods: `schedule_name`, `replay_command_count`, `replay_error`,
   `encode_token_cpu`, `encode_token_icb`, `ensure_replay`,
   `build_token_replay` (C3/C4: **add**, do not replace mixed helpers).
5. `step` / `step_complete`: `ensure_replay` + ICB/CPU branch. Keep
   HEAD’s fallbacks check. ICB’s `position >= max_seq_len` guard is
   already on HEAD `encode_gqa`; lift to `step` as ICB did, or leave
   — either is fine.
6. `pub use`: append `Qwen38Schedule` (C5).

`build_token_replay` **must refuse** when `!self.weights.mixed.is_empty()`.
The ICB graph is `push_q4` only. mixed-sub15 is a **new** graph
(Binary / Residual / Hgravs / Uniform), not this port.

`finish()` stays strict: 964 stages or fail-closed to `SerialGroup`.
Do **not** silently switch G0 from 964-encoder to `begin_serial_group`
if ICB build fails: that is a different, **unmeasured** topology.
If ICB fails, stay on `MultiEncoder` (HEAD default) and surface
`replay_error`. This is a port correction vs `7400acf1b` (which
fell back to `SerialGroup`).

Default-on: `7400acf1b` measured ICB as default. Resident G0 is live
and this lane must not flip it. Land with the same env surface;
**do not enable on the resident process** until a serialized GPU
lane coherence-gates. Recommend shipping default-on only after that
gate, or default-off (`HAWKING_QWEN38_ICB=1` to enable) if the
organism must keep bit-identical encode until gated. GLM later
rejected default-on ICB (`Reject default-on ICB replay at token gate`);
that kill is a different model (dynamic MoE). It does not transfer
as a Qwen3.8 Q4 kill. It **does** argue for an env kill switch,
which already exists.

### P3 — `crates/hawking-core/examples/ascension_qwen38_hybrid_greedy.rs`

Keep `DENSE_W_MATERIALIZED`. Add ICB prints + named-fixed aggregation
from `7400acf1b`. ~70 lines.

### P4 — do not touch

- `qwen38_device_activations.metal` / `gk_family.metal` (receipt
  `not_touched`; no rewrite required).
- mixed encode path, catalog, codec-4 (other lane).
- receipts (already on HEAD).
- resident Genesis process.

### P5 — GPU lane after this file (not this lane)

1. Compile `ascension_qwen38_hybrid_greedy`.
2. Coherence-gate vs `QWEN38_COHERENCE_SEAL.json` with ICB on,
   uniform-q4-v1. Expect 0 fallbacks, 964 `ICB_COMMANDS`, no
   `ICB_ERROR`.
3. One complete-wall pair (ICB vs HEAD multi-encoder) **on the same
   process** if a serialized GPU lane wants to replace the cross-
   session 659,766 with a paired number. Do **not** re-discover ICB
   vs encode as if new (`g1-byte-deletions.md:635`;
   `g1-fusion-persistent.md:441`).
4. Expected if the Q4 graph still matches: encode ~91 µs, named-fixed
   ~0.67 ms, wait−gpu up ~136 µs. ESTIMATED.

---

## 9. How much of 659,766 survives at HEAD

| path | survives | tag |
|---|---|---|
| G0 Q4 `step` (mixed empty, 964 GEMV/Q4 graph) | **most of it: ESTIMATED 0.60–0.70 ms named-fixed** | mechanism still present; encode still 0.886–0.919 ms class (authority / G024); ICB still deletes that class |
| complete-token wall 1.533 ms | **do not spend** | dirty GPU |
| intra-CB 336,926 ns | **unclaimed** | possible extra if encoder collapse is real; not in 659,766 |
| mixed-sub15 native | **0** | `ReplayGraphBuilder` is `push_q4` only; HEAD mixed uses Binary/Residual/Hgravs/Uniform + `qwen38_hgravu_embedding_lookup` |
| ceremony 1.653 ms as a second well | **0 additional** | ICB **is** that well’s encode line |

Not re-measured on HEAD (no GPU). Live G0 39.33 ms vs authority
38.22 ms is session + 86 commits, not proof encode vanished. G024
encode 919,250 ns still exists as a measured host_preparation row
on an ancestor genome whose Q4 `step` shape HEAD still has.

Lower-bound sanity (ESTIMATED, not a claim): if HEAD encode is still
G024’s 919 µs and ICB encode is still 91 µs, encode deletion is
~828 µs. Net named-fixed then depends on the wait tail. Authority
wait 426 µs + ICB’s +136 µs ≈ 562 µs after, net ~660 µs. If HEAD
wait is G024’s 384 µs and ICB wait is 562 µs, net ≈ 828−178 = 650 µs.
The 659,766 sits in that band. It is **not** guaranteed until a
same-process pair on this tree.

---

## 10. KILLS / REOPEN_IF

- **KILL:** treating ICB as a path to 10 ms TOKEN_NS / 100 TPS.
  Already FALSIFIED by fusion-persistent. ICB does not touch
  weight_addressing 21.293 ms.
- **KILL:** cherry-picking `7400acf1b` as a whole commit onto HEAD.
  5 textual conflicts; ICB `open()` would delete
  `Qwen38HybridWeights` / `attach` / mixed.
- **KILL:** applying this ReplayGraphBuilder to mixed-sub15 and
  calling it the same 659,766. Different graph, unmeasured.
- **KILL:** spending 1.533 ms wall or 0.975 ms GPU as the ceremony
  claim.
- **KILL:** subtracting 659,766 from G024 1.653 ms and also banking
  659,766 as extra.
- **REOPEN_IF:** a serialized GPU lane runs a **same-process** ICB vs
  multi-encoder complete-wall on this HEAD and the named-fixed delta
  leaves the 0.60–0.70 ms band. Then update the number; do not
  invent a new mechanism.
- **REOPEN_IF:** Q4 `step` dispatch count leaves 964, or the
  production kernel family changes. `finish()` will fail-closed.
- **REOPEN_IF:** mixed-sub15 native ships and someone writes a
  second `ReplayGraphBuilder` over `encode_*_mixed`. New measurement
  required; this receipt does not transfer.

---

## 11. Evidence appendix (command output / excerpts)

### 11.1 Ancestry

```
$ git rev-parse 7400acf1b
7400acf1b46e3f2ebf4b6bde2a1caec8bd4a6f1c
$ git log -1 --format='%H%n%s%n%ci%n%P' 7400acf1b
7400acf1b46e3f2ebf4b6bde2a1caec8bd4a6f1c
qwen38-kill-fixed-overhead: ICB replay, encode 886us -> 91us
2026-08-16 17:28:59 -0400
9c87c500dc4c1e4b96962a99a1b017a23f273906
$ git merge-base --is-ancestor 7400acf1b HEAD; echo $?
1
$ git merge-base --is-ancestor 9c87c500dc4c1e4b96962a99a1b017a23f273906 HEAD; echo $?
0
$ git merge-base HEAD 7400acf1b
9c87c500dc4c1e4b96962a99a1b017a23f273906
$ git rev-list --count 9c87c500dc4c1e4b96962a99a1b017a23f273906..HEAD
86
$ git rev-list --count HEAD..7400acf1b
1
```

### 11.2 Receipt named-fixed (excerpt)

`git show HEAD:receipts/ascent-2026-08-16/QWEN38_FIXED_OVERHEAD_DELETED.json`

```
"commit": "9c87c500dc4c1e4b96962a99a1b017a23f273906"
"timing_label": "DIRTY_ENGINEERING"
"named_fixed_components_ns": {
  "encode_host_prepare": {"before_mean": 886200, "after_mean": 90981, "delta": -795219}
  "wait_minus_gpu":      {"before_mean": 425900, "after_mean": 561994, "delta": 136094}
  "submit":              {"before_mean": 10500,  "after_mean": 9420,   "delta": -1080}
  "tokenizer":           {"before_mean": 6300,   "after_mean": 6831,   "delta": 531}
  "epilogue":            {"before_mean": 1800,   "after_mean": 1708,   "delta": -92}
}
"named_fixed_sum": {
  "before_authority_mean_ms": 1.3307,
  "after_mean_ms": 0.670934,
  "after_mean_ns": 670934
}
"complete_token_wall": {
  "before_headline_ms": 38.216792,
  "after_headline_ms": 36.683916,
  "before_gpu_ms": 36.987458,
  "after_gpu_ms": 36.01225
}
"verification_pasted.complete_wall_stdout": {
  "SCHEDULE": "icb",
  "ICB_COMMANDS": 964,
  "ENCODE_HOST_PREPARE_NS": 90981,
  "WAIT_MINUS_GPU_NS": 561994,
  "NAMED_FIXED_SUM_MS": 0.670934
}
```

### 11.3 HEAD `step` (no ICB)

`crates/hawking-core/src/model/qwen38_hybrid_decode.rs:3292–3312`

```
pub fn step(&mut self, token: u32) -> Result<(u32, CommandBufferTiming)> {
    ...
    let mut tcb = TokenCommandBuffer::new(&self.context);
    self.encode_embed(&mut tcb, token)?;
    self.encode_layers(&mut tcb)?;
    self.encode_terminal(&mut tcb)?;
    ...
    let mut timing = tcb.commit_and_wait_timed()?;
```

`git show HEAD:…qwen38_hybrid_decode.rs | rg -n -i 'icb|IndirectCommand|Qwen38Schedule|ReplayGraph'` → no matches.

### 11.4 One wait

`crates/hawking-core/src/metal/mod.rs:5371` (`commit_and_wait_split`):

```
let t_sync = Instant::now();
cmd.wait_until_completed();
let wait_d = t_sync.elapsed();
```

### 11.5 964 dispatches (MEASURED, G024 ledger)

`receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json`:

```
"dispatches": {
  "embed": 1, "mixer_prefix": 576, "mlp_suffix": 384,
  "terminal": 3, "total": 964,
  "production_command_buffers": 1
}
"kernel_runtime_genome": "Qwen38HybridDecodeSession + qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 + qwen38_gated_delta_decode_vi + …; deltanet_vi_parallel=true concurrent_independent=false; 1 production CB / 964 dispatches; uniform-q4-v1 BPW=4.252735126866492"
"commit": "57ee82ccef7aba803416ec3562c8981277120fd4"
"median_encode_ns": 919250
"wait_minus_gpu_ns": 384250
```

### 11.6 G024 ceremony rows

`receipts/ascent-2026-08-16/G024_QWEN38_TOKEN_NS.json` `ranked_by_ns`:
host_preparation 919250, synchronization 384250, unattributed_residual
341925, command_submission 12084. Sum of those four = 1,657,509.
Anatomy intra-CB 336,926 + encode + wait + submit = **1,652,510**.

### 11.7 merge-tree conflict files

```
changed in both  crates/hawking-core/examples/ascension_qwen38_hybrid_greedy.rs  markers=1
changed in both  crates/hawking-core/src/metal/mod.rs                            markers=0
changed in both  crates/hawking-core/src/model/qwen38_hybrid_decode.rs           markers=4
```

(plus greedy C1 + decode C2–C5 = 5 hunks total; C2 is the import hunk
in the decode file.)

### 11.8 ICB `encode_token_icb` (`7400acf1b` `:2480–2491`)

```
fn encode_token_icb(&self, tcb: &mut TokenCommandBuffer<'_>, token: u32) -> Result<()> {
    ...
    write_u32(&replay.scalars, replay.token_off, token);
    write_u32(&replay.scalars, replay.position_off, self.position as u32);
    write_u32(&replay.scalars, replay.seq_len_off, (self.position + 1) as u32);
    tcb.execute_replayable_graph(&replay.graph)
}
```

### 11.9 HEAD metal still HashMap-per-replay (`:4696–4718`)

```
let mut resources = Vec::<ReplayResource>::new();
let mut resource_slots = HashMap::<u64, usize>::new();
for graph in graphs {
    for resource in &graph.resources {
        let address = resource.buffer.gpu_address();
        ...
        resources.push(ReplayResource { buffer: resource.buffer.clone(), usage: resource.usage });
```

### 11.10 Line counts

```
$ git show HEAD:crates/hawking-core/src/model/qwen38_hybrid_decode.rs | wc -l
    3905
$ git show 7400acf1b:crates/hawking-core/src/model/qwen38_hybrid_decode.rs | wc -l
    3027
$ git show 7400acf1b^:crates/hawking-core/src/model/qwen38_hybrid_decode.rs | wc -l
    2094
$ git diff --stat 9c87c500 HEAD -- crates/hawking-core/src/model/qwen38_hybrid_decode.rs
 .../qwen38_hybrid_decode.rs | 2111 ++++++++++++++++++--
 1 file changed, 1961 insertions(+), 150 deletions(-)
```

Decode commits on HEAD after ICB parent:
`d204f3642` genesis-children, `ef17aa24e` native-mixed-reader,
`f9d477c89` g002-per-facet-ab.

---

## 12. Cheapest experiment this lane did not run

Same-process complete-wall A/B on this HEAD, ICB vs multi-encoder,
uniform-q4-v1, 6 warm reps, 31 decode steps. Produces a **paired**
named-fixed delta that can replace 659,766. Serialized GPU lane only.
This lane is forbidden to take the device.

---

STATUS: IMPLEMENT_READY (Q4 surgical port). 659,766 MEASURED on
parent+ICB, ESTIMATED to survive on HEAD Q4, UNVERIFIED on this tree.
