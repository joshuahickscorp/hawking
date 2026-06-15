# Host-loop / GPU-sampling audit (Bible §7.5)

**Task I.** Where does host work sit on the per-token decode critical path, what
is already folded onto the GPU, and what is the *genuine, exact* §7.5 lever
(the "~12–15% residual" the bible claims)?

**Method.** Source read of the decode loop + sampler + GPU argmax kernel +
TokenCommandBuffer commit semantics, cross-checked against the measured
gpu_prod trace and the prior host-side kill family. GPU-FREE: no binary run;
all numbers below are tagged **(measured)** = read off an existing
trace/bench artifact, **(prior-measured)** = from a committed prior report,
**(estimate)** = arithmetic on those, **(proxy)** = none used here.

---

## TL;DR (honest verdict)

- **Greedy decode (the locked-config / bench path, `temp==0`) has essentially
  no remaining host lever.** The entire 36-layer graph + final norm + LM-head
  GEMV + **GPU argmax** + KV append are encoded into ONE `TokenCommandBuffer`,
  committed once, and only **4 bytes (the next-token u32)** cross the bus per
  token. The post-commit CPU work is `kv.seq_len += 1` + a detokenize +
  `sink()` callback — µs-scale. **This is `[[dead_levers.md → CPU+GPU pipelining]]`,
  re-confirmed against the current tree.**
- **The "~12–15% host gap" in Bible §7.5 is mostly mislabeled.** The
  *wall-minus-GPU-busy* residual at the current locked config is **~33.6% =
  ~11.0 ms/token (measured)**, but that residual was *directly measured to be
  GPU-side idle*, not host glue: CPU dispatch-encode is **5.77% of decode wall
  (measured, gpu_prod)** and the historical fine-grained split put genuine host
  glue (encode + commit stall) at **<3% (prior-measured)**. Driving "host glue"
  to zero buys **single-digit %, not 12–15%**, on the greedy path.
- **The one genuinely-claimable §7.5 lever is GPU-side NON-greedy sampling.**
  Default `SamplingParams` is `temperature=0.7, top_k=40, top_p=0.9` — i.e. the
  *default API path is non-greedy*, and non-greedy **cannot use the TCB path
  at all** (`qwen_dense.rs:1575-1580`). It falls back to `forward_token`, which
  runs the **whole forward host-orchestrated** and copies the **full ~32K-vocab
  f32 logit vector (~128 KB) back to the CPU every token** for a CPU sampler.
  The GPU kernels to fix this **already exist and are unwired**
  (`shaders/sample.metal`: `sample_temperature`, `sample_repetition`,
  `sample_topk`, `sample_topp`, `sample_multinomial`). This is exact, no-quality-
  cost (bit-exact-able), and the §7.5 deliverable.
- **Verdict: NEEDS-MEASUREMENT for the magnitude, GO on the build.** A weight-only
  CPU proxy cannot legitimately put a tps number on a Metal host-loop change. The
  single decisive gate is named below.

---

## 1. The decode per-token loop — host-step inventory

Loop body: `crates/dismantle-core/src/model/qwen_dense.rs:2463-2517` (macOS).
The per-step fork is the load-bearing line:

```
2470  let next_id = if use_tcb && !ffn_capturing {
2471      self.forward_token_greedy_tcb(last_id, pos)?      // GPU-folded, 4 B back
2472  } else {
2473      let mut logits = self.forward_token(last_id, pos)?; // FULL logits → CPU
2474      self.sampler.sample(&mut logits, &req.sampling)     // CPU sampler
2475  };
```

`use_tcb = DISMANTLE_QWEN_TCB && req.sampling.temperature == 0.0`
(`qwen_dense.rs:1579-1580`). So **the fork is decided by sampling mode**, and
the two branches have completely different host-cost profiles.

### Branch A — greedy TCB path (`forward_token_greedy_tcb`, line 3290; LM-head+argmax at 4446-4658)

| host step on critical path | where | crosses bus? | cost | status |
|---|---|---|---|---|
| embed lookup | GPU `embed_lookup_f32` (in the one TCB) | no | <0.1% GPU (measured, A4) | **folded** |
| 36× layer graph | GPU dispatches in the one TCB | no | 21.7 ms GPU busy (measured) | **folded** |
| final norm + LM-head GEMV | GPU (`gemv_q4_k_v4_predec*`) in TCB | no | part of the 21.7 ms | **folded** |
| **argmax / sampling** | GPU `sample_argmax_f32` in TCB (`4570/4593/4648`) | no | <0.1% GPU (measured) | **folded** |
| **single commit + sync** | `tcb.commit_and_wait()` (`4572/4650`; impl `metal/mod.rs:1120-1144` = one `commit()` + one `wait_until_completed()`) | — | the *only* per-token CPU↔GPU sync | inherent |
| **logit copy-back** | **none** — `token_buf.contents()` reads **4 bytes** (`4574-4575`, `4656-4657`) | **4 B** | µs | **folded** |
| KV pointer bump | CPU `self.kv.seq_len += 1` (`4573/4654`) | no | µs | trivial |
| vocab-prune remap | CPU `map.get(idx)` (`4578-4581`) | no | 1 hashmap/array lookup | trivial |
| detokenize | CPU `tokenizer.decode_one` (`2506`) | no | µs | trivial, off the GPU's critical path |
| usage-capture / sink | CPU `record_argmax` + `sink(Token)` (`2509-2510`) | no | µs (no-op unless capture flag) | trivial |

**Critical-path host total (greedy): the commit/sync + ~4 B readback + a few
µs of bookkeeping.** Everything compute-heavy is on the GPU inside one CB.

### Branch B — non-greedy fallback (`forward_token`, line 2675; CPU sampler `sample/mod.rs:43-102`)

This path is reached whenever `temperature != 0.0` (the **default**), or TCB is
off. It is a *different machine*:

| host step | where | crosses bus? | cost |
|---|---|---|---|
| embed lookup | CPU `embed_lookup` (`2685`) | — | small |
| per-layer rmsnorm/Q/K/V/RoPE/MHA/FFN | **host-orchestrated dispatches**, results read back into `Vec<f32>` per op (`2706-...`); MHA + biases + RoPE run on CPU | many round-trips/layer | the dominant cost |
| **full logits** | returns **`Vec<f32>` of length `vocab` (~32K → ~128 KB)** to the CPU **every token** | **~128 KB/token** | bus + alloc |
| repetition penalty | CPU loop over `recent` window (`sample/mod.rs:45-57`) | — | ≤64 iters |
| temp / softmax / sort / top-k / top-p / draw | CPU (`sample/mod.rs:62-101`); **`indexed.sort_by` is an O(V log V) sort of the full 32K vector** | — | the sort dominates the sampler |

Branch B is the path that **`[[dead_levers.md → CPU+GPU pipelining]]`'s
resurrection clause explicitly flags**: *"if a sampling mode other than
greedy/argmax becomes the primary (top-k/temperature with CPU-heavy logic),
re-measure."* The default API params make non-greedy the primary path for any
real user request — the bench just happens to pin `temp=0`.

---

## 2. What is ALREADY dead (with pointers) — do not re-attack

These were measured and killed; the Kill Protocol classification is in
`reports/dead_levers.md`. Summary so the next session does not re-spawn them:

1. **CPU dispatch-encode overlap / ICB / megakernel-for-dispatch-count.**
   - **Measured:** CPU encode = **0.22 ms = 0.51% of wall** (old config,
     `[[v230_icb_dead]]`, `[[cpu_gpu_pipelining_audit]]`); ICB POC was **+32%
     per-dispatch encode but +0.9% e2e**; concurrent Q/K/V encoder **+1.68%
     e2e** (below the +5% gate); PSO-transition batching **1.06×** (free).
   - **Current trace re-confirm (measured):** `dispatch_wall_pct_of_decode =
     5.77%` and `rss_mb_after_run` fine — the dispatch-encode wall is a single-
     digit % of decode even at the current 616-dispatch/token count.
   - **Type-1** (a measured property: the GPU graph is one CB; encode is cheap).

2. **"The decode gap is host work."** **FALSE / killed.**
   `[[gpu_us_accuracy_verified_2026_05_24]]`: at production-scale dispatches
   `host_wall / Σgpu_us = 1.03×` (CSB is accurate to 3%), and
   `[[decode_gap_anatomy_2026_05_24]]`: the residual is **real GPU-side idle**
   ("the GPU is doing something OTHER than executing the dispatched kernels"),
   not CPU. Every host candidate (encode, commit stall, driver per-dispatch,
   concurrent dispatch, PSO) was eliminated by direct measurement. **Type-1.**
   ⇒ The §7.5 residual is **not** harvestable by a tighter host loop on the
   greedy path; it is inter-dispatch GPU scheduling slack, whose real lever is
   *fewer/faster kernels* (Bible §6 / A5-A6 vectorized Q4_K), not host glue.

3. **Megakernel / persistent GPU loop** to remove the residual: bible-
   deprioritized (`plans/...:150`), and `[[megakernel_revival_nlayer_bench]]`:
   8-layer f16 fused = **4.4× slower**. Not a host lever.

---

## 3. What GENUINELY remains — the exact §7.5 lever

### 3.1 GPU-side non-greedy sampling (the real, claimable item)

**The gap, exactly.** For `temp != 0` (default) the decode loop:
- runs `forward_token` (host-orchestrated, per-op readback), and
- copies the **full f32 logit vector** to the host every token, then
- runs a **CPU sort of the whole vocab** for top-k/top-p.

**Why it is claimable and not yet done:** the GPU kernels already exist and are
*unwired* in production:
- `shaders/sample.metal`: `sample_temperature`, `sample_repetition`,
  `sample_topk` (K≤64, K rounds of parallel argmax), `sample_topp` (nucleus),
  `sample_multinomial` (CDF draw with a host-supplied uniform variate).
- `metal/mod.rs:413-418` already maps these kernel names for tracing.
- Only `sample_argmax_f32_tcb` (`kernels/mod.rs:6886`) is wired into the decode
  TCB. The greedy LM-head→argmax→4-byte-readback structure
  (`qwen_dense.rs:4570-4582`) is the **exact template** to extend: append the
  top-k/top-p/multinomial dispatches into the same TCB and read back the same 4
  bytes. The host supplies one f32 uniform per token (negligible) instead of
  the whole logit vector.

**What this removes from the critical path (per token):**
- the **~128 KB logit copy-back** → **4 B** (bus traffic cut ~32,000×);
- the **CPU O(V log V) sort** of ~32K logits → a bounded GPU top-K (K≤64);
- the per-op host orchestration of `forward_token` → the single-CB TCB graph
  the greedy path already uses (this is the larger structural win — non-greedy
  currently doesn't even use the fast forward).

**Magnitude — explicitly NOT asserting a tps number (Type-2-error guard).**
A weight-only CPU proxy cannot measure a Metal host-loop / bus-sync change;
asserting "+X%" here would be a wrong simulation. The honest bracket:
- *Upper-bound framing (estimate):* bible §7.5 ceiling for the whole host-loop
  item is "up to ~10%+", but that bound was set against the *mislabeled* 12–15%
  residual; the greedy-path host fraction is **<6% measured**, so the greedy
  ceiling is well under that.
- *Where the real win lives (estimate):* the non-greedy path is structurally
  *slower than greedy today* (no TCB, full readback, full sort). Folding it onto
  the GPU plausibly brings non-greedy decode **up to ~greedy-path tps** — i.e.
  the win is "make sampled generation as fast as greedy", a **path-relative**
  win, not a +X% on the already-fast greedy bench. The size depends entirely on
  how far below greedy the current CPU non-greedy path sits, which is **unmeasured**.

### 3.2 Persistent zero-alloc host loop (the residual-shrink item) — small, bounded

`forward_token_greedy_tcb` already reuses a `DenseDecodeArena` (pinned buffers),
and `forward_token` allocates per-op `Vec`s but is the non-production path.
The remaining greedy-path allocation churn is minor; the bible's "persistent
zero-allocation host loop" is **bounded by the <6% host fraction** and is
**not** where the 11 ms/token residual lives (that is GPU-side, §2.2). Keep as a
standing-discipline cleanup, **not** a tps lever. **NEEDS-MEASUREMENT only if a
future kernel restructuring pushes dispatches/token far past 616 so encode
crosses ~1 ms/token** (same gate as the ICB kill).

### 3.3 Draft-loop host overhead (speculation paths) — context, not a §7.5 item

- **n-gram / user-draft propose** (`speculate/user_ngram.rs:155-196`): a CPU
  HashMap chain of ≤K lookups — **µs-scale**, lossless-by-construction, and
  *overlappable* with GPU verify. Not on the bottleneck.
- **Eagle5 head propose** (`speculate/eagle5_forward.rs`,
  `eagle5.rs:532 propose_rollout_chained`): a **full fp32 CPU mini-forward per
  drafted token** (in_proj + block + lm_head over ~32K vocab). This is *real*
  host compute on the spec critical path, but it belongs to §7.2 (speculation),
  and EAGLE is currently NO-GO (`[[dead_levers]]`, τ=0.877). Folding the draft
  head onto the GPU is a §7.2 question, not the §7.5 host-loop item. Flagged so
  it is not double-counted.

---

## 4. GPU-lane validation (the decisive gate)

**Single decisive gate (names the measurement that settles §7.5):**

> A **paired GPU bench** (`tools/bench/paired_lever.sh`-style, A/B, Claude-open
> OK per `[[feedback_bench_with_claude_open]]`) of **non-greedy decode**
> (`temperature=0.7, top_k=40, top_p=0.9`) at the locked config:
> **A =** current CPU sampler path (`forward_token` + `Sampler::sample`);
> **B =** GPU-folded path (TCB forward + `sample_topk`/`sample_topp`/
> `sample_multinomial` in-CB, 4-byte readback).
> Report `dec_tps_B / dec_tps_A` (the lever's actual size) **and** the
> §1 oracle `host_wall − Σgpu_us` per token for each (it should drop on B).
> **Correctness gate:** seeded-RNG token parity — feeding the same per-token
> uniform variate to the GPU multinomial as the CPU draw must reproduce the
> CPU sampler's exact token stream (the kernels are written to match: see the
> tie-break and CDF-convention comments in `sample.metal`). If a fully bit-
> identical match is infeasible across CPU↔GPU float order, the looser gate is
> distributional (same top-k membership + KL of the sampled histogram over
> N≥1000 tokens within noise).

**Why a proxy can't do it:** the lever is a *bus-sync + host-orchestration*
change on Metal. NumPy on dequantized weights can confirm the kernels are
numerically correct (already covered by the existing parity intent in
`sample.metal`), but **cannot** measure the wall-clock effect of removing the
logit copy-back and the CPU sort — that is intrinsically a paired GPU bench.
Recording a NO-GO from a CPU proxy here would be a Type-2 error.

**Secondary cross-check (greedy, confirms §2 not §3):** re-run
`DISMANTLE_TCB_TRACE=gpu_prod` on greedy decode and confirm
`dispatch_wall_pct_of_decode` stays ~6% and `Σgpu_us/wall ≈ 66%` — i.e. the
greedy residual is still GPU-side idle, so **do not** chase it with a host loop.

---

## Appendix — measured reconciliation (current locked config)

Source: `reports/traces/qwen3b_decode_gpu_prod_2026_05_31.json` (the A4 trace,
`DISMANTLE_QWEN_TCB=1 VOCAB_PRUNE=32000 Q4K_LMHEAD=1 FFN_DOWN_Q4K=1 Q4K_PREDEC=1
LMHEAD_PREDEC=1`, greedy), analyzed per `tools/bench/analyze_tcb_trace.py`.
Sibling clean untraced baseline: `reports/a4_clean_walltime.json` (31.0 dec_tps).

| quantity | value | tag |
|---|---|---|
| decode wall (traced run) | **32.71 ms/token** (30.57 dec_tps, 128 tok) | measured |
| decode wall (clean untraced sibling) | 32.26 ms/token (31.0 dec_tps median) | measured |
| GPU busy Σgpu_us / token | **21.73 ms/token** | measured |
| **GPU-busy fraction** | **21.73 / 32.71 = 66.4%** | measured |
| **wall − GPU-busy residual** | **~11.0 ms/token = ~33.6%** | estimate (from the two measured rows) |
| CPU dispatch-encode wall fraction | **5.77%** (`dispatch_wall_pct_of_decode`) | measured |
| dispatches / token | **616** (19712 samples ÷ 32 traced tokens) | measured |
| historical genuine host glue (encode+commit-stall) | **<3%** | prior-measured (`[[gpu_us_accuracy_verified]]`, `[[decode_gap_anatomy]]`) |

**Reading:** the **33.6% residual is real but ~85–95% of it is GPU-side idle**
(prior-measured), not host. The bible's "12–15% host gap" overstates the *host*
portion by conflating it with the GPU-idle residual. The only host work that is
both large and on the critical path is **non-greedy sampling's full-logit
copy-back + CPU sort** — which is *off the bench* (the bench is greedy) and is
the genuine §7.5 build. **GO to build the GPU-sampling wiring; NEEDS-MEASUREMENT
for its tps via the paired non-greedy bench above; the greedy residual is NOT a
host lever (Type-1, do not re-attack).**
