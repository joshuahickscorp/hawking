# Kill-ledger reconciliation + §3 envelope re-anchor

**Date:** 2026-05-31 · **Author:** Task-C (GPU-free reconciliation pass) · **Mode:** PROPOSE only — no edits applied to the bible or `dead_levers.md`.

Resolves the internal inconsistencies between `reports/dead_levers.md`, the bible
(`plans/throughput_bible_2026_05_30.md` §3 envelope + §8.3.1 Kill Protocol/ledger), and the
memory files MEMORY.md indexes. Ground truth was read from git (`git show`/`git log`), the
oracle output files, and the per-fact memory notes. Every number is tagged
(measured)/(proxy)/(estimate) per the QTIP-oracle house style.

---

## 0. TL;DR — the three live inconsistencies and their resolutions

| # | Inconsistency | Ground truth | Where it is still wrong |
|---|---|---|---|
| **1** | L1.4 data-aware SVD oracle: "READY-but-UNRUN" vs "ran 2026-05-31 and died" | **It RAN and DIED.** Commit `dba6ed6` created the oracle; `f5d5f53` recorded NO-GO; output `reports/oracle_dataaware_lowrank.md`. (measured, offline NumPy) | Bible §8.3.1 ledger **lines 559 & 560** still say "Oracle written, not yet run." `dead_levers.md` L1.4 entry (lines 99–101) never recorded the reframe ran. Kill-audit memory body (lines 32–36) still says "WRITTEN + compiles, NOT YET RUN." |
| **2** | L1.3 cross-layer delta status | **Doubly dead:** weight-space (Type-1, `oracle_interlayer_delta.md`) AND data-aware (`oracle_dataaware_lowrank.md` L1.3 §, data-wt cross-layer cos ≈ 0). Both → Type-1. (measured) | `dead_levers.md` L1.3 (lines 21–26) is **correct** (carries the 2026-05-31 update). Bible §8.3.1 ledger **line 559** is stale ("not yet run" + classifies Type-2). |
| **3** | §3 decode anchor: ~39 vs ~31 | **~31 is the honest anchor.** Clean-room `clean_room_batch.sh` §B = **29.12 dec_tps** (measured, contamination-free); A1 paired 30.94→31.55; A4 31.0 median. Quote **~29–31**. ~39 was optimistic/contaminated. | Bible **line 17** (§0/§2 diagnosis) still states "clean ~39 tps … ~47% of peak" with **no superseded flag**. Bible **line 101** cites a stale "L1.4 data-aware-SVD reframe if it resurrects" (it resolved dead → cannot resurrect cheaply). Lines 109/115 carry ~39 but ARE flagged. |

The memory files (`clean_room_baselines_2026_05_31`, `moat_status_forward_path_2026_05_31`)
are internally consistent and already correct (~31 anchor, ~39 dead, Q3_K dead, QTIP the byte-cut path).
The drift is in the **bible** (the §8.3.1 ledger and the §0/§2 diagnosis line) and a **stale L1.4 note**
in `dead_levers.md` + the kill-audit memory body.

---

## 1. (a) L1.4 low-rank data-AWARE SVD oracle — RESOLVED: it RAN and it DIED

**Verdict: the oracle RAN on 2026-05-31 and the lever is NO-GO. Now Type-1.** This is the
authoritative status; "READY-but-UNRUN" is the stale earlier state.

**Evidence (git, measured):**
- `dba6ed6` *"bench: data-aware low-rank reframe oracle for L1.4/L1.3 (Type-2 retest)"* — created `tools/bench/oracle_dataaware_lowrank.py`.
- `f5d5f53` *"bench: L1.4/L1.3 data-aware reframe RESOLVED — NO-GO (in-sample artifact)"* — committed the result.
- `57eefb6` *"docs: register EAGLE-3 NO-GO + close the L1.3/L1.4 data-aware reframe"* — updated `dead_levers.md` (L1.3 only) but **did not touch the bible**, which is why the bible ledger is now stale.
- Output file present: `reports/oracle_dataaware_lowrank.md` (29 KB, 36 layers × {ffn_gate, ffn_up}, 800 tok/layer, 70/30 train/held split, peak RSS 3.37 GB, wall 1254 s).

**The three deciding numbers (measured, mean over sampled FFN tensors, r=64):**
- data-norm energy@64 = **0.990** (vs Frobenius 0.117) — looked like a strong GO …
- … but activation **X effective rank (99% energy) = 63** (participation ratio 3.1 of 2048) → `data-E64≈1.0` is the **trivial low-activation-rank artifact**, true for ANY weight, saving **no weight bytes**.
- held-out data-norm error @64 = **0.1390** vs in-sample **0.0787** (1.8× blow-up; 2/72 tensors blow up >3× above gate) — the in-sample energy is partly overfit.
- byte gate: **1/72** sampled FFN tensors have a sub-Q4_K-byte config under the 0.02 functional-error gate (in-sample, the optimistic side).

**Type classification: Type-1 (now).** The data-aware reframe was the named Type-2 escape; it
was run and it died on a *measured property of the captured activations* (they are themselves
rank-≤64), which no codec cleverness changes — a low-rank-in-data-norm weight codec would still
have to store the full row-space of W to serve tokens whose activations rotate out of the
captured subspace. **Caveat (honest):** target W is the dequantized Q4_K weight (only the Q4_K_M
GGUF is local), so this is a **lower bound** on what AWQ-from-f16 could do; a NO-GO is decisive
here, a marginal GO would have deferred to the f16/AWQ lane. The NO-GO is decisive → L1.4 is dead
in both forms.

---

## 2. (b) L1.3 cross-layer delta — RESOLVED: doubly dead, Type-1

**Verdict: NO-GO in both the weight-space and the data-aware form → Type-1.**

- **Weight-space (Type-1, measured):** `reports/oracle_interlayer_delta.md` — pairs (0,1)/(17,18)/(34,35); mean cosine(W[L],W[L+1]) = **+0.0003**; std(delta)/std(orig) = **1.61** (anti-compressible); top-64 SVD energy of delta = **0.23**; optimal affine α* ≈ 0 (a learned gain buys nothing); 0/7 tensor types beat Q4_K bytes.
- **Data-aware (measured):** `reports/oracle_dataaware_lowrank.md` L1.3 section — data-weighted cross-layer cosine ≈ 0 across all 35 layer pairs (ffn_up ≈ 0; ffn_gate shows a *positional* gate-cosine but `E64(delta) ≈ E64(W[L+1])`, i.e. the resident W[L] gives W[L+1] no free basis). The reframe Frobenius-cosine≠data-norm-cosine was tested; the data norm does not rescue it.

`dead_levers.md` L1.3 (lines 21–26) is **already correct** — it carries the 2026-05-31 update
recording the data-aware NO-GO. Only the bible §8.3.1 ledger (line 559) is stale (still "not yet
run" and classified Type-2 where it is now Type-1).

---

## 3. (c) The §3 decode anchor — RESOLVED to ~31 (measured); every stale ~39 flagged

**Honest anchor: ~29–31 dec_tps clean (measured), NOT ~39.**

**Evidence (measured, contamination-free):** `tools/bench/clean_room_batch.sh` §B with Claude
quit, greedy temp=0, 256 tok, locked fast-path → **clean dec_tps = 29.12**. Corroborated by A1
(paired 30.94→31.55) and A4 (31.0 median). The canonical figure is a thermal-protocol median
(`clean_bench.sh` ×N); expect ~29–31. With the opt-in A6.5 f16-scales flag, ≈33–34 (= ~31 + the
+6–9% A6.5 delta, measured paired). The ~39 figure was an optimistic/contaminated baseline (a
running agent session inflates dec_tps — the bench-contamination finding measured up to ~4–5×;
the mild in-session EAGLE bench reported 36.9).

### Every place the dead ~39 is still cited (bible)

| Bible line | Citation | Flagged? | Action |
|---|---|---|---|
| **17** | "dismantle's clean ~39 tps means ~70 GB/s … ≈ ~47% of the 150 GB/s peak" (§0/§2 diagnosis) | ❌ **NO flag** | **Add superseded note.** The *physical argument* (kernel-bound, not gap-bound) is unaffected — recomputed at ~31: ~31 tps × 1.9 GB ≈ **59 GB/s ≈ 39% of peak**, which strengthens "kernels are the wall" (further below llama's ~60%). Do not delete; annotate. |
| **101** | "the L1.4 data-aware-SVD reframe if it resurrects" | n/a (not a tps cite) | **Stale** — L1.4 resolved dead (§1 above); it cannot resurrect cheaply. Re-word to drop the live-resurrection implication. |
| **109** | "Anchored at ~39 tps clean … theoretical-100% ≈ 78 tps" | ✅ inline `⚠ SUPERSEDED` | OK as-is (already flagged). |
| **115** | table row "now (clean) | ~39 | ~50% | measured" | ⚠ only via §3.0 block above | **Optional:** annotate the row inline so it isn't quoted out of context. |

The §3.0 status block (lines 99–105) is correct and already says the re-projection from ~31 is
the remaining attended task. **This reconciliation does NOT do the full re-projection of the
stacked envelope** (that is an attended-session product per §3.0) — it only (i) flags the one
unflagged ~39 at line 17, (ii) corrects the stale L1.4-resurrection phrase at line 101.

---

## 4. Authoritative dead-lever status table

Legend — **Triple** = carries the Kill-Protocol 3-part record (Type-1/2 + reframe considered +
why-reframe-dies/oracle pointer), per CLAUDE.md + bible §8.3.1. Pre-protocol legacy kills
(before 2026-05-30) predate the protocol; the protocol retro-filled only the four Phase-A kills,
so a missing tag on a legacy kill is **expected, not a defect** (flagged "legacy").

| Lever (dead_levers.md) | Status | Type | Triple? | Reconciliation note |
|---|---|---|---|---|
| L1.1 KV working-set eviction | NO-GO 05-31 | Type-1 | ✅ full | Consistent. |
| **L1.3 cross-layer delta** | NO-GO | **Type-1** (both forms) | ✅ (entry) | **Entry correct.** Bible ledger line 559 stale (Type-2 + "not yet run"). |
| **L1.4 low-rank+residual** | NO-GO | **Type-1** (data-aware ran+died) | ⚠ **partial** | **Entry stale:** still framed as the 05-30 Frobenius kill; never records the data-aware reframe ran. Bible ledger line 560 stale. **Fix below.** |
| L1.5 learned codebook | NO-GO 05-30 | Type-1 (gather) | ✅ (entry+ledger) | Consistent (reframe = QTIP). |
| L2.2 FFN block-256 sparsity | NO-GO 05-30 | Type-1 | ✅ full | Consistent (reframe = trained-for L5.1). |
| L3.1 vocab screen | NO-GO 05-31 | Type-1 | ✅ full | Consistent. |
| Q3_K sub-Q4 byte-cut | NO-GO 05-31 | Type-1 | ✅ full | Consistent (reframe = QTIP). |
| Q4_K MMA rows≤cols | NO-GO 05-31 (partial) | Type-1 | ✅ full | Consistent; rows>cols GO, deferred. |
| A10 access-order repack | NO-GO 05-31 | Type-1 | ✅ full | Consistent. |
| Semantic cache | NO-GO 05-31 (parked) | **Type-2** | ✅ full | Consistent; alive behind named oracle on real logs. |
| EAGLE-3 trained head | NO-GO 05-31 | (reframe = n-gram, sub-gate) | ⚠ no explicit Type tag | Has reframe + oracle-gate in resurrection check; **add explicit "Type-1 (offline ceiling τ=0.877 ≪ 2.5)"** for completeness. Optional. |
| Mixed-precision / W4A8 | HELD 05-24 | — | ⚠ legacy | Held, not a hard NO-GO; resurrection check names AWQ-from-f16 path. Pre-protocol. |
| CPU+GPU pipelining | killed 05-22 | — | ⚠ legacy | Pre-protocol; cost-share kill (0.51%). |
| Host-side per-dispatch family | exhausted 05-24 | — | ⚠ legacy | Pre-protocol; family kill. |
| ICB | killed 05-14 | — | ⚠ legacy | Pre-protocol. |
| MoE megakernel | killed 05-14 | — | ⚠ legacy | Pre-protocol (HW sync wall — effectively Type-1). |
| MoE serial dispatch | killed 05-11 | — | ⚠ legacy | Pre-protocol. |
| MLA Phase 4 rewrite | killed 05-22 | — | ⚠ legacy | Pre-protocol; cost-share (attn 2.4%). |
| Phase Y sumy-trick v3 | killed 05-11 | — | ⚠ legacy | Pre-protocol; register-pressure. |
| Predec 4-row ILP | parked 05-30 | — | ⚠ legacy | Parked pending profiling, not a hard kill. |
| Q5_0 simd_shuffle | killed 05-14 | — | ⚠ legacy | Pre-protocol (HW coalescing — effectively Type-1). |
| Q8-KV layer-diff | killed 05-21 | — | ⚠ legacy | Pre-protocol (uniform routing). |
| Eagle5 v1 routing-mask | killed 05-21 | — | ⚠ legacy | Pre-protocol (uniform routing). |
| f16 residual stream | killed 05-11 | — | ⚠ legacy | Pre-protocol (accumulated error — Type-1). |
| Spec-decode ExactShared | regression 05-11 | — | ⚠ legacy | Pre-protocol; resurrection = batched verify. |
| LM head simdmat | killed 05-11 | — | ⚠ legacy | Pre-protocol; cost-share (~4%). |

**Kills missing the full triple:** the one *actionable* gap is **L1.4** (status text never caught
up to the reframe having run). EAGLE-3 lacks an explicit Type tag (low priority — reframe+oracle
are present in prose). All other untagged entries are **pre-2026-05-30 legacy kills** that the
protocol explicitly did not require retro-filling beyond the four Phase-A levers; converting them
is a discretionary cleanup, NOT a correctness fix, and several (MoE megakernel, Q5_0 shuffle, f16
residual) are obviously Type-1 if a future pass wants to tag them.

---

## 5. EXACT proposed diffs (orchestrator applies; do NOT apply here)

### 5.1 Bible `plans/throughput_bible_2026_05_30.md`

**Diff B1 — §8.3.1 ledger, line 559 (L1.3 row): mark the reframe RAN + Type-1.**
Replace the final cell of the L1.3 row:
- OLD: `**Oracle written, not yet run:** \`tools/bench/oracle_dataaware_lowrank.py\` (L1.3 section). Prior is *against* it (weight orthogonality + full-ish activation rank); and a GO collapses into L1.4/L5.1, not a standalone lever. Confirm-or-kill cheaply, then stop. |`
- NEW: `**RAN 2026-05-31 — NO-GO → now Type-1** (\`reports/oracle_dataaware_lowrank.md\` L1.3 §): data-weighted cross-layer cosine ≈ 0 and \`E64(delta) ≈ E64(W[L+1])\` — the resident W[L] gives W[L+1] no free basis even in the data norm. Both weight-space and data-aware cross-layer reference are dead; do not re-spawn. |`

Also update the L1.3 row's **Type** cell (2nd column) from `**Type-2 (narrow)**` to
`**Type-2 → RESOLVED Type-1 (2026-05-31)**`.

**Diff B2 — §8.3.1 ledger, line 560 (L1.4 row): mark the reframe RAN + Type-1.**
Replace the final cell of the L1.4 row:
- OLD: `**Oracle written, not yet run:** \`oracle_dataaware_lowrank.py\` (L1.4 section) — measures data-weighted energy@r and the functional-error-at-matched-bytes vs Q4_K. The one genuinely-too-early kill; decisive offline. (Target = Q4_K-recon, a lower bound; a marginal GO defers to the f16/AWQ lane.) |`
- NEW: `**RAN 2026-05-31 — NO-GO → now Type-1** (\`reports/oracle_dataaware_lowrank.md\`): data-norm E64≈0.990 was an **in-sample artifact** — the captured activations are themselves effectively rank-≤64 (participation 3.1/2048, rank99% 63), so \`data-E64≈1.0\` holds for ANY weight and saves no WEIGHT bytes; held-out error blows up (0.139 vs 0.079) and 1/72 tensors beat Q4_K bytes. Lower-bound caveat: target W is dequantized Q4_K, so a marginal GO would have deferred to f16/AWQ — but the NO-GO is decisive. QTIP/mixed-prec remain the live byte-cut levers. |`

Also update the L1.4 row's **Type** cell from `**Type-2 (strong)**` to
`**Type-2 → RESOLVED Type-1 (2026-05-31)**`.

**Diff B3 — line 17 (§0 diagnosis): flag the stale ~39 and recompute at ~31.**
- OLD: `The decisive physical check: both dismantle and llama.cpp read the same ~1.9 GB of Q4_K_M weights per token, so dismantle's clean ~39 tps means ~70 GB/s wall-averaged ≈ ~47% of the 150 GB/s peak. A 70%-idle GPU would require pushing all 1.9 GB during the busy 30% at ~155% of peak — impossible. The GPU is busy ~85% of the wall, running the Q4_K GEMVs at **~37–47% of peak vs llama's ~60% and MLX's higher still.**`
- NEW: `The decisive physical check: both dismantle and llama.cpp read the same ~1.9 GB of Q4_K_M weights per token. *(Anchor corrected 2026-05-31 to ~31 dec_tps clean — §3.0 Correction 2; the ~39 below is the superseded figure, left for the audit trail.)* At the honest ~31 tps that is ~59 GB/s wall-averaged ≈ ~39% of the 150 GB/s peak (the old ~39-tps reading gave ~70 GB/s ≈ ~47%); either way a 70%-idle GPU would require pushing all 1.9 GB during the busy 30% at well over 100% of peak — impossible. The GPU is busy ~85% of the wall, running the Q4_K GEMVs at **~39–47% of peak vs llama's ~60% and MLX's higher still** — the lower anchor only *strengthens* the kernel-bound conclusion.`

**Diff B4 — line 101 (Correction 1): drop the stale "L1.4 reframe if it resurrects".**
- OLD: `The path to >34 dense tps now runs ONLY through **fewer bytes** (Q3_K wiring, QTIP, or the L1.4 data-aware-SVD reframe if it resurrects) or the **spec / stateful axes** — NOT decode kernels.`
- NEW: `The path to >34 dense tps now runs ONLY through **fewer bytes** (QTIP — Q3_K is clean-confirmed dead at 33.3 GB/s, and the L1.4 data-aware-SVD reframe RAN 2026-05-31 and is dead, §8.3.1) or the **spec / stateful axes** — NOT decode kernels.`

**Diff B5 (optional) — line 115 table row: annotate the superseded anchor inline.**
- OLD: `| now (clean) | ~39 | ~50% | measured |`
- NEW: `| now (clean) | ~39 → **~31** ⚠ | ~50% → ~39% | measured (re-anchored 2026-05-31, §3.0 Corr. 2) |`

### 5.2 `reports/dead_levers.md`

**Diff D1 — L1.4 entry (lines 99–101): record the data-aware reframe ran + died (mirror the L1.3 update pattern).**
Insert after line 101 (after the `**Resurrection check:**` line of the L1.4 entry), a new line:
- INSERT:
  `**Update 2026-05-31 — data-aware reframe RAN and DIED → now Type-1:** the activation-aware SVD reframe (the named Type-2 escape; ASVD/SVD-LLM on \`W·C^{1/2}\`) was tested (\`reports/oracle_dataaware_lowrank.md\` + \`tools/bench/oracle_dataaware_lowrank.py\`, 36 layers × {ffn_gate, ffn_up}, 800 tok/layer, 70/30 held-out split). NO-GO: data-norm E64≈0.990 is an **in-sample artifact** — captured activations are effectively rank-≤64 (participation 3.1/2048, mean rank99% 63), so \`data-E64≈1.0\` holds for any weight and saves no WEIGHT bytes; held-out error blows up (0.139 vs 0.079 in-sample); 1/72 FFN tensors beat Q4_K bytes at the 0.02 gate. Lower-bound caveat (target W = dequantized Q4_K, only the GGUF is local) noted, but the NO-GO is decisive. Both the data-free (2026-05-30) and data-aware forms are dead → QTIP is the surviving byte-cut codec.`

**Diff D2 — L1.4 Status line (line 99): reflect Type-1 + reframe-resolved.**
- OLD: `**Status:** killed 2026-05-30 by offline byte-budget oracle (before any kernel)`
- NEW: `**Status:** NO-GO — killed 2026-05-30 (data-free SVD) + **data-aware reframe RAN and confirmed dead 2026-05-31** (→ now **Type-1**; see Update below)`

**Diff D3 (optional) — EAGLE-3 entry (line 32): add an explicit Type tag for triple-completeness.**
- OLD: `**Status:** NO-GO concluded 2026-05-31 — doubly confirmed (offline held-out + on-device)`
- NEW: `**Status:** NO-GO concluded 2026-05-31 — doubly confirmed (offline held-out + on-device); **Type-1** (offline ceiling τ=0.877 ≪ 2.5 gate is a measured property of the trained head on this workload; n-gram τ=1.43 is the surviving reframe, also sub-gate)`

### 5.3 Memory (out-of-scope for the orchestrator's repo-doc apply, listed for the attended memory pass)

- `memory/kill_protocol_reframe_audit_2026_05_30.md` **body lines 32–36** still say the oracle is
  "WRITTEN + compiles, **NOT YET RUN**." This contradicts the same file's own lines 23–30 (which
  record it ran and died). The MEMORY.md index line for this file is already correct
  ("L1.4 data-aware SVD oracle READY-but-UNRUN" → should read "RAN 2026-05-31, NO-GO"). Proposed:
  update the body's "Artifact" paragraph to "RAN 2026-05-31 (`f5d5f53`); NO-GO — in-sample
  artifact (activations rank-63)" and update the MEMORY.md index one-liner. *(Flagged only; memory
  edits are an attended `consolidate-memory` pass, not this reconciliation's scope.)*

---

## 6. What this reconciliation deliberately does NOT do

- **It does not re-project the full §3 stacked envelope from ~31.** The §3.0 block already names that
  as the remaining attended task; re-deriving every "% of the way to ~50" row is an attended-session
  product (the bible is an attended doc). This pass only flags the one *unflagged* ~39 (line 17) and
  removes a stale resurrection phrase (line 101).
- **It does not record any new Kill.** Per the Kill Protocol, a CPU/weight-only proxy may only
  *confirm* an already-measured death; the L1.3/L1.4 deaths were already measured (NumPy on
  dequantized GGUF weights) and committed — this pass propagates them, it does not manufacture them.
- **It does not retro-fill the pre-2026-05-30 legacy kills** with Type tags. The protocol scoped
  retro-fill to the four Phase-A kills; the legacy entries' missing tags are expected. Tagging them
  is a discretionary future cleanup.

---

## 7. Decisive gate (what would settle anything still open here)

Nothing in this reconciliation is *unsettled* — L1.3/L1.4 are decisively dead on offline NumPy
oracles (a NO-GO on the dequantized-Q4_K lower bound is decisive), and the ~31 anchor is a clean
contamination-free measurement. The only forward gate is the one the §3.0 block already names: the
**canonical thermal-protocol decode median** (`clean_bench.sh` ×N, Claude quit) to convert the
single 29.12 clean run into the quotable ~29–31 anchor — and that is a bench task, not a ledger
task. (For L1.4's lower-bound caveat, the *only* thing that could revive it is a **Colab f16/AWQ**
re-run, but the Q4_K-recon NO-GO makes that a low-prior bet, not a pending gate.)
