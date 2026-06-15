# Next session — opening prompt for dismantle (paste this whole file)

> Self-contained continuation prompt. Branch `codex/maximal-spec-colab`. Commits authored
> `Joshua Hicks <joshuahicksboba@gmail.com>` **inline** (`git -c user.name=... -c user.email=...`),
> **NO AI attribution of any kind** (no `Co-Authored-By`, no "Generated with"). **Do NOT push** — 26
> commits are local-ahead of origin; the user pushes when they decide. Read `CLAUDE.md` (operating
> contract) and `plans/session_closeout_2026_05_31_evening.md` (what just shipped) first.

## What dismantle is
From-scratch Rust + Metal single-stream inference engine for Qwen2.5-3B-Q4_K_M on an M3 Pro (18 GB).
Strategy: `plans/throughput_bible_2026_05_30.md` — read **§3.0** (status correction) and **§8** (the
stateful moat). **Decode is at its kernel ceiling — clean ~29–31 tps (MEASURED 2026-05-31), NOT ~39
(that anchor is dead).** The decode-GEMV micro-opt track is **CLOSED**. Forward tps runs only through
**fewer bytes** (QTIP) or **spec** (draft tuning), not decode kernels.

## Current state (all MEASURED clean unless noted — `reports/clean_room_baselines_2026_05_31.md`)
- **Decode anchor = ~29–31 tps** (clean 29.12). ~39 is superseded. **Energy floor = 0.17 J/tok.**
- **Byte-cut = QTIP only.** Q3_K is Type-1 dead (clean 23% peak, compute-bound).
- **Draft tuning** merged lossless **default-OFF**; it's **−78% when on** until its verify fast-path lands.
- **Prefix cache** shipped default-on (~45% reuse). **Semantic cache** parked Type-2; **vocab screen**
  Type-1 dead; **LoRA** untested.
- Tree clean except 3 untracked junk files. `stash@{0}` = P1 prefill-MMA partial (preserved).

## The two forward moves — do them IN THIS ORDER
### MOVE 1 (RECOMMENDED FIRST): the QTIP quality oracle — gates the only byte-cut bet
Build the offline oracle from `plans/qtip_bytecut_design_2026_05_31.md` §5.1 (it names it
`oracle_qtip_quality.py`, Colab/NumPy, CPU — in-session-safe, no GPU). **Measure:** QTIP-**from-f16**
(never requant-from-Q4_K) reconstruction quality vs Q4_K_M at ~3 bits on **code** — logit-cosine / KL /
argmax-agreement, the metric class the W4A8 quality work uses. **GO** if QTIP at 3 bits matches-or-beats
Q4_K_M quality (then, and only then, the §5.2 decode-cost oracle, then a kernel). **NO-GO** → QTIP dies
and the byte-cut axis is closed → Kill-Protocol entry. This is the single remaining sub-Q4 byte-cut
bet; the kernel is gated on this oracle clearing first (no kernel before the oracle).

### MOVE 2 (highest-leverage decode-tps unblock): the pruned-Q4K batched-verify fast-path
Draft tuning is merged + lossless but **−78% when on** because `forward_tokens_verify`
(`qwen_dense.rs` ~5006) falls back to a CPU full-vocab batched pass at ~18% acceptance. Build the
**pruned-Q4K batched-verify fast-path** so the K-wide verify GEMM uses the pruned-Q4K LM head (mirror
the decode predec/pruned path). Then re-bench paired (draft OFF vs ON) — the lever flips positive only
when verify cost drops below the acceptance payoff. GPU source lane; parity is already bit-identical
(re-verify token-identity yourself after the change). This is what turns the GO'd draft oracle into
real tps.

## The one attended-strategy item (decide before the §6 rewrite)
**Does ~50 dense stay the target?** At a measured ~31 ceiling with kernels closed, ~50 needs QTIP's full
multiplier **and** the spec unblock both landing — reachable, but no longer "high-confidence parity."
Once decided: **re-project the §3 envelope from ~31** (it's marked SUPERSEDED, not yet redrawn) and do
the **§6 / roadmap consistency pass** — the ~7 "path-to-50" lines are doubly stale (kernels closed +
anchor moved); the rewrite is unblocked the moment the goalpost is set.

## Cheap owed measurement
**Thermal-median** (`clean_bench.sh ×N`, clean room) to pin the canonical anchor at 29 vs 31. Tightening
a known number, not reopening it.

## Discipline (CLAUDE.md + bible §8.3.1)
- **Oracle-before-body**, always. **Kill Protocol:** every NO-GO records Type-1/Type-2 + the reframe +
  why it dies / its named oracle, in `reports/dead_levers.md`. Never re-test a Type-1; never resurrect on vibes.
- **Contamination:** absolute tps/GB/s/J are valid **only clean** (run `clean_room_batch.sh` Claude-quit);
  in-session, **paired relative deltas only**. CPU/NumPy oracles are immune.
- **Verify yourself:** re-run the parity/bit-identity gate in main before merging any decode/kernel lane;
  never trust an agent's "parity passed." Worktree agents: fast-forward to current HEAD at start
  (stale-base pitfall) + symlink `models/`.
- One purpose per commit; commit only the files the lever touches; no sweep commits; don't push.

## Don't re-litigate (recorded kills — `reports/dead_levers.md`)
Q3_K byte-cut, lm_head SVD + vocab screen, EAGLE-3 head, L1.1 KV working-set, FFN block-256 sparsity,
the four weight-structure kills, semantic cache (parked, real-logs only), the decode-kernel micro-opt
track, f16s-default-on, MMA on rows≤cols.

## Pointers (verified present 2026-05-31)
- Closeout: `plans/session_closeout_2026_05_31_evening.md` · Baselines: `reports/clean_room_baselines_2026_05_31.md`
- Designs: `plans/qtip_bytecut_design_2026_05_31.md`, `plans/stateful_moat_continuation_design_2026_05_31.md`
- P1 handoff: `plans/p1_prefill_mma_integration_handoff_2026_05_31.md` · Kills: `reports/dead_levers.md`
- Draft body: `crates/dismantle-core/src/stateful/usage_capture.rs`, `.../speculate/user_ngram.rs`,
  verify seam `qwen_dense.rs` ~5006 · Clean-room: `tools/bench/clean_room_batch.sh`
