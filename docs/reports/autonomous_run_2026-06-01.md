# Autonomous run — 2026-06-01 — HALT (queue unpopulated) + verified harvest

**Branch:** `codex/maximal-spec-colab` · **Started:** 2026-06-01 (in-session, Claude.app live → absolute GPU benches contaminated) · **Push:** none (local only).
**Halt class:** clean halt — the delivered queue was unpopulated placeholders. No work fabricated. GPU-free / self-verifying / already-owed work banked; everything build-or-strategic STAGED below.

## The governing finding (read first)
The queue handed to this run was a **template skeleton**, not a populated list. Every concrete slot was a literal placeholder:
- Wave 1: `T1. <task> — precondition: none. done-gate: <artifact + self-test green>. T2. ...`
- Wave 2: `T#. <pre-gated build> — precondition: <its parity test exists>.`
- Clean-room: `D1. append <absolute-GPU-bench> to tools/bench/clean_room_queue.sh`

Only the STANDING RULES, the S1 synthesis description, and the FINAL-REPORT format were concrete. Per the run's own rule — *"ON ANY AMBIGUITY (missing file, unclear precondition…): HALT that task, log it, continue with independent tasks. Do not guess"* — I did **not** invent T1/T2/Wave-2 builds and auto-execute them. Two independent facts reinforce that posture:
1. **Wave-2 is empty by construction.** The run says Wave-2 runs "ONLY tasks pre-authorized here." Nothing is pre-authorized because the queue is blank ⇒ **zero source/kernel builds authorized this run.**
2. **The branch is under another agent's authorship.** `codex/maximal-spec-colab` was committed to by "Codex" at 08:21–08:40 today (a Rust cleanup/refactor series), ~30 min before this session, and carries **2 uncommitted Colab-synced "maximal spec" headbank artifacts** (see §Surfaced). Autonomously building/committing source here risks colliding with live concurrent work.

So this run executed only what is unambiguous, safe, reversible, GPU-free, self-verifying, and **already owed in writing** by the team's own handoff docs (`plans/next_session_opener_2026_05_31_evening.md`, `plans/overnight_build_queue_2026_05_31.md`), then produced the S1 ranked build-list and staged the rest.

## Per-task status
| Task | As delivered | Status | Reason |
|---|---|---|---|
| T1…Tn (Wave 1) | `<task>` placeholders | **HALT** | Slot unpopulated. Treated the team's next-session opener as the *candidate* harvest; banked only its GPU-free, self-verifying, already-owed items (below). |
| S1 (Synthesis) | concrete | **DONE** | Ranked, gate-sorted build-list produced (§Ranked build-list). Clearly-factual doc corrections: **none safely applicable** — the only stale lines (§6/roadmap "path-to-50", §3 envelope) are judgment-laden/attended (depend on the goalpost call) → STAGED, not edited. |
| T# (Wave 2 builds) | `<pre-gated build>` placeholders | **HALT** | None pre-authorized (queue blank). GPU source against the live decode path + a concurrently-owned branch. Candidates ranked + STAGED (B1–B4). |
| D1 (clean-room) | `<absolute-GPU-bench>` → `clean_room_queue.sh` | **HALT → reconciled** | Named file does not exist; capability already lives in `tools/bench/clean_room_batch.sh` (gated, Claude-quit-refusing, covers Q3 byte-cut + anchor recon + energy). Naming/gap reconciliation STAGED (§Clean-room runbook). No fabricated duplicate created. |

### Banked this run (GPU-free, self-verified)
- **QTIP quality oracle self-test — GREEN (re-verified in-session).** `python3 tools/bench/oracle_qtip_quality.py --selftest` → exit 0. All checks pass: RHT orthogonal+exact-inverse; RHT Gaussianizes (excess-kurtosis 32.06→0.21); Lloyd-Max 8-level SQNR 14.58 dB (≈14.6 expected); bracket ordering 4b<3b MSE; QTIP upper<lower @ 98 B/blk (3.06 eff bits); NumPy Q4_K_M baseline rel-RMSE 0.0781; logit cos/KL/argmax plumbing. (Optional `gguf` Q4_0 cross-check skipped — module absent, non-fatal.) This is the only in-session-verifiable gate for MOVE-1; the **decisive** QTIP verdict is `--colab` (f16 weights + GPU) and **cannot run here**.
- **Stale-test morning-to-do — RESOLVED (verified, no action needed).** The flagged `v1_1_phase5A_batched_forward_parity.rs` (now at `crates/dismantle-core/tests/`) no longer references `SpeculateMode::NGram` anywhere, and `cargo check -p dismantle-core --test v1_1_phase5A_batched_forward_parity` **compiles clean** (`Finished dev … in 4.28s`, warnings only). Today's b219556 (parity-helper dedup) fixed it. The opener/overnight "stale compile error" caveat is stale-good.

## Artifacts committed
| Hash | Line |
|---|---|
| _(this file)_ | `reports/autonomous_run_2026-06-01.md` — this run's audit trail (force-added; `reports/` is gitignored by convention). |

No source/kernel/bench files were created or modified — nothing else to commit. The QTIP self-test and the compile check changed no tracked files (verification only).

## Ranked build-list (S1) — sorted BY THE GATE EACH LEVER MUST CLEAR
A clean oracle buys a **MEASUREMENT** slot, never a BUILD slot. A BUILD slot requires an explicit authorization that this run did not have.

### MEASUREMENT-gated (owed oracle / clean-room — no build until the number lands)
- **M1 — QTIP decisive quality gate (`--colab`, f16 vs Q4_K_M on code).** In-session selftest GREEN (above); local proxy is **CAUTIONARY** (needs ~1.2-bit RHT+trellis gain to match Q4_K_M RMSE — upper edge of TCQ). Decisive verdict needs f16 weights + GPU on Colab. **Blocked-on-Colab; cannot run in-session.** No QTIP kernel until this clears (oracle-before-body). The single remaining sub-Q4 byte-cut bet.
- **M2 — Clean-room absolute batch** (anchor thermal-median ~29 vs ~31; energy J/tok). Lives in `tools/bench/clean_room_batch.sh`; user runs Claude-quit (§Clean-room runbook). Gap: a thermal-median *N-repeat* (the script itself flags Section B as a single run).
- **M3 — Semantic cache re-gate on real file-interleaved session logs.** Oracle built; Type-2 PARKED. Re-run on real transcripts, not the consecutive-edit git proxy.
- **M4 — imatrix mixed-prec + small-dense-draft accept oracles → data.** Both oracles built (`oracle_imatrix_mixprec.py` 69e94a6, `draft_accept_oracle.py` ecf683b); NEEDS-MEASUREMENT.
- **M5 — KV-working-set at 16–32K.** Type-2, **blocked on a cheaper attention-capture path first** (CPU capture hit the time governor at >586 tok). Build the cheaper instrument (GPU `scores` spill / cumulative-curve-only) before any long-ctx test.

### BUILD-gated (parity known/exists — require explicit authorization; NOT run this run)
- **B1 — Pruned-Q4K batched-verify fast-path (MOVE 2). [#1 leverage]** Turns the GO'd draft-tuning lever from **−78% (on)** to positive by making `forward_tokens_verify` (`qwen_dense.rs` ~5006) use the pruned-Q4K LM head instead of a CPU full-vocab pass. Parity is already bit-identical (re-verify token-identity yourself after the change). GPU source lane. Highest-leverage decode-tps unblock.
- **B2 — Prefix-cache default-on hardening + [HIGH] disk-tier fix.** Two items: (a) ship a non-None `PrefixCacheBudget` default (RAM cache is currently unbounded → OOM risk; `evict_to` is correct but unarmed) + `debug_assert` on cache-shape collision; (b) **[HIGH correctness] disk-tier prefill cache stores STALE KV on the TCB path** (`qwen_dense.rs:1519` `cache.store` runs before `mirror_arena_kv_into_self`/RAM mirror → silent corruption on a later-process disk hit with `DISMANTLE_PREFIX_CACHE_DIR`+TCB). Pre-existing, not from any recent haul. Gate: a disk hit reproduces no-cache output bit-identical.
- **B3 — P1 prefill-MMA → TTFT (silicon #8 port).** GO but deferred: stale worktree base + MMA is v3w-layout while the shipped batched path is predec (dormant where it wins). Needs a predec-MMA twin + fresh worktree base. Recipe: `plans/p1_prefill_mma_integration_handoff_2026_05_31.md`. Gate: bit-identical prefill + cold-prompt TTFT win.
- **B4 — P2 n-gram batched-verify-with-logits.** Sub-gate τ=1.43, but batched-verify could bank a copy-heavy-code win. Gate: bit-identical greedy + dec_tps win.

### DEAD — do not build, do not re-test (Type-1 unless noted; `reports/dead_levers.md`)
Q3_K sub-Q4 byte-cut (clean-confirmed compute-bound; QTIP is the Type-2 reframe = M1); lm_head SVD + usage-frequency vocab screen; data-free **and** data-aware low-rank codec; decode-kernel micro-opt track (A5 uint4, A6 occupancy, A7 MMA-decode, A10 layout — GEMV at the Apple-GPU memory-model optimum); MMA on rows≤cols (occupancy; tall-shape variant is GO-but-dormant); FFN block-256 sparsity; the four weight-structure kills; **f16s default-on** (SETTLED opt-in: 8.2% category-correlated corpus drift).

## STAGED decisions awaiting you (I did NOT make these)
1. **Authorize the queue.** The run's queue was blank. Confirm whether the intended Wave-1/Wave-2 is the candidate harvest above (from your own opener) — or paste the real queue. **Nothing was built pending this.**
2. **Goalpost (the standing attended item): does ~50 dense stay the target?** At a measured ~31 ceiling with decode kernels closed, ~50 needs **both** QTIP's full multiplier (M1) **and** the spec unblock (B1) to land — reachable, no longer "high-confidence parity."
3. **§3 envelope re-projection from ~31** (currently marked SUPERSEDED, not redrawn) + **§6 / roadmap "path-to-50" consistency pass** (~7 doubly-stale lines: kernels closed + anchor moved). Unblocked the moment #2 is decided. Judgment-laden → not auto-edited.
4. **Default-on flips** (each clean/exact, just needs the safety wiring = B2): prefix-cache (after byte cap + disk fix), and a broader drift sweep is the only thing between f16s and a constrained-workload default — but f16s default-on is already SETTLED NO-GO for open-ended gen.

## Surfaced — uncommitted working-tree changes I did NOT touch
Two **Colab-synced** files are modified in the working tree (not by me; left exactly as found):
- `docs/archive/.../headbank_500u_v2/headbank_manifest.json` (+74 lines): enriched with `awq_scales`, `gguf_name`, `head_path`, `head_sha256`, `hf_id`, `runtime_profile`, `rust_serving_verified: true`, and richer metrics (`depth1_accept_rate`, `tau`, `policy_kind`) for q05b/q3b/q7b. Paths point at `/content/drive/MyDrive/...`.
- `docs/archive/.../maximal_spec_500u/eval/q1p5/.../frontier.json` (+379 lines).

These look like outputs of the "maximal spec" Colab head-training run that names this branch. **Your call to commit or discard** — I did not stage, commit, or revert them (not in any manifest scope; possibly mid-flight Codex work).

## Clean-room runbook (the "clean_room_queue.sh" the run asked for — reconciled)
The file `tools/bench/clean_room_queue.sh` **does not exist**; the capability is `tools/bench/clean_room_batch.sh` (committed, gated to **refuse to run while Claude / a `claude` CLI / `MASTER_LOOP` / `slm` is live**). It already runs the three owed absolute-metric sections. **Run it with the app fully closed:**

```sh
# 1. Quit the Claude Code desktop app (Cmd+Q) and any `claude` CLI / loop sessions.
# 2. Quit slm / any GPU-heavy process. Open a fresh Terminal.
cd /Users/scammermike/Downloads/dismantle

# Safe anytime (even with Claude open) — prints the plan + contamination gates, runs nothing:
./tools/bench/clean_room_batch.sh --gates-only

# The real run (Claude QUIT):
./tools/bench/clean_room_batch.sh
#   SECTION A — Q3 byte-cut microbench  → GB/s vs 150 peak; ~50% = GO repack, ~30% = NO-GO (QTIP only). [Q3 already Type-1 dead; this re-confirms]
#   SECTION B — clean decode-tps anchor → prints vs ~39 (old) and ~31 (recent); pins which anchor is real.
#   SECTION C — energy J/token          → needs:  brew install macmon   (else SECTION C skips).

# Morning one-liner for the energy + f16s-compare leg specifically:
brew install macmon && tools/bench/measure_joules.sh --tokens 256 --f16s
```
Parse targets the morning report wants (already echoed by the script as `SECTION x VERDICT:` lines): Section B prints `clean tps tracks the ~31 / ~39 anchor`; Section C prints `joules/token`. **Owed gap to add when convenient:** a thermal-median *N-repeat* wrapper around Section B (the script notes it does a single run; pin 29 vs 31 with a median).

## Factual items verified this run (no action needed)
- Working tree clean except the 2 Colab JSONs above; the evening closeout's "3 untracked junk files" (`on_smoke.err`, etc.) are gone.
- Stale parity test compiles clean at HEAD (resolved by b219556) — see Banked.
- QTIP oracle self-test green — see Banked.
- No `slm` and no separate `codex`/`claude`-agent process is live right now; Claude.app + this CLI session are the only relevant processes (absolute GPU numbers in-session remain contaminated — none were run).

## Clean-room results (user-run 2026-06-01, Claude quit — closes M2, re-confirms the kills)
Ran `clean_room_batch.sh` + `measure_joules.sh --f16s` with the app quit (gates passed; one caveat: pre-flight WARNed WindowServer at ~49% CPU — matters only to the energy leg, see C).

- **A — Q3 byte-cut: NO-GO, clean-RE-confirmed.** f32-predec-Q3 best-shape **36.3 GB/s = 24.2% of 150 peak** (prior clean run: 33.3 GB/s / 22%). Q3_K is **38–46% slower in µs** than Q4_predec on all 3 shapes despite ~half the bytes → compute/residual-bound, not BW-bound. The recorded Type-1 kill holds; **QTIP stays the only byte-cut path.**
- **B — decode anchor: ~31, RESOLVED clean.** 31.13 dec_tps (256 tok, greedy temp=0 seed=0, locked fast-path) = **+0.4% vs the ~31 recent anchor, −20.2% vs the dead ~39.** Corroborated by 31.66 / 31.59 in the energy leg. The ~39 envelope is dead; canonical anchor ≈ **31** (tightening toward 31, not 29). tps is robust to the WindowServer WARN (GPU-bound + nice/taskpolicy-isolated). The §3.0 "~39 vs ~31 unreconciled" open item can close (factual); the envelope **re-projection** stays your call (#2/#3).
- **C — energy: NOISY, NOT a clean replacement for the 0.17 floor.** Readings: 0.2328 J/tok (Section C) and 0.2586 / 0.2549 (baseline / f16s) in the --f16s leg. All ran with WindowServer at ~49% CPU (it drives SoC GPU+CPU power that macmon attributes package-wide) and the --f16s leg ran **during a Homebrew auto-update** → package power inflated to 7.4–8.8 W (vs the prior quiet 4.87 W) and GPU power to 5.0–5.3 W (vs prior 3.73 W). J/tok scales directly with that power, so **0.23–0.26 is an inflated reading; the 0.17 J/tok @ 3.73 W floor stands pending a settled re-run** (screen idle, no brew). **Owed: re-run `measure_joules.sh` on a quiet machine.**
- **Bonus (paired, contamination-robust): f16s = +9.3% tps for −1.4% J/tok.** Same-run A/B (31.59→34.52 tps, 0.2586→0.2549 J/tok) → the relative delta survives the noise: the A6.5 f16s lever is **faster AND marginally more energy-efficient** (finishes sooner, idles). Supports the opt-in.

## Honest one-line state
**Moved:** clean-room run (user, app quit) **resolved the decode anchor to ~31** (the ~39 is dead) and **re-confirmed Q3 byte-cut NO-GO** (24.2% peak → QTIP-only); QTIP oracle green; stale-test caveat resolved; f16s paired +9.3% tps / −1.4% J/tok. **Blocked:** queue was never populated (no builds authorized); QTIP decisive verdict is Colab-only; **energy reading came back noisy** (WindowServer + brew) → the 0.17 floor needs a quiet re-run. **Needs your call:** authorize/paste the real queue, the goalpost + §3/§6 re-projection, and the 2 uncommitted Colab JSONs.
