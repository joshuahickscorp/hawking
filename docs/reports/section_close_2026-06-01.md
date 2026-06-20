# Section-close run — 2026-06-01

Final run to close this section of dismantle. Three phases: P1 parallel
autonomous construction → P2 build the two Colab notebooks + HALT for the user
→ P3 resume autonomous (act on verdicts, consolidate, reversible wipe).

Authoritative inline decisions (confirmed at run start):
- **CONFIRM-1 = DELETE** silicon-builds/ (629 MB, untracked) in P3, only after its
  kills are in the canonical ledger AND silicon #8/#13 are re-derived parity-green.
- **CONFIRM-2 = REWRITE** CLAUDE.md to current reality in P3.
- **CONFIRM-3 = STAGE D-2** (no Llama/Gemma GGUF on disk; only Qwen2.5 + deepseek).

Branch: `codex/maximal-spec-colab`. Base at run start: `b3ec26a`. All commits below
are local (unpushed), authored `Joshua Hicks <joshuahicksboba@gmail.com>`, no AI
attribution. The two modified `docs/archive/**/*.json` (Codex Colab artifacts) were
left untouched throughout (PROTECTED).

---

## PHASE 1 — construction state (done-gate: every task GREEN or STAGED)

| Task | Status | Commit | Gate |
|------|--------|--------|------|
| P1-B Qwen zero-copy loader | ✅ GREEN | `a08fa17` | bit-identical A/B (copy vs no-copy), 5 prompts × 8 greedy tokens |
| P1-F server HTTP/SSE + error codes | ✅ GREEN | `acf1fd7` | 11 new + 9 existing dismantle-serve tests; re-verified in main |
| P1-C 3-arm clean-room queue + parity confirm | ✅ GREEN | `54a5997` | 6/6 user_draft bit-identity tests pass; tps deferred to clean-room |
| P1-A Q4_K simdgroup-MMA prefill twin | ✅ GREEN | `14434f5` | parity 4/4 (atol 1e-3 + rtol 1e-4) + token-identity (MMA on==off) |
| P1-E fallible alloc + corrupt-GGUF tests | ✅ GREEN | `c1e18d6` | load_robustness 6/6; model still loads bit-identical |
| P1-D1 ArchReader config-core dedup | ✅ GREEN | `4f0c16c` | ArchReader 3/3 + qwen real-load BYTE-IDENTICAL; lib 94/9/5 |
| P1-D2 2nd-arch fast-decode + parity | ⏸ STAGED | — | CONFIRM-3: no Llama/Gemma GGUF on disk; user action (see below) |

Integrated state verified green: `cargo build --release --workspace` + `cargo test
--workspace --lib` (dismantle-bench 5 / dismantle-core 94 / dismantle-serve 9, 0 failed),
plus per-task integration/parity tests.

### Per-task detail

**P1-B — Qwen zero-copy mmap loader (`a08fa17`).** `qwen_dense.rs` weight loading
switched from `new_buffer_with_bytes` (full ~1.93 GB copy of the mmap into a
StorageModeShared buffer) to `new_buffer_no_copy` over the borrowed GGUF mmap,
mirroring `deepseek_v2.rs:754`. Realizes silicon #13: −1.9 GB RSS, −324 ms TTFT.
Sound because `self.gguf` owns the mmap for the model lifetime. **Verified
bit-identical** by a true A/B: built copy-path and no-copy-path binaries, ran
`batch-hash` greedy 8-token on Qwen2.5-3B across 5 prompts — blake3 hashes match
exactly. Highest-value cheapest win.

**P1-F — OpenAI server HTTP/SSE integration test + structured error codes
(`acf1fd7`).** Added `crates/dismantle-serve/tests/http_integration.rs` (11 tests)
driving the axum Router in-process via `tower::ServiceExt::oneshot` against a stub
engine — fast, hermetic, no model. Covers chat/completions + completions success
(SSE + JSON) and structured-error paths; added an OpenAI-shape `ApiError`
(`{error:{message,type,code}}`). Built by a worktree agent and **re-verified in
main** (build + `cargo test -p dismantle-serve` green). Honest note: SSE
mid-stream engine failures are best-effort-dropped (status already committed) —
correct per OpenAI's model; a richer in-band SSE error frame is a follow-up.

**P1-C — clean-room queue + parity confirm (`54a5997`).** Confirmed C's 6
`user_draft_parity_e2e` bit-identity tests pass 6/6 (293 s, Qwen2.5-3B). The
absolute 3-arm tps verdict is contamination-sensitive, so DEFERRED to clean-room:
added `tools/bench/clean_room_queue.sh` (section-close deferred-absolute-bench
runner: user-draft 3-arm + prefill-MMA TTFT, with the Claude-quit preamble),
complementing `clean_room_batch.sh`. NOTE: the prompt's `clean_room_queue.sh`
did not exist; created it (the established sibling is `clean_room_batch.sh`).

**P1-A — Q4_K simdgroup-MMA prefill twin (`14434f5`).** Re-derived off current
HEAD from `stash@{0}`'s two `quant.metal` kernels (NOT the stale `c9b1c07`
branch). Wired `gemm_q4_k_m_batched_v3w_mma(_predec)` into the batched prefill
`batched_proj!` macro, gated `DISMANTLE_QWEN_Q4K_MMA` and shape-gated `rows>cols`
so it only swaps on ffn_gate/ffn_up (11008×2048) — the measured rows>cols win;
q/k/v/o (square) and ffn_down (wide) stay on tuned v3w/predec (MMA loses there:
Type-1 occupancy). Added 2 wrappers (32-thread/1-simdgroup TG, fixed 576-f32
shmem), prewarm entries, and `q4k_batched_mma_parity.rs`. **Parity 4/4** with a
combined tol atol 1e-3 + rtol 1e-4 (the reduction-reorder fp noise on ~1e3-mag
outputs is ~3e-6 relative — below the fp32 floor that makes pure atol 1e-3
unsatisfiable there; rtol stays ~100× tighter than any real bug). **Token-identity
green** (MMA-on == MMA-off, 8 greedy tokens × 5 prompts, BATCH_PREFILL=1). Perf
(prefill-TTFT vs predec-ON) DEFERRED to clean-room.

**P1-E — fallible Metal alloc + corrupt-GGUF tests (`c1e18d6`).**
`new_buffer_checked` / `new_buffer_with_bytes_checked` reject (graceful `Err`) a
request exceeding the device's recommended working-set ceiling — the dominant OOM
cause (model too large for device). metal-rs 0.29's `new_buffer` ASSERTS non-null
and aborts the instant Metal returns nil, so the guard is an up-front size
pre-check, not a post-check; `max_buffer_length()` is unusable (0.29 hardcodes it
to 1 GB on macOS≥12). Wired the two largest model-driven load allocs (embed +
lm-head) through it. `tests/load_robustness.rs` 6/6: GgufFile::open errs (not
panics) on empty/bad-magic/bad-version/truncated-header/truncated-real-GGUF, plus
the oversize-alloc Err test. Residual: a transient OOM *within* the ceiling still
aborts inside metal-rs — fully fallible alloc needs the objc2-metal binding
(follow-up). Scoped: did NOT mass-rewrite the ~724 unwraps.

**P1-D1 — ArchReader config-core dedup (`4f0c16c`).** Scope correction: the
prompt's premise ("merge 4 near-identical readers into one ArchConfig struct")
does NOT hold at HEAD — qwen/llama/gemma2/phi3 share only a ~5-field required-read
core; they diverge in head_dim policy, vocab fallback chains, per-arch defaults
(rope 1e6/500k/10k, eps 1e-6/1e-5, ctx 32768/8192/4096) and per-arch extra struct
fields (llama rope_scaling+arch, gemma2 soft-caps+scales, phi3/llama
sliding-window). So this is a shared READER helper (`model::arch_config::ArchReader`),
not a shared struct: each loader keeps its `*Config` + vocab + extras byte-for-byte
and routes only the duplicated core through it. **Verified**: ArchReader unit tests
3/3; qwen real-load greedy output BYTE-IDENTICAL to the pre-refactor baseline.
Caveat: llama/gemma2/phi3 lack on-disk GGUFs, so those three are verified by the
helper tests + mechanical equivalence + the smoke harness, not a full real-model
load. mixtral/deepseek (MoE, heavier extras) left for a follow-up.

**P1-D2 — STAGED (user action).** Generalizing Qwen's fast-decode core
(predec / vocab-prune / Q4K-LM-head) to a 2nd architecture and running greedy
parity vs llama.cpp requires a small Llama or Gemma GGUF; none on disk (only
Qwen2.5 0.5/1.5/3/7B + deepseek-v2-lite). **Your action:** drop a small Llama or
Gemma `.gguf` in `models/`, then the 2nd-arch fast-decode generalization + parity
can run autonomously. D-1 (ArchReader) already lowered the cost of adding an arch.

### Orchestration note (why the approach changed mid-run)
The Agent tool's `isolation: worktree` bases new worktrees off **`main`
(`22dd6f4`)**, which is ~5 commits behind the `codex/maximal-spec-colab` HEAD —
`qwen_dense.rs` differs by 3014 lines there, and `gemma2/llama/phi3.rs` did not yet
exist. The first fan-out (A/D1/F agents) therefore built on a stale base: **F was
salvageable** (dismantle-serve unchanged main→HEAD, clean cherry-pick + re-verify),
but **A and D1 were invalid** (stale base; D1 even missed 3 readers). Pivoted to
**sequential in-main execution** for A/D1/E with parity gates after each — slower
wall-clock but eliminates the base/merge hazard, consistent with the
correctness-over-speed standing rule. The MMA `stash@{0}` + the existing
`plans/prefill_mma_build_plan.md` made A a clean re-derivation off HEAD.

### Followups surfaced during Phase 1 (out of scope, logged here)
- **Flaky pre-existing test:** `speculate::eagle5::tests::trained_loader_reads_synthetic_head`
  SIGABRTs intermittently (`ptr::copy_nonoverlapping requires aligned`) — confirmed
  present on clean HEAD, unrelated to any Phase-1 change. Latent unaligned read in
  the eagle5 trained-head loader; the debug-mode precondition check catches it
  alignment-dependently. Triage separately.
- **Fully fallible Metal alloc** needs objc2-metal (metal-rs 0.29 aborts on nil).
- **mixtral/deepseek** config readers not yet on ArchReader.
- **Stale agent worktrees** (`.claude/worktrees/agent-a4b4747… / -af965a7… / -aed0334…`)
  on stale-main branches — prune in P3 wipe (not the PROTECTED locked worktrees).

---

## PHASE 2 — Colab notebooks + handoff (done-gate: both turnkey + handoff written → HALT)

Both gates are turnkey "Run all" notebooks (self-contained: build llama.cpp,
fetch f16/q8 Qwen2.5-3B + code corpora, write a machine-readable verdict JSON,
`files.download()`); every leg is guarded (can't-run → `null` → NEEDS-MEASUREMENT,
never fabricated). I CANNOT run L4/A100 notebooks — the user runs them.

| Gate | Notebook | Status | Verdict JSON |
|------|----------|--------|--------------|
| P2-G QTIP 3-bit trellis vs Q4_K_M | `colab/03_qtip_3bit.ipynb` (hardened `4fe9e3b`) | ✅ CONFIRMED turnkey | `qtip_3bit_results.json` |
| P2-H imatrix mixed-prec vs Q4_K_M | `colab/04_imatrix_mixprec_gate.ipynb` (BUILT this run) | ✅ BUILT + py_compile/JSON-valid | `imatrix_mixprec_results.json` |
| P2-H AWQ-smoothing W4A8 (optional, wired downstream) | `colab/awq_w4a8_validate.py` (existing) | ✅ CONFIRMED runnable | `awq_results.json` |

- **P2-G CONFIRMED.** `reports/qtip_colab_readiness.md` documents the hardened
  notebook cell-by-cell: real Cornell-RelaxML QTIP RHT+trellis K=3 (cell 5,
  guarded bracket fallback that won't fabricate), Hessians on code calib,
  `RUN_REAL_QTIP = vram>=24` (cell 1), and all three legs → `qtip_3bit_results.json`
  (cells 6/7/8/9). Verdict GO/NO-GO/NEEDS-MEASUREMENT mapping was dry-run-verified
  ×3. JSON valid, 11 cells. Matches every P2-G requirement.
- **P2-H BUILT.** The imatrix oracle's decisive logit-cos/KL/argmax leg did not
  exist as a notebook (`01_bytecut` runs a real imatrix but only the uniform-Q3
  *PPL* leg). Built `colab/04_imatrix_mixprec_gate.ipynb` implementing the oracle's
  §"DECISIVE (Colab) gate" runbook: llama-imatrix on the near-lossless source →
  uniform Q4_K_M (gold) + imatrix mixed-prec GGUF (attn+ffn_gate@Q4_K,
  ffn_down+ffn_up@Q3_K via `--tensor-type`; falls back to uniform Q3_K_M+imatrix
  if the build lacks overrides, logged) → recon-budget + PPL + the decisive logit
  leg (llama-cpp-python: cos/KL/argmax vs the q8≈f16 reference). py_compile OK,
  11 cells, valid JSON. Stays in GGUF → a GO needs loader byte-accounting only,
  no new kernel. The AWQ flavour (the wired `DISMANTLE_QWEN_AWQ` downstream) is
  the existing `awq_w4a8_validate.py` (py_compile OK), referenced as the optional
  3rd run.
- **Artifact location:** `colab/` and `reports/` are gitignored+untracked
  (`c8bd80d` removed dev tooling from the shipped repo), so the notebook +
  `COLAB_HANDOFF.md` + this report live ON DISK (not committed) — the established
  dev-artifact workflow. Run the notebook by uploading it to Colab, or push the
  branch and use the `colab.research.google.com/github/...` launcher.

Full run commands + PASS criteria + the return-data schema Phase 3 parses:
**`reports/COLAB_HANDOFF.md`**. Drop verdict JSONs in `reports/colab_verdicts/`.

### Phase-2 execution update (user-run on an RTX PRO 6000 Blackwell, 102 GB)
The two gates were consolidated into ONE master notebook
`colab/05_combined_quality_gates.ipynb` (supersedes 01/03/04; shared setup once;
imatrix gate first, then QTIP; each independently guarded; QTIP real codec
Drive-resumable + opt-in via `ALLOW_FRESH_QTIP_CODEC`). Fixes landed across
pushes `a093c2c..22b9a82`: bf16-safe weight load (numpy can't read Qwen's bf16),
glog/primefac install + captured QTIP subprocess errors, SDPA fallback (no
Blackwell sm_120 flash-attn wheel), LEG1 GGUF↔HF transpose auto-pick,
CMAKE_cloud-GPU_ARCHITECTURES=native, and `--allow-requantize` on the q8-source
quantizes. All colab/ artifacts pushed to origin (colab/ + reports/ otherwise
gitignored). No prebuilt Linux-cloud-GPU llama.cpp exists (only win-cloud-gpu / linux
cpu+vulkan); the ~10-15 min cloud-GPU build is unavoidable from source (Vulkan
prebuilt is a possible instant-GPU swap if re-running — investigated, not wired).

**VERDICTS RETURNED** — persisted at `reports/colab_verdicts/{qtip_3bit_results.json, imatrix_mixprec_results.json}`:
- **QTIP 3-bit = NEEDS-MEASUREMENT, leaning NO-GO.** LEG1 transpose fix verified
  (q4k_rmse sane ~0.07 Q4_K / ~0.018 Q6_K). Corrected bracket bits_needed =
  [+1.37, +0.44] — both positive → even best-case QTIP-3bit is ~0.44 bits short
  of Q4_K_M weight quality (agrees with proxy ~1.20). Decisive real codec NOT run
  (`ALLOW_FRESH_QTIP_CODEC=False`, deliberate CU save); bracket alone can't record
  a kill (Type-2).
- **imatrix mixed-prec = NO-GO (Type-1).** The Q4/Q3 `--tensor-type` split
  wouldn't take in this llama.cpp build → tested uniform Q3_K_M+imatrix: 15%
  smaller (1.66 vs 1.96 GiB) but worse on every quality axis — PPL 4.68 > 4.59;
  logit cos 0.983 < 0.993, KL 0.053 > 0.022, argmax 0.911 < 0.922. (True ~3.82-bit
  mixed wasn't logit-tested, but the local weight-RMSE oracle already showed it
  trails Q4_K on 7/7 tensors → axis verdict sound.)

**Net: the entire sub-Q4 byte-cut axis (trellis + weight-mixing) has NO live bet.
No GOs → no default flips. Phase 2 DONE.**

**HALT / HANDOFF.** Session ended here awaiting the user's go-ahead for the full
Phase-3 finish (includes the irreversible-ish wipe).

## PHASE 3 — verdicts, consolidation, wipe  (✅ COMPLETE 2026-06-01)

Resumed autonomously on the user's "go" (with the single irreversible-step
confirmation honored — see P3-WIPE Tier 3). All four sub-steps landed.

### P3-ACT-ON-VERDICTS — both recorded as kills in the canonical ledger; NO default flips
- **imatrix mixed-prec → Type-1 kill** recorded in `reports/dead_levers.md`
  ("imatrix mixed-precision" entry): uniform-Q3 requant loses to Q4_K on PPL
  (4.68 > 4.59) + logits (cos 0.983 < 0.993, KL 0.053 > 0.022, argmax 0.911 <
  0.922); the true mixed Q4/Q3 split is bounded by the weight-RMSE oracle (trails
  Q4_K 7/7) → reframe dies → Type-1.
- **QTIP 3-bit → leaning NO-GO (Type-2 open)** recorded in `reports/dead_levers.md`
  ("QTIP 3-bit trellis" entry): measured weight bracket bits_needed [+1.37, +0.44]
  both positive; the decisive Cornell-RelaxML codec was deliberately NOT run
  (`ALLOW_FRESH_QTIP_CODEC=False`), so per the Kill Protocol the bracket alone
  cannot *record* a decisive kill — named oracle in-hand (re-run the master
  notebook with the flag True, ~20–40 min) for a Type-1 close.
- **No GO cleared any parity/bit-id gate** → no config defaults changed.
  `DISMANTLE_QWEN_AWQ` stays as-is; W4A8 still HELD at 1.115×. The entire sub-Q4
  byte-cut axis (trellis + weight-mixing) has no live bet — AWQ-from-f16 is the
  only surviving smart-quant path (requant-from-quantized forms are dead).

### P3-CONSOLIDATE — keepers written
- **`reports/dead_levers.md`** is now THE one canonical kill-ledger (400 lines):
  merged the bible/dead-lever kills + the 16-solution silicon audit (transcribed
  from `silicon-builds/*/VERDICT.md` + `SUMMARY.md` before that untracked tree was
  deleted) + the 2 Colab verdicts. Header re-anointed it canonical; cross-refs added.
- **`README.md`** rewritten to honest current state: dense+MoE engine (7 families,
  Qwen2.5-3B primary), ~31 dec_tps clean-room anchor + ceiling note, the moat
  (prefix-cache reuse + spec-on-code + low-RAM zero-copy #13), and the verdict
  outcomes. **`ARCHITECTURE.md`** corrected (the stale "MoE-only / no dense path"
  framing → dense + MoE both first-class; shared `ArchReader`).
- **`CLAUDE.md`** rewritten to reality (CONFIRM-2): retired `tools/haul/` runner →
  `tools/bench/*` + `crates/dismantle-core/tests/` parity; kill-protocol pointer →
  `plans/bible_archive.md` §8.3.1 + `reports/dead_levers.md`; corrected test counts
  (15 → ~94 core / 9 serve / 5 bench); added bench-contamination / clean-room discipline.
- Roadmap/envelope reconciliation: the ~31 anchor is already canonical in
  `plans/bible_active.md` §3.0; the keep-50 goalpost stays STAGED. Not re-litigated.

### P3-COMMIT-STATE — recovery point `e3ac8c6` (BEFORE any deletion)
`section close: ship README/ARCHITECTURE + canonical kill-ledger + CLAUDE.md to
reality`. Authored `Joshua Hicks <joshuahicksboba@gmail.com>`, no AI attribution,
local (unpushed). 6 files: README.md, ARCHITECTURE.md, CLAUDE.md (tracked-modified)
+ reports/dead_levers.md + reports/colab_verdicts/{imatrix,qtip}_*.json (force-added
out of gitignored `reports/` so the silicon kills survive the tree wipe). The 2
PROTECTED Codex `docs/archive/**/*.json` were verified UNSTAGED (still `M`).

### P3-WIPE — manifest printed first; executed in 3 tiers
- **Tier 1 — SAFE-AUTO (tracked):** **NONE** (honest empty tier). Every candidate
  evaluated + rejected: the `throughput_bible_2026_05_30.md` redirect stub the plan
  named is **load-bearing** (16 tracked inbound refs — deleting it dangles all 16);
  the session-logs/handoffs/roadmaps are a cross-referenced curated audit trail; no
  `.bak/.orig/~` orphans exist; `tests/correctness/.keep` is intentional. Same
  discipline as the dead-code audit + Kill Protocol — do not manufacture removals.
- **Tier 2 — PRE-AUTHORIZED worktree prune (~3.5 GB, reversible):** removed
  `.claude/worktrees/agent-{a4b47476 (714M), af965a7b (1.7G), aed03341 (1.1G)}`
  (`git worktree remove --force` after unlock; branch refs survive → re-addable).
  The 4 PROTECTED locked worktrees (c9b1c07/521ae73/9e03270/4a684e1) + `/tmp`
  amx-spike + all 6 stashes confirmed intact.
- **Tier 3 — APPROVAL-GATED irreversible:** **`silicon-builds/` (629 MB, untracked)
  DELETED** after the user's explicit one-time confirm ("Delete it"). Precondition
  verified MET at delete time: the 16 kills present in `reports/dead_levers.md` @
  HEAD; #8 MMA (`14434f5`) + #13 zero-copy (`a08fa17`) shipped parity-green in main.
  Post-delete: 5 GGUFs + 6 stashes + the 2 Codex JSONs all intact.

### Before / after
| Metric | Run start | After Phase 3 |
|---|---:|---:|
| Repo disk (incl. untracked, excl. /tmp wt) | ~33 GB | **~29 GB** (−4.1 GB) |
| `.claude/worktrees/` | 7.4 GB (7 worktrees) | **3.9 GB** (4 protected locked) |
| `silicon-builds/` | 629 MB (untracked) | **0** (deleted) |
| Tracked files | 455 | **458** (+ledger +2 verdict JSONs; 0 removed) |
| Tracked `plans/` | 29 | 29 (unchanged — curated, kept) |
| Kill-ledger | scattered (dead_levers + 15 VERDICT.md + 2 JSON) | **1 canonical, 400 lines** |

### Per-task hashes (full section)
| Task | Commit | State |
|------|--------|-------|
| P1-B zero-copy loader | `a08fa17` | shipped, bit-identical |
| P1-F server HTTP/SSE + errors | `acf1fd7` | shipped, 11+9 tests |
| P1-C clean-room 3-arm + parity | `54a5997` | 6/6 parity; tps deferred → clean_room_queue.sh |
| P1-A Q4_K MMA prefill twin | `14434f5` | parity 4/4 + token-identity; TTFT deferred |
| P1-E fallible alloc + corrupt-GGUF | `c1e18d6` | load_robustness 6/6 |
| P1-D1 ArchReader dedup | `4f0c16c` | 3/3 + qwen byte-identical |
| P1-D2 2nd-arch fast-decode | — | STAGED (no Llama/Gemma GGUF on disk) |
| Phase-2 master Colab gates | `22b9a82` | both verdicts returned (NO-GO / leaning) |
| P3 keepers (recovery point) | `e3ac8c6` | README+ARCH+CLAUDE+ledger+verdict JSONs |

### Kill-ledger cross-check (machine-verified)
All 16 silicon solutions (#1–#16, the 15 crate dirs incl. dispatch=#4+#9 and
mixedprec=#16+#17) present; both Colab verdicts (imatrix, QTIP) present; 28 `🪦`
bible/dead-lever + Colab entries intact. ✓

### Deferred clean-room bench queue (`tools/bench/clean_room_queue.sh`, P1-C)
Run by the user with Claude fully quit (absolute tps/TTFT inflate ~4–5× in-session):
(1) `user_draft_3arm_bench.sh` — propose-first vs bonus-first vs plain user-ngram
draft tps; (2) `prefill_mma_ttft_bench.sh` — Q4_K MMA prefill-TTFT MMA-on vs -off
(both predec-ON; guarded, SKIPs if the binary lacks `DISMANTLE_QWEN_Q4K_MMA`).
`--gates-only` prints the plan safely with Claude open.

### Followups surfaced (out of scope — logged, not fixed)
- `docs/autotune.md` still references the retired `tools/haul` path (single stale
  doc line; non-blocking).
- Flaky pre-existing `eagle5::tests::trained_loader_reads_synthetic_head` SIGABRT
  (unaligned read; present on clean HEAD, unrelated to this section).
- Fully-fallible Metal alloc needs objc2-metal; mixtral/deepseek not yet on ArchReader.

---

## ✅ SECTION CLOSED

Verdicts recorded (imatrix Type-1, QTIP leaning-NO-GO — sub-Q4 byte-cut axis has no
live bet, no default flips); keepers committed (`e3ac8c6`, recovery point); CLAUDE.md
rewritten to reality; tree wiped per confirms (3 stale worktrees + the 629 MB
silicon-builds/ deleted with explicit approval; ~4.1 GB reclaimed; nothing tracked
or protected lost). The one canonical kill-ledger holds all 16 silicon kills + the
2 Colab verdicts. **What remains is the user's: P1-D2** (drop a small Llama/Gemma
GGUF in `models/` → 2nd-arch fast-decode + parity can run) and the **deferred
clean-room bench queue** (absolute tps/TTFT, Claude quit). Decode tps stays at the
accepted ~31 anchor — ceiling proven, not on this checklist. Commits are local;
push is the user's call.
