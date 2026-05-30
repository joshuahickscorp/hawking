# Staging session CLOSEOUT — 2026-05-30

Executed `plans/staging_session_worklist_2026_05_30.md` end-to-end on the
CPU-scaffold lane. **10 commits landed** on `codex/maximal-spec-colab`
(interleaved with the concurrent colab/EAGLE session's `a551acc`, `8f6a70a`,
which commit under the same `Joshua Hicks` identity). Build green, lib tests
pass, working tree clean, nothing pushed.

## Commits (this session, newest → oldest)
- `ccc4e1f` feat: Q3_K + f16 predec scale-table builders (1.6+1.2 CPU halves, tested)
- `09986d9` bench: long-context harness (0.5) + spec oracle on real transcripts (3.1)
- `43df3c9` chore: gitignore venv/__pycache__/silicon-builds
- `d331c0b` docs: Tier-1 lever scaffold handoffs (fill-in-the-body guides)
- `599437b` bench: xctrace export + analysis pipeline (0.1 profiling unblock)
- `4ddc0b4` docs: dead levers (host-side/W4A8/predec-4r) + roadmap snapshot
- `fe065f3` bench: reusable §1-gated paired lever harness (paired_lever.sh)
- `41caa29` bench: map remaining 16 kernels in static_kernel_name (§1 INV2)
- `fba2ab2` bench: TCB dispatch-cost diagnostic + Stage-2 bible notes
- `d2343cc` perf: path-to-50 Q4_K predec +32.5% (2r default-on, gate+up pair, ffn_down predec)

(Plus the EAGLE venv at `.venv/`, torch 2.12.0, not committed — gitignored.)

## State by worklist item

**DONE (committed, CPU lane):**
- 4.1 path-to-50 +32.5% secured (`d2343cc` + diagnostic `fba2ab2`)
- 0.4 kernel-name map — decode traces now pass §1 INV2 (`41caa29`)
- 0.2 `tools/bench/paired_lever.sh` — reusable A/B + §1-gate harness (`fe065f3`)
- 0.1 `tools/bench/mst_export.sh` + `mst_analyze.py` — xctrace → per-kernel GPU-busy, `--cross-check` vs §1 (`599437b`); parser validated on synthetic XML
- 0.3 torch venv (`.venv/`)
- 0.5 `tools/bench/long_context_bench.sh` + 3.1 `tools/bench/spec_oracle_on_transcripts.sh` (`09986d9`)
- 1.2 `predecode_q4_k_scale_table_f16` + 1.6 `predecode_q3_k_scale_table` — CPU halves, unit-tested (`ccc4e1f`)
- 4.2 MEMORY.md 29.2KB→13.5KB (folded 8 dup index lines; no memory deleted)
- 4.3 dead-lever registry + `roadmap_2026_05_30.md` (`4ddc0b4`)
- 4.4 (safe part) gitignore junk (`43df3c9`)

**GPU-BENCH LANE (bodies written/specified; need a free M3 to parity+bench):**
- 1.2 f16-scales kernel `gemm_q4_k_v4_predec_f16s` + cache wire (CPU half done)
- 1.6 Q3_K kernel `gemm_q3_k_v4_predec` + Q3_K predec cache (CPU half done)
- 1.5 LM-head→predec WIRING (no new kernel; bit-identical) — recipe below
- 1.1 prefill MMA port (`gemm_q4k_mma_nwide` → v3w path) — full kernel source + ABI in handoff
- 1.7 simdgroup-matrix decode — XL; gate on a real 0.1 capture first

**CLOUD LANE (separate machine):** EAGLE num_blocks=2 retrain (active), AWQ-from-f16 byte-cut source.

## 1.5 LM-head→predec wiring recipe (the one lever I did NOT code)
Lowest-value Tier-1 lever (+1-2%) but highest touch-count, so deferred to keep
the hot-path file safe. Exact spots (qwen_dense.rs):
1. Struct field `lm_head_q4k_buf` decl at ~199 → add sibling `lm_head_predec_scales: Option<PinnedBuffer>`.
2. Build it where the Q4K LM head is made (~842-860): after `quantize_q4_k(&src_f32, &mut q4k_bytes)`, call `kernels::predecode_q4_k_scale_table(&q4k_bytes)` → pin → assign.
3. Struct literal sets `lm_head_q4k_buf` at THREE sites (821, 1097, 1194) — add the new field to all three (most can be `None` if they don't build the Q4K head).
4. Dispatch at ~3867 (`gemv_q4_k_m_v3_8r_pinned_tcb`) → behind gate `DISMANTLE_QWEN_LMHEAD_PREDEC`, swap to `gemv_q4_k_v4_predec_pinned_tcb` with the scales. Bit-identical; bench with `paired_lever.sh --label lmhead_predec`.

## DEFERRED: corpus purge (decision = purge, NOT executed)
You chose purge+regenerate for the ~190MB committed EAGLE corpus
(`colab/data/eagle5_corpus/*.parquet`, 10 shards). **Not done** — it collides
with active work: the concurrent session committed `8f6a70a` tuning nb02 to the
619-seq corpus, the capture watcher is growing it toward the num_blocks=2
retrain, and zero-upload Colab reads it from the repo. A true (history-shrinking)
purge also needs `git filter-repo` + force-push, which is unsafe mid-flight on
this shared branch. **Execute after the corpus→retrain cycle ships:**
`git rm -r colab/data/eagle5_corpus/` (the `*.parquet` ignore rule already
exists at .gitignore:86), then optionally a history rewrite in an attended,
single-session window.

## Next single step for the GPU lane
Take ONE Metal System Trace (`tools/bench/mst_capture.sh`), run
`mst_export.sh` + `mst_analyze.py --cross-check`, and let it rank which Q4_K
GEMV stall to attack — THEN write the highest-ranked kernel body (likely 1.1
prefill MMA for TTFT, or 1.6 Q3_K for the byte-cut). Don't guess geometry (the
`_4r` lesson). Per-lever fill-in guides: `plans/tier1_scaffold_handoffs_2026_05_30.md`.
