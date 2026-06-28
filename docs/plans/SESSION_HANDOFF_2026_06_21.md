# Session Handoff — 2026-06-21 (Sonnet → h)

Protocol: `docs/plans/h_autoverify_protocol.md`

## ⚠️ Priority 0 — PARITY BUG (fix before anything else)

**Golden Rule 1 is violated.** The EH parity property gate (`event_horizon_parity_prop`) fails
6/20 pseudo-random prompts. EH-ON output diverges from EH-OFF (greedy reference) at the final
token of repeated-suffix sequences. Example: `[77, 30, 77, 30, ...]` → EH-ON emits an extra `77`
where EH-OFF emits `30`.

**Diagnosis (from Wave 3 agent):** boundary condition in SuffixArray or UserNgram proposer —
the proposer is extending a match across a position boundary the reference path does not cross.
The `'ud_loop` EH-ON path routes through `Verifier::verify_line()`, so if the proposer over-proposes
at a boundary, the verifier may accept a draft that matches a stale KV position.

Failing prompt indices: 6, 7, 12, 15, 17, 19 (seeded LCG, deterministic — re-runnable).
Test ran 607s with weights present at `models/qwen2.5-3b-instruct-q4_k_m.gguf`.

**Fix target:** `crates/hawking-core/src/model/qwen_dense.rs` — the `'ud_loop` EH-ON path,
specifically the `kv.seq_len` update or the verify position vector when all drafts are accepted
(na=dlen, full-accept boundary). Cross-check against `'ud_loop` OFF path.

**Gate to pass:**
```bash
cargo test -p hawking-core --test event_horizon_parity_prop 2>&1 | tail -20
# Must show: 20/20 prompts bit-identical
```

---

## Build wave results — 2026-06-21

### Wave 1 — New files (all created, all compiling)

| Item | File | Status | Notes |
|------|------|--------|-------|
| A | `tools/bench/eh_market_bench.py` + `.sh` | SKIP-NO-BINARY | Syntax OK; hawking binary not yet built; runs after `cargo build --release` |
| B | `tools/training/eh_eagle_tau_sweep.py` | PASS (mock) | `--smoke --no-gpu` ran, τ=0.000 NO-GO (expected for mock); kill-ledger enforced |
| C | `tools/eh_free_market_tau.py` | PASS | CPU sim; user_ngram τ=1.057, suffix τ=1.057, retrieval τ=1.077 on NL corpus; run on code corpus for real numbers |
| D | `crates/hawking-core/src/speculate/parallel_draft.rs` | PASS | 8103 bytes; 4 unit tests pass; `requires_hidden=true`; never enabled; `enable_neural_slot` refuses without GO |
| E | `crates/hawking-core/tests/event_horizon_parity_prop.rs` | **FAIL 6/20** | See Priority 0 above |
| F | `crates/hawking-core/src/speculate/suffix_automaton.rs` | PASS | 18852 bytes; 5 unit tests pass; online SAM; behind `HAWKING_EH_SAM` sub-flag |
| G | `docs/plans/hawking_event_horizon_status.md` | PASS | As-built matrix written |

### Wave 2 — Bug fix + wiring

| Item | Status | Evidence |
|------|--------|----------|
| H (`'udpf_loop` fix) | **PASS** | `user_draft_propose_first_bit_identical_default` → ok in 35.7s with weights |
| I (register `parallel_draft` + `suffix_automaton`) | PASS | `mod.rs` + `router.rs` updated; `ProposerId::ParallelDraft` added |
| Integration | PASS | `cargo build` clean; 79/79 speculate lib tests; 8/8 e2e tests |

### Wave 3 — Validation summary

| Gate | Result |
|------|--------|
| EH parity property (N=20) | **FAIL 6/20** |
| propose-first bit-identical | PASS |
| speculate lib tests | PASS 79/79 |
| smoke A (market bench) | SKIP-NO-BINARY |
| smoke B (τ sweep, mock) | PASS |
| smoke C (replay-τ, CPU) | PASS |

### Golden rules audit

| Rule | Status |
|------|--------|
| Nothing committed this wave | ✓ (pre-existing EH commits 6e9f271, 4b7ffc1 not counted) |
| `HAWKING_QWEN_EVENT_HORIZON` default OFF | ✓ (`is_ok()` check) |
| No neural slot enabled in production | ✓ |
| No new Metal shaders | ✓ |
| No head trained | ✓ |
| No full GPU run | ✓ (max_new_tokens=16 cap throughout) |

---

## Wall-clock chain (h executes in order)

### Item 0 — KD eval re-run (CPU-only, run immediately)
Chunked-flag mismatch: train used `--use-chunked --chunk-size 32`, eval omitted it → forward-path swap.
Re-eval **all three variants** with `--use-chunked --chunk-size 32`.
Accept should track loss after fix: 75M ≥ 35M ≥ 26M.
SFT baseline location: `artifacts/lowbit_rwkv7/runs/draft_*_probe/final` is empty — locate or flag
as KD-absolute-only.

**KD frontier status (completed 04:14:34):**
- draft_26m_probe: loss=? accept=0.1975 (pre-fix, non-chunked)
- draft_35m_probe: loss=5.0802 accept=? (just finished)
- draft_75m_probe: loss=4.64 accept=0.05125 (pre-fix, non-chunked)

### Items 1–4 — GPU runs (GPU free as of 04:14:34)

1. **Fix parity bug** (Priority 0) → re-run `event_horizon_parity_prop` → 20/20
2. **Market bench smoke**: `cargo build --release -p hawking` → `bash tools/bench/eh_market_bench.sh`
   (3 arms × 16 tokens = 48 forward steps → writes `reports/eh_market_bench_smoke.md`)
3. **τ sweep (real weights)**: `python3 tools/training/eh_eagle_tau_sweep.py --smoke --model models/qwen2.5-3b-instruct-q4_k_m.gguf`
4. **Parity property uncapped**: after fixing parity bug, un-cap `MAX_NEW_TOKENS_CAP=16` → 256
   in `event_horizon_parity_prop.rs` and re-run

---

## Files created this wave

```
tools/bench/eh_market_bench.py
tools/bench/eh_market_bench.sh
tools/training/eh_eagle_tau_sweep.py
tools/eh_free_market_tau.py
crates/hawking-core/src/speculate/parallel_draft.rs
crates/hawking-core/src/speculate/suffix_automaton.rs
crates/hawking-core/tests/event_horizon_parity_prop.rs
docs/plans/hawking_event_horizon_status.md
docs/plans/h_autoverify_protocol.md  ← the handoff contract
docs/plans/SESSION_HANDOFF_2026_06_21.md  ← this file
reports/eh_market_bench_smoke.md  (placeholder)
reports/eagle_tau_sweep.md
reports/free_market_tau.md
```
