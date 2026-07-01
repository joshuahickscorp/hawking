# Event Horizon Auto-Handoff Protocol (2026-06-21)

## Sessions
- **Source**: Sonnet (this session) — ran the 3-wave build workflow, writes handoff, then stops
- **Target**: h (`local_262ba758-a9ab-483d-9a94-f6a721b34e80`) — ingests, auto-verifies, implements, chains

## Baton-pass hygiene (h does NOT proceed until clean)
```
ps aux | grep -E 'cargo|rustc' | grep -v grep
```
If any cargo/rustc procs are running: wait. The repo must be idle before h touches it.

## h's ingest target
The legacy 2026-06-21 session handoff was consolidated into
`/Users/scammermike/Downloads/PROJECT_RETROSPECTIVE_CHECKPOINTS_2026_06_28.md`. Fresh ingest
should use the current Studio plan and active operational docs instead of reviving session handoffs.

## h's auto-verify sequence (non-negotiable, run in order)
```bash
cd /Users/scammermike/Downloads/hawking

# 1. Clean build
cargo build -p hawking-core 2>&1 | tail -20

# 2. All speculate lib tests (expect 69 + new unit tests from D and F)
cargo test -p hawking-core --lib 'speculate::' 2>&1 | tail -40

# 3. Property parity gate (N=20 prompts × lengths, EH-OFF vs EH-ON)
cargo test -p hawking-core --test event_horizon_parity_prop 2>&1 | tail -40

# 4. Propose-first bit-identical test (item H fix)
cargo test -p hawking-core --test user_draft_parity_e2e \
  user_draft_propose_first_bit_identical_default 2>&1 | tail -20
```
Tests 3 and 4 SKIP silently without model weights — SKIP is a pass. FAIL is not.

## h's auto-implement scope
- Items A–G (new files): verify they exist, syntax-check Python with `python3 -c "import ast; ast.parse(...)"`, no edits unless broken
- Item H (`'udpf_loop` fix): verify the fix is in place; if test fails with weights, find and fix the position-advance error, re-gate
- Item I (module registration): verify `mod.rs` has `parallel_draft` + `suffix_automaton`; verify `router.rs` has `ProposerId::ParallelDraft`
- Any gate failure: fix the root cause, re-run the full gate sequence, commit only after all gates green

## Safety rails (hard — no exceptions, no deliberation)
| Rule | Why |
|------|-----|
| NEVER `enable_neural_slot` with `"GO"` | kill-ledger: EAGLE τ=0.877 < 2.5 gate — permanent NO-GO |
| NEVER train any head | GPU is KD-owned; head training is out of scope |
| NEVER flip `HAWKING_QWEN_EVENT_HORIZON` default to on | additive + flag-gated invariant |
| NEVER write a new Metal shader | out of scope |
| NEVER commit without parity gate passing | losslessness is non-negotiable |
| NEVER commit with AI attribution | per global AGENT.md |

## Wall-clock chain (run in priority order after source stops)

### Chain item 0 — KD eval re-run (CPU-only, highest priority, run first)
**Provenance (airtight, 2026-06-21):**
- TRAIN: `--use-chunked --chunk-size 32` (confirmed from live `/tmp/kd_pipeline.sh`, relaunched 01:25)
- EVAL: `--draft-k 4 --device cpu` — no `--use-chunked` (confirmed same script)
- Accept JSONs: `26m=0.1975`, `75m=0.05125` from this run's `kd_draft_*_probe_eval.txt`

The forward-path swap (chunked WKV vs recurrent) compounds with depth — deeper 75M diverges hard
despite lower loss (4.64). The "better loss, worse accept" inversion is the config mismatch, not
a real regression.

**Re-eval ALL THREE variants** (not just 75M — the 26M's 19.75% was also non-chunked and only
looked plausible by luck of being shallow):
```bash
# Re-eval each variant with --use-chunked --chunk-size 32 matching training:
# Find the eval command in tools/training/g1a_v2_expansion_chain.sh or
# tools/training/launch_draft_sweep.sh — add --use-chunked --chunk-size 32 to all eval calls.
# Accept should track loss: 75M ≥ 35M ≥ 26M after the fix.
```

**SFT baseline location sub-task** (doesn't block KD-internal verdict, but needed for KD-vs-SFT):
`artifacts/lowbit_rwkv7/runs/draft_*_probe/final` is empty. Locate non-KD SFT checkpoints
(likely under the sweep output dir) and re-eval with the same chunked config. If missing: flag
comparison as KD-absolute-only and document in the handoff. Do NOT block chain item 0 on this.

### Chain items 1–4 — GPU runs (GPU is FREE as of 04:14:34 — fire immediately on pickup)
KD frontier completed: draft_35m_probe exit=0, loss=5.0802. All three variants trained.
**No waiting — GPU is available now.**

1. **Market bench (full)**: `bash tools/bench/eh_market_bench.sh` — un-cap `max_new_tokens` to 256+
2. **τ sweep (real weights)**: `python3 tools/training/eh_eagle_tau_sweep.py --smoke` — drop `--no-gpu`
3. **Parity property (full)**: un-cap `max_new_tokens=16` → 256 in `event_horizon_parity_prop.rs`,
   re-run `cargo test -p hawking-core --test event_horizon_parity_prop`
4. **Paired bench**: validate accepted-tokens/s with full free market vs baseline

### Chain item 5 — after τ sweep result
If best triplet τ ≥ 2.5 (unlikely given existing evidence but measure honestly):
Document in `docs/plans/eagle_tau_gate_result.md`. Do NOT call `enable_neural_slot` until
a full replay-oracle run (not just a sweep smoke) confirms τ ≥ 2.5 on the target workload.
If τ < 2.5: update `docs/dead_levers.md` with the new measurement and close the loop.
