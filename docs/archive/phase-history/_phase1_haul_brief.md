# Phase 1 — Haul 1 brief

Execution protocol for the autonomous agent running Haul 1. The
manifest (`_phase1_haul_manifest.md`) is the *what*; this brief is
the *how*. Read both before starting.

## §1 Halt protocol

Single-purpose commits, one per item. On item-halt:
1. The runner (`tools/haul/run-gates.sh`) writes a stub at
   `_phase1_haul_attempt<N>_blocked.md`. Fill in the stub:
   root cause, what ran up to halt, what attended work unblocks,
   followups for next session.
2. Land a halt commit with subject `phase 1: HALT — <gate>: <root cause>`
   (no body changes; the blocked doc carries the context).
3. Apply the halt-budget rule:
   - **G1.1: 1 halt = end haul.** Do not attempt items 2–4.
   - **G1.2 / G1.3 / G1.4: 2 halts in this group = end haul.**
     First port-failure: continue to next *independent* item.
4. Do NOT peel-onion fix the underlying issue mid-haul. Halts are
   for surfacing gaps; the next attended session bundles the fix.

## §2 State drift

Before each item:

```
git status                          # must be clean before item starts
./tools/haul/coexist.sh probe       # exit 0 to proceed; 1 = retry; 2 = halt
```

If `git status` is dirty (uncommitted from a prior halt), do not
start the next item — that's a state-drift halt with
`reason: dirty_worktree`.

## §3 Precondition stack

Before haul 1 runs, the runner asserts (via Phase A install
verification):
- `cargo build --release --workspace` exits 0
- `cargo test --workspace --lib` passes 15 tests
- `models/deepseek-v2-lite-q4.gguf` exists
- `tools/haul/coexist.sh probe` exits 0 OR 1 (not 2)
- `_phase1_kernel_baseline.hashes` exists (header-only is fine for
  haul 1 — items populate rows)

Any precondition failure halts the haul before it starts. Result:
`_phase1_install_blocked.md` written, no items run.

## §4 Per-item dependency map

```
G1.1 ──┐
       ├── G1.2 (independent of G1.3, G1.4)
       ├── G1.3 (independent of G1.2, G1.4)
       └── G1.4 (independent of G1.2, G1.3)
```

If G1.1 halts → no later items run. If G1.2 halts → still attempt
G1.3 and G1.4.

## §5 Time budget

- Per-item soft ceiling: 60 min (45 min impl + 15 min validation)
- Haul hard ceiling: 4 hr
- On hard ceiling, halt cleanly with `reason: ceiling`. The runner
  writes the in-flight item to a blocked doc; next attended
  session re-scopes.

Phase 0's CPU baseline is ~6 min per `dismantle generate` 32-token
run. Token-baseline regression (3 tokens) is roughly half that. The
4hr ceiling has plenty of headroom for 4 items even at Phase-0
speeds — the ceiling exists to catch *runaway thrash* (memory
swapping, infinite loops) not normal slowness.

## §6 Per-item validation

Each item's manifest entry lists its validation step. The runner
captures evidence triples to `tools/haul/_evidence/$GATE/`:
- `pre.json` — git HEAD, memory state, test-count baseline
- `post.json` — exit code of validator, files modified
- `verify.json` — independent re-run attestation

Any of: post.exit_code != 0, verify.attestation == false → halt.

## §7 Pre-flight checklist (run once at haul start)

```
git status                                   # clean
./tools/haul/coexist.sh probe                # exit 0 or 1
cargo build --release --workspace            # exit 0
cargo test --workspace --lib                 # 15 pass
./target/release/dismantle version           # prints "dismantle 0.0.1"
```

## §8 Post-flight checklist (after haul ends, success or halt)

```
git log --oneline phase0-baseline..HEAD      # shows item commits + halt commits
./tools/haul/verify-evidence.sh phase1       # exit 0 if all gates green
ls _phase1_*.md                              # closeout or blocked docs present
cargo test --workspace --lib                 # still 15 + new parity tests
```

If any post-flight assertion fails, append a `## post-flight failure`
section to the closeout/blocked doc explaining what went wrong.

## §9 What this haul does NOT do

(Verbatim from the manifest, restated as a check the agent applies
to every "while I'm here" temptation.)

- Implement wedge 2 (fused Q4_K_M dequant inside FMA loop).
- Implement wedge 1 (single-launch fused MoE kernel).
- Touch attention beyond `o_proj` (no flash-attention, no MLA
  decompress kernel).
- Move sampling to GPU.
- Change quant code, tokenizer, engine API, or Cargo deps.
- Push to remote.

If any in-haul fix appears to require any of the above, **halt
the item, write the blocked doc, and end work on that item.** The
blocked doc explicitly proposes the change as scope-creep that the
next attended session adopts or rejects.

## §10 Honest scope caveat

After Haul 1 lands all 4 items, dismantle's decode tok/s is
expected to land between **3 and 50 tok/s** depending on:
- How well naive Metal GEMV uses simdgroup matrix ops
- Lazy-dequant scratch-buffer round-trip cost per call
- Co-existence throttling under slm training pressure

Haul 1's job is **correctness, not speed**. We're getting off
CPU GEMV. The wedges (haul 2+) deliver the perf.

## Co-existence reminder (Phase 1 specific)

Per `CLAUDE.md § Memory-coexist rule`:
- Probe memory before each item. Sleep 30s × up to 5 retries on
  degraded; halt with `reason: memory_pressure_critical` after 5
  min critical.
- Every dismantle subprocess: `nice -n 19 taskpolicy -b ./target/release/dismantle ...`
- RSS sentinel: 5 GB ceiling. Halt with `reason: rss_ceiling` if
  exceeded.
- 30s inter-item cool-down (the runner enforces).
- Synthetic-first parity tests (no model load); integration smoke
  only if `coexist.sh probe` returns 0; record `PASS-PARITY-ONLY`
  otherwise.
- 3-token (not 5) regression validation.
