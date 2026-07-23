# First Mandatory Implementation Receipt - Harness Spec

Edition: 2026-07-19 · Bible Book XXI §76 + RIP §1.5 (Prove). The first milestone is not a feature menu; it is one full task through the real app, no mocks, no facade, ending with the user accepting the patch.

## 1. The task

A private multi-file Rust or TypeScript bug or feature in HIDE itself (dogfood). Candidate for the first run: a small, self-contained defect with a clear failing test, touching 2-3 files, so the loop exercises read/search, plan, edit, build, test, and a failure-then-recovery. Example class: a bug in a `hide-*` crate with a reproducible unit test.

## 2. Required behaviors (the 20 steps) mapped to what must be wired

| # | Behavior | Wired by (phase) | Status after Phase 1b Increment 1 |
|---|---|---|---|
| 1 | Open + trust repository | trust gate (Phase 1/Sec) | pending (trust gate NEW) |
| 2 | Start session | hide-backend session | EXISTS |
| 3 | Set goal + acceptance | goal/plan (kernel) | EXISTS (kernel plan) |
| 4 | Compile context from real index | hawking-context + hawking-index + startup indexer | Increment 1 feeds compiled context; startup indexer still NEW |
| 5 | Plan with declared oracles | hide-kernel RuntimePlanner | Increment 2 |
| 6 | Read + search files | hide-tools | EXISTS (needs kernel dispatch, Increment 2) |
| 7 | Create a reproduction | kernel reproduction-first | Increment 2 |
| 8 | Edit transactionally | hide-tools verifying applier | EXISTS (needs dispatch) |
| 9 | Build, typecheck, tests | hide-kernel ProcessOracle | EXISTS (needs with_standard_oracles wired) |
| 10 | React to a failure | kernel repair/replan | EXISTS (driver) |
| 11 | Accept a steering message | InterruptHub -> kernel | partial (single-shot has it; loop needs forwarding) |
| 12 | Green verification receipt | hide-kernel verify.result events | EXISTS (driver persists verdicts) |
| 13 | Show diff | hide-tools diff + FE HunkReview | EXISTS (FE) |
| 14 | Persist session | hide-backend event log | EXISTS |
| 15 | Restart server | hide-serve | EXISTS |
| 16 | Resume thread | BackendReplayService | EXISTS |
| 17 | Fork thread for review | replay fork | EXISTS (packed) |
| 18 | Search transcript | event log scan | EXISTS (needs a search verb) |
| 19 | Compare reviewer result | agent-tree synthesis | Increment 2 / Phase 6 |
| 20 | Export receipt | this harness | NEW (Phase 1c) |

Gap to the receipt after Increment 1: the kernel-loop dispatch (Increment 2, steps 5-11), a startup code indexer (step 4 fully), a trust gate (step 1), and the receipt exporter (step 20). Those are the Phase 1b Increment 2 + Phase 1c scope.

## 3. Passing gate

The user accepts the patch. No mock data, no disconnected UI, no raw-prompt one-shot, no "should work". A run that reaches step 12 with a green oracle verdict and a reviewable diff, persisted and resumable, is a pass once the human accepts.

## 4. Receipt schema (RIP §1.5) - what the exporter emits

```yaml
# hide.receipt.v1
task: "<id + one-line>"
repo_snapshot: "<git rev + dirty hash>"
hardware: "Apple M3 Ultra 96GB"
model: "<served model + quant>"
effort_policy: "Interactive | Thorough | Fleet"
tools_enabled: ["fs.read","search","edit","proc.test","git"]
context:
  pack_id: "ctx_..."
  used_tokens: <n>
  retained_spans: <n>
  sources: ["repo://...","memory://..."]
actions: [ {kind, tool, effect, ok} ]
effects_approved: <n>
timings_ms: { wall, model, tool, verify }
compute: { tokens_out, prefill_reused, gpu_ms }
patch: "<content-addressed diff id>"
tests: { command, before: "red", after: "green" }
regression: "no_worse | worse:<...>"
interventions: <n>
failure_modes: ["<...>"]
baseline: "<thin-mode or one-shot comparison>"
accepted: true|false
```

## 5. Measurement discipline

- The receipt is emitted from the REAL app path, not a script that simulates it (Bible law 25: completion is a receipt; law 18: no facade).
- Every number pins its method (hardware, model, quant, context policy). No "fastest/densest" claim without a receipt (carried over from the prior research package's evidence bar).
- The thin-mode baseline (a minimal one-shot loop) runs the same task so the receipt shows whether the harness (context + kernel + verify) actually helped - RIP §1.5 Prove. A mechanism that does not beat its baseline is killed (Bible law 19).
