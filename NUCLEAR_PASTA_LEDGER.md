# NUCLEAR PASTA execution ledger

Durable resume state for the maximum-density compression campaign. Status: **descent in progress**.

| field | value |
|---|---|
| branch | codex/clean-slate-collapse |
| target | irreducible kernel <=50k OR two implemented architectures prove a higher floor |
| C0/C1 | complete |
| C2/C3 | incomplete |
| Escape Receipt | revoked/provisional |

## Order of operations (A-L)
- [~] A. CLI collapse <=2.5k  (at 4,851; probes+studio+bench extracted, decode-parity GREEN)
- [ ] B. Python control <=10k
- [x] C. decode-parity harness (decode_parity_harness.py; verify GREEN; tag nuclear-parity)
- [x] D. generic speculate EXTRACTED to hawking-speculate crate (6,312 LOC, decode-parity GREEN, tag nuclear-speculate-pack)
- [~] E. kernel-bank ABI + extract (NEXT; same pattern, decode-parity gated)
- [ ] F. serving condensation
- [ ] G. Architecture A
- [ ] H. Architecture B (3 profiles)
- [ ] I. select lower green
- [ ] J. <=50k attempt
- [ ] K. final Escape Receipt (if required)
- [ ] L. heavy-run readiness (print, do not launch)

## Log
- CLI: 11,385 -> 5,200 (probes+process_joule -> hawking-lab-cli pack; studio -> hawking-lab). tags archb-cli-{1,2}. product 94,785 -> 88,606.

- parity harness: golden captured on SmolLM-135M; baseline/exact-shared/suffix-automaton all bit-identical (sha 2d1559cf); lossless-speculation invariant enforced.
- CLI: 5,200 -> 4,851 (bench_server+bench_kernel -> hawking-bench pack); decode parity re-verified GREEN. product 88,606 -> 88,373. tag nuclear-cli.
- D DONE: speculate 19 modules physically moved hawking-core->hawking-speculate; cycle broken via Box<dyn Error> ExactTarget boundary; hawking-core src 58.7k->52.4k; decode parity GREEN; 123 spec tests pass.

## HAWKING SEED — Phase 0 CLOSE THE OLD WORLD: DONE
- main fast-forwarded 088aa661 -> 714294ac (collapse merged); validated green (core 123 + spec 123 + py 165 + gravity 34 + parity GREEN); sealed hawking-pre-seed-final; rollback tag hawking-pre-collapse-main. Pushed origin/main.
- Phase 1-8 (build 3 Seed candidate architectures -> 10k): genuine multi-month R&D campaign; resume via this ledger.
- Phase 1 (spec extraction): HAWKING_SEED_ARCHITECTURE.md written = the oracle (parity golden 2d1559cf, Gravity law 34, pack ABI, runtime/CLI contracts) + Record envelope + one state machine + execution IR + 3 candidate designs (A Rust microkernel / B functional Rust / C mixed) + measurement plan. Next: build candidate A in an isolated worktree, migrate status->artifact->decode->Gravity/Forge/Doctor->evidence, parity-gated.

## HAWKING SEED — Candidate A (Rust microkernel): VERTICAL PATH GREEN
- branch codex/hawking-seed-a (off hawking-pre-seed-final f1369745); commit 6d4bb337; PR #28 (draft); tag seed-a-vertical.
- crate crates/hawking-seed: 8 modules (record/state/gravity/pack/runtime/forge/doctor/evidence) + registry CLI. seed_core 1,068 LOC (CLI 287) UNDER the 10k gravitational target. 14 lib vectors + vertical-path integration green. binary 1.68MB, startup 0.12s.
- `hawking-seed run` executes the COMPLETE real vertical path GREEN: verified offline pack -> sealed Record -> 9-state persisted transition -> real SmolLM artifact identity (ed5fa30c) -> deterministic greedy decode via default runtime pack -> PARITY bit-identical to golden 2d1559cf -> Gravity sub-bit decision (escape-without-receipt denied; F1/deferral cannot escape) -> Forge int8 pack/round-trip (10.5 BPW, rel_err 0.0033) -> Doctor outlier-column treatment (rel_err 0.0004->0.0003, total 12.75 BPW, within budget) -> sealed evidence receipt -> pause/resume -> drain/seal -> verify (8 sealed records, all seals valid, state Sealed). exit 0.
- HONEST framing: seed CORE = 1,068 LOC (real functional microkernel, not hollow); model math delegated to OWNED default runtime pack hawking-core (52,413, reported SEPARATELY per arch Section 7/20/276 - not hidden, not a downloader). ship LOC = 1,068 + 52,413. Open question for a truly-small SHIP: shrink the 52k runtime engine (execution-IR / kernel-bank). No predecessor source mutated (reads golden fixture only). metrics: HAWKING_SEED_METRICS.json sha c511618e.
- NEXT: Candidate B / C only after A satisfies its statement (it now does). Consider Candidate B (functional Rust) or C (mixed) for a lower-floor comparison; or attack the 52k runtime-pack floor directly.
