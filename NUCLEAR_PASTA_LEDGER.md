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
- main fast-forwarded 088aa661 -> 714294ac (collapse merged); validated green; sealed hawking-pre-seed-final; rollback tag hawking-pre-collapse-main.
- Phase 1 (spec extraction): HAWKING_SEED_ARCHITECTURE.md written = the oracle (parity golden 2d1559cf, Gravity law, pack ABI, runtime/CLI contracts) + 3 candidate designs + measurement plan.

## HAWKING SEED — ALL THREE CANDIDATES BUILT + MEASURED; WINNER SELECTED
- Candidate A (codex/hawking-seed-a, PR #28 CLOSED/sealed): Rust microkernel, seed core 1,068 LOC, ship ~53,481 (delegates decode to the 52k predecessor hawking-core). tag seed-a-vertical. Bit-identical parity.
- Candidate B (codex/hawking-seed-b, PR #29 CLOSED/sealed): self-contained scalar runtime, ship 1,945 LOC, 9.68 tok/s, 726MB RSS, bit-identical parity, NO predecessor. INDEPENDENT AUDIT sealed (CANDIDATE_B_INDEPENDENT_AUDIT.{json,md}): no hawking-core, no subprocess, no hardcoded golden, 16 distinct per-step logit shas + adversarial tests. PR #29 was CONFLICTING (docs-only ledger conflict + stale Cargo.lock) -> resolved (merged main, synced lock) -> MERGEABLE, then closed as sealed experiment.
- Candidate C (codex/hawking-seed-c, PR #30 OPEN = final Seed merge PR; tag hawking-seed-final): Event Horizon engine. DIRECT-QUANT (mmap, per-row tile dequant, NO dense f32 shadow) + METAL (15x LM-head, measured, argmax agrees) + SUB-BIT direct execution (ternary latent factor 0.401 BPW) + Doctor rescue + MoE-ready IR + bounded 120B F2 bridge. ship 2,204 LOC (UNDER 3,000 gravitational target), 212MB RSS (3.5x < B), bit-identical parity, 23 tests. NO predecessor. tags seed-c-{record,gguf-view,direct-quant-cpu,metal-linear,ir,smollm-parity,subbit,doctor,120b-f2,vertical}.
- 120B F2: real GPT-OSS-120B ABSENT on this machine (the 2026-07-17 65GiB MXFP4 source is gone) -> FAIL CLOSED on the real path + synthetic MoE fixture (Route/Expert/SubBitExpert/DoctorRescue/WeightedCombine). No 120B capability claimed.
- SELECTION: WINNER = C (Event Horizon). Not by LOC alone: C dominates on memory, hardware-native Metal, sub-bit direct execution, and giant-model extensibility (MoE IR + F2) - the mission axes - at parity auditability with B. A rejected (delegates to 52k engine). B superb but dense-shadow (no direct-quant/Metal/sub-bit/MoE). Comparison sealed HAWKING_SEED_CANDIDATE_COMPARISON.{md,json} sha 535ef92f; C metrics HAWKING_SEED_C_METRICS.json sha 5e3ce03d.
- FINAL SEED = Candidate C, tag hawking-seed-final, merge PR #30 (merge under user review/CI policy; not auto-merged). A & B retained as sealed experiments. Highest-value next optimization: wire the validated Metal LM-head into the decode loop (near-tie guard) for C's memory + Metal speed at bit-identical tokens.
