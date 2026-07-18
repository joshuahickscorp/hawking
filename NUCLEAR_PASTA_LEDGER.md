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
