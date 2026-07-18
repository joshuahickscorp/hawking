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
- Phase 1 (spec extraction): HAWKING_SEED_ARCHITECTURE.md written = the oracle (parity golden 2d1559cf, Gravity law 34, pack ABI, runtime/CLI contracts) + Record envelope + one state machine + execution IR + 3 candidate designs (A Rust microkernel / B functional Rust / C mixed) + measurement plan.

## HAWKING SEED — Candidate A (Rust microkernel): VERTICAL PATH GREEN
- branch codex/hawking-seed-a; commit 6d4bb337; PR #28; tag seed-a-vertical. crate crates/hawking-seed, seed_core 1,068 LOC (CLI 287). Full vertical path green; decode DELEGATED to owned default runtime pack hawking-core (52,413, reported separately). ship ~53,481 LOC. binary 1.68MB. Metrics HAWKING_SEED_METRICS.json.

## HAWKING SEED — Candidate B (self-contained execution-IR runtime): VERTICAL PATH GREEN, NO PREDECESSOR ENGINE
- branch codex/hawking-seed-b (off hawking-pre-seed-final f1369745); commit 0685cdd1; tags seed-b-{record,gguf,tokenizer,ir,ops,runtime,parity,vertical}.
- THE IDEOLOGICAL RESULT: Hawking executes SmolLM2-135M on its OWN compact runtime. crate crates/hawking-seed-b: from-scratch GGUF reader, four GGML dequantizers (Q8_0/Q5_0/Q6_K/Q4_K), GPT-2 byte-level BPE tokenizer, compact execution IR (register file + 10 op kinds), scalar CPU ops (RMSNorm f64 / NeoX RoPE / out-major GEMV / softmax / SiLU), KV cache, Llama forward, greedy sampler. NO hawking-core, NO `hawking` binary, NO predecessor decode loop. Deps: serde/serde_json/sha2/thiserror + half (f16 primitive only).
- `hawking-seed-b run` GREEN: pack -> GGUF -> IR(484 ops) -> tensors(106M f32 + tied f16 embed, 65ms) -> tokenize (== golden) -> prefill -> decode 16 tok -> BIT-IDENTICAL greedy tokens + completion sha 2d1559cf -> Gravity -> Forge (through B gemv) -> Doctor (billed) -> sealed evidence -> drain/resume/verify. exit 0. Numerics matched to predecessor AS REFERENCE ONLY. Golden captured black-box via --trace-tokens (no white-box dump exists; intermediate stages covered by exact full-decode self-consistency).
- MEASUREMENTS: complete SHIPPED = 1,945 impl LOC (authority 417, runtime+adapter 1090, forge/doctor 139, CLI 299); default_pack_LOC = 0; nothing excluded as a pack. ALL targets crushed (seed<=2000, rt+adapter<=8000, cli<=750, ship<=12000, STRETCH<=10000). Binary 0.55MB. 20 unit + 1 real-decode integration test green. Peak RSS 726MB (f32 dequant-at-load, honest tradeoff). Metrics HAWKING_SEED_B_METRICS.json sha 1116a975. vs A: A ship ~53,481 delegated; B ship 1,945 self-contained = ~27.5x smaller AND independent. Candidate B's required statement is now TRUE.
- NEXT: Candidate C (mixed) only after B's statement holds (it does). Or optimize B (Metal/SIMD/quantized-resident) to close perf/memory; or unify the shared authority + IR into one Seed both candidates summon.
