# ADAPTIVE TRANSFER LADDER PRECHECK

Generated 2026-07-19. Campaign: F0 gpt-oss-120b, F1 Qwen3-235B, F2 Qwen3.5-397B, F3 DeepSeek-V3.2, F4 Kimi-K2.6.

## Live truth

- Git: HEAD == origin/main == `4fbca8bc`, branch main, remote `joshuahickscorp/hawking`. Uncommitted this
  campaign: the real forward, the G4 controller, the prechecks, G4 artifacts. This /goal authorizes
  commits/pushes/PRs/merges when green (Claude is never attributed in git).
- G4 controller pid 48797 ALIVE and healthy (do not interrupt). 6 of 12 rows sealed: all 6
  original-parent rows done, now on the first packed row `gen_paris__rvq1.0`. ETA ~30 min.
- Real forward validated: "The capital of France is" gives " Paris" p=0.405 over 201088 vocab.
- First-ever real parent perplexity (the reference the quality contract requires): code_py 1.92,
  reason 3.46, math 5.14, gen_science 6.68, instr 12.26, gen_paris 27.43. Domain-ordered, which
  corroborates forward correctness.

## Qwen3-235B admitted

- Repo `Qwen/Qwen3-235B-A22B-Instruct-2507`, immutable revision
  `ac9c66cc9b46af7306746a9250f23d47083d689e`, license apache-2.0.
- Architecture verified live and matches the prompt: qwen3_moe, 94 layers, 128 experts, top-8, GQA
  64 Q / 4 KV, hidden 4096, moe_intermediate 1536, vocab 151936, bf16, untied embeddings.
- 437.9 GiB, 118 weight shards. Metadata (config, tokenizer, index) downloaded to
  `models/qwen3-235b-a22b/_meta`. Receipt: `QWEN3_235B_SOURCE_ADMISSION.json`.

## Storage and the honest gating chain

- 559.7 GiB free. Qwen full source (438 GiB) does NOT coexist with the 61 GB 120B source plus
  reserve. Required mode: full transfer after 120B source release, or shard-serial streaming
  (one-parent storage law).
- Chain to a live Qwen transfer controller: (1) G4 completes and seals; (2) tensor-class correction
  wave C0..C5 and the adaptive G5 decision, or an honest boundary, sealed; (3) the 15 120B
  source-release gates pass, releasing 61 GB (rehydratable from openai/gpt-oss-120b) or committing to
  shard-streaming; (4) download 438 GiB Qwen or shard-stream; (5) Qwen adapter and Q0/Q1/Q2 synthetic
  tests green; (6) launch the durable Qwen controller. Steps 2 to 4 are multi-hour and gate the heavy
  transfer. This session makes maximal real progress; "advancing Qwen checkpoints" is downstream of
  the 120B seal and source release and will not be faked.

## Resources

M3 Ultra 96 GB / 28 c, ~72 GB free, 559.7 GiB disk, network reachable (HF HTTP 200), HF client
present (huggingface_hub 1.13.0, hf_transfer, hf CLI). One heavy lease, held by G4. CUDA blocked
(no sealed budget), Apple-only lane.

## Rollback

`git reset --hard 4fbca8bc`; `kill 48797` stops G4. No model source mutated (byte-range read-only);
only Qwen metadata downloaded so far.
