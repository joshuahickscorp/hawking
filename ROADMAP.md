# Roadmap

What is likely to happen next. Every item here comes from something already written down in
this repo: an open defect, a "still to do" note, or a gate that has not been run. No dates,
and nothing here is a promise.

The engine is close to a stable state. Most of the remaining work is finishing wiring that
already exists, adding tests to things that only have code paths, and tidying up before a
release — not new features.

## Near term

- Bring the docs back in line with the code. `ARCHITECTURE.md` and `MODELS.md` still
  describe Gemma 2, Phi-3, OLMoE, Mamba2 and Mixtral as part of the engine; those adapters
  were moved out of the tree into `packs/hawking-adapters-extra.json`.
- Fix the doc links that point at files which were archived (`BASELINES.md`, `FAILURES.md`
  and `WATCHLIST.md` all cite a plan doc that no longer exists).
- Settle the version numbers before tagging a release: `Cargo.toml` says 0.2.2, while the
  top of `CHANGELOG.md` is headed "Unreleased (post-v2.0.0)".
- Wire the per-channel int4 KV cache into the decode path and gate it on a real perplexity
  run. The kernels and the parity test are built and passing; only the wiring and the
  quality check are left.
- Wire the RWKV-7 decode engine into the serve path, so it is reachable over the HTTP API
  the same way Qwen2.5 is. The engine and its CPU/Metal parity tests already exist.
- Add parity or quality gates for the families that currently only "Run" or are "Untested"
  in `MODELS.md` — Qwen2.5-1.5B/7B, Llama 3.x, Mistral, Qwen3-MoE, DeepSeek-V2-Lite.
- Add a CPU reference decode path for MoE models. CPU/Metal parity is dense-only today.
- Pick one canonical home for the pinned-requirements file that currently exists in two
  places, since two tools point operators at different copies.
- Actually start running the monthly competitor check in `WATCHLIST.md`. It was written
  down but has never been run.

## Later

- Finish the Eagle5 speculative decoding port. Two pieces are missing: real capture-layer
  plumbing (the trained head currently runs with its inputs zeroed, which drops the accept
  rate to roughly 0.05–0.15 against a projected 0.70) and a batched verify step. Until both
  land, speculative decoding does not make anything faster.
- Close the remaining single-stream decode gap to llama.cpp and MLX. The harness that
  measures it already exists.
- Finish the deferred tool consolidation. Each piece was left alone for a stated reason and
  needs an idle process, an accepted plan-text change, or a behavioral test first.
- Implement the bake path for `hawking press`, so it can produce a compressed model instead
  of only printing a plan. Uncertain: this is described in the code as owner-gated, so the
  timing is not purely an engineering question.

## Compression research track

The repo also contains a research track on low-bit model compression. One caveat matters
more than any result in it: every number it reports is measured on bytes and weights, not
on whether the compressed model still answers correctly. The repo states this directly and
nothing here changes it. Remaining work:

- Fix the memory blow-up in the packer so the vocabulary tensor can be packed at all. The
  k-means distance matrix grows with tensor size (about 61 GB for the 951M-weight embedding
  and output layer), so one shard has never packed and an artifact would finish at 281 of
  282 shards. A chunked rewrite was tried and reverted after it caused GPU faults.
- Make the product-quantization packer deterministic, or correct the docstring that claims
  it already is. Identical weights and seed currently produce different hashes.
- Close the authorization gap where a sealed receipt could approve deleting source weights
  without the file it describes ever being opened.
- Run the six planned pilot windows, then replace the current rate allocation — which is
  positional, and says so rather than pretending it was earned — with one based on measured
  per-role sensitivity.
- Run the real parent-vs-packed forward pass on the frozen holdout. The structural
  expert-reduction arm is implemented with an exact byte ledger, but no claim about model
  quality can be made until this runs.
- Finish the 1200-token routing calibration that replaces an earlier 88-token sample, which
  was contaminated because calibration and scoring used the same prompts.
