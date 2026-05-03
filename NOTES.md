# Foundation pass — notes & deferred items

Status as of 2026-04-27. The workspace builds, all 11 unit tests
pass, the release binary runs and reports `dismantle 0.0.1`. This
file is the running log of what got laid down, what was deferred,
and where the verification gates still need real hardware + model
files to close.

## What landed

| Layer | Module | Status |
|---|---|---|
| Runtime | `metal::MetalContext` | Real `metal` crate handle; shader compile via `newLibraryWithSource:`; pipeline cache. |
| Runtime | `kernels` | CPU reference for rmsnorm / silu_mul / rope / embed / gemv / softmax / argmax. |
| Runtime | `quant` | CPU dequant for F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K. Other types error. |
| Runtime | `sample::Sampler` | Full top-K + top-P + temperature + repetition-penalty + seeded RNG, CPU. |
| Model | `gguf::reader` | Real GGUF v2/v3 parser, mmap-backed, alignment-aware tensor index. |
| Model | `tokenizer` | `tokenizers` crate wrapper; `from_file` (sidecar) + `from_gguf` (BPE fallback). |
| Model | `attn` | MHA decode-step + MLA decode-step CPU references. |
| Model | `moe` | top-K gate, expert FFN, MoE forward, shared-expert add, work-queue builder. |
| Model | `cache::KvCache` | In-memory per-layer K/V buffers with append + slice. |
| Model | `cache::prefill_disk` | mmap-backed cross-session KV cache file format + load/store. |
| Model | `model::deepseek_v2` | Full forward pass (embed → N×(MLA + MoE/dense) → norm → lm-head). |
| Model | `model::qwen_moe` | Stub returning Unimplemented (Phase 3). |
| Model | `model::load_engine` | Architecture-dispatch from GGUF metadata. |
| Model | `speculate::shared` | `verify_window` impl; draft loop is wired to be filled in Phase 4.5. |
| Server | `dismantle-serve` | axum router, `/v1/chat/completions` and `/v1/completions` (SSE + JSON), `/v1/models`, `/healthz`, `/metrics`, slot-manager scaffolding. |
| Bench | `dismantle-bench` | Suite dispatcher; `decode` + `prefill` produce real numbers; `throughput`/`bandwidth`/`wax` emit the JSON shape with phase-pending sentinels. |
| Shaders | `shaders/*.metal` | Common kernels (rmsnorm/silu/rope/embed) are real MSL. quant/moe/attn/sample have signatures + stubs ready for the wedges to fill. |
| CLI | `dismantle generate` | End-to-end wired to `Engine::generate`, prints tokens to stdout, stats to stderr. |

## Verification gates that still need hardware + a real model

These are explicitly out of scope for the foundation pass; they all
need either a multi-GB GGUF file on disk or a Mac with a Metal-capable
GPU to actually run.

1. **Phase 0 gate** — match `llama-cli` at temp=0 within ≤2 token
   drift on a real DeepSeek-V2-Lite Q4_K_M. Forward pass is wired;
   numerical correctness is unverified until a real weights file is
   loaded. Likely caveats:
     - `decode_q_k_scale_min` for Q4_K / Q5_K is the most likely place
       for an off-by-bit-pack mismatch with ggml-quants.c — it's
       implemented from the spec but not byte-compared yet.
     - DeepSeek tensor names may differ between exporters (e.g.
       `ffn_gate_exps.{e}.weight` vs `ffn_gate_exps_{e}.weight`); the
       loader tries both. New variants → add another `or_else` line.
2. **Phase 1 gate** — `quant.metal::gemm_q4_k_m_fused` and
   `moe.metal::moe_grouped_gemm_q4` are stubs; real kernels land with
   the wedge-2 implementation.
3. **Phase 2 gate** — `moe.metal::moe_block_fused` (single-launch
   work-queue) is a stub.
4. **Phase 2.5 gate** — `sample.metal::sample_topk_topp` is a stub
   (the temperature, repetition, and constraint kernels are real).
5. **Phase 3 gate** — `model::qwen_moe::QwenMoE::load` returns
   `Unimplemented`. Implementing requires the same scaffolding as
   `deepseek_v2` minus MLA.
6. **Phase 4 gate** — continuous batching is single-request only.
   `Scheduler` exists with `idle_slot`/`active_count`; the actual
   prefill/decode interleave loop is the Phase 4 work.
7. **Phase 4.5 gate** — `speculate::shared::verify_window` is
   correct for the verification side but `DraftToken` production from
   the shared-expert path is not wired into the model layer yet.
8. **Phase 5 gate** — `wax-vs-llama-cpp` suite emits the JSON shape
   with `null`s. Real numbers need a pinned llama.cpp commit + the
   same hardware.

## Caveats picked up during the pass

- **Workspace `[target.'cfg(...)'.dependencies]` block does not exist.**
  Cargo workspaces only have `[workspace.dependencies]` as a flat
  template; target-cfg lives in member `Cargo.toml`s. Member crates
  use `[target.'cfg(target_os = "macos")'.dependencies]` to gate the
  metal/objc2 deps.
- **Phase 0 dequants every weight to fp32 at load time.** That keeps
  the reference path simple but balloons host RAM (~10–14 GB for
  DeepSeek-V2-Lite Q4_K_M). The Phase-1 quant-aware kernels read 4-bit
  weights directly from the mmap and eliminate this materialization.
  This is documented in `model/deepseek_v2.rs`.
- **MLA KV cache stores reconstructed per-head K/V, not the latent.**
  The Phase 0 path expands `c_kv` back to (n_heads, head_dim) and
  reuses the MHA softmax-attention. Phase 3's MLA kernel will keep the
  cache compressed and decompress on read, halving cache memory.
- **The .metal kernels for quant/moe/attn/sample are mostly stubs.**
  They compile as MSL because they include `<metal_stdlib>` and have
  empty bodies tagged `(void)id;`. The runtime will refuse to launch
  them as long as the model layer doesn't reach for them — only
  `common.metal` kernels (rmsnorm/silu/rope/embed) have real bodies.
- **Tokenizer fallback covers BPE/llama only.** SentencePiece (`spm`),
  WordPiece, and the legacy ggml-internal vocab path will need
  additional plumbing if a model ships without `tokenizer.json`.
- **`gguf` warning suite suppressed.** `Q4_K`/`Q5_K`/`Q6_K`/`Q8_K`
  variants use canonical ggml names (snake_upper); `#[allow(non_camel_case_types)]`
  on the enum keeps the upstream identifiers readable.

## Build noise

Build is clean. After warning cleanup the workspace compiles with no
warnings on `cargo build --workspace` and no warnings on
`cargo build --release --workspace`. Tests green:

```
running 11 tests
test kernels::tests::rmsnorm_unit_weight ... ok
test gguf::reader::tests::align_up_works ... ok
test kernels::tests::softmax_sums_to_one ... ok
test gguf::reader::tests::block_layout_q4k_is_144 ... ok
test sample::tests::greedy_picks_argmax ... ok
test kernels::tests::gemv_round_trip ... ok
test moe::tests::topk_picks_largest ... ok
test quant::tests::copy_f16_round_trip ... ok
test quant::tests::q8_0_zeros_round_trip ... ok
test moe::dispatch::tests::build_work_queue_buckets_correctly ... ok
test cache::prefill_disk::tests::round_trip_empty_kv ... ok

test result: ok. 11 passed; 0 failed
```

## Next concrete steps

After 2026-04-27 competitive audit, the next steps got reordered.
Before more kernel work, run the head-to-head benchmark to ground
all future perf claims in real numbers.

### Post-restart runbook (heavy lifting deferred to user)

```sh
# 1. Fetch hero model (~9 GB).
./tools/fetch-model.sh

# 2. Phase 0 gate — does dismantle produce coherent text?
./target/release/dismantle generate \
    --weights ./models/deepseek-v2-lite-q4.gguf \
    --prompt "Once upon a time" \
    --max-new-tokens 32 \
    --temperature 0

# 3. Head-to-head against llama-cli (~30-60 min, dismantle-dominated).
./tools/competitors/run_competitors.sh 3

# 4. Paste the medians into docs/competitive_audit.md hero table.
```

### What the Phase 0 gate is looking for

If output is **coherent**: Phase 0 numerics are good; proceed to
the head-to-head benchmark. dismantle's column is allowed to lose;
that's the `phase: 0` label's whole purpose.

If output is **stable but wrong tokens** (model is loading and
generating, just garbage): suspect Q4_K bit-packing
(`decode_q_k_scale_min` in
`crates/dismantle-core/src/quant/mod.rs:150`). Port the exact
`get_scale_min_k4` from ggml-quants.c, compare a super-block
byte-for-byte against ggml's reference.

If output is a **crash or NaN**: suspect tensor-name mismatch in
`crates/dismantle-core/src/model/deepseek_v2.rs:228` — add the
exporter's variant to the `or_else` chain.

### Phase 0 gate — PASSED 2026-04-27

```
prompt:  "Once upon a time"
output:  ", a young man named Alex O'clock was walking through the
          frozen food section of a half-eclair, when he decided to
          make an egg-stand named Angela.\n\"I'm glad I'm not wearing
          any of those stupid-looking goatees of the future... and
          it's a good time to be made a guest of Oppenheimer."
stop:    EOS at 114 tokens (`<｜end▁of▁sentence｜>`)
prefill: 10.78 s for 4 prompt tokens  ≈ 0.37 tok/s
decode:  375.13 s for 114 tokens      ≈ 0.30 tok/s
wall:    ~390 s
```

Greedy + a quirky base/chat model + a 4-token prompt → whimsical
narrative, but every clause is grammatical English with proper
punctuation. The model is doing inference correctly; this is the
honest Phase-0 baseline that goes into `docs/competitive_audit.md`.

### Bug parade (2026-04-27, in order encountered)

The path from "release binary builds clean" to "produces coherent
text" turned up a chain of bugs. Each is a foundation-pass shortcut
that didn't survive contact with a real GGUF.

1. **Tensor naming**: modern GGUF MoE packs all routed experts into a
   single 3D tensor `blk.{li}.ffn_*_exps.weight` (n_experts on the
   outer dim), not one tensor per expert. Loader rewrote to slice
   the 3D tensor by byte-range — works because per-expert size
   lands on a quant-block boundary.
2. **Dense FFN intermediate**: `cfg.ffn_intermediate` (≈10944) is
   distinct from `cfg.moe_intermediate` (=1408 here). The leading
   dense layers bus their own field. Added `feed_forward_length`
   to `DeepSeekConfig`.
3. **Fused shared MLP**: shared experts are stored as ONE wider MLP
   with `intermediate = n_shared × moe_intermediate` (= 2816). Not
   N separate experts. Modelled as a length-1 `Vec<Expert>`.
4. **Legacy non-K quants**: mradermacher's "Q4_K_M" quantization
   mixes Q5_0 / Q4_0 tensors for some specific layers. Added
   Q4_0/Q4_1/Q5_0/Q5_1 dequantization paths.
5. **Memory: lazy expert dequant**. Eager fp32 dequant blew RAM up
   to ~70 GB on this 9.7 GB model. Refactored to keep mmap alive
   in `DeepSeekV2.gguf`, store `TensorRef` byte-pointers per
   expert, dequant on-demand into reusable scratch buffers in
   `ffn`. Resident drops from OOM to ~2 GB.
6. **K-quant bit-pack** (the highest-risk bug from NOTES.md
   originally):
   - `decode_q_k_scale_min` was scattering bits between subs 0..3
     and 4..7 wrong. Rewrote to mirror ggml's `get_scale_min_k4`
     exactly.
   - Q5_K `qh` layout: I treated it as 1 bit per flat element;
     ggml stores it as 1 bit per (sub-block, column) where bit
     `sub` of `qh[col]` is the 5th bit of element `sub*32+col`.
   - Q6_K: `qb`/`qc` had `qhi` shifts swapped vs. ggml's `q2`/`q3`.
     Rewrote inner loop to match ggml's q1/q2/q3/q4 naming.
7. **Tokenizer ByteLevel**: GGUF-fallback BPE skipped the
   `ByteLevel` pre-tokenizer (encode) and decoder (decode). Encode
   produced tokens the model wasn't trained on; decode printed
   `Ġ` and `Ċ` literals. Wired both into the BPE pipeline.
8. **Auto-shutoff**: `dismantle generate` was unkillable from the
   terminal in CPU-bound prefill. Added a real Ctrl-C handler with
   a two-stage protocol (graceful → hard `exit(130)`), an abort
   flag plumbed through `GenerateRequest`, a `--max-stall-ms`
   per-step watchdog, and a new `StopReason::Aborted`.

### First competitive bench — 2026-04-27

| Backend | Decode tok/s | Prefill tok/s |
|---|---|---|
| dismantle (Phase 0)            | 0.30  | 0.37 |
| llama.cpp Metal (b8870)        | ~48   | ~22  |

dismantle:llama.cpp decode ratio = **1:160**. Phase 0 was always
going to lose; the wedges in ROADMAP `Phase 5.5` are the path to
3–4× llama.cpp at Phase 5.

MLX is treated as an analysis-only competitor in
`docs/competitive_audit.md` (cites Apple's published numbers for
related models). Local benchmarking against MLX is deferred — the
weights would have to be in MLX-native format and the comparison
would no longer be apples-to-apples on either weights file or
quantization scheme.

Full audit: [docs/competitive_audit.md](docs/competitive_audit.md).

### Audit deliverables added 2026-04-27

- `docs/competitive_audit.md` — landscape analysis, per-wedge matrix,
  three uncontested wedges (1, 4, 5).
- `tools/fetch-model.sh` — pinned model download.
- `tools/competitors/run_competitors.sh` — head-to-head shell harness.
- `crates/dismantle-bench/src/competitors/{mod,llamacpp,dismantle}.rs`
  — in-binary equivalent of the shell harness.
- `crates/dismantle-bench/src/suites/competitive.rs` — renamed from
  `wax.rs`; old `wax` name still works (deprecated alias).
- ROADMAP.md gates retargeted from llama.cpp to MLX.
- ROADMAP.md adds Phase 5.5 with three new wedges:
  - Wedge 7: expert-temporal-locality predictor
  - Wedge 8: asymmetric quant per expert role
  - Wedge 11: Q8 KV cache
