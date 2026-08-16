# hawking

A Rust + Metal inference stack for Apple Silicon. The live work is getting two
large models — Qwen3-Coder-Next (Q80) and DeepSeek-V4-Flash (DSV4F) — to generate
on one 96 GB M3 Ultra at complete physical density ≤ 1.5 bits per weight. The
`hawking` CLI still loads GGUF and some `.gravity` artifacts and serves them over
an OpenAI-compatible HTTP API. The numbers that moved this week come from
in-tree example runtimes, not from that server.

The ledger of record is
[receipts/ascent-2026-08-16/ASCENT_STATE.json](receipts/ascent-2026-08-16/ASCENT_STATE.json).
The machine those receipts name is an Apple M3 Ultra, 60 GPU cores, 96 GB
unified memory. A sequential DRAM-row probe on this box measured 560–647 GB/s
clean; the geometry-sweep run's own control was 495 GB/s dirty. Use the dirty
number with that run.

## What runs today

**Q80 uniform-Q4** (4.259 BPW). Abandoned as a density target; kept as a
correctness reference. 48 layers, 512 experts, top-10. Hybrid greedy decode of
"Write a function that reverses a string" produced `Here's a simple function in
Python that reverses a string` at 2.479 tok/s (403 ms/token, 11 steady-state
tokens). Label: `DIRTY_ENGINEERING_P1_INVALID_SINGLE_RUN`. A later whole-token
ledger measured 559 ms as a decode mean. Both are dirty. They disagree. A clean
paired Q80 baseline is still owed.
([Q80_BASELINE_2026_08_16.json](receipts/ascent-2026-08-16/Q80_BASELINE_2026_08_16.json))

**Q80 mixed ≤1.5** (`mixed-1p5-v1`). On-disk complete physical BPW 1.44445,
13.4 GiB, 74,391 tensors. Same prompt, twelve greedy tokens, six identical
reps: `Here’s a function that reverses a string (i.e`. First token matches the
Q4 vehicle; the rest diverges and is still English that answers the prompt.
Numeric drift vs the artifact oracle 3.58e-7. No dense `W`. Median 1,249
ms/token (0.80 tok/s) — slower than the abandoned Q4 vehicle.
`fallbacks.host_sample = 28`. The density gate is earned. BASE_TRUE_TPS is not.
([Q80_PACK.json](receipts/ascent-2026-08-16/Q80_PACK.json),
[Q80_MIXED_GENERATE.json](receipts/ascent-2026-08-16/Q80_MIXED_GENERATE.json))

A 4-layer drift probe had extrapolated 16,211× error at depth 48. A later
40-of-48-layer measurement (the lane timed out; layers 40–47 were not run)
found geometric growth 1.035/layer and last-token rel-L2 0.623 at L39, sitting
under a matched-magnitude null. That is consistent with the short coherent
generation. It is not a capability suite.

**DSV4F native 43-layer token graph** on the streamed ~150 GB source (the
source does not fit resident on 96 GB). Host-wall authority, six GPU-locked
reps, cold discarded: 1,205 ms/token wall (0.83 tok/s), 1,038 ms body, 399 ms
Metal GPU, 137 command buffers, `hc_sha c94da765`, 0 fallbacks. After composing
no-copy expert binds, command-buffer collapse, and simdgroup KV QAT: body 830
ms median, still bit-identical. Several later host and GPU wins exist only as
isolated measurements and are not on this composed token.
([DSV4F_HOST_WALL_BASELINE.json](receipts/ascent-2026-08-16/DSV4F_HOST_WALL_BASELINE.json))

**Packed matvec, isolated organ, not a token.** A geometry sweep took the Q4
organ 209,250 → 6,709 ns (31.2×, 83 GB/s) and binary 60,459 → 6,750 ns (9.0×,
22 GB/s). Numeric gates held (binary 1.62e-5, Q4 1.00e-5, tol 2e-5). Zero
fallbacks. Register pressure was not the limiter: every survivor reported
`max_total_threads_per_threadgroup=1024`. Against MLX 0.32.0 incremental cost
on the same 512×2048 Q80 shape (~3,694 ns/organ), Hawking is now ~1.8×. The
earlier "230× occupancy gap" compared that 0.59 MiB organ to a 64 MiB DRAM-row
probe; MLX itself only does 2.1–2.4 GB/s isolated on the Q80 shape. These
kernel wins have not been re-measured on a full mixed-generate token.
`down_proj` hgravs was not re-swept.
([matvec-geometry-sweep.json](receipts/ascent-2026-08-16/matvec-geometry-sweep.json),
[matvec-mlx-reference.json](receipts/ascent-2026-08-16/matvec-mlx-reference.json))

**Qwen3.8-27B** is on disk (54.74 GB download). Census: dense, not MoE; hybrid
linear/full attention (48+16); multimodal. No token-ns. No runtime.

The binaries that produced the Q80 and DSV4F token numbers are
`ascension_qwen80_uniform_q4_hybrid_greedy`,
`ascension_qwen80_mixed_hybrid_greedy`, and
`gravity_deepseek_v4_native_token_graph`.

## Two crate families

22 crates under `crates/`.

`hawking-*` is the inference engine and the HCLI substrate around it.
`hawking-core` owns the Metal kernels, the GGUF loaders (Qwen dense/MoE, Llama,
DeepSeek-V2, Mixtral, RWKV-7), the Gravity loaders (Llama / GLM / DeepSeek /
Mixtral `.gravity` artifacts), and the Q80/DSV4F campaign runtimes. `hawking`
is the CLI. `hawking-serve` is the HTTP surface (`/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, `/healthz`, `/metrics`).
`hawking-speculate` and `hawking-bench` sit on that path. The remaining
`hawking-*` crates (context, index, orch, research, eval, events, adapters,
perception, comms) are HIDE/HCLI support: some execute, perception is stubs,
comms is a sealed-packet format with no live KV transfer.

`hide-*` is HIDE, a local agent IDE. Eight crates: `hide-core`, `hide-backend`,
`hide-kernel`, `hide-serve`, `hide-protocol`, `hide-fleet`, `hide-gateway`,
`hide-acp`. `hide-backend` supervises `hawking serve` as a child over HTTP and
has no Rust dependency on `hawking-core`. `hide-kernel` does depend on
`hawking-context` and `hawking-index`. These crates are workspace members, not
default-members; they compile in their own CI job. `app/` is the React
front-end (typecheck, test, and a live-transport production build are gated).
The Tauri v2 shell under `app/src-tauri` is a scaffold. There is no compile
target on disk.

## Measured vs target

50 tok/s (20 ms/token) on both Q80 and DSV4F is the session floor. 100
BASE_TRUE_TPS is the tournament entry gate. 333 is TG3. Current dirty
full-token figures are 0.80 tok/s (mixed Q80), 2.48 tok/s (Q4 Q80,
single-run), and 0.83 tok/s (DSV4F wall). Nothing measured is closer than 20×
off the floor.

The physical floor for Q80 at 1.392 BPW is 757 µs/token if the box sustained
peak bandwidth. The Q4 vehicle sits 532× above that floor. Those ceilings are
arithmetic, not measurements, and they assume near-unity efficiency. Isolated
packed Q4 now does 83 GB/s. A full token does not.

A 588-recipe screen said NO_GO for sub-0.655 BPW against a 0.8604 organ-cosine
bar. The same 1.44445 artifact then generated the text above with `down_proj`
holdout cosine 0.7684, so that bar predicted failure where generation worked.
Where the capability cliff sits is unproven.

## What is not true yet

- Neither contender is near 50 tok/s. Mixed Q80 is slower than abandoned Q4.
- The mixed artifact is not tournament-valid: host sample fallbacks, no
  BASE_TRUE_TPS, no capability contract, no HCLI, no restart receipt.
- Occupancy and geometry wins are isolated organs. They are not a measured token.
- Several Q80 GPU wins (simdgroup mixer, device top-k, first-touch bind) are
  measured and still masked by host `moe_table_build` / expert first-touch on
  the Q4 vehicle.
- DSV4F ≤1.5 is arithmetic, not a packed artifact. Source experts are already
  16-level FP4; Q80 rates may not transfer. Determined teacher-X capture is
  the blocker.
- Q30 is abandoned (≤1.5 failed coherence). Qwen3.8 has no decode path.
- `hawking serve` is not the Q80 mixed path and not the DSV4F native path.
  `hawking gravity serve` covers Llama / GLM / DeepSeek / Mixtral artifacts,
  not the mixed-1p5 catalog.
- No adapter family is PRODUCTION
  ([families.md](crates/hawking-adapters/goldens/families.md)). The registry
  describes Llama GGUF as CPU-only and degenerate against llama.cpp. Gemma,
  Phi, and Mamba2 live in a pack that is not in this tree.
- `hawking gravity plan` / `press` is metadata-only; it does not bake.
  `BakeSidecar` prints a plan and writes nothing. `profiles/` is gone.
- Ramanujan is a fixture-only scaffold, blocked on Hawking completion
  ([HAWKING_COMPLETION_GATE.json](ramanujan/governance/boundary/HAWKING_COMPLETION_GATE.json)).
  Five small CPU classifiers were trained under non-production authority:
  retriever and value beat their baselines; formalizer, prover, and repair did
  not beat majority class. Odyssey here is inventory and membership tooling,
  not a launched continued-training run.
- CI is defined (fmt / clippy / unit tests on macos-14, plus a hide job and a
  frontend job). Whether the last push is green is not verified from this
  checkout. `hawking-core/tests` holds 157 Rust files; many Metal and
  model-weight gates are `#[ignore]`. Default CI runs `--lib --bins`.

## Next

Compose the organ-level matvec winners into a full mixed-generate token and
re-measure. Land the held DSV4F residency and occupancy patches and re-measure
the composed token, not the parts. Recalibrate the subbit bar from generation
rather than organ cosine. Qwen3.8 still has no token. 50 tok/s remains the
floor. It has not been approached.

MIT. Joshua Hicks.
