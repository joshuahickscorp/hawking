# Path-to-50 — one-shot self-contained prompt

**How to use:** copy everything below the next `---` divider and paste
into a fresh Claude Code session opened in
`/Users/scammermike/Downloads/dismantle`. Everything that session needs
to execute the full path-to-50 workflow is inline below — the three
lever specs, the orchestration, the constraints, the done conditions.
No need to read other files in `reports/` to get started.

---

You are picking up an in-progress effort to take dismantle (Apple Silicon
MoE inference engine for DeepSeek-V2-Lite-Chat) from **~27 dec_tps to
~50 dec_tps**. The corpus build and calibration analysis are already
done. Your job is to execute the path-to-50 workflow end-to-end.

## 0. State of the world

- **Current production tps:** ~26.87 dec_tps (clean-bench, 2026-05-20).
- **What's on disk:**
  - `artifacts/calibration/v2_lite_corpus/shard_*.parquet` — 141
    shards / 4512 sequences / 256 tokens each (note: ~1500-2000 unique
    sequences; the rest are duplicates from `iter_chat_sequences`
    restarting at row 0 on each watchdog launch — calibration uses
    are fine; dedupe-quality work would need a re-capture)
  - `artifacts/calibration/analysis/vocab_freq.json` (top-5000 token frequencies)
  - `artifacts/calibration/analysis/vocab_whitelist_995.json` (full 23,628-id whitelist)
  - `artifacts/calibration/analysis/per_layer_residual_stats.json` (mean_abs, abs_max, abs_p99, p99.9, int8_scale per layer for all 27 layers)
  - `artifacts/calibration/analysis/expert_load_per_layer.json` (per-MoE-layer per-expert routing freq + balance scores)
  - `artifacts/calibration/analysis/summary.md` (the digest)
- **Foundation Rust module already in:**
  `crates/dismantle-core/src/vocab_prune.rs` (260 lines, 11/11 tests
  pass). Public API: `PrunedVocab::load`, `slice_lm_head_f16`,
  `pruned_to_original`, `original_to_pruned`, `validate`. NOT yet
  called from anywhere — sits dormant pending wiring.

## 1. Required first read (project memory)

Before doing anything, read:

- `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/corpus_complete_analysis_landed.md` (today's wrap)
- `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/path_to_100_repath.md` (why 100 is out of scope here)
- `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/v110_path30_findings.md` (LM head is ~4% of decode-time; MoE GEMMs dominate)
- `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/bench_contamination.md` and `feedback_bench_with_claude_open.md` (bench discipline)
- `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/feedback_kernel_parity_gate.md` (bit-identical 3-token greedy gold standard)

## 2. The three levers and why exactly three

The plan stacks three independent levers to take 27 → 50:

| # | Lever | Status | Projected gain |
|---|---|---|---|
| 1 | **Vocab-prune** | foundation landed, impl-ready spec below (§3) | +1-3 tps |
| 2 | **Mixed-precision quant** | analysis ready; design needed first (§4) | +5-10 tps |
| 3 | **Eagle5 v2 (activation sparsity)** | nothing yet; design needed first (§5) | +5-12 tps |

**Why not four:** Q8-KV layer-differential is **cancelled** per the
analysis. Routing balance scores 0.987-0.995 across all MoE layers
means no signal to drive per-layer KV precision tuning. Stick with
uniform Q8 (already shipped).

Stacked best-case: ~52 tps. Midpoint estimate: ~44 tps. Hitting 50
needs all three landing at upper-mid; not guaranteed but achievable.

**Why not 100:** 100 requires a working spec-decode runtime (the Rust
half that turns offline acceptance into wall-clock). Per memory
`path_to_100_repath.md`, "all spec-decode REGRESSES" as of 2026-05-20.
Recovering that runtime is a SEPARATE workstream gated on a trained
head (Eagle5 v2 produces one). Re-baseline after 50 and decide then.

## 3. LEVER 1 — Vocab-prune wiring (impl-ready)

The foundation (`crates/dismantle-core/src/vocab_prune.rs`) is landed
and tested. Wire it into the LM-head load path and the sampler so the
live decode produces correct tokens against a smaller LM head.

### 3.1 Context

- Corpus calibration found 23,628 of 102,400 vocab tokens cover 99.5%
  of the corpus. Slicing the LM head 102.4k → 23.6k is a 76.9%
  reduction in the output GEMV + argmax. LM head is ~4% of
  decode-time per `v1.1.0` findings → +1-3 tps standalone, +2-5 tps
  per the path-to-30 plan.
- **What's already done:** `PrunedVocab::load(path)`,
  `slice_lm_head_f16(orig, hidden)`, `pruned_to_original(pruned_id)`,
  `original_to_pruned(orig_id)`, `validate(tokenizer_vocab_size)`, full
  test suite. Re-read `crates/dismantle-core/src/vocab_prune.rs` —
  module-level docs spell out the contract.
- **Whitelist file:** `artifacts/calibration/analysis/vocab_whitelist_995.json` (23,628 ids).
- **Model in scope:** DeepSeek-V2-Lite-Chat ONLY. Has explicit
  `output.weight` (NOT tied). See `crates/dismantle-core/src/model/deepseek_v2.rs`
  line ~494: `let lm_head = if gguf.tensor("output.weight").is_some()`.
  Other models (Qwen3 dense in `qwen_dense.rs`) should be left alone in
  this PR — guard the prune path with a profile flag.

### 3.2 Tasks (in order)

**Task 3.2.1 — Profile config (opt-in switch)**

File: `crates/dismantle-core/src/profile.rs`

Add a new field to `Profile` (~line 50):

```rust
/// Path to a vocab whitelist JSON (see vocab_prune module). When set,
/// the LM head is sliced to the pruned vocab at GGUF load time.
/// None ⇒ full vocab (current behavior).
#[serde(default)]
pub vocab_prune_path: Option<std::path::PathBuf>,
```

Add default `None` in `Profile::default()` (line ~265). Existing
profile JSON files must still parse (`#[serde(default)]` handles it).

**Task 3.2.2 — Load + slice at model construction**

File: `crates/dismantle-core/src/model/deepseek_v2.rs`

a. Load the whitelist once at model construction. Suggested spot:
   just before the existing `let lm_head = if gguf.tensor("output.weight").is_some() {…}`
   (~line 494). Reach the profile via the existing config plumbing —
   trace from `Engine::new`; pass the path through `Model::load_*` if needed.

b. Slice the weight right after the existing `dequant_f16("output.weight")`:

```rust
let mut lm_head = Self::dequant_f16(&gguf, "output.weight")?;
let mut effective_vocab_size = cfg.vocab_size;
if let Some(pv) = &pruned_vocab {  // Option<PrunedVocab>, loaded above
    pv.validate(cfg.vocab_size)?;  // fail fast on mismatch
    lm_head = pv.slice_lm_head_f16(&lm_head, cfg.hidden_size)?;
    effective_vocab_size = pv.pruned_len();
}
let lm_head = Some(lm_head);
```

c. Override `cfg.vocab_size` with `effective_vocab_size` BEFORE any
   downstream allocation (logits_buf, argmax dispatch). The pinned
   `lm_head_buf` (line ~671) and the `logits_buf` allocation (line ~835:
   `ctx.new_buffer(cfg.vocab_size * std::mem::size_of::<f32>())`) both
   read from `cfg.vocab_size` — make sure they see the pruned value.

d. Hold the `PrunedVocab` on the `Model` struct (~line 118-150):

```rust
pub pruned_vocab: Option<PrunedVocab>,
```

**Task 3.2.3 — Sampler translation**

Find where the argmax kernel's output token id is consumed:

```sh
grep -rn "argmax\|sample_token" crates/dismantle-core/src/sample crates/dismantle-core/src/model
```

The argmax kernel returns `pruned_id ∈ 0..effective_vocab_size`. Before
that id is handed back to the streaming layer / detokenizer:

```rust
let token_id = match &model.pruned_vocab {
    Some(pv) => pv.pruned_to_original(pruned_id),
    None => pruned_id,
};
```

Audit all sampling paths (greedy + top-k + temperature). They all
operate on logits indices in pruned space and need the translation at
the end.

**Task 3.2.4 — Parity test**

File: new `crates/dismantle-core/tests/vocab_prune_parity.rs`

Acceptance gate: **with the same prompt and a fixed seed, the pruned
model must produce the same token sequence as the unpruned model**,
provided every emitted token is in the whitelist. Steps:

1. Load V2-Lite GGUF unpruned, generate 64 tokens from a fixed prompt
   (use whatever fixture `q8_kv_parity.rs` uses).
2. Verify all 64 emitted tokens are in the whitelist (read it from the
   JSON; if any are not, choose a different prompt).
3. Load V2-Lite GGUF pruned, generate from same prompt + seed, compare
   token-by-token. Must be bit-identical.

Skip on non-Mac in CI (Metal-only). Follow the pattern existing parity
tests use.

**Task 3.2.5 — Microbench**

`crates/dismantle-core/src/kernel_bench.rs` already has
`(102400, 2048, "v2_lite_lm_head")`. Add
`(23628, 2048, "v2_lite_lm_head_pruned")` and run via the existing
harness. Expected: ~4× faster GEMV (76.9% fewer FLOPs).

**Task 3.2.6 — End-to-end tps measurement**

Compare `tools/bench/*` on the same prompt with `vocab_prune_path =
None` vs `Some("artifacts/calibration/analysis/vocab_whitelist_995.json")`.
Target: **+1-3 dec_tps minimum**.

### 3.3 What NOT to do (lever 1)

- Don't touch the input embedding. The user's prompt may contain any
  token; embedding lookup must still work for the full original vocab.
- Don't break the tied-embedding code path in `model/qwen_dense.rs`.
  Guard the prune logic on the model being DeepSeek-V2.
- Don't enable pruning by default. Opt-in via profile config until
  parity is proven.
- Don't merge if parity test fails — a divergence even at token 1 is a
  load-path bug.
- Don't micro-optimize the HashMap reverse map. Fine as-is.

### 3.4 Done condition (lever 1)

- `cargo test -p dismantle-core` green
- `vocab_prune_parity` test passes on Mac M-series
- Microbench shows ~4× speedup on `v2_lite_lm_head_pruned`
- End-to-end bench shows ≥+1 dec_tps with the prune profile

## 4. LEVER 2 — Mixed-precision quant (design THEN impl)

This lever has no foundation yet. **Phase A: produce a design doc**
(`reports/mixed_precision_quant_wiring_handoff.md`) matching the style
of §3 above. **Phase B: implement per the design.** Stop and ask the
user to review the design before implementing.

### 4.1 What you have

- `artifacts/calibration/analysis/per_layer_residual_stats.json` — every
  layer's mean_abs, abs_p99, abs_max, suggested int8_scale.
- `artifacts/calibration/analysis/summary.md` — tiered map already
  derived from those stats:
  - **layers 0-3:** abs_max < 24, mean_abs 0.05-0.08 → **q4 candidates**
  - **layers 4-24:** abs_max ~1180 with abs_p99 ≪ abs_max (outlier-channel
    pattern) → **need q8 headroom**
  - **layers 25-26:** abs_max 645-845 → **q6 sufficient**
- `memory/v110_path30_findings.md` — LM head is ~4% of decode-time; MoE
  GEMMs dominate.
- `memory/per_kernel_time_breakdown.md` — MoE GEMMs 50.5%, rmsnorm+add
  24%, attention 2.4%. **The big quant gain is in MoE expert weights**,
  NOT the residual norm projections that dominate quant-paper benchmarks.
- `crates/dismantle-core/src/model/deepseek_v2.rs` — existing weight
  loading. Read end-to-end before designing; specifically the `dequant`,
  `dequant_f16`, and per-layer load loops.
- `crates/dismantle-core/src/quant/` — existing quantization kernels;
  q4_K, q5_0, q6_K, q8_0 are all there.
- `crates/dismantle-core/src/profile.rs` — add the new option here
  following the `vocab_prune_path` pattern from §3.2.1.

### 4.2 What the design doc must specify

1. **The tier table as a config artifact.** Where does the per-layer
   bit-width map live? Inline in profile.rs? A separate JSON?
   Per-layer override pattern. Must support per-layer choice between
   q4_K, q6_K, q8_0.

2. **What to actually quantize.** Analysis was on `residual_in_per_layer`
   (the residual stream BETWEEN layers). Big weight tensors in V2-Lite
   are the **MoE expert weights** (`blk.N.ffn_gate_exps.weight`,
   `blk.N.ffn_up_exps.weight`, `blk.N.ffn_down_exps.weight`). Decide
   whether the tier map applies to:
   - (a) the expert weights of layer N (most decode-time savings, but
     the calibration data is for residual not expert outputs);
   - (b) the attention projection weights of layer N;
   - (c) both;
   - and justify with the analysis data.

3. **Dispatch path.** When layer N is q4 and layer N+1 is q8, which
   kernel runs? Look at the existing per-shape dispatcher (`kernels/mod.rs`
   suffix matcher — source of last week's `_v2t_v3` bug per
   `memory/feedback_kernel_parity_gate.md`). Make sure mixing
   bit-widths per layer doesn't hit a kernel-not-found path.

4. **GGUF reality.** GGUF files are pre-quantized. Can't change weights
   at runtime without dequant+requant. Two design paths:
   - **Path A (runtime re-quant):** load fp16, requantize per layer at
     boot to the tier-map bit-width, cache the quantized blocks. Cost:
     ~30-60 sec extra startup, but no GGUF surgery.
   - **Path B (separate GGUF):** pre-bake a mixed-precision GGUF via a
     Python script that reads the tier map + dequants the source GGUF +
     re-quantizes per layer. Ship the new GGUF.
   Pick one with a tradeoff explanation. Recommendation: Path A for
   v1, Path B if startup time becomes painful.

5. **Acceptance gates.** Per `memory/feedback_kernel_parity_gate.md`,
   bit-identical 3-token greedy is the gold standard. Specify:
   - parity test (mixed-precision vs uniform fp16, on a fixed prompt)
   - per-layer perplexity-on-corpus delta vs baseline (informational)
   - end-to-end dec_tps delta vs baseline
   - target: **+5-10 tps** with **no token divergence in first 256
     tokens** of a held-out test prompt

### 4.3 What NOT to do (lever 2)

- Don't quantize the attention V projection more aggressively than the
  rest until measured — V outliers blow up quant error more than MLP.
- Don't touch the MLA absorbed-V if you go there (see
  `memory/mla-phase4-queued.md` — separate workstream).
- Don't quantize the LM head (vocab-prune handles that).
- Don't re-quant the embedding table — input embeddings are tiny and
  quant error compounds.

### 4.4 Out of scope for this lever

- Eagle5 (lever 3)
- KV cache precision (already at Q8)
- Activation quantization (this is weight quant only)

### 4.5 File-level diff list the design doc should produce

- `crates/dismantle-core/src/profile.rs` — tier map field (~10 LOC)
- `crates/dismantle-core/src/model/deepseek_v2.rs` — per-layer re-quant
  in the load path (~50 LOC)
- `crates/dismantle-core/src/quant/mod.rs` or new module — re-quant
  helpers (dequant fp16 → quant qN_K) (~80 LOC)
- New `crates/dismantle-core/tests/mixed_precision_parity.rs` (~120 LOC)
- New `crates/dismantle-core/src/kernel_bench.rs` entries (~5 LOC)
- Possibly `tools/quant/rebuild_mixed_gguf.py` for Path B (~150 LOC)

### 4.6 Done condition (lever 2)

The design doc lands at `reports/mixed_precision_quant_wiring_handoff.md`,
matches the rigor of §3 above, and the user has reviewed it. Then the
impl phase produces:

- `cargo test -p dismantle-core` green including the new parity test
- No token divergence in first 256 tokens vs uniform fp16
- End-to-end bench shows ≥+5 dec_tps with the mixed-precision profile

Estimated impl-session effort: **2-3 days**.

## 5. LEVER 3 — Eagle5 v2 (activation sparsity) (design THEN impl)

No foundation yet. **Phase A: produce a design doc**
(`reports/eagle5_v2_wiring_handoff.md`). **Phase B: implement.**
Stop and ask the user to review the design before implementing.

### 5.1 Why this approach

The **old eagle5 plan (predict which experts get routed) is dead** per
the calibration analysis — MoE routing balance scores are 0.987-0.995
across all 26 MoE layers (essentially uniform), so there's nothing for
a routing predictor to exploit.

Eagle5 v2 predicts **token-level activation sparsity** instead. For
each MoE layer, the routed experts' FFN output (in
`intermediate_per_layer`) shows which **hidden dimensions** activate
strongly per token. Even with uniform routing, the per-token
per-dimension activation pattern is highly sparse: a small subset of
intermediate channels carry most of the signal at any given token
position.

A head that predicts WHICH channels matter for the next token can:
- Skip dense GEMM ops over inactive channels (sparse decode), OR
- Drive a smaller draft model that only computes the predicted-active
  channels and verifies via the full model

The acceptance metric (τ-at-depth-K) translates this directly into
wall-clock speed-up like in eagle4.

### 5.2 What you have

- **Reference architecture:** `eagle4/eagle4.py` (~500 LOC), the
  trained head + trainer. Read this end-to-end before designing. Also
  `eagle4/ARCHITECTURE.md`, `eagle4/tau_eval.py` (the metric that
  matters), `eagle4/README.md` (with τ-at-depth-K numbers; eagle4
  shipped at τ=3.57).
- **Corpus:** `artifacts/calibration/v2_lite_corpus/shard_*.parquet`
  (141 shards, 4512 sequences). Columns:
  - `tokens` (input_ids)
  - `residual_in_per_layer` (27 layers × n_tok × 2048 hidden) ← training input
  - `expert_idx_per_layer` (26 MoE layers × n_tok × top_k=6)
  - `routing_topk_weight_per_layer` (26 layers × n_tok × top_k)
  - `intermediate_per_layer` (per-MoE-layer first-expert output, used
    as a probe for FFN activations) ← **this is the sparsity signal**
- **Analysis:** `artifacts/calibration/analysis/summary.md` confirms
  routing is uniform; you need sparsity in a different dimension.
- **Memory:** `corpus_complete_analysis_landed.md` documents the
  pivot. `path_to_100_repath.md` documents why the Rust+Metal
  spec-decode runtime is a SEPARATE workstream — your design's output
  is the head + trainer + offline-acceptance-eval; runtime wiring is
  out of scope.

### 5.3 What the design doc must specify

1. **Architecture.** What does the head look like? Eagle4 reads
   `(hidden_low, hidden_mid, hidden_high, shared_hidden)` at fusion
   layers (2, 13, 25). Eagle5 v2 should read residual + the
   intermediate-activation tensor and predict NEXT-token-activation-
   sparsity. Design:
   - Inputs: which layer's residual + which layer's intermediate signal
   - Output head: per-token, per-channel sparsity mask (or top-K
     channel indices) for the next token's MoE FFN
   - Param count budget: ≤ 50M params (eagle4 was ~30M)
   - MLX-or-PyTorch trainer: pick one and justify (eagle4 used MLX;
     PyTorch may be simpler given the corpus is already parquet-numpy)

2. **Loss function.** Two-headed loss:
   - Channel-sparsity prediction: BCE over the (channel-active) mask
     derived from `intermediate_per_layer` ground truth
   - Next-token logits: standard CE against the target token (keeps the
     head a real draft model, not just a sparsity predictor)
   Balance terms; reference eagle4's loss balance in `eagle4.py`.

3. **Training pipeline.** Concrete script names + corpus reader code.
   - Where: probably new `tools/training/eagle5_train.py`
   - Reads from `artifacts/calibration/v2_lite_corpus/shard_*.parquet`
   - Reuses the zero-copy Arrow flatten approach from `analyze_corpus.py`
   - Expected runtime: 5-10 epochs × ~1h/epoch = 5-10h on M3 Pro
   - Checkpoints to `checkpoints/eagle5_v2/`

4. **Eval — the τ-at-depth-K metric.** Specify the eval harness:
   - Greedy rollout of the head autoregressively for K=4 steps
   - Compare each predicted token to V2-Lite's argmax at that position
   - Report τ = mean accepted prefix length (max 4)
   - Acceptance gate: **τ ≥ 3.0** (eagle3 was 2.15, eagle4 was 3.57;
     eagle5 v2 with a different angle should at least beat eagle3)
   - Plus single-step argmax acceptance: ≥ 85% (eagle3 was 73.8%)

5. **Quantization plan.** Eagle4 quantizes its head to q4 for
   deployment (see `eagle4/q4_parity.py`). Plan for the same — q4
   quantized head with parity check vs bf16.

6. **Integration interface.** What does the head expose to the Rust
   runtime? Specify:
   - Input format: residual + intermediate slices
   - Output format: top-K channel ids + their softmax weights
   - GGUF or safetensors export format
   - The Rust runtime wiring is OUT OF SCOPE — but the head's I/O
     contract must be specified clearly enough that runtime wiring can
     be done in a follow-up workstream

### 5.4 Acceptance gates (lever 3)

- Trained head: τ-at-depth-4 ≥ 3.0
- Single-step argmax accept ≥ 85%
- Quantized (q4) head: τ within 5% of bf16 head
- Bit-identical greedy on held-out test set: not required (probabilistic
  head, not a verifier; verifier is V2-Lite itself via spec-decode
  runtime, which is separate)

### 5.5 What NOT to do (lever 3)

- Don't try to predict expert routing. That's the dead angle.
- Don't ignore eagle4. Use it as architectural reference; if designing
  something dramatically different, justify why eagle4's approach
  wouldn't extend trivially.
- Don't design the Rust runtime. Separate workstream. Stay in
  Python/MLX.
- Don't try to capture intermediate activations for ALL experts. Our
  corpus only captures the first expert per MoE layer as a probe (per
  `build_corpus.py` line ~199). Either work with what's there, or spec
  a re-capture (cost: ~3 hours of capture compute).
- Don't gate on absolute tps. Requires runtime wiring. Gate on
  τ-at-depth-K and accept rate.

### 5.6 Out of scope (lever 3)

- Rust+Metal spec-decode runtime (separate doc)
- LM head pruning (lever 1)
- Mixed-precision residual quant (lever 2)
- Any cross-model generalization (V2-Lite only)

### 5.7 Done condition (lever 3)

The design doc lands at `reports/eagle5_v2_wiring_handoff.md`, user
reviews. Then the impl phase produces a trained + quantized head
meeting the acceptance gates in §5.4. Estimated impl effort:
**~1 week** of focused work (design+code+training+iterate). Pure
training compute: ~5-10 hours of M3 Pro wall-clock.

## 6. Execution sequence

```
Phase 1 (sequential design work, this session):
  1. Read §0, §1, §2 of this prompt + the memory files listed in §1.
  2. Produce reports/mixed_precision_quant_wiring_handoff.md from §4.
     STOP and ask user to review.
  3. Produce reports/eagle5_v2_wiring_handoff.md from §5.
     STOP and ask user to review.

Phase 2 (sequential implementation, after both designs approved):
  1. Implement vocab-prune per §3 (1-2 days). Re-baseline dec_tps.
  2. Implement mixed-precision quant per the §4 handoff (2-3 days).
     Re-baseline dec_tps.
  3. Implement eagle5 v2 per the §5 handoff (~1 week, training-heavy).
     Re-baseline dec_tps.

Phase 3 (wrap):
  - Land memory/path_to_50_complete.md summarizing what shipped, what
    each lever actually returned vs projected, and what the new
    bottleneck is for any future path-to-100 attempt.
```

## 7. Bench discipline

- For paired deltas (with-lever vs without-lever), Claude Code can be
  open — contamination cancels per `feedback_bench_with_claude_open.md`.
- For absolute tps numbers (e.g. "did we hit 50?"), ask the user to
  fully quit Claude Code first.
- Always run benches via `tools/bench/*` and force-add reports under
  `reports/`. The repo workflow is the post-prune model
  (`memory/post_prune_operating_model.md`).

## 8. Constraints

- **Don't attribute to Claude in git.** User's global rule. No
  `Co-Authored-By` trailers, no "Generated with Claude" footers.
- **Don't commit autonomously.** Ask for review before each commit.
- **Don't try to do all three implementations in parallel.** They
  share too much codebase surface; sequential lands cleanly.
- **Don't redo the corpus build.** It's at `artifacts/calibration/v2_lite_corpus/`
  with the known duplicate caveat documented. If a lever truly needs a
  deduped corpus, flag it and ask the user.
- **Don't pursue Q8-KV layer-differential.** Cancelled per the analysis.

## 9. Done condition for the whole workflow

- Both Phase-1 handoff docs exist in `reports/` and pass user review.
- All three levers from Phase 2 are wired, tested, and benchmarked.
- End-to-end clean-bench (Claude quit) shows ≥45 dec_tps; 50 is the
  stretch goal. If you can't hit 45 after all three levers, report the
  delta vs projection per lever and ask whether to dig deeper or stop.
- A wrap memo lands at `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/path_to_50_complete.md`
  summarizing what shipped, what each lever actually returned vs
  projected, and what the new bottleneck is.

Estimated wall clock: **~3 weeks** of part-time work. **~15-20 hours
of pure M3 Pro compute** of which eagle5 training dominates (~10h).
Reality multiplier on compute: 1.5-2× with debug iterations.

---

End of one-shot prompt. Begin by reading §1 (project memory).
