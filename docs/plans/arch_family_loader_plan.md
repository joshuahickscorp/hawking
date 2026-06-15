# Architecture-family loader generalization — parameterization plan (§9.1)

**Task D, wave 2026-05-31. PLAN ONLY — no code edited. GPU-free desk study.**
Companion: Throughput Bible `plans/throughput_bible_2026_05_30.md` §9.1 (lines 598–611)
and §9.1.a scope discipline.

---

## 0. TL;DR / verdict

**The loader is already ~80% generalized at the *correctness* layer and ~0%
generalized at the *fast-decode* layer.** This is the single most important
finding and it reframes §9.1 entirely.

- **Config + dispatch + tensor-naming + tokenizer are already metadata-driven
  and multi-arch** (verified by reading source, below). Four dense loaders
  already exist and ship: `qwen_dense.rs`, `llama.rs`, `gemma2.rs`, `phi3.rs`,
  each with its own `<arch>.*`-prefixed `from_gguf`. The `Engine` trait already
  abstracts forward/generate; `model::load_engine` already peeks
  `general.architecture` and dispatches. `profile::arch_family` already folds
  point-releases (`qwen2.5`→`qwen2`, `llama3.2`→`llama`).
- **The gap §9.1 actually names is NOT "the loader is hardcoded to Qwen
  dims."** It is **(a) four near-duplicate loaders that should share one
  parameterized core**, and **(b) the entire optimized Metal decode pipeline
  — TCB pinned buffers, predec scale-tables, Q4K_FAST, vocab-prune,
  Q4K-LM-head, eagle5, prefix-cache — lives ONLY in `qwen_dense.rs`.** The
  other three run a slower CPU/Metal-hybrid `forward_token` path (llama.rs
  docstring lines 16–20 states this explicitly).
- **Decisive measurement gate is BLOCKED on local weights.** §9.1's named
  oracle is "wire a *second family* (Llama-3.x / Mistral GGUF) and greedy-parity
  it vs llama.cpp." **There is NO Llama / Mistral / Gemma / Phi GGUF on this
  machine.** The only non-Qwen GGUF is `deepseek-v2-lite` (9.7 GB, `deepseek2`
  MoE — explicitly out of §9.1.a dense scope, and would blow the 3 GB RSS
  ceiling). So the cross-family parity gate **NEEDS-MEASUREMENT** and the very
  first checklist item is *acquire a small Llama/Mistral Q4_K_M GGUF*.
- **A second *intra-family* shape is already provable today** with zero new
  downloads: Qwen2.5 **1.5B** (28 layers, hidden 1536, 12 heads, 2 KV) and
  **7B** (28 layers, hidden 3584, 28 heads, 4 KV, head_dim 128) are present and
  exercise different layer counts / head counts / head_dim than 3B (36 layers,
  hidden 2048, 16 heads, 2 KV, head_dim 128). This proves *dim-parameterization*
  but NOT *family-prefix / structural* generalization. It is the cheap
  pre-flight, not the §9.1 oracle.

**Recommendation: do §9.1 as two separately-gated sub-moves.**
- **D-1 "unify the config reader"** (low risk, no behavior change, parity-
  provable on Qwen variants today): collapse the four `<arch>.*` `from_gguf`
  blocks into one prefix-parameterized `ArchConfig::from_gguf(prefix, gguf)`.
- **D-2 "port the fast decode path to a second family"** (the real breadth
  win; blocked on a non-Qwen GGUF): make `forward_token_greedy_tcb` consume a
  small `ArchSpec` (norm placement, activation, bias presence, RoPE variant,
  logit softcap) so llama/gemma/phi can route through the TCB pipeline instead
  of the CPU-hybrid fallback.

Do **not** record any Kill here — nothing died. This is a build plan with a
named, currently-unavailable measurement gate.

---

## 1. What is already generic (verified in source — do NOT re-build)

| Concern | Where | Status |
|---|---|---|
| GGUF parse (metadata→typed `HashMap`, tensor index, mmap-nocopy) | `gguf/reader.rs` (whole file) | **fully arch-agnostic.** `MetaValue` + `as_u32/as_f32/as_str/...` accessors; no arch strings anywhere. Nothing to do. |
| Arch dispatch by `general.architecture` | `model/mod.rs:55–94` `load_engine` | **already multi-arch.** Maps `qwen2/qwen2.5/qwen` → `QwenDense`, `llama/llama2/3/3.1/3.2/mistral` → `LlamaDense`, `gemma2`, `phi3`, `deepseek2`, `qwen2moe`. |
| Point-release folding for profile match | `profile.rs:22–30` `arch_family` | **done.** `qwen2.5`→`qwen2`, `llama3.2`→`llama`, etc. Tested (`profile.rs:470–486`). |
| Tensor names (`blk.{li}.attn_q.weight`, `token_embd.weight`, `output_norm.weight`, `output.weight`) | `qwen_dense.rs:794–823` | **GGUF-canonical, NOT Qwen-specific.** Every llama.cpp-produced GGUF (llama/mistral/gemma/phi) uses the same `blk.{i}.{suffix}` scheme. The same `lp` closure works for all dense families. |
| Tie-vs-untied LM head | `qwen_dense.rs:798–802` | **already handled generically:** `if gguf.tensor("output.weight").is_some()` → own head, else tie to embed. Family-independent. |
| Q/K/V bias presence | `qwen_dense.rs:817–819` (`dequant_f32_opt(...).unwrap_or_default()`) + forward guard `if !layer.q_bias.is_empty()` (`:603–611`) | **already optional.** Qwen carries them, Llama omits them; the loader reads `_opt` and the forward path no-ops on empty. This is *already* a parameterized structural difference. |
| GQA (n_heads ≠ n_kv_heads) | `qwen_dense.rs:47–48,555–642`; `mha_decode_step(n_heads, n_kv_heads, …)` | **already parameterized** from metadata. |
| Per-arch config from metadata | `qwen_dense.rs:33–86`, `llama.rs:67–`, `gemma2.rs:72–`, `phi3.rs` | **done per-arch, but duplicated 4×.** Each reads `<arch>.block_count` / `.embedding_length` / `.attention.head_count[_kv]` / `.feed_forward_length` / `.context_length` / `.rope.freq_base` / `.attention.layer_norm_rms_epsilon`, with identical fallback logic. **This duplication is D-1's target.** |
| Structural variants already implemented per-arch | gemma2 (sandwich norms + GeGLU + logit softcap + √hidden embed-scale, `gemma2.rs:7–13,159–161,358–399`); llama (NTK RoPE `Llama3RopeScaling`, no bias, θ=500k, `llama.rs:8–14,29–31,59`); qwen (Q/K/V bias, θ=1e6) | **the four structural shapes are already each handled** — just in four separate files. |

**Conclusion of §1:** the claim "dismantle is hardcoded to Qwen2.5-3B dims" is
**false at the dimension level** — every dim is read from GGUF metadata. The
real targets are duplication (D-1) and fast-path coverage (D-2).

---

## 2. What is hardcoded / Qwen-specific (the actual surface to parameterize)

Ordered by how arch-specific each truly is.

### 2.a Metadata-key *prefix* string (cheap; D-1)
`QwenConfig::from_gguf` hardcodes the literal `"qwen2."` in every key
(`qwen_dense.rs:38–83`). The *values* are read from the file — only the prefix
is baked. llama/gemma/phi each repeat the same pattern with their own literal.
**Generalize:** one `from_gguf(prefix: &str, g: &GgufFile)` that formats keys as
`{prefix}.block_count` etc. The arch→prefix map already exists implicitly in
`load_engine`; surface it as `arch_meta_prefix(arch) -> &str`.

### 2.b The fast-decode pipeline is Qwen-only (expensive; D-2 — the real breadth move)
Everything below lives in `qwen_dense.rs` and has **no llama/gemma/phi twin**:
- `forward_token_greedy_tcb` + `forward_tokens_batch_tcb` (the TCB pinned-buffer
  decode/prefill path — the reason clean dec_tps is ~29–31 (measured, bible
  §3 Corr.2), vs the CPU-hybrid fallback the other families use).
- `weights_mmap_buf` whole-mmap pin + `gemv_q4_k_m_v*_pinned_tcb` windowed reads.
- Predec scale tables: `q4k_predec_cache`, `…_f16`, `lm_head_pruned_predec`,
  `ensure_q4k_predec_cache[_f16]` (default-on, bible-confirmed +8–12%).
- `q4k_fast_buf` sidecar path.
- Q4K-LM-head (`lm_head_q4k_buf`) + vocab-prune (`lm_head_pruned_buf`).
- eagle5 head + capture (`eagle5_*`), prefix cache (`ram_prefix_cache`), FFN
  capture (`ffn_capture`).
llama.rs:16–20 confirms its path "runs Q4_K projections, f16 LM head, rmsnorm on
Metal; the rest (Q6_K weights, attention) on the CPU reference path. The full
TCB pinned-buffer + predec arena is a follow-up best done with a real GGUF in
hand to bench against." **That follow-up *is* D-2.**

### 2.c vocab-prune "first-N" heuristic — Qwen-BPE-specific (structural caveat)
`qwen_dense.rs:206–223,989–1035`: the legacy `DISMANTLE_QWEN_VOCAB_PRUNE=N`
fast path assumes "the first N IDs of Qwen's BPE vocab are the most frequent by
construction," so `pruned_idx ≡ original_id` with no remap. **This is a Qwen
tokenizer fact and will silently mis-rank on a SentencePiece/Llama vocab.** The
corpus-derived path (`DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS=N`, builds an explicit
`vocab_prune_remap`) is tokenizer-agnostic and is the correct general form.
**Generalize:** when porting the prune lever to a non-Qwen family, force the
corpus/remap path; gate the first-N identity path behind a "frequency-ordered
vocab" capability flag (true for Qwen `gpt2`-style tokenizer, false otherwise).
Note `VOCAB_PRUNE_CORPUS` (the seed text, `:445`) is English-centric — fine for
the bench regime, but it is a calibration input, not an arch constant.

### 2.d Tokenizer family
All four local Qwen GGUFs and deepseek-v2-lite report `tokenizer.ggml.model =
gpt2` (BPE) (measured via `gguf` reader). Llama-2/3 GGUFs use `llama`/`spm` or
`gpt2`(llama-3); Gemma uses `spm`. `Tokenizer::from_gguf` is the existing
abstraction (`qwen_dense.rs:785–789` prefers a sidecar `tokenizer.json` then
falls back to GGUF). **No hardcode in the loader**, but the prune caveat (2.c)
and any special-token assumptions (`eos_id()`, chat template `im_start`/`im_end`
— Qwen-specific) live above the loader and must not be assumed by the generic
core. Chat templating is a *serve-layer* concern, out of scope for §9.1's
greedy-parity oracle (which uses raw token-in/token-out).

### 2.e Defaults that differ by family (already per-arch; fold into one table)
| Param | Qwen default | Llama default | Gemma2 | Source |
|---|---|---|---|---|
| `rope.freq_base` | 1_000_000 | 500_000 | 10_000 | qwen `:81`, llama `:111`, gemma `:107` |
| `rms_norm_eps` | 1e-6 | 1e-5 | 1e-6 | qwen `:82`, llama `:112`, gemma `:108` |
| `context_length` | 32768 | 8192 | 8192 | qwen `:83`, llama `:113`, gemma `:109` |
| `head_dim` | hidden/n_heads (derived) | `key_length` else derived | explicit `key_length` (256) | qwen `:51`, llama `:91`, gemma `:89` |
These are **fallbacks only** — when the GGUF carries the key, the file wins.
The fallbacks matter solely for GGUFs that omit the key, so keep them
per-family.

---

## 3. Local 2nd-family validation targets (what is actually on disk)

`find … -name '*.gguf'` (measured) — full inventory:

| File | Size | arch (measured) | dense? | Usable as §9.1 oracle? |
|---|---|---|---|---|
| `models/qwen2.5-3b-instruct-q4_k_m.gguf` | 1.8 GB | `qwen2` | yes | baseline (the reference) |
| `models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1.0 GB | `qwen2` | yes | **intra-family 2nd shape** (28L/1536/12h/2kv) |
| `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` | 469 MB | `qwen2` | yes | intra-family smoke (smallest) |
| `models/qwen2.5-7b-instruct-q4_k_m.gguf` | 4.4 GB | `qwen2` | yes | intra-family 2nd shape (28L/3584/28h/4kv/hd128) |
| `models/deepseek-v2-lite-q4.gguf` | 9.7 GB | `deepseek2` | **NO — MoE** (64 experts, 6 used; measured) | **out of §9.1.a scope; 9.7 GB blows 3 GB RSS** |

**Hard conclusion:** the §9.1-named cross-family oracle (Llama/Mistral) is
**not satisfiable with local weights.** The work that *can* be measured today
is intra-family dim-robustness on Qwen 0.5B/1.5B/7B. To satisfy §9.1 properly,
**acquire one small Llama-family Q4_K_M GGUF** (recommended: Llama-3.2-1B-
Instruct-Q4_K_M ≈0.8 GB, or Mistral-7B-Instruct-v0.3-Q4_K_M ≈4.4 GB — v0.3 to
dodge the SWA warning llama.rs:120 emits). Llama-3.2-1B is the cheapest:
smallest download, exercises the `key_length`-explicit head_dim path
(llama.rs:88–93), and its f16 reference for parity is trivially producible.

---

## 4. The parameterization diff plan (concrete)

### D-1 — unify the config reader (no behavior change; parity-provable today)

**New file** `crates/dismantle-core/src/model/arch_config.rs` (additive; does
not touch existing loaders until they opt in):

```rust
/// Family-independent dense-transformer dims, read from GGUF metadata
/// under a family prefix. Replaces the duplicated `*Config::from_gguf`.
pub struct ArchConfig {
    pub n_layers, hidden, n_heads, n_kv_heads, head_dim, intermediate,
        vocab_size, max_seq_len: usize,
    pub rope_theta, rms_norm_eps: f32,
}
pub struct ArchDefaults { pub rope_theta: f32, pub rms_norm_eps: f32, pub ctx: usize }

impl ArchConfig {
    pub fn from_gguf(prefix: &str, g: &GgufFile, d: ArchDefaults) -> Result<Self> {
        let u = |k: &str| g.metadata.get(&format!("{prefix}.{k}")).and_then(|v| v.as_u32());
        let f = |k: &str| g.metadata.get(&format!("{prefix}.{k}")).and_then(|v| v.as_f32());
        let n_layers = u("block_count").ok_or_else(|| Error::Model(format!("missing {prefix}.block_count")))? as usize;
        let hidden   = u("embedding_length").ok_or(...)? as usize;
        let n_heads  = u("attention.head_count").ok_or(...)? as usize;
        let n_kv_heads = u("attention.head_count_kv").unwrap_or(n_heads as u32) as usize;
        let head_dim = u("attention.key_length").map(|v| v as usize).unwrap_or(hidden / n_heads);
        let intermediate = u("feed_forward_length").ok_or(...)? as usize;
        let vocab_size = u("vocab_size").map(|v| v as usize)
            .unwrap_or_else(|| /* max-dim of token_embd.weight, the existing fallback */);
        Ok(Self { n_layers, hidden, n_heads, n_kv_heads, head_dim, intermediate, vocab_size,
                  rope_theta: f("rope.freq_base").unwrap_or(d.rope_theta),
                  rms_norm_eps: f("attention.layer_norm_rms_epsilon").unwrap_or(d.rms_norm_eps),
                  max_seq_len: u("context_length").unwrap_or(d.ctx as u32) as usize })
    }
}
pub fn arch_meta_prefix(arch: &str) -> &'static str { /* qwen2.5→"qwen2", llama3.2→"llama", … — mirror profile::arch_family */ }
```

Then each existing `*Config::from_gguf` becomes a thin wrapper that calls
`ArchConfig::from_gguf(prefix, g, family_defaults)` and copies the eight shared
fields, keeping ONLY its family-extra fields (gemma: `head_dim_scale`,
`attn/final_logit_softcap`, `query_pre_attn_scalar`; llama: `rope_scaling`,
`arch`). **`head_dim` fallback is now uniform** — note this changes Qwen's
`head_dim` derivation only if a Qwen GGUF ever ships `attention.key_length`
(none of the four local ones do; Qwen value equals hidden/n_heads anyway:
2048/16 = 128, matching the explicit-key result, so **bit-identical**).

**Risk:** none to runtime — the *values produced are identical* for all five
local GGUFs (verify by asserting the new reader reproduces the old structs).
This is a pure de-duplication.

### D-2 — port the fast decode path to a second family (the breadth win)

Introduce a small **capability descriptor** (not new dims — new *structural
switches*) consumed by the TCB forward path so it stops assuming Qwen shape:

```rust
pub struct ArchSpec {
    pub norm: NormPlacement,          // PreOnly (qwen/llama) | Sandwich (gemma2)
    pub activation: GatedAct,         // SiLU (qwen/llama) | GeluTanh (gemma2)
    pub qkv_bias: bool,               // true qwen, false llama/gemma
    pub rope: RopeVariant,            // Plain{theta} | Llama3Ntk(Llama3RopeScaling)
    pub embed_scale_sqrt_hidden: bool,// gemma2 only
    pub attn_logit_softcap: f32,      // 0 = off (qwen/llama); gemma2 sets it
    pub final_logit_softcap: f32,     // 0 = off
    pub freq_ordered_vocab: bool,     // true qwen gpt2 → first-N prune legal; else force corpus-remap
}
```

The TCB pipeline currently inlines Qwen's choices. The diff is to branch the
**five** structural points on `ArchSpec`:
1. **Norm site** — add the two extra sandwich-norm dispatches when
   `norm == Sandwich` (mirror gemma2.rs:366,383); skip when `PreOnly`.
2. **Activation kernel** — select `silu_mul` vs `gelu_mul` (both already exist;
   gemma2.rs:378 uses `gelu_mul`).
3. **Bias adds** — already guarded by `is_empty()`; just ensure non-Qwen
   families load `_opt` biases (they will be empty → free).
4. **RoPE** — route through `rope_inplace_scaled` + optional `Llama3RopeScaling`
   (llama.rs already has this) instead of plain `rope_inplace`.
5. **Logit softcap + embed scale** — apply `logit_softcap_inplace` /
   √hidden scale when the spec sets them (gemma2 path already has the kernels).

**Tensor-layout caveat (phi3 is harder than llama/gemma):** the ArchSpec above
covers *compute* switches but assumes the GGUF stores **separate**
`attn_q/k/v.weight` and `ffn_gate/up.weight` (true for qwen/llama/gemma). **phi3
stores FUSED projections** — one `attn_qkv.weight` row-split into Q/K/V and one
`ffn_up` split into gate(first half)/up(second half) (phi3.rs:6,130–136,
213–226) — plus per-dimension long/short-context RoPE factors (phi3.rs:150). So
phi3 needs an extra `proj_layout: { Separate | FusedQkv + FusedGateUp }` switch
in `ArchSpec` and a row-offset `TensorRef` split at load (phi3.rs already
implements the split — promote it, don't re-derive). **Do phi3 LAST**; llama
(only RoPE differs) then gemma2 (norm+act+softcap) are the cleaner first ports.

**Where it plugs in:** rather than four copies of the TCB pipeline, the cleanest
shape is a single generic `DenseDecodeCore` parameterized by `ArchConfig +
ArchSpec` that owns `forward_token_greedy_tcb` / `forward_tokens_batch_tcb` /
the predec + pinned-buffer machinery, with the four `Engine` impls becoming
thin owners of a `DenseDecodeCore`. **This is a large refactor of a 5378-line
file and is the bulk of D-2's effort** — scope it as its own attended haul, not
a drive-by. The intermediate, lower-risk option is to *promote llama.rs's path
to the TCB pipeline first* (llama is the closest sibling — PreOnly norm, SiLU,
no bias, plain/NTK RoPE, no softcap → only RoPE differs from Qwen), prove the
cross-family parity gate on Llama-3.2-1B, **then** generalize gemma/phi.

**Levers that attach unchanged (per §9.1 oracle clause c):** prefix-cache
(`ram_prefix_cache`) and draft-tuning are KV/logits-level and **family-agnostic
by construction** — a matched token prefix's KV is a pure function of
model+tokenizer+tokens regardless of arch. They require no per-family work; the
gate is simply that they still attach to `DenseDecodeCore`. eagle5 is the
exception: its trained head is Qwen-shape-specific (capture layer, in_proj dims)
and does **not** transfer — leave it Qwen-gated.

---

## 5. Minimal 2nd-family bring-up checklist

Run in order; each line is a gate. **`[BLOCKED]` = needs a non-Qwen GGUF.**

1. **`[do-today]`** D-1 land: new `ArchConfig::from_gguf` + `arch_meta_prefix`;
   wrap all four `*Config::from_gguf`. Assert produced dims byte-match the old
   structs for all 5 local GGUFs (CPU-only, milliseconds; no Metal).
2. **`[do-today]`** Intra-family dim smoke: `cargo test --workspace --lib`
   green, then a 3-token greedy generate on Qwen **1.5B** and **7B** through the
   *existing* qwen TCB path — proves the parameterized config drives a different
   shape with no hardcoded-dim panic. (This is the cheap pre-flight; it is NOT
   the §9.1 cross-family oracle.) GPU run — schedule on a GPU-allowed session.
3. **`[BLOCKED: acquire GGUF]`** Download Llama-3.2-1B-Instruct-Q4_K_M (≈0.8 GB)
   to `models/`. Confirm `load_engine` dispatches it to `LlamaDense` and it
   loads with **no hardcoded-dim failure** (§9.1 oracle clause a).
4. **`[BLOCKED]`** D-2 step-1: promote `LlamaDense` to the TCB pipeline (RoPE is
   the only structural delta vs Qwen). Parity-gate the new TCB Llama path vs the
   existing CPU-hybrid Llama path *on the same Llama GGUF* — internal self-parity
   first (atol=1e-3 fp16, per CLAUDE.md verification rule).
5. **`[BLOCKED]`** §9.1 oracle clause b: greedy parity of dismantle-Llama vs
   **llama.cpp** on the same prompt/seed (first-3-token-ID match minimum, per the
   token-output gate; extend to 16/32-tok for confidence).
6. **`[BLOCKED]`** §9.1 oracle clause c: confirm `ram_prefix_cache` and
   draft-tuning attach to the Llama core unchanged (a 2nd same-prefix request
   skips re-prefill; bit-identical KV reuse).
7. **`[regression]`** Re-run the Qwen parity gate (below) — generalization MUST
   NOT move a single Qwen logit.
8. **`[expand]`** Repeat 3–7 for gemma2 (sandwich+GeGLU+softcap) then phi3
   (FUSED qkv/gate-up + per-dim RoPE — do it LAST, §4 caveat), parity-gating
   each before adding to the supported list.

---

## 6. The parity gate proving generalization did not regress Qwen

This is the non-negotiable regression wall for D-1 **and** D-2. Two layers:

**Layer A — config bit-identity (D-1, CPU-only, do first).**
The new `ArchConfig::from_gguf("qwen2", g, qwen_defaults)` must reproduce the
exact `QwenConfig` the old code produced, for the 3B/1.5B/0.5B/7B GGUFs. A
unit test comparing all 10 fields field-by-field. **Expected: identical**
(verified by inspection — same keys, same fallbacks; the only theoretical
divergence is `head_dim` if a Qwen GGUF carried `attention.key_length`, which
none do, and where the value would equal hidden/n_heads anyway).

**Layer B — logit / token bit-identity (D-2, the real gate).**
Use the project's existing locked baselines so this is a *regression* test, not
a new oracle:
- **Kernel parity:** `tests/correctness/phase1_kernel_parity.rs` and the locked
  `tests/golden/_phase1_kernel_baseline.hashes` — the refactored TCB path must
  reproduce the Qwen Q4_K/rmsnorm/rope kernel outputs at **atol=1e-3 fp16**
  (CLAUDE.md verification rule).
- **Token parity:** `tests/golden/_phase1_token_baseline.hashes` — first **3**
  token IDs on the locked Qwen-3B prompt must match exactly (CLAUDE.md token
  gate; mismatch = the refactor changed Qwen behavior = halt).
- **Paired decode parity:** the existing `tools/bench/paired_lever.sh` Qwen-3B
  run must stay **bit-identical** (the predec/pinned path is bit-identical by
  design; the refactor must preserve that). Verify per the worktree-parity
  memory note: re-run parity *yourself* on the baseline before trusting any
  "parity passed" claim.

**Pass condition:** Layer A field-identical AND Layer B kernel-atol-1e-3 +
token-ID-exact on Qwen-3B, with the second family (Llama-1B) independently
greedy-matching llama.cpp. If Qwen tokens move at all, the generalization
leaked a Qwen assumption into a shared path — halt and localize.

**Why this gate is sufficient:** §9.1's three oracle clauses (loads / parity /
levers-attach) on the second family prove *forward* generalization; the locked
Qwen baselines prove *no backward regression*. The decode-kernel optimum is
already established (~29–31 tps clean, bible §3 Corr.2) so D-2 success is "second
family runs through the *same* fast path at parity," not a new tps number — the
breadth win is coverage, not throughput (consistent with §9 framing that breadth
≠ tps).

---

## 7. Scope discipline (§9.1.a — what this plan deliberately excludes)

- **deepseek-v2-lite (MoE) is NOT a target.** MoE routing is explicitly out of
  §9.1.a; the file is also 9.7 GB (>3 GB RSS). It already has its own loader.
- **No new format, no QTIP, no layout repack** — those are §9.2/§9.3, gated
  separately and later.
- **No chat-template / serve-layer work** — §9.1's oracle is raw greedy
  token-parity; chat templating (Qwen `im_start`/`im_end`) is a serve concern.
- **eagle5 stays Qwen-gated** — its trained head is shape-specific; porting it
  is a separate trained-artifact task, not loader parameterization.
- **The rule (bible §9.1.a):** widen to the dense Llama-family shape dismantle
  is already good at; do not widen toward "a worse llama.cpp."

---

## 8. Numbers in this doc, tagged

- "clean dec_tps ~29–31" — **(measured)**, bible §3 Corr.2 / clean-room
  2026-05-31; carried as the *coverage-not-throughput* anchor (D-2 success is
  parity at this number, not beating it).
- Local GGUF inventory + per-model dims + `general.architecture` /
  `tokenizer.ggml.model` / deepseek expert counts — **(measured)** this session
  via the `gguf` Python reader (header KV only; no weight load; peak RSS well
  under 3 GB).
- "~80% generalized at correctness / ~0% at fast-decode" — **(estimate)**, a
  qualitative read of the four-loader source surface, not a metric.
- Llama-3.2-1B ≈0.8 GB, Mistral-7B-v0.3 ≈4.4 GB download sizes — **(estimate)**
  from typical Q4_K_M sizing; confirm at download time.
- `head_dim` bit-identity for Qwen under the unified reader — **(proxy)**:
  reasoned from 2048/16 = 128 and the absence of `attention.key_length` in the
  four local Qwen GGUFs; the *decisive* confirmation is the Layer-A field-by-
  field unit test in §6.

**The single decisive gate:** a Llama-family Q4_K_M GGUF on disk, run through
the parameterized TCB path, greedy-matching llama.cpp on a fixed prompt/seed
(§5 step 5) **while** the locked Qwen token baseline stays bit-identical (§6
Layer B). Everything upstream of that is preparation; that one paired result is
what settles "the family generalization is sound."
