# aud07 — Apple GPU / prefill / context

Audit of git `04193ccbc` (2026-09-02). Freeze date 2026-08-27 from the campaign's `H-ROADMAP.md` copy (that file is **not** in git HEAD). This is source archaeology. No GPU benchmark was run. Nothing here is `PHYSICALLY_MEASURED` by this lane.

## The claim

> Native prefill was previously using the SINGLE-TOKEN DECODE path, and a first batched GEMM path now exists.

**Verdict: SPLIT.**

| Clause | On HEAD (`04193ccbc`) | Elsewhere |
|---|---|---|
| Native prefill is `session.step` per prompt token | **CONFIRMED** — `generate_greedy` at `qwen38_hybrid_decode.rs:7454` | same |
| A batched GEMM path is wired into the Qwen3.8 resident | **REFUTED** | **CONFIRMED** on unmerged `8391a0ef8` (`grok/prefill-gemm-20260901-232724`) |
| Some batched GEMM prefill exists in this repo | **CONFIRMED**, but it is QwenDense + opt-in env, not the native resident | — |

HEAD still does this:

```rust
for (i, &token) in prompt.iter().enumerate() {
    let (sampled, timing) = session.step(token)?;
}
```

`step` builds `encode_full_token` (embed + 64 layers + lm_head) and `TokenCommandBuffer::new` every token. That is the decode graph. G117/G118/G119 and `PREFILL_KV.json` (2026-08-24) measured exactly this topology. The producer has not changed.

A first Qwen3.8 GEMM prefill **was written** after the freeze (`8391a0ef8`, 2026-09-02 00:10). It adds `qwen38_hybrid_prefill.rs`, `shaders/qwen38_prefill.metal`, and rewires `generate_greedy` to `prefill_prompt` when `HAWKING_QWEN38_BATCH_PREFILL` is not `0`. **`git merge-base --is-ancestor 8391a0ef8 HEAD` is false.** Those files are not at HEAD.

`docs/PREFILL.md` *is* at HEAD (landed in an HCLI commit, `b4a3857d4`, that does not touch `hawking-core`). It describes the unmerged files and a 1.85× (78.3 vs 42.7 tok/s) as if they were in this tree. The kernel commit itself says it was unmeasured in the sandbox. There is no `receipts/` artifact for 78.3. Treat the doc as `STALE_ROADMAP_TEXT`.

Separately, HEAD's **generic** QwenDense engine already has `HAWKING_QWEN_BATCH_PREFILL=1` → `forward_tokens_batch_tcb` → `gemm_q4_k_m_batched_v3w_mma_n32_pinned_tcb`. Default is off. `prefill_slot` (what `hawking-serve` actually calls) **refuses** that path because chunked attention is not prefix-causal and produced a prompt-independent first token. So: a GEMM primitive exists; a correct native-resident prefill backend does not.

## Should PREFILL be its own gene?

**Yes — as an I-D gene, not as a civilization, and not stuffed under II-D.**

II-D "State / Tokenizer / Decoding" is the wrong home. That civilization is about *non-weight* organs (KV, tokenizer, sampling, LM head, long-context policy). Prefill-as-GEMM is a *weight-execution phase*: different shaders, different workspace, different encode functions, different tests, different scaling (compute-bound GEMM + quadratic GQA vs bandwidth-bound GEMV). The unmerged module is already a separate file included into the session. I-D already lists **phase-aware backend selection**; that gene is unexpressed as a shipped backend. Name PREFILL there (or rename that gene so PREFILL is the first expressed phase). Do not add Era VI. Do not touch 0.7%.

The school item in H-ROADMAP §11.3 ("Phase-aware prefill vs decode backends") is not a gene. Leaving the work only as school prose is how `docs/PREFILL.md` outran the tree.

## Other gene questions

**Context Computer / Context Gravity — do not mint a civilization. Split the name.**

Two codebases share the word:

1. `crates/hawking-context::ContextCompiler` is a HIDE prompt packer. `hide-backend` and `hide-kernel` call `.compile(CompileInput{...})`. HCLI `context_budget.resolve` is called from `backends.py` / `engine.py`. Acceptance receipts `HCLI_CONTEXT_*` are ACCEPTED. That is product, already wired.
2. GPU *state gravity* (GQA KV grows with seq, DeltaNet rec+conv does not, state overtakes weights at long context) is a measured school in `PREFILL_KV.json` / `STATE_GRAVITY.json` and live Metal buffers on the session. It is not a compiler.

A "Context Computer" civilization would duplicate HIDE and break the five-era rule. Express the existing II-D genes (KV/state compression, persistent recurrent state, long-context policy). Do not add a fourth name for the same organs.

**DeltaNet-state checkpoint+reuse — do not add a gene. The gene already exists and has no caller.**

II-D already lists `persistent recurrent state`. `sstate_disk.rs` defines `SstateDiskCache` and nothing calls it. `generate_greedy` always `session.reset()`. G007 (`hcli/test_deltanet_state_checkpoint.py`) is a red gate with no `receipts/sovereign/G007_deltanet_state.json`. Adding a gene with the same meaning is nomenclature. Wire a caller, then discharge G007.

**KV/state tiering — yes, explicit II-D gene.**

Compression and *where bytes live* are different. QwenDense `generate()` already has RAM (`InMemoryPrefixCache`) then disk (`PrefillDiskCache::lookup_longest_prefix`). `hawking-serve` has `SystemPromptKvBank`. The native Qwen38 resident has one GPU workspace and resets it every HCLI request. That split is load-bearing. A gene named only "KV/state compression" will keep scheduling quant recipes (STATE_GRAVITY's H2O/KIVI/rank-32, capability cost ABSENT) and miss tiers the generic engine already runs.

**Long-context physical accounting — do not add a duplicate. Express `long-context policy`.**

II-D already has it. G006 is the verifier (red; no receipt). G118 fitted quadratic cost; 32k was predicted, not walked. The sealed profile is `max_seq_len: 8192`. The resident does not return a live `(prefill_s, kv_bytes, rec_state_bytes, admitted_limit)` object a gate can hash. That is an unexpressed gene, not a missing heading.

## Surface classifications (HEAD)

Strongest state the *source* supports. Tests named below were **not** executed by this lane.

| Surface | Class | Why |
|---|---|---|
| Native resident | INTEGRATED | HCLI `complete_payload` → `ascension_qwen38_resident` → `generate_greedy` |
| Decode kernels | INTEGRATED | `step` → `encode_full_token`; 964-dispatch ledger; fusions subtract for real |
| Prefill (native) | INTEGRATED as decode | prompt loop is `step`; not a GEMM backend |
| Qwen38 GEMM prefill | ABSENT | files/branch not on HEAD |
| QwenDense batched GEMM | CALLABLE | env-gated; serve `prefill_slot` avoids it |
| f32 prefill gap | CONCEPT_ONLY | 402 Q4 + 353 f32 is a catalog constant; no GEMM coverage to gap on HEAD |
| Long context | CALLABLE | clamped to 8192; 131k/262k G006 red |
| DeltaNet decode | INTEGRATED | `encode_layers` → `encode_deltanet` (48 layers) |
| DeltaNet checkpoint | SCAFFOLDED | `sstate_disk` has zero callers |
| GQA | INTEGRATED | `encode_gqa`; f32 KV |
| MLP | INTEGRATED | `encode_dense_mlp`; `HAWKING_QWEN38_FUSE_MLP` is read |
| LM head | INTEGRATED | `encode_terminal` → matvec + `sample_argmax_f32_tcb` |
| Persistent workspace/KV | INTEGRATED | allocated at attach, reused across tokens |
| Persistent command buffers | ABSENT | `TokenCommandBuffer::new` every `step` |
| Dispatch reduction | INTEGRATED (decode) | live fusion flags; prefill collapse ABSENT |
| Hybrid KV/state | INTEGRATED in-session | reset every request; no prefix share |
| QwenDense KV tiers | CALLABLE | RAM then disk; not on hybrid |
| HIDE ContextCompiler | INTEGRATED | `.compile` called from hide-backend/kernel |
| HCLI context budget | INTEGRATED | `resolve_context_budget` callers; acceptance ACCEPTED |
| No-copy (Qwen38 hybrid) | ABSENT | `new_buffer_with_bytes_checked` copies |
| No-copy (GGUF engines) | CALLABLE | `new_buffer_no_copy` on llama/qwen_dense/mixtral/DSV2 |
| HCLI overhead | CONCEPT_ONLY | 2× claim has no receipt; not remeasured |
| Protected bench machinery | CALLABLE | runner exists; `QWEN27_PROTECTED_BASELINE` verdict BLOCKED (not quiet) |
| Qwen80 mixed prefill | INTEGRATED | `forward_token` per prompt id |
| Llama packed prefill | SCAFFOLDED | two unsafe env flags; comment says slower than serial |
| `docs/PREFILL.md` | STALE_ROADMAP_TEXT | describes sources git cannot resolve at HEAD |

Sovereign red gates with **no** receipt: G005 (f32 GEMM coverage), G006 (131k/262k), G007 (DeltaNet checkpoint). The tests are load-bearing verifiers, not passing measurements.

## What moved since freeze

After 2026-08-27, the only Qwen3.8 prefill *backend* change found is `8391a0ef8` — **off HEAD**. Decode-side work on HEAD since freeze (MLP bitcast/unpack, DeltaNet widen_f4, fusion promotion G126, etc.) does not change the prefill topology: still one `step` per prompt token, still 964-class GEMV.

Qwen80 mixed greedy is the same shape (`forward_token` per prompt token).

## Surprises

1. **GEMM prefill is not on HEAD.** Settled by merge ancestry, not by more prose.
2. **`docs/PREFILL.md` shipped without the producer.** Settled by a G005-shaped receipt from a named Metal command.
3. **QwenDense already had batched GEMM**, and serve refuses it for a documented correctness bug. "First GEMM path" must name the *caller*.
4. **Unmerged `encode_prefill_gemm` errors** rather than falling back to matvec. The "353 f32 tensors still go through matvec" sentence is coarser than that function.
5. **`H-ROADMAP.md` is not in git.** Gene arguments here used the Downloads copy the campaign named.
6. **Native weights are copies.** No-copy is a GGUF-engine fact.
7. **`generate_greedy` always `reset()`s.** Prefix reuse and DeltaNet checkpoint cannot matter until a production caller stops doing that.

Machine-readable companion: `receipts/audit/aud07-accelerator-prefill-context.json`. Every capability entry has a call site of the symbol itself, or an explicit blocker.
