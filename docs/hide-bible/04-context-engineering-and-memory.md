# 04 · Context Engineering & Memory

> **Purpose (one line).** Because Hawking owns the runtime, context is not a fixed paid window we rent from a cloud — it is an *elastic, compressed, persistent, on-disk, model-cooperative* substrate we engineer end to end: a per-turn **Context Compiler** that budgets and packs the model window from many ranked sources, sitting on top of a **tiered KV store** (GPU→RAM→disk), an **extended-context** position/attention layer, and a **forever, user-editable project memory** — designed so the model effectively never forgets and never runs out of room.

---

## Table of contents

1. [Purpose & scope](#1-purpose--scope)
2. [Tenets](#2-tenets)
3. [State of the art + limits (cited)](#3-state-of-the-art--limits-cited)
4. [The Hawking design (concrete)](#4-the-hawking-design-concrete)
   - 4.1 [System architecture & module layout](#41-system-architecture--module-layout)
   - 4.2 [The Context Compiler](#42-the-context-compiler)
   - 4.3 [The context manifest (UI + replay contract)](#43-the-context-manifest-ui--replay-contract)
   - 4.4 [Elastic / extended context length](#44-elastic--extended-context-length)
   - 4.5 [KV cache as a first-class managed store](#45-kv-cache-as-a-first-class-managed-store)
   - 4.6 [Hierarchical memory (working / episodic / semantic / procedural)](#46-hierarchical-memory-working--episodic--semantic--procedural)
   - 4.7 [Forever project memory (CLAUDE.md on steroids)](#47-forever-project-memory-claudemd-on-steroids)
   - 4.8 [Context compression & distillation](#48-context-compression--distillation)
   - 4.9 [Per-task context profiles](#49-per-task-context-profiles)
5. [How we EXCEED — local superpowers & "cloud literally cannot do this"](#5-how-we-exceed)
6. [Failure modes & mitigations (overflow, rot, stale memory)](#6-failure-modes--mitigations)
7. [Extensibility / plugin points](#7-extensibility--plugin-points)
8. [Bleeding-edge / moonshots (ranked)](#8-bleeding-edge--moonshots-ranked)
9. [Open questions & dials](#9-open-questions--dials)
10. [Cross-references](#10-cross-references)
- [Appendix A — Binding contracts (schemas other chapters import)](#appendix-a--binding-contracts)
- [Appendix B — Source register](#appendix-b--source-register)

---

## 1. Purpose & scope

This chapter specifies the **context and memory substrate** of HIDE — the layer that decides, every single turn, *exactly which tokens the model sees*, and the layer that remembers everything across turns, sessions, and weeks. It is the chapter where the user's explicit example lives:

> *"I said something about the context length — these kinds of things we can adjust, we can work on, we can offer people better opportunities."*

A cloud coding agent rents a **fixed, metered window**. Every token in that window costs money and the window is a hard wall. Their context strategy is therefore *defensive*: spend as few tokens as possible, evict aggressively, summarize to survive. HIDE is the opposite. We own `hawking-serve` and `hawking-core`; we control attention, the KV cache, and the RoPE positions directly; **there is no per-token bill**, so we can keep enormous context resident, recompute and re-compress freely, and back the whole thing with local disk and RAM the model treats as effectively unbounded. Context is a *lever we engineer*, not a budget we ration.

### In scope

- **The Context Compiler** — the per-turn subsystem that budgets the window and packs it from ranked sources (system, plan, retrieved code, symbols, tool outputs, memory tiers, scratchpad), emitting a **context manifest** the UI renders as a live "context stack."
- **Elastic / extended context length** — exploiting runtime ownership: RoPE/YaRN scaling, attention sinks / StreamingLLM, sliding-window + sink, routing unbounded-history tasks to the SSM path (RWKV-7 / Mamba-2), and per-task context-length profiles the user can dial.
- **KV cache as a managed store** — paging, tiered residency (GPU/RAM/disk), eviction & compression (quantized KV, heavy-hitter), prefix/KV reuse banking, and KV checkpoint/restore for resume.
- **Hierarchical memory** — the working/episodic/semantic/procedural taxonomy, the schemas, stores, promotion/decay policies, and retrieval.
- **Forever project memory** — persistent, versioned, user-editable, auto-maintained, provenance-tagged memory; how it is injected and how it is prevented from rotting.
- **Context compression/distillation** — recursive summarization, summary/beacon tokens, draft-model-powered cheap compaction, learned eviction.
- **Anti-context-rot measures.**

### Out of scope / deferred (with explicit gates)

| Item | Where it lives | Status |
|---|---|---|
| The HF / remote-model surface | ch.06 | **Deferred** — this chapter designs against the **local HTTP surface first**; remote is a strict superset and binds the same manifest. |
| `.tq` sub-4-bit serving, 32B residency | *Hawking Condense* + ch.06 | **Runtime testing, not shell-gating.** Context profiles reference model footprint but never block on it. |
| GPU-resident KV surgery kernels (int4-KV append, paged-attention block table on Metal) | `hawking-core` model/arena | **Later / not shell-gating.** Designed here, marked at every hook; the shell works against today's HTTP surface without them. |

> **Scoping rule, restated as a hard invariant.** Everything in §4 that touches *inside-the-runtime* KV memory carries a **[RUNTIME-SIDE — LATER]** tag and a **[SHELL-TODAY]** fallback that works against the existing `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/hawking/generate`, `/v1/hawking/tokens`, and `/metrics` endpoints. The shell is never blocked on a kernel landing.

### Ground truth this chapter binds to (verified in-tree)

The runtime is further along than a greenfield design would assume. This chapter binds to **real types**, not aspirations:

- **`hawking-serve`** (`crates/hawking-serve/src/`): continuous batching (`batch/`), per-slot SSE, OpenAI-compatible routes (`http.rs`), a **`SystemPromptKvBank`** (`system_kv_bank.rs`) that maps `hash(system-prefix) → source_slot` for cross-request KV reuse (re-verified, **greedy-lossless**), and spec governance (`spec_gov.rs`).
- **`hawking-core`** (`crates/hawking-core/src/`):
  - **Engine trait** (`engine.rs`) exposing the seams this chapter drives: `prefill_slot`, `prefill_slot_from_pos(slot, ids, start_pos)`, `copy_kv_prefix_to_slot(src, dst, prefix_len)`, `kv_fingerprint_at_pos`, `forward_multiseq_*`, `embed` (powers `/v1/embeddings`), and `reset_kv_for_test`.
  - **A working-set eviction framework already stubbed** (`stateful/working_set.rs`): a `KvEvictionPolicy` trait with `StreamingLlmPolicy`, `H2OPolicy`, `SnapKvPolicy`, and `LosslessPolicy` (the greedy-lossless escape hatch), plus `EvictionPlan { Drop | Compress }`, `WorkingSetBudget`, `WorkingSetMode { Bounded | Lossless }`. **The taxonomy of §4.5 is already the code's taxonomy.**
  - **A tiered prefix-KV cache** (`stateful/prefix_cache.rs` RAM tier + `cache/prefill_disk.rs` on-disk `DSPRFKV2` tier) keyed by a rolling prefix hash bound to `(model_id, tokenizer_signature, prompt_ids)`, with `HAWKING_PREFIX_CACHE_DIR` / `HAWKING_PREFIX_CACHE_BUDGET_MB` levers.
  - **SSM path**: `model/rwkv7.rs` (O(1) recurrent state, ~6 MiB at any depth) and `model/mamba2.rs` (no KV cache; per-layer SSM state). This is the structural long-context win.
  - **KV precision levers**: `HAWKING_QWEN_F16_KV`, `HAWKING_QWEN_INT4_KV` (staged), plus ~284 `HAWKING_*` levers including vocab pruning.
  - **Constrained decode** (`json_constrain.rs`) — used by the manifest's structured-output path.
  - **Attention capture** (`stateful/attn_capture.rs`) — the hook that feeds real attention scores to H2O/SnapKV.

Where this chapter says "we already have X," it has been checked against these files. Where it says "[RUNTIME-SIDE — LATER]," the seam exists but the body is `todo!()` or env-gated-off.

---

## 2. Tenets

1. **Context is engineered, not rented.** Every turn, a deterministic compiler chooses the highest-value token set under a budget. We never "just stuff the window" and we never let the model see junk because junk was free to keep.
2. **The smallest high-signal set wins.** Per Anthropic's context-engineering guidance and the OP-RAG inverted-U, *more context past a point is actively harmful*. The compiler's objective is **signal density**, not fill ratio. ([Anthropic 2025], [Jin et al. 2025])
3. **Nothing is ever truly lost — it is tiered.** A token leaving the model window is demoted (KV→RAM→disk, or raw→summary→memory), never deleted. Retrieval can always pull it back. This is the local superpower: disk is cheap and ours.
4. **Lossless by default, lossy by opt-in and gated.** The default working-set mode is `Lossless` (`is_lossless() == true`); eviction/compression is explicit, oracle-gated, and per-task. The greedy-exact path is always available for correctness-critical work. (Mirrors the existing `WorkingSetMode::Lossless` default.)
5. **Reuse before recompute.** A prefix we have already processed is never re-prefilled. KV banking, prefix cache, and copy-KV are first-class, not optimizations. (We already ship the `SystemPromptKvBank` + disk prefix cache.)
6. **The model is a cooperating party.** We can ask the model to attend differently (sinks), position differently (RoPE scaling), summarize itself (draft-model compaction), and emit memory edits as tool calls. Cloud treats the model as a sealed box; we co-design with it.
7. **Determinism & replay.** The exact context that produced any turn is reconstructable byte-for-byte from the manifest + content-addressed stores. A session is a *replayable state machine*, which makes "why did the agent do that" answerable and makes debugging the agent loop (ch.02) tractable.
8. **Provenance on every token.** Each span in the window carries its source, age, and trust. The UI shows it; the compiler scores by it; rot mitigations key on it.
9. **Memory is the user's, forever, and editable.** Project memory is a structured, versioned, human-readable store the user can open, grep, correct, and pin. It never silently resets. (CLAUDE.md is the seed; we make it queryable and auto-maintained.)
10. **Honor the architecture's grain.** Transformers grow KV `O(n)`; SSMs carry `O(1)` state but compress history lossily. We *route by task*: exact-recall work to attention, unbounded-history/flat-throughput work to RWKV-7/Mamba-2. We do not pretend one architecture is free.

---

## 3. State of the art + limits (cited)

Tagged **[PROVEN-IN-PROD]** / **[RESEARCH-PROVEN]** / **[SPECULATIVE]**, with *difficulty* (build cost for us) and *impact*. Full source register in [Appendix B](#appendix-b--source-register).

### 3.1 Context-window extension (position & attention)

| Technique | Mechanism (compressed) | Headline | Maturity | Diff / Impact |
|---|---|---|---|---|
| **Position Interpolation (PI)** | Linearly down-scale position indices into the trained range; short finetune. | LLaMA 7–65B → 32k in <1k steps. | [PROVEN-IN-PROD] | low / med — the basis of everything below. ([Chen et al. 2023]) |
| **NTK-aware / Dynamic NTK** | Scale the RoPE base θ non-uniformly across dims; make the factor a function of current length. Often training-free. | 8k+ with minimal ppl loss, no finetune; graceful length generalization. | [PROVEN-IN-PROD] (HF `rope_scaling: dynamic`) | low / med. ([YaRN paper 2024]) |
| **YaRN** | "NTK-by-parts" (3 frequency bands) + attention-temperature `√(1/t)=0.1·ln(s)+1`. Finetune. | LLaMA-2 4k→128k (**32×**), ppl@128k=2.37, <0.1% pretrain data. **The workhorse** (Qwen, DeepSeek, Mistral). | [PROVEN-IN-PROD] | low–med / **high**. ([Peng et al. 2024]) |
| **LongRoPE / LongRoPE2** | Evolutionary search for per-dim rescale + progressive schedule; LongRoPE2 uses needle-driven ppl to fix short-ctx regression. | **2M tokens** (LongRoPE); 128k at ~97–98% short-ctx retention (LongRoPE2, 2025). | [RESEARCH-PROVEN] | high / high. ([Ding et al. 2024], [LongRoPE2 2025]) |
| **SelfExtend** | Training-free bi-level attention: grouped (coarse, long-range) + neighbor (precise, local). Inference-time patch. | Llama-2 → 16k+ matching finetuned methods, **no training**. | [RESEARCH-PROVEN] | low / high (no weights touched). ([Jin et al. 2024]) |
| **StreamingLLM / attention sinks** | Keep first **4** tokens (sinks) + rolling recent window; evict middle. Fixed cache, infinite stream. | Stable over **4M+** tokens; up to **22.2×** vs sliding-window-recompute. Does *not* extend effective recall — keeps the model coherent & fast. | [PROVEN-IN-PROD] | low / high. ([Xiao et al. 2023]) |
| **Sliding-window + global (Mistral SWA / Longformer)** | Each layer attends a window W; receptive field stacks to ~W·layers; few global tokens see all. | Linear attention cost; constant rotating-buffer KV. Recall beyond multi-hop window is lossy. | [PROVEN-IN-PROD] | med / med. ([Jiang et al. 2023], [Beltagy et al. 2020]) |
| **Activation Beacon** | Learned "beacon" tokens distill a context chunk into their own KV; raw KV dropped. | 4k → **400k** (~100× of distilled span), small quality cost. | [RESEARCH-PROVEN] | high (trains a module) / high. ([Zhang et al. 2024]) |

### 3.2 KV-cache compression, eviction & management

| Technique | Mechanism | Headline | Maturity | Diff / Impact |
|---|---|---|---|---|
| **H2O (Heavy-Hitter Oracle)** | Keep recent window + running-top tokens by cumulative attention mass. | ~20% budget ≈ lossless; up to **29×** throughput vs HF-Accelerate. | [RESEARCH-PROVEN] | low–med (needs attn scores; fights FlashAttention) / high. ([Zhang et al. 2023]) |
| **Scissorhands** | "Persistence of importance" — historically high-attention tokens stay important. | **5×** KV cut lossless; **20×** with 4-bit. | [RESEARCH-PROVEN] | low–med / high. ([Liu et al. 2023b]) |
| **SnapKV** | Recent "observation window" votes (via attention) on which *prompt* positions to keep; pooled across heads, applied at prefill-end. | Near-lossless on LongBench/NIAH at large prompt-KV cuts. **De-facto eviction baseline.** | [RESEARCH-PROVEN] | low–med / high. ([Li et al. 2024]) |
| **FastGen** | Per-head policy (local / special-token / punctuation / full). Adaptive, training-free. | ~40% KV cut (65B) keeping >95% of attn map. | [RESEARCH-PROVEN] | med / med. ([Ge et al. 2024]) |
| **PyramidKV** | Layer-varying budget (large low layers → small high layers). | Full quality at **~12%** of KV. | [RESEARCH-PROVEN] | low (atop SnapKV) / med. ([Cai et al. 2024]) |
| **Ada-KV** | Allocate the eviction budget *across heads* by contribution. | Consistent gains at equal budget. | [RESEARCH-PROVEN] | low / med. ([Feng et al. 2024]) |
| **KIVI (2-bit KV)** | Keys per-channel, values per-token, asymmetric 2-bit + tiny FP residual. | **2-bit**, ~2.6× mem, 2.35–3.47× throughput, near-lossless. | [RESEARCH-PROVEN] | med (custom kernels) / high. ([Liu et al. 2024]) |
| **KVQuant (3-bit)** | Per-channel **pre-RoPE** key quant + sensitivity datatypes + sparse outliers. | **3-bit**, <0.1 ppl, **1M tokens on one A100**. | [RESEARCH-PROVEN] | high / high. ([Hooper et al. 2024]) |
| **RotateKV** | RHT-style rotation for KV quant (outlier-aware), 2-bit. | 2-bit accuracy beyond KIVI. **Directly adjacent to the in-tree STRAND/RHT work.** | [RESEARCH-PROVEN] | high / med. ([RotateKV 2025]) |
| **PagedAttention (vLLM)** | OS-style virtual memory: KV in fixed non-contiguous **16-token blocks** + block table; copy-on-write sharing. | Waste **<4%** (vs 60–80%); 2–4× throughput. **Lossless — pure layout.** | [PROVEN-IN-PROD] | high to build / high. ([Kwon et al. 2023]) |
| **RadixAttention (SGLang)** | All live/recent KV in a **radix tree** keyed on token IDs; longest-shared-prefix walk skips prefill; **LRU-on-leaves** eviction with refcount pinning. | Up to **5×** on prefix-sharing workloads (chat, agents, tree-search). | [PROVEN-IN-PROD] | high / high. ([Zheng et al. 2024]) |
| **Automatic prefix caching (vLLM/TRT-LLM)** | Hash KV blocks by content+position; dedup identical prefixes across requests. | Eliminates shared-prefix recompute; one flag. | [PROVEN-IN-PROD] | low to use / high. (vLLM APC docs) |
| **LMCache / CacheGen / Mooncake** | Tiered KV (GPU→DRAM→SSD→remote); CacheGen encodes KV to a streamable bitstream; Mooncake disaggregates prefill/decode with a cluster KV pool. | Any-position reuse; Mooncake **+59% to +498%** capacity under SLO. **Bottleneck: PCIe — a 50 GB cache is ~15 ms from HBM but ~800 ms from DRAM**, so transfer must be amortized. | [PROVEN-IN-PROD] (enterprise) | high / high. ([LMCache 2025], [CacheGen 2024], [Mooncake 2024]) |

> **Limit to internalize.** Attention-score eviction is *fragile* — dropped tokens are gone permanently and failures are silent and task-dependent ("Pitfalls of KV Cache Compression," arXiv:2510.00231). HIDE therefore: (a) defaults to lossless, (b) prefers **demotion over deletion** (a "dropped" token's KV streams to disk, recoverable), and (c) gates lossy modes per-task with a recall oracle.

### 3.3 Recurrent / compressive memory (the O(1)-state alternative)

- **The structural fact.** Transformer KV grows `O(n)` → memory-bandwidth-bound at long context, and KV eventually exceeds VRAM. SSMs (Mamba-2) and modern linear-attention RNNs (**RWKV-7 "Goose"**) carry a **fixed-size recurrent state → O(1) per step**, compute-bound. ([Dao & Gu 2024], [RWKV-7 2025])
- **Measured flat throughput.** Mamba state stays ~constant (1K vs 128K tokens); reaches ~220K seq on a 24 GB GPU; ~5× decode throughput with constant memory ([arXiv:2507.12442]). **In-tree, RWKV-7 measured flat ~118→119 tps to 8k vs Qwen ~40→8.6 (≈14× at 8k)** — consistent with the published flat-decode claim. This is the Hawking long-context moat.
- **The quality cost — real, and we respect it.** On needle-in-haystack *within* the trained window: **Transformer++ ≈ RWKV-7 > Mamba**; pure recurrence is distraction-prone. The fixed state is a *lossy compression of the past* — exact long-range associative recall is the structural weakness. *Beyond* the trained window it flips (attention fails, SSMs extrapolate). ([arXiv:2506.11305]) → **HIDE routes by task** (§4.4, §4.9) and the field's convergence on *hybrids* (a few attention layers among SSM layers) is on our moonshot list (§8).

### 3.4 Agent memory systems & taxonomy

- **CoALA taxonomy (the vocabulary we adopt).** Four memory types: **Working** (in-context, current cycle), **Episodic** (past experiences/trajectories), **Semantic** (world/project facts), **Procedural** (skills — in weights + in code/prompts). ([Sumers et al. 2023]) §4.6 maps each to a concrete store.
- **MemGPT / Letta.** OS-style virtual context: **main context** (in-window: system + working scratchpad + FIFO of recent msgs) vs **external context** (recall = full history, archival = DB); the LLM moves data across the boundary via *function calls*; memory-pressure → recursive summarization. ([Packer et al. 2023]) HIDE's compiler is this boundary, made deterministic.
- **Mem0.** Extract→update (ADD/UPDATE/DELETE/NOOP) over retrieved similar memories. **LOCOMO: 66.88% vs OpenAI-memory 52.90% (+26% rel.), ~7k vs ~26k tokens (>90% savings), p95 latency 1.44s vs 17.12s (~91% lower).** ([Chhikara et al. 2025]) The headline lesson: *pruning the window beats filling it.*
- **Generative Agents.** Memory stream + retrieval score `α_rec·recency + α_imp·importance + α_rel·relevance` (all min-max'd to [0,1], all α=1; recency = exp decay, factor 0.995; importance = LLM 1–10; relevance = cosine). Reflection fires when summed importance > 150. ([Park et al. 2023]) HIDE's episodic/semantic retrieval uses exactly this scoring shape (§4.6).
- **Reflexion.** Verbal self-reflection stored as episodic memory, prepended next attempt. **HumanEval 91% (>GPT-4's 80%); +22% AlfWorld, +20% HotPotQA.** ([Shinn et al. 2023]) → our procedural/episodic "lessons" store.
- **A-MEM (2025).** Zettelkasten-style auto-linked notes; new memories trigger evolution of old ones. ([Xu et al. 2025])
- **Letta sleep-time compute (2025).** A background agent reorganizes memory during idle time, off the critical path. ([Letta 2025]) → our background "memory gardener" (§4.7, §8).

### 3.5 Context engineering & failure modes

- **Lost in the Middle.** U-shaped curve: answer recalled well when *first* or *last*, badly in the *middle* (≈15–30 pt drop). ([Liu et al. 2023a]) → ordering policy (§4.2: pin to head/tail).
- **Context Rot (Chroma 2025).** All 18 frontier models degrade *continuously as input grows, below the window limit*; worse as needle–question similarity drops; **gradual, not a cliff**. A 200k-window model can degrade by ~50k tokens. ([Hong et al. 2025]) → §6 is built around this.
- **Anthropic, Effective Context Engineering (2025).** "Smallest possible set of high-signal tokens"; **just-in-time retrieval** (keep identifiers, load at runtime — "progressive disclosure"); **compaction** (summarize near the limit, reinitiate; cheapest step = tool-result clearing); **structured note-taking** (NOTES.md persisted outside the window); **sub-agent isolation** (clean windows, return 1–2k-token summaries); context as a finite "attention budget." ([Anthropic 2025]) HIDE implements every one of these as a typed mechanism.
- **RAG vs long-context.** 4k+retrieval can match 16k-finetuned ([Xu et al. 2023b]); **OP-RAG**: keep chunks in *document order*, inverted-U — ∞Bench EN.QA Llama3.1-70B full-context (~117k tok) **34.26 F1** vs OP-RAG @48k tok **47.25 F1** (better at ~60% fewer tokens) ([Yu et al. 2024]); **hard negatives** from strong retrievers can *hurt* ([Jin et al. 2025]). → the compiler caps chunk count at the inverted-U peak and preserves order.
- **Recursive/hierarchical summarization.** Running-summary update `(prev + new) → fresh summary`; hierarchical merge up a tree. Underpins production auto-compaction. ([Wang et al. 2023])
- **Peer practice.** Claude Code: auto-compaction near saturation, CLAUDE.md, subagents with fresh windows, JIT `glob`/`grep`. Cursor: semantic codebase index (Turbopuffer), 2-stage retrieve+rerank, @-mentions. Windsurf: local-first index + passive "awareness engine." (ch.05 owns the index; this chapter consumes it.)

---

## 4. The Hawking design (concrete)

### 4.1 System architecture & module layout

The substrate is four cooperating layers. Data flows **down** on a turn (sources → compiled window) and **up** over time (window → memory tiers → disk).

```
                          ┌───────────────────────────────────────────────┐
   user turn  ───────────▶│  (A) CONTEXT COMPILER   (per turn, deterministic)│
                          │   budget → gather → score → pack → manifest     │
                          └───────────────┬───────────────────────────────┘
                                          │ emits ContextManifest + packed token stream
                                          ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ (B) EXTENDED-CONTEXT LAYER   position/attention policy per task profile     │
   │     RoPE/YaRN scale · attention sinks · sliding-window+sink · SSM route     │
   └───────────────┬──────────────────────────────────────────────────────────┘
                   │ forward()/prefill_slot_from_pos()/copy_kv_prefix_to_slot()
                   ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ (C) KV STORE   GPU(slots) ⇄ RAM(prefix cache) ⇄ disk(DSPRFKV2) ⇄ checkpoints│
   │     paging · eviction(KvEvictionPolicy) · quant(f16/int4) · KV bank · resume│
   └───────────────┬──────────────────────────────────────────────────────────┘
                   │ promotion/decay, content-addressed writes
                   ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ (D) MEMORY   working · episodic · semantic · procedural · forever-project   │
   │     schemas · stores(SQLite+vec+files) · promotion/decay · retrieval        │
   └──────────────────────────────────────────────────────────────────────────┘
```

**Module layout** (proposed; binds to existing files where they exist):

```
hawking-context/                      # NEW crate — the shell-side compiler & memory (HTTP-only deps)
  src/
    compiler.rs        # ContextCompiler: budget→gather→score→pack→manifest
    manifest.rs        # ContextManifest schema (Appendix A.1) — serde + replay
    budget.rs          # TokenBudget, per-region reservations, tokenizer-accurate counting
    sources/           # one module per ContextSource (trait-based, pluggable)
      system.rs  plan.rs  code.rs  symbols.rs  tool_output.rs  memory.rs  scratchpad.rs  diagnostics.rs
    pack.rs            # the packing/knapsack algorithm + ordering (head/tail anti-LITM)
    compress/          # summarization & distillation
      recursive.rs  draft_compact.rs  beacon.rs  dedup.rs
    memory/            # the (D) layer
      store.rs         # MemoryStore trait + SQLite/FTS5/sqlite-vec impl
      working.rs  episodic.rs  semantic.rs  procedural.rs
      project.rs       # forever project memory (.hide/memory/*) + provenance + versioning
      retrieval.rs     # recency/importance/relevance scorer (Generative-Agents shape)
      decay.rs         # promotion/decay/garbage-gardener policies
    profiles.rs        # per-task ContextProfile registry (§4.9)
    kvclient.rs        # talks to hawking-serve over HTTP (today) / FFI (later)

hawking-core/src/stateful/            # EXISTING — runtime-side KV intelligence
  working_set.rs       # KvEvictionPolicy{Streaming,H2O,SnapKV,Lossless} — already stubbed
  prefix_cache.rs      # RAM prefix-KV tier — interfaces present, bodies oracle-gated
  attn_capture.rs      # attention-score capture feeding H2O/SnapKV
hawking-core/src/cache/
  prefill_disk.rs      # on-disk DSPRFKV2 KV tier — shipped
hawking-serve/src/
  system_kv_bank.rs    # hash(prefix)→source_slot bank — shipped
  http.rs              # the surface the shell binds to; gains optional /v1/hawking/kv/* (later)
```

> **Why a separate `hawking-context` crate.** The compiler and memory are **shell concerns** (per scoping: "shell first"). They depend only on the *HTTP surface* of `hawking-serve` plus a tokenizer, so they ship and test without any kernel or `.tq` dependency. The runtime-side KV intelligence stays in `hawking-core` where the model lives. The two meet at a thin `kvclient.rs` seam.

---

### 4.2 The Context Compiler

The Context Compiler runs **once per agent turn** (and on demand for previews). It is a pure function of `(state, sources, budget, profile)` → `(packed_window, manifest)`. Determinism is a hard requirement: same inputs ⇒ byte-identical window ⇒ replayable turn.

#### 4.2.1 Inputs

```rust
struct CompileInput<'a> {
    profile: &'a ContextProfile,          // §4.9 — task-specific budget & policy
    model: ModelDescriptor,               // ctx_len, tokenizer, arch (transformer|ssm), footprint
    state: &'a SessionState,              // turn index, plan, open files, recent tool calls
    sources: Vec<Box<dyn ContextSource>>, // pluggable providers (§4.7 plugin point)
    fixed_prefix: Option<PrefixHandle>,   // already-banked system KV (skip re-emit if reusable)
    now: Timestamp,
}
```

A **`ContextSource`** is the universal provider interface (the plugin seam, §7):

```rust
trait ContextSource {
    fn kind(&self) -> SourceKind;         // System|Plan|Code|Symbol|ToolOutput|Memory|Scratchpad|Diagnostic|Custom
    /// Produce candidate spans. MUST be cheap: return *handles/identifiers* and
    /// estimated token cost; defer full-body materialization to `realize()`
    /// (Anthropic "just-in-time / progressive disclosure").
    fn candidates(&self, ctx: &CompileCtx) -> Vec<Candidate>;
    /// Materialize the chosen candidate's actual tokens (called only for spans
    /// the packer selected — never for rejected ones).
    fn realize(&self, c: &Candidate, budget_tokens: usize) -> RealizedSpan;
    /// Optional cheaper rendering at a tighter budget (truncation/summary).
    fn degrade(&self, c: &Candidate, target_tokens: usize) -> Option<RealizedSpan> { None }
}

struct Candidate {
    id: SpanId,                 // content-addressed (blake3 of canonical form)
    kind: SourceKind,
    title: String,             // human label for the manifest/UI
    est_tokens: usize,         // estimate before realize()
    base_priority: Priority,   // source-declared importance band (see scoring)
    provenance: Provenance,    // path:line range, tool-call id, memory id, commit sha…
    recency: Timestamp,        // for decay scoring
    pin: PinState,             // User-pinned | Never-evict | Normal
    relevance_key: RelevanceKey, // embedding handle or token set for relevance scoring
}
```

#### 4.2.2 Scoring

Each candidate gets a scalar **value** and a **token cost**; packing maximizes total value under the budget. The score blends the source's declared band with the three signals the literature converges on (recency / importance / relevance), plus a pin override and an **anti-rot redundancy penalty**.

```
value(c) =  w_band   · band(c)                      # source-declared band, [0,1]
          + w_rel    · relevance(c, query)           # cosine(embed(c), embed(query)) ∈ [0,1]
          + w_rec    · recency(c, now)               # exp decay, half-life by profile
          + w_imp    · importance(c)                 # LLM/heuristic salience ∈ [0,1]
          − w_redund · redundancy(c, selected)       # max cos-sim to already-selected spans
          + PIN_BONUS · 𝟙[pin == User-pinned]        # large constant; user pins float to top
value(c) := +∞   if pin == Never-evict               # system prompt, safety rails, hard rules
value(c) := −∞   if staleness(c) marks it invalidated (e.g. file changed since capture)
```

- Weights `w_*` are **per-profile** (§4.9): a *debugging* profile weights diagnostics + recency; a *refactor* profile weights symbols + relevance.
- `relevance` reuses the runtime's `embed()` (already powering `/v1/embeddings`) so code/symbol/memory candidates are embedded with the *same model* that will read them.
- `recency` half-life differs by tier: tool outputs decay fast (minutes), project memory barely decays.
- `redundancy` is the explicit **anti-context-rot** term: two near-duplicate spans (e.g. the same function pasted by two tools) cannot both win — the second's value collapses. This directly fights the "fill the window with distractors" failure (Context Rot, hard negatives).

#### 4.2.3 The packing algorithm (pseudocode)

Packing is a **budgeted, reservation-aware, ordering-sensitive knapsack**. Exact 0/1 knapsack is NP-hard, but our item count is small (tens–low hundreds) and items are tiered, so a **greedy density pass with reserved bands + a bounded local-improvement sweep** is both fast and near-optimal in practice. Ordering then defeats lost-in-the-middle.

```python
def compile_context(input) -> (Window, Manifest):
    B = TokenBudget.from_profile(input.profile, input.model.ctx_len)
    # --- 0. Hard reservations (carved out before anything competes) ---
    reserve = {
        "system":     B.reserve_pct(input.profile.system_pct),     # never-evict
        "response":   B.reserve_pct(input.profile.response_pct),   # leave room to GENERATE
        "scratchpad": B.reserve_pct(input.profile.scratchpad_pct), # agent working notes
    }
    free = B.total - sum(reserve.values())

    # --- 1. Gather cheap candidates (NO body materialization) ---
    cands = []
    for src in input.sources:
        cands += src.candidates(ctx)            # handles + est_tokens only
    # Reuse: if a fixed system prefix is already banked, mark it zero-cost-to-emit
    if input.fixed_prefix and prefix_reusable(input.fixed_prefix):
        mark_banked(cands, input.fixed_prefix)  # KV reused; tokens not re-prefilled (§4.5)

    # --- 2. Score ---
    selected = []                               # for redundancy penalty (greedy, order matters)
    for c in cands:
        c.value = score(c, input.query, input.profile, selected=[])  # base, pre-redundancy
    # Pinned + never-evict are admitted unconditionally (subtract their cost first)
    pinned = [c for c in cands if c.pin in (UserPinned, NeverEvict)]
    free  -= sum(min(c.est_tokens, c.max_tokens) for c in pinned)
    selected += pinned
    pool = sorted([c for c in cands if c not in pinned],
                  key=lambda c: c.value / max(c.est_tokens, 1), reverse=True)  # value density

    # --- 3. Greedy density fill with on-the-fly degrade ---
    for c in pool:
        if free <= 0: break
        c.value = score(c, input.query, input.profile, selected)  # NOW penalize redundancy vs selected
        if c.value <= 0:  # dominated by an already-selected near-duplicate, or stale
            record_drop(c, reason="redundant_or_low_value"); continue
        cost = c.est_tokens
        if cost <= free:
            span = c.source.realize(c, budget_tokens=cost)         # materialize ONLY now
            selected.append(with_span(c, span)); free -= span.tokens
        else:
            # try a degraded (summarized/truncated) rendering that fits
            deg = c.source.degrade(c, target_tokens=free)
            if deg and deg.tokens <= free and value_of(deg) > 0:
                selected.append(with_span(c, deg)); free -= deg.tokens
                record_compaction(c, deg)
            else:
                record_drop(c, reason="no_fit")

    # --- 4. Bounded local improvement (swap a low-density included span for a
    #         higher-value excluded one that now fits after a degrade). O(k²) capped. ---
    local_improve(selected, dropped, free, max_iters=profile.improve_iters)

    # --- 5. ORDER to defeat lost-in-the-middle: pin highest-value spans to the
    #         HEAD and TAIL of the window; bury filler in the middle. ---
    ordered = order_head_tail(selected, profile.ordering)
    #   layout (transformer): [system | high-value-A | … middle (low-value) … | high-value-B | recent | scratchpad]
    #   sinks: ensure the first ATTENTION_SINKS positions are stable (§4.4)

    window   = render(ordered, reserve)         # final token stream
    manifest = ContextManifest.build(ordered, dropped, B, reserve, input.profile)  # Appendix A.1
    assert tokens(window) + reserve["response"] <= B.total   # invariant
    return window, manifest
```

**Key properties.**
- **Just-in-time materialization.** `realize()` runs *only* for selected spans — we never pay to render context that loses. This is Anthropic's progressive disclosure made mechanical, and it is cheap because our sources return identifiers (file paths, symbol ids, memory ids), not bodies.
- **Reserve-then-fill** guarantees the model can *generate* (`response` reservation) and *think* (`scratchpad` reservation) — overflow that eats the response budget is the #1 naive-stuffing bug.
- **Degrade ladder** means a span never silently disappears because it was one token too big; it shrinks (truncate → summary → reference-only).
- **Ordering** is where we beat the U-curve: the compiler *places* high-value content at the extremes deliberately.
- **Determinism.** Stable sort tie-breaks on `SpanId`; `realize()` is pure given content hash; the manifest records exact ordering — so the turn replays.

#### 4.2.4 Where it runs against today's surface **[SHELL-TODAY]**

The compiler emits a final **token stream**, which the shell sends as the prompt to `/v1/hawking/generate` (lean, token-or-text in) or as the assembled `messages`/`prompt` to `/v1/chat/completions` // `/v1/completions`. The fixed-system-prefix reuse is *already* served by `SystemPromptKvBank` on the serve side — the shell just keeps the system span byte-stable across turns and the bank does the rest. Embeddings for relevance scoring come from `/v1/embeddings`. **No kernel work is required for the compiler to ship.**

---

### 4.3 The context manifest (UI + replay contract)

The manifest is the **single source of truth** for what the model saw, why, and what was dropped. The UI renders it as a live, inspectable "context stack" (ch.01/ch.09 own the surface); the agent loop (ch.02) reads it for replay; ch.05 cross-checks code spans against the index. **This is a binding contract — see [Appendix A.1] for the normative schema.**

Design points:
- **Every retained span** carries `{id, kind, title, provenance, tokens, value, signals{recency,importance,relevance,redundancy}, pin, banked?, compacted_from?}`.
- **Every dropped/degraded candidate** is recorded with a reason (`no_fit | redundant | stale | low_value`) and its would-be cost — so the user can *see* what was left out and one-click pin it back in.
- **Budget accounting** is explicit: total, per-region reservations, used, free, and the model's `ctx_len` and effective (post-scaling) length.
- **Content-addressing**: `id = blake3(canonical_span)`. Two turns that include the same span share the id → cross-turn diffing is trivial and the replay store deduplicates.
- **Replayability**: manifest + content store ⇒ exact window reconstruction. The manifest is small (handles, not bodies); bodies live in a content-addressed cache keyed by `id`.

UI affordances this enables (cheap because the data is already there): a bar showing window fill by source color; hover-to-see-provenance; "dropped (12) ▸" expander; drag-to-pin; "why is this here?" tracing a span back to the tool call or memory that produced it.

---

### 4.4 Elastic / extended context length

This is the section the user pointed at directly. **Because we own the runtime, context length is a dial, not a SKU.** Four levers, composed and selected per task profile.

#### 4.4.1 RoPE / YaRN position scaling **[RUNTIME-SIDE — partly LATER]**

- **What.** Apply position-index rescaling so the model attends coherently beyond its native trained length. PI for the simple case; **YaRN** (3-band NTK-by-parts + attention-temperature) as the workhorse; **Dynamic NTK** so short contexts are untouched and the factor grows only as the sequence does.
- **Why we can.** RoPE is applied *inside our forward pass*. We set the scaling factor at load or per-request; cloud cannot let a user dial θ.
- **Design.** A `PositionPolicy` carried on the `ContextProfile`:
  ```rust
  enum PositionPolicy {
      Native,                              // no scaling (default for ≤ ctx_len)
      Pi      { scale: f32 },              // linear interpolation
      Yarn    { scale: f32, beta_fast: f32, beta_slow: f32, attn_temp: f32 },
      DynamicNtk { trained_len: usize },   // factor = f(current_len), short-ctx safe
  }
  ```
  The temperature default follows YaRN: `attn_temp = 0.1·ln(scale) + 1`.
- **Status.** **[RUNTIME-SIDE — LATER]** for a new request-time override (needs a `rope_scaling` field threaded through `prefill`/forward). **[SHELL-TODAY]** fallback: if a model is loaded with a fixed YaRN config (as Qwen ships), the shell simply *uses* the longer window — the dial is "pick the right model build," and the profile records the effective length.
- **Gotcha.** Scaling trades a little short-context precision for long-context reach; LongRoPE2's needle-driven recovery shows this is the regression to watch. The profile therefore only scales when the *task* needs it (a 200-line file edit stays `Native`).

#### 4.4.2 Attention sinks / StreamingLLM **[RUNTIME-SIDE — LATER]**, with shell-side analog today

- **What.** Keep the first **4** "sink" tokens pinned plus a rolling recent window; evict the middle. The model stays coherent over effectively infinite streams at fixed cache.
- **Why we can.** Sinks are a KV-management policy *inside* the cache. We already have the policy type: `StreamingLlmPolicy { sinks, recent }` in `working_set.rs`. Wiring it means the cache stops growing and the head/tail stay stable.
- **Design.** Streaming is a `WorkingSetMode::Bounded` + `StreamingLlmPolicy(sinks=4, recent=R)` choice on the profile. The compiler cooperates by **always placing the system/sink content in the first positions** so the sinks are meaningful, and by keeping the recent window aligned with the scratchpad/last-tool region.
- **Status.** **[RUNTIME-SIDE — LATER]** to wire the policy into the live arena (the `on_admit`/`select_evictions` bodies are `todo!()`). **[SHELL-TODAY]** fallback: the compiler emulates the *effect* at the prompt level — it caps the window, pins the system head, and keeps a recent tail — so even before the GPU policy lands, the *shell* delivers bounded-window streaming against `/v1/chat/completions`.

#### 4.4.3 Sliding-window + global/sink hybrid

- **What.** Local window per layer (cheap, linear) + a few global tokens that everything attends to (the system rules, the current file's signature, the task statement). Combine with sinks for the best of both.
- **Why it matters here.** For *very* long sessions where most context is locally relevant but a handful of facts must be globally visible, this is the right shape — and it composes with the compiler's head/tail ordering (the "global" tokens are exactly the head-pinned spans).
- **Status.** **[RESEARCH-PROVEN]**, **[RUNTIME-SIDE — LATER]** as a Metal attention variant; profile-selectable when present.

#### 4.4.4 Route unbounded-history tasks to the SSM path (RWKV-7 / Mamba-2)

- **What.** For tasks whose value is *flat throughput over a huge running history* rather than exact long-range lookup — "watch this log stream," "follow this long refactor narrative," "summarize as we go" — route to **RWKV-7** or **Mamba-2**, which carry `O(1)` state (RWKV-7 ~6 MiB at any depth) and decode at *flat* tps where the transformer collapses (in-tree: ~118→119 vs ~40→8.6 to 8k).
- **Why we can.** We ship both engines. The model layer (ch.06) is a routing decision, and the compiler is architecture-aware (`ModelDescriptor.arch`).
- **The honest tradeoff.** SSMs lose on exact needle recall within-window. So the routing rule is explicit: **exact-recall / multi-file precise edit → transformer (+ scaling if needed); unbounded-history / streaming / "keep the gist of everything" → SSM.** Hybrids (§8) are the long-term answer.
- **Status.** **[PROVEN-IN-PROD in-tree for RWKV-7 serve]** (the SSM moat is measured); routing policy is **[SHELL-TODAY]** (pick the endpoint/model per profile).

#### 4.4.5 Per-task context-length profiles the user can dial

The user gets a **dial** (literally a UI control + a profile field): *Tight / Standard / Long / Unbounded*, mapping to `(model, position_policy, working_set_mode, eviction_policy, source weights)`. See §4.9 for the table. The point the user made — *"these kinds of things we can adjust… offer people better opportunities"* — is realized as a first-class, persisted, per-project, per-task setting.

---

### 4.5 KV cache as a first-class managed store

We treat the KV cache like a **memory-mapped, tiered, paged store** — because we own it. This subsection defines the store interface, the tiers, eviction/compression, reuse banking, and checkpoint/restore. **The interface is a binding contract — [Appendix A.4].**

#### 4.5.1 Tiers & residency

```
TIER 0  GPU slots          continuous-batch multiseq arena; the live window's KV (fast)
TIER 1  RAM prefix cache    stateful/prefix_cache.rs — recently-used prefixes' KV in host RAM
TIER 2  DISK (DSPRFKV2)     cache/prefill_disk.rs — post-prefill KV on SSD, rolling-hash addressed
TIER 3  CHECKPOINTS         named, durable KV snapshots for session resume (§4.5.5)
```

- **Promotion/demotion.** On a turn, the needed prefix is looked up longest-first across T1→T2 (rolling prefix hash; both tiers agree on the address by construction — `prefix_cache.rs` is byte-compatible with `prefill_disk.rs`). A hit **promotes** the bytes into a GPU slot via `copy_kv_prefix_to_slot` and prefills only the tail with `prefill_slot_from_pos`. A finished slot's prefix **demotes** to T1 (and async to T2) instead of being dropped.
- **Bottleneck honored.** Per the LMCache/PCIe reality (a 50 GB cache is ~15 ms from HBM but ~800 ms from DRAM), demotion to T1/T2 is **asynchronous and amortized**, and we only promote a prefix we will actually reuse (the bank's job, below). Budgets: `HAWKING_PREFIX_CACHE_BUDGET_MB` (RAM), disk budget on `PrefillDiskCache::with_budget_bytes`.

#### 4.5.2 Paging

- **[RUNTIME-SIDE — LATER].** The PagedAttention model (fixed 16-token blocks + block table, <4% waste, lossless) is the target layout for the Metal arena so KV is non-contiguous and shareable copy-on-write. Today's arena is slot-strided; paging is the upgrade that makes T0↔T1 movement and cross-slot prefix sharing cheap.
- **[SHELL-TODAY] fallback.** The disk tier already gives us the *durability and reuse* benefit without paging; paging is a throughput/footprint optimization, not a correctness gate.

#### 4.5.3 Eviction & compression (the `KvEvictionPolicy` framework — already in-tree)

The runtime *already* defines the abstraction this chapter needs (`stateful/working_set.rs`): a `KvEvictionPolicy` trait with `on_admit` → `observe_attention` → `select_evictions`, returning an `EvictionPlan` of `(position, EvictionAction::{Drop | Compress})`, governed by a `WorkingSetBudget`, in `WorkingSetMode::{Bounded | Lossless}`. The policies are stubbed: `StreamingLlmPolicy`, `H2OPolicy`, `SnapKvPolicy`, `LosslessPolicy`.

HIDE's contribution is to (a) **wire the bodies** (oracle-gated, §6) and (b) **make the policy a per-profile choice** with a crucial twist:

- **`EvictionAction::Compress` is "demote, don't delete."** Where cloud eviction *drops* a token forever (the fragility cited in §3.2), our `Compress` action re-encodes the position's K/V at lower precision (`f16`→`int4` via the existing KV-quant levers) *and* streams the full-precision bytes to T2 disk. The position stays attendable at low precision in-window and is *recoverable* at full precision from disk if the task later needs it. **This is the local superpower applied to eviction: nothing is truly evicted.**
- **Policy selection:**
  - `LosslessPolicy` (default) — keep all; greedy-exact; `is_lossless()==true`.
  - `StreamingLlmPolicy(4, R)` — infinite-stream tasks (Unbounded profile).
  - `SnapKvPolicy` — compress the *prompt* KV at prefill-end (huge code prompts) down to load-bearing positions.
  - `H2OPolicy` — long generative tasks; keep recent + heavy-hitters by cumulative attention (fed by `attn_capture.rs`).
  - `PyramidKv`/`Ada-KV` budget-shaping — **[NEW]** layer-/head-varying budgets atop the above; high impact, low marginal cost.
- **Quantized KV** is the compression codec: `f16-KV` (shipped lever) for a safe 2× footprint cut; `int4-KV` (staged, `HAWKING_QWEN_INT4_KV_EXPERIMENTAL`) and the **KIVI** per-channel-key/per-token-value scheme as the aggressive target; **RotateKV** (RHT-for-KV) is a natural fold-in given the in-tree STRAND/RHT rotation work.

#### 4.5.4 Prefix / KV reuse banking

- **Shipped:** `SystemPromptKvBank` (`system_kv_bank.rs`) maps `hash(system-prefix) → source_slot`, surviving a request finishing, so serial chat with a shared system prompt never re-prefills the system block. It stores **zero KV bytes** (a routing hint), and every hit is re-verified by the bit-identical `copy_kv_prefix_to_slot` + `prefill_slot_from_pos` path → **greedy-lossless**, with FNV-folded 128-bit keys, LRU to `max_entries`, and `/metrics` counters.
- **HIDE extension — the *block store* the bank routes into.** The bank's documented "deferred half" is a **slot-independent KV-block store** so reuse survives even when *no* slot currently holds the bytes. HIDE specifies this as **T1/T2** above (`prefix_cache.rs` RAM + `prefill_disk.rs` disk), addressed by the *same rolling prefix hash*. The bank becomes the **router**; the tiered store is the **backing**. Net: a coding session that re-sends "system + CLAUDE.md + open-file headers" every turn pays the prefill *once ever*, then reuses across turns, restarts (disk), and slots.
- **Why this is decisive for coding.** Coding workloads re-send the same files, imports, and scaffolding constantly — the ideal prefix-reuse case (the `prefix_cache.rs` doc says exactly this). On a single-user local box, the hit rate is high and the savings are pure (skipped forward passes = power not drawn = the energy verdict is "genuine").

#### 4.5.5 KV checkpoint / restore (resume)

- **What.** Name and snapshot the full KV state of a session to disk (T3), and restore it to resume *instantly* — no re-prefill of the whole history — after a quit, crash, or machine sleep.
- **Why we can.** The KV is ours and the disk format already exists (`DSPRFKV2` stores layer-major f32 K/V + the token ids + model/tokenizer hashes for validation). A checkpoint is a *named* DSPRFKV2-style blob (plus the position policy + manifest) keyed to `(model, tokenizer, session_id, turn)`.
- **Design (interface in [Appendix A.4]):**
  ```
  kv_checkpoint(session_id, label) -> CheckpointId       # snapshot live KV + manifest + position policy
  kv_restore(CheckpointId) -> RestoredSession            # validate model/tokenizer hash; warm GPU slot
  kv_list(session_id) -> [CheckpointMeta]                # for a "resume / time-travel" UI
  ```
- **Determinism dividend.** Because the manifest + checkpoint fully describe the state, "resume" and "replay from turn N" are the same operation. This is the substrate for the agent loop's time-travel/undo (ch.02) and for reproducing a bug report.
- **Status.** **[RUNTIME-SIDE — LATER]** for live-GPU snapshot/restore (needs a serve endpoint, e.g. `POST /v1/hawking/kv/checkpoint`). **[SHELL-TODAY]** fallback: the *shell* checkpoints the **conversation + manifest** (not the raw KV) and on resume re-establishes context via the prefix cache — slower to warm but correct and available now.

---

### 4.6 Hierarchical memory (working / episodic / semantic / procedural)

We adopt the **CoALA taxonomy** verbatim and give each type a concrete store, schema, promotion/decay policy, and retrieval path. **The schemas are binding contracts — [Appendix A.2].**

| Type | What it holds (coding-agent) | Store | Lifetime |
|---|---|---|---|
| **Working** | The current turn's compiled window: plan, open files, last tool outputs, scratchpad. | In-context (the compiled window itself) + a small RAM ring. | One turn → compacted into episodic. |
| **Episodic** | Past *experiences*: "edited `auth.rs` to fix the JWT bug, tests passed"; tool-call traces; **Reflexion-style lessons** from failures. | SQLite rows + vec index. | Decays; salient ones promote to semantic. |
| **Semantic** | Project *facts*: "the DB layer is in `db/`, uses sqlx"; API contracts; user preferences; durable conclusions. | Project memory store (§4.7) + vec index. | Forever (versioned), low decay. |
| **Procedural** | *Skills*: repo-specific recipes ("to add a migration: …"), the agent's own prompt/tool config, learned command sequences. | Versioned files in `.hide/memory/procedural/` + (implicitly) model weights. | Forever; user-editable. |

#### 4.6.1 Stores (concrete)

- **Primary store: SQLite** with **FTS5** (keyword) + **sqlite-vec** (vector) extensions — one embedded, file-backed, transactional store at `.hide/memory/memory.db`. No server, no daemon; it is *the user's file*, greppable and backup-able. (Local superpower: the memory is on disk, durable, and inspectable.)
- **Project memory** additionally mirrors to **human-readable files** (`.hide/memory/*.md` with structured front-matter) so the user can open and edit in the IDE (§4.7).
- **Embeddings** come from the runtime `embed()` (`/v1/embeddings`) — the memory is embedded by the *same model family* that reads it, so relevance is calibrated.

#### 4.6.2 Memory record schema (normative shape; full in A.2)

```jsonc
{
  "id": "mem_01H…",                  // ULID, time-sortable
  "type": "episodic|semantic|procedural",
  "content": "…natural-language fact/experience/skill…",
  "embedding_ref": "vec:…",          // handle into sqlite-vec
  "importance": 0.0,                 // [0,1]; LLM 1–10 rescaled (Generative-Agents)
  "created_at": "…", "last_access": "…", "access_count": 0,
  "decay_half_life_days": 30,        // type-dependent; semantic ≫ episodic
  "provenance": {                    // WHERE it came from — non-negotiable
    "source": "tool_call|user_edit|reflection|file_scan|conversation",
    "ref": "auth.rs:42-88 @commit abc123 | toolcall_… | turn_17",
    "confidence": 0.0                // [0,1]
  },
  "links": ["mem_…", …],             // A-MEM-style graph edges (related/supersedes)
  "supersedes": "mem_…",             // version chain; old facts are retired, not erased
  "pinned": false,                   // user override → never decays, always retrievable
  "tags": ["auth","security"],
  "version": 3
}
```

#### 4.6.3 Retrieval (Generative-Agents scoring, made deterministic)

```
retrieve(query, k, type_filter) =
   top_k over store where type ∈ filter of
     score = α_rec·recency(now − last_access)        # exp decay, factor per half-life
           + α_imp·importance
           + α_rel·cosine(embed(query), embedding)
     (all three min-max normalized to [0,1]; α default 1,1,1; per-profile override)
   then re-rank by an LLM/cross-encoder pass for the final small set (Cursor-style 2-stage)
   then ON ACCESS: bump access_count, last_access (feeds recency next time)
```

#### 4.6.4 Promotion & decay

- **Promotion (episodic → semantic).** When an episodic memory is accessed ≥ τ times *or* a reflection synthesizes it into a general fact, it is promoted (or a new semantic memory minted with a `links` edge back). Mirrors Generative-Agents reflection (trigger: accumulated importance over a window exceeds a threshold) and Mem0's UPDATE op.
- **Decay.** Each retrieval recomputes recency; un-accessed episodic memories sink below the retrieval cutoff (effectively forgotten) but are **not deleted** — they stay in SQLite, recoverable by explicit search. Semantic/procedural decay is near-zero.
- **Consolidation (background "memory gardener," §4.7/§8).** Off the critical path (Letta sleep-time-compute pattern): de-duplicate near-identical memories (MERGE), retire superseded facts (SUPERSEDE with a version edge), and re-summarize clusters. This is what keeps the store from rotting.

---

### 4.7 Forever project memory (CLAUDE.md on steroids)

The single most user-visible promise: **a project memory that never resets, that the user can read, edit, and trust.** CLAUDE.md is the seed; HIDE makes it *structured, queryable, auto-maintained, versioned, and provenance-tagged* — without losing its hand-editability.

#### 4.7.1 Layout & format

```
.hide/memory/
  project.md            # the human-facing "CLAUDE.md on steroids": curated, sectioned, editable
  semantic/             # auto-maintained structured facts (one file per topic) w/ front-matter
    architecture.md  conventions.md  apis.md  preferences.md  gotchas.md
  procedural/           # recipes/skills (how-to-do-X for THIS repo)
  episodic/             # rolled-up experience logs (rotated, summarized)
  memory.db             # the SQLite/FTS5/vec index over all of the above (derived, rebuildable)
  .provenance.jsonl     # append-only provenance & version log (audit trail)
```

Each structured memory file is **markdown with YAML front-matter** so it is *both* human-readable in the IDE *and* machine-parseable:

```markdown
---
id: sem_arch_db
type: semantic
importance: 0.9
provenance: { source: file_scan, ref: "db/mod.rs @abc123", confidence: 0.95 }
last_verified: 2026-06-24
pinned: true
version: 4
---
# Database layer
The project uses `sqlx` against Postgres. Migrations live in `db/migrations/`,
applied via `make migrate`. Connection pool is built in `db/pool.rs`.
```

#### 4.7.2 Injection (how it reaches the model)

Project memory is **not** dumped wholesale into the window (that is how CLAUDE.md rots and bloats). Instead:
- The **system/never-evict band** gets only the *pinned, high-importance core* (`project.md` curated head + pinned facts) — small, always present.
- Everything else enters via the **compiler as a `MemorySource`**: candidates are *retrieved* by relevance to the current task (§4.6.3) and compete in packing like any other source — **just-in-time**, not always-on. This is the Anthropic "progressive disclosure" principle applied to memory.

#### 4.7.3 Auto-maintenance (how it is kept *fresh* and prevented from rotting)

Rot is the failure mode that kills long-lived memory. Mitigations, all concrete:

1. **Provenance + verification.** Every fact records where it came from and `last_verified`. A background pass re-checks facts whose `ref` (file path/symbol) changed since `last_verified` (cross-checked against ch.05's code index) and flags **stale** facts for re-derivation or demotion. A fact about `db/mod.rs` is auto-suspected when `db/mod.rs` changes.
2. **Versioning, not overwriting.** Updates create a new `version` with a `supersedes` edge; the old fact is retired (hidden from retrieval) but kept in `.provenance.jsonl`. The user can diff and roll back. Memory has *git-like history*.
3. **Confidence + decay.** Low-confidence or long-unverified facts decay below the retrieval cutoff; they do not silently mislead.
4. **User as the final authority.** Any fact is **editable in the IDE** (it is a file) and **pinnable** (pin → never decays, always injected, marked authoritative). User edits set `confidence=1.0, source=user_edit`. The user can also **delete/quarantine** a wrong memory.
5. **The "memory gardener"** (background, idle-time): de-dup, supersede-resolution, cluster re-summarization, stale-flag sweep. Letta-sleep-time pattern — never on the hot path.
6. **Contradiction detection.** When a new fact contradicts a pinned one, the compiler surfaces a **conflict** in the manifest rather than silently picking one — the user resolves it.

#### 4.7.4 Status

**[SHELL-TODAY] entirely.** Project memory is files + SQLite + retrieval over the HTTP `embed()` surface. It needs no kernel and no `.tq`. It is the highest-leverage, lowest-risk piece of this chapter — and the most directly user-felt.

---

### 4.8 Context compression & distillation

When content must shrink (a 4k-line file, a 30-turn history, a giant tool dump), we compress *cooperatively* — and, crucially, **for free**, because compaction can run on a cheap local draft model with no per-token bill.

#### 4.8.1 Recursive / hierarchical summarization

- **Conversation compaction.** Maintain a running summary; near saturation (or on `/compact`), fold `(prev_summary + new_turns) → fresh_summary`, reinitialize the window with it (Anthropic compaction; recursive-summarization literature). **Preserve** architectural decisions, unresolved bugs, file/symbol references, open TODOs; **discard** redundant tool outputs first ("tool-result clearing" is the cheapest step).
- **Document/file compaction.** Chunk → summarize each chunk → hierarchical merge up a tree to a target size (preserving symbol signatures and the spans the task touched).

#### 4.8.2 Draft-model-powered cheap compaction **[local superpower]**

- **What.** Run summarization/compaction on a **small fast local model** (the same draft model the runtime already loads for speculative decode, or a tiny dedicated summarizer). The big model never spends its turn budget compacting; a 0.5B/1.5B model does it in the background.
- **Why cloud can't.** Cloud compaction spends *paid* tokens of the *same* expensive model (or a separate paid call). We have a free, local, always-warm draft model and idle GPU cycles. **Compaction is effectively free and continuous for us.**
- **Continuous, not just-at-the-cliff.** Because it is free, we compact *proactively in the background* (a tool output is summarized the moment it lands, the long-tail of history is rolled up while the user reads) — so the window is *already* dense when the next turn compiles, instead of scrambling at 95% saturation.

#### 4.8.3 Summary / beacon tokens **[RUNTIME-SIDE — LATER, moonshot-adjacent]**

- **Activation-Beacon-style** learned compression tokens that distill a context chunk into a few KV positions, then drop the raw KV (4k→400k of distilled span). High impact, high cost (trains a module). Deferred to §8, but the *architecture* (we own the KV) is what makes it possible at all.

#### 4.8.4 Learned eviction

- The `KvEvictionPolicy` framework (§4.5.3) *is* learned/attention-driven eviction (H2O/SnapKV). The compression side (`EvictionAction::Compress` → quantized KV + disk demotion) is HIDE's "lossy-but-recoverable" twist that distinguishes it from the cloud's lossy-and-gone eviction.

---

### 4.9 Per-task context profiles

A **`ContextProfile`** bundles every knob into a named, user-selectable, per-project-overridable preset. The user's dial. **(Binding shape in [Appendix A.3].)**

```rust
struct ContextProfile {
    name: String,                         // "tight"|"standard"|"long"|"unbounded"|custom
    model_hint: ModelSelector,            // which engine/build (arch-aware)
    target_ctx_tokens: usize,             // effective window after scaling
    position_policy: PositionPolicy,      // Native | Pi | Yarn | DynamicNtk (§4.4.1)
    working_set_mode: WorkingSetMode,     // Lossless | Bounded
    eviction: EvictionChoice,             // Lossless | StreamingLlm{..} | SnapKv{..} | H2O{..} (+budget shaping)
    kv_precision: KvPrecision,            // F16 | Int4 | Native (the compression codec)
    reservations: Reservations,           // system_pct, response_pct, scratchpad_pct
    source_weights: SourceWeights,        // w_band/w_rel/w_rec/w_imp/w_redund per SourceKind
    recency_half_life: Duration,
    compaction: CompactionPolicy,         // off | reactive(threshold) | proactive(draft_model)
    ordering: OrderingPolicy,             // head_tail anti-LITM placement
    retrieval_k: usize,                   // memory/code candidates pulled per turn
    improve_iters: usize,                 // packer local-improvement budget
}
```

Default presets:

| Profile | Window / position | Working set / eviction | KV prec | Compaction | Best for |
|---|---|---|---|---|---|
| **Tight** | Native (e.g. 8k) | Lossless | Native | reactive | quick edits, single file, lowest latency/energy |
| **Standard** | Native or mild YaRN | Lossless | f16-KV | proactive(draft) | day-to-day agentic coding (default) |
| **Long** | YaRN ~4–8× | Bounded + **SnapKV** (compress big prompt) | f16/int4 | proactive(draft) | multi-file refactor, large files, long sessions |
| **Unbounded** | **SSM (RWKV-7/Mamba-2)** *or* StreamingLLM sinks on transformer | StreamingLLM(4, R) | f16/int4 | proactive(draft) | "watch this stream / follow this long narrative," flat-throughput history |

The dial maps user intent → a coherent, *validated* bundle, instead of forcing the user to understand RoPE. Advanced users edit the profile; the manifest always records which profile (and effective length) produced the turn.

---

## 5. How we EXCEED

The structural thesis, stated plainly:

> **Cloud gives a fixed, paid window. We give an elastic, compressed, persistent, on-disk, model-cooperative context.**

### 5.1 The exceed-list (cloud literally cannot do this)

1. **Dial the context length.** We rescale RoPE/YaRN, add attention sinks, or switch to an `O(1)`-state SSM *per task*. A cloud user gets the SKU's fixed window; **they cannot reach into attention or positions.** (The user's exact ask — delivered.)
2. **Free, continuous, background compaction.** Summarization runs on a local draft model with **no per-token bill**, proactively, while the user reads. Cloud pays for every compaction token of an expensive model and only does it at the saturation cliff.
3. **Enormous resident context, recomputed/recompressed freely.** No per-token cost means we keep huge context warm and re-derive it at will. Cloud rations because every resident token is metered.
4. **Tiered, persistent, disk-backed context the model treats as unbounded.** KV demotes GPU→RAM→disk and *comes back*; "evicted" tokens stream to SSD instead of vanishing. Cloud KV is ephemeral and per-request; their eviction is *lossy-and-gone*. **We demote; we never truly delete.**
5. **Prefix/KV reuse banking across turns, restarts, and slots.** We already bank `hash(system-prefix)→slot` and back it with a disk KV tier addressed by the same rolling hash. The "system + CLAUDE.md + file headers" prefix is prefilled **once ever**. Cross-request prefix caching exists in vLLM/SGLang — but they cannot give a *single local user* a forever, on-disk, cross-restart bank as cheaply as we can.
6. **KV checkpoint/restore = instant resume + time-travel.** Snapshot the literal KV to disk; restore to resume a long session with zero re-prefill; "replay from turn N" is the same operation. Cloud cannot hand you the KV.
7. **Forever, editable, versioned project memory the user owns.** A structured, queryable, provenance-tagged, git-like memory living as files + SQLite on the user's disk. It never resets and the user can open, correct, pin, and roll it back. Cloud "memory" is a managed, opaque, remote service.
8. **Determinism & replay.** Manifest + content-addressed stores reconstruct any turn byte-for-byte. The agent loop is debuggable and reproducible. Cloud context assembly is invisible.
9. **Model-cooperative compression (beacons/summary tokens).** Because we own the KV, learned context-compression tokens are *available to us*; a sealed cloud box can't expose that.
10. **Energy as a real dividend.** Every reused prefix and skipped prefill is power not drawn — a genuine, measurable local win (the in-tree energy verdict), not just a cost line.

### 5.2 Where cloud is genuinely ahead (so we route, not pretend)

Honesty per tenet 10: cloud has bigger native windows on the largest models and more GPUs for brute long-context attention. Our answer is **not** "match their window on a transformer" — it is **(a)** elastic scaling + sinks to stretch our window when needed, **(b)** SSM routing for unbounded-history tasks where their KV wall and per-token bill both bite, and **(c)** retrieval + memory + compaction so we *need* fewer tokens to be just as effective (Mem0's >90% token cut at *better* accuracy is the proof that curated-and-persistent beats big-and-metered).

---

## 6. Failure modes & mitigations

| # | Failure | Symptom | Mitigation (mechanism) |
|---|---|---|---|
| F1 | **Window overflow** | Compiled context + response exceeds `ctx_len`; truncation eats the answer. | **Reserve-then-fill** packer (§4.2.3) carves `response_pct` *first*; the `assert tokens(window)+reserve.response ≤ total` invariant cannot be violated; degrade ladder shrinks spans before dropping. |
| F2 | **Context rot** (degradation as input grows, sub-limit; Chroma) | Quality silently drops on long inputs even within the window. | **Signal density over fill** (objective is value, not fill); **redundancy penalty** kills near-duplicates; **proactive compaction**; the **dial** caps window per task; SSM routing for the truly-long. Don't fill the window just because it's free. |
| F3 | **Lost in the middle** (U-curve) | Mid-window facts ignored. | **Head/tail ordering** (§4.2.3) *places* high-value spans at the extremes; sinks pin the head; recent tail aligned with scratchpad. |
| F4 | **Stale memory** | Memory asserts a fact the code no longer supports. | **Provenance + `last_verified`** with background re-check against ch.05's index; **decay** of unverified facts; **conflict surfacing** in the manifest; **versioning** so a wrong fact is retired not silently trusted. |
| F5 | **Lossy eviction loses something the task needed** (the §3.2 fragility) | A dropped token was load-bearing; silent wrong answer. | **`Compress`=demote-to-disk, not delete** → recoverable at full precision; **`Lossless` default**; eviction modes **oracle-gated** (recall oracle on real coding transcripts before a mode ships, per `prefix_cache.rs`'s gating discipline). |
| F6 | **Prefix-cache / bank false hit** | Reused KV doesn't match the prompt → corrupt output. | Already solved in-tree: the bank is a *routing hint*; every hit is **re-verified** by the bit-identical `copy_kv_prefix_to_slot`+`prefill_slot_from_pos`; a stale slot simply fails the copy and falls back to a cold prefill. Keys bind `(model, tokenizer, tokens)`. **Greedy-lossless guaranteed.** |
| F7 | **Tier-movement stall** (PCIe) | Promoting a big cold prefix blocks the turn (~800 ms from DRAM). | **Async, amortized demotion**; **promote only what we'll reuse** (bank decides); budgets (`HAWKING_PREFIX_CACHE_BUDGET_MB`, disk budget); CacheGen-style compact encoding on T2 (moonshot). |
| F8 | **Memory store bloat** | SQLite/files grow unbounded; retrieval slows; dup facts. | **Decay** below cutoff (not deleted, just not retrieved); **gardener** de-dups/supersedes/re-summarizes off the critical path; FTS5+vec keep retrieval O(log n). |
| F9 | **SSM recall miss** | Unbounded-profile (SSM) misses an exact long-range needle. | **Route by task** — exact-recall tasks never go to SSM; **hybrid** models (§8) recover recall; the dial documents the tradeoff so the user opts in knowingly. |
| F10 | **RoPE-scaling short-ctx regression** | Scaling hurts a short task. | **Dynamic NTK** / scale only when the task needs it (profile-gated); short tasks stay `Native`; LongRoPE2-style needle recovery if/when we train scales. |
| F11 | **Non-determinism breaks replay** | A turn can't be reproduced. | Stable sort + content-addressed `realize()` + manifest records exact ordering & ids; greedy-lossless KV reuse; the checkpoint == replay equivalence. |
| F12 | **Provenance/trust poisoning** | A tool injects a "fact" that becomes trusted memory. | Provenance `confidence` < 1 for tool-sourced facts; only **user_edit** sets confidence 1.0; contradiction-with-pinned surfaces a conflict; the user can quarantine. |

---

## 7. Extensibility / plugin points

Everything that *produces* or *stores* context is a trait, so third parties (and ch.05/ch.06) extend without forking the compiler.

1. **`ContextSource`** (§4.2.1) — the universal provider seam. A plugin registers a source (a custom retriever: a vector DB, a docs site, a ticket system, a different code index, a teammate's shared memory) by implementing `candidates`/`realize`/`degrade`. The compiler ranks and packs it like any built-in. *This is how ch.05's codebase intelligence plugs in as the `Code`/`Symbol` sources.*
2. **`MemoryStore`** (A.2) — swap the backing store (SQLite default; a plugin could back semantic memory with a remote vector DB, a knowledge graph, or a team-shared store). Promotion/decay policies are injectable.
3. **`KvEvictionPolicy`** (in-tree trait) — add a new eviction/compression policy (e.g. PyramidKV budget shaping, a learned policy) behind the existing interface; it becomes profile-selectable.
4. **`PositionPolicy`** — register a new context-extension scheme (a new RoPE-scaling variant, SelfExtend-style grouped attention) as a profile option.
5. **`CompactionPolicy`** — plug a custom summarizer (a different draft model, a structured-extraction compactor, a domain-specific distiller).
6. **`Retriever`/`Reranker`** — the two-stage retrieval (candidate → rerank) is pluggable end to end.
7. **Profiles** — ship/share `ContextProfile` presets per language/stack/team as files.
8. **Manifest consumers** — the manifest is a stable, versioned, public schema (A.1); any tool (UI panels, audit/export, eval harnesses) can read it.

Stability contract: `ContextManifest` and the memory record schema are **versioned** (`schema_version`); additive changes bump minor, breaking changes bump major; consumers pin a major.

---

## 8. Bleeding-edge / moonshots (ranked)

Ranked by **(impact × feasibility) ÷ cost**, each tagged maturity / difficulty.

1. **Wire the existing `KvEvictionPolicy` bodies (StreamingLLM → SnapKV → H2O) + `Compress`=demote-to-disk.** *Highest leverage:* the framework is already in-tree and stubbed; bodies + the lossy-but-recoverable twist deliver bounded-RAM long context that *no one else makes recoverable*. **[RESEARCH-PROVEN tech / med build]**. Gate on the recall oracle.
2. **The slot-independent KV block store backing the `SystemPromptKvBank` (T1+T2).** Turns the shipped bank from "router with no backing when no slot holds it" into forever cross-restart prefix reuse. Mostly composition of existing `prefix_cache.rs`+`prefill_disk.rs`. **[med build / high impact].**
3. **Forever project memory (§4.7) end to end.** Files + SQLite + retrieval over `embed()`; pure shell; biggest *user-felt* differentiator; lowest risk. **[low–med build / very high impact].** *Do this early.*
4. **KV checkpoint/restore for instant resume + time-travel** (`POST /v1/hawking/kv/checkpoint`). The disk format exists; this is a serve endpoint + warm-restore. **[med build / high impact].**
5. **PagedAttention block layout on the Metal arena.** Lossless throughput+footprint+sharing win; unlocks cheap T0↔T1 movement and copy-on-write prefix sharing. **[PROVEN-IN-PROD elsewhere / high build].**
6. **Request-time RoPE/YaRN override + Dynamic NTK** threaded through `prefill`/forward, exposed on the profile. Makes the "dial the length" promise fully runtime-side. **[PROVEN-IN-PROD tech / med build].**
7. **KIVI/RotateKV 2-bit KV** as the aggressive compression codec, folding in the in-tree STRAND/RHT rotation work (RotateKV ≈ RHT-for-KV). **[RESEARCH-PROVEN / high build].**
8. **Hybrid attention–SSM model** (a few full-attention layers among RWKV-7/Mamba-2 layers) to recover exact recall while keeping `O(1)`-ish state and flat throughput — the field's convergence point; resolves F9. **[SPECULATIVE→RESEARCH-PROVEN / high build]** (ties to ch.06 + Hawking Condense).
9. **Background "memory gardener" (Letta sleep-time-compute pattern)** on the always-idle local GPU: de-dup, supersede, re-summarize, stale-sweep, *and pre-compute likely-next context*. **[RESEARCH-PROVEN pattern / med build].**
10. **Activation-Beacon / learned summary tokens** for 100× distilled-span compression. **[RESEARCH-PROVEN / very high build — trains a module].** Furthest out; the KV ownership is what makes it *possible*.
11. **CacheGen-style compact KV encoding on T2** so cold prefixes stream from SSD faster than recompute. **[RESEARCH-PROVEN / high build].**
12. **A-MEM-style self-organizing memory graph** (auto-linked notes that evolve old memories on insert). **[RESEARCH-PROVEN / med build].**

---

## 9. Open questions & dials

- **Eviction oracle thresholds.** What recall hit on *real coding transcripts* is acceptable before SnapKV/H2O ship non-lossless? (Reuses the `prefix_cache.rs` oracle discipline — measure before flipping.)
- **Packer optimality vs cost.** Is greedy-density + bounded local-improvement enough, or do certain profiles want a true DP/ILP pass? Dial: `improve_iters`.
- **Memory promotion threshold τ** (episodic→semantic) and **decay half-lives** per type — defaults from Generative-Agents (0.995/window) but need tuning on coding workloads.
- **Scoring weights** `w_band/w_rel/w_rec/w_imp/w_redund` per profile — to be learned from accept/edit signals (ties to ch.02's outcome telemetry).
- **When does RAG beat long-context for *code*?** The OP-RAG inverted-U peak (chunk count) is task-dependent; dial: `retrieval_k`. We expect retrieval to win for "find the relevant function in a 1M-line repo," long-context to win for "edit this 2k-line file coherently."
- **SSM↔transformer routing policy** — explicit rules now; a learned router later. What's the precise boundary between "exact recall" and "unbounded history" tasks?
- **Position scaling: ship-with-the-model vs request-time.** Request-time override is more flexible but riskier (short-ctx regression). Default conservative, dial to scale.
- **KV precision floor.** How low (f16→int4→2-bit) before coding quality cracks? Reuse Hawking Condense's quality methodology; the `kv_precision` dial.
- **Memory privacy/sharing.** Project memory is the user's files — what is the model for *team-shared* memory (a `MemoryStore` plugin) without leaking secrets?
- **Checkpoint size budget.** Full-KV snapshots are large; dial between full-KV (instant resume) and manifest-only (cheap, re-warm) checkpoints.

---

## 10. Cross-references

- **ch.02 · Agent loop.** The compiler runs once per agent turn; the manifest is the loop's *replay/undo* substrate (checkpoint==replay); sub-agent context isolation (clean windows returning 1–2k-token summaries) is an agent-loop policy that *consumes* this chapter's compiler. Outcome telemetry (accept/edit) feeds the scoring weights (§9). **The agent loop must bind `ContextManifest` (A.1).**
- **ch.05 · Codebase intelligence.** Provides the `Code`/`Symbol` `ContextSource` implementations (semantic index, symbol graph, retrieval); provides the file/symbol change signals that drive memory **staleness re-verification** (F4); is the retriever behind just-in-time code loading. **Binds `ContextSource` (§4.2.1) and consumes provenance refs.**
- **ch.06 · Model layer.** Owns engine selection (transformer vs SSM routing, §4.4.4), the RoPE/YaRN scaling implementation (§4.4.1), KV precision codecs (f16/int4/KIVI), and exposes the `embed()`/prefill/copy-KV seams this chapter drives. **Binds `ModelDescriptor` and the KV-store interface (A.4).** *Hawking Condense* governs the `.tq` footprint that sets which model fits which profile.
- **ch.01 / ch.09 · UI / observability.** Render the context-stack from the manifest, the budget bar, the dropped-spans expander, drag-to-pin, and the memory editor/inspector. The `/metrics` bank/cache counters surface here.

---

## Appendix A — Binding contracts

> These are the schemas other chapters import. They are **versioned**; additive = minor bump, breaking = major bump. Consumers pin a major version.

### A.1 `ContextManifest` (schema_version 1)

```jsonc
{
  "schema_version": 1,
  "turn_id": "turn_01H…",            // ULID
  "session_id": "sess_…",
  "created_at": "2026-06-24T…Z",
  "profile": { "name": "standard", "target_ctx_tokens": 16384,
               "position_policy": "yarn(scale=2,attn_temp=1.07)",
               "working_set_mode": "lossless", "kv_precision": "f16" },
  "model": { "id": "qwen-7b", "arch": "transformer",   // "transformer" | "ssm"
             "ctx_len_native": 32768, "ctx_len_effective": 16384,
             "tokenizer_sig": "blake3:…" },
  "budget": { "total": 16384, "used": 14210, "free": 2174,
              "reservations": { "system": 1200, "response": 2048, "scratchpad": 1024 } },
  "spans": [                          // RETAINED, in final window order (head→tail)
    {
      "id": "blake3:…",               // content address (dedup + replay key)
      "kind": "System",               // System|Plan|Code|Symbol|ToolOutput|Memory|Scratchpad|Diagnostic|Custom
      "title": "System + project core",
      "order_index": 0,
      "tokens": 1200,
      "value": 1.0,                   // +inf encoded as 1.0 here; "pin" carries the reason
      "signals": { "recency": 1.0, "importance": 1.0, "relevance": 0.0, "redundancy": 0.0 },
      "pin": "never_evict",           // never_evict | user_pinned | normal
      "banked": true,                 // KV reused from prefix bank/cache (not re-prefilled)
      "compacted_from": null,         // or { "original_id": "blake3:…", "method": "draft_summary", "ratio": 0.18 }
      "provenance": { "source": "memory", "ref": ".hide/memory/project.md#core", "confidence": 1.0 }
    }
    // … more retained spans …
  ],
  "dropped": [                        // candidates NOT included (for the UI "what was left out")
    { "id": "blake3:…", "kind": "ToolOutput", "title": "full `cargo build` log",
      "would_be_tokens": 4200, "reason": "no_fit",   // no_fit | redundant | stale | low_value
      "value": 0.31, "provenance": { "source": "tool_call", "ref": "toolcall_…" } }
  ],
  "conflicts": [                      // F12/F4: surfaced contradictions for user resolution
    { "between": ["blake3:…","blake3:…"], "note": "pinned arch fact vs new scan", "resolved": false }
  ],
  "kv": { "prefix_reuse_tokens": 1200, "bank_hit": true,    // accounting for §4.5
          "tiers_touched": ["gpu","ram"], "checkpoint_id": null },
  "compaction_events": [ { "original_id": "blake3:…", "result_id": "blake3:…",
                           "method": "draft_summary", "model": "qwen-0.5b", "ratio": 0.18 } ],
  "replay": { "deterministic": true, "content_store": "cas://.hide/cache/spans" }
}
```

### A.2 `MemoryRecord` (schema_version 1) — the store contract

```jsonc
{
  "schema_version": 1,
  "id": "mem_01H…",                   // ULID, time-sortable
  "type": "episodic",                 // working|episodic|semantic|procedural
  "content": "…natural language…",
  "embedding_ref": "vec:…",
  "importance": 0.0,                  // [0,1] (Generative-Agents 1–10 rescaled)
  "created_at": "…", "last_access": "…", "access_count": 0,
  "decay_half_life_days": 30,
  "provenance": { "source": "tool_call|user_edit|reflection|file_scan|conversation",
                  "ref": "auth.rs:42-88 @abc123", "confidence": 0.0 },  // [0,1]
  "links": ["mem_…"], "supersedes": "mem_…",
  "pinned": false, "tags": ["…"], "version": 1
}
```
Retrieval API (binding):
```
MemoryStore::retrieve(query: &str, k: usize, types: &[MemType]) -> Vec<ScoredMemory>
  // score = α_rec·recency + α_imp·importance + α_rel·relevance, all min-max'd, then rerank
MemoryStore::upsert(rec: MemoryRecord) -> MemId           // creates a new version on conflict
MemoryStore::supersede(old: MemId, new: MemoryRecord) -> MemId
MemoryStore::pin(id: MemId, pinned: bool)
```

### A.3 `ContextProfile` (schema_version 1)

The struct in §4.9 is the normative shape; serialized form is the obvious JSON/TOML of those fields. `name` is the dial value the UI exposes; `Tight|Standard|Long|Unbounded` are reserved built-ins; anything else is a user/team custom profile file under `.hide/profiles/`.

### A.4 KV-store interface (the `kvclient` ↔ `hawking-core` seam)

```rust
/// Shell-facing KV operations. Today: HTTP to hawking-serve. Later: FFI.
/// Lossless-by-construction: every reuse is re-verified by the engine's
/// bit-identical copy+prefill-from-pos path (greedy-lossless guarantee).
trait KvStore {
    /// Longest-prefix lookup across tiers; returns a handle if reusable.
    fn lookup_prefix(&self, key: &PrefixKey) -> Option<PrefixHandle>;
    /// Promote a hit into a live slot and prefill only the tail.
    fn warm_into_slot(&mut self, h: &PrefixHandle, full_ids: &[u32]) -> Result<SlotId>;
    /// Demote a finished slot's prefix to RAM (and async to disk). Never deletes.
    fn demote(&mut self, slot: SlotId, prefix_len: usize) -> Result<()>;
    /// Set/inspect the working-set eviction policy + budget for a slot.
    fn set_policy(&mut self, slot: SlotId, p: EvictionChoice, b: WorkingSetBudget);
    /// Checkpoint / restore (resume + time-travel). [RUNTIME-SIDE — LATER]
    fn checkpoint(&mut self, session: &str, label: &str) -> Result<CheckpointId>;
    fn restore(&mut self, id: CheckpointId) -> Result<RestoredSession>;
    fn list_checkpoints(&self, session: &str) -> Vec<CheckpointMeta>;
    /// Stats for the manifest / /metrics.
    fn stats(&self) -> KvStoreStats; // {bank_hits, prefix_reuse_tokens, tier_bytes, evictions, …}
}
```
`PrefixKey` binds `(model_hash, tokenizer_hash, rolling_prefix_hash, n_tokens)` — **byte-compatible with the in-tree `prefix_cache::PrefixKey` / `prefill_disk::PrefillKey`** so RAM and disk tiers agree on a prefix's address, and the `SystemPromptKvBank` routes into them.

---

## Appendix B — Source register

Numbers carrying *(paper-claim)* are authors' reported figures (not independently reproduced) and are gated before entering the bible as measured fact. In-tree measurements (RWKV-7 tps, energy verdict, the existing modules) are HIDE's own.

**Context extension.** PI — Chen et al., "Extending Context Window of LLMs via Position Interpolation," 2023, arXiv:2306.15595. · YaRN — Peng et al., ICLR 2024, arXiv:2309.00071. · LongRoPE — Ding et al., ICML 2024, arXiv:2402.13753. · LongRoPE2 — 2025, arXiv:2502.20082. · SelfExtend — Jin et al., ICML 2024, arXiv:2401.01325. · StreamingLLM — Xiao et al., ICLR 2024, arXiv:2309.17453. · Mistral 7B/SWA — Jiang et al., 2023, arXiv:2310.06825. · Longformer — Beltagy et al., 2020, arXiv:2004.05150. · Activation Beacon — Zhang et al., 2024, arXiv:2401.03462.

**KV compression / management.** H2O — Zhang et al., NeurIPS 2023, arXiv:2306.14048. · Scissorhands — Liu et al., NeurIPS 2023, arXiv:2305.17118. · SnapKV — Li et al., NeurIPS 2024, arXiv:2404.14469. · FastGen — Ge et al., ICLR 2024, arXiv:2310.01801. · PyramidKV — Cai et al., 2024, arXiv:2406.02069. · Ada-KV — Feng et al., 2024, arXiv:2407.11550. · KIVI — Liu et al., ICML 2024, arXiv:2402.02750. · KVQuant — Hooper et al., NeurIPS 2024, arXiv:2401.18079. · RotateKV — 2025, arXiv:2501.16383. · PagedAttention — Kwon et al., SOSP 2023, arXiv:2309.06180. · RadixAttention/SGLang — Zheng et al., NeurIPS 2024, arXiv:2312.07104. · LMCache — 2025, arXiv:2510.09665. · CacheGen — SIGCOMM 2024, arXiv:2310.07240. · Mooncake — 2024, arXiv:2407.00079. · "Pitfalls of KV Cache Compression," 2025, arXiv:2510.00231.

**SSM / recurrent.** Mamba-2 ("Transformers are SSMs") — Dao & Gu, ICML 2024, arXiv:2405.21060. · RWKV-7 "Goose" — 2025, arXiv:2503.14456. · SSM/Hybrid long-ctx characterization — 2025, arXiv:2507.12442. · "Don't Pay Attention" (Transformer/Mamba/RWKV-7 needle head-to-head) — 2025, arXiv:2506.11305.

**Agent memory.** MemGPT — Packer et al., 2023, arXiv:2310.08560. · Mem0 — Chhikara et al., 2025, arXiv:2504.19413 (LOCOMO: 66.88% vs 52.90%; >90% tokens; p95 1.44s vs 17.12s). · Generative Agents — Park et al., 2023, arXiv:2304.03442 (score = recency+importance+relevance, all α=1, decay 0.995, importance 1–10, reflection>150). · CoALA — Sumers et al., 2023, arXiv:2309.02427 (working/episodic/semantic/procedural). · Reflexion — Shinn et al., NeurIPS 2023, arXiv:2303.11366 (HumanEval 91%). · A-MEM — Xu et al., NeurIPS 2025, arXiv:2502.12110. · Letta sleep-time compute — 2025, arXiv:2504.13171.

**Context engineering / failure modes.** Lost in the Middle — Liu et al., TACL 2023, arXiv:2307.03172. · Context Rot — Hong et al. (Chroma), 2025, research.trychroma.com/context-rot. · Anthropic, "Effective Context Engineering for AI Agents," 2025, anthropic.com/engineering/effective-context-engineering-for-ai-agents. · Retrieval meets Long Context — Xu et al., 2023, arXiv:2310.03025. · OP-RAG — Yu et al., 2024, arXiv:2409.01666 (∞Bench EN.QA: 34.26 → 47.25 F1 at ~60% fewer tokens). · Long-Context LLMs Meet RAG (hard negatives) — Jin et al., ICLR 2025, arXiv:2410.05983. · Recursive summarization — Wang et al., 2023, arXiv:2308.15022.

**In-tree (HIDE's own ground truth).** `crates/hawking-serve/src/system_kv_bank.rs` (KV bank, greedy-lossless). · `crates/hawking-core/src/stateful/working_set.rs` (`KvEvictionPolicy`: Streaming/H2O/SnapKV/Lossless). · `crates/hawking-core/src/stateful/prefix_cache.rs` + `cache/prefill_disk.rs` (RAM+disk KV tiers, `DSPRFKV2`). · `crates/hawking-core/src/model/rwkv7.rs`, `model/mamba2.rs` (SSM, O(1) state ~6 MiB). · `crates/hawking-core/src/engine.rs` (prefill/copy-KV/embed seams). · RWKV-7 measured flat ~118→119 tps to 8k vs Qwen ~40→8.6 (in-tree campaign).
