# Hawking model-family adapter registry

Generated from `hawking-adapters` — do not hand-edit.

**No family is PRODUCTION today.**

| Family | Level | Executes | Serve-registered | Module |
|---|---|---|---|---|
| DeepSeek V2 | SOURCE_HEADER_VALIDATED | true | true | `crates/hawking-core/src/model/deepseek_v2.rs` |
| Gemma 2 | DECLARED | false | false | `packs/hawking-adapters-extra (gemma2)` |
| GLM (gravity glm_moe_dsa) | SMALL_REAL_CHECKPOINT | true | true | `crates/hawking-core/src/model/gravity_engine.rs` |
| Kimi K2.x | SYNTHETIC_PARITY | false | false | `KIMI_K26_ADAPTER_TWIN.json (reference twin; no in-tree serve module)` |
| Llama (dense GGUF + gravity) | SOURCE_HEADER_VALIDATED | true | true | `crates/hawking-core/src/model/llama.rs` |
| MiniMax | DECLARED | false | false | `(none — declared only)` |
| Mistral / Mixtral | SOURCE_HEADER_VALIDATED | true | true | `crates/hawking-core/src/model/llama.rs (+ pack mixtral)` |
| Phi-3 | DECLARED | false | false | `packs/hawking-adapters-extra (phi3)` |
| Qwen (dense + MoE) | SOURCE_HEADER_VALIDATED | true | true | `crates/hawking-core/src/model/qwen_dense.rs` |
| State-space (RWKV7 + Mamba2) | DECLARED | true | true | `crates/hawking-core/src/model/rwkv7.rs` |

## Gaps

### deepseek

- not FULL_PARENT_VALIDATED: no sealed full-size parent receipt in registry evidence
- not PRODUCTION
- DeepSeek V3/V4 MLA+DSA ladder rungs are NOT this family's shipping GGUF deepseek2 path

### gemma

- module not in shipping load_engine
- pack hydrate required to execute
- not PRODUCTION

### glm

- not PRODUCTION
- gravity_glm.rs is another lane's sealed path — not claimed as open production serve
- full parent source safetensors not the parity authority (gravity bytes are)

### kimi

- not serve-registered in load_engine
- no SMALL_REAL_CHECKPOINT sealed receipt for full generate path
- not PRODUCTION

### llama

- no standing PRODUCTION parity receipt
- smoke and gravity_llama_forward skip when weights/artifacts are absent
- REAL_TENSOR_DECODE / SMALL_REAL_CHECKPOINT require committed fixtures or on-disk parents
- executes on CPU only: dispatches_per_forward=0 and device_id=null, so no Metal path is reached
- output is degenerate on both a raw prompt and a correct Llama-3 chat template, while llama.cpp reads the same GGUF at 338 tok/s against our 3.63

### minimax

- no in-tree engine module
- not serve-registered
- not PRODUCTION

### mistral_mixtral

- mixtral MoE not in shipping load_engine
- seed-c ArchAdapter does not execute
- no PRODUCTION receipt
- SMALL_REAL_CHECKPOINT smoke skips when no GGUF is on disk

### phi

- module not in shipping load_engine
- pack hydrate required to execute
- not PRODUCTION

### qwen

- not PRODUCTION: no standing production parity receipt under continuous serve
- large MoE parents (235B/397B) are campaign-side, not this registry's PRODUCTION claim
- integration/parity tests skip or are #[ignore] without on-disk parents

### state_space

- mamba2 not in shipping load_engine
- not PRODUCTION
- family spans RWKV (executes) and Mamba2 (declared pack only)
