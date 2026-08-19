I'll pull official model cards, `config.json` files, and papers for all four patients in parallel and only keep numbers those sources actually state.Gemma’s HF files are gated, so I’m pulling the technical report, official docs, and tokenizer/config from public sources next.Several architecture numbers are still missing (Gemma 1B dims, Qwen router, commits, official quants). Fetching those next.# ODYSSEY-I static architecture fact-sheets

All numbers below are from official cards, `config.json`, papers, or official repos. Unverified fields are **UNKNOWN**. Gemma-3-1b-it Hub files are gated; 1B dims are taken from Google’s official architecture dump plus Hub API metadata (revision/param count).

---

## O000  `google/gemma-3-1b-it`

### IDENTITY

| field | value | source |
|---|---|---|
| repo | `google/gemma-3-1b-it` | [HF card](https://huggingface.co/google/gemma-3-1b-it) |
| latest revision | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` (2025-04-04) | [HF API](https://huggingface.co/api/models/google/gemma-3-1b-it) |
| tokenizer | SentencePiece, Gemini-2.0 vocab; files `tokenizer.model` + `tokenizer.json`; chat tokens `<bos>`, `<start_of_turn>`, `<end_of_turn>` | [paper](https://arxiv.org/html/2503.19786), [HF siblings](https://huggingface.co/api/models/google/gemma-3-1b-it) |
| vocab | **262144** (paper table also says “256k entries”) | [Google blog 1B dump](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/), [paper Table 1](https://arxiv.org/html/2503.19786) |
| source precision | **bf16** | [HF card “Tensor type BF16”](https://huggingface.co/google/gemma-3-1b-it), [HF API safetensors](https://huggingface.co/api/models/google/gemma-3-1b-it) |
| license | **Gemma** (not Apache) | [HF `license:gemma`](https://huggingface.co/api/models/google/gemma-3-1b-it), [terms](https://ai.google.dev/gemma/terms) |
| params (total) | **999,885,952** (~1.00B). Paper split: 302M embed + 698M non-embed | [HF API](https://huggingface.co/api/models/google/gemma-3-1b-it), [paper Table 1](https://arxiv.org/html/2503.19786) |
| params (active) | n/a (dense) | — |

### ARCHITECTURE

| field | value | source |
|---|---|---|
| class | **dense** decoder-only, **text-only** (`Gemma3ForCausalLM` / `gemma3_text`). **No vision tower** | [paper Table 1 Vision Encoder=0](https://arxiv.org/html/2503.19786), [HF docs](https://huggingface.co/docs/transformers/en/model_doc/gemma3) |
| n_layers | **26** | [Google 1B dump](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/) |
| hidden_size | **1152**; intermediate **6912**; head_dim **256** | same |
| attention | **GQA → MQA**: 4 Q / **1** KV; QK-norm (no Gemma-2 softcap) | same + [paper §2](https://arxiv.org/html/2503.19786) |
| hybrid attn | **5 local : 1 global**, first layer local. Pattern period **6** | [paper](https://arxiv.org/html/2503.19786), [blog](https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/) |
| local window | Family paper/blog: **1024**. Released 1B configs: **`sliding_window`: 512** | [paper](https://arxiv.org/html/2503.19786) vs [mlx-community copy of official config](https://huggingface.co/mlx-community/gemma-3-1b-it-4bit/raw/main/config.json) |
| RoPE | global θ **1e6**; local θ **10k**. 1B context **32,768** (not 128k) | [paper](https://arxiv.org/html/2503.19786), [model card](https://ai.google.dev/gemma/docs/core/model_card_3) |
| modality | **text only**. Larger Gemma-3 (4/12/27B) use frozen SigLIP ~417M, 896², 256 tokens/image | [paper Table 1](https://arxiv.org/html/2503.19786) |
| MTP / draft | **none** on Gemma 3 1B | not in paper/config |

### KNOWN QUANT / RUNTIME

| item | fact | source |
|---|---|---|
| official QAT | **Yes.** ~5k QAT steps; official int4 / Q4_0 / SFP8. 1B BF16 2.0 GB → int4 **0.5 GB** | [paper §2.3](https://arxiv.org/html/2503.19786), [QAT blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/) |
| official GGUF | **Yes** — [google Gemma-3 QAT collection](https://huggingface.co/collections/google/gemma-3-qat-67ee61ccacbf2be4195c265b) | same |
| official MLX | **Yes** — [mlx-community Gemma-3 QAT](https://huggingface.co/collections/mlx-community/gemma-3-qat-68002674cd5afc6f9022a0ae); also `gemma.cpp` | same |
| transformers | `>=4.50.0`; 1B via `AutoModelForCausalLM` (not VLM class) | [HF card](https://huggingface.co/google/gemma-3-1b-it), [HF docs](https://huggingface.co/docs/transformers/en/model_doc/gemma3) |
| Apple Silicon gotcha | Hybrid **local/global KV cache** (`cache_implementation: hybrid`). Must not treat all 26 layers as full-seq KV. Use official QAT GGUF/MLX, not naive PTQ. | [QAT blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/), [mlx config](https://huggingface.co/mlx-community/gemma-3-1b-it-4bit/raw/main/config.json) |

### GRAVITY-RELEVANT NOVELTY (vs plain Qwen)

- **5:1 local/global attention + tiny local window (512/1024)** — KV is *not* uniform across depth; Hawking must emit two cache layouts.
- **MQA (1 KV head) + head_dim 256 + QK-norm + dual RoPE bases** — not Qwen GQA/128/single-θ.
- **1B is text-only** while the rest of the family is SigLIP-multimodal — do not attach a vision tower.

### COMPILER HYPOTHESES

| # | hypothesis | cheapest discriminator | expected physical win | Doctor risk |
|---|---|---|---|---|
| H1 | Local layers only need a 512-token ring KV; global layers need full 32k | Measure KV bytes/layer at prefill 8k: 5/6 layers should stay flat after 512 | ~5–6× KV vs all-global 32k | If window is actually 1024, ring is short and quality dies |
| H2 | QK-norm + query_pre_attn_scalar=256 replace 1/√d scale | One-layer logit dump vs `1/√256` | Wrong scale → garbage first tokens | Silent quality cliff, not a crash |
| H3 | Official QAT Q4_0 is the quality floor vs PTQ | 50-prompt PPL: official QAT GGUF vs self-PTQ | paper: **54% less PPL drop** vs naive Q4_0 | Wasted convert time if QAT already exists |

---

## O001  `tiiuae/Falcon-H1-7B-Instruct`

### IDENTITY

| field | value | source |
|---|---|---|
| repo | `tiiuae/Falcon-H1-7B-Instruct` | [HF card](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct) |
| latest revision | `41e72f27effbab80cd45b6e884688452253a3686` (2025-07-31) | [HF API](https://huggingface.co/api/models/tiiuae/Falcon-H1-7B-Instruct) |
| tokenizer | BPE, family **Falcon-H1-131k**; `tokenizer_class=PreTrainedTokenizer`; bos `<\|begin_of_text\|>`, eos `<\|end_of_text\|>` | [paper Table 5](https://arxiv.org/html/2507.22448), [tokenizer_config](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/raw/main/tokenizer_config.json) |
| vocab | config **130049**; paper table **130,048**; tokenizer name **131,048** | [config.json](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/raw/main/config.json), [paper Table 1](https://arxiv.org/html/2507.22448) |
| source precision | **bf16** | [config `torch_dtype`](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/raw/main/config.json), [HF API 7,585,654,880 BF16](https://huggingface.co/api/models/tiiuae/Falcon-H1-7B-Instruct) |
| license | **Falcon-LLM License** (Apache-2.0-based, `license: other`) | [HF card](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct), [terms](https://falconllm.tii.ae/falcon-terms-and-conditions.html) |
| params (total) | paper **7.59B**; Hub tensors **7,585,654,880** | [paper Table 1](https://arxiv.org/html/2507.22448), [HF API](https://huggingface.co/api/models/tiiuae/Falcon-H1-7B-Instruct) |
| params (active) | n/a (dense hybrid, not MoE) | — |

### ARCHITECTURE

| field | value | source |
|---|---|---|
| class | **hybrid-head**: **parallel** Attention + **Mamba-2 SSM**, then MLP (`SA_M`). **Every** layer. `attn_layer_indices: null` | [paper Fig.1 / §2.1](https://arxiv.org/html/2507.22448), [config](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/raw/main/config.json) |
| n_layers | **44** | [config](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/raw/main/config.json), [paper Table 1](https://arxiv.org/html/2507.22448) |
| hidden | **3072**; MLP intermediate **12288** | same |
| attention | **GQA 12 Q / 2 KV**, head_dim **128** | same |
| SSM | Mamba-2: **24** heads, d_head **128**, **d_state 256**, d_ssm **3072**, expand **2**, n_groups **1**, conv **4**, chunk **256**, RMSNorm, SiLU gate | [config](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/raw/main/config.json) |
| RoPE | θ **1e11** (one hundred billion). No scaling object | [config `rope_theta`](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/raw/main/config.json), [paper §2.3.1](https://arxiv.org/html/2507.22448) |
| context | **262,144** | same |
| μP | many non-1 multipliers (`embedding_multiplier`, `ssm_multipliers`, `attention_out_multiplier`, …) | [config](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/raw/main/config.json) |
| modality | **text**; 18 languages | [HF card](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct) |
| MTP | **none** | not in config/paper |
| MoE | **none** | — |

### KNOWN QUANT / RUNTIME

| item | fact | source |
|---|---|---|
| official QAT | **No** claim | — |
| official GGUF | **Yes** — [`tiiuae/Falcon-H1-7B-Instruct-GGUF`](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct-GGUF) (Q2–Q8, BF16) | [card](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct-GGUF) |
| official MLX | **Yes, native** `mlx-lm` | [official deploy docs](https://tiiuae.github.io/Falcon-H1/deployment/) |
| transformers | install **from source**; `torch.bfloat16` not fp16 | same |
| vLLM | card `>=0.9.0`; **cap `--max-model-len`** (default 262144 blows KV) | [HF card](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct), [deploy](https://tiiuae.github.io/Falcon-H1/deployment/) |
| llama.cpp | **native**; official GGUFs | same |
| Apple Silicon | Official M4 Max llama.cpp demo. **temp 0.1** recommended (higher “may largely drop”) | [deploy](https://tiiuae.github.io/Falcon-H1/deployment/) |
| extra | optional mamba-ssm fork for a stochastic CUDA bug — N/A on Metal | [deploy](https://tiiuae.github.io/Falcon-H1/deployment/) |

### GRAVITY-RELEVANT NOVELTY (vs plain Qwen)

- **Parallel hybrid mixer** (SSM ∥ attn, concat, then MLP) in **all 44 layers** — not sequential interleave, not MoE.
- **d_state=256 Mamba-2 recurrence** + document-boundary state reset — new state tensor, not a KV cache.
- **RoPE θ=1e11** + μP scale vector — positional/init assumptions from Qwen (θ=1e4/1e6, unit scales) are wrong.

### COMPILER HYPOTHESES

| # | hypothesis | cheapest discriminator | expected physical win | Doctor risk |
|---|---|---|---|---|
| H1 | Prefill throughput beats a 7B transformer once seq ≳ 8–16k (SSM linear) | Same-box time-to-first @ 2k vs 32k vs Qwen3-8B | paper: up to **4× prefill / 8× decode** at long ctx (34B vs Qwen2.5-32B; 7B unstated) | Short-ctx may *lose* to mature attn kernels |
| H2 | SSM state (24×128×256) dominates KV (2×128) at long ctx | Byte-count state vs KV at 4k and 64k | KV almost flat; state constant per layer | Wrong d_state packing → silent numerical drift |
| H3 | Sampling temp >0.1 is a quality bug, not a compile bug | 20-prompt IFEval-style at 0.1 vs 0.7 | official: higher temp “may largely drop” | False “compiler regresses quality” |

---

## O005  `Qwen/Qwen3-30B-A3B`

### IDENTITY

| field | value | source |
|---|---|---|
| repo | `Qwen/Qwen3-30B-A3B` | [HF card](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| latest revision | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` (2025-07-26) | [HF API](https://huggingface.co/api/models/Qwen/Qwen3-30B-A3B) |
| tokenizer | **Qwen2Tokenizer** (BBPE); `merges.txt`+`vocab.json`+`tokenizer.json` | [tokenizer_config](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/tokenizer_config.json), [paper](https://arxiv.org/html/2505.09388) |
| vocab | config **151936**; paper **151,669** | [config](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json), [paper §2](https://arxiv.org/html/2505.09388) |
| source precision | **bf16** | [config](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json), [HF 30,532,122,624 BF16](https://huggingface.co/api/models/Qwen/Qwen3-30B-A3B) |
| license | **Apache-2.0** | [HF card](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| params (total) | **30.5B** (card); Hub **30,532,122,624**; non-embed **29.9B** | [HF card](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| params (active) | **3.3B / token** | same |

### ARCHITECTURE

| field | value | source |
|---|---|---|
| class | **MoE** `Qwen3MoeForCausalLM` | [config](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json) |
| n_layers | **48**; **all MoE** (`decoder_sparse_step=1`, `mlp_only_layers=[]`) | same |
| hidden | **2048**; dense-MLP intermediate **6144**; expert intermediate **768** | same |
| attention | **GQA 32 Q / 4 KV**, head_dim **128**, **QK-norm**, no attn bias | [config](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json), [modeling](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) |
| MoE | **128 routed**, **top-8**, **0 shared**; **softmax** then top-k; `norm_topk_prob=true`; aux coef **0.001**; global-batch load-balance (paper) | [config](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json), [Qwen3MoeTopKRouter](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py), [paper §2](https://arxiv.org/html/2505.09388) |
| context | native **32,768**; config `max_position_embeddings` **40960** (32k out + 8k prompt reserve); **131,072 via YaRN** (static; enable only for long ctx) | [HF card](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| RoPE | θ **1,000,000**; `rope_scaling: null` as shipped | [config](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json) |
| modality | **text**; 119 langs | [paper](https://arxiv.org/html/2505.09388) |
| MTP | **none** | not in config |
| extra | unified **thinking / non-thinking** via chat template (`enable_thinking`, `/think`, `/no_think`) | [HF card](https://huggingface.co/Qwen/Qwen3-30B-A3B) |

### KNOWN QUANT / RUNTIME

| item | fact | source |
|---|---|---|
| official QAT | **No** | — |
| official GGUF | **Yes** — [`Qwen/Qwen3-30B-A3B-GGUF`](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF) | official org repo |
| official MLX | **Supported** (card lists MLX-LM); community MLX conversions | [HF card](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| transformers | **`>=4.51.0`** or `KeyError: 'qwen3_moe'` | [HF card](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| vLLM | `>=0.8.5` (`--enable-reasoning --reasoning-parser deepseek_r1`) | same |
| SGLang | `>=0.4.6.post1` (`--reasoning-parser qwen3`) | same |
| Apple Silicon | 30.5B bf16 ~61 GB; 4-bit ~16 GB. Thinking mode default can emit 32k traces — **cap max_new_tokens**. Static YaRN **hurts short ctx** — leave `rope_scaling` null unless >32k | [HF card YaRN warning](https://huggingface.co/Qwen/Qwen3-30B-A3B) |

### GRAVITY-RELEVANT NOVELTY (vs a *dense* Qwen)

This **is** a Qwen transformer — novelty is MoE + dual-mode, not a new mixer:

- **Fine-grained 128-expert / top-8 / no shared expert** + softmax-then-renormalize (`norm_topk_prob`).
- **QK-norm** (new vs Qwen2.5) + 4 KV heads only.
- **Single checkpoint, two decode graphs** (think vs empty `<think></think>`).

### COMPILER HYPOTHESES

| # | hypothesis | cheapest discriminator | expected physical win | Doctor risk |
|---|---|---|---|---|
| H1 | Expert I/O, not matmul, dominates decode (8×768 SwiGLU vs 2048 dense) | nsys/osx Instruments: memcpy vs GEMM share, 1-token decode | grouped-GEMM / packed expert cache | Wrong grouping → 128 serial GEMMs |
| H2 | Softmax+renorm ≠ sigmoid; dropping renormalize shifts mix | Compare layer-0 router weights with/without `norm_topk_prob` vs HF ref | bitwise-close to transformers | Quality drift blamed on quant |
| H3 | `enable_thinking=False` is a **template** change, not a weight change | Same weights, two templates; first tokens `<think>` vs not | huge latency cut on chat | Serving stack that always thinks OOMs 32k |

---

## O010  `zai-org/GLM-4.5-Air`

### IDENTITY

| field | value | source |
|---|---|---|
| repo | `zai-org/GLM-4.5-Air` | [HF card](https://huggingface.co/zai-org/GLM-4.5-Air) |
| latest revision | `a24ceef6ce4f3536971efe9b778bdaa1bab18daa` (2025-08-11) | [HF API](https://huggingface.co/api/models/zai-org/GLM-4.5-Air) |
| tokenizer | `PreTrainedTokenizer`; GLM specials `[gMASK]`, `<sop>`, `<\|user\|>`, `<think>`, `/nothink` | [tokenizer_config](https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/tokenizer_config.json) |
| vocab | **151552** | [config](https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json) |
| source precision | **bf16** (+ 5888 F32 scalars) | [HF API](https://huggingface.co/api/models/zai-org/GLM-4.5-Air) |
| license | **MIT** | [HF card](https://huggingface.co/zai-org/GLM-4.5-Air), [repo](https://github.com/zai-org/GLM-4.5) |
| params (total) | paper/card **106B**; Hub tensors **110,468,824,832** (embeds + MTP counted on Hub, not in paper Table 1) | [paper Table 1](https://arxiv.org/html/2508.06471), [HF API](https://huggingface.co/api/models/zai-org/GLM-4.5-Air) |
| params (active) | **12B** | [paper Table 1](https://arxiv.org/html/2508.06471), [HF card](https://huggingface.co/zai-org/GLM-4.5-Air) |

### ARCHITECTURE

| field | value | source |
|---|---|---|
| class | **MoE + MTP**, `Glm4MoeForCausalLM` | [config](https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json) |
| n_layers | **46** decoder: **1 dense** (`first_k_dense_replace=1`) + **45 MoE**. Plus **1 MTP** layer | [config](https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json), [paper Table 1](https://arxiv.org/html/2508.06471) |
| hidden | **4096**; dense FFN **10944**; expert FFN **1408** | same |
| attention | **GQA 96 Q / 8 KV**, head_dim **128**, **partial RoPE 0.5**, attn **bias=true**, **`use_qk_norm=false`** (Air; full 4.5 has QK-norm) | [config](https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json), [paper Table 1](https://arxiv.org/html/2508.06471) |
| MoE | **128 routed + 1 shared**, **top-8**; **sigmoid** gates; **loss-free** balance (`n_group=1`, `topk_group=1`); `routed_scaling_factor=1.0`; `norm_topk_prob=true` | [paper §2.1](https://arxiv.org/html/2508.06471), [config](https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json) |
| context | **131,072**; RoPE θ **1e6** (raised from 1e4 at 32k mid-train) | [config](https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json), [paper §2.4](https://arxiv.org/html/2508.06471) |
| MTP | **1** next-n predict layer (MoE); speculative decode | [config `num_nextn_predict_layers=1`](https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json), [paper](https://arxiv.org/html/2508.06471) |
| modality | **text**; hybrid think / no-think | [HF card](https://huggingface.co/zai-org/GLM-4.5-Air) |
| note | HF transformers **does not include MTP** in the modeling class; vLLM/SGLang do (`glm4_moe_mtp`) | [HF glm4_moe docs](https://huggingface.co/docs/transformers/en/model_doc/glm4_moe) |

### KNOWN QUANT / RUNTIME

| item | fact | source |
|---|---|---|
| official QAT | **No** | — |
| official low-bit | **Yes, FP8** — [`zai-org/GLM-4.5-Air-FP8`](https://huggingface.co/zai-org/GLM-4.5-Air-FP8) | [official README](https://github.com/zai-org/GLM-4.5) |
| official GGUF / MLX | **No official**. Community GGUF exists (Unsloth etc.) | Hub tree + [README](https://github.com/zai-org/GLM-4.5) |
| transformers | `glm4_moe`, `>=4.54.0` (config). **No MTP** | [config](https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json), [HF docs](https://huggingface.co/docs/transformers/en/model_doc/glm4_moe) |
| vLLM / SGLang | official parsers + MTP/EAGLE (`--speculative-*`) | [README](https://github.com/zai-org/GLM-4.5) |
| Apple Silicon | BF16 **~221 GB**. 4-bit ~55 GB. Needs ≥128 GB unified **and** a working Metal MoE+MTP path — **UNKNOWN if mlx-lm supports this checkpoint**. Do not assume Qwen3-MoE kernels apply (sigmoid + shared + partial RoPE + MTP) | sizes from [HF API](https://huggingface.co/api/models/zai-org/GLM-4.5-Air) |

### GRAVITY-RELEVANT NOVELTY (vs plain Qwen / Qwen3-MoE)

- **Sigmoid + 1 shared expert + loss-free routing** (DeepSeek-style), not Qwen3 softmax / no-shared.
- **Deeper/narrower + 96 heads on 4096 + partial RoPE ½** — attn layout is not `head_dim * n_heads = hidden`.
- **First layer dense, rest MoE, plus a real MTP draft head** — two graphs (main + draft) and a layer-type split.

### COMPILER HYPOTHESES

| # | hypothesis | cheapest discriminator | expected physical win | Doctor risk |
|---|---|---|---|---|
| H1 | MTP draft (1 MoE layer) gives accepted-token speedup on-device | vLLM/SGLang `--speculative-num-draft-tokens 4` vs off; record accept rate | official serving assumes MTP on | transformers path has **no MTP** — false “no draft weights” |
| H2 | Shared expert is a dense GEMM every token; routed 8 are sparse | FLOP counter: 1×1408 always + 8×1408 routed | fuse shared with residual; don’t scatter it | treating shared as 129th routed expert |
| H3 | Only rotate **first 64 of 128** dims (`partial_rotary_factor=0.5`) | Compare RoPE-on-all-128 vs first-64 vs HF | mismatch = immediate PPL blowup | looks like a “RoPE kernel bug” |

---

## NEXT

| patient | first cheapest discriminator once weights land |
|---|---|
| **O000** gemma-3-1b-it | Dump per-layer KV bytes at prefill 2048: confirm 5/6 layers cap at **512** (released) not 1024 (paper). Then load **official QAT Q4_0** GGUF/MLX and match 20-prompt greedy tokens vs bf16. |
| **O001** Falcon-H1-7B-Instruct | `mlx_lm.generate` **or** official GGUF at temp **0.1**, ctx 4096: confirm SSM state shape `(24, 128, 256)` per layer and that attn KV is only 2 heads. |
| **O005** Qwen3-30B-A3B | One decode step through official HF: assert router is **softmax → top-8 → renormalize**, **no shared**, 48/48 MoE. Then same step with `enable_thinking=False` and assert empty `<think>` block. |
| **O010** GLM-4.5-Air | (1) Confirm layer 0 is dense FFN 10944 and layers 1–45 are 128+1 MoE. (2) Confirm RoPE applies to **64/128** dims. (3) Check whether MTP tensors exist and whether the runtime you picked actually runs them. |

---

**UNKNOWN leftover (do not invent):** Gemma-3-1b-it gated raw `config.json` (window 512 vs paper 1024 still a released-vs-paper conflict); Falcon tokenizer vocab 130049 vs 130048 vs 131048; whether mlx-lm has a first-class GLM-4.5-Air path; exact Apple-Silicon tokens/s for any patient (needs weights).
