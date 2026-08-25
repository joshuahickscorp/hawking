# Odyssey MoE architecture + representation-opportunity map

Steer S042. Read-only archaeology. No weights were loaded or downloaded.
Configs came from the already-cached Hugging Face snapshots. Framework
source is the installed `mlx_lm` / `transformers` under
`~/.grok-vision`. Decoder citations are this worktree
(`crates/hawking-core/src/model/qwen38_hybrid_decode.rs` and friends).
The Qwen3.8 legacy index lives on the parent working tree at
`/Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-18/`
(not in this sparse checkout and not in `HEAD`; the files exist on
disk and were read from there).

Every opportunity is tagged **UNIVERSAL** / **MOE-SPECIFIC** /
**UNRESOLVED**. Closed Qwen3.8 negatives are not re-opened unless the
index or the contract marks them dense-specific.

---

## 0. Candidates and configs

| | Candidate A | Candidate B |
|---|---|---|
| Hub id | `huihui-ai/Huihui-Qwen3-30B-A3B-Thinking-2507-abliterated` | `ArliAI/GLM-4.5-Air-Derestricted` |
| Snapshot | `174f8bf573e9c5536451248d7419618064596edb` | `09fba4859d9d902b9efa14c738775cda9b7fbc5f` |
| `architectures` | `Qwen3MoeForCausalLM` | `Glm4MoeForCausalLM` |
| `model_type` | `qwen3_moe` | `glm4_moe` |
| Marketing | ~3B active / 30B stored | ~12B active / 106B stored |
| Config-derived (this file) | 3.042B active incl. lm_head / 30.532B stored | 12.803B active incl. lm_head / 106.852B stored (no MTP) |

Only `config.json` is present in each snapshot. No safetensors index
and no weights were opened.

---

## 1. Exact topology — Candidate A (`qwen3_moe`)

Sources: cached `config.json`;
`mlx_lm/models/qwen3_moe.py`;
`transformers/models/qwen3_moe/{configuration,modeling}_qwen3_moe.py`;
`mlx_lm/models/switch_layers.py`.

### 1.1 Config (authoritative numbers)

From the cached config (not the transformers defaults):

- `num_hidden_layers = 48`, `hidden_size = 2048`, `intermediate_size = 6144`
- `moe_intermediate_size = 768`, `num_experts = 128`, `num_experts_per_tok = 8`
- `decoder_sparse_step = 1`, `mlp_only_layers = []`
- `norm_topk_prob = true` (overrides transformers default `False` at
  `configuration_qwen3_moe.py:106`)
- `num_attention_heads = 32`, `num_key_value_heads = 4`, `head_dim = 128`
- `attention_bias = false`, `use_sliding_window = false`
- `rms_norm_eps = 1e-6`, `rope_theta = 1e7`, `rope_scaling = null`
- `vocab_size = 151936`, `tie_word_embeddings = false`
- `max_position_embeddings = 262144`
- `hidden_act = silu`

`intermediate_size = 6144` is the **dense-MLP** width. It is unused
on this checkpoint: every layer is sparse (below).

### 1.2 Layer recipe

`Qwen3MoeDecoderLayer.__init__` (`modeling_qwen3_moe.py:305–316`) and
the mlx twin `Qwen3MoeDecoderLayer.__init__` (`qwen3_moe.py:142–159`):

```
if (layer_idx not in mlp_only_layers)
   and num_experts > 0
   and (layer_idx + 1) % decoder_sparse_step == 0:
    mlp = Qwen3MoeSparseMoeBlock
else:
    mlp = Qwen3MoeMLP(intermediate_size)
```

With `mlp_only_layers = []` and `decoder_sparse_step = 1`,
`(layer_idx + 1) % 1 == 0` for every layer. **All 48 layers are
`Qwen3MoeSparseMoeBlock`. There is no dense MLP layer and no shared
expert.**

Forward (`modeling_qwen3_moe.py:319–348`, mlx `qwen3_moe.py:161–171`)
is the standard pre-norm residual sandwich:

1. `input_layernorm` → `self_attn` → add residual
2. `post_attention_layernorm` → `mlp` → add residual

### 1.3 Attention — GQA + QK-norm + full RoPE

`Qwen3MoeAttention` (`modeling_qwen3_moe.py:122–190`) /
mlx `Attention` (`qwen3_moe.py:37–96`):

| Tensor | Shape | Bias |
|---|---|---|
| `self_attn.q_proj.weight` | `[32*128, 2048] = [4096, 2048]` | none |
| `self_attn.k_proj.weight` | `[4*128, 2048] = [512, 2048]` | none |
| `self_attn.v_proj.weight` | `[512, 2048]` | none |
| `self_attn.o_proj.weight` | `[2048, 4096]` | none |
| `self_attn.q_norm.weight` | `[128]` (RMS on head dim) | — |
| `self_attn.k_norm.weight` | `[128]` | — |

- GQA groups = `32/4 = 8`.
- QK-norm is **unconditional** (constructed in `__init__`, applied
  before RoPE). mlx `Attention.__call__` lines 76–81; transformers
  `forward` lines 162–163.
- RoPE is **full `head_dim`** (`nn.RoPE(head_dim, traditional=False,
  base=rope_theta)` at mlx lines 59–63). Not the Qwen3.8 partial
  rotary (`QWEN38_PARTIAL_ROTARY_FACTOR = 0.25`,
  `QWEN38_GQA_ROTARY_DIM = 64` in `qwen38_geometry.rs:32,44`).
- No attention-output sigmoid gate. Qwen3.8 GQA dispatches
  `qwen38_attention_apply_sigmoid_gate`
  (`qwen38_hybrid_decode.rs:3325–3336`). That kernel must **not**
  run on A.
- Decode cache is standard KV (`cache.update_and_fetch` /
  `past_key_values.update`). No DeltaNet, no conv1d, no `A_log`.

This GQA shape is **identical** to the already-typed
`QWEN30_CODER_GQA_TOPOLOGY` (`qwen_moe.rs:59–65`: 32/4/128,
`architecture = "Qwen3MoeForCausalLM"`).
`qwen30_gqa_topology_from_hf_config` (`qwen_moe.rs:155–179`) would
accept this config. `dispatch_qwen30_gqa_attention_component`
(`qwen_moe.rs:254–292`) is the existing Metal component — after
Q/K projection and RoPE, not a full decoder.

### 1.4 Router — softmax, top-8, renormalize

mlx `Qwen3MoeSparseMoeBlock.__call__` (`qwen3_moe.py:123–139`):

```
gates = softmax(Linear(x, [128, 2048]), axis=-1, precise=True)   # [*, 128]
inds  = argpartition(gates, kth=-8)[..., -8:]
scores = take_along_axis(gates, inds)
scores /= sum(scores)                                            # norm_topk_prob
y = SwitchGLU(x, inds)
y = (y * scores[..., None]).sum(axis=-2)
```

transformers `Qwen3MoeTopKRouter.forward` (`modeling_qwen3_moe.py:258–267`)
is the same math: `F.linear` → `softmax` → `topk(8)` → optional
renorm. Weight tensor is `gate.weight` of shape `[128, 2048]`, no
bias, no `e_score_correction_bias`, no group routing, no
`routed_scaling_factor`.

This is the same 128-expert / top-8 topology as
`QWEN30_CODER_ROUTE_TOPOLOGY` (`qwen_moe.rs:30–35`).
`qwen30_route_topology_from_hf_config` (`qwen_moe.rs:134–149`) would
accept this config. The live Metal gate is
`kernels::moe_topk_gate_tcb` / `moe_topk_gate_tcb_ex`
(`kernels/mod.rs:12721–12780`), kernel `moe_topk_gate` in
`shaders/moe.metal:153`. That kernel is **softmax + mask-and-pick
max**, with optional in-kernel top-k renormalization
(`normalize_topk=true` is exactly Qwen3 `norm_topk_prob`). It is
the right router for A. It is the **wrong** router for B.

mlx `Model.quant_predicate` (`qwen3_moe.py:249–255`) forces the
router (`path.endswith("mlp.gate")`) to **8-bit group-64**. Keep
the router in a higher-precision lane than the experts.

### 1.5 Experts — 128 independent SwiGLU, no shared expert

Each expert is a bias-free SwiGLU of width 768:

| Tensor | Shape |
|---|---|
| `mlp.experts.{e}.gate_proj.weight` | `[768, 2048]` |
| `mlp.experts.{e}.up_proj.weight` | `[768, 2048]` |
| `mlp.experts.{e}.down_proj.weight` | `[2048, 768]` |

mlx packs them at load time (`Model.sanitize`, `qwen3_moe.py:232–246`)
into `mlp.switch_mlp.{gate,up,down}_proj.weight` stacked as
`[128, out, in]`. `SwitchGLU.__call__` (`switch_layers.py:176–199`)
does `mx.gather_mm` / `mx.gather_qmm` indexed by the top-8 ids,
then `swiglu(gate, up)` and the down-proj. Tokens are sorted when
`indices.size >= 64` (prefill); a single decode token does not
sort.

transformers `Qwen3MoeExperts` (`modeling_qwen3_moe.py:209–246`)
stores a packed `gate_up_proj[128, 1536, 2048]` +
`down_proj[128, 2048, 768]` and loops only over **hit** experts
(`expert_hit`), `index_add_`-ing the weighted SwiGLU. Same function.

There is no `shared_experts` module. CPU helper
`moe::add_shared_experts` (`moe.rs:115–117`) documents
“Qwen3-MoE has 0”.

### 1.6 Embed / head / norm

- `model.embed_tokens.weight` `[151936, 2048]`
- `lm_head.weight` `[151936, 2048]` (untied; mlx `Model.__call__`
  lines 226–229 uses `lm_head` when `tie_word_embeddings` is false)
- `model.norm.weight` `[2048]`
- per-layer `input_layernorm.weight`, `post_attention_layernorm.weight`

No MTP. No vision tower. No `language_model.` prefix (that prefix
is Qwen3.8-VLM-specific: `qwen38_layer_name` at
`qwen38_geometry.rs:119–121`).

---

## 2. Exact topology — Candidate B (`glm4_moe`)

Sources: cached `config.json`;
`mlx_lm/models/glm4_moe.py`;
`transformers/models/glm4_moe/{configuration,modeling}_glm4_moe.py`.

### 2.1 Config (authoritative numbers)

- `num_hidden_layers = 46`, `hidden_size = 4096`, `intermediate_size = 10944`
- `moe_intermediate_size = 1408`
- `n_routed_experts = 128`, `n_shared_experts = 1`, `num_experts_per_tok = 8`
- `first_k_dense_replace = 1`
- `n_group = 1`, `topk_group = 1`, `routed_scaling_factor = 1.0`
- `norm_topk_prob = true`
- `num_attention_heads = 96`, `num_key_value_heads = 8`, `head_dim = 128`
- `attention_bias = true`, `use_qk_norm = false`
- `partial_rotary_factor = 0.5` (transformers `__post_init__` default
  at `configuration_glm4_moe.py:123`; present in the cached config)
- `rms_norm_eps = 1e-5`, `rope_theta = 1e6`
- `vocab_size = 151552`, `tie_word_embeddings = false`
- `max_position_embeddings = 131072`
- `num_nextn_predict_layers = 1` (MTP; mlx field name
  `num_mtp_layers` / attribute map at config line 87–89)

mlx `ModelArgs` also carries `scoring_func = "sigmoid"` and
`topk_method = "noaux_tc"` (`glm4_moe.py:45–46`). `MoEGate.__init__`
asserts `topk_method == "noaux_tc"` (line 178).

### 2.2 Layer recipe — one dense layer, then MoE

`Glm4MoeDecoderLayer.__init__` (`modeling_glm4_moe.py:412–415`) and
mlx `DecoderLayer.__init__` (`glm4_moe.py:232–239`):

```
mlp = MoE(config) if layer_idx >= first_k_dense_replace else MLP(config)
```

- **Layer 0**: dense `Glm4MoeMLP` / mlx `MLP` with
  `intermediate_size = 10944`.
- **Layers 1–45** (45 layers): `Glm4MoeMoE` / mlx `MoE`.
- Layer 46 is the MTP block. mlx `Model.sanitize` (`glm4_moe.py:317–337`)
  drops `model.layers.{num_hidden_layers}.*`. transformers
  `_keys_to_ignore_on_load_unexpected` includes
  `r"model\.layers\.46.*"` (`modeling_glm4_moe.py:470`).
  **MTP weights are not part of the 46-layer decode graph.** They
  are a speculative-decoding add-on (`Glm4MoeConfig` docstring
  lines 36–39). G26 named MTP as the highest-leverage unused TPS
  path; it is available on B and absent on A.

### 2.3 Attention — wide GQA, partial RoPE, bias, no QK-norm

`Glm4MoeAttention` (`modeling_glm4_moe.py:185–260`) /
mlx `Attention` (`glm4_moe.py:49–108`):

| Tensor | Shape | Bias |
|---|---|---|
| `self_attn.q_proj.weight` | `[96*128, 4096] = [12288, 4096]` | **yes** `[12288]` |
| `self_attn.k_proj.weight` | `[8*128, 4096] = [1024, 4096]` | **yes** `[1024]` |
| `self_attn.v_proj.weight` | `[1024, 4096]` | **yes** `[1024]` |
| `self_attn.o_proj.weight` | `[4096, 12288]` | no |
| `q_norm` / `k_norm` | **absent** (`use_qk_norm = false`) | — |

- GQA groups = `96/8 = 12`.
- Partial RoPE: mlx `nn.RoPE(int(head_dim * partial_rotary_factor))`
  (`glm4_moe.py:70–74`) so rotary dim = `64`. transformers
  `apply_rotary_pos_emb` (`modeling_glm4_moe.py:170–181`) splits
  `q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]` and
  concatenates. Qwen3.8 GQA also uses a 64-dim rotary, but on a
  **256**-dim head (`QWEN38_GQA_HEAD_DIM` /
  `QWEN38_GQA_ROTARY_DIM`). The existing
  `qwen38_gqa_qk_norm_rope_cache_*` kernel is the wrong geometry
  (24 heads / 4 KV / 256 dim / always QK-norm / sigmoid gate).
- No QK-norm on this checkpoint. mlx constructs `q_norm`/`k_norm`
  only when `use_qk_norm` is true (lines 65–68).
- No attention sigmoid gate.

### 2.4 Router — sigmoid + correction bias + (degenerate) grouped top-k

This is **not** the Qwen3 softmax gate.

mlx `group_expert_select` (`glm4_moe.py:132–163`), compiled:

1. `scores = sigmoid(gates.float32)`  — stored as `orig_scores`
2. `scores = scores + e_score_correction_bias`  — selection only
3. If `n_group > 1`: reshape to `(n_group, n_experts/n_group)`,
   score each group as the sum of its top-2, keep `topk_group`
   groups, zero the rest.
4. `argpartition` top-`k` on the **biased** scores; gather
   **unbiased** `orig_scores` as weights.
5. If `top_k > 1 and norm_topk_prob`: renormalize.
6. Multiply by `routed_scaling_factor`.

`MoEGate.__call__` (`glm4_moe.py:180–189`) is
`group_expert_select(x @ weight.T, e_score_correction_bias, ...)`.

transformers `Glm4MoeTopkRouter.forward` (`modeling_glm4_moe.py:292–317`)
is the same sigmoid + bias + group-mask + `topk` + gather-unbiased +
renorm + `routed_scaling_factor`. `e_score_correction_bias` is an
`nn.Buffer` kept in fp32 (`_keep_in_fp32_modules_strict` at line 469;
mlx `cast_predicate` skips it at `glm4_moe.py:399–403`).

On **this** checkpoint `n_group = 1` and `topk_group = 1`, so the
group mask is a no-op (mlx even skips the branch when
`n_group > 1` is false). The live difference versus A is still
real: **sigmoid, not softmax; bias used for ranking only; weights
come from unbiased sigmoid; then `* routed_scaling_factor`.**
`moe_topk_gate` cannot run B. A new
`glm4_moe_sigmoid_bias_topk` (or a mode on the existing kernel)
is required.

Router tensors per MoE layer:

| Tensor | Shape |
|---|---|
| `mlp.gate.weight` | `[128, 4096]` |
| `mlp.gate.e_score_correction_bias` | `[128]` fp32 |

### 2.5 Experts — 128 routed + 1 shared, both width 1408

mlx `MoE.__init__` (`glm4_moe.py:192–208`):

- `switch_mlp = SwitchGLU(4096, 1408, 128)` — routed experts
- `shared_experts = MLP(intermediate_size = 1408 * n_shared_experts)`
  = one SwiGLU of width 1408

`MoE.__call__` (`glm4_moe.py:212–225`):

```
inds, scores = gate(x)
y = (switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
y = y + shared_experts(x)          # always-on, unweighted
```

transformers `Glm4MoeMoE.forward` (`modeling_glm4_moe.py:395–402`)
is the same: routed `experts(...)` plus `shared_experts(residuals)`.

| Tensor | Shape | When touched |
|---|---|---|
| `mlp.experts.{e}.gate_proj.weight` | `[1408, 4096]` | if routed |
| `mlp.experts.{e}.up_proj.weight` | `[1408, 4096]` | if routed |
| `mlp.experts.{e}.down_proj.weight` | `[4096, 1408]` | if routed |
| `mlp.shared_experts.gate_proj.weight` | `[1408, 4096]` | **every** MoE token |
| `mlp.shared_experts.up_proj.weight` | `[1408, 4096]` | every MoE token |
| `mlp.shared_experts.down_proj.weight` | `[4096, 1408]` | every MoE token |

Layer-0 dense MLP uses the **wide** intermediate:

| Tensor | Shape |
|---|---|
| `mlp.gate_proj.weight` | `[10944, 4096]` |
| `mlp.up_proj.weight` | `[10944, 4096]` |
| `mlp.down_proj.weight` | `[4096, 10944]` |

mlx `sanitize` (`glm4_moe.py:321–330`) stacks per-expert
`{w1,w2,w3}` / `{gate,down,up}_proj` into `mlp.switch_mlp.*`.

### 2.6 Embed / head / norm / MTP

- `model.embed_tokens.weight` `[151552, 4096]`
- `lm_head.weight` `[151552, 4096]` (untied)
- `model.norm.weight` `[4096]`
- MTP at `model.layers.46.*` is stripped / ignored (see 2.2)

---

## 3. ACTIVE vs STORED parameter split

Counts are **parameter elements** derived from the configs and the
module constructors above. They do not include optimizer state,
KV cache, or MTP. BF16 bytes = elements × 2.

### 3.1 Candidate A — every token, every layer

**Stored (always resident if the whole model is loaded):**

| Block | Formula | Elements |
|---|---|---|
| Attn + 2×RMS + QK-norm, ×48 | `48 * ((4096+512+512+4096)*2048 + 128+128+2048+2048)` | 906,178,560 |
| Router, ×48 | `48 * 128 * 2048` | 12,582,912 |
| 128 experts × 3 GEMVs, ×48 | `48 * 128 * 3 * 2048 * 768` | 28,991,029,248 |
| Embed + lm_head + final RMS | `2*151936*2048 + 2048` | 622,331,904 |
| **Total stored** | | **30,532,122,624** |

Matches the 30B label (30.53B).

**Active per decode token** (tensors whose values are read to
produce the next logit):

| Touched | Elements | Notes |
|---|---|---|
| Embed **row** | 2,048 | lookup, not the full table |
| Per layer, always: Q/K/V/O + QK-norm + 2×RMS | 18,878,720 | 48 × this |
| Per layer, always: `mlp.gate.weight` | 262,144 | 48 × this |
| Per layer, **8 of 128** expert triplets | 8 × 4,718,592 = 37,748,736 | 48 × this |
| Final RMS + **full** lm_head | 311,166,976 | vocab GEMV |
| **Active incl. lm_head** | **3,041,869,824** | ~3.04B |
| Active body (no embed row, no lm_head) | 2,730,700,800 | ~2.73B |

Untouched per token: **120 / 128 experts × 48 layers** =
5,760 expert triplets = 27.179B parameters (89.0% of stored).
That is the entire representation opening.

At a coherent ~3.3 bpw (legacy `PATIENT_IDENTITY_VECTOR` /
`STRATEGIC_FINDING_100TPS.json`), **active** bytes including
lm_head are `3.041869824e9 * 3.3 / 8 ≈ 1.25 GB`. Stored at the
same bpw is `30.532122624e9 * 3.3 / 8 ≈ 12.6 GB`.

### 3.2 Candidate B — layer 0 dense, then 8-of-128 + shared

**Stored (no MTP):**

| Block | Elements |
|---|---|
| Attn + biases + 2×RMS, ×46 | 5,017,423,872 |
| Layer-0 dense SwiGLU 10944 | 134,479,872 |
| Router + bias, ×45 | 23,598,720 |
| Shared expert, ×45 | 778,567,680 |
| 128 routed experts, ×45 | 99,656,663,040 |
| Embed + lm_head + final RMS | 1,241,518,080 |
| **Total stored** | **106,852,251,264** |

Matches the 106B label (106.85B).

**Active per decode token:**

| Touched | Elements |
|---|---|
| Embed row | 4,096 |
| Layer 0 attn+norm + dense 10944 MLP | 243,554,304 |
| Layers 1–45 attn+norm | 45 × 109,074,432 |
| Layers 1–45 router + bias | 45 × 524,416 |
| Layers 1–45 shared expert | 45 × 17,301,504 |
| Layers 1–45, 8 routed experts | 45 × 138,412,032 |
| Final RMS + lm_head | 620,761,088 |
| **Active incl. lm_head** | **12,803,376,768 (~12.80B)** |
| Active body (no embed row, no lm_head) | 12,182,611,584 (~12.18B) |

Untouched per token: **120 / 128 routed experts × 45 layers** =
5,400 expert triplets = 93.428B parameters (87.4% of stored).
The shared expert and layer-0 dense MLP are **never** cold.

At ~3.3 bpw: active ≈ `12.803376768e9 * 3.3 / 8 ≈ 5.28 GB`
(above the ~4.7 GB / 100 TPS bandwidth line in
`STRATEGIC_FINDING_100TPS.json`); stored ≈ 44.1 GB.

### 3.3 What is *not* in the active stream

| Tensor class | A | B |
|---|---|---|
| Cold routed experts (120/128) | unused | unused |
| Shared expert | n/a | **always on** |
| Layer-0 wide MLP | n/a | **always on** |
| Router | always on (tiny) | always on (tiny) |
| Attention | always on | always on |
| Embed table minus 1 row | unused | unused |
| MTP (`layers.46`) | n/a | not in greedy decode |
| `e_score_correction_bias` | n/a | always on, 128 fp32 |

---

## 4. What `qwen38_hybrid_decode` would need to run either natively

The decoder is Qwen3.8-hybrid-specific. The file header says so
(`qwen38_hybrid_decode.rs:1–6`: “Dense SwiGLU suffix”). The
schedule comment says the same
(`qwen38_64_layer_execution_schedule.rs:1–4`: “Dense SwiGLU
suffix replaces Q80's 14-dispatch MoE suffix”). This is not a
config tweak. It is a new token graph that can **reuse GEMV
kinds and some Qwen30/Q80 MoE primitives**.

### 4.1 Hard refuses already in tree

| Site | What it does | A | B |
|---|---|---|---|
| `qwen38_accept_config` `qwen38_geometry.rs:141–173` | Accepts only `qwen3_5` / `qwen3_5_text`. Explicitly refuses `qwen3_moe`. Refuses any config containing `num_experts` or `moe_intermediate_size` (“qwen38 is dense”). Locks 64 / 5120 / 17408 / 248320 / 24 / 4 / 256 / interval-4. | refuse | refuse |
| `qwen38_mixer_kind` `qwen38_geometry.rs:86–97` | GQA iff `(layer+1)%4==0`, else DeltaNet. | wrong (A is all-GQA) | wrong (B is all-GQA) |
| `qwen38_assert_schedule_intact` + `QWEN38_DENSE_MLP_SUFFIX_KERNELS` `qwen38_64_layer_execution_schedule.rs:42–49,94` | Locks a 6-dispatch dense SwiGLU suffix (`mlp.gate/up/down_proj` only). | refuse | refuse |
| `qwen38_layer_name` `qwen38_geometry.rs:119–121` | `language_model.model.layers.{L}.{suffix}` | wrong prefix | wrong prefix |
| `classify_qwen38_mixed_payload` `qwen38_hybrid_decode.rs:160–200` + test `unknown_codec_5_still_refuses` `:4406–4414` | Codecs 0–4 only. Codec 5 (“HGRAVF01”) is **refused on this worktree**. | cannot bind affine2 | same |
| `assert_mixed_mlp_native_kinds` `:250–276` | Requires every layer to have exactly `mlp.{gate,up,down}_proj.weight` as Binary/Uniform, Residual/Uniform, Hgravs/Uniform. | no such tensors | only layer 0 |
| `QwenMoE::load` `qwen_moe.rs:298–303` | `Error::Unimplemented("qwen-moe: lands in Phase 3")`. | no Engine | no Engine |

### 4.2 Token-graph functions that cannot be reused as-is

`Qwen38HybridDecodeSession::encode_layers`
(`qwen38_hybrid_decode.rs:3352–3360`) is the whole story:

```
for layer in 0..QWEN38_LAYERS {          // 64, not 48 or 46
    match qwen38_mixer_kind(layer)? {    // DeltaNet / GQA
        DeltaNet => encode_deltanet(...)
        Gqa      => encode_gqa(...)
    }
    encode_dense_mlp(..., first_residual) // always-on wide SwiGLU
}
```

| Function | Lines | Why it does not run A or B |
|---|---|---|
| `encode_deltanet` / `encode_deltanet_mixed` | 3092–3216, 3493+ | A/B have no `linear_attn.*`, no conv state, no recurrent state. 48/64 Qwen3.8 layers take this path. |
| `encode_gqa` / `encode_gqa_mixed` | 3218–3350 | Hardcoded `QWEN38_GQA_{HEADS,KV_HEADS,HEAD_DIM,ROTARY_DIM}` = 24/4/256/64; always binds `q_norm`/`k_norm`; always dispatches `qwen38_attention_apply_sigmoid_gate`. A is 32/4/128/128 full-RoPE + QK-norm, no gate. B is 96/8/128/64 partial-RoPE, no QK-norm, **q/k/v bias**, no gate. |
| `encode_dense_mlp` / `encode_dense_mlp_mixed` | 3040–3090, 3441–3491 | One `mlp.gate_proj` + `mlp.up_proj` + `swiglu_f32` + `mlp.down_proj` at `QWEN38_INTERMEDIATE = 17408`. A has 128×768 experts. B has that pattern only on layer 0, at 10944. |
| `encode_mlp_matvecs_only` | 2119–2142 | Same three names. |
| `encode_named_matvec` / `encode_mixed_matvec` | 1541–1576 | Fine as a **primitive** (name → `MixedGpuWeight` → `dispatch_{binary,residual,hgravs,uniform}`). Missing: indexed/batched expert GEMV and codec-5 affine. |
| `encode_mixer` | 2051–2056 | Hybrid mixer only. |
| Workspace (`qwen38_workspace_bytes`, ~688–760) | sized for hidden 5120 / intermediate 17408 / 48 DeltaNet slots / 16 GQA slots | A needs hidden 2048 / mid 768×8. B needs hidden 4096 / mid 1408×8 + shared 1408 + dense 10944, and 46 GQA slots of 8×128. |

`MixedMlpNativeKind` (`:208–213`) is `{Binary, Residual, Hgravs, Uniform}`.
That set is the packed-GEMV vocabulary and is reusable. What is
missing is a **role** that is “expert *e* of layer *L*”, a router
lane, and (on this worktree) `AffineScaleBias`.

### 4.3 New kinds / dispatch / gather — named against real functions

Do **not** extend `qwen38_accept_config` to lie. Add a sibling
geometry + session (the Q80 pattern:
`qwen80_mixed_hybrid_decode.rs` + `qwen80_device_expert_table.rs`).

#### Shared (both candidates)

1. **New geometry module** (mirror `qwen38_geometry.rs`) with the
   config numbers in §1.1 / §2.1. New `accept_config` that
   *requires* `num_experts` / `moe_intermediate_size` (the inverse
   of `qwen38_accept_config:169–172`).
2. **New layer schedule** that replaces
   `QWEN38_DENSE_MLP_SUFFIX_KERNELS` with a router + expert-wave
   suffix. `qwen38_assert_schedule_intact` must not be called.
3. **Catalog names** `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight`
   (the scheme `ensure_expert` already uses at
   `qwen80_mixed_hybrid_decode.rs:1877–1879`), plus
   `model.layers.{L}.mlp.gate.weight`. Drop the
   `language_model.` prefix.
4. **Reuse GEMV kinds as-is**: `classify_qwen38_mixed_payload`
   codecs 0–4, `MixedGpuKind`, `dispatch_binary` /
   `dispatch_residual` / `dispatch_hgravs` / `dispatch_uniform`
   (`qwen38_hybrid_decode.rs:1658–1840`). These are shape-generic.
   `qwen38_hgravu01_geo_tpr64_launch` (`:564–581`) only requires
   `group_size==64` and `cols % 64 == 0`. A’s 2048-col GEMVs
   qualify; B’s 4096-col GEMVs qualify; A’s 768-row down-proj
   is a **small-M** launch (occupancy question, not a refuse).
5. **Expert gather / combine**, pick one of the already-written
   stacks rather than inventing a fourth:
   - Device table: `build_qwen80_device_expert_table` /
     `dispatch_qwen80_device_expert_table_tcb`
     (`qwen80_device_expert_table.rs:347,1032`) — today locked to
     `QWEN80_EXPERTS=512`, `top_k=10`,
     `QWEN80_MOE_INTERMEDIATE`, `QWEN80_HIDDEN`. Needs a
     128-expert / top-8 / (768 or 1408) specialization.
   - Host-bind wave: `ensure_expert` + `routed_wave_fused`
     (`qwen80_mixed_hybrid_decode.rs:1865–2078`) using
     `qwen80_expert_table_silu_mul` and
     `qwen80_expert_table_weighted_sum`.
   - GGUF-indexed Q4: `MetalContext::moe_block_batched_indexed_metal`
     (`kernels/mod.rs:7912`) — `moe_batched_gemm_q4_indexed_v2`
     + `moe_batched_gemm_q8_0_indexed`, optional shared-expert
     offsets. Closest to B’s “routed + shared” block, but Q4_K /
     Q8_0, not HQ38M20 mixed codecs.
   - CPU oracle only: `moe::moe_forward_token` +
     `moe::expert_ffn` + `moe::add_shared_experts`
     (`moe.rs:49–140`).
6. **Router GEMV** is just another `encode_named_matvec` of
   `mlp.gate.weight` (tiny: 128×2048 or 128×4096). Keep it
   high-precision (mlx 8-bit g64 predicate).
7. **All-GQA mixer**: new `encode_gqa_*` parameterized by
   `(n_heads, n_kv, head_dim, rotary_dim, qk_norm, attn_bias,
   sigmoid_gate=false)`. Reuse `mha_decode_f32_tcb` /
   `kernels::mha_decode_f32_metal` (already used by
   `dispatch_qwen30_gqa_attention_component`). Do **not**
   dispatch `qwen38_attention_apply_sigmoid_gate`. Do **not**
   call `encode_deltanet`.
8. **RMS + residual + argmax + embed lookup**
   (`encode_rmsnorm`, `qwen_next_add_residual_tcb`,
   `sample_argmax_f32_tcb`, `encode_embed_mixed`) are reusable
   once hidden/vocab constants change.

#### Candidate A only (closer)

9. Router: `moe_topk_gate_tcb_ex(..., n_experts=128, top_k=8,
   normalize_topk=true)` (`kernels/mod.rs:12742`). Or
   `dispatch_qwen_router_component` (`qwen_moe.rs:208`) after
   the router GEMV. `qwen30_route_topology_from_hf_config` +
   `qwen30_gqa_topology_from_hf_config` **already accept this
   config**. The missing piece is the rest of the token graph,
   not the topology types.
10. No shared expert. No dense layer. No attention bias.
11. QK-norm + full-head RoPE: new kernel or a generalized
    `qwen38_gqa_qk_norm_rope_cache_*` with `n_heads=32`,
    `n_kv=4`, `head_dim=128`, `rotary_dim=128`.

#### Candidate B only (farther)

12. **New router kernel**: sigmoid → add
    `e_score_correction_bias` → top-8 on biased scores → gather
    unbiased sigmoid → renorm → `* routed_scaling_factor`.
    `group_expert_select` (`glm4_moe.py:132–163`) /
    `Glm4MoeTopkRouter.forward` (`modeling_glm4_moe.py:292–317`).
    `n_group=1` so the group-mask branch can wait. Softmax
    `moe_topk_gate` is incorrect even as an approximation
    (ranking and weights both change).
13. **Shared expert** every MoE layer: one extra SwiGLU of width
    1408, added (not routed-weighted). CPU already has
    `add_shared_experts`. Device path: the shared-offset arm of
    `moe_block_batched_indexed_metal` (`kernels/mod.rs:8000–8142`)
    or a dedicated `encode_dense_mlp` at width 1408.
14. **Layer-0 dense MLP** at 10944: this *is*
    `encode_dense_mlp` with different constants. Reuse the
    function, not the Qwen3.8 schedule lock.
15. **Attention bias**: `q_proj`/`k_proj`/`v_proj` GEMVs need a
    bias-add that Qwen3.8 GQA never does (`attention_bias=false`
    there). New epilogue or fused GEMV+bias.
16. **Partial RoPE 0.5 on head_dim 128** (rotary 64, pass-through
    64), **no** QK-norm. New rope kernel; do not call
    `qwen38_gqa_qk_norm_rope_cache_*`.
17. **MTP** (`num_nextn_predict_layers=1`): out of scope for a
    greedy native decoder; it is the G26 TPS lever if B is
    chosen (`G26_RUNTIME_DIAGNOSIS.json` “MTP multi-token
    (2-3x, untapped)”).
18. Hidden 4096 / 96 heads makes every always-on GEMV **4×** A’s
    attention bytes and **2×** A’s width. Workspace and
    residency are a different machine than A.

### 4.4 Honest native-readiness

| | A | B |
|---|---|---|
| Can this worktree’s `Qwen38HybridDecodeSession` load it? | **No.** Admission refuse + dense MLP + hybrid mixer + wrong names. | **No.** Same, plus sigmoid router, shared expert, attn bias, 96-head GQA. |
| Typed topology already in tree? | **Yes.** `QWEN30_CODER_{ROUTE,GQA}_TOPOLOGY` is this shape. | **No.** Nothing named `glm4`. |
| Router kernel exists? | **Yes.** `moe_topk_gate` + `normalize_topk`. | **No.** Need sigmoid+bias. |
| Expert wave exists? | Pattern yes (Q80 table / mixed `ensure_expert`); geometry no (512×10×512 vs 128×8×768). | Pattern yes; also need always-on shared + layer-0 dense. |
| Engine? | `QwenMoE` unimplemented. | none |
| HGRAVF01 (affine2) in *this* decoder? | **No.** Codec 5 refused (`:4406`). Receipts document it as UNIVERSAL kernel work (G23/G24/G25). Parent working tree has `MixedCatalogLane::Affine` / `dispatch_affine`; this worktree does not. | same |

A is a **new decoder that can steal Qwen30 component contracts
and Q80 expert-table machinery**. B is that plus a new router,
shared-expert residual, attention bias, and a 96-head GQA
kernel. Neither is a flag on `qwen38_hybrid_decode.rs`.

---

## 5. Representation opportunities (conservative tags)

Index tags are quoted from
`receipts/ascent-2026-08-18/QWEN38_LEGACY_INDEX.json`. Where the
index says UNIVERSAL and the contract says the finding is
dense-specific, both are recorded and the opportunity is
**UNRESOLVED** rather than re-opened as if the negative never
happened.

### 5.1 Newly viable on MoE

#### Per-expert codec — **MOE-SPECIFIC**

Qwen3.8 locked one kind per MLP *role* across all 64 layers
(`assert_mixed_mlp_native_kinds`, `:250–276`: gate Binary, up
Residual, down Hgravs, or Uniform on any). MoE has 128
independent triplets per layer. Nothing in the source requires
they share a codec. Hot experts can stay HGRAVU01-q3 / affine2;
cold experts can go residual / HGRAVS / heavier entropy.

Prior that stays closed: G21/G22 (`QWEN38_LEGACY_INDEX` tag
**UNIVERSAL**) — a `bits=2` *label* is 3.22 bpw effective, the
same density class as Hawking q3. Do not advertise “2-bit
experts” without the S034 effective-bpw gate.

Prior that is reusable: G23 affine2 format + G25 kernel parity
(index **UNIVERSAL**). `HGRAVF01` =
`w = q*scale + bias`, group-32, codec 5
(`G23_AFFINE2_FORMAT.json`, `G24_NATIVE_DESIGN.json`). The
kernel is a GEMV and does not care that the matrix is an
expert. This worktree still has to grow codec 5 before that
sentence is operational.

#### Cold-expert compression — **MOE-SPECIFIC**

87–89% of stored parameters are not read on a given token
(§3). Compressing them changes **resident RAM**, not the
per-token GEMV stream. That is exactly the split G26 recorded
on the dense patient (“DENSITY work buys STORED density / RAM
but NOT TPS”, `G26_RUNTIME_DIAGNOSIS.json`) — and G26 is tagged
**DENSE-SPECIFIC** in the index. On MoE the RAM half becomes
the product: a 30B model at coherent bpw is ~12.6 GB stored
but only ~1.25 GB active (A); a 106B model is ~44 GB stored
/ ~5.3 GB active (B).

Do **not** claim this raises TPS. Claim it as a residency /
admission lever (`ExpertCache::evict_cold` /
`note_access`, `expert_cache.rs:87–144`; Q80
`device_residency` LRU named in
`qwen80_device_expert_table.rs:16–19`).

#### Router-aware placement / prefetch — **MOE-SPECIFIC**

The router is a 128-wide GEMV whose output *is* the residency
hint for the next expert wave. Q80 already pays a host
readback for this (`qwen80_mixed_hybrid_decode.rs:98–111`:
“512 f32 router logits must return to the host so
`source_qwen80_topk_router` can pick 10 experts”). A 128-wide
softmax/sigmoid is small enough that a device-side top-k
(`moe_topk_gate`) plus a 128-entry address table (Q80 table
shrunk) removes that readback.

`ExpertCache::mark_warm` (`expert_cache.rs:124–133`) is the
advisory `MADV_WILLNEED` hook; it is not wired to a Qwen3 /
GLM router.

Untested: whether A/B routers are sticky enough across tokens
that prefetch of the previous token’s experts wins. Label
the *mechanism* MOE-SPECIFIC, the *hit-rate* **UNRESOLVED**.

#### Matryoshka experts — **MOE-SPECIFIC** structure, **UNRESOLVED** quality

Two different ideas, do not conflate:

1. **G11 2-tier NR** (`G11_MATRYOSHKA.md`, MEASURED_WIN on
   Qwen3.8 dense MLP): HGRAVU01-q3 base + q2 residual
   correction. `PHASE_B_HYBRID_REFUTED.json` (index
   **QWEN-SPECIFIC**) already says this is a **quality** lever
   (more bytes, better error), not a byte-reduction lever.
   On MoE the new move is: store the correction plane on every
   expert, **apply it only for high router weight** (or only
   for the top-1 of the top-8). That does not exist on the
   dense patient. Structure: MOE-SPECIFIC. Whether Doctor
   holds: **UNRESOLVED**.
2. **Nested expert width** (run a 384-wide prefix of a 768
   expert, etc.). No measurement. **UNRESOLVED**. Do not
   borrow G11’s MEASURED_WIN for this.

#### Shared-expert / dense-layer as the “always-on” quality floor (B only) — **MOE-SPECIFIC**

B spends 17.3M params/layer on a shared expert that every
token hits, plus a 134M layer-0 dense MLP. Those organs are
the natural place to spend q3/affine2 quality. Routed experts
can be cheaper. A has no such organ — its always-on MLP-side
bytes are just the 262k router.

#### MTP decode (B only) — **MOE-SPECIFIC**

`num_nextn_predict_layers = 1`. G26 (index **DENSE-SPECIFIC**
as a Qwen3.8 TPS diagnosis, but the *lever* is architectural)
named MTP as the highest-leverage unused path to >40 TPS on
the dense patient. B actually has the weights (stripped from
the 46-layer graph, present as layer 46). A does not. Native
MTP is a new decode loop, not a representation.

### 5.2 Reopenable because they were dense-specific

| Finding | Index tag | Why MoE can differ | Tag here |
|---|---|---|---|
| Shared/grouped operator distillation (`DENSITY_LEVER_HONEST.json`, G3_*) | Index: **UNIVERSAL** for the honest verdict; G3_* **DENSE-SPECIFIC**; contract: operator-distillation is dense-specific | The dense failure was a **width bottleneck** (`m=6144` vs 17408). An expert is already 768 or 1408 wide — a shared-across-experts operator is a *different* question, and `QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json` already measured near-zero pairwise cosine on a 96-expert layer (gate mean direction cosine 0.122, pairwise 0.004). | **UNRESOLVED** for “one op across 128 experts”. **CLOSED** for “one op replaces a 17408 dense MLP”. Do not re-run G3 as if it were news. |
| Density-doesn’t-buy-TPS (`G26_RUNTIME_DIAGNOSIS.json`) | **DENSE-SPECIFIC** | Measured on always-on 11.7→10.1 GB GEMVs at 5120×17408. A’s active GEMVs are 768×2048 × 8 experts; launch/occupancy, not the G26 bandwidth-vs-kernel story. | **UNRESOLVED** for TPS-from-density on *active* experts. **CLOSED** as a reason to expect TPS from compressing *cold* experts. |
| G10 Metal shared-operator blueprint | **DENSE-SPECIFIC** | Same as distillation. | leave closed until an MoE-specific operator is proposed |
| G21_TEXT_ACTIVE | **DENSE-SPECIFIC** | Empty finding body; do not reuse. | ignore |
| Attention / DeltaNet active-byte reduction (`DENSITY_LEVER_HONEST.still_untested`) | listed as untested even on Qwen3.8 | A/B have **no DeltaNet**. Attention *is* always-on GQA, so an attention codec is UNIVERSAL-shaped and still untested. | Attention codec: **UNRESOLVED** (and newly a larger fraction of *active* bytes on A, because the MLP collapsed). DeltaNet work: **CLOSED** (no such mixer). |

### 5.3 100 TPS physics — **UNRESOLVED** on A, still hostile on B

`STRATEGIC_FINDING_100TPS.json` (index **UNIVERSAL** *for
Qwen3.8-27B*) says 100 TPS needs ≤~4.7 GB active at ~471 GB/s,
and that a 27B×3.3 bpw patient cannot get there.
It also names the reopener: **“DIFFERENT BASE MODEL… roughly
≤11B params at 3.3 BPW.”**

- **A**: 3.042B active × 3.3 bpw ≈ **1.25 GB** ≪ 4.7 GB.
  Bandwidth ceiling would be hundreds of TPS. That does **not**
  mean 100 TPS is delivered. A decode token is ~48 × (4 attn
  GEMVs + 1 router + 8×3 expert GEMVs) unless the expert wave
  is fused. G26’s other half still applies: this runtime can
  be kernel/dispatch bound. Treat 100 TPS on A as
  **UNRESOLVED**, physics-unblocked, dispatch-blocked until
  measured.
- **B**: 12.803B active × 3.3 bpw ≈ **5.28 GB** > 4.7 GB.
  Bandwidth ceiling ≈ 471/5.28 ≈ 89 TPS even before kernel
  tax. 100 TPS at coherent bpw is still physics-hostile
  unless MTP or a cheaper active set lands. **UNRESOLVED**
  only via MTP / colder shared-expert / lower-bpw active
  organs; not via cold-expert compression.

---

## 6. Qwen3.8 negatives that stay closed

Do not re-test these. Cite the index.

| Receipt | Index tag | Why it stays closed on MoE |
|---|---|---|
| `G23_AFFINE2_FORMAT.json` | UNIVERSAL | Format semantics (`w=q*scale+bias`, g32). Transplant, don’t rediscover. |
| `G24_NATIVE_DECODE.json` / `G24_NATIVE_DESIGN.json` | UNIVERSAL | Native affine2 GEMV works; mlx’s 1.2× is kernel/runtime, not representation. The kernel is reusable; the 31.5-vs-37.7 TPS number is Qwen3.8-shaped. |
| `G25_KERNEL_PARITY.json` | UNIVERSAL | Bit-exact affine2 GEMV vs CPU oracle. Don’t rebuild the dequant. |
| `EXEC_LEVER_MAP.json` | UNIVERSAL | `geo_tpr64` is already the HGRAVU01 default (`dispatch_uniform` → `qwen38_hgravu01_geo_tpr64_launch`). No leftover 3–4× flag. Small-M expert GEMVs may want a *different* launch, but that is a new bench, not a replay of geo vs simd3 on 17408×5120. |
| `CORRECTION_MLP_INPUT_TENSOR.json` | UNIVERSAL | Distillation on the wrong activation is a methodology bug. If any MoE operator work happens, the input is `post_attention_layernorm(h)`, matching `Qwen3MoeDecoderLayer.forward:345–346` / `Glm4MoeDecoderLayer.forward:446–447`. |
| `METHODOLOGY_AUDIT.json` | UNIVERSAL | Leakage / aggregation / wrong-input can fake a “beats q3” headline. Applies to any future operator claim. |
| `G21_G22_FINDINGS.json` / `G21_2BIT_CENSUS.json` | UNIVERSAL | `bits=2` ≠ 2.0 bpw. Effective-bpw gate stays on. |
| `DENSITY_LEVER_HONEST.json` assembled-Doctor | UNIVERSAL *as a methodology + “narrow bottleneck cannot replace a wide dense MLP”* | Do not try to replace A’s 768-wide expert *or* B’s 1408-wide expert with an even narrower shared op and expect Doctor to hold. The G3 “breakthrough” receipts (DENSE-SPECIFIC) are the refuted ones. |
| `PHASE_B_FUNCTIONAL_LOWRANK.json` | QWEN-SPECIFIC | Activation-aware functional low-rank overfit on the dense MLP. Not a reason to try the same fit on a 768-wide expert without a new protocol; also not a reason to declare experts low-rank. Leave closed until a new, pre-registered expert-rank study exists. |
| `PHASE_B_HYBRID_REFUTED.json` | QWEN-SPECIFIC | q2+correction cannot beat q3 on *active* bytes. Still the Pareto statement for any **always-on** organ (A’s attention, B’s shared expert + layer-0 MLP). Does not forbid storing a correction plane on *cold* experts. |
| `MLP_ACTIVATION_SPARSITY.json` | QWEN-SPECIFIC | Dense-MLP activation sparsity, ~2× cap, compounds over 62 layers. MoE already *is* the conditional-compute lever (8/128). Do not re-hunt silu-zeros inside a 768-wide expert as a TPS program. |
| `KV_STATE_COST.json` | QWEN-SPECIFIC | On the *hybrid* (DeltaNet+GQA) patient, state was 2.3% of the token. A/B are all-GQA: KV grows with context (`48*4*128*2` or `46*8*128*2` elements per token of context). The Qwen3.8 “state is not a speed lever” sentence does **not** transfer; the measurement method does. Re-measure if context ≫ 128; don’t quote the 2.3% number. |
| `G15_ZERO_FALLBACK.json` / `G17_RECONSTRUCTION_COST.json` | QWEN-SPECIFIC | Process rules (refuse on Err, no dense-W expand). Keep the rules; they are not findings to re-test. |
| `G29_GENESIS_NR.json` | UNIVERSAL | NR schema. Any MoE pack still binds `catalog_content_sha256`. |
| `ABLITERATED_SOURCE_PROVENANCE.json` / `PATIENT_IDENTITY_VECTOR.json` | QWEN-SPECIFIC | Those documents are about the Qwen3.8 abliterated patient. A is a *different* abliterated checkpoint; B is derestricted GLM. Identity/Doctor work must be redone per patient. The *existence* of an abliterated Qwen3-30B-A3B is not a Qwen3.8 finding to replay. |

### 6.1 Affine2 / HGRAVF01 / native GEMV — UNIVERSAL, status on this worktree

Contract: “affine2/HGRAVF01 kernel + native GEMV are UNIVERSAL;
do not re-test.”

- Format and kernel parity: **closed**, G23+G25.
- This worktree’s decoder: codec 5 **refused**
  (`qwen38_hybrid_decode.rs:198–200, 4406–4414`). Native GEMV
  kinds that *are* wired: Binary / Residual / HGRAVS /
  HGRAVU01 Uniform (`MixedGpuKind` match at `:1366–1506`,
  dispatch at `:1571–1576`).
- Parent working tree (not this branch) has
  `MixedCatalogLane::Affine`, `MixedMlpNativeKind::AffineScaleBias`,
  `GpuAffine`, `dispatch_affine`. That is integration, not new
  science.
- `encode_named_matvec` is the universal “name → packed GEMV”
  hook any MoE decoder should call.

---

## 7. Side-by-side topology (for the campaign fork)

| | A `qwen3_moe` | B `glm4_moe` | Qwen3.8 decoder today |
|---|---|---|---|
| Layers | 48, all MoE | 46 (1 dense + 45 MoE) + ignored MTP | 64 (48 DeltaNet + 16 GQA), all dense MLP |
| Hidden | 2048 | 4096 | 5120 |
| Routed experts | 128 / top-8 | 128 / top-8 | 0 |
| Shared expert | 0 | 1 × 1408 | n/a |
| Expert width | 768 | 1408 | MLP 17408 |
| Router | softmax, renorm | sigmoid + bias, renorm, ×1.0 | none |
| Attention | GQA 32/4/128, QK-norm, full RoPE, no bias, no gate | GQA 96/8/128, no QK-norm, RoPE 64/128, **bias**, no gate | hybrid; GQA 24/4/256, QK-norm, RoPE 64/256, **sigmoid gate** |
| Active / stored | 3.04B / 30.53B | 12.80B / 106.85B | ~27B always-on (dense) |
| Native steal | Qwen30 128/8 + 32/4/128 contracts | almost nothing typed | this file |
| 100 TPS physics | unblocked on bytes, blocked on dispatch until measured | still above the 4.7 GB line at 3.3 bpw | closed |

---

## 8. Recommended next measurements (not this contract)

Weights are being fetched separately. When they land, the first
honest numbers are:

1. Router histogram on a real decode trace (per-layer unique
   experts / 1k tokens). This sizes the cold-expert and
   prefetch claims. **UNRESOLVED** until then.
2. One-layer native expert wave at A’s 768×2048 top-8, using
   `dispatch_qwen80_device_expert_table_tcb` retargeted, vs
   eight serial `encode_named_matvec`s. This is the A TPS
   question. **UNRESOLVED**.
3. Do not start a shared-operator training run. The index
   already spent that money on the dense patient, and
   `QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json` is a prior
   negative on expert collinearity.

---

## Sources

Framework (installed):

- `~/.grok-vision/lib/python3.12/site-packages/mlx_lm/models/qwen3_moe.py`
- `~/.grok-vision/lib/python3.12/site-packages/mlx_lm/models/glm4_moe.py`
- `~/.grok-vision/lib/python3.12/site-packages/mlx_lm/models/switch_layers.py`
- `~/.grok-vision/lib/python3.12/site-packages/transformers/models/qwen3_moe/configuration_qwen3_moe.py`
- `~/.grok-vision/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py`
- `~/.grok-vision/lib/python3.12/site-packages/transformers/models/glm4_moe/configuration_glm4_moe.py`
- `~/.grok-vision/lib/python3.12/site-packages/transformers/models/glm4_moe/modeling_glm4_moe.py`

Configs (cache only, no weights):

- `~/.cache/huggingface/hub/models--huihui-ai--Huihui-Qwen3-30B-A3B-Thinking-2507-abliterated/snapshots/174f8bf573e9c5536451248d7419618064596edb/config.json`
- `~/.cache/huggingface/hub/models--ArliAI--GLM-4.5-Air-Derestricted/snapshots/09fba4859d9d902b9efa14c738775cda9b7fbc5f/config.json`

Native decoder / MoE primitives (this worktree):

- `crates/hawking-core/src/model/qwen38_hybrid_decode.rs`
- `crates/hawking-core/src/model/qwen38_geometry.rs`
- `crates/hawking-core/src/model/qwen38_64_layer_execution_schedule.rs`
- `crates/hawking-core/src/model/qwen_moe.rs`
- `crates/hawking-core/src/model/qwen80_mixed_hybrid_decode.rs`
- `crates/hawking-core/src/model/qwen80_device_expert_table.rs`
- `crates/hawking-core/src/moe.rs`
- `crates/hawking-core/src/model/expert_cache.rs`
- `crates/hawking-core/src/kernels/mod.rs` (`moe_topk_gate_tcb`, `moe_block_batched_indexed_metal`)
- `crates/hawking-core/shaders/moe.metal`

Legacy index (parent working tree, not in `HEAD`):

- `receipts/ascent-2026-08-18/QWEN38_LEGACY_INDEX.json`
- `receipts/ascent-2026-08-18/DENSITY_LEVER_HONEST.json`
- `receipts/ascent-2026-08-18/G26_RUNTIME_DIAGNOSIS.json`
- plus G23/G24/G25, `EXEC_LEVER_MAP.json`, `STRATEGIC_FINDING_100TPS.json`,
  `G11_MATRYOSHKA.md`, `PHASE_B_HYBRID_REFUTED.json`,
  `receipts/QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json`
