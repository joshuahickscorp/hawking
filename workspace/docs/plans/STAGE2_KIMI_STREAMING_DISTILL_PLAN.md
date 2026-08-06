# Stage-2 Kimi Streaming Distill Plan

**Status:** PLAN ONLY — no streaming, no model consumption, no runtime edits in this lane.  
**Seal companion:** `workspace/campaign/evidence/models/frankenstein/STAGE2_KIMI_STREAMING_DISTILL_PLAN.json`  
**Depends on:** Stage-1 Proto seal (DeepSeek-V4-Flash body + GLM math inheritance sealed)  
**Produces:** Final Frankenstein = DeepSeek + GLM + Kimi strategic inheritance

---

## 1. Settled place in the programme

| Stage | Inheritance | Result name |
|------|-------------|-------------|
| 1 (active) | GLM mathematical → DeepSeek-V4-Flash body | **Proto-Frankenstein** |
| **2 (this plan)** | **Kimi K3 strategic/agentic → Proto** | **Final Frankenstein** |
| 3 | Odyssey + formal tools + verifier + Q-gauntlet + sandbox qualification | **Ramanujan** |

Stage 2 does **not** redesign the fusion operation. It executes the already-sealed
`block_wise_streaming_distillation_via_latent_bridge` path for the
**`KIMI_STRATEGIC_BRIDGE` only**, after Proto shards from Stage 1 are sealed and
storage has been cleared for a single-donor Kimi window.

---

## 2. Source identity (do not invent; pin these)

### Student body (read-only)

| Field | Value | Source |
|------|-------|--------|
| Repo | `deepseek-ai/DeepSeek-V4-Flash` | official model card + local admission |
| Revision (pinned) | `60d8d70770c6776ff598c94bb586a859a38244f1` | `DEEPSEEK_V4_FLASH_SOURCE_ADMISSION.json` |
| Total / active params | **284B / 13B** | [DeepSeek-V4-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |
| Layers / hidden | 43 / 4096 | local fusion geometry + admission |
| MoE | 256 routed, 1 shared, **6 experts/token** | admission |
| Context | 1M tokens | model card |
| Source weight bytes (blobs) | ~159.6 GB | admission `source_gb_from_blobs` |
| Native precision note | FP4 experts + FP8 rest (fp4-in-I8 packed); **not** a clean BF16 body | admission `source_precision` |
| Full gravity stream status | `FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY` | pipeline plan / pilot schedule |

### Strategic donor (Kimi K3)

| Field | Value | Source |
|------|-------|--------|
| Repo | `moonshotai/Kimi-K3` | official model card + local admission |
| Revision (pinned) | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` | `KIMI_K3_SOURCE_ADMISSION.json` |
| Total / active params | **2.8T / 104B** | [Kimi-K3 model card](https://huggingface.co/moonshotai/Kimi-K3) |
| Layers / hidden | 93 / 7168 | model card + admission |
| MoE | **896 experts, 16 selected/token, 2 shared** | model card |
| Context | 1,048,576 | model card |
| Weight shards | 96 shards, **1,560,936,091,448 bytes (~1.56 TB)** | admission |
| Multimodal | native text + image (MoonViT-V2, 401M) | model card |
| Admission status | `KIMI_K3_OFFICIAL_SOURCE_ADMITTED_METADATA_ONLY` | admission seal |
| Weight materialisation | **0 shards downloaded at admission** | admission storage |

**Law:** metadata admission is **not** permission to restream the body or acquire
teacher traces. Stage-2 stream launch requires separate owner/runtime gates
(listed in §8).

### Projection geometry (already sealed in fusion op)

```
a_d  : [B, S, 7168]   # Kimi hidden
W    : [7168, 4096]   # project_kimi_k3_to_student
a_d' : [B, S, 4096]   # student latent
A    : residual adapter on student activations (zero-init)
a_out = a_s + A(a_s)
```

Direct weight transplant is **impossible** and **prohibited**
(`direct_weight_transplant=false`). Elementwise averaging of Kimi/GLM/DeepSeek
weights is undefined (H ∈ {7168, 6144, 4096}, L ∈ {93, 78, 43}).

---

## 3. Preconditions (gate checklist)

Stage-2 **must not start** until all of the following hold:

1. **Proto seal complete**
   - Stage-1 GLM blocks sealed under the fusion archive contract.
   - Progress cursor shows all `GLM_MATH_BRIDGE` production blocks complete
     (or an explicit owner-scoped pilot with residual plan recorded).
   - Capability gate on Proto: **measure, do not assume** — Proto must clear
     the frozen quality probes that Stage-1 defines before Kimi inheritance
     is allowed to land on top.

2. **Clear storage (hard requirement of this plan)**
   - Evict **all** GLM donor windows, GLM scratch, and any residual Stage-1
     Xet/HF caches that are not part of the sealed Proto archive or the
     read-only DeepSeek body.
   - Re-check free disk against the working-set invariant before the first
     Kimi window fetch.
   - Floor: `hard_free_floor_bytes = 16_106_127_360` (15 GiB) from the sealed
     Frankenstein storage contract. **Do not start a Kimi window if
     `free - (working_set + next_window) < floor`.**

3. **Working-set invariant (unchanged)**
   ```
   DeepSeek body (read-only, already on volume)
     + at most ONE donor window (Kimi only)
     + current output block
     + scratch
   ```
   Holding GLM **and** Kimi resident is `PROHIBITED_BY_DISK_CONTRACT`.

4. **DeepSeek forward gate**
   - Pilot schedules still mark `fit_gate: DEEPSEEK_FORWARD_PENDING` on Kimi
     blocks. Live residual fitting requires a student forward that produces
     activations at each transplant point. Until that gate opens, blocks may
     only materialise fixtures / sealed stubs — not claim inheritance.

5. **Authority**
   - Ramanujan research authorisation remains a **separate** Stage-3 concern.
   - Stage-2 is engineering inheritance into the DeepSeek child, not Odyssey
     teacher-trace acquisition. Do not conflate the two.

---

## 4. What Kimi donates (strategic / agentic inheritance)

Stage-2 targets **behavioural and latent alignment** at the preserved
`KIMI_STRATEGIC_BRIDGE` transplant points — **not** dumping 1.56 TB of weights
into the child.

### Inheritance targets (capability classes)

| Class | Intent | Notes |
|------|--------|-------|
| Long-horizon strategy | Multi-step plan formation, mid-course correction | Align residual at deep layers + final hidden |
| Knowledge work | Research synthesis, structured report / dashboard style work | Corpus must include multi-document prompts |
| Coding breadth | Repo-scale navigation, tool-mediated coding | Prefer agentic coding corpora over single-file puzzles |
| Context management | Use of long context without collapse / thrash | Kimi card: 1M ctx; student also 1M — train compaction behaviour, not raw length brag |
| Agent orchestration | Tool choice, multi-tool plans, MCP-style workflows | Bind to HCLI tool-action decision points where present |
| Critique & synthesis | Independent alternative, falsifier, arbitration tone | Matches Odyssey's later K3 role as independent alternative — Stage-2 seeds the substrate |

### Explicit non-targets

- Multimodal vision weights as a full second body (MoonViT may inform
  **future** work; Stage-2 text/agentic bridge is the default scope unless
  owner expands).
- Replacing DeepSeek routers/experts with Kimi experts.
- Simultaneous dual-donor averaging.
- Claiming "Final has Kimi's 104B active capacity" — the student remains
  284B/13B active; density rises via adapters + better use of existing
  experts, not by growing the body.

---

## 5. Double-stream law (why sequential, not all-at-once)

**Double-stream** means two **sequential** inheritance streams over the **same
small student body**:

1. Stream A — GLM math (Stage 1) → Proto  
2. Stream B — Kimi strategic (Stage 2) → Final  

Not: materialise both donors, average, and hope.

### Why this is the intended win (MoE / router mechanism)

DeepSeek-V4-Flash is already a sparse MoE: **256 routed experts, top-6 per
token, ~13B active of 284B total**. Capability density is not the same as
parameter count.

| Mechanism | How sequential streams raise density without growing the body |
|-----------|----------------------------------------------------------------|
| **Expert specialisation** | Residual adapters + route-aware targets push different token classes toward different expert coalitions (math-like vs agentic/coding-like). Specialisation is learned in latent space, not by pasting donor expert matrices. |
| **Router capacity** | The router has finite bits per token. Teaching it one coherent specialisation (math), sealing, then teaching a second (strategy/agentic) avoids gradient conflict from two huge mismatched donors in one joint objective. |
| **Active-bytes / token** | Decode cost tracks **active** experts + attention, not total params. Inheritance that improves *which* experts fire and *how* residuals steer them can raise capability while active-bytes/token stay near the existing roofline (~11.7 GB teacher active path analytical; see runtime accounting). |
| **Bounded memory** | Only one donor window is ever resident. Peak disk/RAM is set by window size + body + scratch, not by 1.56 TB + GLM + student. |
| **Reversibility** | Each stream seals **separately content-addressed residual adapters**. Stage-2 can be rolled back without erasing Stage-1; Stage-3 does not need to re-touch raw Kimi weights. |

**Density claim (honest):** sequential streams are intended to increase
*effective capability per resident byte* and *per active byte/token*. They do
**not** magically compress Kimi's 2.8T into the child. Promotion still
requires the four-way ablation (see `FRANKENSTEIN_PROGRAM.md`).

---

## 6. Streaming schedule (same bounded-evicting pattern as GLM)

Mirror the GLM lifecycle; substitute Kimi as the sole donor.

### Per-block lifecycle

1. `disk_floor_check`  
2. `stream_one_donor_window` (Kimi shard range only)  
3. `verify_range_identity_and_provenance` (LFS SHA-256 from admission inventory)  
4. `project_donor_activations` (7168 → 4096) **or** record `PENDING`  
5. `student_forward` **or** `DEEPSEEK_FORWARD_PENDING`  
6. `fit_residual_adapter` **or** stub  
7. `seal_output_block_raw_no_gravity`  
8. `evict_donor_window_and_scratch`  
9. `append_progress_cursor`

### Window budget (planning bounds — measure before production)

| Budget item | Planning value | Authority |
|-------------|----------------|-----------|
| Donor window budget | 32 GiB (`34_359_738_368`) | pilot schedule `working_set` |
| Output block budget | 2 GiB | pilot |
| Scratch | 4 GiB | pilot |
| Working set total | ~40.8 GB | pilot |
| Kimi full body | ~1.56 TB | admission — **never resident whole** |
| Max simultaneous donors | **1** | storage contract |
| Gravity on Stage-2 output | **false** during fusion archive | fusion op — gravity is a later recomposition stage |

Kimi has **96 weight shards**. A production schedule must:

- Cover every shard needed for the layers that map onto student layers
  under  
  `donor_layer = round(student_layer * (93-1) / (43-1))`.
- Prefer **layer-contiguous windows** so a student layer's mapped donor
  activations can be computed without refetch thrash.
- Autotune window size against APFS allocation, reconstruction scratch,
  swap, thermals — same spirit as `GLM52_XET_AUTOTUNE_PLAN`, but for Kimi.
  **Do not ship a window size from arithmetic alone.**

### Layer map examples (from fusion op)

| Student layer | Kimi donor layer |
|---------------|------------------|
| 0 | 0 |
| 42 | 92 |

### Bridge / transplant points for Stage-2

Primary bridge: **`KIMI_STRATEGIC_BRIDGE`**.

Default production transplant point (pilot already freezes this as the first
live shape): **`post_moe_hidden_state`**.

Full transplant-point catalogue (frozen v3, no weight graft) includes among
others:

- `pre_norm_hidden_state`, `post_attention_hidden_state`
- `pre_router_hidden_state`, `router_logits`, `selected_expert_ids`,
  `route_probabilities_and_margins`
- `post_moe_hidden_state`, `mhc_state`, `attention_index_state`
- `final_hidden_state`, `lm_head_logits`
- `hcli_tool_action_decision`

**Stage-2 prioritisation (recommended executable order):**

1. `post_moe_hidden_state` — full 43 student layers (strategic residual body)  
2. `final_hidden_state` + `lm_head_logits` — output behaviour  
3. `hcli_tool_action_decision` — agent orchestration surface  
4. Router-family points (`pre_router_*`, `router_logits`, `selected_expert_ids`)
   — only after forward gate is green; these are the density levers for
   expert specialisation  
5. Attention-index / MHC — only if capability ablation shows residual value  

Do **not** schedule all 12 points × 43 layers on day one. Expand only when
the four-way ablation shows contribution.

### Loss (from fusion op; gated on student forward)

```
mse_projected_donor:  || (a_s + A(a_s)) - a_d' ||_2^2 / (B·S·H)   weight 1.0
cosine_alignment:     1 - cos(a_s + A(a_s), a_d')   weight 0.1
```

Token alignment: **prompt-id + char-span**, not raw token id (tokenizers
differ; Kimi vocab 160K vs DeepSeek 129280).

---

## 7. Corpus for Stage-2 (planning requirements)

Stage-2 needs a **shared prompt corpus** that can be tokenized by both
tokenizers and that stresses strategic/agentic behaviour:

- Long-horizon multi-file coding sessions  
- Tool-use / HCLI-style action decisions  
- Knowledge-work synthesis with sources  
- Critique / independent-alternative prompts (prefigures Odyssey K3 role)  
- Context-management prompts (summarise, compact, resume after 100K+)  

**Unknown until measured:** corpus size required for stable residual fit.
Plan to start with a sealed pilot corpus (small, hash-bound), fit L00 only,
then scale.

---

## 8. Executable phase list (Stage-2 only)

| Phase | Work | Human? | Exit gate |
|------|------|--------|-----------|
| S2.0 | Proto seal inventory + free-disk report | no | Proto complete; floor free |
| S2.1 | **Clear storage** — evict GLM windows/caches | no (operator confirm) | only Proto archive + DSV body + floor |
| S2.2 | Freeze Stage-2 streaming schedule JSON (full or pilot→full) | no | schedule seal; every needed shard scheduled once |
| S2.3 | Open DeepSeek forward / activation capture for all 43 layers | eng | `DEEPSEEK_FORWARD` green |
| S2.4 | Bounded Kimi window stream + fit residuals | eng | progress cursor advances; each block sealed |
| S2.5 | Stage-2 archive seal (adapters + projections, no gravity) | eng | content-addressed archive |
| S2.6 | Capability probes: Proto vs Final on strategic suite | eng | Final ≥ Proto on strategic axes; no math regression |
| S2.7 | Mark Final Frankenstein candidate | eng | ready for optional gravity recomposition + Stage-3 |

**Upload / cloud seal of Final is MANUAL and lives in
`POST_FINAL_GRAVITY_HCLI_PLAN.md` — not automated here.**

---

## 9. Prohibitions

- Do not hold two donors resident.  
- Do not average misaligned tensors.  
- Do not transplant Kimi weights into DeepSeek slots.  
- Do not retain raw donor weights, logits, KV, or hidden states as the
  inheritance payload after the block seals.  
- Do not claim Final = 2.8T or 104B active.  
- Do not start Stage-2 while GLM windows still occupy the floor.  
- Do not treat metadata admission as stream authorisation.  
- Do not gravity-compress Stage-2 blocks until a separate recomposition plan
  says so.

---

## 10. Open measurement items (do not invent)

| Item | Plan stance |
|------|-------------|
| Peak resident RAM of one Kimi window + student forward | **measure, do not assume** |
| Optimal window size / shard count | autotune after S2.1 |
| Whether router-point adapters beat post-MoE-only | ablation after S2.6 |
| Multimodal inheritance value on this Mac | out of Stage-2 default scope |
| Wall-clock for full 43-layer Kimi residual fit | unknown until forward gate + first window |

---

## 11. References (local + cards)

- Local fusion: `workspace/campaign/evidence/models/frankenstein/FRANKENSTEIN_FUSION_OPERATION.json`  
- Pilot schedule: `.../FRANKENSTEIN_DIRECT_STREAMING_SCHEDULE_PILOT.json`  
- Kimi admission: `workspace/campaign/evidence/models/kimi-k3/KIMI_K3_SOURCE_ADMISSION.json`  
- DSV admission: `workspace/campaign/evidence/models/deepseek-v4/DEEPSEEK_V4_FLASH_SOURCE_ADMISSION.json`  
- Pipeline (K3 admitted): `.../DEEPSEEK_V4_FRANKENSTEIN_PIPELINE_PLAN_K3_ADMITTED.json`  
- GLM streaming pattern: `workspace/campaign/evidence/models/glm52/GLM52_STREAMING_SCHEDULE.md`  
- Cards: [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash), [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
