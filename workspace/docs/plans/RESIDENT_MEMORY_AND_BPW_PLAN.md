# Resident Memory & BPW Reality Plan

**Status:** PLAN ONLY — honest memory/density law for the 96 GB box.  
**Seal companion:** `workspace/campaign/evidence/models/frankenstein/RESIDENT_MEMORY_AND_BPW_PLAN.json`  
**Machine:** Apple M3 Ultra Studio, **96 GB UMA**, ~819 GB/s advertised BW, 1 TB SSD  
**Process admission budget (coding-agent tenant):** **78 GiB**, not full 96 GB  
  (see `workspace/docs/reference/BASELINES.md`)

---

## 1. Verdict first (read this before any density claim)

| Claim | Verdict |
|-------|---------|
| "1.5 BPW is proven capability-safe for flagship MoE" | **FALSE as stated** — 1.5 BPW is a **target / planning floor**, not a proven threshold |
| "Sub-1 BPW on GLM was fine because integrity sealed" | **FALSE** — sealed integrity ≠ capability; live generation collapsed |
| "DSV4F @ 1.5 BPW = 53 GB, Qwen @ 1.5 BPW = 5.7 GB, therefore both co-resident in 96 GB" | **ARITHMETIC FLOOR ONLY** — co-residency is decided by **measured resident memory**, not this sum |
| "Force every tensor to 1.5 BPW" | **Not the preferred MoE strategy** — prefer moderate total size + low active-bytes/token + expert cache |
| Default resident model on the box | **Qwen Gravity (Luna)** — always-on coding executor |
| DeepSeek / Frankenstein / Ramanujan | **Load-on-demand (Terra / Sol)** |

---

## 2. The GLM sub-1 BPW collapse (cite, do not repeat)

Local sealed evidence in Odyssey substrate capability:

**Artifact:** `GLM-5.2-H0.98-Math-Preserve`  
- `complete_bpw ≈ 0.977`  
- Integrity: **SEALED_AND_COMPLETE** (282/282 shards, frozen decisions match)  
- Capability: **REFUSED** — `H1_ARTIFACT / SEMANTIC_COLLAPSE`  
- Live probes: "The capital of France is" → `combust`; **"2 + 2 =" → `rus`**  
- Source: `workspace/campaign/governance/odyssey/program/launch/SUBSTRATE_CAPABILITY.json`

**Artifact:** `GLM-5.2-General-R0`  
- `complete_bpw ≈ 0.88–0.92` even with lm_head / embed_tokens native protection  
- Capability: **REFUSED** — standing note `R0_IS_THE_PRACTICAL_CEILING` for that family path

**Dead-lever table** (`workspace/docs/guides/dead_levers.md`):

| Gate | Status | Note |
|------|--------|------|
| GLM-5.2 sub-bit MoE expert path | NO-GO Type-1 | Four families **0.116–0.157 cos @ 0.75 BPW**; none beat null **0.898** |

**Law for this programme:**

> Integrity sealing, allocation headroom, and "math preserve" labels are not
> capability. Any BPW target below a **measured** capability gate is a research
> aspiration. **1.5 BPW is a planning target, not a license to ship.**

Related negative transfer (Qwen3-235B foundry atlas): collapses observed near
~1.0 and ~0.5 complete BPW on real forwards — another reason not to treat
"~1 BPW" as safe by default.

---

## 3. Raw arithmetic floors (planning only)

Formula: `bytes ≈ N_params × BPW / 8`.

| Model | Params (card) | @ 1.5 BPW | Notes |
|-------|---------------|----------:|-------|
| DeepSeek-V4-Flash total | 284B | **≈ 53.25 GB** | Weight-only floor; ignores everything in §4 |
| DeepSeek-V4-Flash active | 13B | **≈ 2.44 GB** | Active-path floor for *hot* experts only |
| Qwen3-Coder-30B-A3B total | 30.5B | **≈ 5.72 GB** | Weight-only floor |
| Qwen3-Coder-30B-A3B active | 3.3B | **≈ 0.62 GB** | Active-path floor |
| Kimi-K3 total | 2.8T | **not a resident target** | ~1.56 TB source shards; stream-only donor |
| Kimi-K3 active | 104B | **not resident on this box as teacher** | Stage-2 streams windows |

**Native DSV4F on disk is already mixed precision** (FP4 experts + FP8 rest;
local admission ~159.6 GB source blobs). That is **not** "284B × 16-bit".
Any "1.5 BPW recompress" must be measured **against capability gates**, not
against the marketing total alone.

**Sum of weight floors at 1.5 BPW:**  
`53.25 + 5.72 ≈ 59 GB` of *ideal weight bytes* — still **not** a co-residency
proof on a 96 GB / 78 GiB-admission machine once §4 is charged.

---

## 4. What actually occupies resident memory

Any honest resident budget must charge **all** of these. Unknowns stay
unknown until measured.

| Bucket | What it is | Planning stance |
|--------|------------|-----------------|
| **Packed weights** | Gravity / quant tensors actually mapped | measure RSS after load |
| **Native / protected tensors** | Often higher precision: `lm_head`, `embed_tokens`, norms, routers | **do not force 1.5 BPW** without gates; GLM R0 still failed after protecting some of these |
| **Scales / codebooks / LUTs** | FP8/FP4 scales, PQ codebooks, Doctor side-info | charge complete BPW, not "weight file BPW" only |
| **Kernel tables** | Metal pipelines, shader libraries, fused kernel workspaces | process baseline; measure cold vs warm |
| **Runtime buffers** | Activations, scratch GEMM, MoE gather/scatter | scales with batch and hidden size |
| **KV cache** | Grows with context length × layers × KV dims × dtype | **dominant at long context**; 1M context is not free |
| **Expert cache** | Hot experts retained; cold experts faulted | MoE's real lever — see §5 |
| **HCLI / agent overhead** | Tool schemas, session state, hide-kernel, orchestrator | not zero; coding-agent is a protected tenant |
| **macOS / system reserve** | WindowServer, kernel, other apps | full 96 GB is never yours |
| **Admission budget** | **78 GiB** process budget for the coding-agent tenant | BASELINES |
| **Capability gates** | Quality probes that refuse a dense-but-dead artifact | hard stop |

### Context-length warning

Both DSV4F and Kimi advertise **1M context**. Resident plan for interactive
use should pick **measured working contexts** (e.g. 4K / 32K / 128K profiles)
and publish KV bytes at each. **Do not claim 1M-context co-residency** without
a measurement.

### Disk floors (orthogonal but binding)

- Frankenstein fusion contract: `hard_free_floor_bytes = 15 GiB`.  
- BASELINES storage planning: current free with **150 GB hard floor + 64 GB
  scratch + 32 GB HF/Xet cache** for heavy procure/processing.  
  Use the **stricter applicable floor** for the operation class
  (fusion window vs frontier procure).

---

## 5. MoE strategy: prefer active-bytes + expert cache over uniform 1.5 BPW

DeepSeek-V4-Flash is **sparse by design**:

- 256 routed experts, **6 active / token**, 1 shared  
- Card: **13B active of 284B total**  
- Local runtime accounting (analytical, bandwidth 417.7 GB/s measured):  
  - teacher complete-token active path ≈ **11.7 GB**  
  - functional student path ≈ **7.7 GB**  
  - attention dominates token traffic; experts already FP4 so further MoE
    byte wins are limited (~9.8× on MoE vs teacher analytical, whole-token
    win smaller)

**Argument (planning law):**

1. **Moderate total size** of a Gravity-packed child can exceed a naive 1.5
   BPW uniform pack **if** protected tensors stay healthy and capability
   gates pass.  
2. **Low active-bytes/token** is what sets decode roofline and much of peak
   working set during generation.  
3. **Strong expert caching** (hot-N experts resident, cold on SSD) can beat
   "force every expert tensor to 1.5 BPW" both for quality and for effective
   RAM, because most experts are cold on any single token.  
4. Uniform sub-bit assault on **all** experts is exactly the class of lever
   that died on GLM (0.75 BPW cos collapse).

Therefore for Terra / Sol packaging:

```
priority:
  1) capability gates green
  2) minimize measured active-bytes/token
  3) expert cache hit-rate under real HCLI workloads
  4) total resident RSS under admission budget
  5) only then chase average BPW headlines
```

1.5 BPW remains a **useful target line on a spreadsheet**. It is **not** the
promotion criterion.

---

## 6. Switching plan for the 96 GB box

### Default topology

| Slot | Model | Policy |
|------|-------|--------|
| **Luna (default resident)** | Qwen3-Coder-30B-A3B Gravity | Always loaded for coding-agent loops |
| **Terra (on demand)** | DeepSeek-V4-Flash Gravity | Load when general/agent body needed |
| **Sol (on demand)** | Final Frankenstein / Ramanujan | Load for flagship strategy / hard tasks |
| **Donors (never default resident)** | GLM-5.2, Kimi-K3 | Stream windows only during inheritance builds |

### Switching rules

1. **Qwen stays hot** unless an explicit exclusive heavy-lease says otherwise
   (BASELINES: exclusive heavy-work lease shared by processing + Studio).  
2. **Load Terra or Sol** into the remaining admission headroom after Qwen's
   **measured** RSS + KV for the active session profile.  
3. **Never assume** Terra+Luna co-residency from 53+5.7 GB arithmetic.  
   - If measured `RSS_qwen + RSS_terra + KV + HCLI + reserve ≤ 78 GiB`
     admission and capability gates hold → co-residency **allowed**.  
   - Else → **swap**: unload one body before loading the other; keep
     conversation state external so reload is cold-start weights only.  
4. **Sol vs Terra:** Sol (Frankenstein) is a **successor body + adapters** of
   the DeepSeek family, not a third simultaneous Dense+MoE giant. Plan for
   **Sol replacing Terra in RAM** when flagship is needed, not Sol+Terra+Luna
   all hot, unless measurement says otherwise.  
5. **Kimi body** never becomes a third interactive resident on this Mac.

### Measurement protocol (required before co-residency claims)

For each of Qwen Gravity, DSV4F Gravity, Final (when exists):

1. Cold load; record peak UMA, process RSS, swap delta.  
2. Warm generate at context profiles {4K, 32K, 128K} (extend only if safe).  
3. Record active-bytes/token (or proxy: bytes moved / token).  
4. Run capability gate suite.  
5. Dual-load attempt: Qwen + candidate; if pressure/swap/thermal trip →
   mark `CO_RESIDENT=false` with receipt.  
6. Seal results under campaign evidence; **update this plan's JSON**, do not
   hand-wave.

Until those receipts exist, all dual-load statements remain:
**"measure, do not assume."**

---

## 7. Gravity packaging implications

- Gravity is the **serve format** for Terra and Luna after Post-Final
  re-download (see post-final plan).  
- Frankenstein fusion archive during Stages 1–2 is **raw residual adapters
  (no gravity)** per fusion op; recomposition/gravity is a later stage.  
- When gravitizing:
  - Protect tensors that historically collapse first (lm_head, embeddings,
    routers) until gates say they can go lower.  
  - Prefer representation escalation before BPW reduction
    (`representation-before-BPW` in substrate capability notes).  
  - Charge **complete BPW** (side streams, scales, adapters), not marketing
    file BPW.

---

## 8. HCLI memory note

HCLI is both a **capability surface** (tool-action decisions are transplant
points) and a **memory consumer** (sessions, grants, tool schemas,
orchestrator). The Frankenstein programme must not treat "model RSS" as the
whole process. Admission budget **78 GiB** already anticipates a protected
interactive tenant.

Kernel tuning priority after Post-Final (brokers of open-source models) is
about **latency and correctness under this memory law**, not about stuffing
two full flagships into UMA by optimism.

---

## 9. Open questions (owner / measurement)

1. Measured RSS of Qwen3-Coder-30B-A3B Gravity at the intended quant on this
   box?  
2. Measured RSS of DSV4F Gravity (native mixed vs recompressed)?  
3. Expert-cache hot-N that maximises HCLI hit-rate under real sessions?  
4. Acceptable context profiles for interactive Sol vs batch?  
5. Owner tolerance bands for capability regression when lowering BPW?

---

## 10. References

- BASELINES: `workspace/docs/reference/BASELINES.md`  
- Dead levers: `workspace/docs/guides/dead_levers.md`  
- Substrate capability: `workspace/campaign/governance/odyssey/program/launch/SUBSTRATE_CAPABILITY.json`  
- DSV runtime accounting: `workspace/campaign/evidence/models/deepseek-v4/DEEPSEEK_V4_RUNTIME_ACCOUNTING.json`  
- DSV source admission: `.../DEEPSEEK_V4_FLASH_SOURCE_ADMISSION.json`  
- Cards: DeepSeek-V4-Flash, Kimi-K3, Qwen3-Coder-30B-A3B-Instruct
