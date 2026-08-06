# Post-Final Gravity + HCLI Plan

**Status:** PLAN ONLY — after Final Frankenstein exists.  
**Seal companion:** `workspace/campaign/evidence/models/frankenstein/POST_FINAL_GRAVITY_HCLI_PLAN.json`  
**Upstream:** Final Frankenstein seal from Stage-2 (`STAGE2_KIMI_STREAMING_DISTILL_PLAN.md`)  
**Downstream:** Stage-3 Ramanujan (Odyssey / Q-gauntlet) uses Final as substrate; HCLI serves Terra + Luna (+ Sol on demand)

---

## 1. Purpose

Once **Final Frankenstein** is sealed locally as the DeepSeek-family child +
GLM + Kimi residual inheritance:

1. **HUMAN** seals / uploads Final to the cloud for durable off-box backup.  
2. Machine is reclaimed: re-download **clean** DeepSeek-V4-Flash and
   **Qwen3-Coder-30B-A3B**.  
3. **Gravity** both clean parents into Terra / Luna serve artifacts.  
4. Wire them into **HCLI** as the Terra + Luna tiers (Sol = Final / later
   Ramanujan, load-on-demand).  
5. Continue **kernel tuning**, prioritising the main open-source-model
   brokers.

This plan marks every step that is **manual (human)** vs **automatable
engineering**. Upload/seal to cloud are **never** automated in this programme.

---

## 2. Preconditions

| Gate | Requirement |
|------|-------------|
| Final candidate | Stage-2 archive sealed; progress cursor complete for scheduled Kimi blocks |
| Ablation | At least A/B/C comparison started; C not worse than B on math beyond owner tolerance |
| Storage | Local free space re-checked; Final archive either externalised or small enough to retain beside clean re-downloads |
| Identity pins | DSV4F revision `60d8d70770c6776ff598c94bb586a859a38244f1`; Qwen3-Coder-30B-A3B official Instruct revision **pin at download time** (record SHA; do not leave floating `main`) |
| Floor | Obey fusion 15 GiB free floor and BASELINES procure reserves for large pulls |

---

## 3. Phase plan

### Phase PF.0 — Local Final freeze (engineering)

1. Seal Final Frankenstein adapter archive + student body reference
   (content-addressed).  
2. Write a **Final identity receipt**: hashes of body, adapters, projections,
   corpus pins, schedule seal, capability probe summary.  
3. Stop further Stage-2 mutation of that archive (append-only receipts only).

**Exit:** `FINAL_FRANKENSTEIN_LOCALLY_SEALED` (local name; not cloud).

---

### Phase PF.1 — Cloud seal / upload (**HUMAN ONLY**)

| Step | Actor | Notes |
|------|-------|-------|
| Choose off-box destination | **Human** | Private object store / HF private repo / cold disk — owner decision |
| Upload Final archive + identity receipt | **Human** | Tools may assist with `rclone`/UI, but **authorisation and "this is the sealed Final" confirmation are human** |
| Verify remote checksums | **Human** (or human-triggered script) | Match local seal_sha256 / file inventory |
| Record remote URI + checksums in campaign evidence | eng after human OK | Do not invent a silent auto-upload agent |

**Explicit non-automation:**

- No cron upload.  
- No agent with cloud credentials in this plan.  
- No "seal" claim based only on local files without human cloud confirmation
  if the programme's durability requirement is off-box.

**Exit:** `FINAL_FRANKENSTEIN_CLOUD_SEALED` with human-signed or human-attested
receipt path.

---

### Phase PF.2 — Storage reclaim (engineering, after PF.1)

Only after cloud (or alternate durable) seal is attested:

1. Evict Kimi windows, Xet/HF donor caches, intermediate fit scratch.  
2. Optionally offload bulky intermediate Proto windows if Final supersedes
   them and Proto is also backed up.  
3. **Retain** Final archive (or a verified pointer) and any required student
   body until Gravity Terra is independently verified — same law as pipeline:
   do not evict the only verified source before a successor exists.  
4. Re-check free disk against BASELINES procure budget before re-download.

**Exit:** free space report sealed; floor preserved.

---

### Phase PF.3 — Re-download clean parents (engineering)

#### 3a. DeepSeek-V4-Flash (Terra source)

| Field | Value |
|------|-------|
| Repo | `deepseek-ai/DeepSeek-V4-Flash` |
| Pin | `60d8d70770c6776ff598c94bb586a859a38244f1` (or newer **owner-approved** pin with new admission) |
| Expected scale | ~159.6 GB blobs (prior admission); **re-measure** |
| Path | public Xet / presigned-range profile already frozen for V4 — reuse; do not re-tune from LAN folklore |
| Goal | **Clean** source for Gravity Terra — not the Frankenstein adapter tree |

#### 3b. Qwen3-Coder-30B-A3B (Luna source)

| Field | Value |
|------|-------|
| Repo | `Qwen/Qwen3-Coder-30B-A3B-Instruct` (confirm exact Hub id at fetch) |
| Card specs | **30.5B total / 3.3B activated** |
| Pin | record commit SHA at download; seal admission JSON |
| Goal | Clean coding MoE for Gravity Luna |

**Rules:**

- One heavy source window at a time if disk is tight.  
- Verify LFS/file hashes before any Gravity pack.  
- Do **not** "rehydrate" from partial Frankenstein scratch and call it clean.

**Exit:** two source admission receipts (DSV4F + Qwen) green.

---

### Phase PF.4 — Gravity pack Terra + Luna (engineering)

1. **Terra:** Gravity-pack clean DeepSeek-V4-Flash for serve.  
   - Respect native mixed precision (FP4 experts / FP8 rest).  
   - Do **not** blindly recompress to 1.5 BPW; follow
     `RESIDENT_MEMORY_AND_BPW_PLAN.md` (capability gates first).  
   - Measure resident RSS + active-bytes/token + capability suite → A in the
     four-way comparison.  
2. **Luna:** Gravity-pack Qwen3-Coder-30B-A3B.  
   - Optimise for always-resident coding loops.  
   - Measure RSS at intended context profiles.  
3. Optional later: Gravity **recompose** Final Frankenstein (C) for Sol serve
   — separate from clean Terra; do not overwrite Terra with Final without
   renaming.

**Exit:** `TERRA_GRAVITY_SEALED`, `LUNA_GRAVITY_SEALED` with load+generate
receipts.

---

### Phase PF.5 — Apply to HCLI as Terra + Luna (engineering)

Wire into the hide/HCLI tier map:

| HCLI tier | Artifact | Load policy |
|-----------|----------|-------------|
| **Luna** | Qwen Gravity | Default resident |
| **Terra** | DeepSeek-V4-Flash Gravity | On demand |
| **Sol** | Final Frankenstein (then Ramanujan) | On demand; may share body lineage with Terra but is **not** the plain Terra weights |

Work items:

1. Register model paths / manifests in HCLI config surfaces.  
2. Ensure tool-action / broker routing can address Luna for code iteration
   and Terra/Sol for heavier reasoning.  
3. Smoke: multi-turn tool call on Luna; one Terra load/unload cycle under
   memory supervisor.  
4. Record peak UMA during dual-load attempt; apply co-residency verdict from
   memory plan.

**Exit:** HCLI live suite receipts for Luna + Terra; Sol path declared even
if weight load is deferred.

---

### Phase PF.6 — Kernel tuning (ongoing engineering)

**Priority:** the most important kernels are the **main open-source-model
brokers** — the paths that actually move tokens for DeepSeek-family MoE and
Qwen-Coder MoE on Metal/UMA — not speculative research kernels.

Suggested order:

1. Broker correctness: load, route, expert gather, decode loop parity.  
2. Hot-expert cache + fault path (MoE reality on 96 GB).  
3. Attention / decode bandwidth (runtime accounting already flags attention
   as binding for DSV4F).  
4. HCLI tool-loop latency (scheduler, not only GEMM).  
5. Only then exotic quant/sub-bit work — and never without capability gates.

Dead levers in `workspace/docs/guides/dead_levers.md` stay dead unless new
evidence resurrects them.

**Exit:** never "done"; track in kernel/benchmark receipts. This phase
**overlaps** Stage-3 and product use.

---

### Phase PF.7 — Hand-off to Stage-3 Ramanujan (owner-gated)

- Final (Sol) becomes the Odyssey substrate candidate.  
- Substrate capability gate applies: unknown hash = REFUSED until probed.  
- Q-gauntlet and sandbox qualification proceed under Ramanujan governance
  (`RAMANUJAN_RESEARCH_AUTHORIZED` etc.) — **not** implied by Post-Final.  
- Additive law: formal training must not erase coding/agentic/tool-use.

---

## 4. Manual vs automated checklist

| Step | Manual (human) | Automated / eng |
|------|----------------|-----------------|
| Decide Final is "the" Final | **yes** | assist with probes |
| Cloud upload / off-box seal | **yes** | checksum tools only |
| Cloud checksum attestation | **yes** | script may print hashes |
| Evict donor caches | confirm | eng |
| Re-download DSV4F + Qwen | confirm pins | eng |
| Gravity pack | no | eng |
| HCLI wire-up | product accept | eng |
| Kernel tuning | prioritisation input | eng |
| Odyssey research authorise | **yes (owner)** | eng under fence |

---

## 5. Risk register

| Risk | Mitigation |
|------|------------|
| Upload incomplete; local reclaim deletes only copy | **Never reclaim before human-attested remote checksum** |
| "Clean" re-download drifts revision | Pin commit; refuse floating `main` |
| Gravity Terra confuses with Final | Separate paths and names; Terra has **no** Frankenstein adapters |
| Co-residency assumed | Memory plan: measure |
| Kernel work distracts from brokers | Priority list in PF.6 is binding for this plan |
| Stage-3 starts on unprobed substrate | Odyssey substrate capability gate |

---

## 6. Success criteria (Post-Final complete)

1. Final Frankenstein identity receipt exists locally **and** durable off-box
   attestation is human-recorded.  
2. Clean DSV4F + Qwen3-Coder-30B admitted and Gravity-packed.  
3. HCLI serves **Luna default**, **Terra on demand**, **Sol path declared**.  
4. Memory co-residency verdict sealed from measurement.  
5. Kernel broker backlog tracked; no claim of "kernels finished."  
6. Stage-3 may begin only under separate Ramanujan authority.

---

## 7. References

- Programme: `FRANKENSTEIN_PROGRAM.md`  
- Memory: `RESIDENT_MEMORY_AND_BPW_PLAN.md`  
- Stage-2: `STAGE2_KIMI_STREAMING_DISTILL_PLAN.md`  
- DSV admission / pipeline: `workspace/campaign/evidence/models/deepseek-v4/`  
- BASELINES: `workspace/docs/reference/BASELINES.md`  
- Cards:  
  - https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash  
  - https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct  
  - https://huggingface.co/moonshotai/Kimi-K3 (Final donor; not re-downloaded as interactive body here)
