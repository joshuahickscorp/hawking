# HAWKING ODYSSEY-I — CANONICAL PLAN
Single source of truth. Unifies the bible (`/Users/scammermike/Downloads/h_odyssey.md`, 100 §)
and steers S001-S004 (`~/.claude/ultragoal/steers/1b50b8c3-...md`, append-only history) into one
head. The autonomous loop EXECUTES from `ODYSSEY_STATE.json` / `ODYSSEY_COMPLETIONS.json` /
`ODYSSEY_POLICY.json` / `ODYSSEY_MANIFEST.json`; this file is the DOCTRINE + STATE the humans and
future steers read. A steer maps to a section below (doctrine / architecture / targets / state).
Last canonized: 2026-08-19.

---
## 0. GATES — THE GOAL (G-series)
The campaign is tracked as advancing G1 → G7. Simple sequential nomenclature; future steers
reference G-numbers (`applies-to: G2`). Internal machine ids (O000-O013 patients, mechanism-ids)
stay as implementation detail under these gates. Status: DONE / ACTIVE / PENDING.

- **G1 — AUTONOMY.** Replay-proof completion index + write-scope serialization + hardened harvest
  + launchd driver; first autonomous seal→retire proven. **DONE** (O005 retired unattended).
- **G2 — SUPER-DETACHMENT MACHINE.** memgate (multi-model, swap≤30GiB) + candidate-search-as-data
  + predeclared manifest + cost-model/detachment-metrics + novelty engine + anti-complacency
  retirement gate, all wired into the loop. **DONE** (6-lane fan-out + integration merged; multi-model
  admission proven — 2 concurrent disjoint-patient lanes under swap≤30; driver re-enabled on full machine).
- **G3 — CYCLE 1 SEALED.** Each of O001/O003/O004/O005/O006: anchor → aggressive probe → frontier
  localized → NX attempt → packet-shows-frontier → retire. **ACTIVE** (O005 done; rest running).
- **G4 — COHORT 2.** Acquire the next storage-fitting cohort (O007+ by info-value/GB under disk+swap),
  overlapping download with compute. **PENDING** (on Cycle-1 empty).
- **G5 — ORCHESTRATOR v2.** Distill the Cycle-1 trace into deterministic code (swgrok); prove
  Claude-free-patient, Grok-only-novelty, Opus=0 patient + cohort, anti-complacency/conventionality/
  localization/anti-dogma tests. **PENDING** (post-Cycle-1).
- **G6 — DSV4F REPLAY (O011).** Mandatory compiler-evolution control vs old Gravity. **PENDING.**
- **G7 — ODYSSEY-I SUCCESS.** Bible §76: ladder substantially complete · packets · real NXs · rulebase
  · transfer matrix · negatives · machine genome improved · streamed-large science · Opus cost ≪ naive.
  Hands to Odyssey-II (DSV4F deep replay → super-kernel → genesis tournament). **PENDING.**

---
## 1. MISSION (prime directive)
Odyssey-I perfects **HAWKING**, not the models. It is an information + compiler-learning campaign,
NOT a quantization benchmark. Every patient is a teacher; every patient must leave the compiler more
capable. Product = **Gravity / Doctor / Tabula / NX / NVM-HIDE / Machine-Genome vNEXT + rulebase +
transfer matrix + negative-science corpus + cost model**. The end state is **Hawking autonomously
running the Odyssey — not Claude** (S004). Odyssey-II replays the important architectures (DSV4F
mandatory) with the matured compiler; then super-kernel; then genesis tournament over NX objects.

---
## 2. OPERATING DOCTRINE (unified)
**Cheapest reliable layer** (bible §10): deterministic-software → Grok → local-model → Opus. Always
ask "can this decision be lowered one level? if yes, lower it" (S004 §51).
**Ultra-detachment** (S004 §51-74): the normal loop runs with NO Claude/Opus. Every repeated
scientific decision (threshold/rule/state-transition/cost-fn/Pareto/search-policy/predicate) becomes
CODE. Grok = default novelty engine (NOVELTY_PACKET → fanout; never arithmetic a script can do).
Local models = cheap workers (never self-certify; pass through tests/Doctor/physics/receipts). **Opus
= interrupt handler, default ASLEEP** — wake ONLY for: irreconcilable high-quality contradiction /
unprecedented architecture / suspected systemic Goodhart / project-wide universal rule / ontology
change / strategic model-resource fork. NEVER for candidate-failed, BPW-missed, patient-finished,
Doctor-pass/fail, frontier-moved. Same escalation class twice → encode to software; three times → defect.
**Anti-complacency** (S004 §0-50): FIRST-VALID ≠ BEST-USEFUL ≠ FRONTIER. A Doctor-valid conventional
q3/q4 is a CONVENTIONAL_ANCHOR, never automatically a frontier. Push the ladder: anchor → aggressive
quant (q2/ternary/mixed/per-expert alloc) → structural (base+correction, tiers, shared/entropy metadata)
→ active NX. On failure, LOCALIZE (which organ) and generate a targeted repair — never global retreat
to q4 (last resort). Search allocation over organ/tensor/layer/expert/family/channel/tier/route, and
BETWEEN integer bit-widths. Retirement REFUSES a patient on conventional anchor alone (see §4 gate).
**Determinism first** (bible §13): tools MEASURE, models interpret. Never ask a model a fact a script
can compute (params/tensors/experts/entropy/bytes/BPW/TPS/DRAM/hashes).
**Protected GPU / clean-room** (bible §14, user 2026-08-19): contaminated TPS timing is VOID.
Clean-room (exclusive) ONLY for protected TPS/TOKEN_NS; ordinary SPECIMEN experiments run concurrently.
**Memory** (user 2026-08-19, `ODYSSEY_POLICY.detachment.memory`): admit MULTIPLE concurrent model
experiments while **projected swap ≤ 30 GiB** (96 GiB box, ~12 reserve); `tools/odyssey_memgate.py`
enforces. Serialize only same-write_set lanes + clean-room lanes.
**Disk is a cache / cohort cycling** (bible §16, S003): active window = current cohort + compiler
cache. A CYCLE = one storage-resident cohort of patients; seal → retire bulk (after receipts/hashes/
provenance/best-NX/rules preserved) → acquire next fitting cohort. Bulk moves, science accumulates.
**Receipt epistemology** (bible §18): label VERIFIED/MEASURED/DERIVED/INFERRED/HYPOTHESIS/SPECULATIVE/
REFUTED/STALE/UNKNOWN. Receipts are machine memory; chat is dev history.
**False-win gates** (bible §60, S004 §29-34): no fake density (complete bytes incl scales/metadata/
correction/alignment/state; nominal_bits vs complete_bpw), no fake active density (measure bytes/token
+ DRAM), no fake TPS (protected lane), anti-Goodhart BPW/Doctor(held-out)/info-count(novelty not count).

---
## 3. AUTONOMY ARCHITECTURE (the machine)
Controller `tools/odyssey_ctl.py` — subcommands status/queue/value/harvest/packet/admit/completions/
run/cycle/retire/acquire-next. Driver `tools/odyssey_driver.sh` under launchd `com.hawking.odyssey`
(15-min `cycle --go`), auto-commits DATA (code → REVIEW_QUEUE).
- **Replay-proof scheduler**: `ODYSSEY_COMPLETIONS.json` is the canonical workflow-state (separate from
  packet=patient, receipts=evidence). Scheduler queries it; never packet markers. Harvest writes
  completions dynamically from the launched obligation's mechanism.
- **Write-scope serialization**: obligations declare write_set; intersecting → serialize, disjoint →
  parallel. Data-producing lanes (external/route/sensitivity/gravity/nx/transfer) complete from
  RECEIPTS and DROP incidental code tweaks (harvest classifies by template, not diff-files).
- **Retirement gate (anti-complacency)**: required set per class + REFUSE if conventional-anchor-only
  and no nonconventional probe attempted and cheap mechanisms remain (exception = LOW_INFORMATION_VALUE
  receipt). Retire → seal packet + `_PATIENT_SEAL.json` + patient-sealed completion + mark reclaimable.
- **Storage-cohort cycling**: `acquire-next` downloads the next manifest patient (disk-gated, swap-safe,
  overlap acquisition with compute) when the ready frontier empties.
- **Modules (super-detachment fan-out, 2026-08-19)**: `odyssey_memgate.py` (multi-model swap≤30 admit),
  `odyssey_candgen.py` + `candidate_families.json` (aggressive search-space AS DATA, deterministic
  gen/prune, no LLM per candidate), `ODYSSEY_MANIFEST.json` (predeclared O000-O013 sources+targets),
  `odyssey_costmodel.py` (compile-economics + frontier-depth + detachment metrics → `ODYSSEY_COST_MODEL.json`).
- **Runner** `tools/odyssey_patient_runner.py` — mlx SPECIMEN external science (external/route/sensitivity/
  gravity `--gravity <spec>`/nx-gather/ssm) for archs the Rust `load_engine` can't run (Qwen3-MoE/Falcon/
  Gemma/Mamba). Native NX = new Rust primitive (bible §53, deferred). Census `tools/odyssey_census.py`.
- **Governors reused**: worker_gate, machine_state.clean_box_ok, reclaim_safe.sh, doctor_seal.seal.

---
## 4. PATIENT LADDER + COHORTS (bible §34-48, manifest = authority)
O000 gemma-3-1b (tiny dense lab, GATED) · O001 Falcon-H1-7B (hybrid attn+Mamba) · O002 gemma-3-4b
(MM dense, GATED) · O003 Kimi-VL-A3B (MM MoE) · O004 Mistral-Small-3.1-24B (dense MM control) ·
O005 Qwen3-30B-A3B (small-active MoE) · O006 Qwen3-VL-30B-A3B (MM MoE, transfer sibling of O005) ·
O007 Kimi-Linear-48B-A3B (linear-attn MoE) · O008 Jamba-1.5-Mini (Mamba+attn MoE) · O009 Qwen2.5-72B
(large dense) · O010 GLM-4.5-Air (106B/12B MoE) · **O011 DSV4F (MANDATORY legacy replay control)** ·
O012 GLM-4.5 (355B/32B very-large) · O013 Kimi-K3 (2.8T streamed capstone, native-QAT).
Cohorts: **Cycle 1 = O001/O003/O004/O005/O006** (on-disk). Numbers = scientific ladder; scheduler =
cheapest storage-fitting order. Gemma (O000/O002) BLOCKED on HF license — do not stall; backfill later.

---
## 5. TARGETS (soft pressure, arch-specific; anti-complacency not pass-thresholds)
Zones: ≤3 BPW reachable-or-explained · ≤2.5 pressure · <2 aggressive · 1-1.5 structural/correction/tier
· <1 frontier-with-full-accounting. Arch objective: dense=stored-density, MoE=active-bytes/token,
hybrid=state-residency, MM=modality organs, MTP=tokens/traversal, streamed=residency+IO. Per-patient
targets live in `ODYSSEY_MANIFEST.json`. Odyssey-I minimum per patient (bible §24): provenance · Doctor
baseline · arch census · physical baseline · route/state map · sensitivity map · ≥1 Gravity build · ≥1
aggressive/nonconventional probe · ≥1 NX/exec attempt · moderate kernel pass · packet · transfer class · seal.

---
## 6. COMPILER FRONTIER (durable outputs)
`GRAVITY_RULEBASE.json` (rules: conditions/supporting/negative patients/confidence/reopen_if/rationale) ·
`TRANSFER_MATRIX.json` (rule × patient: UNCHANGED/RETUNED/ARCH-SPECIFIC/PATIENT-SPECIFIC/FAILED/HARMFUL/
NOT_TESTED) · `NEGATIVE_SCIENCE.json` (kills + premise + reopen_if; predicates over blacklists) ·
Machine Genome = M3 Ultra 96GB (ambient) · `ODYSSEY_COST_MODEL.json` (post-Cycle-1). A rule is universal
only after surviving sufficiently different patients — never "worked twice on two Qwens".

---
## 7. STATE / PROGRESS ARCHIVE (2026-08-19 session)
**Built + committed on branch `odyssey-i`** (main untouched): census tool; patient-runner (MoE/dense/
hybrid + gravity/nx/sensitivity modes); controller (replay-proof completion index, write-scope
serialization, retirement, storage-aware acquisition, gravity/nx templates); hardened harvest;
launchd driver (fixed EX_CONFIG; PATH; data-auto-commit); policy `ODYSSEY_POLICY.json`; O005/O001/O003/
O004/O006 packets; seeds rulebase/transfer/negative.
**Autonomy PROVEN live**: launchd driver fires unattended → harvest→complete→retire→launch. **First
autonomous patient transition: O005 SEALED+RETIRED** (external/route/sensitivity/gravity-moe/nx-gather-moe).
**Science (all SPECIMEN, honestly labeled)**: O005 census 30.53B/3.35B-active(11%), experts=95% of body;
route entropy 6.16/7.0, **0 cold experts**, MoE-universal sparse path holds vs abliterated; first Gravity
q3-experts 4.03 stored/4.23 active bpw, battery 10/12, Δ0 → CONVENTIONAL_ANCHOR (reclassified per S004);
NX-gather selected/full 0.0625 = **16× active-byte lever**. O001 Falcon: hybrid, H2 REFUTED (KV beats
SSM state past ~1557 tok), sensitivity done. O003 Kimi-VL: 64-exp top-6, entropy 5.46/6.0, 0 cold,
battery 12/12. O006 Qwen3-VL: text body ≈ O005, transfer 4/9 cells. **Cross-patient: 0 cold experts on
O005/O003/O006 → cold-expert compression NOT universal; sparse-active-gather is the lever.**
**In flight (2026-08-19 fan-out)**: anti-complacency-v1 (aggressive gravity + retirement gate) + 5
super-detachment module lanes (memgate/multi-model swap≤30, candgen search-as-data, manifest, costmodel,
novelty) + O003 gravity / O004 external science. Driver PAUSED for the integration.
**Integration learnings (durable)**: harvest classify-by-template (drop incidental runner tweaks);
launchd EX_CONFIG = plist WorkingDirectory/dup-StdOut keys (not TCC); launchd needs PATH incl ~/.grok/bin;
worker_gate over-strict on stale swap → memgate + ODYSSEY_HEADROOM_ADMIT; serialize same-file lanes.

---
## 8. NEXT (forward plan)
1. **Integration pass** (when fan-out lands): wire memgate (multi-model swap≤30 admission, replacing
   one-at-a-time) + candgen + manifest + costmodel + novelty into `ctl.py`; fix data-lane write_sets so
   model experiments PARALLELIZE; re-enable driver.
2. **Run Cycle 1 to completion** aggressively: each remaining patient gets anchor → aggressive probe →
   localize → (structural / base+correction / tiers) → NX attempt → packet-shows-frontier → retire.
3. **Cohort transition**: on Cycle-1 empty, acquire next fitting cohort (O007/O008/O009/O010 by
   info-value/GB under disk+swap), overlapping download with compute.
4. **Post-Cycle-1 Orchestrator v2 (swgrok, S004 §70)**: distill the Cycle-1 trace into deterministic
   code — "what did Opus/Claude do that can become code/data/rule/test/state-transition/grok-policy?"
   Prove: Claude-free-patient, Grok-only-novelty, Opus=0-patient, anti-complacency, full-cohort tests.
5. **O011 DSV4F** mandatory replay (compiler-evolution control). Then O012/O013 streamed science.

---
## 9. STEER INDEX (append-only history in STEERS.md)
S001 (launch authorization: replay-proof P0, enable driver) · S002 (continuous patient cycling, bounded
info-budget, prove seal→retire→next) · S003 (storage-window cohorts + compile-economics/cost-model) ·
S004 (**anti-complacency** + **ultra-detachment** — canonized into §2 doctrine + `ODYSSEY_POLICY.json`).
Future steers: append to STEERS.md, then fold the constraint into the matching section here + the
machine-readable policy/manifest. Opus stays the synthesis authority for this file; the machine executes it.
