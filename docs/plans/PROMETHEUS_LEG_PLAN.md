# PROMETHEUS LEG PLAN
## Capability-conditioned allocation as the next leg of Gravity — connected to the codebase as it actually stands, sequenced, gated, and droppable

```text
status:      DETERMINISTIC SPINE BUILT — L1 + L2-control run on real metadata.
             Capability campaign (L3+) still GATED on the two preconditions in §1.
opens when:  MODEL LADDER DONE  and  HAWKING SHIPPABLE   (see §1, gates the CAMPAIGN)
source doc:  Hawking Prometheus & Ramanujan · Canonical Master Plan · Revision 4
this doc:    the execution bridge from that plan to /Users/scammermike/Downloads/hawking
authority:   live git state, sealed receipts, and measured hardware override every
             number below. Where this doc and a sealed receipt disagree, the receipt wins.
```

> **THEORY → ACTION DELTA (2026-07-23).** The gate-independent half of the leg is built and
> tested. What is deterministic — allocation arithmetic, tiering, byte accounting, and the
> RandomPolicy control — does not need a served runtime, so it was taken from prose to
> running code ahead of the gate. What needs the runtime — capability evaluation (G1–G8),
> cartography coalition membership — stays gated and is marked so in every output. See §5.5.

This is not a restatement of Revision 4. Revision 4 is the science. This document is the
wiring diagram: for each of the six arms it names the files that already exist, the files
that do not, and the order to build the missing ones — starting from what is already on
disk, not from a blank tree.

The name is deliberate. Revision 4 calls the compression science **Prometheus**. This repo
already has **Gravity** — an executable law in `crates/hawking-seed-c/src/gravity.rs` that
crushes representation toward the sub-bit singularity uniformly. Prometheus is the same
crush made **selective**: it destroys the matter that does not serve a declared purpose and
protects the matter that does. That is why the consolidation in §7 is the real endpoint —
Prometheus is not a sibling of Gravity, it is Gravity with a capability-conditioned
allocation policy where the uniform policy used to be. The black hole does not stop
crushing; it starts choosing what to crush.

---

# 0. WHAT THIS DOCUMENT IS AND IS NOT

**It is** an arm-by-arm map from Revision 4 to real paths, plus a leg sequence (L0–L7) that
starts from the proven Gravity machinery and adds only the missing pieces, cheapest-citable
result first.

**It is not** a green light. Nothing here executes until the gate in §1 is true. It is not a
second master plan — Revision 4 stays canonical for hypotheses, gates, and falsifiers. When
a leg says "per Revision 4 §X," go read that section; this doc does not copy it.

**Provenance tags** carry over from Revision 4: `[M]` measured here, `[V]` vendor, `[D]`
derived, `[P]` projected, `[U]` unknown. An untagged number in this doc is a defect.

---

# 1. THE GATE — WHEN THIS LEG OPENS

Two preconditions, both stated by the user, both concrete against this repo:

### 1.1 MODEL LADDER DONE

The ladder is `HAWKING_LADDER_V3.json` / `.md` — rungs F0–F8. Current state `[M from ladder seal]`:

```text
F0 gpt-oss-120b        A8 closed
F1 Qwen3-235B-A22B      A7 live   (was ~layer 48/94 at last seal)
F2 Qwen3.5-397B-A17B    A2        (organ inventory + window plan created this session)
F3 MiniMax-M3           A2
F4 DeepSeek-V3.2        A2
F5 GLM-5.2              A2/Gen-B   (functional-partial, HELD — see §3)
F6 Kimi-K2.6            A2         (a full campaign already sealed: KIMI_K26_*)
F7 DeepSeek-V4-Pro      A2
F8 Kimi-K3              A0         (does not exist publicly — do not wait on it)
```

"Ladder done" is **not** "every rung streamed." It is: the parents Prometheus needs as
substrates have passed **Gate S0** (Revision 4 Arm I) at least on the two the MVP requires —
the control substrate (Qwen3-235B, F1) and the build-first substrate (Qwen3.5-397B, F2).
GLM-5.2 (F5) is the flagship *target*, not a precondition, exactly as Revision 4 §I.4 argues.

**Concrete check when this opens:** F1 and F2 each have a sealed source-format ledger, a
frozen `.gravity` decode-parity pass, and a baseline capability capture. F1's analog exists
in spirit already (`GLM52_SOURCE_FORMAT_LEDGER.json`, `GLM52_REFERENCE_PARITY.json` are the
template); reproduce that pair for F1 and F2.

### 1.2 HAWKING SHIPPABLE

The one measured blocker from memory: **the runtime does not yet serve a `.gravity`
artifact end to end** (the long-standing "serve gap — `qwen_dense.rs` doesn't read `.tq`"
lesson, now the `.gravity` equivalent). Shippable means:

```text
a real .gravity shard loads through crates/hawking-core and produces correct logits,
at a measured tokens/sec, on this box, without the original repo present.
```

This is **Gate S0.8** (cost-model calibration) from Revision 4 §I.5, and it is the single
`[U]` that gates every throughput and schedule number downstream. Until a `.gravity` file
decodes *and serves*, Prometheus has no runtime to prove Claim B on.

> **If either precondition is false when you return to this doc, stop. Finish the ladder
> and the serve path first. Prometheus on an unshippable runtime is a paper with no
> instrument behind it.**

---

# 2. CODEBASE REALITY MAP — THE CORE OF THIS DOCUMENT

Six arms, each: **what exists** (real paths), **what is a gap**, **the one-line verdict.**

## ARM I — SUBSTRATE (decoder + format)  ·  Revision 4 Arm I

**Exists — this arm is ~70% built:**

```text
.gravity container (= Paper 1 "Hawking format"), FROZEN v1:
  docs/gravity/GRAVITY_CONTAINER_SPEC.md          the ABI, self-describing shard
  docs/gravity/GRAVITY_CONTAINER_SCHEMA.json      header + tensor descriptor schema
  docs/gravity/GRAVITY_CONTAINER_TEST_VECTORS.json
  docs/gravity/GRAVITY_CONTAINER_COMPATIBILITY.json
  → the byte-honesty Revision 4 §II.6.1 demands is ALREADY LAW here: descriptor.bytes
    is never zero, protected tensors must carry payload. This is the exact Generation-A
    defect (billed protected tensors at 16 BPW, wrote them nowhere) burned into the spec.

decoder / runtime:
  crates/hawking-core/src/{engine.rs,attn.rs,moe.rs,model/,metal/,backend/,quant.rs,sample.rs}
  crates/hawking-core/src/mixed_quant_store.rs    per-tensor mixed representation
  crates/hawking-core/src/quant_tier_map.rs       per-layer bit tiers (T0–T3 PRECURSOR)

source ingestion + per-family adapters:
  crates/hawking-seed-c/src/providers/{adapters.rs,source_formats.rs(44KB),ir.rs(41KB),
    registry.rs,validation.rs,verify.rs,tokenizer_protocols.rs(43KB),source_decl.rs}
  crates/hawking-seed-c/src/{safetensors.rs,pack.rs,record.rs,state.rs}
  tools/condense/glm52_adapter.py(57KB)           MLA+DSA+IndexShare handled here
  tools/condense/deepseek_v4_adapter.py           FP4/FP8 family
  tools/condense/qwen35_moe_adapter.py            Gated-DeltaNet family (NEW, untracked)

S0-class gate machinery (per-parent, already the working pattern):
  tools/condense/corpus_integrity.py + GLM52_CORPUS_INTEGRITY.json     (S0.4/S0.5)
  GLM52_SOURCE_FORMAT_LEDGER.json, GLM52_ARCHITECTURE_CONTRACT.json     (S0.1/S0.2)
  GLM52_SOURCE_ADMISSION.json, GLM52_REFERENCE_PARITY.json             (S0.3)
  GLM52_STREAMING_SCHEDULE.json                                        (S0.4 streaming)
```

**Gap:**

```text
- Revision 4 §Appendix C wants per-attention-family decoder CRATES
  (gqa/ gated-deltanet/ mla/ dsa/ indexshare/). Reality: families live inside ir.rs +
  the Python adapters, not as isolated crates. This is a REFACTOR, not new science —
  and Revision 4's own build order (GQA→DeltaNet→MLA/DSA) is already how F1→F2→F5 ran.
- S0.8 cost calibration (measured tok/s at target bitrate through a served .gravity):
  [U]. This is the §1.2 shippability blocker. It is Arm I's last mile.
- S0.7 baseline capability at source precision uses Math Profile v1 (Arm II) — so the
  real baseline capture is blocked on Arm II gates existing. Today's baselines are
  cosine-fidelity, not capability gates.
```

**Verdict:** the format is done and frozen; the decoders work per-parent; the missing
piece is *serving* a `.gravity` at a measured rate. Finish that (L0), do not rebuild.

## ARM II — COMPRESSION / PROMETHEUS  ·  Revision 4 Arm II  ·  THE BIG BUILD

**Exists — the *engine* exists, the *conditioning* does not:**

```text
Gravity law (executable):
  crates/hawking-seed-c/src/gravity.rs       Rate(num/den), Ask, Evidence, Decision;
                                             sub-bit-first; Escape Receipt gate >1 BPW
  crates/hawking-seed-c/src/gravity_run.rs
  → this IS the thing Prometheus consolidates into (§7).

allocation + frontier machinery (Python, scattered):
  tools/condense/gravity_frontier_{g1..g4}*.py, gravity_frontier_controller.py
  tools/condense/gravity_global_allocator.py       byte budget across tensors
  tools/condense/glm52_allocation_probe.py         causal sensitivity probing (P3 precursor)
  tools/condense/glm52_functional_*.py             auction/cascade/gauntlet/student/roofline
  tools/condense/glm52_lowrank.py, glm52_moe_student*.py

the HONEST metric (the "capacity honest" instrument Revision 4 §2.4 demands):
  tools/condense/hawking_null_metric.py            centered cosine, fit-split null,
                                                   512-resample bootstrap lower bound
  HAWKING_NULL_CORRECTED_METRIC_CONTRACT.md/.json  raw cosine can't override the gate

tier precursor:
  crates/hawking-core/src/quant_tier_map.rs        per-layer bit tiers, but keyed on
                                                   residual-stat headroom, NOT capability
partial vocab hook:
  crates/hawking-core/src/profile.rs:80            RuntimeLevers.vocab_prune: Option<usize>
                                                   (a load-time lever, not a corpus-driven stage)
dead-expert census concept:
  tools/condense/eco_activation.py, glm52_allocation_probe.py
```

**Gap — this is the leg's real work:**

```text
- NO `crates/hawking-prometheus` crate. The Appendix-C pipeline (graph/ profiles/ probes/
  cartography/ tiering/ narrowing/ optimizer/ candidates/ evaluation/ reports/ adapters/)
  exists only as loose Python. P0–P10 (Rev4 §II.1) is not a pipeline, it is a folder.
- NO capability PROFILES. profiles/ holds runtime perf configs (qwen*.json) only. There is
  no math-v1 / general-v1 / uniform-v1 / random-v1 / mathnarrow-v1 ALLOCATION profile.
- NO RandomPolicy control — the single control that makes Claim A credible (Rev4 §II.7).
  Today's null is constant-mean (hawking_null_metric), which answers "beats a constant?"
  NOT "beats arbitrary non-uniform allocation at matched bytes?" Different question.
- NO Math Profile v1 gates G1–G8 (Rev4 §II.2). Especially G3 formalization + G8 proof-
  criticism — the discriminating gates. Current eval is fidelity, not capability.
- NO vocabulary-pruning STAGE (Rev4 §II.5.1, "largest cheap win"). Only the runtime lever.
- NO T0–T3 capability tier assignment (Rev4 §II.4). quant_tier_map is bit-headroom, not
  capability-conditioned.
- NO iso-memory frontier harness (Rev4 §II.6 / Paper 3).
- NO preregistrations/ dir (Rev4 §V.1).
```

**Verdict:** Gravity crushes; it does not yet crush *by capability*. Everything in this leg
is: (1) give the scattered Python a spine, (2) add the profile abstraction, (3) build the
RandomPolicy control, (4) build the G1–G8 gates. That is L1–L3.

## ARM III — INSTRUMENTATION / FORGE  ·  Revision 4 Arm III

**Exists — but it is a different Forge:**

```text
crates/hawking-seed-c/src/providers/forge.rs           (7KB)
tools/condense/forge_{actaware,controller_integration,f2_fixture,giant_adapters,
  pre_run_readiness}.py, gravity_forge.py, gravity_forge_run.py
reports/condense/gravity_forge/*.json
docs/plans/HAWKING_GRAVITY_FORGE.md
```

**Gap:** today's "Forge" is a **compression-frontier** forge (adapters, readiness, F2
residual). Revision 4's Forge is a **capability-attribution** layer: the five planes
(capability/policy/permission/evidence/resource), the matched-paraphrase protocol
(§III.3.1), the limit registry (§III.5), the false-refusal metric. **None of that exists.**
This arm is a reframe + new build, and Revision 4 §III.1 already downgraded it (open-weight
math models barely refuse; expected false-refusal <1% `[P]`). It stays droppable-partial.

**Verdict:** keep the name collision in mind — do not confuse the two Forges. The attribution
Forge is L6, and it is optional.

## ARM IV — RAMANUJAN  ·  Revision 4 Arm IV

**Exists:** nothing. Confirmed zero code — no `ramanujan/`, no ledger, no tribunal, no Lean,
no roles, no verifier lattice. `grep -rilE 'ramanujan|tribunal|conjecturer|mathlib'` over
`*.rs`/`*.py` returns empty.

**Gap:** the entire Appendix-C `ramanujan/` tree, plus the **P5 human-mathematician gate**
(§VI.6) — a *person*, not code, and Revision 4 calls it "the most under-resourced element in
the entire plan."

**Verdict:** fully greenfield, fully droppable (Rev4 says so), and gated behind the MVP
landing *and* an external expert being secured. This is L7 and it may never open.

## ARM V — SCIENCE  ·  Revision 4 Arm V

**Exists:** the sealed receipts *are* proto-evidence — `KIMI_K26_SCIENCE_PUBLICATION_MANIFEST.json`,
`KIMI_K26_SCIENTIFIC_LAW.json`, `GLM52_CORRECTED_SCIENTIFIC_LAW.md`, the negative-result
ledgers. The discipline (tag every number, publish negatives) is already the house culture.

**Gap:** no actual papers; no `preregistrations/`. Papers 1–12 (Rev4 §V.2) are unwritten.

**Verdict:** the evidence pipeline exists; the writing does not. Each leg below names the
paper it feeds so writing is never a separate project.

## ARM VI — PROGRAMME  ·  Revision 4 Arm VI

**Exists:** the goal-loop skill, the campaign-contract pattern, and a strong kill-criteria
culture (the memory index is one long list of "sealed NEGATIVE / DEAD / held" verdicts —
this project already stops cleanly).

**Gap:** the effort-hours schedule (§VI.2) and the declared weekly capacity (§VI.1, `[U]`)
are not written down as a living doc. Without the capacity number, every calendar estimate
is fiction, as Revision 4 states.

**Verdict:** the operating discipline is here; the honest schedule is a one-page write-up
away. Do it in L0.

---

# 3. EVIDENCE CORRECTION — CAPACITY HONESTY APPLIED TO THE PLAN ITSELF

Revision 4 leans on "**GLM compressed to 0.7 effective weight bits `[M]`**" (§I.3.3, §II.6)
to derisk Claim B. The latest sealed evidence in this repo **contests that premise**, and
the plan's own doctrine ("capacity honest," §2.4) requires flagging it before any schedule
is built on it:

```text
HAWKING_ASCENSION_STATUS.md, sealed 2026-07-23:

  weight-space verdict:  ZERO of 22 artifacts beat a constant. Corrected block-output
                         skill −1.21 to −5.38, including −1.77 at 2.0169 BPW (2× ceiling).
                         Every weight-space family closed: PQ, low-rank, shared-expert
                         basis, by-role hybrid, functional-block.

  functional escape:     glm52.functional.moe.v1 reaches centered 0.7309 at 0.0104 BPW
                         organ-local — but it is HELD. The GLM-5.2 residual stream is
                         EXPANSIVE at every magnitude (2.41× amplification at 0.77% error),
                         no contractive regime. Four functional layers drop router top-1
                         0.850 → 0.446. Decision: FUNCTIONAL_PARTIAL_ONLY, DO NOT STREAM.

  complete-model rate:   protected 2.11% alone costs 0.3378 BPW at source precision —
                         above the one-third rung before a single expert byte is stored.
```

**Consequences for this leg (do not skip):**

1. The "0.7 achieved" number is a **weight-space** claim that the corrected null-metric
   later **falsified**. Prometheus must not inherit it as a baseline. The real, honest
   baseline is: *no weight-space family beats a constant at any legal rate on GLM-5.2.*
2. Claim B ("frontier reasoner at sub-bit on this box") is therefore **not derisked** by
   GLM. It may still hold on Qwen3.5 (F2), whose residual stream has not been shown
   expansive. **This is a reason the build-first substrate leads, not the flagship** — and
   it strengthens Revision 4 §I.4's inverted build order.
3. The `[M]` tag on "0.7" in Revision 4 should read `[M, weight-space, later falsified by
   the null-corrected metric]`. Carry that correction into any paper.

This is not a reason to abandon Prometheus. It is the reason the RandomPolicy control (L2)
matters *more*: if even conditioned allocation cannot beat a constant on an expansive
stream, that is Claim A resolving **negative on GLM**, which is a publishable result (Rev4
K3) — and the iso-memory frontier (Paper 3) still stands regardless, because it is about
parameter-count-vs-bitrate, not about beating a null.

---

# 4. WHERE PROMETHEUS LIVES IN THE REPO (target layout)

Create these when L1 opens. Mirror Appendix C but *seed each dir from the existing Python*,
do not write from scratch:

```text
crates/hawking-prometheus/                 NEW crate; thin Rust orchestrator over the
  src/                                     proven Python stages (call out, don't reimplement)
    graph.rs        ← lift from providers/ir.rs tensor graph
    profiles.rs     ← NEW: the profile abstraction (§5, L1)
    probes.rs       ← wrap tools/condense/glm52_allocation_probe.py
    cartography.rs  ← wrap glm52_functional_* causal maps
    tiering.rs      ← promote quant_tier_map.rs to capability tiers
    narrowing.rs    ← NEW: vocab prune stage + dead-expert excision
    optimizer.rs    ← wrap gravity_global_allocator.py
    evaluation.rs   ← NEW: G1–G8 gate runner (§L3)
profiles/                                  NEW allocation profiles (today: perf configs only)
  general-v1.json  math-v1.json  mathnarrow-v1.json  uniform-v1.json  random-v1.json
preregistrations/                          NEW (Rev4 §V.1) — hash+commit BEFORE first artifact
gauntlets/prometheus/                      NEW — the flagship experiment arms (§L5)
```

**Ponytail note:** the Rust crate is an orchestrator, not a rewrite. The science already
runs in Python and is sealed. `hawking-prometheus` gives it a typed pipeline spine and a
single entry point; it shells to the proven `tools/condense/*.py` for the heavy math until
a stage proves worth porting. Do not port `glm52_functional_*` to Rust on spec — that is the
over-build this repo has burned on before.

---

# 5. EXECUTION LEGS

Each leg: **objective · connects-to (real files) · do · gate · deliverable · droppable.**
Sequenced cheapest-citable-first, which is Revision 4's MVP path (§VI.4), not arm-numeric
order. L0–L5 = the MVP. L6–L7 = upside.

---

## 5.5 — BUILT AHEAD OF THE GATE  (2026-07-23)

The deterministic, gate-independent core of L1 and L2 is implemented, tested, and sealed on
real GLM-5.2 metadata. This is the theory→action delta. It runs today because allocation
arithmetic needs no served model; only capability *evaluation* does, and that stays gated.

```text
BUILT                                                              MAPS TO
tools/prometheus/prometheus.py         P0 graph · P1 profile ·     L1 spine
                                       P2 census · P5 tiering ·
                                       P6 allocation · byte-decomp
tools/prometheus/test_prometheus.py    8 invariants, ALL PASS      L1/L2 proof
profiles/prometheus/{uniform,general,  the 5 allocation policies   L1 profiles +
  math,mathnarrow,random}-v1.json      (random = the control)      L2 control
preregistrations/PROM-001-*.{md,json}  iso-memory + RandomPolicy,  L2 / Rev4 §V.1
                                       hashed 01c38ca1...
reports/prometheus/GLM52_allocation_   real byte-accounted plans   L1 output
  plans.json                           on the sealed GLM ledger
```

**What the sealed run shows on real GLM-5.2 (753.3B logical weights, from the ledger):**

```text
profile         complete_bpw   physical_GB   excised_Bw   protected_GB
uniform-v1         0.4582         43.14          0.00         19.02
general-v1         0.4960         46.70          0.00         22.82
math-v1            0.4960         46.70          0.08         22.82
mathnarrow-v1      0.5927         29.96        348.91         22.82
random-v1          0.4960         46.70          0.08         22.82   (byte-matched to math)

coalition sweep, Math vs Random total bytes (control validity, every fraction):
  frac 0.02 → 36.368 GB == 36.368 GB   frac 0.10 → 63.918 GB == 63.918 GB
  frac 0.05 → 46.701 GB == 46.701 GB   frac 0.20 → 98.335 GB == 98.335 GB   all matched
```

**Three honest facts the machinery already surfaces, none of which needed a served model:**

1. **The control is valid at every knob.** Math and Random reconcile to identical total
   bytes across the whole coalition-fraction sweep. When L3 opens, any Math-beats-Random
   result cannot be "Math spent more bytes" — the pipeline asserts it cannot.
2. **At category granularity, conditioning barely moves the number** (uniform 0.458 vs math
   0.496 bpw), because routed experts are 97.5% of the weights and every profile keeps them
   instrumental. **The real Claim-A surface is *which experts* form the coalition** — which
   is exactly the GATED cartography piece. The machinery makes this concrete instead of
   hoped-for: it splits routed_expert into coalition + remainder and accepts a coalition
   size today, a membership list when P4 lands.
3. **mathnarrow is the excision-accounting trap, caught.** It excises 348.9B weights (physical
   drops to 29.96 GB) yet its complete_bpw *rises* to 0.593, because bitrate is over
   *retained* weights and it kept the expensive protected head. This is precisely why
   Revision 4 §II.5.3 forbids folding excision into bitrate — and the report keeps the two
   columns separate, so the trap is visible rather than hidden in a headline.

**What is NOT built (correctly gated, not skipped):** every capability number. `prometheus.py`
emits a `gated_stages` block on every run naming P3 (causal probe), P4 (coalition membership),
and P8 (G1–G8 evaluation) as requiring the served runtime + the L3 evaluator. No capability
claim is made by any artifact above. The `crates/hawking-prometheus` Rust orchestrator (§4) is
deliberately deferred — the plan's own rule: don't port to Rust until a stage proves worth it.

Run it:

```bash
python3 tools/prometheus/test_prometheus.py
python3 tools/prometheus/prometheus.py --profile math-v1 --coalition-fraction 0.05
```

---

## L0 — OPEN THE GATE  (Arm I finish + Arm VI honesty)

```text
objective    make the two §1 preconditions concretely true and measured
connects-to  crates/hawking-core (serve path), docs/gravity/GRAVITY_CONTAINER_SPEC.md,
             HAWKING_LADDER_V3.json, tools/condense/corpus_integrity.py
do           1. Serve one real .gravity shard through hawking-core to correct logits at a
                measured tok/s on this box, source repo absent. This is Gate S0.8. [M]
             2. Reproduce the F1 (Qwen3-235B) and F2 (Qwen3.5-397B) source-format ledger +
                decode-parity pair, using the GLM52_* receipts as the template.
             3. Write ONE page: declared sustained weekly capacity (Rev4 §VI.1 [U]→[M]) and
                the effort-hours schedule picked from §VI.2. Without it, every date is fiction.
gate         a .gravity file serves; F1+F2 pass S0.1–S0.4; capacity number exists
deliverable  measured tok/s + GB/token (converts Rev4's [P] throughput to [M]); Paper 1 data
droppable    NO. This is the instrument. Nothing downstream runs without it.
```

## L1 — THE PROMETHEUS SPINE  (Arm II structure)  ·  SPINE BUILT (§5.5)

```text
objective    turn scattered allocation Python into the P0–P10 pipeline + the profile abstraction
connects-to  gravity.rs, gravity_global_allocator.py, glm52_allocation_probe.py,
             glm52_functional_*.py, quant_tier_map.rs, profile.rs:80 (vocab_prune hook)
do           1. [DEFERRED] crates/hawking-prometheus Rust orchestrator — per the plan's own
                rule, not until a Python stage proves worth porting. Spine lives in
                tools/prometheus/prometheus.py today.
             2. [DONE] Profile schema (hawking.prometheus.profile.v1): purpose → capability
                gates → category→tier policy → exact-rational tier rates.
             3. [DONE] uniform-v1 + general-v1 written — AND math/mathnarrow/random too, keyed
                on canonical category so the tier map exists ahead of L3's gate calibration.
             4. [DONE structural / GATED refinement] P5 tiering reads category→tier from the
                profile today; keying the routed_expert coalition on PROBE sensitivity (not
                just structure) is the P4 cartography piece, still gated.
gate         [MET] `prometheus.py --profile uniform-v1` emits a byte-accounted candidate whose
             decomposition reconciles against physical bytes (asserted in test_prometheus.py;
             Rev4 §II.6.1, already law in the container spec)
deliverable  [DONE] reproducible pipeline + the uniform baseline arm on real GLM metadata
droppable    NO. Everything else in Arm II hangs off this spine.
```

## L2 — THE CONTROL THAT MAKES IT SCIENCE  (Arm II + Paper 3)  ·  CONTROL BUILT (§5.5)

```text
objective    build RandomPolicy + the iso-memory harness — the cheapest citable result
connects-to  tools/prometheus/prometheus.py, hawking_null_metric.py, gravity.rs (Rate)
do           1. [DONE] RandomPolicy: Prometheus machinery, coalition membership randomized at
                MATCHED bytes (Rev4 §II.7), seed 20260723 recorded in random-v1.json + PROM-001.
                Byte-match asserted across the whole coalition sweep. This answers "does
                conditioning beat arbitrary non-uniform?" — the constant-mean null cannot.
             2. [PARTIAL] iso-memory: the byte/bitrate arithmetic at a fixed budget runs today
                (coalition sweep 36→98 GB). The parameter-count × bitrate cross-substrate curve
                (397B@H15 vs 744B@H08) awaits F2's logical-weight ledger (S0-class, gated).
             3. [DONE] PRE-REGISTERED first: preregistrations/PROM-001-*, hashed 01c38ca1...,
                filed before any capability artifact exists.
gate         [PARTIAL] Random arm reconciles to Math bytes at every knob (MET, deterministic);
             the capability run on F2 end to end is GATED on S0.8 + L3.
deliverable  Paper 3 (iso-memory frontier) data — Rev4's highest-citation result — and the
             control every later Arm-II claim needs
droppable    NO. Without RandomPolicy, "our allocation works" is indistinguishable from
             "any non-uniform allocation works," and only the first is a paper.
```

## L3 — MATH PROFILE v1 GATES  (Arm II §II.2)

```text
objective    build G1–G8, especially the discriminating G3 (formalization) + G8 (proof-criticism)
connects-to  hawking-prometheus/evaluation.rs, hawking-serve (to run the artifact under load)
do           1. G1 competition-math retention, G2 derivation coherence, G4 tool orchestration,
                G5 trajectory stability, G6 notation fidelity, G7 counterexample.
             2. G3: Lean-4 encoding success of stated claims (needs a Lean-lite harness —
                the FIRST touch of anything Ramanujan-shaped, kept minimal here).
             3. G8: planted-error detection.
             4. Write math-v1 + mathnarrow-v1 profiles now that gates exist to define them.
gate         all eight gates run against a source-precision baseline (this is the real S0.7)
             and against one compressed candidate
deliverable  Math Profile v1; the capability baseline Arm I's S0.7 actually needs
droppable    partially — G1/G2/G5/G6 are cheap and high-value; G3/G8 are the expensive
             discriminators. If capacity is tight, ship G1/G2/G5/G6 and defer G3/G8, and
             SAY SO in the paper (a Math profile without G3 is weaker but still publishable).
```

## L4 — STRUCTURAL NARROWING  (Arm II §II.5)

```text
objective    the largest cheap byte win — vocab pruning — plus dead-expert excision
connects-to  narrowing.rs, profile.rs:80 (vocab_prune runtime lever already exists),
             eco_activation.py / glm52_allocation_probe.py (activation census),
             GRAVITY_CONTAINER_SPEC.md (excised_components must be reported separately)
do           1. Vocab prune: profile token frequency over a math corpus (papers, Lean, CAS),
                rank by frequency × notation-criticality, prune embedding+unembedding rows,
                remap tokenizer with unknown fallback. Gate on G4+G6 (Rev4 §II.5.1). The
                runtime already has the load-time hook; this adds the corpus-driven stage.
             2. Dead/cold expert census + excision, router renormalized over survivors.
                Measure degradation vs parameter fraction (the routing-shift effect is [U]
                and measuring it is itself a contribution, Rev4 §II.5.2).
             3. REPORTING RULE (already container law): excision bytes reported separately
                from bitrate, always. Never fold a removed vision tower into a headline BPW.
gate         vocab prune reclaims bytes with no G4/G6 regression; excision degradation measured
deliverable  Paper 6 (vocab pruning + structural narrowing); 2–3 GB [P] reclaimed on flagship
droppable    yes — vocab prune alone is worth doing; expert excision can defer if routing
             shift proves nasty (that nastiness is itself the finding)
```

## L5 — THE FLAGSHIP EXPERIMENT  (Arm II §II.7 + Papers 2,4,7)

```text
objective    the five-arm ladder that resolves Claim A
connects-to  everything above; gauntlets/prometheus/
do           At each sampled bitrate (dense 0.4–0.7, sparse H09/H15), on F2 then F5:
               <Base>-H<rate>-Uniform / -Random / -General / -Math / -MathNarrow
             Measure per arm: G1–G8, per-subdomain retention, trajectory divergence, tok/s,
             full byte decomposition. Pre-register the whole matrix first.
             Key comparisons: Math vs Uniform (headline), Math vs Random (the control that
             matters), MathNarrow vs Math (does elimination beat allocation — Rev4 H3).
gate         the five arms run at ≥3 bitrates on ≥1 substrate; comparisons are statistically
             separable OR cleanly not (either resolves Claim A)
deliverable  Papers 2 (conditioned compression), 4 (MoE cartography vs published Qwen3-235B
             coordinates — pre-register BEFORE re-reading arXiv:2505.14681), 7 (trajectory)
droppable    NO. This is why Arm II exists. But note §3: on GLM the honest prior is that this
             resolves NEGATIVE, and that is still a paper (Rev4 K3). Run it on F2 first, where
             the stream is not known-expansive.
```

**End of L5 = the MVP.** Papers 2,3,4,6,7 exist. Stop here and the leg is a complete research
result (Rev4 §VI.8). L6–L7 are upside only.

## L6 — THE ATTRIBUTION FORGE  (Arm III — reframe, droppable-partial)

```text
objective    the five-planes capability-attribution layer (NOT the compression Forge already here)
connects-to  providers/forge.rs (name only — different Forge), hawking-serve (turn events)
do           1. F0 DIAGNOSE: ~500-prompt math refusal probe + matched-paraphrase pairs
                (§III.3.1). Predicted false-refusal <1% [P] — if confirmed, F2-restore is
                unnecessary and this arm reduces to instrumentation.
             2. The five planes never impersonate each other (§III.2); the limit registry
                (§III.5); the metrics (hidden-intervention-rate→0, attribution→1).
             3. Paper 9 (refusal degradation under sub-bit) is FREE measurement on artifacts
                L5 already produced — do it even if you skip the rest of Arm III.
gate         attribution completeness 1.0 + hidden-intervention 0 across the probe suite
deliverable  Papers 8, 9
droppable    partially — the diagnostic layer is needed for Arm-IV honesty; the findings are
             optional. Paper 9 is nearly free, so at minimum take that.
```

## L7 — RAMANUJAN  (Arm IV — fully greenfield, fully droppable)

```text
objective    the verifier-grounded research economy (Rev4 Arm IV in full)
connects-to  NOTHING — zero existing code. Greenfield ramanujan/ tree (Appendix C).
prereq       (a) MVP (L5) landed AND resolved Claim A non-negatively, AND
             (b) a named external mathematician secured for the P5 gate (§VI.6) — a PERSON,
                 tracked as a hard gate, not a detail. Do not start L7 without one.
do           per Rev4 §IV in full: ledger, four-tier verifier lattice, nine roles, Tribunal,
             seven memory stores, branch economics, Lean+Mathlib pinning, adversarial
             formalization red-team (§IV.4.3, a HARD gate before trusting any Tier-2 claim).
gate         no claim reaches Tier 3 without a reproducible compiler artifact, demonstrated
             adversarially (Rev4 K5)
deliverable  Papers 10, 11 (compute-to-evidence curve), 12; Claim D is the far horizon
droppable    YES — Rev4 says so explicitly. Arms I/II/III/V stand complete without it.
```

---

# 6. DEPENDENCY + DROP GRAPH

```text
L0 ─┬─ L1 ─┬─ L2 ───────────────┐
    │      ├─ L3 ─┬─ L4 ─┐      │
    │      │      └──────┴─ L5 ─┴─ (MVP complete: Papers 2,3,4,6,7)
    │      │                     │
    │      │                     ├─ L6 (Arm III, droppable-partial: Papers 8,9)
    │      │                     └─ L7 (Arm IV, droppable + needs a mathematician)
non-droppable: L0, L1, L2, L5-core.   droppable: L3(G3/G8), L4(excision), L6, L7.
```

Single points of failure, in order: **L0 serve path** (no runtime = nothing), then
**L2 RandomPolicy** (no control = no credible Claim A). Guard those two hardest.

---

# 7. THE PROMETHEUS → GRAVITY CONSOLIDATION

The user's framing: *once Prometheus is confirmed working, consolidate it into Gravity —
it adds to the matter-destroying nature of a black hole.* Here is what that means concretely
against `crates/hawking-seed-c/src/gravity.rs`:

**Today** Gravity authorizes representation/BPW escalation uniformly. Its `Evidence` struct
weighs `representation_families_tried`, `doctor_bytes_in_budget`, and two traps (`f1_only`,
`scheduler_deferred`). It does not know *what capability* a tensor carries.

**After consolidation** Prometheus's capability map becomes an input to Gravity's `Decision`.
The allocator stops asking only "how few bits can this tensor survive?" and starts asking
"how few bits can this tensor survive *given its tier under the declared profile*?" — T0
(protected math core) resists budget pressure, T2 (vestigial) gets the floor, T3 gets
excised. The sub-bit crush becomes selective. Same law, one new axis.

**Do not consolidate until Prometheus is proven** (the user is right). The decision gate:

```text
consolidate INTO gravity.rs only when ALL hold:
  1. Claim A resolved (L5) — conditioned allocation measurably beats RandomPolicy at
     matched bytes on ≥1 substrate. If it does NOT, Prometheus stays a separate,
     publishable NEGATIVE and Gravity stays uniform (Rev4 K3).
  2. The profile → tier → allocation path is deterministic and reproducible (L1–L3 sealed).
  3. The consolidation is additive: existing uniform Gravity behavior is the profile
     `uniform-v1`, bit-identical to today. No existing .gravity artifact changes.
```

Until then Prometheus lives beside Gravity (`crates/hawking-prometheus` calling the Gravity
law), not inside it. Merging early would couple an unproven policy to the frozen law and
risk the exact Generation-A class of defect the container spec was written to prevent.

---

# 8. IMMEDIATE FIRST ACTIONS WHEN THIS OPENS

The deterministic spine actions are DONE ahead of the gate (§5.5). What remains needs the
runtime. Ordered, each one session or less (Rev4 §Appendix D adapted to this repo):

```text
DONE ahead of the gate (2026-07-23):
  ✓ Profile schema + all 5 profiles (profiles/prometheus/)
  ✓ P0–P6 pipeline + byte-decomposition, tested on real GLM metadata (tools/prometheus/)
  ✓ RandomPolicy control, byte-match asserted across the coalition sweep
  ✓ PROM-001 pre-registered and hashed BEFORE any capability artifact

REMAINING, gated on the runtime:
1. L0.1 — serve one .gravity shard to correct logits at a measured tok/s. Gate S0.8.
          This alone converts every [P] throughput number in Rev4 to [M]. THE blocker.
2. L0.2 — reproduce the F1+F2 source-format + decode-parity pair from the GLM52_* template,
          and seal F2's logical-weight ledger so the spine runs byte-exact on Qwen3.5 too.
3. L0.3 — write the one-page capacity + effort-hours schedule. Without it, no dates.
4. L3   — build the G1–G8 evaluator against a served artifact (the real S0.7 baseline). This
          is what un-gates every capability number the spine currently marks GATED.
5. Begin identifying the P5 mathematician NOW (long lead time) even though L7 is far off.
```

---

# 9. KILL CRITERIA (carried from Rev4 §VI.5, mapped to legs)

```text
K1  GLM-5.2 experts not BF16          → already resolved: F5 source is BF16 (ladder-verified).
K2  serve path can't hit S0.8         → L0 blocks; fix runtime before any Arm-II work.
K3  Math indistinguishable from Random → Claim A false; PUBLISH the negative (L2 makes it
                                         credible); do NOT build L7. §3 says expect this on GLM.
K4  cartography ≠ published Qwen coords → method not measuring what it claims; halt L5, revise.
K5  adversarial-formalization FN too high → Tier 2 untrustworthy; L7 reduces to Tier 1+3.
K6  weekly capacity <8h for two quarters → freeze at the current LEG boundary, write up, stop.
                                           Most likely to fire; stated first-class on purpose.
```

Every leg ends at a sealed artifact or a paper. Stopping after any leg leaves finished work,
not an abandoned system. That property is the point (Rev4 §1.3).

---

```text
END — PROMETHEUS LEG PLAN
saved:   docs/plans/PROMETHEUS_LEG_PLAN.md
built:   tools/prometheus/ (spine+control, 8/8 tests pass on real GLM metadata)
         profiles/prometheus/ (5 policies) · preregistrations/PROM-001 (hashed)
         reports/prometheus/GLM52_allocation_plans.json (sealed)
opens:   capability campaign when MODEL LADDER DONE and HAWKING SHIPPABLE (§1)
becomes: a capability-conditioned allocation axis inside gravity.rs, once proven (§7)
```
