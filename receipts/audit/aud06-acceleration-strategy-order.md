# aud06 — acceleration strategy vs current roadmap order

**Discovery audit. No H-ROADMAP rewrite. No implementation.**

Canonical decree: `/Users/scammermike/Downloads/H-ROADMAP.md` (not a git blob), freeze 2026-08-27, mtime 2026-08-29, sha256 `d43a6b07ab9590bc11c265bfe8a1466131cce291b0622c076370a01d811328e4`. HEAD `04193ccbc`. Machine-readable twin: `receipts/audit/aud06-acceleration-strategy-order.json`.

This pass did not run Metal, ANE, or any board. Classifications are SOURCE_INSPECTION / STATIC_VERIFICATION of source, call sites, and receipts already on disk. Nothing here is MEASURED by this auditor.

## Answer

The CRISPR spec **agrees** with the acceleration strategy in the places that were written as law: Apple/Metal is the first physical software school (§11.3, I-D), U50DD is textbook FPGA #1 (§15, §17.2), high-end FPGA / custom board / ASIC wait for U50 tuition (§25, I.23). Five eras, three Odysseys, no Era VI, FPGA inside Accelerator/Fusion, Theia not a civilization, 0.7% untouched.

It **contradicts** that strategy in four live orders, and those are not to be preserved merely because they were frozen:

1. **Appendix I still puts commodity eGPU Fusion in Phase II, before FPGA Alpha in Phase V.** That is the unreordered Physical Manifestation roadmap. §17.2 and I.23 in the same file say software → U50 FPGA → eGPU.
2. **Era IV numbers Fusion before HMF, and DGX before eGPU**, while Fusion’s own genes already include eGPU/DGX, and the Fusion acceptance gate is blocked on HMF.
3. **CONTEXT + PREFILL is not a first-class school.** §11.1 never names it. Overnight and Appendix P never schedule it. On this HEAD, Qwen38 `generate_greedy` still decode-steps every prompt token.
4. **ANE / Forbidden Fruit is not a named independent school.** The freeze never writes “Forbidden Fruit”. ANE is SING-F71–F78 inside Flash closure. The lab exists in `hcli` and has a PASSED receipt the freeze does not know the name of.

§32 claims the old `software→eGPU→nodes→FPGA` progression was “retained and **reordered** under five eras”. Appendix I shows the reordering was claimed, not done.

## Strategy order (the test, not a constitution change)

1. Apple GPU / Metal — first physical software school
2. Context + prefill — first-class runtime/physical optimization
3. ANE / Forbidden Fruit — independent clean-room machine-discovery lane
4. U50DD — first programmable hardware school
5. HMF / HGVAS / Fusion — software-defined object/memory truth
6. eGPU / DGX Spark / additional Macs — later alien-machine schools
7. High-end FPGA fabric — only after U50 tuition
8. Custom board — only if integration is justified
9. ASIC — only after a recurring stable primitive survives

## Where the freeze matches

| Strategy | Freeze location | Lines |
|---|---|---|
| Metal first | §11.3 Apple / ACORE / Kernel Forge; I-D genes | 1533–1547, 506–522 |
| U50 textbook FPGA #1 | §15 header; §17.2 priority | 2422–2426, 2890–2899 |
| High-end FPGA after U50 | §25; Appendix I Phase XII | 3620–3634, 8783–8787 |
| Custom board / ASIC last | §25.3; I.23; Phases XVI, XXI–XXII | 3628–3668, 8811–8880 |
| Five eras, no Era VI | constitution + do-not-do | 8, 4003 |
| Three Odysseys, no IV | §6 | 997–1014 |
| FPGA inside Accelerator/Fusion | I-D gene “pre-board FPGA school”; IV-A gene FPGA | 522, 777 |
| Theia is a bounty model | §19; not in the 25 programs | 2972–2974 |
| 0.7% | header + accounting | 5, 68 |

Section 34’s V3 gene index still lists Stage O eGPU before Stage Q HMF. The freeze itself says those headings are **lineage labels, not active competing plans** (4058–4060). They are not counted as live contradictions. Appendix I is counted, because it is a live appendix.

## Contradictions (do not preserve because frozen)

### C-APP-I-EGPU-BEFORE-FPGA — HIGH — STALE_ROADMAP_TEXT

Appendix I Phases I–XXII (8706–8738): Phase II Commodity eGPU Fusion, Phase III additional nodes, Phase V FPGA Alpha.

§17.2 (2890–2899): software Accelerator → **U50 FPGA school** → eGPU → DGX.

I.23 in the **same appendix** (8860–8880): SOFTWARE → FPGA SIMULATION → TEXTBOOK FPGA → … → ASIC. No eGPU.

This is the loudest ordering defect. Strategy: U50 first programmable hardware school; eGPU later. The freeze contains both orders. The later, local law (§17.2 / I.23) matches the strategy. Appendix I phases match the absorbed 2026-08 Physical Manifestation doc and were not reordered.

### C-SEC32-REORDER-CLAIM-FALSE — HIGH — STALE_ROADMAP_TEXT

Line 4032: Physical Manifestation progression `software→eGPU→nodes→FPGA→…` “retained and reordered under five eras”.

Appendix I (8703–8717) is still that progression. The sentence is a claim about an edit that did not happen.

### C-ERA-IV-FUSION-BEFORE-HMF — HIGH — STALE_ROADMAP_TEXT

Era IV (771–817): IV-A Fusion (genes: Apple, FPGA, DGX, eGPU, additional Macs) then IV-B HMF/HGVAS.

Section 16 (2801–2864) is HMF object/memory truth, then section 17 is Fusion/eGPU/DGX.

`receipts/acceptance/FUSION_FIRST_HETEROGENEOUS_EXECUTABLE.json` verdict **BLOCKED** on `HMF_DEVICE_VISIBLE_TRUST` / `NO_SECOND_PHYSICAL_DOMAIN`. Invoked symbol: `tools.accelerator.fusion_planner.topology_apple_alone`. Wake conditions include `HMF_PRESENT` and one of U50/eGPU/DGX.

Strategy groups HMF/HGVAS/Fusion as memory truth **before** alien-machine schools. Era numbering puts Fusion first, stuffs eGPU/DGX into Fusion’s genes, then repeats DGX and eGPU as IV-C/IV-D. The acceptance producer already treats HMF as a Fusion prerequisite. Era IV-A/IV-B is the stale order. This is a label swap inside Era IV, not a sixth era.

### C-ERA-IV-DGX-BEFORE-EGPU — MEDIUM — STALE_ROADMAP_TEXT

§17.2: eGPU then DGX. Era list: IV-C DGX Spark (820–838), IV-D eGPU (841–861). Smaller than eGPU-before-U50, still a live internal inversion.

### C-CONTEXT-PREFILL-NOT-FIRST-CLASS — HIGH — STALE_ROADMAP_TEXT

§11.1 physical optimization order (1505–1518) is eliminate information/bytes/FLOPs/dispatches/…/saturate. Prefill is not a step. Runtime context is not a step.

Prefill appears once in the accelerator school, as bullet “Phase-aware prefill vs decode backends” (1544). §35 still asks “what if neural acceleration should own prefill” (5049). Overnight 2.3 (269–287) and Appendix P tonight (9552–9559) schedule Flash, Qwen27, ModelLake, FPGA pre-board — not prefill.

On this HEAD the Qwen38 resident path is still:

```7433:7468:crates/hawking-core/src/model/qwen38_hybrid_decode.rs
        // Every prompt token is stepped once, followed by at most
        // `max_new_tokens` sampled ids.
        ...
        for (i, &token) in prompt.iter().enumerate() {
            let (sampled, timing) = session.step(token)?;
```

`Engine::prefill_slot` default (engine.rs:481) returns `Unimplemented`. Real `prefill_slot` impls exist for QwenDense (qwen_dense.rs:2997) and RWKV7 (rwkv7.rs:1805) — CALLABLE on those families, not INTEGRATED Qwen38 prefill.

AgentOS `ContextCompiler::compile` **is** called (hide-kernel/src/lib.rs:87, hide-backend/src/connectors.rs:550). That is prompt packing (“CONTEXT IS A CACHE”), not KV/prefill physical optimization. Counting it as the strategy’s CONTEXT+PREFILL school would repeat the import-inflation failure mode.

`docs/PREFILL.md` is dated 2026-09-02, after the freeze, and describes a GEMM prefill branch whose files are **not on this HEAD**. A document named PREFILL is not a finished school.

### C-ANE-FF-NOT-A-SCHOOL — HIGH — STALE_ROADMAP_TEXT

Strategy: ANE / Forbidden Fruit is an independent clean-room machine-discovery lane.

H-ROADMAP never contains the string “Forbidden Fruit”. ANE is “Measurement, ANE and proof closure (SING-F61..SING-F80)” (1899–1950), with an explicit rule that the ANE lane cannot block SING-F01..F42. C41–C46 (2169–2185) are “non-blocking device lanes”. Overnight and Appendix P omit ANE.

The lab exists. `hcli/tool_registry.py:1573` calls `forbidden_fruit.run_forbidden_fruit_lab`. `receipts/headless/HCLI_FORBIDDEN_FRUIT_LAB.json` status PASSED, `ane_placement_observed_this_run: false` (CPU preferred). That is TESTED public-API placement, not an ANE-resident organ, and not a school in the freeze.

`ANEProvider(...)` is constructed only in `hcli/test_ane_provider.py`. A test constructor is not a production caller. Atlas receipt status is `ATLAS_SCAFFOLD_COMPILE_BOUNDARY`.

### C-PB03-SIMFPGA-ABSENT — MEDIUM — ABSENT

§15.15 “Pre-board campaign — executable now” lists PB-03 SimFPGAProvider (2758). Seed bank FPGA-002 is the same name (3950). `git grep SimFPGAProvider HEAD` is empty. `MockFPGAProvider` exists (`fpga_preboard.py:192`). `FPGAProvider.execute` (line 187) raises “no physical backend selected”. The freeze presents a missing symbol as a live next step.

## Capability table (call site or blocker, no inflation)

| ID | Claim | Class | Call site or blocker |
|---|---|---|---|
| CAP-METAL-PRIMARY-BACKEND | Metal is the wired primary backend | INTEGRATED | `MetalBackend::new` qwen_dense.rs:6119; `session.step` qwen38_hybrid_decode.rs:7461 |
| CAP-PREFILL-QWEN38-GENERATE | Qwen38 greedy does GEMM prefill | ABSENT | generate_greedy still `session.step`s prompt tokens; hybrid_prefill.rs not in HEAD; default `prefill_slot` Unimplemented |
| CAP-PREFILL-SLOT-DENSE-RWKV | prefill_slot exists for QwenDense/RWKV7 | CALLABLE | hawking-serve/src/lib.rs:875 `engine.prefill_slot`; impl qwen_dense.rs:2997 |
| CAP-CONTEXT-COMPILER-AGENTOS | ContextCompiler packs HCLI context | CALLABLE | hide-kernel/src/lib.rs:87, hide-backend connectors.rs:550 — **not** runtime prefill |
| CAP-FORBIDDEN-FRUIT-LAB | Forbidden Fruit lab is reachable | TESTED | tool_registry.py:1573 `run_forbidden_fruit_lab`; receipt PASSED, no ANE placement |
| CAP-ANE-ATLAS | Measured ANE MLProgram atlas | SCAFFOLDED | `ANEProvider(` only in test_ane_provider.py:10; atlas `ATLAS_SCAFFOLD_COMPILE_BOUNDARY` |
| CAP-U50DD-PHYSICAL | U50DD physical execution | BLOCKED_HARDWARE | FPGAProvider.execute raises; preboard receipt physical_board ABSENT; ROADMAP_STATE U50 NOT_STARTED |
| CAP-FPGA-PREBOARD-SIM | MockFPGAProvider simulation | CALLABLE | agentos_cli.py:1019 `run_fpga_preboard` — SIMULATED, not a board |
| CAP-SIMFPGA-PROVIDER | SimFPGAProvider as PB-03 | ABSENT | no symbol in HEAD |
| CAP-HMF-DEVICE-VISIBLE | HMF device-visible domain | BLOCKED_HARDWARE | acceptance HMF_DEVICE_VISIBLE_TRUST BLOCKED on HMF_PRESENT |
| CAP-FUSION-HETEROGENEOUS-EXECUTABLE | Two-domain Fusion executable | BLOCKED_HARDWARE | acceptance BLOCKED NO_SECOND_PHYSICAL_DOMAIN; nodes=['APPLE'] |
| CAP-EGPU | eGPU enclosure present | BLOCKED_HARDWARE | fusion_planner eGPU link `physical=False` SIMULATED; hardware_doctor no enclosure |
| CAP-DGX-SPARK | DGX Spark present | BLOCKED_HARDWARE | machine_genome: no DGX; IV-C NOT_STARTED |
| CAP-HIGH-END-FPGA | High-end fabric in play | CONCEPT_ONLY | §25.1 prose; U50 tuition unpaid |
| CAP-CUSTOM-BOARD | Custom board justified | CONCEPT_ONLY | Phase XVI / §25; no U50 tuition |
| CAP-ASIC | ASIC primitive earned | CONCEPT_ONLY | §25.3; no caller, no receipt |

## Surprises (loud)

**Appendix I vs I.23.** Same appendix, two orders. Phases: eGPU then FPGA. Law: FPGA then (maybe) fusion fabric, never eGPU. What would settle: a freeze that deletes or SUPERSEDES the phase list, or rewrites it to match I.23/§17.2. This audit does not edit the freeze.

**Forbidden Fruit is unnamed in the freeze.** The lab, registry tool, tests, and a PASSED receipt exist. H-ROADMAP does not know the name. What would settle: whether the lab was added after 2026-08-27 on purpose, or dropped during absorption of the ANE docs.

**Prefill branch not on this HEAD.** `docs/PREFILL.md` describes `qwen38_hybrid_prefill.rs` / `qwen38_prefill.metal` on `grok/prefill-gemm-20260901-232724`. Those paths are absent here. `generate_greedy` still decode-steps the prompt. What would settle: whether that branch landed anywhere reachable from this HEAD.

**H-ROADMAP is not in git.** Authority is a Downloads path. civilization/build_state.py and tools/roadmap/parse.py both hard-code it. The freeze can drift without a commit.

**Fusion’s own gate depends on HMF.** Era IV-A Fusion, IV-B HMF, but `FUSION_FIRST_HETEROGENEOUS_EXECUTABLE` is blocked on `HMF_DEVICE_VISIBLE_TRUST`. The producer already has the strategy’s order. The era numbers do not.

## Constitution (explicitly not in play)

Exactly five eras. Exactly three Odysseys. No Era VI. FPGA stays inside Hawking Accelerator / Fusion. Theia stays one generalist bounty model. 0.7% is not rewritten. None of the contradictions above require breaking those rules. They require not treating Appendix I phases, Era IV-A/IV-B numbering, Flash-buried ANE, and a missing prefill school as if they were still the acceleration strategy.

## Not done

Did not rewrite H-ROADMAP.md. Did not start an implementation campaign. Did not touch `hcli/`, `tools/`, `crates/`, or `civilization/`.
