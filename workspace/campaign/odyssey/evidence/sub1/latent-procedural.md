I'll start by reading the frontier packet in full so the proposals match its guardrails and output schema.I’ll pull prior Hawking science, patient codecs, and related art so the base+correction proposals stay grounded.I’ll pull the Odyssey packet, prior BPW receipts, and residual-encoding notes next so the proposals cite measured numbers.I’ll hunt GLM 0.167 BPW, Qwen3.8 proxy collapse, rice residual numbers, and bible §29/§58 so the proposals stay on measured ground.I'll search the named trees for the GLM cosine/BPW numbers, Qwen sub-bit collapse, Bible sections, residual-tier mechanisms, and BitNet citations.I'll read the listed receipt and policy files and extract only the requested numbers.Grep isn’t in the listed tools; I’ll try it anyway so we can hit the actual numbers.Need remaining residual-encoding rows (up_proj @ 2%) and GLM 0.167 / hgravs01_r160_b3 hits.I'll use the Sourcegraph MCP grep next so we can pull exact line hits.=== /Users/scammermike/Downloads/hawking/receipts/QWEN80_RESIDUAL_ENCODING.json [L16-L17,L142-L145,L1204-L1466,L1745-L1748,L2807-L3025] ===
reconstruction: "binary sign/scale base plus a sparse additive correction at the selected positions; value_bits<16 changes only the stored correction magnitude"
bits_per_outlier_definition: "8 * (codec_payload_bytes - binary_payload_bytes) / outlier_count; payload includes the binary base, container header, and residual body"
selection: "identical to _residual_codec: global top-k by |W - binary(W)|"
residual_budget_bpw_8bit_nonexpert: 0.1742352396249771

organ model.layers.10.mlp.experts.453.gate_proj.weight
binary cosine=0.8932464137407012 expert_bpw=1.126922607421875 payload_bytes=147708

outlier_frac=0.02 outlier_count=20972
rice_q1_rms cosine=0.9165506181334256 expert_bpw=1.291839599609375 bits_per_outlier=8.2456608811749
rice_q1_mean_abs cosine=0.9165184192825984 expert_bpw=1.2918777465820312 bits_per_outlier=8.247568186152966
rice_q4_absmax cosine=0.9169100871532744 expert_bpw=1.3519744873046875 bits_per_outlier=11.25233644859813
rice_fp16 cosine=0.9171842982674411 expert_bpw=1.5915679931640625 bits_per_outlier=23.231737554835018
group_local_fp16 cosine=0.9171842982674411 expert_bpw=1.6209640502929688 bits_per_outlier=24.70150677093267
bitmap_fp16 cosine=0.9171842982674411 expert_bpw=2.3836746215820312 bits_per_outlier=62.836162502384134
legacy_u32_fp16 cosine=0.9171842982674411 expert_bpw=2.0881423950195312 bits_per_outlier=48.059889376311276

organ model.layers.10.mlp.experts.453.up_proj.weight
binary cosine=0.8275162668981674 expert_bpw=1.126922607421875 payload_bytes=147708

outlier_frac=0.02 outlier_count=20972
rice_q1_rms cosine=0.8641625485550589 expert_bpw=1.29180908203125 bits_per_outlier=8.244135037192446
rice_q1_mean_abs cosine=0.864072736244655 expert_bpw=1.2918472290039062 bits_per_outlier=8.246042342170513
rice_q4_absmax cosine=0.8649987010121568 expert_bpw=1.3519439697265625 bits_per_outlier=11.250810604615678
rice_fp16 cosine=0.865143437435382 expert_bpw=1.5915374755859375 bits_per_outlier=23.230211710852565
group_local_fp16 cosine=0.865143437435382 expert_bpw=1.6209640502929688 bits_per_outlier=24.70150677093267
bitmap_fp16 cosine=0.865143437435382 expert_bpw=2.3903884887695312 bits_per_outlier=63.171848178523746
legacy_u32_fp16 cosine=0.865143437435382 expert_bpw=2.0881423950195312 bits_per_outlier=48.059889376311276

=== /Users/scammermike/Downloads/hawking/receipts/QWEN80_REPRESENTATION_FRONTIER_SWEEP.json ===
bar=0.8604 expert_bpw_allowance=1.3011578470468521

layers.10.experts.453.gate_proj
binary_g cosine=0.8932464137407012 expert_bpw=1.126922607421875 verdict=PASS
binary+resid_2pct cosine=0.9171842982674411 expert_bpw=2.0881423950195312 verdict=over-budget
binary+resid_5pct cosine=0.9348738551218126 expert_bpw=3.5281448364257812 verdict=over-budget

layers.10.experts.453.up_proj  [binary fails bar; residual clears bar]
binary_g cosine=0.8275162668981674 expert_bpw=1.126922607421875 verdict=fail
binary+resid_2pct cosine=0.865143437435382 expert_bpw=2.0881423950195312 verdict=over-budget
binary+resid_5pct cosine=0.8915749276089628 expert_bpw=3.5281448364257812 verdict=over-budget

layers.3.experts.494.gate_proj  [binary fails bar; residual clears bar]
binary_g cosine=0.8585935762823004 expert_bpw=1.126922607421875 verdict=fail
binary+resid_2pct cosine=0.8921522719311777 expert_bpw=2.0881423950195312 verdict=over-budget
binary+resid_5pct cosine=0.9144990308556648 expert_bpw=3.5281448364257812 verdict=over-budget

layers.3.experts.494.up_proj  [binary fails bar; residual clears bar]
binary_g cosine=0.8258422168325341 expert_bpw=1.126922607421875 verdict=fail
binary+resid_2pct cosine=0.8664641631396747 expert_bpw=2.0881423950195312 verdict=over-budget
binary+resid_5pct cosine=0.8938995949108425 expert_bpw=3.5281448364257812 verdict=over-budget

=== /Users/scammermike/Downloads/hawking/receipts/QWEN80_DOWN_PROJ_FRONTIER_SWEEP.json ===
bar=0.8604 expert_bpw_allowance_8bit=1.3011578470468521
hgravs01_r160_b3: UNKNOWN (file has hgravs01_r192_b3 and hgravs01_r256_b3 only)
codecs with verdict_vs_1.3012 PASS: none (no candidate both clears_bar and fits_1.3012)

layers.1.experts.265.down
binary_g 0.8264830117545535 / 1.126922607421875 fail
binary+resid_0.25pct 0.8372991279272517 / 1.2481460571289062 fail
binary+resid_0.5pct 0.843962509210505 / 1.3681259155273438 fail
binary+resid_1pct 0.8550145042177935 / 1.6081314086914062 fail
binary+resid_1.5pct 0.8634967358629458 / 1.84814453125 over-budget
binary+resid_2pct 0.8705929134278911 / 2.0881423950195312 over-budget
binary+resid_3pct 0.8827235658613926 / 2.5681610107421875 over-budget
hgravs01_r192_b3 0.9132099561676162 / 1.536773681640625 over-budget
hgravs01_r256_b3 0.9209323874563516 / 2.044586181640625 over-budget

layers.32.experts.179.down
binary_g 0.811376440972125 / 1.126922607421875 fail
binary+resid_0.25pct 0.82168169276342 / 1.2481460571289062 fail
binary+resid_0.5pct 0.8284206326483243 / 1.3681259155273438 fail
binary+resid_1pct 0.838735380463715 / 1.6081314086914062 fail
binary+resid_1.5pct 0.8472146949035896 / 1.84814453125 fail
binary+resid_2pct 0.854365364712183 / 2.0881423950195312 fail
binary+resid_3pct 0.8668136217243277 / 2.5681610107421875 over-budget
hgravs01_r192_b3 0.9069425738311184 / 1.536773681640625 over-budget
hgravs01_r256_b3 0.9177530983145028 / 2.0445938110351562 over-budget

layers.46.experts.428.down
binary_g 0.8128680052862013 / 1.126922607421875 fail
binary+resid_0.25pct 0.8235168580253623 / 1.2481460571289062 fail
binary+resid_0.5pct 0.8298679120639975 / 1.3681259155273438 fail
binary+resid_1pct 0.8409951528762011 / 1.6081314086914062 fail
binary+resid_1.5pct 0.8495737733113499 / 1.84814453125 fail
binary+resid_2pct 0.8565039810578796 / 2.0881423950195312 fail
binary+resid_3pct 0.8694233147198769 / 2.5681610107421875 over-budget
hgravs01_r192_b3 0.8738972143071755 / 1.5367584228515625 over-budget
hgravs01_r256_b3 0.877784744477751 / 2.0445785522460938 over-budget

layers.35.experts.330.down
binary_g 0.806691122739587 / 1.126922607421875 fail
binary+resid_0.25pct 0.8167804982543675 / 1.2481460571289062 fail
binary+resid_0.5pct 0.8237162888368988 / 1.3681259155273438 fail
binary+resid_1pct 0.8347133349279652 / 1.6081314086914062 fail
binary+resid_1.5pct 0.843411588211054 / 1.84814453125 fail
binary+resid_2pct 0.850731175086746 / 2.0881423950195312 fail
binary+resid_3pct 0.863110252724947 / 2.5681610107421875 over-budget
hgravs01_r192_b3 0.9179733754931663 / 1.5367660522460938 over-budget
hgravs01_r256_b3 0.9278137063752688 / 2.0445785522460938 over-budget

=== /Users/scammermike/Downloads/hawking/receipts/QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json ===
gate_proj binary_group expert_bpw=1.1269 cosine_range=[0.8586, 0.8932]
up_proj binary + rice_q1_rms sparse residual @2% bits_per_outlier=8.24 expert_bpw=1.2918 cosine_range=[0.86416, 0.86524]
down_proj hgravs01_r160_b3 expert_bpw=1.27 cosine_range=[0.8862, 0.8978]
mixed_expert_bpw=1.22957 complete_bpw nonexpert_8bit=1.43051

=== /Users/scammermike/Downloads/hawking/receipts/odyssey-i/O005_SENSITIVITY.json ===
stored_bpw: UNKNOWN
active_bpw: UNKNOWN
cold_experts: UNKNOWN
doctor baseline battery=10/12 refusals=0/2 seal_verdict=PASS_WITH_WARNINGS

=== /Users/scammermike/Downloads/hawking/receipts/odyssey-i/O005_GRAVITY_q3-g32-experts.json ===
stored_bpw=4.0253 active_bpw=4.2305 stored_bytes=15362682880
doctor battery=10/12 refusals=0/2 hits=10 seal_verdict=PASS_WITH_WARNINGS verdict=CANDIDATE_PASS
cold_experts: UNKNOWN in this file

=== /Users/scammermike/Downloads/hawking/receipts/odyssey-i/O005_NX_gather.json ===
cold_experts=0 (route.cold_experts) cold_set=[] tokens_observed=93 n_experts=128 topk=8
hot_cold_verdict="mildly peaked: entropy 5.36/7.00, top16=23%, most-pop=1.8733%"
per_layer cold (layer,cold): 0:40 1:33 2:55 3:49 4:56 5:48 6:69 7:68 8:55 9:59 10:50 11:54 12:57 13:53 14:52 15:63 16:49 17:52 18:67 19:74 20:60 21:66 22:55 23:51 24:57 25:53 26:55 27:60 28:50 29:57 30:63 31:63 32:61 33:61 34:56 35:51 36:53 37:55 38:51 39:53 40:52 41:55 42:60 43:60 44:59 45:63 46:61 47:59
stored_bpw: UNKNOWN in this file
doctor: UNKNOWN in this file

=== /Users/scammermike/Downloads/hawking/receipts/odyssey-i/O003_SENSITIVITY.json ===
cold expert counts: UNKNOWN (no cold_experts field; inventory expert=78 shared_expert=78)
baseline battery=12/12

=== /Users/scammermike/Downloads/hawking/receipts/odyssey-i/O006_SENSITIVITY.json ===
cold expert counts: UNKNOWN (no cold_experts field; inventory expert=144)
baseline battery=12/12

=== /Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/GRAVITY_RULEBASE.json ===
base+correction rules: none named base+correction
related: R-organ-inversion "allocate bits to gate/up first, spend slack on down"
R-uniform-routing-no-cold-compress never_routed_experts == 0 supporting O005
R-sparse-active-expert-gather supporting O005

=== /Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/contracts/auto/o005_novelty-representation.md ===
patient stalled primary=1.7305 pressure=2.5 stored_bpw=4.0253 active_bpw=4.2305
anchor complete_bpw=4.0253 spec=q3-g32-experts
search: base+correction (sparse/residual/selected-hi-prec-channels/expert- or route-conditioned/procedural)

=== /Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/ODYSSEY_POLICY.json ===
families includes "base+correction (sparse/residual/selected-hi-prec-channels/expert- or route-conditioned/procedural)"
target TOTAL complete_bpw sub-1 / sub-0.2
component_floors_bpw=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
target_pressure_zones_bpw: reachable_or_explained=3.0 pressure=2.5 aggressive=2.0 structural_correction_tier=1.5 frontier_full_accounting=1.0
no numeric "correction budget" key other than residual in QWEN80_RESIDUAL_ENCODING residual_budget_bpw_8bit_nonexpert=0.1742352396249771

=== GLM 0.167 BPW ===
MISSING /Users/scammermike/Downloads/hawking/glm_recon.py
MISSING /Users/scammermike/Downloads/hawking/tools/odyssey/glm_recon.py
MISSING /Users/scammermike/Downloads/hawking/workspace/docs/glm.md
MISSING /Users/scammermike/Downloads/hawking/workspace/docs/GLM.md
0.167: UNKNOWN (not present in the files successfully read)Locator is list-dir only; I can’t open files or grep, so I’ll report path-level candidates and UNKNOWN for every quoted number.

**Locator limitation:** this session only has `list_dir`. No Grep, no Glob, no Read. **No `file:line` snippets and no quoted numbers can be extracted.** Paths below are directory hits only, not content evidence.

---

### 1. GLM activation-aware 0.755 cosine @ 0.167 BPW (real activations)
**UNKNOWN** (content not searchable)

Path-name candidates:
- `/Users/scammermike/Downloads/hawking/lab/operators/glm52_activation_aware_pack.py`
- `/Users/scammermike/Downloads/hawking/lab/operators/glm52_activation_aware_pack_v2.py`
- `/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/SUB1_FRONTIER.md`
- `/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/GRAVITY_RULEBASE.json`

### 2. Qwen3.8 gaussian-proxy sub-bit collapse (0.5–0.8 BPW, output-div ~0.69)
**UNKNOWN**

Path-name candidates:
- `/Users/scammermike/Downloads/hawking/tools/qwen38_capture_v2.py`
- `/Users/scammermike/Downloads/hawking/tools/qwen38_sub15_pack.py`
- `/Users/scammermike/Downloads/hawking/lab/qwen38_protected_run_verifier.py`
- `/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/SUB1_FRONTIER.md`

### 3. Bible §29 (stored vs ACTIVE sub-1 for MoE)
**UNKNOWN** — no file named `*bible*` under listed trees. `workspace/campaign/odyssey/evidence/sub1/` exists and is empty in listing.

### 4. Bible §58 / §92 (base+correction, matryoshka tiers)
**UNKNOWN**

Path-name candidates:
- `/Users/scammermike/Downloads/hawking/tools/matryoshka_pack.py`
- `/Users/scammermike/Downloads/hawking/tools/gravity_correction_topology.py`

### 5. Existing base+correction / residual-tier / binary+sparse residual in Hawking
**UNKNOWN** (no line evidence)

Path-name candidates:
- `/Users/scammermike/Downloads/hawking/tools/llama_residual_pq_pack.py`
- `/Users/scammermike/Downloads/hawking/receipts/QWEN80_RESIDUAL_ENCODING.json`
- `/Users/scammermike/Downloads/hawking/receipts/DSV4F_RESIDUAL_COMPOSITION_ORACLE.json`
- `/Users/scammermike/Downloads/hawking/lab/operators/dsv4f_residual_composition_oracle.py`
- `/Users/scammermike/Downloads/hawking/lab/operators/ascension_qwen80_first_residual_bridge_launcher.py`
- `/Users/scammermike/Downloads/hawking/workspace/docs/plans/ascension/RESIDUAL_TEACHER_ADMISSION_GATE.md`
- `/Users/scammermike/Downloads/hawking/workspace/docs/plans/RESIDENT_MEMORY_AND_BPW_PLAN.md`

### 6. BitNet / ternary / 1-bit PTQ citations
**UNKNOWN** — no filename containing bitnet/ternary/ptq in the listed dirs.

### `receipts/QWEN80_RESIDUAL_ENCODING.json` rice_q1_rms vs binary @ 2% outliers
**UNKNOWN** — file exists at `/Users/scammermike/Downloads/hawking/receipts/QWEN80_RESIDUAL_ENCODING.json`; `expert_bpw` / cosine for `up_proj` and `gate_proj` not extractable without Read.

---

**Also present (names only):**  
`/Users/scammermike/Downloads/hawking/workspace/campaign/odyssey/ODYSSEY.md`, `GRAVITY_RULEBASE.json`, `NEGATIVE_SCIENCE.json`, `h_odyssey.md` **not listed** at repo root.

**0 hits** with snippets (cannot search file bodies).  
**PATTERNS TRIED:** none (Grep unavailable). Directory walk only: `workspace/campaign/odyssey/`, `workspace/docs/`, `receipts/`, `tools/`, `spec/`, `ramanujan/`, `docs/`, `lab/operators/`.The GLM 0.167 path and correction-topology tools are the remaining citations. I’ll read those next.The byte math is easy. A 0.3-BPW structured base plus a small correction **beats a naked 2-BPW codec on complete_bpw whenever correction stays under ~1.9 BPW** (rice-q1 at 20% outliers is still cheaper than Q80 `uniform_b2` at 2.252). Quality is the binding constraint, not bits.

**Naked “2-bit” is not 2.0 complete.** Affine-2 g32 is **3.00** payload (2 + f16 scale + f16 bias per 32; G21 whole-model **3.22**). Q80 `uniform_b2` is **2.252**. O005 `q2-g32-experts` **Doctor-failed** (gate/up + router). Compare complete-to-complete.

**Measured instance of the family (not 0.3-base):** Q80 `up_proj` binary-g128 + `rice_q1_rms` @ 2% = **1.292 expert_bpw, cosine 0.864** vs `uniform_b2` **2.252 / 0.801**. Same reconstruction, cheaper index (`receipts/QWEN80_RESIDUAL_ENCODING.json`). Gate often needs **no** residual (binary **0.893 @ 1.127**). Down does **not** sparse-residual (bar only at 1.5–3% and then over-budget); it wants activation-weighted low-rank. Mixed screen: expert **1.230**, complete **1.431** at 8-bit non-expert — **not packed, not generated, not Doctor** (`QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json`).

**Packed binary cannot make stored sub-1.** Binary-g128 complete is **1.127** (1 sign + f16/128 + ~0.002 header). O005 experts are ~95% of mass: `0.95*1.13 + 0.05*8 ≈ 1.57`. Stored sub-1 needs expert complete **≲ 0.63–0.90**. That is why a **0.3-BPW base** has to be **structured keep**, not 1-bit of every weight.

**Active sub-1 is an attention/head problem once experts are crushed.** O005: ~0.54 of *active* params are selected experts, ~0.27 attn, ~0.19 embed/head/router (always-on). Experts at 0.3 + attn at 4 + other at 8 → active **~2.8**, not sub-1. Bible §29 still holds as an *axis*; hitting the number requires the same family on attn.

---

### Guardrails (how every proposal is scored)

| gate | how this lane honors it |
|---|---|
| complete accounting | payload + f16 scales + occupancy + rice/group-local index + residual values + container + **expanded GPU index buffers if that is how it runs** |
| activation-aware only | residual rank by `d_j·resid²` (G070), not `\|W−binary(W)\|`; cosine on real X; Doctor on held-out. Qwen3.8 gaussian-proxy sub-bit is the trap (packet; output-div ~0.69) |
| Doctor authority | Q80 mixed is a **screen**. Fast battery ≠ seal. Gibberish is useful iff it names the broken organ |
| native / no expand | HGRAVB01 + HGRAVR02 already bind in `qwen38_hybrid_decode.rs`. Load currently **expands Rice → u32 indices** — bill that as active/working-set until a fused kernel exists. G24: 1.16× fewer bytes → 1.05× TPS (kernel-bound) |
| stored vs active | MoE stored sub-1 ≠ active sub-1. 0 cold experts on O005/O003/O006 — no popularity skip |

**Do not re-propose:** NS-raw-weight-PQ @ ~1 bit (F1 1.0075 and 0.493 collapsed 6/6); F0 `D4_pq_doctor` residual-codebook @ 0.15 (collapsed); NS-uniform-subbit; NS-ternary-**factorization**; NS-posthoc-scalar-gain on k-means; bitmap residual index (Q80 63–127 bits/outlier > legacy 48); dense q2 correction plane (G11 +2.25 BPW stored; PHASE_B_HYBRID q2+LR cannot recover q3 at matched bytes); expand-to-dense GEMV (G17 `dense_w_materialized=0` is the bar).

**Prior art to steal, not rediscover:** BiLLM / PB-LLM (1-bit + salient split); SqueezeLLM/SpQR (outlier channels); AWQ (salient channels); AQLM (additive codebooks = stacked correction); BitNet b1.58 (ternary *base*, not factorization); QuIP# RHT as **pack-time** rotation only (QTIP Metal trellis = Type-1 tps kill). GLM packet claim **0.755 cos @ 0.167 BPW on real X** — cited from `SUB1_FRONTIER.md`; this lane did not locate the receipt (UNKNOWN path). Distinct from GLM-5.2 weight-space expert PQ **0.116–0.157 cos @ 0.75 BPW**, Type-1 (`dead_levers.md`).

---

### Break-even (bytes only; quality separate)

Assume 0.30 structured-binary base. Rice-q1 residual ≈ **8.24 bits/outlier** (Q80 MEASURED @ 2%).

| correction frac (of full tensor) | corr BPW | complete (base+corr) | vs `uniform_b2` 2.252 | vs affine-2 3.00 |
|---:|---:|---:|---|---|
| 2% | 0.165 | 0.465 | beats | beats |
| 5% | 0.412 | 0.712 | beats | beats |
| 10% | 0.824 | 1.124 | beats | beats |
| 20% | 1.648 | 1.948 | beats | beats |
| 24% | 1.978 | 2.278 | lose | beats |

**Bytes never kill 0.3+corr against naked 2 until residual ≳ 23%.** If Doctor needs more than that, the topology or the base is wrong — do not globally retreat to q2.

---

## BC1 — organ-mixed binary + rice-q1 residual (the measured 1.3-band)

**mechanism:** Gate = naked `HGRAVB01` g128; up = `HGRAVR02` binary + rice_q1_rms @ ~2%; down = **not** sparse residual (`HGRAVS01` / hi-prec). Heterogeneous organs, one family.

**complete_byte_accounting**

| slot | formula (Q80 MEASURED where noted) |
|---|---|
| binary signs | 1 bit/W |
| group scales | f16 / 128 = 0.125 |
| container | ~0.002 (Q80 1.127−1.125) |
| residual index | Rice deltas; **8.24 bits/outlier all-in** @ 2% (includes header) |
| residual values | 1-bit sign × one f16 RMS scale (in the 8.24) |
| **not counted unless used** | codebook, router, embed, attn, alignment pad, **expanded u32 index working set** |

Q80 mixed screen: gate 1.127 + up 1.292 + down 1.27 → expert **1.230**; complete **1.431** (8-bit non-expert) / **1.312** (4-bit).

**stored_bpw / active_bpw**

| patient | stored (DERIVED envelope, not packed) | active |
|---|---|---|
| Q80 experts | 1.23 expert → 1.43 complete @ 8-bit NE | top-10/512 of that + full attn/head |
| O005 | ~0.95×1.23 + 0.05×4 ≈ **1.27** if attn stays 4-bit | still ~3+ until attn crushed (see BC6) |
| O001/O004 dense | stored = active ≈ 1.2–1.5 on MLP mass only | no MoE subsidy |

**expected_reachable_bpw:** **1.2–1.5 complete** on MoE expert body if Doctor holds. **Not sub-1 stored.** Sub-0.2: no.

**quality_risk / limiter:** Up_proj and the **worst gate** (Q80 L3 e494 gate binary 0.859, *just under* 0.8604 bar — 2% residual clears). Down sparse-residual is the wrong organ (NS-organ-inversion / Q80 down sweep). Screen ≠ generate (`claim_boundary.coherence_generation_tested=false`).

**cheapest_falsifier:** One O005 expert `up_proj` + one `gate_proj` on **real routed X**: binary vs binary+rice_q1@2% vs `uniform_b2` at counted complete_bpw; then 12-item fast-Doctor on a packed mixed catalog. Kill if mixed does not beat q2 **and** does not hold q3-anchor hits. Localize organ; do not globally go to q3.

**execution_path:** Native codecs 0/1/2 already in mixed catalog (`HGRAVB01`/`HGRAVR02`/`HGRAVS01`). Residual load **expands Rice → u32 + row_ptr** (`qwen38_hybrid_decode.rs` Residual arm). Fused binary+scatter GEMV: UNKNOWN if a kernel does it without expanded indices. Cost/token: <REDACTED> index DRAM + 1 add at outliers; G24 says kernel, not BW, dominates.

**applicability:** MoE experts first (O005/O006/O003). Dense MLP (O004, O001 mlp). **Not** SSM `A_log`/`D`/`dt_bias`, router, untied embed/lm_head.

**confidence:** HIGH that it beats naked 2-bit on **organ cosine** (measured). LOW that Doctor holds on Odyssey patients (unpacked). MEDIUM transfer Q80→O005 (same Qwen-MoE family, different expert width 512 vs 768).

**transfer:** O006 language-MoE 8/128 matches O005; vision tower out of expert path. O003 has **shared experts always-on** — treat like down/router (protect). O001: MLP only. Does **not** auto-transfer to SSM.

---

## BC2 — same bytes, functional residual selection (G070)

**mechanism:** Keep BC1 budgets. Rank correction by **output error** `d_j · resid²` with `d` = input-channel energy on **held-in real X**, not `\|W−binary(W)\|` (Q80 selection). G070: weight-magnitude is the G069 mistake.

**complete_byte_accounting:** Identical slots to BC1. Only the *set* of positions changes. If topology stays scattered, index entropy may change — **re-measure** Rice bits/outlier; do not reuse 8.24.

**stored_bpw / active_bpw:** Matched to BC1 by construction. UNKNOWN delta after re-encode.

**expected_reachable_bpw:** Same 1.2–1.5 band; possible **lower residual frac** for the same bar (UNKNOWN).

**quality_risk / limiter:** Fit/hold split on X (PHASE_B functional-LR **overfit** 0.20 fit → 0.39 hold). Calibrate d on ≥1000 tokens disjoint from Doctor (NS-calibration-88). Proxy X = instant kill.

**cheapest_falsifier:** One tensor, two index sets, **same k and same encoding**. Hold-out `‖XWᵀ−XŴᵀ‖/‖XWᵀ‖` on unused tokens. If functional selection does not beat `|resid|` at matched k, G070 is dead on that organ — stop.

**execution_path:** Same HGRAVR02. Pack-time only.

**applicability:** Any organ BC1 touches. Highest EV on up_proj (binary fails bar, 2% just clears).

**confidence:** MEDIUM. Script exists (`tools/gravity_correction_topology.py`); **patient G070 numbers UNKNOWN**.

**transfer:** Method-universal if X is real. d is layer- and patient-specific.

---

## BC3 — 0.27-keep binary as a true 0.3-BPW base + restore

**mechanism:** The only honest 0.3-BPW *binary* base: keep **~27% of input-channels (or output-rows)** at full binary-g128; drop the rest to 0. Occupancy = 1 bit/channel (not per-weight). Restore the dropped units that dominate Doctor-delta as ROW/COL hi-prec (8-bit) or rice residual. This is **not** 1-bit of everything.

keep_frac × 1.127 ≈ 0.30 ⇒ keep_frac ≈ **0.266**. Wanda-style keep: `‖X_j‖·‖W_{:,j}‖` on real X.

**complete_byte_accounting**

| slot | bpw (DERIVED; g=128) |
|---|---|
| kept signs | 0.266 × 1.0 = 0.266 |
| kept scales | 0.266 × 0.125 = 0.033 |
| channel mask | ~1/width (O005 expert in=2048 → **0.0005**) |
| restore f% of *full* tensor @ rice-q1 | f × 8.24 |
| container | ~0.002 |
| **example f=5%** | **0.30 + 0.41 = 0.71 complete** on that tensor |

O005 stored envelope if *all* experts use 0.71 and non-expert 8-bit: `0.95×0.71 + 0.05×8 ≈ 1.07` (near sub-1). At f=2%: `0.95×0.46 + 0.05×8 ≈ **0.84 stored**`. That is the stored-sub-1 path. **UNMEASURED quality.**

**stored_bpw / active_bpw:** Stored as above. Active: only selected experts’ keep-mask + their restore plane (see BC5). Dense: stored=active.

**expected_reachable_bpw:** **0.45–1.1** stored on MoE experts *if* a 2–8% restore holds Doctor (UNKNOWN). Sub-0.2 stored: only if keep_frac≪0.15 **and** restore≪2% — **not expected** on current patients. Qwen3.8 `post_swiglu` needs **37–57% dims for 99% energy**, exact0=0 (`MLP_ACTIVATION_SPARSITY.json`) — 73% drop will lose more than 1% energy unless MoE experts are narrower/more concentrated (UNKNOWN).

**quality_risk / limiter:** The dropped 73%. Limiter = **input channels of up/gate** (energy-diffuse). Down even worse for drop (already fails at full binary). Embed/lm_head: GLM R0 sub-bit embed was catastrophic (Q80 ledger). Layer-0 may be a different source (NS-layer-zero).

**cheapest_falsifier:** One O005 expert `up_proj`, keep top-26.6% columns by d on half of real routed X; score hold-out cosine vs full binary and vs `uniform_b2`. If keep-0.27 cosine ≪ full-binary (0.83) by more than a 5% rice restore recovers, **0.3-base is dead** on that organ. Do not pack a model.

**execution_path:** Channel-sparse binary GEMV = gather kept columns into a skinny matvec (native path **does not exist** today; mlx `gather_qmm` is expert-axis). Fallback: dense binary with zeros — **fake 0.3** (you still touch 1.13 bits). Must have a gather kernel or it is not a native win. Cost: index of kept cols (cheap) + irregular GEMV (kernel risk; G24).

**applicability:** MoE expert up/gate. **Not** down, router, SSM, embed. Dense MLP only after MoE screen.

**confidence:** LOW–MEDIUM as a Doctor path; HIGH as the *only* honest 0.3 binary-base construction. Confidence the byte math beats 2-BPW: HIGH. Confidence Doctor: UNKNOWN.

**transfer:** Keep-frac will not transfer; re-measure energy CDF per organ/patient.

---

## BC4 — correction topology at matched complete_bpw (not matched k)

**mechanism:** G070: scattered entries pay `16+⌈log2(n)⌉` each; a whole row pays one row-id for `cols` values. Q80 used **global top-k scattered** (then Rice). At a **fixed bit budget**, ROW / COLUMN / BLOCK / LOW-RANK buy different *counts*. Topology is the cheapest falsifier (`candidate_families.base_plus_correction`).

**complete_byte_accounting:** One budget B ∈ {0.16, 0.25, 0.50} extra BPW on top of the binary base. Spend B entirely in one topology; itemize index vs values.

| topology | index | Q80 2% lesson |
|---|---|---|
| SCATTERED + rice-q1 | Rice deltas | **8.24 b/outlier** (winner vs legacy 48) |
| SCATTERED + u32+f16 | 32+16 | 2.088 BPW, over-budget |
| BITMAP | occupancy+mask | **63–127 b/outlier — dead** |
| ROW | ⌈log2(rows)⌉ | UNKNOWN on these patients |
| COLUMN | ⌈log2(cols)⌉ | UNKNOWN; matches activation axis if y=xWᵀ |
| BLOCK g64 | ⌈log2(n/64)⌉ | UNKNOWN |
| LOW-RANK | 0 index; r(m+n)·16 | Q80 down: `hgravs01_r192_b3` 0.91 cos @ 1.54, still over 1.30; PHASE_B_HYBRID: q2+LR cannot recover q3 |

**stored_bpw / active_bpw:** Matched B by construction. Active: ROW/COL are gather-friendly; scattered is scatter-add (worse kernel).

**expected_reachable_bpw:** Same envelopes as BC1/BC3; topology can **cut residual frac** needed to clear the bar (UNKNOWN magnitude).

**quality_risk / limiter:** Picking SCATTERED because “outliers are scattered” without pricing the index. Qwen3.8 FFN block-256 sparsity was Type-1 dead (scattered neurons, 0.2% skip @ 99% recall) — **weight** sparsity ≠ **correction** sparsity, but the kernel rhyme is real.

**cheapest_falsifier:** Run `gravity_correction_topology.py` on one O005 expert gate + one up + one down, budgets `0.16,0.25,0.50`, real X, hold-out. Winner = lowest hold-out err at matched B. If SCATTERED+rice already wins, stop. If ROW/COL wins, pack that, not k%.

**execution_path:** ROW/COL = extra hi-prec GEMV on a skinny slice (native affine/uniform kernels exist). Scattered = current Residual expand. Block = group-scale override (close to mixed-q).

**applicability:** All classes. Down: expect LOW-RANK to win (already measured). Gate/up: expect COLUMN or SCATTERED+rice.

**confidence:** HIGH that topology is the right question. LOW on which wins (patient numbers UNKNOWN).

**transfer:** Topology ranking is organ-specific; do not copy Q80 scattered onto down.

---

## BC5 — route-conditioned correction (stored vs active)

**mechanism:** Store the full expert correction inventory. **Fetch/apply T1 only for the routed expert set.** Base binary of selected experts always runs. 0 cold experts (O005/O003/O006 MEASURED) ⇒ cannot drop stored experts; **can** drop active correction traffic. 16× selection is an execution lever, not a compression stat (packet; O005 8/128=0.0625).

**complete_byte_accounting**

| axis | what to bill |
|---|---|
| stored | Σ_experts (binary + correction + masks) + attn + embed + head + router + container |
| active/token | top-k × (binary_e + **correction_e if T1 hits**) + shared_expert + attn + embed + head + router |
| **illegal** | `stored × topk/N` as “active_bpw” (`no_fake_active_density`) |

O005 NX: mlx `gather_qmm` already gathers compute; **full expert body stays resident** (`O005_NX_gather.json`). Active-byte win requires **residency** drop, not just gather.

**stored_bpw / active_bpw:** Stored = BC1/BC3. Active correction = (topk/N) × corr_bpw × hit_rate. If hit_rate=1, O005 expert-corr active ≈ 0.0625 × 0.16 = **0.01 BPW** — rounding. The rest of active is attn/head.

**expected_reachable_bpw:** Active expert-side **≪ 1** even when stored experts ~1.2. Whole-token active sub-1 still needs BC6.

**quality_risk / limiter:** Applying T0-only on a “cheap” expert that was not actually cheap (uniform routing: most-popular **1.4–1.9%**, entropy 5.4–6.2 / 7). Hit_rate≈1 under uniform route ⇒ you still *compute* all selected corrections. Win is **movement/residency**, not skip. Transition overlap ~0.41 — prefetch of next expert set is a different rule (`R-predictable-route-prefetch`, P(E_t|E_{t-1}) UNKNOWN as a lever).

**cheapest_falsifier:** Profile one O005 decode: bytes moved for (binary only) vs (binary+corr) with corr buffers **not resident** until route. If TPS unchanged (kernel-bound, G24/A3B caveat), this is a **memory win only** — still valid stored/active science, not an NX tps claim.

**execution_path:** Expert-gather of two planes. mlx specimen is not Hawking NX. Native gather of HGRAVR02 for top-k: **does not exist** as a residency-dropping kernel.

**applicability:** MoE only (O005/O006 8/128; O003 6/64 + 2 shared — **shared always pays T1**).

**confidence:** HIGH on accounting. MEDIUM on NX (residency not implemented). LOW on tps.

**transfer:** MoE-universal gather; route-uniform ⇒ no cold-compress (R-uniform-routing-no-cold-compress MEASURED O005, O006 transfer cold_experts=0).

---

## BC6 — attention/head binary+correction (required for active sub-1)

**mechanism:** Same family on **always-on** GEMVs. `tools/qwen38_sub15_pack.py` already recipes attn rice_q1@2% from BF16; embed/lm_head stayed Q4. Without this, crushing experts cannot produce active sub-1 (mass math above).

**complete_byte_accounting:** Per attn projection: binary 1.13 + rice residual + QK-norm (keep f32; O005 norms unquantized). Embed/lm_head: **do not** 0.3-base (Q80 ledger: sub-bit embed catastrophic on GLM R0). Bill them at 4–8 bit.

**stored_bpw / active_bpw:** Attn is ~100% active. O005 attn 1.81 GB bf16. At 1.29 BPW: 0.146 GB. Combined with expert-active at 0.3–1.3: still plus embed/head.

Rough O005 active envelope (DERIVED, not measured):

| expert BPW | attn BPW | other BPW | ~active_bpw |
|---|---|---|---|
| 1.29 | 4 | 8 | ~3.3 |
| 0.46 | 1.29 | 8 | ~2.0 |
| 0.46 | 1.29 | 4 | ~1.2 |
| 0.46 | 1.13 | 4 | ~1.15 |

**Active sub-1 on O005 needs other (embed/head) ≲ 2-bit *and* attn ≲ 1.3 — UNKNOWN quality, high risk.** Stored sub-1 does **not** need that (experts dominate stored).

**expected_reachable_bpw:** Active **1.1–2.0** if attn accepts binary+2% (UNKNOWN on O005). Active **<1**: not expected without touching embed/head.

**quality_risk / limiter:** Attn / QK-norm / router. O005 localization of q2 failure already names **gate + router**. Router is 0.03 GB — keep 8-bit (`R-protect-router-if-sensitive`). DeltaNet/SSM (O001, Qwen3.8): recurrent error accumulates — **not** a 0.3-base candidate.

**cheapest_falsifier:** In-place: binary+rice one attn projection family (q_proj) on O005 4-bit specimen vs untouched; fast-Doctor delta_hits. If q_proj alone drops hits, attn is not crushable with this family.

**execution_path:** Same HGRAVB01/R02. Native mixed already runs attn as Uniform/Affine; Residual-on-attn is a catalog-role extension (today Residual allowed on `up_proj` only in the MLP lock). **Lock must change** or this is non-native.

**applicability:** Dense+MoE attn. Hybrid: attn yes, SSM no.

**confidence:** HIGH that this axis is required for active sub-1. LOW that it holds Doctor.

**transfer:** Attn sensitivity is patient-specific (O005 zero-ablate attn: 0/12). Do not assume Qwen3.8 affine-2 attn coherence (5/6) transfers to 1-bit+2% residual.

---

## BC7 — sparse Matryoshka T0/T1 (executable 1.13, optional 0.16)

**mechanism:** T0 = full binary-g128, **standalone executable** (G11 lesson: base must generate without the plane). T1 = rice-q1 residual, stored, applied when T0 organ error exceeds a threshold (per-token/per-expert). **Not** G11’s dense q2 plane (+2.25 BPW, hierarchy 5.50). PHASE_B_HYBRID: q3+LR correction **generalizes but adds bytes** (quality lever, not density). Sparse T1 is the density-compatible sibling.

**complete_byte_accounting:** Stored = T0+T1 always (unless a residency mode drops T1). Active = T0 + 1_{hit} T1. Report stored tiers, active tiers, hit rate (`candidate_families.matryoshka_tiers`).

**stored_bpw / active_bpw:** Stored ≈ BC1 (1.13+0.16). Active ≈ 1.13 if hit_rate=0 (then T1 is dead weight — G11 cheapest falsifier: if T1 never hits, drop it). If hit_rate=1, active=stored.

**expected_reachable_bpw:** Stored 1.2–1.5; active 1.13–1.5. Sub-0.2: no.

**quality_risk / limiter:** Threshold Goodhart (tune on the Doctor battery). T0-only gibberish on up_proj is expected (binary 0.83). If T1 must always fire on up, this collapses to BC1 with extra machinery.

**cheapest_falsifier:** Decode T0-only vs T0+T1 on fast-Doctor. If T1 never restores a hit T0 lost, drop T1. If T0 never loses a hit, T1 is wasted stored.

**execution_path:** Two-plane Residual kernel. Skip T1 = skip the scatter-add. Native path: Residual already *is* T0+T1 fused; skipping T1 is a branch. Cost: one extra pass or predicated add.

**applicability:** All. Most useful where some experts/layers tolerate T0 (gate) and some do not (up).

**confidence:** MEDIUM. G11 proved hierarchy **structure**; PHASE_B_HYBRID proved dense T1 is not a byte win. Sparse T1 untested on Odyssey.

**transfer:** Pattern transfers; thresholds do not.

---

## BC8 — BitNet-style ternary base + sparse repair (not factorization)

**mechanism:** Per-tensor (or per-row) scale × ternary codes `{−1,0,+1}` as T0; rice/row repair as T1. **Distinct from NS-ternary-factorization** (ternary *factors* vs VQ, killed on gpt-oss-120b). Packed ternary ≈ 1.58 entropy, 2 bits naive, or 5-trits/byte ≈ 1.60 + scale.

**complete_byte_accounting:** Trit payload (measure packed size, do not quote 1.58 as complete) + scale(s) + T1. If unpacked to 2-bit, complete **> binary 1.13** and this **loses to BC1 on bytes** unless zeros are dense enough to entropy-code below ~0.9.

**stored_bpw / active_bpw:** Likely **1.6–2.2** stored before T1 (UNKNOWN pack). Only interesting if measured packed complete **< binary 1.13** on a real organ (many exact zeros).

**expected_reachable_bpw:** UNKNOWN. If trit-pack ≲ 0.9, then +2% rice could land ~1.1 and still beat uniform_b2. 0.3-base: only with keep-frac (BC3 applied to ternary) or if ~70% are exact 0 **and** occupancy is structured.

**quality_risk / limiter:** 1-bit PTQ without QAT (BitNet is trained ternary). NS-raw-weight family is about PQ/VQ, not ternary — still expect a capability cliff without source change. Reopen QAT/distill only as **source-changing** (F1: not bound by the raw-weight kill).

**cheapest_falsifier:** Pack one O005 `gate_proj` ternary-per-row-scale vs binary-g128, **matched complete_bpw** (pad the cheaper one). Real-X cosine + one fast-Doctor organ swap. If ternary loses at matched complete, **dead vs BC1** — do not stack T1.

**execution_path:** No native ternary GEMV in the mixed catalog. Affine-2 can *store* 2-bit codes including a zero bin (q∈{0,1,2,3}) — that is **not** balanced ternary. New kernel or map onto affine-2 with a zero code (semantic mismatch; G23 affine ≠ absmax).

**applicability:** Same as BC1, **after** BC1 falsifier. Do not lead with this.

**confidence:** LOW. Exists to avoid ignoring BitNet; expected dominated by binary+rice on frozen weights.

**transfer:** Weak. Trained-ternary checkpoints (if a patient is native BitNet) reopen this as a **decode** problem, not PTQ.

---

### Campaign order (info/cost)

| # | experiment | kills | cost |
|---|---|---|---|
| 1 | BC4 topology + BC2 selection on **one** O005 up + one gate, real routed X | wrong topology / `|resid|` selection | CPU, one tensor |
| 2 | BC1 packed mixed vs `q2-g32-experts` vs `q3-g32-experts` fast-Doctor | family as a Doctor lever | one mlx/native catalog |
| 3 | BC3 keep-0.27 screen on the same up_proj | 0.3-base | CPU |
| 4 | BC7 T0-only vs T0+T1 Doctor | extra stored tier | same artifact, decode flag |
| 5 | BC6 q_proj-only | attn crush | one organ |
| 6 | BC5 residency probe | tps-from-bytes | clean-room TPS only if 2–5 hold |
| 7 | BC8 ternary vs binary matched complete | BitNet-PTQ | only if 1–3 stall |

If (1)+(2) fail, **stop the lane** on that patient: residual repair of a 1-bit base is not the remaining delta (O005 still sits at 4.03 stored). Then the deficit is structural/shared-basis or latent/generated, not base+correction.

**UNKNOWN (not guessed):** G070 numbers on Odyssey organs; GLM 0.167 receipt path; fused Residual GEMV vs expanded-index working set; O005 per-expert sensitivity; whether MoE expert activation energy is concentrated enough for BC3; Doctor under any HGRAVB01/R02 Odyssey pack.
