# G1 capability gate — replacement that can fail

Date: 2026-08-17. HEAD at write: this worktree. No GPU, no Metal, no resident mutation.
Write scope: this file only.

STATUS: **IMPLEMENT_READY**. Current capability bookkeeping cannot fail. The
replacement is specified below as files, functions, thresholds, and a required
negative-control test on mixed-2p0-v1.

Every number is tagged MEASURED (this lane), CLAIMED (prior receipt), or
PROJECTED. A component cosine is not a token-level claim.

---

## 0. What is broken (established + re-verified)

Three independent “1.0”s were assigned, not measured.

| field | what it claims | what it is | evidence |
|---|---|---|---|
| lineage `capability` 1.0/1.0/1.0 | measured coherence / discipline / engineering | `DEFAULT_CAPABILITY_CONTRACT` copied by `make_qwen38_genesis` | `lab/lineage/identity.py:28-31,252` |
| `artifact_sha` / `runtime_sha` / `kernel_genome_sha` | content identity | `sha256("hawking.lineage/"+label)` | `lab/lineage/canon.py:29-30`, `identity.py:245-249` |
| manifest `min_q4_cosine: 1.0` | min dequant cosine vs BF16 | `fold(1.0, min)` over 402 `None`s | `qwen38_pack.rs:680-684,312` |

The 2026-08-17 promotion-hardening receipt made six *paperwork* forgeries REJECT
(`receipts/ascent-2026-08-16/GENESIS_PROMOTION_GATE_HARDENING.json`). It did not
make capability a measurement. The “preimage” the hardened clause hashes is
still the label string `hawking.lineage/artifact/child-g1`
(`lab/lineage/testing.py:37-42`). Swapping catalog bytes at a path does not
move any seated hash.

The live resident already knows the lineage hash is stale and loads anyway:

```
genesis-resident: lineage identity is stale; loading measured artifact ... sha d650a757...
```

`tools/agentos/genesis_body/src/main.rs:839-848` compares
`sha256(manifest.json)` to `CURRENT.artifact_sha` (a labeled hash) and
continues. Health reports the manifest sha. Lineage keeps the labeled sha.
They are different numbers (MEASURED this lane, §3).

G0 capability itself now measures clean on the live body (wave-1 remasure):
6/6 oracle-32 match + `17*19=323`. That measurement is **not** what lineage
stores. Lineage stores the default contract.

---

## 1. Capability seal a Qwen3.8 candidate must contain

Schema: `hawking.genesis.qwen38_capability_seal.v1`.
One JSON document, written next to the artifact as `capability_seal.json`,
and copied into promotion evidence as `evidence["capability_seal"]`.
No axis float may appear on a `GenesisInstance` unless it is copied from this
document. Missing seal is PENDING, never 1.0.

```
{
  "schema": "hawking.genesis.qwen38_capability_seal.v1",
  "verdict": "PASS" | "FAIL" | "PENDING",
  "parent": {
    "model": "PocketAiHub/Qwen3.8-27B-Abliterated-MLX",
    "base_rev": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "bf16_index_sha256": "<sha256(model.safetensors.index.json)>",
    "g0_artifact_content_sha": "f590664c259cbea8fe90889e06e2f78f09c57f03f34f97b26635e524e5e06b5e"
  },
  "artifact": {
    "root": "<path>",
    "kind": "uniform_q4" | "hq38m20",
    "manifest_or_catalog_sha256": "<sha256 of manifest.json OR catalog.hq38m20 bytes>",
    "catalog_merkle_sha256": "<§3>",
    "artifact_content_sha": "<sha256(manifest_sha || merkle_sha)>",
    "complete_physical_bpw": "<recomputed 8*payload_bytes/elements, not trusted field>",
    "tensor_count": 755 | 851,
    "payload_bytes": <int>,
    "source_weight_elements": 26895998464
  },
  "runtime": {
    "executable_sha256": "<sha256 of the bytes that ran generate>",
    "kernel_source_sha256": "<sha256 of dispatched .metal sources, concatenated in name order>",
    "dispatch_names": ["qwen_uniform_q4_group64_matvec_geo_tpr64_tg128", "..."],
    "fallbacks": 0,
    "dense_w_materialized": 0,
    "vehicle": "native_reader"
  },
  "tokenizer_sha256": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
  "weight_screen": {
    "definition": "dequant(Wq) cosine vs BF16 parent, f64, streamed, 100% of GEMV tensors",
    "n_catalog_gemv": <int>,
    "n_measured": <int>,
    "n_none": 0,
    "min": <float>,
    "p10": <float>,
    "median": <float>,
    "argmin": "<tensor name>",
    "by_role": { "gate": {}, "up": {}, "down": {}, "attn": {}, "embed": {}, "lm_head": {} },
    "status": "PASS" | "FAIL"
  },
  "generation": {
    "vehicle": "native_reader",
    "template": "chat_template + tokenizer bytes hashed above",
    "max_new_tokens": 32,
    "prompts": [ {prompt, rendered, prompt_ids, new_token_ids, text, class, reasons, task, task_ok, parent_prefix_match} ],
    "oracle32_say_hi": { "match": true|false, "ids": [...] },
    "n_coherent": <int>,
    "n_fail_classes": <int>,
    "status": "PASS" | "FAIL"
  },
  "derived_capability": {
    "coherence": <n_coherent / n_required_probes>,
    "complete_token_discipline": <1.0 iff timing_authority present and same-stopwatch; else omitted>,
    "engineering": <1.0 iff fallbacks==0 and vehicle==native_reader and dense_w==0>
  }
}
```

Refuse to write `derived_capability` if `weight_screen.n_none > 0` or
`generation.status` is missing. Refuse `vehicle` other than `native_reader`.
The expand-to-float path
(`engine: mlx_lm_weights_overwritten_from_mixed_pack` on
`mixed-2p0-v1/GENERATE.json`) is FAIL, not evidence. Wave 1: both prior
sub-1.5 verdicts were that confound.

`tools/coherence_gate.py` is **not** this seal. It only diffs greedy ids
against a previous seal. It is a subroutine for the genome-only (same
`artifact_content_sha`) case.

---

## 2. Per-tensor dequant cosine vs BF16 — how to compute it, and when None is FAIL

### 2.1 Definition (WEIGHT cosine, not output cosine)

For one HQ30UQ4 tensor, the number `pack_uniform_q4_group64` already computes
when it actually packs (`qwen80_uniform_q4.rs:278-314`):

```
reconstructed[i] = (nibble_i - 8) * f16_scale_of_group
cosine = dot(src, recon) / (||src|| * ||recon||)     # accumulators f64
rel_l2 = ||src - recon|| / ||src||
```

Codec (same file:195-275, decode at 330-344):

- flat groups of 64
- stored scale = `f16(max_abs/7)`, FP16 value is authority
- `q ∈ [-8, 7]`, even local index in the low nibble, odd in the high
- ties-to-even via `rint` (`qwen80_uniform_q4.rs:174-187`)

`src` is the BF16 parent widened `f32::from_bits(u16 << 16)`
(`widen_source_to_f32`, same file:365-376). For fused catalog names
(`in_proj_qkvz`, `in_proj_ba`) `src` is
`fuse_in_proj_qkvz` / `fuse_in_proj_ba`
(`qwen38_geometry.rs:291-390`), not a BF16 tensor of that name.

This is **weight** cosine. It is a screen. It is not capability.
Q80 transfer (`q80-recalibrate-capability-bar.SUMMARY.json`): a packed
artifact generated coherent text at down_proj holdout **output** cosine
0.7684, below the 0.8604 organ bar. mixed-2p0 has
`mean_component_cosine: 0.9069688696406788` over 851 rows
(`PACK_REPORT.json`) and is INCOHERENT on native generate. Cosine cannot
pass-certify.

### 2.2 None is FAIL, never 1.0

Current fold (`qwen38_pack.rs:680-684`):

```
let min_q4_cosine = rows
    .iter()
    .filter(|row| row.kind == "q4")
    .filter_map(|row| row.cosine)
    .fold(1.0f64, f64::min);
```

`try_reuse_q4` writes `cosine: None` (`qwen38_pack.rs:312`). The second
reuse branch in `pack_q4_named` does the same (`:403`). Live G0 catalog:

```
q4_tensors 402
cosine is None: 402
present: 0
fold identity: 1.0
manifest.min_q4_cosine: 1.0
```

MEASURED this lane on
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json`.

Replacement fold:

```
if any(row.kind=="q4" and row.cosine is None): FAIL, min=null, n_none=N
if any(cosine non-finite): FAIL
else min = min(cosines), n_none=0
```

Never seed with 1.0. Never `filter_map` away None. A reseal that skips a
tensor is FAIL, not a smaller N.

f32v2 rows today write `cosine: Some(1.0)` without comparing
(`pack_f32_named`, `qwen38_pack.rs:462`; `try_reuse_f32`:342). Against raw
BF16 that number is a lie. MEASURED L0 `input_layernorm`:

```
stored_first=0.046875   bf16_first=1.046875
cosine vs raw BF16     = -0.5882498371575625
cosine vs (BF16 - 1.0) = 1.0     rel_l2=0
```

The packer converts MLX residual norms to HF delta (`mlx_residual_norm_to_delta`,
`:430-441`). The 1.0 is “we stored what we intended”, not “equals BF16”.
f32v2 cosine vs raw BF16 must not enter `min_q4_cosine`. Compare f32v2
against the converted target, or omit them from the Q4 min.

### 2.3 Streaming procedure (production; peak << 20 GB)

Do not materialise embed/lm_head as two f32 copies (2 × 5.08 GB).

1. mmap HQ30UQ4 payload; parse header (`parse_uniform_q4_header`).
2. mmap the parent safetensors shard; take `[data0+start, data0+end)`.
3. For groups of 64 (chunk 4096–8192 groups):
   widen BF16 → f32, dequant Q4 group, accumulate dot / src_sq / recon_sq / err_sq in f64.
4. Record `{name, shape, elements, cosine, rel_l2, rmse, n_nonfinite}`.

This lane’s helper self-check (random N(0,0.02) 256-vector, same quantiser):
`self_check_cosine=0.9951864747866593`, `max_abs_err=0.004397718235850334`.
Dequant matched pack recon bit-for-bit on that vector.

### 2.4 MEASURED G0 ceiling (this lane, no GPU)

Command: `python3 /tmp/g1_capability_gate_full_cosine.py`
(streamed dequant vs live BF16 under
`.../qwen38-27b/bf16/`). Peak RSS of the 10-tensor + merkle pass: 432 MB.
Full unfused pass wall 83.56 s.

| set | n | min | p10 | median | max |
|---|---:|---:|---:|---:|---:|
| unfused Q4 vs BF16 | 306 | **0.9894979639251519** | 0.99351682 | 0.99397953 | 0.99421285 |
| skipped fused `in_proj_{qkvz,ba}` | 96 | — | — | — | — |

argmin: `language_model.model.layers.63.self_attn.k_proj.weight`
(5,242,880 elems, rel_l2=0.14611).
argmax: `language_model.model.embed_tokens.weight` (0.99421285).
`n_below_0.993=17`, `n_below_0.990=1`, `n_below_0.98948=0`, `n_none=0`,
`n_nonfinite=0`.

Named 10-tensor cut (includes embed + lm_head):

| tensor | cosine | rel_l2 | elems |
|---|---:|---:|---:|
| L0 `linear_attn.out_proj` | 0.9935414911194438 | 0.11425 | 31,457,280 |
| L0 `mlp.gate_proj` | 0.9941447925762601 | 0.10874 | 89,128,960 |
| L0 `mlp.up_proj` | 0.9941928218031617 | 0.10829 | 89,128,960 |
| L0 `mlp.down_proj` | 0.99400840 | 0.11000 | 89,128,960 |
| L3 `self_attn.o_proj` | 0.99374735 | 0.11242 | 26,214,400 |
| L3 `self_attn.q_proj` | 0.99395537 | 0.11050 | 26,214,400 |
| L63 `self_attn.o_proj` | 0.99255989 | 0.12271 | 26,214,400 |
| L63 `mlp.down_proj` | 0.99325121 | 0.11682 | 89,128,960 |
| embed | 0.99421285 | 0.10810 | 1,271,398,400 |
| lm_head | 0.99375788 | 0.11230 | 1,271,398,400 |

L0 `out_proj` 0.9935414911194438 matches wave-1
`g1-sparse-exact-islands.md` `weight_cosine` 0.993541491119444 on the
same tensor. Independent recompute.

CLAIMED pack-time min `0.98948` (`qwen38-native-bringup.json`
`correctness.numeric_gate`; `THREE_MODEL_REGIME_SPLIT.json`
`correctness.q4_min_cosine_vs_bf16`) sits 1.8e-5 below this lane’s unfused
min 0.98949796. That is the same number to five decimals. The missing 96
fused tensors are the remaining gap; they are FAIL-None until
`fuse_in_proj_*` is applied (cheapest experiment, §10).

rel_l2 on G0 Q4 is **0.108–0.146**, not ~0. A cosine of 0.993 is an 11%
relative L2. Do not describe G0 Q4 as “lossless”.

### 2.5 What the screen may fail on

Q4-class / HGRAVU01 (attention, embed, lm_head, and any tensor the
candidate claims is the G0 Q4 codec):

- any None / non-finite / n_measured < n_catalog_gemv → FAIL
- min < **0.9890** → FAIL
  (G0 measured min 0.989498; claimed pack min 0.98948; 5e-4 absolute
  slack so a resealed G0 passes; a tensor under 0.9890 is below G0)

MLP mixed codecs (HGRAVB01 / HGRAVR02 / HGRAVS01):

- any None / non-finite → FAIL
- cosine < **0.30** → FAIL (wrong tensor or zero pack)
- **do not pass-certify** at 0.90. mixed-2p0 mean 0.90697 killed tokens.

Output cosine (Wx vs Wq x on captured X) stays a doctor screen. It is not
in the seal’s pass rule. Q3 MLP hold min 0.9679 is ESTIMATED on 6 layers,
generation untested (`g1-doctor-tensor-map.md`).

---

## 3. Artifact identity binds to bytes, not to a path string

### 3.1 MEASURED split

```
labeled_sha("artifact/qwen38-27b/uniform-q4-v1")
  = 56dd65d465f31741f8d40a86d84de779a939fdd9b9b90ecd3d1cb4f82aa4287a
labeled_sha("runtime/ascension_qwen38_hybrid_greedy")
  = ecfc1cac8742d51dac35bca3c702520a7409089914b9dd637d7927baae0cfe72
labeled_sha("genome/Qwen38HybridDecodeSession+qwen_uniform_q4_group64")
  = 688d8b87bddc6baa7bd083229f1b1c7c96ea01adb893c42a98ad534c3341cd7e
sha256(uniform-q4-v1/manifest.json)
  = d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
sha256(mixed-2p0-v1/catalog.hq38m20)
  = 6a16f7fdb58a91925dc154f37f4fa3e2f1364e95bb21a9aed80b61b1a92e52da
```

`GENESIS_LINEAGE_CURRENT.json` `slots.CURRENT.artifact_sha` equals the
**labeled** artifact hash, not the manifest hash. Wave-1 inventory/remeasure
reports that treated those as equal were wrong about the seated file.

### 3.2 Required bind

For a uniform-Q4 artifact:

```
manifest_sha = sha256(manifest.json bytes)
file_sha[i]  = sha256(bytes of tensors/<artifact> named by tensors[i])
merkle       = sha256( concat_i  name_i || 0x00 || hex(file_sha[i]) || 0x0a )
artifact_content_sha = sha256( bytes.fromhex(manifest_sha) || bytes.fromhex(merkle) )
```

MEASURED on live G0 (755 files, wall 9.21 s):

```
merkle               = c33d59d8811669760eaf6c27a39338f855fce97a48563b2bcab00c2e310c9641
artifact_content_sha = f590664c259cbea8fe90889e06e2f78f09c57f03f34f97b26635e524e5e06b5e
embed file sha256    = 8bf67af581661b60dced866143eed48ae14fe7d5d15ab6dd64bc856268aaed62
lm_head file sha256  = e86b8de6f6df4207d128e482a3d5955e01dc3614b3538145d23c3cdfe56313ba
```

Replacing one tensor file changes `file_sha`, `merkle`, and
`artifact_content_sha`. Replacing the path string does not.

For HQ38M20:

```
catalog_sha = sha256(catalog.hq38m20 bytes)          # already on mixed-2p0 candidate json
segment_sha[j] = sha256(segments/<file> bytes)
merkle = sha256( concat  catalog_sha || segment_id || segment_sha )
```

mixed-2p0 catalog sha is already
`6a16f7fdb58a91925dc154f37f4fa3e2f1364e95bb21a9aed80b61b1a92e52da`
(candidate json `catalog.sha256` and this lane’s recompute agree).

Runtime / kernel:

```
runtime_sha       = sha256(executable file bytes)     # not labeled_sha("runtime/...")
kernel_genome_sha = sha256(concat metal sources in dispatch-name order
                           || 0x00 || each dispatch name)
```

Wave-1 remasure already had content hashes for the live body
(resident `ae0bc8defd84…`, Q4 shader `51abdf7be388…`). Those are the
shape. They are not what lineage stores.

Promotion clause `artifact_identity_exact` today
(`promotion.py:167-176,` `_computed_artifact_sha`):
`sha256(receipt.preimage)` must equal `child.artifact_sha`. Change the
allowed preimage to `{manifest_or_catalog_bytes, merkle_receipt}` and
require `child.artifact_sha == artifact_content_sha`. A string starting
`hawking.lineage/` is FAIL.

`CLAUSE_RUNTIME_GENOME` today compares two asserted hex strings. Require
`genome.runtime_preimage` / `kernel_preimage` bytes (or a list of file
sha256s) and recompute.

---

## 4. Generation checks that separate coherent from fluent nonsense

### 4.1 Why greedy parent==child is the wrong single bar

`CLAUSE_GREEDY_TOKEN_IDS` (`promotion.py:55,` MIN_GREEDY_PROMPTS=3) requires
`parent_ids == child_ids` on three prompts. Two failures of that rule:

1. **Too weak.** Evidence is caller-supplied. Fixture IDs are
   `[383, 1024, 17, 88]` (`testing.py:30-34`), which are not G0’s France
   prefix `[248068, 198, 760, …]`. Forged matching IDs PASS without the
   model running. Protected-test rows are asserted `"PASS"` strings
   (`REQUIRED_PROTECTED_TESTS`, `promotion.py:75-79`).
2. **Too strong for a new pack.** G1 may change representation of the
   same BF16 parent. A coherent lower-BPW pack can drift greedy ids and
   still be the same model. Exact 32-id match is required only when
   `artifact_content_sha` equals G0’s (genome-only child).

`tools/coherence_gate.py` is the same id-identity idea. Its own docstring
says a 12-token single-prompt check cannot certify lm_head / embed /
sampling changes. `QWEN38_COHERENCE_SEAL.json` is that 12-token seal.

### 4.2 Classes

Operate on `new_token_ids` + decoded text of a **native** generate.
Do not use the MLX overwrite receipt.

| class | rule (first match) |
|---|---|
| INCOHERENT | n=0; or unique≤2 and n≥8; or top_frac≥0.70 and n≥8; or newline_frac≥0.50; or punct_only_frac≥0.85; or unigram entropy < 1.0 bit and n≥8 |
| ECHO | stripped text equals the prompt, or is a ≥8-char prefix of the prompt |
| FLUENT_NONSENSE | class would be COHERENT on degeneracy, but a tasked prompt misses the required token (`paris`, `323`) |
| DEGRADED | tasked code prompt does not address the task (no reverse/slice) but is not collapse |
| COHERENT | none of the above |

These thresholds were calibrated on two sealed receipts this lane, not on
a sweep. mixed-2p0 hits INCOHERENT on rule 1 before any task check. G0
oracle-32 hits COHERENT (unique 29/32, entropy 4.8125 bits, newline_frac
0.031). A grammatical France answer without “Paris” is the fluent-nonsense
case the degeneracy rules miss — that is why the task tokens exist.

### 4.3 Required probe set

Render with the hashed tokenizer + the same chat template the resident
uses. Seal `prompt_ids` so a template drift is visible.

| id | prompt | pass condition |
|---|---|---|
| P1 | `Say hi.` | If `artifact_content_sha == G0`: 32-id exact match to oracle-32 below. Else: class COHERENT, not ECHO. |
| P2 | `What is the capital of France?` | class COHERENT **and** `paris` in text (case-insensitive) |
| P3 | `What is 17 times 19? Reply with the integer product, then one short sentence showing the arithmetic. No other preamble.` | class COHERENT **and** `323` in text |
| P4 | `Write a function that reverses a string.` | class COHERENT (not ECHO); text addresses reverse |
| P5 | `def fibonacci(n):` | class COHERENT (catches punct collapse) |
| P6 | `The three primary colors are` | class COHERENT (catches punct collapse) |

`max_new_tokens ≥ 32`. 12 tokens is not enough for P2/P3.

G0 oracle-32 for P1 (MEASURED live, 6/6, `g1-baseline-remeasure.md`;
prefix equals `QWEN38_COHERENCE_SEAL.json` 12-id):

```
[248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149,
 1061, 369, 264, 1546, 4145, 11, 2050, 1622, 13, 353, 3172, 1066, 1910,
 15131, 303, 264, 11321, 11, 5629, 1560]
```

Text (truncated at 32, as designed):

```
<think>
The user simply wants me to say "hi." This is a very simple, direct request. I'll just say hi in a friendly, natural way
```

G0 P3 is MEASURED on the same body (`g1-baseline-remeasure.md` §5b):
`ok=true fallbacks=0 n_new=168`, body contains `323` and
`17 × 19 = 17 × (20 − 1) = 340 − 17 = 323`.

G0 P2/P4/P5/P6 to 32+ tokens are **not** sealed as full text this lane.
The 12-id France seal starts `<think>` and is not a Paris certificate.
Cheapest experiment: one native 64-token generate on P2/P4/P5/P6 when
the GPU lock is free. Do not claim G0 passes P2 until that receipt exists.

Vehicle constraints (all FAIL if violated):

- `fallbacks == 0`
- `dense_w_materialized == 0`
- loader path is the native reader (`catalog.hq38m20` → mixed arms;
  else uniform-Q4). Expand-to-Q4 / MLX overwrite is FAIL.
- `load_count` unchanged across the probe set (no silent reload).

### 4.4 Classifier on mixed-2p0 (MEASURED this lane, no GPU)

Receipt: `receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json`
(copied to `/tmp/QWEN38_NATIVE_MIXED_2P0_GENERATE.json`).
`lane=qwen38-coherence-generate`, `fallbacks_total=0`,
`dense_w_materialized_total=0`. Native path, not the MLX overwrite.

| prompt | class | reasons | top_id | top_frac | unique | text |
|---|---|---|---:|---:|---:|---|
| Say hi. | INCOHERENT | degenerate_cycle | 198 | 1.00 | 1 | 16× newline |
| Write a function that reverses a string. | INCOHERENT | majority_token_collapse | 1076 | 0.75 | 3 | `......)...)...` |
| What is the capital of France? | INCOHERENT | degenerate_cycle | 198 | 0.94 | 2 | 15× newline + `)` |
| Explain what a hash map is in one sentence. | INCOHERENT | newline_collapse | 198 | 0.62 | 3 | newlines + `)..)).` |
| def fibonacci(n): | INCOHERENT | newline_collapse | 198 | 0.69 | 3 | `)))))))))` + newlines |
| The three primary colors are | INCOHERENT | degenerate_cycle | 198 | 0.69 | 2 | `)` + newlines + `))))` |

`n_coherent=0/6`. Gate verdict **FAIL**.

Same artifact’s `GENERATE.json` is a different vehicle
(`engine: mlx_lm_weights_overwritten_from_mixed_pack`, 851 tensors
replaced, 1 new token `<|im_end|>`). The seal must refuse it.

G0 oracle-32 under the same classifier: class **COHERENT**, reasons `[]`,
unique_ratio 0.90625, entropy 4.8125 bits.

---

## 5. Pass thresholds, given the ceilings

| check | PASS | FAIL | ceiling that sets the number |
|---|---|---|---|
| `n_none` on GEMV cosine | 0 | any | G0 catalog is 402 Nones today; that must not PASS |
| Q4-class min weight cosine | ≥ 0.9890 | < 0.9890 | G0 unfused min 0.989498; claimed pack min 0.98948 |
| MLP-mixed min weight cosine | ≥ 0.30 and not None | None or < 0.30 | mixed-2p0 mean 0.907 is **not** a pass bar |
| output cosine | recorded, never decides | — | Q80 generated at 0.768 holdout; 0.8604 bar is dead |
| vehicle | native, fallbacks=0, dense_w=0 | expand / MLX overwrite / fallback | wave-1 confound |
| P1–P6 class | all COHERENT | any INCOHERENT / ECHO / FLUENT_NONSENSE | mixed-2p0 is 6× INCOHERENT; G0 P1/P3 MEASURED COHERENT |
| P2 token | contains `paris` | missing | fluent-nonsense discriminator |
| P3 token | contains `323` | missing | remasure G0 emits 323 |
| oracle-32 | exact iff same `artifact_content_sha` as G0 | mismatch on a genome-only child | remasure 6/6 |
| derived `coherence` | 1.0 only if 6/6 COHERENT | < 1.0 | do not assign 1.0 |
| derived `engineering` | 1.0 only if fallbacks=0 and native | else 0.0 | mixed-2p0 native has fallbacks=0 and still FAIL generation — engineering 1.0 does not save it |
| `capability_ge_parent_contract` | child derived ≥ parent **measured** derived | child 1.0 from default | parent must be resealed; default contract deleted |

A child may not lower the parent floor via `evidence.parent_contract`
(already hardened). After this change the parent floor is the resealed
G0 derived map, not `{1,1,1}`.

`complete_token_discipline` stays a promotion timing clause, not a
capability invention. Do not default it to 1.0.

---

## 6. Prove the gate works: it must FAIL mixed-2p0-v1

A gate never observed failing is not evidence. mixed-2p0-v1 is the
negative control: native generate, 0 fallbacks, 0 dense-W, 2.0856 BPW,
attention still 4.250, MLP 0.848, down_proj 0.1316, mean component cosine
0.907, tokens are newlines and `)`.

### 6.1 What today’s gate does with it

Honest ids vs G0 P1 would FAIL `greedy_token_ids_agree`
(`[198]×16` ≠ `[248068, 198, 760, …]`). That is accidental. The capability
clause would still **PASS** if someone copies
`DEFAULT_CAPABILITY_CONTRACT`. The cosine field is not consulted. The
artifact clause PASSES on `sha256("hawking.lineage/artifact/…")`.
Protected tests PASS as strings. There is no fixture that loads
`QWEN38_NATIVE_MIXED_2P0_GENERATE.json`.

mixed-2p0 decode wall is ~1.55 s / 16 tokens ≈ 97 ms/token (CLAIMED from
that receipt) vs G0 39.3 ms, so `complete_token_wall_improves_materially`
would FAIL on real walls. That is a speed fail, not a capability fail.
A faster incoherent child would still be a capability PASS.

### 6.2 Required tests (must stay red)

File: `lab/tests/test_qwen38_capability_gate.py`.

1. `test_mixed_2p0_v1_is_capability_reject`
   - load `receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json`
   - bind `catalog_sha256 == 6a16f7fdb58a91925dc154f37f4fa3e2f1364e95bb21a9aed80b61b1a92e52da`
   - every prompt class == INCOHERENT
   - `evaluate_capability_seal(...).verdict == "FAIL"`
   - `derived_capability` absent or coherence == 0.0
   - `evaluate_promotion` with those ids and with
     `capability={1,1,1}` still REJECT on `generation_class_coherent`
   - no GPU

2. `test_g0_oracle32_and_323_are_capability_pass`
   - sealed remasure ids + 323 text → COHERENT
   - no GPU

3. `test_none_cosine_fold_is_fail_not_one`
   - 402× `cosine: None` → status FAIL, min is null, never 1.0
   - live G0 manifest reproduces this FAIL until resealed

4. `test_labeled_sha_is_not_artifact_identity`
   - preimage `hawking.lineage/artifact/qwen38-27b/uniform-q4-v1` is FAIL
   - `56dd65d4… != d650a757… != f590664c…`

5. `test_fluent_nonsense_france_without_paris_is_fail`
   - ids with high unique ratio, text
     `"The capital of France is a major European cultural centre."`
     → FLUENT_NONSENSE, seal FAIL

6. `test_expand_vehicle_is_fail`
   - `mixed-2p0-v1/GENERATE.json` `engine` contains
     `mlx_lm_weights_overwritten` → FAIL regardless of tokens

7. `test_g0_current_manifest_min_q4_cosine_is_unmeasured`
   - reading live `min_q4_cosine==1.0` with 402 Nones is FAIL
   - this is the “gate observed failing on the seated artifact’s
     paperwork” proof, complementary to mixed-2p0

If test 1 is deleted or marked xfail, CI is red. Put it in
`lab/tests/test_genesis_promotion_gate_adversarial.py` as well so the
adversarial file keeps its “every new clause watched going red” contract.

### 6.3 What a later GPU lane adds (not this lane)

One native generate of P1–P6 on mixed-2p0 is already on disk. Do not rerun
it. A G0 reseal of P2/P4/P5/P6 to 64 tokens is the only missing positive
control. Serialized GPU lane. Do not touch the resident.

---

## 7. Exact files and functions to change

This lane does not edit them. The implementer does, in this order.

### 7.1 New

| path | what |
|---|---|
| `lab/lineage/capability_seal.py` | `CapabilitySeal`, `fold_min_or_none`, `classify_generation`, `evaluate_capability_seal`, `derive_capability_floats`. None→FAIL. `hawking.lineage/` preimage→FAIL. |
| `lab/operators/qwen38_dequant_cosine.py` | streaming dequant vs BF16; `catalog_merkle`; call `fuse_in_proj_qkvz` / `fuse_in_proj_ba` for the 96 fused names (port or FFI). |
| `lab/operators/qwen38_generation_class.py` | P1–P6 table, class rules §4.2, oracle-32 constant. |
| `lab/tests/test_qwen38_capability_gate.py` | the seven tests in §6.2 |
| `tools/qwen38_capability_gate.py` | CLI `seal|verify|fail-control`. `fail-control` runs test 1 and exits 1 if mixed-2p0 ever PASSes. |

### 7.2 Change

| path | function / site | change |
|---|---|---|
| `lab/lineage/identity.py` | `DEFAULT_CAPABILITY_CONTRACT` `:28-31` | delete the 1.0/1.0/1.0 assignment. Replace with no default. Construction without a seal is IdentityError. |
| `lab/lineage/identity.py` | `make_qwen38_genesis` `:240-260` | `artifact_sha=artifact_content_sha` (f59066… once resealed); `runtime_sha`/`kernel_genome_sha` from file bytes; `capability` from seal. |
| `lab/lineage/canon.py` | `labeled_sha` `:29-30` | keep for tests; document as banned in production identity. Add `content_sha256`, `catalog_merkle`. |
| `lab/lineage/promotion.py` | `ALL_CLAUSES` `:57-73` | add `dequant_cosine_none_is_fail`, `generation_class_coherent`, `native_vehicle_only`. Strengthen `artifact_identity_exact` to content preimage. Split `greedy_token_ids_agree`: exact match only when `artifact_content_sha` equals parent. |
| `lab/lineage/promotion.py` | `_computed_artifact_sha` `:167-176` | reject `hawking.lineage/` strings; require manifest+merkle bytes. |
| `lab/lineage/promotion.py` | `CLAUSE_CAPABILITY` handler | compare derived floats from `evidence["capability_seal"]`, not `child.capability` asserted map. |
| `lab/lineage/testing.py` | `artifact_preimage_for` `:37-42`, `GREEDY_PAIRS` `:30-34`, `passing_evidence` | content preimage; real G0 oracle-32 / 323 text; no labeled_sha child identity. |
| `lab/tests/test_genesis_promotion_gate.py` | happy path + `_artifact_swap` | happy path must carry a seal; labeled preimage no longer ACCEPTs. |
| `lab/tests/test_genesis_promotion_gate_adversarial.py` | add mixed-2p0 case | test 1 from §6.2 |
| `crates/hawking-core/src/model/qwen38_pack.rs` | `try_reuse_q4` `:290-313` | do not write `cosine: None`. Recompute via `decode_uniform_q4_group64` + parent, or refuse to emit `min_q4_cosine`. |
| `crates/hawking-core/src/model/qwen38_pack.rs` | `pack_q4_named` reuse `:403` | same. |
| `crates/hawking-core/src/model/qwen38_pack.rs` | min fold `:680-684` | None → error / `min_q4_cosine: null` + `min_q4_cosine_status: "UNMEASURED"`. Never fold 1.0. |
| `crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs` | `pack_quality` `:278-314`, `decode_uniform_q4_group64` `:330-344` | **do not change the formula**. Call it from reseal. |
| `crates/hawking-core/src/model/qwen38_geometry.rs` | `fuse_in_proj_qkvz` `:291`, `fuse_in_proj_ba` `:360` | expose to the sealer for the 96 fused names. |
| `tools/genesis_seat.py` | `seat` `:33-50` | refuse to seat without a PASS seal; write content hashes. |
| `tools/agentos/genesis_body/src/main.rs` | `artifact_manifest_sha` `:394-404`, load `:839-848` | bind merkle, not just manifest; a labeled CURRENT sha is a hard error, not a “stale; loading anyway” log. |
| `tools/coherence_gate.py` | whole file | keep as id-identity subroutine; do not call it the capability gate. |

### 7.3 Do not change

- receipts (any)
- AgentOS scaffolding, HCLI UI, World State, HIDE, packaging
- residency / workers / scheduler / research bus
- Q80 / DSV4F vehicles (transfer science only: generation is the
  certificate, organ cosine is a screen, 0.8604 is dead)
- the live resident process
- G0 catalog bytes (reseal writes a new `capability_seal.json` beside
  them, and a corrected `min_q4_cosine` only if the packer is rerun)

### 7.4 Implementation sequence

1. Land `capability_seal.py` + `qwen38_generation_class.py` + tests 1,3,4,5,6.
   mixed-2p0 FAIL is red on HEAD the day the test exists. That is the proof.
2. Land `qwen38_dequant_cosine.py` + test 3 against the live G0 manifest
   (UNMEASURED FAIL) and against the 306-row reseal (min 0.989498 PASS).
3. Port fuse, score the 96, close n_none on G0.
4. Wire `evaluate_promotion` to the seal. Kill `DEFAULT_CAPABILITY_CONTRACT`.
5. Retarget `make_qwen38_genesis` / `genesis_seat.py` / genesis-resident bind.
6. Reseal G0 P2/P4/P5/P6 on the serialized GPU lane. Then test 2 expands.

---

## 8. KILLS / REOPEN_IF

| mechanism | verdict | REOPEN_IF |
|---|---|---|
| `DEFAULT_CAPABILITY_CONTRACT` 1.0/1.0/1.0 as a seated measurement | KILLS | never; floats must be derived from a seal |
| `labeled_sha` as `artifact_sha` / `runtime_sha` / `kernel_genome_sha` | KILLS for identity | never; labels may remain test helpers |
| `min_q4_cosine` fold seeded at 1.0 over None | KILLS | never |
| organ / weight cosine as a GO certificate | KILLS | a new receipt where cosine predicts generate class on Qwen3.8 *and* mixed-2p0 is explained |
| 0.8604 Q80 residual-identity bar transferred to Qwen3.8 | KILLS | — |
| exact 32-id match as the only generate bar on a new pack | KILLS | the child is genome-only (`artifact_content_sha` equals G0) |
| expand-to-float / expand-to-Q4 generate as capability evidence | KILLS | a complete-token measurement shows the expansion is a net physical win (wave-1 standing rule) |
| 12-token single-prompt id seal as capability | KILLS | — |
| generator+residual family as a G1 candidate | KILLS (wave 1, not this lane) | new evidence the residual quantizes better |

---

## 9. What this lane did not do

- No GPU generate. mixed-2p0 / G0 P1 / G0 P3 used sealed receipts.
- Did not score the 96 fused `in_proj_*` Q4 tensors (need `fuse_in_proj_*`).
  Cheapest experiment: port the two fuse functions and rerun the 83 s
  scanner. Until then n_none on a full 402-row G0 seal is 96 and the
  seal is FAIL. That is correct.
- Did not reseal G0 P2/P4/P5/P6 to 64 tokens. 12-id France is not Paris.
- Did not dequant HGRAVB01/R02/S01 on mixed-2p0 (native generate already
  kills it; cosine would be a screen, not the verdict).
- Did not edit any tracked file except this one.

---

## 10. Evidence appendix (command output, not paraphrase)

### 10.1 Labeled vs content hashes + G0 None-fold + mixed codec census

```
labeled artifact 56dd65d465f31741f8d40a86d84de779a939fdd9b9b90ecd3d1cb4f82aa4287a
labeled runtime  ecfc1cac8742d51dac35bca3c702520a7409089914b9dd637d7927baae0cfe72
labeled genome   688d8b87bddc6baa7bd083229f1b1c7c96ea01adb893c42a98ad534c3341cd7e
manifest sha     d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df bytes 238879
mixed catalog    6a16f7fdb58a91925dc154f37f4fa3e2f1364e95bb21a9aed80b61b1a92e52da bytes 158970
mixed ver 1 n_tensors 851 n_segments 66
by_codec {'HGRAVU01': 659, 'HGRAVB01': 64, 'HGRAVR02': 64, 'HGRAVS01': 64}
G0 q4 402 none 402 present 0 fold 1.0 field 1.0
```

### 10.2 Streaming dequant + merkle + classifier

```
COS out_proj ...layers.0.linear_attn.out_proj.weight  0.99354149 rel_l2=1.142526e-01
COS gate     ...layers.0.mlp.gate_proj.weight         0.99414479
COS up       ...layers.0.mlp.up_proj.weight           0.99419282
COS down     ...layers.0.mlp.down_proj.weight         0.99400840
COS out_proj ...layers.3.self_attn.o_proj.weight      0.99374735
COS q_proj   ...layers.3.self_attn.q_proj.weight      0.99395537
COS out_proj ...layers.63.self_attn.o_proj.weight     0.99255989
COS down     ...layers.63.mlp.down_proj.weight        0.99325121
COS embed    model.embed_tokens.weight                0.99421285
COS lm_head  lm_head.weight                           0.99375788
F32 raw cosine=-0.5882498371575625 stored_first=0.046875 src_first=1.046875
F32 dlt cosine=1.0 rel_l2=0.0
MIXED CLASS {'INCOHERENT': 6} verdict FAIL
MIN_COS 0.9925598851191002 MAX_COS 0.9942128523312148   # 10-tensor cut
ARTIFACT_CONTENT_SHA f590664c259cbea8fe90889e06e2f78f09c57f03f34f97b26635e524e5e06b5e
MERKLE c33d59d8811669760eaf6c27a39338f855fce97a48563b2bcab00c2e310c9641 wall 9.207s
maximum resident set size 432095232
```

Full unfused scan (`/tmp/g1_capability_gate_full_cosine.json`):

```
n_q4_catalog 402
n_measured 306
n_skipped_fused_or_missing 96
min 0.9894979639251519
argmin language_model.model.layers.63.self_attn.k_proj.weight
p01 0.9914656219054088
p10 0.9935168211827524
median 0.9939795275553678
max 0.9942128523312154
n_below_0.98948 0
wall_s 83.55571129200689
```

### 10.3 mixed-2p0 pack report (CLAIMED fields, path quoted)

`.../mixed-2p0-v1/PACK_REPORT.json`:

```
complete_physical_bpw 2.0855934079220506
mlp_physical_bpw      0.8480504639008466
nonmlp_physical_bpw   4.250142713483966
mean_component_cosine 0.9069688696406788
quality_rows_with_cosine 851
organ_breakdown.mlp_down_proj.physical_bpw 0.13161714918473189
claim_boundary.generation_is_the_gate true
```

### 10.4 Source excerpts

`lab/lineage/identity.py:28-31,245-252` (from `git show HEAD`):

```
DEFAULT_CAPABILITY_CONTRACT: dict[str, float] = {
    "coherence": 1.0,
    "complete_token_discipline": 1.0,
    "engineering": 1.0,
}
...
artifact_sha=labeled_sha("artifact/qwen38-27b/uniform-q4-v1"),
...
capability=dict(DEFAULT_CAPABILITY_CONTRACT),
```

`lab/lineage/canon.py:29-30`:

```
def labeled_sha(label: str) -> str:
    return hashlib.sha256(f"hawking.lineage/{label}".encode("utf-8")).hexdigest()
```

`crates/hawking-core/src/model/qwen38_pack.rs:312,680-684`:

```
cosine: None,
...
let min_q4_cosine = rows
    .iter()
    .filter(|row| row.kind == "q4")
    .filter_map(|row| row.cosine)
    .fold(1.0f64, f64::min);
```

`tools/agentos/genesis_body/src/main.rs:844-848`:

```
if cur.artifact != args.artifact_root || cur.artifact_sha != measured_artifact_sha {
    eprintln!(
        "genesis-resident: lineage identity is stale; loading measured artifact {} sha {}",
        ...
```

---

```
STATUS
IMPLEMENT_READY

CLAIMS
1. Seated G0 capability 1.0/1.0/1.0 is DEFAULT_CAPABILITY_CONTRACT, not a measurement. Evidence: lab/lineage/identity.py:28-31,252.
2. Seated artifact/runtime/kernel hashes are sha256 of path labels, not of bytes. Evidence: canon.py:29-30; identity.py:245-249; this-lane hashes in §10.1; GENESIS_LINEAGE_CURRENT.json artifact_sha=56dd65d4… equals labeled_sha, not manifest d650a757….
3. min_q4_cosine=1.0 is a fold of 1.0 over 402 None. Evidence: qwen38_pack.rs:680-684,312; this-lane count 402/402 None.
4. G0 Q4 weight-cosine ceiling vs BF16 is min 0.989498 (L63 k_proj) on 306 unfused tensors; claimed pack min 0.98948 is the same number to five decimals. Evidence: /tmp/g1_capability_gate_full_cosine.json; qwen38-native-bringup.json correctness.numeric_gate.
5. Weight cosine is a screen. mixed-2p0 mean_component_cosine 0.90697 and still native-INCOHERENT. Evidence: PACK_REPORT.json; QWEN38_NATIVE_MIXED_2P0_GENERATE.json; this-lane classifier 6/6 INCOHERENT.
6. None must be FAIL. A resealed G0 Q4-class min below 0.9890 is FAIL. MLP mixed cosine must not pass-certify at 0.90. Evidence: §2.5, §5.
7. Artifact identity must be sha256(manifest_or_catalog) bound to a manifest-ordered merkle of file bytes. G0 content sha MEASURED f590664c…. Evidence: §3.2, §10.2.
8. Generation certificate is class(COHERENT) + task tokens (paris, 323) on a native reader. Exact oracle-32 only when artifact_content_sha equals G0. Evidence: §4; remasure 6/6 + 323; Q80 bar transfer.
9. The replacement is proven by a required test that FAILs mixed-2p0-v1 on sealed native generate, and FAILs the live G0 manifest’s None-fold. Evidence: §6.2 tests 1 and 7.
10. Exact edit list is §7. First land lab/tests/test_qwen38_capability_gate.py so mixed-2p0 is observed failing.

EVIDENCE
- /tmp/g1_capability_gate_measure.json
- /tmp/g1_capability_gate_full_cosine.json
- /tmp/QWEN38_NATIVE_MIXED_2P0_GENERATE.json  (git show HEAD:receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json)
- workspace/superwave/g1/g1-baseline-audit.md §1 (None-fold, labeled_sha)
- workspace/superwave/g1/g1-baseline-remeasure.md §5 (oracle-32, 323)
- workspace/superwave/g1/g1-artifact-inventory.md §5.3 (mixed-2p0 loadable, native INCOHERENT)
- receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json
- receipts/ascent-2026-08-16/GENESIS_PROMOTION_GATE_HARDENING.json
- receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json
- receipts/ascent-2026-08-16/q80-recalibrate-capability-bar.SUMMARY.json
- live artifacts under /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/

CHANGES
- created workspace/superwave/g1/g1-capability-gate.md
- no other tracked path

TESTS
- test -s / wc -l / git status --porcelain: see session completion message

RISKS
- 96 fused in_proj tensors still unscored; a seal that ignores them reintroduces None→1.0. Spec requires FAIL until fused.
- G0 P2/P4/P5/P6 full text not sealed; do not claim G0 passes Paris.
- Classifier thresholds calibrated on two artifacts; a fluent-nonsense model that also names Paris and 323 would still need a wider probe. Name that as a reopen, do not pretend six prompts are lm_head certification (Q80 Q4 already flips 11–13 ids).
- Implementer must materialize lab/lineage (not in this sparse checkout) before editing it.

UNRESOLVED
- True min over all 402 Q4 tensors (96 fused pending fuse_in_proj_*).
- G0 64-token text for P2/P4/P5/P6 (serialized GPU lane).
- mixed-2p0 per-tensor (not mean) component cosine. Not needed for the FAIL; native generate already kills it.
- Whether a coherent pack between 2.0856 and 4.2527 BPW exists. Out of scope; this gate is how that bisection becomes honest.

NEXT
1. Implement §7.4 step 1: tests that FAIL mixed-2p0 and FAIL the None-fold, no GPU.
2. Stream-reseal 306 unfused G0 cosines into capability_seal.json; then the 96 fused.
3. Delete DEFAULT_CAPABILITY_CONTRACT; reseat G0 from the seal.
4. Serialized GPU lane: G0 P2/P4/P5/P6 64-token seal.
```
