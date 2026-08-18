# G1 Tabula / Gravity provenance genome

Lane: `69-tabula-genome`. Spec only. No GPU. No generate. No pack. No resident touch.
Every number is `MEASURED` (this process), `RECEIPT` (quoted field), `SOURCE` (file:line), or `UNBOUND` (named, cheapest experiment attached).

STATUS: IMPLEMENT_READY.

---

## 0. Verdict

Today's lineage identity is naming, not provenance.

- Git-tracked G0 (`git show HEAD:receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json`) stores `artifact_sha` / `runtime_sha` / `kernel_genome_sha` as `labeled_sha(path-string)`. `MEASURED` below.
- Live G0 (`/Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json`) rebound those three fields to `sha256(manifest.json)`, `sha256(genesis-resident)`, `sha256(qwen_uniform_q4.metal)`. Still not a merkle of weight bytes. Capability is still the assigned `{1,1,1}` contract. No Tabula. No Gravity. `research_state` is null.
- Packed tensor filenames are `sha256(tensor_name)`, not `sha256(file bytes)`. `MEASURED` below.
- Promotion "computes" artifact identity from a UTF-8 label preimage, not from artifact bytes. `SOURCE` `lab/lineage/promotion.py:193-200` + `lab/lineage/testing.py:35-42`.

A G1 that changes packing (Gravity) or weights/behavior (Tabula) cannot be attributed unless those are separate sealed documents, each hashed over the bytes it claims, with a mechanical invariant that a write to one cannot rewrite the other.

This file specifies those two documents, the content-hash binding, the non-breaking attachment to `hawking.lineage.genesis_instance.v1`, the G0 migration, and the files/functions that must change.

---

## 1. What is seated today

### 1.1 Two G0 records exist

| copy | path | `CURRENT.artifact_sha` | `CURRENT.runtime_sha` | `CURRENT.kernel_genome_sha` | `CURRENT.complete_token_ns` |
|---|---|---|---|---|---:|
| git HEAD | `receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json` | `56dd65d465f31741…aa4287a` | `ecfc1cac8742d51d…ae0cfe72` | `688d8b87bddc6baa…3341cd7e` | 35227918 |
| live box | `/Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json` | `d650a757c4cffed4…fc47b9df` | `ae0bc8defd84a8a1…f84d32c2` | `51abdf7be388d62b…09d3efcd1` | 37879375 |

This worktree does not materialize the receipts path. Git value is from `git show HEAD:…`. Live value is from a `json.loads` of the box file. `MEASURED` 2026-08-17.

Live events: `install` + `snapshot_lkg` at `2026-08-17T13:10:59Z`, then `bind_live_observed` (`schema: hawking.lineage.live_binding.v1`) at `2026-08-17T13:15:23Z`. `RECEIPT` live file `events`. CITED `workspace/superwave/g1/g1-resident-harvest.md:240`.

Live `LAST_KNOWN_GOOD` was **not** rebound. It still carries the labeled hashes and `complete_token_ns=35227918`. CURRENT and LKG of the same `instance_id` / generation therefore disagree on what the three sha fields mean. That is already a silent coupling bug: a rollback would restore labeled hashes into a resident that compares them to `sha256(manifest.json)`.

### 1.2 Git hashes are labeled path strings

`SOURCE` `lab/lineage/canon.py:29-30`:

```
def labeled_sha(label: str) -> str:
    return hashlib.sha256(f"hawking.lineage/{label}".encode("utf-8")).hexdigest()
```

`SOURCE` `lab/lineage/identity.py:240-248` `make_qwen38_genesis`:

```
artifact_sha=labeled_sha("artifact/qwen38-27b/uniform-q4-v1"),
runtime_sha=labeled_sha("runtime/ascension_qwen38_hybrid_greedy"),
kernel_genome_sha=labeled_sha("genome/Qwen38HybridDecodeSession+qwen_uniform_q4_group64"),
```

`MEASURED` this process:

```
56dd65d465f31741f8d40a86d84de779a939fdd9b9b90ecd3d1cb4f82aa4287a  artifact/qwen38-27b/uniform-q4-v1
ecfc1cac8742d51dac35bca3c702520a7409089914b9dd637d7927baae0cfe72  runtime/ascension_qwen38_hybrid_greedy
688d8b87bddc6baa7bd083229f1b1c7c96ea01adb893c42a98ad534c3341cd7e  genome/Qwen38HybridDecodeSession+qwen_uniform_q4_group64
aa6990ca04bcb26dd53a0ccc13b27b2275a2b2e3b4965ba0c6a8274a3e2e38dd  bench/complete-token/qwen38/greedy/3prompt/gpu-cb-timestamps
```

These equal git `CURRENT.*_sha` and `benchmark_fingerprint`. They are not hashes of any file on disk.

### 1.3 Live hashes are still not packed-byte merkeles

`MEASURED` `shasum -a 256` of live `uniform-q4-v1/manifest.json`:

```
d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
```

Equals live `CURRENT.artifact_sha` and live `identity.artifact_sha_authority = "sha256(manifest.json bytes)"`. The manifest lists 755 tensors by **name-hash filename** and `bytes`; it does not carry per-file content sha (`MEASURED` tensor0 keys = `artifact, bytes, cosine, elements, kind, name, shape`).

`SOURCE` packer `crates/hawking-core/src/model/qwen38_pack.rs:270-273`:

```
fn artifact_filename(name: &str, ext: &str) -> String {
    let digest = Sha256::digest(name.as_bytes());
    format!("{:x}.{ext}", digest)
}
```

`MEASURED` on live `layers.0.input_layernorm` (`20488` B `.f32v2`):

```
name    language_model.model.layers.0.input_layernorm.weight
name_sha   b562b986ad0a175dff2803bfe05a6582e815edf279aa777207d81f32556d9630
content_sha 4487e59fb774e13c79f8668afb1f49f7e41f268118653d52bc55192857847de3
equal   False
```

Same constructor in `tools/qwen38_sub15_pack.py:109-110` (`artifact_filename`). Sub15 packed blobs are named the same way.

Live `runtime_sha` is `identity.resident_executable_sha256` of `workspace/ops/build/rust/release/genesis-resident`. Live `kernel_genome_sha` is `identity.kernel_source_sha256` of **one** shader (`crates/hawking-core/shaders/qwen_uniform_q4.metal`). A Qwen3.8 token is 964 dispatches across multiple shader families (`SOURCE` `workspace/superwave/g1/g1-kernel-inventory.md:50-70`). One-file shader hash is not a kernel genome.

### 1.4 Capability is assigned, not measured

`SOURCE` `lab/lineage/identity.py:28-32`:

```
DEFAULT_CAPABILITY_CONTRACT = {
    "coherence": 1.0,
    "complete_token_discipline": 1.0,
    "engineering": 1.0,
}
```

Both git and live `CURRENT.capability` are that map. Promotion `CLAUSE_CAPABILITY` compares these floats (`lab/lineage/promotion.py` `_capability_losses`). It never reads an oracle receipt.

Live measured capability (different document): 6/6 oracle-32 match + `17*19=323`. `RECEIPT` `workspace/superwave/g1/g1-baseline-remeasure.md:14,268`. Not stored on the instance.

Live measured TOKEN_NS median of 6 paired reps: `39,326,090` → TPS `25.4284`. `RECEIPT` same file:12-13. Lineage CURRENT still holds `37,879,375` (DIRTY wall). LKG still holds `35,227,918` (`GENESIS_COMPLETE_TOKEN_NS`). Three numbers, one generation, no hash-kind discriminator.

### 1.5 Resident already disagrees with git lineage

`SOURCE` `tools/agentos/genesis_body/src/main.rs:394-401` `artifact_manifest_sha` = `sha256(manifest.json bytes)`.

Startup (`:839-868`): if `lineage.artifact_sha != measured`, log `"lineage identity is stale"` and load anyway, storing the **measured** sha on the body.

Reload (`:508-521`): **requires** `artifact_sha == sha256(manifest.json)` or returns error. `maybe_reload_lineage` (`:553-563`) then sets `last_reload_error`.

Against git lineage this is a permanent stale/reload-fail loop. Against live lineage startup matches and reload is allowed — but only because `artifact_sha` was overwritten to mean manifest bytes, while LKG and `make_qwen38_genesis()` still mean labeled paths.

`tools/genesis_seat.py:33-40` `seat()` rebuilds from `make_qwen38_genesis()` and `LineageState.to_dict()`. Running it again would wipe the live bind and restore labeled hashes. **KILLS re-seat as a migration step.**

### 1.6 Promotion hashes a label, not bytes

`SOURCE` `lab/lineage/testing.py:35-42` `artifact_preimage_for`:

```
return f"hawking.lineage/{label}"   # e.g. hawking.lineage/artifact/child-g1
```

`SOURCE` `lab/lineage/promotion.py:193-200` `_computed_artifact_sha` sha256s that UTF-8 string. Clause `artifact_identity_exact` (`:429-473`) then requires `child.artifact_sha == measurement.artifact_sha == receipt.sha == sha256(preimage)`. Passing the gate proves the label is self-consistent. It does not prove the packed bytes exist.

`CLAUSE_RUNTIME_GENOME` (`:497-514`) compares asserted `evidence.genome.runtime_sha` / `kernel_genome_sha` to the child's fields. No binary or shader bytes are read.

### 1.7 What already reads the instance

| reader | function | fields used | extra keys |
|---|---|---|---|
| `tools/genesis_seat.py` `seat` | `make_qwen38_genesis` + `LineageState.install` + `to_dict` | all first-class | **wiped** |
| `tools/genesis_seat.py` `status` | `json.loads` | generation, instance_id, representation_bpw, complete_token_ns, tps | ignored |
| `tools/genesis_peek.py` `main` | `json.loads` `slots` | generation, representation_bpw, complete_token_ns, tps, live, launched | ignored |
| `tools/ascent_daemon.py` | `slots["CANDIDATE"]` truthiness | CANDIDATE presence only | ignored |
| `genesis_body` `read_lineage_current` | CURRENT.generation, identity.artifact, artifact_sha | those three | ignored |
| `GenesisInstance.from_mapping` | listed fields only `:213-237` | drops unknown top-level (`physical_bpw` on live CURRENT is already dropped on load) | `identity` dict **survives**; `research_state` survives |
| `GenesisInstance.to_dict` / `copy` | `:187-210`, `:167` | emits `identity` as-is | top-level extras dropped |
| `evaluate_promotion` | instance fields + `identity.model` | `CLAUSE_MODEL_IDENTITY` only | other identity keys ignored |
| `normalize_generation` `lab/lineage/continuity.py` | different schema: artifact_sha, runtime_sha, repo_head, physical_bpw, complete_token_ns | unchanged | n/a |
| `validate_message` `lab/lineage/bus.py` | artifact_sha, runtime_sha on the **message** | unchanged | n/a |
| `handover` `lab/lineage/state.py` | sets `successor.research_state = accepted["payload"]` | **replaces** research_state | do not store profiles there |

Non-breaking attachment that survives `from_mapping`/`to_dict`/`copy` **without** editing `identity.py`: string pointers inside `identity`. Top-level new fields are dropped until `from_mapping`/`to_dict`/`copy`/`make_qwen38_genesis` are extended.

`research_state` is the wrong home: G0 is null; handover overwrites it with the eight-key transfer payload (`lab/lineage/transfer.py:17-26`). Extra transfer keys are refused.

---

## 2. The two sciences

| | Tabula | Gravity |
|---|---|---|
| object | behavioral weights and their interventions | physical representation of a fixed Tabula checkpoint |
| current checkpoint | BF16 (or other uncompressed) language tensors | packed catalog the runtime loads |
| change that must not move the other | new abliteration / SFT / merge / base rev | new codec, group size, allocation, reconstruction vehicle |
| G0 fact | `PocketAiHub/Qwen3.8-27B-Abliterated-MLX` from `Qwen/Qwen3.8-27B` @ `1d4bf0f2…` via refusal-direction projection | `uniform-q4-v1` HQ30UQ4 g64 + f32v2, native, 4.252735126866492 BPW |
| G1-relevant fact | same Tabula unless someone swaps the source | `mixed-sub15-v1` is a **different Gravity** of the same Tabula |

Doctor receipts measure the **weights** (what to protect). They live on Tabula. Gravity's per-organ allocation may **cite** a Doctor receipt sha. That is an explicit hash edge, not a field alias.

Packed bytes of a Gravity artifact are **not** Tabula's current checkpoint. If they were, every repack would silently rewrite Tabula. That is the coupling this schema forbids.

---

## 3. Content hash

Schema `hawking.lineage.content_merkle.v1`. Function `lab/lineage/content_hash.py:merkle_listed` (new). Uses `lab/provenance.py:sha256_file` (streaming 1 MiB) + `lab/lineage/canon.py:digest`.

```
merkle = digest({
  "schema": "hawking.lineage.content_merkle.v1",
  "root": "<posix relpath or abs, recorded not hashed as authority>",
  "files": [
    {"relpath": "<posix, sorted>", "sha256": "<sha256_file>", "size": <int>},
    ...
  ]
})
```

Rules:

- Hash file **bytes**, never the path string, never `labeled_sha`.
- Include only files the claim says are loaded or are the checkpoint. G0 Gravity: `manifest.json` + every manifest `tensors[].artifact`. Exclude undeclared `*.f32bin` (353 files, 10_584_840 B, `RECEIPT` `g1-artifact-inventory.md:114-116`).
- Tabula current checkpoint: the BF16 files listed in `bf16/artifact-manifest.json` `files[]` whose path is a language-weight or identity document (`abliteration-manifest.json`, `config.json`, language shards). Vision shards may be listed as `out_of_scope` with their claimed sha; they are not in the G0 pack (`skipped_vision_tensors=333`).
- A claimed sha in a manifest is `CLAIMED` until `sha256_file` matches. This lane verified `abliteration-manifest.json` bytes = `a6e35878e969a319d49570ec266d93762870fefb2737fc2dc4815a2e3380875e`, which matches `artifact-manifest.json`. Shard shas (~50 GiB) are `CLAIMED`, not re-hashed. Cheapest experiment: stream-hash the 11 safetensors + compare.
- `st_dev` / mount device is not an identity key. `RECEIPT` `receipts/q30-startup-latency/LATENCY_GENOME.json` stage `admit_source_chain_device_identity`.

Hash-kind discriminator (required on every sha field that has ever been labeled):

| kind | meaning |
|---|---|
| `hawking.lineage.labeled_path.v1` | `labeled_sha(label)` |
| `hawking.lineage.content_sha256.v1` | `sha256(one file's bytes)` |
| `hawking.lineage.content_merkle.v1` | merkle above |

A field without `*_kind` is `UNBOUND`. Live CURRENT overwrote sha fields in place with no kind. Forbidden going forward.

---

## 4. TABULA_PROFILE

Schema: `hawking.lineage.tabula_profile.v1`

Sealed document. `profile_sha256 = digest(body without profile_sha256 and seal_sha256)`. Then `lab/receipts.py:seal`.

```
{
  "schema": "hawking.lineage.tabula_profile.v1",
  "profile_sha256": "<digest of body>",
  "instance_id": "genesis-qwen38-g0",
  "generation": 0,
  "source_checkpoint": {
    "repo": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "content_merkle": "<UNBOUND | merkle>",
    "content_merkle_kind": "hawking.lineage.content_merkle.v1",
    "evidence": "crates/hawking-core/src/model/qwen38_geometry.rs:15-16 + bf16/artifact-manifest.json metadata.sourceRevision",
    "status": "CLAIMED_REVISION_UNHASHED_TREE"
  },
  "current_checkpoint": {
    "repo": "PocketAiHub/Qwen3.8-27B-Abliterated-MLX",
    "local_path": "workspace/campaign/records/runs/qwen38-27b/bf16",
    "content_merkle": "<UNBOUND | merkle of artifact-manifest files after verify>",
    "claimed_file_shas": "bf16/artifact-manifest.json files[]",
    "abliteration_manifest_sha256": "a6e35878e969a319d49570ec266d93762870fefb2737fc2dc4815a2e3380875e",
    "status": "LOCAL_TREE_PRESENT_SHARDS_CLAIMED"
  },
  "intervention": {
    "type": "ABLITERATION",
    "method": "refusal-direction orthogonal weight projection",
    "direction_sha256": "3958f6bba70f35e869c2918b61a4858a1fa53fbd3c5f31ac594c5b2d87105c51",
    "direction_source_layer": 53,
    "destination_layers": [24, ..., 63],
    "target_kinds": ["full_attention_out", "linear_attention_out", "mlp_down"],
    "scale": 1.0,
    "norm_preserve": true,
    "modified_tensor_count": 80,
    "modified_by_kind": {"full_attention_out": 10, "linear_attention_out": 30, "mlp_down": 40},
    "evidence": "bf16/abliteration-manifest.json + bf16/artifact-manifest.json metadata.abliteration",
    "status": "EVIDENCE_ON_DISK",
    "direction_bytes_verified": false
  },
  "training_or_merge_history": [
    {
      "kind": "BASE_PRETRAIN",
      "who": "Qwen/Qwen3.8-27B",
      "status": "UNBOUND",
      "note": "no local training card; do not invent a mergekit graph"
    },
    {
      "kind": "ABLITERATION",
      "who": "PocketAiHub conversion via mlx-vlm 0.6.8",
      "status": "EVIDENCE_ON_DISK",
      "not_a_merge": true
    }
  ],
  "behavioral_deltas": [
    {
      "axis": "refusal_vs_base_qwen",
      "status": "UNBOUND",
      "cheapest_experiment": "paired greedy ids, same prompts, BF16 current vs a local Qwen/Qwen3.8-27B @ 1d4bf0f2 checkout. GPU lane holds the lock."
    }
  ],
  "capability_deltas": {
    "contract": {
      "coherence": 1.0,
      "complete_token_discipline": 1.0,
      "engineering": 1.0,
      "origin": "ASSIGNED",
      "source": "lab/lineage/identity.py:28-32 DEFAULT_CAPABILITY_CONTRACT"
    },
    "measured": {
      "oracle_32": {"match": "6/6", "status": "MEASURED", "evidence": "g1-baseline-remeasure.md:14,268"},
      "arith_17x19": {"emitted": "323", "status": "MEASURED", "evidence": "same"},
      "status": "NOT_WRITTEN_ON_INSTANCE"
    }
  },
  "doctor_receipts": [
    {
      "id": "qwen38-activation-capture-v1",
      "schema": "hawking.ascension.qwen38_bf16_post_swiglu_activation_capture.v1",
      "path": "workspace/campaign/records/runs/qwen38-27b/activation-capture-v1/capture-result.json",
      "n_tokens": 256,
      "site": "post-norm hidden 5120, not post-SwiGLU 17408, not mixer_x",
      "sha256_self": "fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512",
      "file_sha256": "01db2f814fba99a1b7dac4668e30e20d69247ee3a4efa83b9ce4665718aedcbe",
      "status": "CAPTURED_UNDERDETERMINED",
      "evidence": "g1-doctor-recovery.md:47-77"
    },
    {
      "id": "qwen38-doctor-sensitivity",
      "status": "STRANDED_NOT_ON_HEAD",
      "commit": "7c5f323d9913e6981b19aaa93026db05651bae10",
      "content_sha256_summary": "bc07351926f9a685ae224f3798e563a56e0f811ed3bf1ac554b8409a0d51c83c",
      "evidence": "g1-doctor-recovery.md:15-32"
    }
  ],
  "open_questions": [
    "n_tokens=256 vs in-dim 6144 / 17408 is underdetermined (rows-per-dim 0.0417 / 0.0147). NS-014 analog. Contract context.",
    "mixer_x never captured; out_proj scored in weight space only. g1-out-proj-forensics / campaign context.",
    "1-bit and 2-bit doctor rungs were the same RTN operator; 1-bit floor never measured. Contract context.",
    "No paired generate vs unabliterated base. Abliteration effect on oracle-32 is UNBOUND.",
    "direction_sha256 not re-hashed this lane."
  ],
  "negative_science": [
    {"id": "NS-009", "class": "REFUTED", "why": "synthetic X is not a Tabula or Doctor fit source"},
    {"id": "NS-014", "class": "REFUTED", "why": "do not trust scores from n_fit < rank/dim"},
    {"id": "NS-018", "class": "REFUTED", "why": "decoded-W cache is not a Tabula fact"},
    {"id": "capture-256-underdetermined", "class": "OPEN", "why": "current capture is thinner than NS-014's 0.0449"}
  ],
  "does_not_include": [
    "packed catalog bytes",
    "codec / group / allocation",
    "reconstruction vehicle",
    "runtime binary",
    "Metal shaders"
  ],
  "seal_sha256": "<lab.receipts.seal>"
}
```

Intervention enum (closed): `ABLITERATION | SFT | MERGE | BASELINE | UNKNOWN`. G0 is `ABLITERATION`, not `MERGE`. Evidence on disk; do not upgrade to `MEASURED` until direction bytes and a behavioral delta exist.

Doctor receipt objects on this profile are pointers + content sha, not the 667 kB map. The map stays in its file.

---

## 5. GRAVITY_PROFILE

Schema: `hawking.lineage.gravity_profile.v1`

Same seal rule. `profile_sha256` independent of Tabula.

```
{
  "schema": "hawking.lineage.gravity_profile.v1",
  "profile_sha256": "<digest of body>",
  "instance_id": "genesis-qwen38-g0",
  "generation": 0,
  "tabula_profile_sha256": "<pointer only; Tabula body is not inlined>",
  "representation_genome": {
    "catalog_schema": "hawking.ascent.qwen38_language_uniform_q4.v1",
    "status": "CANDIDATE_QWEN38_LANGUAGE_Q4_FUSED_INPROJ",
    "codecs": [
      {"magic": "HQ30UQ4", "ext": "hq30uq4", "group": 64, "n": 402, "role": "GEMV"},
      {"magic": "f32v2", "ext": "f32v2", "n": 353, "role": "small_vector"}
    ],
    "fusion": ["in_proj_qkvz", "in_proj_ba"],
    "tensors": 755,
    "source_weight_elements": 26895998464,
    "kernel_entry": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "dispatch_count": 964,
    "production_tiles": {"tg": 128, "simdgroups": 4, "rows_per_tg": 2, "threads_per_row": 64},
    "evidence": "qwen38_pack.rs:27-34; g1-kernel-inventory.md:40-56; g1-artifact-inventory.md:163-168"
  },
  "packing_recipe": {
    "packer": "crates/hawking-core/src/model/qwen38_pack.rs",
    "entry": "pack_qwen38_language_uniform_q4",
    "source_dir_field": "request.source_dir (BF16)",
    "output": "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1",
    "filename_rule": "sha256(tensor_name)+ext  # NAME hash, not content",
    "filename_rule_status": "NAMING_NOT_PROVENANCE",
    "packer_content_sha256": "<sha256 of qwen38_pack.rs at the packing HEAD; UNBOUND this lane>"
  },
  "per_organ_allocation": {
    "rule": "UNIFORM",
    "organs": [
      {"class": "mlp.gate_proj", "n": 64, "codec": "HQ30UQ4_g64", "physical_bpw": 4.25000},
      {"class": "mlp.up_proj", "n": 64, "codec": "HQ30UQ4_g64", "physical_bpw": 4.25000},
      {"class": "mlp.down_proj", "n": 64, "codec": "HQ30UQ4_g64", "physical_bpw": 4.25000},
      {"class": "attention_gemv", "n": 402-64*3-2", "codec": "HQ30UQ4_g64", "physical_bpw": 4.25},
      {"class": "embed", "n": 1, "codec": "HQ30UQ4_g64", "physical_bpw": 4.250000251691366},
      {"class": "lm_head", "n": 1, "codec": "HQ30UQ4_g64", "physical_bpw": 4.250000251691366},
      {"class": "small", "n": 353, "codec": "f32v2", "physical_bpw": 32.00853977162764}
    ],
    "complete_physical_bpw": 4.252735126866492,
    "formula": "8 * tensor_payload_bytes / source_weight_elements",
    "doctor_receipt_sha256": null,
    "note": "G0 allocation is not Doctor-driven. A future mixed allocation MUST cite Tabula.doctor_receipts[].sha256 here."
  },
  "reconstruction_recipe": {
    "mode": "NATIVE",
    "expand_to_q4": false,
    "expand_to_float": false,
    "vehicle": null,
    "native_catalog": "manifest.json (uniform-q4), no catalog.hq38m20",
    "loader": "Qwen38HybridWeights::load → load_qwen38_manifest (no load_mixed)",
    "evidence": "qwen38_hybrid_decode.rs:508-513"
  },
  "binding": {
    "artifact_root": "workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1",
    "manifest_sha256": "d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df",
    "manifest_sha256_kind": "hawking.lineage.content_sha256.v1",
    "packed_bytes_merkle": null,
    "packed_bytes_merkle_kind": "hawking.lineage.content_merkle.v1",
    "packed_bytes_merkle_status": "UNBOUND",
    "cheapest_experiment": "stream-hash the 755 listed tensor files + manifest; RSS stays pages of the hasher. Do not numpy-load tensors.",
    "catalog_hq38m20_sha256": null,
    "runtime_bytes_sha256": "ae0bc8defd84a8a1a5cd1c4598224f370c0cfce83a0904e275cbb33df84d32c2",
    "runtime_bytes_kind": "hawking.lineage.content_sha256.v1",
    "runtime_path": "workspace/ops/build/rust/release/genesis-resident",
    "kernel_source_merkle": null,
    "kernel_source_merkle_status": "UNBOUND",
    "kernel_one_file_sha256": "51abdf7be388d62ba080d13a1f97a18ab8b1114c0a6968e9d0f04d109d3efcd1",
    "kernel_one_file": "crates/hawking-core/shaders/qwen_uniform_q4.metal",
    "kernel_one_file_note": "insufficient; token uses qwen80_device_activations / qwen38_device_activations / qwen_next as well"
  },
  "does_not_include": [
    "abliteration direction",
    "BF16 shard bytes",
    "Doctor fit rows",
    "capability contract floats",
    "behavioral deltas"
  ],
  "seal_sha256": "<lab.receipts.seal>"
}
```

### 5.1 G1 candidate Gravity (not seated): mixed-sub15-v1

Same Tabula. Different Gravity. That is the point.

| field | value | label |
|---|---|---|
| complete_physical_bpw | 1.2910781930062503 | RECEIPT `PACK_REPORT.json` / `g1-sub15-native-gap.md:10-13` |
| mlp.gate | HGRAVB01 binary_g128 copied from mixed-2p0 | RECEIPT |
| mlp.up | HGRAVR02 rice_q1_rms@2% copied from mixed-2p0 | RECEIPT |
| mlp.down | HGRAVS01 r160_b3 real-X copied from mixed-2p0 | RECEIPT |
| attention GEMV | rice_q1_rms@2% from BF16 | RECEIPT |
| embed / lm_head | HQ30UQ4 g64 | RECEIPT |
| reconstruction | EXPAND_TO_Q4 vehicle (hardlinked uniform-q4 overwritten with reconstructed Q4) | SOURCE packer docstring L13-15 |
| native catalog.hq38m20 | ABSENT | MEASURED `g1-sub15-native-gap.md:52-53` |
| recorded generate | INCOHERENT on the Q4 vehicle; not a native-codec verdict | RECEIPT `QWEN38_SUB15_INCOHERENT.json` |

`reconstruction_recipe.mode` for a seated G1 of this pack, after the native-load lane, must become `NATIVE` with `catalog.hq38m20` content sha and `expand_to_q4: false`. Until that catalog exists, a generate of this pack is confounded and **cannot** be a Tabula delta.

mixed-2p0 allocation is the cautionary Gravity: attention left at 4.250, `down_proj` crushed to 0.1316. `RECEIPT` `g1-arch-gravity-runs.md:151,169-171`. That is an allocation fact, not a Tabula fact.

---

## 6. Attachment (do not break readers)

### 6.1 Phase 0 — no `identity.py` edit

Sidecar files (new, not inside the instance body):

```
receipts/ascent-2026-08-16/profiles/genesis-qwen38-g0.tabula.v1.json
receipts/ascent-2026-08-16/profiles/genesis-qwen38-g0.gravity.v1.json
```

Pointers as **string** keys on `slots.CURRENT.identity` and `slots.LAST_KNOWN_GOOD.identity` (each slot independently):

```
identity.tabula_profile_sha256
identity.tabula_profile_path
identity.gravity_profile_sha256
identity.gravity_profile_path
identity.artifact_sha_kind
identity.runtime_sha_kind
identity.kernel_genome_sha_kind
```

Why this works: `from_mapping:228` `identity=dict(...)`, `to_dict:201` `identity=dict(...)`, `copy:167` `identity=dict(...)`. Peek, daemon, promotion `identity.model` check, resident `identity.artifact` are unaffected.

Why not top-level fields today: `from_mapping` drops them. Live `physical_bpw` is already such a dropped field.

Why not `research_state`: `handover` replaces it (`lab/lineage/state.py` successor.research_state = accepted payload). Transfer extra keys refused (`transfer.py:17-26`).

Why not rewrite `artifact_sha` in place again: git, LKG, fixtures, and `make_qwen38_genesis` still mean labeled_path. Live CURRENT already did this once. Add kind; do not shift meaning.

### 6.2 Phase 1 — additive `genesis_instance.v1` fields

Same schema string. Optional fields, default null. Edit:

- `lab/lineage/identity.py` `GenesisInstance` dataclass
- `to_dict` `:187-210`
- `from_mapping` `:212-237`
- `copy` `:155-176`
- `make_qwen38_genesis` `:240-265` (keep labeled sha values; set `*_kind` to `labeled_path.v1`; set profile pointers)

Do not bump to `v2`. Readers that ignore unknown keys stay valid. `from_mapping` must start preserving the new fields or Phase 0 identity pointers remain the only durable path.

### 6.3 Transfer / bus / continuity

- `transfer.genomes` is a free-form dict (`testing.py:50`). Nest `genomes.tabula_profile_sha256` and `genomes.gravity_profile_sha256`. Do **not** add required keys to `TRANSFER_PAYLOAD_KEYS` (that is a breaking refuse).
- `MessageType.PROFILE_DELTA` already exists (`bus.py:42`) and already requires a receipt (`:71`). New body contract:
  - `body.profile` ∈ `{TABULA, GRAVITY}` exactly one
  - `body.old_sha`, `body.new_sha`
  - `affected_subsystem` = `tabula` or `gravity`
  - a payload that names both profiles is `BusRefusal`
- `normalize_generation` (`continuity.py`) stays as-is for workers. Additive optional `tabula_profile_sha256` / `gravity_profile_sha256` later; missing is not a refuse in v1.

### 6.4 Promotion clauses (additive, after Phase 1)

Existing 15 clauses stay. New clauses, all `PENDING` when profiles are absent (G0 after Phase 0 may still promote under old rules; G1 must not):

| clause | pass condition |
|---|---|
| `tabula_profile_bound` | child.identity.tabula_profile_sha256 == digest(sidecar minus seal) and sidecar seal verifies |
| `gravity_profile_bound` | same for Gravity; `packed_bytes_merkle` equals recomputed merkle of listed files |
| `science_separation` | mutating a copy of Tabula and resealing leaves gravity sha unchanged, and vice versa (unit test, not a live GPU check) |
| `behavior_attribution` | if greedy ids ≠ parent: exactly one of (tabula_sha, gravity_sha) differs, **or** verdict is `UNATTRIBUTED` and the child is not ACCEPT |
| `artifact_identity_exact` | **redefined**: preimage must be file bytes or a merkle receipt whose files hash. Label preimages (`hawking.lineage/artifact/…`) become `FAIL` with detail `preimage is a path label, not content` |

`lab/lineage/testing.py:artifact_preimage_for` and `passing_evidence` must stop returning the label string once the clause is armed. Until then, fixtures stay green and G1 can still fake identity.

`CLAUSE_REPRESENTATION_BPW` continues to compare `representation_bpw` to Gravity `complete_physical_bpw` via `bpw_key` (6 decimals). Gravity stores the full float; the instance field stays the 6-dec key. Do not silently replace `4.2527` with `4.252735126866492` in the instance — that is a Gravity fact.

---

## 7. Separation invariant

Name: `INVARIANT_SCIENCE_SEPARATION`.

1. `tabula_profile_sha256` is a function of the Tabula document only.
2. `gravity_profile_sha256` is a function of the Gravity document only.
3. Neither document inlines the other's body. Cross-edges are sha256 pointers.
4. Gravity may cite `doctor_receipt_sha256`. That citation changing is a Gravity write (allocation evidence changed). The Doctor receipt file changing is a Tabula write. Both shas move only if both documents are resealed on purpose.
5. Tabula `current_checkpoint` is never a packed catalog path.
6. Gravity `packed_bytes_merkle` is never a BF16 shard merkle.
7. A writer that updates one sidecar must not rewrite the other file. Enforcement: `assert_science_separation(before, after)` in `lab/lineage/profiles.py` — if tabula digest changed, gravity digest must be bit-identical, and vice versa, unless the call is an explicit `dual_reseal` that records `UNATTRIBUTED` for any subsequent behavior claim.
8. Behavior change (greedy token ids, oracle-32, capability_256) is attributable only by a 2×2:

| | Gravity sha same | Gravity sha different |
|---|---|---|
| Tabula sha same | no science change; if behavior moved, measurement/runtime confound | **GRAVITY** |
| Tabula sha different | **TABULA** | **UNATTRIBUTED** |

`UNATTRIBUTED` is not ACCEPT. It is a refuse-to-claim. Mixed-sub15 native generate vs G0, same Tabula, is the GRAVITY cell. Abliteration vs base Qwen at the same HQ30UQ4 recipe is the TABULA cell (Gravity `recipe` same, `packed_bytes_merkle` will differ because source bytes differ — that packed-byte move does **not** by itself attribute behavior to Gravity; attribution uses `profile_sha256`, and Gravity profile_sha includes packed merkle, so a source change that is only re-packed **must** reseal Tabula and may reseal Gravity. Attribution then needs an explicit control: same packed recipe on both Tabula checkpoints, compare greedy ids, **or** same Tabula under two Gravities).

To keep the 2×2 honest when source bytes change:

- Gravity `profile_sha256` is computed over `representation_genome + packing_recipe + per_organ_allocation + reconstruction_recipe` **plus** `binding.packed_bytes_merkle` stored as a **cited** field that is **excluded** from `profile_sha256`.
- `binding.packed_bytes_merkle` lives on the document and is sealed, but `profile_sha256` (the science identity) excludes `binding` payload hashes and `seal_sha256`.
- Instance `identity.gravity_profile_sha256` = that science digest.
- Instance also stores `identity.gravity_packed_merkle` (content). A source-only change: Tabula sha moves, Gravity science sha stays, packed merkle moves. Attribution cell = TABULA.
- A recipe/allocation change: Gravity science sha moves. If Tabula sha stays: GRAVITY.

This split is load-bearing. If `profile_sha256` includes packed bytes, every Tabula change looks like a Gravity change.

Unit test (no GPU): `test_tabula_reseal_leaves_gravity_science_sha`, `test_gravity_recipe_change_leaves_tabula_sha`, `test_source_repack_moves_packed_merkle_not_gravity_science`, `test_dual_change_is_unattributed`.

---

## 8. G0 migration

Do **not** run `tools/genesis_seat.py seat`.

### 8.1 Git-tracked copy (this repo, what tests install)

Unbound today:

| field | current | kind | migrate to |
|---|---|---|---|
| artifact_sha | `56dd65d4…` | labeled_path.v1 | keep value; set `identity.artifact_sha_kind` |
| runtime_sha | `ecfc1cac…` | labeled_path.v1 | keep; set kind |
| kernel_genome_sha | `688d8b87…` | labeled_path.v1 | keep; set kind |
| capability | `{1,1,1}` | ASSIGNED | keep as contract; measured block lives on Tabula |
| research_state | null | — | leave null |
| profiles | absent | — | write sidecars + identity pointers |
| packed merkle | absent | — | UNBOUND on Gravity until stream-hash |
| Tabula intervention | model name only | — | fill from `bf16/abliteration-manifest.json` |

`make_qwen38_genesis` keeps emitting labeled shas so `lab/tests/test_genesis_*.py` stay green. Tests that compare `parent.artifact_sha == labeled_sha(...)` must not be rewritten in the same PR as the promotion preimage change.

### 8.2 Live box copy

CURRENT already overwrote the three sha fields. Migration:

1. Snapshot the live file before any write (`bind_live_observed` is the only evidence of the overwrite).
2. Set `identity.artifact_sha_kind = content_sha256.v1` (manifest bytes). Record `identity.legacy_labeled_artifact_sha = 56dd65d4…`.
3. Set `identity.runtime_sha_kind = content_sha256.v1` (resident executable). Record `identity.legacy_labeled_runtime_sha = ecfc1cac…`.
4. Set `identity.kernel_genome_sha_kind = content_sha256.v1` and `identity.kernel_genome_coverage = "single_shader_qwen_uniform_q4.metal"`. Record legacy labeled kernel sha `688d8b87…`.
5. Do **not** copy CURRENT's rebound shas onto LKG. LKG stays labeled until a deliberate LKG refresh. Document that rollback today would reintroduce the stale/reload-fail path in `genesis_body`.
6. Write the two sidecars. Point both slots' identity at them (same Tabula, same Gravity science; LKG packed merkle still UNBOUND).
7. Stale `identity.resident_pid = 40316` is not provenance. Leave it. Peek/health own liveness. CITED `g1-artifact-inventory.md:67`, `g1-resident-harvest.md:257`.

### 8.3 Resident follow-up (not this lane; named so it is not forgotten)

`Body::reload` and `maybe_reload_lineage` must compare using `identity.artifact_sha_kind`:

- `labeled_path.v1` → do not compare to `artifact_manifest_sha`; compare `identity.artifact` path only, or refuse reload.
- `content_sha256.v1` → current compare is correct **if and only if** the field is the manifest sha.
- `content_merkle.v1` → compare merkle, not manifest-only.

Startup "stale; load measured" stays as a safety hatch, but must log both kinds.

### 8.4 What "bound" means for G0 after migration

G0 is **partially bound**:

- Tabula intervention: EVIDENCE_ON_DISK (abliteration manifest).
- Tabula source tree merkle: UNBOUND (shards CLAIMED, not verified).
- Tabula behavioral delta vs base: UNBOUND.
- Tabula capability measured: MEASURED in a lane report, not on the instance.
- Gravity recipe/allocation/reconstruction: EVIDENCE_ON_DISK.
- Gravity manifest sha: MEASURED.
- Gravity packed merkle: UNBOUND.
- Gravity kernel genome: UNBOUND (one shader hashed).
- Science separation: not yet mechanically tested (no `profiles.py`).

A G1 nomination that cannot fill Gravity packed merkle and Tabula current-checkpoint merkle is `PENDING`, not ACCEPT.

---

## 9. Files and functions

### 9.1 New (implementation lane)

| path | symbols |
|---|---|
| `lab/lineage/content_hash.py` | `HASH_KIND_LABELED`, `HASH_KIND_FILE`, `HASH_KIND_MERKLE`, `file_sha256`, `merkle_listed`, `refuse_labeled_as_content` |
| `lab/lineage/profiles.py` | `SCHEMA_TABULA`, `SCHEMA_GRAVITY`, `TabulaProfile`, `GravityProfile`, `seal_tabula`, `seal_gravity`, `profile_science_sha` (excludes `binding` payload hashes), `assert_science_separation`, `attribute_behavior`, `attach_profile_pointers`, `g0_tabula_document`, `g0_gravity_document` |
| `receipts/ascent-2026-08-16/profiles/genesis-qwen38-g0.tabula.v1.json` | sidecar |
| `receipts/ascent-2026-08-16/profiles/genesis-qwen38-g0.gravity.v1.json` | sidecar |
| `lab/tests/test_genesis_tabula_gravity_profiles.py` | separation + merkle + from_mapping survival + promotion PENDING/FAIL |

### 9.2 Existing writers / identity

| path | symbols | change |
|---|---|---|
| `lab/lineage/canon.py` | `labeled_sha`, `digest`, `canonical`, `require_sha256` | keep `labeled_sha`; do not reuse it for content |
| `lab/lineage/identity.py` | `SCHEMA`, `DEFAULT_CAPABILITY_CONTRACT`, `GenesisInstance`, `to_dict`, `from_mapping`, `copy`, `make_qwen38_genesis` | Phase 1 optional fields; kinds; do not change labeled values |
| `lab/lineage/testing.py` | `artifact_preimage_for`, `make_child`, `passing_evidence` | content preimage when clause armed |
| `lab/lineage/promotion.py` | `_computed_artifact_sha`, `_preimage_bytes`, `CLAUSE_ARTIFACT_IDENTITY`, `CLAUSE_RUNTIME_GENOME`, `ALL_CLAUSES`, `evaluate_promotion` | content preimage; new clauses |
| `lab/lineage/state.py` | `install`, `snapshot`, `handover`, `LineageState.to_dict` | do not put profiles in research_state; install must not strip identity extras |
| `lab/lineage/transfer.py` | `TRANSFER_PAYLOAD_KEYS`, `normalize_payload`, `pack_state`, `parent_research_payload` | nest under `genomes` only |
| `lab/lineage/bus.py` | `MessageType.PROFILE_DELTA`, `validate_message` | one-profile body refuse |
| `lab/lineage/continuity.py` | `normalize_generation` | later additive shas |
| `lab/provenance.py` | `sha256_file`, `sha256_bytes`, `pin_digest` | reuse |
| `lab/receipts.py` | `seal`, `verify` | reuse |
| `tools/genesis_seat.py` | `seat`, `status`, `STATE_FILE` | **forbid wipe**; merge pointers; never call bare `make_qwen38_genesis` onto a live bind |
| `tools/genesis_peek.py` | `main` | optional one-line profile shas; not required |
| `tools/ascent_daemon.py` | CANDIDATE check | no change |
| `tools/agentos/genesis_body/src/main.rs` | `artifact_manifest_sha`, `read_lineage_current`, `Body::reload`, `maybe_reload_lineage`, `main` | kind-aware compare |
| `crates/hawking-core/src/model/qwen38_pack.rs` | `artifact_filename`, `pack_qwen38_language_uniform_q4`, `Qwen38CatalogRow` | later: emit per-file content sha in catalog; do not rename files (would break load) |
| `crates/hawking-core/src/model/qwen38_hybrid_decode.rs` | `load`, `load_mixed`, `parse_qwen38_mixed_catalog` | later: verify merkle at load |
| `crates/hawking-core/src/model/qwen38_geometry.rs` | `QWEN38_SOURCE_REPOSITORY`, `QWEN38_BASE_REPOSITORY`, `QWEN38_BASE_REVISION` | Tabula constants already correct |
| `tools/qwen38_sub15_pack.py` | `artifact_filename`, recipe constants, `CODEC_*` | G1 Gravity recipe authority |
| `lab/operators/doctor6/verify.py` | `verify` → `hawking.doctor6.verify.v1` | Q30-oriented; Qwen38 Doctor is the stranded sensitivity instrument, not doctor6 |
| `receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json` | slots + events | Phase 0 pointer patch only |
| `receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json` | entries NS-009, NS-014, NS-018, NS-019 | cited, not rewritten |

### 9.3 Tests that pin current (broken) meaning

| test | pin |
|---|---|
| `lab/tests/test_genesis_lineage_state.py` | install/handover/rollback on `hawking.lineage.genesis_instance.v1` |
| `lab/tests/test_genesis_promotion_gate.py` `test_qwen38_parent_identity_is_the_seated_genesis` | labeled parent |
| `lab/tests/test_genesis_promotion_gate.py` artifact swap cases | labeled receipt sha |
| `lab/tests/test_genesis_promotion_gate_adversarial.py` `artifact_preimage_missing` / `wrong` | label preimage |
| `lab/lineage/testing.py` `passing_evidence` | the green path that hashes a string |

These must stay green through Phase 0. Phase 1 promotion redefinition needs a flag or a second clause name so the old tests do not silently invert.

---

## 10. Worked attribution examples

1. **Native mixed-sub15 generate, same BF16/abliterated tree.** Tabula sha same. Gravity science sha different (recipe + allocation + reconstruction). If greedy ids ≠ G0: **GRAVITY**. If they match: Gravity changed representation without a measured behavioral move (still not a Tabula claim).
2. **Expand-to-Q4 generate of mixed-sub15 (the recorded INCOHERENT).** Confounded. Attribution **UNATTRIBUTED**. Reconstruction vehicle is a second Gravity. Do not write a Tabula behavioral_delta from `QWEN38_SUB15_INCOHERENT.json`.
3. **Re-abliterate destination layers or scale, same packer recipe.** Tabula sha different. Gravity science sha same. Packed merkle different. If greedy ids change: **TABULA**.
4. **Doctor recapture (4096 tokens) then same uniform-Q4 pack.** Tabula sha different (new doctor receipt). Gravity science sha same if allocation unchanged. No generate change expected; if generate changes, measurement confound (different runtime), not a science claim.
5. **Doctor recapture then reallocate organs and pack mixed-sub15.** Both shas move. Any generate delta is **UNATTRIBUTED** until a control fills a 2×2 cell.

---

## 11. Cheapest experiments this spec will not run

| gap | experiment | who |
|---|---|---|
| G0 packed merkle UNBOUND | stream-hash 755 listed files + manifest | CPU, no GPU, no numpy load |
| BF16 shard shas CLAIMED | stream-hash 11 safetensors vs `artifact-manifest.json` | ~50 GiB read; not this box-busy lane |
| Abliteration direction bytes | hash the stored direction tensor vs `3958f6bb…` | if the tensor is on disk next to the manifest; else UNBOUND |
| Behavioral delta vs base Qwen | paired greedy, GPU lock | serialized measurement lane |
| Kernel genome merkle | sha256 the shader set named in `g1-kernel-inventory.md` §2 + `qwen38_hybrid_decode.rs` + `qwen38_64_layer_execution_schedule.rs` | CPU |
| Qwen38 Doctor on HEAD | land stranded `7c5f323d9` instrument or the replacement capture lane | other lanes |

---

## 12. KILLS / REOPEN_IF

- **KILLS** treating `labeled_sha` or `sha256(tensor_name)` as content provenance. REOPEN_IF a field carries `*_kind` and the preimage is file bytes.
- **KILLS** stuffing profiles into `research_state` or adding required transfer keys. REOPEN_IF transfer schema is deliberately bumped and all pack/accept callers migrate.
- **KILLS** running `genesis_seat.py seat` as a bind step. REOPEN_IF `seat` gains a merge-from-live path that preserves identity extras and does not call bare `make_qwen38_genesis`.
- **KILLS** attributing mixed-sub15 INCOHERENT to Tabula or to a native Gravity codec. REOPEN_IF a native HQ38M20 generate exists with 0 fallbacks.
- **KILLS** collapsing Doctor (what to protect) into Gravity rescaling rules. Contract context + `g1-out-proj-forensics`: activation magnitude for protection ≠ activation-weighted scaling.
- **KILLS** concluding a coherence floor from a confounded expand-to-Q4 failure. Already campaign law; restated so this schema cannot launder that verdict as a Tabula delta.

---

## 13. Required tests for this lane

```
$ test -s workspace/superwave/g1/g1-tabula-genome.md
# see completion report TESTS

$ wc -l workspace/superwave/g1/g1-tabula-genome.md
# see completion report TESTS

$ git status --porcelain
# see completion report TESTS
```

---

```
STATUS
IMPLEMENT_READY

CLAIMS
1. Git G0 artifact/runtime/kernel shas are labeled_sha(path), not content. EVIDENCE §1.2 MEASURED python + identity.py:245-248 + canon.py:29-30 + git GENESIS_LINEAGE_CURRENT.json.
2. Live G0 rebound those fields to sha256(manifest.json) / genesis-resident / one shader; still not packed-byte merkle; capability still assigned. EVIDENCE §1.1, §1.3, §1.4 live JSON dump + shasum + DEFAULT_CAPABILITY_CONTRACT.
3. Tensor filenames are sha256(name). EVIDENCE qwen38_pack.rs:270-273 + MEASURED layernorm name_sha ≠ content_sha.
4. Promotion artifact clause hashes a label preimage. EVIDENCE promotion.py:193-200 + testing.py:35-42.
5. Extra top-level instance keys drop on from_mapping; identity string keys survive; research_state is handover-clobbered. EVIDENCE identity.py:201,228,235 + state.py handover + transfer.py:17-26.
6. TABULA_PROFILE / GRAVITY_PROFILE schemas, content merkle, Phase 0/1 attachment, G0 migration, and INVARIANT_SCIENCE_SEPARATION are specified with file:function names. EVIDENCE §3–§9.
7. A behavior change is attributable only via the 2×2 on profile science shas; dual change is UNATTRIBUTED. EVIDENCE §7, §10.

EVIDENCE
- labeled_sha match: this process python (four hex lines in §1.2).
- git lineage CURRENT artifact_sha 56dd65d4…: git show HEAD:receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json.
- live lineage CURRENT artifact_sha d650a757…, identity keys, bind_live_observed: json.loads of /Users/scammermike/Downloads/hawking/receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json.
- manifest sha d650a757…: shasum -a 256 …/uniform-q4-v1/manifest.json.
- layernorm content_sha 4487e59f… ≠ name_sha b562b986…: this process sha256_file of the 20488-byte f32v2.
- abliteration-manifest sha a6e35878… matches artifact-manifest: this process hashlib of the file.
- Resident compare: tools/agentos/genesis_body/src/main.rs:394-521, 839-868.
- Readers: tools/genesis_seat.py:33-40, 47-60; tools/genesis_peek.py slots loop; tools/ascent_daemon.py:187-198.

CHANGES
- Created workspace/superwave/g1/g1-tabula-genome.md only.

TESTS
- test -s / wc -l / git status --porcelain: run after write; paste in the session completion report.

RISKS
- Live and git lineage already disagree; a naive seat() or a from_mapping roundtrip of the live file drops physical_bpw and can restore labeled shas.
- Arming content preimage in promotion without a flag breaks 55 lineage tests that hash labels.
- Including packed merkle inside Gravity science sha collapses the 2×2 (source repack looks like Gravity).

UNRESOLVED
- Packed-byte merkle of 755 G0 tensors not computed (stream-hash left as cheapest experiment).
- BF16 shard shas not re-verified (~50 GiB).
- Abliteration direction bytes not hashed.
- Kernel-set merkle not computed.
- No profiles.py yet; invariant is specified, not executed.

NEXT
- Implementation lane: lab/lineage/content_hash.py + profiles.py + Phase 0 sidecars + identity pointer patch. Do not seat(). Do not GPU.
```
