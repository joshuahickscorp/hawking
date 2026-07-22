# GLM-5.2 Corpus Integrity

**Status:** PASS

This is an offline corpus-integrity admission, not a model-quality result. No model payload was fetched.

## Bound tokenizer

- Repository: `zai-org/GLM-5.2`
- Revision: `b4734de4facf877f85769a911abafc5283eab3d9`
- SHA-256: `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`
- Vocabulary: 154,856 IDs
- Loader: direct local `tokenizer.json`; network disabled

## Coverage

- Nine partitions: 9
- Domains per core split: 15
- Core records: 135
- Long-context records: 36
- Atomic segments: 33,804
- Declared document families: 153
- Matched ladder repeats not counted as independent samples: 18
- Embedding-claim target IDs: 9

## Semantic and near-duplicate admission

Numbers and long generated identifiers are redacted before comparison. Exact redacted skeleton reuse across splits fails, independently of the shingle gates.

| View | Width | All-split limit | Maximum | Train/eval limit | Maximum |
|---|---:|---:|---:|---:|---:|
| Character shingles | 9 | 0.350 | 0.238938 | 0.250 | 0.143939 |
| Official-token-ID shingles | 3 | 0.350 | 0.260870 | 0.280 | 0.200000 |

Pairs checked: 12,996 cross-split, including 5,415 training-versus-evaluation pairs.

## Context ladder

| Rung | Admission | Records | Exact policy |
|---:|---|---:|---|
| 2K | ADMITTED | 9 | all five disclosed buckets per rung |
| 8K | ADMITTED | 9 | all five disclosed buckets per rung |
| 32K | ADMITTED | 9 | all five disclosed buckets per rung |
| 128K | ADMITTED | 9 | all five disclosed buckets per rung |
| 256K | NOT_ADMITTED_RESOURCE_VALIDATION_PENDING | 0 | admit only with sealed resource-valid execution evidence |
| 1M | NOT_ADMITTED_EXACT_RUNTIME_PENDING | 0 | admit only when the exact runtime executes it safely |

## Hard-fail gates

- character shingle near duplicate: **PASS**
- cross split context overlap: **PASS**
- cross split document family: **PASS**
- domain imbalance hidden by averages: **PASS**
- evaluation prompt leakage: **PASS**
- missing provenance: **PASS**
- number or identifier salted template: **PASS**
- official token shingle near duplicate: **PASS**
- position only leakage: **PASS**
- repeated segment inflation: **PASS**
- repeated semantic family inflation: **PASS**
- tamper evident seal: **PASS**

Every core split contains exactly one record for every required domain. Long-context token volume is reported separately and cannot hide a missing domain cell. Source-document identity/content, normalized segment, context window, prompt, and embedding-claim token ownership are independently checked.

Matched query families reused across context lengths remain inside one partition and are counted as ladder controls, not independent capability samples. Generated distractor rows likewise contribute context length, not sample count.

This admission establishes corpus hygiene only. A capability claim remains forbidden until the real model has produced sealed per-domain and per-rung scores.

The 256K rung is not admitted without resource-valid execution evidence. The 1M rung is explicitly not admitted until the exact runtime executes it safely; shorter tests do not support a 1M preservation claim.

Manifest seal: `7f5eff81a10c01b231e829d5870d6e8290a745a4bacdf4dfe74c4d0e5080e767`
