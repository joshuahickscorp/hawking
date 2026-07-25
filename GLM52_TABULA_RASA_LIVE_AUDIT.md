# GLM-5.2 tabula rasa live audit

Generated 2026-07-23T04:11:49.250641Z. Seal `2cab73096e5d9dc4`.

Endpoint status: **TABULA_RASA_PREP_IN_PROGRESS**. This is a preparation campaign; it does not begin the new stream.

## Repository

- `/Users/scammermike/Downloads/hawking` on `campaign/glm52-bf16-xet-gravity` at `83f28f5e4b3a`
- unpushed: 1 commit(s) -- 83f28f5e Revert "Bound the k-means distance matrix by the codebook, not the tensor"
- dirty tracked: 12, untracked: 3 (pre-existing, not this session's)
- worktrees: 6

    - `/Users/scammermike/Downloads/hawking                       83f28f5e [campaign/glm52-bf16-xet-gravity]`
    - `/Users/scammermike/Downloads/hawking-1000tps               5981b835 [campaign/hawking-true-batch1-1000tps]`
    - `/Users/scammermike/Downloads/hawking-breakthrough          d0448a11 [campaign/glm52-inference-breakthrough]`
    - `/Users/scammermike/Downloads/hawking-hide-build            4fbca8bc [build/hide-impl-2026-07-19]`
    - `/Users/scammermike/Downloads/hawking-kimi-release-executor 11b17604 [codex/kimi-release-executor]`
    - `/Users/scammermike/HawkingWorktrees/subbit-reset           c2019114 [campaign/subbit-capability-density-reset]`

## Runtime

- controller PID `14145`, lease `/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity/source_fetch/fetch.lock` held by PID `15406`
- controller env: `{}`
- caffeinate present: True
- launchd GLM jobs: 7
- foreign heavy jobs (MoP): 4 at nice 20 -- **in-memory only, reverts when they restart**

## Source

- `zai-org/GLM-5.2` @ `b4734de4facf`, 282 weight shards
- verified-ever 147/282, probed 124, packed 111, resident now 35
- source root 186.2 GB

## Teacher evidence

- 8 sealed capsules covering layers [0, 2, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 38, 39, 40]
- calibration: **SEALED_SYNTHETIC_TOKEN_ID_PROBE (not natural text)**
- capsules declaring a chain gap: ['L02_L02', 'L16_L18', 'L28_L30', 'L38_L40']
- ledger 14 rows at `/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity/source_fetch/teacher/GLM52_TEACHER_EVIDENCE_LEDGER.jsonl`

## Artifacts

- Desktop output `/Users/scammermike/Desktop/GLM52-Gravity-SubBit`: 111 `.gravity`, 32.3 GB
- representation: BASELINE_A / glm52 PQ R0 (NEGATIVE CONTROL, unverdicted)

## Protected

- `/Users/scammermike/.cache/huggingface` and 5 MoP roots -- HARD_PROTECTED_NEVER_WRITE_NEVER_DELETE

## Host

- 28 cores, free disk 417 GB of 995 GB
- PhysMem: 95G used (15G wired, 9400M compressor), 192M unused.
- swap total = 3072.00M  used = 1501.50M  free = 1570.50M  (encrypted), thermal green True

## Open defects found in this audit

| id | severity | gate | component | defect |
|---|---|---|---|---|
| D1 | MEDIUM | 8.1 live-process truth | `glm52_source_fetch.safe_to_leave/_controller` | pgrep -f 'glm52_source_fetch.py run' also matches an auditing shell whose command line contains the string, so PID/env/lease can be attributed to the wrong process |
| D2 | HIGH | 5.1 largest-tensor safety | `gravity_forge._kmeans` | [N,K] distance allocation is proportional to tensor size; 61 GB for the 951,582,720-weight embedding/lm_head, causing MPS OOM and ~67 GB wired retention |
| D3 | MEDIUM | 5.5 determinism policy | `gravity_forge.pack_product_quant` | non-deterministic on MPS: identical weights and seed give 2 distinct hashes over 3 fits; docstring claims determinism |
| D4 | HIGH | 6.1/6.2 natural corpus + sequential lineage | `glm52_teacher_capture` | calibration is a synthetic token-id probe, and non-zero-start runs declare chain_gap_layers; neither is an authoritative full-run trajectory |
| D5 | HIGH | 5.1 | `glm52_pack.pack_shard` | model-00001 (embedding + lm_head) has never packed; artifact would complete at 281/282 missing the vocabulary organ |
| D6 | MEDIUM | 5.2 dynamic resource controller | `resource policy` | no automatic backoff; 6 pack workers drove PhysMem to 95G with 207MB unused and 33GB compressor, manually reduced to 4 |

D2 and D5 are the same root cause: the k-means distance matrix scales with the tensor, so the
vocabulary organ cannot be packed and the artifact would complete at 281/282. A chunked rewrite
was attempted and reverted after it produced GPU address faults on real tensors.

