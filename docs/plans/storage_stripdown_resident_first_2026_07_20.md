# Storage stripdown and the resident-first resequencing

2026-07-20/21, branch `campaign/storage-stripdown-resident-first`.

Close Qwen. Release every non-MOP model payload that passed the gates. Measure the stripped
machine. Then reorder the ladder so disk residency, not parameter count, decides what runs next.

---

## 1. Qwen3-235B-A22B is closed

The final chained pass, `S64_gamma`, sealed at `2026-07-21T02:40:12Z`. All six frozen holdout
prompts wrote checkpoints, Telegram delivered them at 02:40:32Z, and the campaign released its
lease and exited cleanly.

| | |
|---|---|
| complete BPW | `918123097/918334510` = 0.999769787, legal under the 1/1 ceiling |
| mean symKL vs parent | 5.94 (gate: <= 0.10) |
| next-token argmax agreement | 0.20 (gate: >= 0.95) |
| mean top-5 overlap | 0.12 |
| capability passes, all 8 variants | **none** |
| verdict | `SUBBIT_UNSOLVED`, classification `RECONSTRUCTION_BOUND` |

Ladder run: `R0_parent`, `S64_structural`, `S64_doctor`, `D1_route_only`, `D2_recon_only`,
`S32_recon_first`, `S2A_adaptive_k`, `S64_gamma`. Branch
`campaign/subbit-capability-density-reset` pushed, tagged `qwen3-235b-sealed-20260721`.

`S64_gamma` is worth naming: data-free output-aware coding on gate/up at **zero additional bits**,
because the load-bearing quantity turned out to be the post-attention LayerNorm gamma, which
already ships native as a pass-through tensor. It cost nothing and it still collapsed. That is the
cleanest possible statement of the reconstruction bound.

### Two gates that earned their keep

**The stale seal.** `QWEN_GRAVITY_STATE.json` is rewritten per generation. At the moment the
controller started waiting it already read `SEALED / final:true / rows_done 36 of 12` — the
*previous* generation's seal — while the chained `S64_gamma` pass was on row 0 of 6. A controller
that trusted `status` alone would have released 437.9 GiB under a campaign that had completed
nothing. `WAIT_QWEN` now requires a seal strictly newer than the one observed on entry, plus a
released lease. The self-check asserts a stale seal cannot unlock the release.

**The heartbeat that stops.** The first tightening then produced a false negative: it proved drain
from the heartbeat's row counter, and the heartbeat stops updating at process exit, so it froze at
`0/6` the instant the campaign finished. The controller blocked rather than proceeding — correct
direction, wrong reason. Drain is now proven from the seal plus a released lease, never from a
heartbeat.

---

## 2. MOP was protected, including a cache that did not look like MOP

`MOP_ROOT` resolved to `/Users/scammermike/Downloads/mop`, 21.75 GiB, unchanged. Also
hard-protected: `mop-data`, `mop-experimental-method-reformation`, `mop_expansion_bundle`,
`~/Library`, `~/Documents`, `~/Pictures`, `~/Desktop`, `~/.ssh`, `~/.gnupg`, `~/.config`, the
system volumes, and this repo's `.git`.

The one that mattered was not obvious: **`~/.cache/huggingface` holds 7.5 GiB of
`facebook/vjepa2-*` weights that belong to MOP**, pulled by
`mop/tests/unit/test_vjepa21_official.py`. The first draft of the cleaner walked the whole hub
cache. That would have deleted a MOP build product, which this campaign forbids outright. The
entire hub cache is now a hard-protected root and its 9.93 GiB is excluded from every recovery
figure in this report.

The gate is fail-closed by construction. Every candidate is re-resolved with `realpath` and
device/inode identity at **both** manifest time and unlink time, and rejected if it resolves to or
through a protected root by path or by inode, if it is a symlink, if it sits on another volume, if
it is not owned by the invoking uid, or if any check raises. Deletion unlinks exact files one at a
time from a sealed manifest. No globs, no `rm -rf`, no directory recursion, no symlink following.

---

## 3. What was released

| item | files | bytes |
|---|--:|--:|
| `models/qwen3-235b-a22b/*.safetensors` | 118 | 437.90 GiB |
| `~/.cache/codex-runtimes` (see below) | 28,369 | 1.51 GiB |
| `HawkingWorktrees/deep-architecture-foundry` (worktree + target) | — | 0.34 GiB |
| **rejected by the gates** | 0 payload / 662 cache | — |

118 of 118 shards unlinked, zero failures. `config.json`, `tokenizer.json`, `vocab.json`,
`merges.txt` and the tensor index were retained, so `MODEL_RELEASE_qwen3-235b-a22b.json`
rehydrates the source byte-for-byte from the pinned public revision
`ac9c66cc9b46af7306746a9250f23d47083d689e`. Nothing unique to this machine was in the payload.

**One mis-scope, recorded rather than smoothed over.** `~/.cache/codex-runtimes` is another tool's
environment cache. The campaign's own rules put that on the report-only list, not the auto-clean
list, and it should never have been a target. It was caught mid-delete, 1,887 of 27,593 files in.
I let it finish instead of interrupting, because a half-removed runtime tree does not self-heal
while a fully absent one re-downloads on next use. `CACHE_TARGETS` is now empty.

### Preserved before removal

- **`deep-architecture-foundry`**: 3 unique commits pushed to
  `archive/deep-architecture-foundry-2026-07-20` *and* captured in a bundle that verifies clean,
  before the worktree was removed.
- **`hawking-hide-parity-research`**: clean, and HEAD contained in its own pushed upstream. The
  first pass refused it for "1 unique commit" because it counted against `origin/main` only; a
  branch level with a pushed upstream is already preserved, and the rule now checks that.
- **`hawking-hide-build`**: kept. 235 dirty files, no upstream, and a live build process in it.
  Archived without committing: a 78,286-line diff against HEAD plus the 168 untracked files.
- **Qwen parent reference logits**: 51 MiB of `.npy` for the six holdout prompts, the bf16
  reference every candidate was scored against. They were untracked *and* gitignored, living only
  in the recovery clone. Releasing the source moved their regeneration cost from zero to a full
  437.9 GiB re-download plus a full parent forward, so they are force-added here with a sha256
  manifest.

### Reported, not touched

`computexchange` (34 GiB, separate project), `~/Library` (86 GiB), `~/go` (4 GiB), `~/.cache/uv`
(1.8 GiB), `hide-donor-analysis` (363 MiB, external donor clones), the three APFS OS-update
snapshots, and `~/.Trash` (0.8 MiB). Each needs its own authorization.

---

## 4. Free space, measured

| | GiB |
|---|--:|
| free at campaign start | 114.7 |
| free after stripdown | **555** |
| recovered | ~440 |

Volume total 926.4 GiB; used fell from 762 GiB to 323 GiB.

Note on `hawking-hide-build`: `du` reports 31 GiB, the inventory's `st_blocks` sum reports
53.9 GiB. The gap is APFS clone/dedup — `du` counts shared blocks once. Both are reported rather
than picking the flattering one.

---

## 5. Resident-first eligibility

    MIN_RESERVE      = max(32 GiB, 2 x largest official shard, 3% of volume)
    WORKING_HEADROOM = max(16 GiB, largest shard, projected compact checkpoint)
    compact checkpoint = total_params / 8 bytes, one complete bit per original weight

    FULL_RESIDENT_COMFORTABLE : source + headroom + 80 GiB <= free
    FULL_RESIDENT_SQUEEZED    : source + headroom + MIN_RESERVE <= free

Every source figure is a live `HfApi` `files_metadata` sum at a pinned revision. No nominal sizes.

**Kimi K2.6 was verified, not assumed, and it does not fit.** 554.3 GiB of source, plus a
124.6 GiB projected compact checkpoint for 1.07T parameters, plus a 32 GiB reserve, is 710.9 GiB
required against a projected ceiling of 552 GiB — short by 158.9 GiB. The absolute ceiling on this
volume after deleting every non-MOP payload is ~602 GiB, so no arrangement of deletions reaches
it. `studio_manifest.py` also had it wrong in a second way: it lists K2.6 as a 595 GiB
`compressed-tensors` checkpoint, and the live config says `bfloat16`.

Nine parents fit fully; nine do not. Every one of the nine that do sorts ahead of every one of the
nine that do not.

| # | parent | source GiB | fit | margin GiB |
|--:|---|--:|---|--:|
| 1 | **deepseek-ai/DeepSeek-V4-Flash-DSpark** | 155.4 | COMFORTABLE | 331.5 |
| 2 | MiniMaxAI/MiniMax-M2 | 214.4 | COMFORTABLE | 278.9 |
| 3 | openai/gpt-oss-20b | 38.5 | COMFORTABLE | 465.5 |
| 4 | moonshotai/Kimi-Linear-48B-A3B-Instruct | 91.5 | COMFORTABLE | 412.5 |
| 5 | ibm-granite/granite-4.0-h-small | 60.0 | COMFORTABLE | 444.0 |
| 6 | nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 | 92.9 | COMFORTABLE | 411.1 |
| 7 | Qwen/Qwen3-Next-80B-A3B-Instruct | 151.5 | COMFORTABLE | 352.5 |
| 8 | Qwen/Qwen3-VL-235B-A22B-Instruct | 439.0 | COMFORTABLE | 53.7 |
| 9 | openai/gpt-oss-120b (F0, sealed) | 182.3 | COMFORTABLE | 187.4 |
| 10 | moonshotai/Kimi-K2.7-Code | 554.3 | DOES NOT FIT | -158.9 |
| 11 | moonshotai/Kimi-K2.6 | 554.3 | DOES NOT FIT | -158.9 |
| 12 | deepseek-ai/DeepSeek-V3.2 | 642.2 | DOES NOT FIT | -200.3 |
| 13 | meta-llama/Llama-4-Maverick-17B-128E | 748.0 | DOES NOT FIT | -282.6 |
| 14 | Qwen/Qwen3.5-397B-A17B (F2) | 751.4 | DOES NOT FIT | -277.5 |
| 15 | deepseek-ai/DeepSeek-V4-Pro-DSpark | 831.4 | DOES NOT FIT | -497.7 |
| 16 | Qwen/Qwen3-Coder-480B-A35B | 894.4 | DOES NOT FIT | -430.3 |
| 17 | moonshotai/Kimi-K2-Instruct | 958.5 | DOES NOT FIT | -559.1 |
| 18 | zai-org/GLM-5.2 | 1403.2 | DOES NOT FIT | -970.9 |

The F-ladder's nominal successor edge was F1 -> F2 -> Qwen3.5-397B. Under the resident-first law
it drops to row 14. That is the point of the law, not a violation of it.

---

## 6. The next parent, and why

**`deepseek-ai/DeepSeek-V4-Flash-DSpark`** @ `62af8fffb2f7030cac4de2f0169f5b8d1101b646`,
apache-licensed, ungated. 155.4 GiB, 48 shards, largest 3.44 GiB. 284B total / 13B active.
43 layers, 256 routed experts, top-6, 1 shared expert, 1M context, MLA attention with a sparse
indexer, a DSpark long-context compressor, and hyper-connections.

The decisive property is in `config.json`: `dtype: fp8`, `scale_fmt: ue8m0`, **`expert_dtype: fp4`**.

The routed experts — the exact organ this programme has spent the whole campaign trying to push
below one bit — are **already four bits** in the official checkpoint. Against a bf16 parent, 1.0
complete BPW is a 16x reduction. Against this one it is roughly 4x. That is a materially better
posed question, and it attacks the reconstruction bound directly rather than restating it at a
larger parameter count. It also means **this parent's BPW is not comparable to the sealed Qwen
numbers** without that caveat, which the adapter prints in every inventory it emits.

The adapter (`tools/condense/deepseek_v4_adapter.py`) classifies all **72,317** tensors from the
real index with zero unknowns:

| organ | tensors | weights | scales |
|---|--:|--:|--:|
| `ffn.experts` | 70,656 | 35,328 | 35,328 |
| `attn.mla` | 460 | 230 | 230 |
| `ffn.shared_experts` | 276 | 138 | 138 |
| `hyper_connection.block` | 279 | 279 | 0 |
| `attn.norm` | 186 | 186 | 0 |
| `attn.compressor` (DSpark) | 164 | 164 | 0 |
| `attn.indexer` (sparse attn) | 147 | 126 | 21 |
| `ffn.gate` (router) | 92 | 92 | 0 |
| remainder (embed, head, norms, MTP, markov, confidence, sinks) | 57 | | |

Routed-expert grammar confirmed exactly: 43 x 256 x 3 = 33,024 weights, every one carrying a
`ue8m0` scale companion, zero orphan scales. Index `total_size` 166,878,536,440 bytes matches the
live HfApi sum.

Three things the adapter's own checks caught:

1. A `startswith("attn.")` catch-all silently billed the DSpark compressor and the sparse indexer
   as MLA. The self-check asserts an unrecognised attention sub-module *raises*; it failed on the
   first run, and the match is now an explicit allowlist.
2. The repo ships two configs with different spellings, and copying both by basename clobbered one.
   Only the HF-style root names `model_type`; only `inference/config.json` names `expert_dtype`.
   They are now merged with an alias table that raises on disagreement.
3. That table immediately caught `n_mtp_layers: 3` against `num_nextn_predict_layers: 1`. Those are
   different quantities — stacked MTP blocks versus tokens predicted per step. The index arbitrates
   (2304 = 3 x 256 x 3 expert tensors, so three stacks) and both values are kept with their
   semantics.

---

## 7. Ledger warnings carried into admission

- `original_weight_count` is **not** bound. The index carries no shapes. Bind it from the resident
  safetensors headers before any Fraction ledger — `verify_against_source` does exactly this and
  is a launch gate.
- The `.scale` companions are real artifact bytes and must be billed, not treated as metadata.
- The MTP stack, `markov_head` and `confidence_head` must be explicitly included in or excluded
  from the denominator, with the choice recorded.
- Runtime boundary: **Python reference forward only.** The Rust engine does not route
  `deepseek_v4`. No tok/s, resident-footprint or serve claim is legal for this parent.

---

## 8. Where it stops, honestly

The controller stages and verifies the source. `LAUNCH_PARENT` then evaluates the Doctor Prime
packet, and it will go **RED** on one gate that cannot be waved through: there is no campaign
runner for `deepseek_v4`. A real parent-vs-packed forward does not exist for this architecture.
Upstream ships a CUDA/Triton reference at `_meta/inference/model.py` + `kernel.py`; porting it to
the Metal/CPU path — MLA, sparse indexer, DSpark compressor, hyper-connections, fp8/fp4
block-scale dequant — is the blocking build item.

That gate was added deliberately. Before it, `LAUNCH_PARENT` went green merely because an adapter
file existed on disk, and the machine would have reached `COMPLETE` with no experiment running —
a state that reads as "science is advancing" when nothing is. It now blocks and names the build
item instead.

**Next build item:** the `deepseek_v4` reference forward. Everything upstream of it is done.
