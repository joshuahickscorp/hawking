# G1 mixed-sub15-v1 native generate — 1.291 BPW

Lane: `90-sub15-native-generate`. Date: 2026-08-17.
Write scope in-repo: this file only.
Catalog install (outside repo): `mixed-sub15-v1/catalog.hq38m20`.

Every number is `MEASURED` (this process), `RECEIPT` (quoted field), or
`LABELED` (not a complete-token wall).

---

## 0. Verdict

**INCOHERENT.** Native mixed HQ38M20 path. `fallbacks_total=0`.
`DENSE_W_MATERIALIZED=0`. `refused=0`. `expanded_to_q4=0`.
`expanded_to_float_gemv=0`. Not VOID.

All six Phase-A prompts (16 new tokens) emitted the identical period-1
cycle of token `279` (` the`). No Paris. No 323. No `<think>`. Failure
mode: **degenerate cycle** (unigram collapse to ` the`), not punctuation
collapse, not early EOS.

Phase B (France@128 / 17×19@256) was not run: 16 tokens may establish
INCOHERENT and cannot establish COHERENT. No complete-token wall.

This is a codec verdict on the catalog's **resolved** bytes, which are
**not** self-contained under `mixed-sub15-v1`. See §2.

---

## 1. Live G0 manifest sha256

Path:
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json`

| when | sha256 | inode | size | mtime |
|---|---|---:|---:|---|
| pre-run (before catalog install) | `d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df` | 315064607 | 238879 | Aug 16 03:15:06 2026 |
| post-install | same | same | same | same |
| post-generate | same | same | same | same |

Expected seal: `d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df`.
**Unchanged.** Embed inode `314847693` and lm_head inode `315002072`
unchanged; they remain hardlinked between G0 and mixed-sub15-v1.

Resident pid 50196 stayed ALIVE through the generate. Lock was released
after the run (`LOCK_FREE`). This lane did not stop, restart, or RPC the
resident.

---

## 2. Self-containment — FAIL

Exact artifact path used as `--artifact-root`:

```
/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1
```

Catalog installed (this lane, `cp -n`, 309294 B, sha256
`ab014699aca1c098cb98ee390dc0cc6af7ad90f109b5b86097f867383ebe321d`):

```
.../mixed-sub15-v1/catalog.hq38m20
```

Catalog `segment_mode` is absolute paths. `MEASURED` parse of this file:

| | n | bytes addressed |
|---|---:|---:|
| segments | 723 | — |
| segments absolute | 723 / 723 | — |
| segments under `mixed-sub15-v1` | 659 | 2_526_544_096 |
| segments under `mixed-2p0-v1/segments` | 64 | 1_814_060_541 |
| tensors resolving outside own root | **192** (all MLP) | **1_814_060_541** |
| missing payloads | 0 | — |
| range faults | 0 | — |
| whole-file `nbytes == lstat` | 659 | — |
| sliced-in-range (the 192 MLP) | 192 | — |

MLP (gate/up/down, 64+64+64) is **only** in
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1/segments/Lxx.hq38seg`.
Attention rice (304), embed, lm_head, and 353 f32v2 live under
`mixed-sub15-v1`. 355 catalog tensors share G0 inodes (embed + lm_head +
353 f32v2). 353 unused `*.f32bin` sidecars also share G0 inodes; they
are not in the catalog and were not uploaded.

**Is the run still a valid codec verdict?** Yes, with this label: the
loader opened `catalog.hq38m20` in this root, bound the named absolute
payloads, and generated with `fallbacks=0` / no expand. That is a
verdict on **those resolved bytes** (family-A MLP at 0.848 BPW + rice
attention + oracle embed/lm_head/small).

**Is it a verdict about mixed-sub15-v1 as a self-contained artifact?**
**No.** A silent read of another artifact directory is not a claim that
this directory alone holds 1.291 BPW. The 1.814 GB MLP leak is explicit
in the catalog, not an accident of the loader.

---

## 3. Complete BPW (recomputed from real bytes)

`N = 26_895_998_464` language elements (`MEASURED` sum of catalog
shapes; matches `source_weight_elements`).

Numerator = sum of catalog `nbytes` for all 851 rows. Every in-root
whole file has `nbytes == lstat`. Every MLP slice is in-range inside
its mixed-2p0 segment.

```
payload_bytes     = 4_340_604_637     MEASURED
complete_physical = 8 * 4340604637 / 26895998464
                  = 1.2910781930062503   MEASURED
```

Equals `PACK_REPORT.json` `complete_physical_bpw` (`RECEIPT`).
`8 * (payload + catalog_file) / N = 1.291170190037084` is a different
quotient (catalog file is not in the G0 numerator).

| class | n | elems | bytes | physical BPW |
|---|---:|---:|---:|---:|
| mlp.gate_proj HGRAVB01 | 64 | 5704253440 | 802177344 | 1.1250234267290902 |
| mlp.up_proj HGRAVR02 | 64 | 5704253440 | 918036000 | 1.2875108157887178 |
| mlp.down_proj HGRAVS01 | 64 | 5704253440 | 93847197 | 0.13161714918473189 |
| MLP combined | 192 | 17112760320 | 1814060541 | 0.8480504639008466 |
| attention_gemv HGRAVR02 | 304 | 7237795840 | 1165098376 | 1.2877935788805008 |
| embed HQ30UQ4 | 1 | 1271398400 | 675430440 | 4.250000251691366 |
| lm_head HQ30UQ4 | 1 | 1271398400 | 675430440 | 4.250000251691366 |
| small f32v2 | 353 | 2645504 | 10584840 | 32.00853977162764 |
| **complete** | **851** | **26895998464** | **4340604637** | **1.2910781930062503** |

All BPW in this table are `MEASURED` (`8 * bytes / elems`).

---

## 4. Vehicle

Worktree HEAD `62aa302ea`. Branch `grok/90-sub15-native-generate-20260817-120128`.

```
CARGO_TARGET_DIR=workspace/ops/build/rust \
  cargo build --release -p hawking-core --example ascension_qwen38_hybrid_greedy --offline
Finished release in 3m 15s  EXIT:0
```

```
-rwxr-xr-x  size=4673040  mtime=Aug 17 12:08:34 2026  ino=325824612
sha256=0299a2959498120e6dbe8a1d40c87da47e5245fd9fa46cbf2e7aefd6fa4d6101
workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy
```

This tree's `qwen38_hybrid_decode.rs` sha256
`582725ef0761d2f821f66f33e9be7b6892ae0eeda4cfd07069d7ff71217e378e`
matches the lane-50 native reader. The Aug 16 main-repo greedy binary
was not executed.

Chat render: default (no `--raw-prompt`).
`--max-seq-len 512`. `--max-new-tokens 16`.
Tokenizer:
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json`
Prompts:
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/coherence_prompts.txt`
(171 B).

No `HAWKING_*` / `MLX` / `RECON` env was set.

---

## 5. GPU lock (no resident stop)

```
WAIT_START=Mon Aug 17 12:09:22 EDT 2026
LOCK_OWNER_PRE=genesis-resident:parent
LOCK_PID_PRE=50196
RESIDENT_50196=ALIVE
LOCK_ACQUIRED=Mon Aug 17 12:10:42 EDT 2026
LOCK_OWNER_NOW=qwen38-sub15-native
LOCK_PID_NOW=61180
RESIDENT_50196_AT_ACQUIRE=ALIVE
WAIT_OR_RUN_END=Mon Aug 17 12:11:11 EDT 2026
EXIT=0
RESIDENT_50196_POST=ALIVE
```

Wait for lock ~80 s. Resident stayed up (unlike the q7 lane, where pid
74869 was already dead at acquire). Session open `8.183s`. Lock free
after exit.

---

## 6. Codec census at load (GPU stderr, verbatim)

```
qwen38-decode opening mixed HQ38M20 + 851 catalog tensors (no reconstruct-to-Q4)
qwen38-decode mixed upload 0/851
qwen38-decode mixed upload 50/851
qwen38-decode mixed upload 100/851
qwen38-decode mixed upload 150/851
qwen38-decode mixed upload 200/851
qwen38-decode mixed upload 250/851
qwen38-decode mixed upload 300/851
qwen38-decode mixed upload 350/851
qwen38-decode mixed upload 400/851
qwen38-decode mixed upload 450/851
qwen38-decode mixed upload 500/851
qwen38-decode mixed upload 550/851
qwen38-decode mixed upload 600/851
qwen38-decode mixed upload 650/851
qwen38-decode mixed upload 700/851
qwen38-decode mixed upload 750/851
qwen38-decode mixed upload 800/851
qwen38-decode mixed upload 850/851
qwen38-decode mixed census: tensors=851 binary=64 residual=368 hgravs=64 uniform=0 q4=2 f32=353 refused=0 expanded_to_q4=0 expanded_to_float_gemv=0
qwen38-decode mixed bind: K-complete; recon_fuse=ON uses q80_binary_group_matvec_simd_bytes / q80_binary_group_csr_matvec_bytes when cols>2048 (256-col tiles; this model K in {5120,6144}); cols<=2048 stay on tg256; recon_fuse=0 walks every column via gk_matvec_binary
qwen38 session open 8.183s for 6 prompts
qwen38 prompt 1/6 tokens=11 text="Say hi."
[hawking] HAWKING_TCB_TRACE="(unset)" → mode=Off
qwen38 prompt 2/6 tokens=17 text="Write a function that reverses a string."
qwen38 prompt 3/6 tokens=15 text="What is the capital of France?"
qwen38 prompt 4/6 tokens=19 text="Explain what a hash map is in one sentence."
qwen38 prompt 5/6 tokens=12 text="def fibonacci(n):"
qwen38 prompt 6/6 tokens=13 text="The three primary colors are"
wrote /tmp/qwen38-sub15-native-generate/QWEN38_SUB15_GENERATE_16.json
```

| check | result |
|---|---|
| first line `opening mixed HQ38M20` | **yes**. Not `opening Metal + 755 catalog tensors`. |
| reconstruct-to-Q4 | log says `no reconstruct-to-Q4` |
| load census | 851 / bin 64 / res 368 / hgravs 64 / uni 0 / q4 2 / f32 353 |
| refused / expanded_to_q4 / expanded_to_float_gemv | **0 / 0 / 0** |
| bind | K-complete, recon_fuse=ON, simd_bytes for cols>2048 |
| `fallbacks_total` | **0** (JSON + every prompt `FALLBACKS: 0`) |
| `DENSE_W_MATERIALIZED` | **0** (every prompt + JSON total) |

A nonzero fallback or any dense materialization would be VOID. This run
is not VOID.

CPU catalog peek (pre-generate, magics at catalog offset):
HGRAVB01×64, HGRAVR02×368, HGRAVS01×64, HQ30UQ4×2, f32v2 u64-numel
headers×353. Matches the GPU census.

---

## 7. Phase A — raw emitted text (verbatim)

Rendered (harness):
`<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n`

### 7.1 stdout, verbatim

```
PROMPT: Say hi.
GENERATED_TEXT_VERBATIM:  the the the the the the the the the the the the the the the the
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 11
NEW_TOKENS: [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279]
WALL_NS: 3196566584

PROMPT: Write a function that reverses a string.
GENERATED_TEXT_VERBATIM:  the the the the the the the the the the the the the the the the
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 17
NEW_TOKENS: [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279]
WALL_NS: 3595960583

PROMPT: What is the capital of France?
GENERATED_TEXT_VERBATIM:  the the the the the the the the the the the the the the the the
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 15
NEW_TOKENS: [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279]
WALL_NS: 3374803000

PROMPT: Explain what a hash map is in one sentence.
GENERATED_TEXT_VERBATIM:  the the the the the the the the the the the the the the the the
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 19
NEW_TOKENS: [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279]
WALL_NS: 3826678125

PROMPT: def fibonacci(n):
GENERATED_TEXT_VERBATIM:  the the the the the the the the the the the the the the the the
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 12
NEW_TOKENS: [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279]
WALL_NS: 3037292667

PROMPT: The three primary colors are
GENERATED_TEXT_VERBATIM:  the the the the the the the the the the the the the the the the
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 13
NEW_TOKENS: [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279]
WALL_NS: 3153744333
```

`WALL_NS` is prompt-inclusive wall (prefill + 16-token decode). It is
**not** a complete-token measurement. No TOKEN_NS / TPS is claimed.

### 7.2 Failure mode

Token `279` decodes to ` the` (leading space + "the").

| prompt | new ids | raw text | period-1 cycle ≥8 | only EOS | contains Paris/323 | `<think>` 248068 |
|---|---|---|---|---|---|---|
| Say hi. | 279 × 16 | ` the` × 16 | yes | no | no | no |
| reverses a string | 279 × 16 | ` the` × 16 | yes | no | no | no |
| capital of France | 279 × 16 | ` the` × 16 | yes | no | no | no |
| hash map | 279 × 16 | ` the` × 16 | yes | no | no | no |
| fibonacci | 279 × 16 | ` the` × 16 | yes | no | no | no |
| primary colors | 279 × 16 | ` the` × 16 | yes | no | no | no |

All six fire INCOHERENT. Characterisation: **degenerate cycle**, period
1, same attractor on every prompt. Not punctuation-only (q7 was token
`8` = `)`; 2p0 was `{198, 8, 13, 1076, 578, 220}`). Not early EOS.
Not a semantic answer that happens to be wrong — the model never leaves
the unigram.

G0 / seal `Say hi.` first 12 (`RECEIPT` from prior lanes):

```
[248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149]
```

This run: `[279] × 16`. Prefix match **0/12**.

---

## 8. JSON receipt (authoritative machine record)

`/tmp/qwen38-sub15-native-generate/QWEN38_SUB15_GENERATE_16.json`
(not written under `receipts/` — this lane's in-repo write is this file).

```json
{
  "artifact_root": "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1",
  "dense_w_materialized_total": 0,
  "fallbacks_total": 0,
  "lane": "qwen38-coherence-generate",
  "max_new_tokens": 16,
  "prompts": [
    {
      "decode_wall_ns": 1691333167,
      "dense_w_materialized": 0,
      "fallbacks": 0,
      "generated_text": " the the the the the the the the the the the the the the the the",
      "new_token_ids": [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279],
      "prefill_wall_ns": 1505230833,
      "prompt": "Say hi.",
      "prompt_ids": [248045, 846, 198, 44240, 15131, 13, 248046, 198, 248045, 74455, 198],
      "prompt_len": 11,
      "rendered": "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n",
      "wall_ns": 3196566584
    },
    {
      "decode_wall_ns": 1687888125,
      "dense_w_materialized": 0,
      "fallbacks": 0,
      "generated_text": " the the the the the the the the the the the the the the the the",
      "new_token_ids": [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279],
      "prefill_wall_ns": 1908071666,
      "prompt": "Write a function that reverses a string.",
      "prompt_ids": [248045, 846, 198, 7734, 264, 709, 421, 16915, 287, 264, 886, 13, 248046, 198, 248045, 74455, 198],
      "prompt_len": 17,
      "rendered": "<|im_start|>user\nWrite a function that reverses a string.<|im_end|>\n<|im_start|>assistant\n",
      "wall_ns": 3595960583
    },
    {
      "decode_wall_ns": 1692314333,
      "dense_w_materialized": 0,
      "fallbacks": 0,
      "generated_text": " the the the the the the the the the the the the the the the the",
      "new_token_ids": [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279],
      "prefill_wall_ns": 1682485250,
      "prompt": "What is the capital of France?",
      "prompt_ids": [248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198],
      "prompt_len": 15,
      "rendered": "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
      "wall_ns": 3374803000
    },
    {
      "decode_wall_ns": 1691193417,
      "dense_w_materialized": 0,
      "fallbacks": 0,
      "generated_text": " the the the the the the the the the the the the the the the the",
      "new_token_ids": [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279],
      "prefill_wall_ns": 2135482583,
      "prompt": "Explain what a hash map is in one sentence.",
      "prompt_ids": [248045, 846, 198, 814, 20139, 1092, 264, 5010, 2336, 369, 303, 799, 11316, 13, 248046, 198, 248045, 74455, 198],
      "prompt_len": 19,
      "rendered": "<|im_start|>user\nExplain what a hash map is in one sentence.<|im_end|>\n<|im_start|>assistant\n",
      "wall_ns": 3826678125
    },
    {
      "decode_wall_ns": 1688314750,
      "dense_w_materialized": 0,
      "fallbacks": 0,
      "generated_text": " the the the the the the the the the the the the the the the the",
      "new_token_ids": [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279],
      "prefill_wall_ns": 1348966958,
      "prompt": "def fibonacci(n):",
      "prompt_ids": [248045, 846, 198, 727, 73111, 1393, 1590, 248046, 198, 248045, 74455, 198],
      "prompt_len": 12,
      "rendered": "<|im_start|>user\ndef fibonacci(n):<|im_end|>\n<|im_start|>assistant\n",
      "wall_ns": 3037292667
    },
    {
      "decode_wall_ns": 1689402834,
      "dense_w_materialized": 0,
      "fallbacks": 0,
      "generated_text": " the the the the the the the the the the the the the the the the",
      "new_token_ids": [279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279],
      "prefill_wall_ns": 1464337458,
      "prompt": "The three primary colors are",
      "prompt_ids": [248045, 846, 198, 760, 2250, 5839, 7736, 513, 248046, 198, 248045, 74455, 198],
      "prompt_len": 13,
      "rendered": "<|im_start|>user\nThe three primary colors are<|im_end|>\n<|im_start|>assistant\n",
      "wall_ns": 3153744333
    }
  ]
}
```

---

## 9. What this does and does not decide

Does decide:

- The native mixed path loads this catalog without expand-to-Q4, without
  MLX, without fallbacks, with the expected 851-tensor census.
- Greedy decode on that bind is INCOHERENT at complete BPW
  **1.2910781930062503**.
- Rice attention (cosine 0.834–0.847, pack metadata) plus the 0.848 MLP
  does not produce tokens. Same qualitative death as mixed-2p0-v1
  (2.0856, same MLP) and mixed-floor-q7-v1 (3.1768, same MLP, Q7 attn).
  The attractor this time is `279` rather than `)` / newline.

Does not decide:

- mixed-sub15-v1 as a self-contained root (it is not).
- Families B (`mixed-q4down-v1`) and C (`mixed-q3mlp-v1`): still
  `assert_mixed_mlp_native` refuse. The MLP-floor hypothesis is
  **supported**, not closed.
- Speed. No complete-token wall. Projected TPS from `PACK_REPORT` is
  not a measurement.
- That rice attention is the cause. The shared dead organ across the
  three native family-A fails is the 0.848 MLP.

---

## 10. Outside-repo residue

Left in place (requested install, not a tracked file):

```
.../mixed-sub15-v1/catalog.hq38m20   309294 B
```

Scratch (not in repo): `/tmp/qwen38-sub15-native-generate/`,
`/tmp/sub15-native-census.json`.
