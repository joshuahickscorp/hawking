# G1 bracket generate — first native coherence datapoint inside Qwen3.8

Date: 2026-08-17. GPU lane. Write scope: this file only.

Followed `g1-bracket-bisection.md` Path A (HEAD as-is, no Rust change).
Family B (`mixed-q4down-v1`) and family C (`mixed-q3mlp-v1`) remain
LOAD_REFUSE on `assert_mixed_mlp_native`. They were not generated.
Family A′ / q8 were not generated: Path A stops after one family-A run.

**Headline:** `mixed-floor-q7-v1` generated natively, `fallbacks=0`,
`DENSE_W_MATERIALIZED=0`, mixed HQ38M20 path, no expand-to-Q4 / MLX.
Phase A 16×6 is punctuation-only collapse on every prompt.
Verdict: **INCOHERENT**. This is a family-A fact. It does **not** locate
a 1-D Qwen3.8 floor and does **not** license `floor > 3.1768`.

---

## 0. What this lane is allowed to decide

Authority: `workspace/superwave/g1/g1-bracket-bisection.md` §§3.3, 5.3–5.4.

- Rust frozen → Path A only. One generate: `mixed-floor-q7-v1`.
- Binding for a codec verdict: `opening mixed HQ38M20`, `fallbacks_total==0`,
  `DENSE_W_MATERIALIZED: 0`, vehicle = `ascension_qwen38_hybrid_greedy`
  built from this worktree HEAD. Expand-to-Q4 / MLX is void.
- 16 new tokens × 6 prompts can decide **INCOHERENT** only.
- **COHERENT** requires Phase B: France@128 contains `Paris` and
  17×19@256 contains `323`, both well-formed. Phase B is not run on collapse.
- A family-A fail does not raise the Qwen3.8 floor to 3.18. Families B and C
  remain untested.

---

## 1. Vehicle (this binary, not a stale one)

Worktree HEAD: `bf0b4dc0250041a8ff9237065c89404eff48506b`
Branch: `grok/71-bracket-generate-20260817-113742`

```
CARGO_TARGET_DIR=/Users/scammermike/.claude-grok/worktrees/71-bracket-generate-20260817-113742/workspace/ops/build/rust \
  cargo build --release -p hawking-core --example ascension_qwen38_hybrid_greedy --offline
```

```
    Finished `release` profile [optimized] target(s) in 3m 07s
EXIT:0
Mon Aug 17 11:43:15 EDT 2026
-rwxr-xr-x  1 scammermike  staff  4673040 Aug 17 11:43 workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy
04108ec95b1d659ea07d9e27b75120cf773f089194bc2af386fd78525f4008a7  workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy
inode=325650794 size=4673040 mtime=Aug 17 11:43:14 2026
```

The main-repo binary at
`/Users/scammermike/Downloads/hawking/workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy`
is a different file (mtime Aug 16 22:36, size 4656528,
sha256 `a01452066e8811bcb0fb327247a74f74d21a9f4a6b3980796be8fcc76d15233f`).
It was not executed.

Chat render: default (no `--raw-prompt`).
`--max-seq-len 512`. `--max-new-tokens 16` for Phase A.

---

## 2. GPU lock and resident coexistence

Lock script: `tools/gpu_lane_lock.sh` → `/tmp/hawking-gpu-lane.lock`.

```
WAIT_START=2026-08-17 11:43:53 EDT
RESIDENT_PRE=74869 1367568 01:12:33     # genesis-resident G0, lock owner genesis-resident:parent
LOCK_ACQUIRED=2026-08-17 11:47:19 EDT
LOCK_OWNER=qwen38-bracket-a
LOCK_PID=3579
RESIDENT_AT_ACQUIRE=DEAD
MEM_AT_ACQUIRE free_gb=16.35 inactive_gb=35.59 reclaimable_gb=51.94
WAIT_OR_RUN_END=2026-08-17 11:47:50 EDT
EXIT=0
RESIDENT_POST=DEAD                      # pid 74869 specifically; see below
```

Wait for the lock was ~206 s. This lane did not signal, stop, RPC, or
restart the resident. Pid 74869 was already gone when the lock came free.

`genesis_forever.sh` started a replacement on its own:

```
ps -p 15546 -o lstart → Mon Aug 17 11:47:18 2026
genesis-resident.log: loading uniform-q4-v1, opening Metal + 755 catalog tensors
genesis-resident: body resident 14.625s weight_bytes=14297675776
genesis-resident: listening ... pid=15546
log mtime 11:47:33
```

So G0 reload (11:47:18–11:47:33) overlapped this lane's mixed upload
(session open 9.209 s from 11:47:19) and the first decode steps.
That is a coexistence caveat for any wall / TOKEN_NS number.
It is **not** treated as a fallback. Binding logs below are the mixed
path with `fallbacks=0`. Collapse shape matches the already-measured
native 2p0 family-A failure (punctuation / short cycle), not a Metal
fault or a missing-GEMV abort.

This lane did not kill 74869 or 15546. After the oneshot exited, 15546
was still serving G0 (`uniform-q4-v1`). No complete-token wall was
taken (gate failed; walls would also have been dirty).

---

## 3. Artifact attempted — `mixed-floor-q7-v1` (family A)

### 3.1 Path and bytes (recomputed this lane)

Exact path:

```
/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-floor-q7-v1
```

Parser: HQ38M20 header + 128-byte records as
`parse_qwen38_mixed_catalog` (`qwen38_hybrid_decode.rs:96-174`).
Payload = sum of catalog `nbytes`. N = 26_895_998_464.
G0 complete BPW = `8 * payload / N` (`qwen38_pack.rs:673-679`).
Catalog file itself is not in the G0 numerator.
Every referenced `[offset, offset+nbytes)` was inside its segment file
(0 range faults). 0 missing magics.

```
magic b'HQ38M20\x00'
version 1  n_tensors 851  n_segments 66  catalog_bytes 158970
payload_bytes 10680295260
elements 26895998464   elems_match N = True
complete_physical_bpw (G0 def) = 3.1767685514394888
PACK_REPORT-style 8*(payload+catalog)/N = 3.1768158357967402
range_faults 0
```

PACK_REPORT.json on disk claims `tensor_payload_bytes` 10680295260 and
`complete_physical_bpw` 3.17681583579674 (catalog included). The G0
number used below is **3.1767685514394888**.

### 3.2 Codec census at catalog + header (CPU, before generate)

| organ | n | elems | bytes | BPW | catalog codec | magic |
|---|---:|---:|---:|---:|---|---|
| mlp.gate_proj | 64 | 5_704_253_440 | 802_177_344 | 1.1250234267290902 | 0 | HGRAVB01 ×64 |
| mlp.up_proj | 64 | 5_704_253_440 | 918_036_000 | 1.2875108157887178 | 1 | HGRAVR02 ×64 |
| mlp.down_proj | 64 | 5_704_253_440 | 93_847_197 | 0.1316171491847319 | 2 | HGRAVS01 ×64 |
| attn+embed+norm | 659 | 9_783_238_144 | 8_866_234_719 | 7.2501432253799178 | 3 | HGRAVU01 ×659 bits=7 |
| **total** | **851** | **26_895_998_464** | **10_680_295_260** | **3.1767685514394888** | | |

U01 bits histogram: `{7: 659}`. No bits=4 / bits=8 U01 on this pack.
MLP recipe is byte-identical to mixed-2p0 (0.8480504639008466).

L0 headers actually read (JSON after magic+u32le):

- gate L0: magic `HGRAVB01`, schema `hawking.gravity.binary_sign_scale.v1`,
  `group_size=128`, `representation=binary_sign_scale`.
- up L0: magic `HGRAVR02`, schema `hawking.gravity.binary_outlier_residual.v2`,
  `outlier_ratio_requested=0.02`, `outlier_count=1782580`, `rice_k=5`,
  `value_bits=1`.
- down L0: magic `HGRAVS01`, schema
  `hawking.gravity.activation_weighted_svd_low_rank.v1`,
  `representation=activation_weighted_svd_low_rank_q`, `rank=160`,
  `factor_bits=3`, `factor_group_size=64`. Matches
  `QWEN38_MIXED_HGRAVS_{RANK,BITS,GROUP}`.
- lm_head: magic `HGRAVU01`, `bits=7`, `group_size=64`,
  `representation=uniform_q7_group_scale`.

### 3.3 Binding at load (GPU)

stderr, verbatim:

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
qwen38 session open 9.209s for 6 prompts
qwen38 prompt 1/6 tokens=11 text="Say hi."
[hawking] HAWKING_TCB_TRACE="(unset)" → mode=Off
qwen38 prompt 2/6 tokens=17 text="Write a function that reverses a string."
qwen38 prompt 3/6 tokens=15 text="What is the capital of France?"
qwen38 prompt 4/6 tokens=19 text="Explain what a hash map is in one sentence."
qwen38 prompt 5/6 tokens=12 text="def fibonacci(n):"
qwen38 prompt 6/6 tokens=13 text="The three primary colors are"
wrote /tmp/qwen38-bracket-generate/QWEN38_FLOOR_Q7_GENERATE_16.json
```

Required bindings:

| check | result |
|---|---|
| `opening mixed HQ38M20` | **yes** (line 1). Not `opening Metal + 755 catalog tensors`. |
| reconstruct-to-Q4 | log says `no reconstruct-to-Q4` |
| `fallbacks_total` | **0** (JSON + every prompt `FALLBACKS: 0`) |
| `DENSE_W_MATERIALIZED` | **0** (printed 0 on every prompt; JSON `dense_w_materialized_total: 0`) |
| expand-to-Q4 / MLX | **no**. Vehicle is this worktree's `ascension_qwen38_hybrid_greedy`. |
| assert refuse | **no**. Process wrote the JSON. |

A fallback-tainted run would be VOID. This one is not.

---

## 4. Phase A — 16 new tokens × 6 prompts (raw)

Prompts file (171 B, the native-2p0 set):

```
/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/coherence_prompts.txt
Say hi.
Write a function that reverses a string.
What is the capital of France?
Explain what a hash map is in one sentence.
def fibonacci(n):
The three primary colors are
```

Tokenizer:
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json`

Rendered (harness, no `--raw-prompt`):
`<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n`

### 4.1 stdout, verbatim

```
PROMPT: Say hi.
GENERATED_TEXT_VERBATIM: ))))))))))))))))
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 11
NEW_TOKENS: [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
WALL_NS: 4924585417

PROMPT: Write a function that reverses a string.
GENERATED_TEXT_VERBATIM: ,,,,,,,,,,,,,,,)
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 17
NEW_TOKENS: [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 8]
WALL_NS: 3464549708

PROMPT: What is the capital of France?
GENERATED_TEXT_VERBATIM: ))))))))))))))))
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 15
NEW_TOKENS: [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
WALL_NS: 3258138292

PROMPT: Explain what a hash map is in one sentence.
GENERATED_TEXT_VERBATIM: ,,,,,,,,,,,,))))
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 19
NEW_TOKENS: [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 8, 8, 8, 8]
WALL_NS: 3688050417

PROMPT: def fibonacci(n):
GENERATED_TEXT_VERBATIM: ))))))))))))))))
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 12
NEW_TOKENS: [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
WALL_NS: 2929374750

PROMPT: The three primary colors are
GENERATED_TEXT_VERBATIM: ))
)
<10 newlines>
))
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 13
NEW_TOKENS: [8, 8, 198, 8, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 8, 8]
WALL_NS: 3038953333
```

The sixth `GENERATED_TEXT_VERBATIM` in the raw stdout is the two-line
form with embedded newlines (token 198). JSON stored it as
`"))\n)\n\n\n\n\n\n\n\n\n\n))"`.

### 4.2 Per-prompt collapse rules (`g1-bracket-bisection.md` §5.4)

Token map from the emitted text: `8` = `)`, `11` = `,`, `198` = `\n`.
The 2p0 collapse alphabet is `{198, 8, 13, 1076, 578, 220}`. Comma (11)
is additional punctuation, still not an English/code word of length ≥ 3.

| prompt | new ids | raw text | punctuation/ws only | cycle ≤4 over ≥8 | only EOS | word ≥3 | `<think>` 248068 |
|---|---|---|---|---|---|---|---|
| Say hi. | 8 × 16 | `))))))))))))))))` | yes | yes (period 1) | no | no | no |
| reverses a string | 11 × 15, 8 | `,,,,,,,,,,,,,,,)` | yes | yes (period 1) | no | no | no |
| capital of France | 8 × 16 | `))))))))))))))))` | yes | yes (period 1) | no | no | no |
| hash map | 11 × 12, 8 × 4 | `,,,,,,,,,,,,))))` | yes | yes (period 1) | no | no | no |
| fibonacci | 8 × 16 | `))))))))))))))))` | yes | yes (period 1) | no | no | no |
| primary colors | 8,8,198,8,198×10,8,8 | `))\n)\n…\n))` | yes | yes (198 run) | no | no | no |

All six fire the INCOHERENT rules. None start with the G0 think preamble.

### 4.3 Greedy-id oracle (signal, not gate)

G0 / seal `Say hi.` first 12
(`QWEN38_COHERENCE_SEAL.json`):

```
[248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149]
= "<think>\nThe user simply wants me to say \"hi.\""
```

This run: `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]`. Prefix match **0/12**.

France seal 12:

```
[248068, 198, 760, 1156, 369, 9859, 264, 4145, 57879, 3296, 25, 3437]
= "<think>\nThe user is asking a simple factual question: What"
```

This run: `[8] × 16`. Prefix match **0/12**. Paris is not present (and
16 tokens cannot establish France→Paris even on G0).

Native 2p0 (already measured, 0 fallbacks) was the same *shape*:
newlines / `)` / `...` / `))`. Q7 attention did not leave that basin.

### 4.4 JSON receipt (authoritative machine record)

`/tmp/qwen38-bracket-generate/QWEN38_FLOOR_Q7_GENERATE_16.json`
(not written under `receipts/` — this lane's write scope is this file).

```json
{
  "artifact_root": "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-floor-q7-v1",
  "dense_w_materialized_total": 0,
  "fallbacks_total": 0,
  "lane": "qwen38-coherence-generate",
  "max_new_tokens": 16,
  "prompts": [
    {
      "generated_text": "))))))))))))))))",
      "new_token_ids": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
      "fallbacks": 0,
      "dense_w_materialized": 0,
      "prompt": "Say hi.",
      "prompt_len": 11,
      "prefill_wall_ns": 3301736292,
      "decode_wall_ns": 1622839500,
      "wall_ns": 4924585417
    },
    {
      "generated_text": ",,,,,,,,,,,,,,,)",
      "new_token_ids": [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 8],
      "fallbacks": 0,
      "dense_w_materialized": 0,
      "prompt": "Write a function that reverses a string.",
      "prompt_len": 17,
      "prefill_wall_ns": 1839516250,
      "decode_wall_ns": 1625030875,
      "wall_ns": 3464549708
    },
    {
      "generated_text": "))))))))))))))))",
      "new_token_ids": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
      "fallbacks": 0,
      "dense_w_materialized": 0,
      "prompt": "What is the capital of France?",
      "prompt_len": 15,
      "prefill_wall_ns": 1621868917,
      "decode_wall_ns": 1636266000,
      "wall_ns": 3258138292
    },
    {
      "generated_text": ",,,,,,,,,,,,))))",
      "new_token_ids": [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 8, 8, 8, 8],
      "fallbacks": 0,
      "dense_w_materialized": 0,
      "prompt": "Explain what a hash map is in one sentence.",
      "prompt_len": 19,
      "prefill_wall_ns": 2059778625,
      "decode_wall_ns": 1628270542,
      "wall_ns": 3688050417
    },
    {
      "generated_text": "))))))))))))))))",
      "new_token_ids": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
      "fallbacks": 0,
      "dense_w_materialized": 0,
      "prompt": "def fibonacci(n):",
      "prompt_len": 12,
      "prefill_wall_ns": 1301421417,
      "decode_wall_ns": 1627951875,
      "wall_ns": 2929374750
    },
    {
      "generated_text": "))\n)\n\n\n\n\n\n\n\n\n\n))",
      "new_token_ids": [8, 8, 198, 8, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 8, 8],
      "fallbacks": 0,
      "dense_w_materialized": 0,
      "prompt": "The three primary colors are",
      "prompt_len": 13,
      "prefill_wall_ns": 1409825291,
      "decode_wall_ns": 1629125666,
      "wall_ns": 3038953333
    }
  ]
}
```

Decode walls after the first prompt sit in a tight band
(~1.623–1.636 s / 16 tokens). That is reported as a cleanliness
observation only. It is **not** a TOKEN_NS or complete-token-wall
result (coexistence + no paired reps).

---

## 5. Phase B and complete-token wall — not run

Phase A collapsed. Per §5.3 / Path A:

- France@128 **not run**. Gate result: n/a. The 16-token France probe
  is `))))))))))))))))` and does not contain `Paris`; that is the
  collapse screen, not the COHERENT gate.
- 17×19@256 **not run**. Gate result: n/a.
- Complete-token wall **not measured**. The artifact did not pass the
  gate. A wall taken under a concurrent G0 reload would also have been
  a dirty number.

---

## 6. Verdict for the attempted artifact

```
artifact: /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-floor-q7-v1
family:   A (2p0 MLP 0.848 + U01 attention/embed bits=7)
complete_physical_bpw (G0, this lane): 3.1767685514394888
codec census at load: B01:64 R02:64 S01:64 U01:659 (u01_bits {7:659})
fallbacks_total: 0
dense_w_materialized_total: 0
path: mixed HQ38M20, no reconstruct-to-Q4
France@128: not run (Phase A collapsed)
arith@256:  not run (Phase A collapsed)
VERDICT: INCOHERENT
reason: binding held; all 6 Phase A prompts are punctuation/whitespace
        only and are period-1 cycles over 16 tokens. Same basin as
        native mixed-2p0. Q7 attention did not rescue the 0.848 MLP.
```

H4 (Q7/Q8 attention rescues the 2p0 MLP) is **falsified for q7**.
Path A says stop. Do not run `mixed-floor-q8-v1` or
`mixed-floor-q8-up10-v1` — same MLP, more attention bits.

**Do not write `Qwen3.8 floor > 3.1768`.** That would smuggle a
family-A statement into the 1-D bracket. Families B and C are still
untested. The open interval remains (2.0856, 4.2527] as a *campaign*
bracket; this lane only adds `floor_A > 3.1767685514394888` for this
exact allocation (crushed MLP + U01-q7 attention).

---

## 7. Artifacts not generated (and why)

CPU catalog parse this lane (same parser as §3.1). No GPU load.

| pack | path | payload | G0 complete BPW | family | HEAD generate | this lane |
|---|---|---:|---:|---|---|---|
| mixed-q4down-v1 | `.../qwen38-27b/mixed-q4down-v1` | 9_948_135_693 | 2.9589935339460913 | B | REFUSE (`down` not S01) | **not attempted** |
| mixed-floor-q7-v1 | `.../qwen38-27b/mixed-floor-q7-v1` | 10_680_295_260 | 3.1767685514394888 | A | RUN | **INCOHERENT** |
| mixed-floor-q8-v1 | `.../qwen38-27b/mixed-floor-q8-v1` | 11_903_200_220 | 3.5405118678698031 | A | RUN | skipped (Path A stop) |
| mixed-q3mlp-v1 | `.../qwen38-27b/mixed-q3mlp-v1` | 12_149_632_429 | 3.6138111608720234 | C | REFUSE (MLP all U01) | **not attempted** |
| mixed-floor-q8-up10-v1 | `.../qwen38-27b/mixed-floor-q8-up10-v1` | 12_203_836_482 | 3.6299337236607006 | A′ | RUN | skipped (Path A stop) |

Family B census (CPU): codecs B01:64 R02:64 U01:723, u01_bits `{4:723}`,
down U01 q4 `uniform_q4_group_scale` g64. `assert_mixed_mlp_native`
requires down = Hgravs. Needs the ~20-line Uniform accept on MLP.
Not a code change this lane is allowed to make. A GPU load would
upload every tensor and then refuse — wasted lock time, not a
coherence verdict.

Family C census (CPU): codecs U01:851, u01_bits `{4:659, 3:192}`,
gate/up/down all `uniform_q3_group_scale` g64. Same assert refuse
on gate/up/down. This is the information-optimal first generate
*after* the assert patch (Path B). Still untested.

Control (not one of the five, not re-run):

| pack | G0 BPW | prior native verdict |
|---|---:|---|
| mixed-2p0-v1 | 2.0855385872764454 | INCOHERENT, 0 fallbacks |
| uniform-q4-v1 (G0) | 4.252735126866492 | COHERENT, 6/6 oracle-32 |

---

## 8. What this does and does not license

Licenses:

- Family A at Q7 attention, native mixed kernels, 0 fallbacks, is
  below its token floor. H4 is false at q7.
- `floor_A` for *this* allocation (B01+R02@2%+S01-r160 down 0.132,
  U01-q7 attention/embed) is `> 3.1767685514394888`.
- Wave-1 attribution (2p0 died of the 0.848 MLP, not of Q4 attention)
  survives a richer-attention check.

Does **not** license:

- `Qwen3.8 floor > 3.18` as a 1-D fact.
- Any statement about family B (q4down, 2.959) or family C (q3mlp, 3.614).
- A 3.18 BPW pack with a different MLP.
- Skipping the assert patch. Path B is still the cheapest *location*
  procedure, and it is still blocked on ~20 lines in
  `assert_mixed_mlp_native`.
- TOKEN_NS / TPS / complete-wall numbers for mixed q7.
- Anything about Q8 attention (not run; same MLP).

---

## 9. Evidence index

- This file.
- `/tmp/qwen38-bracket-generate/q7_phaseA.stdout`
- `/tmp/qwen38-bracket-generate/q7_phaseA.stderr`
- `/tmp/qwen38-bracket-generate/QWEN38_FLOOR_Q7_GENERATE_16.json`
- `/tmp/qwen38-bracket-generate/catalog_census.json`
- Binary sha256 `04108ec95b1d659ea07d9e27b75120cf773f089194bc2af386fd78525f4008a7`
  at `workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy`
- `crates/hawking-core/src/model/qwen38_hybrid_decode.rs:511-512, 588-590, 667, 958-1004`
- `workspace/superwave/g1/g1-bracket-bisection.md` Path A / §5.4
- Prior native 2p0: `receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json`

```
STATUS
MEASURED_NEGATIVE

CLAIMS
1. mixed-floor-q7-v1 generated natively on a just-built
   ascension_qwen38_hybrid_greedy (sha256 04108ec9…), mixed HQ38M20
   path, fallbacks_total=0, DENSE_W_MATERIALIZED=0.
2. Complete BPW recomputed from catalog nbytes: 3.1767685514394888.
   Census B01:64 R02:64 S01:64 U01:659 bits=7. 0 range faults.
3. Phase A 6/6 is punctuation-only collapse (token 8=')', 11=',',
   198=newline). INCOHERENT. Phase B not run. No complete-token wall.
4. H4 is false at Q7. Family-A floor for this allocation is
   > 3.1767685514394888. The campaign 1-D interval is not updated.
5. B and C were not generated. They still die in
   assert_mixed_mlp_native. q8 / q8-up10 skipped by Path A.

EVIDENCE
- §3–§5 command output and raw emitted text
- /tmp/qwen38-bracket-generate/QWEN38_FLOOR_Q7_GENERATE_16.json

CHANGES
Created workspace/superwave/g1/g1-bracket-generate.md only.

TESTS
$ test -s workspace/superwave/g1/g1-bracket-generate.md && echo 'test -s: PASS'
test -s: PASS
$ wc -l workspace/superwave/g1/g1-bracket-generate.md
     592 workspace/superwave/g1/g1-bracket-generate.md
$ git status --porcelain
?? workspace/superwave/g1/g1-bracket-generate.md

RISKS
- genesis-resident 74869 died of its own supervisor recycle ~1 s
  before lock acquire; 15546 reloaded G0 during this oneshot.
  Collapse shape matches 2p0, binding held, so INCOHERENT stands.
  Walls are not evidence.
- A reader will write "floor > 3.18". That is wrong. Family B/C open.

UNRESOLVED
- Family C (q3mlp) and family B (q4down) still have zero generate
  receipts. Both need the assert relaxation.
- Isolated organ floors. No overlay packs.
- Live mixed TOKEN_NS. Not this lane.

NEXT
GPU/code owner: land the assert_mixed_mlp_native Uniform accept
(~20 lines), then Path B Run 1 = mixed-q3mlp-v1 Phase A.
Do not generate q8. Do not treat this file as a 1-D floor raise.
```
