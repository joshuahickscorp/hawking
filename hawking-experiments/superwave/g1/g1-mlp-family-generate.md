# G1 MLP family generate — mixed-q3mlp-v1 / mixed-q4down-v1

Lane: `92-mlp-family-generate`. Native generate only. No runtime/loader/shader/packer
edit. No repack. Live G0 not touched. Resident Genesis not stopped.

Every number is **MEASURED** unless tagged PACK_REPORT or DIRTY_ENGINEERING.

---

## 0. Verdict

| artifact | complete BPW (G0 def) | MLP BPW | load | fallbacks | dense W | gate | verdict |
|---|---:|---:|---|---:|---:|---|---|
| mixed-q3mlp-v1 | 3.6138111608720234 | 3.2500251321231617 | mixed HQ38M20, 0 refuse | 0 | 0 | France 128 contains Paris (×8); 17×19 256 contains 323 (×3) | **COHERENT** |
| mixed-q4down-v1 | 2.9589935339460913 | 2.2208531248803234 | mixed HQ38M20, 0 refuse | 0 | 0 | 16-token newline/blank collapse; France has no Paris | **INCOHERENT** |

MLP hypothesis: **supported**. Family-A 0.848 MLP (sub15 / 2p0 / floor-q7) all collapsed
with richer attention. Restoring only `down_proj` to Q4 (family B, MLP 2.221, gate still
HGRAVB01 @ 1.125, up still HGRAVR02 @ 1.288) still collapses. Replacing **all three**
MLP GEMVs with HGRAVU01 Q3 (family C, MLP 3.250) clears the campaign gate.

The constraint is MLP recipe, not complete BPW and not attention richness. Bits belong
on gate/up/down as Uniform, not on Binary-gate + rice-up + crushed-down.

q3mlp is a G1 promotion **candidate** (3.614 < 4.2527, native, 0 fallbacks). It is not
a 6/6 oracle-32 seal: after the correct fact it loops (`Assistant`, `few`, fence
ticks). The gate is substring, and it passed. Quality is below live G0.

q3mlp complete-token wall (DIRTY_ENGINEERING, genesis resident + other lanes live):

```
headline complete_wall = 148588917 ns/tok = 148.588917 ms/tok
headline TPS           = 6.7300   MEASURED (1e9/148588917)
headline GPU           = 146963124 ns/tok
rep medians (A1 B1 A2 B2 A3 B3) ns:
  148424792  148588917  148460333  148480250  148721583  148600250
spread min/max         = 148424792 / 148721583   (0.297 ms)
control decode wall    = 148614061 ns/tok
```

G0 live baseline (prior, not remeasured here): 4.252735126866492 BPW, TOKEN_NS
39,326,090, TPS 25.4284. q3mlp is slower — Q3 Uniform hits `q80_hgravs01_factor_matvec_simd3`,
not the G0 Q4 GEMV. Do not substitute a projected TPS.

Neither run is VOID.

---

## 1. Identity of this process

```
HEAD        b2ed67cb084d3b50162d03b63c6ebe2babfaefa9
            (lane 91 MLP role-lock unlock; this worktree)
decode.rs   sha256 5638aab3e77c3b829c43d71c76999238cc60d41f0b8792fe4b6b65c144ae7ac6
binary      /Users/scammermike/.claude-grok/worktrees/91-mlp-rolelock-unlock-20260817-120135/workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy
            sha256 a9d41d09856ff7ffb7e32ff4fd4f7ad49cafc4301afd4f6814f51f8418898fab
            built this process 2026-08-17T16:27:44Z from this HEAD
            cargo build --release -p hawking-core --example ascension_qwen38_hybrid_greedy --offline
            CARGO_TARGET_DIR = 91-lane release target (same HEAD, incremental)
vehicle     ascension_qwen38_hybrid_greedy
            --max-seq-len 512   no --raw-prompt
tokenizer   .../qwen38-27b/bf16/tokenizer.json
N           26895998464
```

Stale Aug-16 binary at Downloads/hawking `workspace/ops/build/rust/release/examples/`
was **not** used (mtime 2026-08-16 22:36, pre-unlock).

---

## 2. Live G0 seal

Path: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json`

```
PRE  (before first generate)  d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
after q3mlp 16                d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
after q3mlp France 128        d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
after q3mlp arith 256         d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
after q3mlp complete-wall     d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
after q4down 16               d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
POST (end of lane)            d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
```

Unchanged. Required value. Resident pid 50196 **ALIVE** at every check
(`genesis-resident` RSS ~16.9 GB, `weight_bytes=14297675776`). Socket still listening.
This lane never sent the resident a request and never wrote into `uniform-q4-v1`.

---

## 3. mixed-q3mlp-v1

### 3.1 Path / self-containment / BPW

```
root   /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-v1
```

Catalog parse this process (HQ38M20 v1, 851 tensors, 258 segments):

| quantity | value | tag |
|---|---:|---|
| sum(catalog nbytes) | 12_149_632_429 | MEASURED |
| elements | 26_895_998_464 | MEASURED (= N) |
| complete physical BPW = 8 × nbytes / N | **3.6138111608720234** | MEASURED, G0 definition |
| PACK_REPORT complete_physical_bpw | 3.6138647373176767 | PACK_REPORT (includes catalog.hq38m20 180124 B) |
| MLP bytes / elems / BPW | 6_952_112_640 / 17_112_760_320 / **3.2500251321231617** | MEASURED |
| attn BPW | 4.2501555848196455 | MEASURED |
| embed / lm_head BPW | 4.250001799593266 | MEASURED |
| gate = up = down BPW | 3.2500251321231617 each | MEASURED |

Self-containment **MEASURED**:

- 0 absolute segment filenames.
- 0 paths resolve outside this root.
- 0 missing segment files.
- 0 payloads past EOF.
- 258/258 unique segment paths live under `mixed-q3mlp-v1/segments/`.
- 66 of those files are hardlinked (nlink=3) with `mixed-2p0-v1` and `mixed-q4down-v1`
  (embed + Lxx attention slices). Catalog names are relative `segments/<file>`.
  The loader never leaves this root.
- **standalone_root = YES.** This is a verdict about this artifact as a root, not
  about a silent sibling open. Contrast mixed-sub15-v1 (absolute filenames into
  another pack). Slack leftover 2p0 MLP payloads may sit unused inside hardlinked
  Lxx files; they are **not** in the nbytes sum and **not** in BPW.

L0 MLP magics (peek at catalog offset):

```
layers.0.mlp.gate_proj  codec 3  HGRAVU01  shape [17408,5120]  nbytes 36208920
layers.0.mlp.up_proj    codec 3  HGRAVU01  shape [17408,5120]  nbytes 36208920
layers.0.mlp.down_proj  codec 3  HGRAVU01  shape [5120,17408]  nbytes 36208920
```

All 192 MLP rows codec 3. All 851 catalog rows codec 3.

### 3.2 Codec census at load (every open)

Log line, identical on 16 / France 128 / arith 256 / wall:

```
qwen38-decode opening mixed HQ38M20 + 851 catalog tensors (no reconstruct-to-Q4)
qwen38-decode mixed census: tensors=851 binary=0 residual=0 hgravs=0 uniform=498 q4=0 f32=353 refused=0 expanded_to_q4=0 expanded_to_float_gemv=0
```

Never `opening Metal + 755 catalog tensors`. 498 Uniform + 353 f32 (HGRAVU01
vectors ≤65536 dequant to f32, not GEMV expand) = 851. `q4=0`.

```
fallbacks_total          = 0     MEASURED (every prompt, every invocation)
dense_w_materialized     = 0     MEASURED (example prints 0; load census
                                       expanded_to_q4=0 expanded_to_float_gemv=0
                                       is the authority)
```

### 3.3 Phase A — 16 tokens × 6 prompts

Command:

```
tools/gpu_lane_lock.sh qwen38-mlp-family-q3mlp \
  $BIN --artifact-root $ART/mixed-q3mlp-v1 --tokenizer $TOK \
  --prompts-file $ART/coherence_prompts.txt \
  --max-new-tokens 16 --max-seq-len 512
session open 5.231s
```

Waited on `genesis-resident:child_b` lock ~2 min, then ran. 2026-08-17T16:29:00Z–16:31:53Z.

Raw text is Python `repr` of `generated_text`:

| prompt | verbatim repr | ids | fb | wall_ns |
|---|---|---|---:|---:|
| Say hi. | `'\nHi! How can I help you today? How can I help you today'` | `[198, 12675, 0, 2500, 628, 353, 1438, 488, 3242, 30, 2500, 628, 353, 1438, 488, 3242]` | 0 | 3952441917 |
| Write a function that reverses a string. | `'```python\nreverse_string = lambda x: x[::-1]\n```\n'` | `[71093, 12305, 198, 25075, 3773, 283, 12102, 830, 25, 830, 60059, 16, 60, 198, 71093, 198]` | 0 | 4750275083 |
| What is the capital of France? | `'Assistant\nAssistant\nAssistant\nAssistant\nAssistant\nAssistant\nAssistant\nAssistant\n'` | `[69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198]` | 0 | 4454956500 |
| Explain what a hash map is in one sentence. | `'```\n``` ```\n```\n```\n```\n```\n```\n```'` | `[71093, 198, 71093, 52451, 198, 71093, 198, 71093, 198, 71093, 198, 71093, 198, 71093, 198, 71093]` | 0 | 5050095625 |
| def fibonacci(n): | `'```python\ndef fibonacci(n):\n    if n <= 0:\n'` | `[71093, 12305, 198, 727, 73111, 1393, 1590, 198, 262, 413, 307, 2564, 220, 15, 25, 198]` | 0 | 4011372917 |
| The three primary colors are | `'```\nThe three primary colors are red, blue, and yellow.\n```'` | `[71093, 198, 760, 2250, 5839, 7736, 513, 2438, 11, 6105, 11, 321, 13358, 13, 198, 71093]` | 0 | 4152760667 |

16-token France has **no** Paris. 16-token cannot declare COHERENT. Not
punctuation-only collapse (hi / reverse / fibonacci / colors are words), so Phase B
ran.

### 3.4 Phase B — France 128

Prompt: `What is the capital of France?`  (chat-templated; prompt_len=15)
`--max-new-tokens 128`. 2026-08-17T16:32:32Z. FALLBACKS 0. DENSE_W 0.

**GENERATED_TEXT_VERBATIM:**

```
Assistant
Assistant
Assistant
Assistant
Assistant
Assistant
Assistant
Assistant
Assistant
Answer: Paris. Answer: Paris. Answer:
Answer: Answer: Answer:
Answer: Paris. Answer: Paris. Answer: What is the capital of
What is the capital of France? Answer: Paris. Answer: Paris. Answer: Paris. Answer: Paris. Answer: What is the

What is the
What is the capital of France? Answer: multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple many many many many many many
What is the capital of France? What is
```

**new_token_ids** (n=128):

```
[69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 15666, 25, 11751, 13, 21134, 25, 11751, 13, 21134, 25, 198, 15666, 25, 21134, 25, 21134, 25, 198, 15666, 25, 11751, 13, 21134, 25, 11751, 13, 21134, 25, 3437, 369, 279, 6511, 314, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 25, 11751, 13, 21134, 25, 11751, 13, 21134, 25, 11751, 13, 21134, 25, 11751, 13, 21134, 25, 3437, 369, 279, 271, 3710, 369, 279, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 25, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 1599, 1599, 1599, 1599, 1599, 1599, 198, 3710, 369, 279, 6511, 314, 9338, 30, 3437, 369]
```

`"Paris" in text` = **True** (8 times). Token 11751 (`Paris`) count = 8.
wall_ns=21352690042  prefill=2307428125  decode=19045259917
median_gpu_ns_per_token=148290791

### 3.5 Phase B — 17×19 at 256

Prompt: `What is 17 times 19? Reply with the integer product, then one short sentence showing the arithmetic. No other preamble.`
prompt_len=36. `--max-new-tokens 256`. FALLBACKS 0. DENSE_W 0.

**GENERATED_TEXT_VERBATIM** (tilde fence: the model emitted markdown fences):

~~~~
```
323
323 = 17 × 19
```
```
323
3
```
```
```

```
32
```
```
```
```
```

```
```
```
```
```
```
few few few few few few few few few few few few few few few few few few few few few few few few few few few few
few few few few few few
```
```
```

```
```

```
```

```

```
```
```
replies: few few few few few few few few
replies: few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few few
```
```
```

```
```

```
 asked asked asked asked asked asked asked asked asked asked asked asked
 asked asked asked asked asked asked
```
```
```
```
```

```
```
```
replies: few few few few few
replies:
~~~~

**new_token_ids** (n=256):

```
[71093, 198, 18, 17, 18, 198, 18, 17, 18, 283, 220, 16, 22, 23985, 220, 16, 24, 198, 71093, 198, 71093, 198, 18, 17, 18, 198, 18, 198, 71093, 198, 71093, 198, 71093, 198, 198, 71093, 198, 18, 17, 198, 71093, 198, 71093, 198, 71093, 198, 71093, 198, 71093, 198, 198, 71093, 198, 71093, 198, 71093, 198, 71093, 198, 71093, 198, 71093, 198, 68336, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 198, 68336, 2342, 2342, 2342, 2342, 2342, 198, 71093, 198, 71093, 198, 71093, 198, 198, 71093, 198, 71093, 198, 198, 71093, 198, 71093, 198, 198, 71093, 198, 198, 71093, 198, 71093, 198, 71093, 198, 265, 6976, 25, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 198, 265, 6976, 25, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 2342, 198, 71093, 198, 71093, 198, 71093, 198, 198, 71093, 198, 71093, 198, 198, 71093, 198, 4439, 4439, 4439, 4439, 4439, 4439, 4439, 4439, 4439, 4439, 4439, 4439, 198, 4439, 4439, 4439, 4439, 4439, 4439, 198, 71093, 198, 71093, 198, 71093, 198, 71093, 198, 71093, 198, 198, 71093, 198, 71093, 198, 71093, 198, 265, 6976, 25, 2342, 2342, 2342, 2342, 2342, 198, 265, 6976, 25]
```

`"323" in text` = **True** (3 times). Opens `323` then `323 = 17 × 19`, then
degenerates to fence ticks / `few` / `asked`.
wall_ns=43880746500  prefill=5439322125  decode=38441420333
median_gpu_ns_per_token=148925625

### 3.6 Gate and quality

Campaign gate (this contract): France 128 contains Paris **and** 17×19 256 contains
323. **Both true. COHERENT.**

Not claimed: 6/6 oracle-32, parent-prefix match, or G0-equal text. After the fact
the model loops. A promotion still needs a capability seal if one is required
beyond this gate.

### 3.7 Complete-token wall (run because gate cleared)

Machine at wall start 2026-08-17T16:34:14Z:

```
GENESIS_ALIVE pid=50196
MEM free=19.01GB anon=40.08GB file=32.46GB compressor=4.30GB
LOCK FREE
```

Command:

```
tools/gpu_lane_lock.sh qwen38-mlp-family-q3mlp-wall \
  $BIN --artifact-root $ART/mixed-q3mlp-v1 --tokenizer $TOK \
  --prompt "Say hi." --complete-wall --pairs 3 \
  --max-new-tokens 32 --max-seq-len 512
session open 2.606s
```

Census at open: same 851 / uniform=498 / f32=353 / refused=0 / expanded=0.
fallbacks=0 on cold, 6 warm, and control. Greedy ids identical across warm reps
(example fails the process on drift).

```
COMPLETE_WALL_NS_PER_TOKEN: 148588917
COMPLETE_WALL_MS_PER_TOKEN: 148.588917
COMPLETE_WALL_TPS: 6.7300
GPU_NS_PER_TOKEN: 146963124
WALL_MINUS_GPU_NS: 1625793
REP_MEDIANS_NS: [148424792, 148588917, 148460333, 148480250, 148721583, 148600250]
CONTROL_DECODE_WALL_NS_PER_TOKEN: 148614061
GENERATED_TEXT_VERBATIM:
Hi! How can I help you today? How can I help you today? How can I help
you today?

"How can I help you today
FALLBACKS: 0
```

Authority set = 3 A/B pairs = 6 warm in-process generates after one discarded cold.
Headline = median of the 6 per-rep medians of per-step complete_wall_ns on decode
steps (31 decode steps/rep).

```
spread_rep_median_complete_wall_ns
  n=6 min=148424792 median=148588917 mean=148546020.833 max=148721583
spread_rep_median_gpu_ns
  n=6 min=146875874 median=146963124 mean=146962381.167 max=147087916
pooled_steady n=186
  wall min=147840125 median=148586125 max=149567959
  gpu  median=146969916
timing_label = DIRTY_ENGINEERING
timing_label_reason = GPU lock held for the series; other CPU/memory lanes may
                      still be live. Not offered as CLEAN_CANDIDATE or BASE_TRUE_TPS.
```

COLD first step wall=237170958 ns gpu=149488124 ns prefill=1720457375 ns
(discarded from authority).

Greedy ids (32):

```
[198, 12675, 0, 2500, 628, 353, 1438, 488, 3242, 30, 2500, 628, 353, 1438, 488, 3242, 30, 2500, 628, 353, 1438, 198, 9053, 3242, 30, 271, 68552, 628, 353, 1438, 488, 3242]
```

Spread is 0.297 ms across 6 rep medians on a busy box. The 148.6 ms / 6.73 TPS
figure is a **measured complete-token wall**, dirty, not a projection. It is not
G0's 25.43 TPS. Q3 Uniform is a different kernel.

---

## 4. mixed-q4down-v1

### 4.1 Path / self-containment / BPW

```
root   /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-q4down-v1
```

| quantity | value | tag |
|---|---:|---|
| sum(catalog nbytes) | 9_948_135_693 | MEASURED |
| elements | 26_895_998_464 | MEASURED |
| complete physical BPW = 8 × nbytes / N | **2.9589935339460913** | MEASURED, G0 definition |
| PACK_REPORT complete_physical_bpw | 2.9590429283570026 | PACK_REPORT (catalog 166064 B) |
| MLP bytes / elems / BPW | 4_750_615_904 / 17_112_760_320 / **2.2208531248803234** | MEASURED |
| gate BPW (HGRAVB01) | 1.1250234267290902 | MEASURED |
| up BPW (HGRAVR02) | 1.2875108157887178 | MEASURED |
| down BPW (HGRAVU01 bits=4) | 4.250025132123162 | MEASURED |
| attn / embed / lm_head | same as q3mlp (hardlinked Q4 U01) | MEASURED |

Self-containment **MEASURED**: same shape as q3mlp. 0 abs filenames, 0 outside
paths, 0 missing, 130/130 segment files under this root. 66 hardlinked with
2p0 + q3mlp. **standalone_root = YES.** Slack leftover 2p0 S01 down payloads
inside hardlinked Lxx files are unreferenced and not in BPW.

L0 MLP magics:

```
layers.0.mlp.gate_proj  codec 0  HGRAVB01  [17408,5120]  nbytes 12534021  in L00.hq38seg
layers.0.mlp.up_proj    codec 1  HGRAVR02  [17408,5120]  nbytes 14344242  in L00.hq38seg
layers.0.mlp.down_proj  codec 3  HGRAVU01  [5120,17408]  nbytes 47350040  replace_..._down_proj_weight.hq38seg
```

Catalog: codec 0 ×64, codec 1 ×64, codec 3 ×723.

### 4.2 Codec census at load

```
qwen38-decode opening mixed HQ38M20 + 851 catalog tensors (no reconstruct-to-Q4)
qwen38-decode mixed census: tensors=851 binary=64 residual=64 hgravs=0 uniform=370 q4=0 f32=353 refused=0 expanded_to_q4=0 expanded_to_float_gemv=0
session open 4.281s for 6 prompts
```

64 B01 + 64 R02 + 370 U01 (64 Q4 down + 306 attn/embed GEMV) + 353 f32 = 851.
`hgravs=0` (down is Uniform, not S01). `q4=0` (no HQ30UQ4 lane). Unlock admitted
Uniform-on-down; load did not refuse.

```
fallbacks_total = 0
dense_w_materialized = 0
```

Not VOID.

### 4.3 Phase A — 16 tokens × 6 prompts

2026-08-17T16:35:18Z–16:35:49Z. Lock free.

Raw text is Python `repr` of `generated_text` (so newlines stay visible):

| prompt | verbatim repr | ids | fb | wall_ns |
|---|---|---|---:|---:|
| Say hi. | `'6....\n\nSay hi.\n\n\n\n\n\n00.0'` | `[21, 13, 13, 13, 13, 271, 44240, 15131, 13, 271, 271, 271, 15, 15, 13, 15]` | 0 | 3860669917 |
| Write a function that reverses a string. | `'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'` | `[198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198]` | 0 | 4638546458 |
| What is the capital of France? | `'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'` | `[198, 198, 198, 198, 271, 271, 271, 271, 271, 271, 271, 271, 271, 271, 271, 271]` | 0 | 4347733250 |
| Explain what a hash map is in one sentence. | `'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'` | `[271, 271, 271, 271, 271, 271, 271, 271, 271, 271, 271, 271, 271, 271, 271, 271]` | 0 | 4925779875 |
| def fibonacci(n): | `'def fibonacci(n):\ndef\ndef\n\n\n\n\n\n\n\n'` | `[727, 73111, 1393, 1590, 198, 727, 198, 727, 198, 198, 198, 198, 198, 198, 198, 198]` | 0 | 3914892417 |
| The three primary colors are | `'\n\nThe three primary colors are the\n\n\n\n\n\n\n\n'` | `[198, 198, 760, 2250, 5839, 7736, 513, 279, 198, 198, 198, 198, 198, 198, 198, 198]` | 0 | 4056299125 |

France 16: no Paris. Degenerate newline/blank cycle, same family as native
mixed-2p0 (`[198]×16`). **INCOHERENT** by the 16-token rule. Phase B not run
(16-token establishes INCOHERENT and never COHERENT). No complete-token wall.

---

## 5. What this does to the floor

Prior native 0-fallback collapses, all MLP 0.848:

```
mixed-sub15-v1     1.2910781930062503   MLP 0.848   "the the the..." / space-a cycle
mixed-2p0-v1       2.0855934079220506   MLP 0.848   [198]×16 / "))))..."
mixed-floor-q7-v1  3.1767685514394888   MLP 0.848   token 8 ×16
```

This lane:

```
mixed-q4down-v1    2.9589935339460913   MLP 2.221   INCOHERENT  (gate B01 + up R02 kept)
mixed-q3mlp-v1     3.6138111608720234   MLP 3.250   COHERENT    (all MLP U01 Q3)
```

So:

1. Attention-side bits do not rescue 0.848 MLP. Already known; still true.
2. Restoring **only down** to Q4 does **not** rescue. H3 (bracket-bisection) is
   **refuted**. Gate B01 @ 1.125 and/or rice up @ 1.288 remain below their floor
   even when down is Q4.
3. Q3 Uniform on **gate + up + down** (cosine hold ≥ 0.9679, pack min 0.9653)
   **does** clear the campaign gate. H2 is **supported**.
4. The floor is not a 1-D complete-BPW cut. q4down at 2.959 complete is dead;
   q3mlp at 3.614 complete lives; floor-q7 at 3.177 complete was dead because
   its MLP was still 0.848.
5. Packer implication: spend the next bits on Uniform gate/up (lift off B01/R02),
   not on attention above Q4, not on down-only Q4 while leaving Binary+rice.

Not measured here: Q3-gate + Q3-up + S01-down; Q3-gate + R02-up + Q4-down; a
Uniform-Q3 pack with attention also cut. The two on-disk MLP-varying artifacts
were generated. That is the lane.

---

## 6. Non-goals held

- Runtime / loader / shader / packer: not modified.
- No new model artifact.
- Live G0 directory and hardlinked inodes: not written.
- Resident Genesis: not stopped, not restarted, not pointed at these roots.
- Only write: this file.
- No commit, push, merge, deploy.

---

## 7. Raw load logs (authority for VOID check)

q3mlp 16 stderr (census + bind):

```
qwen38-decode opening mixed HQ38M20 + 851 catalog tensors (no reconstruct-to-Q4)
qwen38-decode mixed upload 0/851
...
qwen38-decode mixed upload 850/851
qwen38-decode mixed census: tensors=851 binary=0 residual=0 hgravs=0 uniform=498 q4=0 f32=353 refused=0 expanded_to_q4=0 expanded_to_float_gemv=0
qwen38-decode mixed bind: K-complete; recon_fuse=ON uses q80_binary_group_matvec_simd_bytes / q80_binary_group_csr_matvec_bytes when cols>2048 (256-col tiles; this model K in {5120,6144}); cols<=2048 stay on tg256; recon_fuse=0 walks every column via gk_matvec_binary
qwen38 session open 5.231s for 6 prompts
```

q4down 16 stderr (census + bind):

```
qwen38-decode opening mixed HQ38M20 + 851 catalog tensors (no reconstruct-to-Q4)
qwen38-decode mixed census: tensors=851 binary=64 residual=64 hgravs=0 uniform=370 q4=0 f32=353 refused=0 expanded_to_q4=0 expanded_to_float_gemv=0
qwen38-decode mixed bind: K-complete; recon_fuse=ON uses q80_binary_group_matvec_simd_bytes / q80_binary_group_csr_matvec_bytes when cols>2048 (256-col tiles; this model K in {5120,6144}); cols<=2048 stay on tg256; recon_fuse=0 walks every column via gk_matvec_binary
qwen38 session open 4.281s for 6 prompts
```
