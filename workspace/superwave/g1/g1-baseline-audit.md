# G1 baseline audit — claimed G0 numbers

Audit HEAD: `2eee9a00493a8631ec7aede5807a3b2292f8370c` (`main`, 2026-08-17).
No GPU / Metal / inference was run. Artifact bytes were read from the live
uniform-q4-v1 catalog on this box. Timing claims were traced only to
receipts already in git.

Claimed G0 triple (unverified campaign numbers):

| quantity | claimed |
|---|---|
| complete BPW | 4.2527 |
| TPS | 26.4 |
| TOKEN_NS | 37,900,000 |

These three numbers are **not one measurement**. BPW is a pack-catalog
quotient. 26.4 and 37,900,000 are a rounded TEXT n=1 wall. The seated G0
identity uses a third pair. A later locked complete-token authority uses a
fourth.

## Verdicts

| claimed | verdict | what it actually is |
|---|---|---|
| 4.2527 BPW | **DEFENSIBLE** | `complete_physical_bpw` of `workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1`. Independently recomputed from the live manifest. Storage quotient, not a runtime. |
| 26.4 TPS | **REFUTED** as the G0 baseline | Seated G0 TPS is **28.3866**. 26.4 is `GENESIS_POOL.md` / `GENESIS_CHILDREN_CAPACITY.json` N=1 TEXT, no GPU lock, 1 completion. |
| 37,900,000 TOKEN_NS | **REFUTED** as the G0 baseline | No receipt field equals 37,900,000. It is `37855081` rounded to 100 k. Seated G0 TOKEN_NS is **35,227,918**. Locked complete-token authority is **38,216,792**. |

The claimed triple as a single G0 baseline is **FALSIFIED**.

---

## 1. 4.2527 BPW — DEFENSIBLE

### Receipt

Live artifact, not a git blob (the catalog is not in the repo):

`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json`

```
complete_physical_bpw: 4.252735126866492
source_weight_elements: 26895998464
tensor_payload_bytes: 14297694680
q4_tensors: 402
f32_tensors: 353
skipped_vision_tensors: 333
min_q4_cosine: 1.0
status: CANDIDATE_QWEN38_LANGUAGE_Q4_FUSED_INPROJ
schema: hawking.ascent.qwen38_language_uniform_q4.v1
```

Formula in `crates/hawking-core/src/model/qwen38_pack.rs:673-678`:

```
let source_weight_elements: u64 = rows.iter().map(|row| row.elements).sum();
let tensor_payload_bytes: u64 = rows.iter().map(|row| row.bytes).sum();
let complete_physical_bpw = if source_weight_elements == 0 {
    0.0
} else {
    (tensor_payload_bytes as f64 * 8.0) / source_weight_elements as f64
};
```

Independent recompute (this lane, no GPU):

```
$ python3 - <<'PY'
import hashlib, json
p='/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json'
d=json.load(open(p))
print(d['complete_physical_bpw'])
print(d['tensor_payload_bytes']*8.0/d['source_weight_elements'])
print('q4_cosine_None', sum(1 for t in d['tensors'] if t['kind']=='q4' and t.get('cosine') is None))
print('manifest_sha256', hashlib.sha256(open(p,'rb').read()).hexdigest())
PY
4.252735126866492
4.252735126866492
q4_cosine_None 402
manifest_sha256 d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
```

Tensor-row sum equals the manifest fields (755 rows, 0 delta). Catalog files
on disk sum to 14,297,694,680 B, matching `tensor_payload_bytes`.

4.2527 is the 4-decimal rounding of that quotient. Same literal is hardcoded
as `UNIFORM_Q4_V1_BPW` in `crates/hawking-core/src/model/qwen38_token_ns_ledger.rs:30`
and as `GENESIS_BPW` in `lab/lineage/identity.py:24`.

### What the number is

MEASURED (catalog arithmetic), not projected.

It is **complete physical storage BPW of the language-only pack**:
payload bits / source elements, including the embed table
(`language_model.model.embed_tokens.weight`, 1,271,398,400 elems,
675,430,440 B Q4) that is not streamed per token except one gathered row.

Active-per-token bytes from
`receipts/ascent-2026-08-16/QWEN38_ACTIVE_BUDGET_MEASURED.json`:
`active_bytes_per_token = 13622264240` (embed table excluded).
That is a different numerator. Because embed is the same Q4 codec, excluding
it from both sides changes BPW only in the fourth decimal
(`13622264240 * 8 / (26895998464 - 1271398400) = 4.2518…`).
The claimed 4.2527 is the storage number, not the served-per-token number.

Q4 tensors alone are 4.250025 BPW (group-64 codes + f16 scale). The extra
0.0027 is 353 f32 norm tensors (2,645,504 elems at 32 BPW).

Vision is excluded by construction (`skipped_vision_tensors: 333`). That is
the G0 vehicle, not a hidden omission.

### Failure-mode checks

| mode | result |
|---|---|
| stale HEAD | N/A for a pack quotient. Artifact mtime 2026-08-16 03:15. Number does not depend on runtime HEAD. |
| wrong artifact | Path matches every G0 receipt. Bytes match the manifest. |
| silent kernel/codec fallback | Not a runtime number. |
| component as token | Not a timing number. |
| projected as measured | Not projected. |
| page-cache unpaired | N/A. |
| skipped-green | **YES — quality gate.** See below. |
| capability never checked | BPW is not capability. Related: `min_q4_cosine` is not a measurement. |

`min_q4_cosine: 1.0` is the fold identity, not a vs-bf16 cosine.

`qwen38_pack.rs:305-312` (`try_reuse_q4`) writes `cosine: None` when a Q4
file already exists. `qwen38_pack.rs:680-684`:

```
let min_q4_cosine = rows
    .iter()
    .filter(|row| row.kind == "q4")
    .filter_map(|row| row.cosine)
    .fold(1.0f64, f64::min);
```

All 402 Q4 rows in the live manifest have `cosine: None`. The fold never
sees a value and reports 1.0. This is the "checked nothing, reported
success" pattern. A different receipt
(`THREE_MODEL_REGIME_SPLIT.json:50`) claims `q4_min_cosine_vs_bf16: 0.98948`.
That number is **not** on the sealed catalog.

353 leftover `*.f32bin` files sit next to the catalog (10,584,840 B). They
are not in the manifest and are not in the BPW numerator. Harmless for the
quotient; the tensors directory is not a clean catalog.

`GENESIS_CHILDREN_CAPACITY.json:8` labels the same artifact
`artifact_on_disk_gb: 8.5`. Payload is 14.30 GB (13.31 GiB). 8.5 is false.
IOAccelerator dirty on the same receipt is 14,476,197,888 B, which matches
the 14.3 GB catalog, not 8.5. They timed the real pack and mislabeled its
size.

G0 `artifact_sha` is not a content hash of this catalog.

`lab/lineage/canon.py:29-30`:

```
def labeled_sha(label: str) -> str:
    return hashlib.sha256(f"hawking.lineage/{label}".encode("utf-8")).hexdigest()
```

`lab/lineage/identity.py:245`:

```
artifact_sha=labeled_sha("artifact/qwen38-27b/uniform-q4-v1"),
```

```
$ python3 -c "import hashlib; print(hashlib.sha256(b'hawking.lineage/artifact/qwen38-27b/uniform-q4-v1').hexdigest())"
56dd65d465f31741f8d40a86d84de779a939fdd9b9b90ecd3d1cb4f82aa4287a
```

That equals `GENESIS_LINEAGE_CURRENT.json:9`. Manifest sha256
`d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df` is never
bound. Swapping the catalog would not move `artifact_sha`.

### Cheapest re-measurement

No GPU. Re-sum the live catalog (already done). To bind identity, replace
`labeled_sha("artifact/...")` with sha256 of `manifest.json` plus a
manifest-ordered hash of catalog file bytes. To make `min_q4_cosine` real,
recompute dequant-vs-bf16 on a named tensor subset (or refuse to report 1.0
when every cosine is `None`).

---

## 2. 26.4 TPS — REFUTED (as G0 baseline)

### Where 26.4 actually lives

`receipts/ascent-2026-08-16/GENESIS_POOL.md:37-41`:

```
Complete-token wall (DIRTY, TEXT, no GPU lock), aggregate tok/s = n / token_s:

| n | complete token | aggregate tok/s | vs N=1 |
| 1 | 37.855 ms      | 26.4            | 1.00   |
```

`receipts/ascent-2026-08-16/GENESIS_CHILDREN_CAPACITY.json:28-41` and `:144`:

```
"n": 1,
"seq_len": 128,
"max_new_tokens": 64,
"complete_token_ns": 37855081,
"completions": 1,
"label": "MEASURED"

"n1": {"complete_token_ns": 37855081, "tok_per_s": 26.42}
```

```
1e9 / 37855081 = 26.416533093668455
```

MEASURED, not projected. Regime: TEXT, `hold_gpu_lock: false`, one
completion, DIRTY. The same file names the locked timing authority as a
**different** number (`:11`, `:165`):

```
"complete_token_wall_ns_authority": 35227918
"why_lock": "A contended TIMING is not a number. The 35,227,918 ns ledger was taken under the lock."
```

### What G0 actually sealed

`lab/lineage/identity.py:23-25`:

```
GENESIS_COMPLETE_TOKEN_NS = 35_227_918
GENESIS_BPW = 4.2527
GENESIS_TPS = 28.4
```

`receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json:12-14`
(`instance_id: genesis-qwen38-g0`, `generation: 0`):

```
"representation_bpw": 4.2527,
"complete_token_ns": 35227918,
"tps": 28.386576805362157,
```

```
1e9 / 35227918 = 28.386576805362157
```

That TPS is derived from the G024 / ledger wall, not from 37.855 ms.

### Competing locked complete-token TPS (not 26.4 either)

`receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json:5-7,32-36`:

```
"timing_label": "DIRTY_ENGINEERING",
"not_base_true_tps": true,
"not_clean_candidate": true,
"complete_wall_ns_per_token": 38216792,
"complete_tps": 26.1665,
```

6 warm A/B reps under `./tools/gpu_lane_lock.sh`. Headline = median of
per-rep medians. This is the receipt whose merge commit
`23cbb80d3` (2026-08-16 16:13:37 -0400) is titled
"Merge the Qwen3.8 complete-token wall authority, **which corrects my estimate**".

G0 was seated later the same day (`61b138210`, 22:39:54 -0400) on the
**uncorrected** 35.2 M / 28.4 TPS pair.

Tournament paired speed (`GENESIS_TOURNAMENT_RESULT.json:25-29`),
label `CLEAN_CANDIDATE`, GPU lock, one prompt, 67 tokens:

```
"steady_ns_per_token": 37576039, "tps": 26.6
```

### Failure-mode checks

| mode | result |
|---|---|
| stale HEAD | G024 timing commit is `57ee82ccef7aba803416ec3562c8981277120fd4`. `git rev-list --count 57ee82c..HEAD` = **143**. `qwen38_hybrid_decode.rs` changed +3159/−245 in that range (mixed reader + wall instrumentation). Production kernel `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` body is unchanged (diagnostic probes appended after it). `load()` takes mixed only if catalog magic is `HQ38M20`; uniform-q4-v1 still takes the Q4 path. Timing at HEAD is **unmeasured**. |
| wrong artifact | Same `uniform-q4-v1` path. Children header `8.5 GB` is wrong; resident IOAccelerator 14.48 GB matches the real catalog. |
| silent fallback | Receipts report `fallbacks: 0`. Decode refuses silent codec/GEMV fallback (`qwen38_hybrid_decode.rs` "refusing silent fallback" / "refuses a run after a fallback"). No evidence this TEXT n=1 run fell off `geo_tpr64_tg128`. Identity `kernel_genome_sha` is `labeled_sha("genome/Qwen38HybridDecodeSession+qwen_uniform_q4_group64")` and cannot detect a swap. |
| component as token | 26.4 is a complete-token wall, not a component. Not this failure. |
| projected as measured | 26.4 is measured. The **projected** figures are the density ladder (`ms * target_bpw/4.2527`): 63.4 TPS at 2.0 BPW from G015 GPU, 55.64 TPS from the 38.217 ms authority wall (`QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json:89-93`). Do not confuse those with 26.4. |
| page-cache unpaired | **YES.** `completions: 1`. No A/B pair. No GPU lock. TEXT. Adjacent receipt `GENESIS_RESIDENT_BODY.json` on the same artifact the same day: "A preceding Qwen3.8 lane had just dropped the same artifact, so file/Metal upload was cache-hot." |
| skipped-green | Children receipt labels the run `MEASURED` with n=1. No skipped test produced 26.4. |
| capability never checked | This is a timing number. Capability scores on the same G0 object are assigned 1.0 (section 4). |

26.4 is a real TEXT-regime sample. It is not the seated G0 TPS, not the
locked authority TPS, and not a BASE_TRUE number. Attributing it as "G0
baseline TPS" is a wrong-receipt claim.

### Cheapest re-measurement

Owned by the serialized GPU lane, not this one.

`./tools/gpu_lane_lock.sh` + `target/release/examples/ascension_qwen38_hybrid_greedy`
on HEAD, same `uniform-q4-v1` artifact, authority definition
(discard 1 cold generate; 3 A/B pairs; median of per-rep medians of
decode-step `complete_wall_ns`; new-tokens[1..] only). Report that
headline and `1e9/headline_ns`. Do not use TEXT n=1 and do not reuse
35,227,918.

---

## 3. 37,900,000 TOKEN_NS — REFUTED (as G0 baseline)

### No receipt equals 37,900,000

Nearest MEASURED field: `37855081` in
`GENESIS_CHILDREN_CAPACITY.json:38` (same TEXT n=1 run as 26.4).

```
round(37855081, -5) = 37900000
```

That is the claimed integer. It is a rounding of a single unlocked TEXT
completion, not a ledger identity.

### What the receipts actually record

| source | ns | label | lock | pairing |
|---|---:|---|---|---|
| `QWEN38_TOKEN_NS_LEDGER.json:median_wall_ns` / `TOKEN_NS_QWEN38.json:TOTAL_TOKEN_NS` / `G024_QWEN38_TOKEN_NS.json:22` | **35,227,917** | DIRTY_ENGINEERING | implied (G024 production) | 3 paired generates after 4 discarded tokens |
| seated G0 / `identity.py:23` / `GENESIS_LINEAGE_CURRENT.json:13` | **35,227,918** | G024 sum-of-components (1 ns rounding) | cited as locked | same as G024 |
| tournament `CLEAN_CANDIDATE` | **37,576,039** | CLEAN_CANDIDATE | GPU lock | one prompt, 67 tokens |
| children TEXT n=1 seq=128 | **37,855,081** | MEASURED, TEXT, no lock | no | 1 completion |
| **claimed** | **37,900,000** | rounded children n=1 | no | 1 |
| authority min of 6 rep-medians | 37,922,375 | DIRTY_ENGINEERING, not_base_true | GPU lock | 6 A/B |
| authority headline | **38,216,792** | same | GPU lock | 6 A/B |
| authority 16-new confirmation | 38,543,084 | same | GPU lock | 6 A/B |
| children TEXT n=1 seq=8192 | 40,996,195 | MEASURED, TEXT | no | 1 |

G024 / ledger (`G024_QWEN38_TOKEN_NS.json:8-25,36-41`,
`QWEN38_TOKEN_NS_LEDGER.json:7-10`):

```
"label": "DIRTY_ENGINEERING",
"commit": "57ee82ccef7aba803416ec3562c8981277120fd4",
"median_wall_ns": 35227917,
"steady_decode_steps": "all post-prompt steps across 3 generates",
"total_token_ns": 35227917,
"sum_components_ns": 35227918,
```

G024 is a closed 12-component ledger that equals **its** wall. That does
not make 35.2 M the complete-token definition used by the later authority.

The authority receipt (`QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json:79-86`)
explicitly retracts the 33.537 ms GPU / leftover-prefill denominator that
G024 treated as in-band (`this_run_minus_g015_ms: 0.375`):

```
"answer": "NO",
"recorded_g015_gpu_ms": 33.537,
"this_session_gpu_ms": 36.987,
"this_session_complete_wall_ms": 38.217,
"why": "... 33.537 is a different session's GPU median over a different
denominator (drop first step only; leftover prefill kept)."
```

GPU 33.912 ms (G024) vs 36.987 ms (authority) is a 3.075 ms gap on the
same named artifact, both DIRTY. Wall 35.228 ms vs 38.217 ms is 8.5%.
G0 was seated on the lower number after the higher number was already
merged as the correction.

### Failure-mode checks

| mode | result |
|---|---|
| stale HEAD | Same as TPS. G024 at `57ee82c`, 143 commits behind HEAD. Authority merge is also not HEAD. No TOKEN_NS receipt is attributed to `2eee9a004`. |
| wrong artifact | Same `uniform-q4-v1`. Ledger `weight_bytes.active_bytes = 13618141856` is **geometry**, not the measured catalog `13622264240`. Delta 4,122,384 B. Byte budget in the 35.2 M ledger is not the live artifact sum. |
| silent fallback | `fallbacks: 0` on G024, authority, resident body. Production Q4 kernel body unchanged since `57ee82c`. Mixed path is magic-gated. |
| component as token | **Partial.** 37.9 M is presented as TOKEN_NS (token-level) and is a TEXT complete-token wall, not a component. The seated 35.2 M **is** a component-sum that equals a wall taken under a looser denominator ("all post-prompt steps") than the authority ("new-tokens[1..] only"). G024 rank-1 component `weight_addressing = 21,293,103` is not being sold as 37.9 M. |
| projected as measured | 37.9 M is a rounded measurement, not a projection. |
| page-cache unpaired | **YES** for the claimed integer (n=1 TEXT). Authority 38.2 M is paired (6 A/B) and still DIRTY / not_base_true. |
| skipped-green | No skipped test emitted 37,900,000. |
| capability never checked | TOKEN_NS is timing. See section 4. |

### Cheapest re-measurement

Same locked complete-token wall as section 2. The TOKEN_NS to promote
against is that headline `complete_wall_ns`, not 35,227,918 and not
37,900,000. Re-emit `TOKEN_NS_QWEN38` from that run. This lane must not
do it.

---

## 4. Capability and skipped-green (cross-cutting)

G0 capability is assigned, not scored.

`lab/lineage/identity.py:28-32,252`:

```
DEFAULT_CAPABILITY_CONTRACT = {
    "coherence": 1.0,
    "complete_token_discipline": 1.0,
    "engineering": 1.0,
}
...
capability=dict(DEFAULT_CAPABILITY_CONTRACT),
```

`GENESIS_LINEAGE_CURRENT.json:15-18` copies those 1.0s onto the live
`genesis-qwen38-g0` object. `runtime_sha` and `kernel_genome_sha` are
also `labeled_sha(...)` of strings
(`identity.py:246-248`). They do not hash the binary or the metallib.

What **was** checked, narrowly:

- `QWEN38_COHERENCE_SEAL.json`: 12 greedy ids on 3 prompts (`Say hi.`,
  reverse-a-string, capital of France). Binary path is a scratch
  `ascension_qwen38_hybrid_greedy`.
- G024 / authority: 16-id prefix match on `Say hi.` only;
  `greedy_bit_identical: true`, `fallbacks: 0`.
- Tournament: Qwen3.8 beat Q80 on 4/5 tasks. T3 ENGINEER "measured the
  PLAN, not a verified complete-system improvement"
  (`GENESIS_TOURNAMENT_RESULT.json` `what_this_tournament_did_NOT_measure`).

What was **not** checked at seat time:

- No numeric rubric produced `coherence=1.0` / `engineering=1.0` /
  `complete_token_discipline=1.0`.
- `benchmark_fingerprint` is `labeled_sha("bench/complete-token/qwen38/greedy/3prompt/gpu-cb-timestamps")`,
  not a hash of any bench output.
- `min_q4_cosine=1.0` is an empty fold (section 1).
- `GENESIS_LINEAGE_REPRODUCTION.json:161-167`: required
  `python3 -m pytest -q lab/tests/` is `BLOCKED_PREEXISTING`
  (`64 failed, 889 passed, 6 skipped, 7 errors`). The 55 lineage tests
  that passed are gate **unit** tests, including
  `protected_test_skipped` → REJECT. They do not run the G0 vehicle.

Capability for G0 is therefore **never actually checked** at the contract
layer. A 12-token 3-prompt seal exists beside it. Those are different
objects.

---

## 5. What G1 may treat as G0

| quantity | use | do not use |
|---|---|---|
| BPW | 4.252735126866492 complete-physical storage of the live `uniform-q4-v1` catalog. Language-only, vision skipped, embed included. | 8.5 GB implied BPW. `min_q4_cosine=1.0` as quality. |
| TOKEN_NS / TPS | No existing number is BASE_TRUE at HEAD. Closest locked complete-token wall is 38,216,792 ns / 26.17 TPS (`not_base_true_tps: true`, DIRTY, 6 A/B, 2026-08-16, not this HEAD). | 26.4 / 37,900,000 (TEXT n=1). 28.4 / 35,227,918 (seated, superseded denominator, stale commit). 29.9 GPU-only (`THREE_MODEL_REGIME_SPLIT.json:7-8`). 63.4 / 55.64 projected. |
| capability | Unscored. Re-run the 3-prompt seal on HEAD before claiming preservation. | The 1.0/1.0/1.0 map. |

A G1 candidate that beats 26.4 TPS without a locked complete-token wall
on the same definition has not beaten G0.

---

## 6. Command log (this lane)

```
$ git rev-parse HEAD
2eee9a00493a8631ec7aede5807a3b2292f8370c

$ git rev-list --count 57ee82ccef7aba803416ec3562c8981277120fd4..HEAD
143

$ git log -1 --format='%h %ci %s' 57ee82ccef7aba803416ec3562c8981277120fd4
57ee82cce 2026-08-16 14:56:19 -0400 Qwen3.8 is at 98.7 percent of the decode ceiling - question resolved from existing data

$ git log -1 --format='%h %ci %s' 23cbb80d3
23cbb80d3 2026-08-16 16:13:37 -0400 Merge the Qwen3.8 complete-token wall authority, which corrects my estimate

$ git log -1 --format='%h %ci %s' 61b138210
61b138210 2026-08-16 22:39:54 -0400 Seat Genesis generation 0 and put the resident model in the loop

$ git diff --stat 57ee82ccef7aba803416ec3562c8981277120fd4 HEAD -- \
    crates/hawking-core/src/model/qwen38_hybrid_decode.rs \
    crates/hawking-core/shaders/qwen_uniform_q4.metal \
    crates/hawking-core/examples/ascension_qwen38_hybrid_greedy.rs
 .../examples/ascension_qwen38_hybrid_greedy.rs     |  640 +++-
 crates/hawking-core/shaders/qwen_uniform_q4.metal  |   99 +
 .../hawking-core/src/model/qwen38_hybrid_decode.rs | 3159 ++++++++++++++++++--
 3 files changed, 3653 insertions(+), 245 deletions(-)
```

Independent catalog / SHA / arithmetic: section 1 and the python block
under Verdicts' supporting numbers.

No Metal, no generate, no pack, no live Genesis process touched.

---

## Completion report

```
STATUS
FALSIFIED

CLAIMS
- CLAIM 4.2527 BPW is complete_physical_bpw of uniform-q4-v1: DEFENSIBLE. Evidence: live manifest complete_physical_bpw=4.252735126866492; recomputed payload*8/elems identical; qwen38_pack.rs:675-678.
- CLAIM 26.4 TPS is the G0 baseline: REFUTED. Evidence: identity.py:25 GENESIS_TPS=28.4; GENESIS_LINEAGE_CURRENT.json:14 tps=28.386576805362157; 26.4 is GENESIS_POOL.md:41 / CHILDREN_CAPACITY.json:144 TEXT n=1 no lock.
- CLAIM 37,900,000 TOKEN_NS is the G0 baseline: REFUTED. Evidence: no field equals 37900000; 37855081 rounded; seated 35227918 (identity.py:23, LINEAGE_CURRENT.json:13); locked authority 38216792 (COMPLETE_TOKEN_WALL_AUTHORITY.json:34).
- CLAIM the three numbers are one G0 measurement: REFUTED. Evidence: BPW is pack arithmetic; 26.4/37.9M are TEXT n=1; seated G0 pairs 4.2527 with 28.4/35.2M.
- CLAIM G0 capability 1.0/1.0/1.0 was measured: REFUTED. Evidence: identity.py:28-32,252 DEFAULT_CAPABILITY_CONTRACT assigned; labeled_sha identity hashes; min_q4_cosine empty fold.
- CLAIM 35,227,918 remains the honest locked complete-token wall: SUSPECT / superseded. Evidence: 23cbb80d3 "corrects my estimate"; AUTHORITY.json:79-86 leftover-prefill retraction; GPU 33.912 vs 36.987 ms.

EVIDENCE
- HEAD 2eee9a00493a8631ec7aede5807a3b2292f8370c; 143 commits after G024 commit 57ee82cce.
- Live catalog /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json sha256 d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df.
- receipts/ascent-2026-08-16/{G024_QWEN38_TOKEN_NS.json,QWEN38_TOKEN_NS_LEDGER.json,TOKEN_NS_QWEN38.json,QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json,GENESIS_LINEAGE_CURRENT.json,GENESIS_CHILDREN_CAPACITY.json,GENESIS_POOL.md,GENESIS_TOURNAMENT_RESULT.json,QWEN38_ACTIVE_BUDGET_MEASURED.json,GENESIS_LINEAGE_REPRODUCTION.json,GENESIS_RESIDENT_BODY.json}.
- lab/lineage/identity.py, lab/lineage/canon.py, crates/hawking-core/src/model/qwen38_pack.rs.

CHANGES
- Created workspace/superwave/g1/g1-baseline-audit.md only.

TESTS
- test -s / wc -l / git status --porcelain: see operator paste after this file exists.

RISKS
- Timing at HEAD is unmeasured (decode.rs +3k lines since G024). Production Q4 kernel body looks unchanged; that is a source diff, not a timing result.
- Live Genesis was not inspected (forbidden). LINEAGE_CURRENT in git may not match the running process.
- Artifact lives outside the worktree (Downloads/hawking/...). This worktree has no copy.

UNRESOLVED
- Honest locked complete-token wall on commit 2eee9a004. Cheapest answer: GPU-lock authority protocol on this HEAD.
- Whether the resident organism is still serving uniform-q4-v1 via geo_tpr64_tg128.
- Real vs-bf16 cosine of the sealed catalog.

NEXT
- GPU lane: re-measure complete-token wall on HEAD; retire 26.4/37.9M and 28.4/35.2M as G0 timing.
- Bind artifact_sha to manifest+catalog bytes.
- Treat DEFAULT_CAPABILITY_CONTRACT 1.0 as unset until a live 3-prompt seal is re-run.
```
