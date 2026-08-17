# G0 baseline remeasure — 2026-08-17

HEAD `2eee9a00493a8631ec7aede5807a3b2292f8370c`  
Host Apple M3 Ultra, 96 GB, 28 cores, 60 GPU cores  
Label: **DIRTY_ENGINEERING** (Claude.app + 6 Claude CLI sessions + ChatGPT/Codex + parallel Grok CPU lanes + live Genesis daemon all up). Absolute numbers do not survive this contamination. Relative regime does.

## Headline (today, live G0)

| quantity | value | class | definition |
|---|---|---|---|
| complete BPW | **4.252735126866492** | MEASURED | `8 * declared_on_disk_bytes / catalog_elements` |
| TOKEN_NS | **39,326,090** | MEASURED | median of 6 paired decode-phase means; see method |
| TPS | **25.4284** | DERIVED | `1e9 / 39_326_090` |
| capability | **coherent** | MEASURED | oracle 32-id match + `17*19=323` emitted |

Historical claim `4.2527 BPW / 26.4 TPS / 37,900,000 TOKEN_NS` is the same genome. BPW matches to every recorded digit. The 37.9 M / 26.4 TPS figure is a **prior complete-wall per-step median** (`QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL`, 00:54 today), **not re-run here**. Today's live-organism decode-phase mean is **+3.8 %** slower. Same regime. Not a material falsification. Today's number is the live baseline.

## 1. Artifact identity (the one the resident actually has loaded)

Live argv (pid 74869, started 10:31):

```
genesis-resident
  --artifact-root /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1
  --tokenizer     /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json
  --socket        /Users/scammermike/Downloads/hawking/workspace/ops/genesis-resident.sock
```

Health after the pair series (`serve_count=14`, `load_count=1`, `generation=0`):

```
artifact     = .../qwen38-27b/uniform-q4-v1
artifact_sha = d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
resident_weight_bytes = 14297675776
pid          = 74869
```

`artifact_sha` is `sha256(manifest.json)` and equals lineage `CURRENT.artifact_sha` and `CURRENT.identity.artifact_sha`.

```
manifest_sha256 d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
manifest_size 238879
schema hawking.ascent.qwen38_language_uniform_q4.v1
```

Tokenizer: `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`  
Resident executable: `ae0bc8defd84a8a1a5cd1c4598224f370c0cfce83a0904e275cbb33df84d32c2`  
(matches `CURRENT.runtime_sha` / `CURRENT.identity.resident_executable_sha256`)  
Q4 shader at this HEAD: `51abdf7be388d62ba080d13a1f97a18ab8b1114c0a6968e9d0f04d109d3efcd1`  
(matches `CURRENT.identity.kernel_source_sha256`)

`load_count` stayed **1** across health-before, all 6 timing reps, and the capability generate. No silent reload. `reload_error=null`.

## 2. Complete BPW — MEASURED from real bytes / real elements

```
source_weight_elements     26895998464
tensor_payload_bytes       14297694680
declared_on_disk_bytes     14297694680   missing=0  byte_mismatch=0
q4_tensors 402  f32_tensors 353  tensor_count 755
catalog.hq38m20_exists     False
bpw_declared_disk          4.252735126866492
complete_physical_bpw      4.252735126866492   # manifest field; same arithmetic
all_tensor_dir_bytes       14308279520
undeclared_bytes           10584840            # leftover *.f32bin, not in catalog
```

Formula (same as `qwen38_pack.rs` 673–679):

```
complete_physical_bpw = 8 * tensor_payload_bytes / source_weight_elements
                      = 8 * 14297694680 / 26895998464
                      = 4.252735126866492
```

`source_weight_elements` is the sum of catalog `elements` over the 755 language tensors (vision skipped at pack: `skipped_vision_tensors=333`). Geometry-major tensors (embed + lm_head + 64×SwiGLU + 48×DeltaNet + 16×GQA + norms) account for 26,810,127,360 of the 26,895,998,464; the 85,871,104 residue is `conv1d` + split `k_proj`/`v_proj` + `A_log`/`dt_bias`/`q_norm`/`k_norm`. The catalog sum is the parameter count.

Resident uploaded bytes `14,297,675,776` are 18,904 below on-disk declared (Q4/`f32v2` headers stripped at upload). That is not a second BPW; it is the same tensors after header peel.

Leftover 353 undeclared `*.f32bin` files (10,584,840 B) sit next to the declared `*.f32v2` payloads of equal total size. Runtime `Qwen38HybridWeights::load` only reads `manifest.json` `artifact` names. They are not in the loaded set.

**Claim 4.2527 BPW: SUPPORTED, exact.**

## 3. Codec the runtime actually selected

`load()` (`crates/hawking-core/src/model/qwen38_hybrid_decode.rs:508–514`) prefers `catalog.hq38m20` (mixed) if present; otherwise the uniform-q4 manifest. The live artifact has **no** `catalog.hq38m20`.

Live load log (`workspace/ops/genesis-resident.log`, pid 74869):

```
qwen38-decode opening Metal + 755 catalog tensors
qwen38-decode upload 0/755
...
qwen38-decode upload 750/755
genesis-resident: body resident 3.435s weight_bytes=14297675776
genesis-resident: listening .../genesis-resident.sock pid=74869
```

That string is the uniform-q4 branch, not `opening mixed HQ38M20`. A missing codec fails the run (`refusing silent fallback` on unknown magic).

On-disk Q4 magic of a declared tensor:

```
q4_sample ...hq30uq4  b'HQ30UQ4\x00'
```

Catalog kinds: 402 `q4` (`*.hq30uq4`) + 353 `f32` (`*.f32v2`). Nominal codec BPW 4.25, group size 64. Dispatch kernel used by this session is `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` (same genome as the complete-wall identity; shader sha above).

No silent fallback to mixed / HGRAVU / expand-to-float.

## 4. TOKEN_NS / TPS — MEASURED on the live body

### Method (what this is, and is not)

Paired A/B × 3 on the **already-resident** process, session `protected_test`, `protected_capability_prompt_preserved`, prompt `Say hi.`, `max_new_tokens=32`.

Timer is `generate_greedy` (`qwen38_hybrid_decode.rs:3377–3415`):

- **Prefill** = teacher-forced walk of all `prompt_len=11` ids. The last prefill step emits new-token[0] (first-token latency).
- **Decode** = the loop that produces new-tokens[1..]. `decode_steps = n_new - 1 = 31`.
- **TOKEN_NS_rep** = `decode_wall_ns / decode_steps` (mean of the decode-phase wall).
- Headline = median of the 6 rep means (`sorted[len//2]`).
- TPS = `1e9 / headline`. Not measured separately. Not projected.

This is **complete-token adjacent**: `session.step()` wall includes encode + submit + wait + GPU. It is **not** the `generate_greedy_complete_wall` per-step median (that also folds tokenizer-decode + bookkeeping into each step; historical tokenizer_decode median was 6,208 ns, ~0.016 % of 39 ms).

`ascension_qwen38_hybrid_greedy --complete-wall` was **built from this HEAD** (`release-fast`, sha `7ea00e23715ec31f18eb4745bff1c5cee00c4afef1d8c8e6814aa50d66baeb35`) and **not executed**. Reason: resident holds the 13.6 GB weight set; `pages free` during the measurement window was 7,856–168,486 (128 MB–2.7 GB). A second Metal upload would swap. The resident was not stopped.

### Pair table (all 6 ok, fallbacks=0, load_count=1, pid=74869)

Source: `/tmp/g1-baseline-remeasure/resident_measure_v2.json`

| rep | decode_wall_ns | decode_steps | TOKEN_NS (mean) | TPS (derived) | prefill_wall_ns | n_new | oracle32 |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 | 1,206,114,084 | 31 | 38,906,905.94 | 25.7024 | 549,837,042 | 32 | match |
| B1 | 1,209,772,791 | 31 | 39,024,928.74 | 25.6246 | 445,613,709 | 32 | match |
| A2 | 1,219,108,791 | 31 | 39,326,090.03 | 25.4284 | 435,372,209 | 32 | match |
| B2 | 1,226,198,209 | 31 | 39,554,780.94 | 25.2814 | 445,172,917 | 32 | match |
| A3 | 1,228,417,834 | 31 | 39,626,381.74 | 25.2357 | 434,395,042 | 32 | match |
| B3 | 1,217,522,500 | 31 | 39,274,919.35 | 25.4615 | 449,805,166 | 32 | match |

Spread of the 6 TOKEN_NS means:

```
all    [38906905.94, 39024928.74, 39274919.35, 39326090.03, 39554780.94, 39626381.74]
min    38906905.94
median 39326090.03
max    39626381.74
range  719475.81   (1.83 % of median)
```

Tight spread. This is a real number, not a page-cache draw.

A discarded earlier A1 (v1, same body, same prompt) was 39,075,629 ns — inside the same band.

Prefill is separated: 434–550 ms for 11 prompt tokens (~39–50 ms/token including the first-new-token emit). A1 prefill is the cold-of-series outlier; B1–B3 sit at 434–450 ms, consistent with 11 × ~39 ms.

### Versus the historical claim

| | TOKEN_NS | TPS | method | when |
|---|---:|---:|---|---|
| claimed / lineage CURRENT | 37,879,375 | 26.3996 | complete-wall per-step median, 6 warm reps | 00:54 today |
| this lane, live G0 | 39,326,090 | 25.4284 | decode-phase mean, 6 warm reps | 11:10 today |
| delta | +1,446,715 (+3.8 %) | −0.97 | different timer, dirtier box | |

`QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.verification.json` `derived_rep_median_complete_wall_ns` = `[37869000, 37858625, 37837833, 37879542, 37959250, 37879375]`. That series is tighter and ~1.4 ms faster. It was `release-fast` `ascension_qwen38_hybrid_greedy` sha `f9e03b7e5114dc2cc3f49b1833f728412aa27fca9092bd2feb83cdb9101a34c3`, not the live `release` `genesis-resident`. I did not re-execute that binary.

G1 targets (`TOKEN_NS <= 10,000,000`, `TPS >= 100`) are **not** the current G0. G0 is ~25 TPS.

## 5. Capability — MEASURED, not a checkbox

### 5a. Greedy identity (timing prompt)

All 6 reps emitted the same 32 ids, matching the current-main receipt and the 16-id coherence seal prefix:

```
[248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149,
 1061, 369, 264, 1546, 4145, 11, 2050, 1622, 13, 353, 3172, 1066, 1910,
 15131, 303, 264, 11321, 11, 5629, 1560]
```

`fallbacks=0` on every rep. Text (truncated at 32 tokens, as designed):

```
<think>
The user simply wants me to say "hi." This is a very simple, direct request. I'll just say hi in a friendly, natural way
```

That is the same opening the 00:54 complete-wall receipt recorded.

### 5b. Arithmetic (live body, 256 new tokens, 11:11:56)

Prompt: `What is 17 times 19? Reply with the integer product, then one short sentence showing the arithmetic. No other preamble.`

```
<think>
The user wants me to calculate 17 × 19 and reply with just the integer product followed by one short sentence showing the arithmetic. No other preamble.

17 × 19 = 17 × 20 - 17 = 340 - 17 = 323

Let me verify: 17 × 19 = 17 × (20 - 1) = 340 - 17 = 323. Yes.

Format: integer product, then one short sentence showing the arithmetic. No other preamble.
</think>

323

17 × 19 = 17 × (20 − 1) = 340 − 17 = 323.
```

`ok=true fallbacks=0 n_new=168`. The model is not a fluent nonsense server.

Raw: `/tmp/g1-baseline-remeasure/capability_256.json`.

## 6. Machine state (absolute numbers are dirty)

At measurement (11:01–11:12):

- `genesis-resident` pid 74869 live; `ascent_daemon.py loop` + `genesis_forever.sh` live; daemon injects 900-token parent proposes between windows. Pair series ran in an 18 s idle gap.
- GPU lock `/tmp/hawking-gpu-lane.lock` owner `genesis-resident:parent` pid 74869. Not stolen. Not released.
- Claude.app pid 73987 open. Six `claude` CLI sessions. ChatGPT/Codex.app open. Ollama, Blender (9876), Chrome, OrbStack.
- Parallel Grok CPU lanes holding multiple-GB python jobs (`g1_sparse_exact_islands.py`, `g1_vq_sweep.py`, `qwen38_shared_basis.py`, `g1_entropy_measure.py`, earlier `analyze_weights.py` at 14 GB).
- load averages 9–13. `pmset -g therm`: no thermal / performance / CPU-power warning recorded. `powermetrics` not available without sudo.
- IOAccelerator during the pair series: Device Utilization 97–99 %. After: 0 %.
- `pages free` swung 1.2 M → 7 k → 168 k. Swapins 8.8 M / swapouts 15.6 M already on the box before this lane.
- This worktree built `ascension_qwen38_hybrid_greedy` `release-fast` in 42.48 s to `workspace/ops/build/rust` (gitignored). That binary was **not** the measurement vehicle.

## 7. What was not measured

- Per-step `complete_wall_ns` median (`--complete-wall --pairs 3 --max-new-tokens 32`). Built, not run. Cheapest experiment: wait until `pages free > ~1e6` (16 GB) **and** the resident is idle, then

  ```
  workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy \
    --artifact-root /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1 \
    --tokenizer     /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json \
    --prompt "Say hi." --complete-wall --pairs 3 --max-new-tokens 32 \
    --out /tmp/g1-baseline-remeasure/QWEN38_COMPLETE_TOKEN_WALL.json
  ```

  Do not start that while the resident is generating. Do not kill the resident to free the GPU.

- Isolated-family GPU timestamps / TOKEN_NS ledger decomposition. Component microbenchmarks are not token-level claims; none are offered as TOKEN_NS.
- Bandwidth roof. Bytes/quoted-peak is not a floor. Not computed.

## 8. Verdict on the claimed historical regime

| claim | verdict |
|---|---|
| complete BPW 4.2527 | **SUPPORTED** — exact, from live artifact bytes / catalog elements |
| TOKEN_NS 37,900,000 | **same regime, not re-run as complete-wall**. Live decode-phase headline today is 39,326,090. Prior complete-wall median 37,879,375 (00:54, different binary, cleaner timer). +3.8 % is inside dirty-box noise + mean-vs-median. |
| TPS 26.4 | **DERIVED from the 37.9 M claim**. Today's derived TPS is 25.43. |
| capability | **SUPPORTED** — oracle ids + emitted `323` |

G0 is a 4.25 BPW uniform-Q4 Qwen3.8 that decodes at ~25 TPS on this box today. G1 still has to beat that, not the 100 TPS poster.

---

```
STATUS
SUPPORTED

CLAIMS
1. complete BPW = 4.252735126866492 (MEASURED). Evidence: §2 command dump; manifest sha d650a757…; 8*14297694680/26895998464.
2. Live G0 TOKEN_NS headline = 39,326,090 (MEASURED, decode-phase mean, 6 paired reps). Evidence: /tmp/g1-baseline-remeasure/resident_measure_v2.json spread_rep_mean_decode_token_ns.
3. TPS = 25.4284 (DERIVED = 1e9/39326090). Evidence: same file headline_tps_from_decode_phase_mean.
4. Codec selected at load = uniform HQ30UQ4 catalog, 755 tensors, no mixed fallback (MEASURED). Evidence: genesis-resident.log "opening Metal + 755 catalog tensors"; catalog.hq38m20 absent; magic HQ30UQ4\\0.
5. Artifact measured = live resident path + sha d650a757… + load_count=1 (MEASURED). Evidence: health JSON; argv of pid 74869.
6. Capability preserved (MEASURED). Evidence: 6/6 oracle-32 match; capability_256.json emits "323".
7. Historical 37.9 M / 26.4 TPS is the same genome's earlier complete-wall median, not today's live decode-phase number (MEASURED vs prior receipt). Evidence: QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.verification.json vs §4 table.

EVIDENCE
- /tmp/g1-baseline-remeasure/resident_measure_v2.json
- /tmp/g1-baseline-remeasure/capability_256.json
- /tmp/g1-baseline-remeasure/resident_measure.json (v1 A1 only)
- /Users/scammermike/Downloads/hawking/workspace/ops/genesis-resident.log
- /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json
- receipts/ascent-2026-08-16/QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.verification.json (prior, not re-run)
- crates/hawking-core/src/model/qwen38_hybrid_decode.rs:508-514, 3377-3415
- crates/hawking-core/src/model/qwen38_pack.rs:673-679

CHANGES
workspace/superwave/g1/g1-baseline-remeasure.md (new)

TESTS
```
$ test -s workspace/superwave/g1/g1-baseline-remeasure.md && echo PASS
PASS
$ wc -l workspace/superwave/g1/g1-baseline-remeasure.md
     304 workspace/superwave/g1/g1-baseline-remeasure.md
$ git status --porcelain
?? workspace/superwave/g1/g1-baseline-remeasure.md
```

RISKS
- TOKEN_NS is decode-phase mean on the live release body, not a complete-wall per-step median. Tokenizer-decode (~6 µs historical) is outside the interval.
- Box is DIRTY. +3.8 % vs 00:54 complete-wall can be contamination, mean-vs-median, or release vs release-fast. Not partitioned.
- Second-copy complete-wall was not run. That is the remaining hole, not a license to treat 37.9 M as today's number.

UNRESOLVED
Official `--complete-wall` series on the this-HEAD binary. Blocked by coresident 13.6 GB + <3 GB free. Recipe in §7.

NEXT
G1 work should treat 4.2527 BPW / ~25.4 TPS / 39.3 M TOKEN_NS as the live G0 to beat, and 37.9 M as an earlier cleaner complete-wall on the same artifact. Do not plan as if G0 were 100 TPS.
```
