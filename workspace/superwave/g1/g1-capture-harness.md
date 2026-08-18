# G1 capture harness — the activation substrate every later fit is missing

Lane: `51-capture-harness`. Three new files. No GPU. No forward. No model load.
No resident touch. The existing 256-token cube and its receipt were not modified.

STATUS: IMPLEMENT_READY. The tool is ready to fire. A separate GPU lane runs it
when the machine is quiet. This lane proved the adequacy gate by making it FAIL.

Every number is labeled MEASURED (command or on-disk file), CITED (wave-1 / doctor
recovery, not re-derived), or ESTIMATED (arithmetic on those).

---

## 0. Why this lane exists

The live cube at

```
/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1
```

is MEASURED `n_tokens=256`, `wall_s=14.967979082999591`, `source=PARENT_BF16_REAL`.
It is a real BF16 post-norm hidden dump of width 5120. It is not a mixer site,
not a SwiGLU site, and not a confirmed final-norm site.

Fitting anything of dimension 6144 or 17408 on 256 rows gives

| claim | n | dim | rpd | vs NS-014 92/2048 = 0.044921875 |
|---|---:|---:|---:|---:|
| out_proj / act_colscale | 256 | 6144 | 0.041667 | **worse** |
| down_proj | 256 | 17408 | 0.014706 | 3× worse |
| this gate, prompt-held | 185 | 6144 | 0.030111 | **worse** |

The last row is MEASURED by the new gate on the existing receipt: prompt-level
25% hold-out leaves `n_fit=185`, `n_hold=71`. The activation-column-scale result
that dropped L0 `out_proj` cosine `0.9922374383267348 → 0.9186496062432181` is
the same underdetermined class. It is dead as allocator input until this
harness has been run.

---

## 1. What was built

| path | role |
|---|---|
| `tools/qwen38_capture_v2.py` | harness + adequacy gate + `--plan` / `--run` / `--check-adequacy` |
| `tools/tests/test_qwen38_capture_v2.py` | 16 tests, including the must-fail 256×6144 case |
| `workspace/superwave/g1/g1-capture-harness.md` | this report |

The recovered doctor instrument at `7c5f323d9` stays unmerged. The gate here is
the tightened allocation form of `unit_determination`: **n_fit >= fit_dim or
the number is REFUSED**. REFUSED writes `score=null`, `emit_score=false`, and
raises if `.require()` is called. It does not emit 1.0, 0.0, or any other
foldable default.

Parent BF16 remains the only legal fit source. The Q4 twin is additional.
`mixed-2p0` / `mixed-sub15` / expand-to-float are REFUSED.

---

## 2. The gate was observed FAILING

That is the point of this lane. A gate never seen to fail is not evidence.

### 2.1 Existing 256-token capture vs a 6144-dim fit — REFUSED

```
$ python3 tools/qwen38_capture_v2.py --check-adequacy --fit-dim 6144 --procedure fit_from_X
```

MEASURED output (exit 1):

```
{
  "status": "REFUSED",
  "determination": "UNDERDETERMINED",
  "procedure": "fit_from_X",
  "n_rows": 256,
  "n_fit": 185,
  "n_hold": 71,
  "n_prompts": 5,
  "fit_dim": 6144,
  "rows_per_dim": 0.030110677083333332,
  "eval_thin": true,
  "emit_score": false,
  "score": null,
  "reason": "REFUSED: fit_from_X n_fit=185 < fit_dim=6144 (rows_per_dim=0.030111). NS-014 wreck was 92/2048=0.044922. This number is not emitted.",
  "holdout_by_prompt": true,
  "x_source": "PARENT_BF16_REAL",
  "worse_than_ns014": true,
  "verdict": "REFUSED"
}
VERDICT: REFUSED
```

`gated_score` does not call the score function on a refused unit. Test
`test_gated_score_does_not_call_fn_or_emit_default_on_refuse` asserts the
callback is never invoked and that `1.0` is not substituted.

### 2.2 Synthetic adequate case — ACCEPTED

`N=23216`, 69 sequences, designated prompt hold `n_hold=5804`, `n_fit=17412`.
`17412 >= 17408`. Test `test_synthetic_adequate_case_is_accepted` passes.

### 2.3 Other refusals the gate actually emits

- NS-014 shape `92 × 2048`
- `rank = min(budget, n_fit)` silent starve
- interpolated layer (UNMEASURED)
- row-shuffle holdout (leaks the same prompt into fit and hold)
- `mixed-2p0` / incoherent BPW as X
- `eval_weight_only` on 256×6144 is ACCEPTED *as an eval* and flagged
  `eval_thin`. It is not a fit and must not set a bit on an unswept layer.

---

## 3. Capture spec the GPU lane will run

Settled in `g1-doctor-recovery.md` §5–§6. Implemented, not redesigned.

| item | value |
|---|---|
| N | **23216** |
| sequences | **69** (>= 64) |
| hold | prompt-level 25%, designated `n_hold=5804`, `n_fit=17412` |
| min / max seq | 32 / 2048 (the 10-token v1 mass is gone) |
| mass mix | prose 5804, code 4643, math 3482, instruction 3482, long 2322, multilingual 2322, adversarial 1161 |
| long sequences | 3 × 774 tokens, in [512, 2048] |
| sites, every layer 0..63 | `post_input_norm` 5120, `post_attn_norm` 5120, `post_swiglu` 17408, `mixer_x` 6144 |
| plus | `final_norm` 5120 × 1 (confirmed `language_model.model.norm`, not L63 post-norm) |
| vehicles | parent BF16 (legal fit source) + Q4 twin (additional, native, no expand) |
| storage | f16, site-split N, stream per layer, never a resident cube |
| mixer_x | `out_proj` / `o_proj` **input**. DeltaNet: gated recurrent mix after `norm(recurrent, z)`. GQA: `repeat(v)*sigmoid(q_gate)` after softmax. A `v*silu(z)` proxy is labeled DEGENERATE and REFUSED. |

Site-split N (the 67 GB option):

| site | store_n | n_fit at 25% | fit_dim | adequacy |
|---|---:|---:|---:|---|
| post_input_norm | 6827 | 5120 | 5120 | ACCEPTED |
| post_attn_norm | 6827 | 5120 | 5120 | ACCEPTED |
| post_swiglu | 23216 | 17412 | 17408 | ACCEPTED |
| mixer_x | 8192 | 6144 | 6144 | ACCEPTED |
| final_norm | 6827 | 5120 | 5120 | ACCEPTED |

Receipt is v1-readable: `n_tokens`, `n_layers`, `hidden`, `prompts`, `per_layer`,
`wall_s`, `fit_kind`, `sha256_self`, `source`. v2 adds `sites`, `adequacy`,
`mixer_x_kind`, `holdout`. `per_layer` points at `post_input_norm` so existing
JSON consumers still parse.

---

## 4. Exact GPU-lane command

Do **not** run this from a builder lane. Do **not** run it while the resident
holds the GPU lock. Do **not** expand Q4 to float.

```
QWEN38_CAPTURE_I_HOLD_GPU=1 \
python3 tools/qwen38_capture_v2.py --run \
  --vehicles parent \
  --store-mode site-split-n \
  --out /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v2
```

Wrap it in `tools/gpu_lane_lock.sh` so it is the only GPU occupant.

Q4 is additional. The same command with `--vehicles parent,q4` will write an
explicit `REFUSED` receipt for the Q4 stream unless `HAWKING_Q4_CAPTURE_BIN`
points at a native Hawking reader. The Python harness will not dequant Q4
into mlx. That is the binding that kept the parent legal.

`--plan` (this lane, already run) never loads a model and never touches the GPU.

---

## 5. Expected cost (ESTIMATED from MEASURED 14.967979 s / 256 tokens)

### 5.1 Wall

| quantity | value | label |
|---|---:|---|
| v1 wall | 14.967979082999591 s | MEASURED |
| s/token | 0.05846866829296715 | MEASURED / 256 |
| linear scale 23216/256 | **1357.4086030895255 s ≈ 22.62 min** | ESTIMATED |
| + f16 write at 1.5 GB/s | **1402.2026652548589 s ≈ 23.37 min** | ESTIMATED lower |
| GQA-adjusted (16/64 layers quadratic) | **3207.32608458925 s ≈ 53.46 min** | ESTIMATED |
| upper bound | **10800 s = 3 h** | ESTIMATED, long GQA |
| parent + Q4 sequential, linear | 2765.530829389718 s | ESTIMATED |

The 23-minute figure is the short-prompt lower bound in doctor recovery §5.6.
Ten percent of the mass is 512–2048 tokens. GQA is quadratic. Treat 53 minutes
as the working estimate and 3 hours as the bound. Two vehicles, sequential,
roughly doubles the linear number.

### 5.2 Peak memory

| quantity | bytes | GB (1e9) | label |
|---|---:|---:|---|
| BF16 weights | 53,791,996,928 | 53.791997 | MEASURED geometry × 2 |
| one-layer microbatch (2048 × 17408 × f16) | 71,303,168 | 0.071 | ESTIMATED |
| peak capture process | 56,265,953,280 | **56.266** | ESTIMATED |
| resident payload | 14,297,675,776 | 14.298 | MEASURED harvest |
| peak if resident stays | 70,563,629,056 | **70.564** | ESTIMATED |
| unified | 96,000,000,000 | 96 | box |

Streaming never keeps a full site. Peak extra activation RAM is one layer ×
one sequence of the widest site, plus the GQA score tensor on a 2048-token
prompt. BF16 weights dominate.

### 5.3 Disk

| stream | bytes | GB (1e9) |
|---|---:|---:|
| parent, site-split N, f16 | **67,191,093,248** | **67.191** |
| parent, same, f32 (not written) | 134,382,186,496 | 134.382 |
| Q4 twin, census N=2048, f16 | 8,879,341,568 | 8.879 |
| both | 76,070,434,816 | 76.070 |

67.191 GB is the site-split-N figure doctor recovery estimated at ~67.19 GB.
`--store-mode full` (every site at 23216) is 100.668 GB and is not the default.

---

## 6. What the GPU lane must check before starting

`--plan` already printed a live preflight. Re-run it at fire time. This lane
observed, and did not change:

| check | observed this lane | must at fire |
|---|---|---|
| parent is `.../qwen38-27b/bf16` | exists, looks bf16 | refuse any other source |
| Q4 is `.../uniform-q4-v1` | exists | native only; no expand |
| candidate under test | none on those paths | refuse mixed-2p0 / mixed-sub15 |
| GPU lock | **held by `qwen38-bracket-a`** | do not start until this lane holds `gpu_lane_lock.sh` |
| resident | **alive**, socket exists, 5 `pgrep` lines | do not stop / restart / talk |
| `hw.memsize` | 103,079,215,104 (96 GiB) | 96 GB unified |
| disk free | 328,433,356,800 | need ≥ 67.2 GB + slack; plan used 86 GB with Q4 |
| peak if resident stays | 70.56 GB of 96 GB | GPU lane decides whether to pause the organism. This tool will not. |

`--run` refuses to start if the GPU lock is held and `QWEN38_CAPTURE_I_HOLD_GPU`
is unset, if the parent path is not bf16, if the path looks like a candidate
under test, or if disk is short. It never sends a signal to genesis-resident.

If 96 GB cannot hold resident 14.3 GB + BF16 53.8 GB + a 56 GB capture
working set, the GPU lane pauses the resident *itself* and records that it
did. Do not let the capture evict the organism by surprise.

---

## 7. Command output (this lane)

```
$ python3 -m pytest tools/tests/test_qwen38_capture_v2.py -q
................                                                         [100%]
16 passed in 0.19s
```

```
$ python3 tools/qwen38_capture_v2.py --plan
```

Full stdout is in §8. `gpu: NOT TOUCHED`, `model_load: NO`, `forward: NO`.

```
$ test -s workspace/superwave/g1/g1-capture-harness.md; echo $?
0
```

```
$ git status --porcelain
?? tools/qwen38_capture_v2.py
?? tools/tests/
?? workspace/superwave/g1/g1-capture-harness.md
```

(`g1-capture-harness.md` is this file; porcelain after it exists includes it.)

---

## 8. Plan-mode stdout (MEASURED, this process)

```
QWEN38 CAPTURE V2 PLAN
======================
gpu: NOT TOUCHED
model_load: NO
forward: NO

n_tokens: 23216
n_sequences: 69
n_prompts: 69
hold_frac: 0.25
n_fit: 17412
n_hold: 5804
n_fit >= 17408: YES
holdout: prompt-level (not row shuffle)

sites:
  post_input_norm  width=5120  layers=64 store_n=6827  fit_dim=5120  n_fit=5120  bytes_f16=4474142720 gb_f16=4.474143 adequacy=ACCEPTED
    consumed_by: q/k/v, in_proj_qkv/z/a/b (attention in-dim)
  post_attn_norm   width=5120  layers=64 store_n=6827  fit_dim=5120  n_fit=5120  bytes_f16=4474142720 gb_f16=4.474143 adequacy=ACCEPTED
    consumed_by: gate_proj, up_proj (MLP in-dim after mixer residual)
  post_swiglu      width=17408 layers=64 store_n=23216 fit_dim=17408 n_fit=17412 bytes_f16=51730448384 gb_f16=51.730448 adequacy=ACCEPTED
    consumed_by: down_proj. silu(x@Wg.T)*(x@Wu.T), stored, not reconstructed
  mixer_x          width=6144  layers=64 store_n=8192  fit_dim=6144  n_fit=6144  bytes_f16=6442450944 gb_f16=6.442451 adequacy=ACCEPTED
    consumed_by: out_proj / o_proj. True recurrent mix or GQA gated softmax mix
  final_norm       width=5120  layers=1  store_n=6827  fit_dim=5120  n_fit=5120  bytes_f16=69908480 gb_f16=0.069908 adequacy=ACCEPTED
    consumed_by: lm_head. Confirmed model.norm, not L63 post-norm hidden
store_mode: site-split-n
parent_bytes_f16: 67191093248
parent_gb_f16: 67.191093
parent_bytes_f32: 134382186496
parent_gb_f32: 134.382186
q4_store_mode: census
q4_bytes_f16: 8879341568
q4_gb_f16: 8.879342
total_bytes_f16_parent_only: 67191093248
total_gb_f16_parent_only: 67.191093
total_bytes_f16_both: 76070434816
total_gb_f16_both: 76.070435

vehicles:
  parent: PARENT_BF16_REAL  path=/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16
          calibration source; must be preserved; refuse mixed-2p0 / mixed-sub15
  q4:     COHERENT_Q4_VEHICLE path=/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1
          additional, not a replacement; native reader only; expand-to-float REFUSED

time (ESTIMATED from MEASURED 14.967979 s / 256 tokens):
  v1_wall_s: 14.967979082999591
  v1_n_tokens: 256
  s_per_token_measured: 0.05846866829296715
  wall_s_linear: 1357.4086030895255
  wall_s_gqa_adjusted: 3207.32608458925
  wall_s_lower: 1402.2026652548589
  wall_s_upper: 10800.0
  wall_s_both_vehicles_linear: 2765.530829389718
  bound: ~23 min (short prompts) to a few hours (long GQA)

memory (ESTIMATED):
  bf16_weight_bytes: 53791996928
  bf16_weight_gb: 53.791997
  resident_weight_bytes: 14297675776
  resident_weight_gb: 14.297676
  one_layer_microbatch_bytes: 71303168
  peak_process_bytes: 56265953280
  peak_process_gb: 56.265953
  peak_if_resident_stays_bytes: 70563629056
  peak_if_resident_stays_gb: 70.563629
  unified_memory_bytes: 96000000000

preflight (GPU lane must check before --run; this lane does not mutate):
  mutate: False
  parent_dir_exists: True
  parent_looks_bf16: True
  q4_dir_exists: True
  candidate_under_test: None
  gpu_lock_exists: True
  gpu_lock_owner: qwen38-bracket-a
  hw_memsize: 103079215104
  disk_free_bytes: 328433356800
  need_bytes: 86070434816
  disk_enough: True
  resident: {'checked': True, 'alive': True, 'action': 'DO_NOT_STOP_RESTART_OR_TALK', 'socket_exists': True, 'pgrep_rc': 0, 'pgrep_n_lines': 5}
  unified_gb: 96
  must_not_stop_resident: True
  must_not_expand_q4_to_float: True
  advice: Pause the resident only from the GPU lane if 96 GB cannot hold resident (14.297676 GB) + BF16 (53.791997 GB) + activations. This tool will not pause it.

adequacy law: n_fit >= fit_dim or the number is REFUSED
  NS-014 wreck: 92/2048 = 0.044921875
  v1 256/6144 = 0.041666666666666664
  v1 256/17408 = 0.014705882352941176
  mixer_x is required. A v*silu(z) proxy is DEGENERATE and is not mixer_x.

out_dir: /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v2
```

---

## 9. What this lane did not do

- Did not run the capture, any GPU work, any forward, or any model load.
- Did not stop, restart, or talk to the live Genesis process.
- Did not modify `activation-capture-v1` or its receipt.
- Did not merge `7c5f323d9`.
- Did not implement a native Q4 forward. The Q4 stream is specified and
  refused-unless-native. Expand-to-float is not a fallback.

---

## Completion report

```
STATUS
IMPLEMENT_READY

CLAIMS
1. tools/qwen38_capture_v2.py is runnable. --plan prints n_tokens=23216, the
   five sites, per-site bytes, 67.191093 GB parent f16, 56.266 GB peak process,
   1357.41 s linear wall, without touching the GPU. MEASURED. Evidence: §8.
2. adequacy_gate is a callable and is wired into --check-adequacy, --run
   receipts, and gated_score. A refused verdict has score=null and
   emit_score=false. MEASURED. Evidence: test_gated_score_*; §2.1.
3. The existing 256-token capture against a 6144-dim fit returns REFUSED
   (n_fit=185, rpd=0.030111, worse than NS-014 0.044922). The test passes.
   MEASURED. Evidence: pytest; --check-adequacy exit 1.
4. A synthetic N=23216 / n_fit=17412 / fit_dim=17408 case is ACCEPTED. The
   test passes. MEASURED. Evidence: test_synthetic_adequate_case_is_accepted.
5. GPU-lane command, wall, memory, disk, and preflight are in §4–§6. The
   resident is alive and the GPU lock is held by another lane; this tool
   will not start under that lock without QWEN38_CAPTURE_I_HOLD_GPU=1 and
   will not pause the organism. MEASURED preflight; ESTIMATED costs.

EVIDENCE
- python3 -m pytest tools/tests/test_qwen38_capture_v2.py -q → 16 passed
- python3 tools/qwen38_capture_v2.py --plan → §8
- python3 tools/qwen38_capture_v2.py --check-adequacy --fit-dim 6144 → REFUSED
- workspace/campaign/.../activation-capture-v1/capture-result.json (unmodified)
- crates/hawking-core/src/model/qwen38_geometry.rs
- workspace/superwave/g1/g1-doctor-recovery.md §5–§6
- NS-014 92/2048 = 0.044921875

CHANGES
tools/qwen38_capture_v2.py
tools/tests/test_qwen38_capture_v2.py
workspace/superwave/g1/g1-capture-harness.md

TESTS
see §7

RISKS
- 56 GB capture + 14 GB resident + 96 GB unified is tight. GPU lane owns the
  pause decision. Surprise eviction of the organism is the failure mode.
- True mixer_x depends on hooking mlx out_proj/o_proj input. If a future
  mlx_lm fuse removes that Linear, --run REFUSES rather than storing
  v*silu(z). Do not weaken that refuse.
- Q4 twin is specified, not implemented in Python. A GPU lane that wants
  vehicle-X ranks needs a native binary.
- GQA 2048-token sequences make the 23 min linear estimate a lower bound.

UNRESOLVED
- The capture itself. This lane builds and proves. GPU lane fires.
- Whether parent-X and Q4-X rank the same organs.
- Whether mixer_x hot columns stay disjoint from |W| once the true mix
  is stored (forensics REOPEN_IF).

NEXT
GPU lane: hold gpu_lane_lock.sh, re-run --plan, decide the resident, then
the command in §4. After capture, refuse any fit that the wired gate
does not ACCEPT. Do not pack from the 256-token cube.
```
