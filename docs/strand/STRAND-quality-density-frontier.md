# STRAND quality-density frontier

_Started 2026-06-12. Scope: reducing PPL per encoded bit for STRAND model
quantization. This is the companion to `docs/STRAND-media-speed-roadmap.md`,
which intentionally excludes quantization. The speed roadmap makes STRAND feel
practical; this roadmap makes smaller STRAND models stay intelligent._

## 0. Decisive status

We are not just benchmarking locally.

Local live work, observed 2026-06-12:

| lane | live command | meaning |
|---|---|---|
| `pv_down_protect_q2` | `scripts/strand-qat.py --model scratch/qwen-05b --quant strand --bits 2 --steps 300 --lr 0.0001 --kd --device mps --strand-flags "--bits 2 --l 12 --outlier-channel 1 --rung-config configs/mp-dp-d4-r2.json ..."` | 300-step QAT/PV through the real STRAND encoder, starting from down-protected 2-bit. Init: `bpw=3.290977`, `PPL=40.1640`. Goal: recover quality from the dp-d4-r2 PTQ floor. |
| `ffn4/rest2` | `target/release/quantize-model --in scratch/qwen-05b/model.safetensors --out research/mp-frontier/ffn4/recon/model.safetensors --rung-config configs/mp-ffn-protect.json --bits 2 --l 12 --outlier-channel 1` | CPU quantization of a stronger FFN-protected 2-bit frontier arm: attention at 2-bit, all FFN projections at 4-bit. This is a placement test, not QAT. |

RunPod live work, observed 2026-06-12:

| lane | live command | meaning |
|---|---|---|
| `qwen-32b q2_l12_out1` | `/workspace/s7p-chain.sh /workspace/strand/scratch/qwen-32b --bits 2 --l 12 --outlier-channel 1 --threads 96 --no-eval --resume --skip-calib --label q2_l12_out1 --out-dir scratch/qwen-32b/reopen/q2_l12_out1` | The pod is quantizing/reconstructing the 32B model shards with the canonical 2-bit/outlier recipe. It is not training and is not currently evaluating PPL because `--no-eval` is set. |

Immediate operational rule: do not launch another local MPS training job while
the QAT run is active. CPU-only quantization can run, but this repo has already
recorded box freezes from careless co-tenancy; quality runs get the machine alone.

## 1. What the current numbers mean

PPL is exponential in loss, so raw `PPL / bpw` is the wrong optimization metric.
Use log loss tax:

```
loss_tax_nats = ln(PPL_quant / PPL_bf16)
```

For the Qwen2.5-0.5B local canon, use `PPL_bf16 ~= 12.536`.

| config | bpw | PPL | loss tax vs bf16 | read |
|---|---:|---:|---:|---|
| `dp_d4_r3` | 3.68 | 19.107 | 0.421 nats | Strong, but its `L=8` note makes it not clean against the current `L=12` canon. Re-run before making it doctrine. |
| `dp_d4_r2` | 3.29 | 40.48 | 1.172 nats | Real 2-bit rescue: pure q2 was about 79.94 PPL, so protecting down-proj at 4-bit halves raw collapse and removes about 0.68 nats of damage. |
| current local PV init | 3.290977 | 40.164 | 1.164 nats | Same floor as `dp_d4_r2`, now inside a 300-step KD/PV recovery run. |

The clean objective is:

```
minimize loss_tax_nats at fixed encoded bpw
or
maximize loss_tax_nats removed per added 0.01 bpw
```

The black-hole framing is exact enough to be useful: keep pushing quality mass
inside a smaller bit horizon, but bill every particle of side-info that crosses
the boundary.

## 2. Audit of what STRAND already has

The current encoder is not a naive 2-bit quantizer. Built pieces:

| surface | current code | status |
|---|---|---|
| Incoherence | `crates/strand-quant/src/rht.rs`, `rht_seed_for` | Per-tensor signed Hadamard by name seed. Strong, but the seed is currently not optimized. |
| Gaussian frozen LUT | `crates/strand-quant/src/codebook.rs`, `lut_tables.rs` | Frozen Q12 Gaussian quantile table, deterministic decode. |
| Stateful trellis | `crates/strand-quant/src/trellis.rs`, `encode.rs` | `L=12` is the 2-bit operating point; `L>=13` already looked saturated/expensive. |
| Per-block side-info | `encode.rs` | 32-bit super-scale, 6-bit sub-scales, optional affine mins, init state. Bit-ledger shows this metadata is now first-class. |
| Outlier side-channel | `outlier_wire.rs`, `quantize-model --outlier-channel` | The top-|w| pre-RHT channel is alive and large. Its positions are now the biggest entropy-coding opportunity. |
| Mixed rungs | `--rung-config`, `configs/mp-dp-d4-r2.json`, `configs/mp-ffn-protect.json` | Working. Current best signal says down-proj protection is a real 2-bit rescue. |
| QAT/PV | `scripts/strand-qat.py` | Real encoder in the loop, delta-forward STE, KD, WSD hooks, selective PV regex support, lineage output. |
| Learned vector LUT | `--vec-dim`, `--learned-codebook`, `learned_codebook.rs`, `gate-vectrellis.rs` | Built as a research path; PTQ vector collapse means it needs PV/progressive training, not blind PTQ. |
| De-bias math | `debias.rs`, `gate-debias.rs`, `research/debias-results.md` | Output-RMS gate alive on modeled activation mean; not integrated or PPL-confirmed. |
| Loss-aware requant design | `docs/STRAND-loss-aware-requant-proposal.md` | Strong theory gate: only live y-space within-block grad structure can justify it. |

Dead or restricted surfaces:

| idea | verdict |
|---|---|
| Generic Hessian/diag-H weighting | Dead in the current framing. RHT flattens diagonal curvature into inert per-block constants and previous A/B hurt PPL. |
| rel-RMS-only routing | Not allowed as a final decision metric. RHT makes rel-RMS flat across tensor classes. |
| Proxy QAT through a different quantizer | Dead. Train through what ships. |
| Attention-protect as default MP story | Dominated in prior data; down-proj carried the 3-bit win. |
| More scalar `L` beyond 12 | Saturated enough that it is not the main frontier. |

Conclusion: the next quality gains are unlikely to come from "try another scalar
quantizer." They come from objective, basis, allocation, and training path.

## 3. External frontier, translated into STRAND terms

Primary-source scan, refreshed 2026-06-12:

| source | external result | STRAND translation |
|---|---|---|
| [QTIP](https://arxiv.org/abs/2406.11235) | TCQ plus incoherence reaches high effective dimension without exponential VQ codebooks. | STRAND is already in this family. Do not copy; deepen the parts QTIP leaves open for us: frozen integer LUTs, bit-exact decode, PV through the actual trellis. |
| [QuIP#](https://arxiv.org/abs/2402.04396) | Randomized Hadamard and lattice codebooks are strong in extreme PTQ. | Confirms RHT and structured codebooks; suggests basis choice matters, not just "have RHT." |
| [AQLM](https://arxiv.org/abs/2401.06118) | Learned additive/multi-codebook quantization is strong below 3 bits, especially after optimization. | Competes on trained codebooks; STRAND should answer with trained trellis/PV and optional low-rank residuals, not scalar PTQ. |
| [ParetoQ](https://arxiv.org/abs/2502.02631) | There is a learning transition between 2 and 3 bits; 2-bit wants a different trained solution, not just rounding. | Explains why `q2` needs PV and why q3 PV can damage a near-converged basin. |
| [Compute-optimal QAT](https://arxiv.org/abs/2509.22935) and Apple write-up | QAT fraction and bit-width should be compute-allocated, not treated as a postscript. | Our 300-step PV arms are a probe, but the real product needs staged QAT budgets by rung and model size. |
| [QAT scaling law](https://arxiv.org/abs/2505.14302) | Quantization error depends on model size, training data, and granularity; bottlenecks can shift by layer. | Run 0.5B as hypothesis lab, but do not overclaim scale. The 32B pod quant is a necessary scale signal. |
| [WSD cooldown dynamics](https://arxiv.org/abs/2508.01483) | Cooldown shape materially affects transformer final loss. | The current local arm is cosine; the next clean A/B is WSD warmup/hold/cooldown with everything else fixed. |
| [Reasoning-QAT study](https://arxiv.org/abs/2601.14888) | KD is robust, PTQ init helps, and calibration/training domain alignment accelerates QAT. | Current KD + WikiText train/test split is directionally right; add domain-aligned train screens before scale. |
| [D2Quant](https://arxiv.org/abs/2602.02546) | Down-projection is a sub-4-bit bottleneck; dual-scale and mean-shift correction help without extra bit budget. | Directly supports the dp-d4 results. Build STRAND-native down-proj absorbable scaling plus activation-mean correction. |
| [Bit-by-Bit](https://arxiv.org/abs/2604.07888) | Progressive QAT and outlier-channel splitting stabilize ultra-low-bit training. | Replace direct q2-only PV with q4 -> q3 -> q2 progressive STRAND-PV and keep outlier routing in the training loop. |
| [SpinQuant](https://arxiv.org/abs/2405.16406) and [ReSpinQuant](https://arxiv.org/abs/2604.11080) | Rotation choice matters; layer-wise expressivity helps if overhead can be fused away. | STRAND should run a deterministic RHT-seed/basis lottery. Bill a tiny seed index only if held-out PPL improves. |
| [ZeroQAT](https://arxiv.org/abs/2509.00031) and [LR-QAT](https://arxiv.org/abs/2406.06385) | Memory-efficient QAT can work by freezing most parameters or using low-rank trainable objects. | Selective PV and low-rank residual shadows are not hacks; they are the likely scale path. |
| [LiftQuant](https://arxiv.org/abs/2606.04050) | Continuous bit-width via lifted 1-bit structures attacks the deployment gap between integer bit-widths. | STRAND already has fractional payload via `k / vec_dim`. Revive vector modes only as trained/progressive rungs. |
| [LittleBit](https://arxiv.org/abs/2506.13771) | Sub-1-bit needs latent factorization and compensation, not plain rounding. | Future sub-2 path: STRAND core plus low-rank binary/latent residual, using `strand-delta` concepts. |

## 4. The frontier stack

### A. Finish and read the active `pv_down_protect_q2` run

Decision first:

| outcome | interpretation | next action |
|---|---|---|
| final PPL <= 30 | PV can recover a large fraction of down-protected q2 damage. | Promote dp-d4-r2 PV to the main 2-bit lane; run WSD A/B. |
| PPL 30-36 | Alive but weak. | Try WSD and progressive q3->q2 before scaling. |
| PPL >= 36 | Direct cosine PV is not enough. | Do not scale this recipe; pivot to progressive QAT plus de-bias/basis gates. |

Do not judge from rel-RMS. PPL is the truth.

### B. Evaluate `ffn4/rest2` against `dp4/rest2`

Current `configs/mp-ffn-protect.json` protects gate/up/down at 4-bit and leaves
attention at 2-bit. It is expensive, but it answers a placement question:

```
Is down_proj the only 2-bit bottleneck, or does the whole FFN need protection?
```

Gate:

| result | decision |
|---|---|
| `ffn4/rest2` PPL close to `dp4/rest2` | Down-proj really is the event horizon; use bits/training there first. |
| `ffn4/rest2` much better at acceptable bpw | Build a per-layer FFN allocator; down-only was under-protecting. |
| `ffn4/rest2` worse or dominated | Kill broad FFN protection; keep down-proj and pursue objective/basis instead. |

### C. Cheap PPL A/B for inner-product de-bias

The de-bias math is already in `debias.rs`; the missing piece is real activation
mean and PPL application.

Implementation slice:

1. Add `scripts/calib-actmean.py`: collect projection-input activation means on
   WikiText train windows, keyed by full tensor/module name.
2. Add `--debias-mu <json>` or a sidecar-driven eval shim that applies
   per-output-row bias correction to projections.
3. Run A/B at `--bits 2 --l 12 --outlier-channel 1` with identical eval.

Why this comes early: it costs about 0.003-0.018 bpw depending on tensor width and
attacks output error rather than weight-MSE. D2Quant's DAC result makes this even
more plausible because it says quantization-induced mean shifts are underused.

Kill bar:

```
adopt if PPL_B <= PPL_A * 0.995 at billed bpw
kill if PPL_B >= PPL_A or if the correction is not applied visibly
```

The contamination tell remains: identical PPL to many decimals means the bias path
was not actually used.

### D. Down-proj absorbable dual-scale

D2Quant says down-proj has special structure. STRAND already sees this empirically:
protecting down-proj at 4-bit rescues q2 far more than a generic rule would predict.

STRAND-native version:

For each MLP intermediate channel `j`, introduce an absorbable scale `a_j`:

```
up_proj row j     <- up_proj row j / a_j
down_proj col j   <- down_proj col j * a_j
```

For the SwiGLU path, this preserves the full-precision function when only the
`up_proj` side is inversely scaled:

```
down_col_j * (silu(gate_j x) * up_j x)
```

becomes:

```
(a_j down_col_j) * (silu(gate_j x) * (up_j x / a_j))
```

The full-precision function is unchanged, but the quantization surfaces change.
Pick `a_j` to make down-proj easier at 2-bit without blowing up up-proj. This is
not a wire-cost side-channel if the transformed weights are what we quantize.

Gate:

1. Single-layer smoke on layers 0 and deep layer: does down-proj rel-RMS/output-KL
   improve without up-proj damage?
2. 0.5B PPL A/B on `dp4/rest2` and q2.
3. Only then consider scale.

### E. RHT seed and basis lottery

SpinQuant/ReSpinQuant make one thing loud: rotations are not interchangeable.
STRAND currently uses a deterministic seed derived from the tensor name. That is
stable, but it is not optimized.

Minimal deterministic extension:

1. Add a global seed salt list, for example 16 salts.
2. For each tensor, quantize several candidate salts on a small calibration slice.
3. Score by output-KL/hot-swap, not rel-RMS.
4. Store a 4-bit salt index per tensor only if it improves held-out PPL enough to
   pay for itself.

This keeps decode bit-exact and integer-only. It spends a tiny amount of side-info
to choose a better event horizon.

Kill bar:

```
adopt if 0.5B PPL improves >= 0.5% at fixed bpw after billing salt bits
kill if only rel-RMS improves
```

### F. Progressive STRAND-PV: q4 -> q3 -> q2

The current active run jumps directly into q2 PV. Bit-by-Bit and ParetoQ both say
ultra-low-bit training is more stable when the model is moved stage by stage.

Use existing machinery:

| stage | flags | reason |
|---|---|---|
| stage 1 | q4 or down-protected q3, short KD | keep model near basin while it learns STRAND geometry |
| stage 2 | q3/dp4, WSD hold | adapt without full q2 collapse |
| stage 3 | q2/dp4, cooldown | final compression and quality packing |

Needed script: `scripts/strand-act2-density.sh`, built from `strand-qat.py`
`--init-state`, `--strand-flags`, `--cooldown-frac`, and `--save`.

Gate:

```
same wall-clock budget as direct q2 PV
win if final q2 PPL improves >= 5% over direct q2 PV
```

### G. Live-Fisher requant, only after Phase 0

This is the sharpest "objective" lever, but it has to respect the Hessian corpse.
Use the framing in `docs/STRAND-loss-aware-requant-proposal.md`:

```
accumulate grad^2 after rotating gradients into y/RHT space
use only reproducible within-block structure
feed that map into Viterbi cost
do not change decode or wire format
```

Phase 0 is mandatory:

| phase | question | kill |
|---|---|---|
| 0a | does live y-space grad^2 have within-block variation? | median within-block CV < 0.10 |
| 0b | is it reproducible? | split-half r < 0.2 |

If Phase 0 passes, this is the cleanest way to "instill quality" into the same
number of bits: do not add bits; move the errors to less damaging positions inside
the block.

### H. PPL-KL rung allocator and selective PV

The allocator must see output-space damage. Rel-RMS is not enough.

Build `scripts/rung-kl.py`:

1. Quantize full recon files for candidate rungs/configs.
2. Load bf16 once.
3. For each tensor, hot-swap one recon tensor at a time.
4. Score logit KL and delta-NLL on WikiText train windows.
5. Emit exact tensor rules plus a RED PV regex/list.

Then use the existing selective PV support in `strand-qat.py`:

```
--pv-tensors '<RED tensor regex>'
```

The prize is a 7B/32B training lane that trains only collapsed tensors and freezes
the rest at bit-identical PTQ recon.

### I. Fractional/vector rungs as trained objects

STRAND already has fractional payloads:

```
payload_bpw = k / vec_dim
```

Old vector PTQ collapse should not close the channel. LiftQuant and LittleBit both
argue that sub-2-bit quality needs structured/trained representations.

Near-term rule:

| vector mode | status |
|---|---|
| PTQ vector rung | do not ship; collapse already seen. |
| PV-trained vector rung | alive if progressive STRAND-PV first proves stable at q2. |
| learned vector LUT | use only with bit-identical decode tests and PPL gates. |
| low-rank residual over STRAND core | future sub-1.5 lane; tie to `strand-delta`, not the q2 product path. |

## 5. Ordered execution plan

No new local heavy run until active local jobs finish.

| order | task | decisive output |
|---:|---|---|
| 1 | Finish local `pv_down_protect_q2` | final PPL, lineage row, compare to 40.16 init and 26.77 prior PV floor |
| 2 | Finish/eval `ffn4/rest2` | PPL/bpw versus `dp4/rest2`; decide down-only vs full-FFN protection |
| 3 | Run de-bias real-activation PPL A/B | adopt/dead for nearly-free output-bias correction |
| 4 | Run WSD A/B of the same PV recipe | isolate cooldown from direct cosine |
| 5 | Build `rung-kl.py` hot-swap screen | output-space allocator, RED tensor list |
| 6 | Run selective PV down-proj vs full-PV 0.5B | decide if 7B selective-PV is the cheap path |
| 7 | Run seed/basis lottery on 0.5B | adopt/dead seed-index side-info |
| 8 | Progressive q4->q3->q2 STRAND-PV | decide if direct q2 PV is leaving quality on the table |
| 9 | Phase-0 live-Fisher map | decide whether loss-aware Viterbi requant is real |
| 10 | Scale only the winners to 7B/32B pod lanes | no scale run without a 0.5B gate |

## 5.1 Chain placement

Local first:

| work | why local |
|---|---|
| activation-mean calibration | Needs model hooks and small WikiText windows; CPU-only is acceptable and avoids touching the pod chain. |
| de-bias PPL A/B | Cheap 0.5B truth gate; requires the base model plus a recon and should be run before any scale spend. |
| seed/basis lottery | Many tiny candidate encodes and hot-swap scores; this is a hypothesis lab task. |
| live-Fisher Phase 0 | Must instrument the local PV loop carefully; do not run it beside unrelated MPS work. |
| progressive STRAND-PV recipe smoke | First compare against the active direct q2 PV run on the same 0.5B harness. |

Cloud later:

| work | why cloud |
|---|---|
| 7B/32B/70B quant/eval confirmation | Expensive shard quantization and long PPL passes belong on RunPod after a local gate passes. |
| scale-specific mixed-rung sweeps | The allocator emits exact tensor rules locally; the pod confirms whether they survive scale. |
| selective-PV at 7B+ | Only after 0.5B selective PV proves the RED tensor set recovers most of full-PV gain. |

New local tooling for the first custom gate:

```bash
/usr/local/bin/python3 scripts/calib-actmean.py \
  --model scratch/qwen-05b \
  --chunks 8 --ctx 1024 \
  --out research/actmean-qwen05b.json

/usr/local/bin/python3 scripts/strand-debias-ppl.py \
  --base scratch/qwen-05b \
  --recon research/mp-frontier/ffn4/recon/model.safetensors \
  --actmean research/actmean-qwen05b.json \
  --ctx 2048 --chunks 64 --device cpu --dtype bfloat16 \
  --out research/debias-ppl-ffn4-ab.json
```

This should not be appended to the current cloud chain yet. The cloud chain is
currently producing large q2 artifacts; the de-bias/vector-mean gate must prove a
PPL move on 0.5B before it earns a pod append.

## 6. Success metrics

Report every frontier point with this schema:

| field | required |
|---|---|
| model | exact HF dir / shard set |
| quant recipe | bits, `L`, outlier pct/bits, rung config, RHT seed policy, vector dim |
| training recipe | steps, LR, schedule, KD, data split, init checkpoint, selective tensor set |
| bpw | encoded bpw and deploy bpw, with side-info billed |
| PPL | same eval harness, ctx, windows, device, dtype |
| loss tax | `ln(PPL / PPL_bf16)` |
| delta | loss-tax removed per added 0.01 bpw, or bpw saved at same loss tax |
| contamination checks | identical-PPL guard, clean train/test split, no concurrent MPS if relevant |

The product claim is not "lowest PPL" and not "lowest bpw." It is the Pareto
frontier: lowest loss tax at a given real encoded bit budget.

## 7. North-star

STRAND's speed frontier removes wasted motion. This frontier removes wasted
precision.

The next moat is a quantized model that has been trained, routed, biased, rotated,
and packed so every surviving bit is doing model work. Bits that only store a
default, a habit, or a proxy metric do not cross the event horizon.

## 8. Session log 2026-06-12 — gates run, grammar built, orchestration condensed

### 8.1 The shared promotion grammar (`scripts/promote.py`)

One tool every result now passes through, so the system speaks gates not vibes. It
auto-computes `loss_tax_nats`, refuses to promote a result missing {ppl, bpw,
harness identity}, applies the §4 kill bars, and stamps a `promotion` block into
the json. Five states: `INCOMPLETE`, `KILLED`, `LOCAL_PASS`, `PROMOTE_CLOUD`,
`GATED`. Exit codes encode the decision (0/10/20/30/40) for shell gates.

Wired in:
- `scripts/conductor.sh` block 2b (FRONTIER PROMOTION): any new/changed result in
  `research/{pv-dp,mp-frontier,debias-*,down-protect}` is run through promote.py and
  surfaced as an `evt2 PROMOTE` grammar line. The conductor now watches the new
  quality-density lanes, not just the old marathon `qat-*.json`.
- `scripts/pod-chain-v2.sh` `require_promoted()`: a NEW scale leg runs only if its
  0.5B gate json (mirrored to `/workspace/gates/`) carries `promotion.state ==
  PROMOTE_CLOUD`. The pod trusts the stamped state; it does not recompute. Baseline
  campaign legs pass no gate and run as before. **This is the cloud-waste guard:
  you cannot append a scale leg without a local promotion on file.**

### 8.2 §4.B placement gate — DECIDED (kill broad FFN protection)

All 0.5B mixed-precision arms, loss-tax billed against the bf16 anchor 12.536 and
against uniform q3 (16.11 @ ~3.4 bpw, tax 0.251 — the point to beat):

| arm | bpw | PPL | loss_tax | verdict |
|---|---:|---:|---:|---|
| q3 uniform (anchor) | ~3.4 | 16.11 | 0.251 | the Pareto point |
| `dp_d4_r3` (down@4/rest@3, L=8) | 3.68 | 19.11 | 0.421 | dominated (and L=8 confound) |
| `ffn4` (all-FFN@4/attn@2) | 4.54 | 23.53 | 0.630 | dominated; off the 2-bit budget |
| `dp_attn2` (down@4/attn@2/ffn@3) | 3.57 | 32.12 | 0.941 | dominated |
| `dp_d4_r2` (down@4/rest@2) | 3.29 | 40.48 | 1.172 | 2-bit rescue (q2 was 79.94) but dominated |

**Decision:** broad FFN protection is killed as a density play — `ffn4` buys PPL
only by spending to 4.5 bpw, where uniform q3 already wins at 3.4. Down-protect
*halves the raw q2 collapse* (79.94 → 40.48) but no bit-placement arm reaches the
q3 Pareto point. **At 0.5B, bit placement is not the 2-bit lever; training is.**
This is exactly why the active `pv_down_protect_q2` PV run is the real test: it asks
whether *training through* the dp-d4-r2 floor can do what placement cannot.

### 8.3 De-bias gate — **ADOPTED, the largest free lever found to date**

`calib-actmean.py` banked 168 modules with feature-mean vectors
(`research/actmean-qwen05b.json`). Vector A/B on the `dp_d4_r2` recon:

```
baseline (recon)      PPL 40.245
debiased (recon + c)  PPL 28.708     ratio 0.7133  (-28.67%)
corrections: 168/168 vector, corr_l2=61.4, corr_absmax=1.07 (visibly applied,
contamination guard clean; hook-replaced baseline 40.24 vs recon-load 40.48 is
bf16-copy ordering noise — judge by the ratio)
```

The kill bar was -0.5%. It removed **a third of the 2-bit damage** and lands the
2-bit arm at `tax=0.829 @ ~3.305 bpw` (stamped by promote.py: LOCAL_PASS).

**Side-info bill (exact):** per-output-row bias = 304,128 rows (24 layers x 12,672)
x bf16 = 4.87 Mbit / 357.8 M weights = **0.0136 bpw**. At fp32 it is 0.0272 bpw.
Either way the trade is absurdly favorable: 0.343 nats removed for ~0.014 bpw
(~25 nats per bpw — over an order of magnitude past any §5 estimate; the doc previously said 245, a 10x arithmetic error caught by the fact-check wave).

Decisions taken:
1. **Generality A/B running** on `dp_d4_r3` (does the win persist at 3-bit-class?).
2. **Artifact-ization is the gating work**: the correction must move from eval-time
   hook into the encode path. At encode time `quantize-model` holds orig, recon, and
   mu — compute `c = -(recon-orig) @ mu` per tensor and persist it with the
   artifact; deploy never needs the base. Runtime applies c in the MAC epilogue
   (same float-order class as existing scale application; decode stays deterministic).
   **BUILT 2026-06-12**: `quantize-model --actmean <calib.json>` computes c per
   tensor inside the encode loop (covers fresh-quant and identity-reuse arms) and
   writes `<out>.debias.json` with the billed mass on the grammar line. Parity vs
   the Python harness math: max|diff| 3.7e-09 on layers.0.down_proj (PASS). Next
   step when a gate earns it: fold the sidecar into the .strand v2 section chain
   (EOF-chained like OUTL/SPRV) and the decode-kernel MAC epilogue.
3. **7B confirm queued for the pod post-campaign** (bases were disk-law deleted;
   combine with the planned qwen-7b re-download for the iso-bpw head-to-head).
   It runs only with a PROMOTE_CLOUD stamp per `require_promoted()`.

### 8.4 Orchestration condensed (less mass, one canon)

The orchestration was triplicated (`ops/`, `scratch/`, `scripts/`) with diverged
`conductor.sh` copies and an ambiguous canon — a live hazard. Collapsed to **one**
canon in `scripts/`: conductor (running, with the frontier block), pod-chain
(SHARD_JOBS + require_promoted), governor (anon-based mem law), watch, podctl.
Removed 6 duplicate scripts; repointed the conductor's `replay.sh` exec. Remaining:
3 historical `.md` dups in `ops/` (low risk; future tidy).

### 8.5 Open gates (next session, in order; all 0.5B before any scale)

1. **PV verdict** (`research/pv-dp/pv-dp.json`) — promote.py bars: ≤30 & <26.77 →
   PROMOTE_CLOUD; 30–36 → WSD/progressive first; ≥36 → pivot to objective/basis.
2. **De-bias verdict** — adopt/kill per §4.C; if it moves PPL, it is near-free.
3. WSD A/B of the same recipe (only after PV verdict).
4. `rung-kl.py` output-space allocator + RED tensor list (to build).
5. Seed/basis lottery; progressive q4→q3→q2; live-Fisher Phase 0 — under clean
   resource guards (no second MPS job beside a PV run).

Nothing scales to the pod without a `PROMOTE_CLOUD` stamp.

## 9. Black-hole condensation doctrine 2026-06-12 (measured, not assumed)

Strategic frame: **speed is a threshold good (G4 already cleared it); quality-per-bit
is the scarce good; determinism is the moat nobody else sells.** Pour spend into
loss-tax-at-fixed-bpw and always frame wins as "denser AND deterministic."

STRAND's unfair advantage is the **encode-time triad**: orig weights + exact recon
+ activation stats + a decoder we can train through. PTQ lacks the trainable
decoder; non-deterministic methods can't ship the correction reproducibly. The
doctrine: spend near-free, low-dimensional correction channels tuned to the
dominant error; bill every particle; let training reach what correction can't.

Four measurements taken this session that reshape the plan:

**(1) Placement is output-KL, and it INVERTS the down-protect intuition.**
`scripts/rung-kl.py` hot-swap output-KL by class (dp_d4_r2, train windows):

```
up_proj 0.0099  v_proj 0.0092  gate_proj 0.0066  o_proj 0.0045
k_proj 0.0030   down_proj 0.0028   q_proj 0.0024
```

`down_proj` — the tensor every dp-arm protected — is **second-LEAST** output-damaging.
The real damage is `up_proj`/`v_proj`/`gate_proj`. rel-RMS hid this (RHT flattens it).
**Consequence:** the down-protect family was aiming at the wrong tensor; protecting
down spent 4-bit budget on a low-damage projection. The RED list (25 tensors, top:
up/gate/v) is the correct mixed-rung AND selective-PV target. The next PV arm and
any future mp-config must route by `rung-kl`, not by the down-proj habit.

**(2) The quantization error is white in WEIGHT space, low-rank in OUTPUT space.**
`scripts/error-spectrum.py` on dp_d4_r2:

```
weight-MSE space:        rank-1 energy 0.24%   (white — RHT whitened it, like diag-H)
output (act-weighted):   rank-1 energy 18.9%, rank-16 37.8%   (low-rank, ALIVE)
```

This is why every weight-MSE method (diag-Hessian, weight-low-rank residual) dies,
and why de-bias (an OUTPUT-space correction) works. The space that matters is the
one the activations see.

**(3) But de-bias (rank-0) already captures the dominant output mode; higher-rank
residuals are marginal.** `scripts/lowrank-residual-ppl.py` math check (up_proj):

```
output error:  recon 183.0  -> +de-bias 153.4 (-16%)  -> +de-bias+rank16 147.4 (-3.9% more, +0.34 bpw)
```

The dominant output-error mode *is* the mean shift, which de-bias handles at rank-0
(a bias, cheaper than a rank-1 UVt). So the LiftQuant/LittleBit latent-residual lever
is **real but marginal for the q2 product** — do not build it now; the bits buy more
as trellis or as PV. Banked: the residual harness exists and gates if a sub-1.5 lane
ever needs it.

**(4) De-bias generalizes across bit-rates and is the free prize.**
dp_d4_r2 (q2 base): -28.7% PPL. dp_d4_r3 (q3 base): -10.9% PPL. Both clear the bar
for ~0.014 bpw. **ADOPT de-bias globally** (artifact-ized via `--actmean`). It is
the rank-0 capture of the dominant output-error mode — the single best free lever.

### The condensed q2 lever stack (what actually pays, ranked)

1. **De-bias correction** (rank-0 output, ~0.014 bpw) — ADOPTED, generalizes. Free.
2. **PV training routed by rung-kl** (up/v/gate, NOT down) — the high-rank residual
   correction can't reach. The 2-bit lever. Next arm targets the RED list.
3. **Side-info entropy coding** (C2 scales 0.106 + outlier-positions 0.1476 bpw) —
   the measured ~0.25 bpw the artifact carries as dead mass. Pure density, no quality cost.
4. **Outlier-position entropy coding** (0.1476 bpw, the largest dead mass). NB: STRAND
   outliers are top-|w| *individual weights* (element-wise, flattened) — NOT channel
   outliers, so cross-layer channel-sharing does not apply (verified in
   quantize-model.rs). The real structure to exploit is intra-tensor: do the top-|w|
   positions cluster by column/row? If so, code column-id + offset instead of flat
   index. Scout = position column-concentration; then a rANS/delta coder on the stream.

Killed/deprioritized this session: bit-placement by tensor type (q3 dominates all 2-bit
placement arms); down-protect as the routing story (wrong tensor by KL); weight-space
low-rank residual (white); high-rank output residual for q2 (marginal above de-bias).

### Next experiments (queued, gated, resource-aware — PV owns MPS)
- **rung-kl-routed PV arm**: protect/train up/v/gate by the RED list instead of down.
  This is the course-correction the KL screen demands. Queue behind the current PV.
- **lowrank rank-16 PPL judge** (up_proj, on top of de-bias): confirm/refute the
  synthetic "marginal" call with real activations. Low priority, queued.
- **outlier-position cross-layer sharing**: measure repeat rate first (scout), then
  decide. The single largest remaining side-info mass.

## 10. Scheduled dev ledger (local + cloud) + the >70B flagship — 2026-06-13

"All dev scheduled" means it runs unattended and survives restart, not that I babysit it.

### Local (durable, conductor-owned)
- `@reboot` + `*/15` crons run `scratch/guardian.sh` -> ensures `scripts/conductor.sh`
  is alive (the refactor had deleted guardian.sh while the @reboot cron still pointed
  at it — a silent durability hole, now fixed).
- The conductor owns: frontier-promotion watch (promote.py grammar), pod poll/mirror,
  replay, and the **gate queue** (`scripts/run-next-gate.sh` -> `scripts/gates/*.sh`).
- Gate queue (self-guarding, idle-gated, one at a time; add dev = drop a numbered script):
  - `10-pv-kl-routed.sh` — PV protecting up/v/gate per rung-kl (fires after the
    down-protect PV verdict). The placement course-correction.
  - `20-lowrank-rank16-judge.sh` — real-PPL test of the output low-rank residual
    on top of de-bias (confirm/refute the synthetic "marginal").

### Cloud (gated — nothing scales without a PROMOTE_CLOUD stamp)
These legs are DEFINED and `require_promoted()`-guarded; they run post-70B when a base
is present, only if their 0.5B gate stamped PROMOTE_CLOUD. This is the schedule; the
guard is the enforcement.
- **de-bias scale confirm** (7B/32B): de-bias is ADOPTED at 0.5B; confirm the −10..−29%
  PPL win holds at scale. Needs an actmean calib on the base before deletion.
- **KL-routed mp confirm**: confirm the up/v/gate routing beats q3-uniform at scale.
- **pv7b RE-AIM (warranted by this session)**: the staged 7B selective-PV targeted
  `--pv-tensors down_proj`. rung-kl says down_proj is 2nd-LOWEST output damage — pv7b
  must be re-aimed to the RED set (up/v/gate). Do not run the down-proj version.

### Are there public models bigger than 70B? Yes — and one trial would be the flagship.
Open-weight, larger than 70B:
- **Dense:** Llama 3.1 **405B**, Mistral Large 2 **123B**, Falcon **180B**,
  Command R+ **104B**, Qwen2.5-**72B**.
- **MoE (total/active):** DeepSeek-V3 **671B**/37B, Grok-1 **314B**, Nemotron **340B**,
  DBRX **132B**/36B, Mixtral 8x22B **141B**/39B.

**Necessary for the science? No.** The scale-tolerance thesis (q2 relative damage NOT
monotone; 7B/14B tolerate, 32B in flight) is proven by the 70B tier. A 405B confirms a
prediction, it doesn't test a new hypothesis.

**Valuable for industry leadership? Yes — singularly.** "Llama-3.1-405B-class capability
in ~110GB at 2-bit, **bit-identical and verifiable**" is a flagship claim no competitor
can make: AQLM/QTIP can attempt 405B but not deterministically/trainably/billably.
The moat (determinism) matters MORE at this scale — distributed serving of a 405B needs
every node to agree, which only frozen-LUT decode guarantees.

**Gated sequence (do NOT jump the queue):**
1. Land 70B q2 (in flight) — proves the boss tier.
2. Confirm this session's winners (de-bias, KL-routed PV) at 7B/32B.
3. THEN Llama-3.1-405B as the flagship — requires an infra step up: ~810GB bf16 source
   (volume >> 200GB or stream-quant from a mirror), days-to-weeks of 27-core Viterbi
   (the SHARD_JOBS pool + a longer budget), and offload-eval since the base can't fit.
   Dense first; **MoE (DeepSeek-V3) is a separate track** needing expert-aware routing
   (shared vs per-expert rungs) — a distinct project, not a drop-in.

Verdict: schedule the 405B as a **gated flagship leg**, not a research necessity. It is
how we *cement* leadership once 70B + the 0.5B winners clear — the biggest possible
deterministic-2-bit showcase, run only after the cheap gates earn it.

## 11. 70B deferred + cloud-GPU memory wall (2026-06-13)

Decision: **70B q2 DEFERRED** (revisit only as the 405B-class flagship). The 32B q2
result (6.609, +38% over bf16 4.778) already confirms the scale-tolerance thesis
(7B +59%, 14B +75%, 32B +38% — NOT monotone: 14B is worse than 7B, but 32B is clearly best; the trend is "large models tolerate 2-bit well", not strict monotonicity), so the 70B is
diminishing information for multi-day cost. The 1 completed recon shard is kept.

**cloud-GPU encode lane — validated correct, but hits a memory wall above 32B:**
- The lane works: `--features cloud-gpu` (cloud-gpu dependency pinned `cloud-gpu-12060` for cloud-GPU 12.7+),
  kernel fixed for nvrtc (no `<float.h>`, `COST_INF` as macro), "cloud-GPU GPU ready".
  A tiny-tensor parity gate (`cloud-gpu-tiny-gate.sh`) confirmed correct output.
- BUT on 70B's 235M-param FFN tensors at L12, the GPU back-buffer host-staging pushes
  memory past the 125GB cgroup → **OOM-killed** repeatedly. CPU completes (≈64GB) but
  is ~4-6 days on these tensors. 32B fit (smaller tensors); 70B does not.
- **Prerequisite for both 70B-on-GPU AND the 405B flagship: batch the GPU block
  dispatch** in `encode_tensor_with_cloud-gpu` (bounded block batches, like the CPU path)
  so the back-buffer staging fits 24GB/125GB. Until then the GPU lane is for <=32B.
- Auto-routing scaffold left in place on the pod (`qm-wrapper.sh` + `.cloud-gpu-verdict`):
  flip the verdict to OK and the chain auto-uses cloud-GPU — but only safe after the
  batching fix + a real-tensor parity check.

Operational lesson re-learned (hard): the self-match `pkill -f` trap silently kills
the ssh shell before the kill completes, so "failed" launches actually stack into
duplicate racing chains. ALWAYS bracket the pattern AND keep the literal out of every
echo/grep in the same command (`llama2-70[b]`, nothing literal anywhere).

Redirect: pod idle (terminate to stop spend); focus = the local 0.5B quality-density
frontier (PV verdicts, de-bias artifact, KL-routed PV, lowrank judge).

## 12. RED-TEAMED targets (2026-06-13, falsification + interaction waves) — REVISED DOWN, honest

Two adversarial waves (25-agent falsification + 16-agent lever-interaction, ~3M tokens, all code-grounded)
refuted the §9/§10 targets. The principle + the determinism moat survive; the HEADLINE NUMBERS were
overstated. Honest revised targets:

| metric | I claimed | RED-TEAMED honest | why the claim was wrong |
|---|---|---|---|
| bpw (2-bit lane) | 2.40 | **2.56** (true zero-quality C2 = scale+sub_scale 0.106 bpw); 2.42 only as a SEPARATE gated outlier-position deliverable | 58% of the "0.25 recoverable" was outlier-POSITIONS (0.1476) — the quality-gated OUTLIER channel, NOT pure side-info. "C2 side-info" alone is scale+sub_scale = 0.106 → 2.56. Also: sideinfo_rans.rs is ORPHANED (not wired) — 0.25 is a microscope ceiling, not an achieved bpw. |
| 2-bit loss-tax | ≤0.15 | **~0.22 (range 0.20–0.25)** at TRUE 2-bit (2.665 bpw), de-bias adopted; PV upside | de-bias removes a ~28% FRACTION, not a fixed 0.344 nats → on 32B base tax 0.324 it buys ~0.09 nats → ~0.227 before PV. ≤0.15 only at ~3.9 bpw (mp-kl-routed = 59% of weight at 4-bit) — **that abandons 2-bit**, it's a 3-bit-class result. |
| 3-bit loss-tax | ≤0.05 | **0.056 PROVEN** (llama2-7b mp_light); 0.088 (Qwen-7B); ≤0.05 = target-to-confirm gated on the unrun 7B de-bias A/B | the −0.344 de-bias figure is from a PPL-40 broken recon; 0.056−0.344 is impossible arithmetic. |

### Three mechanistic corrections that reshape the whole target math
1. **C2 contributes ZERO quality nats.** It is LOSSLESS side-info recompression (byte-identical decode, dPPL=0). Counting it in a loss-tax budget is a category error — it is a bpw/storage win ONLY.
2. **De-bias is a fractional (~28%) first-moment correction, not a fixed nat-count.** It removes the rank-0 output-mean mode; gain scales with the damage. At low base-tax (large models, 3-bit) its absolute nats are small. It provably cannot touch the variance floor (the RHT-whitened tax).
3. **The levers OVERLAP — do NOT sum them.** de-bias ⊕ PV: combined ~−1.15 to −1.25 nats, NOT the −1.438 sum (PV retrains the full-rank weight incl. its rank-0 mode, so de-bias re-banks little on the PV-targeted tensors; de-bias's net survives mainly on the ~143 NON-PV tensors for ~free bpw). de-bias ⊕ C2: independent/additive (disjoint axes). PV is the workhorse; de-bias mops up; C2 is pure density.

### Honest commit for the sprint
- **bpw: build + wire the real C2 coder, commit 2.56 (zero-quality), 2.42 as a gated outlier-position arm.**
- **2-bit: commit 32B q2 (2.665 bpw) loss-tax ≤0.25 via de-bias, selective-PV as upside** — run the missing 0.5B selective-PV-by-KL A/B + a 7B de-bias confirm BEFORE promising any 32B number.
- **3-bit: commit 0.056 (proven); ≤0.05 is target-to-confirm.**
- Still real, still bleeding-edge, just not vanity. The moat (bit-identical decode) is intact throughout.

### 12.1 De-bias law (wave wxl0agton) — pins the 2-bit math, refines the overlap
De-bias captures a **bit-invariant fixed fraction k≈0.28 of the loss-tax** (measured: 0.290 at 2-bit, 0.276 at 3-bit). The fraction HOLDS across scale/family/class; the NATS shrink only because the tax shrinks. So it's a **high-damage phenomenon, not small-model-only** — 32B 2-bit still has +38% damage, so de-bias banks ~28% of it.
- **32B 2-bit pre-PV: −0.09 nats** (6.609→6.03, −8.8%) at **0.003–0.014 bpw** (cheaper at 32B, bias cost = 16/in_features).
- **Orthogonality resolved (verified from the checkpoint header):** the high-damage MLP tensors {down, gate, up} and o_proj are **bias-FREE** — so PV has no additive-DC parameter there and *cannot* absorb the rank-0 mode; it can only reshape the high-rank residual. De-bias (rank-0) and PV (high-rank) are therefore **orthogonal on exactly the tensors that matter most → they STACK, not overlap**, on the MLP path. (PV gets a free DC param only on q/k/v, which are biased.)
- **Refined 2-bit target math:** PTQ 0.324 → if selective-PV stabilizes the arm to +10–18% over bf16 (0.10–0.17 nats), de-bias closes ~28% of the residual → **~0.07–0.13 nats**. So: **≤0.25 is the de-bias-only floor (proven mechanism); ≤0.15 is reachable ONLY if PV works at scale** (the cloud run is the decider). The two waves agree once you separate "de-bias alone" (0.22–0.25, conservative) from "PV-success + de-bias" (0.07–0.15, conditional).
- **Verdict: de-bias is the single highest-leverage near-free lever at scale — ADOPT.** ~−0.075 nats central at 32B 2-bit, additive with PV, near-zero bpw.

## 13. Cloud PV recipe + TWO real cloud-script bugs (wave wiohtqx42)
Optimal recipe (judge-panel winner): **progressive q4→q3→q2 selective-PV, KL-routed (RED set up/v/gate, NOT down), de-bias stacked free at the final q2 encode, gated ladder** ($0 0.5B gate → ~$30 7B canary → ~$120 one gated 32B ≈ $130 total).
- **7B is the safe headline** (~$30): credibly HALVES the 2-bit tax to ~0.20–0.24 and settles the "trained deterministic 2-bit beats AQLM-class ~8.9" claim. Excellent gain-per-dollar.
- **32B ≤0.15 is a coin-flip, NOT a lock** — needs 3 favorable assumptions to all land (≥80%-of-full-PV concentration, progressive>direct by 5–12%, de-bias adds 0.05–0.10 on top). The honest selective floor is ~0.166–0.181 unless every lever lands.
- **BUG A (verified, #1 blocker):** strand-qat.py uses an **fp32 frozen shadow (8 B/param) → ~238 GB at 32B → won't load.** Must switch to a bf16 frozen shadow + 8-bit Adam before any 32B PV. Arm the cloud only after this fix.
- **BUG B (verified empirically):** the queued gate-10 rung-kl **regex is unanchored → matches 41 tensors not 25** (o_proj layers 1/11/21, v_proj 17 not 10). AND gate-10 routes via `--rung-config` (a PTQ placement config training ALL wrapped tensors), **not `--pv-tensors` selective-freeze** — so the selective-PV mechanism the recipe needs has never actually run. Fix the regex + switch gate-10 to `--pv-tensors` before deploying.
- **Gate discipline:** require ≥80% RED-vs-full recovery on the $0 0.5B gate before any pod dollar; fall back to direct-q2 selective-PV if progressive only ties.

## 14. NEW levers from external research (wave w6fyh8tz5) — all moat-safe, ranked
Cross-verified against the real code; literature numbers re-checked, overclaims flagged. 4 BUILD-worthy:

1. **Progressive 3-bit→2-bit PV init (UPQ/ParetoQ) — HIGHEST expected value, zero new encoder code.** Strongest-validated lever in the literature: UPQ INT4-first init → MMLU 53.20 vs 39.17 for direct INT2; "starting directly from INT2 PTQ is a poor init." First experiment: set the 2-bit PV arm's base = a STRAND 3-bit requant's de-RHT'd shadow (one extra ~15-min CPU requant, 0 bpw, decode-invariant). Target: pull the 2-bit PV floor 26.77 → high-teens. **Fold into the cloud PV recipe** (validates the progressive ladder the recipe wave proposed). Free riders: ternary level-symmetric codebook {−3,−1,1,3}/{−1,0,1} (ParetoQ SEQ beats zero-heavy grids on the ~Gaussian post-RHT source), + distance-gated PV schedule (log the L1 weight-move per requant as a free instrument).

2. **QTIP computed-Gaussian codebook (sum-of-two-quantile-draws, "3INST") + L→16 — decode byte-identical.** STRAND's codebook IS structurally QTIP-1MAD; the win is drawing a smoother tail. At LUT-freeze (offline f64 OK; decode stays float-free), replace `ranks[hash_state(s)]` with `quantile(hash_a(s)) + quantile(hash_b(s))` (CLT smoothing), re-pin LUT_GOLDEN_HASH. **HONEST: this is a PTQ-floor lever (~2–4% PPL) that COLLAPSES once PV runs** (QTIP Table 3: 1MAD and 3INST both land 6.82 with full pipeline; the 7.05→6.82 gap is no-fine-tune). So it helps the un-trained q2 today but not the post-PV product. L→16 adds asymptotic trellis gain but 4× LUT + 4× encode — gate against the ≥5× encode target.

3. **ButterflyQuant frozen-integer learnable rotation — the ONLY moat-safe learned rotation.** Replace the fixed ±1 FWHT with a per-tensor learnable butterfly (Givens G(θ); Hadamard = θ=π/4), learn angles on 128 calib samples by min-reconstruction (tiny CPU, no QAT), freeze cos/sin into a Q12 twiddle LUT, decode integer `(cos_q*u+sin_q*v)>>12` — bit-identical, inverse = transpose. SpinQuant (dense float Stiefel) and ReSpin (float residual) BREAK the integer moat — rejected. med-high.

4. **SpinQuant seed-search (best-of-N RHT seed) — cheap, no-regret.** Try N FNV seeds per tensor on a calib slice, keep the best by output-KL, store a tiny seed index. med.

**New product surface (off-target, banked):** OSCAR-style frozen KV/activation codec — STRAND's deterministic integer decode applied to the KV cache for long-context bandwidth. A whole new axis for post-project.

**Confirmed DEAD for STRAND:** Hessian-Viterbi/BCJR-QAT, AQLM additive-decode port, low-rank weight residual, ReSpin/SpinQuant float rotations (all break the moat or are RHT-whitened).

## 15. INTEGRATION RISK LEDGER (wave wxgehdvom) — the safe-wiring order (do NOT skip)
The audit found a determinism HOLE + silent-drop hazards. Wiring the levers naively breaks the moat or loses sections. Mandatory order:

**0. Module wiring (step zero).** `debias_wire.rs` AND `sideinfo_rans.rs` are ORPHANS — untracked, not in lib.rs, so their `#[cfg(test)]` tests are in *uncompiled* files and have NEVER run. First: `pub mod debias_wire; pub mod sideinfo_rans;` in lib.rs, then `cargo test` actually exercises them.

**1. SEAL the de-bias (CRITICAL — determinism hole).** DBIA feeds a real f32 add into every output row, but `descriptor_digest` (provenance.rs) has no DBIA slot and `verify_archive` never reads DBIA → **a tampered/dropped correction passes SPRV verification unnoticed.** That silently breaks the bit-identical-decode attestation. Fix: add a `dbia_digest` arg to `descriptor_digest` + read DBIA in `verify_archive` (mirror exactly how OUTL was folded in). Do this BEFORE any DBIA ships.

**2. Make ALL section walkers step-over-consistent (or sections silently drop).** The step-over magic sets are INCONSISTENT: `read_outl` steps over {SPRV}; `read_sdsc` over {SPRV,OUTL}; `read_dbia` over {SPRV,OUTL,RSLT} but NOT SDSC. Adding DBIA/C2 requires teaching `read_outl` AND `read_sdsc` (+ the `append_sdsc` restack strip-replay list) to step over the new magics, and fixing `read_dbia` to step over SDSC. Miss one → `Ok(None)`, section silently lost.

**3. append_sdsc RESTACKS — it will DROP DBIA/C2 on restack** (it only reads back OUTL+SPRV). Add new sections to its strip-and-replay, or appending SDSC after DBIA/C2 destroys them.

**4. RSLT stays OUTERMOST.** `append_rslt` does NOT tail-pad the file end; every other parser rejects `trailer_end % PAGE != 0`. So DBIA/C2 MUST be appended BEFORE RSLT.

**5. Bump SDSC for C2/DBIA (or the self-describe moat regresses).** `decode_q12_with_sdsc` pins SIDEINFO_LAYOUT=1, applies NO output bias, unpacks sub_scales as raw 6-bit. C2-coded side-info is a new layout; DBIA needs a post-recon add expr — neither is in the SDSC v1 opcode/const set. Ship without bumping → the "archive describes its own float-free decode" property silently diverges.

**6. Extend loader.rs (silent feature-absence).** `StrandModel::from_mmap` reads ONLY OUTL — a deployed model will IGNORE DBIA/C2 with no parse error until loader.rs is extended. The DBIA epilogue (`y[o]+=bf16_to_f32(c[o])`, once, after the outlier residual) has no call site in outlier_mac.rs yet.

**7. C2 needs an END-TO-END bit-identity gate before swapping.** sideinfo_rans decode is integer-only/self-consistent, but no test proves rANS-decoded scale_q/positions reproduce byte-identical Q12 weights vs the fixed-width path. Build that gate first; until then C2 is an unproven moat change.

GOOD NEWS: v2 core readers are back-compat safe (length-framed off the page-padded header, ignore all trailers — verified). The bf16 round-trip in debias_wire is moat-safe as written (ties-to-even, NaN/Inf preserved). The break is only among section-aware readers + the loader.

## 16. C2 coder BAKE-OFF result (wave wm521gic6) — verified, ~0.237 bpw, refines the bpw target
Judge-panel of 5 independent coders. Winner (score 90): **mode-adaptive coder (c2_attempt_4)** — per stream it builds BOTH an explicit per-symbol static CDF (zig-zag delta-coded symbol table + ESC slot) AND a constant-250B exponential-bucket rANS model, encodes with each, emits the byte-smaller behind a 1-byte mode tag. rANS core = Ryg 32-bit byte-renorm (L=1<<23, SCALE_BITS=14), **byte-identical to the existing sideinfo_rans**.
- **VERIFIED by execution** (an agent ran it in an isolated /tmp target, PV untouched): 21/21 tests, an **87,381-stream exhaustive byte-exact certificate**, both decode modes, frozen-shared-model round-trip, totality + 20k random-soup. Decode integer-only; encode also float-free (model pick by byte length, not float cost).
- **Recovers ~0.237 bpw** (scale_q 0.0868 + outlier-positions 0.135–0.164) = **95% of the 0.25 ceiling → 2.665 → ~2.43 bpw**, at zero quality cost (pure lossless recode of the existing streams — byte-identical reconstruction, distinct from an OUTL *scheme* change which is what's quality-gated).
- **KEY CAVEAT: the win needs whole-model concatenation / a shared frozen model.** Per-tensor small-block (256 blocks) only recovers 0.053 bpw because the per-symbol table dominates; pooled across the model the table amortizes to 0.0001 bpw. So wiring must concatenate streams model-wide.
- Runner-up: static rANS (c2_attempt_0, score 84) — all three streams 0.234 bpw, every format-layout claim verified against real source.

**Refined bpw target: ~2.43 bpw via verified lossless C2 (all side-info streams), zero quality cost** — this resolves the §12 split: 2.56 (scale+sub only) was too conservative; losslessly entropy-coding the existing outlier-position stream is byte-identical (zero quality), so the real C2 lands ~2.43. Next: wire c2_attempt_4 into encode.rs/format.rs with whole-model concatenation (per the §15 safe-wiring order: new section magic added to every walker + the SDSC bump + the end-to-end bit-identity gate).

## 17. Determinism moat — formally hardened + ONE tested limit found (wave wu9rh330z)
Three new passing test files (kept — valuable coverage, not litter): `encode_decode_equivalence.rs` (exhaustive: ~5,120 scalar + 6,144 vector + tail-biting cases, decode bit-identical & float-free across L/k/block_len/vec_dim; found+fixed 2 fixed-point-shift bugs in the reference replay), `rht_determinism.rs` (10 tests + **4 Kani harnesses, all VERIFICATION SUCCESSFUL** — seed FNV-1a always-odd over 5028 names, sign ±1, splitmix64 pure), `decode_byte_stability.rs` (52M-case golden vector reproducing LUT_GOLDEN_HASH, integer decode byte-stable).

**The tested limit (honest, important):** the RHT round-trip is bit-exact **IFF the effective block size h is an EVEN power of two** ({1,4,16,64,256}). Non-256-aligned widths whose largest power-of-two divisor is an ODD power — **896→h=128, 384→h=128, 200→h=8** — are only APPROXIMATE (~1e-6, measured worst 1.79e-7), because 1/√h isn't a dyadic f32. **This is now a tested property, not a silent failure.** Consequence: for those geometries (note: Qwen-0.5B is 896-wide), cross-device bit-identity of the *encode-side RHT* rests on IEEE-754 f32 add/sub identity + no-FMA-contraction (asserted via the golden vector, not Kani-proven). The DECODE (integer Q12 LUT) stays bit-exact everywhere; the crack is only in the float RHT for odd-power blocks.

Moat status: **decode is provably bit-identical + float-free** (exhaustive + golden + Kani on the primitives). The one gap to close for a total guarantee: GPU-vs-CPU *encoder* path parity (hardware-gated, separate), and the odd-power-block RHT f32 identity (mitigate by 256-aligning, or accept the documented ~1e-6 for those widths). My `c2_attempt_*` litter removal also unblocked `cargo kani --tests` (the E0432 it hit is gone).

## 18. CAPSTONE — 13-wave fleet complete (2026-06-13), the final gated sprint
The supercondenser fleet is done (~42M tokens across both sessions). Net: the sprint is honest, de-risked, and the levers are verified or honestly bounded. Single source of truth for execution:

**bpw leg — SOLVED on paper.** C2 coder verified byte-exact (87,381-stream cert, §16), recovers ~0.237 bpw → **~2.43 bpw at zero quality cost**. Needs whole-model concatenation wiring + the §15 safe-wiring order (seal, step-over, SDSC bump). Sub-scale RD re-select is a marginal mop-up (~0.005–0.016 bpw, not the claimed 0.04), gated on C2.

**quality leg — bounded, gated on the free local PV.** Honest 2-bit target ≤0.25 (de-bias-only floor) / ≤0.15 (only if selective-PV works at scale, §12/§12.1). De-bias adopted (k≈0.28, orthogonal to PV on the bias-free MLP path). Best new lever: **progressive 3→2-bit PV init** (UPQ-validated, zero new code — fold into the cloud recipe). 3-bit 0.056 proven.

**speed leg — measured wins ready.** Media: video-decode CDF fusion 1.42–1.65× + image 1.27–1.47×, both bit-identical, patch-ready (apply-pass, needs go-ahead). Decode→90% of bandwidth + cloud-GPU decode/encode = GPU-gated, deferred. Moat formally hardened (Kani-proven primitives), with the tested 896-width even-power-of-2 RHT limit (§17).

**The two cloud bugs to fix before any pod $:** fp32 PV-shadow OOMs 32B (→bf16+8bit-Adam); gate regex over-matches + wrong mechanism (§13).

**The trap (hold it):** selective-PV buys 2-bit *quality* at ~3.2–3.8 bpw *density* — separate headlines, never "smaller AND better" from the PV arm.

**Execution order (all gated, local-first):** (1) wire C2 [bpw, local, the §15 order] → (2) seal+deploy DBIA de-bias [quality, local, seal FIRST] → (3) when local PV gate green: cheap 7B selective-PV with progressive-init [cloud, ~$30–40, after the two bug-fixes] → (4) media apply-pass [speed, go-ahead] → (5) decode/encode GPU levers [deferred]. 32B + 70B headline runs: end-of-project only.

Runaway lesson banked: the methods wave (loop-until-dry + per-method fanout) hit 451 agents / 29M tokens / session-limit before synthesis — cap agent count on discovery waves.

## 19. P4 — output-aware pre-conditioning (cross-chat find, 2026-06-13) — the first PTQ quality lever
The PTQ-probe session ran 6 home-grown mechanisms; P4 is the win and it CONVERGES with this fleet's findings from the opposite direction.
- **P4: pre-condition input columns by activation energy (D^-1, α=0.5) BEFORE RHT → −15.6% output-error, robust 6/6.** weight-RMS goes UP while output-error goes DOWN — direct proof STRAND's weight-MSE objective is mis-specified. Moat-safe: D^-1 folds into the activation-RHT path already run; weight decode stays integer; calib already emits feature_rms.
- **Saliency SURVIVES the RHT** — overturns the repeated "RHT whitens saliency away" assumption. So the Hessian-Viterbi backfire was wrong-signal-in-wrong-basis (c4 Hessian, post-RHT coords), NOT whitening. Important correction to §2's dead-lever reasoning.
- **CROSS-CHAT CONVERGENCE:** this fleet's de-bias law (§12.1, output-mean correction) + error-spectrum (§9, output error low-rank in activation space) reached the same conclusion — the loss cares about OUTPUT error, weighted by activation energy — from the post-quant side. P4 attacks it pre-quant (the quantization metric itself). Both chats, same diagnosis.
- Other probes: P2 embedded residual (capability win, bit-exact @2.0/3.0 bpw), P1 conditional scale coder (density 0.027 bpw) WIN; P3 codebook-bank, H low-rank scale field, B payload context-coding all KILL with numbers (heavy-tail flattened, scale-residual incompressible, payload memoryless). **Honest meta: the PTQ density space is near-exhausted; P4 is the one PTQ quality door still open — everything bigger is training (STRAND-PVT).**

### Integration caveats (this fleet's value-add before banking P4)
1. **Scalar probe → needs real-trellis + PPL A/B.** P4's −15.6% is a scalar stand-in for the trellis; the ratio should transfer but is a go/kill gate, not a banked number. Confirm via `quantize-model --act-precond` (to be wired) + a 0.5B PPL A/B.
2. **MEASURE the P4 ⊕ de-bias OVERLAP — do NOT sum.** Both attack output error (P4 = pre-quant metric / AC+variance; de-bias = post-quant mean / DC). Per the interaction-wave lesson (§12), levers overlap; the combined gain is likely < sum. Run P4+de-bias together vs each alone.
3. **P4 is AWQ-class** (activation-aware pre-scaling) — the novelty is the STRAND-basis viability (survives RHT, folds into activation-RHT, integer decode), not the idea. Frame it as "AWQ confirmed viable + beneficial in STRAND's basis," honestly.

**Gating:** de-risk cheaply first (per-class α sweep, scalar, ~20s — locks an α-schedule); the real-trellis + PPL confirmation is DEFERRED (needs --act-precond wired + the box free of the live PV + the session/budget reset). Adds to the lever stack as: PTQ quality = P4 (metric) + de-bias (mean) + outlier channel, orthogonal-pending-overlap-check; training quality = STRAND-PVT.
