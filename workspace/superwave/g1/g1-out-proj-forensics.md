# G1 — why Qwen3.8 attention resists the current codec

Date: 2026-08-17
Lane: 05-out-proj-forensics
Source tensors: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16` (`PocketAiHub/Qwen3.8-27B-Abliterated-MLX-BF16`, base `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`)
Activations: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1` (`sha256_self=fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512`, 256 real BF16 post-norm hidden rows, width 5120)
Measurement JSON: `/tmp/qwen38_out_proj_forensics.json` sha256 `c85975745dbee0fd7b0bfc8d54f71161aacf73c067ac0609eba3ee224512243f`
Follow-up JSON: `/tmp/qwen38_out_proj_forensics_followup.json` sha256 `4110044edbe580a5ef143225da72934306d8bd8b7b75f08f184b7d803f70e89c`
Wall: 254.7 s + 22.2 s. Peak RSS: 7.050 GB (under 20 GB). CPU/numpy only. No GPU, no generate, no pack.

Every number below is **measured** on the real BF16 tensors unless marked **claimed** (prior receipt, unverified here) or **projected** (arithmetic on measured mass fractions).

---

## Named root cause

**Absmax group scale tracks |W|. Residual / mixer-output error tracks mixer-site |X| and write-gain. Those two axes are nearly orthogonal.**

Attached numbers (L0 `linear_attn.out_proj`, HGRAVU01 Q3 g=64, hold-out odd rows of the 256-token capture):

| quantity | L0 out_proj | L0 down_proj | Δ |
|---|---:|---:|---:|
| weight cosine | 0.9668 | 0.9686 | 0.0018 |
| group-normalized entropy (256-bin) | 7.5915 bit | 7.6162 bit | 0.0247 |
| mixer-output cosine | 0.9531 | 0.9915 | 0.0384 |
| output/weight rel-L2 amplification | 1.492 | 0.469 | |
| residual-proxy cosine (add write to post-norm hidden) | 0.99745 | 0.99952 | |
| write RMS / residual RMS | 0.353 | 0.381 | |
| \|W\|-hottest 42 cols ∩ \|X\|-hottest 42 cols | **0 / 42** | 530 / 2588 | |

Protecting the 42 fattest **weight** columns and Q3-ing the rest changes L0 out_proj output cosine **0.9531 → 0.9533** (+0.0002).
Protecting the 42 hottest **activation-energy** columns changes it **0.9531 → 0.9762** (+0.0231).
Those two 42-column sets are disjoint.

This is why the current codec cannot take attention below Q4 at the 0.99 **mixer-output** bar, and why that fact is not evidence that attention is incompressible.

---

## Mechanism class that fixes this root cause

**In-register group quantizer whose scale (or bit-width) minimizes `||(W−Ŵ) X_mixer||` (mixer-output MSE) or `||Δ residual||`, not `||W−Ŵ||_∞`.**

Late GQA `o_proj` additionally needs a write-gain term in the bit budget: L63 write/residual RMS = **1.845**, so residual-proxy Q3 cosine **0.9775** ≈ mixer-output cosine **0.9604**. Early DeltaNet `out_proj` does not: L0 write/residual = **0.353**, residual-proxy Q3 = **0.9974** while mixer-output is **0.9531**.

What this class is not:

- Not “extract |W| outliers then absmax the body”. Measured: extract-k=1 → 0.9703; extract-k=2 → 0.9759; sparse-top-1% |W| → 0.9719; fattest-1% rows exact → 0.9638. All fail 0.99 on L0 out_proj.
- Not “regroup to head boundaries”. g=64 is already head-aligned (128/64=2, 256/64=4). Intra-head slot error ratio 1.014. g=48 (straddles 128-d heads) is *better* (0.9602), not worse.
- Not Hadamard-then-absmax. L0 out_proj Q3: 0.9601.
- Not naive per-column X-RMS fold into W. Prior probe (claimed): `HGRAVU01_q3_g64_act_colscale` on L0 out_proj output cosine **0.8426**, worse than unscaled Q3 **0.9532**. Receipt: `receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json` tensor `qwen38.L0.linear_attn.out_proj`.
- Not expand-to-float-then-generic-GEMV. Binding forbids that path unless a complete-token measurement shows a net win. This lane did not run tokens.

A shipping kernel for this class is a Qn-family matvec that already exists for absmax scales (`qwen_uniform_q4_group64_matvec` / `qwen_uniform_qn_*`), with the scale tensor computed from `X_mixer` (or from `X_mixer` Gram per group) instead of `max |W|`. That is a pack-time change plus the same in-register dequant.

This lane did **not** implement that codec. The column-ablation is the existence proof that the |X| axis is the one that moves output cosine, and the |W| axis is not.

---

## Starting-point claim (prior, not re-derived as a floor)

Claimed in `receipts/ascent-2026-08-16/QWEN38_DENSITY_ROOT_CAUSE.json`:

> Qwen3.8's BPW floor is set by ATTENTION, not by the MLP. The MLP is already at 0.848 BPW. Attention+embed+norms sit at 4.250 BPW and are 5.20 GB of the 7.01 GB artifact - 74%. Complete BPW is therefore attention-dominated at 2.0856.

That is a **pack-composition** fact about `mixed-2p0-v1`, not a statement about attention entropy. Confirmed on disk at `.../qwen38-27b/mixed-2p0-v1/PACK_REPORT.json`:

```
complete_physical_bpw  2.0855934079220506
mlp_physical_bpw       0.8480504639008466
nonmlp_physical_bpw    4.250142713483966
mlp_down_proj          0.13161714918473189 BPW   (HGRAVS01)
attention_embed_norm   4.250142713483966 BPW
```

`mixed-2p0-v1` kept attention at Q4 and crushed `down_proj` to 0.13 BPW. Native generate on that artifact is **claimed** incoherent (`receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json`). That incoherence cannot be charged to attention compression: attention was not compressed. This lane did not re-run generate.

Descent scored every `attn_out` candidate `quality_space=weight_only`, `n_fit=0` (`QWEN38_BPW_DESCENT.json`, L0/L3/L47/L63 `attn_out`). That is a measurement hole, not a property of the tensor.

---

## Codec under test

Shipping HGRAVU01 / HQ30UQ4: flatten W in C-order `[out, in]`, groups of 64 consecutive elements, one f16 scale = absmax / (2^{b-1}−1), round-to-nearest.

```3:6:crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs
//! Layout matches `qwen_uniform_q4.metal`:
//!   * magic `HQ30UQ4\\0`, version 1, group_size 64
//!   * FP16 scale per group of 64 flat elements
//!   * 32 code bytes per group (even nibble low, odd high; q = nibble - 8)
```

```173:189:tools/qwen38_sub15_pack.py
def pack_hq30uq4(values: np.ndarray, shape: list[int]) -> bytes:
    """Match hawking-core pack_uniform_q4_group64 (HQ30UQ4 v1)."""
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    ...
    grouped = padded.reshape(groups, Q4_GROUP)
    max_abs = np.max(np.abs(grouped), axis=1)
    scale = (max_abs / 7.0).astype(np.float16)
    quant = np.rint(grouped * inv[:, None]).clip(-8.0, 7.0)
```

For `out_proj` / `o_proj` shape `[5120, 6144]`, `6144 % 64 == 0`, so groups never straddle output rows. Geometry (`crates/hawking-core/src/model/qwen38_geometry.rs`):

- DeltaNet `out_proj` in-dim 6144 = 48 value heads × 128. `128 % 64 == 0` → 2 groups/head, aligned.
- GQA `o_proj` in-dim 6144 = 24 heads × 256. `256 % 64 == 0` → 4 groups/head, aligned.
- GQA layers: `(layer+1) % 4 == 0`.

`out_proj` X is the same mixer-site proxy the density probe used (not the recurrent DeltaNet mix, not softmax GQA): `v * silu(z)` or `repeat(v) * sigmoid(q_gate)`. Labelled as a site proxy. Residual proxy uses captured post-norm hidden (width 5120) as the stand-in residual; that is **not** the pre-norm residual (RMSNorm is not inverted). Labeled below.

Hold-out: odd rows of the 256-token capture, matching `lab/operators/qwen_attention_density_probe.py`. L0 out_proj Q3 output cosine 0.95310 vs probe 0.95317 (same tensor, same codec).

---

## Hypothesis verdicts

### A. Outlier channels destroy group absmax scale — KILLS as stated. A refined |W|-vs-|X| form SUPPORTS.

Discriminating measurement: (i) per-output vs per-input channel RMS spread; (ii) drop fattest 1% output rows and re-Q3 the body; (iii) store fattest |W| columns exact, Q3 the rest; (iv) store hottest |X| columns exact, Q3 the rest; (v) overlap of those two column sets.

Measured:

- L0 out_proj excess kurtosis **149.36**. After dropping the fattest 1% of **output rows**, kurtosis **1.25**. The kurtosis is a few fat residual-side rows.
- Those fat rows have their own groups (row-major g=64). They cannot pollute other rows’ scales.
- Input-channel RMS max/median = **1.24**. There are no fat **input** channels in |W|.
- Fraction of groups with absmax > 8× group-median: **0.58%** (L0 out) vs **0.22%** (L0 down). Not a group-scale collapse.
- Keep fattest 1% **rows** exact, Q3 the body: output cosine **0.9638**. Fail.
- Q3 the body after **dropping** those rows: **0.9605**. Fail.
- Store 42 fattest |W| **columns** exact: **0.9533**. Fail. Overlap with 42 hottest |X| columns: **0**.
- Store 42 hottest |X| columns exact: **0.9762**. Moves. Still below 0.99.
- Q3 **only** the 16 hottest |X| columns and keep the other 6128 exact: **0.9865**. Most of the *concentrated* damage is on those 16 columns; the body still costs ~0.033 cosine to 0.9531.

KILLS: “a few outlier channels destroy group absmax, and extracting them lets Q3 through.”
SUPPORTS: bit allocation by |W| is the wrong axis. |X| is the axis.

REOPEN_IF: a group-MSE-against-X (not absmax, not column-RMS-fold) codec is scored on the same hold-out and still misses residual-proxy 0.99 on late GQA `o_proj`.

### B. Attention weights are genuinely higher entropy per element — KILLS.

Discriminating measurement: histogram entropy of (i) z-scored raw W, (ii) W / group-absmax, (iii) same after dropping the group-max element. Same codec, same g=64, attention vs MLP on the same layers.

Measured (256-bin):

| tensor | raw z-score H | group-norm H | group-norm H drop-max | Q3 weight cosine |
|---|---:|---:|---:|---:|
| L0 out_proj | 5.9808 | 7.5915 | 7.5577 | 0.9668 |
| L0 down_proj | 6.0294 | 7.6162 | 7.5826 | 0.9686 |
| L0 gate_proj | 6.0448 | 7.6243 | 7.5910 | 0.9695 |
| L0 in_proj_qkv | 6.0344 | 7.6214 | 7.5884 | 0.9694 |
| L3 o_proj | 6.0045 | 7.6060 | 7.5728 | 0.9677 |
| L32 out_proj | 6.0376 | 7.6041 | 7.5708 | 0.9687 |
| L63 o_proj | 6.0209 | 7.4824 | 7.4459 | 0.9619 |
| L63 down_proj | 6.0386 | 7.5433 | 7.5080 | 0.9653 |

Attention is not a higher-entropy organ. Q3 weight cosine is the same number, 0.962–0.970, on every tensor measured. The “MLP compresses, attention does not” split is not in the weight distribution.

### C. Group size interacts badly with head boundaries — KILLS.

Discriminating measurement: (i) does g=64 straddle a head? (ii) Q3 error by intra-head group slot; (iii) Q3 output cosine at g=48 (straddles 128-d heads), g=64 (aligned), g=96, g=128 (one group per DeltaNet head).

Measured L0 out_proj:

```
groups_aligned_to_head: true
groups_cannot_straddle_output_rows: true
groups_per_head: 2
mean_sq_err_by_intra_head_slot: [1.725e-5, 1.749e-5]
slot_max_over_min: 1.014
```

L3 o_proj slots (4 per 256-d head): ratio **1.019**. L32: **1.001**. L63: **1.061**.

Q3 output cosine vs group size, L0 out_proj: g48=0.9602, g64=0.9531, g96=0.9434, g128=0.9443.
Smaller groups help (more scales). Head-aligned g=128 is worse. Straddling g=48 is better. This is the ordinary absmax group-size tradeoff, not a head-boundary defect.

### D. The error metric was weight-space rather than output-space — PARTIAL. Process hole: yes. Explanation of the floor: no.

Discriminating measurement: score the same Ŵ in both spaces, attention vs MLP, and also in residual-proxy space.

Descent `attn_out` is `quality_space=weight_only` (`QWEN38_BPW_DESCENT.json`). That hole is real.

It is not why Q3 fails a 0.99 **output** bar: this lane and the density probe both score output space, and L0 out_proj Q3 is 0.9531 there.

What D gets right is that the **0.99 mixer-output bar is the wrong residual-stream bar on early DeltaNet out_proj**:

| layer / organ | Q3 mixer-output cosine | Q3 residual-proxy cosine | write / residual RMS |
|---|---:|---:|---:|
| L0 out_proj | 0.9531 | **0.99745** | 0.353 |
| L0 down_proj | 0.9915 | 0.99952 | 0.381 |
| L3 o_proj | 0.9834 | **0.99109** | 1.353 |
| L3 down_proj | 0.9807 | 0.99963 | 0.155 |
| L32 out_proj | 0.9679 | 0.98738 | 0.843 |
| L32 down_proj | 0.9740 | 0.99730 | 0.316 |
| L63 o_proj | 0.9604 | 0.97749 | **1.845** |
| L63 down_proj | 0.9728 | 0.97324 | **2.792** |

If the bar is residual-proxy ≥ 0.99, L0 out_proj Q3 **passes** (0.99745) and L3 o_proj Q3 **passes** (0.99109). L32 and L63 fail. The density-probe 0.99 mixer-output bar fails L0 for a write that is 0.35× the residual and nearly orthogonal to it (Y·R = 0.0076).

MLP was previously judged against the 0.8604 organ bar (`QWEN38_BPW_DESCENT.json` `bars.q80_residual_identity`). Against the 0.99 output bar used for attention, L3/L32/L63 `down_proj` Q3 also fail (0.9807 / 0.9740 / 0.9728). “MLP compresses to <1 bit, attention cannot” is a **bar mismatch**, not an entropy mismatch.

### E. o_proj error compounds through the residual differently than MLP error — SUPPORTS on late GQA. KILLS as the L0 story.

Discriminating measurement: write RMS / residual RMS, write·residual cosine, residual-proxy cosine after adding Y vs Yq. Same captured hidden as residual stand-in for both organs. Not a multi-layer generate. Not pre-norm residual.

Measured write-gain:

- L0 DeltaNet out_proj: write/R = 0.353, Y·R = 0.0076, residual-proxy Q3 = 0.99745. Diluted, orthogonal. Residual compounding is **not** why L0 looks bad on mixer-output cosine.
- L3 GQA o_proj: write/R = **1.353**, residual-proxy Q3 = 0.99109. Write larger than residual; still clears 0.99 residual.
- L32 DeltaNet out_proj: write/R = 0.843, residual-proxy Q3 = 0.98738. Fails 0.99 residual.
- L63 GQA o_proj: write/R = **1.845**, residual-proxy Q3 = 0.97749 ≈ output 0.9604. Not diluted.
- L63 down_proj: write/R = **2.792**, residual-proxy Q3 = 0.97324. Late MLP write is also larger than residual.

o_proj is not uniquely a residual-compounder. Late layers of **both** organs have write-gain > 1. Early DeltaNet out_proj is the opposite. A codec that only “protects o_proj because residual” will over-protect L0 and under-protect L63 down_proj.

---

## Full Q3/Q4 table (measured)

HGRAVU01 absmax, g=64, hold-out odd rows. `clears 0.99` is mixer-output cosine.

| tensor | Q4 out | Q4 min-row | Q3 out | Q3 min-row | Q3 w | Q3 amp | Q2 out |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 out_proj | 0.9922 | 0.9877 | 0.9531 | 0.9154 | 0.9668 | 1.492 | 0.7063 |
| L0 in_proj_qkv | 0.9961 | 0.9937 | 0.9795 | 0.9670 | 0.9694 | 0.809 | 0.8387 |
| L0 down_proj | 0.9984 | 0.9964 | 0.9915 | 0.9804 | 0.9686 | 0.469 | 0.8866 |
| L0 gate_proj | 0.9964 | 0.9951 | 0.9810 | 0.9742 | 0.9695 | 0.756 | 0.8480 |
| L3 o_proj | 0.9967 | 0.9939 | 0.9834 | 0.9739 | 0.9677 | 0.658 | 0.8110 |
| L3 q_proj | 0.9970 | 0.9957 | 0.9838 | 0.9772 | 0.9685 | 0.717 | 0.8691 |
| L3 down_proj | 0.9964 | 0.9951 | 0.9807 | 0.9742 | 0.9694 | 0.806 | 0.8366 |
| L32 out_proj | 0.9938 | 0.9930 | 0.9679 | 0.9637 | 0.9687 | 1.009 | 0.7686 |
| L32 down_proj | 0.9950 | 0.9941 | 0.9740 | 0.9687 | 0.9682 | 0.904 | 0.8020 |
| L63 o_proj | 0.9925 | 0.9894 | 0.9604 | 0.9456 | 0.9619 | 0.888 | 0.7233 |
| L63 q_proj | 0.9983 | 0.9979 | **0.9909** | 0.9885 | 0.9661 | 0.506 | 0.9084 |
| L63 down_proj | 0.9950 | 0.9903 | 0.9728 | 0.9519 | 0.9653 | 0.856 | 0.7878 |
| L63 gate_proj | 0.9988 | 0.9978 | **0.9937** | 0.9883 | 0.9678 | 0.423 | 0.9356 |

Q4 clears 0.99 mixer-output on every tensor in this set (L0 out_proj min-row 0.9877 sits under 0.99; mean 0.9922 clears). Q3 clears 0.99 mixer-output only on L0 down_proj, L63 q_proj, L63 gate_proj. Q2 clears none of the 0.99 bars.

---

## Counterfactuals on L0 out_proj Q3 (the worst mixer-output cell)

| codec | output cosine | clears 0.99 | note |
|---|---:|---|---|
| absmax q3 g64 (incumbent) | 0.9531 | no | |
| absmax q3 g48 | 0.9602 | no | straddles 128-d heads |
| absmax q3 g128 | 0.9443 | no | one group / DeltaNet head |
| p99-scale q3 | 0.9627 | no | clip group absmax to p99 |
| extract 1 max/group exact | 0.9703 | no | +0.344 BPW if every group |
| extract 2 max/group exact | 0.9759 | no | +0.688 BPW |
| sparse top 0.01 \|W\| exact | 0.9719 | no | |
| fattest 1% rows exact | 0.9638 | no | |
| Hadamard then absmax q3 | 0.9601 | no | |
| drop fattest 1% rows, Q3 body | 0.9605 | no | |
| exact 16 |X|-hot cols, rest Q3 | 0.9680 | no | +0.042 BPW |
| exact 42 |X|-hot cols, rest Q3 | 0.9762 | no | +0.109 BPW |
| exact 3555 |X|-hot cols, rest Q3 | 0.9950 | yes | +9.26 BPW; not sparse |
| Q3 only 16 |X|-hot, rest exact | 0.9865 | no | localises damage |
| exact 42 fattest \|W\| cols | 0.9533 | no | overlap with |X|-hot = 0 |

No cheap |W|-side patch reaches 0.99 mixer-output. A cheap |X|-side patch of 42 columns does not either. Reaching 0.99 mixer-output by exacting columns requires 58% of the in-dim (3555/6144). The residual-proxy bar, on the same Ŵ, is already 0.99745 without any patch.

---

## Mixer-site X structure (why |W| and |X| diverge)

L0 out_proj X = `v * silu(z)` from fused in_proj on captured hidden:

- excess kurtosis **5538.7**
- 50% of energy in **16 / 6144** columns (0.26%)
- 90% of energy in **42 / 6144** columns (0.68%)
- 99% of energy in **3555 / 6144** columns (57.9%) — long tail after the spike
- column RMS max/median **83.4**

L0 down_proj X = `silu(X@Wg.T)*(X@Wu.T)`:

- excess kurtosis **22444** (spikier)
- 50% of energy in 16 / 17408 (0.09%)
- 90% of energy in **2588 / 17408 (14.9%)**
- The spike sits on a fat body. Q3 averages. Amplification 0.47.

L3 o_proj X (GQA `repeat(v)*sigmoid(gate)`): 50% energy in **29.2%** of columns. Not a 16-column spike. Q3 output 0.9834, residual-proxy 0.9911.

The L0 DeltaNet mixer proxy is a near-one-hot write into 42 of 6144 inputs. Absmax(|W|) does not see those 42 columns (in-channel |W| RMS max/median 1.24; overlap 0). That is the L0-specific half of the root cause. The L63 half is write-gain 1.845, not a spike.

---

## Projected complete BPW if this root cause is fixed (not measured)

Mass fractions from `QWEN38_BPW_DESCENT.json` (claimed, used as arithmetic only): mlp 0.6363, attention+norms 0.2692, embed+lm_head 0.0945.

- Attention → Q3 (3.25), MLP stays 0.848, embed Q4: **projected** complete BPW `0.6363*0.848 + 0.2692*3.25 + 0.0945*4.25 = 1.817`.
- Attention → Q3, MLP stays Q4: **projected** 3.981.
- Attention → Q2 (2.25), MLP 0.848, embed Q4: **projected** 1.548. Q2 L0 out_proj mixer-output cosine is 0.7063. Dead under any bar used in this lane.
- G1 target 1.5 with embed at Q4 and attention at Q3 forces MLP **projected** ≤ 0.35 BPW. That is the `mixed-2p0` down_proj-0.13 regime, claimed incoherent.

These are storage-mass projections, not token-level claims, not generate claims.

---

## Command output (measurement)

```
$ /opt/homebrew/bin/python3 /tmp/qwen38_out_proj_forensics.py
[10:58:57] rss_max=0.033G ===== layer 0 gqa=False =====
[10:58:57] rss_max=1.096G eval L0.linear_attn.in_proj_qkv shape=(10240, 5120) full=False
[10:59:05] rss_max=3.115G eval L0.linear_attn.out_proj shape=(5120, 6144) full=True
[10:59:16] rss_max=3.391G eval L0.mlp.down_proj shape=(5120, 17408) full=True
[10:59:48] rss_max=7.050G eval L0.mlp.gate_proj shape=(17408, 5120) full=False
[10:59:59] rss_max=7.050G checkpointed L0 -> /tmp/qwen38_out_proj_forensics.json
[10:59:59] rss_max=7.050G ===== layer 3 gqa=True =====
[10:59:59] rss_max=7.050G eval L3.self_attn.q_proj shape=(12288, 5120) full=False
[11:00:08] rss_max=7.050G eval L3.self_attn.o_proj shape=(5120, 6144) full=True
[11:00:21] rss_max=7.050G eval L3.mlp.down_proj shape=(5120, 17408) full=True
[11:00:55] rss_max=7.050G eval L3.mlp.gate_proj shape=(17408, 5120) full=False
[11:01:07] rss_max=7.050G checkpointed L3 -> /tmp/qwen38_out_proj_forensics.json
[11:01:07] rss_max=7.050G ===== layer 32 gqa=False =====
[11:01:07] rss_max=7.050G eval L32.linear_attn.in_proj_qkv shape=(10240, 5120) full=False
[11:01:15] rss_max=7.050G eval L32.linear_attn.out_proj shape=(5120, 6144) full=True
[11:01:26] rss_max=7.050G eval L32.mlp.down_proj shape=(5120, 17408) full=True
[11:01:57] rss_max=7.050G eval L32.mlp.gate_proj shape=(17408, 5120) full=False
[11:02:09] rss_max=7.050G checkpointed L32 -> /tmp/qwen38_out_proj_forensics.json
[11:02:09] rss_max=7.050G ===== layer 63 gqa=True =====
[11:02:09] rss_max=7.050G eval L63.self_attn.q_proj shape=(12288, 5120) full=False
[11:02:18] rss_max=7.050G eval L63.self_attn.o_proj shape=(5120, 6144) full=True
[11:02:29] rss_max=7.050G eval L63.mlp.down_proj shape=(5120, 17408) full=True
[11:03:00] rss_max=7.050G eval L63.mlp.gate_proj shape=(17408, 5120) full=False
[11:03:12] rss_max=7.050G checkpointed L63 -> /tmp/qwen38_out_proj_forensics.json
[11:03:12] rss_max=7.050G DONE wall_s=254.7 rss_max_gb=7.050
```

```
$ /opt/homebrew/bin/python3 /tmp/qwen38_out_proj_forensics_followup.py
[11:05:57] rss_max=0.034G followup L0
[11:06:03] rss_max=4.006G checkpoint L0
[11:06:03] rss_max=4.069G followup L3
[11:06:09] rss_max=4.069G checkpoint L3
[11:06:09] rss_max=4.069G followup L32
[11:06:16] rss_max=4.069G checkpoint L32
[11:06:16] rss_max=4.069G followup L63
[11:06:19] rss_max=4.069G checkpoint L63
[11:06:19] rss_max=4.069G DONE 22.2s
```

```
$ shasum -a 256 /tmp/qwen38_out_proj_forensics.json /tmp/qwen38_out_proj_forensics_followup.json
c85975745dbee0fd7b0bfc8d54f71161aacf73c067ac0609eba3ee224512243f  /tmp/qwen38_out_proj_forensics.json
4110044edbe580a5ef143225da72934306d8bd8b7b75f08f184b7d803f70e89c  /tmp/qwen38_out_proj_forensics_followup.json
```

JSON excerpts (L0 out_proj weight / group / Q3; follow-up ablation):

```
# /tmp/qwen38_out_proj_forensics.json  layers.0.out_proj
weight.excess_kurtosis                              149.3577
weight.excess_kurtosis_drop_fattest_1pct_rows         1.2540
weight.per_output_channel_rms.max_over_median        20.7009
weight.per_input_channel_rms.max_over_median          1.2414
group_g64.median_med_over_absmax                      0.2515
group_g64.frac_groups_absmax_gt_8x_median             0.0058
group_g64.group_normalized_entropy_bits_256bin        7.5915
codecs.absmax_q3_g64.weight_cosine                    0.96675
codecs.absmax_q3_g64.output_cosine                    0.95310
codecs.absmax_q3_g64.output_over_weight_rel_l2        1.49197
head_alignment_q3_g64.groups_aligned_to_head          true
head_alignment_q3_g64.slot_max_over_min               1.01410
```

```
# /tmp/qwen38_out_proj_forensics_followup.json  layers.0.out_proj
x_energy.n50 / n90 / n99                     16 / 42 / 3555
baseline_q3.output_cosine                     0.95310
baseline_q3.residual_proxy_cosine             0.99745
baseline_q3.write_rms_over_residual_rms       0.3528
exact_fattest_w_cols_n90_rest_q3.output_cosine 0.95331
exact_fattest_w_cols_n90_rest_q3.overlap_with_x90  0
exact_top90e_rest_q3.output_cosine            0.97616
q3_only_top50e_rest_exact.output_cosine       0.98653
```

```
# /tmp/qwen38_out_proj_forensics_followup.json  layers.63.o_proj
baseline_q3.output_cosine                     0.96040
baseline_q3.residual_proxy_cosine             0.97749
baseline_q3.write_rms_over_residual_rms       1.8450
```

---

## Prior receipt excerpts

`receipts/ascent-2026-08-16/QWEN38_DENSITY_ROOT_CAUSE.json` (claimed pack composition):

```
WATCHDOG_L4_ROOT_CAUSE: "Qwen3.8's BPW floor is set by ATTENTION, not by the MLP.
The MLP is already at 0.848 BPW. Attention+embed+norms sit at 4.250 BPW and are
5.20 GB of the 7.01 GB artifact - 74%. Complete BPW is therefore
attention-dominated at 2.0856."
evidence.complete_physical_bpw = 2.0855934079220506
evidence.mlp_physical_bpw      = 0.8480504639008466
evidence.nonmlp_physical_bpw   = 4.250142713483966
```

`receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json` (claimed; this lane remeasured the Q3 numbers and matches):

```
"if_q3_all_attention_rejected": {
  "why": "out_proj Q3 cosine 0.953 (L0, kurtosis 149) / 0.968 (L32).
          in_proj Q3 0.975-0.980. Not a Q4-quality pack.",
  "verdict": "QUALITY_FAIL"
}
```

`receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json` `coherence_floor.capture_limit`:

```
"attn_out in-dim is 6144 and was weight-scored only."
```

`receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json` (claimed generate; not re-run):

```
"4.2527_BPW_q4_oracle": "COHERENT"
"2.0856_BPW_mixed-2p0-v1": "INCOHERENT (native, verified twice)"
```

`mixed-2p0-v1` attention BPW is 4.250. The incoherent artifact did not compress attention.

---

## What this lane did not measure

- Generate / coherence / token identity. Serialized GPU lane owns that. Residual-proxy ≥ 0.99 is **not** a generate claim.
- Pre-norm residual (RMSNorm not inverted). Residual-proxy uses post-norm hidden as stand-in.
- True DeltaNet recurrent mix or softmax GQA mix as `out_proj` X. Site proxy only, same as the density probe.
- Output-MSE scale search (closed-form / 1-D per group against X). Column ablation is the cheaper discriminator; the search is the first implementation step of the mechanism class.
- Layers other than 0, 3, 32, 63.
- lm_head / embed.
- A packed artifact or a Metal kernel.

Cheapest experiment that would turn the mechanism class into a generate-facing claim: pack L0/L32 `out_proj` and L3/L63 `o_proj` with per-group scale = argmin_s ||X_mixer (w − quant_s(w))|| on the 192-row fit split, keep HGRAVU01 codes, consume with the existing uniform-Qn kernel, and let the GPU lane score native generate against the Q4 oracle. Do not expand to float.

---

## Completion report

```
STATUS
SUPPORTED

CLAIMS
1. Named root cause: HGRAVU01 absmax scale tracks |W|; mixer-output and residual error track mixer-site |X| and write-gain; on L0 out_proj those column sets are disjoint (0/42). Evidence: /tmp/qwen38_out_proj_forensics_followup.json layers.0.out_proj exact_fattest_w_cols_n90_rest_q3.overlap_with_x90 == 0 and exact_top90e_rest_q3.output_cosine == 0.97616 vs baseline 0.95310.
2. Attention is not higher-entropy than MLP. Group-norm entropy 7.5915 vs 7.6162 bit; Q3 weight cosine 0.9668 vs 0.9686. Evidence: /tmp/qwen38_out_proj_forensics.json layers.0.out_proj and layers.0.down_proj.
3. Outlier-channel extraction does not recover Q3 mixer-output 0.99 (best cheap |W| patch 0.9759). Evidence: layers.0.out_proj.codecs.extract_k2_exact_body_q3_g64.output_cosine.
4. g=64 is head-aligned; intra-head slot error ratio 1.014; straddling g=48 is better (0.9602). Evidence: layers.0.out_proj.head_alignment_q3_g64 and codecs.absmax_q3_g48.
5. L0 out_proj Q3 residual-proxy cosine is 0.99745 at write/R=0.353; L63 o_proj is 0.97749 at write/R=1.845. Evidence: follow-up JSON layers.0.out_proj.baseline_q3 and layers.63.o_proj.baseline_q3.
6. mixed-2p0 2.0856 BPW incoherence is not evidence attention cannot compress: that pack left attention at 4.250 BPW and crushed down_proj to 0.1316 BPW. Evidence: mixed-2p0-v1/PACK_REPORT.json organ_breakdown.

EVIDENCE
- /tmp/qwen38_out_proj_forensics.json sha256 c85975745dbee0fd7b0bfc8d54f71161aacf73c067ac0609eba3ee224512243f
- /tmp/qwen38_out_proj_forensics_followup.json sha256 4110044edbe580a5ef143225da72934306d8bd8b7b75f08f184b7d803f70e89c
- measurement command log in this file, wall 254.7 s + 22.2 s, rss_max 7.050 GB
- crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs:3-6
- tools/qwen38_sub15_pack.py:173-189
- crates/hawking-core/src/model/qwen38_geometry.rs:20-52
- receipts/ascent-2026-08-16/QWEN38_DENSITY_ROOT_CAUSE.json
- receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json (attn_out quality_space=weight_only)
- receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json / VERDICT.json
- .../qwen38-27b/mixed-2p0-v1/PACK_REPORT.json
- .../qwen38-27b/activation-capture-v1/capture-result.json sha256_self fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512

CHANGES
workspace/superwave/g1/g1-out-proj-forensics.md (this file, untracked). No other path touched.

TESTS
see end of lane message (test -s, wc -l, git status --porcelain)

RISKS
- out_proj X is a mixer-site proxy, not the recurrent / softmax mix. A true-mix X could move the hot-column set. REOPEN_IF a captured mixer-output X (width 6144) exists and the |W|∩|X| overlap is no longer ~0.
- residual-proxy uses post-norm hidden, not pre-norm residual. Magnitude of write/R can shift under inverse-RMSNorm. Directional claims (diluted vs not) should survive.
- 256 tokens / 4 layers. Not a 64-layer census.
- Residual-proxy ≥ 0.99 is not generate.

UNRESOLVED
- Output-MSE-optimal group scale was not fitted. Column ablation is the discriminator, not the codec.
- No generate. GPU lane owns that.
- Whether late-layer write-gain > 1 is a Qwen3.8-wide fact or a 4-layer sample.

NEXT
Fit per-group scale = argmin ||X_mixer (w − q(w,s))|| on the 192-row fit split for out_proj/o_proj; pack HGRAVU01 codes with those scales; consume with the existing uniform-Qn kernel; hand to the GPU lane for native generate vs the Q4 oracle. Do not extract |W| outliers. Do not regroup to heads. Do not fold column RMS into W (already failed).
```
