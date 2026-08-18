#!/usr/bin/env python3
"""G1 cost model: fit on named anchors, predict the rest, score mechanisms.

No GPU. Arithmetic only. Every printed tag is MEASURED / DERIVED / PROJECTED.
"""
from __future__ import annotations

N = 26_895_998_464
E_MLP = 17_112_760_320
E_ATTN = 7_237_795_840
E_TAB = 2_542_796_800
E_SMALL = 2_645_504
E_GEMV_ALL = 26_893_352_960  # includes embed+lm_head
E_GEMV_NO_EMBED = E_GEMV_ALL - 1_271_398_400
E_ATTN_PROJ = 7_214_202_880  # VQ report, projections only

# G0 artifact
G0_PAYLOAD = 14_297_694_680
G0_BPW = 8.0 * G0_PAYLOAD / N
G0_UPLOADED = 14_297_675_776
G0_GEMV = 13_611_663_360
G0_CODES = 12_810_977_280
G0_SCALES = 800_686_080
G0_EMBED_TABLE = 675_430_440
G0_EMBED_ROW = 2_720
G0_F32_RESIDENT = 10_582_016
G0_ORPHAN_F32BIN = 10_584_840
G0_HEADERS = 18_904
ACTIVE_BUDGET = 13_622_264_240
GEOM_ACTIVE = 13_618_141_856

# Bandwidth MEASURED (HONEST_ROOF). 411.51 is REFUTED and unused.
BW_SINGLE_ADDR = 699.5736545106142
BW_SINGLE_DECODE = 683.7970139656385
BW_SINGLE_FULL = 666.6814921907636
BW_TILED_ADDR = 591.1317979446468
BW_CATALOG_ADDR = 530.6544688491846
BW_CATALOG_FULL = 505.8100047843556
BW_SEALED = 639.2522348492898
BW_UNIQUE_ONCE_13G = 375.6517695934827
BW_REFUTED_411 = 411.51  # do not use

SINGLE_ADDR_NS = 19_457_084
SINGLE_FULL_NS = 20_417_041  # 13611663360 / 666.6814921907636
TILED_ADDR_NS = 22_988_750
TILED_BYTES = 13_589_381_120
TILED_ORGANS = 287
CATALOG_ADDR_NS = 25_650_709
CATALOG_DISPS = 401
SEALED_ADDR_NS = 21_293_102.5

# Walls
LIVE_TOKEN_NS = 39_326_090
LIVE_TPS = 1e9 / LIVE_TOKEN_NS
G024_WALL = 35_227_917
G024_GPU = 33_912_333
G024_ENCODE = 919_250
G024_SUBMIT = 12_084
G024_SYNC = 384_250
G024_RESIDUAL = 341_925
AUTH_WALL = 38_216_792
AUTH_GPU = 36_987_458
ICB_WALL = 36_683_916
ICB_ENCODE = 90_981
Q3MLP_WALL = 148_588_917
Q3MLP_GPU = 146_963_124
Q3MLP_BPW = 3.6138111608720234
Q3_MSE_BPW = 4.133684715546539  # packed q4-mse-g128-v1, better screen than G0

# G024 components
DELTANET = 3_732_794.93
GQA = 2_443_470.71
NORM = 2_367_415.00
DECODE_RECON = 1_808_227.35
SWIGLU = 1_004_197.53
KV = 537_665.00
TERM = 383_534.95
CEREMONY = G024_ENCODE + G024_SUBMIT + G024_SYNC + G024_RESIDUAL  # 1,657,509

# Isolated families MEASURED
ISO_MLP = 15_853_666
ISO_DN = 5_560_749
ISO_GQA = 1_817_416
ISO_LM = 1_017_458
ISO_GEMV = ISO_MLP + ISO_DN + ISO_GQA + ISO_LM
ISO_EMBED = 4_999
ISO_ARGMAX = 335_499
ISO_SILU = 160_958
ISO_MLP_RES = 134_208
ISO_MIX_RES = 118_250
ISO_REARR = 350_999
ISO_BA = 139_374
ISO_GRMS = 1_295_500
ISO_GDELTA = 2_146_166
ISO_ROPE = 1_562_625
ISO_MHA = 666_500
ISO_SIG = 43_625
ISO_IN_NORM = 1_137_250
ISO_POST_NORM = 1_210_874
ISO_FINAL_NORM = 19_291
ISO_REC = 467_374
ISO_CONV = 19_000
ISO_GQA_K = 24_375
ISO_GQA_V = 26_916

# Probe fracs MEASURED
ADDR_FRAC = dict(mlp=0.8716916907999576, dn=0.9050872853807689,
                 gqa=0.8302650920366828, lm=0.9157069126980056)

# Bytes per class (geometry Q4)
MLP_B = 9_091_153_920
DN_B = 2_953_789_440
GQA_B = 891_289_600
LM_B = 675_430_400

# Dispatch
DISP_TOKEN = 964
DISP_GEMV = 401
DISP_MLP = 192
DISP_DN = 144
DISP_GQA = 64
DISP_LM = 1

# Codebook MEASURED. 460.041 µs = 460_041 ns. Do not treat µs as ms.
CB_EL = 14336 * 4096
CB_1S_NS = 460_041
CB_4S_NS = 672_625
# recon HGRAVS
HGRAVS_DOWN_NS = 71_458
F32_DOWN_NS = 7_083
GATE_F32_NS = 15_125

# State
REC_B = 150_994_944
CONV_B = 5_898_240
GQA_SLOT = 16 * 2 * 4 * 256 * 4  # 131072 write
SEQ_LEDGER = 19
WS_SEQ128 = 175_361_796
RSS_G0 = 15_511_666_688
PHYS_G0 = 15_385_816_488
ORGANISM_GIB = 17.9
UNIFIED = 103_079_215_104

# q3mlp candidate GEMV bytes (scale-fetch): 11,472,568,320
Q3MLP_GEMV_B = 11_472_568_320

# mixed / sub15
MIXED2P0_BPW = 2.0855934079220506
MIXED2P0_MLP_BPW = 0.8480504639008466
SUB15_BPW = 1.2910781930062503
SUB15_BYTES = 4_340_604_637
ENTROPY_H = 3.478937682977414
SHANNON_COMPLETE = 3.732

# serial vs vi
SERIAL_GDELTA = 11_485_249
VI_GDELTA = 2_146_166
SERIAL_GPU = 42_734_499
VI_GPU_A = 33_449_499

GB = 1e9


def ns_from_bytes(nbytes: float, gbps: float) -> float:
    return nbytes / (gbps * GB) * 1e9


def bpw_of(nbytes: float) -> float:
    return 8.0 * nbytes / N


def rel_err(pred: float, meas: float) -> float:
    return (pred - meas) / meas


def pct(x: float) -> str:
    return f"{100.0 * x:+.2f}%"


print("=" * 72)
print("G1 COST MODEL — arithmetic run")
print("=" * 72)

# ---------------------------------------------------------------------------
# 0. Identities
# ---------------------------------------------------------------------------
print("\n## 0 IDENTITIES")
print(f"N                          {N}")
print(f"G0_BPW                     {G0_BPW:.15f}  MEASURED identity 8*payload/N")
print(f"G0_BPW match contract      {G0_BPW == 4.252735126866492}")
print(f"G0_GEMV codes+scales       {G0_CODES}+{G0_SCALES}={G0_CODES+G0_SCALES}  vs {G0_GEMV}")
print(f"scale/gemv                 {G0_SCALES/G0_GEMV:.12f}  = 2/34 {2/34:.12f}")
print(f"LIVE_TPS                   {LIVE_TPS:.10f}  DERIVED 1e9/39326090")
print(f"CEREMONY G024              {CEREMONY}  DERIVED encode+submit+sync+residual")
print(f"contract ceremony cited    1653000  (rounded)")
print(f"addr/G024_wall             {SEALED_ADDR_NS/G024_WALL:.10f}  cited 60.44%")
print(f"addr/LIVE_wall             {SEALED_ADDR_NS/LIVE_TOKEN_NS:.10f}  NOT 60.44")
print(f"411.51 REFUTED unused      {BW_REFUTED_411}")

# FLOPs
flops_mlp = 64 * 3 * 2 * 17408 * 5120
flops_dn_qkvz = 48 * 2 * 16384 * 5120
flops_dn_ba = 48 * 2 * 96 * 5120
flops_dn_out = 48 * 2 * 5120 * 6144
flops_gqa_q = 16 * 2 * 12288 * 5120
flops_gqa_kv = 16 * 2 * 2 * 1024 * 5120
flops_gqa_o = 16 * 2 * 5120 * 6144
flops_lm = 2 * 248320 * 5120
flops_gemv = (flops_mlp + flops_dn_qkvz + flops_dn_ba + flops_dn_out
              + flops_gqa_q + flops_gqa_kv + flops_gqa_o + flops_lm)
print(f"dense-equivalent GEMV FLOP {flops_gemv}  DERIVED 2*rows*cols")
print(f"  mlp                      {flops_mlp}")
print(f"  dn                       {flops_dn_qkvz+flops_dn_ba+flops_dn_out}")
print(f"  gqa                      {flops_gqa_q+flops_gqa_kv+flops_gqa_o}")
print(f"  lm_head                  {flops_lm}")

# State
kv_read_19 = 16 * 2 * SEQ_LEDGER * 4 * 256 * 4
state_rw_19 = 2 * (REC_B + CONV_B) + GQA_SLOT + kv_read_19
print(f"state R+W seq=19           {state_rw_19}  DERIVED (ledger 316407808)")
print(f"workspace seq=128          {WS_SEQ128}")
print(f"resident G0 uploaded       {G0_UPLOADED}")
print(f"RSS G0 receipt             {RSS_G0}")
print(f"unified                    {UNIFIED}")
print(f"G0 fraction of 96GiB       {G0_UPLOADED/UNIFIED:.6f}")

# ---------------------------------------------------------------------------
# 1. FIT
# ---------------------------------------------------------------------------
print("\n## 1 FIT SET (not used as predictions)")
print(f"FIT BW_SINGLE_ADDR         {BW_SINGLE_ADDR} GB/s  MEASURED 1-dispatch 13.612GB addr_probe")
print(f"FIT BW_SINGLE_FULL         {BW_SINGLE_FULL} GB/s  MEASURED same topology full kernel")
alu_tax_component = 1.0 - BW_SINGLE_FULL / BW_SINGLE_ADDR  # 4.70% of addr; receipt language
alu_tax_token = DECODE_RECON / SEALED_ADDR_NS  # 8.492%; includes input+FMA probe split
print(f"FIT ALU tax component      {alu_tax_component:.6f}  DERIVED 1-full/addr (4.70%)")
print(f"FIT ALU tax TOKEN decode   {alu_tax_token:.6f}  MEASURED 1.808ms/21.293ms (use this)")
alu_tax = alu_tax_token
print(f"FIT SINGLE_ADDR_NS         {SINGLE_ADDR_NS}")
print(f"FIT TILED_ADDR_NS          {TILED_ADDR_NS}  {TILED_ORGANS} identical 17408x5120")
print(f"FIT TILED_BYTES            {TILED_BYTES}")

# Corrected per-dispatch tax: tiled minus the stream time of tiled bytes at single-addr BW
tiled_stream = ns_from_bytes(TILED_BYTES, BW_SINGLE_ADDR)
t_disp = (TILED_ADDR_NS - tiled_stream) / (TILED_ORGANS - 1)
print(f"FIT tiled_stream @single   {tiled_stream:.3f} ns")
print(f"FIT t_disp same-shape      {t_disp:.6f} ns/extra-dispatch  DERIVED")
print(f"FIT encode/dispatch        {G024_ENCODE/DISP_TOKEN:.6f} ns  MEASURED 919250/964")
print(f"FIT encoder-gap/dispatch   {336926/DISP_TOKEN:.6f} ns  MEASURED residual/964")
print(f"FIT n_sync                 1  MEASURED one waitUntilCompleted")
print(f"FIT non-GEMV isolated sum  used as genome-fixed remainder (see §3)")

t_enc = G024_ENCODE / DISP_TOKEN
t_gap = 336926 / DISP_TOKEN

# Non-GEMV isolated (same shaders on any packed-Qn token that keeps the mixer)
non_gemv = (
    ISO_IN_NORM + ISO_POST_NORM + ISO_FINAL_NORM
    + ISO_SILU + ISO_MLP_RES
    + ISO_MIX_RES
    + ISO_REARR + ISO_BA + ISO_GRMS + ISO_GDELTA
    + ISO_ROPE + ISO_MHA + ISO_SIG
    + ISO_ARGMAX + ISO_EMBED
    + ISO_REC + ISO_CONV + ISO_GQA_K + ISO_GQA_V
)
print(f"FIT non-GEMV isolated      {non_gemv} ns  APPLIED (q3 lane 9863783; recompute)")

# ---------------------------------------------------------------------------
# 2. HOLD-OUT predictions
# ---------------------------------------------------------------------------
print("\n## 2 HOLD-OUT (never in the fit)")

# H1 catalog addr
h1_pred = ns_from_bytes(G0_GEMV, BW_SINGLE_ADDR) + (CATALOG_DISPS - 1) * t_disp
h1_meas = CATALOG_ADDR_NS
print(f"H1 catalog addr  pred={h1_pred:.1f} meas={h1_meas} err={pct(rel_err(h1_pred,h1_meas))}")
print(f"   residual attributed to shape-mix (ba=96, k/v=1024)  DERIVED")

# H2 isolated family FULL times: bytes/BW_FULL + n*t_disp
def fam_full(nbytes, ndisp):
    return ns_from_bytes(nbytes, BW_SINGLE_FULL) + ndisp * t_disp

h2 = [
    ("mlp",  fam_full(MLP_B, DISP_MLP), ISO_MLP),
    ("dn",   fam_full(DN_B, DISP_DN), ISO_DN),
    ("gqa",  fam_full(GQA_B, DISP_GQA), ISO_GQA),
    ("lm",   fam_full(LM_B, DISP_LM), ISO_LM),
]
print("H2 isolated FULL @ single_full + t_disp:")
for name, pred, meas in h2:
    print(f"   {name:4s} pred={pred:.1f} meas={meas} err={pct(rel_err(pred,meas))}")
h2_sum_pred = sum(p for _, p, _ in h2)
print(f"   SUM  pred={h2_sum_pred:.1f} meas={ISO_GEMV} err={pct(rel_err(h2_sum_pred,ISO_GEMV))}")

# H3 sealed addressing from single+401*t_disp  (production is class-isolated, not catalog)
h3_pred = ns_from_bytes(G0_GEMV, BW_SINGLE_ADDR) + (DISP_GEMV - 1) * t_disp
h3_meas = SEALED_ADDR_NS
print(f"H3 sealed addr as 401-disp catalog-like")
print(f"   pred={h3_pred:.1f} meas={h3_meas} err={pct(rel_err(h3_pred,h3_meas))}")
print(f"   (expected miss: sealed is isolated-class, 639.25 not 530.65)")

# H3b: sealed from single with NO dispatch (lower bound)
h3b = ns_from_bytes(G0_GEMV, BW_SINGLE_ADDR)
print(f"H3b sealed as single-addr roof")
print(f"   pred={h3b:.1f} meas={h3_meas} err={pct(rel_err(h3b,h3_meas))}")

# H4 unique_once vs Q4 (qualitative: different kernel must be slower)
h4_pred_if_q4 = ns_from_bytes(G0_GEMV, BW_SINGLE_ADDR)
h4_meas = ns_from_bytes(G0_GEMV, BW_UNIQUE_ONCE_13G)
print(f"H4 unique_once 13.6GB  meas={h4_meas:.1f}  Q4-single would be {h4_pred_if_q4:.1f}")
print(f"   unique_once/Q4 = {h4_meas/h4_pred_if_q4:.4f}x  MEASURED kernel gap")

# H5 q3mlp GEMV from SOURCE trip×TPR (not fitted to 137ms)
# bits=3 192 MLP: trips 2x, TPR 1/2 → 4x isolated MLP
# bits=4 305 mixer+lm: trips 16x, TPR 1/2 → 8x isolated mixer+lm
iso_mixer_lm = ISO_DN + ISO_GQA + ISO_LM
h5_gemv = 4.0 * ISO_MLP + 8.0 * iso_mixer_lm
h5_wall = h5_gemv + non_gemv + (Q3MLP_WALL - Q3MLP_GPU)  # use measured q3 ceremony
# but wait-gpu/encode on q3 is MEASURED 1,625,793 — using it is mildly in-sample for ceremony only
q3_ceremony = Q3MLP_WALL - Q3MLP_GPU
h5_wall = h5_gemv + non_gemv + q3_ceremony
h5_gemv_meas = Q3MLP_GPU - non_gemv  # inferred
print(f"H5 q3mlp GEMV  SOURCE 4x MLP + 8x mixer+lm")
print(f"   pred_gemv={h5_gemv:.1f} inferred_meas={h5_gemv_meas:.1f} err={pct(rel_err(h5_gemv,h5_gemv_meas))}")
print(f"   pred_wall={h5_wall:.1f} meas={Q3MLP_WALL} err={pct(rel_err(h5_wall,Q3MLP_WALL))}")
print(f"   q3 effective GB/s inferred {Q3MLP_GEMV_B / (h5_gemv_meas/1e9) / GB:.4f}")

# H5b: if we wrongly scaled q3mlp by BPW (the check that must fail)
h5b = LIVE_TOKEN_NS * (Q3MLP_BPW / G0_BPW)
print(f"H5b NAIVE bpw-scale live  pred={h5b:.1f} meas={Q3MLP_WALL} err={pct(rel_err(h5b,Q3MLP_WALL))}")
print(f"   (this check MUST fail; kernel class changed)")

# H6 4-stage codebook from 1-stage * stages
h6_pred = CB_1S_NS * 4
h6_meas = CB_4S_NS
print(f"H6 RVQ-4stage = 1stage*4  pred={h6_pred:.0f} meas={h6_meas} err={pct(rel_err(h6_pred,h6_meas))}")
print(f"   linear-stage model FAILS; both sit in 460-673 µs gather class")

# H7 codebook vs Q4 sequential of SAME 14336x4096 geometry
cb_q4_bytes = CB_EL * 4.25 / 8.0
h7_q4 = ns_from_bytes(cb_q4_bytes, BW_SEALED)
print(f"H7 14336x4096 Q4 @sealed  pred={h7_q4:.1f} ns")
print(f"   codebook 1-stage      meas={CB_1S_NS} ns  ratio={CB_1S_NS/h7_q4:.3f}x")
print(f"   codebook 4-stage      meas={CB_4S_NS} ns  ratio={CB_4S_NS/h7_q4:.3f}x")

# H8 HGRAVS two-small-GEMV vs measured recon
h8_pred = 2 * F32_DOWN_NS
h8_meas = HGRAVS_DOWN_NS
print(f"H8 HGRAVS as 2x f32-disc  pred={h8_pred} meas={h8_meas} err={pct(rel_err(h8_pred,h8_meas))}")
print(f"   naive two-GEMV FAILS; simd3 occupancy {h8_meas/F32_DOWN_NS:.3f}x vs f32")

# H9 ICB encode: if encoder-create is the 953 ns and ICB deletes it
# leftover ~ set_bytes of 3 u32s. Predict encode ~ 100 µs class.
# We do NOT have a first-principles number; compare measured 90981 vs 964*94.4
# (ICB is a hold-out for "ceremony ∝ encoder count")
h9_pred_if_linear = G024_ENCODE * (1 / 964)  # one encoder
# actually ICB is 1 executeCommandsInBuffer; encode should collapse toward
# a small constant. Measured 90981 vs G024 919250.
print(f"H9 ICB encode  meas={ICB_ENCODE}  G024 encode={G024_ENCODE}  ratio={ICB_ENCODE/G024_ENCODE:.4f}")
print(f"   encode collapsed  {G024_ENCODE-ICB_ENCODE} ns  MEASURED later genome")
print(f"   ICB wall {ICB_WALL} vs authority {AUTH_WALL}  delta={AUTH_WALL-ICB_WALL}")

# H10 serial vs vi gated_delta (genome, not bytes)
print(f"H10 gated_delta serial={SERIAL_GDELTA} vi={VI_GDELTA} ratio={SERIAL_GDELTA/VI_GDELTA:.3f}x")
print(f"    production GPU serial={SERIAL_GPU} vi={VI_GPU_A} delta={SERIAL_GPU-VI_GPU_A}")

# H11 live vs G024 (definition + dirty session; model does not claim to predict dirt)
print(f"H11 live-G024  {LIVE_TOKEN_NS-G024_WALL} ns  ({100*(LIVE_TOKEN_NS-G024_WALL)/G024_WALL:.2f}%)")
print(f"    live-authority {LIVE_TOKEN_NS-AUTH_WALL} ns")
print(f"    not a kernel prediction; dirty-box + decode-phase-mean vs encode+submit+wait")

# H12 reconstruction-is-free: Q4 gate vs f32 gate at tpr64 (COMPONENT)
print(f"H12 recon-free gate tpr64  f32={GATE_F32_NS} (Q4 ~15125-15500, excess 0)")

print("\n## 2b HOLD-OUT SUMMARY")
rows = [
    ("H1 catalog addr", h1_pred, float(h1_meas)),
    ("H2 mlp isolated", h2[0][1], float(h2[0][2])),
    ("H2 dn isolated", h2[1][1], float(h2[1][2])),
    ("H2 gqa isolated", h2[2][1], float(h2[2][2])),
    ("H2 lm isolated", h2[3][1], float(h2[3][2])),
    ("H2 SUM isolated", h2_sum_pred, float(ISO_GEMV)),
    ("H3 sealed as catalog", h3_pred, float(h3_meas)),
    ("H3b sealed as single", h3b, float(h3_meas)),
    ("H5 q3mlp GEMV", h5_gemv, h5_gemv_meas),
    ("H5 q3mlp wall", h5_wall, float(Q3MLP_WALL)),
    ("H5b naive BPW scale", h5b, float(Q3MLP_WALL)),
    ("H6 4stage=1s*4", float(h6_pred), float(h6_meas)),
    ("H8 HGRAVS=2*f32", float(h8_pred), float(h8_meas)),
]
print(f"{'id':<24} {'pred':>14} {'meas':>14} {'rel_err':>10}")
for name, pred, meas in rows:
    print(f"{name:<24} {pred:14.1f} {meas:14.1f} {pct(rel_err(pred,meas)):>10}")

# ---------------------------------------------------------------------------
# 3. MODEL
# ---------------------------------------------------------------------------
print("\n## 3 MODEL")
print("""
TOKEN_NS = t_gemv(kernel_class, active_bytes, n_gemv, n_streams)
         + t_nongemv_fixed          # 9.86 ms isolated mixer/norm/silu/kv/argmax
         + t_ceremony(n_disp, n_sync)
         + t_session_def            # 0 on G024 def; +4.098 ms live-vs-G024 ESTIMATED dirt

t_gemv:
  REGISTER_SEQ (geo_tpr64 Qn/binary/lattice):
      t_addr  = active_bytes / 639.2522348492898e9   # sealed production regime
      t_dec   = alu_tax * t_addr                      # 4.70% at tpr64
      t_disp  = 0  (already inside sealed 639.25)
  ISSUE_NARROW (HGRAVU simd 1-wide / simd3):
      t_gemv  = 4*ISO_MLP + 8*ISO_mixer_lm   if same 192+305 split
              or active_bytes / 83.68e9      (q3mlp inferred class BW)
  CODEBOOK_GATHER:
      t_gemv  = (n_elements / 58720256) * 460.041e-3 s
  TWO_STAGE_SIMD3:
      t_gemv  >= 10.09 * t_register_of_materialized   (COMPONENT, not unique-once)
      unique-once lower bound = factor_bytes/639.25 + n_factor_disp * t_disp
  EXPAND_THEN_Q4:
      t_gemv  = t_expand + t_REGISTER_SEQ(Q4 bytes)   # strictly worse than Q4

t_ceremony = n_disp * 953.63 + n_sync * 384250 + 12084 + n_disp * 349.51
             (G024); ICB replaces first term with ~91 µs.

411.51 GB/s is REFUTED and is not a term.
""")

SESSION_LIVE = LIVE_TOKEN_NS - G024_WALL  # 4,098,173

def ceremony(n_disp, icb=False):
    if icb:
        return ICB_ENCODE + G024_SUBMIT + 561_994 + 0  # ICB wait-gpu rose; residual gone
    return n_disp * t_enc + G024_SUBMIT + G024_SYNC + n_disp * t_gap


def register_seq(active_bytes, n_disp=DISP_GEMV, icb=False, wall="g024"):
    t_addr = ns_from_bytes(active_bytes, BW_SEALED)
    t_dec = alu_tax * t_addr
    t_ng = non_gemv
    t_cer = ceremony(DISP_TOKEN if n_disp else DISP_TOKEN, icb=icb)
    # When only GEMV bytes change, non-GEMV + ceremony stay; use G024 remainder
    # more honestly: t_addr + t_dec + (G024_WALL - SEALED_ADDR_NS - DECODE_RECON)
    remainder = G024_WALL - SEALED_ADDR_NS - DECODE_RECON
    tot_g024 = t_addr + t_dec + remainder
    tot_live = tot_g024 + SESSION_LIVE
    return dict(t_addr=t_addr, t_dec=t_dec, remainder=remainder,
                token_ns_g024=tot_g024, token_ns_live=tot_live,
                tps_g024=1e9/tot_g024, tps_live=1e9/tot_live)


def score_register(name, payload_bytes, active_bytes, n_disp=401, n_sync=1,
                   extra_disp=0, note=""):
    r = register_seq(active_bytes)
    art_bpw = bpw_of(payload_bytes)
    act_bpw = 8.0 * active_bytes / N
    print(f"\n[{name}]")
    print(f"  artifact_bytes           {payload_bytes}  complete_bpw={art_bpw:.12f}")
    print(f"  active_bytes/token       {active_bytes}  active_bpw={act_bpw:.12f}")
    print(f"  dram_traffic_weights     {active_bytes}  (D_w≈A_w this class)")
    print(f"  t_addr @639.25           {r['t_addr']:.1f}")
    print(f"  t_decode 4.70%           {r['t_dec']:.1f}")
    print(f"  t_nongemv+cer G024 rest  {r['remainder']:.1f}")
    print(f"  TOKEN_NS G024-def        {r['token_ns_g024']:.1f}  TPS={r['tps_g024']:.4f}")
    print(f"  TOKEN_NS live-def        {r['token_ns_live']:.1f}  TPS={r['tps_live']:.4f}")
    print(f"  kernel_launches          {DISP_TOKEN + extra_disp}")
    print(f"  gemv_launches            {n_disp}")
    print(f"  syncs                    {n_sync}")
    if note:
        print(f"  note                     {note}")
    return r


print("\n## 4 G0 SCORECARD (model applied to incumbent; should recover)")
g0 = score_register("G0 uniform-q4 geo_tpr64", G0_PAYLOAD, G0_GEMV,
                    note="fit-adjacent: sealed BW is this token's addressing")
print(f"  recover G024 addr        pred={g0['t_addr']:.1f} meas={SEALED_ADDR_NS} err={pct(rel_err(g0['t_addr'],SEALED_ADDR_NS))}")
print(f"  recover G024 wall        pred={g0['token_ns_g024']:.1f} meas={G024_WALL} err={pct(rel_err(g0['token_ns_g024'],G024_WALL))}")
print(f"  recover live wall        pred={g0['token_ns_live']:.1f} meas={LIVE_TOKEN_NS} err={pct(rel_err(g0['token_ns_live'],LIVE_TOKEN_NS))}")
print(f"  compute FLOP             {flops_gemv} dense-eq + mixer ALU (untimed as FLOP)")
print(f"  decode compute           {DECODE_RECON} ns MEASURED (addr vs decode probe)")
print(f"  generation compute       FMA remainder in GEMV classes (~4.7% with decode)")
print(f"  resident RAM             uploaded {G0_UPLOADED} + ws {WS_SEQ128} + RSS {RSS_G0}")
print(f"  KV interaction           rec+conv R+W {2*(REC_B+CONV_B)} + GQA w {GQA_SLOT} + GQA r(seq)")
print(f"  KV ns                    {KV} MEASURED seq≈19")

# ICB G0
print("\n[G0 + ICB] ceremony cut, same bytes")
icb_g024 = G024_WALL - G024_ENCODE + ICB_ENCODE + (561_994 - G024_SYNC)
# ICB complete-wall was 36.684 vs authority 38.217, not vs G024
print(f"  TOKEN_NS authority-def   {ICB_WALL} MEASURED later genome")
print(f"  vs authority             delta={ICB_WALL-AUTH_WALL} ({100*(ICB_WALL-AUTH_WALL)/AUTH_WALL:.2f}%)")
print(f"  launches still           964 ICB commands, 1 executeCommandsInBuffer, 1 wait")

# ---------------------------------------------------------------------------
# 5. Mechanisms
# ---------------------------------------------------------------------------
print("\n## 5 MECHANISMS")

# 5.1 Register-seq density ladder (bytes scale, kernel held)
print("\n### 5.1 REGISTER_SEQ density ladder (kernel held at geo_tpr64, bytes scale)")
for label, bpw in [
    ("G0 4.2527", G0_BPW),
    ("q4-mse 4.1337", Q3_MSE_BPW),
    ("Shannon Q4-index 3.732", SHANNON_COMPLETE),
    ("uniform Q3 3.25+scales+small", 3.250 + 0.002735),  # approx body+header like G0 extra
    ("mixed-2p0 2.086 (if tpr64 consume)", MIXED2P0_BPW),
    ("G1 1.5 target", 1.5),
    ("sub15 1.291 (if tpr64 consume)", SUB15_BPW),
    ("0.7 claimed", 0.7),
]:
    # active ≈ gemv payload scales with (bpw / G0_BPW) * G0_GEMV
    # more honest: complete bytes = bpw * N / 8; active ≈ complete - embed_table + embed_row
    complete_b = bpw * N / 8.0
    # embed table still resident; if tables stay Q4, embed table is 675430440
    # For a uniform scale assume active_gemv = complete_b - embed_table + small corrections
    # G0: complete 14297694680 - 675430440 + 2720 = 13622302960 ≈ ACTIVE_BUDGET
    active = complete_b - G0_EMBED_TABLE + G0_EMBED_ROW
    r = register_seq(active)
    print(f"  {label:<40} complete_b={complete_b:,.0f} active={active:,.0f} "
          f"addr={r['t_addr']/1e6:6.2f}ms  g024={r['token_ns_g024']/1e6:6.2f}ms "
          f"live={r['token_ns_live']/1e6:6.2f}ms  tps_live={r['tps_live']:6.2f}")

print("  NOTE: 1.5 BPW live-def still ~25.3 ms (39.6 TPS) because 18.0 ms remainder does not scale.")
print("  10 ms / 100 TPS is NOT implied. Remainder today 18.03 ms live / 13.93 ms G024.")

# 5.2 ISSUE_NARROW q3mlp (MEASURED + model)
print("\n### 5.2 ISSUE_NARROW — HGRAVU simd/simd3 (q3mlp MEASURED)")
print(f"  artifact complete_bpw    {Q3MLP_BPW}  MEASURED")
print(f"  active GEMV bytes        {Q3MLP_GEMV_B}  DERIVED (11.47 GB; MLP q3 + attn q4)")
print(f"  TOKEN_NS                 {Q3MLP_WALL}  MEASURED")
print(f"  GPU                      {Q3MLP_GPU}  MEASURED")
print(f"  inferred GEMV            {h5_gemv_meas:.1f}")
print(f"  effective BW             {Q3MLP_GEMV_B / (h5_gemv_meas/1e9) / GB:.3f} GB/s")
print(f"  vs G0 live               {Q3MLP_WALL/LIVE_TOKEN_NS:.4f}x")
print(f"  vs naive byte scale      {Q3MLP_WALL/h5b:.4f}x slower than BPW would imply")
print(f"  launches                 964 + 96 fuse  ESTIMATED extra 96")
print(f"  VERDICT                  DEAD as a speed path. Coherent at 3.614 BPW but 3.78x slower.")
print(f"  REOPEN_IF                same bytes consumed by geo_tpr64 (bound-7), complete-token A/B ≤ G0.")

# 5.3 Codebook / shared dictionary
print("\n### 5.3 CODEBOOK_GATHER — shared dictionary (PQ / RVQ / gravity VQ)")
ns_per_el = CB_1S_NS / CB_EL
print(f"  ns/element 1-stage       {ns_per_el:.6f}  MEASURED 460041e3/{CB_EL}")
for label, el in [
    ("one 14336x4096 FFN (meas)", CB_EL),
    ("all Qwen38 GEMV no embed", E_GEMV_NO_EMBED),
    ("attention projections only", E_ATTN_PROJ),
    ("MLP only", E_MLP),
    ("lm_head only", 1_271_398_400),
    ("one gate 17408x5120", 17408 * 5120),
]:
    t = el * ns_per_el
    print(f"  {label:<32} el={el:>12}  t_gemv={t/1e6:8.3f} ms  PROJECTED from 460.041µs")

attn_pq = E_ATTN_PROJ * ns_per_el
all_pq = E_GEMV_NO_EMBED * ns_per_el
# complete token: gather GEMV + non-GEMV + ceremony
pq_attn_wall = attn_pq + non_gemv + ceremony(DISP_TOKEN)
pq_all_wall = all_pq + non_gemv + ceremony(DISP_TOKEN)
print(f"  attention-only token     {pq_attn_wall/1e6:.3f} ms PROJECTED  (VQ report 56.52 ms attn GEMV)")
print(f"  all-GEMV token           {pq_all_wall/1e6:.3f} ms PROJECTED")
print(f"  vs live G0               attn {pq_attn_wall/LIVE_TOKEN_NS:.2f}x  all {pq_all_wall/LIVE_TOKEN_NS:.2f}x")
print(f"  claimed BPW examples     PQ d=4 K=256 attn → complete 1.4799 (VQ §10)")
print(f"                           gravity D=32 K=256 → complete 1.0095")
print(f"                           RVQ S=3 d=16 K=256 → complete 1.3462")
print(f"  artifact small ≠ token cheap: gather is per-element, not per-artifact-byte")
print(f"  VERDICT                  DEAD on complete token at any useful codebook rate.")
print(f"  REOPEN_IF                new kernel ≤ Q4 ns on 10240x5120 AND quality ≥ 0.99.")

# 4-stage
print(f"  RVQ 4-stage 14336x4096   {CB_4S_NS/1e3:.3f} µs MEASURED (1.46x 1-stage, not 4x)")
rvq4_attn = (E_ATTN_PROJ / CB_EL) * CB_4S_NS
print(f"  RVQ4 attention PROJECTED {rvq4_attn/1e6:.3f} ms")

# 5.4 Hierarchical additive streams
print("\n### 5.4 HIERARCHICAL additive streams")
# S register-seq streams: bytes add, decode adds if not fused
print("  (a) fused additive q2q2 g64 at tpr64")
add_bpw = 4.500040929457721  # descent one gate MEASURED
add_complete_b = add_bpw * (E_GEMV_NO_EMBED + E_SMALL) / 8.0 + G0_EMBED_TABLE
# simpler: if all GEMV become 4.50 instead of 4.25
add_active = G0_GEMV * (4.50 / 4.25)
r_add = register_seq(add_active)
print(f"      physical BPW body    4.50 > Q4 4.25  MEASURED one gate")
print(f"      active PROJECTED     {add_active:.0f}")
print(f"      TOKEN_NS live        {r_add['token_ns_live']:.1f}  WORSE than G0")
print(f"      recon excess         0 at tpr64  MEASURED COMPONENT")
print(f"      VERDICT              DEAD as a density path (more bytes, same launch).")

print("  (b) unfused S streams (each a separate DRAM walk / dispatch)")
for S, bpw_each in [(2, 2.25), (3, 1.5), (4, 1.125)]:
    # if each stream is register-seq of bpw_each and they are NOT fused
    active = S * (bpw_each / 4.25) * G0_GEMV
    extra = (S - 1) * DISP_GEMV
    r = register_seq(active)
    # extra dispatches pay encode+gap; GEMV time already in larger active
    extra_cer = extra * (t_enc + t_gap)
    print(f"      S={S} x {bpw_each} BPW unfused  active={active:.0f}  "
          f"addr={r['t_addr']/1e6:.2f}ms  extra_cer={extra_cer/1e6:.2f}ms  "
          f"live≈{(r['token_ns_live']+extra_cer)/1e6:.2f}ms")
print("      VERDICT              unfused multi-stream ≥ one stream of the sum. No free lunch.")
print("      If streams are gathers: S * codebook class → DEAD (see 5.3).")

print("  (c) generator r=64 f16 + quantized residual  (per-tensor SVD, FALSIFIED quality)")
# factor bytes all GEMV at r=64 f16: 2*r*(m+n) per tensor. Use energy report projection.
# Approximate: r(m+n)*2 over classes. Use 2.436e9 captured / still need residual of 91%.
# Cost: 2 extra GEMVs per tensor (L, R) + residual stream.
# r=64 f16 factors for all language GEMV:
# MLP 192 * 64*(17408+5120)*2 = 192*64*22528*2 = 553,648,128 B? 192*2*64*22528
fac_mlp = 192 * 2 * 64 * (17408 + 5120)
fac_dn = 48 * 2 * 64 * ((16384 + 5120) + (96 + 5120) + (5120 + 6144))
fac_gqa = 16 * 2 * 64 * ((12288 + 5120) + 2 * (1024 + 5120) + (5120 + 6144))
fac_tab = 2 * 2 * 64 * (248320 + 5120)
fac = fac_mlp + fac_dn + fac_gqa + fac_tab
print(f"      r=64 f16 factor bytes {fac}  DERIVED")
print(f"      factor BPW            {bpw_of(fac):.6f}")
print(f"      residual still ~91% Frobenius  MEASURED; residual Q4 ≈ 0.91 * 4.25 if naive")
# honest: they measured no pair beats Q4 error at lower total BPW
print(f"      quality                no (r, codec) pair hits Q4 error at lower total BPW  MEASURED")
print(f"      consume if materialize W then GEMV: generate + full GEMV → strictly worse")
print(f"      consume L@(R@x)+res:   2*401 extra GEMVs + residual stream")
print(f"      extra launches         {2*401}")
print(f"      VERDICT                DEAD as constructed (quality + extra streams).")
print(f"      REOPEN_IF              activation-weighted / trained generator with in-register consume.")

print("  (d) hierarchical 3-level: coarse register + mid correction + exact island")
# binary 1.125 + rice 2% + 0.1% exact
# bytes: binary body + residual + island values + indices
# TOKEN: binary sequential (cheap) + CSR irregular + island add
bin_active = G0_GEMV * (1.125 / 4.25)
r_bin = register_seq(bin_active)
print(f"      binary sequential addr {r_bin['t_addr']/1e6:.2f} ms PROJECTED @639.25")
print(f"      + CSR residual 2%      irregular; shipping tile 2048 does not divide K=5120")
print(f"      + island 0.1% bf16     {0.001 * E_GEMV_NO_EMBED * 2 / 1e6:.1f} MB values")
print(f"      live if ONLY binary    {r_bin['token_ns_live']/1e6:.2f} ms PROJECTED")
print(f"      quality                binary+1% islands hold 0.894 vs Q4 0.997  MEASURED")
print(f"      VERDICT                binary sequential is physically viable;")
print(f"                             adding gather-style corrections can kill the win.")
print(f"                             3-level as a TOKEN win is UNMEASURED; quality FALSIFIED at these rates.")

# 5.5 Generated blocks HGRAVS
print("\n### 5.5 GENERATED BLOCKS — HGRAVS r160 b3 L@(R@x)")
# mixed-2p0 downs
down_elems = 64 * 5120 * 17408
fac_elems = 64 * 160 * (5120 + 17408)
fac_bytes_q3 = fac_elems * 3.25 / 8.0
print(f"  64 downs factor elems      {fac_elems}")
print(f"  factor bytes q3            {fac_bytes_q3:.0f}  ({bpw_of(fac_bytes_q3):.6f} complete)")
print(f"  MEASURED down physical     0.131617 BPW on downs (mixed-2p0)")
t_fac_bw = ns_from_bytes(fac_bytes_q3, BW_SEALED)
t_fac_disp = 128 * t_disp
print(f"  unique-once lower bound    {t_fac_bw:.1f} ns + {t_fac_disp:.1f} ns disp = {t_fac_bw+t_fac_disp:.1f}")
q4_down_b = MLP_B / 3
t_q4_down = ns_from_bytes(q4_down_b, BW_SEALED)
print(f"  Q4 downs addr              {t_q4_down:.1f} ns")
print(f"  COMPONENT recon            {HGRAVS_DOWN_NS} vs f32 {F32_DOWN_NS} = {HGRAVS_DOWN_NS/F32_DOWN_NS:.2f}x")
print(f"  if simd3 occupancy scales to unique-once: {10.09 * t_q4_down:.1f} ns  PROJECTED upper")
print(f"  VERDICT                    byte-viable; CURRENT simd3 consume is a token risk.")
print(f"                             mixed-2p0 INCOHERENT (quality). mixed-q4down INCOHERENT.")
print(f"  REOPEN_IF                  factor kernel at geo_tpr64 AND generate-coherent.")

# 5.6 Sparse exact islands
print("\n### 5.6 SPARSE CORRECTIONS — index + exact values")
for frac, scheme_bpw, hold in [
    (0.0, 1.1250, 0.847040),
    (1e-4, 1.1281, 0.862334),
    (1e-3, 1.1527, 0.871833),
    (1e-2, 1.3669, 0.893734),
    (3e-2, 1.8049, 0.914771),
]:
    # if applied to attention only, rest G0 Q4
    attn_b = scheme_bpw * E_ATTN / 8.0
    rest_b = (MLP_B + LM_B)  # Q4
    active = attn_b + rest_b
    r = register_seq(active)
    print(f"  frac={frac:<6} attn_bpw={scheme_bpw:.4f} hold={hold:.4f} "
          f"active={active:.0f} live={r['token_ns_live']/1e6:.2f}ms  (REGISTER_SEQ assume)")
print("  CSR/index consume is NOT register-seq. Shipping tg256 TILE=2048 KILLS K=5120.")
print("  VERDICT                    quality FALSIFIED (need ~1% for weak MLP bar, never Q4-class).")
print("                             index gather is a second CODEBOOK-class risk.")
print("                             DEAD as a G1 density path (quality). Token-viable only if")
print("                             residual is sequential and kernel is K-complete.")

# 5.7 Shared cross-layer dictionary
print("\n### 5.7 SHARED CROSS-LAYER BASIS / DICTIONARY")
print("  adjacent cosine 1e-5..8e-3  MEASURED  → shared rank-256 leaves 69-93% residual")
print("  lookup consume              codebook gather (5.3) or extra R@x")
print("  VERDICT                    DEAD quality on this parent. Token would add a gather.")
print("  REOPEN_IF                  new parent with adj cos mean ≥ 0.05.")

# 5.8 Expand-then-Q4
print("\n### 5.8 EXPAND-TO-Q4-THEN-GENERIC-GEMV")
# expand traffic: write dense W (or Q4) then read it
# worst: materialize f32 = 2x source bytes
exp_f32 = E_GEMV_NO_EMBED * 4
t_exp = ns_from_bytes(exp_f32, BW_SEALED)  # write
t_read = ns_from_bytes(exp_f32, BW_SEALED)  # read as GEMV
print(f"  materialize f32 write      {t_exp/1e6:.2f} ms PROJECTED")
print(f"  then f32 GEMV read         {t_read/1e6:.2f} ms PROJECTED")
print(f"  sum vs G0 addr             {(t_exp+t_read)/SEALED_ADDR_NS:.2f}x just for traffic")
print(f"  VERDICT                    DEAD unless a complete-token A/B shows a net win.")
print(f"  REOPEN_IF                  measured token with expand < G0. Not observed.")

# 5.9 Entropy coding
print("\n### 5.9 ENTROPY CODING of incumbent Q4 indices")
shan_complete_b = SHANNON_COMPLETE * N / 8.0
shan_active = shan_complete_b - G0_EMBED_TABLE + G0_EMBED_ROW
r_sh = register_seq(shan_active)
print(f"  Shannon complete BPW       {SHANNON_COMPLETE}  MEASURED ceiling this quantizer")
print(f"  prize                      0.521 bpw / 1.75 GB")
print(f"  TOKEN if RA-legal          live {r_sh['token_ns_live']/1e6:.2f} ms PROJECTED")
print(f"  vs G0 live                 {r_sh['token_ns_live']/LIVE_TOKEN_NS:.4f}x")
print(f"  VERDICT                    cannot reach 1.5. Random-access-legal codecs recover little.")
print(f"                             sequential ANS/range would break tpr64 gather-free load.")
print(f"  REOPEN_IF                  group-local RA codec that stays register-seq.")

# 5.10 Fusion / persistent / residency
print("\n### 5.10 FUSION / PERSISTENCE / RESIDENCY (not representations)")
print(f"  ICB                        MEASURED −0.744 ms named-fixed on later genome")
print(f"  8-layer f16 megakernel     MEASURED_NEGATIVE (fusion-persistent)")
print(f"  1-TG persist               KILLS")
print(f"  use_resource batching      KILLS as TOKEN_NS primary (≤2 ms, not 21 ms)")
print(f"  skip DRAM via cache        FALSIFIED (13.6 GB >> ~64 MiB cache cliff)")
print(f"  VERDICT                    none of these make a small artifact win;")
print(f"                             none resurrect a gather/issue-bound consume.")

# ---------------------------------------------------------------------------
# 6. DEAD set
# ---------------------------------------------------------------------------
print("\n## 6 DEAD ON COMPLETE TOKEN regardless of artifact size")
print("""
A mechanism is DEAD-TOKEN if its consume path has a per-element or per-dispatch
cost that exceeds G0 even when stored bytes → 0 (or when bytes are already smaller).

1. Direct codebook-lookup GEMV (PQ / RVQ / gravity VQ / shared dictionary lookup)
   Why: 460.041 µs MEASURED on 58.7 M weights; scales with elements, not artifact
   bytes. Attention-only PROJECTED 56.5 ms > live G0 39.3 ms. All-GEMV ~201 ms.
   KILLS. REOPEN_IF new kernel ≤ Q4 ns on real 10240×5120 AND quality ≥ 0.99.

2. Multi-stage additive GATHER (RVQ S>1)
   Why: 672.625 µs MEASURED 4-stage; still gather-class. Linear-S model fails
   (1.46x not 4x) but both sit 9–14× above Q4 sequential of the same matrix.
   KILLS. REOPEN_IF fused register-only residual (no gather) at tpr64.

3. HGRAVU 1-wide / simd3 consume of a dense Uniform body (q3mlp kernel class)
   Why: MEASURED 148.6 ms at 3.614 BPW = 3.78× live G0. Issue-bound at ~84 GB/s.
   Further bit cuts do not help until the launch is geo_tpr64.
   KILLS as a speed path. REOPEN_IF geo_tpr64 bound-7 A/B ≤ G0.

4. Expand-to-float/Q4 then generic GEMV
   Why: pays materialize traffic + the Q4/f32 stream. G0 already in-register.
   KILLS. REOPEN_IF complete-token A/B net win (not observed).

5. Materialize generated W then GEMV
   Why: generate + full stream. Strictly worse than consuming L@(R@x) or Q4.
   KILLS. REOPEN_IF never; use two-stage in-register instead.

6. Cross-token weight residency / reconstruction cache
   Why: 13.6 GB unique-once vs ~64 MiB cache cliff. FALSIFIED.
   KILLS TOKEN lever. REOPEN_IF a representation whose HOT working set ≤ cache.

7. Megakernel / 1-TG persistent layer
   Why: MEASURED_NEGATIVE / KILLS (fusion-persistent). Occupancy death.
   REOPEN_IF inline packed consume AND occupancy ≥ geo_tpr64 AND token A/B.

8. Unfused hierarchical streams whose sum of walks ≥ G0 walk
   Why: S sequential streams of the same mass cost S× (or the sum of bytes).
   A 0.7 BPW artifact split into 6 unfused 0.7-class walks is not 0.7.
   KILLS the 'hide cost in reconstruction' cheat. REOPEN_IF fused in-register.

NOT dead-token (physically viable consume; quality is a separate gate):
  - geo_tpr64 register-seq Qn / binary / lattice / Hadamard (TOKEN ~ bytes/639.25 + 14–18 ms)
  - ICB (small ceremony win)
  - fused tiny-kernel attack on the 14–18 ms remainder (UNMEASURED token)
  - HGRAVS two-stage IF the factor kernel is geo_tpr64 not simd3 (byte-viable; token UNMEASURED)
  - K-complete binary/rice sequential (byte-viable; TILE=2048 bind is currently wrong)
""")

# ---------------------------------------------------------------------------
# 7. Viability at claimed BPW
# ---------------------------------------------------------------------------
print("\n## 7 CLAIMED-BPW SCOREBOARD")
print(f"{'mechanism':<42} {'bpw':>8} {'token_ms':>9} {'vsG0':>7} {'class':<16} {'viable?'}")

def row(mech, bpw, token_ns, kclass, viable):
    print(f"{mech:<42} {bpw:8.3f} {token_ns/1e6:9.2f} {token_ns/LIVE_TOKEN_NS:7.2f}x {kclass:<16} {viable}")

row("G0 HQ30UQ4 geo_tpr64", G0_BPW, LIVE_TOKEN_NS, "REGISTER_SEQ", "YES (incumbent)")
row("q4-mse-g128-v1 (if same kernel)", Q3_MSE_BPW, register_seq((Q3_MSE_BPW*N/8)-G0_EMBED_TABLE+G0_EMBED_ROW)['token_ns_live'], "REGISTER_SEQ", "YES if tpr64")
row("Shannon-coded Q4 indices", SHANNON_COMPLETE, r_sh['token_ns_live'], "REGISTER_SEQ", "YES token / NO to 1.5")
row("q3mlp HGRAVU simd/simd3", Q3MLP_BPW, Q3MLP_WALL, "ISSUE_NARROW", "NO token (YES quality)")
row("mixed-2p0 if tpr64 consume", MIXED2P0_BPW, register_seq((MIXED2P0_BPW*N/8)-G0_EMBED_TABLE+G0_EMBED_ROW)['token_ns_live'], "REGISTER_SEQ", "token YES / quality NO")
row("sub15 if tpr64 consume", SUB15_BPW, register_seq((SUB15_BPW*N/8)-G0_EMBED_TABLE+G0_EMBED_ROW)['token_ns_live'], "REGISTER_SEQ", "token YES / quality NO")
row("G1 1.5 register-seq", 1.5, register_seq((1.5*N/8)-G0_EMBED_TABLE+G0_EMBED_ROW)['token_ns_live'], "REGISTER_SEQ", "token YES / 100TPS NO")
# mixed-2p0 MLP sequential + PQ attention gather
mlp_2p0_b = MIXED2P0_MLP_BPW * E_MLP / 8.0
pq_mixed_token = attn_pq + ns_from_bytes(mlp_2p0_b, BW_SEALED) + non_gemv + ceremony(DISP_TOKEN)
rvq4_token = rvq4_attn + non_gemv + ceremony(DISP_TOKEN)
row("PQ attn d4K256 + mixed MLP", 1.4799, pq_mixed_token, "CODEBOOK+SEQ", "NO token")
row("PQ all GEMV", 2.00, pq_all_wall, "CODEBOOK", "NO token")
row("gravity VQ D32K256 all", 1.0095, pq_all_wall, "CODEBOOK", "NO token")
row("RVQ S=4 attn", 1.50, rvq4_token, "CODEBOOK", "NO token")
row("additive q2q2 fused tpr64", 4.50, r_add['token_ns_live'], "REGISTER_SEQ", "NO (worse bytes)")
row("generator r64 + Q4 res", 4.25 + bpw_of(fac), LIVE_TOKEN_NS + 2*401*(t_enc+t_gap) + ns_from_bytes(fac, BW_SEALED), "2-STAGE+SEQ", "NO (extra streams)")
row("binary+1% islands attn", 1.37, register_seq(1.3669*E_ATTN/8 + MLP_B + LM_B)['token_ns_live'], "SEQ+INDEX", "token maybe / quality NO")
row("HGRAVS downs + Q4 rest", MIXED2P0_BPW, register_seq((MIXED2P0_BPW*N/8)-G0_EMBED_TABLE+G0_EMBED_ROW)['token_ns_live'], "2-STAGE simd3", "bytes YES / quality NO / token RISK")
row("expand-then-Q4", G0_BPW, LIVE_TOKEN_NS + ns_from_bytes(E_GEMV_NO_EMBED*4, BW_SEALED), "EXPAND", "NO")
row("shared cross-layer dict", 0.50, pq_all_wall, "CODEBOOK", "NO token + quality")
row("G0+ICB", G0_BPW, ICB_WALL, "REGISTER_SEQ", "YES small win")
row("0.7 BPW register-seq", 0.7, register_seq((0.7*N/8)-G0_EMBED_TABLE+G0_EMBED_ROW)['token_ns_live'], "REGISTER_SEQ", "token YES / 100TPS NO / quality UNKNOWN")

# remainder math for 100 TPS
print("\n## 8 100 TPS / 10 ms")
for bpw in (4.252735126866492, 3.732, 2.0856, 1.5, 0.7, 0.0):
    active = max(0.0, (bpw * N / 8.0) - G0_EMBED_TABLE + G0_EMBED_ROW)
    t_addr = ns_from_bytes(active, BW_SEALED) if active else 0.0
    rem_g024 = G024_WALL - SEALED_ADDR_NS
    rem_live = LIVE_TOKEN_NS - SEALED_ADDR_NS
    print(f"  bpw={bpw:<8.4f} addr={t_addr/1e6:6.2f}ms  "
          f"+g024_rest { (t_addr+rem_g024)/1e6:6.2f}ms  "
          f"+live_rest {(t_addr+rem_live)/1e6:6.2f}ms  "
          f"need_rest≤{max(0,10e6-t_addr)/1e6:.2f}ms for 10ms")

print("\nDONE")
