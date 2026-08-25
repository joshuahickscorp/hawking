# Coordinate transform probe (N044 / S026 §78)

The 2.25 MLP composition floor was measured in the un-rotated
parameterization. S026 §117: the information floor of a parameterization is
not necessarily the floor of the function. This is the cheap CPU
discriminator that decides whether a GPU reopening of
`QWEN_MLP_ROTATED_TERNARY` is worth it.

Source: `tools/headless/coordinate_transform_probe.py`
Receipt: `receipts/headless/COORDINATE_TRANSFORM_PROBE.json`

## Verdict

`ROTATION_MOVES_BARRIER = false`

The 2.25 floor is coordinate-robust and stays closed. QWEN_MLP_2_25 remains
closed for the un-rotated family (S026 §11). No bounded reopening frontier
is named.

## What was measured

High-sensitivity MLP blocks from N036 (uniform injury, earliest L0
`up_proj`, worst mean `down_proj`): layers 0 and 31, organs
gate/up/down, plus the composed SwiGLU+down MLP. Real
`capture_diverse2` post_attn_norm activations, 512 held-out tokens, fit/hold
from the capture manifest. Parent BF16 streamed one tensor at a time;
`~/noetic/NOETIC_PARENT_A` catalog read-only, not mutated. No GPU, no
second 27B decode, no cargo/Metal.

Function-preserving transforms (T then T^{-1} absorbed):

| transform | role |
|---|---|
| identity | no-op control; must match the un-rotated fit |
| hadamard_b1024 | block-diagonal Walsh-Hadamard, G032 tile, 0 stored bytes |
| pca_orth_b1024 | learned block-orthogonal from fit-activation PCA |
| bad_nonorth_b1024 | non-orthogonal control; must not spuriously help |

Codecs re-fit in each coordinate system: binary g64 (1.25), ternary g64
(~1.58 code / 1.85 packed 5-in-8), q2f g64 (2.25). Gate is held-out
composition (rel_fro / argmax agreement), not weight-space.

## Measured MLP-composition deltas (L0+L31 mean)

| arm | Δrel_fro | identity → rotated | Δargmax |
|---|---|---|---|
| hadamard / binary | -0.0169 | 0.8054 → 0.7885 | +0.0947 |
| pca / binary | -0.0142 | 0.8054 → 0.7912 | +0.0859 |
| hadamard / ternary | -0.0093 | 0.5701 → 0.5608 | +0.0137 |
| pca / ternary | -0.0005 | 0.5701 → 0.5696 | +0.0010 |

Material is Δrel_fro ≤ -0.03 (larger than G032's 0.0082 Q2 hold delta) and
closing the gap to unrotated q2f, or crossing `local_survives` on the MLP.
None of the candidate arms meet it. The bad control explodes (rel_fro
thousands) and does not count.

Hadamard does lift *organ-local* ternary meanabs over the 0.50/0.50 GEMV
bar on several tensors. MLP composition does not follow. That is the
composition ladder (N011), not a reopening.

Absorbed Hadamard/PCA bill 0 runtime bytes (S026 §9, §93).
