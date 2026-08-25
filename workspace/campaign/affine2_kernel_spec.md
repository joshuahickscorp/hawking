# MLX affine-2 group-32 pack layout and GEMV kernel spec

Ground-truth read of the **installed** packages, not GitHub:

| package | version | install root |
|---|---|---|
| `mlx` | **0.32.1** | `~/.grok-vision/lib/python3.12/site-packages/mlx` |
| `mlx_lm` | **0.31.3** | `~/.grok-vision/lib/python3.12/site-packages/mlx_lm` |
| `mlx_metal` | **0.32.1** | ships `mlx/lib/mlx.metallib` |

This is the representation used by the PocketAiHub Qwen3.8-27B-Abliterated-MLX 2-bit artifact (affine, `bits=2`, `group_size=32`, `mode="affine"`): 2-bit packed codes + fp16 scale + fp16 bias per 32-group ≈ **3.0 bpw**. Hawking must reimplement the representation and a native Metal GEMV, not wrap mlx.

No files under `crates/` or `site-packages` were modified.

---

## 1. Call path: mlx_lm → QuantizedLinear → quantized_matmul

`qwen3_5.py` does **not** implement its own quantized linear. Text layers are ordinary `nn.Linear` / `nn.Embedding`:

- Attention: `Qwen3NextAttention.{q,k,v,o}_proj` — `mlx_lm/models/qwen3_next.py` lines 89–108.
- MLP: `Qwen3NextMLP.{gate,up,down}_proj` — `qwen3_next.py` lines 161–169.
- Gated-delta (linear-attn) layers in `qwen3_5.py`: `GatedDeltaNet.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}` — lines 114–128.
- `lm_head`: `TextModel` `nn.Linear(hidden, vocab, bias=False)` — `qwen3_5.py` line 285.

Quantization is applied at load time by `mlx_lm.utils.load_model` (`mlx_lm/utils.py` 348–366):

```python
def _quantize(quantization):
    def class_predicate(p, m):
        if p in config["quantization"]:
            return config["quantization"][p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in weights
    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization.get("mode", "affine"),
        class_predicate=class_predicate,
    )
```

`nn.Linear.to_quantized` (`mlx/nn/layers/linear.py` 72–107) returns `QuantizedLinear.from_linear(...)`.

`QuantizedLinear` (`mlx/nn/layers/quantized.py` 205–307) stores three arrays plus an optional dense output bias:

| member | role |
|---|---|
| `self.weight` | packed codes, `dtype=uint32` |
| `self.scales` | per-group scale, same floating dtype as the original weight |
| `self.biases` | per-group affine bias (the `β` / intercept — **not** the Linear `b`) |
| `self.bias` | optional dense `nn.Linear` bias, added **after** the matmul |

Forward (`quantized.py` 270–283):

```python
def __call__(self, x):
    x = mx.quantized_matmul(
        x,
        self["weight"],
        scales=self["scales"],
        biases=self.get("biases"),
        transpose=True,          # y = x @ W.T
        group_size=self.group_size,
        bits=self.bits,
        mode=self.mode,
    )
    if "bias" in self:
        x = x + self["bias"]
    return x
```

C++ entry point (`mlx/include/mlx/ops.h` 1566–1576):

```cpp
MLX_API array quantized_matmul(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases = std::nullopt,
    bool transpose = true,
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "affine",
    StreamOrDevice s = {});
```

Python stub (`mlx/core/__init__.pyi` 3356–3382) documents the same: one floating scale and bias per `group_size` elements; each code occupies `bits` bits packed in `uint32`.

Defaults if unspecified (`quantized.py` `_defaults_for_mode`, and the mode table in `__init__.pyi` 3422–3431): affine is `group_size=64`, `bits=4`. The 2-bit g32 artifact **must** carry `quantization: {group_size: 32, bits: 2, mode: "affine"}` in `config.json`. Affine **does** support `group_size ∈ {32, 64, 128}` and `bits ∈ {2, 3, 4, 5, 6, 8}`.

C++ primitive: `mlx::core::QuantizedMatmul` (`mlx/include/mlx/primitives.h` 1616–1648) with `QuantizationMode::Affine` (`primitives.h` 155). GPU eval lives only in `libmlx.dylib` (`QuantizedMatmul::eval_gpu` at `0xdfbc28`); there is no `quantized.cpp` in the wheel.

---

## 2. Exact affine2 (bits=2, g32) pack layout

### 2.1 Tensor shapes and dtypes

Let dense `W` have shape `(N, K)` = `(output_dims, input_dims)`, row-major, last axis `K` divisible by `group_size=32`.

After `mx.quantize` / as stored in the safetensors:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `weight` (`w_q`) | `uint32` | `(N, K * bits / 32)` = `(N, K/16)` | packed 2-bit codes |
| `scales` | **same as original `W`** (fp16 or bf16) | `(N, K / 32)` | per-group `s` |
| `biases` | same as `scales` | `(N, K / 32)` | per-group `β` |

Cite:

- Mode table: `__init__.pyi` 3427: affine scale type is “same as input”, bias = yes.
- `QuantizedLinear._extra_repr` (`quantized.py` 262–268): `in_dims = (in_dims * 32) // self.bits` recovers `K` from the packed last dim — i.e. the last dim of `weight` is counted in **uint32 words**, 16 codes per word at 2-bit.
- Common helper (`mlx/include/mlx/backend/common/quantized.h` 4–12) and Metal twin (`quantized.h` 17–26):

```cpp
inline constexpr short get_pack_factor(int bits, int wsize = 8) {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}
inline constexpr short get_bytes_per_pack(int bits, int wsize = 8) {
  bool power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}
```

For `bits=2`:

| `wsize` | `pack_factor` | `bytes_per_pack` | used by |
|---|---|---|---|
| 8 (byte) | 4 codes / byte | 1 | `affine_quantize`, `affine_dequantize`, `QuantizedBlockLoader`, `qdot` |
| 32 (word) | 16 codes / `uint32` | 4 | `qmv_fast_impl` / `qmv_impl` GEMV |

Bytes per packed row = `K * bits / 8` = `K/4`. Uint32s per packed row = `K/16`. Groups per row = `K/32`. **Two uint32 words hold one group of 32 codes.**

Scale/bias are **separate buffers**, not interleaved into the packed words. Index of the group covering column `k` is `g = k / 32`; the pair `(scales[n, g], biases[n, g])` applies to `W[n, 32g : 32g+32)`.

Effective bpw for fp16 scale+bias: `2 + 16/32 + 16/32 = 3.0`.

### 2.2 Bit order inside a uint32 (and inside a byte)

Python API (`__init__.pyi` 3448–3452), written for 4-bit but the same rule is used for every power-of-two width:

> After the above computation, `ŵ_i` fits in `b` bits and is packed in an unsigned 32-bit integer **from the lower to upper bits**. For instance, for 4-bit quantization we fit 8 elements in an unsigned 32 bit integer where the 1st element occupies the 4 least significant bits, the 2nd bits 4-7 etc.

Encoder (`affine_quantize`, `quantized.h` 2683–2688):

```metal
uint8_t val = min(round((w_thread[i] - bias) / scale), n_bins);
output |= val << (bits * (i % pack_factor));
```

Decoder (`affine_dequantize`, `quantized.h` 2769–2781), `bits==2` path:

```metal
uint val = w[offset];          // one byte
for (int i = 0; i < pack_factor; i++) {   // pack_factor = 4
    uint8_t d = (val >> (bits * i)) & 0x03;
    out[i] = scale * d + bias;
}
```

Fused GEMV extract (`qdot` bits==2, `quantized.h` 205–212) is the same layout without an explicit shift: it ANDs the unshifted fields `0x03, 0x0c, 0x30, 0xc0` and compensates by pre-dividing `x` (see §3.2).

**Worked map, one group of 32 codes → two little-endian uint32 words.**

Let `q[0..31] ∈ {0,1,2,3}` be the codes of `W[n, 32g : 32g+32)` in K-order.

Byte `b` (uint8, `b = 0..7` along the group) holds four codes:

```
byte[b] = q[4b]
        | (q[4b+1] << 2)
        | (q[4b+2] << 4)
        | (q[4b+3] << 6)
```

Two words (Apple is little-endian, so byte 0 is the low 8 bits of the uint32):

```
word0 = byte[0] | (byte[1] << 8) | (byte[2] << 16) | (byte[3] << 24)
      = q[0]  in bits [ 1: 0]
      | q[1]  in bits [ 3: 2]
      | q[2]  in bits [ 5: 4]
      | ...
      | q[15] in bits [31:30]

word1 = same packing of q[16..31]
```

Extract of code `i` in a word: `(word >> (2*i)) & 0x3` for `i ∈ 0..15`.

For `g32`, GEMV `qmv_fast_impl` (`quantized.h` 769–774, 791) assigns **one uint32 (16 codes) per SIMD lane**; two consecutive lanes cover one group of 32 and share one `(scale, bias)` (`scale_step_per_thread = group_size / values_per_thread = 32/16 = 2`).

### 2.3 Row-major, last-axis groups

`mx.quantize` docs (`__init__.pyi` 3388–3390): *every `group_size` elements in a row of `w` are quantized together. Hence the last dimension of `w` should be divisible by `group_size`.*

Packed `weight[n, :]` is the concatenation of groups along `K` for output row `n`. There is no column-major / blocked / interleaved tile layout at rest; tiling is a GEMM-loader concern only (`QuantizedBlockLoader`, §4.2).

---

## 3. Exact reconstruction formula

### 3.1 Decode (what GEMV must implement)

Python (`__init__.pyi` 3502–3508):

```
w_i = s * ŵ_i + β
```

Metal `dequantize<>` (`quantized.h` 483–502), comment on line 483: *“Decode one quantized block (`scale * q + bias`) into `w_local`.”*

For `bits==2`:

```metal
float sc[4] = {s, s / 4.0f, s / 16.0f, s / 64.0f};
w_local[4*i + 0] = sc[0] * (w[i] & 0x03) + b;   // q0 = bits [1:0]
w_local[4*i + 1] = sc[1] * (w[i] & 0x0c) + b;   // q1 = bits [3:2], unshifted
w_local[4*i + 2] = sc[2] * (w[i] & 0x30) + b;
w_local[4*i + 3] = sc[3] * (w[i] & 0xc0) + b;
```

`s/4 * (byte & 0x0c)` ≡ `s * ((byte >> 2) & 0x03)`. Same identity for the 16 and 64 divisors.

`affine_dequantize` (`quantized.h` 2774–2781) writes the shifted form, which is the one to copy:

```
d = (byte >> (2*i)) & 0x03     // d ∈ {0,1,2,3}, unsigned
w = scale * float(d) + bias
```

`q` is **unsigned**. There is no zero-point subtraction at decode time; any zero-point is folded into `scale`/`bias` by the encoder (§3.3). **`scale` may be negative.**

Fused matvec identity used by `qdot` (`quantized.h` 289):

```
sum_i x_i * (scale * q_i + bias)  =  scale * sum_i (q_i * x_i)  +  bias * sum_i x_i
```

implemented as `return scale * accum + sum * bias`.

### 3.2 Fused `qdot` extract (GEMV fast path)

`load_vector` for `bits==2` (`quantized.h` 37–44) pre-divides the activation so `qdot` can multiply raw bitfields:

```
x'[0] = x[0]
x'[1] = x[1] / 4
x'[2] = x[2] / 16
x'[3] = x[3] / 64
```

`qdot` (`quantized.h` 205–212):

```
accum += x'[0]*(w & 0x03) + x'[1]*(w & 0x0c) + x'[2]*(w & 0x30) + x'[3]*(w & 0xc0)
```

Bit-exact with `scale * q + bias` as long as the divisions are in float (the kernel uses `float U`). A Hawking kernel may either (a) copy this trick or (b) shift-and-mask `q` then FMA `x * (scale*q + bias)`. (b) is clearer; (a) is what mlx runs.

### 3.3 Encode (how the safetensors were produced)

Do **not** treat the Python docstring as bit-exact for `scale`/`bias`. The docstring (`__init__.pyi` 3435–3456) is the textbook map:

```
α = max_i w_i
β = min_i w_i
s = (α − β) / (2^b − 1)
ŵ_i = round((w_i − β) / s)
```

with `s` stored as `scales` and `β` as `biases`. The **installed Metal encoder** (`affine_quantize`, `quantized.h` 2619–2722) is the code that actually packed the artifact. For `bits=2`, `group_size=32`:

```
n_bins = (1 << bits) - 1          // 3
eps    = 1e-7
values_per_reduce = group_size / 32 = 1   // one SIMD of 32 lanes = one group
```

Per group (simd_min / simd_max over 32 lanes):

```metal
float w_min = Limits<T>::max;   // +inf
float w_max = 0;                // NOT −inf. All-negative groups force max to 0.

// then per-element min/max, then:
w_min = simd_min(w_min);
w_max = simd_max(w_max);

float scale = max((w_max - w_min) / n_bins, eps);
bool  side  = abs(w_min) > abs(w_max);
scale = side ? scale : -scale;          // negate if |max| >= |min|
float edge  = side ? w_min : w_max;     // larger-magnitude endpoint
float q0    = round(edge / scale);
bool  at_zero = q0 == 0.0f;
scale = at_zero ? scale : edge / q0;    // snap so `edge` is exactly representable
float bias  = at_zero ? 0 : edge;

uint8_t q = min(round((w - bias) / scale), n_bins);   // then uint8 saturates q<0 → 0
```

Consequences for a reimplementer:

1. Reconstruction is always `ŵ = scale * q + bias` with `q ∈ {0,1,2,3}`.
2. `scale` is not necessarily `(max−min)/3` and is not necessarily positive.
3. `bias` is not necessarily `min`. It is the snapped endpoint (or `0` when `round(edge/scale)==0`).
4. `w_max` initialized to `0` pulls all-negative groups onto a range that includes zero.
5. `eps = 1e-7` floors a degenerate range.

Host wrappers in the dylib (not in headers): `mlx::core::affine_quantize`, `mlx::core::affine_dequantize`. Primitive: `mlx::core::fast::Quantize` (`mlx/include/mlx/fast_primitives.h` 328–362) with `dequantize_` flag.

For GEMV you only need §3.1. For a native re-quantizer that must match mlx safetensors, copy §3.3.

---

## 4. How the GEMV consumes the pack: in-register, not expand-to-dense

### 4.1 Decode / GEMV (what token generation hits)

`QuantizedLinear` uses `transpose=True`, so the op is `y = x @ W.T` with `x` shape `(..., K)` and `W` shape `(N, K)`. For `M = 1` (decode) this is a quantized **matrix–vector**: each output `y[n]` is the dot of packed row `n` with `x`.

That path **never materializes a dense `(N, K)` weight**. Three in-register kernels:

| kernel | impl | dequant style |
|---|---|---|
| `affine_qmv_fast` | `qmv_fast_impl` (`quantized.h` 757–822) | fused `qdot` in registers |
| `affine_qmv` | `qmv_impl` (`quantized.h` 824–983) | same `qdot`, plus remainder-safe `qdot_safe` |
| `affine_qmv_quad` | `qmv_quad_impl` (`quantized.h` 700–755) | same `qdot`, quad reductions, only `K ∈ {64, 128}` |
| `affine_qmv_wide` | `qmv_wide_impl` (`quantized.h` 985–1080) | `dequantize` into an 8-value **register** chunk, reused across `vecs_per_tg` vectors |

`qmv_wide_impl` comment (`quantized.h` 985–987):

> Affine analog of fp_qmv_wide. Weights carry a scale and bias per group, so each group is decoded in 8-value sub-chunks (`scale * q + bias`, **registers bounded for any group_size**) and reused across the `vecs_per_tg` vectors.

`qmv_wide` is the small-`M` (2..5) path, not the `M=1` decode path. It still does not write a dense matrix to device memory.

There is a separate **expand-to-dense** kernel, `affine_dequantize` (`quantized.h` 2724–2784). That is `mx.dequantize`, used by embedding gather and by `dequantize_model`. `QuantizedLinear.__call__` does **not** call it.

### 4.2 Prefill / GEMM (for contrast, not the decode GEMV)

`affine_qmm_t` (`quantized.h` 1192–1318, 1901+) uses `QuantizedBlockLoader` (`quantized.h` 571–698). The loader **does** dequant packed bytes into a **threadgroup** tile `Ws[BN * BK_padded]` as dense `T`, then Steel `BlockMMA`. Tile is `BM=BN=BK=32` in the non-NAX kernel; NAX variants in `quantized_nax.h` use `BM=BN=BK=64`. That expansion is tile-local, not a full-matrix unpack.

NAX specialized kernels exist **only** for QMM (`affine_qmm_t_nax`, `affine_qmm_n_nax`, gather-qmm). There is **no** `affine_qmv_*_nax` in `mlx.metallib`. Decode GEMV is the SIMD `qmv_*` kernels.

### 4.3 Host launchers (dylib, no header)

Demangled from `mlx/lib/libmlx.dylib`:

```
mlx::core::qmv(...)
mlx::core::qmv_quad(...)
mlx::core::qmv_wide(...)
mlx::core::qvm(...)
mlx::core::qvm_split_k(...)
mlx::core::qmm(...)
mlx::core::qmm_nax(...)
mlx::core::qmm_splitk(...)
mlx::core::QuantizedMatmul::eval_gpu(...)   // 0xdfbc28
```

`eval_gpu` is not shipped as source. Dispatch is inferred from (a) what each kernel can legally run and (b) the specialized names in `mlx.metallib`. For affine2 g32 the metallib contains (dtype × batch) specializations of:

```
affine_qmv_fast_{float16_t,bfloat16_t,float}_gs_32_b_2_batch_{0,1}
affine_qmv_{float16_t,bfloat16_t,float}_gs_32_b_2_batch_{0,1}
affine_qmv_quad_*_gs_32_b_2_d_{64,128}_batch_{0,1}
affine_qmv_wide_*_gs_32_b_2_nv_{2,3,4,5}_kl_8_batch_{0,1}
affine_qvm_*_gs_32_b_2_batch_{0,1}
affine_qvm_split_k_*_gs_32_b_2_spk_{8,32}
affine_quantize_*_gs_32_b_2
affine_dequantize_*_gs_32_b_2
affine_qmm_t_*_gs_32_b_2_alN_{true,false}_batch_{0,1}
```

Name grammar: `gs_32` = group size, `b_2` = bits, `d_64/128` = `qmv_quad` `D`, `nv_*` = `vecs_per_tg`, `kl_8` = `k_lanes`, `batch_{0,1}` = the `batched` template flag.

Inferred launch rules (kernel-legal, not a dump of `eval_gpu`):

- `transpose=True`, `M=1`, `K ∈ {64,128}` → `affine_qmv_quad` (`D` matches `K`).
- `transpose=True`, `M=1`, `K` multiple of the fast block (for bits=2: `block_size = 16 * 32 = 512`, `quantized.h` 769–775) → `affine_qmv_fast`. `qmv_fast_impl` has **no** remainder loop; launching it with `K % 512 != 0` would OOB.
- `transpose=True`, `M=1`, otherwise → `affine_qmv` (`qmv_impl` remainder-safe).
- `transpose=True`, `M ∈ {2,3,4,5}` → `affine_qmv_wide` with `vecs_per_tg = M`, `k_lanes = 8`.
- `transpose=True`, larger `M` → `affine_qmm_t` (threadgroup expand + MMA). NAX if the device has it.
- `transpose=False` → `affine_qvm` / `affine_qvm_split_k` (`qouter` in registers). `QuantizedLinear` never takes this path (`transpose=True`).

---

## 5. SIMD / tile geometry (visible in the kernel)

`SIMD_SIZE = 32`, `QUAD_SIZE = 4` (`quantized.h` 14–15).

### 5.1 `qmv_fast_impl` / `qmv_impl` — decode workhorse

Compile-time constants (`quantized.h` 769–776 and 836–844):

```
packs_per_thread      = (bits == 2) ? 1 : 2     // 1 at 2-bit
num_simdgroups        = 2
results_per_simdgroup = 4
pack_factor           = get_pack_factor<bits, 32>() = 16
bytes_per_pack        = 4
values_per_thread     = 16
block_size            = 16 * 32 = 512           // K-step per iteration
scale_step_per_thread = group_size / 16 = 2     // two lanes share one (s, β)
```

Threadgroup: **2 simdgroups × 32 lanes = 64 threads**. Each TG writes **8 output rows** (`2 * 4`).

Pointer setup (`quantized.h` 786–795):

```
out_row = tid.y * 8 + simd_gid * 4
ws     += out_row * (K/4) + simd_lid * 1 * 4     // uint8 view; 4 bytes = 1 uint32
scales += out_row * (K/32) + simd_lid / 2
biases += out_row * (K/32) + simd_lid / 2
x      += tid.x * K + simd_lid * 16
y      += tid.x * N + out_row
```

Lane `simd_lid` owns `x[16*lid : 16*lid+16]` and the matching 16 packed codes of each of the 4 rows this simdgroup produces. Lanes 0 and 1 share group 0’s `(scale, bias)`, lanes 2 and 3 share group 1, …

Loop: for `k = 0; k < K; k += 512`:

1. `sum = load_vector<16, bits=2>(x)` — also fills `x_thread[16]` with the pre-divided values.
2. For `row = 0..3`: `result[row] += qdot(wl_row, x_thread, scale, bias, sum)`.
3. Advance `ws`, `scales`, `biases`, `x` by one block.

Reduce: `result[row] = simd_sum(result[row])`; lane 0 of the simdgroup stores `y[row]`.

`qmv_impl` is the same geometry with a remainder pass via `load_vector_safe` / `qdot_safe` and a “last tile slides back” guard when `N < 8`.

Grid (inferred from `tid.x` = batch/M, `tid.y` = output-row tile):

```
threadgroupSize = (32, 2, 1)     // 2 simdgroups
grid            = (M,  ceil(N / 8),  B)   // B = product of batch dims, or 1
```

When `batched=true`, `adjust_matrix_offsets` (`quantized.h` ~1540) rebases `x, w, scales, biases, y` from `tid.z` before the impl.

### 5.2 `qmv_quad_impl` — tiny K

`values_per_thread = D / 4` with specialized `D ∈ {64, 128}`. One quad (4 threads) holds the whole `K`. `results_per_quadgroup = 8`. Reduction is `quad_sum`, not `simd_sum`. Not used for LLM `K` of several thousand.

### 5.3 `qmv_wide_impl` — tiny M, large N

```
k_lanes = 8
results_per_simdgroup = 32 / 8 = 4
num_simdgroups = 2          // 8 output rows / TG, same as fast
sub = 8                     // dequant 8 values at a time
vecs_per_tg ∈ {2,3,4,5}
```

Each lane walks groups with stride `k_lanes`: `for (g = k_lane; g < K/32; g += 8)`. For each group it dequants 32/8 = 4 sub-chunks of 8 values into `w_dq[8]`, then dots those against each of the `vecs_per_tg` x-rows. Reduction is a shuffle-down ladder of width `k_lanes`, not a full `simd_sum` (because a simdgroup spans multiple output rows).

### 5.4 `qvm_impl` — `x @ W` (no transpose)

Each simdgroup owns `tn * pack_factor = (32/16)*16 = 32` output columns (one group). Lane `simd_lid` walks the **rows** of `W` (`block_size = 32`), loads one activation `x_local`, one `(scale, bias)`, one packed chunk, and `qouter`s into a 32-wide register result. Not the Linear path.

### 5.5 Buffer bindings (decode QMV)

`affine_qmv` / `affine_qmv_fast` (`quantized.h` 1607–1656, 1664–1713):

| buffer | name | contents |
|---|---|---|
| 0 | `w` | `device uint32_t*` packed codes |
| 1 | `scales` | `device T*` |
| 2 | `biases` | `device T*` |
| 3 | `x` | `device T*` activations, row-major `(M, K)` |
| 4 | `y` | `device T*` output `(M, N)` |
| 5 | `in_vec_size` | `constant int` = `K` |
| 6 | `out_vec_size` | `constant int` = `N` |
| 7–14 | batch metadata | `x_batch_ndims`, `x_shape`, `x_strides`, `w_batch_ndims`, `w_shape`, `w_strides`, `s_strides`, `b_strides` |

`T` is `float16_t`, `bfloat16_t`, or `float`. Accumulators inside the kernel are **`float`** (`typedef float U`).

`affine_qmv_wide` additionally binds `constant int& M` (true row count, so the last TG can clamp `vec0 + v < M`).

---

## 6. Native-Metal affine2 g32 GEMV plan (Hawking)

Goal: `y[m, n] = sum_{k=0}^{K-1} x[m, k] * (scale[n, k/32] * q[n, k] + bias[n, k/32])`, with `q` packed as §2. Match mlx decode: **in-register dequant, never a dense `W`**.

### 6.1 Host-side buffers

Assume decode `M=1` first; the same kernel extends to small `M` by looping `m` in registers (mlx’s `qmv_wide`) or by launching `M` in `tid.x` (mlx’s `qmv_fast`).

```
W_packed : MTLBuffer  uint32[N * (K/16)]     // row-major, 16 codes / word
scales   : MTLBuffer  half [N * (K/32)]
biases   : MTLBuffer  half [N * (K/32)]
X        : MTLBuffer  half [M * K]
Y        : MTLBuffer  half [M * N]
params   : K, N, M
```

Row `n` of `W_packed` starts at `n * (K/16)`. Group `g` of that row: words `W_packed[n*(K/16) + 2*g + {0,1}]`, scale `scales[n*(K/32) + g]`.

If the Linear also has a dense `bias` (`QuantizedLinear.bias`), add it in a separate vector-add after the GEMV. Do not confuse it with per-group `biases`.

### 6.2 Recommended kernel: match `qmv_fast_impl` for `M=1`

Specialization: `bits=2`, `group_size=32`, `T=half`. Require `K % 32 == 0` (group) and, for the fast kernel, `K % 512 == 0`. If `K` is only a multiple of 32, use the remainder-safe variant (§6.4).

```metal
// Bindings identical to affine_qmv_fast.
kernel void affine2_g32_gemv(
    device const uint32_t* w     [[buffer(0)]],
    device const half*     scales[[buffer(1)]],
    device const half*     biases[[buffer(2)]],
    device const half*     x     [[buffer(3)]],
    device half*           y     [[buffer(4)]],
    constant int& K              [[buffer(5)]],
    constant int& N              [[buffer(6)]],
    uint3  tid     [[threadgroup_position_in_grid]],
    uint   simd_gid[[simdgroup_index_in_threadgroup]],
    uint   simd_lid[[thread_index_in_simdgroup]])
{
    const int values_per_thread = 16;
    const int block = 512;                 // 32 lanes * 16
    const int rows_per_sg = 4;
    const int nsg = 2;
    const int out_row = tid.y * (nsg * rows_per_sg) + simd_gid * rows_per_sg;
    if (out_row >= N) return;

    const int packed_row_uints = K / 16;   // uint32s per weight row
    const int groups_per_row   = K / 32;

    device const uint32_t* wrow = w + out_row * packed_row_uints + simd_lid;
    device const half* srow = scales + out_row * groups_per_row + simd_lid / 2;
    device const half* brow = biases + out_row * groups_per_row + simd_lid / 2;
    device const half* xcol = x + tid.x * K + simd_lid * values_per_thread;

    float acc[4] = {0, 0, 0, 0};

    for (int k = 0; k < K; k += block) {
        // 16 activations for this lane
        float xv[16];
        float xsum = 0;
        for (int i = 0; i < 16; i++) {
            xv[i] = float(xcol[i]);
            xsum += xv[i];
        }
        for (int r = 0; r < 4; r++) {
            if (out_row + r >= N) break;
            uint32_t word = *(wrow + r * packed_row_uints);   // 16 codes
            float s = float(srow[r * groups_per_row]);
            float b = float(brow[r * groups_per_row]);
            float qdot = 0;
            for (int i = 0; i < 16; i++) {
                uint32_t q = (word >> (2 * i)) & 0x3u;
                qdot += xv[i] * float(q);
            }
            acc[r] += s * qdot + b * xsum;
        }
        wrow += block / 16;                 // +32 uint32s = +512 codes; lane
                                            // spacing (1 uint32) is preserved
        srow += block / 32;                 // +16 groups; lid/2 pairing holds
        brow += block / 32;
        xcol += block;
    }
    for (int r = 0; r < 4; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lid == 0 && out_row + r < N)
            y[tid.x * N + out_row + r] = half(acc[r]);
    }
}
```

Launch:

```
MTLSize tgs = {32, 2, 1};                 // 64 threads, 2 simdgroups
MTLSize grid = { (uint)M, (uint)((N + 7) / 8), 1 };
```

Per-iteration pointer math matches `qmv_fast_impl` (`quantized.h` 810–813): the simd as a whole consumes 512 codes = 32 uint32s of a row, so each lane’s `uint32*` advances by 32 (`block/16`) and still reads words `lid, lid+32, lid+64, …`. Scales advance by `512/32 = 16` groups; the `simd_lid/2` pairing is unchanged.

### 6.3 Threadgroup / occupancy notes

- No threadgroup memory on the decode path. Registers: `x[16]`, four `float` accumulators, one `uint32` word per row.
- Do not load a full dequantized row. 2-bit decode bandwidth is `K/4` bytes of codes + `K/32 * 4` bytes of fp16 scale+bias per output row, plus `2K` bytes of fp16 `x` (broadcast across the TG’s 8 rows — `x` is re-read per TG; mlx also re-reads `x` per TG rather than staging it).
- Keep the FMA in `float`. mlx does.

### 6.4 Remainder-safe variant (`K % 512 != 0`)

Copy `qmv_impl` (`quantized.h` 824–983): full blocks as above, last partial block uses `min(16, K - k - 16*simd_lid)` valid lanes/elements, zero-fill the rest, and skip `q` terms for the padded tail. If `N < 8`, only emit `out_row + r < N`. mlx also slides the last output tile back so the final TG always has 8 live rows when `N ≥ 8`; a first Hawking kernel can instead guard stores.

### 6.5 Small-M extension

For `M ∈ 2..5` (speculative draft, multi-token chunk) prefer mlx’s `qmv_wide` shape: 8 K-lanes per simdgroup, dequant each group once into `w_dq[8]` × 4 sub-chunks, reuse against `M` x-rows. Bindings as `affine_qmv_wide` (`quantized.h` 1715–1772). For prefill (`M ≳ 16`) do not GEMV; write a QMM that dequants a `BK=32` tile into threadgroup memory (`QuantizedBlockLoader` + MMA), or convert once and use a dense GEMM — that is a different kernel.

### 6.6 Numeric checklist against mlx

A native kernel is correct iff, for random `x` and a matrix quantized by `mx.quantize(..., group_size=32, bits=2, mode="affine")`:

1. Packed `weight` last dim is `K/16`, dtype uint32.
2. `(word >> (2*i)) & 3` recovers `q`.
3. `y[n] = sum_k x[k] * (float(scales[n,k/32]) * q[n,k] + float(biases[n,k/32]))` in fp32 accumulate, stored as fp16.
4. Matches `mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=32, bits=2, mode="affine")` within ordinary fp16 matvec noise (not bit-identical to the fused `qdot` pre-divide, but should be within a few ulp unless you copy `qdot` exactly).

To **produce** matching safetensors, the encoder must copy §3.3, not the Python docstring.

### 6.7 What not to do

- Do not call `mx.dequantize` then dense GEMV. That is the slow path mlx itself avoids for Linear.
- Do not treat `biases` as a zero-point to subtract from `q`. Decode is `s*q + β`, and `s` can be negative.
- Do not assume `q` is two’s-complement or shifted by 2. Codes are `{0,1,2,3}`.
- Do not pack MSB-first. LSB of each byte/word is `q[0]` of that pack.
- Do not put scale/bias in the packed word. They are sidecar arrays, one pair per 32 K-elements per row.

---

## 7. Source index (every claim above)

| fact | file (under `site-packages`) |
|---|---|
| affine mode table, pack-from-LSB docs, `w = s·q + β` | `mlx/core/__init__.pyi` 3356–3508 |
| C++ `quantize` / `dequantize` / `quantized_matmul` signatures | `mlx/include/mlx/ops.h` 1566–1597 |
| `QuantizationMode::Affine`, `QuantizedMatmul` | `mlx/include/mlx/primitives.h` 155, 1616–1648 |
| `Quantize` primitive | `mlx/include/mlx/fast_primitives.h` 328–362 |
| `get_pack_factor` / `get_bytes_per_pack` | `mlx/include/mlx/backend/common/quantized.h`; Metal copy `.../metal/kernels/quantized.h` 17–26 |
| `qdot` / `load_vector` / `dequantize` / `qouter` | `mlx/include/mlx/backend/metal/kernels/quantized.h` 28–569 |
| `QuantizedBlockLoader` (GEMM tile expand) | same file 571–698 |
| `qmv_quad_impl` / `qmv_fast_impl` / `qmv_impl` / `qmv_wide_impl` / `qvm_impl` | same file 700–1188 |
| kernel entry points + buffer bindings | same file 1549–1780 |
| `affine_quantize` encoder (true scale/bias) | same file 2619–2722 |
| `affine_dequantize` expand-to-dense | same file 2724–2784 |
| `QuantizedLinear` store + `quantized_matmul(..., transpose=True)` | `mlx/nn/layers/quantized.py` 205–307 |
| `Linear.to_quantized` | `mlx/nn/layers/linear.py` 72–107 |
| mlx_lm load-time `nn.quantize` | `mlx_lm/utils.py` 348–366, 774–850 |
| Qwen3.5 Linear sites | `mlx_lm/models/qwen3_5.py`, `mlx_lm/models/qwen3_next.py` 80–169 |
| specialized kernel names | `mlx/lib/mlx.metallib` strings |
| host launchers `qmv` / `qmv_fast` / `qmv_wide` / `qmm` | `mlx/lib/libmlx.dylib` (`nm`) |

Host `QuantizedMatmul::eval_gpu` dispatch thresholds are **not** in the installed headers; §4.3 states only what the kernels themselves constrain.
)
