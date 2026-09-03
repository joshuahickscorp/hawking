# Chunked prefill for the Qwen3.8 hybrid resident

Read-only design. Derived from the shipped kernels and geometry, not from what
the architecture is usually said to do. Nothing here is implemented yet.

## Why

Prefill steps one prompt token at a time through the same session function decode
uses. Measured on one live call: 4,467 prompt tokens, 2,590,860 dispatches,
190 s of wall at 98% GPU, `host_control_share_of_wall` 0.0107. That is 580
dispatches per token across 64 layers, or **9.06 per layer per token**.

Nine kernels per layer is tight. There is no fat to fuse inside a token, so the
only remaining lever is to stop paying decode prices for prompt tokens: process
many positions together and turn per-layer GEMV into GEMM.

## 1-2. What the layers are

| | count | mixer |
|---|---|---|
| DeltaNet (linear, recurrent) | 48 | gated delta rule |
| GQA (ordinary full attention) | 16 | every 4th layer |

`QWEN38_FULL_ATTENTION_INTERVAL = 4`. DeltaNet geometry: 16 key heads, 48 value
heads (3 values per key head), key and value head dim 128. Carried state is
therefore `48 x 128 x 128` floats per layer -- 3.1 MB at f32, **151 MB across the
48 recurrent layers**. There is also a causal depthwise conv of kernel width 4
ahead of the delta rule.

## The recurrence, read off the kernel

`shaders/qwen_next.metal::qwen_next_gated_delta_decode_single`, per head, with
state `S` indexed `[k][v]`, scalar decay `d` and scalar beta `b`:

```
S     <- d * S                       // decay in place
kv    =  S^T k                       // read after decay
delta =  (v - kv) * b
S     <- S + k (x) delta             // rank-1 update
out   =  S^T q
```

Substituting, and keeping S as a (K, V) matrix:

```
S_t = d_t * S_{t-1} + b_t * k_t (v_t - d_t S_{t-1}^T k_t)^T
    = d_t (I - b_t k_t k_t^T) S_{t-1}  +  b_t k_t v_t^T
```

## 4-5. The scientific question: yes, and here is why

Write `S_t = A_t S_{t-1} + B_t` with

```
A_t = d_t (I - b_t k_t k_t^T)     (K x K, identity plus rank one, scaled)
B_t = b_t k_t v_t^T               (K x V, rank one)
```

The state transition is **affine in S**, so it composes associatively:

```
(A_2, B_2) . (A_1, B_1) = (A_2 A_1,  A_2 B_1 + B_2)
```

That is the whole answer. A chunk of T tokens collapses to a single `(A, B)`
pair; chunks combine by the same rule; the combination is associative, so it is a
scan and not a serial dependency.

The product of the `A_t` does not have to be formed as T dense K x K matmuls.
Each factor is a scaled rank-1 update of the identity, so the product over a
chunk has the WY form

```
prod_t (I - b_t k_t k_t^T)  =  I - K_c^T W
```

with `K_c` the chunk's keys (T x K) and `W` (T x K) built by a short sequential
recurrence over T rows of small vectors -- not over the K x V state. The state is
touched once per chunk instead of T times.

**So the serial dependency is real but it is O(T) work on T x K objects, not on
the 128 x 128 state.** That is the difference between the current program and the
chunked one.

## 3, 7. What becomes GEMM

Computable for the entire chunk before any state advancement:

- input RMSNorm
- q, k, v projections, and the `ba` projection that yields decay and beta
- the causal conv of width 4 (needs a 3-token halo from the previous chunk)
- RoPE on the GQA layers
- the MLP: router, gate, up, down
- the output projection after the mixer

Inside the chunk, once `W` is formed:

- the intra-chunk attention-like terms are T x T and T x K GEMMs
- the GQA layers are ordinary blocked attention over the chunk, which is the
  easy half: no recurrence at all, just a causal mask

Irreducibly sequential: the scan that builds `W`, and the single state advance
per chunk.

## 6. What crosses a chunk boundary

- the DeltaNet state `S` per layer: 48 x 128 x 128 (3.1 MB f32)
- 3 tokens of conv history per layer
- the GQA KV cache, exactly as today

Nothing else. The existing `prefix_checkpoint` already captures the recurrent
carry, so the boundary object is not new.

## 8. Equivalence gates

Bit-parity is the wrong bar and claiming it would be dishonest: chunking changes
the reduction order, so f32 sums differ in the last places. The kernel comment
says the bar itself -- "a tiled SIMD implementation may replace it only after
parity against this operator is retained". Concretely:

1. **Operator parity.** Same random q/k/v/decay/beta, one head, T = 1: chunked
   path must equal the serial kernel bitwise, since T = 1 is the same arithmetic.
2. **Numerical parity.** T in {16, 64, 256}: max abs error on `S` and on `out`
   within a stated tolerance of the serial reference, reported not assumed.
3. **Whole-model parity.** Same prompt, greedy: identical token ids for at least
   256 generated tokens against the current binary. A single divergent argmax
   fails the gate.
4. **State parity across the boundary.** Prefill N tokens chunked, checkpoint,
   restore, decode; compare against prefill N serially then decode.

## 9. Smallest prototype

One DeltaNet head, host-side, in Rust, no Metal: implement the serial recurrence
and the chunked `(A, B)` composition side by side and assert gate 1 and gate 2.
This is a few hundred lines and needs no GPU. It either confirms the algebra
above on real weights or refutes it before anything is ported to a kernel.

Only then: one chunked Metal kernel for one layer, benchmarked against 
`qwen_next_gated_delta_decode_single` called T times.

## 10. What would justify integrating it

Prefill is 98% of the wall on a long call. A chunk of T should divide the
per-layer dispatch count by roughly T, so the arithmetic ceiling is large, but
the GEMMs are not free and the scan is not free.

The bar: **a measured 4x or better on end-to-end prefill wall for a 4,000-token
prompt, with gates 1-4 green.** Below 4x the integration risk against a sealed
resident is not worth it; the checkpoint work already landed is the cheaper win
and should be exhausted first.

## Order

This waits. The autonomy gates come first: HCLI must land its own mutations
before the machine spends weeks on its own prefill. When they pass, this becomes
an HCLI mission -- it profiles, reads the recurrence, proposes the decomposition
and writes the tests, and Claude verifies the mathematics and the equivalence.
