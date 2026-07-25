# `.gravity` functional codec specification

```
codec:      glm52.functional.moe.v1
generator:  gen.splitmix64_boxmuller.v1
authority:  tools/condense/gravity_functional_codec.py
metal:      tools/condense/gravity_functional_metal.py
```

Every other codec in the container stores an approximation of a weight matrix. This one
stores a function. It has no tensor to decode, so the container's usual "decode then
matmul" contract does not apply and the descriptor says so explicitly.

## Payload layout

```
offset  size  field
0       8     magic       "GRVFUNC\0"
8       4     version     uint32, currently 1
12      4     width       uint32, input dimension (6144)
16      4     hidden      uint32, feature count; 0 means "no feature map, direct linear"
20      4     out_width   uint32, output dimension (6144)
24      4     rank        uint32, 0 when the readout is unfactored
28      4     activation  uint32, 0 identity, 1 SiLU
32      8     seed        uint64
40      4     scale       float32
44     20     zero padding to 64
64      *     readout     float16 [hidden, out_width], or [hidden, rank] when factored
 *      *     right       float16 [rank, out_width], present only when rank > 0
end     4     layer       uint32
```

## The generator is the specification

The feature map is reproduced from the seed, never stored. That only works if every
implementation agrees on it bit for bit, so the generator is frozen as a stateless hash
rather than a library RNG:

```
splitmix64(state):
    state += 0x9E3779B97F4A7C15
    z = state
    z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z XOR (z >> 27)) * 0x94D049BB133111EB
    return z XOR (z >> 31)

uniform(seed, stream, index):
    key  = seed * 0xD1342543DE82EF95 + stream * 0xA24BAED4963EE407
    bits = splitmix64(key XOR index)
    return (float(bits >> 11) + 0.5) * 2^-53

projection[i][j], index = i * hidden + j:
    u1 = uniform(seed, 1, index)
    u2 = uniform(seed, 2, index)
    return sqrt(-2 ln u1) * cos(2*pi*u2) / sqrt(width)
```

All arithmetic is unsigned 64-bit with wraparound. Two properties matter and both are
load-bearing:

- **Stateless.** Element `(i, j)` is a pure function of `(seed, i, j)`, so a kernel thread
  can produce exactly the elements it needs without producing any others. NumPy's PCG64
  cannot do this, which is why it is not the frozen generator even though the original fit
  used it.
- **Portable.** Ten lines of integer arithmetic reproduce in Rust, C, or Metal Shading
  Language. The Metal transliteration lives in `gravity_functional_metal.METAL_SOURCE` and
  is parity-checked against the CPU authority at 7.9e-7 relative L2.

Conformance test for any reimplementation: reproduce `projection(width=8, hidden=4,
seed=17)` exactly. The golden values are in `GLM52_FUNCTIONAL_CPU_PARITY.json`.

## Execution grammars

| grammar | feature map | resident per layer | measured | vs teacher traffic |
|---|---|---:|---:|---:|
| **FRT-B** procedural | generated in-kernel | 12.58 MB | **2.15 ms** | **48×** |
| FRT-A explicit | held resident | 25.17 MB | 2.90 ms | 24× |
| FRT-D direct linear | none (`hidden = 0`) | 75.50 MB | 2.52 ms | 8× |

FRT-B wins on both axes: 6.29 million in-kernel generator calls per layer are cheaper than
reading 12.58 MB of projection. That is the whole argument for a seeded feature map, and it
is measured rather than assumed.

FRT-C, a structured fast transform, is not implemented: the rank-256 seeded projection
already reaches skill 0.505 at 0.0026 local BPW, so the cheap-transform slot is filled by
something that needed no new kernel.

All three pass the parity gate against the CPU authority. The gate is relative L2 below
5e-2 and cosine above 0.999; a float16 readout under a float32 accumulator will not agree
bit for bit across two reduction orders and is not asked to. Worst measured case is 2.0e-4.

## Container integration

A functional organ is a normal `.gravity` tensor with an abnormal descriptor:

```json
{
  "name": "model.layers.38.mlp.__functional__",
  "codec": "glm52.functional.moe.v1",
  "generator": "gen.splitmix64_boxmuller.v1",
  "disposition": "REPLACED_BY_FUNCTIONAL_CODEC",
  "elements": 9702998272,
  "replaces": ["model.layers.38.mlp.experts.*",
               "model.layers.38.mlp.shared_experts.*",
               "model.layers.38.mlp.gate.weight",
               "model.layers.38.mlp.gate.e_score_correction_bias"],
  "runtime": { "artifact_bytes": ..., "expanded_feature_map_bytes": ..., ... }
}
```

`elements` is the count of **source logical weights the organ stands in for**, not the size
of its own readout. That makes the container's existing rate check meaningful without
changing it: bytes divided by elements is the organ-local rate, and the header says in
`rate_basis` that this is an organ rate and not the model rate.

`replaces` is why no source tensor silently disappears from a manifest. Routing is removed
rather than approximated, so the router matrix and its correction bias are named as
replaced, not omitted.

The shard header carries `"representation": "FUNCTIONAL_MODEL"` so a runtime can refuse a
model it does not know how to execute instead of decoding garbage.

## What a runtime must do

```
read header
-> refuse unless codec and generator are both recognised
-> read payload
-> either generate the feature map once (FRT-A) or nothing (FRT-B)
-> per token: hidden state -> features -> activation -> readout -> MoE replacement
-> the block adds it to the post-attention residual
```

No router executes and no expert executes. If a future variant needs routing for quality,
the contract requires it to be billed and executed explicitly rather than assumed away.

## Verified end to end

`glm52_functional_integration.py run 38` writes a real shard, verifies its body hash and
per-tensor hashes through `gravity_format`, reads the payload back out, executes it on real
captured states, parity-checks all Metal grammars, substitutes the result inside the real
GLM-5.2 block, carries the residual into layer 39 through real attention and a real router,
and takes a greedy token from the logit lens. Artifact 12,584,647 bytes, block skill 0.880,
layer-39 router top-1 agreement 0.851, greedy token agreement 0.552.
