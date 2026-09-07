# Measured frontier input — the KV cache is stored at twice the model's dtype — 2026-09-05

Everything below was read from source constants and the artifact config in this worktree.
No estimates, no historical figures.

## THE FACT

crates/hawking-core/src/model/qwen38_geometry.rs

    QWEN38_LAYERS 64   QWEN38_GQA_LAYERS 16   QWEN38_GQA_KV_HEADS 4   QWEN38_GQA_HEAD_DIM 256

crates/hawking-core/src/model/qwen38_hybrid_decode.rs allocates the GQA KV cache twice
(once K, once V) through:

    f32b(QWEN38_GQA_LAYERS * max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)

/Users/scammermike/noetic/NOETIC_PARENT_A/config.json, text_config:

    dtype                    bfloat16
    num_key_value_heads      4
    head_dim                 256
    num_hidden_layers        64
    full_attention_interval  4          -> 16 full-attention layers, matching the constant
    max_position_embeddings  262144

The model is bfloat16. Its KV cache is f32.

## THE ARITHMETIC

    16 * 4 * 256 * 4 B * 2  =  131,072 B/token  =  128 KB/token   as allocated today
    16 * 4 * 256 * 2 B * 2  =   65,536 B/token  =   64 KB/token   at the model's own dtype

    context      KV at f32     KV at bf16
      9,728        1.27 GB       0.64 GB
     32,768        4.29 GB       2.15 GB
    131,072       17.18 GB       8.59 GB
    262,144       34.36 GB      17.18 GB

The body is ~11 GB and the box had ~49 GB free. At f32, the 262,144 rung needs 34.4 GB of
KV on top of the body. At bf16 it needs 17.2 GB. The admission gate on this machine is the
METAL WORKING SET, not free RAM.

## WHY THIS IS NOT A COMPRESSION SCHEME

This is not sub-bit work and it is not a codec. It is a correction: the cache is being held
at twice the precision the model declares. The DeltaNet recurrent+conv state is a separate,
FIXED 156,893,184 B and is untouched by this.

## WHY IT IS STILL NOT FREE

f32 storage keeps precision that bf16 storage discards, and attention reads this cache every
step for every full-attention layer. A wall improvement with a changed answer is a FAILURE,
not a win.

Acceptance is SEMANTIC EXECUTION, never reconstruction arithmetic:

  - generated token identity, or a stated and measured capability-equivalence, on real
    prompts through the real path
  - the protected gate receipts/sovereign/G007_deltanet_state.json requires
    `output_equivalent is True` and `full_attention_kv_handled is True`
  - measure resident bytes and bytes/token before and after, and the prompt/decode latency

## THE QUESTION FOR THIS MISSION

Establish, from the source, whether the GQA attention kernels can read a bf16 KV cache while
still accumulating in f32 -- that is the shape that usually preserves the answer. If the
kernels require an f32 cache, say so and name the kernel: that is the real boundary and it
should be recorded rather than worked around.

If they do not, emit the smallest mutation that stores the GQA KV cache at the model's own
dtype, and name a test that fails before it and passes after.

## CONSTRAINTS

- Every file in receipts/sovereign/VERIFIER_MANIFEST.json is PROTECTED. Landing refuses any
  proposal touching them.
- The sealed artifact must not be mutated (G007 asserts sealed_artifact_mutated is False).
- Smallest executable change. One operation.
