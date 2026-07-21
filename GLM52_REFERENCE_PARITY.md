# GLM-5.2 adapter, twin, and reference parity

Adapter: **PASS_SYNTHETIC_TWIN_AND_OFFICIAL_HEADER_TOKENIZER_SCHEMA**. Reference: **PASS_SYNTHETIC_MAIN_AND_MTP_SELF_CONSISTENCY_SOURCE_PARENT_PENDING**.

Official header graph: 59,585 tensors and 753,329,940,480 logical weights, exact.
Synthetic main HF/reference: max abs `0.00239688`, relative Frobenius `0.000332304`, cosine `0.999999821`, top-1 `1.000`.
Strict causally effective DSA index agreement: `0.9875`; tie-aware agreement: `1.0000`; raw score max error: `2.91e-11`.
Router expert-set agreement: `1.000`. The synthetic indexer shape probe reached 1,048,576 keys; it is not a full-model or 1M capability result.
Metal: `PASS`.

MTP status is synthetic source conformance and self-consistency only. The pinned external runtime semantics were inspected but that runtime was not executed.

The official BF16 parent forward remains pending because no payload shard has yet been admitted. No synthetic result is presented as source-parent or capability evidence.

Adapter seal: `ca98d39c66d6bccaf40459d45d1666a88d92a604896775591fdddfafb7c44665`.
Reference seal: `4d5793444ba55583d22f1887f828e8985244d38dfd53248aaa2912815a745f5c`.
