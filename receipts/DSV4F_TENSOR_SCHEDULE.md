# DSV4F tensor schedule

Read-only analysis of the sealed DeepSeek-V4-Flash 43-layer source stream.
Masses come from `manifest["tensors"]`. No chunk body was opened.

## Artifact identity

- path: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/deepseek-v4/full-43-layer-stream.gravity`
- schema: `hawking.gravity.deepseek_v4.full_stream.v1`
- status: `FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY`
- seal_sha256: `ba9039bfe71328e2e47ced782bd1f931e2d412055382da0ea669092c1d90bfed`
- content_addressed_chunk_sha256: `15e00fb1b91ac074b7f24686de4e289f76d66eb1c3fb4ad643de027adc78ca13`
- chunks: 69837
- tensor_count: 69187
- total_tensor_bytes: 159609485896
- repository: `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1`
- MTP tensors are named `mtp.0.*`, not `layers.43.*`. They classify into the same organs; only the four MTP-unique projections stay in `other`.
- The BYTE_AUCTION `158.07B logical` figure is **stored elements** (`prod(shape)` = 158069433298). Unpacked FP4 logical params are larger because each I8 holds two e2m1 values.

## Coverage

- classified tensors: **69187 / 69187**
- byte mass sum: **159609485896**
- byte residual vs manifest total: **0**
- logical unpacked params: **290942289362**
- stored elements (prod of physical shapes): **158069433298**
- shape/dtype undetermined: **0**
- other (enumerated): **4**
- scope counts: base=67606 mtp=1575 global=6
- covers_all_tensors: **True**

## Per-organ mass

| organ | tensors | bytes | % bytes | logical params | % params | source bpw (logical) |
|---|---:|---:|---:|---:|---:|---:|
| routed_expert | 67584 | 150592290816 | 94.3505 | 283467841536 | 97.4310 | 4.2500 |
| shared_expert | 264 | 1107363840 | 0.6938 | 1107296256 | 0.3806 | 8.0005 |
| mla | 484 | 4706307584 | 2.9486 | 4706011904 | 1.6175 | 8.0005 |
| mhc | 270 | 138945864 | 0.0871 | 34736466 | 0.0119 | 32.0000 |
| indexer_compressor | 311 | 801075968 | 0.5019 | 487194752 | 0.1675 | 13.1541 |
| hash_layers | 3 | 18616320 | 0.0117 | 0 | 0.0000 | — |
| embeddings | 1 | 1059061760 | 0.6635 | 529530880 | 0.1820 | 16.0000 |
| lm_head | 1 | 1059061760 | 0.6635 | 529530880 | 0.1820 | 16.0000 |
| norms | 180 | 888832 | 0.0006 | 444416 | 0.0002 | 16.0000 |
| router_gate | 85 | 92316672 | 0.0578 | 46147840 | 0.0159 | 16.0036 |
| other | 4 | 33556480 | 0.0210 | 33554432 | 0.0115 | 8.0005 |

Routed-expert subroles (w1/w3/w2):

| subrole | tensors | bytes | logical params |
|---|---:|---:|---:|
| w1 | 22528 | 50197430272 | 94489280512 |
| w2 | 22528 | 50197430272 | 94489280512 |
| w3 | 22528 | 50197430272 | 94489280512 |

### Other (fully enumerated)

These names matched no organ suffix after `layers.L.` / `mtp.N.` stripping. MTP-unique projections live here on purpose.

- `mtp.0.e_proj.scale`
- `mtp.0.e_proj.weight`
- `mtp.0.h_proj.scale`
- `mtp.0.h_proj.weight`

## Dtype / source precision

| stored dtype | family | tensors | bytes | % bytes | logical params |
|---|---|---:|---:|---:|---:|
| BF16 | bf16 | 433 | 2830518528 | 1.7734 | 1415259264 |
| F32 | f32 | 417 | 144672072 | 0.0906 | 36168018 |
| F8_E4M3 | fp8_e4m3 | 375 | 6023020544 | 3.7736 | 6023020544 |
| F8_E8M0 | scale_ue8m0 | 34167 | 8858737664 | 5.5503 | 0 |
| I64 | i64 | 3 | 18616320 | 0.0117 | 0 |
| I8 | fp4_e2m1fn_x2_packed_i8 | 33792 | 141733920768 | 88.8004 | 283467841536 |

Experts are already FP4-native (`I8` packed e2m1fn_x2 + `F8_E8M0` scales). Further 'compression' of experts is a second packing on top of FP4, not a BF16→low-bit collapse. Control path is FP8 e4m3 + UE8M0. Embeddings, lm_head, norms, gate scores, and compressor weights are BF16. mHC and APE tables are F32. Hash `tid2eid` is I64.

## BPW feasibility envelope

```
complete_bpw = f_expert * expert_bpw + f_nonexpert * nonexpert_bpw
```

- f_expert = 0.978115414  (284575137792 logical params)
- f_nonexpert = 0.021884586  (6367151570 logical params)
- source expert bpw (logical, includes scales) = 4.264593339615759
- source non-expert bpw (logical) = 9.938298032380592
- source complete bpw (logical unpacked) = 4.388760018242892
- source complete bpw (stored elements) = 8.077943094543604
- source complete bpw vs claimed 158.07B = 8.07791413404188

Required non-expert bits to hold `complete_bpw = 1.5`:

| expert_bpw | required_nonexpert_bpw | feasible |
|---:|---:|---|
| 1.0 | 23.847131 | True |
| 1.2 | 14.908278 | True |
| 1.3 | 10.438852 | True |
| 1.4 | 5.969426 | True |
| 1.4609 | 3.247546 | True |

Source expert_bytes already include UE8M0 scales. The envelope table treats expert_bpw as the complete expert rate a child would bill (weights+scales+codebooks) per logical expert param.

## Per-layer streaming

A layer executes in this order (official `inference/model.py` `Block.forward`):

- embed.weight (once; evict after expanding hc_mult copies)
- for layer L in 0..42:
-   hc_pre attn (hc_attn_fn/base/scale) — input dim 16384
-   attn_norm
-   MLA: wq_a, q_norm, wq_b, wkv, kv_norm, optional compressor, optional indexer, attn_sink, wo_a, wo_b
-   hc_post attn
-   hc_pre ffn (hc_ffn_fn/base/scale)
-   ffn_norm
-   gate.weight (+ bias or tid2eid) then route top-6
-   6 routed experts (w1,w3,w2 + scales) streamed after the route is known
-   1 shared expert (w1,w3,w2 + scales)
-   hc_post ffn
- hc_head_fn/base/scale + norm.weight + head.weight
- MTP layer 43 is excluded from the base streamed token path

Peak **full-layer** resident weights: **3596055640** bytes at layer 2 (all 256 routed experts present).
Peak **streamed-decode** resident weights (keep non-experts + shared + top-6 routed): **253719640** bytes at layer 2.

| L | mode | gate | tensors | full bytes | streamed peak | MLA | indexer/compr | 6 experts | shared |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | sliding_window_only | hash_token_id_to_expert_ids | 1565 | 3566148952 | 223812952 | 106961536 | 0 | 80216064 | 25167360 |
| 1 | sliding_window_only | hash_token_id_to_expert_ids | 1565 | 3566148952 | 223812952 | 106961536 | 0 | 80216064 | 25167360 |
| 2 | ratio_4_with_indexer | hash_token_id_to_expert_ids | 1576 | 3596055640 | 253719640 | 106961536 | 29906688 | 80216064 | 25167360 |
| 3 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 4 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 5 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 6 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 7 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 8 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 9 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 10 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 11 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 12 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 13 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 14 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 15 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 16 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 17 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 18 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 19 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 20 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 21 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 22 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 23 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 24 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 25 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 26 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 27 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 28 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 29 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 30 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 31 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 32 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 33 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 34 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 35 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 36 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 37 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 38 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 39 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 40 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |
| 41 | ratio_128 | learned_scores_with_selection_bias | 1569 | 3568596312 | 226260312 | 106961536 | 8651776 | 80216064 | 25167360 |
| 42 | ratio_4_with_indexer | learned_scores_with_selection_bias | 1576 | 3589851224 | 247515224 | 106961536 | 29906688 | 80216064 | 25167360 |

MTP layer 43 is **not** on the base token path (1575 tensors, 3593787756 bytes, sliding_window_only).

## Activation X capture (sizes the later capture lane)

Organs that **must** retain X for a ≤1.5 complete activation-weighted fit:

| organ | input dim | required |
|---|---:|---|
| `routed_expert.w1` | 4096 | yes |
| `routed_expert.w3` | 4096 | yes |
| `routed_expert.w2` | 2048 | yes |
| `shared_expert.w1` | 4096 | yes |
| `shared_expert.w3` | 4096 | yes |
| `shared_expert.w2` | 2048 | yes |
| `mla.wq_a` | 4096 | yes |
| `mla.wq_b` | 1024 | yes |
| `mla.wkv` | 4096 | yes |
| `mla.wo_a` | 4096 | yes |
| `mla.wo_b` | 8192 | yes |
| `indexer.wq_b` | 1024 | yes |
| `indexer.weights_proj` | 4096 | yes |
| `compressor.wkv` | 4096 | yes |
| `compressor.wgate` | 4096 | yes |
| `indexer.compressor.wkv` | 4096 | yes |
| `indexer.compressor.wgate` | 4096 | yes |
| `router_gate.weight` | 4096 | yes |

Distinct X matrices (this is the capture surface):

- `h_post_attn_norm` dim=4096 → mla.wq_a, mla.wkv, compressor.wkv, compressor.wgate, indexer.weights_proj, indexer.compressor.wkv, indexer.compressor.wgate
- `q_lora_qr` dim=1024 → mla.wq_b, indexer.wq_b
- `attn_out_grouped` dim=4096 → mla.wo_a
- `o_lora` dim=8192 → mla.wo_b
- `h_post_ffn_norm` dim=4096 → router_gate.weight, routed_expert.w1, routed_expert.w3, shared_expert.w1, shared_expert.w3
- `swiglu_hidden_routed` dim=2048 → routed_expert.w2
- `swiglu_hidden_shared` dim=2048 → shared_expert.w2
- `hc_flat_pre_attn` dim=16384 → mhc.hc_attn_fn
- `hc_flat_pre_ffn` dim=16384 → mhc.hc_ffn_fn
- `h_final` dim=4096 → lm_head

## Prior 145 MB vs 103 MB

The 145 MB vs 103 MB split is a PER-TOKEN SERVED traffic figure, not a stored-parameter split. Stored byte mass is overwhelmingly routed experts. The 102,760,448 MoE number reproduces exactly as 6*routed_fp4_weights + shared_fp8_weights + gate.weight (NO scales). The 144,703,488 attention number reproduces exactly as MLA_fp8_weights + shared_fp8_weights + ONE routed_fp4_expert_weights — that is not a clean attention-only mass. A 1.5 COMPLETE budget still cannot ignore MLA/mHC/indexer/router: they dominate served bytes/token even though they are a small fraction of stored mass.

- reconstructed MoE teacher-active: 102760448 (match=True)
- reconstructed 'attention' 138 MiB identity: 144703488 (equals claimed=True)
- MLA fp8 weights only: 106954752

## Undetermined

None. Every tensor has a known dtype, shape, and bytes that reconcile.

