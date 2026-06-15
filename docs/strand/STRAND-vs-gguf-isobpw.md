# STRAND vs GGUF — the iso-bpw head-to-head

_The launch doc. One harness, both formats, sorted by bits-per-weight. Settles
"have we beaten GGUF" with measurements, not memory._

**Machine:** Apple M3 Pro, 18 GB, 12 logical cores. **Model:** Qwen2.5-0.5B
(`scratch/qwen-05b`), a deliberate **small-model proxy** — the 7B/14B confirms
queue on the pod (will.md §10 has the 7B/14B STRAND points; the GGUF side at 7B
is the open item). **Dataset/harness:** WikiText-2, the STRAND canon eval
(`ops/eval-ppl.py` → `tools/strand_eval`), ctx 2048, 64 non-overlap windows,
device cpu, dtype bf16. Date: 2026-06-11.

## What "iso-bpw" means here, and why it's the only honest frame

bf16 (PPL ceiling) is **not** the benchmark — it's the unbeatable ceiling. The
real question is: **at the same file size (same bits-per-weight), which format
gives lower perplexity?** Reporting "% over bf16" hides the win/loss; this doc
reports the Pareto frontier of (bpw, PPL) with both formats on it.

### True iso-harness (not cross-harness)

Both formats are scored by the **same** eval. GGUF quants are **dequantized back
to HF safetensors** (`tools/gguf/gguf_to_hf.py`, using gguf's reference
dequantizer) and run through `ops/eval-ppl.py` — the identical code path STRAND
uses. No llama-perplexity / eval-ppl cross-harness caveat is needed. The bf16
anchor is the original HF weights through the same eval.

### Matched bpw denominator

STRAND bills bpw over the **quantized projection weights** (q/k/v/o/gate/up/
down_proj 2D matrices), including its outlier side-channel; norms, biases and
embeddings are not in STRAND's quantized set. The GGUF side is billed the
**same way** (`tools/gguf/gguf_bpw.py`): bits in those same 7 projection-weight
tensors / their element count. Both denominators are **357,826,560 elements —
verified identical**. `file_bpw` (whole-model) is reported alongside for context.

## The GGUF bpw surprise on 0.5B — the dim-896 fallback (the STRAND edge, concrete)

Qwen2.5-0.5B has **896-dim** projection tensors. 896 is not a clean multiple of
llama.cpp's 256-element K-quant superblock, so **the K-quants silently fall back
to 32-blocked legacy types** for these tensors. Measured projection-weight
composition and effective proj-bpw:

| nominal | actual proj-weight types | **proj_bpw** | file_bpw |
|---|---|---|---|
| Q2_K   | Q4_0×120, Q5_0×24, Q3_K×24            | **4.197** | 5.387 |
| IQ3_S  | IQ4_NL×120, Q5_0×24, IQ3_S×24         | **4.197** | 5.387 |
| Q3_K_M | Q4_0×96, Q5_0×46, Q5_1×2, Q4_K×23, Q5_K×1 | **4.574** | 5.660 |
| Q4_K_M | Q5_0×132, Q6_K×12, Q8_0×12, Q4_K×12   | **5.521** | 6.345 |

So "Q2_K" is **not 2-bit here — it is 4.20 bpw on the projection weights.** This
is exactly the will.md edge ("Q4_K is forced to 5.52 bpw on 896-dim tensors")
made fully concrete and extended to the whole K-quant family. STRAND hits a
**uniform, requested** bpw on any dimension (its row-aware RHT handles ragged
896-dim directly); GGUF cannot tile 896 at low bits and pays a large fallback
tax. At the 0.5B this is the headline structural win independent of PPL.

IQ3_S was quantized **without an imatrix** (144/290 tensors fell back); a
calibrated imatrix would help IQ-quant PPL but does not change the bpw tiling
story. Noted as a fairness caveat.

## The table (Pareto frontier — bpw vs PPL, both formats)

<!-- AUTO: regenerate with  python3 tools/gguf/isobpw_table.py  -->
_(filled by the harness; STRAND mp_light already in from the canon run)_

| bpw (proj) | config | PPL | format | note |
|---|---|---|---|---|
| 3.806 | STRAND mp_light (attn4/ffn3) | 15.039 | strand | eff_bpw 3.80564, canon 64w |
| … | (pending) | … | | |

## How to reproduce

```
bash scripts/isobpw-headtohead.sh          # full pipeline, resumable, CPU-gated
python3 tools/gguf/isobpw_table.py          # rebuild the table
```

Tools: `tools/gguf/convert_hf_to_gguf.py` (llama.cpp b6000, gguf 0.19.0),
`tools/gguf/gguf_to_hf.py` (dequant→HF), `tools/gguf/gguf_bpw.py` (matched
billing), `tools/gguf/isobpw_table.py` (join).

## Verdict

_(written once all PPLs land — see the table)_
