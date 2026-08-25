# TOKENIZER GRAVITY

Source: S026 §30-38, §82, §93, §109, §118. Measurement:
`receipts/headless/TOKENIZER_GRAVITY.json` via `tools/headless/tokenizer_gravity.py`.

Tokenizer topology is part of the model's physical program. Vocab size
changes model bytes, output-head work, sequence length, and KV growth.
Complete executable closure counts tokenizer data (tokenizer.json,
merges, vocab, chat template) — those bytes are MODEL_SPECIFIC, not free.

## Control (not the default)

The published Qwen3.8 ASCII-prune (bsaleh03, 248,320 → 129,006 rows) is
reproduced exactly. Predicate: drop a row iff its GPT-2-decoded payload is
valid UTF-8 containing a non-ASCII codepoint. Embed and output rows are
gathered; the rest of the model is unchanged. All 256 byte-level tokens and
all 276 special/reserved tail rows survive.

On this box the LM head is ~5% of decode FLOPs and ~5% of active weight
bytes. GQA KV grows at 131,072 bytes/position; DeltaNet state does not
grow with sequence. Effective sequence cost is

    n_tokens × (active_weight_bytes_per_token + kv_bytes_per_position)

A 10% smaller head that emits 25% more tokens loses (§35). Here the head
cut is ~48% of the head and ~5% of the token, so any inflation ≳ 5% loses
on decode work.

Measured TOKEN_INFLATION_RATIO (ASCII-prune CONTROL, this receipt):

| domain | I | net-beneficial |
|---|---|---|
| ASCII English (constructed) | 1.000 | yes |
| code / JSON / shell / paths / structured | 1.000–1.001 | yes |
| AgentOS English docs | 1.009 | yes (under the 5% head saving) |
| math (Δ, ≈) | 1.014 | yes |
| typographic English (curly quotes, em dash) | 1.207 | **no** |
| French probe | 1.517 | **no** |
| CJK probe | 2.873 | **no** |
| AgentOS mix (code+JSON+shell+paths+docs+math+tools) | 1.002 | yes |

q4 embed+output bytes removed: 649,068,160. Output-head FLOPs removed per token: 1,221,775,360 (2 × 119,314 × 5,120). GQA KV positions that those bytes would buy at 131,072 B/pos: ~4,952. DeltaNet state does not grow with sequence.

ASCII-only is therefore a CONTROL, not the default (§31).

## Default: hot / warm / cold residency

Do not delete rows that carry generation capability because an AgentOS
mix is English+code. Byte-fallback preserves *representability*, not
generation (§109). Tool, JSON, code, path, and schema tokens are
protected and live in HOT (§82).

| tier | what | default |
|---|---|---|
| HOT | byte alphabet, added/specials, AgentOS-observed tokens, protected tool/JSON/code/path/schema surfaces | always in the LM-head GEMV |
| WARM | remaining Latin/ASCII/Greek/math/symbol rows | on disk; page into the GEMV when the session needs them; do not delete |
| COLD | CJK / Cyrillic / Arabic / Hangul / Thai / … | on disk; page-in-on-demand; do not delete by default |

A two-stage head (GEMV over HOT every token, WARM/COLD as a gather-able
residual) removes work from every decode step (§36) without the inflation
of deleting unseen English subwords. A HOT-only *deletion* whitelist of
the observed ~5k tokens inflates even ASCII English and is forbidden.

If a campaign must delete rows, the measured alternative is *script-cold*
deletion (keep Latin/Greek/symbols, drop CJK/Cyrillic/Arabic/…): French
and typographic English stay at ratio 1.00. That still deletes cold-script
generation capability and is not the default.

## What this lane did not do

No GPU. No cargo/Metal benchmark. No 27B weight decode. tokenizer.json
only. NOETIC_PARENT_A was not opened.
