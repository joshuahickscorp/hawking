# Test Matrix — what was measured, how, result (autonomous campaign, 2026-06-21)

Harness: `./target/release/hawking generate --weights <M> --prompt <P> --max-new-tokens N --temperature 0 --seed 5`,
`HAWKING_QWEN_USER_DRAFT=0`, warm = median of ≥3-5 fresh runs (PSO warms across runs). Model `M` =
`models/qwen2.5-3b-instruct-q4_k_m.gguf` unless noted. Quality = argmax-identity (greedy diff vs the bit-identical default).

## Speed — config levers (warm tps)
| Config | ctx | tps (warm median) | Δ vs default | quality | verdict |
|---|---|---|---|---|---|
| default | short | 38.3–40.6 | — | (reference) | baseline |
| default (COLD 1-run) | short | 29.7 | — | — | ⚠️ PSO-compile, not steady-state |
| `--profile fast` | short | ~41.2 | **~+3–7%** (noise floor) | 83% identity (12 prompts), 90% on adversarial 10 | ✅ speed-priority, mild trade |
| `FFN_DOWN_Q4K=1` | short | 39.6 | +1.4% (noise) | 100% identity (30 prompts) | ❌ no warm gain |
| f16-scales | short | 40.7 | +0.3% | identical (3 prompts) | opt-in, ~0% |
| Q6_K ffn_down **1r** (`Q6K_SWIGLU_2R=0`) | short | 39.9 | −1.4% | bit-identical | slower |
| Q6_K ffn_down **2r** (default) | short | **40.5** | best | bit-identical | ✅ optimal default |
| Q6_K ffn_down **4r** (`Q6K_SWIGLU_4R=1`) | short | 40.0 | −1.1% | ~1e-4 | not better |

## Config attribution + predec verify (warm 5-trial median A/B) — NOISE-FLOOR FINDING
| Config (vs default) | Δ | note |
|---|---|---|
| `HAWKING_QWEN_Q4K_PREDEC=0` (predec OFF) | **−46.7%** (40.4→21.6) | ✅ **predec is the headline decode win (~2× when on)**; already default-on. Clean, well above noise. |
| `--profile fast` | +2.5% … +7.5% | real but small: fast stably ~41.2 vs default's noisy 38–40 → ~+3–7%, NOT a clean +7.5% |
| `VOCAB_PRUNE=1` | +7.6% (1 run) | largest `fast` component, but at the noise floor |
| `Q4K_LMHEAD=1` | +0.4% | ~0 |
| `PREDEC_F16SCALES=1` | nominal +32% — **but the default arm was COLD (30.5) = artifact**, not a real +32% |
| `VOCAB_PRUNE + Q4K_LMHEAD` | −12.3% | inconsistent (cold cfg arm) = noise |

⚠️ **NOISE FLOOR (method finding):** the `default` baseline drifted **30.5–40.4** across rows — the fresh-process 5-trial
median has a **±several-% noise floor** (per-process cold PSO residual). **Sub-10% config deltas are NOT cleanly resolvable
this way.** Only predec (−47%) and the architecture moat (RWKV-7) are above the floor. For fine (<10%) deltas, use a SINGLE
long warm run (many tokens in one process), not N fresh processes. This refines (downgrades) the earlier "+7.5% clean" claim.

## Compression — KV levers
| Config | ctx | tps | footprint | quality | verdict |
|---|---|---|---|---|---|
| default (f32 KV) | short / 2.5k / 8k | 40.6 / 18.8 / 8.6 | 0.28 GiB @4k | — | KV wall = 4.6× drop |
| `F16_KV=1` | short | 40.2 | −50% KV | 88% identity (8 prompts) | ✅ footprint, ~0% short tps |
| `F16_KV=1` | ~2.5k | 19.1 | −50% KV | — | +1.9% (scales with depth) |
| `INT4_KV=1` (per-ROW) | ~2.5k | 17.7 | −75% KV | **0% identity** | ❌ slower + collapse |
| per-CHANNEL int4-KV | — | — | −75% KV | cosine 0.998 (real K/V) | 🔵 wiring (not yet e2e) |

## Architecture (long-context regime)
| Model | short tps | ~2.5k | ~8k | note |
|---|---|---|---|---|
| Qwen2.5-3B-Q4_K_M | 40 | 18.8 | 8.6 | transformer KV wall (4.6×) |
| RWKV-7 0.4B (`rwkv7-g1-04-sft`) | **118.6** | **110.6** | **119.4** | **FLAT — SSM, no KV wall; ~14× faster than Qwen @8k** |
| mamba2-370M | ~11 | ~11 | ✗ FAIL | **FLAT short→mid (10.6→10.4 / 11.7→11.7 across runs)** corroborates the SSM property across a 2nd family. 8k returned 0.00 = a bug in the UNOPTIMIZED mamba2 long-ctx kernel path (a pure SSM has NO context limit, so it's not a model cap). Absolute ~11 tps = unoptimized kernel, not the model. **Moat stands on RWKV-7** (primary SSM, 8k ✅); mamba2 8k-fix is a separate, low-priority kernel task. |

**Moat magnitude @8k ctx:** RWKV-7 119 vs Qwen 8.6 = ~14× (size + flatness). Flatness (the SSM property) is the structural
differentiator: RWKV-7 short→8k is ~0% change; Qwen short→8k is −78%. The long-context product belongs on the SSM path.

## Quality suites run
- `--profile fast` vs default: 12 diverse prompts (TCP/UDP, math, haiku, SQL, regex, linked-list, history, geography, bash) → 83% argmax-identity.
- `FFN_DOWN_Q4K`: 30 prompts (12 + 18 adversarial @160 tok) → 100% argmax-identity.
- `F16_KV`: 8 prompts → 88%. `int4-KV` per-row: 6 prompts → 0%.
- **Adversarial suite** (10: code/math/JSON/multilingual/haiku/primes/SQL/TCP/regex/sort, via `tools/bench/ratios.sh qual`):
  `--profile fast` = **90%** (9/10); `F16_KV` = **100%** (10/10). Harness self-validated. → `--profile fast` is a confirmed
  mild trade (~85-90% across suites); `F16_KV` quality is high (divergence is rare/prompt-specific).

## Validation status / remaining gaps
- **Lib unit suite RUN: 182/182 PASS** (0 failed, 1 ignored). Re-run in CPU-only hardening report
  `reports/overnight/20260622T020448Z/` finished in 524.85s — core engine logic validated post-crash.
- Parity subset **RUN, 3/3 PASS** (post-crash codebase validated): `perchannel_int4kv_survives_outliers_qwen_geometry` ✅
  (the deferred int4-KV-PC scheme's numerics are sound — survives the outlier case that killed per-row); `q6k_swiglu_2r`
  **bit-identical to 1r** ✅; `q6k_swiglu_4r` ≈ 2r ✅. Re-run in `reports/overnight/20260622T020448Z/`:
  all 3 PASS. Full suite still pending (many GPU model-load tests = expensive).
- Real-model **perplexity** gate for per-channel int4-KV — **pending** (the documented ship gate, never run).
- Long-ctx (8-16k) tps for per-channel int4-KV once wired — **pending**.
- Adversarial quality suite **RE-VERIFIED** (10 prompts: code/math/JSON/multilingual/format/edge): `--profile fast`
  **90%** (9/10), `F16_KV` **100%** (10/10) argmax-identical. Via `tools/bench/ratios.sh qual`.
- f16-x GEMV, GQA-coalesced MHA, MLX-diff prototype benches — **design stage**.

## Serve-path validation
| Gate | model | result | evidence | note |
|---|---|---|---|---|
| SSM serve smoke | `models/rwkv7-g1-04-sft-Q4_K_M.gguf` | **FAIL/ATTENTION** | `reports/serve-smoke/20260622T022233Z/` | `hawking serve` loads the model; `/healthz`, `/v1/models`, and `/metrics` pass. Native `/v1/hawking/generate` timed out after 180s. Metrics after request: queued=1, admitted=0, tokens=0. Top SSM production bug: queued request is not admitted/decode-stepped. |
| SSM serve smoke (post-fix) | `rwkv7-g1-04-sft` | **ADMISSION FIXED, decode bug remains** | `reports/serve-smoke/20260622T023814Z/` | Added RWKV's 3 batch trait methods (`encode_prompt_for_batch`/`decode_token_for_batch`/`eos_id_for_batch`, rwkv7.rs). Now `admitted=1` (was 0), `queued=0`, **no 180s hang** — request admitted + decode-stepped in ~7s. **Remaining (separate, higher-care):** produces 1 token, empty text (immediate EOS) vs `generate`'s coherent 119-tps → multiseq serve decode / prefill-state-handoff or EOS/template bug. Flagged in run-log Phase 5. |
| RWKV prefill→multiseq parity gate | `rwkv7-g1-04-sft` | **✅ PASSES (R1 FIXED)** | `tests/rwkv7_prefill_slot_multiseq_parity.rs` | Pre-fix reproduced the bug (`multi=[37138,45,21265]`≠solo); **post-fix GREEN both slots: solo==multi==[37138,47,11]**. Root cause was the **stale bundle-wide `fresh` flag** (NOT a layout bug — proven identical); fix clears `g.fresh=false` in `prefill_slot` after the state copy (`rwkv7.rs:1222`). Run: `cargo test --release -p hawking-core --test rwkv7_prefill_slot_multiseq_parity -- --ignored --test-threads=1`. |
| SSM serve smoke (FIXED) | `rwkv7-g1-04-sft` | **✅ fail=0 — coherent e2e** | `reports/serve-smoke/20260622T121046Z/` | With R1 + the SSE stream-terminator fix (`{stats}`/`[DONE]` on any stream end): **16 coherent tokens** (" Paris. The capital of the United Kingdom is…") + `{stats}` + `[DONE]`; admitted=1, tokens_generated=16, queued=0. NB serve dec_tps ~7.8 = B=8-arena overhead (perf follow-up R1b). The "Summarize…" prompt yields 1 eos token = raw-completion chat-template gap (R3), not a bug. |
