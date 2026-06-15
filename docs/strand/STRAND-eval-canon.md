# STRAND eval canon — the measurement protocol, defined once

*2026-06-11. The companion to `tools/strand_eval/` (the one canonical eval module) and
`research/results-ledger.jsonl` (the results ledger). This document states the protocol;
the module enforces it by construction. Built from audit `research/global-audit/
measurement.md` §3.1–3.3 — the "bug-factory kill". will.md stays king for strategy;
this file is the measurement reference.*

## 1. The protocol (what a PPL number means here)

- **Dataset:** WikiText-2-raw-v1, **test** split. Load via the fallback chain
  `"wikitext"` → `"Salesforce/wikitext"` (older hubs accept the bare legacy id and may
  serve it from local cache; newer hubs require `namespace/name`). Same split either way;
  every result records the **dataset id used** AND a **content fingerprint** (sha256 of
  the `"\n\n"`-joined raw text, 16 hex) so "same split" is proven, not promised.
- **Tokenization:** the model's own tokenizer over the single joined string.
- **Windowing:** **non-overlapping** ctx-token windows, `ctx = 2048`.
  - **64 windows = screening** (the cheap rung; most reopen-matrix numbers).
  - **146 windows = anchors** (the §3 canon table on the 7B; bf16 = 7.7362).
  - 64w and 146w numbers are **NOT comparable** (7B bf16: 6.629 @ 64w vs 7.7362 @ 146w).
    The chunk count is inside the harness_key, so cross-window comparison is flagged
    mechanically — never rely on the "~comparable" prose again.
- **Loss:** sum-CE over shifted logits per window, accumulated in float64 via `.item()`;
  **ppl = exp(Σnll / Σtok)**, with Σtok counting shift labels (ctx−1 per window).
- **Logits in float32** before the CE (`.float()`), model forward in the load dtype.

## 2. Dtype rules

- **bf16 is the canon dtype.** Qwen2.5 in fp16 **overflows to NaN** — the module prints a
  loud warning if you ask for fp16. fp32 is legal for QAT-side evals (recorded as such).
- The recon artifacts are bf16 (`--recon-dtype bf16` in the quant pipeline).

## 3. Device notes

- `auto` resolves cuda → mps → cpu. **Records always carry the RESOLVED device**, never
  the mode string (an eval-ppl.py divergence the audit caught).
- **mps:** eager attention is mandatory (MPS SDPA cannot broadcast Qwen GQA); the
  whole-model allocator warmup is disabled (fails on 7B even when tensor-by-tensor
  loading fits); CE runs in 512-row slices (full-vocab log_softmax over 2047×152k rows
  is a ~2.5GB transient) — CE-sum is additive over rows, so Σnll is the same quantity;
  defrag+retry once on allocator-fragmentation OOM. 18 GB is a hard ceiling: evals run
  ALONE next to nothing heavy (will.md §7 freeze trap).
- **cpu / cuda:** whole-window CE (bit-faithful to the historical heredoc; A/B-proven
  bit-identical, 2026-06-11, cpu/bf16/2ch: 11.839819038689043 old == new).
- **offload:** `device_map=auto` with a GPU budget (`--gpu-gb`, or env `EVAL_GPU_GB`,
  default 21 GiB) — for models larger than VRAM (the 14B/32B/70B pod legs).
- PPL is a float forward pass: it is **not** bit-deterministic across devices, and is not
  supposed to be (the determinism contract governs *decode*). Hence device ∈ harness_key.

## 4. Comparability rules (the harness_key)

Every record carries `harness_key` = {module version, resolved device, dtype, ctx,
chunks, dataset id} and its 8-hex hash `harness_key8`.

1. **Two PPLs are directly comparable IFF their `harness_key8` match.**
2. The ledger checker WARNs on any model with results under multiple keys, and warns
   loudly when the same (model, tag) spans keys — those are different measurements.
3. **The 15-digit tell, as code:** two records with bit-identical ppl but different
   (model, tag, key) is a contamination/bug, every time (this caught `MP_FALLBACK=4`).
   The ledger checker makes it an ERROR; conductor's ledger hook raises a `LEDGER_TELL`
   judgment event on it.
4. Records without a harness_key (legacy jsons ingested from the pod) are visibly
   **un-provenanced** — they never silently enter a canon comparison.
5. Bill everything (will.md §5.11): the module records sidecar `effective_bpw` when the
   model dir is a recon with quantize-model sidecars.

## 5. Names and locations — by construction

- Result files are **always** `ppl_<model-id>_<tag>.json`. The writer derives the model
  id from the model dir (skipping generic leaves like `recon/`, `hf/`); a same-name
  write from a *different* model dir gets a path-fingerprint suffix instead of
  overwriting (the llama2-overwrote-qwen incident is unrepresentable).
- The module self-locates via `__file__`, proven by tests from `/`, via symlink, and
  from a copied path; a copy shipped off-repo **refuses to guess** and demands
  `STRAND_ROOT` (the `//Cargo.toml` REPO_ROOT incident dies at startup, loudly).

## 6. The pieces

| thing | where | role |
|---|---|---|
| canon module | `tools/strand_eval/` (`core.py`, `ledger.py`, `cli.py`, `qat_shim.py`) | THE eval; consolidates the 3 historical copies |
| CLI | `scripts/strand-eval` | `run` / `ledger check` / `ledger ingest` / `where` |
| ledger | `research/results-ledger.jsonl` | append-only jsonl; one line per result |
| pod shim | `ops/eval-ppl.py` | same argv contract as before, delegates to the module. **Not shipped to the pod during the live campaign** — migration is post-campaign |
| QAT shim | `tools/strand_eval/qat_shim.py` | drop-in `eval_ppl(model, eval_ch, device, tag)` for strand-qat.py (that file is owned by the PV agent; swap is 2 lines, documented in the shim header) |
| run registry | `ops/podctl.sh` (`runs`, `launch`, `stop-run`) | setsid PGIDs registered to `/workspace/strand-run-registry.tsv`; tree kills by PGID, no pattern matching |
| ledger hook | `ops/conductor.sh` pod poll | idempotent `ledger ingest` of mirrored pod results + `LEDGER_TELL` event on checker errors |
| canon driver | `scripts/strand-7b-ppl.sh` | UNCHANGED (run-frozen during campaigns); its heredoc remains the historical reference the module is A/B-locked to |

## 7. Tests

`/usr/local/bin/python3 -m unittest discover -s tools/strand_eval/tests -v`
— naming/key/ledger units + the self-location proofs (24 tests). The 0.5B smoke
(2-chunk canon-shape run on `scratch/qwen-05b`) is gated: `STRAND_EVAL_SMOKE=1`,
optional `STRAND_EVAL_SMOKE_DEVICE=mps|cpu`.

Landing gates already run (2026-06-11, idle M3 Pro 18GB, box free of science runs):
- **A/B continuity:** old heredoc vs module, cpu/bf16/2ch on qwen-05b →
  `11.839819038689043` both, bit-identical.
- **Ledger replay:** banked `scratch/pod-results/*.json` ingested → the historical
  llama2/qwen name-collision pair lands as two distinct models; 0 false errors.
- MPS smoke: ppl 11.85 @ 2w (consistent with the 12.55 canon @ 64w).
