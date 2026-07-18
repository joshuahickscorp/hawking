# Candidate B — independent audit

Branch `codex/hawking-seed-b` @ `99f1f503`, PR #29. Audited from the committed crate + real 105 MB
SmolLM fixture. Verdict: **Candidate B is a real self-contained runtime.**

## Static proofs (source + dependency graph)
- **No `hawking-core`**: the only occurrences of the string are comments documenting its absence; the
  dependency tree is `half, serde, serde_json, sha2, thiserror` — no predecessor crate, no GGUF/tokenizer/
  linear-algebra library.
- **No subprocess**: zero `Command::new` / `exec` / `spawn` — Candidate B never shells out to the `hawking`
  binary or anything else.
- **No hardcoded golden**: the completion ids (7042/38634/…) and sha `2d1559cf` appear nowhere in `src/`.
  The golden lives only in `seed_b_golden.json`, loaded at runtime and compared.
- **No prompt special-casing**: "The capital of France is" is a CLI constant; the runtime (`model.rs`,
  `tokenizer.rs`, `quant.rs`) never branches on the prompt text.

## Dynamic proofs (behavior)
- Parses actual GGUF metadata and reads actual tensors (30-layer config extracted from the file).
- Executes every transformer layer: the adapter emits a 484-op IR the runtime interprets per token.
- Generates from **calculated logits** — 16 **distinct** per-step logit checksums captured (a constant or
  hardcoded table could not produce 16 different sha256 logit vectors that still yield the golden tokens).
- **Variable prompt** → different valid output: `"Once upon a time"` → `".\n\nThe time has passed,"`.
- Deterministic (same prompt → same output) and length-consistent (decode-4 prefixes decode-16).

## Adversarial tests (`tests/audit.rs`, 6, all green)
| case | result |
|---|---|
| different prompt | different valid sequence |
| different token count | length-N greedy prefixes length-M |
| altered model byte (output_norm) | output changes |
| corrupt quant block (Q6_K scale) | different dequantized values |
| tokenizer multi-prompt | deterministic + ASCII round-trip |
| per-step logit checksums | deterministic, distinct across steps |

Note: an initial altered-byte test flipped one byte in an **unused special-token embedding row** and the
output did **not** change — correct localized behavior, not a defect. The test now corrupts `output_norm`
(on the critical path for every logit) and the output changes as required.

## PR #29 state
OPEN, was `CONFLICTING/DIRTY`. Cause: B branched from the pre-Seed reference `hawking-pre-seed-final`
(714294ac); `main` (f1369745) later added a Phase-1 section to `NUCLEAR_PASTA_LEDGER.md` → a **docs-only**
conflict, plus a stale `Cargo.lock`. No source conflict. Resolved by syncing the lockfile and merging main
(keeping both ledger sections). **Not merged** — per the directive, B stays independent until the A/B/C
comparison selects the winner. Repo has no CI configured (checks empty).

## Verdict
Candidate B genuinely executes SmolLM-135M on its own runtime with the claimed quantization decoders, and
its golden parity is corroborated by multi-step logit checksums — it is sound to use as a design reference.
