# Measured frontier input — G002's producer states something no longer true — 2026-09-05

## THE CLAIM

`tools/sovereign/g002_overhead.py --attribute` reports all twelve stages unattributable and
explains why:

    "model_calls[] persists only wall_s/prompt_tokens/completion_tokens. The connector's
     prefill/decode/gpu_ns envelope is computed per call and discarded. With zero
     pure-prefill samples, prefill and decode are not separable: a least-squares split of
     wall on (prompt, completion) returns a negative fixed term, which is unphysical."

## THE MEASUREMENT THAT CONTRADICTS IT

Verified in `.hcli/receipts/46878d6b-4f2c-4dc6-907a-a3664bb9829d.json`, model_calls[0] keys:

    prefill_profile          -> totals: wall_ns, gpu_ns, encode_ns, submit_ns, wait_ns,
                                        dispatches
    prefill_tokens_stepped
    prefix_reused_tokens
    prefix_source
    wall_s
    completion_tokens

So `native_prefill_ns` is measured DIRECTLY (prefill_profile.totals.wall_ns) and
`native_decode_ns` follows by subtraction from wall_s. The least-squares split the producer
calls unphysical is not needed for those two stages.

Two of the twelve stages are attributable from data already on disk. The producer says zero.

## THE TASK

Make `--attribute` report `native_prefill_ns` and `native_decode_ns` from the receipts when
`prefill_profile` is present, and list only the genuinely unattributable stages as
unattributable. Update `why_unattributable` so it describes what is actually missing rather
than what used to be.

Do NOT invent the other ten. They are HCLI-side stages with no counters yet, and reporting a
zero for an unmeasured stage is exactly what that gate exists to catch. The receipt must stay
honest: the producer already REFUSES to emit a partial receipt, and that refusal must survive.

## CONSTRAINTS

- `tools/sovereign/g002_overhead.py` is a PRODUCER, not a gate. It is not in
  receipts/sovereign/VERIFIER_MANIFEST.json and may be edited.
- `hcli/test_hcli_overhead.py` IS protected. Do not touch it.
- Smallest executable change. One operation.
- Name a test that fails before your change and passes after it.
