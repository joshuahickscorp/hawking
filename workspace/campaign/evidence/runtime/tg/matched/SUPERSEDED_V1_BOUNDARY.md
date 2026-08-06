# Matched Llama receipts: v1 boundary notice

`hawking_ctx2048.json`, `llama_cpp_ctx2048.json`, `hawking_ctx8192.json`,
and `llama_cpp_ctx8192.json` are non-certifying diagnostic receipts only.
They are retained because their greedy output agreement is useful, but the
v1 llama.cpp runner timed the GPU evaluation plus argmax without token-ID to
text-piece conversion. Hawking's `generate` timing includes that streamed
token work. The runner was corrected in v2 before any performance promotion.

No TPS ratio from v1 may be called parity, surpass, ship, dominance, or
moonshot. Both v1 and v2 remain blocked until current-context K0 and resource
instrumentation are complete.
