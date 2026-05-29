# Eagle5 qwen Phase C — diagnostic on the 0% accept rate

Continuation of `reports/eagle5_phase_c_initial_bench.md`. The initial
Q4_K_M bench showed 1.1% accept; the hypothesis was train/serve precision
shift (fp16 train → Q4_K_M serve). To test that hypothesis we downloaded
the official `Qwen/Qwen2.5-1.5B-Instruct-GGUF` F16 GGUF and re-ran with
the matched q1p5 trained head.

## Result: matched fp16 didn't help

| Config | dec_tps | draft_accepted | draft_rejected | accept rate |
|---|---|---|---|---|
| baseline F16, no spec | **33.63** | 0 | 0 | n/a |
| Eagle5 + trained q1p5 + capture (layer 22) | **6.80** | 0 | 90 | **0.0%** |

So the precision hypothesis was wrong. The matched model produces
**zero accepted drafts**, even worse than the mismatched Q4_K_M run
which got 1/90.

## Real root cause: drafts depend on residual, not prev_token

`DISMANTLE_QWEN_EAGLE5_CAPTURE_DEBUG=1` exposes the per-cycle drafts:

```
last_id="time" → drafts=["!", "!", "!", "!"]
last_id=","    → drafts=[" ", " ", " ", " "]
last_id=" was" → drafts=[" was", " was", " was", " was"]  ← head repeats input!
```

The **K-identical drafts within a cycle** are the smoking gun. Eagle5's
auto-regressive propose is supposed to use the previous draft as
`prev_token` for the next step:

```
draft_0 = head.argmax(prev=last_id, residual, intermediate)
draft_1 = head.argmax(prev=draft_0, residual, intermediate)  # same residual!
draft_2 = head.argmax(prev=draft_1, residual, intermediate)
```

Within a cycle, `residual` and `intermediate` are CONSTANT (snapshot from
the previous verifier forward). Only `prev_token` changes. So if the
head's output depends primarily on `residual`, K drafts will be ~identical.

Looking at the head's architecture (from `eagle5_train_pytorch.py:217-218`):

```python
baseline = _rms_norm(residual_in, self._output_norm, RMS_EPS)
draft_hidden = baseline.to(x.dtype) + self.residual_gate * x
```

`residual_gate` initializes at 0.05 (line 185) and may stay small even
after training. So `draft_hidden ≈ baseline = rmsnorm(residual_in)`. The
`prev_token`-influenced term `self.residual_gate * x` (where `x` is the
transformer block's output) is heavily attenuated.

**The head's `draft_hidden @ _lm_head` argmax is therefore dominated by
`residual_in`** — captures determine the prediction, prev_token has
minimal effect.

## So what predicts what?

The head was trained: given residual + intermediate captured at layer L
when the verifier processes ground-truth token `t_N`, predict `t_{N+1}`.

At inference, what my dispatch ACTUALLY captures: residual + intermediate
from the LAST forward in the verify loop. In my current Phase B.4 serial
verify:

```
for i in 0..K {
    pred = forward(tmp_last, pos+i)  // ← captures populate during this forward
    if pred != draft[i] { break }    // ← but maybe this iteration's draft was wrong
    tmp_last = pred
}
```

The captures are written by EVERY forward in the loop, so by the time
we exit, captures hold state from the **last forward** — which is
either:
- The forward that rejected the K-th draft (most often), OR
- The K-th draft itself if all accepted.

Either way, captures end up from a **draft-position forward**, not from
a verified-token forward. The next cycle's `propose` then sees these
"wrong-position captures" and outputs garbage.

The trainer captured from ground-truth runs (every position got a
correct token). So at runtime the head sees out-of-distribution captures
and predicts essentially random tokens (modulated by whatever scale the
captures land at).

## Fix: restructure verify-then-propose

The deepseek_v2 reference uses BATCHED verify, where the K+1 positions
are processed in one forward AND the captures naturally come from the
last verified position. Adapting that:

```
loop {
    // Forward at the last-verified position. Returns logits at next pos
    // AND populates captures for the head to use.
    bonus_logits = forward(last_verified, last_verified_pos)

    // Propose K drafts with FRESH captures from the verified position.
    drafts = head.propose(last_verified, captures, K)

    // Batched verify of [draft_0, ..., draft_{K-1}] at [pos+1..pos+K]
    // (or serial K-1 forwards). Compare each pred to next draft.
    // ...
}
```

This restructures the dispatch loop so the FORWARD that produces the
verifier's prediction also provides clean captures for the next propose.

Effort: 1-2 days attended. Touches the Eagle5 verify branch in
`qwen_dense.rs:1530-1620ish`. Requires verifying greedy parity preserved
+ measuring the new accept rate.

## Honest scope assessment

The 13 commits tonight shipped functionally complete Eagle5 infrastructure
— loader, forward, dispatch, batched verify, capture plumbing. All
parity tests green. Greedy correctness preserved across all modes.

But the dispatch's verify-then-propose ORDERING produces wrong captures
at runtime. The numerical-parity work was correct; the architectural
ordering wasn't. The fix is **structural** (re-order the loop), not
numerical.

The trained head from Colab is NOT bad — it's responding to its input
(different drafts across cycles). It just sees the wrong input at
runtime because of the dispatch ordering.

## What this means for the user

- **The colab training work is salvageable.** Once dispatch ordering is
  fixed, the same `q3b_eagle6_long.safetensors` and `q1p5_eagle6_long.safetensors`
  may work fine.
- **Engineering work remains.** ~1-2 days of attended work to restructure
  the dispatch loop.
- **No re-training required.** The previous "fp16 vs Q4_K_M precision"
  hypothesis was wrong; this is purely a runtime-dispatch issue.

## Debug instrumentation kept in code

`DISMANTLE_QWEN_EAGLE5_CAPTURE_DEBUG=1` env flag prints per-cycle
residual stats + draft tokens. Useful for the attended session that
restructures the dispatch loop. Cost: zero when flag is unset.
