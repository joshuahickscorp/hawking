# Qwen3.8 shared sessions — measured 2026-08-16

Process-pool children do **not** share artifact pages. Four
`ascension_qwen38_hybrid_greedy` children of `uniform-q4-v1` at
`--max-seq-len 2048` (the number this contract was given, not re-derived):

| quantity | value |
|---|---|
| children alive | 4 |
| sum per-process RSS | 35.09 GB = 8.77 GB/child |
| machine-wide consumed | 40.67 GB = 10.2 GB/child |
| free at the load spike | 0.37 GB |
| free once warm | 5.11 GB |

That design tops out around 6–8 children and dies on the **simultaneous load
spike**. It is the fallback, with staggered spawn (see `lab/genesis_pool.py`).

## Primary: one process, N sessions, one resident weight set

`Qwen38HybridDecodeSession::attach(Arc<Qwen38HybridWeights>, max_seq_len)`
allocates only workspace / KV. `open` is load + attach. There is no
code-level blocker: Metal weight buffers are `metal::Buffer` handles, the
context is `Clone` via `Arc`, and `generate_greedy_parallel` compiles
because the session is `Send`.

Measured this box, `uniform-q4-v1`, `--max-seq-len 128`, 4 attached
sessions, GPU lock held
(`receipts/ascent-2026-08-16/QWEN38_SHARED_SESSIONS.json`):

| quantity | measured |
|---|---|
| `Arc` shared | true (ptr_eq) |
| resident weight buffers | 14,297,675,776 B |
| workspace formula | 175,361,796 B |
| RSS after load | 15,120,416,768 B |
| RSS after session 0..3 | 15.294, 15.468, 15.642, 15.815 GB |
| **marginal RSS / session** | **173,703,168 B** (identical on each of 3 deltas) |
| 1-session steady tok/s | 26.653 |
| 1-session median GPU | 36,099,333 ns |
| 4-session aggregate tok/s | 9.427 (worse than 1) |
| lm_head 1× GPU | 1,013,791 ns |
| lm_head 4× serial GPU | 4,144,541 ns = 4.09× |
| lm_head 4× concurrent GPU | 4,022,124 ns = 3.97× |

Sharing weights **saves memory**. It does **not** amortize DRAM. Four
independent GEMVs against the same `lm_head` still pay ~4× GPU time,
concurrent encoder or not. Four in-process decode threads on one command
queue **slow each other down** (26.7 → 9.4 aggregate tok/s).

## Child ceiling (this design)

Two different ceilings:

1. **Resident sessions** (KV live, step one at a time): marginal 174 MB at
   seq=128, 427 MB workspace at seq=2048. After one 15.1 GB load and a 4 GB
   no-swap reserve, this box admits on the order of **~120 sessions at
   seq=128** / **~40 at seq=2048**. The gate refused 144 seq=128 sessions
   (`QWEN38_ADMISSION_REFUSE_SESSIONS.json`).
2. **Concurrent decode**: **1**. Extra in-process sessions do not raise
   tokens/s. The process-pool fallback measured a 1.37× knee at N=2
   independent processes and a swap at N=4; that is not the primary path.

## What parallelizes, what is lock-bound

| work | parallel? | lock? |
|---|---|---|
| gravity-recipe (TEXT generate) | many resident sessions; **step serially** | no (flag recorded) |
| kernel-floor (GPU timestamps) | no — one session | yes (`--lock-held`) |
| weight load | once per process | n/a |
| process-pool fallback | spawn **staggered**, run concurrent | TEXT vs TIMING as above |

## Admission

Refuses before swap (`free - cost < 4 GiB`). Demonstrated:

- `QWEN38_ADMISSION_REFUSE.json` — over-subscription refuse, no model
- `QWEN38_ADMISSION_REFUSE_SESSIONS.json` — 144 shared sessions refuse
- attach path in the probe would have refused the same way

Process-pool fallback still uses the measured 10.2 GB/child machine cost
and must spawn one-at-a-time.

## Verify

```
cargo build --release -p hawking-core --example ascension_qwen38_hybrid_greedy
cargo test -p hawking-core --release qwen38
```
