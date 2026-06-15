# Handoff — correctness + security review of the overnight-haul branch

> Paste this whole file as the opening prompt for a FRESH Claude Code session
> (local or cloud). It is self-contained. The session running the build lanes is
> busy on the GPU — this review is **static / read-only** and runs in parallel
> with zero conflict. Do NOT run GPU benches (the GPU is contended); `cargo test
> -p dismantle-core --lib` (CPU) is fine if you want to spot-check.

## What dismantle is
A from-scratch Rust + Metal single-stream inference engine for Qwen2.5-3B-Q4_K_M
on an M3 Pro (18 GB). Decode is bandwidth-bound; the strategy doc is
`plans/throughput_bible_2026_05_30.md` (read §0, §1, and the new §3.0 status
correction). Branch: `codex/maximal-spec-colab`. Commits are authored
`Joshua Hicks <joshuahicksboba@gmail.com>` with **no AI attribution** — preserve
that convention if you commit anything (you should NOT; this is review-only).

## Why this review now
An overnight autonomous haul landed ~21 commits (manifest +
per-step log: `plans/overnight_build_queue_2026_05_31.md`). **Two levers are
about to be flipped DEFAULT-ON**, so they need a correctness/security pass first:
1. **f16-scales predec GEMV** (`DISMANTLE_QWEN_PREDEC_F16SCALES`, commit
   `0899137`) — +6-9% decode, but **NOT bit-identical** (f16 scale rounding).
2. **In-RAM prefix cache** (`DISMANTLE_QWEN_PREFIX_CACHE`, commit `ebfc57a`) —
   ~84% prefill cut, claims **bit-identical KV reuse**.

## Your task — review the haul commits for CORRECTNESS bugs + SECURITY issues
Scope (review the diffs; `git show <sha>` / `git diff <sha>^..<sha>`):

**PRIORITY 1 — the two default-on candidates:**
- **f16s (`0899137`):** Is the flag-OFF path *truly* byte-identical to before (default decode unchanged)? Is the f16 scale table built correctly (the f16 widening matches the kernel's `(float)half` read)? Does `gemm_q4_k_v4_predec_pair_f16s` preserve the exact FMA order of the f32 pair kernel except for the scale precision? Any shape/edge case (rows%16, ragged tail) mishandled? Is rel-L2 2.5e-4 a safe ship bar, or could a pathological input drift the greedy argmax?
- **prefix cache (`ebfc57a`, `crates/dismantle-core/src/stateful/prefix_cache.rs` + `qwen_dense.rs`):** Scrutinize the **bit-identical-reuse guarantee** — is `PrefixKey{model_hash, tokenizer_hash, prefix_hash, n_tokens}` collision-safe (could two different prefixes hash-collide → wrong KV served → silent corruption)? Is the "longest STRICT prefix" lookup correct (never returns the full prompt)? The `mirror_arena_kv_into_self` TCB-arena fix — is the restored KV exactly the recomputed KV in all paths (partial hit, eviction mid-session)? **Known latent bug to confirm:** the agent reported the on-disk `PrefillDiskCache` (`cache/prefill_disk.rs`) has the SAME TCB-arena stale-KV dependency unfixed — verify + flag severity.

**PRIORITY 2 — a bug CLASS Lane 2 surfaced:**
- Commit `e0fdf80` fixed `gemv_q3_k_pinned_tcb` writing rows/cols as two `set_bytes` when the shader wanted ONE `ArgbufRowsCols` struct (→ all-zero output, hit by Mixtral Q3_K). **Audit the OTHER `*_pinned_tcb` GEMV wrappers in `src/kernels/mod.rs`** for the same `set_bytes`-vs-struct / buffer-index mismatch class.

**PRIORITY 3 — security (local single-user, so low bar, but scan):**
- Prefix cache writes KV to disk — any path traversal / unsafe deserialize / cache-poisoning via the cache dir or key? Env-flag parsing safe? `unsafe` blocks in the new code sound (the `msg_send` GPU-timestamp reads, any `from_raw_parts`)? No secrets/paths leaked in logs.

## Method + constraints
- **Static review** (read code). The GPU is busy — do NOT run benches or GPU
  tests. `cargo test -p dismantle-core --lib` (CPU) is OK to spot-check.
- **Read-only:** do NOT edit source, commit, or push. (If you find a bug, describe
  the fix; don't apply it.)
- The kernel-microopt track is CLOSED (bible §3.0) — don't propose decode-kernel
  optimizations; this is a correctness/security pass only.

## Output
A findings list, each: **[severity: high/med/low] file:line — issue — why it
matters — suggested fix**. Lead with anything that could make a DEFAULT-ON lever
silently corrupt output (the bit-identical-reuse + flag-off-unchanged guarantees
are the load-bearing claims). End with a GO / GO-WITH-FIXES / NO-GO recommendation
for flipping f16s and prefix-cache default-on. Keep it under ~600 words.

## Pointers
- Manifest + closeout: `plans/overnight_build_queue_2026_05_31.md`
- Strategy + the §3.0 correction: `plans/throughput_bible_2026_05_30.md`
- Dead levers (don't re-litigate these): `reports/dead_levers.md`
- Commit range to review: `git log --oneline f119791..HEAD` (the post-push haul) —
  prioritize `0899137`, `ebfc57a`, `e0fdf80`, `0e6eb14`.
