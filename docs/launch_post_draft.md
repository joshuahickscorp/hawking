# r/LocalLLaMA launch post draft — dismantle v2.0

Two versions below — pick whichever feels right. Both lead with the
honest perf number to avoid the "yet another llama.cpp killer" reception
pattern.

---

## Version A — short, technical, lower-stakes

**Title:** dismantle v2.0 — pure-Rust MoE inference engine for Apple Silicon

**Body:**

I built a Mixture-of-Experts inference engine in Rust + Apple Metal —
no Python at runtime, no llama.cpp dependency. Single binary. Loads
GGUF weights via mmap and runs them through hand-rolled Metal kernels.

Honest perf on M3 Pro 18 GB:
- DeepSeek-V2-Lite Q4_K_M: **~17 dec_tps** (95% CI [17.25, 17.42])
- Mixtral 8×7B Q3_K_M: ~0.1 dec_tps (functional but SSD-bandwidth-bound
  on 18 GB; better on 32+ GB Macs)

llama.cpp Metal on the same machine is roughly 3× faster on V2-Lite.
This release isn't competitive on raw throughput. It's a working
auditable Rust codebase to build on.

What's there:
- MoE-first architecture (DeepSeek-V2 family + Mixtral)
- OpenAI-compatible HTTP API (`dismantle serve`)
- Reproducible kernel autotune that picks per-shape Q4_K kernel variants
- Reproducible benchmarking — every perf claim above is repeatable with
  one command, with statistical CIs
- Memory-conscious expert dispatch (`--max-routed-expert-ram-mb` budget)
- All written for Apple's hardware specifically

What's NOT there:
- Speculative decoding that actually delivers e2e wins (infrastructure
  is correct but batched verify is sequential-in-CB on current
  architecture; documented in repo)
- Apple Neural Engine integration (planned, post-v2.0)
- Match for llama.cpp's bandwidth utilization (~50% vs our ~20%)

GitHub: https://github.com/joshuahickscorp/dismantle

Posting at honest baseline rather than chasing a number. Open to PRs
on the perf gap.

---

## Version B — longer, narrative, more story

**Title:** I spent two months building a pure-Rust MoE inference engine for Apple Silicon. Here's the honest reckoning.

**Body:**

dismantle is a Mixture-of-Experts inference engine I built in Rust + Apple
Metal over the past two months. Single binary, no Python at runtime, no
llama.cpp dependency. It loads GGUF weights via mmap and runs them through
hand-rolled Metal compute kernels.

GitHub: https://github.com/joshuahickscorp/dismantle

**The honest number:** ~17 dec_tps on DeepSeek-V2-Lite Q4_K_M on an
M3 Pro 18 GB. llama.cpp Metal on the same hardware does ~60. That's a
3× gap. dismantle is not competitive with llama.cpp on raw throughput.

**Why post anyway:** the architecture and methodology are interesting
even if the numbers aren't yet. I've measured exactly where the gap
comes from (we're at ~20% memory bandwidth utilization vs llama.cpp's
~50% — see `reports/v1.1.0_architecture_audit.md` in the repo) and
documented every dead-end I hit along the way. If you've thought about
building a Rust ML runtime for Apple Silicon, this might be useful as
either a starting point or a cautionary tale.

**What's actually working:**
- DeepSeek-V2-Lite Q4_K_M generation, stable, ~17 dec_tps
- Mixtral 8×7B Q3_K_M loads and generates on 18 GB Macs (slow due to
  SSD page-faults on cold expert weights — much faster on 32+ GB)
- OpenAI-compatible HTTP API, single binary, no Python
- Reproducible kernel autotune with per-shape Q4_K variants
- Statistical bench harness with CIs and cross-commit diffing in tree
- MoE-aware expert offloading via `posix_madvise` budget caps

**What I learned the hard way:**

1. **Per-kernel optimization hits a ceiling.** I ported llama.cpp's Q4_K
   GEMV faithfully, added MPSGraph for the LM head, did fp16 KV cache,
   built a top-K LM head. Each won 0-5% in isolation. Combined: +0.86%
   e2e. The bench-first commit gate caught it; nothing shipped as
   default. But it cost weeks.

2. **Speculative decoding without K-parallel verify doesn't work.**
   N-gram spec at α=1.0 (perfect 51/51 acceptance on a repetitive
   prompt) gave -0.5% e2e — flat. Root cause: the "batched" verify
   processes K tokens sequentially within one Metal command buffer.
   Real spec requires K-parallel compute. That's a multi-week refactor
   I haven't done.

3. **The TCB trace is gold.** When I added per-kernel timing inside the
   single command buffer, it instantly revealed that 69% of decode is
   the TCB compute (MoE + attention) and 30% is the LM head — and MoE
   GEMV is 0.1% of trace-visible dispatch time. Most of my "MoE
   optimization" work after that was confirmed wasted effort.

**What's next:** post-v2.0, Apple Neural Engine integration for the LM
head + attention projections. ANE has 16 cores designed for f16 matmuls
and llama.cpp doesn't use it. If I can plumb it through Core ML, that's
the next architectural lever. No promises on timeline.

**Reproduce the numbers:**
```sh
git clone https://github.com/joshuahickscorp/dismantle
cd dismantle && cargo build --release --workspace
./tools/fetch-model.sh
TRIALS=4 TOKENS=24 bash tools/bench/coexist_bench.sh
```

The bench reports median + 95% CI + IQR. Tell me if your number is
different from mine and I'll dig in.

MIT licensed, single Mac developer, weekends and evenings. Feedback
and PRs welcome — especially on the bandwidth utilization gap.

---

## Notes for posting

**When to post:**
- Tuesday or Wednesday morning Pacific (avoids Friday afternoon dump)
- After the GitHub release tag is up so the link is real
- After README + CHANGELOG are committed and on main

**Comments to prep for:**
- "Why not use llama.cpp?" → "I wanted a pure-Rust runtime; this is
  also a learning project; the architecture is documented in the repo"
- "What's the point if it's slower?" → "Single binary, no Python, MoE
  arch from the kernel level up, methodology is reproducible. If
  none of those matter to you, llama.cpp is the right answer."
- "Apple Silicon perf comparison?" → "see README perf table; CIs
  documented; reproduce with one command"
- "When will it match llama.cpp?" → "honestly unsure. ANE integration
  is the next architectural lever I want to try. No promises."

**Don't engage with:**
- "rewrite it in C++" / language wars
- "use MLX instead" arguments (different goal)
- requests for benchmarks against models we don't support yet

**What success looks like:**
- 50+ upvotes (modest target — this isn't a perf-leader story)
- Constructive comments about the architecture
- A few people clone and try it
- Someone files an issue or PR about a kernel optimization
- Discussion thread that doesn't devolve

Anything beyond that is bonus.
