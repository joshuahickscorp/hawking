# Hawking — Summary Seed for a New Chat

Paste this into a fresh chat to request a full project-standing review.

---

You are picking up **Hawking**, an Apple-Silicon-first Rust/Metal LLM inference runtime at
`/Users/scammermike/Downloads/hawking`. A long autonomous campaign just finished a condensation pass. Before doing
anything, read the standing packet in this order:

1. `docs/campaign/project_standing_snapshot.md` — one-page state + speed table + what's ready/experimental/blocked/dead
2. `docs/campaign/open_risks_and_gates.md` — every open risk, the evidence to close it, the best next gate
3. `docs/campaign/commit_plan.md` — the dirty tree split into clean commit lanes (NOT yet committed)
4. `docs/campaign/pruning_inventory.md` — doc condensation candidates (no code deletion)
5. `docs/campaign/roadmap.md`, `kill_ledger.md`, `test_matrix.md`, `findings_summary.md` — evidence + rejected ideas
6. `docs/campaign/ssm_model_selection.md`, `ssm_productionization.md` — the SSM product path

**The 60-second standing:**
- **Validated moat:** the RWKV-7 SSM path removes the transformer long-context KV wall — flat ~114–119 tps to 8k vs
  Qwen-3B's ~8.6 tps (**~13–14× at 8k**). This is the strategic differentiator.
- **Qwen transformer path is mature:** `predec` is the headline decode win (−46.7% when off); `--profile fast` only
  ~+3–7% (noisy, mild trade); F16_KV is a −50% KV footprint / long-ctx lever. Decode kernels are near the Apple
  memory-model optimum (bible §3.0). Spec-decode is dead for speed (overhead wall).
- **The one open fix that unlocks the most value:** RWKV **serve decode correctness**. Admission is fixed (no more 180s
  hang), but the GPU-slot recurrent-state handoff diverges at token 2. A parity gate reproduces it exactly:
  `cargo test --release -p hawking-core --test rwkv7_prefill_slot_multiseq_parity -- --ignored --test-threads=1`
  (`solo=[37138,47,11]` vs `multi=[37138,45,21265]`; first token correct, diverges at token 2). Fixing
  `copy_cpu_state_to_gpu_slot` (or building `prefill_slot` on the GPU path) turns it green → coherent RWKV serve →
  unblocks the SSM product AND valid instruct quality eval (via `/v1/chat/completions`).

**Hard rails (persist these):**
- No destructive git; preserve user/agent changes; commit only narrow reviewed lanes (see `commit_plan.md`); never stage `reports/`.
- **No git AI-attribution trailers** — commits must look human-authored.
- Do NOT run global `cargo fmt` (repo-wide pre-existing drift; would balloon every diff).
- Keep GPU jobs sequential (check `ps` first). RWKV/Metal state + Qwen hot paths + quant wiring are higher-care — gate before changing.

**Good first request to make:** "Give me the current Hawking project standing, then either (a) attempt the gated RWKV
serve decode fix, keeping it tiny and rerunning the parity gate, or (b) execute Lane 1–3 of the commit plan."
