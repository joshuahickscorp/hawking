# Hawking — Summary Seed for a New Chat

Paste this into a fresh chat to request a full project-standing review.

---

You are picking up **Hawking**, an Apple-Silicon-first Rust/Metal LLM inference runtime at
`/Users/scammermike/Downloads/hawking`. A long autonomous campaign just finished a condensation pass. Before doing
anything, read the standing packet in this order:

1. `docs/campaign/hawking_ship_finalization_prompt.md` — canonical ship-finalization prompt and rails
2. `docs/plans/hawking_shippability_masterplan_2026_06_22.md` — P1/P2/P3 shippability roadmap
3. `docs/campaign/project_standing_snapshot.md` — one-page state + speed table + what's ready/experimental/blocked/dead
4. `docs/campaign/open_risks_and_gates.md` — every open risk, the evidence to close it, the best next gate
5. `docs/campaign/commit_plan.md` — the dirty tree split into clean commit lanes (NOT yet committed)
6. `docs/campaign/pruning_inventory.md` — doc condensation candidates (no code deletion)
7. `docs/campaign/roadmap.md`, `kill_ledger.md`, `test_matrix.md`, `findings_summary.md` — evidence + rejected ideas
8. `docs/campaign/ssm_model_selection.md`, `ssm_productionization.md` — the SSM product path
9. `docs/plans/condense_frontier_2026_06_22.md` — post-finalization Model Press frontier:
   memory-budgeted out-of-core quantization/condensation of too-large open-weight parents
10. `docs/plans/condense_naming_migration_2026_06_22.md` — rename the public legacy STRAND surface to Condense

**The 60-second standing:**
- **Validated moat:** the RWKV-7 SSM path removes the transformer long-context KV wall — flat ~114–119 tps to 8k vs
  Qwen-3B's ~8.6 tps (**~13–14× at 8k**). This is the strategic differentiator.
- **Qwen transformer path is mature:** `predec` is the headline decode win (−46.7% when off); `--profile fast` only
  ~+3–7% (noisy, mild trade); F16_KV is a −50% KV footprint / long-ctx lever. Decode kernels are near the Apple
  memory-model optimum (bible §3.0). Spec-decode is dead for speed (overhead wall).
- **RWKV serve correctness is fixed:** admission, recurrent-state handoff, and stream termination are all green;
  `ssm_serve_smoke.sh` is fail=0 with coherent output. Keep
  `rwkv7_prefill_slot_multiseq_parity` green as the proof.
- **The one open fix that unlocks the most runtime value:** RWKV **serve throughput**. Serve decodes about 7.8 tps because
  the B=8 multiseq arena does 8-stream work for 1 active stream; single-stream `generate` is about 119 tps. Size decode work
  to active slots, then rerun parity, `ssm_serve_smoke.sh`, and `ssm_product_gate.sh`.
- **The parallel product gate:** valid instruct quality via `/v1/chat/completions`; raw `hawking generate` is not a
  chat-template quality gate.
- **The post-finalization expansion:** Condense. Hawking should become able to plan, stream, quantize,
  recover, and verify 4/3/2/1-bit artifacts from parents that cannot be fully resident on the user's machine. Start with
  dry-run planning and small-model out-of-core proofs; GLM-class downloads/cloud runs need owner approval. Publicly call
  this low-bit line **Condense**; treat STRAND as legacy/internal until aliases and gates exist.

**Hard rails (persist these):**
- No destructive git; preserve user/agent changes; commit only narrow reviewed lanes (see `commit_plan.md`); never stage `reports/`.
- **No git AI-attribution trailers** — commits must look human-authored.
- Do NOT run global `cargo fmt` (repo-wide pre-existing drift; would balloon every diff).
- Keep GPU jobs sequential (check `ps` first). RWKV/Metal state + Qwen hot paths + quant wiring are higher-care — gate before changing.

**Good first request to make:** "Read `docs/campaign/hawking_ship_finalization_prompt.md`, give me the current Hawking
project standing, then either (a) attack RWKV serve throughput while keeping parity green, or (b) execute Lane 1–3 of the
commit plan."
