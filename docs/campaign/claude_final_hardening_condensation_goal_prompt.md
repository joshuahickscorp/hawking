# Claude Final Hardening / Pruning / Condensation Goal Prompt

Copy/paste this into Claude for one last autonomous pass before stopping the
loop and asking for a clean project-standing summary.

````text
/goal Final Hawking hardening, pruning, and condensation pass in:

  /Users/scammermike/Downloads/hawking

Mission:
Convert the current fast-moving Hawking campaign into a clean, defensible project
state. Do one last long-shot pass that prioritizes production hardening,
evidence consolidation, safe pruning inventory, stale-doc cleanup, and an
operator-ready project-standing packet. Keep making progress, but do not chase
ungated hot-path changes or delete code without proof and approval.

Read first:
1. docs/campaign/autonomous_run_log.md
2. docs/campaign/test_matrix.md
3. docs/campaign/ssm_model_selection.md
4. docs/campaign/ssm_productionization.md
5. docs/campaign/change_manifest.md
6. docs/campaign/roadmap.md
7. docs/campaign/claude_moonshot_goal_prompt.md
8. docs/env_flags.md
9. docs/architecture.md

Current numbers to preserve:
- Qwen2.5-3B Q4_K_M transformer path:
  - short context: ~38-41 tps warm
  - ~2.5k context: ~18.8 tps
  - ~8k context: ~8.6 tps
  - long-context drop: ~4.6x / about -78%
  - `--profile fast`: only ~+3-7%, noisy, mild quality trade
  - `predec`: the real Qwen decode win; turning it off is about -46.7%
  - F16_KV: -50% KV footprint; quality generally high; speed mostly footprint/long-context lever
  - per-row int4-KV: rejected, slower and quality collapse
  - per-channel int4-KV: numerics pass but not wired end-to-end
- RWKV-7 0.4B SSM path:
  - original moat: 118.6 / 110.6 / 119.4 tps short / mid / ~8k
  - latest overnight matrix: about 114.6 / 113.9 / 110.5 tps
  - interpretation: flat long-context decode, about 13-14x Qwen at ~8k
- mamba2-370M:
  - short/mid around ~11 tps
  - long/8k returns 0.00 in the matrix: unoptimized/buggy long path, not product-ready
- Serve state:
  - RWKV serve admission bug is fixed: admitted=1, queued=0, no 180s hang
  - RWKV serve decode still wrong: one empty token / no final stats
  - parity gate exists and reproduces: `solo=[37138,47,11]` vs `multi=[37138,45,21265]`
  - root: first token is correct; recurrent GPU slot state diverges at token 2
- Quality state:
  - Existing argmax-identity gates for Qwen levers are useful: fast ~83-90%, F16_KV high
  - Raw `hawking generate` is not a valid instruct/Q&A quality gate because it does not apply chat templates
  - Valid instruct quality should use `/v1/chat/completions` after RWKV serve correctness, or manual templates
- Training/KD state:
  - KD beat SFT in small-draft experiments, best 75M KD agreement ~19.4% vs SFT 17.7%
  - This is undertrained, not no-go. Converged KD needs more corpus and a separate training campaign

Rails:
- No destructive git. Do not revert, delete, or reset user/agent work.
- Do not run global `cargo fmt`; preflight fmt failure is known repo-wide drift.
- Keep GPU jobs sequential. Check `ps` before any model job.
- Reports under `reports/` are evidence and should not be staged.
- Do not commit unless scope is narrow, reviewed, and excludes unrelated dirty lanes.
- Sonnet is fine for docs, harnesses, inventory, and validation. Use higher-care review for RWKV/Metal state, Qwen hot paths,
  quantization wiring, or output semantics.

Primary deliverable: Project-standing packet
Create or update:
1. `docs/campaign/project_standing_snapshot.md`
   - one-page current state
   - exact speed table
   - quality/validation status
   - what is production-ready, experimental, blocked, and dead
   - exact next commands
2. `docs/campaign/open_risks_and_gates.md`
   - every remaining risk
   - evidence required to close it
   - current best next gate
3. `docs/campaign/pruning_inventory.md`
   - candidates for deletion, condensation, or archival
   - classify as safe-doc-cleanup / needs-attended-review / do-not-delete
   - include evidence and owners; do not delete code in this pass unless trivial and obviously generated
4. `docs/campaign/commit_plan.md`
   - split dirty tree into clean commit lanes:
     - RWKV serve admission and HTTP robustness
     - SSM productization harness/docs
     - diagnostic tests
     - training/KD experiment lane
     - campaign documentation

Stretch lane A - harden what already exists:
1. Inspect `reports/overnight/20260622T025008Z/summary.md` and command logs.
2. Record the final RWKV/mamba/Qwen numbers in the snapshot.
3. Run only low-risk checks:
   - `bash -n tools/ci/overnight_hardening.sh`
   - `bash -n tools/ci/ssm_serve_smoke.sh`
   - `bash -n tools/ci/ssm_product_gate.sh` if present
   - `bash -n tools/ci/ssm_quality_suite.sh` if present
   - `git diff --check`
4. If any check fails due to your edits, fix it. If it fails due unrelated drift, record it.

Stretch lane B - RWKV serve correctness, only if gated:
1. Run or inspect:
   `cargo test --release -p hawking-core --test rwkv7_prefill_slot_multiseq_parity -- --ignored --test-threads=1`
2. If it fails as expected, document the exact signature.
3. Do not attempt a Metal/state-copy fix unless you can keep it tiny and rerun the parity gate.
4. If fixed, rerun:
   `tools/ci/ssm_serve_smoke.sh models/rwkv7-g1-04-sft-Q4_K_M.gguf`
5. Success means useful text, final stats, `[DONE]`, admitted>=1, queued=0, tokens_generated>1.

Stretch lane C - pruning and condensation:
1. Collapse contradictory docs into one source of truth where safe.
2. Mark stale docs as superseded instead of deleting.
3. Identify dead/no-go code paths, but do not remove hot-path code without attended approval.
4. Create a concise "ask me for project standing" summary seed:
   `docs/campaign/summary_seed_for_new_chat.md`
   This should be the file the user can paste into a new chat to ask for a full project standing.

Stretch lane D - quality gate repair:
1. Add a note to any quality script/report that raw `generate` is not a chat-template instruct gate.
2. If low-risk, add manual chat-template prompts for Qwen and RWKV to make a temporary quality gate.
3. Otherwise, leave quality as blocked on RWKV serve `/v1/chat/completions` correctness.

Stretch lane E - final recommendation:
At the end, write the clearest possible recommendation:
- what to ship now
- what to keep researching
- what to stop doing
- what single next technical fix unlocks the most value

Useful commands:

```bash
git status --short
git diff --stat
git diff --check

ps ax -o pid= -o ppid= -o etime= -o %cpu= -o %mem= -o command= \
  | rg 'hawking (generate|serve)|overnight_hardening|cargo (build|check|test)|mamba2|rwkv7' \
  | rg -v 'rg '

sed -n '1,260p' reports/overnight/20260622T025008Z/summary.md
tail -160 reports/overnight/20260622T025008Z/commands.log
sed -n '1,220p' docs/campaign/test_matrix.md
sed -n '1,260p' docs/campaign/autonomous_run_log.md
```

Stop condition:
Stop when the project-standing packet exists, the pruning inventory exists, checks are recorded, and the next single best
technical fix is unmistakable. Do not mark the whole Hawking project done; mark the campaign summarized and ready for a
new-chat standing review.
````
