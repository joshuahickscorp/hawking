# HIDE Minimum Lovable Vertical

Run date: 2026-07-19 · Grounding: `HIDE_PRIORITIZED_BUILD_LADDER.md` (Phase 0/1), `HIDE_CLAUDE_CODE_UX_GENOME.md`, `HIDE_LIVE_ARCHAEOLOGY.md`.
Bible §96: the one daily-usable vertical proving a Claude-Code-quality chat loop on a real local Hawking model with real tools, context, verification, diff/review, IDE sync, and session resume.

## 1. The slice, in one sentence

> A developer opens a real repo in HIDE, trusts it, describes a multi-file bug, watches HIDE read the repo, plan, edit transactionally, run the tests, self-correct on a failure, and present a reviewable diff with proof, then resumes the same session tomorrow, all on a local Hawking model with no meter and no egress, and can steer the run from either Chat or the editor.

This is deliberately *not* the signature demo (warm-state best-of-N). It is the smallest thing that is **lovable daily**, which means it must nail the top love genes (steerable autonomy, reversibility, legibility, proof-of-done), not the most novel one.

## 2. Why this slice and not a flashier one

The love/pain map shows the love concentrates in the *ordinary* loop done well, not in a novel trick. The archaeology shows this loop is a reconnection of already-tested parts, so it is the highest-P, lowest-risk path to a product people use every day. The signature advantage (state forks) rides on top of it later. Shipping the trick before the loop would violate the Apple-vs-Samsung doctrine (a novel feature with a rough ordinary loop loses).

## 3. Components (each mapped to its source and readiness)

| Component | Source | Readiness | What "lovable" requires |
|---|---|---|---|
| Trust-before-config open | new (security floor) | missing | a calm dialog listing what the repo would grant; nothing runs before accept |
| Flat kernel loop (plan-as-data, RuntimePlanner) | `hide-kernel` | real+unwired | replaces the 256-token single-shot turn |
| Reserve-then-fill ContextPack fed to the model | `hawking-context` | real+unwired | the compiled window actually reaches generation (fixes S3) |
| Living code index grounding | `hawking-index` | real+unwired | the agent "holds the repo in its head" |
| Typed tools: read/search, transactional edit, build/test, git | `hide-tools` | real+unwired | edits are reviewable and reversible |
| Deterministic verify oracles (build/typecheck/test/lint) | `hide-kernel` | real+unwired | "done" = green tests, not a claim |
| Interrupt-and-keep + soft steer | FE SteerBar + live turn | ui_only -> wire | the #1 love gene; must feel instant |
| Plan mode + graded approval | FE PlanCard + tool gate | ui_only -> wire | the #2 love gene; write block enforced |
| Diff review (per-hunk accept/reject) | FE HunkReview | real (mock-fed) -> wire | native side-by-side, Tab/Esc |
| Durable transcript + resume | `hide-backend` replay | real+unwired | crash/restart costs nothing |
| CLAUDE.md migration read | new (migration reader) | missing | an existing Claude Code repo works unchanged |
| Local model on Hawking serve | `hawking-serve` + a coder | partial | a capable-enough local coder (see caveat) |
| Chat<->IDE one-session sync | FE two chambers + bridge | partial -> wire | click a file/diff in Chat, review in IDE, same session |

## 4. The end-to-end path (what runs)

```text
open repo -> trust gate -> [session core created]
  user: "the CSV importer drops the last row and mislabels booleans; fix it"
    -> index/retrieval (hawking-index) -> ranked exact spans
    -> reserve-then-fill ContextPack (hawking-context) -> fed to the model
    -> plan mode: agent proposes a plan; user approves (graded)
    -> flat loop: edit (transactional) -> run tests (ProcessOracle) -> FAIL on booleans
    -> self-correct: re-read failure -> edit -> re-run -> GREEN
    -> diff review: per-hunk accept/reject in Chat or IDE
    -> "done" summary: what changed, why, test evidence, remaining risk, rollback
    -> commit (permission prompt, 'don't ask again' persists per-repo)
  [next day] resume -> full conversation + warm state restored, no re-prefill
```

## 5. Acceptance criteria (the exit gate, restated as a test)

1. A stranger opens the app, trusts a repo, and completes the CSV task through the real UI (not a mock).
2. Every edit is a reviewable transaction; rewind restores code + conversation.
3. Interrupt (Esc) cancels the in-flight tool and keeps prior work; a typed correction steers without stopping.
4. Plan mode structurally blocks edits until approval.
5. Tests decide "done"; the summary shows the evidence, not a celebration.
6. Resume the next day restores the session; the warm state is reused with no re-prefill (RWKV lane) or an honest "re-prefilled" note (transformer lane).
7. The same session is steerable from both Chat and the editor.
8. An existing Claude Code repo's `CLAUDE.md` loads unchanged.
9. A complete critical-path trace exists; no unsandboxed exec path; no egress unless explicitly granted.

## 6. The honest caveat (what could make it not-lovable)

- **Local model judgment** is the risk. The harness is provably reconstructable; the *quality* of plans/edits depends on the local coder. The vertical should ship first on the strongest currently-serveable local coder (dense/MoE ACTIVE in the archaeology), with the Qwen3-Coder-Next feasibility branch isolated from the ship path (`HIDE_LOCAL_MODEL_TOPOLOGY.md`). If the local coder is too weak for the CSV-class task, the vertical is not lovable yet, and that is a `blocked_on_model` fact to state, not paper over.
- **Interrupt latency and diff polish** are the two ergonomics that, if rough, sink the slice regardless of correctness. They get disproportionate polish budget.

## 7. Why this proves the thesis

If this slice ships and is used daily, it proves: (a) the packed harness reconnects into a Claude-Code-quality loop; (b) it runs on a real local model with no meter and no egress; (c) the two surfaces share one session. That is parity demonstrated on the real app path, which is the precondition for every supremacy claim (`HIDE_SUPREMACY_THESIS.md` §4: no "fastest/densest" claim before a receipt on the real path).
