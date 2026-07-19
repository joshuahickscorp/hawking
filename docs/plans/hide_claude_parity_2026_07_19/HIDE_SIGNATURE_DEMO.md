# HIDE Signature Demo

Run date: 2026-07-19 · Grounding: `HIDE_STATE_CAPSULE_ABI.md`, `HIDE_SUPREMACY_THESIS.md` §2.4, `HIDE_SIGNATURE_DEMO` gates in `HIDE_EXPERIMENT_MENU.md`.
Bible §97: the polished demonstration of the one thing Claude Code cannot structurally provide.

## 1. The demo, in one sentence

> From one warmed project state, fork several isolated agents that pursue different solutions, watch them work in parallel, let execution verify all of them, compare, accept the winning diff, with no cloud bill and a provable no-egress run.

## 2. The 90-second script

```text
0:00  One warm session on a real repo. A hard task: "make the parser 2x faster
      without changing output; there are at least two viable approaches."
0:10  User hits "Fork and try N". HIDE forks the warm state capsule into 3 agents.
      The Context Stack shows: prefill avoided (shared prefix), fork latency (ms),
      resident memory per fork. No re-prefill. No dollar meter, anywhere.
0:20  Three fleet cards appear, each an isolated worktree + forked state:
        A: memoize + precompute lookup tables
        B: streaming single-pass rewrite
        C: SIMD-friendly restructuring
      Each streams its own plan and edits. The main session stays interactive
      (a warm side-fork answers a /btw question with zero added latency).
0:45  Execution verifies each: build + the existing test suite + a micro-benchmark
      oracle. A fails a correctness test (boolean edge case) and self-corrects.
      C builds but only reaches 1.4x. B reaches 2.3x, tests green.
1:05  Compare view: three diffs side by side with their verification receipts
      (tests, benchmark delta, diff size, review notes). B is the execution winner.
1:20  Accept B's diff (per-hunk). The losing forks are dropped (near-zero cost).
      A no-egress badge shows nothing reached the network the entire run.
1:30  Done: what changed, why, the benchmark proof, rollback available.
```

## 3. Why this is structurally impossible for Claude Code

- **Cost.** Claude Code meters every token; running 3 full approaches to completion plus verification is 3x+ tokens against a weekly cap. The ~7x-token agent-teams warning [DOCUMENTED] exists precisely to discourage this. HIDE has no meter.
- **Fork.** Claude Code's "fork" is a prompt-cache hit; each agent still re-establishes context from text. Cursor's `/best-of-n` provisions N cloud VMs + re-tokenizes. HIDE's fork is a pointer-copy of a resident state capsule (`RwkvState::fork`, byte-exact memcpy, no re-prefill).
- **Egress.** The whole run is local; the no-egress badge is a hardware fact, not a proxy allowlist.

## 4. Honest readiness (what must ship before this demo is real)

This demo is the payoff of the supremacy thesis, and it is **not shippable today**. The archaeology is explicit about the gates:

| Gate | Status | Blocks |
|---|---|---|
| `RwkvState::fork` (memcpy, byte-exact) | real+unwired | the fork itself works; needs a caller |
| GPU->CPU recurrent readback (exact live capture) | missing | forking mid-run on the default GPU path is not byte-exact yet |
| `/v1/hawking/state/{save,load,fork}` + session->slot affinity | missing | no way for a client to drive the fork |
| Fleet orchestration + isolated worktrees + merge | real+unwired (packed `hide-fleet`) | the N-agent board and compare view |
| Execution tie-break (build/test/benchmark oracles) | real+unwired (packed `hide-kernel`) | "execution decides the winner" |
| No-egress proof mode | partial | the badge |

So the demo ships on the **RWKV lane after Phase 2** of the ladder (state exposure), with the transformer/Hybrid lane (a Qwen3-Coder-Next-class coder) following once the transformer capsule exists (`blocked_on_model`). **Presenting this demo before those gates would be a demo of unwired primitives** and is explicitly forbidden by the honesty discipline.

## 5. What the demo must measure (not just show)

Per `HIDE_CAPABILITY_DENSITY_EVAL.md`, the demo doubles as a measured experiment:

- fork latency and prefill avoided (the "no re-prefill" claim);
- resident memory per fork (the resource envelope);
- verified gain: does best-of-3 beat one stronger single attempt at equal wall-clock, and at what energy? (the honest question from the falsification list: "cheap state forks make best-of-N worthwhile" must be *proven*, not assumed);
- merge success rate on the accepted winner.

If best-of-N does not beat one stronger attempt per second/joule, the demo is downgraded from "signature advantage" to "situational tool," and that finding is reported, not hidden. The demo's credibility comes from the receipts, not the animation.

## 6. The fallback demo (if the state moat measurements disappoint)

If GPU->CPU readback proves too costly or best-of-N does not pay off, the honest fallback signature is **the shared-state daemon** (`HIDE_SUPREMACY_THESIS.md` §2.5): one warm resident session, attached simultaneously from Chat, the IDE, and a headless SDK client, with a surface switch costing a pointer copy and zero re-prefill, plus a provable no-egress run. That advantage is structural and does not depend on the fork-best-of-N economics holding.
