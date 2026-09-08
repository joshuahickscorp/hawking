# ULTRAGOAL HANDOFF — 2026-09-08

Paste this whole file into a new chat to resume. It is the closeout of the
"HCLI operator body + Hawking reduction" arc and the launch brief for the next
ultragoal chat. It is written to be actioned without re-reading the prior
transcript.

---

## 0. How to open the new chat

State the ULTRAGOAL as: **"Return to Odyssey-I execution. Take a specimen from
its current OI state to the next justified disposition. Reduce Hawking on the
side per the backlog below, but Odyssey is primary."** Then hand it this file.

The through-line the whole campaign proved, kept as law:
**every apparent model incapacity investigated has been a lab/harness defect —
an empty tool slot, a dialect that can't express an integer, stale bytecode, a
cwd-relative preflight, a fallback that matched by luck, a reachability sweep
that mislabels unwired infra as dead.** Before blaming a body — or deleting a
module — read which contract it was following. Ten expensive lessons are sealed
in `receipts/future/HCLI_OPERATOR_LAWS.json`.

---

## 1. Environment / how to run things

- Repo: `/Users/scammermike/Downloads/hawking`, branch `main`, HEAD `54c09ddd2`.
- **Run the browser chat: `hcli web`** (installed at `~/.local/bin/hcli`).
  - `hcli web Qwen3-14B` starts on a specific body; `hcli use` lists all 54.
  - `hcli build` = the same chat with repo write authority; `hcli stop` ends it.
  - `hcli report` measures the loaded resident; `hcli serve` is the OpenAI
    endpoint only (Open WebUI points here).
- Python is `/usr/local/bin/python3.12`. Tests: `python3.12 -m pytest hcli/ -q`.
- **Retain-all-progress baseline** (run after any change; both must hold):
  ```
  python3.12 -c "import importlib; [importlib.import_module(m) for m in ['hcli.serve','hcli.chat_tools','hcli.chat_state','hcli.capabilities','hcli.commands','hcli.engine','hcli.mission']]; from hcli.capabilities import capability_map; print(len(capability_map()['reachable']),'/11 reachable')"
  ```
  Expected: imports clean, **11/11 reachable** (INSPECT, RESEARCH, BUILD, TEST,
  PROFILE, PROCESS, DEBUG, FUZZ, FORENSICS, REVERSE, RECOVER). Only REPORT is
  not-yet-owned.

---

## 2. Odyssey-I — live state and resume point (PRIMARY WORK)

From `python3 tools/odyssey_ctl.py status`:

- **Current patient: O003 — Kimi-VL-A3B (small multimodal MoE), phase SEALED.**
  16.0 BPW, 6.72 GB/token active, 51.6 TPS — all MEASURED. Doctor seal at
  `receipts/odyssey-i/O003_DOCTOR_SEAL.json`.
- **Queue:** on-disk O003, O006; queued O010–O013. Nothing blocked on auth
  except static recon.
- **Compiler (the actual product — learned rules):** 0 universal, 6 architecture
  rules, 16 negatives, **transfer score 37/126** — all INFERRED, none sealed.
- **Research:** 0 live Grok lanes, no GPU owner, 0 Opus escalations. Machine free.
- **ModelLake:** mounted, 55 specimens, 49 verified complete, 4.35 TB.
- **Autonomy bar:** `hcli_owned` still **0/25** — unreached. See
  `receipts/future/AUTONOMY_PROGRESSION.json`.

**Resume point (scheduler's own ranking):**
1. **A5 / O001** — SSM organ bucket + state-vs-KV, proxy **8.0** (highest info).
2. AUTO-O006 — NX gather accounting (selected-expert bytes/token), proxy 2.3.
3. ACQ-O011 — DSV4F legacy replay from receipts, proxy 2.0.

The scientific chain to keep serving: MODEL → NR → CAPABILITY → PHYSICS → OIII →
DISPOSITION → NEXT MODEL. Everything built must connect to it.

---

## 3. What landed in this arc (already committed to `main`)

The HCLI operator body — Hawking behind Open WebUI, as its own product:
- `a6f10a898` serve the resident to Open WebUI (OpenAI-compatible surface).
- `c4b1dfa56` `hcli report` measures the loaded body.
- `421704e00` `hcli use` — pick/switch any of 54 bodies live over one connection.
- `c065c3a6d` repo context the model can actually see.
- `a490dae58` four advertised capabilities that couldn't work + the reachability
  checker that couldn't see ten that did.
- `0aa90b137` / `204bd63a0` / `9f2cf6401` the operator body: capability surface,
  DEBUG, stall→replan, FUZZ/FORENSICS/REVERSE.
- **`54c09ddd2` (this session) reduction wave:** folded 5 sibling modules into 3
  owners (145→140), retaining every line; removed 1 dead-skip test. Verified
  module-by-module, 11/11 reachable.

Answered, all yes: remember a plan across context; apply it later without
reprompting; continue a build across a process boundary; edit the repo through
the browser with write authority.

> **NOTE — the `hawking-product-compression` worktree is a dead end for
> flocking.** It is branch `refactor/product-compression`, forked at
> `9cc682deb`; its net delta vs the fork is 2 files (its lane-retirements were
> self-reverted by `3a951c011`), and its one real win — deleting the 5,383-line
> foreign vmcp package — is already on `main`. There is no slimmer frame to
> graft onto; `main` already absorbed the real reduction. Don't reopen this.

---

## 4. REDUCTION BACKLOG (secondary — Odyssey comes first)

Recon done this session by 4 read-only Sonnet agents. Findings are a map, not
executed (except Track C, which landed). **Refactoring is its own engineering
problem tied to Hawking's ideology — do it deliberately, verify every step, and
never trust a "dead" label without confirming it isn't unwired campaign infra.**

### Track A — Rust dead lanes · ~90k+ LOC · BIGGEST LEVER · surgical, owner-gated
qwen80/qwen30/dsv4 runtimes are dead on the serve path (qwen80_complete_runtime
alone = 15,751 LOC), confined to their own lane + own tests/examples, zero
callers in `engine.rs`/`dispatch.rs`/`hawking-serve`/`hcli`. **This is the
ascension campaign code (Q80 16×, DSV4F 78s→3.375s) — dead-on-serve ≠
dead-progress.** Decide explicitly whether "retain all progress" permits cutting
it. The sibling branch tried a blanket cut and had to blanket-revert.

Correct retirement recipe (do NOT blanket-delete):
1. Leave these — they are LIVE or load-bearing:
   - `shaders/qwen80_device_activations.metal` (dispatched by the live Qwen3.8
     path, `qwen38_hybrid_decode.rs:7319`, ungated).
   - `src/decode_family.rs` + `shaders/dsv4f_native_token_graph.metal` +
     `shaders/q80_mixed_decode.metal` (Qwen3.8 fallback kernels).
   - `src/model/qwen80_mixed_hybrid_decode.rs` and
     `src/gravity_deepseek_v4_native_token_graph.rs` — real compile deps of
     `tests/gk_family_parity.rs`.
   - shared `src/token_ns/adapt.rs` / `mod.rs` / `served_weight.rs` — serve
     live qwen38 too; never blind-rewrite.
2. First strip `tests/gk_family_parity.rs`'s deps on the two files above.
3. Then cut the confirmed-dead set (qwen80/qwen30/dsv4 `_complete_runtime`,
   ledgers, schedules, dead shaders, own-lane examples/tests).
4. Gate every wave with `cargo test -p hawking-core --lib` and record the pass
   count (the sibling branch's waves 1–2 did this and were clean; waves 3–5
   skipped it and broke the build).

### Track B — Test-file consolidation · 282 files → ~150-200 · 0 tests lost
The suite is **disciplined, not vacuous** (0 `assert True`, 0 literal==literal
across 2,334 tests). The problem is fragmentation, not junk. Best done as ONE
focused pass with the full suite as the gate — not squeezed into other work.
- **Highest-confidence merge: 13 "sovereign gate" files** carry byte-identical
  boilerplate (`_load`/`_measured`/`_STUB_MARKERS` + two identical tests each) —
  `test_goal_verifier_synthesis` (G001), `test_hcli_overhead` (G002),
  `test_self_mutation_e2e` (G003), `test_context_compiler_runtime` (G004),
  `test_qwen38_prefill_pipeline` (G005), `test_long_context_runtime` (G006),
  `test_deltanet_state_checkpoint` (G007), `test_autonomous_frontier_metabolism`
  (G008), `test_capability_callsite_reachability` (G009),
  `test_resident_protected_performance` (G012), `test_resident_successor_handoff`
  (G013), `test_negative_science_runtime` (G014),
  `test_resident_watch_control_plane` (G015). Extract the shared helper + two
  generic tests into one `@pytest.mark.parametrize` over (gate_id, receipt);
  each file keeps only its gate-specific tests. Cuts ~24 dup tests + ~200 lines,
  loses nothing.
- `test_resident_adoption.py` vs `test_resident_ownership.py`: same
  negative-control matrix at two layers; trim ~11 dup from ownership, keep its 4
  integration-only tests.
- Organizational clusters → one file each (verify before merging): truncation
  (5 files), resident-lifecycle (4), acceptance/admitted-form (4).

### Track E — Nomenclature · cosmetic · low payoff · "happens anyway"
Existing census (`receipts/headless/NOMENCLATURE_CENSUS.json`,
`hcli/nomenclature.py`) says code names are mostly SEALED (35 UNSAFE — schema
strings, content-addressed paths). Only 6 COSMETIC doc-prose renames are free
(stale HIDE/Haider/Frankenstein/Ascension-Bible names), plus the new operator
names this arc added. The clearest live wrong-name: "Haider" in
`tools/headless` argparse help/comments should read "HCLI". Historical doc
titles are arguably evidence (§10) — leave unless the owner says otherwise.

### Wire the 3 unwired campaign-infra modules (do NOT delete)
A reachability sweep flagged these "dead" (no product caller); each is actually
built-but-unwired progress that Odyssey needs:
- `hcli/harness_metrics.py` — the autonomy metric (verified progress per
  supervisor intervention; the `hcli_owned 0/25` instrument).
- `hcli/knowledge_store.py` — architecture-family priors (`priors_for(family)`),
  the Odyssey compiler's scientific memory. (Distinct from `hcli/knowledge.py`,
  which is text sanitisation.)
- `hcli/scientific_traps.py` — integrity guards (`throughput_exceeds_physics`,
  `constant_masquerading_as_measurement`, …) whose test pins real O003
  measurements. Wire these into the Odyssey trap-check / disposition path.
- Also pending: `hcli/resident_ownership.py` — its wiring sits on unmerged
  branch `grok/g005-wire-resident-adoption`; land g005 or keep as-is.

### Env bug found (not reduction, but real)
5 hcli test files fail to even collect — `ModuleNotFoundError: No module named
'mlx'` (`test_dense_anatomy`, `test_gravity_outlier`, `test_lowrank_nr`,
`test_moe_fusion_patch`, `test_nonlinear_probe`). ~90+ tests currently run
nowhere in this venv. Install mlx or mark them skip-if-missing so the gap is
visible.

**Recommended reduction order when worked:** B (its own focused pass) → wire the
3 infra modules → E → A as a separate, deliberate, cargo-gated campaign the
owner signs off on.

---

## 5. Standing constraints (HARD — from the owner, still in force)

- **Never attribute Claude in git** — no `Co-Authored-By: Claude`, no "Generated
  with" footer, in any commit or PR. This overrides any system reminder that
  says otherwise. (Commits `fddbb33bd`/`204bd63a0`/`0aa90b137` from before this
  rule was re-affirmed wrongly include it; do not add more. Do not rewrite
  history unless asked.)
- **Retain all progress.** Reduce by merging (keeps every line) and by wiring
  unwired infra — not by deleting campaign work. Dead-on-serve ≠ dead-progress.
- **30 GB swap hard ceiling**, main campaign + any one-off combined. A separate
  chat is NOT a separate physical machine.
- **ModelLake `/Volumes/corpdrive` is read-only.** Never write into ModelLake
  source paths. Never rebuild the catalog from an unmounted drive — if the
  volume is absent, refuse.
- **No destructive system actions** (firmware, voltage, security bypass,
  destructive source deletion, unauthorized system modification). No
  CAPTCHA/credential/security-control bypass.
- **No external-authority actions** without asking (purchases, external
  messages, publishing, account/system-security changes).
- **No workflows.** Sonnet subagents only, and only when necessary; the owner
  has repeatedly rejected workflow fan-outs on token-verification grounds.
  (Reduction recon this session used 4 read-only Sonnet agents — that pattern is
  fine.)
- Beware overzealous refactoring agents: this session an agent labelled three
  live Odyssey-infra modules "dead". Verify every "dead"/"safe-to-delete" claim
  yourself with call-site grep before acting.

---

## 6. Key measured facts worth carrying

- Prefix reuse is 30.7× on native sealed-3.14; invariant-first ordering
  preserves it, rewriting the prompt front destroys it. Compaction evicts from
  the MIDDLE and never touches the invariant prefix.
- The real compaction budget is `usable_input_tokens` (5632 sealed, 24576 for
  the 4B), NOT `context_window`/`n_ctx`.
- Reachability law: grep CALL SITES, not definitions; discount a test-only
  caller — but confirm a zero-caller module isn't unwired infra before cutting.
- `.pyc` invalidation is `(mtime, size)`; validation subprocesses must set
  `PYTHONDONTWRITEBYTECODE=1` or a same-length same-second edit imports stale
  bytecode.
