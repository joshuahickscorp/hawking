# HIDE Blind Preference Study

Run date: 2026-07-19
Grounding: `HIDE_LIVE_ARCHAEOLOGY.md`, `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json`, `HIDE_USER_LOVE_PAIN_MAP.md`, `HIDE_STATE_CAPSULE_ABI.md`, `HIDE_TWO_SURFACE_ARCHITECTURE.md`; dossier `docs/plans/hawking_ide_frontier_2026_07_19.md` sec 5.12 / 8 / 9.
Bible: sec 89 (this study), sec 15 (model-vs-harness decomposition), sec 33 (response-style eval).
Status: specification for a human-subject preference study. It is the SUBJECTIVE complement to `HIDE_CAPABILITY_DENSITY_EVAL.md` (objective, oracle-graded coding capability). Read together: this measures what users prefer, the sibling measures what is correct. Neither substitutes for the other, and preference is never reported as capability.

## 1. Purpose and the single confound it exists to kill

A naive "HIDE vs Claude Code, which do you prefer" test is uninterpretable, because it conflates two independent factors: the **model** (Claude cloud model vs the local Hawking model) and the **harness** (the agent loop, context, tools, permissions, review, and response surface around the model). Cursor reports that model and harness jointly determine quality, that they must be co-tuned, and that swapping a model mid-session produces out-of-distribution history plus a cache miss (dossier sec 5.12, model-specific harness profiles). If HIDE loses a blind preference test, a naive design cannot tell whether the harness is worse or the local model is simply weaker, and those two conclusions have opposite build implications.

The local model is, today, the plausible loser: HIDE's capability-density model bet (Qwen3-Coder-Next on Hawking) is an open research bet with an explicit kill criterion (dossier sec 9; open decision 2, sec 11), not a shipped equal to a frontier Claude model. This study is therefore built as a **factorial decomposition** (Bible sec 15) that separates the model contribution from the harness contribution, so we can state honestly: how much of any preference gap is the model (a gated model build item), and how much is the harness (what HIDE can fix without model changes).

The study answers five per-comparison questions, elicited as forced paired choices (Section 10): which feels **more capable**, **clearer**, **faster**, **more trustworthy**, and **which would you use daily**.

## 2. Honest precondition: what is runnable today vs gated

The production HIDE vertical slice is broken (`HIDE_LIVE_ARCHAEOLOGY.md` sec 0: the frontend speaks `/v1/hide/*` on 8744 to a `hide-serve` that no longer builds; live `hawking-serve` serves a disjoint OpenAI-compatible surface on 8080). A full "HIDE" arm cannot be run until the Phase 0/1 reconnection lands (`HIDE_LIVE_ARCHAEOLOGY.md` sec 6.1). The study is staged to match the build ladder so no arm is faked.

| Arm cell | Runnable today | Blocked on |
|---|---|---|
| Claude model in Claude Code | yes | nothing |
| Claude model in minimal harness | yes | nothing |
| Other cloud model in Claude-Code-like harness | yes | model-endpoint override (see Section 5) |
| Hawking model in minimal harness | **yes** | nothing (`hawking-serve` ACTIVE, OpenAI-compatible, `HIDE_LIVE_ARCHAEOLOGY.md` sec 3.2) |
| Hawking model in HIDE | no | Phase 1 reconnect (`hide-core`+`hide-serve`+context+index+kernel+tools wired) |
| HIDE warm-resume / fork / two-surface supremacy variants | no | state-capsule exposure + `HTTP state routes` + session-slot affinity; each variant names its gate in Section 7 |

**Experiment 0 (runnable now, highest decision value):** run only the two minimal-harness cells (Claude model vs Hawking model, harness held identical). This isolates the raw local-model gap with zero harness confound and is the number that should gate whether HIDE harness reconnection is worth funding before the model bet resolves (dossier sec 9 kill criterion for the Qwen3-Coder-Next bet). Everything else in this document is the full study that becomes runnable as the ladder advances.

## 3. The factorial: model x harness decomposition (Bible sec 15)

Two factors. Model in {Claude cloud, Hawking local}; harness in {Claude Code, minimal control, HIDE}. The five load-bearing cells the decomposition needs:

| Cell | Model | Harness | Role |
|---|---|---|---|
| C1 | Claude | Claude Code | the reference users love (`HIDE_USER_LOVE_PAIN_MAP.md` sec 1) |
| C2 | Claude | minimal | isolates the value Claude Code's harness adds over the model |
| C3 | Other cloud | Claude-Code-like | tests harness portability across a model swap |
| C4 | Hawking | HIDE | the target product |
| C5 | Hawking | minimal | isolates the value HIDE's harness must add over the Hawking model |

The **minimal harness** is a mini-SWE-agent-class control (single flat loop, bare tool set, no context compiler, no fleet), retained deliberately as the baseline that every layer of orchestration must beat to justify its latency (dossier sec 5.12, small deterministic maps / mini-SWE-agent as control; sec 8). It is the same binary across C2 and C5, so the model is the only difference between them.

Linear contrasts recovered from the cells (each estimated with a bootstrap CI, Section 12):

| Contrast | Cells | Estimates |
|---|---|---|
| Pure model gap | C2 vs C5 | Claude-vs-Hawking with harness held constant (the cleanest capability contrast) |
| Claude Code harness value | C1 vs C2 | what the loved harness adds on top of its own model |
| HIDE harness value | C4 vs C5 | what HIDE adds on top of the Hawking model = **what HIDE can fix without model changes** |
| Harness portability | C1 vs C3 | whether a Claude-shaped harness survives a model swap (Cursor OOD/cache-miss risk, sec 5.12) |
| Product gap | C1 vs C4 | the felt end-to-end gap; decomposed, not reported bare |

**The decomposition inference (the whole point):** the raw C1-over-C4 product preference is `model gap + harness gap`. If HIDE harness value (C4-C5) is greater than or equal to Claude Code harness value (C1-C2), then HIDE's harness is at or above parity and the residual product gap is attributable to the model, which is a separate gated build item, not a harness failure. This is the only construction that prevents a model gap from masquerading as a harness verdict, and prevents a harness win from masquerading as model quality.

## 4. Participant selection

Target: practicing developers who use agentic coding tools weekly. Recruit n = 30 for the full paired study (power sketch below); Experiment 0 needs fewer (n ~ 16) because the within-subject model contrast has lower variance. Screen out non-coders and pure spectators.

Stratify (recorded as covariates, balanced across arms):

| Stratum | Levels | Why it is a confound |
|---|---|---|
| Claude Code familiarity | power user / occasional / never | brand-habit is the dominant lock-in and biases blinded preference (`HIDE_USER_LOVE_PAIN_MAP.md` sec 4: captivity is habit + config, not technical). Power users recognize the Claude Code response style even blinded. |
| Primary language | Rust / TypeScript / other | tasks are Rust+TS to match the private eval suite (dossier sec 8.2); language mismatch inflates task difficulty noise |
| Privacy posture | regulated/air-gapped / neutral / cloud-native | regulated teams already point Claude Code at local models as a workaround (`HIDE_USER_LOVE_PAIN_MAP.md` sec 5) and will over-weight the no-egress wedge |
| Seniority | senior / mid / junior | affects tolerance for autonomy and review depth |

Familiarity is the sharpest confound and must be a fixed effect in the model (Section 12), not just balanced. Compensate participants at a flat rate independent of which tool they prefer, to avoid demand characteristics.

## 5. Arm instantiation

- **C1 Claude model / Claude Code:** stock Claude Code, but with the status line configured to **suppress the dollar/cost meter** (the JSON status-line contract is scriptable, parity id `loop.status_line`), so the presence-of-a-meter tell does not leak the tool identity in blinded tracks (Section 9).
- **C2 Claude model / minimal:** the mini-SWE-agent-class control driving the Claude API.
- **C3 Other cloud model / Claude-Code-like:** feasible because Claude Code (and Claude-Code-shaped harnesses) can be pointed at a non-Claude model endpoint; regulated-team practice already does this (`HIDE_USER_LOVE_PAIN_MAP.md` sec 5). If the stock product refuses a given endpoint, reconstruct the harness shape (same prompt/edit/tool ABI) over the other model. Record which was used per session (dossier sec 5.12 warns the harness may be out of distribution for the swapped model; that is the effect C3 measures, not a bug to hide).
- **C4 Hawking model / HIDE:** the reconnected vertical slice (Phase 1). Blocked today (Section 2).
- **C5 Hawking model / minimal:** the same minimal harness as C2, driving `hawking-serve` on 8080. Runnable today.

The Hawking model in C4/C5 is fixed by `HIDE_LOCAL_MODEL_TOPOLOGY.md` (the coding lane, Qwen3-Coder-Next-class per the model bet, dossier sec 9). C4 and C5 must use byte-identical weights, quantization, and tokenizer so the harness is the only difference; record the `.tq`/model id in the receipt (Section 14).

## 6. Task battery

Nine tasks, each a real-work shape mapped to a parity behavior and graded by a deterministic oracle first where one exists (dossier sec 5.12 actor/evaluator: deterministic test/build/policy oracles first, a model evaluator only for ambiguous acceptance; decision 8). Each task is time-boxed; a task can succeed, partially succeed, or fail, and that objective outcome is recorded separately from preference so a liked-but-wrong run cannot pass as capable.

| Task | Exercises (parity id / doc) | Deterministic oracle | HIDE readiness |
|---|---|---|---|
| Cold-repo bug fix | index + context compiler + kernel loop | failing test flips to green | needs Phase 1 (index/context/kernel packed, `HIDE_LIVE_ARCHAEOLOGY.md` sec 3.5) |
| Unfamiliar-repo question (read-only) | repo map / retrieval; Aider RepoMap baseline (sec 5.12) | rubric vs gold answer key (no edit) | needs Phase 1 |
| Plan-then-implement | `perm.plan_mode` (ui_only) | plan artifact exists + post-plan diff passes tests | needs kernel + tool gate |
| Test-failure repair | core loop + verification plane; edit-to-green | test suite green | needs Phase 1 |
| Multi-file refactor | verifying edit applier (`hide-tools`, packed); `apply_patch` correctness flagged NEEDS PROBE (`HIDE_LIVE_ARCHAEOLOGY.md` sec 4) | tests green + no unintended file touched | needs `hide-tools` wired; probe apply_patch first |
| Git conflict resolution | `hide-tools` git worktree trio (packed) | merge resolves + tests green | needs `hide-tools` wired |
| Session resume | `session.resume_picker` (partial), `session.durable_transcript` (packed_unwired), `session.checkpoint_rewind` (ui_only) | resumed session continues to a correct patch | PARITY needs Phase 1; **SUPREMACY** (instant warm-state resume, no re-prefill) gated on `HIDE_STATE_CAPSULE_ABI.md` sec 8 exposure |
| IDE handoff (Chat to editor) | `ide.two_surface_bridge` (partial); `HIDE_TWO_SURFACE_ARCHITECTURE.md` | edit applied at the handed-off location, tests green | PARITY needs bridge; **SUPREMACY** (one warm resident context, zero round-trip) gated on `/v1/hide/*` reconnect |
| Permission denial | `perm.rule_engine` (packed_unwired), `trust.workspace_gate` (absent), `perm.persistence_tiers` (packed_unwired) | a denied action is physically blocked, not merely refused in prose | needs `hide-security`+`hide-tools` wired; SUPREMACY (OS/Seatbelt enforcement) gated |

Session-resume, IDE-handoff, and permission-denial each carry a PARITY variant (reproduce the Claude Code behavior) and a gated SUPREMACY variant (the warm-state or OS-enforced version). Only the PARITY variant is measured until its gate clears; the SUPREMACY variant is added to the battery when the named build item lands, and its result is what `HIDE_SUPREMACY_THESIS.md` may cite, never before.

## 7. Two-track design (live vs frozen)

Full double-blind is impossible for a live local tool (Section 9). We therefore split measurement:

- **Track A (live, interactive):** the participant drives the arm themselves on the task. Measures the felt end-to-end experience (steerability, cancel, plan gate, latency texture) that users actually love (`HIDE_USER_LOVE_PAIN_MAP.md` sec 1). Blinding is single-blind at best.
- **Track B (frozen, matched transcript):** the participant does not drive; they compare two pre-recorded transcripts of the same task rendered in a neutral common viewer (common typeface, no tool chrome, no meter, injected-jitter timing). Blinding is much stronger because the live tells are stripped. Cost: it loses interactivity realism.

Every comparison is run in both tracks. Divergence between Track A and Track B preference is itself a finding: it localizes how much preference comes from live interaction feel vs from the rendered content.

## 8. Counterbalancing

Within-subject: each participant sees every task once and every arm multiple times, never the same task twice. Assign arm-to-task with a Graeco-Latin square so neither task order nor arm order is confounded with the arm effect, and randomize the starting cell per participant. Rotate which member of a pair is shown left/right in forced-choice screens. Insert one attention-check pair (an obviously-broken run vs a green run) to detect click-through; drop participants who fail it.

## 9. Blinding and its hard limits

A local, no-meter tool is hard to fully blind. The tells and their mitigations:

| Tell | Source | Mitigation |
|---|---|---|
| No cost/usage meter | HIDE doctrine forbids a budget meter (parity id `cost.usage_transparency`; `HIDE_USER_LOVE_PAIN_MAP.md` sec 6; `HIDE_LIVE_ARCHAEOLOGY.md` sec 3.4 Digest hides the token meter) | suppress the meter in ALL arms, including Claude Code (scriptable status line, id `loop.status_line`), so meter-absence stops discriminating |
| Latency signature (local TTFT vs cloud round-trip) | local inference has no network hop | inject randomized network-latency jitter into Track B stimuli; in Track A, disclose that latency differs and treat felt-speed as paired-relative only (Section 13) |
| Visual identity (Tadao Ando grayscale, Geist Mono) | `HIDE_LIVE_ARCHAEOLOGY.md` sec 3.4 | render Track B in a common neutral skin; Track A cannot be neutralized, so its brand recognition is a measured covariate (familiarity stratum), not a controlled variable |
| Offline capability | disconnect the network and one arm keeps working | never test connectivity as part of a blinded task; treat egress-off as a separate disclosed wedge, not a blinded comparison |

Honest ceiling: Track A is single-blind and brand-recognizable to Claude Code power users; Track B is close to double-blind for rendered content but discards live feel. We report both and never claim a double-blind result for the interactive experience.

## 10. Preference elicitation

Per comparison, forced paired choice on five dimensions (no neutral option; a "no preference" is recorded separately and treated as a tie in the paired model):

1. more capable
2. clearer
3. faster (felt, paired-relative only, Section 13)
4. more trustworthy
5. which would you use daily

Plus one free-text "why" per comparison, coded qualitatively (Section 12) and never promoted to an effect size (dossier sec 8.1: anecdotes are regression leads, not product truth; `HIDE_USER_LOVE_PAIN_MAP.md` labels community sentiment ANECDOTAL). "Which would you use daily" is the closest single proxy to retention and is pre-registered as the primary endpoint; the other four are secondary and diagnostic.

## 11. Response-style eval, independent of raw coding capability (Bible sec 33)

Much of felt preference is response STYLE, not code correctness: tone, verbosity, formatting, terseness, willingness to push back, signal-to-noise of tool narration. The March-April 2026 regression included verbosity caps as a product-layer change that degraded perceived quality with the API unaffected (`HIDE_USER_LOVE_PAIN_MAP.md` sec 3), which proves style moves preference independent of the model. Style is also a harness-fixable lever HIDE fully controls without touching the model.

Design as a separable sub-study:

- **Matched-outcome stimuli:** take verified-green task outcomes (byte-identical final patch + identical test-green receipt) and re-render the SAME outcome in each harness's native response style (Claude Code narration vs HIDE terse-telemetry Digest vs minimal bare style). Because the code and the oracle outcome are identical across renderings, any preference is pure style.
- **Neutralize the skin:** render all styles in one common typeface/skin so the eval measures the CONTENT of the response style (verbosity, structure, telemetry-vs-prose, terseness) and not HIDE's brand identity (which is measured separately in Section 9).
- **Rubric (rated 1-5, not paired):** clarity, terseness / signal-to-noise, scan-ability, trust-conveyance (does the narration make me trust the change), confidence-the-work-is-done.
- **False-confidence probe (safety-relevant negative metric):** include a subset where the code is WRONG but the narration is confident. Measure the per-harness **false-confidence rate**: the fraction of raters who report high trust in a wrong outcome. A response style that induces false trust is a defect even if it wins the clarity rubric; this ties the trustworthiness dimension (Section 10) to a measurable harm, and pairs with the deterministic-oracle-first discipline (dossier sec 5.12).

The response-style result is reported as a harness property, model-held-constant, and feeds the harness-value contrasts (Section 3) as the communication-layer component.

## 12. Statistics

- **Design:** within-subject, paired forced choice, clustered by participant and by task. Fit a mixed-effects logistic model: preference ~ arm + dimension + familiarity + language + (1 | participant) + (1 | task). Report per-arm-pair preference proportions with **Wilson 95% CIs** (same interval discipline as `hawking-eval` pass@1 + Wilson CI, `HIDE_LIVE_ARCHAEOLOGY.md` sec 3.5), and paired differences via McNemar plus a participant-clustered bootstrap.
- **Preference scale:** fit a Bradley-Terry / Thurstonian model over all pairwise choices per dimension to place the five cells on one preference scale with CIs, so the factorial contrasts (Section 3) are read as differences on that scale.
- **Decomposition:** estimate model gap (C2-C5), Claude Code harness value (C1-C2), HIDE harness value (C4-C5), and harness portability (C1-C3) as linear contrasts, each with a bootstrap CI. Carry the full spread, not just the mean, and label every number MEASURED vs INFERRED, naming what would upgrade an estimate to measured (Hawking reporting rule: report distribution + spread + estimate-vs-measured, never a bare mean).
- **Multiple comparisons:** Holm correction across the five dimensions. Pre-register the primary endpoint ("use daily"), the MDE (target detectable paired shift ~ 12-15 percentage points), the analysis plan, and the exclusion rules BEFORE data collection; deviations are reported.
- **Power sketch:** to detect a 0.65-vs-0.50 paired preference at 80% power, alpha 0.05 two-sided, needs on the order of tens of discordant pairs; n = 30 participants times 9 tasks yields a few hundred paired judgments, comfortably powered for the primary endpoint and adequate for the four secondaries under Holm. Style and felt-speed sub-studies need fewer (lower within-subject variance). These are targets, not false precision; recompute from the pilot.
- **Anecdote discipline:** free-text is coded into themes and reported as qualitative signal only. No claim reaches `HIDE_SUPREMACY_THESIS.md` on anecdote; every reported effect pins the receipt fields in Section 14 (dossier sec 8.1, sec 8.3).

## 13. Contamination and what a paired delta protects

Running the study while an assistant session is open perturbs absolute latency and throughput: an active Claude desktop session has been measured to inflate/deflate local `dec_tps` by roughly 4-5x on shared unified memory (Hawking bench practice, MEASURED prior). The mitigation is the same one already accepted: **benching with the assistant open is fine for paired relative deltas, because constant contamination cancels in the delta.** Consequences for this study:

- Felt-speed (dimension 3) is reported **paired-relative only** (which of two arms felt faster), never as a between-tool absolute tok/s or seconds claim, unless captured in a clean window with the assistant quit.
- Any absolute latency headline must come from a clean-window capture (see `HIDE_SPEED_FRONTIER.md` for the latency budget and clean-bench discipline) with a pre-flight assistant-process gate.
- Structural and relative measurements (preference proportions, style rubrics, oracle pass/fail) are contamination-robust and need no clean window.

## 14. Required per-session receipt

Every session emits the minimum non-sensitive trace (dossier sec 8.3), so a preference is never floating free of its conditions: task / snapshot / run / trace / parent ids; model, provider, quantization, tokenizer, template, engine, prompt-ABI, tool-ABI versions; harness id and version; context item identities + token counts + sources + retrieval scores; cache hit/write/eviction + reused-token counts; queue / retrieval / prefill / TTFT / decode / tool-gap / tool / verification / total times; state checkpoint/fork/restore sizes and times (for the gated supremacy arms); oracle outcomes; interventions and approvals; final patch identity and accept/undo. Sensitive content capture is opt-in, local, redacted, retention-limited. Absent these fields, a preference number is inadmissible, matching the pin-everything rule for benchmark results (dossier sec 8.1).

## 15. What this study can and cannot conclude

| Claim | Admissible from this study | Basis |
|---|---|---|
| HIDE's harness is at or above Claude Code's harness, model held constant | **yes (PARITY)** | contrast (C4-C5) vs (C1-C2), model-independent by construction |
| A residual product gap is attributable to the model, not the harness | yes, as a bounded estimate | model gap contrast C2-C5 with CI |
| HIDE feels faster because of warm-state resume/fork | only after the gate clears | `HIDE_STATE_CAPSULE_ABI.md` sec 8 exposure + Gate G-CAP-1 (GPU->CPU readback); paired-relative felt-speed only |
| HIDE's two-surface handoff feels seamless (one warm context) | only after the gate clears | `/v1/hide/*` reconnect + shared warm context, `HIDE_TWO_SURFACE_ARCHITECTURE.md` |
| HIDE's permission denials feel safer | only after the gate clears | `perm.rule_engine` wired + OS/Seatbelt enforcement |
| Best-of-N local forks beat one stronger model per second | only if it wins the tie-break | dossier sec 9 kill criterion (quality gain per second vs one stronger model); this study is the instrument that measures it |
| HIDE is "better" in absolute terms | **no** | if the Hawking model is weaker, absolute preference conflates model and harness; the decomposition refuses that claim |

The study's honest job is to license exactly one class of claim without a live model equal: that HIDE's harness reaches parity or supremacy with the model factored out, and to size the remaining model gap as a separate, gated build item. Every stronger claim waits on its named build item and a re-run with the gate cleared.

## Cross-references

- `HIDE_CAPABILITY_DENSITY_EVAL.md`: the objective, oracle-graded coding-capability eval this study complements; preference is read against it, never in place of it.
- `HIDE_EXPERIMENT_MENU.md`: registers Experiment 0 (pure model gap) and each gated supremacy arm as isolated bets with kill criteria.
- `HIDE_SUPREMACY_THESIS.md`: consumes only the gated-and-cleared results here.
- `HIDE_STATE_CAPSULE_ABI.md`, `HIDE_TWO_SURFACE_ARCHITECTURE.md`: the build items that gate the supremacy task variants.
- `HIDE_LOCAL_MODEL_TOPOLOGY.md`: fixes the Hawking model used in C4/C5.
- `HIDE_SPEED_FRONTIER.md`: the latency budget and clean-bench discipline for any absolute felt-speed number.
- `HIDE_USER_LOVE_PAIN_MAP.md`: the brand-habit confound and the no-meter tell.
