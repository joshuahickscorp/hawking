# HIDE User Love / Pain / Captivity Map

Run date: 2026-07-19 · Sources: clean-room facet 9 (user-love-pain) + facet 11 (2026 changelog), independently verified. Community sentiment is labeled ANECDOTAL; pricing/policy facts are DOCUMENTED where a primary source exists.

## 1. What developers love (and would miss most)

| Loved behavior | Evidence | Why it retains |
|---|---|---|
| Terminal-native autonomous loop: understands the whole repo, reads/edits/runs tests/commits end-to-end | ANECDOTAL (strong, consistent) | "Holds the codebase in its head"; removes friction and cognitive load; a working change in minutes without babysitting |
| Steerable autonomy (Esc interrupt-and-keep, soft steer, plan gate) | DOCUMENTED + ANECDOTAL | Turns a monologue into a conversation; catch a wrong turn early instead of paying for a full bad run |
| Reversibility (checkpoints / rewind) | DOCUMENTED | Makes people *allow* autonomy - a bad multi-file swing is one gesture to undo |
| Judgment and willingness to push back | ANECDOTAL | Feels like a capable teammate, not an autocomplete |
| Subagent context isolation | DOCUMENTED | Main thread stays clean and cheap; long sessions don't degrade |
| Persistent project memory (CLAUDE.md) + resume | DOCUMENTED | No re-explaining; pick work back up; **the main source of stickiness** |

**Design implication for HIDE:** the love is concentrated in *steerable autonomy over a genuinely-understood repo*, not in any single feature. HIDE's live archaeology shows the harness that produces this (index + context compiler + kernel loop + tools) exists but is unwired - so the first lovable vertical is a *reconnection*, and it must nail interrupt/plan/rewind polish, because those are the genes users would feel missing on day one.

## 2. What they tolerate (the friction they accept to keep it)

| Tolerated pain | Evidence | Texture |
|---|---|---|
| Two-tier metering: rolling 5-hour session window + weekly caps (all-models + an **Opus-specific** cap), reset 7 days after a session starts | DOCUMENTED (costs page); verifier corrected "Sonnet-only" → Opus-specific | The weekly cap is "the wall" |
| Usage anxiety worsened by `/usage` being computed **only from local session history** (not whole-account) | DOCUMENTED | Users can't trust the number near a wall |
| Cost: a $100 Max plan exhausts quickly on big projects | ANECDOTAL | Especially with agent teams (~7x tokens; reported ~$50-65/day for 5 agents) |
| Permission fatigue: heavy prompting, ~93% approval rate | DOCUMENTED (93% MEASURED); the "~100 prompts/hour" figure is **not** in the source [verifier] | The reason `/sandbox` (−84% prompts) and auto mode exist |
| Context/auto-compact warnings (distinct from the usage wall) | DOCUMENTED | A separate recurring friction |
| Model-access asymmetry (Opus reserved/rate-limited vs Sonnet default) | DOCUMENTED | "Switching to deeper reasoning means switching tools" |

## 3. The trust wound (a durable, specific pain)

The **March 4 - April 20, 2026 quality regression** [DOCUMENTED, post-mortem April 23]: three overlapping *product-layer* changes to Claude Code's harness (a reasoning-effort downgrade high→low, a caching bug, verbosity caps) degraded quality for weeks; the API was unaffected; a ~3% coding-eval drop was measured for both Opus 4.6 and 4.7 [verifier-refined]. Users were **powerless to opt out** of a silent server-side change.

This is the single most important captivity crack: it proves that a cloud harness can regress overnight with no user recourse. It is not a feature gap - it is a *structural* property of a remotely-controlled runtime.

## 4. Captivity / switching friction (what actually locks users in)

| Lock-in source | Strength | Nature |
|---|---|---|
| Workflow muscle memory (CLAUDE.md, skills, plugins, permission model, keybindings) | High | **Soft** - habit + config, not technical |
| No hard technical lock | - | In 2026 developers increasingly route work across multiple tools |
| Ecosystem (marketplaces, shared team skills/plugins) | Medium | Network effect, portable in principle |
| Data gravity (transcripts, memory) | Low - Medium | Local files; portable |

**Conclusion:** captivity is almost entirely *habit and config compatibility*, which is exactly why `HIDE_CLAUDE_CODE_CONFIGURATION_COMPATIBILITY.md` (read CLAUDE.md / agents / skills / MCP verbatim) is the wedge that neutralizes switching cost. Interop is not a cost - it is HIDE's entry ticket.

## 5. The default that most users don't foreground (but is the deepest wedge)

- **Egress is the default architecture** [DOCUMENTED]: Claude Code always transmits code/context to Anthropic's cloud by default. It is not merely a "concern"; it is how the tool works.
- **Training on data by default** for Free/Pro/Max since the Aug/Sept 2025 consumer-terms update, unless the user opts out, with extended (~5-year) retention for those who allow [DOCUMENTED, verifier-added]. API/Console/Enterprise are excluded.

This is understated in most community discussion and is the strongest privacy wedge: regulated/contractual/air-gapped teams currently resort to pointing Claude Code at *local* models as a workaround - evidence that a first-class local runtime is a demanded, not hypothetical, product.

## 6. The wedge map - where a local Hawking runtime structurally wins

Ranked by strength of the opening (INFERRED from the pain map, each tied to a HIDE mechanism):

1. **No usage meter / no weekly wall / no per-token anxiety.** Removes the *entire* tolerated-pain cluster in §2. HIDE replaces `/usage` with performance telemetry (tokens/s, energy, best-of-N depth) - capability headroom, not a countdown. Mechanism: local inference.
2. **No code egress; true offline / air-gapped.** Eliminates the privacy wedge (§5) as a *hardware fact*, not a proxy allowlist. Mechanism: local inference + egress-default-off.
3. **No silent server-side regressions.** A version-pinned local runtime cannot be remotely degraded; "my tool got worse overnight" is impossible by construction (§3). Mechanism: local weights + harness + prompts, user-controlled updates.
4. **Instant warm-state resume / fork instead of cloud context reprocessing.** Resume restores the KV/recurrent capsule (no re-prefill); best-of-N and forks are near-free. Mechanism: state capsule ABI (RWKV lane today; transformer lane a build item).
5. **Unmetered parallelism.** Agent teams / background fleets bounded by local hardware, not a 10x-quota warning. Mechanism: warm-state forks + local fleet.

## 7. What HIDE must NOT do with the pain

- Do not gloat about or foreground competitor pain in the product surface. The doctrine (`DESIGN_DOCTRINE.md`) already forbids a budget meter; keep it forbidden.
- Do not import the anxiety loop by adding any dollar HUD "for familiarity."
- Do not over-promise offline capability the model can't back: state that local capability density is gated on a capable local coder (Qwen3-Coder-Next-class), which is a model build item, not a harness one. Honesty here is itself a trust advantage over the party that regressed silently.

## 8. One-line synthesis

> Users love Claude Code for steerable autonomy over a repo it understands, tolerate its meter and prompts, were wounded by a silent regression, and are held mostly by habit and config. HIDE wins the transition by reading their config verbatim, then removes the meter, the egress, and the remote-regression risk - turning every tolerated pain into a structural advantage.
