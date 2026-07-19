# HIDE Permission and Effect System

Run date: 2026-07-19
Doctrine home: Bible Book X (ch.10) sec 66 (permission gate), sec 20, sec 34.
Grounding: `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json` (ids `perm.rule_engine`, `perm.persistence_tiers`, `perm.auto_mode`, `perm.mode_cycle`, `trust.workspace_gate`, `perm.plan_mode`, `security.sandbox`); `HIDE_LIVE_ARCHAEOLOGY.md` §3.2, §3.5, §5. All `hide-*` file:line evidence is pinned to the sealed backend commit `5a99d0e2` (packed; recover with `git checkout 5a99d0e2 -- crates`), unless labeled otherwise.
Sibling division of labor: `HIDE_SECURITY_CONSTITUTION.md` owns trust-before-config (Article I), egress default-off (Article III), untrusted-content isolation (Article V), secrets (Article VI), and the sandbox charter (Article VII). This document owns the **effect taxonomy, the rule engine, persistence tiers, permission modes, the auto-mode classifier, batching/simulate, and the typed effect ledger**, which Article IV of the Constitution delegates here by name.
Status: specification for a Hawking-native mechanism; every claim is tagged by the readiness of the primitive it depends on. Readiness key: **real-and-wired** (reachable on a shipping path), **real-but-unwired** (built and tested, no live caller), **partial**, **stub**, **missing**.

## 1. The thesis: a typed effect ledger, not a prompt string

Claude Code's permission layer matches actions against string patterns (`Bash(git commit:*)`) and enforces a sandbox at the OS boundary; it is a strong design and the parity target. HIDE's structural improvement is one level deeper: **every action is first lowered to a typed `Effect` object with a content hash, a risk level, and provenance, and the permission decision is a function of that object, not of a re-parsed command string.** The effect set is simultaneously the thing the user approves, the thing the audit chain records, and the thing the sandbox is rendered from. This is why the substrate already exists as real code:

- `EffectKind` and `Effect` are a live typed enum with a `bytes_hash` and a `risk` field (`hide-core/src/types.rs:99-117`) [VERIFIED REPO @5a99d0e2].
- The permission engine consumes an `Effect` list, not a raw string (`PermissionRequest.effects`, `hide-core/src/permission.rs`) [VERIFIED REPO].
- Grants can be **bound to an exact effect hash** so a stored "allow" only re-fires for the identical effect (`CapabilityGrant.exact_effect_hash`, config default `require_exact_effect_grants: true`, `config.rs`) [VERIFIED REPO].

A pure-prompt system cannot make a stored grant safe against a subtly different command; an effect-hash-bound grant can. That is the whole moat: the ledger is typed, so precedence, persistence, simulation, sandbox rendering, and audit are all reads of one object.

**Honest status:** the ledger substrate is real but **packed and unwired** (`perm.rule_engine.hide_status = packed_unwired`; archaeology §3.5). Nothing on the live turn evaluates it today. This document specifies the object and the build items that make it load-bearing; it does not present a packed primitive as shipping.

## 2. Effect taxonomy

The parity contract requires a rich effect vocabulary (Constitution Article IV): `read · write · process · network · git-mutation · package-install · secret-access · destructive · external-side-effect`. The live enum is a coarser real substrate; the extended taxonomy is a build item that refines it without a schema break (additive variants, same evaluation).

| Parity effect class | Live `EffectKind` today | Readiness | Note / build item |
|---|---|---|---|
| read | `Read` (`types.rs:100`) | real-but-unwired | maps 1:1 |
| write | `Write` (`types.rs:101`) | real-but-unwired | maps 1:1 |
| destructive | `Delete` (`types.rs:102`) | real-but-unwired | `Delete` is the destructive substrate; refine into `destructive` (irreversible) vs ordinary `write` |
| process | `Execute` (`types.rs:103`) | real-but-unwired | maps 1:1 (`shell.exec`) |
| network | `Network` (`types.rs:104`) | real-but-unwired | already carries the one implemented pre-gate (Section 4.1) |
| git-mutation | (subclass of `Execute`/`Write`) | missing | add a first-class variant so `git commit`/`push` are gated distinctly from a build |
| package-install | (subclass of `Execute`/`Network`) | missing | `npm i`/`pip install` = process + network + write; add a fused variant so it is one approvable effect |
| secret-access | (guarded in redaction/sandbox, not an `EffectKind`) | partial | denied at the sandbox read layer (Section 11) and scrubbed by the redactor; promote to a first-class effect for the ledger |
| external-side-effect | (subclass of `Network`) | missing | messages/PRs/webhooks: irreversible egress, always `ask` at minimum, never speculated (Section 10) |
| model | `Model` (`types.rs:105`) | real-but-unwired | Hawking-native: a local decode is a $0.00 effect, still ledgered |
| plugin | `Plugin` (`types.rs:106`) | real-but-unwired | plugin/MCP tool invocation |
| (unclassified) | `Unknown` (`types.rs:107`) | real-but-unwired | fail-safe: `Unknown` must route to `ask`, never `allow` |

Each effect is a typed record, not a label:

```text
Effect {                               // hide-core/src/types.rs:110-117 [VERIFIED @5a99d0e2]
  kind:       EffectKind
  target:     String                   // path, domain, argv-join, tool id
  bytes_hash: Option<String>           // content hash of the write/patch/command -> grant binding
  risk:       RiskLevel                // Trivial | Low | Medium | High | Critical (types.rs:19-25)
  metadata:   BTreeMap<String,String>  // provenance, tool, cwd, ...
}
EffectSet { effects: Vec<Effect> }     // types.rs:119-122; the batching unit (Section 10)
```

**Rule (fail-safe):** an action whose effect cannot be classified is `Unknown` and therefore `ask`; classification is never optional. Provenance-labelled untrusted content that reaches a tool is handled by Constitution Article V, not here.

## 3. The rule engine

Parity id `perm.rule_engine` requires: an ordered `deny -> ask -> allow` evaluator, first-match-wins, shell compound-command decomposition, word-boundary globs, gitignore-style path scoping, the domain/tool/agent matchers, a protected-path pre-gate, and a non-defeatable destructive circuit breaker. The live engine implements the **precedence core** and a **narrow glob**; the matchers and decomposition are the build items. This is stated honestly below, matcher by matcher.

### 3.1 Precedence (real-but-unwired, matches parity)

`StaticPermissionEngine::evaluate` (`hide-core/src/permission.rs`) already enforces the required order [VERIFIED REPO @5a99d0e2]:

1. **Pre-gate** (implemented, one case): any `Network` effect at `risk >= High` is forced to `Ask` before any rule is consulted. This is the pattern the protected-path pre-gate (Section 5) generalizes.
2. **Deny wins:** the first matching rule with `Decision::Deny` returns `Deny`.
3. **Then allow:** the first matching rule with `Decision::Allow` returns `Allow`.
4. **Else default:** `PermissionPolicy.default_decision`, which defaults to `Ask` (`permission.rs`; `SecurityConfig.default_decision = Ask`, `config.rs`).

So `deny -> ask(default) -> allow` precedence and first-match-wins are **real**. `Decision` is exactly the parity triad `{Allow, Ask, Deny}` (`types.rs:29-33`). The default posture is deny-by-approval: `network_default = Deny`, `shell_default = Ask`, `workspace_write_default = Ask` (`config.rs`) [VERIFIED REPO].

### 3.2 Matchers (the build-out)

A `PermissionRule` today is `{capability_kind, scope_pattern, decision, max_risk, reason}` and matches when `capability_kind` is an exact string equal, `pattern_matches(scope_pattern, target)` holds, and `request.risk <= max_risk` (`permission.rs`) [VERIFIED REPO]. The live `pattern_matches` supports only `*` / `**` (any), `prefix/**` (subpath), and exact equality. Everything richer is unbuilt:

| Parity matcher (`perm.rule_engine`) | Live support | Readiness | Build item |
|---|---|---|---|
| `deny -> ask -> allow`, first-match-wins | yes (`evaluate`) | real-but-unwired | wire onto the live turn |
| gitignore-style path scoping (`Read`/`Edit`) | `prefix/**` + exact only | partial | replace `pattern_matches` with a gitignore-semantics matcher (negation, `**` mid-path, extension globs); resolve canonical path + symlink **before** matching (Constitution Article II) |
| word-boundary globs (`Bash(git commit:*)`) | exact `capability_kind` only | missing | tokenize the command, match on the program + word-boundary argument globs, not substring |
| shell compound-command decomposition | none | missing | split on `;`, `&&`, `\|\|`, pipes, `$()`, backticks into sub-commands and evaluate **each**; the whole compound is allowed only if every sub-command is allowed (deny/ask on any sub-command gates the whole) |
| `WebFetch(domain:*)` | none | missing | domain matcher over the `Network` effect target; pairs with the egress broker (Constitution Article III) |
| `mcp__server__tool` | none | missing | matcher over `Plugin` effect target `server/tool` |
| `Agent(Name)` | none | missing | matcher gating subagent spawn by name (feeds `subagents.file_defined`) |
| symlink dual-path (allow needs both, deny needs either) | none | missing | evaluate both the link path and its canonical target; a deny on **either** denies, an allow requires **both** |

**Compound decomposition is the highest-value correctness item.** Without it, `capability_kind == "shell.exec"` matches the whole `argv`, so `safe_cmd && rm -rf /` can slip past a naive allow. The build item makes the effect lowering emit one `Effect` per decomposed sub-command, after which the existing precedence core gates each one. The circuit breaker (Section 5) is the backstop, not the primary defense.

## 4. Read-only allowlist that never prompts

Parity (`perm.rule_engine`, `perm.persistence_tiers`): a read-only Bash allowlist never prompts in any mode; read-only effects never prompt. HIDE has the honest classifier for this already in the tool ABI:

```text
Purity { Pure, PureFs, Impure }        // hide-core/src/tool.rs:230 [VERIFIED @5a99d0e2]
```

`Tool::purity()` declares whether a tool is `Pure` (no filesystem, no side effect), `PureFs` (read-only filesystem), or `Impure`. **Rule:** an effect set whose every effect is `Read`/`Model` from a `Pure`/`PureFs` tool auto-resolves to `Allow` and is never surfaced, in every permission mode. This is the mechanism behind the parity claim "read-only never prompts" and is the single largest prompt-fatigue reducer (Claude Code measured ~93% of prompts approved: fatigue is a security problem, not a UX one; `perm.auto_mode` evidence, Constitution Article IV).

Readiness: `Purity` is **real-but-unwired**; the auto-allow rule that reads it is a build item on the reintegrated gate.

## 5. Protected-path pre-gate and the destructive circuit breaker

Parity (`perm.rule_engine`): protected paths (`.git`, `.claude`, rc files) route to a prompt **before** any allow rule can fire, and an `rm -rf /`-class circuit breaker fires **even in a bypass mode**.

**Circuit breaker (real-but-unwired, coarse).** `dangerous_command(argv)` is live code (`hide-backend/src/host.rs:1096-1131`, tested `:1282-1293`) [VERIFIED REPO @5a99d0e2]. It catches `sudo`/`doas`, `mkfs`/`mkfs.*`, `dd of=/dev/*`, `rm -rf` (also `-fr`, `-r -f`) targeting `/`, `~`, or `/*`, `curl|wget ... | sh|bash`, the `:(){ :|:& };:` fork bomb, and recursive `chmod`/`chown` on `/` or `~`. It is deliberately conservative: `rm -rf node_modules`, `cargo test`, `git push origin main` pass. A caught command is **not dropped**: it is parked in a bounded `GateBook` under a fresh gate id (`hold`/`take`/`approve_gate`/`deny_gate`, `host.rs:1048-1090`, gate applied at `host.rs:358-361`) awaiting an explicit approve/deny round-trip. A second `CATASTROPHIC` list backstops the shell tool (`hide-tools/src/shell.rs:40`).

Honest gaps versus parity:
- **Substring scan, not decomposed argv.** `dangerous_command` lowercases and joins the argv and scans; it does not decompose a compound command (Section 3.2). Build item: run the breaker over each decomposed sub-command so `echo ok && rm -rf /` is caught on the second clause.
- **Not yet proven non-defeatable in bypass.** The breaker must sit **above** the mode system so `bypassPermissions`/auto cannot skip it. Today it is a gate on one shell path, not a global invariant. Build item: make the breaker a pre-mode invariant in the reintegrated turn.

**Protected-path pre-gate (missing).** No live rule routes `.git`/`.claude`/rc writes to a prompt ahead of allow rules. Build item: add a pre-gate (mirroring the implemented network pre-gate in `evaluate`, Section 3.1) that forces `Ask` on writes/deletes to protected paths regardless of a matching allow rule. Canonical-path + symlink resolution runs first (Constitution Article II).

## 6. Tiered persistence

Parity id `perm.persistence_tiers`: the prompt offers **allow-once / allow-for-session / allow-permanently (persisted, inspectable, per-repo) / deny**, with persistence scope matched to risk; read-only never prompts; a shell "don't ask again" persists permanently per-repo+command; a file-edit approval persists only until session end.

The grant object already carries the fields to express all four tiers [VERIFIED REPO @5a99d0e2]:

```text
CapabilityGrant {                      // hide-core/src/permission.rs
  id, capabilities, decision,
  granted_by:       User | Policy | System,
  run_id:           Option<RunId>,     // session/run scoping -> allow-for-session
  expires_at_ms:    Option<TimestampMs>,// allow-once (short TTL) / expiry
  exact_effect_hash:Option<String>,    // grant binds to the exact effect (config require_exact_effect_grants=true)
  ...
}
```

| Tier | Mechanism | Readiness |
|---|---|---|
| allow-once | `expires_at_ms` short TTL or single-use consume | real-but-unwired (field present; consume logic to wire) |
| allow-for-session | grant scoped to `run_id`; dropped at session end | real-but-unwired |
| allow-permanently (per-repo) | grant persisted to the per-repo settings store, inspectable/revocable | partial: the durable grant store is the build item; the grant record is real |
| deny | `Decision::Deny` rule, always wins (Section 3.1) | real-but-unwired |

**Risk-matched persistence (the parity nuance):** a `shell.exec` grant may persist permanently per-repo+command; a `Write`/`Delete` grant should default to session scope. Because `require_exact_effect_grants` defaults true, a persisted grant only re-fires for the **identical effect hash** (`bytes_hash`), so a permanent shell allow cannot silently cover a different command; a re-shaped command mismatches the hash and re-prompts. This is the effect-ledger advantage over a pure string allowlist, stated in Section 1, made concrete for persistence. Build item: the per-repo durable grant file (the analogue of `settings.local.json`) plus an inspection/revocation surface. Migration and precedence of an imported Claude Code `settings.local.json` are covered by `HIDE_CLAUDE_CODE_CONFIGURATION_COMPATIBILITY.md`.

## 7. Permission modes and the always-visible badge

Parity id `perm.mode_cycle`: a single cycling chord (Shift+Tab) over an ordered mode set `default(Manual) / acceptEdits / plan (+ optional bypassPermissions / auto)`, an always-visible mode badge, plus a `--permission-mode` flag and a `permissions.defaultMode` setting. Parity id `perm.plan_mode`: plan mode blocks source edits at the executor (not by prompt) until a written plan is approved via a graded dialog.

| Mode | Effect-gate behavior | Live status |
|---|---|---|
| Manual (default) | every non-`Pure`/`PureFs` effect gated per Section 3 | rule engine real-but-unwired |
| acceptEdits | `Write` to workspace auto-allowed; `Delete`/`Network`/`Execute` still gated | build item over the rule engine |
| plan | executor structurally blocks `Write`/`Delete`/`Execute` (not a prompt); reads + `Pure` shell allowed until the plan is approved | FE PlanCard exists (`ui_only`); enforcement needs the kernel + tool gate |
| bypassPermissions (optional) | allow rules auto-fire; **the circuit breaker (Section 5) still fires** | build item; breaker must be pre-mode |
| auto (optional) | the classifier gates (Section 8) | missing |

Live truth: the FE has a bypass/auto-approve gate and a partial mode surface (`perm.mode_cycle.hide_status = partial`; `perm.plan_mode.hide_status = ui_only`; archaeology §3.4), but the full ordered cycle, the enforced write-block, and the always-visible badge need the reintegrated kernel and tool gate. The mode is a permission-mode field on the session that the effect gate reads; because both surfaces read one store (`HIDE_TWO_SURFACE_ARCHITECTURE.md`), the badge and the gate are identical in Chat and IDE by construction. **Supremacy (`perm.plan_mode`, gated on state-capsule fork exposure):** best-of-N candidate plans can be executed speculatively in isolated local forks so the user picks the plan that already passed its oracles, not a guess (`HIDE_STATE_CAPSULE_ABI.md` §7; gated on the fork-exposure build items there).

## 8. The auto-mode classifier

Parity id `perm.auto_mode` (MEASURED, Sonnet-4.6-class: ~83% catch / 0.4% false-positive / 17% false-negative): a **separate classifier model** gates every non-trivial action; it sees the user messages, the tool calls, and the rules, but **NOT raw tool-result content**; it auto-approves benign work, blocks escalation/exfiltration/destructive actions, falls back to prompting after repeated blocks, and drops broad allow rules on entry.

Live status: **missing** (`perm.auto_mode.hide_status = absent`). This is a build item, specified here so it is built with the isolation discipline the measured numbers depend on:

1. **Independent pass.** The classifier is a distinct evaluation over the `EffectSet` + the rule state + the user turn, run **before** an `Impure` effect executes. It is not the coding model judging itself.
2. **Isolated from hostile tool-result content.** The classifier input **excludes raw tool-result bytes** (the injection vector). It reads the typed effects, the declared targets, and provenance labels, not the untrusted payload. Untrusted-content isolation is Constitution Article V; the classifier is the consumer of that boundary.
3. **Documented block taxonomy.** Blocks are typed (privilege escalation, exfiltration, destructive/irreversible, protected-path, secret-access), so a block is auditable and a fallback is explainable, not a black-box refusal.
4. **Graceful fallback.** After N consecutive blocks the mode falls back to prompting the user rather than looping; entering auto mode drops broad allow rules so a poisoned prior grant cannot pre-authorize.

**Supremacy (gated on state-capsule fork exposure).** HIDE runs the classifier as a **warm local fork** (`HIDE_STATE_CAPSULE_ABI.md`): no network round-trip, no metered cost, no transcript egress, so it can run per action (even mid-turn) instead of being rationed, and **best-of-N safety judges** can vote to push the single-classifier 17% false-negative down. Because egress is default-off (Constitution Article III), the exfiltration block class is structurally smaller: there is no approved-domain path for a classifier to miss. Every one of these gains is gated on the fork-exposure build items in `HIDE_STATE_CAPSULE_ABI.md` §8; until they land, the classifier is a single local pass, which is already a parity-complete design.

## 9. Simulate-first and explain-effect

Parity (Constitution Article IV): simulate-first and explain-effect on demand. HIDE's tool ABI already carries the simulate seam [VERIFIED REPO @5a99d0e2]:

```text
Tool::simulate(&args, ctx) -> Option<EffectSet>   // hide-core/src/tool.rs:219
ToolCall.x.dry_run: bool                            // tool.rs:67; ToolOutput.from_dry_run stat tool.rs:138
```

`ShellRun::simulate` returns the `EffectSet` a command **would** produce without running it (`hide-tools/src/shell.rs:138`). This is the substrate for two behaviors:
- **explain-effect:** the gate renders the simulated `EffectSet` (targets, kinds, risks, and the sandbox profile that would apply) so the user approves the *effects*, not a command string they have to parse.
- **simulate-first:** high-risk effects can be dry-run to populate the ledger before a real execution.

Readiness: `simulate` is **real-but-unwired**; wiring it into the gate render is a build item. **Speculation limit (Constitution Article IV / dossier §5.7):** only read-only, local, private, side-effect-free, idempotent effects may be speculatively *prepared*; HIDE must **never** speculate a mutation, an external request, a secret access, a message, a write, a delete, or a credential entry (even a discarded external request leaks intent). The `Purity` classifier (Section 4) plus the effect kinds are exactly the predicate that decides speculation eligibility.

## 10. Effect batching

Approvals are for meaningful exceptions and irreversible effects, not every low-risk step (Constitution Article IV). The batching unit is the typed `EffectSet` (`types.rs:119-122`), not a per-call prompt:

- A plan step that emits many `Read`/`PureFs` effects resolves to a single auto-allow (Section 4), never N prompts.
- A multi-file edit presents **one** grouped approval over its `EffectSet` (one diff, one decision), and the resulting `CapabilityGrant` binds to the set's combined effect hash.
- Coalescing is safe **only** within one risk tier and one effect class; a batch that mixes a `Write` with a `Delete`/`Network`/`external-side-effect` splits so the irreversible effect gets its own explicit approval. `Unknown` never batches.

Readiness: the `EffectSet` type is real-but-unwired; the batching policy is a build item on the gate. This is where the parity claim "avoid approval spam" is honored without weakening the deny/circuit-breaker guarantees.

## 11. OS-syscall enforcement: a denied capability is physically absent

The typed effect ledger decides *policy*; the Seatbelt sandbox makes a denial *physical*. Probabilistic model defenses cannot replace deterministic environment boundaries (dossier §5.9). HIDE's Seatbelt renderer is **pure logic, real and tested** [VERIFIED REPO @5a99d0e2]:

- `render_macos_seatbelt` / `render_macos_seatbelt_with` (`hide-security/src/sandbox.rs`) emit an SBPL profile that is **`(deny default)`**, with a **process-exec allowlist** (empty allowlist renders `(deny process-exec*)`: nothing may exec unless the exact binary is granted), **filesystem read broad-but-bounded / write confined to the worktree**, **secret read-denies** (`~/.ssh`, `~/.aws`, `~/.config/gh`, `.env`, `*.pem`, `*.key`), and a **`.hide/log` write-deny** so the audit log is untouchable from inside the sandbox.
- `SandboxProfile { tier, read_roots, write_roots, allowed_commands, network }` (`hide-core/src/security.rs:65`); `NetworkPolicy` defaults to `Deny` (`security.rs:90`); the only rendered egress route is the host proxy port.
- **Fail-closed:** on a platform with no usable OS sandbox the spawn **refuses** rather than running unconfined (`shell.rs:363`; `sandbox_exec_available()` checks `/usr/bin/sandbox-exec`, `shell.rs:336`).

So when the effect ledger denies `Network`, the rendered profile has no egress route: the capability is not merely un-approved, it is **absent from the process**. A prompt-only system leaves the syscall reachable; HIDE removes it.

**Honest readiness (`security.sandbox.hide_status = packed_unwired`; archaeology §3.5, §5):** the profile rendering and the `sandbox-exec` spawn are real logic and pass tests, and macOS OS enforcement runs through `sandbox-exec`. The **egress proxy** (the out-of-sandbox broker the profile points its one network route at) and Linux `bubblewrap`+seccomp / microVM paths are **seams**, not built. The mapping from a denied effect kind to a rendered profile clause is the build item that closes the loop between Sections 3 and 11. Egress-broker design and the network-default-off doctrine are owned by `HIDE_SECURITY_CONSTITUTION.md` Article III; the trust-before-config gate (project `settings.json` allow-rules inert until a trust dialog; `trust.workspace_gate = absent`; serve currently binds `0.0.0.0` with no auth, archaeology §3.2 G10) is owned by Article I. This document assumes both as prerequisites: **no rule, grant, hook, or MCP definition from project config is evaluated before trust is accepted.**

## 12. Durability of the decision (audit + at-rest)

Every evaluated effect and every grant is an event on the tamper-evident chain, so the permission system is also a record:

- **blake3 hash-chain** over `prev_hash || canonical_event_bytes`, with a **per-workspace genesis salt** so a chain cannot be transplanted between logs (`hide-security/src/audit.rs`, `compute_event_chain` / `verify_event_chain` + salted variants) [VERIFIED REPO @5a99d0e2].
- **Secret redaction before durability:** signature detectors (AWS, GitHub/GitLab PAT, PEM, JWT, Slack) plus a Shannon-entropy detector; a hit becomes `«redacted:detector»` and `redact_json` emits RFC 6901 pointers for `Event.redactions`, so a secret in a tool result never enters the log while the redaction stays auditable (`hide-security/src/redaction.rs`).
- **AES-256-GCM at-rest** with a random 256-bit workspace data key and a per-segment 96-bit nonce; authenticated open (a tampered segment fails the GCM tag). Default off, `enabled()` posture on (`hide-security/src/storage.rs`).

Readiness: audit chain, redaction, and at-rest are **real-but-unwired** (packed logic + tests; no live caller). Constitution Article VIII owns the durability charter; this document notes only that a permission decision is a first-class, redacted, chained event.

## 13. Readiness ledger (what ships vs what is a build item)

| Mechanism | Parity id | Live location (@5a99d0e2 unless noted) | Readiness |
|---|---|---|---|
| Typed `Effect` / `EffectSet` object | `perm.rule_engine` | `types.rs:99-122` | real-but-unwired |
| `deny -> ask -> allow`, first-match-wins | `perm.rule_engine` | `permission.rs` `evaluate` | real-but-unwired |
| Network high-risk pre-gate | `perm.rule_engine` | `permission.rs` | real-but-unwired |
| Effect-hash-bound grants | `perm.persistence_tiers` | `permission.rs` + `config.rs` | real-but-unwired |
| Read-only auto-allow (`Purity`) | `perm.rule_engine` | `tool.rs:230` | real-but-unwired |
| Circuit breaker (`dangerous_command`) | `perm.rule_engine` | `host.rs:1096`, `shell.rs:40` | real-but-unwired (coarse; needs decomposition + pre-mode placement) |
| GateBook park-then-approve | `perm.rule_engine` | `host.rs:1048` | real-but-unwired |
| Simulate / explain-effect | Constitution IV | `tool.rs:219`, `shell.rs:138` | real-but-unwired |
| Persistence tiers (grant fields) | `perm.persistence_tiers` | `permission.rs` | real-but-unwired (durable per-repo store missing) |
| Seatbelt render + fail-closed spawn | `security.sandbox` | `sandbox.rs`, `shell.rs:336-363` | real-but-unwired (OS enforce via `sandbox-exec` real; egress proxy + Linux paths = seams) |
| blake3 audit chain / redaction / AES-256-GCM | Constitution VIII | `audit.rs`, `redaction.rs`, `storage.rs` | real-but-unwired |
| Compound-command decomposition | `perm.rule_engine` | none | missing |
| Word-boundary / gitignore / domain / mcp / agent matchers | `perm.rule_engine` | `pattern_matches` (glob only) | partial/missing |
| Protected-path pre-gate | `perm.rule_engine` | none | missing |
| Permission mode cycle + always-visible badge | `perm.mode_cycle` | FE partial (archaeology §3.4) | partial |
| Plan-mode enforced write-block | `perm.plan_mode` | FE PlanCard | ui_only |
| Auto-mode classifier | `perm.auto_mode` | none | missing |
| Trust-before-config gate | `trust.workspace_gate` | none (Constitution Article I) | missing |

## 14. What this buys, stated conservatively

**Parity (reproduce Claude Code):** the precedence core, effect-hash-bound grants, the read-only allowlist, the circuit breaker, the simulate seam, and the Seatbelt renderer are already real; parity is the reintegration of the packed `hide-core`/`hide-security`/`hide-backend` gate onto the live turn plus the four missing matchers, the protected-path pre-gate, the durable per-repo grant store, the full mode cycle, and the auto-mode classifier. None of it is greenfield invention.

**Supremacy (do better than Claude Code), each gated:**

| Supremacy claim | Gated on |
|---|---|
| A denied capability is physically absent (not just un-approved) | Seatbelt OS-enforcement seam closed + effect-kind -> profile-clause mapping (Section 11) |
| Exfiltration block class is structurally smaller | egress default-off (Constitution Article III) |
| Grant safety survives command drift | effect-hash binding wired (real; needs the durable store) |
| Classifier runs per action at $0.00, best-of-N judges | state-capsule fork exposure (`HIDE_STATE_CAPSULE_ABI.md` §8) |
| Speculative best-of-N plan execution in isolated forks | state-capsule fork exposure |
| Reversible pre-run snapshot before a risky effect | state-capsule save/fork exposure |

No supremacy claim is asserted as shipping. Every one names the build item it waits on, in keeping with the honest-readiness discipline of `HIDE_STATE_CAPSULE_ABI.md` and the archaeology's core finding: the parts are real, the wiring is the work.
