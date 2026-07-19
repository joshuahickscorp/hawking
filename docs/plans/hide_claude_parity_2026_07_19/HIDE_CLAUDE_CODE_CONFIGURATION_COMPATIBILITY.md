# HIDE Claude Code Configuration Compatibility

Run date: 2026-07-19 · Target: Claude Code v2.1.x · All items DOCUMENTED unless noted.
Goal (Bible §18): *a Claude Code project should migrate to HIDE with minimal configuration rewriting.* This document specifies the exact file classes, locations, and precedence a HIDE **migration reader** must honor. It does not promise byte-exact compatibility until tested - it defines what to build and test against.

## 1. Strategy: read verbatim, own the runtime

HIDE reads Claude Code config files **verbatim** (same filenames, same shapes) so an existing repo works unchanged, while HIDE keeps its own native config namespace for HIDE-only capabilities (state capsules, model-role topology, fleet). A repo may carry both; HIDE's native files take precedence over the imported Claude-compatible ones at the same scope.

Two independent precedence orders must be implemented separately - they are not the same:

- **Instructions** (CLAUDE.md family): `Managed > User > Project > Local`, loaded broad-to-specific so **more-specific is read last and wins**.
- **Settings** (settings.json family): `Managed > CLI > Local > Project > User`, with **permissions MERGING** rather than overriding, and `deny` always winning.

## 2. Instruction files (the CLAUDE.md family)

| File | Location | Load behavior |
|---|---|---|
| Project instructions | `./CLAUDE.md` and `./.claude/CLAUDE.md` | Load every session; git-shared |
| Ancestor instructions | every `CLAUDE.md` walking up to repo root | Load at launch, root-first concatenation |
| Subdirectory instructions | `CLAUDE.md` in a subtree | **Lazy** - load only when a file in that subtree is read |
| Local overrides | `CLAUDE.local.md` | Appended after the shared file in the same dir; gitignored |
| Modular rules | `.claude/rules/**/*.md` | Un-scoped rules load at launch (CLAUDE.md priority); `paths:` frontmatter globs load a rule only on Read of a matching file |
| User global | `~/.claude/CLAUDE.md` | Loaded for all projects |
| Managed | `managed-settings.json` `claudeMd` key | Injected as managed instructions |

Reader obligations:
- `@path` imports: resolve relative to the importing file, plus absolute and `~/` paths; recurse to **depth 4**; skip `@` inside backticks / fenced code; first external import gates behind a one-time approval. [DOCUMENTED]
- Strip block-level HTML comments (`<!-- ... -->`) before injection; preserve comments inside fenced code. [verifier-added]
- Honor `claudeMdExcludes` globs.
- `--add-dir` does NOT load that dir's CLAUDE.md unless `CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD=1`. [verifier-added]
- After compaction, project-root CLAUDE.md is re-read from disk; nested subdir files are NOT auto re-injected (reload on next touch). [verifier-added - matters for HIDE's warm-state caching]

## 3. Auto-memory

- Location: `~/.claude/projects/<repo-derived-key>/memory/MEMORY.md` + linked topic files; machine-local; repo-keyed.
- Load: first **200 lines / 25KB** of MEMORY.md at start.
- Migration: HIDE reads an existing tree at start. HIDE's own memory is a warm-state capsule (no 200-line cliff); on migration it ingests MEMORY.md as seed facts. The `#` quick-capture shortcut is **not in current docs** [verifier] - HIDE provides quick-capture but need not name it `#`.

## 4. Settings JSON layer

| Scope | Path | VCS |
|---|---|---|
| User | `~/.claude/settings.json` | private |
| Project | `.claude/settings.json` | shared |
| Local | `.claude/settings.local.json` | gitignored |
| Managed | OS-specific `managed-settings.json` (+ `managed-settings.d/*.json`, numeric-prefix ordered) | admin |

- Precedence `Managed > CLI > Local > Project > User`; **permissions arrays MERGE**; `deny` wins at every scope.
- Managed Windows path is `C:\Program Files\ClaudeCode\managed-settings.json` (legacy `C:\ProgramData\...` dropped v2.1.75 - a compat reader on old machines may still find it). [verifier-added]
- Key surface is large (not ~15): includes `permissions.*`, `availableModels`/`enforceAvailableModels`, `sandbox.*`, `editorMode`, `alwaysThinkingEnabled`, `fastMode`, `effortLevel`, `autoCompactEnabled`, `statusLine`, `outputStyle`, `hooks`, `env`, `cleanupPeriodDays`, `pluginConfigs`, managed-only lockdown keys. [verifier-added] HIDE maps the ones with a HIDE analogue and preserves the rest as pass-through it can display.

## 5. Permission rules (the security-critical part)

The rule engine is a first-class migration target (see `HIDE_PERMISSION_AND_EFFECT_SYSTEM.md` for the model). A reader must faithfully evaluate:
- `allow` / `ask` / `deny` arrays, **deny → ask → allow**, first-match-wins.
- Matcher grammar: `Bash(git commit:*)` word-boundary globs with **compound-command splitting** (each subcommand must independently match); process-wrapper stripping (`timeout`, `nice`, `nohup`, `xargs`) but NOT env-runners (`npx`, `docker exec`); `Read`/`Edit` gitignore-style path scoping (`//abs`, `~/home`, `/settings-relative`, `./cwd`); `WebFetch(domain:*)`; `mcp__server__tool`; `Agent(Name)`; `Cd`.
- Symlink dual-path: `allow` requires both link and target to match; `deny` matches if either does. [verifier-added]
- Protected paths (`.git`, `.claude`, shell rc, package configs) route to prompt **before** allow rules; `rm -rf /`/`~` circuit breaker even in bypass (detects `$(...)`, backticks, `<(...)`). [verifier-added]

## 6. Extensibility config

| Class | Location | Migration note |
|---|---|---|
| Subagents | `.claude/agents/*.md` + `~/.claude/agents/*.md` | Frontmatter: name, description, tools, disallowedTools, model(`inherit`), skills, mcp, hooks, memory, permissions. `disallowedTools` applied before allow. Read verbatim. |
| Skills | `.claude/skills/<name>/SKILL.md` (+ `~/.claude/skills`, plugin skills) | Frontmatter: name, description, allowed-tools, disable-model-invocation, user-invocable, context(fork+agent), model, effort, paths. `${CLAUDE_SKILL_DIR}` dir var. Custom `.claude/commands/*.md` merged into skills. |
| Plugins | `.claude-plugin/plugin.json` + marketplace.json | Bundles skills/agents/hooks/.mcp.json/.lsp.json/monitors/themes/output-styles/settings.json + userConfig(sensitive). SHA-pinned. |
| Hooks | `hooks` in settings / plugin `hooks/` | Event taxonomy + matchers + JSON-decision contract (see parity spec `hooks.lifecycle`). |
| MCP | `.mcp.json` (project), `~/.claude.json` (user/local), managed | Transports stdio/http(streamable-http)/sse/ws; OAuth; `${VAR:-default}` expansion. |
| Output styles | `.claude/output-styles/*.md` (user/project/managed) | Selected via `outputStyle` setting (standalone `/output-style` removed v2.1.91). [verifier-corrected] |

## 7. Migration reader test matrix (what "compatible" must prove)

A migration is only claimed compatible when HIDE, pointed at a real Claude Code repo, reproduces:
1. The exact resolved instruction set and order (verify against a `/context`-style "what loaded and why" view).
2. The exact permission decision on a battery of commands (allow/ask/deny) including compound commands, symlinks, and protected paths.
3. Subagent/skill discovery and frontmatter semantics (tools allow/deny, model inherit, progressive disclosure).
4. MCP server resolution across scopes with the correct precedence and no merge where whole-entry-wins applies.
5. Hooks firing at the right events with the right decision effect, **and never before trust**.

## 8. Honest boundaries

- HIDE does **not** promise to execute a Claude Code hook/skill script identically - scripts are arbitrary code; HIDE runs them in its own (stronger, OS-sandboxed) execution environment, which may differ in available env/paths. Migration guarantees *config reading and semantics*, not bug-for-bug script behavior.
- Cloud-only Claude Code features (Routines, `--cloud`/`--teleport`, web session store) have no local equivalent to migrate; HIDE's analogue is a local always-on host (see `HIDE_DURABLE_AGENT_SPEC.md`).
- Anything the clean-room verifiers downgraded to ANECDOTAL (e.g. the `#` shortcut) is implemented as a *capability*, not a named-identical feature.
