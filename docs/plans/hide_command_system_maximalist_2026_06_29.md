# HIDE — The Maximalist Command System (trigger grammar + local-unlimited depth)

The thesis, in one line: **a trigger is a detonator.** In a metered cloud tool a slash command is a
macro. In HIDE, where inference is local and there is no token meter, a single `/`, `@`, or `>` can
detonate a fleet, a speculative preview, a generate-N-and-pick, or an always-on ambient pipeline. Other
tools must choose "try one" or "try five" because each variant costs money. HIDE generates N and keeps
the best at the same cost as one. So the design rule is: **copy the winning trigger paradigms, then
expand each one with the multipliers only free local compute allows** and tame the result with a single
grammar plus progressive disclosure so it reads as calm, not gaudy-cluttered.

The audit confirms the bones already exist: a Cmd+P palette (`ui.tsx:CommandPalette`), the Executor
composer (`Chat.tsx:Composer`, no trigger interception yet), `wire.ts:CUSTOM_NAMES` (20 custom intents),
a backend `CommandRouter` (validates + logs `custom{name}` as an event but **never dispatches it** — the
handler is a no-op today), and an unused `ExtensionContribution::Command` seam (`hide-core/plugin.rs`, no
loader). Honest framing (the critique corrected an earlier "mostly wiring" claim): the *surfaces* exist,
but the load-bearing pieces are **genuinely new engineering** — a trigger parser in the composer, a
command-file loader + registry, a `CommandDispatcher` (the missing name->handler step), a typed-arg
overlay, and the speculative-preview sandbox. The MVP is small; the full system is real work, staged.

---

## 1. One grammar, six triggers (the vocabulary)

All triggers live in the **one input the user already types into** (the Executor composer), plus the
Cmd+P palette as the keyboard twin. One grammar, learnable in a minute, each character a different verb:

| Trigger | Verb | Copies | HIDE expansion (free + local) |
|---|---|---|---|
| `/` | run a command | Slack/Discord/Notion/the coding agent | command can declare `fleets: N`, `ambient`, a `pipeline`, an arg schema; user-authored as files |
| `@` | pull into context | Cursor @files/@symbols | `@file` silently background-summarizes into the Context Stack; `@symbol` spawns a 3-agent usage/callers/tests fleet; `@state` loads a saved RWKV state; `@agent` spins a fleet |
| `#` | reference memory | GitHub #issue, Slack #channel | `#tag` pulls structured Project-Brain records (Spine B memory store) by label; `#decision`, `#constraint`, `#file-fact` |
| `>` | do an action | VS Code palette `>` | `>format/test/build/deploy`; `>agent:name` runs a named fleet; ambient oracles already running make the result instant |
| `:` | insert a template | Slack `:emoji:`, snippets | `:snippet:name` parameterized insert; `:agent:role` injects a persona; `:state:name` |
| `!` | run shell | Jupyter `!cmd` | `!cmd` in the session PTY with streamed output; `!~` fuzzy shell-history |
| `Cmd+K` | inline edit | Cursor/VS Code Cmd+K | edit-in-place that forks **3 attempts in parallel** (quick / thorough / creative); pick or blend |

They **compose in one input**: `@auth.ts refactor the token path using #decision:retry-policy, then >test`.
And commands **pipe**: `/format src/**/*.ts | /lint --fix | /test --fast`, where a pipe stage can fan its
input across a Try-N fleet.

## 2. The maximalist multipliers (what every trigger inherits because it is free)

These are the "be expensive because local" levers. Any command/trigger can opt into them; good defaults
turn the safe ones on:

- **Fleet-spawn** — `fleets: N` forks the RWKV state N times (memcpy, no re-prefill) and runs N variant
  prompts; an oracle ladder ranks; you see winner + runner-up. No tool that meters tokens can offer this.
- **Speculative preview = a fleet, not a single run.** As you type arguments, HIDE forks N throwaway
  state-copies (free memcpy), runs the variants in parallel, and the autocomplete hint shows the
  **oracle-ranked top result + runner-up** ("`/format --style ` -> Prettier `1st` · Black `2nd`", each
  with a preview diff). This lifts the Cmd+K 3-attempt pattern (quick/thorough/creative) onto *every*
  command's argument entry. A metered tool can preview zero variants; HIDE previews several before you
  even press Enter. (Debounced on settle, cancelled on next keystroke, concurrency-capped — see §8.)
- **Generate-N-and-pick, with adaptive N.** The default for any generative command: produce several, rank
  by build->test (cheapest-first, short-circuit), surface winner + runner-up. N **scales to the stakes**:
  a one-line tweak gets 3; a "refactor the token path" gets 9; an ambiguous architectural ask gets a
  15-branch tree. Cost is identical to one, so the only ceiling is local concurrency, not budget.
- **Ambient = a multi-tier oracle hierarchy, always on.** Not a vague "small budget" but a concrete
  roster: **Tier 1** (on keystroke, file-local, instant: typecheck, spell) · **Tier 2** (on save,
  build-affected: lint + the affected tests) · **Tier 3** (on idle ~5s, whole-project: search-index
  refresh, security-gate candidates, suggestion ranking). Each emits a `tool_progress`/`ambient_finding`
  with a confidence score; only `> 0.7` pins to a quiet Context-Stack "Ambient" line. A hard CPU ceiling
  (~20% while editing) and a fixed pool keep it calm. The point: the lint/test/security verdict is
  *already there* before you ask, because running it constantly is free.
- **Conditional pipelines + retries** — commands declare steps with on-success/on-failure/max-retries;
  loops are free, so a `/fix` can self-retry against the test oracle until green or it asks you.
- **Batch** — a `file-pattern` argument forks one agent per matched file, in parallel.

## 3. User-authored commands + composition (expand it yourself)

The feature set is **open**, not a fixed menu:

- **Commands as files** — `.hide/commands/**/*.md`: YAML frontmatter (`name`, `description`, `icon`,
  `args:` typed schema with hints/enums/defaults, `fleets:`, `ambient:`, `context:` to auto-pin spans,
  `pipeline:` steps) + a markdown body that is the prompt (with `$ARG` substitution). Filesystem path is
  the namespace. This steals Raycast's script-command model and other coding agents' file-based slash-command
  configs, typed like Discord's options.
- **Pipelines as files** — `.hide/pipelines/*.pipeline`: a prose/DSL chain of commands with typed magic
  variables passed between stages (Apple-Shortcuts-style), optionally fanned across a fleet.
- **Macros** — record a sequence of actions, save as a command. HIDE can even *learn* one: notice a
  repeated manual sequence and offer to bottle it.
- **Prompt/command library** — git-versioned in the repo so a team shares and forks commands like code.

## 4. Discovery and learnability (so maximalism stays inviting)

Maximalism only works if it is discoverable. The discovery layer is itself a place to spend free compute:

- **Fuzzy palette** ranked by recency + frequency + favorites; typo-tolerant; keyboard-first.
- **Contextual surfacing** — a selection or cursor position narrows the palette to the 6-12 actions that
  apply, each with a one-line "why" and its keybinding. Right place, right moment.
- **Inline arg hints + speculative preview** — never "argument soup"; every arg is typed, hinted, and
  previewed live.
- **Did-you-mean + similar-commands** — a near-miss surfaces ranked candidates.
- **Ambient shortcut surfacing** — after you use a command a few times by mouse, its keybinding starts
  showing in terse mono next to it. No modal onboarding; hints appear inline as you act.

## 5. The clutter discipline (gaudy depth, calm surface)

The honest risk the research flagged is the Microsoft-Word-ribbon failure: depth becomes noise. The
countermeasures, all consistent with the concrete/no-meter doctrine:

- **One grammar** (six triggers, one input) instead of scattered menus and toolbars.
- **Progressive disclosure** — the surface shows the few likely actions; depth is one keystroke away, not
  always on screen. Power-user depth lives behind the palette and Cmd+Shift+?.
- **Good defaults over options** — the safe multipliers (generate-N, ambient lint) are on; the loud ones
  (auto-apply, deep fleets) are opt-in per command.
- **Throttle ambient** — a small fixed budget of always-on agents; findings surface only above a
  confidence bar, as a quiet Context-Stack line, never a popup storm.
- **Doctrine** — Geist Mono, concrete, no budget meter, terse telemetry voice; the palette is text, not
  chrome.

## 6. Where it binds (audit hooks) + the real-vs-new ledger

**Real today (wire it):** `ui.tsx:CommandPalette` (registry + fuzzy + keyboard nav); `Chat.tsx:Composer`
(the input to intercept); `wire.ts:CUSTOM_NAMES` (20 names) + the `custom` intent (already the extensible
escape hatch); `commands.rs:CommandRouter` (validates + logs intents); `plugin.rs:ExtensionContribution::Command`
(the seam); `CodeActions.tsx` (selection actions).

**New code (build it):**
1. **Trigger parser** in the composer — intercept `/ @ # > : !` to open an autocomplete overlay (reuse the
   palette component); parse a trailing pipeline (`|`).
2. **Command registry** that merges built-ins with files loaded from `.hide/commands/**` (a loader behind
   `ExtensionContribution::Command`), exposing `{name, description, args schema, multipliers}`.
3. **Annotate `CUSTOM_NAMES`** with `{description, example, readyToRun(state)}` so the slash surface
   auto-generates from the registry and the backend validates names bidirectionally.
4. **Dispatch-by-name** in `CommandRouter` — today custom is logged but unhandled; route `custom{name}` to
   a handler registry (built-in handlers + file-command -> SubmitTurn / ForkSession×N / pipeline runner).
5. **Speculative-preview sandbox** — a throwaway session that runs a command for the autocomplete hint
   (this is where the RWKV state fork + the ambient lane pay off).
6. **Keybinding registry** (replaces the 4 hardcoded chords in `App.tsx`) so commands bind keys and a
   Settings surface can edit them.

**Loader, dispatch, and safety specifics** (the critique flagged these as under-specified):
- **Namespace + conflict rule:** built-ins occupy a base namespace; a file at `.hide/commands/refactor/lint.md`
  maps to `/refactor:lint`; a file command with the same name **overrides** the built-in. Invalid files
  (missing `name`/`description`/bad schema) are logged and skipped, never crash boot. The registry exposes
  `conflicts()` and a startup line: "loaded 23 commands (18 built-in, 5 from .hide/commands, 1 override".
- **`CommandDispatcher`:** the missing step. `CommandRouter.handle(custom{name,payload})` routes to a
  name->handler map: built-in handlers, or a file command -> `SubmitTurn` / `ForkSession×N` (fleet) /
  pipeline runner. Today this is a no-op after logging.
- **Safety / scope:** `.hide/commands` is **repo-local and runs in the same trust domain as the repo's
  code** (a malicious repo command is no worse than a malicious build script, but say so): no network
  fetch of commands; `!` shell and file-writing commands go through the existing security gate; preview
  runs are sandboxed and side-effect-free.
- **`flavor` keeps defaults simple:** each command declares one mutually-exclusive
  `flavor: single | generate-n | fleet | ambient` (default `single`), so authors don't juggle independent
  `fleets:`/`ambient:` flags and the composer can show the flavor as a small badge.

## 7. Sequencing (MVP first, depth in layers)

1. **MVP — `/` in the composer** over the existing palette registry + a handful of built-ins
   (`/explain`, `/test`, `/fork`), arg-free, single-run. Proves the grammar with almost no new backend.
2. **`@` and `>`** — context injection (`@file`/`@symbol`) and actions (`>format/test`), wired to the
   existing connectors + intents.
3. **User-authored command files** — the `.hide/commands` loader + YAML arg schema + `$ARG` substitution +
   dispatch-by-name in `CommandRouter`.
4. **The multipliers** — `fleets:N` (over Fork-&-Try-N), generate-N-pick default, speculative preview,
   then `ambient`.
5. **Composition** — pipelines + macros + the prompt library.
6. **Discovery polish** — recency/frequency ranking, contextual narrowing, ambient shortcut surfacing.

## 8. Honesty / doctrine guardrails + pitfalls (each with a countermeasure)
- No budget meter; ambient findings are quiet Context-Stack lines, not popups. Throttle ambient agents to
  a small fixed pool.
- Speculative preview must be debounced + sandboxed + cancellable, or every keystroke becomes a DoS on the
  local box. Concrete budget: fire on ~250ms settle, cancel on next keystroke, **at most one preview fleet
  in flight per trigger**, 2s timeout, results discarded if stale. The ambient pool obeys the ~20%-CPU
  ceiling separately.
- Too many trigger types is cognitive load — hold the line at six, one input, one grammar; resist a
  seventh.
- Mention/tag resolution failures must be visible (a soft inline "not found"), never silent.
- Composition without type safety silently breaks — pipeline stages pass typed magic variables, validated
  at construction.
- Never market "infinite" anything; the command vocabulary is "always loaded, instantly resumed, never
  billed, never truncated."

## 9. Frontier paradigms (the next depth, once the grammar lands)
The critique pushed for depth beyond the trigger grammar. These exploit local-unlimited compute hardest
and are where HIDE becomes a cut above; they sit on top of the §7 sequencing, not in the MVP:

- **Command output as first-class objects.** A run does not just emit text into the composer; it produces
  a typed, queryable result (a diff set, a file list, a test report, a ranked variant list) that you can
  pipe, pin to the Context Stack, re-open, or feed into the next `/`. Output is data, not transcript.
- **Time-travel over command runs.** The State timeline already scrubs RWKV state; extend it to the
  **command history** — scrub to any past command, see its inputs/output/diffs, and **fork the session
  from there** to try a different command at that point. Re-run a past command against the current code.
- **Voice + multimodal triggers.** Triggers are not text-only: speak a command (local Whisper) and it
  becomes a `/`; while a fleet runs, a spoken phrase is an `Interrupt::Steer`; drop an image/screenshot as
  a trigger argument. Latency-free because local.
- **Agent-authored commands.** The Executor can write a new `.hide/commands/*.md` when it notices a
  repeated multi-step task ("you have done lint->fix->test by hand three times; save `/tidy`?"), proposing
  it for one-click adoption. The feature set grows itself.
- **Cross-file refactor previews.** `/refactor @auth.ts @session.ts @guards.ts` runs the fleet and shows a
  **multi-file diff feed** (each file a card, accept/reject per file or all), previewed before commit.
- **Command-chaining hints.** After `/test` passes, the palette surfaces the common next move
  ("`/estimate-impact`?", "`/commit`?") learned from this repo's usage. The system teaches its own
  pipelines.
