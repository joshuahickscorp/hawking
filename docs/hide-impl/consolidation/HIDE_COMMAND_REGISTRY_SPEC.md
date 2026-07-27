# HIDE Command Registry Spec

Consolidation gate task 2: the ONE unified command registry (schema authority plus codegen),
model-free. This documents the single canonical table from which every surface (button, shortcut,
context menu, palette, chat, IDE, ACP, SDK) resolves a command, so no surface re-declares its own
bindings.

House note: hyphens and parentheses only, no long dashes.

Status: built and green. `hide-protocol` owns the table; `hide-sdk` projects it. No FE control is
touched yet (this task builds the authority only). No model is staged.

---

## Owner decision: hide-protocol plus hide-sdk, not a new crate

The single authoritative owner is `hide-protocol` (already THE schema authority: it defines the
semantic object model and the wire protocol in Rust, with serde AND schemars derived from the same
types) plus `hide-sdk` codegen (already the crate that emits the TypeScript the FE consumes). The
command registry is a schemars table exactly like `Method` and `Notification`, so it belongs where
the other wire shapes already live.

A dedicated `hide-command-registry` crate was rejected: it would add a compile unit and invite
naming-symmetry ("there is a registry, so there is a crate") with nothing behind it. The catalog is
a static table plus deterministic checks; it does not warrant its own crate. Concretely:

- `crates/hide-protocol/src/command.rs` defines `CommandSpec`, its enums, the canonical
  `command_catalog()`, and the integrity constants (`INTENT_NAMES`, `WIRE_CUSTOM_NAMES`,
  `HOST_CAPABILITIES`). Re-exported from `hide_protocol` lib root. (`PENDING_CUSTOM_NAMES` was
  retired by the contract-cleanup stage; see the section of that name below.)
- `crates/hide-sdk/src/command.rs` projects the catalog into two artifacts the FE consumes:
  `goldens/command_catalog.json` (the serialized table) and `goldens/commands.d.ts` (the
  `CommandSpec` type plus the `COMMAND_CATALOG` array). Emitted by the existing codegen bin
  (`cargo run -p hide-sdk --bin hide-sdk-codegen`) and pinned by golden tests, the same pattern as
  the protocol goldens.

Both crates stay model-free (RIP doctrine): a static table and deterministic codegen, no model, no
socket, no runtime bytes.

---

## The CommandSpec schema

One row per user-invocable command. Fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | `String` | Stable unique id (the capability name); the key every surface resolves on. |
| `title` | `String` | Human title for menus and the palette. |
| `description` | `String` | One-line description. |
| `category` | `Category` | The domain (drives the coverage ledger). |
| `primary_surface` | `Surface` | Where the command lives first. |
| `available_surfaces` | `Vec<Surface>` | Every surface that offers it. |
| `required_selection` | `RequiredSelection` | `None`, `Text`, `File`, `Hunk`, `PlanStep`, or `Any`. |
| `required_capabilities` | `Vec<String>` | Negotiated server capabilities it needs. |
| `effects` | `Vec<EffectClass>` | The effect classes it may cause. |
| `approval_policy` | `ApprovalPolicy` | `Auto`, `Ask`, `RequireSandbox`, or `Deny`. |
| `keyboard_shortcut` | `Option<String>` | `Mod` is Cmd on macOS, Ctrl elsewhere. |
| `command_palette` | `bool` | Whether the palette lists it. |
| `context_menu` | `bool` | Whether a context menu offers it. |
| `toolbar_binding` | `Option<String>` | A toolbar button id it binds to. |
| `backend_binding` | `BackendBinding` | How a surface reaches the backend. |
| `undo_strategy` | `UndoStrategy` | `None`, `Inverse`, `Checkpoint`, or `Reject`. |
| `receipt_kind` | `Option<String>` | The receipt it seals, if any. |
| `telemetry` | `Option<String>` | A telemetry event name, if instrumented. |

Enums: `Surface` (Chat, Ide, Home, ContextStack, StatusBar, StateTimeline, Terminal, DiffReview,
Settings, Fleet, Palette, Editor), `Category`, `RequiredSelection`, `ApprovalPolicy`,
`UndoStrategy`, and `BackendBinding` (`Intent(String) | Custom(String) | Rpc(String) | LocalOnly`).
All derive `Serialize`, `Deserialize`, and `schemars::JsonSchema`, so the JSON Schema and the TS
type are generated from the same definitions and can never silently drift.

### EffectClass mirrors the protocol effect enum

`EffectClass` is a type alias of `hide_protocol::plan::Effect` (ReadFs, WriteFs, Network, Process,
Shell, Vcs, Environment, Approval, AgentSpawn, State, Other), the schema authority's own effect
enum, which already derives `JsonSchema`. Reusing it keeps the command table self-contained and
avoids a second effect taxonomy in the same crate.

Note the OTHER `Effect` in the tree: `hide-extension-registry` owns a SECURITY least-privilege
ranking (Read, Write, GitMutation, Execute, Process, Network, SecretAccess, ExternalMutation,
Irreversible, Privileged). That is a different taxonomy (for capability resolution) and does not
derive `JsonSchema`; reusing it would add a cross-crate dependency and a schema mismatch. The
command registry mirrors the protocol effect classes instead. If a later pass wants one effect
vocabulary across the tree, map protocol `Effect` to the security `Effect` at the policy boundary,
not here.

---

## How every surface resolves a command from the ONE catalog

`command_catalog()` is the single source. Each surface resolves against it by a different key, but
always the same rows:

- Command palette: lists every row where `command_palette == true`, keyed by `id` and `title`.
- Keyboard shortcut: matches `keyboard_shortcut`. The parity invariant guarantees a shortcut OR a
  palette entry, so no command is unreachable.
- Toolbar button: a button with id `X` invokes the row whose `toolbar_binding == Some("X")` (for
  example `composer.send`, `chat.new`, `steer.redirect`).
- Context menu: offers rows where `context_menu == true`, filtered by `required_selection` against
  the live selection (`Hunk` for diff hunks, `File` in the explorer, `Text` for an editor span).
- Chat / IDE / any panel surface: renders the rows whose `available_surfaces` contains that
  `Surface`; `primary_surface` decides the default home.
- ACP peer and SDK: read the same table. The SDK exposes it as `commands.d.ts` (`COMMAND_CATALOG`
  plus the `CommandSpec` type); an ACP peer reads `command_catalog.json`. Neither hand-declares
  commands.

Invocation is uniform: a surface reads `backend_binding` and dispatches:

- `Intent(name)`: post the typed `hide-core` `Intent` to `/v1/hide/intent` (the path the shipped FE
  already speaks). `name` is a real `api.rs` variant.
- `Custom(name)`: post `Intent::Custom{ name, payload }` to the same route. `name` is live in
  `wire.ts` `CUSTOM_NAMES`, which now means exactly "the host has an arm for it". There is no
  pending tier.
- `Rpc(name)`: an elevated capability, either a real `Method` string (for example `turn/steer`,
  `item/list`, `goal/get`) or a census-confirmed host capability (for example
  `run_static_analysis`, `memory_add`). Per the census, the lazy FE path for these is a custom
  intent once wired, not a second `/rpc` client.
- `LocalOnly`: a pure FE action, no backend call (none in the seed catalog today; reserved).

### The catalog (52 commands)

Grouped by `primary_surface`:

- Chat: `submit_turn`, `cancel_run`, `pause_run`, `resume_run`, `create_side_chat`,
  `merge_side_chat`, `steer`
- DiffReview: `accept_diff`, `reject_diff`
- StateTimeline: `scrub_to_event`, `fork_session`, `checkpoint_create`, `checkpoint_restore`,
  `checkpoint_rewind`, `checkpoint_replay`, `checkpoint_fork`, `checkpoint_compare`,
  `checkpoint_inspect`
- ContextStack: `memory_add`, `memory_supersede`, `memory_record_outcome`, `memory_revalidate`,
  `approve_plan`, `edit_plan_step`, `reorder_plan`, `skip_step`, `repair_step`
- Home: `goal_set`, `goal_get`, `goal_clear`, `goal_evaluate`, `workspace_set_repo_trust`,
  `environment_switch`
- StatusBar: `run_static_analysis`, `promote_run`, `resume_run_foreground`
- Ide: `open_file`
- Terminal: `run_command`, `pty_input`, `pty_resize`
- Palette: `run_search`

Added by the contract-reconciliation pass (host-handled names that had NO CommandSpec, so surfaces
dispatched them raw with no palette row and no shortcut parity):

- `new_session` (Home / Chat): the launcher and New-chat "New thread" gesture, `handle_intent`
  launcher arm.
- `revert_diff`, `edit_hunk` (DiffReview): the diff-review stage already binds `revert_diff`;
  both route through the `diff_action` arm of `handle_intent` (`revert_diff` / `apply_hunk`).

The first 27 rows were seeded from the ranked census priority in
`HIDE_BACKEND_WITHOUT_SURFACE_REPORT.md` so each maps a REAL capability to a REAL existing control
(verify to StatusBar Problems, side chat to the dead New-chat buttons, checkpoints to StateTimeline,
memory to the ContextStack Memory stratum, goals to the HomeComposer goal field, steer to SteerBar,
workspace trust to the Add-folder flow), plus environment switch, transcript search, and the ten
already-working core intents.

The 15 rows added this session catch the registry up to the surface the backend grew, all bound to
custom names the host already handles (`crates/hide-backend/src/host.rs`):

- Checkpoint family (StateTimeline): `checkpoint_rewind`, `checkpoint_replay`, `checkpoint_fork`,
  `checkpoint_compare`, `checkpoint_inspect` (`handle_goal_checkpoint_intent`).
- Plan domain (ContextStack / Chat): `approve_plan`, `edit_plan_step`, `reorder_plan`, `skip_step`,
  `repair_step` (`handle_plan_intent`); a new `Plan` category.
- Background jobs (StatusBar / Home): `promote_run`, `resume_run_foreground`
  (`handle_background_intent`); a new `Background` category. Pause / stop / fork of a promoted run
  reuse `pause_run` / `cancel_run` / `fork_session`, which already route by run id, so no new command.
- Terminal input (Terminal): `pty_input`, `pty_resize` (`handle_pty_intent`, live wire names). The
  process attach / stop / capture host methods (`attach_process` host.rs:1991, `stop_process` 2001,
  `capture_process_artifact` 2006) have NO wire trigger (no Intent or Custom arm dispatches to them;
  only the Trace D test drives them directly), so NO command is minted for them, per "note it rather
  than invent a name".
- Search (Palette): `run_search`, the LIVE transcript search over `/intent` (the FE dials `/intent`,
  never `/rpc`). The `item/list`-bound `search_transcript` that once sat beside it was COLLAPSED into
  this row by the contract-cleanup stage: one capability, one command.

Two new `Category` members carry the new domains: `Plan` and `Background`. The coverage test still
asserts the seven priority domains; the extra categories keep the ledger honest.

Binding split after the contract-cleanup stage: 10 `Intent`, 2 `Rpc`, 40 `Custom`.

### Contract reconciliation: Rpc bindings that were really custom-handled

Eight commands were bound `Rpc`, so `runCommand` in `app/src/store.ts` refused them (the app posts
`/v1/hide/intent` only and deliberately has no `/rpc` client), yet `host.rs` `handle_intent` already
dispatches an equivalent `Intent::Custom` name for every one of them. They are re-bound to `Custom`:

| command | was | now | host.rs arm |
| --- | --- | --- | --- |
| `steer` | `Rpc("turn/steer")` | `Custom("redirect_run")` | `steer_action` (`"redirect_run" \| "steer"`), raises a real `InterruptHub::Steer` |
| `memory_add` | `Rpc("memory_add")` | `Custom("memory_add")` | `memory_workspace_env_action` |
| `memory_supersede` | `Rpc("memory_supersede")` | `Custom("memory_supersede")` | same |
| `memory_record_outcome` | `Rpc("memory_record_outcome")` | `Custom("memory_record_outcome")` | same |
| `memory_revalidate` | `Rpc("memory_revalidate")` | `Custom("memory_revalidate")` | same |
| `goal_evaluate` | `Rpc("goal_evaluate")` | `Custom("goal_evaluate")` | same |
| `workspace_set_repo_trust` | `Rpc("workspace_set_repo_trust")` | `Custom("workspace_set_repo_trust")` | same |
| `environment_switch` | `Rpc("environment_switch")` | `Custom("environment_switch")` | same |

`steer` also drops its stale `toolbar_binding` `steer.redirect` (the retired SteerBar) for
`composer.steer`: the composer owns the gesture now, and the binding is what stops the shell from
re-binding `Mod+/` on top of the control that already owns it.

Still `Rpc`, honestly unreachable from the app, because there is NO custom arm:

- `run_static_analysis`: `host.rs` exposes it as a method only; no `Intent::Custom` name reaches it.
- `goal_get` (`goal/get`): RETIRED as a command (remediation). A real `Method` string with no custom
  arm, so no surface could dispatch it, while its spec still declared `command_palette: true`. The
  Method stays; the catalog row is gone, and the catalog now carries NO `Rpc` binding at all.
- `search_transcript` (`item/list`): RETIRED as a separate command. Keeping it beside `run_search`
  left two catalog ids for one capability, and its `Mod+Shift+F` could never be registered (the
  shortcut derivation drops `Rpc` bindings, and a bare chord carries no query). The host answers
  `run_search`, `search` and `search_transcript` on the same arm, so `run_search` is the one row.

---

## The parity invariant

Every command is reachable: it has a `keyboard_shortcut` OR `command_palette == true`. No orphan
actions. Asserted over the whole catalog by
`command::tests::every_command_has_a_shortcut_or_lives_in_the_palette`. Both branches are exercised:
`accept_diff` and `reject_diff` are shortcut-only (palette off, context menu on), so the OR is not a
tautology, and a future command added with neither reachability path fails the build.

---

## Backend-binding integrity (nothing is silently invented)

`command::tests::backend_bindings_resolve_to_real_targets` asserts, over the whole catalog:

- every `Intent(name)` is in `INTENT_NAMES` (a mirror of the snake_case `#[serde(tag = "type")]`
  variants in `crates/hide-core/src/api.rs`);
- every `Custom(name)` is in `WIRE_CUSTOM_NAMES` (a mirror of `app/src/wire.ts` `CUSTOM_NAMES`);
- every `Rpc(name)` is a real `Method::as_str()` OR in `HOST_CAPABILITIES` (census-confirmed host
  methods with `host.rs` refs).

Both directions are guarded now. `wire_custom_names_mirror_wire_ts` reads `app/src/wire.ts` itself
and compares the parsed name list in order, so the mirror cannot silently go stale (it once sat 17
names behind while a same-file comparison kept passing). `every_live_custom_name_has_a_command`
asserts the other direction: a live wire name with no `CommandSpec` is a capability the palette, the
shortcut map and the SDK cannot see, which is how eight host-handled names ended up dispatched raw.

---

## PENDING_CUSTOM_NAMES (RETIRED by the contract-cleanup stage)

This section used to hold a whitelist of host-handled names that `wire.ts` did not yet carry, so a
`Custom` binding could name one before the FE could send it. It is gone, and so is the const.

Two reasons. First, all 17 pending names were added to `wire.ts` `CUSTOM_NAMES` in one pass, so the
list was empty in substance while the Rust mirror still declared them, which is what made the
disjointness guard vacuous. Second, a pending tier is a way for the contract to carry a name that
cannot work; the contract now carries exactly the names `host.rs handle_intent` acts on, asserted in
both directions (see the integrity section above), and an unhandled name gets an honest negative ack
rather than `accepted: true`.

The same pass RETIRED 17 names in the other direction, orphans with no host arm anywhere in
`crates/`: `save_file`, `inline_edit`, `mention_in_chat`, `quick_fix`, `queue_turn`, `rerun_step`,
`fleet_run`, `resolve_conflict`, `pin_span`, `unpin_span`, `switch_profile`, `switch_model`,
`toggle_confidence`, `focus_run`, `dismiss`, `create_pr`, `switch_branch`.

CORRECTION (remediation): `save_file` is no longer on that list. It came back as a live command with
a host arm (`host.rs` `save_file`) and a release arm, precisely so the editor save stops being a raw
connector write with no gate. The other 16 remain retired.

### Added by the contract-cleanup stage (the host-handled names with no spec)

| Command | Custom name | Surface it was already dispatched from |
| --- | --- | --- |
| Create worktree | `create_worktree` | HomeComposer worktree chip |
| Open session | `open_session` | Home recents list |
| Approve held command | `approve_gate` | store `approveGate` (the gate overlay) |
| Deny held command | `deny_gate` | store `denyGate` (the gate overlay) |
| Approve effectful step | `approve_effect` | the effect-approval gate |
| Deny effectful step | `deny_effect` | the effect-approval gate |

CORRECTION (remediation): this table listed two more, `compact_context` and `open_folder`. Both are
RETIRED: neither has a catalog row, a wire name or a host arm any more, because each had an empty
host arm and no reader for the record it wrote.

The five that address a live object (`gate`, `run_id`, `session_id`) carry that id in
`REQUIRED_ARGS` (`app/src/store.ts`), which is also what keeps them out of the palette: a bare
palette gesture cannot name the gate it is approving. `create_worktree` needs no argument and is an
ordinary palette row.

### Collapsed

`search_transcript` (bound `Rpc(item/list)`, carrying a `Mod+Shift+F` that nothing could register)
folded into `run_search`. The host answers `run_search`, `search` and `search_transcript` on the same
arm, so two ids were one capability counted twice.

### HOST_CAPABILITIES (Rpc targets that are not yet a Method)

Census-confirmed host methods with no `Method` string yet, referenced by `Rpc` bindings:
`run_static_analysis` (host.rs:1373), `memory_add` (1870), `memory_supersede` (1906),
`memory_record_outcome` (1931), `memory_revalidate` (1957), `memory_list` (1885), `goal_evaluate`
(1572), `workspace_set_repo_trust` (1136), `environment_switch` (1193). When a later increment gives
one of these a real `Method`, move the binding to that `Method` string and drop the host-cap entry.

---

## Tests and verification

hide-protocol (`command::tests`, 8, all green):

- `catalog_is_non_empty_with_unique_ids`
- `every_command_has_a_shortcut_or_lives_in_the_palette` (the parity invariant)
- `backend_bindings_resolve_to_real_targets` (Intent / Custom / Rpc integrity)
- `wire_custom_names_mirror_wire_ts` (reads `app/src/wire.ts` and compares in order)
- `every_live_custom_name_has_a_command`
- `catalog_covers_the_seven_priority_domains` (verify, side_chat, checkpoint, memory, goal, steer,
  workspace)
- `specs_round_trip_through_serde_json`
- `catalog_data_carries_no_en_or_em_dashes`

hide-sdk (`tests/golden.rs`, 2 new plus 2 extended, all green):

- `command_catalog_json_golden_is_stable`, `command_typescript_golden_is_stable` (byte-compare, so a
  catalog change fails the build until `cargo run -p hide-sdk --bin hide-sdk-codegen` refreshes the
  goldens);
- `generation_is_deterministic_across_runs` and `generated_artifacts_carry_no_en_or_em_dashes`
  extended to cover the two new artifacts.

Regenerate the goldens after any intended catalog change:

```
cargo run -p hide-sdk --bin hide-sdk-codegen
```
