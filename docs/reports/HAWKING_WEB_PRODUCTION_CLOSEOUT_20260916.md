# Hawking Web production closeout — 2026-09-16

## Scope

This is the final bounded H-Web closeout plus the requested markdown
upload/download path. Capture/OCR, Layer II, Ghidra/Rizin, Flash, Pulsar,
ModelLake, and native model admission were not started or changed.

## Current live owner

`h web` and `hawking web` use one launcher implementation,
`hawking.web_launcher`. The public path no longer starts resident Kimi P0,
invokes `hcli serve`, uses MLX, or creates a second daemon/scheduler. It checks
the canonical Keychain-first OpenRouter resolver, validates the current
Hawking API/Web contract, reuses a healthy compatible `hawkingd`, and performs
only the existing identity-checked recovery when required.

The latest exact workflow was:

```text
cd /Users/scammermike/Downloads/hawking
unset OPENROUTER_API_KEY
h web
```

It returned exit 0 through the current launcher and recovered the compatible
`hawkingd` owner as PID `60107` after the bounded source reload. Subsequent
healthy launches reuse that owner. `/health` was 200,
`/hawking/web/markdown` was present, the active accepted Web release was
`web-6347550e08ae`, and no candidate remained. The output reported the
Keychain credential as ready without printing the credential.

The current main-chat seam is Hawking-native where evidence is needed:
`/hawking/web/chat` attaches the bound workspace/session identity and enters the
server-owned read-only worker loop for source/Goal/runtime/artifact intent. Its
H-Web-specific fence exposes only bounded source, filesystem, Git, runtime,
permission, Goal, artifact, and memory reads; mutation, shell, browser action,
and macOS action doors remain unavailable. Tool calls dispatch through the
existing registry or canonical H-NOTES/local doors, and the result carries the
real Hawking receipt/tool trace without exposing provider credentials. Ordinary
conversation keeps the provider's true incremental SSE stream; tool-loop turns
are emitted as a bounded accepted result chunk after Hawking settles the
observation loop.

## Markdown capability

The current page admits only bounded `.md`/`.markdown` bytes through Hawking's
content-addressed artifact owner. The browser fixture was opened through the
real file chooser, removed, re-attached, sent to a real DeepSeek worker, and
survived page reload. The worker's evidence included Hawking `fs.read` of the
admitted artifact rather than a browser-only text copy.

Plan/roadmap/blueprint/specification requests now keep the full answer in the
conversation and additionally create a durable `hawking-plan.md` result. The
page shows a small `↓ .md` link. The live download route returned HTTP 200 with
`Content-Disposition: attachment; filename*=UTF-8''hawking-plan.md`; the
artifact and link remained after session reload. Ordinary question answers are
not truncated or forced into markdown files.

The final accepted asset renders user prompts in a right-aligned column and
assistant output in a left-aligned column. Long submitted prompts expose
`Show more`/`Show less`; a dark centered scroll-to-latest button appears while
the reader is away from the newest turn; real generation shows a loading
indicator and hides TPS when complete; visible turns render bounded Markdown;
assistant output copies on double-click or a two-finger touch gesture. These
behaviors were checked on the live page.

## Live H-Web evidence

- Current page: `http://hawking.localhost:8014/`.
- Cloud catalog: DeepSeek V4.1 Flash, Kimi K3, and Qwen3.8 Flash, returned by
  Hawking's catalog/search path.
- Auto was visible as the cloud route and selected the current DeepSeek worker
  plan in the live control surface.
- Local remained honest/withheld: `KIMI_P0_OPERATIONAL` was not selectable.
- Real tiny cloud calls for DeepSeek, Kimi, Qwen, and Auto are retained in the
  existing H-Web session evidence; no new benchmark campaign was run.
- Disposable attached Goal `GOAL-7FD0AAE541` completed as a canonical Hawking
  Goal with worker `hawking-worker-66aaa7cc9c0246e8`, model
  `deepseek/deepseek-v4.1-flash`, and recorded worker cost `$0.0083810292`.
  Its worker evidence covered `tools.catalog`, `tool.reachable`, `fs.search`,
  `fs.read`, `repo.edit`, `tests.run`, `git.status`, and `git.diff`, with a
  structured completion/result receipt. The Goal bubble and Goal modal were
  visible in H-Web, and the Goal/session identity survived reload.
- The main operator query was run after the controlled daemon recovery; the
  live response used the canonical projection and named
  `GOAL-7FD0AAE541` with its completed result. The cloud streaming path now
  receives a bounded, turn-local Goal projection only for matching operator
  questions; Goal state remains owned by Hawking files.
- A current H-Web read-only source-tool smoke completed through Hawking with a
  real DeepSeek receipt and 17 bounded registry/local observations; no source
  mutation was admitted. A separate real streamed DeepSeek turn returned SSE
  frames, usage, and Hawking metadata. These disposable smoke sessions were
  removed afterward; existing user conversation/Goal state was preserved.

## Final acceptance matrix

| Check | Result | Evidence / boundary |
|---|---|---|
| H_WEB_COMMAND | PASS | exact Keychain-only `h web`, exit 0 |
| LEGACY_HCLI_REMOVED_FROM_PUBLIC_WEB_PATH | PASS | one canonical Hawking launcher |
| LEGACY_KIMI_RESIDENT_NOT_STARTED | PASS | no resident startup in source or live launch |
| DAEMON_REUSE | PASS | healthy `hawkingd` owner reused; current PID `60107` |
| SAFE_RECOVERY | PASS | controlled recovery exercised when the route contract changed |
| KEYCHAIN_ONLY_CREDENTIAL | PASS | environment unset; resolver reported presence only |
| SECRET_NONEXPOSURE | PASS | no key/header in argv, logs, API projection, or report |
| CURRENT_WEB_RELEASE | PASS | `web-6347550e08ae`, candidate absent |
| NORMAL_CHAT | PASS | actual H-Web cloud responses and persisted turns |
| STREAMING | PASS | ordinary H-Web turn delivered 7 real SSE frames; tool-loop result is bounded after observation |
| LIVE_TPS | PASS | shown only during actual generation; no hard-coded model speed |
| CLOUD_AUTO | PASS | live cloud Auto route; DeepSeek selected in current plan |
| DEEPSEEK | AVAILABLE | exact catalog/search route and live call |
| KIMI | AVAILABLE | exact catalog/search route and live call |
| QWEN | AVAILABLE | exact catalog/search route and live call |
| LOCAL_RUNTIME | WITHHELD | no admitted executable local artifact |
| MODEL_SEARCH | PASS | Hawking-owned `/v1/models` search |
| MARKDOWN_UPLOAD | PASS | real file chooser and Hawking artifact admission |
| MARKDOWN_REMOVE_REATTACH | PASS | browser token/remove/re-add path |
| MARKDOWN_CHAT_READ | PASS | real worker received durable attachment reference |
| MARKDOWN_DOWNLOAD | PASS | HTTP 200 plus attachment header and bytes |
| MARKDOWN_REFRESH_PERSISTENCE | PASS | plan link and attachment remained after reload |
| CHAT_LAYOUT | PASS | prompts right; assistant responses left on the live page |
| LONG_TURN_COLLAPSE | PASS | long submitted `/status` fixture showed `Show more` |
| SCROLL_TO_LATEST | PASS | dark centered control appeared away from latest |
| GENERATION_LOADER | PASS | live H-Web request showed `loading` before completion |
| MARKDOWN_RENDERING | PASS | live heading/list output rendered as structured content |
| OUTPUT_COPY_GESTURE | PASS | live assistant double-click populated browser clipboard |
| GOAL_CREATION_FROM_WEB | PASS | `GOAL-7FD0AAE541` |
| WORKER_VISIBLE | PASS | live Goal bubble/modal shows worker model and state |
| TOOL_ACTIVITY_VISIBLE | PASS | canonical worker activity/evidence retained for H-Web Goal |
| GOAL_REFRESH_RECOVERY | PASS | same session, Goal, result, and export after reload |
| STATUS | PASS | H-Web status path uses canonical `/health`; live health 200 |
| STEER | NOT NEEDED | the bounded Goal was already terminal when final control checks ran |
| SELF_BUILD_GOAL | PASS | accepted Hawking worker receipts, including prior `GOAL-945600ECD0` |
| REPO_EDIT | PASS | Hawking-owned `repo.edit` evidence |
| TESTS_RUN | PASS | deterministic worker/test evidence |
| GIT_STATUS | PASS | grounded worker receipt |
| GIT_DIFF | PASS | grounded worker receipt |
| GROUNDED_COMPLETION | PASS | structured result and terminal Goal receipt |
| MAIN_CHAT_GOAL_SYNTHESIS | PASS | live response named `GOAL-7FD0AAE541` from canonical projection |
| MAIN_CHAT_HAWKING_TOOL_LOOP | PASS | H-Web source-tool smoke dispatched through Hawking registry/local doors |
| CLOSE_AND_REOPEN_H_WEB | PASS | repeated launcher path reuses current daemon/release |
| DUPLICATE_DAEMON | NO | one live `hawkingd` authority |
| DUPLICATE_GOAL | NO | no duplicate created by refresh/export |
| FLASH_PULSAR_MODELLAKE_DISTURBED | NO | protected lanes untouched |

The inspected H-Web worker-event stream records `$0.1950952224` across its
retained worker-event subtotal; the final bounded attachment Goal accounts for
`$0.0083810292`. No new model benchmark or unnecessary remote campaign was
started for the markdown feature.

## Verification

- Broad selected H-Web/Arsenal suite: **107 passed**.
- Focused chat/attachment/launcher recheck: **141 passed**; the post-projection
  serve/launcher/attachment recheck was **181 passed**.
- `cargo test -p hawking-web --offline`: **2 passed**.
- Python compile of all changed H-Web/attachment/launcher modules: passed.
- H-Web JavaScript syntax check: passed after the final layout/interaction pass.
- Latest focused native-main-chat/artifact/serve verification: **169 passed**.
- Controlled daemon reload: passed; one `hawkingd` owner remained on port 8014.
- `git diff --check`: passed.

## Durable production boundary

The canonical parent Goal remains `GOAL-B33C69F193`, `RUNNING`, at
`E_MACOS_READ_ONLY_FOUNDATION_ACCEPTED`, with the accepted semantic-action
checkpoint preserved. Its exact next action remains a fresh bounded
`E_MACOS_CAPTURE_OCR` WorkUnit only after a red-before-green
`document.extract` contract is available; capture/OCR remains deferred.

`HAWKING_WEB_PRODUCTION_READY=YES` for the defined H-Web operational loop:
terminal launch, real cloud chat, durable markdown input/output, Hawking-owned
Goals/workers/tools/evidence, and session persistence. Tool-loop turns are
intentionally settled before their final bounded SSE result; ordinary chat
continues to expose provider chunk cadence.

## Capability-projection repair addendum — 2026-09-16

The prior “main-chat Hawking-tool smoke” was too narrow: it exercised an
explicit, non-browser research turn and a separate ordinary streaming turn. It
did not prove that the production H-Web streaming executor attached schemas.
The operator correctly found that the normal browser route used
`provider.generate_stream()` directly, with `tool_count=0`, while Goal workers
used the schema-bearing WorkUnit loop.

That split is now removed. Every workspace-bound H-Web main-chat turn receives
the bounded Hawking READ projection; no mutation schema is present. The
stream-aware continuation loop preserves provider SSE deltas, pauses only for
Hawking-validated tool execution, and resumes the provider with the resulting
observations. Health now carries
`hweb-main-chat-read-tools-v2`; `h web` treats an older daemon as incompatible
and uses the existing controlled recovery rather than silently pairing new Web
assets with an old executor.

Fresh evidence:

- Keychain-only `h web --no-browser` safely recovered stale PID `75113` to
  PID `2818`, then later reused PID `2818` without a duplicate daemon.
- Live pinned DeepSeek source discovery: provider boundary exposed **21** READ
  schemas; Hawking executed `source.owner`, `fs.search`, `fs.read`, and related
  bounded reads; effect ceiling was `READ`; `repo.edit` was absent; the final
  answer arrived across **208** actual SSE frames.
- The actual H-Web browser Auto turn visibly streamed source-tool activity and
  grounded `hawking/web_launcher.py::main`; browser refresh retained the same
  completed transcript.
- The existing 21,484-byte pasted Markdown artifact was reattached to a fresh
  bounded turn. Its provider continuation carried 21 schemas and tool-result
  turns, then returned `hawking/web_launcher.py`; original artifact admission
  and body-read behavior remain intact.

Focused deterministic regression coverage: **14 passed, 1 deselected** across
the H-Web native main-chat and OpenRouter gateway suites. The deselected legacy
catalog assertion is pre-existing/environment-coupled (`KIMI_P0_OPERATIONAL`
was returned by the current canonical catalog instead of the test fixture's
legacy fixed catalog); it is unrelated to this repair. Python compilation and
`git diff --check` passed.

## Public `h` bootstrap cutover addendum — 2026-09-16

The installed public `h` router still contained a second stale boundary after
the earlier `h web` repair: it special-cased only `web` and forwarded `build`
plus other control verbs into the historical HCLI staging router. The active
public router now resolves the current Hawking checkout for every control verb;
only `infer` remains a deliberately separate native inference executable.

`h web` and `h build` now share `hawking.web_launcher`. The former clears any
old builder browser binding and opens the normal READ surface. The latter
requires `hweb-builder-session-v1`, creates a short-lived local handoff, and
binds the resulting BUILD session with an HttpOnly, SameSite local cookie. The
durable Session records only a hash of the one-time handoff and browser proof.
An arbitrary session id, browser field, prompt, share route, or copied URL does
not grant builder authority. A builder turn remains READ by default; a bounded
source-mutation request is the only interactive route that receives the
existing typed `repo.edit` / `tests.run` doors, still through the canonical
engine and mutation lock.

Fresh bootstrap evidence:

- `unset OPENROUTER_API_KEY; h web --no-browser` detected the new daemon
  contract and used controlled recovery once, producing healthy `hawkingd`
  PID `78485` on accepted release `web-6347550e08ae`.
- `unset OPENROUTER_API_KEY; h build --no-browser` reused that PID with the
  canonical Keychain resolver and reported server-owned BUILD authority; no
  resident Kimi startup or HCLI invocation occurred.
- A local handoff redemption proved a durable `authority_profile=build` and a
  consumed single-use handoff without printing a token or credential.
- Focused launcher/H-Web regression suite: **15 passed**. Python compilation
  and `git diff --check` passed. The broad legacy catalog fixture remains
  separately classified as above.

No remote mutation was launched for this bootstrap repair. The live remote
builder-tool execution remains an operator-authorized next acceptance, not a
claim made by the launcher cutover itself.

## OpenRouter fast-route roster addendum — 2026-09-16

The default cloud and Auto roster is now exactly the two economical Hawking
workers: `deepseek/deepseek-v4.1-flash` and `qwen/qwen3.8-flash`. Kimi K3 is
not an enabled default, but remains discoverable through Hawking model search
for explicit manual admission. Existing Goals and receipts were not rewritten.

Hawking's gateway policy now keeps the requested model fixed while preferring
the live OpenRouter endpoint tags `together` for DeepSeek and `alibaba` for
Qwen; provider fallback is permitted only within that same requested model.
Provider receipts retain requested/resolved model, preferred/resolved provider,
provider request identity, usage/cost, and timing.

Fresh evidence, with the OpenRouter credential resolved from Keychain only:

- H-Web catalog enabled exactly the two default models; Kimi was returned only
  by an explicit Hawking search.
- A tool-capable DeepSeek turn resolved through Together; a tool-capable Qwen
  turn resolved through Alibaba.
- A real streamed Auto turn selected DeepSeek and resolved through Together.
- The no-cost Auto-plan endpoint likewise selected DeepSeek for a read-only
  repository task.
- Focused Auto, orchestration, gateway, launcher, and native-main-chat tests:
  **34 passed, 1 deselected**. The deselected legacy catalog assertion remains
  environment-coupled and unrelated. Python compilation and `git diff --check`
  passed.

## `h build` builder-session repair — 2026-09-16

The original terminal error was not a Hawking-native diagnosis: the launcher
collapsed every `urllib.error.HTTPError` into the bare word `HTTPError`, losing
the status and server reason needed to distinguish route, schema, workspace, or
authority failures. The historical response itself was not retained, so its
exact old status cannot be honestly reconstructed. The current live endpoint
was then exercised directly and returned `200` with
`hawking.web.builder_handoff.v1`.

The production seam is:

```text
h build -> hawking.web_launcher.launch(profile="build")
        -> POST /hawking/web/build/session {workspace}
        -> hawking.serve.Handler.do_POST
        -> _mint_web_builder_session
```

The launcher now renders a bounded status/reason on a future HTTP rejection
without printing a handoff, request body, credential, or response dump. The
builder contract revision is `hweb-builder-session-v2-revocable`, forcing
controlled recovery rather than reuse of a daemon that cannot enforce the
current boundary.

The session remains local-only, workspace-bound, short-lived (five minutes),
one-time, and server-owned: only a token hash persists; redemption establishes
an HttpOnly SameSite local cookie proof. The explicit `h web` READ transition
now both expires that cookie and records server-side revocation, preventing an
old copied cookie from retaining BUILD authority after an operator downgrade.

Fresh acceptance:

- `unset OPENROUTER_API_KEY; h build` reused healthy `hawkingd` PID `9822` and
  opened a builder handoff with the canonical Keychain resolver. No HCLI or
  resident Kimi path was invoked.
- The same cookie-bound H-Web session performed actual `fs.search`,
  `source.symbol`, and `fs.read` calls for `context.recall`; `repo.edit` was
  absent from that READ turn.
- A deliberately inert Markdown append did reach `repo.edit`, ran the focused
  test and `git.status`/`git.diff`, then was atomically rolled back by the
  existing `NOT_RED_BEFORE` gate. This is correct fail-closed behavior, not an
  accepted self-build mutation: an already-green Markdown fixture cannot prove
  a source change. No production or fixture bytes were left behind.
- A live local handoff followed by `/hawking/web/read` cleared the cookie,
  changed the session profile to `read`, and persisted `revoked_at` on the
  handoff. A following `h build` minted a fresh handoff.
- Focused launcher and H-Web control tests: **17 passed**. Python compilation
  and `git diff --check` passed.

### Workspace-identity follow-up

The operator supplied the missing live discriminator: entering the same
case-insensitive macOS directory as `~/downloads/hawking` caused the public
shim to forward that textual spelling, while the daemon had recorded
`~/Downloads/hawking`. The builder mint path compared the two resolved path
strings and returned `HTTP 400: builder workspace does not match this Hawking
daemon`, despite both names identifying the same filesystem object.

`_mint_web_builder_session` now compares filesystem identity with `samefile`,
failing closed for missing or genuinely distinct workspaces. The builder
contract revision is now `hweb-builder-session-v3-workspace-identity`, so a
pre-fix daemon cannot be silently reused. Fresh acceptance from the exact
lowercase shell path recovered `hawkingd` as PID `17746`, then successfully
minted the builder handoff. A following lowercase-path `h web` reused that
daemon and opened the normal READ surface. Focused launcher/H-Web control
coverage is now **18 passed**; Python compilation and `git diff --check`
passed.
