# Hawking Arsenal Archaeology + Harness Collapse Probe

**Date:** 2026-09-15
**Scope:** current dirty Hawking main worktree, live `hawkingd`, H-Web baseline, current receipts/tests, and reachable repository history.
**Authority:** current source and measured runtime state outrank historical prompts, roadmaps, and legacy names.

This is the completed probe requested by the attached Arsenal document. It is a
baseline and a smallest-next-delta decision record, not a claim that the full
Arsenal has already been implemented.

## VERIFIED BASELINE

- Repository: `/Users/scammermike/Downloads/hawking`
- Git: `main`, `HEAD 8f10e8bb8`, with the user’s large in-progress migration and
  many linked worktrees preserved. No reset, checkout, cleanup, or commit was
  performed.
- Live daemon: `hawkingd`, PID `96603`, bound to `127.0.0.1:8014` with the
  `remote_goal_write` authority. H-Web is available at
  `http://hawking.localhost:8014/`.
- Health: HTTP 200 on `/health`, `/api/runtime/roles`, `/v1/models`,
  `/v1/goals`, `/v1/workers`, and `/hawking/web/models`.
- Native role state is honest and fail-closed: Pulsar, Magnetar, and Themis are
  withheld; no native provider child is running. The current Gravity helper is
  still running and was not disturbed.
- Secure credential gate: `OPENROUTER_KEY_PRESENT=PASS`. No credential or
  authorization header was printed, copied, or written to a file.
- Existing H-Web cloud connectivity was already proven for DeepSeek, Kimi, Qwen,
  and Auto. This probe did not repeat those remote tests.
- Current H-Web cloud roster remains the configured three-model pool:
  DeepSeek V4.1 Flash, Kimi K3, and Qwen3.8 Flash. The public model catalog does
  not expose the provider catalog by default.
- No remote spend was incurred by this probe: `REMOTE_COST_USD=0.00`.

## CURRENT HARNESS

| Plane | Current owner | Finding |
|---|---|---|
| Perception | `hawking.perception` | One local perception surface; file, terminal, behavior, doctor, process inspection, and connected acts are real. Browser/desktop perception is parked. |
| Action | Hawking tool registry and workunit owner | Filesystem writes, tests, Git, receipts, and guarded shell operations are Hawking-owned and authority checked. |
| Knowledge | context/index/receipt paths | Durable receipts and indexed repository knowledge exist; the generated corpus needs narrower search boundaries. |
| Orchestration | Goal surface, workunit owner, Auto plan | Goals and workers are durable Hawking objects; remote provider calls are subordinate invocations. |
| Tool fabric | `default_tool_registry()` | 108 names resolve to 42 canonical tools plus 66 aliases. Aliases are discoverable/callable through the existing authority path. |
| Control surface | H-Web and Hawking CLI/API | H-Web is the accepted minimal operator surface and is backed by Hawking APIs; legacy HCLI/HIDE/Grok labels are not live authorities. |

### Goal/workunit evidence

- Live `/v1/goals` currently reports no active Goal and four distinct recent
  Goals, all terminal. `/v1/workers` reports four terminal Hawking workers.
- `GOAL-945600ECD0` has a real isolated linked worktree at
  `.worktrees/hweb-live-qualification-20260915`, two harmless fixture files,
  16 evidence entries, a grounded diff, and seal
  `GOAL-945600ECD0_SEAL_B1`. It did not touch production or the protected
  Flash/P0 lane.
- That historical WorkUnit receipt exposed the harness-collapse symptom: the
  machine-completed Goal state could be terminal while `last_outcome` retained
  `CONTINUE`. The source now has a narrow machine-edge repair that writes a
  coherent terminal state and preserves the old receipt unchanged.

## ABSORPTION / RENAMING GRAPH

| Historical surface | Current disposition | Evidence and decision |
|---|---|---|
| HCLI | **ABSORBED/RENAMED** | Staged migration renames the Python package and public docs to Hawking; current entry points are `hawking`, `h`, and `hawkingd`. Legacy state/docs remain evidence only. Do not revive HCLI. |
| AgentOS | **LIVE BUT HIDDEN / PROTECTED INTERNAL** | `hawking/agentos` is still imported by current Flash-science paths. It is not a separate public authority and must not be removed during this probe. |
| HIDE | **PARTIAL / COEXISTING RUST SUBSTRATE** | Five `hide-*` crates remain in the Cargo workspace beside Hawking crates, about 94.8k Rust LOC. Package names/descriptions still carry HIDE. Deletion is unsafe until caller/dependency parity is measured. |
| VMCP / VisionMCP | **EXTERNAL RETAINED BOUNDARY / PARKED** | `visionmcp` is a nested external repository and current docs treat it as an external sensory boundary. No second Hawking visual authority was added. |
| Grok | **ADAPTER / INERT LEGACY** | `grok_bridge.py` and compatibility references remain, but the live provider is OpenRouter and no Grok child is running. Do not activate it. |
| MLX | **DORMANT COMPATIBILITY** | MLX backend/model compatibility code remains, but `mlx` and `mlx_lm` are absent and the live remote path does not use it. Do not remove it in the dirty migration without a protected-lane audit. |
| WorldIR / PaintIR | **DOC/HISTORICAL ABSENT** | No current named runtime package or authority was found. |
| Ghidra / PyGhidra | **CANDIDATE / DOC-ONLY ABSENT** | No current executable or integrated Hawking wrapper was found. |
| Rizin / radare2 | **HOST-AVAILABLE / UNWRAPPED** | Host has `rizin 0.9.1` and `radare2 6.1.8`; no current Hawking registry wrapper was found. |
| Tesseract / OCRmyPDF | **HOST-AVAILABLE / UNWRAPPED** | Host has `tesseract 5.5.2` and `ocrmypdf 17.5.0`; no current Hawking registry wrapper was found. |
| tree-sitter | **PARTIAL / CONNECTED** | Rust index crates already depend on tree-sitter language grammars; Python bindings are absent. |
| Playwright / browser automation | **EXTERNAL CANDIDATE** | Playwright is present only in the retained external vision material/docs, not as a Hawking-native browser authority. |

## HAWKING-NATIVE COMPUTER USE

`HAWKING_NATIVE_COMPUTER_USE=NO`.

Hawking has real local file/terminal/process perception and PTY evidence on this
host, but no Hawking-owned AXUIElement, ScreenCaptureKit, Vision/DOM, CGEvent,
CDP, or Playwright control surface. The Codex CUA browser can verify H-Web as an
external operator surface; that must not be reported as a Hawking-native worker
capability.

## CENSUS AND PHYSICAL PROFILE

### Code and state

| Area | Measured size |
|---|---:|
| Python `hawking` source | 510 files / 186,712 LOC |
| Python `hawking` tests | 332 files / 61,538 LOC |
| `hawking/agentos` | 39 files / 18,060 LOC |
| Rust workspace | 754 files / 512,384 LOC |
| Hawking-named Rust crates | about 376,749 LOC |
| HIDE-named Rust crates | about 94,766 LOC |
| `tools` | 832 files / 345,990 LOC |
| `research` | 167 files / 73,487 LOC |
| H-Web review artifact | 2,146 LOC |
| `.hawking` state | about 9,008 files / 1.5G |
| `workspace/campaign` | about 1,019,179 files |

The repository is in a user-owned mass rename/migration state. The unusually
large receipt and campaign corpus is a real search and storage cost, not a
reason to delete evidence.

### Runtime and host

- Python 3.14.6, Node 26.7.0, Cargo 1.96.0, arm64 Darwin.
- Mac Studio with Apple M3 Ultra, 28 CPU cores, 60 GPU cores, 96 GB RAM, Metal
  4. The main native compute helper is present, while protected native model
  roles remain withheld.
- Data volume: about 83G free of 926G (91% used). This is an operational risk
  for generated receipts/campaign data and worktrees.

### Startup and warm endpoint observations

| Probe | Observation |
|---|---:|
| `import hawking` | about 0.03s |
| `python3 -m hawking --help` | about 0.04s |
| `python3 -m hawking tools --workspace . --repo-root .` | about 0.10s |
| `python3 -m hawking connectivity ... --no-network` | about 0.86s |
| `/health` warm p50 / p95 | 1.11ms / 1.31ms |
| `/v1/models` warm p50 / p95 | 18.77ms / 20.07ms |
| `/v1/workers` warm p50 / p95 | 0.88ms / 1.03ms |
| `/hawking/web/models` warm p50 / p95 | 0.52ms / 0.66ms |

### Tool registry and latency

The registry contains 108 names, 42 canonical tools, and 66 aliases. Measured
mutation classes are read-only 21, research 4, costly 7, workspace-write 2,
reversible-repo 3, reversible-runtime 4, and destructive 1. Fresh direct
read-only invokes measured approximately:

- `tools.catalog`: 1.50ms
- scoped Hawking `fs.search`: 14.64ms
- `fs.read` of `hawking/serve.py`: 0.76ms
- Git status tool: 115.89ms

## ARSENAL INVENTORY AND RECOVERY TABLE

| Capability family | Current usable power | Missing/parked power | Recovery path |
|---|---|---|---|
| Source search/read | `fs.search`, `fs.read`, index, Git | Broad-root search is noisy and expensive | Default to scoped roots/globs and exclude generated receipts/campaign output. |
| Repository mutation | guarded filesystem write, worktrees, Git status/diff | Registry does not expose every engine mutation spelling | Unify `repo.edit` with the canonical typed registry contract under the same guard. |
| Tests/repair | test tools, WorkUnit receipts, isolated linked worktrees | Historical completion contract could leave stale outcome fields | Keep the new coherent machine completion edge and add RED tests for the full evidence seal. |
| Binary/reverse | host Rizin/radare2 | No Hawking-owned typed adapter | Add one read-only bounded adapter with timeout, path scope, and receipt, only after a deterministic test. |
| Document/OCR | host Tesseract/OCRmyPDF, Python image/PDF libraries | No registry adapter or redaction contract | Add a narrow document extraction adapter; never grant arbitrary filesystem/network access. |
| Browser/desktop | external Codex CUA can inspect H-Web | no native Hawking browser/desktop control | Separate future local secure bridge from H-Web; require explicit capability and evidence boundaries. |
| Context/index | Rust tree-sitter dependencies and current context tools | generated-state compaction/exclusion needs work | Add corpus boundaries and measurement before adding another index. |
| Model routing | configured cloud Auto pool and worker packets | accepted-work metrics are not yet the routing authority | Persist cost, accepted result, repair, latency, and reviewer-overturn evidence per task class. |

## TOP WASTES AND HARNESS GAPS

1. **Search surface waste:** `.hawking` contains roughly 9,008 files and the
   campaign corpus roughly 1,019,179 files. A broad search can walk about 5,001
   generated receipt files before finding useful source. Search must be scoped
   by repository owner, source kind, and explicit glob.
2. **Duplicate naming and coupling:** `tools`, `research`, Hawking Python, and
   coexisting HIDE Rust crates are all large. The next cleanup must be graph-
   driven; deleting names first would destroy migration provenance.
3. **Mutation contract split:** `hawking/chat_tools.py` and engine handling
   contain a `repo.edit` operation, while `default_tool_registry()` exposes
   `filesystem.write` as the canonical workspace mutation. This is the most
   actionable harness-collapse gap found.
4. **Completion evidence split:** the historical live Goal reached a coherent
   result seal but retained a stale `last_outcome`. The machine edge is now
   repaired for future discrete Goals; the old receipt is intentionally not
   rewritten.
5. **Native computer-use boundary:** local perception is real, but browser and
   desktop control are not owned by Hawking. External CUA evidence must remain
   classified as operator verification.
6. **Disk pressure:** 83G free with large generated state makes uncontrolled
   search, indexing, or worktree fan-out risky.

## SMALLEST ARSENAL DELTA ALREADY APPLIED

`hawking/workunit_owner.py` now uses a single
`_complete_discrete_goal(record, owner, emit=None)` machine edge for the two
discrete completion paths. It sets terminal state, outcome, phase, worker
release, kill reason, and next action together before saving and emitting the
completion event.

New regression coverage:
`hawking/tests/test_discrete_goal_completion_state.py` proves a stale
`last_outcome=CONTINUE` cannot survive machine completion. The focused suite
passed:

```text
31 passed in 0.71s
```

The applied change is deliberately small and does not weaken authorization,
protected-command checks, provider attribution, or native-lane withholding.

## IMPLEMENTATION WAVES

### Wave 0 — completion and evidence coherence (current)

- Keep the completion helper and regression test.
- Add the missing RED contract around applied edit, grounded passing test,
  result packet, and terminal state as one acceptance boundary.
- Keep historical receipts immutable.

### Wave 1 — typed Arsenal adapters

Add only bounded read-only adapters that earn their place through tests:

- one `reverse.inspect` adapter over the already-installed Rizin/radare2 tools;
- one `document.extract` adapter over existing OCR/PDF capabilities;
- explicit root/path scope, timeout, output truncation, redaction, and receipt;
- canonical registry discovery, with aliases retained only for compatibility.

### Wave 2 — native perception decision

Choose one Hawking-owned local browser/desktop boundary only after a security
and capability decision: a local Playwright/CDP bridge, Apple AXUIElement /
ScreenCaptureKit / CGEvent bindings, or a hardened retained external VMCP
boundary. Do not add an Internet-facing daemon or claim external CUA as native.

### Wave 3 — HIDE/Hawking collapse

Build a caller/dependency graph and differential tests for the five `hide-*`
crates versus their Hawking-named consumers. Retire only proven duplicate
surfaces, preserving history and the protected Flash-science callers.

### Wave 4 — context and accepted-work routing

Exclude generated corpora by default, measure context/token waste, and make
Auto learn from accepted WorkUnits, repair probability, cost, latency, and
review overturns rather than benchmark labels alone.

## EXPECTED DELTAS

| Change | Expected effect |
|---|---|
| Current completion helper + test | roughly 30–45 LOC; no new dependency; startup/endpoint latency unchanged; remote spend $0 |
| Typed local Arsenal adapters | small additive LOC; no remote spend; bounded subprocess latency with explicit timeouts |
| Search exclusions/compaction | lower filesystem traversal and model-context waste; must preserve receipt reachability |
| HIDE/Hawking collapse | potentially large LOC reduction, but only after parity evidence; no safe estimate before graphing callers |
| Native computer-use bridge | dependency and permission cost depends on selected substrate; not part of this probe |

## EXACT NEXT BUILD MISSION

Run one bounded local Hawking Goal in an isolated `.worktrees/` worktree:

> Map `repo.edit`, `filesystem.write`, `tests.run`, and `git.*` into one
> canonical typed mutation contract without weakening any authority guard. Seed
> a RED test proving that a Goal cannot become COMPLETE until an authorized edit,
> grounded passing test, result packet, and coherent terminal state all exist.
> Implement the smallest fix, run the focused suite, inspect Git status and the
> grounded diff, and preserve all receipts. Do not touch Flash/Pulsar, do not
> activate HCLI/HIDE/Grok, and do not spend remote budget.

This is the highest-leverage next mission because it closes the observed
operator-visible harness gap before adding more Arsenal breadth.

## HAWKING-GROK-STAGING REVIEW

The requested staging directory is a container for two linked Git worktrees,
not one disposable checkout:

| Worktree | State | Useful material | Decision |
|---|---|---|---|
| `passport-check` (`worker/passport-check`) | At the current `main` commit; untracked `.hawking-worker/` and `docs/worker/` | `PASSPORT_CHECK_GAP_EVIDENCE.md` identifies a deterministic fail-closed gap where `reuse_method` does not consult sealed passport Scars. `PASSPORT_CHECK_PROPOSED_FIX.md` gives a bounded schema, executor gate, and focused-test plan. | Preserve. No code was applied in that worktree. |
| `worker-integration-snapshot` (`worker/WORKER_INTEGRATION_SNAPSHOT`) | HEAD `b11b11d4e`, with extensive uncommitted source, tests, and receipt changes | Historical `SEAL_B83` gate-key/digest-bind evidence plus a large worker campaign and receipts. | Preserve. It is useful evidence but not a clean merge candidate. |

The integration snapshot’s unique commit still edits legacy `hcli` paths and
its branch carries a broad divergent campaign history. The corresponding
gate-key and digest-bind concepts are already present in the current dirty
Hawking source, so cherry-picking the old commit would add conflict and
provenance risk without a demonstrated current-tree gain. Both worktrees are
dirty; deleting either would destroy uncommitted worker evidence. Therefore:

```text
HAWKING_GROK_STAGING_USEFUL=YES
HAWKING_GROK_STAGING_DELETED=NO
HAWKING_GROK_STAGING_MERGED=NO_STALE_DIRTY_BRANCH
```

The passport-check gap is a strong candidate for a later protected Flash/Gravity
mission, but it is outside this probe’s safe additive delta and was not activated
or spent against.

## FINAL PROBE STATE

```text
OPENROUTER_KEY_PRESENT=PASS
REMOTE_COST_USD=0.00
HAWKING_NATIVE_COMPUTER_USE=NO
MAIN_GOAL_DISTURBED=NO
PROTECTED_NATIVE_LANES=WITHHELD
READY_FOR_NEXT_ARSENAL_WAVE=YES
```

The accepted H-Web baseline remains live. Qualification was not reopened, no
provider was resurrected, no legacy authority was restored, and no credential
was exposed.
