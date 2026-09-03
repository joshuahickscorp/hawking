# aud15 — function merge candidates

Discovery only. No code was rewritten, moved, merged, or deleted.
Evidence is `git grep` / `git show` of `HEAD` `04193ccbc` (2026-09-02).
This worktree is sparse: `hcli/` is **not on disk**. A live HCLI daemon holds
~110 uncommitted `hcli/` files against some checkout; that tree was **not**
diffed. Grade of every claim: `SOURCE_INSPECTION` / `STATIC_VERIFICATION`.
Nothing here is `MEASURED`. Tests were listed, not executed.

Machine-readable twin: `receipts/audit/aud15-function-merge-candidates.json`.

## What this audit is allowed to say

Call sites are syntactic calls of the symbol. A module import is not a caller.
Definitions with zero calls of themselves are reported as dead, not as
capability. Roadmap prose was not used as evidence.

## Headline

The known starting point is real, and it is worse than advertised.

`hcli.persist.atomic_write_bytes` **already exists** (`hcli/persist.py:45`)
and is already called (`hcli/session.py:228`). The three flash-body modules
import `atomic_write_json` from that same module and still ship a private
`_atomic_write_bytes` whose body is the persist contract (temp + fsync +
`os.replace`). The G028 test that claims “one atomic writer” only checks that
`resources` / `runtime` re-export `atomic_write_text`. It does not fail.

That is the only merge I would call **high-confidence and high-value**.

Around it sit two large, boring, safe families (chunked file SHA-256, dict-or-None
`_read_json`) and a list of lookalikes that must **not** be DRY’d: PID liveness,
repo-root locators, phys_footprint vs RSS, per-gate causality text, Engine vs
Grok vs Ledger receipts.

## Classification counts (25 candidates)

| class | n | ids |
|---|---|---|
| EXACT_DUPLICATE | 5 | C01, C05, C07, C08, C25 |
| PARAMETERIZABLE | 8 | C04, C06, C10, C13, C14, C16-sub, C18, C19 |
| SHARED_CORE_WITH_SPECIALIZED_EDGE | 2 | C11, C16 |
| SEMANTICALLY_SIMILAR_BUT_MUST_REMAIN_DISTINCT | 8 | C02, C09, C12, C15, C17, C22, C23, C24 |
| HISTORICAL_DUPLICATE | 2 | C20, C21 |
| NOT_MERGEABLE | 1 | C03 |

Safe first HCLI-py batch, if a later lane implements: **C01, C05, C07, C08, C10, C13**.
C04 and C06 are safe only with an explicit `missing=` / “dict-only” parameter.
Do not start with C09, C11, C12, C15, C17, C18, C22, C23.

## Surprises (do not smooth these over)

1. **Canonical bytes writer already exists.** C01 is a regression against persist’s
   own docstring (“callers that used to ship a private `_atomic_write*` re-export
   these names”).
2. **`TestAtomicWriteOneAuthority` cannot see C01.** It would stay green after a
   gut of persist.atomic_write_bytes as long as the text re-exports remain.
   What would settle: a grep assertion that `def _atomic_write_bytes` does not
   reappear under `hcli/`, mutation-checked.
3. **`flash_vector_component_body._sha256_file` is dead.** Defined, never called
   in that file. Copied and left.
4. **Flash `LAKE_ROOT` ignores `HCLI_MODEL_LAKE_ROOT`.** `specimens._lake_root`
   and `tool_registry._model_lake_roots` honor the env. Flash body writers only
   overlay `HCLI_FLASH_NEXT_ROOT`. A redirected lake is split-brain. Settling
   this is a behaviour change, not DRY.
5. **`tools/future/_common.write_receipt` is not crash-safe.** It seals, then
   `Path.write_text`. persist is the fsync+replace path.
6. **`PermissionError` splits PID liveness.** `goal_bank._pid_alive` → True
   (owner exists, not ours). `process_alive` / `background._pid_alive` → False
   (doubt is dead). Same `os.kill(pid, 0)`.
7. **This sparse tree cannot see the live daemon’s uncommitted `hcli/`.** Diff
   that tree against `HEAD:hcli/` before merging anything.

## Per-candidate reports

### C01 — EXACT_DUPLICATE — flash `_atomic_write_bytes` vs `persist.atomic_write_bytes`

- **Symbols.** `_atomic_write_bytes` in
  `hcli/agentos/flash_component_body.py:59`,
  `flash_vector_component_body.py:50`,
  `flash_matrix_component_body.py:62`;
  `hcli.persist.atomic_write_bytes` at `persist.py:45`.
- **Behaviour.** mkdir, `.{name}.{pid}.{uuid}.tmp` in the same directory, write,
  flush, fsync, `os.replace`, unlink on error. persist additionally `TypeError`s
  if payload is not `bytes`.
- **Callers (the write, not the import).**
  `flash_component_body.py:213`,
  `flash_matrix_component_body.py:213`,
  `flash_vector_component_body.py:141`,
  `session.py:228` (persist).
  Vector already `import flash_component_body as component_body` and still
  does not call that module’s writer.
- **Tests.** `tools/haider/hcli/tests/test_core_authorities.py` covers text/json
  re-exports only. **No test calls `atomic_write_bytes`.** Capability: `CALLABLE`,
  not `TESTED`.
- **Merge safety.** High. Canonical API already exists.
- **Deletion.** Three local defs. `from hcli.persist import atomic_write_bytes`.
- **Risk.** Low. Callers already pass bytes.

### C02 — MUST_REMAIN_DISTINCT — `lab/lineage` compare-and-swap write

`lab/lineage/lifecycle.py:107` takes `expected_previous_sha256`, preserves mode,
fsyncs the directory, raises `LifecycleError` if another controller won.
Callers: `:170`, `:863`. Last-writer-wins persist would drop the CAS.
Not an HCLI-py merge.

### C03 — NOT_MERGEABLE — Rust `atomic_write_bytes`

`crates/hawking-core/src/model/qwen_complete_binary/qwen80_uniform_q4.rs:426`.
Same contract, different language. This lane must not touch crates.

### C04 — PARAMETERIZABLE — chunked file SHA-256

~21 copies of “1 MiB chunks, `hashlib.sha256`, hexdigest”.

Two modes, proven by the body not the name:

| mode | behaviour | examples |
|---|---|---|
| soft | `except OSError: return None` | most of `hcli/agentos/*`, `flash_next.py`, `architecture.py`, `vmcp_adapter.py` |
| hard | no try; OSError propagates | `autonomy_gate.py:215`, `tool_registry.py:193` |

`autonomy_gate` uses the digest as a verifier expected value (`:619`). Softening
it to `None` would let two missing files compare equal.

**Dead:** `flash_vector_component_body._sha256_file` (def at `:39`, no call).

Canonical proposal: `hcli.persist.sha256_file(path, *, missing="none"|"raise")`.
Do not default autonomy_gate onto `"none"`.

### C05 — EXACT_DUPLICATE — SHA-256 of bytes

Ten `return hashlib.sha256(value).hexdigest()` copies in flash_* plus
`tool_registry._sha256_bytes`. `engine._sha256_bytes` is the same plus
`None → None`. Safe to put on persist. Callers include
`flash_component_body.py:214`, `flash_vector_component_body.py:142`,
`flash_loader_roundtrip.py:100`.

### C06 — PARAMETERIZABLE — soft dict `_read_json`

**19 character-identical copies** (18 named `_read_json`, plus
`modellake_supervisor._read` at `:20`): utf-8, OSError/Unicode/JSONDecode →
`None`, non-dict JSON → `None`. Densest caller: `flash_executable.py` (~20 calls).

**Must not swallow:**

- `delegate._read_json` (`:122`) returns `(value, defect)`. Missing file is a
  defect string. Callers unpack the pair (`delegate.py:441` and 9 more).
- `resident.ResidentStore._read_json` (`:255`) returns `Any` so inbox/knowledge
  **lists** survive. Dict-only would empty the inbox (`:295`).
- `frontier_scheduler._read_json` (`:148`) has no encoding and no dict check.

Canonical: `hcli.persist.read_json_object(path) -> Optional[dict]` for the 19.
Leave delegate and resident.

### C07 — EXACT_DUPLICATE — `_final_root`

Seven copies. `value or HCLI_FLASH_NEXT_ROOT` else
`(LAKE_ROOT / "specimens" / LAKE_SLUG).resolve()`.
Callers at each module’s `run_*` plus
`flash_router_representation_ab.py:145` calling `transform._final_root(None)`.

See C18 before “fixing” the default root.

### C08 — EXACT_DUPLICATE — agentos `_repo_root = parents[2]`

Five defs (`accelerator_regression`, `autonomy_gate`, `charge`,
`qwen27_runtime_identity`, `qwen38_fusion_audit`). Empty string is falsy in
both the ternary and `if value:` spellings.

Real extra **calls** (imports, then the symbol is invoked):
`protected_accelerator_benchmark.py:397`,
`protected_benchmark_watcher.py:203`,
`qwen27_mlp_diagnostic.py:173`.

This is not `find_repo_root`. Do not replace with C09.

### C09 — MUST_REMAIN_DISTINCT — repo-root locators

| helper | markers | why it exists |
|---|---|---|
| `hcli.paths.find_repo_root` | `.git`/`Cargo.toml` + `tools/headless` or `crates` | find **this** hawking checkout; silent fallback to the running code |
| `session_ledger.discover_repo_root` | `git rev-parse --show-toplevel` first | `/land` must not retarget a scratch repo at the live checkout. Docstring at `:79` is the incident. |
| `commands._land_repo_root` | wraps discover | git-mutating verbs |
| `agentos.runtime._find_repo_root` | `hcli/` + `pyproject.toml` | refuse to treat an external mission workspace as a fake repo |
| `runtime_iface.default_repo_root` / `genomes.runtime_genome._repo_root` | `hcli/runtime.py` + `receipts/headless`, else cwd | third marker set |
| `resources._default_repo_root` / `machine.default_repo_root` | re-export `find_repo_root` | not clones |

Tests: `hcli/test_checkpoint.py:194`, `hcli/test_session_ledger.py:73`,
`hcli/test_machine_status.py`. Capability: `TESTED`.

The only mild sub-merge is `runtime_genome._repo_root` ↔ `runtime_iface.default_repo_root`.

### C10 — PARAMETERIZABLE — gate `_write_receipt`

`native_gate:391`, `native_mission_gate:148`, `resident_gate:257`,
`modellake_gate._write:399`. Six identical lines; only the default filename
changes. Persistence is already `atomic_write_json`. Callers: one per `run_*`
(resident_gate twice). Do not merge with C22.

### C11 — SHARED_CORE_WITH_SPECIALIZED_EDGE — `causality_payload`

Eight gates. Shared empty envelope and `claim_kind` ternary. **`probe_performed`
/ `direct_observation` / `interpretation` are the gate’s identity** in the
status-causality ledger. Every file calls its own function (no cross-gate
import). Extracting the empty envelope is fine. Unifying the probe strings is
the verifier/proposer trap named in the brief.

### C12 — MUST_REMAIN_DISTINCT — PID liveness

| helper | pid 0 / None | PermissionError | recycled pid |
|---|---|---|---|
| `background._pid_alive` | False (`pid <= 0`) | False (OSError) | not checked |
| `goal_bank._pid_alive` | False (`pid <= 1`) | **True** | not checked |
| `grok_bridge.process_alive` | False (`if not pid`) | False | not checked |
| `resident._pid_matches` | False | False | `process_start_token` must match |

`process_alive` is `TESTED`
(`tools/haider/hcli/tests/test_grok_identity.py:73`,
`test_grok_cancel_orphan.py`). `mission.py:145` calls `process_alive`.
`_worker_live` is `_pid_matches`, not `_pid_alive`.

Merging goal_bank into process_alive would treat another uid’s live owner as
dead and drop the claim.

### C13 — PARAMETERIZABLE — `_json_safe`

Three copies: `json.loads(json.dumps(..., default=str))`.
`autonomy_gate` and `runtime` set `sort_keys=True`; `resident` does not.
Callers are in-module only (autonomy_gate `:473` ff, resident `:290`,
runtime `:247`). Low risk.

### C14 — PARAMETERIZABLE — temporary generation env

`native_gate._temporary_generation_env` is a contextmanager taking
`model_tokens` (`:374`, used `:611`).
`native_mission_gate` is save/restore with tokens hardcoded to 64
(`:156/:164`, used `:246/:348`). Same two env vars. Confirm the restore sits
in `finally` before deleting the pair.

`hcli.backends.structured_output_attempts` is already the single retry-budget
literal. This is not a second retry policy.

### C15 — MUST_REMAIN_DISTINCT — `_within` vs `Engine._safe_path`

- `background._within` (`:36`): does **not** resolve `path` first. One caller (`:166`).
- `tool_registry._within` (`:201`): resolves candidate first. Six-plus callers
  including ModelLake download containment (`:1091`).
- `Engine._safe_path` (`:3625`): rejects absolute paths and `..`, joins to
  `self.root`, then resolved symlink policy. Mutation authority. Five callers.

Merging toward the weaker `within` is a containment regression. Canonical, if
any: tool_registry’s resolved form. Engine stays.

### C16 — SHARED_CORE — context budget leftover

`hcli/context_budget.py` is already the stated single authority
(`per_seq_context`, `resolve`, `preflight`, `native_profile_limits`).
`TESTED` by `hcli/test_native_context_budget.py`.

Leftover: `engine._CHARS_PER_TOKEN = 3` (comment in context_budget says it
must match) and `Engine._estimate_prompt_tokens` which returns `max(1, …)`
so empty messages become **1**, while `estimate_tokens` returns **0** for empty
text. Import the constant; do not silently unify the 0-vs-1 edge.

`native_profile_limits` vs `tools/odyssey/contracts.load_profile` share no
behaviour (hawking-native JSON vs math-v1). Not a family.

### C17 — MUST_REMAIN_DISTINCT — memory accounting

`processes._footprint_bytes` (`:126`) runs `footprint -p` and parses
**phys_footprint**. The comment is a named incident: `ps -o rss` read 1.19 GB
while Activity Monitor showed 12 GB.

`resident_health.rss_bytes_of` is RSS via libproc then `ps`.
`resident.memory_decision` (`:403`) is host pressure/free/swap, no pid.
`TESTED`: `hcli/test_processes.py:119`, `hcli/test_memory_admission_is_live.py:56`.

Do not name a merged helper `memory()`.

### C18 — PARAMETERIZABLE, behaviour change — ModelLake root

`specimens._lake_root` (`:34`) and `_model_lake_roots` honor
`HCLI_MODEL_LAKE_ROOT`, default `/Volumes/corpdrive/hawking-modellake`.
Flash `LAKE_ROOT` is the hardcoded path. See surprise S04. A DRY that
“just uses `_lake_root()`” retargets Flash bodies. Decide the overlay first.

### C19 — SHARED_CORE — resident-watch test fixtures

`hcli/test_resident_watch.py:17` vs `test_resident_watch_stream.py:143`.
Same daemon + `m-watch` + unit G001. Stream adds `session_id`, omits
`mission.log`. Cosmetic. `session_id` is load-bearing for the stream test.

### C20 — HISTORICAL — tools `sha256_file`

20 files matching `def sha256_file` under `tools/`. Plus lab/operators and
headless copies. `tools/future/_common.py:577` already looks like a tools
canonical (hard hasher). Not on the HCLI daemon SCC. Opportunistic, not a
campaign. Do not start hashing ModelLake trees; `modellake_gate` refuses that.

### C21 — HISTORICAL — rust example `sha256_hex`

71 example files with `fn sha256_hex`. A few production copies under
`crates/hawking-core/src/`. Out of scope (`DO_NOT_TOUCH crates`).

### C22 — MUST_REMAIN_DISTINCT — Engine / Grok / Ledger `_write_receipt`

Same method name, three schemas:

- Engine `:4762` — `.hcli/receipts/<goal_id>.json` via `build_result_envelope`
  (6 call sites).
- GrokBridge `:1202` — grok-run task receipt (`command_run`, dry_run, …).
- Ledger `:935` — obligation verify sidecar + in-memory flush.

They already share persist for I/O. Unifying documents would mix authorities.

### C23 — MUST_REMAIN_DISTINCT — tools `write_receipt` vs persist

`tools/future/_common.write_receipt` (`:157`): seal, refuse foreign overwrite,
then **`Path.write_text`**. persist: fsync+replace, no seal. Combining them
drops one of those contracts.

### C24 — MUST_REMAIN_DISTINCT — `mission_state` name collision

`hcli.agentos.states.mission_state` (`:42`) → `AgentState` enum.
`hcli.mission.mission_state_path` (`:60`) → `Path`.
`delegate.py:442` reads the path. `mission.py:574` calls the enum mapper.
Types already disagree. Do not alias.

### C25 — EXACT_DUPLICATE, not worth a campaign — `_now`

`background._now`, `resident._now`, `providers._now` are `time.time()`.
`odyssey._now` (`:200`) returns UTC ISO **strings** for `recorded_at`.
Inline the floats if you must; do not touch odyssey.

## Hunt categories with no merge family

| hunt | finding |
|---|---|
| benchmark arithmetic | `flash_executable._ebpw_budget` is unique. No second solver. |
| retry/backoff | `backends.structured_output_attempts` is already the single literal. Grok `sleep` is poll-wait. |
| evidence normalization | no generic `normalize_evidence()`. Closest: C11 / C13. Domain `_normalize_*` stay. |
| profile parsing | `native_profile_limits` is unique. Odyssey `load_profile` is a different document. |
| resident lifecycle | `resident.py` is the HCLI authority. `tools/agentos/genesis_resident.body_is_up` is a different body. |

## Recommended later-lane order (not this lane)

This file is not a mandate to implement.

1. C01 bytes writer → persist (extend G028 test so it would have caught this).
2. C05 bytes hasher, C13 json_safe, C10 gate receipt default-path.
3. C07 `_final_root` **without** changing lake-root env semantics.
4. C08 parents[2] `_repo_root`, still not `find_repo_root`.
5. C04 / C06 only with `missing=` and dict-only parameters, plus a test that
   autonomy_gate still raises and resident inbox still accepts a JSON list.
6. Stop. Do not touch C09, C11 observation text, C12, C15, C17, C18, C22, C23.

## Blockers

- **B01 sparse checkout.** `hcli/` missing on disk. Evidence is `HEAD` only.
- **B02 tests not run.** `TESTED` means a test file calls the symbol.
- **B03 git index not writable from this executor.** `git add --sparse -A`
  failed with `index.lock: Operation not permitted` against
  `Downloads/hawking/.git/worktrees/...`. Object DB is not writable
  (`com.apple.macl`). The two receipt files exist on disk as untracked
  `receipts/audit/aud15-function-merge-candidates.{json,md}`. Uncommitted
  work is destroyed at cleanup — commit them from a process that can write
  the parent `.git`.

What would settle B01/B02: materialize `hcli/`, diff against the live daemon tree,
run `python3 -m pytest tools/haider/hcli/tests/test_core_authorities.py
hcli/test_native_context_budget.py hcli/test_processes.py
hcli/test_memory_admission_is_live.py -o addopts=""`.

What would settle B03:

```
git add --sparse receipts/audit/aud15-function-merge-candidates.json \
                 receipts/audit/aud15-function-merge-candidates.md
git status
git commit -m "audit(aud15): function merge candidates from call-site behaviour"
```
