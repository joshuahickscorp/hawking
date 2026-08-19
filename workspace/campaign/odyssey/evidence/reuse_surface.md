I'll scout the Hawking repo read-only and map the exact reusable APIs. Starting with the named files and a wide search for callers, schemas, and Odyssey patient-queue machinery.Key files are located. Next I’ll read the gate/controller/pipeline sources and fan out greps for callers, schemas, and TPS receipts.Core gate files are in. Next I’ll pull doctor/seal/dispatch/odyssey sources plus TPS receipts and the ascent state schema.Doctor and dispatch sources are in. I’ll now pull the ascent state schema, spawn callers, TPS receipts, and Odyssey patient-queue surfaces.The 29.3 TPS specimen is in the A3B recon receipt. Next I’ll read that pipeline plus remaining gate, contract, and Odyssey patient-queue sources.I’ve confirmed the 29.3 TPS number is an mlx_lm specimen. Next I’ll pull remaining APIs: spawn callers, Metal headroom, contract formation, and Odyssey patient surfaces.Worker_gate is only imported by the NOS pipeline. Next I’ll map Metal admission, contract formation, remaining architectures, and Odyssey patient machinery.Architecture dispatch is Qwen/Llama/DeepSeek/RWKV plus Gravity artifacts. I’ll now fill Metal admission, Mixtral/Falcon/Mamba edges, TPS scripts, and Odyssey patient machinery.A few contradictions remain (Mixtral in-tree vs extra pack, no Metal headroom in worker_gate). I’ll read the remaining admission, target, Odyssey, and baseline sources.I still need the ascent target schema, Mixtral dispatch truth, DeepSeek-V4 load path, spawn callers, and Odyssey patient-queue files.I have the controller/state loop. Next I’ll close remaining gaps: Gravity load dispatch, spawn-gating callers, Odyssey/foundry patient machinery, and the mlx vs transformers baseline.A few contradictions remain (Mixtral in-tree vs extra-pack comments). I’ll read spawn admission, Gravity arch match, Odyssey modules, and the generate/TPS entrypoints.A few last facts: native TPS measurement, hawking generate entry, Odyssey module inventory, and whether ascent is G001-hardwired.# ODYSSEY-I reuse-surface recon

Read-only scout of `/Users/scammermike/Downloads/hawking`. No files were edited. There is no `tools/machine_state.py`; the live module is `tools/agentos/machine_state.py`.

---

## 1. MEMORY / WORKER GATE

### `tools/worker_gate.py` — wired-memory spawn gate (G133)

Calibrated on `ascension_qwen38_hybrid` workers. **Does not inspect Metal headroom.** Gates on wired pages, compressor, swap, and a 32 GB non-wired reserve.

Constants (`tools/worker_gate.py:38–44`):

```38:44:tools/worker_gate.py
BASELINE_WIRED_GB = 4.63      # G131, measured with zero workers
PER_WORKER_WIRED_GB = 15.59   # G131, 40/43 samples at n=3
MEASURED_MAX_N = 3            # G131, held 235s stable
CRASHED_AT_N = 5              # directive S18, "do not repeat the five-worker crash"
RESERVE_GB = 32.0             # between the 51.7 that worked and the 20.5 that crashed
COMPRESSOR_FLOOR_GB = 0.10    # G131, flat through the entire three-worker run
PROC = "ascension_qwen38_hybrid"
```

**Signatures**

```47:47:tools/worker_gate.py
def observe() -> dict:
```

Live `vm_stat` / `sysctl` / `ps` snapshot (`:59–68`):

| key | source |
|---|---|
| `total_gb` | `sysctl -n hw.memsize` / 1e9 |
| `wired_gb` | Pages wired down × page size |
| `free_gb` | Pages free |
| `inactive_gb` | Pages inactive |
| `compressed_gb` | Pages occupied by compressor |
| `swap_used_mb` | `vm.swapusage` `used = N M` |
| `workers_resident` | `ps` lines whose comm contains `ascension_qwen38_hybrid` |
| `worker_rss_total_gb` | sum of those RSS |

```71:71:tools/worker_gate.py
def gate(obs: dict, reserve: float = RESERVE_GB) -> dict:
```

Decision (`:82–112`):

- If `n = obs["workers_resident"] >= 1`: `per = (wired - 4.63) / n`, else G131 `15.59`.
- `projected = wired + per`, `headroom = total - projected`, `spawn_target = n + 1`.
- REFUSE if **any** of: `swap_used_mb > 0`; `compressed_gb > 0.10 * 1.5`; `headroom < reserve`.
- If permitted and `spawn_target > 3`: `unvalidated=True` (n=4 never measured; n=5 crashed).

Return dict (`:99–112`):

```python
{
  "workers_resident": int,
  "spawn_target": int,                 # n+1
  "per_worker_wired_gb": float,
  "per_worker_source": str,
  "current_wired_gb": float,
  "projected_wired_gb": float,
  "projected_headroom_gb": float,
  "reserve_gb": float,
  "decision": "PERMIT" | "REFUSE",
  "unvalidated": bool,
  "note": str,
  "reasons": list[str],
}
```

```115:115:tools/worker_gate.py
def sweep(upto: int, obs: dict) -> list[dict]:
```

Projects `gate()` for `n in range(upto)` using the G131 linear law, not live wired.

**CLI**

```
./tools/worker_gate.py --sweep 6
./tools/worker_gate.py --sweep 6 --out receipts/.../G133.json
```

`--out` writes schema `hawking.nos.worker_spawn_gate.v1` (`:172–204`). Self-check: injected swap/compressor must REFUSE (`sys.exit(1)` if they do not); sweep must have a first REFUSE (`:166–168`).

**Metal headroom is not here.** Wired Metal shows up only as “Pages wired down”. Dedicated Metal/IOAccelerator accounting is in `lab/genesis_pool.py` and `crates/hawking-core/src/model/qwen38_host_admission.rs` (below).

### `tools/agentos/machine_state.py` — clean-box / resource snapshot (S001)

```89:101:tools/agentos/machine_state.py
def snapshot() -> dict:
    total_gib, free_gib = _ram_gib()
    du = shutil.disk_usage(str(Path.home()))
    return {
        "type": "machine_state",
        "chip": _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model"),
        "ncpu": os.cpu_count(),
        "ram_total_gib": round(total_gib, 1) if total_gib else None,
        "ram_free_gib": round(free_gib, 1) if free_gib is not None else None,
        "disk_free_gib": round(du.free / 1024**3, 1),
        "load_1m": _load_1m(),
        "active_grok_lanes": _active_grok_lanes(),
    }
```

```104:115:tools/agentos/machine_state.py
def clean_box_ok(snap: dict | None = None, min_free_gib: float = 15.0) -> tuple[bool, str]:
    """A clean GPU/TPS measurement is safe only if no other heavy lane is live and
    disk is above the emergency floor. Conservative: any live worktree counts."""
    ...
    if lanes: return False, f"{len(lanes)} live lane(s) may contend GPU/memory: {lanes}"
    if (s.get("disk_free_gib") or 0) < min_free_gib: return False, ...
    if (s.get("load_1m") or 0) > (s.get("ncpu") or 1) * 0.5: return False, ...
    return True, "no live lanes, disk ok, load ok"
```

CLI: `python3 tools/agentos/machine_state.py` → JSON + `clean_box_ok` / `clean_box_reason`.

This is a **measurement-box** gate (do not time while grok lanes live), not a worker-admit gate. It does not look at wired, compressor, swap, or Metal.

### Adjacent admit APIs (not `worker_gate`, but they actually spawn)

| API | File | What it gates |
|---|---|---|
| `GenesisPool._refuse_if_full` / `spawn` | `lab/genesis_pool.py:645–679` | `alive >= safe_n` (default 3); optional `min_free_bytes`; swap>256 MiB and free<4 GiB. Raises `AdmissionRefused`. **Does not import `worker_gate`.** Counts Metal as `phys_footprint` / IOAccelerator (`:48–54`, `:358–364`). |
| `plan` / `admit` | `tools/genesis_capacity.py:71,110–132` | Shared-session fill to 90%, minus 14.08 GiB generation reserve and 4 GiB no-swap floor. `admit <n>` exit **0** ADMIT / **1** REFUSE / **2** missing n. |
| `decide_admission(memory, request, reserve_bytes) -> AdmissionDecision` | `crates/hawking-core/src/model/qwen38_host_admission.rs:176` | Free pages − `cost_bytes` vs 4 GiB reserve. Verdict `Admit`/`Refuse`. Schema `hawking.qwen38.host_admission.v1`. |
| `memory_lane_cap()` | `tools/ascent_daemon.py:323` | Concurrent **Grok lanes** from free+inactive+purgeable, 6 GiB/lane, 90% fill. Not a model worker. |

### How existing code gates a spawn

Repo-wide `import worker_gate` / `worker_gate.gate` hits **only** `tools/nos_pipeline.py` (`:32,54–58,139`). Production spawn of Qwen3.8 children is `lab/genesis_pool.py:spawn` → `_refuse_if_full` (count + optional free/swap). Ascent lanes use `machine_state.clean_box_ok` + `memory_lane_cap`, not `worker_gate`.

`nos_pipeline` usage (`:53–58`):

```53:58:tools/nos_pipeline.py
    if want_worker:
        obs = worker_gate.observe()
        g = worker_gate.gate(obs)
        trace.append({"stage": "spawn", "verdict": g["decision"], "note": g["note"]})
        if g["decision"] == "REFUSE":
            raise GateRefused("spawn", g["note"])
```

**Reuse note:** Odyssey-I should call `worker_gate.gate(observe())` before a **new process** that looks like `ascension_*` (wired ~15.6 GB). For a different binary, `PROC` and the G131 coefficients are wrong. Metal-private copies need `genesis_pool` / `qwen38_host_admission`, not this gate.

---

## 2. DISK RECLAIM — `tools/reclaim_safe.sh`

There is **no `--disk-floor` flag**. Floor lives in the Python governors (`DISK_FLOOR_GIB = 15.0` in `ascent_controller.py:35` and `ascent_daemon.py:54`). The script always reclaims; callers decide *when*.

**Invocation**

```
bash tools/reclaim_safe.sh          # reclaim
DRY=1 bash tools/reclaim_safe.sh    # print DRY: …, do nothing
```

Callers: `ascent_controller.reclaim_if_tight` (`:86–90`) when `disk_free_gib < 40`; `ascent_daemon.govern` (`:265–272`) when `disk_free_gib < 90` (daemon also deletes `~/.claude-grok/tasks/**/diff.patch` >50M and reaps finished grok worktrees).

**What it does** (`:23–75`)

1. Build caches: `rm -rf workspace/ops/build/*` unless `pgrep -x cargo`.
2. Grok task dirs under `~/.claude-grok/tasks`: if `pgrep -f $id`, keep; else delete everything **except** `grok-report.md` and `metadata.json`.
3. This-repo grok worktrees (`~/.claude-grok/worktrees/` in `git worktree list`): skip running / `<1d` old; dirty → commit `wip: preserve before reclaim` then `git worktree remove --force`; clean → remove (branch kept); `git worktree prune`.
4. `brew cleanup -s`; wipe `~/Library/Caches/pip` and Homebrew downloads.

**Hard protections** (header `:5–10`): never touch `workspace/campaign/records/runs/**`, `workspace/**/quality-diagnostics/**capture**`, ACTIVE lanes, FOREIGN worktrees, DIRTY trees without first committing.

**Exit codes:** `set -euo pipefail`. Success → 0. No named codes. Individual `rm`/`brew`/`worktree remove` are `|| true`. A failing `df`/`git worktree list`/`pgrep` (without `|| true`) can abort non-zero. No disk-floor halt inside the script.

---

## 3. DETACHED CONTROLLER — `tools/ascent_controller.py` (full) + daemon that actually forms contracts

### CLI

```
python3 tools/ascent_controller.py status
python3 tools/ascent_controller.py run [--max-lanes N]   # default 4
python3 tools/ascent_controller.py selfcheck
```

Exit: `0` queue dry / lanes launched; `2` disk below 15 GiB (`:168–170`).

### State file

`receipts/ascent-2026-08-16/ASCENT_STATE.json` (`:32`). Empty default (`:75–78`):

```json
{ "schema": "hawking.ascent_controller.v1", "targets": [], "history": [] }
```

Live file also has `generated`, `note`, `incumbents` (q80 / dsv4f science), `sources_on_device`. **Target object** (hand-written example `:540–556`; auto-generated in daemon `:1276–1291`):

```json
{
  "id": "q80-coherence-probe",
  "model": "q80",
  "hypothesis": "...",
  "target_stage": "density/representation",
  "contract": "workspace/ops/ascent-lanes/q80-coherence-probe.md",
  "resource_class": "GPU_EXCLUSIVE",
  "probability_of_success": 0.9,
  "recoverable_ns_per_token": 0,
  "density_frontier_gain_ns_equiv": 0,
  "information_gain": 900000000,
  "transfer_value": 300000000,
  "experiment_cost": 1.0,
  "status": "pending|running|retained|rejected|launch_failed|stale_no_process|deauthorised|mechanism_refused|admission_refused",
  "tier1": "…",
  "task_id": "optional, controller-only",
  "mechanism": "daemon-only, required to launch",
  "auto_generated": true,
  "from_bottleneck": "…",
  "from_lane": "…",
  "tier1_command": "optional override",
  "tier1_expect": "optional",
  "tier1_forbid": "optional",
  "profile": "gate (default)"
}
```

History row (`controller:191–193`): `{id, task_id, outcome, tier1}`.

### Loop (`run`, `:162–201`)

```
while launched < max_lanes:
  snap = machine_snapshot()          # agentos.snapshot + clean_box_ok(15 GiB)
  reclaim_if_tight(snap)             # if disk < 40 GiB → bash reclaim_safe.sh
  if disk < 15 GiB: return 2
  pending = targets with status==pending
  if none: return 0
  target = max(pending, key=value)
  launch(target) → grok-run delegate --background
  wait_for(task_id) up to 14400s, poll grok-run status every 60s
  if done: tier1_verify else reject
  status = retained | rejected; append history; grok-run cleanup --id
```

### Ranking (`value`, `:60–72`)

```
VALUE = p_success * (recoverable_ns + density_gain + info + transfer) / max(cost, 0.1)
```

Default `p_success=0.5`. Ordering only. Selfcheck asserts large uncertain win > small certain win.

### Contract formation — **not in the controller**

`launch` (`:128–143`) requires `target["contract"]` already on disk. It never writes a contract.

Contract synthesis is `tools/ascent_daemon.py:generate_targets` (`:1153–1433`): harvest `NEXT_BOTTLENECK:` from `~/.claude-grok/tasks/*/grok-report.md`; ask Genesis resident (`genesis_proposes`); write `workspace/ops/ascent-lanes/<tid>.md` with a **Qwen3.8/Genesis-hardwired** body (uniform-q4-v1, `gpu_lane_lock.sh`, Qwen greedy-id oracle, DENY list). Then `target["contract"] = str(path)`.

### Lane launch (`controller:128–143`)

```
~/.claude-grok/bin/grok-run delegate
  --task <target.id>
  --contract <path>
  --repo <REPO>
  --profile <target.profile or "gate">
  --background
```

Task id extracted by regex `<slug>-YYYYMMDD-HHMMSS`.

Daemon launch (`ascent_daemon.py:144–189`) is **foreground grok-run in a detached process group** (so it has a real `launcher_pid`), same argv minus `--background`.

### Tier-1 (`controller:93–114`)

Reject-only, 1800s, `shell=True`:

- no `tier1_command` → **pass** (“not a promotion”)
- nonzero exit → fail
- `tier1_expect` missing from stdout+stderr → fail
- `tier1_forbid` present → fail

Daemon `tier1` (`:1438–1456`) adds per-model defaults (`TIER1` `:82–100`): `cargo build --profile release-fast -p hawking-core` (qwen38) or a named example (q80 / dsv4f), expect `Finished`, forbid `error[`.

### Ledger update

Controller: mutate target `status`/`tier1`, append `history`, `save(STATE)`. Daemon also writes `PROMOTION_QUEUE.json` (`hawking.ascent.promotion_queue.v1`) with `disposition` MERGE_READY / NEEDS_COMPOSITION / … and **never merges**.

### VERDICT: extend vs Genesis-hardwired

`ascent_controller.py` has **zero** `G001`/`G0xx` references. The loop is a generic JSON queue + grok-run + optional shell gate. Reusable as an Odyssey patient-queue **skeleton**.

It is **not** a drop-in patient controller:

- State path is hard-wired to `receipts/ascent-2026-08-16/`.
- Value keys are ns/token experiment economics, not patient identity.
- Contracts are opaque markdown files; the writer (`ascent_daemon.generate_targets`) is Genesis/Qwen3.8-specific (`ACTIVE_MODELS = {"qwen38"}` `:445`, sealed-model `DEAUTHORISED_PATTERNS`, resident prompt, artifact `uniform-q4-v1`).
- Daemon `one_pass` interleaves Genesis lifecycle + AgentOS HCLI + GPU lane lock. That is the old organism, not a generic queue.

**Use the controller loop. Do not extend the daemon as Odyssey-I.** Write a new target schema + contract factory.

---

## 4. NOS GATE PIPELINE — `tools/nos_pipeline.py`

### CLI

```
./tools/nos_pipeline.py selftest
./tools/nos_pipeline.py selftest --out receipts/.../G27.json
```

Only command is `selftest`. Receipt schema `hawking.nos.pipeline.v1`. Exit 0/1 from selftest boolean.

### Entry

```46:47:tools/nos_pipeline.py
def qualify_and_promote(candidate: dict, parent: dict, *, want_worker=True,
                        timing_fn=None, provenance_links=None) -> dict:
```

Raises `GateRefused(stage, reason)`. Success:

```python
{"candidate": str, "final": "PROMOTE"|"DEVELOP", "trace": [stage dicts], "rebind": dict|None}
```

### Stages (order is the law)

| # | Stage | Function | Input | Refuse when |
|---|---|---|---|---|
| 1 SPAWN | `worker_gate.observe()` + `worker_gate.gate(obs)` | live machine; skipped if `want_worker=False` | `decision == "REFUSE"` |
| 2 TIMING | `gpu_lane_guard.guard(timing_fn, label=f"qualify:{name}")` | only if `timing_fn` given | `verdict == "VOID"` (other `ascension_qwen38_hybrid` process before or after) |
| 3 DOCTOR | `doctor_seal.seal(candidate["seal"])` | seal dict (below) | `REFUSED` |
| 4 PROVENANCE | `provenance_chain.seal(links)` + `check(links, root)` | `provenance_links` or `candidate["provenance_links"]`; skipped if none | check false |
| 5 PROMOTE | `successor_select.rank_against_parent(candidate["metrics"], parent["metrics"])` | metrics: `name, bpw, token_ns, doctor_pass, provenance_valid, native_path, no_hidden_fallback` | `Refused` (hard gate); silent T regression → `DEVELOP` not raise |
| 6 REBIND | `worker_checkpoint.checkpoint` + `rebind` | only if decision `PROMOTE`; needs `checkpoint_state` with 6 S16 fields | `Incomplete` |

### Downstream signatures you must satisfy

```49:51:tools/gpu_lane_guard.py
def guard(fn, pattern: str = PROC, label: str = "timing"):
    """Run fn() between two contention witnesses. Returns (result, verdict)."""
```

Verdict: `{label, verdict: VALID|VOID, witness_before, witness_after, wall_s, why}`.

```41:42:tools/successor_select.py
def rank_against_parent(cand: dict, parent: dict, t_tol: float = 0.02,
                        allow_t_regression: bool = False) -> dict:
```

Hard gate keys (`:28`): `doctor_pass`, `provenance_valid`, `native_path`, `no_hidden_fallback`. Returns `{name, decision: PROMOTE|DEVELOP, reason}`.

```58:59:tools/worker_checkpoint.py
def checkpoint(worker: str, parent: str, obligations: list[str], state: dict,
               store: pathlib.Path = STORE) -> pathlib.Path:
```

```83:83:tools/worker_checkpoint.py
def rebind(path: pathlib.Path, new_parent: str) -> dict:
```

Required state keys: `hypothesis, code_changes, measurements, negative_science, blocker, next_experiment`. Rebind **invalidates measurements**.

### Standalone call

```python
sys.path.insert(0, "tools")
import nos_pipeline
trace = nos_pipeline.qualify_and_promote(candidate, parent, want_worker=True)
# or want_worker=False to skip live wired gate (selftest path)
```

There is no “qualify this artifact directory” CLI. You assemble the candidate dict yourself (`_good_candidate` `:107–126` is the fixture).

---

## 5. DOCTOR

Four different instruments. None is a generic “grade this HuggingFace checkpoint” API.

### 5a. Seal (structural) — `tools/doctor_seal.py`

```33:55:tools/doctor_seal.py
def seal(candidate: dict):
    """Returns (verdict, reasons). PASS requires all four fields ..."""
```

Required keys (`:30`): `tabula_drift`, `observed_controls`, `stated_test_width`, `known_blind_spots`. Empty list/dict/str ≡ missing → `REFUSED`. If no control has `watched_to_fail: True` → `REFUSED`. If `tabula_drift.instrument_validated` is false → `PASS_WITH_WARNINGS`.

CLI: `./tools/doctor_seal.py --self-test [--out PATH]`. Schema `hawking.nos.doctor_seal.v1`.

This is what NOS stage 3 calls. **It never runs a model.**

### 5b. Fast / tensor gate — `tools/gravity_doctor_gate.py`

```183:201:tools/gravity_doctor_gate.py
def gate(W, Wh, X, ref=None, seed=None):
```

Inputs: numpy `W` (reference), `Wh` (candidate), `X` (activation rows). Axes: `observed` (X@W vs X@Wh cosine), `probed` (isotropic), `worst_unit`, `gain`. Relative mode vs honest Q4 g128 reference + `AXIS_MARGIN`.

**Fast path (no weights):**

```
python3 tools/gravity_doctor_gate.py --demo
```

Synthetic 256×64; must HEALTHY on faithful Q4, UNHEALTHY on visible-subspace cheat.

**Full path (Qwen3.8-27B only):**

```
python3 tools/gravity_doctor_gate.py
python3 tools/gravity_doctor_gate.py --tensor language_model.model.layers.0.mlp.gate_proj.weight --layer 0 --json out.json
```

Hard-coded roots (`:32–33`): `workspace/campaign/records/runs/qwen38-27b/{bf16,activation-capture-v1}`. Loads safetensors + capture `X`. **Not a live runtime. Not logits. Not MLX.**

### 5c. Capability battery — `tools/gravity_doctor_capability.py`

```84:101:tools/gravity_doctor_capability.py
def score(root, max_new=260, max_seq=768):
```

```
python3 tools/gravity_doctor_capability.py \
  --artifact SOME_DIR_NAME \
  --control-pass uniform-q4-v1 \
  --control-fail mixed-q4down-v1 \
  [--max-new-tokens 260] [--max-seq-len 768] [--json out.json]
```

Drives **only** `workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy` with `--artifact-root workspace/campaign/records/runs/qwen38-27b/<name>`. 10 prompts, 7 dims (factual/arithmetic/code/tool/instruction/multilingual/reasoning). Degeneracy = `<3` non-EOT tokens. Controls must separate or **no verdict**.

**Hawking NX artifact only.**

### 5d. Six dimensions — `tools/gravity_doctor_dimensions.py` (G076)

```
./tools/gravity_doctor_dimensions.py \
  --artifact uniform-q4-v1 --artifact compact-q3attn-r1p2-v1 \
  [--known-bad mixed-q4down-v1] [--n-per-dim 12] [--max-new-tokens 320] \
  [--seed 20260818] [--out PATH]
```

Dims (never averaged): FACTUAL, REASONING, PROCEDURAL, LANGUAGE, TOOL, IDENTITY. Same greedy binary under `gpu_lane_lock.sh`, same `qwen38-27b/<artifact>` tree. Scores text after `</think>`; unclosed think = truncated, not wrong. Schema `hawking.nos.doctor_dimensions.v1`.

### 5e. “Fast doctor” in campaign practice

| Intent | Command | Runtime |
|---|---|---|
| Structural seal | `doctor_seal.seal(dict)` / `--self-test` | none |
| Tensor health | `gravity_doctor_gate.py --demo` or `--tensor/--layer` | numpy + on-disk W/X |
| Two-needle coherence | `genesis_nos.doctor(artifact)` (`tools/genesis_nos.py:39–58`) | hawking greedy, France/Paris + 17×19 |
| External stock model | **no doctor tool**; `workspace/campaign/odyssey/a3b_recon.py` 12-item mlx battery | mlx_lm |

**Can it grade an external MLX / transformers run?** No existing Doctor entrypoint accepts logits, an mlx handle, or a transformers `generate()` transcript. Capability/dimensions are wired to `ascension_qwen38_hybrid_greedy` + a Hawking artifact directory. To grade Odyssey patients A3B/GLM/Falcon you either (a) wrap their text in a new scorer that reuses `judge()` / BATTERY predicates, or (b) pack a Hawking NX artifact first.

---

## 6. RUNTIME / ARCH SUPPORT — `crates/hawking-core/src/model/*`

### Shipping GGUF dispatch — `load_engine` (`mod.rs:80–172`)

```80:80:crates/hawking-core/src/model/mod.rs
pub fn load_engine(weights: &Path, mut config: EngineConfig) -> Result<Box<dyn Engine>> {
```

Used by `hawking generate` (`crates/hawking/src/main.rs:4185`) and `hawking serve`.

Order:

1. Memory budget (`memory_limit_mb`, `0` = 80% RAM).
2. If file is `.gravity` / activation-aware → `GravityEngine::load` (Metal-only on macOS).
3. Else GGUF `general.architecture`:

| GGUF arch string | Engine | Status |
|---|---|---|
| `llama` **and** Mixtral expert tensors | `MixtralEngine` | **In-tree and dispatched** (`mod.rs:143–145`, `mixtral.rs:1552`) despite comments claiming extraction |
| `llama`, `llama2`, `llama3`, `llama3.1`, `llama3.2`, `mistral` | `LlamaDense` | ships |
| `deepseek2`, `deepseek-v2`, `deepseek2-lite` | `DeepSeekV2` | ships (V2, not V4) |
| `qwen2`, `qwen2.5`, `qwen` | `QwenDense` | ships |
| `qwen2moe`, `qwen3moe`, `qwen-moe` | `QwenMoE` | **match arm exists; `Engine::load` is `Unimplemented`** (`qwen_moe.rs:298–302`) |
| `rwkv7`, `rwkv-7` | `RwkvSeven` | ships |
| anything else | error | names extra pack |

Error text (`:169–170`):

```
unknown architecture {other:?}; the shipping engine supports llama + deepseek2 + qwen2
 + qwen-moe + rwkv7. gemma2/phi3/olmoe/mamba2/mixtral are in the hawking-adapters-extra pack
```

That error is **stale on Mixtral** (still in `load_engine`) and **optimistic on qwen-moe** (dispatch yes, forward no).

### Gravity artifact dispatch (`gravity_engine.rs:411–451`)

`architecture.model_type`:

- `llama` / `mistral` / `qwen2` → `GravityLlamaGpu`
- `glm_moe_dsa` → `GravityGlmGpu` (multi-shard dir)
- `deepseek2` / `deepseek_v2` → `GravityDeepSeek`
- `mixtral` → `MixtralEngine` (generation on Gravity path is `Unimplemented`)

No `deepseek_v4`, no Falcon-H1, no Gemma, no Mamba.

### Qwen3 / A3B / DSV4F (campaign runtimes, not `load_engine`)

These are **example binaries**, not GGUF engines:

- Qwen3.8-27B hybrid: `qwen38_hybrid_decode.rs` + `ascension_qwen38_hybrid_greedy`
- Qwen3-Coder-30B complete-binary: `qwen30_complete_runtime.rs` (artifact runtime, comment at `mod.rs:13–15`)
- Qwen3-Coder-Next 80B: `qwen80_*`
- DSV4F: `dsv4f_activation_capture.rs` + example `gravity_deepseek_v4_native_token_graph` — capture/token-graph, not `load_engine`

`qwen_moe.rs` documents A3B topology only (`:30–35`): 128 experts, top-8, `Qwen3MoeForCausalLM`. Comment `:4–10`: “complete Qwen decoder remains unavailable… component primitives only.”

### Asked families

| Family | In shipping `load_engine`? |
|---|---|
| Gemma / Gemma-3 | **No.** Adapter registry `gemma` is DECLARED, `executes: false`, pack-only (`families.rs:687–697`). `gemma2_smoke.rs` expects `model_arch()=="gemma2"` and will fail if a GGUF is present. Soft-cap kernel `mha_decode_step_gemma` exists in `attn.rs` but is not wired to a loader. |
| Falcon-H1 | **No.** Only vendor `convert_hf_to_gguf.py` + download log `workspace/campaign/odyssey/downloads/O001_falcon-h1-7b.log`. |
| Mamba-SSM / Mamba2 | **No** in dispatch. `state_space` family aliases include `mamba2` but module is `rwkv7.rs`. `mamba2_smoke.rs` calls `load_engine` and asserts `model_arch()=="mamba2"` — that path is dead without the extra pack. |
| Generic MoE | **No.** Mixtral-shaped GGUF yes; Qwen3-MoE load stub; GLM only as `.gravity` `glm_moe_dsa`. |
| DeepSeek-V4 | **Not** GGUF. Separate DSV4F campaign runtime. |
| Qwen family | Dense GGUF yes. Qwen3.8/Q30/Q80 are **artifact** runtimes. A3B complete decode **no**. |

Adapter registry families (`families.rs:595–727`, 10 ids): `llama`, `mistral_mixtral`, `qwen`, `glm`, `deepseek` (V2), `kimi` (no engine), `minimax` (declared), `gemma` (pack), `phi` (pack), `state_space` (RWKV7 executes).

### CLI entry

```
hawking generate --weights <gguf|.gravity> --prompt "..." [--max-new-tokens 256] [--max-seq-len 4096] [--memory-limit-mb N]
```

(`crates/hawking/src/main.rs:417–429`). Architecture is not a flag; `load_engine` peeks the file.

`dispatch.rs` is **not** arch dispatch — it is RMSNorm/GEMV Metal-vs-CPU helpers.

---

## 7. BASELINE TPS / token_ns

Two stacked authorities. Do not mix them.

### A. External stock-model specimen — **mlx_lm**, not transformers

Exact script: `workspace/campaign/odyssey/a3b_recon.py`.

```46:48:workspace/campaign/odyssey/a3b_recon.py
generate(model,tok,prompt="Hi",max_tokens=4,verbose=False)
t=time.time(); long=generate(model,tok,prompt="Explain step by step how photosynthesis works.",max_tokens=64,verbose=False); dt=time.time()-t
tps=64/dt
```

- Runtime: `from mlx_lm import load, generate` + `mlx.core`.
- Weights: `workspace/campaign/records/runs/qwen3-30b-a3b/bf16`.
- Formula: **64 / wall_seconds** of one `generate(..., max_tokens=64)` after a 4-token warmup. Host wall, includes prompt processing. Rounded to 1 decimal.
- Receipt: `receipts/ascent-2026-08-18/A3B_RECON.json` → `"tps_specimen": 29.3` with explicit caveat: “SPECIMEN mlx_lm… indicative, not full corpus”; “29.3 is the mlx SPECIMEN”.

Sister script `workspace/campaign/odyssey/glm_recon.py` is the same pattern (`tps=48/dt`).

`tools/ascent/matvec_mlx_reference.py` is **kernel GB/s**, not model TPS.

Transformers is used in `workspace/campaign/moe_arch_map.md` for **static config archaeology**, not TPS.

### B. Native Hawking token_ns (Qwen3.8 NX, not stock A3B)

| Surface | How |
|---|---|
| Greedy binary | `ascension_qwen38_hybrid_greedy` prints `STEADY_DECODE_WALL_NS_PER_TOKEN` / `WALL_NS` (`lab/genesis_pool.py:67–70` parses them). |
| Sealed receipt | `receipts/ascent-2026-08-18/GROUND_TRUTH_TPS.json` — uniform-q4-v1: `steady_decode_wall_ns_per_token: <REDACTED> → **33.10 complete-wall TPS**; GPU-only 34.44. Lane-clean, 128 new tokens. |
| Ledger | schema `hawking.ascension.qwen38_token_ns_ledger.v1` (`qwen38_token_ns_ledger.rs:25`); GPU time = `MTLCommandBuffer GPUEndTime−GPUStartTime`. |
| Legal BASE_TRUE_TPS | `tools/agentos/agentos.py:60–66` `base_true_tps(steps_us) = n / sum(us)/1e6` **including drain**. `base_true_tps_ok()` refuses unless `clean_box_ok()`. |
| Llama scorecard | `tools/llama_tps_contract.py` — **does not run an engine**; validates signed JSON receipts (`decode_tps`, …). |

A3B 29.3 is **not** a Hawking native number and is **not** `BASE_TRUE_TPS`.

---

## 8. EXISTING ODYSSEY CONTROL

There are **three** “Odyssey” layers and **no** model-patient queue.

### A. Training-data Odyssey — `tools/odyssey/` (OLD, still live)

Package docstring (`__init__.py:1–10`): inventory, membership, ingest, contamination barrier. **No downloads, no training.**

CLI (`cli.py:192–221`):

```
python3 -m tools.odyssey.cli inventory [--out]
python3 -m tools.odyssey.cli membership-check
python3 -m tools.odyssey.cli ingest-fixture [--raw] [--corpus-id] [--out-dir]
python3 -m tools.odyssey.cli barrier-report [--out]
python3 -m tools.odyssey.cli teacher
python3 -m tools.odyssey.cli all
```

Schemas: `hawking.odyssey.data_inventory.v1`, `.membership_record.v1`, `.contamination_barrier.v1`, plus T0 `hawking.odyssey.t0.v1` (`t0_run.py`), tournament `hawking.odyssey.checkpoint_tournament.v1` (math-profile + support-halo, **training checkpoints**, not NX patients).

Source files present: `ingest, dedup, contamination, inventory, membership, normalize, data_verify, feasibility, hidden_memberships, known_failures, runtime_authority, substrate_*, teacher_assess, tournament, contracts, t0_run, cli`. `__pycache__` still has deleted modules (`scheduler, apparatus, trainer, qat, trajectory, …`) — **not on disk as `.py`**.

Governance tree: `workspace/campaign/governance/odyssey/` (fence, T0 receipts, eval contracts, checkpoints). This is the **training launch** program.

Ramanujan twin: `ramanujan/scaffold/research/odyssey.py` — fixture-only control plane, `SCHEMA = hawking.ramanujan.odyssey_fixture.v1`, never promotes.

### B. Odyssey-I recon (NEW, this campaign) — `workspace/campaign/odyssey/`

| Path | What |
|---|---|
| `a3b_recon.py` | mlx A3B baseline + route map → `A3B_RECON.json` |
| `glm_recon.py` | mlx GLM-4.5-Air same |
| `contracts/reuse_surface.md` | this recon contract |
| `contracts/arch_archaeology.md` | patients O000 Gemma-3-1B, O001 Falcon-H1-7B, O005 Qwen3-30B-A3B, O010 GLM-4.5-Air |
| `downloads/O001_*.log`, `O005_*.log` | HF download logs |
| `PROVENANCE.json` | provenance stub |

`workspace/campaign/moe_arch_map.md` is the static MoE map (mlx_lm + transformers configs, no weights).

### C. Patient-queue / packet / transfer-matrix / gravity-rulebase?

**No patient-queue. No patient-packet type. No gravity-rulebase module.** Grep for those names hits only this contract.

What *does* exist and is easy to confuse:

| Artifact | Schema | Role |
|---|---|---|
| `receipts/ascent-2026-08-18/PATIENT_IDENTITY_VECTOR.json` | `hawking.nos.patient_identity_vector.v1` | **One** Qwen3.8 artifact identity (capability/tabula/tools). Not a queue. |
| `WIDE_BATTERY_PATIENT.json`, `TABULA_PATIENT.json` | — | Same patient, inputs to `genesis_nos.g124_seal` |
| `lab/operators/ascension_dual_gravity_worker.py` `_update_transfer_matrix` | `hawking.ascension.transfer_matrix.v1` | Per-organ representation transfer **inside Qwen gravity search** |
| `tools/foundry/post_parent_review.py:update_cross_parent_matrix` | `hawking.foundry.cross_parent_transfer_matrix.v1` | Cross-parent assumption harvest (gpt-oss / qwen3-235b) |
| `tools/foundry/NEGATIVE_TRANSFER_ATLAS.json` | — | Dead-lever atlas |
| `tools/foundry/GRAVITY_METHOD_REGISTRY.json` | — | Method priors + doctor method names `doctor_static` / `doctor_conditional` |
| `receipts/ascent-2026-08-16/G002_TRANSFER_MATRIX.json` | `hawking.ascension.g002_transfer_matrix.v1` | Ascent-era transfer receipt |
| `receipts/ascent-2026-08-16/ASCENT_STATE.json` + `PROMOTION_QUEUE.json` | ascent controller/daemon | Experiment lanes, not patients |
| `tools/campaign/knowledge_plane.py` | mentions transfer-matrix hash binding | Campaign knowledge, not a patient scheduler |

### `receipts/` vs Odyssey-I

No `receipts/odyssey-i/` tree. Relevant receipts are `receipts/ascent-2026-08-18/{A3B_RECON,GLM_RECON,PATIENT_IDENTITY_VECTOR,…}` and the old training records under `workspace/campaign/governance/odyssey/records/`.

---

## REUSE VERDICT

| component | reuse as-is | extend | build-new | note |
|---|---|---|---|---|
| `worker_gate.observe/gate` | yes, for ~15 GB `ascension_*` processes | retune `PROC` + coefficients if the patient binary is not Qwen3.8 hybrid | — | **No Metal headroom.** Call before process spawn. Only NOS imports it today. |
| `agentos.machine_state` | yes, as clean-box / disk / load snapshot | — | — | Not a worker admit. `clean_box_ok` blocks on **any** live grok worktree. |
| `genesis_pool._refuse_if_full` / `qwen38_host_admission` | pattern only | generalize costs | Odyssey admit if patients are multi-process Metal copies | These are the real Metal/swap gates. |
| `genesis_capacity.admit` | no (Qwen3.8 session bytes) | — | per-patient working-set table | 15.12 GB body is Qwen3.8-specific. |
| `reclaim_safe.sh` | **yes** | — | — | No `--disk-floor`. Pair with 15 GiB floor in the controller. Preserve reports. |
| `ascent_controller.py` loop | yes as queue skeleton | new STATE path + target schema + value() | — | Not G001-hardwired. Does not form contracts. |
| `ascent_daemon.py` | no | cherry-pick `harvest` / `govern` / `launch_target` | Odyssey patient controller | Hard-wired `ACTIVE_MODELS={"qwen38"}`, Genesis resident, uniform-q4 contracts. |
| grok-run launch argv | yes | — | — | `delegate --task --contract --repo --profile gate`. |
| `nos_pipeline.qualify_and_promote` | after you can fill the candidate dict | generalize `PROC` witnesses + metrics | — | CLI is selftest-only. Promote/rebind is Genesis succession. |
| `doctor_seal.seal` | yes | — | — | Structural. Reuse for every patient seal. |
| `gravity_doctor_gate` | only on Hawking W/X tensors | new loaders for foreign safetensors + captures | — | `--demo` is the fast self-check. |
| `gravity_doctor_capability` / `_dimensions` / `wide_battery` | no, for external models | extract `judge()` / BATTERY / dim builders | mlx/transformers runner + same predicates | Binaries and artifact roots are Qwen3.8 NX only. |
| `genesis_nos.doctor` | no (two-needle on greedy NX) | — | per-patient coherence script (A3B already has mlx battery) | |
| `load_engine` | yes for Llama/Qwen-dense/DSV2/RWKV7/Mixtral-GGUF / `.gravity` llama/glm/dsv2 | — | **Falcon-H1, Gemma-3, Mamba, Qwen3-MoE complete, DSV4, GLM-HF** | `qwen3moe` match arm raises Unimplemented. |
| mlx A3B/GLM recon | **yes** as external TPS+route specimen | parameterize patient root / token count | — | This is how 29.3 was produced. Not BASE_TRUE_TPS. |
| `agentos.base_true_tps` | yes, for native Hawking timings | — | — | Refuse unless `clean_box_ok`. |
| `tools/odyssey/` | yes for **train/eval contamination** if you ingest corpora | — | — | Not a model-patient control plane. |
| patient queue / packet / rulebase | — | — | **build-new** | Does not exist. Closest analog is ascent target JSON + foundry/ascension transfer matrices (different object). |
| `PATIENT_IDENTITY_VECTOR` | schema inspiration | generalize dimensions | per-patient identity writer | Today one Qwen3.8 artifact document. |
| foundry transfer matrix / negative atlas | read as priors | — | Odyssey transfer matrix if you want cross-patient lever transfer | Not a scheduler. |

**Bottom line:** reuse the **governors** (`worker_gate`, `machine_state`, `reclaim_safe`, `doctor_seal`, mlx recon, controller loop). Treat `ascent_daemon` and all Qwen3.8 greedy/Doctor batteries as **Genesis-specific**. Odyssey-I still needs a new patient-queue schema, per-arch runtimes (A3B/Falcon/Gemma/Mamba are not in `load_engine`), and an external Doctor runner.

---

**Completion report**

- Scope: read-only recon of the eight reuse surfaces; no edits.
- Deliverable: this report (signatures, `file:line`, CLI, schemas, reuse table).
- Uncertainty: Mixtral comments vs `load_engine` match arm disagree — **code wins** (Mixtral still dispatched). Extra-pack Gemma/Mamba/Phi were not opened. `tools/odyssey` pycache lists modules whose `.py` is gone.
