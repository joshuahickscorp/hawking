# HIDE Durable and Proactive Agent Spec

Run date: 2026-07-19
Grounding: `HIDE_LIVE_ARCHAEOLOGY.md` §3.4, §3.5, §5; `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json` (ids `session.background_supervisor`, `teams.coordinated`, `goal.evaluator_loop`, `session.durable_transcript`, `session.checkpoint_rewind`, `subagents.fork_worker`); Bible Book XII (durable and proactive agents). Sealed-backend evidence is pinned at git `5a99d0e2` (the only lifeline; the offline archive is gone from disk, `HIDE_LIVE_ARCHAEOLOGY.md` §2b).
Readiness key (per Bible discipline): **real-and-wired** / **real-but-unwired (packed)** / **partial** / **stub** / **missing**.

## 1. The one-sentence honest status

HIDE has a real, event-sourced, crash-recoverable job fabric and a real runtime supervisor, both packed at `5a99d0e2` and **unwired from any live path**, plus a digest projection and an interrupt hub. What it does not have is a detached session-supervisor daemon, the rich trigger set, quiet-hours or notification-threshold policy, and warm-state (not just text) durability. Durable and proactive agents are therefore a **reintegration plus a thin trigger/notify layer on proven parts**, not a greenfield build. Every claim below is labeled by the readiness of the primitive it rests on.

Separation held throughout: **PARITY** = reproduce Claude Code's background supervisor, teams, and goal loop; **SUPREMACY** = what a resident local runtime does better, each gated on a named build item.

## 2. What exists today (verified substrate)

All rows are packed at `5a99d0e2`, recoverable via `git checkout 5a99d0e2 -- crates`, absent from the active tree (`4fbca8bc`), and **not HTTP-reachable** (`HIDE_LIVE_ARCHAEOLOGY.md` §3.5, parity `session.background_supervisor.hide_evidence`).

| Primitive | Role | Readiness | Evidence (5a99d0e2) |
|---|---|---|---|
| `RuntimeSupervisor` | keep the `hawking serve` child alive: boot, healthz poll, restart backoff ladder + per-window cap, `runtime.lock` | real-but-unwired | `hide-backend/supervisor.rs:59,84,154` |
| `BackendReplayService` | rebuild / scrub / fork a session by folding the durable event log | real-but-unwired | `hide-backend/replay.rs:14,41,92,127` |
| `InterruptHub` | per-run interrupt mailbox (pause/steer/abort, abort supersedes) drained into the kernel | real-but-unwired | `hide-backend/interrupt.rs:26,33,63` |
| `SessionRegistry` | stable open-or-create session id recorded in durable KV, survives workspace reopen | real-but-unwired | `hide-backend/services.rs:32,43` |
| `BackendServices` (Project Brain) | SQLite-on-disk long-term memory (decisions, test results, failed approaches) | real-but-unwired | `hide-backend/services.rs:73` |
| digest projection | verified deliverables to `home`/`sessions` FE patches; streak, activity heatmap, branch, worktrees | real-but-unwired | `hide-backend/digest.rs:31,142,254` |
| `AgentJob` + `JobGraph` | the durable job record; graph is a fold of the event log, rebuilt on startup (crash recovery) | real-but-unwired | `hide-fleet/queue.rs:198,315,491` |
| `JobSchedule` + `ScheduleGate` | earliest-start + gate conditions that must all hold before a batch fires | real-but-unwired | `hide-fleet/queue.rs:180,189` |
| `FleetGovernor` | resource admission, model-slot ceiling, spawn-rate circuit breaker, thermal-aware | real-but-unwired | `hide-fleet/scheduler.rs:110,176` |
| `WorktreeManager` / `PortAllocator` | per-run worktree + port isolation leases | real-but-unwired | `hide-fleet/isolate.rs:63,148` |
| merge funnel | footprint-overlap plan, tournament select, three-way merge, integration result | real-but-unwired | `hide-fleet/merge.rs:61,146,237,451` |
| `FleetView` | machine-wide live counters + governor banner, projected from `job.*`/`governor.*` events | real-but-unwired | `hide-fleet/fleetview.rs:31,49` |
| ACP remote | JSON-RPC 2.0 over persistent WebSocket, durable server-side sessions independent of the socket | real-but-unwired | `hide-fleet/remote.rs:4,12` |
| FE FleetView / StateTimeline / Digest | dashboard, scrub/fork, digest surfaces (optimistic, mock-fed, "plan 2") | ui_only | `HIDE_LIVE_ARCHAEOLOGY.md` §3.4 |
| detached session-supervisor daemon (roster/jobs on disk) | host detached sessions across terminal close | missing | no such daemon at either commit |
| goal evaluator loop | separate transcript-only judge with reason feedback | missing | parity `goal.evaluator_loop.hide_status = absent` |

**Load-bearing distinction:** `BackendReplayService::fork_session` (`replay.rs:127`) forks the **event log** (the durable text projection), re-appending the prefix under a fresh `SessionId`. It does **not** fork the warm KV/recurrent capsule. Warm-state fork is the separate moat specified in `HIDE_STATE_CAPSULE_ABI.md`, and it is gated on GPU readback + serve state routes. Background-agent durability is text-durable today; warm-state durability is a further gate. Do not conflate them.

## 3. The DurableGoalBinding

A durable agent is a goal **bound** to everything needed to resume it deterministically after any interruption. The packed `AgentJob` (`hide-fleet/queue.rs:198`) already carries most of this contract; the binding below names each field, its packed backing, and the gaps.

```text
DurableGoalBinding {
  repository        // base_ref: the commit each worktree forks from      [AgentJob.base_ref]
  branch/worktree   // isolation: Worktree|Overlay|Container|None          [AgentJob.isolation, isolate.rs]
  session           // session_id: stable, KV-recorded, reopen-recovered   [AgentJob.session_id, services.rs:43]
  state_capsule     // warm KV/recurrent handle at a committed boundary     [MISSING -> HIDE_STATE_CAPSULE_ABI]
  plan              // objective + pattern + profile + max_steps            [JobSpec, queue.rs:157]
  permissions       // trust domain + tool allowlist for unattended run     [partial -> HIDE_SECURITY_CONSTITUTION]
  budget            // resource_hint (memory/gpu/slots) + max_steps         [ResourceRequest, queue.rs:119]
  schedule          // earliest_start_ms + gates                           [JobSchedule, queue.rs:180]
  triggers          // gate set that fires the run (Section 4)             [partial -> ScheduleGate, queue.rs:189]
  checkpoints       // event-log prefix (text) + capsule ref (warm)        [text real-but-unwired; warm MISSING]
  verification      // acceptance oracle + retry/stall policy              [attempts/max_attempts real; oracle -> kernel]
  notifications     // thresholds, quiet hours, priority, dedup            [MISSING policy -> Section 6]
}
```

Two fields are the honest weak points. `state_capsule` is `None` for every non-RWKV session (`HIDE_STATE_CAPSULE_ABI.md` §6), so a durable binding today resumes from the **text prefix**, not warm state, for transformer models. `permissions` for an unattended run must bind to a trust domain and an explicit tool allowlist; the packed `hide-security` logic is real but its OS enforcement is a seam (`HIDE_LIVE_ARCHAEOLOGY.md` §3.5). Unattended autonomy is gated on both, not asserted.

**Identity binding rule (inherited from `HIDE_STATE_CAPSULE_ABI.md` §4):** a resumed binding MUST verify its capsule's `IdentityBinding` (weights, arch, tokenizer, engine build, security domain) and refuse on mismatch. A goal bound across a model swap is a correctness hazard, not a silent degrade. `AgentJob.schema_version` (`queue.rs:198`) is the migration hook for the binding record itself.

## 4. Triggers

A durable agent fires on a gate condition. The packed `ScheduleGate` enum (`hide-fleet/queue.rs:189`) is the **whole** trigger vocabulary today; the parity target requires a richer set that is a straightforward extension of the enum plus an event-source/watcher layer that emits `job.*` events. Manual firing maps to `CreatedBy::User`; scheduled firing to `CreatedBy::Schedule` (`queue.rs:42`).

| Trigger | Parity intent | Readiness | Backing / gap |
|---|---|---|---|
| time / cron | run at a wall-clock instant or schedule | real-but-unwired | `ScheduleGate::Cron` + `earliest_start_ms` (`queue.rs:180,189`) |
| idle | run only when the machine is idle | real-but-unwired | `ScheduleGate::Idle` (`queue.rs:189`) |
| AC power | run only on wall power | real-but-unwired | `ScheduleGate::AcPower` (`queue.rs:189`) |
| thermal-ok | defer under thermal pressure | real-but-unwired | `ScheduleGate::ThermalOk` + `resources.rs:19,61` |
| manual | user launches explicitly | real-but-unwired | `CreatedBy::User` (`queue.rs:42`) |
| git-push | fire on a push to a watched ref | missing | add gate + git watcher; `base_ref` already models the ref |
| PR opened / updated | fire on PR event | missing | add gate + forge webhook or poll |
| issue opened / labeled | fire on issue event | missing | add gate + forge webhook or poll |
| CI failure | fire on a failing pipeline | missing | add gate + CI status source |
| file-change | fire on a workspace path change | missing | add gate + fs watcher emitting `workspace.*` |
| dependency-advisory | fire on a new advisory for a locked dep | missing | add gate + advisory feed |
| monitoring-alert | fire on an external alert | missing | add gate + local alert intake |

**Build item T-TRIG-1:** extend `ScheduleGate` with the event-driven variants and add a watcher layer that translates git/forge/CI/fs/advisory/alert events into gate-satisfaction and `job.enqueued` events on the durable log. The gate-all-hold semantics (`JobSchedule.gates`, `queue.rs:180`) already compose the safety gates (idle/AC/thermal) with the event gates, so "run on push, but only when idle and on AC" is expressible without new control flow.

## 5. Recovery survival matrix

Durability means the goal survives every interruption and resumes from disk, never re-launching completed work. The load-bearing invariant is already coded: `JobGraph::project_from` folds the event log to rebuild the graph on startup and **"folding records data, it never re-launches a run"** (`queue.rs:491`, comment T3). Recovery is a replay of durable events, not a re-execution.

| Failure mode | Survives via | Readiness | Evidence |
|---|---|---|---|
| IDE close (surface gone) | session core is authoritative, not the surface; `SessionRegistry` reopen-recovers the stable id | real-but-unwired | `services.rs:43`; `HIDE_TWO_SURFACE_ARCHITECTURE.md` §1 |
| terminal close (chat surface gone) | same session core; both surfaces are thin clients | real-but-unwired | `HIDE_TWO_SURFACE_ARCHITECTURE.md` §2 |
| model restart / serve crash | `RuntimeSupervisor` restart backoff ladder + per-window cap, healthz-gated | real-but-unwired | `supervisor.rs:59,154` |
| backend restart | `JobGraph::project_from` rebuilds the job graph from the log; `BackendReplayService::rebuild_session` rebuilds session projections | real-but-unwired | `queue.rs:491`; `replay.rs:41` |
| reboot | `AgentJob` is serde with `schema_version`; graph + sessions replay from the durable log/KV | real-but-unwired | `queue.rs:198,491`; `services.rs:29` |
| network loss | ACP durable sessions persist server-side independent of the socket | real-but-unwired | `remote.rs:4,12` |
| tool failure | `attempts`/`max_attempts` retry; `JobStatus::Failed` is terminal and recorded | real-but-unwired | `queue.rs:87,198` |
| detached-across-terminal supervision | no per-user session-supervisor daemon hosting detached jobs | missing | build item R-SUP-1 |
| warm-state resume (no re-prefill) | capsule persistence behind the goal binding | missing | `HIDE_STATE_CAPSULE_ABI.md` §8 (G-CAP-1) |

**What is real vs what remains:** the event-sourced spine gives text-durable survival across restart, reboot, and network loss **once wired to a live path**. Two gaps keep it from Claude Code's documented `claude agents` daemon: R-SUP-1, a per-user supervisor that hosts detached jobs surviving terminal close and sleep (the packed `RuntimeSupervisor` supervises the serve **child**, not detached sessions), and warm-state persistence so resume skips re-prefill (`HIDE_STATE_CAPSULE_ABI.md` build items). Until R-SUP-1 lands, "background sessions survive terminal close" is INFERRED from the durable log, not demonstrated end-to-end.

## 6. Quiet proactive behavior

A proactive agent must earn attention, not spend it. Bible Book XII and the dossier are explicit: "a calm activity projection that does not equate busyness with progress" and "make the overnight digest a projection of verified deliverables" (frontier dossier §4.7, Phase 4 step 8). The FE already enforces the no-metering doctrine and hides the token meter in the Digest (`HIDE_LIVE_ARCHAEOLOGY.md` §3.4).

| Behavior | Definition | Readiness | Backing / gap |
|---|---|---|---|
| digest | periodic roll-up of **verified deliverables**, not activity | real-but-unwired | `digest.rs:31` computes `home`/`sessions` patches |
| priority | which agent may preempt for a model slot | real-but-unwired | `PriorityClass` Interactive>High>Normal>Batch>Idle (`queue.rs:75`) |
| attention budget | admission ceiling + spawn-rate breaker bound concurrent work | real-but-unwired | `FleetGovernor::can_admit`, `BreakerState` (`scheduler.rs:110,176`) |
| dedup | overlapping-footprint work is serialized / suppressed | partial | `Footprint::overlaps` + `plan_footprints` (`merge.rs:43,61`); semantic duplicate-detection is a build item |
| notification thresholds | only surface events above a salience bar (blocker, approval, conflict, done) | missing | policy layer over the event log (T-NOTIF-1) |
| quiet hours | suppress non-urgent notification in a user window | missing | no quiet-hours primitive; extend the gate/notify layer (T-NOTIF-1) |

**Build item T-NOTIF-1:** a notification policy that folds the durable event log into an attention inbox (blockers, approvals, conflicts, completed deliverables, frontier dossier §4.7), applies a salience threshold and per-user quiet hours, deduplicates by footprint and by session, and delivers push only above the bar. This is a projection layer, not new autonomy: it reads the same `job.*`/`workspace.*` events the `FleetView` already folds (`fleetview.rs:49`). The digest is its low-frequency, verified-deliverables-only sibling.

## 7. The background dashboard

Parity `session.background_supervisor` requires "a one-screen dashboard grouped by state (working/needs-input/idle/done/failed) with peek/reply/attach, per-session worktree isolation, conditional auto-PR." The packed pieces map cleanly.

**State groups.** `JobStatus` (`queue.rs:87`) provides Queued/Admitted/Running/Paused/Preempted/Merging/Done/Failed/Cancelled with `is_terminal`/`is_live` helpers; the dashboard groups **working** = live statuses, **needs-input** = a run with a pending `InterruptHub` reply or a permission gate, **idle** = queued/paused, **done**/**failed** = terminal. `FleetView` (`fleetview.rs:31,49`) already projects machine-wide live counters plus a governor banner from the event log, and `RunRow` (`fleetview.rs:16`) carries the worktree path and leased ports per run.

**Peek / reply / attach.** Peek is a read-only projection (`FleetView`, `BackendReplayService::ui_events` at `replay.rs:77`). Reply and steer route through `InterruptHub::signal` (pause/steer/abort, `interrupt.rs:33`), drained into the live kernel between transitions (`interrupt.rs:63`); abort supersedes a buffered pause (last-write-wins). Attach re-fronts a durable session by its stable id (`services.rs:43`) in either surface (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §4-5). Readiness: real-but-unwired (the FE surfaces are `ui_only`, optimistic, mock-fed until the `/v1/hide/*` boundary is restored, `HIDE_LIVE_ARCHAEOLOGY.md` §3.4).

**Worktree isolation.** Each run gets a `WorktreeLease` + `PortLease` from `WorktreeManager`/`PortAllocator` (`isolate.rs:63,148`), forking from `AgentJob.base_ref`. Write-heavy workers are isolated; read-heavy workers (`ConcurrencyClass::CpuOnly`, `queue.rs:65`) run wide without a model slot.

**Conditional auto-PR (hard safety invariants).** The merge funnel is packed (footprint plan, tournament select, three-way merge, `merge.rs:61,146,237`), but the auto-PR **policy guard is a build item and MUST encode these invariants as code, not convention:**

- never commit or push to `main` (or any protected ref);
- never force-push;
- only ever push to the run's own isolated worktree branch;
- open a **draft** PR, never auto-merge;
- refuse to act outside the run's `base_ref` lineage.

This mirrors Claude Code's documented supervisor behavior (auto worktree commit/push/draft-PR, never main, never force-push; parity `session.background_supervisor`). Readiness: merge machinery real-but-unwired; the never-main/never-force-push guard is **missing** and is build item R-PR-1. Autonomy ships **after** the guard and after `hide-security` OS enforcement, per the phase-zero security gate (frontier dossier §5.9).

## 8. Parity: local always-on machine vs Claude Code cloud routines

Claude Code / Anthropic runs scheduled routines and background work on **managed cloud infrastructure**: ephemeral VMs, metered per run, source and context egress to the provider, sandbox state wiped between runs. HIDE runs the identical durable-goal machinery on the **user's own always-on machine**. The structural contrasts (each a consequence of local residency, not a new feature):

| Dimension | Claude Code cloud routines | HIDE local routines |
|---|---|---|
| per-run cost | metered VM compute per invocation | none beyond electricity; local hardware |
| egress | source + context leave the machine | no egress; state stays on-device |
| between-run state | ephemeral sandbox wiped | persistent worktrees, event log, Project Brain survive (`services.rs:73`, `queue.rs:491`) |
| resume | re-established from transcript | text-durable now; warm-state resume gated (`HIDE_STATE_CAPSULE_ABI.md`) |
| parallelism ceiling | metered quota (10 parallel warned as 10x quota) | bounded only by local cores/RAM/thermal (`FleetGovernor`, `scheduler.rs:110`) |

This is a **parity** claim on behavior (a routine fires on a schedule, does work in isolation, reports deliverables) and a **structural** claim on economics/privacy that needs no experiment: it follows from the routine running on hardware the user already owns. It becomes real on a live path only after the reintegration in Section 10.

## 9. Supremacy (gated on named build items)

Each claim is separated from parity and gated on the specific item it needs; none is asserted as shipping.

1. **Agent teams become a default, not a rationed premium.** Claude Code's experimental teams cost **~7x tokens** (parity `teams.coordinated`, evidence DOCUMENTED, MEASURED ~7x). With resident warm forks, N teammates share one warmed prefix and the marginal model cost collapses toward one, because a fork is a pointer-copy of a resident capsule (`HIDE_STATE_CAPSULE_ABI.md` §7). **Gated on:** state-capsule exposure (serve `/v1/hawking/state/fork` + session-slot affinity, `HIDE_STATE_CAPSULE_ABI.md` §8), and on the RWKV lane first (transformer/`Hybrid` capsule unbuilt).
2. **Zero-marginal-cost goal evaluator.** The `/goal` loop runs a separate transcript-only judge (default Haiku) after each turn (parity `goal.evaluator_loop`, DOCUMENTED). HIDE runs the judge as a local Haiku-class fork at near-zero marginal cost, so it can evaluate more often (even mid-turn) and run best-of-N judges to cut single-evaluator error. **Gated on:** the evaluator loop is **missing** in HIDE (parity `goal.evaluator_loop.hide_status = absent`); it is a build item (an evaluator role over the kernel, see `HIDE_AGENT_KERNEL_OPTIONS.md`), plus fork exposure for the near-free claim.
3. **Best-of-N background workers from one warm start.** A durable goal can fan out into isolated worktrees (`isolate.rs`), each a warm-state fork, reconciled by execution evidence via the tournament merge (`merge.rs:146`), never by averaging states (`HIDE_STATE_CAPSULE_ABI.md` §7). **Gated on:** fork exposure + the never-main/force-push guard (R-PR-1).
4. **Unlimited local retention.** No 100-checkpoint cap or 30-day cleanup (Claude Code's documented ring, parity `session.checkpoint_rewind`); the durable log and content-addressed artifacts are local and bounded only by disk. **Structural**, gated only on wiring the log to a live path.

## 10. Exposure: the ordered build items

The primitives are packed and unwired. To make durable and proactive agents load-bearing, in order:

1. **R-SLICE (Phase 0/1 reintegration).** Lift `hide-core` + `hide-serve` + `hide-backend` (supervisor, replay, interrupt, services, digest) + `hide-fleet` out of `5a99d0e2` onto a live path, and restore the `/v1/hide/*` boundary so the FE FleetView/StateTimeline/Digest stop being mock-fed (`HIDE_LIVE_ARCHAEOLOGY.md` §6, `HIDE_TWO_SURFACE_ARCHITECTURE.md` §7). This alone makes the dashboard, recovery, and digest real.
2. **R-SUP-1.** A per-user session-supervisor daemon that hosts detached jobs surviving terminal close and sleep, on top of the packed `RuntimeSupervisor` (which supervises the serve child) and the `JobGraph` (which recovers from the log). Parity target for `session.background_supervisor`.
3. **T-TRIG-1.** Extend `ScheduleGate` with the event-driven triggers (git-push, PR, issue, CI-failure, file-change, dependency-advisory, monitoring-alert) plus the watcher layer that emits durable gate/enqueue events (Section 4).
4. **T-NOTIF-1.** The notification/attention-inbox policy: salience thresholds, quiet hours, footprint/session dedup, digest as the verified-deliverables projection (Section 6).
5. **R-PR-1.** The auto-PR safety guard encoding never-main / never-force-push / draft-only / lineage-bound as code (Section 7), landing **before** any unattended autonomy, gated behind `hide-security` OS enforcement (frontier dossier §5.9).
6. **Warm-state durability.** Persist the state capsule alongside the text prefix so resume skips re-prefill and teams/best-of-N are near-free. This is the `HIDE_STATE_CAPSULE_ABI.md` build set (GPU readback G-CAP-1, serve state routes, `SstateDiskCache` wiring); it upgrades every "text-durable" row above to "warm-durable" and unlocks the Section 9 supremacy claims.

## 11. Honest envelope

| Claim | Status | Basis |
|---|---|---|
| Job graph survives crash/reboot and never re-launches completed runs | supported once wired | `queue.rs:491` (fold-not-relaunch, T3) |
| Runtime child restarts under a backoff ladder | supported once wired | `supervisor.rs:59,154` |
| Background sessions survive terminal close (detached) | not yet | R-SUP-1 missing; INFERRED from durable log only |
| Dashboard grouped by state with peek/reply/steer | supported once wired | `fleetview.rs`, `interrupt.rs`, FE `ui_only` |
| Triggers on time/idle/AC/thermal/manual | supported once wired | `ScheduleGate` (`queue.rs:189`) |
| Triggers on git-push/PR/issue/CI/file/advisory/alert | not yet | T-TRIG-1 build item |
| Quiet hours + notification thresholds | not yet | T-NOTIF-1 build item |
| Conditional auto-PR, never main, never force-push | not yet | merge machinery packed; R-PR-1 guard missing |
| Local routines with no per-run cost / no egress / no ephemeral wipe | structural, once wired | residency on the user's machine (Section 8) |
| Agent teams near-free vs Claude Code ~7x tokens | gated | fork exposure, RWKV lane first (`HIDE_STATE_CAPSULE_ABI.md`) |
| Zero-cost, more-frequent, best-of-N goal evaluator | gated | evaluator loop missing + fork exposure |
| Warm-state resume with no re-prefill | not yet | `HIDE_STATE_CAPSULE_ABI.md` G-CAP-1 + state routes |

The supremacy thesis (`HIDE_SUPREMACY_THESIS.md`) may claim always-on local durable agents as a structural advantage over metered cloud routines, but only text-durable behavior is defensible before R-SLICE, and every warm-state and near-free-teams claim is gated on the exposure build items above, never asserted.
