/*
  board.tsx: the fleet board + THE WEDGE + the per-run timeline strip.

  The fleet board is a CALM GRID of agent cards (not a swarm, Self-check C15): each card is an
  objective + one live-feed line (its current action) + status by glow (breathing gold = active,
  jade = done, amber = needs you, red = failed). Status is shape+glow, never color alone.

  THE WEDGE (HIDE_PLAN §111, the headline brand frame): one control, "fork and try N". It dispatches
  ForkSession{at_event} x N then Custom:fleet_run per branch, and at the fork moment plays the
  RadiationEdge `travel` sheen from the parent card to each freshly spawned child. The gold edge IS
  the state memcpy, rendered: no five spinners. A Cormorant number resolves the beat ("5 branches.
  0 re-reads.").

  The timeline strip is the per-run scrub/fork rail (ScrubToEvent / ForkSession), the OpenHands
  event-stream filmstrip re-housed: a lane of dots in seq order, scrub back/forward, fork at a dot.

  Local view-state seeds from the store's fleet slice (the source of truth) and layers the
  fork/timeline interaction the mock transport does not script, so the wedge is ALIVE with no backend.
*/
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { sendIntent } from "../../ipc";
import { useStore, type FleetRun } from "../../store";
import { intent } from "../../wire";
import { Display, Panel, RadiationEdge, SectionLabel } from "../../ui";

const MOCK_SESSION = "ses_mock0000000000000000000";

// status -> the marker color (always with the shape/label, never color alone).
const STATE_DOT: Record<FleetRun["state"], string> = {
  active: "var(--radiation)",
  waiting: "var(--warning)",
  done: "var(--success)",
  failed: "var(--danger)",
};
const STATE_LABEL: Record<FleetRun["state"], string> = {
  active: "active",
  waiting: "needs you",
  done: "done",
  failed: "failed",
};

// One believable live-feed line per run (the calm tool-feed, not a log firehose).
const FEED: Record<string, string> = {
  run_a: "editing crates/pool/src/guard.rs",
  run_b: "running cargo test exhausted_pool",
};

// ---- A forked branch is local view-state (the mock does not script forks). ----
export interface Branch extends FleetRun {
  parent?: string;
  feed: string;
  justForked?: boolean; // drives the one-shot travel sheen
}

const APPROACHES = [
  "drop past retry boundary",
  "scope guard with permit",
  "RAII release on acquire fail",
  "tokio Semaphore owned permit",
  "explicit permit return",
  "try_acquire with backoff",
];

export function FleetBoard() {
  const fleet = useStore((s) => s.fleet);
  // local board seeds from the store, then layers forked children + the live feed line.
  const [branches, setBranches] = useState<Branch[]>([]);
  const [wedgeFor, setWedgeFor] = useState<string | null>(null);
  const [lastFork, setLastFork] = useState<{ from: string; n: number } | null>(null);
  const clearTimers = useRef<ReturnType<typeof setTimeout>[]>([]);

  // fold store fleet into the board, preserving any locally-forked children + feed lines.
  useEffect(() => {
    setBranches((prev) => {
      const forked = prev.filter((b) => b.parent);
      const base = fleet.map((r) => ({ ...r, feed: FEED[r.id] ?? "thinking" }));
      return [...base, ...forked];
    });
  }, [fleet]);

  useEffect(() => () => clearTimers.current.forEach(clearTimeout), []);

  const fork = (parent: Branch, n: number) => {
    // THE WEDGE: ForkSession x N on the live state, then fleet_run per branch.
    const children: Branch[] = Array.from({ length: n }, (_, i) => {
      const id = `${parent.id}_b${i + 1}`;
      // each child forks at the parent's current event and starts its own fleet run.
      void sendIntent(intent.forkSession(MOCK_SESSION, `evt_${parent.id}_${parent.step}`));
      void sendIntent(intent.custom("fleet_run", { run_id: id, parent: parent.id, approach: APPROACHES[i % APPROACHES.length] }));
      return {
        id,
        parent: parent.id,
        objective: APPROACHES[i % APPROACHES.length],
        state: "active" as const,
        step: parent.step,
        steps: parent.steps,
        feed: "warm state cloned, thinking",
        justForked: true,
      };
    });
    setBranches((b) => [...b, ...children]);
    setLastFork({ from: parent.objective, n });
    setWedgeFor(null);
    // the travel sheen is one-shot: clear the flag after it sweeps (1.4s, matches radiation-travel).
    const t = setTimeout(
      () => setBranches((b) => b.map((x) => (x.justForked ? { ...x, justForked: false } : x))),
      1500,
    );
    clearTimers.current.push(t);
  };

  const active = branches.filter((b) => b.state === "active").length;

  return (
    <section>
      <div style={{ display: "flex", alignItems: "center", gap: "var(--s3)" }}>
        <SectionLabel count={branches.length}>Fleet</SectionLabel>
        {lastFork ? (
          // the Cormorant beat that resolves the fork: "N branches. 0 re-reads."
          <Display size={20} style={{ color: "var(--text-mid)", marginLeft: "auto" }}>
            {lastFork.n} branches. 0 re-reads.
          </Display>
        ) : (
          <span style={{ marginLeft: "auto", color: "var(--text-low)", fontSize: "var(--text-xs)" }}>
            {active} breathing
          </span>
        )}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
          gap: "var(--s3)",
          marginTop: "var(--s2)",
        }}
      >
        {branches.length === 0 ? (
          <Panel pad="var(--s5)" style={{ color: "var(--text-low)" }}>
            No runs yet. Fan out agents from Chat, or fork an objective and try N.
          </Panel>
        ) : (
          branches.map((b) => (
            <AgentCard
              key={b.id}
              branch={b}
              wedgeOpen={wedgeFor === b.id}
              onWedge={() => setWedgeFor((w) => (w === b.id ? null : b.id))}
              onFork={(n) => fork(b, n)}
            />
          ))
        )}
      </div>
    </section>
  );
}

function AgentCard({
  branch,
  wedgeOpen,
  onWedge,
  onFork,
}: {
  branch: Branch;
  wedgeOpen: boolean;
  onWedge: () => void;
  onFork: (n: number) => void;
}) {
  const live = branch.state === "active";
  const card = (
    <Panel
      active={live && !branch.justForked}
      pad="var(--s4)"
      style={{ display: "flex", flexDirection: "column", gap: "var(--s3)", height: "100%" }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "var(--s2)" }}>
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: STATE_DOT[branch.state],
            boxShadow: branch.state === "done" ? "0 0 6px 0 var(--success)" : undefined,
          }}
        />
        <span style={{ fontSize: "var(--text-xs)", color: "var(--text-low)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          {STATE_LABEL[branch.state]}
        </span>
        {branch.parent ? (
          <span style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}>fork of {branch.parent}</span>
        ) : null}
        <span style={{ marginLeft: "auto", color: "var(--text-low)", fontSize: "var(--text-xs)" }}>
          step {branch.step}/{branch.steps}
        </span>
      </div>

      <div style={{ color: "var(--text-hi)" }}>{branch.objective}</div>

      {/* the ONE live-feed line: current action, the calm tool feed (no firehose). */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--s2)",
          color: live ? "var(--radiation)" : "var(--text-low)",
          fontSize: "var(--text-sm)",
          minHeight: "calc(var(--text-sm) * var(--leading-ui))",
        }}
      >
        {live ? <span style={{ color: "var(--radiation)" }}>›</span> : null}
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{branch.feed}</span>
      </div>

      {/* progress as a recessed bar (real work, not a spinner). */}
      <div style={{ height: 4, borderRadius: 999, background: "var(--surface-2)", boxShadow: "inset 0 0 0 1px var(--rim)" }}>
        <div
          style={{
            width: `${(branch.step / Math.max(branch.steps, 1)) * 100}%`,
            height: "100%",
            borderRadius: 999,
            background: live ? "var(--radiation)" : STATE_DOT[branch.state],
            transition: "width 320ms ease",
          }}
        />
      </div>

      <RunTimeline runId={branch.id} steps={branch.steps} at={branch.step} />

      <div style={{ display: "flex", gap: "var(--s2)", flexWrap: "wrap" }}>
        <CardBtn label="Pause" onClick={() => void sendIntent(intent.pauseRun(branch.id))} disabled={!live} />
        <CardBtn label="Stop" onClick={() => void sendIntent(intent.cancelRun(branch.id))} disabled={!live} />
        {/* THE WEDGE control: fork and try N. */}
        {!branch.parent ? (
          <CardBtn label="Fork and try N" tone="gold" onClick={onWedge} active={wedgeOpen} />
        ) : null}
      </div>

      {wedgeOpen ? <Wedge onFork={onFork} /> : null}
    </Panel>
  );

  // at the fork moment the new child wears the travelling gold sheen: the state memcpy, rendered.
  return branch.justForked ? <RadiationEdge mode="travel">{card}</RadiationEdge> : card;
}

// THE WEDGE control body: pick N, one key fans out.
function Wedge({ onFork }: { onFork: (n: number) => void }) {
  const opts = [3, 5, 8];
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--s2)",
        padding: "var(--s2)",
        borderRadius: "var(--radius)",
        background: "var(--surface-1)",
        boxShadow: "inset 0 0 0 1px var(--rim)",
      }}
    >
      <span style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}>try</span>
      {opts.map((n) => (
        <button
          key={n}
          onClick={() => onFork(n)}
          style={{
            padding: "2px 12px",
            borderRadius: "var(--radius)",
            color: "var(--radiation-bright)",
            fontSize: "var(--text-sm)",
            background: "var(--surface-0)",
            boxShadow: "inset 0 0 0 1px var(--radiation)",
          }}
        >
          {n}
        </button>
      ))}
      <span style={{ color: "var(--text-low)", fontSize: "var(--text-xs)", marginLeft: "auto" }}>approaches at once</span>
    </div>
  );
}

// ---- The per-run timeline strip: scrub / fork over the event log (OpenHands filmstrip, re-housed). ----
export function RunTimeline({ runId, steps, at }: { runId: string; steps: number; at: number }) {
  const [scrub, setScrub] = useState<number | null>(null);
  const here = scrub ?? at;
  const dots = useMemo(() => Array.from({ length: Math.max(steps, 1) }, (_, i) => i + 1), [steps]);

  const goto = (i: number) => {
    setScrub(i);
    void sendIntent(intent.scrubToEvent(MOCK_SESSION, `evt_${runId}_${i}`));
  };
  const forkHere = (i: number) => void sendIntent(intent.forkSession(MOCK_SESSION, `evt_${runId}_${i}`));

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "var(--s2)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, flex: 1 }}>
        {dots.map((i) => {
          const past = i <= here;
          const cur = i === here;
          return (
            <button
              key={i}
              title={`event ${i}   click scrub, shift-click fork`}
              onClick={(e) => (e.shiftKey ? forkHere(i) : goto(i))}
              style={dotStyle(past, cur)}
            />
          );
        })}
      </div>
      {scrub != null ? (
        <button
          onClick={() => setScrub(null)}
          style={{ color: "var(--text-low)", fontSize: "var(--text-xs)" }}
          title="return to live"
        >
          live
        </button>
      ) : null}
    </div>
  );
}

function dotStyle(past: boolean, cur: boolean): CSSProperties {
  return {
    width: cur ? 9 : 7,
    height: cur ? 9 : 7,
    padding: 0,
    borderRadius: "50%",
    background: past ? "var(--radiation)" : "var(--surface-2)",
    boxShadow: cur
      ? "0 0 0 1px var(--radiation-bright), 0 0 8px 0 var(--radiation-bloom)"
      : "inset 0 0 0 1px var(--rim)",
    transition: "all 160ms ease",
  };
}

function CardBtn({
  label,
  onClick,
  disabled,
  tone,
  active,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  tone?: "gold";
  active?: boolean;
}) {
  const gold = tone === "gold";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "3px 10px",
        borderRadius: "var(--radius)",
        fontSize: "var(--text-xs)",
        color: disabled ? "var(--text-low)" : gold ? "var(--radiation-bright)" : "var(--text-mid)",
        boxShadow: active
          ? "inset 0 0 0 1px var(--radiation), 0 0 12px -4px var(--radiation-bloom)"
          : gold
            ? "inset 0 0 0 1px var(--radiation)"
            : "inset 0 0 0 1px var(--rim)",
        background: active ? "var(--surface-2)" : "var(--surface-1)",
      }}
    >
      {label}
    </button>
  );
}
