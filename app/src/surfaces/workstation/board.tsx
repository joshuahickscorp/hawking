/*
  board.tsx: the fleet board + THE WEDGE + the per-run timeline strip.

  Doctrine v3 (Tadao Ando grayscale concrete): the fleet board is a CALM GRID of .volume slabs
  floating in generous void (not a swarm). Each card is an objective + one live-feed line (its
  current action) + STATE READ BY LIGHT, never a colored badge: an active card breathes (.alive
  via LightEdge), a done card rests steady, a card that NEEDS YOU is lit and steady (the agent
  asking for you, the cross of light), a failed card carries the --bad glyph and quiet text.
  There is no third color and no amber: "needs you" is a glyph + neutral text + the steady light.

  THE WEDGE (the headline brand frame): one control, "fork and try N". It dispatches
  ForkSession{at_event} x N then Custom:fleet_run per branch, and at the fork moment plays the
  LightEdge mode="travel" sheen from the parent card to each freshly spawned child. The light edge
  IS the state memcpy, rendered: no five spinners. A Display number resolves the beat ("5 branches.
  0 re-reads.").

  The timeline strip is the per-run scrub/fork rail (ScrubToEvent / ForkSession), a lane of dots
  in seq order rendered in light (never gold), scrub back/forward, fork at a dot.

  Local view-state seeds from the store's fleet slice (the source of truth) and layers the
  fork/timeline interaction the mock transport does not script, so the wedge is ALIVE with no backend.
*/
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { sendIntent } from "../../ipc";
import { useStore, type FleetRun } from "../../store";
import { intent } from "../../wire";
import { Display, LightEdge, SectionLabel, Volume } from "../../ui";

const MOCK_SESSION = "ses_mock0000000000000000000";

// status -> a glyph + label. State is read by light (breathe / steady / lit) plus this glyph,
// never by color alone. Only "failed" carries a pigment (--bad), always with its glyph.
const STATE_GLYPH: Record<FleetRun["state"], string> = {
  active: "›",
  waiting: "◆", // the agent is asking for you
  done: "●",
  failed: "✕",
};
const STATE_LABEL: Record<FleetRun["state"], string> = {
  active: "active",
  waiting: "needs you",
  done: "done",
  failed: "failed",
};
// the glyph tone: light for the states the eye should land on, --bad only for failure.
const STATE_TONE: Record<FleetRun["state"], string> = {
  active: "var(--light)",
  waiting: "var(--light)",
  done: "var(--text-2)",
  failed: "var(--bad)",
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
    // the travel sheen is one-shot: clear the flag after it sweeps (matches the light-travel keyframe).
    const t = setTimeout(
      () => setBranches((b) => b.map((x) => (x.justForked ? { ...x, justForked: false } : x))),
      1500,
    );
    clearTimers.current.push(t);
  };

  const active = branches.filter((b) => b.state === "active").length;

  return (
    <section>
      <div style={{ display: "flex", alignItems: "center", gap: "var(--ma-4)" }}>
        <SectionLabel count={branches.length}>Fleet</SectionLabel>
        {lastFork ? (
          // the Display beat that resolves the fork: "N branches. 0 re-reads."
          <Display style={{ fontSize: "20px", letterSpacing: "-0.02em", color: "var(--text-2)", marginLeft: "auto" }}>
            {lastFork.n} branches. 0 re-reads.
          </Display>
        ) : (
          <span className="t-micro" style={{ marginLeft: "auto" }}>{active} breathing</span>
        )}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
          gap: "var(--ma-8)",
          marginTop: "var(--ma-6)",
        }}
      >
        {branches.length === 0 ? (
          <Volume pad="var(--ma-8)" style={{ color: "var(--text-3)" }}>
            No runs yet. Fan out agents from Chat, or fork an objective and try N.
          </Volume>
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
  const needsYou = branch.state === "waiting";
  // a card that NEEDS YOU holds a steady lit edge (the agent asking, not breathing);
  // an active card breathes; done and failed rest quiet. State is read by light, never a badge.
  const litSteady: CSSProperties = needsYou
    ? { boxShadow: "var(--hairline-strong), var(--light-bloom), var(--inner-glow)" }
    : {};
  const card = (
    <Volume
      alive={live && !branch.justForked}
      pad="var(--ma-6)"
      style={{ display: "flex", flexDirection: "column", gap: "var(--ma-4)", height: "100%", ...litSteady }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "var(--ma-2)" }}>
        <span style={{ color: STATE_TONE[branch.state], fontSize: "11px", lineHeight: 1 }}>
          {STATE_GLYPH[branch.state]}
        </span>
        <span className="t-label">{STATE_LABEL[branch.state]}</span>
        {branch.parent ? (
          <span className="t-micro">fork of {branch.parent}</span>
        ) : null}
        <span className="t-micro" style={{ marginLeft: "auto" }}>
          step {branch.step}/{branch.steps}
        </span>
      </div>

      <div className="t-title" style={{ color: "var(--text-1)" }}>{branch.objective}</div>

      {/* the ONE live-feed line: current action, the calm tool feed (no firehose). */}
      <div
        className="t-code"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--ma-2)",
          color: live ? "var(--text-1)" : "var(--text-2)",
          minHeight: "calc(13.5px * 1.55)",
        }}
      >
        {live ? <span style={{ color: "var(--light)" }}>›</span> : null}
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{branch.feed}</span>
      </div>

      {/* progress as a recessed bar (real work, not a spinner); fill in light, never color. */}
      <div style={{ height: 4, borderRadius: 999, background: "var(--concrete-1)", boxShadow: "var(--hairline)" }}>
        <div
          style={{
            width: `${(branch.step / Math.max(branch.steps, 1)) * 100}%`,
            height: "100%",
            borderRadius: 999,
            background: branch.state === "failed" ? "var(--bad)" : live || needsYou ? "var(--light)" : "var(--text-3)",
            transition: "width var(--dur-slow) var(--ease)",
          }}
        />
      </div>

      <RunTimeline runId={branch.id} steps={branch.steps} at={branch.step} />

      <div style={{ display: "flex", gap: "var(--ma-2)", flexWrap: "wrap" }}>
        <CardBtn label="Pause" onClick={() => void sendIntent(intent.pauseRun(branch.id))} disabled={!live} />
        <CardBtn label="Stop" onClick={() => void sendIntent(intent.cancelRun(branch.id))} disabled={!live} />
        {/* THE WEDGE control: fork and try N. */}
        {!branch.parent ? (
          <CardBtn label="Fork and try N" lit onClick={onWedge} active={wedgeOpen} />
        ) : null}
      </div>

      {wedgeOpen ? <Wedge onFork={onFork} /> : null}
    </Volume>
  );

  // at the fork moment the new child wears the travelling light sheen: the state memcpy, rendered.
  return branch.justForked ? <LightEdge mode="travel">{card}</LightEdge> : card;
}

// THE WEDGE control body: pick N, one key fans out.
function Wedge({ onFork }: { onFork: (n: number) => void }) {
  const opts = [3, 5, 8];
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--ma-2)",
        padding: "var(--ma-3)",
        borderRadius: "var(--radius)",
        background: "var(--concrete-1)",
        boxShadow: "var(--hairline)",
      }}
    >
      <span className="t-micro">try</span>
      {opts.map((n) => (
        <button
          key={n}
          onClick={() => onFork(n)}
          className="t-code"
          style={{
            padding: "2px 12px",
            borderRadius: "var(--radius)",
            color: "var(--light)",
            background: "var(--concrete-3)",
            boxShadow: "var(--hairline-strong), var(--light-bloom)",
          }}
        >
          {n}
        </button>
      ))}
      <span className="t-micro" style={{ marginLeft: "auto" }}>approaches at once</span>
    </div>
  );
}

// ---- The per-run timeline strip: scrub / fork over the event log. ----
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
    <div style={{ display: "flex", alignItems: "center", gap: "var(--ma-2)" }}>
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
          className="t-micro"
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
    // dots are light, never gold: past steps glow faintly, the current step blooms.
    background: past ? "var(--light)" : "var(--concrete-4)",
    boxShadow: cur ? "var(--light-bloom)" : "var(--hairline)",
    transition: "all var(--dur) var(--ease)",
  };
}

function CardBtn({
  label,
  onClick,
  disabled,
  lit,
  active,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  lit?: boolean;
  active?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="t-micro"
      style={{
        padding: "3px 10px",
        borderRadius: "var(--radius)",
        color: disabled ? "var(--text-3)" : lit ? "var(--light)" : "var(--text-2)",
        boxShadow: active
          ? "var(--hairline-strong), var(--light-bloom)"
          : lit
            ? "var(--hairline-strong)"
            : "var(--hairline)",
        background: active ? "var(--concrete-3)" : "var(--concrete-2)",
      }}
    >
      {label}
    </button>
  );
}
