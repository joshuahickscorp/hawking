import { useEffect, useMemo, useRef, useState } from "react";
import { sendIntent } from "../../ipc";
import { useStore, type FleetRun } from "../../store";
import { intent } from "../../wire";
import { LightEdge, SectionLabel, Volume } from "../../ui";

const MOCK_SESSION = "ses_mock0000000000000000000";

const STATE_GLYPH: Record<FleetRun["state"], string> = {
  active: "●",
  waiting: "◆",
  done: "✓",
  failed: "✕",
};

const STATE_LABEL: Record<FleetRun["state"], string> = {
  active: "active",
  waiting: "needs you",
  done: "done",
  failed: "failed",
};

const STATE_TONE: Record<FleetRun["state"], string> = {
  active: "var(--light)",
  waiting: "var(--light)",
  done: "var(--ok)",
  failed: "var(--bad)",
};

const FEED: Record<string, string> = {
  run_a: "editing crates/pool/src/guard.rs",
  run_b: "running cargo test exhausted_pool",
};

const APPROACHES = [
  "drop past retry boundary",
  "scope guard with permit",
  "RAII release on acquire fail",
  "tokio semaphore owned permit",
  "explicit permit return",
  "try_acquire with backoff",
];

export interface Branch extends FleetRun {
  parent?: string;
  feed: string;
  justForked?: boolean;
}

export function FleetBoard() {
  const fleet = useStore((s) => s.fleet);
  const [branches, setBranches] = useState<Branch[]>([]);
  const [wedgeFor, setWedgeFor] = useState<string | null>(null);
  const [lastFork, setLastFork] = useState<{ from: string; n: number } | null>(null);
  const clearTimers = useRef<ReturnType<typeof setTimeout>[]>([]);

  useEffect(() => {
    setBranches((prev) => {
      const forked = prev.filter((b) => b.parent);
      const base = fleet.map((r) => ({ ...r, feed: FEED[r.id] ?? "working" }));
      return [...base, ...forked];
    });
  }, [fleet]);

  useEffect(() => () => clearTimers.current.forEach(clearTimeout), []);

  const fork = (parent: Branch, n: number) => {
    const children: Branch[] = Array.from({ length: n }, (_, i) => {
      const id = `${parent.id}_b${i + 1}`;
      void sendIntent(intent.forkSession(MOCK_SESSION, `evt_${parent.id}_${parent.step}`));
      void sendIntent(intent.custom("fleet_run", { run_id: id, parent: parent.id, approach: APPROACHES[i % APPROACHES.length] }));
      return {
        id,
        parent: parent.id,
        objective: APPROACHES[i % APPROACHES.length],
        state: "active" as const,
        step: parent.step,
        steps: parent.steps,
        feed: "state cloned, thinking",
        justForked: true,
      };
    });

    setBranches((b) => [...b, ...children]);
    setLastFork({ from: parent.objective, n });
    setWedgeFor(null);

    const t = setTimeout(
      () => setBranches((b) => b.map((x) => (x.justForked ? { ...x, justForked: false } : x))),
      1500,
    );
    clearTimers.current.push(t);
  };

  const active = branches.filter((b) => b.state === "active").length;

  return (
    <section className="section-block">
      <div className="section-head">
        <SectionLabel count={branches.length}>Fleet</SectionLabel>
        <span className="t-micro">{lastFork ? `${lastFork.n} branches, 0 re-reads` : `${active} active`}</span>
      </div>

      <div className="fleet-grid">
        {branches.length === 0 ? (
          <Volume style={{ color: "var(--text-3)" }}>No runs yet</Volume>
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
  const waiting = branch.state === "waiting";
  const body = (
    <Volume alive={live && !branch.justForked} className={["agent-card", waiting && "agent-card--waiting"].filter(Boolean).join(" ")}>
      <div className="agent-card__top">
        <span aria-hidden style={{ color: STATE_TONE[branch.state], fontSize: 11 }}>{STATE_GLYPH[branch.state]}</span>
        <span className="t-label">{STATE_LABEL[branch.state]}</span>
        {branch.parent ? <span className="t-micro">fork of {branch.parent}</span> : null}
        <span className="t-micro" style={{ marginLeft: "auto" }}>step {branch.step}/{branch.steps}</span>
      </div>

      <div className="agent-card__objective t-title">{branch.objective}</div>

      <div className="agent-feed t-code">
        {live ? <span aria-hidden style={{ color: "var(--light)" }}>›</span> : null}
        <span>{branch.feed}</span>
      </div>

      <div className="progress-track" aria-label={`${branch.step} of ${branch.steps}`}>
        <div
          className="progress-fill"
          style={{
            width: `${(branch.step / Math.max(branch.steps, 1)) * 100}%`,
            background: branch.state === "failed" ? "var(--bad)" : live || waiting ? "var(--light)" : "var(--text-3)",
          }}
        />
      </div>

      <RunTimeline runId={branch.id} steps={branch.steps} at={branch.step} />

      <div className="agent-card__actions">
        <CardBtn label="Pause" onClick={() => void sendIntent(intent.pauseRun(branch.id))} disabled={!live} />
        <CardBtn label="Stop" onClick={() => void sendIntent(intent.cancelRun(branch.id))} disabled={!live} />
        {!branch.parent ? <CardBtn label="Fork" lit onClick={onWedge} active={wedgeOpen} /> : null}
      </div>

      {wedgeOpen ? <Wedge onFork={onFork} /> : null}
    </Volume>
  );

  return branch.justForked ? <LightEdge mode="travel">{body}</LightEdge> : body;
}

function Wedge({ onFork }: { onFork: (n: number) => void }) {
  return (
    <div className="wedge">
      <span className="t-micro">try</span>
      {[3, 5, 8].map((n) => (
        <button key={n} className="ghost-button t-code" onClick={() => onFork(n)} style={{ color: "var(--light)" }}>
          {n}
        </button>
      ))}
      <span className="t-micro" style={{ marginLeft: "auto" }}>branches</span>
    </div>
  );
}

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
    <div className="timeline-strip">
      <div className="timeline-dots">
        {dots.map((i) => (
          <button
            key={i}
            className={[
              "timeline-dot",
              i <= here && "timeline-dot--past",
              i === here && "timeline-dot--current",
            ].filter(Boolean).join(" ")}
            title={`event ${i}, click scrub, shift click fork`}
            onClick={(e) => (e.shiftKey ? forkHere(i) : goto(i))}
          />
        ))}
      </div>
      {scrub != null ? (
        <button className="t-micro" onClick={() => setScrub(null)} title="return to live">
          live
        </button>
      ) : null}
    </div>
  );
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
      className="ghost-button t-micro"
      onClick={onClick}
      disabled={disabled}
      style={{
        color: disabled ? "var(--text-3)" : lit ? "var(--light)" : undefined,
        boxShadow: active ? "var(--hairline-strong), var(--light-bloom), var(--inner-glow)" : undefined,
      }}
    >
      {label}
    </button>
  );
}
