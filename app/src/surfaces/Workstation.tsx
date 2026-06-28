/*
  Workstation.tsx: the AI Workstation surface frame (01-surfaces §D). The front door.
  A calm board of agent cards, each a run with a status glow (breathing gold = active,
  green = done, amber = waiting). Fed by projection_patch(fleet) via the store. The big
  Cormorant editorial number is the 032c "morning digest" moment.
  Skeleton: fleet grid + digest. Merge-review and the per-run timeline are the surface pass.
  Sends: Custom:fleet_run, PauseRun/CancelRun per run.
*/
import { sendIntent } from "../ipc";
import { useStore, type FleetRun } from "../store";
import { intent } from "../wire";
import { Display, Panel, SectionLabel } from "../ui";

const STATE_DOT: Record<FleetRun["state"], string> = {
  active: "var(--radiation)",
  waiting: "var(--warning)",
  done: "var(--success)",
  failed: "var(--danger)",
};

export function Workstation() {
  const fleet = useStore((s) => s.fleet);
  const active = fleet.filter((r) => r.state === "active").length;
  const needs = fleet.filter((r) => r.state === "waiting").length;

  return (
    <div style={{ height: "100%", overflowY: "auto", padding: "var(--s6)" }}>
      {/* The morning digest: one editorial number in Cormorant. */}
      <header style={{ maxWidth: 980, margin: "0 auto var(--s6)" }}>
        <Display size={40}>
          {fleet.length} agent{fleet.length === 1 ? "" : "s"} running. {needs} need{needs === 1 ? "s" : ""} you.
        </Display>
        <p style={{ color: "var(--text-low)", marginTop: "var(--s3)" }}>
          {active} active, {fleet.filter((r) => r.state === "done").length} ready to review. Spend lavishly, locally.
        </p>
      </header>

      <div style={{ maxWidth: 980, margin: "0 auto" }}>
        <SectionLabel count={fleet.length}>Fleet</SectionLabel>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
            gap: "var(--s3)",
          }}
        >
          {fleet.length === 0 ? (
            <Panel pad="var(--s5)" style={{ color: "var(--text-low)" }}>
              No runs yet. Fan out agents from Chat or a fleet objective.
            </Panel>
          ) : (
            fleet.map((r) => <RunCard key={r.id} run={r} />)
          )}
        </div>
      </div>
    </div>
  );
}

function RunCard({ run }: { run: FleetRun }) {
  const live = run.state === "active";
  return (
    <Panel active={live} pad="var(--s4)" style={{ display: "flex", flexDirection: "column", gap: "var(--s3)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "var(--s2)" }}>
        <span style={{ width: 8, height: 8, borderRadius: "50%", background: STATE_DOT[run.state] }} />
        <span style={{ fontSize: "var(--text-xs)", color: "var(--text-low)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          {run.state}
        </span>
        <span style={{ marginLeft: "auto", color: "var(--text-low)", fontSize: "var(--text-xs)" }}>
          step {run.step}/{run.steps}
        </span>
      </div>
      <div style={{ color: "var(--text-hi)" }}>{run.objective}</div>
      {/* step progress as a recessed bar (real work, not a spinner) */}
      <div style={{ height: 4, borderRadius: 999, background: "var(--surface-2)", boxShadow: "inset 0 0 0 1px var(--rim)" }}>
        <div
          style={{
            width: `${(run.step / Math.max(run.steps, 1)) * 100}%`,
            height: "100%",
            borderRadius: 999,
            background: live ? "var(--radiation)" : STATE_DOT[run.state],
          }}
        />
      </div>
      <div style={{ display: "flex", gap: "var(--s2)" }}>
        <CardBtn label="Pause" onClick={() => void sendIntent(intent.pauseRun(run.id))} disabled={!live} />
        <CardBtn label="Stop" onClick={() => void sendIntent(intent.cancelRun(run.id))} disabled={!live} />
      </div>
    </Panel>
  );
}

function CardBtn({ label, onClick, disabled }: { label: string; onClick: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "3px 10px",
        borderRadius: "var(--radius)",
        fontSize: "var(--text-xs)",
        color: disabled ? "var(--text-low)" : "var(--text-mid)",
        boxShadow: "inset 0 0 0 1px var(--rim)",
        background: "var(--surface-1)",
      }}
    >
      {label}
    </button>
  );
}
