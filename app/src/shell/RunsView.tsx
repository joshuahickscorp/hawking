/*
  RunsView.tsx — the Source-Control-style "Runs" sidebar view (the folded fleet). A compact list of
  agent runs with a status glyph + objective + step counter. Phase 4 wires selecting a run/change to
  open its diff in the editor; for now it lists live fleet state.
*/
import { useStore, type FleetRun } from "../store";

const STATE_MARK: Record<FleetRun["state"], { glyph: string; color: string; label: string }> = {
  active: { glyph: "●", color: "var(--accent)", label: "active" },
  waiting: { glyph: "◆", color: "var(--accent)", label: "needs you" },
  done: { glyph: "✓", color: "var(--green)", label: "done" },
  failed: { glyph: "✕", color: "var(--red)", label: "failed" },
};

export function RunsView() {
  const fleet = useStore((s) => s.fleet);

  if (fleet.length === 0) {
    return <div className="sidebar__empty">No active runs</div>;
  }

  return (
    <div className="runs-view">
      <div className="runs-subhead">Runs</div>
      <ul className="runs-list">
        {fleet.map((run) => {
          const m = STATE_MARK[run.state];
          return (
            <li key={run.id}>
              <button className="ghost-button runs-row" title={`${run.objective} — ${m.label}`}>
                <span className="runs-row__glyph" style={{ color: m.color }}>{m.glyph}</span>
                <span className="runs-row__name">{run.objective}</span>
                <span className="runs-row__meta">{run.step}/{run.steps}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
