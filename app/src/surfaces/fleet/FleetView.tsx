/*
  FleetView.tsx: Fork and Try N. Renders store.fleet as compact branch cards.

  HONEST SCOPE (remediation stage): no Rust code publishes a `fleet` projection. The ONLY producer is
  the mock transport (src/ipc.ts), so `store.fleet` is empty on every live host and this surface never
  renders a card there. Nothing is seeded by App any more either. It is kept because the mock run is
  the demo of the fork-and-try shape, not because a live fleet exists.

  RETIRED with this stage: the "keep best" button. It fired the `focus_run` custom name, which no host
  arm handles, under a tooltip promising to discard the other branches. Nothing was kept and nothing
  was discarded. The catalog's `promote_run` is a different capability (promote ONE run to a durable
  background job, offered in Home's Background rail), not a keep-one-discard-the-rest, so there was
  nothing honest to re-point this at. `stop` stays: cancel_run is real.
*/
import { noticeFailure, runCommand, useStore, type FleetRun } from "../../store";
import { Radiate } from "../../shell/Radiate";

const STATE: Record<FleetRun["state"], { label: string; color: string }> = {
  active: { label: "radiating", color: "var(--light)" },
  waiting: { label: "needs you", color: "var(--light)" },
  done: { label: "done", color: "var(--ok)" },
  failed: { label: "failed", color: "var(--bad)" },
};

export function FleetView() {
  const fleet = useStore((s) => s.fleet);
  const tools = useStore((s) => s.tools);
  const lastTool = tools[tools.length - 1]?.message;

  if (fleet.length === 0) {
    return (
      <div className="fleet">
        <p className="fleet__pitch">
          No attempts yet. Parallel attempts are a demo of the fork-and-try shape; no host publishes
          a fleet yet, so nothing runs here on a live workspace.
        </p>
      </div>
    );
  }

  const active = fleet.filter((r) => r.state === "active").length;
  return (
    <div className="fleet">
      <div className="fleet__pitch">
        {fleet.length} attempt{fleet.length === 1 ? "" : "s"}, free, local{active ? `, ${active} radiating` : ""}
      </div>
      <ul className="fleet__list">
        {fleet.map((run) => (
          <Branch key={run.id} run={run} live={run.state === "active" ? lastTool : undefined} />
        ))}
      </ul>
    </div>
  );
}

function Branch({ run, live }: { run: FleetRun; live?: string }) {
  const s = STATE[run.state];
  const pct = run.steps ? Math.round((run.step / run.steps) * 100) : 0;
  const isActive = run.state === "active";
  // Through the spine: the card used to build a cancel_run Intent itself and skip its guards.
  const stop = () => void runCommand("cancel_run", { run_id: run.id }).catch(noticeFailure("command"));
  return (
    <li className={["branch", isActive && "branch--alive"].filter(Boolean).join(" ")}>
      <div className="branch__top">
        <Radiate size={13} active={isActive} title={s.label} />
        <span className="branch__state" style={{ color: s.color }}>{s.label}</span>
        <span className="branch__steps">{run.step}/{run.steps}</span>
      </div>
      <div className="branch__obj">{run.objective}</div>
      {isActive && live ? <div className="branch__feed">› {live}</div> : null}
      <div className="branch__track">
        <div className="branch__fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="branch__actions">
        <button className="branch__stop" onClick={stop} disabled={!isActive} title="Stop this attempt">
          stop
        </button>
      </div>
    </li>
  );
}
