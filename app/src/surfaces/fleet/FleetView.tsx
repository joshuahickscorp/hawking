/*
  FleetView.tsx — Fork & Try N. The move cloud tools can't price: fork the agent's RWKV state into N
  parallel attempts (a memcpy, ~free on already-paid silicon), watch them all radiate, keep the best.
  Renders store.fleet as compact branch cards. (Backend forks real state in plan 2; until then App
  seeds the branches optimistically, honestly framed as local.)
*/
import { sendIntent } from "../../ipc";
import { useStore, type FleetRun } from "../../store";
import { intent } from "../../wire";
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
          No attempts yet. Fork the agent state into N branches and run them in parallel, free, on your
          machine, then keep the best. Use "try 3 / 5 / 8" in the toolbar.
        </p>
      </div>
    );
  }

  const active = fleet.filter((r) => r.state === "active").length;
  return (
    <div className="fleet">
      <div className="fleet__pitch">
        {fleet.length} attempt{fleet.length === 1 ? "" : "s"} · free · local{active ? ` · ${active} radiating` : ""}
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
  const keep = () => void sendIntent(intent.custom("focus_run", { run_id: run.id }));
  const stop = () => void sendIntent(intent.cancelRun(run.id));
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
        <button className="branch__keep" onClick={keep} title="Keep this branch, discard the rest">
          keep best
        </button>
        <button className="branch__stop" onClick={stop} disabled={!isActive} title="Stop this attempt">
          stop
        </button>
      </div>
    </li>
  );
}
