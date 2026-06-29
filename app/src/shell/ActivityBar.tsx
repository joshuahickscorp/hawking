/*
  ActivityBar.tsx — the far-left rail, consolidated to three views: Explorer (files + search folded in),
  Agents (runs + Fork-&-Try-N branches), Context (the agent's state). Chat lives in the toolbar (✦),
  not here. The active item shows a 2px left border.
*/
import { Icon, type IconName } from "./icons";

export type SideView = "explorer" | "agents" | "context";

const TOP: { view: SideView; icon: IconName; label: string }[] = [
  { view: "explorer", icon: "files", label: "Explorer" },
  { view: "agents", icon: "fleet", label: "Agents" },
  { view: "context", icon: "layers", label: "Context" },
];

export function ActivityBar({
  view,
  sidebarOpen,
  onView,
  onSettings,
}: {
  view: SideView;
  sidebarOpen: boolean;
  onView: (v: SideView) => void;
  onSettings: () => void;
}) {
  return (
    <div className="activitybar">
      <div className="activitybar__group">
        {TOP.map((t) => (
          <button
            key={t.view}
            className={["activitybar__btn", sidebarOpen && view === t.view && "activitybar__btn--active"]
              .filter(Boolean)
              .join(" ")}
            title={t.label}
            aria-label={t.label}
            aria-pressed={sidebarOpen && view === t.view}
            onClick={() => onView(t.view)}
          >
            <Icon name={t.icon} size={24} strokeWidth={1.5} />
          </button>
        ))}
      </div>

      <div className="activitybar__spacer" />

      <div className="activitybar__group">
        <button className="activitybar__btn" title="Account" aria-label="Account">
          <Icon name="user" size={24} strokeWidth={1.5} />
        </button>
        <button className="activitybar__btn" title="Settings" aria-label="Settings" onClick={onSettings}>
          <Icon name="settings" size={24} strokeWidth={1.5} />
        </button>
      </div>
    </div>
  );
}
