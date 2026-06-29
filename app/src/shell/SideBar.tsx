/*
  SideBar.tsx — the primary sidebar, consolidated to three views. Explorer folds Search into its
  top filter; Agents folds the runs list + Fork-&-Try-N branches into one; Context is the state stack.
  (SearchView / RunsView remain in the tree but are superseded by the folded surfaces.)
*/
import { ContextStack } from "../surfaces/ContextStack";
import { Explorer } from "../surfaces/ide/Explorer";
import { FleetView } from "../surfaces/fleet/FleetView";
import type { SideView } from "./ActivityBar";

const HEADS: Record<SideView, string> = {
  explorer: "Explorer",
  agents: "Agents",
  context: "Context",
};

export function SideBar({
  view,
  openPath,
  onOpen,
}: {
  view: SideView;
  openPath: string | null;
  onOpen: (path: string) => void;
}) {
  return (
    <nav className="sidebar" aria-label={view}>
      <div className="sidebar__head">{HEADS[view]}</div>
      <div className="sidebar__body">
        {view === "explorer" ? <Explorer activePath={openPath} onOpen={onOpen} /> : null}
        {view === "agents" ? <FleetView /> : null}
        {view === "context" ? <ContextStack /> : null}
      </div>
    </nav>
  );
}
