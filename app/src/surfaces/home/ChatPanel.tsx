/*
  ChatPanel.tsx — the active-chat side panel (Claude Code's Terminal / Diff / Preview, recast). A
  full-height right column beside the conversation; the switcher lives in the Chat stage and toggles which
  face shows. Terminal and the diff review are the real IDE components, reused; Preview is the local view;
  Tools is the agent's live tool feed; Artifacts is what the run produced.
*/
import { Terminal } from "../ide/Terminal";
import { HunkReview, type HunkAction } from "../ide/HunkReview";
import type { DiffDoc, Hunk } from "../ide/types";
import { Icon } from "../../shell/icons";
import { Preview } from "./Preview";
import { useStore } from "../../store";

export type ChatPanelKind = "terminal" | "diff" | "preview" | "tools" | "artifacts";

const TITLE: Record<ChatPanelKind, string> = {
  terminal: "Terminal",
  diff: "Diff",
  preview: "Preview",
  tools: "Tools",
  artifacts: "Artifacts",
};

// The agent's tool calls, newest first. Real data (tool_progress stream), not a placeholder.
function ToolsPanel() {
  const tools = useStore((s) => s.tools);
  if (!tools.length) {
    return <div className="home-panel__empty t-body">No tool activity yet. The agent's moves show here.</div>;
  }
  return (
    <ul className="toolfeed">
      {tools
        .slice()
        .reverse()
        .map((t) => (
          <li key={t.call_id + t.ts} className="toolfeed__row">
            <Icon name="tool" size={13} />
            <span className="toolfeed__msg">{t.message}</span>
          </li>
        ))}
    </ul>
  );
}

// What the run produced: the file the current diff touches, plus anything loaded in Preview.
function ArtifactsPanel({ diff }: { diff: DiffDoc | null }) {
  if (!diff) {
    return (
      <div className="home-panel__empty t-body">No artifacts yet. Files and previews the run produces show here.</div>
    );
  }
  const added = diff.hunks.filter((h) => h.status === "accepted").length;
  return (
    <ul className="toolfeed">
      <li className="toolfeed__row">
        <Icon name="box" size={13} />
        <span className="toolfeed__msg">{diff.path}</span>
        <span className="toolfeed__meta">{added ? `${added} applied` : "proposed"}</span>
      </li>
    </ul>
  );
}

export function ChatPanel({
  panel,
  onClose,
  diff,
  onDiffAct,
}: {
  panel: ChatPanelKind;
  onClose: () => void;
  diff: DiffDoc | null;
  onDiffAct: (hunk: Hunk, action: HunkAction) => void;
}) {
  return (
    <aside className="home-panel" aria-label={TITLE[panel]}>
      <div className="home-panel__head">
        <span className="t-label">{TITLE[panel]}</span>
        <button className="home-panel__close" title="Close panel" aria-label="Close panel" onClick={onClose}>
          <Icon name="close" size={15} />
        </button>
      </div>
      <div className="home-panel__body">
        {panel === "terminal" ? <Terminal /> : null}
        {panel === "diff" ? (
          diff ? (
            <HunkReview doc={diff} onAct={onDiffAct} />
          ) : (
            <div className="home-panel__empty t-body">No changes yet. Edits the agent proposes show here.</div>
          )
        ) : null}
        {panel === "preview" ? <Preview /> : null}
        {panel === "tools" ? <ToolsPanel /> : null}
        {panel === "artifacts" ? <ArtifactsPanel diff={diff} /> : null}
      </div>
    </aside>
  );
}
