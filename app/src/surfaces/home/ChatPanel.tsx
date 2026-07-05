/*
  ChatPanel.tsx — the active-chat side panel (Claude Code's Terminal / Diff / Preview, recast). A
  full-height right column beside the conversation; the switcher lives in the Chat stage and toggles which
  face shows. Terminal and the diff review are the real IDE components, reused; Preview is the local view.
*/
import { Terminal } from "../ide/Terminal";
import { HunkReview, type HunkAction } from "../ide/HunkReview";
import type { DiffDoc, Hunk } from "../ide/types";
import { Icon } from "../../shell/icons";
import { Preview } from "./Preview";

export type ChatPanelKind = "terminal" | "diff" | "preview";

const TITLE: Record<ChatPanelKind, string> = { terminal: "Terminal", diff: "Diff", preview: "Preview" };

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
      </div>
    </aside>
  );
}
