/*
  EditorArea.tsx — the center column: a VS Code tab strip over the Monaco editor group, plus a
  collapsible bottom panel hosting the integrated terminal. State (open tabs, active path, diff) is
  owned by App and threaded through.
*/
import { EditorGroup } from "../surfaces/ide/Editor";
import { Terminal } from "../surfaces/ide/Terminal";
import type { DiffDoc } from "../surfaces/ide/types";
import { Icon } from "./icons";
import { StateTimeline } from "./StateTimeline";

export function EditorArea({
  openPath,
  tabs,
  diff,
  panelOpen,
  onSelectTab,
  onCloseTab,
  onDiffChange,
  onTogglePanel,
}: {
  openPath: string | null;
  tabs: string[];
  diff: DiffDoc | null;
  panelOpen: boolean;
  onSelectTab: (path: string) => void;
  onCloseTab: (path: string) => void;
  onDiffChange: (next: DiffDoc | null) => void;
  onTogglePanel: () => void;
}) {
  return (
    <div className="editor-area">
      <StateTimeline />
      <div className="tabstrip" role="tablist">
        {tabs.map((path) => {
          const name = path.split("/").pop() ?? path;
          const active = path === openPath;
          return (
            <div
              key={path}
              role="tab"
              aria-selected={active}
              className={["tab", active && "tab--active"].filter(Boolean).join(" ")}
              onClick={() => onSelectTab(path)}
              title={path}
            >
              <span className="tab__dot" data-ext={name.split(".").pop()} />
              <span className="tab__name">{name}</span>
              <button
                className="tab__close"
                title="Close"
                aria-label={`Close ${name}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onCloseTab(path);
                }}
              >
                <Icon name="close" size={13} strokeWidth={1.8} />
              </button>
            </div>
          );
        })}
      </div>

      <div className="editor-host">
        <EditorGroup openPath={openPath} diff={diff} onDiffChange={onDiffChange} />
      </div>

      {panelOpen ? (
        <section className="bottom-panel" aria-label="Panel">
          <div className="bottom-panel__tabs">
            <span className="bottom-panel__tab bottom-panel__tab--active">Terminal</span>
            <span className="bottom-panel__tab">Problems</span>
            <span className="bottom-panel__tab">Output</span>
            <div style={{ marginLeft: "auto", display: "flex" }}>
              <button className="icon-button" title="Close panel" aria-label="Close panel" onClick={onTogglePanel}>
                <Icon name="close" size={15} />
              </button>
            </div>
          </div>
          <div className="bottom-panel__body">
            <Terminal />
          </div>
        </section>
      ) : null}
    </div>
  );
}
