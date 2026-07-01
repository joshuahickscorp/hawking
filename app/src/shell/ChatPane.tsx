/*
  ChatPane.tsx — the right-hand AI pane (Cursor's chat panel). A VS Code-style header (title + model
  + new-chat + close) over the Chat surface.
*/
import { useStore } from "../store";
import { Chat } from "../surfaces/Chat";
import { Icon } from "./icons";

export function ChatPane({ onClose, onFloat }: { onClose: () => void; onFloat?: () => void }) {
  const manifest = useStore((s) => s.manifest);
  const model = manifest?.model?.id ?? "qwen2.5-7b";

  return (
    <section className="chatpane" aria-label="Executor">
      <div className="chatpane__head">
        <span className="chatpane__title">Executor</span>
        <span className="chatpane__model">{model}</span>
        <div className="chatpane__actions">
          <button className="icon-button chatpane__icon" title="New chat" aria-label="New chat">
            <Icon name="plus" size={16} />
          </button>
          {onFloat ? (
            <button className="icon-button chatpane__icon" title="Float" aria-label="Float chat" onClick={onFloat}>
              <Icon name="split" size={15} />
            </button>
          ) : null}
          <button className="icon-button chatpane__icon" title="Close" aria-label="Close chat" onClick={onClose}>
            <Icon name="close" size={16} />
          </button>
        </div>
      </div>
      <div className="chatpane__body">
        <Chat />
      </div>
    </section>
  );
}
