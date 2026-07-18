/*
  ChatPane.tsx — the right-hand AI pane (Cursor's chat panel). A VS Code-style header (title + model +
  pop-to-chat + new-chat + float + close) over the Chat surface. Resizable: drag the left edge to size it
  for the moment (shrink down to just the composer). Width persists.
*/
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { useStore } from "../store";
import { Chat } from "../surfaces/Chat";
import { Icon } from "./icons";

const MIN_W = 300; // enough for "Describe a task"
const MAX_W = 760;

function readWidth(): number {
  try {
    const v = Number.parseInt(localStorage.getItem("hide.chatW") ?? "", 10);
    if (Number.isFinite(v)) return Math.min(MAX_W, Math.max(MIN_W, v));
  } catch {
    /* storage unavailable */
  }
  return 384;
}

export function ChatPane({
  onClose,
  onFloat,
  onPopToChat,
}: {
  onClose: () => void;
  onFloat?: () => void;
  onPopToChat?: () => void;
}) {
  const manifest = useStore((s) => s.manifest);
  const model = manifest?.model?.id ?? "qwen2.5-7b";

  const [width, setWidth] = useState(readWidth);
  const drag = useRef<{ startX: number; startW: number } | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem("hide.chatW", String(width));
    } catch {
      /* storage unavailable */
    }
  }, [width]);

  const onPointerDown = (e: ReactPointerEvent) => {
    drag.current = { startX: e.clientX, startW: width };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    e.preventDefault();
  };
  const onPointerMove = (e: ReactPointerEvent) => {
    if (!drag.current) return;
    // Dragging the left edge leftward widens the pane.
    const next = drag.current.startW + (drag.current.startX - e.clientX);
    setWidth(Math.min(MAX_W, Math.max(MIN_W, next)));
  };
  const onPointerUp = (e: ReactPointerEvent) => {
    drag.current = null;
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
  };

  return (
    <section className="chatpane" aria-label="Executor" style={{ width }}>
      <div
        className="chatpane__resize"
        role="separator"
        aria-label="Resize Executor"
        aria-orientation="vertical"
        title="Drag to resize"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />
      <div className="chatpane__head">
        <span className="chatpane__title">Executor</span>
        <span className="chatpane__model">{model}</span>
        <div className="chatpane__actions">
          <button className="icon-button chatpane__icon" title="New chat" aria-label="New chat">
            <Icon name="plus" size={16} />
          </button>
          {onPopToChat ? (
            <button className="icon-button chatpane__icon" title="Open in Chat (picture in picture)" aria-label="Open in Chat" onClick={onPopToChat}>
              <Icon name="pip" size={15} />
            </button>
          ) : null}
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
