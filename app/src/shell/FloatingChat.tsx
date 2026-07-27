/*
  FloatingChat.tsx — the summoned assistant as a draggable Liquid-Glass panel that floats over the
  editor (Xcode-assistant style), instead of a fixed dock. Drag by the header. Dock/close from the
  header. Hosts the existing <Chat/> surface unchanged. (The docked ChatPane is kept as the alternate.)
*/
import { useEffect, useRef, type PointerEvent as ReactPointerEvent } from "react";
import { useStore } from "../store";
import { Chat, NewChatButton } from "../surfaces/Chat";
import { Icon } from "./icons";
import { modelId } from "./ModelChooser";

export function FloatingChat({
  pos,
  onPos,
  onClose,
  onDock,
  onPopToChat,
}: {
  pos: { x: number; y: number };
  onPos: (p: { x: number; y: number }) => void;
  onClose: () => void;
  onDock: () => void;
  onPopToChat: () => void;
}) {
  const manifest = useStore((s) => s.manifest);
  const model = modelId(manifest);
  const drag = useRef<{ dx: number; dy: number } | null>(null);

  const onPointerDown = (e: ReactPointerEvent) => {
    drag.current = { dx: e.clientX - pos.x, dy: e.clientY - pos.y };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: ReactPointerEvent) => {
    if (!drag.current) return;
    const x = Math.max(8, Math.min(window.innerWidth - 120, e.clientX - drag.current.dx));
    const y = Math.max(44, Math.min(window.innerHeight - 80, e.clientY - drag.current.dy));
    onPos({ x, y });
  };
  const onPointerUp = (e: ReactPointerEvent) => {
    drag.current = null;
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
  };

  // Re-clamp into view on window resize (otherwise a panel parked at the old right edge can end up
  // off-screen after the window shrinks). Same bounds as the drag clamp.
  useEffect(() => {
    const clamp = () => {
      const x = Math.max(8, Math.min(window.innerWidth - 120, pos.x));
      const y = Math.max(44, Math.min(window.innerHeight - 80, pos.y));
      if (x !== pos.x || y !== pos.y) onPos({ x, y });
    };
    window.addEventListener("resize", clamp);
    return () => window.removeEventListener("resize", clamp);
  }, [pos.x, pos.y, onPos]);

  return (
    <section className="floatchat glass" style={{ left: pos.x, top: pos.y }} role="dialog" aria-label="Executor">
      <div
        className="floatchat__head"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <span className="floatchat__spark" aria-hidden>
          <Icon name="sparkle" size={15} strokeWidth={1.4} />
        </span>
        <span className="floatchat__title">Executor</span>
        <span className="floatchat__model">{model}</span>
        <div className="floatchat__actions">
          {/* One shared New-chat control (surfaces/Chat.tsx), identical here and in ChatPane. */}
          <NewChatButton className="floatchat__icon" size={15} />
          <button className="floatchat__icon" title="Open in Chat (picture in picture)" aria-label="Open in Chat" onClick={onPopToChat}>
            <Icon name="pip" size={14} />
          </button>
          <button className="floatchat__icon" title="Dock to side" aria-label="Dock" onClick={onDock}>
            <Icon name="split" size={14} />
          </button>
          <button className="floatchat__icon" title="Close (Esc)" aria-label="Close" onClick={onClose}>
            <Icon name="close" size={15} />
          </button>
        </div>
      </div>
      <div className="floatchat__body">
        <Chat />
      </div>
    </section>
  );
}
