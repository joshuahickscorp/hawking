/*
  CodeActions.tsx — highlight-to-100x. Select code in the editor and a small Liquid-Glass popover
  offers one-tap leverage: explain / refactor / test / fork & try 3. A short prompt, big value. Each
  dispatches a real intent (submit_turn / inline_edit / fleet_run); the chat/fleet surfaces pick it up.
  Keyboard: opens with focus on the first action; Esc dismisses (returns focus to the editor),
  Up/Down move between actions, Enter activates.
*/
import { useEffect, useRef } from "react";
import { sendIntent } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";

export function CodeActions({
  text,
  top,
  left,
  onDone,
}: {
  text: string;
  top: number;
  left: number;
  onDone: () => void;
}) {
  const pushNotice = useStore((s) => s.pushNotice);
  const sessionId = useStore((s) => s.sessionId);
  const ref = useRef<HTMLDivElement>(null);
  const sel = text.length > 600 ? text.slice(0, 600) + "\n[cut]" : text;

  // Focus the first action when the popover opens (keyboard users land inside it).
  useEffect(() => {
    ref.current?.querySelector<HTMLButtonElement>(".codeactions__btn")?.focus();
  }, []);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      onDone();
      return;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const btns = Array.from(ref.current?.querySelectorAll<HTMLButtonElement>(".codeactions__btn") ?? []);
      const i = btns.indexOf(document.activeElement as HTMLButtonElement);
      const next = e.key === "ArrowDown" ? (i + 1) % btns.length : (i - 1 + btns.length) % btns.length;
      btns[next]?.focus();
    }
  };

  const act = (label: string, run: () => void) => {
    run();
    pushNotice({ kind: "info", code: "code", message: label });
    onDone();
  };

  return (
    <div ref={ref} className="codeactions glass" style={{ top, left }} role="menu" aria-label="Code actions" onKeyDown={onKeyDown}>
      <button className="codeactions__btn" role="menuitem" onClick={() => act("explain selection", () => void sendIntent(intent.submitTurn(sessionId, `Explain this code:\n\n${sel}`)))}>
        explain
      </button>
      <button className="codeactions__btn" role="menuitem" onClick={() => act("refactor selection", () => void sendIntent(intent.custom("inline_edit", { instruction: "refactor", selection: sel })))}>
        refactor
      </button>
      <button className="codeactions__btn" role="menuitem" onClick={() => act("write tests for selection", () => void sendIntent(intent.submitTurn(sessionId, `Write tests for this code:\n\n${sel}`)))}>
        test
      </button>
    </div>
  );
}
