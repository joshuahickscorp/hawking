/*
  CodeActions.tsx — highlight-to-100x. Select code in the editor and a small Liquid-Glass popover
  offers one-tap leverage: explain / refactor / test / fork & try 3. A short prompt, big value. Each
  dispatches a real intent (submit_turn / inline_edit / fleet_run); the chat/fleet surfaces pick it up.
*/
import { sendIntent } from "../../ipc";
import { useStore } from "../../store";
import { intent } from "../../wire";

const SESSION = "ses_live000000000000000000";

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
  const sel = text.length > 600 ? text.slice(0, 600) + "\n…" : text;
  const act = (label: string, run: () => void) => {
    run();
    pushNotice({ kind: "info", code: "code", message: label });
    onDone();
  };

  return (
    <div className="codeactions glass" style={{ top, left }} role="menu" aria-label="Code actions">
      <button className="codeactions__btn" role="menuitem" onClick={() => act("explain selection", () => void sendIntent(intent.submitTurn(SESSION, `Explain this code:\n\n${sel}`)))}>
        explain
      </button>
      <button className="codeactions__btn" role="menuitem" onClick={() => act("refactor selection", () => void sendIntent(intent.custom("inline_edit", { instruction: "refactor", selection: sel })))}>
        refactor
      </button>
      <button className="codeactions__btn" role="menuitem" onClick={() => act("write tests for selection", () => void sendIntent(intent.submitTurn(SESSION, `Write tests for this code:\n\n${sel}`)))}>
        test
      </button>
      <button className="codeactions__btn codeactions__btn--accent" role="menuitem" onClick={() => act("forked 3 attempts on the selection", () => void sendIntent(intent.custom("fleet_run", { task: `improve: ${text.slice(0, 60)}`, n: 3 })))}>
        ⑂ try 3
      </button>
    </div>
  );
}
