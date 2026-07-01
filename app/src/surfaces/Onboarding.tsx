/*
  Onboarding.tsx — the no-folder first-run surface. Shown until the user opens a project folder (the
  native Tauri dialog in the desktop app) or chooses to continue with the bundled sample workspace.
  Quiet concrete glass, the HIDE mark, the local-first pitch, and a compact shortcut reference. The
  choice persists (App owns the flag) so this is shown once. Esc / "sample workspace" never dead-ends.
*/
import { useState } from "react";
import { LogoMark } from "../shell/Mark";
import { useFocusTrap } from "../shell/a11y";
import { pickWorkspaceFolder, isTauri } from "../shell/onboarding";

const KEYS: [string, string][] = [
  ["Command palette", "Cmd P"],
  ["Describe the work", "Cmd I"],
  ["Toggle terminal", "Cmd J"],
  ["Toggle navigator", "Cmd B"],
];

export function Onboarding({ onChoose }: { onChoose: (folder: string | null) => void }) {
  const trapRef = useFocusTrap<HTMLDivElement>();
  const [busy, setBusy] = useState(false);

  const openFolder = async () => {
    setBusy(true);
    const path = await pickWorkspaceFolder();
    setBusy(false);
    // A Tauri build returns the chosen path; web/dev (no native dialog) returns null and we fall
    // through to the sample workspace so the primary button is never a dead end.
    onChoose(path);
  };

  return (
    <div className="onboarding-overlay" role="presentation">
      <div
        className="onboarding glass"
        role="dialog"
        aria-modal="true"
        aria-label="Welcome to HIDE"
        tabIndex={-1}
        ref={trapRef}
      >
        <LogoMark size={44} />
        <h1 className="t-display onboarding__title">Open a project to begin</h1>
        <p className="t-body onboarding__pitch">
          Your whole project, always loaded. Local first, never billed, never truncated.
        </p>

        <div className="onboarding__actions">
          <button className="gate" onClick={openFolder} disabled={busy}>
            {busy ? "Opening…" : "Open folder…"}
          </button>
          <button className="text-button" onClick={() => onChoose(null)}>
            Continue with the sample workspace
          </button>
        </div>

        {!isTauri() ? (
          <p className="onboarding__note t-label">The native folder picker opens in the desktop app.</p>
        ) : null}

        <div className="onboarding__keys">
          {KEYS.map(([name, key]) => (
            <div key={name} className="settings__row">
              <span className="settings__k">{name}</span>
              <span className="settings__v">{key}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
