/*
  Settings.tsx — a real Settings surface (the gear opened the command palette before). A quiet glass
  panel: model + endpoint (with switch), engine/transport + the live glass path, the keyboard map, and
  an About line. Read-mostly by design; the deep config lands as the app grows. Esc or backdrop closes.
*/
import { useEffect, useState, type ReactNode } from "react";
import { sendIntent, TRANSPORT_KIND } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Icon } from "../shell/icons";
import { useFocusTrap } from "../shell/a11y";
import { checkForUpdate } from "../shell/updater";

const SHORTCUTS: [string, string][] = [
  ["Command palette", "Cmd P"],
  ["Toggle navigator", "Cmd B"],
  ["Toggle terminal", "Cmd J"],
  ["Toggle Executor", "Cmd I"],
  ["Save file", "Cmd S"],
  ["Apply diff / reject", "Tab / Esc"],
];

export function Settings({ onClose }: { onClose: () => void }) {
  const manifest = useStore((s) => s.manifest);
  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const pushNotice = useStore((s) => s.pushNotice);
  const model = manifest?.model?.id ?? "qwen2.5";
  const glass = typeof document !== "undefined" ? document.documentElement.dataset.glass ?? "frost" : "frost";
  const endpoint = (import.meta.env.VITE_HIDE_BASE as string | undefined) ?? "127.0.0.1:8744";
  const trapRef = useFocusTrap<HTMLDivElement>();

  useEffect(() => {
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [onClose]);

  const switchModel = () => {
    void sendIntent(intent.custom("switch_model", {}));
    pushNotice({ kind: "info", code: "model", message: "switch model" });
  };

  // Auto-update: the real check lights up in the desktop app (Tauri updater plugin); on web/dev it
  // degrades to a clear "managed by the desktop app" note rather than a dead button.
  const [update, setUpdate] = useState<string>("");
  const checkUpdates = async () => {
    setUpdate("checking…");
    const res = await checkForUpdate();
    if (res == null) setUpdate("updates are managed by the desktop app");
    else if (res.available) setUpdate(`update available: ${res.version ?? "newer"}`);
    else setUpdate("up to date");
  };

  return (
    <div className="gate-overlay" role="presentation" onClick={onClose}>
      <div
        className="settings glass"
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
        tabIndex={-1}
        ref={trapRef}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="settings__head">
          <span className="settings__title">Settings</span>
          <button className="icon-button" title="Close" aria-label="Close" onClick={onClose}>
            <Icon name="close" size={16} />
          </button>
        </div>

        <Section label="Model">
          <Row k="model" v={model} />
          <Row k="status" v={runtimeStatus} />
          <Row k="endpoint" v={endpoint} />
          <button className="settings__btn" onClick={switchModel}>switch model</button>
        </Section>

        <Section label="Engine">
          <Row k="transport" v={TRANSPORT_KIND} />
          <Row k="glass" v={glass === "refract" ? "refract (Chromium)" : "frost (WebKit-safe)"} />
        </Section>

        <Section label="Keyboard">
          {SHORTCUTS.map(([name, key]) => (
            <Row key={name} k={name} v={key} />
          ))}
        </Section>

        <Section label="About">
          <Row k="HIDE" v="0.1.0" />
          <button className="settings__btn" onClick={checkUpdates}>check for updates</button>
          {update ? <Row k="update" v={update} /> : null}
          <p className="settings__note">Local first. Your whole project, always loaded, never billed, never truncated.</p>
        </Section>
      </div>
    </div>
  );
}

function Section({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="settings__section">
      <div className="settings__label">{label}</div>
      {children}
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="settings__row">
      <span className="settings__k">{k}</span>
      <span className="settings__v">{v}</span>
    </div>
  );
}
