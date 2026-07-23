/*
  Settings.tsx — a real Settings surface (the gear opened the command palette before). A quiet glass
  panel: model, the workspace read and its environment, engine/transport + the live glass path, the
  keyboard map, and an About line. Esc or backdrop closes.

  Two consolidation landings here:
  - decision 3.4: the "switch model" button was the third copy of an empty `switch_model` payload
    against a host with no model-switch capability. It is replaced by the shared ModelChooser.
  - the Workspace section is the multi-repo workspace graph READ (root, repo, branch, worktrees, as
    the host home projection reports them) plus `environment_switch`, which keeps this session and
    its history and re-scopes file roots and tool permissions.
*/
import { useEffect, useState, type ReactNode } from "react";
import { TRANSPORT_KIND } from "../ipc";
import { boundShortcuts, surfaceShortcuts, useStore } from "../store";
import { Icon } from "../shell/icons";
import { ModelChooser } from "../shell/ModelChooser";
import { useFocusTrap } from "../shell/a11y";
import { checkForUpdate } from "../shell/updater";
import { useActions } from "./contextstack/state";
import { ENVIRONMENT_NOTE, environmentPlan, workspaceRows } from "./home/actions";
import { keyLabel } from "./chat/actions";

export function Settings({ onClose }: { onClose: () => void }) {
  const manifest = useStore((s) => s.manifest);
  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const pushNotice = useStore((s) => s.pushNotice);
  const home = useStore((s) => s.home);
  const sessionId = useStore((s) => s.sessionId);
  const glass = typeof document !== "undefined" ? document.documentElement.dataset.glass ?? "frost" : "frost";
  const endpoint = (import.meta.env.VITE_HIDE_BASE as string | undefined) ?? "127.0.0.1:8744";
  const trapRef = useFocusTrap<HTMLDivElement>();

  // The environment target is typed, not picked: environments are host workspace-graph nodes and no
  // projection enumerates them, so a dropdown here would be a list this app cannot honestly fill.
  const [envId, setEnvId] = useState("");
  const actions = useActions((message) => pushNotice({ kind: "error", code: "settings", message }));
  const envState = actions.stateOf("environment");
  const switchEnv = () => {
    const id = envId.trim();
    if (id) void actions.run("environment", environmentPlan(sessionId, id));
  };

  useEffect(() => {
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, [onClose]);

  // Auto-update: the real check lights up in the desktop app (Tauri updater plugin); on web/dev it
  // degrades to a clear "managed by the desktop app" note rather than a dead button.
  const [update, setUpdate] = useState<string>("");
  const checkUpdates = async () => {
    setUpdate("checking");
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
          <ModelChooser manifest={manifest} tone="panel" />
          <Row k="status" v={runtimeStatus} />
          <Row k="endpoint" v={endpoint} />
        </Section>

        <Section label="Workspace">
          {workspaceRows(home?.workspace).map(([k, v]) => (
            <Row key={k} k={k} v={v} />
          ))}
          <label className="settings__row" htmlFor="settings-env">
            <span className="settings__k">environment</span>
            <input
              id="settings-env"
              className="ctx-note-input"
              value={envId}
              onChange={(e) => setEnvId(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") switchEnv();
              }}
              placeholder="environment id"
              aria-label="Environment id from the workspace graph"
              aria-describedby="settings-env-note"
            />
          </label>
          <button
            className="settings__btn"
            onClick={switchEnv}
            disabled={!envId.trim() || envState === "pending"}
            aria-label={`Switch this session to environment ${envId.trim() || "none named"}`}
          >
            {envState === "pending" ? "switching" : "switch environment"}
          </button>
          {envState === "done" ? <Row k="environment" v="switched" /> : null}
          {envState === "failed" ? <Row k="environment" v={actions.messageOf("environment") ?? "refused"} /> : null}
          <p className="settings__note" id="settings-env-note">
            {ENVIRONMENT_NOTE}
          </p>
        </Section>

        <Section label="Engine">
          <Row k="transport" v={TRANSPORT_KIND} />
          <Row k="glass" v={glass === "refract" ? "refract (Chromium)" : "frost (WebKit-safe)"} />
        </Section>

        <Section label="Keyboard">
          {/* The bindings, READ from the command catalog (store.boundShortcuts). The hand-written
              table that used to sit here listed six literal "Cmd X" pairs: it went stale, it was
              wrong on Linux and Windows, and it was the per-surface action table the consolidation
              forbids. Surface-owned chords are shown on their own surface (the composer hint, the
              diff bar) rather than copied into a second list here. */}
          {boundShortcuts().map((b) => (
            <Row key={b.id} k={b.title} v={keyLabel(b.shortcut)} />
          ))}
          {/* Surface-owned chords: the shell does not bind them (they need a hunk, or the open
              buffer), but they ARE keyboard bindings, so they are listed with the surface that
              answers them rather than being discoverable only by guessing. */}
          {surfaceShortcuts().map((b) => (
            <Row key={b.id} k={`${b.title} (on ${b.surface})`} v={keyLabel(b.shortcut)} />
          ))}
        </Section>

        <Section label="About">
          {/* No version row: the one that stood here was the literal "0.1.0", rendered as live state
              next to genuinely derived rows, and app/package.json ships "0.0.0". Nothing in the app
              is fed a build version, so there is no honest value to bind. The update check below is
              the real read. */}
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
