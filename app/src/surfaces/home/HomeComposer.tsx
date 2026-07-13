/*
  HomeComposer.tsx — the courtyard composer: where a session begins. A single volume floating in air,
  with the workspace context chips (Local . repo . branch . worktree) above a growing textarea, and the
  instrument row below (attach, voice, model, effort, permission mode, send).

  The worktree chip creates a real git worktree (git.worktree.add via the create_worktree intent) so a
  session can be isolated on its own branch. Sending launches the turn and hands off to the Code chamber.
*/
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { sendIntent } from "../../ipc";
import { useStore } from "../../store";
import { intent } from "../../wire";
import { Icon } from "../../shell/icons";
import { Radiate } from "../../shell/Radiate";
import { pickWorkspaceFolder } from "../../shell/onboarding";

export type PermMode = "ask" | "auto" | "bypass";
const PERM_LABEL: Record<PermMode, string> = { ask: "Ask each step", auto: "Auto run", bypass: "Bypass permissions" };
const PERM_NEXT: Record<PermMode, PermMode> = { ask: "auto", auto: "bypass", bypass: "ask" };

const EFFORTS = ["Standard", "Extra", "Max"] as const;
type Effort = (typeof EFFORTS)[number];

const fmtBytes = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}MB` : n >= 1e3 ? `${Math.round(n / 1e3)}KB` : `${n}B`);

export function HomeComposer({
  onPopToCode,
  permMode,
  onPermMode,
}: {
  onPopToCode: () => void;
  permMode: PermMode;
  onPermMode: (m: PermMode) => void;
}) {
  const sessionId = useStore((s) => s.sessionId);
  const runtimeReady = useStore((s) => s.runtimeStatus === "ready");
  const manifest = useStore((s) => s.manifest);
  const home = useStore((s) => s.home);
  const pushUserMessage = useStore((s) => s.pushUserMessage);
  const pushNotice = useStore((s) => s.pushNotice);

  const [text, setText] = useState("");
  const [effort, setEffort] = useState<Effort>("Standard");
  const [files, setFiles] = useState<File[]>([]);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [addMenu, setAddMenu] = useState(false);
  const rec = useRef<{ mr: MediaRecorder; stream: MediaStream; timer: ReturnType<typeof setInterval> } | null>(null);
  const ref = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const addRef = useRef<HTMLDivElement>(null);

  // The plus button adds context, like Claude Code: a working folder (native picker) or attachments.
  const addFolder = async () => {
    setAddMenu(false);
    const path = await pickWorkspaceFolder();
    if (path) {
      void sendIntent(intent.custom("open_folder", { path }));
      pushNotice({ kind: "info", code: "folder", message: `workspace set, ${path.split("/").pop() ?? path}` });
    } else {
      pushNotice({ kind: "info", code: "folder", message: "folder picker opens in the desktop app" });
    }
  };

  // Close the add menu on an outside click or Escape.
  useEffect(() => {
    if (!addMenu) return;
    const onDown = (e: MouseEvent) => {
      if (addRef.current && !addRef.current.contains(e.target as Node)) setAddMenu(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setAddMenu(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [addMenu]);

  const model = manifest?.model?.id ?? "local model";
  const repo = home?.workspace?.repo ?? home?.workspace?.root?.split("/").pop() ?? "workspace";
  const branch = home?.workspace?.branch ?? "main";

  // Auto-grow the textarea. useLayoutEffect measures after layout so scrollHeight is accurate; runtimeReady
  // is a dep so the one field re-measures when it flips from disabled (runtime down at boot) to enabled,
  // which otherwise leaves a stale mount-time height.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    // Never measure at zero width: the placeholder would wrap to one char per line and balloon
    // scrollHeight. Clear the inline height so the CSS single-row min applies until real layout.
    if (!el.clientWidth) {
      el.style.height = "";
      return;
    }
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 220) + "px";
  }, [text, runtimeReady, recording]);

  useEffect(() => () => stopRec(false), []); // tear down a live recording on unmount // eslint-disable-line react-hooks/exhaustive-deps

  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  const stopRec = (transcribe: boolean) => {
    const r = rec.current;
    if (!r) return;
    clearInterval(r.timer);
    try {
      r.mr.stop();
    } catch {
      /* already stopped */
    }
    r.stream.getTracks().forEach((t) => t.stop());
    rec.current = null;
    setRecording(false);
    if (transcribe) pushNotice({ kind: "info", code: "voice", message: `voice ${fmt(elapsed)} captured, transcribing locally` });
    setElapsed(0);
  };

  const toggleMic = async () => {
    if (recording) return stopRec(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mr.start();
      const timer = setInterval(() => setElapsed((e) => e + 1), 1000);
      rec.current = { mr, stream, timer };
      setElapsed(0);
      setRecording(true);
    } catch {
      pushNotice({ kind: "error", code: "voice", message: "microphone unavailable" });
    }
  };

  // Attachments: accept anything (no type filter), so a multimodal model is never held back. Real upload
  // to the blob store is a backend seam; here the picker, the chips, and add/remove all work locally.
  const addFiles = (list: FileList | null) => {
    if (list && list.length) setFiles((f) => [...f, ...Array.from(list)]);
  };
  const removeFile = (i: number) => setFiles((f) => f.filter((_, j) => j !== i));

  // Send stays on the Chat page: the reply streams into the conversation right here (Claude Code style).
  // Use the pop-out button to open the same conversation in the Code chamber.
  const submit = async () => {
    const t = text.trim();
    if (!t || !runtimeReady) return;
    const attached = files.length;
    pushUserMessage(attached ? `${t}\n\n(${attached} attachment${attached === 1 ? "" : "s"})` : t);
    setText("");
    setFiles([]);
    const ack = await sendIntent(intent.submitTurn(sessionId, t));
    if (!ack.accepted) pushNotice({ kind: "error", code: "rejected", message: ack.message ?? "turn rejected" });
  };

  const createWorktree = () => {
    void sendIntent(intent.custom("create_worktree", { branch }));
    pushNotice({ kind: "info", code: "worktree", message: `worktree on ${branch}` });
  };
  const createPr = () => {
    void sendIntent(intent.custom("create_pr", { branch }));
    pushNotice({ kind: "info", code: "pr", message: `opening pull request for ${branch}` });
  };
  const switchModel = () => void sendIntent(intent.custom("switch_model", {}));
  const cycleEffort = () => {
    const next = EFFORTS[(EFFORTS.indexOf(effort) + 1) % EFFORTS.length];
    setEffort(next);
    void sendIntent(intent.custom("switch_profile", { profile: next }));
  };

  const placeholder = recording ? `listening ${fmt(elapsed)}` : runtimeReady ? "Describe a task" : "Runtime not ready";
  const armed = !!text.trim() && runtimeReady;

  return (
    <div className={"hc" + (recording ? " hc--recording" : "")}>
      <div className="hc__chips">
        <span className="hc__chip" title="Runs on this machine, offline">
          <span className="hc__dot" aria-hidden /> Local
        </span>
        <span className="hc__chip" title="Workspace">
          <Icon name="files" size={13} /> {repo}
        </span>
        <span className="hc__chip" title="Git branch">
          <Icon name="source-control" size={13} /> {branch}
        </span>
        <button className="hc__chip hc__chip--action" onClick={createWorktree} title="Create an isolated git worktree">
          <Icon name="fork" size={13} /> worktree
        </button>
        <button className="hc__chip hc__chip--pr" onClick={createPr} title="Open a pull request for this branch">
          <Icon name="source-control" size={13} /> Create PR
        </button>
      </div>

      {files.length ? (
        <div className="hc__files">
          {files.map((f, i) => (
            <span key={i} className="hc__file" title={`${f.name} (${f.type || "file"})`}>
              <span className="hc__file-name">{f.name}</span>
              <span className="hc__file-size">{fmtBytes(f.size)}</span>
              <button className="hc__file-x" onClick={() => removeFile(i)} aria-label={`Remove ${f.name}`}>
                ×
              </button>
            </span>
          ))}
        </div>
      ) : null}

      <textarea
        ref={ref}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void submit();
          }
        }}
        rows={1}
        placeholder={placeholder}
        disabled={!runtimeReady || recording}
        className="hc__input"
        aria-label="Describe a task"
      />

      <div className="hc__row">
        <div className="hc__group">
          <button className="hc__perm" type="button" onClick={() => onPermMode(PERM_NEXT[permMode])} title="Permission mode">
            {PERM_LABEL[permMode]}
          </button>
          <div className="hc__add" ref={addRef}>
            <button
              className={"hc__icon" + (addMenu ? " hc__icon--pressed" : "")}
              type="button"
              title="Add a folder or files"
              aria-label="Add context"
              aria-haspopup="menu"
              aria-expanded={addMenu}
              onClick={() => setAddMenu((v) => !v)}
            >
              <Icon name="plus" size={16} />
            </button>
            {addMenu ? (
              <div className="hc__addmenu" role="menu">
                <button className="hc__addmenu__item" role="menuitem" type="button" onClick={addFolder}>
                  <Icon name="files" size={14} />
                  Add folder
                </button>
                <button
                  className="hc__addmenu__item"
                  role="menuitem"
                  type="button"
                  onClick={() => {
                    setAddMenu(false);
                    fileRef.current?.click();
                  }}
                >
                  <Icon name="plus" size={14} />
                  Attach files
                </button>
              </div>
            ) : null}
          </div>
          <input
            ref={fileRef}
            type="file"
            multiple
            hidden
            onChange={(e) => {
              addFiles(e.target.files);
              e.currentTarget.value = "";
            }}
          />
          <button
            className={"hc__icon" + (recording ? " hc__icon--on" : "")}
            type="button"
            onClick={toggleMic}
            title={recording ? "Stop voice" : "Voice (local, no time limit)"}
            aria-label={recording ? "Stop voice" : "Voice"}
            aria-pressed={recording}
          >
            <Icon name="mic" size={15} />
          </button>
        </div>

        <div className="hc__group">
          <button className="hc__meta" type="button" onClick={switchModel} title="Switch model">
            {model}
          </button>
          <button className="hc__meta" type="button" onClick={cycleEffort} title="Reasoning effort">
            {effort}
          </button>
          <button
            className="hc__icon"
            type="button"
            onClick={onPopToCode}
            title="Open this conversation in Code (picture in picture)"
            aria-label="Open in Code"
          >
            <Icon name="pip" size={15} />
          </button>
          <button className="hc__send" type="button" onClick={() => void submit()} disabled={!armed} title="Send" aria-label="Send">
            {armed ? <Radiate size={15} active /> : <Icon name="send" size={15} />}
          </button>
        </div>
      </div>
    </div>
  );
}
