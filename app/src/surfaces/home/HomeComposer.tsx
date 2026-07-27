/*
  HomeComposer.tsx - the courtyard composer: where a session begins. A single volume floating in air,
  with the workspace context chips (Local . repo . branch . worktree) above a growing textarea, and
  the instrument row below (permission mode, add, model, pop out, send).

  Three consolidation landings, all inside controls that already existed:

  A. GOAL. The durable goal domain (host goal_set / goal_clear / goal_evaluate over the KV store) is
     bound to the composer, not to a new panel: the add menu binds the composer text as an acceptance
     condition, and the goal then rides in the existing chips row as one chip that runs the
     deterministic acceptance check. A task can carry a goal, so a run can stop exactly when done.

  B. ATTACHMENTS. submit_turn has always carried an attachments field on the contract and this
     composer staged File objects and dropped them at submit. They are now real BlobRefs (name, real
     content digest, size, media type), so the field stops being dead.

  C. WORKSPACE TRUST. A repo enters the host workspace graph UNTRUSTED, and while untrusted its
     instruction and policy files are inert. The add-folder flow therefore ends in an explicit trust
     decision inside the same menu. Nothing is auto-trusted.

  F. RETIRED here (decisions 3.2 and 3.4): the voice mic (it recorded, then discarded the recording,
     because no transcription capability exists) and the third `switch_model` copy (empty payload,
     log-only host-side, no model-switch capability). The model is now a plain label, and the ONE
     place that choice is presented is shell/ModelChooser. Retired with the remediation stage for the
     same reason: the "Create PR" chip (`create_pr`) and the reasoning-effort cycle
     (`switch_profile`), both custom names with no host arm and no capability behind them.
*/
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { runCommand, useStore } from "../../store";
import { Icon } from "../../shell/icons";
import { modelSwitchNote, modelId } from "../../shell/ModelChooser";
import { Radiate } from "../../shell/Radiate";
import { branchLabel } from "../../shell/StatusBar";
import { pickWorkspaceFolder } from "../../shell/onboarding";
import { useActions } from "../contextstack/state";
import { runChatAction } from "../chat/actions";
import {
  GOAL_HINT,
  TRUST_MEANING,
  goalPlan,
  repoIdFor,
  stageAttachments,
  trustPlan,
  worktreeNotice,
  type Trust,
} from "./actions";

/*
  Permission mode. RETIRED here: the third mode, "Auto run". Nothing in the app ever read it: the
  security gate is auto-approved for "bypass" and prompts for everything else (App.tsx), so "Auto
  run" was a label that changed nothing and sat between the two real modes in the cycle. Two modes
  now, and each states what it does, because one of them switches the approval gate off.
*/
export type PermMode = "ask" | "bypass";
const PERM_LABEL: Record<PermMode, string> = { ask: "Ask each step", bypass: "Bypass permissions" };
const PERM_NEXT: Record<PermMode, PermMode> = { ask: "bypass", bypass: "ask" };
const PERM_MEANING: Record<PermMode, string> = {
  ask: "Every gated step waits for your approval.",
  bypass: "Every approval gate is auto-approved, including unsandboxed commands. It never survives a restart.",
};

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
  const activeRunId = useStore((s) => s.activeRunId);
  const manifest = useStore((s) => s.manifest);
  const home = useStore((s) => s.home);
  const pushUserMessage = useStore((s) => s.pushUserMessage);
  const pushNotice = useStore((s) => s.pushNotice);

  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [addMenu, setAddMenu] = useState(false);
  // The goal the host currently holds for this session, as this composer last set it.
  const [goal, setGoal] = useState<string | null>(null);
  // A folder added in this session that still needs, or has just been given, a trust decision.
  const [repo, setRepo] = useState<{ id: string; path: string; trust: Trust | null } | null>(null);
  const ref = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const addRef = useRef<HTMLDivElement>(null);

  const actions = useActions((message) => pushNotice({ kind: "error", code: "composer", message }));

  // The plus button adds context, like Claude Code: a working folder (native picker) or attachments.
  // Adding a folder does NOT end here: the folder enters untrusted and the menu stays open on the
  // trust decision, because trusting a repo activates its instruction and policy files. The trust
  // decision IS the host call, and it carries the folder's PATH, which is what puts the repo into
  // the host workspace graph (there is no separate add-repo wire name, so an id alone addressed a
  // node that was never created). The retired `open_folder` intent only wrote a log record no
  // reader ever read; the workspace root itself is relaunched by the desktop shell, app/src-tauri.
  const addFolder = async () => {
    const path = await pickWorkspaceFolder();
    if (!path) {
      setAddMenu(false);
      pushNotice({ kind: "info", code: "folder", message: "folder picker opens in the desktop app" });
      return;
    }
    setRepo({ id: repoIdFor(path), path, trust: null });
  };

  const decideTrust = async (trust: Trust) => {
    if (!repo) return;
    const ok = await actions.run("trust", trustPlan(repo.id, repo.path, trust));
    if (ok) setRepo({ ...repo, trust });
    setAddMenu(false);
  };

  const setGoalFromText = async () => {
    const condition = text.trim();
    if (!condition) return;
    setAddMenu(false);
    if (await actions.run("goal", goalPlan.set(sessionId, condition))) setGoal(condition);
  };
  const clearGoal = async () => {
    setAddMenu(false);
    if (await actions.run("goal", goalPlan.clear(sessionId))) setGoal(null);
  };
  // The acceptance check: deterministic, graded against real verification results, never a guess.
  const evaluateGoal = () => void actions.run("goal", goalPlan.evaluate(sessionId));

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

  const model = modelId(manifest);
  const modelNote = modelSwitchNote(manifest);
  const repoName = home?.workspace?.repo ?? home?.workspace?.root?.split("/").pop() ?? "workspace";
  const branch = branchLabel(home?.workspace?.branch);

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
  }, [text, runtimeReady]);

  // Attachments: accept anything (no type filter), so a multimodal model is never held back.
  const addFiles = (list: FileList | null) => {
    if (list && list.length) setFiles((f) => [...f, ...Array.from(list)]);
  };
  const removeFile = (i: number) => setFiles((f) => f.filter((_, j) => j !== i));

  // Send stays on the Chat page: the reply streams into the conversation right here (Claude Code style).
  // Use the pop-out button to open the same conversation in the Code chamber.
  //
  // Through the spine like everything else. This used to be the ONE courtyard call that built its
  // own Intent, because store.ts intentFor("submit_turn") dropped the attachments argument; that
  // argument is threaded now, so the exception is gone with it.
  const submit = async () => {
    const t = text.trim();
    if (!t || !runtimeReady) return;
    const staged = files;
    pushUserMessage(staged.length ? `${t}\n\n(${staged.length} attachment${staged.length === 1 ? "" : "s"})` : t);
    setText("");
    setFiles([]);
    const attachments = await stageAttachments(staged);
    try {
      const ack = await runCommand("submit_turn", { session_id: sessionId, text: t, attachments });
      if (!ack.accepted) pushNotice({ kind: "error", code: "rejected", message: ack.message ?? "turn rejected" });
    } catch (err) {
      pushNotice({ kind: "error", code: "command", message: (err as Error).message });
    }
  };

  // Mod+/ is the catalog chord for `steer`, and the background-run detail in the rail tells the user
  // to steer "from the composer with Mod+/". Only the Executor's composer handled it, so in the Chat
  // chamber that instruction was false. Same shared dispatch the Executor uses, no new control.
  const steer = async () => {
    const t = text.trim();
    if (!t) return;
    if (!activeRunId) {
      pushNotice({ kind: "error", code: "steer", message: "There is no run in flight to steer" });
      return;
    }
    setText("");
    try {
      const ack = await runChatAction("steer", { sessionId, runId: activeRunId, text: t });
      if (!ack.accepted) pushNotice({ kind: "error", code: "rejected", message: ack.message ?? "steer rejected" });
    } catch (err) {
      pushNotice({ kind: "error", code: "command", message: (err as Error).message });
    }
  };

  // The ack decides what the strip says. The host holds the unsandboxed `git worktree add` at an
  // approval gate, so even an accepted request is not a finished worktree, and it says so. No branch
  // is sent: the host names the new worktree branch itself (`hide/<slug>`), so passing the branch
  // the user is standing on described an operation that never happened.
  const createWorktree = async () => {
    try {
      pushNotice(worktreeNotice(await runCommand("create_worktree")));
    } catch (err) {
      pushNotice({ kind: "error", code: "worktree", message: (err as Error).message });
    }
  };

  const placeholder = runtimeReady ? "Describe a task" : "Runtime not ready";
  const armed = !!text.trim() && runtimeReady;
  const goalState = actions.stateOf("goal");
  const goalNote = actions.messageOf("goal");

  return (
    <div className="hc">
      <div className="hc__chips">
        <span className="hc__chip" title="Runs on this machine, offline">
          <span className="hc__dot" aria-hidden /> Local
        </span>
        <span className="hc__chip" title="Workspace">
          <Icon name="files" size={13} /> {repoName}
        </span>
        <span className="hc__chip" title="Git branch">
          <Icon name="source-control" size={13} /> {branch}
        </span>
        <button className="hc__chip hc__chip--action" onClick={() => void createWorktree()} title="Create an isolated git worktree">
          <Icon name="fork" size={13} /> worktree
        </button>
        {/* RETIRED (this stage): the "Create PR" chip. It fired the `create_pr` custom name, which no
            host arm handles, and pushed a success notice without reading the ack. No pull-request
            capability exists anywhere in the backend, so the honest move is to remove the control
            rather than keep a button that only produced a toast. */}
        {/* Exists only while a goal exists. Pressing it runs the acceptance check. */}
        {goal ? (
          <button
            className="hc__chip hc__chip--action"
            onClick={evaluateGoal}
            aria-busy={goalState === "pending"}
            title={`Goal: ${goal}. ${GOAL_HINT} Press to run the acceptance check. Clear it from the add menu.`}
            aria-label={`Goal, ${goal}. Run the acceptance check.`}
          >
            <Icon name="sparkle" size={13} /> goal
            {goalState === "pending" ? ", checking" : goalState === "failed" ? ", check refused" : ""}
          </button>
        ) : null}
        {/* Exists only after a folder is added in this session, and states the security meaning. */}
        {repo ? (
          <span
            className="hc__chip"
            title={repo.trust ? TRUST_MEANING[repo.trust] : TRUST_MEANING.untrusted}
            aria-label={`Folder ${repo.id} is ${repo.trust ?? "untrusted"}. ${
              repo.trust ? TRUST_MEANING[repo.trust] : TRUST_MEANING.untrusted
            }`}
          >
            <Icon name="files" size={13} /> {repo.id} {repo.trust ?? "untrusted"}
          </span>
        ) : null}
      </div>

      {goalState === "failed" && goalNote ? (
        <div className="hc__files t-micro" role="status">
          {goalNote}
        </div>
      ) : null}

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
          if (e.key === "/" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            void steer();
          } else if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void submit();
          }
        }}
        rows={1}
        placeholder={placeholder}
        disabled={!runtimeReady}
        className="hc__input"
        aria-label="Describe a task"
      />

      <div className="hc__row">
        <div className="hc__group">
          <button
            className="hc__perm"
            type="button"
            onClick={() => onPermMode(PERM_NEXT[permMode])}
            title={`Permission mode: ${PERM_LABEL[permMode]}. ${PERM_MEANING[permMode]} Select for ${PERM_LABEL[PERM_NEXT[permMode]]}.`}
            aria-label={`Permission mode, ${PERM_LABEL[permMode]}. ${PERM_MEANING[permMode]} Select to switch to ${PERM_LABEL[PERM_NEXT[permMode]]}.`}
          >
            {PERM_LABEL[permMode]}
          </button>
          <div className="hc__add" ref={addRef}>
            <button
              className={"hc__icon" + (addMenu ? " hc__icon--pressed" : "")}
              type="button"
              title="Add a folder, files, or a goal"
              aria-label="Add context"
              aria-haspopup="menu"
              aria-expanded={addMenu}
              onClick={() => setAddMenu((v) => !v)}
            >
              <Icon name="plus" size={16} />
            </button>
            {addMenu ? (
              <div className="hc__addmenu" role="menu" aria-label="Add context">
                {repo && repo.trust === null ? (
                  <>
                    {/* The trust decision is the end of the add-folder flow, never skipped for the user. */}
                    <span className="hc__addmenu__item t-micro" role="presentation">
                      {repo.id} was added untrusted
                    </span>
                    <button
                      className="hc__addmenu__item"
                      role="menuitem"
                      type="button"
                      onClick={() => void decideTrust("trusted")}
                      title={TRUST_MEANING.trusted}
                    >
                      <Icon name="play" size={14} />
                      Trust this folder
                    </button>
                    <button
                      className="hc__addmenu__item"
                      role="menuitem"
                      type="button"
                      onClick={() => void decideTrust("untrusted")}
                      title={TRUST_MEANING.untrusted}
                    >
                      <Icon name="stop" size={14} />
                      Keep it untrusted
                    </button>
                  </>
                ) : (
                  <>
                    <button className="hc__addmenu__item" role="menuitem" type="button" onClick={() => void addFolder()}>
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
                    <button
                      className="hc__addmenu__item"
                      role="menuitem"
                      type="button"
                      disabled={!text.trim()}
                      onClick={() => void setGoalFromText()}
                      title={GOAL_HINT}
                    >
                      <Icon name="sparkle" size={14} />
                      Set goal from this message
                    </button>
                    {goal ? (
                      <button
                        className="hc__addmenu__item"
                        role="menuitem"
                        type="button"
                        onClick={() => void clearGoal()}
                        title={`Clear the goal: ${goal}`}
                      >
                        <Icon name="close" size={14} />
                        Clear goal
                      </button>
                    ) : null}
                  </>
                )}
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
        </div>

        <div className="hc__group">
          <span className="hc__meta" title={modelNote} aria-label={`Model ${model}. ${modelNote}`}>
            {model}
          </span>
          {/* RETIRED (this stage): the reasoning-effort cycle. It fired `switch_profile`, which no host
              arm handles, and nothing else in the app carried the choice, so the label moved and the
              run did not. No effort/profile capability exists to re-point it at. */}
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
