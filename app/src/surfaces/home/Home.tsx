/*
  Home.tsx — the Chat chamber, the front door. Claude Code style: you arrive to the digest and recents,
  describe a task, and the reply streams into the conversation right here (it stays in chat). Empty shows
  the launcher (greeting, digest, recents, fleet); active shows the conversation with a Terminal / Diff /
  Preview side panel (the Claude Code active-chat panels). The composer is always at the foot. Pop-out
  opens the same conversation in the Code chamber (Cursor style), and back again.

  This and the Executor render the same <Conversation/> from one store, so they are one session with one
  context.
*/
import { useEffect, useMemo, useState } from "react";
import { TRANSPORT_KIND } from "../../ipc";
import { boundShortcuts, hasSessionActivity, noticeFailure, runCommand, useStore } from "../../store";
import { keyLabel } from "../chat/actions";
import { Icon, type IconName } from "../../shell/icons";
import { LogoH } from "../../shell/Mark";
import { readDiagnostics } from "../../shell/StatusBar";
import { FleetView } from "../fleet/FleetView";
import { Conversation } from "../chat/Conversation";
import { useActions } from "../contextstack/state";
import { MOCK_DIFF, applyHunkStatus, parseDiff, type DiffDoc, type HunkStatus } from "../ide/types";
import { ChatPanel, type ChatPanelKind } from "./ChatPanel";
import { Digest } from "./Digest";
import { HomeComposer, type PermMode } from "./HomeComposer";
import {
  JOB_PHASE_GLYPH,
  JOB_PHASE_LABEL,
  jobActionEnabled,
  jobLabel,
  jobPhase,
  jobPlan,
  readJobNotice,
  type JobView,
} from "./actions";
import { fillGreeting, nextGreetingIndex } from "./greetings";

const PANELS: { kind: ChatPanelKind; icon: IconName; label: string }[] = [
  { kind: "terminal", icon: "terminal", label: "Terminal" },
  { kind: "diff", icon: "source-control", label: "Diff" },
  { kind: "preview", icon: "globe", label: "Preview" },
  { kind: "tools", icon: "tool", label: "Tools" },
  { kind: "artifacts", icon: "box", label: "Artifacts" },
  // The Context Stack: what went into the window, what it cost, what was left out, and the state
  // controls (snapshot / fork / memory). One tab in a bar that already exists.
  { kind: "context", icon: "layers", label: "Context" },
];

/** The chord for Settings, read from the ONE bound-key table (empty when nothing binds it). */
const settingsChord = (): string => {
  const b = boundShortcuts().find((k) => k.id === "open.settings");
  return b ? ` (${keyLabel(b.shortcut)})` : "";
};

export function Home({
  mode,
  onMode,
  onPopToCode,
  onSettings,
  permMode,
  onPermMode,
  panel,
  onPanel,
  onClosePanel,
}: {
  mode: "chat" | "code";
  onMode: (m: "chat" | "code") => void;
  onPopToCode: () => void;
  onSettings: () => void;
  permMode: PermMode;
  onPermMode: (m: PermMode) => void;
  // The side panel is shell state now (App.tsx), so the palette rows and these icon buttons are the
  // same toggle rather than two of them.
  panel: ChatPanelKind | null;
  onPanel: (k: ChatPanelKind) => void;
  onClosePanel: () => void;
}) {
  const home = useStore((s) => s.home);
  const sessions = useStore((s) => s.sessions);
  const fleet = useStore((s) => s.fleet);
  const startNewSession = useStore((s) => s.startNewSession);
  const openSession = useStore((s) => s.openSession);
  const ready = useStore((s) => s.runtimeStatus === "ready");
  const diffPatch = useStore((s) => s.projections.diff);
  // Once the session has something to show, the page becomes the conversation (the digest gives way
  // to the chat). Shared with the palette so one rule decides where the side panels exist.
  const hasConversation = useStore(hasSessionActivity);

  // The diff shown in the Diff panel: the host's proposed diff, with the mock sample as a demo fallback.
  const hostDiff = useMemo(() => parseDiff(diffPatch as Record<string, unknown> | undefined), [diffPatch]);
  const [diff, setDiff] = useState<DiffDoc | null>(null);
  useEffect(() => {
    if (hostDiff) setDiff(hostDiff);
    else if (TRANSPORT_KIND === "mock") setDiff((d) => d ?? MOCK_DIFF);
  }, [hostDiff]);
  // Optimistic local flip ONLY. This used to also send accept_diff / reject_diff with no hunk_id,
  // which the host reads as the WHOLE diff while the panel marked a single hunk. HunkReview now owns
  // the dispatch for this panel too, so one hunk decided here is one hunk decided on disk.
  const onDiffStatus = (hunkId: string, status: HunkStatus) =>
    setDiff((d) => (d ? applyHunkStatus(d, hunkId, status) : d));

  const name = home?.user?.name ?? "there";
  // The opening line rotates per visit (index fixed at mount, name fills reactively).
  const [greetIx] = useState(() => nextGreetingIndex());
  const greeting = fillGreeting(greetIx, name);

  // New session: reset to a blank chat so the composer is ready for a fresh task.
  const newSession = () => {
    startNewSession();
    onClosePanel();
    void runCommand("new_session").catch(noticeFailure("session"));
  };
  // Open a recent: the conversation loads in place (stays in chat). The mock/live branch lives in the
  // store now, so the palette's per-recent rows open a session exactly the way this rail does.

  const composer = (
    <div className={"home-composer-zone" + (ready ? "" : " home-composer-zone--waiting")}>
      <HomeComposer onPopToCode={onPopToCode} permMode={permMode} onPermMode={onPermMode} />
    </div>
  );

  return (
    <div className="home">
      <aside className="home-rail" aria-label="Sessions">
        <div className="home-switch" role="tablist" aria-label="Chamber">
          <button
            role="tab"
            aria-selected={mode === "chat"}
            className={"home-switchbtn" + (mode === "chat" ? " home-switchbtn--on" : "")}
            onClick={() => onMode("chat")}
          >
            <Icon name="chat" size={14} /> Chat
          </button>
          <button
            role="tab"
            aria-selected={mode === "code"}
            className={"home-switchbtn" + (mode === "code" ? " home-switchbtn--on" : "")}
            onClick={() => onMode("code")}
          >
            <Icon name="split" size={14} /> Code
          </button>
        </div>
        <button className="home-new" onClick={newSession}>
          <Icon name="plus" size={15} /> New session
        </button>
        {/* The "Artifacts" rail item is RETIRED (consolidation decision 3.3): its onClick opened the
            Code chamber and no artifact store exists. The Artifacts side panel on a live conversation
            is the honest surface, and it shows what the run actually produced. */}
        {/* Chord READ from the bound-key table, like every other one in the shell. */}
        <button className="home-nav" title={`Settings${settingsChord()}`} onClick={onSettings}>
          <Icon name="settings" size={15} /> Customize
        </button>

        <BackgroundRun />

        <div className="home-recents">
          <div className="t-label home-recents__head">Recents</div>
          {sessions.length ? (
            <ul className="home-recents__list">
              {sessions.map((s) => (
                <li key={s.id}>
                  <button className="home-recent" onClick={() => openSession(s.id)} title={s.title}>
                    <span className={"home-recent__dot" + (s.state === "active" ? " home-recent__dot--live" : "")} aria-hidden />
                    <span className="home-recent__title">{s.title}</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="home-recents__empty t-micro">No sessions yet</div>
          )}
        </div>

        <div className="home-id">
          <span className="home-id__avatar" aria-hidden>
            {(name[0] ?? "?").toUpperCase()}
          </span>
          <span className="home-id__name">{name}</span>
          {home?.user?.plan ? <span className="home-id__plan">{home.user.plan}</span> : null}
        </div>
      </aside>

      {hasConversation ? (
        <main className="home-stage home-stage--live">
          <div className="home-convo">
            <div className="home-panelbar" role="tablist" aria-label="Panels">
              {PANELS.map((p) => (
                <button
                  key={p.kind}
                  role="tab"
                  aria-selected={panel === p.kind}
                  aria-label={p.label}
                  className={"home-panelbtn" + (panel === p.kind ? " home-panelbtn--on" : "")}
                  title={p.label}
                  onClick={() => onPanel(p.kind)}
                >
                  <Icon name={p.icon} size={15} />
                </button>
              ))}
            </div>
            <Conversation onOpenDiff={onPopToCode} />
            {composer}
          </div>
          {panel ? <ChatPanel panel={panel} onClose={onClosePanel} diff={diff} onDiffStatus={onDiffStatus} /> : null}
        </main>
      ) : (
        <main className="home-stage">
          <div className="home-scroll">
            <div className="home-hero">
              <span className="home-hero__mark" aria-hidden>
                <LogoH size={20} />
              </span>
              <h1 className="t-display home-hero__title">{greeting}</h1>
            </div>
            {/* Live work outranks retrospective stats: when agents are running, the fleet sits above the
                digest so it is seen first instead of buried below a tall card at the fold. */}
            {fleet.length > 0 ? (
              <section className="home-fleet" aria-label="Running agents">
                <div className="home-fleet__head">
                  <span className="t-label">Running</span>
                  <span className="home-fleet__hint t-micro">
                    Parallel attempts of one task, forked and run locally. Stop the ones you do not want.
                  </span>
                </div>
                <FleetView />
              </section>
            ) : null}
            <Digest digest={home?.digest ?? null} />
          </div>
          {composer}
        </main>
      )}
    </div>
  );
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/*
  BackgroundRun: background work inside the EXISTING sessions rail, not a second orchestration
  product. One collapsed line while a run exists, and everything else behind progressive disclosure,
  so an idle courtyard is byte-for-byte the courtyard it was. Zero permanent controls: the whole
  block unmounts when there is no run and no job.

  Every value in it is real store state: the run phase and run id from the turn projection, the
  pending approval from the security gate, the current verification from the same `diagnostics`
  projection the status bar binds, the process line from the tool stream, and the durable job id from
  the host's own job lifecycle event.

  Steer and fork are deliberately NOT repeated here: `redirect_run` already steers by run id from the
  chat composer (Mod+/), and `fork_session` needs an EventId that only the State Timeline holds.
*/
function BackgroundRun() {
  const sessionId = useStore((s) => s.sessionId);
  const activeRunId = useStore((s) => s.activeRunId);
  const runPhase = useStore((s) => s.runPhase);
  const gate = useStore((s) => s.gate);
  const tools = useStore((s) => s.tools);
  const diagnostics = useStore((s) => s.projections.diagnostics);
  const notices = useStore((s) => s.notices);
  const pushNotice = useStore((s) => s.pushNotice);
  const [open, setOpen] = useState(false);
  const [job, setJob] = useState<{ jobId: string; label: string } | null>(null);
  const actions = useActions((message) => pushNotice({ kind: "error", code: "job", message }));

  // The host publishes its job lifecycle as Custom UiEvents and store.ts has no job slice, so they
  // land in the notices strip as truncated JSON. Read the job id and the event back out of it.
  useEffect(() => {
    for (const n of notices) {
      if (n.code !== "custom") continue;
      const seen = readJobNotice(n.message);
      if (seen) setJob({ jobId: seen.jobId, label: seen.label });
    }
  }, [notices]);

  const d = readDiagnostics(diagnostics);
  const view: JobView = {
    phase: jobPhase(runPhase, !!gate),
    jobId: job?.jobId ?? null,
    jobEvent: job?.label ?? null,
    approval: gate?.message ?? null,
    verification: d ? `${plural(d.errors, "error")}, ${plural(d.warnings, "warning")}` : null,
    process: tools[tools.length - 1]?.message ?? null,
  };
  const runId = activeRunId ?? "";
  if (!runId && !view.jobId) return null;

  const rows: { id: "promote" | "pause" | "resume" | "stop" | "foreground"; label: string; title: string }[] = [
    { id: "promote", label: "Run in background", title: "Promote this run to a durable background job. It keeps running, no restart." },
    { id: "pause", label: "Pause", title: "Hold the run where it is." },
    { id: "resume", label: "Resume", title: "Continue a paused run." },
    { id: "foreground", label: "Resume in foreground", title: "Reattach the background job to this window." },
    { id: "stop", label: "Stop this run", title: "Cancel the run. Work already applied to the tree stays applied." },
  ];

  const fire = (id: (typeof rows)[number]["id"]) => {
    const plan =
      id === "promote"
        ? jobPlan.promote(runId, sessionId)
        : id === "pause"
          ? jobPlan.pause(runId)
          : id === "resume"
            ? jobPlan.resume(runId)
            : id === "stop"
              ? jobPlan.stop(runId)
              : jobPlan.foreground(view.jobId ?? "");
    void actions.run(id, plan);
  };

  return (
    <div className="home-recents">
      <div className="t-label home-recents__head">Background</div>
      <ul className="home-recents__list">
        <li>
          <button
            className="home-recent"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
            title={jobLabel(view)}
            aria-label={`${jobLabel(view)}. Select to ${open ? "hide" : "show"} detail and controls.`}
          >
            <span className={"home-recent__dot" + (view.phase === "active" ? " home-recent__dot--live" : "")} aria-hidden />
            <span className="home-recent__title">
              {JOB_PHASE_GLYPH[view.phase]} {JOB_PHASE_LABEL[view.phase]}
            </span>
          </button>
          {open ? (
            <div className="home-recents__empty t-micro" role="group" aria-label="Background run detail">
              <div>{view.jobId ? `job ${view.jobId}, ${view.jobEvent}` : "foreground only, not promoted yet"}</div>
              {view.approval ? <div>approval needed, {view.approval}</div> : null}
              <div>verification, {view.verification ?? "no static analysis has run"}</div>
              <div>process, {view.process ?? "no step reported yet"}</div>
              <div>Steer this run from the composer with Mod+/. Fork it from the state timeline.</div>
              {rows.map((r) => {
                const state = actions.stateOf(r.id);
                return (
                  <button
                    key={r.id}
                    className="settings__btn"
                    type="button"
                    title={r.title}
                    aria-busy={state === "pending"}
                    disabled={!jobActionEnabled(r.id, view, !!runId)}
                    aria-label={`${r.label}${state === "failed" ? `, failed, ${actions.messageOf(r.id) ?? ""}` : state === "done" ? ", done" : ""}`}
                    onClick={() => fire(r.id)}
                  >
                    {r.label}
                    {state === "pending" ? ", working" : state === "failed" ? ", failed" : state === "done" ? ", done" : ""}
                  </button>
                );
              })}
            </div>
          ) : null}
        </li>
      </ul>
    </div>
  );
}
