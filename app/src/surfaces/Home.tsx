/*
  Home.tsx — the Chat chamber, the front door. Claude Code style: you arrive to the digest and recents,
  describe a task, and the reply streams into the conversation right here (it stays in chat). Empty shows
  the launcher (greeting, digest, recents, fleet); active shows the conversation with a Terminal / Diff /
  Preview side panel (the Claude Code active-chat panels). The composer is always at the foot. Pop-out
  opens the same conversation in the Code chamber (Cursor style), and back again.

  This and the Executor render the same <Conversation/> from one store, so they are one session with one
  context. The secret auto-compaction (see shell/autocompact) works on this conversation.
*/
import { useEffect, useMemo, useState } from "react";
import { sendIntent, TRANSPORT_KIND } from "../ipc";
import { useStore } from "../store";
import { intent } from "../wire";
import { Icon, type IconName } from "../shell/icons";
import { LogoH } from "../shell/components";
import { FleetView } from "./FleetView";
import { Conversation } from "./ChatConversation";
import { MOCK_DIFF, applyHunkStatus, parseDiff, type DiffDoc, type Hunk } from "./types";
import type { HunkAction } from "./HunkReview";
import { ChatPanel, type ChatPanelKind } from "./ChatPanel";
import { Digest } from "./Digest";
import { HomeComposer, type PermMode } from "./HomeComposer";
import { fillGreeting, nextGreetingIndex } from "./greetings";

const PANELS: { kind: ChatPanelKind; icon: IconName; label: string }[] = [
  { kind: "terminal", icon: "terminal", label: "Terminal" },
  { kind: "diff", icon: "source-control", label: "Diff" },
  { kind: "preview", icon: "globe", label: "Preview" },
  { kind: "tools", icon: "tool", label: "Tools" },
  { kind: "artifacts", icon: "box", label: "Artifacts" },
];

export function Home({
  mode,
  onMode,
  onPopToCode,
  onSettings,
  permMode,
  onPermMode,
}: {
  mode: "chat" | "code";
  onMode: (m: "chat" | "code") => void;
  onPopToCode: () => void;
  onSettings: () => void;
  permMode: PermMode;
  onPermMode: (m: PermMode) => void;
}) {
  const home = useStore((s) => s.home);
  const sessions = useStore((s) => s.sessions);
  const fleet = useStore((s) => s.fleet);
  const sessionId = useStore((s) => s.sessionId);
  const startNewSession = useStore((s) => s.startNewSession);
  const pushUserMessage = useStore((s) => s.pushUserMessage);
  const ready = useStore((s) => s.runtimeStatus === "ready");
  const diffPatch = useStore((s) => s.projections.diff);
  // Once there is a turn, the page becomes the conversation (the digest gives way to the chat).
  const hasConversation = useStore((s) => s.messages.length > 0);

  const [panel, setPanel] = useState<ChatPanelKind | null>(null);

  // The diff shown in the Diff panel: the host's proposed diff, with the mock sample as a demo fallback.
  const hostDiff = useMemo(() => parseDiff(diffPatch as Record<string, unknown> | undefined), [diffPatch]);
  const [diff, setDiff] = useState<DiffDoc | null>(null);
  useEffect(() => {
    if (hostDiff) setDiff(hostDiff);
    else if (TRANSPORT_KIND === "mock") setDiff((d) => d ?? MOCK_DIFF);
  }, [hostDiff]);
  const onDiffAct = (hunk: Hunk, action: HunkAction) => {
    setDiff((d) => (d ? applyHunkStatus(d, hunk.id, action === "accept" ? "accepted" : "rejected") : d));
    if (diff) {
      void sendIntent(action === "accept" ? intent.acceptDiff(diff.run_id, diff.diff_id) : intent.rejectDiff(diff.run_id, diff.diff_id));
    }
  };

  const name = home?.user?.name ?? "there";
  // The opening line rotates per visit (index fixed at mount, name fills reactively).
  const [greetIx] = useState(() => nextGreetingIndex());
  const greeting = fillGreeting(greetIx, name);

  // New session: reset to a blank chat so the composer is ready for a fresh task.
  const newSession = () => {
    startNewSession();
    setPanel(null);
    void sendIntent(intent.custom("new_session", {}));
  };
  // Open a recent: the conversation loads in place (stays in chat). On mock we replay the session's task
  // as a live exchange so the demo is a working chat; on a real host we ask it to rebuild the session.
  const openSession = (id: string) => {
    startNewSession();
    if (TRANSPORT_KIND === "mock") {
      const task = sessions.find((s) => s.id === id)?.title ?? "continue the session";
      pushUserMessage(task);
      void sendIntent(intent.submitTurn(sessionId, task));
    } else {
      void sendIntent(intent.custom("open_session", { session_id: id }));
    }
  };

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
        <button className="home-nav" onClick={onPopToCode}>
          <Icon name="box" size={15} /> Artifacts
        </button>
        <button className="home-nav" onClick={onSettings}>
          <Icon name="settings" size={15} /> Customize
        </button>

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
                  onClick={() => setPanel((cur) => (cur === p.kind ? null : p.kind))}
                >
                  <Icon name={p.icon} size={15} />
                </button>
              ))}
            </div>
            <Conversation onOpenDiff={onPopToCode} />
            {composer}
          </div>
          {panel ? <ChatPanel panel={panel} onClose={() => setPanel(null)} diff={diff} onDiffAct={onDiffAct} /> : null}
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
                    Parallel attempts of one task, forked and run locally. Keep the best.
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
