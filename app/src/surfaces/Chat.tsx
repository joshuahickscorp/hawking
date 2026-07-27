/*
  Chat.tsx — the Executor's content: the shared Conversation transcript plus the steering composer. This
  is the same conversation the full-page Chat surface shows (both render <Conversation/> from one store),
  so popping between them (picture-in-picture) never loses context.

  The composer is honest about what Enter does. When no turn is running Enter starts one (submit_turn);
  while a turn IS running Enter STEERS it (the catalog `steer` capability, delivered over the reachable
  redirect_run intent) instead of quietly starting an unrelated turn. The old "Queue turn" relabel is
  gone: the host has no turn queue, so no gesture here claims one.

  Retired here (docs/hide-impl/consolidation/HIDE_CONSOLIDATION_DECISIONS.md 3.1/3.2/3.3): the dead
  Attach button (no onClick, and Chat submit drops attachments), the voice mic (it recorded audio then
  discarded it, no transcription capability exists), and the "Queue turn" relabel. Their slot now
  carries the composer's state label, which is also the submit menu, so the control count went DOWN.
*/
import { useEffect, useRef, useState, type CSSProperties, type RefObject } from "react";
import { noticeFailure, runCommand, useStore } from "../store";
import { Icon } from "../shell/icons";
import { Radiate } from "../shell/Radiate";
import type { DiffChipPatch, PlanPatch } from "./chat/parts";
import { SteerBar } from "./chat/SteerBar";
import { Conversation } from "./chat/Conversation";
import {
  actionBlockedReason,
  actionEnabled,
  actionSpec,
  composerMode,
  COMPOSER_MENU,
  defaultAction,
  keyLabel,
  MODE_HINT,
  MODE_LABEL,
  NEW_CHAT_MENU,
  runChatAction,
  type ChatActionId,
  type ComposerMode,
} from "./chat/actions";

// The empty-composer prompt. Default terse (flight-log voice); flip DREAM_BIG to restore "dream big".
const DREAM_BIG = false;
const IDLE_PLACEHOLDER = DREAM_BIG ? "dream big" : "Describe a task";

// Oracle ladder for the radiate ring: how many distinct verify stages have reported this run.
const LADDER = ["build", "typecheck", "test", "lint"];
function oracleStage(tools: { message: string }[]): number | undefined {
  const seen = new Set<number>();
  for (const t of tools) {
    const m = t.message.toLowerCase();
    LADDER.forEach((k, i) => {
      if (m.includes(k)) seen.add(i);
    });
  }
  return seen.size > 0 ? seen.size : undefined;
}

export function Chat() {
  const tools = useStore((s) => s.tools);
  const runtimeReady = useStore((s) => s.runtimeStatus === "ready");
  const runPhase = useStore((s) => s.runPhase);
  const activeRunId = useStore((s) => s.activeRunId);
  const pushUserMessage = useStore((s) => s.pushUserMessage);
  const pushNotice = useStore((s) => s.pushNotice);
  const sessionId = useStore((s) => s.sessionId);
  const plan = useStore((s) => s.projections.plan as PlanPatch | undefined);
  const diffPatch = useStore((s) => s.projections.diff_chip as DiffChipPatch | undefined);

  const [text, setText] = useState("");
  const mode = composerMode(runtimeReady, runPhase);
  const live = mode === "steer";
  const runId = activeRunId ?? plan?.run_id ?? diffPatch?.chips?.[0]?.run_id ?? "";
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // One handler for every composer gesture: Enter, the modifier keys, and each menu entry all land
  // here, so a keystroke and a menu click can never mean two different things.
  const act = async (id: ChatActionId) => {
    const t = text.trim();
    if (!actionEnabled(id, mode, !!t, !!runId, !!useStore.getState().lastCheckpointId, !!useStore.getState().lastSideChatParent))
      return;
    if (id === "start" || id === "steer") {
      pushUserMessage(t);
      setText("");
    } else if (id === "goal_set") {
      setText("");
    }
    try {
      const ack = await runChatAction(id, { sessionId, runId, text: t });
      if (!ack.accepted)
        pushNotice({ kind: "error", code: "rejected", message: ack.message ?? `${actionSpec(id).label} rejected` });
    } catch (err) {
      pushNotice({ kind: "error", code: "command", message: (err as Error).message });
    }
    inputRef.current?.focus(); // focus stays on the composer after any action completes
  };

  // Through the spine, like every other gesture: the SteerBar used to build these three Intents
  // itself, which skipped the run-scope guard and could send an empty run_id.
  const steerRun = (id: "pause_run" | "resume_run" | "cancel_run") => () =>
    void runCommand(id, { run_id: runId }).catch(noticeFailure("command"));
  const pause = steerRun("pause_run");
  const resume = steerRun("resume_run");
  const cancel = steerRun("cancel_run");

  return (
    <div className="chat-shell">
      <Conversation />
      <div className="composer-zone">
        {live ? <SteerBar phase={runPhase} onPause={pause} onResume={resume} onCancel={cancel} /> : null}
        <Composer
          inputRef={inputRef}
          text={text}
          onText={setText}
          onAct={act}
          mode={mode}
          hasRun={!!runId}
          stage={oracleStage(tools)}
        />
      </div>
    </div>
  );
}

function Composer({
  inputRef,
  text,
  onText,
  onAct,
  mode,
  hasRun,
  stage,
}: {
  inputRef: RefObject<HTMLTextAreaElement | null>;
  text: string;
  onText: (v: string) => void;
  onAct: (id: ChatActionId) => void;
  mode: ComposerMode;
  hasRun: boolean;
  stage?: number;
}) {
  const hasCheckpoint = !!useStore((s) => s.lastCheckpointId);
  // The PARENT, not the branch id: both are recorded together, and the merge needs the session the
  // side chat came from (the active session is the side chat itself by then).
  const hasSideChat = !!useStore((s) => s.lastSideChatParent);
  const [menu, setMenu] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  useDismiss(menu, wrap, () => setMenu(false));

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 180) + "px";
  }, [text, inputRef]);

  const placeholder =
    mode === "blocked" ? "Runtime not ready" : mode === "steer" ? "Steer this run" : IDLE_PLACEHOLDER;
  const armed = !!text.trim() && mode !== "blocked";
  const fallback = defaultAction(mode);

  return (
    <div className="composer">
      <div className="hc__add" ref={wrap}>
        <button
          type="button"
          className="composer__attach"
          style={MODE_BUTTON}
          aria-haspopup="menu"
          aria-expanded={menu}
          aria-label={`Submit action, ${MODE_LABEL[mode]}. Open submit menu`}
          title={`${MODE_HINT[mode]}. Click for more`}
          onClick={() => setMenu((v) => !v)}
        >
          {MODE_LABEL[mode]}
        </button>
        {menu ? (
          <ActionMenu
            ids={COMPOSER_MENU}
            reason={(id) => actionBlockedReason(id, mode, !!text.trim(), hasRun, hasCheckpoint, hasSideChat)}
            onPick={(id) => {
              setMenu(false);
              onAct(id);
            }}
          />
        ) : null}
      </div>
      <textarea
        ref={inputRef}
        value={text}
        onChange={(e) => onText(e.target.value)}
        onKeyDown={(e) => {
          // Modifier gestures mirror the catalog: Mod+Enter is submit_turn's binding, Mod+/ is steer's.
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            onAct("start");
          } else if (e.key === "/" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            onAct("steer");
          } else if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onAct(fallback);
          }
        }}
        rows={1}
        placeholder={placeholder}
        disabled={mode === "blocked"}
        className="composer__input"
      />
      <button
        className="composer__send"
        type="button"
        onClick={() => onAct(fallback)}
        disabled={!armed}
        title={`${MODE_LABEL[mode]} (Enter)`}
        aria-label={MODE_LABEL[mode]}
      >
        {mode === "steer" ? <Radiate size={16} active stage={stage} /> : <Icon name="send" size={16} />}
      </button>
    </div>
  );
}

/*
  The ONE New-chat control, shared by the docked pane header and the floating panel header, so the two
  can never drift (decision 3.1: both buttons were dead, both now resolve the same catalog commands).
  Presentation differs only by the caller's className and icon size.
*/
export function NewChatButton({ className, size }: { className: string; size: number }) {
  const sessionId = useStore((s) => s.sessionId);
  const activeRunId = useStore((s) => s.activeRunId);
  const runtimeReady = useStore((s) => s.runtimeStatus === "ready");
  const pushNotice = useStore((s) => s.pushNotice);
  const hasCheckpoint = !!useStore((s) => s.lastCheckpointId);
  // The PARENT, not the branch id: both are recorded together, and the merge needs the session the
  // side chat came from (the active session is the side chat itself by then).
  const hasSideChat = !!useStore((s) => s.lastSideChatParent);
  const [menu, setMenu] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useDismiss(menu, wrap, () => setMenu(false));

  const mode = composerMode(runtimeReady, "idle");

  const fire = async (id: ChatActionId) => {
    setMenu(false);
    try {
      const ack = await runChatAction(id, { sessionId, runId: activeRunId ?? "", text: "" });
      if (!ack.accepted)
        pushNotice({ kind: "error", code: "rejected", message: ack.message ?? `${actionSpec(id).label} rejected` });
    } catch (err) {
      pushNotice({ kind: "error", code: "command", message: (err as Error).message });
    }
    trigger.current?.focus(); // focus returns to the control that opened the menu
  };

  return (
    <div className="hc__add" ref={wrap}>
      <button
        ref={trigger}
        className={className}
        type="button"
        title="New chat, side chat, or fork"
        aria-label="New chat"
        aria-haspopup="menu"
        aria-expanded={menu}
        onClick={() => setMenu((v) => !v)}
      >
        <Icon name="plus" size={size} />
      </button>
      {menu ? (
        <ActionMenu
          ids={NEW_CHAT_MENU}
          reason={(id) => actionBlockedReason(id, mode, false, !!activeRunId, hasCheckpoint, hasSideChat)}
          onPick={(id) => void fire(id)}
          style={DROP_DOWN}
        />
      ) : null}
    </div>
  );
}

/** The shared menu body. Every entry names its catalog command and shows its shortcut. */
function ActionMenu({
  ids,
  reason,
  onPick,
  style,
}: {
  ids: ChatActionId[];
  /** Empty string means the entry can fire; anything else disables it and IS the stated reason. */
  reason: (id: ChatActionId) => string;
  onPick: (id: ChatActionId) => void;
  style?: CSSProperties;
}) {
  return (
    <div className="hc__addmenu" role="menu" style={style}>
      {ids.map((id) => {
        const spec = actionSpec(id);
        const why = reason(id);
        const on = why === "";
        return (
          <button
            key={id}
            className="hc__addmenu__item"
            role="menuitem"
            type="button"
            disabled={!on}
            aria-disabled={!on}
            style={{ opacity: on ? 1 : 0.4, cursor: on ? "pointer" : "default", gap: "var(--ma-4)" }}
            title={on ? (spec.shortcut ? `${spec.label} (${keyLabel(spec.shortcut)})` : spec.label) : `${spec.label}. ${why}`}
            onClick={() => onPick(id)}
          >
            <span style={{ flex: 1 }}>{spec.label}</span>
            {spec.shortcut ? (
              <span style={{ color: "var(--text-2)", fontSize: "var(--fs-small)" }}>{keyLabel(spec.shortcut)}</span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}

/** Close a popover on outside click or Escape (the HomeComposer add-menu pattern). */
function useDismiss(open: boolean, ref: RefObject<HTMLElement | null>, close: () => void) {
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, ref, close]);
}

// The state label reuses the retired attach slot: same pill, sized for a short word instead of a glyph.
const MODE_BUTTON: CSSProperties = {
  width: "auto",
  padding: "0 var(--ma-2)",
  fontSize: "var(--fs-label)",
  whiteSpace: "nowrap",
};

// The pane and float headers sit at the TOP of their panel, so their menu drops down, not up.
const DROP_DOWN: CSSProperties = { bottom: "auto", top: "calc(100% + var(--ma-2))", left: "auto", right: 0 };
