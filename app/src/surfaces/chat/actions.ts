/*
  chat/actions.ts: the composer's semantic actions, resolved through the ONE command spine.

  Every entry here names a catalog command id (src/generated/command_catalog.json, mirrored from
  crates/hide-protocol command_catalog) so a button, a modifier key, a context menu and the palette
  all mean the same thing. Nothing in this file invents a verb: an action with no CommandSpec carries
  command: null and says so, and an action with no honest host capability is simply absent.

  Deliberately ABSENT, with the reason (see docs/hide-impl/consolidation/HIDE_CONSOLIDATION_DECISIONS.md):
    queue after current step  -> `queue_turn` is reserved on the wire with NO host handler and no
                                 CommandSpec (decision section 5 = RETIRE), so Enter is only ever
                                 "start" or "steer" and there is no queue state to show.
    plan only / verify work   -> no CommandSpec for plan-only. `run_static_analysis` IS reachable
                                 now (Custom-bound, real host arm), but it is file-scoped and the
                                 composer holds no file, so it is bound where a path exists (the
                                 editor selection menu and the hunk review). `goal_evaluate` is
                                 reachable too but belongs to the goal surface, not the composer.
    attach file/code/diff/... -> submit_turn carries BlobRef attachments but there is no blob route
                                 in the app and no attach CommandSpec; the real attach flow is the
                                 HomeComposer file input (decision 3.1).
    fork from current turn    -> `fork_session` needs the EventId of a boundary STEP, which the
                                 timeline owns (StateTimeline history menu); the composer has no
                                 step selection, so offering it here would have to guess one.

  NO LONGER absent: "fork from checkpoint" is offered now that store.ts keeps `lastCheckpointId`
  (folded from the host's checkpoint_created Custom UiEvent), and it is disabled with a stated
  reason until a checkpoint actually exists. Same for "merge side chat": this file CREATES side
  chats, so it is where the merge belongs, and `merge_side_chat` had a CommandSpec and a real host
  arm (spawn_merge_side_chat) with no dispatch site anywhere in the app, meaning a side chat could
  be started and never folded back.
*/
import { runCommand, useStore } from "../../store";
import type { RunPhase } from "../../store";
import type { IntentAck } from "../../wire";

/** Phases where a turn is in flight, so the composer steers instead of starting. */
const STEERABLE = new Set<RunPhase>(["planning", "executing", "paused", "awaiting"]);

/** What pressing Enter does right now. There is no "queue" member on purpose (see the header). */
export type ComposerMode = "blocked" | "start" | "steer";

export function composerMode(ready: boolean, phase: RunPhase): ComposerMode {
  if (!ready) return "blocked";
  return STEERABLE.has(phase) ? "steer" : "start";
}

export const MODE_LABEL: Record<ComposerMode, string> = {
  blocked: "Runtime down",
  start: "Start turn",
  steer: "Steer run",
};

export const MODE_HINT: Record<ComposerMode, string> = {
  blocked: "The runtime is not ready, so Enter cannot send yet",
  start: "Enter starts a new turn",
  steer: "Enter steers the running turn. Use the menu or Mod+Enter to start a separate new turn",
};

export type ChatActionId =
  | "start"
  | "steer"
  | "goal_set"
  | "new_thread"
  | "side_chat"
  | "research_fork"
  | "checkpoint"
  | "fork_checkpoint"
  | "merge_side_chat";

export interface ChatActionSpec {
  id: ChatActionId;
  label: string;
  /** Catalog command id, or null when the host capability has no CommandSpec. */
  command: string | null;
  /** Catalog keyboard_shortcut, shown in the menu and the tooltip. */
  shortcut?: string;
  /** True when the action consumes the composer text (so it is offered only with text). */
  needsText: boolean;
}

export const CHAT_ACTIONS: ChatActionSpec[] = [
  // `steer` binds Custom("redirect_run") now (host.rs matches "redirect_run" | "steer" and raises a
  // real InterruptHub Steer), so it dispatches through the spine like everything else.
  { id: "steer", label: "Steer run now", command: "steer", shortcut: "Mod+/", needsText: true },
  { id: "start", label: "Start a separate new turn", command: "submit_turn", shortcut: "Mod+Enter", needsText: true },
  { id: "goal_set", label: "Update goal from this message", command: "goal_set", needsText: true },
  { id: "new_thread", label: "New thread", command: "new_session", needsText: false },
  { id: "side_chat", label: "Side chat, inherits this history", command: "create_side_chat", shortcut: "Mod+Shift+N", needsText: false },
  { id: "research_fork", label: "Research fork, fresh context", command: "create_side_chat", needsText: false },
  { id: "checkpoint", label: "Checkpoint this point", command: "checkpoint_create", needsText: false },
  { id: "fork_checkpoint", label: "Fork from the last checkpoint", command: "checkpoint_fork", needsText: false },
  // The other half of the side-chat pair. The composer text IS the summary the parent gains (the
  // host records a typed SideChatResult, never the child's whole transcript), so it needs text.
  { id: "merge_side_chat", label: "Merge the side chat back with this summary", command: "merge_side_chat", needsText: true },
];

export const actionSpec = (id: ChatActionId): ChatActionSpec =>
  CHAT_ACTIONS.find((a) => a.id === id) as ChatActionSpec;

/** The composer submit menu (hung off the existing submit control, not a new toolbar). */
export const COMPOSER_MENU: ChatActionId[] = [
  "steer",
  "start",
  "goal_set",
  "side_chat",
  // The merge lives on the COMPOSER menu, not the New-chat header: the typed text is the summary
  // the parent gains, and the header has no text, so it could only ever render disabled there.
  "merge_side_chat",
  "research_fork",
  "checkpoint",
];

/** The ONE New-chat menu, shared by the docked pane header and the floating panel header. */
export const NEW_CHAT_MENU: ChatActionId[] = [
  "new_thread",
  "side_chat",
  "research_fork",
  "checkpoint",
  "fork_checkpoint",
];

export interface ChatActionCtx {
  sessionId: string;
  runId: string;
  text: string;
}

/** Why an entry cannot fire right now, in words (empty string means it can). Offered-but-dead is
 *  worse than visibly unavailable, and a disabled entry that says nothing is worse than both. */
export function actionBlockedReason(
  id: ChatActionId,
  mode: ComposerMode,
  hasText: boolean,
  hasRun: boolean,
  hasCheckpoint = false,
  hasSideChat = false,
): string {
  if (mode === "blocked") return "The runtime is not ready";
  if (actionSpec(id).needsText && !hasText)
    return id === "merge_side_chat" ? "Type the summary the parent should gain" : "Type a message first";
  if (id === "steer" && !(mode === "steer" && hasRun)) return "There is no run in flight to steer";
  if (id === "fork_checkpoint" && !hasCheckpoint)
    return "No checkpoint yet. Checkpoint this point first, so there is a sealed boundary to fork";
  if (id === "merge_side_chat" && !hasSideChat)
    return "No side chat yet. Start one first, so there is a branch to fold back";
  return "";
}

/** Whether an entry can fire right now. */
export function actionEnabled(
  id: ChatActionId,
  mode: ComposerMode,
  hasText: boolean,
  hasRun: boolean,
  hasCheckpoint = false,
  hasSideChat = false,
): boolean {
  return actionBlockedReason(id, mode, hasText, hasRun, hasCheckpoint, hasSideChat) === "";
}

/** The default action for a bare Enter press. */
export const defaultAction = (mode: ComposerMode): ChatActionId => (mode === "steer" ? "steer" : "start");

/** Render a catalog shortcut for a human: Mod is Cmd on macOS, Ctrl elsewhere. */
export function keyLabel(shortcut: string): string {
  const mod = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform ?? "") ? "Cmd" : "Ctrl";
  return shortcut.replace("Mod", mod).replace("Enter", "Enter");
}

/**
 * THE dispatch point for every chat-composer and New-chat gesture. Buttons, modifier keys and both
 * menus call this, so the two New-chat headers cannot drift and no surface builds its own intent.
 * Throws (never silently no-ops) when the spine refuses, so the caller can surface a notice.
 */
export async function runChatAction(id: ChatActionId, ctx: ChatActionCtx): Promise<IntentAck> {
  switch (id) {
    case "start":
      return runCommand("submit_turn", { session_id: ctx.sessionId, text: ctx.text });

    case "steer":
      if (!ctx.runId) throw new Error("Steer run needs an active run");
      return runCommand("steer", { run_id: ctx.runId, session_id: ctx.sessionId, text: ctx.text });

    case "goal_set":
      return runCommand("goal_set", { session_id: ctx.sessionId, condition: ctx.text });

    case "new_thread":
      // Mirrors the courtyard's new-session gesture: clear the local transcript optimistically, then
      // let the host mint the real session id.
      useStore.getState().startNewSession();
      return runCommand("new_session", {});

    case "side_chat":
      return runCommand("create_side_chat", { session_id: ctx.sessionId, inherit: true });

    case "research_fork":
      // Same host capability, inherit:false, so the branch starts with no inherited history.
      return runCommand("create_side_chat", { session_id: ctx.sessionId, inherit: false });

    case "checkpoint":
      return runCommand("checkpoint_create", {
        session_id: ctx.sessionId,
        label: ctx.text.trim() || "checkpoint",
      });

    case "merge_side_chat": {
      // The id the host minted on side_chat_created; the menu keeps this entry disabled until it
      // exists, so nothing here invents a branch.
      const { lastSideChat: sideChat, lastSideChatParent: parent } = useStore.getState();
      if (!sideChat) throw new Error("Merge the side chat needs a side chat");
      // NOT ctx.sessionId: `side_chat_created` arrives under the NEW session id and the store adopts
      // it, so the active session IS the side chat and the merge would land its summary back on the
      // branch it came from. The parent the host recorded is the one the summary belongs to.
      if (!parent) throw new Error("Merge the side chat needs the session it branched from");
      return runCommand("merge_side_chat", {
        side_chat: sideChat,
        parent,
        summary: ctx.text.trim(),
      });
    }

    case "fork_checkpoint": {
      // The id the host minted on checkpoint_created; the menu keeps this entry disabled until it exists.
      const checkpointId = useStore.getState().lastCheckpointId;
      if (!checkpointId) throw new Error("Fork from the last checkpoint needs a checkpoint");
      return runCommand("checkpoint_fork", { checkpoint_id: checkpointId });
    }
  }
}
