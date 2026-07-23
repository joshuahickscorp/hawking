/*
  actions.test.ts: the chat composer's semantics as a gate.

  Asserts the three things a user can be lied to about: what Enter does right now, which catalog
  command each menu entry actually dispatches, and that the retired controls (dead Attach, mock voice
  mic, the "Queue turn" relabel, the two dead New-chat buttons) are really gone from the source.
*/
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";

// The transport seam, stubbed so each test reads exactly what went on the wire.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const { sent } = vi.hoisted(() => ({ sent: [] as any[] }));
vi.mock("../../ipc", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sendIntent: async (i: any) => {
    sent.push(i);
    return { accepted: true, event_seq: 1, message: null };
  },
  subscribeUi: () => () => {},
  callConnector: async () => null,
  TRANSPORT_KIND: "mock",
}));

import { COMMANDS, useStore } from "../../store";
import {
  actionBlockedReason,
  actionEnabled,
  actionSpec,
  CHAT_ACTIONS,
  composerMode,
  COMPOSER_MENU,
  defaultAction,
  keyLabel,
  MODE_LABEL,
  NEW_CHAT_MENU,
  runChatAction,
} from "./actions";

const CTX = { sessionId: "ses_test", runId: "run_1", text: "tighten the loop" };
const last = () => sent[sent.length - 1];

beforeEach(() => {
  sent.length = 0;
});

describe("composer state label", () => {
  it("says start when idle, steer while a turn is in flight, blocked when the runtime is down", () => {
    expect(composerMode(true, "idle")).toBe("start");
    expect(composerMode(true, "done")).toBe("start");
    expect(composerMode(true, "failed")).toBe("start");
    expect(composerMode(true, "planning")).toBe("steer");
    expect(composerMode(true, "executing")).toBe("steer");
    expect(composerMode(true, "paused")).toBe("steer");
    expect(composerMode(true, "awaiting")).toBe("steer");
    expect(composerMode(false, "executing")).toBe("blocked");
  });

  it("labels every mode and never claims a queue the host does not have", () => {
    expect(MODE_LABEL.start).toBe("Start turn");
    expect(MODE_LABEL.steer).toBe("Steer run");
    expect(MODE_LABEL.blocked).toBe("Runtime down");
    expect(Object.values(MODE_LABEL).join(" ").toLowerCase()).not.toContain("queue");
  });

  it("routes a bare Enter to the action the label names", () => {
    expect(defaultAction(composerMode(true, "idle"))).toBe("start");
    expect(defaultAction(composerMode(true, "executing"))).toBe("steer");
  });
});

describe("submit dispatch", () => {
  it("starts a turn when no run is active", async () => {
    await runChatAction("start", { ...CTX, runId: "" });
    expect(last().type).toBe("submit_turn");
    expect(last().data).toMatchObject({ session_id: "ses_test", text: "tighten the loop" });
  });

  it("steers the running turn instead of starting an unrelated one", async () => {
    await runChatAction("steer", CTX);
    // The catalog binds `steer` to Rpc(turn/steer); the reachable route to the same host capability
    // is the redirect_run custom intent (hide-backend matches both names, raises InterruptHub Steer).
    expect(last().type).toBe("custom");
    expect(last().data.name).toBe("redirect_run");
    expect(last().data.payload).toMatchObject({ run_id: "run_1", text: "tighten the loop" });
    expect(sent.some((i) => i.type === "submit_turn")).toBe(false);
  });

  it("refuses to steer with no run rather than silently starting a turn", async () => {
    await expect(runChatAction("steer", { ...CTX, runId: "" })).rejects.toThrow(/active run/);
    expect(sent).toHaveLength(0);
  });
});

describe("menu entries dispatch the right command", () => {
  it("side chat inherits history, research fork does not", async () => {
    await runChatAction("side_chat", CTX);
    expect(last().data).toMatchObject({ name: "create_side_chat", payload: { session_id: "ses_test", inherit: true } });
    await runChatAction("research_fork", CTX);
    expect(last().data).toMatchObject({ name: "create_side_chat", payload: { session_id: "ses_test", inherit: false } });
  });

  it("checkpoint marks this point, labelled from the composer text", async () => {
    await runChatAction("checkpoint", CTX);
    expect(last().data).toMatchObject({ name: "checkpoint_create", payload: { session_id: "ses_test", label: "tighten the loop" } });
    await runChatAction("checkpoint", { ...CTX, text: "  " });
    expect(last().data.payload.label).toBe("checkpoint");
  });

  it("update goal sends the message as the durable goal condition", async () => {
    await runChatAction("goal_set", CTX);
    expect(last().data).toMatchObject({ name: "goal_set", payload: { session_id: "ses_test", condition: "tighten the loop" } });
  });

  it("new thread clears the local transcript and asks the host for a fresh session", async () => {
    useStore.getState().pushUserMessage("stale");
    expect(useStore.getState().messages.length).toBeGreaterThan(0);
    await runChatAction("new_thread", { ...CTX, text: "" });
    expect(useStore.getState().messages).toHaveLength(0);
    expect(last().data.name).toBe("new_session");
  });
});

describe("the catalog is the authority", () => {
  it("every action naming a command id resolves in the generated catalog", () => {
    for (const a of CHAT_ACTIONS) {
      if (a.command === null) continue;
      expect(COMMANDS.find((c) => c.id === a.command), `${a.id} -> ${a.command}`).toBeTruthy();
    }
  });

  it("borrows its shortcuts from the catalog, never invents one", () => {
    for (const a of CHAT_ACTIONS) {
      if (!a.shortcut || a.command === null) continue;
      expect(COMMANDS.find((c) => c.id === a.command)?.keyboard_shortcut, a.id).toBe(a.shortcut);
    }
  });

  it("renders a shortcut for a human without inventing notation", () => {
    expect(keyLabel("Mod+Enter")).toMatch(/^(Cmd|Ctrl)\+Enter$/);
  });

  it("offers no entry whose capability does not exist", () => {
    const ids = [...COMPOSER_MENU, ...NEW_CHAT_MENU];
    for (const id of ids) expect(actionSpec(id), id).toBeTruthy();
    // queue_turn, plan-only, verify, attach and fork_session are deliberately absent.
    expect(CHAT_ACTIONS.map((a) => a.id)).not.toContain("queue");
    expect(CHAT_ACTIONS.map((a) => a.command)).not.toContain("queue_turn");
    expect(CHAT_ACTIONS.map((a) => a.command)).not.toContain("fork_session");
  });
});

describe("availability", () => {
  it("gates steer on a live run and text, not on hope", () => {
    expect(actionEnabled("steer", "steer", true, true)).toBe(true);
    expect(actionEnabled("steer", "steer", false, true)).toBe(false);
    expect(actionEnabled("steer", "steer", true, false)).toBe(false);
    expect(actionEnabled("steer", "start", true, true)).toBe(false);
  });

  it("gates text-consuming entries on text and everything on a live runtime", () => {
    expect(actionEnabled("start", "start", false, false)).toBe(false);
    expect(actionEnabled("goal_set", "start", true, false)).toBe(true);
    expect(actionEnabled("side_chat", "start", false, false)).toBe(true);
    expect(actionEnabled("side_chat", "blocked", true, true)).toBe(false);
  });

  it("offers fork from checkpoint only with a real checkpoint, and says why when it cannot", () => {
    expect(actionEnabled("fork_checkpoint", "start", false, false, true)).toBe(true);
    expect(actionEnabled("fork_checkpoint", "start", false, false)).toBe(false);
    expect(actionBlockedReason("fork_checkpoint", "start", false, false)).toMatch(/no checkpoint/i);
    expect(actionBlockedReason("fork_checkpoint", "start", false, false, true)).toBe("");
  });

  it("offers the merge only with a real side chat and a summary, and says why when it cannot", () => {
    expect(actionEnabled("merge_side_chat", "start", true, false, false, true)).toBe(true);
    expect(actionEnabled("merge_side_chat", "start", true, false, false, false)).toBe(false);
    expect(actionBlockedReason("merge_side_chat", "start", true, false, false, false)).toMatch(/no side chat/i);
    expect(actionBlockedReason("merge_side_chat", "start", false, false, false, true)).toMatch(/summary/i);
  });

  it("the merge folds the side chat the store recorded back onto the session it branched from", async () => {
    useStore.setState({ lastSideChat: null, lastSideChatParent: null });
    await expect(runChatAction("merge_side_chat", CTX)).rejects.toThrow(/needs a side chat/);
    // `side_chat_created` arrives under the NEW session id and the store adopts it, so the active
    // session is the side chat. The parent from the record is what the summary lands on; naming the
    // active session sent the merge to the branch itself.
    useStore.setState({ lastSideChat: "ses_side", lastSideChatParent: "ses_parent" });
    await runChatAction("merge_side_chat", { ...CTX, sessionId: "ses_side", text: "the child found the leak" });
    expect(last().data).toMatchObject({
      name: "merge_side_chat",
      payload: { side_chat: "ses_side", parent: "ses_parent", summary: "the child found the leak" },
    });
  });

  it("fork from checkpoint addresses the id the store folded, and refuses without one", async () => {
    useStore.setState({ lastCheckpointId: null });
    await expect(runChatAction("fork_checkpoint", CTX)).rejects.toThrow(/needs a checkpoint/);
    useStore.setState({ lastCheckpointId: "ckpt_42" });
    await runChatAction("fork_checkpoint", CTX);
    expect(last().data).toMatchObject({ name: "checkpoint_fork", payload: { checkpoint_id: "ckpt_42" } });
  });

  it("steer goes through the one spine now that it binds Custom, not Rpc", async () => {
    await runChatAction("steer", CTX);
    expect(last().data.name).toBe("redirect_run");
    expect(COMMANDS.find((c) => c.id === "steer")?.backend_binding).toEqual({
      kind: "custom",
      target: "redirect_run",
    });
  });
});

describe("retired controls are gone", () => {
  const read = (p: string) => readFileSync(join(__dirname, p), "utf8");
  const chat = read("../Chat.tsx");
  const steerbar = read("./SteerBar.tsx");
  const pane = read("../../shell/ChatPane.tsx");
  const float = read("../../shell/FloatingChat.tsx");
  const code = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

  it("drops the dead Attach button and the mock voice mic from the chat composer", () => {
    expect(code(chat)).not.toContain("composer__mic");
    expect(code(chat)).not.toContain("MediaRecorder");
    expect(code(chat)).not.toContain('aria-label="Attach"');
  });

  it("drops the dishonest Queue turn relabel", () => {
    expect(code(chat)).not.toContain("Queue turn");
    expect(code(chat)).not.toContain("Queue a turn");
  });

  it("retires the duplicate steer input now that the composer steers", () => {
    expect(code(steerbar)).not.toContain("Redirect this run");
    expect(code(steerbar)).not.toContain("onRedirect");
  });

  it("gives both New-chat headers the one shared control", () => {
    for (const [name, src] of [["ChatPane", pane], ["FloatingChat", float]] as const) {
      const body = code(src);
      expect(body, name).toContain("<NewChatButton");
      expect(body, name).not.toMatch(/title="New chat"/);
    }
  });
});
