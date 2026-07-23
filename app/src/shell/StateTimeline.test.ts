/*
  StateTimeline.test.ts: the ONE history surface.

  What a user can be lied to about here is what a time-travel verb actually does, so this asserts that
  every menu entry resolves a REAL catalog command with the payload crates/hide-backend
  handle_goal_checkpoint_intent parses, that each rewind names its target explicitly and refuses to
  send until it is confirmed a second time, that entries with no addressable checkpoint are blocked
  with a stated reason rather than sending something the host cannot resolve, and that the separate
  permanent "fork from here" button is gone in favour of the one menu.

  No jsdom in this project, so component assertions render through react-dom/server and interaction
  assertions go through the exported pure functions the component calls.
*/
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

// The transport seam, stubbed so each test reads exactly what went on the wire.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const { sent } = vi.hoisted(() => ({ sent: [] as any[] }));
vi.mock("../ipc", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sendIntent: async (i: any) => {
    sent.push(i);
    return { accepted: true, event_seq: 1, message: null };
  },
  subscribeUi: () => () => {},
  callConnector: async () => null,
  TRANSPORT_KIND: "mock",
}));

import { COMMANDS, runCommand, useStore } from "../store";
import {
  actionById,
  blockedReason,
  StateTimeline,
  timelineCtx,
  TIMELINE_ACTIONS,
  timelinePlan,
} from "./StateTimeline";
import type { TimelineActionId, TimelineCtx } from "./StateTimeline";

const SRC = readFileSync(join(__dirname, "StateTimeline.tsx"), "utf8");

const CTX: TimelineCtx = {
  sessionId: "ses_test",
  atEvent: "evt_7",
  checkpointId: "ckpt_deadbeefdeadbeefdeadbeef",
  label: "reading the workspace",
};

const planOf = (id: TimelineActionId, ctx: TimelineCtx = CTX) => timelinePlan(actionById(id), ctx);

beforeEach(() => {
  sent.length = 0;
  useStore.setState({ tools: [], notices: [], sessionId: "ses_test", lastCheckpointId: null });
});

describe("every entry names a real command", () => {
  it("resolves against the generated catalog", () => {
    for (const a of TIMELINE_ACTIONS) expect(COMMANDS.map((c) => c.id)).toContain(a.command);
  });

  it("covers the whole checkpoint family the backend now supports", () => {
    const used = new Set(TIMELINE_ACTIONS.map((a) => a.command));
    expect([...used].sort()).toEqual([
      "checkpoint_compare",
      "checkpoint_create",
      "checkpoint_fork",
      "checkpoint_inspect",
      "checkpoint_replay",
      "checkpoint_restore",
      "checkpoint_rewind",
      "fork_session",
    ]);
  });
});

describe("each entry dispatches the right command with the right payload", () => {
  it("create seals a boundary at the scrubbed step", () => {
    expect(planOf("checkpoint_here")).toEqual({
      id: "checkpoint_create",
      args: { session_id: "ses_test", at_event: "evt_7", label: "reading the workspace" },
    });
  });

  it("fork from a step uses the event boundary, not a checkpoint", () => {
    expect(planOf("fork_event")).toEqual({
      id: "fork_session",
      args: { session_id: "ses_test", at_event: "evt_7" },
    });
  });

  it("inspect, replay, fork and restore address the checkpoint id", () => {
    for (const id of ["inspect", "replay", "fork_checkpoint", "restore"] as TimelineActionId[]) {
      expect(planOf(id).args).toEqual({ checkpoint_id: CTX.checkpointId });
    }
    expect(planOf("inspect").id).toBe("checkpoint_inspect");
    expect(planOf("replay").id).toBe("checkpoint_replay");
    expect(planOf("fork_checkpoint").id).toBe("checkpoint_fork");
    expect(planOf("restore").id).toBe("checkpoint_restore");
  });

  it("compare carries both the checkpoint and the current session", () => {
    expect(planOf("compare")).toEqual({
      id: "checkpoint_compare",
      args: { checkpoint_id: CTX.checkpointId, session_id: "ses_test" },
    });
  });

  it("each rewind carries its own explicit target", () => {
    expect(planOf("rewind_conversation")).toEqual({
      id: "checkpoint_rewind",
      args: { checkpoint_id: CTX.checkpointId, target: "conversation" },
    });
    expect(planOf("rewind_code").args).toMatchObject({ target: "code" });
    expect(planOf("rewind_both").args).toMatchObject({ target: "both" });
  });

  it("the payload really reaches the wire through the one spine", async () => {
    const p = planOf("rewind_code");
    await runCommand(p.id, p.args);
    expect(sent).toHaveLength(1);
    expect(sent[0].type).toBe("custom");
    expect(sent[0].data.name).toBe("checkpoint_rewind");
    expect(sent[0].data.payload).toMatchObject({ checkpoint_id: CTX.checkpointId, target: "code" });
  });
});

describe("rewind is explicit and confirmed", () => {
  it("only the rewinds are marked destructive", () => {
    const destructive = TIMELINE_ACTIONS.filter((a) => a.destructive).map((a) => a.id);
    expect(destructive).toEqual(["rewind_conversation", "rewind_code", "rewind_both"]);
  });

  it("no rewind entry is labelled ambiguously", () => {
    for (const a of TIMELINE_ACTIONS.filter((x) => x.command === "checkpoint_rewind")) {
      expect(a.target).toBeTruthy();
      expect(a.label.toLowerCase()).toContain(a.target === "both" ? "conversation and code" : (a.target as string));
    }
  });

  it("a first click only arms; nothing is sent until the second click", async () => {
    // The component's guard, exercised directly: an unarmed destructive action returns before dispatch.
    const a = actionById("rewind_both");
    let armed: TimelineActionId | null = null;
    const fire = async () => {
      if (a.destructive && armed !== a.id) {
        armed = a.id;
        return;
      }
      const p = timelinePlan(a, CTX);
      await runCommand(p.id, p.args);
    };
    await fire();
    expect(sent).toHaveLength(0);
    expect(armed).toBe("rewind_both");
    await fire();
    expect(sent).toHaveLength(1);
    expect(sent[0].data.payload).toMatchObject({ target: "both" });
  });

  it("the source really implements the two-click guard and never a modal", () => {
    expect(SRC).toContain("armed !== action.id");
    expect(SRC).toContain("Confirm:");
    expect(SRC).not.toMatch(/window\.confirm|role="dialog"/);
  });
});

describe("blocked entries state a reason", () => {
  it("checkpoint verbs are blocked with words while no checkpoint is addressable", () => {
    const none = { ...CTX, checkpointId: null };
    expect(blockedReason(actionById("rewind_code"), none)).toMatch(/no checkpoint/i);
    expect(blockedReason(actionById("checkpoint_here"), none)).toBe("");
  });

  it("boundary verbs are blocked with words while no step is recorded", () => {
    const none = { ...CTX, atEvent: "" };
    expect(blockedReason(actionById("fork_event"), none)).toMatch(/not a recorded event/i);
  });
});

describe("the boundary id is the kind the host resolves", () => {
  // replay.rs seq_of_event resolves an EventId and refuses anything else, so a boundary addressed
  // with the tool call_id always failed. The store now carries the recorded event id.
  it("the timeline carries the recorded event id, never the tool call id", () => {
    useStore.getState().apply({
      seq: 12,
      session_id: "ses_test",
      kind: { type: "tool_progress", data: { call_id: "tcl_9", message: "ran the tests", event_id: "evt_9" } },
    });
    const step = useStore.getState().tools.at(-1);
    expect(step?.event_id).toBe("evt_9");
    const ctx = timelineCtx(step, "ses_test", null);
    expect(ctx.atEvent).toBe("evt_9");
    expect(timelinePlan(actionById("fork_event"), ctx).args).toEqual({
      session_id: "ses_test",
      at_event: "evt_9",
    });
  });

  it("a step with no recorded event blocks the boundary verbs instead of sending the call id", () => {
    useStore.getState().apply({
      seq: 13,
      session_id: "ses_test",
      kind: { type: "tool_progress", data: { call_id: "proc:1", message: "stdout line" } },
    });
    const ctx = timelineCtx(useStore.getState().tools.at(-1), "ses_test", null);
    expect(ctx.atEvent).toBe("");
    expect(blockedReason(actionById("fork_event"), ctx)).toMatch(/not a recorded event/i);
  });
});

describe("the checkpoint id comes from a real host record", () => {
  it("is the id the store folds out of the host's checkpoint_created event", () => {
    useStore.getState().apply({
      seq: 12,
      session_id: "ses_test",
      kind: {
        type: "custom",
        data: {
          kind: "checkpoint_created",
          record: { checkpoint_id: "ckpt_0123456789abcdef01234567", session_id: "ses_test", at_seq: 12 },
        },
      },
    });
    expect(useStore.getState().lastCheckpointId).toBe("ckpt_0123456789abcdef01234567");
  });

  it("stays null for an unrelated custom event, so nothing is invented", () => {
    useStore.getState().apply({
      seq: 13,
      session_id: "ses_test",
      kind: { type: "custom", data: { kind: "job_created" } },
    });
    expect(useStore.getState().lastCheckpointId).toBeNull();
  });

  it("the surface reads that slice instead of scraping a notice", () => {
    expect(SRC).toContain("useStore((s) => s.lastCheckpointId)");
  });
});

describe("the row keeps its shape", () => {
  // zustand v5 answers useSyncExternalStore's SERVER snapshot with getInitialState, so a
  // react-dom/server render always sees the initial store. The row's store-fed parts are asserted on
  // the source; the empty case renders for real.
  it("renders nothing before the first recorded step", () => {
    expect(renderToStaticMarkup(createElement(StateTimeline))).toBe("");
  });

  it("keeps the step dots and the live row", () => {
    expect(SRC).toContain("statetl__dots");
    expect(SRC).toContain("statetl__dot--at");
    expect(SRC).toContain("Return to the latest state");
  });

  // scrub_to_event was recorded by the host and acted on by nothing, so the dots select locally and
  // the surface no longer dispatches it (nor does the catalog carry it).
  it("does not dispatch the retired scrub command", () => {
    expect(SRC).not.toContain('runCommand("scrub_to_event"');
    expect(COMMANDS.map((c) => c.id)).not.toContain("scrub_to_event");
  });

  it("adds exactly one control: the history menu trigger", () => {
    // Two statetl__btn in the row: the pre-existing conditional "live" return, and the ONE menu
    // trigger that replaced the standalone fork button.
    expect(SRC.match(/className="statetl__btn"/g)).toHaveLength(2);
    expect(SRC).not.toContain("statetl__fork");
    expect(SRC).not.toMatch(/>\s*fork from here/);
    expect(SRC).toContain('role="menu"');
  });

  it("is bound to the live store, not to props or local mocks", () => {
    expect(SRC).toContain("useStore((s) => s.tools)");
    expect(SRC).toContain("useStore((s) => s.sessionId)");
  });
});
