/*
  structure.test.tsx: the PlanCard bound to the REAL host plan projection, and the two retirements
  that go with it.

  Asserts the things a user can be lied to about here: that the card shows the host's declared
  contract (acceptance, dependencies, effects, files, owner) and live state (status, verification,
  blocker) rather than a frontend invention; that every plan verb lands on its catalog command with
  the payload crates/hide-backend host.rs handle_plan_intent parses; that a write-blocked step is
  visibly gated and refuses every effectful verb; that the DiffChipRow exposes exactly one control
  per chip (the dead in-chat Accept/Reject pair is gone); and that the inline gate uses the ONE store
  handler pair instead of a second inline copy.

  No jsdom in this project, so component assertions render through react-dom/server and interaction
  assertions go through the exported pure functions the components call.
*/
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
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

import {
  DiffChipRow,
  PLAN_ACTIONS,
  PLAN_STEP_MENU,
  PlanCard,
  StepDetail,
  planActionEnabled,
  planActionReason,
  reorderedIds,
  runPlanAction,
  stepAriaLabel,
  type PlanProjection,
  type PlanProjectionStep,
} from "./structure";
import { commandById } from "../../store";

const SESSION = "ses_plan_test";

// A two-step record exactly as crates/hide-backend plan_domain.rs emits it under suggest_only:
// a read-only investigate step that completed, and an effectful edit step that is write-blocked.
const PLAN: PlanProjection = {
  plan_id: "plan_1",
  title: "Tighten the turn loop",
  objective: "One flat loop",
  status: "active",
  autonomy: "suggest_only",
  approved: false,
  steps: [
    {
      id: "s1",
      text: "Read the kernel loop",
      status: "completed",
      dependencies: [],
      acceptance: "the loop entry point is named",
      effects: ["read_fs"],
      related_files: ["crates/hide-kernel/src/lib.rs"],
      owner_agent: "root",
      verification: "passed",
      approved: true,
      write_blocked: false,
    },
    {
      id: "s2",
      text: "Collapse the phase machine",
      status: "ready",
      dependencies: ["s1"],
      acceptance: "run_turn_core drives one loop",
      effects: ["write_fs"],
      related_files: ["crates/hide-backend/src/turn.rs"],
      owner_agent: "root",
      verification: "pending",
      approved: false,
      write_blocked: true,
    },
  ],
};

const step = (id: string): PlanProjectionStep => PLAN.steps!.find((s) => s.id === id)!;

const FAILED: PlanProjectionStep = {
  id: "s3",
  text: "Run the parity gate",
  status: "failed",
  verification: "failed",
  blocker: "acceptance not met",
  effects: ["shell"],
  write_blocked: false,
};

const src = (name: string) => readFileSync(join(__dirname, name), "utf8");

beforeEach(() => {
  sent.length = 0;
});

describe("PlanCard bound to the real plan projection", () => {
  const html = renderToStaticMarkup(createElement(PlanCard, { plan: PLAN, sessionId: SESSION }));

  it("renders every host step, in order, with the plan title and completed count", () => {
    expect(html).toContain("Read the kernel loop");
    expect(html).toContain("Collapse the phase machine");
    expect(html.indexOf("Read the kernel loop")).toBeLessThan(html.indexOf("Collapse the phase machine"));
    expect(html).toContain("Tighten the turn loop");
    expect(html).toContain("1/2"); // one completed of two
  });

  it("carries status, verification and the gate in each step's accessible name", () => {
    expect(html).toContain(stepAriaLabel(step("s1"), 0));
    expect(stepAriaLabel(step("s1"), 0)).toContain("completed");
    expect(stepAriaLabel(step("s1"), 0)).toContain("verification passed");
    expect(stepAriaLabel(step("s2"), 1)).toContain("write blocked");
  });

  it("offers Approve while the plan is unapproved, and says so once it is", () => {
    expect(html).toContain("Approve plan");
    const approved = renderToStaticMarkup(createElement(PlanCard, { plan: { ...PLAN, approved: true }, sessionId: SESSION }));
    expect(approved).not.toContain('aria-label="Approve plan"');
    expect(approved).toContain("approved");
  });

  it("renders nothing when the projection carries no steps", () => {
    expect(renderToStaticMarkup(createElement(PlanCard, { plan: { steps: [] }, sessionId: SESSION }))).toBe("");
  });
});

describe("expanded step detail", () => {
  it("shows the declared contract and the live state, not a frontend invention", () => {
    const html = renderToStaticMarkup(createElement(StepDetail, { step: step("s2") }));
    expect(html).toContain("run_turn_core drives one loop"); // acceptance
    expect(html).toContain("Verification");
    expect(html).toContain("pending");
    expect(html).toContain("s1"); // dependency
    expect(html).toContain("write_fs"); // effects
    expect(html).toContain("crates/hide-backend/src/turn.rs"); // related files
    expect(html).toContain("root"); // owner agent
  });

  it("shows a failed step's blocker and its failed verification", () => {
    const html = renderToStaticMarkup(createElement(StepDetail, { step: FAILED }));
    expect(html).toContain("acceptance not met");
    expect(html).toContain("Blocker");
    expect(html).toContain("failed");
  });

  it("explains the write block in words, not by color", () => {
    const html = renderToStaticMarkup(createElement(StepDetail, { step: step("s2") }));
    expect(html).toContain("write blocked");
    expect(html).toContain("gated by the run autonomy");
    // The gated marker differs from the blocked and failed markers by glyph AND by words.
    expect(html).not.toContain("⊗");
  });
});

describe("write blocking is preserved", () => {
  it("refuses every effectful action for a write-blocked step", () => {
    const gated = step("s2");
    for (const spec of PLAN_ACTIONS.filter((a) => a.effectful))
      expect(planActionEnabled(spec.id, gated)).toBe(false);
    expect(planActionReason("repair", gated)).toContain("write blocked");
  });

  it("still offers the read-only verbs on a gated step", () => {
    const gated = step("s2");
    expect(planActionEnabled("approve_step", gated)).toBe(true);
    expect(planActionEnabled("side_chat", gated)).toBe(true);
    expect(planActionEnabled("fork_alternative", gated)).toBe(true);
  });

  it("refuses to dispatch a gated effectful verb even if something asks for it", async () => {
    await expect(runPlanAction("repair", { sessionId: SESSION, step: step("s2") })).rejects.toThrow(/write blocked/);
    expect(sent).toHaveLength(0);
  });

  it("shows the gate badge on the gated step and not on the ungated one", () => {
    const html = renderToStaticMarkup(createElement(PlanCard, { plan: PLAN, sessionId: SESSION }));
    expect(html.match(/write blocked/g)?.length).toBe(2); // the badge + the accessible name
  });
});

describe("step availability", () => {
  it("only repairs a step whose verification failed", () => {
    expect(planActionEnabled("repair", FAILED)).toBe(true);
    expect(planActionEnabled("repair", step("s1"))).toBe(false);
  });

  it("does not re-approve an approved step, and does not edit or skip a terminal one", () => {
    expect(planActionEnabled("approve_step", step("s1"))).toBe(false);
    expect(planActionEnabled("edit", step("s1"))).toBe(false);
    expect(planActionEnabled("skip", step("s1"))).toBe(false);
    expect(planActionEnabled("edit", step("s2"))).toBe(true);
  });

  it("names a real catalog command for every menu entry", () => {
    for (const id of PLAN_STEP_MENU) {
      const spec = PLAN_ACTIONS.find((a) => a.id === id)!;
      expect(commandById(spec.command), `${id} -> ${spec.command}`).toBeTruthy();
    }
  });
});

describe("plan actions dispatch the catalog commands the host parses", () => {
  const last = () => sent[sent.length - 1];

  it("approves the whole plan with no step_id, and one step with its id", async () => {
    await runPlanAction("approve_plan", { sessionId: SESSION });
    expect(last().data).toMatchObject({ name: "approve_plan" });
    expect(last().data.payload).toEqual({ session_id: SESSION });

    await runPlanAction("approve_step", { sessionId: SESSION, step: step("s2") });
    expect(last().data.payload).toEqual({ session_id: SESSION, step_id: "s2" });
  });

  it("skips only with a reason, and sends it as the host's blocker", async () => {
    await expect(runPlanAction("skip", { sessionId: SESSION, step: step("s2") })).rejects.toThrow(/reason/);
    await expect(
      runPlanAction("skip", { sessionId: SESSION, step: step("s2"), reason: "   " }),
    ).rejects.toThrow(/reason/);
    expect(sent).toHaveLength(0);

    await runPlanAction("skip", { sessionId: SESSION, step: step("s2"), reason: "covered by s1" });
    expect(last().data).toMatchObject({ name: "skip_step" });
    expect(last().data.payload).toEqual({ session_id: SESSION, step_id: "s2", reason: "covered by s1" });
  });

  it("repairs a failed step through repair_step", async () => {
    await runPlanAction("repair", { sessionId: SESSION, step: FAILED });
    expect(last().data).toMatchObject({ name: "repair_step" });
    expect(last().data.payload).toEqual({ session_id: SESSION, step_id: "s3" });
  });

  it("edits a step's text, and refuses an empty edit", async () => {
    await expect(runPlanAction("edit", { sessionId: SESSION, step: step("s2"), text: " " })).rejects.toThrow(/text/);
    await runPlanAction("edit", { sessionId: SESSION, step: step("s2"), text: "Collapse it" });
    expect(last().data).toMatchObject({ name: "edit_plan_step" });
    expect(last().data.payload).toEqual({ session_id: SESSION, step_id: "s2", text: "Collapse it" });
  });

  it("reorders with the full permutation reorder_plan requires", async () => {
    expect(reorderedIds(PLAN.steps!, 1, -1)).toEqual(["s2", "s1"]);
    expect(reorderedIds(PLAN.steps!, 1, 1)).toEqual(["s1", "s2"]); // clamped at the end
    await runPlanAction("reorder", { sessionId: SESSION, order: reorderedIds(PLAN.steps!, 1, -1) });
    expect(last().data).toMatchObject({ name: "reorder_plan" });
    expect(last().data.payload).toEqual({ session_id: SESSION, order: ["s2", "s1"] });
  });

  it("opens a side chat about a step, and forks an alternative with fresh context", async () => {
    await runPlanAction("side_chat", { sessionId: SESSION, step: step("s2") });
    expect(last().data.payload).toEqual({ session_id: SESSION, inherit: true });
    await runPlanAction("fork_alternative", { sessionId: SESSION, step: step("s2") });
    expect(last().data.payload).toEqual({ session_id: SESSION, inherit: false });
  });
});

describe("DiffChipRow keeps exactly one control", () => {
  const chips = [{ diff_id: "d1", path: "crates/a/src/lib.rs", added: 4, removed: 1 }];

  it("renders one open/review button per chip and no accept/reject pair", () => {
    const html = renderToStaticMarkup(createElement(DiffChipRow, { chips, onOpen: () => {} }));
    expect(html.match(/<button/g)?.length).toBe(1);
    expect(html).not.toMatch(/>Accept</);
    expect(html).not.toMatch(/>Reject</);
    expect(html).toContain('aria-label="Review crates/a/src/lib.rs in the editor"');
  });

  it("has no accept/reject handler left in the source", () => {
    const s = src("structure.tsx");
    expect(s).not.toContain("onAccept");
    expect(s).not.toContain("onReject");
  });
});

describe("the inline gate uses the ONE store handler pair", () => {
  it("Conversation takes approveGate and denyGate from the store, not a second inline copy", () => {
    const s = src("Conversation.tsx");
    expect(s).toContain("useStore((s) => s.approveGate)");
    expect(s).toContain("useStore((s) => s.denyGate)");
    // No inline re-implementation: the gate intents are built in the store and nowhere else.
    expect(s).not.toContain("approve_gate");
    expect(s).not.toContain("deny_gate");
  });
});

describe("the per-step menu keeps its focus contract", () => {
  const s = src("structure.tsx");

  it("moves focus into the menu on open, so its Escape handler can fire", () => {
    // It opens from the keyboard (Shift+F10) and its Escape lived on the container, so a menu opened
    // that way could never be closed with Escape and could be stranded open.
    expect(s).toContain('querySelector<HTMLButtonElement>(".hc__addmenu__item:not([disabled])")?.focus()');
    expect(s).toContain('if (e.key === "ContextMenu" || (e.shiftKey && e.key === "F10"))');
  });

  it("closes on an outside click as well, and hands focus back to the row head", () => {
    expect(s).toContain('document.addEventListener("mousedown", onDown)');
    expect(s).toContain('document.removeEventListener("mousedown", onDown)');
    expect(s).toContain("const restore = () => headRef.current?.focus();");
  });
});
