import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { sent } = vi.hoisted(() => ({ sent: [] as any[] }));
vi.mock("../ipc", () => ({
  sendIntent: async (i: any) => {
    sent.push(i);
    return { accepted: true, event_seq: 1, message: null };
  },
  subscribeUi: () => () => {},
  callConnector: async () => null,
  TRANSPORT_KIND: "mock",
}));

import {
  DIFF_ACTIONS,
  HUNK_ACTIONS,
  HunkDetail,
  citeHunk,
  exportReviewReceipt,
  hunkActionSpec,
  hunkActionsFor,
  hunkBaseHash,
  hunkId,
  hunkProvenance,
  invalidatedNote,
  reviewKeysActive,
  reviewReceipt,
  reviewSurface,
  runDiffAction,
  runHunkAction,
  statusAfter,
} from './ide_HunkReview';
import {
  DISPATCHED,
  SEL_ACTIONS,
  citeRef,
  hashText,
  isStale,
  refBody,
  runSelectionAction,
  selActionSpec,
  sourceRef,
  symbolOf,
} from './ide_CodeActions';
import { commandById } from "../store";
import type { DiffDoc, Hunk } from './ide_types';

const HUNK: Hunk = {
  id: "h1",
  header: "@@ -4,4 +4,4 @@ pub async fn acquire",
  status: "pending",
  lines: [
    { kind: "ctx", text: "    if conn.is_stale() {", oldNo: 4, newNo: 4 },
    { kind: "del", text: "        drop(permit);", oldNo: 5, newNo: null },
    { kind: "add", text: "        let fresh = retry().await?;", oldNo: null, newNo: 5 },
  ],
};

const RICH = {
  ...HUNK,
  hunk_id: "hunk_7f3a",
  file: "crates/pool/src/guard.rs",
  base_hash: "b3aa11ee22",
  provenance: { plan_step: "step_2", agent: "edit.search_replace", turn: 3 },
} as Hunk;

const DOC: DiffDoc = {
  diff_id: "diff_abc",
  run_id: "run_1",
  path: "crates/pool/src/guard.rs",
  lang: "rust",
  before: "",
  after: "",
  stale: false,
  hunks: [RICH],
};

const CTX = { diffId: DOC.diff_id, runId: DOC.run_id, sessionId: "ses_test", path: DOC.path, hunk: RICH };
const last = () => sent[sent.length - 1];

beforeEach(() => {
  sent.length = 0;
});

describe("hunk addressing", () => {
  it("addresses the wire id, not the local one, and falls back when the host sent none", () => {
    expect(hunkId(RICH)).toBe("hunk_7f3a");
    expect(hunkId(HUNK)).toBe("h1");
  });

  it("carries the hunk_id on a per-hunk accept and reject", async () => {
    await runHunkAction("accept", CTX);
    expect(last().type).toBe("accept_diff");
    expect(last().data).toMatchObject({ run_id: "run_1", diff_id: "diff_abc", hunk_id: "hunk_7f3a" });

    await runHunkAction("reject", CTX);
    expect(last().type).toBe("reject_diff");
    expect(last().data.hunk_id).toBe("hunk_7f3a");
  });

  it("routes revert and re-apply to the same host verbs, so one action never wears two controls", async () => {
    await runHunkAction("revert", CTX);
    expect(last().type).toBe("reject_diff");
    expect(last().data.hunk_id).toBe("hunk_7f3a");

    await runHunkAction("reapply", CTX);
    expect(last().type).toBe("accept_diff");
    expect(last().data.hunk_id).toBe("hunk_7f3a");
  });

  it("omits the hunk_id on every whole-diff path, which is how the host reads all of it", async () => {
    await runDiffAction("accept_all", { diffId: "diff_abc", runId: "run_1" });
    expect(last().type).toBe("accept_diff");
    expect(last().data.hunk_id).toBeNull();

    await runDiffAction("revert_all", { diffId: "diff_abc", runId: "run_1" });
    expect(last().type).toBe("custom");
    expect(last().data.name).toBe("revert_diff");
    expect(last().data.payload).toMatchObject({ diff_id: "diff_abc" });
  });
});

describe("hunk actions", () => {
  it("names a real catalog command for every entry that claims one", () => {
    for (const a of [...HUNK_ACTIONS, ...DIFF_ACTIONS]) {
      if (a.command) expect(commandById(a.command), a.id).toBeTruthy();
    }
  });

  it("offers only what the current status can do", () => {
    expect(hunkActionsFor("pending")).toContain("accept");
    expect(hunkActionsFor("pending")).not.toContain("revert");
    expect(hunkActionsFor("accepted")).toContain("revert");
    expect(hunkActionsFor("accepted")).not.toContain("accept");
    expect(hunkActionsFor("rejected")).toContain("reverify");
    expect(hunkActionsFor("pending")).not.toContain("reverify");
  });

  it("flips the optimistic status only for the actions that change disk", () => {
    expect(statusAfter("accept", "pending")).toBe("accepted");
    expect(statusAfter("revert", "accepted")).toBe("rejected");
    expect(statusAfter("explain", "pending")).toBe("pending");
    expect(statusAfter("concern", "accepted")).toBe("accepted");
  });

  it("cites the hunk when it asks the agent anything", async () => {
    await runHunkAction("explain", { ...CTX, runId: "" });
    expect(last().type).toBe("submit_turn");
    expect(last().data.text).toContain(citeHunk(RICH, DOC.path));
    expect(last().data.text).toContain("drop(permit);");
  });

  it("steers a live run for an alternative, and opens a turn when nothing is running", async () => {
    await runHunkAction("alternative", CTX);
    expect(last().type).toBe("custom");
    expect(last().data.name).toBe("redirect_run");

    await runHunkAction("alternative", { ...CTX, runId: "" });
    expect(last().type).toBe("submit_turn");
  });

  it("refuses an empty concern instead of sending a blank steer", async () => {
    await expect(runHunkAction("concern", { ...CTX, note: "   " })).rejects.toThrow(/needs a note/);
    await runHunkAction("concern", { ...CTX, note: "this drops the permit twice" });
    expect(last().data.payload.text).toContain("this drops the permit twice");
  });
});

describe("provenance on screen", () => {
  const render = (status: Hunk["status"]) =>
    renderToStaticMarkup(
      createElement(HunkDetail, {
        hunk: { ...RICH, status },
        path: DOC.path,
        busy: false,
        onAct: () => {},
      }),
    );

  it("renders the originating plan step, the agent, the turn and the base hash", () => {
    const html = render("pending");
    expect(html).toContain("step_2");
    expect(html).toContain("edit.search_replace");
    expect(html).toContain("b3aa11ee22");
    expect(html).toContain("crates/pool/src/guard.rs");
  });

  it("says so when the host recorded no provenance, instead of showing a blank", () => {
    const html = renderToStaticMarkup(
      createElement(HunkDetail, { hunk: HUNK, path: DOC.path, busy: false, onAct: () => {} }),
    );
    expect(html).toContain("not recorded");
    expect(html).toContain("no originating plan step");
    expect(hunkProvenance(HUNK)).toBeNull();
    expect(hunkBaseHash(HUNK)).toBeNull();
  });

  it("shows the invalidated verification on a rejected hunk and offers the rerun", () => {
    const html = render("rejected");
    expect(html).toContain(invalidatedNote("crates/pool/src/guard.rs"));
    expect(html).toContain(hunkActionSpec("reverify").label);
  });

  it("RUNS the checks on the file the reject reverted, rather than asking the agent to", async () => {
    await runHunkAction("reverify", { ...CTX, runId: "", hunk: { ...RICH, status: "rejected" } });
    expect(last().type).toBe("custom");
    expect(last().data.name).toBe("run_static_analysis");
    expect(last().data.payload.paths).toEqual(["crates/pool/src/guard.rs"]);
  });
});

describe("review receipt", () => {
  it("keeps every hunk's provenance and base hash, and counts the statuses", () => {
    const doc = { ...DOC, hunks: [{ ...RICH, status: "accepted" as const }] };
    const r = reviewReceipt(doc);
    expect(r.counts.accepted).toBe(1);
    expect(r.hunks[0]).toMatchObject({
      hunk_id: "hunk_7f3a",
      status: "accepted",
      base_hash: "b3aa11ee22",
      provenance: { plan_step: "step_2", agent: "edit.search_replace", turn: 3 },
    });
  });

  it("exports through the writer it is given, so the export is a real effect not a toast", async () => {
    const written: string[] = [];
    await exportReviewReceipt(DOC, async (t) => void written.push(t));
    expect(written).toHaveLength(1);
    const parsed = JSON.parse(written[0]);
    expect(parsed.diff_id).toBe("diff_abc");
    expect(parsed.hunks[0].provenance.agent).toBe("edit.search_replace");
    expect(parsed.note).toMatch(/host seals/);
  });
});

describe("selection source ref", () => {
  const REF = sourceRef("crates/pool/src/guard.rs", 4, 6, "if conn.is_stale() {\n    drop(permit);\n}");
  const SEL = { sessionId: "ses_test", runId: "" };

  it("resolves to path, line range and a content hash", () => {
    expect(REF.path).toBe("crates/pool/src/guard.rs");
    expect(REF.startLine).toBe(4);
    expect(REF.endLine).toBe(6);
    expect(REF.hash).toBe(hashText(REF.text));
    expect(citeRef(REF)).toBe("crates/pool/src/guard.rs:4-6");
    expect(citeRef(sourceRef("a.rs", 9, 9, "x"))).toBe("a.rs:9");
  });

  it("detects a stale selection when the buffer moved or the range vanished", () => {
    expect(isStale(REF, REF.text)).toBe(false);
    expect(isStale(REF, REF.text + " // touched")).toBe(true);
    expect(isStale(REF, null)).toBe(true);
  });

  it("refuses to act on a stale ref rather than citing lines that moved", async () => {
    await expect(runSelectionAction("explain", REF, SEL, true)).rejects.toThrow(/changed since it was selected/);
    expect(sent).toHaveLength(0);
  });

  it("names a real catalog command for every entry that claims one", () => {
    for (const a of SEL_ACTIONS) {
      if (a.command) expect(commandById(a.command), a.id).toBeTruthy();
    }
  });

  it("dispatches the command each entry advertises", async () => {
    await runSelectionAction("explain", REF, SEL);
    expect(last().type).toBe("submit_turn");
    expect(last().data.text).toContain("crates/pool/src/guard.rs:4-6");

    await runSelectionAction("side_chat", REF, SEL);
    expect(last().type).toBe("custom");
    expect(last().data.name).toBe("create_side_chat");

    await runSelectionAction("history", REF, SEL);
    expect(last().type).toBe("run_command");
    expect(last().data.argv).toEqual([
      "git",
      "log",
      "-L",
      "4,6:crates/pool/src/guard.rs",
      "--max-count=20",
    ]);

    await runSelectionAction("attach", REF, { sessionId: "ses_test", runId: "run_1" });
    expect(last().data.name).toBe("redirect_run");
    await runSelectionAction("attach", REF, SEL);
    expect(last().type).toBe("submit_turn");
  });

  it("covers every dispatched entry in the menu", async () => {
    for (const id of DISPATCHED) {
      sent.length = 0;
      await runSelectionAction(id, REF, SEL);
      expect(sent, selActionSpec(id).label).toHaveLength(1);
    }
  });

  it("looks a reference up by symbol, not by the whole block", () => {
    expect(symbolOf("  acquire_retry(&self)")).toBe("acquire_retry");
    expect(symbolOf("   {}   ")).toBe("");
  });

  it("cuts a long selection so one turn is never a whole file", () => {
    const big = sourceRef("a.rs", 1, 400, "x".repeat(5000));
    expect(refBody(big).length).toBeLessThan(700);
    expect(refBody(big)).toContain("[cut]");
  });
});

describe("retired controls stay retired", () => {
  const read = (p: string) => readFileSync(join(__dirname, p), "utf8");

  it("no longer dispatches the log-only inline_edit, and no longer advertises fork and try 3", () => {
    const src = read("ide_CodeActions.tsx");
    expect(src).not.toContain('intent.custom("inline_edit"');
    expect(src).not.toContain('intent.custom("fleet_run"');
    const body = src.slice(src.indexOf("\nimport "));
    expect(body).not.toMatch(/inline_edit|fleet_run/);
  });

  it("lets the keyboard walk the references list, not just the action buttons", () => {
    const src = read("ide_CodeActions.tsx");
    expect(src).toContain('querySelectorAll<HTMLButtonElement>(".codeactions__btn, .search-hit")');
    expect(src).toContain("if (next >= SEL_ACTIONS.length) setHitSel(next - SEL_ACTIONS.length)");
  });

  it("keeps the diff bar at two controls in both phases", () => {
    const src = read("ide_Editor.tsx");
    expect(src.match(/<button className="diffbar__btn/g)?.length).toBe(4); // two phases, two each
  });
});

describe("the review keys are scoped, and Escape is never destructive", () => {
  const read = (p: string) => readFileSync(join(__dirname, p), "utf8");

  it("acts only while the diff surface holds focus", () => {
    expect(reviewKeysActive(false, "BUTTON")).toBe(false);
    expect(reviewKeysActive(false, undefined)).toBe(false);
    expect(reviewKeysActive(true, "BUTTON")).toBe(true);
    expect(reviewKeysActive(true, undefined)).toBe(true);
  });

  it("still leaves a text field inside the surface its own keystrokes", () => {
    expect(reviewKeysActive(true, "INPUT")).toBe(false);
    expect(reviewKeysActive(true, "TEXTAREA")).toBe(false); // Monaco's hidden input, the concern box
  });

  it("scopes to the whole diff review region when mounted in one, else to the panel", () => {
    const panel = { closest: () => null } as unknown as Element;
    const region = {} as Element;
    const inside = { closest: (sel: string) => (sel === ".diffreview" ? region : null) } as unknown as Element;
    expect(reviewSurface(inside)).toBe(region);
    expect(reviewSurface(panel)).toBe(panel);
    expect(reviewSurface(null)).toBe(null);
  });

  it("binds the window listener behind that check, not behind a tag name alone", () => {
    const src = read("ide_HunkReview.tsx");
    expect(src).toContain("reviewKeysActive(!!surface?.contains(document.activeElement)");
    expect(src).not.toContain('el.tagName === "INPUT" || el.tagName === "TEXTAREA"');
  });

  it("binds NO bare key to a whole-diff verb, and traps focus with none of them", () => {
    const src = read("ide_Editor.tsx");
    expect(src).not.toMatch(/"Escape"[\s\S]{0,120}revert_all/);
    expect(src).not.toContain('runWhole("revert_all")\n');
    expect(src).not.toMatch(/"Tab"/);
    expect(src).not.toContain("<kbd>Tab</kbd>");
    expect(src).not.toContain("<kbd>Esc</kbd>");
    expect(src).not.toContain("onKeyDown");
    expect(src).toContain('onClick={() => runWhole("revert_all")}');
    expect(src).toContain('onClick={() => runWhole("accept_all")}');
  });

  it("leaves Tab to the browser everywhere in the region, so focus can get out", () => {
    expect(read("ide_Editor.tsx")).not.toContain("preventDefault()");
    expect(read("ide_HunkReview.tsx")).not.toMatch(/case "Tab"/);
  });

  it("re-audit: every bare key still bound in the region acts on ONE hunk, or on nothing", () => {
    const src = read("ide_HunkReview.tsx");
    for (const [key, call] of [
      ["a", "act(sel, \"accept\")"],
      ["r", "act(sel, \"reject\")"],
    ] as const) {
      expect(src).toMatch(new RegExp(`case "${key}":[\\s\\S]{0,80}${call.replace(/[(),"]/g, "\\$&")}`));
    }
    expect(src).not.toMatch(/case "[a-z]":[\s\S]{0,120}(runWhole|accept_all|reject_all)/);
  });
});

describe("the editor save is a command, not a private write", () => {
  const read = (p: string) => readFileSync(join(__dirname, p), "utf8");

  it("dispatches the catalog command and sends the base_hash the concurrency guard needs", () => {
    const src = read("ide_Editor.tsx");
    expect(src).toContain('runCommand("save_file"');
    expect(src).toContain("base_hash: body.hash ?? null");
    expect(src).not.toContain('callConnector("fs", "write_file"');
    expect(commandById("save_file")?.backend_binding).toMatchObject({ kind: "custom", target: "save_file" });
    expect(commandById("save_file")?.keyboard_shortcut).toBe("Mod+S");
  });

  it("surfaces the host's own reason for a refused or held save", () => {
    const src = read("ide_Editor.tsx");
    expect(src).toContain("ack.message ?? `save refused ${openPath}`");
    expect(src).not.toContain("`save failed ${openPath}`");
  });
});
