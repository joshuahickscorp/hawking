/*
  ui.test.ts: THE one search experience.

  Asserts the things a user can be lied to about here: that the default scope follows the surface the
  search was opened from; that a transcript search really dials the host `run_search` custom name with
  the query, the structured filters and the limit hide-backend host.rs handle_search_intent parses,
  and reads the hits back off the `search_results` UiEvent; that the code-index leg sends the shape
  the connector can actually deserialize (the old `{ q, limit }` call could not) and normalizes the
  `{ results: [{ span }] }` answer; that a result resolves back to its real source (path and line, or
  session and event id); that arrows plus Enter are enough to select and act with no mouse; that
  attach and side chat land on real host capabilities; and that the palette still lists the DERIVED
  catalog entries rather than a second command list.

  No jsdom in this project, so component assertions render through react-dom/server and interaction
  assertions go through the exported pure functions the components call.
*/
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

// The transport seam, stubbed so each test reads exactly what went on the wire. `run_search` is
// answered the way hide-backend does: an accepted ack, then a `search_results` Custom UiEvent.
const { sent, calls, host } = vi.hoisted(() => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sent: [] as any[],
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  calls: [] as any[],
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  host: { hits: [] as any[], connector: null as any, fail: false },
}));

vi.mock("./ipc", () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const listeners = new Set<(ev: any) => void>();
  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    sendIntent: async (i: any) => {
      sent.push(i);
      if (i.type === "custom" && i.data.name === "run_search") {
        const query = i.data.payload.query;
        queueMicrotask(() => {
          for (const l of [...listeners])
            l({
              seq: 1,
              session_id: null,
              kind: { type: "custom", data: { kind: "search_results", query, count: host.hits.length, hits: host.hits } },
            });
        });
      }
      return { accepted: true, event_seq: 1, message: null };
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    subscribeUi: (on: any) => {
      listeners.add(on);
      return () => listeners.delete(on);
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    callConnector: async (id: string, method: string, params: any) => {
      calls.push({ id, method, params });
      if (host.fail) throw new Error("index offline");
      return host.connector;
    },
    TRANSPORT_KIND: "mock",
  };
});

import {
  CommandPalette,
  SCOPES,
  SEARCH_LIMIT,
  citeHit,
  defaultScopes,
  hitActionFor,
  hitLabel,
  hitProvenance,
  indexHits,
  nextIndex,
  referenceHits,
  runHitAction,
  searchAll,
  searchOrigin,
  searchPayload,
  setSearchOrigin,
  transcriptHits,
  type SearchHit,
} from "./ui";
import { COMMANDS, SHELL_COMMANDS, boundShortcuts, paletteCommands } from "./store";
import { keyLabel } from "./surfaces/chat/actions";
import { Explorer } from "./surfaces/ide/Explorer";

const CTX = { sessionId: "ses_test", runId: "" };

// A code-index answer in the connector's real shape (hawking-index SearchResult).
const INDEX_ANSWER = {
  results: [
    {
      span: { path: "crates/pool/src/guard.rs", range: { start_line: 42, start_col: 0, end_line: 44, end_col: 0 } },
      title: "guard",
      snippet: "pub struct  Guard {",
      score: 0.9,
      source: "lexical",
    },
  ],
};

// A hide-backend TranscriptHit, exactly as publish_search_results serializes it.
const TRANSCRIPT_ANSWER = [
  {
    session_id: "ses_other",
    event_id: "ev_7",
    seq: 7,
    kind: "agent.message",
    role: "assistant",
    snippet: "parse_port returns None on failure",
    ts: 1,
  },
];

const FILE_HIT: SearchHit = {
  key: "files:a.rs:42:0",
  scope: "files",
  title: "a.rs",
  preview: "fn main",
  path: "a.rs",
  line: 42,
};

const LOG_HIT: SearchHit = {
  key: "transcript:ev_7",
  scope: "transcript",
  title: "assistant item",
  preview: "parse_port returns None",
  session_id: "ses_other",
  event_id: "ev_7",
  role: "assistant",
};

beforeEach(() => {
  sent.length = 0;
  calls.length = 0;
  host.hits = [];
  host.connector = null;
  host.fail = false;
  setSearchOrigin("global");
});

describe("default scope by origin", () => {
  it("differs per surface and only ever names a scope with a real backend", () => {
    expect(defaultScopes("editor")).toEqual(["files", "symbols", "references"]);
    expect(defaultScopes("chat")).toEqual(["transcript", "threads", "tools"]);
    expect(defaultScopes("terminal")).toEqual(["tools"]);
    expect(defaultScopes("plan")).toEqual(["transcript", "tools"]);
    expect(defaultScopes("editor")).not.toEqual(defaultScopes("chat"));
    // global spans the whole object model.
    expect(defaultScopes("global")).toEqual(SCOPES.map((s) => s.id));
    for (const origin of ["editor", "chat", "diff", "terminal", "plan", "global"] as const)
      for (const id of defaultScopes(origin)) expect(SCOPES.some((s) => s.id === id)).toBe(true);
  });

  it("follows the surface that claimed the origin", () => {
    expect(searchOrigin()).toBe("global");
    setSearchOrigin("editor");
    expect(defaultScopes(searchOrigin())).toEqual(["files", "symbols", "references"]);
  });
});

describe("run_search dispatch", () => {
  it("carries the query, the structured filters and the limit", () => {
    expect(searchPayload("port", "transcript", "ses_1", 5)).toEqual({ query: "port", limit: 5, session_id: "ses_1" });
    // threads deliberately omits session_id so the host searches every session.
    expect(searchPayload("port", "threads", "ses_1", 5)).toEqual({ query: "port", limit: 5 });
    expect(searchPayload("port", "tools", "ses_1", 5)).toEqual({ query: "port", limit: 5, kind: "tool.result" });
  });

  it("dials the host custom name and reads the hits off the search_results event", async () => {
    host.hits = TRANSCRIPT_ANSWER;
    const out = await searchAll("parse_port", ["transcript"], { ...CTX, limit: 3 });
    expect(sent).toHaveLength(1);
    expect(sent[0]).toEqual({
      type: "custom",
      data: { name: "run_search", payload: { query: "parse_port", limit: 3, session_id: "ses_test" } },
    });
    expect(out.errors).toEqual([]);
    expect(out.hits).toHaveLength(1);
    expect(out.hits[0]).toMatchObject({ scope: "transcript", event_id: "ev_7", session_id: "ses_other", role: "assistant" });
  });

  it("defaults the limit and never asks for a semantic leg", async () => {
    host.connector = INDEX_ANSWER;
    await searchAll("guard", ["files"], CTX);
    expect(calls[0]).toEqual({
      id: "code_index",
      method: "search",
      params: {
        query: { text: "guard", limit: SEARCH_LIMIT, include_symbols: false, include_lexical: true, include_semantic: false },
      },
    });
    const symbols = calls.length;
    await searchAll("guard", ["symbols"], CTX);
    expect(calls[symbols].params.query.include_symbols).toBe(true);
    expect(calls[symbols].params.query.include_semantic).toBe(false);
  });

  it("truncates the host answer to the limit", () => {
    const many = Array.from({ length: 9 }, (_, i) => ({ ...TRANSCRIPT_ANSWER[0], event_id: `ev_${i}` }));
    expect(transcriptHits(many, "threads", 4)).toHaveLength(4);
  });

  it("surfaces a failing leg instead of swallowing it, and keeps the others", async () => {
    host.fail = true;
    const out = await searchAll("guard", ["files"], CTX);
    expect(out.hits).toEqual([]);
    expect(out.errors[0]).toContain("Files search failed");
  });

  it("runs no leg at all for a blank query", async () => {
    expect(await searchAll("   ", ["files", "transcript"], CTX)).toEqual({ hits: [], errors: [] });
    expect(calls).toEqual([]);
    expect(sent).toEqual([]);
  });
});

describe("wire normalizers", () => {
  it("reads the code index answer the connector really returns", () => {
    const hits = indexHits(INDEX_ANSWER, "files", 10);
    expect(hits).toHaveLength(1);
    expect(hits[0]).toMatchObject({ path: "crates/pool/src/guard.rs", line: 42, scope: "files" });
    expect(hits[0].preview).toBe("pub struct Guard {");
    // A bare array (the mock transport) and a flat row still normalize.
    expect(indexHits([{ path: "a.rs", line: 3, preview: "x" }], "files", 10)[0]).toMatchObject({ path: "a.rs", line: 3 });
    expect(indexHits(null, "files", 10)).toEqual([]);
  });

  it("reads reference occurrences", () => {
    const hits = referenceHits({ occurrences: [{ symbol: "Guard", file: "a.rs", range: { start_line: 9 }, role: "call" }] }, 10);
    expect(hits[0]).toMatchObject({ scope: "references", path: "a.rs", line: 9, role: "call" });
  });
});

describe("provenance", () => {
  it("resolves a hit back to its source", () => {
    expect(hitProvenance(FILE_HIT)).toBe("a.rs:42");
    expect(hitProvenance(LOG_HIT)).toBe("ev_7 in ses_other");
    // The accessible name never depends on color or position.
    expect(hitLabel(FILE_HIT)).toContain("a.rs:42");
    expect(hitLabel(LOG_HIT)).toContain("session log");
    expect(citeHit(FILE_HIT)).toBe("a.rs:42\nfn main");
  });
});

describe("keyboard navigation", () => {
  it("moves with arrows and clamps at both ends", () => {
    expect(nextIndex(3, 0, "ArrowDown")).toBe(1);
    expect(nextIndex(3, 2, "ArrowDown")).toBe(2);
    expect(nextIndex(3, 0, "ArrowUp")).toBe(0);
    expect(nextIndex(3, 1, "End")).toBe(2);
    expect(nextIndex(0, 0, "ArrowDown")).toBe(0);
    expect(nextIndex(3, 1, "a")).toBe(1);
  });

  it("maps the modifier on Enter to the action, so no row needs a button", () => {
    expect(hitActionFor({ metaKey: false, ctrlKey: false, shiftKey: false })).toBe("open");
    expect(hitActionFor({ metaKey: true, ctrlKey: false, shiftKey: false })).toBe("attach");
    expect(hitActionFor({ metaKey: false, ctrlKey: true, shiftKey: true })).toBe("side_chat");
  });

  it("selects with arrows and activates the selection with Enter", async () => {
    const hits = [FILE_HIT, LOG_HIT];
    const sel = nextIndex(hits.length, 0, "ArrowDown");
    await runHitAction(hitActionFor({ metaKey: false, ctrlKey: false, shiftKey: false }), hits[sel], CTX);
    expect(sent[0].data).toMatchObject({ name: "open_session", payload: { session_id: "ses_other" } });
  });
});

describe("acting on a result", () => {
  it("opens a file hit at its line and a log hit in its session", async () => {
    await runHitAction("open", FILE_HIT, CTX);
    expect(sent[0]).toEqual({ type: "open_file", data: { path: "a.rs", line: 42 } });
    await runHitAction("open", LOG_HIT, CTX);
    // NOT scrub_to_event: the host recorded that intent and acted on it nowhere.
    expect(sent[1].data).toMatchObject({ name: "open_session", payload: { session_id: "ses_other" } });
  });

  it("refuses honestly when a hit has no source", async () => {
    await expect(
      runHitAction("open", { ...FILE_HIT, path: undefined, session_id: undefined }, CTX),
    ).rejects.toThrow(/no source/);
  });

  it("attaches to the RUNNING turn by steering it", async () => {
    await runHitAction("attach", FILE_HIT, { sessionId: "ses_test", runId: "run_1" });
    expect(sent[0]).toEqual({
      type: "custom",
      data: {
        name: "redirect_run",
        payload: { run_id: "run_1", session_id: "ses_test", text: "Referring to a.rs:42\nfn main" },
      },
    });
  });

  it("attaches with no run in flight by opening the turn", async () => {
    await runHitAction("attach", FILE_HIT, CTX);
    expect(sent[0]).toEqual({
      type: "submit_turn",
      data: { session_id: "ses_test", text: "Referring to a.rs:42\nfn main", attachments: [] },
    });
  });

  it("starts a side chat forked AT the hit's event, and without one for a file hit", async () => {
    await runHitAction("side_chat", LOG_HIT, CTX);
    expect(sent[0]).toEqual({
      type: "custom",
      data: { name: "create_side_chat", payload: { session_id: "ses_other", inherit: true, at_event: "ev_7" } },
    });
    await runHitAction("side_chat", FILE_HIT, CTX);
    expect(sent[1].data.payload).toEqual({ session_id: "ses_test", inherit: true });
  });
});

describe("the navigator field (the second entry to the SAME engine)", () => {
  it("renders the tree with no query and announces what it searches", () => {
    const html = renderToStaticMarkup(
      createElement(Explorer, { activePath: null, onOpen: () => {} }),
    );
    expect(html).toContain('aria-label="Filter or search files, symbols and references"');
    expect(html).toContain('role="combobox"');
    expect(html).toContain('role="tree"');
    // No search runs until the user types, so the navigator costs nothing at rest.
    expect(calls.filter((c) => c.id === "code_index")).toEqual([]);
  });
});

describe("the palette", () => {
  const render = (open: boolean) =>
    renderToStaticMarkup(
      createElement(CommandPalette, {
        open,
        onClose: () => {},
        commands: (() => {
          const keys = new Map(boundShortcuts().map((b) => [b.id, b.shortcut]));
          return [...SHELL_COMMANDS, ...paletteCommands()].map((c) => ({
            id: c.id,
            label: c.title,
            shortcut: keys.get(c.id) ?? null,
            run: () => {},
          }));
        })(),
      }),
    );

  it("still lists the DERIVED catalog entries, not a second command list", () => {
    const html = render(true);
    // one local shell command and several catalog commands, by their catalog titles
    expect(html).toContain("Command Palette");
    expect(html).toContain("Cancel run");
    expect(html).toContain("New side chat");
    const derived = paletteCommands();
    expect(derived.length).toBeGreaterThan(5);
    for (const c of derived) expect(html).toContain(c.title);
  });

  it("retires the catalog's own empty-query Search row (the input is that command)", () => {
    // The retirement lives in the derivation now: run_search declares command_palette:true, and the
    // required-argument rule keeps it out because a bare gesture carries no query.
    expect(COMMANDS.find((c) => c.id === "run_search")?.command_palette).toBe(true);
    expect(paletteCommands().some((c) => c.id === "run_search")).toBe(false);
    const html = render(true);
    expect(html).not.toContain(">Search<");
  });

  it("shows the binding on the rows that have one, since the palette aggregates every command", () => {
    const html = render(true);
    // Mod+P is the shell chord for the palette itself and Mod+. is the catalog chord for cancel_run.
    expect(html).toContain(keyLabel("Mod+P"));
    expect(html).toContain(keyLabel("Mod+."));
    // and it can only show a chord that is really bound: nothing invents one per row
    for (const b of boundShortcuts()) expect(html).toContain(keyLabel(b.shortcut));
  });

  it("is a listbox with an accessible name and renders nothing when closed", () => {
    const html = render(true);
    expect(html).toContain('role="listbox"');
    expect(html).toContain('aria-label="Command palette and search"');
    expect(render(false)).toBe("");
  });
});
