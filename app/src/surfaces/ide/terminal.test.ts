/*
  terminal.test.ts: the terminal as a session-aware process surface.

  What a user can be lied to about here, and what these assert instead: that a typed line really
  reaches the host's SANDBOXED run path through the one command spine (not a private exec), that
  streamed output is appended incrementally row by row rather than echoed once, that a process the
  user navigated away from keeps streaming and re-attaches on the way back, that the interrupt is a
  real host call, and that the state row states sandbox, process, and exit state in words. Also that
  this surface dispatches ONLY catalog commands, so it cannot grow a control the host cannot serve.

  No jsdom in this project, so the component assertions render through react-dom/server and the
  wiring assertions read the source.
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

import { COMMANDS, useStore, type ToolEvent } from "../../store";
import {
  foldProcesses,
  interruptProcess,
  latestProc,
  newSince,
  runTerminalLine,
  stateRow,
  TerminalStateBar,
} from "./Terminal";

const SRC = readFileSync(join(__dirname, "Terminal.tsx"), "utf8");

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const apply = (kind: any, seq = 1) =>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (useStore.getState().apply as any)({ seq, session_id: "ses_term", kind });

const line = (call_id: string, message: string, seq = 1) =>
  apply({ type: "tool_progress", data: { call_id, message } }, seq);

const tools = () => useStore.getState().tools;

const row = (opts: Parameters<typeof stateRow>[1]) => stateRow(tools(), opts);
const BASE = { pending: null, root: "/w/hawking", repo: "hawking", branch: "main", session: "ses_term" };

beforeEach(() => {
  sent.length = 0;
  useStore.setState({ tools: [], toolSeq: 0, home: null, sessionId: "ses_term" });
});

describe("running a command", () => {
  it("dispatches the host RunCommand path (sandbox confined), not a private exec", async () => {
    useStore.setState({ home: { workspace: { root: "/w/hawking" } } });
    const ack = await runTerminalLine("cargo  test --lib");
    expect(ack.ok).toBe(true);
    expect(sent).toEqual([
      { type: "run_command", data: { argv: ["cargo", "test", "--lib"], cwd: "/w/hawking" } },
    ]);
    // The confinement lives in the host RunCommand path; this surface must not build its own.
    expect(SRC).not.toMatch(/sendIntent\(/);
    expect(SRC).toMatch(/runCommand\("run_command"/);
  });

  it("surfaces a refusal instead of claiming the command ran", async () => {
    // An unknown command id is the one refusal the spine raises synchronously.
    const ack = await runTerminalLine("");
    expect(ack.ok).toBe(false);
    expect(sent).toEqual([]);
  });

  it("only ever dispatches ids that exist in the command catalog", () => {
    const ids = [...SRC.matchAll(/runCommand\("([a-z_]+)"/g)].map((m) => m[1]);
    expect(ids.length).toBeGreaterThan(0);
    for (const id of ids) expect(COMMANDS.some((c) => c.id === id)).toBe(true);
  });
});

describe("streamed output", () => {
  it("appends every new row incrementally, not just the newest", () => {
    line("proc:1", "compiling hide-backend", 1);
    line("proc:1", "warning: unused import", 2);
    let cursor: ToolEvent | null = null;
    const first = newSince(tools(), cursor);
    expect(first.map((e) => e.message)).toEqual(["compiling hide-backend", "warning: unused import"]);
    cursor = first[first.length - 1];

    line("proc:1", "test result: ok", 3);
    const next = newSince(tools(), cursor);
    expect(next.map((e) => e.message)).toEqual(["test result: ok"]);
    // Nothing new means nothing repainted (no duplicated tail).
    expect(newSince(tools(), next[next.length - 1])).toEqual([]);
  });

  it("folds the streamed rows into one process keyed by the host process id", () => {
    line("proc:1", "a", 1);
    line("proc:1", "b", 2);
    line("call_agent_7", "shell.run . reading", 3);
    const procs = foldProcesses(tools());
    expect(procs).toHaveLength(1);
    expect(procs[0]).toMatchObject({ id: "proc:1", lines: 2, last: "b", state: "streaming" });
  });

  it("reads the host fail-closed sandbox refusal as a blocked process", () => {
    line("proc:2", "SANDBOX_UNAVAILABLE: refusing to run unconfined (no macOS sandbox-exec / Linux bwrap)", 1);
    expect(latestProc(tools())?.state).toBe("refused");
    expect(row(BASE).sandbox).toBe("refused");
    expect(row(BASE).processTone).toBe("blocked");
  });
});

describe("navigating away and back", () => {
  it("keeps the process and replays its whole buffer on re-attach", () => {
    line("proc:3", "server listening on 7717", 1);
    const cursor = tools()[tools().length - 1];
    expect(newSince(tools(), cursor)).toEqual([]);

    // Navigate away: the panel unmounts, the store keeps folding the host stream.
    line("proc:3", "GET /v1/hide/events 200", 2);
    line("proc:3", "GET /healthz 200", 3);

    // Navigate back: a fresh terminal has no cursor, so it re-attaches to the whole buffer.
    const replay = newSince(tools(), null);
    expect(replay.map((e) => e.message)).toEqual([
      "server listening on 7717",
      "GET /v1/hide/events 200",
      "GET /healthz 200",
    ]);
    const proc = latestProc(tools());
    expect(proc).toMatchObject({ id: "proc:3", lines: 3, state: "streaming" });
    // The mount path is the replay path (no separate attach route to drift from).
    expect(SRC).toMatch(/for \(const ev of seeded\) writeRow\(term, ev\)/);
  });
});

describe("interrupting", () => {
  it("writes the interrupt byte to the process stdin through pty_input", async () => {
    const ack = await interruptProcess("proc:4");
    const ETX = String.fromCharCode(3); // the interrupt byte a real terminal sends on Ctrl+C
    expect(ack.ok).toBe(true);
    expect(sent).toHaveLength(1);
    expect(sent[0].type).toBe("custom");
    expect(sent[0].data.name).toBe("pty_input");
    expect(sent[0].data.payload).toMatchObject({ process: "proc:4", data: ETX });
    // Honest about its ceiling: no host stop trigger exists yet.
    expect(ack.message).toMatch(/no host stop trigger/);
  });

  it("does nothing (and says so) when no process has streamed", async () => {
    const ack = await interruptProcess(null);
    expect(ack.ok).toBe(false);
    expect(sent).toEqual([]);
  });
});

describe("the state row", () => {
  it("states env, cwd, sandbox, process, exit and owning task from host state", () => {
    line("proc:5", "running", 1);
    const r = row(BASE);
    expect(r).toEqual({
      env: "hawking @ main",
      cwd: "/w/hawking",
      sandbox: "confined",
      process: "proc:5 streaming",
      processTone: "active",
      exit: "not reported",
      task: "ses_term",
    });
  });

  it("distinguishes idle, pending and active without relying on colour", () => {
    expect(row(BASE)).toMatchObject({ process: "none", processTone: "idle" });
    expect(row({ ...BASE, pending: "cargo test" })).toMatchObject({
      process: "pending cargo test",
      processTone: "pending",
    });
    line("proc:6", "ok", 1);
    expect(row(BASE)).toMatchObject({ process: "proc:6 streaming", processTone: "active" });
  });

  it("renders the state in words with an accessible name", () => {
    line("proc:7", "ok", 1);
    const html = renderToStaticMarkup(createElement(TerminalStateBar, { row: row(BASE) }));
    for (const word of ["sandbox", "confined", "proc:7 streaming", "exit", "not reported", "ses_term"]) {
      expect(html).toContain(word);
    }
    expect(html).toMatch(/role="status"/);
    // the region is NAMED, and its six fields are content: the name no longer concatenates them, so
    // a change to one field does not re-announce the whole bar
    expect(html).toMatch(/aria-label="Terminal process state"/);
  });

  it("carries no control that the catalog cannot serve", () => {
    // stop / attach / capture-artifact have host methods but no wire trigger, so they must not be
    // rendered as buttons that would only log. No <button> lives in this surface at all.
    expect(SRC).not.toMatch(/<button/);
  });
});
