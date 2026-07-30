import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeEach, describe, it, expect, vi } from "vitest";
import {
  boundShortcuts,
  COMMANDS,
  commandById,
  hasSessionActivity,
  matchesShortcut,
  paletteCommands,
  runCommand,
  SHELL_COMMANDS,
  shortcutCommands,
  surfaceShortcuts,
  useStore,
} from "./store";
import { ackState, CUSTOM_NAMES } from "./wire";
import catalog from "./generated/command_catalog.json";
import { applyKind, flush, walk } from "./test_fixtures";

const { sent, reply } = vi.hoisted(() => ({ sent: [] as any[], reply: { accepted: true } }));
vi.mock("./ipc", () => ({
  sendIntent: async (i: any) => {
    sent.push(i);
    return {
      accepted: reply.accepted,
      event_seq: 1,
      message: reply.accepted ? null : "the host refused that",
    };
  },
  subscribeUi: () => () => {},
  callConnector: async () => null,
  TRANSPORT_KIND: "mock",
}));

const apply = (kind: any, session_id: string | null = "ses_x", seq = 1) =>
  applyKind(useStore.getState().apply as any, kind, session_id, seq);

describe("store.apply", () => {
  it("tracks the active session and runtime status from events", () => {
    apply({ type: "runtime_status", data: { status: "ready", detail: null } }, "ses_live");
    expect(useStore.getState().sessionId).toBe("ses_live");
    expect(useStore.getState().runtimeStatus).toBe("ready");
  });

  it("coalesces streamed tokens into one assistant message", () => {
    apply({ type: "token_batch", data: { stream_id: "s1", text: "Hello " } }, "ses_live", 2);
    apply({ type: "token_batch", data: { stream_id: "s1", text: "world" } }, "ses_live", 3);
    const msgs = useStore.getState().messages;
    const last = msgs[msgs.length - 1];
    expect(last.role).toBe("assistant");
    expect(last.text).toBe("Hello world");
  });

  it("renders a replayed transcript and re-adopts the checkpoint id, deduped by durable event id", () => {
    useStore.getState().startNewSession();
    useStore.setState({ notices: [] });
    const line = {
      type: "custom",
      data: {
        kind: "transcript_message",
        event_id: "evt_01KY15DVQ9GY0PWG3ATP5GJ7T3",
        role: "user",
        text: "prove the transcript replays after a reload",
      },
    };
    apply(line, "ses_replay", 801);
    apply(line, "ses_replay", 801); // open_session and the catch-up both replay it
    const msgs = useStore.getState().messages;
    expect(msgs.length).toBe(1);
    expect(msgs[0]).toMatchObject({ role: "user", text: "prove the transcript replays after a reload" });
    expect(useStore.getState().notices.some((n) => n.code === "custom")).toBe(false);

    apply(
      {
        type: "custom",
        data: { kind: "checkpoint_created", record: { checkpoint_id: "ckpt_ef335a2ce9515e43a2f5439f" } },
      },
      "ses_replay",
      803,
    );
    expect(useStore.getState().lastCheckpointId).toBe("ckpt_ef335a2ce9515e43a2f5439f");
    expect(hasSessionActivity(useStore.getState())).toBe(true);
    useStore.setState({ notices: [] }); // the store is shared across cases in this file
  });

  it("folds a context_manifest projection into the live manifest", () => {
    apply(
      { type: "projection_patch", data: { projection: "context_manifest", patch: { ctx_len_effective: 131072, tq_multiplier: 4 } } },
      "ses_live",
      4,
    );
    expect(useStore.getState().manifest?.ctx_len_effective).toBe(131072);
    expect(useStore.getState().manifest?.tq_multiplier).toBe(4);
  });

  it("folds the home digest and deep-merges partial workspace patches", () => {
    apply(
      { type: "projection_patch", data: { projection: "home", patch: { user: { name: "Joshua-Hicks" }, workspace: { repo: "hawking", branch: "main" }, digest: { sessions: 1182 } } } },
      null,
      5,
    );
    apply({ type: "projection_patch", data: { projection: "home", patch: { workspace: { branch: "wt/feat" } } } }, null, 6);
    const home = useStore.getState().home;
    expect(home?.user?.name).toBe("Joshua-Hicks");
    expect(home?.workspace?.repo).toBe("hawking");
    expect(home?.workspace?.branch).toBe("wt/feat");
    expect(home?.digest?.sessions).toBe(1182);
  });

  it("replaces the sessions list from a sessions projection", () => {
    apply({ type: "projection_patch", data: { projection: "sessions", patch: { items: [{ id: "ses_a", title: "a", state: "active", updated_ms: 1 }] } } }, null, 7);
    expect(useStore.getState().sessions.map((s) => s.id)).toEqual(["ses_a"]);
  });

  it("keeps the host-minted ids a Custom UiEvent carries, instead of only a truncated notice", () => {
    apply({ type: "custom", data: { kind: "checkpoint_created", record: { checkpoint_id: "ckpt_9", at_seq: 8 } } }, "ses_live", 8);
    apply({ type: "custom", data: { kind: "session_forked", record: { session_id: "ses_fork" } } }, "ses_fork", 9);
    apply({ type: "custom", data: { kind: "side_chat_created", record: { session_id: "ses_side" } } }, "ses_side", 10);
    const s = useStore.getState();
    expect(s.lastCheckpointId).toBe("ckpt_9");
    expect(s.lastForkedSession).toBe("ses_fork");
    expect(s.lastSideChat).toBe("ses_side");
    expect(s.notices.filter((n) => n.code === "custom")).toHaveLength(3);
  });

  it("routes search results to their consumer instead of dumping JSON into the notice area", () => {
    useStore.setState({ notices: [] });
    apply({ type: "custom", data: { kind: "search_results", query: "loop", count: 2, hits: [{ event_id: "evt_1" }] } }, "ses_x", 11);
    expect(useStore.getState().notices.filter((n) => n.code === "custom")).toHaveLength(0);
    apply({ type: "custom", data: { kind: "job_created" } }, "ses_x", 12);
    expect(useStore.getState().notices.filter((n) => n.code === "custom")).toHaveLength(1);
  });

  it("never synthesizes runtime readiness from the static role registry", () => {
    const src = readFileSync(join(__dirname, "store.ts"), "utf8");
    expect(src).not.toContain('"roles.list"');
    expect(src).toContain('callConnector<{ state?: RuntimeState; detail?: string | null }>("runtime", "state"');
  });

  it("startNewSession clears the local transcript", () => {
    useStore.getState().pushUserMessage("hello");
    expect(useStore.getState().messages.length).toBeGreaterThan(0);
    useStore.getState().startNewSession();
    expect(useStore.getState().messages.length).toBe(0);
    expect(useStore.getState().runPhase).toBe("idle");
  });
});

describe("the command spine", () => {
  beforeEach(() => {
    sent.length = 0;
    useStore.setState({ activeRunId: "run_1", sessionId: "ses_1", gate: null, gateQueue: [] });
  });

  it("builds the typed Intent for an Intent-bound command, filling the run id from live state", async () => {
    await runCommand("cancel_run");
    expect(sent).toEqual([{ type: "cancel_run", data: { run_id: "run_1" } }]);
  });

  it("carries the additive hunk_id on accept_diff and defaults it to the whole diff", async () => {
    await runCommand("accept_diff", { diff_id: "d1", hunk_id: "h2" });
    await runCommand("accept_diff", { diff_id: "d1" });
    expect(sent[0].data).toEqual({ run_id: "run_1", diff_id: "d1", hunk_id: "h2" });
    expect(sent[1].data.hunk_id).toBeNull();
  });

  it("builds a custom Intent for a Custom-bound command, filling the session id from live state", async () => {
    await runCommand("checkpoint_create", { label: "before refactor" });
    expect(sent).toEqual([
      {
        type: "custom",
        data: { name: "checkpoint_create", payload: { label: "before refactor", session_id: "ses_1" } },
      },
    ]);
  });

  it("carries submit_turn attachments, so no surface needs its own Intent builder", async () => {
    const blob = { id: "file:patch.diff", hash: "sha256:ab", size_bytes: 4, media_type: "text/plain" };
    await runCommand("submit_turn", { session_id: "ses_a", text: "review this", attachments: [blob] });
    expect(sent[0]).toEqual({
      type: "submit_turn",
      data: { session_id: "ses_a", text: "review this", attachments: [blob] },
    });
  });

  it("never overwrites a session id the caller supplied", async () => {
    await runCommand("checkpoint_create", { label: "x", session_id: "ses_other" });
    expect(sent[0].data.payload.session_id).toBe("ses_other");
  });

  it("dispatches the commands that were Rpc-bound before the contract reconciliation", async () => {
    const calls: [string, Record<string, unknown>][] = [
      ["steer", { run_id: "run_1", text: "try the other branch" }],
      ["memory_add", { claim: "the pool guard is reentrant" }],
      ["goal_evaluate", {}],
      ["workspace_set_repo_trust", { repo_id: "hawking", trust: "trusted" }],
      ["environment_switch", { env_id: "env_2" }],
    ];
    for (const [id, args] of calls) await runCommand(id, args);
    expect(sent.map((i) => i.data.name)).toEqual([
      "redirect_run",
      "memory_add",
      "goal_evaluate",
      "workspace_set_repo_trust",
      "environment_switch",
    ]);
    expect(paletteCommands().map((c) => c.id)).toEqual(
      expect.arrayContaining(["goal_evaluate", "new_session"]),
    );
  });

  it("reaches static analysis over the intent channel, so the Problems counter has a producer", async () => {
    expect(commandById("run_static_analysis")?.backend_binding.kind).toBe("custom");
    await expect(runCommand("run_static_analysis")).rejects.toThrow(/needs paths/);
    await runCommand("run_static_analysis", { paths: ["src/a.rs"] });
    expect(sent).toEqual([
      {
        type: "custom",
        data: { name: "run_static_analysis", payload: { paths: ["src/a.rs"], session_id: "ses_1" } },
      },
    ]);
  });

  it("leaves no command bound Rpc for a surface to trip over", () => {
    expect(COMMANDS.filter((c) => c.backend_binding.kind === "rpc").map((c) => c.id)).toEqual([]);
  });

  it("refuses an unknown command, a missing argument, and a run-scoped command with no run", async () => {
    await expect(runCommand("not_a_command")).rejects.toThrow(/unknown command/);
    await expect(runCommand("open_file")).rejects.toThrow(/needs path/);
    useStore.setState({ activeRunId: null });
    await expect(runCommand("pause_run")).rejects.toThrow(/active run/);
    expect(sent).toEqual([]);
  });

  it("keeps every Custom binding in the catalog on the wire contract", () => {
    const names = new Set<string>(CUSTOM_NAMES);
    const customs = COMMANDS.filter((c) => c.backend_binding.kind === "custom");
    expect(customs.length).toBeGreaterThan(0);
    for (const c of customs) {
      expect(names.has((c.backend_binding as { target: string }).target)).toBe(true);
    }
  });

  it("derives palette entries from the catalog and offers nothing it cannot run", () => {
    const ids = paletteCommands().map((c) => c.id);
    expect(ids.length).toBeGreaterThan(5);
    for (const c of paletteCommands()) {
      expect(c.command_palette).toBe(true);
      expect(c.backend_binding.kind).not.toBe("rpc");
      expect(c.backend_binding.kind).not.toBe("local_only");
      expect(c.required_selection).toBe("none");
    }
    expect(ids).toEqual(expect.arrayContaining(["cancel_run", "new_session", "checkpoint_create"]));
  });

  it("offers no palette row that cannot carry the payload its host arm requires", async () => {
    const ids = paletteCommands().map((c) => c.id);
    const unsatisfiable = [
      "checkpoint_restore",
      "checkpoint_rewind",
      "checkpoint_replay",
      "checkpoint_fork",
      "checkpoint_compare",
      "checkpoint_inspect",
      "merge_side_chat",
      "goal_set",
      "environment_switch",
      "workspace_set_repo_trust",
      "memory_record_outcome",
      "memory_revalidate",
      "reorder_plan",
      "promote_run",
      "resume_run_foreground",
      "revert_diff",
      "run_search",
      "pty_input",
      "pty_resize",
      "run_static_analysis",
      "attach_process",
      "stop_process",
      "capture_process_artifact",
      "export_review_receipt",
    ];
    for (const id of unsatisfiable) {
      expect(COMMANDS.find((c) => c.id === id)?.command_palette).toBe(true);
      expect(ids).not.toContain(id);
      await expect(runCommand(id, {})).rejects.toThrow(/needs /);
    }
    expect(sent).toEqual([]);
  });

  it("derives shell shortcuts from the catalog and leaves composer-owned keys to the composer", () => {
    const ids = shortcutCommands().map((c) => c.id);
    expect(ids).not.toContain("submit_turn");
    expect(ids).not.toContain("steer");
    expect(ids).toContain("create_side_chat");
    expect(ids).toContain("cancel_run");
    for (const c of shortcutCommands()) expect(c.keyboard_shortcut).toBeTruthy();
  });

  it("binds no two commands to the same shortcut", () => {
    const keys = boundShortcuts().map((b) => b.shortcut);
    expect(new Set(keys).size).toBe(keys.length);
  });

  it("lets no two catalog commands claim one chord on a surface they share", () => {
    const byChord = new Map<string, typeof COMMANDS>();
    for (const c of COMMANDS.filter((c) => c.keyboard_shortcut))
      byChord.set(c.keyboard_shortcut as string, [...(byChord.get(c.keyboard_shortcut as string) ?? []), c]);
    const shared: string[] = [];
    for (const [chord, specs] of byChord)
      for (const a of specs)
        for (const b of specs)
          if (a.id < b.id && a.available_surfaces.some((s) => b.available_surfaces.includes(s)))
            shared.push(`${chord}: ${a.id} and ${b.id}`);
    expect(shared).toEqual([]);
    expect(byChord.get("Mod+Enter")?.map((c) => c.id)).toEqual(["submit_turn", "accept_diff"]);
  });

  it("preserves the existing shell shortcuts", () => {
    const keys = boundShortcuts().map((b) => b.shortcut);
    for (const k of ["Mod+P", "Mod+J", "Mod+B", "Mod+I"]) expect(keys).toContain(k);
  });

  it("matches a shortcut string against a keyboard event, modifiers and all", () => {
    const ev = (key: string, mods: Partial<KeyboardEvent> = {}) =>
      ({ key, metaKey: false, ctrlKey: false, shiftKey: false, altKey: false, ...mods }) as KeyboardEvent;
    expect(matchesShortcut("Mod+.", ev(".", { metaKey: true }))).toBe(true);
    expect(matchesShortcut("Mod+.", ev(".", { ctrlKey: true }))).toBe(true);
    expect(matchesShortcut("Mod+.", ev("."))).toBe(false);
    expect(matchesShortcut("Mod+Shift+F", ev("F", { metaKey: true, shiftKey: true }))).toBe(true);
    expect(matchesShortcut("Mod+P", ev("p", { metaKey: true, shiftKey: true }))).toBe(false);
    expect(commandById("cancel_run")?.keyboard_shortcut).toBe("Mod+.");
  });

  it("shares ONE pair of gate handlers between both presentations", async () => {
    useStore.setState({ gate: { gate: "g1", message: "rm -rf" } });
    useStore.getState().approveGate();
    await flush();
    expect(useStore.getState().gate).toBeNull();
    useStore.setState({ gate: { gate: "g2", message: "rm -rf" } });
    useStore.getState().denyGate();
    await flush();
    expect(useStore.getState().gate).toBeNull();
    expect(sent).toEqual([
      { type: "custom", data: { name: "approve_gate", payload: { gate: "g1", session_id: expect.any(String) } } },
      { type: "custom", data: { name: "deny_gate", payload: { gate: "g2", session_id: expect.any(String) } } },
    ]);
  });

  it("keeps the gate up until the decision is RECORDED, and puts it back when it is not", async () => {
    useStore.setState({ gate: { gate: "g3", message: "rm -rf /" } });
    useStore.getState().approveGate();
    expect(useStore.getState().gate).toMatchObject({ gate: "g3", deciding: true });
    useStore.getState().denyGate();
    await flush();
    expect(sent.map((i) => i.data.name)).toEqual(["approve_gate"]);
    expect(useStore.getState().gate).toBeNull();

    reply.accepted = false;
    useStore.setState({ gate: { gate: "g4", message: "rm -rf /" }, notices: [] });
    useStore.getState().denyGate();
    await flush();
    reply.accepted = true;
    expect(useStore.getState().gate).toMatchObject({ gate: "g4", deciding: false });
    expect(useStore.getState().notices.map((n) => n.message)).toEqual(["the host refused that"]);
  });

  it("queues a second gate instead of orphaning the first", async () => {
    useStore.setState({ gate: null, gateQueue: [] });
    apply({ type: "security_gate", data: { gate: "g_a", message: "save a" } }, "ses_1", 20);
    apply({ type: "security_gate", data: { gate: "g_b", message: "save b" } }, "ses_1", 21);
    expect(useStore.getState().gate?.gate).toBe("g_a");
    expect(useStore.getState().gateQueue.map((g) => g.gate)).toEqual(["g_b"]);

    apply({ type: "security_gate", data: { gate: "g_b", message: "save b" } }, "ses_1", 22);
    apply({ type: "security_gate", data: { gate: "g_a", message: "save a" } }, "ses_1", 23);
    expect(useStore.getState().gateQueue).toHaveLength(1);

    useStore.getState().approveGate();
    await flush();
    expect(useStore.getState().gate?.gate).toBe("g_b");
    expect(useStore.getState().gateQueue).toHaveLength(0);
    useStore.getState().approveGate();
    await flush();
    expect(useStore.getState().gate).toBeNull();
  });

  it("answers a paused effectful step with approve_effect / deny_effect through the same handlers", async () => {
    apply(
      {
        type: "custom",
        data: {
          kind: "approval_requested",
          run_id: "run_7",
          step_id: "stp_3",
          summary: "write src/retry.rs",
          effects: ["write_fs"],
        },
      },
      "ses_1",
      11,
    );
    expect(useStore.getState().gate).toMatchObject({
      gate: "stp_3",
      message: "write src/retry.rs (write_fs)",
      effect: { run_id: "run_7", step_id: "stp_3" },
    });

    useStore.getState().approveGate();
    await flush();
    expect(useStore.getState().gate).toBeNull();

    apply({ type: "custom", data: { kind: "approval_requested", run_id: "run_8", step_id: "stp_4" } }, "ses_1", 12);
    useStore.getState().denyGate();
    await flush();

    expect(sent).toEqual([
      {
        type: "custom",
        data: {
          name: "approve_effect",
          payload: { run_id: "run_7", step_id: "stp_3", session_id: expect.any(String) },
        },
      },
      {
        type: "custom",
        data: {
          name: "deny_effect",
          payload: { run_id: "run_8", step_id: "stp_4", session_id: expect.any(String) },
        },
      },
    ]);
  });
});

describe("keyboard and palette parity", () => {
  const read = (p: string) => readFileSync(join(__dirname, p), "utf8");

  it("gives Settings and the permission mode a palette path", () => {
    const ids = SHELL_COMMANDS.map((c) => c.id);
    expect(ids).toContain("open.settings");
    expect(ids).toContain("perm.ask");
    expect(ids).toContain("perm.bypass");
    expect(boundShortcuts().find((b) => b.id === "open.settings")?.shortcut).toBe("Mod+,");
    expect(SHELL_COMMANDS.find((c) => c.id === "perm.bypass")?.title).toContain("auto-approve every gate");
  });

  it("has a handler for every shell command, so no palette row is a dead entry", () => {
    const app = read("App.tsx");
    for (const c of SHELL_COMMANDS) expect(app).toContain(`"${c.id}":`);
  });

  it("binds the chord the chat menus advertise for a side chat", () => {
    expect(read("surfaces/chat_actions.ts")).toContain('shortcut: "Mod+Shift+N"');
    expect(boundShortcuts().find((b) => b.id === "create_side_chat")?.shortcut).toBe("Mod+Shift+N");
  });

  it("handles Mod+/ in the composer the courtyard tells the user to steer from", () => {
    expect(read("surfaces/home_Home.tsx")).toContain("Steer this run from the composer with Mod+/");
    expect(read("surfaces/home_HomeComposer.tsx")).toContain('e.key === "/" && (e.metaKey || e.ctrlKey)');
  });

  it("gives the five conversation side panels a palette path", () => {
    const ids = SHELL_COMMANDS.map((c) => c.id);
    for (const k of ["terminal", "diff", "preview", "tools", "artifacts"]) expect(ids).toContain(`panel.${k}`);
  });

  it("gives open_session and create_worktree a palette path", () => {
    expect(paletteCommands().map((c) => c.id)).toContain("create_worktree");
    expect(paletteCommands().map((c) => c.id)).not.toContain("open_session");
    expect(read("App.tsx")).toContain("`Open session: ${s.title}`");
  });

  it("advertises no catalog chord that nothing binds", () => {
    const bound = new Set(boundShortcuts().map((b) => b.id));
    const surfaceOwned = new Set(surfaceShortcuts().map((b) => b.id));
    expect([...surfaceOwned]).toContain("save_file");
    for (const c of COMMANDS.filter((c) => c.keyboard_shortcut))
      expect(bound.has(c.id) || surfaceOwned.has(c.id)).toBe(true);
    expect(commandById("open_file")?.keyboard_shortcut).toBeNull();
  });

  it("never lets Escape approve, deny or dismiss the security gate", () => {
    const app = read("App.tsx");
    expect(app).not.toMatch(/"Escape"[\s\S]{0,120}onDeny/);
    expect(app).toContain("!paletteOpen && !settingsOpen && !gate");
    expect(app).toContain("disabled={deciding}");
  });

  it("withholds the five panel rows while there is no conversation to show them beside", () => {
    const app = read("App.tsx");
    expect(app).toContain('SHELL_COMMANDS.filter((c) => hasConversation || !c.id.startsWith("panel."))');
    expect(app).toContain("useStore(hasSessionActivity)");
    expect(read("surfaces/home_Home.tsx")).toContain("useStore(hasSessionActivity)");
    const base = useStore.getState();
    expect(hasSessionActivity({ ...base, messages: [], tools: [] })).toBe(false);
    expect(
      hasSessionActivity({ ...base, messages: [], tools: [{ call_id: "t1", message: "started edit.write_file", ts: 0 }] }),
    ).toBe(true);
  });

  it("keeps every advertised chord live in the chamber the user is standing in", () => {
    const app = read("App.tsx");
    for (const id of ["toggle.chat", "toggle.float", "toggle.panel", "toggle.sidebar"])
      expect(app).toMatch(new RegExp(`"${id}": inCode\\(`));
    expect(read("shell/Toolbar.tsx")).toContain('title={`Settings${chord("open.settings")}`}');
    expect(read("surfaces/home_Home.tsx")).toContain("title={`Settings${settingsChord()}`}");
  });

  it("renders the Settings keyboard map from the catalog, with no second hand-written table", () => {
    const s = read("surfaces/Settings.tsx");
    expect(s).toContain("boundShortcuts().map");
    expect(s).toContain("surfaceShortcuts().map");
    expect(s).not.toContain("const SHORTCUTS");
    expect(s).not.toContain('"Cmd P"');
  });
});

describe("held acks are never read as done", () => {
  const read = (p: string) => readFileSync(join(__dirname, p), "utf8");

  it("resolves the three outcomes, and treats a missing held flag as the old two-state meaning", () => {
    expect(ackState({ accepted: true, held: true, event_seq: 1, message: "held for approval: gate=g" })).toBe("held");
    expect(ackState({ accepted: true, event_seq: 1, message: null })).toBe("accepted");
    expect(ackState({ accepted: true, held: false, event_seq: 1, message: null })).toBe("accepted");
    expect(ackState({ accepted: false, event_seq: null, message: "no" })).toBe("refused");
  });

  it("every surface that can dispatch a held-capable command reads the third state", () => {
    const heldCapable = [
      ...(catalog as { id: string; approval_policy?: string }[])
        .filter((c) => c.approval_policy === "ask")
        .map((c) => c.id),
      "save_file",
      "run_command",
      "reject_diff",
    ];
    const READS_STATE = /ackState|\.held|heldNote|worktreeNotice/;
    const files = walk(__dirname).filter((f) => !/\.test\.|\/generated\//.test(f));
    let checked = 0;
    for (const file of files) {
      const src = readFileSync(file, "utf8");
      const dispatches = heldCapable.filter((id) => src.includes(`"${id}"`));
      // wire_types.ts is pure Intent/UiEvent type projection (no dispatch UI).
      if (!dispatches.length || /\/(wire|wire_types|store|ipc)\.ts$/.test(file)) continue;
      checked++;
      expect(READS_STATE.test(src), `${file} dispatches ${dispatches.join(", ")} without reading the held state`).toBe(
        true,
      );
    }
    expect(checked).toBeGreaterThan(4);
  });

  it("does not flip diff hunks or print done on a hold", () => {
    const e = read("surfaces/ide_Editor.tsx");
    expect(e.indexOf('state === "held"')).toBeLessThan(e.indexOf("${spec.label}: done"));
    expect(e).toContain("heldNote(spec.label)");
  });

  it("keeps a held timeline verb pending, not done", () => {
    const t = read("shell/StateTimeline.tsx");
    expect(t).toContain('state === "held"');
    expect(t).toContain('state: "pending", message: heldNote(action.label)');
  });

  it("surfaces a refusal and a hold from the palette and from a catalog chord", () => {
    const a = read("App.tsx");
    expect(a).not.toContain("void runCommand(id).catch(");
    expect(a).toContain("was refused");
    expect(a).toContain("heldNote(commandById(id)?.title ?? id)");
  });
});
