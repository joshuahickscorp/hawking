import { readFileSync, readdirSync, statSync } from "node:fs";
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

/** Every non-generated source file under app/src, so a derived invariant sees new files too. */
function walk(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else if (/\.(ts|tsx)$/.test(name)) out.push(p);
  }
  return out;
}

// The transport seam, stubbed so the spine tests read exactly what went on the wire.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
// `reply` lets one test answer an intent with a REFUSAL; every other test leaves it accepted.
const { sent, reply } = vi.hoisted(() => ({ sent: [] as any[], reply: { accepted: true } }));
vi.mock("./ipc", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sendIntent: async (i: any) => {
    sent.push(i);
    return { accepted: reply.accepted, event_seq: 1, message: reply.accepted ? null : "the host refused that" };
  },
  subscribeUi: () => () => {},
  callConnector: async () => null,
  TRANSPORT_KIND: "mock",
}));

// Test fixtures: `apply` takes a UiEvent; these are loose literals (the test file is excluded from the
// production tsc, and vitest transpiles without typecheck).
// Let the dispatch promise settle: a gate decision is recorded before the prompt closes, so the
// assertion has to run after the ack, not before it.
const flush = () => new Promise((r) => setTimeout(r, 0));
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const apply = (kind: any, session_id: string | null = "ses_x", seq = 1) =>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (useStore.getState().apply as any)({ seq, session_id, kind });

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

  // The two frames below are copied verbatim from a live catch-up against hide-serve:
  //   GET /v1/hide/events?after_seq=0&session_id=ses_01KY14JZP0040EK22033Z9V5SD
  // which returned seq 801 transcript_message and seq 803 checkpoint_created. A recorded session
  // used to replay as truncated JSON in the status bar with an empty conversation, and the
  // checkpoint id existed only on the live bus, so a reload disabled seven history verbs.
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
    // routed, so it is not ALSO dumped into the status bar as a 200-char JSON blob
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
    // and the stage that mounts the Context Stack now exists for this session
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
    // a later partial patch (a new branch only) must not wipe the repo or digest
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
    // the info notice still lands, so nothing is silently dropped
    expect(s.notices.filter((n) => n.code === "custom")).toHaveLength(3);
  });

  it("routes search results to their consumer instead of dumping JSON into the notice area", () => {
    // ui.tsx awaitSearchResults renders these in the search surface. Echoing them here too made
    // every search keystroke print three truncated JSON blobs in the status bar.
    useStore.setState({ notices: [] });
    apply({ type: "custom", data: { kind: "search_results", query: "loop", count: 2, hits: [{ event_id: "evt_1" }] } }, "ses_x", 11);
    expect(useStore.getState().notices.filter((n) => n.code === "custom")).toHaveLength(0);
    // an unrouted custom event still surfaces, so nothing is silently dropped
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
    // The courtyard composer kept a private sendIntent for exactly one reason: intentFor dropped
    // this argument. It is threaded now, so the staged files ride the spine.
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
    // Each has a real Intent::Custom arm in hide-backend host.rs handle_intent, so the app's
    // /v1/hide/intent channel reaches them; before the re-bind runCommand threw on all of them.
    // The payloads are the ones those arms require: an empty one is refused here, not on the host.
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
    // It was the last Rpc-bound command and the reason the diagnostics projection had no writer in
    // this app. Custom-bound now (host.rs handle_static_analysis_intent), and the argument rule
    // still refuses the empty payload the host would reject.
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
    // The catalog is what this app can dispatch, and it speaks /v1/hide/intent only. goal_get was
    // the last Rpc row: it advertised itself in the palette while being reachable by nothing, so it
    // is retired from the catalog (goal/get is still a real elevated-protocol Method).
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
    // Every one of these is command_palette:true with required_selection "none", so only the
    // argument model keeps them out. Each was a one-click row that reached the host and came back
    // `missing(...)`, or (merge_side_chat) was silently dropped there.
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
      // the same rule guards the wire, so a click path cannot send the empty payload either
      await expect(runCommand(id, {})).rejects.toThrow(/needs /);
    }
    expect(sent).toEqual([]);
  });

  it("derives shell shortcuts from the catalog and leaves composer-owned keys to the composer", () => {
    const ids = shortcutCommands().map((c) => c.id);
    // the composer binds Mod+Enter and Mod+/ on its own textarea; the shell must not rebind them
    expect(ids).not.toContain("submit_turn");
    expect(ids).not.toContain("steer");
    // a BUTTON binding (chat.new) owns no chord, so its command's chord is the shell's to bind
    expect(ids).toContain("create_side_chat");
    expect(ids).toContain("cancel_run");
    for (const c of shortcutCommands()) expect(c.keyboard_shortcut).toBeTruthy();
  });

  it("binds no two commands to the same shortcut", () => {
    const keys = boundShortcuts().map((b) => b.shortcut);
    expect(new Set(keys).size).toBe(keys.length);
  });

  it("lets no two catalog commands claim one chord on a surface they share", () => {
    // The old collision test only saw boundShortcuts(), which excludes every surface-owned chord, so
    // it could not see submit_turn and accept_diff both declaring Mod+Enter. This one reads the
    // catalog: two commands may share a chord only when no surface can offer both at once.
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
    // and the pair that DOES share Mod+Enter is disjoint by surface, which is why the diff review
    // scopes its window listener to the focused surface (HunkReview.reviewKeysActive)
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
    // Through the spine now (both are catalog commands), so the payload also carries the session
    // fill-in runCommand adds to every custom binding.
    expect(sent).toEqual([
      { type: "custom", data: { name: "approve_gate", payload: { gate: "g1", session_id: expect.any(String) } } },
      { type: "custom", data: { name: "deny_gate", payload: { gate: "g2", session_id: expect.any(String) } } },
    ]);
  });

  it("keeps the gate up until the decision is RECORDED, and puts it back when it is not", async () => {
    // The overlay used to clear synchronously, before and regardless of what the dispatch returned:
    // the app's one security-facing control dismissed itself on a decision that may never have
    // landed, leaving the effect parked with nothing left to answer it.
    useStore.setState({ gate: { gate: "g3", message: "rm -rf /" } });
    useStore.getState().approveGate();
    expect(useStore.getState().gate).toMatchObject({ gate: "g3", deciding: true });
    // and a second press cannot send a second decision while the first is in flight
    useStore.getState().denyGate();
    await flush();
    expect(sent.map((i) => i.data.name)).toEqual(["approve_gate"]);
    expect(useStore.getState().gate).toBeNull();

    // a refused decision is not a recorded one: the prompt comes back, with the reason
    reply.accepted = false;
    useStore.setState({ gate: { gate: "g4", message: "rm -rf /" }, notices: [] });
    useStore.getState().denyGate();
    await flush();
    reply.accepted = true;
    expect(useStore.getState().gate).toMatchObject({ gate: "g4", deciding: false });
    expect(useStore.getState().notices.map((n) => n.message)).toEqual(["the host refused that"]);
  });

  it("queues a second gate instead of orphaning the first", async () => {
    // One overlay, and it used to be overwritten unconditionally: the replaced gate's id lived
    // nowhere else, so its parked effect could never be approved or denied by anything. Two saves
    // under the shipped write policy is enough to reach it.
    useStore.setState({ gate: null, gateQueue: [] });
    apply({ type: "security_gate", data: { gate: "g_a", message: "save a" } }, "ses_1", 20);
    apply({ type: "security_gate", data: { gate: "g_b", message: "save b" } }, "ses_1", 21);
    expect(useStore.getState().gate?.gate).toBe("g_a");
    expect(useStore.getState().gateQueue.map((g) => g.gate)).toEqual(["g_b"]);

    // the reconnect catch-up replays the same gate; it may not queue twice
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
    // The host announces the pause as a Custom UiEvent (announce_approval_request). Before this the
    // FE had no dispatcher at all, so a SuggestOnly turn paused on its first effect forever.
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

/*
  Keyboard and palette PARITY: a control a mouse can reach must be reachable without one, and a chord
  advertised to the user must be bound somewhere. These read the surfaces as text because the suite
  runs in the node environment (vitest.config.ts), so there is no DOM to dispatch a key into.
*/
describe("keyboard and palette parity", () => {
  const read = (p: string) => readFileSync(join(__dirname, p), "utf8");

  it("gives Settings and the permission mode a palette path", () => {
    // Settings had two permanent click entry points (the toolbar gear, the Home rail) and no palette
    // or keyboard path at all; the permission mode, which can switch the security gate off, had none
    // either. Both are shell commands now, so the ONE derivation carries them.
    const ids = SHELL_COMMANDS.map((c) => c.id);
    expect(ids).toContain("open.settings");
    expect(ids).toContain("perm.ask");
    expect(ids).toContain("perm.bypass");
    expect(boundShortcuts().find((b) => b.id === "open.settings")?.shortcut).toBe("Mod+,");
    // and the bypass row names what it does rather than reading "bypass"
    expect(SHELL_COMMANDS.find((c) => c.id === "perm.bypass")?.title).toContain("auto-approve every gate");
  });

  it("has a handler for every shell command, so no palette row is a dead entry", () => {
    const app = read("App.tsx");
    for (const c of SHELL_COMMANDS) expect(app).toContain(`"${c.id}":`);
  });

  it("binds the chord the chat menus advertise for a side chat", () => {
    // Mod+Shift+N was rendered as a key hint in the composer menu and the New-chat menu, and bound
    // nowhere: the catalog gives it a BUTTON toolbar_binding, which owns no chord.
    expect(read("surfaces/chat/actions.ts")).toContain('shortcut: "Mod+Shift+N"');
    expect(boundShortcuts().find((b) => b.id === "create_side_chat")?.shortcut).toBe("Mod+Shift+N");
  });

  it("handles Mod+/ in the composer the courtyard tells the user to steer from", () => {
    expect(read("surfaces/home/Home.tsx")).toContain("Steer this run from the composer with Mod+/");
    expect(read("surfaces/home/HomeComposer.tsx")).toContain('e.key === "/" && (e.metaKey || e.ctrlKey)');
  });

  it("gives the five conversation side panels a palette path", () => {
    // They were mouse-only icon buttons on the Chat stage. Shell commands now, so the ONE derivation
    // carries them and App.tsx owns the handlers (asserted by the shell-handler test above).
    const ids = SHELL_COMMANDS.map((c) => c.id);
    for (const k of ["terminal", "diff", "preview", "tools", "artifacts"]) expect(ids).toContain(`panel.${k}`);
  });

  it("gives open_session and create_worktree a palette path", () => {
    // create_worktree carries no argument, so the catalog derivation already offers it.
    expect(paletteCommands().map((c) => c.id)).toContain("create_worktree");
    // open_session needs a session id, which a bare gesture cannot invent, so App.tsx offers one row
    // per recent session and the row supplies the argument.
    expect(paletteCommands().map((c) => c.id)).not.toContain("open_session");
    expect(read("App.tsx")).toContain("`Open session: ${s.title}`");
  });

  it("advertises no catalog chord that nothing binds", () => {
    // open_file declared Mod+P (the palette's own chord) while required_selection=file kept it out of
    // every derived key map. Every catalog chord must now be bound, or belong to a surface that
    // binds it itself (the composer's Mod+Enter / Mod+/, the diff review's accept / reject).
    const bound = new Set(boundShortcuts().map((b) => b.id));
    // A surface-owned chord is not shell-bound, but it is not invisible either: Settings renders
    // surfaceShortcuts() beside the shell table, which is what Cmd+S (bound inside Monaco, in no
    // table at all) was missing.
    const surfaceOwned = new Set(surfaceShortcuts().map((b) => b.id));
    expect([...surfaceOwned]).toContain("save_file");
    for (const c of COMMANDS.filter((c) => c.keyboard_shortcut))
      expect(bound.has(c.id) || surfaceOwned.has(c.id)).toBe(true);
    expect(commandById("open_file")?.keyboard_shortcut).toBeNull();
  });

  it("never lets Escape approve, deny or dismiss the security gate", () => {
    // Three unscoped Escape listeners were live at once, and one of them denied a paused step of a
    // running turn: one press to close the palette also answered the gate and closed the Executor.
    const app = read("App.tsx");
    expect(app).not.toMatch(/"Escape"[\s\S]{0,120}onDeny/);
    // the outermost listener stands down whenever something nearer the user is open
    expect(app).toContain("!paletteOpen && !settingsOpen && !gate");
    // and the gate answers with its two buttons, which are held while the decision is recorded
    expect(app).toContain("disabled={deciding}");
  });

  it("withholds the five panel rows while there is no conversation to show them beside", () => {
    // They were lifted into SHELL_COMMANDS for a keyboard path, but Home mounts the panel bar and
    // the panel only when the session has something to show, so on a fresh boot every row was a
    // silent no-op. ONE selector decides it, and both call sites read that selector: the gate used
    // to be spelled out twice, and the copy that said "messages only" made the Context Stack -
    // mounted nowhere else - unopenable on a host whose composer is disabled for want of a model.
    const app = read("App.tsx");
    expect(app).toContain('SHELL_COMMANDS.filter((c) => hasConversation || !c.id.startsWith("panel."))');
    expect(app).toContain("useStore(hasSessionActivity)");
    expect(read("surfaces/home/Home.tsx")).toContain("useStore(hasSessionActivity)");
    // A replayed session with no transcript (no model ever ran) still has its recorded tool feed,
    // and that is enough for the stage that mounts the panels to exist.
    const base = useStore.getState();
    expect(hasSessionActivity({ ...base, messages: [], tools: [] })).toBe(false);
    expect(
      hasSessionActivity({ ...base, messages: [], tools: [{ call_id: "t1", message: "started edit.write_file", ts: 0 }] }),
    ).toBe(true);
  });

  it("keeps every advertised chord live in the chamber the user is standing in", () => {
    // Settings lists boundShortcuts() and the palette prints each row's chord, so a chord that only
    // acts in the Code chamber walks there rather than flipping state under a chamber that renders
    // nothing, the same way the panel rows walk to Chat.
    const app = read("App.tsx");
    for (const id of ["toggle.chat", "toggle.float", "toggle.panel", "toggle.sidebar"])
      expect(app).toMatch(new RegExp(`"${id}": inCode\\(`));
    // and the one chord this campaign added is shown by the button that opens it
    expect(read("shell/Toolbar.tsx")).toContain('title={`Settings${chord("open.settings")}`}');
    expect(read("surfaces/home/Home.tsx")).toContain("title={`Settings${settingsChord()}`}");
  });

  it("renders the Settings keyboard map from the catalog, with no second hand-written table", () => {
    const s = read("surfaces/Settings.tsx");
    expect(s).toContain("boundShortcuts().map");
    // and the surface-owned chords beside them, so a keyboard binding cannot hide from the table
    expect(s).toContain("surfaceShortcuts().map");
    expect(s).not.toContain("const SHORTCUTS");
    expect(s).not.toContain('"Cmd P"');
  });
});

/*
  ACK SEMANTICS: the host answers an intent with THREE outcomes, and a held one is the dangerous
  case. A surface that branched on `accepted` alone rendered a destructive command parked at the
  approval gate as finished, and flipped its own optimistic state as though the effect had run.
*/
describe("held acks are never read as done", () => {
  const read = (p: string) => readFileSync(join(__dirname, p), "utf8");

  it("resolves the three outcomes, and treats a missing held flag as the old two-state meaning", () => {
    expect(ackState({ accepted: true, held: true, event_seq: 1, message: "held for approval: gate=g" })).toBe("held");
    expect(ackState({ accepted: true, event_seq: 1, message: null })).toBe("accepted");
    expect(ackState({ accepted: true, held: false, event_seq: 1, message: null })).toBe("accepted");
    expect(ackState({ accepted: false, event_seq: null, message: "no" })).toBe("refused");
  });

  it("every surface that can dispatch a held-capable command reads the third state", () => {
    // DERIVED, not enumerated. The old version was a hand-written list of five files, so the two
    // held paths it did not list (save_file's policy hold, and HunkReview, which reaches the gated
    // whole-diff revert by payload shape) were invisible to it and shipped a "done" for a hold.
    const heldCapable = [
      // Every ApprovalPolicy::Ask row, read off the generated catalog: adding one extends this.
      ...(catalog as { id: string; approval_policy?: string }[])
        .filter((c) => c.approval_policy === "ask")
        .map((c) => c.id),
      // A write the permission policy refuses is held at the same gate carrying its reason.
      "save_file",
      // A destructive argv is parked at the dangerous-command gate.
      "run_command",
      // host effect_command resolves reject_diff with no hunk_id to the gated whole-diff revert.
      "reject_diff",
    ];
    // The one shared helper that resolves the state on its caller's behalf.
    const READS_STATE = /ackState|\.held|heldNote|worktreeNotice/;
    const files = walk(__dirname).filter((f) => !/\.test\.|\/generated\//.test(f));
    let checked = 0;
    for (const file of files) {
      const src = readFileSync(file, "utf8");
      const dispatches = heldCapable.filter((id) => src.includes(`"${id}"`));
      // The spine itself renders nothing: wire.ts is the contract, store.ts the dispatcher,
      // ipc.ts the transport. Everything else that names a held-capable id shows a user a result.
      if (!dispatches.length || /\/(wire|store|ipc)\.ts$/.test(file)) continue;
      checked++;
      expect(READS_STATE.test(src), `${file} dispatches ${dispatches.join(", ")} without reading the held state`).toBe(
        true,
      );
    }
    expect(checked).toBeGreaterThan(4);
  });

  it("does not flip diff hunks or print done on a hold", () => {
    const e = read("surfaces/ide/Editor.tsx");
    // The held branch returns BEFORE the `: done` note and before the applyHunkStatus rewrite.
    expect(e.indexOf('state === "held"')).toBeLessThan(e.indexOf("${spec.label}: done"));
    expect(e).toContain("heldNote(spec.label)");
  });

  it("keeps a held timeline verb pending, not done", () => {
    const t = read("shell/StateTimeline.tsx");
    expect(t).toContain('state === "held"');
    expect(t).toContain('state: "pending", message: heldNote(action.label)');
  });

  it("surfaces a refusal and a hold from the palette and from a catalog chord", () => {
    // Both gestures resolve through runFromSpine, which used to `void runCommand(id).catch(...)` and
    // throw the ack away, so an honest negative ack never reached the keyboard user.
    const a = read("App.tsx");
    expect(a).not.toContain("void runCommand(id).catch(");
    expect(a).toContain("was refused");
    expect(a).toContain("heldNote(commandById(id)?.title ?? id)");
  });
});
