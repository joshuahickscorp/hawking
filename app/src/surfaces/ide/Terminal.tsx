/*
  Terminal.tsx: the integrated terminal as a session-aware process surface. @xterm/xterm mounted and
  re-skinned to the v3 doctrine: --void grayscale background, a LIGHT caret (no gold), Geist Mono, and
  lichen/oxide ANSI mapped to the only two semantic tokens (--ok / --bad). NO terminal-green, NO blue,
  NO purple, NO gold. The prompt glyph is light entering the dark.

  What is real here (and only what is real):

  * A typed line resolves the catalog command `run_command` through the ONE command spine
    (store.runCommand), so the palette, a shortcut, and this prompt all reach the same semantics. The
    host runs it through the sandboxed process surface (hide-backend spawn_supervised -> confine),
    which is fail closed: with no OS sandbox it refuses rather than running unconfined.
  * Output streams INCREMENTALLY. The host publishes every stdout/stderr line as a tool_progress
    event tagged with the process id (call_id = "proc:N"), so rows are appended as they arrive, not
    echoed once at the end. Every new row is written, never just the newest one.
  * Re-attach on mount. The store folds the host's tool_progress stream whether or not this panel is
    mounted, so a process the user navigated away from keeps running and its buffered output is
    replayed into a freshly mounted terminal.
  * Ctrl+C writes 0x03 to the live process's stdin through `pty_input`, and the process geometry goes
    out once per process through `pty_resize`. Both are real host paths.
  * One compact state row reports what the host actually tells us: workspace env, cwd, sandbox state,
    process id and state, exit state, and the owning task.

  Deliberately NOT surfaced (no wire trigger exists, see crates/hide-protocol/src/command.rs, the
  terminal section): stop_process, attach_process/detach_process, capture_process_artifact, and
  starting a persistent or service process. The host methods exist; no command in the catalog reaches
  them, so this surface does not grow a button that would only log. Same reason the exit field reads
  "not reported": the supervisor streams output but publishes no terminal-status event yet.
*/
import { useEffect, useRef, useState } from "react";
import { Terminal as Xterm } from "@xterm/xterm";
import type { ITheme } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import { runCommand, useStore, type ToolEvent } from "../../store";
import { ackState, heldNote } from "../../wire";
import { MONO_FONT } from "./ideConstants";

// The v3 xterm theme: every color a theme.css token value (xterm needs literal hex). Grayscale
// concrete with light as the only accent; ok/bad are the lone two colors, glyph-paired in the feed.
const XTERM_THEME: ITheme = {
  background: "#070707", // --void
  foreground: "#9B9A95", // --text-2
  cursor: "#F4F2EE", // --light caret (the cross of light, never gold)
  cursorAccent: "#070707",
  selectionBackground: "#F4F2EE22",
  selectionInactiveBackground: "#F4F2EE12",
  scrollbarSliderBackground: "#F4F2EE10",
  scrollbarSliderHoverBackground: "#F4F2EE1C",
  black: "#070707",
  red: "#C0807A", // --bad (oxide)
  green: "#7E9E86", // --ok (lichen)
  yellow: "#9B9A95", // remap yellow -> neutral (no gold/yellow, anywhere)
  blue: "#9B9A95", // remap blue -> neutral (no blue, anywhere)
  magenta: "#9B9A95", // remap magenta -> neutral (no purple)
  cyan: "#7E9E86",
  white: "#ECEAE6", // --text-1
  brightBlack: "#8A887F", // --text-3
  brightRed: "#C0807A",
  brightGreen: "#7E9E86",
  brightYellow: "#ECEAE6", // remap -> chalk (no gold)
  brightBlue: "#9B9A95",
  brightMagenta: "#9B9A95",
  brightCyan: "#7E9E86",
  brightWhite: "#F4F2EE", // --light
};

// The prompt: a light "hide" lit against the dark, then a muted caret glyph (no gold sequence).
const PROMPT = "\x1b[38;2;244;242;238mhide\x1b[0m \x1b[38;2;110;109;104m›\x1b[0m ";
const DIM = (s: string) => "\x1b[38;2;110;109;104m" + s + "\x1b[0m";
const OUT = (s: string) => "\x1b[38;2;155;154;149m" + s + "\x1b[0m";
const BAD = (s: string) => "\x1b[38;2;192;128;122m" + s + "\x1b[0m";
// Erase the live prompt line before appending streamed output, then repaint prompt + pending input.
const CLEAR_LINE = "\r\x1b[2K";

/* ---- Process state, folded from the host's streamed rows -------------------------------------
   The supervisor tags every output line with the process id, so the tool feed already carries the
   process registry; nothing here invents state the host did not send. */

export const PROC_PREFIX = "proc:";
export const isProcEvent = (ev: ToolEvent) => ev.call_id.startsWith(PROC_PREFIX);

/** The host's fail-closed refusal marker (process.rs `confine`): the one sandbox fact FE-observable. */
export const SANDBOX_REFUSED = "SANDBOX_UNAVAILABLE";

export interface TerminalProc {
  id: string;
  lines: number;
  last: string;
  /** "streaming" = output observed; "refused" = the host refused to run it unconfined. */
  state: "streaming" | "refused";
}

/** Every process the host has streamed a row for, oldest first. */
export function foldProcesses(tools: ToolEvent[]): TerminalProc[] {
  const byId = new Map<string, TerminalProc>();
  for (const ev of tools) {
    if (!isProcEvent(ev)) continue;
    const prev = byId.get(ev.call_id);
    const refused = ev.message.startsWith(SANDBOX_REFUSED) || prev?.state === "refused";
    byId.set(ev.call_id, {
      id: ev.call_id,
      lines: (prev?.lines ?? 0) + 1,
      last: ev.message,
      state: refused ? "refused" : "streaming",
    });
  }
  return [...byId.values()];
}

/** The process this terminal is currently mirroring (the most recent one the host streamed). */
export function latestProc(tools: ToolEvent[]): TerminalProc | null {
  const all = foldProcesses(tools);
  return all.length ? all[all.length - 1] : null;
}

/** Rows the buffer has not written yet. A `last` that has aged out of the ring replays the whole
 *  buffer, which is exactly the re-attach case (mount with `last = null` replays everything). */
export function newSince(tools: ToolEvent[], last: ToolEvent | null): ToolEvent[] {
  if (!last) return tools;
  const i = tools.indexOf(last);
  return i < 0 ? tools : tools.slice(i + 1);
}

/* ---- The two real dispatches ----------------------------------------------------------------- */

export interface TerminalAck {
  ok: boolean;
  message: string;
}

/** Run a line through the command spine. `run_command` is the sandbox-confined host path. */
export async function runTerminalLine(cmd: string): Promise<TerminalAck> {
  const argv = cmd.split(/\s+/).filter(Boolean);
  if (!argv.length) return { ok: false, message: "empty command" };
  const cwd = useStore.getState().home?.workspace?.root ?? null;
  try {
    const ack = await runCommand("run_command", { argv, cwd });
    // A destructive argv is PARKED at the security gate, not started. Reading `accepted` alone
    // printed "started ... (sandbox confined)" for a command the host refused to run and did not
    // confine, because it never spawned it.
    switch (ackState(ack)) {
      case "held":
        return { ok: false, message: heldNote(argv.join(" ")) };
      case "accepted":
        return { ok: true, message: "started " + argv.join(" ") + " (sandbox confined)" };
      default:
        return { ok: false, message: ack.message ?? "command rejected" };
    }
  } catch (err) {
    return { ok: false, message: err instanceof Error ? err.message : String(err) };
  }
}

/** Ctrl+C: write 0x03 to the process stdin via `pty_input`. Honest about its ceiling: the host has
 *  no process-stop wire trigger, and a one-shot command is spawned with a null stdin, so the host
 *  answers with a real error rather than a silent success. */
export async function interruptProcess(procId: string | null): Promise<TerminalAck> {
  if (!procId) return { ok: false, message: "no process to interrupt" };
  try {
    const ack = await runCommand("pty_input", { process: procId, data: "\u0003" });
    return ack.accepted
      ? { ok: true, message: "interrupt written to " + procId + " stdin (no host stop trigger yet)" }
      : { ok: false, message: ack.message ?? "interrupt rejected" };
  } catch (err) {
    return { ok: false, message: err instanceof Error ? err.message : String(err) };
  }
}

/* ---- The compact state row -------------------------------------------------------------------
   One line, no buttons: the only permanent addition, and it exists because cwd, sandbox posture,
   process identity, and owning task were otherwise invisible in this panel. */

export interface TerminalStateRow {
  env: string;
  cwd: string;
  sandbox: "confined" | "refused";
  process: string;
  processTone: "idle" | "pending" | "active" | "blocked";
  exit: string;
  task: string;
}

/** Everything the row shows, derived from host state only. `pending` is a run this surface accepted
 *  before the host streamed its first row (so it has no process id yet). */
export function stateRow(
  tools: ToolEvent[],
  opts: { pending: string | null; root?: string; repo?: string; branch?: string; session: string },
): TerminalStateRow {
  const proc = latestProc(tools);
  const env = [opts.repo, opts.branch].filter(Boolean).join(" @ ") || "workspace unknown";
  const refused = proc?.state === "refused";
  return {
    env,
    cwd: opts.root ?? "host workspace root",
    sandbox: refused ? "refused" : "confined",
    process: proc ? proc.id + " " + proc.state : opts.pending ? "pending " + opts.pending : "none",
    processTone: refused ? "blocked" : proc ? "active" : opts.pending ? "pending" : "idle",
    exit: "not reported",
    task: opts.session,
  };
}

const FIELD_TITLE: Record<string, string> = {
  env: "The workspace this terminal runs in (repo and branch from the host home projection). The process inherits the host environment; this surface sets no extra variables.",
  cwd: "The working directory sent with run_command. Blank means the host uses its workspace root.",
  sandbox:
    "Every terminal process is spawned through the host OS confinement and is fail closed: refused means the host declined to run it unconfined.",
  process: "The host process id and what it is doing. Pending means accepted but no output streamed yet.",
  exit: "The host streams process output but publishes no terminal-status event yet, so the exit code is not reported to this surface.",
  task: "The session that owns the process (the host records it as the process owner).",
};

function Field({ label, value, tone }: { label: string; value: string; tone?: "bad" | "warn" }) {
  return (
    <span
      title={FIELD_TITLE[label]}
      style={{ display: "inline-flex", gap: 4, whiteSpace: "nowrap", alignItems: "baseline" }}
    >
      <span style={{ color: "var(--text-3)" }}>{label}</span>
      <span
        style={{
          color: tone === "bad" ? "var(--bad)" : tone === "warn" ? "var(--text-1)" : "var(--text-2)",
          fontStyle: tone === "warn" ? "italic" : undefined,
        }}
      >
        {value}
      </span>
    </span>
  );
}

export function TerminalStateBar({ row }: { row: TerminalStateRow }) {
  const tone = row.processTone === "blocked" ? "bad" : row.processTone === "pending" ? "warn" : undefined;
  return (
    <div
      role="status"
      aria-live="polite"
      // The NAME of the region, not its contents. It used to concatenate all six fields, so a
      // change to any one of them re-announced the whole bar; the fields below are the content a
      // live region announces, and they already read as "label value".
      aria-label="Terminal process state"
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "var(--ma-3)",
        padding: "var(--ma-1) var(--ma-4)",
        fontFamily: MONO_FONT,
        fontSize: 11,
        lineHeight: 1.6,
        color: "var(--text-2)",
      }}
    >
      <Field label="env" value={row.env} />
      <Field label="cwd" value={row.cwd} />
      <Field label="sandbox" value={row.sandbox} tone={row.sandbox === "refused" ? "bad" : undefined} />
      <Field label="process" value={row.process} tone={tone} />
      <Field label="exit" value={row.exit} />
      <Field label="task" value={row.task} />
    </div>
  );
}

export function Terminal() {
  const hostRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<Xterm | null>(null);
  const lineRef = useRef("");
  const lastRef = useRef<ToolEvent | null>(null);
  // A run this surface accepted before the host streamed its first row (so it has no id yet).
  const [pending, setPending] = useState<string | null>(null);

  // ponytail: the fold runs over the store's bounded tool ring (50 rows), so re-deriving it on every
  // streamed line is cheap. Memoize if the ring ever grows.
  const tools = useStore((s) => s.tools);
  const home = useStore((s) => s.home);
  const session = useStore((s) => s.sessionId);
  const procId = latestProc(tools)?.id ?? null;
  const row = stateRow(tools, {
    pending,
    root: home?.workspace?.root,
    repo: home?.workspace?.repo,
    branch: home?.workspace?.branch,
    session,
  });

  useEffect(() => {
    if (!hostRef.current) return;
    const term = new Xterm({
      theme: XTERM_THEME,
      fontFamily: MONO_FONT,
      fontSize: 13,
      lineHeight: 1.4,
      letterSpacing: 0.2,
      cursorBlink: true,
      cursorStyle: "bar",
      convertEol: true,
      scrollback: 2000,
      allowProposedApi: true,
    });
    term.open(hostRef.current);
    termRef.current = term;

    term.writeln(DIM("hawking shell . sandbox confined . output streams live"));
    // Re-attach: a process started before this panel was mounted (or before the user navigated away
    // and back) kept running host-side, and the store kept folding its rows. Replay them.
    const seeded = useStore.getState().tools;
    for (const ev of seeded) writeRow(term, ev);
    lastRef.current = seeded.length ? seeded[seeded.length - 1] : null;
    term.write(PROMPT);

    // Local line editor. Enter resolves the catalog command; the host streams the output back.
    const sub = term.onData((data) => {
      for (const ch of data) {
        const code = ch.charCodeAt(0);
        if (ch === "\r") {
          const cmd = lineRef.current.trim();
          term.write("\r\n");
          lineRef.current = "";
          if (!cmd) {
            term.write(PROMPT);
            continue;
          }
          setPending(cmd);
          void runTerminalLine(cmd).then((ack) => {
            term.write(CLEAR_LINE);
            term.writeln(ack.ok ? DIM(ack.message) : BAD(ack.message));
            if (!ack.ok) setPending(null);
            term.write(PROMPT + lineRef.current);
          });
        } else if (code === 127) {
          if (lineRef.current.length) {
            lineRef.current = lineRef.current.slice(0, -1);
            term.write("\b \b");
          }
        } else if (code === 3) {
          // Ctrl+C: abandon the local line, and signal the live process for real.
          lineRef.current = "";
          term.write("^C\r\n");
          const target = latestProc(useStore.getState().tools)?.id ?? null;
          void interruptProcess(target).then((ack) => {
            term.write(CLEAR_LINE);
            term.writeln(ack.ok ? DIM(ack.message) : BAD(ack.message));
            term.write(PROMPT + lineRef.current);
          });
        } else if (code >= 32) {
          lineRef.current += ch;
          term.write(ch);
        }
      }
    });

    // Stream: append EVERY row the host published since the last paint, not only the newest.
    const unsub = useStore.subscribe((s) => {
      const rows = newSince(s.tools, lastRef.current);
      if (!rows.length) return;
      lastRef.current = s.tools[s.tools.length - 1];
      term.write(CLEAR_LINE);
      for (const ev of rows) writeRow(term, ev);
      term.write(PROMPT + lineRef.current);
    });

    return () => {
      unsub();
      sub.dispose();
      term.dispose();
      termRef.current = null;
    };
  }, []);

  // Geometry goes to the host once per process (`pty_resize`). ponytail: no fit addon is installed,
  // so xterm's cols/rows do not change at runtime; send on each new process instead of on resize.
  useEffect(() => {
    const term = termRef.current;
    if (!procId) return;
    setPending(null); // the host answered the accepted run with a real process id
    if (!term) return;
    void runCommand("pty_resize", { process: procId, cols: term.cols, rows: term.rows }).catch(() => {});
  }, [procId]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div
        ref={hostRef}
        style={{
          flex: 1,
          minHeight: 0,
          overflow: "hidden",
          padding: "var(--ma-3) var(--ma-4)",
          background: "var(--void)",
          borderRadius: "var(--radius)",
          boxShadow: "var(--hairline), var(--inner-glow)",
        }}
      />
      <TerminalStateBar row={row} />
    </div>
  );
}

/** One streamed row. Process output is the shell's own text (rendered raw); an agent tool row keeps
 *  the light glyph so the two sources stay distinguishable in one buffer. */
function writeRow(term: Xterm, ev: ToolEvent) {
  if (isProcEvent(ev)) {
    term.writeln(ev.message.startsWith(SANDBOX_REFUSED) ? BAD(ev.message) : OUT(ev.message));
  } else {
    term.writeln("\x1b[38;2;244;242;238m●\x1b[0m " + OUT(ev.message));
  }
}
