/*
  Terminal.tsx: the integrated terminal (D1.2 / D4.4 #5). @xterm/xterm mounted and re-skinned to the
  doctrine: near-black --void background, gold cursor, Geist Mono, jade/red/orange ANSI mapped to the
  Part C semantic tokens (NO terminal-green, NO blue, per C3). Human input over the (future) PTY
  WebSocket; here, a typed line dispatches RunCommand{argv,cwd} and the agent's tool_progress rows
  (the shell tool's mirrored output) are echoed into the same buffer so the surface is ALIVE.

  Re-housed from the standard xterm IDE terminal; the chrome (prompt glyph, header) is HIDE mono.
*/
import { useEffect, useRef } from "react";
import { Terminal as Xterm } from "@xterm/xterm";
import type { ITheme } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import { sendIntent } from "../../ipc";
import { useStore, type ToolEvent } from "../../store";
import { intent } from "../../wire";
import { MONO_FONT } from "./monacoTheme";

// The doctrine xterm theme: every color a Part C token value (xterm needs literal hex).
const XTERM_THEME: ITheme = {
  background: "#060606",
  foreground: "#A8A6A1",
  cursor: "#FFD888", // gold caret
  cursorAccent: "#060606",
  selectionBackground: "#F0B95B33",
  selectionInactiveBackground: "#F0B95B1A",
  scrollbarSliderBackground: "#FFFFFF14",
  scrollbarSliderHoverBackground: "#FFFFFF26",
  black: "#060606",
  red: "#E5635E",
  green: "#6FBF8B",
  yellow: "#E08A3C",
  blue: "#A8A6A1", // remap blue -> neutral (C3: no blue, anywhere)
  magenta: "#A8A6A1", // remap magenta -> neutral (no purple)
  cyan: "#6FBF8B",
  white: "#F2F0EC",
  brightBlack: "#7C7A75",
  brightRed: "#E5635E",
  brightGreen: "#6FBF8B",
  brightYellow: "#F0B95B",
  brightBlue: "#A8A6A1",
  brightMagenta: "#A8A6A1",
  brightCyan: "#8FD0A6",
  brightWhite: "#F2F0EC",
};

const PROMPT = "\x1b[38;2;240;185;91mhide\x1b[0m \x1b[38;2;124;122;117m›\x1b[0m ";

export function Terminal() {
  const hostRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<Xterm | null>(null);
  const lineRef = useRef("");
  const lastToolSeq = useRef(0);

  useEffect(() => {
    if (!hostRef.current) return;
    const term = new Xterm({
      theme: XTERM_THEME,
      fontFamily: MONO_FONT,
      fontSize: 12,
      lineHeight: 1.35,
      letterSpacing: 0.2,
      cursorBlink: true,
      cursorStyle: "bar",
      convertEol: true,
      scrollback: 2000,
      allowProposedApi: true,
    });
    term.open(hostRef.current);
    termRef.current = term;

    term.writeln("\x1b[38;2;124;122;117mhawking shell. agent runs mirror here.\x1b[0m");
    term.write(PROMPT);

    // Local line editor: enter dispatches RunCommand, the host mirrors back as tool_progress.
    const sub = term.onData((data) => {
      for (const ch of data) {
        const code = ch.charCodeAt(0);
        if (ch === "\r") {
          const cmd = lineRef.current.trim();
          term.write("\r\n");
          lineRef.current = "";
          if (cmd) runLine(term, cmd);
          else term.write(PROMPT);
        } else if (code === 127) {
          if (lineRef.current.length) {
            lineRef.current = lineRef.current.slice(0, -1);
            term.write("\b \b");
          }
        } else if (code === 3) {
          // Ctrl+C: abandon the line.
          lineRef.current = "";
          term.write("^C\r\n");
          term.write(PROMPT);
        } else if (code >= 32) {
          lineRef.current += ch;
          term.write(ch);
        }
      }
    });

    // Seed from any tool events already folded into the store, then live-subscribe below.
    lastToolSeq.current = useStore.getState().toolSeq;

    return () => {
      sub.dispose();
      term.dispose();
      termRef.current = null;
    };
  }, []);

  // Echo new tool_progress rows (the shell tool's mirrored output) into the buffer as they arrive.
  useEffect(() => {
    const unsub = useStore.subscribe((s) => {
      const term = termRef.current;
      if (!term) return;
      if (s.toolSeq > lastToolSeq.current && s.tools.length) {
        lastToolSeq.current = s.toolSeq;
        const ev: ToolEvent = s.tools[s.tools.length - 1];
        writeToolLine(term, ev);
      }
    });
    return unsub;
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div
        ref={hostRef}
        style={{
          flex: 1,
          minHeight: 0,
          overflow: "hidden",
          padding: "var(--s2) var(--s3)",
          background: "var(--void)",
          borderRadius: "var(--radius)",
          boxShadow: "var(--panel-inset)",
        }}
      />
    </div>
  );
}

// Dispatch the command and acknowledge in-buffer; a rejected ack is surfaced (never swallowed).
function runLine(term: Xterm, cmd: string) {
  const argv = cmd.split(/\s+/);
  void sendIntent(intent.runCommand(argv, null)).then((ack) => {
    if (!ack.accepted) {
      term.writeln("\x1b[38;2;229;99;94m" + (ack.message ?? "command rejected") + "\x1b[0m");
    } else {
      term.writeln("\x1b[38;2;124;122;117mqueued › " + cmd + "\x1b[0m");
    }
    term.write(PROMPT);
  });
}

// Render a tool_progress row as a mirrored agent action (the shell tool side-effect).
function writeToolLine(term: Xterm, ev: ToolEvent) {
  term.writeln("\x1b[38;2;240;185;91m●\x1b[0m \x1b[38;2;168;166;161m" + ev.message + "\x1b[0m");
  term.write(PROMPT);
}
