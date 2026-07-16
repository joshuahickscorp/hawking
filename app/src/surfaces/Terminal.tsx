/*
  Terminal.tsx: the integrated terminal (the workshop's mirrored shell). @xterm/xterm mounted and
  re-skinned to the v3 doctrine: --void grayscale background, a LIGHT caret (no gold), Geist Mono, and
  lichen/oxide ANSI mapped to the only two semantic tokens (--ok / --bad). NO terminal-green, NO blue,
  NO purple, NO gold. Human input over the (future) PTY WebSocket; here, a typed line dispatches
  RunCommand{argv,cwd} and the agent's tool_progress rows (the shell tool's mirrored output) are echoed
  into the same buffer so the surface is ALIVE. The prompt glyph is light entering the dark.

  Re-housed from the standard xterm IDE terminal; the chrome (prompt glyph, header) is HIDE mono.
*/
import { useEffect, useRef } from "react";
import { Terminal as Xterm } from "@xterm/xterm";
import type { ITheme } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import { sendIntent } from "../ipc";
import { useStore, type ToolEvent } from "../store";
import { intent } from "../wire";
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

    term.writeln("\x1b[38;2;110;109;104mhawking shell\x1b[0m");
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
          padding: "var(--ma-3) var(--ma-4)",
          background: "var(--void)",
          borderRadius: "var(--radius)",
          boxShadow: "var(--hairline), var(--inner-glow)",
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
      // oxide for a refusal (the only "bad" color), direct and blame-free.
      term.writeln("\x1b[38;2;192;128;122m" + (ack.message ?? "command rejected") + "\x1b[0m");
    } else {
      term.writeln("\x1b[38;2;110;109;104mqueued › " + cmd + "\x1b[0m");
    }
    term.write(PROMPT);
  });
}

// Render a tool_progress row as a mirrored agent action (the shell tool side-effect). The leading
// glyph is light (the agent's work entering the room), the message in --text-2.
function writeToolLine(term: Xterm, ev: ToolEvent) {
  term.writeln("\x1b[38;2;244;242;238m●\x1b[0m \x1b[38;2;155;154;149m" + ev.message + "\x1b[0m");
  term.write(PROMPT);
}
