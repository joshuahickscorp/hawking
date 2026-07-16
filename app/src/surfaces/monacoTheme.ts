/*
  monacoTheme.ts: the DOCTRINE Monaco theme (v3, Tadao Ando grayscale concrete + light).
  This is the load-bearing re-skin (Self-check: "Monaco looks like VS Code"). Every Monaco color is
  pinned to a v3 theme.css token value so the editor reads as a HIDE concrete chamber where the only
  accent is LIGHT entering the dark, NOT VS Code blue and NOT the retired v2 gold.

  Monaco's theme API takes literal hex, not CSS variables, so the v3 token values are mirrored here as
  constants. This file is the ONLY place those values are duplicated, and they trace 1:1 to theme.css.
  Absolute nevers: no blue, no purple, no gold, no yellow. Keywords and types are LIGHT; strings are ok
  (lichen); comments are text-3; diffs are ok/bad (lichen/oxide), always paired with +/- gutter markers.
  The caret is light. Geist Mono is set on the editor instance options (fontFamily), not in the theme.
*/
import { loader } from "@monaco-editor/react";
import type * as Monaco from "monaco-editor";
import { MONO_FONT } from "./ideConstants";

// Mirror of the v3 tokens (theme.css) that Monaco needs as literal hex (no leading #).
const T = {
  void: "070707", // --void (the unlit chamber, editor background)
  concrete1: "0E0E0F", // --concrete-1
  concrete2: "141416", // --concrete-2
  concrete3: "1B1B1E", // --concrete-3
  concrete4: "222226", // --concrete-4
  text1: "ECEAE6", // --text-1 (chalk)
  text2: "9B9A95", // --text-2
  text3: "8A887F", // --text-3
  mute: "5C5B57", // --mute
  light: "F4F2EE", // --light (the only accent; keywords + types live here)
  ok: "7E9E86", // --ok (lichen)  -> strings, additions
  bad: "C0807A", // --bad (oxide) -> deletions, errors
  okBg: "7E9E8614", // ~ --ok-bg, hex8 for Monaco line/text bg (lichen @ ~8%)
  badBg: "C0807A14", // ~ --bad-bg (oxide @ ~8%)
} as const;

export const HIDE_THEME = "hide-observatory";

// Build the theme object. Tokens are deliberately near-monochrome warm ash-grays; LIGHT is the one
// luminous thing (keywords, types, the caret), with ok/bad reserved for strings and diffs only.
export function hideMonacoTheme(): Monaco.editor.IStandaloneThemeData {
  return {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "", foreground: T.text1, background: T.void },
      { token: "comment", foreground: T.text3, fontStyle: "italic" },
      { token: "keyword", foreground: T.light },
      { token: "keyword.control", foreground: T.light },
      { token: "operator", foreground: T.text2 },
      { token: "string", foreground: T.ok },
      { token: "number", foreground: T.text1 },
      { token: "type", foreground: T.light },
      { token: "type.identifier", foreground: T.light },
      { token: "function", foreground: T.text1 },
      { token: "variable", foreground: T.text1 },
      { token: "variable.predefined", foreground: T.text2 },
      { token: "identifier", foreground: T.text1 },
      { token: "delimiter", foreground: T.text2 },
      { token: "tag", foreground: T.light },
      { token: "attribute.name", foreground: T.light },
      { token: "invalid", foreground: T.bad },
    ],
    colors: {
      "editor.background": "#" + T.void,
      "editor.foreground": "#" + T.text1,
      // Gutter / chrome from the concrete ramp, never the VS Code blue-grey.
      "editorGutter.background": "#" + T.void,
      "editorLineNumber.foreground": "#" + T.mute,
      "editorLineNumber.activeForeground": "#" + T.light,
      "editorIndentGuide.background1": "#" + T.concrete2,
      "editorIndentGuide.activeBackground1": "#" + T.concrete4,
      "editorWhitespace.foreground": "#" + T.concrete4,
      // The LIGHT caret + selection accents (the one luminous thing; no gold).
      "editorCursor.foreground": "#" + T.light,
      "editor.selectionBackground": "#F4F2EE22",
      "editor.inactiveSelectionBackground": "#F4F2EE12",
      "editor.selectionHighlightBackground": "#F4F2EE18",
      "editor.wordHighlightBackground": "#F4F2EE12",
      "editor.findMatchBackground": "#F4F2EE33",
      "editor.findMatchHighlightBackground": "#F4F2EE1A",
      "editor.lineHighlightBackground": "#" + T.concrete1,
      "editor.lineHighlightBorder": "#07070700",
      // Material surfaces for the floaty bits (concrete tiers, no blue).
      "editorWidget.background": "#" + T.concrete2,
      "editorWidget.border": "#" + T.concrete3,
      "editorSuggestWidget.background": "#" + T.concrete2,
      "editorSuggestWidget.selectedBackground": "#" + T.concrete3,
      "editorSuggestWidget.highlightForeground": "#" + T.light,
      "editorHoverWidget.background": "#" + T.concrete2,
      "editorHoverWidget.border": "#" + T.concrete3,
      "scrollbarSlider.background": "#F4F2EE10",
      "scrollbarSlider.hoverBackground": "#F4F2EE1C",
      "scrollbarSlider.activeBackground": "#F4F2EE2A",
      // Diff colors: lichen-add / oxide-del at ~8%, paired ALWAYS with +/- gutter markers.
      "diffEditor.insertedTextBackground": "#" + T.okBg,
      "diffEditor.removedTextBackground": "#" + T.badBg,
      "diffEditor.insertedLineBackground": "#7E9E860F",
      "diffEditor.removedLineBackground": "#C0807A0F",
      "diffEditorGutter.insertedLineBackground": "#7E9E861C",
      "diffEditorGutter.removedLineBackground": "#C0807A1C",
      "diffEditor.diagonalFill": "#" + T.concrete2,
      // Bracket matching glows with light, not blue, not gold.
      "editorBracketMatch.background": "#F4F2EE14",
      "editorBracketMatch.border": "#F4F2EE33",
      "editorOverviewRuler.border": "#07070700",
      "editorError.foreground": "#" + T.bad,
      "editorWarning.foreground": "#" + T.text2,
    },
  };
}

let registered = false;

// Register the theme once with the shared monaco instance the @monaco-editor/react loader uses.
// Called from a beforeMount handler so it is in place before the first editor paints.
export function registerHideTheme(monaco: typeof Monaco): void {
  if (registered) return;
  monaco.editor.defineTheme(HIDE_THEME, hideMonacoTheme());
  registered = true;
}

// Editor options shared by the plain editor and the diff editor: Geist Mono, calm chrome,
// no minimap clutter, accessible. NO VS Code default look survives this.
export const HIDE_EDITOR_OPTIONS: Monaco.editor.IStandaloneEditorConstructionOptions = {
  fontFamily: MONO_FONT,
  fontSize: 13,
  lineHeight: 21,
  fontLigatures: true,
  letterSpacing: 0.2,
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  renderLineHighlight: "line",
  cursorBlinking: "smooth",
  cursorSmoothCaretAnimation: "on",
  smoothScrolling: true,
  roundedSelection: false,
  guides: { indentation: true, bracketPairs: false },
  padding: { top: 16, bottom: 16 },
  overviewRulerLanes: 0,
  scrollbar: { verticalScrollbarSize: 10, horizontalScrollbarSize: 10, useShadows: false },
  renderWhitespace: "none",
  occurrencesHighlight: "singleFile",
  contextmenu: false,
  automaticLayout: true,
};

/*
  Point the @monaco-editor/react loader at the locally-bundled monaco (the npm package, no CDN:
  honors the air-gap ethos in theme.css) and wire its web workers via Vite's ?worker imports
  (so editor features run off the main thread; no CDN worker fetch). Idempotent.
*/
import EditorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import TsWorker from "monaco-editor/esm/vs/language/typescript/ts.worker?worker";
import JsonWorker from "monaco-editor/esm/vs/language/json/json.worker?worker";
import CssWorker from "monaco-editor/esm/vs/language/css/css.worker?worker";
import HtmlWorker from "monaco-editor/esm/vs/language/html/html.worker?worker";

let loaderConfigured = false;

export function configureMonacoLoader(): void {
  if (loaderConfigured) return;
  // The web-worker factory Monaco reads at runtime (label -> worker for that language service).
  (self as unknown as { MonacoEnvironment: { getWorker: (id: string, label: string) => Worker } }).MonacoEnvironment = {
    getWorker(_id, label) {
      if (label === "typescript" || label === "javascript") return new TsWorker();
      if (label === "json") return new JsonWorker();
      if (label === "css" || label === "scss" || label === "less") return new CssWorker();
      if (label === "html" || label === "handlebars" || label === "razor") return new HtmlWorker();
      return new EditorWorker();
    },
  };
  loader.config({ monaco });
  loaderConfigured = true;
}

// Import the bundled monaco so the loader uses it instead of fetching from a CDN.
import * as monaco from "monaco-editor";
