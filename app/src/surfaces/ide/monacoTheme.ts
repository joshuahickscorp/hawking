/*
  monacoTheme.ts: the DOCTRINE Monaco theme. This is the load-bearing re-skin (Risk R5 / Self-check
  C15: "Monaco looks like VS Code"). Every Monaco color is pinned to a Part C token value so the editor
  reads as HIDE near-black anodized material with the gold radiation accent, NOT VS Code blue.

  Monaco's theme API takes literal hex, not CSS variables, so the Part C token values are mirrored here
  as constants. This file is the ONLY place those values are duplicated, and they trace 1:1 to theme.css.
  No blue, no purple anywhere (C3 off-limits). Gold cursor/selection. Geist Mono is set on the editor
  instance options (fontFamily), not in the theme.
*/
import { loader } from "@monaco-editor/react";
import type * as Monaco from "monaco-editor";

// Mirror of the Part C tokens (theme.css) that Monaco needs as literal hex.
const T = {
  void: "060606",
  surface0: "0B0B0C",
  surface1: "111113",
  surface2: "18181B",
  textHi: "F2F0EC",
  textMid: "A8A6A1",
  textLow: "7C7A75",
  radiation: "F0B95B",
  radiationBright: "FFD888",
  success: "6FBF8B",
  danger: "E5635E",
  warning: "E08A3C",
  diffAddBg: "6FBF8B1A", // jade @ ~10% (matches --diff-add-bg)
  diffDelBg: "E5635E1A", // red @ ~10%
} as const;

export const HIDE_THEME = "hide-observatory";
export const MONO_FONT = '"Geist Mono", ui-monospace, "SF Mono", Menlo, monospace';

// Build the theme object. Tokens are deliberately near-monochrome warm off-whites with ONE gold
// accent (C3 mood: muted desaturated across 95% of the surface, gold the one luminous thing).
export function hideMonacoTheme(): Monaco.editor.IStandaloneThemeData {
  return {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "", foreground: T.textHi, background: T.void },
      { token: "comment", foreground: T.textLow, fontStyle: "italic" },
      { token: "keyword", foreground: T.radiation },
      { token: "keyword.control", foreground: T.radiation },
      { token: "operator", foreground: T.textMid },
      { token: "string", foreground: T.success },
      { token: "number", foreground: T.warning },
      { token: "type", foreground: T.radiationBright },
      { token: "type.identifier", foreground: T.radiationBright },
      { token: "function", foreground: T.textHi },
      { token: "variable", foreground: T.textHi },
      { token: "variable.predefined", foreground: T.textMid },
      { token: "identifier", foreground: T.textHi },
      { token: "delimiter", foreground: T.textMid },
      { token: "tag", foreground: T.radiation },
      { token: "attribute.name", foreground: T.radiationBright },
      { token: "invalid", foreground: T.danger },
    ],
    colors: {
      "editor.background": "#" + T.void,
      "editor.foreground": "#" + T.textHi,
      // Gutter / chrome from the near-black ramp, never the VS Code blue-grey.
      "editorGutter.background": "#" + T.void,
      "editorLineNumber.foreground": "#" + T.textLow,
      "editorLineNumber.activeForeground": "#" + T.radiation,
      "editorIndentGuide.background1": "#1A1A1D",
      "editorIndentGuide.activeBackground1": "#2A2A2E",
      "editorWhitespace.foreground": "#222226",
      // The gold cursor + selection accents (C3: the one luminous thing).
      "editorCursor.foreground": "#" + T.radiationBright,
      "editor.selectionBackground": "#" + T.radiation + "33",
      "editor.inactiveSelectionBackground": "#" + T.radiation + "1A",
      "editor.selectionHighlightBackground": "#" + T.radiation + "22",
      "editor.wordHighlightBackground": "#" + T.radiation + "1A",
      "editor.findMatchBackground": "#" + T.radiation + "44",
      "editor.findMatchHighlightBackground": "#" + T.radiation + "22",
      "editor.lineHighlightBackground": "#0B0B0C",
      "editor.lineHighlightBorder": "#00000000",
      // Material surfaces for the floaty bits (no blue).
      "editorWidget.background": "#" + T.surface1,
      "editorWidget.border": "#1C1C20",
      "editorSuggestWidget.background": "#" + T.surface1,
      "editorSuggestWidget.selectedBackground": "#" + T.surface2,
      "editorSuggestWidget.highlightForeground": "#" + T.radiation,
      "editorHoverWidget.background": "#" + T.surface1,
      "editorHoverWidget.border": "#1C1C20",
      "scrollbarSlider.background": "#FFFFFF14",
      "scrollbarSlider.hoverBackground": "#FFFFFF26",
      "scrollbarSlider.activeBackground": "#FFFFFF33",
      // Diff colors: jade-add / red-del at ~10%, paired ALWAYS with +/- gutter markers (C3 / C14).
      "diffEditor.insertedTextBackground": "#" + T.diffAddBg,
      "diffEditor.removedTextBackground": "#" + T.diffDelBg,
      "diffEditor.insertedLineBackground": "#6FBF8B12",
      "diffEditor.removedLineBackground": "#E5635E12",
      "diffEditorGutter.insertedLineBackground": "#6FBF8B22",
      "diffEditorGutter.removedLineBackground": "#E5635E22",
      "diffEditor.diagonalFill": "#1A1A1D",
      // Bracket matching glows gold, not blue.
      "editorBracketMatch.background": "#" + T.radiation + "22",
      "editorBracketMatch.border": "#" + T.radiation + "55",
      "editorOverviewRuler.border": "#00000000",
      "editorError.foreground": "#" + T.danger,
      "editorWarning.foreground": "#" + T.warning,
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
  fontSize: 12.5,
  lineHeight: 20,
  fontLigatures: false,
  letterSpacing: 0.2,
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  renderLineHighlight: "line",
  cursorBlinking: "smooth",
  cursorSmoothCaretAnimation: "on",
  smoothScrolling: true,
  roundedSelection: false,
  guides: { indentation: true, bracketPairs: false },
  padding: { top: 10, bottom: 10 },
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
