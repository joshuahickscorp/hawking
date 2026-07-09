/*
  ideConstants.ts — plain IDE constants with NO Monaco import, so lightweight consumers (the xterm
  terminal) can share them without pulling the ~4.5MB editor into their chunk. monacoTheme.ts re-uses
  MONO_FONT from here; keeping it Monaco-free is what lets Monaco stay lazy behind the Code chamber.
*/
export const MONO_FONT = '"Geist Mono", ui-monospace, "SF Mono", Menlo, monospace';
