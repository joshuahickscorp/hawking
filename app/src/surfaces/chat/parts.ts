/*
  chat/parts.ts: the shapes the AI Chat surface reads out of the store's generic projection bag,
  plus the small style atoms shared across the chat helper components.

  The host emits plan/diff_chip state as projection_patch{projection:"plan"|"diff_chip", patch}
  (D-reference, the Custom-name + projection registry). store.routeProjection has no typed slice for
  these yet, so they land in s.projections.plan / s.projections.diff_chip as a shallow-merged bag.
  These interfaces are the FE's read view of that bag (kept tolerant: every field optional, because a
  patch is a partial state-diff and an early patch may carry only a subset).
*/
import type { CSSProperties } from "react";

// projection_patch:plan -> the ordered, steerable plan the agent proposes then executes.
export type PlanStepStatus = "pending" | "active" | "done" | "failed" | "skipped";
export interface PlanStep {
  id: string;
  title: string;
  status?: PlanStepStatus;
  detail?: string;
}
export interface PlanPatch {
  run_id?: string;
  // suggest-only -> the plan waits for Approve; auto -> it is already executing.
  awaiting_approval?: boolean;
  steps?: PlanStep[];
}

// projection_patch:diff_chip -> a produced diff, rendered as a chip that opens the hunk review.
export interface DiffChip {
  diff_id: string;
  run_id?: string;
  path: string;
  added?: number;
  removed?: number;
  status?: "proposed" | "applied" | "rejected" | "stale";
}
export interface DiffChipPatch {
  // the host may send the whole set or a single chip; tolerate both.
  chips?: DiffChip[];
}

// ---- shared style atoms (kept here so every chat helper reads from one place) ----

// A chip: a small recessed pill of concrete, shadow-line for its edge, mono telemetry label.
// No fill accent: a chip is a quiet object in the void, brightened only by its glyph when it matters.
export const chip: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: "var(--ma-2)",
  padding: "var(--ma-1) var(--ma-3)",
  borderRadius: "var(--radius-pill)",
  fontSize: "12px",
  color: "var(--text-2)",
  background: "var(--concrete-2)",
  boxShadow: "var(--hairline)",
  maxWidth: "100%",
};

// The dim mono caps label used to title an inline structure block (plan, tools, diffs).
// Mirrors the .t-label role: 600/uppercase/tracked, in --mute.
export const blockLabel: CSSProperties = {
  fontWeight: 500,
  fontSize: "12px",
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  color: "var(--mute)",
};

// A small control button (Cancel / Pause / Resume / Approve), shape + tone, never color-only.
// `lit` is the engaged/affirmative control: it catches the light (text -> --light, a soft bloom);
// the rest are quiet recessed concrete that brightens on the focus ring.
export function ctlStyle(lit: boolean): CSSProperties {
  return {
    padding: "var(--ma-1) var(--ma-3)",
    borderRadius: "var(--radius)",
    fontSize: "12px",
    fontWeight: 500,
    color: lit ? "var(--light)" : "var(--text-2)",
    background: "var(--concrete-3)",
    boxShadow: lit ? "var(--hairline-strong), var(--light-bloom)" : "var(--hairline)",
  };
}

// Status -> a marker glyph + tone, so the plan-step / diff state never reads as color alone.
// The two pigments (--ok / --bad) appear only on terminal done/failed; in-flight and pending
// states are read by light and grayscale (the active step also breathes via .alive at the call site).
export const STEP_MARK: Record<PlanStepStatus, { glyph: string; color: string }> = {
  pending: { glyph: "○", color: "var(--text-3)" },
  active: { glyph: "◉", color: "var(--light)" },
  done: { glyph: "●", color: "var(--ok)" },
  failed: { glyph: "✕", color: "var(--bad)" },
  skipped: { glyph: "/", color: "var(--text-3)" },
};
