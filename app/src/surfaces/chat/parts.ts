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

// A chip: a small rim-lit recessed pill in the near-black material, mono caps-ish telemetry label.
export const chip: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: "var(--s2)",
  padding: "3px 9px",
  borderRadius: 999,
  fontSize: "var(--text-xs)",
  color: "var(--text-mid)",
  background: "var(--surface-1)",
  boxShadow: "inset 0 0 0 1px var(--rim)",
  maxWidth: "100%",
};

// The dim mono caps label used to title an inline structure block (plan, tools, diffs).
export const blockLabel: CSSProperties = {
  fontSize: "var(--text-xs)",
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  color: "var(--text-low)",
};

// A small lit/ghost control button (Cancel / Pause / Resume / Approve), shaped not color-only.
export function ctlStyle(lit: boolean): CSSProperties {
  return {
    padding: "4px 11px",
    borderRadius: "var(--radius)",
    fontSize: "var(--text-xs)",
    color: lit ? "var(--void)" : "var(--text-mid)",
    background: lit ? "var(--radiation-bright)" : "var(--surface-2)",
    boxShadow: lit ? "0 0 14px -4px var(--radiation-bloom)" : "inset 0 0 0 1px var(--rim)",
  };
}

// Status -> a marker glyph + color, so the plan-step / diff state never reads as color alone.
export const STEP_MARK: Record<PlanStepStatus, { glyph: string; color: string }> = {
  pending: { glyph: "○", color: "var(--text-low)" },
  active: { glyph: "◉", color: "var(--radiation)" },
  done: { glyph: "●", color: "var(--success)" },
  failed: { glyph: "✕", color: "var(--danger)" },
  skipped: { glyph: "-", color: "var(--text-low)" },
};
