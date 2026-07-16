/*
  ChatParts.ts: shapes the AI Chat surface reads from the generic projection bag,
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

// A chip: a small flat pill (VS Code style), surface-2 fill with a thin border, muted label.
export const chip: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: "var(--ma-2)",
  padding: "3px var(--ma-2)",
  borderRadius: "var(--radius-sm)",
  fontSize: "var(--fs-small)",
  color: "var(--text-muted)",
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  maxWidth: "100%",
};

// The dim caps label used to title an inline structure block (plan, tools, diffs).
export const blockLabel: CSSProperties = {
  fontWeight: 600,
  fontSize: "var(--fs-label)",
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  color: "var(--text-dim)",
};

// A small control button (Cancel / Pause / Resume / Approve), chip-style flat button.
// `lit` is the engaged/affirmative control: filled accent. The rest are quiet bordered surface.
export function ctlStyle(lit: boolean): CSSProperties {
  return {
    padding: "3px var(--ma-2)",
    borderRadius: "var(--radius-sm)",
    fontSize: "var(--fs-small)",
    fontWeight: 500,
    color: lit ? "var(--accent-text)" : "var(--text)",
    background: lit ? "var(--accent)" : "var(--surface-2)",
    border: lit ? "1px solid var(--accent)" : "1px solid var(--border-strong)",
  };
}

// Status -> a marker glyph + tone, so the plan-step / diff state never reads as color alone.
export const STEP_MARK: Record<PlanStepStatus, { glyph: string; color: string }> = {
  pending: { glyph: "○", color: "var(--text-dim)" },
  active: { glyph: "◉", color: "var(--accent)" },
  done: { glyph: "●", color: "var(--green)" },
  failed: { glyph: "✕", color: "var(--red)" },
  skipped: { glyph: "/", color: "var(--text-dim)" },
};
