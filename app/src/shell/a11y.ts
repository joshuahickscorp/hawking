/*
  a11y.ts — shared accessibility primitives. Two halves:

  1. Pure helpers (no DOM) so the focus-cycling and tree-navigation math is unit-testable in the
     node-environment vitest suite: `cycleIndex`, `flattenVisible`, `treeKeyTarget`.
  2. `useFocusTrap` — a focus trap + restore for modal dialogs (Settings, the security gate, the
     command palette): focus the first control on mount, keep Tab/Shift+Tab inside, and return focus
     to whatever was focused before the dialog opened.
*/
import { useEffect, useRef } from "react";
import type { FileNode } from "../surfaces/ide/types";

// Selector for the elements a Tab cycle should visit inside a trapped container.
export const FOCUSABLE_SELECTOR =
  'a[href],button:not([disabled]),input:not([disabled]),textarea:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])';

// Next index when Tab (forward) or Shift+Tab (backward) wraps around a list of `count` focusables.
// Pure so the wrap logic is tested without a DOM. Returns 0 for an empty list.
export function cycleIndex(count: number, current: number, backward: boolean): number {
  if (count <= 0) return 0;
  const step = backward ? -1 : 1;
  return (((current + step) % count) + count) % count;
}

// A row in the flattened, currently-visible tree (collapsed dirs hide their children).
export interface FlatRow {
  path: string;
  dir: boolean;
  level: number;
  expanded: boolean;
}

// Depth-first flatten of the visible rows. A dir is "collapsed" when collapsed[path] === true
// (matching Explorer's state, where absence means open). Children of a collapsed dir are skipped.
export function flattenVisible(nodes: FileNode[], collapsed: Record<string, boolean>, level = 0): FlatRow[] {
  const out: FlatRow[] = [];
  for (const n of nodes) {
    const expanded = n.dir ? !collapsed[n.path] : false;
    out.push({ path: n.path, dir: n.dir, level, expanded });
    if (n.dir && expanded && n.children?.length) {
      out.push(...flattenVisible(n.children, collapsed, level + 1));
    }
  }
  return out;
}

export type TreeKey = "ArrowDown" | "ArrowUp" | "ArrowRight" | "ArrowLeft" | "Home" | "End";

// Resolve a tree keystroke against the flattened rows to either a navigation target (move focus to
// another row's path) or a structural action (expand/collapse the focused dir). Returns null for a
// no-op (e.g. ArrowRight on a file). Keeps the keyboard model in one tested place.
export function treeKeyTarget(
  rows: FlatRow[],
  focusedPath: string,
  key: TreeKey,
): { kind: "focus"; path: string } | { kind: "toggle"; path: string; expand: boolean } | null {
  const i = rows.findIndex((r) => r.path === focusedPath);
  if (i < 0) return rows.length ? { kind: "focus", path: rows[0].path } : null;
  const row = rows[i];
  switch (key) {
    case "ArrowDown":
      return i < rows.length - 1 ? { kind: "focus", path: rows[i + 1].path } : null;
    case "ArrowUp":
      return i > 0 ? { kind: "focus", path: rows[i - 1].path } : null;
    case "Home":
      return rows.length ? { kind: "focus", path: rows[0].path } : null;
    case "End":
      return rows.length ? { kind: "focus", path: rows[rows.length - 1].path } : null;
    case "ArrowRight":
      if (row.dir && !row.expanded) return { kind: "toggle", path: row.path, expand: true };
      if (row.dir && row.expanded && i < rows.length - 1) return { kind: "focus", path: rows[i + 1].path };
      return null;
    case "ArrowLeft":
      if (row.dir && row.expanded) return { kind: "toggle", path: row.path, expand: false };
      // collapse-to-parent: jump to the nearest shallower row above.
      for (let j = i - 1; j >= 0; j--) {
        if (rows[j].level < row.level) return { kind: "focus", path: rows[j].path };
      }
      return null;
    default:
      return null;
  }
}

// Focus trap for a modal container. Pass a stable boolean; when true, traps focus inside `ref` and
// restores it to the previously-focused element on teardown. SSR/jsdom-safe (guards on document).
export function useFocusTrap<T extends HTMLElement>(active = true) {
  const ref = useRef<T>(null);
  useEffect(() => {
    if (!active || typeof document === "undefined") return;
    const node = ref.current;
    if (!node) return;
    const prev = document.activeElement as HTMLElement | null;
    const list = () =>
      Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
        (el) => el.offsetParent !== null || el === document.activeElement,
      );
    const items = list();
    (items[0] ?? node).focus?.();

    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      const focusables = list();
      if (focusables.length === 0) {
        e.preventDefault();
        return;
      }
      const here = focusables.indexOf(document.activeElement as HTMLElement);
      const next = cycleIndex(focusables.length, here < 0 ? -1 : here, e.shiftKey);
      // Only intercept at the edges so normal Tab between controls still works natively.
      const atEdge = e.shiftKey ? here <= 0 : here === focusables.length - 1 || here < 0;
      if (atEdge) {
        e.preventDefault();
        focusables[next].focus();
      }
    };
    node.addEventListener("keydown", onKey);
    return () => {
      node.removeEventListener("keydown", onKey);
      prev?.focus?.();
    };
  }, [active]);
  return ref;
}
