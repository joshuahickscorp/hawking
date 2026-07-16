import { useEffect, useRef } from "react";
import { sendIntent } from "../ipc";
import { useStore, type RunPhase } from "../store";
import type { FileNode } from "../surfaces/types";
import { intent } from "../wire";

export function initGlass(): void {
  const root = document.documentElement;
  let refract = false;
  try {
    refract = typeof CSS !== "undefined" && !!CSS.supports && CSS.supports("backdrop-filter", "url(#p)");
  } catch {
    refract = false;
  }
  root.dataset.glass = refract ? "refract" : "frost";
  if (!refract || document.getElementById("hide-glass-defs")) return;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.id = "hide-glass-defs";
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("style", "position:absolute;width:0;height:0;pointer-events:none");
  svg.innerHTML =
    '<defs><filter id="hide-glass-refract" x="-2%" y="-2%" width="104%" height="104%">' +
    '<feTurbulence type="fractalNoise" baseFrequency="0.012 0.018" numOctaves="1" seed="7" result="n"/>' +
    '<feGaussianBlur in="n" stdDeviation="1.4" result="nb"/>' +
    '<feDisplacementMap in="SourceGraphic" in2="nb" scale="5" xChannelSelector="R" yChannelSelector="G"/>' +
    "</filter></defs>";
  document.body.appendChild(svg);
}

export const ONBOARDING_DONE_KEY = "hide.folderOpened";

export function shouldShowOnboarding(folderOpened: boolean): boolean {
  return !folderOpened;
}

export function isTauri(): boolean {
  if (typeof window === "undefined") return false;
  const value = window as unknown as Record<string, unknown>;
  return Boolean(value.__TAURI_INTERNALS__) || Boolean(value.__TAURI__);
}

export async function pickWorkspaceFolder(): Promise<string | null> {
  if (typeof window === "undefined") return null;
  const tauri = (window as unknown as { __TAURI__?: { dialog?: { open?: (opts: unknown) => Promise<unknown> } } }).__TAURI__;
  const open = tauri?.dialog?.open;
  if (!open) return null;
  try {
    const picked = await open({ directory: true, multiple: false, title: "Open a project folder" });
    return typeof picked === "string" ? picked : null;
  } catch {
    return null;
  }
}

export type Watermark = "normal" | "soft" | "warn" | "critical";
export interface CompactSignal {
  watermark?: Watermark;
  occupancy?: number;
  active: boolean;
}
export interface CompactPolicyState {
  lastFiredOccupancy: number;
  armed: boolean;
}
export const INITIAL_POLICY: CompactPolicyState = { lastFiredOccupancy: -1, armed: true };
const SOFT_GLIDE = 0.68;
const REARM_MARGIN = 0.15;
const ACTIVE_PHASES: ReadonlySet<RunPhase> = new Set(["planning", "executing", "awaiting"]);

export function shouldCompact(signal: CompactSignal, state: CompactPolicyState): boolean {
  if (!signal.active || !state.armed) return false;
  const watermark = signal.watermark ?? "normal";
  const occupancy = signal.occupancy ?? 0;
  return watermark === "critical" || watermark === "warn" || (watermark === "soft" && occupancy >= SOFT_GLIDE);
}

export function nextPolicyState(fired: boolean, occupancy: number, state: CompactPolicyState): CompactPolicyState {
  if (fired) return { lastFiredOccupancy: occupancy, armed: false };
  return !state.armed && occupancy <= Math.max(0, state.lastFiredOccupancy - REARM_MARGIN)
    ? { ...state, armed: true }
    : state;
}

export function useAutoCompact(): void {
  const watermark = useStore((state) => state.manifest?.live?.watermark) as Watermark | undefined;
  const occupancy = useStore((state) => state.manifest?.live?.occupancy);
  const runPhase = useStore((state) => state.runPhase);
  const sessionId = useStore((state) => state.sessionId);
  const state = useRef<CompactPolicyState>({ ...INITIAL_POLICY });

  useEffect(() => {
    const signal: CompactSignal = { watermark, occupancy, active: ACTIVE_PHASES.has(runPhase) };
    const fire = shouldCompact(signal, state.current);
    if (fire) {
      void sendIntent(
        intent.custom("compact_context", {
          session_id: sessionId,
          reason: "watermark",
          occupancy: signal.occupancy ?? null,
          watermark: signal.watermark ?? null,
        }),
      );
    }
    state.current = nextPolicyState(fire, signal.occupancy ?? 0, state.current);
  }, [watermark, occupancy, runPhase, sessionId]);
}

export const FOCUSABLE_SELECTOR =
  'a[href],button:not([disabled]),input:not([disabled]),textarea:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])';

export function cycleIndex(count: number, current: number, backward: boolean): number {
  if (count <= 0) return 0;
  const step = backward ? -1 : 1;
  return (((current + step) % count) + count) % count;
}

export interface FlatRow {
  path: string;
  dir: boolean;
  level: number;
  expanded: boolean;
}

export function flattenVisible(nodes: FileNode[], collapsed: Record<string, boolean>, level = 0): FlatRow[] {
  const out: FlatRow[] = [];
  for (const node of nodes) {
    const expanded = node.dir ? !collapsed[node.path] : false;
    out.push({ path: node.path, dir: node.dir, level, expanded });
    if (node.dir && expanded && node.children?.length) {
      out.push(...flattenVisible(node.children, collapsed, level + 1));
    }
  }
  return out;
}

export type TreeKey = "ArrowDown" | "ArrowUp" | "ArrowRight" | "ArrowLeft" | "Home" | "End";

export function treeKeyTarget(
  rows: FlatRow[],
  focusedPath: string,
  key: TreeKey,
): { kind: "focus"; path: string } | { kind: "toggle"; path: string; expand: boolean } | null {
  const index = rows.findIndex((row) => row.path === focusedPath);
  if (index < 0) return rows.length ? { kind: "focus", path: rows[0].path } : null;
  const row = rows[index];
  switch (key) {
    case "ArrowDown":
      return index < rows.length - 1 ? { kind: "focus", path: rows[index + 1].path } : null;
    case "ArrowUp":
      return index > 0 ? { kind: "focus", path: rows[index - 1].path } : null;
    case "Home":
      return rows.length ? { kind: "focus", path: rows[0].path } : null;
    case "End":
      return rows.length ? { kind: "focus", path: rows[rows.length - 1].path } : null;
    case "ArrowRight":
      if (row.dir && !row.expanded) return { kind: "toggle", path: row.path, expand: true };
      return row.dir && row.expanded && index < rows.length - 1
        ? { kind: "focus", path: rows[index + 1].path }
        : null;
    case "ArrowLeft":
      if (row.dir && row.expanded) return { kind: "toggle", path: row.path, expand: false };
      for (let i = index - 1; i >= 0; i--) {
        if (rows[i].level < row.level) return { kind: "focus", path: rows[i].path };
      }
      return null;
  }
}

export function useFocusTrap<T extends HTMLElement>(active = true) {
  const ref = useRef<T>(null);
  useEffect(() => {
    if (!active || typeof document === "undefined") return;
    const node = ref.current;
    if (!node) return;
    const previous = document.activeElement as HTMLElement | null;
    const list = () =>
      Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
        (element) => element.offsetParent !== null || element === document.activeElement,
      );
    (list()[0] ?? node).focus?.();
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const focusable = list();
      if (focusable.length === 0) {
        event.preventDefault();
        return;
      }
      const here = focusable.indexOf(document.activeElement as HTMLElement);
      const next = cycleIndex(focusable.length, here < 0 ? -1 : here, event.shiftKey);
      const edge = event.shiftKey ? here <= 0 : here === focusable.length - 1 || here < 0;
      if (edge) {
        event.preventDefault();
        focusable[next].focus();
      }
    };
    node.addEventListener("keydown", onKey);
    return () => {
      node.removeEventListener("keydown", onKey);
      previous?.focus?.();
    };
  }, [active]);
  return ref;
}

export const UPDATE_TARGETS = ["darwin-aarch64", "darwin-x86_64"] as const;
export type UpdateTarget = (typeof UPDATE_TARGETS)[number];
export interface UpdatePlatform {
  signature: string;
  url: string;
}
export interface UpdateManifest {
  version: string;
  notes: string;
  pub_date: string;
  platforms: Partial<Record<UpdateTarget, UpdatePlatform>>;
}
export interface ReleaseInput {
  version: string;
  notes?: string;
  pubDate: string;
  platforms: Partial<Record<UpdateTarget, UpdatePlatform>>;
}

export function buildUpdateManifest(input: ReleaseInput): UpdateManifest {
  if (!input.version.trim()) throw new Error("update manifest: version must not be empty");
  const targets = Object.keys(input.platforms) as UpdateTarget[];
  if (targets.length === 0) throw new Error("update manifest: at least one platform is required");
  for (const target of targets) {
    if (!UPDATE_TARGETS.includes(target)) throw new Error(`update manifest: unknown target ${target}`);
    const platform = input.platforms[target]!;
    if (!platform.signature.trim() || !platform.url.trim()) {
      throw new Error(`update manifest: ${target} needs both a signature and a url`);
    }
  }
  return {
    version: input.version,
    notes: input.notes ?? "",
    pub_date: input.pubDate,
    platforms: input.platforms,
  };
}

interface TauriUpdater {
  check?: () => Promise<{ available?: boolean; version?: string; currentVersion?: string } | null>;
}
function updater(): TauriUpdater | undefined {
  if (typeof window === "undefined") return undefined;
  return (window as unknown as { __TAURI__?: { updater?: TauriUpdater } }).__TAURI__?.updater;
}
export function isUpdaterAvailable(): boolean {
  return typeof updater()?.check === "function";
}
export interface UpdateCheck {
  available: boolean;
  version?: string;
}
export async function checkForUpdate(): Promise<UpdateCheck | null> {
  const check = updater()?.check;
  if (!check) return null;
  try {
    const result = await check();
    return result
      ? { available: Boolean(result.available), version: result.version }
      : { available: false };
  } catch {
    return null;
  }
}
