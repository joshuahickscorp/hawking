/**
 * Shared store / IPC fixtures for app test files.
 * Keeps mock transport, store apply helpers, and source-walkers in one place
 * so surface tests stop re-declaring the same stubs.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { vi } from "vitest";

/** Every TypeScript source file under a directory (recursive). */
export function walk(dir: string, opts: { excludeTests?: boolean } = {}): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p, opts));
    else if (/\.(ts|tsx)$/.test(name) && !(opts.excludeTests && /\.test\.ts$/.test(name))) out.push(p);
  }
  return out;
}

/** Strip block and line comments so source assertions match code, not prose. */
export const stripComments = (src: string): string =>
  src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

/** Read a source file relative to `baseDir`, optionally stripping comments. */
export const readSrc = (baseDir: string, rel: string, strip = true): string => {
  const raw = readFileSync(join(baseDir, rel), "utf8");
  return strip ? stripComments(raw) : raw;
};

/** Let a microtask / dispatch promise settle. */
export const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

/**
 * Hoisted IPC mock state shared by store-facing tests.
 * Call `installIpcMock()` once at module top (before importing store).
 */
export type IpcMockState = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  sent: any[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  calls: any[];
  reply: { accepted: boolean; held?: boolean; message?: string | null };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  host: { hits: any[]; connector: any; fail: boolean };
  reset: () => void;
};

/** Create hoisted mock bags. Pass the result into vi.mock factories. */
export function createIpcMockState(): IpcMockState {
  const state: IpcMockState = {
    sent: [],
    calls: [],
    reply: { accepted: true },
    host: { hits: [], connector: null, fail: false },
    reset() {
      state.sent.length = 0;
      state.calls.length = 0;
      state.reply.accepted = true;
      state.reply.held = undefined;
      state.reply.message = undefined;
      state.host.hits = [];
      state.host.connector = null;
      state.host.fail = false;
    },
  };
  return state;
}

/**
 * Standard accept-everything IPC mock. `state` must come from vi.hoisted.
 * Optional `onIntent` lets a suite inject side effects (search results, etc.).
 */
export function ipcMockModule(
  state: IpcMockState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onIntent?: (intent: any, listeners: Set<(ev: any) => void>) => void,
) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const listeners = new Set<(ev: any) => void>();
  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    sendIntent: async (i: any) => {
      state.sent.push(i);
      onIntent?.(i, listeners);
      return {
        accepted: state.reply.accepted,
        held: state.reply.held,
        event_seq: 1,
        message:
          state.reply.message ??
          (state.reply.accepted ? null : "the host refused that"),
      };
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    subscribeUi: (on: any) => {
      listeners.add(on);
      return () => listeners.delete(on);
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    callConnector: async (id: string, method: string, params: any) => {
      state.calls.push({ id, method, params });
      if (state.host.fail) throw new Error("index offline");
      return state.host.connector;
    },
    TRANSPORT_KIND: "mock",
  };
}

/**
 * Apply a UiEvent kind to the store under test. `apply` is the store method.
 * Loose typing: vitest suites are excluded from production tsc.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const applyKind = (apply: (ev: any) => void, kind: any, session_id: string | null = "ses_x", seq = 1) =>
  apply({ seq, session_id, kind });
