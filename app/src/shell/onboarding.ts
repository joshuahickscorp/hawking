/*
  onboarding.ts — first-run / no-folder logic, kept transport- and DOM-free so the decision and the
  optional native folder-picker are unit-testable in the node-environment vitest suite.

  The first-run surface (`surfaces/Onboarding.tsx`) shows until the user opens a project folder or
  chooses to continue with the sample workspace; the choice persists in localStorage so it is shown
  once. The actual native folder picker is the Tauri dialog plugin, detected at runtime via the global
  Tauri API (`withGlobalTauri`) so this module has NO build-time dependency on the plugin: when the
  plugin is bundled the picker lights up; in the browser / dev / a build without it, it degrades to
  "no native dialog" and the sample-workspace path still works.
*/

// localStorage key for "the user has chosen a workspace (or skipped to the sample one)".
export const ONBOARDING_DONE_KEY = "hide.folderOpened";

// Whether to show the first-run onboarding surface. Shown until a folder is opened (or skipped).
export function shouldShowOnboarding(folderOpened: boolean): boolean {
  return !folderOpened;
}

// Runtime detection of the Tauri webview (no import, so it is build-safe everywhere).
export function isTauri(): boolean {
  if (typeof window === "undefined") return false;
  const w = window as unknown as Record<string, unknown>;
  return Boolean(w.__TAURI_INTERNALS__) || Boolean(w.__TAURI__);
}

// Open the native folder picker if the Tauri dialog plugin is present, else return null (web/dev, or
// a build without the plugin). Returns the chosen absolute path, or null if cancelled/unavailable.
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
