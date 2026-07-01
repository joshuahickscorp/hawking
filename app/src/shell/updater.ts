/*
  updater.ts — auto-update scaffold. Two halves, both transport/DOM-free so they unit-test in the
  node vitest suite:

  1. `buildUpdateManifest` — produces the Tauri v2 updater feed (`latest.json`) shape from release
     inputs, so the release pipeline can generate a correct feed deterministically.
  2. `isUpdaterAvailable` / `checkForUpdate` — runtime-detected access to the Tauri updater plugin via
     the global Tauri API (`withGlobalTauri`). NO build-time dependency on `@tauri-apps/plugin-updater`
     (which is fetched at release time): when the plugin is bundled the check lights up; on web / dev /
     a build without it, it degrades to "no updater" so the Settings affordance is never a dead end.

  Going live additionally needs (all infra, not code): the updater plugin added to the desktop crate +
  `@tauri-apps/plugin-updater`, a generated minisign keypair (`tauri signer generate`), the `plugins.
  updater` block in tauri.conf.json (template in docs/plans/hide_release_autoupdate.md), and a host that
  serves the feed. See that doc for the exact steps.
*/

// The Tauri updater build targets HIDE ships (Apple Silicon today; the others are ready for when the
// release matrix widens).
export const UPDATE_TARGETS = ["darwin-aarch64", "darwin-x86_64"] as const;
export type UpdateTarget = (typeof UPDATE_TARGETS)[number];

export interface UpdatePlatform {
  /** minisign signature of the artifact (from `tauri signer sign`). */
  signature: string;
  /** absolute URL of the update artifact (e.g. the `.app.tar.gz`). */
  url: string;
}

// The `latest.json` Tauri v2 updater feed shape.
export interface UpdateManifest {
  version: string;
  notes: string;
  pub_date: string;
  platforms: Partial<Record<UpdateTarget, UpdatePlatform>>;
}

export interface ReleaseInput {
  version: string;
  notes?: string;
  /** ISO-8601 publish timestamp (the caller stamps it; this module never reads the clock). */
  pubDate: string;
  platforms: Partial<Record<UpdateTarget, UpdatePlatform>>;
}

// Build a valid updater feed from a release input. Throws on an empty version or no platforms so a
// malformed feed can never be published silently.
export function buildUpdateManifest(input: ReleaseInput): UpdateManifest {
  if (!input.version.trim()) throw new Error("update manifest: version must not be empty");
  const targets = Object.keys(input.platforms) as UpdateTarget[];
  if (targets.length === 0) throw new Error("update manifest: at least one platform is required");
  for (const t of targets) {
    if (!UPDATE_TARGETS.includes(t)) throw new Error(`update manifest: unknown target ${t}`);
    const p = input.platforms[t]!;
    if (!p.signature.trim() || !p.url.trim()) {
      throw new Error(`update manifest: ${t} needs both a signature and a url`);
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

// Whether the Tauri updater plugin is reachable at runtime.
export function isUpdaterAvailable(): boolean {
  return typeof updater()?.check === "function";
}

export interface UpdateCheck {
  available: boolean;
  version?: string;
}

// Check the feed for an update. Returns null when the updater is unavailable (web / dev / no plugin)
// so callers can show "updates are managed by the desktop app" rather than fail.
export async function checkForUpdate(): Promise<UpdateCheck | null> {
  const check = updater()?.check;
  if (!check) return null;
  try {
    const res = await check();
    if (!res) return { available: false };
    return { available: Boolean(res.available), version: res.version };
  } catch {
    return null;
  }
}
