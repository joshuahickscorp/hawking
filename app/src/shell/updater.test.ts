import { describe, it, expect, afterEach } from "vitest";
import {
  buildUpdateManifest,
  isUpdaterAvailable,
  checkForUpdate,
  UPDATE_TARGETS,
  type ReleaseInput,
} from "./policies";

const base: ReleaseInput = {
  version: "0.2.0",
  notes: "Faster state forks.",
  pubDate: "2026-06-29T00:00:00Z",
  platforms: {
    "darwin-aarch64": { signature: "sig-aarch64", url: "https://dl.hide.dev/HIDE_0.2.0_aarch64.app.tar.gz" },
  },
};

describe("buildUpdateManifest", () => {
  it("produces the Tauri v2 feed shape", () => {
    const m = buildUpdateManifest(base);
    expect(m).toEqual({
      version: "0.2.0",
      notes: "Faster state forks.",
      pub_date: "2026-06-29T00:00:00Z",
      platforms: {
        "darwin-aarch64": { signature: "sig-aarch64", url: "https://dl.hide.dev/HIDE_0.2.0_aarch64.app.tar.gz" },
      },
    });
  });
  it("defaults notes to an empty string", () => {
    const m = buildUpdateManifest({ ...base, notes: undefined });
    expect(m.notes).toBe("");
  });
  it("rejects an empty version", () => {
    expect(() => buildUpdateManifest({ ...base, version: "  " })).toThrow(/version/);
  });
  it("rejects a feed with no platforms", () => {
    expect(() => buildUpdateManifest({ ...base, platforms: {} })).toThrow(/platform/);
  });
  it("rejects an unknown target", () => {
    expect(() =>
      buildUpdateManifest({ ...base, platforms: { "linux-x86_64": { signature: "s", url: "u" } } as never }),
    ).toThrow(/unknown target/);
  });
  it("rejects a platform missing a signature or url", () => {
    expect(() =>
      buildUpdateManifest({ ...base, platforms: { "darwin-aarch64": { signature: "", url: "u" } } }),
    ).toThrow(/signature/);
  });
  it("exposes the known targets", () => {
    expect(UPDATE_TARGETS).toContain("darwin-aarch64");
  });
});

describe("isUpdaterAvailable / checkForUpdate (runtime-detected)", () => {
  const g = globalThis as unknown as { window?: unknown };
  afterEach(() => {
    delete g.window;
  });

  it("is unavailable without the Tauri updater plugin", async () => {
    expect(isUpdaterAvailable()).toBe(false);
    g.window = {};
    expect(isUpdaterAvailable()).toBe(false);
    await expect(checkForUpdate()).resolves.toBeNull();
  });

  it("reports an available update when the plugin returns one", async () => {
    g.window = { __TAURI__: { updater: { check: async () => ({ available: true, version: "0.2.0" }) } } };
    expect(isUpdaterAvailable()).toBe(true);
    await expect(checkForUpdate()).resolves.toEqual({ available: true, version: "0.2.0" });
  });

  it("reports no update when the plugin returns null", async () => {
    g.window = { __TAURI__: { updater: { check: async () => null } } };
    await expect(checkForUpdate()).resolves.toEqual({ available: false });
  });

  it("degrades to null when the check throws", async () => {
    g.window = { __TAURI__: { updater: { check: async () => { throw new Error("offline"); } } } };
    await expect(checkForUpdate()).resolves.toBeNull();
  });
});
