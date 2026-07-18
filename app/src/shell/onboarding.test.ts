import { describe, it, expect, afterEach } from "vitest";
import { shouldShowOnboarding, isTauri, pickWorkspaceFolder, ONBOARDING_DONE_KEY } from "./onboarding";

describe("shouldShowOnboarding", () => {
  it("shows until a folder has been opened", () => {
    expect(shouldShowOnboarding(false)).toBe(true);
    expect(shouldShowOnboarding(true)).toBe(false);
  });
  it("exposes a stable persistence key", () => {
    expect(ONBOARDING_DONE_KEY).toBe("hide.folderOpened");
  });
});

describe("isTauri / pickWorkspaceFolder (no Tauri global)", () => {
  const g = globalThis as unknown as { window?: unknown };
  afterEach(() => {
    delete g.window;
  });

  it("isTauri is false without a window or Tauri global", () => {
    expect(isTauri()).toBe(false);
  });

  it("isTauri detects the injected Tauri global", () => {
    g.window = { __TAURI__: {} };
    expect(isTauri()).toBe(true);
  });

  it("pickWorkspaceFolder returns null when no native dialog is present", async () => {
    g.window = {}; // a window, but no __TAURI__.dialog
    await expect(pickWorkspaceFolder()).resolves.toBeNull();
  });

  it("pickWorkspaceFolder returns the chosen path when the dialog plugin is present", async () => {
    g.window = { __TAURI__: { dialog: { open: async () => "/Users/me/project" } } };
    await expect(pickWorkspaceFolder()).resolves.toBe("/Users/me/project");
  });

  it("pickWorkspaceFolder returns null when the picker is cancelled", async () => {
    g.window = { __TAURI__: { dialog: { open: async () => null } } };
    await expect(pickWorkspaceFolder()).resolves.toBeNull();
  });
});
