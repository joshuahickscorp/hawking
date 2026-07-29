import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";

vi.mock("../../ipc", () => ({
  TRANSPORT_KIND: "live",
  sendIntent: async () => ({ accepted: true, event_seq: 1, message: null }),
  subscribeUi: () => () => {},
  callConnector: async () => null,
}));

import { COMMANDS } from "../../store";
import { MOCK_DIFF, MOCK_FILE_BODY, MOCK_TREE, mockOnly } from "./types";

const here = fileURLToPath(new URL(".", import.meta.url));
const read = (p: string) => readFileSync(here + p, "utf8");

function sources(dir: string): string[] {
  const out: string[] = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = dir + e.name;
    if (e.isDirectory()) out.push(...sources(p + "/"));
    else if (/\.tsx?$/.test(e.name) && !e.name.endsWith(".test.ts")) out.push(p);
  }
  return out;
}

describe("mock fixtures on a live transport", () => {
  it("hands back nothing, whatever fixture is asked for", () => {
    expect(mockOnly(MOCK_TREE)).toBeNull();
    expect(mockOnly(MOCK_FILE_BODY)).toBeNull();
    expect(mockOnly(MOCK_DIFF)).toBeNull();
    expect(mockOnly(MOCK_TREE) ?? []).toEqual([]);
    expect(mockOnly(MOCK_FILE_BODY)?.["crates/pool/src/guard.rs"] ?? null).toBeNull();
  });

  it("is the ONLY way the file surfaces reach a fixture", () => {
    for (const [file, name] of [
      ["Explorer.tsx", "MOCK_TREE"],
      ["Editor.tsx", "MOCK_FILE_BODY"],
    ] as const) {
      const uses = read(file)
        .split("\n")
        .filter((l) => l.includes(name) && !l.trimStart().startsWith("import"));
      expect(uses.length).toBeGreaterThan(0);
      for (const line of uses) expect(line).toContain(`mockOnly(${name})`);
    }
  });

  it("never opens a fixture path as the boot tab", () => {
    const app = read("../../App.tsx");
    expect(app).toContain('TRANSPORT_KIND === "mock" ? "crates/pool/src/guard.rs" : null');
    expect(app).toContain("INITIAL_FILE ? [INITIAL_FILE] : []");
  });

  it("never mounts an editable buffer over a body that did not load", () => {
    const src = read("Editor.tsx");
    expect(src).not.toContain("host streams this buffer as projection_patch");
    expect(src).toContain("if (loadError || !body)");
  });
});

describe("retired custom names", () => {
  const RETIRED = ["create_pr", "switch_profile", "focus_run", "edit_hunk", "open_folder", "compact_context"];
  const dispatched = (src: string, name: string) => src.includes(`runCommand("${name}"`);

  it("scans the form the app really dispatches in, so this test can fail", () => {
    const anyLive = sources(here + "../../").some((f) => dispatched(readFileSync(f, "utf8"), "submit_turn"));
    expect(anyLive).toBe(true);
  });

  it("are dispatched by no surface and carried by no catalog row", () => {
    const ids = COMMANDS.map((c) => c.id);
    for (const name of RETIRED) expect(ids).not.toContain(name);
    for (const file of sources(here + "../../")) {
      const src = readFileSync(file, "utf8");
      for (const name of RETIRED) {
        expect(dispatched(src, name), `${file} dispatches retired ${name}`).toBe(false);
        expect(src).not.toContain(`intent.custom("${name}"`);
      }
    }
  });
});

describe("the home diff panel", () => {
  it("dispatches no whole-diff intent of its own", () => {
    const home = read("../home/Home.tsx");
    expect(home).not.toContain("intent.acceptDiff");
    expect(home).not.toContain("intent.rejectDiff");
  });

  it("binds the hunk-addressed callback, so one hunk decided is one hunk decided", () => {
    expect(read("../home/ChatPanel.tsx")).toContain("onStatus={onDiffStatus}");
    expect(read("HunkReview.tsx")).not.toContain("legacyOnAct");
  });
});
