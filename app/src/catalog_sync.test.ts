import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, it, expect } from "vitest";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "../..");
const read = (p: string) => readFileSync(join(root, p), "utf8");

describe("command catalog FE mirror", () => {
  it("is byte-identical to the hide-protocol golden", () => {
    expect(read("app/src/generated/command_catalog.json")).toBe(
      read("crates/hide-protocol/generated/command_catalog.json"),
    );
  });

  it("parses to commands with unique ids", () => {
    const catalog = JSON.parse(read("app/src/generated/command_catalog.json")) as {
      id: string;
    }[];
    expect(catalog.length).toBeGreaterThan(0);
    expect(new Set(catalog.map((c) => c.id)).size).toBe(catalog.length);
  });
});
