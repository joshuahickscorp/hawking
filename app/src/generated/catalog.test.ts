/*
  The drift guard. command_catalog.json here is a COPY of the Rust authority's golden
  (crates/hide-sdk/goldens/command_catalog.json, generated from hide-protocol command_catalog()).
  If the two ever differ by a single byte, the frontend suite fails, so the FE can never resolve a
  command table the backend has moved on from. Fix by re-copying the golden, never by hand editing.
*/
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, it, expect } from "vitest";

const here = fileURLToPath(new URL(".", import.meta.url));
const read = (p: string) => readFileSync(here + p, "utf8");

describe("generated command catalog", () => {
  it("is byte-identical to the Rust golden", () => {
    expect(read("command_catalog.json")).toBe(read("../../../crates/hide-sdk/goldens/command_catalog.json"));
  });

  it("parses to commands with unique ids", () => {
    const catalog = JSON.parse(read("command_catalog.json")) as { id: string }[];
    expect(catalog.length).toBeGreaterThan(0);
    expect(new Set(catalog.map((c) => c.id)).size).toBe(catalog.length);
  });
});
