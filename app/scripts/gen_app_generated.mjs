#!/usr/bin/env node
/**
 * App wire-type emitter (convenience only; outputs are counted active source).
 *
 * Writes:
 *   app/src/wire_types.ts
 *
 * Does NOT write command_catalog.json — hide-sdk-codegen is the sole writer of
 * both protocol and FE command catalog mirrors.
 *
 * Run: node app/scripts/gen_app_generated.mjs
 */
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "../..");

const wire = spawnSync(process.execPath, [join(here, "gen_wire.mjs")], {
  cwd: root,
  encoding: "utf8",
});
if (wire.status !== 0) {
  console.error(wire.stdout);
  console.error(wire.stderr);
  process.exit(wire.status ?? 1);
}
process.stdout.write(wire.stdout || "");
