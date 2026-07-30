#!/usr/bin/env node
/**
 * Generate app/src/wire_types.ts from hide-core wire types.
 *
 * Authority: crates/hide-core/src/api.rs (+ types.rs, runtime.rs).
 * Handwritten FE logic (ackState, intent builders, CUSTOM_NAMES mirror) stays in
 * app/src/wire.ts and re-exports these types.
 *
 * Run: node app/scripts/gen_app_generated.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "../..");

const api = readFileSync(join(root, "crates/hide-core/src/api.rs"), "utf8");
const types = readFileSync(join(root, "crates/hide-core/src/types.rs"), "utf8");
const runtime = readFileSync(join(root, "crates/hide-core/src/runtime.rs"), "utf8");

/** @param {string} src @param {string} name */
function extractBlock(src, kind, name) {
  const re = new RegExp(`pub ${kind} ${name}\\s*\\{([\\s\\S]*?)\\n\\}`, "m");
  const m = src.match(re);
  if (!m) throw new Error(`${kind} ${name} not found`);
  return m[1];
}

/** @param {string} ty */
function rustTypeToTs(ty) {
  const t = ty.replace(/\s+/g, " ").trim();
  if (t === "String") return "string";
  if (t === "bool") return "boolean";
  if (
    t === "u64" ||
    t === "u32" ||
    t === "i64" ||
    t === "i32" ||
    t === "usize" ||
    t === "isize"
  )
    return "number";
  if (t === "Value" || t === "serde_json::Value") return "unknown";
  if (t === "SessionId" || t === "RunId" || t === "EventId" || t === "BlobId") return t;
  if (t === "BlobRef") return "BlobRef";
  if (t === "UiEventKind") return "UiEventKind";
  const opt = t.match(/^Option<\s*(.+)\s*>$/);
  if (opt) return `${rustTypeToTs(opt[1])} | null`;
  const vec = t.match(/^Vec<\s*(.+)\s*>$/);
  if (vec) return `${rustTypeToTs(vec[1])}[]`;
  return "unknown";
}

/** @param {string} name */
function toSnake(name) {
  return name.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
}

/**
 * Parse a rust enum body whose variants are either `Name { fields }` or `Name(Type)`.
 * @param {string} body
 * @param {{ customUntagged?: boolean }} [opts]
 */
function enumToTaggedUnion(body, opts = {}) {
  const parts = body.split(/\n\s{4}([A-Z][A-Za-z0-9_]*)\s*/);
  const arms = [];
  for (let i = 1; i < parts.length; i += 2) {
    const name = parts[i];
    const rest = parts[i + 1] ?? "";
    const snake = toSnake(name);

    // newtype / untagged: Custom(Value)
    const tuple = rest.match(/^\(([^)]*)\)/);
    if (tuple) {
      const inner = tuple[1].trim();
      if (opts.customUntagged && name === "Custom") {
        arms.push(`  | { type: "${snake}"; data: ${rustTypeToTs(inner)} }`);
      } else if (!inner) {
        arms.push(`  | { type: "${snake}" }`);
      } else {
        arms.push(`  | { type: "${snake}"; data: ${rustTypeToTs(inner)} }`);
      }
      continue;
    }

    const brace = rest.match(/^\{([\s\S]*?)\n\s{4}\}/);
    if (!brace) {
      arms.push(`  | { type: "${snake}" }`);
      continue;
    }
    const fieldsBody = brace[1];
    /** @type {string[]} */
    const fields = [];
    let pendingOptional = false;
    for (const line of fieldsBody.split("\n")) {
      if (line.includes("#[serde(default")) pendingOptional = true;
      const fl = line.match(/^\s*(?:pub\s+)?(\w+)\s*:\s*([^,]+),?\s*$/);
      if (!fl) continue;
      const fname = fl[1];
      if (fname.startsWith("/")) continue;
      const fty = fl[2].trim();
      const ts = rustTypeToTs(fty);
      // serde(default) Option fields are optional on the wire
      if (pendingOptional && fty.startsWith("Option<")) {
        const base = ts.replace(/ \| null$/, "");
        fields.push(`${fname}?: ${base} | null`);
      } else {
        fields.push(`${fname}: ${ts}`);
      }
      pendingOptional = false;
    }
    if (fields.length === 0) {
      arms.push(`  | { type: "${snake}" }`);
    } else {
      arms.push(`  | { type: "${snake}"; data: { ${fields.join("; ") } } }`);
    }
  }
  return arms.join("\n");
}

/**
 * @param {string} body
 * @param {{ optionalFields?: Set<string> }} [opts]
 */
function structToInterface(body, opts = {}) {
  const optional = opts.optionalFields ?? new Set();
  /** @type {string[]} */
  const fields = [];
  let pendingOptional = false;
  for (const line of body.split("\n")) {
    if (line.includes("#[serde(default")) pendingOptional = true;
    const fl = line.match(/^\s*pub\s+(\w+)\s*:\s*([^,]+),?\s*$/);
    if (!fl) continue;
    const fname = fl[1];
    const fty = fl[2].trim();
    if (pendingOptional || optional.has(fname)) {
      if (fty.startsWith("Option<") || fty === "bool") {
        const ts = rustTypeToTs(fty).replace(/ \| null$/, "");
        fields.push(`  ${fname}?: ${fty === "bool" ? "boolean" : `${ts} | null`};`);
      } else {
        fields.push(`  ${fname}: ${rustTypeToTs(fty)};`);
      }
      pendingOptional = false;
      continue;
    }
    fields.push(`  ${fname}: ${rustTypeToTs(fty)};`);
    pendingOptional = false;
  }
  return fields.join("\n");
}

const blobBody = extractBlock(types, "struct", "BlobRef");
const intentBody = extractBlock(api, "enum", "Intent");
const ackBody = extractBlock(api, "struct", "IntentAck");
const kindBody = extractBlock(api, "enum", "UiEventKind");
const eventBody = extractBlock(api, "struct", "UiEvent");

const rtMatch = runtime.match(/pub enum RuntimeSupervisorState\s*\{([\s\S]*?)\n\}/);
if (!rtMatch) throw new Error("RuntimeSupervisorState not found");
const runtimeStates = [...rtMatch[1].matchAll(/([A-Z][A-Za-z0-9_]*)/g)].map((x) =>
  toSnake(x[1]),
);

const out = `// Generated by app/scripts/gen_app_generated.mjs (via gen_wire). DO NOT EDIT BY HAND.
// Authority: crates/hide-core/src/api.rs (serde tag="type", content="data", rename_all="snake_case").
// Regenerate: node app/scripts/gen_app_generated.mjs
// CUSTOM_NAMES / PROJECTION_NAMES / ack helpers stay in app/src/wire.ts (hide-protocol mirror).

export type SessionId = string;
export type RunId = string;
export type EventId = string;
export type BlobId = string;

export interface BlobRef {
${structToInterface(blobBody)}
}

export type Intent =
${enumToTaggedUnion(intentBody)};

export interface IntentAck {
${structToInterface(ackBody, { optionalFields: new Set(["held"]) })}
}

export type UiEventKind =
${enumToTaggedUnion(kindBody, { customUntagged: true })};

export interface UiEvent {
${structToInterface(eventBody)}
}

export type RuntimeState = ${runtimeStates.map((s) => `"${s}"`).join(" | ")};
`;

const dest = join(here, "../src/wire_types.ts");
writeFileSync(dest, out);
console.log(`wrote ${dest} (${out.split("\n").length} lines)`);
