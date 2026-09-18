#!/usr/bin/env node
/**
 * Hawking's private Playwright mechanics sidecar.
 *
 * This process deliberately has no Goal, worker, receipt, or permission
 * authority. It owns one Chromium process and its Playwright contexts; the
 * Python Hawking owner supplies session identity, authorization, durable
 * WorldState and artifact paths over a local Unix socket.
 */
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import process from "node:process";
import { randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import { chromium } from "playwright";

const SCHEMA = "hawking.browser.helper.v1";
const MAX_EVENTS = 64;
const MAX_NODES = 128;
const MAX_TEXT = 12000;
const require = createRequire(import.meta.url);
const playwrightVersion = require("playwright/package.json").version;

const socketPath = process.argv[2] === "--serve" ? process.argv[3] : "";
if (!socketPath) {
  console.error("usage: hawking_browser_helper.mjs --serve <unix-socket>");
  process.exit(64);
}

let browser = null;
let serial = Promise.resolve();
let idleTimer = null;
const sessions = new Map();

function bounded(value, max) {
  const text = String(value ?? "");
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

function asInt(value, fallback, maximum) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(0, Math.min(maximum, Math.floor(number)));
}

function protocolError(code, message) {
  const error = new Error(message);
  error.code = code;
  return error;
}

function requireId(value, label) {
  const text = String(value ?? "").trim();
  if (!/^[A-Za-z0-9_-]{1,120}$/.test(text)) {
    throw protocolError("INVALID_ARGUMENTS", `${label} is required and must be a safe identifier`);
  }
  return text;
}

function requireUrl(value) {
  const text = String(value ?? "").trim();
  let parsed;
  try {
    parsed = new URL(text);
  } catch {
    throw protocolError("INVALID_ARGUMENTS", "url must be an absolute http(s) or file URL");
  }
  if (!["http:", "https:", "file:"].includes(parsed.protocol)) {
    throw protocolError("INVALID_ARGUMENTS", "url protocol must be http, https, or file");
  }
  return parsed.href;
}

async function ensureBrowser() {
  if (browser && browser.isConnected()) return browser;
  browser = await chromium.launch({
    headless: process.env.HAWKING_BROWSER_HEADLESS !== "0",
    chromiumSandbox: true,
  });
  return browser;
}

function scheduleIdleShutdown() {
  if (idleTimer) clearTimeout(idleTimer);
  if (sessions.size > 0) return;
  idleTimer = setTimeout(async () => {
    try {
      if (browser) await browser.close();
    } finally {
      browser = null;
    }
  }, 5 * 60 * 1000);
  idleTimer.unref();
}

function recordEvent(session, kind, payload) {
  session.events.push({
    kind,
    at: new Date().toISOString(),
    ...payload,
  });
  if (session.events.length > MAX_EVENTS) {
    session.events.splice(0, session.events.length - MAX_EVENTS);
  }
}

function pageIdFor(session, page) {
  for (const [id, candidate] of session.pages.entries()) {
    if (candidate === page) return id;
  }
  const id = `PAGE-${randomUUID().replaceAll("-", "").slice(0, 16).toUpperCase()}`;
  session.pages.set(id, page);
  page.on("console", (message) => recordEvent(session, "console", {
    page_id: id,
    level: message.type(),
    text: bounded(message.text(), 2000),
  }));
  page.on("pageerror", (error) => recordEvent(session, "console", {
    page_id: id,
    level: "error",
    text: bounded(error?.message, 2000),
  }));
  page.on("requestfailed", (request) => recordEvent(session, "network", {
    page_id: id,
    kind: "requestfailed",
    url: bounded(request.url(), 2000),
    method: request.method(),
    failure: bounded(request.failure()?.errorText, 800),
  }));
  page.on("response", (response) => {
    if (response.status() >= 400) {
      recordEvent(session, "network", {
        page_id: id,
        kind: "http_error",
        url: bounded(response.url(), 2000),
        status: response.status(),
      });
    }
  });
  page.on("download", (download) => recordEvent(session, "download", {
    page_id: id,
    suggested_filename: bounded(download.suggestedFilename(), 512),
    url: bounded(download.url(), 2000),
  }));
  page.on("close", () => {
    session.pages.delete(id);
    recordEvent(session, "page", { page_id: id, kind: "closed" });
  });
  return id;
}

function materializePages(session) {
  for (const page of session.context.pages()) pageIdFor(session, page);
}

async function describePage(session, page) {
  const pageId = pageIdFor(session, page);
  const title = await page.title().catch(() => "");
  return {
    page_id: pageId,
    url: page.url(),
    title: bounded(title, 2000),
    closed: page.isClosed(),
  };
}

function getSession(args) {
  const id = requireId(args.browser_session_id, "browser_session_id");
  const session = sessions.get(id);
  if (!session) throw protocolError("UNKNOWN_SESSION", `unknown browser session ${id}`);
  return session;
}

function getPage(session, args) {
  materializePages(session);
  const requested = String(args.page_id ?? "").trim();
  if (requested) {
    const page = session.pages.get(requested);
    if (!page || page.isClosed()) throw protocolError("UNKNOWN_PAGE", `unknown page ${requested}`);
    return page;
  }
  const page = [...session.pages.values()].find((candidate) => !candidate.isClosed());
  if (!page) throw protocolError("NO_PAGE", "browser session has no open page");
  return page;
}

async function openSession(args) {
  const sessionId = requireId(args.browser_session_id, "browser_session_id");
  let session = sessions.get(sessionId);
  if (!session) {
    const current = await ensureBrowser();
    const context = await current.newContext({ acceptDownloads: true });
    session = { id: sessionId, context, pages: new Map(), events: [] };
    sessions.set(sessionId, session);
    context.on("page", (page) => pageIdFor(session, page));
    recordEvent(session, "session", { kind: "opened" });
  }
  materializePages(session);
  let page = [...session.pages.values()].find((candidate) => !candidate.isClosed());
  if (!page) page = await session.context.newPage();
  if (args.url) {
    await page.goto(requireUrl(args.url), {
      waitUntil: args.wait_until || "domcontentloaded",
      timeout: asInt(args.timeout_ms, 30000, 120000),
    });
  }
  return {
    browser_session_id: sessionId,
    page: await describePage(session, page),
    pages: await Promise.all([...session.pages.values()].filter((item) => !item.isClosed()).map((item) => describePage(session, item))),
  };
}

async function nodeRows(page, maxNodes = MAX_NODES) {
  const selector = "button,a,input,textarea,select,[role],[contenteditable='true']";
  return page.locator(selector).evaluateAll((nodes, cap) => nodes.slice(0, cap).map((node, index) => {
    const el = /** @type {HTMLElement} */ (node);
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute("role") || (tag === "a" ? "link" : tag === "button" ? "button" : tag === "select" ? "combobox" : tag === "textarea" ? "textbox" : tag === "input" ? (el.getAttribute("type") === "checkbox" ? "checkbox" : "textbox") : "");
    const rawText = el.getAttribute("aria-label") || el.getAttribute("title") || el.textContent || el.getAttribute("value") || "";
    const name = rawText.replace(/\s+/g, " ").trim().slice(0, 500);
    const stable = el.getAttribute("data-testid") || el.id || el.getAttribute("name") || "";
    return {
      node_id: stable ? `${role || tag}:${stable}` : `${role || tag}:${index}:${name.slice(0, 80)}`,
      tag,
      role,
      name,
      label: el.getAttribute("aria-label") || "",
      test_id: el.getAttribute("data-testid") || "",
      enabled: !(/** @type {HTMLInputElement} */ (el)).disabled,
      visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length),
      value: "value" in el ? String((/** @type {HTMLInputElement} */ (el)).value || "").slice(0, 500) : "",
    };
  }), maxNodes);
}

async function observe(session, args) {
  const page = getPage(session, args);
  const nodeCap = Math.max(1, Math.min(MAX_NODES, asInt(args.max_nodes, 48, MAX_NODES)));
  const [description, nodes, visibleText] = await Promise.all([
    describePage(session, page),
    nodeRows(page, nodeCap).catch(() => []),
    page.locator("body").innerText({ timeout: 5000 }).catch(() => ""),
  ]);
  return {
    ...description,
    loading: false,
    focused_node: await page.evaluate(() => {
      const el = document.activeElement;
      if (!el || el === document.body) return null;
      return {
        tag: el.tagName.toLowerCase(),
        id: el.id || "",
        role: el.getAttribute("role") || "",
        label: el.getAttribute("aria-label") || "",
      };
    }).catch(() => null),
    visible_text: bounded(visibleText.replace(/\s+/g, " ").trim(), MAX_TEXT),
    nodes,
    recent_console: session.events.filter((item) => item.kind === "console").slice(-16),
    recent_network: session.events.filter((item) => item.kind === "network").slice(-16),
  };
}

function locatorFor(page, rawTarget) {
  if (!rawTarget || typeof rawTarget !== "object" || Array.isArray(rawTarget)) {
    throw protocolError("INVALID_ARGUMENTS", "target must be an object with one semantic selector");
  }
  const target = rawTarget;
  const exact = target.exact === true;
  if (target.role) {
    const role = String(target.role);
    const options = target.name ? { name: String(target.name), exact } : {};
    return { locator: page.getByRole(role, options), resolved_by: { role, ...(target.name ? { name: String(target.name), exact } : {}) } };
  }
  if (target.label) {
    return { locator: page.getByLabel(String(target.label), { exact }), resolved_by: { label: String(target.label), exact } };
  }
  if (target.test_id) {
    return { locator: page.getByTestId(String(target.test_id)), resolved_by: { test_id: String(target.test_id) } };
  }
  if (target.text) {
    return { locator: page.getByText(String(target.text), { exact }), resolved_by: { text: String(target.text), exact } };
  }
  if (target.css) {
    return { locator: page.locator(String(target.css)), resolved_by: { css: String(target.css) } };
  }
  throw protocolError("INVALID_ARGUMENTS", "target needs role, label, test_id, text, or css");
}

async function resolveUnique(page, target) {
  const selected = locatorFor(page, target);
  const count = await selected.locator.count();
  if (count !== 1) {
    throw protocolError(count === 0 ? "TARGET_NOT_FOUND" : "TARGET_AMBIGUOUS", `${count} nodes match the supplied semantic target`);
  }
  return selected;
}

async function find(session, args) {
  const page = getPage(session, args);
  const selected = locatorFor(page, args.target);
  const count = await selected.locator.count();
  const max = Math.max(1, Math.min(32, asInt(args.max_results, 8, 32)));
  const matches = await selected.locator.evaluateAll((nodes, cap) => nodes.slice(0, cap).map((node) => {
    const el = /** @type {HTMLElement} */ (node);
    return {
      tag: el.tagName.toLowerCase(),
      text: (el.getAttribute("aria-label") || el.textContent || "").replace(/\s+/g, " ").trim().slice(0, 500),
      enabled: !(/** @type {HTMLInputElement} */ (el)).disabled,
      visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length),
    };
  }), max).catch(() => []);
  return { page_id: pageIdFor(session, page), resolved_by: selected.resolved_by, count, matches, truncated: count > matches.length };
}

async function action(session, args, kind) {
  const page = getPage(session, args);
  const timeout = asInt(args.timeout_ms, 15000, 120000);
  let resolvedBy = null;
  if (kind === "scroll") {
    await page.mouse.wheel(Number(args.delta_x || 0), Number(args.delta_y || 720));
  } else if (kind === "key") {
    if (args.target) {
      const selected = await resolveUnique(page, args.target);
      await selected.locator.focus({ timeout });
      resolvedBy = selected.resolved_by;
    }
    await page.keyboard.press(String(args.key ?? ""));
  } else {
    const selected = await resolveUnique(page, args.target);
    resolvedBy = selected.resolved_by;
    if (kind === "click") await selected.locator.click({ timeout });
    if (kind === "type") await selected.locator.fill(String(args.text ?? ""), { timeout });
    if (kind === "select") await selected.locator.selectOption(args.option, { timeout });
  }
  const result = {
    action: kind,
    page_id: pageIdFor(session, page),
    resolved_by: resolvedBy,
    verified: true,
  };
  recordEvent(session, "action", result);
  return result;
}

async function waitFor(session, args) {
  const page = getPage(session, args);
  const timeout = asInt(args.timeout_ms, 15000, 120000);
  if (args.target) {
    const selected = await resolveUnique(page, args.target);
    await selected.locator.waitFor({ state: args.state || "visible", timeout });
    return { page_id: pageIdFor(session, page), waited_for: "target", resolved_by: selected.resolved_by };
  }
  if (args.load_state) {
    await page.waitForLoadState(args.load_state, { timeout });
    return { page_id: pageIdFor(session, page), waited_for: `load:${args.load_state}` };
  }
  const delay = asInt(args.delay_ms, 100, 10000);
  await page.waitForTimeout(delay);
  return { page_id: pageIdFor(session, page), waited_for: `delay:${delay}` };
}

async function screenshot(session, args) {
  const page = getPage(session, args);
  const outputPath = path.resolve(String(args.output_path || ""));
  if (!outputPath) throw protocolError("INVALID_ARGUMENTS", "output_path is required");
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  await page.screenshot({ path: outputPath, fullPage: args.full_page === true });
  const bytes = fs.statSync(outputPath).size;
  return { page_id: pageIdFor(session, page), artifact: { kind: "browser_screenshot", path: outputPath, bytes } };
}

async function verify(session, args) {
  const page = getPage(session, args);
  const checks = [];
  if (args.url_includes) checks.push({ check: "url_includes", expected: String(args.url_includes), actual: page.url(), passed: page.url().includes(String(args.url_includes)) });
  if (args.text) {
    const text = await page.locator("body").innerText({ timeout: 5000 }).catch(() => "");
    checks.push({ check: "text", expected: String(args.text), passed: text.includes(String(args.text)) });
  }
  if (args.target) {
    try {
      const selected = await resolveUnique(page, args.target);
      const visible = await selected.locator.isVisible();
      const enabled = await selected.locator.isEnabled();
      checks.push({ check: "target", resolved_by: selected.resolved_by, visible, enabled, passed: args.enabled === undefined ? visible : visible && enabled === Boolean(args.enabled) });
    } catch (error) {
      checks.push({ check: "target", passed: false, error: bounded(error?.message, 1000) });
    }
  }
  if (checks.length === 0) throw protocolError("INVALID_ARGUMENTS", "verify requires url_includes, text, or target");
  return { page_id: pageIdFor(session, page), verified: checks.every((item) => item.passed), checks };
}

async function handle(args) {
  const op = String(args.op || "");
  if (op === "health") {
    // A routine Hawking WorldState commit asks for helper facts.  It must not
    // revive Chromium merely because the final session was just closed.  The
    // explicit browser.health tool supplies probe=true when it needs to prove
    // the executable browser path end-to-end.
    const current = args.probe === true ? await ensureBrowser() : browser;
    scheduleIdleShutdown();
    return {
      schema: SCHEMA,
      health: "READY",
      playwright_version: playwrightVersion,
      browser_version: current?.isConnected() ? current.version() : null,
      browser_connected: Boolean(current?.isConnected()),
      session_count: sessions.size,
    };
  }
  if (op === "session.open") return openSession(args);
  if (op === "session.close") {
    const session = getSession(args);
    await session.context.close();
    sessions.delete(session.id);
    scheduleIdleShutdown();
    return { browser_session_id: session.id, closed: true };
  }
  if (op === "pages") {
    const session = getSession(args);
    materializePages(session);
    return { browser_session_id: session.id, pages: await Promise.all([...session.pages.values()].filter((page) => !page.isClosed()).map((page) => describePage(session, page))) };
  }
  if (op === "goto") {
    const session = getSession(args); const page = getPage(session, args);
    await page.goto(requireUrl(args.url), { waitUntil: args.wait_until || "domcontentloaded", timeout: asInt(args.timeout_ms, 30000, 120000) });
    const result = { action: "goto", page: await describePage(session, page), verified: true };
    recordEvent(session, "action", result); return result;
  }
  if (op === "observe") return observe(getSession(args), args);
  if (op === "find") return find(getSession(args), args);
  if (["click", "type", "select", "key", "scroll"].includes(op)) return action(getSession(args), args, op);
  if (op === "wait") return waitFor(getSession(args), args);
  if (op === "screenshot") return screenshot(getSession(args), args);
  if (op === "console" || op === "network" || op === "downloads") {
    const session = getSession(args); const kind = op === "downloads" ? "download" : op;
    return { browser_session_id: session.id, events: session.events.filter((item) => item.kind === kind).slice(-asInt(args.limit, 32, 64)) };
  }
  if (op === "verify") return verify(getSession(args), args);
  if (op === "shutdown") {
    for (const session of sessions.values()) await session.context.close().catch(() => {});
    sessions.clear();
    if (browser) await browser.close().catch(() => {});
    browser = null;
    setTimeout(() => process.exit(0), 10).unref();
    return { stopped: true };
  }
  throw protocolError("UNKNOWN_OPERATION", `unsupported browser helper operation: ${op}`);
}

function reply(socket, payload) {
  socket.end(`${JSON.stringify(payload)}\n`);
}

function serve(socket) {
  let buffer = "";
  socket.setEncoding("utf8");
  socket.on("data", (chunk) => {
    buffer += chunk;
    const newline = buffer.indexOf("\n");
    if (newline < 0) return;
    const line = buffer.slice(0, newline);
    socket.pause();
    serial = serial.then(async () => {
      let request;
      try {
        request = JSON.parse(line);
        const result = await handle(request);
        reply(socket, { schema: SCHEMA, id: request.id ?? "", ok: true, result });
      } catch (error) {
        reply(socket, {
          schema: SCHEMA,
          id: request?.id ?? "",
          ok: false,
          error: { code: error?.code || "HELPER_ERROR", message: bounded(error?.message || error, 2000) },
        });
      }
    }).catch(() => {}).finally(() => socket.resume());
  });
}

if (fs.existsSync(socketPath)) fs.unlinkSync(socketPath);
fs.mkdirSync(path.dirname(socketPath), { recursive: true });
const server = net.createServer(serve);
server.listen(socketPath, () => fs.chmodSync(socketPath, 0o600));
server.on("error", (error) => { console.error(error.stack || error.message); process.exit(1); });
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, async () => {
  try { if (browser) await browser.close(); } finally { server.close(() => process.exit(0)); }
});
