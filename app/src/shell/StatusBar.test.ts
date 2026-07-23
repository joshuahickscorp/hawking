/*
  StatusBar.test.ts: the status bar as an honest read of host state.

  Two things a user can be lied to about here, and both were being lied about before this stage: the
  problem counts (a hardcoded 0 / 0) and the branch (a hardcoded "main" on a button that did nothing).
  So this asserts the counter reflects a REAL `diagnostics` projection patch (including a clean patch,
  which must read as a real zero and not as "not run"), that the branch is bound to the host `home`
  projection, that the dead Branch button chrome is gone from the source, and that status is carried in
  words rather than by colour alone.

  No jsdom in this project, so component assertions render through react-dom/server.
*/
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../ipc", () => ({
  sendIntent: async () => ({ accepted: true, event_seq: 1, message: null }),
  subscribeUi: () => () => {},
  callConnector: async () => null,
  TRANSPORT_KIND: "mock",
}));

import { useStore } from "../store";
import {
  analysisPaths,
  branchLabel,
  DiagnosticsDetail,
  jobsLabel,
  problemTone,
  problemsLabel,
  readDiagnostics,
  readWriteLease,
  leaseLabel,
  runningJobs,
  StatusBar,
  WriteLeaseDetail,
} from "./StatusBar";

const SRC = readFileSync(join(__dirname, "StatusBar.tsx"), "utf8");

// zustand v5 answers useSyncExternalStore's SERVER snapshot with getInitialState, so a
// react-dom/server render always sees the initial store. Store-fed assertions therefore go through
// the exported readers and the prop-driven detail panel; the binding itself is asserted on the source.
const detail = (d: unknown) => renderToStaticMarkup(createElement(DiagnosticsDetail, { d: readDiagnostics(d) }));

const DIRTY = {
  errors: 2,
  warnings: 3,
  by_file: [
    { file: "src/net.rs", errors: 2, warnings: 1 },
    { file: "src/lib.rs", errors: 0, warnings: 2 },
  ],
  last_verification_id: "ver_abc123",
};
const CLEAN = { errors: 0, warnings: 0, by_file: [], last_verification_id: "ver_clean" };

beforeEach(() => {
  useStore.setState({ projections: {}, home: null, fleet: [], notices: [], manifest: null });
});

describe("the bar keeps its shape", () => {
  it("still renders branch, problems, phase, model, transport and runtime on one line", () => {
    const html = renderToStaticMarkup(createElement(StatusBar));
    expect(html).toContain("vsc-statusbar");
    expect(html).toContain("Workspace branch");
    expect(html).toContain("phase: idle");
    expect(html).toContain("mock transport");
    expect(html).toContain("Runtime down");
  });
});

describe("diagnostics projection", () => {
  it("reads the host shape", () => {
    const d = readDiagnostics(DIRTY);
    expect(d?.errors).toBe(2);
    expect(d?.warnings).toBe(3);
    expect(d?.by_file.map((f) => f.file)).toEqual(["src/net.rs", "src/lib.rs"]);
    expect(d?.last_verification_id).toBe("ver_abc123");
  });

  it("a clean patch is a real zero, not 'not run'", () => {
    const d = readDiagnostics(CLEAN);
    expect(d).not.toBeNull();
    expect(d?.errors).toBe(0);
    expect(problemTone(d)).toBe("clean");
  });

  it("no patch at all is unknown, never a fabricated zero", () => {
    expect(readDiagnostics(undefined)).toBeNull();
    expect(readDiagnostics({})).toBeNull();
    expect(problemTone(null)).toBe("unknown");
    expect(problemsLabel(null)).toMatch(/no static analysis has run/i);
  });

  it("tone separates errors from warnings from clean", () => {
    expect(problemTone(readDiagnostics(DIRTY))).toBe("bad");
    expect(problemTone(readDiagnostics({ errors: 0, warnings: 4, by_file: [] }))).toBe("warn");
  });

  it("the accessible name states counts and verification in words, not colour", () => {
    const label = problemsLabel(readDiagnostics(DIRTY));
    expect(label).toContain("2 errors");
    expect(label).toContain("3 warnings");
    expect(label).toContain("ver_abc123");
    expect(problemsLabel(readDiagnostics({ errors: 1, warnings: 1, by_file: [] }))).toContain("1 error,");
  });
});

describe("the detail panel renders the real patch", () => {
  it("shows the counts, the per-file breakdown and the sealing verification id", () => {
    const html = detail(DIRTY);
    expect(html).toContain("2 errors");
    expect(html).toContain("3 warnings");
    expect(html).toContain("ver_abc123");
    expect(html).toContain("src/net.rs");
    expect(html).toContain("src/lib.rs");
  });

  it("a clean patch is a real zero with an empty breakdown", () => {
    const html = detail(CLEAN);
    expect(html).toContain("0 errors, 0 warnings");
    expect(html).toContain("No file has a finding");
    expect(html).not.toMatch(/no static analysis has run/i);
  });

  it("no patch says nothing has run instead of showing 0 / 0", () => {
    const html = detail(undefined);
    expect(html).toMatch(/No static analysis has run/i);
    expect(html).not.toContain("0 errors");
  });
});

describe("background work", () => {
  it("counts only work that is still moving", () => {
    const runs = runningJobs([
      { id: "run_a", objective: "refactor pool guard", state: "active", step: 3, steps: 6 },
      { id: "run_b", objective: "add retry tests", state: "waiting", step: 2, steps: 4 },
      { id: "run_c", objective: "port the tokenizer", state: "done", step: 4, steps: 4 },
    ]);
    expect(runs.map((r) => r.id)).toEqual(["run_a", "run_b"]);
    expect(jobsLabel(runs)).toContain("refactor pool guard");
    expect(jobsLabel(runs)).toContain("2 background runs");
    expect(jobsLabel([])).toBe("No background work");
  });

  it("an idle bar carries no chip", () => {
    expect(useStore.getState().fleet).toHaveLength(0);
    expect(renderToStaticMarkup(createElement(StatusBar))).not.toContain("running");
  });
});

describe("bindings and retirements", () => {
  it("the counter is bound to the host diagnostics projection", () => {
    expect(SRC).toContain("useStore((s) => s.projections.diagnostics)");
    expect(SRC).toContain("readDiagnostics(diagnosticsPatch)");
  });

  it("the branch is bound to the host home projection, never hardcoded", () => {
    expect(SRC).toContain("useStore((s) => s.home?.workspace?.branch)");
    expect(branchLabel("build/hide-impl-2026-07-19")).toBe("build/hide-impl-2026-07-19");
    expect(branchLabel(undefined)).toBe("no branch");
    expect(branchLabel("   ")).toBe("no branch");
  });

  it("the dead Branch button chrome is gone", () => {
    expect(SRC).not.toMatch(/<button[^>]*title="Branch"/);
    expect(SRC).not.toMatch(/<span>main<\/span>/);
    expect(renderToStaticMarkup(createElement(StatusBar))).not.toContain(">main<");
  });

  it("no hardcoded problem counts survive", () => {
    expect(SRC).not.toMatch(/<span>0<\/span>/);
    // With the initial (empty) store the bar shows the honest unknown state, not a fabricated zero.
    const html = renderToStaticMarkup(createElement(StatusBar));
    expect(html).toContain("not run");
    expect(html).not.toContain("<span>0</span>");
  });
});

/*
  The counter is also the PRODUCER now: run_static_analysis binds Custom, so this is the one place in
  the app that can write the diagnostics projection the bar reads. It must never send a payload the
  host would refuse, so the file list is derived and the control is disabled when it is empty.
*/
describe("the Problems counter can fill itself", () => {
  it("derives the file list from the diff, the last receipt, and the open tabs, deduped", () => {
    expect(analysisPaths({ path: "src/net.rs" }, readDiagnostics(DIRTY), ["src/net.rs", "src/main.rs"])).toEqual([
      "src/net.rs",
      "src/lib.rs",
      "src/main.rs",
    ]);
    expect(analysisPaths(undefined, null, [])).toEqual([]);
    expect(analysisPaths(undefined, null, ["  ", ""])).toEqual([]);
  });

  it("offers the analysis run, and disables it with a reason when there is no file to analyse", () => {
    const withPaths = renderToStaticMarkup(
      createElement(DiagnosticsDetail, { d: readDiagnostics(DIRTY), paths: ["src/net.rs"] }),
    );
    expect(withPaths).toContain("Run static analysis (1)");
    expect(withPaths).not.toContain("disabled");

    const withNone = renderToStaticMarkup(createElement(DiagnosticsDetail, { d: null, paths: [] }));
    expect(withNone).toContain("disabled");
    expect(withNone).toContain("no file is open and no change has been proposed");
  });

  // The write lease. The bar may only claim a lease the HOST says is in force: an unleased session
  // and a revoked one look identical here, which is what stops a revoked lease reading as active.
  it("shows a write lease only while the host reports one active", () => {
    expect(readWriteLease(undefined)).toBeNull();
    expect(readWriteLease({})).toBeNull();
    expect(readWriteLease({ write_lease: { active: false, note: "revoked by the user" } })).toBeNull();

    const l = readWriteLease({
      write_lease: { active: true, note: "granted", lease_id: "gnt_1", repo_id: "hide", scopes: ["/w/hide"] },
    });
    expect(l).toEqual({ lease_id: "gnt_1", repo_id: "hide", scopes: ["/w/hide"], note: "granted" });
    expect(leaseLabel(l!)).toContain("Write lease active on hide");
    expect(leaseLabel(l!)).toContain("1 declared scope");
    expect(leaseLabel(l!)).toContain("outside it still ask");
  });

  it("puts the scopes and the one de-escalating gesture in the popover, not on the bar", () => {
    const l = readWriteLease({
      write_lease: { active: true, note: "granted", lease_id: "gnt_1", repo_id: "hide", scopes: ["/w/hide/app"] },
    })!;
    const markup = renderToStaticMarkup(createElement(WriteLeaseDetail, { lease: l }));
    expect(markup).toContain("/w/hide/app");
    expect(markup).toContain("Revoke write lease");
    expect(markup).toContain("still asks");
    // The bar itself carries one short line and opens the rest in place: no new permanent control.
    expect(SRC).toContain('aria-controls="statusbar-write-lease"');
    expect(SRC).toContain("{lease ? (");
  });
});
