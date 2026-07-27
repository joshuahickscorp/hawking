/*
  StatusBar.tsx - the 22px status bar. Left: the real workspace branch, the real diagnostics counter,
  the newest notice, and a background-work chip that only exists while work is running. Right: phase,
  model, transport, runtime.

  Two consolidation decisions land here (docs/hide-impl/consolidation/HIDE_CONSOLIDATION_DECISIONS.md
  3.1 and 3.2):

  1. The Branch item was dead button chrome: a <button> with no onClick whose label was the string
     "main". No switch-branch capability exists (decision section 5 RETIRES `switch_branch`), so the
     button is retired to a plain label bound to the REAL branch the host reports in the `home`
     projection (hide-backend digest.rs git_branch reads .git/HEAD).
  2. The Problems counter was a hardcoded "0 / 0" mock. It now binds to the host `diagnostics`
     projection (hide-backend StaticAnalysisReceipt::diagnostics_projection:
     { errors, warnings, by_file: [{ file, errors, warnings }], last_verification_id }).

  Density is preserved by progressive disclosure: the bar still shows one short line, and the per-file
  breakdown plus the sealing verification id live in a popover on the counter itself, so no new
  permanent control appears. Status is never colour-only: every item carries words in its accessible
  name (counts, verification state, runtime state).

  Deliberately NOT surfaced here, because the app has no honest source for them yet (reported with the
  stage, not faked):
    pending approvals   -> a SecurityGate already raises a blocking overlay in App.tsx; a second
                           indicator would duplicate an existing control.
    repository trust    -> the CONTROL is real and bound (Custom, host-handled) in the add-folder
                           flow that makes the decision, HomeComposer. What is missing is a READ: no
                           projection carries a trust value, so a status item would have nothing to
                           show. Bound where the decision is made, not mirrored here.
    environment         -> same shape: `environment_switch` is bound in Settings (Workspace section),
                           and its UiEvent is not folded into a typed slice, so there is no live
                           value for this bar to display.
    checkpoint count    -> checkpoint records arrive as Custom UiEvents and store.ts holds no
                           checkpoint slice (see StateTimeline.tsx).
*/
import { useState } from "react";
import { TRANSPORT_KIND } from "../ipc";
import { noticeFailure, runCommand, useStore } from "../store";
import type { FleetRun } from "../store";
import { Icon } from "./icons";
import { modelId } from "./ModelChooser";

export interface FileDiagnostics {
  file: string;
  errors: number;
  warnings: number;
}

/** The host `diagnostics` projection, as the FE reads it. */
export interface Diagnostics {
  errors: number;
  warnings: number;
  by_file: FileDiagnostics[];
  last_verification_id: string | null;
}

const num = (v: unknown): number => (typeof v === "number" && Number.isFinite(v) ? v : 0);

/**
 * Read the `diagnostics` projection patch. Returns null ONLY when no patch has arrived, so a CLEAN
 * analysis (0 errors, 0 warnings, empty by_file) reports as a real verified zero and is never
 * confused with "nothing has run".
 */
export function readDiagnostics(patch: unknown): Diagnostics | null {
  if (!patch || typeof patch !== "object") return null;
  const p = patch as Record<string, unknown>;
  if (p.errors === undefined && p.warnings === undefined) return null;
  const rows = Array.isArray(p.by_file) ? (p.by_file as Record<string, unknown>[]) : [];
  return {
    errors: num(p.errors),
    warnings: num(p.warnings),
    by_file: rows.map((f) => ({ file: String(f.file ?? ""), errors: num(f.errors), warnings: num(f.warnings) })),
    last_verification_id: typeof p.last_verification_id === "string" ? p.last_verification_id : null,
  };
}

/** unknown = never analysed, clean = analysed with nothing to report. Shape, not colour. */
export type ProblemTone = "unknown" | "clean" | "warn" | "bad";

export function problemTone(d: Diagnostics | null): ProblemTone {
  if (!d) return "unknown";
  if (d.errors > 0) return "bad";
  if (d.warnings > 0) return "warn";
  return "clean";
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/** The accessible name: counts AND verification state in words, so nothing is carried by colour. */
export function problemsLabel(d: Diagnostics | null): string {
  if (!d) return "Problems, no static analysis has run in this session yet";
  const verified = d.last_verification_id
    ? `sealed by verification ${d.last_verification_id}`
    : "no verification receipt";
  return `Problems, ${plural(d.errors, "error")}, ${plural(d.warnings, "warning")}, ${verified}`;
}

/** The branch the host reports. Never the hardcoded "main" the retired button rendered. */
export function branchLabel(branch: string | undefined): string {
  const b = (branch ?? "").trim();
  return b || "no branch";
}

/** Work that is still moving. The chip only exists while this is non-empty, so an idle bar is unchanged. */
export function runningJobs(fleet: FleetRun[]): FleetRun[] {
  return fleet.filter((r) => r.state === "active" || r.state === "waiting");
}

/** The host `status` projection's `write_lease` half (hide-backend host.rs publish_write_lease). */
export interface WriteLease {
  lease_id: string | null;
  repo_id: string | null;
  scopes: string[];
  note: string;
}

/**
 * Read the active write lease, or null when none is in force. Only an `active: true` patch counts:
 * a revocation publishes `active: false` so the bar CLEARS rather than keeping a lease on screen
 * that the host no longer honours.
 */
export function readWriteLease(patch: unknown): WriteLease | null {
  const l = (patch as { write_lease?: Record<string, unknown> } | undefined | null)?.write_lease;
  if (!l || l.active !== true) return null;
  return {
    lease_id: typeof l.lease_id === "string" ? l.lease_id : null,
    repo_id: typeof l.repo_id === "string" ? l.repo_id : null,
    scopes: Array.isArray(l.scopes) ? l.scopes.filter((s): s is string => typeof s === "string") : [],
    note: typeof l.note === "string" ? l.note : "granted",
  };
}

/** Never colour-only: the scope count and the repo are in the accessible name. */
export function leaseLabel(l: WriteLease): string {
  return `Write lease active on ${l.repo_id ?? "this workspace"}, ${plural(l.scopes.length, "declared scope")}. Writes outside it still ask`;
}

/**
 * The files this app can honestly hand the analyzer, in the order it prefers them: the file the live
 * diff touches, every file the last receipt already reported on, then the open editor tabs. Deduped,
 * and EMPTY when the app knows of no file, in which case the control says so instead of sending a
 * payload the host would refuse.
 */
export function analysisPaths(diffPatch: unknown, d: Diagnostics | null, openPaths: string[]): string[] {
  const diffPath = (diffPatch as { path?: unknown } | undefined | null)?.path;
  const all = [
    ...(typeof diffPath === "string" ? [diffPath] : []),
    ...(d?.by_file ?? []).map((f) => f.file),
    ...openPaths,
  ];
  return [...new Set(all.filter((p) => p && p.trim()))];
}

export function jobsLabel(runs: FleetRun[]): string {
  if (runs.length === 0) return "No background work";
  const detail = runs.map((r) => `${r.objective} ${r.state} step ${r.step} of ${r.steps}`).join("; ");
  return `${plural(runs.length, "background run")}: ${detail}`;
}

export function StatusBar({ openPaths = [] }: { openPaths?: string[] } = {}) {
  const runtimeStatus = useStore((s) => s.runtimeStatus);
  const runtimeDetail = useStore((s) => s.runtimeDetail);
  const runPhase = useStore((s) => s.runPhase);
  const manifest = useStore((s) => s.manifest);
  const notices = useStore((s) => s.notices);
  const diagnosticsPatch = useStore((s) => s.projections.diagnostics);
  const diffPatch = useStore((s) => s.projections.diff);
  const statusPatch = useStore((s) => s.projections.status);
  const branch = useStore((s) => s.home?.workspace?.branch);
  const fleet = useStore((s) => s.fleet);
  const [detail, setDetail] = useState(false);
  const [leaseOpen, setLeaseOpen] = useState(false);

  const model = modelId(manifest);
  const latest = notices[notices.length - 1];
  const diagnostics = readDiagnostics(diagnosticsPatch);
  const tone = problemTone(diagnostics);
  const jobs = runningJobs(fleet);
  const lease = readWriteLease(statusPatch);
  const dotClass =
    runtimeStatus === "ready"
      ? "status-dot status-dot--ok"
      : runtimeStatus === "failed" || runtimeStatus === "down"
        ? "status-dot status-dot--bad"
        : "status-dot status-dot--light";

  return (
    <footer className="vsc-statusbar">
      {/* Retired: the dead Branch BUTTON. A label, because no branch-switch capability exists. */}
      <span className="vsc-statusbar__item" aria-label={`Workspace branch, ${branchLabel(branch)}`}>
        <Icon name="source-control" size={13} strokeWidth={1.8} />
        <span>{branchLabel(branch)}</span>
      </span>

      {/* The real diagnostics counter. The counts stay on the bar; the per-file breakdown and the
          sealing verification id open in place, so the bar keeps its height and its one line. */}
      <span
        className="hc__add"
        style={{ height: "100%" }}
        onBlur={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDetail(false);
        }}
        onKeyDown={(e) => {
          if (e.key === "Escape") setDetail(false);
        }}
      >
        <button
          type="button"
          className="vsc-statusbar__item vsc-statusbar__item--button"
          aria-expanded={detail}
          aria-controls="statusbar-diagnostics"
          aria-label={`${problemsLabel(diagnostics)}. Open the diagnostics detail`}
          title={problemsLabel(diagnostics)}
          data-tone={tone}
          onClick={() => setDetail((v) => !v)}
        >
          <Icon name="error" size={13} strokeWidth={1.8} />
          <span>{diagnostics ? diagnostics.errors : "-"}</span>
          <Icon name="warning" size={13} strokeWidth={1.8} style={{ marginLeft: 6 }} />
          <span>{diagnostics ? diagnostics.warnings : "-"}</span>
          {tone === "unknown" ? <span style={{ marginLeft: 6, color: "var(--text-3)" }}>not run</span> : null}
        </button>
        {detail ? <DiagnosticsDetail d={diagnostics} paths={analysisPaths(diffPatch, diagnostics, openPaths)} /> : null}
      </span>

      {/* Only while a lease is actually in force, like the jobs chip: an unleased bar is unchanged.
          The one line says that writes are leased; the declared scopes and the revoke live in the
          same progressive-disclosure popover the Problems counter uses, so no new permanent control
          appears. Revoke is here because this is the only surface that knows a lease exists, and a
          de-escalation must always be one gesture away from what shows it. */}
      {lease ? (
        <span
          className="hc__add"
          style={{ height: "100%" }}
          onBlur={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setLeaseOpen(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Escape") setLeaseOpen(false);
          }}
        >
          <button
            type="button"
            className="vsc-statusbar__item vsc-statusbar__item--button"
            aria-expanded={leaseOpen}
            aria-controls="statusbar-write-lease"
            aria-label={`${leaseLabel(lease)}. Open the write lease detail`}
            title={leaseLabel(lease)}
            onClick={() => setLeaseOpen((v) => !v)}
          >
            <Icon name="tool" size={13} strokeWidth={1.8} />
            <span>write lease</span>
          </button>
          {leaseOpen ? <WriteLeaseDetail lease={lease} /> : null}
        </span>
      ) : null}

      {/* Only while work is actually moving: an idle bar shows nothing extra. */}
      {jobs.length ? (
        <span className="vsc-statusbar__item" aria-label={jobsLabel(jobs)} title={jobsLabel(jobs)}>
          <Icon name="fleet" size={13} strokeWidth={1.8} />
          <span>{jobs.length} running</span>
        </span>
      ) : null}

      {latest ? (
        <span className="vsc-statusbar__item" style={{ color: latest.kind === "error" ? "var(--red)" : "var(--text-muted)" }}>
          {latest.message}
        </span>
      ) : null}

      <span className="vsc-statusbar__spacer" />

      <span className="vsc-statusbar__item">phase: {runPhase}</span>
      <span className="vsc-statusbar__item">{model}</span>
      <span className="vsc-statusbar__item">{TRANSPORT_KIND} transport</span>
      <span className="vsc-statusbar__item" title={runtimeDetail ?? undefined} aria-label={`Runtime ${runtimeStatus}`}>
        <span className={dotClass} style={{ width: 8, height: 8 }} />
        <span style={{ textTransform: "capitalize" }}>{runtimeStatus}</span>
      </span>
    </footer>
  );
}

/** The progressive half: what the one-line counter cannot say, plus the ONE action that fills it.
 *  The counter is the producer's natural home: `run_static_analysis` binds Custom, so this is the
 *  only place in the app that can write the `diagnostics` projection the bar reads. */
export function DiagnosticsDetail({ d, paths = [] }: { d: Diagnostics | null; paths?: string[] }) {
  const [state, setState] = useState<"idle" | "running">("idle");
  const run = () => {
    setState("running");
    void runCommand("run_static_analysis", { paths })
      .catch(noticeFailure("verify"))
      .finally(() => setState("idle"));
  };
  return (
    <div id="statusbar-diagnostics" className="hc__addmenu" role="group" aria-label="Diagnostics detail" style={DETAIL_POP}>
      <button
        type="button"
        className="settings__btn"
        disabled={!paths.length || state === "running"}
        title={
          paths.length
            ? `Analyse ${plural(paths.length, "file")}: ${paths.slice(0, 3).join(", ")}`
            : "Nothing to analyse yet: open a file or let the agent propose a change first"
        }
        aria-label={
          paths.length
            ? `Run static analysis on ${plural(paths.length, "file")}${state === "running" ? ", working" : ""}`
            : "Run static analysis, unavailable, no file is open and no change has been proposed"
        }
        onClick={run}
      >
        {state === "running" ? "Running analysis" : `Run static analysis (${paths.length})`}
      </button>
      {d ? (
        <>
          <span style={ROW}>
            {plural(d.errors, "error")}, {plural(d.warnings, "warning")}
          </span>
          <span style={MUTED}>
            {d.last_verification_id
              ? `sealed by verification ${d.last_verification_id}`
              : "no verification receipt sealed yet"}
          </span>
          {d.by_file.length ? (
            d.by_file.slice(0, 8).map((f) => (
              <span key={f.file} style={ROW}>
                <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>{f.file}</span>
                <span style={{ color: "var(--text-2)" }}>
                  {f.errors}e {f.warnings}w
                </span>
              </span>
            ))
          ) : (
            <span style={MUTED}>No file has a finding.</span>
          )}
          {d.by_file.length > 8 ? <span style={MUTED}>{d.by_file.length - 8} more files</span> : null}
        </>
      ) : (
        <span style={MUTED}>
          No static analysis has run in this session, so there are no real counts to show yet.
        </span>
      )}
    </div>
  );
}

/** The progressive half of the lease item: WHAT is leased, and the one gesture that ends it. */
export function WriteLeaseDetail({ lease }: { lease: WriteLease }) {
  const [state, setState] = useState<"idle" | "running">("idle");
  const revoke = () => {
    setState("running");
    void runCommand("revoke_write_lease", {})
      .catch(noticeFailure("revoke write lease"))
      .finally(() => setState("idle"));
  };
  return (
    <div id="statusbar-write-lease" className="hc__addmenu" role="group" aria-label="Write lease detail" style={DETAIL_POP}>
      <span style={MUTED}>
        This task may edit files inside the scopes below without asking per file. Everything else
        (writes outside them, shell, git, network) still asks.
      </span>
      {lease.scopes.length ? (
        lease.scopes.map((s) => (
          <span key={s} style={ROW}>
            <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", direction: "rtl" }}>{s}</span>
          </span>
        ))
      ) : (
        <span style={MUTED}>No scope is declared, so nothing is covered.</span>
      )}
      <button
        type="button"
        className="settings__btn"
        disabled={state === "running"}
        aria-label={`Revoke the write lease on ${lease.repo_id ?? "this workspace"}${state === "running" ? ", working" : ""}`}
        onClick={revoke}
      >
        {state === "running" ? "Revoking" : "Revoke write lease"}
      </button>
    </div>
  );
}

// Bottom-anchored bar, so the popover opens upward (the .hc__addmenu default) and to the right edge
// of the counter rather than off the left of the window.
const DETAIL_POP = { left: 0, minWidth: 260, maxWidth: 420 } as const;
const ROW = { display: "flex", gap: "var(--ma-3)", padding: "2px var(--ma-2)", fontSize: "var(--fs-small)" } as const;
const MUTED = { ...ROW, color: "var(--text-3)" } as const;
