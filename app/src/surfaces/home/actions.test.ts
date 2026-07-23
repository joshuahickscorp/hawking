/*
  actions.test.ts - the courtyard bindings: goals, attachments, workspace trust, background jobs,
  environment, and the duplicate cleanup. Behaviour, not line count: every plan must name a REAL
  catalog command, the staged files must reach the submit_turn attachments argument, the add-folder
  flow must make the trust decision explicit, and the retired mocks must actually be gone.
*/
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, it, expect } from "vitest";
import { commandById } from "../../store";
import {
  ENVIRONMENT_NOTE,
  GOAL_HINT,
  JOB_PHASE_GLYPH,
  JOB_PHASE_LABEL,
  TRUST_MEANING,
  environmentPlan,
  goalPlan,
  jobActionEnabled,
  jobLabel,
  jobPhase,
  jobPlan,
  readEnvironmentNotice,
  readJobNotice,
  repoIdFor,
  stageAttachments,
  trustPlan,
  workspaceRows,
  worktreeNotice,
  type JobView,
} from "./actions";
import { MODEL_ID_UNKNOWN, modelSwitchNote, modelId } from "../../shell/ModelChooser";

// Source assertions read the CODE, not the prose: comments explain the retirements and would
// otherwise match the very strings a retirement is supposed to remove.
const stripComments = (src: string) => src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");
const read = (rel: string) => stripComments(readFileSync(join(__dirname, rel), "utf8"));
const HOME = read("Home.tsx");
const COMPOSER = read("HomeComposer.tsx");
const SIDEBAR = read(join("..", "..", "shell", "SideBar.tsx"));
const SETTINGS = read(join("..", "Settings.tsx"));

const SES = "ses_test000000000000000000";

describe("A. goals ride the existing composer", () => {
  it("set, clear and evaluate all name real catalog commands", () => {
    for (const plan of [goalPlan.set(SES, "tests pass"), goalPlan.clear(SES), goalPlan.evaluate(SES)]) {
      expect(commandById(plan.id), `${plan.id} must be in the command catalog`).toBeTruthy();
    }
  });

  it("set carries the session and the acceptance condition", () => {
    expect(goalPlan.set(SES, "tests pass")).toEqual({
      id: "goal_set",
      args: { session_id: SES, condition: "tests pass" },
    });
  });

  it("clear carries only the session, so nothing keeps grading the run", () => {
    expect(goalPlan.clear(SES)).toEqual({ id: "goal_clear", args: { session_id: SES } });
  });

  it("evaluate is the deterministic acceptance check, session scoped", () => {
    expect(goalPlan.evaluate(SES)).toEqual({ id: "goal_evaluate", args: { session_id: SES } });
  });

  it("the composer dispatches all three through the shared plan runner, not ad hoc intents", () => {
    expect(COMPOSER).toContain("goalPlan.set(sessionId");
    expect(COMPOSER).toContain("goalPlan.clear(sessionId)");
    expect(COMPOSER).toContain("goalPlan.evaluate(sessionId)");
    expect(COMPOSER).not.toContain('intent.custom("goal');
  });

  it("says what a goal is for, so the control reads without documentation", () => {
    expect(GOAL_HINT.toLowerCase()).toContain("acceptance");
  });

  it("goal lives in the composer, not in a new panel", () => {
    expect(COMPOSER).toContain("hc__chip");
    expect(HOME).not.toContain("goalPlan");
  });
});

describe("B. staged files reach the submit_turn attachments argument", () => {
  const file = (name: string, body: string, type = "text/plain") => new File([body], name, { type });

  it("no files means an empty attachments list, never a fabricated one", async () => {
    expect(await stageAttachments([])).toEqual([]);
  });

  it("each staged file becomes a BlobRef with a real content digest", async () => {
    const [ref] = await stageAttachments([file("notes.txt", "hello")]);
    expect(ref.id).toBe("file:notes.txt");
    expect(ref.size_bytes).toBe(5);
    expect(ref.media_type).toBe("text/plain");
    // sha256("hello"), so the hash is the file's real digest and not a placeholder.
    expect(ref.hash).toBe("sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
  });

  it("different bytes give different digests", async () => {
    const [a, b] = await stageAttachments([file("a.txt", "one"), file("b.txt", "two")]);
    expect(a.hash).not.toBe(b.hash);
  });

  it("the composer stages the files it holds and sends them through the spine", () => {
    expect(COMPOSER).toContain("stageAttachments(staged)");
    expect(COMPOSER).toContain('runCommand("submit_turn", { session_id: sessionId, text: t, attachments })');
    // the private Intent builder that existed only because the spine dropped attachments is gone
    expect(COMPOSER).not.toContain("submitTurnWith");
  });
});

describe("C. add folder ends in an explicit trust decision", () => {
  it("trust names the real host capability and both states", () => {
    expect(commandById("workspace_set_repo_trust")).toBeTruthy();
    expect(trustPlan("web", "/Users/x/code/web", "trusted")).toEqual({
      id: "workspace_set_repo_trust",
      args: { repo_id: "web", root_path: "/Users/x/code/web", trust: "trusted" },
    });
    expect(trustPlan("web", "/Users/x/code/web", "untrusted").args.trust).toBe("untrusted");
    // The folder path travels with the decision: it is what puts the repo in the host graph, and
    // without it the trust call has nothing to act on.
    expect(trustPlan("web", "/Users/x/code/web", "trusted").args.root_path).toBe("/Users/x/code/web");
  });

  it("the repo id is the folder the host graph keys on", () => {
    expect(repoIdFor("/Users/x/code/web")).toBe("web");
    expect(repoIdFor("/Users/x/code/web/")).toBe("web");
    expect(repoIdFor("web")).toBe("web");
  });

  it("states the security meaning of each choice, not just a label", () => {
    expect(TRUST_MEANING.untrusted.toLowerCase()).toContain("inert");
    expect(TRUST_MEANING.trusted.toLowerCase()).toContain("active");
  });

  it("the flow adds the folder, then asks, and never auto-trusts", () => {
    // `open_folder` is retired: its host arm was empty and nothing read the record it wrote.
    expect(COMPOSER).not.toContain("open_folder");
    // The folder's PATH goes with the decision: it is what enters the repo into the host graph, so
    // the trust call has a node to act on instead of a silent miss.
    expect(COMPOSER).toContain("trustPlan(repo.id, repo.path, trust)");
    expect(COMPOSER).toContain('decideTrust("trusted")');
    expect(COMPOSER).toContain('decideTrust("untrusted")');
    // The added repo starts with NO decision recorded; only a user gesture writes one.
    expect(COMPOSER).toContain("trust: null");
  });
});

describe("D. background jobs use the existing rail with progressive disclosure", () => {
  const base: JobView = {
    phase: "active",
    jobId: null,
    jobEvent: null,
    approval: null,
    verification: null,
    process: null,
  };

  it("every job control names a real catalog command", () => {
    const plans = [
      jobPlan.promote("run_1", SES),
      jobPlan.pause("run_1"),
      jobPlan.resume("run_1"),
      jobPlan.stop("run_1"),
      jobPlan.foreground("job_1"),
    ];
    for (const p of plans) expect(commandById(p.id), `${p.id} must be in the catalog`).toBeTruthy();
  });

  it("promote carries the run and the session the host needs", () => {
    expect(jobPlan.promote("run_1", SES)).toEqual({
      id: "promote_run",
      args: { run_id: "run_1", session_id: SES },
    });
  });

  it("resume in foreground carries the durable job id", () => {
    expect(jobPlan.foreground("job_abc")).toEqual({
      id: "resume_run_foreground",
      args: { job_id: "job_abc" },
    });
  });

  it("control gestures on a promoted run reuse the run commands, no invented job verbs", () => {
    expect(jobPlan.pause("run_1").id).toBe("pause_run");
    expect(jobPlan.resume("run_1").id).toBe("resume_run");
    expect(jobPlan.stop("run_1").id).toBe("cancel_run");
  });

  it("phase comes from real run state, and a blocking gate outranks it", () => {
    expect(jobPhase("executing", false)).toBe("active");
    expect(jobPhase("planning", false)).toBe("pending");
    expect(jobPhase("paused", false)).toBe("paused");
    expect(jobPhase("awaiting", false)).toBe("blocked");
    expect(jobPhase("failed", false)).toBe("failed");
    expect(jobPhase("done", false)).toBe("done");
    expect(jobPhase("idle", false)).toBe("idle");
    expect(jobPhase("executing", true)).toBe("blocked");
  });

  it("pending, active, blocked, failed and completed all read differently without colour", () => {
    const keys = ["pending", "active", "blocked", "failed", "done"] as const;
    const words = keys.map((k) => JOB_PHASE_LABEL[k]);
    const glyphs = keys.map((k) => JOB_PHASE_GLYPH[k]);
    expect(new Set(words).size).toBe(keys.length);
    expect(new Set(glyphs).size).toBe(keys.length);
  });

  it("reads the durable job id and lifecycle event out of the host job event", () => {
    const notice = JSON.stringify({
      kind: "job_promoted",
      record: { job_id: "job_7f3a", session_id: SES, repo_id: null, goal_id: null },
    }).slice(0, 200);
    expect(readJobNotice(notice)).toEqual({
      jobId: "job_7f3a",
      event: "job_promoted",
      label: "running in the background",
    });
  });

  it("survives the 200 character truncation the notices strip applies", () => {
    const full = JSON.stringify({
      kind: "job_resumed_foreground",
      record: { job_id: "job_deadbeef", session_id: SES, triggers: Array(40).fill("tick") },
    });
    expect(full.length).toBeGreaterThan(200);
    expect(readJobNotice(full.slice(0, 200))?.jobId).toBe("job_deadbeef");
  });

  it("ignores unrelated custom events", () => {
    expect(readJobNotice(JSON.stringify({ kind: "search_results", count: 3 }))).toBeNull();
    expect(readJobNotice("not json at all")).toBeNull();
  });

  it("the row states approval, verification, blocker and process in words", () => {
    const label = jobLabel({
      phase: "blocked",
      jobId: "job_1",
      jobEvent: "running in the background",
      approval: "shell.run needs approval",
      verification: "2 errors, 0 warnings",
      process: "cargo test",
    });
    expect(label).toContain("waiting on you");
    expect(label).toContain("job job_1");
    expect(label).toContain("approval needed");
    expect(label).toContain("verification");
    expect(label).toContain("cargo test");
  });

  it("offers only what can actually fire right now", () => {
    expect(jobActionEnabled("promote", base, true)).toBe(true);
    expect(jobActionEnabled("promote", base, false)).toBe(false);
    // Already promoted: promoting again is not a thing.
    expect(jobActionEnabled("promote", { ...base, jobId: "job_1" }, true)).toBe(false);
    expect(jobActionEnabled("pause", base, true)).toBe(true);
    expect(jobActionEnabled("pause", { ...base, phase: "paused" }, true)).toBe(false);
    expect(jobActionEnabled("resume", { ...base, phase: "paused" }, true)).toBe(true);
    expect(jobActionEnabled("stop", { ...base, phase: "done" }, true)).toBe(false);
    // Resuming in the foreground needs a real job id, never a guessed one.
    expect(jobActionEnabled("foreground", base, true)).toBe(false);
    expect(jobActionEnabled("foreground", { ...base, jobId: "job_1" }, false)).toBe(true);
  });

  it("lives in the existing rail, collapsed, and unmounts when there is nothing running", () => {
    expect(HOME).toContain("home-recents__list");
    expect(HOME).toContain('aria-expanded={open}');
    expect(HOME).toContain("if (!runId && !view.jobId) return null;");
  });

  it("does not duplicate steer or fork, which existing surfaces already own", () => {
    expect(HOME).not.toContain("redirect_run");
    expect(HOME).not.toContain("fork_session");
  });
});

describe("E. environment and the workspace graph read", () => {
  it("environment switch names the real capability and keeps the session", () => {
    expect(commandById("environment_switch")).toBeTruthy();
    expect(environmentPlan(SES, "staging")).toEqual({
      id: "environment_switch",
      args: { session_id: SES, env_id: "staging", reason: "switched from settings" },
    });
  });

  it("explains that a switch re-scopes roots and permissions", () => {
    expect(ENVIRONMENT_NOTE.toLowerCase()).toContain("file roots");
    expect(ENVIRONMENT_NOTE.toLowerCase()).toContain("permissions");
  });

  it("reads back the environment the host actually switched to", () => {
    const notice = JSON.stringify({
      kind: "environment_switch",
      record: { session_id: SES, previous_env: "dev", new_env: "staging" },
    }).slice(0, 200);
    expect(readEnvironmentNotice(notice)).toBe("staging");
    expect(readEnvironmentNotice(JSON.stringify({ kind: "job_created" }))).toBeNull();
  });

  it("the workspace read is honest when the host reports nothing", () => {
    expect(workspaceRows(undefined)).toEqual([
      ["root", "no folder opened"],
      ["repo", "none"],
      ["branch", "no branch"],
      ["worktrees", "none"],
    ]);
  });

  it("renders what the host does report", () => {
    const rows = workspaceRows({ root: "/w", repo: "hawking", branch: "main", worktrees: ["wt-a", "wt-b"] });
    expect(rows).toContainEqual(["repo", "hawking"]);
    expect(rows).toContainEqual(["worktrees", "wt-a, wt-b"]);
  });

  it("settings hosts the workspace section and dispatches the switch", () => {
    expect(SETTINGS).toContain("workspaceRows(home?.workspace)");
    expect(SETTINGS).toContain("environmentPlan(sessionId, id)");
  });
});

describe("F. duplicate and mock cleanup", () => {
  it("the three switch_model copies are gone", () => {
    for (const [name, src] of [
      ["Home.tsx", HOME],
      ["HomeComposer.tsx", COMPOSER],
      ["SideBar.tsx", SIDEBAR],
      ["Settings.tsx", SETTINGS],
    ] as const) {
      expect(src.includes('intent.custom("switch_model"'), `${name} still fires switch_model`).toBe(false);
      expect(src.includes("switch model"), `${name} still offers a switch-model control`).toBe(false);
    }
  });

  it("one chooser component is the single place that choice is presented", () => {
    expect(SIDEBAR).toContain('from "./ModelChooser"');
    expect(SETTINGS).toContain('from "../shell/ModelChooser"');
    expect(COMPOSER).toContain('from "../../shell/ModelChooser"');
    expect(SIDEBAR).toContain("<ModelChooser");
    expect(SETTINGS).toContain("<ModelChooser");
  });

  it("the chooser is honestly labelled rather than pretending", () => {
    expect(modelSwitchNote(null).toLowerCase()).toContain("no model-switch capability");
    // and the note tells the truth about what is loaded: it used to assert "One local model is
    // loaded" unconditionally, printed beside this file's own "no model reported".
    expect(modelSwitchNote(null).toLowerCase()).toContain("no model is loaded");
    expect(
      modelSwitchNote({ model: { id: "qwen3-coder", arch: "qwen", ctx: 1, profile: "p", sampling: "s" } }).toLowerCase(),
    ).toContain("one local model is loaded");
    // no manifest = unknown, never an invented id (four surfaces used to print two different ones)
    expect(modelId(null)).toBe(MODEL_ID_UNKNOWN);
    expect(MODEL_ID_UNKNOWN).not.toMatch(/qwen/i);
    expect(modelId({ model: { id: "qwen3-coder", arch: "qwen", ctx: 1, profile: "p", sampling: "s" } })).toBe(
      "qwen3-coder",
    );
  });

  it("the composer voice mic mock is retired, recorder and all", () => {
    expect(COMPOSER).not.toContain("MediaRecorder");
    expect(COMPOSER).not.toContain("getUserMedia");
    expect(COMPOSER).not.toContain('name="mic"');
  });

  it("the misleading Artifacts rail item is retired", () => {
    expect(HOME).not.toContain("home-nav\" onClick={onPopToCode}");
    expect(HOME).not.toMatch(/<Icon name="box" size=\{15\} \/> Artifacts/);
  });
});

describe("F. a refused command surfaces the refusal, never a success notice", () => {
  it("says what the host said when the worktree request is refused", () => {
    const n = worktreeNotice({ accepted: false, event_seq: null, message: "no git repo here" });
    expect(n.kind).toBe("error");
    expect(n.message).toBe("no git repo here");
  });

  it("falls back to a refusal, not a success, when the host gives no reason", () => {
    const n = worktreeNotice({ accepted: false, event_seq: null, message: null });
    expect(n.kind).toBe("error");
    expect(n.message).toContain("refused");
  });

  it("does not call an accepted request a finished worktree, and names no branch it did not use", () => {
    const n = worktreeNotice({ accepted: true, event_seq: 9, message: null });
    expect(n.kind).toBe("info");
    expect(n.message).toContain("requested");
    // the host creates hide/<slug>, so the notice must not claim the branch the user is standing on
    expect(n.message).toContain("hide/");
  });

  it("a HELD request says the gate is waiting, and never reads as done", () => {
    const n = worktreeNotice({ accepted: true, held: true, event_seq: 9, message: "held for approval: gate=g1" });
    expect(n.kind).toBe("info");
    expect(n.message).toContain("approve");
    expect(n.message).not.toContain("done");
  });

  it("the composer pushes the ack-derived notice and nothing optimistic", () => {
    expect(COMPOSER).toContain('worktreeNotice(await runCommand("create_worktree"))');
    // The two chips that only ever produced a toast are gone, control and handler both.
    expect(COMPOSER).not.toContain("createPr");
    expect(COMPOSER).not.toContain("cycleEffort");
    expect(COMPOSER).not.toContain("Create PR");
  });

  it("the fleet keep-best control, which promised to discard the other branches, is retired", () => {
    const fleet = read(join("..", "fleet", "FleetView.tsx"));
    expect(fleet).not.toContain("branch__keep");
    expect(fleet).not.toContain("keep best");
    expect(fleet).toContain('runCommand("cancel_run"'); // stop is real and stays, through the spine
  });
});
