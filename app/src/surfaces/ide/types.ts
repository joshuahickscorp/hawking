/*
  ide/types.ts: the local view-model the IDE surface folds out of the store's generic
  projection bag (s.projections.diff / .editor / .file_external). These mirror what the host's
  projection_patch{diff} carries (D1.2: a diff_id with hunks, status flips applied on accept).
  Kept local to the surface (HARD RULE: no edits to the shared store/wire), parsed defensively
  so a partial host patch never throws. When the mock is the transport, MOCK_DIFF stands in.
*/

// One reviewable hunk: a contiguous run of -/+ lines, reviewed as a unit (j/k to move, a/r to act).
export type HunkStatus = "pending" | "accepted" | "rejected" | "applied";
export type LineKind = "ctx" | "add" | "del";

export interface DiffLine {
  kind: LineKind;
  text: string;
  // 1-based line numbers in the old / new file; null on the side a line does not exist.
  oldNo: number | null;
  newNo: number | null;
}

export interface Hunk {
  id: string;
  header: string; // the @@ -a,b +c,d @@ context label
  lines: DiffLine[];
  status: HunkStatus;
}

export interface DiffDoc {
  diff_id: string;
  run_id: string;
  path: string;
  lang: string; // monaco language id
  before: string; // full old-side text (for the Monaco diff model)
  after: string; // full new-side text
  hunks: Hunk[];
  stale: boolean; // file changed under a pending diff (D1.2 Stale state): never apply onto drift
}

export interface FileNode {
  path: string;
  name: string;
  dir: boolean;
  // git/agent badges shown in the tree (file_external + tool_progress feed these in the live host).
  badge?: "added" | "modified" | "touched";
  children?: FileNode[];
}

// ---- Defensive folds from the generic projection bag (Record<string, unknown>) ----

const asStr = (v: unknown, fallback = ""): string => (typeof v === "string" ? v : fallback);

export function parseDiff(patch: Record<string, unknown> | undefined): DiffDoc | null {
  if (!patch) return null;
  const hunks = Array.isArray(patch.hunks) ? (patch.hunks as Hunk[]) : null;
  if (!hunks || !patch.diff_id) return null;
  return {
    diff_id: asStr(patch.diff_id),
    run_id: asStr(patch.run_id),
    path: asStr(patch.path, "untitled"),
    lang: asStr(patch.lang, "plaintext"),
    before: asStr(patch.before),
    after: asStr(patch.after),
    hunks,
    stale: patch.stale === true,
  };
}

// Apply the host's status-flip (a follow-up projection_patch{diff} after AcceptDiff) by id.
export function applyHunkStatus(doc: DiffDoc, hunkId: string, status: HunkStatus): DiffDoc {
  return { ...doc, hunks: doc.hunks.map((h) => (h.id === hunkId ? { ...h, status } : h)) };
}

/*
  MOCK_DIFF: a believable agent edit for the scripted run (the mock transport emits no diff
  projection, so the surface seeds this so the core gesture is ALIVE with no backend). It is the
  same change the mock's tool_progress narrates: "edit guard.rs: moved drop past retry".
*/
const BEFORE = `pub async fn acquire(&self) -> Result<Conn, PoolError> {
    let permit = self.sem.acquire().await?;
    let conn = self.checkout()?;
    if conn.is_stale() {
        drop(permit);
        return self.acquire_retry().await;
    }
    Ok(Conn::new(conn, permit))
}`;

const AFTER = `pub async fn acquire(&self) -> Result<Conn, PoolError> {
    let permit = self.sem.acquire().await?;
    let conn = self.checkout()?;
    if conn.is_stale() {
        let fresh = self.acquire_retry().await?;
        return Ok(fresh);
    }
    Ok(Conn::new(conn, permit))
}`;

export const MOCK_DIFF: DiffDoc = {
  diff_id: "diff_guard_retry_1",
  run_id: "run_mock0000000000000000000",
  path: "crates/pool/src/guard.rs",
  lang: "rust",
  before: BEFORE,
  after: AFTER,
  stale: false,
  hunks: [
    {
      id: "h1",
      header: "@@ -4,4 +4,4 @@ pub async fn acquire",
      status: "pending",
      lines: [
        { kind: "ctx", text: "    if conn.is_stale() {", oldNo: 4, newNo: 4 },
        { kind: "del", text: "        drop(permit);", oldNo: 5, newNo: null },
        { kind: "del", text: "        return self.acquire_retry().await;", oldNo: 6, newNo: null },
        { kind: "add", text: "        let fresh = self.acquire_retry().await?;", oldNo: null, newNo: 5 },
        { kind: "add", text: "        return Ok(fresh);", oldNo: null, newNo: 6 },
        { kind: "ctx", text: "    }", oldNo: 7, newNo: 7 },
      ],
    },
  ],
};

// A small believable tree for the Explorer until the code_index connector is live (mock returns []).
export const MOCK_TREE: FileNode[] = [
  {
    path: "crates",
    name: "crates",
    dir: true,
    children: [
      {
        path: "crates/pool/src",
        name: "pool/src",
        dir: true,
        children: [
          { path: "crates/pool/src/guard.rs", name: "guard.rs", dir: false, badge: "modified" },
          { path: "crates/pool/src/lib.rs", name: "lib.rs", dir: false },
          { path: "crates/pool/src/retry.rs", name: "retry.rs", dir: false, badge: "touched" },
        ],
      },
      {
        path: "crates/hide-core/src",
        name: "hide-core/src",
        dir: true,
        children: [
          { path: "crates/hide-core/src/api.rs", name: "api.rs", dir: false },
          { path: "crates/hide-core/src/event.rs", name: "event.rs", dir: false },
        ],
      },
    ],
  },
  {
    path: "tests",
    name: "tests",
    dir: true,
    children: [{ path: "tests/pool_exhausted.rs", name: "pool_exhausted.rs", dir: false, badge: "added" }],
  },
];

// Stub file bodies so an OpenFile click in the mock shows real content in the editor.
export const MOCK_FILE_BODY: Record<string, { lang: string; text: string }> = {
  "crates/pool/src/guard.rs": { lang: "rust", text: AFTER },
  "crates/pool/src/lib.rs": {
    lang: "rust",
    text: `mod guard;\nmod retry;\n\npub use guard::Conn;\npub use guard::Pool;\n`,
  },
  "crates/pool/src/retry.rs": {
    lang: "rust",
    text: `impl Pool {\n    pub(crate) async fn acquire_retry(&self) -> Result<RawConn, PoolError> {\n        self.checkout()\n    }\n}\n`,
  },
  "crates/hide-core/src/api.rs": {
    lang: "rust",
    text: `pub enum Intent {\n    SubmitTurn { session_id: SessionId, text: String },\n    AcceptDiff { run_id: RunId, diff_id: String },\n}\n`,
  },
  "tests/pool_exhausted.rs": {
    lang: "rust",
    text: `#[tokio::test]\nasync fn exhausted_pool_releases_permit() {\n    let pool = Pool::new(1);\n    let _a = pool.acquire().await.unwrap();\n    // second acquire must not deadlock after a failed retry\n}\n`,
  },
};
