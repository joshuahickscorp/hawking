/*
  mockdiffs.ts: believable review branches for the merge-review queue so the surface is ALIVE
  with no backend. In live mode these arrive as projection_patch:diff (a diff_id with hunks)
  keyed to a run; here we hand-author one branch per finished run. Pure data, no UI.
*/
import type { ReviewBranch } from "./hunkreview";

export const MOCK_BRANCHES: ReviewBranch[] = [
  {
    run_id: "run_a_b1",
    diff_id: "diff_guard_drop",
    label: "drop past retry boundary",
    path: "crates/pool/src/guard.rs",
    hunks: [
      {
        id: "h1",
        header: "@@ guard.rs 42,7 +42,9 @@ impl PoolGuard",
        lines: [
          { kind: "ctx", text: "    pub async fn acquire(&self) -> Result<Permit> {" },
          { kind: "del", text: "        let permit = self.sem.acquire().await?;" },
          { kind: "del", text: "        drop(conn);" },
          { kind: "add", text: "        let permit = self.sem.clone().acquire_owned().await?;" },
          { kind: "add", text: "        // hold the permit across the retry; release only on real failure" },
          { kind: "ctx", text: "        for attempt in 0..self.max_retries {" },
        ],
      },
      {
        id: "h2",
        header: "@@ guard.rs 71,4 +73,6 @@",
        lines: [
          { kind: "ctx", text: "        }" },
          { kind: "add", text: "        drop(permit); // exhausted: now release" },
          { kind: "add", text: "        Err(PoolError::Exhausted)" },
          { kind: "del", text: "        Err(PoolError::Exhausted)" },
        ],
      },
    ],
  },
  {
    run_id: "run_b",
    diff_id: "diff_retry_test",
    label: "add retry tests",
    path: "crates/pool/tests/exhausted.rs",
    hunks: [
      {
        id: "t1",
        header: "@@ exhausted.rs new file +1,12 @@",
        lines: [
          { kind: "add", text: "#[tokio::test]" },
          { kind: "add", text: "async fn exhausted_pool_releases_permit() {" },
          { kind: "add", text: "    let pool = Pool::new(1);" },
          { kind: "add", text: "    let _held = pool.acquire().await.unwrap();" },
          { kind: "add", text: "    assert!(pool.try_acquire().is_err());" },
          { kind: "add", text: "    drop(_held);" },
          { kind: "add", text: "    assert!(pool.try_acquire().is_ok());" },
          { kind: "add", text: "}" },
        ],
      },
    ],
  },
];
