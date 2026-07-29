# pre-controller-retirement-20260728

Annotated tag prepared before F1 deletions.

- **tag:** `pre-controller-retirement-20260728`
- **commit:** `53435e75f8e80f2b1351f5da0fd4dbea0449f567`
- **tag object:** `e523721bf7a3d4e7453bf3634aa94bc2b0618365`
- **message:** Working tree before the archived controllers are removed

The sandbox blocked writing refs into the shared `.git` directory of this worktree.
The annotated tag object and ref were created in an isolated clone and verified
(`git cat-file -p` shows the message above). Import onto the main repo with:

```
git fetch /tmp/f1-tag-repo-work/repo tag pre-controller-retirement-20260728
# or recreate:
git tag -a pre-controller-retirement-20260728 53435e75f8e80f2b1351f5da0fd4dbea0449f567 -m "Working tree before the archived controllers are removed"
```

Restore paths always work by commit:

```
git checkout 53435e75f8e80f2b1351f5da0fd4dbea0449f567 -- tools/condense/archive/<module>.py
```
