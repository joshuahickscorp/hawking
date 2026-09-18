# Arsenal dependency and third-party notice — 2026-09-15

The implementation adds no new network or provider dependency.

| Addition | Dependencies | Disk/runtime effect | Status |
|---|---|---|---|
| `hawking/source_tools.py` | Python standard library only (`ast`, `difflib`, `hashlib`, filesystem APIs) | no model or browser launch | active registry adapter |
| `crates/hawking-web` | existing workspace `serde` and `serde_json` | one workspace-package lock entry; no new third-party package | compiled/tested staging contract |

The existing `crates/hawking-index` Tree-sitter dependency set remains the
structural index authority. No copy of its index, parser cache, model weights,
credentials, or receipts was created.

No OpenRouter call was made and remote spend attributable to this checkpoint is
`$0.00`. Historical HIDE/AgentOS crates and Claude sources remain in place;
they were not deleted or rewritten.
