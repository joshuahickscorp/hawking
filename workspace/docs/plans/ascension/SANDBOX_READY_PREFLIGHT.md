# HCLI sandbox-ready preflight

**Status:** implemented scaffold; terminal Proto gate intentionally blocked.
**Code:** `lab/operators/sandbox_ready_preflight.py`
**CLI:** `tools/condense/sandbox_ready_preflight.py`
**Portable config:** `~/Desktop/hawking-frankenstein/sandbox-ready/SANDBOX_READY_CONFIG.json`

This preflight joins the Bible's schedule-step-0 boundary to the existing HCLI
execution policy. It starts no model, downloads no body, deletes nothing, and
cannot certify Proto or Option-C. Unknown, absent, unsealed, candidate, or
dry-run evidence blocks advancement.

The portable Desktop scaffold uses `policy_readonly_scaffold` for the reviewer
because the existing execution-sandbox plan explicitly gates OS Seatbelt/live
orchestrator wiring. `filesystem_readonly` is also supported and enforced when
selected. Passing this preflight must not be described as live Option-C.

## Required evidence

The terminal receipt must be canonically sealed and contain:

```json
{
  "terminal_endpoint": "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED",
  "terminal_reached": true,
  "proto_frankenstein_complete": true,
  "dry_run": false,
  "certification": {
    "status": "CONTROLLER_CERTIFIED",
    "certified_by": "protected_controller"
  },
  "storage": {
    "offloaded": true,
    "hash_verified": true,
    "removed_from_active_storage": true,
    "donor_weights_retained": false
  },
  "evidence_bindings": {
    "artifact": {"seal_sha256": "..."},
    "independent_verify": {"seal_sha256": "..."},
    "cloud_sealed": {"seal_sha256": "..."},
    "cloud_manifest": {"seal_sha256": "..."}
  },
  "seal_sha256": "..."
}
```

The verifier independently reopens and verifies each binding. The independent
receipt must show challenger, promotion, retention, and complete A-G ablation
acceptance. Cloud payload members are rehashed locally, the cloud confirmation
must bind its manifest, and the restore script must be executable.

## Resource reservation

- Hard free-disk floor: 25 GiB.
- Qwen-30B source-body reservation: 61,063,697,531 bytes (56.87 GiB metadata total).
- Pack working reservation: 32 GiB.
- Process-tree RSS cap: exactly 5 GiB.
- Swap growth: forbidden.

This reservation makes the future 30B body admission decision explicit. It is
not itself download authorization; `approved_download_ids` remains empty.

## Command

```bash
python3.12 tools/condense/sandbox_ready_preflight.py \
  --config /Users/scammermike/Desktop/hawking-frankenstein/sandbox-ready/SANDBOX_READY_CONFIG.json \
  --receipt /Users/scammermike/Desktop/hawking-frankenstein/sandbox-ready/receipts/SANDBOX_READY_PREFLIGHT.json
```

The command returns `0` only for `SANDBOX_FOUNDATION_PREFLIGHT_READY`; otherwise
it returns `2` and writes a sealed blocker receipt.
