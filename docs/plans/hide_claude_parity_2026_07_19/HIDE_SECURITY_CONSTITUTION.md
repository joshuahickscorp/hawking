# HIDE Security Constitution

Run date: 2026-07-19 · Grounding: clean-room facet 8 (security), `HIDE_LIVE_ARCHAEOLOGY.md` (hide-security), `HIDE_PERMISSION_AND_EFFECT_SYSTEM.md`.
Governing thesis (Claude Code's, adopted and extended): **deterministic, OS-enforced boundaries beat probabilistic model defenses**, because the deterministic boundary is what holds when everything probabilistic misses. [DOCUMENTED]

For an autonomous, execution-capable IDE this is not a feature; it is a phase-zero gate. The archaeology shows HIDE has strong *logic* for this (blake3 hash-chain audit, AES-256-GCM at-rest, secret redaction, macOS Seatbelt profile rendering, all in the packed `hide-security`, pure logic real and tested) but the OS *enforcement* (egress proxy, microVM, endpoint monitor) is a documented seam, and the live serve binds `0.0.0.0` with no auth. Security lands before autonomy.

## 1. Threat model (what HIDE must survive)

```text
malicious repository            malicious project config         malicious prompt
prompt injection in docs/       secret exfiltration              MCP compromise
issues/tool output              dependency compromise            shell injection
path traversal                  symlink attack                   localhost listener abuse
agent collusion                 memory poisoning                 extension compromise
```

Real precedent: CVE-2026-33068 (GHSA-mmgp-wc2j-qcv7) [DOCUMENTED] was exactly a repo-committed `settings.json` setting `bypassPermissions` that was resolved *before* the trust gate. HIDE's constitution is written to make that class impossible by construction.

## 2. Article I: trust before configuration (non-negotiable)

Treat project-open like an inbound untrusted request. On first open of a not-yet-trusted directory:

- Project-supplied capability-granting config (`allow` rules, additional directories, MCP servers, hooks, skills that pre-approve tools) is **parsed but INERT**. [parity `trust.workspace_gate`]
- **No permission mode is resolved before the gate** (the CVE fix). A repo cannot set itself to bypass.
- Restrictive config (`deny`/`ask`) applies immediately.
- The trust dialog **enumerates exactly what the folder would grant**.
- Trust persists keyed to the git repo root; home-directory trust is session-only.
- **No project hook, skill script, or MCP server executes before trust.** [Bible §60, §63]

## 3. Article II: filesystem boundaries

- Default access: workspace + explicitly-granted additional directories + scoped caches.
- **Canonical path and symlink resolution before scope checks.** Allow rules require both the symlink and its target to match; deny rules match if either matches. [DOCUMENTED]
- Writes confined to cwd + session temp, **enforced by the OS (Seatbelt/bubblewrap) and inherited by every subprocess**, not an in-process path check (a subprocess must not escape). [DOCUMENTED]
- Protected paths (`.git`, `.claude`/HIDE config, shell rc, package configs) route to a prompt *before* allow rules; `rm -rf /` and `~` circuit breakers fire even in the most permissive mode, detecting substitution-wrapped forms (`$(...)`, backticks, `<(...)`).
- Modes: read-write-no-delete and read-only as first-class execution modes. Recoverable deletes; transactional writes.

## 4. Article III: network egress (HIDE's strongest structural advantage)

Claude Code's own containment writeup names its **out-of-sandbox network proxy as its biggest source of failures**; egress is its *default architecture*. HIDE inverts this:

- **Local Hawking inference needs no egress at all.** There is no `api.anthropic.com` to allowlist. HIDE ships **egress default-fully-off** and is still a complete coding tool. [structural advantage, love/pain wedge #2]
- When a tool genuinely needs the network, route it through a mediating proxy *outside* the sandbox: pre-allow nothing, prompt on first new domain (allow-for-session), `allowedDomains`/`deniedDomains`, and a managed-only lockdown that blocks rather than prompts.
- `curl`/`wget` and network-fetching Bash are never auto-approved. Web fetches and untrusted tool output run in an **isolated context** (Article V).
- This eliminates the entire "approved-domain exfiltration to an attacker's account" class that survived even Claude Code's proxy in 24-of-25 red-team exfil attempts. [DOCUMENTED]

## 5. Article IV: the typed effect system

Every tool call declares its effect class; the effect ledger is the audit and the gate (see `HIDE_PERMISSION_AND_EFFECT_SYSTEM.md`):

```text
read · write · process · network · git-mutation · package-install · secret-access · destructive · external-side-effect
```

- Approvals are for meaningful exceptions and irreversible effects, not every low-risk step (Claude Code measured ~93% approval / ~84% reduction from sandboxing: approval fatigue is a security problem, not just UX).
- Effect batching when safe; simulate-first and explain-effect on demand.
- Read-only, local, private, side-effect-free, idempotent reads may be speculatively prepared **only** when the effect system proves them safe; **never speculate mutation, external requests, secrets, messages, writes, deletes, purchases, or credentials** (even discarded external requests leak intent, per "Ghost Tool Calls"). [Bible §47, dossier §5.7]

## 6. Article V: untrusted content isolation

- Web fetches and network-fetching commands run in a **context isolated from the main instruction stream**; connector/tool output is treated as **data even when the connector is "audited."** [DOCUMENTED]
- Provenance labels are immutable on all tool and external content; untrusted output is inspected/sanitized before entering model context.
- Subagent summaries are scanned/annotated for content imitating harness control output or permission grants; **no inter-agent message counts as user consent** (enforced at the runtime layer). [DOCUMENTED]
- Hawking superiority: quarantine untrusted content in a **forked read-only sub-state whose conclusions must be explicitly merged back** (a structural boundary stronger than a separate context window). [gated on state-capsule fork exposure]

## 7. Article VI: secrets and credentials

- Secrets and home-directory credentials (`~/.aws`, `~/.ssh`) are outside reach unless explicitly granted; a subprocess env-scrub by default.
- `deny` (unset env / block file read) and `mask` (per-session sentinel swapped for the real secret only on approved hosts by the proxy, requires TLS termination, fail-closed if misconfigured) modes. [DOCUMENTED]
- **HIDE never needs the user's cloud-LLM credentials in the environment at all** (no remote model call), shrinking the secret surface a sandboxed command can reach to only the user's own tool tokens. [structural advantage]
- Never enter financial/government/password credentials into any field; credential brokerage only via an approved local mechanism.

## 8. Article VII: sandboxes and isolation

- Native process sandbox (Seatbelt macOS; bubblewrap+seccomp Linux/WSL2) for **every** execution path including the terminal; native Windows unsupported (match Claude Code's honest limitation).
- Optional container/microVM for unattended fleet work; worktree isolation for write-heavy agents; filesystem overlay; network namespace; resource quotas (thermal/RAM, from the packed `hide-fleet` admission scheduler).
- Warm-state fork as a *disposable* sandbox: run a risky/first-run-untrusted action against a forked capsule; drop the fork if it misbehaves (near-zero-cost rollback the process sandbox alone cannot give). [gated on fork exposure]

## 9. Article VIII: durable integrity and audit

- Single-writer append-only event log with crash-safe framing and tail repair; a workspace single-writer lock (from the packed `hide-backend` event bus).
- blake3 tamper-evident hash-chain audit with genesis salt and signed anchors (from `hide-security`, real and tested).
- AES-256-GCM at-rest with an OS-keychain-wrapped key and fail-closed layout validation.
- Memory records treated as an injection-persistence surface (revalidate before use; supersede, do not silently trust; see `HIDE_MEMORY_SPEC.md`).
- Audit export without exposing sensitive content by default (redaction: regex + Shannon-entropy detectors).

## 10. Article IX: organization / managed policy

- A managed tier that lower tiers cannot override; `deny`/narrowing rules compose from any scope and cannot be removed; capability-widening (`allow`) lockable to managed-only.
- Managed lockdown keys (analogues of `allowManagedPermissionRulesOnly`, `allowManagedReadPathsOnly`, `allowManagedDomainsOnly`, fail-closed startup).
- **Local-only operation + no telemetry means an org can enforce policy without any code or prompt leaving the machine** (compliance without a data-residency review). [structural advantage]

## 11. Article X: autonomous-merge gate

Irreversible and high-risk operations retain human gates unless explicitly authorized (Bible §72). What evidence permits what is specified in `HIDE_AGENT_KERNEL_OPTIONS.md`; the constitution's floor: auto-commit/push to a non-default branch may be allowed under passing deterministic oracles; **auto-merge to the default branch and any force-push always require a human gate.**

## 12. Honesty clause

HIDE ships a plain-language limitations page (what the sandbox does and does not stop), warns when widening one boundary undoes another, and makes the enforcement layer inspectable/open. A local open runtime with no server component means **the entire trust boundary is on the user's machine and readable end-to-end** (there is no opaque cloud proxy for HIDE to get wrong). Candor and auditability are themselves the loved security behavior. [DOCUMENTED that this is what security-conscious developers value]
