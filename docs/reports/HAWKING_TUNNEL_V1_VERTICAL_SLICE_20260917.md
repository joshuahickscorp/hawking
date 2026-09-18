# Hawking Tunnel V1 protocol/admission slice — 2026-09-17

This slice extends the existing `hawking.share_bridge` owner and the existing
`hawkingd` H-Web server. It does not create a second daemon, VPN, screen-share
service, provider runtime, or mutation executor.

## Implemented

- one-use, short-lived device invitations with only invitation digests stored;
- host-share ∩ device-request ∩ session-authority capability intersection;
- direct-P2P selection with encrypted-relay fallback metadata;
- AES-256-GCM application frames with authenticated request IDs and sequences;
- reconnect cursors, connection disconnect, request idempotency, and replayable
  Tunnel events;
- fail-closed share revocation for all paired connections;
- scoped list-chats, list-models, result retrieval, prompt admission, and
  Goal-only Build admission through `/s/<share>/tunnel/*`;
- local host invitation control at `/hawking/web/tunnel`.

The share token, invitation, and reconnect secret are never persisted in raw
form. Pairing and request state remains inside the existing object-scoped share
record, so the Goal/Session stores remain the authority for data and actions.

## Evidence

```text
python3 -m pytest -q \
  hawking/tests/test_share_bridge.py \
  hawking/tests/test_web_live_contract.py \
  hawking/tests/test_web_projection.py \
  hawking/tests/test_hweb_controls.py
19 passed

python3 -m py_compile hawking/share_bridge.py hawking/serve.py \
  hawking/tests/test_share_bridge.py
passed

git diff --check
passed
```

## Claim boundary

This is a protocol and admission slice. `stream_prompt` and `scoped_build`
produce a bounded dispatch envelope for the canonical H-Web owner; they do not
execute a provider or mutate a repository from inside the Tunnel adapter.
Actual different-network rendezvous/relay hosting and remote H-Web streaming
remain an integration frontier for P18 and must be promoted only after the
P1/P2 graph dependencies are accepted.
