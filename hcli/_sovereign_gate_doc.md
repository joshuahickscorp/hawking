Protected sovereign verifiers. Each gate is RED until its capability exists.

Design rules, enforced by construction in every gate below:
  - gates on a DURABLE RECEIPT produced by a real run, never on a definition
  - the receipt must carry MEASURED fields; a placeholder/zero/absent field fails
  - a negative control proves the gate bites (a stub receipt must NOT pass)
  - "the model said done" is never accepted
  - a receipt whose producer is broken does not pass
  - self-contained: no shared helper, so no single file can weaken many gates
