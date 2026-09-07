# Odyssey data-pipeline disposition

The former `tools/odyssey` corpus-ingestion sidecar was a closed fixture/data
CLI, not part of HCLI, ModelLake, or runtime admission. It had no production
callers outside its own tests. Its durable signal is preserved here:

- declared corpora were classified as present, absent, or partial;
- content-addressed membership used normalized text and exact/near-duplicate
  rejection before any training use;
- contamination checks rejected exact and near matches against hidden eval;
- teacher-trace assessment recorded the gap without upgrading a fixture into
  a capability claim.

The CLI, ingestion helpers, and dedicated fixture test were retired. Original
implementations remain recoverable in Git history; current HCLI research tools
and ModelLake admission remain the active owners.

The dormant T0 checkpoint-tournament judge, synthetic runtime reproduction,
and Odyssey Fusion Bridge adapter were also retired. Tournament decisions and
runtime identity remain in the sealed governance/evidence records; physical
graph and accelerator bridge implementations remain current.
