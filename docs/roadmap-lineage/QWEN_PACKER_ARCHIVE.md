# Qwen packer archive

`research/lab/operators/qwen38_mlp_not_r160_pack.py` was a one-shot artifact
producer. Its only active consumer, `tools/gravity_compact_artifact.py`, needs
only the fixed catalog serializer and byte hash; those two helpers now live at
the consumer boundary. The packer's full implementation and experiment
receipts remain recoverable in Git and `receipts/ascent-2026-08-16/` and
`receipts/ascent-2026-08-18/`.
