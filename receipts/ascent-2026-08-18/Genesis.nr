{
  "nr_version": "1.0.0",
  "nr_kind": "hawking.nos.noetic_representation",
  "semantic_provenance": {
    "parent_model": "Qwen3.8-27B (Genesis patient, abliterated)",
    "parent_revision": "bf16",
    "parameter_count": 26895998464,
    "catalog_content_sha256": "4e5b11a8d55ee2a269fd56626e419f4f7fb713b47feb887a389d68b39378cbcd",
    "pack_schema": "hawking.ascent.qwen38_mlp_not_r160.v1",
    "patient_note": "abliterated parent; Tabula drift is a Doctor axis and is NOT recorded here because NR states what the representation IS, not how it scored"
  },
  "representation": {
    "tensors": {
      "count": 851,
      "encoded_count": 496,
      "copied_count": 355,
      "payload_bytes": 11244907853,
      "codec_families": [
        {
          "family": "grouped_absmax",
          "bits": 3,
          "group": 64,
          "applies_to": "mlp_and_attention_encoded_tensors",
          "count": 496,
          "organ_bits_per_weight": {
            "mlp": 3.2500251321231617,
            "non_mlp": 3.510326662656368
          }
        },
        {
          "family": "grouped_absmax",
          "bits": 4,
          "group": 64,
          "applies_to": "endpoints_and_copied_tensors",
          "count": 355
        }
      ],
      "complete_bits_per_weight": 3.3447723007722434
    },
    "entropy_streams": [],
    "shared_structures": [],
    "generated_structures": [],
    "latent_codes": [],
    "correction_planes": [],
    "exact_islands": [],
    "route_graph": null,
    "absent_sections_are_measured_not_unfilled": "empty for the same reasons as the uniform NR: G035/G062 refuted sharing, G042 records GENERATED and CORRECTION at 0, G043 records ROUTING_FLOPS at 0 for this dense model, and no packer emits an entropy stream yet (the ~2.5 BPW rANS ladder was refuted on-disk, session f5169750)."
  },
  "kernel_requirements": [
    {
      "requires": "grouped_absmax_decoder",
      "bits": 3,
      "group": 64,
      "note": "the mixed q3 MLP/attention decoder family; naming a specific kernel, threadgroup geometry or device here would make this NX, not NR"
    },
    {
      "requires": "grouped_absmax_decoder",
      "bits": 4,
      "group": 64,
      "note": "the q4 endpoints/copied decoder family"
    },
    {
      "requires": "gated_delta_recurrence",
      "note": "the DeltaNet mixer family the representation assumes; no implementation named"
    }
  ]
}
