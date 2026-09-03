#!/usr/bin/env python3
"""G103: the NR container -- protected representation with NO machine-specific fields.

NR is what the patient IS. NX is how one machine runs it. The whole value of the
split is that an NR file must be portable, so the spec's teeth are not in what it
ALLOWS but in what it REFUSES: any field that could only be true of one machine
belongs in NX and is rejected here.

That distinction has one genuinely subtle case, and the spec has to get it right.
The obligation lists "kernel requirements" as an NR field, which sounds
machine-specific and is not: NR may say "this stream needs a grouped-absmax
decoder at group 64", because that is a property of the REPRESENTATION. It may not
say "dispatch qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 with threadgroup 128",
because that is a property of a MACHINE. Requirement is portable; binding is not.

The negative test is the point of the exercise. A validator that has never been
watched REJECT something has not been shown to work, so --negative-test injects a
real machine-specific field and the run is only meaningful if that exits non-zero.

  ./tools/nr_container.py --serialize uniform-q4-v1 --out /tmp/nr.json
  ./tools/nr_container.py --validate /tmp/nr.json          # expect exit 0
  ./tools/nr_container.py --negative-test /tmp/nr.json     # expect exit 1
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/campaign/records/runs/qwen38-27b"
SCHEMA = ROOT / "docs/spec/nr_container.schema.json"

# Fields that can only be true of one machine. Presence of any of these ANYWHERE in
# an NR document is a rejection, not a warning.
MACHINE_SPECIFIC = {
    "kernel", "kernels", "kernel_name", "shader", "metallib", "threadgroup",
    "threadgroup_size", "tg_size", "grid", "dispatch", "dispatches", "device",
    "device_id", "machine_genome", "gpu", "residency_plan", "cache_plan",
    "schedule", "ps_per_element", "gb_s", "tps", "token_ns", "occupancy",
    "register_pressure", "simd_width", "r_tiling", "k_amortization",
}


def walk_keys(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k, f"{path}.{k}" if path else k
            yield from walk_keys(v, f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk_keys(v, f"{path}[{i}]")


def validate(doc):
    """Returns (ok, problems). Machine-specific fields are the hard rejection."""
    bad = []
    for k, path in walk_keys(doc):
        if k.lower() in MACHINE_SPECIFIC:
            bad.append(f"MACHINE-SPECIFIC FIELD in NR: {path}")
    req = ["nr_version", "semantic_provenance", "representation", "kernel_requirements"]
    for r in req:
        if r not in doc:
            bad.append(f"missing required section: {r}")
    prov = doc.get("semantic_provenance", {})
    for r in ("parent_model", "parent_revision", "parameter_count"):
        if r not in prov:
            bad.append(f"semantic_provenance missing {r}")
    for kr in doc.get("kernel_requirements", []):
        # A REQUIREMENT names a decoder family and its parameters. If it names an
        # implementation, it is a binding and belongs in NX.
        if "implementation" in kr or "kernel" in kr:
            bad.append(f"kernel_requirements entry names an implementation, not a requirement: {kr}")
    return (not bad), bad


def serialize_catalog(name, root):
    """Build the NR for a mixed catalog artifact (patient family) from PACK_REPORT.json.
    Content-addressed: the catalog's sha256 binds this NR to the exact on-disk representation."""
    import hashlib
    rep = json.loads((root / "PACK_REPORT.json").read_text())
    cat = root / "catalog.hq38m20"
    cat_sha = hashlib.sha256(cat.read_bytes()).hexdigest() if cat.is_file() else None
    return {
        "nr_version": "1.0.0",
        "nr_kind": "hawking.nos.noetic_representation",
        "semantic_provenance": {
            "parent_model": "Qwen3.8-27B (Genesis patient, abliterated)",
            "parent_revision": pathlib.Path(rep.get("source_bf16", "unknown")).name,
            "parameter_count": rep.get("source_weight_elements"),
            "catalog_content_sha256": cat_sha,
            "pack_schema": rep.get("schema"),
            "patient_note": "abliterated parent; Tabula drift is a Doctor axis and is NOT recorded "
                            "here because NR states what the representation IS, not how it scored",
        },
        "representation": {
            "tensors": {
                "count": rep.get("tensor_count"),
                "encoded_count": rep.get("encoded_tensors"),
                "copied_count": rep.get("copied_tensors"),
                "payload_bytes": rep.get("tensor_payload_bytes"),
                "codec_families": [
                    {"family": "grouped_absmax", "bits": 3, "group": 64,
                     "applies_to": "mlp_and_attention_encoded_tensors",
                     "count": rep.get("encoded_tensors"),
                     "organ_bits_per_weight": {"mlp": rep.get("mlp_physical_bpw"),
                                               "non_mlp": rep.get("nonmlp_physical_bpw")}},
                    {"family": "grouped_absmax", "bits": 4, "group": 64,
                     "applies_to": "endpoints_and_copied_tensors",
                     "count": rep.get("copied_tensors")},
                ],
                "complete_bits_per_weight": rep.get("complete_physical_bpw"),
            },
            "entropy_streams": [],
            "shared_structures": [],
            "generated_structures": [],
            "latent_codes": [],
            "correction_planes": [],
            "exact_islands": [],
            "route_graph": None,
            "absent_sections_are_measured_not_unfilled": (
                "empty for the same reasons as the uniform NR: G035/G062 refuted sharing, "
                "G042 records GENERATED and CORRECTION at 0, G043 records ROUTING_FLOPS at 0 "
                "for this dense model, and no packer emits an entropy stream yet (the ~2.5 BPW "
                "rANS ladder was refuted on-disk, session f5169750)."),
        },
        "kernel_requirements": [
            {"requires": "grouped_absmax_decoder", "bits": 3, "group": 64,
             "note": "the mixed q3 MLP/attention decoder family; naming a specific kernel, "
                     "threadgroup geometry or device here would make this NX, not NR"},
            {"requires": "grouped_absmax_decoder", "bits": 4, "group": 64,
             "note": "the q4 endpoints/copied decoder family"},
            {"requires": "gated_delta_recurrence",
             "note": "the DeltaNet mixer family the representation assumes; no implementation named"},
        ],
    }


def serialize(name):
    root = RUNS / name
    # Two on-disk artifact formats: the uniform-q4 manifest.json, and the mixed
    # catalog (PACK_REPORT.json + catalog.hq38m20 + segments/). NR must represent
    # either; it states what the representation IS, never how one machine runs it.
    if not (root / "manifest.json").is_file() and (root / "PACK_REPORT.json").is_file():
        return serialize_catalog(name, root)
    man = json.loads((root / "manifest.json").read_text())
    tensors = man.get("tensors", {})
    if isinstance(tensors, dict):
        n_tensors = len(tensors)
    else:
        n_tensors = man.get("tensor_count", 0)
    return {
        "nr_version": "1.0.0",
        "nr_kind": "hawking.nos.noetic_representation",
        "semantic_provenance": {
            "parent_model": "Qwen3.8-27B (Genesis patient)",
            "parent_revision": man.get("source_dir", "unknown"),
            "parameter_count": man.get("source_weight_elements"),
            "patient_note": "abliterated parent; Tabula drift is a Doctor axis and is NOT recorded "
                            "here because NR states what the representation IS, not how it scored",
        },
        "representation": {
            "tensors": {
                "count": n_tensors,
                "payload_bytes": man.get("tensor_payload_bytes"),
                "codec_families": [
                    {"family": "grouped_absmax", "bits": 4, "group": man.get("q4_group_size", 64),
                     "applies_to": "q4_tensors", "count": man.get("q4_tensors")},
                    {"family": "raw_f32", "applies_to": "f32_tensors",
                     "count": man.get("f32_tensors")},
                ],
                "complete_bits_per_weight": man.get("complete_physical_bpw"),
            },
            "entropy_streams": [],
            "shared_structures": [],
            "generated_structures": [],
            "latent_codes": [],
            "correction_planes": [],
            "exact_islands": [],
            "route_graph": None,
            "absent_sections_are_measured_not_unfilled": (
                "entropy_streams, shared_structures, generated_structures, latent_codes, "
                "correction_planes, exact_islands and route_graph are EMPTY because this campaign "
                "measured them absent or refuted: G035/G062 refuted sharing, G042 records "
                "GENERATED_BPW_EQUIVALENT and CORRECTION_BPW at 0, G043 records ROUTING_FLOPS at 0 "
                "for this dense model, and no packer emits an entropy stream yet."),
        },
        "kernel_requirements": [
            {"requires": "grouped_absmax_decoder", "bits": 4,
             "group": man.get("q4_group_size", 64),
             "note": "a REQUIREMENT on the decoder family and its parameters. Naming a specific "
                     "kernel, threadgroup geometry or device here would make this NX, not NR."},
            {"requires": "gated_delta_recurrence",
             "note": "the mixer family the representation assumes; no implementation named"},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serialize"); ap.add_argument("--validate"); ap.add_argument("--negative-test")
    ap.add_argument("--out", type=pathlib.Path); ap.add_argument("--write-schema", action="store_true")
    a = ap.parse_args()

    if a.write_schema:
        SCHEMA.parent.mkdir(parents=True, exist_ok=True)
        SCHEMA.write_text(json.dumps({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Hawking NR container",
            "description": "Protected representation. Contains what the patient IS. Any field that "
                           "could only be true of one machine belongs in the NX genome (G104) and "
                           "is rejected by the validator.",
            "type": "object",
            "required": ["nr_version", "semantic_provenance", "representation", "kernel_requirements"],
            "properties": {
                "nr_version": {"type": "string"},
                "semantic_provenance": {"type": "object",
                    "required": ["parent_model", "parent_revision", "parameter_count"]},
                "representation": {"type": "object"},
                "kernel_requirements": {"type": "array", "items": {"type": "object",
                    "required": ["requires"],
                    "not": {"anyOf": [{"required": ["implementation"]}, {"required": ["kernel"]}]}}},
            },
            "x-rejected-field-names": sorted(MACHINE_SPECIFIC),
        }, indent=2) + "\n")
        print(f"wrote {SCHEMA.relative_to(ROOT)}")
        return 0

    if a.serialize:
        doc = serialize(a.serialize)
        ok, bad = validate(doc)
        txt = json.dumps(doc, indent=2) + "\n"
        if a.out:
            a.out.write_text(txt)
        print(f"serialized {a.serialize} -> {a.out or 'stdout'}  "
              f"({len(txt)} bytes, sha256 {hashlib.sha256(txt.encode()).hexdigest()[:16]})")
        print(f"self-validate: {'PASS' if ok else 'FAIL ' + str(bad)}")
        return 0 if ok else 1

    if a.validate:
        doc = json.loads(pathlib.Path(a.validate).read_text())
        # ROUND TRIP: re-serialize and confirm the document is stable under it.
        rt = json.loads(json.dumps(doc))
        stable = rt == doc
        ok, bad = validate(doc)
        print(f"round-trip stable: {stable}")
        print(f"validate: {'PASS' if ok else 'FAIL'}")
        for b in bad:
            print(f"  {b}")
        return 0 if (ok and stable) else 1

    if a.negative_test:
        doc = json.loads(pathlib.Path(a.negative_test).read_text())
        doc["representation"]["threadgroup_size"] = 128
        doc["kernel_requirements"][0]["kernel"] = \
            "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128"
        ok, bad = validate(doc)
        print("NEGATIVE TEST: injected a threadgroup_size and a named kernel into NR")
        print(f"validator says: {'PASS -- THE CHECK IS BROKEN' if ok else 'REJECTED, as required'}")
        for b in bad:
            print(f"  {b}")
        return 1 if not ok else 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
