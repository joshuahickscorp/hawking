# Mandatory combined revision: rare-route basis pilot

This is one implementation revision round covering both sealed audit
corrections. Before editing, read these files completely:

1. `/Users/scammermike/Downloads/hawking/control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-1.md`
2. `/Users/scammermike/Downloads/hawking/control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-2.md`

Implement and test every requirement in both files. Neither file is advisory
and later text does not waive earlier text. Where Revision 2 adds detail, use
the stricter combined interpretation.

The implementation/revision task must remain source-body-free: do not fetch,
read, release, or modify any real source body; do not touch MOP; do not execute
real acquire, measure, release, or aggregate operations.

Required outcome:

- only the four intended pilot/preflight deliverables are changed;
- fake-only tests exercise the full combined adversarial matrix;
- selftest and preflight trap network, downloader subprocess, and
  `*.safetensors` access;
- deterministic preflight regeneration is byte-identical;
- existing v2 and route-census suites still pass;
- every authorization, HIDE, Odyssey, traversal, capability, Ramanujan, and
  MOP fence remains false;
- report exact test counts, hashes, byte totals, and any unimplemented or
  uncertain item honestly.

Do not claim readiness for real source lifecycle unless every combined
requirement is implemented and proven.
