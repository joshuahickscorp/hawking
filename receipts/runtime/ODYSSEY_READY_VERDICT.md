# Odyssey readiness verdict — preserved evidence, machinery retired

`tools/odyssey_ready.py` is deleted by Event Horizon. Its RESULT is preserved here,
which is the correct pattern: keep the durable finding, retire the campaign machinery.

## Why the machinery is not restored

The gate printed `READY=13/13`. Its `callers()` ran
`git grep -l --fixed-strings <module-stem>` and counted every file containing the
module's NAME, not its call sites. A gate that counts word occurrences cannot fail.

Measured, word-mentions versus real production call sites:

    compare_candidates    240  ->  1
    enumerate_specimens   257  ->  2
    schedule_followups     88  -> 14
    derive_laws_scars      11  ->  1

## The honest verdict, after the checker was repaired (ascension, 2026-09-05)

    READY = 11        PARTIAL = 2

    gravity_experiments      -> tools/gravity_verify_source.py
                                0 production call sites, 0 test call sites.
                                Its ~9 references are a roadmap catalog entry naming
                                the path as DATA, a CAPABILITY_GRAPH.json entry, a doc,
                                and its own usage docstring. A hand-run CLI, not wired
                                capability.

    capability_qualification -> tools/odyssey/performance_qualification.py
                                0 production, 2 test call sites. Imported only by
                                test_gpu_cleanliness.py and
                                test_performance_qualification.py.

Event Horizon subsequently deleted `tools/odyssey/performance_qualification.py` and its
two test importers as one coherent cluster, with zero dangling references — which is
consistent with this verdict rather than contradicted by it.

Source: ascension-isolated, tools/odyssey_ready.py at 57fd2ad54.
Git remains the museum for the implementation.
