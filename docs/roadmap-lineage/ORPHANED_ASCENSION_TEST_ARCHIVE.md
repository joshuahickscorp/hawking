# Orphaned Ascension test disposition

`test_ascension_campaign.py` and `test_ascension_lifecycle.py` were retired
because their imports target the already-removed
`ascension_foundation_contracts`, `ascension_kernel_registry`,
`ascension_execution_plan`, and `ascension_knowledge_contract` modules.
Neither file could collect or exercise a current owner. The surviving
`ascension_contracts` tests cover the current blocked-state and planning
contracts; the deleted files remain recoverable from Git history.

This disposition does not authorize deletion of the still-referenced physical
gatekeeper/lifecycle cluster. That cluster has a separate missing-dependency
chain and requires an ownership audit before any broader retirement.
